"""Grocery Intel integration."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
import json
import logging
import os
import re
import shutil
import base64
from difflib import SequenceMatcher
import io
import subprocess
import tempfile
from typing import Any
import mimetypes
import time
import hashlib
import functools
from urllib.parse import quote_plus

import voluptuous as vol

import aiohttp
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_time_change, async_track_time_interval
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import dt as dt_util

from .locale import DEFAULT_MONTH_MAP, LocaleProfile, get_locale_profile

from .activity import ActivityLog
from .auto_shopping import AUTO_RUN_HOUR_LOCAL, async_run_auto_shopping
from .export import ExportFilters, build_export_payload, write_export_file
from .inventory_images import (
    cleanup_inventory_images_archive_sync,
    scan_inventory_images_inbox_sync,
    normalize_items_with_confidence_from_llm_result,
    _read_file_base64_sync as _inventory_read_file_b64_sync,
    _extract_taken_at_iso_sync as _inventory_extract_taken_at_iso_sync,
    async_analyze_inventory_image,
)
from .storage import ReceiptStorage
from .sensor import GrocerySpendCoordinator
from .const import (
    DOMAIN,
    CONF_CURRENCY_SYMBOL,
    CONF_RECEIPTS_INBOX_PATH,
    CONF_RECEIPTS_ARCHIVE_PATH,
    CONF_RECEIPTS_ARCHIVE_TTL_DAYS,
    CONF_INBOX_SCAN_INTERVAL_SEC,
    CONF_ON_SUCCESS,
    CONF_OCR_ENDPOINT_URL,
    CONF_OCR_LANGUAGE,
    CONF_OCR_API_TOKEN,
    CONF_OCR_API_TOKEN_HEADER,
    CONF_EXTRACTOR_MODE,
    CONF_LLM_PROVIDER,
    CONF_LLM_MODEL,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_EXTRA_INSTRUCTIONS,
    CONF_AZURE_API_VERSION,
    DEFAULT_CURRENCY_SYMBOL,
    DEFAULT_RECEIPTS_INBOX_PATH,
    DEFAULT_RECEIPTS_ARCHIVE_PATH,
    DEFAULT_RECEIPTS_ARCHIVE_TTL_DAYS,
    DEFAULT_INBOX_SCAN_INTERVAL_SEC,
    DEFAULT_ON_SUCCESS,
    DEFAULT_OCR_ENDPOINT_URL,
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_OCR_API_TOKEN,
    DEFAULT_OCR_API_TOKEN_HEADER,
    DEFAULT_EXTRACTOR_MODE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_EXTRA_INSTRUCTIONS,
    DEFAULT_AZURE_API_VERSION,
    SERVICE_ADD_RECEIPT,
    SERVICE_UNDO_ACTIVITY,
    SERVICE_REPROCESS_RECEIPTS,
    SERVICE_SCAN_RECEIPTS_INBOX,
    SERVICE_RUN_OCR,
    SERVICE_REPARSE_RECEIPTS,
    SERVICE_CLEAR_ALL_DATA,
    SERVICE_UPDATE_RECEIPT,
    SERVICE_RUN_AUTO_SHOPPING,
    CONF_INVENTORY_IMAGES_INBOX_PATH,
    CONF_INVENTORY_IMAGES_ARCHIVE_PATH,
    CONF_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
    CONF_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
    DEFAULT_INVENTORY_IMAGES_INBOX_PATH,
    DEFAULT_INVENTORY_IMAGES_ARCHIVE_PATH,
    DEFAULT_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS,
    DEFAULT_INVENTORY_IMAGES_SCAN_INTERVAL_SEC,
    CONF_TELEGRAM_BOT_TOKEN,
    CONF_TELEGRAM_ALLOWED_CHAT_IDS,
    CONF_TELEGRAM_AUTO_DETECT,
    CONF_TELEGRAM_SEND_FEEDBACK,
    DEFAULT_TELEGRAM_BOT_TOKEN,
    DEFAULT_TELEGRAM_ALLOWED_CHAT_IDS,
    DEFAULT_TELEGRAM_AUTO_DETECT,
    DEFAULT_TELEGRAM_SEND_FEEDBACK,
    CONF_EXPORTS_PATH,
    DEFAULT_EXPORTS_PATH,
    SERVICE_SCAN_INVENTORY_IMAGES_INBOX,
    SERVICE_RUN_INVENTORY_VISION,
    SERVICE_RESET_STUCK_RECEIPTS,
    SERVICE_TELEGRAM_INGEST,
    SERVICE_EXPORT_DATA,
    SERVICE_DEDUPE_STORES,
)

PLATFORMS: list[Platform] = [Platform.SENSOR]

SIGNAL_REFRESH = "grocery_intel_refresh"

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
HEIC_EXTENSIONS = {".heic", ".heif"}

# Safety guard for Telegram ingests (avoid downloading huge files into HA memory).
# Telegram Bot API file sizes vary by client/type; keep conservative.
TELEGRAM_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB

_LOGGER = logging.getLogger(__name__)


@dataclass
class GroceryIntelData:
    hass: HomeAssistant
    storage: ReceiptStorage
    activity: ActivityLog
    coordinator: GrocerySpendCoordinator
    ocr_semaphore: asyncio.Semaphore
    unsub_scan: callable | None = None
    unsub_auto_shopping: callable | None = None
    unsub_inventory_images_scan: callable | None = None


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration from YAML (not supported)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Grocery Intel from a config entry."""
    storage = ReceiptStorage(hass)
    activity = ActivityLog(hass)
    await storage.async_load()
    await activity.async_load()

    coordinator = GrocerySpendCoordinator(hass, storage, activity, entry)
    await coordinator.async_config_entry_first_refresh()

    data = GroceryIntelData(
        hass=hass,
        storage=storage,
        activity=activity,
        coordinator=coordinator,
        ocr_semaphore=asyncio.Semaphore(2),
    )
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = data

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    _register_services(hass)

    interval = entry.options.get(CONF_INBOX_SCAN_INTERVAL_SEC, DEFAULT_INBOX_SCAN_INTERVAL_SEC)

    async def _handle_scan(now) -> None:
        await _async_scan_receipts_inbox(hass)

    data.unsub_scan = async_track_time_interval(
        hass,
        _handle_scan,
        timedelta(seconds=interval),
    )

    inventory_interval = entry.options.get(
        CONF_INVENTORY_IMAGES_SCAN_INTERVAL_SEC, DEFAULT_INVENTORY_IMAGES_SCAN_INTERVAL_SEC
    )

    async def _handle_inventory_images_scan(now) -> None:
        await _async_scan_inventory_images_inbox(hass)

    data.unsub_inventory_images_scan = async_track_time_interval(
        hass,
        _handle_inventory_images_scan,
        timedelta(seconds=int(inventory_interval)),
    )

    async def _handle_auto_shopping(now) -> None:
        payload = await async_run_auto_shopping(hass, entry, data, dry_run=False)
        if payload.get("enabled"):
            await data.activity.async_add_activity(
                kind="shopping_list_auto_run",
                description="Auto-added shopping list items",
                payload=payload,
            )
            await data.coordinator.async_refresh()

    data.unsub_auto_shopping = async_track_time_change(
        hass,
        _handle_auto_shopping,
        hour=AUTO_RUN_HOUR_LOCAL,
        minute=0,
        second=0,
    )

    await _async_scan_receipts_inbox(hass)
    await _async_scan_inventory_images_inbox(hass)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        data = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        if data and data.unsub_scan:
            data.unsub_scan()
        if data and data.unsub_auto_shopping:
            data.unsub_auto_shopping()
        if data and data.unsub_inventory_images_scan:
            data.unsub_inventory_images_scan()

    if unload_ok and not hass.config_entries.async_entries(DOMAIN):
        _unregister_services(hass)

    return unload_ok


def _register_services(hass: HomeAssistant) -> None:
    def _reg(name: str, handler, schema) -> None:
        if hass.services.has_service(DOMAIN, name):
            return
        hass.services.async_register(DOMAIN, name, handler, schema=schema)

    line_item_schema = vol.Schema(
        {
            vol.Required("raw_name"): cv.string,
            vol.Required("line_total"): vol.Coerce(float),
            vol.Optional("qty_raw"): cv.string,
            vol.Optional("unit_price_raw"): vol.Coerce(float),
        },
        extra=vol.ALLOW_EXTRA,
    )

    async def handle_add_receipt(call: ServiceCall) -> None:
        data = _get_data(hass)
        if data is None:
            return

        total = call.data["total"]
        date_str = call.data.get("date")
        store = call.data.get("store")
        raw_text = call.data.get("raw_text")
        line_items = call.data.get("line_items")

        entry = _get_entry(hass)
        currency = None
        if entry:
            currency = entry.options.get(CONF_CURRENCY_SYMBOL, DEFAULT_CURRENCY_SYMBOL)

        try:
            receipt = await data.storage.async_add_receipt(
                total=total,
                date_str=date_str,
                store=store,
                raw_text=raw_text,
                currency=currency,
                line_items=line_items,
            )
        except ValueError as err:
            _LOGGER.warning("Invalid date provided: %s", err)
            return

        if receipt.get("store_name"):
            store_eid, canonical = await _async_assign_store_entity(
                data,
                store_name=receipt.get("store_name"),
                branch_name=None,
                merchant_hints=None,
            )
            if store_eid or canonical:
                await data.storage.async_update_receipt(
                    receipt["id"],
                    {
                        "store_entity_id": store_eid,
                        "store_name": canonical or receipt.get("store_name"),
                    },
                )

        item_count = len(receipt.get("line_items_raw", []))
        await data.activity.async_add_activity(
            kind="receipt_added",
            description=f"Receipt added (total: {total}, items: {item_count})",
            payload={"receipt_id": receipt["id"], "item_count": item_count},
        )

        await data.coordinator.async_refresh()

    async def handle_update_receipt(call: ServiceCall) -> None:
        data = _get_data(hass)
        if data is None:
            return

        receipt_id = call.data["receipt_id"]
        receipts = await data.storage.async_list_receipts()
        receipt = next((r for r in receipts if r.get("id") == receipt_id), None)
        if not receipt:
            _LOGGER.warning("Receipt not found: %s", receipt_id)
            return

        updates: dict[str, Any] = {}
        previous_store_name = receipt.get("store_name")

        if "store_name" in call.data:
            updates["store_name"] = data.storage.resolve_store_alias(
                _normalize_store_name(call.data.get("store_name"))
            )

        if "purchased_at" in call.data:
            purchased_raw = (call.data.get("purchased_at") or "").strip()
            if purchased_raw:
                parsed = _parse_date_from_text(purchased_raw)
                if parsed is not None:
                    updates["purchased_at"] = parsed.isoformat()
                else:
                    _LOGGER.warning("Invalid purchased_at for receipt_id=%s", receipt_id)
                    return
            else:
                updates["purchased_at"] = None

        if "total" in call.data:
            updates["total"] = float(call.data["total"]) if call.data["total"] is not None else None

        if "currency" in call.data:
            updates["currency"] = (call.data.get("currency") or "").strip() or None

        clear_items = bool(call.data.get("clear_line_items", False))
        if clear_items:
            updates["line_items_raw"] = []
        elif "line_items" in call.data:
            items = _clean_line_items(
                list(call.data.get("line_items") or []),
                float(updates.get("total") or receipt.get("total"))
                if (updates.get("total") or receipt.get("total")) is not None
                else None,
            )
            updates["line_items_raw"] = items

        if not updates:
            return

        if updates.get("store_name"):
            store_eid, canonical = await _async_assign_store_entity(
                data,
                store_name=updates.get("store_name"),
                branch_name=None,
                merchant_hints=receipt.get("merchant_hints")
                if isinstance(receipt.get("merchant_hints"), dict)
                else None,
            )
            if store_eid:
                updates["store_entity_id"] = store_eid
            if canonical:
                updates["store_name"] = canonical

        await data.storage.async_update_receipt(receipt_id, updates)

        # Learning: if the user corrected store name, remember it as an alias.
        new_store = updates.get("store_name")
        if (
            isinstance(previous_store_name, str)
            and previous_store_name.strip()
            and isinstance(new_store, str)
            and new_store.strip()
            and previous_store_name.strip() != new_store.strip()
        ):
            changed = await data.storage.async_add_store_alias(previous_store_name, new_store)
            if changed:
                await data.activity.async_add_activity(
                    kind="store_alias_learned",
                    description=f"Learned store alias: {previous_store_name} -> {new_store}",
                    payload={"from": previous_store_name, "to": new_store, "receipt_id": receipt_id},
                )

        reprocess = bool(call.data.get("reprocess", True))
        if reprocess and (
            "line_items_raw" in updates
            or "store_name" in updates
            or "purchased_at" in updates
        ):
            await data.storage.async_reprocess_receipts(receipt_id, 1)

        await data.activity.async_add_activity(
            kind="receipt_updated",
            description=f"Receipt updated: {receipt.get('filename') or receipt_id}",
            payload={"receipt_id": receipt_id, "fields": sorted(list(updates.keys()))},
        )

        await data.coordinator.async_refresh()

    async def handle_undo_activity(call: ServiceCall) -> None:
        data = _get_data(hass)
        if data is None:
            return

        activity_id = call.data["activity_id"]
        ok = await data.activity.async_undo_activity(activity_id, data.storage)
        if not ok:
            _LOGGER.warning("Undo failed for activity_id=%s", activity_id)
            return

        await data.coordinator.async_refresh()

    async def handle_reprocess(call: ServiceCall) -> None:
        data = _get_data(hass)
        if data is None:
            return

        receipt_id = call.data.get("receipt_id")
        limit = call.data.get("limit", 50)
        processed = await data.storage.async_reprocess_receipts(receipt_id, limit)

        await data.activity.async_add_activity(
            kind="receipts_reprocessed",
            description=f"Reprocessed receipts ({processed})",
            payload={"receipt_id": receipt_id, "count": processed},
        )

        await data.coordinator.async_refresh()

    async def handle_scan(call: ServiceCall) -> None:
        await _async_scan_receipts_inbox(hass)

    async def handle_run_ocr(call: ServiceCall) -> None:
        data = _get_data(hass)
        entry = _get_entry(hass)
        if data is None or entry is None:
            return

        extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
        if extractor_mode == "llm":
            receipt_id = call.data.get("receipt_id")
            overwrite = bool(call.data.get("overwrite", False))
            receipts = await data.storage.async_list_receipts()
            if receipt_id:
                receipts = [r for r in receipts if r.get("id") == receipt_id]
            else:
                receipts = [
                    r
                    for r in receipts
                    if r.get("file_path")
                    and r.get("extract_status") in {"pending", "failed"}
                ]
            queued_any = False
            for receipt in receipts:
                rid = receipt.get("id")
                if rid:
                    await data.storage.async_update_receipt(
                        rid,
                        {
                            "extract_status": "queued",
                            "extract_queued_at": dt_util.now().isoformat(),
                        },
                    )
                    queued_any = True
                hass.async_create_task(
                    _async_run_llm_for_receipt_file(
                        hass,
                        entry,
                        data,
                        receipt,
                        overwrite=overwrite,
                        force=bool(receipt_id),
                    )
                )
            if queued_any:
                await data.coordinator.async_refresh()
            return

        receipt_id = call.data.get("receipt_id")
        receipts = await data.storage.async_list_receipts()

        if receipt_id:
            receipts = [r for r in receipts if r.get("id") == receipt_id]
        else:
            receipts = [
                r
                for r in receipts
                if r.get("file_path")
                and r.get("extract_status") in {"pending", "failed"}
            ]

        queued_any = False
        for receipt in receipts:
            rid = receipt.get("id")
            if rid and receipt.get("extract_status") in {"pending", "failed"}:
                await data.storage.async_update_receipt(
                    rid,
                    {
                        "extract_status": "queued",
                        "extract_queued_at": dt_util.now().isoformat(),
                    },
                )
                queued_any = True
            hass.async_create_task(_async_run_ocr_for_receipt(hass, entry, data, receipt))
        if queued_any:
            await data.coordinator.async_refresh()

    async def handle_reparse(call: ServiceCall) -> None:
        data = _get_data(hass)
        entry = _get_entry(hass)
        if data is None or entry is None:
            return

        receipt_id = call.data.get("receipt_id")
        limit = call.data.get("limit", 50)
        overwrite = bool(call.data.get("overwrite", False))
        receipts = await data.storage.async_list_receipts()
        receipts.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        if receipt_id:
            receipts = [r for r in receipts if r.get("id") == receipt_id]
        else:
            receipts = receipts[:limit]

        updated = 0
        for receipt in receipts:
            rid = receipt.get("id")
            text = receipt.get("ocr_text") or ""
            if not rid or not text:
                continue

            updates = await _async_extract_receipt_fields(
                hass=hass,
                entry=entry,
                receipt=receipt,
                text=text,
                filename=receipt.get("filename") or "",
                overwrite=overwrite,
            )

            if not updates:
                continue

            if "store_name" in updates:
                updates["store_name"] = data.storage.resolve_store_alias(
                    _safe_str(updates.get("store_name"))
                )
            if updates.get("store_name"):
                store_eid, canonical = await _async_assign_store_entity(
                    data,
                    store_name=updates.get("store_name"),
                    branch_name=None,
                    merchant_hints=None,
                )
                if store_eid:
                    updates["store_entity_id"] = store_eid
                if canonical:
                    updates["store_name"] = canonical

            await data.storage.async_update_receipt(rid, updates)
            updated += 1

        if updated:
            await data.activity.async_add_activity(
                kind="receipts_reparsed",
                description=f"Re-parsed receipts ({updated})",
                payload={"receipt_id": receipt_id, "count": updated},
            )
            await data.coordinator.async_refresh()

    async def handle_clear_all(call: ServiceCall) -> None:
        data = _get_data(hass)
        if data is None:
            return

        if call.data.get("confirm") is not True:
            _LOGGER.warning(
                "Refusing to clear all Grocery Intel data without confirm=true"
            )
            return

        await data.storage.async_clear_all_data()
        await data.activity.async_clear_all()
        await data.coordinator.async_refresh()

    async def handle_run_auto_shopping(call: ServiceCall) -> None:
        data = _get_data(hass)
        entry = _get_entry(hass)
        if data is None or entry is None:
            return

        payload = await async_run_auto_shopping(
            hass,
            entry,
            data,
            dry_run=bool(call.data.get("dry_run", False)),
        )
        if payload.get("enabled"):
            await data.activity.async_add_activity(
                kind="shopping_list_auto_run",
                description="Auto-added shopping list items",
                payload=payload,
            )
            await data.coordinator.async_refresh()

    async def handle_scan_inventory_images(call: ServiceCall) -> None:
        await _async_scan_inventory_images_inbox(hass)

    async def handle_run_inventory_vision(call: ServiceCall) -> None:
        data = _get_data(hass)
        entry = _get_entry(hass)
        if data is None or entry is None:
            return

        image_id = (call.data.get("image_id") or "").strip() or None
        images = await data.storage.async_list_inventory_images()
        if image_id:
            images = [i for i in images if i.get("image_id") == image_id]
        else:
            images = [i for i in images if i.get("status") in {"pending", "failed"}]

        for img in images:
            hass.async_create_task(_async_run_inventory_vision_for_image(hass, entry, data, img))

    async def handle_reset_stuck_receipts(call: ServiceCall) -> None:
        data = _get_data(hass)
        entry = _get_entry(hass)
        if data is None or entry is None:
            return

        older_than_minutes = int(call.data.get("older_than_minutes") or 30)
        reset = await _async_reset_stuck_receipts(
            hass,
            entry,
            data,
            older_than_minutes=older_than_minutes,
            include_missing_started=True,
        )
        if reset:
            await data.activity.async_add_activity(
                kind="receipts_reset_stuck",
                description=f"Reset stuck receipts ({reset})",
                payload={"count": reset, "older_than_minutes": older_than_minutes},
            )
            await data.coordinator.async_refresh()

    async def handle_telegram_ingest(call: ServiceCall) -> None:
        """Ingest a Telegram attachment into receipts or inventory images."""
        data = _get_data(hass)
        entry = _get_entry(hass)
        if data is None or entry is None:
            return

        token = (entry.options.get(CONF_TELEGRAM_BOT_TOKEN, DEFAULT_TELEGRAM_BOT_TOKEN) or "").strip()
        if not token:
            _LOGGER.warning("Telegram ingest called but telegram_bot_token is not configured")
            return

        allowed_raw = entry.options.get(
            CONF_TELEGRAM_ALLOWED_CHAT_IDS, DEFAULT_TELEGRAM_ALLOWED_CHAT_IDS
        )
        allowed = _parse_telegram_allowed_chat_ids(allowed_raw)
        chat_id = _coerce_int(call.data.get("chat_id"))
        if chat_id is None:
            _LOGGER.warning("Telegram ingest missing/invalid chat_id")
            return
        if allowed and chat_id not in allowed:
            _LOGGER.warning("Telegram ingest rejected from chat_id=%s (not in allowlist)", chat_id)
            return

        file_id = str(call.data.get("file_id") or "").strip()
        if not file_id:
            _LOGGER.warning("Telegram ingest missing file_id")
            return

        kind = str(call.data.get("kind") or "auto").strip().lower()
        if kind not in {"auto", "receipt", "inventory"}:
            kind = "auto"
        caption = _safe_str(call.data.get("caption"))
        message_id = _coerce_int(call.data.get("message_id"))
        filename_override = _safe_str(call.data.get("filename"))

        send_feedback = bool(
            entry.options.get(CONF_TELEGRAM_SEND_FEEDBACK, DEFAULT_TELEGRAM_SEND_FEEDBACK)
        )

        try:
            info = await _async_telegram_get_file_info(hass, token=token, file_id=file_id)
            file_path = _safe_str(info.get("file_path"))
            if not file_path:
                raise RuntimeError("Telegram getFile returned no file_path")
            name_for_log = filename_override or os.path.basename(str(file_path)) or file_id
            file_size = info.get("file_size")
            try:
                file_size_i = int(file_size) if file_size is not None else None
            except Exception:
                file_size_i = None
            if file_size_i is not None and file_size_i > TELEGRAM_MAX_FILE_BYTES:
                msg = (
                    f"File too large ({file_size_i/1024/1024:.1f} MB). "
                    f"Limit is {TELEGRAM_MAX_FILE_BYTES/1024/1024:.0f} MB."
                )
                _LOGGER.warning(
                    "Telegram ingest rejected oversized file (size=%s): %s", file_size_i, name_for_log
                )
                if send_feedback:
                    await _async_telegram_send_message(
                        hass,
                        token=token,
                        chat_id=chat_id,
                        text=msg,
                        reply_to_message_id=message_id,
                    )
                return
            content = await _async_telegram_download_file_bytes(hass, token=token, file_path=file_path)
        except Exception as err:
            _LOGGER.warning("Telegram ingest download failed: %s", err)
            if send_feedback:
                await _async_telegram_send_message(
                    hass,
                    token=token,
                    chat_id=chat_id,
                    text=f"Failed to download file from Telegram: {err}",
                    reply_to_message_id=message_id,
                )
            return

        detected_name = filename_override or os.path.basename(str(file_path))
        detected_ext = os.path.splitext(detected_name)[1].lower()
        if not detected_ext:
            detected_ext = _detect_ext_from_bytes(content) or ""
            detected_name = (detected_name or "telegram_upload") + (detected_ext or "")

        auto_detect = bool(entry.options.get(CONF_TELEGRAM_AUTO_DETECT, DEFAULT_TELEGRAM_AUTO_DETECT))
        final_kind = kind
        confidence = 1.0
        reason = "explicit"
        if kind == "auto":
            if not auto_detect:
                final_kind = "receipt"
                confidence = 0.4
                reason = "auto_detect_disabled_default_receipt"
            else:
                final_kind, confidence, reason = await _async_classify_telegram_upload(
                    hass,
                    entry=entry,
                    filename=detected_name,
                    caption=caption,
                    content=content,
                )

        source_meta = {
            "chat_id": chat_id,
            "message_id": message_id,
            "file_id": file_id,
            "filename": detected_name,
            "caption": caption,
            "classified_as": final_kind,
            "classify_confidence": round(float(confidence), 3),
            "classify_reason": reason,
        }

        if final_kind == "inventory":
            image_id, archived_path, duplicate = await _async_ingest_inventory_image_bytes(
                hass,
                entry=entry,
                data=data,
                filename=detected_name,
                content=content,
                source_meta=source_meta,
            )
            if send_feedback:
                if duplicate:
                    await _async_telegram_send_message(
                        hass,
                        token=token,
                        chat_id=chat_id,
                        text=f"Duplicate inventory image archived: {os.path.basename(archived_path)}",
                        reply_to_message_id=message_id,
                    )
                elif image_id:
                    await _async_telegram_send_message(
                        hass,
                        token=token,
                        chat_id=chat_id,
                        text=f"Inventory image received ({os.path.basename(archived_path)}). Queued for analysis.",
                        reply_to_message_id=message_id,
                    )
        else:
            receipt_id, archived_path, duplicate = await _async_ingest_receipt_bytes(
                hass,
                entry=entry,
                data=data,
                filename=detected_name,
                content=content,
                source_meta=source_meta,
            )
            if send_feedback:
                if duplicate:
                    await _async_telegram_send_message(
                        hass,
                        token=token,
                        chat_id=chat_id,
                        text=f"Duplicate receipt archived: {os.path.basename(archived_path)}",
                        reply_to_message_id=message_id,
                    )
                elif receipt_id:
                    await _async_telegram_send_message(
                        hass,
                        token=token,
                        chat_id=chat_id,
                        text=f"Receipt received ({os.path.basename(archived_path)}). Queued for extraction.",
                        reply_to_message_id=message_id,
                    )

    async def handle_export_data(call: ServiceCall) -> None:
        data = _get_data(hass)
        entry = _get_entry(hass)
        if data is None or entry is None:
            return

        scope = str(call.data.get("scope") or "analytics").strip().lower()
        since = _safe_str(call.data.get("since"))
        until = _safe_str(call.data.get("until"))
        include_undated = bool(call.data.get("include_undated", False))

        exports_dir = (entry.options.get(CONF_EXPORTS_PATH, DEFAULT_EXPORTS_PATH) or "").strip()
        if not exports_dir:
            exports_dir = DEFAULT_EXPORTS_PATH
        if not hass.config.is_allowed_path(exports_dir):
            reason = f"Exports path not allowed by Home Assistant: {exports_dir}"
            _LOGGER.warning(reason)
            await data.activity.async_add_activity(
                kind="export_failed",
                description="Export failed",
                payload={"scope": scope, "since": since, "until": until, "reason": reason},
            )
            try:
                from homeassistant.components import persistent_notification

                persistent_notification.async_create(
                    hass,
                    title="Grocery Intel export failed",
                    message=reason,
                    notification_id="grocery_intel_export_failed",
                )
            except Exception:
                pass
            return

        filename = _safe_str(call.data.get("filename"))
        if not filename:
            ts = dt_util.now().strftime("%Y%m%d_%H%M%S")
            filename = f"grocery_intel_export_{ts}_{scope}.json"
        # Ensure filename cannot escape the exports directory.
        filename = os.path.basename(filename).strip() or "grocery_intel_export.json"
        if not filename.lower().endswith(".json"):
            filename += ".json"

        receipts = await data.storage.async_list_receipts()
        line_items = await data.storage.async_list_line_items()
        products = await data.storage.async_list_products()
        observations = await data.storage.async_list_observations()
        stores = await data.storage.async_list_stores()
        inventory_images = await data.storage.async_list_inventory_images()
        activities = await data.activity.async_list_activities() if scope == "full" else None

        filters = ExportFilters(since=since or None, until=until or None, include_undated=include_undated)
        payload = build_export_payload(
            hass,
            scope=scope,
            filters=filters,
            receipts=receipts,
            line_items=line_items,
            products=products,
            observations=observations,
            stores=stores,
            inventory_images=inventory_images,
            activities=activities,
        )

        try:
            written = await hass.async_add_executor_job(
                functools.partial(
                    write_export_file,
                    hass,
                    exports_dir=exports_dir,
                    filename=filename,
                    payload=payload,
                )
            )
        except Exception as err:
            _LOGGER.warning("Export failed: %s", err)
            await data.activity.async_add_activity(
                kind="export_failed",
                description="Export failed",
                payload={"scope": scope, "since": since, "until": until, "reason": str(err)},
            )
            return

        await data.activity.async_add_activity(
            kind="export_created",
            description=f"Export created: {os.path.basename(written)}",
            payload={"path": written, "scope": scope, "since": since, "until": until},
        )

        # Best-effort user feedback.
        try:
            from homeassistant.components import persistent_notification

            persistent_notification.async_create(
                hass,
                title="Grocery Intel export created",
                message=f"Export written to: {written}",
                notification_id="grocery_intel_export_created",
            )
        except Exception:
            pass

        await data.coordinator.async_refresh()

    async def handle_dedupe_stores(call: ServiceCall) -> None:
        data = _get_data(hass)
        if data is None:
            return

        mode = str(call.data.get("mode") or "hybrid").strip().lower()
        dry_run = bool(call.data.get("dry_run", True))
        delete_orphans = bool(call.data.get("delete_orphans", False))
        max_preview = int(call.data.get("max_preview", 20))

        result = await data.storage.async_dedupe_stores(
            mode=mode,
            dry_run=dry_run,
            delete_orphans=delete_orphans,
            max_preview=max_preview,
        )

        title = "Grocery Intel store dedupe (dry run)" if dry_run else "Grocery Intel store dedupe"
        summary = (
            f"Mode: {result.get('mode')}\n"
            f"Stores: {result.get('stores_before')} -> {result.get('stores_after')}\n"
            f"Merges: {result.get('store_merges')}\n"
            f"Receipts updated: {result.get('receipts_updated')}\n"
            f"Orphans deleted: {result.get('orphans_deleted')}\n"
        )

        preview = result.get("preview") or []
        if isinstance(preview, list) and preview:
            summary += "\nPreview:\n"
            for row in preview[: max_preview or 0]:
                if not isinstance(row, dict):
                    continue
                summary += (
                    f"- {row.get('from')} -> {row.get('to')} "
                    f"({row.get('chain')}, from_receipts={row.get('from_receipts')})\n"
                )

        await data.activity.async_add_activity(
            kind="stores_deduped" if not dry_run else "stores_dedupe_dry_run",
            description="Store dedupe completed" if not dry_run else "Store dedupe dry-run computed",
            payload={
                "mode": result.get("mode"),
                "dry_run": bool(dry_run),
                "stores_before": result.get("stores_before"),
                "stores_after": result.get("stores_after"),
                "store_merges": result.get("store_merges"),
                "receipts_updated": result.get("receipts_updated"),
                "orphans_deleted": result.get("orphans_deleted"),
                "preview": preview[: min(len(preview), 20)] if isinstance(preview, list) else [],
            },
        )

        try:
            from homeassistant.components import persistent_notification

            persistent_notification.async_create(
                hass,
                title=title,
                message=summary.strip(),
                notification_id="grocery_intel_store_dedupe",
            )
        except Exception:
            pass

        await data.coordinator.async_refresh()

    _reg(
        SERVICE_ADD_RECEIPT,
        handle_add_receipt,
        vol.Schema(
            {
                vol.Required("total"): vol.Coerce(float),
                vol.Optional("date"): cv.string,
                vol.Optional("store"): cv.string,
                vol.Optional("raw_text"): cv.string,
                vol.Optional("line_items"): vol.All(cv.ensure_list, [line_item_schema]),
            }
        ),
    )

    _reg(
        SERVICE_UPDATE_RECEIPT,
        handle_update_receipt,
        vol.Schema(
            {
                vol.Required("receipt_id"): cv.string,
                vol.Optional("store_name"): cv.string,
                vol.Optional("purchased_at"): cv.string,
                vol.Optional("total"): vol.Coerce(float),
                vol.Optional("currency"): cv.string,
                vol.Optional("line_items"): vol.All(cv.ensure_list, [line_item_schema]),
                vol.Optional("clear_line_items", default=False): cv.boolean,
                vol.Optional("reprocess", default=True): cv.boolean,
            }
        ),
    )

    _reg(
        SERVICE_UNDO_ACTIVITY,
        handle_undo_activity,
        vol.Schema({vol.Required("activity_id"): cv.string}),
    )

    _reg(
        SERVICE_REPROCESS_RECEIPTS,
        handle_reprocess,
        vol.Schema(
            {
                vol.Optional("receipt_id"): cv.string,
                vol.Optional("limit", default=50): vol.All(int, vol.Range(min=1, max=500)),
            }
        ),
    )

    _reg(SERVICE_SCAN_RECEIPTS_INBOX, handle_scan, vol.Schema({}))

    _reg(
        SERVICE_RUN_OCR,
        handle_run_ocr,
        vol.Schema(
            {
                vol.Optional("receipt_id"): cv.string,
                vol.Optional("overwrite", default=False): cv.boolean,
            }
        ),
    )

    _reg(
        SERVICE_REPARSE_RECEIPTS,
        handle_reparse,
        vol.Schema(
            {
                vol.Optional("receipt_id"): cv.string,
                vol.Optional("limit", default=50): vol.All(int, vol.Range(min=1, max=500)),
                vol.Optional("overwrite", default=False): cv.boolean,
            }
        ),
    )

    _reg(
        SERVICE_CLEAR_ALL_DATA,
        handle_clear_all,
        vol.Schema({vol.Required("confirm"): cv.boolean}),
    )

    _reg(
        SERVICE_RUN_AUTO_SHOPPING,
        handle_run_auto_shopping,
        vol.Schema({vol.Optional("dry_run", default=False): cv.boolean}),
    )

    _reg(SERVICE_SCAN_INVENTORY_IMAGES_INBOX, handle_scan_inventory_images, vol.Schema({}))

    _reg(
        SERVICE_RUN_INVENTORY_VISION,
        handle_run_inventory_vision,
        vol.Schema({vol.Optional("image_id"): cv.string}),
    )

    _reg(
        SERVICE_RESET_STUCK_RECEIPTS,
        handle_reset_stuck_receipts,
        vol.Schema({vol.Optional("older_than_minutes", default=30): vol.All(int, vol.Range(min=1, max=1440))}),
    )

    _reg(
        SERVICE_TELEGRAM_INGEST,
        handle_telegram_ingest,
        vol.Schema(
            {
                vol.Required("chat_id"): vol.Any(int, cv.string),
                vol.Required("file_id"): cv.string,
                vol.Optional("filename"): cv.string,
                vol.Optional("caption"): cv.string,
                vol.Optional("message_id"): vol.Any(int, cv.string),
                vol.Optional("kind", default="auto"): cv.string,
            }
        ),
    )

    _reg(
        SERVICE_EXPORT_DATA,
        handle_export_data,
        vol.Schema(
            {
                vol.Optional("scope", default="analytics"): cv.string,
                vol.Optional("since"): cv.string,
                vol.Optional("until"): cv.string,
                vol.Optional("include_undated", default=False): cv.boolean,
                vol.Optional("filename"): cv.string,
            }
        ),
    )

    _reg(
        SERVICE_DEDUPE_STORES,
        handle_dedupe_stores,
        vol.Schema(
            {
                vol.Optional("mode", default="hybrid"): vol.In(["hybrid", "strict", "chain_only"]),
                vol.Optional("dry_run", default=True): cv.boolean,
                vol.Optional("delete_orphans", default=False): cv.boolean,
                vol.Optional("max_preview", default=20): vol.All(int, vol.Range(min=0, max=200)),
            }
        ),
    )


def _unregister_services(hass: HomeAssistant) -> None:
    hass.services.async_remove(DOMAIN, SERVICE_ADD_RECEIPT)
    hass.services.async_remove(DOMAIN, SERVICE_UPDATE_RECEIPT)
    hass.services.async_remove(DOMAIN, SERVICE_UNDO_ACTIVITY)
    hass.services.async_remove(DOMAIN, SERVICE_REPROCESS_RECEIPTS)
    hass.services.async_remove(DOMAIN, SERVICE_SCAN_RECEIPTS_INBOX)
    hass.services.async_remove(DOMAIN, SERVICE_RUN_OCR)
    hass.services.async_remove(DOMAIN, SERVICE_REPARSE_RECEIPTS)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_ALL_DATA)
    hass.services.async_remove(DOMAIN, SERVICE_RUN_AUTO_SHOPPING)
    hass.services.async_remove(DOMAIN, SERVICE_SCAN_INVENTORY_IMAGES_INBOX)
    hass.services.async_remove(DOMAIN, SERVICE_RUN_INVENTORY_VISION)
    hass.services.async_remove(DOMAIN, SERVICE_RESET_STUCK_RECEIPTS)
    hass.services.async_remove(DOMAIN, SERVICE_TELEGRAM_INGEST)
    hass.services.async_remove(DOMAIN, SERVICE_EXPORT_DATA)
    hass.services.async_remove(DOMAIN, SERVICE_DEDUPE_STORES)


def _get_entry(hass: HomeAssistant) -> ConfigEntry | None:
    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        return None
    return entries[0]


def _get_data(hass: HomeAssistant) -> GroceryIntelData | None:
    entry = _get_entry(hass)
    if not entry:
        _LOGGER.error("No config entry found for Grocery Intel")
        return None

    return hass.data.get(DOMAIN, {}).get(entry.entry_id)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _parse_telegram_allowed_chat_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        out: set[int] = set()
        for v in value:
            i = _coerce_int(v)
            if i is not None:
                out.add(i)
        return out
    text = str(value).strip()
    if not text:
        return set()
    out: set[int] = set()
    for part in re.split(r"[,\s]+", text):
        part = part.strip()
        if not part:
            continue
        i = _coerce_int(part)
        if i is not None:
            out.add(i)
    return out


async def _async_telegram_get_file_info(
    hass: HomeAssistant, *, token: str, file_id: str
) -> dict[str, Any]:
    session = async_get_clientsession(hass)
    url = f"https://api.telegram.org/bot{token}/getFile?file_id={quote_plus(file_id)}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        data = await resp.json()
    if not isinstance(data, dict) or not data.get("ok"):
        raise RuntimeError(f"Telegram getFile failed: {data}")
    result = data.get("result")
    return result if isinstance(result, dict) else {}


async def _async_telegram_download_file_bytes(
    hass: HomeAssistant, *, token: str, file_path: str
) -> bytes:
    session = async_get_clientsession(hass)
    url = f"https://api.telegram.org/file/bot{token}/{file_path.lstrip('/')}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"Telegram download failed HTTP {resp.status}: {body[:200]}")
        return await resp.read()


async def _async_telegram_send_message(
    hass: HomeAssistant,
    *,
    token: str,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
) -> None:
    session = async_get_clientsession(hass)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
    try:
        async with session.post(
            url, timeout=aiohttp.ClientTimeout(total=30), json=payload
        ) as resp:
            _ = await resp.text()
    except Exception as err:
        _LOGGER.debug("Telegram sendMessage failed: %s", err)


def _format_iso_dt_for_user(hass: HomeAssistant, value: Any) -> str:
    """Format an ISO datetime/date string in Home Assistant's local timezone."""
    if not value:
        return "Unknown"
    raw = str(value).strip()
    if not raw:
        return "Unknown"
    dt = dt_util.parse_datetime(raw)
    if dt is None:
        # Support date-only strings.
        d = dt_util.parse_date(raw)
        if d is None:
            return raw
        dt = dt_util.as_utc(dt_util.start_of_local_day(d))
    local = dt_util.as_local(dt)
    try:
        # Example: "Fri 13 Feb 2026 22:11"
        return local.strftime("%a %d %b %Y %H:%M")
    except Exception:
        return local.isoformat()


def _detect_ext_from_bytes(content: bytes) -> str | None:
    if not content:
        return None
    if content.startswith(b"%PDF"):
        return ".pdf"
    if content[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"RIFF") and b"WEBP" in content[:16]:
        return ".webp"
    # HEIC/HEIF (ISO BMFF) signatures typically contain ftyp* brands.
    if content[4:12] == b"ftypheic" or content[4:12] == b"ftypheif":
        return ".heic"
    return None


async def _async_classify_telegram_upload(
    hass: HomeAssistant,
    *,
    entry: ConfigEntry,
    filename: str,
    caption: str | None,
    content: bytes,
) -> tuple[str, float, str]:
    ext = os.path.splitext(filename)[1].lower()
    low_caption = (caption or "").casefold()

    inv_keywords = {"inventory", "fridge", "pantry", "cupboard", "kyl", "skafferi"}
    rec_keywords = {"receipt", "kvitto", "kvittot"}
    if any(k in low_caption for k in inv_keywords):
        return "inventory", 0.95, "caption_keyword"
    if any(k in low_caption for k in rec_keywords):
        return "receipt", 0.95, "caption_keyword"

    if ext == ".pdf" or content.startswith(b"%PDF"):
        return "receipt", 0.95, "pdf_default_receipt"

    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
        return "receipt", 0.4, "unknown_extension_default_receipt"

    llm_provider = (entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or "").strip().lower()
    llm_model = (entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or "").strip()
    llm_base_url = (entry.options.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL) or "").strip()
    llm_api_key = (entry.options.get(CONF_LLM_API_KEY, DEFAULT_LLM_API_KEY) or "").strip()

    if not llm_provider or not llm_model:
        return "receipt", 0.4, "no_llm_config_default_receipt"

    # Write to a temporary file so we can reuse existing HEIC conversion logic.
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(prefix="grocery_intel_tg_", suffix=ext or ".bin", delete=False)
        tmp.write(content)
        tmp.close()
        img_b64 = await hass.async_add_executor_job(_inventory_read_file_b64_sync, tmp.name)
    finally:
        if tmp is not None:
            try:
                os.remove(tmp.name)
            except OSError:
                pass

    if not img_b64:
        return "receipt", 0.4, "image_read_failed_default_receipt"

    image_mime = "image/jpeg"
    if ext in {".png"}:
        image_mime = "image/png"
    elif ext in {".webp"}:
        image_mime = "image/webp"
    elif ext in HEIC_EXTENSIONS:
        image_mime = "image/jpeg"

    try:
        if llm_provider == "openai":
            if not llm_api_key:
                return "receipt", 0.4, "openai_missing_key_default_receipt"
            base = llm_base_url or "https://api.openai.com"
            out = await _async_llm_openai_vision_classify(
                hass=hass,
                base_url=base,
                api_key=llm_api_key,
                model=llm_model,
                filename=filename,
                image_b64=img_b64,
                image_mime=image_mime,
            )
            return out
        if llm_provider == "ollama":
            base = llm_base_url or "http://host.docker.internal:11434"
            out = await _async_llm_ollama_vision_classify(
                hass=hass,
                base_url=base,
                model=llm_model,
                filename=filename,
                image_b64=img_b64,
            )
            return out
    except Exception:
        pass

    return "receipt", 0.4, "classifier_failed_default_receipt"


async def _async_llm_openai_vision_classify(
    *,
    hass: HomeAssistant,
    base_url: str,
    api_key: str,
    model: str,
    filename: str,
    image_b64: str,
    image_mime: str,
) -> tuple[str, float, str]:
    session = async_get_clientsession(hass)
    url = _join_url(base_url, "/v1/responses")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": ["receipt", "inventory"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["kind", "confidence", "reason"],
    }
    data_url = f"data:{image_mime};base64,{image_b64}"
    system = (
        "You classify user uploads for a Home Assistant system.\n"
        "Task: determine if the image is a grocery RECEIPT (paper receipt with totals/prices) or an INVENTORY photo "
        "(fridge/pantry/cupboard contents). Return JSON only."
    )
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"filename: {filename}\nClassify the attached image."},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "classification",
                "schema": schema,
                "strict": True,
            }
        },
    }
    async with session.post(
        url,
        timeout=aiohttp.ClientTimeout(total=60),
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"OpenAI HTTP {resp.status}: {body[:200]}")
        data = await resp.json()
    out_text = _extract_openai_output_text(data) if isinstance(data, dict) else ""
    parsed = _extract_first_json_object(out_text)
    kind = str(parsed.get("kind") or "").strip().lower()
    try:
        conf = float(parsed.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    reason = _safe_str(parsed.get("reason")) or "openai_classifier"
    if kind not in {"receipt", "inventory"}:
        return "receipt", 0.4, "openai_classifier_invalid"
    return kind, conf, reason


async def _async_llm_ollama_vision_classify(
    *,
    hass: HomeAssistant,
    base_url: str,
    model: str,
    filename: str,
    image_b64: str,
) -> tuple[str, float, str]:
    system = (
        "You classify user uploads for a Home Assistant system.\n"
        "Decide if the image is a grocery RECEIPT (paper receipt with totals/prices) or an INVENTORY photo "
        "(fridge/pantry/cupboard contents).\n"
        "Return JSON only with keys: kind (receipt|inventory), confidence (0-1), reason."
    )
    session = async_get_clientsession(hass)
    url = _join_url(base_url, "/api/chat")
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"filename: {filename}\nClassify the attached image.",
                "images": [image_b64],
            },
        ],
        "options": {"temperature": 0},
    }
    async with session.post(
        url, timeout=aiohttp.ClientTimeout(total=60), json=payload
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"Ollama HTTP {resp.status}: {body[:200]}")
        data = await resp.json()
    msg = data.get("message", {}) if isinstance(data, dict) else {}
    content = msg.get("content") if isinstance(msg, dict) else ""
    parsed = _extract_first_json_object(content) if isinstance(content, str) else {}
    kind = str(parsed.get("kind") or "").strip().lower()
    try:
        conf = float(parsed.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    reason = _safe_str(parsed.get("reason")) or "ollama_classifier"
    if kind not in {"receipt", "inventory"}:
        return "receipt", 0.4, "ollama_classifier_invalid"
    return kind, conf, reason


async def _async_ingest_receipt_bytes(
    hass: HomeAssistant,
    *,
    entry: ConfigEntry,
    data: GroceryIntelData,
    filename: str,
    content: bytes,
    source_meta: dict[str, Any],
) -> tuple[str | None, str, bool]:
    archive_path = entry.options.get(CONF_RECEIPTS_ARCHIVE_PATH, DEFAULT_RECEIPTS_ARCHIVE_PATH)
    if not hass.config.is_allowed_path(archive_path):
        raise RuntimeError("Receipts archive path is not allowed")
    os.makedirs(archive_path, exist_ok=True)

    content_hash = hashlib.sha256(content).hexdigest()
    fingerprint = f"sha256:{content_hash}"
    processed = await data.storage.async_get_processed_fingerprints()
    processed |= await data.storage.async_get_receipt_content_hash_fingerprints()

    now_ts = time.time()
    if fingerprint in processed:
        dest = _unique_dest_path(archive_path, filename, suffix="_duplicate")
        await hass.async_add_executor_job(_write_bytes_sync, dest, content, now_ts)
        await data.activity.async_add_activity(
            kind="receipt_duplicate_file",
            description=(
                "Archived duplicate receipt file with suffix: "
                f"{filename} -> {os.path.basename(dest)}"
            ),
            payload={"filename": filename, "to_path": dest, "source": "telegram"},
        )
        return None, dest, True

    dest = _unique_dest_path(archive_path, filename)
    await hass.async_add_executor_job(_write_bytes_sync, dest, content, now_ts)

    receipt = await data.storage.async_add_receipt(
        total=None,
        date_str=None,
        store=None,
        raw_text=None,
        currency=None,
        line_items=None,
        source_type="telegram",
        file_path=dest,
        filename=os.path.basename(dest),
    )
    await data.storage.async_update_receipt(
        receipt["id"], {"content_hash": content_hash, "source_meta": source_meta}
    )
    await data.storage.async_mark_processed(
        fingerprint,
        {
            "fingerprint": fingerprint,
            "content_hash": content_hash,
            "archived_path": dest,
            "filename": os.path.basename(dest),
            "size": len(content),
            "processed_at": dt_util.now().isoformat(),
            "source": "telegram",
        },
    )
    await data.activity.async_add_activity(
        kind="receipt_imported_file",
        description=f"Imported receipt file (Telegram): {os.path.basename(dest)}",
        payload={"receipt_id": receipt["id"], "filename": os.path.basename(dest), "source": "telegram"},
    )

    extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
    await data.storage.async_update_receipt(
        receipt["id"],
        {"extract_status": "queued", "extract_queued_at": dt_util.now().isoformat()},
    )
    if extractor_mode == "llm":
        hass.async_create_task(
            _async_run_llm_for_receipt_file(hass, entry, data, receipt, overwrite=False, force=True)
        )
    else:
        hass.async_create_task(_async_run_ocr_for_receipt(hass, entry, data, receipt))
    await data.coordinator.async_refresh()
    return receipt["id"], dest, False


async def _async_ingest_inventory_image_bytes(
    hass: HomeAssistant,
    *,
    entry: ConfigEntry,
    data: GroceryIntelData,
    filename: str,
    content: bytes,
    source_meta: dict[str, Any],
) -> tuple[str | None, str, bool]:
    archive_path = entry.options.get(
        CONF_INVENTORY_IMAGES_ARCHIVE_PATH, DEFAULT_INVENTORY_IMAGES_ARCHIVE_PATH
    )
    if not hass.config.is_allowed_path(archive_path):
        raise RuntimeError("Inventory images archive path is not allowed")
    os.makedirs(archive_path, exist_ok=True)

    fingerprint = hashlib.sha256(content).hexdigest()
    processed = await data.storage.async_get_processed_inventory_images()
    processed |= await data.storage.async_get_inventory_image_fingerprints()

    now_ts = time.time()
    if fingerprint in processed:
        dest = _unique_dest_path(archive_path, filename, suffix="_duplicate")
        await hass.async_add_executor_job(_write_bytes_sync, dest, content, now_ts)
        await data.activity.async_add_activity(
            kind="inventory_image_duplicate_file",
            description=(
                "Archived duplicate inventory image with suffix: "
                f"{filename} -> {os.path.basename(dest)}"
            ),
            payload={"filename": filename, "to_path": dest, "source": "telegram"},
        )
        return None, dest, True

    dest = _unique_dest_path(archive_path, filename)
    await hass.async_add_executor_job(_write_bytes_sync, dest, content, now_ts)
    taken_at = await hass.async_add_executor_job(_inventory_extract_taken_at_iso_sync, dest)

    row = await data.storage.async_add_inventory_image(
        archived_path=dest,
        filename=os.path.basename(dest),
        fingerprint=fingerprint,
        taken_at=taken_at,
        source_type="telegram",
        source_meta=source_meta,
    )
    await data.storage.async_mark_inventory_image_processed(
        fingerprint,
        {
            "fingerprint": fingerprint,
            "archived_path": dest,
            "filename": os.path.basename(dest),
            "size": len(content),
            "processed_at": dt_util.now().isoformat(),
            "source": "telegram",
        },
    )
    await data.activity.async_add_activity(
        kind="inventory_image_imported",
        description=f"Imported inventory image (Telegram): {row.get('filename')}",
        payload={"image_id": row.get("image_id"), "filename": row.get("filename"), "source": "telegram"},
    )
    hass.async_create_task(_async_run_inventory_vision_for_image(hass, entry, data, row))
    await data.coordinator.async_refresh()
    return row.get("image_id"), dest, False


def _write_bytes_sync(path: str, content: bytes, now_ts: float) -> None:
    with open(path, "wb") as f:
        f.write(content)
    try:
        os.utime(path, (now_ts, now_ts))
    except OSError:
        pass


def _get_telegram_settings(entry: ConfigEntry) -> tuple[str, bool]:
    token = (entry.options.get(CONF_TELEGRAM_BOT_TOKEN, DEFAULT_TELEGRAM_BOT_TOKEN) or "").strip()
    enabled = bool(entry.options.get(CONF_TELEGRAM_SEND_FEEDBACK, DEFAULT_TELEGRAM_SEND_FEEDBACK))
    return token, enabled


async def _async_maybe_notify_telegram_receipt(
    data: GroceryIntelData,
    *,
    receipt_id: str,
    status: str,
    reason: str | None = None,
) -> None:
    hass = data.hass
    entry = _get_entry(hass)
    if entry is None:
        return
    token, enabled = _get_telegram_settings(entry)
    if not token or not enabled:
        return

    receipt = await data.storage.async_get_receipt(receipt_id) or {}
    if receipt.get("source_type") != "telegram":
        return
    meta = receipt.get("source_meta") if isinstance(receipt.get("source_meta"), dict) else {}
    chat_id = _coerce_int(meta.get("chat_id"))
    reply_to = _coerce_int(meta.get("message_id"))
    if chat_id is None:
        return

    filename = receipt.get("filename") or "receipt"
    if status == "done":
        store = receipt.get("store_name") or "Unknown"
        purchased_at = _format_iso_dt_for_user(hass, receipt.get("purchased_at"))
        total = receipt.get("total")
        currency = receipt.get("currency") or entry.options.get(CONF_CURRENCY_SYMBOL, DEFAULT_CURRENCY_SYMBOL) or ""
        try:
            items_n = len(receipt.get("line_items_raw") or [])
        except Exception:
            items_n = 0
        try:
            total_str = f"{float(total):.2f} {currency}".strip() if total is not None else "Unknown total"
        except Exception:
            total_str = f"{total} {currency}".strip() if total is not None else "Unknown total"
        text = (
            f"Receipt analyzed: {filename}\n"
            f"- Store: {store}\n"
            f"- When: {purchased_at}\n"
            f"- Total: {total_str}\n"
            f"- Line items: {items_n}"
        )
    else:
        text = f"Receipt failed: {filename}\nReason: {reason or 'unknown'}"

    await _async_telegram_send_message(
        hass,
        token=token,
        chat_id=chat_id,
        text=text,
        reply_to_message_id=reply_to,
    )


async def _async_maybe_notify_telegram_inventory_image(
    data: GroceryIntelData,
    *,
    image_row: dict[str, Any],
    status: str,
    reason: str | None = None,
    detected_rows: list[dict[str, Any]] | None = None,
) -> None:
    hass = data.hass
    entry = _get_entry(hass)
    if entry is None:
        return
    token, enabled = _get_telegram_settings(entry)
    if not token or not enabled:
        return
    if image_row.get("source_type") != "telegram":
        return
    meta = image_row.get("source_meta") if isinstance(image_row.get("source_meta"), dict) else {}
    chat_id = _coerce_int(meta.get("chat_id"))
    reply_to = _coerce_int(meta.get("message_id"))
    if chat_id is None:
        return

    filename = image_row.get("filename") or "image"
    taken_at = _format_iso_dt_for_user(
        hass, image_row.get("taken_at") or image_row.get("created_at")
    )
    if status == "done":
        items = detected_rows or (image_row.get("detected_products") or [])
        try:
            items_n = len(items)
        except Exception:
            items_n = 0
        top = []
        for row in (items or [])[:5]:
            label = row.get("label")
            conf = row.get("confidence")
            if label:
                top.append(f"{label} ({conf})")
        top_str = ", ".join(top) if top else "None"
        text = (
            f"Inventory image analyzed: {filename}\n"
            f"- Taken/imported: {taken_at}\n"
            f"- Detected items: {items_n}\n"
            f"- Top: {top_str}"
        )
    else:
        text = f"Inventory image failed: {filename}\nReason: {reason or 'unknown'}"

    await _async_telegram_send_message(
        hass,
        token=token,
        chat_id=chat_id,
        text=text,
        reply_to_message_id=reply_to,
    )


async def _async_scan_receipts_inbox(hass: HomeAssistant) -> None:
    data = _get_data(hass)
    entry = _get_entry(hass)
    if data is None or entry is None:
        return

    # Reset old queued/running receipts so they can be retried after restarts.
    reset = await _async_reset_stuck_receipts(
        hass,
        entry,
        data,
        older_than_minutes=30,
        include_missing_started=False,
    )
    if reset:
        await data.activity.async_add_activity(
            kind="receipts_reset_stuck",
            description=f"Reset stuck receipts ({reset})",
            payload={"count": reset, "older_than_minutes": 30},
        )

    inbox_path = entry.options.get(CONF_RECEIPTS_INBOX_PATH, DEFAULT_RECEIPTS_INBOX_PATH)
    archive_path = entry.options.get(CONF_RECEIPTS_ARCHIVE_PATH, DEFAULT_RECEIPTS_ARCHIVE_PATH)
    ttl_days = int(entry.options.get(CONF_RECEIPTS_ARCHIVE_TTL_DAYS, DEFAULT_RECEIPTS_ARCHIVE_TTL_DAYS))
    on_success = entry.options.get(CONF_ON_SUCCESS, DEFAULT_ON_SUCCESS)

    if not hass.config.is_allowed_path(inbox_path):
        _LOGGER.warning("Inbox path is not allowed: %s", inbox_path)
        return
    if not hass.config.is_allowed_path(archive_path):
        _LOGGER.warning("Archive path is not allowed: %s", archive_path)
        return

    processed = await data.storage.async_get_processed_fingerprints()
    # Also include hashes from already-imported receipts so we can dedupe across
    # restarts or after clearing processed_files.
    processed |= await data.storage.async_get_receipt_content_hash_fingerprints()

    result = await hass.async_add_executor_job(
        _scan_inbox_sync, inbox_path, archive_path, processed, on_success
    )

    imported = result.get("imported", [])
    for record in imported:
        receipt = await data.storage.async_add_receipt(
            total=None,
            date_str=None,
            store=None,
            raw_text=None,
            currency=None,
            line_items=None,
            source_type="file_inbox",
            file_path=record["archived_path"],
            filename=record["filename"],
        )
        if record.get("content_hash"):
            await data.storage.async_update_receipt(receipt["id"], {"content_hash": record.get("content_hash")})
        await data.activity.async_add_activity(
            kind="receipt_imported_file",
            description=f"Imported receipt file: {record['filename']}",
            payload={"receipt_id": receipt["id"], "filename": record["filename"]},
        )
        await data.storage.async_mark_processed(record["fingerprint"], record)

    duplicates = result.get("duplicates", [])
    for record in duplicates:
        await data.activity.async_add_activity(
            kind="receipt_duplicate_file",
            description=(
                "Archived duplicate receipt file with suffix: "
                f"{record['filename']} -> {os.path.basename(record['archived_path'])}"
            ),
            payload={
                "filename": record["filename"],
                "from_path": record["path"],
                "to_path": record["archived_path"],
            },
        )

    if imported:
        await data.coordinator.async_refresh()

    # Best-effort cleanup of old archived files.
    await hass.async_add_executor_job(_cleanup_archive_sync, archive_path, ttl_days)

    extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
    if extractor_mode == "llm":
        receipts = await data.storage.async_list_receipts()
        queued_any = False
        for receipt in receipts:
            if receipt.get("file_path") and receipt.get("extract_status") in {"pending", "failed"}:
                rid = receipt.get("id")
                if rid:
                    await data.storage.async_update_receipt(
                        rid,
                        {
                            "extract_status": "queued",
                            "extract_queued_at": dt_util.now().isoformat(),
                        },
                    )
                    queued_any = True
                hass.async_create_task(
                    _async_run_llm_for_receipt_file(
                        hass, entry, data, receipt, overwrite=False, force=False
                    )
                )
        if queued_any:
            await data.coordinator.async_refresh()
        return

    endpoint = entry.options.get(CONF_OCR_ENDPOINT_URL, DEFAULT_OCR_ENDPOINT_URL)
    if not endpoint:
        return

    receipts = await data.storage.async_list_receipts()
    queued_any = False
    for receipt in receipts:
        if receipt.get("file_path") and receipt.get("extract_status") == "pending":
            rid = receipt.get("id")
            if rid:
                await data.storage.async_update_receipt(
                    rid,
                    {
                        "extract_status": "queued",
                        "extract_queued_at": dt_util.now().isoformat(),
                    },
                )
                queued_any = True
            hass.async_create_task(_async_run_ocr_for_receipt(hass, entry, data, receipt))
    if queued_any:
        await data.coordinator.async_refresh()


async def _async_scan_inventory_images_inbox(hass: HomeAssistant) -> None:
    data = _get_data(hass)
    entry = _get_entry(hass)
    if data is None or entry is None:
        return

    inbox_path = entry.options.get(
        CONF_INVENTORY_IMAGES_INBOX_PATH, DEFAULT_INVENTORY_IMAGES_INBOX_PATH
    )
    archive_path = entry.options.get(
        CONF_INVENTORY_IMAGES_ARCHIVE_PATH, DEFAULT_INVENTORY_IMAGES_ARCHIVE_PATH
    )
    ttl_days = int(
        entry.options.get(
            CONF_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS, DEFAULT_INVENTORY_IMAGES_ARCHIVE_TTL_DAYS
        )
    )

    if not hass.config.is_allowed_path(inbox_path):
        _LOGGER.warning("Inventory images inbox path is not allowed: %s", inbox_path)
        return
    if not hass.config.is_allowed_path(archive_path):
        _LOGGER.warning("Inventory images archive path is not allowed: %s", archive_path)
        return

    processed = await data.storage.async_get_processed_inventory_images()
    processed |= await data.storage.async_get_inventory_image_fingerprints()
    result = await hass.async_add_executor_job(
        scan_inventory_images_inbox_sync,
        inbox_path,
        archive_path,
        processed,
    )

    added_any = False

    imported = result.get("imported", [])
    for record in imported:
        archived_path = record.get("archived_path")
        if not archived_path:
            continue
        row = await data.storage.async_add_inventory_image(
            archived_path=archived_path,
            filename=str(record.get("filename") or os.path.basename(archived_path)),
            fingerprint=str(record.get("fingerprint") or ""),
            taken_at=record.get("taken_at"),
        )
        await data.activity.async_add_activity(
            kind="inventory_image_imported",
            description=f"Imported inventory image: {row.get('filename')}",
            payload={"image_id": row.get("image_id"), "filename": row.get("filename")},
        )
        await data.storage.async_mark_inventory_image_processed(record["fingerprint"], record)
        added_any = True
        hass.async_create_task(_async_run_inventory_vision_for_image(hass, entry, data, row))

    duplicates = result.get("duplicates", [])
    for record in duplicates:
        await data.activity.async_add_activity(
            kind="inventory_image_duplicate_file",
            description=(
                "Archived duplicate inventory image with suffix: "
                f"{record['filename']} -> {os.path.basename(record['archived_path'])}"
            ),
            payload={
                "filename": record["filename"],
                "from_path": record["path"],
                "to_path": record["archived_path"],
            },
        )
        added_any = True

    if added_any:
        await data.coordinator.async_refresh()

    await hass.async_add_executor_job(cleanup_inventory_images_archive_sync, archive_path, ttl_days)


async def _async_reset_stuck_receipts(
    hass: HomeAssistant,
    entry: ConfigEntry,
    data: GroceryIntelData,
    *,
    older_than_minutes: int,
    include_missing_started: bool,
) -> int:
    """Reset receipts stuck in queued/running state back to pending."""
    if older_than_minutes <= 0:
        return 0

    now = dt_util.as_local(dt_util.now())
    cutoff = now - timedelta(minutes=older_than_minutes)
    receipts = await data.storage.async_list_receipts()
    to_reset: list[str] = []

    for r in receipts:
        rid = r.get("id")
        if not rid:
            continue
        status = r.get("extract_status")
        if status not in {"queued", "running"}:
            continue
        ts_raw = (
            r.get("extract_started_at")
            if status == "running"
            else r.get("extract_queued_at")
        )
        ts = dt_util.parse_datetime(ts_raw) if ts_raw else None
        if ts is None and include_missing_started:
            ts = dt_util.parse_datetime(r.get("created_at")) if r.get("created_at") else None
        if ts is None:
            continue
        if dt_util.as_local(ts) <= cutoff:
            to_reset.append(rid)

    if not to_reset:
        return 0

    for rid in to_reset:
        await data.storage.async_update_receipt(
            rid,
            {
                "extract_status": "pending",
                "extract_started_at": None,
                "extract_queued_at": None,
                "extract_finished_at": None,
                "extract_duration_ms": None,
                "extract_queue_delay_ms": None,
            },
        )

    # Re-trigger processing immediately.
    extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
    if extractor_mode == "llm":
        receipts = await data.storage.async_list_receipts()
        for r in receipts:
            if r.get("id") in set(to_reset):
                await data.storage.async_update_receipt(
                    r["id"],
                    {
                        "extract_status": "queued",
                        "extract_queued_at": dt_util.now().isoformat(),
                    },
                )
                hass.async_create_task(
                    _async_run_llm_for_receipt_file(hass, entry, data, r, overwrite=False, force=True)
                )
    else:
        endpoint = entry.options.get(CONF_OCR_ENDPOINT_URL, DEFAULT_OCR_ENDPOINT_URL)
        if endpoint:
            receipts = await data.storage.async_list_receipts()
            for r in receipts:
                if r.get("id") in set(to_reset):
                    await data.storage.async_update_receipt(
                        r["id"],
                        {
                            "extract_status": "queued",
                            "extract_queued_at": dt_util.now().isoformat(),
                        },
                    )
                    hass.async_create_task(_async_run_ocr_for_receipt(hass, entry, data, r))

    return len(to_reset)


async def _async_run_inventory_vision_for_image(
    hass: HomeAssistant,
    entry: ConfigEntry,
    data: GroceryIntelData,
    image_row: dict[str, Any],
) -> None:
    image_id = image_row.get("image_id")
    if not image_id:
        return
    if image_row.get("status") in {"done", "running"}:
        return

    file_path = image_row.get("file_path")
    filename = image_row.get("filename") or (os.path.basename(file_path) if file_path else "image")
    if not file_path:
        await data.storage.async_update_inventory_image(
            image_id, {"status": "failed", "attempts": int(image_row.get("attempts", 0)) + 1}
        )
        return

    if not hass.config.is_allowed_path(file_path):
        await data.storage.async_update_inventory_image(
            image_id, {"status": "failed", "attempts": int(image_row.get("attempts", 0)) + 1}
        )
        return

    await data.storage.async_update_inventory_image(
        image_id,
        {"status": "running", "attempts": int(image_row.get("attempts", 0)) + 1},
    )

    async with data.ocr_semaphore:
        try:
            img_b64 = await hass.async_add_executor_job(_inventory_read_file_b64_sync, file_path)
        except Exception:
            img_b64 = ""

        if not img_b64:
            await data.storage.async_update_inventory_image(image_id, {"status": "failed"})
            await data.activity.async_add_activity(
                kind="inventory_image_failed",
                description=f"Inventory image analysis failed: {filename}",
                payload={"image_id": image_id, "filename": filename, "reason": "read_failed"},
            )
            await data.coordinator.async_refresh()
            await _async_maybe_notify_telegram_inventory_image(
                data, image_row=image_row, status="failed", reason="read_failed"
            )
            return

        result = await async_analyze_inventory_image(
            hass,
            entry=entry,
            filename=str(filename),
            image_b64=img_b64,
        )

    if not isinstance(result, dict) or not result:
        await data.storage.async_update_inventory_image(image_id, {"status": "failed"})
        await data.activity.async_add_activity(
            kind="inventory_image_failed",
            description=f"Inventory image analysis failed: {filename}",
            payload={"image_id": image_id, "filename": filename, "reason": "llm_unavailable_or_failed"},
        )
        await data.coordinator.async_refresh()
        await _async_maybe_notify_telegram_inventory_image(
            data, image_row=image_row, status="failed", reason="llm_unavailable_or_failed"
        )
        return

    items = normalize_items_with_confidence_from_llm_result(result)
    now = dt_util.as_local(dt_util.now())

    per_product: dict[str, dict[str, Any]] = {}
    detected_rows: list[dict[str, Any]] = []
    boosts: list[dict[str, Any]] = []

    for item in items:
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        try:
            conf = float(item.get("confidence", 0.7))
        except Exception:
            conf = 0.7
        conf = max(0.0, min(1.0, conf))

        pid, match_conf = await data.storage.async_match_or_create_product(label)
        prev_state = data.storage.get_shopping_product_state(pid)
        prev_seen_at = prev_state.get("last_seen_at")
        prev_seen_conf = prev_state.get("last_seen_confidence")
        try:
            prev_seen_conf_f = float(prev_seen_conf) if prev_seen_conf is not None else None
        except Exception:
            prev_seen_conf_f = None

        new_conf = conf if prev_seen_conf_f is None else max(prev_seen_conf_f, conf)

        existing = per_product.get(pid)
        if existing and float(existing.get("last_seen_confidence", 0.0)) >= new_conf:
            continue

        per_product[pid] = {"last_seen_at": now.isoformat(), "last_seen_confidence": new_conf}
        detected_rows.append(
            {
                "label": label,
                "confidence": round(conf, 3),
                "product_id": pid,
                "match_confidence": int(match_conf),
            }
        )
        boosts.append(
            {
                "product_id": pid,
                "label": label,
                "confidence": round(conf, 3),
                "previous": {"last_seen_at": prev_seen_at, "last_seen_confidence": prev_seen_conf},
                "new": {"last_seen_at": now.isoformat(), "last_seen_confidence": round(new_conf, 3)},
            }
        )

    if per_product:
        await data.storage.async_bulk_update_shopping_product_state(per_product)

    await data.storage.async_update_inventory_image(
        image_id,
        {
            "status": "done",
            "raw_result": json.dumps(result, ensure_ascii=False),
            "detected_products": detected_rows,
        },
    )

    await data.activity.async_add_activity(
        kind="inventory_image_analyzed",
        description=f"Inventory image analyzed: {filename} ({len(per_product)} items)",
        payload={
            "image_id": image_id,
            "filename": filename,
            "boosts": boosts,
            "detected": detected_rows,
        },
    )
    await data.coordinator.async_refresh()
    await _async_maybe_notify_telegram_inventory_image(
        data, image_row=image_row, status="done", detected_rows=detected_rows
    )


async def _async_run_ocr_for_receipt(
    hass: HomeAssistant,
    entry: ConfigEntry,
    data: GroceryIntelData,
    receipt: dict[str, Any],
) -> None:
    receipt_id = receipt.get("id")
    if not receipt_id:
        return

    extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
    if extractor_mode == "llm":
        return

    if receipt.get("extract_status") in {"done", "running"}:
        return

    file_path = receipt.get("file_path")
    if not file_path:
        await _async_mark_extract_failed(
            data,
            receipt,
            "Missing receipt file path",
        )
        return

    if not hass.config.is_allowed_path(file_path):
        await _async_mark_extract_failed(
            data,
            receipt,
            "Receipt file path not allowed",
        )
        return
    if not await hass.async_add_executor_job(os.path.exists, file_path):
        await _async_mark_extract_failed(data, receipt, "Receipt file missing")
        return

    endpoint = entry.options.get(CONF_OCR_ENDPOINT_URL, DEFAULT_OCR_ENDPOINT_URL)
    language = entry.options.get(CONF_OCR_LANGUAGE, DEFAULT_OCR_LANGUAGE)
    if not endpoint:
        await _async_mark_extract_failed(data, receipt, "OCR endpoint URL not configured")
        return

    ocr_api_token = entry.options.get(CONF_OCR_API_TOKEN, DEFAULT_OCR_API_TOKEN) or ""
    ocr_api_token_header = entry.options.get(
        CONF_OCR_API_TOKEN_HEADER, DEFAULT_OCR_API_TOKEN_HEADER
    )

    t0 = time.perf_counter()
    async with data.ocr_semaphore:
        now_iso = dt_util.now().isoformat()
        queue_delay_ms = _ms_between_iso(receipt.get("extract_queued_at"), now_iso)
        await data.storage.async_update_receipt(
            receipt_id,
            {
                "extract_status": "running",
                "extract_started_at": now_iso,
                "extract_method": "ocr",
                "extract_provider": "ocr",
                "extract_model": None,
                "extract_queue_delay_ms": queue_delay_ms,
            },
        )
        session = async_get_clientsession(hass)

        async def _do_ocr_request() -> dict[str, Any]:
            def _read_for_upload(path: str, fname: str) -> tuple[bytes, str]:
                ext = os.path.splitext(fname)[1].lower()
                if ext in HEIC_EXTENSIONS:
                    converted = _convert_heic_to_jpeg_bytes_sync(path)
                    if not converted:
                        return b"", fname
                    base = os.path.splitext(fname)[0]
                    processed = _preprocess_receipt_image_bytes_sync(
                        converted, filename=fname
                    ) or converted
                    return processed, f"{base}.jpg"
                if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                    try:
                        with open(path, "rb") as f:
                            raw = f.read()
                    except Exception:
                        return b"", fname
                    processed = _preprocess_receipt_image_bytes_sync(raw, filename=fname)
                    if processed:
                        base = os.path.splitext(fname)[0]
                        return processed, f"{base}.jpg"
                    return raw, fname
                with open(path, "rb") as f:
                    return f.read(), fname

            filename = receipt.get("filename") or os.path.basename(file_path)
            orig_ext = os.path.splitext(filename)[1].lower()
            content, upload_filename = await hass.async_add_executor_job(
                _read_for_upload, file_path, filename
            )
            if not content:
                raise RuntimeError("Failed to read/convert receipt file")

            # If we've preprocessed, it's always JPEG bytes.
            upload_ext = os.path.splitext(upload_filename)[1].lower()
            if upload_ext in {".jpg", ".jpeg"} and orig_ext in (
                HEIC_EXTENSIONS | {".jpg", ".jpeg", ".png", ".webp"}
            ):
                content_type = "image/jpeg"
                if not upload_filename.lower().endswith((".jpg", ".jpeg")):
                    upload_filename = os.path.splitext(upload_filename)[0] + ".jpg"
            else:
                content_type = mimetypes.guess_type(upload_filename)[0] or "application/octet-stream"

            headers = {
                "Content-Type": content_type,
                "X-Filename": upload_filename,
                "X-OCR-Lang": language,
            }
            if ocr_api_token:
                headers[str(ocr_api_token_header)] = str(ocr_api_token)

            async with session.post(
                endpoint,
                timeout=aiohttp.ClientTimeout(total=120),
                data=content,
                headers=headers,
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientError(f"HTTP {response.status}")
                payload = await response.json()

            # If the OCR output likely missed the header (common for phone photos),
            # run a lightweight second OCR pass on the cropped header region and
            # prepend it so store/date extraction has better context.
            if content_type == "image/jpeg" and orig_ext in (
                HEIC_EXTENSIONS | {".jpg", ".jpeg", ".png", ".webp"}
            ):
                try:
                    full_text = payload.get("text") if isinstance(payload, dict) else ""
                    lines = [ln.strip() for ln in str(full_text or "").splitlines() if ln.strip()]
                    first_line = lines[0] if lines else ""
                    store_guess = _parse_store_from_text(str(full_text or ""), filename=filename) or ""
                    low_store = store_guess.casefold()
                    looks_like_slogan = any(
                        kw in low_store
                        for kw in ("affärsidé", "affarside", "welcome", "välkommen", "kvitto", "receipt")
                    )
                    if len(first_line) <= 3 or looks_like_slogan:
                        header_bytes = await hass.async_add_executor_job(
                            _crop_receipt_header_jpeg_bytes_sync, content
                        )
                        if header_bytes:
                            header_headers = dict(headers)
                            header_headers["X-Filename"] = f"header_{upload_filename}"
                            async with session.post(
                                endpoint,
                                timeout=aiohttp.ClientTimeout(total=60),
                                data=header_bytes,
                                headers=header_headers,
                            ) as hresp:
                                if hresp.status < 400:
                                    hpayload = await hresp.json()
                                    htext = hpayload.get("text") if isinstance(hpayload, dict) else ""
                                    if (
                                        isinstance(htext, str)
                                        and htext.strip()
                                        and isinstance(payload, dict)
                                    ):
                                        # Only prepend header text if it looks plausible; otherwise it can
                                        # pollute store parsing (e.g., random symbols from a bad crop).
                                        sample = htext.strip()[:200]
                                        non_ws = [ch for ch in sample if not ch.isspace()]
                                        letters = sum(ch.isalpha() for ch in non_ws)
                                        digits = sum(ch.isdigit() for ch in non_ws)
                                        other = len(non_ws) - letters - digits
                                        letter_ratio = letters / max(1, len(non_ws))
                                        other_ratio = other / max(1, len(non_ws))
                                        has_word = any(
                                            len("".join(ch for ch in w if ch.isalpha())) >= 4
                                            for w in sample.split()
                                        )
                                        if letters >= 10 and has_word and letter_ratio >= 0.30 and other_ratio <= 0.45:
                                            payload["text"] = f"{htext.strip()}\n{full_text or ''}"
                except Exception:
                    # Best-effort only; never fail the main OCR request because of header OCR.
                    pass

            if not isinstance(payload, dict):
                raise RuntimeError("OCR service returned invalid response")
            return payload

        try:
            payload = await asyncio.wait_for(_do_ocr_request(), timeout=180)
        except asyncio.TimeoutError:
            _LOGGER.warning("OCR request timed out for %s", file_path)
            await _async_mark_extract_failed(data, receipt, "OCR request timed out")
            return
        except Exception as err:
            _LOGGER.warning("OCR request failed for %s: %s", file_path, err)
            await _async_mark_extract_failed(data, receipt, "OCR request failed")
            return

    if not isinstance(payload, dict) or not payload.get("ok"):
        await _async_mark_extract_failed(data, receipt, "OCR service returned failure")
        return

    text = payload.get("text") or ""
    confidence = _normalize_confidence(payload.get("confidence"))

    # Hybrid mode: if OCR output for a photo looks unusable, fall back to LLM vision
    # so we don't persist nonsense store/date/total.
    extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
    ext = os.path.splitext(receipt.get("filename") or os.path.basename(file_path))[1].lower()
    if (
        extractor_mode == "hybrid"
        and ext in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
        and _ocr_text_looks_low_quality(text)
    ):
        llm_provider = (entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or "").strip()
        llm_model = (entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or "").strip()
        if llm_provider.lower() == "ollama" and llm_model:
            now_iso = dt_util.now().isoformat()
            await data.storage.async_update_receipt(
                receipt_id,
                {
                    "ocr_text": text,
                    "ocr_confidence": confidence,
                    "extract_status": "queued",
                    "extract_started_at": None,
                    "extract_queued_at": now_iso,
                    "extract_finished_at": None,
                    "extract_duration_ms": None,
                    "extract_queue_delay_ms": None,
                    "extract_method": "llm",
                    "extract_provider": llm_provider.lower(),
                    "extract_model": llm_model,
                },
            )
            await data.activity.async_add_activity(
                kind="ocr_low_quality_fallback",
                description=f"OCR low quality; falling back to vision for {receipt.get('filename', 'receipt')}",
                payload={"receipt_id": receipt_id, "filename": receipt.get("filename")},
            )
            hass.async_create_task(
                _async_run_llm_for_receipt_file(
                    hass,
                    entry,
                    data,
                    {**receipt, "ocr_text": text, "ocr_confidence": confidence, "extract_status": "queued"},
                    overwrite=False,
                    force=True,
                )
            )
            await data.coordinator.async_refresh()
            return

    duration_ms = int((time.perf_counter() - t0) * 1000)
    finished_iso = dt_util.now().isoformat()
    updates: dict[str, Any] = {
        "ocr_text": text,
        "ocr_confidence": confidence,
        "extract_status": "done",
        "extract_started_at": None,
        "extract_queued_at": None,
        "extract_finished_at": finished_iso,
        "extract_duration_ms": duration_ms,
    }

    try:
        extracted_updates = await _async_extract_receipt_fields(
            hass=hass,
            entry=entry,
            receipt=receipt,
            text=text,
            filename=receipt.get("filename") or os.path.basename(file_path),
            overwrite=False,
        )
        if "store_name" in extracted_updates:
            extracted_updates["store_name"] = data.storage.resolve_store_alias(
                _safe_str(extracted_updates.get("store_name"))
            )
        updates.update(extracted_updates)
    except Exception as err:
        _LOGGER.warning("Field extraction failed for %s: %s", file_path, err)
        await data.activity.async_add_activity(
            kind="extract_fields_failed",
            description=f"Field extraction failed for {receipt.get('filename', 'receipt')}",
            payload={"receipt_id": receipt_id, "filename": receipt.get("filename"), "reason": str(err)},
        )

    if updates.get("store_name"):
        store_eid, canonical = await _async_assign_store_entity(
            data,
            store_name=updates.get("store_name"),
            branch_name=None,
            merchant_hints=None,
        )
        if store_eid:
            updates["store_entity_id"] = store_eid
        if canonical:
            updates["store_name"] = canonical

    await data.storage.async_update_receipt(receipt_id, updates)
    if "line_items_raw" in updates:
        await data.storage.async_reprocess_receipts(receipt_id, 1)

    await data.activity.async_add_activity(
        kind="extract_text_completed",
        description=f"Text extraction completed for {receipt.get('filename', 'receipt')}",
        payload={"receipt_id": receipt_id, "filename": receipt.get("filename")},
    )

    # Receipt status changed to "done", so refresh analytics (includes processing counts).
    await data.coordinator.async_refresh()
    await _async_maybe_notify_telegram_receipt(data, receipt_id=receipt_id, status="done")


async def _async_run_llm_for_receipt_file(
    hass: HomeAssistant,
    entry: ConfigEntry,
    data: GroceryIntelData,
    receipt: dict[str, Any],
    *,
    overwrite: bool = False,
    force: bool = False,
) -> None:
    receipt_id = receipt.get("id")
    if not receipt_id:
        return

    if not force and receipt.get("extract_status") in {"done", "running"}:
        return

    file_path = receipt.get("file_path")
    if not file_path:
        await _async_mark_extract_failed(data, receipt, "Missing receipt file path")
        return

    if not hass.config.is_allowed_path(file_path):
        await _async_mark_extract_failed(data, receipt, "Receipt file path not allowed")
        return
    if not await hass.async_add_executor_job(os.path.exists, file_path):
        await _async_mark_extract_failed(data, receipt, "Receipt file missing")
        return

    filename = receipt.get("filename") or os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    def _store_needs_help(store_name: str | None) -> bool:
        s = _safe_str(store_name)
        if not s:
            return True
        low = s.casefold()
        # Addresses / metadata / slogans.
        if any(k in low for k in ("vägen", "vagen", "gatan", "gata", "street", "road", "tel", "orgnr", "org nr")):
            return True
        if any(k in low for k in ("affärsidé", "affarside", "welcome", "välkommen", "kvitto", "receipt")):
            return True
        # If it contains many digits it's likely an address/phone.
        if sum(ch.isdigit() for ch in s) >= 3:
            return True
        # Very short / garbage-ish.
        if len(s) < 3:
            return True
        return False

    def _norm_digits(value: str | None) -> str | None:
        if not value:
            return None
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return digits or None

    def _norm_text(value: str | None) -> str | None:
        if not value:
            return None
        t = re.sub(r"\s{2,}", " ", str(value).strip())
        return t or None

    async def _infer_store_from_history(hints: dict[str, Any]) -> str | None:
        """Infer store_name using merchant hints matched against prior receipts."""
        org = _norm_digits(_safe_str(hints.get("org_number")))
        phone = _norm_digits(_safe_str(hints.get("phone")))
        store_id = _norm_digits(_safe_str(hints.get("store_id")))
        address = _norm_text(_safe_str(hints.get("address")))
        postal = _norm_digits(_safe_str(hints.get("postal_code")))
        city = _norm_text(_safe_str(hints.get("city")))

        if not any([org, phone, store_id, address, postal, city]):
            return None

        receipts = await data.storage.async_list_receipts()
        best_score = 0
        best_store: str | None = None

        for r in receipts:
            store = _safe_str(r.get("store_name"))
            if not store:
                continue
            mh = r.get("merchant_hints") or {}
            if not isinstance(mh, dict):
                mh = {}

            r_org = _norm_digits(_safe_str(mh.get("org_number")))
            r_phone = _norm_digits(_safe_str(mh.get("phone")))
            r_store_id = _norm_digits(_safe_str(mh.get("store_id")))
            r_addr = _norm_text(_safe_str(mh.get("address")))
            r_postal = _norm_digits(_safe_str(mh.get("postal_code")))
            r_city = _norm_text(_safe_str(mh.get("city")))

            score = 0
            if org and r_org and org == r_org:
                score += 10
            if phone and r_phone and phone == r_phone:
                score += 8
            if store_id and r_store_id and store_id == r_store_id:
                score += 8
            if postal and r_postal and postal == r_postal:
                score += 4
            if city and r_city and city.casefold() == r_city.casefold():
                score += 3
            if address and r_addr:
                # Token overlap is robust across minor OCR differences.
                a_tokens = {t for t in re.split(r"[^\\w]+", address.casefold()) if len(t) >= 3}
                r_tokens = {t for t in re.split(r"[^\\w]+", r_addr.casefold()) if len(t) >= 3}
                if a_tokens and r_tokens:
                    overlap = len(a_tokens & r_tokens) / max(1, len(a_tokens))
                    if overlap >= 0.6:
                        score += 6
                    elif overlap >= 0.4:
                        score += 4

            if score > best_score:
                best_score = score
                best_store = store

        # Only accept confident matches.
        if best_score >= 10:
            return best_store
        if best_score >= 8 and (org or phone or store_id):
            return best_store
        return None

    # Concurrency guard (shared with OCR).
    async with data.ocr_semaphore:
        t0 = time.perf_counter()
        now_iso = dt_util.now().isoformat()
        queue_delay_ms = _ms_between_iso(receipt.get("extract_queued_at"), now_iso)
        await data.storage.async_update_receipt(
            receipt_id,
            {
                "extract_status": "running",
                "extract_started_at": now_iso,
                "extract_method": "llm",
                "extract_provider": str(entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or "").lower()
                or None,
                "extract_model": str(entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or "") or None,
                "extract_queue_delay_ms": queue_delay_ms,
            },
        )
        try:
            if ext == ".pdf":
                text = await hass.async_add_executor_job(_extract_pdf_text_sync, file_path)
                if not text.strip():
                    llm_provider = entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or ""
                    llm_model = entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or ""
                    llm_base_url = entry.options.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL) or ""
                    llm_api_key = entry.options.get(CONF_LLM_API_KEY, DEFAULT_LLM_API_KEY) or ""
                    llm_extra = entry.options.get(
                        CONF_LLM_EXTRA_INSTRUCTIONS, DEFAULT_LLM_EXTRA_INSTRUCTIONS
                    ) or ""

                    provider = str(llm_provider).lower()
                    if not llm_model:
                        await _async_mark_extract_failed(data, receipt, "LLM model not configured")
                        return

                    pdf_img_b64 = await hass.async_add_executor_job(
                        _render_pdf_first_page_base64_sync, file_path
                    )
                    if not pdf_img_b64:
                        await _async_mark_extract_failed(
                            data,
                            receipt,
                            "Failed to render PDF to image for vision parsing",
                        )
                        return

                    fields: dict[str, Any] = {}
                    if provider == "ollama":
                        base = llm_base_url or "http://host.docker.internal:11434"
                        fields = await _async_llm_ollama_image_extract(
                            hass=hass,
                            base_url=base,
                            model=llm_model,
                            filename=filename,
                            image_b64=pdf_img_b64,
                            system_prompt=_llm_system_prompt(str(llm_extra)),
                        )
                    elif provider == "openai":
                        base = llm_base_url or "https://api.openai.com"
                        if not llm_api_key:
                            await _async_mark_extract_failed(data, receipt, "LLM API key not configured")
                            return
                        fields = await _async_llm_openai_vision_extract(
                            hass=hass,
                            base_url=base,
                            api_key=llm_api_key,
                            model=llm_model,
                            filename=filename,
                            image_b64=pdf_img_b64,
                            image_mime="image/png",
                            system_prompt=_llm_system_prompt(str(llm_extra)),
                        )
                    else:
                        await _async_mark_extract_failed(
                            data,
                            receipt,
                            "PDF has no extractable text; vision requires llm_provider=ollama or llm_provider=openai",
                        )
                        return

                    combined_fields = dict(fields) if isinstance(fields, dict) else {}
                    combined_fields = _sanitize_llm_fields(combined_fields)

                    # Second pass (OpenAI only): extract merchant identifiers (org/phone/address/store id).
                    # This helps infer store_name when the receipt omits the store name.
                    if provider == "openai" and (
                        _store_needs_help(combined_fields.get("store_name"))
                        or receipt.get("merchant_hints") is None
                    ):
                        store_only = await _async_llm_openai_vision_store_extract(
                            hass=hass,
                            base_url=llm_base_url or "https://api.openai.com",
                            api_key=llm_api_key,
                            model=llm_model,
                            filename=filename,
                            image_b64=pdf_img_b64,
                            image_mime="image/png",
                            system_prompt=_llm_store_only_system_prompt(str(llm_extra)),
                        )
                        if isinstance(store_only, dict):
                            sname = _safe_str(store_only.get("store_name"))
                            sloc = _safe_str(store_only.get("store_location"))
                            if sname and _store_needs_help(combined_fields.get("store_name")):
                                combined_fields["store_name"] = _normalize_store_name(sname)
                    
                            combined_fields["merchant_hints"] = {
                                "org_number": _safe_str(store_only.get("org_number")),
                                "phone": _safe_str(store_only.get("phone")),
                                "store_id": _safe_str(store_only.get("store_id")),
                                "address": _safe_str(store_only.get("address")),
                                "postal_code": _safe_str(store_only.get("postal_code")),
                                "city": _safe_str(store_only.get("city")),
                                "branch_name": sloc,
                            }

                    # Second pass: try line items only (helps vision models focus)
                    if (overwrite or not (receipt.get("line_items_raw") or [])) and not _coerce_line_items(
                        combined_fields.get("line_items")
                    ):
                        items_only: dict[str, Any] = {}
                        if provider == "ollama":
                            base = llm_base_url or "http://host.docker.internal:11434"
                            items_only = await _async_llm_ollama_image_extract(
                                hass=hass,
                                base_url=base,
                                model=llm_model,
                                filename=filename,
                                image_b64=pdf_img_b64,
                                system_prompt=_llm_line_items_only_system_prompt(str(llm_extra)),
                            )
                        elif provider == "openai":
                            base = llm_base_url or "https://api.openai.com"
                            items_only = await _async_llm_openai_vision_extract(
                                hass=hass,
                                base_url=base,
                                api_key=llm_api_key,
                                model=llm_model,
                                filename=filename,
                                image_b64=pdf_img_b64,
                                image_mime="image/png",
                                system_prompt=_llm_line_items_only_system_prompt(str(llm_extra)),
                            )
                        if isinstance(items_only, dict) and "line_items" in items_only:
                            combined_fields["line_items"] = items_only.get("line_items")

                    # Assign a canonical store entity when we have a store name and/or merchant hints.
                    store_eid, canonical = await _async_assign_store_entity(
                        data,
                        store_name=combined_fields.get("store_name"),
                        branch_name=(combined_fields.get("merchant_hints") or {}).get("branch_name")
                        if isinstance(combined_fields.get("merchant_hints"), dict)
                        else None,
                        merchant_hints=combined_fields.get("merchant_hints")
                        if isinstance(combined_fields.get("merchant_hints"), dict)
                        else None,
                    )
                    if store_eid:
                        combined_fields["store_entity_id"] = store_eid
                    if canonical:
                        combined_fields["store_name"] = canonical

                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    finished_iso = dt_util.now().isoformat()
                    updates: dict[str, Any] = {
                        "extract_status": "done" if combined_fields else "failed",
                        "extract_started_at": None,
                        "extract_queued_at": None,
                        "extract_finished_at": finished_iso,
                        "extract_duration_ms": duration_ms,
                        "raw_text": json.dumps(combined_fields, ensure_ascii=False) if combined_fields else None,
                    }

                    if combined_fields:
                        def should_set(field: str) -> bool:
                            if overwrite:
                                return True
                            return receipt.get(field) is None

                        store_val = _safe_str(combined_fields.get("store_name"))
                        if store_val and should_set("store_name"):
                            updates["store_name"] = data.storage.resolve_store_alias(_normalize_store_name(store_val))

                        store_eid_val = _safe_str(combined_fields.get("store_entity_id"))
                        if store_eid_val:
                            updates["store_entity_id"] = store_eid_val

                        merchant_hints = combined_fields.get("merchant_hints")
                        if isinstance(merchant_hints, dict):
                            updates["merchant_hints"] = merchant_hints

                        total_val = combined_fields.get("total")
                        if should_set("total"):
                            if isinstance(total_val, (int, float)):
                                updates["total"] = float(total_val)
                            elif isinstance(total_val, str):
                                parsed = _parse_amount(total_val)
                                if parsed is not None:
                                    updates["total"] = float(parsed)

                        purchased_val = _safe_str(combined_fields.get("purchased_at")) or ""
                        if purchased_val and should_set("purchased_at"):
                            parsed_dt = _parse_date_from_text(purchased_val)
                            if parsed_dt is not None:
                                updates["purchased_at"] = parsed_dt.isoformat()

                        if overwrite or not (receipt.get("line_items_raw") or []):
                            items = _coerce_line_items(combined_fields.get("line_items"))
                            if items:
                                total_for_clean = updates.get("total")
                                if total_for_clean is None:
                                    total_for_clean = receipt.get("total")
                                try:
                                    total_for_clean_f = float(total_for_clean) if total_for_clean is not None else None
                                except Exception:
                                    total_for_clean_f = None
                                updates["line_items_raw"] = _clean_line_items(items, total_for_clean_f)

                        await data.activity.async_add_activity(
                            kind="llm_completed",
                            description=f"LLM parsed receipt PDF (vision): {filename}",
                            payload={"receipt_id": receipt_id, "filename": filename},
                        )
                    else:
                        await data.activity.async_add_activity(
                            kind="llm_failed",
                            description=f"LLM failed to parse receipt PDF (vision): {filename}",
                            payload={"receipt_id": receipt_id, "filename": filename},
                        )

                    await data.storage.async_update_receipt(receipt_id, updates)
                    if "line_items_raw" in updates:
                        await data.storage.async_reprocess_receipts(receipt_id, 1)
                    await data.coordinator.async_refresh()
                    await _async_maybe_notify_telegram_receipt(data, receipt_id=receipt_id, status="done")
                    return

                duration_ms = int((time.perf_counter() - t0) * 1000)
                finished_iso = dt_util.now().isoformat()
                updates: dict[str, Any] = {
                    "ocr_text": text,
                    "ocr_confidence": None,
                    "extract_status": "done",
                    "extract_started_at": None,
                    "extract_queued_at": None,
                    "extract_finished_at": finished_iso,
                    "extract_duration_ms": duration_ms,
                    "extract_method": "pdf_text",
                    "extract_provider": None,
                    "extract_model": None,
                }
                try:
                    extracted = await _async_extract_receipt_fields(
                        hass=hass,
                        entry=entry,
                        receipt=receipt,
                        text=text,
                        filename=filename,
                        overwrite=overwrite,
                    )
                    updates.update(extracted)
                except Exception as err:
                    _LOGGER.warning("Field extraction failed for %s: %s", filename, err)
                    await data.activity.async_add_activity(
                        kind="extract_fields_failed",
                        description=f"Field extraction failed for {filename}",
                        payload={"receipt_id": receipt_id, "filename": filename, "reason": str(err)},
                    )

                if "store_name" in updates:
                    updates["store_name"] = data.storage.resolve_store_alias(_safe_str(updates.get("store_name")))
                if updates.get("store_name"):
                    store_eid, canonical = await _async_assign_store_entity(
                        data,
                        store_name=updates.get("store_name"),
                        branch_name=None,
                        merchant_hints=None,
                    )
                    if store_eid:
                        updates["store_entity_id"] = store_eid
                    if canonical:
                        updates["store_name"] = canonical
                await data.storage.async_update_receipt(receipt_id, updates)
                if "line_items_raw" in updates:
                    await data.storage.async_reprocess_receipts(receipt_id, 1)
                await data.activity.async_add_activity(
                    kind="llm_completed",
                    description=f"LLM parsed receipt file: {filename}",
                    payload={"receipt_id": receipt_id, "filename": filename},
                )
                await data.coordinator.async_refresh()
                await _async_maybe_notify_telegram_receipt(data, receipt_id=receipt_id, status="done")
                return

            elif ext in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
                llm_provider = entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or ""
                llm_model = entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or ""
                llm_base_url = entry.options.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL) or ""
                llm_api_key = entry.options.get(CONF_LLM_API_KEY, DEFAULT_LLM_API_KEY) or ""
                llm_extra = entry.options.get(
                    CONF_LLM_EXTRA_INSTRUCTIONS, DEFAULT_LLM_EXTRA_INSTRUCTIONS
                ) or ""

                provider = str(llm_provider).lower()
                if not llm_model:
                    await _async_mark_extract_failed(data, receipt, "LLM model not configured")
                    return

                img_b64, img_mime = await hass.async_add_executor_job(
                    _read_receipt_image_base64_and_mime_sync, file_path
                )
                if not img_b64:
                    await _async_mark_extract_failed(data, receipt, "Failed to read receipt image")
                    return

                fields: dict[str, Any] = {}
                if provider == "ollama":
                    base = llm_base_url or "http://host.docker.internal:11434"
                    fields = await _async_llm_ollama_image_extract(
                        hass=hass,
                        base_url=base,
                        model=llm_model,
                        filename=filename,
                        image_b64=img_b64,
                        system_prompt=_llm_system_prompt(str(llm_extra)),
                    )
                elif provider == "openai":
                    base = llm_base_url or "https://api.openai.com"
                    if not llm_api_key:
                        await _async_mark_extract_failed(data, receipt, "LLM API key not configured")
                        return
                    fields = await _async_llm_openai_vision_extract(
                        hass=hass,
                        base_url=base,
                        api_key=llm_api_key,
                        model=llm_model,
                        filename=filename,
                        image_b64=img_b64,
                        image_mime=img_mime,
                        system_prompt=_llm_system_prompt(str(llm_extra)),
                    )
                else:
                    await _async_mark_extract_failed(
                        data,
                        receipt,
                        "Image receipts require llm_provider=ollama or llm_provider=openai in llm mode",
                    )
                    return

                combined_fields = dict(fields) if isinstance(fields, dict) else {}
                combined_fields = _sanitize_llm_fields(combined_fields)

                # Second pass (OpenAI only): extract merchant identifiers (org/phone/address/store id).
                # This is useful both for fixing missing store names and for enabling history-based inference
                # when a future receipt omits the store name but includes address/ids.
                if provider == "openai" and (
                    _store_needs_help(combined_fields.get("store_name"))
                    or receipt.get("merchant_hints") is None
                ):
                    store_only = await _async_llm_openai_vision_store_extract(
                        hass=hass,
                        base_url=llm_base_url or "https://api.openai.com",
                        api_key=llm_api_key,
                        model=llm_model,
                        filename=filename,
                        image_b64=img_b64,
                        image_mime="image/jpeg",
                        system_prompt=_llm_store_only_system_prompt(str(llm_extra)),
                    )
                    if isinstance(store_only, dict):
                        sname = _safe_str(store_only.get("store_name"))
                        sloc = _safe_str(store_only.get("store_location"))
                        if sname and _store_needs_help(combined_fields.get("store_name")):
                            combined_fields["store_name"] = _normalize_store_name(sname)

                        combined_fields["merchant_hints"] = {
                            "org_number": _safe_str(store_only.get("org_number")),
                            "phone": _safe_str(store_only.get("phone")),
                            "store_id": _safe_str(store_only.get("store_id")),
                            "address": _safe_str(store_only.get("address")),
                            "postal_code": _safe_str(store_only.get("postal_code")),
                            "city": _safe_str(store_only.get("city")),
                            "branch_name": sloc,
                        }

                        # Resolve store via canonical store table (supports matching by address/ids).
                        store_eid, canonical = await _async_assign_store_entity(
                            data,
                            store_name=combined_fields.get("store_name"),
                            branch_name=sloc,
                            merchant_hints=combined_fields.get("merchant_hints"),
                        )
                        if store_eid:
                            combined_fields["store_entity_id"] = store_eid
                        if canonical and _store_needs_help(combined_fields.get("store_name")):
                            combined_fields["store_name"] = canonical

                # Second pass: try line items only (helps vision models focus)
                if (overwrite or not (receipt.get("line_items_raw") or [])) and not _coerce_line_items(
                    combined_fields.get("line_items")
                ):
                    items_only: dict[str, Any] = {}
                    if provider == "ollama":
                        base = llm_base_url or "http://host.docker.internal:11434"
                        items_only = await _async_llm_ollama_image_extract(
                            hass=hass,
                            base_url=base,
                            model=llm_model,
                            filename=filename,
                            image_b64=img_b64,
                            system_prompt=_llm_line_items_only_system_prompt(str(llm_extra)),
                        )
                    elif provider == "openai":
                        base = llm_base_url or "https://api.openai.com"
                        items_only = await _async_llm_openai_vision_extract(
                            hass=hass,
                            base_url=base,
                            api_key=llm_api_key,
                            model=llm_model,
                            filename=filename,
                            image_b64=img_b64,
                            image_mime=img_mime,
                            system_prompt=_llm_line_items_only_system_prompt(str(llm_extra)),
                        )
                    if isinstance(items_only, dict) and "line_items" in items_only:
                        combined_fields["line_items"] = items_only.get("line_items")

                # Assign a canonical store entity when we have a store name and/or merchant hints.
                store_eid, canonical = await _async_assign_store_entity(
                    data,
                    store_name=combined_fields.get("store_name"),
                    branch_name=(combined_fields.get("merchant_hints") or {}).get("branch_name")
                    if isinstance(combined_fields.get("merchant_hints"), dict)
                    else None,
                    merchant_hints=combined_fields.get("merchant_hints")
                    if isinstance(combined_fields.get("merchant_hints"), dict)
                    else None,
                )
                if store_eid:
                    combined_fields["store_entity_id"] = store_eid
                if canonical:
                    combined_fields["store_name"] = canonical
                duration_ms = int((time.perf_counter() - t0) * 1000)
                finished_iso = dt_util.now().isoformat()
                updates: dict[str, Any] = {
                    "extract_status": "done" if combined_fields else "failed",
                    # Store raw model output for debugging (not OCR text).
                    "raw_text": json.dumps(combined_fields, ensure_ascii=False)
                    if combined_fields
                    else None,
                    "extract_started_at": None,
                    "extract_queued_at": None,
                    "extract_finished_at": finished_iso,
                    "extract_duration_ms": duration_ms,
                }
                if combined_fields:
                    def should_set(field: str) -> bool:
                        if overwrite:
                            return True
                        return receipt.get(field) is None

                    store_val = _safe_str(combined_fields.get("store_name"))
                    if store_val and should_set("store_name"):
                        updates["store_name"] = data.storage.resolve_store_alias(
                            _normalize_store_name(store_val)
                        )

                    store_eid_val = _safe_str(combined_fields.get("store_entity_id"))
                    if store_eid_val:
                        updates["store_entity_id"] = store_eid_val

                    merchant_hints = combined_fields.get("merchant_hints")
                    if isinstance(merchant_hints, dict):
                        updates["merchant_hints"] = merchant_hints

                    total_val = combined_fields.get("total")
                    if should_set("total"):
                        if isinstance(total_val, (int, float)):
                            updates["total"] = float(total_val)
                        elif isinstance(total_val, str):
                            parsed = _parse_amount(total_val)
                            if parsed is not None:
                                updates["total"] = float(parsed)

                    purchased_val = _safe_str(combined_fields.get("purchased_at")) or ""
                    if purchased_val and should_set("purchased_at"):
                        parsed_dt = _parse_date_from_text(purchased_val)
                        if parsed_dt is not None:
                            updates["purchased_at"] = parsed_dt.isoformat()

                    if overwrite or not (receipt.get("line_items_raw") or []):
                        items = _coerce_line_items(combined_fields.get("line_items"))
                        if items:
                            total_for_clean = updates.get("total")
                            if total_for_clean is None:
                                total_for_clean = receipt.get("total")
                            try:
                                total_for_clean_f = (
                                    float(total_for_clean) if total_for_clean is not None else None
                                )
                            except Exception:
                                total_for_clean_f = None
                            updates["line_items_raw"] = _clean_line_items(items, total_for_clean_f)

                    await data.activity.async_add_activity(
                        kind="llm_completed",
                        description=f"LLM parsed receipt image: {filename}",
                        payload={"receipt_id": receipt_id, "filename": filename},
                    )
                else:
                    await data.activity.async_add_activity(
                        kind="llm_failed",
                        description=f"LLM failed to parse receipt image: {filename}",
                        payload={"receipt_id": receipt_id, "filename": filename},
                    )

                await data.storage.async_update_receipt(receipt_id, updates)
                if "line_items_raw" in updates:
                    await data.storage.async_reprocess_receipts(receipt_id, 1)
                await data.coordinator.async_refresh()
                if updates.get("extract_status") == "done":
                    await _async_maybe_notify_telegram_receipt(
                        data, receipt_id=receipt_id, status="done"
                    )
                else:
                    await _async_maybe_notify_telegram_receipt(
                        data,
                        receipt_id=receipt_id,
                        status="failed",
                        reason="LLM returned no usable fields",
                    )
                return

            else:
                await _async_mark_extract_failed(data, receipt, f"Unsupported file extension: {ext}")
                return
        except Exception as err:
            _LOGGER.warning("LLM file parse failed for %s: %s", filename, err)
            await _async_mark_extract_failed(data, receipt, "LLM file parse failed")


def _read_receipt_image_base64_and_mime_sync(path: str) -> tuple[str, str]:
    """Read an image for receipt vision extraction.

    Returns (base64, mime). Preprocessing is best-effort; if preprocessing
    succeeds we return JPEG bytes + `image/jpeg`. Otherwise we return the
    original bytes + a mime inferred from filename.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in HEIC_EXTENSIONS:
        raw = _convert_heic_to_jpeg_bytes_sync(path)
        base_mime = "image/jpeg"
    else:
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except Exception:
            raw = b""
        if ext in {".png"}:
            base_mime = "image/png"
        elif ext in {".webp"}:
            base_mime = "image/webp"
        else:
            base_mime = "image/jpeg"

    if not raw:
        return "", "image/jpeg"

    processed = _preprocess_receipt_image_bytes_sync(raw, filename=os.path.basename(path))
    if processed:
        return base64.b64encode(processed).decode("ascii"), "image/jpeg"

    return base64.b64encode(raw).decode("ascii"), base_mime


def _read_file_base64_sync(path: str) -> str:
    """Legacy helper: base64-encode file for LLM/OCR transport."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
        b64, _mime = _read_receipt_image_base64_and_mime_sync(path)
        if b64:
            return b64
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return ""



def _convert_heic_to_jpeg_bytes_sync(path: str) -> bytes:
    """Best-effort HEIC/HEIF -> JPEG conversion without hard dependencies."""
    # 1) Pillow (optional HEIF support via pillow-heif)
    try:
        try:
            from pillow_heif import register_heif_opener  # type: ignore

            register_heif_opener()
        except Exception:
            pass

        from PIL import Image  # type: ignore

        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue()
    except Exception:
        pass

    # 2) External tools
    tool = shutil.which("magick") or shutil.which("convert") or shutil.which("heif-convert")
    if not tool:
        return b""

    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            prefix="grocery_intel_heic_", suffix=".jpg", delete=False
        )
        tmp_path = tmp.name
        tmp.close()

        base = os.path.basename(tool)
        if base == "heif-convert":
            cmd = [tool, path, tmp_path]
        elif base == "magick":
            cmd = [tool, path, "-auto-orient", "-strip", "-quality", "90", tmp_path]
        else:
            cmd = [tool, path, "-auto-orient", "-strip", "-quality", "90", tmp_path]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return b""
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception:
        return b""
    finally:
        if tmp is not None:
            try:
                os.remove(tmp.name)
            except OSError:
                pass


def _preprocess_receipt_image_bytes_sync(content: bytes, *, filename: str) -> bytes:
    """Best-effort preprocessing to improve OCR/vision on receipt photos."""
    if not content:
        return b""

    try:
        from PIL import Image, ImageEnhance, ImageOps  # type: ignore
    except Exception:
        return b""

    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception:
        return b""

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode not in ("RGB", "L"):
        try:
            img = img.convert("RGB")
        except Exception:
            return b""

    # Resize to a reasonable max dimension.
    max_dim = 2000
    try:
        w, h = img.size
        scale = max(w / max_dim, h / max_dim, 1.0)
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)))
    except Exception:
        pass

    # Attempt to auto-crop to bright receipt area (paper on darker background).
    try:
        gray = img.convert("L")
        thresh = 185
        mask = gray.point(lambda p: 255 if p >= thresh else 0)
        bbox = mask.getbbox()
        if bbox:
            x0, y0, x1, y1 = bbox
            bw = x1 - x0
            bh = y1 - y0
            iw, ih = img.size
            if bw * bh >= int(iw * ih * 0.20) and bw >= int(iw * 0.25) and bh >= int(ih * 0.25):
                pad = max(8, int(min(iw, ih) * 0.01))
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(iw, x1 + pad)
                y1 = min(ih, y1 + pad)
                img = img.crop((x0, y0, x1, y1))
    except Exception:
        pass

    # Enhance for OCR: grayscale + autocontrast + contrast + sharpness.
    try:
        gray = img.convert("L")
        gray = ImageOps.autocontrast(gray)
        gray = ImageEnhance.Contrast(gray).enhance(1.8)
        gray = ImageEnhance.Sharpness(gray).enhance(1.5)
        img = gray
    except Exception:
        pass

    out = io.BytesIO()
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out, format="JPEG", quality=88, optimize=True)
        return out.getvalue()
    except Exception:
        return b""


def _crop_receipt_header_jpeg_bytes_sync(jpeg_bytes: bytes) -> bytes:
    """Crop the top header region of a receipt photo for improved store detection."""
    if not jpeg_bytes:
        return b""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return b""

    import io

    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img.load()
    except Exception:
        return b""

    try:
        w, h = img.size
        if w <= 0 or h <= 0:
            return b""
        # Keep a generous top slice so we don't crop away logos.
        header_h = max(1, int(h * 0.28))
        crop = img.crop((0, 0, w, header_h))
        out = io.BytesIO()
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        crop.save(out, format="JPEG", quality=88, optimize=True)
        return out.getvalue()
    except Exception:
        return b""


def _ocr_text_looks_low_quality(text: str) -> bool:
    """Heuristic to detect unusable OCR output (so we can fall back to vision)."""
    if not text:
        return True
    t = str(text)
    stripped = t.strip()
    if len(stripped) < 80:
        return True

    non_ws = [ch for ch in stripped if not ch.isspace()]
    if len(non_ws) < 80:
        return True

    letters = sum(ch.isalpha() for ch in non_ws)
    digits = sum(ch.isdigit() for ch in non_ws)
    other = len(non_ws) - letters - digits

    letter_ratio = letters / max(1, len(non_ws))
    other_ratio = other / max(1, len(non_ws))

    # If it's mostly symbols/garbage, treat as low quality.
    if letter_ratio < 0.18 and digits < 20:
        return True
    if other_ratio > 0.45:
        return True

    # Some very common receipt anchors (multilingual-lite).
    low = stripped.casefold()
    anchors = ("total", "totalt", "sek", "moms", "vat", "kvitto", "receipt", "summa")
    if not any(a in low for a in anchors) and letter_ratio < 0.25:
        return True

    return False


def _extract_pdf_text_sync(path: str) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as err:
        raise RuntimeError("pypdf is required to parse PDFs in llm mode") from err

    reader = PdfReader(path)
    chunks: list[str] = []
    for page in reader.pages[:10]:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            chunks.append(t.strip())
    # Keep payload size reasonable for LLM calls.
    joined = "\n\n".join(chunks).strip()
    return joined[:30000]


def _render_pdf_first_page_base64_sync(path: str) -> str:
    try:
        import fitz  # type: ignore
    except Exception as err:
        raise RuntimeError("PyMuPDF is required to render PDFs in llm mode") from err

    doc = fitz.open(path)
    if doc.page_count < 1:
        return ""
    page = doc.load_page(0)
    # Moderate scale for readability without huge payloads.
    matrix = fitz.Matrix(1.5, 1.5)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    return base64.b64encode(png_bytes).decode("ascii")


async def _async_llm_ollama_image_extract(
    *,
    hass: HomeAssistant,
    base_url: str,
    model: str,
    filename: str,
    image_b64: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    session = async_get_clientsession(hass)
    url = _join_url(base_url, "/api/chat")
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt or _llm_system_prompt()},
            {
                "role": "user",
                "content": _llm_user_prompt("See attached receipt image.", filename),
                "images": [image_b64],
            },
        ],
        "options": {"temperature": 0},
    }
    try:
        async with session.post(
            url,
            timeout=aiohttp.ClientTimeout(total=120),
            json=payload,
        ) as response:
            if response.status >= 400:
                raise aiohttp.ClientError(f"HTTP {response.status}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (Ollama vision) request failed for %s: %s", filename, err)
        return {}

    if not isinstance(data, dict):
        return {}
    msg = data.get("message", {})
    content = msg.get("content") if isinstance(msg, dict) else ""
    if isinstance(content, str):
        return _extract_first_json_object(content)
    return {}

async def _async_mark_extract_failed(
    data: GroceryIntelData, receipt: dict[str, Any], reason: str
) -> None:
    receipt_id = receipt.get("id")
    if not receipt_id:
        return

    current = await data.storage.async_get_receipt(receipt_id) or {}
    attempts = int(current.get("extract_attempts", receipt.get("extract_attempts", 0)) or 0) + 1
    finished_iso = dt_util.now().isoformat()
    started_iso = _safe_str(current.get("extract_started_at")) or _safe_str(receipt.get("extract_started_at"))
    duration_ms = _ms_between_iso(started_iso, finished_iso)
    await data.storage.async_update_receipt(
        receipt_id,
        {
            "extract_status": "failed",
            "extract_attempts": attempts,
            "extract_started_at": None,
            "extract_queued_at": None,
            "extract_finished_at": finished_iso,
            "extract_duration_ms": duration_ms,
            "ocr_text": None,
            "ocr_confidence": None,
        },
    )

    await data.activity.async_add_activity(
        kind="extract_failed",
        description=f"Extraction failed for {receipt.get('filename', 'receipt')}",
        payload={"receipt_id": receipt_id, "filename": receipt.get("filename"), "reason": reason},
    )

    # Receipt status changed to "failed", so refresh analytics (includes processing counts).
    await data.coordinator.async_refresh()
    await _async_maybe_notify_telegram_receipt(
        data, receipt_id=receipt_id, status="failed", reason=reason
    )


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ms_between_iso(start_iso: str | None, end_iso: str | None) -> int | None:
    """Return milliseconds between two ISO timestamps, or None if not computable."""
    if not start_iso or not end_iso:
        return None
    start_dt = dt_util.parse_datetime(start_iso)
    end_dt = dt_util.parse_datetime(end_iso)
    if start_dt is None or end_dt is None:
        return None
    try:
        delta_s = (end_dt - start_dt).total_seconds()
    except Exception:
        return None
    if delta_s < 0:
        delta_s = 0
    return int(delta_s * 1000)


async def _async_assign_store_entity(
    data: GroceryIntelData,
    *,
    store_name: str | None,
    branch_name: str | None = None,
    merchant_hints: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Return (store_entity_id, canonical_chain_name)."""
    resolved_store_name = data.storage.resolve_store_alias(_safe_str(store_name))
    hints = merchant_hints if isinstance(merchant_hints, dict) else None

    store = await data.storage.async_match_or_create_store(
        chain_name=resolved_store_name,
        branch_name=_safe_str(branch_name),
        merchant_hints=hints,
    )
    if not isinstance(store, dict):
        return None, resolved_store_name
    return _safe_str(store.get("store_entity_id")), _safe_str(store.get("chain_name")) or resolved_store_name


def _extract_first_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1].strip()
    try:
        parsed = json.loads(candidate)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sanitize_llm_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Make LLM output safer: drop obviously-wrong values instead of persisting them."""
    cleaned: dict[str, Any] = {}

    store_val = _safe_str(fields.get("store_name"))
    if store_val:
        store_val = _normalize_store_name(store_val)
    if store_val and 2 <= len(store_val) <= 80:
        # Avoid very short acronym-like "stores" (often hallucinations).
        if len(store_val) <= 4 and store_val.isupper():
            store_val = None
    cleaned["store_name"] = store_val

    # Total: accept only sane ranges.
    total_val = fields.get("total")
    total_num: float | None = None
    if isinstance(total_val, (int, float)):
        total_num = float(total_val)
    elif isinstance(total_val, str):
        parsed = _parse_amount(total_val)
        if parsed is not None:
            total_num = float(parsed)
    if total_num is not None and not (0.0 <= total_num <= 50000.0):
        total_num = None
    cleaned["total"] = total_num

    purchased_val = _safe_str(fields.get("purchased_at")) or ""
    dt = _parse_date_from_text(purchased_val) if purchased_val else None
    # Drop dates far in the future.
    if dt is not None:
        future_cutoff = dt_util.as_local(dt_util.now()) + timedelta(days=2)
        if dt_util.as_local(dt) > future_cutoff:
            dt = None
    cleaned["purchased_at"] = dt.isoformat() if dt is not None else None

    # Line items: keep as-is (coercion happens later), but ensure list.
    li = fields.get("line_items")
    cleaned["line_items"] = li if isinstance(li, list) else (li if li is None else [])
    return cleaned

def _coerce_line_items(value: Any) -> list[dict[str, Any]]:
    """Coerce LLM line_items into the storage format expected by async_add_receipt."""
    if not isinstance(value, list):
        return []

    items: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue

        raw_name = _safe_str(raw.get("raw_name"))
        if not raw_name:
            continue

        line_total_val = raw.get("line_total")
        line_total: float | None = None
        if isinstance(line_total_val, (int, float)):
            line_total = float(line_total_val)
        elif isinstance(line_total_val, str):
            parsed = _parse_amount(line_total_val)
            if parsed is not None:
                line_total = float(parsed)
        if line_total is None:
            continue

        qty_raw = _safe_str(raw.get("qty_raw"))

        unit_price_val = raw.get("unit_price_raw")
        unit_price_raw: float | None = None
        if isinstance(unit_price_val, (int, float)):
            unit_price_raw = float(unit_price_val)
        elif isinstance(unit_price_val, str):
            parsed = _parse_amount(unit_price_val)
            if parsed is not None:
                unit_price_raw = float(parsed)

        items.append(
            {
                "raw_name": raw_name,
                "line_total": line_total,
                "qty_raw": qty_raw,
                "unit_price_raw": unit_price_raw,
            }
        )

    return items[:80]


def _normalize_store_name(value: str | None) -> str | None:
    if not value:
        return None
    name = re.sub(r"\s{2,}", " ", str(value).strip())
    if not name:
        return None

    # Remove common OCR punctuation noise inside the name (keep letters, numbers, spaces, & and apostrophes).
    # Examples: "WILLY:S" -> "WILLYS", "I.C.A" -> "ICA".
    name = re.sub(r"[^\w\s&'’-]", " ", name, flags=re.UNICODE)
    name = re.sub(r"\s{2,}", " ", name).strip()

    lower = name.lower()
    if lower in {"kvitto", "receipt"}:
        return None
    if "kvitto" in lower and len(name) <= 10:
        return None

    # Join sequences like "G i mi" -> "Gimi".
    parts = name.split(" ")
    if parts and all(len(p) == 1 for p in parts):
        name = "".join(parts)
    else:
        rebuilt: list[str] = []
        buf: list[str] = []
        for p in parts:
            if len(p) == 1 and p.isalpha():
                buf.append(p)
                continue
            if buf:
                rebuilt.append("".join(buf))
                buf = []
            rebuilt.append(p)
        if buf:
            rebuilt.append("".join(buf))
        # If OCR split a trailing letter off a word (e.g. "WILLY S"), re-join.
        if len(rebuilt) >= 2 and len(rebuilt[-1]) == 1 and rebuilt[-1].isalpha():
            prev = rebuilt[-2]
            if prev.isalpha() and len(prev) >= 3:
                rebuilt = rebuilt[:-2] + [prev + rebuilt[-1]]
        name = " ".join(rebuilt)

    name = name.strip(" -–—|")
    name = re.sub(r"\s{2,}", " ", name).strip()
    if len(name) < 2:
        return None

    # If it's mostly uppercase letters, normalize casing for readability.
    letters = [ch for ch in name if ch.isalpha()]
    if letters:
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if upper_ratio >= 0.85:
            name = " ".join(w.capitalize() for w in name.split())
    return name


_AMOUNT_RE = re.compile(r"-?[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})")


def _clean_line_items(
    items: list[dict[str, Any]], receipt_total: float | None
) -> list[dict[str, Any]]:
    if not items:
        return []

    skip_contains = (
        "att betala",
        "summa",
        "subtotal",
        "totalt",
        "total",
        "moms",
        "vat",
        "växel",
        "change",
        "kontant",
        "kort",
        "orgnr",
        "org nr",
        "sparat",
        "kvitto",
        "receipt",
    )

    cleaned: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_name = _safe_str(item.get("raw_name")) or ""
        if not raw_name:
            continue

        lower = raw_name.lower()
        # Keep explicit discounts.
        is_discount = "rabatt" in lower or lower.startswith(("-", "discount"))

        if not is_discount and any(s in lower for s in skip_contains):
            continue

        # Rows that are only numbers are typically footer summaries.
        if re.fullmatch(r"[0-9.,\s]+", raw_name):
            continue

        # Rows containing multiple amounts are often totals blocks.
        if not is_discount and len(_AMOUNT_RE.findall(raw_name)) >= 2:
            continue

        line_total = item.get("line_total")
        try:
            line_total_f = float(line_total) if line_total is not None else None
        except Exception:
            line_total_f = None

        # If a row's line_total equals the receipt total but the name isn't a product, drop it.
        if (
            not is_discount
            and receipt_total is not None
            and line_total_f is not None
            and abs(line_total_f - float(receipt_total)) <= 0.01
            and not re.search(r"[A-Za-zÅÄÖåäö]", raw_name)
        ):
            continue

        # Guardrail: if a single line exceeds the receipt total, it's almost certainly mis-parsed.
        if (
            not is_discount
            and receipt_total is not None
            and line_total_f is not None
            and line_total_f > float(receipt_total) + 1.0
        ):
            continue

        cleaned.append(
            {
                "raw_name": raw_name,
                "line_total": item.get("line_total"),
                "qty_raw": _safe_str(item.get("qty_raw")),
                "unit_price_raw": item.get("unit_price_raw"),
            }
        )

    return cleaned[:80]


def _heuristic_line_items_from_text(text: str) -> list[dict[str, Any]]:
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    skip_contains = (
        "orgnr",
        "org nr",
        "tele",
        "kvitto",
        "receipt",
        "moms",
        "vat",
        "summa",
        "att betala",
        "totalt",
        "subtotal",
        "växel",
        "kort",
        "kontant",
    )

    amt_re = re.compile(r"(-?[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2}))\s*$")
    items: list[dict[str, Any]] = []

    for line in lines:
        lower = line.lower()
        if lower.startswith(("---", "===", "***")):
            continue
        if any(s in lower for s in skip_contains):
            continue

        m = amt_re.search(line)
        if not m:
            continue

        amount = _parse_amount(m.group(1))
        if amount is None:
            continue

        name_part = line[: m.start()].strip()
        name_part = re.sub(r"\s{2,}", " ", name_part)
        if not name_part or len(name_part) < 2:
            continue

        qty_raw = None
        # Common patterns like "2st*46,90" or "2 st" or "2pcs"
        qty_m = re.search(r"(\d+(?:\.\d+)?)\s*(st|pcs|pc)\b", lower)
        if qty_m:
            qty_raw = qty_m.group(0)
        else:
            qty_m = re.search(r"\b(\d+)\s*st\*", lower)
            if qty_m:
                qty_raw = qty_m.group(0)

        items.append(
            {
                "raw_name": name_part,
                "line_total": float(amount),
                "qty_raw": qty_raw,
                "unit_price_raw": None,
            }
        )

    # Guardrail: avoid runaway extraction on very noisy text
    return items[:80]


def _llm_system_prompt(extra_instructions: str | None = None) -> str:
    base = (
        "Extract receipt fields. Return JSON only with keys: "
        "store_name (string|null), purchased_at (string|null, ISO 8601 date or datetime), "
        "total (number|null), line_items (array|null). Do not include any extra keys.\n\n"
        "Rules:\n"
        "- Do NOT guess. Only extract values you can clearly read from the receipt. If uncertain, use null.\n"
        "- total must be the grand total to pay (not subtotal, not tax, not change, not a line item).\n"
        "- Only set total if it is explicitly labeled (e.g., 'Total', 'Totalt', 'Att betala') or clearly the final sum.\n"
        "- If amounts use a decimal comma, convert to a JSON number with a dot (e.g., 531,92 -> 531.92).\n"
        "- purchased_at should be the purchase date/time; if multiple dates exist, choose the receipt/purchase date.\n"
        "- Only set purchased_at if the date/time is explicitly printed on the receipt.\n"
        "- store_name should be the merchant/store name (avoid generic words like 'kvitto'/'receipt').\n"
        "- store_name should come from the merchant header/logo line(s), not from street/city/address lines.\n"
        "- line_items should be an array of objects with keys: raw_name (string), line_total (number), "
        "qty_raw (string|null), unit_price_raw (number|null). Use line_total as the total price for that line.\n"
        "- If you cannot extract line items, set line_items to an empty array [].\n"
        "- If the receipt is too blurry or unreadable, return {\"store_name\": null, \"purchased_at\": null, \"total\": null, \"line_items\": []}."
    )
    extra = (extra_instructions or "").strip()
    if not extra:
        return base
    return base + "\n\nUser instructions:\n" + extra


def _llm_line_items_only_system_prompt(extra_instructions: str | None = None) -> str:
    base = (
        "Extract receipt line items. Return JSON only with keys: line_items (array). "
        "Do not include any extra keys.\n\n"
        "Rules:\n"
        "- Do NOT guess. Only include a line item if you can clearly read its name and amount.\n"
        "- line_items must be an array of objects with keys: raw_name (string), line_total (number), "
        "qty_raw (string|null), unit_price_raw (number|null).\n"
        "- Use line_total as the total price for that line (include negative line_total for discounts).\n"
        "- If amounts use a decimal comma, convert to a JSON number with a dot.\n"
        "- If you cannot extract any line items, return an empty array []."
    )
    extra = (extra_instructions or "").strip()
    if not extra:
        return base
    return base + "\n\nUser instructions:\n" + extra


def _llm_store_only_system_prompt(extra_instructions: str | None = None) -> str:
    base = (
        "Extract merchant identification from a receipt image.\n"
        "Return JSON only with keys: store_name (string|null), store_location (string|null), "
        "org_number (string|null), phone (string|null), store_id (string|null), "
        "address (string|null), postal_code (string|null), city (string|null).\n"
        "Do not include any extra keys.\n\n"
        "Rules:\n"
        "- Do NOT guess. Only extract values you can clearly read.\n"
        "- store_name must be the chain/merchant name from the header/logo line(s), not an address.\n"
        "- store_location is optional (e.g., city/branch) and must be near the header.\n"
        "- org_number is a company/org/tax identifier on the receipt (e.g., OrgNr, org.nr, NIT, VAT).\n"
        "- store_id is a store/branch identifier (e.g., Butik, StoreID) if present.\n"
        "- address/postal_code/city should come from the merchant address block.\n"
        "- If unclear, use nulls."
    )
    extra = (extra_instructions or "").strip()
    if not extra:
        return base
    return base + "\n\nUser instructions:\n" + extra


def _llm_user_prompt(text: str, filename: str) -> str:
    return f"filename: {filename}\n\ntext:\n{text}\n\nReturn JSON only."


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _extract_openai_output_text(payload: dict[str, Any]) -> str:
    # Prefer output_text if present (Responses API convenience field).
    out = payload.get("output_text")
    if isinstance(out, str) and out.strip():
        return out

    # Fallback: traverse payload["output"].
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                chunks.append(txt)
    return "\n".join(chunks).strip()


async def _async_llm_openai_extract(
    *,
    hass: HomeAssistant,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    filename: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    session = async_get_clientsession(hass)
    url = _join_url(base_url, "/v1/responses")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "store_name": {"type": ["string", "null"]},
            "purchased_at": {"type": ["string", "null"]},
            "total": {"type": ["number", "null"]},
            "line_items": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "raw_name": {"type": "string"},
                        "line_total": {"type": "number"},
                        "qty_raw": {"type": ["string", "null"]},
                        "unit_price_raw": {"type": ["number", "null"]},
                    },
                    # OpenAI's strict json_schema requires required to include every key in properties.
                    "required": ["raw_name", "line_total", "qty_raw", "unit_price_raw"],
                },
            },
        },
        # OpenAI's strict json_schema requires required to include every key in properties.
        "required": ["store_name", "purchased_at", "total", "line_items"],
    }

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt or _llm_system_prompt()},
            {"role": "user", "content": _llm_user_prompt(text, filename)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "receipt_fields",
                "schema": schema,
                "strict": True,
            }
        },
    }

    try:
        async with session.post(
            url,
            timeout=aiohttp.ClientTimeout(total=60),
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"OpenAI HTTP {response.status}: {body[:500]}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (OpenAI) request failed for %s: %s", filename, err)
        raise

    if not isinstance(data, dict):
        raise RuntimeError("OpenAI returned non-JSON response")
    out_text = _extract_openai_output_text(data)
    return _extract_first_json_object(out_text)


async def _async_llm_openai_vision_extract(
    *,
    hass: HomeAssistant,
    base_url: str,
    api_key: str,
    model: str,
    filename: str,
    image_b64: str,
    image_mime: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """OpenAI Responses API vision extraction for receipt images."""
    session = async_get_clientsession(hass)
    url = _join_url(base_url, "/v1/responses")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "store_name": {"type": ["string", "null"]},
            "purchased_at": {"type": ["string", "null"]},
            "total": {"type": ["number", "null"]},
            "line_items": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "raw_name": {"type": "string"},
                        "line_total": {"type": "number"},
                        "qty_raw": {"type": ["string", "null"]},
                        "unit_price_raw": {"type": ["number", "null"]},
                    },
                    # OpenAI's strict json_schema requires required to include every key in properties.
                    "required": ["raw_name", "line_total", "qty_raw", "unit_price_raw"],
                },
            },
        },
        # OpenAI's strict json_schema requires required to include every key in properties.
        "required": ["store_name", "purchased_at", "total", "line_items"],
    }

    data_url = f"data:{image_mime};base64,{image_b64}"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt or _llm_system_prompt()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _llm_user_prompt("See attached receipt image.", filename),
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "receipt_fields",
                "schema": schema,
                "strict": True,
            }
        },
    }

    try:
        async with session.post(
            url,
            timeout=aiohttp.ClientTimeout(total=120),
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"OpenAI HTTP {response.status}: {body[:500]}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (OpenAI vision) request failed for %s: %s", filename, err)
        raise

    if not isinstance(data, dict):
        raise RuntimeError("OpenAI returned non-JSON response")
    out_text = _extract_openai_output_text(data)
    return _extract_first_json_object(out_text)


async def _async_llm_openai_vision_store_extract(
    *,
    hass: HomeAssistant,
    base_url: str,
    api_key: str,
    model: str,
    filename: str,
    image_b64: str,
    image_mime: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """OpenAI Responses API vision extraction for merchant identifiers."""
    session = async_get_clientsession(hass)
    url = _join_url(base_url, "/v1/responses")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "store_name": {"type": ["string", "null"]},
            "store_location": {"type": ["string", "null"]},
            "org_number": {"type": ["string", "null"]},
            "phone": {"type": ["string", "null"]},
            "store_id": {"type": ["string", "null"]},
            "address": {"type": ["string", "null"]},
            "postal_code": {"type": ["string", "null"]},
            "city": {"type": ["string", "null"]},
        },
        # OpenAI's strict json_schema requires required to include every key in properties.
        "required": [
            "store_name",
            "store_location",
            "org_number",
            "phone",
            "store_id",
            "address",
            "postal_code",
            "city",
        ],
    }

    data_url = f"data:{image_mime};base64,{image_b64}"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt or _llm_store_only_system_prompt()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"filename: {filename}\n\n"
                            "Extract merchant identifiers from the receipt (store name if present, plus org/phone/address/store id). "
                            "Return JSON only."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "receipt_store",
                "schema": schema,
                "strict": True,
            }
        },
    }

    try:
        async with session.post(
            url,
            timeout=aiohttp.ClientTimeout(total=120),
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"OpenAI HTTP {response.status}: {body[:500]}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (OpenAI vision store) request failed for %s: %s", filename, err)
        return {}

    if not isinstance(data, dict):
        return {}
    out_text = _extract_openai_output_text(data)
    return _extract_first_json_object(out_text)


async def _async_llm_azure_extract(
    *,
    hass: HomeAssistant,
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str,
    text: str,
    filename: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    session = async_get_clientsession(hass)
    url = _join_url(
        endpoint,
        f"/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt or _llm_system_prompt()},
            {"role": "user", "content": _llm_user_prompt(text, filename)},
        ],
        "temperature": 0,
        "max_tokens": 250,
    }
    try:
        async with session.post(
            url,
            timeout=aiohttp.ClientTimeout(total=60),
            headers={"api-key": api_key},
            json=payload,
        ) as response:
            if response.status >= 400:
                raise aiohttp.ClientError(f"HTTP {response.status}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (Azure OpenAI) request failed for %s: %s", filename, err)
        return {}

    if not isinstance(data, dict):
        return {}
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = ""
    return _extract_first_json_object(content if isinstance(content, str) else "")


async def _async_llm_anthropic_extract(
    *,
    hass: HomeAssistant,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    filename: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    session = async_get_clientsession(hass)
    url = _join_url(base_url, "/v1/messages")
    payload = {
        "model": model,
        "max_tokens": 250,
        "system": system_prompt or _llm_system_prompt(),
        "messages": [{"role": "user", "content": _llm_user_prompt(text, filename)}],
        "temperature": 0,
    }
    try:
        async with session.post(
            url,
            timeout=aiohttp.ClientTimeout(total=60),
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json=payload,
        ) as response:
            if response.status >= 400:
                raise aiohttp.ClientError(f"HTTP {response.status}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (Anthropic) request failed for %s: %s", filename, err)
        return {}

    if not isinstance(data, dict):
        return {}
    content = data.get("content")
    if isinstance(content, list) and content:
        part = content[0]
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return _extract_first_json_object(part["text"])
    return {}


async def _async_llm_google_extract(
    *,
    hass: HomeAssistant,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    filename: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    session = async_get_clientsession(hass)
    url = _join_url(base_url, f"/v1beta/models/{model}:generateContent?key={api_key}")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": _llm_user_prompt(text, filename)}]}],
        "systemInstruction": {"parts": [{"text": system_prompt or _llm_system_prompt()}]},
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    try:
        async with session.post(
            url,
            timeout=aiohttp.ClientTimeout(total=60),
            json=payload,
        ) as response:
            if response.status >= 400:
                raise aiohttp.ClientError(f"HTTP {response.status}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (Google) request failed for %s: %s", filename, err)
        return {}

    if not isinstance(data, dict):
        return {}
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        content = ""
    return _extract_first_json_object(content if isinstance(content, str) else "")


async def _async_llm_ollama_extract(
    *,
    hass: HomeAssistant,
    base_url: str,
    model: str,
    text: str,
    filename: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    session = async_get_clientsession(hass)
    url = _join_url(base_url, "/api/chat")
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt or _llm_system_prompt()},
            {"role": "user", "content": _llm_user_prompt(text, filename)},
        ],
        "options": {"temperature": 0},
    }
    try:
        async with session.post(
            url,
            timeout=aiohttp.ClientTimeout(total=60),
            json=payload,
        ) as response:
            if response.status >= 400:
                raise aiohttp.ClientError(f"HTTP {response.status}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (Ollama) request failed for %s: %s", filename, err)
        return {}

    if not isinstance(data, dict):
        return {}
    msg = data.get("message", {})
    content = msg.get("content") if isinstance(msg, dict) else ""
    if isinstance(content, str):
        return _extract_first_json_object(content)
    return {}


async def _async_extract_receipt_fields(
    *,
    hass: HomeAssistant,
    entry: ConfigEntry,
    receipt: dict[str, Any],
    text: str,
    filename: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    profile = get_locale_profile(hass)

    extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
    llm_provider = entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or ""
    llm_model = entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or ""
    llm_api_key = entry.options.get(CONF_LLM_API_KEY, DEFAULT_LLM_API_KEY) or ""
    llm_base_url = entry.options.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL) or ""
    azure_api_version = entry.options.get(CONF_AZURE_API_VERSION, DEFAULT_AZURE_API_VERSION) or ""
    llm_extra = entry.options.get(CONF_LLM_EXTRA_INSTRUCTIONS, DEFAULT_LLM_EXTRA_INSTRUCTIONS) or ""
    system_prompt = _llm_system_prompt(str(llm_extra))

    def should_set(field: str) -> bool:
        if overwrite:
            return True
        return receipt.get(field) is None

    def should_set_items() -> bool:
        if overwrite:
            return True
        existing = receipt.get("line_items_raw") or []
        return not existing

    # Heuristic extraction (built-in)
    if extractor_mode in {"heuristic", "hybrid"}:
        parsed_total = _parse_total_from_text(text, profile=profile)
        parsed_date = _parse_date_from_text(text, filename=filename, profile=profile)
        parsed_store = _parse_store_from_text(text, filename=filename, profile=profile)

        if parsed_total is not None and should_set("total"):
            updates["total"] = parsed_total
        if parsed_date is not None and should_set("purchased_at"):
            updates["purchased_at"] = parsed_date.isoformat()
        if parsed_store and should_set("store_name"):
            updates["store_name"] = parsed_store

    # LLM extraction
    needs_line_items = extractor_mode in {"llm", "hybrid"} and should_set_items()

    needs_llm = extractor_mode == "llm" or needs_line_items or (
        extractor_mode == "hybrid"
        and (
            (should_set("total") and "total" not in updates)
            or (should_set("purchased_at") and "purchased_at" not in updates)
            or (should_set("store_name") and "store_name" not in updates)
        )
    )

    if needs_llm:
        if not llm_provider or not llm_model:
            _LOGGER.warning(
                "Extractor mode=%s requires LLM provider+model; using heuristic-only",
                extractor_mode,
            )
            return updates

        llm_fields: dict[str, Any] = {}
        provider = str(llm_provider).lower()
        if provider == "openai":
            base = llm_base_url or "https://api.openai.com"
            if not llm_api_key:
                _LOGGER.warning("LLM extractor (OpenAI) missing API key; skipping")
            else:
                try:
                    llm_fields = await _async_llm_openai_extract(
                        hass=hass,
                        base_url=base,
                        api_key=llm_api_key,
                        model=llm_model,
                        text=text,
                        filename=filename,
                        system_prompt=system_prompt,
                    )
                except Exception:
                    llm_fields = {}
        elif provider == "azure":
            if not llm_base_url:
                _LOGGER.warning("LLM extractor (Azure OpenAI) missing base URL; skipping")
            elif not llm_api_key:
                _LOGGER.warning("LLM extractor (Azure OpenAI) missing API key; skipping")
            else:
                try:
                    llm_fields = await _async_llm_azure_extract(
                        hass=hass,
                        endpoint=llm_base_url,
                        api_key=llm_api_key,
                        deployment=llm_model,
                        api_version=azure_api_version or DEFAULT_AZURE_API_VERSION,
                        text=text,
                        filename=filename,
                        system_prompt=system_prompt,
                    )
                except Exception:
                    llm_fields = {}
        elif provider == "anthropic":
            base = llm_base_url or "https://api.anthropic.com"
            if not llm_api_key:
                _LOGGER.warning("LLM extractor (Anthropic) missing API key; skipping")
            else:
                try:
                    llm_fields = await _async_llm_anthropic_extract(
                        hass=hass,
                        base_url=base,
                        api_key=llm_api_key,
                        model=llm_model,
                        text=text,
                        filename=filename,
                        system_prompt=system_prompt,
                    )
                except Exception:
                    llm_fields = {}
        elif provider == "google":
            base = llm_base_url or "https://generativelanguage.googleapis.com"
            if not llm_api_key:
                _LOGGER.warning("LLM extractor (Google) missing API key; skipping")
            else:
                try:
                    llm_fields = await _async_llm_google_extract(
                        hass=hass,
                        base_url=base,
                        api_key=llm_api_key,
                        model=llm_model,
                        text=text,
                        filename=filename,
                        system_prompt=system_prompt,
                    )
                except Exception:
                    llm_fields = {}
        elif provider == "ollama":
            base = llm_base_url or "http://host.docker.internal:11434"
            try:
                llm_fields = await _async_llm_ollama_extract(
                    hass=hass,
                    base_url=base,
                    model=llm_model,
                    text=text,
                    filename=filename,
                    system_prompt=system_prompt,
                )
            except Exception:
                llm_fields = {}
        else:
            _LOGGER.warning("Unknown LLM provider: %s", llm_provider)

        llm_fields = _sanitize_llm_fields(llm_fields if isinstance(llm_fields, dict) else {})

        if should_set("total") and "total" not in updates:
            total_val = llm_fields.get("total")
            if isinstance(total_val, (int, float)):
                updates["total"] = float(total_val)
        if should_set("purchased_at") and "purchased_at" not in updates:
            purchased_val = _safe_str(llm_fields.get("purchased_at")) or ""
            if purchased_val:
                parsed_dt = _parse_date_from_text(purchased_val)
                if parsed_dt is not None:
                    updates["purchased_at"] = parsed_dt.isoformat()
        if should_set("store_name") and "store_name" not in updates:
            store_val = _safe_str(llm_fields.get("store_name"))
            if store_val:
                updates["store_name"] = store_val

        # Fallback: if LLM didn't produce header fields, try heuristics on available text.
        # This improves robustness for PDF text-layer parsing when the LLM endpoint is unavailable.
        if extractor_mode == "llm":
            if should_set("total") and "total" not in updates:
                parsed_total = _parse_total_from_text(text, profile=profile)
                if parsed_total is not None:
                    updates["total"] = parsed_total
            if should_set("purchased_at") and "purchased_at" not in updates:
                parsed_date = _parse_date_from_text(text, profile=profile)
                if parsed_date is not None:
                    updates["purchased_at"] = parsed_date.isoformat()
            if should_set("store_name") and "store_name" not in updates:
                parsed_store = _parse_store_from_text(text, profile=profile)
                if parsed_store:
                    updates["store_name"] = parsed_store

        if needs_line_items and should_set_items():
            items = _coerce_line_items(llm_fields.get("line_items"))
            if not items and text:
                items = _heuristic_line_items_from_text(text)
            if items:
                updates["line_items_raw"] = items

    if "store_name" in updates:
        updates["store_name"] = _normalize_store_name(_safe_str(updates.get("store_name")))

    if "line_items_raw" in updates:
        total = updates.get("total")
        if total is None:
            total = receipt.get("total")
        try:
            total_f = float(total) if total is not None else None
        except Exception:
            total_f = None
        updates["line_items_raw"] = _clean_line_items(
            list(updates.get("line_items_raw") or []), total_f
        )

    return updates


def _normalize_confidence(value: Any) -> int | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= num <= 1:
        num *= 100
    return max(0, min(100, int(round(num))))


def _parse_total_from_text(text: str, *, profile: LocaleProfile | None = None) -> float | None:
    if not text:
        return None
    p = profile
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    scored: list[tuple[int, int, float]] = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        if p and any(k in lower for k in p.total_negative):
            continue
        if "subtotal" in lower:
            continue

        score = 0
        if p:
            if any(k in lower for k in p.total_strong):
                score += 5
            if any(k in lower for k in p.total_weak):
                score += 3
        else:
            if any(k in lower for k in ("att betala", "attbetala", "summa att betala", "totalt att betala")):
                score += 5
            if any(k in lower for k in ("summa", "total", "totalt", "belopp")):
                score += 3
            if any(k in lower for k in ("amount", "balance", "grand")):
                score += 2
        if score == 0:
            continue

        amounts = _extract_amounts([line])
        for amount in amounts:
            scored.append((score, idx, amount))

    if scored:
        # Prefer strongest keyword; if tie, prefer later line (often the final total).
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]

    all_amounts = _extract_amounts(lines)
    if all_amounts:
        return max(all_amounts)
    return None


def _extract_amounts(lines: list[str]) -> list[float]:
    amounts: list[float] = []
    pattern = re.compile(r"([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2}))")
    for line in lines:
        for match in pattern.findall(line):
            parsed = _parse_amount(match)
            if parsed is not None:
                amounts.append(parsed)
    return amounts


def _parse_amount(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    if value.count(",") and value.count("."):
        value = value.replace(",", "")
        try:
            return float(value)
        except ValueError:
            return None
    if value.count(",") and not value.count("."):
        parts = value.split(",")
        if len(parts[-1]) == 2:
            value = ".".join(parts)
        else:
            value = value.replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def _parse_date_from_text(
    text: str,
    *,
    filename: str | None = None,
    profile: LocaleProfile | None = None,
):
    """Best-effort purchase date extraction from OCR/LLM text (with optional filename fallback)."""
    def _parse_date_like(value: str):
        value = (value or "").strip()
        if not value:
            return None

        # First try HA's parser directly.
        dt = dt_util.parse_datetime(value)
        if dt is not None:
            return dt_util.as_local(dt)
        date_obj = dt_util.parse_date(value)
        if date_obj is not None:
            return dt_util.as_local(dt_util.start_of_local_day(date_obj))

        # Normalize common separators.
        normalized = re.sub(r"[./]", "-", value)

        # YYYYMMDD
        m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", normalized)
        if m:
            y, mo, d = m.groups()
            date_obj = dt_util.parse_date(f"{y}-{mo}-{d}")
            if date_obj is not None:
                return dt_util.as_local(dt_util.start_of_local_day(date_obj))

        # DD-MM-YY / DD-MM-YYYY / YYYY-MM-DD with 1-2 digit components.
        m = re.fullmatch(r"(\d{1,4})-(\d{1,2})-(\d{1,4})", normalized)
        if m:
            a, b, c = m.groups()
            # Heuristic: if first group is 4-digit year => Y-M-D, else D-M-Y.
            if len(a) == 4:
                y, mo, d = a, b, c
            else:
                d, mo, y = a, b, c
            if len(y) == 2:
                y = "20" + y
            date_obj = dt_util.parse_date(f"{y}-{int(mo):02d}-{int(d):02d}")
            if date_obj is not None:
                return dt_util.as_local(dt_util.start_of_local_day(date_obj))

        # Month names (locale-aware), e.g. "11 feb 2024"
        month_map = (profile.month_map if profile else DEFAULT_MONTH_MAP).copy()
        m = re.search(r"\b(\d{1,2})\s*([A-Za-zÅÄÖåäö]+)\s*(\d{2,4})\b", value)
        if m:
            d_s, mon_s, y_s = m.groups()
            mon_key = mon_s.casefold()
            mo = month_map.get(mon_key)
            if mo:
                if len(y_s) == 2:
                    y_s = "20" + y_s
                date_obj = dt_util.parse_date(f"{y_s}-{mo:02d}-{int(d_s):02d}")
                if date_obj is not None:
                    return dt_util.as_local(dt_util.start_of_local_day(date_obj))

        return None

    def _from_filename(name: str) -> Any:
        if not name:
            return None
        # Match YYYY-MM-DD / YYYY_MM_DD / YYYY.MM.DD / YYYYMMDD anywhere in filename.
        m = re.search(r"(\d{4})[-_.](\d{2})[-_.](\d{2})", name)
        if m:
            y, mo, d = m.groups()
            return _parse_date_like(f"{y}-{mo}-{d}")
        m = re.search(r"\b(\d{4})(\d{2})(\d{2})\b", name)
        if m:
            y, mo, d = m.groups()
            return _parse_date_like(f"{y}-{mo}-{d}")
        return None

    if not text:
        return _from_filename(filename or "") if filename else None

    now = dt_util.as_local(dt_util.now())
    future_cutoff = now + timedelta(days=2)

    if profile:
        positive_kw = profile.date_positive
        negative_kw = profile.date_negative
    else:
        positive_kw = ("datum", "date", "köp", "kop", "purchase", "kassa", "tid", "time")
        negative_kw = ("bäst före", "bast fore", "best before", "expiry", "förfall", "forfall", "due")

    candidates: list[tuple[int, Any]] = []
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]

    # Extract from keyword-weighted lines first.
    for ln in lines[:80]:
        low = ln.casefold()
        score = 0
        if any(k in low for k in positive_kw):
            score += 5
        if any(k in low for k in negative_kw):
            score -= 5

        for m in re.finditer(r"\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b", ln):
            dt = _parse_date_like(m.group(0))
            if dt and dt <= future_cutoff:
                candidates.append((score + 2, dt))
        for m in re.finditer(r"\b\d{1,2}[-./]\d{1,2}[-./]\d{2,4}\b", ln):
            dt = _parse_date_like(m.group(0))
            if dt and dt <= future_cutoff:
                candidates.append((score + 1, dt))
        for m in re.finditer(r"\b\d{8}\b", ln):
            dt = _parse_date_like(m.group(0))
            if dt and dt <= future_cutoff:
                candidates.append((score, dt))
        dt = _parse_date_like(ln)
        if dt and dt <= future_cutoff:
            candidates.append((score, dt))

    if candidates:
        # Prefer highest score; tie-break by latest date.
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][1]

    # Fallback: parse from filename (common for archived PDFs like "... 2024-02-22.pdf").
    return _from_filename(filename or "") if filename else None

def _clean_store_candidate(line: str) -> str:
    cleaned = "".join(ch for ch in line if ch.isalpha() or ch in " &'-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _parse_store_from_text(
    text: str,
    *,
    filename: str = "",
    profile: LocaleProfile | None = None,
) -> str | None:
    """Heuristic store name detection intended to work globally (no country DB)."""
    def _from_filename(name: str) -> str | None:
        if not name:
            return None
        base = os.path.splitext(os.path.basename(name))[0].strip()
        if not base:
            return None
        # Strip common trailing date fragments.
        base = re.sub(r"[\s_-]*(\d{4}[-_.]\d{2}[-_.]\d{2}|\d{8})$", "", base).strip()
        # Strip trailing parenthetical copies.
        base = re.sub(r"\s*\(\d+\)\s*$", "", base).strip()
        cleaned = _clean_store_candidate(base)
        if 3 <= len(cleaned) <= 80:
            return cleaned
        return None

    if not text:
        return _from_filename(filename)

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]

    if profile:
        stop_exact = set(profile.store_stop_exact)
        stop_contains = profile.store_stop_contains
        brand_hints = profile.store_brand_hints
        address_hints = profile.store_address_hints
    else:
        # Generic stopwords + receipt boilerplate (multilingual-lite).
        stop_exact = {
            "kvitto",
            "receipt",
            "butik",
            "store",
            "total",
            "summa",
            "datum",
            "date",
            "tid",
            "time",
            "tack",
            "thanks",
            "välkommen",
            "welcome",
        }
        stop_contains = (
            "org.nr",
            "org nr",
            "vat",
            "moms",
            "affärsidé",
            "affarside",
            "tel",
            "telefon",
            "phone",
            "www.",
            "http",
            "address",
            "adress",
            "kvitt",
            "receipt",
            "terminal",
            "kassa",
            "cashier",
        )
        # A small set of chain hints helps in Sweden but doesn't block global usage.
        brand_hints = (
            "ica",
            "coop",
            "willys",
            "hemköp",
            "city gross",
            "lidl",
            "maxi",
            "7-eleven",
            "seven eleven",
        )
        address_hints = (
            "gata",
            "gatan",
            "väg",
            "vagen",
            "vägen",
            "street",
            "st ",
            "road",
            "rd ",
            "ave",
            "avenue",
            "postcode",
            "postnr",
            "postnummer",
        )

    def _has_brand_hint(candidate: str) -> bool:
        low = candidate.casefold()
        if any(h in low for h in brand_hints):
            return True
        # Fuzzy match common OCR quirks: WILLY:S, W1LLYS, etc.
        tokens = [t for t in re.split(r"[\s\-–—|]+", low) if t]
        for token in tokens:
            if len(token) < 4:
                continue
            for hint in brand_hints:
                hint_compact = hint.replace(" ", "")
                tok_compact = token.replace(" ", "")
                if len(tok_compact) < 4:
                    continue
                if SequenceMatcher(None, tok_compact, hint_compact).ratio() >= 0.86:
                    return True
        return False

    def _strip_trailing_contact(raw_line: str) -> str:
        line = str(raw_line or "")
        if not line.strip():
            return ""
        low = line.casefold()
        cut = None
        # Prefer cutting at explicit contact/metadata keywords.
        m = re.search(r"\b(tel|tele|telefon|phone)\b", low)
        if m:
            cut = m.start()
        # Otherwise cut before the first digit (phones, org numbers, etc).
        m2 = re.search(r"\d", line)
        if m2:
            cut = m2.start() if cut is None else min(cut, m2.start())
        if cut is None:
            return line
        return line[:cut].strip(" -–—|,;:")

    def _looks_like_location_line(raw_line: str) -> str | None:
        candidate = _strip_trailing_contact(raw_line)
        cleaned = _clean_store_candidate(candidate)
        if not (3 <= len(cleaned) <= 50):
            return None
        low = cleaned.casefold()
        if low in stop_exact:
            return None
        if any(s in low for s in stop_contains):
            return None
        if any(h in low for h in address_hints):
            return None
        if any(k in low for k in ("summa", "total", "datum", "date", "tid", "time")):
            return None
        return cleaned

    scored: list[tuple[int, int, str]] = []
    max_lines = min(40, len(lines))
    for idx, line in enumerate(lines[:max_lines]):
        candidate = _strip_trailing_contact(line) if any(ch.isdigit() for ch in line) else line
        cleaned = _clean_store_candidate(candidate)
        if not (3 <= len(cleaned) <= 80):
            continue

        lower = cleaned.casefold()
        if lower in stop_exact:
            continue
        if any(s in lower for s in stop_contains):
            continue
        # Avoid rows that look like addresses or metadata.
        if sum(ch.isdigit() for ch in line) >= 6 and not _has_brand_hint(cleaned):
            continue

        score = 0
        # Early lines are more likely to be merchant name.
        if idx <= 2:
            score += 6
        elif idx <= 6:
            score += 3
        elif idx <= 12:
            score += 1

        # Prefer lines with brand hints when present.
        if _has_brand_hint(cleaned):
            score += 6

        # Prefer mostly-letter lines.
        letters = sum(ch.isalpha() for ch in cleaned)
        digits = sum(ch.isdigit() for ch in cleaned)
        if letters >= 5:
            score += 2
        score -= min(3, digits)

        # Penalize lines with common non-merchant words.
        if any(k in lower for k in ("summa", "total", "datum", "date", "tid", "time")):
            score -= 4

        scored.append((score, idx, cleaned))

    # Add filename hint as a candidate with medium strength.
    file_hint = _from_filename(filename)
    if file_hint:
        scored.append((4, -1, file_hint))

    if not scored:
        return None

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    best_score, _, best = scored[0]
    if best_score < 2:
        return None

    # Combine multi-line store headers: e.g. "WILLYS" + "HELSINGBORG VÄLA".
    # Only do this when we have a real line index and the base line looks like a chain/brand.
    best_idx = scored[0][1]
    if best_idx >= 0:
        best_low = best.casefold()
        has_brand = _has_brand_hint(best)
        token_count = len(best.split())
        if has_brand and token_count <= 3:
            parts = [best]
            for offset in (1, 2):
                if best_idx + offset >= len(lines):
                    break
                loc = _looks_like_location_line(lines[best_idx + offset])
                if not loc:
                    break
                # Don't repeat the brand name.
                if loc.casefold() == best_low:
                    continue
                parts.append(loc)
            if len(parts) > 1:
                return " ".join(parts).strip()

    return best


def _scan_inbox_sync(
    inbox_path: str,
    archive_path: str,
    processed: set[str],
    on_success: str,
) -> dict[str, list[dict[str, Any]]]:
    imported: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    try:
        os.makedirs(inbox_path, exist_ok=True)
    except OSError:
        return {"imported": imported, "duplicates": duplicates}
    if not os.path.isdir(inbox_path):
        return {"imported": imported, "duplicates": duplicates}

    os.makedirs(archive_path, exist_ok=True)
    now_ts = time.time()
    seen = set(processed)

    for entry in os.scandir(inbox_path):
        if not entry.is_file():
            continue

        ext = os.path.splitext(entry.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        try:
            stat = entry.stat()
            content_hash = _sha256_file_sync(entry.path)
            # Primary dedupe key: content hash (stable even if filename/path changes).
            fingerprint = f"sha256:{content_hash}" if content_hash else f"{entry.path}|{stat.st_size}|{int(stat.st_mtime)}"
            if fingerprint in seen:
                dest_path = _unique_dest_path(archive_path, entry.name, suffix="_duplicate")
                shutil.move(entry.path, dest_path)
                try:
                    os.utime(dest_path, (now_ts, now_ts))
                except OSError:
                    pass
                duplicates.append(
                    {
                        "fingerprint": fingerprint,
                        "content_hash": content_hash,
                        "path": entry.path,
                        "archived_path": dest_path,
                        "filename": entry.name,
                        "size": stat.st_size,
                        "mtime": int(stat.st_mtime),
                        "processed_at": dt_util.now().isoformat(),
                    }
                )
                continue
            seen.add(fingerprint)

            dest_path = entry.path
            if on_success == "archive":
                dest_path = _unique_dest_path(archive_path, entry.name)
                shutil.move(entry.path, dest_path)
                try:
                    os.utime(dest_path, (now_ts, now_ts))
                except OSError:
                    pass

            imported.append(
                {
                    "fingerprint": fingerprint,
                    "content_hash": content_hash,
                    "path": entry.path,
                    "archived_path": dest_path,
                    "filename": entry.name,
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime),
                    "processed_at": dt_util.now().isoformat(),
                }
            )
        except Exception:
            _LOGGER.exception("Failed to process receipt file: %s", entry.path)

    return {"imported": imported, "duplicates": duplicates}


def _sha256_file_sync(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cleanup_archive_sync(archive_path: str, ttl_days: int) -> int:
    if ttl_days <= 0:
        return 0
    if not os.path.isdir(archive_path):
        return 0

    cutoff = time.time() - (ttl_days * 86400)
    deleted = 0
    for entry in os.scandir(archive_path):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        ext = os.path.splitext(entry.name)[1].lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            continue
        try:
            st = entry.stat()
            if st.st_mtime > cutoff:
                continue
            os.remove(entry.path)
            deleted += 1
        except OSError:
            continue
    return deleted


def _unique_dest_path(dest_dir: str, filename: str, *, suffix: str = "") -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, f"{base}{suffix}{ext}")
    if not os.path.exists(candidate):
        return candidate

    idx = 1
    while True:
        candidate = os.path.join(dest_dir, f"{base}{suffix}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1
