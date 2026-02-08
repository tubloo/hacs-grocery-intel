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
from typing import Any
import mimetypes

import voluptuous as vol

import aiohttp
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import dt as dt_util

from .activity import ActivityLog
from .storage import ReceiptStorage
from .sensor import GrocerySpendCoordinator
from .const import (
    DOMAIN,
    CONF_CURRENCY_SYMBOL,
    CONF_RECEIPTS_INBOX_PATH,
    CONF_RECEIPTS_ARCHIVE_PATH,
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
)

PLATFORMS: list[Platform] = [Platform.SENSOR]

SIGNAL_REFRESH = "grocery_intel_refresh"

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

_LOGGER = logging.getLogger(__name__)


@dataclass
class GroceryIntelData:
    storage: ReceiptStorage
    activity: ActivityLog
    coordinator: GrocerySpendCoordinator
    ocr_semaphore: asyncio.Semaphore
    unsub_scan: callable | None = None


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration from YAML (not supported)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Grocery Intel from a config entry."""
    storage = ReceiptStorage(hass)
    activity = ActivityLog(hass)
    await storage.async_load()
    await activity.async_load()

    coordinator = GrocerySpendCoordinator(hass, storage, entry)
    await coordinator.async_config_entry_first_refresh()

    data = GroceryIntelData(
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

    await _async_scan_receipts_inbox(hass)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        data = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        if data and data.unsub_scan:
            data.unsub_scan()

    if unload_ok and not hass.config_entries.async_entries(DOMAIN):
        _unregister_services(hass)

    return unload_ok


def _register_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, SERVICE_ADD_RECEIPT):
        return

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

        item_count = len(receipt.get("line_items_raw", []))
        await data.activity.async_add_activity(
            kind="receipt_added",
            description=f"Receipt added (total: {total}, items: {item_count})",
            payload={"receipt_id": receipt["id"], "item_count": item_count},
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
                    if r.get("file_path") and r.get("ocr_status") in {"pending", "failed"}
                ]
            for receipt in receipts:
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
            return

        receipt_id = call.data.get("receipt_id")
        receipts = await data.storage.async_list_receipts()

        if receipt_id:
            receipts = [r for r in receipts if r.get("id") == receipt_id]
        else:
            receipts = [
                r
                for r in receipts
                if r.get("file_path") and r.get("ocr_status") in {"pending", "failed"}
            ]

        for receipt in receipts:
            hass.async_create_task(_async_run_ocr_for_receipt(hass, entry, data, receipt))

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

    hass.services.async_register(
        DOMAIN,
        SERVICE_ADD_RECEIPT,
        handle_add_receipt,
        schema=vol.Schema(
            {
                vol.Required("total"): vol.Coerce(float),
                vol.Optional("date"): cv.string,
                vol.Optional("store"): cv.string,
                vol.Optional("raw_text"): cv.string,
                vol.Optional("line_items"): vol.All(cv.ensure_list, [line_item_schema]),
            }
        ),
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_UNDO_ACTIVITY,
        handle_undo_activity,
        schema=vol.Schema({vol.Required("activity_id"): cv.string}),
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_REPROCESS_RECEIPTS,
        handle_reprocess,
        schema=vol.Schema(
            {
                vol.Optional("receipt_id"): cv.string,
                vol.Optional("limit", default=50): vol.All(int, vol.Range(min=1, max=500)),
            }
        ),
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_SCAN_RECEIPTS_INBOX,
        handle_scan,
        schema=vol.Schema({}),
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RUN_OCR,
        handle_run_ocr,
        schema=vol.Schema(
            {
                vol.Optional("receipt_id"): cv.string,
                vol.Optional("overwrite", default=False): cv.boolean,
            }
        ),
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_REPARSE_RECEIPTS,
        handle_reparse,
        schema=vol.Schema(
            {
                vol.Optional("receipt_id"): cv.string,
                vol.Optional("limit", default=50): vol.All(int, vol.Range(min=1, max=500)),
                vol.Optional("overwrite", default=False): cv.boolean,
            }
        ),
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_ALL_DATA,
        handle_clear_all,
        schema=vol.Schema({vol.Required("confirm"): cv.boolean}),
    )


def _unregister_services(hass: HomeAssistant) -> None:
    hass.services.async_remove(DOMAIN, SERVICE_ADD_RECEIPT)
    hass.services.async_remove(DOMAIN, SERVICE_UNDO_ACTIVITY)
    hass.services.async_remove(DOMAIN, SERVICE_REPROCESS_RECEIPTS)
    hass.services.async_remove(DOMAIN, SERVICE_SCAN_RECEIPTS_INBOX)
    hass.services.async_remove(DOMAIN, SERVICE_RUN_OCR)
    hass.services.async_remove(DOMAIN, SERVICE_REPARSE_RECEIPTS)
    hass.services.async_remove(DOMAIN, SERVICE_CLEAR_ALL_DATA)


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


async def _async_scan_receipts_inbox(hass: HomeAssistant) -> None:
    data = _get_data(hass)
    entry = _get_entry(hass)
    if data is None or entry is None:
        return

    inbox_path = entry.options.get(CONF_RECEIPTS_INBOX_PATH, DEFAULT_RECEIPTS_INBOX_PATH)
    archive_path = entry.options.get(CONF_RECEIPTS_ARCHIVE_PATH, DEFAULT_RECEIPTS_ARCHIVE_PATH)
    on_success = entry.options.get(CONF_ON_SUCCESS, DEFAULT_ON_SUCCESS)

    if not hass.config.is_allowed_path(inbox_path):
        _LOGGER.warning("Inbox path is not allowed: %s", inbox_path)
        return
    if not hass.config.is_allowed_path(archive_path):
        _LOGGER.warning("Archive path is not allowed: %s", archive_path)
        return

    processed = await data.storage.async_get_processed_fingerprints()

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

    extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
    if extractor_mode == "llm":
        receipts = await data.storage.async_list_receipts()
        for receipt in receipts:
            if receipt.get("file_path") and receipt.get("ocr_status") in {"pending", "failed"}:
                hass.async_create_task(
                    _async_run_llm_for_receipt_file(
                        hass, entry, data, receipt, overwrite=False, force=False
                    )
                )
        return

    endpoint = entry.options.get(CONF_OCR_ENDPOINT_URL, DEFAULT_OCR_ENDPOINT_URL)
    if not endpoint:
        return

    receipts = await data.storage.async_list_receipts()
    for receipt in receipts:
        if receipt.get("file_path") and receipt.get("ocr_status") == "pending":
            hass.async_create_task(_async_run_ocr_for_receipt(hass, entry, data, receipt))


async def _async_run_ocr_for_receipt(
    hass: HomeAssistant,
    entry: ConfigEntry,
    data: GroceryIntelData,
    receipt: dict[str, Any],
    *,
    allow_when_llm: bool = False,
) -> None:
    receipt_id = receipt.get("id")
    if not receipt_id:
        return

    extractor_mode = entry.options.get(CONF_EXTRACTOR_MODE, DEFAULT_EXTRACTOR_MODE)
    if extractor_mode == "llm" and not allow_when_llm:
        return

    if receipt.get("ocr_status") in {"done", "running"}:
        return

    file_path = receipt.get("file_path")
    if not file_path:
        await _async_mark_ocr_failed(
            data,
            receipt,
            "Missing receipt file path",
        )
        return

    if not hass.config.is_allowed_path(file_path):
        await _async_mark_ocr_failed(
            data,
            receipt,
            "Receipt file path not allowed",
        )
        return

    endpoint = entry.options.get(CONF_OCR_ENDPOINT_URL, DEFAULT_OCR_ENDPOINT_URL)
    language = entry.options.get(CONF_OCR_LANGUAGE, DEFAULT_OCR_LANGUAGE)
    if not endpoint:
        await _async_mark_ocr_failed(data, receipt, "OCR endpoint URL not configured")
        return

    ocr_api_token = entry.options.get(CONF_OCR_API_TOKEN, DEFAULT_OCR_API_TOKEN) or ""
    ocr_api_token_header = entry.options.get(
        CONF_OCR_API_TOKEN_HEADER, DEFAULT_OCR_API_TOKEN_HEADER
    )

    await data.storage.async_update_receipt(receipt_id, {"ocr_status": "running"})

    async with data.ocr_semaphore:
        session = async_get_clientsession(hass)
        try:
            def _read_bytes(path: str) -> bytes:
                with open(path, "rb") as f:
                    return f.read()

            content = await hass.async_add_executor_job(_read_bytes, file_path)
            filename = receipt.get("filename") or os.path.basename(file_path)
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

            headers = {
                "Content-Type": content_type,
                "X-Filename": filename,
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
        except Exception as err:
            _LOGGER.warning("OCR request failed for %s: %s", file_path, err)
            await _async_mark_ocr_failed(data, receipt, "OCR request failed")
            return

    if not isinstance(payload, dict) or not payload.get("ok"):
        await _async_mark_ocr_failed(data, receipt, "OCR service returned failure")
        return

    text = payload.get("text") or ""
    confidence = _normalize_confidence(payload.get("confidence"))

    updates: dict[str, Any] = {
        "ocr_text": text,
        "ocr_confidence": confidence,
        "ocr_status": "done",
    }

    extracted_updates = await _async_extract_receipt_fields(
        hass=hass,
        entry=entry,
        receipt=receipt,
        text=text,
        filename=receipt.get("filename") or os.path.basename(file_path),
        overwrite=False,
    )
    updates.update(extracted_updates)

    await data.storage.async_update_receipt(receipt_id, updates)
    if "line_items_raw" in updates:
        await data.storage.async_reprocess_receipts(receipt_id, 1)

    await data.activity.async_add_activity(
        kind="ocr_completed",
        description=f"OCR completed for {receipt.get('filename', 'receipt')}",
        payload={"receipt_id": receipt_id, "filename": receipt.get("filename")},
    )

    if "total" in updates or "purchased_at" in updates or "line_items_raw" in updates:
        await data.coordinator.async_refresh()


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

    if not force and receipt.get("ocr_status") in {"done", "running"}:
        return

    file_path = receipt.get("file_path")
    if not file_path:
        await _async_mark_ocr_failed(data, receipt, "Missing receipt file path")
        return

    if not hass.config.is_allowed_path(file_path):
        await _async_mark_ocr_failed(data, receipt, "Receipt file path not allowed")
        return

    filename = receipt.get("filename") or os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    await data.storage.async_update_receipt(receipt_id, {"ocr_status": "running"})

    # Concurrency guard (shared with OCR).
    async with data.ocr_semaphore:
        try:
            if ext == ".pdf":
                text = await hass.async_add_executor_job(_extract_pdf_text_sync, file_path)
                if not text.strip():
                    llm_provider = entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or ""
                    llm_model = entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or ""
                    llm_base_url = entry.options.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL) or ""
                    llm_extra = entry.options.get(
                        CONF_LLM_EXTRA_INSTRUCTIONS, DEFAULT_LLM_EXTRA_INSTRUCTIONS
                    ) or ""

                    if str(llm_provider).lower() != "ollama" or not llm_model:
                        await _async_mark_ocr_failed(
                            data,
                            receipt,
                            "PDF has no extractable text; configure an Ollama vision model or OCR endpoint",
                        )
                        return

                    base = llm_base_url or "http://host.docker.internal:11434"
                    pdf_img_b64 = await hass.async_add_executor_job(
                        _render_pdf_first_page_base64_sync, file_path
                    )
                    if not pdf_img_b64:
                        await _async_mark_ocr_failed(
                            data,
                            receipt,
                            "Failed to render PDF to image for vision parsing",
                        )
                        return

                    fields = await _async_llm_ollama_image_extract(
                        hass=hass,
                        base_url=base,
                        model=llm_model,
                        filename=filename,
                        image_b64=pdf_img_b64,
                        system_prompt=_llm_system_prompt(str(llm_extra)),
                    )

                    updates: dict[str, Any] = {
                        "ocr_text": None,
                        "ocr_confidence": None,
                        "ocr_status": "done" if fields else "failed",
                        "raw_text": json.dumps(fields, ensure_ascii=False) if fields else None,
                    }

                    if fields:
                        def should_set(field: str) -> bool:
                            if overwrite:
                                return True
                            return receipt.get(field) is None

                        store_val = _safe_str(fields.get("store_name"))
                        if store_val and should_set("store_name"):
                            updates["store_name"] = store_val

                        total_val = fields.get("total")
                        if should_set("total"):
                            if isinstance(total_val, (int, float)):
                                updates["total"] = float(total_val)
                            elif isinstance(total_val, str):
                                parsed = _parse_amount(total_val)
                                if parsed is not None:
                                    updates["total"] = float(parsed)

                        purchased_val = _safe_str(fields.get("purchased_at")) or ""
                        if purchased_val and should_set("purchased_at"):
                            parsed_dt = _parse_date_from_text(purchased_val)
                            if parsed_dt is not None:
                                updates["purchased_at"] = parsed_dt.isoformat()

                        if overwrite or not (receipt.get("line_items_raw") or []):
                            items = _coerce_line_items(fields.get("line_items"))
                            if items:
                                updates["line_items_raw"] = items

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
                    return

                updates: dict[str, Any] = {
                    "ocr_text": text,
                    "ocr_confidence": None,
                    "ocr_status": "done",
                }
                extracted = await _async_extract_receipt_fields(
                    hass=hass,
                    entry=entry,
                    receipt=receipt,
                    text=text,
                    filename=filename,
                    overwrite=overwrite,
                )
                updates.update(extracted)
                await data.storage.async_update_receipt(receipt_id, updates)
                if "line_items_raw" in updates:
                    await data.storage.async_reprocess_receipts(receipt_id, 1)
                await data.activity.async_add_activity(
                    kind="llm_completed",
                    description=f"LLM parsed receipt file: {filename}",
                    payload={"receipt_id": receipt_id, "filename": filename},
                )
                await data.coordinator.async_refresh()
                return

            if ext in {".jpg", ".jpeg", ".png"}:
                llm_provider = entry.options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER) or ""
                llm_model = entry.options.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL) or ""
                llm_base_url = entry.options.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL) or ""
                llm_extra = entry.options.get(
                    CONF_LLM_EXTRA_INSTRUCTIONS, DEFAULT_LLM_EXTRA_INSTRUCTIONS
                ) or ""

                if str(llm_provider).lower() != "ollama":
                    await _async_mark_ocr_failed(
                        data,
                        receipt,
                        "Image parsing requires llm_provider=ollama",
                    )
                    return
                if not llm_model:
                    await _async_mark_ocr_failed(data, receipt, "LLM model not configured")
                    return

                base = llm_base_url or "http://host.docker.internal:11434"

                img_b64 = await hass.async_add_executor_job(_read_file_base64_sync, file_path)
                fields = await _async_llm_ollama_image_extract(
                    hass=hass,
                    base_url=base,
                    model=llm_model,
                    filename=filename,
                    image_b64=img_b64,
                    system_prompt=_llm_system_prompt(str(llm_extra)),
                )

                updates: dict[str, Any] = {
                    "ocr_text": None,
                    "ocr_confidence": None,
                    "ocr_status": "done" if fields else "failed",
                    # Store raw model output for debugging (not OCR text).
                    "raw_text": json.dumps(fields, ensure_ascii=False) if fields else None,
                }
                if fields:
                    def should_set(field: str) -> bool:
                        if overwrite:
                            return True
                        return receipt.get(field) is None

                    # Use the same coercion logic as text LLM extraction.
                    store_val = _safe_str(fields.get("store_name"))
                    if store_val and should_set("store_name"):
                        updates["store_name"] = store_val

                    total_val = fields.get("total")
                    if should_set("total"):
                        if isinstance(total_val, (int, float)):
                            updates["total"] = float(total_val)
                        elif isinstance(total_val, str):
                            parsed = _parse_amount(total_val)
                            if parsed is not None:
                                updates["total"] = float(parsed)

                    purchased_val = _safe_str(fields.get("purchased_at")) or ""
                    if purchased_val and should_set("purchased_at"):
                        parsed_dt = _parse_date_from_text(purchased_val)
                        if parsed_dt is not None:
                            updates["purchased_at"] = parsed_dt.isoformat()

                    if overwrite or not (receipt.get("line_items_raw") or []):
                        items = _coerce_line_items(fields.get("line_items"))
                        if items:
                            updates["line_items_raw"] = items

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
                return

            await _async_mark_ocr_failed(data, receipt, f"Unsupported file extension: {ext}")
        except Exception as err:
            _LOGGER.warning("LLM file parse failed for %s: %s", filename, err)
            await _async_mark_ocr_failed(data, receipt, "LLM file parse failed")


def _read_file_base64_sync(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


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

async def _async_mark_ocr_failed(
    data: GroceryIntelData, receipt: dict[str, Any], reason: str
) -> None:
    receipt_id = receipt.get("id")
    if not receipt_id:
        return

    attempts = int(receipt.get("ocr_attempts", 0)) + 1
    await data.storage.async_update_receipt(
        receipt_id,
        {
            "ocr_status": "failed",
            "ocr_attempts": attempts,
            "ocr_text": None,
            "ocr_confidence": None,
        },
    )

    await data.activity.async_add_activity(
        kind="ocr_failed",
        description=f"OCR failed for {receipt.get('filename', 'receipt')}",
        payload={"receipt_id": receipt_id, "filename": receipt.get("filename"), "reason": reason},
    )


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _llm_system_prompt(extra_instructions: str | None = None) -> str:
    base = (
        "Extract receipt fields. Return JSON only with keys: "
        "store_name (string|null), purchased_at (string|null, ISO 8601 date or datetime), "
        "total (number|null), line_items (array|null). Do not include any extra keys.\n\n"
        "Rules:\n"
        "- total must be the grand total to pay (not subtotal, not tax, not change, not a line item).\n"
        "- If amounts use a decimal comma, convert to a JSON number with a dot (e.g., 531,92 -> 531.92).\n"
        "- purchased_at should be the purchase date/time; if multiple dates exist, choose the receipt/purchase date.\n"
        "- store_name should be the merchant/store name (avoid generic words like 'kvitto'/'receipt').\n"
        "- line_items should be an array of objects with keys: raw_name (string), line_total (number), "
        "qty_raw (string|null), unit_price_raw (number|null). Use line_total as the total price for that line.\n"
        "- If you cannot extract line items, set line_items to null."
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
                    "required": ["raw_name", "line_total"],
                },
            },
        },
        "required": ["store_name", "purchased_at", "total"],
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
                raise aiohttp.ClientError(f"HTTP {response.status}")
            data = await response.json()
    except Exception as err:
        _LOGGER.warning("LLM extractor (OpenAI) request failed for %s: %s", filename, err)
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
        parsed_total = _parse_total_from_text(text)
        parsed_date = _parse_date_from_text(text)
        parsed_store = _parse_store_from_text(text)

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
                llm_fields = await _async_llm_openai_extract(
                    hass=hass,
                    base_url=base,
                    api_key=llm_api_key,
                    model=llm_model,
                    text=text,
                    filename=filename,
                    system_prompt=system_prompt,
                )
        elif provider == "azure":
            if not llm_base_url:
                _LOGGER.warning("LLM extractor (Azure OpenAI) missing base URL; skipping")
            elif not llm_api_key:
                _LOGGER.warning("LLM extractor (Azure OpenAI) missing API key; skipping")
            else:
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
        elif provider == "anthropic":
            base = llm_base_url or "https://api.anthropic.com"
            if not llm_api_key:
                _LOGGER.warning("LLM extractor (Anthropic) missing API key; skipping")
            else:
                llm_fields = await _async_llm_anthropic_extract(
                    hass=hass,
                    base_url=base,
                    api_key=llm_api_key,
                    model=llm_model,
                    text=text,
                    filename=filename,
                    system_prompt=system_prompt,
                )
        elif provider == "google":
            base = llm_base_url or "https://generativelanguage.googleapis.com"
            if not llm_api_key:
                _LOGGER.warning("LLM extractor (Google) missing API key; skipping")
            else:
                llm_fields = await _async_llm_google_extract(
                    hass=hass,
                    base_url=base,
                    api_key=llm_api_key,
                    model=llm_model,
                    text=text,
                    filename=filename,
                    system_prompt=system_prompt,
                )
        elif provider == "ollama":
            base = llm_base_url or "http://host.docker.internal:11434"
            llm_fields = await _async_llm_ollama_extract(
                hass=hass,
                base_url=base,
                model=llm_model,
                text=text,
                filename=filename,
                system_prompt=system_prompt,
            )
        else:
            _LOGGER.warning("Unknown LLM provider: %s", llm_provider)

        if should_set("total") and "total" not in updates:
            total_val = llm_fields.get("total")
            if isinstance(total_val, (int, float)):
                updates["total"] = float(total_val)
        if should_set("purchased_at") and "purchased_at" not in updates:
            parsed_dt = _parse_date_from_text(_safe_str(llm_fields.get("purchased_at")) or "")
            if parsed_dt is not None:
                updates["purchased_at"] = parsed_dt.isoformat()
        if should_set("store_name") and "store_name" not in updates:
            store_val = _safe_str(llm_fields.get("store_name"))
            if store_val:
                updates["store_name"] = store_val

        if needs_line_items and should_set_items():
            items = _coerce_line_items(llm_fields.get("line_items"))
            if items:
                updates["line_items_raw"] = items

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


def _parse_total_from_text(text: str) -> float | None:
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    scored: list[tuple[int, int, float]] = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        if "subtotal" in lower:
            continue

        score = 0
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


def _parse_date_from_text(text: str):
    if not text:
        return None

    candidates = []
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",  # 2025-05-18
        r"\b\d{4}/\d{2}/\d{2}\b",  # 2025/05/18
        r"\b\d{4}\.\d{2}\.\d{2}\b",  # 2025.05.18
        r"\b\d{2}/\d{2}/\d{4}\b",  # 18/05/2025
        r"\b\d{2}-\d{2}-\d{4}\b",  # 18-05-2025
        r"\b\d{2}\.\d{2}\.\d{4}\b",  # 18.05.2025
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            normalized = match.replace(".", "-").replace("/", "-")
            dt = dt_util.parse_datetime(normalized)
            if dt is None:
                date_obj = dt_util.parse_date(normalized)
                if date_obj is None:
                    continue
                dt = dt_util.start_of_local_day(date_obj)
            candidates.append(dt_util.as_local(dt))

    if not candidates:
        return None
    return max(candidates)

def _clean_store_candidate(line: str) -> str:
    cleaned = "".join(ch for ch in line if ch.isalpha() or ch in " &'-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _parse_store_from_text(text: str) -> str | None:
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    stopwords = {
        "kvitto",
        "receipt",
        "butik",
        "store",
        "total",
        "summa",
        "datum",
        "tid",
        "tack",
        "välkommen",
        "välkommen åter",
    }
    brand_hints = (
        "ica",
        "coop",
        "willys",
        "hemköp",
        "city gross",
        "lidl",
        "tempo",
        "pressbyrån",
        "7-eleven",
        "seven eleven",
        "maxi",
    )

    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines[:12]):
        cleaned = _clean_store_candidate(line)
        if not (3 <= len(cleaned) <= 60):
            continue
        lower = cleaned.lower()
        if lower in stopwords:
            continue
        if "org" in lower and "nr" in lower:
            continue

        score = 1
        if any(hint in lower for hint in brand_hints):
            score += 5
        if any(ch.isdigit() for ch in line):
            score -= 1
        scored.append((score, idx, cleaned))

    if scored:
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return scored[0][2]
    return None


def _scan_inbox_sync(
    inbox_path: str,
    archive_path: str,
    processed: set[str],
    on_success: str,
) -> dict[str, list[dict[str, Any]]]:
    imported: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    if not os.path.isdir(inbox_path):
        return {"imported": imported, "duplicates": duplicates}

    os.makedirs(archive_path, exist_ok=True)

    for entry in os.scandir(inbox_path):
        if not entry.is_file():
            continue

        ext = os.path.splitext(entry.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        try:
            stat = entry.stat()
            fingerprint = f"{entry.path}|{stat.st_size}|{int(stat.st_mtime)}"
            if fingerprint in processed:
                dest_path = _unique_dest_path(archive_path, entry.name, suffix="_duplicate")
                shutil.move(entry.path, dest_path)
                duplicates.append(
                    {
                        "fingerprint": fingerprint,
                        "path": entry.path,
                        "archived_path": dest_path,
                        "filename": entry.name,
                        "size": stat.st_size,
                        "mtime": int(stat.st_mtime),
                        "processed_at": dt_util.now().isoformat(),
                    }
                )
                continue

            dest_path = entry.path
            if on_success == "archive":
                dest_path = _unique_dest_path(archive_path, entry.name)
                shutil.move(entry.path, dest_path)

            imported.append(
                {
                    "fingerprint": fingerprint,
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
