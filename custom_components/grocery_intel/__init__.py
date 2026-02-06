"""Grocery Intel integration."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
import logging
import os
import re
import shutil
from typing import Any

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
    DEFAULT_CURRENCY_SYMBOL,
    DEFAULT_RECEIPTS_INBOX_PATH,
    DEFAULT_RECEIPTS_ARCHIVE_PATH,
    DEFAULT_INBOX_SCAN_INTERVAL_SEC,
    DEFAULT_ON_SUCCESS,
    DEFAULT_OCR_ENDPOINT_URL,
    DEFAULT_OCR_LANGUAGE,
    SERVICE_ADD_RECEIPT,
    SERVICE_UNDO_ACTIVITY,
    SERVICE_REPROCESS_RECEIPTS,
    SERVICE_SCAN_RECEIPTS_INBOX,
    SERVICE_RUN_OCR,
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
    data.unsub_scan = async_track_time_interval(
        hass,
        lambda now: _async_scan_receipts_inbox(hass),
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

        receipt_id = call.data.get("receipt_id")
        receipts = await data.storage.async_list_receipts()

        if receipt_id:
            receipts = [r for r in receipts if r.get("id") == receipt_id]
        else:
            receipts = [r for r in receipts if r.get("ocr_status") == "pending"]

        for receipt in receipts:
            hass.async_create_task(_async_run_ocr_for_receipt(hass, entry, data, receipt))

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
        schema=vol.Schema({vol.Optional("receipt_id"): cv.string}),
    )


def _unregister_services(hass: HomeAssistant) -> None:
    hass.services.async_remove(DOMAIN, SERVICE_ADD_RECEIPT)
    hass.services.async_remove(DOMAIN, SERVICE_UNDO_ACTIVITY)
    hass.services.async_remove(DOMAIN, SERVICE_REPROCESS_RECEIPTS)
    hass.services.async_remove(DOMAIN, SERVICE_SCAN_RECEIPTS_INBOX)
    hass.services.async_remove(DOMAIN, SERVICE_RUN_OCR)


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

    if imported:
        await data.coordinator.async_refresh()


async def _async_run_ocr_for_receipt(
    hass: HomeAssistant,
    entry: ConfigEntry,
    data: GroceryIntelData,
    receipt: dict[str, Any],
) -> None:
    receipt_id = receipt.get("id")
    if not receipt_id:
        return

    if receipt.get("ocr_status") == "done":
        return

    file_path = receipt.get("file_path")
    if not file_path:
        await _async_mark_ocr_failed(
            data,
            receipt,
            "Missing receipt file path",
        )
        return

    if receipt.get("ocr_status") != "pending":
        await data.storage.async_update_receipt(receipt_id, {"ocr_status": "pending"})

    if not hass.config.is_allowed_path(file_path):
        await _async_mark_ocr_failed(
            data,
            receipt,
            "Receipt file path not allowed",
        )
        return

    endpoint = entry.options.get(CONF_OCR_ENDPOINT_URL, DEFAULT_OCR_ENDPOINT_URL)
    language = entry.options.get(CONF_OCR_LANGUAGE, DEFAULT_OCR_LANGUAGE)

    async with data.ocr_semaphore:
        session = async_get_clientsession(hass)
        try:
            async with session.post(
                endpoint,
                json={"path": file_path, "lang": language},
                timeout=aiohttp.ClientTimeout(total=30),
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

    parsed_total = _parse_total_from_text(text)
    parsed_date = _parse_date_from_text(text)
    parsed_store = _parse_store_from_text(text)

    if receipt.get("total") is None and parsed_total is not None:
        updates["total"] = parsed_total
    if receipt.get("purchased_at") is None and parsed_date is not None:
        updates["purchased_at"] = parsed_date.isoformat()
    if receipt.get("store_name") is None and parsed_store:
        updates["store_name"] = parsed_store

    await data.storage.async_update_receipt(receipt_id, updates)

    await data.activity.async_add_activity(
        kind="ocr_completed",
        description=f"OCR completed for {receipt.get('filename', 'receipt')}",
        payload={"receipt_id": receipt_id, "filename": receipt.get("filename")},
    )

    if "total" in updates or "purchased_at" in updates:
        await data.coordinator.async_refresh()


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
    keyword_lines = []
    for line in lines:
        lower = line.lower()
        if "subtotal" in lower:
            continue
        if any(key in lower for key in ("total", "amount", "balance", "grand")):
            keyword_lines.append(line)

    candidates = _extract_amounts(keyword_lines) if keyword_lines else []
    if candidates:
        return max(candidates)

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
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            dt = dt_util.parse_datetime(match)
            if dt is None:
                date_obj = dt_util.parse_date(match)
                if date_obj is None:
                    continue
                dt = dt_util.start_of_local_day(date_obj)
            candidates.append(dt_util.as_local(dt))

    if not candidates:
        return None
    return max(candidates)


def _parse_store_from_text(text: str) -> str | None:
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:5]:
        cleaned = re.sub(r"[^A-Za-z &'-]", "", line).strip()
        if 3 <= len(cleaned) <= 40:
            lower = cleaned.lower()
            if lower in {"receipt", "store", "total"}:
                continue
            return cleaned
    return None


def _scan_inbox_sync(
    inbox_path: str, archive_path: str, processed: set[str], on_success: str
) -> dict[str, list[dict[str, Any]]]:
    imported: list[dict[str, Any]] = []
    if not os.path.isdir(inbox_path):
        return {"imported": imported}

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
                continue

            dest_path = entry.path
            if on_success == "archive":
                dest_path = _unique_archive_path(archive_path, entry.name)
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

    return {"imported": imported}


def _unique_archive_path(archive_path: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(archive_path, filename)
    if not os.path.exists(candidate):
        return candidate

    idx = 1
    while True:
        candidate = os.path.join(archive_path, f"{base}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1
