"""Storage layer for receipts, products, and observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import uuid
from difflib import SequenceMatcher

from homeassistant.core import HomeAssistant
from homeassistant.helpers import storage
from homeassistant.util import dt as dt_util

STORE_VERSION = 5
STORE_KEY = "grocery_intel.data"

MATCH_THRESHOLD = 0.75


@dataclass
class Receipt:
    id: str
    total: float | None
    purchased_at: str | None
    store_name: str | None
    currency: str | None
    raw_text: str | None
    ocr_text: str | None
    ocr_confidence: int | None
    ocr_status: str
    ocr_attempts: int
    created_at: str
    line_items_raw: list[dict[str, Any]]
    source_type: str | None
    file_path: str | None
    filename: str | None


class ReceiptStorage:
    """Persist receipts, products, line items, and observations."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store = storage.Store(hass, STORE_VERSION, STORE_KEY)
        self._data: dict[str, Any] = {
            "receipts": {},
            "line_items": {},
            "products": {},
            "observations": {},
            "processed_files": {},
        }

    async def async_load(self) -> None:
        data = await self._store.async_load()
        if data is None:
            self._data = {
                "receipts": {},
                "line_items": {},
                "products": {},
                "observations": {},
                "processed_files": {},
            }
        else:
            self._data = data
            self._data.setdefault("receipts", {})
            self._data.setdefault("line_items", {})
            self._data.setdefault("products", {})
            self._data.setdefault("observations", {})
            self._data.setdefault("processed_files", {})
            for receipt in self._data["receipts"].values():
                receipt.setdefault("ocr_text", None)
                receipt.setdefault("ocr_confidence", None)
                if "ocr_status" not in receipt:
                    receipt["ocr_status"] = "pending" if receipt.get("file_path") else "done"
                receipt.setdefault("ocr_attempts", 0)

    async def async_save(self) -> None:
        await self._store.async_save(self._data)

    async def async_add_receipt(
        self,
        *,
        total: float | None,
        date_str: str | None,
        store: str | None,
        raw_text: str | None,
        currency: str | None,
        line_items: list[dict[str, Any]] | None,
        source_type: str | None = None,
        file_path: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        receipt_id = uuid.uuid4().hex
        created_at = dt_util.now().isoformat()
        purchased_at = _parse_date(date_str) if date_str else None
        ocr_pending = bool(file_path)
        ocr_status = "pending" if ocr_pending else "done"

        receipt = {
            "id": receipt_id,
            "total": float(total) if total is not None else None,
            "purchased_at": purchased_at.isoformat() if purchased_at else None,
            "store_name": store,
            "currency": currency,
            "raw_text": raw_text,
            "ocr_text": None,
            "ocr_confidence": None,
            "ocr_status": ocr_status,
            "ocr_attempts": 0,
            "created_at": created_at,
            "line_items_raw": line_items or [],
            "source_type": source_type,
            "file_path": file_path,
            "filename": filename,
        }

        self._data["receipts"][receipt_id] = receipt

        if line_items:
            await self._add_line_items(receipt_id, store, purchased_at, line_items)

        await self.async_save()
        return receipt

    async def async_delete_receipt(self, receipt_id: str) -> bool:
        if receipt_id not in self._data["receipts"]:
            return False
        self._data["receipts"].pop(receipt_id)
        self._delete_line_items_for_receipt(receipt_id)
        self._delete_observations_for_receipt(receipt_id)
        await self.async_save()
        return True

    async def async_get_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        return self._data["receipts"].get(receipt_id)

    async def async_update_receipt(self, receipt_id: str, updates: dict[str, Any]) -> bool:
        receipt = self._data["receipts"].get(receipt_id)
        if not receipt:
            return False
        receipt.update(updates)
        await self.async_save()
        return True

    async def async_clear_receipt_ocr(self, receipt_id: str) -> bool:
        receipt = self._data["receipts"].get(receipt_id)
        if not receipt:
            return False
        receipt["ocr_text"] = None
        receipt["ocr_confidence"] = None
        receipt["ocr_attempts"] = 0
        receipt["ocr_status"] = "pending" if receipt.get("file_path") else "done"
        await self.async_save()
        return True

    async def async_list_receipts(self) -> list[dict[str, Any]]:
        return list(self._data["receipts"].values())

    async def async_list_line_items(self) -> list[dict[str, Any]]:
        return list(self._data["line_items"].values())

    async def async_list_products(self) -> list[dict[str, Any]]:
        return list(self._data["products"].values())

    async def async_list_observations(self) -> list[dict[str, Any]]:
        return list(self._data["observations"].values())

    async def async_reprocess_receipts(
        self, receipt_id: str | None, limit: int
    ) -> int:
        receipts = list(self._data["receipts"].values())
        receipts.sort(key=lambda r: r.get("purchased_at", ""), reverse=True)

        if receipt_id:
            receipts = [r for r in receipts if r.get("id") == receipt_id]
        else:
            receipts = receipts[:limit]

        processed = 0
        for receipt in receipts:
            rid = receipt.get("id")
            line_items = receipt.get("line_items_raw", [])
            if not rid or not line_items:
                continue
            purchased_at = dt_util.parse_datetime(receipt.get("purchased_at"))
            if purchased_at is None:
                continue
            store = receipt.get("store_name")

            self._delete_line_items_for_receipt(rid)
            self._delete_observations_for_receipt(rid)
            await self._add_line_items(rid, store, purchased_at, line_items)
            processed += 1

        if processed:
            await self.async_save()
        return processed

    async def async_get_processed_fingerprints(self) -> set[str]:
        return set(self._data.get("processed_files", {}).keys())

    async def async_mark_processed(self, fingerprint: str, record: dict[str, Any]) -> None:
        self._data.setdefault("processed_files", {})[fingerprint] = record
        await self.async_save()

    async def _add_line_items(
        self,
        receipt_id: str,
        store: str | None,
        purchased_at,
        line_items: list[dict[str, Any]],
    ) -> None:
        if purchased_at is None:
            return
        for item in line_items:
            raw_name = _safe_str(item.get("raw_name"))
            if not raw_name:
                continue

            line_total = _safe_float(item.get("line_total"))
            if line_total is None:
                continue

            qty_raw = _safe_str(item.get("qty_raw"))
            unit_price_raw = _safe_float(item.get("unit_price_raw"))

            product_id, confidence = self._match_or_create_product(raw_name)

            line_item_id = uuid.uuid4().hex
            line_item = {
                "line_item_id": line_item_id,
                "receipt_id": receipt_id,
                "raw_name": raw_name,
                "line_total": line_total,
                "qty_raw": qty_raw,
                "unit_price_raw": unit_price_raw,
                "matched_product_id": product_id,
                "match_confidence": confidence,
            }
            self._data["line_items"][line_item_id] = line_item

            unit_price, unit_type = _derive_unit_price(
                line_total, qty_raw, unit_price_raw
            )
            if unit_price is None:
                continue

            observation_id = uuid.uuid4().hex
            observation = {
                "observation_id": observation_id,
                "product_id": product_id,
                "store_name": store,
                "observed_at": purchased_at.isoformat(),
                "pack_price": line_total,
                "unit_price": unit_price,
                "unit_type": unit_type,
                "confidence": confidence,
                "receipt_id": receipt_id,
                "line_item_id": line_item_id,
            }
            self._data["observations"][observation_id] = observation

    def _match_or_create_product(self, raw_name: str) -> tuple[str, int]:
        normalized = _normalize_name(raw_name)
        best_id = None
        best_score = 0.0
        for product in self._data["products"].values():
            candidate_names = [product.get("canonical_name", "")] + list(
                product.get("aliases", [])
            )
            for name in candidate_names:
                score = SequenceMatcher(
                    None, normalized, _normalize_name(str(name))
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_id = product.get("product_id")

        if best_id and best_score >= MATCH_THRESHOLD:
            product = self._data["products"][best_id]
            aliases = set(product.get("aliases", []))
            if raw_name not in aliases:
                aliases.add(raw_name)
                product["aliases"] = sorted(aliases)
            return best_id, int(best_score * 100)

        product_id = uuid.uuid4().hex
        canonical = _canonical_name(raw_name)
        now = dt_util.now().isoformat()
        self._data["products"][product_id] = {
            "product_id": product_id,
            "canonical_name": canonical,
            "aliases": [raw_name],
            "unit_type": "unknown",
            "created_at": now,
            "updated_at": now,
        }
        return product_id, 100

    def _delete_line_items_for_receipt(self, receipt_id: str) -> None:
        to_delete = [
            lid
            for lid, item in self._data["line_items"].items()
            if item.get("receipt_id") == receipt_id
        ]
        for lid in to_delete:
            self._data["line_items"].pop(lid, None)

    def _delete_observations_for_receipt(self, receipt_id: str) -> None:
        to_delete = [
            oid
            for oid, obs in self._data["observations"].items()
            if obs.get("receipt_id") == receipt_id
        ]
        for oid in to_delete:
            self._data["observations"].pop(oid, None)


def _parse_date(date_str: str | None):
    if not date_str:
        return dt_util.now()

    dt = dt_util.parse_datetime(date_str)
    if dt is not None:
        return dt_util.as_local(dt)

    date_obj = dt_util.parse_date(date_str)
    if date_obj is not None:
        return dt_util.as_local(dt_util.start_of_local_day(date_obj))

    raise ValueError(f"Invalid date string: {date_str}")


def _normalize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _canonical_name(value: str) -> str:
    value = value.strip()
    return value.title() if value else value


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_unit_price(
    line_total: float, qty_raw: str | None, unit_price_raw: float | None
) -> tuple[float | None, str]:
    if unit_price_raw is not None:
        return float(unit_price_raw), "unknown"

    if not qty_raw:
        return None, "unknown"

    qty_raw = qty_raw.lower().strip()

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(kg|g|l|ml|pcs|pc)", qty_raw)
    if not match:
        count_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", qty_raw)
        if count_match:
            qty = float(count_match.group(1))
            if qty > 0:
                return line_total / qty, "pcs"
        return None, "unknown"

    qty = float(match.group(1))
    unit = match.group(2)

    if qty <= 0:
        return None, "unknown"

    if unit == "g":
        qty = qty / 1000.0
        unit = "kg"
    if unit == "ml":
        qty = qty / 1000.0
        unit = "l"
    if unit == "pc":
        unit = "pcs"

    return line_total / qty, unit
