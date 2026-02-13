"""Storage layer for receipts, products, and observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import uuid
from difflib import SequenceMatcher
import unicodedata

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
    extract_status: str
    extract_attempts: int
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
            "shopping": {"product_state": {}},
            "inventory_images": {},
            "inventory_images_processed": {},
            "store_aliases": {},
            "stores": {},
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
                "shopping": {"product_state": {}},
                "inventory_images": {},
                "inventory_images_processed": {},
                "store_aliases": {},
                "stores": {},
            }
        else:
            self._data = data
            self._data.setdefault("receipts", {})
            self._data.setdefault("line_items", {})
            self._data.setdefault("products", {})
            self._data.setdefault("observations", {})
            self._data.setdefault("processed_files", {})
            self._data.setdefault("shopping", {"product_state": {}})
            self._data["shopping"].setdefault("product_state", {})
            self._data.setdefault("inventory_images", {})
            self._data.setdefault("inventory_images_processed", {})
            self._data.setdefault("store_aliases", {})
            self._data.setdefault("stores", {})
            for receipt in self._data["receipts"].values():
                receipt.setdefault("ocr_text", None)
                receipt.setdefault("ocr_confidence", None)
                receipt.setdefault("source_meta", {})

                if "extract_status" not in receipt:
                    receipt["extract_status"] = "pending" if receipt.get("file_path") else "done"
                receipt.setdefault("extract_attempts", 0)
                receipt.setdefault("extract_started_at", None)
                receipt.setdefault("extract_queued_at", None)
                receipt.setdefault("extract_finished_at", None)
                receipt.setdefault("extract_duration_ms", None)
                receipt.setdefault("extract_queue_delay_ms", None)
                receipt.setdefault("extract_method", None)
                receipt.setdefault("extract_provider", None)
                receipt.setdefault("extract_model", None)
                receipt.setdefault("store_entity_id", None)
                receipt.setdefault("content_hash", None)

                # Don't persist "queued" across restarts.
                if receipt.get("extract_status") == "queued":
                    receipt["extract_status"] = "pending" if receipt.get("file_path") else "done"

            for row in (self._data.get("inventory_images", {}) or {}).values():
                row.setdefault("taken_at", None)
                row.setdefault("source_type", None)
                row.setdefault("source_meta", {})

    async def async_save(self) -> None:
        await self._store.async_save(self._data)

    async def async_clear_all_data(self) -> None:
        """Clear all stored Grocery Intel data."""
        self._data = {
            "receipts": {},
            "line_items": {},
            "products": {},
            "observations": {},
            "processed_files": {},
            "shopping": {"product_state": {}},
            "inventory_images": {},
            "inventory_images_processed": {},
            "store_aliases": {},
            "stores": {},
        }
        await self.async_save()

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
        extract_status = "pending" if file_path else "done"

        receipt = {
            "id": receipt_id,
            "total": float(total) if total is not None else None,
            "purchased_at": purchased_at.isoformat() if purchased_at else None,
            "store_name": store,
            "store_entity_id": None,
            "currency": currency,
            "raw_text": raw_text,
            "ocr_text": None,
            "ocr_confidence": None,
            "extract_status": extract_status,
            "extract_attempts": 0,
            "extract_started_at": None,
            "extract_queued_at": None,
            "extract_finished_at": None,
            "extract_duration_ms": None,
            "extract_queue_delay_ms": None,
            "extract_method": None,
            "extract_provider": None,
            "extract_model": None,
            "content_hash": None,
            "source_meta": {},
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
        receipt["extract_attempts"] = 0
        receipt["extract_status"] = "pending" if receipt.get("file_path") else "done"
        receipt["extract_started_at"] = None
        receipt["extract_queued_at"] = None
        receipt["extract_finished_at"] = None
        receipt["extract_duration_ms"] = None
        receipt["extract_queue_delay_ms"] = None
        receipt["extract_method"] = None
        receipt["extract_provider"] = None
        receipt["extract_model"] = None
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

    async def async_match_or_create_product(self, raw_name: str) -> tuple[str, int]:
        """Match or create a product for a free-form label and persist changes."""
        product_id, confidence = self._match_or_create_product(raw_name)
        await self.async_save()
        return product_id, confidence

    async def async_reprocess_receipts(
        self, receipt_id: str | None, limit: int
    ) -> int:
        receipts = list(self._data["receipts"].values())
        receipts.sort(key=lambda r: r.get("purchased_at") or "", reverse=True)

        if receipt_id:
            receipts = [r for r in receipts if r.get("id") == receipt_id]
        else:
            receipts = receipts[:limit]

        processed = 0
        for receipt in receipts:
            rid = receipt.get("id")
            if not rid:
                continue

            # Always clear derived data so edits/removals are reflected.
            self._delete_line_items_for_receipt(rid)
            self._delete_observations_for_receipt(rid)

            line_items = receipt.get("line_items_raw", [])
            if not line_items:
                continue
            purchased_at = dt_util.parse_datetime(receipt.get("purchased_at"))
            if purchased_at is None:
                continue
            store = receipt.get("store_name")

            await self._add_line_items(rid, store, purchased_at, line_items)
            processed += 1

        if processed or receipt_id:
            await self.async_save()
        return processed

    async def async_get_processed_fingerprints(self) -> set[str]:
        processed = self._data.get("processed_files", {}) or {}
        if isinstance(processed, dict):
            return set(processed.keys())
        return set()

    async def async_get_receipt_content_hash_fingerprints(self) -> set[str]:
        """Return sha256 fingerprints derived from already-imported receipts.

        This allows inbox scans to dedupe across restarts or after clearing
        processed_files, as long as receipt records remain.
        """
        out: set[str] = set()
        for receipt in (self._data.get("receipts", {}) or {}).values():
            ch = receipt.get("content_hash")
            if isinstance(ch, str) and ch:
                out.add(f"sha256:{ch}")
        return out

    async def async_mark_processed(self, fingerprint: str, record: dict[str, Any]) -> None:
        self._data.setdefault("processed_files", {})[fingerprint] = record
        # Best-effort: also index by content_hash when available, so older path-based fingerprints
        # don't cause duplicates if a file is re-uploaded under a new name.
        ch = record.get("content_hash")
        if isinstance(ch, str) and ch:
            self._data.setdefault("processed_files", {})[f"sha256:{ch}"] = record
        await self.async_save()

    async def async_get_processed_inventory_images(self) -> set[str]:
        return set(self._data.get("inventory_images_processed", {}).keys())

    async def async_get_inventory_image_fingerprints(self) -> set[str]:
        """Return fingerprints from already-imported inventory image rows."""
        out: set[str] = set()
        for row in (self._data.get("inventory_images", {}) or {}).values():
            fp = row.get("fingerprint")
            if isinstance(fp, str) and fp:
                out.add(fp)
        return out

    async def async_mark_inventory_image_processed(self, fingerprint: str, record: dict[str, Any]) -> None:
        self._data.setdefault("inventory_images_processed", {})[fingerprint] = record
        await self.async_save()

    async def async_add_inventory_image(
        self,
        *,
        archived_path: str,
        filename: str,
        fingerprint: str,
        taken_at: str | None = None,
        source_type: str | None = None,
        source_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_id = uuid.uuid4().hex
        now = dt_util.now().isoformat()
        row = {
            "image_id": image_id,
            "file_path": archived_path,
            "filename": filename,
            "fingerprint": fingerprint,
            "taken_at": taken_at,
            "source_type": source_type,
            "source_meta": source_meta or {},
            "status": "pending",
            "attempts": 0,
            "created_at": now,
            "updated_at": now,
            "raw_result": None,
            "detected_products": [],
        }
        self._data.setdefault("inventory_images", {})[image_id] = row
        await self.async_save()
        return row

    async def async_update_inventory_image(self, image_id: str, updates: dict[str, Any]) -> bool:
        row = self._data.get("inventory_images", {}).get(image_id)
        if not row:
            return False
        row.update(updates)
        row["updated_at"] = dt_util.now().isoformat()
        await self.async_save()
        return True

    async def async_list_inventory_images(self) -> list[dict[str, Any]]:
        return list(self._data.get("inventory_images", {}).values())

    def resolve_store_alias(self, store_name: str | None) -> str | None:
        """Resolve a store name via the per-instance alias table."""
        if not store_name:
            return None
        key = _normalize_store_key(store_name)
        if not key:
            return store_name
        aliases = self._data.get("store_aliases", {}) or {}
        mapped = aliases.get(key)
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()
        return store_name

    async def async_add_store_alias(self, raw_store_name: str, canonical_store_name: str) -> bool:
        """Add/update store alias mapping. Returns True if it changed state."""
        raw_key = _normalize_store_key(raw_store_name)
        canonical = (canonical_store_name or "").strip()
        if not raw_key or not canonical:
            return False
        aliases: dict[str, Any] = self._data.setdefault("store_aliases", {})
        prev = aliases.get(raw_key)
        if prev == canonical:
            return False
        aliases[raw_key] = canonical
        await self.async_save()
        return True

    async def async_get_store(self, store_entity_id: str) -> dict[str, Any] | None:
        return self._data.get("stores", {}).get(store_entity_id)

    async def async_list_stores(self) -> list[dict[str, Any]]:
        return list(self._data.get("stores", {}).values())

    async def async_match_or_create_store(
        self,
        *,
        chain_name: str | None,
        branch_name: str | None = None,
        merchant_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Match an existing store or create one.

        - If merchant_hints match an existing store, return that store.
        - Otherwise, create a new store only if chain_name is provided.
        """
        hints = merchant_hints if isinstance(merchant_hints, dict) else {}
        match_id, allow_alias = self._match_store_id(chain_name=chain_name, merchant_hints=hints)
        if match_id:
            store = self._data.get("stores", {}).get(match_id)
            if isinstance(store, dict):
                self._maybe_update_store_from_receipt(
                    store,
                    chain_name=chain_name,
                    branch_name=branch_name,
                    hints=hints,
                    allow_alias=allow_alias,
                )
                await self.async_save()
                return store

        chain = _safe_str(chain_name)
        if not chain:
            return None

        store_entity_id = uuid.uuid4().hex
        now = dt_util.now().isoformat()
        store = {
            "store_entity_id": store_entity_id,
            "chain_name": chain,
            "branch_name": _safe_str(branch_name),
            "address": _safe_str(hints.get("address")),
            "postal_code": _safe_str(hints.get("postal_code")),
            "city": _safe_str(hints.get("city")),
            "org_number": _safe_str(hints.get("org_number")),
            "phone": _safe_str(hints.get("phone")),
            "store_id": _safe_str(hints.get("store_id")),
            "aliases": sorted({chain}),
            "created_at": now,
            "updated_at": now,
        }
        self._data.setdefault("stores", {})[store_entity_id] = store
        await self.async_save()
        return store

    def _maybe_update_store_from_receipt(
        self,
        store: dict[str, Any],
        *,
        chain_name: str | None,
        branch_name: str | None,
        hints: dict[str, Any],
        allow_alias: bool,
    ) -> None:
        changed = False
        now = dt_util.now().isoformat()

        chain = _safe_str(chain_name)
        if chain:
            # Only learn name aliases when we believe it's the same chain (e.g., WILLY:S -> Willys).
            if allow_alias:
                aliases = set(store.get("aliases") or [])
                if chain not in aliases:
                    aliases.add(chain)
                    store["aliases"] = sorted(aliases)
                    changed = True
            # Prefer a "clean" chain name if current is empty.
            if not _safe_str(store.get("chain_name")):
                store["chain_name"] = chain
                changed = True

        # Fill missing fields from hints, but don't overwrite existing ones.
        for key in ("branch_name", "address", "postal_code", "city", "org_number", "phone", "store_id"):
            current = _safe_str(store.get(key))
            if current:
                continue
            if key == "branch_name":
                candidate = _safe_str(branch_name)
            else:
                candidate = _safe_str(hints.get(key))
            if candidate:
                store[key] = candidate
                changed = True

        if changed:
            store["updated_at"] = now

    def _match_store_id(
        self, *, chain_name: str | None, merchant_hints: dict[str, Any]
    ) -> tuple[str | None, bool]:
        stores = self._data.get("stores", {}) or {}
        if not isinstance(stores, dict) or not stores:
            return None, False

        def _norm_digits(value: Any) -> str | None:
            if value is None:
                return None
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            return digits or None

        def _norm_text(value: Any) -> str | None:
            if value is None:
                return None
            t = re.sub(r"\s{2,}", " ", str(value).strip())
            return t or None

        org = _norm_digits(merchant_hints.get("org_number"))
        phone = _norm_digits(merchant_hints.get("phone"))
        store_id = _norm_digits(merchant_hints.get("store_id"))
        address = _norm_text(merchant_hints.get("address"))
        postal = _norm_digits(merchant_hints.get("postal_code"))
        city = _norm_text(merchant_hints.get("city"))

        chain_key = _normalize_store_key(chain_name or "") if chain_name else ""

        best_score = 0
        best_id: str | None = None
        best_allow_alias = False

        for sid, s in stores.items():
            if not isinstance(s, dict):
                continue
            score = 0
            s_org = _norm_digits(s.get("org_number"))
            s_phone = _norm_digits(s.get("phone"))
            s_store_id = _norm_digits(s.get("store_id"))
            s_addr = _norm_text(s.get("address"))
            s_postal = _norm_digits(s.get("postal_code"))
            s_city = _norm_text(s.get("city"))
            s_chain = _normalize_store_key(s.get("chain_name") or "")
            s_aliases = [_normalize_store_key(a) for a in (s.get("aliases") or []) if a]

            strong_id_match = False
            if org and s_org and org == s_org:
                score += 10
                strong_id_match = True
            if phone and s_phone and phone == s_phone:
                score += 8
                strong_id_match = True
            if store_id and s_store_id and store_id == s_store_id:
                score += 8
                strong_id_match = True

            chain_match = False
            if chain_key and (chain_key == s_chain or chain_key in s_aliases):
                chain_match = True
                score += 2

            # If the receipt has a chain name and it doesn't match, do not merge stores unless we have a strong ID match.
            if chain_key and not chain_match and not strong_id_match:
                continue

            # Address/postal/city matching is only safe when the chain matches OR we have a strong ID match.
            allow_location_match = chain_match or strong_id_match
            if postal and s_postal and postal == s_postal:
                score += 4 if allow_location_match else 0
            if city and s_city and city.casefold() == s_city.casefold():
                score += 3 if allow_location_match else 0
            if address and s_addr:
                a_tokens = {t for t in re.split(r"[^\w]+", address.casefold()) if len(t) >= 3}
                s_tokens = {t for t in re.split(r"[^\w]+", s_addr.casefold()) if len(t) >= 3}
                if a_tokens and s_tokens:
                    overlap = len(a_tokens & s_tokens) / max(1, len(a_tokens))
                    if overlap >= 0.6:
                        score += 6 if allow_location_match else 0
                    elif overlap >= 0.4:
                        score += 4 if allow_location_match else 0

            if score > best_score:
                best_score = score
                best_id = str(sid)
                # Only learn aliases when the chain matched (formatting variants of same chain).
                best_allow_alias = chain_match

        if best_score >= 10:
            return best_id, best_allow_alias
        if best_score >= 8 and (org or phone or store_id):
            return best_id, best_allow_alias
        return None, False

    async def async_bulk_update_shopping_product_state(
        self, updates: dict[str, dict[str, Any]]
    ) -> None:
        """Update per-product shopping automation state."""
        state = self._data.setdefault("shopping", {}).setdefault("product_state", {})
        for product_id, product_updates in updates.items():
            row = dict(state.get(product_id, {}))
            row.update(product_updates)
            state[product_id] = row
        await self.async_save()

    def get_shopping_product_state(self, product_id: str) -> dict[str, Any]:
        """Return stored shopping automation state for a product."""
        return dict(
            self._data.get("shopping", {})
            .get("product_state", {})
            .get(product_id, {})
        )

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
    value = (value or "").casefold()
    out: list[str] = []
    last_space = False
    for ch in value:
        # Keep unicode letters + numbers. Convert everything else to spaces.
        if ch.isalnum():
            out.append(ch)
            last_space = False
            continue
        if not last_space:
            out.append(" ")
            last_space = True
    return "".join(out).strip()


def _canonical_name(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw

    alcohol = _canonical_alcohol_category(raw)
    if alcohol:
        return alcohol

    # Best-effort genericization: strip size/unit tokens and digits.
    tokens = [t for t in re.split(r"\s+", raw) if t]
    cleaned: list[str] = []
    unit_tokens = {"kg", "g", "l", "ml", "pcs", "pc", "st", "x"}
    for t in tokens:
        t_norm = _normalize_name(t)
        if not t_norm:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        if t_norm in unit_tokens:
            continue
        cleaned.append(t)

    canonical = " ".join(cleaned) if cleaned else raw
    # Normalize unicode presentation without stripping accents (language-agnostic).
    canonical = unicodedata.normalize("NFKC", canonical).strip()
    return canonical[:128]


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


def _normalize_store_key(value: str) -> str:
    value = (value or "").casefold().strip()
    if not value:
        return ""
    # Keep letters/numbers/spaces only; collapse whitespace.
    out: list[str] = []
    last_space = False
    for ch in value:
        if ch.isalnum():
            out.append(ch)
            last_space = False
            continue
        if not last_space:
            out.append(" ")
            last_space = True
    return "".join(out).strip()


def _canonical_alcohol_category(raw: str) -> str | None:
    """Return a standard canonical alcohol category or None.

    This intentionally prefers coarse categories that work across locales.
    """
    if not raw:
        return None

    lower_raw = raw.casefold()
    normalized = _normalize_name(raw)
    tokens = set(normalized.split())

    # Disqualifiers (avoid misclassifying things like wine vinegar).
    if "vinegar" in tokens or "vinäger" in tokens or "vinager" in tokens:
        return None

    # Non-alcoholic hints: only matter if we also see alcohol-like tokens.
    is_non_alc = False
    if "alkoholfri" in normalized or ("alkohol" in tokens and "fri" in tokens):
        is_non_alc = True
    if "non alcoholic" in normalized or "nonalcoholic" in normalized:
        is_non_alc = True
    if "alcohol free" in normalized or "alcoholfree" in normalized:
        is_non_alc = True
    if re.search(r"\b0[.,]0\b", lower_raw):
        is_non_alc = True

    beer_tokens = {
        "öl",
        "ol",
        "beer",
        "lager",
        "ipa",
        "ale",
        "stout",
        "porter",
        "pils",
        "pilsner",
    }
    wine_tokens = {"vin", "wine", "prosecco", "champagne", "cava", "sparkling"}
    cider_tokens = {"cider"}
    spirit_tokens = {
        "sprit",
        "spirit",
        "spirits",
        "whisky",
        "whiskey",
        "vodka",
        "gin",
        "rom",
        "rum",
        "tequila",
        "brandy",
        "cognac",
        "likör",
        "likor",
        "liqueur",
    }

    is_beer = bool(tokens & beer_tokens)
    is_wine = bool(tokens & wine_tokens)
    is_cider = bool(tokens & cider_tokens)
    is_spirit = bool(tokens & spirit_tokens)

    if not (is_beer or is_wine or is_cider or is_spirit):
        return None

    if is_non_alc:
        return None

    if is_beer:
        return "Beer"
    if is_cider:
        return "Cider"
    if is_wine:
        return "Wine"
    if is_spirit:
        return "Spirits"

    return None
