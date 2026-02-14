"""Export analyzed Grocery Intel data."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util


@dataclass
class ExportFilters:
    since: str | None = None
    until: str | None = None
    include_undated: bool = False


def _parse_dt(value: str | None) -> Any:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    dt = dt_util.parse_datetime(raw)
    if dt is not None:
        return dt_util.as_local(dt)
    d = dt_util.parse_date(raw)
    if d is not None:
        return dt_util.start_of_local_day(d)
    return None


def _receipt_in_range(receipt: dict[str, Any], *, since_dt, until_dt, include_undated: bool) -> bool:
    if since_dt is None and until_dt is None:
        return True
    dt = _parse_dt(receipt.get("purchased_at"))
    if dt is None:
        return include_undated
    if since_dt is not None and dt < since_dt:
        return False
    if until_dt is not None and dt > until_dt:
        return False
    return True


def _strip_keys(obj: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in keys}


def build_export_payload(
    hass: HomeAssistant,
    *,
    scope: str,
    filters: ExportFilters,
    receipts: list[dict[str, Any]],
    line_items: list[dict[str, Any]],
    products: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    stores: list[dict[str, Any]],
    inventory_images: list[dict[str, Any]],
    activities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a versioned export payload.

    Scopes:
      - analytics: receipts + stores + products + observations (+ inventory images metadata)
      - debug: analytics + extraction metadata and source_meta
      - full: debug + activities
    """
    scope = (scope or "analytics").strip().lower()
    if scope not in {"analytics", "debug", "full"}:
        scope = "analytics"

    since_dt = _parse_dt(filters.since)
    until_dt = _parse_dt(filters.until)
    receipts_out = [
        r
        for r in receipts
        if _receipt_in_range(r, since_dt=since_dt, until_dt=until_dt, include_undated=filters.include_undated)
    ]
    receipt_ids = {str(r.get("id") or r.get("receipt_id") or r.get("receiptId") or "") for r in receipts_out}
    receipt_ids.discard("")

    # Filter observations/line_items down to exported receipts when possible.
    observations_out = [
        o for o in observations if not receipt_ids or str(o.get("receipt_id") or "") in receipt_ids
    ]
    line_items_out = [
        li for li in line_items if not receipt_ids or str(li.get("receipt_id") or "") in receipt_ids
    ]

    # Limit stores/products to what’s referenced when possible.
    store_ids = {str(r.get("store_entity_id") or "") for r in receipts_out if r.get("store_entity_id")}
    stores_out = [s for s in stores if not store_ids or str(s.get("store_entity_id") or "") in store_ids]

    product_ids = {
        str(o.get("product_id") or "")
        for o in observations_out
        if o.get("product_id")
    }
    products_out = [p for p in products if not product_ids or str(p.get("product_id") or "") in product_ids]

    # Inventory images are less structured; export metadata only by default.
    inventory_out: list[dict[str, Any]] = []
    for img in inventory_images:
        row = dict(img)
        # Never export local file paths by default.
        row.pop("file_path", None)
        inventory_out.append(row)

    # Strip sensitive fields by scope.
    receipt_strip_common = {
        "file_path",
    }
    receipt_strip_analytics = receipt_strip_common | {
        "ocr_text",
        "raw_text",
        "source_meta",
        "extract_started_at",
        "extract_queued_at",
        "extract_finished_at",
        "extract_duration_ms",
        "extract_queue_delay_ms",
        "extract_method",
        "extract_provider",
        "extract_model",
        "merchant_hints",
    }
    receipt_strip_debug = receipt_strip_common | {
        "ocr_text",
        "raw_text",
    }

    if scope == "analytics":
        receipts_out = [_strip_keys(dict(r), receipt_strip_analytics) for r in receipts_out]
    elif scope in {"debug", "full"}:
        receipts_out = [_strip_keys(dict(r), receipt_strip_debug) for r in receipts_out]

    exported_at = dt_util.now().isoformat()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "exported_at": exported_at,
        "scope": scope,
        "filters": {
            "since": filters.since,
            "until": filters.until,
            "include_undated": filters.include_undated,
        },
        "counts": {
            "receipts": len(receipts_out),
            "stores": len(stores_out),
            "products": len(products_out),
            "observations": len(observations_out),
            "line_items": len(line_items_out),
            "inventory_images": len(inventory_out),
            "activities": len(activities or []) if scope == "full" else 0,
        },
        "data": {
            "receipts": receipts_out,
            "stores": stores_out,
            "products": products_out,
            "observations": observations_out,
            "line_items": line_items_out,
            "inventory_images": inventory_out,
        },
    }
    if scope == "full":
        payload["data"]["activities"] = list(activities or [])
    return payload


def write_export_file(
    hass: HomeAssistant,
    *,
    exports_dir: str,
    filename: str,
    payload: dict[str, Any],
) -> str:
    """Write export payload as JSON to exports_dir and return the written path."""
    exports_dir = (exports_dir or "").strip() or "/media/grocery_intel/exports"
    os.makedirs(exports_dir, exist_ok=True)
    path = os.path.join(exports_dir, filename)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path

