"""Public read model for Grocery Intel consumers."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from statistics import mean
from typing import Any

from homeassistant.util import dt as dt_util

PUBLIC_SCHEMA_VERSION = 1

DEFAULT_QUERY_LIMIT = 100
MAX_QUERY_LIMIT = 500
MAX_AGGREGATE_GROUPS = 500

ANALYTICS_SCOPE = "analytics"
DEBUG_SCOPE = "debug"
FULL_SCOPE = "full"
SCOPES = {ANALYTICS_SCOPE, DEBUG_SCOPE, FULL_SCOPE}


DATASET_SCHEMAS: dict[str, dict[str, Any]] = {
    "receipts": {
        "description": "One row per receipt or dining bill.",
        "key_field": "id",
        "default_sort": [{"field": "purchased_at", "direction": "desc"}],
        "fields": {
            "id": "Stable receipt id.",
            "total": "Receipt total amount.",
            "purchased_at": "Purchase datetime in ISO format.",
            "store_name": "Extracted merchant/store name.",
            "store_entity_id": "Canonical store id when matched.",
            "receipt_category": "Top-level category such as grocery or dining.",
            "receipt_category_source": "Category source such as auto or manual.",
            "receipt_subcategories": "Array of {subcategory,total} rows.",
            "currency": "Receipt currency symbol or code.",
            "extract_status": "Extraction lifecycle status.",
            "created_at": "Import/create datetime.",
            "filename": "Archived source filename.",
            "source_type": "Receipt source such as manual, inbox, or telegram.",
        },
        "relationships": {
            "store": "stores.store_entity_id via store_entity_id",
            "line_items": "line_items.receipt_id via id",
            "observations": "observations.receipt_id via id",
        },
    },
    "line_items": {
        "description": "Raw extracted receipt line items matched to canonical products.",
        "key_field": "line_item_id",
        "fields": {
            "line_item_id": "Stable line item id.",
            "receipt_id": "Parent receipt id.",
            "raw_name": "Line item name as extracted from the receipt.",
            "line_total": "Line item total price.",
            "qty_raw": "Raw quantity text when available.",
            "unit_price_raw": "Raw unit price when extracted.",
            "matched_product_id": "Matched canonical product id.",
            "match_confidence": "Product match confidence from 0 to 100.",
        },
        "relationships": {
            "receipt": "receipts.id via receipt_id",
            "product": "products.product_id via matched_product_id",
            "observation": "observations.line_item_id via line_item_id",
        },
    },
    "products": {
        "description": "Canonical product identities derived from receipt items and inventory images.",
        "key_field": "product_id",
        "fields": {
            "product_id": "Stable product id.",
            "canonical_name": "Canonical display name.",
            "aliases": "Known raw names matched to this product.",
            "unit_type": "Canonical unit type when known.",
            "created_at": "First creation datetime.",
            "updated_at": "Last update datetime.",
            "inventory_last_seen_at": "Latest inventory-image evidence datetime when available.",
            "inventory_last_seen_confidence": "Latest inventory-image evidence confidence.",
        },
        "relationships": {
            "observations": "observations.product_id via product_id",
            "line_items": "line_items.matched_product_id via product_id",
        },
    },
    "observations": {
        "description": "Product purchase observations used for price and cadence analytics.",
        "key_field": "observation_id",
        "default_sort": [{"field": "observed_at", "direction": "desc"}],
        "fields": {
            "observation_id": "Stable observation id.",
            "product_id": "Canonical product id.",
            "store_name": "Store name at purchase time.",
            "observed_at": "Purchase datetime in ISO format.",
            "pack_price": "Observed pack or line price.",
            "unit_price": "Derived unit price.",
            "unit_type": "Derived unit type.",
            "confidence": "Product match confidence from 0 to 100.",
            "receipt_id": "Parent receipt id.",
            "line_item_id": "Source line item id.",
        },
        "relationships": {
            "product": "products.product_id via product_id",
            "receipt": "receipts.id via receipt_id",
            "line_item": "line_items.line_item_id via line_item_id",
            "store": "stores.store_entity_id via receipts.store_entity_id",
        },
    },
    "stores": {
        "description": "Canonical store identities used to group receipts.",
        "key_field": "store_entity_id",
        "fields": {
            "store_entity_id": "Stable store id.",
            "chain_name": "Canonical chain or merchant name.",
            "branch_name": "Branch name when known.",
            "address": "Store address when extracted.",
            "postal_code": "Store postal code when extracted.",
            "city": "Store city when extracted.",
            "aliases": "Known store aliases.",
            "created_at": "First creation datetime.",
            "updated_at": "Last update datetime.",
        },
        "relationships": {
            "receipts": "receipts.store_entity_id via store_entity_id",
        },
    },
    "inventory_images": {
        "description": "Inventory image metadata and detected products.",
        "key_field": "image_id",
        "default_sort": [{"field": "taken_at", "direction": "desc"}],
        "fields": {
            "image_id": "Stable inventory image id.",
            "filename": "Archived source filename.",
            "taken_at": "EXIF capture datetime when available.",
            "source_type": "Image source such as inbox or telegram.",
            "status": "Analysis lifecycle status.",
            "attempts": "Analysis attempts.",
            "created_at": "Import/create datetime.",
            "updated_at": "Last update datetime.",
            "detected_products": "Detected inventory product evidence.",
        },
    },
    "activities": {
        "description": "Activity log rows. Available only with full scope.",
        "key_field": "activity_id",
        "default_sort": [{"field": "created_at", "direction": "desc"}],
        "fields": {
            "activity_id": "Stable activity id.",
            "kind": "Activity kind.",
            "description": "Human-readable activity description.",
            "created_at": "Activity datetime.",
            "payload": "Activity payload.",
            "undo": "Undo metadata when supported.",
        },
    },
}


SENSITIVE_KEYS_BY_SCOPE: dict[str, dict[str, set[str]]] = {
    ANALYTICS_SCOPE: {
        "receipts": {
            "file_path",
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
            "content_hash",
        },
        "inventory_images": {"file_path", "raw_result", "source_meta", "fingerprint"},
    },
    DEBUG_SCOPE: {
        "receipts": {"file_path", "ocr_text", "raw_text", "content_hash"},
        "inventory_images": {"file_path", "raw_result", "fingerprint"},
    },
    FULL_SCOPE: {
        "receipts": {"file_path"},
        "inventory_images": {"file_path"},
    },
}


def normalize_scope(scope: Any) -> str:
    """Normalize a requested public read-model scope."""
    value = str(scope or ANALYTICS_SCOPE).strip().lower()
    return value if value in SCOPES else ANALYTICS_SCOPE


async def async_build_public_read_model(data: Any, *, scope: str = ANALYTICS_SCOPE) -> dict[str, Any]:
    """Build the versioned public read model from integration storage."""
    scope = normalize_scope(scope)
    storage = data.storage
    receipts = await storage.async_list_receipts()
    line_items = await storage.async_list_line_items()
    products = await storage.async_list_products()
    observations = await storage.async_list_observations()
    stores = await storage.async_list_stores()
    inventory_images = await storage.async_list_inventory_images()

    datasets: dict[str, list[dict[str, Any]]] = {
        "receipts": _sanitize_rows("receipts", receipts, scope),
        "line_items": _sanitize_rows("line_items", line_items, scope),
        "products": _sanitize_rows("products", products, scope),
        "observations": _sanitize_rows("observations", observations, scope),
        "stores": _sanitize_rows("stores", stores, scope),
        "inventory_images": _sanitize_rows("inventory_images", inventory_images, scope),
    }
    if scope == FULL_SCOPE:
        datasets["activities"] = await data.activity.async_list_activities()

    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "scope": scope,
        "datasets": datasets,
        "counts": {name: len(rows) for name, rows in datasets.items()},
    }


def describe_public_schema(*, scope: str = ANALYTICS_SCOPE) -> dict[str, Any]:
    """Return the public schema descriptor for agents and external consumers."""
    scope = normalize_scope(scope)
    datasets = {
        name: schema
        for name, schema in DATASET_SCHEMAS.items()
        if name != "activities" or scope == FULL_SCOPE
    }
    return {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "scope": scope,
        "default_scope": ANALYTICS_SCOPE,
        "available_scopes": sorted(SCOPES),
        "datasets": datasets,
        "query": {
            "filter_ops": [
                "eq",
                "ne",
                "lt",
                "lte",
                "gt",
                "gte",
                "contains",
                "icontains",
                "in",
                "not_in",
                "exists",
            ],
            "sort_shape": [{"field": "purchased_at", "direction": "desc"}],
            "pagination": {
                "default_limit": DEFAULT_QUERY_LIMIT,
                "max_limit": MAX_QUERY_LIMIT,
            },
        },
        "aggregate": {
            "metrics": ["count", "sum", "avg", "min", "max"],
            "time_buckets": ["day", "week", "month", "year"],
            "max_groups": MAX_AGGREGATE_GROUPS,
        },
    }


def query_public_read_model(
    read_model: dict[str, Any],
    *,
    dataset: str,
    filters: Any = None,
    sort: Any = None,
    limit: Any = DEFAULT_QUERY_LIMIT,
    offset: Any = 0,
    fields: Any = None,
    include: Any = None,
) -> dict[str, Any]:
    """Query rows from a public read-model dataset."""
    dataset = str(dataset or "").strip()
    datasets = read_model.get("datasets") or {}
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset: {dataset}")

    rows = [dict(row) for row in datasets.get(dataset) or []]
    filtered = [row for row in rows if _matches_filters(row, filters)]
    sorted_rows = _sort_rows(filtered, sort or DATASET_SCHEMAS.get(dataset, {}).get("default_sort"))
    limit_i = _coerce_int(limit, DEFAULT_QUERY_LIMIT)
    limit_i = max(1, min(limit_i, MAX_QUERY_LIMIT))
    offset_i = max(0, _coerce_int(offset, 0))
    page = sorted_rows[offset_i : offset_i + limit_i]
    selected = [_select_fields(row, fields) for row in page]
    included = _build_included(read_model, dataset, page, include)

    return {
        "schema_version": read_model.get("schema_version", PUBLIC_SCHEMA_VERSION),
        "scope": read_model.get("scope", ANALYTICS_SCOPE),
        "dataset": dataset,
        "total_rows": len(rows),
        "matched_rows": len(filtered),
        "returned_rows": len(selected),
        "offset": offset_i,
        "limit": limit_i,
        "rows": selected,
        "included": included,
    }


def aggregate_public_read_model(
    read_model: dict[str, Any],
    *,
    dataset: str,
    filters: Any = None,
    group_by: Any = None,
    metrics: Any = None,
    time_bucket: Any = None,
    sort: Any = None,
    limit: Any = DEFAULT_QUERY_LIMIT,
) -> dict[str, Any]:
    """Aggregate rows from a public read-model dataset."""
    dataset = str(dataset or "").strip()
    datasets = read_model.get("datasets") or {}
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset: {dataset}")

    rows = [dict(row) for row in datasets.get(dataset) or []]
    filtered = [row for row in rows if _matches_filters(row, filters)]
    group_fields = _coerce_string_list(group_by)
    bucket = _normalize_time_bucket(time_bucket)
    if bucket:
        if bucket["name"] not in group_fields:
            group_fields.append(bucket["name"])

    metric_defs = _normalize_metrics(metrics)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        row_for_group = dict(row)
        if bucket:
            row_for_group[bucket["name"]] = _bucket_value(row.get(bucket["field"]), bucket["bucket"])
        key = tuple(row_for_group.get(field) for field in group_fields) if group_fields else ("all",)
        grouped[key].append(row_for_group)

    out: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        row: dict[str, Any] = {}
        if group_fields:
            for idx, field in enumerate(group_fields):
                row[field] = key[idx]
        else:
            row["group"] = "all"
        for metric in metric_defs:
            name = metric["name"]
            op = metric["op"]
            field = metric.get("field")
            row[name] = _compute_metric(group_rows, op=op, field=field)
        out.append(row)

    sorted_rows = _sort_rows(out, sort)
    limit_i = max(1, min(_coerce_int(limit, DEFAULT_QUERY_LIMIT), MAX_AGGREGATE_GROUPS))
    return {
        "schema_version": read_model.get("schema_version", PUBLIC_SCHEMA_VERSION),
        "scope": read_model.get("scope", ANALYTICS_SCOPE),
        "dataset": dataset,
        "source_rows": len(rows),
        "matched_rows": len(filtered),
        "group_count": len(out),
        "returned_groups": min(len(sorted_rows), limit_i),
        "groups": sorted_rows[:limit_i],
    }


def _sanitize_rows(dataset: str, rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    strip_keys = SENSITIVE_KEYS_BY_SCOPE.get(scope, {}).get(dataset, set())
    return [{key: value for key, value in dict(row).items() if key not in strip_keys} for row in rows]


def _matches_filters(row: dict[str, Any], filters: Any) -> bool:
    if not filters:
        return True
    if isinstance(filters, dict):
        for field, expected in filters.items():
            if not _compare_any(_field_values(row, str(field)), "eq", expected):
                return False
        return True
    if isinstance(filters, list):
        for item in filters:
            if not isinstance(item, dict):
                return False
            field = str(item.get("field") or "").strip()
            op = str(item.get("op") or "eq").strip().lower()
            expected = item.get("value")
            if not field or not _compare_any(_field_values(row, field), op, expected):
                return False
        return True
    return False


def _field_values(value: Any, path: str) -> list[Any]:
    parts = [part for part in str(path).split(".") if part]
    values = [value]
    for part in parts:
        next_values: list[Any] = []
        for item in values:
            if isinstance(item, dict) and part in item:
                next_values.append(item.get(part))
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict) and part in child:
                        next_values.append(child.get(part))
                    elif part == "*":
                        next_values.append(child)
        values = next_values
    return values


def _compare_any(values: list[Any], op: str, expected: Any) -> bool:
    if op == "exists":
        exists = any(value is not None and value != "" for value in values)
        return exists if bool(expected) else not exists
    if not values:
        values = [None]
    return any(_compare_value(value, op, expected) for value in values)


def _compare_value(value: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return value == expected or str(value) == str(expected)
    if op == "ne":
        return not _compare_value(value, "eq", expected)
    if op in {"lt", "lte", "gt", "gte"}:
        left, right = _coerce_comparable(value, expected)
        if left is None or right is None:
            return False
        if op == "lt":
            return left < right
        if op == "lte":
            return left <= right
        if op == "gt":
            return left > right
        return left >= right
    if op in {"contains", "icontains"}:
        if value is None:
            return False
        left = str(value)
        right = str(expected)
        if op == "icontains":
            left = left.casefold()
            right = right.casefold()
        return right in left
    if op in {"in", "not_in"}:
        expected_values = expected if isinstance(expected, list) else [expected]
        found = any(_compare_value(value, "eq", item) for item in expected_values)
        return found if op == "in" else not found
    return False


def _coerce_comparable(left: Any, right: Any) -> tuple[Any, Any]:
    left_dt = _parse_datetime_like(left)
    right_dt = _parse_datetime_like(right)
    if left_dt is not None and right_dt is not None:
        return left_dt, right_dt
    try:
        return float(left), float(right)
    except (TypeError, ValueError):
        pass
    if left is None or right is None:
        return None, None
    return str(left), str(right)


def _parse_datetime_like(value: Any) -> datetime | date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    dt = dt_util.parse_datetime(value)
    if dt is not None:
        return dt_util.as_local(dt)
    return dt_util.parse_date(value)


def _sort_rows(rows: list[dict[str, Any]], sort: Any) -> list[dict[str, Any]]:
    sort_defs = _normalize_sort(sort)
    out = list(rows)
    for item in reversed(sort_defs):
        field = item["field"]
        reverse = item["direction"] == "desc"
        out.sort(key=lambda row: _sort_value(row.get(field)), reverse=reverse)
    return out


def _normalize_sort(sort: Any) -> list[dict[str, str]]:
    if not sort:
        return []
    if isinstance(sort, str):
        return [{"field": sort, "direction": "asc"}]
    if isinstance(sort, dict):
        sort = [sort]
    if not isinstance(sort, list):
        return []
    out: list[dict[str, str]] = []
    for item in sort:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        if not field:
            continue
        direction = str(item.get("direction") or "asc").strip().lower()
        out.append({"field": field, "direction": "desc" if direction == "desc" else "asc"})
    return out


def _sort_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return (1, 0, "")
    parsed = _parse_datetime_like(value)
    if parsed is not None:
        return (0, 0, parsed.isoformat())
    try:
        return (0, 1, float(value))
    except (TypeError, ValueError):
        return (0, 2, str(value).casefold())


def _select_fields(row: dict[str, Any], fields: Any) -> dict[str, Any]:
    field_names = _coerce_string_list(fields)
    if not field_names:
        return row
    return {field: row.get(field) for field in field_names if field in row}


def _build_included(
    read_model: dict[str, Any], dataset: str, rows: list[dict[str, Any]], include: Any
) -> dict[str, list[dict[str, Any]]]:
    include_names = set(_coerce_string_list(include))
    if not include_names:
        return {}
    datasets = read_model.get("datasets") or {}
    receipts = {row.get("id"): row for row in datasets.get("receipts") or []}
    products = {row.get("product_id"): row for row in datasets.get("products") or []}
    stores = {row.get("store_entity_id"): row for row in datasets.get("stores") or []}
    line_items = {row.get("line_item_id"): row for row in datasets.get("line_items") or []}

    out: dict[str, dict[Any, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        receipt = None
        if dataset == "receipts":
            receipt = row
        elif row.get("receipt_id"):
            receipt = receipts.get(row.get("receipt_id"))

        if "receipt" in include_names and receipt:
            out["receipts"][receipt.get("id")] = receipt
        if "store" in include_names and receipt and receipt.get("store_entity_id"):
            store = stores.get(receipt.get("store_entity_id"))
            if store:
                out["stores"][store.get("store_entity_id")] = store
        if "product" in include_names:
            product_id = row.get("product_id") or row.get("matched_product_id")
            product = products.get(product_id)
            if product:
                out["products"][product.get("product_id")] = product
        if "line_item" in include_names and row.get("line_item_id"):
            line_item = line_items.get(row.get("line_item_id"))
            if line_item:
                out["line_items"][line_item.get("line_item_id")] = line_item

    return {name: list(rows_by_id.values()) for name, rows_by_id in out.items()}


def _normalize_metrics(metrics: Any) -> list[dict[str, Any]]:
    if not metrics:
        return [{"op": "count", "name": "count"}]
    if isinstance(metrics, dict):
        metrics = [metrics]
    if isinstance(metrics, str):
        metrics = [{"op": metrics}]
    out: list[dict[str, Any]] = []
    for metric in metrics if isinstance(metrics, list) else []:
        if isinstance(metric, str):
            metric = {"op": metric}
        if not isinstance(metric, dict):
            continue
        op = str(metric.get("op") or "count").strip().lower()
        if op not in {"count", "sum", "avg", "min", "max"}:
            continue
        field = str(metric.get("field") or "").strip() or None
        name = str(metric.get("name") or "").strip()
        if not name:
            name = op if not field else f"{op}_{field}"
        out.append({"op": op, "field": field, "name": name})
    return out or [{"op": "count", "name": "count"}]


def _compute_metric(rows: list[dict[str, Any]], *, op: str, field: str | None) -> Any:
    if op == "count":
        return len(rows)
    values = [_safe_float(row.get(field)) for row in rows] if field else []
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    if op == "sum":
        return round(sum(numbers), 2)
    if op == "avg":
        return round(mean(numbers), 4)
    if op == "min":
        return min(numbers)
    if op == "max":
        return max(numbers)
    return None


def _normalize_time_bucket(value: Any) -> dict[str, str] | None:
    if not value:
        return None
    if not isinstance(value, dict):
        return None
    field = str(value.get("field") or "").strip()
    bucket = str(value.get("bucket") or "").strip().lower()
    if not field or bucket not in {"day", "week", "month", "year"}:
        return None
    name = str(value.get("name") or f"{field}_{bucket}").strip()
    return {"field": field, "bucket": bucket, "name": name}


def _bucket_value(value: Any, bucket: str) -> str | None:
    parsed = _parse_datetime_like(value)
    if parsed is None:
        return None
    if isinstance(parsed, datetime):
        parsed_date = parsed.date()
    else:
        parsed_date = parsed
    if bucket == "day":
        return parsed_date.isoformat()
    if bucket == "week":
        iso_year, iso_week, _iso_day = parsed_date.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if bucket == "month":
        return f"{parsed_date.year:04d}-{parsed_date.month:02d}"
    if bucket == "year":
        return f"{parsed_date.year:04d}"
    return None


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
