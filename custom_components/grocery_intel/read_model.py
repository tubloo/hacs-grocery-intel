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
DEFAULT_RECENT_RECEIPT_LIMIT = 10
MAX_RECENT_RECEIPT_LIMIT = 100

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
            "parameter_contract": {
                "filters": {
                    "shape": [
                        {"field": "field_name", "op": "gte", "value": "2026-01-01"}
                    ],
                    "notes": [
                        "Filters are ANDed together.",
                        "A shorthand object is also accepted for equality filters, for example {'store_name': 'ICA'}.",
                        "Nested/list fields can be addressed with dot paths, for example 'receipt_subcategories.subcategory'.",
                    ],
                    "date_values": {
                        "preferred": "Use ISO 8601 date or datetime strings.",
                        "date": "YYYY-MM-DD, for example 2026-01-31.",
                        "datetime": "YYYY-MM-DDTHH:MM:SS or timezone-aware ISO datetime.",
                        "range_recommendation": "Use half-open ranges with gte start and lt next period for precise day/month/year filtering.",
                        "examples": [
                            [
                                {"field": "purchased_at", "op": "gte", "value": "2026-01-01"},
                                {"field": "purchased_at", "op": "lt", "value": "2026-02-01"},
                            ],
                            [
                                {"field": "observed_at", "op": "gte", "value": "2026-07-01T00:00:00+02:00"},
                                {"field": "observed_at", "op": "lt", "value": "2026-08-01T00:00:00+02:00"},
                            ],
                        ],
                    },
                },
                "sort": {
                    "shape": [{"field": "field_name", "direction": "desc"}],
                    "directions": ["asc", "desc"],
                },
                "fields": {
                    "shape": ["field_name"],
                    "description": "Optional list of fields to return from each row.",
                },
                "include": {
                    "shape": ["receipt", "store", "product", "line_item"],
                    "description": "Optional related records to include when relationships are available.",
                },
            },
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
            "parameter_contract": {
                "group_by": {
                    "shape": ["field_name"],
                    "example": ["store_name"],
                },
                "metrics": {
                    "shape": [{"op": "sum", "field": "total", "name": "total_spend"}],
                    "notes": [
                        "count does not require a field.",
                        "sum, avg, min, and max should use numeric fields such as total, line_total, pack_price, or unit_price.",
                    ],
                },
                "time_bucket": {
                    "shape": {"field": "purchased_at", "bucket": "month", "name": "month"},
                    "buckets": ["day", "week", "month", "year"],
                    "output_formats": {
                        "day": "YYYY-MM-DD",
                        "week": "YYYY-Www ISO week, for example 2026-W03",
                        "month": "YYYY-MM",
                        "year": "YYYY",
                    },
                },
                "examples": {
                    "monthly_spend": {
                        "dataset": "receipts",
                        "filters": [
                            {"field": "purchased_at", "op": "gte", "value": "2026-01-01"},
                            {"field": "purchased_at", "op": "lt", "value": "2027-01-01"},
                        ],
                        "metrics": [{"op": "sum", "field": "total", "name": "total_spend"}],
                        "time_bucket": {
                            "field": "purchased_at",
                            "bucket": "month",
                            "name": "month",
                        },
                        "sort": [{"field": "month", "direction": "asc"}],
                    },
                    "spend_by_store": {
                        "dataset": "receipts",
                        "group_by": ["store_name"],
                        "metrics": [
                            {"op": "sum", "field": "total", "name": "total_spend"},
                            {"op": "count", "name": "receipt_count"},
                        ],
                        "sort": [{"field": "total_spend", "direction": "desc"}],
                    },
                },
            },
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


def calculate_grocery_spend_summary(
    read_model: dict[str, Any],
    *,
    start_date: Any = None,
    end_date: Any = None,
    date_field: str = "purchased_at",
    store_name: Any = None,
    category: Any = None,
    subcategory: Any = None,
) -> dict[str, Any]:
    """Return a scalar spend summary for common agent queries."""
    date_field = _normalize_date_field(date_field, default="purchased_at")
    rows = [dict(row) for row in (read_model.get("datasets") or {}).get("receipts") or []]
    filtered = [
        row
        for row in rows
        if _receipt_matches(
            row,
            start_date=start_date,
            end_date=end_date,
            date_field=date_field,
            store_name=store_name,
            category=category,
            subcategory=subcategory,
        )
    ]
    spend_values = [_safe_float(row.get("total")) for row in filtered]
    spend_numbers = [value for value in spend_values if value is not None]
    currencies = sorted({str(row.get("currency")) for row in filtered if row.get("currency")})
    return {
        "schema_version": read_model.get("schema_version", PUBLIC_SCHEMA_VERSION),
        "scope": read_model.get("scope", ANALYTICS_SCOPE),
        "dataset": "receipts",
        "date_range": {"start": start_date or None, "end": end_date or None, "end_exclusive": True},
        "date_field": date_field,
        "filters": _scalar_filter_summary(
            store_name=store_name,
            category=category,
            subcategory=subcategory,
        ),
        "source_rows": len(rows),
        "matched_rows": len(filtered),
        "receipt_count": len(filtered),
        "receipts_with_total": len(spend_numbers),
        "missing_total_count": len(filtered) - len(spend_numbers),
        "total_spend": round(sum(spend_numbers), 2),
        "currency_values": currencies,
        "date_coverage": _date_coverage(rows, date_field),
    }


def list_recent_grocery_receipts(
    read_model: dict[str, Any],
    *,
    limit: Any = DEFAULT_RECENT_RECEIPT_LIMIT,
    start_date: Any = None,
    end_date: Any = None,
    before_date: Any = None,
    date_field: str = "purchased_at",
    store_name: Any = None,
    category: Any = None,
    include_missing_dates: bool = False,
) -> dict[str, Any]:
    """Return recent receipt rows using scalar arguments."""
    date_field = _normalize_date_field(date_field, default="purchased_at")
    rows = [dict(row) for row in (read_model.get("datasets") or {}).get("receipts") or []]
    filtered = [
        row
        for row in rows
        if _receipt_matches(
            row,
            start_date=start_date,
            end_date=end_date,
            before_date=before_date,
            date_field=date_field,
            store_name=store_name,
            category=category,
            include_missing_dates=include_missing_dates,
        )
    ]
    sorted_rows = _sort_rows(filtered, [{"field": date_field, "direction": "desc"}])
    limit_i = max(1, min(_coerce_int(limit, DEFAULT_RECENT_RECEIPT_LIMIT), MAX_RECENT_RECEIPT_LIMIT))
    fields = ["id", "total", "purchased_at", "created_at", "store_name", "receipt_category", "currency", "filename"]
    selected = [_select_fields(row, fields) for row in sorted_rows[:limit_i]]
    return {
        "schema_version": read_model.get("schema_version", PUBLIC_SCHEMA_VERSION),
        "scope": read_model.get("scope", ANALYTICS_SCOPE),
        "dataset": "receipts",
        "date_range": {"start": start_date or None, "end": end_date or None, "end_exclusive": True},
        "before_date": before_date or None,
        "date_field": date_field,
        "filters": _scalar_filter_summary(store_name=store_name, category=category),
        "source_rows": len(rows),
        "matched_rows": len(filtered),
        "returned_rows": len(selected),
        "limit": limit_i,
        "rows": selected,
        "date_coverage": _date_coverage(rows, date_field),
    }


def get_grocery_spend_breakdown(
    read_model: dict[str, Any],
    *,
    start_date: Any = None,
    end_date: Any = None,
    date_field: str = "purchased_at",
    group_by: str = "store",
    store_name: Any = None,
    category: Any = None,
    limit: Any = DEFAULT_QUERY_LIMIT,
) -> dict[str, Any]:
    """Return spend grouped by one common scalar dimension."""
    date_field = _normalize_date_field(date_field, default="purchased_at")
    group_key = _normalize_group_by(group_by)
    rows = [dict(row) for row in (read_model.get("datasets") or {}).get("receipts") or []]
    filtered = [
        row
        for row in rows
        if _receipt_matches(
            row,
            start_date=start_date,
            end_date=end_date,
            date_field=date_field,
            store_name=store_name,
            category=category,
        )
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        for value in _breakdown_values(row, group_key, date_field):
            grouped[value].append(row)

    out: list[dict[str, Any]] = []
    for value, group_rows in grouped.items():
        numbers = [_safe_float(row.get("total")) for row in group_rows]
        spend_numbers = [number for number in numbers if number is not None]
        out.append(
            {
                "group": value,
                "total_spend": round(sum(spend_numbers), 2),
                "receipt_count": len(group_rows),
                "receipts_with_total": len(spend_numbers),
            }
        )
    sorted_rows = _sort_rows(out, [{"field": "total_spend", "direction": "desc"}])
    limit_i = max(1, min(_coerce_int(limit, DEFAULT_QUERY_LIMIT), MAX_AGGREGATE_GROUPS))
    return {
        "schema_version": read_model.get("schema_version", PUBLIC_SCHEMA_VERSION),
        "scope": read_model.get("scope", ANALYTICS_SCOPE),
        "dataset": "receipts",
        "date_range": {"start": start_date or None, "end": end_date or None, "end_exclusive": True},
        "date_field": date_field,
        "group_by": group_key,
        "source_rows": len(rows),
        "matched_rows": len(filtered),
        "group_count": len(out),
        "returned_groups": min(len(sorted_rows), limit_i),
        "groups": sorted_rows[:limit_i],
        "date_coverage": _date_coverage(rows, date_field),
    }


def find_product_price_history(
    read_model: dict[str, Any],
    *,
    product_query: str,
    start_date: Any = None,
    end_date: Any = None,
    store_name: Any = None,
    limit: Any = DEFAULT_RECENT_RECEIPT_LIMIT,
) -> dict[str, Any]:
    """Return price observations matching a product query."""
    query = str(product_query or "").strip()
    datasets = read_model.get("datasets") or {}
    observations = [dict(row) for row in datasets.get("observations") or []]
    products = {row.get("product_id"): row for row in datasets.get("products") or []}
    matched = [
        row
        for row in observations
        if _observation_matches_product(row, products, query)
        and _date_in_range(row.get("observed_at"), start_date=start_date, end_date=end_date)
        and _string_contains(row.get("store_name"), store_name)
    ]
    sorted_rows = _sort_rows(matched, [{"field": "observed_at", "direction": "desc"}])
    limit_i = max(1, min(_coerce_int(limit, DEFAULT_RECENT_RECEIPT_LIMIT), MAX_RECENT_RECEIPT_LIMIT))
    prices = [_safe_float(row.get("unit_price")) for row in matched]
    unit_prices = [value for value in prices if value is not None]
    pack_prices = [_safe_float(row.get("pack_price")) for row in matched]
    pack_price_numbers = [value for value in pack_prices if value is not None]
    rows = []
    for row in sorted_rows[:limit_i]:
        product = products.get(row.get("product_id")) or {}
        rows.append(
            {
                "observation_id": row.get("observation_id"),
                "observed_at": row.get("observed_at"),
                "product_id": row.get("product_id"),
                "product_name": product.get("canonical_name"),
                "store_name": row.get("store_name"),
                "pack_price": row.get("pack_price"),
                "unit_price": row.get("unit_price"),
                "unit_type": row.get("unit_type"),
                "confidence": row.get("confidence"),
                "receipt_id": row.get("receipt_id"),
            }
        )
    latest = rows[0] if rows else None
    oldest = rows[-1] if rows else None
    return {
        "schema_version": read_model.get("schema_version", PUBLIC_SCHEMA_VERSION),
        "scope": read_model.get("scope", ANALYTICS_SCOPE),
        "dataset": "observations",
        "product_query": query,
        "date_range": {"start": start_date or None, "end": end_date or None, "end_exclusive": True},
        "store_name": store_name or None,
        "source_rows": len(observations),
        "matched_rows": len(matched),
        "returned_rows": len(rows),
        "latest": latest,
        "oldest_in_returned_rows": oldest,
        "unit_price_summary": _number_summary(unit_prices),
        "pack_price_summary": _number_summary(pack_price_numbers),
        "rows": rows,
    }


def inspect_grocery_data_quality(
    read_model: dict[str, Any],
    *,
    dataset: str = "receipts",
    issue_type: str = "missing_dates",
    limit: Any = DEFAULT_RECENT_RECEIPT_LIMIT,
) -> dict[str, Any]:
    """Return common read-model data quality issues."""
    datasets = read_model.get("datasets") or {}
    dataset = str(dataset or "receipts").strip()
    rows = [dict(row) for row in datasets.get(dataset) or []]
    issue = str(issue_type or "missing_dates").strip().lower()
    limit_i = max(1, min(_coerce_int(limit, DEFAULT_RECENT_RECEIPT_LIMIT), MAX_RECENT_RECEIPT_LIMIT))

    if dataset == "receipts" and issue == "missing_dates":
        matched = [row for row in rows if not row.get("purchased_at")]
        fields = ["id", "total", "purchased_at", "created_at", "store_name", "filename"]
    elif dataset == "receipts" and issue == "missing_totals":
        matched = [row for row in rows if _safe_float(row.get("total")) is None]
        fields = ["id", "total", "purchased_at", "created_at", "store_name", "filename"]
    elif dataset == "receipts" and issue == "failed_extraction":
        matched = [row for row in rows if str(row.get("extract_status") or "").casefold() == "failed"]
        fields = ["id", "extract_status", "purchased_at", "created_at", "store_name", "filename"]
    elif dataset == "receipts" and issue == "uncategorized":
        matched = [row for row in rows if not row.get("receipt_category")]
        fields = ["id", "total", "purchased_at", "created_at", "store_name", "receipt_category", "filename"]
    elif dataset == "line_items" and issue == "low_confidence":
        matched = [row for row in rows if (_safe_float(row.get("match_confidence")) or 0) < 75]
        fields = ["line_item_id", "receipt_id", "raw_name", "line_total", "matched_product_id", "match_confidence"]
    elif dataset == "observations" and issue == "low_confidence":
        matched = [row for row in rows if (_safe_float(row.get("confidence")) or 0) < 75]
        fields = ["observation_id", "product_id", "store_name", "observed_at", "pack_price", "unit_price", "confidence"]
    else:
        matched = []
        fields = []

    selected = [_select_fields(row, fields) for row in matched[:limit_i]]
    return {
        "schema_version": read_model.get("schema_version", PUBLIC_SCHEMA_VERSION),
        "scope": read_model.get("scope", ANALYTICS_SCOPE),
        "dataset": dataset,
        "issue_type": issue,
        "supported_issue_types": {
            "receipts": ["missing_dates", "missing_totals", "failed_extraction", "uncategorized"],
            "line_items": ["low_confidence"],
            "observations": ["low_confidence"],
        },
        "source_rows": len(rows),
        "matched_rows": len(matched),
        "returned_rows": len(selected),
        "limit": limit_i,
        "rows": selected,
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
        if isinstance(left_dt, datetime) and not isinstance(right_dt, datetime):
            left_dt = left_dt.date()
        elif isinstance(right_dt, datetime) and not isinstance(left_dt, datetime):
            right_dt = right_dt.date()
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


def _receipt_matches(
    row: dict[str, Any],
    *,
    start_date: Any = None,
    end_date: Any = None,
    before_date: Any = None,
    date_field: str = "purchased_at",
    store_name: Any = None,
    category: Any = None,
    subcategory: Any = None,
    include_missing_dates: bool = False,
) -> bool:
    if not include_missing_dates and not row.get(date_field):
        return False
    if not _date_in_range(
        row.get(date_field),
        start_date=start_date,
        end_date=end_date,
        before_date=before_date,
        allow_missing=include_missing_dates,
    ):
        return False
    if not _string_contains(row.get("store_name"), store_name):
        return False
    if category and not _compare_value(row.get("receipt_category"), "eq", category):
        return False
    if subcategory and not _compare_any(
        _field_values(row, "receipt_subcategories.subcategory"), "eq", subcategory
    ):
        return False
    return True


def _date_in_range(
    value: Any,
    *,
    start_date: Any = None,
    end_date: Any = None,
    before_date: Any = None,
    allow_missing: bool = False,
) -> bool:
    if not value:
        return allow_missing and not start_date and not end_date and not before_date
    if start_date and not _compare_value(value, "gte", start_date):
        return False
    if end_date and not _compare_value(value, "lt", end_date):
        return False
    if before_date and not _compare_value(value, "lt", before_date):
        return False
    return True


def _string_contains(value: Any, expected: Any) -> bool:
    if expected is None or str(expected).strip() == "":
        return True
    if value is None:
        return False
    return str(expected).casefold() in str(value).casefold()


def _normalize_date_field(value: Any, *, default: str) -> str:
    field = str(value or default).strip()
    return field if field in {"purchased_at", "created_at", "observed_at", "updated_at", "taken_at"} else default


def _normalize_group_by(value: Any) -> str:
    group_by = str(value or "store").strip().lower()
    return group_by if group_by in {"store", "category", "subcategory", "week", "month", "year"} else "store"


def _breakdown_values(row: dict[str, Any], group_by: str, date_field: str) -> list[str]:
    if group_by == "store":
        return [str(row.get("store_name") or "Unknown")]
    if group_by == "category":
        return [str(row.get("receipt_category") or "uncategorized")]
    if group_by == "subcategory":
        values = _field_values(row, "receipt_subcategories.subcategory")
        return [str(value) for value in values if value] or ["uncategorized"]
    if group_by in {"week", "month", "year"}:
        return [_bucket_value(row.get(date_field), group_by) or "unknown"]
    return ["unknown"]


def _scalar_filter_summary(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None and str(value).strip()}


def _date_coverage(rows: list[dict[str, Any]], date_field: str) -> dict[str, Any]:
    populated = [row.get(date_field) for row in rows if row.get(date_field)]
    return {
        "field": date_field,
        "rows": len(rows),
        "populated": len(populated),
        "missing": len(rows) - len(populated),
    }


def _observation_matches_product(
    row: dict[str, Any], products: dict[Any, dict[str, Any]], query: str
) -> bool:
    if not query:
        return False
    product = products.get(row.get("product_id")) or {}
    values = [
        row.get("product_id"),
        product.get("canonical_name"),
        *list(product.get("aliases") or []),
    ]
    query_cf = query.casefold()
    return any(value is not None and query_cf in str(value).casefold() for value in values)


def _number_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(mean(values), 4),
    }


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
