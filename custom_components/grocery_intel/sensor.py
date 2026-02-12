"""Sensors for Grocery Intel."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from datetime import timedelta
from statistics import median
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .activity import ActivityLog
from .const import (
    DOMAIN,
    CONF_CURRENCY_SYMBOL,
    CONF_OVERPAID_PCT_THRESHOLD,
    CONF_BASELINE_WINDOW_N,
    CONF_TOP_INCREASES_RECENT_DAYS,
    CONF_TOP_INCREASES_PRIOR_DAYS,
    CONF_BEST_STORE_WINDOW_DAYS,
    CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
    DEFAULT_CURRENCY_SYMBOL,
    DEFAULT_OVERPAID_PCT_THRESHOLD,
    DEFAULT_BASELINE_WINDOW_N,
    DEFAULT_TOP_INCREASES_RECENT_DAYS,
    DEFAULT_TOP_INCREASES_PRIOR_DAYS,
    DEFAULT_BEST_STORE_WINDOW_DAYS,
    DEFAULT_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
    SMALL_OVERPAY_FLOOR,
)
from .storage import ReceiptStorage


@dataclass
class GroceryIntelDataSnapshot:
    week_total: float
    month_total: float
    rolling_7d_total: float
    rolling_30d_total: float
    receipt_count_30d: int
    avg_basket_30d: float
    receipt_status_counts: dict[str, int]
    receipt_processing_timing: dict[str, Any]
    top_stores_30d: list[dict[str, Any]]
    recent_receipts: list[dict[str, Any]]
    recent_activities: list[dict[str, Any]]
    inventory_recently_seen: list[dict[str, Any]]
    top_increases: list[dict[str, Any]]
    overpaid_items: list[dict[str, Any]]
    best_store_items: list[dict[str, Any]]


class GrocerySpendCoordinator(DataUpdateCoordinator[GroceryIntelDataSnapshot]):
    """Coordinator to calculate spend totals and analytics."""

    def __init__(
        self,
        hass: HomeAssistant,
        storage: ReceiptStorage,
        activity: ActivityLog,
        entry: ConfigEntry,
    ) -> None:
        super().__init__(hass, logger=logging.getLogger(__name__), name=DOMAIN, update_interval=None)
        self._storage = storage
        self._activity = activity
        self._entry = entry

    async def _async_update_data(self) -> GroceryIntelDataSnapshot:
        receipts = await self._storage.async_list_receipts()
        observations = await self._storage.async_list_observations()
        products = await self._storage.async_list_products()
        activities = await self._activity.async_list_activities()

        week_total, month_total = _compute_spend_totals(receipts)
        rolling_7d_total, rolling_30d_total, receipt_count_30d, avg_basket_30d, top_stores_30d = (
            _compute_rolling_stats(receipts)
        )
        receipt_status_counts = _compute_receipt_status_counts(receipts)
        receipt_processing_timing = _compute_receipt_processing_timing(receipts)
        recent_receipts = _compute_recent_receipts(receipts)
        recent_activities = _compute_recent_activities(activities)

        options = self._entry.options
        overpaid_pct = options.get(CONF_OVERPAID_PCT_THRESHOLD, DEFAULT_OVERPAID_PCT_THRESHOLD)
        baseline_n = options.get(CONF_BASELINE_WINDOW_N, DEFAULT_BASELINE_WINDOW_N)
        recent_days = options.get(CONF_TOP_INCREASES_RECENT_DAYS, DEFAULT_TOP_INCREASES_RECENT_DAYS)
        prior_days = options.get(CONF_TOP_INCREASES_PRIOR_DAYS, DEFAULT_TOP_INCREASES_PRIOR_DAYS)
        best_store_days = options.get(CONF_BEST_STORE_WINDOW_DAYS, DEFAULT_BEST_STORE_WINDOW_DAYS)
        evidence_ttl_days = int(
            options.get(
                CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
                DEFAULT_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
            )
        )

        product_lookup = {p.get("product_id"): p for p in products}
        obs_by_product = _group_observations_by_product(observations)

        baselines = _compute_baselines(obs_by_product, baseline_n)

        overpaid = _compute_overpaid_items(
            obs_by_product,
            baselines,
            product_lookup,
            overpaid_pct,
        )

        increases = _compute_top_increases(
            obs_by_product,
            product_lookup,
            recent_days,
            prior_days,
        )

        best_store = _compute_best_store(
            obs_by_product,
            product_lookup,
            best_store_days,
        )

        inventory_recently_seen = _compute_inventory_recently_seen(
            products,
            self._storage,
            evidence_ttl_days,
        )

        return GroceryIntelDataSnapshot(
            week_total=week_total,
            month_total=month_total,
            rolling_7d_total=rolling_7d_total,
            rolling_30d_total=rolling_30d_total,
            receipt_count_30d=receipt_count_30d,
            avg_basket_30d=avg_basket_30d,
            receipt_status_counts=receipt_status_counts,
            receipt_processing_timing=receipt_processing_timing,
            top_stores_30d=top_stores_30d,
            recent_receipts=recent_receipts,
            recent_activities=recent_activities,
            inventory_recently_seen=inventory_recently_seen,
            top_increases=increases,
            overpaid_items=overpaid,
            best_store_items=best_store,
        )


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities
) -> None:
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator: GrocerySpendCoordinator = data.coordinator

    currency = entry.options.get(CONF_CURRENCY_SYMBOL, DEFAULT_CURRENCY_SYMBOL)

    device_info = DeviceInfo(
        identifiers={(DOMAIN, "grocery_intel")},
        name=entry.title,
        manufacturer="Community",
        model="Grocery Intel",
    )

    entities: list[SensorEntity] = [
        GrocerySpendSensor(
            coordinator,
            f"{DOMAIN}_spend_week",
            "Spend week",
            "spend_week",
            device_info,
            currency,
            "week_total",
        ),
        GrocerySpendSensor(
            coordinator,
            f"{DOMAIN}_spend_month",
            "Spend month",
            "spend_month",
            device_info,
            currency,
            "month_total",
        ),
        GrocerySpendSensor(
            coordinator,
            f"{DOMAIN}_spend_7d",
            "Spend 7d",
            "spend_7d",
            device_info,
            currency,
            "rolling_7d_total",
        ),
        GrocerySpendSensor(
            coordinator,
            f"{DOMAIN}_spend_30d",
            "Spend 30d",
            "spend_30d",
            device_info,
            currency,
            "rolling_30d_total",
        ),
        GrocerySpendSensor(
            coordinator,
            f"{DOMAIN}_avg_basket_30d",
            "Avg basket 30d",
            "avg_basket_30d",
            device_info,
            currency,
            "avg_basket_30d",
        ),
        GroceryCountSensor(
            coordinator,
            f"{DOMAIN}_receipt_count_30d",
            "Receipts 30d",
            "receipt_count_30d",
            device_info,
            "receipt_count_30d",
        ),
        GroceryReceiptProcessingSensor(
            coordinator,
            f"{DOMAIN}_receipt_processing",
            "Receipt processing",
            "receipt_processing",
            device_info,
        ),
        GroceryAnalyticsSensor(
            coordinator,
            f"{DOMAIN}_top_stores_30d",
            "Top stores 30d",
            "top_stores_30d",
            device_info,
            "top_stores_30d",
        ),
        GroceryAnalyticsSensor(
            coordinator,
            f"{DOMAIN}_recent_receipts",
            "Recent receipts",
            "recent_receipts",
            device_info,
            "recent_receipts",
        ),
        GroceryAnalyticsSensor(
            coordinator,
            f"{DOMAIN}_recent_activities",
            "Recent activities",
            "recent_activities",
            device_info,
            "recent_activities",
        ),
        GroceryAnalyticsSensor(
            coordinator,
            f"{DOMAIN}_inventory_recently_seen",
            "Inventory recently seen",
            "inventory_recently_seen",
            device_info,
            "inventory_recently_seen",
        ),
        GroceryAnalyticsSensor(
            coordinator,
            f"{DOMAIN}_top_price_increases",
            "Top price increases",
            "top_price_increases",
            device_info,
            "top_increases",
        ),
        GroceryAnalyticsSensor(
            coordinator,
            f"{DOMAIN}_overpaid_items",
            "Overpaid items",
            "overpaid_items",
            device_info,
            "overpaid_items",
        ),
        GroceryAnalyticsSensor(
            coordinator,
            f"{DOMAIN}_best_store_by_item",
            "Best store by item",
            "best_store_by_item",
            device_info,
            "best_store_items",
        ),
    ]

    async_add_entities(entities)


class GrocerySpendSensor(CoordinatorEntity[GrocerySpendCoordinator], SensorEntity):
    """Representation of spend sensors."""

    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        coordinator: GrocerySpendCoordinator,
        unique_id: str,
        name: str,
        suggested_object_id: str,
        device_info: DeviceInfo,
        currency_symbol: str,
        snapshot_key: str,
    ) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = unique_id
        self._attr_name = name
        self._attr_suggested_object_id = suggested_object_id
        self._attr_device_info = device_info
        self._snapshot_key = snapshot_key
        self._attr_native_unit_of_measurement = currency_symbol

    @property
    def native_value(self) -> float | None:
        data = self.coordinator.data
        if not data:
            return None
        value = getattr(data, self._snapshot_key, None)
        if value is None:
            return None
        return round(float(value), 2)


class GroceryCountSensor(CoordinatorEntity[GrocerySpendCoordinator], SensorEntity):
    """Simple count sensor."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GrocerySpendCoordinator,
        unique_id: str,
        name: str,
        suggested_object_id: str,
        device_info: DeviceInfo,
        snapshot_key: str,
    ) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = unique_id
        self._attr_name = name
        self._attr_suggested_object_id = suggested_object_id
        self._attr_device_info = device_info
        self._snapshot_key = snapshot_key

    @property
    def native_value(self) -> int | None:
        data = self.coordinator.data
        if not data:
            return None
        value = getattr(data, self._snapshot_key, None)
        if value is None:
            return None
        return int(value)


class GroceryAnalyticsSensor(CoordinatorEntity[GrocerySpendCoordinator], SensorEntity):
    """Analytics sensor with list attributes."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GrocerySpendCoordinator,
        unique_id: str,
        name: str,
        suggested_object_id: str,
        device_info: DeviceInfo,
        key: str,
    ) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = unique_id
        self._attr_name = name
        self._attr_suggested_object_id = suggested_object_id
        self._attr_device_info = device_info
        self._key = key

    @property
    def native_value(self) -> int | None:
        data = self.coordinator.data
        if not data:
            return None
        items = getattr(data, self._key)
        return len(items)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        data = self.coordinator.data
        if not data:
            return None
        return {"items": getattr(data, self._key)}


class GroceryReceiptProcessingSensor(CoordinatorEntity[GrocerySpendCoordinator], SensorEntity):
    """Receipt extraction processing status counts."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: GrocerySpendCoordinator,
        unique_id: str,
        name: str,
        suggested_object_id: str,
        device_info: DeviceInfo,
    ) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = unique_id
        self._attr_name = name
        self._attr_suggested_object_id = suggested_object_id
        self._attr_device_info = device_info

    @property
    def native_value(self) -> int | None:
        data = self.coordinator.data
        if not data:
            return None
        counts = data.receipt_status_counts or {}
        return int(counts.get("pending", 0) + counts.get("queued", 0) + counts.get("running", 0))

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        data = self.coordinator.data
        if not data:
            return None
        counts = data.receipt_status_counts or {}
        total = sum(int(v) for v in counts.values())
        in_progress = int(counts.get("pending", 0) + counts.get("queued", 0) + counts.get("running", 0))
        failed = int(counts.get("failed", 0))
        return {
            "total_receipts": total,
            "in_progress": in_progress,
            "failed": failed,
            "status_counts": counts,
            "timing": data.receipt_processing_timing or {},
        }


def _compute_spend_totals(receipts: list[dict[str, Any]]) -> tuple[float, float]:
    now = dt_util.as_local(dt_util.now())
    iso_year, iso_week, _ = now.isocalendar()

    week_total = 0.0
    month_total = 0.0

    for receipt in receipts:
        dt = _parse_receipt_datetime(receipt.get("purchased_at"))
        if dt is None:
            continue

        local_dt = dt_util.as_local(dt)
        if local_dt.isocalendar()[:2] == (iso_year, iso_week):
            week_total += float(receipt.get("total", 0))

        if (local_dt.year, local_dt.month) == (now.year, now.month):
            month_total += float(receipt.get("total", 0))

    return week_total, month_total


def _compute_receipt_status_counts(receipts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for receipt in receipts:
        status = receipt.get("extract_status")
        if not status:
            status = "pending" if receipt.get("file_path") else "done"
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def _compute_receipt_processing_timing(
    receipts: list[dict[str, Any]], *, recent_n: int = 200
) -> dict[str, Any]:
    def _parse_ts(value: Any):
        if not value:
            return None
        dt = dt_util.parse_datetime(str(value))
        return dt_util.as_local(dt) if dt is not None else None

    def _pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        values_sorted = sorted(values)
        idx = int(round((len(values_sorted) - 1) * p))
        idx = max(0, min(len(values_sorted) - 1, idx))
        return values_sorted[idx]

    def _summarize(seconds: list[float]) -> dict[str, Any]:
        if not seconds:
            return {"count": 0}
        return {
            "count": len(seconds),
            "avg_s": round(sum(seconds) / len(seconds), 2),
            "median_s": round(float(median(seconds)), 2),
            "p95_s": round(float(_pct(seconds, 0.95) or 0.0), 2),
        }

    rows: list[dict[str, Any]] = []
    for r in receipts:
        if r.get("extract_status") != "done":
            continue
        ms = r.get("extract_duration_ms")
        try:
            ms_i = int(ms) if ms is not None else None
        except Exception:
            ms_i = None
        if not ms_i or ms_i <= 0:
            continue
        ts = _parse_ts(r.get("extract_finished_at")) or _parse_ts(r.get("created_at"))
        method = (r.get("extract_method") or "unknown").lower()
        provider = (r.get("extract_provider") or "unknown").lower()
        rows.append(
            {
                "ts": ts,
                "method": method,
                "provider": provider,
                "seconds": ms_i / 1000.0,
            }
        )

    rows.sort(key=lambda x: x.get("ts") or dt_util.as_local(dt_util.now()), reverse=True)
    window = rows[: max(0, int(recent_n))]

    overall_seconds = [r["seconds"] for r in window]
    by_method: dict[str, list[float]] = {}
    by_llm_provider: dict[str, list[float]] = {}
    for r in window:
        by_method.setdefault(r["method"], []).append(r["seconds"])
        if r["method"] == "llm":
            by_llm_provider.setdefault(r["provider"], []).append(r["seconds"])

    return {
        "window_n": int(recent_n),
        "considered": len(window),
        "overall": _summarize(overall_seconds),
        "by_method": {k: _summarize(v) for k, v in sorted(by_method.items())},
        "by_llm_provider": {k: _summarize(v) for k, v in sorted(by_llm_provider.items())},
    }


def _compute_rolling_stats(
    receipts: list[dict[str, Any]],
) -> tuple[float, float, int, float, list[dict[str, Any]]]:
    now = dt_util.as_local(dt_util.now())
    cutoff_7 = now - timedelta(days=7)
    cutoff_30 = now - timedelta(days=30)

    total_7 = 0.0
    total_30 = 0.0
    count_30 = 0
    totals_30: list[float] = []
    by_store: dict[str, float] = {}

    for receipt in receipts:
        dt = _parse_receipt_datetime(receipt.get("purchased_at"))
        if dt is None:
            continue
        local_dt = dt_util.as_local(dt)
        total = receipt.get("total")
        try:
            total_f = float(total) if total is not None else None
        except Exception:
            total_f = None
        if total_f is None:
            continue

        if local_dt >= cutoff_7:
            total_7 += total_f
        if local_dt >= cutoff_30:
            total_30 += total_f
            count_30 += 1
            totals_30.append(total_f)
            store = receipt.get("store_name") or "Unknown"
            by_store[store] = by_store.get(store, 0.0) + total_f

    avg_basket = (sum(totals_30) / len(totals_30)) if totals_30 else 0.0

    top_stores = [
        {"store_name": k, "total": round(v, 2)}
        for k, v in sorted(by_store.items(), key=lambda kv: kv[1], reverse=True)[:10]
    ]

    return total_7, total_30, count_30, avg_basket, top_stores


def _compute_recent_receipts(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in receipts:
        dt = _parse_receipt_datetime(r.get("purchased_at"))
        if dt is None:
            continue
        rows.append(
            {
                "receipt_id": r.get("id"),
                "purchased_at": dt_util.as_local(dt).isoformat(),
                "store_name": r.get("store_name"),
                "total": r.get("total"),
                "filename": r.get("filename"),
            }
        )
    rows.sort(key=lambda x: x.get("purchased_at") or "", reverse=True)
    return rows[:20]


def _compute_recent_activities(activities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _compact_payload(payload: Any) -> dict[str, Any] | None:
        """Shrink payload so sensor attributes stay under HA's 16KB limit."""
        if not isinstance(payload, dict):
            return None

        drop_keys = {
            # Very large / repetitive payloads
            "boosts",
            "detected",
            "items",
            "line_items",
            "line_items_raw",
            "raw_result",
            "raw_text",
        }
        out: dict[str, Any] = {}

        def _compact_value(value: Any) -> Any:
            if value is None or isinstance(value, (int, float, bool)):
                return value
            if isinstance(value, str):
                v = value.strip()
                return (v[:117] + "...") if len(v) > 120 else v
            if isinstance(value, list):
                return {"count": len(value)}
            if isinstance(value, dict):
                # Keep only shallow primitive values.
                sub: dict[str, Any] = {}
                for sk, sv in value.items():
                    if sv is None or isinstance(sv, (int, float, bool)):
                        sub[str(sk)] = sv
                    elif isinstance(sv, str):
                        ssv = sv.strip()
                        sub[str(sk)] = (ssv[:117] + "...") if len(ssv) > 120 else ssv
                return sub or {"keys": len(value)}
            return str(value)[:120]

        for k, v in payload.items():
            key = str(k)
            if key in drop_keys:
                continue
            out[key] = _compact_value(v)

        return out or None

    rows: list[dict[str, Any]] = []
    for a in activities:
        description = a.get("description")
        if isinstance(description, str) and len(description) > 180:
            description = description[:177] + "..."
        rows.append(
            {
                "activity_id": a.get("activity_id"),
                "timestamp": a.get("timestamp"),
                "kind": a.get("kind"),
                "description": description,
                "payload": _compact_payload(a.get("payload")),
            }
        )
    rows.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return rows[:25]


def _compute_inventory_recently_seen(
    products: list[dict[str, Any]],
    storage: ReceiptStorage,
    evidence_ttl_days: int,
) -> list[dict[str, Any]]:
    if evidence_ttl_days <= 0:
        return []

    now = dt_util.as_local(dt_util.now())
    rows: list[dict[str, Any]] = []

    for p in products:
        pid = p.get("product_id")
        if not pid:
            continue
        state = storage.get_shopping_product_state(pid)
        last_seen_raw = state.get("last_seen_at")
        if not last_seen_raw:
            continue
        last_seen = dt_util.parse_datetime(str(last_seen_raw))
        if last_seen is None:
            continue
        last_seen_local = dt_util.as_local(last_seen)
        expires = last_seen_local + timedelta(days=evidence_ttl_days)
        if now >= expires:
            continue
        try:
            conf = float(state.get("last_seen_confidence")) if state.get("last_seen_confidence") is not None else None
        except Exception:
            conf = None
        rows.append(
            {
                "product_id": pid,
                "product": p.get("canonical_name", pid),
                "last_seen_at": last_seen_local.isoformat(),
                "expires_at": expires.isoformat(),
                "confidence": round(conf, 3) if isinstance(conf, float) else None,
            }
        )

    rows.sort(key=lambda x: x.get("last_seen_at") or "", reverse=True)
    return rows[:100]


def _group_observations_by_product(observations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        pid = obs.get("product_id")
        if not pid:
            continue
        grouped.setdefault(pid, []).append(obs)

    for obs_list in grouped.values():
        obs_list.sort(key=lambda o: o.get("observed_at", ""))

    return grouped


def _compute_baselines(obs_by_product: dict[str, list[dict[str, Any]]], window_n: int) -> dict[str, dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    for pid, obs_list in obs_by_product.items():
        unit_prices = [
            o.get("unit_price")
            for o in obs_list
            if isinstance(o.get("unit_price"), (int, float))
        ]
        if len(unit_prices) < 3:
            continue
        recent_prices = unit_prices[-window_n:]
        baseline_value = median(recent_prices)
        baselines[pid] = {
            "baseline": baseline_value,
            "count": len(unit_prices),
            "confidence": min(100, len(unit_prices) * 20),
        }
    return baselines


def _compute_overpaid_items(
    obs_by_product: dict[str, list[dict[str, Any]]],
    baselines: dict[str, dict[str, Any]],
    product_lookup: dict[str, dict[str, Any]],
    overpaid_pct: float,
) -> list[dict[str, Any]]:
    now = dt_util.as_local(dt_util.now())
    iso_year, iso_week, _ = now.isocalendar()
    items: list[dict[str, Any]] = []

    for pid, obs_list in obs_by_product.items():
        baseline_info = baselines.get(pid)
        if not baseline_info:
            continue
        baseline = baseline_info["baseline"]
        if baseline <= 0:
            continue
        for obs in obs_list:
            unit_price = obs.get("unit_price")
            if not isinstance(unit_price, (int, float)):
                continue
            dt = _parse_receipt_datetime(obs.get("observed_at"))
            if dt is None:
                continue
            local_dt = dt_util.as_local(dt)
            if local_dt.isocalendar()[:2] != (iso_year, iso_week):
                continue

            overpaid_value = unit_price - baseline
            if unit_price < baseline * (1 + overpaid_pct):
                continue
            if overpaid_value < SMALL_OVERPAY_FLOOR:
                continue

            product = product_lookup.get(pid, {})
            items.append(
                {
                    "product": product.get("canonical_name", pid),
                    "store": obs.get("store_name"),
                    "unit_price": round(unit_price, 2),
                    "baseline": round(baseline, 2),
                    "overpaid_pct": round(overpaid_value / baseline, 3),
                    "observed_at": obs.get("observed_at"),
                }
            )

    items.sort(key=lambda i: i.get("overpaid_pct", 0), reverse=True)
    return items[:10]


def _compute_top_increases(
    obs_by_product: dict[str, list[dict[str, Any]]],
    product_lookup: dict[str, dict[str, Any]],
    recent_days: int,
    prior_days: int,
) -> list[dict[str, Any]]:
    now = dt_util.as_local(dt_util.now())
    items: list[dict[str, Any]] = []
    recent_cutoff = now - timedelta(days=recent_days)
    prior_cutoff = now - timedelta(days=prior_days)

    for pid, obs_list in obs_by_product.items():
        recent_prices = []
        prior_prices = []
        total_prices = []
        unit_type = None
        for obs in obs_list:
            unit_price = obs.get("unit_price")
            if not isinstance(unit_price, (int, float)):
                continue
            dt = _parse_receipt_datetime(obs.get("observed_at"))
            if dt is None:
                continue
            local_dt = dt_util.as_local(dt)
            total_prices.append(unit_price)
            unit_type = unit_type or obs.get("unit_type")
            if local_dt >= recent_cutoff:
                recent_prices.append(unit_price)
            elif local_dt >= prior_cutoff:
                prior_prices.append(unit_price)

        if not prior_prices:
            continue
        if len(recent_prices) < 2 and len(total_prices) < 3:
            continue

        recent_median = median(recent_prices) if recent_prices else None
        prior_median = median(prior_prices)
        if prior_median <= 0:
            continue

        if recent_median is None:
            continue
        increase_pct = (recent_median - prior_median) / prior_median
        if increase_pct <= 0:
            continue

        product = product_lookup.get(pid, {})
        items.append(
            {
                "product": product.get("canonical_name", pid),
                "increase_pct": round(increase_pct, 3),
                "recent_median": round(recent_median, 2),
                "prior_median": round(prior_median, 2),
                "unit_type": unit_type or "unknown",
                "observations": len(total_prices),
            }
        )

    items.sort(key=lambda i: i.get("increase_pct", 0), reverse=True)
    return items[:10]


def _compute_best_store(
    obs_by_product: dict[str, list[dict[str, Any]]],
    product_lookup: dict[str, dict[str, Any]],
    window_days: int,
) -> list[dict[str, Any]]:
    now = dt_util.as_local(dt_util.now())
    window_cutoff = now - timedelta(days=window_days)
    recent_cutoff = now - timedelta(days=60)

    items: list[dict[str, Any]] = []

    for pid, obs_list in obs_by_product.items():
        store_prices: dict[str, list[float]] = {}
        all_prices: list[float] = []
        recent_count = 0
        total_count = 0
        unit_type = None

        for obs in obs_list:
            unit_price = obs.get("unit_price")
            if not isinstance(unit_price, (int, float)):
                continue
            dt = _parse_receipt_datetime(obs.get("observed_at"))
            if dt is None:
                continue
            local_dt = dt_util.as_local(dt)
            if local_dt < window_cutoff:
                continue

            total_count += 1
            if local_dt >= recent_cutoff:
                recent_count += 1

            store = obs.get("store_name") or "Unknown"
            store_prices.setdefault(store, []).append(unit_price)
            all_prices.append(unit_price)
            unit_type = unit_type or obs.get("unit_type")

        if total_count < 4 and recent_count < 2:
            continue
        if len(all_prices) < 2:
            continue

        overall_median = median(all_prices)
        if overall_median <= 0:
            continue

        best_store = None
        best_median = None
        for store, prices in store_prices.items():
            if len(prices) < 2:
                continue
            store_median = median(prices)
            if best_median is None or store_median < best_median:
                best_median = store_median
                best_store = store

        if best_store is None or best_median is None:
            continue

        savings_pct = (overall_median - best_median) / overall_median
        if savings_pct < 0.05:
            continue

        product = product_lookup.get(pid, {})
        items.append(
            {
                "product": product.get("canonical_name", pid),
                "best_store": best_store,
                "best_median": round(best_median, 2),
                "overall_median": round(overall_median, 2),
                "savings_pct": round(savings_pct, 3),
                "unit_type": unit_type or "unknown",
            }
        )

    items.sort(key=lambda i: i.get("savings_pct", 0), reverse=True)
    return items[:10]


def _parse_receipt_datetime(value: Any):
    if not value:
        return None
    if isinstance(value, str):
        dt = dt_util.parse_datetime(value)
        if dt is None:
            date_obj = dt_util.parse_date(value)
            if date_obj is None:
                return None
            return dt_util.start_of_local_day(date_obj)
        return dt
    return None
