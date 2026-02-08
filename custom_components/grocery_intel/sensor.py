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

from .const import (
    DOMAIN,
    CONF_CURRENCY_SYMBOL,
    CONF_OVERPAID_PCT_THRESHOLD,
    CONF_BASELINE_WINDOW_N,
    CONF_TOP_INCREASES_RECENT_DAYS,
    CONF_TOP_INCREASES_PRIOR_DAYS,
    CONF_BEST_STORE_WINDOW_DAYS,
    DEFAULT_CURRENCY_SYMBOL,
    DEFAULT_OVERPAID_PCT_THRESHOLD,
    DEFAULT_BASELINE_WINDOW_N,
    DEFAULT_TOP_INCREASES_RECENT_DAYS,
    DEFAULT_TOP_INCREASES_PRIOR_DAYS,
    DEFAULT_BEST_STORE_WINDOW_DAYS,
    SMALL_OVERPAY_FLOOR,
)
from .storage import ReceiptStorage


@dataclass
class GroceryIntelDataSnapshot:
    week_total: float
    month_total: float
    top_increases: list[dict[str, Any]]
    overpaid_items: list[dict[str, Any]]
    best_store_items: list[dict[str, Any]]


class GrocerySpendCoordinator(DataUpdateCoordinator[GroceryIntelDataSnapshot]):
    """Coordinator to calculate spend totals and analytics."""

    def __init__(self, hass: HomeAssistant, storage: ReceiptStorage, entry: ConfigEntry) -> None:
        super().__init__(hass, logger=logging.getLogger(__name__), name=DOMAIN, update_interval=None)
        self._storage = storage
        self._entry = entry

    async def _async_update_data(self) -> GroceryIntelDataSnapshot:
        receipts = await self._storage.async_list_receipts()
        observations = await self._storage.async_list_observations()
        products = await self._storage.async_list_products()

        week_total, month_total = _compute_spend_totals(receipts)

        options = self._entry.options
        overpaid_pct = options.get(CONF_OVERPAID_PCT_THRESHOLD, DEFAULT_OVERPAID_PCT_THRESHOLD)
        baseline_n = options.get(CONF_BASELINE_WINDOW_N, DEFAULT_BASELINE_WINDOW_N)
        recent_days = options.get(CONF_TOP_INCREASES_RECENT_DAYS, DEFAULT_TOP_INCREASES_RECENT_DAYS)
        prior_days = options.get(CONF_TOP_INCREASES_PRIOR_DAYS, DEFAULT_TOP_INCREASES_PRIOR_DAYS)
        best_store_days = options.get(CONF_BEST_STORE_WINDOW_DAYS, DEFAULT_BEST_STORE_WINDOW_DAYS)

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

        return GroceryIntelDataSnapshot(
            week_total=week_total,
            month_total=month_total,
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
            "week",
        ),
        GrocerySpendSensor(
            coordinator,
            f"{DOMAIN}_spend_month",
            "Spend month",
            "spend_month",
            device_info,
            currency,
            "month",
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
        key: str,
    ) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = unique_id
        self._attr_name = name
        self._attr_suggested_object_id = suggested_object_id
        self._attr_device_info = device_info
        self._key = key
        self._attr_native_unit_of_measurement = currency_symbol

    @property
    def native_value(self) -> float | None:
        data = self.coordinator.data
        if not data:
            return None
        if self._key == "week":
            return round(data.week_total, 2)
        return round(data.month_total, 2)


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
