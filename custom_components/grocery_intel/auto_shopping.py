"""Daily auto-approve shopping list automation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from statistics import median
from typing import Any

from homeassistant.util import dt as dt_util

from .const import (
    CONF_SHOPPING_AUTO_APPROVE_ENABLED,
    CONF_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
    CONF_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
    CONF_SHOPPING_PAUSE_WHEN_ALL_AWAY,
    CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
    DEFAULT_SHOPPING_AUTO_APPROVE_ENABLED,
    DEFAULT_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
    DEFAULT_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
    DEFAULT_SHOPPING_PAUSE_WHEN_ALL_AWAY,
    DEFAULT_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
)
from .shopping_list_api import async_add_item, async_get_items


AUTO_RUN_HOUR_LOCAL = 7
AWAY_MIN_HOURS = 48
MIN_PURCHASES = 3

LEVEL_PLENTY = "plenty"
LEVEL_MEDIUM = "medium"
LEVEL_LIKELY_LOW = "likely_low"


@dataclass
class ShoppingCandidate:
    product_id: str
    name: str
    level: str
    confidence: float
    reason: str


def _all_people_away_for(hass, *, hours: int) -> bool:
    people = hass.states.async_all("person")
    if not people:
        return False
    now = dt_util.now()
    for st in people:
        if st.state == "home":
            return False
        if (now - st.last_changed) < timedelta(hours=hours):
            return False
    return True


def _compute_level_and_confidence(purchase_dts: list) -> tuple[str, float, float, float]:
    """Return (level, confidence, cadence_days, days_since_last)."""
    if len(purchase_dts) < MIN_PURCHASES:
        return LEVEL_MEDIUM, 0.0, 0.0, 0.0

    purchase_dts = sorted(purchase_dts)
    deltas = [
        max(0.0, (purchase_dts[i] - purchase_dts[i - 1]).total_seconds() / 86400.0)
        for i in range(1, len(purchase_dts))
    ]
    if not deltas:
        return LEVEL_MEDIUM, 0.0, 0.0, 0.0

    cadence = float(median(deltas))
    if cadence <= 0:
        return LEVEL_MEDIUM, 0.0, 0.0, 0.0

    now = dt_util.as_local(dt_util.now())
    last = dt_util.as_local(purchase_dts[-1])
    days_since_last = max(0.0, (now - last).total_seconds() / 86400.0)

    ratio = days_since_last / cadence
    if ratio < 0.7:
        level = LEVEL_PLENTY
    elif ratio < 1.0:
        level = LEVEL_MEDIUM
    else:
        level = LEVEL_LIKELY_LOW

    mad = float(median([abs(x - cadence) for x in deltas])) if deltas else 0.0
    stability = 1.0 - min(1.0, (mad / cadence)) if cadence > 0 else 0.0
    history = min(1.0, len(purchase_dts) / 10.0)
    confidence = max(0.0, min(1.0, 0.6 * stability + 0.4 * history))

    return level, confidence, cadence, days_since_last


def _norm_key(value: str) -> str:
    return (value or "").casefold().strip()


async def async_run_auto_shopping(hass, entry, data, *, dry_run: bool = False) -> dict[str, Any]:
    """Compute likely-low items and auto-add them to the default shopping list."""
    options = entry.options
    enabled = options.get(CONF_SHOPPING_AUTO_APPROVE_ENABLED, DEFAULT_SHOPPING_AUTO_APPROVE_ENABLED)
    cooldown_days = int(
        options.get(
            CONF_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
            DEFAULT_SHOPPING_AUTO_APPROVE_COOLDOWN_DAYS,
        )
    )
    threshold = float(
        options.get(
            CONF_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
            DEFAULT_SHOPPING_AUTO_APPROVE_CONFIDENCE_THRESHOLD,
        )
    )
    pause_when_away = bool(
        options.get(CONF_SHOPPING_PAUSE_WHEN_ALL_AWAY, DEFAULT_SHOPPING_PAUSE_WHEN_ALL_AWAY)
    )
    evidence_ttl_days = int(
        options.get(
            CONF_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
            DEFAULT_INVENTORY_IMAGES_EVIDENCE_TTL_DAYS,
        )
    )

    now = dt_util.as_local(dt_util.now())
    result: dict[str, Any] = {
        "enabled": bool(enabled),
        "dry_run": bool(dry_run),
        "started_at": now.isoformat(),
        "added": [],
        "skipped": {
            "not_enabled": [],
            "not_enough_history": [],
            "cooldown": [],
            "already_present": [],
            "low_confidence": [],
            "not_low": [],
            "recently_seen": [],
        },
    }

    if not enabled:
        return result

    if pause_when_away and _all_people_away_for(hass, hours=AWAY_MIN_HOURS):
        result["skipped"]["not_enabled"].append("paused_all_away")
        return result

    observations = await data.storage.async_list_observations()
    products = await data.storage.async_list_products()
    product_lookup = {p.get("product_id"): p for p in products if p.get("product_id")}

    purchase_by_product: dict[str, list] = {}
    for obs in observations:
        pid = obs.get("product_id")
        if not pid:
            continue
        dt = dt_util.parse_datetime(obs.get("observed_at"))
        if dt is None:
            continue
        purchase_by_product.setdefault(pid, []).append(dt)

    candidates: list[ShoppingCandidate] = []
    for pid, dts in purchase_by_product.items():
        if len(dts) < MIN_PURCHASES:
            result["skipped"]["not_enough_history"].append(pid)
            continue
        level, conf, cadence, days_since_last = _compute_level_and_confidence(dts)
        if level != LEVEL_LIKELY_LOW:
            result["skipped"]["not_low"].append(pid)
            continue
        if conf < threshold:
            result["skipped"]["low_confidence"].append(pid)
            continue

        product = product_lookup.get(pid, {})
        name = (product.get("canonical_name") or "").strip() or pid

        state = data.storage.get_shopping_product_state(pid)
        last_added_raw = state.get("last_auto_added_at")
        last_added = dt_util.parse_datetime(last_added_raw) if last_added_raw else None
        if last_added is not None:
            last_added_local = dt_util.as_local(last_added)
            if now < (last_added_local + timedelta(days=cooldown_days)):
                result["skipped"]["cooldown"].append(pid)
                continue

        last_seen_raw = state.get("last_seen_at")
        last_seen = dt_util.parse_datetime(last_seen_raw) if last_seen_raw else None
        if last_seen is not None and evidence_ttl_days > 0:
            last_seen_local = dt_util.as_local(last_seen)
            if now < (last_seen_local + timedelta(days=evidence_ttl_days)):
                result["skipped"]["recently_seen"].append(pid)
                continue

        reason = f"Typical cadence ~{cadence:.1f}d; last bought {days_since_last:.0f}d ago"
        candidates.append(
            ShoppingCandidate(
                product_id=pid,
                name=name,
                level=level,
                confidence=conf,
                reason=reason,
            )
        )

    existing_items = await async_get_items(hass)
    existing_names = {
        _norm_key(i.get("name", ""))
        for i in existing_items
        if not i.get("complete", False)
    }

    added_rows: list[dict[str, Any]] = []
    state_updates: dict[str, dict[str, Any]] = {}

    for cand in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        if _norm_key(cand.name) in existing_names:
            result["skipped"]["already_present"].append(cand.product_id)
            continue

        if dry_run:
            added_rows.append(
                {
                    "product_id": cand.product_id,
                    "name": cand.name,
                    "reason": cand.reason,
                    "confidence": round(cand.confidence, 3),
                    "shopping_list_item_id": None,
                }
            )
            continue

        created = await async_add_item(hass, cand.name)
        if not created or not created.get("id"):
            continue

        item_id = str(created.get("id"))
        existing_names.add(_norm_key(cand.name))
        added_rows.append(
            {
                "product_id": cand.product_id,
                "name": cand.name,
                "reason": cand.reason,
                "confidence": round(cand.confidence, 3),
                "shopping_list_item_id": item_id,
            }
        )
        state_updates[cand.product_id] = {"last_auto_added_at": now.isoformat()}

    if state_updates:
        await data.storage.async_bulk_update_shopping_product_state(state_updates)

    result["added"] = added_rows
    return result
