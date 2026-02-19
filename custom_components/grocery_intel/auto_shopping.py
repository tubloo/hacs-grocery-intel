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
from .shopping_list_api import async_add_item, async_get_items, async_update_item


AUTO_RUN_HOUR_LOCAL = 7
AWAY_MIN_HOURS = 48
MIN_PURCHASES = 3

# Store recommendation + tagging (for auto-added items).
STORE_RECO_LOOKBACK_DAYS = 120
STORE_RECO_MAX_STALE_DAYS = 60
STORE_RECO_MIN_WINNER_N = 3
STORE_RECO_MIN_ALT_N = 2
STORE_RECO_MIN_MATCH_CONF = 75  # observation.confidence is 0-100
STORE_RECO_MIN_GAP_PCT = 0.02
STORE_TAG_CONF_THRESHOLD = 0.75
STORE_TAG_RETAG_HYSTERESIS = 0.10

MAX_STORES_PER_RUN = 2
MULTI_STORE_PENALTY_PCT = 0.05

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


def _split_store_tag(name: str) -> tuple[str, str | None]:
    """Split 'Item @ Store' -> ('Item', 'Store'). Returns (name, None) if untagged."""
    raw = (name or "").strip()
    if not raw:
        return "", None
    if " @ " not in raw:
        return raw, None
    base, store = raw.rsplit(" @ ", 1)
    base = base.strip()
    store = store.strip()
    if not base or not store:
        return raw, None
    return base, store


def _pct25_75(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 4:
        return None
    vs = sorted(values)
    n = len(vs)

    def _pick(p: float) -> float:
        idx = int(round((n - 1) * p))
        idx = max(0, min(n - 1, idx))
        return float(vs[idx])

    return _pick(0.25), _pick(0.75)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


async def _recommend_store_for_product(
    *,
    data,
    product_id: str,
    now_local,
    observations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a store recommendation dict or None if unknown/low confidence."""
    cutoff = now_local - timedelta(days=STORE_RECO_LOOKBACK_DAYS)

    rows: list[dict[str, Any]] = []
    for obs in observations:
        if obs.get("product_id") != product_id:
            continue
        store_raw = (obs.get("store_name") or "").strip()
        if not store_raw:
            continue
        try:
            unit_price = float(obs.get("unit_price")) if obs.get("unit_price") is not None else None
        except Exception:
            unit_price = None
        if unit_price is None or unit_price <= 0:
            continue
        unit_type = (obs.get("unit_type") or "").strip() or "unknown"
        conf = obs.get("confidence")
        try:
            conf_i = int(conf) if conf is not None else 0
        except Exception:
            conf_i = 0
        if conf_i < STORE_RECO_MIN_MATCH_CONF:
            continue

        dt = dt_util.parse_datetime(obs.get("observed_at"))
        if dt is None:
            continue
        dt_local = dt_util.as_local(dt)
        if dt_local < cutoff:
            continue

        store_norm = (data.storage.resolve_store_alias(store_raw) or store_raw).strip()
        if not store_norm:
            continue

        rows.append(
            {
                "store": store_norm,
                "unit_price": unit_price,
                "unit_type": unit_type,
                "observed_at": dt_local,
            }
        )

    if not rows:
        return None

    # Use the most common unit_type to avoid comparing apples to oranges.
    counts: dict[str, int] = {}
    for r in rows:
        key = str(r.get("unit_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    unit_type = max(counts.items(), key=lambda kv: kv[1])[0]
    rows = [r for r in rows if (r.get("unit_type") or "unknown") == unit_type]
    if len(rows) < STORE_RECO_MIN_WINNER_N:
        return None

    by_store: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_store.setdefault(str(r["store"]), []).append(r)

    store_stats: dict[str, dict[str, Any]] = {}
    for store, srows in by_store.items():
        prices = [float(r["unit_price"]) for r in srows if r.get("unit_price") is not None]
        if not prices:
            continue
        last_seen = max(r["observed_at"] for r in srows if r.get("observed_at") is not None)
        store_stats[store] = {
            "n": len(prices),
            "median": float(median(prices)),
            "prices": prices,
            "last_seen": last_seen,
        }

    if not store_stats:
        return None

    candidates = []
    for store, st in store_stats.items():
        n = int(st.get("n") or 0)
        if n < STORE_RECO_MIN_WINNER_N:
            continue
        last_seen = st.get("last_seen")
        if last_seen is None:
            continue
        days_stale = (now_local - last_seen).total_seconds() / 86400.0
        if days_stale > STORE_RECO_MAX_STALE_DAYS:
            continue
        candidates.append((store, st))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: float(pair[1].get("median") or 1e18))
    winner_store, winner = candidates[0]

    alt = [
        (s, st)
        for s, st in candidates[1:]
        if int(st.get("n") or 0) >= STORE_RECO_MIN_ALT_N
    ]
    if not alt:
        return None
    runner_store, runner = alt[0]

    winner_m = float(winner.get("median") or 0.0)
    runner_m = float(runner.get("median") or 0.0)
    if winner_m <= 0 or runner_m <= 0:
        return None

    gap_pct = (runner_m - winner_m) / runner_m
    if gap_pct < STORE_RECO_MIN_GAP_PCT:
        return None

    n_w = int(winner.get("n") or 0)
    n_r = int(runner.get("n") or 0)
    suff_w = _clamp01(n_w / STORE_RECO_MIN_WINNER_N)
    suff_r = _clamp01(n_r / STORE_RECO_MIN_ALT_N)
    suff = 0.6 * suff_w + 0.4 * suff_r

    margin = _clamp01((gap_pct - 0.02) / 0.10)

    last_seen_w = winner.get("last_seen")
    if last_seen_w is None:
        return None
    days_since = max(0.0, (now_local - last_seen_w).total_seconds() / 86400.0)
    if days_since <= 7:
        recency = 1.0
    elif days_since <= 30:
        recency = 1.0 - ((days_since - 7) / (30 - 7)) * 0.4
    elif days_since <= 60:
        recency = 0.6 - ((days_since - 30) / (60 - 30)) * 0.6
    else:
        recency = 0.0
    recency = _clamp01(recency)

    stability = 0.6
    if n_w >= 4:
        prices_w = list(winner.get("prices") or [])
        qs = _pct25_75([float(x) for x in prices_w if x is not None])
        if qs:
            q25, q75 = qs
            iqr = max(0.0, float(q75) - float(q25))
            rel_iqr = iqr / max(1e-9, winner_m)
            stability = _clamp01(1.0 - (rel_iqr / 0.25))

    raw = 0.45 * margin + 0.30 * stability + 0.25 * recency
    confidence = _clamp01(suff * raw)

    price_by_store = {s: float(st.get("median") or 0.0) for s, st in store_stats.items()}

    return {
        "store": winner_store,
        "confidence": round(confidence, 3),
        "unit_type": unit_type,
        "winner_median": round(winner_m, 4),
        "runner_up_store": runner_store,
        "runner_up_median": round(runner_m, 4),
        "gap_pct": round(gap_pct, 4),
        "price_by_store": price_by_store,
    }


def _pick_multi_store_plan(recs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pick a simple <=2-store plan for this run."""
    store_counts: dict[str, int] = {}
    for r in recs.values():
        store = (r.get("store") or "").strip()
        conf = float(r.get("confidence") or 0.0)
        if not store or conf < STORE_TAG_CONF_THRESHOLD:
            continue
        store_counts[store] = store_counts.get(store, 0) + 1

    if not store_counts:
        return {"stores": [], "assignments": {}}

    primary = max(store_counts.items(), key=lambda kv: kv[1])[0]

    remaining: dict[str, int] = {}
    for pid, r in recs.items():
        store = (r.get("store") or "").strip()
        conf = float(r.get("confidence") or 0.0)
        if not store or conf < STORE_TAG_CONF_THRESHOLD:
            continue
        if store == primary:
            continue
        remaining[store] = remaining.get(store, 0) + 1
    secondary = max(remaining.items(), key=lambda kv: kv[1])[0] if remaining else None

    stores = [primary]
    if secondary and secondary != primary and MAX_STORES_PER_RUN >= 2:
        stores.append(secondary)

    assignments: dict[str, str] = {}
    for pid, r in recs.items():
        best = (r.get("store") or "").strip()
        conf = float(r.get("confidence") or 0.0)
        if not best or conf < STORE_TAG_CONF_THRESHOLD:
            continue

        price_by_store = r.get("price_by_store") or {}
        try:
            best_price = float(price_by_store.get(best) or 0.0)
        except Exception:
            best_price = 0.0
        if best_price <= 0:
            continue

        chosen = best
        if primary in price_by_store:
            try:
                p_primary = float(price_by_store.get(primary) or 0.0)
            except Exception:
                p_primary = 0.0
            if p_primary > 0:
                penalty = (p_primary - best_price) / best_price
                if penalty <= MULTI_STORE_PENALTY_PCT:
                    chosen = primary

        if chosen not in stores:
            # Pick the cheapest allowed store with known price.
            best_allowed = None
            best_allowed_price = None
            for s in stores:
                try:
                    p = float(price_by_store.get(s) or 0.0)
                except Exception:
                    p = 0.0
                if p <= 0:
                    continue
                if best_allowed_price is None or p < best_allowed_price:
                    best_allowed = s
                    best_allowed_price = p
            if best_allowed:
                chosen = best_allowed
            else:
                continue

        assignments[pid] = chosen

    return {
        "stores": stores,
        "primary": primary,
        "secondary": secondary,
        "penalty_pct": MULTI_STORE_PENALTY_PCT,
        "assignments": assignments,
    }


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
        "renamed": [],
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
    existing_open = [i for i in existing_items if not i.get("complete", False)]
    existing_by_base: dict[str, dict[str, Any]] = {}
    existing_base_keys: set[str] = set()
    for item in existing_open:
        name = str(item.get("name", "") or "")
        base, _store = _split_store_tag(name)
        key = _norm_key(base)
        if not key:
            continue
        existing_base_keys.add(key)
        existing_by_base.setdefault(key, item)

    added_rows: list[dict[str, Any]] = []
    renamed_rows: list[dict[str, Any]] = []
    state_updates: dict[str, dict[str, Any]] = {}

    store_recs: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        rec = await _recommend_store_for_product(
            data=data,
            product_id=cand.product_id,
            now_local=now,
            observations=observations,
        )
        if rec:
            store_recs[cand.product_id] = rec

    plan = _pick_multi_store_plan(store_recs)
    assignments = plan.get("assignments") or {}

    for cand in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        base_name = cand.name
        base_key = _norm_key(base_name)
        if not base_key:
            continue

        desired_store = assignments.get(cand.product_id)
        rec = store_recs.get(cand.product_id) or {}
        try:
            store_conf = float(rec.get("confidence") or 0.0)
        except Exception:
            store_conf = 0.0

        desired_name = base_name
        if desired_store and store_conf >= STORE_TAG_CONF_THRESHOLD:
            desired_name = f"{base_name} @ {desired_store}"

        existing_item = existing_by_base.get(base_key)
        if existing_item:
            existing_id = str(existing_item.get("id") or "")
            existing_name = str(existing_item.get("name") or "")
            _existing_base, existing_store = _split_store_tag(existing_name)

            if _norm_key(existing_name) == _norm_key(desired_name):
                result["skipped"]["already_present"].append(cand.product_id)
                continue

            should_rename = False
            if desired_name != base_name:
                if existing_store is None:
                    should_rename = True
                else:
                    state = data.storage.get_shopping_product_state(cand.product_id)
                    prev_conf = state.get("last_store_tag_confidence")
                    try:
                        prev_conf_f = float(prev_conf) if prev_conf is not None else 0.0
                    except Exception:
                        prev_conf_f = 0.0
                    if store_conf >= max(STORE_TAG_CONF_THRESHOLD, prev_conf_f + STORE_TAG_RETAG_HYSTERESIS):
                        should_rename = True

            if not should_rename or dry_run:
                result["skipped"]["already_present"].append(cand.product_id)
                continue

            ok = await async_update_item(hass, existing_id, name=desired_name)
            if not ok:
                continue

            renamed_rows.append(
                {
                    "product_id": cand.product_id,
                    "shopping_list_item_id": existing_id,
                    "old_name": existing_name,
                    "new_name": desired_name,
                    "store": desired_store,
                    "store_confidence": round(store_conf, 3),
                    "reason": cand.reason,
                }
            )
            existing_by_base[base_key] = dict(existing_item, name=desired_name)
            state_updates[cand.product_id] = {
                "last_auto_added_at": now.isoformat(),
                "last_store_tag": desired_store,
                "last_store_tag_confidence": round(store_conf, 3),
            }
            continue

        if dry_run:
            added_rows.append(
                {
                    "product_id": cand.product_id,
                    "name": desired_name,
                    "reason": cand.reason,
                    "confidence": round(cand.confidence, 3),
                    "store": desired_store,
                    "store_confidence": round(store_conf, 3) if desired_store else None,
                    "shopping_list_item_id": None,
                }
            )
            continue

        if base_key in existing_base_keys:
            result["skipped"]["already_present"].append(cand.product_id)
            continue

        created = await async_add_item(hass, desired_name)
        if not created or not created.get("id"):
            continue

        item_id = str(created.get("id"))
        existing_base_keys.add(base_key)
        added_rows.append(
            {
                "product_id": cand.product_id,
                "name": desired_name,
                "reason": cand.reason,
                "confidence": round(cand.confidence, 3),
                "store": desired_store,
                "store_confidence": round(store_conf, 3) if desired_store else None,
                "shopping_list_item_id": item_id,
            }
        )
        state_updates[cand.product_id] = {
            "last_auto_added_at": now.isoformat(),
            "last_store_tag": desired_store,
            "last_store_tag_confidence": round(store_conf, 3) if desired_store else None,
        }

    if state_updates:
        await data.storage.async_bulk_update_shopping_product_state(state_updates)

    result["added"] = added_rows
    result["renamed"] = renamed_rows
    result["store_plan"] = {
        "stores": plan.get("stores") or [],
        "primary": plan.get("primary"),
        "secondary": plan.get("secondary"),
        "max_stores": MAX_STORES_PER_RUN,
        "penalty_pct": MULTI_STORE_PENALTY_PCT,
    }
    return result
