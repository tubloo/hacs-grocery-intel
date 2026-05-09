"""Thin wrapper around Home Assistant's built-in shopping list."""
from __future__ import annotations

from typing import Any


def _to_bool_complete(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"completed", "complete", "true", "yes", "1"}
    return bool(value)


def _item_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    result: dict[str, Any] = {}
    for key in ("id", "name", "complete"):
        if hasattr(item, key):
            result[key] = getattr(item, key)
    return result


def _guess_shopping_todo_entity(hass) -> str | None:
    """Best-effort resolution of the shopping-list todo entity."""
    if hass.states.get("todo.shopping_list") is not None:
        return "todo.shopping_list"
    candidates = hass.states.async_all("todo")
    if not candidates:
        return None
    for st in candidates:
        if st.entity_id.endswith("shopping_list"):
            return st.entity_id
    for st in candidates:
        friendly = str(st.attributes.get("friendly_name", "") or "").casefold()
        if "shopping" in friendly:
            return st.entity_id
    return candidates[0].entity_id


def _extract_todo_items(response: Any, entity_id: str) -> list[dict[str, Any]]:
    """Normalize response from todo.get_items across HA versions."""
    if isinstance(response, dict):
        bucket = response.get(entity_id)
        if isinstance(bucket, dict):
            items = bucket.get("items")
            if isinstance(items, list):
                return [dict(i) for i in items if isinstance(i, dict)]
        items = response.get("items")
        if isinstance(items, list):
            return [dict(i) for i in items if isinstance(i, dict)]
    if isinstance(response, list):
        out: list[dict[str, Any]] = []
        for row in response:
            if not isinstance(row, dict):
                continue
            items = row.get("items")
            if isinstance(items, list):
                out.extend(dict(i) for i in items if isinstance(i, dict))
        return out
    return []


async def async_get_items(hass) -> list[dict[str, Any]]:
    """Return current shopping list items."""
    try:
        from homeassistant.components import shopping_list as sl  # type: ignore
    except Exception:
        sl = None  # type: ignore[assignment]
    if sl is not None and hasattr(sl, "async_get_items"):
        try:
            items = await sl.async_get_items(hass)  # type: ignore[attr-defined]
            return [_item_to_dict(i) for i in (items or [])]
        except Exception:
            pass

    entity_id = _guess_shopping_todo_entity(hass)
    if not entity_id:
        return []
    try:
        response = await hass.services.async_call(
            "todo",
            "get_items",
            {"status": "needs_action"},
            blocking=True,
            return_response=True,
            target={"entity_id": entity_id},
        )
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for item in _extract_todo_items(response, entity_id):
        name = str(item.get("summary", "") or "").strip()
        if not name:
            continue
        item_id = item.get("uid") or item.get("id") or name
        out.append(
            {
                "id": str(item_id),
                "name": name,
                "complete": _to_bool_complete(item.get("status")),
            }
        )
    return out


async def async_add_item(hass, name: str) -> dict[str, Any] | None:
    """Add an item and return the created row (including id) when possible."""
    # 1) Try module helper (preferred when available).
    try:
        from homeassistant.components import shopping_list as sl  # type: ignore
        if hasattr(sl, "async_add_item"):
            item = await sl.async_add_item(hass, name)  # type: ignore[attr-defined]
            return _item_to_dict(item)
    except Exception:
        pass

    # 2) Fall back to service call, then resolve created item id by diffing list items.
    before_ids = {str(i.get("id")) for i in await async_get_items(hass) if i.get("id") is not None}
    try:
        await hass.services.async_call(
            "shopping_list",
            "add_item",
            {"name": name},
            blocking=True,
        )
    except Exception:
        return None

    after_items = await async_get_items(hass)
    for row in after_items:
        row_id = row.get("id")
        if row_id is None:
            continue
        if str(row_id) in before_ids:
            continue
        if str(row.get("name", "")).strip() == str(name).strip():
            return row
    return None


async def async_remove_item(hass, item_id: str) -> bool:
    """Remove an item by id."""
    try:
        from homeassistant.components import shopping_list as sl  # type: ignore
    except Exception:
        return False
    if not hasattr(sl, "async_remove_item"):
        return False
    ok = await sl.async_remove_item(hass, item_id)  # type: ignore[attr-defined]
    return bool(ok)


async def async_update_item(hass, item_id: str, *, name: str | None = None, complete: bool | None = None) -> bool:
    """Update an existing shopping list item.

    Best-effort across HA versions:
    - Prefer the shopping_list module helper (async_update_item) when available.
    - Fall back to the shopping_list.update_item service if present.
    """
    if not item_id:
        return False

    # 1) Try module helper (preferred).
    try:
        from homeassistant.components import shopping_list as sl  # type: ignore

        if hasattr(sl, "async_update_item"):
            await sl.async_update_item(hass, item_id, name=name, complete=complete)  # type: ignore[attr-defined]
            return True
    except Exception:
        pass

    # 2) Fall back to service call.
    try:
        service_data: dict[str, Any] = {"item_id": item_id}
        if name is not None:
            service_data["name"] = name
        if complete is not None:
            service_data["complete"] = bool(complete)

        await hass.services.async_call("shopping_list", "update_item", service_data, blocking=True)
        return True
    except Exception:
        return False
