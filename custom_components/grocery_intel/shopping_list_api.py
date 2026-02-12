"""Thin wrapper around Home Assistant's built-in shopping list."""
from __future__ import annotations

from typing import Any


def _item_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    result: dict[str, Any] = {}
    for key in ("id", "name", "complete"):
        if hasattr(item, key):
            result[key] = getattr(item, key)
    return result


async def async_get_items(hass) -> list[dict[str, Any]]:
    """Return current shopping list items."""
    try:
        from homeassistant.components import shopping_list as sl  # type: ignore
    except Exception:
        return []
    if not hasattr(sl, "async_get_items"):
        return []
    items = await sl.async_get_items(hass)  # type: ignore[attr-defined]
    return [_item_to_dict(i) for i in (items or [])]


async def async_add_item(hass, name: str) -> dict[str, Any] | None:
    """Add an item and return the created row (including id) when possible."""
    try:
        from homeassistant.components import shopping_list as sl  # type: ignore
    except Exception:
        return None
    if not hasattr(sl, "async_add_item"):
        return None
    item = await sl.async_add_item(hass, name)  # type: ignore[attr-defined]
    return _item_to_dict(item)


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

