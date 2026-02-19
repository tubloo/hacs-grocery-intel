"""Activity log for Grocery Intel."""
from __future__ import annotations

from typing import Any
import uuid

from homeassistant.core import HomeAssistant
from homeassistant.helpers import storage
from homeassistant.util import dt as dt_util

from .storage import ReceiptStorage

ACTIVITY_STORE_VERSION = 3
ACTIVITY_STORE_KEY = "grocery_intel.activity"

UNDOABLE_KINDS = {
    "receipt_added",
    "receipt_imported_file",
    "ocr_completed",
    "ocr_failed",
    "shopping_list_auto_run",
    "inventory_image_analyzed",
}


class ActivityLog:
    """Persist activity records using Home Assistant storage."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        self._store = storage.Store(hass, ACTIVITY_STORE_VERSION, ACTIVITY_STORE_KEY)
        self._data: dict[str, Any] = {"activities": []}

    async def async_load(self) -> None:
        data = await self._store.async_load()
        if data is None:
            self._data = {"activities": []}
        else:
            self._data = data
            self._data.setdefault("activities", [])

    async def async_save(self) -> None:
        await self._store.async_save(self._data)

    async def async_clear_all(self) -> None:
        """Clear all activity records."""
        self._data = {"activities": []}
        await self.async_save()

    async def async_add_activity(
        self, *, kind: str, description: str, payload: dict[str, Any], save: bool = True
    ) -> dict[str, Any]:
        activity = {
            "activity_id": uuid.uuid4().hex,
            "timestamp": dt_util.now().isoformat(),
            "description": description,
            "kind": kind,
            "payload": payload,
        }
        self._data["activities"].append(activity)
        if save:
            await self.async_save()
        return activity

    async def async_get_activity(self, activity_id: str) -> dict[str, Any] | None:
        for activity in self._data["activities"]:
            if activity.get("activity_id") == activity_id:
                return activity
        return None

    async def async_list_activities(self) -> list[dict[str, Any]]:
        return list(self._data["activities"])

    async def async_undo_activity(
        self, activity_id: str, receipt_storage: ReceiptStorage
    ) -> bool:
        activity = await self.async_get_activity(activity_id)
        if not activity:
            return False

        if activity.get("kind") not in UNDOABLE_KINDS:
            return False

        kind = activity.get("kind")
        if kind not in {"shopping_list_auto_run", "inventory_image_analyzed"}:
            receipt_id = activity.get("payload", {}).get("receipt_id")
            if not receipt_id:
                return False

        if kind in {"receipt_added", "receipt_imported_file"}:
            deleted = await receipt_storage.async_delete_receipt(receipt_id)
            if not deleted:
                return False
        elif kind in {"ocr_completed", "ocr_failed"}:
            cleared = await receipt_storage.async_clear_receipt_ocr(receipt_id)
            if not cleared:
                return False
        elif kind == "shopping_list_auto_run":
            try:
                from .shopping_list_api import async_get_items, async_remove_item, async_update_item
            except Exception:
                return False

            items = await async_get_items(self._hass)
            name_by_id = {str(i.get("id")): i.get("name") for i in (items or [])}

            added = list(activity.get("payload", {}).get("added") or [])
            removed = 0
            for row in added:
                item_id = row.get("shopping_list_item_id")
                if not item_id:
                    continue
                ok = await async_remove_item(self._hass, str(item_id))
                if ok:
                    removed += 1

            renamed = list(activity.get("payload", {}).get("renamed") or [])
            restored = 0
            skipped_renames = 0
            for row in renamed:
                item_id = row.get("shopping_list_item_id")
                old_name = row.get("old_name")
                new_name = row.get("new_name")
                if not item_id or not old_name or not new_name:
                    continue
                current = name_by_id.get(str(item_id))
                if current is not None and str(current).strip() != str(new_name).strip():
                    skipped_renames += 1
                    continue
                ok = await async_update_item(self._hass, str(item_id), name=str(old_name))
                if ok:
                    restored += 1

            await self.async_add_activity(
                kind="shopping_list_undone",
                description=f"Undid auto shopping list run ({removed} removed, {restored} renamed reverted)",
                payload={
                    "activity_id": activity_id,
                    "removed": removed,
                    "restored_renames": restored,
                    "skipped_renames": skipped_renames,
                },
            )
            return True
        elif kind == "inventory_image_analyzed":
            boosts = list(activity.get("payload", {}).get("boosts") or [])
            to_update: dict[str, dict[str, Any]] = {}
            for row in boosts:
                pid = row.get("product_id")
                if not pid:
                    continue
                prev = row.get("previous", {})
                if not isinstance(prev, dict):
                    prev = {}
                to_update[str(pid)] = {
                    "last_seen_at": prev.get("last_seen_at"),
                    "last_seen_confidence": prev.get("last_seen_confidence"),
                }

            if to_update:
                await receipt_storage.async_bulk_update_shopping_product_state(to_update)

            await self.async_add_activity(
                kind="inventory_image_undone",
                description="Undid inventory image analysis",
                payload={"activity_id": activity_id, "restored": len(to_update)},
            )
            return True

        await self.async_add_activity(
            kind="receipt_undone" if kind in {"receipt_added", "receipt_imported_file"} else "ocr_undone",
            description=(
                f"Undid receipt {receipt_id}"
                if kind in {"receipt_added", "receipt_imported_file"}
                else f"Undid OCR for receipt {receipt_id}"
            ),
            payload={"receipt_id": receipt_id, "activity_id": activity_id},
        )
        return True
