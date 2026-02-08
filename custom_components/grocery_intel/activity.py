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
}


class ActivityLog:
    """Persist activity records using Home Assistant storage."""

    def __init__(self, hass: HomeAssistant) -> None:
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
        self, *, kind: str, description: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        activity = {
            "activity_id": uuid.uuid4().hex,
            "timestamp": dt_util.now().isoformat(),
            "description": description,
            "kind": kind,
            "payload": payload,
        }
        self._data["activities"].append(activity)
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

        receipt_id = activity.get("payload", {}).get("receipt_id")
        if not receipt_id:
            return False

        kind = activity.get("kind")
        if kind in {"receipt_added", "receipt_imported_file"}:
            deleted = await receipt_storage.async_delete_receipt(receipt_id)
            if not deleted:
                return False
        elif kind in {"ocr_completed", "ocr_failed"}:
            cleared = await receipt_storage.async_clear_receipt_ocr(receipt_id)
            if not cleared:
                return False

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
