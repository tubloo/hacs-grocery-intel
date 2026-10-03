"""Receipt quota regressions, runnable without a Home Assistant installation."""
import ast
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock


# Load the actual helpers without importing Home Assistant's integration runtime.
SOURCE = Path(__file__).resolve().parents[1] / "custom_components/grocery_intel/__init__.py"
NAMES = {
    "_receipt_should_auto_extract", "_is_openai_quota_error",
    "_friendly_llm_failure_reason", "_async_mark_extract_failed",
    "_async_ingest_receipt_bytes",
}
tree = ast.parse(SOURCE.read_text())
module = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    *(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
      and node.name in NAMES),
], type_ignores=[])
namespace = {}
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


class QuotaRetryTests(unittest.IsolatedAsyncioTestCase):
    def test_quota_codes_and_temporary_throttling(self):
        for code in (
            "insufficient_quota", "credit_balance_exhausted",
            "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
            "organization_usage_limit_exceeded",
        ):
            error = RuntimeError(f'OpenAI HTTP 429: {{"error": {{"code": "{code}"}}}}')
            self.assertTrue(namespace["_is_openai_quota_error"](error))
            reason = namespace["_friendly_llm_failure_reason"](error)
            self.assertIn("automatic retries are paused", reason)
            self.assertIn("send the same receipt again in Telegram", reason)
        for body in ('{"code": "rate_limit_exceeded"}', '{"code": "slow_down"}'):
            error = RuntimeError(f"OpenAI HTTP 429: {body}")
            self.assertFalse(namespace["_is_openai_quota_error"](error))
            self.assertIn("rate limit", namespace["_friendly_llm_failure_reason"](error))

    async def test_quota_pause_is_saved_logged_and_notified_once(self):
        receipt = {"id": "receipt-1", "filename": "receipt.jpg", "file_path": "/media/receipt.jpg",
                   "extract_status": "running", "extract_attempts": 2}
        async def update(_receipt_id, updates):
            receipt.update(updates)
        data = SimpleNamespace(
            storage=SimpleNamespace(async_get_receipt=AsyncMock(return_value=receipt),
                                    async_update_receipt=AsyncMock(side_effect=update)),
            activity=SimpleNamespace(async_add_activity=AsyncMock()),
            request_refresh=Mock(),
        )
        notify = AsyncMock()
        namespace.update({
            "dt_util": SimpleNamespace(now=lambda: datetime.now(timezone.utc)),
            "_safe_str": lambda value: str(value) if value else None,
            "_ms_between_iso": lambda *_args: None,
            "_apply_auto_receipt_category": Mock(),
            "_apply_auto_receipt_subcategories": Mock(),
            "_async_maybe_notify_telegram_receipt": notify,
        })
        await namespace["_async_mark_extract_failed"](data, receipt, "Quota exhausted", auto_retry=False)
        self.assertEqual(receipt["extract_status"], "failed")
        self.assertEqual(receipt["extract_attempts"], 3)
        self.assertFalse(receipt["extract_auto_retry"])
        data.activity.async_add_activity.assert_awaited_once()
        notify.assert_awaited_once()
        # Repeated scans (including after a restart) must leave this receipt alone.
        for _ in range(3):
            self.assertFalse(namespace["_receipt_should_auto_extract"](dict(receipt)))
        # A manual attempt that fails transiently restores normal retry behavior.
        await namespace["_async_mark_extract_failed"](data, receipt, "Timed out")
        self.assertTrue(namespace["_receipt_should_auto_extract"](receipt))

    def test_legacy_receipts_and_explicit_pending_reset(self):
        eligible = namespace["_receipt_should_auto_extract"]
        self.assertTrue(eligible({"file_path": "receipt.jpg", "extract_status": "failed"}))
        self.assertTrue(eligible({"file_path": "receipt.jpg", "extract_status": "pending",
                                  "extract_auto_retry": False}))
        for status in ("done", "running", "queued"):
            self.assertFalse(eligible({"file_path": "receipt.jpg", "extract_status": status}))
        self.assertFalse(eligible({"extract_status": "failed"}))

    async def test_telegram_reupload_retries_existing_failed_receipt(self):
        for status, chat_id, should_retry in (
            ("failed", 42, True), ("done", 42, False),
            ("running", 42, False), ("queued", 42, False),
            ("failed", 99, False),
        ):
            with self.subTest(status=status, chat_id=chat_id):
                content = b"same receipt image"
                receipt = {
                    "id": "existing-receipt", "content_hash": hashlib.sha256(content).hexdigest(),
                    "extract_status": status, "extract_auto_retry": False,
                    "source_type": "telegram", "source_meta": {"chat_id": 42, "message_id": 1},
                    "file_path": "/media/expired-original.jpg",
                }
                async def update(_receipt_id, updates):
                    receipt.update(updates)
                data = SimpleNamespace(
                    storage=SimpleNamespace(
                        async_get_processed_fingerprints=AsyncMock(return_value=set()),
                        async_get_receipt_content_hash_fingerprints=AsyncMock(
                            return_value={f'sha256:{receipt["content_hash"]}'}),
                        async_list_receipts=AsyncMock(return_value=[receipt]),
                        async_update_receipt=AsyncMock(side_effect=update),
                        async_add_receipt=AsyncMock(),
                    ),
                    activity=SimpleNamespace(async_add_activity=AsyncMock()),
                    request_refresh=Mock(),
                )
                # Close the scheduled coroutine without running network extraction.
                hass = SimpleNamespace(
                    config=SimpleNamespace(is_allowed_path=lambda _path: True),
                    async_add_executor_job=AsyncMock(),
                    async_create_task=Mock(side_effect=lambda coroutine: coroutine.close()),
                )
                namespace.update({
                    "hashlib": hashlib,
                    "os": SimpleNamespace(path=os.path, makedirs=Mock()),
                    "time": SimpleNamespace(time=lambda: 0),
                    "dt_util": SimpleNamespace(now=lambda: datetime.now(timezone.utc)),
                    "CONF_RECEIPTS_ARCHIVE_PATH": "archive",
                    "DEFAULT_RECEIPTS_ARCHIVE_PATH": "/media/archive",
                    "_sanitize_archive_filename": lambda value, **_kwargs: value,
                    "_unique_dest_path": lambda *_args, **_kwargs: "/media/archive/receipt_duplicate.jpg",
                    "_write_bytes_sync": Mock(),
                    "_coerce_int": lambda value: int(value) if value is not None else None,
                    "_async_run_llm_for_receipt_file": AsyncMock(),
                })
                result = await namespace["_async_ingest_receipt_bytes"](
                    hass, entry=SimpleNamespace(options={}), data=data,
                    filename="receipt.jpg", content=content,
                    source_meta={"chat_id": chat_id, "message_id": 2},
                )
                data.storage.async_add_receipt.assert_not_awaited()
                if should_retry:
                    self.assertEqual(result, ("existing-receipt", "/media/archive/receipt_duplicate.jpg", False))
                    self.assertEqual(receipt["extract_status"], "queued")
                    self.assertEqual(receipt["source_meta"]["message_id"], 2)
                    self.assertEqual(receipt["file_path"], result[1])
                    hass.async_create_task.assert_called_once()
                    kinds = [call.kwargs["kind"] for call in data.activity.async_add_activity.await_args_list]
                    self.assertIn("receipt_extraction_retry_requested", kinds)
                else:
                    self.assertTrue(result[2])
                    data.storage.async_update_receipt.assert_not_awaited()
                    hass.async_create_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
