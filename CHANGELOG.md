# Changelog

## Unreleased

- Telegram intake: new `grocery_intel.telegram_ingest` service to ingest receipts/inventory images from Telegram attachments (via automation), with optional auto-detect and Telegram feedback.
- Inventory images: vision analysis supports `llm_provider=openai` in addition to Ollama.
- Receipts: improved image preprocessing (best-effort) to increase OCR/vision extraction reliability.
- Receipts: `.webp` support across inbox scanning, OCR upload, and LLM vision parsing.
- Safety: Telegram ingests reject files larger than 25 MB.

## 0.6.0

- Receipt imports: content-based (SHA-256) dedupe (prevents duplicates even if re-uploaded under a different filename).
- Stores: canonical store entity model for more reliable grouping across receipts.
- Extraction pipeline: improved tracking for `pending/queued/running/done/failed`, and richer timing metrics exposed via `sensor.grocery_intel_receipt_processing`.
- Activity sensor: payload compaction + caps to avoid exceeding Home Assistant recorder attribute limits.
- Inventory images: HEIC/HEIF support (best-effort conversion) and EXIF `taken_at` capture time (when available) to improve freshness.
