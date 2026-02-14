# Changelog

## Unreleased

## 1.0.2 (2026-02-14)

- Docs: dashboard YAML example now includes `sensor.grocery_intel_spend_ytd` and other key entities.

## 1.0.1 (2026-02-14)

- Fix: `scan_receipts_inbox` now creates the receipts inbox folder if it doesn't exist (important for fresh installs using `/media`).
- Docs: clarify that the 30-day receipt count sensor may appear as `sensor.grocery_intel_receipts_30d` on older installs.

## 1.0.0 (2026-02-14)

- Analytics: new `sensor.grocery_intel_spend_ytd` (calendar year-to-date spend).

## 0.7.0 (2026-02-14)

- Telegram intake: new `grocery_intel.telegram_ingest` service to ingest receipts/inventory images from Telegram attachments (via automation), with optional auto-detect and Telegram feedback.
- Export: new `grocery_intel.export_data` service to export analyzed data to JSON under the configured exports folder.
- Inventory images: vision analysis supports `llm_provider=openai` in addition to Ollama.
- Receipts: improved image preprocessing (best-effort) to increase OCR/vision extraction reliability.
- Receipts: `.webp` support across inbox scanning, OCR upload, and LLM vision parsing.
- Safety: Telegram ingests reject files larger than 25 MB.
- Defaults: inbox/archive paths now default to `/media/grocery_intel/...` for better privacy (note: container installs must mount a host folder to `/media`).

## 0.6.0

- Receipt imports: content-based (SHA-256) dedupe (prevents duplicates even if re-uploaded under a different filename).
- Stores: canonical store entity model for more reliable grouping across receipts.
- Extraction pipeline: improved tracking for `pending/queued/running/done/failed`, and richer timing metrics exposed via `sensor.grocery_intel_receipt_processing`.
- Activity sensor: payload compaction + caps to avoid exceeding Home Assistant recorder attribute limits.
- Inventory images: HEIC/HEIF support (best-effort conversion) and EXIF `taken_at` capture time (when available) to improve freshness.
