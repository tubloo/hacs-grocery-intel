# Changelog

## Unreleased

## 1.0.19 (2026-03-17)

- Analytics: added `sensor.grocery_intel_spend_by_month_12m` with trailing 12-month receipt-date buckets in `items` (`month`, `month_start`, `grocery_total`, `grocery_ex_vice`, `vice_total`, `dining_total`, `total`).
- Consistency: reuse a shared vice-subcategory calculator in period totals and monthly analytics.
- Docs: updated `README.md` entities/sensor descriptions and dashboard example; updated `AGENTS.md` analytics summary.

## 1.0.18 (2026-03-16)

- New period sensors for category-focused dashboards:
  - `sensor.grocery_intel_grocery_week|month|ytd`
  - `sensor.grocery_intel_dining_week|month|ytd`
  - `sensor.grocery_intel_vice_week|month|ytd`
- `Vice` is defined as alcohol subcategories plus `tobacco_nicotine` on grocery receipts.
- Docs: updated `README.md` entity/sensor descriptions for the new period sensors.

## 1.0.17 (2026-03-16)

- Sensors: `sensor.grocery_intel_recent_receipts` now includes `receipt_category` and `receipt_category_source` in each `items` row for dashboard accuracy.
- Docs: updated `README.md` to document the expanded `recent_receipts` payload.
- Agent docs: added explicit SemVer-based release numbering strategy to `AGENTS.md`.

## 1.0.16 (2026-03-16)

- Receipt categories: renamed `eating_out` to `dining` across services/docs/sensors, with storage migration for legacy values.
- Prompting/options: removed `Eating-out keywords`; kept customization via `Receipt category LLM prompt`.
- Subcategories: expanded grocery subcategory taxonomy (e.g., `baby_child`, `pet_care`, `pharmacy_health`, `ready_meals`, `fees_deposits`, `tobacco_nicotine`, `alcohol_*`) and split `dairy_eggs` into `dairy` + `eggs` (legacy alias kept).
- Extraction prompts: improved receipt schema guidance and locale-aware context (HA language/country/currency/timezone + local chain hints) for text and vision receipt parsing.
- Inventory vision: added locale context guidance to inventory-image prompting.
- Analytics: grocery subcategory 30d sensor now relies on persisted receipt subcategories (legacy line-item fallback removed).

## 1.0.11 (2026-03-14)

- Security/safety: sanitize ingest/archive filenames so Telegram or upload filename overrides cannot inject path components.
- Undo: `undo_activity` for auto-shopping runs now restores per-product shopping state metadata (`last_auto_added_at`, store-tag fields), not only shopping-list item edits.
- Sensors: spend totals now skip non-numeric totals safely instead of risking conversion errors.
- Export: date-only `until` filters now include the full local day (end-of-day semantics).

## 1.0.10 (2026-02-19)

- Fix: ensure sensors refresh even if the debounced refresh callback is missed (failsafe refresh, no restart needed).

## 1.0.9 (2026-02-19)

- Shopping list: auto-added items can include a recommended store suffix (e.g., `Eggs @ Willys`) based on price history, and may rename existing untagged items to include the recommendation.

## 1.0.8 (2026-02-19)

- Fix: prevent analytics/sensor refresh from getting stuck indefinitely after a missed debounced refresh callback.
- New: `grocery_intel.force_refresh` service to force an immediate refresh when entities appear stale.

## 1.0.7 (2026-02-18)

- Perf: debounce analytics refreshes to avoid repeated recomputation during bursty updates.
- Perf: batch storage/activity saves during receipts/inventory inbox scans to reduce repeated full-store writes.

## 1.0.6 (2026-02-15)

- Fix: prevent duplicate receipts when the same Telegram receipt is ingested as both a PDF and an image (semantic de-dupe after extraction).

## 1.0.5 (2026-02-15)

- Fix: `grocery_intel.dedupe_stores` dry-run notification now includes projected store counts after orphan deletion.

## 1.0.4 (2026-02-15)

- New: `grocery_intel.dedupe_stores` service to merge duplicate store entities and update receipts (dry-run by default).

## 1.0.3 (2026-02-15)

- Fix: prevent duplicate store entities when receipts only contain a store name (no merchant hints/location).
- Docs: clarify store matching behavior and how to consolidate older duplicates.

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
