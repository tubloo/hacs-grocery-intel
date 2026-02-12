# Grocery Intel — Current Context

This file describes the **current state of the system as implemented**.
It is the primary handoff document for Codex and future work.

---

## What exists today

### 1. Receipt ingestion

Receipts can be ingested via:
- Manual service calls
- File inbox scanning

#### File inbox
- Files are manually copied into:
  `/config/www/receipts_inbox`
- Supported formats:
  - PDF
  - JPG / JPEG
  - PNG
  - WEBP
  - HEIC / HEIF
- The integration:
  - Detects new files
  - Creates a receipt record referencing the file
  - Logs an activity entry
  - Moves the file to:
    `/config/www/receipts_archive`
- Processed files are tracked to prevent re-import.
  - Receipt dedupe is content-based (SHA-256), so re-uploading the same receipt under a different filename will not create duplicates.
  - Already-processed duplicates are archived with a `_duplicate` suffix.

---

### 2. Extraction pipeline

Receipts go through an **extraction step** that may use:
- Heuristics
- LLMs (Ollama and OpenAI, including vision)
- Hybrid approaches

Key properties:
- Controlled by an `extractor_mode`
- Receipt `extract_status` values:
  - `pending` / `queued` / `running` / `done` / `failed`
- PDFs:
  - Try text-layer extraction first
  - Fallback to rendering and vision if needed
- Images:
  - Sent directly to vision models
- Prompts enforce structured (JSON) output
- Extraction may produce:
  - total
  - date
  - store
  - currency
  - line_items (if available)
- Missing archived files are treated as failures (auto-marked `failed`) to avoid retry loops.

Extraction failure does NOT invalidate the receipt.

---

### 3. OCR / LLM infrastructure

- Heavy extraction runs **outside Home Assistant**
- Typically via a local Ollama instance
- HA integration orchestrates calls and stores results
- No cloud dependency is required

---

### 4. Receipt editing & reprocessing

A service exists to:
- Manually update receipt fields
- Replace or clear line items
- Trigger reprocessing of derived data

This is a core trust feature.

---

### 5. Analytics & sensors

The integration exposes:
- Weekly spend
- Monthly spend
- Rolling spend metrics (e.g., 7d / 30d)
- Receipt counts and averages
- Receipt extraction status counts + timing metrics (`sensor.grocery_intel_receipt_processing`)
- List-style insights via sensor attributes
  (e.g., recent receipts, top stores)

Analytics automatically reprocess when receipt data changes.

---

### 6. Activity log & undo

- Every automatic action creates an activity entry
- Activities include:
  - Receipt added
  - File imported
  - Extraction completed / failed
  - Manual updates
- Undo is supported where logically possible
- Undo never crashes or corrupts state
- The `Recent activities` sensor caps and compacts payloads to stay under Home Assistant's recorder attribute size limits.

---

## Automated shopping list (v1)

The integration can auto-add items to Home Assistant's default Shopping List.

- Runs daily at **07:00 local time**
- Uses probabilistic **inventory levels** derived from purchase cadence:
  - `plenty` / `medium` / `likely_low`
- Auto-add eligibility (defaults):
  - Minimum purchase history: **3 purchases**
  - Cooldown after auto-add: **7 days** (configurable)
  - Only auto-add when confidence is above a configurable threshold
- Never auto-removes items; user clears the list manually
- Duplicate prevention: skips adding if the item is already on the list
- Every run is logged as an activity, and undo removes items that were added by that run

---

## Inventory images (v1)

Users can drop fridge/pantry/cupboard photos into an inbox folder.
The integration can analyze these with a vision-capable LLM to improve inventory confidence.

- Inbox → archive flow (separate from receipts):
  - Default inbox: `/config/www/inventory_images_inbox`
  - Default archive: `/config/www/inventory_images_archive`
  - Duplicates are detected by file fingerprint and archived with a `_duplicate` suffix
- Freshness:
  - When available, the integration stores `taken_at` (EXIF capture time) for the image.
  - Otherwise it falls back to import time (`created_at`).
- Vision analysis:
  - Uses existing LLM settings (Ollama vision for images)
  - Extracts a list of generic detected items (`milk`, `eggs`, etc.)
  - Normalizes to products (creates new products when needed)
  - Applies **boost-only evidence** by setting `last_seen_at` on products (no negative inference)
  - Evidence suppresses auto-add suggestions for a configurable TTL
- Every import/analyze action is logged; undo restores previous evidence state

---

## What is stable vs fluid

Stable:
- Inbox → archive flow
- Receipt model
- Activity + undo semantics
- Spend sensors

Fluid / evolving:
- Extraction quality
- Prompt design
- Line-item normalization
- Advanced analytics

---

## Active development focus

- Improving extraction reliability
- Making edits + reprocessing safer and clearer
- Incrementally improving analytics without adding entities

No major architectural rewrites are planned.
