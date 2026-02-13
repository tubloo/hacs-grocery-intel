# Grocery Intel

> This project was created with the assistance of AI tooling (e.g., OpenAI Codex/ChatGPT).

Local-first Home Assistant integration for tracking grocery receipts and spend totals.

Grocery Intel is a local-first Home Assistant integration that turns grocery receipts and pantry/fridge photos into practical household insights. It automatically ingests receipts, extracts totals, dates, stores, and (when available) line items using OCR and/or LLMs, then builds analytics like spend trends, store comparisons, price changes, and a canonical list of the items you typically buy. For day-to-day usefulness, it maintains a probabilistic “inventory confidence” (not exact counts) and can suggest shopping list items based on purchase cadence and low-confidence signals. Every automated action is logged, explainable, and undoable, and all data is stored locally in Home Assistant.

## Privacy & Data Handling

- Grocery Intel stores receipts and derived data locally in your Home Assistant storage (e.g., under `/config/.storage/`).
- If you configure OCR, your receipt images/text may be sent to the OCR endpoint you specify. Choose an endpoint you trust.

## Disclaimer

- This is an unofficial, community integration and is not affiliated with Home Assistant / Nabu Casa.
- Receipt parsing/OCR and totals can be incomplete or incorrect. Use this as a convenience, not as financial or accounting advice.
- You are responsible for securing your Home Assistant instance and any external services you configure (like OCR).

## Installation

### HACS (Custom Repository)
1. HACS → three-dot menu → **Custom repositories**
2. Add this repo URL and select **Integration**
3. Install **Grocery Intel** and restart Home Assistant
4. Add the integration in **Settings → Devices & Services**

### Manual (advanced)
1. Copy `custom_components/grocery_intel` into your HA config:
   - `/config/custom_components/grocery_intel`
2. Restart Home Assistant
3. Add the integration in **Settings → Devices & Services**

## Features
- Receipt inbox ingestion (PDF/images)
- Spend sensors (weekly/monthly) + basic price analytics
- OCR and/or LLM-based field extraction (configurable)
- Activity log with undo for supported actions
- Daily auto-add to Home Assistant Shopping List (optional)
- Inventory images (fridge/pantry/cupboard) inbox + vision analysis (optional)
- Alcohol item normalization (Beer/Wine/Cider/Spirits)

## What you can analyze
- Spend trends: weekly/monthly totals, rolling 7/30 days, average basket size, receipt counts
- Store insights: top stores by spend and spend split across stores (within the selected window)
- Receipt timeline: recent receipts (store/date/total/filename) to spot gaps and spikes
- Item-level analytics (when line items are available): price history, top increases, overpaid items, best store by item
- Purchase cadence: how frequently items appear across receipts (useful for shopping list suggestions)
- Inventory freshness: items “recently seen” via pantry/fridge images (boost-only evidence; no exact counts)
- Pipeline health: receipt processing status counts and timing breakdowns (overall/by method/provider)

## Data model (high level)
Grocery Intel stores its richer data in Home Assistant storage (`/config/.storage/grocery_intel.data`). Sensors are summaries over that data.

- Receipts: one row per receipt (source file, `purchased_at`, `total`, `store_name`, `extract_status`, timing fields, optional `content_hash`)
- Stores: canonical store entities (`store_entity_id`) used to group receipts even when names vary
- Line items: raw line items per receipt (when extraction provides them)
- Products: normalized/canonical products created from line items and inventory-image detections
- Observations: per-product purchase “events” derived from line items (used for price/pattern analytics)
- Inventory images: inbox-imported pantry/fridge/cupboard photos with `taken_at` (EXIF, when available), analysis status, and detected items
- Processed file indexes: fingerprints/hashes used to avoid importing the same file twice
- Activity log: auditable history of automatic/manual actions (imports, duplicates, extraction done/failed, auto-shopping runs, image analysis) with undo where possible

## Entities
- `sensor.grocery_intel_spend_week`
- `sensor.grocery_intel_spend_month`
- `sensor.grocery_intel_spend_7d`
- `sensor.grocery_intel_spend_30d`
- `sensor.grocery_intel_avg_basket_30d`
- `sensor.grocery_intel_receipt_count_30d`
- `sensor.grocery_intel_receipt_processing`
- `sensor.grocery_intel_top_stores_30d`
- `sensor.grocery_intel_recent_receipts`
- `sensor.grocery_intel_recent_activities`
- `sensor.grocery_intel_inventory_recently_seen`
- `sensor.grocery_intel_top_price_increases`
- `sensor.grocery_intel_overpaid_items`
- `sensor.grocery_intel_best_store_by_item`

## Services
- `grocery_intel.add_receipt`
- `grocery_intel.update_receipt`
- `grocery_intel.undo_activity`
- `grocery_intel.scan_receipts_inbox`
- `grocery_intel.run_ocr` (also used to run LLM parsing in `llm` mode)
- `grocery_intel.reparse_receipts`
- `grocery_intel.reprocess_receipts`
- `grocery_intel.clear_all_data`
- `grocery_intel.run_auto_shopping` (manual trigger for the daily job; supports `dry_run`)
- `grocery_intel.scan_inventory_images_inbox`
- `grocery_intel.run_inventory_vision`
- `grocery_intel.reset_stuck_receipts`
- `grocery_intel.telegram_ingest`

## Configuration
- Add the integration via the Home Assistant UI.
- Optional: set a currency symbol in the integration options.
- Receipt extraction: use `Extractor mode` (default `heuristic`).
  - `heuristic`: requires an OCR endpoint URL; built-in parsing extracts total/date/store.
  - `llm`: parses receipts via an LLM.
    - Images (`.jpg/.png/.webp/.heic/.heif`) are sent to a vision-capable LLM when `llm_provider=ollama` or `llm_provider=openai`.
    - PDFs are parsed from their text layer (via `pypdf`). If the PDF has no text layer, you'll need to OCR/convert it outside Home Assistant (the integration avoids heavy native dependencies).
    - The integration asks the LLM for `total`, `store_name`, `purchased_at`, and `line_items` (best-effort). For images/PDF-vision it will do a second “line items only” pass to improve extraction.
  - `hybrid`: uses OCR + heuristics first, then LLM to fill missing fields (and attempt line item extraction).
- Optional: `LLM extra instructions` lets you add fine-tuning instructions; the integration always enforces a JSON-only contract and appends your instructions.
- Tip (Home Assistant in Docker): `.local` hostnames may not resolve; prefer an IP like `http://192.168.x.x:11434` for `LLM base URL`.
- Defaults (Options): inbox `/config/www/receipts_inbox`, archive `/config/www/receipts_archive`.
  - Receipt dedupe is content-based (SHA-256), so re-uploading the same receipt under a different filename will not create duplicates.
  - Already-processed duplicates are archived with a `_duplicate` suffix.
- Archive retention: archived receipt files are deleted after `Archive retention (days)` (default 30 days, configurable 1–90).
- Receipt processing status: see `sensor.grocery_intel_receipt_processing` (includes `status_counts` and `timing` in attributes).
- Shopping list automation (Options):
  - `Auto-add shopping items (daily)`: runs at 07:00 local time and adds items when inventory is likely low
  - `Auto-add cooldown (days)`: default 7
  - `Auto-add confidence threshold`: default 0.75
  - `Pause auto-add when all people away ≥48h`: optional
- Inventory images (Options):
  - Upload images to `Inventory images inbox path` (default `/config/www/inventory_images_inbox`)
  - The integration archives them to `Inventory images archive path` and analyzes them (vision requires a vision-capable LLM; supported providers include `ollama` and `openai`)
  - When available, the integration stores `taken_at` (EXIF capture time) for freshness; otherwise it falls back to import time (`created_at`)
  - Evidence boosts inventory (no exact counts) and suppresses auto-add for `Inventory evidence TTL (days)`

### Telegram intake (optional)
You can ingest receipts and inventory images from Telegram by calling `grocery_intel.telegram_ingest` from a Home Assistant automation triggered by incoming Telegram messages.

- Configure options:
  - `Telegram bot token`: required (used to download files and send feedback)
  - `Telegram allowed chat IDs`: recommended for security (comma-separated allowlist)
  - `Telegram auto-detect receipt vs inventory`: when enabled, PDFs default to receipts; images use caption keywords first (e.g., `receipt`, `inventory`, `fridge`, `pantry`) and may use your configured LLM (OpenAI/Ollama vision) to classify when available
  - `Telegram send analysis feedback`: replies in Telegram when queued and when analysis completes/fails (timestamps are formatted in Home Assistant local time)
  - Limits:
    - Telegram ingests reject files larger than **25 MB** (to avoid large in-memory downloads inside Home Assistant).

#### Example automation (Telegram attachment → Grocery Intel)
Telegram uploads typically arrive as the `telegram_attachment` event. Use **Developer Tools → Events** to listen for `telegram_attachment` and confirm the exact payload fields in your HA instance, then adapt the templates below if needed.

```yaml
alias: Grocery Intel - Telegram ingest
mode: queued
trigger:
  - platform: event
    event_type: telegram_attachment
action:
  - service: grocery_intel.telegram_ingest
    data:
      chat_id: "{{ trigger.event.data.chat_id }}"
      message_id: "{{ trigger.event.data.id | default(none) }}"
      file_id: "{{ trigger.event.data.get('attachment', {}).get('file_id', '') }}"
      filename: >-
        {{ trigger.event.data.get('attachment', {}).get('file_name')
           or trigger.event.data.get('attachment', {}).get('file_unique_id')
           or '' }}
      caption: "{{ trigger.event.data.get('caption', '') }}"
      kind: auto  # or 'receipt' / 'inventory'
```

## Dashboard (example)
Example Lovelace cards (YAML mode):

```yaml
type: entities
title: Grocery Intel
entities:
  - entity: sensor.grocery_intel_spend_7d
  - entity: sensor.grocery_intel_spend_30d
  - entity: sensor.grocery_intel_avg_basket_30d
  - entity: sensor.grocery_intel_receipt_count_30d
  - entity: sensor.grocery_intel_receipt_processing
  - entity: sensor.grocery_intel_top_stores_30d
  - entity: sensor.grocery_intel_recent_receipts
  - entity: sensor.grocery_intel_top_price_increases
  - entity: sensor.grocery_intel_overpaid_items
  - entity: sensor.grocery_intel_best_store_by_item
```

The list-style sensors store their details in attributes under `items` (shown in **Developer Tools → States**).
