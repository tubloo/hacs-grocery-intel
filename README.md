# Grocery Intel

> This project was created with the assistance of AI tooling (e.g., OpenAI Codex/ChatGPT).

Local-first Home Assistant integration for tracking grocery receipts and spend totals.

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

## Entities
- `sensor.grocery_intel_spend_week`
- `sensor.grocery_intel_spend_month`
- `sensor.grocery_intel_top_price_increases`
- `sensor.grocery_intel_overpaid_items`
- `sensor.grocery_intel_best_store_by_item`

## Services
- `grocery_intel.add_receipt`
- `grocery_intel.undo_activity`
- `grocery_intel.scan_receipts_inbox`
- `grocery_intel.run_ocr` (also used to run LLM parsing in `llm` mode)
- `grocery_intel.reparse_receipts`
- `grocery_intel.reprocess_receipts`
- `grocery_intel.clear_all_data`

## Configuration
- Add the integration via the Home Assistant UI.
- Optional: set a currency symbol in the integration options.
- Receipt extraction: use `Extractor mode` (default `heuristic`).
  - `heuristic`: requires an OCR endpoint URL; built-in parsing extracts total/date/store.
  - `llm`: parses receipts via an LLM.
    - Images (`.jpg/.png`) are sent directly to Ollama **vision** models (e.g., `llava:7b`) when `llm_provider=ollama`.
    - PDFs are parsed from their text layer (via `pypdf`); if the PDF has no text layer, the first page is rendered to an image (via `PyMuPDF`) and sent to Ollama vision.
    - The integration asks the LLM for `total`, `store_name`, `purchased_at`, and `line_items` (when possible).
  - `hybrid`: uses OCR + heuristics first, then LLM to fill missing fields (and attempt line item extraction).
- Optional: `LLM extra instructions` lets you add fine-tuning instructions; the integration always enforces a JSON-only contract and appends your instructions.
- Tip (Home Assistant in Docker): `.local` hostnames may not resolve; prefer an IP like `http://192.168.x.x:11434` for `LLM base URL`.
- Defaults (Options): inbox `/config/www/receipts_inbox`, archive `/config/www/receipts_archive` (already-processed files are archived with a `_duplicate` suffix).
