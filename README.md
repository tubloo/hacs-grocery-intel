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

## Features (V0.1)
- Receipt ingestion service
- Weekly and monthly spend sensors
- Activity log with undo

## Entities
- `sensor.grocery_intel_spend_week`
- `sensor.grocery_intel_spend_month`
- `sensor.grocery_intel_top_price_increases`
- `sensor.grocery_intel_overpaid_items`
- `sensor.grocery_intel_best_store_by_item`

## Services
- `grocery_intel.add_receipt`
- `grocery_intel.undo_activity`

## Configuration
- Add the integration via the Home Assistant UI.
- Optional: set a currency symbol in the integration options.
