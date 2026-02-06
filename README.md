# Grocery Intel

Local-first Home Assistant integration for tracking grocery receipts and spend totals.

## Features (V0.1)
- Receipt ingestion service
- Weekly and monthly spend sensors
- Activity log with undo

## Entities
- `sensor.grocery_spend_week`
- `sensor.grocery_spend_month`

## Services
- `grocery_intel.add_receipt`
- `grocery_intel.undo_activity`

## Configuration
- Add the integration via the Home Assistant UI.
- Optional: set a currency symbol in the integration options.
