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
- Stores: canonical store entities (`store_entity_id`) used to group receipts even when names vary.
  - Matching prefers strong merchant hints when available (org/store ID/phone/address/postal/city).
  - If a receipt only yields a store/chain name (no hints), Grocery Intel reuses an existing matching store entity (by normalized `chain_name`/aliases) to avoid creating many empty duplicates.
  - Note: if you shop at multiple branches of the same chain and your receipts don’t include location/IDs, those branches may be grouped together until richer hints are available.
- Line items: raw line items per receipt (when extraction provides them)
- Products: normalized/canonical products created from line items and inventory-image detections
- Observations: per-product purchase “events” derived from line items (used for price/pattern analytics)
- Inventory images: inbox-imported pantry/fridge/cupboard photos with `taken_at` (EXIF, when available), analysis status, and detected items
- Processed file indexes: fingerprints/hashes used to avoid importing the same file twice
- Activity log: auditable history of automatic/manual actions (imports, duplicates, extraction done/failed, auto-shopping runs, image analysis) with undo where possible

## Entities
- `sensor.grocery_intel_spend_week`
- `sensor.grocery_intel_spend_month`
- `sensor.grocery_intel_spend_ytd`
- `sensor.grocery_intel_spend_7d`
- `sensor.grocery_intel_spend_30d`
- `sensor.grocery_intel_avg_basket_30d`
- `sensor.grocery_intel_receipt_count_30d` (may appear as `sensor.grocery_intel_receipts_30d` on older installs)
- `sensor.grocery_intel_receipt_processing`
- `sensor.grocery_intel_top_stores_30d`
- `sensor.grocery_intel_recent_receipts`
- `sensor.grocery_intel_recent_activities`
- `sensor.grocery_intel_inventory_recently_seen`
- `sensor.grocery_intel_top_price_increases`
- `sensor.grocery_intel_overpaid_items`
- `sensor.grocery_intel_best_store_by_item`

### Sensors explained
- `sensor.grocery_intel_spend_week`: current ISO-week spend total.
- `sensor.grocery_intel_spend_month`: current calendar-month spend total.
- `sensor.grocery_intel_spend_ytd`: calendar year-to-date spend total (based on `purchased_at` in HA local time).
- `sensor.grocery_intel_spend_7d`: rolling 7-day spend total.
- `sensor.grocery_intel_spend_30d`: rolling 30-day spend total.
- `sensor.grocery_intel_avg_basket_30d`: average receipt total over the last 30 days.
- `sensor.grocery_intel_receipt_count_30d`: receipt count over the last 30 days (entity_id may be `sensor.grocery_intel_receipts_30d` if it was created before the suggested id changed; use the one shown in your HA Entities list).
- `sensor.grocery_intel_receipt_processing`: pipeline health; state is the number of receipts in `pending+queued+running`, and attributes include `status_counts` and `timing` summaries (avg/median/p95 by method/provider).

List-style sensors: the state is a count, and details are in the `items` attribute.
- `sensor.grocery_intel_top_stores_30d`: top stores by spend (last 30 days, up to 10).
- `sensor.grocery_intel_recent_receipts`: latest receipts (up to 20: `receipt_id`, `purchased_at`, `store_name`, `total`, `filename`).
- `sensor.grocery_intel_recent_activities`: recent activity log (up to 25, payload is compacted to fit HA attribute limits).
- `sensor.grocery_intel_inventory_recently_seen`: inventory evidence from vision (up to 100: `product`, `last_seen_at`, `expires_at`, `confidence`).
- `sensor.grocery_intel_top_price_increases`: largest increases by median unit price (up to 10; requires line-item observations).
- `sensor.grocery_intel_overpaid_items`: “overpaid vs baseline” items (up to 10; requires line-item observations).
- `sensor.grocery_intel_best_store_by_item`: best store for an item by median unit price (up to 10; requires enough history).

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
- `grocery_intel.export_data`
- `grocery_intel.dedupe_stores` (dry-run by default; merges duplicate store entities and updates receipts)

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
- Recommended (privacy): use `/media` paths so receipts/photos are protected by Home Assistant authentication.
  - **Home Assistant OS / Supervised:** `/media` is available by default.
  - **Home Assistant Container:** you must mount a host folder into the container at `/media` (e.g., in docker-compose: `./media:/media`).
  - If `/media` isn’t available for your install, you can change the inbox/archive paths in the integration options.
- Defaults (Options): inbox `/media/grocery_intel/receipts_inbox`, archive `/media/grocery_intel/receipts_archive`.
  - Receipt dedupe is content-based (SHA-256), so re-uploading the same receipt under a different filename will not create duplicates.
  - Already-processed duplicates are archived with a `_duplicate` suffix.
  - Telegram semantic dedupe (PDF + image): if the same receipt is ingested as both a PDF and an image with matching date/store/total, the integration will keep one receipt record and remove the duplicate automatically (file remains archived).
- Store grouping:
  - Receipts are grouped by a canonical `store_entity_id`, derived from extracted merchant hints when available.
  - If your extraction only produces a store name (no merchant IDs/location), Grocery Intel falls back to name-based matching to prevent store duplication. This can group multiple branches under one entity if the receipt lacks branch/location details.
  - Existing duplicates (from older versions): updating will stop new duplicate store entities from being created. To consolidate historical receipts, run `grocery_intel.dedupe_stores` (start with `dry_run: true`) or re-save `store_name` via `grocery_intel.update_receipt`.
- Archive retention: archived receipt files are deleted after `Archive retention (days)` (default 30 days, configurable 1–90).
- Receipt processing status: see `sensor.grocery_intel_receipt_processing` (includes `status_counts` and `timing` in attributes).
- Shopping list automation (Options):
  - `Auto-add shopping items (daily)`: runs at 07:00 local time and adds items when inventory is likely low
  - `Auto-add cooldown (days)`: default 7
  - `Auto-add confidence threshold`: default 0.75
  - `Pause auto-add when all people away ≥48h`: optional
- Inventory images (Options):
  - Upload images to `Inventory images inbox path` (default `/media/grocery_intel/inventory_images_inbox`)
  - The integration archives them to `Inventory images archive path` and analyzes them (vision requires a vision-capable LLM; supported providers include `ollama` and `openai`)
  - When available, the integration stores `taken_at` (EXIF capture time) for freshness; otherwise it falls back to import time (`created_at`)
  - Evidence boosts inventory (no exact counts) and suppresses auto-add for `Inventory evidence TTL (days)`
  - Exports:
    - `Exports folder path` (default `/media/grocery_intel/exports`)
    - Run `grocery_intel.export_data` and download the JSON from the Media browser (Local Media).

### Using exports for further analysis
The JSON produced by `grocery_intel.export_data` is designed to be portable and easy to analyze outside Home Assistant.

- Reporting tools:
  - Convert `data.receipts`, `data.line_items`, and `data.observations` to CSV and import into Excel/Google Sheets/Power BI/Tableau.
  - Keep `product_id` and `store_entity_id` so you can join to `data.products` (canonical names) and `data.stores` (canonical store identities).
- Data/BI workflows:
  - Load the JSON into Python (`pandas`), DuckDB, or a notebook to compute YTD/MTD spend, store comparisons, price inflation, and outlier baskets.
- LLM/chat analysis:
  - You can upload the exported JSON to a chat tool for ad-hoc questions (e.g., “What are my top stores YTD?”).
  - Privacy note: exports may contain sensitive information (store locations, purchase timing, item names). Use `scope: analytics` where possible and avoid sharing `debug/full` exports with third parties unless you are comfortable with the data leaving your network.

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
  - entity: sensor.grocery_intel_spend_week
  - entity: sensor.grocery_intel_spend_month
  - entity: sensor.grocery_intel_spend_ytd
  - entity: sensor.grocery_intel_spend_7d
  - entity: sensor.grocery_intel_spend_30d
  - entity: sensor.grocery_intel_avg_basket_30d
  # Receipt count entity_id can be `sensor.grocery_intel_receipt_count_30d` or (older installs) `sensor.grocery_intel_receipts_30d`.
  - entity: sensor.grocery_intel_receipt_count_30d
  - entity: sensor.grocery_intel_receipt_processing
  - entity: sensor.grocery_intel_top_stores_30d
  - entity: sensor.grocery_intel_recent_receipts
  - entity: sensor.grocery_intel_recent_activities
  - entity: sensor.grocery_intel_inventory_recently_seen
  - entity: sensor.grocery_intel_top_price_increases
  - entity: sensor.grocery_intel_overpaid_items
  - entity: sensor.grocery_intel_best_store_by_item
```

The list-style sensors store their details in attributes under `items` (shown in **Developer Tools → States**).

### Dashboard cards (friendlier examples)

These examples are designed to be easy to read in a Lovelace dashboard.

- Markdown cards support HA templates.
- The spend snapshot example uses the **Mushroom** card set (`custom:mushroom-*`).

#### Spend snapshot (Mushroom)

```yaml
type: vertical-stack
cards:
  - type: custom:mushroom-title-card
    title: Spend snapshot
    subtitle: At-a-glance totals
  - type: grid
    columns: 2
    square: false
    cards:
      - type: custom:mushroom-template-card
        icon: mdi:calendar-week
        primary: Week
        secondary: >-
          {% set cur = state_attr('sensor.grocery_intel_spend_month','currency')
            or state_attr('sensor.grocery_intel_spend_30d','currency') or 'kr' %}
          {{ '{:,.2f}'.format(states('sensor.grocery_intel_spend_week')|float(0)) }} {{ cur }}
      - type: custom:mushroom-template-card
        icon: mdi:calendar-month
        primary: Month
        secondary: >-
          {% set cur = state_attr('sensor.grocery_intel_spend_month','currency')
            or state_attr('sensor.grocery_intel_spend_30d','currency') or 'kr' %}
          {{ '{:,.2f}'.format(states('sensor.grocery_intel_spend_month')|float(0)) }} {{ cur }}
      - type: custom:mushroom-template-card
        icon: mdi:calendar-range
        primary: Last 30 days
        secondary: >-
          {% set cur = state_attr('sensor.grocery_intel_spend_month','currency')
            or state_attr('sensor.grocery_intel_spend_30d','currency') or 'kr' %}
          {{ '{:,.2f}'.format(states('sensor.grocery_intel_spend_30d')|float(0)) }} {{ cur }}
      - type: custom:mushroom-template-card
        icon: mdi:calendar
        primary: Year to date
        secondary: >-
          {% set cur = state_attr('sensor.grocery_intel_spend_month','currency')
            or state_attr('sensor.grocery_intel_spend_30d','currency') or 'kr' %}
          {{ '{:,.2f}'.format(states('sensor.grocery_intel_spend_ytd')|float(0)) }} {{ cur }}
```

#### Overpaid items (Markdown)

```yaml
type: markdown
title: Overpaid items (recent)
content: |
  _Flag “bad buys”: paid well above your baseline._
  {% set cur = state_attr('sensor.grocery_intel_spend_month','currency')
    or state_attr('sensor.grocery_intel_spend_30d','currency') or 'kr' %}
  {% set items = state_attr('sensor.grocery_intel_overpaid_items','items') or [] %}
  {% if items|length == 0 %}
  No items.
  {% else %}
  {% for it in items[:10] %}
  {% set d = as_local(as_datetime(it.observed_at)).strftime('%d-%b') if it.observed_at is defined else '' %}
  {% set paid = it.unit_price | float(0) %}
  {% set base = it.baseline | float(0) %}
  {% set diff = (paid - base) %}
  - {{ it.product }} @ {{ it.store }}: {{ '{:,.2f}'.format(paid) }} {{ cur }} vs {{ '{:,.2f}'.format(base) }} {{ cur }}
    ({{ '+' if diff >= 0 else '' }}{{ '{:,.2f}'.format(diff) }} {{ cur }}, +{{ (it.overpaid_pct * 100) | round(0) }}%) on {{ d }}
  {% endfor %}
  {% endif %}
```

#### Recent receipts (Markdown, de-duped)

```yaml
type: markdown
title: Recent receipts
content: |
  _Latest receipts (deduped by day + store + total)._
  {% set cur = state_attr('sensor.grocery_intel_spend_month','currency')
    or state_attr('sensor.grocery_intel_spend_30d','currency') or 'kr' %}
  {% set items = state_attr('sensor.grocery_intel_recent_receipts','items') or [] %}
  {% if items|length == 0 %}
  No receipts yet.
  {% else %}
  {% set seen = namespace(keys=[]) %}
  {% set out = namespace(rows=0) %}
  {% for r in items %}
  {% if out.rows >= 15 %}{% break %}{% endif %}
  {% set dt = r.get('purchased_at') %}
  {% set dkey = as_local(as_datetime(dt)).strftime('%Y-%m-%d') if dt else '' %}
  {% set d = as_local(as_datetime(dt)).strftime('%d-%b') if dt else '' %}
  {% set store = (r.get('store_name') or '') %}
  {% set total = (r.get('total')|float(0)) %}
  {% set key = dkey ~ '|' ~ store ~ '|' ~ ('%.2f'|format(total)) %}
  {% if key not in seen.keys %}
  {% set seen.keys = seen.keys + [key] %}
  {% set out.rows = out.rows + 1 %}
  - {{ d }} — {{ store }} — {{ '{:,.2f}'.format(total) }} {{ cur }} ({{ r.get('filename','') }})
  {% endif %}
  {% endfor %}
  {% endif %}
```

#### Inventory recently seen (Markdown, grouped by day)

```yaml
type: markdown
title: Inventory recently seen
content: |
  _Grouped by day; confidence shown as percent._
  {% set items = state_attr('sensor.grocery_intel_inventory_recently_seen','items') or [] %}
  {% if items|length == 0 %}
  No recent inventory evidence.
  {% else %}
  {% set ns = namespace(days=[]) %}
  {% for it in items[:50] %}
  {% set raw = it.get('last_seen_at') %}
  {% if raw %}
  {% set day = as_local(as_datetime(raw)).strftime('%d-%b') %}
  {% if day not in ns.days %}
  {% set ns.days = ns.days + [day] %}
  {% endif %}
  {% endif %}
  {% endfor %}
  {% for day in ns.days %}
  **{{ day }}**
  {% for it in items[:50] %}
  {% set raw = it.get('last_seen_at') %}
  {% if raw and as_local(as_datetime(raw)).strftime('%d-%b') == day %}
  {% set c = it.get('confidence') %}
  {% set pct = ((c * 100) | round(0)) if c is number else (c|string) %}
  - {{ it.get('product','') }} ({{ pct }}%)
  {% endif %}
  {% endfor %}
  {% endfor %}
  {% endif %}
```

#### Spend by category (Markdown)

```yaml
type: markdown
title: Spend by Category
content: |
  _Categories inferred from purchases (last 30d → avg month)._
  {% set cur = state_attr('sensor.grocery_intel_spend_month','currency')
    or state_attr('sensor.grocery_intel_spend_30d','currency') or 'kr' %}
  {% set items = state_attr('sensor.grocery_intel_top_stores_30d','items') or [] %}
  {% set factor = 30.4375 / 30 %}

  {% set groceries = ['ica', 'maxi', 'kvantum', 'willys'] %}
  {% set alc_tob = ['systembolaget', 'tobak', 'tobacco', 'cig', 'cigarette', 'snus'] %}
  {% set discount_household = ['dollarstore'] %}
  {% set specialty_food = ['spicenord', 'indiska spice'] %}
  {% set food_delivery = ['uber eats', 'ubereats', 'wolt', 'foodora', 'doordash', 'deliveroo'] %}
  {% set dining_out = ['restaurant', 'pizzeria', 'pizza', 'sushi', 'cafe', 'café', 'bar', 'grill', 'thai', 'indian'] %}

  {% set g = namespace(sum=0.0) %}
  {% set at = namespace(sum=0.0) %}
  {% set d = namespace(sum=0.0) %}
  {% set s = namespace(sum=0.0) %}
  {% set fd = namespace(sum=0.0) %}
  {% set do = namespace(sum=0.0) %}
  {% set o = namespace(sum=0.0) %}

  {% for it in items %}
    {% set name = (it.get('store_name','') | lower) %}
    {% set amt = (it.get('total',0) | float(0)) %}
    {% if groceries | select('in', name) | list | length > 0 %}
      {% set g.sum = g.sum + amt %}
    {% elif alc_tob | select('in', name) | list | length > 0 %}
      {% set at.sum = at.sum + amt %}
    {% elif discount_household | select('in', name) | list | length > 0 %}
      {% set d.sum = d.sum + amt %}
    {% elif specialty_food | select('in', name) | list | length > 0 %}
      {% set s.sum = s.sum + amt %}
    {% elif food_delivery | select('in', name) | list | length > 0 %}
      {% set fd.sum = fd.sum + amt %}
    {% elif dining_out | select('in', name) | list | length > 0 %}
      {% set do.sum = do.sum + amt %}
    {% else %}
      {% set o.sum = o.sum + amt %}
    {% endif %}
  {% endfor %}

  | Category | 30d spend | Avg / month |
  |---|---:|---:|
  | Groceries | {{ '{:,.2f}'.format(g.sum) }} {{ cur }} | {{ '{:,.2f}'.format(g.sum * factor) }} {{ cur }} |
  | Alcohol & Tobacco | {{ '{:,.2f}'.format(at.sum) }} {{ cur }} | {{ '{:,.2f}'.format(at.sum * factor) }} {{ cur }} |
  | Discount / household | {{ '{:,.2f}'.format(d.sum) }} {{ cur }} | {{ '{:,.2f}'.format(d.sum * factor) }} {{ cur }} |
  | Specialty food | {{ '{:,.2f}'.format(s.sum) }} {{ cur }} | {{ '{:,.2f}'.format(s.sum * factor) }} {{ cur }} |
  | Food delivery | {{ '{:,.2f}'.format(fd.sum) }} {{ cur }} | {{ '{:,.2f}'.format(fd.sum * factor) }} {{ cur }} |
  | Dining out | {{ '{:,.2f}'.format(do.sum) }} {{ cur }} | {{ '{:,.2f}'.format(do.sum * factor) }} {{ cur }} |
  | Other | {{ '{:,.2f}'.format(o.sum) }} {{ cur }} | {{ '{:,.2f}'.format(o.sum * factor) }} {{ cur }} |
```
