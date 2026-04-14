# Grocery Intel — Agent Instructions (Single Source of Truth)

This file is the canonical agent handoff for this repo. Prefer it over any other prompt/spec/context docs.

## Start here (every session)

1. Read this file end-to-end.
2. For any non-trivial change, skim `README.md` and `custom_components/grocery_intel/services.yaml` for user-facing expectations.
3. If behavior is unclear, inspect the code rather than guessing (especially prompt strings and service schemas).

## Compatibility baseline

- Minimum supported Home Assistant Core version: `2024.1.0` (see `custom_components/grocery_intel/manifest.json`).

## Debug info requests (how to ask users)

When you need debug info from Home Assistant, prefer providing a **ready-to-run** snippet for **Developer Tools → Template** that prints a single JSON blob the user can paste back.

Guidelines:
- Ask users to redact secrets (tokens/API keys); receipt IDs, filenames, and chat IDs are OK.
- If the requested info cannot be accessed via templates (e.g., system logs), ask for it separately and narrowly (time window + integration filter).

## Non-negotiable principles

- **Work with the existing architecture**; do not redesign it.
- **No new subsystems** unless explicitly requested.
- Prefer extending existing flows over creating parallel ones.
- Do not refactor unrelated code.
- Keep changes minimal, reviewable, and consistent with existing patterns.
- Respect Home Assistant async patterns; avoid blocking I/O in the event loop.
- Avoid new dependencies unless clearly justified.
- **User trust first**:
  - Every automatic action must be logged as an activity.
  - Actions should be explainable and undoable where logically possible.

## What exists today (high-level)

### Receipt ingestion

- Sources: manual service calls, file inbox scanning, Telegram intake (optional).
- Optional receipt categorization: `receipt_category` (`grocery` or `dining`) can be set manually and is auto-detected from merchant/file/text hints.
  - Manual `receipt_category` edits are persisted as `receipt_category_source=manual` and protected from automatic reclassification.
- Subcategories are persisted in one unified `receipt_subcategories` array (single-row for dining, multi-row for grocery).
  - `add_receipt` and `update_receipt` accept manual `receipt_subcategories` overrides (`[{subcategory,total}, ...]`) with category-aware validation.
- Receipts inbox → archive:
  - Inbox: `/media/grocery_intel/receipts_inbox`
  - Archive: `/media/grocery_intel/receipts_archive`
  - Defaults: scan interval 300s; archive TTL 30 days (configurable)
  - Dedupe is SHA-256 content-based; duplicates archived with a `_duplicate` suffix.
  - Telegram receipt completion feedback includes receipt summary and unknown-subcategory review details; very long messages are split into continuation messages only when they exceed Telegram limits.

### Extraction pipeline (receipts)

- Mode: LLM-only receipt extraction.
- Options wizard is provider-first (7 steps): LLM Provider → LLM Connection → LLM Prompting → Receipts → Inventory → Automation → Review.
- Status lifecycle: `pending` / `queued` / `running` / `done` / `failed`.
- PDFs attempt text-layer extraction first; may fall back to rendering + vision.
- Images go to vision-capable extraction when configured.
- If vision extraction cannot parse `purchased_at`, it falls back to current Home Assistant local datetime.
- Prompts enforce structured JSON output.
- Missing archived files are treated as failures to avoid retry loops.
- `reparse_receipts` supports optional `since` / `until` purchased-date filters (inclusive; date-only `until` includes full local day).

### Analytics, activities, and undo

- Rich data stored in HA storage; sensors are summaries.
- Spend analytics now include dynamic period sensors for categories and subcategories (`spend_by_category_periods`, `spend_by_subcategory_periods`) driven by persisted receipt categories/subcategories using week/month/year and 12-month buckets.
- Automatic actions create an activity record; undo exists for supported actions and must be safe.

### Automated shopping list (v1)

- Runs daily at 07:00 local time and can be invoked manually.
- Auto-add candidates are based on per-product purchase observations (cadence), with:
  - Minimum purchase history: 3 observations
  - Cooldown after auto-add: 7 days (default)
  - Confidence threshold: 0.75 (default)
- Optional name marker can be applied to auto-added shopping list items (configurable prefix/suffix).
- Optional translation can localize auto-added item names to HA language using the configured LLM, gated by a confidence threshold and cached per product.
- Never auto-removes items; undo removes only items added by that run.
- Undo also restores per-product shopping state updates written by that run (for example `last_auto_added_at` and store-tag metadata).

### Inventory images (v1)

- Inventory images inbox → archive:
  - Inbox: `/media/grocery_intel/inventory_images_inbox`
  - Archive: `/media/grocery_intel/inventory_images_archive`
  - Defaults: scan interval 300s; archive TTL 30 days; evidence TTL 7 days (configurable)
  - Dedupe by fingerprint; duplicates archived with `_duplicate` suffix.
- Vision analysis (Ollama/OpenAI when configured):
  - Extracts generic item labels.
  - Applies **boost-only evidence** (`last_seen_at`) to suppress auto-add for the evidence TTL (no negative inference).

## Doc + schema hygiene

- Keep user-facing docs and schemas in sync:
  - `README.md`
  - `custom_components/grocery_intel/services.yaml`
  - This `AGENTS.md`
- If you change a service signature, option default, entity name, or user workflow, update the docs in the same PR.

## Prompt hygiene

- Do not guess prompts; locate them in code:
  - Receipt prompt helpers live in `custom_components/grocery_intel/__init__.py`
  - Inventory-image prompt lives in `custom_components/grocery_intel/inventory_images.py`

## Commits and releases (safety rules)

- Before any commit: do a thorough review to ensure functionality works as intended and code follows existing standards/conventions (including async patterns and activity/undo expectations).
- Do **not** create tags/releases or publish artifacts unless explicitly asked.
- When asked to make a release:
  - Request confirmation immediately before running any tagging/release commands.
  - Update documentation as needed (at minimum `README.md`, `custom_components/grocery_intel/services.yaml`, and `AGENTS.md`).
  - Follow this strict sequence to avoid tag/manifest drift:
    1. Update `custom_components/grocery_intel/manifest.json` to the target release version.
    2. Update `CHANGELOG.md` for that exact version/date.
    3. Run compile check (`python3 -m py_compile custom_components/grocery_intel/*.py`).
    4. Commit release changes.
    5. Push `main` and verify local/remote alignment (`git rev-parse HEAD` equals `git rev-parse origin/main`).
    6. Create annotated tag from the verified `HEAD` (`git tag -a vX.Y.Z ...`).
    7. Push the tag.
    8. Create GitHub release from that tag.
    9. Verify post-release that tag content is correct (at minimum: `manifest.json` version under `ref=vX.Y.Z` matches `X.Y.Z`).
    10. Verify `main` and `origin/main` still match after release commands.
  - Never create/publish a release from a tag that already existed with mismatched contents; bump to a new patch version instead.
  - Provide a crisp summary of what changed (user-facing + developer-facing).

## Release numbering strategy

Use Semantic Versioning (`MAJOR.MINOR.PATCH`) with the current repo convention (`1.0.x`):

- `PATCH`:
  - Backward-compatible fixes/improvements.
  - Prompt/schema tuning, sensor payload additions, migrations that preserve compatibility.
- `MINOR`:
  - Backward-compatible feature additions (new service capabilities, significant new workflows/entities).
- `MAJOR`:
  - Breaking changes that require user dashboard/automation/config updates or incompatible model changes.

Guidance:
- Prefer compatibility aliases/migrations where practical to avoid MAJOR bumps.
- If a breaking change is unavoidable, call it out clearly in `CHANGELOG.md` with concrete migration notes.
