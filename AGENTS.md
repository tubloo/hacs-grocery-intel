# Grocery Intel — Agent Instructions (Single Source of Truth)

This file is the canonical agent handoff for this repo. Prefer it over any other prompt/spec/context docs.

## Start here (every session)

1. Read this file end-to-end.
2. For any non-trivial change, skim `README.md` and `custom_components/grocery_intel/services.yaml` for user-facing expectations.
3. If behavior is unclear, inspect the code rather than guessing (especially prompt strings and service schemas).

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
- Receipts inbox → archive:
  - Inbox: `/media/grocery_intel/receipts_inbox`
  - Archive: `/media/grocery_intel/receipts_archive`
  - Defaults: scan interval 300s; archive TTL 30 days (configurable)
  - Dedupe is SHA-256 content-based; duplicates archived with a `_duplicate` suffix.

### Extraction pipeline (receipts)

- Modes: heuristic / LLM / hybrid.
- Status lifecycle: `pending` / `queued` / `running` / `done` / `failed`.
- PDFs attempt text-layer extraction first; may fall back to rendering + vision.
- Images go to vision-capable extraction when configured.
- Prompts enforce structured JSON output.
- Missing archived files are treated as failures to avoid retry loops.

### Analytics, activities, and undo

- Rich data stored in HA storage; sensors are summaries.
- Automatic actions create an activity record; undo exists for supported actions and must be safe.

### Automated shopping list (v1)

- Runs daily at 07:00 local time and can be invoked manually.
- Auto-add candidates are based on per-product purchase observations (cadence), with:
  - Minimum purchase history: 3 observations
  - Cooldown after auto-add: 7 days (default)
  - Confidence threshold: 0.75 (default)
- Never auto-removes items; undo removes only items added by that run.

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
  - Provide a crisp summary of what changed (user-facing + developer-facing).
