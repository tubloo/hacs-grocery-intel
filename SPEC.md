# Grocery Intel — Specification (Living)

Grocery Intel is a **local-first Home Assistant integration** that helps households
understand grocery spend and purchasing patterns from receipts.

This document defines **non-negotiable principles and architecture**, not a roadmap.

---

## Core principles

- **Local-first**
  - No mandatory cloud services
  - Data stays on the Home Assistant host
- **Home Assistant native**
  - Custom integration (not an add-on)
  - Uses HA’s configured database (SQLite / MariaDB / Postgres)
- **User trust first**
  - Every automatic action is logged
  - Actions are explainable and undoable
  - User can always correct parsed data
- **Resource-conscious**
  - Minimal entities
  - Heavy computation delegated to optional external services
- **Extensible, not brittle**
  - New capabilities extend existing flows
  - No “one-off” pipelines

---

## Architectural truths

- Domain: `grocery_intel`
- Runs inside HA Core
- Event-driven (no tight polling loops)
- Rich data stored in HA storage; sensors are summaries
- Receipt lifecycle:
  1. Ingest (manual service or file inbox)
  2. Extract (heuristic / LLM / hybrid)
  3. Review/edit (optional)
  4. Reprocess derived analytics

---

## Explicit non-goals

- Mandatory cloud dependency
- Silent automation without activity logging
- Entity explosion (per-item sensors, etc.)
- Forcing correctness over user control

---

This spec is intentionally minimal.
Details live in CONTEXT.md.
