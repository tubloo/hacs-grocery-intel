# Codex Working Instructions — Grocery Intel

Before doing anything:
1. Read `SPEC.md`
2. Read `CONTEXT.md`

Rules:
- Work with the existing architecture; do not redesign it.
- Do not invent new subsystems unless explicitly asked.
- Prefer extending existing flows over creating parallel ones.
- Do not refactor unrelated code.
- If something is unclear, leave a TODO or ask rather than guessing.
- Keep changes minimal and reviewable.

When implementing:
- Respect Home Assistant async patterns.
- Avoid new dependencies unless clearly justified.
- Ensure new automatic actions are logged as activities.
- Preserve undo semantics where possible.

After changes:
- Clearly state which files were modified.
- Avoid speculative features.

Then proceed with the requested task.
