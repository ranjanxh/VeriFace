# Archive

This directory holds the original, unmodified project artifacts from before the
2026-08 restructure. Nothing in here is imported by `src/`, `app/`, `scripts/`,
or `tests/` — it exists purely as a historical reference so nothing is lost.

- `notebooks/` — every original `.ipynb` file (`DeepfakevFinal.ipynb` plus the
  9 notebooks that used to live in `notebooks/`). Their logic has been
  extracted, cleaned up, and moved into `src/` and `app/` — see `HANDOFF.md`
  at the repo root, section "What Changed From the Original Codebase", for a
  file-by-file mapping of old notebook -> new module.
- `legacy_app.py` — the Streamlit app that used to be committed as `app.py` at
  the repo root. It was the *simpler* of two app variants that existed in
  this project's history (the more feature-complete variant — PDF reports,
  gauges, scan history — only ever existed inside `DeepfakevFinal.ipynb` and
  was never the checked-in `app.py`). The new `app/main.py` merges the best of
  both.
- `legacy_src_process_data/` — the original `src/process_data/*.py` scripts,
  before being moved/renamed into `src/data/`.

Do not import from this directory. Do not edit it. If you need to recover
original behavior for comparison, read it here, then make the change in the
real `src/`/`app/` tree.
