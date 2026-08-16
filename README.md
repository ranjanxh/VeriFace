# VeriFace

Dual-stream deepfake video detector: a spatial (frame-level) CNN and a
temporal (sequence-level) RNN each score a video independently, and a
calibrated logistic-regression ensemble fuses the two into a final verdict.
Ships as a Streamlit app with a confidence gauge, per-frame timeline, scan
history, and a downloadable PDF forensic report.

> **New to this repo?** Read `HANDOFF.md` first — it has the exact dataset
> layout this code expects, environment requirements, the full run order,
> and what's still unverified because this codebase was restructured and
> hardened in an environment with no GPU and no dataset available.

## Architecture

```
                 ┌──────────────────┐
   video ──────► │  MTCNN face crop │
                 └────────┬─────────┘
                          │ N sampled frames
                          ▼
              ┌────────────────────────┐
              │  Spatial: EfficientNet-B3│──► per-frame P(fake)
              │  (src/models/spatial.py) │
              └───────────┬────────────┘
                          │ 1536-d embeddings
                          ▼
              ┌────────────────────────┐
              │  Temporal: Bi-LSTM +    │──► sequence-level logit
              │  attention pooling      │
              │  (src/models/temporal.py)│
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Ensemble: calibrated   │──► final P(fake)
              │  logistic regression    │
              │  (src/models/ensemble.py)│
              └────────────────────────┘
```

- **Spatial stream** (`src/models/spatial.py`): EfficientNet-B3 backbone
  (via `timm`) + a small MLP head, trained per-frame on face crops to detect
  spatial/texture manipulation artifacts.
- **Temporal stream** (`src/models/temporal.py`): a 2-layer bidirectional
  LSTM with learned attention pooling over the sequence of per-frame spatial
  embeddings, trained to detect motion/consistency artifacts across a video.
- **Ensemble** (`src/models/ensemble.py` + `src/models/checkpoint.py`): a
  `CalibratedClassifierCV(LogisticRegression)` over 5 hand-engineered
  features (spatial mean/max/std/top-3-mean + the temporal logit).

## Repository layout

```
src/
  config.py       # single source of truth for all data/checkpoint paths
  data/           # split generation, frame extraction, face cropping, embeddings
  models/         # model definitions + the shared checkpoint schema
  train/          # training entrypoints for all three stages
  inference/      # THE canonical inference pipeline (used by app + benchmark)
  eval/           # leakage checker, benchmark, metrics helpers
app/
  main.py         # Streamlit app entrypoint
  report.py       # PDF forensic report generation
scripts/
  run_pipeline.sh # documents the exact, numbered, ordered pipeline commands
tests/            # pytest unit tests — no GPU/dataset required
archive/          # original notebooks + scripts, kept for reference only
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training/inference, install the CUDA build of torch/torchvision
*before* the line above (plain PyPI wheels are CPU-only) — see `Dockerfile`
for the exact command, or use the provided Docker image:

```bash
docker build -t veriface .
docker run --gpus all -p 8501:8501 -v /path/to/data:/workspace/Dataset veriface
```

## Running the pipeline

See `scripts/run_pipeline.sh` (run with no arguments to print the full
numbered stage list) and `HANDOFF.md` → "Exact Run Order" for prerequisites,
expected inputs/outputs, and rough runtime per stage. In short:

```
1. build_splits (+ leakage check)  →  2. extract_frames  →  3. train_spatial
→  4. extract_embeddings  →  5. train_temporal  →  6. train_ensemble
→  7. benchmark  →  8. run the app
```

**None of this has been run in this repository** — there is no dataset or
GPU in the environment this was restructured in. See `HANDOFF.md` for what's
needed to actually execute it.

## Running the app

```bash
streamlit run app/main.py
```

Requires trained checkpoints under `checkpoints/{spatial,temporal,ensemble}/`
(see `src/config.py` for exact paths, overridable via `VERIFACE_*` env
vars). If a mandatory checkpoint (spatial or temporal) is missing or fails
to load, the app shows a visible error and refuses to serve predictions —
it will not silently fall back to randomly-initialized weights.

## Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -v
```

All tests run on CPU with synthetic/tiny data — no GPU or dataset required.

## Benchmarks: pending

There is no verified accuracy or latency number for this pipeline yet.
The app deliberately shows "not yet evaluated" / "not yet benchmarked"
rather than a fabricated figure until `src/train/train_ensemble.py` and
`src/eval/benchmark.py` have actually been run on real hardware with real
data. **See `HANDOFF.md`** for exactly what's needed to produce those
numbers, and for the known overfitting/identity-leakage concern that should
be resolved (via `src/eval/leakage_check.py` + a corrected retrain) before
trusting whatever numbers come out.

## License

Not specified in the original project — add one before treating this as a
public/shareable repository.
