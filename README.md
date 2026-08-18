# VeriFace

VeriFace is a dual-stream deepfake video classifier. A spatial model scores
individual face-cropped frames for texture-level manipulation artifacts, a
temporal model scores the frame sequence for motion-consistency artifacts,
and a calibrated logistic-regression ensemble fuses both into a final
per-video verdict. The pipeline covers dataset preparation (with
identity-aware splitting to prevent train/test leakage), training for all
three stages, end-to-end latency benchmarking, and a Streamlit app that
serves the trained pipeline with a PDF forensic report.

## Results

Trained and evaluated on DFDC: 19,909 videos (2,512 real / 17,397 fake),
split by identity group (train 15,767 / val 2,957 / test 985 / reserved
200), with zero identity leakage across splits, verified by
`src/eval/leakage_check.py`. The test split is the class-balanced,
true held-out measure and is the headline result: **0.9959 AUC / 96.14%
accuracy**.

| Stage | Level | Split | n | AUC | Accuracy |
|---|---|---|---|---|---|
| Spatial (EfficientNet-B3) | frame | val | — | 0.9901 | — |
| Temporal (Bi-LSTM + attention) | video | val | — | 0.9975 | — |
| Ensemble (calibrated logistic regression) | video | train | 15,767 | 1.0000 | 99.89% |
| Ensemble (calibrated logistic regression) | video | val | 2,957 | 0.9984 | 98.38% |
| **Ensemble (calibrated logistic regression)** | **video** | **test** | **985** | **0.9959** | **96.14%** |

The temporal model's train AUC reaches 1.0000 quickly, and val AUC/loss show
clear overfitting past that point. This is an expected characteristic of a
Bi-LSTM trained on a comparatively small number of video-level sequences
(thousands, not the frame-level millions the spatial model sees), not a
defect in the training procedure — it's why the ensemble is calibrated
rather than trusting the temporal model's raw output directly, and why the
test-split ensemble number, not the temporal model's val AUC, is the number
to trust for real-world accuracy.

### Latency

Measured via `src.eval.benchmark`, timing the full pipeline — video decode,
MTCNN face detection, spatial inference, temporal inference, and ensemble
fusion — over 20 sample videos:

| Metric | Value |
|---|---|
| Mean | 2918.8 ms/video |
| p95 | 6637.9 ms/video |

MTCNN face detection and video decode, not model inference, are the
dominant cost. This replaces an earlier, unverified `<250ms` figure that
existed only as a hardcoded UI string with no measurement behind it.

## Architecture

```
raw video
   │
   ▼
┌─────────────────────┐
│ MTCNN face detection │  N evenly-spaced frames per video, face-cropped
└──────────┬───────────┘
           │ face crops [N, 3, 224, 224]
           ▼
┌───────────────────────────┐
│ Spatial: EfficientNet-B3   │  per-frame embedding [N, 1536] + per-frame P(fake)
│ (src/models/spatial.py)    │
└──────────┬─────────────────┘
           │ embeddings [N, 1536]
           ▼
┌───────────────────────────┐
│ Temporal: Bi-LSTM +         │  sequence-level logit
│ attention pooling           │
│ (src/models/temporal.py)    │
└──────────┬─────────────────┘
           │ spatial stats (mean/max/std/top-3) + temporal logit
           ▼
┌───────────────────────────┐
│ Ensemble: calibrated        │  final P(fake)
│ logistic regression         │
│ (src/models/ensemble.py)    │
└──────────┬─────────────────┘
           ▼
        verdict
```

**Why two streams instead of one model.** Spatial and temporal artifacts
are different signals: a single manipulated frame can look correct in
isolation (spatial) while the manipulation still breaks physical continuity
across frames (temporal), and vice versa. Splitting them lets each model
specialize and lets the temporal model train on precomputed 1536-d
embeddings rather than re-running the CNN backbone over every frame of
every sequence — several orders of magnitude cheaper than joint end-to-end
training over raw frame sequences.

**Why calibration on the ensemble.** A plain `LogisticRegression`'s
`decision_function` output is a ranking score, not a probability. Wrapping
it in `CalibratedClassifierCV` (sigmoid/Platt scaling via cross-validated
folds) turns the ensemble's output into a number that means what it says —
`0.96` is actually close to a 96% likelihood of being correct, not just
"higher than 0.5." This matters both for the UI's displayed confidence
value and for any downstream thresholding.

**Why identity-grouped splitting instead of random splitting.** DFDC fakes
are derived from real source videos of the same identity; FaceForensics++
stems directly encode a `<target_id>_<source_id>` pair. Splitting videos at
random risks putting a real video and the fakes derived from it — the same
face — on both sides of the train/test boundary, which lets a model learn
to recognize the *person* rather than the *manipulation*, inflating
held-out metrics without the model having learned anything transferable.
`src/data/build_splits.py` groups videos by identity first (via
`src/data/identity.py`), then assigns whole groups to splits, so no
identity crosses a split boundary — verified directly by
`src/eval/leakage_check.py` rather than assumed.

## Engineering notes

A few real issues found and fixed during development, kept here because
they're the kind of thing that silently corrupts results if missed:

- **Checkpoint contract mismatch.** Training originally saved the ensemble
  as a bare `scikit-learn` object to `ensemble_best.pkl`; the serving code
  expected a `{"calibrator": ...}` dict at a different filename,
  `ensemble_final.joblib`. Loading a real trained checkpoint into the app
  raised `TypeError: 'CalibratedClassifierCV' object is not subscriptable`.
  Fixed by giving both sides exactly one schema to import from
  (`src/models/checkpoint.py`), so the two can't drift apart silently again.

- **MTCNN output-range bug.** `src/data/preprocess_raw.py` assumed MTCNN's
  face-crop tensor was already in `[0, 255]` and clamped it directly before
  casting to `uint8`. MTCNN's actual output range depends on configuration
  and isn't `[0, 255]`, so the clamp silently zeroed out nearly every pixel
  — producing all-black face crops across the dataset with no error or
  warning. Fixed by rescaling from each tensor's own observed min/max before
  casting.

- **DFDC identity leakage.** `src/data/identity.py` originally treated every
  DFDC video as an opaque, unlinkable identity, since no source/original
  mapping had survived into the project. That's now closed for this dataset
  pull: per-part DFDC `metadata.json` files were recovered and combined into
  one mapping (fake stem → source real-video stem), which `identity.py`
  loads via `VERIFACE_DFDC_METADATA_PATH` and unions into the same
  leakage-detection graph FaceForensics++ pairs use. Coverage is scoped to
  what DFDC's metadata actually encodes: fake-to-source pairing. Two real
  videos of the same person with no fake derived from either are still
  treated as separate identities, since DFDC's metadata exposes no
  independent actor/person ID beyond that relationship.

- **Training stability.** Gradient clipping (`--grad-clip-norm`, default
  `1.0`) and explicit non-finite loss/gradient-norm guards (skip the batch
  rather than let a NaN propagate into the optimizer state) were added to
  `src/train/train_spatial.py` after observing instability in mixed-precision
  training on real data.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.2.2` / `torchvision==0.17.2` (CPU wheels
from plain PyPI). For a CUDA build:

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

The provided `Dockerfile` builds on `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04`
and installs its own pinned `torch==2.13.0`/`torchvision==0.28.0` via the
same `cu129` index — check `Dockerfile` directly before relying on a
specific version for GPU work, since it and `requirements.txt` are not
currently pinned to the same torch release.

```bash
docker build -t veriface .
docker run --gpus all -p 8501:8501 -v /path/to/data:/workspace/Dataset veriface
```

DFDC-side leakage detection additionally requires the combined metadata
file referenced above:

```bash
export VERIFACE_DFDC_METADATA_PATH=/path/to/dfdc_combined_metadata.json
```

## Usage

Full pipeline, in order:

```bash
# 0. Raw video -> face-cropped frames, per (dataset, label)
python -m src.data.preprocess_raw \
    --input-dir raw/dfdc_real --output-dir Dataset/ \
    --dataset-name dfdc --label real --workers 8
python -m src.data.preprocess_raw \
    --input-dir raw/dfdc_fake --output-dir Dataset/ \
    --dataset-name dfdc --label fake --workers 8

# 1. Identity-grouped train/val/test/reserved split + leakage check
python -m src.data.build_splits --check-leakage

# 2. Sample frames per video from the split file lists
python -m src.data.extract_frames --splits train val test

# 3. Train the spatial (frame-level) model
python -m src.train.train_spatial --epochs 12 --batch-size 32

# 4. Extract per-frame embeddings using the trained spatial backbone
python -m src.data.extract_embeddings

# 5. Train the temporal (video-level) model on embeddings
python -m src.train.train_temporal --epochs 25 --batch-size 16

# 6. Fit the calibrated ensemble on spatial + temporal features
python -m src.train.train_ensemble

# 7. Measure real end-to-end latency
python -m src.eval.benchmark --videos-dir /path/to/sample_videos --n-frames 20

# 8. Run the app
streamlit run app/main.py
```

Every stage reads its defaults from `src/config.py` (overridable via
`VERIFACE_*` environment variables) and `--help` lists every flag.
`src/eval/leakage_check.py` can also be run standalone against any split
files:

```bash
python -m src.eval.leakage_check --train data/train.txt --val data/val.txt --test data/test_internal.txt --out results/leakage_report.json
```

The app requires trained checkpoints under
`checkpoints/{spatial,temporal,ensemble}/`. If the spatial or temporal
checkpoint is missing or fails to load, the app shows a visible error and
refuses to serve predictions rather than falling back to
randomly-initialized weights.

### Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -v
```

All tests run on CPU against synthetic/tiny data — no GPU or dataset
required.

## Repository structure

```
src/
  config.py           # single source of truth for all data/checkpoint paths
  data/
    build_splits.py     # identity-grouped train/val/test/reserved split + leakage gate
    identity.py         # FF++ and DFDC identity-token parsing and grouping
    preprocess_raw.py    # raw video -> MTCNN face crops -> *_Face_only_data layout
    extract_frames.py    # split-aware frame sampling
    extract_embeddings.py # per-frame spatial embeddings via the trained backbone
    face_crop.py          # standalone MTCNN cropping utility (frame-folder input)
    augmentations.py      # ffmpeg re-encoding helpers for compression augmentation
  models/
    spatial.py           # EfficientNet-B3 + MLP head
    temporal.py           # Bi-LSTM + attention pooling
    ensemble.py           # ensemble feature-vector contract
    checkpoint.py         # the one shared save/load schema for all three stages
  train/
    train_spatial.py, train_temporal.py, train_ensemble.py
  inference/
    pipeline.py          # the one canonical inference path (app + benchmark both use it)
  eval/
    leakage_check.py      # identity-leakage detection/reporting
    benchmark.py           # real end-to-end latency measurement
    metrics.py             # results/metrics.json read/write contract
app/
  main.py               # Streamlit app entrypoint
  report.py             # PDF forensic report generation
scripts/
  run_pipeline.sh         # numbered pipeline stage reference
tests/                  # pytest unit tests — CPU-only, no GPU/dataset required
archive/                # original pre-restructure notebooks/scripts, reference only
results/                # results/metrics.json (accuracy + latency), written by train_ensemble.py / benchmark.py
Untitled.ipynb          # ad hoc training/experimentation notebook used for GPU runs
```

## Limitations / future work

- **Latency is not real-time.** 2.9s mean / 6.6s p95 per video is dominated
  by MTCNN face detection and video decode, not model inference. The clear
  next step is optimizing or replacing the face-detection/decode path
  (batched detection, a faster detector, hardware-accelerated decode)
  before model-level optimization would matter.
- **DFDC leakage coverage is fake-to-source only.** `src/eval/leakage_check.py`
  closes DFDC identity leakage for videos with a resolvable fake→source
  pairing in the recovered metadata. It does not link two real videos of
  the same person that have no fake derived from either — DFDC's own
  metadata doesn't expose an actor/person ID beyond that pairing, so this
  is a real, documented boundary of what the checker can verify, not an
  oversight.

## License

Not specified in the original project — add one before treating this as a
public/shareable repository.
