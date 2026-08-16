#!/usr/bin/env bash
# Numbered, ordered pipeline: raw dataset -> trained models -> running app.
#
# This is documentation-as-a-script, not a "just run it" one-liner: every
# stage depends on real dataset/checkpoint files that do not exist in this
# repo (see HANDOFF.md "Dataset Required"). It is meant to be read, and run
# stage-by-stage, on a machine that actually has the dataset and a GPU
# (see HANDOFF.md "Environment Requirements" for what's assumed — an
# RTX Pro 6000 / H200 class GPU).
#
# Usage: bash scripts/run_pipeline.sh [stage_number]
#   With no argument, prints the stage list and exits (does not run anything).
#   With a stage number (1-8), runs only that stage.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE="${1:-}"

run_stage() {
    local n="$1"; shift
    echo "=== Stage $n: $* ==="
    "$@"
}

case "$STAGE" in
  1) run_stage 1 python -m src.data.build_splits --check-leakage ;;
  2) run_stage 2 python -m src.data.extract_frames --splits train val test ;;
  3) run_stage 3 python -m src.train.train_spatial ;;
  4) run_stage 4 python -m src.data.extract_embeddings ;;
  5) run_stage 5 python -m src.train.train_temporal ;;
  6) run_stage 6 python -m src.train.train_ensemble ;;
  7)
    echo "=== Stage 7: benchmark (requires a directory of sample videos) ==="
    echo "python -m src.eval.benchmark --videos-dir <path/to/sample_videos>"
    ;;
  8) run_stage 8 streamlit run app/main.py ;;
  "")
    cat <<'EOF'
VeriFace pipeline stages (see HANDOFF.md "Exact Run Order" for details,
prerequisites, and rough runtime per stage):

  1. python -m src.data.build_splits --check-leakage
     Build identity-grouped train/val/test splits + labels.json from
     Dataset/*_Face_only_data/. Runs the leakage checker immediately after
     writing splits and fails loudly if any identity crosses splits.

  2. python -m src.data.extract_frames --splits train val test
     Sample 8 frames/video into preprocessed/frames/<split>/<stem>/.

  3. python -m src.train.train_spatial
     Train the EfficientNet-B3 spatial classifier on extracted frames.
     Writes checkpoints/spatial/spatial_best_valAUC.pth.

  4. python -m src.data.extract_embeddings
     Use the trained spatial backbone to extract per-frame embeddings for
     every video -> embeddings/<split>/<stem>.npy.

  5. python -m src.train.train_temporal
     Train the Bi-LSTM+attention temporal classifier on embeddings.
     Writes checkpoints/temporal/temporal_best_valAUC.pth.

  6. python -m src.train.train_ensemble
     Fit the calibrated logistic-regression fusion head on spatial+temporal
     features. Writes checkpoints/ensemble/ensemble_final.joblib AND the
     "accuracy" section of results/metrics.json.

  7. python -m src.eval.benchmark --videos-dir <path/to/sample_videos>
     Measure real end-to-end latency on the target GPU. Writes the
     "latency" section of results/metrics.json.

  8. streamlit run app/main.py
     Launch the app. Reads results/metrics.json for the hero metrics —
     shows "pending" badges until stages 6/7 have actually been run.

Run one stage:  bash scripts/run_pipeline.sh <N>
EOF
    ;;
  *)
    echo "Unknown stage: $STAGE (expected 1-8, or no argument to see the list)" >&2
    exit 1
    ;;
esac
