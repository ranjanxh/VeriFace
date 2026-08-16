# results/

This directory is where `metrics.json` lives once it's been generated on
real hardware. It is **empty in this checkout** — no training or
benchmarking has been run in this environment (no GPU, no dataset). See
`HANDOFF.md` at the repo root, "What Still Needs Real Hardware to Complete".

`app/main.py` reads `results/metrics.json` and shows an explicit "not yet
evaluated" / "not yet benchmarked" badge for any section that's missing,
rather than a fabricated number. Do not hand-write this file with plausible
numbers — populate it by actually running the two scripts below.

## Expected schema

```json
{
  "schema_version": "1.0",
  "accuracy": {
    "evaluated_at_utc": "...",
    "train": {"n": 0, "auc": 0.0, "accuracy": 0.0},
    "val":   {"n": 0, "auc": 0.0, "accuracy": 0.0},
    "test":  {"n": 0, "auc": 0.0, "accuracy": 0.0},
    "note": "..."
  },
  "latency": {
    "measured_at_utc": "...",
    "device": "cuda",
    "gpu_name": "...",
    "n_frames_sampled_per_video": 20,
    "n_videos": 0,
    "n_failures": 0,
    "failures": [],
    "mean_ms": 0.0,
    "median_ms": 0.0,
    "p95_ms": 0.0,
    "min_ms": 0.0,
    "max_ms": 0.0,
    "note": "..."
  }
}
```

- `accuracy` is written by `src/train/train_ensemble.py` after fitting the
  ensemble on real embeddings.
- `latency` is written by `src/eval/benchmark.py` after timing real
  end-to-end inference (video decode + face detection + all three models)
  over a directory of sample videos on the target GPU.

Both scripts merge-write via `src.eval.metrics.update_metrics_section`, so
running one never wipes out the other's results.
