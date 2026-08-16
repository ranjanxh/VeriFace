"""Measure real end-to-end inference latency and populate the ``latency``
section of ``results/metrics.json``.

This is the script that is supposed to produce the number the original app
hardcoded as "<250ms" with no measurement behind it at all (see
KNOWLEDGE_BASE.md / HANDOFF.md). It calls the exact same
``src.inference.pipeline.analyze_video`` the app uses, so the measurement is
representative of real app latency, not just the model forward pass (the
original's ``inference_ms`` only timed the two model calls and excluded
video decode + MTCNN face detection, which is almost certainly the dominant
cost).

Do NOT run this without: (a) real trained checkpoints in
``checkpoints/{spatial,temporal,ensemble}/``, and (b) a directory of sample
videos to benchmark against, and (c) a GPU matching the target deployment
(RTX Pro 6000 class) if you want numbers representative of production. None
of that is available in this environment. See HANDOFF.md.

Usage (once the above are available):

    python -m src.eval.benchmark --videos-dir /path/to/sample_videos --n-frames 20
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from src import config
from src.eval.metrics import update_metrics_section
from src.inference.pipeline import InferenceError, analyze_video, load_models

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi")


def discover_videos(videos_dir: Path) -> list[Path]:
    return sorted(p for ext in SUPPORTED_EXTENSIONS for p in videos_dir.glob(f"*{ext}"))


def run_benchmark(videos: list[Path], models, n_frames: int) -> dict:
    per_video_ms: list[float] = []
    failures: list[dict] = []

    for video_path in videos:
        try:
            result, _previews, _meta = analyze_video(video_path, models, n_frames=n_frames)
            per_video_ms.append(result.total_ms)
            logger.info("%s: %.1f ms", video_path.name, result.total_ms)
        except InferenceError as exc:
            failures.append({"video": str(video_path), "error": str(exc)})
            logger.warning("Skipping %s: %s", video_path, exc)

    if not per_video_ms:
        raise RuntimeError(
            f"No videos produced a timing measurement (n_videos={len(videos)}, "
            f"n_failures={len(failures)}). Cannot compute a benchmark summary."
        )

    sorted_ms = sorted(per_video_ms)

    def percentile(p: float) -> float:
        idx = min(int(len(sorted_ms) * p), len(sorted_ms) - 1)
        return sorted_ms[idx]

    return {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "device": str(models.device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_frames_sampled_per_video": n_frames,
        "n_videos": len(per_video_ms),
        "n_failures": len(failures),
        "failures": failures,
        "mean_ms": round(statistics.fmean(per_video_ms), 1),
        "median_ms": round(statistics.median(per_video_ms), 1),
        "p95_ms": round(percentile(0.95), 1),
        "min_ms": round(min(per_video_ms), 1),
        "max_ms": round(max(per_video_ms), 1),
        "note": (
            "Full end-to-end latency: video decode + MTCNN face detection + "
            "spatial + temporal + ensemble inference, timed via "
            "src.inference.pipeline.analyze_video. This is the same function "
            "path the Streamlit app uses."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory of sample .mp4/.mov/.avi videos to benchmark against")
    parser.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    parser.add_argument("--n-frames", type=int, default=20)
    args = parser.parse_args(argv)

    if not args.videos_dir.exists():
        logger.error("--videos-dir not found: %s", args.videos_dir)
        return 2

    videos = discover_videos(args.videos_dir)
    if not videos:
        logger.error("No videos with extensions %s found under %s", SUPPORTED_EXTENSIONS, args.videos_dir)
        return 2
    logger.info("Found %d videos to benchmark", len(videos))

    models = load_models(checkpoint_dir=args.checkpoint_dir)
    if not models.critical_ok:
        logger.error("Models not ready: %s. Cannot benchmark an untrained/failed pipeline.", models.degraded_reason)
        return 2

    summary = run_benchmark(videos, models, args.n_frames)
    update_metrics_section("latency", summary)
    logger.info("Wrote latency section of %s: mean=%.1fms p95=%.1fms", config.METRICS_FILE, summary["mean_ms"], summary["p95_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
