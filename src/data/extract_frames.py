"""Sample N frames per video into ``preprocessed/frames/<split>/<video_stem>/frame_NN.jpg``.

Consolidates two previously-divergent implementations found in the archived
codebase:

- ``archive/notebooks/extract_frames.ipynb`` ("02 - Sample frames from
  face-cropped videos") — reads the split files written by
  ``build_splits.py`` and is the version actually wired into the rest of the
  pipeline (``train_spatial.py`` expects exactly this
  ``<split>/<stem>/frame_*.jpg`` layout).
- ``archive/legacy_src_process_data/extract_frames.py`` — a generic,
  split-unaware CLI utility (``--src``/``--out``) that samples frames from a
  flat folder of videos. Appears to be an earlier/exploratory variant.

This module keeps the split-aware behavior (the one the rest of the
pipeline actually depends on) and keeps the CLI ergonomics of the standalone
script. It also handles the "face-only" videos being already-cropped-face
*frame folders* rather than actual video files, matching the notebook's
``copy_first_images_from_folder`` fallback.

Do NOT run this without the dataset in place — see HANDOFF.md.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2

from src import config

logger = logging.getLogger(__name__)

FRAMES_PER_VIDEO_DEFAULT = 8


def sample_frames_from_video(video_path: Path, out_dir: Path, n: int) -> str:
    """Extract n evenly-spaced frames from a video file. Returns a status string."""
    if (out_dir / "frame_00.jpg").exists():
        return "skipped"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return "no_frames"
        indices = [int(i * total / n) for i in range(n)]
        saved = 0
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            cv2.imwrite(str(out_dir / f"frame_{i:02d}.jpg"), frame)
            saved += 1
    finally:
        cap.release()
    return "saved" if saved > 0 else "no_frames"


def copy_first_frames_from_folder(folder_path: Path, out_dir: Path, n: int) -> str:
    """Fallback for inputs that are already folders of extracted frame images
    rather than video files (as produced by some face-cropping pipelines)."""
    if (out_dir / "frame_00.jpg").exists():
        return "skipped"
    imgs = sorted(list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")))
    if not imgs:
        return "no_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, p in enumerate(imgs[:n]):
        shutil.copy(str(p), str(out_dir / f"frame_{i:02d}.jpg"))
        saved += 1
    return "saved" if saved > 0 else "no_frames"


def extract_split(split_name: str, list_path: Path, out_root: Path, n: int) -> dict[str, int]:
    counts = {"saved": 0, "skipped": 0, "no_frames": 0, "missing": 0}
    if not list_path.exists():
        logger.warning("[%s] split file not found: %s (skipping)", split_name, list_path)
        return counts

    lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("[%s] processing %d entries", split_name, len(lines))

    for line in lines:
        src = Path(line)
        stem = src.stem
        out_dir = out_root / split_name / stem
        if not src.exists():
            counts["missing"] += 1
            continue
        status = (
            copy_first_frames_from_folder(src, out_dir, n)
            if src.is_dir()
            else sample_frames_from_video(src, out_dir, n)
        )
        counts[status] = counts.get(status, 0) + 1

    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=list(config.SPLITS), help="Which splits to process")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.FRAMES_DIR)
    parser.add_argument("--frames-per-video", type=int, default=FRAMES_PER_VIDEO_DEFAULT)
    args = parser.parse_args(argv)

    split_files = {
        "train": args.data_dir / "train.txt",
        "val": args.data_dir / "val.txt",
        "test": args.data_dir / "test_internal.txt",
    }

    total_counts: dict[str, int] = {}
    for split_name in args.splits:
        list_path = split_files.get(split_name)
        if list_path is None:
            logger.error("Unknown split %r (expected one of %s)", split_name, list(split_files))
            return 2
        counts = extract_split(split_name, list_path, args.out_dir, args.frames_per_video)
        logger.info("[%s] done: %s", split_name, counts)
        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

    logger.info("Totals across requested splits: %s", total_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
