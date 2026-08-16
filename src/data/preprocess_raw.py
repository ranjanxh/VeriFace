"""Raw video -> ``<DATASET>_Face_only_data`` preprocessing — the missing
"Step 0" flagged in HANDOFF.md ("Dataset Required" / "Known Limitations"):
nothing in the original codebase documented how raw downloaded DFDC/FF++/
Celeb-DF videos become the pre-cropped ``*_Face_only_data`` folders that
``src/data/build_splits.py`` expects to already exist. This module is that
step.

Input: a folder of raw ``.mp4``/``.mov``/``.avi`` files, all belonging to
one dataset (``dfdc`` | ``ffpp`` | ``celebdf``) and one label
(``real`` | ``fake``) — matching how these datasets are naturally organized
on disk (DFDC and Celeb-DF ship real/fake in separate folders already; FF++
labels usually come from a manifest, handled below).

Output: for each video, ``N`` evenly-spaced frames are sampled
(``N = src.config.FRAMES_PER_VIDEO``, the same value
``src/data/extract_frames.py`` already uses — not a new hardcoded number),
MTCNN face-cropped (same detector configuration as
``src/data/face_crop.py``'s ``crop_faces_from_frames`` —
``keep_all=False, select_largest=True``; kept in sync by hand since that
module is out of scope for this change), and written to::

    <output_root>/<DATASET_PREFIX>_Face_only_data/<video_stem>/frame_00.jpg
    <output_root>/<DATASET_PREFIX>_Face_only_data/<video_stem>/frame_01.jpg
    ...

matching the exact layout documented in HANDOFF.md "Dataset Required" and
read by ``src/data/build_splits.py``:

    dfdc    + real   -> DFDC_REAL_Face_only_data
    dfdc    + fake   -> DFDC_FAKE_Face_only_data
    ffpp    + either -> FF_Face_only_data          (+ metadata.csv, see below)
    celebdf + real   -> Celeb_real_face_only
    celebdf + fake   -> Celeb_fake_face_only

**FF++ needs a label file, not just a folder split**: unlike DFDC/Celeb-DF,
both FF++ labels land in the same ``FF_Face_only_data`` folder (there is no
separate real/fake top-level folder for it in ``build_splits.py``'s
expected layout). ``build_splits.py`` resolves FF++ labels from
``FF_Face_only_data/metadata.csv`` (columns ``video,label``) when present,
falling back to checking whether "fake"/"real" appears in the *parent
folder path* — which never actually matches for a fixed single dirname like
``FF_Face_only_data`` (a pre-existing quirk in that script, out of scope for
this change since it isn't touched here). So for ``--dataset-name ffpp``,
this module also appends rows to that ``metadata.csv`` — otherwise the
labels this stage assigns would be silently unrecoverable downstream.

**Failure handling**: every video-level anomaly (corrupt/unreadable file,
shorter than ``N`` frames, zero faces detected in any sampled frame) is
appended to a dedicated failures log (default:
``<config.LOG_DIR>/preprocessing_failures.log``) as well as logged via the
standard logger — never silently skipped. Videos that partially succeed
(some sampled frames had no detectable face, but at least one did) are
still saved, with the shortfall noted in the failures log.

**Parallelism**: designed for thousands of videos. ``--workers > 1`` uses a
``multiprocessing`` pool (spawn context, one MTCNN instance built once per
worker process, not per video). GPU MTCNN under multiple worker processes
is unverified in this environment (no GPU here) — CPU is the
tested/recommended device for parallel runs; see HANDOFF.md.

Do NOT run this without real raw video files in place — see HANDOFF.md.

Usage::

    python -m src.data.preprocess_raw \\
        --input-dir /path/to/raw_dfdc_real_videos \\
        --output-dir Dataset/ \\
        --dataset-name dfdc --label real \\
        --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

from src import config

logger = logging.getLogger(__name__)

VALID_DATASET_NAMES = ("dfdc", "ffpp", "celebdf")
VALID_LABELS = ("real", "fake")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")

FACE_ONLY_DIRNAMES: dict[tuple[str, str], str] = {
    ("dfdc", "real"): "DFDC_REAL_Face_only_data",
    ("dfdc", "fake"): "DFDC_FAKE_Face_only_data",
    ("ffpp", "real"): "FF_Face_only_data",
    ("ffpp", "fake"): "FF_Face_only_data",
    ("celebdf", "real"): "Celeb_real_face_only",
    ("celebdf", "fake"): "Celeb_fake_face_only",
}

FFPP_METADATA_FILENAME = "metadata.csv"
DEFAULT_WORKERS = 4


def resolve_face_only_dir(output_root: Path, dataset_name: str, label: str) -> Path:
    """Map (dataset_name, label) to the exact folder name build_splits.py
    reads from — see this module's docstring and HANDOFF.md "Dataset
    Required" for the full table."""
    key = (dataset_name, label)
    if key not in FACE_ONLY_DIRNAMES:
        raise ValueError(
            f"Unknown (dataset_name, label) combination: {key}. "
            f"Expected dataset_name in {VALID_DATASET_NAMES} and label in {VALID_LABELS}."
        )
    return output_root / FACE_ONLY_DIRNAMES[key]


def build_mtcnn(device: str = "cpu") -> MTCNN:
    """Same MTCNN configuration as src/data/face_crop.py's
    crop_faces_from_frames — kept in sync by hand (that module is out of
    scope for this change): keep_all=False, select_largest=True."""
    return MTCNN(keep_all=False, select_largest=True, device=device)


def discover_videos(input_dir: Path) -> list[Path]:
    return sorted(p for ext in VIDEO_EXTENSIONS for p in input_dir.glob(f"*{ext}"))


def _sample_frame_indices(total_frames: int, n_frames: int) -> tuple[np.ndarray, bool]:
    """Evenly-spaced, de-duplicated frame indices across [0, total_frames).

    Returns (indices, was_short). ``was_short`` is True when the video had
    fewer frames than requested — the caller must log this loudly rather
    than silently sampling duplicate frames to pad out to n_frames (which is
    what naive ``np.linspace(0, total-1, n_frames)`` would otherwise produce
    when total < n_frames).
    """
    if total_frames <= 0:
        return np.array([], dtype=int), False
    actual_n = min(n_frames, total_frames)
    indices = np.unique(np.linspace(0, total_frames - 1, actual_n, dtype=int))
    return indices, actual_n < n_frames


@dataclass
class VideoResult:
    video_path: str
    status: str  # "saved" | "already_done" | "corrupt" | "no_faces_at_all"
    faces_saved: int
    frames_sampled: int
    frames_missing_face: int
    detail: str = ""


def process_video(video_path: Path, out_dir: Path, mtcnn: MTCNN, n_frames: int) -> VideoResult:
    """Sample, face-crop, and save one video. Never raises for expected
    failure modes (corrupt file, no face, short video) — those are reported
    in the returned VideoResult's status/detail for the caller to log."""
    if (out_dir / "frame_00.jpg").exists():
        return VideoResult(str(video_path), "already_done", 0, 0, 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return VideoResult(str(video_path), "corrupt", 0, 0, 0, "cv2.VideoCapture could not open file")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return VideoResult(str(video_path), "corrupt", 0, 0, 0, "no readable frames (CAP_PROP_FRAME_COUNT <= 0)")

        indices, was_short = _sample_frame_indices(total, n_frames)

        frames_missing_face = 0
        pending_saves: list[tuple[int, np.ndarray]] = []

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                frames_missing_face += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                face = mtcnn(Image.fromarray(rgb))
            except Exception as exc:  # noqa: BLE001 - MTCNN can raise on odd frames; treat as no-face, don't abort the whole video
                logger.debug("MTCNN error on frame %d of %s: %s", i, video_path, exc)
                face = None

            if face is None:
                frames_missing_face += 1
                continue

            arr = face.permute(1, 2, 0).clamp(0, 255).byte().numpy().astype(np.uint8)
            pending_saves.append((i, arr))
    finally:
        cap.release()

    short_note = f"video shorter than requested n_frames={n_frames} (had {total} frames, sampled {len(indices)})" if was_short else ""

    if not pending_saves:
        detail = f"no face detected in any of {len(indices)} sampled frames"
        if short_note:
            detail += f"; {short_note}"
        return VideoResult(str(video_path), "no_faces_at_all", 0, len(indices), frames_missing_face, detail)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in pending_saves:
        Image.fromarray(arr).save(out_dir / f"frame_{i:02d}.jpg")

    detail_parts = [p for p in (short_note, f"{frames_missing_face}/{len(indices)} sampled frames had no detectable face" if frames_missing_face else "") if p]
    return VideoResult(str(video_path), "saved", len(pending_saves), len(indices), frames_missing_face, "; ".join(detail_parts))


# --------------------------------------------------------------------------
# Multiprocessing worker plumbing
# --------------------------------------------------------------------------

_worker_mtcnn: MTCNN | None = None
_worker_device = "cpu"


def _init_worker(device: str) -> None:
    global _worker_mtcnn, _worker_device
    _worker_device = device
    _worker_mtcnn = build_mtcnn(device=device)


def _process_one(task: tuple[str, str, int]) -> VideoResult:
    global _worker_mtcnn
    video_path_str, out_dir_str, n_frames = task
    if _worker_mtcnn is None:  # e.g. pool used without the initializer (shouldn't happen via main(), but defensive)
        _worker_mtcnn = build_mtcnn(device=_worker_device)
    return process_video(Path(video_path_str), Path(out_dir_str), _worker_mtcnn, n_frames)


# --------------------------------------------------------------------------
# FF++ metadata.csv (label recovery — see module docstring)
# --------------------------------------------------------------------------


def update_ffpp_metadata(metadata_csv: Path, results: list[VideoResult], label: str) -> None:
    """Merge newly-labeled video stems into FF_Face_only_data/metadata.csv,
    de-duplicating against whatever's already there. Only videos that now
    have (or already had) an output folder get a row — build_splits.py only
    ever lists directories that exist, so anything else would be inert."""
    label_value = 1 if label == "fake" else 0

    existing: dict[str, int] = {}
    if metadata_csv.exists():
        with open(metadata_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["video"]] = int(row["label"])

    for r in results:
        stem = Path(r.video_path).stem
        if r.status == "saved":
            existing[stem] = label_value  # this run is authoritative for freshly-processed videos
        elif r.status == "already_done":
            existing.setdefault(stem, label_value)  # don't overwrite a differing label from a prior run

    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "label"])
        for video, lbl in sorted(existing.items()):
            writer.writerow([video, lbl])


# --------------------------------------------------------------------------
# Failures log
# --------------------------------------------------------------------------

_FAILURE_STATUSES = ("corrupt", "no_faces_at_all")


def _is_notable(result: VideoResult) -> bool:
    """Anything the task's failure-handling requirement calls out explicitly:
    corrupt/unreadable video, zero faces detected, or a short-video/partial
    detection anomaly on an otherwise-saved video."""
    return result.status in _FAILURE_STATUSES or bool(result.detail)


def write_failures_log(failures_log: Path, results: list[VideoResult]) -> int:
    failures_log.parent.mkdir(parents=True, exist_ok=True)
    notable = [r for r in results if _is_notable(r)]
    if not notable:
        return 0

    with open(failures_log, "a", encoding="utf-8") as f:
        for r in notable:
            line = f"{datetime.now(UTC).isoformat()}\t{r.status}\t{r.video_path}\t{r.detail}\n"
            f.write(line)
            logger.warning("%s: %s (%s)", r.video_path, r.status, r.detail or "no detail")

    return len(notable)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of raw .mp4/.mov/.avi files, one dataset+label per run")
    parser.add_argument("--output-dir", type=Path, default=config.DATASET_DIR, help="Root that *_Face_only_data folders are written under (default: src.config.DATASET_DIR)")
    parser.add_argument("--dataset-name", choices=VALID_DATASET_NAMES, required=True)
    parser.add_argument("--label", choices=VALID_LABELS, required=True)
    parser.add_argument("--n-frames", type=int, default=config.FRAMES_PER_VIDEO, help="Frames sampled per video (default: src.config.FRAMES_PER_VIDEO)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel worker processes (1 = run serially, no multiprocessing)")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda' for MTCNN. CPU is the tested/recommended device for workers > 1 — see HANDOFF.md.")
    parser.add_argument("--failures-log", type=Path, default=None, help="Default: <config.LOG_DIR>/preprocessing_failures.log")
    args = parser.parse_args(argv)

    if not args.input_dir.exists():
        logger.error("--input-dir not found: %s", args.input_dir)
        return 2

    videos = discover_videos(args.input_dir)
    if not videos:
        logger.error("No videos with extensions %s found under %s", VIDEO_EXTENSIONS, args.input_dir)
        return 2
    logger.info("Found %d videos in %s (dataset=%s, label=%s)", len(videos), args.input_dir, args.dataset_name, args.label)

    face_only_dir = resolve_face_only_dir(args.output_dir, args.dataset_name, args.label)
    face_only_dir.mkdir(parents=True, exist_ok=True)

    failures_log = args.failures_log or (config.LOG_DIR / "preprocessing_failures.log")

    tasks = [(str(v), str(face_only_dir / v.stem), args.n_frames) for v in videos]

    if args.workers <= 1:
        mtcnn = build_mtcnn(device=args.device)
        results = [process_video(Path(v), Path(o), mtcnn, n) for v, o, n in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(args.device,)) as pool:
            results = pool.map(_process_one, tasks)

    if args.dataset_name == "ffpp":
        update_ffpp_metadata(face_only_dir / FFPP_METADATA_FILENAME, results, label=args.label)

    counts = Counter(r.status for r in results)
    logger.info("Done: %s", dict(counts))

    n_notable = write_failures_log(failures_log, results)
    if n_notable:
        logger.warning("%d/%d videos had a notable issue — see %s", n_notable, len(results), failures_log)

    n_total_failures = counts.get("corrupt", 0) + counts.get("no_faces_at_all", 0)
    if n_total_failures == len(results):
        logger.error("Every video in this batch failed (corrupt or no face detected) — check --device/--input-dir.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
