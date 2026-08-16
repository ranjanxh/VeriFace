"""MTCNN face cropping utility: ``frames_root/<video_stem>/*.jpg`` ->
``out_root/<video_stem>/*_face.jpg``.

Cleaned up from ``archive/legacy_src_process_data/face_crop.py``. Its exact
place in the overall pipeline is not fully documented in the original
codebase — see HANDOFF.md "Dataset Required" for the open question about how
raw dataset videos become the ``*_Face_only_data`` folders that
``src/data/build_splits.py`` expects to already exist. This utility operates
on already-extracted frame images, not raw video files, so it is one
plausible piece of that missing step, not a verified end-to-end replacement
for it.

Do NOT run this without frame images in place — see HANDOFF.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 224


def crop_faces_from_frames(frames_root: Path, out_root: Path, device: str = "cpu") -> dict[str, int]:
    """Runs MTCNN over every ``frames_root/<video_stem>/*.jpg`` and writes the
    largest detected face crop to ``out_root/<video_stem>/<name>_face.jpg``.

    Returns counts: {"processed": N, "saved": N, "no_face": N, "errors": N}.
    """
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
    out_root.mkdir(parents=True, exist_ok=True)

    frames = list(frames_root.glob("*/*.jpg"))
    logger.info("Found %d frames under %s", len(frames), frames_root)

    counts = {"processed": 0, "saved": 0, "no_face": 0, "errors": 0}
    for frame_path in frames:
        counts["processed"] += 1
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open %s: %s", frame_path, exc)
            counts["errors"] += 1
            continue

        try:
            face = mtcnn(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTCNN failed on %s: %s", frame_path, exc)
            counts["errors"] += 1
            continue

        if face is None:
            counts["no_face"] += 1
            continue

        arr = face.permute(1, 2, 0).clamp(0, 255).byte().numpy().astype(np.uint8)
        out_dir = out_root / frame_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out_dir / f"{frame_path.stem}_face.jpg")
        counts["saved"] += 1

    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    args = parser.parse_args(argv)

    if not args.frames_root.exists():
        logger.error("frames-root does not exist: %s", args.frames_root)
        return 2

    counts = crop_faces_from_frames(args.frames_root, args.out_root, device=args.device)
    logger.info("Done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
