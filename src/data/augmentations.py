"""Video re-encoding helpers, used to generate compression-augmented training
data (e.g. simulating social-media re-compression). Moved unchanged (aside
from explicit error surfacing) from ``archive/legacy_src_process_data/augmentations.py``.

Requires an ``ffmpeg`` binary on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_QF_TO_CRF = {85: 18, 75: 23}


def reencode_video(input_path: Path | str, output_path: Path | str, crf: int = 23) -> str:
    """Re-encode with libx264 at the given CRF (lower = higher quality, ~17-28 typical)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — required for video re-encoding.")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        "-c:a",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return str(output_path)


def reencode_h264_qf(input_path: Path | str, output_path: Path | str, qf: int = 85) -> str:
    """Convert an approximate 'quality fraction' to a CRF value. This mapping
    is a rough heuristic carried over from the original code (only 75/85 are
    calibrated; anything else falls back to CRF 23) — not a precise
    perceptual-quality model.
    """
    crf = _QF_TO_CRF.get(int(qf), 23)
    return reencode_video(input_path, output_path, crf=crf)
