"""Extract per-frame spatial embeddings for every video: writes
``embeddings/<split>/<video_stem>.npy`` (shape ``[T, 1536]``).

Consolidated from ``archive/notebooks/extract_embeddings.ipynb``. These
embeddings are the input to the temporal model (``src/train/train_temporal.py``)
and the ensemble feature builder (``src/models/ensemble.py``).

Do NOT run this without a trained spatial checkpoint + extracted frames in
place — see HANDOFF.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src import config
from src.models.checkpoint import load_torch_checkpoint
from src.models.spatial import SpatialModel

logger = logging.getLogger(__name__)

IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 16

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def _load_frame_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _PREPROCESS(img)


@torch.no_grad()
def extract_for_video(
    model: SpatialModel,
    device: torch.device,
    frames_dir: Path,
    out_path: Path,
    batch_size: int,
    overwrite: bool,
) -> str:
    if out_path.exists() and not overwrite:
        return "exists"
    if not frames_dir.exists():
        return "missing_frames"

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return "no_frames"

    tensors = [_load_frame_tensor(p) for p in frame_files]
    embeddings = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size], dim=0).to(device)
        feats = model.embed(batch)
        embeddings.append(feats.cpu().numpy())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.concatenate(embeddings, axis=0).astype(np.float32))
    return "saved"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=list(config.SPLITS))
    parser.add_argument("--frames-dir", type=Path, default=config.FRAMES_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.EMBEDDINGS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=config.SPATIAL_BEST_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test-count", type=int, default=0, help="If >0, only process this many videos per split (for a quick sanity run).")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialModel(pretrained=False).to(device)
    meta = load_torch_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    logger.info("Loaded spatial checkpoint (epoch=%s, val_auc=%.4f)", meta.epoch, meta.val_auc)

    manifest: dict[str, dict[str, int]] = {}
    for split in args.splits:
        split_dir = args.frames_dir / split
        if not split_dir.exists():
            logger.warning("Skipping missing split dir: %s", split_dir)
            continue

        stems = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if args.smoke_test_count > 0:
            stems = stems[: args.smoke_test_count]

        counts = {"saved": 0, "exists": 0, "missing_frames": 0, "no_frames": 0}
        logger.info("Extracting embeddings: split=%s videos=%d", split, len(stems))
        for stem in stems:
            status = extract_for_video(
                model,
                device,
                split_dir / stem,
                args.out_dir / split / f"{stem}.npy",
                args.batch_size,
                args.overwrite,
            )
            counts[status] = counts.get(status, 0) + 1
        logger.info("[%s] done: %s", split, counts)
        manifest[split] = counts

    logger.info("Summary: %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
