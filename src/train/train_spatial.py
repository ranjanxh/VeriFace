"""Train the spatial (frame-level) EfficientNet-B3 classifier.

Consolidated from ``archive/legacy_src_process_data/train_spatial.py``
("04 - Train spatial EfficientNet-B3"). Behavior is unchanged (mixed
precision, resumable, best-val-AUC checkpointing) except:

- checkpoints are now saved/loaded through ``src.models.checkpoint``, so
  their on-disk shape can never silently drift from what
  ``src/inference/pipeline.py`` and the app expect (see that module's
  docstring for the bug this fixes on the ensemble side).
- paths come from ``src.config`` instead of a notebook-local ``Path.cwd()``.
- uses the modern ``torch.amp`` API instead of the deprecated
  ``torch.cuda.amp``.

Do NOT run this without extracted frames + labels.json in place — see
HANDOFF.md for expected runtime (dataset size dependent; not measurable in
this environment).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import config
from src.eval.metrics import compute_auc
from src.models.checkpoint import load_torch_checkpoint, save_torch_checkpoint
from src.models.spatial import SpatialModel

logger = logging.getLogger(__name__)

IMG_SIZE = 224
SEED_DEFAULT = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_label_by_stem(stem: str, labels_map: dict[str, int]) -> int:
    if stem in labels_map:
        return int(labels_map[stem])
    candidates = [v for k, v in labels_map.items() if k.startswith(stem)]
    if len(candidates) == 1:
        return int(candidates[0])
    for k, v in labels_map.items():
        if stem in k:
            return int(v)
    raise KeyError(f"Label for stem '{stem}' not found in labels.json")


class FrameDataset(Dataset):
    """``frames_root/<split>/<video_stem>/frame_*.jpg`` — each frame is one
    training sample, labeled by its parent video's stem."""

    def __init__(self, frames_root: Path, split: str, labels_map: dict[str, int], transform=None) -> None:
        self.root = frames_root / split
        self.transform = transform
        if not self.root.exists():
            raise RuntimeError(f"Frames directory not found: {self.root}")

        items = []
        for video_folder in sorted(self.root.iterdir()):
            if not video_folder.is_dir():
                continue
            stem = video_folder.name
            try:
                label = get_label_by_stem(stem, labels_map)
            except KeyError:
                logger.warning("Skipping %s: no label found", stem)
                continue
            for frame_path in sorted(video_folder.glob("frame_*.jpg")):
                items.append((str(frame_path), label, stem))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, stem = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), stem


def _transforms(train: bool) -> T.Compose:
    ops = [T.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        ]
    ops += [T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    return T.Compose(ops)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_preds, all_labels = [], []

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for imgs, labels, _ in tqdm(loader, desc="train" if is_train else "val"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * imgs.size(0)
            all_preds.append(torch.sigmoid(logits).detach())
            all_labels.append(labels.detach())

    preds = torch.cat(all_preds).cpu().numpy() if all_preds else np.array([])
    labels_np = torch.cat(all_labels).cpu().numpy() if all_labels else np.array([])
    avg_loss = running_loss / max(len(loader.dataset), 1)
    auc = compute_auc(labels_np, preds)
    return avg_loss, auc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", type=Path, default=config.FRAMES_DIR)
    parser.add_argument("--labels-json", type=Path, default=config.LABELS_JSON)
    parser.add_argument("--checkpoint-dir", type=Path, default=config.SPATIAL_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint-dir/spatial_last.pth if present")
    args = parser.parse_args(argv)

    if not args.frames_dir.exists() or not args.labels_json.exists():
        logger.error(
            "Expected inputs not found (frames-dir=%s exists=%s, labels-json=%s exists=%s). "
            "This requires the dataset pipeline to have been run first — see HANDOFF.md.",
            args.frames_dir, args.frames_dir.exists(), args.labels_json, args.labels_json.exists(),
        )
        return 2

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    labels_map = json.loads(args.labels_json.read_text(encoding="utf-8"))

    train_ds = FrameDataset(args.frames_dir, "train", labels_map, transform=_transforms(train=True))
    val_ds = FrameDataset(args.frames_dir, "val", labels_map, transform=_transforms(train=False))
    logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    model = SpatialModel(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    pos_count = sum(labels_map.values())
    neg_count = len(labels_map) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 0
    best_val_auc = 0.0
    last_ckpt = args.checkpoint_dir / "spatial_last.pth"
    if args.resume and last_ckpt.exists():
        meta = load_torch_checkpoint(model, last_ckpt, map_location=device, optimizer=optimizer)
        start_epoch = meta.epoch + 1
        best_val_auc = meta.best_val_auc
        logger.info("Resumed from epoch %d, best_val_auc=%.4f", start_epoch, best_val_auc)

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_auc = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_auc = run_epoch(model, val_loader, device, criterion)
        scheduler.step(val_auc)

        save_torch_checkpoint(model, last_ckpt, epoch=epoch, val_auc=val_auc, best_val_auc=best_val_auc, optimizer=optimizer)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_torch_checkpoint(
                model,
                args.checkpoint_dir / "spatial_best_valAUC.pth",
                epoch=epoch,
                val_auc=val_auc,
                best_val_auc=best_val_auc,
            )
            logger.info("New best model at epoch %d, val_auc=%.4f", epoch, val_auc)

        logger.info(
            "Epoch %d: train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f",
            epoch, train_loss, train_auc, val_loss, val_auc,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
