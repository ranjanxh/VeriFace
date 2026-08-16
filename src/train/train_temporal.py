"""Train the temporal (video-level) Bi-LSTM + attention classifier over
per-frame spatial embeddings.

Consolidated from ``archive/legacy_src_process_data/train_temporal.py``
("06 - Train temporal model"). Behavior unchanged (packed variable-length
sequences, gradient clipping, best-val-AUC checkpointing) except checkpoints
now go through ``src.models.checkpoint`` and paths come from ``src.config``.

Do NOT run this without extracted embeddings in place — see HANDOFF.md.
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import config
from src.eval.metrics import compute_auc as safe_auc
from src.models.checkpoint import save_torch_checkpoint
from src.models.temporal import TemporalModel

logger = logging.getLogger(__name__)

SEED_DEFAULT = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_label_from_stem(stem: str, labels_map: dict[str, int]) -> int:
    if stem in labels_map:
        return int(labels_map[stem])
    for k, v in labels_map.items():
        if stem in k:
            return int(v)
    raise KeyError(stem)


class VideoEmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir: Path, split: str, labels_map: dict[str, int]) -> None:
        self.root = embeddings_dir / split
        self.items = sorted(self.root.glob("*.npy"))
        self.labels_map = labels_map

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path = self.items[idx]
        emb = np.load(path).astype(np.float32)
        label = get_label_from_stem(path.stem, self.labels_map)
        return torch.from_numpy(emb), torch.tensor(label, dtype=torch.float32), path.stem


def collate_fn(batch):
    seqs, labels, stems = zip(*batch, strict=True)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    maxlen = int(lengths.max())
    feat_dim = seqs[0].shape[1]

    out = torch.zeros(len(seqs), maxlen, feat_dim)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s

    return out, lengths, torch.stack(labels), list(stems)


def run_epoch(model, loader, device, criterion, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    preds, labels_all = [], []

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for seqs, lengths, labels, _ in tqdm(loader, desc="train" if is_train else "val"):
            seqs, lengths, labels = seqs.to(device), lengths.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits, _ = model(seqs, lengths)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            loss_sum += loss.item() * seqs.size(0)
            preds.append(torch.sigmoid(logits).detach().cpu())
            labels_all.append(labels.cpu())

    preds_np = torch.cat(preds).numpy() if preds else np.array([])
    labels_np = torch.cat(labels_all).numpy() if labels_all else np.array([])
    avg_loss = loss_sum / max(len(loader.dataset), 1)
    auc = safe_auc(labels_np, preds_np)
    return avg_loss, auc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, default=config.EMBEDDINGS_DIR)
    parser.add_argument("--labels-json", type=Path, default=config.LABELS_JSON)
    parser.add_argument("--checkpoint-dir", type=Path, default=config.TEMPORAL_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args(argv)

    if not args.embeddings_dir.exists() or not args.labels_json.exists():
        logger.error(
            "Expected inputs not found (embeddings-dir=%s exists=%s, labels-json=%s exists=%s). "
            "Run src.data.extract_embeddings first — see HANDOFF.md.",
            args.embeddings_dir, args.embeddings_dir.exists(), args.labels_json, args.labels_json.exists(),
        )
        return 2

    set_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    labels_map = json.loads(args.labels_json.read_text(encoding="utf-8"))

    train_ds = VideoEmbeddingDataset(args.embeddings_dir, "train", labels_map)
    val_ds = VideoEmbeddingDataset(args.embeddings_dir, "val", labels_map)
    if len(train_ds) == 0:
        logger.error("No training embeddings found under %s/train", args.embeddings_dir)
        return 2

    feat_dim = np.load(train_ds.items[0]).shape[1]
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

    model = TemporalModel(feat_dim=feat_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    last_ckpt = args.checkpoint_dir / "temporal_last.pth"

    for epoch in range(args.epochs):
        train_loss, train_auc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_auc = run_epoch(model, val_loader, device, criterion)
        scheduler.step(val_auc)

        save_torch_checkpoint(model, last_ckpt, epoch=epoch, val_auc=val_auc, best_val_auc=best_val_auc, optimizer=optimizer)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_torch_checkpoint(
                model,
                args.checkpoint_dir / "temporal_best_valAUC.pth",
                epoch=epoch,
                val_auc=val_auc,
                best_val_auc=best_val_auc,
            )

        logger.info("Epoch %d: train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f", epoch, train_loss, train_auc, val_loss, val_auc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
