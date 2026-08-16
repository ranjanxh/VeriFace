"""Train the ensemble (fusion) stage: a calibrated logistic regression over
5 hand-engineered spatial+temporal features, one row per video.

Consolidates two notebooks that had diverged in the archived codebase:
``train_ensemble.ipynb`` (a more defensive version with bad-data logging and
adaptive calibration-fold count) and ``train_ensemble3.ipynb`` (a simpler,
more readable version). This keeps the defensive behavior and the simpler
structure.

Also fixes the concrete bug this whole restructure was partly motivated by:
both original notebooks saved the raw sklearn object via
``joblib.dump(ensemble_clf, "ensemble_best.pkl")``, while ``app.py`` expected
a ``{"calibrator": ...}`` dict at a *different* filename
(``ensemble_final.joblib``). This script now saves through
``src.models.checkpoint.save_ensemble_checkpoint``, which is the same
function ``src/inference/pipeline.py`` and the app load through — the
contract cannot drift apart again silently.

On successful completion, writes the ``accuracy`` section of
``results/metrics.json`` (test-split AUC + accuracy) via
``src.eval.metrics.update_metrics_section`` — this is the mechanism by which
the app's "Accuracy: pending real measurement" badge gets a real number,
once this is actually run on real hardware with real data.

Do NOT run this without extracted embeddings + trained spatial/temporal
checkpoints in place — see HANDOFF.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src import config
from src.eval.metrics import compute_auc, update_metrics_section
from src.models.checkpoint import load_torch_checkpoint, save_ensemble_checkpoint
from src.models.ensemble import build_ensemble_features
from src.models.spatial import SpatialModel
from src.models.temporal import TemporalModel

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_CV = 5
DEFAULT_LR_MAX_ITER = 2000


def get_label(stem: str, labels_map: dict[str, int]) -> int:
    if stem in labels_map:
        return int(labels_map[stem])
    for k, v in labels_map.items():
        if stem in k:
            return int(v)
    raise KeyError(f"Label not found for {stem}")


@torch.no_grad()
def build_features_for_split(
    split: str,
    embeddings_dir: Path,
    labels_map: dict[str, int],
    spatial_model: SpatialModel,
    temporal_model: TemporalModel,
    device: torch.device,
    cache_dir: Path,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / f"{split}.npz"
    if cache_path.exists() and not overwrite:
        logger.info("Loading cached features for %s from %s", split, cache_path)
        d = np.load(cache_path)
        return d["X"], d["y"]

    files = sorted((embeddings_dir / split).glob("*.npy"))
    logger.info("Building features for %s (%d videos)", split, len(files))

    rows, labels = [], []
    for path in tqdm(files, desc=f"features:{split}"):
        emb = np.load(path)
        if emb.shape[0] == 0:
            logger.warning("Skipping %s: empty embedding", path)
            continue

        emb_t = torch.from_numpy(emb).float().to(device)
        spatial_probs = torch.sigmoid(spatial_model.head(emb_t)).cpu().numpy()

        lengths = torch.tensor([emb_t.shape[0]], device=device)
        temporal_logit, _ = temporal_model(emb_t.unsqueeze(0), lengths)
        temporal_logit = temporal_logit.item()

        try:
            row = build_ensemble_features(spatial_probs, temporal_logit)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        try:
            label = get_label(path.stem, labels_map)
        except KeyError:
            logger.warning("Skipping %s: no label found", path)
            continue

        rows.append(row[0])
        labels.append(label)

    x = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=x, y=y)
    return x, y


def _fit_calibrated_classifier(x_train: np.ndarray, y_train: np.ndarray, seed: int, requested_cv: int):
    """Adaptive calibration fold count: falls back to a plain (uncalibrated)
    LogisticRegression if the training set is too small to support the
    requested number of stratified folds — carried over from the more
    defensive of the two original notebooks."""
    base_clf = LogisticRegression(max_iter=DEFAULT_LR_MAX_ITER, solver="lbfgs")

    cv = requested_cv
    n_samples = len(y_train)
    while cv > 1 and n_samples < cv * 2:
        cv -= 1

    if cv < 2:
        logger.warning("Training set too small for calibration (n=%d) — using plain LogisticRegression.", n_samples)
        base_clf.fit(x_train, y_train)
        return base_clf

    try:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        calibrated = CalibratedClassifierCV(base_clf, cv=skf, method="sigmoid")
        calibrated.fit(x_train, y_train)
        return calibrated
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"CalibratedClassifierCV failed ({exc}); falling back to plain LogisticRegression.",
            stacklevel=2,
        )
        base_clf.fit(x_train, y_train)
        return base_clf


def evaluate(calibrator, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {"n": 0, "auc": float("nan"), "accuracy": float("nan")}
    probs = calibrator.predict_proba(x)[:, 1]
    preds = (probs > 0.5).astype(int)
    return {
        "n": int(len(y)),
        "auc": compute_auc(y, probs),
        "accuracy": float((preds == y).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", type=Path, default=config.EMBEDDINGS_DIR)
    parser.add_argument("--labels-json", type=Path, default=config.LABELS_JSON)
    parser.add_argument("--spatial-checkpoint", type=Path, default=config.SPATIAL_BEST_CHECKPOINT)
    parser.add_argument("--temporal-checkpoint", type=Path, default=config.TEMPORAL_BEST_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=config.ENSEMBLE_CHECKPOINT_DIR)
    parser.add_argument("--feature-cache-dir", type=Path, default=config.ROOT / "ensemble_features")
    parser.add_argument("--calibration-cv", type=int, default=DEFAULT_CALIBRATION_CV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-feature-cache", action="store_true")
    args = parser.parse_args(argv)

    for label, path in [("embeddings-dir", args.embeddings_dir), ("labels-json", args.labels_json), ("spatial-checkpoint", args.spatial_checkpoint), ("temporal-checkpoint", args.temporal_checkpoint)]:
        if not path.exists():
            logger.error("Required input --%s not found: %s. See HANDOFF.md.", label, path)
            return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_map = json.loads(args.labels_json.read_text(encoding="utf-8"))

    spatial_model = SpatialModel(pretrained=False).to(device)
    load_torch_checkpoint(spatial_model, args.spatial_checkpoint, map_location=device)
    spatial_model.eval()

    sample_emb = np.load(next((args.embeddings_dir / "train").glob("*.npy")))
    temporal_model = TemporalModel(feat_dim=sample_emb.shape[1]).to(device)
    load_torch_checkpoint(temporal_model, args.temporal_checkpoint, map_location=device)
    temporal_model.eval()

    splits = {}
    for split in config.SPLITS:
        splits[split] = build_features_for_split(
            split, args.embeddings_dir, labels_map, spatial_model, temporal_model, device,
            args.feature_cache_dir, overwrite=args.overwrite_feature_cache,
        )
        logger.info("%s: %d feature rows", split, len(splits[split][1]))

    x_train, y_train = splits["train"]
    if len(y_train) == 0:
        logger.error("No training samples available — cannot train ensemble.")
        return 2

    calibrator = _fit_calibrated_classifier(x_train, y_train, args.seed, args.calibration_cv)

    results = {split: evaluate(calibrator, x, y) for split, (x, y) in splits.items()}
    for split, r in results.items():
        logger.info("%s: n=%d auc=%.4f accuracy=%.4f", split, r["n"], r["auc"], r["accuracy"])

    save_ensemble_checkpoint(calibrator, args.out_dir)

    update_metrics_section(
        "accuracy",
        {
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "train": results["train"],
            "val": results["val"],
            "test": results["test"],
            "note": (
                "Populated by src/train/train_ensemble.py. See HANDOFF.md 'Known "
                "Limitations' regarding possible identity leakage inflating these "
                "numbers if src/eval/leakage_check.py was not run and clean beforehand."
            ),
        },
    )
    logger.info("Wrote accuracy section of %s", config.METRICS_FILE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
