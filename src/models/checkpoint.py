"""Shared checkpoint schema for all three models (spatial, temporal, ensemble).

This module exists to close a real bug found in the original codebase: the
ensemble-training notebooks saved a bare, un-wrapped scikit-learn object via
``joblib.dump(ensemble_clf, "ensemble_best.pkl")``, while ``app.py`` expected
a dict shaped like ``{"calibrator": ...}`` loaded from a *different* filename
(``ensemble_final.joblib``). Loading a real trained checkpoint into the app
would raise ``TypeError: 'CalibratedClassifierCV' object is not
subscriptable``. See KNOWLEDGE_BASE.md / HANDOFF.md for the original audit.

The fix is structural, not cosmetic: training code (``src/train/*``) and
serving code (``app/*``, ``src/inference/*``) both import the save/load
functions from *this one module*. There is no longer a second place where the
on-disk shape of a checkpoint is decided, so the two sides cannot drift apart
silently again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TORCH_CHECKPOINT_SCHEMA_VERSION = "1.0"
ENSEMBLE_CHECKPOINT_SCHEMA_VERSION = "1.0"
ENSEMBLE_CHECKPOINT_FILENAME = "ensemble_final.joblib"

# Order matters: this is the exact feature vector layout the ensemble
# calibrator was (and must continue to be) trained on.
ENSEMBLE_FEATURE_NAMES: tuple[str, ...] = (
    "spatial_mean",
    "spatial_max",
    "spatial_std",
    "spatial_top3_mean",
    "temporal_logit",
)


class CheckpointError(Exception):
    """Base class for all checkpoint-related failures."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint path does not exist on disk."""


class CheckpointSchemaError(CheckpointError):
    """Raised when a checkpoint exists but does not match the expected schema."""


class CheckpointLoadError(CheckpointError):
    """Raised when a checkpoint matches the schema but fails to load into a model."""


# --------------------------------------------------------------------------
# Torch checkpoints (spatial / temporal models)
# --------------------------------------------------------------------------


@dataclass
class TorchCheckpointMeta:
    """Metadata stored alongside every spatial/temporal checkpoint."""

    epoch: int
    val_auc: float
    best_val_auc: float
    schema_version: str = TORCH_CHECKPOINT_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


def save_torch_checkpoint(
    model: nn.Module,
    path: Path | str,
    *,
    epoch: int,
    val_auc: float,
    best_val_auc: float,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a spatial/temporal checkpoint in the one shared on-disk schema.

    Schema (dict keys): ``model_state``, ``optimizer_state`` (optional),
    ``epoch``, ``val_auc``, ``best_val_auc``, ``schema_version``.
    Tensors are moved to CPU before saving so checkpoints are portable across
    machines/devices, matching the original training scripts' intent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "epoch": epoch,
        "val_auc": val_auc,
        "best_val_auc": best_val_auc,
        "schema_version": TORCH_CHECKPOINT_SCHEMA_VERSION,
    }
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        cpu_state = {}
        for k, v in opt_state.get("state", {}).items():
            cpu_state[k] = {
                sk: (sv.cpu() if isinstance(sv, torch.Tensor) else sv)
                for sk, sv in v.items()
            }
        payload["optimizer_state"] = {
            "state": cpu_state,
            "param_groups": opt_state.get("param_groups", []),
        }
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)
    logger.info("Saved torch checkpoint to %s (epoch=%s, val_auc=%.4f)", path, epoch, val_auc)
    return path


def load_torch_checkpoint(
    model: nn.Module,
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
) -> TorchCheckpointMeta:
    """Load a spatial/temporal checkpoint into ``model`` in place.

    Raises ``CheckpointNotFoundError`` / ``CheckpointSchemaError`` /
    ``CheckpointLoadError`` instead of ever silently leaving ``model`` with
    randomly-initialized weights. Callers (e.g. the Streamlit app) are
    expected to catch these and surface a visible failure state rather than
    serve predictions from an untrained model.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointNotFoundError(f"Checkpoint not found: {path}")

    try:
        # weights_only=False: recent torch defaults to True, which is the right
        # security posture for untrusted checkpoints, but our payload is a plain
        # dict of tensors/ints/floats produced by save_torch_checkpoint() above,
        # from checkpoints we trust (our own training runs). Explicit, not an
        # oversight.
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - re-raised with context, not swallowed
        raise CheckpointLoadError(f"Failed to deserialize checkpoint at {path}: {exc}") from exc

    if not isinstance(payload, dict) or "model_state" not in payload:
        raise CheckpointSchemaError(
            f"Checkpoint at {path} does not match the expected schema "
            f"(dict with a 'model_state' key). Got type={type(payload)!r} "
            f"keys={list(payload.keys()) if isinstance(payload, dict) else 'n/a'}."
        )

    try:
        model.load_state_dict(payload["model_state"], strict=strict)
    except Exception as exc:  # noqa: BLE001 - re-raised with context, not swallowed
        raise CheckpointLoadError(
            f"Checkpoint at {path} matched the schema but failed to load into "
            f"{type(model).__name__} (strict={strict}): {exc}"
        ) from exc

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return TorchCheckpointMeta(
        epoch=payload.get("epoch", -1),
        val_auc=payload.get("val_auc", float("nan")),
        best_val_auc=payload.get("best_val_auc", float("nan")),
        schema_version=payload.get("schema_version", "unknown"),
        extra=payload.get("extra", {}),
    )


# --------------------------------------------------------------------------
# Ensemble checkpoint (calibrated logistic-regression fusion head)
# --------------------------------------------------------------------------


@dataclass
class EnsembleCheckpoint:
    """In-memory representation of a loaded ensemble checkpoint."""

    calibrator: Any  # fitted sklearn estimator exposing predict_proba(X)
    feature_names: tuple[str, ...]
    schema_version: str = ENSEMBLE_CHECKPOINT_SCHEMA_VERSION

    def predict_proba_fake(self, features: list[list[float]] | Any) -> float:
        """Return P(fake) for a single feature row built in ``feature_names`` order."""
        return float(self.calibrator.predict_proba(features)[0, 1])


def save_ensemble_checkpoint(
    calibrator: Any,
    out_dir: Path | str,
    *,
    feature_names: tuple[str, ...] = ENSEMBLE_FEATURE_NAMES,
    filename: str = ENSEMBLE_CHECKPOINT_FILENAME,
) -> Path:
    """Save the calibrated ensemble classifier in the one shared schema.

    On-disk shape: ``joblib``-serialized dict
    ``{"calibrator": <fitted estimator>, "feature_names": [...], "schema_version": "1.0"}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    payload = {
        "calibrator": calibrator,
        "feature_names": list(feature_names),
        "schema_version": ENSEMBLE_CHECKPOINT_SCHEMA_VERSION,
    }
    joblib.dump(payload, path)
    logger.info("Saved ensemble checkpoint to %s", path)
    return path


def load_ensemble_checkpoint(path: Path | str) -> EnsembleCheckpoint:
    """Load the ensemble checkpoint, validating it matches the shared schema.

    Raises ``CheckpointNotFoundError`` / ``CheckpointSchemaError`` instead of
    ever silently falling back to a fabricated/random classifier.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointNotFoundError(f"Ensemble checkpoint not found: {path}")

    try:
        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context, not swallowed
        raise CheckpointLoadError(f"Failed to deserialize ensemble checkpoint at {path}: {exc}") from exc

    if not isinstance(payload, dict) or "calibrator" not in payload:
        raise CheckpointSchemaError(
            f"Ensemble checkpoint at {path} does not match the expected schema "
            f"(dict with a 'calibrator' key). Got type={type(payload)!r}. "
            "This is exactly the drift this module was written to prevent — "
            "make sure it was saved via save_ensemble_checkpoint()."
        )

    feature_names = tuple(payload.get("feature_names", ENSEMBLE_FEATURE_NAMES))
    if feature_names != ENSEMBLE_FEATURE_NAMES:
        raise CheckpointSchemaError(
            f"Ensemble checkpoint at {path} was trained with feature order "
            f"{feature_names}, but the current pipeline expects "
            f"{ENSEMBLE_FEATURE_NAMES}. Refusing to load to avoid silently "
            "feeding features to the calibrator in the wrong order."
        )

    return EnsembleCheckpoint(
        calibrator=payload["calibrator"],
        feature_names=feature_names,
        schema_version=payload.get("schema_version", "unknown"),
    )
