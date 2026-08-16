"""Ensemble stream: fuses spatial + temporal scores into a final verdict.

The "model" here is a calibrated logistic regression (``CalibratedClassifierCV``
wrapping ``LogisticRegression``, Platt/sigmoid calibration) trained in
``src/train/train_ensemble.py`` over a small, fixed set of hand-engineered
features. This module defines that feature contract in exactly one place —
the original codebase built this 5-element feature vector independently (and
slightly differently) in ``app.py``, ``train_ensemble.ipynb``, and
``train_ensemble3.ipynb``. Checkpoint save/load lives in
``src/models/checkpoint.py``; ``ENSEMBLE_FEATURE_NAMES`` is re-exported from
there so there is a single definition of "what a feature row means".
"""

from __future__ import annotations

import numpy as np

from src.models.checkpoint import ENSEMBLE_FEATURE_NAMES

__all__ = ["ENSEMBLE_FEATURE_NAMES", "build_ensemble_features"]


def build_ensemble_features(spatial_probs: np.ndarray, temporal_logit: float) -> np.ndarray:
    """Build the fixed 5-feature row the ensemble calibrator expects.

    Args:
        spatial_probs: per-frame P(fake) from the spatial model, shape [N].
        temporal_logit: raw (pre-sigmoid) logit from the temporal model.

    Returns:
        A ``(1, 5)`` float32 array in ``ENSEMBLE_FEATURE_NAMES`` order:
        (spatial_mean, spatial_max, spatial_std, spatial_top3_mean, temporal_logit).
    """
    spatial_probs = np.asarray(spatial_probs, dtype=np.float64)
    if spatial_probs.size == 0:
        raise ValueError("build_ensemble_features requires at least one per-frame spatial score")

    top3_mean = (
        np.sort(spatial_probs)[-3:].mean() if spatial_probs.size >= 3 else spatial_probs.max()
    )

    row = np.array(
        [
            [
                spatial_probs.mean(),
                spatial_probs.max(),
                spatial_probs.std(),
                top3_mean,
                float(temporal_logit),
            ]
        ],
        dtype=np.float32,
    )
    assert row.shape == (1, len(ENSEMBLE_FEATURE_NAMES))
    return row
