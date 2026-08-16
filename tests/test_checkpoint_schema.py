"""Save/load round-trip tests for the shared checkpoint schema
(src/models/checkpoint.py). This is the module that fixes the original
ensemble-checkpoint filename/shape mismatch bug — these tests exist
specifically to make sure that bug class can't come back silently: an
old-shaped checkpoint (or one at the wrong path) must fail loudly, not
fall through to a fabricated/random result.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from src.models.checkpoint import (
    ENSEMBLE_FEATURE_NAMES,
    CheckpointNotFoundError,
    CheckpointSchemaError,
    load_ensemble_checkpoint,
    load_torch_checkpoint,
    save_ensemble_checkpoint,
    save_torch_checkpoint,
)
from src.models.spatial import SpatialModel


def test_torch_checkpoint_round_trip(tmp_path: Path):
    model = SpatialModel(pretrained=False)
    path = tmp_path / "spatial_best.pth"

    save_torch_checkpoint(model, path, epoch=3, val_auc=0.91, best_val_auc=0.91)

    loaded_model = SpatialModel(pretrained=False)
    meta = load_torch_checkpoint(loaded_model, path, map_location="cpu")

    assert meta.epoch == 3
    assert meta.val_auc == pytest.approx(0.91)
    for p1, p2 in zip(model.parameters(), loaded_model.parameters(), strict=True):
        assert torch.allclose(p1, p2)


def test_torch_checkpoint_missing_file_raises(tmp_path: Path):
    model = SpatialModel(pretrained=False)
    with pytest.raises(CheckpointNotFoundError):
        load_torch_checkpoint(model, tmp_path / "does_not_exist.pth", map_location="cpu")


def test_torch_checkpoint_wrong_shape_raises(tmp_path: Path):
    """A file that exists but isn't our schema (e.g. a bare state_dict saved
    by some other tool) must raise, not silently half-load."""
    model = SpatialModel(pretrained=False)
    path = tmp_path / "bare_state_dict.pth"
    torch.save(model.state_dict(), path)  # NOT wrapped in {"model_state": ...}

    with pytest.raises(CheckpointSchemaError):
        load_torch_checkpoint(model, path, map_location="cpu")


def test_ensemble_checkpoint_round_trip(tmp_path: Path):
    calibrator = LogisticRegression().fit([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], [0, 1])

    saved_path = save_ensemble_checkpoint(calibrator, tmp_path)
    assert saved_path.name == "ensemble_final.joblib"

    loaded = load_ensemble_checkpoint(saved_path)
    assert loaded.feature_names == ENSEMBLE_FEATURE_NAMES
    prob = loaded.predict_proba_fake([[1, 1, 1, 1, 1]])
    assert 0.0 <= prob <= 1.0


def test_ensemble_checkpoint_missing_file_raises(tmp_path: Path):
    with pytest.raises(CheckpointNotFoundError):
        load_ensemble_checkpoint(tmp_path / "ensemble_final.joblib")


def test_ensemble_checkpoint_rejects_bare_object(tmp_path: Path):
    """This is the exact original bug, reproduced as a regression test: a
    bare (un-wrapped) sklearn object saved directly via joblib.dump, which
    is what the old training notebooks did. Loading it must raise a clear
    error, not TypeError: 'CalibratedClassifierCV' object is not
    subscriptable deep inside app code."""
    calibrator = LogisticRegression().fit([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], [0, 1])
    bad_path = tmp_path / "ensemble_final.joblib"
    joblib.dump(calibrator, bad_path)  # bare object, no {"calibrator": ...} wrapper

    with pytest.raises(CheckpointSchemaError):
        load_ensemble_checkpoint(bad_path)


def test_ensemble_checkpoint_rejects_mismatched_feature_order(tmp_path: Path):
    calibrator = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])
    path = tmp_path / "ensemble_final.joblib"
    joblib.dump({"calibrator": calibrator, "feature_names": ["a", "b"], "schema_version": "1.0"}, path)

    with pytest.raises(CheckpointSchemaError):
        load_ensemble_checkpoint(path)
