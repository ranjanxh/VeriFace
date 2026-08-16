"""Shape/contract tests for the three model definitions. CPU-only, tiny
random tensors — no GPU, no trained weights, no dataset required. These
would have caught, for example, a mismatched feature dimension between the
spatial backbone and the temporal model's expected input size.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.ensemble import ENSEMBLE_FEATURE_NAMES, build_ensemble_features
from src.models.spatial import SpatialModel
from src.models.temporal import TemporalModel


def test_spatial_model_forward_shape():
    model = SpatialModel(pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4,)


def test_spatial_model_embed_matches_feat_dim():
    model = SpatialModel(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = model.embed(x)
    assert feats.shape == (2, model.feat_dim)


def test_spatial_model_head_consumes_embeddings_directly():
    """The ensemble/embedding pipeline applies model.head() directly to
    precomputed embeddings (skipping the backbone) — this must stay valid."""
    model = SpatialModel(pretrained=False)
    model.eval()
    fake_embeddings = torch.randn(5, model.feat_dim)
    with torch.no_grad():
        logits = model.head(fake_embeddings).squeeze(1)
    assert logits.shape == (5,)


def test_temporal_model_forward_no_padding():
    model = TemporalModel(feat_dim=16, hidden=8, layers=1)
    model.eval()
    x = torch.randn(1, 10, 16)
    with torch.no_grad():
        logits, attn = model(x)
    assert logits.shape == (1,)
    assert attn.shape == (1, 10)
    assert torch.allclose(attn.sum(dim=1), torch.ones(1), atol=1e-5)


def test_temporal_model_forward_with_lengths_matches_unpadded():
    """Batch size 1, no padding, called with an explicit lengths tensor must
    be numerically equivalent to calling without lengths at all — this is
    the invariant that let us merge the training and inference code paths
    into one model class (see src/models/temporal.py docstring)."""
    torch.manual_seed(0)
    model = TemporalModel(feat_dim=16, hidden=8, layers=1)
    model.eval()
    x = torch.randn(1, 10, 16)
    lengths = torch.tensor([10])

    with torch.no_grad():
        logits_no_lengths, _ = model(x)
        logits_with_lengths, _ = model(x, lengths)

    assert torch.allclose(logits_no_lengths, logits_with_lengths, atol=1e-5)


def test_temporal_model_padding_is_masked_out():
    """A padded tail must not influence the pooled result — enforced by the
    combination of pack_padded_sequence (the LSTM never sees timesteps past
    `lengths`) and AttentionPool's mask (belt-and-suspenders for any path
    where padding does reach the pooling step)."""
    torch.manual_seed(0)
    model = TemporalModel(feat_dim=16, hidden=8, layers=1)
    model.eval()

    real = torch.randn(1, 6, 16)
    padded = torch.cat([real, torch.randn(1, 4, 16)], dim=1)  # 4 frames of "garbage" padding
    lengths = torch.tensor([6])

    with torch.no_grad():
        logits_real_only, _ = model(real, torch.tensor([6]))
        logits_padded, _ = model(padded, lengths)

    assert torch.allclose(logits_real_only, logits_padded, atol=1e-5)


def test_build_ensemble_features_shape_and_order():
    spatial_probs = torch.tensor([0.1, 0.9, 0.4, 0.6, 0.95]).numpy()
    row = build_ensemble_features(spatial_probs, temporal_logit=1.23)
    assert row.shape == (1, len(ENSEMBLE_FEATURE_NAMES))
    # row is float32 (matches model dtypes elsewhere in the pipeline), so
    # compare with a tolerance rather than exact equality against the
    # float64 literal 1.23.
    assert row[0, -1] == pytest.approx(1.23, abs=1e-6)  # temporal_logit is always the last feature


def test_build_ensemble_features_rejects_empty_input():
    with pytest.raises(ValueError):
        build_ensemble_features(np.array([]), temporal_logit=0.0)
