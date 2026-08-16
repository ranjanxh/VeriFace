"""Temporal stream: video-level sequence classifier over per-frame embeddings.

Bidirectional LSTM + attention pooling over the sequence of per-frame
EfficientNet-B3 embeddings (from ``SpatialModel.embed``). Scores the *whole
sequence* for motion/temporal-consistency artifacts.

The original codebase had two copies of this model that had quietly
diverged: the training script (``train_temporal.py``, archived) used
``pack_padded_sequence`` with an explicit ``lengths`` tensor and a masked
attention pool (correct for batched, variable-length training); the shipped
app (``app.py`` / the notebook) used an unmasked, unpacked variant that only
happened to be equivalent because inference always runs a single,
un-padded sequence (batch size 1). This module merges them into one
definition: ``lengths`` is optional, and the model is numerically equivalent
in both modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

LSTM_HIDDEN = 512
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
BIDIRECTIONAL = True
HEAD_HIDDEN_DIM = 256
HEAD_DROPOUT = 0.3


class AttentionPool(nn.Module):
    """Learned attention pooling over the time dimension, with optional masking."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, dim], lengths: [B] (valid length per sequence, or None for no padding).

        Returns (pooled: [B, dim], attention_weights: [B, T]).
        """
        b, t, _ = x.shape
        scores = self.att(x).squeeze(-1)  # [B, T]

        if lengths is not None:
            lengths = torch.clamp(lengths, min=1)
            mask = torch.arange(t, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-1e9"))

        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class TemporalModel(nn.Module):
    """Bi-LSTM + attention pooling over per-frame spatial embeddings."""

    def __init__(
        self,
        feat_dim: int = 1536,
        hidden: int = LSTM_HIDDEN,
        layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
        head_hidden: int = HEAD_HIDDEN_DIM,
        head_dropout: float = HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim,
            hidden,
            layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_dim = hidden * (2 if bidirectional else 1)
        self.attn = AttentionPool(self.out_dim)
        self.head = nn.Sequential(
            nn.Linear(self.out_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, feat_dim], lengths: [B] or None (assumes full length, no padding).

        Returns (logits: [B], attention_weights: [B, T]).
        """
        if lengths is None:
            out, _ = self.lstm(x)
            pooled, att = self.attn(out, lengths=None)
        else:
            lengths_sorted, idx = lengths.sort(descending=True)
            x_sorted = x[idx]

            packed = rnn_utils.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            packed_out, _ = self.lstm(packed)
            out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)

            _, inv = idx.sort()
            out = out[inv]
            lengths = lengths[inv]

            pooled, att = self.attn(out, lengths=lengths)

        logits = self.head(pooled).squeeze(1)
        return logits, att
