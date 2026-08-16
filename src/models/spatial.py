"""Spatial stream: frame-level deepfake artifact classifier.

EfficientNet-B3 backbone (via ``timm``) + a small MLP head. Scores a single
face-cropped frame for spatial/texture manipulation artifacts. This module
contains the model definition only — see ``src/train/train_spatial.py`` for
training and ``src/inference/pipeline.py`` for serving.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

BACKBONE_NAME = "efficientnet_b3"
HEAD_HIDDEN_DIM = 512
HEAD_DROPOUT = 0.4


class SpatialModel(nn.Module):
    """EfficientNet-B3 backbone + binary classification head.

    ``backbone`` is exposed separately from ``head`` because the rest of the
    pipeline (embedding extraction, the temporal model, the ensemble) all
    consume the 1536-d backbone feature vector directly, independent of the
    frame-level classification head.
    """

    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        pretrained: bool = False,
        head_hidden: int = HEAD_HIDDEN_DIM,
        dropout: float = HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.feat_dim: int = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> logits: [B]"""
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> backbone feature vectors: [B, feat_dim].

        Used by embedding extraction and the temporal/ensemble stages, which
        consume backbone features directly rather than the classification
        head's output.
        """
        return self.backbone(x)
