"""The one canonical inference pipeline: face extraction -> spatial scoring
-> temporal scoring -> ensemble fusion.

Both ``app/main.py`` (interactive Streamlit UI) and ``src/eval/benchmark.py``
(offline latency measurement) import ``load_models`` and ``analyze_video``
from here. In the original codebase this logic was duplicated (with subtle
differences) between the committed ``app.py`` and the notebook's "advanced"
app variant. There is now exactly one implementation.

This module also fixes two concrete bugs found in the original app:

1. **Silent random-weight fallback**: the original had a bare
   ``except: pass`` around checkpoint loading, so a missing/corrupt
   checkpoint silently left the model at random initialization and kept
   serving predictions with no visible warning. ``load_models`` now returns
   an explicit, inspectable load status per component; the app is expected
   to refuse to serve predictions (``ModelBundle.critical_ok is False``)
   rather than silently continue. See ``ModelsNotReadyError`` below, which
   ``analyze_video`` raises defensively even if a caller forgets to check.

2. **Dead "frames to sample" control**: the original UI slider set
   ``st.session_state.n_frames`` but the extraction function underneath
   ignored it and always used a hardcoded constant. ``extract_faces`` here
   takes ``n_frames`` as a real parameter.

Latency: ``analyze_video`` times the *entire* call (video decode + MTCNN
face detection/cropping + both model forward passes + ensemble fusion) with
one ``time.perf_counter()`` span. The original only timed the two model
forward passes and called that "inference_ms", then displayed an unrelated
hardcoded "<250ms" marketing badge next to it. See
``src/eval/benchmark.py`` and ``results/metrics.json`` for how a real,
measured, end-to-end latency number is meant to reach the UI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

from src import config
from src.models.checkpoint import (
    CheckpointError,
    load_ensemble_checkpoint,
    load_torch_checkpoint,
)
from src.models.ensemble import build_ensemble_features
from src.models.spatial import SpatialModel
from src.models.temporal import TemporalModel

logger = logging.getLogger(__name__)

IMG_SIZE = 224
DEFAULT_FRAMES_TO_SAMPLE = 20
MIN_FRAMES_TO_SAMPLE = 8
MAX_FRAMES_TO_SAMPLE = 40
MAX_PREVIEW_FRAMES = 6

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class InferenceError(Exception):
    """Base class for all user-facing inference failures."""


class VideoDecodeError(InferenceError):
    """The file could not be opened/decoded as a video at all (corrupt file,
    unsupported codec/container, zero-length, etc.)."""


class NoFaceDetectedError(InferenceError):
    """The video decoded fine, but MTCNN found no face in any sampled frame."""


class ModelsNotReadyError(InferenceError):
    """Raised if analyze_video() is called with a ModelBundle whose
    mandatory components (spatial, temporal) failed to load. Serving a
    prediction in this state would mean serving it from randomly
    initialized weights — refused by design, not just by UI convention."""


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


@dataclass
class ModelLoadStatus:
    name: str
    ok: bool
    detail: str


@dataclass
class ModelBundle:
    spatial: SpatialModel
    temporal: TemporalModel
    ensemble: object | None  # EnsembleCheckpoint, or None if running in degraded/heuristic mode
    mtcnn: MTCNN
    device: torch.device
    load_statuses: list[ModelLoadStatus] = field(default_factory=list)

    @property
    def critical_ok(self) -> bool:
        """True only if BOTH mandatory models (spatial, temporal) loaded
        real trained weights. The ensemble is optional — its absence
        degrades gracefully to a heuristic average, clearly marked as such,
        never silently."""
        critical = {s.name for s in self.load_statuses if s.name in ("spatial", "temporal") and s.ok}
        return critical == {"spatial", "temporal"}

    @property
    def degraded_reason(self) -> str | None:
        if self.critical_ok:
            return None
        failed = [s for s in self.load_statuses if s.name in ("spatial", "temporal") and not s.ok]
        return "; ".join(f"{s.name}: {s.detail}" for s in failed)


def load_models(
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    device: torch.device | None = None,
) -> ModelBundle:
    """Load all three model stages, never silently falling back to random
    weights for the two mandatory ones. Always returns a ``ModelBundle`` —
    check ``.critical_ok`` before calling ``analyze_video`` with it."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    statuses: list[ModelLoadStatus] = []

    spatial_model = SpatialModel(pretrained=False).to(device)
    try:
        meta = load_torch_checkpoint(
            spatial_model, checkpoint_dir / "spatial" / "spatial_best_valAUC.pth", map_location=device
        )
        statuses.append(ModelLoadStatus("spatial", True, f"loaded (epoch={meta.epoch}, val_auc={meta.val_auc:.4f})"))
    except CheckpointError as exc:
        statuses.append(ModelLoadStatus("spatial", False, str(exc)))
        logger.error("Spatial checkpoint failed to load: %s", exc)
    spatial_model.eval()

    temporal_model = TemporalModel().to(device)
    try:
        meta = load_torch_checkpoint(
            temporal_model, checkpoint_dir / "temporal" / "temporal_best_valAUC.pth", map_location=device
        )
        statuses.append(ModelLoadStatus("temporal", True, f"loaded (epoch={meta.epoch}, val_auc={meta.val_auc:.4f})"))
    except CheckpointError as exc:
        statuses.append(ModelLoadStatus("temporal", False, str(exc)))
        logger.error("Temporal checkpoint failed to load: %s", exc)
    temporal_model.eval()

    ensemble = None
    try:
        ensemble = load_ensemble_checkpoint(checkpoint_dir / "ensemble" / "ensemble_final.joblib")
        statuses.append(ModelLoadStatus("ensemble", True, "loaded"))
    except CheckpointError as exc:
        statuses.append(
            ModelLoadStatus(
                "ensemble", False, f"{exc} -- degraded mode: using (spatial_mean + temporal_prob) / 2 heuristic"
            )
        )
        logger.warning("Ensemble checkpoint unavailable, using fallback heuristic: %s", exc)

    mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
    statuses.append(ModelLoadStatus("mtcnn", True, "ready"))

    return ModelBundle(
        spatial=spatial_model, temporal=temporal_model, ensemble=ensemble, mtcnn=mtcnn, device=device, load_statuses=statuses
    )


# --------------------------------------------------------------------------
# Face extraction
# --------------------------------------------------------------------------


@dataclass
class VideoMeta:
    total_frames: int
    fps: float
    duration_sec: float
    faces_detected: int


def extract_faces(
    video_path: str | Path,
    mtcnn: MTCNN,
    n_frames: int = DEFAULT_FRAMES_TO_SAMPLE,
) -> tuple[torch.Tensor, list[Image.Image], VideoMeta]:
    """Sample ``n_frames`` evenly-spaced frames, run MTCNN, return
    preprocessed face-crop tensors + a handful of preview images + metadata.

    Raises ``VideoDecodeError`` if the file can't be opened/decoded at all,
    or ``NoFaceDetectedError`` if it decodes but no face is found in any
    sampled frame.
    """
    n_frames = max(MIN_FRAMES_TO_SAMPLE, min(MAX_FRAMES_TO_SAMPLE, n_frames))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise VideoDecodeError(f"Could not open video file: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0:
            raise VideoDecodeError(f"Video file has no readable frames: {video_path}")

        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        crops: list[torch.Tensor] = []
        previews: list[Image.Image] = []
        faces_detected = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                crop = mtcnn(Image.fromarray(rgb))
            except Exception as exc:  # noqa: BLE001 - MTCNN can raise on odd frames; skip, don't abort the whole video
                logger.debug("Face extraction error on frame %d of %s: %s", idx, video_path, exc)
                continue
            if crop is None:
                continue
            crops.append(_PREPROCESS(transforms.ToPILImage()(crop)))
            if len(previews) < MAX_PREVIEW_FRAMES:
                previews.append(Image.fromarray(rgb))
            faces_detected += 1

        meta = VideoMeta(total_frames=total, fps=round(fps, 2), duration_sec=round(total / fps, 2), faces_detected=faces_detected)
    finally:
        cap.release()

    if not crops:
        raise NoFaceDetectedError(f"No face detected in any of {n_frames} sampled frames of {video_path}")

    return torch.stack(crops), previews, meta


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class InferenceResult:
    per_frame_probs: np.ndarray
    spatial_mean: float
    spatial_max: float
    spatial_std: float
    temporal_prob: float
    final_prob: float
    is_fake: bool
    used_ensemble: bool
    total_ms: float


@torch.no_grad()
def run_inference(feats: torch.Tensor, models: ModelBundle) -> InferenceResult:
    """Score an already-extracted batch of face crops: [N, 3, H, W] -> InferenceResult.

    Does not include face extraction time — see ``analyze_video`` for the
    full end-to-end timer used by the UI and the benchmark script.
    """
    feats = feats.to(models.device)

    backbone_feats = models.spatial.embed(feats)
    spatial_logits = models.spatial.head(backbone_feats).squeeze(1)
    spatial_probs = torch.sigmoid(spatial_logits).cpu().numpy()

    temporal_logit, _ = models.temporal(backbone_feats.unsqueeze(0))
    temporal_logit = temporal_logit.item()
    temporal_prob = float(torch.sigmoid(torch.tensor(temporal_logit)).item())

    used_ensemble = models.ensemble is not None
    if used_ensemble:
        feature_row = build_ensemble_features(spatial_probs, temporal_logit)
        final_prob = models.ensemble.predict_proba_fake(feature_row)
    else:
        final_prob = (float(spatial_probs.mean()) + temporal_prob) / 2

    return InferenceResult(
        per_frame_probs=spatial_probs,
        spatial_mean=float(spatial_probs.mean()),
        spatial_max=float(spatial_probs.max()),
        spatial_std=float(spatial_probs.std()),
        temporal_prob=temporal_prob,
        final_prob=final_prob,
        is_fake=final_prob > 0.5,
        used_ensemble=used_ensemble,
        total_ms=float("nan"),  # filled in by analyze_video()
    )


def analyze_video(
    video_path: str | Path,
    models: ModelBundle,
    n_frames: int = DEFAULT_FRAMES_TO_SAMPLE,
) -> tuple[InferenceResult, list[Image.Image], VideoMeta]:
    """Full pipeline: decode -> detect faces -> score -> fuse, timed end to end.

    This is the ONE function both the app and the benchmark script call.
    Raises ``ModelsNotReadyError`` / ``VideoDecodeError`` / ``NoFaceDetectedError``
    — callers should catch these and show a clear message, never a raw
    stack trace (see app/main.py).
    """
    if not models.critical_ok:
        raise ModelsNotReadyError(
            f"Cannot run inference: {models.degraded_reason}. Refusing to serve "
            "predictions from a model that failed to load its trained weights."
        )

    t0 = time.perf_counter()
    feats, previews, meta = extract_faces(video_path, models.mtcnn, n_frames=n_frames)
    result = run_inference(feats, models)
    result.total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return result, previews, meta
