"""Shared metric helpers + the results/metrics.json read/write contract.

``results/metrics.json`` is the single file the app reads to display
accuracy/latency figures. It intentionally does not exist in this
environment (no GPU/dataset to measure anything) — see HANDOFF.md. Once
``src/train/train_ensemble.py`` and ``src/eval/benchmark.py`` are actually
run on real hardware, they populate the ``accuracy`` and ``latency``
sections respectively via ``update_metrics_section`` below, so one script
can never accidentally wipe out the other's results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src import config

METRICS_SCHEMA_VERSION = "1.0"
METRICS_SECTIONS = ("accuracy", "latency")


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """ROC-AUC, returning NaN (never raising) for degenerate inputs such as
    a single-class batch — callers should treat NaN as "undefined this
    epoch/batch", not as an error."""
    if len(y_true) == 0 or np.isnan(y_scores).any():
        return float("nan")
    try:
        return roc_auc_score(y_true, y_scores)
    except Exception:  # noqa: BLE001
        return float("nan")


def load_metrics(path: Path = config.METRICS_FILE) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": METRICS_SCHEMA_VERSION, "accuracy": None, "latency": None}
    return json.loads(path.read_text(encoding="utf-8"))


def update_metrics_section(section: str, data: dict[str, Any], path: Path = config.METRICS_FILE) -> dict[str, Any]:
    """Merge-write one section of results/metrics.json without disturbing
    the other section (accuracy vs latency are produced by different
    scripts, run at different times, possibly on different machines)."""
    if section not in METRICS_SECTIONS:
        raise ValueError(f"Unknown metrics section {section!r}, expected one of {METRICS_SECTIONS}")

    metrics = load_metrics(path)
    metrics[section] = data
    metrics["schema_version"] = METRICS_SCHEMA_VERSION

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
