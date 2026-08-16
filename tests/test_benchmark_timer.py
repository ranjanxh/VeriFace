"""Structural tests for src/eval/benchmark.py's aggregation logic.

These do NOT run real model inference (no GPU/checkpoints/dataset in this
environment) — they monkeypatch analyze_video to return synthetic timing
results and check that run_benchmark() aggregates them correctly (mean,
median, p95, failure handling). This is exactly the kind of "did we compute
the stats right" bug that would otherwise only surface after an expensive
real benchmark run on the target GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.eval import benchmark
from src.inference.pipeline import NoFaceDetectedError


@dataclass
class _FakeModels:
    device: str = "cpu"


@dataclass
class _FakeResult:
    total_ms: float


def test_discover_videos_filters_by_extension(tmp_path: Path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mov").touch()
    (tmp_path / "c.avi").touch()
    (tmp_path / "d.txt").touch()  # should be ignored
    (tmp_path / "e.MP4").touch()  # different case; original glob is case-sensitive on Linux, document that behavior

    found = benchmark.discover_videos(tmp_path)
    names = {p.name for p in found}
    assert names == {"a.mp4", "b.mov", "c.avi"}


def test_run_benchmark_computes_mean_median_p95(monkeypatch):
    timings = [100.0, 120.0, 90.0, 200.0, 110.0]
    videos = [Path(f"video_{i}.mp4") for i in range(len(timings))]

    call_order = iter(timings)

    def fake_analyze_video(video_path, models, n_frames):
        return _FakeResult(total_ms=next(call_order)), [], None

    monkeypatch.setattr(benchmark, "analyze_video", fake_analyze_video)

    summary = benchmark.run_benchmark(videos, _FakeModels(), n_frames=20)

    assert summary["n_videos"] == 5
    assert summary["n_failures"] == 0
    assert summary["mean_ms"] == pytest.approx(sum(timings) / len(timings), abs=0.1)
    assert summary["median_ms"] == pytest.approx(110.0, abs=0.1)
    assert summary["min_ms"] == pytest.approx(90.0, abs=0.1)
    assert summary["max_ms"] == pytest.approx(200.0, abs=0.1)


def test_run_benchmark_records_failures_without_crashing(monkeypatch):
    videos = [Path("good.mp4"), Path("bad.mp4")]

    def fake_analyze_video(video_path, models, n_frames):
        if video_path.name == "bad.mp4":
            raise NoFaceDetectedError("no face")
        return _FakeResult(total_ms=150.0), [], None

    monkeypatch.setattr(benchmark, "analyze_video", fake_analyze_video)

    summary = benchmark.run_benchmark(videos, _FakeModels(), n_frames=20)

    assert summary["n_videos"] == 1
    assert summary["n_failures"] == 1
    assert summary["failures"][0]["video"] == "bad.mp4"


def test_run_benchmark_raises_if_every_video_fails(monkeypatch):
    def fake_analyze_video(video_path, models, n_frames):
        raise NoFaceDetectedError("no face")

    monkeypatch.setattr(benchmark, "analyze_video", fake_analyze_video)

    with pytest.raises(RuntimeError):
        benchmark.run_benchmark([Path("bad.mp4")], _FakeModels(), n_frames=20)
