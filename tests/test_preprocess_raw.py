"""Tests for src/data/preprocess_raw.py using tiny synthetic videos —
generated programmatically with cv2.VideoWriter, no real dataset, no GPU
required (MTCNN runs on CPU).

Solid-color synthetic frames have no detectable face, which is deliberately
useful here: real MTCNN genuinely finds no face in them, so the
"no face detected" failure-logging path is exercised with the real
detector, not a mock. For the "a face WAS found" success path (folder
structure, frame numbering, ffpp metadata.csv), MTCNN's detection call is
monkeypatched to return a synthetic crop — there's no way to synthesize an
image a real face detector will reliably recognize as a face without a real
photo, and mocking the detector is the standard way to test the surrounding
save/organize logic in isolation from the CV model itself.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from src.data import preprocess_raw as pr


def _write_synthetic_video(path, n_frames=6, size=(64, 64), color=(0, 0, 0)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, size)
    frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


# --------------------------------------------------------------------------
# resolve_face_only_dir
# --------------------------------------------------------------------------


def test_resolve_face_only_dir_dfdc(tmp_path):
    assert pr.resolve_face_only_dir(tmp_path, "dfdc", "real") == tmp_path / "DFDC_REAL_Face_only_data"
    assert pr.resolve_face_only_dir(tmp_path, "dfdc", "fake") == tmp_path / "DFDC_FAKE_Face_only_data"


def test_resolve_face_only_dir_ffpp_same_folder_for_both_labels(tmp_path):
    """Both FF++ labels land in the same folder — build_splits.py has no
    separate real/fake top-level dir for FF++, unlike DFDC/Celeb-DF."""
    real_dir = pr.resolve_face_only_dir(tmp_path, "ffpp", "real")
    fake_dir = pr.resolve_face_only_dir(tmp_path, "ffpp", "fake")
    assert real_dir == fake_dir == tmp_path / "FF_Face_only_data"


def test_resolve_face_only_dir_celebdf(tmp_path):
    assert pr.resolve_face_only_dir(tmp_path, "celebdf", "real") == tmp_path / "Celeb_real_face_only"
    assert pr.resolve_face_only_dir(tmp_path, "celebdf", "fake") == tmp_path / "Celeb_fake_face_only"


def test_resolve_face_only_dir_rejects_unknown_combination(tmp_path):
    with pytest.raises(ValueError):
        pr.resolve_face_only_dir(tmp_path, "not_a_real_dataset", "real")


# --------------------------------------------------------------------------
# _sample_frame_indices
# --------------------------------------------------------------------------


def test_sample_frame_indices_normal_video():
    indices, was_short = pr._sample_frame_indices(100, 8)
    assert len(indices) == 8
    assert not was_short
    assert indices[0] == 0
    assert indices[-1] == 99


def test_sample_frame_indices_short_video_is_flagged_not_silently_padded():
    indices, was_short = pr._sample_frame_indices(3, 8)
    assert was_short
    assert len(indices) <= 3  # never duplicated up to 8


def test_sample_frame_indices_empty_video():
    indices, was_short = pr._sample_frame_indices(0, 8)
    assert len(indices) == 0
    assert not was_short


# --------------------------------------------------------------------------
# process_video — real MTCNN, real (solid-color, faceless) synthetic videos
# --------------------------------------------------------------------------


def test_process_video_no_face_detected_is_reported_not_silently_skipped(tmp_path):
    video = tmp_path / "solid.mp4"
    _write_synthetic_video(video, n_frames=6)
    out_dir = tmp_path / "out" / "solid"
    mtcnn = pr.build_mtcnn(device="cpu")

    result = pr.process_video(video, out_dir, mtcnn, n_frames=6)

    assert result.status == "no_faces_at_all"
    assert not out_dir.exists()
    assert "no face detected" in result.detail


def test_process_video_corrupt_file_is_reported(tmp_path):
    bad_video = tmp_path / "not_a_video.mp4"
    bad_video.write_bytes(b"this is not a video file")
    out_dir = tmp_path / "out" / "bad"
    mtcnn = pr.build_mtcnn(device="cpu")

    result = pr.process_video(bad_video, out_dir, mtcnn, n_frames=6)

    assert result.status == "corrupt"
    assert not out_dir.exists()


def test_process_video_short_video_flagged_in_detail(tmp_path):
    video = tmp_path / "short.mp4"
    _write_synthetic_video(video, n_frames=2)
    out_dir = tmp_path / "out" / "short"
    mtcnn = pr.build_mtcnn(device="cpu")

    result = pr.process_video(video, out_dir, mtcnn, n_frames=8)

    # Solid color -> still no face, but the short-video condition must be
    # visible in the detail regardless of the face-detection outcome.
    assert "shorter than requested" in result.detail


def test_process_video_already_done_is_skipped(tmp_path):
    out_dir = tmp_path / "out" / "done"
    out_dir.mkdir(parents=True)
    (out_dir / "frame_00.jpg").touch()
    mtcnn = pr.build_mtcnn(device="cpu")

    result = pr.process_video(tmp_path / "irrelevant.mp4", out_dir, mtcnn, n_frames=4)

    assert result.status == "already_done"


# --------------------------------------------------------------------------
# process_video — mocked detection, verifying output folder/file structure
# --------------------------------------------------------------------------


def test_process_video_creates_correct_folder_structure_when_faces_found(tmp_path, monkeypatch):
    video = tmp_path / "fake_face.mp4"
    _write_synthetic_video(video, n_frames=4)
    out_dir = tmp_path / "out" / "fake_face"
    mtcnn = pr.build_mtcnn(device="cpu")

    # __call__ is a dunder method: Python looks it up on the *class*, not the
    # instance, so patching mtcnn.__call__ directly on the instance would be
    # silently ignored (mtcnn(img) would still run the real detector). Patch
    # the class instead — same technique used in the CLI-level test below.
    fake_face_tensor = torch.zeros(3, 160, 160)
    monkeypatch.setattr(pr.MTCNN, "__call__", lambda self, img: fake_face_tensor)

    result = pr.process_video(video, out_dir, mtcnn, n_frames=4)

    assert result.status == "saved"
    assert result.faces_saved == 4
    saved = sorted(p.name for p in out_dir.glob("frame_*.jpg"))
    assert saved == ["frame_00.jpg", "frame_01.jpg", "frame_02.jpg", "frame_03.jpg"]


# --------------------------------------------------------------------------
# update_ffpp_metadata
# --------------------------------------------------------------------------


def test_update_ffpp_metadata_writes_new_rows(tmp_path):
    metadata_csv = tmp_path / "metadata.csv"
    results = [
        pr.VideoResult("005_010.mp4", "saved", 8, 8, 0),
        pr.VideoResult("099_001.mp4", "no_faces_at_all", 0, 8, 8, "no face detected"),
    ]

    pr.update_ffpp_metadata(metadata_csv, results, label="fake")

    rows = metadata_csv.read_text().strip().splitlines()
    assert rows[0] == "video,label"
    assert "005_010,1" in rows
    assert not any("099_001" in row for row in rows)  # never saved -> no folder -> no row


def test_update_ffpp_metadata_preserves_existing_label_for_already_done(tmp_path):
    metadata_csv = tmp_path / "metadata.csv"
    metadata_csv.write_text("video,label\n005_010,0\n")

    results = [pr.VideoResult("005_010.mp4", "already_done", 0, 0, 0)]
    pr.update_ffpp_metadata(metadata_csv, results, label="fake")  # label=fake must NOT overwrite the existing 0

    rows = metadata_csv.read_text().strip().splitlines()
    assert "005_010,0" in rows


# --------------------------------------------------------------------------
# CLI end-to-end
# --------------------------------------------------------------------------


def test_cli_end_to_end_serial_all_faceless_reports_failure(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    _write_synthetic_video(input_dir / "v1.mp4", n_frames=6)
    _write_synthetic_video(input_dir / "v2.mp4", n_frames=6)

    output_dir = tmp_path / "Dataset"
    failures_log = tmp_path / "logs" / "preprocessing_failures.log"

    exit_code = pr.main(
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--dataset-name", "dfdc",
            "--label", "fake",
            "--workers", "1",
            "--device", "cpu",
            "--failures-log", str(failures_log),
        ]
    )

    # Every video is solid-color (no face) -> the whole batch "fails" to find
    # a face. This is the scenario under test: the CLI must still run to
    # completion, without crashing, and must report this loudly rather than
    # exiting 0 as if everything were fine.
    assert exit_code == 1
    assert failures_log.exists()
    content = failures_log.read_text()
    assert content.count("no_faces_at_all") == 2
    assert (output_dir / "DFDC_FAKE_Face_only_data").exists()


def test_cli_end_to_end_with_mocked_detection_writes_ffpp_metadata(tmp_path, monkeypatch):
    fake_face_tensor = torch.zeros(3, 160, 160)
    monkeypatch.setattr(pr.MTCNN, "__call__", lambda self, img: fake_face_tensor)

    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    _write_synthetic_video(input_dir / "005_010.mp4", n_frames=6)

    output_dir = tmp_path / "Dataset"
    failures_log = tmp_path / "logs" / "preprocessing_failures.log"

    exit_code = pr.main(
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--dataset-name", "ffpp",
            "--label", "fake",
            "--workers", "1",
            "--device", "cpu",
            "--failures-log", str(failures_log),
        ]
    )

    assert exit_code == 0
    video_out_dir = output_dir / "FF_Face_only_data" / "005_010"
    assert sorted(p.name for p in video_out_dir.glob("frame_*.jpg")) == [f"frame_{i:02d}.jpg" for i in range(6)]

    metadata_csv = output_dir / "FF_Face_only_data" / "metadata.csv"
    assert metadata_csv.exists()
    rows = metadata_csv.read_text().strip().splitlines()
    assert rows[0] == "video,label"
    assert "005_010,1" in rows


def test_cli_multiprocess_workers_runs_end_to_end_without_gpu(tmp_path):
    """Real multiprocessing.Pool path (workers=2), CPU-only, no mocking
    (monkeypatches don't cross process boundaries) — solid-color videos, so
    this exercises "no face detected" through the parallel code path."""
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    for i in range(3):
        _write_synthetic_video(input_dir / f"vid_{i}.mp4", n_frames=6)

    output_dir = tmp_path / "Dataset"
    failures_log = tmp_path / "logs" / "preprocessing_failures.log"

    exit_code = pr.main(
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--dataset-name", "celebdf",
            "--label", "real",
            "--workers", "2",
            "--device", "cpu",
            "--failures-log", str(failures_log),
        ]
    )

    assert exit_code == 1
    assert failures_log.read_text().count("no_faces_at_all") == 3
    assert (output_dir / "Celeb_real_face_only").exists()


def test_cli_missing_input_dir_returns_error_code(tmp_path):
    exit_code = pr.main(
        [
            "--input-dir", str(tmp_path / "does_not_exist"),
            "--output-dir", str(tmp_path / "Dataset"),
            "--dataset-name", "dfdc",
            "--label", "real",
        ]
    )
    assert exit_code == 2
