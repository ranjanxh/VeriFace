"""PDF forensic report generation, extracted from the "Report Generation"
section of the abandoned advanced app variant that only ever existed inside
``archive/notebooks/DeepfakevFinal.ipynb`` (cell 3) — it was never part of
the committed ``app.py``. See HANDOFF.md "What Changed" for the full story.

Callable from both the Streamlit app (``app/main.py``) and, if useful later,
a standalone CLI script — kept free of any ``streamlit`` import so it has no
UI framework dependency.

Uses ``fpdf2`` (the actively maintained fork of the original, unmaintained
``fpdf`` PyPI package — it is a drop-in replacement that keeps the
``from fpdf import FPDF`` import name, so this is the only file that needed
to change to move off a dead dependency).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF

from src.inference.pipeline import InferenceResult, VideoMeta


def build_pdf_report(filename: str, meta: VideoMeta, result: InferenceResult) -> bytes:
    """Generate a concise PDF forensic report and return raw bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(14, 17, 23)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "VeriFace Forensic Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True, align="C")
    pdf.ln(4)

    verdict = "FAKE -- SYNTHETIC MEDIA DETECTED" if result.is_fake else "AUTHENTIC -- NO MANIPULATION DETECTED"
    r, g, b = (255, 75, 75) if result.is_fake else (0, 204, 150)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, verdict, ln=True, align="C", fill=True)
    pdf.ln(6)

    pdf.set_text_color(30, 30, 30)

    def section(title: str) -> None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(230, 230, 245)
        pdf.cell(0, 8, f"  {title}", ln=True, fill=True)
        pdf.ln(2)

    def row(label: str, value: str) -> None:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(70, 7, label, border="B")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 7, value, border="B", ln=True)

    section("Source File")
    row("Filename", filename)
    row("Duration", f"{meta.duration_sec} seconds")
    row("Frame Rate", f"{meta.fps} FPS")
    row("Total Frames", str(meta.total_frames))
    row("Faces Sampled", str(meta.faces_detected))
    pdf.ln(4)

    section("Forensic Scores")
    conf = result.final_prob if result.is_fake else 1 - result.final_prob
    row("Final Confidence", f"{conf * 100:.1f}%  ({'FAKE' if result.is_fake else 'REAL'})")
    row("Ensemble Used", "Yes (calibrated fusion model)" if result.used_ensemble else "No (degraded: heuristic average — see HANDOFF.md)")
    row("Spatial Anomaly (mean)", f"{result.spatial_mean:.4f}")
    row("Spatial Anomaly (max)", f"{result.spatial_max:.4f}")
    row("Temporal Inconsistency", f"{result.temporal_prob:.4f}")
    row("Total Inference Time", f"{result.total_ms} ms (end-to-end: decode + face detection + models)")
    pdf.ln(4)

    section("Per-Frame Spatial Scores")
    pdf.set_font("Helvetica", "", 8)
    for i, score in enumerate(result.per_frame_probs):
        tag = " <- HIGH RISK" if score > 0.7 else ""
        pdf.cell(0, 5, f"  Frame sample {i + 1:>2}:  {score:.4f}{tag}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, "VeriFace | Generated for evaluation purposes -- not a certified forensic instrument", align="C", ln=True)

    output = pdf.output()
    return bytes(output)


def report_filename(source_filename: str) -> str:
    stem = Path(source_filename).stem
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"veriface_{stem}_{timestamp}.pdf"
