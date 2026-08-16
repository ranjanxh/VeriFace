"""VeriFace — Deepfake Detection Platform (Streamlit app).

The single canonical app entrypoint. Merges what the abandoned "advanced"
notebook app (``archive/notebooks/DeepfakevFinal.ipynb``, cell 3) had that
the previously-committed ``app.py`` lacked — PDF report export, a confidence
gauge, a per-frame timeline chart, and scan history — while fixing several
concrete bugs found during the 2026-08 audit/restructure (see HANDOFF.md
"What Changed From the Original Codebase" for the full list):

- The notebook version had a broken ``import cv2ngrok`` line that would
  crash on startup; ngrok tunneling was a separate, unrelated notebook cell
  and has no place in the app module itself.
- Checkpoint loading no longer uses a bare ``except: pass`` — see
  ``src.inference.pipeline.load_models``; if the spatial or temporal model
  fails to load, this app refuses to serve predictions and shows a visible
  error instead of silently falling back to randomly-initialized weights.
- The "frames to sample" sidebar slider is now actually wired to the
  extraction call (it used to be display-only).
- The hardcoded "sub-250ms" / "99.2% accuracy" hero badges are gone. Real
  numbers are read from ``results/metrics.json``, which does not exist in a
  fresh checkout — the UI explicitly shows "Not yet benchmarked" / "Not yet
  evaluated" until ``src/eval/benchmark.py`` and
  ``src/train/train_ensemble.py`` have actually been run on real hardware.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

# Streamlit's execution model puts this file's own directory on sys.path,
# not the repo root — add the repo root explicitly so `import src....` works
# regardless of the working directory `streamlit run` was invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.graph_objects as go
import streamlit as st
import torch

from app.report import build_pdf_report, report_filename
from src import config
from src.eval.metrics import load_metrics
from src.inference.pipeline import (
    DEFAULT_FRAMES_TO_SAMPLE,
    MAX_FRAMES_TO_SAMPLE,
    MIN_FRAMES_TO_SAMPLE,
    InferenceError,
    ModelBundle,
    ModelsNotReadyError,
    NoFaceDetectedError,
    VideoDecodeError,
    analyze_video,
    load_models,
)

MAX_FILE_MB = 500
MAX_HISTORY = 8

st.set_page_config(
    page_title="VeriFace | Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, .main { background-color: #0E1117; font-family: 'Inter', sans-serif; }
        h1 {
            background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: 800 !important; font-size: 2.6rem !important; letter-spacing: -1.5px;
        }
        h2, h3 { color: #FAFAFA; font-weight: 600; }
        .metric-card {
            background: linear-gradient(145deg, #1e2030, #262730);
            border: 1px solid #3a3a4a; border-radius: 14px; padding: 22px 16px;
            text-align: center; height: 100%;
        }
        .metric-card .label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .metric-card .value { font-size: 2rem; font-weight: 700; color: #FAFAFA; }
        .verdict-fake {
            background: linear-gradient(135deg, rgba(255,75,75,0.15), rgba(255,75,75,0.05));
            border: 1px solid rgba(255,75,75,0.5); border-radius: 14px; padding: 20px; text-align: center;
        }
        .verdict-real {
            background: linear-gradient(135deg, rgba(0,204,150,0.15), rgba(0,204,150,0.05));
            border: 1px solid rgba(0,204,150,0.5); border-radius: 14px; padding: 20px; text-align: center;
        }
        .pending-badge {
            background: rgba(255,200,0,0.08); border: 1px solid rgba(255,200,0,0.4);
            border-radius: 10px; padding: 10px 14px; font-size: 0.82rem; color: #d4a72c;
        }
        .history-item {
            background: #1e2030; border-radius: 10px; padding: 10px 14px;
            margin-bottom: 8px; border-left: 4px solid; font-size: 0.82rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_models() -> ModelBundle:
    return load_models(checkpoint_dir=config.CHECKPOINT_DIR)


def render_gauge(probability: float, is_fake: bool) -> go.Figure:
    color = "#FF4B4B" if is_fake else "#00CC96"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#555", "tickfont": {"color": "#888"}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(0,204,150,0.12)"},
                    {"range": [40, 60], "color": "rgba(255,200,0,0.10)"},
                    {"range": [60, 100], "color": "rgba(255,75,75,0.12)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": probability * 100},
            },
            title={"text": "Fake Probability", "font": {"color": "#aaa", "size": 14}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#fff"},
        margin={"t": 40, "b": 10, "l": 20, "r": 20}, height=240,
    )
    return fig


def render_frame_timeline(per_frame_probs) -> go.Figure:
    x = list(range(1, len(per_frame_probs) + 1))
    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(255,75,75,0.06)", line_width=0)
    fig.add_trace(
        go.Scatter(
            x=x, y=per_frame_probs, mode="lines+markers",
            line={"color": "#6C63FF", "width": 2.5},
            marker={"size": 6, "color": per_frame_probs, "colorscale": [[0, "#00CC96"], [0.5, "#FFD166"], [1, "#FF4B4B"]], "cmin": 0, "cmax": 1, "showscale": False},
            fill="tozeroy", fillcolor="rgba(108,99,255,0.12)", name="Spatial Anomaly Score",
            hovertemplate="Frame %{x}<br>Score: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,100,100,0.5)", annotation_text="Decision Boundary", annotation_font_color="#FF4B4B")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,21,30,0.8)", font={"color": "#ccc"},
        xaxis={"title": "Frame Sample Index", "gridcolor": "#2a2a3a", "color": "#888"},
        yaxis={"title": "Anomaly Score", "range": [0, 1], "gridcolor": "#2a2a3a", "color": "#888"},
        margin={"t": 20, "b": 40, "l": 60, "r": 20}, height=260,
    )
    return fig


def render_hero_metrics() -> None:
    """Read results/metrics.json and show real numbers, or an explicit
    'pending' badge — never a fabricated placeholder. See HANDOFF.md."""
    metrics = load_metrics()
    acc_col, lat_col = st.columns(2)

    accuracy = metrics.get("accuracy")
    with acc_col:
        if accuracy and accuracy.get("test", {}).get("n", 0) > 0:
            st.metric("Test Accuracy", f"{accuracy['test']['accuracy'] * 100:.1f}%", help=f"n={accuracy['test']['n']}, measured {accuracy.get('evaluated_at_utc', '?')}")
        else:
            st.markdown('<div class="pending-badge">Accuracy: <b>not yet evaluated</b><br>run src/train/train_ensemble.py on real hardware</div>', unsafe_allow_html=True)

    latency = metrics.get("latency")
    with lat_col:
        if latency and latency.get("n_videos", 0) > 0:
            st.metric("Median Latency", f"{latency['median_ms']:.0f} ms", help=f"n={latency['n_videos']} videos, p95={latency['p95_ms']}ms, device={latency.get('device', '?')}")
        else:
            st.markdown('<div class="pending-badge">Latency: <b>not yet benchmarked</b><br>run src/eval/benchmark.py on real hardware</div>', unsafe_allow_html=True)


def render_results(result, meta, filename: str, previews: list) -> None:
    is_fake = result.is_fake
    disp_prob = result.final_prob if is_fake else 1 - result.final_prob
    verdict = "FAKE" if is_fake else "REAL"
    color = "#FF4B4B" if is_fake else "#00CC96"
    icon = "🚨" if is_fake else "✅"
    css_class = "verdict-fake" if is_fake else "verdict-real"

    st.markdown("### 📊 Forensic Report")

    if not result.used_ensemble:
        st.warning("⚠️ Running in degraded mode: the ensemble fusion checkpoint is unavailable, so this verdict is a simple average of the spatial and temporal scores, not the calibrated model. See the sidebar for details.")

    st.markdown(
        f"""
        <div class="{css_class}">
            <div style="font-size:2.4rem; font-weight:800; color:{color};">{icon} {verdict}</div>
            <div style="color:#ccc; margin-top:6px; font-size:0.95rem;">
                Confidence: <strong style="color:{color};">{disp_prob * 100:.1f}%</strong>
                &nbsp;|&nbsp; Frames Analyzed: <strong>{meta.faces_detected}</strong>
                &nbsp;|&nbsp; Total Latency: <strong>{result.total_ms} ms</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    cards = [
        ("Spatial Anomaly", f"{result.spatial_mean:.3f}", "Avg per-frame score"),
        ("Spatial Peak", f"{result.spatial_max:.3f}", "Max per-frame score"),
        ("Temporal Score", f"{result.temporal_prob:.3f}", "Motion consistency"),
        ("Std Deviation", f"{result.spatial_std:.3f}", "Frame-level variance"),
    ]
    for col, (label, val, sub) in zip(st.columns(4), cards, strict=True):
        col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{val}</div><div style="font-size:0.72rem;color:#666;margin-top:4px;">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Confidence Gauge", "📈 Frame Timeline", "🖼️ Face Samples", "ℹ️ Video Info"])

    with tab1:
        col_g, col_i = st.columns([1.2, 1])
        col_g.plotly_chart(render_gauge(result.final_prob, is_fake), use_container_width=True)
        with col_i:
            st.markdown("<br><br>", unsafe_allow_html=True)
            risk = "🔴 HIGH" if result.final_prob > 0.7 else ("🟡 MEDIUM" if result.final_prob > 0.45 else "🟢 LOW")
            st.markdown(f"**Risk Level:** {risk}")
            st.markdown(f"**Raw Fake Prob:** `{result.final_prob:.6f}`")
            st.markdown(f"**Fusion Mode:** {'Calibrated ensemble' if result.used_ensemble else 'Heuristic average (degraded)'}")

    with tab2:
        st.plotly_chart(render_frame_timeline(result.per_frame_probs), use_container_width=True)
        st.caption("Each point = one sampled frame. Scores above 0.5 indicate potential manipulation.")

    with tab3:
        if previews:
            # strict=False: columns are deliberately over-repeated (x3) to cycle
            # through a row of up to 3 columns for up to 6 preview images; the
            # two sequences are not meant to be the same length.
            for i, (col, img) in enumerate(zip(st.columns(min(len(previews), 3)) * 3, previews[:6], strict=False)):
                score = result.per_frame_probs[i] if i < len(result.per_frame_probs) else 0
                badge = "🔴" if score > 0.5 else "🟢"
                col.image(img, caption=f"{badge} Sample {i + 1} · {score:.3f}", use_container_width=True)
        else:
            st.info("No preview frames available.")

    with tab4:
        r1, r2 = st.columns(2)
        r1.metric("Duration", f"{meta.duration_sec}s")
        r1.metric("Frame Rate", f"{meta.fps} FPS")
        r2.metric("Total Frames", str(meta.total_frames))
        r2.metric("Faces Sampled", str(meta.faces_detected))

    st.markdown("<br>", unsafe_allow_html=True)
    try:
        pdf_bytes = build_pdf_report(filename, meta, result)
        st.download_button("⬇️ Download Forensic Report (PDF)", data=pdf_bytes, file_name=report_filename(filename), mime="application/pdf")
    except Exception as exc:  # noqa: BLE001 - PDF export is a nice-to-have; surface the error, don't crash the results view
        st.warning(f"PDF generation unavailable: {exc}")


def update_history(filename: str, result) -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(
        0,
        {
            "filename": filename,
            "verdict": "FAKE" if result.is_fake else "REAL",
            "prob": result.final_prob,
            "timestamp": datetime.now(UTC).strftime("%H:%M:%S"),
            "color": "#FF4B4B" if result.is_fake else "#00CC96",
        },
    )
    st.session_state.history = st.session_state.history[:MAX_HISTORY]


def main() -> None:
    with st.spinner("🔄 Initializing VeriFace Engine…"):
        models = get_models()

    with st.sidebar:
        st.markdown("## 🛡️ VeriFace")
        st.caption("Restructured build · see HANDOFF.md")
        st.markdown("---")

        if models.critical_ok:
            st.success("**Engine:** Online")
        else:
            st.error("**Engine:** Not ready")

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        st.info(f"**Accelerator:** {'GPU · ' + gpu_name if torch.cuda.is_available() else 'CPU'}")

        with st.expander("📦 Model Load Status", expanded=not models.critical_ok):
            for status in models.load_statuses:
                icon = "✅" if status.ok else "❌"
                st.caption(f"{icon} **{status.name}**: {status.detail}")

        st.markdown("---")
        st.markdown("### ⚙️ Config")
        n_frames = st.slider(
            "Frames to sample",
            MIN_FRAMES_TO_SAMPLE,
            MAX_FRAMES_TO_SAMPLE,
            DEFAULT_FRAMES_TO_SAMPLE,
            key="n_frames",
            help="More frames = higher accuracy, slower scan. This is actually wired into extraction (the original UI had this control but it was ignored).",
        )
        st.markdown("---")

        st.markdown("### 🕓 Scan History")
        history = st.session_state.get("history", [])
        if history:
            for entry in history:
                conf = entry["prob"] if entry["verdict"] == "FAKE" else 1 - entry["prob"]
                st.markdown(
                    f'<div class="history-item" style="border-left-color:{entry["color"]};">'
                    f'<strong style="color:{entry["color"]};">{entry["verdict"]}</strong> · {conf * 100:.0f}%<br>'
                    f'<span style="color:#888;">{entry["filename"][:28]} · {entry["timestamp"]}</span></div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("No scans yet.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Deepfake Detection Platform")
    st.markdown("#### Dual-stream AI: spatial artifact detection + temporal consistency scoring.")
    render_hero_metrics()
    st.markdown("---")

    if not models.critical_ok:
        st.error(f"🚨 Engine initialization failed: {models.degraded_reason}")
        st.info("This app refuses to serve predictions when a mandatory model failed to load its trained weights, rather than silently falling back to random ones. See HANDOFF.md to provision real checkpoints.")
        st.stop()

    st.markdown("### 📤 Upload Video for Analysis")
    uploaded_file = st.file_uploader("Drag & drop or click to browse", type=["mp4", "mov", "avi"], help=f"Max file size: {MAX_FILE_MB} MB")

    if not uploaded_file:
        st.info("👆 Upload a video file to begin.")
        return

    file_size_mb = len(uploaded_file.getvalue()) / (1024**2)
    if file_size_mb > MAX_FILE_MB:
        st.error(f"File too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_FILE_MB} MB.")
        return

    col_left, col_right = st.columns([1.4, 2], gap="large")

    with col_left:
        st.markdown("### 📺 Source Media")
        if "temp_path" not in st.session_state or st.session_state.get("last_filename") != uploaded_file.name:
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.flush()
            st.session_state.temp_path = tmp.name
            st.session_state.last_filename = uploaded_file.name
            st.session_state.pop("result", None)

        st.video(st.session_state.temp_path)
        st.caption(f"📁 {uploaded_file.name} · {file_size_mb:.1f} MB")
        run_scan = st.button("🚀 Run Authenticator", type="primary")

    if run_scan:
        with col_right:
            with st.spinner(f"Analyzing {n_frames} sampled frames…"):
                try:
                    result, previews, meta = analyze_video(st.session_state.temp_path, models, n_frames=n_frames)
                    st.session_state.result = result
                    st.session_state.meta = meta
                    st.session_state.previews = previews
                    update_history(uploaded_file.name, result)
                except NoFaceDetectedError:
                    st.error("❌ No face detected. Please upload a video with a clearly visible face.")
                    st.session_state.pop("result", None)
                except VideoDecodeError as exc:
                    st.error(f"❌ Could not read this video file: {exc}")
                    st.session_state.pop("result", None)
                except ModelsNotReadyError as exc:
                    st.error(f"🚨 {exc}")
                    st.session_state.pop("result", None)
                except InferenceError as exc:
                    st.error(f"❌ Analysis failed: {exc}")
                    st.session_state.pop("result", None)
                except Exception as exc:  # noqa: BLE001 - last-resort guard so users see a message instead of a blank crash
                    st.error("❌ An unexpected error occurred while analyzing this video.")
                    with st.expander("Technical details"):
                        st.code(f"{exc}\n\n{traceback.format_exc()}")
                    st.session_state.pop("result", None)

    if "result" in st.session_state:
        with col_right:
            render_results(st.session_state.result, st.session_state.meta, uploaded_file.name, st.session_state.previews)


main()
