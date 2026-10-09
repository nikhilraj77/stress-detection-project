"""
PPG-Based Stress Detection Dashboard

Two top-level system modes:
1. Real-Time Replay Mode
   - WESAD session replay
   - New raw-PPG dataset replay (CSV/NPY) without retraining
   - Interactive timeline and automatic streaming replay
2. Live Sensor Mode
   - Placeholder for future MAX30102 + ESP32 / other raw-PPG hardware.

Interactive Timeline behavior:
The selected 60-second window drives the PPG waveform, HRV values, stress
probability/state, available ground truth, and the marker on the stress trend.
For uploaded datasets, no ground truth is assumed.

Run:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler

import config
from model import CNNLSTMFusion
from ppg_stream import GenericFileReplayStream, LiveSensorStream, WESADReplayStream
from realtime_engine import RealtimeEngine, load_deployment_scaler, load_model_ensemble
from preprocessing import bandpass_filter, extract_hrv_features, zscore_normalize


st.set_page_config(
    page_title="PPG Stress Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


HRV_NAMES = [
    "Mean HR (bpm)",
    "RMSSD (ms)",
    "SDNN (ms)",
    "pNN50 (%)",
    "LF/HF Ratio",
]


# -----------------------------------------------------------------------------
# Cached data/model loading
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset() -> dict[str, np.ndarray] | None:
    if not os.path.exists(config.PROCESSED_DATASET_PATH):
        return None
    data = np.load(config.PROCESSED_DATASET_PATH)
    return {
        "X_raw": data["X_raw"],
        "X_hrv": data["X_hrv"],
        "y": data["y"],
        "subject_id": data["subject_id"],
        "window_time": data["window_time"],
    }


@st.cache_data(show_spinner=False)
def load_loso_results() -> dict[str, Any] | None:
    path = os.path.join(config.MODELS_DIR, "loso_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_deployment_resources():
    """Deployment-time model ensemble + scaler; no retraining."""
    scaler_path = os.path.join(config.MODELS_DIR, "deployment_scaler.pkl")
    if not os.path.exists(scaler_path):
        from prepare_deployment_assets import prepare
        prepare()
    return load_model_ensemble(), load_deployment_scaler()


@st.cache_resource(show_spinner=False)
def load_model_for_subject(
    subject_id: int,
    _X_hrv_all: np.ndarray,
    _subject_ids: np.ndarray,
):
    """Load the exact LOSO model/scaler used for manual WESAD replay."""
    model_path = os.path.join(config.MODELS_DIR, f"model_S{subject_id}.pt")
    if not os.path.exists(model_path):
        return None, None

    train_mask = _subject_ids != subject_id
    scaler = StandardScaler()
    scaler.fit(_X_hrv_all[train_mask])

    model = CNNLSTMFusion(n_hrv_features=_X_hrv_all.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, scaler


@st.cache_data(show_spinner=False)
def run_inference_for_subject(subject_id: int, _data: dict[str, np.ndarray]):
    """Pre-compute all WESAD window predictions for Interactive Timeline mode."""
    model, scaler = load_model_for_subject(
        subject_id,
        _data["X_hrv"],
        _data["subject_id"],
    )
    if model is None:
        return None

    mask = _data["subject_id"] == subject_id
    order = np.argsort(_data["window_time"][mask])

    X_raw = _data["X_raw"][mask][order]
    X_hrv = _data["X_hrv"][mask][order]
    y = _data["y"][mask][order]
    time_s = _data["window_time"][mask][order]

    X_hrv_scaled = scaler.transform(X_hrv)
    raw_t = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
    hrv_t = torch.tensor(X_hrv_scaled, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.sigmoid(model(raw_t, hrv_t)).numpy()

    return {
        "time_min": time_s / 60.0,
        "probs": probs,
        "y_true": y,
        "hrv_raw": X_hrv,
        "raw_signal": X_raw,
        "source_name": f"Subject S{subject_id}",
        "has_ground_truth": True,
    }


@st.cache_data(show_spinner=True)
def run_inference_for_uploaded_ppg(
    raw_bytes: bytes,
    filename: str,
    native_fs: float,
    column: str | None,
):
    """Build a timeline from an arbitrary raw PPG CSV/NPY without retraining.

    The uploaded signal is resampled to the project's 64 Hz working rate, then
    processed using the same 60-s / 50%-overlap windows and the existing
    deployment ensemble + deployment scaler.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".npy"}:
        raise ValueError("Upload a .csv or .npy raw PPG/BVP file.")
    if native_fs <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        column_arg: str | int | None = column
        if isinstance(column, str) and column.strip().isdigit():
            column_arg = int(column.strip())

        source = GenericFileReplayStream(
            file_path=tmp_path,
            native_fs=float(native_fs),
            column=column_arg,
        )
        source.open()
        signal = np.asarray(source.resampled_signal, dtype=float)
        source.close()

        window_n = int(config.WINDOW_SAMPLES)
        step_n = int(config.STEP_SAMPLES)
        if len(signal) < window_n:
            raise ValueError(
                f"The uploaded signal is only {len(signal) / config.BVP_FS:.1f}s long. "
                f"At least {config.WINDOW_SECONDS}s is required."
            )

        models, scaler = load_deployment_resources()

        windows = []
        hrv_values = []
        times_s = []
        for start in range(0, len(signal) - window_n + 1, step_n):
            window = signal[start:start + window_n]
            filtered = bandpass_filter(window)
            normalized = zscore_normalize(filtered)
            hrv = extract_hrv_features(normalized)
            windows.append(normalized)
            hrv_values.append(hrv)
            times_s.append(start / config.BVP_FS)

        X_raw = np.asarray(windows, dtype=np.float32)
        X_hrv = np.asarray(hrv_values, dtype=np.float32)
        X_hrv_scaled = scaler.transform(X_hrV := X_hrv)

        raw_t = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
        hrv_t = torch.tensor(X_hrv_scaled, dtype=torch.float32)

        # Batched ensemble inference keeps the uploaded-file mode responsive.
        probs_accum = np.zeros(len(X_raw), dtype=float)
        with torch.no_grad():
            for model in models:
                probs_accum += torch.sigmoid(model(raw_t, hrv_t)).numpy()
        probs = probs_accum / max(len(models), 1)

        return {
            "time_min": np.asarray(times_s) / 60.0,
            "probs": probs,
            "y_true": None,
            "hrv_raw": X_hrv,
            "raw_signal": X_raw,
            "source_name": filename,
            "has_ground_truth": False,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def ensure_uploaded_file_path(uploaded_file) -> str:
    """Persist a Streamlit UploadedFile into a local temp file for replay."""
    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    old_signature = st.session_state.get("uploaded_file_signature")
    old_path = st.session_state.get("uploaded_file_path")

    if signature != old_signature or not old_path or not os.path.exists(old_path):
        if old_path and os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

        suffix = Path(uploaded_file.name).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.close()

        st.session_state["uploaded_file_signature"] = signature
        st.session_state["uploaded_file_path"] = tmp.name

    return st.session_state["uploaded_file_path"]


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _stress_periods(time_min: np.ndarray, y_true: np.ndarray):
    """Return contiguous stress intervals when ground truth is available."""
    if y_true is None:
        return []
    values = list(y_true)
    if not values or any(v is None for v in values):
        return []

    periods = []
    in_stress = False
    start = None

    for t, label in zip(time_min, values):
        stress = int(label) == 1
        if stress and not in_stress:
            start = float(t)
            in_stress = True
        elif not stress and in_stress:
            periods.append((start, float(t)))
            in_stress = False

    if in_stress and len(time_min):
        periods.append((start, float(time_min[-1])))
    return periods


def _rolling_average(values: np.ndarray, window: int = 5):
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_session_trend(
    time_min: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray | None,
    current_index: int,
    title: str,
    uncertainty: np.ndarray | None = None,
):
    """Full-session stress trend with synchronized current-window marker."""
    fig, ax = plt.subplots(figsize=(12.5, 4.6))

    for start, end in _stress_periods(time_min, y_true):
        ax.axvspan(start, end, color="#C44E52", alpha=0.10)

    ax.scatter(
        time_min,
        probs,
        s=18,
        color="#B9B9B9",
        alpha=0.80,
        zorder=2,
        label="Per-window P(stress)",
    )

    trend = _rolling_average(probs, 5)
    if len(trend) == len(probs):
        ax.plot(
            time_min,
            trend,
            color="#4267A8",
            linewidth=2.6,
            zorder=3,
            label="Rolling trend (window=5)",
        )

    current_index = int(np.clip(current_index, 0, len(time_min) - 1))
    current_t = float(time_min[current_index])
    current_p = float(probs[current_index])

    ax.axvline(
        current_t,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        zorder=4,
    )
    ax.scatter(
        [current_t],
        [current_p],
        s=95,
        color="#C44E52",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label="Selected window",
    )

    if uncertainty is not None and current_index < len(uncertainty):
        ax.errorbar(
            [current_t],
            [current_p],
            yerr=[float(uncertainty[current_index])],
            fmt="none",
            ecolor="#666666",
            capsize=4,
            alpha=0.85,
            zorder=4,
        )

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlim(float(time_min[0]), float(time_min[-1]))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time into session (minutes)")
    ax.set_ylabel("Predicted P(stress)")
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(alpha=0.20)
    ax.legend(loc="upper left", ncol=3, frameon=True)
    plt.tight_layout()
    return fig


def plot_current_waveform(signal: np.ndarray, source_label: str, start_time_min: float):
    """Plot the exact 60-second normalized PPG window selected by the timeline."""
    x = np.arange(len(signal)) / config.BVP_FS
    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    ax.plot(x, signal, linewidth=1.1, color="#4267A8")
    ax.set_xlabel("Time within selected window (seconds)")
    ax.set_ylabel("Normalized BVP amplitude")
    ax.set_title(
        f"{source_label} — Current 60-second PPG Window "
        f"(starts at {start_time_min:.1f} min)",
        fontsize=12,
        pad=8,
    )
    ax.grid(alpha=0.20)
    ax.set_xlim(0, config.WINDOW_SECONDS)
    plt.tight_layout()
    return fig


def render_loso_sidebar(results: dict[str, Any]):
    st.sidebar.markdown("### Model Validation")
    c1, c2, c3 = st.sidebar.columns(3)
    c1.metric("Accuracy", f"{results['mean_accuracy'] * 100:.1f}%")
    c2.metric("F1", f"{results['mean_f1']:.3f}")
    c3.metric("AUC", f"{results['mean_auc']:.3f}")
    st.sidebar.caption(
        f"LOSO validation: {len(results['per_fold'])} held-out subjects "
        f"(mean ± std: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f} accuracy)."
    )


def render_selected_metrics(
    probability: float,
    current_time: float,
    ground_truth: int | None,
    hrv: np.ndarray,
    model_spread: float | None = None,
):
    """Render all current-window values together."""
    status = "🔴 STRESSED" if probability >= 0.5 else "🟢 CALM"
    pred_text = "Stress" if probability >= 0.5 else "Non-Stress"

    c1, c2, c3 = st.columns([1.05, 1.25, 1.0])

    with c1:
        st.subheader("Current Reading")
        st.metric("Session time", f"{current_time:.1f} min")
        st.markdown(f"## {status}")
        st.metric("Stress probability", f"{probability * 100:.1f}%")
        st.progress(float(np.clip(probability, 0.0, 1.0)))
        if model_spread is not None:
            st.caption(f"11-model spread: ±{model_spread:.3f}")

    with c2:
        st.subheader("HRV Snapshot — Selected Window")
        cols = st.columns(2)
        for idx, (name, value) in enumerate(zip(HRV_NAMES, hrv)):
            cols[idx % 2].metric(name, f"{float(value):.2f}")

    with c3:
        st.subheader("Window Comparison")
        st.metric("Model prediction", pred_text)
        if ground_truth is None:
            st.metric("Ground truth", "Not available")
            st.caption("Uploaded datasets do not contain WESAD stress labels.")
        else:
            gt_text = "Stress" if int(ground_truth) == 1 else "Non-Stress"
            match = pred_text == gt_text
            st.metric("WESAD ground truth", gt_text)
            st.metric("Agreement", "✓ Match" if match else "✕ Different")


# -----------------------------------------------------------------------------
# General checks
# -----------------------------------------------------------------------------

st.title("❤️ PPG-Based Stress Detection")
st.caption(
    "A dynamic wearable-style stress monitoring interface. The selected 60-second "
    "window controls the waveform, HRV values, stress probability, and trend marker. "
    "Real-Time Replay can also replay a new raw PPG dataset without retraining the model."
)

data = load_dataset()
results = load_loso_results()

if data is None:
    st.error("Processed dataset not found. Run `python src/build_dataset.py` first.")
    st.stop()
if results is None:
    st.error("LOSO results not found. Run `python src/train.py` first.")
    st.stop()

subjects_present = sorted(set(int(x) for x in data["subject_id"].tolist()))


# -----------------------------------------------------------------------------
# Top-level mode selection
# -----------------------------------------------------------------------------

mode = st.sidebar.radio(
    "System Mode",
    ["Real-Time Replay Mode", "Live Sensor Mode"],
    index=0,
)

render_loso_sidebar(results)


# -----------------------------------------------------------------------------
# MODE 1 — REAL-TIME REPLAY
# -----------------------------------------------------------------------------

if mode == "Real-Time Replay Mode":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Replay Controls")

    source_type = st.sidebar.radio(
        "Replay source",
        ["WESAD Session", "New PPG Dataset"],
        index=0,
        help=(
            "WESAD uses the existing labelled sessions. New PPG Dataset accepts a "
            "raw PPG/BVP CSV or NPY file and replays it through the same trained model "
            "without retraining."
        ),
    )

    generic_uploaded_file = None
    generic_native_fs = None
    generic_column = None
    generic_source_signature = None

    if source_type == "WESAD Session":
        subject_id = st.sidebar.selectbox(
            "Subject session",
            subjects_present,
            format_func=lambda x: f"Subject S{x}",
            key="replay_subject",
        )
        source_signature = f"wesad:{subject_id}"
    else:
        generic_uploaded_file = st.sidebar.file_uploader(
            "Upload raw PPG/BVP",
            type=["csv", "npy"],
            help=(
                "CSV: select the PPG column below. NPY: a 1-D array is expected. "
                "The signal must be raw PPG/BVP-like data, not HR values or a screenshot."
            ),
        )
        generic_native_fs = st.sidebar.number_input(
            "Original sampling rate (Hz)",
            min_value=1.0,
            max_value=2000.0,
            value=64.0,
            step=1.0,
            help="Sampling rate used when the uploaded PPG signal was recorded.",
        )
        generic_column = st.sidebar.text_input(
            "CSV PPG column (optional)",
            value="",
            help="Leave blank to use the first CSV column. You can also enter a column index such as 0.",
        ).strip() or None

        if generic_uploaded_file is None:
            st.sidebar.info("Upload a raw PPG CSV/NPY file to enable replay.")
            source_signature = "generic:none"
        else:
            generic_source_signature = (
                f"generic:{generic_uploaded_file.name}:{generic_uploaded_file.size}:"
                f"{generic_native_fs}:{generic_column}"
            )
            source_signature = generic_source_signature

    replay_type = st.sidebar.radio(
        "Replay type",
        ["Interactive Timeline", "Auto Replay"],
        index=0,
        help=(
            "Interactive Timeline lets you move through precomputed windows. "
            "Auto Replay streams the signal continuously through the real-time engine."
        ),
    )

    # Generic datasets support Interactive Timeline too, but unlike WESAD they
    # do not have ground-truth labels.
    if source_type == "New PPG Dataset" and generic_uploaded_file is None:
        replay_type = "Interactive Timeline"

    active_engine = st.session_state.get("rt_engine")
    active_source = st.session_state.get("rt_source_signature")

    if active_engine is not None and active_source != source_signature:
        active_engine.stop()
        st.session_state["rt_engine"] = None
        st.session_state.pop("rt_source_signature", None)
        st.session_state.pop("rt_source_label", None)
        active_engine = None

    if replay_type == "Interactive Timeline":
        # ---- Prepare timeline inference data ----
        try:
            if source_type == "WESAD Session":
                inf = run_inference_for_subject(subject_id, data)
            else:
                if generic_uploaded_file is None:
                    st.info("Upload a raw PPG/BVP CSV or NPY file to build its replay timeline.")
                    st.stop()
                inf = run_inference_for_uploaded_ppg(
                    generic_uploaded_file.getvalue(),
                    generic_uploaded_file.name,
                    float(generic_native_fs),
                    generic_column,
                )
        except Exception as exc:
            st.error(f"Could not prepare the replay timeline: {exc}")
            st.stop()

        if inf is None:
            st.error("Could not build the selected replay source.")
            st.stop()

        times = np.asarray(inf["time_min"], dtype=float)
        if len(times) == 0:
            st.error("No valid 60-second windows are available for this source.")
            st.stop()

        slider_key = (
            f"window_slider_wesad_{subject_id}"
            if source_type == "WESAD Session"
            else "window_slider_generic"
        )
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0

        source_label = inf.get("source_name", "PPG dataset")
        st.markdown("### Session Timeline")
        st.caption(
            "Drag the timeline. The PPG waveform, HRV values, stress probability, "
            "and trend marker all update to the selected 60-second window."
        )

        selected_index = st.slider(
            "Window position",
            min_value=0,
            max_value=len(times) - 1,
            value=int(np.clip(st.session_state[slider_key], 0, len(times) - 1)),
            step=1,
            format="Window %d",
            key=slider_key,
        )
        selected_index = int(selected_index)

        current_time = float(inf["time_min"][selected_index])
        previous_time = float(times[selected_index - 1]) if selected_index > 0 else current_time
        next_time = float(times[selected_index + 1]) if selected_index < len(times) - 1 else current_time

        def _set_window(target_index):
            st.session_state[slider_key] = int(
                np.clip(target_index, 0, len(times) - 1)
            )

        b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
        with b1:
            st.button(
                "⏮ Previous",
                use_container_width=True,
                on_click=_set_window,
                args=(selected_index - 1,),
            )
        with b2:
            st.button(
                "Next ⏭",
                use_container_width=True,
                on_click=_set_window,
                args=(selected_index + 1,),
            )
        with b3:
            st.button(
                "↺ Reset",
                use_container_width=True,
                on_click=_set_window,
                args=(0,),
            )
        with b4:
            st.info(
                f"Selected: **{current_time:.1f} min**  •  "
                f"window ≈ {current_time:.1f}–{current_time + config.WINDOW_SECONDS / 60.0:.1f} min  •  "
                f"{selected_index + 1}/{len(times)}"
            )

        current_prob = float(inf["probs"][selected_index])
        current_hrv = inf["hrv_raw"][selected_index]
        current_true = None
        if inf.get("has_ground_truth", False):
            current_true = int(inf["y_true"][selected_index])

        render_selected_metrics(
            current_prob,
            current_time,
            current_true,
            current_hrv,
        )

        st.markdown("---")
        st.subheader("Current PPG Window")
        wave_col, trend_col = st.columns([1.05, 1.35])

        with wave_col:
            st.pyplot(
                plot_current_waveform(
                    inf["raw_signal"][selected_index],
                    source_label,
                    current_time,
                ),
                clear_figure=True,
            )
            st.caption(
                "Exact 60-second window currently entering the model. "
                "Move the timeline to change every downstream value."
            )

        with trend_col:
            st.pyplot(
                plot_session_trend(
                    inf["time_min"],
                    inf["probs"],
                    inf.get("y_true"),
                    selected_index,
                    title=f"{source_label} — Stress Trend",
                ),
                clear_figure=True,
            )
            if inf.get("has_ground_truth", False):
                st.caption("Pink shading = WESAD stress periods. Red marker = selected window.")
            else:
                st.caption("Uploaded raw PPG has no ground-truth labels; the trend shows model predictions only.")

        st.markdown("---")
        st.subheader("Current Window Details")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Window start", f"{current_time:.1f} min")
        with d2:
            st.metric("Previous window", f"{previous_time:.1f} min")
        with d3:
            st.metric("Next window", f"{next_time:.1f} min")

        st.markdown("---")
        st.subheader("Explainability")
        st.caption(
            "SHAP and saliency plots describe the trained model globally. The timeline-synchronized "
            "sections above are specific to the selected PPG window."
        )

        shap_files = sorted(
            f for f in os.listdir(config.MODELS_DIR) if f.startswith("shap_summary")
        )
        saliency_files = sorted(
            f for f in os.listdir(config.MODELS_DIR) if f.startswith("saliency")
        )
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**SHAP — HRV feature importance**")
            if shap_files:
                st.image(
                    os.path.join(config.MODELS_DIR, shap_files[0]),
                    use_container_width=True,
                )
            else:
                st.info("Run `python src/explain.py` to generate SHAP output.")
        with e2:
            st.markdown("**Saliency — raw PPG waveform sensitivity**")
            if saliency_files:
                st.image(
                    os.path.join(config.MODELS_DIR, saliency_files[0]),
                    use_container_width=True,
                )
            else:
                st.info("Run `python src/explain.py` to generate saliency output.")

    # -------------------------------------------------------------------------
    # Auto replay mode — WESAD or generic uploaded raw PPG.
    # -------------------------------------------------------------------------
    else:
        st.header("⏱️ Real-Time Replay")
        if source_type == "WESAD Session":
            source_caption = (
                f"Recorded WESAD Subject S{subject_id} is streamed chunk-by-chunk as though "
                "it were arriving from a wearable sensor."
            )
            source_ready = True
            source_label = f"Subject S{subject_id}"
        else:
            source_caption = (
                "The uploaded raw PPG/BVP file is streamed chunk-by-chunk through the same "
                "trained model. No retraining is required."
            )
            source_ready = generic_uploaded_file is not None
            source_label = generic_uploaded_file.name if generic_uploaded_file is not None else "Uploaded PPG"

        st.caption(
            f"{source_caption} Predictions use a sliding 60-second window. "
            "New raw PPG datasets are accepted as CSV/NPY and resampled to the project's 64 Hz working rate."
        )

        c1, c2, c3 = st.sidebar.columns(3)
        with c1:
            refresh_seconds = st.selectbox(
                "Refresh",
                [5.0, 10.0],
                index=0,
                format_func=lambda x: f"{int(x)}s",
                key="rt_refresh",
            )
        with c2:
            speed = st.selectbox(
                "Speed",
                [1.0, 2.0, 4.0, 8.0, 16.0],
                index=3,
                format_func=lambda x: f"{x:g}×",
                key="rt_speed",
            )
        with c3:
            chunk = st.selectbox(
                "Chunk",
                [0.5, 1.0, 2.0],
                index=1,
                format_func=lambda x: f"{x:g}s",
                key="rt_chunk",
            )

        old_engine = st.session_state.get("rt_engine")
        old_source = st.session_state.get("rt_source_signature")
        if old_engine is not None and old_source != source_signature:
            old_engine.stop()
            st.session_state["rt_engine"] = None
            st.session_state.pop("rt_source_signature", None)
            st.session_state.pop("rt_source_label", None)
            old_engine = None

        if source_type == "New PPG Dataset" and not source_ready:
            st.warning("Upload a raw PPG/BVP CSV or NPY file before starting Replay.")

        b1, b2, b3 = st.columns(3)
        with b1:
            start_disabled = (
                old_engine is not None
                and old_engine.is_running()
            ) or not source_ready
            if st.button("▶ Start Replay", use_container_width=True, disabled=start_disabled):
                try:
                    scaler_path = os.path.join(config.MODELS_DIR, "deployment_scaler.pkl")
                    if not os.path.exists(scaler_path):
                        from prepare_deployment_assets import prepare
                        prepare()

                    if source_type == "WESAD Session":
                        stream = WESADReplayStream(
                            subject_id=subject_id,
                            speed_multiplier=float(speed),
                        )
                    else:
                        saved_path = ensure_uploaded_file_path(generic_uploaded_file)
                        column_arg: str | int | None = generic_column
                        if isinstance(generic_column, str) and generic_column.isdigit():
                            column_arg = int(generic_column)
                        stream = GenericFileReplayStream(
                            file_path=saved_path,
                            native_fs=float(generic_native_fs),
                            column=column_arg,
                        )

                    engine = RealtimeEngine(
                        stream=stream,
                        refresh_seconds=float(refresh_seconds),
                        chunk_seconds=float(chunk),
                        speed_multiplier=float(speed),
                    )
                    st.session_state["rt_engine"] = engine
                    st.session_state["rt_source_signature"] = source_signature
                    st.session_state["rt_source_label"] = source_label
                    engine.start()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not start replay: {exc}")

        with b2:
            if st.button("■ Stop", use_container_width=True):
                engine = st.session_state.get("rt_engine")
                if engine is not None:
                    engine.stop()
                st.session_state["rt_engine"] = None
                st.rerun()

        with b3:
            if st.button("↺ Reset", use_container_width=True):
                engine = st.session_state.get("rt_engine")
                if engine is not None:
                    engine.stop()
                st.session_state["rt_engine"] = None
                st.session_state.pop("rt_source_signature", None)
                st.session_state.pop("rt_source_label", None)
                st.rerun()

        engine = st.session_state.get("rt_engine")
        if engine is None:
            st.info(
                "Press **Start Replay**. After the first 60 seconds of signal are buffered, "
                "the waveform, HRV values, stress probability, and trend will update automatically. "
                "For a new dataset, no retraining is required as long as the upload contains a raw PPG/BVP waveform."
            )
        else:
            running_text = "streaming" if engine.is_running() else "finished"
            st.success(
                f"{source_label} is {running_text} at {speed:g}×. "
                "The dashboard is reading the latest state from the real-time engine."
            )

            def live_panel():
                state = engine.get_latest_state()
                latest = state["latest"]

                if state.get("error"):
                    st.error(f"Real-time engine error: {state['error']}")
                    st.caption(
                        f"Buffered signal before the error: {state['buffer_seconds_available']:.1f}s"
                    )
                    return

                if latest is None:
                    st.info(
                        f"Collecting the first {config.WINDOW_SECONDS}-second window… "
                        f"buffered: {state['buffer_seconds_available']:.1f}s"
                    )
                    st.progress(
                        float(
                            np.clip(
                                state["buffer_seconds_available"] / config.WINDOW_SECONDS,
                                0.0,
                                1.0,
                            )
                        )
                    )
                    return

                history = state["history"]
                probs = np.asarray([h["probability"] for h in history], dtype=float)
                times = np.asarray([h["time_sec"] / 60.0 for h in history], dtype=float)
                uncertainty = np.asarray([h["uncertainty"] for h in history], dtype=float)

                if any(h.get("ground_truth") is not None for h in history):
                    gt = np.asarray(
                        [h.get("ground_truth") for h in history],
                        dtype=object,
                    )
                else:
                    gt = None

                latest_window = state.get("latest_window")
                latest_hrv = state.get("latest_hrv")
                if latest_window is None or latest_hrv is None:
                    st.info("Waiting for the first complete inference window…")
                    return

                current_prob = float(latest["probability"])
                current_time = float(latest["time_sec"] / 60.0)
                current_gt = latest.get("ground_truth")
                if current_gt is not None:
                    current_gt = int(current_gt)

                render_selected_metrics(
                    current_prob,
                    current_time,
                    current_gt,
                    latest_hrv,
                    model_spread=float(latest.get("uncertainty", 0.0)),
                )

                st.markdown("---")
                wcol, tcol = st.columns([1.05, 1.35])
                with wcol:
                    st.subheader("Live PPG Window")
                    st.pyplot(
                        plot_current_waveform(
                            latest_window,
                            source_label,
                            current_time - config.WINDOW_SECONDS / 60.0,
                        ),
                        clear_figure=True,
                    )
                    st.caption("Latest 60-second rolling window entering the model.")

                with tcol:
                    st.subheader("Live Stress Trend")
                    st.pyplot(
                        plot_session_trend(
                            times,
                            probs,
                            gt,
                            len(times) - 1,
                            title=f"{source_label} — Live Stress Trend",
                            uncertainty=uncertainty,
                        ),
                        clear_figure=True,
                    )
                    if gt is None:
                        st.caption("No ground-truth labels are available for this uploaded dataset.")
                    else:
                        st.caption("The red marker moves as new predictions are generated.")

                st.caption(
                    f"Buffered signal: {state['buffer_seconds_available']:.1f}s  •  "
                    f"Predictions: {len(history)}  •  "
                    f"Replay running: {'Yes' if state['is_running'] else 'Finished'}"
                )

            fragment = getattr(st, "fragment", None)
            if fragment is not None:
                @fragment(run_every=1.0)
                def _live_fragment():
                    live_panel()
                _live_fragment()
            else:
                live_panel()
                if st.button("Refresh display"):
                    st.rerun()
                st.warning(
                    "Your Streamlit version does not support automatic fragments. "
                    "The engine can still run; refresh the page to see the latest state."
                )


# -----------------------------------------------------------------------------
# MODE 2 — LIVE SENSOR (future hardware)
# -----------------------------------------------------------------------------

else:
    st.header("🔌 Live Sensor Mode")
    st.info(
        "This mode is reserved for future physical PPG acquisition. The software "
        "input interface is already separated from buffering, preprocessing, model "
        "inference, and the dashboard."
    )

    st.markdown(
        """
### Planned path

```text
MAX30102 optical PPG sensor
        ↓
      ESP32
        ↓
   Bluetooth / USB
        ↓
   LiveSensorStream
        ↓
 Rolling Buffer
        ↓
Preprocessing + HRV
        ↓
 CNN + BiLSTM + HRV
        ↓
 Stress Probability
        ↓
 Dynamic Dashboard
```

The hardware does **not** need to behave like a complete smartwatch. It only
needs to collect a usable optical PPG waveform.
"""
    )

    st.markdown("### Current implementation status")
    status_cols = st.columns(4)
    status_cols[0].success("PPG stream interface\n\nReady")
    status_cols[1].success("Rolling buffer\n\nReady")
    status_cols[2].success("Realtime inference\n\nReady")
    status_cols[3].warning("Physical sensor\n\nNot connected")

    if st.button("Test Live Sensor Interface"):
        try:
            LiveSensorStream().open()
        except NotImplementedError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Sensor interface error: {exc}")


st.markdown("---")
st.caption(
    "WESAD provides the recorded physiological signal used for model development. "
    "Real-Time Replay Mode can replay WESAD or any compatible raw PPG/BVP CSV/NPY without retraining. "
    "Live Sensor Mode is reserved for future raw-PPG hardware such as MAX30102 + ESP32."
)
