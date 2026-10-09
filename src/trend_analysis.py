"""
Objective 4: "Provide stress trends over time"

IMPORTANT SCOPE NOTE (read this before writing your report):
WESAD contains a single ~1-2 hour recording session per subject, not
multi-day/longitudinal data. So this script produces a WITHIN-SESSION
stress trajectory -- how predicted stress probability rises and falls
minute-by-minute across one recording session -- NOT long-term burnout
or multi-day trend prediction. Be precise about this distinction in your
report; it's an honest, buildable scope for this dataset, and is a
legitimate reading of "stress trends over time" without overclaiming.

For each subject, this script:
  - Reloads that subject's LOSO model (the one trained WITHOUT them,
    exactly as used for their test evaluation -- no extra leakage)
  - Runs inference on all of their windows, IN TEMPORAL ORDER
  - Plots predicted stress probability over session time, with a rolling
    average trend line
  - Flags rising/falling stress segments using simple slope detection
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import config
from model import CNNLSTMFusion

TREND_PLOTS_DIR = os.path.join(config.MODELS_DIR, "trend_plots")
ROLLING_WINDOW = 5  # number of consecutive windows to average for the trend line


def load_subject_trend_data(subject_id: int):
    data = np.load(config.PROCESSED_DATASET_PATH)
    X_raw, X_hrv, y, subject_ids, window_time = (
        data["X_raw"], data["X_hrv"], data["y"],
        data["subject_id"], data["window_time"],
    )

    test_mask = subject_ids == subject_id
    train_mask = ~test_mask

    if test_mask.sum() == 0:
        raise ValueError(f"No data found for subject S{subject_id}")

    # sort by time within this subject (should already be in order, but be safe)
    order = np.argsort(window_time[test_mask])

    X_raw_s = X_raw[test_mask][order]
    X_hrv_s = X_hrv[test_mask][order]
    y_s = y[test_mask][order]
    time_s = window_time[test_mask][order]

    scaler = StandardScaler()
    scaler.fit(X_hrv[train_mask])
    X_hrv_scaled = scaler.transform(X_hrv_s)

    model_path = os.path.join(config.MODELS_DIR, f"model_S{subject_id}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run train.py first -- it saves one "
            f"model per LOSO fold, which is what we reuse here."
        )
    model = CNNLSTMFusion(n_hrv_features=X_hrv.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    raw_t = torch.tensor(X_raw_s, dtype=torch.float32).unsqueeze(1)
    hrv_t = torch.tensor(X_hrv_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(raw_t, hrv_t)
        probs = torch.sigmoid(logits).numpy()

    return time_s, probs, y_s


def rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_trend_segments(time_min: np.ndarray, trend: np.ndarray, slope_threshold=0.01):
    """Simple slope-based rising/falling segment detection between
    consecutive rolling-average points. Returns a list of
    (start_min, end_min, direction) tuples for segments where the trend
    is consistently rising or falling."""
    segments = []
    if len(trend) < 2:
        return segments

    diffs = np.diff(trend)
    current_dir = None
    seg_start_idx = 0

    for i, d in enumerate(diffs):
        direction = "rising" if d > slope_threshold else (
            "falling" if d < -slope_threshold else "flat"
        )
        if direction != current_dir:
            if current_dir in ("rising", "falling") and i > seg_start_idx:
                segments.append((time_min[seg_start_idx], time_min[i], current_dir))
            seg_start_idx = i
            current_dir = direction

    if current_dir in ("rising", "falling") and len(time_min) - 1 > seg_start_idx:
        segments.append((time_min[seg_start_idx], time_min[-1], current_dir))

    # keep only segments lasting at least 2 minutes -- ignore tiny noise blips
    segments = [s for s in segments if (s[1] - s[0]) >= 2.0]
    return segments


def plot_subject_trend(subject_id: int):
    time_s, probs, y_true = load_subject_trend_data(subject_id)
    time_min = time_s / 60.0

    trend = rolling_average(probs, ROLLING_WINDOW)
    segments = detect_trend_segments(time_min, trend)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(time_min, probs, s=15, color="lightgray",
               label="Per-window predicted P(stress)", zorder=1)
    ax.plot(time_min, trend, color="#4C72B0", linewidth=2.5,
             label=f"Rolling trend (window={ROLLING_WINDOW})", zorder=2)

    # shade actual ground-truth stress periods for comparison
    in_stress = False
    seg_start = None
    for t, label in zip(time_min, y_true):
        if label == 1 and not in_stress:
            seg_start = t
            in_stress = True
        elif label == 0 and in_stress:
            ax.axvspan(seg_start, t, color="#C44E52", alpha=0.12)
            in_stress = False
    if in_stress:
        ax.axvspan(seg_start, time_min[-1], color="#C44E52", alpha=0.12)

    for start, end, direction in segments:
        color = "#C44E52" if direction == "rising" else "#55A868"
        ax.axvspan(start, end, color=color, alpha=0.08)
        ax.annotate(
            f"{direction}", xy=((start + end) / 2, 1.02), xycoords=("data", "axes fraction"),
            ha="center", fontsize=8, color=color, fontweight="bold",
        )

    ax.set_xlabel("Time into session (minutes)")
    ax.set_ylabel("Predicted P(stress)")
    ax.set_title(
        f"Subject S{subject_id} -- Within-Session Stress Trend\n"
        f"(pink shading = ground-truth stress periods; annotated bands = detected trend direction)"
    )
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(TREND_PLOTS_DIR, f"trend_S{subject_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved -> {out_path}")

    if segments:
        print(f"  Detected {len(segments)} trend segment(s) for S{subject_id}:")
        for start, end, direction in segments:
            print(f"    {start:.1f}-{end:.1f} min: stress {direction}")
    else:
        print(f"  No sustained rising/falling trend segments detected for S{subject_id} "
              f"(stress level stayed relatively flat throughout).")


if __name__ == "__main__":
    os.makedirs(TREND_PLOTS_DIR, exist_ok=True)

    data = np.load(config.PROCESSED_DATASET_PATH)
    subjects_present = sorted(set(data["subject_id"].tolist()))

    print(f"Generating within-session trend plots for {len(subjects_present)} subjects...")
    print("(Scope note: this shows trend WITHIN one recorded session, not "
          "multi-day/burnout trend -- see the module docstring for why.)\n")

    for sid in subjects_present:
        print(f"Subject S{sid}:")
        plot_subject_trend(sid)
        print()

    print(f"All trend plots saved in: {TREND_PLOTS_DIR}")
    print("Pick 1-2 representative examples (e.g. one clear case, one flat "
          "case) for your report rather than including all of them.")
