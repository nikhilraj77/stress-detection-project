"""
Explainability, run AFTER train.py.

1. SHAP on the HRV branch: treats the model as a function of the 5 HRV
   features (holding a representative raw-signal input fixed-ish via a
   wrapper), so SHAP's KernelExplainer can attribute importance to
   mean_hr, rmssd, sdnn, pnn50, lf_hf_ratio.
2. Saliency on the CNN branch: gradient of the stress-probability output
   w.r.t. the raw BVP input, to see which parts of the waveform the model
   actually attends to.

Use one representative fold's model (pick the one with median performance,
or just the first fold) -- doing this for all 15 folds is unnecessary for
a project report; 1-3 folds' worth of explanation plots is plenty.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from sklearn.preprocessing import StandardScaler

import config
from model import CNNLSTMFusion


def load_fold(fold_name: str):
    data = np.load(config.PROCESSED_DATASET_PATH)
    X_raw, X_hrv, y, subject_id = (
        data["X_raw"], data["X_hrv"], data["y"], data["subject_id"]
    )
    test_subject = int(fold_name.replace("S", ""))
    test_mask = subject_id == test_subject
    train_mask = ~test_mask

    scaler = StandardScaler()
    scaler.fit(X_hrv[train_mask])

    model = CNNLSTMFusion(n_hrv_features=X_hrv.shape[1])
    model.load_state_dict(torch.load(
        os.path.join(config.MODELS_DIR, f"model_{fold_name}.pt"),
        map_location="cpu",
    ))
    model.eval()

    return model, scaler, X_raw[test_mask], X_hrv[test_mask], y[test_mask]


def shap_on_hrv_branch(model, scaler, X_raw_test, X_hrv_test, fold_name):
    """SHAP explanation for the HRV feature branch.

    We fix the raw-signal input to a representative "average" window (mean
    across the test set) and vary only the HRV features, so SHAP explains
    the HRV branch's contribution specifically. This is a reasonable
    simplification for a project report -- be upfront about it as a design
    choice, not a hidden shortcut.
    """
    feature_names = ["mean_hr", "rmssd", "sdnn", "pnn50", "lf_hf_ratio"]

    X_hrv_scaled = scaler.transform(X_hrv_test)
    representative_raw = torch.tensor(
        X_raw_test.mean(axis=0, keepdims=True), dtype=torch.float32
    ).unsqueeze(1)  # (1,1,L)

    def predict_fn(hrv_batch: np.ndarray) -> np.ndarray:
        hrv_tensor = torch.tensor(hrv_batch, dtype=torch.float32)
        raw_batch = representative_raw.repeat(len(hrv_batch), 1, 1)
        with torch.no_grad():
            logits = model(raw_batch, hrv_tensor)
            probs = torch.sigmoid(logits).numpy()
        return probs

    # background = small random sample of training-distribution-like HRV values
    background = X_hrv_scaled[np.random.choice(
        len(X_hrv_scaled), size=min(30, len(X_hrv_scaled)), replace=False
    )]
    explainer = shap.KernelExplainer(predict_fn, background)

    sample = X_hrv_scaled[:min(50, len(X_hrv_scaled))]
    shap_values = explainer.shap_values(sample, nsamples=100)

    plt.figure()
    shap.summary_plot(
        shap_values, sample, feature_names=feature_names, show=False
    )
    out_path = os.path.join(config.MODELS_DIR, f"shap_summary_{fold_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved SHAP summary plot -> {out_path}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda t: -t[1])
    print("  HRV feature importance ranking (mean |SHAP value|):")
    for name, val in ranking:
        print(f"    {name}: {val:.4f}")

    return ranking


def saliency_on_cnn_branch(model, X_raw_test, X_hrv_test, scaler, fold_name, n_examples=3):
    """Gradient-based saliency: for a few example windows, show which time
    points in the raw BVP the model's stress-probability output is most
    sensitive to."""
    X_hrv_scaled = scaler.transform(X_hrv_test)

    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]

    for i in range(n_examples):
        raw = torch.tensor(X_raw_test[i:i+1], dtype=torch.float32).unsqueeze(1)
        raw.requires_grad_(True)
        hrv = torch.tensor(X_hrv_scaled[i:i+1], dtype=torch.float32)

        logit = model(raw, hrv)
        prob = torch.sigmoid(logit)
        prob.backward()

        saliency = raw.grad.abs().squeeze().numpy()
        signal = raw.detach().squeeze().numpy()

        ax = axes[i]
        ax.plot(signal, color="gray", alpha=0.6, label="BVP signal")
        ax2 = ax.twinx()
        ax2.plot(saliency, color="red", alpha=0.8, label="saliency")
        ax.set_title(f"Example {i+1} -- predicted P(stress)={prob.item():.2f}")
        ax.set_ylabel("normalized BVP")
        ax2.set_ylabel("|gradient|")

    plt.tight_layout()
    out_path = os.path.join(config.MODELS_DIR, f"saliency_{fold_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved saliency plot -> {out_path}")


def physiological_sanity_check(ranking):
    """Print a short discussion comparing SHAP ranking against known stress
    physiology -- fill this in with your own written analysis for the report."""
    print("\n--- Physiological sanity check (fill in for your report) ---")
    print("Known physiology: acute stress is typically associated with")
    print("  - decreased RMSSD/SDNN (reduced parasympathetic/vagal activity)")
    print("  - increased LF/HF ratio (sympathetic dominance)")
    print("  - increased mean heart rate")
    print(f"\nModel's actual top feature: {ranking[0][0]}")
    print("Does this match expectation? Write 2-3 sentences in your report")
    print("discussing whether it does, and if not, what that might mean")
    print("(e.g. model relying on artifact/noise rather than physiology).")


if __name__ == "__main__":
    results_path = os.path.join(config.MODELS_DIR, "loso_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError("Run train.py first to produce trained models.")

    with open(results_path) as f:
        summary = json.load(f)

    # pick the fold with the median F1 score as "representative"
    per_fold = summary["per_fold"]
    per_fold_sorted = sorted(per_fold, key=lambda m: m["f1"])
    median_fold = per_fold_sorted[len(per_fold_sorted) // 2]["fold"]
    print(f"Using fold {median_fold} (median F1) for explainability analysis.\n")

    model, scaler, X_raw_test, X_hrv_test, y_test = load_fold(median_fold)

    print("Running SHAP on HRV branch...")
    ranking = shap_on_hrv_branch(model, scaler, X_raw_test, X_hrv_test, median_fold)

    print("\nRunning saliency on CNN branch...")
    saliency_on_cnn_branch(model, X_raw_test, X_hrv_test, scaler, median_fold)

    physiological_sanity_check(ranking)
