"""
Generates report-ready plots from the LOSO results, WITHOUT retraining:
  1. Per-fold accuracy / F1 / AUC bar charts (grouped)
  2. Aggregate confusion matrix (pooled predictions across all folds)
  3. Aggregate ROC curve (pooled across all folds)
  4. A summary metrics table saved as both CSV and a plotted image

Run this AFTER train.py has produced models/loso_results.json and the
per-fold model weights in models/.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

import config
from model import CNNLSTMFusion

PLOTS_DIR = os.path.join(config.MODELS_DIR, "report_plots")


def load_results():
    path = os.path.join(config.MODELS_DIR, "loso_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Run train.py first to produce loso_results.json")
    with open(path) as f:
        return json.load(f)


def plot_per_fold_metrics(summary):
    per_fold = summary["per_fold"]
    folds = [m["fold"] for m in per_fold]
    acc = [m["accuracy"] for m in per_fold]
    f1 = [m["f1"] for m in per_fold]
    auc_vals = [m["auc"] if not np.isnan(m["auc"]) else 0 for m in per_fold]

    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, acc, width, label="Accuracy", color="#4C72B0")
    ax.bar(x, f1, width, label="F1 Score", color="#DD8452")
    ax.bar(x + width, auc_vals, width, label="AUC", color="#55A868")

    ax.axhline(summary["mean_accuracy"], color="#4C72B0", linestyle="--", alpha=0.5,
               label=f"Mean Acc = {summary['mean_accuracy']:.3f}")
    ax.axhline(summary["mean_f1"], color="#DD8452", linestyle="--", alpha=0.5,
               label=f"Mean F1 = {summary['mean_f1']:.3f}")

    ax.set_xlabel("Held-out subject (LOSO fold)")
    ax.set_ylabel("Score")
    ax.set_title("Per-Subject LOSO Performance: Accuracy, F1, AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "per_fold_metrics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_summary_bar(summary):
    """Single clean bar chart with error bars -- good for a slide/summary."""
    metrics = ["Accuracy", "F1 Score", "AUC"]
    means = [summary["mean_accuracy"], summary["mean_f1"], summary["mean_auc"]]
    stds = [summary["std_accuracy"], summary["std_f1"], summary["std_auc"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(metrics, means, yerr=stds, capsize=8,
                   color=["#4C72B0", "#DD8452", "#55A868"])
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.03,
                 f"{mean:.3f}", ha="center", fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Overall LOSO Cross-Validation Performance\n(mean \u00b1 std across all subjects)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "summary_metrics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def pooled_predictions(summary):
    """Reload every fold's saved model, run inference on its held-out
    subject, and pool all predictions together (this is the standard way
    to build one aggregate confusion matrix / ROC curve from LOSO)."""
    data = np.load(config.PROCESSED_DATASET_PATH)
    X_raw, X_hrv, y, subject_id = (
        data["X_raw"], data["X_hrv"], data["y"], data["subject_id"]
    )

    all_labels, all_probs = [], []

    for m in summary["per_fold"]:
        fold = m["fold"]
        test_subject = int(fold.replace("S", ""))
        train_mask = subject_id != test_subject
        test_mask = subject_id == test_subject

        scaler = StandardScaler()
        scaler.fit(X_hrv[train_mask])
        X_hrv_test_scaled = scaler.transform(X_hrv[test_mask])

        model = CNNLSTMFusion(n_hrv_features=X_hrv.shape[1])
        model.load_state_dict(torch.load(
            os.path.join(config.MODELS_DIR, f"model_{fold}.pt"),
            map_location="cpu",
        ))
        model.eval()

        raw_t = torch.tensor(X_raw[test_mask], dtype=torch.float32).unsqueeze(1)
        hrv_t = torch.tensor(X_hrv_test_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = model(raw_t, hrv_t)
            probs = torch.sigmoid(logits).numpy()

        all_labels.extend(y[test_mask].tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_probs):
    y_pred = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Non-Stress", "Stress"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Aggregate Confusion Matrix\n(pooled across all LOSO folds)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")

    return cm


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Aggregate ROC Curve\n(pooled across all LOSO folds)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def save_summary_table(summary, cm):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    rows = [
        ("Mean Accuracy", f"{summary['mean_accuracy']:.3f} \u00b1 {summary['std_accuracy']:.3f}"),
        ("Mean F1 Score", f"{summary['mean_f1']:.3f} \u00b1 {summary['std_f1']:.3f}"),
        ("Mean AUC", f"{summary['mean_auc']:.3f} \u00b1 {summary['std_auc']:.3f}"),
        ("Pooled Precision", f"{precision:.3f}"),
        ("Pooled Recall (Sensitivity)", f"{recall:.3f}"),
        ("Pooled Specificity", f"{specificity:.3f}"),
        ("Validation Protocol", "Leave-One-Subject-Out (LOSO)"),
        ("Num. Folds (Subjects)", str(len(summary["per_fold"]))),
        ("Model Parameters", f"{summary['per_fold'][0]['n_params']:,}"),
        ("CPU Inference Latency", f"{summary['per_fold'][0]['cpu_inference_latency_ms']:.2f} ms/window"),
    ]

    csv_path = os.path.join(PLOTS_DIR, "summary_table.csv")
    with open(csv_path, "w") as f:
        f.write("Metric,Value\n")
        for name, val in rows:
            f.write(f"{name},{val}\n")
    print(f"Saved -> {csv_path}")

    fig, ax = plt.subplots(figsize=(7, len(rows) * 0.5))
    ax.axis("off")
    table = ax.table(
        cellText=[[name, val] for name, val in rows],
        colLabels=["Metric", "Value"],
        loc="center", cellLoc="left", colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white")

    plt.title("Model Performance Summary", fontweight="bold", pad=20)
    plt.tight_layout()
    img_path = os.path.join(PLOTS_DIR, "summary_table.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {img_path}")


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    summary = load_results()

    print("Generating per-fold metric bar chart...")
    plot_per_fold_metrics(summary)

    print("Generating summary bar chart...")
    plot_summary_bar(summary)

    print("Reloading models to pool predictions for confusion matrix / ROC...")
    y_true, y_probs = pooled_predictions(summary)

    print("Generating confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_probs)

    print("Generating ROC curve...")
    plot_roc_curve(y_true, y_probs)

    print("Generating summary table...")
    save_summary_table(summary, cm)

    print(f"\nAll plots saved in: {PLOTS_DIR}")
    print("These are the ones to drop straight into your report / slides.")
