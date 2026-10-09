"""
Leave-One-Subject-Out (LOSO) training and evaluation.

For each subject S:
  - train on all other subjects
  - test on S
  - save the model, and metrics for that fold

Also logs parameter count and per-window CPU inference latency, since these
are part of the "lightweight, honestly validated" contribution from the
project plan (Option C).
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import config
from dataset import WESADDataset
from model import CNNLSTMFusion


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_fold(X_raw_train, X_hrv_train, y_train,
                    X_raw_test, X_hrv_test, y_test,
                    device, fold_name):
    # fit HRV scaler on TRAIN ONLY to avoid leaking test-subject stats
    hrv_scaler = StandardScaler()
    hrv_scaler.fit(X_hrv_train)

    train_ds = WESADDataset(X_raw_train, X_hrv_train, y_train, hrv_scaler)
    test_ds = WESADDataset(X_raw_test, X_hrv_test, y_test, hrv_scaler)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = CNNLSTMFusion(n_hrv_features=X_hrv_train.shape[1]).to(device)

    # handle class imbalance (stress is usually the minority class)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    best_test_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for x_raw, x_hrv, y in train_loader:
            x_raw, x_hrv, y = x_raw.to(device), x_hrv.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_raw, x_hrv)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_raw, x_hrv, y in test_loader:
                x_raw, x_hrv, y = x_raw.to(device), x_hrv.to(device), y.to(device)
                logits = model(x_raw, x_hrv)
                loss = criterion(logits, y)
                test_loss += loss.item() * len(y)
        test_loss /= len(test_ds)

        print(f"    [{fold_name}] epoch {epoch+1}/{config.NUM_EPOCHS} "
              f"train_loss={train_loss:.4f} test_loss={test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"    [{fold_name}] early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # --- final evaluation metrics ---
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for x_raw, x_hrv, y in test_loader:
            x_raw, x_hrv = x_raw.to(device), x_hrv.to(device)
            logits = model(x_raw, x_hrv)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # happens if test subject has only one class present

    # --- latency benchmark (single-window CPU inference, Option C metric) ---
    cpu_model = CNNLSTMFusion(n_hrv_features=X_hrv_train.shape[1])
    cpu_model.load_state_dict(best_state)
    cpu_model.eval().to("cpu")
    sample_raw = train_ds.X_raw[:1]
    sample_hrv = train_ds.X_hrv[:1]
    timings = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            cpu_model(sample_raw, sample_hrv)
            timings.append(time.perf_counter() - t0)
    latency_ms = float(np.median(timings) * 1000)

    metrics = {
        "fold": fold_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "n_test_windows": len(all_labels),
        "n_params": model.count_parameters(),
        "cpu_inference_latency_ms": latency_ms,
    }
    return model, hrv_scaler, metrics


def run_loso():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    if not os.path.exists(config.PROCESSED_DATASET_PATH):
        raise FileNotFoundError(
            f"{config.PROCESSED_DATASET_PATH} not found. Run "
            f"`python src/build_dataset.py` first."
        )

    data = np.load(config.PROCESSED_DATASET_PATH)
    X_raw, X_hrv, y, subject_id = (
        data["X_raw"], data["X_hrv"], data["y"], data["subject_id"]
    )

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    all_fold_metrics = []

    subjects_present = sorted(set(subject_id.tolist()))
    for test_subject in subjects_present:
        fold_name = f"S{test_subject}"
        print(f"\n=== LOSO fold: holding out subject {fold_name} ===")

        train_mask = subject_id != test_subject
        test_mask = subject_id == test_subject

        if y[test_mask].sum() == 0 or y[test_mask].sum() == test_mask.sum():
            print(f"  Note: {fold_name} has only one class in test set "
                  f"(AUC will be undefined for this fold).")

        model, scaler, metrics = train_one_fold(
            X_raw[train_mask], X_hrv[train_mask], y[train_mask],
            X_raw[test_mask], X_hrv[test_mask], y[test_mask],
            device, fold_name,
        )

        print(f"  -> {fold_name}: acc={metrics['accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} auc={metrics['auc']:.3f} "
              f"params={metrics['n_params']} "
              f"latency={metrics['cpu_inference_latency_ms']:.2f}ms")

        torch.save(model.state_dict(),
                   os.path.join(config.MODELS_DIR, f"model_{fold_name}.pt"))
        all_fold_metrics.append(metrics)

    # --- aggregate results ---
    accs = [m["accuracy"] for m in all_fold_metrics]
    f1s = [m["f1"] for m in all_fold_metrics]
    aucs = [m["auc"] for m in all_fold_metrics if not np.isnan(m["auc"])]

    summary = {
        "per_fold": all_fold_metrics,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "std_auc": float(np.std(aucs)) if aucs else None,
    }

    with open(os.path.join(config.MODELS_DIR, "loso_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Mean accuracy: {summary['mean_accuracy']:.3f} +/- {summary['std_accuracy']:.3f}")
    print(f"Mean F1:       {summary['mean_f1']:.3f} +/- {summary['std_f1']:.3f}")
    if summary["mean_auc"] is not None:
        print(f"Mean AUC:      {summary['mean_auc']:.3f} +/- {summary['std_auc']:.3f}")
    print(f"\nFull results saved to {config.MODELS_DIR}/loso_results.json")
    print("\nReport BOTH mean and std in your project report -- this is what "
          "makes LOSO validation more credible than a single accuracy number.")


if __name__ == "__main__":
    run_loso()
