"""
End-to-end dataset build:
  for each subject -> load -> filter -> normalize -> segment -> HRV features
  -> stack everything -> save to data/processed/dataset.npz
 
Output arrays:
  X_raw      : (N, WINDOW_SAMPLES)  filtered+normalized raw BVP windows
  X_hrv      : (N, 5)               HRV feature vectors
  y          : (N,)                 binary label (0=non-stress, 1=stress)
  subject_id : (N,)                 which subject each window came from
                                     (needed for LOSO cross-validation)
"""
import os
 
import numpy as np
from tqdm import tqdm
 
import config
from data_loader import load_subject_pkl, get_wrist_bvp, get_labels
from preprocessing import (
    bandpass_filter,
    zscore_normalize,
    segment_signal,
    extract_hrv_features,
)
 
 
def build():
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
 
    all_X_raw, all_X_hrv, all_y, all_subject_id, all_window_time = [], [], [], [], []
 
    for sid in tqdm(config.SUBJECT_IDS, desc="Processing subjects"):
        path = os.path.join(config.RAW_DATA_DIR, f"S{sid}.pkl")
        if not os.path.exists(path):
            print(f"  skipping S{sid} (file not found)")
            continue
 
        data = load_subject_pkl(sid)
        bvp_raw = get_wrist_bvp(data)
        labels_700hz = get_labels(data)
 
        bvp_filtered = bandpass_filter(bvp_raw)
        bvp_norm = zscore_normalize(bvp_filtered)
 
        windows, window_labels, window_times = segment_signal(bvp_norm, labels_700hz)
 
        if len(windows) == 0:
            print(f"  WARNING: no valid windows extracted for S{sid}")
            continue
 
        # binarize: stress vs everything else
        y_binary = np.array([
            1 if lbl == config.STRESS_LABEL else 0 for lbl in window_labels
        ])
 
        hrv_feats = np.array([
            extract_hrv_features(w) for w in windows
        ])
 
        all_X_raw.append(windows)
        all_X_hrv.append(hrv_feats)
        all_y.append(y_binary)
        all_subject_id.append(np.full(len(windows), sid))
        all_window_time.append(window_times)
 
        n_stress = y_binary.sum()
        print(f"  S{sid}: {len(windows)} windows "
              f"({n_stress} stress / {len(windows) - n_stress} non-stress)")
 
    if not all_X_raw:
        raise RuntimeError(
            "No data was processed. Check that data/raw/ contains the "
            "WESAD .pkl files (run `python src/data_loader.py --inspect` first)."
        )
 
    X_raw = np.concatenate(all_X_raw, axis=0)
    X_hrv = np.concatenate(all_X_hrv, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_id = np.concatenate(all_subject_id, axis=0)
    window_time = np.concatenate(all_window_time, axis=0)
 
    print(f"\nFinal dataset: {X_raw.shape[0]} windows total")
    print(f"  X_raw shape: {X_raw.shape}")
    print(f"  X_hrv shape: {X_hrv.shape}")
    print(f"  Class balance: {y.sum()} stress / {len(y) - y.sum()} non-stress "
          f"({100 * y.mean():.1f}% stress)")
    print(f"  Subjects included: {sorted(set(subject_id.tolist()))}")
 
    np.savez_compressed(
        config.PROCESSED_DATASET_PATH,
        X_raw=X_raw, X_hrv=X_hrv, y=y, subject_id=subject_id,
        window_time=window_time,
    )
    print(f"\nSaved to {config.PROCESSED_DATASET_PATH}")
 
 
if __name__ == "__main__":
    build()