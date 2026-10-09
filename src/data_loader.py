"""
Loads raw WESAD .pkl files.

Expected structure of each SX.pkl (standard WESAD release):
    {
        'signal': {
            'wrist': {'BVP': array, 'EDA': array, 'TEMP': array, 'ACC': array},
            'chest': {...}
        },
        'label': array,   # sampled at 700 Hz, synced across the whole session
        'subject': 'SX'
    }

If your Kaggle copy has a different structure, run this file with --inspect
first and adjust WRIST_BVP_KEY / LABEL_KEY below to match.
"""
import argparse
import os
import pickle

import numpy as np

import config


def load_subject_pkl(subject_id: int) -> dict:
    """Load a single subject's raw pickle file."""
    path = os.path.join(config.RAW_DATA_DIR, f"S{subject_id}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Make sure you've unzipped the Kaggle "
            f"download into data/raw/ so files are named S2.pkl, S3.pkl, etc."
        )
    with open(path, "rb") as f:
        # WESAD pkl files were pickled in Python 2 -> need latin1 encoding
        data = pickle.load(f, encoding="latin1")
    return data


def get_wrist_bvp(subject_data: dict) -> np.ndarray:
    """Extract the raw wrist BVP (PPG) signal."""
    return np.asarray(subject_data["signal"]["wrist"]["BVP"]).flatten()


def get_labels(subject_data: dict) -> np.ndarray:
    """Extract the label channel (700 Hz)."""
    return np.asarray(subject_data["label"]).flatten()


def inspect_all_subjects():
    """Quick sanity check: confirm every expected subject file loads and
    print basic shape / label distribution info. Run this FIRST after
    downloading the dataset, before anything else."""
    print(f"Looking for raw files in: {config.RAW_DATA_DIR}\n")
    found, missing = [], []

    for sid in config.SUBJECT_IDS:
        path = os.path.join(config.RAW_DATA_DIR, f"S{sid}.pkl")
        if os.path.exists(path):
            found.append(sid)
        else:
            missing.append(sid)

    print(f"Found {len(found)}/{len(config.SUBJECT_IDS)} subject files.")
    if missing:
        print(f"Missing subjects: {missing}")
        print("(This is fine to proceed with fewer subjects, but note it "
              "in your report as a dataset limitation.)\n")

    if not found:
        print("No files found — check your data/raw/ folder and file names.")
        return

    # deep-inspect the first subject we found
    sid = found[0]
    print(f"\n--- Deep inspection of subject S{sid} ---")
    data = load_subject_pkl(sid)
    print("Top-level keys:", list(data.keys()))
    print("signal keys:", list(data["signal"].keys()))
    print("wrist keys:", list(data["signal"]["wrist"].keys()))

    bvp = get_wrist_bvp(data)
    labels = get_labels(data)
    print(f"\nBVP shape: {bvp.shape}  (expected ~64Hz * session length)")
    print(f"Label shape: {labels.shape}  (expected ~700Hz * session length)")
    print(f"Session length (BVP): {len(bvp) / config.BVP_FS / 60:.1f} minutes")

    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution (raw 700Hz label channel):")
    for u, c in zip(unique, counts):
        print(f"  label {u}: {c} samples ({c / config.LABEL_FS:.1f} sec)")

    print("\nIf this looks reasonable (BVP is a 1D float array, labels "
          "include values 1-4 for baseline/stress/amusement/meditation), "
          "you're good to proceed to build_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                         help="Run a sanity check over all subject files")
    args = parser.parse_args()

    if args.inspect:
        inspect_all_subjects()
    else:
        print("Nothing to do. Run with --inspect to sanity-check your data, "
              "or import this module's functions from other scripts.")
