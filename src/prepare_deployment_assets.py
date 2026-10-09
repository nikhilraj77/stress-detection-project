"""
Run this ONCE before using real-time mode. Live/replay deployment has no
"held-out subject" concept the way LOSO training folds did -- the person
being monitored live is, by definition, not any of the 11 training
subjects. So instead of reusing one fold's scaler (which was fit
excluding one specific subject), we fit a fresh HRV scaler on ALL
available subjects combined, and save it for real-time use.

This does NOT retrain the model -- it only prepares a scaler artifact.
The 11 already-trained model_S*.pt files are reused as-is (see
realtime_engine.py, which ensembles all 11 for inference).
"""
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

import config


def prepare():
    data = np.load(config.PROCESSED_DATASET_PATH)
    X_hrv = data["X_hrv"]

    scaler = StandardScaler()
    scaler.fit(X_hrv)

    out_path = f"{config.MODELS_DIR}/deployment_scaler.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Fit HRV scaler on {X_hrv.shape[0]} windows across all available subjects.")
    print(f"Saved -> {out_path}")
    print("This is used by realtime_engine.py for both Real-Time Replay Mode "
          "and (later) Live Sensor Mode.")


if __name__ == "__main__":
    prepare()
