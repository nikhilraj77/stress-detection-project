"""PyTorch Dataset wrapper around the processed .npz arrays."""
import numpy as np
import torch
from torch.utils.data import Dataset


class WESADDataset(Dataset):
    def __init__(self, X_raw, X_hrv, y, hrv_scaler=None):
        """
        X_raw: (N, window_samples) raw BVP windows
        X_hrv: (N, n_hrv_features) HRV feature vectors
        y: (N,) binary labels
        hrv_scaler: optional sklearn-style scaler already fit on TRAIN data,
                    used to transform X_hrv (avoids leaking test stats)
        """
        if hrv_scaler is not None:
            X_hrv = hrv_scaler.transform(X_hrv)

        self.X_raw = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.X_hrv = torch.tensor(X_hrv, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_raw[idx], self.X_hrv[idx], self.y[idx]
