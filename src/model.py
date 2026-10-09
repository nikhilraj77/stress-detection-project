"""
CNN + LSTM + HRV fusion model.

Architecture:
  raw BVP window -> 1D CNN (local waveform patterns)
                  -> BiLSTM (temporal dynamics across the window)
                  -> pooled feature vector
  HRV features    -> small MLP branch
  [CNN-LSTM features ; HRV features] -> concat -> dense -> sigmoid

Kept intentionally small: WESAD has ~15 subjects, so a deep/wide network
will just overfit. Small models also support the "lightweight, deployable"
angle from the project plan (Option C).
"""
import torch
import torch.nn as nn


class CNNLSTMFusion(nn.Module):
    def __init__(self, n_hrv_features=5, cnn_channels=(16, 32, 64),
                 lstm_hidden=32, dropout=0.3):
        super().__init__()

        # --- CNN branch (raw BVP) ---
        c1, c2, c3 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3), nn.BatchNorm1d(c1),
            nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.BatchNorm1d(c2),
            nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(c2, c3, kernel_size=3, padding=1), nn.BatchNorm1d(c3),
            nn.ReLU(), nn.MaxPool1d(2),
        )

        # --- LSTM branch (temporal dynamics over CNN feature maps) ---
        self.lstm = nn.LSTM(
            input_size=c3, hidden_size=lstm_hidden,
            batch_first=True, bidirectional=True,
        )

        # --- HRV branch ---
        self.hrv_mlp = nn.Sequential(
            nn.Linear(n_hrv_features, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 16), nn.ReLU(),
        )

        # --- fusion head ---
        fused_dim = (lstm_hidden * 2) + 16
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_raw, x_hrv):
        # x_raw: (B, 1, L)
        cnn_out = self.cnn(x_raw)                   # (B, C, L')
        cnn_out = cnn_out.permute(0, 2, 1)           # (B, L', C) for LSTM

        lstm_out, (h_n, _) = self.lstm(cnn_out)
        # concat final forward + backward hidden states
        lstm_feat = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, 2*hidden)

        hrv_feat = self.hrv_mlp(x_hrv)               # (B, 16)

        fused = torch.cat([lstm_feat, hrv_feat], dim=1)
        logit = self.classifier(fused).squeeze(-1)    # (B,)
        return logit

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
