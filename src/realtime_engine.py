"""
The real-time engine: pulls from any PPGStreamSource, pushes into a
RollingBuffer, and on a sliding refresh interval, preprocesses the latest
60-second window and runs it through the ensemble of all 11 trained LOSO
models -- exactly the models you already trained, no retraining.

Runs as a background thread so a Streamlit dashboard (or anything else)
can poll `engine.get_latest_state()` at any time without blocking.
"""
import os
import pickle
import threading
import time

import numpy as np
import torch

import config
from model import CNNLSTMFusion
from preprocessing import bandpass_filter, zscore_normalize, extract_hrv_features
from rolling_buffer import RollingBuffer
from ppg_stream import PPGStreamSource, simulate_realtime_delay


def load_model_ensemble():
    """Load all 11 already-trained LOSO fold models. We ensemble them
    (average their predictions) for real-time/deployment use, since no
    single fold's model is "the" deployment model -- each was trained
    excluding a different subject."""
    models = []
    for sid in config.SUBJECT_IDS:
        path = os.path.join(config.MODELS_DIR, f"model_S{sid}.pt")
        if not os.path.exists(path):
            continue
        m = CNNLSTMFusion()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        models.append(m)
    if not models:
        raise RuntimeError(
            "No trained models found in models/. Run train.py first."
        )
    return models


def load_deployment_scaler():
    path = os.path.join(config.MODELS_DIR, "deployment_scaler.pkl")
    if not os.path.exists(path):
        raise RuntimeError(
            "deployment_scaler.pkl not found. Run "
            "prepare_deployment_assets.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


class RealtimeEngine:
    def __init__(self, stream: PPGStreamSource, refresh_seconds: float = 5.0,
                 chunk_seconds: float = 1.0, speed_multiplier: float = 1.0):
        """
        stream: any PPGStreamSource (WESAD replay, generic file replay, or
                later, a live sensor) -- the engine doesn't care which.
        refresh_seconds: how often a new prediction is produced, using a
                          SLIDING window over the latest 60s of buffered
                          signal (not waiting for a full new 60s window).
        chunk_seconds: how large a bite the stream is read in at a time
                        (smaller = smoother simulated real-time feel).
        speed_multiplier: 1.0 = real playback speed, 4.0 = 4x faster, etc.
                           Only meaningful for replay sources.
        """
        self.stream = stream
        self.refresh_seconds = refresh_seconds
        self.chunk_seconds = chunk_seconds
        self.speed_multiplier = speed_multiplier

        self.buffer = RollingBuffer()
        self.models = load_model_ensemble()
        self.scaler = load_deployment_scaler()

        self.history = []  # list of dicts: {time_sec, probability, ground_truth}
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_inference_time = -1e9
        self._latest_window = None
        self._latest_hrv = None
        self._error = None

    def _run_inference_on_window(self, window: np.ndarray):
        filtered = bandpass_filter(window)
        normalized = zscore_normalize(filtered)

        hrv_feats = extract_hrv_features(normalized).reshape(1, -1)
        hrv_scaled = self.scaler.transform(hrv_feats)

        raw_t = torch.tensor(normalized, dtype=torch.float32).reshape(1, 1, -1)
        hrv_t = torch.tensor(hrv_scaled, dtype=torch.float32)

        probs = []
        with torch.no_grad():
            for m in self.models:
                logit = m(raw_t, hrv_t)
                probs.append(torch.sigmoid(logit).item())
        return (
            float(np.mean(probs)),
            float(np.std(probs)),
            hrv_feats[0].copy(),
            normalized.copy(),
        )

    def _loop(self):
        self.stream.open()
        try:
            while self._running:
                chunk, label_chunk = self.stream.read_chunk(self.chunk_seconds)
                if chunk is None:
                    break  # stream exhausted (replay reached the end)

                self.buffer.push(chunk, label_chunk)
                simulate_realtime_delay(self.chunk_seconds, self.speed_multiplier)

                elapsed = self.buffer.elapsed_seconds
                if elapsed - self._last_inference_time < self.refresh_seconds:
                    continue
                if self.buffer.seconds_available() < config.WINDOW_SECONDS:
                    continue  # not enough history yet for a full window

                window, label_window = self.buffer.get_latest_window()
                mean_prob, std_prob, hrv_feats, normalized_window = self._run_inference_on_window(window)

                ground_truth = None
                if self.stream.has_ground_truth() and label_window:
                    valid = [l for l in label_window if l is not None]
                    if valid:
                        ground_truth = 1 if max(set(valid), key=valid.count) == config.STRESS_LABEL else 0

                with self._lock:
                    # Store the same normalized window that actually entered the model
                    # so the live dashboard waveform matches the training-time representation.
                    self._latest_window = np.asarray(normalized_window, dtype=np.float32).copy()
                    self._latest_hrv = np.asarray(hrv_feats, dtype=np.float32).copy()
                    self.history.append({
                        "time_sec": elapsed,
                        "probability": mean_prob,
                        "uncertainty": std_prob,
                        "ground_truth": ground_truth,
                    })
                self._last_inference_time = elapsed
        except Exception as exc:
            # Surface background-thread errors to the dashboard instead of
            # silently leaving it stuck at "waiting for inference".
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self.stream.close()
            self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self.history = []
        self._last_inference_time = -1e9
        self._latest_window = None
        self._latest_hrv = None
        self._error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._running

    def get_latest_state(self):
        """Thread-safe snapshot of everything the dashboard needs right now."""
        with self._lock:
            history_copy = list(self.history)
            error = self._error
        latest = history_copy[-1] if history_copy else None
        return {
            "history": history_copy,
            "latest": latest,
            "buffer_seconds_available": self.buffer.seconds_available(),
            "is_running": self._running,
            "latest_window": None if self._latest_window is None else self._latest_window.copy(),
            "latest_hrv": None if self._latest_hrv is None else self._latest_hrv.copy(),
            "error": error,
        }
