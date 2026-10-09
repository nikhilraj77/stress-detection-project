"""
A sliding buffer that accumulates incoming PPG samples (from any
PPGStreamSource) and can hand back the latest 60-second window at any time
-- not just once every 60 seconds, but continuously, so predictions can
refresh every few seconds using a sliding window (like a live monitor)
rather than only once a full new window has arrived.
"""
from collections import deque

import numpy as np

import config


class RollingBuffer:
    def __init__(self, capacity_seconds: float = None, fs: int = config.BVP_FS):
        """capacity_seconds: how much history to retain. Defaults to just
        over one window's worth, so old data outside the window is dropped
        automatically and memory doesn't grow unbounded during a long replay."""
        self.fs = fs
        self.capacity_seconds = capacity_seconds or (config.WINDOW_SECONDS + 5)
        self.capacity_samples = int(self.capacity_seconds * fs)

        self.samples = deque(maxlen=self.capacity_samples)
        self.labels = deque(maxlen=self.capacity_samples)  # 700Hz-rate labels aligned separately
        self.elapsed_seconds = 0.0  # total real signal time pushed in, ever (for the trend x-axis)

    def push(self, chunk: np.ndarray, label_chunk: np.ndarray = None):
        """Add a new chunk of samples (and optional matching labels) to
        the buffer. Old samples beyond capacity are dropped automatically
        (deque maxlen)."""
        self.samples.extend(chunk.tolist())
        self.elapsed_seconds += len(chunk) / self.fs

        if label_chunk is not None and len(label_chunk) > 0:
            # labels may be at a different native rate (e.g. WESAD's 700Hz);
            # store the majority label for this chunk as one representative
            # value repeated per BVP sample, so it can be windowed the same way
            values, counts = np.unique(label_chunk, return_counts=True)
            majority = int(values[np.argmax(counts)])
            self.labels.extend([majority] * len(chunk))
        else:
            self.labels.extend([None] * len(chunk))

    def seconds_available(self) -> float:
        return len(self.samples) / self.fs

    def get_latest_window(self, window_seconds: float = config.WINDOW_SECONDS):
        """Return the most recent `window_seconds` of buffered signal, or
        None if not enough data has accumulated yet."""
        n_needed = int(window_seconds * self.fs)
        if len(self.samples) < n_needed:
            return None, None

        window = np.array(list(self.samples)[-n_needed:])
        label_window = list(self.labels)[-n_needed:]
        return window, label_window
