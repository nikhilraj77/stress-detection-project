"""
PPG Stream Interface -- the seam that makes "where does the signal come
from" independent of everything downstream (buffering, preprocessing,
model inference). Three implementations:

  WESADReplayStream    -- replays one WESAD subject's real recorded session
  GenericFileReplayStream -- replays ANY raw PPG signal you provide (a CSV
                             or .npy file), at whatever sampling rate you
                             tell it -- this is what lets you feed in a new
                             dataset without retraining anything
  LiveSensorStream      -- stub for a real physical sensor (e.g. MAX30102),
                            not implemented yet, same interface as the above

All three yield samples already resampled to config.BVP_FS (64 Hz), so
nothing downstream (rolling_buffer.py, realtime_engine.py) needs to know
or care which source it's reading from.
"""
import time
from abc import ABC, abstractmethod

import numpy as np

import config
from data_loader import load_subject_pkl, get_wrist_bvp, get_labels


class PPGStreamSource(ABC):
    """Common interface every stream source must implement."""

    @abstractmethod
    def open(self):
        """Prepare the source (open a file, connect to hardware, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self, chunk_seconds: float):
        """Return up to `chunk_seconds` worth of new samples as a 1D numpy
        array at config.BVP_FS, plus optional matching labels (or None if
        the source has no ground truth). Returns (samples, labels_or_None).
        Returns (None, None) when the stream is exhausted."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def has_ground_truth(self) -> bool:
        """Whether this source can supply real labels for comparison
        (WESAD replay: yes. A new uploaded file / live sensor: no)."""
        return False


class WESADReplayStream(PPGStreamSource):
    """Replays one WESAD subject's real recorded BVP session, at real
    playback speed (or faster, via speed_multiplier, for a quicker demo)."""

    def __init__(self, subject_id: int, speed_multiplier: float = 1.0):
        self.subject_id = subject_id
        self.speed_multiplier = speed_multiplier
        self.bvp = None
        self.labels_700hz = None
        self._pos = 0  # current read position, in BVP-rate samples

    def open(self):
        data = load_subject_pkl(self.subject_id)
        self.bvp = get_wrist_bvp(data)  # already native 64 Hz, WESAD's own rate
        self.labels_700hz = get_labels(data)
        self._pos = 0

    def has_ground_truth(self) -> bool:
        return True

    def read_chunk(self, chunk_seconds: float):
        if self.bvp is None:
            raise RuntimeError("Stream not opened -- call open() first.")

        n_samples = int(chunk_seconds * config.BVP_FS)
        end = self._pos + n_samples
        if self._pos >= len(self.bvp):
            return None, None  # exhausted

        chunk = self.bvp[self._pos:end]

        # map this BVP sample range to the 700Hz label channel to fetch
        # matching ground-truth labels for the same time span
        label_ratio = config.LABEL_FS / config.BVP_FS
        lstart = int(self._pos * label_ratio)
        lend = int(end * label_ratio)
        label_chunk = self.labels_700hz[lstart:lend]

        self._pos = end
        return chunk, label_chunk

    def close(self):
        self.bvp = None
        self.labels_700hz = None


class GenericFileReplayStream(PPGStreamSource):
    """Replays ANY raw PPG signal from a CSV or .npy file, resampled to
    config.BVP_FS. This is what lets you plug in a brand-new dataset (not
    WESAD) and replay it through the exact same trained model, with zero
    retraining -- as long as it's a genuine raw PPG/BVP-like waveform.

    CSV files: pass `column` (name or index) to pick which column holds
    the signal; if there's only one column you can leave it as None.
    """

    def __init__(self, file_path: str, native_fs: float, column=None):
        self.file_path = file_path
        self.native_fs = native_fs
        self.column = column
        self.resampled_signal = None
        self._pos = 0

    def open(self):
        if self.file_path.endswith(".npy"):
            raw = np.load(self.file_path).astype(float).flatten()
        elif self.file_path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(self.file_path)
            if self.column is not None:
                col = df[self.column] if isinstance(self.column, str) else df.iloc[:, self.column]
            else:
                col = df.iloc[:, 0]
            raw = col.to_numpy(dtype=float)
        else:
            raise ValueError(
                f"Unsupported file type for {self.file_path}. Use .csv or .npy."
            )

        # resample from whatever rate the file was recorded at, to our
        # standard 64 Hz working rate, via linear interpolation over time
        n_native = len(raw)
        duration_sec = n_native / self.native_fs
        native_times = np.linspace(0, duration_sec, n_native)
        target_times = np.linspace(0, duration_sec, int(duration_sec * config.BVP_FS))
        self.resampled_signal = np.interp(target_times, native_times, raw)
        self._pos = 0

    def has_ground_truth(self) -> bool:
        return False  # a generic uploaded file has no WESAD-style labels

    def read_chunk(self, chunk_seconds: float):
        if self.resampled_signal is None:
            raise RuntimeError("Stream not opened -- call open() first.")

        n_samples = int(chunk_seconds * config.BVP_FS)
        end = self._pos + n_samples
        if self._pos >= len(self.resampled_signal):
            return None, None

        chunk = self.resampled_signal[self._pos:end]
        self._pos = end
        return chunk, None  # no ground truth available

    def close(self):
        self.resampled_signal = None


class LiveSensorStream(PPGStreamSource):
    """Placeholder for a real physical PPG sensor (e.g. MAX30102 + ESP32
    streaming over serial/BLE). Implements the same interface as the replay
    streams above, so realtime_engine.py and the dashboard don't need any
    changes to switch from replay to a live sensor later -- only this class
    needs to be filled in.
    """

    def __init__(self, port: str = None):
        self.port = port

    def open(self):
        raise NotImplementedError(
            "Live Sensor Mode isn't built yet -- this is a placeholder for "
            "when a physical PPG sensor (e.g. MAX30102) is connected. The "
            "rest of the pipeline (buffering, preprocessing, inference, "
            "dashboard) already works with any PPGStreamSource, so wiring "
            "this up later won't require changing anything downstream."
        )

    def has_ground_truth(self) -> bool:
        return False

    def read_chunk(self, chunk_seconds: float):
        raise NotImplementedError

    def close(self):
        pass


def simulate_realtime_delay(chunk_seconds: float, speed_multiplier: float = 1.0):
    """Sleep for the real-world duration a chunk represents, divided by a
    speed multiplier (e.g. 4.0 = 4x faster than real time), so replay mode
    genuinely feels like a live stream rather than an instant dump."""
    if speed_multiplier <= 0:
        return
    time.sleep(chunk_seconds / speed_multiplier)
