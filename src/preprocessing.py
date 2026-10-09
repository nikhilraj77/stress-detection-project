"""
Preprocessing steps for the raw BVP signal:
  1. Bandpass filter (remove baseline wander + high freq noise)
  2. Per-subject z-score normalization
  3. Windowing / segmentation aligned with the label channel
  4. HRV feature extraction per window (using neurokit2)
"""
import numpy as np
from scipy.signal import butter, filtfilt
 
import config
 
try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    print("WARNING: neurokit2 not installed. HRV feature extraction will be "
          "skipped. Run: pip install neurokit2")
 
 
def bandpass_filter(signal: np.ndarray, fs: int = config.BVP_FS) -> np.ndarray:
    """Butterworth bandpass filter to isolate the physiologically relevant
    PPG frequency band and remove drift/noise."""
    nyq = 0.5 * fs
    low = config.BANDPASS_LOW / nyq
    high = config.BANDPASS_HIGH / nyq
    b, a = butter(config.FILTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal)
 
 
def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Per-subject normalization -- important because raw PPG amplitude
    varies a lot across people/sensor placement."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        return signal - mean
    return (signal - mean) / std
 
 
def majority_label_for_window(label_window_700hz: np.ndarray) -> int:
    """Given the 700Hz label slice corresponding to a BVP window, return the
    majority label. Returns -1 if the window doesn't have a clear majority
    valid label (i.e. mostly transient/undefined) -- caller should drop these."""
    valid_mask = np.isin(label_window_700hz, list(config.VALID_LABELS))
    if valid_mask.sum() < 0.5 * len(label_window_700hz):
        # more than half the window is transient/undefined -> discard
        return -1
    valid_labels = label_window_700hz[valid_mask]
    values, counts = np.unique(valid_labels, return_counts=True)
    return int(values[np.argmax(counts)])
 
 
def segment_signal(bvp: np.ndarray, labels_700hz: np.ndarray):
    """Slice the filtered/normalized BVP signal into fixed windows, and
    assign each window a single majority label. Returns (windows, labels,
    window_start_times_sec).
 
    window_start_times_sec records where each window falls within the
    session timeline (seconds since session start) -- this is what makes
    within-session trend plotting possible later (see trend_analysis.py)."""
    windows, window_labels, window_times = [], [], []
 
    n_samples = len(bvp)
    label_ratio = config.LABEL_FS / config.BVP_FS  # samples-per-sample ratio
 
    start = 0
    while start + config.WINDOW_SAMPLES <= n_samples:
        end = start + config.WINDOW_SAMPLES
 
        bvp_window = bvp[start:end]
 
        # map this BVP window's sample range to the label channel's range
        label_start = int(start * label_ratio)
        label_end = int(end * label_ratio)
        label_window = labels_700hz[label_start:label_end]
 
        maj_label = majority_label_for_window(label_window)
        if maj_label != -1:
            windows.append(bvp_window)
            window_labels.append(maj_label)
            window_times.append(start / config.BVP_FS)  # seconds into session
 
        start += config.STEP_SAMPLES
 
    return np.array(windows), np.array(window_labels), np.array(window_times)
 
 
def extract_hrv_features(bvp_window: np.ndarray, fs: int = config.BVP_FS) -> np.ndarray:
    """Extract a compact HRV feature vector from one BVP window using
    neurokit2's PPG peak detection + HRV analysis.
 
    Returns a fixed-length vector: [mean_hr, rmssd, sdnn, pnn50, lf_hf_ratio]
    If HRV can't be computed for this window (e.g. too few peaks), returns
    zeros -- these windows should be flagged/dropped by the caller if needed.
    """
    n_features = 5
    if not HAS_NEUROKIT:
        return np.zeros(n_features)
 
    try:
        signals, info = nk.ppg_process(bvp_window, sampling_rate=fs)
        peaks = info["PPG_Peaks"]
 
        if len(peaks) < 3:
            return np.zeros(n_features)
 
        hrv_time = nk.hrv_time(peaks, sampling_rate=fs, show=False)
        rmssd = hrv_time["HRV_RMSSD"].values[0]
        sdnn = hrv_time["HRV_SDNN"].values[0]
        pnn50 = hrv_time["HRV_pNN50"].values[0]
 
        mean_hr = np.mean(signals["PPG_Rate"])
 
        try:
            hrv_freq = nk.hrv_frequency(peaks, sampling_rate=fs, show=False)
            lf = hrv_freq["HRV_LF"].values[0]
            hf = hrv_freq["HRV_HF"].values[0]
            lf_hf = lf / hf if hf and hf > 1e-8 else 0.0
        except Exception:
            lf_hf = 0.0
 
        feats = np.array([mean_hr, rmssd, sdnn, pnn50, lf_hf])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats
 
    except Exception:
        # PPG processing can fail on noisy/short windows -- fail gracefully
        return np.zeros(n_features)