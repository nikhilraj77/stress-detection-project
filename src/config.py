"""
Central place for all paths and hyperparameters.
Change things here rather than hunting through every script.
"""
import os

# --- paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "dataset.npz")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# WESAD subject IDs actually present in the public release (no S1 or S12)
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# --- signal params ---
BVP_FS = 64          # Empatica E4 BVP sampling rate (Hz)
LABEL_FS = 700        # WESAD label channel sampling rate (Hz), synced to chest device

# WESAD label codes (from the official readme):
# 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
# 4 = meditation, 5/6/7 = should-be-ignored extra conditions
BASELINE_LABEL = 1
STRESS_LABEL = 2
AMUSEMENT_LABEL = 3
MEDITATION_LABEL = 4
VALID_LABELS = {BASELINE_LABEL, STRESS_LABEL, AMUSEMENT_LABEL, MEDITATION_LABEL}
NON_STRESS_LABELS = {BASELINE_LABEL, AMUSEMENT_LABEL, MEDITATION_LABEL}

# --- windowing ---
WINDOW_SECONDS = 60
WINDOW_OVERLAP = 0.5     # 50% overlap between consecutive windows
WINDOW_SAMPLES = WINDOW_SECONDS * BVP_FS
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP))

# --- filtering ---
BANDPASS_LOW = 0.5   # Hz
BANDPASS_HIGH = 8.0  # Hz
FILTER_ORDER = 3

# --- training ---
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 6
RANDOM_SEED = 42

DEVICE = "cuda"  # train.py will fall back to "cpu" automatically if unavailable
