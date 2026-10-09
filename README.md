# PPG Stress Detection — WESAD (CNN-LSTM + HRV fusion, LOSO validated, SHAP-explained)

## 0. What this repo does
- Loads raw WESAD `.pkl` subject files (wrist Empatica E4 data)
- Filters + segments the BVP (PPG) signal into fixed-length windows
- Extracts HRV features per window (RMSSD, SDNN, pNN50, LF/HF, mean HR)
- Trains a CNN+LSTM branch on raw BVP fused with an HRV feature branch
- Validates with **Leave-One-Subject-Out (LOSO)**, not leaky k-fold
- Explains predictions with SHAP (HRV branch) and saliency (CNN branch)

## 1. Get the dataset from Kaggle
1. Go to Kaggle and download one of the WESAD mirrors, e.g. search "WESAD Wearable Stress and Affect Detection" — look for a dataset that contains **per-subject `.pkl` files** (`S2.pkl`, `S3.pkl`, ... `S17.pkl` — note there's no S1 or S12 in the original release). Avoid pre-extracted-feature CSV versions for now; we want raw signal access for the CNN branch.
2. Unzip the download so the `.pkl` files sit directly in `data/raw/`, i.e.:
   ```
   data/raw/S2.pkl
   data/raw/S3.pkl
   ...
   data/raw/S17.pkl
   ```
3. **If your Kaggle copy only gives you CSVs / already-extracted features**, tell me — the pipeline will need to change (skip the raw-signal CNN branch or adapt column names). Check `python src/data_loader.py --inspect` after downloading; it'll print what it finds.

## 2. VS Code setup
```bash
# from the project root
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick the `.venv` one you just created.

## 3. Run the pipeline, in order
```bash
# 1. Sanity-check that your pkl files load correctly
python src/data_loader.py --inspect

# 2. Build the processed dataset (filtering, segmentation, HRV features)
python src/build_dataset.py

# 3. Train with LOSO cross-validation
python src/train.py

# 4. Run explainability (after training, uses saved models)
python src/explain.py
```

## 4. Project structure
```
data/raw/          <- put downloaded WESAD .pkl files here (not committed to git)
data/processed/    <- output of build_dataset.py
src/
  config.py        <- all hyperparameters / paths in one place
  data_loader.py    <- reads raw .pkl files
  preprocessing.py  <- filtering, segmentation, HRV extraction
  build_dataset.py  <- orchestrates loading -> saves data/processed/dataset.npz
  dataset.py        <- PyTorch Dataset wrapper
  model.py          <- CNN + LSTM + HRV fusion architecture
  train.py           <- LOSO training loop, saves per-fold models + metrics
  explain.py         <- SHAP (HRV branch) + saliency (CNN branch)
models/             <- saved model weights per LOSO fold
notebooks/          <- optional scratch space for exploration
```

## 5. Notes on what's already decided (and why)
- **Binary classification** (stress vs non-stress): baseline + amusement + meditation are merged into "non-stress." This matches the standard framing used in most WESAD papers, and is what your project doc proposes.
- **LOSO validation**: fold `i` trains on all subjects except subject `i`, tests on subject `i`, repeated for every subject. This is slower than k-fold but is the only honest way to report accuracy on WESAD — mention this explicitly in your report as your methodological rigor contribution.
- **Per-subject z-score normalization**: normalizing each subject's own signal separately (not globally) because PPG baseline amplitude varies a lot across sensors/skin tone/placement.
