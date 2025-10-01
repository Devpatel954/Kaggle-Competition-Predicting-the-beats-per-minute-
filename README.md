# Kaggle Playground S5E9 — Predicting Beats-per-Minute (BPM)

Result: **Top 24%** · Public LB **RMSE 26.3877**  


## Overview
Lightweight, CPU-only tabular pipeline that avoids one-hot encoding and fits within Kaggle’s RAM limits.  
Core ideas: frequency + OOF target encoding for categoricals, stratified CV on binned target, LGBM/XGBoost with early stopping, NNLS blending, simple calibration, and duplicate-row signature override.

## What’s inside
- **Preprocessing:** median imputation, optional winsorization (clip extremes), all arrays `float32`.
- **Categorical encoding:**  
  - **Frequency encoding** (category counts)  
  - **OOF target encoding** (leak-safe mean target per category)
- **CV:** **StratifiedKFold** on binned `y` for stable OOF.
- **Models:**  
  - **LightGBM** (bagged across 2 seeds)  
  - **XGBoost** (hist, regularized)  
- **Ensemble:** **NNLS** (non-negative least squares) + **mean/variance calibration**.  
- **Extra:** **Exact duplicate row signature override** (use train mean when test row exactly matches a train row).

## Reproduce on Kaggle
1. Open the competition → **Code → New Notebook**.  
2. Paste the notebook from this repo and **Run All**.  
3. Submit generated files:
   - `submission.csv` (ensemble)
   - `submission_lgbm.csv` (single-model hedge)

> For quick tests: set `N_SPLITS=3`, `EARLY=200`, `N_EST=12000`.  
> For final runs: use `5 / 300 / 20000`.

## Run locally
```bash
pip install -r requirements.txt
# (Download competition data separately from Kaggle)
jupyter notebook notebook.ipynb
