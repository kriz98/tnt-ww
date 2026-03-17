# Reproducible pipeline (preprocessing + TabPFN / AutoTabPFN evaluation)

This repository contains code to reproduce the *analysis workflow* described in the manuscript:
1) **Preprocessing (R)** to harmonize two cohorts into a shared feature space and export encoded CSVs  
2) **Modeling + evaluation (Python)** using TabPFN and AutoTabPFN with:
   - 10-fold out-of-fold (OOF) predictions on the development cohort
   - Train on full development cohort (TNT+TME) → external validation on the held-out cohort (TNT+W/W)
   - Bootstrap 95% CIs for AUROC / AUPRC / Brier
   - Optional post-hoc Platt scaling on external predictions

No patient-level data is included here. You must provide de-identified input tables locally.

A GUI to interact with our trianed model from Varghese, Ng et al can be accessed here: https://019c2071-2788-bd90-6439-393728369ded.share.connect.posit.cloud/.<img width="468" height="14" alt="image" src="https://github.com/user-attachments/assets/3f143713-946e-43fc-b843-d3b78cfaa1fe" />
 
---

## Repository structure

- `preprocess.R`  
  Reads two local tables (development + external), harmonizes variable coding, one-hot encodes categorical predictors, and writes:
  - `data/dev_encoded.csv`
  - `data/ext_encoded.csv`

- `analysis.py`  
  Reads the encoded CSVs, runs TabPFN and AutoTabPFN, computes bootstrap CIs, prints a summary table, and saves fitted models + calibration params.

---

## Inputs (you provide locally)

Place your de-identified input files in `data/` and set file names in `preprocess.R`:

- `data/dev_raw.csv` (development cohort)
- `data/ext_raw.csv` (external cohort)

**Requirements**
- One row per subject
- Columns include:
  - predictors (numeric/categorical)
  - a binary outcome column

---

## Outputs

After running preprocessing:
- `data/dev_encoded.csv`
- `data/ext_encoded.csv`

After running Python analysis:
- `results/metrics_summary.csv`
- `results/tabpfn_fitted_on_dev.pkl`
- `results/autotabpfn_fitted_on_dev.pkl`
- `results/platt_params_external.pkl` (optional)

---

## How to run

### 1) R preprocessing
From the repository root:

```r
source("preprocess.R")
```

### 2) Python analysis

```bash
python analysis.py
```

---
