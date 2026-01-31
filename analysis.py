# analysis.py
# Minimal evaluation script
# - Reads one-hot encoded CSVs produced by preprocess.R
# - Runs TabPFN + AutoTabPFN with OOF + external validation on W/W cohort
# - Computes bootstrap 95% CIs for AUROC / AUPRC / Brier
# - Optionally fits Platt scaling on external predictions (saved)

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit

from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier


# -----------------------------
# User-configurable settings
# -----------------------------
DEV_PATH = "data/dev_encoded.csv"
EXT_PATH = "data/ext_encoded.csv"

OUTCOME_DEV = "outcome_dev"   # must match preprocess.R
OUTCOME_EXT = "outcome_ext"   # must match preprocess.R

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Utility: bootstrap 95% CIs
# -----------------------------
def bootstrap_ci(y, p, n_boot=1000, seed=42):
    y = np.asarray(y)
    p = np.asarray(p)

    rng = np.random.RandomState(seed)
    aucs, aps, brs = [], [], []

    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        yt, pt = y[idx], p[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, pt))
        aps.append(average_precision_score(yt, pt))
        brs.append(brier_score_loss(yt, pt))

    def ci(arr):
        return np.percentile(arr, [2.5, 97.5])

    return {
        "AUROC": (roc_auc_score(y, p), ci(aucs)),
        "AUPRC": (average_precision_score(y, p), ci(aps)),
        "Brier": (brier_score_loss(y, p), ci(brs)),
    }


# -----------------------------
# Utility: Platt scaling
# -----------------------------
def fit_platt_scaler(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    logits = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logits, y)

    alpha = float(lr.intercept_[0])
    beta = float(lr.coef_[0][0])
    p_cal = expit(alpha + beta * logits.ravel())
    return p_cal, {"alpha": alpha, "beta": beta}


# -----------------------------
# 1) Load encoded data
# -----------------------------
dev = pd.read_csv(DEV_PATH)
ext = pd.read_csv(EXT_PATH)

if OUTCOME_DEV not in dev.columns:
    raise ValueError(f"Missing outcome in dev: {OUTCOME_DEV}")
if OUTCOME_EXT not in ext.columns:
    raise ValueError(f"Missing outcome in ext: {OUTCOME_EXT}")

X_dev = dev.drop(columns=[OUTCOME_DEV])
y_dev = dev[OUTCOME_DEV].astype(int).to_numpy()

X_ext = ext.drop(columns=[OUTCOME_EXT])
y_ext = ext[OUTCOME_EXT].astype(int).to_numpy()

# Ensure same feature order
missing_in_ext = set(X_dev.columns) - set(X_ext.columns)
missing_in_dev = set(X_ext.columns) - set(X_dev.columns)
if missing_in_ext or missing_in_dev:
    raise ValueError("Feature mismatch between dev and ext. Re-run preprocess.R to harmonize.")

X_ext = X_ext[X_dev.columns]


# -----------------------------
# 2) TabPFN: OOF + external
# -----------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

tabpfn = TabPFNClassifier(random_state=42)

p_dev_tabpfn = cross_val_predict(
    tabpfn, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]

tabpfn.fit(X_dev, y_dev)
p_ext_tabpfn = tabpfn.predict_proba(X_ext)[:, 1]

tabpfn_dev = bootstrap_ci(y_dev, p_dev_tabpfn)
tabpfn_ext = bootstrap_ci(y_ext, p_ext_tabpfn)

joblib.dump(tabpfn, os.path.join(RESULTS_DIR, "tabpfn_fitted_on_dev.pkl"))


# -----------------------------
# 3) AutoTabPFN: OOF + external
# -----------------------------
autotabpfn = AutoTabPFNClassifier(max_time=120, device="cuda")

p_dev_auto = cross_val_predict(
    autotabpfn, X_dev.to_numpy(), y_dev, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]

autotabpfn.fit(X_dev.to_numpy(), y_dev)
p_ext_auto = autotabpfn.predict_proba(X_ext.to_numpy())[:, 1]

auto_dev = bootstrap_ci(y_dev, p_dev_auto)
auto_ext = bootstrap_ci(y_ext, p_ext_auto)

joblib.dump(autotabpfn, os.path.join(RESULTS_DIR, "autotabpfn_fitted_on_dev.pkl"))


# -----------------------------
# 4) Optional: Platt on external
# -----------------------------
p_ext_auto_platt, platt_params = fit_platt_scaler(y_ext, p_ext_auto)
auto_ext_platt = bootstrap_ci(y_ext, p_ext_auto_platt)

joblib.dump(platt_params, os.path.join(RESULTS_DIR, "platt_params_external.pkl"))


# -----------------------------
# 5) Summarize to CSV
# -----------------------------
def fmt(res, key, d=3):
    v, (lo, hi) = res[key]
    return f"{v:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"

summary = pd.DataFrame([
    {"Model/Cohort": "TabPFN / Dev (OOF)",
     "AUROC": fmt(tabpfn_dev, "AUROC", 3),
     "AUPRC": fmt(tabpfn_dev, "AUPRC", 3),
     "Brier": fmt(tabpfn_dev, "Brier", 3)},
    {"Model/Cohort": "TabPFN / Ext",
     "AUROC": fmt(tabpfn_ext, "AUROC", 3),
     "AUPRC": fmt(tabpfn_ext, "AUPRC", 3),
     "Brier": fmt(tabpfn_ext, "Brier", 3)},
    {"Model/Cohort": "AutoTabPFN / Dev (OOF)",
     "AUROC": fmt(auto_dev, "AUROC", 3),
     "AUPRC": fmt(auto_dev, "AUPRC", 3),
     "Brier": fmt(auto_dev, "Brier", 3)},
    {"Model/Cohort": "AutoTabPFN / Ext",
     "AUROC": fmt(auto_ext, "AUROC", 3),
     "AUPRC": fmt(auto_ext, "AUPRC", 3),
     "Brier": fmt(auto_ext, "Brier", 3)},
    {"Model/Cohort": "AutoTabPFN+Platt / Ext",
     "AUROC": fmt(auto_ext_platt, "AUROC", 3),
     "AUPRC": fmt(auto_ext_platt, "AUPRC", 3),
     "Brier": fmt(auto_ext_platt, "Brier", 3)},
])

out_csv = os.path.join(RESULTS_DIR, "metrics_summary.csv")
summary.to_csv(out_csv, index=False)

print(summary.to_string(index=False))
print(f"\nWrote: {out_csv}")
