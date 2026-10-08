"""Shared data/model helpers for frozen TabPFN-3 context-construction experiments.

  * ``load_dataset``  one TabArena dataset (OpenML task) -> numeric matrix + categorical indices
  * ``_model``        a frozen TabPFN-3 classifier/regressor (standard tabpfn checkpoint resolution:
                      downloaded/cached by tabpfn, ``TABPFN_MODEL_CACHE_DIR`` honoured when set)
  * ``_align``        predict_proba re-indexed to a global class list (missing classes -> 0)
  * ``_err``          log loss (classification) / RMSE (regression)
  * ``_geo``          train-mean imputation + train-fit standardization (geometry space)
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SEED = 0


def _runs_dir() -> Path:
    return Path(os.environ.get("TTC_RUNS_DIR", "runs"))


def load_dataset(md, ds: str):
    """OpenML task for one TabArena dataset -> (Xn[float, cats=int codes], y, cat_idx, ptype, task).

    categorical = OpenML-flagged OR any non-numeric dtype (e.g. bank-marketing ships object columns the
    flags miss); ordinal codes are fit on the full X (an arbitrary, label-free relabeling -- no leakage).
    """
    import openml
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder

    tid = int(md.loc[ds, "tid"])
    ptype = md.loc[ds, "problem_type"]
    t = openml.tasks.get_task(tid, download_splits=True, download_data=True)
    X, y, cat_ind, _ = t.get_dataset().get_data(target=t.target_name, dataset_format="dataframe")
    is_cat = [bool(cat_ind[i]) or X.iloc[:, i].dtype == object or str(X.iloc[:, i].dtype) == "category"
              for i in range(X.shape[1])]
    cat_idx = [i for i, c in enumerate(is_cat) if c]
    Xn = X.copy()
    if cat_idx:
        cols = X.columns[cat_idx]
        Xn[cols] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(
            X[cols].astype(str))
    Xn = Xn.apply(lambda s: pd.to_numeric(s, errors="coerce")).to_numpy(dtype=np.float64)
    y = y.to_numpy()
    if ptype == "regression":
        y = y.astype(np.float64)
    return Xn, y, cat_idx, ptype, t


def _geo(Xtr, Xte):
    """Train-mean impute + train-fit StandardScaler -> geometry space for clustering + distances."""
    from sklearn.preprocessing import StandardScaler

    mu = np.nanmean(Xtr, axis=0)
    Xtr2, Xte2 = np.where(np.isnan(Xtr), mu, Xtr), np.where(np.isnan(Xte), mu, Xte)
    sc = StandardScaler().fit(Xtr2)
    return sc.transform(Xtr2), sc.transform(Xte2)


def _model(ptype, cat_idx, n_estimators, device):
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    kw = dict(n_estimators=n_estimators, device=device, ignore_pretraining_limits=True,
              random_state=SEED, categorical_features_indices=cat_idx or None)
    return TabPFNRegressor(**kw) if ptype == "regression" else TabPFNClassifier(**kw)


def _align(m, Xte, classes):
    """predict_proba re-indexed to the global (train) class list; classes missing from a subset -> 0."""
    p = m.predict_proba(Xte)
    out = np.zeros((len(Xte), len(classes)))
    idx = {c: j for j, c in enumerate(classes)}
    for k, c in enumerate(m.classes_):
        out[:, idx[c]] = p[:, k]
    return out


def _err(ptype, y_true, pred, classes):
    from sklearn.metrics import log_loss, mean_squared_error

    if ptype == "regression":
        return float(np.sqrt(mean_squared_error(y_true, pred)))
    return float(log_loss(y_true, pred, labels=classes))
