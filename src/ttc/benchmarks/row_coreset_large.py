"""Large-table data layer + feed-all arm for the row-retrieval study (frozen TabPFN-3).

  * ``load_large``  dataset name -> (X float32, y, cat_idx, ptype). Prepared npy pools from
                    ``$TTC_DATA_DIR`` (see ``row_pool_prep``) or OpenML (Covertype, Poker).
  * ``_split``      the per-seed split: fixed 50,000-row held-out test set + shuffled source pool.
  * ``_eval_full``  the feed-all arm (whole pool as context; OOM is caught and recorded).
  * ``_scores``     log loss / AUC (binary) / accuracy.

The model-aware arm lives in ``ttc.benchmarks.row_modelaware``.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from ttc.benchmarks.hardgather import _align, _err, _model  # noqa: E402
from ttc.benchmarks.row_pool_prep import data_dir           # noqa: E402

try:  # the tabpfn fork wraps CUDA OOM in its own hierarchy (TabPFNError <- Exception, NOT RuntimeError)
    from tabpfn.errors import TabPFNOutOfMemoryError as _TabPFNOOM
except ImportError:                                                # pragma: no cover - older forks
    class _TabPFNOOM(Exception):
        pass

OPENML_DIDS = {"Covertype": 1596, "Poker": 1567}
TEST_N = 50_000


def _runs_dir() -> Path:
    return Path(os.environ.get("TTC_RUNS_DIR", "runs"))


def load_large(name):
    """name -> (X float32, y, cat_idx, ptype).

    OpenML-backed: Covertype, Poker. Anything else is a prepared pool in the data directory:
    ``{name}_X.npy`` (memory-mapped), ``{name}_y.npy`` and ``{name}_meta.json``, e.g.
    ``Higgs_full`` / ``US_Accidents_full`` / ``COMET_MC_full`` or the 3M subsets ``Higgs_3M`` etc.
    """
    if name in OPENML_DIDS:
        from ttc.benchmarks.overflow_data import load_openml
        X, y, cat = load_openml(OPENML_DIDS[name])
        return (np.asarray(X, np.float32), np.asarray(y), cat,
                ("binary" if len(np.unique(y)) == 2 else "multiclass"))
    d = data_dir()
    xp, yp, mp = d / f"{name}_X.npy", d / f"{name}_y.npy", d / f"{name}_meta.json"
    if not (xp.exists() and yp.exists() and mp.exists()):
        raise FileNotFoundError(f"pool {name!r} not found in {d} (expected {xp.name}, {yp.name}, "
                                f"{mp.name}); build it with python -m ttc.benchmarks.row_pool_prep")
    X = np.load(xp, mmap_mode="r")
    y = np.load(yp)
    m = json.load(open(mp))
    return X, y, list(m.get("cat_idx", [])), ("binary" if m["n_classes"] == 2 else "multiclass")


def _split(X, y, seed):
    """Fixed held-out test (TEST_N) + shuffled pool from the rest."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    te, pool = perm[:TEST_N], perm[TEST_N:]
    return np.asarray(X[pool]), np.asarray(y[pool]), np.asarray(X[te]), np.asarray(y[te])


def _k_clusters(n_test):
    """Query-cluster count G = max(4, min(30, n_test // 2000)) (callers also cap at n_test)."""
    return int(max(4, min(30, n_test // 2000)))


def _peak_gb(torch):
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0


def _eval_full(Xtr, ytr, Xte, yte, cat, ptype, classes, n_est, device, torch):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    t0 = time.time()
    try:
        m = _model(ptype, cat, n_est, device).fit(Xtr, ytr)
        pred = m.predict(Xte) if ptype == "regression" else _align(m, Xte, classes)
        return dict(status="ok", **_scores(ptype, yte, pred, classes),
                    secs=round(time.time() - t0, 1), peak_gb=_peak_gb(torch))
    except (RuntimeError, MemoryError, _TabPFNOOM) as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        oom = "out of memory" in str(e).lower() or isinstance(e, (MemoryError, _TabPFNOOM))
        return dict(status="oom" if oom else "err", err=np.nan, auc=np.nan, acc=np.nan,
                    secs=round(time.time() - t0, 1), peak_gb=_peak_gb(torch))


def _scores(ptype, yte, pred, classes):
    from sklearn.metrics import accuracy_score, roc_auc_score
    if ptype == "regression":
        return dict(err=_err(ptype, yte, pred, classes), auc=np.nan, acc=np.nan)
    err = _err(ptype, yte, pred, classes)
    acc = float(accuracy_score(yte, [classes[i] for i in pred.argmax(1)]))
    auc = np.nan
    if len(classes) == 2:
        auc = float(roc_auc_score((yte == classes[1]).astype(int), pred[:, 1]))
    return dict(err=round(err, 5), auc=round(auc, 5) if auc == auc else np.nan, acc=round(acc, 5))
