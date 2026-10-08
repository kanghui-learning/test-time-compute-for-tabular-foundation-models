"""Build the large-table source pools for the row-retrieval study (CPU only).

Deterministic conversion of a complete OpenML source -> ``{name}_full_{X,y}.npy`` + meta.json in
the data directory ($TTC_DATA_DIR, default ``./data/row_large``):
drop constant columns; object/category/bool -> pandas Categorical codes (float32, recorded in
``cat_idx``); numeric -> float32 (non-finite -> NaN); y -> Categorical codes. No row sampling, so
the full pool has no randomness at all.

``--subset-rows 3000000`` additionally writes the 3M-row source subset ``{name}_3M_{X,y}.npy``
used by the paper's within-memory sweep (<= 2.9M rows) on Higgs / US_Accidents / COMET_MC:
rows ``full[np.random.default_rng(0).choice(N_full, 3_000_000, replace=False)]`` in draw order.

Per-seed test split: no file is written. ``ttc.benchmarks.row_coreset_large._split`` draws it at
run time from the loaded pool -- a ``default_rng(seed)`` permutation whose first 50,000 rows are
the test set and whose remainder (in permuted order) is the source pool; a pool of size N is the
first N rows of that remainder. Each pool family (full / 3M subset / OpenML) therefore has its own
seed-specific test set, as in the paper.

Covertype (OpenML 1596) and Poker-Hand (OpenML 1567) need no preparation: they are loaded from
OpenML on the fly (``row_coreset_large.load_large``). The OpenML cache location follows openml's own
``OPENML_CACHE_DIR`` environment variable when set.

    python -m ttc.benchmarks.row_pool_prep                                  # all three
    python -m ttc.benchmarks.row_pool_prep --datasets Higgs
    python -m ttc.benchmarks.row_pool_prep --datasets Higgs COMET_MC US_Accidents --subset-rows 3000000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

DIDS = {"Higgs": 45570, "COMET_MC": 5889, "US_Accidents": 46650}
EXPECT_FEAT = {"Higgs": 28, "COMET_MC": 4, "US_Accidents": 43}
# full source row counts -- the sweep's top rungs are pool==N cells, so a silent row-count drift
# would make the runner skip them (N > len(pool)) without any error; fail loudly here instead
EXPECT_ROWS = {"Higgs": 11_000_000, "COMET_MC": 7_619_400, "US_Accidents": 7_728_394}
SUBSET_SEED = 0


def data_dir() -> Path:
    """Directory holding the prepared npy pools ($TTC_DATA_DIR, default ./data/row_large)."""
    return Path(os.environ.get("TTC_DATA_DIR", "data/row_large"))


def build(name: str, out: Path) -> None:
    import openml
    t0 = time.time()
    print(f"==== {name} (did={DIDS[name]}) ====", flush=True)
    d = openml.datasets.get_dataset(DIDS[name], download_data=True, download_qualities=False,
                                    download_features_meta_data=True)
    X, y, _, _ = d.get_data(target=d.default_target_attribute, dataset_format="dataframe")
    X = X.loc[:, X.nunique(dropna=True) > 1]                     # constant-column drop
    cat_idx = []
    for j, c in enumerate(X.columns):
        if str(X[c].dtype) in ("object", "category", "bool"):
            X[c] = pd.Categorical(X[c].astype("string")).codes.astype(np.float32)
            cat_idx.append(j)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    Xn = X.to_numpy(np.float32)
    Xn = np.where(np.isfinite(Xn), Xn, np.nan)
    yv = pd.Categorical(y.astype("string")).codes.astype(np.int16)
    if Xn.shape[1] != EXPECT_FEAT[name]:
        raise SystemExit(f"{name}: {Xn.shape[1]} features != expected {EXPECT_FEAT[name]} "
                         f"(OpenML schema drifted -- investigate before saving)")
    if Xn.shape[0] != EXPECT_ROWS[name]:
        raise SystemExit(f"{name}: {Xn.shape[0]:,} rows != expected {EXPECT_ROWS[name]:,} "
                         f"(source drifted -- the sweep's exact top rungs would be silently skipped)")
    if (yv < 0).any():
        raise SystemExit(f"{name}: {(yv < 0).sum()} missing target values (Categorical code -1) -- "
                         f"a phantom class would silently corrupt classification")
    np.save(out / f"{name}_full_X.npy", Xn)
    np.save(out / f"{name}_full_y.npy", yv)
    json.dump({"cat_idx": cat_idx, "n_classes": int(len(pd.unique(yv)))},
              open(out / f"{name}_full_meta.json", "w"))
    print(f"  saved {name}_full X={Xn.shape} n_cat={len(cat_idx)} "
          f"classes={len(pd.unique(yv))} ({Xn.nbytes / 1e9:.2f}GB) {time.time() - t0:.0f}s", flush=True)


def build_subset(name: str, out: Path, n_rows: int, seed: int = SUBSET_SEED) -> None:
    """``{name}_{n}M`` subset of the full pool: rows full[default_rng(seed).choice(N, n_rows)], in
    draw order (no sorting). Verified bit-identical (X and y values) to the paper's 3M subsets of
    Higgs and COMET_MC; see docs/retrieval.md for US_Accidents."""
    X = np.load(out / f"{name}_full_X.npy", mmap_mode="r")
    y = np.load(out / f"{name}_full_y.npy")
    idx = np.random.default_rng(seed).choice(len(X), n_rows, replace=False)
    order = np.argsort(idx)                                    # sequential mmap read, then unsort
    Xs = np.empty((n_rows, X.shape[1]), np.float32)
    Xs[order] = X[idx[order]]
    tag = f"{name}_{n_rows // 1_000_000}M" if n_rows % 1_000_000 == 0 else f"{name}_{n_rows}"
    np.save(out / f"{tag}_X.npy", Xs)
    np.save(out / f"{tag}_y.npy", y[idx])
    meta = json.load(open(out / f"{name}_full_meta.json"))
    json.dump(meta, open(out / f"{tag}_meta.json", "w"))
    print(f"  saved {tag} X={Xs.shape} (seed={seed})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(DIDS), choices=list(DIDS))
    ap.add_argument("--subset-rows", type=int, default=0,
                    help="also write a seeded random subset of this many rows (paper: 3000000)")
    ap.add_argument("--skip-full", action="store_true",
                    help="reuse existing {name}_full_*.npy (only build the subset)")
    a = ap.parse_args()
    out = data_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name in a.datasets:
        if not a.skip_full:
            build(name, out)
        if a.subset_rows:
            build_subset(name, out, a.subset_rows)
    print(f"ALL DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
