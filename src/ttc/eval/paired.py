"""Paired error change between two methods on the same TabArena cells.

For every cell (dataset, fold) take the error e of the candidate and of the baseline
(e = 1 - AUC for binary, log loss for multiclass, RMSE for regression; TabArena's
``metric_error``). Per dataset, l_d = mean over its cells of log(e_cand / e_base). The reported
change is 100 * (exp(mean_d l_d) - 1) percent (datasets equally weighted; negative = candidate
better). The 95% interval is a percentile bootstrap over datasets (10,000 resamples, numpy
``default_rng(0)``); wins = number of datasets with l_d < 0.

CLI
---
    python -m ttc.eval.paired --candidate A/results_per_split.csv --baseline B/results_per_split.csv
    # baseline = a reference row of the same CSV:
    python -m ttc.eval.paired --candidate A/results_per_split.csv --baseline-method "TA-TABPFN-3 (default)"
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ttc.eval.elo import N_CELLS, candidate_methods, read_results

N_BOOT = 10_000
SEED = 0


def cell_errors(results: str | Path | pd.DataFrame, method: str | None = None) -> pd.Series:
    """``metric_error`` per (dataset, fold) of one method (default: the single candidate)."""
    data = results if isinstance(results, pd.DataFrame) else read_results(results)
    if method is None:
        own = candidate_methods(data)
        if len(own) != 1:
            raise ValueError(f"expected exactly one non-reference method, got {own}; pass method=")
        method = own[0]
    d = data[data["method"] == method].set_index(["dataset", "fold"]).sort_index()
    if len(d) != N_CELLS or d.index.duplicated().any():
        raise ValueError(f"{method!r}: expected {N_CELLS} unique cells, got {len(d)}")
    return d["metric_error"]


@dataclass
class PairedResult:
    estimate: float  # percent
    lo: float
    hi: float
    wins: int
    datasets: int
    cells: int


def paired_change(candidate: pd.Series, baseline: pd.Series, n_boot: int = N_BOOT,
                  seed: int = SEED) -> tuple[PairedResult, pd.Series]:
    """Paired error change of ``candidate`` vs ``baseline`` (cell-indexed error Series).

    Returns the summary and the per-dataset mean log ratio l_d.
    """
    if not candidate.index.sort_values().equals(baseline.index.sort_values()):
        raise ValueError("candidate and baseline cover different cells")
    log_ratio = np.log(candidate / baseline.reindex(candidate.index))
    per_dataset = log_ratio.groupby(level="dataset").mean()
    v = per_dataset.to_numpy()
    n = len(v)
    boot = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    est, lo, hi = 100 * np.expm1(np.r_[v.mean(), np.quantile(v[boot].mean(1), [.025, .975])])
    return PairedResult(float(est), float(lo), float(hi), int((v < 0).sum()), n, len(log_ratio)), per_dataset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ttc.eval.paired", description=__doc__.split("\n\n")[0])
    p.add_argument("--candidate", required=True, metavar="CSV")
    p.add_argument("--candidate-method", help="default: the single non-reference method of --candidate")
    p.add_argument("--baseline", metavar="CSV", help="default: the --candidate CSV")
    p.add_argument("--baseline-method", help="default: the single non-reference method of --baseline")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--per-dataset", metavar="CSV", help="write per-dataset mean log ratios here")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.baseline is None and args.baseline_method is None:
        p.error("give --baseline and/or --baseline-method")
    cand_data = read_results(args.candidate)
    base_data = cand_data if args.baseline is None else read_results(args.baseline)
    res, per_ds = paired_change(cell_errors(cand_data, args.candidate_method),
                                cell_errors(base_data, args.baseline_method), args.n_boot, args.seed)
    if args.per_dataset:
        per_ds.rename("mean_log_ratio").to_csv(args.per_dataset)
    if args.json:
        print(json.dumps(asdict(res)))
    else:
        print(f"error change {res.estimate:+.3f}%  95% CI [{res.lo:+.3f}%, {res.hi:+.3f}%]  "
              f"wins {res.wins}/{res.datasets}  ({res.cells} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
