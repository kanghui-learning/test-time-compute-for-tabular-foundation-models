#!/usr/bin/env python
"""Build the Hugging Face dataset folder with the paper's per-cell TabArena results.

Layout written to ``--out`` (nothing is uploaded)::

    README.md                         dataset card
    data/results.parquet              all 12 arms in one table (12 x 816 rows)
    data/arms.parquet                 one row per arm: config, Elo, Delta Elo, GPU hours, sha256, ...
    data/arms.csv                     the same table as CSV, for reading without a parquet reader
    raw/<arm>/results_per_split.csv   the arm's own TabArena-format rows, byte-identical lines

For every paper arm only our own rows (the non-reference ``Demo_*`` method) of the run's
``results_per_split.csv`` are copied, line by line, so the Elo fit on ``raw/`` stays bit-identical.
The frozen TabFM arm (``raw/tabfm_frozen``) is also the 68th entry of the fixed reference field.
The 67 official TabArena reference entries are NOT redistributed: ``ttc.eval.elo`` fetches them
from TabArena. Both dataset configs are parquet because ``datasets.load_dataset`` infers one builder
for all configs of a repository (a csv config next to a parquet one fails to load).

    python scripts/export_hf_results.py --runs-dir $TTC_RUNS_DIR/eval \
        --paper-arms docs/paper/artifacts/elo68_paper/paper_arms.csv \
        --points docs/paper/artifacts/headline_cost/points.csv --out hf_export/

Arms are taken from ``points.csv`` (backbone, method, run_tag, Elo, GPU hours; 12 rows) joined with
the normalized score of ``paper_arms.csv``; ``--tags`` restricts the export to a subset of run tags
(the frozen TabFM reference always ships).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "ttc" / "eval" / "elo.py").exists():
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from ttc.eval.elo import CANDIDATE_PREFIX, FROZEN_TABFM_TAG, N_CELLS, REFERENCE_METHODS  # noqa: E402

GITHUB = "https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models"
HF_REPO = "nkh/test-time-compute-for-tabular-foundation-models-results"

BACKBONES = {"TabPFN-3": "tabpfn3", "TabICL v2": "tabiclv2", "TabFM": "tabfm"}
# points.csv method label -> (release config name, method label in the dataset, learning rate)
METHODS = {
    "Frozen": ("frozen", "frozen"),
    "32 views": ("native_views_32", "native views (32)"),
    "Full FT": ("full_ft", "full fine-tuning"),
    "DiagScale": ("diagscale", "DiagScale"),
    "96-config. aggregation": ("aggregation_96", "aggregation (96 configurations)"),
    "DiagScale + aggregation": ("diagscale_aggregation_96", "DiagScale + aggregation"),
}
# Learning rate of each adaptation arm (= ft_lr of configs/<backbone>/<method>.yaml).
LEARNING_RATES = {
    "tabpfn3_full_ft": 1e-6, "tabpfn3_diagscale": 3e-2, "tabpfn3_diagscale_aggregation_96": 1e-2,
    "tabiclv2_full_ft": 1e-5, "tabiclv2_diagscale": 3e-2,
    "tabfm_full_ft": 1e-4, "tabfm_diagscale": 1e-2,
}
ARM_ORDER = [
    "tabpfn3_frozen", "tabpfn3_native_views_32", "tabpfn3_full_ft", "tabpfn3_diagscale",
    "tabpfn3_aggregation_96", "tabpfn3_diagscale_aggregation_96",
    "tabiclv2_frozen", "tabiclv2_full_ft", "tabiclv2_diagscale",
    "tabfm_frozen", "tabfm_full_ft", "tabfm_diagscale",
]
# Textual TabArena columns that are empty for our rows; kept as strings (null), not float NaN.
STRING_COLS = ["dataset", "tabarena_method", "metric", "problem_type", "impute_method", "method_type",
               "method_subtype", "config_type", "ta_name", "ta_suite"]
LEADING_COLS = ["dataset", "fold", "tabarena_method", "metric", "metric_error", "metric_error_val",
                "time_train_s", "time_infer_s"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arms(paper_arms: Path, points: Path) -> pd.DataFrame:
    pts = pd.read_csv(points)
    scores = pd.read_csv(paper_arms).set_index(["backbone", "setting"])["score"]
    rows = []
    for p in pts.itertuples():
        cfg_method, method = METHODS[p.method]
        arm = f"{BACKBONES[p.backbone]}_{cfg_method}"
        rows.append(dict(
            arm=arm, backbone=p.backbone, method=method, learning_rate=LEARNING_RATES.get(arm),
            config=f"configs/{BACKBONES[p.backbone]}/{cfg_method}.yaml",
            elo=p.elo_raw, delta_elo=p.delta_elo_raw, delta_elo_rounded=int(p.delta_elo),
            baseline=p.baseline, normalized_score=scores[(p.backbone, p.method)],
            gpu_hours=p.total_h, source_run=p.run_tag))
    arms = pd.DataFrame(rows)
    assert set(arms.arm) == set(ARM_ORDER), sorted(set(arms.arm) ^ set(ARM_ORDER))
    return arms.set_index("arm").loc[ARM_ORDER].reset_index()


def copy_candidate_rows(src: Path, dst: Path) -> tuple[str, int, int]:
    """Copy header + non-reference rows verbatim; returns (method, n_cells, n_datasets)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    methods, cells = set(), set()
    with open(src, newline="") as f_in, open(dst, "w", newline="") as f_out:
        header = f_in.readline()
        f_out.write(header)
        cols = next(csv.reader([header]))
        i_m, i_d, i_f = cols.index("method"), cols.index("dataset"), cols.index("fold")
        for line in f_in:
            row = next(csv.reader([line]))
            if row[i_m] in REFERENCE_METHODS:
                continue
            methods.add(row[i_m])
            cells.add((row[i_d], row[i_f]))
            f_out.write(line)
    if len(methods) != 1 or not next(iter(methods)).startswith(CANDIDATE_PREFIX):
        raise ValueError(f"{src}: expected one {CANDIDATE_PREFIX}* method, got {sorted(methods)}")
    if len(cells) != N_CELLS:
        raise ValueError(f"{src}: expected {N_CELLS} cells, got {len(cells)}")
    return methods.pop(), len(cells), len({d for d, _ in cells})


def read_raw(path: Path) -> pd.DataFrame:
    """Read a raw results_per_split.csv with exact float parsing and the dataset's column names."""
    df = pd.read_csv(path, float_precision="round_trip").rename(columns={"method": "tabarena_method"})
    for col in df.columns:
        if col in STRING_COLS:
            df[col] = df[col].astype("string")
    df["fold"] = df["fold"].astype("int64")
    return df[LEADING_COLS + [c for c in df.columns if c not in LEADING_COLS]]


def results_table(arms: pd.DataFrame, out: Path) -> pd.DataFrame:
    parts = []
    for a in arms.itertuples():
        df = read_raw(out / a.raw_file)
        df.insert(0, "learning_rate", a.learning_rate)
        df.insert(0, "method", a.method)
        df.insert(0, "backbone", a.backbone)
        df.insert(0, "arm", a.arm)
        parts.append(df)
    res = pd.concat(parts, ignore_index=True)
    for col in ("arm", "backbone", "method"):
        res[col] = res[col].astype("string")
    res["learning_rate"] = res["learning_rate"].astype("float64")
    return res.sort_values(["arm", "dataset", "fold"], kind="stable").reset_index(drop=True)


def fmt_lr(lr) -> str:
    return "" if pd.isna(lr) else f"{lr:.0e}".replace("e-0", "e-")


CARD = """---
license: cc-by-4.0
pretty_name: Test-Time Compute for Tabular Foundation Models - TabArena Results
task_categories:
- tabular-classification
- tabular-regression
tags:
- tabular
- benchmark
- tabarena
- test-time-compute
- tabular-foundation-models
- tabpfn
- tabicl
size_categories:
- 1K<n<10K
configs:
- config_name: results
  data_files: data/results.parquet
  default: true
- config_name: arms
  data_files: data/arms.parquet
---

# Test-Time Compute for Tabular Foundation Models: TabArena Results

Per-cell TabArena test errors for the 12 methods ("arms") in the main comparison of
*Test-Time Compute for Tabular Foundation Models: Mechanisms, Gains, and Limits*.
Every arm covers all 51 TabArena datasets (816 cells), so these files are enough to recompute
the paper's Elo ratings and paired comparisons.

**Code:** [GitHub]({github}) · **Paper:** [arXiv:2610.12005](https://arxiv.org/abs/2610.12005)

## Files

| file | content |
|---|---|
| `data/results.parquet` | config `results`: one row per arm × cell ({n_rows:,} rows) |
| `data/arms.parquet`, `data/arms.csv` | config `arms`: one row per arm (config, Elo, ΔElo, GPU hours, file checksum) |
| `raw/<arm>/results_per_split.csv` | each arm in TabArena's format, the input to `ttc.eval.elo` |

## Arms

ΔElo is measured against TabArena's default entry for TabPFN-3 and TabICL v2, and against our
frozen run for TabFM. GPU hours are fit + predict time over 816 cells on one H100.

{arms_table}

## Columns of `results`

| column | meaning |
|---|---|
| `arm`, `backbone`, `method`, `learning_rate` | arm identity; `learning_rate` is null without fine-tuning |
| `dataset`, `fold` | TabArena dataset and split (`fold` enumerates repeat × fold) |
| `metric`, `metric_error` | `roc_auc` (binary, error = 1 − AUC), `log_loss` (multiclass) or `rmse` (regression); lower is better |
| `time_train_s`, `time_infer_s` | wall-clock seconds per cell |
| `problem_type` | `binary`, `multiclass` or `regression` |
| other columns | TabArena bookkeeping (`tabarena_method`, `imputed`, ...); `metric_error_val` is not recorded (null) |

## Usage

```python
from datasets import load_dataset

repo = "{hf_repo}"
results = load_dataset(repo, "results", split="train")
arms = load_dataset(repo, "arms", split="train")
```

Recompute an Elo rating with the code repository:

```bash
hf download {hf_repo} \\
    --repo-type dataset --local-dir hf_results
python -m ttc.eval.elo \\
    --candidate hf_results/raw/tabpfn3_diagscale/results_per_split.csv \\
    --reference-field hf_results/raw/tabfm_frozen/results_per_split.csv \\
    --baseline "TA-TABPFN-3 (default)"            # ΔElo +26
```

## Notes

- **Elo.** Fixed field of 68 entries: the 67 official TabArena entries (fetched by TabArena, not
  included here) plus our frozen TabFM run. Each arm is added alone and rated with TabArena's
  Bradley–Terry solver, `RF (default)` = 1000.
- GPU fine-tuning is not bitwise deterministic; re-running an adaptation config gives slightly
  different cell errors.

## License

The results are released under CC-BY-4.0. TabArena data and reference results, and the TabPFN-3,
TabICL v2 and TabFM weights (none included), are under their own licenses.

## Citation

```bibtex
@article{{ning2026testtime,
  title   = {{Test-Time Compute for Tabular Foundation Models: Mechanisms, Gains, and Limits}},
  author  = {{Ning, Kanghui and Bilo{{\\v{{s}}}}, Marin and Wilson, James T. and Zhang, Yilang and Rasul, Kashif
             and Song, Dongjin and Schneider, Anderson and Nevmyvaka, Yuriy}},
  journal = {{arXiv preprint arXiv:2610.12005}},
  year    = {{2026}}
}}
```
"""


def arms_markdown(arms: pd.DataFrame) -> str:
    lines = ["| backbone | arm | LR | Elo | ΔElo | GPU h |", "|---|---|---:|---:|---:|---:|"]
    prev = None
    for a in arms.itertuples():
        backbone = "" if a.backbone == prev else a.backbone
        prev = a.backbone
        delta = f"{a.delta_elo_rounded:+d}" if a.delta_elo_rounded else "0"
        lines.append(f"| {backbone} | [`{a.arm}`]({GITHUB}/blob/main/{a.config}) | {fmt_lr(a.learning_rate)} "
                     f"| {a.elo:.0f} | {delta} | {a.gpu_hours:.1f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    runs_default = os.environ.get("TTC_RUNS_DIR")
    ap.add_argument("--runs-dir", type=Path, default=Path(runs_default) / "eval" if runs_default else None,
                    help="folder of <tag>/results_per_split.csv (default $TTC_RUNS_DIR/eval)")
    ap.add_argument("--paper-arms", type=Path, required=True, help="artifacts/elo68_paper/paper_arms.csv")
    ap.add_argument("--points", type=Path, required=True, help="artifacts/headline_cost/points.csv")
    ap.add_argument("--tags", nargs="*", help="export only these run tags (default: all 12 arms)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.runs_dir is None:
        ap.error("--runs-dir (or $TTC_RUNS_DIR) is required")

    arms = load_arms(args.paper_arms, args.points)
    if args.tags:
        keep = arms.source_run.isin(args.tags) | arms.source_run.map(lambda t: Path(t).name).isin(args.tags)
        arms = arms[keep | (arms.source_run == FROZEN_TABFM_TAG)].reset_index(drop=True)

    extra = []
    for a in arms.itertuples():
        src = args.runs_dir / a.source_run / "results_per_split.csv"
        raw_file = Path("raw") / a.arm / "results_per_split.csv"
        dst = args.out / raw_file
        _, n_cells, n_ds = copy_candidate_rows(src, dst)
        extra.append(dict(n_cells=n_cells, n_datasets=n_ds, raw_file=str(raw_file), sha256=sha256(dst)))
        print(f"{a.arm:36s} {a.source_run:60s} {dst.stat().st_size / 1e3:8.1f} kB")
    arms = pd.concat([arms, pd.DataFrame(extra)], axis=1)
    arms = arms[["arm", "backbone", "method", "learning_rate", "config", "elo", "delta_elo",
                 "delta_elo_rounded", "baseline", "normalized_score", "gpu_hours", "n_cells",
                 "n_datasets", "raw_file", "sha256", "source_run"]]

    (args.out / "data").mkdir(parents=True, exist_ok=True)
    res = results_table(arms, args.out)
    res.to_parquet(args.out / "data" / "results.parquet", index=False)
    arms.to_parquet(args.out / "data" / "arms.parquet", index=False)
    arms.to_csv(args.out / "data" / "arms.csv", index=False)
    card = CARD.format(github=GITHUB, hf_repo=HF_REPO, n_rows=len(res), arms_table=arms_markdown(arms))
    (args.out / "README.md").write_text(card)
    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"{len(arms)} arms, {len(res)} rows -> {args.out}  total {total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
