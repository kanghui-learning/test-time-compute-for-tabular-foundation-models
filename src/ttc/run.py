"""Run one TabArena experiment from a YAML config.

    python -m ttc.run configs/tabpfn3/diagscale.yaml

Config schema (all keys optional except ``model``):
    model:  tabpfn3-ft           # registry key (python -m ttc.registry --list)
    mode:   full                 # full (51 datasets, all folds/repeats) | lite | debug
    tag_suffix: diagscale        # REQUIRED when overriding hyperparameters: keeps each
                                 # variant's fit cache separate
    hyperparameters: {}          # model hyperparameter overrides (e.g. ft_lr: 3.0e-2)
    datasets: [name, ...]        # optional: restrict to these TabArena datasets (sharding)
    repeats:  [0]                # optional: restrict TabArena repeats
    folds:    [0, 1, 2]          # optional: restrict TabArena folds
    fresh:    false              # ignore caches and write to a separate <tag>_rerun directory
    raise_on_failure: true       # false = record a failing cell (e.g. CUDA OOM) and continue

Outputs go to $TTC_RUNS_DIR (default ./runs): experiments/<tag>/ (per-cell fit cache) and
eval/<tag>/ (results_per_split.csv, leaderboard.csv, fit logs).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one TabArena experiment from a YAML config.")
    parser.add_argument("config", help="path to a YAML experiment config")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    mode = cfg.get("mode", "full")
    if mode not in ("full", "lite", "debug"):
        raise SystemExit(f"invalid mode '{mode}' (full | lite | debug)")
    if cfg.get("hyperparameters") and not cfg.get("tag_suffix"):
        raise SystemExit("config overrides hyperparameters but sets no tag_suffix; "
                         "variants would collide in the fit cache")

    from ttc.benchmarks.tabarena import run

    run(
        cfg["model"],
        lite=mode == "lite",
        debug=mode == "debug",
        fresh=bool(cfg.get("fresh", False)),
        hyperparameters=cfg.get("hyperparameters"),
        tag_suffix=cfg.get("tag_suffix"),
        repeats=cfg.get("repeats"),
        folds=cfg.get("folds"),
        datasets=cfg.get("datasets"),
        raise_on_failure=bool(cfg.get("raise_on_failure", True)),
    )


if __name__ == "__main__":
    main()
