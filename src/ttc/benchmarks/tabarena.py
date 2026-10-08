"""TabArena benchmark runner: evaluate one registered model on the TabArena protocol.

    python -m ttc.benchmarks.tabarena <model_key> [--debug | --lite] [--fresh]
    python -m ttc.benchmarks.tabarena --list

Configs normally go through ``python -m ttc.run <config.yaml>``, which calls :func:`run`.

Outputs live under $TTC_RUNS_DIR (default ./runs):
    experiments/<tag>/   per-cell fit cache (re-running a finished cell reads the cache)
    eval/<tag>/          results_per_split.csv, leaderboard.csv and per-fit logs
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ttc.registry import MODELS, resolve_model_class

# --debug: one binary classification and one regression dataset, for a quick end-to-end check.
DEBUG_DATASETS = ["diabetes", "airfoil_self_noise"]

DEFAULT_RUNS_DIR = "runs"


def _runs_dir() -> Path:
    """Root for all run outputs (fit caches can be large; point TTC_RUNS_DIR at a data disk)."""
    return Path(os.environ.get("TTC_RUNS_DIR", DEFAULT_RUNS_DIR))


def _classification_only(model_cls) -> bool:
    """True if the model declares it cannot handle regression (from ``supported_problem_types``).

    Such a model runs the 38 classification tasks only; the 13 regression tasks are imputed with
    the RandomForest (default) baseline at compare time, as on the official leaderboard.
    """
    try:
        spt = model_cls.supported_problem_types()
    except Exception:
        return False
    return spt is not None and "regression" not in spt


# Named TabArena size predicates keyed on a model's hard ``ag.max_rows`` cap, which AutoGluon
# asserts rather than subsamples. Datasets above the cap are imputed with RF (default).
_MAX_ROWS_TO_PREDICATE = {2_000: "tiny", 10_000: "small"}


def _row_cap_predicate(model_cls) -> str | None:
    try:
        max_rows = model_cls()._get_default_auxiliary_params().get("max_rows")
    except Exception:
        return None
    if max_rows is None:
        return None
    pred = _MAX_ROWS_TO_PREDICATE.get(max_rows)
    if pred is None:
        print(f"[ttc.benchmarks.tabarena] WARNING: max_rows={max_rows} has no matching size "
              f"predicate; not filtering by rows (the run may hit ag.max_rows assertions).")
    return pred


def run(
    model_key: str,
    *,
    lite: bool = False,
    debug: bool = False,
    fresh: bool = False,
    hyperparameters: dict | None = None,
    tag_suffix: str | None = None,
    repeats: list[int] | None = None,
    folds: list[int] | None = None,
    datasets: list[str] | None = None,
    raise_on_failure: bool = True,
) -> Path:
    import pandas as pd

    from tabarena.benchmark.experiment import AGModelOuterExperiment
    from tabarena.nips2025_utils.end_to_end import EndToEnd
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext

    spec = MODELS[model_key]
    model_cls = resolve_model_class(model_key)

    restrict: list[str] = []
    if _classification_only(model_cls):
        restrict.append("classification")
    row_pred = _row_cap_predicate(model_cls)
    if row_pred:
        restrict.append(row_pred)

    if debug:
        mode = "debug"
        runner_kwargs = {"datasets": DEBUG_DATASETS, "subset": list(dict.fromkeys(["lite", *restrict]))}
    elif lite:
        mode = "lite"
        runner_kwargs = {"subset": list(dict.fromkeys(["small", "lite", *restrict]))}
    else:
        # full protocol: all 51 TabArena datasets, full folds x repeats grid (816 cells)
        mode = "full"
        runner_kwargs = {"subset": restrict or None}

    # Optional restrictions (compose with any mode). Put the scope in tag_suffix so that
    # differently-scoped runs never share a cache.
    if repeats is not None:
        runner_kwargs["repeats"] = list(repeats)
    if folds is not None:
        runner_kwargs["folds"] = list(folds)
    if datasets is not None:
        if mode != "full":
            raise ValueError(f"datasets sharding is only supported in full mode, not {mode!r} "
                             "(debug/lite already pin their own dataset subset)")
        runner_kwargs["datasets"] = list(datasets)

    # Registry defaults overridden by the config's hyperparameters.
    hp = {**spec.hyperparameters, **(hyperparameters or {})}

    tag = model_key
    if tag_suffix:
        tag += f"_{tag_suffix}"
    if mode != "full":
        tag += f"_{mode}"
    if fresh:
        tag += "_rerun"
    root = _runs_dir()
    expname = str(root / "experiments" / tag)
    eval_dir = root / "eval" / tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Models following the ft_* convention write per-fit logs (ft_fits.jsonl) and, for the
    # aggregation models, per-cell member predictions into this run's eval directory.
    if hasattr(model_cls, "_get_ft_params"):
        hp.setdefault("ft_log_dir", str(eval_dir))
        hp.setdefault("ft_run_label", tag_suffix or tag)

    detail = f"  datasets={DEBUG_DATASETS}" if debug else ""
    if restrict:
        detail += f"  [restricted to {restrict}; excluded tasks imputed w/ RF (default)]"
    if fresh:
        detail += "  [FRESH: cache ignored]"
    print(f"[ttc.benchmarks.tabarena] model={model_key} ({spec.display})  mode={mode}{detail}")
    print(f"[ttc.benchmarks.tabarena] runs dir: {root}")

    context = TabArenaContext()
    if datasets is not None:
        known = set(context.task_metadata["dataset"].unique())
        unknown = [d for d in datasets if d not in known]
        if unknown:
            raise ValueError(f"unknown dataset(s): {unknown} (not among the {len(known)} TabArena tasks)")
    experiment = AGModelOuterExperiment(name=spec.display, model_cls=model_cls, model_hyperparameters=hp)
    if fresh:
        runner_kwargs = {**runner_kwargs, "cache_mode": "ignore"}
    runner_kwargs = {**runner_kwargs, "raise_on_failure": raise_on_failure}
    runner = context.make_experiment_batch_runner(expname=expname, **runner_kwargs)
    results_lst = runner.run_all(methods=[experiment])

    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=None, cache=False, cache_raw=False)
    results = end_to_end.to_results()
    new_results = results.get_results(new_result_prefix="Demo_", use_model_results=True)
    # A complete restricted run keeps all 51 tasks so the excluded ones are RF-imputed (full-51 Elo);
    # otherwise the board covers only the tasks this run produced.
    only_valid = (False if (mode == "full" and restrict and datasets is None)
                  else new_results["method"].unique())
    leaderboard = context.compare(
        output_dir=eval_dir,
        only_valid_tasks=only_valid,
        new_results=new_results,
        verbose=False,
    )
    lb = context.leaderboard_to_website_format(leaderboard)

    out_csv = eval_dir / "leaderboard.csv"
    lb.to_csv(out_csv, index=False)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(lb.to_markdown(index=False))
    print(f"\nSaved leaderboard -> {out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a registered model on the TabArena protocol.")
    parser.add_argument("model", nargs="?", choices=sorted(MODELS), help="model key (see --list)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true",
                      help=f"smoke test on {', '.join(DEBUG_DATASETS)}")
    mode.add_argument("--lite", action="store_true", help="small dataset subset, 1 fold")
    parser.add_argument("--fresh", action="store_true", help="ignore cached fits; write to <tag>_rerun")
    parser.add_argument("--list", action="store_true", help="list model keys and exit")
    args = parser.parse_args()

    if args.list:
        for k in sorted(MODELS):
            print(f"{k:<16} {MODELS[k].display}")
        return
    if not args.model:
        parser.error("a model key is required (or use --list)")
    run(args.model, lite=args.lite, debug=args.debug, fresh=args.fresh)


if __name__ == "__main__":
    main()
