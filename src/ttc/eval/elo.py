"""Fixed-reference TabArena Elo used in the paper.

Protocol
--------
* Benchmark: TabArena, 51 datasets, 816 cells (dataset x repeat x fold; the ``fold`` column of a
  TabArena ``results_per_split.csv`` already enumerates repeat x fold).
* Reference field: 68 fixed entries = the 67 official TabArena entries + our frozen TabFM
  reproduction (run ``repro_tabfm_bf16-cellcap``), renamed ``Frozen_TabFM_reference``.
* Each candidate is inserted *alone* into that field (a 69-entry fit). Different candidates are
  never fit together.
* Bradley-Terry ratings come from TabArena's official solver
  (``bencheval.elo_utils.EloHelper.compute_mle_elo``) with pairwise per-cell wins/losses/ties,
  each cell weighted by ``1 / C_d`` (C_d = number of cells of dataset d), calibrated so that
  ``RF (default)`` = 1000.
* Delta Elo = Elo(candidate) - Elo(baseline) *within the same fit*, computed before rounding and
  rounded once. Baselines: ``TA-TABPFN-3 (default)`` (TabPFN-3), ``TABICLV2 (default)``
  (TabICL v2), ``Frozen_TabFM_reference`` (TabFM).
* Normalized score: TabArena's per-dataset min/median normalization of the mean cell error,
  averaged over datasets, computed on the same 69 entries.

The weighted win/loss/tie counts of every pair are compressed into one battle row per outcome
before calling the solver; this is the exact sufficient statistic of the official per-cell battle
likelihood (checked against ``EloHelper.convert_results_to_battles`` in the tests).

Reference rows
--------------
The 67 official rows can come from (``references=``):

``"field"``      the reference-field CSV (the full frozen-TabFM board; what the paper used);
``"candidate"``  the candidate's own ``results_per_split.csv`` (a TabArena ``compare`` run embeds
                 them, already RF-imputed);
``"tabarena"``   TabArena itself (``TabArenaContext().load_results_paper()``, downloaded on first use
                 to TabArena's cache), RF-imputed exactly as ``tabarena ... compare`` does;
``"auto"``       the first of the above that contains all 67 entries;
a file path      any CSV/parquet in TabArena long format containing the 67 entries.

CSV parsing note: the paper's numbers were computed from CSVs read with pandas' default float
parser, which can differ from the stored text by ~1 ulp. CSVs are therefore read with that
parser, and in-memory sources (TabArena, parquet) are passed through the same CSV text round trip
(``_canonical_floats``), so every route yields bit-identical matrices.

CLI
---
    python -m ttc.eval.elo --candidate RUN/results_per_split.csv [--name NAME] \
        --reference-field FROZEN_TABFM/results_per_split.csv \
        --baseline "TA-TABPFN-3 (default)"
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

N_CELLS = 816
N_DATASETS = 51
CANDIDATE_PREFIX = "Demo_"
FROZEN_TABFM_NAME = "Frozen_TabFM_reference"
FROZEN_TABFM_TAG = "repro_tabfm_bf16-cellcap"
CALIBRATION_METHOD = "RF (default)"
CALIBRATION_ELO = 1000
BASELINES = {
    "TabPFN-3": "TA-TABPFN-3 (default)",
    "TabICL v2": "TABICLV2 (default)",
    "TabFM": FROZEN_TABFM_NAME,
}

# The 67 official TabArena entries of the paper's reference field (TabArena results of
# 2025-06-12 .. 2026-05-13 as returned by TabArenaContext().load_results_paper()).
REFERENCE_METHODS: tuple[str, ...] = tuple(
    [f"{m} ({v})" for m in ["CAT", "EBM", "FASTAI", "GBM", "KNN", "LR", "MNCA_GPU", "NN_TORCH",
                            "PB", "REALMLP_GPU", "REALTABPFN-V2.5", "RF", "TA-ILTM", "TABDPT_GPU",
                            "TABM_GPU", "TABPFNV2_GPU", "TABSTAR", "XGB", "XRFM_GPU", "XT"]
     for v in ["default", "tuned", "tuned + ensemble"]]
    + ["AutoGluon 1.4 (best, 4h)", "AutoGluon 1.5 (extreme, 4h)", "MITRA_GPU (default)",
       "TA-TABPFN-3 (default)", "TABICLV2 (default)", "TABICL_GPU (default)",
       "TABPFN-V2.6 (default)"])
assert len(REFERENCE_METHODS) == 67


# ----------------------------------------------------------------------------- loading

def _canonical_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through CSV text with pandas' default parser (see module docstring)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def read_results(path: str | Path) -> pd.DataFrame:
    """Read a TabArena long-format results table (``results_per_split.csv`` or parquet)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return _canonical_floats(pd.read_parquet(path))
    return pd.read_csv(path)


def error_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Pivot long results to a complete (dataset, fold) x method matrix of ``metric_error``."""
    assert not data.duplicated(["dataset", "fold", "method"]).any(), "duplicate (dataset, fold, method)"
    result = data.pivot(index=["dataset", "fold"], columns="method", values="metric_error").sort_index()
    assert len(result) == N_CELLS, f"expected {N_CELLS} cells, got {len(result)}"
    assert result.index.get_level_values("dataset").nunique() == N_DATASETS
    return result


def candidate_methods(data: pd.DataFrame) -> list[str]:
    """Methods that are not official references (our runs are named ``Demo_*``)."""
    return sorted(m for m in data["method"].unique() if m not in REFERENCE_METHODS)


_TABARENA_CACHE: dict[str, pd.DataFrame] = {}


def load_tabarena_references() -> pd.DataFrame:
    """The 67 official TabArena entries, RF-imputed as in ``TabArenaContext.compare``.

    Uses TabArena's own loader (downloads the published results to TabArena's cache on first
    use) and ``tabarena.nips2025_utils.compare.prepare_data`` with ``fillna="RF (default)"``.
    """
    if "refs" not in _TABARENA_CACHE:
        from tabarena.nips2025_utils.compare import prepare_data
        from tabarena.nips2025_utils.tabarena_context import TabArenaContext

        raw = TabArenaContext().load_results_paper(download_results="auto")
        raw = raw.drop(columns=["method_metadata"], errors="ignore")
        missing = set(REFERENCE_METHODS) - set(raw["method"])
        if missing:
            raise RuntimeError(f"TabArena results lack reference entries: {sorted(missing)}")
        raw = raw[raw["method"].isin(REFERENCE_METHODS)]
        filled = prepare_data(raw, fillna=CALIBRATION_METHOD)
        cols = ["dataset", "fold", "method", "metric_error", "imputed"]
        _TABARENA_CACHE["refs"] = _canonical_floats(filled[cols])
    return _TABARENA_CACHE["refs"].copy()


def _has_references(data: pd.DataFrame) -> bool:
    return set(REFERENCE_METHODS) <= set(data["method"].unique())


def reference_matrix(source: str | Path, candidate: pd.DataFrame | None = None,
                     field_data: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """Resolve the 67-column official reference matrix; returns (matrix, description)."""
    src = str(source)
    order = {"auto": ["field", "candidate", "tabarena"], "candidate": ["candidate"],
             "field": ["field"], "tabarena": ["tabarena"]}.get(src)
    if order is None:  # a file path
        data = read_results(src)
        if not _has_references(data):
            raise ValueError(f"{src} does not contain all 67 reference entries")
        return error_matrix(data[data["method"].isin(REFERENCE_METHODS)]), src
    for kind in order:
        if kind == "candidate" and candidate is not None and _has_references(candidate):
            data = candidate
        elif kind == "field" and field_data is not None and _has_references(field_data):
            data = field_data
        elif kind == "tabarena":
            data = load_tabarena_references()
        else:
            continue
        refs = error_matrix(data[data["method"].isin(REFERENCE_METHODS)])
        assert list(refs.columns) == sorted(REFERENCE_METHODS)
        return refs, kind
    raise ValueError(f"no source with all 67 reference entries for references={src!r}")


def frozen_tabfm_column(field_data: pd.DataFrame, method: str | None = None) -> pd.Series:
    """The 68th reference: the single candidate column of the frozen-TabFM board."""
    own = [method] if method else candidate_methods(field_data)
    if len(own) != 1:
        raise ValueError(f"reference-field CSV must hold exactly one non-reference method, got {own}")
    rows = field_data[field_data["method"] == own[0]]
    col = error_matrix(rows)[own[0]]
    col.name = FROZEN_TABFM_NAME
    return col


# ----------------------------------------------------------------------------- fitting

def _elo_helper():
    from bencheval.elo_utils import EloHelper  # TabArena's official solver

    return EloHelper(task_col="dataset", split_col="fold")


def compressed_battles(errors: pd.DataFrame) -> pd.DataFrame:
    """Weighted pairwise win/loss/tie totals (cell weight 1/C_d), one row per pair x outcome."""
    names = errors.columns.to_numpy()
    a, b = np.triu_indices(len(names), 1)
    values = errors.to_numpy()
    counts = errors.groupby(level="dataset").size()
    weights = 1 / counts.reindex(errors.index.get_level_values("dataset")).to_numpy()
    va, vb = values[:, a], values[:, b]
    wins = ((va < vb) * weights[:, None]).sum(axis=0)
    losses = ((va > vb) * weights[:, None]).sum(axis=0)
    ties = ((va == vb) * weights[:, None]).sum(axis=0)
    battles = pd.concat([
        pd.DataFrame({"method_1": names[a], "method_2": names[b], "winner": winner, "weight": weight})
        for winner, weight in [("1", wins), ("2", losses), ("tie", ties)]
    ], ignore_index=True)
    return battles[battles.weight > 0]


def fit_elo(errors: pd.DataFrame) -> pd.Series:
    """Official TabArena MLE Elo of every column of a complete error matrix (RF (default)=1000)."""
    assert np.isfinite(errors.to_numpy()).all(), "error matrix has missing/non-finite cells"
    return _elo_helper().compute_mle_elo(compressed_battles(errors),
                                         calibration_framework=CALIBRATION_METHOD,
                                         calibration_elo=CALIBRATION_ELO)


def normalized_scores(errors: pd.DataFrame) -> pd.Series:
    """TabArena normalized score: 1 - clip((err - best) / (median - best), 0, 1), dataset mean."""
    means = errors.groupby(level="dataset").mean()
    low, med = means.min(axis=1), means.median(axis=1)
    return (1 - means.sub(low, axis=0).div((med - low).clip(lower=1e-5), axis=0).clip(0, 1)).mean()


# ----------------------------------------------------------------------------- candidates

@dataclass
class EloResult:
    name: str
    method: str
    elo: float
    baseline: str
    baseline_elo: float
    delta_elo: float
    delta_rounded: int
    score: float
    n_entries: int
    n_imputed_cells: int
    reference_source: str
    board: pd.DataFrame = field(repr=False)  # elo + score of all entries in this fit

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("board")
        return d


def build_field(candidate: pd.DataFrame, field_data: pd.DataFrame, method: str | None = None,
                references: str | Path = "auto", check_references: bool = True,
                ) -> tuple[pd.DataFrame, str, int, str]:
    """69-column matrix: 67 references + frozen TabFM + one candidate.

    Returns (matrix, candidate method, number of RF-imputed candidate cells, reference source).
    """
    own = [method] if method else candidate_methods(candidate)
    if len(own) != 1:
        raise ValueError(f"candidate CSV must hold exactly one non-reference method "
                         f"(pass method=...), got {own}")
    method = own[0]
    refs, source = reference_matrix(references, candidate, field_data)
    if check_references:  # every embedded copy of the references must be the same field
        for kind, data in (("field", field_data), ("candidate", candidate)):
            if kind != source and _has_references(data):
                other = error_matrix(data[data["method"].isin(REFERENCE_METHODS)])
                # Re-written CSVs may differ by float-parse ulps (rtol 1e-12); anything more is
                # a different reference field.
                pd.testing.assert_frame_equal(refs, other, check_exact=False, rtol=1e-12, atol=0)

    fm = frozen_tabfm_column(field_data)
    rows = candidate[candidate["method"] == method]
    assert not rows.duplicated(["dataset", "fold"]).any()
    cand = rows.set_index(["dataset", "fold"])["metric_error"].reindex(refs.index)
    # TabArena imputes cells a method did not produce with RF (default) (fillna_metrics).
    n_imputed = int(cand.isna().sum())
    cand = cand.fillna(refs[CALIBRATION_METHOD])
    if n_imputed == 0 and "imputed" in rows:
        n_imputed = int(rows["imputed"].fillna(False).astype(bool).sum())
    if method in REFERENCE_METHODS or method == FROZEN_TABFM_NAME:
        raise ValueError(f"candidate name {method!r} collides with a reference entry")
    if cand.equals(fm.rename(method)):
        raise ValueError("candidate is identical to the frozen TabFM reference; it is already "
                         f"in the field (read {FROZEN_TABFM_NAME!r} from any fit's board)")
    errors = refs.copy()
    errors[FROZEN_TABFM_NAME] = fm
    errors[method] = cand
    errors = errors.sort_index(axis=1)
    assert errors.shape == (N_CELLS, 69)
    return errors, method, n_imputed, source


def evaluate_candidate(candidate: str | Path | pd.DataFrame, reference_field: str | Path | pd.DataFrame,
                       baseline: str = BASELINES["TabPFN-3"], name: str | None = None,
                       method: str | None = None, references: str | Path = "auto",
                       check_references: bool = True) -> EloResult:
    """Insert one candidate into the fixed 68-entry field and fit Elo (69 entries)."""
    cand = candidate if isinstance(candidate, pd.DataFrame) else read_results(candidate)
    fdata = reference_field if isinstance(reference_field, pd.DataFrame) else read_results(reference_field)
    errors, method, n_imp, source = build_field(cand, fdata, method, references, check_references)
    if baseline not in errors.columns:
        raise ValueError(f"baseline {baseline!r} not in the field")
    elo, score = fit_elo(errors), normalized_scores(errors)
    delta = float(elo[method] - elo[baseline])
    board = pd.DataFrame({"elo": elo, "score": score}).sort_values("elo", ascending=False)
    return EloResult(name=name or method, method=method, elo=float(elo[method]), baseline=baseline,
                     baseline_elo=float(elo[baseline]), delta_elo=delta, delta_rounded=round(delta),
                     score=float(score[method]), n_entries=errors.shape[1], n_imputed_cells=n_imp,
                     reference_source=source, board=board)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ttc.eval.elo", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", action="append", required=True, metavar="CSV",
                   help="results_per_split.csv of a candidate (repeatable; each fit separately)")
    p.add_argument("--name", action="append", default=[], help="display name per --candidate")
    p.add_argument("--method", action="append", default=[],
                   help="candidate method name per --candidate (default: the single non-reference method)")
    p.add_argument("--reference-field", required=True, metavar="CSV",
                   help=f"results_per_split.csv of the frozen TabFM reproduction ({FROZEN_TABFM_TAG})")
    p.add_argument("--baseline", default=BASELINES["TabPFN-3"],
                   help=f"entry Delta Elo is measured against (e.g. {', '.join(map(repr, BASELINES.values()))})")
    p.add_argument("--references", default="auto",
                   help="source of the 67 official rows: auto | candidate | field | tabarena | PATH")
    p.add_argument("--no-check-references", action="store_true",
                   help="do not require reference rows of all inputs to be bit-identical")
    p.add_argument("--show", action="append", default=[], metavar="ENTRY",
                   help="also print Elo/score of this field entry in each fit (repeatable)")
    p.add_argument("--json", action="store_true", help="print JSON lines instead of a table")
    args = p.parse_args(argv)
    for opt in ("name", "method"):
        vals = getattr(args, opt)
        if vals and len(vals) != len(args.candidate):
            p.error(f"--{opt} must be given once per --candidate")

    field_data = read_results(args.reference_field)
    rows = []
    for i, path in enumerate(args.candidate):
        r = evaluate_candidate(path, field_data, baseline=args.baseline,
                               name=args.name[i] if args.name else None,
                               method=args.method[i] if args.method else None,
                               references=args.references,
                               check_references=not args.no_check_references)
        d = r.to_dict()
        d["candidate_csv"] = str(path)
        for entry in args.show:
            d[f"elo[{entry}]"] = float(r.board.loc[entry, "elo"])
            d[f"score[{entry}]"] = float(r.board.loc[entry, "score"])
        rows.append(d)
        if args.json:
            print(json.dumps(d))
    if not args.json:
        cols = ["name", "elo", "baseline_elo", "delta_elo", "delta_rounded", "score",
                "n_entries", "reference_source"] + [c for c in rows[0] if c.startswith(("elo[", "score["))]
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.float_format", "{:.6f}".format):
            print(f"baseline: {args.baseline}")
            print(pd.DataFrame(rows)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
