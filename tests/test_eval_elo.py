"""Reproduce the paper's Elo / Delta Elo / normalized score / paired-error numbers.

Data-dependent; tests skip when the data are absent. Two sources (set either or both):

* ``TTC_PAPER_RESULTS_DIR``: folder of ``<run tag>/results_per_split.csv`` (the full TabArena boards
  of the original runs, which embed the 67 reference entries).
* ``TTC_HF_RESULTS_DIR``: the Hugging Face results folder, ``raw/<arm>/results_per_split.csv``
  holding only the arm's own rows; the reference entries are then fetched from TabArena.

    TTC_PAPER_RESULTS_DIR=/path/to/eval pytest tests/test_eval_elo.py
    TTC_HF_RESULTS_DIR=/path/to/hf_results pytest tests/test_eval_elo.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "ttc" / "eval" / "elo.py").exists():  # test this checkout, not another installed ttc
    sys.path.insert(0, str(_SRC))

from ttc.eval import elo as E  # noqa: E402
from ttc.eval import paired as P  # noqa: E402

RESULTS_DIR = os.environ.get("TTC_PAPER_RESULTS_DIR")
HF_DIR = os.environ.get("TTC_HF_RESULTS_DIR")
FIELD_TAG = E.FROZEN_TABFM_TAG
TOL = 1e-6

# (arm, run tag, baseline, unrounded Elo, unrounded Delta Elo, normalized score) -- paper values
# (artifacts/elo68_paper/paper_arms.csv, artifacts/headline_cost/points.csv).
ARMS = [
    ("TabPFN-3 frozen", "repro_tabpfn3-ft_diagscale-board-base-tabpfn3", "TA-TABPFN-3 (default)",
     1658.2965704015296, 0.5441288428048665, 0.6759926465203421),
    ("TabPFN-3 32 views", "repro_tabpfn3-dev_agg-nest32", "TA-TABPFN-3 (default)",
     1675.353753626879, 20.718540398403547, 0.6963295855202154),
    ("TabPFN-3 DiagScale", "repro_tabpfn3-ft_diagscale-board-tabpfn3-diagscale-fx3e-2",
     "TA-TABPFN-3 (default)", 1681.940184257033, 25.664866793884812, 0.6982535589084367),
    ("TabPFN-3 full FT", "_paper_adaptation/repro_tabpfn3-ft_full-e30-lr1e-6", "TA-TABPFN-3 (default)",
     1681.2614542388362, 25.26670546458968, 0.7041739006555048),
    ("TabPFN-3 aggregation 96", "repro_tabpfn3-agg_aggonly96", "TA-TABPFN-3 (default)",
     1721.179036193109, 66.9285793722172, 0.7361929036357467),
    ("TabPFN-3 DiagScale + aggregation", "repro_tabpfn3-agg-ft_aggft-board", "TA-TABPFN-3 (default)",
     1747.5084984243956, 92.94964962641257, 0.7662951426193787),
    ("TabICL v2 frozen", "repro_tabicl-ft_diagscale-board-base-tabicl", "TABICLV2 (default)",
     1592.0747612286468, 9.18309120750473, 0.6174946637006296),
    ("TabICL v2 DiagScale", "repro_tabicl-ft_diagscale-board-tabicl-diagscale-fx3e-2", "TABICLV2 (default)",
     1664.0977006192577, 82.1179296530081, 0.6763592466875675),
    ("TabICL v2 full FT", "repro_tabicl-ft_diagscale-board-tabicl-full-fx1e-5", "TABICLV2 (default)",
     1658.6982822904895, 77.2014870443561, 0.670584795414111),
    ("TabFM DiagScale", "repro_tabfm-ft_diagscale-bf16", E.FROZEN_TABFM_NAME,
     1812.1363312729527, 20.331105399292483, 0.8998982251135323),
    ("TabFM full FT", "repro_tabfm-ft_full-bf16-fx1e-4", E.FROZEN_TABFM_NAME,
     1819.9992160263407, 28.470432220896782, 0.909626276192008),
]
# Frozen TabFM is itself the 68th reference; the paper reads it from the TabFM full-FT fit.
FROZEN_TABFM = ("repro_tabfm-ft_full-bf16-fx1e-4", 1791.5287838054446, 0.8814865590636426)

# Paper Table "DiagScale vs full FT" (artifacts/review88/diagscale_paired_summary.csv).
PAIRED = [
    ("repro_tabpfn3-ft_diagscale-board-tabpfn3-diagscale-fx3e-2",
     "_paper_adaptation/repro_tabpfn3-ft_full-e30-lr1e-6",
     -0.09062738970720949, -0.28330071324910033, 0.08365802301939226, 21),
    ("repro_tabicl-ft_diagscale-board-tabicl-diagscale-fx3e-2",
     "repro_tabicl-ft_diagscale-board-tabicl-full-fx1e-5",
     -0.45818822796610137, -1.0640202914185641, -0.01579886508080144, 26),
    ("repro_tabfm-ft_diagscale-bf16", "repro_tabfm-ft_full-bf16-fx1e-4",
     0.10164637791684515, -0.0538022161677372, 0.2859631089747285, 28),
]


# Run tag -> arm name of the Hugging Face results folder (raw/<arm>/results_per_split.csv).
HF_ARMS = {
    "repro_tabpfn3-ft_diagscale-board-base-tabpfn3": "tabpfn3_frozen",
    "repro_tabpfn3-dev_agg-nest32": "tabpfn3_native_views_32",
    "_paper_adaptation/repro_tabpfn3-ft_full-e30-lr1e-6": "tabpfn3_full_ft",
    "repro_tabpfn3-ft_diagscale-board-tabpfn3-diagscale-fx3e-2": "tabpfn3_diagscale",
    "repro_tabpfn3-agg_aggonly96": "tabpfn3_aggregation_96",
    "repro_tabpfn3-agg-ft_aggft-board": "tabpfn3_diagscale_aggregation_96",
    "repro_tabicl-ft_diagscale-board-base-tabicl": "tabiclv2_frozen",
    "repro_tabicl-ft_diagscale-board-tabicl-full-fx1e-5": "tabiclv2_full_ft",
    "repro_tabicl-ft_diagscale-board-tabicl-diagscale-fx3e-2": "tabiclv2_diagscale",
    "repro_tabfm_bf16-cellcap": "tabfm_frozen",
    "repro_tabfm-ft_full-bf16-fx1e-4": "tabfm_full_ft",
    "repro_tabfm-ft_diagscale-bf16": "tabfm_diagscale",
}


@pytest.fixture(scope="module", params=["runs", "hf"])
def source(request) -> str:
    """Which results folder the module runs on: research runs or the Hugging Face export."""
    if request.param == "runs" and not RESULTS_DIR:
        pytest.skip("TTC_PAPER_RESULTS_DIR not set")
    if request.param == "hf" and not HF_DIR:
        pytest.skip("TTC_HF_RESULTS_DIR not set")
    return request.param


def _path(tag: str, source: str) -> Path:
    if source == "hf":
        path = Path(HF_DIR) / "raw" / HF_ARMS[tag] / "results_per_split.csv"
        if not path.exists():
            pytest.skip(f"no {path}")
        return path
    for cand in (Path(RESULTS_DIR) / tag, Path(RESULTS_DIR) / Path(tag).name):
        if (cand / "results_per_split.csv").exists():
            return cand / "results_per_split.csv"
    pytest.skip(f"no results for {tag} under {RESULTS_DIR}")


def _tabarena_available() -> bool:
    try:
        E.load_tabarena_references()
        return True
    except Exception:  # not installed / no network and no cache
        return False


@pytest.fixture(scope="module")
def field(source):
    return E.read_results(_path(FIELD_TAG, source))


def _references_mode(data, field_data) -> str:
    """Full boards carry the 67 references; reduced exports need TabArena."""
    if E._has_references(data) or E._has_references(field_data):
        return "auto"
    if not _tabarena_available():
        pytest.skip("reduced CSVs need the TabArena reference results (tabarena not available)")
    return "tabarena"


@pytest.mark.parametrize("arm,tag,baseline,elo,delta,score", ARMS, ids=[a[0] for a in ARMS])
def test_paper_arm_elo(source, field, arm, tag, baseline, elo, delta, score):
    data = E.read_results(_path(tag, source))
    r = E.evaluate_candidate(data, field, baseline=baseline, references=_references_mode(data, field))
    assert r.n_entries == 69
    assert abs(r.elo - elo) < TOL
    assert abs(r.delta_elo - delta) < TOL
    assert r.delta_rounded == round(delta)
    assert abs(r.score - score) < 1e-9


def test_frozen_tabfm_reference(source, field):
    tag, elo, score = FROZEN_TABFM
    data = E.read_results(_path(tag, source))
    r = E.evaluate_candidate(data, field, baseline=E.FROZEN_TABFM_NAME,
                             references=_references_mode(data, field))
    assert abs(r.board.loc[E.FROZEN_TABFM_NAME, "elo"] - elo) < TOL
    assert abs(r.board.loc[E.FROZEN_TABFM_NAME, "score"] - score) < 1e-9


@pytest.mark.parametrize("arm,tag,baseline,elo,delta,score", ARMS[:1] + ARMS[7:8] + ARMS[9:10],
                         ids=["TabPFN-3 frozen", "TabICL v2 DiagScale", "TabFM DiagScale"])
def test_reduced_csv_with_tabarena_references(source, field, arm, tag, baseline, elo, delta, score):
    """Candidate-only CSV + candidate-only frozen TabFM + TabArena references = same fit."""
    if not _tabarena_available():
        pytest.skip("tabarena reference results not available")
    data = E.read_results(_path(tag, source))
    own = data[data.method.str.startswith(E.CANDIDATE_PREFIX)]
    fm_own = field[field.method.str.startswith(E.CANDIDATE_PREFIX)]
    r = E.evaluate_candidate(own, fm_own, baseline=baseline, references="auto")
    assert r.reference_source == "tabarena"
    assert abs(r.elo - elo) < TOL and abs(r.delta_elo - delta) < TOL
    assert abs(r.score - score) < 1e-9
    if E._has_references(field):  # TabArena download is bit-identical to the embedded rows
        refs, _ = E.reference_matrix("tabarena")
        refs_csv, _ = E.reference_matrix("field", field_data=field)
        assert refs.equals(refs_csv)


def test_compressed_fit_matches_official_battles(field):
    """One-row-per-pair compression == TabArena's per-cell battle conversion."""
    if not E._has_references(field):
        pytest.skip("needs the full frozen-TabFM board")
    from bencheval.elo_utils import EloHelper

    helper = EloHelper(task_col="dataset", split_col="fold")
    direct = helper.compute_mle_elo(helper.convert_results_to_battles(field),
                                    calibration_framework="RF (default)", calibration_elo=1000)
    compressed = E.fit_elo(E.error_matrix(field))
    assert float((direct - compressed).abs().max()) < 1e-5


@pytest.mark.parametrize("cand,base,est,lo,hi,wins", PAIRED, ids=["TabPFN-3", "TabICL v2", "TabFM"])
def test_paired_diagscale_vs_full_ft(source, cand, base, est, lo, hi, wins):
    r, per_ds = P.paired_change(P.cell_errors(_path(cand, source)), P.cell_errors(_path(base, source)))
    assert (r.datasets, r.cells, r.wins) == (51, 816, wins)
    assert abs(r.estimate - est) < 1e-9 and abs(r.lo - lo) < 1e-9 and abs(r.hi - hi) < 1e-9
    assert len(per_ds) == 51


def test_cli_smoke(source, field, capsys):
    tag = ARMS[0][1]
    mode = _references_mode(E.read_results(_path(tag, source)), field)
    assert E.main(["--candidate", str(_path(tag, source)),
                   "--reference-field", str(_path(FIELD_TAG, source)),
                   "--references", mode, "--json"]) == 0
    assert '"delta_rounded": 1' in capsys.readouterr().out


def test_hf_arms_table():
    """data/arms.csv of the Hugging Face folder: paper values and checksums of raw/."""
    if not HF_DIR or not (Path(HF_DIR) / "data" / "arms.csv").exists():
        pytest.skip("TTC_HF_RESULTS_DIR with data/arms.csv not set")
    import pandas as pd

    arms = pd.read_csv(Path(HF_DIR) / "data" / "arms.csv").set_index("arm")
    assert len(arms) == 12 and set(arms.index) == set(HF_ARMS.values())
    for _, tag, baseline, elo, delta, score in ARMS:
        a = arms.loc[HF_ARMS[tag]]
        assert a.baseline == baseline and a.source_run == tag
        assert abs(a.elo - elo) < TOL and abs(a.delta_elo - delta) < TOL
        assert a.delta_elo_rounded == round(delta) and abs(a.normalized_score - score) < 1e-9
    for arm, a in arms.iterrows():
        assert (a.n_cells, a.n_datasets) == (816, 51)
        assert hashlib.sha256((Path(HF_DIR) / a.raw_file).read_bytes()).hexdigest() == a.sha256
