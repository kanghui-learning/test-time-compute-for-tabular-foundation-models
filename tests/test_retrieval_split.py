"""CPU tests for the model-aware retrieval overflow policy ``split``.

The recursive split was first run as a monkey-patch of ``sklearn.cluster.KMeans`` inside
``_eval_modelaware`` (research script, function ``split_labels``). Its cluster assignments and
split logs on the synthetic inputs below are frozen in ``data/retrieval_split_reference.json``;
the integrated ``ttc.benchmarks.row_modelaware.split_labels`` / ``_cluster_queries`` must
reproduce them exactly. The property tests at the end check the policy itself.

    python -m pytest tests/test_retrieval_split.py -q
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from ttc.benchmarks.row_modelaware import _cluster_queries, _indicator, split_labels

REFERENCE = json.loads((Path(__file__).parent / "data" / "retrieval_split_reference.json").read_text())


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _plain(splits):
    """Split logs with plain Python numbers, comparable to the JSON reference."""
    return json.loads(json.dumps(splits, default=lambda o: o.item()))


# ---------------------------------------------------------------- synthetic inputs
def case_disjoint():
    """32 queries with disjoint 3-row retrievals (the reference's own self-test input)."""
    ti = np.arange(96).reshape(32, 3)
    ind = csr_matrix((np.ones(ti.size), (np.repeat(np.arange(32), 3), ti.ravel())), shape=(32, 96))
    return ind, ti


def case_grouped(n_q=400, r=25, pool=2000, n_groups=5, seed=0):
    """Queries in latent groups: 20 of 25 retrievals from the group's preferred 300-row block, 5
    uniform over the pool -> overlapping unions, indicator built as in production (_indicator)."""
    rng = np.random.default_rng(seed)
    blocks = [rng.choice(pool, 300, replace=False) for _ in range(n_groups)]
    g = rng.integers(0, n_groups, n_q)
    ti = np.empty((n_q, r), np.int64)
    for i in range(n_q):
        own = rng.choice(blocks[g[i]], 20, replace=False)
        rest = rng.choice(np.setdiff1d(np.arange(pool), own), r - 20, replace=False)
        ti[i] = np.concatenate([own, rest])
    ind, _ = _indicator(ti)
    return ind, ti


CASES = {
    # name: (inputs, base_k, cap, max_labels)
    "disjoint_multi_split": (case_disjoint, 2, 12, 64),
    "grouped_deep_split": (case_grouped, 3, 250, 200),
    "grouped_some_split": (case_grouped, 4, 900, 64),
    "grouped_no_split": (case_grouped, 4, 2000, 64),
}


# ---------------------------------------------------------------- equality with the reference
@pytest.mark.parametrize("name", sorted(CASES))
def test_split_matches_reference(name):
    ref = REFERENCE["cases"][name]
    make, base_k, cap, max_labels = CASES[name]
    ind, ti = make()
    lab, splits = _quiet(split_labels, ind, ti, base_k=base_k, cap=cap, max_labels=max_labels)
    assert lab.tolist() == ref["labels"]
    assert _plain(splits) == ref["splits"]
    # the policy dispatcher used by _eval_modelaware gives the same labels
    lab2, splits2 = _quiet(_cluster_queries, ind, ti, base_k, "split", cap, max_labels)
    assert lab2.tolist() == ref["labels"] and _plain(splits2) == ref["splits"]


def test_cases_exercise_recursion():
    """Guard that the synthetic cases really need multiple, nested splits."""
    for name in ("disjoint_multi_split", "grouped_deep_split"):
        splits = REFERENCE["cases"][name]["splits"]
        children = {s["child"] for s in splits}
        assert len(splits) >= 3, name
        assert any(s["cluster"] in children for s in splits), f"{name}: no child was split again"


def test_label_budget_exhaustion():
    ind, ti = case_grouped()
    with pytest.raises(AssertionError, match="label budget exhausted"):
        _quiet(split_labels, ind, ti, base_k=3, cap=250, max_labels=10)


def test_monkeypatched_dispatch_equivalence():
    """The monkey-patch replaced KMeans.fit_predict with split_labels at base_k=25, cap=4M,
    max_labels=64. With 50k queries the integrated base K is the paper's G rule = 25, so the
    integrated dispatcher must agree. Scaled down here: same wiring, small cap."""
    from ttc.benchmarks.row_coreset_large import _k_clusters
    assert _k_clusters(50_000) == 25
    ind, ti = case_grouped(n_q=600, r=25, pool=3000, seed=3)
    base_k = min(_k_clusters(len(ti)), len(ti))          # = 4 here
    lab, _ = _quiet(_cluster_queries, ind, ti, base_k, "split", 300, 200)
    assert lab.tolist() == REFERENCE["cases"]["monkeypatch_wiring"]["labels"]


# ---------------------------------------------------------------- properties (no reference needed)
@pytest.mark.parametrize("name", sorted(CASES))
def test_split_properties(name):
    make, base_k, cap, max_labels = CASES[name]
    ind, ti = make()
    lab, splits = _quiet(split_labels, ind, ti, base_k=base_k, cap=cap, max_labels=max_labels)
    assert lab.shape == (ti.shape[0],) and lab.dtype == np.int64
    assert all(len(np.unique(ti[lab == c])) <= cap for c in np.unique(lab))     # every union fits
    assert lab.max() < base_k + len(splits) <= max_labels
    for s in splits:                                                              # balanced halves
        assert s["sizes"] == [s["n_q"] // 2, s["n_q"] - s["n_q"] // 2] and s["demand"] > cap
    lab2, splits2 = _quiet(split_labels, ind, ti, base_k=base_k, cap=cap, max_labels=max_labels)
    assert np.array_equal(lab, lab2) and splits == splits2                      # deterministic


def test_no_split_equals_truncate_clustering():
    """Uncapped groups are untouched: split labels == the truncate policy's plain KMeans labels."""
    ind, ti = case_grouped()
    lab, splits = _quiet(split_labels, ind, ti, base_k=4, cap=10_000)
    plain, none = _cluster_queries(ind, ti, 4, "truncate", 10_000)
    assert not splits and not none
    assert np.array_equal(lab, plain)
    assert np.array_equal(plain, KMeans(n_clusters=4, random_state=0, n_init=4).fit_predict(ind))


def test_overlapping_large_group_not_split():
    """A large query group whose retrievals overlap stays whole if its union fits."""
    ind, _ = case_disjoint()
    repeated = np.tile(np.arange(3), (32, 1))
    _, splits = _quiet(split_labels, ind, repeated, base_k=2, cap=6)
    assert not splits


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
