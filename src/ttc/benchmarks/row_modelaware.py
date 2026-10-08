"""Model-aware (attention-guided) row retrieval for frozen TabPFN-3 on large tables.

Adapts LimiX's retrieval inference to TabPFN-3:
  1. SCORE   fit on the pool; ONE full-context forward on the query (test) set capturing the LAST ICL
             layer's query->context softmax attention, mean over heads -- streamed as per-query
             top-r pool indices + their attention values (r = --retrieval-len, paper 500).
  2. CLUSTER KMeans (random_state=0, n_init=4) over the BINARY retrieval-indicator matrix
             [n_test, n_unique_retrieved], G = min(n_test, max(4, min(30, n_test // 2000))) clusters
             (= 25 for the 50k-query test sets).
  3. UNION   per cluster, unique union of the members' top-r sets -> that cluster's context.
  4. FIT     one frozen TabPFN fit+predict per cluster on (union rows, cluster queries); class
             probabilities realigned to the pool's class order and stitched into query order.

PAST THE MEMORY WALL (feed-all infeasible beyond 4.0-4.5M rows on 80GB): --shard-size S splits the
(shuffled) pool into fixed S-row blocks, remainder as its own last block; ONE scoring fit per block;
each query takes each block's top-(r * |block|/N) (proportional quota, largest-remainder rounded to
sum exactly r); selection uses within-block ranks -- cross-block attention values are never
compared as a ranking (each block's softmax normalizes over its own rows). Steps 2-4 draw from the
WHOLE pool.

UNION CAPACITY (--union-cap C, paper 4M) with --overflow:
  truncate (default)  a cluster whose union exceeds C keeps the C rows with the largest summed
                      retained attention over its queries (block-local values when sharded).
  split               recursively bisect any cluster whose union exceeds C: 2-center KMeans on its
                      queries' binary indicators, queries ordered by distance difference, split into
                      balanced halves, until every union fits (at most --max-clusters clusters).
                      Uses no labels or predictions. Paper: Higgs at 6M-10M.

The feed-all arm (--arms full) fits the whole pool as context (ttc.benchmarks.row_coreset_large).

    python -m ttc.benchmarks.row_modelaware --smoke                                  # CPU checks
    python -m ttc.benchmarks.row_modelaware --datasets Covertype --n-avail 100000    # GPU
    python -m ttc.benchmarks.row_modelaware --datasets Higgs_full --n-avail 10000000 \
        --arms modelaware --shard-size 2000000 --union-size 4000000 --overflow split  # past the wall
"""
from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import ttc.benchmarks.row_coreset_large as rcl                          # noqa: E402
from ttc.benchmarks.hardgather import _align, _model                    # noqa: E402
from ttc.benchmarks.row_coreset_large import (                          # noqa: E402
    _TabPFNOOM, _eval_full, _k_clusters, _peak_gb, _scores, load_large,
)
from ttc.retrieval import host_probe                                    # noqa: E402

OVERFLOW = ("truncate", "split")


def _resolve_call_idx(m, call_idx: int) -> int:
    """-1 -> the LAST ICL layer's test->train call (call order = ICL block order; the trailing
    test->train calls on classification are the class decoder, which n_icl_layers excludes)."""
    if call_idx >= 0:
        return call_idx
    n_icl = host_probe.n_icl_layers(m)
    if not n_icl:
        raise RuntimeError("could not locate icl_blocks to resolve the last ICL layer; pass --call-idx")
    return n_icl - 1


def _quota(k: int, sizes) -> np.ndarray:
    """Per-shard quotas proportional to shard size, largest-remainder rounded to sum exactly k."""
    raw = np.asarray(sizes, float) * (k / float(np.sum(sizes)))
    q = np.floor(raw).astype(int)
    for i in np.argsort(raw - q)[::-1][: k - q.sum()]:
        q[i] += 1
    return q


def _score_attn(Xtr, ytr, Xte, cat, ptype, k_ret, n_est, device, buf_gb, call_idx, shard_size, torch):
    """Attention scoring -> (ti [n_test, k_ret] GLOBAL pool indices, tv, coverage, call_idx, n_shards).

    shard_size=0 (or >= pool) -> single feed-all scoring fit. Else fixed shard_size blocks over the
    shuffled pool (remainder = last block), one fit per block, per-query top-(quota) per block,
    concatenated. Coverage = mean over queries of captured softmax mass / n_shards (each block's
    softmax sums to 1, so this is the captured fraction of the average per-context mass; reduces to
    the plain top-k mass at n_shards=1)."""
    N = len(Xtr)
    step = shard_size if 0 < shard_size < N else N
    bounds = [(b, min(b + step, N)) for b in range(0, N, step)]
    quotas = _quota(min(k_ret, N), [b1 - b0 for b0, b1 in bounds])
    ci, parts_i, parts_v = call_idx, [], []
    for (b0, b1), q in zip(bounds, quotas):
        if q == 0:
            continue
        m = _model(ptype, cat, n_est, device).fit(Xtr[b0:b1], ytr[b0:b1])
        pf = m.predict if ptype == "regression" else m.predict_proba
        if ci < 0:
            ci = _resolve_call_idx(m, call_idx)
        got = host_probe.capture_attention_topk(pf, Xte, b1 - b0, k=int(q), indices=[ci], buf_gb=buf_gb)
        ti_s, tv_s = got[ci]
        parts_i.append(ti_s.astype(np.int64) + b0)         # local block indices -> global pool indices
        parts_v.append(tv_s)
        del m, pf, got                                     # pf pins the estimator -> free before next fit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ti = np.concatenate(parts_i, axis=1)
    tv = np.concatenate(parts_v, axis=1)
    cover = float(tv.sum(1).mean() / len(bounds))
    return ti, tv, cover, ci, len(bounds)


def _indicator(ti: np.ndarray):
    """Binary retrieval-indicator CSR [n_test, n_unique_retrieved] from top-k indices [n_test, k]."""
    import scipy.sparse as sp
    uniq, inv = np.unique(ti, return_inverse=True)
    n_test, kk = ti.shape
    rows = np.repeat(np.arange(n_test), kk)
    ind = sp.csr_matrix((np.ones(ti.size, np.float32), (rows, inv.reshape(-1))),
                        shape=(n_test, len(uniq)))
    return ind, uniq


def split_labels(ind, ti, *, base_k, cap=4_000_000, max_labels=64):
    """Indicator KMeans into ``base_k`` clusters, then balanced recursive bisection of every cluster
    whose retrieved union exceeds ``cap`` rows -> (labels int64 [n_test], split log).

    A cluster is split by a 2-center KMeans (random_state=0, n_init=4) on its queries' binary
    indicators; queries are ordered (stable) by distance-to-center-0 minus distance-to-center-1 and
    the second half of that order becomes a new cluster id. Children are pushed back on the queue
    (LIFO) until every union fits. A cluster with at most cap // r queries cannot exceed the cap and
    is never split. No labels or predictions enter the rule. Raises if more than ``max_labels``
    clusters would be needed."""
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=base_k, random_state=0, n_init=4).fit_predict(ind)
    lab = lab.astype(np.int64)
    next_id, queue, splits = base_k, list(range(base_k)), []
    while queue:
        c = queue.pop()
        mask = lab == c
        nq = int(mask.sum())
        if nq <= cap // ti.shape[1]:
            continue
        demand = len(np.unique(ti[mask]))
        if demand <= cap:
            continue
        km = KMeans(n_clusters=2, random_state=0, n_init=4).fit(ind[mask])
        distances = km.transform(ind[mask])
        order = np.argsort(distances[:, 0] - distances[:, 1], kind="stable")
        idx = np.where(mask)[0]
        lab[idx[order[nq // 2:]]] = next_id
        splits.append(dict(cluster=int(c), n_q=nq, demand=int(demand), child=next_id,
                           sizes=[nq // 2, nq - nq // 2]))
        print(f"split: {splits[-1]}", flush=True)
        queue.extend([c, next_id])
        next_id += 1
        if not next_id <= max_labels:
            raise AssertionError("label budget exhausted")
    if not all(len(np.unique(ti[lab == c])) <= cap for c in np.unique(lab)):
        raise AssertionError("a split cluster still exceeds the union cap")
    return lab, splits


def _cluster_queries(ind, ti, K, overflow="truncate", union_cap=4_000_000, max_clusters=64):
    """Query-cluster labels under the overflow policy -> (labels [n_test], split log)."""
    if overflow == "split":
        return split_labels(ind, ti, base_k=K, cap=union_cap, max_labels=max_clusters)
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=K, random_state=0, n_init=4).fit_predict(ind), []


def _eval_modelaware(Xtr, ytr, Xte, yte, cat, ptype, classes, k_ret, n_est, device, torch, buf_gb,
                     call_idx=-1, shard_size=0, deploy_n_est=0, union_cap=4_000_000,
                     overflow="truncate", max_clusters=64):
    if overflow not in OVERFLOW:
        raise ValueError(f"unknown overflow policy {overflow!r} (choose from {OVERFLOW})")
    if overflow == "split" and not union_cap:
        raise ValueError("overflow='split' needs a positive union cap")
    dep = deploy_n_est or n_est
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    t0 = time.time()
    # 1. SCORE
    ti, tv, coverage, ci, n_sh = _score_attn(Xtr, ytr, Xte, cat, ptype, k_ret, n_est, device,
                                             buf_gb, call_idx, shard_size, torch)
    secs_score, gb_score = round(time.time() - t0, 1), _peak_gb(torch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    # 2. CLUSTER on the binary retrieval-indicator matrix (+ recursive split under overflow=split)
    K = min(_k_clusters(len(Xte)), len(Xte))
    ind, _ = _indicator(ti)
    lab, splits = _cluster_queries(ind, ti, K, overflow, union_cap, max_clusters)
    # 3+4. per-cluster context -> fit -> predict -> class-align -> stitch
    pred = np.zeros(len(Xte)) if ptype == "regression" else np.zeros((len(Xte), len(classes)))
    usz, n_capped, trunc_max = [], 0, 0.0
    for c in range(int(lab.max()) + 1):
        mask = lab == c
        if not mask.any():
            continue
        flat = ti[mask].ravel()                          # unique union of the members' top-k sets
        uniq, inv = np.unique(flat, return_inverse=True)
        if union_cap and len(uniq) > union_cap:
            # truncate at the memory wall: keep the rows the cluster's queries retrieved with the
            # most total attention MASS (= vote count weighted by softmax mass). Under
            # overflow='split' no union exceeds the cap, so this branch never fires.
            w = tv[mask].ravel().astype(np.float64)
            score = np.bincount(inv, weights=w)
            sub = np.sort(uniq[np.argpartition(-score, union_cap - 1)[:union_cap]])
            n_capped += 1
            trunc_max = max(trunc_max, 1.0 - union_cap / len(uniq))
        else:
            sub = uniq
        usz.append(len(sub))
        mm = _model(ptype, cat, dep, device).fit(Xtr[sub], ytr[sub])
        pred[mask] = mm.predict(Xte[mask]) if ptype == "regression" else _align(mm, Xte[mask], classes)
        del mm
    return dict(status="ok", k_clusters=K, **_scores(ptype, yte, pred, classes),
                secs=round(time.time() - t0, 1), peak_gb=max(gb_score, _peak_gb(torch)),
                retrieval_len=k_ret, call_idx=ci, coverage=round(coverage, 4) if coverage == coverage else np.nan,
                union_med=int(np.median(usz)), union_max=int(max(usz)),
                secs_score=secs_score, gb_score=gb_score,
                n_shards=n_sh, n_est_deploy=dep, overflow=overflow, union_cap=union_cap,
                n_capped=n_capped, trunc_max=round(trunc_max, 4),
                n_leaf_clusters=len(usz), n_splits=len(splits))


FIELDS = ["dataset", "ptype", "n_avail", "n_test", "arm", "budget", "k_clusters", "seed", "status",
          "err", "auc", "acc", "secs", "peak_gb",
          "retrieval_len", "call_idx", "coverage", "union_med", "union_max", "secs_score", "gb_score",
          "n_shards", "n_est_deploy", "overflow", "union_cap", "n_capped", "trunc_max",
          "n_leaf_clusters", "n_splits"]


def run(datasets, n_avail, k_ret, seeds, arms, n_est, device, buf_gb, call_idx, out_path,
        shard_size=0, deploy_n_est=0, union_cap=4_000_000, overflow="truncate", max_clusters=64):
    import torch
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval=""); w.writeheader()
        for name in datasets:
            X, y, cat, ptype = load_large(name)
            classes = None if ptype == "regression" else sorted(np.unique(np.asarray(y)).tolist())
            print(f"[{name}] ptype={ptype} X={X.shape}", flush=True)
            for seed in seeds:
                Xpool, ypool, Xte, yte = rcl._split(X, y, seed)
                for N in sorted(n_avail):
                    if N > len(Xpool):
                        print(f"  [{name}] N={N} > pool {len(Xpool)} -- skipped", flush=True)
                        continue
                    Xtr, ytr = Xpool[:N], ypool[:N]
                    base = dict(dataset=name, ptype=ptype, n_avail=N, n_test=len(Xte), seed=seed)
                    if "full" in arms:
                        dep = deploy_n_est or n_est
                        r = _eval_full(Xtr, ytr, Xte, yte, cat, ptype, classes, dep, device, torch)
                        w.writerow(dict(**base, arm="full", budget=N, k_clusters=1,
                                        n_est_deploy=dep, **r)); fh.flush()
                        print(f"  [{name}] N={N} s{seed} full       err={r.get('err')} "
                              f"peak={r.get('peak_gb')}GB {r['secs']}s", flush=True)
                    if "modelaware" in arms:
                        try:
                            r = _eval_modelaware(Xtr, ytr, Xte, yte, cat, ptype, classes, k_ret,
                                                 n_est, device, torch, buf_gb, call_idx,
                                                 shard_size, deploy_n_est, union_cap, overflow,
                                                 max_clusters)
                        except (RuntimeError, MemoryError, _TabPFNOOM) as e:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            oom = "out of memory" in str(e).lower() or isinstance(e, (MemoryError, _TabPFNOOM))
                            r = dict(status="oom" if oom else "err", err=np.nan, auc=np.nan,
                                     acc=np.nan, secs=0, peak_gb=np.nan, overflow=overflow,
                                     n_shards=0, coverage=np.nan, union_med=0, union_max=0)
                        w.writerow(dict(**base, arm="modelaware", budget=k_ret, **r)); fh.flush()
                        print(f"  [{name}] N={N} s{seed} modelaware {r['status']} "
                              f"err={r.get('err')} cover={r.get('coverage')} "
                              f"union_med={r.get('union_med')} sh={r.get('n_shards')} "
                              f"leaves={r.get('n_leaf_clusters')} capped={r.get('n_capped')} "
                              f"peak={r.get('peak_gb')}GB {r['secs']}s "
                              f"(score {r.get('secs_score', 0)}s)", flush=True)
    print(f"DONE -> {out_path}", flush=True)


def _smoke_equivalence(k_ret=200, seed=0):
    """CPU logic check (downloads Covertype from OpenML): streaming top-k == full-map top-k; shard
    quotas/offsets; union-cap truncation; recursive split; run()/CSV path."""
    import torch
    rcl.TEST_N = 2000
    X, y, cat, ptype = load_large("Covertype")
    sub = np.random.default_rng(0).choice(len(X), 7000, replace=False)
    X, y = np.asarray(X[sub]), np.asarray(y[sub])
    classes = sorted(np.unique(y).tolist())
    Xpool, ypool, Xte, yte = rcl._split(X, y, seed)
    Xtr, ytr = Xpool[:4000], ypool[:4000]
    # 1. streaming == full map
    m = _model(ptype, cat, 1, "cpu").fit(Xtr, ytr)
    pf = m.predict_proba
    ci = _resolve_call_idx(m, -1)
    print(f"[smoke] last ICL call idx={ci}")
    full = host_probe.capture_attention(pf, Xte, len(Xtr), indices=[ci])[ci]
    ti, tv = host_probe.capture_attention_topk(pf, Xte, len(Xtr), k=k_ret, indices=[ci],
                                               buf_gb=0.05)[ci]
    ref = np.argsort(-full, axis=1)[:, :k_ret]
    jac = np.mean([len(np.intersect1d(ti[i], ref[i])) / k_ret for i in range(len(ti))])
    mass = np.abs(np.sort(-full, axis=1)[:, :k_ret].sum(1) + tv.sum(1)).max()
    print(f"[smoke] top-{k_ret} overlap vs full map: jaccard={jac:.6f}  max|mass diff|={mass:.2e}")
    assert jac > 0.999 and mass < 1e-4
    # 2. quota: proportional + exact-sum
    assert _quota(500, [2e6] * 5).tolist() == [100] * 5
    q = _quota(500, [2e6, 2e6, 2e6, 1.68e6]); assert q.sum() == 500 and q[3] < q[0]
    # 3. sharded scoring: offsets, per-shard quota counts, shapes (2 blocks: 2500 + 1500)
    ti3, tv3, cov3, _, ns3 = _score_attn(Xtr, ytr, Xte, cat, ptype, k_ret, 1, "cpu", 0.05, ci,
                                         2500, torch)
    q3 = _quota(k_ret, [2500, 1500])
    assert ns3 == 2 and ti3.shape == (len(Xte), k_ret) and tv3.shape == ti3.shape
    assert all(len(np.unique(r)) == k_ret for r in ti3[:50])
    n_lo = (ti3 < 2500).sum(1)
    assert (n_lo == q3[0]).all() and ((ti3 >= 0) & (ti3 < 4000)).all()
    print(f"[smoke] shards: quotas={q3.tolist()} per-query low-block count OK cover={cov3:.3f}")
    # 4. union cap, truncate: mass-ranked truncation bounds every refit context
    rc = _eval_modelaware(Xtr, ytr, Xte, yte, cat, ptype, classes, k_ret, 1, "cpu", torch, 0.05, ci,
                          union_cap=1000)
    assert rc["status"] == "ok" and rc["union_max"] <= 1000 and rc["n_capped"] >= 1
    print(f"[smoke] truncate: n_capped={rc['n_capped']} union_max={rc['union_max']} "
          f"trunc_max={rc['trunc_max']} err={rc['err']}")
    # 5. union cap, split: cap one below the largest uncapped union -> >= 1 split, nothing truncated
    ru = _eval_modelaware(Xtr, ytr, Xte, yte, cat, ptype, classes, k_ret, 1, "cpu", torch, 0.05, ci,
                          union_cap=0)
    cap = ru["union_max"] - 1
    rs = _eval_modelaware(Xtr, ytr, Xte, yte, cat, ptype, classes, k_ret, 1, "cpu", torch, 0.05, ci,
                          union_cap=cap, overflow="split", max_clusters=256)
    assert rs["status"] == "ok" and rs["n_splits"] >= 1 and rs["n_capped"] == 0
    assert rs["union_max"] <= cap and rs["n_leaf_clusters"] > ru["n_leaf_clusters"]
    print(f"[smoke] split: cap={cap} splits={rs['n_splits']} leaves={rs['n_leaf_clusters']} "
          f"union_max={rs['union_max']} err={rs['err']} (uncapped {ru['err']})")
    # 6. run()/CSV path: both arms through the DictWriter (sharded scoring)
    import tempfile
    out = Path(tempfile.mkdtemp()) / "smoke_run.csv"
    run(["Covertype"], [5000], k_ret, [0], ["full", "modelaware"], 1, "cpu", 0.05, -1, out, 2500)
    rows = list(csv.DictReader(open(out)))
    assert [r["arm"] for r in rows] == ["full", "modelaware"], rows
    assert rows[1]["n_shards"] == "2" and rows[1]["status"] == "ok" and rows[0]["status"] == "ok"
    print(f"[smoke] run()/CSV: 2 arms written, modelaware n_shards=2 OK")
    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["Covertype"],
                    help="Covertype, Poker (OpenML) or a prepared pool, e.g. Higgs_full, Higgs_3M")
    ap.add_argument("--n-avail", type=int, nargs="+", default=[100_000],
                    help="source-pool sizes N (first N rows of each seed's shuffled pool)")
    ap.add_argument("--retrieval-len", type=int, default=500, help="per-query top-r (paper: 500)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", nargs="+", default=["full", "modelaware"],
                    choices=["full", "modelaware"])
    ap.add_argument("--shard-size", type=int, default=0,
                    help="0 = single scoring fit; else fixed block size (remainder = own block); "
                         "paper: 2000000 for pools > 4M")
    ap.add_argument("--union-size", "--union-cap", dest="union_cap", type=int, default=4_000_000,
                    help="per-cluster union capacity C (0 = uncapped); paper: 4000000")
    ap.add_argument("--overflow", choices=OVERFLOW, default="truncate",
                    help="what to do with a cluster whose union exceeds C: keep the C rows with the "
                         "largest summed attention (truncate) or recursively bisect it (split)")
    ap.add_argument("--max-clusters", type=int, default=64,
                    help="overflow=split: maximum number of clusters after splitting")
    ap.add_argument("--n-est", type=int, default=1, help="views for the scoring fits (paper: 1)")
    ap.add_argument("--deploy-n-est", type=int, default=0,
                    help="0 = same as --n-est; else n_estimators for DEPLOY fits only "
                         "(feed-all arm + per-cluster refits; scoring stays --n-est)")
    ap.add_argument("--buf-gb", type=float, default=8.0, help="GPU scratch for the scoring pass")
    ap.add_argument("--call-idx", type=int, default=-1, help="-1 = last ICL layer")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="cells CSV (default $TTC_RUNS_DIR/row_modelaware/cells.csv)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        _smoke_equivalence(); return
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = a.out or rcl._runs_dir() / "row_modelaware" / "cells.csv"
    print(f"[row_modelaware] device={device} datasets={a.datasets} N={a.n_avail} "
          f"k={a.retrieval_len} arms={a.arms} shard={a.shard_size} union_cap={a.union_cap} "
          f"overflow={a.overflow} deploy_n_est={a.deploy_n_est or a.n_est}", flush=True)
    run(a.datasets, a.n_avail, a.retrieval_len, a.seeds, a.arms, a.n_est, device, a.buf_gb,
        a.call_idx, out, a.shard_size, a.deploy_n_est, a.union_cap, a.overflow, a.max_clusters)


if __name__ == "__main__":
    main()
