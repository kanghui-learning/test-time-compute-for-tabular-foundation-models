# Large-table row retrieval (model-aware vs feed-all)

Frozen TabPFN-3, single view. Two arms on a fixed 50,000-row test set per seed:

* **feed-all** (`--arms full`): the whole source pool of N rows as context.
* **model-aware** (`--arms modelaware`): attention-guided retrieval. Each query's attention over
  the pool (last ICL layer, query-to-context softmax, mean over heads) gives its top r = 500 rows;
  queries are clustered by their binary retrieval indicators
  (G = min(n_Q, max(4, min(30, floor(n_Q/2000)))) = 25 KMeans clusters, random_state 0, n_init 4);
  each cluster predicts with the union of its queries' rows.

Beyond memory (feed-all fails at 4.0-4.5M rows on an 80 GB GPU), the shuffled pool is scored in
2M-row blocks (`--shard-size 2000000`) with proportional per-block quotas (largest-remainder
rounding) and within-block ranks. Per-cluster unions are capped at C = 4M rows
(`--union-size 4000000`, alias `--union-cap`); `--overflow` decides what happens above C:

* `truncate` (default): keep the C rows with the largest summed retained attention.
* `split`: recursively bisect the cluster (2-center KMeans on its queries' binary indicators,
  queries ordered by distance difference, balanced halves) until every union fits; at most
  `--max-clusters 64` clusters. Uses no labels. Paper: Higgs at 6M-10M.

Code: `ttc.benchmarks.row_modelaware` (pipeline + CLI), `ttc.benchmarks.row_coreset_large`
(data loading, per-seed split, feed-all arm), `ttc.benchmarks.row_pool_prep` (pool preparation),
`ttc.retrieval.host_probe` (streaming attention top-k).

## Environment

| Variable | Default | Used for |
|---|---|---|
| `TTC_DATA_DIR` | `./data/row_large` | prepared npy pools |
| `TTC_RUNS_DIR` | `./runs` | default output, `$TTC_RUNS_DIR/row_modelaware/cells.csv` |
| `OPENML_CACHE_DIR` | openml's default | OpenML downloads (read by openml itself) |
| `TABPFN_MODEL_CACHE_DIR` | tabpfn's default | TabPFN-3 checkpoint, downloaded by tabpfn on first use |

The TabPFN-3 checkpoint is tabpfn's default (`tabpfn-v3-classifier-v3_default.ckpt`); nothing in
these modules points to a local path. Run everything from the repository root with the package
installed (`pip install -e .`) or `PYTHONPATH=src`.

## 1. Prepare the pools (CPU)

Covertype (OpenML 1596) and Poker-Hand (OpenML 1567) are read from OpenML at run time. The three
tables beyond 1M rows are converted once:

```bash
# full tables: Higgs (OpenML 45570, 11M), COMET_MC (5889, 7.62M), US_Accidents (46650, 7.73M)
python -m ttc.benchmarks.row_pool_prep --datasets Higgs COMET_MC US_Accidents
# plus the 3M-row subsets used by the within-memory sweep (up to 2.9M rows)
python -m ttc.benchmarks.row_pool_prep --datasets Higgs COMET_MC US_Accidents \
    --skip-full --subset-rows 3000000
```

This writes `{Higgs,COMET_MC,US_Accidents}_full_{X,y}.npy` + `_meta.json` and
`{...}_3M_{X,y}.npy` + `_meta.json` into `$TTC_DATA_DIR`. The subset is
`full[np.random.default_rng(0).choice(N_full, 3_000_000, replace=False)]` in draw order.

No split files are written. For seed s, `row_coreset_large._split` permutes the loaded table with
`np.random.default_rng(s)`; the first 50,000 rows are the test set and a pool of size N is the
first N of the remaining rows. Each table family (full, 3M subset, OpenML) therefore has its own
seed-specific test set, as in the paper.

## 2. Smoke test (one GPU, minutes)

```bash
python -m ttc.benchmarks.row_modelaware --datasets Covertype --n-avail 20000 --seeds 0 \
    --arms full modelaware --out runs/row_modelaware/smoke.csv
# exercises the split policy (unions of 2,000 queries x 500 rows exceed a 15k cap)
python -m ttc.benchmarks.row_modelaware --datasets Covertype --n-avail 20000 --seeds 0 \
    --arms modelaware --union-size 15000 --overflow split --out runs/row_modelaware/smoke_split.csv
```

`python -m ttc.benchmarks.row_modelaware --smoke` runs the CPU logic checks (streaming top-k equals
the full attention map, block quotas and offsets, truncation, splitting, CSV output) on a
7,000-row Covertype sample. `tests/test_retrieval_split.py` checks the split policy alone (CPU).

## 3. Paper settings

All runs: seeds 0-4, `--n-est 1` (default), `--retrieval-len 500` (default),
`--union-size 4000000` (default; never binds within memory). Paper results come from separate
jobs per dataset and seed group; the commands below give the grid.

**Within memory, unsharded** ("Within memory" columns of the paper's row-construction table;
row-curation figures):

```bash
python -m ttc.benchmarks.row_modelaware --datasets Covertype --seeds 0 1 2 3 4 \
    --n-avail 100000 300000 500000 --arms full modelaware
python -m ttc.benchmarks.row_modelaware --datasets Poker --seeds 0 1 2 3 4 \
    --n-avail 100000 300000 500000 900000 --arms full modelaware
python -m ttc.benchmarks.row_modelaware --datasets Higgs_3M --seeds 0 1 2 3 4 \
    --n-avail 300000 500000 900000 1500000 2000000 2500000 2900000 --arms full modelaware
python -m ttc.benchmarks.row_modelaware --datasets US_Accidents_3M COMET_MC_3M --seeds 0 1 2 3 4 \
    --n-avail 300000 900000 1500000 2000000 2500000 2900000 --arms full modelaware
```

(The paper's model-aware Higgs subset sweep has no 2.5M point; feed-all does.)

**Full tables up to the wall, unsharded** (feed-all references at 2M and 4M):

```bash
python -m ttc.benchmarks.row_modelaware --datasets Higgs_full US_Accidents_full COMET_MC_full \
    --seeds 0 1 2 3 4 --n-avail 2000000 4000000 --arms full modelaware
```

**Two-block scoring control at 4M** (paper appendix "Reaching past the memory wall"):

```bash
python -m ttc.benchmarks.row_modelaware --datasets Higgs_full US_Accidents_full COMET_MC_full \
    --seeds 0 1 2 3 4 --n-avail 4000000 --arms modelaware --shard-size 2000000
```

**Beyond the wall** (sharded scoring, 2M blocks; feed-all is not run, its reference is 4M):

```bash
# Higgs: recursive split (paper main result) ...
python -m ttc.benchmarks.row_modelaware --datasets Higgs_full --seeds 0 1 2 3 4 \
    --n-avail 6000000 8000000 10000000 --arms modelaware \
    --shard-size 2000000 --union-size 4000000 --overflow split --max-clusters 64
# ... and the capped-union control (Higgs union-capacity table)
python -m ttc.benchmarks.row_modelaware --datasets Higgs_full --seeds 0 1 2 3 4 \
    --n-avail 6000000 8000000 10000000 --arms modelaware \
    --shard-size 2000000 --union-size 4000000 --overflow truncate
# US_Accidents (7,678,394 = whole pool) and COMET_MC (7,569,400 = whole pool): truncate
python -m ttc.benchmarks.row_modelaware --datasets US_Accidents_full --seeds 0 1 2 3 4 \
    --n-avail 6000000 7678394 --arms modelaware --shard-size 2000000 --union-size 4000000
python -m ttc.benchmarks.row_modelaware --datasets COMET_MC_full --seeds 0 1 2 3 4 \
    --n-avail 6000000 7569400 --arms modelaware --shard-size 2000000 --union-size 4000000
```

Relative log-loss reduction is `100 * (err_feedall - err_modelaware) / err_feedall`, paired by
dataset, N and seed; beyond the wall the reference is feed-all at 4M on the same full-table seed.

Output columns: `err` (log loss), `auc` (binary), `acc`, `secs`, `peak_gb`, `coverage` (captured
attention mass), `union_med`/`union_max` (per-cluster context sizes), `n_shards`, `n_capped`
and `trunc_max` (truncated clusters and largest truncated fraction), `n_leaf_clusters` and
`n_splits` (final clusters and splits under `--overflow split`). `k_clusters` is the initial
cluster count G.

## Notes on reproducibility

* **Split policy.** The paper's Higgs 6M-10M split runs used the same algorithm applied as a
  patch around the same pipeline (25 initial clusters, C = 4M, at most 64 clusters).
  `--overflow split` integrates it; `tests/test_retrieval_split.py` checks that cluster
  assignments are identical. Those runs report 26-27 final clusters. Seed 0 informed the
  development of the split rule.
* **3M subsets.** The subset builder reproduces the paper's Higgs and COMET_MC 3M subsets
  bit for bit. The paper's US_Accidents 3M subset was drawn by a different, unrecorded procedure,
  so US_Accidents results at 2.9M rows and below come from a different (equally random) subset.
* **Hardware.** Feed-all at 4M rows needs about 80 GB of GPU memory (about 20 GB per 1M rows).
  `--buf-gb` (default 8) bounds the scratch memory of the attention-scoring pass.
* Model-aware scoring and clustering see the whole unlabeled test batch at once, so contexts
  depend on which queries are evaluated together; test labels are used only for scoring.
