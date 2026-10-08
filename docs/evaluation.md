# Evaluation protocol

All numbers in the paper come from TabArena per-cell results (`results_per_split.csv`, one row
per dataset x fold x method) and two computations implemented in `ttc.eval`.

## Benchmark

TabArena: 51 datasets, 816 cells (dataset x repeat x fold; TabArena's `fold` column enumerates
repeat x fold). Cell error `metric_error` is 1 - AUC (binary), log loss (multiclass) or RMSE
(regression). Cells a method did not produce are imputed with `RF (default)`, as TabArena does.

## Elo (`ttc.eval.elo`)

* **Fixed reference field, 68 entries:** the 67 official TabArena entries plus our frozen TabFM
  reproduction (run `repro_tabfm_bf16-cellcap`, entry `Frozen_TabFM_reference`).
* **One candidate per fit:** each candidate is inserted alone (69 entries). Candidates are never
  fit together, so adding an arm never moves another arm's number.
* **Solver:** TabArena's Bradley-Terry MLE (`bencheval.elo_utils.EloHelper.compute_mle_elo`) on
  pairwise per-cell wins/losses/ties, each cell weighted by 1/C_d (C_d = cells of dataset d),
  calibrated so `RF (default)` = 1000. Weighted pair totals are passed as one battle row per
  outcome, which is the exact sufficient statistic of TabArena's per-cell battles (tested).
* **Delta Elo:** Elo(candidate) - Elo(baseline) within the same fit, computed unrounded and
  rounded once. Baselines: `TA-TABPFN-3 (default)` (TabPFN-3 arms), `TABICLV2 (default)`
  (TabICL v2 arms), `Frozen_TabFM_reference` (TabFM arms).
* **Normalized score:** TabArena's: per dataset, mean cell error rescaled to
  1 - clip((e - best)/(median - best), 0, 1) over the 69 entries, averaged over datasets.

```bash
python -m ttc.eval.elo \
    --candidate hf_results/raw/tabpfn3_diagscale/results_per_split.csv \
    --reference-field hf_results/raw/tabfm_frozen/results_per_split.csv \
    --baseline "TA-TABPFN-3 (default)"
# baseline: TA-TABPFN-3 (default)
#  name  elo          baseline_elo  delta_elo  delta_rounded  score     n_entries  reference_source
#  ...   1681.940184  1656.275317   25.664867  26             0.698254  69         tabarena
```

Repeat `--candidate` (and optionally `--name`) to evaluate several arms; each is fit separately.
`--show Frozen_TabFM_reference` also prints another entry's rating in each fit (the paper reads
the frozen TabFM rating from the TabFM full-FT fit). `--json` prints one JSON object per candidate.

**Where the 67 references come from** (`--references`, default `auto` = first available):

| value | source |
|---|---|
| `field` | the `--reference-field` CSV, if it is a full TabArena board (as in the paper) |
| `candidate` | the candidate CSV, if it is a full TabArena board (a `compare` run embeds them) |
| `tabarena` | TabArena itself: `TabArenaContext().load_results_paper()` (downloaded once into TabArena's cache), RF-imputed with `tabarena.nips2025_utils.compare.prepare_data` |
| a path | any TabArena long-format CSV/parquet containing the 67 entries |

All routes give bit-identical reference matrices; when several are present they are checked
against each other. The released results contain only our rows, so they use the `tabarena` route.

API:

```python
from ttc.eval.elo import evaluate_candidate
r = evaluate_candidate("cand.csv", "frozen_tabfm.csv", baseline="TABICLV2 (default)")
r.elo, r.delta_elo, r.delta_rounded, r.score, r.board  # board = Elo + score of all 69 entries
```

Lower-level pieces: `error_matrix`, `reference_matrix`, `build_field`, `fit_elo`,
`normalized_scores`.

## Paired error change (`ttc.eval.paired`)

For candidate m and baseline b on the same cells, l_d = mean over the cells of dataset d of
log(e_m / e_b). Reported: 100 (exp(mean_d l_d) - 1) %, a 95% percentile bootstrap interval over
datasets (10,000 resamples, `numpy.random.default_rng(0)`), and wins = #datasets with l_d < 0.

```bash
python -m ttc.eval.paired \
    --candidate hf_results/raw/tabfm_diagscale/results_per_split.csv \
    --baseline  hf_results/raw/tabfm_full_ft/results_per_split.csv
# error change +0.102%  95% CI [-0.054%, +0.286%]  wins 28/51  (816 cells)
```

`--baseline-method "TA-TABPFN-3 (default)"` compares against a reference row of a full board
(`--baseline` then defaults to the candidate CSV).

## Reproducing the paper

`tests/test_eval_elo.py` checks the 12 headline arms (Elo, Delta Elo to 1e-6, normalized score)
and the DiagScale-vs-full-FT paired table on all three backbones:

```bash
TTC_HF_RESULTS_DIR=hf_results pytest tests/test_eval_elo.py
```

`TTC_HF_RESULTS_DIR` is the downloaded Hugging Face folder (`hf_results/raw/<arm>/results_per_split.csv`,
candidate rows only; TabArena must be installed to supply the references).
`TTC_PAPER_RESULTS_DIR` instead points at `<run tag>/results_per_split.csv` of full TabArena boards
from your own runs. Tests skip when the data are absent.

## Releasing results

`scripts/export_hf_results.py` writes the Hugging Face dataset folder: `data/results.parquet` (all
12 arms, one row per arm x cell), `data/arms.parquet` and `data/arms.csv` (one row per arm: config,
Elo, Delta Elo, GPU hours, sha256), `raw/<arm>/results_per_split.csv` (the arm's candidate rows,
copied byte-for-byte; `raw/tabfm_frozen` is the 68th reference) and the `README.md` dataset card.
It does not upload.
