# Backbone patches

The experiments use small, additive patches on top of three public tabular foundation
model libraries. Each package has

- `patches/<pkg>/NNNN-*.patch`: a `git format-patch` series (commit messages and authorship
  kept; apply with `git am`), and
- `patches/<pkg>.diff`: the same change as one squashed diff (`git apply --3way`), and
- `patches/<pkg>/LICENSE`: the upstream license at the base commit (the patches are
  derivative works distributed under it; see `THIRD_PARTY_NOTICES.md`).

`scripts/install_forks.sh` clones each upstream at the base commit, applies the series,
**checks that the resulting source tree is bit-identical to the one used for the paper**
(git tree hash), and installs it with `pip install -e`.

| package | upstream | base commit | patch range | result tree (= paper fork commit) |
|---|---|---|---|---|
| `tabpfn` | https://github.com/PriorLabs/TabPFN | `95ff6772` = tag **v8.0.7** | 8 commits, `95ff6772..bf9b2c99` | `f2094982` (fork `bf9b2c99`) |
| `tabicl` | https://github.com/soda-inria/tabicl | `46b91961` on `main` (after v2.1.1, before v2.2.0) | 3 commits, `46b9196..efda645` | `2df5bffc` (fork `efda645`) |
| `tabfm` | https://github.com/google-research/tabfm | `2fb67e86` on `main` (before v1.0.1) | 3 commits, `2fb67e8..78beef7` | `7abb8157` (fork `78beef7`) |
| `tabarena` | https://github.com/autogluon/tabarena | `abd24c7f` on `main` | none (pinned only) | — |

All base commits are reachable from the public upstream `main` branch (checked against the
upstream remotes). Every patch only touches the files listed below; none changes default
inference behaviour of the unpatched library except where noted (tabfm bf16 path).

## tabpfn (Prior Labs TabPFN, v8.0.7 + 8 commits)

Files: `src/tabpfn/finetuning/{peft.py (new), finetuned_base.py, finetuned_classifier.py, finetuned_regressor.py}`

1. **PEFT parameter-selection hook.** `finetune_mode` / `finetune_cfg` constructor arguments on
   `FinetunedTabPFNClassifier/Regressor`; `finetuning/peft.py::select_trainable` chooses which
   parameters the optimizer trains (`full` = exact upstream behaviour, `bitfit`, `regex`).
2. **Epoch bookkeeping** exposed after `fit`: `epochs_run_`, `best_epoch_`, `stopped_early_`;
   later also `best_metric_`, `greater_is_better_` (used by the per-task learning-rate sweep).
3. **`auto_scale_n_estimators` disabled in the finetuning training forward only.** Upstream
   scales the ensemble up on wide tables, which breaks the finetuning loss's assertion that
   exactly `n_estimators_finetune` members are returned. Validation and final inference keep
   auto-scaling.
4. **Fall-back to base weights.** With early stopping on, if no epoch beats the
   pre-finetuning validation score (`best_epoch_ == -1`) the base weights are restored instead
   of shipping the last (degraded) epoch.
5. **LoRA mode** (`finetune_mode="lora"`): `LoRALinear` / `inject_lora`, zero-initialised `B`,
   seeded adapter init (reproducible under a fixed `random_state`), guard rails for `r<=0`.
6. **DiagScale mode** (`finetune_mode="diagscale"`): `DiagScale` / `inject_diagscale` wrap the
   attention `SoftmaxScalingMLP` and multiply its output query by a trainable zero-initialised
   per-(head, dim) factor `(1 + delta)`, i.e. a learned diagonal similarity metric.

Excluded on purpose: fork commit `2102d854` (opt-in step-granular validation) and
uncommitted research edits; they are not used by any released result.

License: Prior Labs License v1.2 (Apache-2.0 + additional attribution provision, see
`THIRD_PARTY_NOTICES.md`). The TabPFN-3 model weights are under the separate
non-commercial TABPFN-3 License v1.0 and are downloaded by `tabpfn` itself; we do not
redistribute them.

## tabicl (TabICL v2, 46b9196 + 3 commits)

Files: `src/tabicl/_model/{attention.py, layers.py, peft.py (new)}`, `tests/test_peft.py (new)`

1. **DiagScale and (IA)^3 adapters.** `MultiheadAttention` gains `diag_delta`,
   `ia3_delta_k`, `ia3_delta_v` slots (plain `None` attributes, so released checkpoints still
   load with `strict=True`); `MultiheadAttentionBlock` gains `ia3_delta_ff` on the FFN
   activation. `_model/peft.py` provides `inject_diagscale`, `inject_ia3`, `select_trainable`
   (`full | regex | diagscale | ia3`). Bit-exact identity at `delta = 0`.
2. **`freeze_others` flag** on `select_trainable` (flipping `requires_grad` changes the CUDA
   train-mode forward by ~5e-3; `freeze_others=False` keeps flag parity).
3. **KV-cache fix:** the cache now stores the pre-(IA)^3 K/V (previously the rescale was
   applied twice on cache reuse); `select_trainable` hardening. 12 unit tests.

License: BSD-3-Clause (Soda team @ Inria). The TabICL checkpoints (`jingang/TabICL` on the
Hugging Face Hub, fetched by the library) are also tagged BSD-3-Clause.

## tabfm (Google TabFM, 2fb67e8 + 3 commits)

Files: `tabfm/src/classifier_and_regressor.py`, `tabfm/src/pytorch/{model.py, peft.py (new), peft_test.py (new)}`

1. **bf16 inference.** The PyTorch predict step casts inputs to the model's parameter dtype
   instead of hard-coded float32 (fp32 models unchanged) and casts logits back to fp32 before
   NumPy.
2. **DiagScale adapter:** zero-initialised `delta[nhead, head_dim]` multiplying the final query
   (after RoPE / q-RMSNorm / per-dim scale), `select_trainable(full | regex | diagscale)`.
3. **(IA)^3 baseline mode:** `1 + delta` on keys / values / FFN activation.

Excluded on purpose: fork commit `bdbe4ae` (per-head temperature adapter), not used by any
released result.

License: code Apache-2.0 (Google LLC). The TabFM model weights
(`google/tabfm-1.0.0-pytorch` on Hugging Face) are under the separate **TabFM Non-Commercial
License v1.0**, which forbids distributing the model or any Derivative (including fine-tuned
weights); we do not redistribute weights or fine-tuned adapters.

## tabarena (autogluon/tabarena @ abd24c7, no patch)

Only `packages/bencheval` and `packages/tabarena` are installed (editable). tabarena requires
`autogluon>=1.5,<1.6`. License: Apache-2.0.

## Regenerating the patches (maintainers)

```bash
git -C <tabpfn-fork> format-patch 95ff6772..bf9b2c99 -o patches/tabpfn
git -C <tabpfn-fork> diff 95ff6772 bf9b2c99 > patches/tabpfn.diff
git -C <tabicl-fork> format-patch 46b9196..efda645 -o patches/tabicl
git -C <tabicl-fork> diff 46b9196 efda645 > patches/tabicl.diff
git -C <tabfm-fork>  format-patch 2fb67e8..78beef7 -o patches/tabfm
git -C <tabfm-fork>  diff 2fb67e8 78beef7 > patches/tabfm.diff
```
