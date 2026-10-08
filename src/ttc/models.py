"""Per-task test-time finetuning wrappers (full finetuning and DiagScale) for three backbones.

Each class is an AutoGluon/TabArena model, resolved by ``ttc.registry``. Every hyperparameter
prefixed ``ft_`` drives the finetuning loop and is stripped before the remaining keyword arguments
reach the backbone's estimator; ``ft_epochs: 0`` reproduces the frozen backbone exactly.

Finetuning requires the patched backbones (see ``patches/``): ``tabpfn.finetuning.peft``,
``tabicl._model.peft`` and ``tabfm.src.pytorch.peft``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from tabarena.models import TabPFN3Model
from tabarena.models.tabicl.model import TabICLv2Model

from ttc.backbones.tabfm import TabFMModel


def _current_task_context() -> tuple[str | None, int | None, int | None]:
    """Recover ``(dataset name, fold, repeat)`` from the tabarena ExperimentRunner frame on the
    call stack.

    AutoGluon hands the model only X/y; the task identity — including the CV ``fold``/``repeat``
    — lives in the runner that called us (`ExperimentRunner.task_name/fold/repeat`). Logging
    fold/repeat per fit lets downstream analysis join fits to result cells on the exact split
    instead of fragile file order (which a requeue can duplicate). Read-only stack walk; returns
    ``(None, None, None)`` outside a benchmark run.
    """
    import inspect

    try:
        for fi in inspect.stack():
            obj = fi.frame.f_locals.get("self")
            name = getattr(obj, "task_name", None)
            if isinstance(name, str):
                return name, getattr(obj, "fold", None), getattr(obj, "repeat", None)
    except Exception:
        pass
    return None, None, None


def _single_lr(ft: dict, default: float) -> float:
    """The finetuning learning rate: ``ft_lr``, or a one-element ``ft_lr_grid``.

    The paper uses one learning rate per backbone and method, shared across datasets; per-cell
    learning-rate selection is not part of the released protocol.
    """
    grid = ft.get("ft_lr_grid")
    if grid is not None:
        if len(grid) != 1:
            raise ValueError(f"ft_lr_grid must contain exactly one learning rate, got {grid}")
        return float(grid[0])
    return float(ft.get("ft_lr", default))


class FinetunedTabPFN3Model(TabPFN3Model):
    """TabPFN-3 with per-task test-time finetuning through ``tabpfn.finetuning``.

    Hyperparameters:
        ft_epochs       maximum number of epochs (0 -> the frozen TabPFN-3 fit, unchanged)
        ft_mode         full | diagscale | lora | regex | bitfit  (see tabpfn.finetuning.peft)
        ft_lr           learning rate
        ft_regex        module-name pattern (required for regex; optional narrowing for diagscale/lora)
        ft_time_limit   optional time cap in seconds for the finetuning loop
        ft_log_dir      set by the runner; each fit appends a JSON line (dataset, fold, repeat,
                        epochs_run, best_epoch, ...) to <ft_log_dir>/ft_fits.jsonl

    The finetuner holds out a validation split, selects the best epoch on it, and falls back to the
    pretrained weights when no epoch improves on them (``best_epoch == -1`` in the log).
    """

    ag_key = "TTC-FT-TABPFN3"
    ag_name = "TTC-FT-TabPFN3"

    def _get_ft_params(self) -> dict:
        return {k: v for k, v in super()._get_model_params().items() if k.startswith("ft_")}

    def _get_model_params(self) -> dict:
        # ft_* params drive the finetuning loop only — never leak them into the underlying
        # TabPFN estimator kwargs (the stock fallback path relies on this too).
        return {k: v for k, v in super()._get_model_params().items() if not k.startswith("ft_")}

    def _fit(self, X, y, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        ft = self._get_ft_params()
        epochs = int(ft.get("ft_epochs", 10))
        if epochs <= 0:
            # identity rung: bit-exact stock TabPFN-3
            return super()._fit(X=X, y=y, num_cpus=num_cpus, num_gpus=num_gpus, **kwargs)

        X = self.preprocess(X, y=y, is_train=True)

        from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor

        is_classification = self.problem_type in ["binary", "multiclass"]
        device = self._resolve_tabpfn_device(num_gpus=num_gpus)
        if not isinstance(device, str):
            device = device[0]  # the finetuning loop expects a single device string
        extra = {
            "model_path": self._get_model_checkpoint(),
            "categorical_features_indices": self._categorical_indices,
            "n_jobs": num_cpus,
        }
        ft_mode = ft.get("ft_mode", "full")
        if ft_mode == "lora":
            # ft_regex = target pattern over nn.Linear MODULE names; default ".*" = full-model LoRA.
            # None-safe coercion (a config may pass an explicit YAML null). seed ties adapter init to
            # random_state so LoRA runs are reproducible (see tabpfn.finetuning.peft.inject_lora).
            rank = ft.get("ft_lora_rank")
            alpha = ft.get("ft_lora_alpha")
            drop = ft.get("ft_lora_dropout")
            ft_cfg = {
                "pattern": str(ft.get("ft_regex", ".*")),
                "r": int(rank) if rank is not None else 8,
                "alpha": float(alpha) if alpha is not None else 16.0,
                "dropout": float(drop) if drop is not None else 0.0,
                "seed": int(self.fixed_random_state or 0),
            }
        elif ft_mode == "diagscale":
            # inject the diagonal metric on ALL SoftmaxScalingMLP modules by default ('.'); a config
            # may narrow it via ft_regex (e.g. the mechanism ablation). The fork's diagscale mode
            # requires cfg["pattern"], so this branch must run for diagscale even without ft_regex.
            ft_cfg = {"pattern": str(ft.get("ft_regex", "."))}
        elif "ft_regex" in ft:
            ft_cfg = {"pattern": str(ft["ft_regex"])}
        else:
            ft_cfg = None
        common = {
            "device": device,
            "epochs": epochs,
            "random_state": self.fixed_random_state,
            "time_limit": ft.get("ft_time_limit", kwargs.get("time_limit")),
            "finetune_mode": ft_mode,
            "finetune_cfg": ft_cfg,
        }
        lr = _single_lr(ft, default=1e-5)
        log_dir = ft.get("ft_log_dir")
        dataset, fold, repeat = _current_task_context()
        dataset = dataset or "unknown"

        def _fit_one(lr_val):  # build + fit the official finetuner at one lr
            common_lr = {**common, "learning_rate": float(lr_val)}
            if is_classification:
                m = FinetunedTabPFNClassifier(**common_lr, extra_classifier_kwargs=extra)
            else:
                m = FinetunedTabPFNRegressor(**common_lr, extra_regressor_kwargs=extra)
            return m.fit(X=X, y=y)

        t0 = time.monotonic()
        self.model = _fit_one(lr)
        if log_dir:
            record = {
                "dataset": dataset,
                "fold": fold,
                "repeat": repeat,
                "ft_mode": common["finetune_mode"],
                "ft_lr": lr,
                "ft_epochs_cap": epochs,
                "problem_type": self.problem_type,
                "n_rows": int(len(X)),
                "n_features": int(X.shape[1]),
                "epochs_run": getattr(self.model, "epochs_run_", None),
                "best_epoch": getattr(self.model, "best_epoch_", None),
                "stopped_early": getattr(self.model, "stopped_early_", None),
                "fit_seconds": round(time.monotonic() - t0, 2),
            }
            path = Path(log_dir) / "ft_fits.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")

    # --- device management: after a finetuned fit, self.model is the finetuning wrapper,
    #     which holds the actual TabPFN estimator as `finetuned_estimator_` ---
    def _inner_estimator(self):
        return getattr(self.model, "finetuned_estimator_", None) or self.model

    def get_device(self) -> str:
        base = self._inner_estimator()
        if hasattr(base, "devices_"):
            return base.devices_[0].type
        device = base.device
        if isinstance(device, str):
            return device.split(":")[0]
        from collections.abc import Sequence

        if isinstance(device, Sequence):
            return device[0]
        return device

    def _set_device(self, device: str) -> None:
        self._inner_estimator().to(device)


class FinetunedTabFMModel(TabFMModel):
    """Google TabFM with per-task test-time finetuning (diagscale / (IA)3 / full / regex).

    Extends the zero-shot ``TabFMModel``: ``_fit`` finetunes the backbone on the task's training
    data (``ttc.backbones.tabfm_finetune`` — the ported, validated sandbox loop) and deploys the
    finetuned backbone through the normal TabFM estimator path (so ``predict`` / device management
    are inherited unchanged). Requires the patched ``tabfm`` (``tabfm.src.pytorch.peft``).

    Same ``ft_*`` hyperparameter convention as ``FinetunedTabPFN3Model`` (stripped before any kwarg
    reaches the TabFM estimator):
        ft_epochs      gradient epochs (0 -> EXACT stock ``TabFMModel`` fit; the identity rung)
        ft_mode        full | diagscale | ia3 | regex   (default diagscale)
        ft_lr          learning rate (default 1e-2 for adapters, 1e-5 for full)
        ft_wd          AdamW weight decay (default 0.01)
        ft_n_est_finetune  views per finetuning step (default 2)
        ft_regex       module-name pattern: the injected-attention set for diagscale/ia3 (default
                       "." = all 42 attention modules), or the trainable-param set for regex
        ft_ia3_parts   ia3 only: subset of ("k","v","ff") to inject (default all three)
        ft_patience    early-stop patience on the val split (default 8)
        ft_ctx_qry_cap context+query rows per finetuning step (default 6144)
        ft_log_dir     injected by ttc.benchmarks (= the run's eval dir): append a JSON line per
                       fit to <ft_log_dir>/ft_fits.jsonl (dataset/fold/repeat/best_epoch/timing)

    Deployment is single-view (``n_estimators=1``) by default to match the single-view finetune;
    pass ``n_estimators`` via hyperparameters to ensemble at inference (the trained delta is
    view-agnostic — it lives inside attention, independent of view augmentation). Only the
    "default" preset is supported (the "ensemble" NNLS recipe is incompatible with single-view
    training).
    """

    ag_key = "TTC-FT-TABFM"
    ag_name = "TTC-FT-TabFM"

    # bf16 finetune (2026-07-24): the FT trains+deploys in bf16 (like the base), which PRESERVES the
    # delta (89-104% of the fp32 gain) — the old "bf16 quantizes the delta -> FT must be fp32" premise
    # is FALSE. So the DEPLOY now mirrors the base exactly: bf16, full context, and the SAME two-level
    # cell-budget FEATURE cap (fits at 500 -> 500 = Google, else fall back to _OOM_FALLBACK_FEATURES;
    # constants inherited from TabFMModel). This keeps base(ft_epochs=0)-vs-FT deploy IDENTICAL so the
    # delta is pure finetuning (not a cap mismatch). The caps below are TRAINING-only (val-predict +
    # gradient-episode memory), independent of the deploy. _FT_MAX_FEATURES_CAP kept only as a legacy
    # flat override via the `max_features_cap` hyperparameter.
    _FT_MAX_FEATURES_CAP = 100
    # Row cap for the val-predict + full-context deploy. The 32-view fp32 deploy scales with CONTEXT
    # ROWS (not cells): wide datasets cap at 2.4M/100 = 24k rows (fits ~64GB), but tall NARROW datasets
    # were only bounded by the 100k row_ceiling and OOM'd (GiveMeSomeCredit 10 feat -> 100k rows ->
    # 63+18.8 GB, job 12275). Lowered the ceiling to the same 24k the wide datasets already use, so
    # every dataset deploys at <=24k context rows; base gets the SAME cap -> base-vs-FT stays matched.
    _FT_CONTEXT_CELL_BUDGET = 2_400_000
    _FT_CONTEXT_ROW_CEILING = 24_000
    # Episode cap for the 2-VIEW gradient forward (the real 2-view train binding constraint, NOT the
    # row/val context). Calibrated (job 11352): on wide APSFailure ctx_qry_cap=6144 -> 80.7GB but
    # 4096 -> 60.3GB (below that the ~60GB val floor dominates). 400k cells = ~4k rows @ 100 feat;
    # thin datasets keep the full 6144 (400k/10 = 40k > 6144).
    _FT_EPISODE_CELL_BUDGET = 400_000
    # Absolute finetune-episode ROW cap. The 1.6B backbone OOMs the gradient FF block on tall NARROW
    # datasets where the cell-budget doesn't bind (GiveMeSomeCredit 10 feat: 400k/10 = 40k >> 6144, so
    # take stayed 6144 and OOM'd, job 12272). 2500 rows fits both arms (full-FT + diagscale), same cap
    # for both = fair. Applies UNIFORMLY on top of the cell-budget below.
    _FT_EPISODE_ROW_CAP = 2500

    def _get_ft_params(self) -> dict:
        return {k: v for k, v in super()._get_model_params().items() if k.startswith("ft_")}

    def _get_model_params(self) -> dict:
        # ft_* params drive the finetuning loop only — never leak into the TabFM estimator kwargs.
        return {k: v for k, v in super()._get_model_params().items() if not k.startswith("ft_")}

    # Chunk the deploy inference over TEST ROWS. The 32-view fp32 forward builds ~57 GB of activations
    # on a large test set (APSFailure 25k rows -> 82 GB OOM in predict; _fit itself ends at 6.6 GB, so
    # this is the predict, not a leak). Each chunk attends the SAME stored context, so concatenating is
    # bit-identical to a single forward. Override at the exec-model boundary (self.model.predict[_proba]).
    _FT_DEPLOY_TEST_CHUNK = 8000

    def _chunked(self, method_name, X):
        """Run a parent predict method over TEST-ROW chunks and concatenate (numpy- or pandas-aware).
        Each chunk attends the SAME stored context, so the result is bit-identical to one forward."""
        chunk = getattr(self, "_deploy_chunk", None) or int(
            self._get_ft_params().get("ft_deploy_test_chunk", self._FT_DEPLOY_TEST_CHUNK) or 0)
        parent = getattr(super(), method_name)
        if not chunk or len(X) <= chunk:
            return parent(X)
        parts = [parent(X.iloc[i:i + chunk]) for i in range(0, len(X), chunk)]
        import numpy as np
        import pandas as pd
        if isinstance(parts[0], np.ndarray):
            return np.concatenate(parts, axis=0)
        return pd.concat(parts)

    def _predict_proba(self, X):
        return self._chunked("_predict_proba", X)

    def _predict(self, X):
        return self._chunked("_predict", X)

    def _fit(self, X, y, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        # Reclaim the PREVIOUS cell's GPU memory before this cell allocates its fresh 1.6B deepcopy +
        # 32-view deploy. Without this, the finetuned model + deploy activations from the prior task
        # stay resident (~63 GB after a wide cell) and the next TALL cell's predict OOMs — verified
        # jobs 12278 (APSFailure->GiveMeSomeCredit OOMs) vs 12279 (GiveMeSomeCredit alone fits).
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # TF32 matmul (also set in tabfm_finetune._fresh_model) — set here too so it is active before
        # any op regardless of path, keeping this deploy at the same matmul precision as the base.
        torch.backends.cuda.matmul.allow_tf32 = True
        ft = self._get_ft_params()
        epochs = int(ft.get("ft_epochs", 30))
        if epochs <= 0:
            # identity rung: bit-exact stock TabFM (must reproduce the base 'tabfm' board)
            return super()._fit(X=X, y=y, num_cpus=num_cpus, num_gpus=num_gpus, **kwargs)

        X = self.preprocess(X, y=y)
        is_classification = self.problem_type in ["binary", "multiclass"]
        device = self._resolve_device(num_gpus)

        # DEPLOY feature cap = the base's two-level CELL-BUDGET cap (fits at 500 -> 500 = Google, else
        # fall back to 100), inherited from TabFMModel, so base(ft_epochs=0)-vs-FT deploy use IDENTICAL
        # features (the delta is pure finetuning, not a cap mismatch). The TRAINING caps below (context
        # row cap + episode cap) are separate bf16 memory bounds on the unchunked val-predict and the
        # 2-view gradient step; they don't touch the deploy.
        mp = self._get_model_params()
        flat_cap = int(mp.get("max_features_cap", 0) or 0)              # legacy unconditional flat override
        feat_budget = int(mp.get("feature_cell_budget", self._FEATURE_CELL_BUDGET) or 0)
        cell_budget = int(mp.get("context_cell_budget", self._FT_CONTEXT_CELL_BUDGET) or 0)
        row_ceiling = int(mp.get("context_row_ceiling", self._FT_CONTEXT_ROW_CEILING) or 0)
        n_rows, n_feats = int(X.shape[0]), max(int(X.shape[1]), 1)
        eff500 = min(n_feats, self._ESTIMATOR_MAX_FEATURES)            # the estimator's own 500 default
        if flat_cap and n_feats > flat_cap:
            est_feat_cap = flat_cap
        elif feat_budget and n_rows * eff500 > feat_budget:            # tall+wide -> safe 100 fallback
            est_feat_cap = min(self._OOM_FALLBACK_FEATURES, eff500)
        else:
            est_feat_cap = None                                        # short/narrow -> uncapped (=500)
        eff_feats = est_feat_cap or eff500                            # effective feat count for row/episode caps
        row_cap = cell_budget // eff_feats if cell_budget else 0
        row_cap = min(row_cap, row_ceiling) if (row_cap and row_ceiling) else (row_cap or row_ceiling)
        # feature-aware episode cap so the 2-view gradient forward fits on wide datasets
        episode_cap = int(ft.get("ft_ctx_qry_cap", self._FT_EPISODE_ROW_CAP))
        if self._FT_EPISODE_CELL_BUDGET:
            episode_cap = min(episode_cap, self._FT_EPISODE_CELL_BUDGET // eff_feats)

        ft_mode = str(ft.get("ft_mode", "diagscale"))
        pattern, peft_cfg = None, None
        if ft_mode in ("diagscale", "ia3"):
            pattern = str(ft.get("ft_regex", "."))
            if ft_mode == "ia3" and ft.get("ft_ia3_parts"):
                peft_cfg = {"parts": tuple(ft["ft_ia3_parts"])}
        elif ft_mode == "regex":
            if not ft.get("ft_regex"):
                raise ValueError("ft_mode='regex' requires ft_regex")
            pattern = str(ft["ft_regex"])
        elif ft_mode != "full":
            raise ValueError(f"unknown ft_mode {ft_mode!r} (full | diagscale | ia3 | regex)")
        lr = _single_lr(ft, default=1e-2 if ft_mode in ("diagscale", "ia3") else 1e-5)

        # Finetune on a row-capped context so the val / identity-guard FULL-context predict fits fp32
        # (the training forward/backward is already bounded by ctx_qry_cap). Feature cap via the est.
        Xft, yft = X, y
        if row_cap and n_rows > row_cap:
            idx = np.sort(np.random.default_rng(self.fixed_random_state).choice(
                n_rows, row_cap, replace=False))
            Xft = X.iloc[idx]
            yft = y.iloc[idx] if hasattr(y, "iloc") else np.asarray(y)[idx]
            print(f"[FinetunedTabFM] finetune context cap: {n_rows}x{n_feats} -> "
                  f"{row_cap}x{eff_feats} (bounds the unchunked val-predict)", flush=True)

        dataset, fold, repeat = _current_task_context()
        import torch

        from ttc.backbones.tabfm_finetune import finetune_tabfm

        def _ft(lr_val):  # one finetune at a given lr
            return finetune_tabfm(
                is_classification, Xft, yft, device, self.fixed_random_state,
                mode=ft_mode, pattern=pattern, lr=float(lr_val), epochs=epochs,
                patience=int(ft.get("ft_patience", 8)), ctx_qry_cap=episode_cap,
                weight_decay=float(ft.get("ft_wd", 0.01)), peft_cfg=peft_cfg,
                tag=f"{dataset or 'ft'} {ft_mode}@lr{lr_val:g}", max_num_features=est_feat_cap,
                n_est_finetune=int(ft.get("ft_n_est_finetune", 2)))

        t0 = time.monotonic()
        model_ft, meta = _ft(lr)
        if device == "cuda":
            torch.cuda.empty_cache()

        # FINAL estimator: FULL-train context with the finetuned (bf16) backbone. Deploy MIRRORS the
        # base — bf16, full context, cell-budget feature cap — so base(ft_epochs=0)-vs-FT is precision-
        # and feature-matched. default preset only.
        params = dict(self._get_model_params())
        preset = params.pop("preset", "default")
        if preset != "default":
            raise ValueError(f"FinetunedTabFMModel supports only preset='default' (got {preset!r})")
        for k in ("compute_dtype", "max_features_cap", "feature_cell_budget",
                  "context_cell_budget", "context_row_ceiling"):
            params.pop(k, None)  # control keys; the deploy feature cap is applied below as an est kwarg
        if est_feat_cap and "max_num_features" not in params:
            params["max_num_features"] = est_feat_cap
        # Deploy on FULL context, UNCAPPED rows — the bf16 backbone keeps full context on one 80GB GPU
        # (tall+wide tables already fell back to a 100-feature cap above, which fits full context in bf16),
        # exactly like the base. NO fp32 deploy row cap. bf16 = 2x the fp32 byte-headroom, so size the
        # (defensive, bit-identical) test-row chunk against 2x the cell limit.
        _bf16_budget = self._FP32_CELL_LIMIT * 2
        self._deploy_chunk = int(min(self._FT_DEPLOY_TEST_CHUNK,
                                     max(1000, _bf16_budget // eff_feats - n_rows)))
        params.setdefault(self.seed_name, self.fixed_random_state)
        params.setdefault("n_estimators", 32)  # TabFM official deploy = 32 augmented views
        params.setdefault("verbose", False)
        # Chunk the ENSEMBLE forward (batch_size = #views forwarded at once, NOT #views): fp32 deploy
        # OOMs when all 32 views go through at once on a large test set (job 12277: 18.8 GB predict_proba
        # alloc). Smaller batch_size => lower peak, IDENTICAL result (all 32 views still ensembled).
        bs = ft.get("ft_deploy_batch_size")
        if bs is not None:
            params["batch_size"] = int(bs)
        self.model = self._get_model_class()(model=model_ft, **params).fit(X=X, y=y)

        log_dir = ft.get("ft_log_dir")
        if log_dir:
            record = {
                "dataset": dataset or "unknown", "fold": fold, "repeat": repeat,
                "ft_mode": ft_mode, "ft_lr": meta["lr"], "ft_wd": meta["weight_decay"],
                "ft_epochs_cap": epochs, "problem_type": self.problem_type,
                "n_rows": int(len(X)), "n_features": int(X.shape[1]),
                "epochs_run": meta["epochs_run"], "best_epoch": meta["best_epoch"],
                "val0": meta["val0"], "best_val": meta["best_val"],
                "n_trainable": meta["n_trainable"], "param_drift": meta.get("param_drift"),
                "fit_seconds": round(time.monotonic() - t0, 2),
            }
            path = Path(log_dir) / "ft_fits.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")


class FinetunedTabICLModel(TabICLv2Model):
    """TabICL v2 with per-task test-time finetuning (diagscale / (IA)³ / full / regex).

    Extends the zero-shot ``TabICLv2Model``: ``_fit`` finetunes the backbone on the task's training
    data (``ttc.backbones.tabicl_finetune`` — the ported, validated sandbox loop) and deploys the
    finetuned backbone through an adopted TabICL estimator (so ``predict`` / device management are
    inherited unchanged). Requires the patched ``tabicl`` (``tabicl._model.peft``, including the
    (IA)³ KV-cache fix). Same ``ft_*`` convention as the other FT models:

        ft_epochs      gradient epochs (0 -> EXACT stock ``TabICLv2Model`` fit; the identity rung)
        ft_mode        full | diagscale | ia3 | regex   (default diagscale)
        ft_lr          learning rate (default 1e-2 for adapters, 1e-5 for full)
        ft_wd          AdamW weight decay (default 0.01)
        ft_regex       module-name pattern: injected-attention set for diagscale/ia3 (default "." =
                       all 21 attention modules), or the trainable-param set for regex
        ft_ia3_parts   ia3 only: subset of ("k","v","ff") to inject (default all three)
        ft_patience    early-stop patience on the val split (default 8)
        ft_ctx_qry_cap context+query rows per finetuning episode (default 10000 = official max)
        ft_deploy_n_estimators  inference views for the deployed estimator (default 1 = single-view,
                       matching the single-view finetune; the trained delta is view-agnostic)
        ft_log_dir     injected by ttc.benchmarks: append a JSON line per fit to ft_fits.jsonl

    To keep the finetune episodes and the deployed estimator on IDENTICAL numeric inputs, features
    are ordinal-encoded/imputed up front by ``NumericEncoder`` (mirrors tabicl's own X_encoder_) via
    a ``_preprocess`` override that is inert until ``_fit`` fits the encoder — so ft_epochs=0 stays
    bit-exact stock. Only classification/binary + regression are supported (v2 covers all three).
    """

    ag_key = "TTC-FT-TABICLv2"
    ag_name = "TTC-FT-TabICLv2"

    # Finetune-only GPU-memory cap. TabICL has NO per-view feature cap (unlike TabPFN-3's 500/view), so
    # it processes ALL features per view; the finetune BACKWARD then scales ~ ctx_rows x n_features x
    # n_views and OOMs on wide tables (Bioresponse 1776 feat: 2250x1776x2 = 8.0M cell-views -> 82 GB on
    # an 80 GB H100). Deploy/base do NOT OOM (inference keeps no backward activations), so we cap ONLY
    # the finetune episode ROWS as a function of width (like TabFM's _FT_EPISODE_CELL_BUDGET) — deploy
    # stays full-feature/full-row so base-vs-FT stays matched. Budget chosen so the widest ALREADY-CACHED
    # dataset (APSFailure, 170 feat) keeps its full 10k rows (170x10000x2 = 3.4M), i.e. only the 3 very
    # wide tables (Bioresponse/hiva/QSAR-TID-11, >=1024 feat) get capped. Calibrated on H100 (see below).
    _FT_EPISODE_CELL_BUDGET = 3_400_000

    def _get_ft_params(self) -> dict:
        return {k: v for k, v in super()._get_model_params().items() if k.startswith("ft_")}

    def _get_model_params(self) -> dict:
        return {k: v for k, v in super()._get_model_params().items() if not k.startswith("ft_")}

    def _preprocess(self, X, is_train: bool = False, **kwargs):
        # Standard tabicl preprocessing, then (finetune path only) numericize with the encoder fit
        # in _fit — so episodes and deploy inference see identical inputs. Inert (stock) until the
        # encoder exists, keeping the ft_epochs=0 identity rung bit-exact with base TabICLv2Model.
        X = super()._preprocess(X, is_train=is_train, **kwargs)
        enc = getattr(self, "_ft_encoder", None)
        return enc.transform(X) if enc is not None else X

    def _fit(self, X, y, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        ft = self._get_ft_params()
        epochs = int(ft.get("ft_epochs", 30))
        if epochs <= 0:
            # identity rung: bit-exact stock TabICLv2 (_ft_encoder stays unset -> _preprocess inert)
            return super()._fit(X=X, y=y, num_cpus=num_cpus, num_gpus=num_gpus, **kwargs)

        from torch.cuda import is_available

        device = "cuda" if (num_gpus and num_gpus != 0) else "cpu"
        if device == "cuda" and not is_available():
            raise AssertionError("Fit specified to use GPU, but CUDA is not available.")
        is_classification = self.problem_type in ["binary", "multiclass"]
        seed = int(self._get_model_params().get(self.seed_name) or 0)

        # standard preprocess (categoricals kept — _ft_encoder still unset) -> fit numeric encoder
        Xp = self.preprocess(X, y=y)
        from ttc.backbones.tabicl_finetune import (NumericEncoder, _make_est, _pristine_backbone,
                                                   finetune_tabicl)
        self._ft_encoder = NumericEncoder().fit(Xp)
        Xnum = self._ft_encoder.transform(Xp)

        ft_mode = str(ft.get("ft_mode", "diagscale"))
        pattern, peft_cfg = None, None
        if ft_mode in ("diagscale", "ia3"):
            pattern = str(ft.get("ft_regex", "."))
            if ft_mode == "ia3" and ft.get("ft_ia3_parts"):
                peft_cfg = {"parts": tuple(ft["ft_ia3_parts"])}
        elif ft_mode == "regex":
            if not ft.get("ft_regex"):
                raise ValueError("ft_mode='regex' requires ft_regex")
            pattern = str(ft["ft_regex"])
        elif ft_mode != "full":
            raise ValueError(f"unknown ft_mode {ft_mode!r} (full | diagscale | ia3 | regex)")
        lr = _single_lr(ft, default=1e-2 if ft_mode in ("diagscale", "ia3") else 1e-5)

        ckpt = self.get_checkpoint_version(self._get_model_params())
        pristine = _pristine_backbone(is_classification, ckpt)

        dataset, fold, repeat = _current_task_context()
        import torch

        # Width-aware finetune-episode row cap (GPU-memory bound; see _FT_EPISODE_CELL_BUDGET). Only the
        # very wide tables are reduced; deploy is untouched, so base-vs-FT stays matched.
        n_views = int(ft.get("ft_n_est_finetune", 2))
        base_cap = int(ft.get("ft_ctx_qry_cap", 10000))
        budget = int(ft.get("ft_episode_cell_budget", self._FT_EPISODE_CELL_BUDGET) or 0)
        ctx_cap = base_cap
        if budget:
            ctx_cap = max(256, min(base_cap, budget // (max(1, Xnum.shape[1]) * max(1, n_views))))
        if ctx_cap < base_cap:
            print(f"[tabicl-ft] {dataset}: {Xnum.shape[1]} feat -> episode row cap {base_cap}->{ctx_cap} "
                  f"(budget {budget} cell-views)", flush=True)

        def _ft(lr_val):  # one finetune at a given lr
            return finetune_tabicl(
                pristine, Xnum, y, is_classification, device, seed,
                mode=ft_mode, pattern=pattern, lr=float(lr_val), epochs=epochs,
                patience=int(ft.get("ft_patience", 8)),
                ctx_qry_cap=ctx_cap,
                weight_decay=float(ft.get("ft_wd", 0.01)), peft_cfg=peft_cfg,
                tag=f"{dataset or 'ft'} {ft_mode}@lr{lr_val:g}",
                n_est_finetune=int(ft.get("ft_n_est_finetune", 2)))

        t0 = time.monotonic()
        model_ft, meta = _ft(lr)

        # FINAL estimator: adopt the finetuned backbone; TabICL official deploy = 8 views.
        _model, cfg, path = pristine
        deploy_ne = int(ft.get("ft_deploy_n_estimators", 8))
        est = _make_est(model_ft, cfg, path, is_classification, seed, device, n_estimators=deploy_ne)
        self.model = est.fit(Xnum, np.asarray(y))

        log_dir = ft.get("ft_log_dir")
        if log_dir:
            record = {
                "dataset": dataset or "unknown", "fold": fold, "repeat": repeat,
                "ft_mode": ft_mode, "ft_lr": meta["lr"], "ft_wd": meta["weight_decay"],
                "ft_epochs_cap": epochs, "problem_type": self.problem_type,
                "n_rows": int(len(Xnum)), "n_features": int(Xnum.shape[1]),
                "epochs_run": meta["epochs_run"], "best_epoch": meta["best_epoch"],
                "val0": meta["val0"], "best_val": meta["best_val"],
                "n_trainable": meta["n_trainable"], "param_drift": meta.get("param_drift"),
                "fit_seconds": round(time.monotonic() - t0, 2),
            }
            path_j = Path(log_dir) / "ft_fits.jsonl"
            path_j.parent.mkdir(parents=True, exist_ok=True)
            with open(path_j, "a") as f:
                f.write(json.dumps(record) + "\n")
