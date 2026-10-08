"""TabFMModel — AutoGluon/TabArena wrapper for Google's TabFM (PyTorch backend).

Registry key ``tabfm`` (see ttc.registry): the default preset ``TabFMClassifier(model)`` /
``TabFMRegressor(model)`` == Google's "TabFM" leaderboard entry (32 augmented views, uniform blend).

Mirrors ``tabarena.models.TabPFN3Model`` / ``TabICLModel``: subclass ``AbstractTorchModel`` and, in
``_fit``, preprocess -> build the sklearn-compatible TabFM estimator -> ``.fit``. Prediction is
inherited (``AbstractModel._predict_proba`` calls ``self.model.predict_proba`` for classification
and ``.predict`` for regression), exactly as the TabPFN-3 and TabICL wrappers do.

TabFM ships a PyTorch backend, so we stay entirely inside this repo's torch stack — no JAX. The
backbone weights download from Hugging Face (``google/tabfm-1.0.0-pytorch`` at the pinned
``TABFM_REVISION``) into ``HF_HOME`` on first use; ``tabfm_v1_0_0_pytorch.load`` caches them per
(model_type, device) in-process, so the folds of one task load the backbone once.

Control hyperparameters (consumed here, never forwarded to the TabFM estimator):
    preset   "default" (single recipe) | "ensemble" (Google's heavier ensemble + calibration)
Every other hyperparameter is forwarded verbatim to ``TabFMClassifier`` / ``TabFMRegressor``
(``n_estimators``, ``norm_methods``, ``softmax_temperature``, ``nnls_beta``, feature-cross / SVD
schedules, ...), so the config-tune / views experiments can drive TabFM later with no change here.
Note: the default preset already ensembles 32 augmented views internally — for an apples-to-apples
"single-view base" (to measure our own views axis without double-counting), pass ``n_estimators: 1``
via the config's ``hyperparameters`` (plus a ``tag_suffix``).

License note: the TabFM weights are under a non-commercial license — research / benchmark use only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


TABFM_REPO = "google/tabfm-1.0.0-pytorch"
# Hugging Face revision whose weights the paper used. It ships ``pytorch_model.bin``, the format the
# pinned TabFM loader reads; later revisions of the same repository ship ``model.safetensors``.
TABFM_REVISION = "4315364dbb949483559be57f97284d487b03890f"


def resolve_tabfm_checkpoint() -> str:
    """Local TabFM checkpoint dir (with ``classification/`` and ``regression/`` heads) for the loader.

    ``TABFM_CHECKPOINT_DIR`` overrides; otherwise the pinned revision is downloaded into the Hugging
    Face cache on first use (``HF_HOME`` sets the cache). Base and finetuning paths share this, so
    they always load identical weights.
    """
    import os

    env = os.environ.get("TABFM_CHECKPOINT_DIR")
    if env:
        return env
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=TABFM_REPO, revision=TABFM_REVISION)
    missing = [h for h in ("classification", "regression")
               if not os.path.exists(os.path.join(path, h, "pytorch_model.bin"))]
    if missing:
        raise FileNotFoundError(f"TabFM snapshot {path} lacks pytorch_model.bin for {missing}")
    return path


class TabFMModel(AbstractTorchModel):
    """Google TabFM (zero-shot tabular foundation model) — TabArena integration, PyTorch backend."""

    ag_key = "TA-TABFM"
    ag_name = "TA-TabFM"
    ag_priority = 100
    seed_name = "random_state"

    fixed_random_state: int = 0
    """Fixed seed for reproducibility (matches the TabPFN-3 wrapper convention)."""

    # Row-cap safety net — DISABLED by default. bf16 (see `_load_backbone`) is the real memory fix;
    # capping context rows barely helped on WIDE data because the OOM is feature-dominated (the
    # categorical Fourier embedding over 170+ features), not row-dominated. Re-enable by setting the
    # `context_cell_budget` (>0) / `context_row_ceiling` hyperparameters: caps CONTEXT so
    # rows*features stays under the cell budget and rows under the ceiling. Default preset only —
    # the ensemble preset's enable_nnls is mutually exclusive with max_num_rows.
    _CONTEXT_CELL_BUDGET = 0
    _CONTEXT_ROW_CEILING = 40_000

    # Feature cap — a TWO-LEVEL cap decided by CONTEXT CELLS (train_rows x features), not features
    # alone. Single-GPU OOM is driven by the categorical Fourier embedding over rows*features, so a
    # wide-but-SHORT table (few rows) fits its full features on one 80GB H100 while a tall-wide one
    # does not (measured 2026-07-24, bf16 32-view: MIC 111f x 1.5k rows -> 6.6GB, Bioresponse 1776f x
    # 3.3k = 4.4M -> 20GB, QSAR 1024f x 5.1k = 5.2M -> 29GB all FIT at 500; kddcup 212f x 33k = 7.1M
    # and APSFailure 170f x 51k = 8.6M both OOM). Rule: if the estimator's own 500-feature cap fits
    # the cell budget -> use 500 (= Google's config); else fall back to a safe 100-feature cap (proven
    # to fit on the 1750 board: APSFailure/kddcup/Amazon at 100 = 3-5M cells). The earlier UNCONDITIONAL
    # flat `>100 -> 100` cap wrongly bit short-wide tables too (QSAR-TID-11 was our single worst cell vs
    # Google, +9.4% rel error, purely from 1024f -> 100 when 500 fit in 29GB); this version leaves them
    # at 500 and only the genuinely OOM-ing tall+wide tables fall back to 100 — matching the corrected
    # board exactly (no re-run needed). Tune with `feature_cell_budget` (0 disables); `max_features_cap`
    # (>0) still forces an unconditional flat cap instead. Applies to BOTH presets.
    _FEATURE_CELL_BUDGET = 5_000_000   # max cells (rows x <=500 feat) that fit one 80GB H100 (proven ~4.4M ok / 7.1M OOM)
    _ESTIMATOR_MAX_FEATURES = 500      # the TabFM estimator's own default max_num_features (= Google)
    _OOM_FALLBACK_FEATURES = 100       # safe cap for tables that OOM even at 500 (1750-board-proven to fit)

    # fp32 full-context deploy OOMs above this many context-cells (rows*features, after the feature
    # cap) on one 80GB GPU — measured: 24000x100=2.4M and kddcup 33333x100=3.3M FIT, APSFailure
    # 50666x100=5.1M OOMs. The base (no delta) drops to bf16 above this so it KEEPS full context
    # (the official protocol is full-context); the fp32 finetune path (bf16 would quantize the delta)
    # caps the context instead. Only APSFailure crosses it in TabArena.
    _FP32_CELL_LIMIT = 3_500_000

    def _get_model_class(self):
        from tabfm import TabFMClassifier, TabFMRegressor

        is_classification = self.problem_type in ["binary", "multiclass"]
        return TabFMClassifier if is_classification else TabFMRegressor

    def _load_backbone(self, device: str, compute_dtype: str = "bfloat16"):
        """Load the pre-trained TabFM torch backbone (cached per device); cast to bf16 on GPU.

        TabFM is bf16-native (`# bf16 SDPA` in the model; Google benchmarked on TPU, which computes
        in bf16), and bf16 ~halves activation memory — which is what lets large/WIDE datasets (e.g.
        APSFailure 76k x 170) fit a single 80 GB GPU at FULL context. Pass the `compute_dtype`
        hyperparameter = "float32" to opt out. CPU stays fp32 (bf16 on CPU is slow, no memory win).
        """
        from tabfm import tabfm_v1_0_0_pytorch

        model_type = "classification" if self.problem_type in ["binary", "multiclass"] else "regression"
        # The paper's weights at a pinned revision (or TABFM_CHECKPOINT_DIR); see resolve_tabfm_checkpoint().
        ckpt = resolve_tabfm_checkpoint()
        backbone = tabfm_v1_0_0_pytorch.load(model_type=model_type, device=device, checkpoint_path=ckpt)
        if device == "cuda" and str(compute_dtype).lower() in ("bfloat16", "bf16"):
            import torch

            backbone = backbone.to(torch.bfloat16)
        print(f"[TabFMModel] backbone dtype={next(backbone.parameters()).dtype} on {device}")
        return backbone

    @staticmethod
    def _resolve_device(num_gpus: int) -> str:
        if num_gpus and num_gpus > 0:
            import torch

            if not torch.cuda.is_available():
                raise AssertionError(
                    "Fit specified to use GPU, but CUDA is not available on this machine. "
                    "Please switch to CPU usage instead.",
                )
            return "cuda"
        return "cpu"

    def _fit(self, X, y, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        X = self.preprocess(X, y=y)

        # Match the finetune path (tabfm_finetune._fresh_model): run fp32 matmuls on the TF32 tensor
        # cores instead of the FP32 CUDA cores. ~7x faster than pure fp32 on H100, and the base
        # (identity, no delta) has nothing to lose from the 10-bit mantissa — this keeps the base
        # deploy at the SAME matmul precision as the diagscale/full deploy (their delta is TF32-safe,
        # validated by the TabPFN3/TabICL boards), so base-vs-FT stays precision-matched.
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True

        params = dict(self._get_model_params())
        preset = params.pop("preset", "default")  # control key — not a TabFM estimator kwarg
        params.setdefault(self.seed_name, self.fixed_random_state)

        # Cap context to fit a single GPU (see class docstring). Pop the control keys either way so
        # they never reach the estimator; apply only for the default preset (ensemble sets enable_nnls,
        # which is exclusive with max_num_rows).
        budget = params.pop("context_cell_budget", self._CONTEXT_CELL_BUDGET)
        row_ceiling = params.pop("context_row_ceiling", self._CONTEXT_ROW_CEILING)
        if preset == "default" and budget and "max_num_rows" not in params:
            n_rows, n_feats = int(X.shape[0]), max(int(X.shape[1]), 1)
            cap = min(int(row_ceiling), int(budget) // n_feats)
            if n_rows > cap:
                params["max_num_rows"] = cap
                print(f"[TabFMModel] context cap: {n_rows}x{n_feats} -> max_num_rows={cap} "
                      f"(budget={budget} cells, ceiling={row_ceiling}) to avoid single-GPU OOM")

        # Two-level feature cap decided by CELLS (train_rows x features) — see the class constants.
        # `max_features_cap` (>0) still forces an unconditional flat cap; otherwise: fits at 500 ->
        # 500 (= Google), else fall back to the safe 100-feature cap. Only tall+wide tables fall back.
        flat_cap = int(params.pop("max_features_cap", 0) or 0)
        feat_budget = int(params.pop("feature_cell_budget", self._FEATURE_CELL_BUDGET) or 0)
        n_rows, n_feats = int(X.shape[0]), int(X.shape[1])
        if "max_num_features" not in params:
            if flat_cap and n_feats > flat_cap:
                params["max_num_features"] = flat_cap
                print(f"[TabFMModel] flat feature cap: {n_feats}f -> max_num_features={flat_cap}", flush=True)
            elif feat_budget:
                eff = min(n_feats, self._ESTIMATOR_MAX_FEATURES)  # the estimator already caps to 500
                if n_rows * eff > feat_budget:                    # won't fit at 500 -> safe 100 fallback
                    cap = min(self._OOM_FALLBACK_FEATURES, eff)
                    params["max_num_features"] = int(cap)
                    print(f"[TabFMModel] tall+wide {n_rows}x{n_feats} ({n_rows * eff / 1e6:.1f}M cells @<=500 "
                          f"> {feat_budget / 1e6:.1f}M budget) -> max_num_features={cap} to fit one 80GB GPU",
                          flush=True)

        compute_dtype = params.pop("compute_dtype", "bfloat16")  # control key: bf16 = TabFM-native (TPU-like)
        # Keep FULL context (official protocol) even where fp32 would OOM: fall back to bf16 for those
        # cells only (in TabArena: just APSFailure). bf16-vs-fp32 inference accuracy is negligible, and
        # the base has no delta to protect — so we trade precision, not context. (The finetune path,
        # which DOES need fp32 for its delta, caps the context there instead — see FinetunedTabFMModel.)
        eff_rows = int(params.get("max_num_rows") or X.shape[0])
        # effective feature count = our cap if set, else the estimator's own 500 default (capped by width)
        eff_cols = int(params.get("max_num_features") or min(int(X.shape[1]), self._ESTIMATOR_MAX_FEATURES))
        if compute_dtype == "float32" and eff_rows * eff_cols > self._FP32_CELL_LIMIT:
            compute_dtype = "bfloat16"
            print(f"[TabFMModel] fp32 full-context {eff_rows}x{eff_cols}="
                  f"{eff_rows * eff_cols / 1e6:.1f}M cells > {self._FP32_CELL_LIMIT / 1e6:.1f}M "
                  f"-> bf16 to keep full context on one GPU", flush=True)
        # Adaptive test-row chunk for the chunked deploy predict (FinetunedTabFMModel._chunked): keep
        # the col-FF over (context + chunk) rows under the cell budget (bf16 = 2x the fp32 byte-
        # headroom). Solved from (rows, features), so low-cell datasets get the max chunk (fast) and
        # only high-cell ones near the limit get a small chunk — no hardcoded per-dataset list.
        cell_budget = self._FP32_CELL_LIMIT * (2 if compute_dtype == "bfloat16" else 1)
        self._deploy_chunk = int(min(getattr(self, "_FT_DEPLOY_TEST_CHUNK", 8000),
                                     max(1000, cell_budget // eff_cols - eff_rows)))
        device = self._resolve_device(num_gpus)
        backbone = self._load_backbone(device, compute_dtype)
        estimator_cls = self._get_model_class()

        if preset == "ensemble":
            self.model = estimator_cls.ensemble(model=backbone, **params)
        elif preset == "default":
            self.model = estimator_cls(model=backbone, **params)
        else:
            raise ValueError(f"unknown TabFM preset {preset!r} (expected 'default' | 'ensemble')")

        self.model = self.model.fit(X=X, y=y)

    # --- problem types / limits ---
    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        # Hard architectural cap of the released model (pytorch ClassificationConfig.max_classes == 10).
        # Classification tasks with more classes are excluded and RF-imputed at compare time (same
        # treatment other capped models get); regression is unaffected.
        default_auxiliary_params.update({"max_classes": 10})
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """One fold at a time (parallel folding races the shared HF download) + refit enabled."""
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update(
            {
                "fold_fitting_strategy": "sequential_local",
                "refit_folds": default_ag_args_ensemble.pop("refit_folds", True),
            }
        )
        return default_ag_args_ensemble

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    # --- resources / device ---
    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {"num_cpus": 1, "num_gpus": 1 if is_gpu_available else 0}

    def get_device(self) -> str:
        """Device type of the underlying TabFM torch backbone (held on the sklearn estimator)."""
        backbone = getattr(self.model, "model", None)
        if backbone is None:
            return "cpu"
        try:
            return next(backbone.parameters()).device.type
        except StopIteration:
            return "cpu"

    def _set_device(self, device: str) -> None:
        backbone = getattr(self.model, "model", None)
        if backbone is not None:
            self.model.model = backbone.to(device)

    # --- memory estimate (mirrors the TabPFN-3 wrapper: a comparable full-context ICL FM) ---
    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=self._get_model_params(),
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(cls, *, X: pd.DataFrame, **kwargs) -> int:
        """Heuristic: model + activations baseline plus a multiple of the dataset footprint.
        TabFM is smaller than TabPFN-3, but full-context ICL activations dominate; revisit with
        measured peaks once we have profiling data."""
        baseline_mem_est = 8 * 1e9  # 8 GB (model + activations); conservative
        dataset_mem_est = 5 * get_approximate_df_mem_usage(X).sum()
        return int(baseline_mem_est + dataset_mem_est)


def prefetch_weights() -> None:
    """Warm the HF cache by downloading both TabFM PyTorch checkpoints (classification+regression)."""
    from tabfm import tabfm_v1_0_0_pytorch

    tabfm_v1_0_0_pytorch.load(model_type="classification")
    tabfm_v1_0_0_pytorch.load(model_type="regression")
