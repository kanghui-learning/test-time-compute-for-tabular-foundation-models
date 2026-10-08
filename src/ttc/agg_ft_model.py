"""DiagScale adaptation followed by the configuration pool.

The backbone is adapted once per outer cell, then every configuration in the pool is built from
the adapted backbone and reduced by the same out-of-fold greedy selection as
``AggregationTabPFN3Model``.

The adapter is injected rather than saved as a checkpoint. ``inject_diagscale`` replaces each
``SoftmaxScalingMLP`` with a wrapper that holds the frozen module as ``.base`` and a trainable
``.delta`` (forward: ``base(q, n) * (1 + delta)``), so the adapted state dict cannot be loaded into
a stock TabPFN-3. The trained ``.delta`` tensors are kept in memory and re-injected into each
member after its fit builds the model.

This class overrides two hooks of ``AggregationTabPFN3Model``:
  * ``_fit``              adapts first (``ft_*`` hyperparameters, TabPFN's finetuning loop with its
                          own 90/10 split for early stopping), keeps the deltas, then calls
                          ``super()._fit``;
  * ``_build_estimator``  wraps each member so the deltas are injected after it fits.

Information flow: the adapter is fitted on the outer training split, and the out-of-fold
predictions that choose the reducer weights are computed on that same split. Each inner fold is
held out of the context, but its labels may have influenced the shared adapter, so the selector
inputs are out of fold with respect to the context only. Outer test labels are used only for
evaluation.
"""
from __future__ import annotations

import time

from ttc.agg_model import AggregationTabPFN3Model

# Modes this composition supports: they ADD adapter tensors and leave the pretrained weights frozen,
# so a member can load the stock checkpoint and then have the adapters injected. Weight-modifying
# modes (full / bitfit / regex — regex trains the ORIGINAL parameters selected by name) are rejected:
# there is no separable adapter to carry over.
_ADAPTER_MODES = ("diagscale", "lora")
# ...but tabpfn's select_trainable indexes cfg["pattern"] for these (KeyError without it).
_PATTERN_MODES = ("diagscale", "lora", "regex")


class _DeltaInjectedEstimator:
    """Proxy around a TabPFN estimator that re-injects finetuned adapter params after ``fit``.

    TabPFN builds ``model_`` lazily inside ``fit``, so injection cannot happen at construction time.
    Everything except ``fit`` is forwarded to the wrapped estimator (``predict`` / ``predict_proba``
    / ``classes_`` / ...), so this is a drop-in for ``AggregationTabPFN3Model``'s call pattern.
    """

    def __init__(self, est, adapter_sd: dict, pattern: str, mode: str):
        self._est = est
        self._adapter_sd = adapter_sd
        self._pattern = pattern
        self._mode = mode

    def fit(self, X, y):
        self._est.fit(X, y)
        model = getattr(self._est, "model_", None)
        if model is None:
            raise RuntimeError("member estimator has no model_ after fit — cannot inject the delta")
        if self._mode == "diagscale":
            from tabpfn.finetuning.peft import inject_diagscale
            n = inject_diagscale(model, pattern=self._pattern)
            if n == 0:
                raise RuntimeError(f"inject_diagscale matched 0 modules for pattern {self._pattern!r}")
        elif self._mode == "lora":
            from tabpfn.finetuning.peft import inject_lora
            inject_lora(model, pattern=self._pattern)
        # strict=False: adapter_sd holds ONLY the adapter tensors; every pretrained weight is
        # already loaded from the stock checkpoint and must stay as-is.
        missing, unexpected = model.load_state_dict(self._adapter_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"adapter keys not present in the member model: {list(unexpected)[:3]}")
        return self

    def __getattr__(self, name):  # only called when normal lookup fails -> forwards to the estimator
        return getattr(self._est, name)


class FinetunedAggregationTabPFN3Model(AggregationTabPFN3Model):
    """Aggregation member pool over a per-task finetuned TabPFN-3 backbone.

    Hyperparameters: the union of the two parents' —
        ``aggregation: {...}``  the member pool / reduction knobs (see AggregationTabPFN3Model)
        ``ft_*``                the finetuning knobs (ft_epochs / ft_lr / ft_mode / ft_regex ...);
                                ``ft_epochs <= 0`` skips the finetune entirely, which makes this
                                model bit-identical to plain ``AggregationTabPFN3Model``
                                (the identity rung that validates the composition).
    """

    ag_key = "TTC-AGGFT-TABPFN3"
    ag_name = "TTC-AggFT-TabPFN3"

    # ── every member is built from the stock checkpoint, then gets the deltas injected ──
    def _build_estimator(self, native_cfg: dict, n_estimators: int):
        est = super()._build_estimator(native_cfg, n_estimators)
        sd = getattr(self, "_ft_adapter_sd", None)
        if not sd:
            return est                        # no finetune (identity rung) -> plain aggregation
        return _DeltaInjectedEstimator(est, sd, self._ft_pattern, self._ft_mode)

    # ── finetune the backbone once; keep the trained adapter tensors ──
    def _finetune_backbone(self, X, y, num_cpus: int, num_gpus: int, ft: dict) -> dict:
        import torch
        from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor

        is_clf = self.problem_type in ("binary", "multiclass")
        device = self._resolve_tabpfn_device(num_gpus=num_gpus)
        if not isinstance(device, str):
            device = device[0]
        extra = {
            "model_path": self._get_model_checkpoint(),
            "categorical_features_indices": self._categorical_indices,
            "n_jobs": num_cpus,
        }
        mode = str(ft.get("ft_mode", "diagscale"))
        # tabpfn's select_trainable does cfg["pattern"] unconditionally for the injecting modes, so
        # they MUST get one even when the config omits ft_regex (default "." = every module).
        pattern = str(ft.get("ft_regex", "."))
        ft_cfg = {"pattern": pattern} if mode in _PATTERN_MODES else None
        common = {
            "device": device,
            "epochs": int(ft.get("ft_epochs", 10)),
            "learning_rate": float(ft.get("ft_lr", 1e-2)),
            "random_state": self.fixed_random_state,
            "time_limit": ft.get("ft_time_limit"),
            "finetune_mode": mode,
            "finetune_cfg": ft_cfg,
        }
        cls = FinetunedTabPFNClassifier if is_clf else FinetunedTabPFNRegressor
        kw = {"extra_classifier_kwargs": extra} if is_clf else {"extra_regressor_kwargs": extra}
        t0 = time.monotonic()
        ftm = cls(**common, **kw).fit(X=X, y=y)
        secs = time.monotonic() - t0

        inner = getattr(ftm, "finetuned_estimator_", None)
        model = getattr(inner, "model_", None) if inner is not None else None
        if model is None:
            raise RuntimeError("finetuning produced no finetuned_estimator_.model_")
        # keep ONLY the adapter tensors (diagscale: '*.delta'; lora: '*.lora_A/B'). The pretrained
        # weights are untouched by these modes, so every member gets them from the stock checkpoint.
        suffixes = (".delta", ".lora_A", ".lora_B")
        sd = {k: v.detach().to("cpu").clone()
              for k, v in model.state_dict().items() if k.endswith(suffixes)}
        if not sd:
            raise RuntimeError(f"no adapter tensors found for ft_mode={mode!r} "
                               "(is it a weight-modifying mode? this path supports adapters only)")
        self._ft_mode, self._ft_pattern = mode, pattern
        self._ft_seconds = round(secs, 2)
        n_par = int(sum(v.numel() for v in sd.values()))
        del ftm, inner, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[agg-ft] finetuned {mode}@lr{common['learning_rate']:g} {common['epochs']}ep "
              f"n={len(X)} -> {len(sd)} adapter tensors / {n_par:,} params in {secs:.1f}s", flush=True)
        return sd

    def _fit(self, X, y, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        ft = self._get_ft_params()
        if int(ft.get("ft_epochs", 0)) > 0:
            if str(ft.get("ft_mode", "diagscale")) not in _ADAPTER_MODES:
                raise ValueError(
                    f"ft_mode={ft.get('ft_mode')!r} modifies the pretrained weights; this model "
                    "composes ADAPTER finetunes (diagscale / lora) with aggregation.")
            # preprocess exactly as the parent will (the finetune must see the same feature space)
            Xp = self.preprocess(X, y=y, is_train=True)
            self._ft_adapter_sd = self._finetune_backbone(Xp, y, num_cpus, num_gpus, ft)
        else:
            self._ft_adapter_sd = None        # identity rung == plain aggregation
        return super()._fit(X=X, y=y, num_cpus=num_cpus, num_gpus=num_gpus, **kwargs)
