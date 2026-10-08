"""Per-task test-time finetuning loop for TabICL v2 — the machinery behind ``FinetunedTabICLModel``.

A faithful port of the validated sandbox runner ``ttc.benchmarks.tabicl_diagscale._finetune``
(clone ``ttc-diagscale`` @ ``diagscale``), stripped to what the AutoGluon wrapper needs and fp32-only.
Injects the diagscale / (IA)³ adapter via the tabicl fork's ``tabicl._model.peft`` (``ta_diagscale``
env), trains ONLY the adapter deltas (zero-init -> behaviour-preserving), restores the val-best epoch
(or the pretrained init when no epoch beats base -> never-worse-than-base).

TabICL specifics vs the TabFM loop (all forced by the backbone):
  * classifier and regressor are SEPARATE checkpoints (picked by task type);
  * training episodes use tabicl's own ``_build_meta_batch`` (ctx/qry split + inference-equivalent
    preprocessing + class remap), n_estimators=1 single view; loss = CE (cls) / pinball on the raw
    quantile head (reg) — what the pretrained heads expect;
  * ``select_trainable(..., freeze_others=False)`` — flipping requires_grad on col_embedder weights
    shifts the CUDA train-mode forward ~5e-3 even at δ=0 (kernel selection keyed on the flag); keeping
    the flags on preserves kernel parity, only the returned δ params actually step;
  * singleton-class guard (drop episode-sampled classes with <2 rows, else StratifiedShuffleSplit
    inside the builder raises); ``adapter_state`` keys strip the grad-checkpoint ``.blk.`` infix;
  * the estimator is built with ``use_amp=False, use_fa3=False`` — the wrapper's "auto" turns on fp16
    autocast for n>=1024 rows / >=60 features, quantizing ``1+δ`` to the fp16 grid at eval. Eval is
    always fp32. (The (IA)³ KV-cache double-scaling fix lives inside the fork's attention.py.)

Feature encoding: episodes AND the deployed estimator must see IDENTICAL numeric inputs, so the caller
numericizes up front with ``NumericEncoder`` (mirrors tabicl's own OrdinalEncoder+SimpleImputer) and
feeds the encoded array here; the estimator's internal encoder is then a no-op on already-numeric data.
"""
from __future__ import annotations

import copy
import math
import os

import numpy as np

from ttc.backbones.tabfm_finetune import _delta_norms, _head, _score, _val_metric  # backbone-agnostic


class NumericEncoder:
    """Ordinal-encode object/category/bool/string columns (fit on train, unknown -> -1) and
    mean-impute numeric NaN (train means; all-NaN -> 0), returning a float32 ndarray. Applied UP
    FRONT so the finetune episodes and the deployed TabICL estimator see identical numeric inputs.
    Mirrors the sandbox ``_to_numeric`` / tabicl's own ``X_encoder_``. Accepts a DataFrame or array.
    """

    def fit(self, X) -> "NumericEncoder":
        import pandas as pd
        from sklearn.preprocessing import OrdinalEncoder

        df = pd.DataFrame(X).reset_index(drop=True)
        self.columns_ = list(df.columns)
        self.cat_ = [c for c in df.columns
                     if df[c].dtype == object or str(df[c].dtype) in ("category", "string", "bool")]
        self.enc_ = None
        if self.cat_:
            self.enc_ = OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1).fit(df[self.cat_].astype(str))
        arr = self._encode(df)
        finite = np.isfinite(arr)
        self.col_mean_ = np.where(finite, arr, 0.0).sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
        return self

    def _encode(self, df) -> np.ndarray:
        df = df.copy()
        if self.cat_:
            df[self.cat_] = self.enc_.transform(df[self.cat_].astype(str))
        return np.asarray(df, dtype=np.float32)

    def transform(self, X) -> np.ndarray:
        import pandas as pd

        df = pd.DataFrame(X).reset_index(drop=True)
        if list(df.columns) != self.columns_ and len(df.columns) == len(self.columns_):
            df.columns = self.columns_          # align positional -> fitted names (ndarray inputs)
        arr = self._encode(df)
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.broadcast_to(self.col_mean_, arr.shape)[bad]
        return arr


# ── backbone / estimator construction ───────────────────────────────────────────────────────────
def _pristine_backbone(is_cls: bool, checkpoint_version: str | None = None):
    """Pretrained TabICL v2 on CPU fp32 (classifier or regressor checkpoint); deep-copied per
    finetune. Returns (model, model_config, model_path) so fresh sklearn wrappers can adopt a model
    without reloading weights."""
    from tabicl import TabICLClassifier, TabICLRegressor

    kw = {"n_estimators": 1, "device": "cpu", "allow_auto_download": True}
    if checkpoint_version:
        kw["checkpoint_version"] = checkpoint_version
    loader = (TabICLClassifier if is_cls else TabICLRegressor)(**kw)
    loader._resolve_device()
    loader._load_model()
    return loader.model_.eval(), loader.model_config_, loader.model_path_


def _fresh_model(pristine, device: str):
    import torch

    model, cfg, path = pristine
    m = copy.deepcopy(model).to(device)
    m.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    return m, cfg, path


def _make_est(model, cfg, path, is_cls: bool, seed: int, device: str, n_estimators: int = 1):
    """Estimator adopting ``model`` in place (shadow ``_load_model`` so fit never reloads weights).
    ``use_amp``/``use_fa3`` pinned OFF: the wrapper's "auto" turns fp16 autocast on for n>=1024 rows
    or >=60 features, quantizing ``1+δ`` to the fp16 grid at eval and making eval precision differ
    across datasets. Protocol: eval is always fp32."""
    from tabicl import TabICLClassifier, TabICLRegressor

    est = (TabICLClassifier if is_cls else TabICLRegressor)(
        n_estimators=n_estimators, device=device, random_state=seed, use_amp=False, use_fa3=False)
    est.model_ = model
    est.model_config_ = cfg
    est.model_path_ = path
    est._load_model = lambda: None
    return est


def _checkpoint_blocks(model) -> None:
    """Gradient-checkpoint the transformer stacks during training (transparent under no_grad).
    The fp32 math-backend SDPA saves [heads, T, T] attention probs per ICL block; wrapping keeps
    headroom for the full-FT arm at ctx_qry_cap rows."""
    import torch
    from torch.utils.checkpoint import checkpoint

    class _CkptBlock(torch.nn.Module):
        def __init__(self, blk):
            super().__init__()
            self.blk = blk

        def forward(self, *args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(self.blk, *args, use_reentrant=False, **kwargs)
            return self.blk(*args, **kwargs)

        def forward_with_cache(self, *args, **kwargs):  # KV-cache inference path, no_grad only
            return self.blk.forward_with_cache(*args, **kwargs)

    for blocks in (model.icl_predictor.tf_icl.blocks,
                   model.row_interactor.tf_row.blocks,
                   model.col_embedder.tf_col.blocks):
        for i, blk in enumerate(blocks):
            blocks[i] = _CkptBlock(blk)


# ── finetuning loop ─────────────────────────────────────────────────────────────────────────────
def _episode_loss(model, batch, is_cls: bool, device: str):
    """Loss on one MetaBatch: CE on sliced logits (cls) / pinball on the raw quantile head (reg),
    matching tabicl._finetune classifier/regressor semantics."""
    import torch
    import torch.nn.functional as F
    from tabicl._finetune.data import move_meta_batch
    from tabicl._finetune.regressor import _pinball_loss

    batch = move_meta_batch(batch, torch.device(device))
    if is_cls:
        logits = model(batch.X, batch.y_train.float())
        n_classes = int(batch.y_train.max().item()) + 1
        return F.cross_entropy(logits[..., :n_classes].reshape(-1, n_classes).float(),
                               batch.y_query.long().reshape(-1))
    quantiles = model(batch.X, batch.y_train)
    q = quantiles.shape[-1]
    alpha = torch.linspace(0.0, 1.0, q + 2, device=quantiles.device, dtype=quantiles.dtype)[1:-1]
    return _pinball_loss(quantiles, batch.y_query, alpha)


def _skip_batch(batch, is_cls: bool) -> bool:
    """Skip episodes whose query has classes absent from the context (CE undefined) —
    tabicl._finetune.classifier._task_skip_batch."""
    import torch

    if not is_cls:
        return False
    ctx = torch.unique(batch.y_train.reshape(-1))
    qry = torch.unique(batch.y_query.reshape(-1).to(ctx.dtype))
    return not bool(torch.isin(qry, ctx).all())


def _finetune(pristine, Xtr_num, ytr, is_cls, device, seed, *, mode, pattern, lr, epochs=30,
              patience=8, min_delta=1e-4, ctx_qry_cap=10000, qry_frac=0.2, weight_decay=0.01,
              grad_clip=1.0, peft_cfg=None, tag="", n_est_finetune=2):
    """Finetune a fresh copy of the backbone; return (finetuned model, meta). ``Xtr_num`` must be a
    numeric float32 array (see NumericEncoder). Episodes are built with tabicl's ``_build_meta_batch``
    on the 90% split; validation scores the 10% split through the sklearn predict path."""
    import torch
    from sklearn.model_selection import train_test_split
    from tabicl._finetune.data import _build_meta_batch
    from tabicl._model import peft

    y_arr = np.asarray(ytr)
    if is_cls:  # episode labels must be 0..K-1 ints (the meta-batch builder indexes with them)
        from sklearn.preprocessing import LabelEncoder
        y_ep = LabelEncoder().fit_transform(y_arr).astype(np.int64)
    else:
        y_ep = y_arr.astype(np.float64)

    n_all = len(Xtr_num)
    val_size = int(max(round(0.1 * n_all), min(200, round(0.25 * n_all))))
    try:
        X_ft, X_val, y_ft, y_val, yep_ft, _ = train_test_split(
            Xtr_num, y_arr, y_ep, test_size=val_size, random_state=seed,
            stratify=y_ep if is_cls else None)
    except ValueError:  # ultra-rare classes: stratification impossible
        X_ft, X_val, y_ft, y_val, yep_ft, _ = train_test_split(
            Xtr_num, y_arr, y_ep, test_size=val_size, random_state=seed)

    model, cfg, path = _fresh_model(pristine, device)
    _checkpoint_blocks(model)
    est = _make_est(model, cfg, path, is_cls, seed, device, n_estimators=n_est_finetune).fit(X_ft, y_ft)

    is_adapter = mode in ("diagscale", "ia3")
    if is_adapter:
        probe = _head(X_val, 256)
        p0 = est.predict_proba(probe) if is_cls else est.predict(probe)
    # freeze_others=False on adapters: keep flag parity so injection stays bit-exact on CUDA.
    peft_kwargs = {**({"pattern": pattern} if pattern else {}), **(peft_cfg or {})}
    if is_adapter:
        peft_kwargs["freeze_others"] = False
    trainable = peft.select_trainable(model, mode=mode, **peft_kwargs)
    if is_adapter:
        p1 = est.predict_proba(probe) if is_cls else est.predict(probe)
        p0a, p1a = np.asarray(p0), np.asarray(p1)
        tol = 1e-5 * max(1.0, float(np.abs(p0a).max()))
        if not np.allclose(p0a, p1a, rtol=0.0, atol=tol):
            raise RuntimeError(f"{tag}: {mode} injection broke identity "
                               f"(max diff {np.abs(p0a - p1a).max():.2e})")

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    n = len(X_ft)
    take = min(n, ctx_qry_cap)
    steps_per_epoch = int(max(1, min(10, round(n / take))))
    total_steps, warmup = steps_per_epoch * epochs, max(1, int(steps_per_epoch * epochs * 0.1))

    def _lr_lambda(step):  # 10% linear warmup then cosine to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    val0 = _val_metric(est, X_val, y_val, is_cls)
    best, best_epoch, patience_ct, epochs_run = val0, -1, 0, 0
    best_state = [p.detach().cpu().clone() for p in trainable]
    curve = [val0]
    print(f"[tabicl-ft] {tag}: n_ft={n} n_val={len(X_val)} take={take} "
          f"steps/epoch={steps_per_epoch} val0={val0:.5f}", flush=True)

    for epoch in range(epochs):
        epochs_run = epoch + 1
        rng = np.random.default_rng(seed * 100003 + epoch)  # same episode schedule across lrs
        losses = []
        model.train()  # train-mode forward: raw logits / raw quantile head (dropout is 0.0)
        for step in range(steps_per_epoch):
            idx = rng.permutation(n)[:take]
            if is_cls:  # the builder's StratifiedShuffleSplit raises on singleton classes
                vals, cnts = np.unique(yep_ft[idx], return_counts=True)
                if (cnts < 2).any():
                    idx = idx[~np.isin(yep_ft[idx], vals[cnts < 2])]
            m_ep = len(idx)
            if m_ep < 2:
                continue
            n_qry = int(np.clip(round(m_ep * qry_frac), 1, m_ep - 8)) if m_ep > 8 else 1
            try:
                batch = _build_meta_batch(
                    X_ft[idx], yep_ft[idx], classification=is_cls, n_estimators=n_est_finetune,
                    query_size=n_qry, epoch_seed=seed * 100003 + epoch, chunk_idx=step,
                    norm_methods=None, feat_shuffle_method="latin", class_shuffle_method="shift",
                    outlier_threshold=4.0, preprocessing_seed=seed)
            except ValueError as e:  # residual stratification corner cases: skip the episode, loudly
                print(f"[tabicl-ft] {tag} epoch {epoch + 1} step {step}: episode skipped ({e})",
                      flush=True)
                continue
            if _skip_batch(batch, is_cls):
                continue
            model.zero_grad(set_to_none=True)  # backbone grads exist under freeze_others=False
            loss = _episode_loss(model, batch, is_cls, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            sch.step()
            losses.append(float(loss.detach()))
        model.eval()
        v = _val_metric(est, X_val, y_val, is_cls)
        curve.append(v)
        improved = v < best - min_delta
        loss_str = f"{float(np.mean(losses)):.5f}" if losses else "skipped"
        print(f"[tabicl-ft] {tag} epoch {epoch + 1}/{epochs} "
              f"loss={loss_str} val={v:.5f}{' *' if improved else ''}", flush=True)
        if improved:
            best, best_epoch, patience_ct = v, epoch + 1, 0
            best_state = [p.detach().cpu().clone() for p in trainable]
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[tabicl-ft] {tag} early stop (patience={patience})", flush=True)
                break

    with torch.no_grad():  # restore best epoch (or the pretrained init when best_epoch=-1)
        for p, s in zip(trainable, best_state):
            p.copy_(s.to(p.device))

    tids = {id(t) for t in trainable}
    trainable_names = [n for n, p in model.named_parameters() if id(p) in tids]
    meta = {"mode": mode, "pattern": pattern, "lr": float(lr), "weight_decay": float(weight_decay),
            "best_epoch": best_epoch, "epochs_run": epochs_run, "val0": val0,
            "best_val": float(best), "val_curve": curve, "trainable_names": trainable_names,
            "n_trainable": int(sum(p.numel() for p in trainable))}
    if is_adapter:
        meta["delta_norms"] = _delta_norms(model)
        if best_epoch >= 0 and not any(v[1] > 0 for v in meta["delta_norms"].values()):
            raise RuntimeError(f"{tag}: trained (best_epoch={best_epoch}) but all delta are zero")
    del est
    return model, meta


def finetune_tabicl(pristine, Xtr_num, ytr, is_cls, device, seed, *, mode, pattern, lr, epochs=30,
                    patience=8, ctx_qry_cap=10000, weight_decay=0.01, peft_cfg=None, tag="",
                    n_est_finetune=2) -> tuple:
    """Public entry: finetune a fresh copy of the (already-loaded) TabICL backbone on numeric
    (Xtr_num, ytr); return (finetuned model on ``device``, meta). The caller builds the FINAL
    estimator (``_make_est``) with the returned model and the ``pristine`` config/path.
    ``n_est_finetune`` = ensemble members per episode/val (official protocol default 2; the builder
    pre-shuffles each member's query labels so the CE loss is unchanged from single-view)."""
    return _finetune(pristine, Xtr_num, ytr, is_cls, device, seed, mode=mode, pattern=pattern, lr=lr,
                     epochs=epochs, patience=patience, ctx_qry_cap=ctx_qry_cap,
                     weight_decay=weight_decay, peft_cfg=peft_cfg, tag=tag,
                     n_est_finetune=n_est_finetune)
