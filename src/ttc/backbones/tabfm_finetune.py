"""Per-task test-time finetuning loop for TabFM — the machinery behind ``FinetunedTabFMModel``.

A faithful port of the validated sandbox runner ``ttc.benchmarks.tabfm_diagscale._finetune``
(clone ``ttc-diagscale`` @ ``diagscale``), stripped to the pieces the AutoGluon wrapper needs:
no leaderboard / recovered% / figure / CLI, and fp32-only (protocol v2 trains fp32; the bf16
autocast arm is dropped). The DIFFERENTIABLE finetune injects the diagscale / (IA)3 adapter via the
tabfm fork's ``tabfm.src.pytorch.peft`` (``ta_tabfm_dev`` / ``ta_diagscale`` env), trains ONLY the
adapter deltas (zero-init -> injection is behaviour-preserving), and restores the val-best epoch
(or the pretrained init when no epoch beats base -> never-worse-than-base).

Protocol (mirrors ``tabpfn.finetuning.FinetunedTabPFNBase``, single-view TabFM edition):
  * 10% val split with a 200-row floor (capped at 25% n) — the split does triple duty
    (early-stop epoch, and the caller's lr selection / never-worse fallback);
  * per-epoch random context/query splits of the 90% train part (loss on the query rows), using
    the estimator's ``ensemble_generator_.transform_fold`` — the EXACT inference preprocessing;
  * ``n_estimators=1`` single-view (class shift asserted 0, so the CE labels are unambiguous);
  * AdamW + 10% linear warmup + cosine, grad-clip 1.0, early stopping on val (patience), best-epoch
    restore, and an injection identity guard (delta=0 must reproduce base predictions);
  * every transformer block is gradient-checkpointed so the grad-enabled forward at ctx+qry=6144
    fits an 80 GB GPU (the fp32 math SDPA backend saves attention probs otherwise).

The caller (``FinetunedTabFMModel._fit``) refits the FINAL estimator on the full-train context with
the returned finetuned backbone.
"""
from __future__ import annotations

import copy
import math

import numpy as np


# ── backbone / estimator construction ───────────────────────────────────────────────────────────
def _pristine_backbone(is_cls: bool):
    """Pretrained PyTorch TabFM on CPU fp32 (in-process cached by the tabfm loader); deep-copied per
    finetune so training never mutates the shared cache. Shares ``resolve_tabfm_checkpoint`` with the
    base wrapper so base and finetune load identical weights."""
    from tabfm import tabfm_v1_0_0_pytorch

    from ttc.backbones.tabfm import resolve_tabfm_checkpoint

    mt = "classification" if is_cls else "regression"
    return tabfm_v1_0_0_pytorch.load(model_type=mt, checkpoint_path=resolve_tabfm_checkpoint(), device=None)


def _fresh_model(pristine, device: str):
    import torch

    m = copy.deepcopy(pristine).to(device)
    if device == "cuda":
        # bf16 finetune (train + deploy): halves activation memory and PRESERVES the trained delta
        # (verified 2026-07-24: full-bf16 keeps 89-104% of the fp32 gain; delta RMS ~0.05-0.08 >>
        # bf16 ULP@1 8e-3). Matches the bf16 base so base(ft_epochs=0)-vs-FT stays precision-matched;
        # also removes the fp32 full-context hang on very-tall datasets. CPU stays fp32 (bf16 slow, no win).
        m = m.to(torch.bfloat16)
    m.eval()  # no dropout/BN in TabFM; grads still flow where requested
    torch.backends.cuda.matmul.allow_tf32 = True  # TabFM is bf16-native; tf32 matmul is plenty
    return m


def _make_est(model, is_cls: bool, seed: int, max_num_features: int | None = None,
              n_estimators: int = 1):
    """Estimator adopting ``model``. ``n_estimators`` = ensemble views (1 = single-view; the official
    finetune protocol uses 2 for train/val). With n_estimators>1 each member gets its own feature
    shuffle + (clf) cyclic class shift; the loss un-shifts labels per member (see _finetune).
    ``max_num_features`` caps the columns per member (wide-table memory fix) — needed on the finetune
    est too, since its val / identity-guard predict runs FULL-context inference over the 90% split."""
    from tabfm import TabFMClassifier, TabFMRegressor

    cls = TabFMClassifier if is_cls else TabFMRegressor
    kw = dict(model=model, n_estimators=int(n_estimators), random_state=seed, verbose=False)
    if max_num_features:
        kw["max_num_features"] = int(max_num_features)
    return cls(**kw)


def _head(X, k):
    return X.iloc[:k] if hasattr(X, "iloc") else np.asarray(X)[:k]


def _score(est, X, y, is_cls: bool) -> dict:
    """Held-out error (lower=better): logloss for cls, rmse for reg."""
    from sklearn.metrics import log_loss

    yv = np.asarray(y)
    if is_cls:
        proba = np.asarray(est.predict_proba(X), dtype=np.float64)
        proba /= proba.sum(axis=1, keepdims=True)  # exact renorm (float32 softmax residue)
        return {"logloss": float(log_loss(yv, proba, labels=est.classes_))}
    pred = np.asarray(est.predict(X), dtype=np.float64)
    return {"rmse": float(np.sqrt(np.mean((pred - yv.astype(np.float64)) ** 2)))}


def _val_metric(est, X_val, y_val, is_cls: bool) -> float:
    key = "logloss" if is_cls else "rmse"
    return _score(est, X_val, y_val, is_cls)[key]


def _delta_norms(model) -> dict:
    """‖delta‖ per injected adapter tensor, grouped by top-level component (0 => untrained)."""
    out: dict[str, list[float]] = {}
    for name, p in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "diag_delta" or leaf.startswith("ia3_delta"):
            out.setdefault(name.split(".")[0], []).append(float(p.detach().float().norm()))
    return {c: (len(v), float(np.mean(v))) for c, v in out.items()}


def _checkpoint_blocks(model) -> None:
    """Gradient-checkpoint every transformer block during training (transparent under no_grad).

    The grad-enabled forward at ctx+qry=6144 OOMs an 80GB GPU without this: fp32 SDPA falls back to
    the math backend, which SAVES each block's attention probs for backward. Wrapping keeps only
    block-boundary activations and recomputes inside (as the tabpfn finetuning protocol does).
    """
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

    stacks = [model.icl_predictor.tf_icl.blocks,          # 24x MultiheadAttentionBlock
              model.row_interactor.tf_row.blocks,         # 3x MultiheadAttentionBlock
              model.row_interactor_2.tf_row.blocks,
              model.col_embedder.tf_col.blocks,           # 3x InducedSelfAttentionBlock
              model.col_embedder_2.tf_col.blocks]
    for blocks in stacks:
        for i, blk in enumerate(blocks):
            blocks[i] = _CkptBlock(blk)


def _uncheckpoint_blocks(model) -> None:
    """Reverse _checkpoint_blocks: unwrap every _CkptBlock back to the raw block, so the DEPLOY forward
    carries NO checkpoint machinery. `predict_proba` runs grad-enabled (not under no_grad), so a
    checkpointed block pins its block-boundary activations for a backward that never comes (~50+ GB on a
    tall test set) — the finetuned model OOMs on predict where the stock base does not (verified: identity
    fp32 fits, FT fp32 OOMs on APSFailure)."""
    stacks = [model.icl_predictor.tf_icl.blocks, model.row_interactor.tf_row.blocks,
              model.row_interactor_2.tf_row.blocks, model.col_embedder.tf_col.blocks,
              model.col_embedder_2.tf_col.blocks]
    for blocks in stacks:
        for i, blk in enumerate(blocks):
            if hasattr(blk, "blk"):  # a _CkptBlock wrapper
                blocks[i] = blk.blk


def _forward_tensors(model, Xs, ys_ctx, cat_masks, ds, device: str):
    """model(...) on one prepared ensemble batch (grad-enabled). Mirrors _predict_step_pytorch."""
    import torch

    dt = next(model.parameters()).dtype
    n_ctx = int(ys_ctx.shape[1])
    y_pad = np.pad(ys_ctx, ((0, 0), (0, Xs.shape[1] - n_ctx)), constant_values=-100)
    X_t = torch.from_numpy(np.ascontiguousarray(Xs)).to(device, dt)
    y_t = torch.from_numpy(y_pad).to(device)
    if y_t.dtype == torch.float64:
        y_t = y_t.to(torch.float32)
    ts_t = torch.full((X_t.shape[0],), n_ctx, dtype=torch.long, device=device)
    d_t = torch.from_numpy(np.asarray(ds)).to(device)
    cm_t = torch.from_numpy(np.asarray(cat_masks)).to(device) if cat_masks is not None else None
    return model(X_t, y_t, ts_t, cat_mask=cm_t, d=d_t)


# ── finetuning loop ─────────────────────────────────────────────────────────────────────────────
def _finetune(pristine, Xtr, ytr, is_cls, device, seed, *, mode, pattern, lr, epochs=30,
              patience=8, min_delta=1e-4, ctx_qry_cap=6144, qry_frac=0.2, weight_decay=0.01,
              grad_clip=1.0, peft_cfg=None, tag="", max_num_features=None, n_est_finetune=2):
    """Finetune a fresh copy of the backbone; return (finetuned model, meta). fp32 throughout.
    ``max_num_features`` caps columns on the fit/val estimator (wide-table fp32 memory); the caller
    should also row-subsample ``Xtr`` so the val / identity-guard full-context predict fits."""
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
    from tabfm.src.pytorch import peft

    # 10% val, 200-row floor (capped at 25% n): the val split is too noisy at ~100 rows for the
    # triple duty it does (early-stop epoch / lr selection / never-worse fallback).
    n_all = len(Xtr)
    val_size = int(max(round(0.1 * n_all), min(200, round(0.25 * n_all))))
    try:
        X_ft, X_val, y_ft, y_val = train_test_split(
            Xtr, ytr, test_size=val_size, random_state=seed,
            stratify=np.asarray(ytr) if is_cls else None)
    except ValueError:  # ultra-rare classes: stratification impossible
        X_ft, X_val, y_ft, y_val = train_test_split(Xtr, ytr, test_size=val_size, random_state=seed)

    model = _fresh_model(pristine, device)
    _checkpoint_blocks(model)
    est = _make_est(model, is_cls, seed, max_num_features, n_estimators=n_est_finetune).fit(
        X_ft, np.asarray(y_ft))
    gen = est.ensemble_generator_
    n_classes = int(getattr(est, "n_classes_", 0))

    # Injection identity guard (adapter modes): with delta=0 the predictions must be unchanged.
    is_adapter = mode in ("diagscale", "ia3")
    if is_adapter:
        probe = _head(X_val, 256)
        p0 = est.predict_proba(probe) if is_cls else est.predict(probe)
    cfg = {**({"pattern": pattern} if pattern else {}), **(peft_cfg or {})}
    trainable = peft.select_trainable(model, mode=mode, **cfg)
    if is_adapter:
        p1 = est.predict_proba(probe) if is_cls else est.predict(probe)
        p0a, p1a = np.asarray(p0), np.asarray(p1)
        # scale-aware absolute tolerance (a plain rtol would grant raw-unit regression outputs a
        # per-element slack of 1e-5*|pred| and mask a real identity break)
        tol = 1e-5 * max(1.0, float(np.abs(p0a).max()))
        if not np.allclose(p0a, p1a, rtol=0.0, atol=tol):
            raise RuntimeError(f"{tag}: {mode} injection broke identity "
                               f"(max diff {np.abs(p0a - p1a).max():.2e})")

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    n = len(X_ft)
    take = min(n, ctx_qry_cap)
    steps_per_epoch = int(max(1, min(10, round(n / take))))
    total_steps, warmup = steps_per_epoch * epochs, max(1, int(steps_per_epoch * epochs * 0.1))

    def _lr_lambda(step):  # 10% linear warmup then cosine to 0 (tabpfn finetuning schedule)
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    val0 = _val_metric(est, X_val, y_val, is_cls)
    best, best_epoch, patience_ct, epochs_run = val0, -1, 0, 0
    best_state = [p.detach().cpu().clone() for p in trainable]
    init_state = best_state  # pretrained-init handle: best_state is REBOUND on each val improve, so
                             # this keeps pointing at the epoch-0 clones (used for the drift diagnostic)
    curve, y_enc = [val0], np.asarray(gen.y_)  # y_: encoded (cls) / scaled (reg), row-aligned to X_ft
    print(f"[tabfm-ft] {tag}: n_ft={n} n_val={len(X_val)} take={take} "
          f"steps/epoch={steps_per_epoch} val0={val0:.5f}", flush=True)

    for epoch in range(epochs):
        epochs_run = epoch + 1
        rng = np.random.default_rng(seed * 100003 + epoch)  # same ctx/qry schedule across lrs
        losses = []
        for _ in range(steps_per_epoch):
            idx = rng.permutation(n)[:take]
            n_qry = int(np.clip(round(take * qry_frac), 1, take - 8)) if take > 8 else 1
            qry_idx, ctx_idx = np.sort(idx[:n_qry]), np.sort(idx[n_qry:])
            data, _ = gen.transform_fold(ctx_idx, qry_idx)
            Xs, ys, cat_masks, ds, cfgs = gen.prepare_ensemble_tensors(data)
            E = len(cfgs)                                   # ensemble members (views) this step
            opt.zero_grad(set_to_none=True)
            out = _forward_tensors(model, Xs, ys, cat_masks, ds, device)
            tail = out[:, len(ctx_idx):, :]                # [E, n_qry, n_out]; reshape is member-major
            if is_cls:
                # member i conditions on context labels cyclically shifted by cfgs[i][1] (see the fork
                # generator: y_ctx = (y + shift) % n_classes), so its query target is shifted the same.
                yq = torch.as_tensor(y_enc[qry_idx], dtype=torch.long, device=device)
                tgt = torch.stack([(yq + int(cfgs[i][1])) % n_classes for i in range(E)])
                loss = F.cross_entropy(tail[..., :n_classes].reshape(-1, n_classes).float(),
                                       tgt.reshape(-1))
            else:
                # gen.y_ is a single global target scale (per-member norm affects features, not y),
                # so every member's target is the same query labels, repeated member-major.
                yq = torch.as_tensor(y_enc[qry_idx], dtype=torch.float32, device=device)
                loss = F.mse_loss(tail[..., 0].reshape(-1).float(), yq.repeat(E))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            sch.step()
            losses.append(float(loss.detach()))
        v = _val_metric(est, X_val, y_val, is_cls)
        curve.append(v)
        improved = v < best - min_delta
        print(f"[tabfm-ft] {tag} epoch {epoch + 1}/{epochs} "
              f"loss={float(np.mean(losses)):.5f} val={v:.5f}{' *' if improved else ''}", flush=True)
        if improved:
            best, best_epoch, patience_ct = v, epoch + 1, 0
            best_state = [p.detach().cpu().clone() for p in trainable]
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[tabfm-ft] {tag} early stop (patience={patience})", flush=True)
                break

    # bf16 full-FT underflow diagnostic: total L2 drift of the LAST-epoch weights from the pretrained
    # init, measured BEFORE the val-best restore, so it reflects whether training moved the weights AT
    # ALL (independent of val selection). Near-zero at small lr = AdamW's ~lr-sized steps fell below the
    # bf16 ULP and the weights never actually moved. For diagscale init is 0, so this equals ‖delta‖.
    with torch.no_grad():
        drift = sum(float((p.detach().cpu().float() - s0.float()).pow(2).sum())
                    for p, s0 in zip(trainable, init_state)) ** 0.5
    print(f"[tabfm-ft] {tag} param drift ‖w_last-w_init‖={drift:.4e}", flush=True)

    with torch.no_grad():  # restore best epoch (or the pretrained init when best_epoch=-1)
        for p, s in zip(trainable, best_state):
            p.copy_(s.to(p.device))

    tids = {id(t) for t in trainable}
    trainable_names = [n for n, p in model.named_parameters() if id(p) in tids]
    meta = {"mode": mode, "pattern": pattern, "lr": float(lr), "weight_decay": float(weight_decay),
            "best_epoch": best_epoch, "epochs_run": epochs_run, "val0": val0,
            "best_val": float(best), "val_curve": curve, "trainable_names": trainable_names,
            "param_drift": float(drift),
            "n_trainable": int(sum(p.numel() for p in trainable))}
    if is_adapter:
        meta["delta_norms"] = _delta_norms(model)
        if best_epoch >= 0 and not any(v[1] > 0 for v in meta["delta_norms"].values()):
            raise RuntimeError(f"{tag}: trained (best_epoch={best_epoch}) but all delta are zero")
    _uncheckpoint_blocks(model)  # deploy path must be checkpoint-free (else predict pins activations)
    # Free finetune-only GPU memory before the caller deploys. TabFM's fp32 32-view deploy inference
    # OOMs on tall datasets if the finetune's optimizer state (~13 GB for the 1.6B full-FT arm) + the
    # per-param .grad buffers (~6 GB) are still held — empty_cache can't release memory the live
    # `opt`/grads reference, so drop the references first, then gc + empty_cache.
    import gc
    for p in model.parameters():
        p.grad = None
        p.requires_grad_(False)  # deploy is inference-only: without this the trained (requires_grad=True)
        # params make predict_proba build an autograd graph, whose activations (~50+ GB on a tall test
        # set) are pinned by the output tensor and survive into the NEXT cell -> cross-cell OOM (every
        # dataset fits ALONE but a wide->tall sequence OOMs; jobs 12279 vs 12280). base is unaffected
        # (its params are already requires_grad=False), which is why only the FT arms leaked.
    del est, opt, sch, best_state
    gc.collect()
    if next(model.parameters()).is_cuda:
        torch.cuda.empty_cache()
    return model, meta


def finetune_tabfm(is_cls, Xtr, ytr, device, seed, *, mode, pattern, lr, epochs=30, patience=8,
                   ctx_qry_cap=6144, weight_decay=0.01, peft_cfg=None, tag="",
                   max_num_features=None, n_est_finetune=2) -> tuple:
    """Public entry: load the pretrained TabFM backbone (CPU-cached), finetune a fresh copy on
    (Xtr, ytr), and return (finetuned model on ``device``, meta dict). The caller refits the FINAL
    estimator on the full-train context with the returned model. ``max_num_features`` caps columns
    on the fit/val estimator; row-subsample ``Xtr`` upstream for the row half of the fp32 cap.
    ``n_est_finetune`` = ensemble views per train/val step (official protocol default 2; the loss
    un-shifts clf labels per member — see _finetune).
    """
    pristine = _pristine_backbone(is_cls)
    return _finetune(pristine, Xtr, ytr, is_cls, device, seed, mode=mode, pattern=pattern, lr=lr,
                     epochs=epochs, patience=patience, ctx_qry_cap=ctx_qry_cap,
                     weight_decay=weight_decay, peft_cfg=peft_cfg, tag=tag,
                     max_num_features=max_num_features, n_est_finetune=n_est_finetune)
