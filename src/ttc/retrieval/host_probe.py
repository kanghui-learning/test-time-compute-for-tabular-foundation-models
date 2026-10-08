"""host_probe — the SINGLE module that touches TabPFN-3 internals to capture model-space signals.

Two captures, both run ONE instrumented host forward and return plain numpy (no live-model handle
escapes), so every downstream SIGNAL stays a pure function of an injected SignalContext:

  * capture_embeddings(host, X_test) -> (emb_train [n_train,D], emb_test [n_test,D])
      via TabPFN-3's PUBLIC get_embeddings API (post-transformer row embeddings; train+test in one
      space). No internal hook -> minimal version coupling.

  * capture_attention(host, X_test, n_train, indices=None, chunk=...) -> {call_idx: [n_test,n_train]}
      patches the v3 attention dispatcher, and for each test->train call (q_seq==n_test &
      k_seq==n_train) computes softmax(q·kᵀ/√d) over the train axis, MEAN over heads (Nori
      sample_attention recipe), chunked over queries. The LAST test->train call ~ the last ICL layer
      (regression) — for classification the trailing call(s) are the class-decoder; the layer-selection
      probe picks the best call-index empirically, so we never need to introspect icl_blocks here.

  * capture_attention_topk(predict_fn, X_test, n_train, k, indices) -> {call_idx: (idx, val)}
      the SAME attention, but streamed to per-query top-k on the GPU — O(n_test*k) host memory, no
      full [n_test, n_train] map — the only affordable form at n_train >~ 1M (LimiX-port scoring).

This module is the documented version-coupling point: install_guard() fail-fasts if the patched
dispatcher symbol is missing.
"""
from __future__ import annotations

import numpy as np


def install_guard() -> None:
    """Fail fast if the TabPFN-3 symbols host_probe couples to are absent (version drift)."""
    from tabpfn.architectures import tabpfn_v3 as v3
    if not hasattr(v3, "_batched_scaled_dot_product_attention"):
        raise RuntimeError(
            "host_probe: tabpfn_v3._batched_scaled_dot_product_attention missing — TabPFN-3 internals "
            "changed; the model-space signals (model_key / attention) need re-grounding."
        )


def _estimator(host):
    """The fitted TabPFN estimator (has predict_proba / get_embeddings). host may BE it or wrap it."""
    for obj in (getattr(host, "model", None), host):
        if obj is not None and hasattr(obj, "get_embeddings"):
            return obj
    # last resort: the module-level API on host.model
    return getattr(host, "model", host)


def capture_embeddings(host, X_test):
    """(emb_train [n_train, D], emb_test [n_test, D]) via the public get_embeddings API.
    Averages over the estimator ensemble axis (n_estimators) -> one vector per row."""
    est = _estimator(host)
    if hasattr(est, "get_embeddings"):
        et = est.get_embeddings(X_test, data_source="train")
        ee = est.get_embeddings(X_test, data_source="test")
    else:                                                  # fall back to the module-level function
        from tabpfn.base import get_embeddings
        et = get_embeddings(est, X_test, data_source="train")
        ee = get_embeddings(est, X_test, data_source="test")
    et, ee = np.asarray(et, np.float64), np.asarray(ee, np.float64)
    if et.ndim == 3:                                       # (n_estimators, n_rows, D) -> mean over est
        et = et.mean(0)
    if ee.ndim == 3:
        ee = ee.mean(0)
    return et, ee


def n_icl_layers(host):
    """Count of ICL transformer blocks (= the test->train attention layers) in the loaded TabPFNV3
    module. Used to resolve attention layer=-1 to the LAST ICL layer, EXCLUDING the trailing
    many-class-decoder test->train calls (classification). Returns None if not locatable (the caller
    then falls back to the last test->train call). The sole icl_blocks coupling point."""
    est = _estimator(host)
    roots = []
    ex = getattr(est, "executor_", None)
    if ex is not None:
        for attr in ("models", "model_caches"):
            seq = getattr(ex, attr, None)
            if seq:
                roots += [getattr(m, "model", m) for m in seq]   # _PerDeviceModelCache wraps as .model
    roots += [getattr(est, "model_", None), est]
    for root in roots:
        if root is None:
            continue
        if hasattr(root, "icl_blocks"):
            try:
                return len(root.icl_blocks)
            except TypeError:
                pass
        if callable(getattr(root, "modules", None)):
            try:
                for sub in root.modules():
                    if hasattr(sub, "icl_blocks"):
                        return len(sub.icl_blocks)
            except Exception:
                continue
    return None


def capture_attention(predict_fn, X_test, n_train, indices=None, chunk: int = 2048):
    """{call_idx: softmax attention [n_test, n_train]} for the requested test->train call indices.

    ``predict_fn`` MUST be the UNWRAPPED estimator predict (predict_proba/predict) — the retrieve
    operator wraps host.model.predict_proba, so passing the wrapped one would recurse infinitely.

    indices=None -> capture ALL test->train calls (probe / small data). indices=[i,...] -> only those.
    When any requested index is negative we capture ALL and resolve against the total count at the end.
    """
    import torch
    from tabpfn.architectures import tabpfn_v3 as v3

    n_test = int(np.asarray(X_test).shape[0])
    want_all = indices is None or any(i < 0 for i in indices)
    want = None if want_all else set(int(i) for i in indices)

    maps: dict[int, np.ndarray] = {}
    counter = {"i": 0}
    orig = v3._batched_scaled_dot_product_attention

    def cap(q_BSHD, k_BSJD, v_BSJD, softmax_scaling_layer=None, _backends_override=None):
        out = orig(q_BSHD, k_BSJD, v_BSJD, softmax_scaling_layer, _backends_override)
        if q_BSHD.shape[1] == n_test and k_BSJD.shape[1] == n_train:
            idx = counter["i"]; counter["i"] += 1
            if want is None or idx in want:
                qq = softmax_scaling_layer(q_BSHD, k_BSJD.shape[1]) if softmax_scaling_layer is not None else q_BSHD
                qh = qq.transpose(1, 2).float()            # [B, H, S, D]
                kh = k_BSJD.transpose(1, 2).float()        # [B, H, J, D]
                if kh.shape[1] != qh.shape[1]:             # GQA: expand kv heads to query heads
                    kh = kh.repeat_interleave(qh.shape[1] // kh.shape[1], dim=1)
                d = qh.shape[-1]
                S, J = qh.shape[2], kh.shape[2]
                rows = np.empty((S, J), dtype=np.float32)
                for s0 in range(0, S, chunk):
                    s1 = min(s0 + chunk, S)
                    logits = (qh[:, :, s0:s1] @ kh.transpose(-1, -2)) * (d ** -0.5)   # [B,H,s,J]
                    a = torch.softmax(logits, dim=-1).mean(dim=1)[0]                  # mean heads, batch0
                    rows[s0:s1] = a.cpu().numpy()
                maps[idx] = rows
        return out

    try:
        v3._batched_scaled_dot_product_attention = cap
        with torch.inference_mode():
            predict_fn(X_test)
    finally:
        v3._batched_scaled_dot_product_attention = orig

    if not maps:
        raise RuntimeError(f"capture_attention: no test->train call (q={n_test}, k={n_train}) fired")
    if want_all and indices is not None:                   # resolve negative indices against the total
        keys = sorted(maps)
        return {i: maps[keys[i]] for i in indices}
    return maps


def capture_attention_topk(predict_fn, X_test, n_train, k, indices, buf_gb: float = 8.0):
    """Streaming per-query top-k of the test->train softmax sample attention.

    Same attention semantics as capture_attention (softmax over the train axis, MEAN over heads) but
    never materializes the [n_test, n_train] map: each query-chunk is reduced to its top-k train
    indices ON the GPU, so host memory is O(n_test*k) instead of O(n_test*n_train) (400 GB at
    50k x 2M), and K stays UNEXPANDED under GQA/MQA (grouped-view matmul instead of the
    repeat_interleave copy). GPU scratch = 2*H*chunk*n_train*4 bytes; chunk auto-sizes to buf_gb.

    ``indices`` must be explicit NON-negative call indices (streaming capture-all is unaffordable;
    resolve the last ICL layer via n_icl_layers). Returns {call_idx: (idx [n_test,k] int64,
    val [n_test,k] float32)}; val is softmax mass, so val.sum(1) = per-query captured coverage.
    """
    import torch
    from tabpfn.architectures import tabpfn_v3 as v3

    n_test = int(np.asarray(X_test).shape[0])
    if indices is None or any(int(i) < 0 for i in indices):
        raise ValueError("capture_attention_topk: indices must be explicit non-negative call indices")
    want = set(int(i) for i in indices)

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    counter = {"i": 0}
    orig = v3._batched_scaled_dot_product_attention

    def cap(q_BSHD, k_BSJD, v_BSJD, softmax_scaling_layer=None, _backends_override=None):
        res = orig(q_BSHD, k_BSJD, v_BSJD, softmax_scaling_layer, _backends_override)
        if q_BSHD.shape[1] == n_test and k_BSJD.shape[1] == n_train:
            idx = counter["i"]; counter["i"] += 1
            if idx in want:
                qq = softmax_scaling_layer(q_BSHD, k_BSJD.shape[1]) if softmax_scaling_layer is not None else q_BSHD
                qh = qq.transpose(1, 2).float()            # [B, H, S, D]
                kh = k_BSJD.transpose(1, 2).float()        # [B, Hk, J, D]
                B, H, S, D = qh.shape
                Hk, J = kh.shape[1], kh.shape[2]
                if Hk == H or Hk == 1:                     # MHA, or MQA (broadcasts over H — no copy)
                    kT = kh.transpose(-1, -2)              # [B, Hk, D, J]
                else:                                      # true GQA: expand (correctness over memory)
                    kT = kh.repeat_interleave(H // Hk, dim=1).transpose(-1, -2)
                kk = min(int(k), J)
                chunk = int(max(16, min(4096, buf_gb * 1e9 / (2 * H * J * 4))))
                ti = np.empty((S, kk), dtype=np.int64)
                tv = np.empty((S, kk), dtype=np.float32)
                for s0 in range(0, S, chunk):
                    s1 = min(s0 + chunk, S)
                    logits = (qh[:, :, s0:s1] @ kT) * (D ** -0.5)        # [B, H, s, J]
                    a = torch.softmax(logits, dim=-1).mean(dim=1)[0]     # mean heads, batch0 -> [s, J]
                    del logits
                    v_, i_ = torch.topk(a, kk, dim=-1)
                    del a
                    ti[s0:s1] = i_.cpu().numpy(); tv[s0:s1] = v_.cpu().numpy()
                out[idx] = (ti, tv)
        return res

    try:
        v3._batched_scaled_dot_product_attention = cap
        with torch.inference_mode():
            predict_fn(X_test)
    finally:
        v3._batched_scaled_dot_product_attention = orig

    if set(out) != want:
        raise RuntimeError(f"capture_attention_topk: wanted calls {sorted(want)} but "
                           f"{counter['i']} test->train calls fired (got {sorted(out)})")
    return out
