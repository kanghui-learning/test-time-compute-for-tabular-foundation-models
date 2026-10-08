"""Reducers that combine a pool of member predictions into one prediction.

  ``uniform``  equal weights, no validation data
  ``argmax``   one-hot weight on the member with the best validation loss (best-single)
  ``caruana``  greedy ensemble selection with replacement (Caruana et al., 2004)

Each reducer chooses weights from per-member validation (out-of-fold) predictions; ``combine``
applies the same weights to the per-member test predictions.

Greedy selection follows AutoGluon's ``EnsembleSelection``: add, with replacement, the member that
most improves the validation loss of the running average for ``n_iterations`` rounds (default 40,
TabArena's setting); ``weight[j] = picks[j] / n_iterations``. Ties go to the first member in pool
order and all rounds are kept.

Score functions return a loss (lower is better): log loss for classification probabilities and
RMSE for regression. Both take ``(y_true, prediction)``.
"""
from __future__ import annotations

import numpy as np

# ── score functions (lower = better) ──────────────────────────────────────────────────────


def log_loss(y_true, proba, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of a class-probability matrix ``proba`` [n, C]."""
    proba = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y = np.asarray(y_true).astype(int)
    return float(-np.log(proba[np.arange(len(y)), y]).mean())


def rmse(y_true, pred) -> float:
    """Root mean squared error of a regression prediction vector ``pred`` [n]."""
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def score_fn_for(problem_type: str):
    return log_loss if problem_type in ("binary", "multiclass") else rmse


# ── weight choosers (operate on VALIDATION predictions) ────────────────────────────────────


def uniform_weights(m: int) -> np.ndarray:
    """Equal weights — no validation used."""
    return np.full(m, 1.0 / m, dtype=float)


def argmax_weights(val_preds, y_val, score_fn) -> np.ndarray:
    """One-hot on the single best candidate (= ``tuned``). nanargmin so a NaN-scoring candidate
    is never selected as the min (np.argmin would treat NaN as smallest); caruana is already
    NaN-safe via its best_s=inf init."""
    scores = np.asarray([score_fn(y_val, p) for p in val_preds], dtype=float)
    w = np.zeros(len(val_preds), dtype=float)
    w[int(np.nanargmin(scores)) if np.isfinite(scores).any() else 0] = 1.0
    return w


def caruana_weights(val_preds, y_val, score_fn, n_iterations: int = 40) -> tuple[np.ndarray, list[int]]:
    """Greedy ensemble selection with replacement (= ``tuned+ensemble``).

    Returns ``(weights, picks)`` where ``weights[j] = picks.count(j) / n_iterations`` and ``picks``
    is the per-round selection trace (for diagnostics).

    Vectorized across the m candidates: each round scores ALL candidates ``(ens_sum + preds[j]) /
    (size+1)`` in batched numpy — the identical arithmetic ``score_fn`` does per candidate, with the
    same NaN/tie handling (``nanargmin`` picks the lowest-index finite minimum; all-NaN -> 0, exactly
    the old ``best_s=inf`` strict-``<`` loop) — instead of an m-long Python loop. This is the per-cell
    hot path once k-fold OOF makes the val signal the FULL train (~10x the holdout). Candidates are
    processed in row-memory-bounded blocks so a wide proba never materializes a huge ``[m, n, C]``.
    A score_fn other than the built-in ``log_loss``/``rmse`` falls back to per-candidate scoring.
    """
    m = len(val_preds)
    P = np.stack([np.asarray(p, dtype=float) for p in val_preds])   # [m, n, C] (clf) | [m, n] (reg)
    ens_sum = np.zeros_like(P[0])                                   # running SUM of selected members
    n = P.shape[1]
    per = max(1, int(256 * 1024**2 / (8 * max(1, P[0].size))))      # candidates/block: cand <= ~256 MB
    if score_fn is log_loss:
        y = np.asarray(y_val).astype(int)
    elif score_fn is rmse:
        y = np.asarray(y_val, dtype=float)
    counts = np.zeros(m, dtype=float)
    picks: list[int] = []
    for it in range(n_iterations):
        scores = np.empty(m, dtype=float)
        for s in range(0, m, per):
            cand = (ens_sum + P[s:s + per]) / (it + 1)             # [b, n, C] | [b, n]
            if score_fn is log_loss:
                c = np.clip(cand, 1e-15, 1.0 - 1e-15)
                c = c / c.sum(axis=2, keepdims=True)
                b = c.shape[0]
                g = c[np.arange(b)[:, None], np.arange(n)[None, :], y[None, :]]   # true-class proba
                scores[s:s + per] = -np.log(g).mean(axis=1)
            elif score_fn is rmse:
                scores[s:s + per] = np.sqrt(((cand - y[None, :]) ** 2).mean(axis=1))
            else:                                                  # generic: arbitrary score_fn
                for jj in range(cand.shape[0]):
                    scores[s + jj] = score_fn(y_val, cand[jj])
        best_j = int(np.nanargmin(scores)) if np.isfinite(scores).any() else 0
        ens_sum = ens_sum + P[best_j]
        counts[best_j] += 1.0
        picks.append(best_j)
    return counts / counts.sum(), picks


# ── apply weights to TEST predictions ──────────────────────────────────────────────────────


def combine(preds, weights) -> np.ndarray:
    """Weighted sum of per-candidate predictions. For proba inputs with weights summing to 1
    the result is itself a valid probability matrix (each input row already sums to 1)."""
    weights = np.asarray(weights, dtype=float)
    out = np.zeros_like(np.asarray(preds[0], dtype=float))
    for w, p in zip(weights, preds):
        if w != 0.0:
            out = out + w * np.asarray(p, dtype=float)
    return out
