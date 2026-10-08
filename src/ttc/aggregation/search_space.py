"""Configuration search space for the TabPFN-3 configuration pool.

Taken from TabArena's tuned TabPFN-v2.5 search space (``tabarena/models/tabpfnv2_5/hpo.py``)
without the v2.5-only ``zip_model_path`` (TabPFN-3 ships one checkpoint and an empty search space
of its own). Every remaining setting is a preprocessing or inference option that TabPFN-3 accepts
natively through ``softmax_temperature``, ``balance_probabilities`` and ``inference_config``.

``sample(n, seed, problem_type)`` returns ``n`` configurations as native TabPFN keyword arguments,
plus their path-key form for logging. Two invariants:

  * configuration 0 is always the empty default ``{}``, i.e. the stock TabPFN-3 settings;
  * draws are a fixed random search under ``seed``, so the first K configurations of a pool are
    themselves a valid K-configuration draw (``configs/aggregation_pool/`` lists the seed-0 pool).

The number of views (``n_estimators``) is fixed by the caller, not sampled here.

The path-key to native translation mirrors TabArena's v2.5 model wrapper
(``models/tabpfnv2_5/model.py``): ``preprocessing/{scaling,global,categoricals,append_original}``
collapse into ``inference_config["PREPROCESS_TRANSFORMS"]`` (one transform per scaler) and
``inference_config/X`` keys become ``inference_config[X]``. TabPFN-3's own wrapper passes
hyperparameters straight to ``TabPFNClassifier``, so the translation is done here.
"""
from __future__ import annotations

import numpy as np

# ── search space (tabpfnv2_5/hpo.py, minus zip_model_path) — {path-key: list of choices} ──
SPACE: dict[str, list] = {
    "softmax_temperature": [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5],
    "balance_probabilities": [True, False],  # CLF only
    "inference_config/OUTLIER_REMOVAL_STD": [3, 6, 12],
    "inference_config/POLYNOMIAL_FEATURES": ["no", 25],
    "inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS": [  # REG only
        [None],
        [None, "safepower"],
        ["safepower"],
        ["kdi_alpha_0.3"],
        ["kdi_alpha_1.0"],
        ["kdi_alpha_3.0"],
        ["quantile_uni"],
    ],
    "preprocessing/scaling": [
        ["none"],
        ["quantile_uni_coarse"],
        ["quantile_norm_coarse"],
        ["kdi_uni"],
        ["kdi_alpha_0.3"],
        ["kdi_alpha_3.0"],
        ["safepower", "quantile_uni"],
        ["none", "quantile_uni_coarse"],
        ["squashing_scaler_default", "quantile_uni_coarse"],
        ["squashing_scaler_default"],
    ],
    "preprocessing/categoricals": ["numeric", "onehot", "none"],
    "preprocessing/append_original": [True, False],
    "preprocessing/global": [None, "svd", "svd_quarter_components"],
}

_CLF_ONLY = frozenset({"balance_probabilities"})
_REG_ONLY = frozenset({"inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS"})


def _is_clf(problem_type: str) -> bool:
    return problem_type in ("binary", "multiclass")


def _active_keys(problem_type: str) -> list[str]:
    drop = _REG_ONLY if _is_clf(problem_type) else _CLF_ONLY
    return [k for k in SPACE if k not in drop]


def _choice(rng: np.random.Generator, options: list):
    """Index-based pick (rng.choice mishandles list-valued options like the scaling lists)."""
    return options[int(rng.integers(len(options)))]


def _raw_key(raw: dict) -> str:
    """Order-independent identity of a raw config, for de-duplication."""
    return repr(sorted((k, repr(v)) for k, v in raw.items()))


def to_native(raw: dict, problem_type: str) -> dict:
    """Translate one path-key raw config into native TabPFN kwargs (see module docstring)."""
    is_clf = _is_clf(problem_type)
    native: dict = {}
    if "softmax_temperature" in raw:
        native["softmax_temperature"] = raw["softmax_temperature"]
    if is_clf and "balance_probabilities" in raw:
        native["balance_probabilities"] = raw["balance_probabilities"]

    inference_config: dict = {}
    for k, v in raw.items():
        if k.startswith("inference_config/"):
            inference_config[k.split("/", 1)[1]] = v
    if "preprocessing/scaling" in raw:
        # Mirror v2.5 model.py EXACTLY: it builds the transforms with hps.pop(...), so the
        # global / categoricals / append_original values are CONSUMED by the first scaler and the
        # 2nd+ scalers fall back to defaults (None / "numeric" / True). A non-mutating .get() here
        # would wrongly copy those values onto every transform of a multi-scaler config.
        glob = raw.get("preprocessing/global", None)
        cat = raw.get("preprocessing/categoricals", "numeric")
        app = raw.get("preprocessing/append_original", True)
        transforms = []
        for first, scaler in enumerate(raw["preprocessing/scaling"]):
            transforms.append({
                "name": scaler,
                "global_transformer_name": glob if first == 0 else None,
                "categorical_name": cat if first == 0 else "numeric",
                "append_original": app if first == 0 else True,
            })
        inference_config["PREPROCESS_TRANSFORMS"] = transforms
    # task-specific key already excluded by _active_keys, but guard against a hand-passed raw.
    if is_clf:
        inference_config.pop("REGRESSION_Y_PREPROCESS_TRANSFORMS", None)
    if inference_config:
        native["inference_config"] = inference_config
    return native


def sample(n: int, seed: int, problem_type: str) -> tuple[list[dict], list[dict]]:
    """Return ``(native_configs, raw_configs)``, both length ``n``, config 0 = stock default.

    ``native_configs[i]`` splats into TabPFNClassifier/Regressor; ``raw_configs[i]`` is the
    path-key form kept for logging / reproducibility. Draws are unique under ``seed``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 (got {n})")
    keys = _active_keys(problem_type)
    rng = np.random.default_rng(seed)
    raws: list[dict] = [{}]  # config 0 = stock default (anchors the curve at base)
    seen: set[str] = {_raw_key({})}
    cap = 200 * n  # generous; the grid is ~thousands, so 40 unique is trivial
    attempts = 0
    while len(raws) < n and attempts < cap:
        attempts += 1
        raw = {k: _choice(rng, SPACE[k]) for k in keys}
        key = _raw_key(raw)
        if key in seen:
            continue
        seen.add(key)
        raws.append(raw)
    if len(raws) < n:
        raise RuntimeError(
            f"could only draw {len(raws)} unique configs of {n} after {attempts} attempts"
        )
    native = [to_native(r, problem_type) for r in raws]
    return native, raws
