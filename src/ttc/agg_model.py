"""TabPFN-3 configuration pool with an out-of-fold greedy reducer.

fit(X, y):
  * sample ``n_configs`` configurations (``ttc.aggregation.search_space``); configuration 0 is the
    default;
  * build each configuration's validation predictions, either
      - ``val_folds >= 2``: k-fold out-of-fold predictions over the full training split (the
        paper uses 8 folds; the fold count is reduced when a class has fewer examples), or
      - ``val_folds = 0``: a stratified 90/10 holdout;
    validation predictions use ``n_est_val`` views (2).

predict(X_test):
  * fit each configuration on the full training split and predict the test split with
    ``n_est_test`` views (8);
  * a configuration that fails at validation or test time is replaced by configuration 0 at both;
  * return the greedy-selection (Caruana) combination whose weights were chosen on the
    validation predictions.

When the runner sets ``ft_log_dir``, per-cell member predictions are written to
``<eval>/agg_preds/<dataset>_f<fold>_r<repeat>.npz`` and per-configuration timings to
``<eval>/agg_cost.jsonl``, so other reducers or smaller pools can be evaluated afterwards.

Configured by an ``aggregation`` dict under ``hyperparameters``; ``ft_*`` keys and ``aggregation``
are stripped before the remaining keyword arguments reach the TabPFN estimator.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from tabarena.models import TabPFN3Model

from ttc.models import _current_task_context


def _current_runner():
    """The tabarena ExperimentRunner on the call stack (has ``task``/``fold``/``repeat``).

    Used at predict time to reload THIS cell's exact test split in the task's deterministic row
    order — AutoGluon reorders test rows internally and resets the index before ``_predict_proba``,
    so we sidestep it entirely rather than try to recover the (unrecoverable) permutation."""
    import inspect

    for fi in inspect.stack():
        obj = fi.frame.f_locals.get("self")
        if isinstance(getattr(obj, "task_name", None), str) and hasattr(obj, "task"):
            return obj
    raise RuntimeError("AggregationTabPFN3Model: no ExperimentRunner on the stack at predict time")


def _current_exec_model():
    """The tabarena AbstractExecModel on the call stack — holds the FITTED feature generator
    (``transform_X``) and ``label_cleaner``. We reuse them to process this cell's test split
    through the EXACT same pipeline the model was fit under (raw features would mismatch)."""
    import inspect

    for fi in inspect.stack():
        obj = fi.frame.f_locals.get("self")
        if hasattr(obj, "transform_X") and hasattr(obj, "label_cleaner"):
            return obj
    raise RuntimeError("AggregationTabPFN3Model: no AbstractExecModel on the stack at predict time")


def _fallback_to_config0(preds: list, times: list, failed: list, i: int, stage: str) -> None:
    """Config `i` crashed at `stage` (val / OOF val / test): reuse config 0 — the base anchor — so the
    parallel preds/times/failed lists stay aligned. MUST be called from inside an `except` block: if
    config 0 ITSELF failed (``preds`` empty) the bare ``raise`` re-raises the real error instead of
    masking it as an IndexError on ``preds[0]``. Appends config 0's prediction by REFERENCE — it is
    only read downstream (combine/caruana never mutate it) and the dump's ``np.stack`` copies it."""
    if not preds:  # config 0 (the base anchor) itself failed — nothing to fall back to
        raise
    warnings.warn(f"[agg] config {i} failed at {stage}; falling back to config 0",
                  RuntimeWarning, stacklevel=3)
    preds.append(preds[0])  # reference (not np.array(...)): read-only downstream, dump copies
    times.append(0.0)
    failed.append(True)


class AggregationTabPFN3Model(TabPFN3Model):
    ag_key = "TTC-AGG-TABPFN3"
    ag_name = "TTC-Agg-TabPFN3"

    # n_est_val/test = the FT (2 / 8) footing; n_iterations = TabArena EnsembleSelection default.
    # val_folds=0 -> the 90/10 holdout (default, FT-comparable); val_folds>=2 -> k-fold OOF over the
    # FULL train (TabArena-aligned, e.g. 8 = TabArena's num_bag_folds default).
    DEFAULTS = {"n_configs": 40, "seed": 0, "n_est_val": 2, "n_est_test": 8,
                "val_ratio": 0.1, "n_iterations": 40, "val_folds": 0}

    # ── hyperparameter routing ──
    def _get_ft_params(self) -> dict:
        """Presence of this method triggers tabarena's ft_log_dir / ft_run_label injection."""
        return {k: v for k, v in super()._get_model_params().items() if k.startswith("ft_")}

    def _get_model_params(self) -> dict:
        return {
            k: v for k, v in super()._get_model_params().items()
            if not k.startswith("ft_") and k != "aggregation"
        }

    def _agg_params(self) -> dict:
        raw = super()._get_model_params().get("aggregation") or {}
        return {**self.DEFAULTS, **raw}

    # ── device management: this model holds NO single `self.model` (estimators are built
    #    per-config in fit/predict), so the base get_device/_set_device (which read self.model)
    #    would hit None. Drive device off the stored string instead. ──
    def get_device(self) -> str:
        dev = getattr(self, "_agg_device", None) or "cpu"
        return dev.split(":")[0] if isinstance(dev, str) else "cpu"

    def _set_device(self, device: str) -> None:
        self._agg_device = device

    # ── build one TabPFN estimator for a native config at a given view count ──
    def _build_estimator(self, native_cfg: dict, n_estimators: int):
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        is_clf = self.problem_type in ("binary", "multiclass")
        # base defaults (e.g. ignore_pretraining_limits) then our fixed kwargs then the config.
        kwargs = dict(self._get_model_params())  # control keys already stripped
        kwargs.update(
            model_path=self._get_model_checkpoint(),
            device=self._agg_device,
            n_jobs=self._agg_num_cpus,
            categorical_features_indices=self._categorical_indices,
            n_estimators=int(n_estimators),
        )
        kwargs[self.seed_name] = self.fixed_random_state  # 'random_state'
        kwargs.update(native_cfg)  # softmax_temperature / balance_probabilities / inference_config
        # auto_scale_n_estimators left at the TabPFN default (True) — matches base & FT inference.
        cls = TabPFNClassifier if is_clf else TabPFNRegressor
        return cls(**kwargs)

    # ── fit: sample configs, carve val, run per-config val forwards ──
    def _fit(self, X, y, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        from ttc.aggregation import search_space as S

        p = self._agg_params()
        self._agg_p = p
        is_clf = self.problem_type in ("binary", "multiclass")

        device = self._resolve_tabpfn_device(num_gpus=num_gpus)
        if not isinstance(device, str):
            device = device[0]
        self._agg_device, self._agg_num_cpus = device, num_cpus

        X = self.preprocess(X, y=y, is_train=True)  # detect categoricals; consistent with predict
        self._X_full, self._y_full = X, y            # full train context for test deployment

        native, raws = S.sample(p["n_configs"], p["seed"], self.problem_type)
        self._agg_native, self._agg_raw = native, raws

        # validation signal for the post-hoc reductions: k-fold OOF over the FULL train
        # (val_folds>=2, TabArena-aligned) OR the 90/10 holdout (default, FT-comparable).
        n_folds = int(p.get("val_folds", 0) or 0)
        if n_folds == 1 or n_folds < 0:  # 0 = holdout, >=2 = k-fold OOF; 1/negative is meaningless
            raise ValueError(f"aggregation.val_folds={n_folds} is invalid: use 0 for the 90/10 "
                             "holdout (default) or >=2 for k-fold OOF")
        if n_folds >= 2:
            vp, vt, vf, y_enc, classes = self._val_oof(X, y, native, p, is_clf, n_folds)
        else:
            vp, vt, vf, y_enc, classes = self._val_holdout(X, y, native, p, is_clf)
        self._val_proba, self._val_times, self._val_failed = vp, vt, vf
        self._y_val, self._classes = y_enc, classes
        return self

    # ── validation scheme A: 90/10 holdout (default; FT-comparable) ──
    def _val_holdout(self, X, y, native, p, is_clf):
        """Each config fits the 90% fit-context and predicts the 10% val (SAME stratified split rule
        as FinetunedTabPFN._get_train_val_split). Returns (val_preds, val_times, failed, y_enc,
        classes); y_enc = the encoded 10% val labels, val_preds[i] = config i's prediction on it."""
        import numpy as np
        from sklearn.model_selection import train_test_split

        from ttc.cost import CostLedger

        device = self._agg_device
        y_arr = np.asarray(y)
        test_size = int(len(y_arr) * p["val_ratio"])
        if is_clf:
            test_size = max(test_size, int(np.unique(y_arr).size))
        Xfit, Xval, yfit, yval = train_test_split(
            X, y, test_size=test_size, random_state=self.fixed_random_state,
            stratify=(y if is_clf else None),
        )

        ledger = CostLedger()
        val_preds: list = []
        val_times: list = []
        failed: list = []
        classes = None
        for i, cfg in enumerate(native):
            try:
                est = self._build_estimator(cfg, p["n_est_val"])
                with ledger.measure(f"config{i}", "fit", device, n_members=p["n_est_val"]) as e:
                    est.fit(Xfit, yfit)
                    pv = est.predict_proba(Xval) if is_clf else est.predict(Xval)
                if is_clf and classes is None:
                    classes = np.asarray(est.classes_)
                val_preds.append(np.asarray(pv, dtype=float))
                val_times.append(round(e.wall_s, 4))
                failed.append(False)
                del est
            except Exception:  # one bad config must not kill the whole cell
                _fallback_to_config0(val_preds, val_times, failed, i, "val")

        if is_clf:
            col = {c: j for j, c in enumerate(classes)}
            y_enc = np.array([col.get(v, 0) for v in np.asarray(yval)], dtype=int)
        else:
            y_enc = np.asarray(yval, dtype=float)
            classes = None
        return val_preds, val_times, failed, y_enc, classes

    # ── validation scheme B: k-fold cross-validated OOF over the FULL train (TabArena-aligned) ──
    def _val_oof(self, X, y, native, p, is_clf, n_folds):
        """TabArena-aligned validation (mirrors num_bag_folds=8): each config produces OUT-OF-FOLD
        predictions over the FULL train via k-fold CV (100% coverage) instead of a 10% holdout, so
        the val-greedy selection / ensemble sees a far stronger signal. Each config -> n_folds fits
        on (k-1)/k of the train, predicting its held-out fold; per-fold predict_proba columns are
        re-aligned to the GLOBAL class order (a fold whose train misses a rare class -> 0 there).
        The fold partition is FIXED across all configs (so OOF rows are comparable). Returns
        (val_preds, val_times, failed, y_enc, classes); val_preds[i] is config i's OOF over the full
        train and y_enc is the full-train labels (encoded) — both span 100% of train."""
        import numpy as np
        from sklearn.model_selection import KFold, StratifiedKFold

        from ttc.cost import CostLedger

        device = self._agg_device
        y_np = np.asarray(y)
        n = len(y_np)
        if is_clf:
            classes, counts = np.unique(y_np, return_counts=True)
            n_folds = max(2, min(n_folds, int(counts.min())))  # adapt down to rarest class (stratify-safe)
            col = {c: j for j, c in enumerate(classes)}
            folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                         random_state=self.fixed_random_state).split(np.zeros(n), y_np))
        else:
            classes = None
            n_folds = max(2, min(n_folds, n))  # adapt down to dataset size (KFold needs n_splits<=n)
            folds = list(KFold(n_splits=n_folds, shuffle=True,
                               random_state=self.fixed_random_state).split(np.zeros(n)))
        p["effective_val_folds"] = int(n_folds)  # post-clamp folds -> dumped meta is truthful (not the request)

        def _x(idx):
            return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

        ledger = CostLedger()
        val_preds: list = []
        val_times: list = []
        failed: list = []
        for i, cfg in enumerate(native):
            try:
                oof = np.zeros((n, len(classes)), dtype=float) if is_clf else np.zeros(n, dtype=float)
                with ledger.measure(f"config{i}", "fit", device,
                                    n_members=p["n_est_val"] * n_folds) as e:  # k folds x n_est_val
                    for tr_idx, va_idx in folds:  # SAME folds for every config
                        est = self._build_estimator(cfg, p["n_est_val"])
                        est.fit(_x(tr_idx), y_np[tr_idx])
                        if is_clf:
                            proba = np.asarray(est.predict_proba(_x(va_idx)), dtype=float)
                            for c_col, c in enumerate(np.asarray(est.classes_)):
                                oof[va_idx, col[c]] = proba[:, c_col]  # align to GLOBAL class order
                        else:
                            oof[va_idx] = np.asarray(est.predict(_x(va_idx)), dtype=float)
                        del est
                val_preds.append(oof)
                val_times.append(round(e.wall_s, 4))
                failed.append(False)
            except Exception:  # one bad config must not kill the whole cell
                _fallback_to_config0(val_preds, val_times, failed, i, "OOF val")

        y_enc = np.array([col[v] for v in y_np], dtype=int) if is_clf else y_np.astype(float)
        return val_preds, val_times, failed, y_enc, classes

    # ── predict: per-config forwards on the harness's (shuffled) X, dump preds + aligned y_test ──
    def _predict_proba(self, X, **kwargs):
        import numpy as np

        from tabarena.benchmark.exec_models.base import _make_perm

        from ttc.aggregation import reduce as R
        from ttc.cost import CostLedger

        p = self._agg_p
        is_clf = self.problem_type in ("binary", "multiclass")
        # Predict on the X AutoGluon hands us — already through the exec model's feature generator
        # (so features match training; raw features would silently degrade regression). The exec
        # model first SHUFFLES the test rows (shuffle_test, deterministic _make_perm(shuffle_seed))
        # and inverts the shuffle on the returned output. So our per-config test_proba is in the
        # shuffled order; we reproduce that permutation to align y_test to it (no fragile join).
        X = self.preprocess(X, **kwargs)

        em = _current_exec_model()
        runner = _current_runner()
        _, _, _, yte_raw = runner.task.get_train_test_split(
            fold=runner.fold, repeat=runner.repeat, sample=getattr(runner, "sample", 0))
        y_te = np.asarray(em.label_cleaner.transform(yte_raw))  # encoded, task (te) order
        if getattr(em, "shuffle_test", False):
            perm, _inv = _make_perm(len(y_te), seed=getattr(em, "shuffle_seed", 0))
            y_te = y_te[perm]  # -> shuffled order, matching X / test_proba
        y_test = y_te.astype(int) if is_clf else y_te.astype(float)

        ledger = CostLedger()
        test_preds: list = []
        test_times: list = []
        failed: list = []
        for i, cfg in enumerate(self._agg_native):
            try:
                est = self._build_estimator(cfg, p["n_est_test"])
                with ledger.measure(f"config{i}", "forward", self._agg_device,
                                    n_members=p["n_est_test"]) as e:
                    est.fit(self._X_full, self._y_full)
                    pt = est.predict_proba(X) if is_clf else est.predict(X)
                test_preds.append(np.asarray(pt, dtype=float))
                test_times.append(round(e.wall_s, 4))
                failed.append(False)
                del est
            except Exception:
                _fallback_to_config0(test_preds, test_times, failed, i, "test")

        # A config that failed at EITHER val or test has no valid prediction there; substitute
        # config 0 (= base, a no-op duplicate in the pool) at BOTH stages, so the val-derived Caruana
        # weight lands on config 0's test rows, not a stale/mismatched prediction. Config 0 is
        # guaranteed valid (its failure re-raises above). Keeps val/test/weights/dump consistent.
        combined_failed = [bool(vf or tf) for vf, tf in zip(self._val_failed, failed)]
        if any(combined_failed):
            self._val_proba = [self._val_proba[0] if bad else vp
                               for bad, vp in zip(combined_failed, self._val_proba)]
            test_preds = [test_preds[0] if bad else tp
                          for bad, tp in zip(combined_failed, test_preds)]
            # the substituted slot now holds config 0's prediction (a no-op duplicate), so its
            # marginal test cost is 0 — zero the time too, keeping the per-config cost ledger aligned
            # to the prediction it stores (matches the test-failure fallback, which records 0.0).
            test_times = [0.0 if bad else tt for bad, tt in zip(combined_failed, test_times)]
        self._val_failed = combined_failed

        self._dump(test_preds, test_times, combined_failed, y_test)

        # Board headline = Caruana(full pool); returned in the shuffled order, which the exec model
        # inverts back to the caller's order — so the board scores the real tuned+ensemble.
        sf = R.score_fn_for(self.problem_type)
        w, _ = R.caruana_weights(self._val_proba, self._y_val, sf, n_iterations=p["n_iterations"])
        combined = R.combine(test_preds, w)
        return self._convert_proba_to_unified_form(combined) if is_clf else combined

    def _dump(self, test_preds, test_times, test_failed, y_test) -> None:
        import numpy as np

        log_dir = self._get_ft_params().get("ft_log_dir")
        if not log_dir:
            return
        dataset, fold, repeat = _current_task_context()
        out = Path(log_dir) / "agg_preds"
        out.mkdir(parents=True, exist_ok=True)
        cell = f"{(dataset or 'unknown').replace('/', '_')}_f{fold}_r{repeat}"
        meta = {"dataset": dataset, "fold": fold, "repeat": repeat,
                "problem_type": self.problem_type, **self._agg_p}
        np.savez_compressed(
            out / f"{cell}.npz",
            # float32 (arrays are built float64): halves the dump; ample for post-hoc log_loss/rmse
            # and the rank-based Elo. In-memory float64 (used for the actual board weights) untouched.
            val_proba=np.stack(self._val_proba).astype(np.float32),   # [K, n_val, C] (clf) | [K, n_val] (reg)
            test_proba=np.stack(test_preds).astype(np.float32),        # [K, n_test, C] | [K, n_test]
            y_val=self._y_val,                       # encoded, aligned to val_proba
            y_test=np.asarray(y_test),               # encoded, aligned to test_proba (task order)
            classes=(self._classes if self._classes is not None else np.array([])),
            val_times=np.asarray(self._val_times, dtype=float),
            test_times=np.asarray(test_times, dtype=float),
            val_failed=np.asarray(self._val_failed, dtype=bool),
            test_failed=np.asarray(test_failed, dtype=bool),
            raw_configs=np.array(json.dumps(self._agg_raw)),
            meta=np.array(json.dumps(meta)),
        )
        rec = {"dataset": dataset, "fold": fold, "repeat": repeat,
               "problem_type": self.problem_type,
               "val_s": round(float(np.sum(self._val_times)), 3),
               "test_s": round(float(np.sum(test_times)), 3),
               "n_failed_val": int(np.sum(self._val_failed)),
               "n_failed_test": int(np.sum(test_failed)),
               "per_config": [{"config_id": i, "val_s": self._val_times[i],
                               "test_s": float(test_times[i])} for i in range(len(test_times))]}
        with open(Path(log_dir) / "agg_cost.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
