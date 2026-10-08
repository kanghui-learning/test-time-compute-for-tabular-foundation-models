"""OpenML dataset loader for the large-row study's OpenML-backed tables (Covertype, Poker-Hand).

External OpenML datasets (not TabArena tasks) -> (X_numeric, y, cat_idx). Categoricals are
ordinal-encoded and everything is coerced to float (mirrors ``ttc.benchmarks.hardgather.load_dataset``
so downstream preprocessing matches). The cache location follows openml's own ``OPENML_CACHE_DIR``
environment variable when set.
"""
from __future__ import annotations

import numpy as np


def load_openml(did: int):
    """OpenML dataset id -> (Xn[float, cats=ordinal codes], y, cat_idx).

    Categorical = OpenML-flagged OR object/category dtype; ordinal codes fit on the full column
    (label-free relabeling, no leakage). Non-numeric leftovers coerced to NaN.
    """
    import openml
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder

    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                    download_features_meta_data=True)
    X, y, cat_ind, _ = d.get_data(target=d.default_target_attribute, dataset_format="dataframe")
    is_cat = [bool(cat_ind[i]) or X.iloc[:, i].dtype == object or str(X.iloc[:, i].dtype) == "category"
              for i in range(X.shape[1])]
    cat_idx = [i for i, c in enumerate(is_cat) if c]
    Xn = X.copy()
    if cat_idx:
        cols = X.columns[cat_idx]
        Xn[cols] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=-1).fit_transform(X[cols].astype(str))
    Xn = Xn.apply(lambda s: pd.to_numeric(s, errors="coerce")).to_numpy(dtype=np.float64)
    return Xn, np.asarray(y), cat_idx
