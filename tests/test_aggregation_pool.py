"""The seeded configuration pool must match the reference lists used in the paper."""
import json
from pathlib import Path

import pytest

from ttc.aggregation import search_space as S

POOL = Path(__file__).resolve().parents[1] / "configs" / "aggregation_pool"


@pytest.mark.parametrize("name,problem_type", [("classification", "binary"),
                                               ("classification", "multiclass"),
                                               ("regression", "regression")])
def test_pool_matches_reference(name, problem_type):
    ref = json.loads((POOL / f"{name}.json").read_text())
    native, raw = S.sample(ref["n_configs"], ref["seed"], problem_type)
    assert json.loads(json.dumps(native, default=str)) == ref["configs_native"]
    assert json.loads(json.dumps(raw, default=str)) == ref["configs_raw"]


def test_smaller_pool_is_prefix():
    for pt in ("binary", "regression"):
        assert S.sample(4, 0, pt)[0] == S.sample(96, 0, pt)[0][:4]
