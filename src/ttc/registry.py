"""Model registry: config key -> AutoGluon/TabArena model class and leaderboard display name.

Stdlib-only, so it can be queried without the heavy environment:

    python -m ttc.registry --list

``class_name`` is either a bare name resolved from ``tabarena.models`` or an explicit
``"module:Class"`` path for classes owned by this package.
"""
from __future__ import annotations

import argparse
from typing import NamedTuple


class ModelSpec(NamedTuple):
    class_name: str  # class in tabarena.models, or "module:Class"
    display: str  # leaderboard display name (results are reported as "Demo_<display>")
    hyperparameters: dict = {}  # default model hyperparameters (read-only)


MODELS: dict[str, ModelSpec] = {
    # Frozen TabPFN-3 as shipped in TabArena; used for the native-view runs (n_estimators=V).
    "tabpfn3": ModelSpec("TabPFN3Model", "TabPFN3"),
    # TabPFN-3 with per-task test-time finetuning (full / DiagScale). ft_epochs=0 = frozen baseline.
    "tabpfn3-ft": ModelSpec("ttc.models:FinetunedTabPFN3Model", "FT-TabPFN3"),
    # TabPFN-3 configuration pool + out-of-fold greedy (Caruana) reducer.
    "tabpfn3-agg": ModelSpec("ttc.agg_model:AggregationTabPFN3Model", "Agg-TabPFN3"),
    # DiagScale adaptation followed by the configuration pool + greedy reducer.
    "tabpfn3-agg-ft": ModelSpec("ttc.agg_ft_model:FinetunedAggregationTabPFN3Model", "AggFT-TabPFN3"),
    # TabICL v2 with per-task test-time finetuning (full / DiagScale). ft_epochs=0 = frozen baseline.
    "tabiclv2-ft": ModelSpec("ttc.models:FinetunedTabICLModel", "FT-TabICLv2"),
    # Google TabFM (PyTorch backend, bf16, cell-budget feature cap): the frozen reference model.
    "tabfm": ModelSpec("ttc.backbones.tabfm:TabFMModel", "TabFM"),
    # TabFM with per-task test-time finetuning (full / DiagScale).
    "tabfm-ft": ModelSpec("ttc.models:FinetunedTabFMModel", "FT-TabFM"),
}


def resolve_model_class(key: str):
    """Import and return the model class for a registry key (imports the heavy dependencies)."""
    import importlib

    class_name = MODELS[key].class_name
    module, _, cls = class_name.rpartition(":")
    return getattr(importlib.import_module(module or "tabarena.models"), cls)


def main() -> None:
    parser = argparse.ArgumentParser(description="List the model keys usable in configs.")
    parser.add_argument("--list", action="store_true", help="list all model keys")
    parser.parse_args()
    for k in sorted(MODELS):
        spec = MODELS[k]
        print(f"{k:<16} {spec.display:<16} {spec.class_name}")


if __name__ == "__main__":
    main()
