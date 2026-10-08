"""ttc: test-time compute for tabular foundation models.

    run.py            YAML config -> one TabArena run
    registry.py       config model keys -> model classes
    benchmarks/       TabArena runner; large-table row retrieval
    models.py         per-task finetuning (full / DiagScale) for TabPFN-3, TabICL v2 and TabFM
    backbones/        TabFM model wrapper and the TabICL/TabFM finetuning loops
    agg_model.py      TabPFN-3 configuration pool + greedy reducer
    agg_ft_model.py   DiagScale adaptation followed by the configuration pool
    aggregation/      configuration search space and reducers
    retrieval/        attention-guided row retrieval helpers
    eval/             Elo against the fixed reference field, paired error changes
"""
from __future__ import annotations

__version__ = "1.0.0"
