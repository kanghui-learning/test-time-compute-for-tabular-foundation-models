"""Model-space signal capture for frozen TabPFN-3 retrieval.

  * host_probe.py -- the only module that touches TabPFN-3 internals: streams the last ICL layer's
                     query->context softmax attention (mean over heads) as per-query top-k indices,
                     plus full-map attention and embedding captures for small-scale checks.

The large-table model-aware retrieval pipeline that uses it is ``ttc.benchmarks.row_modelaware``.
"""
from __future__ import annotations
