"""Evaluation utilities for the paper's TabArena protocol.

    elo.py     fixed-68-reference Bradley-Terry Elo, Delta Elo and normalized score
    paired.py  paired error change between two methods (dataset bootstrap, wins)

Both modules are importable as an API and runnable as CLIs (``python -m ttc.eval.elo``,
``python -m ttc.eval.paired``). See ``docs/evaluation.md``.
"""
