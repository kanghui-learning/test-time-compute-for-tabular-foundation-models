"""Wall-clock cost ledger for the aggregation models.

Each phase is timed with ``torch.cuda.synchronize()`` around it, so asynchronous GPU work is
attributed to the phase that launched it. ``fit`` (out-of-fold member predictions, paid once per
task) is kept separate from the per-inference phases.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class CostEntry:
    """One timed slice: which operator, which phase, how long, on what device."""

    operator: str
    phase: str  # "fit" (setup) | "forward" (amortized) | "postprocess"
    wall_s: float
    device: str
    n_members: int | None = None
    n_forward: int | None = None  # reserved: neural-FLOP profile (later)
    n_backward: int | None = None
    extra: dict = field(default_factory=dict)  # operator-specific (FT: epochs_run, best_epoch, ...)

    def to_dict(self) -> dict:
        d = {
            "operator": self.operator,
            "phase": self.phase,
            "wall_s": round(self.wall_s, 4),
            "device": self.device,
        }
        for k in ("n_members", "n_forward", "n_backward"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.extra:
            d["extra"] = self.extra
        return d


def _cuda_sync(device: str) -> None:
    """Block until pending CUDA work on `device` finishes, so wall-clock isn't undercounted.

    Syncs the operator's OWN device (parsed from the string), not the process current device —
    matters once runs span >1 GPU. Warns (rather than silently passing) if torch/CUDA is absent,
    so an accidental non-GPU run doesn't yield silently-uncalibrated timings.
    """
    if not (isinstance(device, str) and device.startswith("cuda")):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        idx = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
        torch.cuda.synchronize(device=idx)
    except Exception as e:
        import warnings

        warnings.warn(f"_cuda_sync({device!r}) skipped: {e!r} — wall_s may be uncalibrated",
                      RuntimeWarning, stacklevel=2)


@dataclass
class CostLedger:
    """Append-only log of CostEntry, with setup/amortized/total roll-ups."""

    entries: list[CostEntry] = field(default_factory=list)

    @contextmanager
    def measure(self, operator: str, phase: str, device: str, **fields):
        """Time a block. Yields the (not-yet-finalized) CostEntry so the caller can annotate it
        (e.g. set ``entry.extra`` / ``entry.n_members``) before wall_s is stamped and it is
        appended. CUDA-synchronized on both ends when the device is a GPU."""
        _cuda_sync(device)
        t0 = time.perf_counter()
        entry = CostEntry(operator=operator, phase=phase, wall_s=0.0, device=str(device), **fields)
        try:
            yield entry
        finally:
            _cuda_sync(device)
            entry.wall_s = time.perf_counter() - t0
            self.entries.append(entry)

    def setup_s(self) -> float:
        return sum(e.wall_s for e in self.entries if e.phase == "fit")

    def amortized_s(self) -> float:
        return sum(e.wall_s for e in self.entries if e.phase != "fit")

    def total_s(self) -> float:
        return sum(e.wall_s for e in self.entries)

    def to_records(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]
