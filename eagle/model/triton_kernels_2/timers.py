import time
from contextlib import contextmanager
from typing import Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

_TIMERS: Dict[str, float] = {}
_COUNTS: Dict[str, int] = {}
_ENABLED: bool = False


def enable_timers(enable: bool = True) -> None:
    global _ENABLED
    _ENABLED = enable


def reset_timers() -> None:
    _TIMERS.clear()
    _COUNTS.clear()


def get_timers() -> Dict[str, float]:
    return dict(_TIMERS)


def get_counts() -> Dict[str, int]:
    return dict(_COUNTS)


def inc(name: str, by: int = 1) -> None:
    _COUNTS[name] = _COUNTS.get(name, 0) + by


def _cuda_sync():
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


@contextmanager
def time_block(name: str, enabled: Optional[bool] = None):
    """
    Time a code block with cuda synchronize before/after for accurate GPU timing.
    Accumulates into a global timer map by name.
    """
    if enabled is None:
        enabled = _ENABLED
    if not enabled:
        yield
        return
    _cuda_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _cuda_sync()
        dt = time.perf_counter() - t0
        _TIMERS[name] = _TIMERS.get(name, 0.0) + dt


def print_timers(title: str = "EAGLE Triton timers", digits: int = 4) -> None:
    if not _TIMERS and not _COUNTS:
        print(f"{title}: no measurements")
        return
    print(f"{title}:")
    total = 0.0
    for k in sorted(set(list(_TIMERS.keys()) + list(_COUNTS.keys()))):
        v = _TIMERS.get(k, 0.0)
        c = _COUNTS.get(k, 0)
        total += v
        if c:
            print(f"  {k}: {round(v, digits)} s, calls: {c}")
        else:
            print(f"  {k}: {round(v, digits)} s")
    print(f"  total_eagle: {round(total, digits)} s")