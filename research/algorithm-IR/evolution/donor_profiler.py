"""Lightweight profiler for donor-side region sampling.

Usage: set env var ALPHADETECT_PROFILE_DONOR=1, run train_gnn.py.
Stats are printed at the end of each _propose_pairs call.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

_ENABLED = os.environ.get("ALPHADETECT_PROFILE_DONOR") in ("1", "true", "yes")

if not _ENABLED:
    # Stub — zero-cost no-ops when profiling is off.
    def _stub(*a, **kw): pass
    class _StubCtx:
        def __enter__(self): pass
        def __exit__(self, *a): pass

    profile_donor = _stub
    profile_donor_ctx = lambda name: _StubCtx()
    donor_profile_report = lambda: ""
else:
    # ── real implementation ────────────────────────────────────────
    _timers: dict[str, list[float]] = defaultdict(list)
    _counters: dict[str, int] = defaultdict(int)

    def profile_donor(name: str, dt: float, **extra) -> None:
        """Record a timing sample."""
        _timers[name].append(dt)
        for k, v in extra.items():
            key = f"{name}:{k}"
            _counters[key] += v if isinstance(v, int) else 1

    class _Ctx:
        def __init__(self, name):
            self.name = name
            self.t0 = 0.0
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *a):
            dt = time.perf_counter() - self.t0
            _timers[self.name].append(dt)

    def profile_donor_ctx(name: str) -> _Ctx:
        return _Ctx(name)

    def donor_profile_report() -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("DONOR SAMPLING PROFILE")
        lines.append("=" * 70)
        for name in sorted(_timers):
            vs = _timers[name]
            if not vs:
                continue
            total = sum(vs)
            n = len(vs)
            mean = total / n
            mx = max(vs)
            mn = min(vs)
            lines.append(
                f"  {name:50s}  n={n:5d}  total={total:8.2f}s  "
                f"mean={mean*1000:6.1f}ms  max={mx*1000:6.0f}ms  min={mn*1000:5.0f}ms"
            )
        lines.append("---")
        for key in sorted(_counters):
            lines.append(f"  {key:50s}  count={_counters[key]}")
        lines.append("=" * 70)
        return "\n".join(lines)


def reset_donor_profile() -> None:
    """Clear accumulated stats (called at start of _propose_pairs)."""
    if _ENABLED:
        _timers.clear()
        _counters.clear()
