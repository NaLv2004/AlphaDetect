"""Replay grafted callables from the visualisation log to capture
exception types. Reads the SOURCE block of each graft sample, runs it
on a small probe, and reports the distribution of exception types."""
from __future__ import annotations
import os, sys, re, traceback
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = r"D:\ChannelCoding\RCOM\AlphaDetect\research\algorithm-IR\results\gnn_training\graft_visualizations.preB2v2.log"

def _make_probe(seed=2026):
    rng = np.random.default_rng(seed)
    nr, nt = 16, 16
    H = (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2)
    x = rng.choice([1+1j, 1-1j, -1+1j, -1-1j], size=nt) / np.sqrt(2)
    n = (rng.standard_normal(nr) + 1j * rng.standard_normal(nr)) / np.sqrt(2 * 10)
    y = H @ x + n
    sigma2 = 0.1
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return H, y, sigma2, constellation

with open(LOG, encoding="utf8", errors="replace") as f:
    txt = f.read()

# Each sample: header line "[#N] child=...  host=... donor=..." then
# verdict line; further down a SOURCE block ("```python" ... "```").
samples = re.split(r"^\[#\d+\] child=", txt, flags=re.MULTILINE)
print(f"Found {len(samples)-1} samples")

H, y, sigma2, constellation = _make_probe()

verdicts: Counter = Counter()
exc_types: Counter = Counter()
exc_examples: dict[str, str] = {}
n_no_source = 0

for s in samples[1:]:
    vmatch = re.search(r"verdict=(\w+)", s)
    verdict = vmatch.group(1) if vmatch else "?"
    verdicts[verdict] += 1

    # Try to find the GRAFTED CALLABLE source block.
    # The viz writes "```python\n<source>\n```" for the materialised graft.
    blocks = re.findall(r"```python\n(.*?)\n```", s, flags=re.DOTALL)
    if not blocks:
        n_no_source += 1
        continue
    src = blocks[-1]  # last block tends to be the grafted child source
    if "def " not in src:
        n_no_source += 1
        continue
    fn_name = re.search(r"def\s+(\w+)\s*\(", src)
    if not fn_name:
        n_no_source += 1
        continue
    fn_name = fn_name.group(1)
    g = {"np": np}
    try:
        exec(compile(src, "<graft>", "exec"), g)
        fn = g[fn_name]
    except Exception as e:
        key = f"COMPILE/{type(e).__name__}"
        exc_types[key] += 1
        exc_examples.setdefault(key, f"{e!r}")
        continue
    try:
        out = fn(H, y, sigma2, constellation)
        # OK
    except Exception as e:
        key = f"RUNTIME/{type(e).__name__}"
        exc_types[key] += 1
        exc_examples.setdefault(key, f"{e!r}")

print("\nVerdicts:", dict(verdicts))
print(f"\nSamples without parsable source: {n_no_source}")
print("\nException type distribution (replay):")
for k, v in exc_types.most_common():
    print(f"  {v:4d}  {k}  -- e.g.: {exc_examples[k][:140]}")
