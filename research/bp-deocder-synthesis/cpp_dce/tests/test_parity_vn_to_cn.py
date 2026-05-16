"""Step 1 gate: cpp vn_to_cn must equal python _build_vn_to_cn byte-for-byte
for several lifted parity configurations.

Run:
    python cpp_dce/tests/test_parity_vn_to_cn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent           # bp-deocder-synthesis/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))  # cpp_dce/ (for the .pyd next to setup.py)

import pushgp_cpp_dce as cdce
from ldpc_5g import build_parity, _build_vn_to_cn


def check(bgn: int, set_idx: int, zc: int) -> None:
    par_py = build_parity(bgn, set_idx, zc)
    h = cdce.build_parity_handle(par_py)

    # cn_to_vn equality
    cpp_cn = h.cn_to_vn()
    assert len(cpp_cn) == par_py.rows, f"M mismatch {len(cpp_cn)} vs {par_py.rows}"
    for c, row in enumerate(par_py.cn_to_vn):
        assert list(map(int, row.tolist())) == cpp_cn[c], (
            f"cn_to_vn mismatch at c={c}: py={row.tolist()} cpp={cpp_cn[c]}")

    # vn_to_cn equality (insertion order matters!)
    py_vn = _build_vn_to_cn(par_py)
    cpp_vn = h.vn_to_cn()
    assert len(cpp_vn) == par_py.cols, f"N mismatch"
    for v in range(par_py.cols):
        py_list = [(int(c), int(p)) for (c, p) in py_vn[v]]
        cpp_list = [tuple(x) for x in cpp_vn[v]]
        assert py_list == cpp_list, (
            f"vn_to_cn mismatch at v={v}: py={py_list} cpp={cpp_list}")

    print(f"[OK] bgn={bgn} set_idx={set_idx} zc={zc:>3}  "
          f"N={h.N} M={h.M} (cn_to_vn + vn_to_cn byte-equal)")


def main() -> int:
    configs = [
        (2, 1, 2),
        (2, 1, 8),
        (2, 1, 24),
        (2, 2, 4),
        (2, 3, 6),
        (1, 1, 4),
        (1, 2, 16),
        (1, 8, 32),
    ]
    for bgn, set_idx, zc in configs:
        check(bgn, set_idx, zc)
    print("\nStep 1 PASS: all parity handles match python _build_vn_to_cn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
