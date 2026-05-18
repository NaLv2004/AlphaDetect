"""One-call poll report for an in-progress evolution run.

For each call:
  * Print per-gen population distribution (best/mean/median/worst, valid count).
  * Show top-K of the latest gen with LIVE math expressions (V2C + C2V) via TraceVM.
  * If champion fitness improved vs prev poll, flag NEW CHAMPION + dump RAW.
  * Provide a mechanical mathematical interpretation of each LIVE expression
    (substitute FVec.PopBack -> m_i (incoming neighbor msg), L_v -> channel LLR,
    EvoConstK -> K_k, recognize sum-product / min-sum / OMS / NMS patterns).
  * Always re-print the OMS baseline (fit=-1.0951) for reference.

Usage:
    python -B experiments/poll_report.py <run_name> [--top-k 5] [--max-chars 360]

Side effect: writes <run_dir>/.poll_state.json so we can detect new gens / new
champion across subsequent invocations.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import numpy as np  # noqa: E402

from pushgp.serialize import dict_to_program  # noqa: E402
from pushgp.trace import trace_program  # noqa: E402


# ------------------------------- locate run --------------------------------
def find_run_dir(run_name: str) -> Path:
    candidates = [
        Path("results") / "logged_evolution" / run_name,
        Path(_PROJ) / "results" / "logged_evolution" / run_name,
        Path(_PROJ).parent.parent / "results" / "logged_evolution" / run_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "run dir not found. tried:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ------------------------------- math interp --------------------------------
# Map raw token -> compact symbol used in math view
_TOKEN_MAP_BASE = {
    "L_v": "Lv",
    "FVec.PopBack": "m",            # generic "incoming neighbor msg"
    "Float.ConstPi": "π",
    "Float.Const0": "0",
    "Float.Const1": "1",
    "Float.ConstNeg1": "-1",
    "Float.ConstHalf": "0.5",
    "Float.Const2": "2",
    "Float.Const0_1": "0.1",
    "Float.Const1e-6": "1e-6",
    "Env.GetIter": "iter",
    "Env.GetMaxIter": "MaxIter",
    "Env.GetDeg": "deg",
    "Env.GetEdgeIndex": "e",
}

_OP_INFIX = {
    "Float.Add": "+", "Float.Sub": "-", "Float.Mul": "*", "Float.Div": "/",
    "Float.LT": "<", "Float.GT": ">", "Float.EQ": "==",
}
_OP_FUNC = {
    "Float.Neg": "neg", "Float.Inv": "inv", "Float.Square": "sq",
    "Float.Sqrt": "sqrt", "Float.Sign": "sign", "Float.Tanh": "tanh",
    "Float.Atanh": "atanh", "Float.Exp": "exp", "Float.Log": "log",
    "Float.Min": "min", "Float.Max": "max", "Float.Abs": "abs",
    "Float.Ceil": "ceil", "Float.Floor": "floor",
}


def _tokenize(expr: str) -> List[str]:
    """Tokenise a TraceVM live expression like
       'Float.Tanh(Float.Div(L_v, Float.Mul(FVec.PopBack, FVec.PopBack)))'."""
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*|[(),]", expr)


def _parse(tokens: List[str], pos: int) -> Tuple[Any, int]:
    head = tokens[pos]; pos += 1
    if pos < len(tokens) and tokens[pos] == "(":
        pos += 1
        args: List[Any] = []
        if tokens[pos] != ")":
            while True:
                arg, pos = _parse(tokens, pos)
                args.append(arg)
                if tokens[pos] == ",":
                    pos += 1
                else:
                    break
        assert tokens[pos] == ")"
        pos += 1
        return (head, args), pos
    return (head, []), pos


def _to_math(node: Any, evo_consts: Optional[List[float]] = None,
             m_counter: Optional[List[int]] = None) -> str:
    if m_counter is None:
        m_counter = [0]
    head, args = node
    # leaf
    if not args:
        if head == "FVec.PopBack":
            m_counter[0] += 1
            return f"m{m_counter[0]}"
        if head in _TOKEN_MAP_BASE:
            return _TOKEN_MAP_BASE[head]
        # EvoConstK -> K_k or actual value
        m = re.match(r"Float\.EvoConst(\d+)$", head)
        if m:
            k = int(m.group(1))
            if evo_consts and k < len(evo_consts):
                return f"K{k}({evo_consts[k]:.3g})"
            return f"K{k}"
        return head
    # operator/function
    rargs = [_to_math(a, evo_consts, m_counter) for a in args]
    if head in _OP_INFIX and len(rargs) == 2:
        return f"({rargs[0]}{_OP_INFIX[head]}{rargs[1]})"
    if head in _OP_FUNC:
        return f"{_OP_FUNC[head]}({', '.join(rargs)})"
    return f"{head}({', '.join(rargs)})"


def interpret(expr: Optional[str], evo_consts: Optional[List[float]]) -> str:
    if not expr:
        return "<empty>"
    try:
        toks = _tokenize(expr)
        node, _ = _parse(toks, 0)
        math = _to_math(node, evo_consts)
    except Exception as e:
        return f"<parse error: {e}>"
    # Light pattern recognition
    notes = []
    e = math
    if "tanh" in e and "atanh" in e:
        notes.append("sum-product family (tanh/atanh)")
    if "min(" in e and "abs(" in e:
        notes.append("min-sum family")
    if re.search(r"sign\(.*\).*max\(.*-\s*K", e):
        notes.append("OMS-like (sign * max(|min|-β,0))")
    if "Lv" in e:
        notes.append("uses channel LLR")
    else:
        notes.append("ignores channel LLR")
    # Count neighbor-msg references.  The renderer uses `m<i>` for
    # incoming reads it can statically resolve, but FVec.At with a
    # dynamic index (e.g. inside Exec.DoTimes using the loop counter)
    # stays as the literal "FVec.At(<arg>)" — those must be counted too,
    # otherwise loop-driven readers look like 0-neighbor programs.
    n_m = len(re.findall(r"\bm\d+\b", e))
    n_dynamic = e.count("FVec.At") + e.count("Env.GetIncomingVec")
    n_neighbor = n_m + n_dynamic
    notes.append(f"refs {n_neighbor} neighbor msgs")
    return e + "    [" + "; ".join(notes) + "]"


# ------------------------------- tracing -----------------------------------
def trace_one(prog_dicts: List[Dict[str, Any]],
              log_consts: List[float], *, has_llr: bool) -> Dict[str, Any]:
    try:
        prog = dict_to_program(prog_dicts)
        evo = np.asarray([10.0 ** float(x) for x in log_consts], dtype=np.float64)
        return trace_program(
            prog,
            ctx_channel_llr=0.5,
            ctx_incoming=np.array([0.6, -0.4, 0.3, -0.7, 0.1, -0.2, 0.5],
                                  dtype=np.float64),
            ctx_noise_var=1.0, ctx_iter=2, ctx_max_iter=8, ctx_deg=8,
            ctx_edge_index=3, ctx_evo_constants=evo,
            ctx_has_channel_llr=has_llr,
        )
    except Exception as exc:
        return {"value": None, "live_expr": None,
                "fault": f"{type(exc).__name__}: {exc}"}


def render_instr(d: Dict[str, Any]) -> str:
    name = d.get("name", "?")
    cb = d.get("code_block"); cb2 = d.get("code_block2")
    if cb is None and cb2 is None:
        return name
    parts = [name, "{"]
    if cb is not None:
        parts.append(" ; ".join(render_instr(c) for c in cb))
    parts.append("}")
    if cb2 is not None:
        parts.append("{"); parts.append(" ; ".join(render_instr(c) for c in cb2)); parts.append("}")
    return "".join(parts)


def render_program(plist: List[Dict[str, Any]], max_chars: int = 360) -> str:
    s = " ; ".join(render_instr(d) for d in plist)
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


# ------------------------------- structural skeleton ------------------------
# A truthful, indented pretty-print of the program structure.  Unlike the LIVE
# trace, this DOES expand Exec.DoTimes / Exec.DoRange / Exec.While / Exec.If /
# Exec.When code blocks so the reader can see the real control flow that makes
# the program permutation-invariant (typically a loop body containing
# FVec.PopBack / FVec.At / FVec.Pop that consumes the incoming vector).
_LOOP_OPS = {"Exec.DoTimes", "Exec.DoRange", "Exec.While", "Exec.DoCount"}
_COND_OPS = {"Exec.If", "Exec.When", "Exec.Unless"}


def _skel_lines(plist: List[Any], indent: int = 0) -> List[str]:
    """Return a list of indented text lines for ``plist``.

    Each instruction either prints as a single line ``<indent><op>`` or, if it
    is a control-flow opcode with code blocks, as a header line followed by
    the recursively-rendered body lines (and a closing ``end <op>`` line).
    """
    lines: List[str] = []
    pad = "  " * indent
    for ins in plist:
        if not isinstance(ins, dict):
            lines.append(f"{pad}{ins!r}")
            continue
        op = ins.get("name", ins.get("op", "?"))
        body1 = ins.get("code_block")
        body2 = ins.get("code_block2")
        # Render extra immediate fields (anything other than name/op/body)
        extras = {
            k: v for k, v in ins.items()
            if k not in ("name", "op", "code_block", "code_block2")
        }
        extras_str = (" " + " ".join(f"{k}={v}" for k, v in extras.items())
                      if extras else "")
        if op in _LOOP_OPS and isinstance(body1, list):
            lines.append(f"{pad}{op}{extras_str} {{")
            lines.extend(_skel_lines(body1, indent + 1))
            lines.append(f"{pad}}}")
        elif op in _COND_OPS:
            if isinstance(body1, list) and isinstance(body2, list):
                lines.append(f"{pad}{op}{extras_str} then {{")
                lines.extend(_skel_lines(body1, indent + 1))
                lines.append(f"{pad}}} else {{")
                lines.extend(_skel_lines(body2, indent + 1))
                lines.append(f"{pad}}}")
            elif isinstance(body1, list):
                lines.append(f"{pad}{op}{extras_str} {{")
                lines.extend(_skel_lines(body1, indent + 1))
                lines.append(f"{pad}}}")
            else:
                lines.append(f"{pad}{op}{extras_str}")
        else:
            lines.append(f"{pad}{op}{extras_str}")
    return lines


def render_skeleton(plist: List[Any], max_lines: int = 40) -> str:
    """Return a structural, indented skeleton of the program.

    Truncates at ``max_lines`` so output stays readable even for size-100
    programs.  The skeleton is deliberately TRUTHFUL: it shows every operator
    in the static program (including ones that may be no-ops at runtime when
    a stack happens to be empty) so the reader can see the real control flow.
    """
    lines = _skel_lines(plist, indent=0)
    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        head.append(f"  ... ({len(lines) - max_lines + 1} more lines)")
        lines = head
    return "\n".join("    " + ln for ln in lines)


# ------------------------------- main --------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--skeleton-lines", type=int, default=30,
                    help="max lines per program in the structural skeleton view")
    ap.add_argument("--no-skeleton", action="store_true",
                    help="hide the structural skeleton view")
    args = ap.parse_args()

    rd = find_run_dir(args.run_name)
    print(f"[poll] run dir: {rd}", flush=True)

    # Baseline OMS reference
    bp = rd / "baseline.json"
    if bp.exists():
        try:
            bj = json.loads(bp.read_text(encoding="utf-8"))
            print(f"[OMS baseline] fit={bj.get('fitness')}  BER={bj.get('ber_per_snr')}",
                  flush=True)
        except Exception:
            pass

    # Per-gen distribution
    gen_sum = load_jsonl(rd / "gen_summary.jsonl")
    print("\n=== generation distribution (best / mean / median / worst) ===")
    for r in gen_sum:
        # gen_summary has aggregates already; show what's there.
        keys = ("gen", "best", "mean", "med", "worst", "n_valid", "pop_size", "t_sec")
        line = "  " + "  ".join(
            f"{k}={r[k]:.4f}" if isinstance(r.get(k), float) else f"{k}={r.get(k, '?')}"
            for k in keys if k in r
        )
        print(line, flush=True)

    # Latest gen pairs
    inds = load_jsonl(rd / "individuals.jsonl")
    if not inds:
        print("\n[poll] no individuals yet (init still running).")
        return 0

    by_gen: Dict[int, List[Dict[str, Any]]] = {}
    for r in inds:
        by_gen.setdefault(r.get("gen", -1), []).append(r)
    latest_gen = max(by_gen)
    recs = sorted(by_gen[latest_gen], key=lambda r: r.get("fitness", float("inf")))

    # Pop distribution from raw individuals (richer than summary)
    fits = [r.get("fitness", float("nan")) for r in recs
            if isinstance(r.get("fitness"), (int, float))]
    if fits:
        print(f"\n=== gen {latest_gen}  pop_size={len(recs)} ===")
        print(f"  fit:  best={min(fits):+.4f}  med={statistics.median(fits):+.4f}  "
              f"mean={statistics.mean(fits):+.4f}  worst={max(fits):+.4f}  "
              f"valid={sum(1 for r in recs if r.get('valid'))}/{len(recs)}")

    # Champion delta tracking
    state_p = rd / ".poll_state.json"
    prev_state = {}
    if state_p.exists():
        try:
            prev_state = json.loads(state_p.read_text(encoding="utf-8"))
        except Exception:
            prev_state = {}
    prev_champ_fit = prev_state.get("champion_fit", float("inf"))
    champ = recs[0]
    champ_fit = champ.get("fitness", float("inf"))
    is_new = champ_fit < prev_champ_fit - 1e-9
    if is_new:
        print(f"\n*** NEW CHAMPION: gen={latest_gen}  fit={champ_fit:+.4f}  "
              f"(prev best={prev_champ_fit:+.4f}) ***")
    else:
        print(f"\n[champion unchanged: fit={champ_fit:+.4f}]")

    # Top-K with live expressions + math interpretation
    top = recs[: args.top_k]
    for i, r in enumerate(top):
        f = r.get("fitness", float("nan"))
        ber = r.get("ber_per_snr", [])
        v_sz = r.get("v2c_size", 0); c_sz = r.get("c2v_size", 0)
        k = r.get("log_constants", [])
        evo_vals = [10.0 ** float(x) for x in k]
        v_tr = trace_one(r.get("v2c", []), k, has_llr=True)
        c_tr = trace_one(r.get("c2v", []), k, has_llr=False)

        ber_str = "[" + ", ".join(f"{b:.3f}" for b in ber) + "]"
        print(f"\n--- #{i+1}  v={r.get('v_idx','?')} c={r.get('c_idx','?')}  "
              f"fit={f:+.4f}  BER={ber_str}  valid={r.get('valid')}")
        print(f"  V2C [size={v_sz}, sample_out={v_tr.get('value')}]:")
        print(f"     LIVE: {(v_tr.get('live_expr') or '<empty>')[:args.max_chars]}")
        print(f"     MATH: {interpret(v_tr.get('live_expr'), evo_vals)[:args.max_chars]}")
        print(f"  C2V [size={c_sz}, sample_out={c_tr.get('value')}]:")
        print(f"     LIVE: {(c_tr.get('live_expr') or '<empty>')[:args.max_chars]}")
        print(f"     MATH: {interpret(c_tr.get('live_expr'), evo_vals)[:args.max_chars]}")
        if not args.no_skeleton:
            print("  V2C SKELETON (truthful structure, loop bodies expanded):")
            print(render_skeleton(r.get("v2c", []), args.skeleton_lines))
            print("  C2V SKELETON (truthful structure, loop bodies expanded):")
            print(render_skeleton(r.get("c2v", []), args.skeleton_lines))
        if is_new and i == 0:
            print(f"  CHAMPION RAW V2C: {render_program(r.get('v2c', []), args.max_chars)}")
            print(f"  CHAMPION RAW C2V: {render_program(r.get('c2v', []), args.max_chars)}")
        if k:
            print(f"  K (log10): [" + ", ".join(f"{x:+.2f}" for x in k) + "]")

    # Write state
    state_p.write_text(json.dumps({
        "champion_fit": champ_fit,
        "champion_gen": latest_gen,
        "champion_v_idx": champ.get("v_idx"),
        "champion_c_idx": champ.get("c_idx"),
        "n_gens_seen": len(by_gen),
    }), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
