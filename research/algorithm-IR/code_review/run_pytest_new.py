"""Run pytest on new test files; write output to a file."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pytest_new.log"
OUT.parent.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "-m", "pytest", "-v",
    "tests/unit/test_const_lifter.py",
    "tests/unit/test_types_lattice.py",
]
with OUT.open("w", encoding="utf-8") as f:
    proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
print(f"== rc={proc.returncode}")
print(f"== last 60 lines of {OUT}:")
for L in OUT.read_text(encoding="utf-8").splitlines()[-60:]:
    print(L)
