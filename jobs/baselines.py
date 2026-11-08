"""Run all baseline evaluations sequentially (each in its own process to free GPU memory)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ["--model", "CohereLabs/tiny-aya-global", "--label", "tiny_aya_global_minimal", "--prompt", "minimal"],
    ["--model", "CohereLabs/tiny-aya-global", "--label", "tiny_aya_global_prompted", "--prompt", "full"],
    ["--model", "CohereLabs/aya-expanse-8b", "--label", "aya_expanse_8b_prompted", "--prompt", "full", "--tp", "2"],
]
extra = sys.argv[1:]
for args in RUNS:
    print("=== baseline", args, flush=True)
    subprocess.run([sys.executable, str(ROOT / "jobs" / "eval.py"), *args, *extra], cwd=ROOT)
