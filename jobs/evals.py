"""Run a list of eval configs sequentially (each in its own process for a clean GPU).

    python jobs/evals.py round2
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = "kd1609/claimsdesk-tiny-aya-global-sft-lora"
ROUNDS = {
    "round2": [
        ["--model", "CohereLabs/tiny-aya-global", "--adapter", SFT, "--label", "tiny_aya_global_sft", "--prompt", "full"],
        ["--model", "CohereLabs/tiny-aya-global", "--label", "tiny_aya_global_prompted", "--prompt", "full",
         "--splits", "test_ood"],
    ],
}
for args in ROUNDS[sys.argv[1]]:
    print("=== eval", args, flush=True)
    subprocess.run([sys.executable, str(ROOT / "jobs" / "eval.py"), *args, *sys.argv[2:]], cwd=ROOT)
