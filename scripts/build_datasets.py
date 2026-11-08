"""Build the frozen task splits.

    python scripts/build_datasets.py --out data

Splits use DISJOINT world seeds (no shared people, claims, limits or city tables):

  train     seeds 10000-   schema S   (SFT demos + GRPO prompts)
  val       seeds 30000-   schema S   (checkpoint selection, self-heal gate dev set)
  test_id   seeds 50000-   schema S   (in-distribution held-out test)
  test_ood  seeds 70000-   schema O   (renamed tables/columns, never seen in training)

Every task is oracle-verified at generation time (see claimsdesk.tasks).
A manifest with per-split SHA-256 hashes is written so test sets can be frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claimsdesk import __version__  # noqa: E402
from claimsdesk.tasks import generate_for_world  # noqa: E402

SPLITS = {
    # name: (first_seed, n_worlds, variant, per-world counts)
    "train": (10000, 700, "S", {"analytics": 2, "policy": 2, "workflow": 2}),
    "val": (30000, 50, "S", {"analytics": 2, "policy": 2, "workflow": 2}),
    "test_id": (50000, 150, "S", {"analytics": 1, "policy": 1, "workflow": 1}),
    "test_ood": (70000, 60, "O", {"analytics": 1, "policy": 1, "workflow": 1}),
}


def build(out: Path, scale: float = 1.0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"generator_version": __version__, "splits": {}}
    for name, (seed0, n_worlds, variant, counts) in SPLITS.items():
        n = max(1, int(n_worlds * scale))
        rng = random.Random(f"split-{name}")
        tasks = []
        for seed in range(seed0, seed0 + n):
            tasks += generate_for_world(seed, variant, counts, rng)
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["splits"][name] = {
            "file": path.name, "n_tasks": len(tasks), "worlds": [seed0, seed0 + n - 1],
            "schema_variant": variant, "sha256": digest,
            "by_type": dict(Counter(t.task_type for t in tasks)),
            "by_template": dict(Counter(t.template for t in tasks)),
            "by_difficulty": {str(k): v for k, v in sorted(Counter(t.difficulty for t in tasks).items())},
        }
        print(f"{name:9s} {len(tasks):5d} tasks  worlds {seed0}-{seed0 + n - 1}  sha256 {digest[:12]}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of worlds (for quick tests)")
    a = ap.parse_args()
    build(Path(a.out), a.scale)
