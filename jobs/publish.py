"""Publish models, datasets, logs and collections to the Hugging Face Hub (CPU job).

    python jobs/publish.py --user kd1609
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GH = "https://github.com/kdt523/LexsiAg3"

MODEL_CARD = """---
base_model: CohereLabs/tiny-aya-global
library_name: peft
tags: [agent, tool-calling, lora, agenttune, auditkit, claims-desk]
license: cc-by-nc-4.0
---
# {title}

LoRA adapter for **CohereLabs/tiny-aya-global**, trained for the *Claims & Spend Desk* agentic environment
(SQL analytics, policy retrieval, expense-claim workflow decisions) with **AgentTune**; evaluated with
replay-verified metrics and **AuditKIT**. Code: {gh}

{body}

**Usage**: load the base model, apply this adapter (or merge it), and use the system prompt and
JSON tool-call format from `claimsdesk/prompt.py` in the repository.
"""

SFT_BODY = """**Status: deployed model.** Agentic distillation: SFT on 1,866 replay-verified trajectories
(Aya Expanse 32B teacher + oracle tool-only experts), assistant-only loss.

| Metric (test_id, n=450) | Base (full prompt) | This adapter |
|---|---|---|
| Task success | 6.0% | **81.3%** |
| Grounded success | 3.1% | **80.7%** |
| Invalid-call rate | 31% | **0%** |
| Redundant calls | 44% | **0%** |
| Calls per task | 9.4 | **3.5** |
| Unsafe approvals | 31% | **2.7%** |

OOD (renamed schema, n=180): 3.3% -> 41.7%."""

HEALED_BODY = """**Status: NOT deployed (rejected by the deployment gate).** Self-healing round: expert iteration
on the SFT model's own best rollouts (hardened reward) plus oracle corrections of classified failures.

| Validation (n=300) | SFT | This candidate |
|---|---|---|
| Task success | 78% | 80% |
| Unsafe approvals | 3% | **19%** |

Gate rule: deploy iff success improves AND unsafe approvals rise by <= 1 pt -> rejected. Published
for transparency as a documented safety-gate outcome."""

DATA_CARD = """---
license: cc-by-4.0
tags: [agent, tool-calling, synthetic, multi-hop, sql, rag]
---
# {title}

{body}

Generator, environment and evaluation code: {gh}
"""

TASKS_BODY = """Procedurally generated, seed-disjoint tasks for the *Claims & Spend Desk* environment. Every task is
oracle-verified twice (Python gold vs. a tool-only oracle agent). Types: analytics (SQL), policy
(multi-hop retrieval with superseded/wrong-entity distractors, FX), workflow (claim decision + reason + clause).

| Split | Worlds (seeds) | Schema | Tasks |
|---|---|---|---|
| train | 10000-10699 | S | 4,200 |
| val | 30000-30049 | S | 300 |
| test_id | 50000-50149 | S | 450 |
| test_ood | 70000-70059 | O (all tables/columns renamed) | 180 |

`manifest.json` holds sha256 hashes of each split."""

LOGS_BODY = """Training and evaluation logs for the Claims & Spend Desk project.

* `evals/` - every evaluation run: AgentTune trajectories, replay-verified per-episode signals, summaries with bootstrap CIs
* `teacher/` - Aya Expanse 32B teacher trajectories (all + replay-verified)
* `self_heal/` - expert-iteration rollouts, built training rows, build stats (incl. naive vs hardened reward-hacking counts)
* `grpo_negative/` - GRPO attempt logs (OOM on T4, documented negative result)
* `results/` - committed summaries, deployment gate decision, AuditKIT reports"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="kd1609")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    u = a.user
    print("whoami:", api.whoami()["name"], flush=True)

    models = {f"{u}/claimsdesk-tiny-aya-global-sft-lora": ("Claims Desk - Tiny Aya Global SFT LoRA", SFT_BODY),
              f"{u}/claimsdesk-tiny-aya-global-healed-lora": ("Claims Desk - Tiny Aya Global self-heal candidate (rejected)", HEALED_BODY)}
    for repo, (title, body) in models.items():
        api.upload_file(path_or_fileobj=MODEL_CARD.format(title=title, body=body, gh=GH).encode(),
                        path_in_repo="README.md", repo_id=repo)
        api.update_repo_settings(repo, private=False)
        print("public:", repo, flush=True)

    tasks_repo, logs_repo = f"{u}/claimsdesk-tasks", f"{u}/claimsdesk-logs"
    api.create_repo(tasks_repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(ROOT / "data"), repo_id=tasks_repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=DATA_CARD.format(title="Claims & Spend Desk tasks", body=TASKS_BODY, gh=GH).encode(),
                    path_in_repo="README.md", repo_id=tasks_repo, repo_type="dataset")
    print("public:", tasks_repo, flush=True)

    logs = Path("/tmp/logs_unz")
    shutil.rmtree(logs, ignore_errors=True)
    found = glob.glob("/kaggle/input/**/claimsdesk_logs.zip", recursive=True)
    if found:
        zipfile.ZipFile(found[0]).extractall(logs)
        src = logs / "logs"
    else:
        src = Path(glob.glob("/kaggle/input/**/logs/teacher", recursive=True)[0]).parent
    api.create_repo(logs_repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(src), repo_id=logs_repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=DATA_CARD.format(title="Claims & Spend Desk logs", body=LOGS_BODY, gh=GH).encode(),
                    path_in_repo="README.md", repo_id=logs_repo, repo_type="dataset")
    print("public:", logs_repo, flush=True)

    out = {}
    for title, items in [("Claims Desk Agent - Models", [(r, "model") for r in models]),
                         ("Claims Desk Agent - Datasets", [(tasks_repo, "dataset"), (logs_repo, "dataset")])]:
        c = api.create_collection(title=title, namespace=u, private=False, exists_ok=True,
                                  description="Lexsi Labs AI Safety Hackathon, Track 3 (AgentTune + AuditKIT)")
        for rid, typ in items:
            try:
                api.add_collection_item(c.slug, item_id=rid, item_type=typ, exists_ok=True)
            except Exception as e:  # noqa: BLE001
                print("collection item:", rid, e)
        out[title] = f"https://huggingface.co/collections/{c.slug}"
        print("collection:", out[title], flush=True)
    Path("/kaggle/working/out").mkdir(parents=True, exist_ok=True)
    Path("/kaggle/working/out/publish.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
