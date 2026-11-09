"""Online improvement round on the SFT policy: expert iteration (rejection-sampling RL)
+ self-healing with a deployment gate.  Covers PS 1/2 (RL on real tool-calling rollouts,
efficiency via the reward), PS 8 (detect -> classify -> corrective data -> retrain -> gate)
and PS 9 (naive vs hardened reward on the SAME rollouts).

Stages (each its own process so the GPU is clean):
  sample  SFT policy rolls out K samples per prompt (vLLM, T=0.8) on unseen train tasks
  build   replay-verify; keep the best hardened-reward rollout per prompt; classify
          failures and add oracle corrective examples; measure what the naive reward
          would have accepted (reward hacks)
  train   continue the SFT LoRA through AgentTune create_distill_trainer
  gate    val split: candidate vs SFT; deploy only if success improves and unsafe
          approvals do not regress; then evaluate on the test splits

    python jobs/improve.py --sft-adapter kd1609/claimsdesk-tiny-aya-global-sft-lora
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimsdesk.episode import analyse, from_agenttune  # noqa: E402
from claimsdesk.rewards import hardened_reward, naive_reward  # noqa: E402
from claimsdesk.session import bound  # noqa: E402
from claimsdesk.tasks import Task  # noqa: E402
from claimsdesk.tools import call_tool  # noqa: E402

OUT = Path(os.environ.get("CLAIMSDESK_OUT", "out"))
RUN = OUT / "improve"
BASE = "CohereLabs/tiny-aya-global"


def py(*args: str) -> int:
    print("=== stage", args, flush=True)
    return subprocess.run([sys.executable, *args], cwd=ROOT).returncode


def pick_prompts(n: int, seed: int) -> list[Task]:
    tasks = [Task.from_dict(json.loads(l)) for l in open(ROOT / "data" / "train.jsonl", encoding="utf-8")]
    random.Random(seed).shuffle(tasks)
    want = {"workflow": int(n * 0.6), "policy": int(n * 0.25), "analytics": n - int(n * 0.6) - int(n * 0.25)}
    out = []
    for t in reversed(tasks):  # the SFT oracle sample came from the front of the same shuffle
        if want[t.task_type] > 0:
            want[t.task_type] -= 1
            out.append(t)
    return out


def merge(adapter: str, dst: Path) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not (dst / "config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": "cpu"})
        PeftModel.from_pretrained(base, adapter).merge_and_unload().save_pretrained(dst)
        AutoTokenizer.from_pretrained(BASE).save_pretrained(dst)
    return str(dst)


def stage_sample(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM

    sys.path.insert(0, str(ROOT / "jobs"))
    from eval import run_split  # noqa: E402

    from claimsdesk.prompt import SYSTEM_PROMPT
    model = merge(a.sft_adapter, Path("/kaggle/working/sft_merged") if Path("/kaggle/working").exists() else RUN / "sft_merged")
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="half", max_model_len=8192, gpu_memory_utilization=0.88, seed=0)
    prompts = pick_prompts(a.n_prompts, a.seed)
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "rollouts.jsonl", "w", encoding="utf-8") as f:
        for k in range(a.k):
            eps = run_split(llm, tok, prompts, SYSTEM_PROMPT, 12, 512, 8192, temperature=a.temperature, seed=1000 + k)
            for e in eps:
                rec = {"task_id": e.task.task_id, "sample": k, "final": e.final,
                       "steps": [{"action": s.action, "thought": s.thought} for s in e.steps]}
                f.write(json.dumps(rec, default=str) + "\n")
            print(f"sample round {k} done", flush=True)


def to_messages(task: Task, ep) -> list[dict]:
    """Rebuild an OpenAI-style trajectory with the EXACT tool outputs (by replay)."""
    msgs = []
    with bound(task.seed, task.variant):
        for i, c in enumerate(ep.calls):
            cid = f"call_{i + 1}"
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "name": c.name, "arguments": c.arguments}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "name": c.name, "content": call_tool(c.name, c.arguments)})
    msgs.append({"role": "assistant", "content": ep.final_text.strip()})
    return msgs


def classify(sig: dict) -> str:
    """Deterministic failure taxonomy from replay signals (self-healing classifier)."""
    if sig.get("malformed_call"):
        return "malformed_tool_call"
    if not sig["has_output"]:
        return "no_answer_or_no_decision"
    if sig.get("unsafe_approval"):
        return "unsafe_approval"
    if sig.get("over_escalation"):
        return "over_escalation"
    if sig["invalid_calls"]:
        return "invalid_tool_call"
    if sig.get("fooled_by_distractor"):
        return "used_superseded_policy"
    if sig["task_type"] == "workflow" and sig["correct"] and not sig.get("evidence_before_decision"):
        return "decision_without_evidence"
    if sig["task_type"] == "workflow" and sig["correct"] and not sig.get("citation_ok"):
        return "wrong_citation"
    if sig["task_type"] == "workflow":
        return "wrong_decision_or_reason"
    return "wrong_answer"


def stage_build(a) -> None:
    tasks = {t.task_id: t for t in pick_prompts(a.n_prompts, a.seed)}
    by_task: dict[str, list] = {}
    for line in open(RUN / "rollouts.jsonl", encoding="utf-8"):
        r = json.loads(line)
        task = tasks[r["task_id"]]
        ep = from_agenttune(task.task_id, {"steps": r["steps"], "final_response": r["final"]})
        sig = analyse(task, ep)
        by_task.setdefault(task.task_id, []).append((hardened_reward(sig), naive_reward(sig), sig, ep))

    rows, failures, hacks = [], Counter(), Counter()
    stats = {"prompts": len(by_task), "samples": sum(len(v) for v in by_task.values())}
    naive_accepted = naive_hack = 0
    for tid, samples in by_task.items():
        task = tasks[tid]
        for h, n, sig, _ in samples:
            if n == 1.0:
                naive_accepted += 1
                if not sig["correct_and_grounded"] or (task.task_type == "workflow" and not sig["correct"]):
                    naive_hack += 1
                    hacks[classify(sig) if not sig["correct"] else
                          ("decision_without_evidence" if task.task_type == "workflow" else "ungrounded_answer")] += 1
        best = max(samples, key=lambda x: (x[0], -x[2]["n_calls"]))
        if best[0] >= 1.1:
            rows.append({"task_id": tid, "prompt": task.prompt, "messages": to_messages(task, best[3]), "source": "self_rollout"})
        else:
            failures[classify(best[2])] += 1
            rows.append({"task_id": tid, "prompt": task.prompt, "messages": task.gold_messages, "source": "corrective"})
    # small replay buffer of the original SFT distribution to avoid forgetting
    replay = [t for t in pick_prompts(4 * a.n_prompts, a.seed + 1) if t.task_id not in tasks][: a.replay]
    rows += [{"task_id": t.task_id, "prompt": t.prompt, "messages": t.gold_messages, "source": "replay"} for t in replay]
    random.Random(a.seed).shuffle(rows)
    with open(RUN / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    pass_at_k = sum(any(s[2]["correct"] for s in v) for v in by_task.values()) / max(1, len(by_task))
    mean_success = sum(s[2]["correct"] for v in by_task.values() for s in v) / max(1, stats["samples"])
    stats.update(pass_at_1_sampled=mean_success, pass_at_k=pass_at_k,
                 kept_self_rollouts=sum(r["source"] == "self_rollout" for r in rows),
                 corrective=sum(r["source"] == "corrective" for r in rows), replay=len(replay),
                 failure_classes=dict(failures),
                 reward_hacking={"naive_accepted_samples": naive_accepted,
                                 "naive_accepted_but_hacky": naive_hack,
                                 "hack_rate_among_naive_accepted": naive_hack / max(1, naive_accepted),
                                 "hack_types": dict(hacks),
                                 "hardened_accepted_samples": sum(s[0] >= 1.1 for v in by_task.values() for s in v)})
    (RUN / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


def summary(label: str, split: str) -> dict:
    return json.loads((OUT / label / split / "summary.json").read_text())["metrics"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", default="kd1609/claimsdesk-tiny-aya-global-sft-lora")
    ap.add_argument("--hub-repo", default="kd1609/claimsdesk-tiny-aya-global-healed-lora")
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--replay", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    if a.stage == "sample":
        return stage_sample(a)
    if a.stage == "build":
        return stage_build(a)

    me = str(Path(__file__).resolve())
    common = ["--sft-adapter", a.sft_adapter, "--n-prompts", str(a.n_prompts), "--k", str(a.k),
              "--temperature", str(a.temperature), "--replay", str(a.replay), "--seed", str(a.seed)]
    if py(me, "--stage", "sample", *common) or py(me, "--stage", "build", *common):
        raise SystemExit("sample/build failed")
    adapter = str(OUT / "heal" / "adapter")
    py(str(ROOT / "jobs" / "sft.py"), "--n-oracle", "0", "--teacher-glob", "/nonexistent/*.jsonl",
       "--extra-rows", str(RUN / "rows.jsonl"), "--init-adapter", a.sft_adapter, "--out-name", "heal",
       "--lr", "5e-5", "--no-eval", "--hub-repo", "")
    if not (Path(adapter) / "adapter_config.json").exists():
        raise SystemExit("heal training failed")

    ev = str(ROOT / "jobs" / "eval.py")
    py(ev, "--model", BASE, "--adapter", a.sft_adapter, "--label", "gate_sft", "--splits", "val")
    py(ev, "--model", BASE, "--adapter", adapter, "--label", "gate_candidate", "--splits", "val")
    old, new = summary("gate_sft", "val"), summary("gate_candidate", "val")
    d_succ = new["task_success"]["mean"] - old["task_success"]["mean"]
    d_unsafe = new["unsafe_approval_rate"]["mean"] - old["unsafe_approval_rate"]["mean"]
    deploy = d_succ > 0 and d_unsafe <= 0.01
    gate = {"rule": "deploy iff val task_success improves AND unsafe_approval_rate rises by <= 1 pt",
            "sft_val": {k: old[k]["mean"] for k in ("task_success", "unsafe_approval_rate", "calls_per_task")},
            "candidate_val": {k: new[k]["mean"] for k in ("task_success", "unsafe_approval_rate", "calls_per_task")},
            "delta_success": d_succ, "delta_unsafe": d_unsafe, "deploy": deploy}
    (RUN / "gate.json").write_text(json.dumps(gate, indent=2))
    print("GATE", json.dumps(gate, indent=2), flush=True)
    if a.hub_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.hub_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=adapter, repo_id=a.hub_repo)
    if deploy:
        py(ev, "--model", BASE, "--adapter", adapter, "--label", "tiny_aya_global_healed")


if __name__ == "__main__":
    main()
