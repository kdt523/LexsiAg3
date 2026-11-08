"""RL on real tool-calling rollouts: GRPO (AgentTune) from the SFT checkpoint, then eval.

    python jobs/grpo.py --reward hardened --sft-adapter kd1609/claimsdesk-tiny-aya-global-sft-lora
    python jobs/grpo.py --reward naive    ...     # reward-hacking ablation (PS 9)

* Rollouts: AgentTune ``create_rollout_fn`` with our tools, the per-episode
  ``pre_step_hook`` (fresh world session per trajectory) and the shared prompt formats.
* Reward: replay-verified (``claimsdesk.episode.analyse``) -> naive or hardened reward.
* Prompts: training tasks the SFT model does NOT solve reliably are where GRPO learns;
  we sample a mix weighted towards workflow/policy (where SFT is weakest).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimsdesk.episode import analyse, from_agenttune  # noqa: E402
from claimsdesk.prompt import SYSTEM_PROMPT, TOOL_RESULT_FORMAT, TOOLS_FALLBACK_PROMPT  # noqa: E402
from claimsdesk.rewards import hardened_reward, naive_reward  # noqa: E402
from claimsdesk.session import pre_step_hook  # noqa: E402
from claimsdesk.tasks import Task  # noqa: E402
from claimsdesk.tools import TOOLS  # noqa: E402

OUT = Path(os.environ.get("CLAIMSDESK_OUT", "out"))


def make_reward(kind: str, log_path: Path):
    fn = hardened_reward if kind == "hardened" else naive_reward

    def reward(prompts=None, completions=None, task_json=None, trajectories=None, **kwargs):
        scores = []
        with open(log_path, "a", encoding="utf-8") as f:
            for i, tj in enumerate(task_json):
                task = Task.from_dict(json.loads(tj))
                try:
                    sig = analyse(task, from_agenttune(task.task_id, trajectories[i]))
                    r = fn(sig)
                except Exception as e:  # noqa: BLE001  (a broken rollout scores like no answer)
                    sig, r = {"error": repr(e)}, (-0.5 if kind == "hardened" else 0.0)
                scores.append(float(r))
                f.write(json.dumps({"task_id": task.task_id, "reward": r, "naive": naive_reward(sig) if "correct" in sig else 0.0,
                                    "hardened": hardened_reward(sig) if "correct" in sig else -0.5,
                                    **{k: sig.get(k) for k in ("correct", "grounded", "n_calls", "invalid_calls",
                                                               "duplicate_calls", "unsafe_approval", "over_escalation",
                                                               "evidence_before_decision", "citation_ok")}},
                                   default=str) + "\n")
        return scores

    reward.__name__ = f"{kind}_reward"
    return reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--reward", choices=["hardened", "naive"], default="hardened")
    ap.add_argument("--n-prompts", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max-turn-tokens", type=int, default=256)
    ap.add_argument("--hub-repo", default="")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from agenttune.agentic.trajectory.dataset import trajectory_writer
    from agenttune.core.backend_factory import create_agentic_trainer
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run = OUT / f"grpo_{a.reward}"
    run.mkdir(parents=True, exist_ok=True)

    # 1. merge the SFT adapter into the base -> GRPO trains a fresh LoRA on top
    merged = Path("/kaggle/working/sft_merged") if Path("/kaggle/working").exists() else run / "sft_merged"
    tok = AutoTokenizer.from_pretrained(a.base)
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float16)
    PeftModel.from_pretrained(base, a.sft_adapter).merge_and_unload().save_pretrained(merged)
    tok.save_pretrained(merged)
    del base
    torch.cuda.empty_cache()

    # 2. prompts
    tasks = [Task.from_dict(json.loads(l)) for l in open(ROOT / "data" / "train.jsonl", encoding="utf-8")]
    rng = random.Random(a.seed)
    rng.shuffle(tasks)
    want = {"workflow": int(a.n_prompts * 0.45), "policy": int(a.n_prompts * 0.35), "analytics": int(a.n_prompts * 0.2)}
    picked = []
    for t in reversed(tasks):  # take from the end: disjoint from the SFT oracle sample (taken from the front)
        if want[t.task_type] > 0:
            want[t.task_type] -= 1
            picked.append(t)
    ds = Dataset.from_list([{"prompt": t.prompt, "task_json": json.dumps(t.to_dict())} for t in picked])

    # 3. rollouts through AgentTune with the per-episode session hook
    rollout_func = create_rollout_fn(
        tools=TOOLS, max_steps=12, system_prompt=SYSTEM_PROMPT, pre_step_hook=pre_step_hook,
        on_trajectory_end=trajectory_writer(run / "trajectories.jsonl"),
        tools_fallback_prompt=TOOLS_FALLBACK_PROMPT, tool_result_format=TOOL_RESULT_FORMAT)

    trainer = create_agentic_trainer(
        "grpo", backend="trl",
        model=str(merged), reward_funcs=[make_reward(a.reward, run / "rewards.jsonl")],
        train_dataset=ds, rollout_func=rollout_func, output_dir=str(run / "ckpt"),
        peft_config=dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
        num_generations=a.num_generations, per_device_train_batch_size=a.num_generations,
        gradient_accumulation_steps=1, max_steps=a.max_steps, learning_rate=a.lr, beta=a.beta,
        max_completion_length=a.max_turn_tokens, temperature=0.9, fp16=True, bf16=False,
        gradient_checkpointing=True, logging_steps=1, save_strategy="no", report_to=[],
        use_vllm=False, seed=a.seed, model_init_kwargs={"dtype": torch.float16},
    )
    res = trainer.train()
    print("train result:", res, flush=True)
    adapter_dir = run / "adapter"
    trainer.trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (run / "train_log.json").write_text(json.dumps(trainer.trainer.state.log_history, default=str, indent=1))
    if a.hub_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.hub_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(adapter_dir), repo_id=a.hub_repo)
    del trainer
    torch.cuda.empty_cache()

    # 4. eval: merged SFT base + GRPO adapter, same harness as every other system
    cmd = [sys.executable, str(ROOT / "jobs" / "eval.py"), "--model", str(merged), "--adapter", str(adapter_dir),
           "--label", f"tiny_aya_global_sft_grpo_{a.reward}", "--prompt", "full"]
    if a.eval_limit:
        cmd += ["--limit", str(a.eval_limit)]
    subprocess.run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
