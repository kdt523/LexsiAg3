"""Batched, AgentTune-faithful evaluation on the frozen splits.

Runs every task of a split in LOCKSTEP: each turn, all unfinished episodes are rendered
with AgentTune's own ``_render_chat_template`` (same tools-fallback prompt, same tool
result folding) and generated in ONE vLLM batch; tool calls are parsed with AgentTune's
``_extract_tool_calls``. This is the same format the trainer uses, at batch throughput.

Outputs per run (``$CLAIMSDESK_OUT/<label>/<split>/``):
  trajectories.jsonl   AgentTune ``Trajectory.to_dict()`` records (AuditKIT ingests these)
  signals.jsonl        replay-verified per-episode signals (claimsdesk.episode.analyse)
  summary.json         metrics with 95% bootstrap CIs + breakdowns

    python jobs/eval.py --model CohereLabs/tiny-aya-global --label base --splits test_id test_ood
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimsdesk import session as S  # noqa: E402
from claimsdesk.episode import analyse, from_agenttune  # noqa: E402
from claimsdesk.metrics import breakdown, summarise  # noqa: E402
from claimsdesk.prompt import MINIMAL_SYSTEM_PROMPT, SYSTEM_PROMPT, TOOL_RESULT_FORMAT, TOOLS_FALLBACK_PROMPT  # noqa: E402
from claimsdesk.session import Session  # noqa: E402
from claimsdesk.tasks import Task  # noqa: E402
from claimsdesk.tools import TOOLS  # noqa: E402
from claimsdesk.world import World  # noqa: E402

OUT = Path(os.environ.get("CLAIMSDESK_OUT", "out"))


class Ep:
    def __init__(self, task: Task, conv: list[dict]):
        self.task = task
        self.conv = conv
        self.steps: list = []
        self.done = False
        self.final = ""
        self.n_iter = 0
        self.sess = Session(World.get(task.seed, task.variant))
        self.stop_reason = ""


def load_split(name: str, limit: int | None) -> list[Task]:
    tasks = [Task.from_dict(json.loads(l)) for l in open(ROOT / "data" / f"{name}.jsonl", encoding="utf-8")]
    return tasks[:limit] if limit else tasks


def run_split(llm, tok, tasks: list[Task], system_prompt: str, max_steps: int, max_tokens: int,
              max_model_len: int, lora=None, temperature: float = 0.0, seed: int = 0) -> list[Ep]:
    from agenttune.agentic.rollout_engines import rollout_factory as RF
    from agenttune.agentic.trajectory.dataset import Step
    from vllm import SamplingParams

    schemas = [RF._build_tool_schema(t) for t in TOOLS]
    tool_map = {t.__name__: t for t in TOOLS}
    fold = not RF._template_renders_tools(tok, schemas)
    fallback = TOOLS_FALLBACK_PROMPT.format(tools=json.dumps(schemas, indent=2))  # same call as AgentTune
    print(f"template renders tools: {not fold}; folding tool results: {fold}", flush=True)

    eps = []
    for t in tasks:
        conv = RF._build_conversation(t.prompt, system_prompt)
        if fold:
            conv[0] = {**conv[0], "content": f"{conv[0]['content']}\n\n{fallback}"}
        eps.append(Ep(t, conv))

    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0 if temperature == 0 else 0.95,
                        seed=seed, skip_special_tokens=False)
    for turn in range(max_steps + 1):
        active = [e for e in eps if not e.done]
        if not active:
            break
        prompts, live = [], []
        for e in active:
            text = RF._render_chat_template(tok, e.conv, schemas, tool_result_format=TOOL_RESULT_FORMAT, tokenize=False)
            if len(tok(text, add_special_tokens=False)["input_ids"]) > max_model_len - max_tokens - 16:
                e.done, e.stop_reason = True, "context_overflow"
                continue
            prompts.append(text)
            live.append(e)
        if not live:
            break
        t0 = time.time()
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        print(f"turn {turn}: {len(live)} active, {time.time() - t0:.1f}s", flush=True)
        for e, out in zip(live, outs):
            raw = out.outputs[0].text
            msg = {"role": "assistant", "content": RF._strip_special_tokens(raw, tok)}
            calls = RF._extract_tool_calls(msg, raw)
            if not calls or e.n_iter >= max_steps:
                e.final = msg["content"]
                e.done = True
                e.stop_reason = "answer" if not calls else "max_steps"
                e.conv.append(msg)
                continue
            calls = RF._assign_tool_call_ids(calls)
            msg["tool_calls"] = calls
            e.conv.append(msg)
            S._local.session = e.sess
            S._global["session"] = e.sess
            results = []
            for tc in calls:
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name") or tc.get("name")
                args = fn.get("arguments", fn.get("parameters", tc.get("arguments", tc.get("parameters", {}))))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                if not name or name not in tool_map:
                    results.append((name or "unknown", {"error": f"Tool '{name}' not found"}))
                    continue
                try:
                    results.append((name, str(tool_map[name](**args) if isinstance(args, dict) else tool_map[name](args))))
                except Exception as ex:  # noqa: BLE001
                    results.append((name, {"error": str(ex)}))
            for tc, (name, res) in zip(calls, results):
                e.conv.append({"role": "tool", "name": name, "content": str(res), "tool_call_id": tc.get("id")})
            if fold:
                e.conv = RF.fold_tool_messages_into_user(e.conv, TOOL_RESULT_FORMAT)
            e.n_iter += 1
            e.steps.append(Step(step_number=e.n_iter, state=f"[turn={e.n_iter}]", action={"tool_calls": calls},
                                observation=str(dict(results)), thought=raw))
    for e in eps:
        if not e.done:
            e.done, e.stop_reason = True, "max_turns"
    return eps


def write_run(eps: list[Ep], run_dir: Path, meta: dict) -> dict:
    from agenttune.agentic.trajectory.dataset import Trajectory

    run_dir.mkdir(parents=True, exist_ok=True)
    sigs = []
    with open(run_dir / "trajectories.jsonl", "w", encoding="utf-8") as ft, \
            open(run_dir / "signals.jsonl", "w", encoding="utf-8") as fs:
        for e in eps:
            traj = Trajectory(task=e.task.prompt, steps=e.steps, final_response=e.final,
                              metadata={"conversation": e.conv, "task_id": e.task.task_id,
                                        "stop_reason": e.stop_reason, **meta})
            rec = traj.to_dict()
            ft.write(json.dumps(rec, default=str) + "\n")
            sig = analyse(e.task, from_agenttune(e.task.task_id, rec))
            sig["stop_reason"] = e.stop_reason
            sigs.append(sig)
            fs.write(json.dumps(sig, default=str) + "\n")
    summary = {"meta": meta, "n": len(sigs), "metrics": summarise(sigs),
               "by_type": breakdown(sigs, "task_type"), "by_difficulty": breakdown(sigs, "difficulty"),
               "by_template": breakdown(sigs, "template")}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "provenance.json").write_text(json.dumps(meta, indent=2, default=str))
    m = summary["metrics"]
    print(f"[{meta['label']}/{meta['split']}] success {m['task_success']['mean']:.3f} "
          f"grounded {m['grounded_success']['mean']:.3f} calls {m['calls_per_task']['mean']:.2f} "
          f"invalid {m.get('invalid_call_rate', {}).get('mean', float('nan')):.3f}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--splits", nargs="+", default=["test_id", "test_ood"])
    ap.add_argument("--prompt", choices=["full", "minimal"], default="full")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path or HF repo")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(model=a.model, dtype="half", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              tensor_parallel_size=a.tp, seed=0)
    lora = None
    if a.adapter:
        # vLLM's LoRA path crashes on Cohere's tied embeddings -> merge on CPU, serve merged weights
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        merged = Path("/tmp/merged_eval") if Path("/kaggle/working").exists() else OUT / "merged_eval"
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map={"": "cpu"})
        PeftModel.from_pretrained(base, a.adapter).merge_and_unload().save_pretrained(merged)
        tok.save_pretrained(merged)
        del base
        kw["model"] = str(merged)
    llm = LLM(**kw)
    if a.adapter:  # weights are now in GPU memory; free the disk
        import shutil
        shutil.rmtree(kw["model"], ignore_errors=True)
    system_prompt = SYSTEM_PROMPT if a.prompt == "full" else MINIMAL_SYSTEM_PROMPT
    for split in a.splits:
        tasks = load_split(split, a.limit)
        t0 = time.time()
        eps = run_split(llm, tok, tasks, system_prompt, a.max_steps, a.max_tokens, a.max_model_len, lora)
        meta = {"label": a.label, "model": a.model, "adapter": a.adapter, "split": split, "prompt": a.prompt,
                "n_tasks": len(tasks), "wall_secs": round(time.time() - t0, 1), "decoding": "greedy",
                "max_steps": a.max_steps}
        write_run(eps, OUT / a.label / split, meta)


if __name__ == "__main__":
    main()
