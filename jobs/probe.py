"""Kaggle probe: de-risk the stack before spending GPU hours.

Stages (each isolated; GPU stages in subprocesses so a crash can't kill the probe):
  env       library versions, GPUs, compute capability
  tests     claimsdesk test-suite inside the pinned venv
  hf        HF auth, discover Cohere model ids, gated-access check
  vllm      raw vLLM load + throughput of Tiny Aya on a T4 (fp16)
  rollout   AgentTune create_rollout_fn on real tasks with our tools + session hook,
            then replay analysis and AuditKIT ingestion of trajectories.jsonl
Writes $CLAIMSDESK_OUT/probe_report.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path(os.environ.get("CLAIMSDESK_OUT", "out"))
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "probe_report.json"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(key: str, value) -> None:
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    rep[key] = value
    REPORT.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n=== {key} ===\n{json.dumps(value, indent=2, default=str)[:4000]}\n", flush=True)


def stage(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                res = fn(*a, **k)
                save(name, {"ok": True, "secs": round(time.time() - t, 1), **(res or {})})
            except BaseException as e:  # noqa: BLE001
                save(name, {"ok": False, "secs": round(time.time() - t, 1), "error": repr(e),
                            "trace": traceback.format_exc()[-3000:]})
        return run
    return deco


@stage("env")
def s_env():
    import importlib
    vers = {}
    for m in ("torch", "transformers", "trl", "peft", "vllm", "agenttune", "auditkit", "accelerate"):
        try:
            vers[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception as e:  # noqa: BLE001
            vers[m] = f"IMPORT FAIL: {e!r}"[:200]
    import torch
    gpus = [{"name": torch.cuda.get_device_name(i), "cap": torch.cuda.get_device_capability(i),
             "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
            for i in range(torch.cuda.device_count())]
    return {"python": sys.version, "versions": vers, "gpus": gpus}


@stage("tests")
def s_tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True)
    return {"rc": r.returncode, "tail": r.stdout[-1500:]}


@stage("hf")
def s_hf():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    who = api.whoami().get("name")
    models = sorted({m.id for q in ("tiny-aya", "aya-expanse") for m in api.list_models(author="CohereLabs", search=q)})
    access = {}
    for mid in models:
        try:
            hf_hub_download(mid, "config.json")
            access[mid] = "ok"
        except Exception as e:  # noqa: BLE001
            access[mid] = type(e).__name__
    return {"whoami": who, "models": models, "access": access}


def pick_model() -> str:
    rep = json.loads(REPORT.read_text())
    ok = [m for m, v in rep.get("hf", {}).get("access", {}).items() if v == "ok"]
    for pref in ("tiny-aya-global",):
        for m in ok:
            if m.lower().endswith(pref):
                return m
    raise RuntimeError(f"no accessible tiny-aya-global among {ok}")


def sub(stage_name: str, model: str) -> None:
    r = subprocess.run([sys.executable, __file__, "--stage", stage_name, model], cwd=ROOT)
    if r.returncode != 0:
        rep = json.loads(REPORT.read_text())
        if stage_name not in rep:
            save(stage_name, {"ok": False, "error": f"subprocess rc={r.returncode} (likely OOM/crash)"})


@stage("vllm")
def s_vllm(model: str):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from claimsdesk.prompt import build_messages
    tok = AutoTokenizer.from_pretrained(model)
    t = time.time()
    llm = LLM(model=model, dtype="half", max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=False)
    load = time.time() - t
    msgs = [build_messages("WS-S000001", f"Say hello and give the number {i}.") for i in range(32)]
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    sp = SamplingParams(max_tokens=256, temperature=0.0)
    t = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    return {"model": model, "load_secs": round(load, 1), "batch32_secs": round(dt, 2),
            "gen_tokens": ntok, "tok_per_s": round(ntok / dt, 1), "sample": outs[0].outputs[0].text[:300],
            "template_head": prompts[0][:600]}


@stage("rollout")
def s_rollout(model: str):
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine, create_rollout_fn
    from agenttune.agentic.trajectory.dataset import trajectory_writer

    from claimsdesk.episode import analyse, from_agenttune
    from claimsdesk.prompt import SYSTEM_PROMPT
    from claimsdesk.session import pre_step_hook
    from claimsdesk.tasks import Task
    from claimsdesk.tools import TOOLS

    tasks = [Task.from_dict(json.loads(l)) for l in open(ROOT / "data" / "test_id.jsonl")]
    pick = []
    for tt in ("analytics", "policy", "workflow"):
        pick += [t for t in tasks if t.task_type == tt][:3]
    run_dir = OUT / "rollout_probe"
    run_dir.mkdir(exist_ok=True)
    traj_path = run_dir / "trajectories.jsonl"
    if traj_path.exists():
        traj_path.unlink()
    backend = "vllm"
    try:
        engine = create_rollout_engine(backend="vllm", model_path=model, dtype="half", max_model_len=8192,
                                       gpu_memory_utilization=0.85, max_completion_length=512,
                                       temperature=0.01, max_num_seqs=16)
    except Exception as e:  # noqa: BLE001
        print("vLLM engine failed, falling back to transformers:", repr(e), flush=True)
        import torch
        backend = "transformers"
        engine = create_rollout_engine(backend="transformers", model_path=model, dtype=torch.float16)
    fn = create_rollout_fn(rollout_engine=engine, tools=TOOLS, max_steps=12, system_prompt=SYSTEM_PROMPT,
                           pre_step_hook=pre_step_hook, on_trajectory_end=trajectory_writer(traj_path))
    t = time.time()
    res = fn([t_.prompt for t_ in pick])
    secs = time.time() - t
    sigs, samples = [], []
    for task, traj in zip(pick, res["trajectories"]):
        ep = from_agenttune(task.task_id, traj)
        sig = analyse(task, ep)
        sigs.append({k: sig[k] for k in ("task_id", "correct", "n_calls", "invalid_calls", "duplicate_calls",
                                         "answer_status", "malformed_call", "grounded")})
        samples.append({"task": task.question[:160], "gold": task.gold_answer_text,
                        "calls": [(c.name, c.arguments) for c in ep.calls][:8],
                        "raw_turns": [r[:400] for r in ep.raw_turns[:3]], "final": ep.final_text[:400]})
    (run_dir / "samples.json").write_text(json.dumps(samples, indent=2, default=str))
    ak = {}
    try:
        from auditkit.agent_eval import episodes_from_agenttune
        eps = episodes_from_agenttune(str(traj_path))
        ak = {"episodes": len(eps), "first_turns": str(eps[0].turns())[:500] if eps else None}
    except Exception as e:  # noqa: BLE001
        ak = {"error": repr(e)}
    return {"backend": backend, "n": len(pick), "secs": round(secs, 1),
            "success": sum(s["correct"] for s in sigs), "signals": sigs, "auditkit": ak,
            "samples_head": samples[:3]}


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--stage":
        {"vllm": s_vllm, "rollout": s_rollout}[sys.argv[2]](sys.argv[3])
        sys.exit(0)
    s_env()
    s_tests()
    s_hf()
    try:
        m = pick_model()
    except Exception as e:  # noqa: BLE001
        save("model_pick", {"ok": False, "error": repr(e)})
        sys.exit(1)
    save("model_pick", {"ok": True, "model": m})
    sub("vllm", m)
    sub("rollout", m)
    print("PROBE DONE", flush=True)
