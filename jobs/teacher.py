"""Agentic distillation, step 1: collect VERIFIED teacher trajectories via the Cohere API.

The teacher (Aya Expanse 32B) sees exactly the student's prompt: same system prompt,
same text tool-call format (TOOLS_FALLBACK_PROMPT) and same folded tool results, so its
trajectories can be replayed into the student's chat template without translation.
Only trajectories that pass replay verification (correct AND grounded, no invalid calls)
are kept. Built for a trial key: rate-limited, call-capped, resumable.

    python jobs/teacher.py --split train --max-calls 900 --types policy workflow
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimsdesk import session as S  # noqa: E402
from claimsdesk.episode import analyse, from_messages  # noqa: E402
from claimsdesk.prompt import SYSTEM_PROMPT, TOOL_RESULT_FORMAT, TOOLS_FALLBACK_PROMPT  # noqa: E402
from claimsdesk.session import Session  # noqa: E402
from claimsdesk.tasks import Task  # noqa: E402
from claimsdesk.tools import TOOLS, call_tool  # noqa: E402
from claimsdesk.world import World  # noqa: E402

OUT = Path(os.environ.get("CLAIMSDESK_OUT", "out"))
API = "https://api.cohere.com/v2/chat"


def tool_schemas() -> list[dict]:
    try:
        from transformers.utils import get_json_schema
        return [get_json_schema(t) for t in TOOLS]
    except Exception:  # noqa: BLE001
        return [{"type": "function", "function": {"name": t.__name__, "description": (t.__doc__ or "").split("\n")[0]}}
                for t in TOOLS]


def parse_calls(text: str) -> list[dict] | None:
    try:
        from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls
        calls = _extract_tool_calls({"role": "assistant", "content": text}, text)
    except Exception:  # noqa: BLE001
        calls = None
    if not calls:
        return None
    out = []
    for tc in calls:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        out.append({"name": fn.get("name") or fn.get("tool_name"), "arguments": args})
    return out


class Cohere:
    def __init__(self, model: str, max_calls: int, min_interval: float):
        self.key = os.environ["COHERE_API_KEY"]
        self.model, self.max_calls, self.min_interval = model, max_calls, min_interval
        self.calls, self.last = 0, 0.0

    def chat(self, messages: list[dict]) -> str:
        if self.calls >= self.max_calls:
            raise RuntimeError("call budget exhausted")
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.0,
                               "max_tokens": 600}).encode()
            req = urllib.request.Request(API, data=body, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    self.calls += 1
                    data = json.loads(r.read())
                return "".join(p.get("text", "") for p in data["message"]["content"] if p.get("type") == "text")
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code == 429:
                    time.sleep(min(120, 15 * (attempt + 1)))
                    continue
                if e.code in (401, 402, 403) or "limit" in msg.lower() and "month" in msg.lower():
                    raise RuntimeError(f"HTTP {e.code}: {msg}")
                time.sleep(5)
        raise RuntimeError("too many retries")


def solve(task: Task, api: Cohere, sys_text: str, max_steps: int) -> list[dict]:
    """Run one episode; returns OpenAI-style messages (assistant tool_calls + tool results)."""
    S._local.session = S._global["session"] = Session(World.get(task.seed, task.variant))
    convo = [{"role": "system", "content": sys_text}, {"role": "user", "content": task.prompt}]
    record = [{"role": "user", "content": task.prompt}]
    for step in range(max_steps + 1):
        text = api.chat(convo)
        calls = parse_calls(text)
        if not calls or step == max_steps:
            record.append({"role": "assistant", "content": text.strip()})
            return record
        convo.append({"role": "assistant", "content": text})
        tcs = [{"id": f"call_{len(record)}_{i}", "name": c["name"], "arguments": c["arguments"]} for i, c in enumerate(calls)]
        record.append({"role": "assistant", "content": "", "tool_calls": tcs})
        folded = []
        for tc in tcs:
            out = call_tool(tc["name"] or "", tc["arguments"])
            record.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": out})
            folded.append(TOOL_RESULT_FORMAT.format(name=tc["name"], content=out))
        convo.append({"role": "user", "content": "\n".join(folded)})
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="c4ai-aya-expanse-32b")
    ap.add_argument("--types", nargs="+", default=["policy", "workflow", "analytics"])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.4, 0.45, 0.15])
    ap.add_argument("--max-calls", type=int, default=900)
    ap.add_argument("--min-interval", type=float, default=6.5, help="seconds between API calls")
    ap.add_argument("--max-steps", type=int, default=12)
    a = ap.parse_args()

    tasks = [Task.from_dict(json.loads(l)) for l in open(ROOT / "data" / f"{a.split}.jsonl", encoding="utf-8")]
    rng = random.Random(0)
    rng.shuffle(tasks)
    pools = {t: [x for x in tasks if x.task_type == t] for t in a.types}
    out_dir = OUT / "teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, kept_path = out_dir / "teacher_all.jsonl", out_dir / "teacher_verified.jsonl"
    done = set()
    if raw_path.exists():
        done = {json.loads(l)["task_id"] for l in open(raw_path, encoding="utf-8")}

    sys_text = SYSTEM_PROMPT + "\n\n" + TOOLS_FALLBACK_PROMPT.format(tools=json.dumps(tool_schemas(), indent=2))
    api = Cohere(a.model, a.max_calls, a.min_interval)
    stats = {"attempted": 0, "kept": 0, "by_type": {}}
    while api.calls < a.max_calls - a.max_steps:
        ttype = rng.choices(a.types, weights=a.weights[:len(a.types)])[0]
        pool = [t for t in pools[ttype] if t.task_id not in done]
        if not pool:
            break
        task = pool.pop()
        done.add(task.task_id)
        try:
            msgs = solve(task, api, sys_text, a.max_steps)
        except RuntimeError as e:
            print("stopping:", e, flush=True)
            break
        sig = analyse(task, from_messages(task.task_id, msgs))
        ok = sig["correct_and_grounded"] and sig["invalid_calls"] == 0 and sig["duplicate_calls"] == 0
        rec = {"task_id": task.task_id, "task_type": ttype, "messages": msgs, "verified": ok,
               "signals": {k: sig[k] for k in ("correct", "grounded", "n_calls", "invalid_calls", "duplicate_calls")}}
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if ok:
            with open(kept_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        stats["attempted"] += 1
        stats["kept"] += ok
        bt = stats["by_type"].setdefault(ttype, [0, 0])
        bt[0] += 1
        bt[1] += ok
        print(f"[{api.calls} calls] {task.task_id} correct={sig['correct']} kept={ok} "
              f"(kept {stats['kept']}/{stats['attempted']})", flush=True)
    stats["api_calls"] = api.calls
    (out_dir / "teacher_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
