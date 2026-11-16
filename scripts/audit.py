"""AuditKIT report over saved evaluation trajectories (CPU only).

    python scripts/audit.py            # writes results/auditkit/<label>__<split>.json + summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import auditkit.agent_eval as A

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "auditkit"
RUNS = {
    ("tiny_aya_global_minimal", "test_id"): "runs/baselines/out",
    ("tiny_aya_global_minimal", "test_ood"): "runs/baselines/out",
    ("tiny_aya_global_prompted", "test_id"): "runs/baselines/out",
    ("tiny_aya_global_sft", "test_id"): "runs/evals/out",
    ("tiny_aya_global_sft", "test_ood"): "runs/evals/out",
    ("gate_sft", "val"): "runs/gate/out",
    ("gate_candidate", "val"): "runs/gate/out",
}
from claimsdesk.tools import TOOLS  # noqa: E402
TOOL_NAMES = [t.__name__ for t in TOOLS]
SCORERS = ["task_completion", "tool_call_validity", "redundant_tool_calls", "agent_loop_detection"]


def audit(label: str, split: str, base: str) -> dict:
    d = ROOT / base / label / split
    eps = A.episodes_from_agenttune(str(d / "trajectories.jsonl"))
    recs = [json.loads(l) for l in open(d / "trajectories.jsonl", encoding="utf-8")]
    sigs = {json.loads(l)["task_id"]: json.loads(l) for l in open(d / "signals.jsonl", encoding="utf-8")}
    cases = []
    for e, r in zip(eps, recs):
        tid = r["metadata"]["task_id"]
        e.case_id = tid
        s = sigs[tid]
        # outcome = replay-verified, grounded correctness from our environment (not string match)
        pred = A.CustomPredicate(lambda ep, case, _s=s: bool(_s.get("correct") and _s.get("grounded")),
                                 version="claimsdesk-replay-v1", name="grounded_correct")
        cases.append(A.AgentCase(id=tid, task=r["task"], category=s.get("task_type", "generic"), outcome=pred,
                               allowed_tools=TOOL_NAMES))
    res = A.AgentEvalRunner().run(A.AgentEvalSpec(cases=cases, episodes=eps, mode="recorded",
                                                  scorers=SCORERS, name=f"{label}/{split}"))
    return res


def to_jsonable(x):
    for attr in ("to_dict", "model_dump", "dict"):
        if hasattr(x, attr):
            return getattr(x, attr)()
    return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))


def summarise_rows(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n, "outcome_success": sum(r["outcome"]["verdict"] == "success" for r in rows) / n}
    for k in ("tool_call_validity", "redundant_tool_calls", "agent_loop_detection"):
        v = [r["diagnostics"][k]["value"] for r in rows if k in r.get("diagnostics", {}) and r["diagnostics"][k]["value"] is not None]
        if v:
            out[k] = round(sum(v) / len(v), 4)
    out["mean_tool_calls"] = round(sum(r["resource_use"]["n_tool_calls"] or 0 for r in rows) / n, 3)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for (label, split), base in RUNS.items():
        if not (ROOT / base / label / split / "trajectories.jsonl").exists():
            print("skip", label, split); continue
        res = audit(label, split, base)
        d = to_jsonable(res)
        d.pop("episodes", None)  # episodes are the trajectories themselves (published separately)
        (OUT / f"{label}__{split}.json").write_text(json.dumps(d, default=str))
        summary[f"{label}/{split}"] = summarise_rows(d["rows"])
        print(label, split, summary[f"{label}/{split}"], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
