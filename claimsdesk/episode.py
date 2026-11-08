"""Normalise any rollout format into an ``Episode`` and verify it by replay.

Sources supported:
* AgentTune ``trajectories.jsonl`` records / ``Trajectory`` objects
* OpenAI-style message lists (our oracle, API teachers, our eval harness)

``replay`` re-executes the episode's tool calls, in order, against a FRESH session of
the task's world. Rewards and metrics are computed from the replayed results and the
replayed final state -- never from what the model *claims* a tool returned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .answers import NUM_RE, answer_matches, extract_answer, number_tolerance, parse_decision, parse_number
from .session import bound
from .tasks import Task
from .tools import call_tool

RETRIEVAL_TOOLS = ("search_policy", "get_clause")
_CALL_HINT = re.compile(r'"tool_name"|"parameters"|<\|START_ACTION\|>|"name"\s*:\s*"(get_schema|run_sql|search_policy|'
                        r'get_clause|get_fx_rate|calculator|check_duplicates|record_decision)"')


@dataclass
class Call:
    name: str
    arguments: Any
    turn: int


@dataclass
class Episode:
    task_id: str
    calls: list[Call]
    final_text: str
    n_turns: int
    source: str = ""
    raw_turns: list[str] = field(default_factory=list)


def _args(a: Any) -> Any:
    if isinstance(a, str):
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else a
        except json.JSONDecodeError:
            return a
    return a if a is not None else {}


def from_messages(task_id: str, messages: list[dict], source: str = "messages") -> Episode:
    calls: list[Call] = []
    turn = 0
    final = ""
    raws: list[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        turn += 1
        tcs = m.get("tool_calls") or []
        raws.append(str(m.get("content") or ""))
        if tcs:
            for tc in tcs:
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                calls.append(Call(fn.get("name") or "", _args(fn.get("arguments", fn.get("parameters"))), turn))
        else:
            final = str(m.get("content") or "")
    return Episode(task_id, calls, final, turn, source, raws)


def from_agenttune(task_id: str, record: Any) -> Episode:
    """From an AgentTune trajectory (``Trajectory`` object or its ``to_dict()``)."""
    rec = record.to_dict() if hasattr(record, "to_dict") else record
    calls: list[Call] = []
    raws: list[str] = []
    turn = 0
    for st in rec.get("steps", []):
        action = st.get("action") or {}
        tcs = action.get("tool_calls") if isinstance(action, dict) else None
        if not tcs and isinstance(action, dict) and action.get("name"):
            tcs = [action]
        if tcs:
            turn += 1
            raws.append(str(st.get("thought") or ""))
            for tc in tcs:
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                calls.append(Call(fn.get("name") or fn.get("tool_name") or "",
                                  _args(fn.get("arguments", fn.get("parameters"))), turn))
    final = str(rec.get("final_response") or "")
    return Episode(task_id, calls, final, turn + 1, "agenttune", raws)


# ---------------------------------------------------------------------------
@dataclass
class Replay:
    results: list[dict]
    decisions: dict[str, dict]
    audit: list[dict]


def replay(task: Task, ep: Episode) -> Replay:
    with bound(task.seed, task.variant) as sess:
        results = []
        for c in ep.calls:
            out = call_tool(c.name, c.arguments)
            try:
                results.append(json.loads(out))
            except json.JSONDecodeError:
                results.append({"error": "unparseable tool output"})
        return Replay(results, dict(sess.decisions), list(sess.audit))


def _canon(c: Call) -> str:
    try:
        return c.name + "|" + json.dumps(c.arguments, sort_keys=True)
    except TypeError:
        return c.name + "|" + str(c.arguments)


def _numbers_in(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for n in NUM_RE.findall(obj):
            try:
                out.append(float(n.replace(",", "")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _numbers_in(v)
    return out


def _clauses_seen(results: list[dict]) -> set[str]:
    seen: set[str] = set()
    for r in results:
        if "results" in r and isinstance(r["results"], list):
            seen |= {x.get("clause_id") for x in r["results"] if isinstance(x, dict)}
        elif "clause_id" in r and "text" in r:
            seen.add(r["clause_id"])
    return seen


# ---------------------------------------------------------------------------
def analyse(task: Task, ep: Episode) -> dict[str, Any]:
    """All per-episode signals used by rewards and metrics (pure function of replay)."""
    rp = replay(task, ep)
    n = len(ep.calls)
    errors = [i for i, r in enumerate(rp.results) if "error" in r]
    seen: set[str] = set()
    dups = 0
    for c in ep.calls:
        k = _canon(c)
        if k in seen:
            dups += 1
        seen.add(k)
    retrieval_calls = sum(c.name in RETRIEVAL_TOOLS for c in ep.calls)
    sql_calls = sum(c.name == "run_sql" for c in ep.calls)
    clauses_seen = _clauses_seen(rp.results)
    answer, ans_status = extract_answer(ep.final_text)
    malformed = bool(_CALL_HINT.search(ep.final_text or "")) and ans_status != "ok"

    sig: dict[str, Any] = {
        "task_id": task.task_id, "task_type": task.task_type, "template": task.template,
        "difficulty": task.difficulty, "variant": task.variant,
        "n_calls": n, "n_turns": ep.n_turns, "invalid_calls": len(errors), "duplicate_calls": dups,
        "retrieval_calls": retrieval_calls, "sql_calls": sql_calls,
        "over_budget_calls": max(0, n - task.step_budget),
        "answer": answer, "answer_status": ans_status, "malformed_call": malformed,
        "gold_clauses_retrieved": all(cl in clauses_seen for cl in task.gold_clauses) if task.gold_clauses else None,
        "distractor_clauses_seen": sorted(c for c in clauses_seen if c and not c.startswith("TEP26")),
    }

    if task.answer_type == "decision":
        cid = task.meta["claim_id"]
        rec = rp.decisions.get(cid)
        sig["decision"] = rec["decision"] if rec else None
        sig["reason_code"] = rec["reason_code"] if rec else None
        cited = rec["cited_clauses"] if rec else []
        sig["cited_clauses"] = cited
        correct = bool(rec) and rec["decision"] == task.gold["decision"] and rec["reason_code"] == task.gold["reason_code"]
        sig["decision_correct"] = bool(rec) and rec["decision"] == task.gold["decision"]
        sig["correct"] = correct
        sig["has_output"] = rec is not None
        sig["unsafe_approval"] = bool(rec) and rec["decision"] == "approve" and task.gold["decision"] != "approve"
        sig["over_escalation"] = bool(rec) and rec["decision"] == "escalate" and task.gold["decision"] != "escalate"
        sig["citation_ok"] = bool(cited) and all(c.startswith("TEP26") for c in cited) and bool(
            set(cited) & set(task.acceptable_citations))
        sig["cited_distractor"] = any(not c.startswith("TEP26") for c in cited)
        # evidence before decision: the record call must come after the claim lookup and
        # after the deciding rule's evidence was obtained.
        rec_idx = next((i for i, c in enumerate(ep.calls) if c.name == "record_decision"
                        and "error" not in rp.results[i]), None)
        evidence = False
        if rec_idx is not None:
            prior = ep.calls[:rec_idx]
            prior_res = rp.results[:rec_idx]
            looked_up = any(c.name == "run_sql" and cid in json.dumps(c.arguments) and "error" not in r
                            for c, r in zip(prior, prior_res))
            if task.gold["reason_code"] == "duplicate":
                rule_ev = any(c.name == "check_duplicates" for c in prior)
            else:
                rule_ev = bool(set(task.acceptable_citations) & _clauses_seen(prior_res))
            evidence = looked_up and rule_ev
        sig["evidence_before_decision"] = evidence
        sig["grounded"] = evidence
        sig["fooled_by_distractor"] = False
    else:
        correct = ans_status == "ok" and answer_matches(task.answer_type, task.gold, answer)
        sig["correct"] = correct
        # a hedge ("7500 or 8000") or a non-numeric reply to a numeric question is no answer
        if ans_status == "ok" and task.answer_type == "number" and parse_number(answer or "") is None:
            ans_status = sig["answer_status"] = "unparseable"
        sig["has_output"] = ans_status == "ok"
        grounded = False
        if ans_status == "ok":
            ok_results = [r for r in rp.results if "error" not in r]
            if task.answer_type == "number":
                v = parse_number(answer or "")
                if v is not None:
                    pool = _numbers_in(ok_results)
                    grounded = any(abs(v - x) <= number_tolerance(v) for x in pool)
            else:
                blob = json.dumps(ok_results).lower()
                grounded = bool(answer) and answer.lower() in blob
        sig["grounded"] = grounded
        fooled = False
        if task.distractor_answer is not None and ans_status == "ok" and not correct:
            fooled = answer_matches("number", task.distractor_answer, answer)
        sig["fooled_by_distractor"] = fooled
        sig["unsafe_approval"] = False
        sig["over_escalation"] = False
        sig["citation_ok"] = None
        sig["cited_distractor"] = False
        sig["evidence_before_decision"] = None
    sig["correct_and_grounded"] = bool(sig["correct"] and sig["grounded"])
    sig["unnecessary_retrieval"] = (not task.requires_retrieval) and retrieval_calls > 0
    sig["missed_retrieval"] = task.requires_retrieval and retrieval_calls == 0
    return sig


__all__ = ["Call", "Episode", "Replay", "analyse", "from_agenttune", "from_messages", "parse_decision", "replay"]
