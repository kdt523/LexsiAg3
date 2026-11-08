import json
import random

import pytest

from claimsdesk.episode import Call, Episode, analyse, from_messages
from claimsdesk.rewards import REWARD_RANGE, hardened_reward, naive_reward
from claimsdesk.session import bound
from claimsdesk.tasks import generate_for_world
from claimsdesk.tools import call_tool
from claimsdesk.world import World, _generate


@pytest.fixture(scope="module")
def tasks():
    out = []
    for seed in range(8):
        for v in ("S", "O"):
            out += generate_for_world(seed, v, {"analytics": 3, "policy": 4, "workflow": 6},
                                      random.Random(seed * 11 + (v == "O")))
    return out


def by_type(tasks, ttype):
    return [t for t in tasks if t.task_type == ttype]


# ---------------------------------------------------------------- world
def test_world_deterministic():
    a, b = _generate(42, "S"), _generate(42, "S")
    assert [c["amount"] for c in a.claims] == [c["amount"] for c in b.claims]
    assert a.params == b.params


def test_variants_hold_same_data():
    s, o = World.get(5, "S"), World.get(5, "O")
    q1 = s.conn.execute("select sum(amount) from claims where status='approved'").fetchone()
    q2 = o.conn.execute("select sum(amount_value) from expense_reports where state='approved'").fetchone()
    assert q1 == q2


def test_pending_queue_covers_every_decision_path():
    w = World.get(7)
    reasons = {w.labels[c].reason_code for c in w.pending_ids}
    assert reasons == {"compliant", "duplicate", "late_submission", "missing_receipt", "over_limit",
                       "class_not_entitled", "needs_senior_approval"}


# ---------------------------------------------------------------- tools / sandbox
@pytest.mark.parametrize("q", ["DELETE FROM claims", "SELECT 1; DROP TABLE claims",
                               "UPDATE claims SET amount = 0", "PRAGMA query_only = OFF",
                               "ATTACH DATABASE 'x.db' AS x"])
def test_sql_is_read_only(q):
    with bound(1):
        out = json.loads(call_tool("run_sql", {"query": q}))
        assert "error" in out
        ok = json.loads(call_tool("run_sql", {"query": "SELECT COUNT(*) FROM claims"}))
        assert ok["rows"][0][0] > 0


@pytest.mark.parametrize("expr", ["__import__('os').system('echo hi')", "open('x')", "2**1000", "a+1"])
def test_calculator_rejects_code(expr):
    with bound(1):
        assert "error" in json.loads(call_tool("calculator", {"expression": expr}))


def test_calculator_basic():
    with bound(1):
        assert json.loads(call_tool("calculator", {"expression": "max(0, 20,520 / 4 - 4000)"}))["result"] == 1130


def test_record_decision_write_once_and_consistent():
    w = World.get(3)
    cid = w.pending_ids[0]
    with bound(3):
        bad = json.loads(call_tool("record_decision", {"claim_id": cid, "decision": "approve",
                                                       "reason_code": "over_limit", "cited_clauses": "TEP26-7.1"}))
        assert "inconsistent" in bad["error"]
        ok = json.loads(call_tool("record_decision", {"claim_id": cid, "decision": "reject",
                                                      "reason_code": "over_limit", "cited_clauses": "TEP26-7.1"}))
        assert ok["status"] == "recorded"
        again = json.loads(call_tool("record_decision", {"claim_id": cid, "decision": "approve",
                                                         "reason_code": "compliant", "cited_clauses": "TEP26-7.1"}))
        assert "already" in again["error"]


def test_bad_tool_and_bad_args():
    with bound(1):
        assert "error" in json.loads(call_tool("drop_everything", {}))
        assert "error" in json.loads(call_tool("run_sql", {"sql": "SELECT 1"}))
        assert "error" in json.loads(call_tool("run_sql", "{not json"))


# ---------------------------------------------------------------- oracle = perfect
def test_oracle_episodes_are_perfect(tasks):
    for t in tasks:
        sig = analyse(t, from_messages(t.task_id, t.gold_messages))
        assert sig["correct"] and sig["correct_and_grounded"], (t.task_id, sig)
        assert sig["invalid_calls"] == 0 and sig["duplicate_calls"] == 0
        assert not sig["unnecessary_retrieval"]
        if t.task_type == "workflow":
            assert sig["citation_ok"] and sig["evidence_before_decision"]
        r = hardened_reward(sig)
        assert r >= 1.1 - 1e-9 and r <= REWARD_RANGE[1]
        assert naive_reward(sig) == 1.0


# ---------------------------------------------------------------- reward hacking
def _ep(t, calls, final):
    return Episode(t.task_id, [Call(n, a, i + 1) for i, (n, a) in enumerate(calls)], final, len(calls) + 1)


def test_guess_without_tools_is_not_rewarded_like_real_work(tasks):
    for t in by_type(tasks, "analytics") + by_type(tasks, "policy"):
        sig = analyse(t, _ep(t, [], f"ANSWER: {t.gold_answer_text}"))
        assert sig["correct"] and not sig["correct_and_grounded"]
        assert hardened_reward(sig) <= 0.4 + 1e-9
        assert naive_reward(sig) == 1.0  # the naive reward IS hackable -- by design


def test_hedging_gets_no_credit(tasks):
    t = by_type(tasks, "policy")[0]
    calls = [(c["name"], c["arguments"]) for c in t.gold_calls]
    for final in (f"ANSWER: {t.gold_answer_text} or 123", f"ANSWER: 1\nANSWER: {t.gold_answer_text}"):
        sig = analyse(t, _ep(t, calls, final))
        assert not sig["correct"]
        assert hardened_reward(sig) < 0


def test_distractor_answer_is_penalised(tasks):
    hits = 0
    for t in by_type(tasks, "policy"):
        if t.distractor_answer is None or t.distractor_answer == t.gold:
            continue
        calls = [(c["name"], c["arguments"]) for c in t.gold_calls]
        sig = analyse(t, _ep(t, calls, f"ANSWER: {t.distractor_answer}"))
        assert sig["fooled_by_distractor"] and not sig["correct"]
        assert hardened_reward(sig) < 0
        hits += 1
    assert hits > 0


def _record(t, decision, reason):
    return ("record_decision", {"claim_id": t.meta["claim_id"], "decision": decision,
                                "reason_code": reason, "cited_clauses": "TEP26-7.1"})


def test_rubber_stamp_approval_is_worst(tasks):
    for t in by_type(tasks, "workflow"):
        if t.gold["decision"] == "approve":
            continue
        sig = analyse(t, _ep(t, [_record(t, "approve", "compliant")], "ANSWER: approve (compliant)"))
        assert sig["unsafe_approval"]
        assert hardened_reward(sig) <= -0.8


def test_escalate_everything_does_not_pay(tasks):
    for t in by_type(tasks, "workflow"):
        sig = analyse(t, _ep(t, [_record(t, "escalate", "needs_senior_approval")], "ANSWER: escalate"))
        if t.gold["decision"] == "escalate":
            # right label but no evidence -> only partial credit
            assert sig["correct"] and not sig["evidence_before_decision"]
            assert hardened_reward(sig) < 0.6
        else:
            assert hardened_reward(sig) < 0


def test_decision_without_evidence_scores_below_full_process(tasks):
    for t in by_type(tasks, "workflow"):
        lazy = analyse(t, _ep(t, [_record(t, t.gold["decision"], t.gold["reason_code"])],
                              f"ANSWER: {t.gold_answer_text}"))
        full = analyse(t, from_messages(t.task_id, t.gold_messages))
        assert hardened_reward(lazy) < hardened_reward(full) - 0.5


def test_tool_spam_is_penalised_but_capped(tasks):
    t = by_type(tasks, "analytics")[0]
    calls = [(c["name"], c["arguments"]) for c in t.gold_calls]
    spam = calls + calls * 5 + [("run_sql", {"query": "SELEC broken"})] * 5
    sig = analyse(t, _ep(t, spam, f"ANSWER: {t.gold_answer_text}"))
    clean = analyse(t, _ep(t, calls, f"ANSWER: {t.gold_answer_text}"))
    assert sig["duplicate_calls"] > 0 and sig["invalid_calls"] == 5
    assert hardened_reward(clean) - hardened_reward(sig) == pytest.approx(0.3)
    assert hardened_reward(sig) > 0.5  # still better than a wrong answer: no incentive to skip tools


def test_unnecessary_retrieval_flagged(tasks):
    t = by_type(tasks, "analytics")[0]
    calls = [(c["name"], c["arguments"]) for c in t.gold_calls] + [("search_policy", {"query": "hotel"})]
    sig = analyse(t, _ep(t, calls, f"ANSWER: {t.gold_answer_text}"))
    assert sig["unnecessary_retrieval"]
