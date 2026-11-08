"""Reward functions: a NAIVE baseline and a HARDENED, hack-resistant reward.

Both are pure functions of the replay analysis (``episode.analyse``), so the model
cannot influence its reward by narrating fake tool results.

Hardened-reward defences (each maps to a hack we test for):

=============================  ==================================================
Hack                           Defence
=============================  ==================================================
Guessing a plausible number    Correct-but-ungrounded answers earn only 0.3; data
                               and limits are regenerated per world.
Hedging ("7500 or 8000")       Multiple numbers / multiple ANSWER lines = no answer.
Quoting a superseded clause    Distractor answer and distractor citations penalised.
Rubber-stamp approvals         Unsafe approval is the largest penalty (-1.0).
Escalate-everything            Over-escalation penalised (-0.25), no outcome credit.
Decide without evidence        Workflow credit needs lookup + rule evidence first.
Tool spam / loops              Invalid, duplicate and over-budget calls penalised.
Gaming the penalty cap         Efficiency penalty capped (0.3) so it never outweighs
                               correctness -- no incentive to skip tools entirely.
=============================  ==================================================
"""

from __future__ import annotations

from typing import Any

W = {
    "correct_grounded": 1.0,
    "correct_ungrounded": 0.3,
    "no_output": -0.5,
    "unsafe_approval": -1.0,
    "over_escalation": -0.25,
    "fooled_by_distractor": -0.25,
    "format": 0.1,
    "citation_ok": 0.1,
    "cited_distractor": -0.2,
    "invalid_call": 0.05,
    "duplicate_call": 0.05,
    "over_budget_call": 0.02,
    "unnecessary_retrieval": 0.05,
    "efficiency_cap": 0.3,
}


def naive_reward(sig: dict[str, Any]) -> float:
    """Outcome only: 1 if the answer (or, for claims, just the decision) matches."""
    if sig["task_type"] == "workflow":
        return 1.0 if sig.get("decision_correct") else 0.0
    return 1.0 if sig["correct"] else 0.0


def hardened_reward(sig: dict[str, Any], weights: dict[str, float] | None = None) -> float:
    w = {**W, **(weights or {})}
    r = 0.0
    if sig["correct"]:
        r += w["correct_grounded"] if sig["grounded"] else w["correct_ungrounded"]
    elif not sig["has_output"]:
        r += w["no_output"]
    else:
        if sig["unsafe_approval"]:
            r += w["unsafe_approval"]
        elif sig["over_escalation"]:
            r += w["over_escalation"]
        if sig["fooled_by_distractor"]:
            r += w["fooled_by_distractor"]

    if sig["has_output"] and not sig["malformed_call"]:
        r += w["format"]
    if sig["task_type"] == "workflow":
        if sig["citation_ok"]:
            r += w["citation_ok"]
        if sig["cited_distractor"]:
            r += w["cited_distractor"]

    pen = (w["invalid_call"] * sig["invalid_calls"] + w["duplicate_call"] * sig["duplicate_calls"]
           + w["over_budget_call"] * sig["over_budget_calls"]
           + (w["unnecessary_retrieval"] if sig["unnecessary_retrieval"] else 0.0))
    r -= min(pen, w["efficiency_cap"])
    return round(r, 4)


REWARD_RANGE = (-1.5, 1.2)
