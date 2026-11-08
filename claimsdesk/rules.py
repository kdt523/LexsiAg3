"""Deterministic claims-decision rules engine: the single source of ground truth.

Rules are applied in a fixed priority order, mirroring the current policy (TEP26):

1. duplicate            (8.1)  -> reject
2. late_submission      (2.1)  -> reject
3. missing_receipt      (2.1)  -> reject
4. class_not_entitled   (5.1)  -> reject   (airfare)
5. over_limit           (3.x / 4.x / 6.1) -> reject
6. needs_senior_approval(7.1)  -> escalate
7. compliant                   -> approve
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from .config import DECISION_FOR_REASON
from .policy import PolicyParams

PREFIX = "TEP26"


@dataclass
class RuleResult:
    decision: str
    reason_code: str
    deciding_clause: str
    amount_inr: float
    facts: dict = field(default_factory=dict)
    # every clause a careful reviewer must consult to reach this decision
    required_clauses: list[str] = field(default_factory=list)


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def claim_order(c: dict) -> tuple[str, int]:
    """Submission order: (submitted_date, numeric part of claim id)."""
    digits = re.sub(r"\D", "", str(c.get("claim_id", ""))) or "0"
    return (c["submitted_date"], int(digits))


def find_duplicates(claim: dict, all_claims: list[dict]) -> list[str]:
    """Ids of EARLIER-submitted claims with same employee, category, expense date, amount."""
    me = claim_order(claim)
    return [c["claim_id"] for c in all_claims
            if c["claim_id"] != claim["claim_id"] and c["emp_id"] == claim["emp_id"]
            and c["category"] == claim["category"] and c["expense_date"] == claim["expense_date"]
            and abs(c["amount"] - claim["amount"]) < 1e-9 and claim_order(c) < me]


def limit_clause_for(claim: dict, params: PolicyParams) -> tuple[str | None, float | None, str]:
    """(clause id, limit value, comparison unit) for a claim's category, or Nones."""
    cat = claim["category"]
    grade = claim["_grade"]
    if cat == "hotel":
        if claim["currency"] == "USD":
            return f"{PREFIX}-3.4", float(params.intl_hotel_usd[grade]), "usd_per_night"
        tier = params.city_tier.get(claim["city"], 3)
        return f"{PREFIX}-3.{tier}", float(params.hotel_limit[(grade, tier)]), "inr_per_night"
    if cat == "meals":
        return f"{PREFIX}-4.1", float(params.meal_per_diem[grade]), "inr"
    if cat == "client_entertainment":
        return f"{PREFIX}-4.2", float(params.entertainment_cap), "inr"
    if cat == "local_transport":
        return f"{PREFIX}-6.1", float(params.local_transport_cap), "inr"
    return None, None, ""


def evaluate_claim(
    claim: dict,
    params: PolicyParams,
    fx: Callable[[str, str], float],
    all_claims: list[dict],
) -> RuleResult:
    """Decide one claim. ``claim`` uses logical column names plus ``_grade``."""
    rate = 1.0 if claim["currency"] == "INR" else fx(claim["currency"], claim["expense_date"])
    amount_inr = round(claim["amount"] * rate, 2)
    facts: dict = {"amount_inr": amount_inr, "fx_rate": rate, "grade": claim["_grade"]}
    req = [f"{PREFIX}-8.1"]

    def result(reason: str, clause: str) -> RuleResult:
        return RuleResult(DECISION_FOR_REASON[reason], reason, clause, amount_inr, facts,
                          list(dict.fromkeys(req)))

    dups = find_duplicates(claim, all_claims)
    facts["duplicates"] = dups
    if dups:
        return result("duplicate", f"{PREFIX}-8.1")

    req.append(f"{PREFIX}-2.1")
    lag = days_between(claim["expense_date"], claim["submitted_date"])
    facts["submission_lag_days"] = lag
    if lag > params.submission_days:
        return result("late_submission", f"{PREFIX}-2.1")
    if not claim["receipt_attached"] and amount_inr > params.receipt_threshold:
        return result("missing_receipt", f"{PREFIX}-2.1")

    if claim["category"] == "airfare":
        req.append(f"{PREFIX}-5.1")
        if claim["cabin_class"] == "business":
            entitled = claim["_grade"] in ("G3", "G4") and claim["flight_hours"] >= params.business_min_hours
            facts["business_entitled"] = entitled
            if not entitled:
                return result("class_not_entitled", f"{PREFIX}-5.1")
    else:
        clause, limit, unit = limit_clause_for(claim, params)
        req.append(clause)  # type: ignore[arg-type]
        if claim["category"] == "hotel":
            if unit == "inr_per_night":
                req.append(f"{PREFIX}-A.{params.city_tier.get(claim['city'], 3)}")
            nightly = claim["amount"] / claim["nights"]  # in claim currency (INR or USD)
            facts.update(nightly_rate=round(nightly, 2), limit=limit, unit=unit)
            if nightly > limit + 1e-9:
                return result("over_limit", clause)  # type: ignore[arg-type]
        else:
            facts.update(limit=limit, unit=unit)
            if amount_inr > limit + 1e-9:  # type: ignore[operator]
                return result("over_limit", clause)  # type: ignore[arg-type]

    req.append(f"{PREFIX}-7.1")
    if amount_inr > params.approval_threshold:
        return result("needs_senior_approval", f"{PREFIX}-7.1")
    deciding = req[-2] if len(req) >= 2 else f"{PREFIX}-7.1"
    return result("compliant", deciding)
