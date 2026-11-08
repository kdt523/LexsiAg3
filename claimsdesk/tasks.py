"""Difficulty-graded, grounded multi-hop task generator + verified oracle solutions.

Three task families share one world:

* ``analytics`` -- business questions answered from the database (data agent).
* ``policy``    -- multi-hop questions over the policy corpus (+DB), with stale and
                   wrong-entity distractor documents (RAG / multi-hop).
* ``workflow``  -- end-to-end processing of a pending claim with rules, approvals and
                   an audit-trail write (decision agent).

Every task's gold answer is computed TWICE, independently:
  1. directly from the world's Python data (``gold``), and
  2. by a scripted oracle agent that only sees TOOL OUTPUTS (SQL rows, policy text).
A task is emitted only if both agree, which proves it is answerable from the tools.
The oracle's trajectory doubles as the reference route for AuditKIT and as expert
demonstrations.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable

from .answers import answer_matches, format_gold
from .config import CATEGORIES, DEPARTMENTS, GRADE_ORDER, GRADES, MONTHS, QUARTER_MONTHS, YEAR
from .prompt import user_prompt
from .schema import render_sql
from .session import bound
from .tools import call_tool
from .world import World

CAT_LABEL = {"hotel": "hotel", "meals": "meals", "airfare": "airfare",
             "local_transport": "local transport", "client_entertainment": "client entertainment"}


@dataclass
class Task:
    task_id: str
    workspace: str
    seed: int
    variant: str
    task_type: str
    template: str
    difficulty: int
    question: str
    prompt: str
    answer_type: str
    gold: Any
    gold_answer_text: str
    gold_calls: list[dict]
    gold_messages: list[dict]
    gold_clauses: list[str]
    acceptable_citations: list[str]
    distractor_answer: Any
    requires_retrieval: bool
    requires_sql: bool
    step_budget: int
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(**d)


@dataclass
class Draft:
    template: str
    task_type: str
    difficulty: int
    question: str
    answer_type: str
    gold: Any
    plan: Callable[["Oracle"], Any]
    gold_clauses: list[str] = field(default_factory=list)
    acceptable_citations: list[str] = field(default_factory=list)
    distractor: Any = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Oracle: a scripted expert that acts ONLY through the tools
# ---------------------------------------------------------------------------
class Oracle:
    def __init__(self, world: World) -> None:
        self.world = world
        self.messages: list[dict] = []
        self.calls: list[dict] = []
        self._schema_done = False

    def call(self, name: str, **arguments: Any) -> dict:
        cid = f"call_{len(self.calls) + 1}"
        self.messages.append({"role": "assistant", "content": "",
                              "tool_calls": [{"id": cid, "name": name, "arguments": arguments}]})
        out = call_tool(name, arguments)
        self.messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": out})
        self.calls.append({"name": name, "arguments": arguments})
        res = json.loads(out)
        if "error" in res:
            raise OracleError(f"{name}{arguments} -> {res['error']}")
        return res

    def schema(self) -> None:
        if not self._schema_done:
            self.call("get_schema")
            self._schema_done = True

    def sql(self, template: str) -> list[list]:
        self.schema()
        return self.call("run_sql", query=render_sql(template, self.world.view))["rows"]

    def scalar(self, template: str) -> Any:
        rows = self.sql(template)
        return rows[0][0] if rows else None

    def policy(self, query: str, want: str) -> str:
        res = self.call("search_policy", query=query)["results"]
        for r in res:
            if r["clause_id"] == want:
                return r["text"]
        return self.call("get_clause", clause_id=want)["text"]

    def calc(self, expr: str) -> float:
        return self.call("calculator", expression=expr)["result"]


class OracleError(RuntimeError):
    pass


def _money(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise OracleError(f"pattern {pattern!r} not found in clause text")
    return float(m.group(1).replace(",", ""))


def grade_value(text: str, grade: str) -> float:
    return _money(text, rf"Grade {grade} \([^)]*\): (?:INR|USD) ([\d,]+)")


def tier_from_annex(o: Oracle, city: str) -> int:
    res = o.call("search_policy", query=f"Annex A city classification tier {city}")["results"]
    for r in res:
        if r["status"] == "current" and r["title"].startswith("Annex A") and city in r["text"]:
            return int(re.search(r"Tier (\d)", r["title"]).group(1))  # type: ignore[union-attr]
    tier = o.world.params.city_tier.get(city, 3)
    text = o.call("get_clause", clause_id=f"TEP26-A.{tier}")["text"]
    if city not in text and tier != 3:
        raise OracleError(f"{city} not in annex")
    return tier


# ---------------------------------------------------------------------------
# helpers over world data (the independent Python gold)
# ---------------------------------------------------------------------------
def _month_of(iso: str) -> int:
    return date.fromisoformat(iso).month


def _q_bounds(q: int) -> tuple[str, str]:
    m0, m1 = QUARTER_MONTHS[q]
    last = {3: 31, 6: 30, 9: 30, 12: 31}[m1]
    return f"{YEAR}-{m0:02d}-01", f"{YEAR}-{m1:02d}-{last}"


def _m_bounds(m: int) -> tuple[str, str]:
    import calendar
    return f"{YEAR}-{m:02d}-01", f"{YEAR}-{m:02d}-{calendar.monthrange(YEAR, m)[1]:02d}"


def _emp(w: World) -> dict[str, dict]:
    return {e["emp_id"]: e for e in w.employees}


# ---------------------------------------------------------------------------
# ANALYTICS templates
# ---------------------------------------------------------------------------
def t_pending_count(w: World, rng: random.Random) -> Draft | None:
    cat = rng.choice(CATEGORIES)
    gold = sum(1 for c in w.claims if c["category"] == cat and c["status"] == "pending")
    if gold == 0:
        return None
    sql = (f"SELECT COUNT(*) FROM {{claims}} WHERE {{claims.category}} = '{cat}' "
           f"AND {{claims.status}} = 'pending'")
    return Draft("pending_count", "analytics", 1,
                 f"How many {CAT_LABEL[cat]} claims are currently pending?",
                 "number", gold, lambda o: o.scalar(sql))


def t_employee_month_count(w: World, rng: random.Random) -> Draft | None:
    e = rng.choice(w.employees)
    m = rng.randint(1, 9)
    lo, hi = _m_bounds(m)
    gold = sum(1 for c in w.claims if c["emp_id"] == e["emp_id"] and lo <= c["submitted_date"] <= hi)
    if gold == 0:
        return None
    sql = (f"SELECT COUNT(*) FROM {{claims}} c JOIN {{employees}} e ON c.{{claims.emp_id}} = e.{{employees.emp_id}} "
           f"WHERE e.{{employees.name}} = '{e['name']}' AND c.{{claims.submitted_date}} BETWEEN '{lo}' AND '{hi}'")
    return Draft("employee_month_count", "analytics", 2,
                 f"How many expense claims did {e['name']} submit in {MONTHS[m - 1]} {YEAR} (by submission date)?",
                 "number", gold, lambda o: o.scalar(sql))


def t_category_quarter_total(w: World, rng: random.Random) -> Draft | None:
    cat = rng.choice(CATEGORIES)
    q = rng.randint(1, 3)
    lo, hi = _q_bounds(q)
    vals = [c["amount"] for c in w.claims if c["category"] == cat and c["status"] == "approved"
            and c["currency"] == "INR" and lo <= c["expense_date"] <= hi]
    if not vals:
        return None
    gold = round(sum(vals), 2)
    sql = (f"SELECT SUM({{claims.amount}}) FROM {{claims}} WHERE {{claims.category}} = '{cat}' "
           f"AND {{claims.status}} = 'approved' AND {{claims.currency}} = 'INR' "
           f"AND {{claims.expense_date}} BETWEEN '{lo}' AND '{hi}'")
    return Draft("category_quarter_total", "analytics", 2,
                 f"What is the total amount of approved {CAT_LABEL[cat]} claims with expense dates in "
                 f"Q{q} {YEAR}? Count only INR-denominated claims; answer in INR.",
                 "number", gold, lambda o: o.scalar(sql))


def t_dept_month_total(w: World, rng: random.Random) -> Draft | None:
    cat = rng.choice(CATEGORIES)
    dept = rng.choice(DEPARTMENTS)
    m = rng.randint(1, 8)
    lo, hi = _m_bounds(m)
    emp = _emp(w)
    vals = [c["amount"] for c in w.claims if c["category"] == cat and c["status"] == "approved"
            and c["currency"] == "INR" and lo <= c["expense_date"] <= hi
            and emp[c["emp_id"]]["department"] == dept]
    if not vals:
        return None
    gold = round(sum(vals), 2)
    sql = (f"SELECT SUM(c.{{claims.amount}}) FROM {{claims}} c JOIN {{employees}} e "
           f"ON c.{{claims.emp_id}} = e.{{employees.emp_id}} WHERE e.{{employees.department}} = '{dept}' "
           f"AND c.{{claims.category}} = '{cat}' AND c.{{claims.status}} = 'approved' "
           f"AND c.{{claims.currency}} = 'INR' AND c.{{claims.expense_date}} BETWEEN '{lo}' AND '{hi}'")
    return Draft("dept_month_total", "analytics", 3,
                 f"What was the total approved {CAT_LABEL[cat]} spend by employees of the {dept} department "
                 f"with expense dates in {MONTHS[m - 1]} {YEAR}? Count only INR-denominated claims; answer in INR.",
                 "number", gold, lambda o: o.scalar(sql))


def t_top_office(w: World, rng: random.Random) -> Draft | None:
    cat = rng.choice(["hotel", "meals", "airfare", "local_transport"])
    q = rng.randint(1, 3)
    lo, hi = _q_bounds(q)
    emp = _emp(w)
    totals: dict[str, float] = {}
    for c in w.claims:
        if (c["category"] == cat and c["status"] == "approved" and c["currency"] == "INR"
                and lo <= c["expense_date"] <= hi):
            city = w.office(emp[c["emp_id"]]["office_id"])["city"]
            totals[city] = totals.get(city, 0.0) + c["amount"]
    if len(totals) < 2:
        return None
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    if ranked[0][1] - ranked[1][1] < 1.0:
        return None  # ambiguous winner
    gold = ranked[0][0]
    sql = (f"SELECT o.{{offices.city}}, SUM(c.{{claims.amount}}) AS total FROM {{claims}} c "
           f"JOIN {{employees}} e ON c.{{claims.emp_id}} = e.{{employees.emp_id}} "
           f"JOIN {{offices}} o ON e.{{employees.office_id}} = o.{{offices.office_id}} "
           f"WHERE c.{{claims.category}} = '{cat}' AND c.{{claims.status}} = 'approved' "
           f"AND c.{{claims.currency}} = 'INR' AND c.{{claims.expense_date}} BETWEEN '{lo}' AND '{hi}' "
           f"GROUP BY o.{{offices.city}} ORDER BY total DESC LIMIT 1")
    return Draft("top_office", "analytics", 3,
                 f"Which office city had the highest total approved {CAT_LABEL[cat]} spend in Q{q} {YEAR}, "
                 f"attributing each claim to the claimant's home office? Count only INR-denominated claims.",
                 "text", gold, lambda o: o.scalar(sql))


def t_budget_remaining(w: World, rng: random.Random) -> Draft | None:
    cat = rng.choice(CATEGORIES)
    dept = rng.choice(DEPARTMENTS)
    q = rng.randint(1, 3)
    lo, hi = _q_bounds(q)
    emp = _emp(w)
    spent = sum(c["amount"] for c in w.claims if c["category"] == cat and c["status"] == "approved"
                and c["currency"] == "INR" and lo <= c["expense_date"] <= hi
                and emp[c["emp_id"]]["department"] == dept)
    if spent == 0:
        return None
    budget = next(b["budget_inr"] for b in w.budgets
                  if b["department"] == dept and b["category"] == cat and b["quarter"] == f"{YEAR}-Q{q}")
    gold = round(budget - spent, 2)
    sql_b = (f"SELECT {{budgets.budget_inr}} FROM {{budgets}} WHERE {{budgets.department}} = '{dept}' "
             f"AND {{budgets.category}} = '{cat}' AND {{budgets.quarter}} = '{YEAR}-Q{q}'")
    sql_s = (f"SELECT SUM(c.{{claims.amount}}) FROM {{claims}} c JOIN {{employees}} e "
             f"ON c.{{claims.emp_id}} = e.{{employees.emp_id}} WHERE e.{{employees.department}} = '{dept}' "
             f"AND c.{{claims.category}} = '{cat}' AND c.{{claims.status}} = 'approved' "
             f"AND c.{{claims.currency}} = 'INR' AND c.{{claims.expense_date}} BETWEEN '{lo}' AND '{hi}'")

    def plan(o: Oracle) -> Any:
        b = o.scalar(sql_b)
        s = o.scalar(sql_s)
        return o.calc(f"{b} - {s}")

    return Draft("budget_remaining", "analytics", 4,
                 f"How much of the {dept} department's {YEAR}-Q{q} {CAT_LABEL[cat]} budget remains after "
                 f"subtracting its approved INR-denominated {CAT_LABEL[cat]} claims with expense dates in that "
                 f"quarter? Answer in INR (negative if over budget).",
                 "number", gold, plan)


# ---------------------------------------------------------------------------
# POLICY (RAG / multi-hop) templates
# ---------------------------------------------------------------------------
def t_hotel_limit_direct(w: World, rng: random.Random) -> Draft | None:
    g, t = rng.choice(GRADE_ORDER), rng.randint(1, 3)
    gold = w.params.hotel_limit[(g, t)]
    want = f"TEP26-3.{t}"
    return Draft("hotel_limit_direct", "policy", 1,
                 f"Under the current travel policy, what is the maximum nightly hotel rate in INR for a "
                 f"Grade {g} ({GRADES[g]}) employee staying in a Tier {t} city?",
                 "number", gold,
                 lambda o: grade_value(o.policy(f"hotel nightly room rate limit tier {t} cities", want), g),
                 gold_clauses=[want], distractor=w.old_params.hotel_limit[(g, t)])


_SCALARS = {
    "submission_days": ("Under the current policy, within how many days of the expense date must an expense claim be submitted?",
                        "claim submission deadline days", "TEP26-2.1", r"within (\d+) days"),
    "receipt_threshold": ("Under the current policy, above what claim amount in INR is an itemised receipt mandatory?",
                          "receipt mandatory amount", "TEP26-2.1", r"exceeds INR ([\d,]+)"),
    "approval_threshold": ("Up to what amount in INR may the Claims Desk approve a compliant claim without senior approval, under the current policy?",
                           "approval authority senior approval amount", "TEP26-7.1", r"up to INR ([\d,]+)"),
    "entertainment_cap": ("What is the current per-event cap in INR for client entertainment claims?",
                          "client entertainment cap per event", "TEP26-4.2", r"capped at INR ([\d,]+)"),
    "local_transport_cap": ("What is the current per-claim cap in INR for local transport claims?",
                            "local transport taxi cap per claim", "TEP26-6.1", r"capped at INR ([\d,]+)"),
    "business_min_hours": ("Under the current policy, what is the minimum flight duration in hours for a Manager or Director to fly business class?",
                           "air travel business class flight hours", "TEP26-5.1", r"flights of ([\d.]+) hours"),
}


def t_policy_scalar(w: World, rng: random.Random) -> Draft | None:
    key = rng.choice(sorted(_SCALARS))
    q, query, want, pat = _SCALARS[key]
    gold = getattr(w.params, key)
    return Draft("policy_scalar", "policy", 1, q, "number", gold,
                 lambda o: _money(o.policy(query, want), pat),
                 gold_clauses=[want], distractor=getattr(w.old_params, key), meta={"field": key})


def t_meal_allowance(w: World, rng: random.Random) -> Draft | None:
    g = rng.choice(GRADE_ORDER)
    return Draft("meal_allowance", "policy", 1,
                 f"What is the current daily meal allowance in INR for a Grade {g} ({GRADES[g]}) employee?",
                 "number", w.params.meal_per_diem[g],
                 lambda o: grade_value(o.policy("meals daily allowance per diem", "TEP26-4.1"), g),
                 gold_clauses=["TEP26-4.1"], distractor=w.old_params.meal_per_diem[g])


def _pick_city(w: World, rng: random.Random) -> str:
    moved = [c for c in w.params.city_tier if w.params.city_tier[c] != w.old_params.city_tier[c]]
    return rng.choice(moved) if moved and rng.random() < 0.5 else rng.choice(sorted(w.params.city_tier))


def t_hotel_limit_city(w: World, rng: random.Random) -> Draft | None:
    g = rng.choice(GRADE_ORDER)
    city = _pick_city(w, rng)
    t = w.params.city_tier[city]
    gold = w.params.hotel_limit[(g, t)]

    def plan(o: Oracle) -> Any:
        tier = tier_from_annex(o, city)
        return grade_value(o.policy(f"hotel nightly room rate limit tier {tier} cities", f"TEP26-3.{tier}"), g)

    return Draft("hotel_limit_city", "policy", 2,
                 f"Under the current policy, what is the maximum nightly hotel rate in INR for a Grade {g} "
                 f"employee staying in {city}?", "number", gold, plan,
                 gold_clauses=[f"TEP26-A.{t}", f"TEP26-3.{t}"],
                 distractor=w.old_params.hotel_limit[(g, w.old_params.city_tier[city])],
                 meta={"city": city, "tier_changed": w.params.city_tier[city] != w.old_params.city_tier[city]})


def t_hotel_limit_employee(w: World, rng: random.Random) -> Draft | None:
    e = rng.choice(w.employees)
    home = w.office(e["office_id"])["city"]
    city = _pick_city(w, rng)
    if city == home:
        return None
    t = w.params.city_tier[city]
    gold = w.params.hotel_limit[(e["grade"], t)]
    sql = (f"SELECT {{employees.grade}} FROM {{employees}} WHERE {{employees.name}} = '{e['name']}'")

    def plan(o: Oracle) -> Any:
        grade = o.scalar(sql)
        tier = tier_from_annex(o, city)
        return grade_value(o.policy(f"hotel nightly room rate limit tier {tier} cities", f"TEP26-3.{tier}"), grade)

    return Draft("hotel_limit_employee", "policy", 3,
                 f"{e['name']} is travelling to {city} on business. Under the current policy, what is the "
                 f"maximum nightly hotel rate in INR they may claim?", "number", gold, plan,
                 gold_clauses=[f"TEP26-A.{t}", f"TEP26-3.{t}"],
                 distractor=w.old_params.hotel_limit[(e["grade"], w.old_params.city_tier[city])],
                 meta={"city": city})


def t_hotel_excess(w: World, rng: random.Random) -> Draft | None:
    emp = _emp(w)
    pool = [c for c in w.claims if c["category"] == "hotel" and c["currency"] == "INR"]
    if not pool:
        return None
    over = [c for c in pool if w.labels[c["claim_id"]].facts.get("nightly_rate", 0)
            > w.labels[c["claim_id"]].facts.get("limit", 1e18)]
    c = rng.choice(over) if over and rng.random() < 0.5 else rng.choice(pool)
    g = emp[c["emp_id"]]["grade"]
    t = w.params.city_tier.get(c["city"], 3)
    limit = w.params.hotel_limit[(g, t)]
    gold = round(max(0.0, c["amount"] / c["nights"] - limit), 2)
    sql = (f"SELECT c.{{claims.city}}, c.{{claims.nights}}, c.{{claims.amount}}, e.{{employees.grade}} "
           f"FROM {{claims}} c JOIN {{employees}} e ON c.{{claims.emp_id}} = e.{{employees.emp_id}} "
           f"WHERE c.{{claims.claim_id}} = '{c['claim_id']}'")

    def plan(o: Oracle) -> Any:
        city, nights, amount, grade = o.sql(sql)[0]
        tier = tier_from_annex(o, city)
        lim = grade_value(o.policy(f"hotel nightly room rate limit tier {tier} cities", f"TEP26-3.{tier}"), grade)
        return o.calc(f"max(0, {amount} / {nights} - {lim:g})")

    old_lim = w.old_params.hotel_limit[(g, w.old_params.city_tier.get(c["city"], 3))]
    return Draft("hotel_excess", "policy", 4,
                 f"By how much, in INR per night, does the nightly rate of hotel claim {c['claim_id']} exceed "
                 f"the current-policy limit that applies to it? Answer 0 if it is within the limit.",
                 "number", gold, plan, gold_clauses=[f"TEP26-A.{t}", f"TEP26-3.{t}"],
                 distractor=round(max(0.0, c["amount"] / c["nights"] - old_lim), 2),
                 meta={"claim_id": c["claim_id"]})


def t_meal_excess(w: World, rng: random.Random) -> Draft | None:
    emp = _emp(w)
    pool = [c for c in w.claims if c["category"] == "meals" and c["currency"] == "INR"]
    if not pool:
        return None
    over = [c for c in pool if c["amount"] > w.params.meal_per_diem[emp[c["emp_id"]]["grade"]]]
    c = rng.choice(over) if over and rng.random() < 0.5 else rng.choice(pool)
    g = emp[c["emp_id"]]["grade"]
    gold = round(max(0.0, c["amount"] - w.params.meal_per_diem[g]), 2)
    sql = (f"SELECT c.{{claims.amount}}, e.{{employees.grade}} FROM {{claims}} c JOIN {{employees}} e "
           f"ON c.{{claims.emp_id}} = e.{{employees.emp_id}} WHERE c.{{claims.claim_id}} = '{c['claim_id']}'")

    def plan(o: Oracle) -> Any:
        amount, grade = o.sql(sql)[0]
        lim = grade_value(o.policy("meals daily allowance per diem", "TEP26-4.1"), grade)
        return o.calc(f"max(0, {amount} - {lim:g})")

    return Draft("meal_excess", "policy", 3,
                 f"By how much in INR does meal claim {c['claim_id']} exceed the claimant's current daily meal "
                 f"allowance? Answer 0 if it is within the allowance.", "number", gold, plan,
                 gold_clauses=["TEP26-4.1"],
                 distractor=round(max(0.0, c["amount"] - w.old_params.meal_per_diem[g]), 2),
                 meta={"claim_id": c["claim_id"]})


# ---------------------------------------------------------------------------
# WORKFLOW template
# ---------------------------------------------------------------------------
_WF_DIFFICULTY = {"duplicate": 1, "late_submission": 2, "missing_receipt": 2, "class_not_entitled": 3,
                  "over_limit": 4, "needs_senior_approval": 4, "compliant": 4}


def t_process_claim(w: World, rng: random.Random, claim_id: str | None = None) -> Draft | None:
    cid = claim_id or rng.choice(w.pending_ids)
    label = w.labels[cid]
    gold = {"decision": label.decision, "reason_code": label.reason_code}
    if label.reason_code == "compliant":
        acceptable = [label.deciding_clause, "TEP26-7.1"]
    else:
        acceptable = [label.deciding_clause]
    sql = ("SELECT c.{claims.claim_id}, e.{employees.grade}, c.{claims.category}, c.{claims.expense_date}, "
           "c.{claims.submitted_date}, CAST(julianday(c.{claims.submitted_date}) - julianday(c.{claims.expense_date}) "
           "AS INTEGER) AS submission_lag_days, c.{claims.city}, c.{claims.nights}, c.{claims.flight_hours}, "
           "c.{claims.cabin_class}, c.{claims.amount}, c.{claims.currency}, c.{claims.receipt_attached}, "
           "c.{claims.status} FROM {claims} c JOIN {employees} e ON c.{claims.emp_id} = e.{employees.emp_id} "
           f"WHERE c.{{claims.claim_id}} = '{cid}'")

    def plan(o: Oracle) -> Any:
        row = o.sql(sql)[0]
        (_, grade, cat, exp_date, _sub, lag, city, nights, fhours, cabin, amount, currency, receipt, _st) = row

        def record(decision: str, reason: str, clauses: list[str]) -> str:
            o.call("record_decision", claim_id=cid, decision=decision, reason_code=reason,
                   cited_clauses=", ".join(clauses))
            return f"{decision} ({reason})"

        if o.call("check_duplicates", claim_id=cid)["earlier_duplicates"]:
            return record("reject", "duplicate", ["TEP26-8.1"])
        t21 = o.policy("claim submission deadline days receipt", "TEP26-2.1")
        if lag > _money(t21, r"within (\d+) days"):
            return record("reject", "late_submission", ["TEP26-2.1"])
        if currency != "INR":
            rate = o.call("get_fx_rate", currency=currency, date_iso=exp_date)["inr_per_unit"]
            amount_inr = o.calc(f"{amount} * {rate}")
        else:
            amount_inr = amount
        if not receipt and amount_inr > _money(t21, r"exceeds INR ([\d,]+)"):
            return record("reject", "missing_receipt", ["TEP26-2.1"])

        limit_clause = None
        if cat == "airfare":
            if cabin == "business":
                limit_clause = "TEP26-5.1"
                t51 = o.policy("air travel business class flight hours", limit_clause)
                if not (grade in ("G3", "G4") and fhours >= _money(t51, r"flights of ([\d.]+) hours")):
                    return record("reject", "class_not_entitled", [limit_clause])
        elif cat == "hotel":
            if currency == "USD":
                limit_clause = "TEP26-3.4"
                lim = grade_value(o.policy("hotel international travel outside India USD", limit_clause), grade)
            else:
                tier = tier_from_annex(o, city)
                limit_clause = f"TEP26-3.{tier}"
                lim = grade_value(o.policy(f"hotel nightly room rate limit tier {tier} cities", limit_clause), grade)
            nightly = o.calc(f"{amount} / {nights}") if nights != 1 else amount
            if nightly > lim:
                return record("reject", "over_limit", [limit_clause])
        else:
            q, limit_clause, pat = {
                "meals": ("meals daily allowance per diem", "TEP26-4.1", None),
                "client_entertainment": ("client entertainment cap per event", "TEP26-4.2", r"capped at INR ([\d,]+)"),
                "local_transport": ("local transport taxi cap per claim", "TEP26-6.1", r"capped at INR ([\d,]+)"),
            }[cat]
            text = o.policy(q, limit_clause)
            lim = grade_value(text, grade) if pat is None else _money(text, pat)
            if amount_inr > lim:
                return record("reject", "over_limit", [limit_clause])

        t71 = o.policy("approval authority escalate senior approval", "TEP26-7.1")
        if amount_inr > _money(t71, r"up to INR ([\d,]+)"):
            return record("escalate", "needs_senior_approval", ["TEP26-7.1"])
        return record("approve", "compliant", ([limit_clause] if limit_clause else []) + ["TEP26-7.1"])

    return Draft("process_claim", "workflow", _WF_DIFFICULTY[label.reason_code],
                 f"Process pending claim {cid}: check it against the current travel & expense policy, record "
                 f"your decision (approve, reject or escalate) with the reason code and the clauses you relied "
                 f"on, then report the outcome.", "decision", gold, plan,
                 gold_clauses=list(label.required_clauses), acceptable_citations=acceptable,
                 meta={"claim_id": cid, "reason_code": label.reason_code, "category": w.claim(cid)["category"],
                       "currency": w.claim(cid)["currency"]})


TEMPLATES: dict[str, list[Callable[[World, random.Random], Draft | None]]] = {
    "analytics": [t_pending_count, t_employee_month_count, t_category_quarter_total, t_dept_month_total,
                  t_top_office, t_budget_remaining],
    "policy": [t_hotel_limit_direct, t_policy_scalar, t_meal_allowance, t_hotel_limit_city,
               t_hotel_limit_employee, t_hotel_excess, t_meal_excess],
    "workflow": [t_process_claim],
}


# ---------------------------------------------------------------------------
# generation driver
# ---------------------------------------------------------------------------
def realise(w: World, draft: Draft, index: int) -> Task:
    """Run the oracle in a fresh session; emit a Task only if it reproduces the gold."""
    with bound(w.seed, w.variant) as sess:
        o = Oracle(w)
        value = draft.plan(o)
        if draft.answer_type == "decision":
            rec = sess.decisions.get(draft.meta["claim_id"])
            if rec is None or rec["decision"] != draft.gold["decision"] or rec["reason_code"] != draft.gold["reason_code"]:
                raise OracleError(f"oracle decision {rec} != gold {draft.gold}")
            cited = set(rec["cited_clauses"])
            if not cited & set(draft.acceptable_citations):
                raise OracleError("oracle citations not acceptable")
        oracle_text = format_gold(draft.answer_type, value) if draft.answer_type != "decision" else str(value)
        if not answer_matches(draft.answer_type, draft.gold, oracle_text):
            raise OracleError(f"oracle answer {oracle_text!r} != gold {draft.gold!r}")
    gold_text = format_gold(draft.answer_type, draft.gold)
    final = {"role": "assistant", "content": f"ANSWER: {gold_text}"}
    names = [c["name"] for c in o.calls]
    question = draft.question
    return Task(
        task_id=f"{w.variant}{w.seed:06d}-{draft.template}-{index}",
        workspace=w.workspace, seed=w.seed, variant=w.variant, task_type=draft.task_type,
        template=draft.template, difficulty=draft.difficulty, question=question,
        prompt=user_prompt(w.workspace, question), answer_type=draft.answer_type, gold=draft.gold,
        gold_answer_text=gold_text, gold_calls=o.calls, gold_messages=o.messages + [final],
        gold_clauses=draft.gold_clauses, acceptable_citations=draft.acceptable_citations,
        distractor_answer=draft.distractor,
        requires_retrieval=any(n in ("search_policy", "get_clause") for n in names),
        requires_sql="run_sql" in names, step_budget=len(o.calls) + 3, meta=draft.meta)


def generate_for_world(seed: int, variant: str, counts: dict[str, int], rng: random.Random,
                       start_index: int = 0) -> list[Task]:
    """Generate ``counts[type]`` verified tasks of each family from one world."""
    w = World.get(seed, variant)
    out: list[Task] = []
    seen_q: set[str] = set()
    idx = start_index
    for ttype, n in counts.items():
        made, attempts = 0, 0
        pending_pool = list(w.pending_ids)
        rng.shuffle(pending_pool)
        while made < n and attempts < n * 30:
            attempts += 1
            if ttype == "workflow":
                if not pending_pool:
                    break
                draft = t_process_claim(w, rng, pending_pool.pop())
            else:
                draft = rng.choice(TEMPLATES[ttype])(w, rng)
            if draft is None or draft.question in seen_q:
                continue
            task = realise(w, draft, idx)  # OracleError propagates: it is a generator bug
            seen_q.add(draft.question)
            out.append(task)
            made += 1
            idx += 1
    return out
