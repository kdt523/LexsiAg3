"""Procedural world generator: one company, one database, one policy corpus per seed.

``World.get(seed, variant)`` is deterministic and cached. A new seed gives new people,
new claims, new policy limits and a new city classification, so nothing can be
memorised across episodes -- the agent must use its tools.
"""

from __future__ import annotations

import math
import random
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

from .config import (
    CATEGORIES, DEPARTMENTS, DOMESTIC_CITIES, FIRST_NAMES, FX_CURRENCIES, GRADE_ORDER,
    INTERNATIONAL_CITIES, LAST_NAMES, REASON_CODES, REGIONS, YEAR,
)
from .policy import BM25Index, Clause, PolicyParams, render_policy, render_uk_annex
from .rules import RuleResult, claim_order, evaluate_claim
from .schema import LOGICAL_SCHEMA, SchemaView

WORKSPACE_RE = re.compile(r"\bWS-([SO])(\d{6})\b")


def workspace_id(seed: int, variant: str = "S") -> str:
    return f"WS-{variant}{seed:06d}"


def parse_workspace(text: str) -> tuple[int, str] | None:
    m = WORKSPACE_RE.search(text or "")
    return (int(m.group(2)), m.group(1)) if m else None


def _d(day_of_year: int) -> str:
    return (date(YEAR, 1, 1) + timedelta(days=day_of_year - 1)).isoformat()


def _doy(iso: str) -> int:
    return date.fromisoformat(iso).timetuple().tm_yday


@dataclass
class World:
    seed: int
    variant: str
    view: SchemaView
    params: PolicyParams
    old_params: PolicyParams
    clauses: list[Clause]
    index: BM25Index
    offices: list[dict]
    employees: list[dict]
    claims: list[dict]
    budgets: list[dict]
    pending_ids: list[str]
    labels: dict[str, RuleResult]
    conn: sqlite3.Connection
    fx_phase: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ lookups
    @property
    def workspace(self) -> str:
        return workspace_id(self.seed, self.variant)

    def fx(self, currency: str, iso_date: str) -> float:
        """Deterministic INR-per-unit rate for a currency on a date."""
        currency = currency.upper()
        if currency == "INR":
            return 1.0
        base = FX_CURRENCIES[currency]
        doy = _doy(iso_date)
        drift = 0.9 * math.sin(doy / 11.0 + self.fx_phase[currency]) + 0.3 * math.sin(doy / 3.7)
        return round(base * (1 + drift / 100.0) + self.fx_phase[currency] / 10.0, 2)

    def clause(self, clause_id: str) -> Clause | None:
        cid = clause_id.strip().upper()
        for c in self.clauses:
            if c.clause_id.upper() == cid:
                return c
        return None

    def employee(self, emp_id: str) -> dict:
        return next(e for e in self.employees if e["emp_id"] == emp_id)

    def claim(self, claim_id: str) -> dict | None:
        return next((c for c in self.claims if c["claim_id"] == claim_id), None)

    def office(self, office_id: str) -> dict:
        return next(o for o in self.offices if o["office_id"] == office_id)

    # ------------------------------------------------------------------ factory
    @staticmethod
    def get(seed: int, variant: str = "S") -> "World":
        return _cached_world(int(seed), variant)


@lru_cache(maxsize=256)
def _cached_world(seed: int, variant: str) -> World:
    return _generate(seed, variant)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def _generate(seed: int, variant: str) -> World:
    rng = random.Random(f"claimsdesk-world-{seed}")
    params = PolicyParams.draw(rng, "2026.1")
    old = params.perturbed(rng, "2024.2")
    fx_phase = {c: rng.uniform(0, 6.28) for c in FX_CURRENCIES}

    clauses = (render_policy(params, "TEP26", "Travel & Expense Policy v2026.1 (effective 2026-01-01)", "current")
               + render_policy(old, "TEP24", "Travel & Expense Policy v2024.2 (superseded 2025-12-31)", "superseded")
               + render_uk_annex(rng))
    rng.shuffle(clauses)

    # offices
    office_cities = rng.sample(sorted(DOMESTIC_CITIES), rng.randint(4, 6))
    offices = [{"office_id": f"OF{i + 1:02d}", "city": c, "region": REGIONS[c]}
               for i, c in enumerate(office_cities)]

    # employees
    n_emp = rng.randint(40, 60)
    names: set[str] = set()
    employees: list[dict] = []
    start = rng.randint(1000, 8000)
    for i in range(n_emp):
        while True:
            nm = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if nm not in names:
                names.add(nm)
                break
        grade = rng.choices(GRADE_ORDER, weights=[40, 30, 20, 10])[0]
        employees.append({"emp_id": f"E{start + i}", "name": nm, "grade": grade,
                          "department": rng.choice(DEPARTMENTS),
                          "office_id": rng.choice(offices)["office_id"], "manager_id": None})
    for e in employees:
        bosses = [b for b in employees if b["department"] == e["department"]
                  and GRADE_ORDER.index(b["grade"]) > GRADE_ORDER.index(e["grade"])]
        if bosses:
            e["manager_id"] = rng.choice(bosses)["emp_id"]

    world = World(seed=seed, variant=variant, view=SchemaView(variant), params=params,
                  old_params=old, clauses=clauses, index=BM25Index(clauses), offices=offices,
                  employees=employees, claims=[], budgets=[], pending_ids=[], labels={},
                  conn=sqlite3.connect(":memory:", check_same_thread=False), fx_phase=fx_phase)

    keys: set[tuple] = set()
    raw: list[dict] = []

    # historical claims (Jan-Aug)
    for e in employees:
        for _ in range(rng.randint(5, 10)):
            c = _random_claim(rng, world, e, doy_range=(1, 243))
            if c is not None and _key(c) not in keys:
                keys.add(_key(c))
                c["claim_id"] = f"TMP{len(raw):05d}"
                raw.append(c)

    # pending queue (Aug-Sep expenses), engineered to cover every decision path
    pending: list[dict] = []
    for reason in REASON_CODES:
        made = 0
        for _attempt in range(200):
            if made >= 5:
                break
            c = _engineered_claim(rng, world, reason, raw)
            if c is None or (reason != "duplicate" and _key(c) in keys):
                continue
            c["status"] = "pending"
            c["claim_id"] = f"TMP{len(raw) + len(pending) + 50000:05d}"
            res = evaluate_claim(_with_grade(world, c), params, world.fx, raw + pending + [c])
            if res.reason_code != reason:
                continue
            keys.add(_key(c))
            pending.append(c)
            made += 1

    # ids in submission order, status for history from the rules engine
    allc = raw + pending
    allc.sort(key=claim_order)
    base_id = rng.randint(10000, 60000)
    for i, c in enumerate(allc):
        c["claim_id"] = f"C-{base_id + i}"
    world.claims = allc
    for c in allc:
        res = evaluate_claim(_with_grade(world, c), params, world.fx, allc)
        world.labels[c["claim_id"]] = res
        if c["status"] != "pending":
            c["status"] = "rejected" if res.decision == "reject" else "approved"
    world.pending_ids = [c["claim_id"] for c in allc if c["status"] == "pending"]
    world.budgets = _budgets(rng, world)
    _materialise(world)
    return world


def _key(c: dict) -> tuple:
    return (c["emp_id"], c["category"], c["expense_date"], round(c["amount"], 2))


def _with_grade(world: World, c: dict) -> dict:
    d = dict(c)
    d["_grade"] = next(e["grade"] for e in world.employees if e["emp_id"] == c["emp_id"])
    return d


_DESCR = {
    "hotel": ["Hotel stay in {city} for client meetings", "Accommodation in {city} - project visit",
              "Hotel in {city}, quarterly review"],
    "meals": ["Meals during {city} visit", "Working meals in {city}", "Team dinner on site in {city}"],
    "airfare": ["Flight to {city}", "Air ticket to {city} for customer workshop", "Return flight {city}"],
    "local_transport": ["Taxi fares in {city}", "Cab to client office, {city}", "Airport transfer {city}"],
    "client_entertainment": ["Client dinner in {city}", "Customer lunch, {city}",
                             "Partner hospitality event in {city}"],
}


def _random_claim(rng: random.Random, world: World, e: dict, doy_range: tuple[int, int],
                  category: str | None = None, intl: bool | None = None) -> dict | None:
    p = world.params
    cat = category or rng.choices(CATEGORIES, weights=[28, 26, 16, 18, 12])[0]
    home_city = world.office(e["office_id"])["city"]
    if intl is None:
        intl = cat != "local_transport" and rng.random() < 0.12
    if cat == "local_transport" and not intl:
        city = home_city
    elif intl:
        city = rng.choice(INTERNATIONAL_CITIES)
    else:
        city = rng.choice([c for c in DOMESTIC_CITIES if c != home_city])
    currency = "USD" if intl and cat != "airfare" else "INR"
    exp_doy = rng.randint(*doy_range)
    lag = rng.randint(1, min(p.submission_days - 1, 25))
    if rng.random() < 0.04:
        lag = p.submission_days + rng.randint(1, 30)
    over = rng.random() < 0.08
    c = {"emp_id": e["emp_id"], "category": cat, "expense_date": _d(exp_doy),
         "submitted_date": _d(exp_doy + lag), "city": city, "nights": None, "flight_hours": None,
         "cabin_class": None, "currency": currency,
         "receipt_attached": 0 if rng.random() < 0.05 else 1, "status": "historical",
         "description": rng.choice(_DESCR[cat]).format(city=city)}
    g = e["grade"]
    if cat == "hotel":
        nights = rng.choices([1, 2, 3, 4, 5, 6, 8], weights=[20, 25, 20, 15, 10, 6, 4])[0]
        if currency == "USD":
            lim = p.intl_hotel_usd[g]
            rate = lim * (rng.uniform(1.05, 1.4) if over else rng.uniform(0.6, 1.0))
            c["amount"] = float(round(rate) * nights)
        else:
            lim = p.hotel_limit[(g, p.city_tier.get(city, 3))]
            rate = lim * (rng.uniform(1.05, 1.4) if over else rng.uniform(0.55, 1.0))
            c["amount"] = float(round(rate / 10) * 10 * nights)
        c["nights"] = nights
    elif cat == "meals":
        lim = p.meal_per_diem[g]
        v = lim * (rng.uniform(1.05, 1.5) if over else rng.uniform(0.35, 1.0))
        c["amount"] = float(round(v / (84 if currency == "USD" else 1)) if currency == "USD" else round(v / 10) * 10)
    elif cat == "airfare":
        hours = round(rng.uniform(4.0, 16.0) if intl else rng.uniform(1.0, 3.5), 1)
        biz_p = {"G1": 0.03, "G2": 0.05, "G3": 0.35, "G4": 0.6}[g]
        cabin = "business" if rng.random() < biz_p else "economy"
        fare = rng.uniform(25000, 70000) if intl else rng.uniform(3000, 12000)
        if cabin == "business":
            fare *= rng.uniform(2.2, 3.0)
        c.update(flight_hours=hours, cabin_class=cabin, amount=float(round(fare / 10) * 10))
    elif cat == "local_transport":
        lim = p.local_transport_cap
        v = lim * (rng.uniform(1.05, 1.6) if over else rng.uniform(0.08, 0.9))
        c["amount"] = float(round(v / 84) if currency == "USD" else round(v / 10) * 10)
    else:  # client_entertainment
        lim = p.entertainment_cap
        v = lim * (rng.uniform(1.05, 1.4) if over else rng.uniform(0.25, 0.95))
        c["amount"] = float(round(v / 84) if currency == "USD" else round(v / 10) * 10)
    if c["amount"] <= 0:
        return None
    return c


def _engineered_claim(rng: random.Random, world: World, reason: str, history: list[dict]) -> dict | None:
    """Build a pending claim whose first-triggered rule is ``reason`` (verified by caller)."""
    p = world.params
    doy_range = (214, 270)  # Aug 2 - Sep 27
    e = rng.choice(world.employees)
    if reason == "duplicate":
        src = rng.choice(history)
        c = dict(src)
        c["status"] = "pending"
        c["submitted_date"] = (date.fromisoformat(src["submitted_date"])
                               + timedelta(days=rng.randint(5, 40))).isoformat()
        return c
    if reason == "class_not_entitled":
        c = _random_claim(rng, world, e, doy_range, category="airfare")
        if c is None:
            return None
        c["cabin_class"] = "business"
        if e["grade"] in ("G3", "G4"):
            c["flight_hours"] = round(rng.uniform(1.0, max(1.1, p.business_min_hours - 0.5)), 1)
        c["amount"] = float(round(c["amount"] * 2.5 / 10) * 10)
        c["receipt_attached"] = 1
        c["submitted_date"] = _d(_doy(c["expense_date"]) + rng.randint(1, min(25, p.submission_days - 1)))
        return c
    if reason == "needs_senior_approval":
        if rng.random() < 0.5:
            # long compliant hotel stay
            c = _random_claim(rng, world, e, doy_range, category="hotel", intl=False)
            if c is None:
                return None
            lim = p.hotel_limit[(e["grade"], p.city_tier.get(c["city"], 3))]
            rate = round(lim * rng.uniform(0.75, 0.98) / 10) * 10
            nights = min(30, math.ceil(p.approval_threshold / rate) + rng.randint(0, 3))
            c.update(nights=nights, amount=float(rate * nights))
        else:
            ent = [x for x in world.employees if x["grade"] in ("G3", "G4")] or [e]
            e = rng.choice(ent)
            c = _random_claim(rng, world, e, doy_range, category="airfare", intl=True)
            if c is None:
                return None
            c.update(cabin_class="business", flight_hours=round(rng.uniform(max(p.business_min_hours, 8), 16), 1),
                     amount=float(round(rng.uniform(p.approval_threshold * 1.1, p.approval_threshold * 3) / 10) * 10))
        c["receipt_attached"] = 1
        c["submitted_date"] = _d(_doy(c["expense_date"]) + rng.randint(1, min(25, p.submission_days - 1)))
        return c

    cat = rng.choice(["hotel", "meals", "local_transport", "client_entertainment", "airfare"])
    if reason == "over_limit":
        cat = rng.choice(["hotel", "hotel", "meals", "local_transport", "client_entertainment"])
    c = _random_claim(rng, world, e, doy_range, category=cat)
    if c is None:
        return None
    c["receipt_attached"] = 1
    c["submitted_date"] = _d(_doy(c["expense_date"]) + rng.randint(1, min(25, p.submission_days - 1)))
    if c["category"] == "airfare":
        c["cabin_class"] = "economy"
        c["amount"] = float(min(c["amount"], round(p.approval_threshold * 0.8 / 10) * 10))
    if reason == "late_submission":
        c["submitted_date"] = _d(_doy(c["expense_date"]) + p.submission_days + rng.randint(1, 40))
    elif reason == "missing_receipt":
        c["receipt_attached"] = 0
    elif reason == "over_limit":
        _scale_to(c, world, e, over=True, rng=rng)
    elif reason == "compliant":
        _scale_to(c, world, e, over=False, rng=rng)
    return c


def _scale_to(c: dict, world: World, e: dict, over: bool, rng: random.Random) -> None:
    p = world.params
    f = rng.uniform(1.08, 1.5) if over else rng.uniform(0.4, 0.97)
    if c["category"] == "hotel":
        if c["currency"] == "USD":
            lim = p.intl_hotel_usd[e["grade"]]
            c["amount"] = float(round(lim * f) * c["nights"])
        else:
            lim = p.hotel_limit[(e["grade"], p.city_tier.get(c["city"], 3))]
            c["amount"] = float(round(lim * f / 10) * 10 * c["nights"])
        return
    lim = {"meals": p.meal_per_diem[e["grade"]], "local_transport": p.local_transport_cap,
           "client_entertainment": p.entertainment_cap}.get(c["category"])
    if lim is None:
        return
    if c["currency"] == "USD":
        rate = world.fx("USD", c["expense_date"])
        c["amount"] = float(max(1, math.floor(lim * f / rate)) if not over else math.ceil(lim * f / rate))
    else:
        c["amount"] = float(round(lim * f / 10) * 10)


def _budgets(rng: random.Random, world: World) -> list[dict]:
    out = []
    emp_dept = {e["emp_id"]: e["department"] for e in world.employees}
    for dept in DEPARTMENTS:
        for q, (m0, m1) in {1: (1, 3), 2: (4, 6), 3: (7, 9)}.items():
            for cat in CATEGORIES:
                spent = sum(world.labels[c["claim_id"]].amount_inr for c in world.claims
                            if emp_dept[c["emp_id"]] == dept and c["category"] == cat
                            and c["status"] == "approved"
                            and m0 <= date.fromisoformat(c["expense_date"]).month <= m1)
                budget = max(5000, round(spent * rng.uniform(0.8, 1.3) / 1000) * 1000)
                out.append({"department": dept, "quarter": f"{YEAR}-Q{q}", "category": cat,
                            "budget_inr": float(budget)})
    return out


def _materialise(world: World) -> None:
    v = world.view
    cur = world.conn.cursor()
    for stmt in v.ddl():
        cur.execute(stmt)
    for table, rows in (("offices", world.offices), ("employees", world.employees),
                        ("claims", world.claims), ("budgets", world.budgets)):
        cols = [c for c, _, _ in LOGICAL_SCHEMA[table]]
        phys = ", ".join(v.c(table, c) for c in cols)
        marks = ", ".join("?" for _ in cols)
        cur.executemany(f"INSERT INTO {v.t(table)} ({phys}) VALUES ({marks})",
                        [tuple(r.get(c) for c in cols) for r in rows])
    world.conn.commit()
    cur.execute("PRAGMA query_only = ON")
