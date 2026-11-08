"""Travel & expense policy: per-world parameters, clause corpus and BM25 search.

Each world carries three policy documents:

* ``TEP26`` -- the **current** India policy (version 2026.1). The only one that applies.
* ``TEP24`` -- the **superseded** 2024 policy: same structure, different numbers and a
  slightly different city classification. Classic stale-document distractor.
* ``TEPUK`` -- the **UK entity annex**: applies to a different legal entity. Distractor.

All limits are drawn per world, so the model cannot memorise them: it has to retrieve.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import DOMESTIC_CITIES, GRADE_ORDER, GRADES


def _round_to(x: float, step: int) -> int:
    return int(step * round(x / step))


@dataclass
class PolicyParams:
    version: str
    hotel_limit: dict[tuple[str, int], int]          # (grade, tier) -> INR per night
    intl_hotel_usd: dict[str, int]                   # grade -> USD per night
    meal_per_diem: dict[str, int]                    # grade -> INR per day
    entertainment_cap: int                           # INR per event
    local_transport_cap: int                         # INR per claim
    business_min_hours: float                        # G3+ may fly business if flight >= this
    receipt_threshold: int                           # INR; receipts required above this
    submission_days: int                             # claim must be submitted within N days
    approval_threshold: int                          # INR; above -> senior approval (escalate)
    city_tier: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def draw(rng: random.Random, version: str) -> "PolicyParams":
        base = rng.choice(range(4000, 6501, 500))
        grade_mult = {"G1": 1.0, "G2": 1.2, "G3": 1.5, "G4": 2.0}
        tier_mult = {1: 1.0, 2: 0.8, 3: 0.65}
        hotel = {(g, t): max(1500, _round_to(base * grade_mult[g] * tier_mult[t], 250))
                 for g in GRADE_ORDER for t in (1, 2, 3)}
        usd_base = rng.choice(range(140, 221, 10))
        intl = {g: _round_to(usd_base * grade_mult[g], 10) for g in GRADE_ORDER}
        meal_base = rng.choice(range(800, 1501, 100))
        meals = {g: _round_to(meal_base * (1 + 0.25 * i), 50) for i, g in enumerate(GRADE_ORDER)}
        return PolicyParams(
            version=version,
            hotel_limit=hotel,
            intl_hotel_usd=intl,
            meal_per_diem=meals,
            entertainment_cap=rng.choice(range(6000, 15001, 1000)),
            local_transport_cap=rng.choice(range(1500, 4001, 250)),
            business_min_hours=float(rng.choice([4, 5, 6, 7])),
            receipt_threshold=rng.choice([500, 750, 1000, 1500, 2000]),
            submission_days=rng.choice([30, 45, 60, 90]),
            approval_threshold=rng.choice(range(40000, 75001, 5000)),
            city_tier=dict(DOMESTIC_CITIES),
        )

    def perturbed(self, rng: random.Random, version: str) -> "PolicyParams":
        """A plausible *other* version: every number differs, two cities change tier."""

        def bump(v: float, step: int) -> int:
            for _ in range(20):
                new = _round_to(v * rng.uniform(0.75, 0.92), step)
                if new != v and new > 0:
                    return new
            return int(v) - step

        tiers = dict(self.city_tier)
        movable = [c for c in tiers if tiers[c] in (2, 3)]
        for city in rng.sample(movable, 2):
            tiers[city] = 3 if tiers[city] == 2 else 2
        days_opts = [d for d in (30, 45, 60, 90) if d != self.submission_days]
        hours_opts = [h for h in (4.0, 5.0, 6.0, 7.0, 8.0) if h != self.business_min_hours]
        return PolicyParams(
            version=version,
            hotel_limit={k: bump(v, 250) for k, v in self.hotel_limit.items()},
            intl_hotel_usd={k: bump(v, 10) for k, v in self.intl_hotel_usd.items()},
            meal_per_diem={k: bump(v, 50) for k, v in self.meal_per_diem.items()},
            entertainment_cap=bump(self.entertainment_cap, 500),
            local_transport_cap=bump(self.local_transport_cap, 250),
            business_min_hours=rng.choice(hours_opts),
            receipt_threshold=bump(self.receipt_threshold, 250),
            submission_days=rng.choice(days_opts),
            approval_threshold=bump(self.approval_threshold, 5000),
            city_tier=tiers,
        )


@dataclass(frozen=True)
class Clause:
    clause_id: str
    policy: str          # human-readable policy name/version
    status: str          # "current" | "superseded" | "not applicable (UK entity)"
    title: str
    text: str

    def as_result(self) -> dict[str, str]:
        return {"clause_id": self.clause_id, "policy": self.policy, "status": self.status,
                "title": self.title, "text": self.text}


def _inr(v: float) -> str:
    return f"INR {int(v):,}"


def _grade_list(values: dict[str, int], fmt) -> str:
    return "; ".join(f"Grade {g} ({GRADES[g]}): {fmt(values[g])}" for g in GRADE_ORDER)


def render_policy(p: PolicyParams, prefix: str, policy_name: str, status: str) -> list[Clause]:
    """Render one India-format policy document into clauses."""
    C = lambda sec, title, text: Clause(f"{prefix}-{sec}", policy_name, status, title, text)  # noqa: E731
    tier_names = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
    clauses = [
        C("2.1", "Claim submission and receipts",
          f"Expense claims must be submitted within {p.submission_days} days of the expense date "
          f"(hotel: check-in date). Claims submitted later must be rejected as late submissions. "
          f"An itemised receipt is mandatory for any claim whose amount exceeds "
          f"{_inr(p.receipt_threshold)} (foreign-currency claims are converted to INR at the "
          f"exchange rate on the expense date); such claims without a receipt must be rejected."),
    ]
    for t in (1, 2, 3):
        limits = {g: p.hotel_limit[(g, t)] for g in GRADE_ORDER}
        clauses.append(C(
            f"3.{t}", f"Hotel accommodation - {tier_names[t]} cities",
            f"Maximum nightly room rate for hotel stays in {tier_names[t]} cities "
            f"(city classification in Annex A.{t}): {_grade_list(limits, _inr)}. "
            f"The nightly rate is the claimed amount divided by the number of nights. "
            f"Claims above the applicable nightly limit must be rejected as over limit."))
    clauses += [
        C("3.4", "Hotel accommodation - international travel",
          f"Maximum nightly room rate for hotel stays outside India, in USD: "
          f"{_grade_list(p.intl_hotel_usd, lambda v: f'USD {v}')}. "
          f"Claims above the applicable nightly limit must be rejected as over limit."),
        C("4.1", "Meals per diem",
          f"Meal claims are per day. Daily meal allowance: {_grade_list(p.meal_per_diem, _inr)}. "
          f"A meal claim above the employee's daily allowance must be rejected as over limit."),
        C("4.2", "Client entertainment",
          f"Client entertainment is capped at {_inr(p.entertainment_cap)} per event for all grades. "
          f"Claims above the cap must be rejected as over limit."),
        C("5.1", "Air travel class",
          f"Economy class is the standard for all employees. Grade G3 (Manager) and Grade G4 "
          f"(Director) employees may travel business class only on flights of "
          f"{p.business_min_hours:g} hours or longer. A business-class airfare claim that does not "
          f"meet this rule must be rejected as class not entitled. Airfare has no amount cap."),
        C("6.1", "Local transport",
          f"Local transport (taxi, ride-hailing, metro) is capped at {_inr(p.local_transport_cap)} "
          f"per claim. Claims above the cap must be rejected as over limit."),
        C("7.1", "Approval authority",
          f"The Claims Desk may approve compliant claims up to {_inr(p.approval_threshold)} "
          f"(INR value; foreign-currency claims converted at the exchange rate on the expense "
          f"date). Compliant claims above this amount must be escalated for senior approval."),
        C("8.1", "Duplicate claims",
          "A claim with the same employee, category, expense date and amount as an "
          "earlier-submitted claim is a duplicate and must be rejected."),
    ]
    for t in (1, 2, 3):
        cities = sorted(c for c, tier in p.city_tier.items() if tier == t)
        clauses.append(C(
            f"A.{t}", f"Annex A - {tier_names[t]} cities",
            f"{tier_names[t]} cities for travel-policy purposes: {', '.join(cities)}."
            + (" Any other Indian city not listed in Annex A is treated as Tier 3." if t == 3 else "")))
    return clauses


def render_uk_annex(rng: random.Random) -> list[Clause]:
    """UK-entity policy: same topics, GBP numbers, does NOT apply to Acme Corp India."""
    name = "UK Entity Expense Annex v2026 (Acme Corp UK Ltd only)"
    status = "not applicable (UK entity)"
    base = rng.choice(range(120, 181, 10))
    hotel = {g: base + 30 * i for i, g in enumerate(GRADE_ORDER)}
    meal = {g: 35 + 10 * i for i, g in enumerate(GRADE_ORDER)}
    return [
        Clause("TEPUK-3.1", name, status, "Hotel accommodation (UK entity)",
               "Maximum nightly room rate for employees of Acme Corp UK Ltd: "
               + _grade_list(hotel, lambda v: f"GBP {v}") + "."),
        Clause("TEPUK-4.1", name, status, "Meals per diem (UK entity)",
               "Daily meal allowance for employees of Acme Corp UK Ltd: "
               + _grade_list(meal, lambda v: f"GBP {v}") + "."),
        Clause("TEPUK-7.1", name, status, "Approval authority (UK entity)",
               f"UK claims up to GBP {rng.choice([500, 750, 1000])} may be approved by the line "
               f"manager; above that they require finance director approval."),
        Clause("TEPUK-2.1", name, status, "Claim submission (UK entity)",
               f"UK claims must be submitted within {rng.choice([28, 56])} days with receipts "
               f"for any amount over GBP {rng.choice([10, 25])}."),
    ]


# ---------------------------------------------------------------------------
# BM25 retrieval (stdlib)
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, clauses: list[Clause], k1: float = 1.4, b: float = 0.75) -> None:
        self.clauses = clauses
        self.k1, self.b = k1, b
        self.docs = [tokenize(f"{c.title} {c.title} {c.text}") for c in clauses]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 3) -> list[Clause]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((s, i))
        # Stable, deterministic tie-break by corpus order (which is shuffled per world).
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [self.clauses[i] for s, i in scores[:top_k] if s > 0]
