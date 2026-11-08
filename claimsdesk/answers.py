"""Final-answer parsing and comparison (strict, hack-resistant)."""

from __future__ import annotations

import re
from typing import Any

ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+")


def extract_answer(text: str | None) -> tuple[str | None, str]:
    """Return (answer string, status). status: ok | missing | ambiguous.

    Several *different* ANSWER lines is treated as ambiguous (a hedging hack).
    """
    if not text:
        return None, "missing"
    found = [m.group(1).strip().strip("*`\"' ").rstrip(".") for m in ANSWER_RE.finditer(text)]
    found = [f for f in found if f]
    if not found:
        return None, "missing"
    if len({f.lower() for f in found}) > 1:
        return None, "ambiguous"
    return found[-1], "ok"


def parse_number(s: str) -> float | None:
    nums = NUM_RE.findall(s.replace("₹", ""))
    if len(nums) != 1:  # hedges like "7500 or 8000" are not accepted
        return None
    try:
        return float(nums[0].replace(",", ""))
    except ValueError:
        return None


def number_tolerance(gold: float) -> float:
    return max(0.011, 1e-4 * abs(gold))


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_decision(s: str) -> tuple[str | None, str | None]:
    m = re.match(r"\s*(approve|reject|escalate)\w*\s*(?:\(\s*([a-z_]+)\s*\))?", s.lower())
    if not m:
        return None, None
    return m.group(1), m.group(2)


def answer_matches(answer_type: str, gold: Any, answer: str | None) -> bool:
    if answer is None:
        return False
    if answer_type == "number":
        v = parse_number(answer)
        return v is not None and abs(v - float(gold)) <= number_tolerance(float(gold))
    if answer_type == "text":
        return norm_text(answer) == norm_text(str(gold))
    if answer_type == "decision":
        d, r = parse_decision(answer)
        return d == gold["decision"] and (r is None or r == gold["reason_code"])
    raise ValueError(answer_type)


def format_gold(answer_type: str, gold: Any) -> str:
    if answer_type == "number":
        g = round(float(gold), 2)
        return str(int(g)) if g.is_integer() else f"{g:.2f}".rstrip("0").rstrip(".")
    if answer_type == "decision":
        return f"{gold['decision']} ({gold['reason_code']})"
    return str(gold)
