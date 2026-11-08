"""Aggregate metrics with bootstrap CIs and paired significance tests (stdlib only).

Every reported number is a mean over episodes with a 95% percentile-bootstrap CI.
System-vs-system comparisons use a *paired* bootstrap over shared task ids plus an
exact McNemar test for binary metrics.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Iterable

# metric name -> (description, extractor or None if boolean field, higher_is_better)
METRICS: dict[str, tuple[str, Callable[[dict], float | None], bool]] = {
    "task_success": ("verified task success (state/answer checked against gold)",
                     lambda s: float(s["correct"]), True),
    "grounded_success": ("success AND answer traceable to tool outputs / evidence before decision",
                         lambda s: float(s["correct_and_grounded"]), True),
    "no_answer_rate": ("no parseable answer / no recorded decision",
                       lambda s: float(not s["has_output"]), False),
    "malformed_call_rate": ("episode ended on a malformed tool call",
                            lambda s: float(s["malformed_call"]), False),
    "invalid_call_rate": ("share of tool calls that errored (bad name/args/SQL/validation)",
                          lambda s: (s["invalid_calls"] / s["n_calls"]) if s["n_calls"] else None, False),
    "redundant_call_rate": ("share of tool calls that exactly repeat an earlier call",
                            lambda s: (s["duplicate_calls"] / s["n_calls"]) if s["n_calls"] else None, False),
    "calls_per_task": ("tool calls per task", lambda s: float(s["n_calls"]), False),
    "turns_per_task": ("assistant turns per task", lambda s: float(s["n_turns"]), False),
    "over_budget_rate": ("episodes exceeding the step budget",
                         lambda s: float(s["over_budget_calls"] > 0), False),
    "unnecessary_retrieval_rate": ("retrieval on tasks that need none (analytics)",
                                   lambda s: float(s["unnecessary_retrieval"]) if s["task_type"] == "analytics" else None, False),
    "missed_retrieval_rate": ("no retrieval on tasks that need policy evidence",
                              lambda s: float(s["missed_retrieval"]) if s["task_type"] != "analytics" else None, False),
    "gold_evidence_recall": ("all gold clauses retrieved (policy/workflow)",
                             lambda s: float(s["gold_clauses_retrieved"]) if s["gold_clauses_retrieved"] is not None else None, True),
    "distractor_fooled_rate": ("answered with the superseded-policy value (policy tasks)",
                               lambda s: float(s["fooled_by_distractor"]) if s["task_type"] == "policy" else None, False),
    "citation_accuracy": ("cited only current clauses incl. the deciding one (workflow)",
                          lambda s: float(bool(s["citation_ok"])) if s["task_type"] == "workflow" else None, True),
    "unsafe_approval_rate": ("approved a claim that should be rejected/escalated (workflow)",
                             lambda s: float(s["unsafe_approval"]) if s["task_type"] == "workflow" else None, False),
    "over_escalation_rate": ("escalated a claim that should not be (workflow)",
                             lambda s: float(s["over_escalation"]) if s["task_type"] == "workflow" else None, False),
    "evidence_before_decision": ("claim lookup + rule evidence before recording (workflow)",
                                 lambda s: float(bool(s["evidence_before_decision"])) if s["task_type"] == "workflow" else None, True),
}


def _values(sigs: list[dict], fn: Callable[[dict], float | None]) -> list[float]:
    return [v for v in (fn(s) for s in sigs) if v is not None]


def bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if not vals:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def summarise(sigs: list[dict], n_boot: int = 2000) -> dict[str, dict[str, float]]:
    out = {}
    for name, (_, fn, _) in METRICS.items():
        vals = _values(sigs, fn)
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals, n_boot)
        out[name] = {"mean": sum(vals) / len(vals), "lo": lo, "hi": hi, "n": len(vals)}
    return out


def breakdown(sigs: list[dict], key: str, metric: str = "task_success") -> dict[Any, dict[str, float]]:
    groups: dict[Any, list[dict]] = defaultdict(list)
    for s in sigs:
        groups[s[key]].append(s)
    fn = METRICS[metric][1]
    res = {}
    for k in sorted(groups, key=str):
        vals = _values(groups[k], fn)
        if vals:
            lo, hi = bootstrap_ci(vals, 1000)
            res[k] = {"mean": sum(vals) / len(vals), "lo": lo, "hi": hi, "n": len(vals)}
    return res


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_compare(base: list[dict], new: list[dict], metric: str = "task_success",
                   n_boot: int = 5000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of (new - base) over shared task ids, plus McNemar if binary."""
    fn = METRICS[metric][1]
    b_map = {s["task_id"]: fn(s) for s in base}
    n_map = {s["task_id"]: fn(s) for s in new}
    ids = sorted(k for k in b_map.keys() & n_map.keys() if b_map[k] is not None and n_map[k] is not None)
    if not ids:
        return {"n": 0}
    diffs = [n_map[i] - b_map[i] for i in ids]  # type: ignore[operator]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    out = {"n": n, "base": sum(b_map[i] for i in ids) / n, "new": sum(n_map[i] for i in ids) / n,  # type: ignore[misc]
           "delta": sum(diffs) / n, "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1],
           "p_boot_two_sided": min(1.0, 2 * min(sum(d <= 0 for d in boots), sum(d >= 0 for d in boots)) / n_boot)}
    if all(v in (0.0, 1.0) for v in list(b_map.values()) + list(n_map.values()) if v is not None):
        bc = sum(1 for i in ids if b_map[i] == 1 and n_map[i] == 0)
        cb = sum(1 for i in ids if b_map[i] == 0 and n_map[i] == 1)
        out["mcnemar_p"] = mcnemar_exact(bc, cb)
    return out


def format_table(systems: dict[str, list[dict]], metrics: Iterable[str] | None = None) -> str:
    """Markdown table: rows = metrics, columns = systems, cells = mean [lo, hi]."""
    metrics = list(metrics or METRICS)
    summ = {k: summarise(v, 1000) for k, v in systems.items()}
    head = "| metric | " + " | ".join(systems) + " |\n|---|" + "---|" * len(systems) + "\n"
    rows = []
    for m in metrics:
        cells = []
        for k in systems:
            s = summ[k].get(m)
            pct = not m.endswith("per_task")
            cells.append("-" if s is None else (
                f"{100 * s['mean']:.1f}% [{100 * s['lo']:.1f}, {100 * s['hi']:.1f}]" if pct
                else f"{s['mean']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"))
        arrow = "↑" if METRICS[m][2] else "↓"
        rows.append(f"| {m} {arrow} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)
