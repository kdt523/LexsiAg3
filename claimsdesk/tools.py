"""The agent's tools. Plain typed callables with Google-style docstrings, so AgentTune
(via ``transformers.utils.get_json_schema``) builds exact JSON schemas from them.

Every tool returns a compact JSON string (AgentTune ``str()``-s tool results) and
never raises: errors come back as ``{"error": ...}`` so the model can recover and the
audit trail records the failure. All tools act on the session bound by
:mod:`claimsdesk.session`.

Production guard-rails: ``run_sql`` is read-only (SQLite authorizer + ``query_only``),
time-boxed and row-capped; ``calculator`` is an AST whitelist, never ``eval``;
``record_decision`` validates enums, cross-checks decision vs reason, and is
write-once per claim.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import sqlite3
from datetime import date
from typing import Any

from .config import DECISION_FOR_REASON, DECISIONS, FX_CURRENCIES, REASON_CODES
from .rules import find_duplicates
from .session import current

MAX_ROWS = 25
SQL_STEP_BUDGET = 5_000_000


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _num(v: Any) -> Any:
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 4)
    return v


def _finish(tool: str, args: dict, payload: dict) -> str:
    err = payload.get("error")
    current().log(tool, args, ok=err is None, error=err, result=payload if err is None else None)
    return _dump(payload)


# ---------------------------------------------------------------------------
def get_schema() -> str:
    """Return the database schema: every table with its columns, types and meanings. Call this once before writing SQL."""
    w = current().world
    payload = {"tables": w.view.describe(),
               "notes": "Dates are ISO strings (YYYY-MM-DD). Claim amounts are in the claim's own "
                        "currency (INR or USD). SQLite dialect; read-only."}
    return _finish("get_schema", {}, payload)


_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                 getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


def _authorizer(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def run_sql(query: str) -> str:
    """Run one read-only SQL SELECT query against the company database and return the rows.

    Args:
        query: A single SQLite SELECT (or WITH ... SELECT) statement.
    """
    args = {"query": query}
    q = (query or "").strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", q):
        return _finish("run_sql", args, {"error": "only a single SELECT/WITH query is allowed"})
    w = current().world
    steps = {"n": 0}

    def progress() -> int:
        steps["n"] += 1
        return 1 if steps["n"] * 1000 > SQL_STEP_BUDGET else 0

    with w.lock:
        w.conn.set_authorizer(_authorizer)
        w.conn.set_progress_handler(progress, 1000)
        try:
            cur = w.conn.execute(q)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as e:
            msg = str(e)
            if "interrupted" in msg:
                msg = "query exceeded the time budget"
            return _finish("run_sql", args, {"error": f"SQL error: {msg}"})
        finally:
            w.conn.set_authorizer(None)
            w.conn.set_progress_handler(None, 0)
    truncated = len(rows) > MAX_ROWS
    rows = [[_num(v) for v in r] for r in rows[:MAX_ROWS]]
    payload: dict[str, Any] = {"columns": cols, "rows": rows, "row_count": len(rows)}
    if truncated:
        payload["truncated"] = True
        payload["note"] = f"only the first {MAX_ROWS} rows are shown; aggregate in SQL instead"
    return _finish("run_sql", args, payload)


# ---------------------------------------------------------------------------
def search_policy(query: str, top_k: int = 3) -> str:
    """Search the travel & expense policy documents and return the best-matching clauses with their full text and status.

    Args:
        query: Keywords describing the rule you need, e.g. "hotel limit tier 2 cities".
        top_k: Number of clauses to return (1-5).
    """
    args = {"query": query, "top_k": top_k}
    try:
        k = max(1, min(5, int(top_k)))
    except (TypeError, ValueError):
        k = 3
    if not str(query or "").strip():
        return _finish("search_policy", args, {"error": "query must not be empty"})
    hits = current().world.index.search(str(query), k)
    return _finish("search_policy", args, {"results": [c.as_result() for c in hits]})


def get_clause(clause_id: str) -> str:
    """Fetch one policy clause by its id, e.g. "TEP26-3.2".

    Args:
        clause_id: The clause identifier as shown in search results.
    """
    args = {"clause_id": clause_id}
    c = current().world.clause(str(clause_id or ""))
    if c is None:
        return _finish("get_clause", args, {"error": f"unknown clause id {clause_id!r}"})
    return _finish("get_clause", args, c.as_result())


# ---------------------------------------------------------------------------
def get_fx_rate(currency: str, date_iso: str) -> str:
    """Get the official exchange rate (INR per 1 unit of the currency) on a given date.

    Args:
        currency: Currency code: USD, EUR, GBP or SGD.
        date_iso: Date in YYYY-MM-DD format.
    """
    args = {"currency": currency, "date_iso": date_iso}
    cur = str(currency or "").strip().upper()
    if cur not in FX_CURRENCIES:
        return _finish("get_fx_rate", args, {"error": f"unsupported currency {currency!r}"})
    try:
        d = date.fromisoformat(str(date_iso).strip())
    except ValueError:
        return _finish("get_fx_rate", args, {"error": "date_iso must be YYYY-MM-DD"})
    rate = current().world.fx(cur, d.isoformat())
    return _finish("get_fx_rate", args, {"currency": cur, "date": d.isoformat(), "inr_per_unit": rate})


# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Supports + - * / % ** parentheses and round, min, max, abs.

    Args:
        expression: The expression, e.g. "(12500 - 9800) / 3".
    """
    args = {"expression": expression}
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(expression or ""))
    if len(expr) > 300:
        return _finish("calculator", args, {"error": "expression too long"})
    try:
        value = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return _finish("calculator", args, {"error": "division by zero"})
    except (SyntaxError, ValueError, TypeError) as e:
        return _finish("calculator", args, {"error": f"invalid expression: {e}"})
    return _finish("calculator", args, {"result": _num(float(value))})


# ---------------------------------------------------------------------------
def check_duplicates(claim_id: str) -> str:
    """List earlier-submitted claims that duplicate this claim (same employee, category, expense date and amount).

    Args:
        claim_id: The claim id, e.g. "C-10042".
    """
    args = {"claim_id": claim_id}
    w = current().world
    c = w.claim(str(claim_id or "").strip())
    if c is None:
        return _finish("check_duplicates", args, {"error": f"unknown claim id {claim_id!r}"})
    return _finish("check_duplicates", args,
                   {"claim_id": c["claim_id"], "earlier_duplicates": find_duplicates(c, w.claims)})


def record_decision(claim_id: str, decision: str, reason_code: str, cited_clauses: str) -> str:
    """Record the final decision on a pending claim in the audit trail. Each claim can be decided only once.

    Args:
        claim_id: The pending claim id, e.g. "C-10042".
        decision: One of: approve, reject, escalate.
        reason_code: One of: compliant, duplicate, late_submission, missing_receipt, over_limit, class_not_entitled, needs_senior_approval.
        cited_clauses: Comma-separated ids of the current-policy clauses that justify the decision, e.g. "TEP26-2.1".
    """
    args = {"claim_id": claim_id, "decision": decision, "reason_code": reason_code,
            "cited_clauses": cited_clauses}
    s = current()
    w = s.world
    cid = str(claim_id or "").strip()
    dec = str(decision or "").strip().lower()
    reason = str(reason_code or "").strip().lower()
    c = w.claim(cid)
    if c is None:
        return _finish("record_decision", args, {"error": f"unknown claim id {claim_id!r}"})
    if c["status"] != "pending":
        return _finish("record_decision", args, {"error": f"claim {cid} is not pending (status {c['status']})"})
    if cid in s.decisions:
        return _finish("record_decision", args, {"error": f"claim {cid} already has a recorded decision"})
    if dec not in DECISIONS:
        return _finish("record_decision", args, {"error": f"decision must be one of {DECISIONS}"})
    if reason not in REASON_CODES:
        return _finish("record_decision", args, {"error": f"reason_code must be one of {REASON_CODES}"})
    if DECISION_FOR_REASON[reason] != dec:
        return _finish("record_decision", args,
                       {"error": f"reason_code {reason} is inconsistent with decision {dec}"})
    if isinstance(cited_clauses, (list, tuple)):
        cited = [str(x).strip().upper() for x in cited_clauses if str(x).strip()]
    else:
        cited = [x.strip().upper() for x in re.split(r"[,;\s]+", str(cited_clauses or "")) if x.strip()]
    unknown = [x for x in cited if w.clause(x) is None]
    if not cited or unknown:
        return _finish("record_decision", args,
                       {"error": "cite at least one valid clause id" + (f"; unknown: {unknown}" if unknown else "")})
    rec = {"decision_id": f"D-{len(s.decisions) + 1:03d}-{cid}", "claim_id": cid, "decision": dec,
           "reason_code": reason, "cited_clauses": cited, "status": "recorded"}
    s.decisions[cid] = rec
    return _finish("record_decision", args, rec)


TOOLS = [get_schema, run_sql, search_policy, get_clause, get_fx_rate, calculator,
         check_duplicates, record_decision]
TOOL_MAP = {t.__name__: t for t in TOOLS}


def call_tool(name: str, arguments: dict | str | None) -> str:
    """Dispatch a tool call by name (used by our replay/eval harness)."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        current().log(name, {}, ok=False, error="unknown tool")
        return _dump({"error": f"unknown tool {name!r}"})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            current().log(name, {"_raw": arguments}, ok=False, error="arguments are not valid JSON")
            return _dump({"error": "arguments are not valid JSON"})
    arguments = arguments or {}
    if not isinstance(arguments, dict):
        current().log(name, {"_raw": arguments}, ok=False, error="arguments must be an object")
        return _dump({"error": "arguments must be a JSON object"})
    try:
        return fn(**arguments)
    except TypeError as e:
        current().log(name, arguments, ok=False, error=f"bad arguments: {e}")
        return _dump({"error": f"bad arguments: {e}"})
