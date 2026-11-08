"""Per-episode session: binds the tools to one world and keeps the audit trail.

In production every agent session is routed to its tenant's data; here the tenant
is the *workspace id* (``WS-S004211``) written in the task prompt. AgentTune calls
``pre_step_hook(step, conversation, tools)`` before generation, and with a trainer
attached it runs trajectories sequentially, so binding at the first step of each
trajectory gives every rollout a fresh, isolated session.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .world import World, parse_workspace


@dataclass
class Session:
    world: World
    decisions: dict[str, dict] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)

    def log(self, tool: str, arguments: dict, ok: bool, error: str | None = None,
            result: Any = None) -> None:
        self.audit.append({"seq": len(self.audit) + 1, "tool": tool, "arguments": arguments,
                           "ok": ok, "error": error, "result": result})


_local = threading.local()
_global: dict[str, Session | None] = {"session": None}


def current() -> Session:
    s = getattr(_local, "session", None) or _global["session"]
    if s is None:
        raise RuntimeError("no workspace bound: the task prompt must contain a workspace id "
                           "like WS-S000123")
    return s


def bind(seed: int, variant: str = "S") -> Session:
    s = Session(World.get(seed, variant))
    _local.session = s
    _global["session"] = s
    return s


def bind_text(text: str) -> Session | None:
    ws = parse_workspace(text)
    return bind(*ws) if ws else None


@contextmanager
def bound(seed: int, variant: str = "S") -> Iterator[Session]:
    prev_local, prev_global = getattr(_local, "session", None), _global["session"]
    s = bind(seed, variant)
    try:
        yield s
    finally:
        _local.session, _global["session"] = prev_local, prev_global


def _message_text(msg: Any) -> str:
    content = msg.get("content") if isinstance(msg, dict) else msg
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def pre_step_hook(step: int, conversation: list[dict] | None = None, tools: Any = None) -> None:
    """AgentTune hook: start a fresh session at the beginning of every trajectory."""
    conversation = conversation or []
    if any(isinstance(m, dict) and m.get("role") == "assistant" for m in conversation):
        return  # mid-trajectory: keep the session (and its recorded decisions)
    for m in conversation:
        if isinstance(m, dict) and m.get("role") in ("user", "system"):
            if bind_text(_message_text(m)) is not None:
                return
