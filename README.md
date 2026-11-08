# Claims & Spend Desk: reliable tool-calling agents on Tiny Aya

Lexsi Labs Hackathon, Track 3 (Agentic Fine-tuning). The agent is trained with **AgentTune** and evaluated with **AuditKIT**. Student model: **Tiny Aya Global (3.35B)**.

A finance-operations agent for a (synthetic) Indian company. It does three kinds of work in one environment:

| Family | Example | Skills |
|---|---|---|
| `analytics` | "Total approved hotel spend in Q2 2026 (INR claims)?" | SQL over a business DB |
| `policy` | "Max nightly hotel rate for Ramesh travelling to Pune?" | multi-hop retrieval with stale/wrong-entity distractors, DB + policy + calculator |
| `workflow` | "Process pending claim C-41187" | rules, approvals, write-once audit-trail decision |

## Why the environment is verifiable (the accuracy guarantee)

* **Procedural worlds.** Every seed generates a new company: people, claims, budgets, policy limits, approval thresholds and city-tier tables. Answers can't be memorised; the agent has to use its tools.
* **Each answer is computed twice.** Gold answers come from Python over the world data. A scripted oracle agent that sees **only tool outputs** has to reproduce them, or the task is dropped. 5,130 tasks were generated with 0 mismatches.
* **Rewards are computed by replay.** Rewards and metrics re-execute the model's tool calls against a fresh session. They never trust the model's narration of tool results.
* **Disjoint splits.** Train, val, test-ID and test-OOD use disjoint world seeds. Test-OOD also renames every table and column.

## Problem-statement coverage

| # | Problem statement | Where |
|---|---|---|
| 1 | Reliable tool calling (RL) | GRPO on multi-turn rollouts (`create_agentic_trainer("grpo")`) |
| 2 | Efficient trajectories | duplicate/invalid/over-budget penalties; calls/turns metrics |
| 3 | Agentic distillation | verified teacher trajectories, then `create_distill_trainer` |
| 4 | RAG fine-tuning | `search_policy` over a corpus with superseded and UK-entity distractors; when-to-retrieve metrics |
| 5 | Synthetic multi-hop data | `claimsdesk.tasks`: difficulty-graded (1-4 hop), evidence-annotated, oracle-verified |
| 6 | Data/database agents | `analytics` family, execution-verified |
| 7 | Workflow/decision agents | `workflow` family: rules, approvals, write-once audit trail |
| 8 | Self-healing | replay-based failure classifier, then corrective data, `SelfHealLoop`, `DeploymentGate` |
| 9 | Reward design and hacking defences | `claimsdesk.rewards` (naive vs hardened) + adversarial tests |

## Layout

```
claimsdesk/      stdlib-only environment core
  world.py         procedural world generator (SQLite DB + policy corpus + FX)
  policy.py        policy parameters, clause rendering, BM25 search
  rules.py         ground-truth decision engine
  tools.py         8 sandboxed tools (read-only SQL, AST calculator, write-once decisions)
  session.py       per-episode binding via workspace id (AgentTune pre_step_hook)
  tasks.py         task families + oracle verification
  episode.py       rollout normalisation + replay analysis
  rewards.py       naive and hardened rewards
  metrics.py       metrics, bootstrap CIs, paired tests
scripts/build_datasets.py
tests/           sandbox, oracle and reward-hacking tests
data/            frozen splits + manifest (sha256)
```

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[dev]"
pytest -q
python scripts/build_datasets.py --out data
```
