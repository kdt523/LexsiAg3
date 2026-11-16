"""Claims & Spend Desk agent - Hugging Face Space.

Recorded, replay-verified episodes of the base vs fine-tuned Tiny Aya Global agent on the same
held-out tasks, a live tool sandbox on the same environment, and the headline metrics.
"""
from __future__ import annotations

import json

import gradio as gr
from huggingface_hub import hf_hub_download

from claimsdesk import session as S
from claimsdesk.tools import TOOL_MAP

LOGS = "kd1609/claimsdesk-logs"
RUNS = {"Base (full prompt)": "tiny_aya_global_prompted", "Fine-tuned (SFT LoRA)": "tiny_aya_global_sft"}
SPLITS = {"test_id": ["tiny_aya_global_prompted", "tiny_aya_global_sft"], "test_ood": ["tiny_aya_global_minimal", "tiny_aya_global_sft"]}


def _jsonl(path):
    f = hf_hub_download(LOGS, path, repo_type="dataset")
    return [json.loads(l) for l in open(f, encoding="utf-8")]


DATA = {}
for split, labels in SPLITS.items():
    for lab in labels:
        try:
            trajs = _jsonl(f"evals/{lab}/{split}/trajectories.jsonl")
            sigs = _jsonl(f"evals/{lab}/{split}/signals.jsonl")
            DATA[(lab, split)] = {t["metadata"]["task_id"]: (t, s) for t, s in zip(trajs, sigs)}
        except Exception as e:  # noqa: BLE001
            print("missing", lab, split, e)
TASKS = {sp: sorted(set.intersection(*[set(DATA[(l, sp)]) for l in labs if (l, sp) in DATA]))
         for sp, labs in SPLITS.items() if all((l, sp) in DATA for l in labs)}


def render(traj, sig) -> str:
    conv = traj["metadata"]["conversation"]
    out = []
    for m in conv[1:]:
        role = m["role"]
        c = m.get("content") or ""
        if role == "user" and out:
            c = c if len(c) < 1500 else c[:1500] + " ..."
            out.append(f"**tool result**\n```\n{c}\n```")
        elif role == "user":
            out.append(f"**task**\n\n{c}")
        elif role == "assistant":
            out.append(f"**agent**\n```\n{c}\n```")
    flags = (f"correct: **{sig['correct']}** | grounded: **{sig['grounded']}** | calls: {sig['n_calls']} | "
             f"invalid: {sig['invalid_calls']} | duplicate: {sig['duplicate_calls']} | unsafe approval: {sig['unsafe_approval']}")
    return flags + "\n\n---\n\n" + "\n\n".join(out)


def labels_for(split):
    return SPLITS[split]


def show(split, task_id):
    a, b = labels_for(split)
    ta, sa = DATA[(a, split)][task_id]
    tb, sb = DATA[(b, split)][task_id]
    return render(ta, sa), render(tb, sb)


def task_choices(split, only):
    ids = TASKS.get(split, [])
    a, b = labels_for(split)
    if only == "SFT fixed it":
        ids = [i for i in ids if DATA[(b, split)][i][1]["correct_and_grounded"] and not DATA[(a, split)][i][1]["correct_and_grounded"]]
    elif only == "SFT still wrong":
        ids = [i for i in ids if not DATA[(b, split)][i][1]["correct_and_grounded"]]
    return gr.update(choices=ids, value=ids[0] if ids else None)


def sandbox(workspace, tool, args):
    try:
        kw = json.loads(args or "{}")
        sess = S.bind_text(workspace)
        if sess is None:
            return "Workspace must look like WS-S050000 (S = train schema, O = renamed OOD schema)."
        return TOOL_MAP[tool](**kw)
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


RESULTS = """
| Metric (held-out test_id, n=450) | Tiny Aya base | Fine-tuned |
|---|---|---|
| Task success | 6.0% | **81.3%** |
| Grounded success (answer backed by tool evidence) | 3.1% | **80.7%** |
| No final answer | 41% | **0%** |
| Invalid tool calls | 31% | **0%** |
| Redundant tool calls | 44% | **0%** |
| Tool calls per task | 9.4 | **3.5** |
| Unsafe approvals (approving a non-compliant claim) | 31% | **2.7%** |
| OOD success (all tables/columns renamed, n=180) | 3.3% | **41.7%** |

**Self-healing round** (expert iteration + oracle corrections) reached 80% vs 78% on validation, but unsafe
approvals rose from 3% to 19%, so the **deployment gate rejected it**.
**Reward hacking:** a naive outcome reward accepted 168 hacky samples (19.2% of what it accepted:
right decision without evidence, distractor citations, hedged answers); the hardened reward rejects them.

Models, data and logs: [models](https://huggingface.co/collections/kd1609/claims-desk-agent-models-6afb03b6772379c4f6b4ddca) |
[datasets](https://huggingface.co/collections/kd1609/claims-desk-agent-datasets-6afb03b7fd099e07d44d1c9c) |
[code](https://github.com/kdt523/LexsiAg3)
"""

with gr.Blocks(title="Claims & Spend Desk Agent") as demo:
    gr.Markdown("# Claims & Spend Desk Agent\nTiny Aya Global fine-tuned with AgentTune for reliable tool use: "
                "SQL analytics, policy retrieval and expense-claim decisions. Episodes are recorded and replay-verified.")
    with gr.Tab("Compare episodes"):
        with gr.Row():
            split = gr.Dropdown(list(TASKS), value="test_id", label="Split")
            only = gr.Radio(["All", "SFT fixed it", "SFT still wrong"], value="SFT fixed it", label="Filter")
            tid = gr.Dropdown([], label="Task")
        with gr.Row():
            left = gr.Markdown()
            right = gr.Markdown()
        split.change(task_choices, [split, only], tid)
        only.change(task_choices, [split, only], tid)
        tid.change(show, [split, tid], [left, right])
        demo.load(task_choices, [split, only], tid)
    with gr.Tab("Tool sandbox"):
        gr.Markdown("Call the agent's tools directly on any generated world. SQL is read-only and sandboxed.")
        ws = gr.Textbox("WS-S050000", label="Workspace")
        tool = gr.Dropdown(list(TOOL_MAP), value="get_schema", label="Tool")
        args = gr.Textbox("{}", label="Arguments (JSON)", lines=2)
        res = gr.Code(label="Result", language="json")
        gr.Button("Run").click(sandbox, [ws, tool, args], res)
        gr.Examples([["WS-S050000", "run_sql", '{"query": "SELECT status, COUNT(*) FROM claims GROUP BY status"}'],
                     ["WS-S050000", "search_policy", '{"query": "hotel limit tier 1 city"}'],
                     ["WS-O070000", "get_schema", "{}"]], [ws, tool, args])
    with gr.Tab("Results"):
        gr.Markdown(RESULTS)

if __name__ == "__main__":
    demo.launch()
