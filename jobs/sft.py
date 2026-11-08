"""Agentic distillation, step 2: SFT Tiny Aya on verified trajectories (AgentTune), then eval.

Data: verified teacher trajectories (Aya Expanse 32B via Cohere) + oracle expert
trajectories, both replay-verified. Each trajectory is rendered in EXACTLY the
inference format (system prompt + tools fallback, JSON tool-call lists, folded tool
results), and loss is applied to assistant tokens only.

Training goes through AgentTune's ``create_distill_trainer`` with a custom
``trainer_factory`` (the built-in factory renders tool calls through the chat
template, which Tiny Aya's template drops).

    python jobs/sft.py --teacher /kaggle/input/.../teacher_verified.jsonl --n-oracle 2000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimsdesk.prompt import SYSTEM_PROMPT, TOOL_RESULT_FORMAT, TOOLS_FALLBACK_PROMPT  # noqa: E402
from claimsdesk.tasks import Task  # noqa: E402
from claimsdesk.tools import TOOLS  # noqa: E402

OUT = Path(os.environ.get("CLAIMSDESK_OUT", "out"))


def to_conversation(prompt: str, messages: list[dict], fallback: str) -> list[dict]:
    """OpenAI-style trajectory -> plain-text conversation in the inference format."""
    conv = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{fallback}"}, {"role": "user", "content": prompt}]
    pending: list[str] = []
    for m in messages:
        if m["role"] == "user":
            continue
        if m["role"] == "assistant":
            if pending:
                conv.append({"role": "user", "content": "\n".join(pending)})
                pending = []
            if m.get("tool_calls"):
                calls = [{"tool_name": c["name"], "parameters": c["arguments"]} for c in m["tool_calls"]]
                conv.append({"role": "assistant", "content": json.dumps(calls, ensure_ascii=False)})
            else:
                conv.append({"role": "assistant", "content": m["content"].strip()})
        elif m["role"] == "tool":
            pending.append(TOOL_RESULT_FORMAT.format(name=m["name"], content=m["content"]))
    return conv


def encode(tok, conv: list[dict], max_len: int) -> dict | None:
    """Token ids for the whole conversation with labels only on assistant turns."""
    ids: list[int] = []
    labels: list[int] = []
    prev = ""
    for i in range(1, len(conv) + 1):
        if conv[i - 1]["role"] == "assistant":
            before = tok.apply_chat_template(conv[:i - 1], tokenize=False, add_generation_prompt=True)
            through = tok.apply_chat_template(conv[:i], tokenize=False)
            if not before.startswith(prev) or not through.startswith(before):
                return None  # template is not prefix-stable for this sample
            ctx = tok(before[len(prev):], add_special_tokens=False)["input_ids"]
            ans = tok(through[len(before):], add_special_tokens=False)["input_ids"]
            ids += ctx + ans
            labels += [-100] * len(ctx) + ans
            prev = through
    if not ids or len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": labels}


def load_rows(teacher_paths: list[str], n_oracle: int, seed: int) -> list[dict]:
    rows = []
    for p in teacher_paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("verified"):
                rows.append({"task_id": r["task_id"], "messages": r["messages"], "source": "teacher"})
    teacher_ids = {r["task_id"] for r in rows}
    tasks = [Task.from_dict(json.loads(l)) for l in open(ROOT / "data" / "train.jsonl", encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(tasks)
    per_type = {"analytics": int(n_oracle * 0.3), "policy": int(n_oracle * 0.35), "workflow": int(n_oracle * 0.35)}
    for t in tasks:
        if t.task_id in teacher_ids or per_type[t.task_type] <= 0:
            continue
        per_type[t.task_type] -= 1
        rows.append({"task_id": t.task_id, "messages": t.gold_messages, "source": "oracle", "prompt": t.prompt})
    prompts = {t.task_id: t.prompt for t in tasks}
    for r in rows:
        r.setdefault("prompt", prompts[r["task_id"]])
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--teacher-glob", default="/kaggle/input/**/teacher_verified.jsonl")
    ap.add_argument("--n-oracle", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--hub-repo", default="")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from agenttune.agentic import create_distill_trainer
    from agenttune.agentic.rollout_engines import rollout_factory as RF
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(a.model)
    schemas = [RF._build_tool_schema(t) for t in TOOLS]
    fallback = TOOLS_FALLBACK_PROMPT.format(tools=json.dumps(schemas, indent=2))
    teacher_paths = sorted(glob.glob(a.teacher_glob, recursive=True))
    rows = load_rows(teacher_paths, a.n_oracle, a.seed)
    data, dropped = [], 0
    for r in rows:
        enc = encode(tok, to_conversation(r["prompt"], r["messages"], fallback), a.max_len)
        if enc is None:
            dropped += 1
            continue
        data.append(enc)
    stats = {"teacher_files": teacher_paths, "rows": len(rows), "encoded": len(data), "dropped": dropped,
             "teacher_rows": sum(r["source"] == "teacher" for r in rows),
             "mean_tokens": sum(len(d["input_ids"]) for d in data) / max(1, len(data))}
    print(json.dumps(stats, indent=2), flush=True)
    (OUT / "sft").mkdir(parents=True, exist_ok=True)
    (OUT / "sft" / "data_stats.json").write_text(json.dumps(stats, indent=2))

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        return {"input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
                "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])}

    class AssistantOnlyTrainer(Trainer):
        """Apply lm_head only to supervised positions: Tiny Aya's 262k vocab makes full
        fp32 logits for a 3k-token conversation ~4 GB, which OOMs a T4."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inner = model.get_base_model()
            h = inner.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
            labels = inputs["labels"][:, 1:]
            sel = labels != -100
            hs = h[:, :-1][sel]
            logits = inner.lm_head(hs)
            scale = getattr(inner, "logit_scale", None)
            if scale is not None:
                logits = logits * scale
            loss = torch.nn.functional.cross_entropy(logits.float(), labels[sel], reduction="sum")
            denom = num_items_in_batch if num_items_in_batch is not None else sel.sum()
            loss = loss / denom
            return (loss, None) if return_outputs else loss

    out_dir = str(OUT / "sft" / "adapter")
    holder: dict = {}

    def factory(*, model, train_dataset=None, **kwargs):
        # AgentTune passes the student id; we load it with LoRA and train on our exact-format data.
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16, device_map={"": 0})
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
        m = get_peft_model(m, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
        for n_, p_ in m.named_parameters():  # fp32 trainable params for stable fp16 AMP on T4
            if p_.requires_grad:
                p_.data = p_.data.float()
        m.print_trainable_parameters()
        holder["model"] = m
        args = TrainingArguments(output_dir=str(OUT / "sft" / "ckpt"), per_device_train_batch_size=1,
                                 gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
                                 learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=10,
                                 fp16=True, max_grad_norm=1.0, logging_steps=10, save_strategy="no",
                                 report_to=[], seed=a.seed, remove_unused_columns=False)
        return AssistantOnlyTrainer(model=m, args=args, train_dataset=data, data_collator=collate)

    trainer = create_distill_trainer(a.model, demonstrations=[("warmup", "ANSWER: 0")],
                                     trainer_factory=factory, output_dir=out_dir)
    result = trainer.train()
    print("train result:", result, flush=True)
    (OUT / "sft" / "train_result.json").write_text(json.dumps(getattr(result, "metrics", {}) or {}, default=str))
    holder["model"].save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if a.hub_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.hub_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=a.hub_repo)
        print("pushed adapter to", a.hub_repo, flush=True)

    del holder["model"]
    torch.cuda.empty_cache()
    cmd = [sys.executable, str(ROOT / "jobs" / "eval.py"), "--model", a.model, "--adapter", out_dir,
           "--label", "tiny_aya_global_sft", "--prompt", "full"]
    if a.eval_limit:
        cmd += ["--limit", str(a.eval_limit)]
    subprocess.run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
