"""Drive Kaggle from the local machine: sync code, push jobs, wait, fetch outputs.

    python scripts/kaggle_ops.py sync-src -m "msg"        # upload claimsdesk + data + jobs as a private dataset
    python scripts/kaggle_ops.py push probe [--cpu] [--args "..."]
    python scripts/kaggle_ops.py wait probe               # poll until finished
    python scripts/kaggle_ops.py output probe             # download log + /kaggle/working/out to runs/<job>/
    python scripts/kaggle_ops.py run probe                # push + wait + output

Requires the Kaggle CLI with credentials (~/.kaggle/kaggle.json). Kernels are private.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "kaggle" / "build"
RUNS = ROOT / "runs"
USER = os.environ.get("KAGGLE_USER", "krushnad523")
SRC_DATASET = f"{USER}/claimsdesk-src"
TOKEN_DATASET = f"{USER}/hf-token"
SRC_ITEMS = ["claimsdesk", "jobs", "tests", "data", "pyproject.toml", "README.md"]


def kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("$ kaggle", " ".join(args), flush=True)
    r = subprocess.run(cmd, capture_output=capture, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        if capture:
            print(r.stdout, r.stderr)
        raise SystemExit(f"kaggle command failed: {' '.join(args)}")
    return r


def _dataset_exists() -> bool:
    r = kaggle("datasets", "status", SRC_DATASET, check=False, capture=True)
    return r.returncode == 0 and "ready" in (r.stdout + r.stderr).lower()


def sync_src(message: str) -> None:
    d = BUILD / "src_dataset"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    zpath = d / "claimsdesk_src.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for item in SRC_ITEMS:
            p = ROOT / item
            if not p.exists():
                continue
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc":
                    continue
                z.write(f, f"claimsdesk_src/{f.relative_to(ROOT).as_posix()}")
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "claimsdesk-src", "id": SRC_DATASET, "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    print(f"packed {zpath.stat().st_size / 1e6:.1f} MB")
    if _dataset_exists():
        kaggle("datasets", "version", "-p", str(d), "-m", message)
    else:
        kaggle("datasets", "create", "-p", str(d))
    for _ in range(60):
        r = kaggle("datasets", "status", SRC_DATASET, check=False, capture=True)
        out = (r.stdout + r.stderr).strip().lower()
        if "ready" in out:
            print("dataset ready")
            return
        time.sleep(10)
    print("WARNING: dataset not ready after 10 min")


def slug(job: str) -> str:
    return f"claimsdesk-{job.replace('_', '-')}"


def push(job: str, args: str = "", gpu: bool = True, timeout: int | None = None, extra_pip: str = "",
         kernel_sources: list[str] | None = None) -> None:
    d = BUILD / slug(job)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    code = (ROOT / "kaggle" / "runner.py").read_text(encoding="utf-8")
    code = code.replace('"__JOB__"', json.dumps(job)).replace('"__ARGS__"', json.dumps(args))
    code = code.replace('"__EXTRA_PIP__"', json.dumps(extra_pip))
    (d / "runner.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{USER}/{slug(job)}", "title": slug(job), "code_file": "runner.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [SRC_DATASET, TOKEN_DATASET, f"{USER}/claimsdesk-api-keys"], "kernel_sources": [f"{USER}/{slug(k)}" for k in (kernel_sources or [])],
        "competition_sources": [], "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    cmd = ["kernels", "push", "-p", str(d)]
    if gpu:
        cmd += ["--accelerator", "NvidiaTeslaT4"]
    if timeout:
        cmd += ["-t", str(timeout)]
    kaggle(*cmd)


def status(job: str) -> str:
    r = kaggle("kernels", "status", f"{USER}/{slug(job)}", check=False, capture=True)
    return (r.stdout + r.stderr).strip()


def wait(job: str, poll: int = 60, max_hours: float = 12.5) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_hours * 3600:
        s = status(job)
        low = s.lower()
        if s != last:
            print(f"[{(time.time() - t0) / 60:6.1f} min] {s}", flush=True)
            last = s
        if any(k in low for k in ("complete", "error", "cancel")):
            return s
        time.sleep(poll)
    return last


def output(job: str) -> Path:
    d = RUNS / job
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    kaggle("kernels", "output", f"{USER}/{slug(job)}", "-p", str(d), check=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync-src")
    s.add_argument("-m", "--message", default="update")
    for name in ("push", "run"):
        p = sub.add_parser(name)
        p.add_argument("job")
        p.add_argument("--args", default="")
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--timeout", type=int, default=None)
        p.add_argument("--extra-pip", default="")
        p.add_argument("--kernel-sources", nargs="*", default=[], help="job names whose outputs to mount")
    for name in ("status", "wait", "output"):
        p = sub.add_parser(name)
        p.add_argument("job")
    a = ap.parse_args()
    if a.cmd == "sync-src":
        sync_src(a.message)
    elif a.cmd == "push":
        push(a.job, a.args, not a.cpu, a.timeout, a.extra_pip, a.kernel_sources)
    elif a.cmd == "run":
        push(a.job, a.args, not a.cpu, a.timeout, a.extra_pip, a.kernel_sources)
        time.sleep(30)
        print(wait(a.job))
        print("output ->", output(a.job))
    elif a.cmd == "status":
        print(status(a.job))
    elif a.cmd == "wait":
        print(wait(a.job))
    elif a.cmd == "output":
        print("output ->", output(a.job))


if __name__ == "__main__":
    main()
