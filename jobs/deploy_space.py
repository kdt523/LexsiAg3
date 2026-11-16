"""Build and deploy the Gradio Space (runs on Kaggle CPU), after a smoke test of the app.

    python jobs/deploy_space.py --repo kd1609/claimsdesk-agent
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = """---
title: Claims & Spend Desk Agent
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: {ver}
app_file: app.py
pinned: false
license: apache-2.0
short_description: Tiny Aya Global fine-tuned for reliable agentic tool use
---
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="kd1609/claimsdesk-agent")
    a = ap.parse_args()
    import gradio
    build = Path("/tmp/space_build")
    shutil.rmtree(build, ignore_errors=True)
    shutil.copytree(ROOT / "space", build)
    shutil.copytree(ROOT / "claimsdesk", build / "claimsdesk", ignore=shutil.ignore_patterns("__pycache__"))
    (build / "README.md").write_text(CARD.format(ver=gradio.__version__), encoding="utf-8")
    (build / "requirements.txt").write_text(f"gradio=={gradio.__version__}\nhuggingface_hub\n")

    # smoke test: build the UI (downloads the logs) and exercise both tabs without launching
    test = ("import app, json\n"
            "ids = app.TASKS['test_id']; print('tasks', {k: len(v) for k, v in app.TASKS.items()})\n"
            "l, r = app.show('test_id', ids[0]); assert 'correct' in l and 'correct' in r\n"
            "print(app.sandbox('WS-S050000', 'run_sql', json.dumps({'query': 'SELECT COUNT(*) FROM claims'})))\n")
    subprocess.run([sys.executable, "-c", test], cwd=build, check=True)

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(build), repo_id=a.repo, repo_type="space",
                      ignore_patterns=["__pycache__/*", "*.pyc"])
    url = f"https://huggingface.co/spaces/{a.repo}"
    print("deployed:", url, flush=True)
    Path("/kaggle/working/out").mkdir(parents=True, exist_ok=True)
    Path("/kaggle/working/out/space.json").write_text(json.dumps({"space": url}))


if __name__ == "__main__":
    main()
