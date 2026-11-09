# Generic Kaggle runner: sets up a pinned Python 3.12 + AgentTune env and runs one job
# from the claimsdesk source dataset. scripts/kaggle_ops.py fills in JOB/ARGS on push.
JOB = "__JOB__"
ARGS = "__ARGS__"
AGENTTUNE_REF = "d4db4c12d52342227c4113a70183d0f64aa2f345"
EXTRA_PIP = "__EXTRA_PIP__"

import glob
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[runner {time.time() - T0:7.1f}s] {msg}", flush=True)


def sh(cmd, check=True, env=None):
    log(f"$ {cmd}")
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=env or os.environ.copy())
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise SystemExit(f"command failed ({rc}): {cmd}")
    return rc


# 1. locate the source dataset (Kaggle may mount it under different roots)
init = [p for p in glob.glob("/kaggle/input/**/claimsdesk/__init__.py", recursive=True)]
if not init:
    zips = glob.glob("/kaggle/input/**/claimsdesk_src*.zip", recursive=True)
    if not zips:
        raise SystemExit("claimsdesk source dataset not attached")
    sh(f"cd {WORK} && unzip -q -o {zips[0]} -d src_unzip")
    init = glob.glob(f"{WORK}/src_unzip/**/claimsdesk/__init__.py", recursive=True)
SRC_RO = os.path.dirname(os.path.dirname(init[0]))
SRC = os.path.join(WORK, "src")
shutil.copytree(SRC_RO, SRC, dirs_exist_ok=True)
log(f"source at {SRC}")

# 2. HF token from the private dataset (secrets are not available to CLI-started runs)
tok_files = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
if tok_files:
    os.environ["HF_TOKEN"] = open(tok_files[0]).read().strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    log("HF token loaded")
else:
    log("WARNING: no hf_token.txt found; gated models will fail")
for f in glob.glob("/kaggle/input/**/api_keys.env", recursive=True) + glob.glob("/kaggle/input/**/api_keys.txt", recursive=True):
    for line in open(f):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ[k] = v
    log("extra API keys loaded")

# 3. Python 3.12 venv with pinned AgentTune (+vLLM) and AuditKIT
VENV = os.path.join(WORK, "venv")
PY = os.path.join(VENV, "bin", "python")
os.environ["UV_CACHE_DIR"] = "/tmp/uvcache"
os.environ["UV_LINK_MODE"] = "copy"
sh(f"{sys.executable} -m pip install -q uv")
sh(f"uv venv {VENV} --python 3.12 --seed -q")
sh(f'uv pip install --python {PY} -q "agenttune[vllm] @ git+https://github.com/Lexsi-Labs/AgentTune.git@{AGENTTUNE_REF}" '
   f'"auditkit==1.1.6" pytest {EXTRA_PIP if EXTRA_PIP != "__EXTRA_PIP__" else ""}')
sh(f"uv pip install --python {PY} -q -e {SRC}")
# T4 (sm_75): FlashInfer JIT kernels fail to build; remove it so vLLM uses Triton attention
sh(f"uv pip uninstall --python {PY} flashinfer-python flashinfer-cubin flashinfer-jit-cache", check=False)
log("environment ready")

# 4. run the job
env = os.environ.copy()
env.update(PYTHONUNBUFFERED="1", CLAIMSDESK_OUT=OUT, TOKENIZERS_PARALLELISM="false",
           VLLM_LOGGING_LEVEL="WARNING",
           # T4 (sm_75): FlashInfer's JIT sampling kernel fails to build -> use PyTorch sampler
           VLLM_USE_FLASHINFER_SAMPLER="0",
           VLLM_ATTENTION_BACKEND="TRITON_ATTN",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
rc = sh(f"cd {SRC} && {PY} jobs/{JOB}.py {ARGS if ARGS != '__ARGS__' else ''}", check=False, env=env)
log(f"job {JOB} finished rc={rc}")
shutil.rmtree(VENV, ignore_errors=True)  # keep /kaggle/working output small
shutil.rmtree(os.path.join(WORK, "src_unzip"), ignore_errors=True)
shutil.rmtree(SRC, ignore_errors=True)
for big in ("merged_eval", "sft_merged"):  # merged weights live on HF, keep outputs small
    shutil.rmtree(os.path.join(WORK, big), ignore_errors=True)
sys.exit(rc)
