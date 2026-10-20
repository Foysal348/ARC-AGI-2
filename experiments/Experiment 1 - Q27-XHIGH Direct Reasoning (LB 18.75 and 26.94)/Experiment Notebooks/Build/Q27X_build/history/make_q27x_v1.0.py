"""Build Q27-XHIGH-v1: Qwen3.8-27B-FP8 with full xhigh thinking on Kaggle 4xL4, REPL-style verification.

Reused from the Kaggle-proven PIV/V3.1 notebook: model + wheelhouse paths, offline vLLM 0.19.0 install with version
parity, engine settings (TP4, language_model_only, CUDA graphs on, custom all-reduce off, prefix cache off), Qwen chat
rendering with reasoning_effort, the documented forced-close stop text, fresh-process train verification + test
execution of solution(grid).
New: continuous scheduling on vLLM's step API (a finished task frees its slot at once; no waves), xhigh thinking up to
THINK_BUDGET then forced close, MTP speculative decoding, deadline admission/drain, dual mode."""
import hashlib
import json
import sys
from pathlib import Path

DST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\ARC AGI 2\Research Base Experiment\Q27-XHIGH-v1.ipynb")

C_HEAD = r'''# Q27-XHIGH v1 — Qwen3.8-27B-FP8, full xhigh thinking, verified `solution()` + direct answer (4×L4)

Dual mode:
* **Commit / interactive run** (`EVAL12`): 12 public evaluation tasks (fixed seed), scored, with throughput measured.
* **Competition rerun** (`KAGGLE_IS_COMPETITION_RERUN`): every hidden test task → `submission.json`.

A valid fallback `submission.json` (test input as both attempts) is written in cell 0, before anything can fail. It
is rewritten after every finished task.

Per task, one generation:
1. Qwen thinks at **xhigh** effort. If thinking has not ended after `THINK_BUDGET` tokens, Qwen's documented stop
   sentence + `</think>` is appended (token-exact) and the answer is generated in a second request.
2. The answer contains a Python `solution(grid)` and a JSON block with the predicted test outputs.
3. `solution()` runs in a fresh process on every training input (exact match required), then on the test inputs.
4. attempt_1 = output of a train-verified program, else the model's direct grid; attempt_2 = the next distinct
   candidate (direct grid, unverified program output, test input).

Scheduling: vLLM step API with continuous admission. A task that finishes frees its slot at once; there are no
waves. MTP speculative decoding is on (falls back to plain decoding if the engine refuses it). Deadline: no new task
after the admission deadline; at the drain deadline in-flight thinking is force-closed; at the hard stop everything
is aborted and the submission written.

Reused from the Kaggle-proven PIV / V3.1 notebook: offline vLLM 0.19.0 install with version parity, the engine
settings, chat rendering, the stop text, fresh-process verification.
'''

C0 = r'''# ==================================================================== CELL 0 — mode + fallback submission (stdlib only)
import os, sys, json, time, random, hashlib, re, shutil, subprocess, threading, traceback, zipfile, math
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures as cf

NOTEBOOK_T0 = time.perf_counter()
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
ARC_ROOT = Path(os.environ.get("Q27X_ARC_ROOT", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"))
if not ARC_ROOT.exists():
    for _alt in ("/kaggle/input/arc-prize-2026-arc-agi-2",):
        if Path(_alt).exists():
            ARC_ROOT = Path(_alt)
WORKING = Path(os.environ.get("Q27X_WORKING", "/kaggle/working")); WORKING.mkdir(parents=True, exist_ok=True)
OUT = WORKING / "q27x"; OUT.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = WORKING / "submission.json"

def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

test_challenges = _load_json(ARC_ROOT / "arc-agi_test_challenges.json")

def write_json_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def identity_submission(challenges):
    return {tid: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in task["test"]]
            for tid, task in challenges.items()}

write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))
print("MODE:", "COMPETITION RERUN (hidden test)" if RERUN else "COMMIT / INTERACTIVE (evaluation)",
      "| fallback submission.json:", len(test_challenges), "tasks | RUN_ID", RUN_ID)
'''

C1 = r'''# ==================================================================== CELL 1 — configuration
PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", "EVAL12").strip().upper()
PROFILE_TABLE = {
    # commit run: 12 public evaluation tasks, scored; also measures throughput for the 12-hour projection
    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},
    # hidden test: every task; Kaggle's limit is 12 h including install + model load
    "RERUN_ALL": {"source": "test", "n_tasks": None, "time_budget_sec": 12.0 * 3600, "admit_frac": 0.875, "drain_frac": 0.925},
}
if PROFILE not in PROFILE_TABLE:
    raise ValueError(f"unknown profile {PROFILE}; choose {list(PROFILE_TABLE)}")
P = PROFILE_TABLE[PROFILE]
SEED = 20261005
TASK_SAMPLE_SEED = 20261005

MODEL_PATH = Path(os.environ.get("Q27X_MODEL", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"))
WHEELHOUSE = Path(os.environ.get("Q27X_WHEELHOUSE", "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"))

# ---- generation budget (Kaggle discussion 745787: Qwen3.8 needs xhigh; median ~64k tokens; forced commit ~49k)
REASONING_EFFORT = "xhigh"
THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 45056))      # tokens of thinking before the forced close
ANSWER_MAX_TOKENS = int(os.environ.get("Q27X_ANSWER_MAX", 8192))   # answer after </think> (code + JSON grids)
DRAIN_ANSWER_MAX_TOKENS = 4096                                      # answer budget for tasks force-closed at the drain
STOP_THINKING_TEXT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
MAX_MODEL_LEN = 65536
PROMPT_TOKEN_LIMIT = MAX_MODEL_LEN - THINK_BUDGET - ANSWER_MAX_TOKENS - 64     # longer prompts get the fallback
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}                 # Qwen thinking-mode recommendation

# ---- engine (PIV/V3.1 C1 settings; context, concurrency and MTP changed)
MAX_NUM_SEQS = int(os.environ.get("Q27X_MAX_NUM_SEQS", 16))   # KV ~650k tokens/GPU (16 full-attn layers, 1 KV head/GPU)
MAX_NUM_BATCHED_TOKENS = 32768
GPU_MEMORY_UTILIZATION = 0.90
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": int(os.environ.get("Q27X_SPEC_TOKENS", 2))}
if os.environ.get("Q27X_NO_SPEC") == "1":
    SPECULATIVE_CONFIG = None
EXPECTED_LIVE_TOKENS_PER_SEQ = 40000      # used to cap concurrency by the KV cache actually allocated

# ---- verification (fresh process per candidate: train first, then test)
VERIFY_TIMEOUT_SEC = 20.0
VERIFY_WORKERS = 4

# ---- time
TOTAL_BUDGET_SEC = float(os.environ.get("Q27X_TIME_BUDGET", P["time_budget_sec"]))
ADMIT_DEADLINE = NOTEBOOK_T0 + TOTAL_BUDGET_SEC * P["admit_frac"]     # no new task after this
DRAIN_DEADLINE = NOTEBOOK_T0 + TOTAL_BUDGET_SEC * P["drain_frac"]     # in-flight thinking is force-closed
HARD_STOP = NOTEBOOK_T0 + TOTAL_BUDGET_SEC - 15 * 60                  # abort everything, write outputs
CONFIG = {k: globals()[k] for k in ["PROFILE", "RERUN", "SEED", "REASONING_EFFORT", "THINK_BUDGET", "ANSWER_MAX_TOKENS",
          "DRAIN_ANSWER_MAX_TOKENS", "MAX_MODEL_LEN", "PROMPT_TOKEN_LIMIT", "SAMPLING", "MAX_NUM_SEQS",
          "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION", "SPECULATIVE_CONFIG", "VERIFY_TIMEOUT_SEC", "TOTAL_BUDGET_SEC"]}
CONFIG["profile_row"] = P
print(json.dumps(CONFIG, indent=1, default=str))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
'''

C2 = r'''# ==================================================================== CELL 2 — offline install (PIV/V3.1 parity block) + tokenizer
import importlib, importlib.metadata
def pkg_version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None

EXPECTED_RUNTIME_PACKAGES = {"vllm": "0.19.0", "transformers": "4.57.6", "tokenizers": "0.22.2", "huggingface-hub": "0.36.2"}
_loaded = {"vllm": "vllm", "transformers": "transformers", "tokenizers": "tokenizers", "huggingface-hub": "huggingface_hub"}
before_stack = {n: pkg_version(n) for n in EXPECTED_RUNTIME_PACKAGES}

def offline_pip_install(requirements):
    if not requirements: return
    if not WHEELHOUSE.exists():
        raise FileNotFoundError(f"wheelhouse not attached: {WHEELHOUSE}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-cache-dir",
                           "--find-links", str(WHEELHOUSE), *requirements])

if os.environ.get("Q27X_SKIP_INSTALL") != "1":     # Q27X_SKIP_INSTALL is for the local harness only
    for _pkg, _mod in _loaded.items():
        if before_stack.get(_pkg) != EXPECTED_RUNTIME_PACKAGES[_pkg] and _mod in sys.modules:
            raise RuntimeError(f"{_pkg} already imported at {before_stack.get(_pkg)!r}; restart the session before Run All.")
    if before_stack["vllm"] != EXPECTED_RUNTIME_PACKAGES["vllm"]:
        offline_pip_install(["vllm==0.19.0"])
    else:
        offline_pip_install([f"{p}=={EXPECTED_RUNTIME_PACKAGES[p]}" for p in ("transformers", "tokenizers", "huggingface-hub")
                             if before_stack[p] != EXPECTED_RUNTIME_PACKAGES[p]])
    after_stack = {n: pkg_version(n) for n in EXPECTED_RUNTIME_PACKAGES}
    for _p, _want in EXPECTED_RUNTIME_PACKAGES.items():
        if after_stack.get(_p) != _want:
            raise RuntimeError(f"runtime parity failure: {_p}={after_stack.get(_p)!r}, expected {_want!r}")
    print("runtime parity PASS:", after_stack)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=False)
_model_cfg = _load_json(MODEL_PATH / "config.json")
assert _model_cfg.get("model_type") == "qwen3_5", _model_cfg.get("model_type")
assert (_model_cfg.get("quantization_config") or {}).get("quant_method") == "fp8"
THINK_CLOSE_IDS = tokenizer.encode("</think>", add_special_tokens=False)
assert len(THINK_CLOSE_IDS) == 1, THINK_CLOSE_IDS
THINK_CLOSE_ID = THINK_CLOSE_IDS[0]
STOP_IDS = tokenizer.encode(STOP_THINKING_TEXT, add_special_tokens=False)
assert THINK_CLOSE_ID in STOP_IDS
print("tokenizer OK | </think> id", THINK_CLOSE_ID, "| stop text", len(STOP_IDS), "tokens")
'''

C3 = r'''# ==================================================================== CELL 3 — tasks + prompts
if P["source"] == "eval":
    _ch = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
    _sol = _load_json(ARC_ROOT / "arc-agi_evaluation_solutions.json")
    TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))
    CHALLENGES = {t: _ch[t] for t in TASK_IDS}
    SOLUTIONS = {t: _sol[t] for t in TASK_IDS}
else:
    CHALLENGES = dict(test_challenges); TASK_IDS = sorted(CHALLENGES); SOLUTIONS = None

def grid_text(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)

def shape(g):
    return f"{len(g)}x{len(g[0]) if g else 0}"

INSTRUCTIONS = """Find the single transformation rule that maps every training input to its output. Check the rule against \
every training example before you commit to it, then apply it to the test input(s).

When you have finished thinking, reply with exactly two parts:
1. A Python code block defining `def solution(grid: list[list[int]]) -> list[list[int]]` that implements the rule for any \
input grid. Use only the standard library and numpy; no printing, files or network.
2. A JSON code block of the form {"test_outputs": [<output grid for test input 1>, ...]} with one output grid per test \
input, in order, each grid a list of rows of integers 0-9."""

def build_user_message(task):
    parts = ["You are solving an ARC-AGI puzzle. A grid is a rectangle of digits 0-9 (each digit is a colour); "
             "it is shown one row per line, top to bottom.", ""]
    for i, ex in enumerate(task["train"], 1):
        parts += [f"Training example {i}", f"Input ({shape(ex['input'])}):", grid_text(ex["input"]),
                  f"Output ({shape(ex['output'])}):", grid_text(ex["output"]), ""]
    for i, ex in enumerate(task["test"], 1):
        parts += [f"Test input {i} ({shape(ex['input'])}):", grid_text(ex["input"]), ""]
    parts.append(INSTRUCTIONS)
    return "\n".join(parts)

def render_prompt(task):
    msgs = [{"role": "user", "content": build_user_message(task)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True, reasoning_effort=REASONING_EFFORT)

PROMPTS, PROMPT_IDS, SKIPPED = {}, {}, {}
for tid in TASK_IDS:
    text = render_prompt(CHALLENGES[tid])
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > PROMPT_TOKEN_LIMIT:
        SKIPPED[tid] = len(ids); continue
    PROMPTS[tid] = text; PROMPT_IDS[tid] = ids
_pl = sorted(len(v) for v in PROMPT_IDS.values())
print(f"{PROFILE}: {len(TASK_IDS)} tasks, {sum(len(CHALLENGES[t]['test']) for t in TASK_IDS)} test queries | "
      f"prompt tokens median {_pl[len(_pl)//2] if _pl else 0}, max {_pl[-1] if _pl else 0} | skipped (too long): {len(SKIPPED)}")
assert "xhigh" not in build_user_message(CHALLENGES[TASK_IDS[0]])
'''

C4 = r'''# ==================================================================== CELL 4 — answer parsing, fresh-process verification, attempts
def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))

_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_JSON_RE = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.S)

def parse_answer(text):
    """answer text after </think> -> (solution code or None, list of direct grids or None)"""
    code = None
    for block in _CODE_RE.findall(text or ""):
        if "def solution" in block:
            code = block                      # last block that defines solution()
    grids = None
    candidates = _JSON_RE.findall(text or "") + [m.group(0) for m in re.finditer(r"\{\s*\"test_outputs\".*\}", text or "", re.S)]
    for block in reversed(candidates):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("test_outputs"), list):
            grids = obj["test_outputs"]; break
    return code, grids

RUNNER = r"""
import sys, json
def _main():
    payload = json.loads(sys.stdin.read())
    ns = {"__name__": "candidate"}
    out = {"outputs": [], "error": None}
    try:
        exec(compile(payload["code"], "<solution>", "exec"), ns)
        fn = ns["solution"]
    except BaseException as e:
        out["error"] = "define: " + repr(e)[:500]; print(json.dumps(out)); return
    for g in payload["inputs"]:
        try:
            r = fn([list(row) for row in g])
            try:
                import numpy as _np
                if isinstance(r, _np.ndarray): r = r.tolist()
            except Exception:
                pass
            out["outputs"].append([[int(v) for v in row] for row in r])
        except BaseException as e:
            out["outputs"].append(None); out["error"] = out["error"] or ("run: " + repr(e)[:500])
    print(json.dumps(out))
_main()
"""

def _limit_resources():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    except Exception:
        pass

def run_solution(code, inputs, timeout=VERIFY_TIMEOUT_SEC):
    """fresh Python process; returns (list of outputs or None per input, error or None)"""
    try:
        r = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps({"code": code, "inputs": inputs}),
                           capture_output=True, text=True, timeout=timeout,
                           preexec_fn=_limit_resources if os.name == "posix" else None)
        line = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else ""
        res = json.loads(line) if line else {"outputs": [None] * len(inputs), "error": "no output: " + (r.stderr or "")[-300:]}
        outs = [o if valid_grid(o) else None for o in res.get("outputs", [])] + [None] * len(inputs)
        return outs[:len(inputs)], res.get("error")
    except subprocess.TimeoutExpired:
        return [None] * len(inputs), "timeout"
    except Exception as e:
        return [None] * len(inputs), f"runner: {e!r}"[:300]

def verify_task(tid, code, direct):
    task = CHALLENGES[tid]
    info = {"has_code": code is not None, "train_pass": False, "train_correct": 0, "n_train": len(task["train"]),
            "code_error": None, "code_test": None, "direct_valid": None}
    if code is not None:
        outs, err = run_solution(code, [ex["input"] for ex in task["train"]])
        info["train_correct"] = sum(o == ex["output"] for o, ex in zip(outs, task["train"]))
        info["train_pass"] = info["train_correct"] == len(task["train"])
        info["code_error"] = err
        info["code_test"], err2 = run_solution(code, [q["input"] for q in task["test"]])
        info["code_error"] = info["code_error"] or err2
    n = len(task["test"])
    d = list(direct) if isinstance(direct, list) else []
    d = (d + [None] * n)[:n]
    info["direct_valid"] = [valid_grid(g) for g in d]
    attempts = []
    for i, q in enumerate(task["test"]):
        cands = []
        ct = (info["code_test"] or [None] * n)[i]
        if info["train_pass"] and ct is not None: cands.append(("verified_code", ct))
        if valid_grid(d[i]): cands.append(("direct", d[i]))
        if not info["train_pass"] and ct is not None: cands.append(("unverified_code", ct))
        cands.append(("test_input", q["input"]))
        uniq = []
        for src, g in cands:
            if all(g != u[1] for u in uniq): uniq.append((src, g))
        while len(uniq) < 2: uniq.append(uniq[-1])
        attempts.append({"attempt_1": uniq[0][1], "attempt_2": uniq[1][1], "src_1": uniq[0][0], "src_2": uniq[1][0]})
    info["attempts"] = attempts
    return info

def score_task(tid, attempts):
    if SOLUTIONS is None: return None
    sol = SOLUTIONS[tid]
    hits = [int(a["attempt_1"] == s or a["attempt_2"] == s) for a, s in zip(attempts, sol)]
    return {"query_hits": hits, "score": sum(hits) / len(sol), "pass1": [int(a["attempt_1"] == s) for a, s in zip(attempts, sol)]}

# ---- self-tests (CPU, every run)
_t = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": [[3, 4, 5]], "output": [[5, 4, 3]]}],
      "test": [{"input": [[6, 7]]}]}
_good = "def solution(grid):\n    return [list(reversed(r)) for r in grid]\n"
_ans = "Rule: reverse rows.\n```python\n" + _good + "```\n```json\n{\"test_outputs\": [[[7, 6]]]}\n```"
_c, _g = parse_answer(_ans); assert _c and "reversed" in _c and _g == [[[7, 6]]], (_c, _g)
CHALLENGES.setdefault("__selftest__", _t)
_v = verify_task("__selftest__", _c, _g)
assert _v["train_pass"] and _v["attempts"][0]["attempt_1"] == [[7, 6]] and _v["attempts"][0]["src_1"] == "verified_code"
_v2 = verify_task("__selftest__", "def solution(grid):\n    return grid\n", [[[9, 9]]])
assert not _v2["train_pass"] and _v2["attempts"][0]["attempt_1"] == [[9, 9]] and _v2["attempts"][0]["src_2"] == "unverified_code"
_v3 = verify_task("__selftest__", "def solution(grid):\n    while True: pass\n", None)
assert _v3["code_error"] == "timeout" or _v3["code_error"], _v3["code_error"]
assert _v3["attempts"][0]["attempt_1"] == [[6, 7]]          # nothing usable -> test input
_v4 = verify_task("__selftest__", None, [[[7, 6]], [[1]]])
assert _v4["attempts"][0]["attempt_1"] == [[7, 6]]
del CHALLENGES["__selftest__"]
assert parse_answer("no blocks at all") == (None, None)
print("cell 4 self-tests PASS")
'''

C5 = r'''# ==================================================================== CELL 5 — engine (PIV/V3.1 C1 settings + 64k context + MTP)
ENGINE_CONFIG = {
    "model": str(MODEL_PATH), "tensor_parallel_size": 4, "distributed_executor_backend": "mp", "dtype": "auto",
    "load_format": "safetensors", "trust_remote_code": False, "language_model_only": True,
    "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS, "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION, "enforce_eager": False, "disable_custom_all_reduce": True,
    "enable_prefix_caching": False, "seed": SEED,
}
ENGINE_INFO = {"speculative": None, "spec_error": None, "load_sec": None, "kv_tokens": None, "concurrency": None}
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
import vllm
_t0 = time.perf_counter()
llm = None
if SPECULATIVE_CONFIG is not None:
    try:
        llm = LLM(**ENGINE_CONFIG, speculative_config=dict(SPECULATIVE_CONFIG))
        ENGINE_INFO["speculative"] = dict(SPECULATIVE_CONFIG)
    except Exception as e:
        ENGINE_INFO["spec_error"] = f"{type(e).__name__}: {e}"[:2000]
        print("MTP speculative decoding refused, retrying without it:", ENGINE_INFO["spec_error"][:500])
        llm = None
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
if llm is None:
    llm = LLM(**ENGINE_CONFIG)
ENGINE_INFO["load_sec"] = round(time.perf_counter() - _t0, 1)
engine = llm.llm_engine
try:   # the engine core runs in its own process; the front-end copy of the cache config may not carry the block count
    _cc = engine.vllm_config.cache_config
    ENGINE_INFO["kv_tokens"] = int(_cc.num_gpu_blocks) * int(_cc.block_size) if _cc.num_gpu_blocks else None
except Exception as e:
    ENGINE_INFO["kv_tokens"] = None; ENGINE_INFO["kv_error"] = repr(e)[:300]
CONCURRENCY = MAX_NUM_SEQS
if ENGINE_INFO["kv_tokens"]:
    CONCURRENCY = max(4, min(MAX_NUM_SEQS, ENGINE_INFO["kv_tokens"] // EXPECTED_LIVE_TOKENS_PER_SEQ))
ENGINE_INFO["concurrency"] = int(CONCURRENCY); ENGINE_INFO["vllm"] = vllm.__version__
print("engine ready:", json.dumps(ENGINE_INFO))

def make_params(max_tokens, seed):
    return SamplingParams(max_tokens=int(max_tokens), seed=int(seed), detokenize=False, skip_special_tokens=False,
                          output_kind=RequestOutputKind.DELTA, **SAMPLING)
'''

C6 = r'''# ==================================================================== CELL 6 — continuous xhigh scheduler (vLLM step API)
RECORDS_PATH = OUT / "records.jsonl"
STATE = {}          # tid -> dict
FINAL = {}          # tid -> attempts (list per test query)
RECORDS = []
LOG_EVERY_SEC = 120.0
GEN_TOKENS_TOTAL = 0
SCHED = {"started": time.perf_counter(), "admitted": 0, "phase2_continue": 0, "phase2_forced": 0, "drained": 0,
         "aborted_at_hard_stop": 0, "not_admitted": 0, "max_active": 0, "steps": 0}

def write_submission():
    sub = identity_submission(test_challenges)
    if not RERUN:
        write_json_atomic(SUBMISSION_PATH, sub); return
    for tid, att in FINAL.items():
        if tid in sub and len(att) == len(sub[tid]):
            sub[tid] = [{"attempt_1": a["attempt_1"], "attempt_2": a["attempt_2"]} for a in att]
    write_json_atomic(SUBMISSION_PATH, sub)

def _req(tid, phase):
    return f"{tid}|{phase}"

def admit(tid):
    st = {"tid": tid, "phase": 1, "prompt_ids": PROMPT_IDS[tid], "p1_ids": [], "p2_ids": [], "p2_prefix": [],
          "p1_finish": None, "p2_finish": None, "t_admit": time.perf_counter(), "t_done": None, "mode": None}
    STATE[tid] = st
    engine.add_request(_req(tid, 1), {"prompt_token_ids": list(st["prompt_ids"])}, make_params(THINK_BUDGET, SEED + len(STATE)))
    SCHED["admitted"] += 1

def start_phase2(st, answer_cap):
    """think closed but answer cut -> continue; think never closed -> append stop text (token-exact) and answer"""
    closed = THINK_CLOSE_ID in st["p1_ids"]
    prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + ([] if closed else list(STOP_IDS))
    room = MAX_MODEL_LEN - len(prefix) - 8
    cap = min(int(answer_cap), room)
    if cap < 64:
        return False
    st["phase"] = 2; st["p2_prefix"] = [] if closed else list(STOP_IDS); st["mode"] = "continue" if closed else "forced_close"
    SCHED["phase2_continue" if closed else "phase2_forced"] += 1
    engine.add_request(_req(st["tid"], 2), {"prompt_token_ids": prefix}, make_params(cap, SEED + 7919 + len(STATE)))
    return True

def answer_text(st):
    ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
    if THINK_CLOSE_ID not in ids:
        return "", 0, False
    k = len(ids) - 1 - ids[::-1].index(THINK_CLOSE_ID)
    return tokenizer.decode(ids[k + 1:], skip_special_tokens=True), k, True

VERIFY_POOL = cf.ThreadPoolExecutor(max_workers=VERIFY_WORKERS)
PENDING = {}        # future -> tid

def finish(st, reason):
    st["phase"] = "done"; st["t_done"] = time.perf_counter(); st["end_reason"] = reason
    text, think_tokens, closed = answer_text(st)
    code, grids = parse_answer(text)
    st.update({"answer_chars": len(text), "think_tokens": int(think_tokens if closed else len(st["p1_ids"])),
               "think_closed": closed, "has_code": code is not None, "has_grids": grids is not None})
    PENDING[VERIFY_POOL.submit(verify_task, st["tid"], code, grids)] = st["tid"]

def collect_verifications(block=False):
    done = [f for f in list(PENDING) if f.done()] if not block else list(cf.as_completed(list(PENDING)))
    for fut in done:
        tid = PENDING.pop(fut)
        st = STATE[tid]
        try:
            info = fut.result()
        except Exception as e:
            info = {"error": repr(e)[:300], "attempts": [{"attempt_1": q["input"], "attempt_2": q["input"], "src_1": "test_input",
                    "src_2": "test_input"} for q in CHALLENGES[tid]["test"]], "train_pass": False}
        FINAL[tid] = info["attempts"]
        sc = score_task(tid, info["attempts"])
        rec = {"task_id": tid, "mode": st["mode"] or "natural", "end_reason": st["end_reason"],
               "prompt_tokens": len(st["prompt_ids"]), "p1_tokens": len(st["p1_ids"]), "p2_tokens": len(st["p2_ids"]),
               "think_tokens": st["think_tokens"], "think_closed": st["think_closed"], "p1_finish": st["p1_finish"],
               "p2_finish": st["p2_finish"], "sec": round(st["t_done"] - st["t_admit"], 1), "has_code": st["has_code"],
               "has_grids": st["has_grids"], "train_pass": info.get("train_pass"), "train_correct": info.get("train_correct"),
               "n_train": info.get("n_train"), "code_error": info.get("code_error"),
               "src": [(a["src_1"], a["src_2"]) for a in info["attempts"]], "score": sc}
        RECORDS.append(rec)
        with open(RECORDS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{(time.perf_counter() - NOTEBOOK_T0) / 60:6.1f} min] {tid} {rec['mode']:12s} think {rec['think_tokens']:6d} "
              f"closed={rec['think_closed']} code={rec['has_code']} train={rec['train_correct']}/{rec['n_train']} "
              f"src={rec['src'][0][0]}" + (f" score={sc['score']:.2f}" if sc else ""), flush=True)
        write_submission()

def run_scheduler():
    global GEN_TOKENS_TOTAL
    queue = [t for t in TASK_IDS if t in PROMPT_IDS]
    last_log = time.perf_counter(); drained = False
    while True:
        now = time.perf_counter()
        active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        if now >= HARD_STOP:
            ids = [_req(s["tid"], s["phase"]) for s in active]
            if ids: engine.abort_request(ids)
            for s in active:
                SCHED["aborted_at_hard_stop"] += 1; finish(s, "hard_stop")
            break
        if now >= DRAIN_DEADLINE and not drained:
            drained = True
            p1 = [s for s in active if s["phase"] == 1]
            if p1: engine.abort_request([_req(s["tid"], 1) for s in p1])
            for s in p1:
                SCHED["drained"] += 1; s["p1_finish"] = "drained"
                if not start_phase2(s, DRAIN_ANSWER_MAX_TOKENS): finish(s, "drain_no_room")
            active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        while queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        SCHED["max_active"] = max(SCHED["max_active"], len(active))
        if not active:
            collect_verifications()
            if not queue or now >= ADMIT_DEADLINE or drained:
                break
            continue
        outs = engine.step(); SCHED["steps"] += 1
        for o in outs:
            tid, phase = o.request_id.rsplit("|", 1); phase = int(phase); st = STATE[tid]
            if st["phase"] != phase: continue
            comp = o.outputs[0] if o.outputs else None
            if comp is not None:
                new = list(comp.token_ids); GEN_TOKENS_TOTAL += len(new)
                (st["p1_ids"] if phase == 1 else st["p2_ids"]).extend(new)
            if o.finished:
                fr = comp.finish_reason if comp is not None else None
                if phase == 1:
                    st["p1_finish"] = fr
                    closed = THINK_CLOSE_ID in st["p1_ids"]
                    if closed and fr != "length":
                        finish(st, "natural")
                    elif not start_phase2(st, ANSWER_MAX_TOKENS):
                        finish(st, "no_room_for_answer")
                else:
                    st["p2_finish"] = fr; finish(st, st["mode"])
        collect_verifications()
        if time.perf_counter() - last_log > LOG_EVERY_SEC:
            last_log = time.perf_counter(); el = last_log - SCHED["started"]
            print(f"  .. {el / 60:6.1f} min | active {len(active)} | queued {len(queue)} | done {len(FINAL)} | "
                  f"gen {GEN_TOKENS_TOTAL} tok | {GEN_TOKENS_TOTAL / max(el, 1e-6):.0f} tok/s", flush=True)
    SCHED["not_admitted"] = len(queue)
    collect_verifications(block=True)
    SCHED["wall_sec"] = round(time.perf_counter() - SCHED["started"], 1)
    SCHED["gen_tokens"] = GEN_TOKENS_TOTAL
    SCHED["tok_per_sec"] = round(GEN_TOKENS_TOTAL / max(SCHED["wall_sec"], 1e-6), 1)

run_scheduler()
write_submission()
print("scheduler done:", json.dumps(SCHED))
'''

C7 = r'''# ==================================================================== CELL 7 — report, projection, bundle
n_done = len(FINAL)
rows = RECORDS
def _mean(xs): return round(sum(xs) / len(xs), 1) if xs else None
report = {"run_id": RUN_ID, "profile": PROFILE, "rerun": RERUN, "config": CONFIG, "engine": ENGINE_INFO, "scheduler": SCHED,
          "tasks_planned": len(TASK_IDS), "tasks_finished": n_done, "skipped_long_prompt": SKIPPED,
          "modes": {m: sum(r["mode"] == m for r in rows) for m in ("natural", "continue", "forced_close")},
          "think_closed_naturally": sum(r["mode"] == "natural" for r in rows),
          "mean_think_tokens": _mean([r["think_tokens"] for r in rows]),
          "mean_total_gen_tokens": _mean([r["p1_tokens"] + r["p2_tokens"] for r in rows]),
          "with_code": sum(bool(r["has_code"]) for r in rows), "with_grids": sum(bool(r["has_grids"]) for r in rows),
          "train_verified": sum(bool(r["train_pass"]) for r in rows),
          "first_attempt_source": {s: sum(r["src"][0][0] == s for r in rows) for s in ("verified_code", "direct", "unverified_code", "test_input")}}
if SOLUTIONS is not None:
    sc = [r["score"]["score"] for r in rows if r["score"]]
    report["eval_score_sum"] = round(sum(sc), 3)
    report["eval_score_pct_of_planned"] = round(100 * sum(sc) / max(len(TASK_IDS), 1), 2)
    report["eval_pass1_queries"] = sum(sum(r["score"]["pass1"]) for r in rows if r["score"])
# projection for the hidden test (240 tasks) from this run's measured aggregate rate and mean tokens/task
if SCHED.get("tok_per_sec") and report["mean_total_gen_tokens"]:
    _rate = SCHED["tok_per_sec"]
    report["projection_240_tasks_hours_at_this_rate"] = round(240 * report["mean_total_gen_tokens"] / _rate / 3600, 2)
    report["projection_note"] = ("aggregate rate measured with up to %d concurrent tasks; the hidden-test run keeps "
                                 "%d slots busy, so its rate is normally higher" % (SCHED["max_active"], CONCURRENCY))
write_json_atomic(OUT / "run_report.json", report)
print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=1, default=str))
ZIP = WORKING / "Q27X_BUNDLE.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file(): z.write(p, p.relative_to(WORKING).as_posix())
    if SUBMISSION_PATH.exists(): z.write(SUBMISSION_PATH, "submission.json")
print("bundle:", ZIP, round(ZIP.stat().st_size / 1e6, 2), "MB")
if SOLUTIONS is not None:
    print(f"EVAL SCORE: {report['eval_score_sum']} / {len(TASK_IDS)} tasks ({report['eval_score_pct_of_planned']}%) | "
          f"train-verified programs {report['train_verified']} | throughput {SCHED['tok_per_sec']} tok/s | "
          f"projected 240-task hours {report.get('projection_240_tasks_hours_at_this_rate')}")
try:
    VERIFY_POOL.shutdown(wait=False)
except Exception:
    pass
'''

cells = [{"cell_type": "markdown", "metadata": {}, "source": C_HEAD.splitlines(keepends=True)}]
for src in (C0, C1, C2, C3, C4, C5, C6, C7):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print("wrote", DST, hashlib.sha256(DST.read_bytes()).hexdigest())
