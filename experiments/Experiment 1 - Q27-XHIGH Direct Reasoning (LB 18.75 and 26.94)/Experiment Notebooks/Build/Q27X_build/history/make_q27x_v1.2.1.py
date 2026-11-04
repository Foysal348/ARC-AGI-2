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

DST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\ARC AGI 2\Research Base Experiment\Q27-XHIGH-v1_2_1.ipynb")

C_HEAD = r'''# Q27-XHIGH v1.2.1 — Qwen3.8-27B-FP8, full xhigh thinking, verified `solution()` + direct answer (4×L4)

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
waves. Concurrency = min(16, KV tokens // 40k) from the KV cache vLLM actually allocated. MTP speculative decoding is
on (falls back to plain decoding, after the GPUs are released, if the engine refuses it); vLLM stats logging is on so
the MTP acceptance rate and KV usage are visible. Deadline: no new task
after the admission deadline; at the drain deadline in-flight thinking is force-closed; at the hard stop everything
is aborted and the submission written.

v1.2.1 (re-checked against discussion 745787): total generation 49,152 + 16,384 = 65,536 (NguyenThanhNhan: "65k
tokens", forced commit after 49k); KV dtype string `fp8_e4m3` exactly as in his config. Already equal: xhigh; MTP;
one pass; linear attention on triton (vLLM FLA kernels); mamba SSM state float32 (vLLM reads the checkpoint's
`mamba_ssm_dtype`); FlashInfer attention (chosen by vLLM for fp8 KV on sm89).

v1.2 (GPT review, each point checked against the sources): thinking budget 49,152 (NguyenThanhNhan's forced commit
point) + answer 12,288, context 81,920; a long prompt gets a smaller thinking budget instead of being skipped; fp8 KV
cache (discussion recipe; falls back to BF16, then to no MTP, if the engine refuses — every attempt is reported);
MTP 3 tokens; temperature 1.0 (the checkpoint's generation_config); Kaggle per-query scoring; train+test verified in
one process, killed as a process group, never past the verification deadline.

v1.1 (independent audit): one fence parser + brace-counting JSON search (answers in any block order / label),
helper blocks joined to `solution()`, several program variants tried; runtime errors in the scheduler no longer end the
notebook with an exception; answers saved to `q27x/answers/`; EOS stripped before a forced close; drain answer 8k;
hard stop 25 min before the limit; deterministic seeds; Python 3.12 gate; numpy sandbox self-test.

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
THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 49152))      # v1.2: NguyenThanhNhan forced the commit after ~49k thinking tokens
ANSWER_MAX_TOKENS = int(os.environ.get("Q27X_ANSWER_MAX", 16384))  # v1.2.1: 49,152 + 16,384 = 65,536 = NguyenThanhNhan's 65k total
DRAIN_ANSWER_MAX_TOKENS = 8192                                      # v1.1 (F11): a 30x30 grid alone is ~2.7k tokens
STOP_THINKING_TEXT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
MAX_MODEL_LEN = int(os.environ.get("Q27X_MAX_MODEL_LEN", 81920))   # v1.2: native context is 262k; fp8 KV makes room
CONTEXT_SAFETY = 256
MIN_THINK_BUDGET = 4096                  # v1.2: a long prompt gets a smaller thinking budget instead of being skipped
PROMPT_TOKEN_LIMIT = MAX_MODEL_LEN - ANSWER_MAX_TOKENS - CONTEXT_SAFETY - MIN_THINK_BUDGET
def phase1_cap(prompt_len):
    return max(0, min(THINK_BUDGET, MAX_MODEL_LEN - prompt_len - ANSWER_MAX_TOKENS - CONTEXT_SAFETY))
assert phase1_cap(MAX_MODEL_LEN) == 0 and phase1_cap(100) == min(THINK_BUDGET, MAX_MODEL_LEN - 100 - ANSWER_MAX_TOKENS - CONTEXT_SAFETY)
if (MAX_MODEL_LEN, THINK_BUDGET, ANSWER_MAX_TOKENS) == (81920, 49152, 16384):
    assert phase1_cap(9000) == 49152 and phase1_cap(16400) == 48880   # longest real prompt ~8.7k keeps the full budget
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}                 # v1.2: the checkpoint's generation_config.json defaults

# ---- engine (PIV/V3.1 C1 settings; context, concurrency and MTP changed)
MAX_NUM_SEQS = int(os.environ.get("Q27X_MAX_NUM_SEQS", 16))   # KV ~650k tokens/GPU (16 full-attn layers, 1 KV head/GPU)
MAX_NUM_BATCHED_TOKENS = 32768
GPU_MEMORY_UTILIZATION = 0.90
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": int(os.environ.get("Q27X_SPEC_TOKENS", 3))}   # v1.2: vLLM Qwen3.8 recipe uses 3
KV_CACHE_DTYPE = os.environ.get("Q27X_KV_DTYPE", "fp8_e4m3")   # v1.2.1: exactly NguyenThanhNhan's --kv-cache-dtype; on sm89 vLLM then uses FlashInfer
if os.environ.get("Q27X_NO_SPEC") == "1":
    SPECULATIVE_CONFIG = None
EXPECTED_LIVE_TOKENS_PER_SEQ = 40000      # concurrency = min(MAX_NUM_SEQS, kv_tokens // this); vLLM copies num_gpu_blocks to the front end

# ---- verification (fresh process per candidate: train first, then test)
VERIFY_TIMEOUT_SEC = 20.0
VERIFY_WORKERS = 4

# ---- time
TOTAL_BUDGET_SEC = float(os.environ.get("Q27X_TIME_BUDGET", P["time_budget_sec"]))
ADMIT_DEADLINE = NOTEBOOK_T0 + TOTAL_BUDGET_SEC * P["admit_frac"]     # no new task after this
DRAIN_DEADLINE = NOTEBOOK_T0 + TOTAL_BUDGET_SEC * P["drain_frac"]     # in-flight thinking is force-closed
HARD_STOP = NOTEBOOK_T0 + TOTAL_BUDGET_SEC - 25 * 60                  # v1.1 (F12): abort everything, write outputs (25 min margin)
VERIFY_DEADLINE = NOTEBOOK_T0 + TOTAL_BUDGET_SEC - 8 * 60              # v1.2: verification never runs past this
CONFIG = {k: globals()[k] for k in ["PROFILE", "RERUN", "SEED", "REASONING_EFFORT", "THINK_BUDGET", "ANSWER_MAX_TOKENS",
          "DRAIN_ANSWER_MAX_TOKENS", "MAX_MODEL_LEN", "PROMPT_TOKEN_LIMIT", "SAMPLING", "MAX_NUM_SEQS",
          "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION", "SPECULATIVE_CONFIG", "KV_CACHE_DTYPE", "VERIFY_TIMEOUT_SEC",
          "TOTAL_BUDGET_SEC", "MIN_THINK_BUDGET"]}
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
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python {sys.version.split()[0]}: the cp312 wheelhouse needs the pinned Python 3.12 Kaggle image")
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
_gen_cfg = _load_json(MODEL_PATH / "generation_config.json") if (MODEL_PATH / "generation_config.json").exists() else {}
_eos = _gen_cfg.get("eos_token_id", [])
EOS_IDS = set(_eos if isinstance(_eos, list) else [_eos])     # v1.1 (F8): stripped before a forced close
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

PROMPTS, PROMPT_IDS, SKIPPED, P1_CAPS = {}, {}, {}, {}
TASK_INDEX = {t: i for i, t in enumerate(TASK_IDS)}          # v1.1 (F13): seeds no longer depend on timing
for tid in TASK_IDS:
    text = render_prompt(CHALLENGES[tid])
    ids = tokenizer.encode(text, add_special_tokens=False)
    cap = phase1_cap(len(ids))
    if cap < min(MIN_THINK_BUDGET, THINK_BUDGET):  # v1.2: only prompts that leave < 4k thinking tokens are skipped
        SKIPPED[tid] = {"prompt_tokens": len(ids), "reason": "insufficient_context"}; continue
    PROMPTS[tid] = text; PROMPT_IDS[tid] = ids; P1_CAPS[tid] = cap
_pl = sorted(len(v) for v in PROMPT_IDS.values())
print(f"{PROFILE}: {len(TASK_IDS)} tasks, {sum(len(CHALLENGES[t]['test']) for t in TASK_IDS)} test queries | "
      f"prompt tokens median {_pl[len(_pl)//2] if _pl else 0}, max {_pl[-1] if _pl else 0} | reduced thinking budget: "
      f"{sum(c < THINK_BUDGET for c in P1_CAPS.values())} | skipped (too long): {len(SKIPPED)}")
assert "xhigh" not in build_user_message(CHALLENGES[TASK_IDS[0]])
'''

C4 = r'''# ==================================================================== CELL 4 — answer parsing, fresh-process verification, attempts
import tempfile
def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))

# v1.1 (F1): one fence scanner. A label the old regex did not know (json, text, Python, python3, ...) no longer shifts
# the pairing of every later fence; blocks are classified by their label afterwards.
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
CODE_LABELS = {"", "python", "py", "python3"}
MAX_PROGRAM_VARIANTS = 4

def _fences(text):
    return [(lab.strip().lower(), body) for lab, body in _FENCE_RE.findall(text or "")]

def _test_outputs_objects(text):
    """every {"test_outputs": ...} object in the text, found by brace counting (no greedy regex)"""
    objs, t = [], text or ""
    for m in re.finditer(r"\{\s*\"test_outputs\"", t):
        depth = 0
        for k in range(m.start(), len(t)):
            ch = t[k]
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[m.start():k + 1])
                        if isinstance(obj, dict) and isinstance(obj.get("test_outputs"), list):
                            objs.append(obj["test_outputs"])
                    except Exception:
                        pass
                    break
    return objs

def parse_answer(text):
    """answer text after </think> -> (list of program variants to try, direct test_outputs or None)
    variants: the last block defining solution(), then that block preceded by every earlier code block (helpers),
    then earlier solution() blocks, newest first."""
    code_blocks = [b for lab, b in _fences(text) if lab in CODE_LABELS and not b.lstrip().startswith(("{", "["))]
    sol_idx = [k for k, b in enumerate(code_blocks) if re.search(r"^\s*def\s+solution\s*\(", b, re.M)]
    variants = []
    for k in reversed(sol_idx):
        for v in (code_blocks[k], "\n\n".join(code_blocks[:k + 1])):
            if v not in variants:
                variants.append(v)
    objs = _test_outputs_objects(text)
    return variants[:MAX_PROGRAM_VARIANTS], (objs[-1] if objs else None)

RUNNER = r"""
import sys, json, os
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    _cpu = int(os.environ.get("Q27X_CPU_LIMIT", "30"))
    resource.setrlimit(resource.RLIMIT_CPU, (_cpu, _cpu + 1))       # v1.2: CPU seconds
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))   # v1.2: no large files
except Exception:
    pass
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
            if hasattr(r, "tolist"): r = r.tolist()
            out["outputs"].append([[int(v) for v in row] for row in r])
        except BaseException as e:
            out["outputs"].append(None); out["error"] = out["error"] or ("run: " + repr(e)[:500])
    sys.stdout.write("\n" + json.dumps(out) + "\n")
_main()
"""
_CHILD_ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")

def run_solution(code, inputs, timeout=VERIFY_TIMEOUT_SEC):
    """fresh Python process (own session) in a temporary directory; the whole process group is killed on timeout.
    v1.2: the timeout never runs past VERIFY_DEADLINE. Returns (outputs or None per input, error or None)."""
    timeout = min(float(timeout), max(1.0, VERIFY_DEADLINE - time.perf_counter()))
    try:
        with tempfile.TemporaryDirectory(prefix="q27x_sbx_") as cwd:
            proc = subprocess.Popen([sys.executable, "-c", RUNNER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, cwd=cwd, env=_CHILD_ENV,
                                    start_new_session=(os.name == "posix"))
            try:
                stdout, stderr = proc.communicate(json.dumps({"code": code, "inputs": inputs}), timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        import signal; os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    proc.kill()
                proc.communicate()
                return [None] * len(inputs), "timeout"
        lines = [l for l in (stdout or "").strip().splitlines() if l.startswith("{\"outputs\"")]
        res = json.loads(lines[-1]) if lines else {"outputs": [None] * len(inputs), "error": "no output: " + (stderr or "")[-300:]}
        outs = [o if valid_grid(o) else None for o in res.get("outputs", [])] + [None] * len(inputs)
        return outs[:len(inputs)], res.get("error")
    except Exception as e:
        return [None] * len(inputs), f"runner: {e!r}"[:300]

def verify_task(tid, variants, direct):
    task = CHALLENGES[tid]
    n = len(task["test"]); n_tr = len(task["train"])
    info = {"has_code": bool(variants), "variants_tried": 0, "train_pass": False, "train_correct": 0,
            "n_train": n_tr, "code_error": None, "code_test": None, "direct_valid": None}
    best = None
    for code in variants or []:
        if time.perf_counter() >= VERIFY_DEADLINE:
            info["code_error"] = "verify_deadline"; break
        info["variants_tried"] += 1
        # v1.2: train and test inputs in ONE process (was two)
        outs, err = run_solution(code, [ex["input"] for ex in task["train"]] + [q["input"] for q in task["test"]])
        correct = sum(o == ex["output"] for o, ex in zip(outs[:n_tr], task["train"]))
        if best is None or correct > best[1]:
            best = (code, correct, err, outs[n_tr:])
        if correct == n_tr:
            break
    if best is not None:
        code, correct, err, test_outs = best
        info["train_correct"] = correct; info["train_pass"] = correct == n_tr
        info["code_error"] = info["code_error"] or err; info["code_test"] = test_outs
    d = direct
    if n == 1 and valid_grid(d):                       # v1.1 (F9): a single grid given without the outer list
        d = [d]
    d = list(d) if isinstance(d, list) else []
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
CHALLENGES.setdefault("__selftest__", _t)
_layouts = {
    "instructed":      "Rule.\n```python\n" + _good + "```\n```json\n{\"test_outputs\": [[[7, 6]]]}\n```",
    "json_first":      "```json\n{\"test_outputs\": [[[7, 6]]]}\n```\n```python\nM = {1: 2}\n" + _good + "```",
    "text_first":      "Grid:\n```text\n76\n```\n```python\n" + _good + "```\n```json\n{\"test_outputs\": [[[7, 6]]]}\n```",
    "Python_label":    "```Python\n" + _good + "```\n{\"test_outputs\": [[[7, 6]]]}",
    "python3_label":   "```python3\n" + _good + "```\nAnswer: {\"test_outputs\": [[[7, 6]]]} and note {x}",
    "helper_block":    "```python\ndef helper(r):\n    return list(reversed(r))\n```\n```python\ndef solution(grid):\n    return [helper(r) for r in grid]\n```\n```json\n{\"test_outputs\": [[[7, 6]]]}\n```",
    "lone_grid":       "```python\n" + _good + "```\n```json\n{\"test_outputs\": [[7, 6]]}\n```",
    "plaintext_first": "```plaintext\nexample\n```\n```python\n" + _good + "```\n```json\n{\"test_outputs\": [[[7, 6]]]}\n```",
}
for _name, _ans in _layouts.items():
    _vs, _g = parse_answer(_ans)
    _v = verify_task("__selftest__", _vs, _g)
    assert _v["train_pass"] and _v["attempts"][0]["attempt_1"] == [[7, 6]], (_name, _vs, _g, _v.get("code_error"))
    assert _v["attempts"][0]["src_1"] == "verified_code", _name
    assert [[7, 6]] in (_v["attempts"][0]["attempt_1"], _v["attempts"][0]["attempt_2"]), _name
_v2 = verify_task("__selftest__", ["def solution(grid):\n    return grid\n"], [[[9, 9]]])
assert not _v2["train_pass"] and _v2["attempts"][0]["attempt_1"] == [[9, 9]] and _v2["attempts"][0]["src_2"] == "unverified_code"
_v3 = verify_task("__selftest__", ["def solution(grid):\n    while True: pass\n"], None)
assert _v3["code_error"], _v3
assert _v3["attempts"][0]["attempt_1"] == [[6, 7]]          # nothing usable -> test input
_v4 = verify_task("__selftest__", [], [[[7, 6]], [[1]]])
assert _v4["attempts"][0]["attempt_1"] == [[7, 6]]
_v5 = verify_task("__selftest__", ["def solution(grid):\n    return [[0]]\n", _good], None)   # second variant passes
assert _v5["train_pass"] and _v5["variants_tried"] == 2
assert parse_answer("no blocks at all") == ([], None)
_np_code = "import numpy as np\ndef solution(grid):\n    return np.fliplr(np.array(grid)).tolist()\n"
_vn = verify_task("__selftest__", [_np_code], None)
SANDBOX_NUMPY_OK = bool(_vn["train_pass"])
del CHALLENGES["__selftest__"]
print("cell 4 self-tests PASS (8 answer layouts) | numpy inside the sandbox:", SANDBOX_NUMPY_OK, _vn.get("code_error") or "")
if not SANDBOX_NUMPY_OK and os.environ.get("Q27X_SKIP_INSTALL") != "1":
    print("WARNING: numpy is not importable in the verification sandbox; numpy-based solutions will fail verification")
'''

C5 = r'''# ==================================================================== CELL 5 — engine (PIV/V3.1 C1 settings + long context + fp8 KV + MTP)
ENGINE_CONFIG = {
    "model": str(MODEL_PATH), "tensor_parallel_size": 4, "distributed_executor_backend": "mp", "dtype": "auto",
    "load_format": "safetensors", "trust_remote_code": False, "language_model_only": True,
    "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS, "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION, "enforce_eager": False, "disable_custom_all_reduce": True,
    "enable_prefix_caching": False, "seed": SEED,
    "disable_log_stats": False,      # v1.1 (F4): LLM() disables stats by default; we need MTP acceptance / KV usage logs
}
ENGINE_INFO = {"attempts": [], "speculative": None, "kv_cache_dtype": None, "load_sec": None, "kv_tokens": None, "concurrency": None}
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
import vllm

def _wait_gpu_release(max_sec=float(os.environ.get("Q27X_GPU_WAIT_MAX", 180))):
    """v1.1 (F7): a failed engine's worker processes may still hold GPU memory"""
    w0 = time.perf_counter()
    while time.perf_counter() - w0 < max_sec:
        try:
            used = [int(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                   capture_output=True, text=True, timeout=20).stdout.split()]
        except Exception:
            used = []
        if used and max(used) < 1024:
            break
        time.sleep(5)
    return round(time.perf_counter() - w0, 1)

# v1.2: ladder, most capable first. Every attempt and its error is recorded in run_report.json (nothing silent).
_LADDER = [(KV_CACHE_DTYPE, SPECULATIVE_CONFIG), ("auto", SPECULATIVE_CONFIG), ("auto", None)]
_seen = []
for _kv, _spec in _LADDER:
    if (_kv, json.dumps(_spec)) in _seen:
        continue
    _seen.append((_kv, json.dumps(_spec)))
    _t0 = time.perf_counter()
    _kw = dict(ENGINE_CONFIG)
    if _kv != "auto": _kw["kv_cache_dtype"] = _kv
    if _spec is not None: _kw["speculative_config"] = dict(_spec)
    try:
        llm = LLM(**_kw)
        ENGINE_INFO["attempts"].append({"kv_cache_dtype": _kv, "speculative": _spec, "ok": True,
                                        "sec": round(time.perf_counter() - _t0, 1)})
        ENGINE_INFO["speculative"] = _spec; ENGINE_INFO["kv_cache_dtype"] = _kv
        break
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:2000]
        print(f"ENGINE START FAILED with kv_cache_dtype={_kv} speculative={_spec}: {err[:600]}", flush=True)
        llm = None
        import gc; gc.collect()
        ENGINE_INFO["attempts"].append({"kv_cache_dtype": _kv, "speculative": _spec, "ok": False, "error": err,
                                        "gpu_release_wait_sec": _wait_gpu_release()})
if llm is None:
    raise RuntimeError("vLLM engine could not start in any configuration: " + json.dumps(ENGINE_INFO["attempts"])[:3000])
ENGINE_INFO["load_sec"] = round(sum(a.get("sec", 0) for a in ENGINE_INFO["attempts"]), 1)
engine = llm.llm_engine
try:   # vLLM copies the engine core's num_gpu_blocks into the front-end cache config at start-up
    _cc = engine.vllm_config.cache_config
    ENGINE_INFO["kv_tokens"] = int(_cc.num_gpu_blocks) * int(_cc.block_size) if _cc.num_gpu_blocks else None
    ENGINE_INFO["cache_dtype_resolved"] = str(getattr(_cc, "cache_dtype", None))
except Exception as e:
    ENGINE_INFO["kv_tokens"] = None; ENGINE_INFO["kv_error"] = repr(e)[:300]
CONCURRENCY = MAX_NUM_SEQS
if ENGINE_INFO["kv_tokens"]:
    CONCURRENCY = max(4, min(MAX_NUM_SEQS, ENGINE_INFO["kv_tokens"] // EXPECTED_LIVE_TOKENS_PER_SEQ))
ENGINE_INFO["concurrency"] = int(CONCURRENCY); ENGINE_INFO["vllm"] = vllm.__version__
print("engine ready:", json.dumps(ENGINE_INFO, default=str))

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
    engine.add_request(_req(tid, 1), {"prompt_token_ids": list(st["prompt_ids"])},
                       make_params(P1_CAPS[tid], SEED + 10 * TASK_INDEX[tid] + 1))
    SCHED["admitted"] += 1

def start_phase2(st, answer_cap):
    """think closed but answer cut -> continue; think never closed -> append stop text (token-exact) and answer"""
    closed = THINK_CLOSE_ID in st["p1_ids"]
    if not closed:                                  # v1.1 (F8): an EOS before </think> must not precede the stop text
        while st["p1_ids"] and st["p1_ids"][-1] in EOS_IDS:
            st["p1_ids"].pop()
    prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + ([] if closed else list(STOP_IDS))
    room = MAX_MODEL_LEN - len(prefix) - 8
    cap = min(int(answer_cap), room)
    if cap < 64:
        return False
    st["phase"] = 2; st["p2_prefix"] = [] if closed else list(STOP_IDS); st["mode"] = "continue" if closed else "forced_close"
    SCHED["phase2_continue" if closed else "phase2_forced"] += 1
    engine.add_request(_req(st["tid"], 2), {"prompt_token_ids": prefix}, make_params(cap, SEED + 10 * TASK_INDEX[st["tid"]] + 2))
    return True

def answer_text(st):
    ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
    if THINK_CLOSE_ID not in ids:
        return "", 0, False
    k = len(ids) - 1 - ids[::-1].index(THINK_CLOSE_ID)
    return tokenizer.decode(ids[k + 1:], skip_special_tokens=True), k, True

VERIFY_POOL = cf.ThreadPoolExecutor(max_workers=VERIFY_WORKERS)
PENDING = {}        # future -> tid

ANSWERS_DIR = OUT / "answers"; ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

def finish(st, reason):
    st["phase"] = "done"; st["t_done"] = time.perf_counter(); st["end_reason"] = reason
    text, think_tokens, closed = answer_text(st)
    variants, grids = parse_answer(text)
    st.update({"answer_chars": len(text), "think_tokens": int(think_tokens if closed else len(st["p1_ids"])),
               "think_closed": closed, "has_code": bool(variants), "has_grids": grids is not None, "grids": grids})
    try:   # v1.1 (F5): keep the answer and the end of the thinking for diagnosis
        _ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
        _tail = tokenizer.decode(_ids[max(0, (think_tokens if closed else len(_ids)) - 1500):(think_tokens if closed else len(_ids))],
                                 skip_special_tokens=False)
        (ANSWERS_DIR / f"{st['tid']}.txt").write_text(
            "\n".join([f"### end_reason={reason} mode={st['mode']}", "### THINKING TAIL", _tail, "### ANSWER", text, ""]),
            encoding="utf-8")
    except Exception as e:
        print("answer save failed:", st["tid"], repr(e)[:200])
    PENDING[VERIFY_POOL.submit(verify_task, st["tid"], variants, grids)] = st["tid"]

def collect_verifications(block=False):
    if block:   # v1.2: wait only until VERIFY_DEADLINE; unfinished verifications fall back to the direct grids
        while PENDING and time.perf_counter() < VERIFY_DEADLINE:
            cf.wait(list(PENDING), timeout=max(0.1, min(5.0, VERIFY_DEADLINE - time.perf_counter())),
                    return_when=cf.FIRST_COMPLETED)
            _collect_done()
        for fut in list(PENDING):
            tid = PENDING.pop(fut); fut.cancel()
            st = STATE[tid]
            _record(tid, st, verify_task(tid, [], st.get("grids")), late=True)
        return
    _collect_done()

def _collect_done():
    done = [f for f in list(PENDING) if f.done()]
    for fut in done:
        tid = PENDING.pop(fut)
        st = STATE[tid]
        try:
            info = fut.result()
        except Exception as e:
            info = verify_task(tid, [], st.get("grids")); info["error"] = repr(e)[:300]
        _record(tid, st, info)

def _record(tid, st, info, late=False):
        FINAL[tid] = info["attempts"]
        sc = score_task(tid, info["attempts"])
        rec = {"task_id": tid, "mode": st["mode"] or "natural", "end_reason": st["end_reason"],
               "prompt_tokens": len(st["prompt_ids"]), "p1_tokens": len(st["p1_ids"]), "p2_tokens": len(st["p2_ids"]),
               "think_tokens": st["think_tokens"], "think_closed": st["think_closed"], "p1_finish": st["p1_finish"],
               "p2_finish": st["p2_finish"], "sec": round(st["t_done"] - st["t_admit"], 1), "has_code": st["has_code"],
               "has_grids": st["has_grids"], "variants_tried": info.get("variants_tried"),
               "train_pass": info.get("train_pass"), "train_correct": info.get("train_correct"),
               "n_train": info.get("n_train"), "code_error": info.get("code_error"),
               "src": [(a["src_1"], a["src_2"]) for a in info["attempts"]], "score": sc,
               "verification_cut_by_deadline": bool(late), "p1_cap": P1_CAPS.get(tid)}
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

# v1.1 (F2): a runtime error in an ~11 h loop must not end the notebook with an exception (that scores zero even
# with a valid submission.json); keep every finished task, write the submission, report the error.
SCHED_ERROR = None
try:
    run_scheduler()
except BaseException:
    SCHED_ERROR = traceback.format_exc()
    print("SCHEDULER ERROR - keeping finished tasks:", flush=True); print(SCHED_ERROR[-3000:], flush=True)
    try:
        _act = [_req(s["tid"], s["phase"]) for s in STATE.values() if s["phase"] in (1, 2)]
        if _act: engine.abort_request(_act)
    except BaseException:
        pass
finally:
    try:
        collect_verifications(block=True)
    except BaseException:
        print("verification collection failed:", traceback.format_exc()[-1500:])
    SCHED.setdefault("wall_sec", round(time.perf_counter() - SCHED["started"], 1))
    SCHED.setdefault("gen_tokens", GEN_TOKENS_TOTAL)
    SCHED.setdefault("tok_per_sec", round(GEN_TOKENS_TOTAL / max(SCHED["wall_sec"], 1e-6), 1))
    SCHED["error"] = SCHED_ERROR[-4000:] if SCHED_ERROR else None
    write_submission()
print("scheduler done:", json.dumps({k: v for k, v in SCHED.items() if k != "error"}), "| error:", bool(SCHED_ERROR))
'''

C7 = r'''# ==================================================================== CELL 7 — report, projection, bundle
# v1.1 (F2): the report must never turn a finished run into a notebook exception
try:
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
        # v1.2: Kaggle metric = per test query (either attempt exact), averaged over ALL planned queries;
        # unfinished / skipped tasks count as misses. The task-normalised figure is kept as a diagnostic.
        _hits = [h for r in rows if r["score"] for h in r["score"]["query_hits"]]
        _planned_q = sum(len(SOLUTIONS[t]) for t in TASK_IDS)
        report["eval_query_hits"] = int(sum(_hits)); report["eval_query_total"] = int(_planned_q)
        report["eval_score_pct"] = round(100 * sum(_hits) / max(_planned_q, 1), 2)
        sc = [r["score"]["score"] for r in rows if r["score"]]
        report["eval_task_normalized_pct"] = round(100 * sum(sc) / max(len(TASK_IDS), 1), 2)
        report["eval_pass1_queries"] = sum(sum(r["score"]["pass1"]) for r in rows if r["score"])
    # projection for the hidden test (240 tasks) from this run's measured aggregate rate and mean tokens/task
    if SCHED.get("tok_per_sec") and report["mean_total_gen_tokens"]:
        _rate = SCHED["tok_per_sec"]
        _gen_h = 240 * report["mean_total_gen_tokens"] / _rate / 3600
        _setup_h = (ENGINE_INFO.get("load_sec") or 0) / 3600 + 0.1          # install + load + verification tail
        report["projection_240_tasks_generation_hours"] = round(_gen_h, 2)
        report["projection_240_tasks_end_to_end_hours"] = round(_gen_h + _setup_h, 2)
        report["projection_note"] = ("measured with up to %d concurrent tasks (%d slots configured); the hidden run's rate "
                                     "may differ (longer contexts, KV pressure, MTP acceptance) - this is not a guarantee"
                                     % (SCHED["max_active"], CONCURRENCY))
    write_json_atomic(OUT / "run_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=1, default=str))
    ZIP = WORKING / "Q27X_BUNDLE.zip"
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(WORKING).as_posix())
        if SUBMISSION_PATH.exists(): z.write(SUBMISSION_PATH, "submission.json")
    print("bundle:", ZIP, round(ZIP.stat().st_size / 1e6, 2), "MB")
    if SOLUTIONS is not None:
        print(f"EVAL SCORE (Kaggle metric, per query): {report['eval_query_hits']} / {report['eval_query_total']} queries = "
              f"{report['eval_score_pct']}% | task-normalised {report['eval_task_normalized_pct']}% | train-verified programs "
              f"{report['train_verified']} | throughput {SCHED['tok_per_sec']} tok/s | projected 240-task end-to-end hours "
              f"{report.get('projection_240_tasks_end_to_end_hours')}")
    try:
        VERIFY_POOL.shutdown(wait=False)
    except Exception:
        pass

except BaseException:
    print("REPORT ERROR (submission.json is already written):"); print(traceback.format_exc()[-3000:])
'''

cells = [{"cell_type": "markdown", "metadata": {}, "source": C_HEAD.splitlines(keepends=True)}]
for src in (C0, C1, C2, C3, C4, C5, C6, C7):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print("wrote", DST, hashlib.sha256(DST.read_bytes()).hexdigest())
