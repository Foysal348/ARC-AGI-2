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

DST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\ARC AGI 2\Research Base Experiment\Q27-R6-CODE-LOOP-v1_1.ipynb")

C_HEAD = r'''# Q27-R6 v1.1 — Qwen3.8-27B-FP8: think -> write solve(grid) -> run on every training pair -> feedback -> repair (4×L4)

R6 (2026-10-06, after run 4: 18/22 queries force-closed at 49k while still testing rule hypotheses by hand or computing
the output grid by hand; the 4 natural finishes were exact or 1 cell off). Per TASK, a conversation of up to MAX_TURNS
turns: the model writes `def solve(grid)`; it runs in a fresh process on every training input (exact match) and on the
test inputs; if a training pair fails, the differences (wrong rows, wrong size, the exception) are sent back and the
model writes a corrected solve(). Earlier turns keep only their code (preserve_thinking=False) so a repair turn
re-prefills a short context. attempt_1 = the train-verified program's output (else the best partial program),
attempt_2 = the next distinct program output, else the test input. Engine, deadlines and submission handling are v1.4's.

v1.1 (checked against discussion 745787 and run 4): turn-1 thinking cap 49,152 (NguyenThanhNhan's commit point; run 4's
natural finishes took 26.5k-42k thinking tokens, two of them above 32k). If no program reproduces every training pair
after MAX_TURNS, one last turn asks for the output grid(s) directly (v1.4's grid-only answer, the source of run 4's
hits); that direct grid becomes attempt_1 unless a program solved at least one training pair.

---
(v1.4 header kept below for reference)

# Q27-XHIGH v1.4 — Qwen3.8-27B-FP8, full xhigh thinking, one prompt per test input, grid-only answer (4×L4)

Dual mode:
* **Commit / interactive run** (`EVAL12`): every test input of 12 public evaluation tasks (fixed seed, the same 12
  tasks as run 3 = 22 queries), scored, with throughput measured.
* **Competition rerun** (`KAGGLE_IS_COMPETITION_RERUN`): every hidden test input (240 tasks / 259 queries) →
  `submission.json`.

A valid fallback `submission.json` (test input as both attempts) is written in cell 0, before anything can fail. It
is rewritten after every finished query.

Per test input (query), one generation:
1. The prompt is russcore's LB-10.14 free-form prompt: the training pairs and ONE test input, grids as rows of
   space-separated digits, "give ONLY the final output grid ... inside a fenced block ```grid".
2. Qwen thinks at **xhigh** effort, temperature 0.6 (russcore). If thinking has not ended after `THINK_BUDGET` tokens
   (49,152, NguyenThanhNhan's commit point), Qwen's documented stop sentence + `</think>` is appended (token-exact) to the
   full context and the grid is generated in a second request (≤ 8,192 tokens; total 57,344 = russcore's budget).
3. attempt_1 = the model's grid; attempt_2 = the test input (or its mirror if the model returned the test input).

Scheduling: vLLM step API with continuous admission. A query that finishes frees its slot at once; there are no
waves. Concurrency = min(16, KV pool tokens // 30k) from the KV cache vLLM actually allocated. MTP speculative decoding is
on (falls back to plain decoding, after the GPUs are released, if the engine refuses it); vLLM stats logging is on so
the MTP acceptance rate and KV usage are visible. Deadline: no new task
after the admission deadline; at the drain deadline in-flight thinking is force-closed; at the hard stop everything
is aborted and the submission written.

v1.4 (run 3, 2026-10-06: EVAL12 1/22; 12/12 tasks hit the 49k cap; only 6/12 JSON grids parsed; references:
russcore free-form LB 10.14, NguyenThanhNhan pure reasoning LB 23.33): the answer interface is russcore's LB-10.14
notebook (one prompt per test input, space-separated grids, grid-only fenced answer, temperature 0.6) and the commit
point is NguyenThanhNhan's (49k thinking, then the grid with the full reasoning in context). No program is requested
or executed. The parser reads the last fenced block that is a valid grid (space-separated, contiguous digits or a JSON
list, also with a missing bracket), else the last run of digit rows in the answer. The engine (fp8 KV, MTP 3,
FlashInfer, 4096 batched tokens, CUDA-graph memory profiling) is unchanged from v1.3.5.

v1.3.5 (Kaggle run 2, 2026-10-05): the libcuda fix worked; fp8 KV + MTP3 + FlashInfer loaded and passed the decode
smoke. Then the first prefill steps of 12 tasks ran out of GPU memory in the MTP drafter (`Tried to allocate 272.00 MiB`
= 32,768 tokens × 4,352 × 2 bytes; 153 MiB free; 0.61 GiB CUDA graphs outside vLLM's budget; 1,013 MiB fragmented).
Two changes, both from the run-2 log / vLLM 0.19 source:
* `MAX_NUM_BATCHED_TOKENS` 32768 → 4096 = NguyenThanhNhan's `max-prefill-tokens 4096` (8× smaller step tensors).
* `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (vLLM's own suggestion in the run-2 log): CUDA-graph memory is taken out
  of the KV budget instead of being allocated beyond it.
Concurrency unchanged (16): the KV pool is shared by 1 attention + 3 GDN groups (781 blocks × 1,600 tokens in run 2;
a 30k-token task needs 19 attention + 12 GDN-state blocks, so 16 tasks need ~496 blocks).

v1.3.4 (run 1 log): the real cause was FlashInfer's JIT LINK step, `/usr/bin/ld: cannot find -lcuda` (Kaggle has
only libcuda.so.1); the model, MTP and fp8 KV (312,000 tokens) had loaded. Cell 5 now creates a `libcuda.so` shim on
LIBRARY_PATH before vLLM starts and proves `-lcuda` links. Ladder with explicit attention backends: fp8 + FlashInfer (only
if the link probe passes) -> fp8 + Triton -> bf16 + FlashAttention-2 -> bf16 without MTP. Concurrency sized from the
measured KV (30k average live tokens per task).

v1.3.3 (Kaggle run 1, 2026-10-05): the first engine attempt (fp8_e4m3 KV + MTP3) died with `EngineDeadError` after
13 min and its worker processes kept ~20.7 GB per GPU, so the ladder stopped. Now every child process is killed after
a failed attempt (then the GPU-release check), and the run starts with bf16 KV + MTP3 (fallback: bf16 without MTP).

v1.3.2 (GPT final review): among train-exact program variants, one that also runs on every test input is preferred;
the 240-task projection is reported only for a clean run (no hard-stop abort, drain, unadmitted task or cut verification).

v1.3.1: parser variant = the target `solution()` plus every non-solution helper block (before or after it), without
the other `solution()` definitions.

v1.3 (GPT review R2, each point checked): deadlines derived in a guaranteed order (the EVAL12 drain used to fall after
the hard stop); every engine attempt must pass a real 16-token decode before it is used (fp8 KV / MTP / FlashInfer);
no further attempt if GPU memory is not released; if no engine starts the notebook still finishes with the identity
submission and a report; vLLM metrics (MTP acceptance, preemptions, KV usage) and the front-end log saved; projection
only from a complete run, with the measured setup time; concurrency sized for 55k live tokens per task; helper code
after `solution()` handled.

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
THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 49152))      # R6 v1.1: turn-1 thinking cap = NguyenThanhNhan's commit point
REPAIR_THINK_BUDGET = int(os.environ.get("Q27X_REPAIR_THINK", 12288))
MAX_TURNS = int(os.environ.get("Q27X_MAX_TURNS", 4))                   # turn 1 + up to 3 repair turns
REPAIR_EFFORT = os.environ.get("Q27X_REPAIR_EFFORT", "xhigh")
GRID_FALLBACK = os.environ.get("Q27X_GRID_FALLBACK", "1") == "1"      # R6 v1.1: last turn asks for the grid directly
GRID_THINK_BUDGET = int(os.environ.get("Q27X_GRID_THINK", 16384))
ANSWER_MAX_TOKENS = int(os.environ.get("Q27X_ANSWER_MAX", 8192))   # v1.4: grid-only answer; 49,152 + 8,192 = 57,344 = russcore's max_tokens
DRAIN_ANSWER_MAX_TOKENS = 8192                                      # v1.1 (F11): a 30x30 grid alone is ~2.7k tokens
STOP_THINKING_TEXT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
MAX_MODEL_LEN = int(os.environ.get("Q27X_MAX_MODEL_LEN", 81920))   # v1.2: native context is 262k; fp8 KV makes room
CONTEXT_SAFETY = 256
MIN_THINK_BUDGET = 4096                  # v1.2: a long prompt gets a smaller thinking budget instead of being skipped
PROMPT_TOKEN_LIMIT = MAX_MODEL_LEN - ANSWER_MAX_TOKENS - CONTEXT_SAFETY - MIN_THINK_BUDGET
def phase1_cap(prompt_len):
    return max(0, min(THINK_BUDGET, MAX_MODEL_LEN - prompt_len - ANSWER_MAX_TOKENS - CONTEXT_SAFETY))
assert phase1_cap(MAX_MODEL_LEN) == 0 and phase1_cap(100) == min(THINK_BUDGET, MAX_MODEL_LEN - 100 - ANSWER_MAX_TOKENS - CONTEXT_SAFETY)
if (MAX_MODEL_LEN, THINK_BUDGET, ANSWER_MAX_TOKENS) == (81920, 49152, 8192):
    assert phase1_cap(24320) == 49152 and phase1_cap(30000) == 43472  # every real prompt (≤ ~9k tokens) keeps the full budget
SAMPLING = {"temperature": float(os.environ.get("Q27X_TEMPERATURE", 0.6)), "top_p": 0.95, "top_k": 20}   # v1.4: russcore LB-10.14 sampling

# ---- engine (PIV/V3.1 C1 settings; context, concurrency and MTP changed)
MAX_NUM_SEQS = int(os.environ.get("Q27X_MAX_NUM_SEQS", 16))   # upper bound; actual slots = KV tokens // EXPECTED_LIVE_TOKENS_PER_SEQ
MAX_NUM_BATCHED_TOKENS = 4096   # v1.3.5: = discussion max-prefill-tokens 4096; 32768 ran out of memory in run 2 (MTP drafter)
GPU_MEMORY_UTILIZATION = 0.90
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": int(os.environ.get("Q27X_SPEC_TOKENS", 3))}   # v1.2: vLLM Qwen3.8 recipe uses 3
KV_CACHE_DTYPE = os.environ.get("Q27X_KV_DTYPE", "fp8_e4m3")   # v1.3.4: back to the discussion recipe; run 1's crash was
# FlashInfer's JIT link step (`ld: cannot find -lcuda`), fixed in cell 5 (libcuda shim + link probe); fp8 KV itself loaded
# fine (312,000 KV tokens). Set Q27X_KV_DTYPE=auto for bf16 KV.
if os.environ.get("Q27X_NO_SPEC") == "1":
    SPECULATIVE_CONFIG = None
EXPECTED_LIVE_TOKENS_PER_SEQ = 30000      # a running task holds on average about half of its final ~60k length
# v1.3.5 correction: cell 5 divides the whole KV pool (run 2: 781 blocks × 1,600 = 1,249,600 tokens) by this. The pool is
# shared by 1 attention + 3 GDN groups; vLLM's "312,000 tokens" log line is pool // 4. A 30k-token task needs 19 attention
# + 12 GDN-state blocks (3 groups × (1 + 3 MTP)), so 16 tasks need ~496 of 781 blocks -> 16 slots (preemptions reported)

# ---- verification (fresh process per candidate: train first, then test)
VERIFY_TIMEOUT_SEC = 20.0
VERIFY_WORKERS = 4

# ---- time
TOTAL_BUDGET_SEC = float(os.environ.get("Q27X_TIME_BUDGET", P["time_budget_sec"]))
# v1.3: deadlines derived in a guaranteed order (v1.2 had DRAIN after HARD_STOP for EVAL12, so the drain never ran)
VERIFY_DEADLINE = NOTEBOOK_T0 + TOTAL_BUDGET_SEC - 8 * 60              # verification never runs past this
HARD_STOP = NOTEBOOK_T0 + TOTAL_BUDGET_SEC - 25 * 60                  # abort everything, write outputs
DRAIN_DEADLINE = min(NOTEBOOK_T0 + TOTAL_BUDGET_SEC * P["drain_frac"], HARD_STOP - 8 * 60)    # force-close thinking
ADMIT_DEADLINE = min(NOTEBOOK_T0 + TOTAL_BUDGET_SEC * P["admit_frac"], DRAIN_DEADLINE - 7 * 60)  # no new task after this
assert NOTEBOOK_T0 < ADMIT_DEADLINE < DRAIN_DEADLINE < HARD_STOP < VERIFY_DEADLINE < NOTEBOOK_T0 + TOTAL_BUDGET_SEC

# v1.3.5 (run 2 OOM): CUDA-graph memory is taken out of the KV budget (vLLM 0.19 envs.py 1654, v1/worker/gpu_worker.py
# 393-420); run 2 captured 0.61 GiB of graphs beyond it. Set here, before cell 5 spawns the vLLM workers (they inherit it).
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
CONFIG = {k: globals()[k] for k in ["PROFILE", "RERUN", "SEED", "REASONING_EFFORT", "THINK_BUDGET", "ANSWER_MAX_TOKENS",
          "DRAIN_ANSWER_MAX_TOKENS", "MAX_MODEL_LEN", "PROMPT_TOKEN_LIMIT", "SAMPLING", "MAX_NUM_SEQS",
          "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION", "SPECULATIVE_CONFIG", "KV_CACHE_DTYPE", "VERIFY_TIMEOUT_SEC",
          "TOTAL_BUDGET_SEC", "MIN_THINK_BUDGET", "REPAIR_THINK_BUDGET", "MAX_TURNS", "REPAIR_EFFORT", "GRID_FALLBACK", "GRID_THINK_BUDGET"]}
CONFIG["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"]
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

C3 = r'''# ==================================================================== CELL 3 — tasks + turn-1 prompts (R6: one conversation per task)
if P["source"] == "eval":
    _ch = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
    _sol = _load_json(ARC_ROOT / "arc-agi_evaluation_solutions.json")
    TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))
    CHALLENGES = {t: _ch[t] for t in TASK_IDS}
    SOLUTIONS = {t: _sol[t] for t in TASK_IDS}
else:
    CHALLENGES = dict(test_challenges); TASK_IDS = sorted(CHALLENGES); SOLUTIONS = None

def render_grid(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)

def shape(g):
    return f"{len(g)}x{len(g[0]) if g else 0}"

# R6: the model writes the rule as code; we run it on every training pair (exact match) and feed back the differences
CODE_INSTRUCTIONS = ("Write the rule as a Python function\n```python\ndef solve(grid: list[list[int]]) -> list[list[int]]:\n    ...\n```\n"
                     "We will run solve() on every example INPUT and compare the result with the example OUTPUT cell by cell, "
                     "then run it on the test input(s). You may use the Python standard library and numpy. Begin the code with a "
                     "short comment that states the rule. Reply with exactly one ```python block and nothing after it.")

def build_task_message(task):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an INPUT grid and its OUTPUT grid (digits 0-9 "
             "are colors; 0 is usually background). Infer the ONE rule that maps every input to its output.\n"]
    for i, p in enumerate(task["train"]):
        parts.append("Example %d INPUT (%s):\n%s\nExample %d OUTPUT (%s):\n%s\n" %
                     (i + 1, shape(p["input"]), render_grid(p["input"]), i + 1, shape(p["output"]), render_grid(p["output"])))
    for k, q in enumerate(task["test"]):
        parts.append("Test INPUT %d (%s):\n%s\n" % (k + 1, shape(q["input"]), render_grid(q["input"])))
    parts.append(CODE_INSTRUCTIONS)
    return "\n".join(parts)

def render_conversation(messages, effort):
    # preserve_thinking=False: earlier turns keep only their answer (the code), not their thinking -> short re-prefill
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                         reasoning_effort=effort, preserve_thinking=False)

UNITS = list(TASK_IDS)                                       # R6: a unit is a task (one program serves all its test inputs)
UNIT_INDEX = {u: i for i, u in enumerate(UNITS)}             # v1.1 (F13): seeds do not depend on timing
N_HIDDEN_TASKS = len(test_challenges)
N_HIDDEN_QUERIES = sum(len(t["test"]) for t in test_challenges.values())

MESSAGES0, PROMPT_IDS, SKIPPED, P1_CAPS = {}, {}, {}, {}
for tid in UNITS:
    MESSAGES0[tid] = [{"role": "user", "content": build_task_message(CHALLENGES[tid])}]
    ids = tokenizer.encode(render_conversation(MESSAGES0[tid], REASONING_EFFORT), add_special_tokens=False)
    cap = phase1_cap(len(ids))
    if cap < min(MIN_THINK_BUDGET, THINK_BUDGET):  # v1.2: only prompts that leave < 4k thinking tokens are skipped
        SKIPPED[tid] = {"prompt_tokens": len(ids), "reason": "insufficient_context"}; continue
    PROMPT_IDS[tid] = ids; P1_CAPS[tid] = cap
_pl = sorted(len(v) for v in PROMPT_IDS.values())
print(f"{PROFILE}: {len(TASK_IDS)} tasks, {sum(len(CHALLENGES[t]['test']) for t in TASK_IDS)} test queries (hidden test: "
      f"{N_HIDDEN_TASKS} tasks / {N_HIDDEN_QUERIES} queries) | turn-1 prompt tokens median {_pl[len(_pl)//2] if _pl else 0}, "
      f"max {_pl[-1] if _pl else 0} | reduced thinking budget: {sum(c < THINK_BUDGET for c in P1_CAPS.values())} | "
      f"skipped (too long): {len(SKIPPED)}")
_m0 = MESSAGES0[UNITS[0]][0]["content"]
assert "xhigh" not in _m0 and "Test INPUT 1" in _m0 and _m0.endswith("nothing after it.")
assert render_grid([[1, 0], [2, 3]]) == "1 0\n2 3"
'''

C4 = r'''# ==================================================================== CELL 4 — code extraction, fresh-process verification, feedback, attempts
import tempfile
def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
CODE_LABELS = {"", "python", "py", "python3"}
_ENTRY_RE = re.compile(r"^\s*def\s+(solve|solution|transform)\s*\(", re.M)

def extract_code(text):
    """all python blocks of the answer in order (helpers + solve); an unclosed final block is accepted"""
    t = text or ""
    blocks = [b for lab, b in _FENCE_RE.findall(t) if lab.strip().lower() in CODE_LABELS]
    if not any(_ENTRY_RE.search(b) for b in blocks):
        m = re.search(r"```(?:python|py|python3)?[ \t]*\n((?:(?!```).)*)$", t, re.S)   # fence opened but never closed
        if m and _ENTRY_RE.search(m.group(1)):
            blocks.append(m.group(1))
    code = "\n\n".join(b for b in blocks if not b.lstrip().startswith(("{", "[")))
    return code if _ENTRY_RE.search(code) else None

RUNNER = r"""
import sys, json, os
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    _cpu = int(os.environ.get("Q27X_CPU_LIMIT", "30"))
    resource.setrlimit(resource.RLIMIT_CPU, (_cpu, _cpu + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
except Exception:
    pass
def _main():
    payload = json.loads(sys.stdin.read())
    ns = {"__name__": "candidate"}
    out = {"outputs": [], "error": None}
    try:
        exec(compile(payload["code"], "<solve>", "exec"), ns)
        fn = ns.get("solve") or ns.get("solution") or ns.get("transform")
        if fn is None:
            raise NameError("solve() is not defined")
    except BaseException as e:
        out["error"] = "define: " + repr(e)[:400]; print(json.dumps(out)); return
    for g in payload["inputs"]:
        try:
            r = fn([list(row) for row in g])
            if hasattr(r, "tolist"): r = r.tolist()
            out["outputs"].append([[int(v) for v in row] for row in r])
        except BaseException as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            where = next((f"line {f.lineno}" for f in reversed(tb) if f.filename == "<solve>"), "")
            out["outputs"].append(None); out["error"] = out["error"] or ("run: " + repr(e)[:300] + (" at " + where if where else ""))
    sys.stdout.write("\n" + json.dumps(out) + "\n")
_main()
"""
_CHILD_ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")

def run_solution(code, inputs, timeout=VERIFY_TIMEOUT_SEC):
    """fresh Python process (own session) in a temporary directory; the whole process group is killed on timeout.
    The timeout never runs past VERIFY_DEADLINE. Returns (outputs or None per input, error or None)."""
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
                return [None] * len(inputs), f"timeout after {timeout:.0f} s"
        lines = [l for l in (stdout or "").strip().splitlines() if l.startswith("{\"outputs\"")]
        res = json.loads(lines[-1]) if lines else {"outputs": [None] * len(inputs), "error": "no output: " + (stderr or "")[-300:]}
        outs = [o if valid_grid(o) else None for o in res.get("outputs", [])] + [None] * len(inputs)
        return outs[:len(inputs)], res.get("error")
    except Exception as e:
        return [None] * len(inputs), f"runner: {e!r}"[:300]

def verify_program(tid, code):
    """run on every training input and every test input in ONE fresh process"""
    task = CHALLENGES[tid]; n_tr = len(task["train"])
    if not code:
        return {"code_found": False, "train_correct": 0, "n_train": n_tr, "per_example": [False] * n_tr,
                "train_outs": [None] * n_tr, "test_outs": [None] * len(task["test"]), "test_ok": False,
                "error": "no python block with solve()", "verified": False, "cells_right": 0.0}
    outs, err = run_solution(code, [ex["input"] for ex in task["train"]] + [q["input"] for q in task["test"]])
    tr, te = outs[:n_tr], outs[n_tr:]
    per = [o == ex["output"] for o, ex in zip(tr, task["train"])]
    test_ok = all(valid_grid(x) for x in te)
    # fraction of correct cells over the training pairs (wrong size counts 0) - ranks unverified programs
    fr = []
    for o, ex in zip(tr, task["train"]):
        e = ex["output"]
        if o is not None and len(o) == len(e) and all(len(a) == len(b) for a, b in zip(o, e)):
            fr.append(sum(x == y for a, b in zip(o, e) for x, y in zip(a, b)) / sum(len(r) for r in e))
        else:
            fr.append(0.0)
    return {"code_found": True, "train_correct": sum(per), "n_train": n_tr, "per_example": per, "train_outs": tr,
            "test_outs": te, "test_ok": test_ok, "error": err, "verified": all(per) and test_ok,
            "cells_right": round(sum(fr) / max(len(fr), 1), 4)}

FEEDBACK_MAX_CHARS = 3000
def feedback_text(tid, res):
    task = CHALLENGES[tid]
    if not res["code_found"]:
        return ("Your reply did not contain a ```python block that defines solve(grid). Reply with exactly one ```python "
                "block that defines solve(grid).")
    lines = ["Your solve() was run on the training examples:"]
    for i, ex in enumerate(task["train"]):
        got, exp = res["train_outs"][i], ex["output"]
        if res["per_example"][i]:
            lines.append(f"- Example {i + 1}: CORRECT")
        elif got is None:
            lines.append(f"- Example {i + 1}: ERROR - {res['error'] or 'no valid grid returned'}")
        elif len(got) != len(exp) or any(len(a) != len(b) for a, b in zip(got, exp)):
            lines.append(f"- Example {i + 1}: WRONG SIZE - expected {shape(exp)}, got {shape(got)}")
        else:
            bad = [r for r in range(len(exp)) if got[r] != exp[r]]
            n = sum(x != y for a, b in zip(got, exp) for x, y in zip(a, b))
            lines.append(f"- Example {i + 1}: WRONG - {n} of {sum(len(r) for r in exp)} cells differ, in rows "
                         + ", ".join(str(r) for r in bad[:12]) + (" ..." if len(bad) > 12 else ""))
            for r in bad[:3]:
                lines.append(f"    row {r}: expected {' '.join(map(str, exp[r]))}")
                lines.append(f"    row {r}: yours    {' '.join(map(str, got[r]))}")
    if all(res["per_example"]) and not res["test_ok"]:
        bad_t = [k + 1 for k, x in enumerate(res["test_outs"]) if not valid_grid(x)]
        lines.append(f"All training examples are correct, but solve() did not return a valid grid for test input(s) "
                     f"{bad_t}: {res['error'] or 'invalid output'}")
    lines.append("Find what is wrong with the rule or the code and reply with a corrected solve() in exactly one "
                 "```python block.")
    txt = "\n".join(lines)
    return txt if len(txt) <= FEEDBACK_MAX_CHARS else txt[:FEEDBACK_MAX_CHARS - 40] + "\n... (truncated)\n" + lines[-1]

# R6 v1.1: v1.4's grid parser (run 4: 20/22 grids parsed) for the direct-grid fallback turn
_SEP_ROW = re.compile(r"\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?")
_JSON_ROW = re.compile(r"\[\s*\d(?:\s*,\s*\d)*\s*\]")

def _parse_row(s):
    s = s.strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return [int(c) for c in s]
    if _SEP_ROW.fullmatch(s):
        return [int(c) for c in re.findall(r"\d", s)]
    return None

def _first_row_run(lines):
    rows = []
    for line in lines:
        r = _parse_row(line)
        if r is None:
            if rows:
                break
            continue
        rows.append(r)
    return rows if valid_grid(rows) else None

def _grid_from_obj(obj):
    if isinstance(obj, dict):
        for k in ("output", "grid", "test_output", "answer", "test_outputs"):
            if k in obj:
                return _grid_from_obj(obj[k])
        return None
    if valid_grid(obj):
        return obj
    if isinstance(obj, list) and obj and valid_grid(obj[-1]):
        return obj[-1]
    return None

def _json_rows_grid(b):
    grids, cur = [], []
    for m in _JSON_ROW.finditer(b):
        cur.append(json.loads(m.group(0)))
        if b[m.end():].lstrip().startswith("]"):
            grids.append(cur); cur = []
    if cur:
        grids.append(cur)
    for g in reversed(grids):
        if valid_grid(g):
            return g
    return None

def _grid_from_block(body):
    b = body.strip()
    if b[:1] in ("[", "{"):
        try:
            g = _grid_from_obj(json.loads(b))
        except Exception:
            g = None
        if g is None:
            g = _json_rows_grid(b)
        if g is not None:
            return g
    return _first_row_run(b.splitlines())

def parse_grid(text):
    for _lab, body in reversed(_FENCE_RE.findall(text or "")):
        g = _grid_from_block(body)
        if g is not None:
            return g
    runs, cur = [], []
    for line in (text or "").splitlines():
        r = _parse_row(line)
        if r is None:
            if cur:
                runs.append(cur); cur = []
        else:
            cur.append(r)
    if cur:
        runs.append(cur)
    for run in reversed(runs):
        if valid_grid(run):
            return run
    return None

def parse_grids(text, n):
    """one grid per test input, in order: the last n fenced blocks that hold a valid grid; a single test input may also be
    answered without a fence"""
    grids = [g for g in (_grid_from_block(b) for _l, b in _FENCE_RE.findall(text or "")) if g is not None]
    if len(grids) >= n:
        return grids[-n:]
    if n == 1 and not grids:
        g = parse_grid(text)
        return [g]
    return grids + [None] * (n - len(grids))

def make_attempts(tid, programs, direct=None):
    """programs: verification results of every code turn. direct: grids of the grid-only fallback turn (or None).
    Verified programs first; else, if a program reproduced at least one training pair, its output first and the direct grid
    second; otherwise the direct grid first. Then other programs, the test input and its mirror; first two DISTINCT win."""
    ranked = sorted(programs, key=lambda p: (p["verified"], p["train_correct"], p["cells_right"], p["turn"]), reverse=True)
    code_first = bool(ranked) and (ranked[0]["verified"] or ranked[0]["train_correct"] > 0)
    atts = []
    for qi, q in enumerate(CHALLENGES[tid]["test"]):
        code_c = [("verified_code" if p["verified"] else "unverified_code", p["test_outs"][qi]) for p in ranked
                  if qi < len(p["test_outs"]) and valid_grid(p["test_outs"][qi])]
        grid_c = [("direct_grid", direct[qi])] if direct and qi < len(direct) and valid_grid(direct[qi]) else []
        cands = (code_c[:1] + grid_c + code_c[1:]) if code_first else (grid_c + code_c)
        cands += [("test_input", q["input"]), ("mirror", [list(reversed(r)) for r in q["input"]])]
        uniq = []
        for src, g in cands:
            if all(g != u[1] for u in uniq):
                uniq.append((src, g))
        while len(uniq) < 2:
            uniq.append(uniq[-1])
        atts.append({"attempt_1": uniq[0][1], "attempt_2": uniq[1][1], "src_1": uniq[0][0], "src_2": uniq[1][0]})
    return atts

def score_task(tid, attempts):
    if SOLUTIONS is None:
        return None
    sol = SOLUTIONS[tid]
    hits = [int(a["attempt_1"] == s or a["attempt_2"] == s) for a, s in zip(attempts, sol)]
    return {"query_hits": hits, "pass1": [int(a["attempt_1"] == s) for a, s in zip(attempts, sol)],
            "score": sum(hits) / len(sol)}

# ---- self-tests (CPU, every run)
_t = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": [[3, 4, 5]], "output": [[5, 4, 3]]}],
      "test": [{"input": [[6, 7]]}]}
CHALLENGES["__selftest__"] = _t
_good = "# reverse every row\ndef solve(grid):\n    return [list(reversed(r)) for r in grid]\n"
_code_layouts = {
    "plain":        "```python\n" + _good + "```",
    "no_label":     "```\n" + _good + "```",
    "helper_after": "```python\ndef solve(grid):\n    return [h(r) for r in grid]\n```\n```python\ndef h(r):\n    return list(reversed(r))\n```",
    "solution_name": "```python\ndef solution(grid):\n    return [list(reversed(r)) for r in grid]\n```",
    "unclosed":     "Here:\n```python\n" + _good,
    "numpy":        "```python\nimport numpy as np\ndef solve(grid):\n    return np.fliplr(np.array(grid)).tolist()\n```",
}
for _n, _a in _code_layouts.items():
    _c = extract_code(_a); _r = verify_program("__selftest__", _c)
    assert _c and _r["verified"] and _r["test_outs"] == [[[7, 6]]], (_n, _c, _r.get("error"))
assert extract_code("no code") is None and extract_code("```json\n[[1]]\n```") is None
_r = verify_program("__selftest__", "def solve(grid):\n    return grid\n")
assert not _r["verified"] and _r["train_correct"] == 0 and _r["cells_right"] == 0.1667, _r["cells_right"]   # (0/2 + 1/3) / 2
_fb = feedback_text("__selftest__", _r)
assert "Example 1: WRONG - 2 of 2 cells differ" in _fb and "row 0: expected 2 1" in _fb and "row 0: yours    1 2" in _fb, _fb
_r = verify_program("__selftest__", "def solve(grid):\n    return [r + [0] for r in grid]\n")
assert "WRONG SIZE - expected 1x2, got 1x3" in feedback_text("__selftest__", _r)
_r = verify_program("__selftest__", "def solve(grid):\n    return grid[5]\n")
assert _r["error"] and "IndexError" in _r["error"] and "ERROR" in feedback_text("__selftest__", _r), _r["error"]
_r = verify_program("__selftest__", "def solve(grid):\n    while True: pass\n")
assert _r["error"] and not _r["verified"]
_r = verify_program("__selftest__", "def solve(grid):\n    if grid == [[6, 7]]:\n        raise ValueError('t')\n    return [list(reversed(r)) for r in grid]\n")
assert all(_r["per_example"]) and not _r["verified"] and "test input(s) [1]" in feedback_text("__selftest__", _r)
assert "did not contain" in feedback_text("__selftest__", verify_program("__selftest__", None))
_pg = dict(verify_program("__selftest__", _good), turn=2); _pb = dict(verify_program("__selftest__", "def solve(g):\n    return [[9, 9]]\n"), turn=3)
_a = make_attempts("__selftest__", [_pb, _pg])
assert (_a[0]["attempt_1"], _a[0]["src_1"], _a[0]["attempt_2"], _a[0]["src_2"]) == ([[7, 6]], "verified_code", [[9, 9]], "unverified_code"), _a
_a = make_attempts("__selftest__", [])
assert (_a[0]["src_1"], _a[0]["src_2"]) == ("test_input", "mirror")
_a = make_attempts("__selftest__", [_pb], [[[7, 6]]])                  # no partial program -> direct grid first
assert (_a[0]["src_1"], _a[0]["attempt_1"], _a[0]["src_2"]) == ("direct_grid", [[7, 6]], "unverified_code"), _a
_pp = dict(verify_program("__selftest__", "def solve(g):\n    return [list(reversed(r)) for r in g] if len(g[0]) == 2 else g\n"), turn=1)
assert _pp["train_correct"] == 1 and not _pp["verified"]
_a = make_attempts("__selftest__", [_pp], [[[1, 1]]])                  # partial program (1 of 2 pairs) -> program first
assert (_a[0]["src_1"], _a[0]["src_2"], _a[0]["attempt_2"]) == ("unverified_code", "direct_grid", [[1, 1]]), _a
assert parse_grids("```grid\n1 2\n```\ntext\n```grid\n3 4\n```", 2) == [[[1, 2]], [[3, 4]]]
assert parse_grids("Final:\n1 2\n", 1) == [[[1, 2]]] and parse_grids("```grid\n1 2\n```", 2) == [[[1, 2]], None]
assert parse_grids("```grid\n9\n```\n```grid\n1 2\n```\n```grid\n3 4\n```", 2) == [[[1, 2]], [[3, 4]]]
del CHALLENGES["__selftest__"]
print(f"cell 4 self-tests PASS ({len(_code_layouts)} code layouts, wrong / size / crash / timeout / test-crash feedback, attempts with and without a direct grid, multi-grid parsing)")
'''

C5 = r'''# ==================================================================== CELL 5 — engine (PIV/V3.1 C1 settings + long context + fp8 KV + MTP)
# ---- v1.3.4: root cause of Kaggle run 1 (2026-10-05). With fp8 KV, vLLM picks FlashInfer on sm89; FlashInfer JIT-compiled
# its kernels and then failed to LINK them: `/usr/bin/ld: cannot find -lcuda` (Kaggle ships only libcuda.so.1). Make
# `-lcuda` linkable BEFORE vLLM starts its worker processes (they inherit LIBRARY_PATH), and prove it with a tiny link.
# Same fix as dfranzen's Kaggle launcher (libcuda.so symlink), done in a private shim directory.
import shutil as _shutil
def ensure_libcuda_linkable(candidates=None, workdir="/tmp/q27x_libcuda_shim"):
    info = {"needed": None, "real": None, "shim": None, "probe": "FAIL"}
    try:
        work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
        if candidates is None:
            existing = [Path(d, "libcuda.so") for d in ("/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs",
                                                        "/usr/lib/x86_64-linux-gnu", "/usr/lib64")]
            existing = [p for p in existing if p.exists()]
        else:
            existing = []
        if existing:
            info.update(needed=False, real=str(existing[0]))
        else:
            cands = list(candidates) if candidates is not None else []
            if candidates is None:
                for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64",
                          "/usr/local/cuda/compat"):
                    try:
                        cands += sorted(str(p) for p in Path(d).glob("libcuda.so.*") if p.is_file())
                    except Exception:
                        pass
                try:
                    for line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout.splitlines():
                        if "libcuda.so.1" in line and "=>" in line:
                            cands.append(line.split("=>")[-1].strip())
                except Exception:
                    pass
            real = next((Path(c) for c in cands if Path(c).is_file()), None)
            if real is None:
                info["reason"] = "libcuda.so.1 not found"
                return info
            link = work / "libcuda.so"
            if link.exists() or link.is_symlink():
                link.unlink()
            try:
                link.symlink_to(real.resolve())
            except OSError:
                _shutil.copyfile(real, link)
            old = os.environ.get("LIBRARY_PATH", "")
            os.environ["LIBRARY_PATH"] = f"{work}:{old}" if old else str(work)
            info.update(needed=True, real=str(real), shim=str(link), library_path=os.environ["LIBRARY_PATH"])
        cxx = _shutil.which("c++") or _shutil.which("g++")
        if not cxx:
            info["reason"] = "no C++ compiler for the link probe"
            return info
        cp = subprocess.run([cxx, "-shared", "-fPIC", "-x", "c++", "-", "-lcuda", "-o", str(work / "probe.so")],
                            input='extern "C" int q27x_probe() { return 0; }\n', text=True, capture_output=True,
                            timeout=120, env=os.environ.copy())
        info["probe"] = "PASS" if cp.returncode == 0 else "FAIL"
        if cp.returncode != 0:
            info["probe_stderr"] = (cp.stderr or "")[-1500:]
    except Exception as e:
        info["reason"] = f"{type(e).__name__}: {e}"[:500]
    return info

os.environ.setdefault("MAX_JOBS", "4")              # bound FlashInfer JIT build parallelism
CUDA_LINK = ensure_libcuda_linkable()
CONFIG["cuda_link"] = CUDA_LINK
print("CUDA/FlashInfer linker preflight:", json.dumps(CUDA_LINK))

ENGINE_CONFIG = {
    "model": str(MODEL_PATH), "tensor_parallel_size": 4, "distributed_executor_backend": "mp", "dtype": "auto",
    "load_format": "safetensors", "trust_remote_code": False, "language_model_only": True,
    "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS, "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION, "enforce_eager": False, "disable_custom_all_reduce": True,
    "enable_prefix_caching": False, "seed": SEED,
    "disable_log_stats": False,      # v1.1 (F4): LLM() disables stats by default; we need MTP acceptance / KV usage
}
ENGINE_INFO = {"attempts": [], "speculative": None, "kv_cache_dtype": None, "attention_backend": None, "load_sec": None,
               "kv_tokens": None, "concurrency": None, "fatal": None, "cuda_link": CUDA_LINK}
import logging
try:   # v1.3: keep vLLM's front-end log (stats lines incl. speculative decoding) in the bundle
    _vllm_log_handler = logging.FileHandler(OUT / "vllm_frontend.log", mode="w", encoding="utf-8")
    _vllm_log_handler.setLevel(logging.INFO)
    _vllm_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("vllm").addHandler(_vllm_log_handler)
except Exception as _e:
    print("vLLM log file handler not installed:", repr(_e)[:200])
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
import vllm

def _wait_gpu_release(max_sec=float(os.environ.get("Q27X_GPU_WAIT_MAX", 180))):
    """v1.1 (F7) + v1.3: wait until every GPU uses < 1 GB; report whether that actually happened"""
    w0 = time.perf_counter(); used = []
    while time.perf_counter() - w0 < max_sec:
        try:
            used = [int(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                   capture_output=True, text=True, timeout=20).stdout.split()]
        except Exception:
            used = []
        if used and max(used) < 1024:
            return {"released": True, "sec": round(time.perf_counter() - w0, 1), "last_used_mb": used}
        time.sleep(5)
    return {"released": False, "sec": round(time.perf_counter() - w0, 1), "last_used_mb": used}

def _kill_child_processes(grace_sec=20):
    """v1.3.3: terminate, then kill, every descendant process of this kernel (vLLM engine core + workers after a failure)"""
    try:
        import psutil
        kids = psutil.Process(os.getpid()).children(recursive=True)
        for k in kids:
            try: k.terminate()
            except Exception: pass
        _gone, alive = psutil.wait_procs(kids, timeout=grace_sec)
        for k in alive:
            try: k.kill()
            except Exception: pass
        psutil.wait_procs(alive, timeout=grace_sec)
        return {"found": len(kids), "killed_hard": len(alive)}
    except Exception as e:
        return {"error": repr(e)[:200]}

def _engine_smoke(llm_obj):
    """v1.3: a constructor that succeeds is not enough - run a real 16-token decode (fp8 KV / MTP / attention kernels)"""
    ids = tokenizer.encode("Reply with the digits 1 2 3 4 5 6 7 8:", add_special_tokens=False)
    out = llm_obj.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=16, temperature=0.0, detokenize=False),
                           use_tqdm=False)
    toks = list(out[0].outputs[0].token_ids) if out and out[0].outputs else []
    if not toks:
        raise RuntimeError("engine decode smoke produced no tokens")
    return len(toks)

# v1.3.4 ladder: (kv cache dtype, speculative config, attention backend), most capable first. The backend is explicit so
# vLLM can never fall back to FlashInfer silently. fp8 + FlashInfer only if `-lcuda` links; fp8 + Triton attention keeps
# the fp8 KV memory saving without FlashInfer's JIT; bf16 + FlashAttention-2 is the PIV/V3.1-proven path.
_LADDER = []
if KV_CACHE_DTYPE.startswith("fp8"):
    if CUDA_LINK.get("probe") == "PASS":
        _LADDER.append((KV_CACHE_DTYPE, SPECULATIVE_CONFIG, "FLASHINFER"))
    _LADDER.append((KV_CACHE_DTYPE, SPECULATIVE_CONFIG, "TRITON_ATTN"))
_LADDER += [("auto", SPECULATIVE_CONFIG, "FLASH_ATTN"), ("auto", None, "FLASH_ATTN")]
_seen = []
llm = None
for _kv, _spec, _backend in _LADDER:
    if (_kv, json.dumps(_spec), _backend) in _seen:
        continue
    _seen.append((_kv, json.dumps(_spec), _backend))
    _t0 = time.perf_counter()
    _kw = dict(ENGINE_CONFIG, attention_config={"backend": _backend})     # LLM(attention_config=...) in vLLM 0.19
    if _kv != "auto": _kw["kv_cache_dtype"] = _kv
    if _spec is not None: _kw["speculative_config"] = dict(_spec)
    _att = {"kv_cache_dtype": _kv, "speculative": _spec, "attention_backend": _backend}
    print(f"ENGINE ATTEMPT: kv_cache_dtype={_kv} speculative={_spec} attention_backend={_backend}", flush=True)
    try:
        llm = LLM(**_kw)
        _att["smoke_tokens"] = _engine_smoke(llm)
        _att.update(ok=True, sec=round(time.perf_counter() - _t0, 1))
        ENGINE_INFO["attempts"].append(_att)
        ENGINE_INFO.update(speculative=_spec, kv_cache_dtype=_kv, attention_backend=_backend)
        break
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:2000]
        print(f"ENGINE ATTEMPT FAILED (kv_cache_dtype={_kv}, speculative={_spec}, backend={_backend}): {err[:600]}", flush=True)
        try:
            if llm is not None and hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "engine_core"):
                llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        llm = None
        import gc; gc.collect()
        # v1.3.3: LLM() itself may raise (run 1), leaving the vLLM workers alive with ~20.7 GB per GPU
        _att["killed_children"] = _kill_child_processes()
        _rel = _wait_gpu_release()
        _att.update(ok=False, error=err, sec=round(time.perf_counter() - _t0, 1), gpu_release=_rel)
        ENGINE_INFO["attempts"].append(_att)
        if not _rel["released"]:
            ENGINE_INFO["fatal"] = "gpu_memory_not_released_after_failed_engine"
            print("GPU memory was not released - no further engine attempts", flush=True)
            break
ENGINE_READY = llm is not None
ENGINE_INFO["load_sec"] = round(sum(a.get("sec", 0) for a in ENGINE_INFO["attempts"]), 1)
if not ENGINE_READY:
    # v1.3: do not raise - the identity submission from cell 0 stays valid, the report explains the failure
    ENGINE_INFO["fatal"] = ENGINE_INFO["fatal"] or "all_engine_attempts_failed"
    engine = None; CONCURRENCY = 0
    print("ENGINE INVALID - keeping the identity submission:", ENGINE_INFO["fatal"], flush=True)
else:
    engine = llm.llm_engine
    try:   # vLLM copies the engine core's num_gpu_blocks into the front-end cache config at start-up
        _cc = engine.vllm_config.cache_config
        ENGINE_INFO["kv_tokens"] = int(_cc.num_gpu_blocks) * int(_cc.block_size) if _cc.num_gpu_blocks else None
        ENGINE_INFO["cache_dtype_resolved"] = str(getattr(_cc, "cache_dtype", None))
    except Exception as e:
        ENGINE_INFO["kv_tokens"] = None; ENGINE_INFO["kv_error"] = repr(e)[:300]
    try:   # what vLLM actually resolved (requested backend is in ENGINE_INFO["attention_backend"])
        ENGINE_INFO["attention_backend_resolved"] = str(getattr(engine.vllm_config.attention_config, "backend", None))
    except Exception as e:
        ENGINE_INFO["attention_backend_resolved"] = None; ENGINE_INFO["attention_backend_error"] = repr(e)[:300]
    CONCURRENCY = MAX_NUM_SEQS
    if ENGINE_INFO["kv_tokens"]:
        CONCURRENCY = max(4, min(MAX_NUM_SEQS, ENGINE_INFO["kv_tokens"] // EXPECTED_LIVE_TOKENS_PER_SEQ))
ENGINE_INFO["concurrency"] = int(CONCURRENCY); ENGINE_INFO["vllm"] = vllm.__version__
print("engine:", json.dumps(ENGINE_INFO, default=str)[:3000])

def make_params(max_tokens, seed):
    return SamplingParams(max_tokens=int(max_tokens), seed=int(seed), detokenize=False, skip_special_tokens=False,
                          output_kind=RequestOutputKind.DELTA, **SAMPLING)

def metrics_snapshot():
    """v1.3: vLLM Prometheus snapshot (speculative decoding, preemptions, KV usage, tokens) as plain JSON"""
    if not ENGINE_READY:
        return []
    try:
        snap = []
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            if not any(k in name for k in ("spec_decode", "preempt", "kv_cache", "generation_tokens", "prompt_tokens",
                                           "num_requests")):
                continue
            row = {"name": name, "type": type(m).__name__, "labels": getattr(m, "labels", None)}
            for f in ("value", "values", "count", "sum"):
                if hasattr(m, f): row[f] = getattr(m, f)
            snap.append(row)
        return snap
    except Exception as e:
        return [{"error": repr(e)[:300]}]
'''

C6 = r'''# ==================================================================== CELL 6 — R6 scheduler: think -> code -> run on train -> feedback -> repair (vLLM step API)
from collections import deque
RECORDS_PATH = OUT / "records.jsonl"
STATE = {}          # tid -> dict
FINAL = {}          # tid -> attempts (list per test query)
RECORDS = []
LOG_EVERY_SEC = 120.0
GEN_TOKENS_TOTAL = 0
SCHED = {"started": time.perf_counter(), "admitted": 0, "turns_started": 0, "repair_turns": 0, "phase2_continue": 0,
         "phase2_forced": 0, "drained": 0, "aborted_at_hard_stop": 0, "not_admitted": 0, "max_active": 0, "steps": 0}
READY = deque()     # tids whose next (repair) turn is prepared and waits for an engine slot
STOP_REPAIRS = [False]

def write_submission():
    sub = identity_submission(test_challenges)
    if not RERUN:
        write_json_atomic(SUBMISSION_PATH, sub); return
    for tid, att in FINAL.items():
        if tid in sub and len(att) == len(sub[tid]):
            sub[tid] = [{"attempt_1": a["attempt_1"], "attempt_2": a["attempt_2"]} for a in att]
    write_json_atomic(SUBMISSION_PATH, sub)

def _req(tid, turn, phase):
    return f"{tid}|t{turn}p{phase}"

def _parse_req(rid):
    tid, tag = rid.rsplit("|", 1)
    m = re.fullmatch(r"t(\d+)p(\d)", tag)
    return tid, int(m.group(1)), int(m.group(2))

def submit_turn(st):
    st.update({"status": "engine", "phase": 1, "p1_ids": [], "p2_ids": [], "p2_prefix": [], "p1_finish": None,
               "p2_finish": None, "mode": None, "t_turn": time.perf_counter()})
    engine.add_request(_req(st["tid"], st["turn"], 1), {"prompt_token_ids": list(st["prompt_ids"])},
                       make_params(st["cap"], SEED + 100 * UNIT_INDEX[st["tid"]] + 10 * st["turn"] + 1))
    SCHED["turns_started"] += 1
    if st["turn"] > 1:
        SCHED["repair_turns"] += 1

def admit(tid):
    st = {"tid": tid, "turn": 1, "kind": "code", "messages": list(MESSAGES0[tid]), "prompt_ids": PROMPT_IDS[tid],
          "cap": P1_CAPS[tid], "programs": [], "turns": [], "direct": None, "t_admit": time.perf_counter(), "t_done": None,
          "gen_tokens": 0}
    STATE[tid] = st
    submit_turn(st)
    SCHED["admitted"] += 1

def start_phase2(st, answer_cap):
    """think closed but answer cut -> continue; think never closed -> append stop text (token-exact) and answer"""
    closed = THINK_CLOSE_ID in st["p1_ids"]
    if not closed:                                  # v1.1 (F8): an EOS before </think> must not precede the stop text
        while st["p1_ids"] and st["p1_ids"][-1] in EOS_IDS:
            st["p1_ids"].pop()
    prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + ([] if closed else list(STOP_IDS))
    cap = min(int(answer_cap), MAX_MODEL_LEN - len(prefix) - 8)
    if cap < 64:
        return False
    st["phase"] = 2; st["p2_prefix"] = [] if closed else list(STOP_IDS); st["mode"] = "continue" if closed else "forced_close"
    SCHED["phase2_continue" if closed else "phase2_forced"] += 1
    engine.add_request(_req(st["tid"], st["turn"], 2), {"prompt_token_ids": prefix},
                       make_params(cap, SEED + 100 * UNIT_INDEX[st["tid"]] + 10 * st["turn"] + 2))
    return True

def answer_text(st):
    ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
    if THINK_CLOSE_ID not in ids:
        return "", len(ids), False
    k = len(ids) - 1 - ids[::-1].index(THINK_CLOSE_ID)
    return tokenizer.decode(ids[k + 1:], skip_special_tokens=True), k, True

VERIFY_POOL = cf.ThreadPoolExecutor(max_workers=VERIFY_WORKERS)
PENDING = {}        # future -> tid
ANSWERS_DIR = OUT / "answers"; ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

def end_generation(st, reason):
    """one turn's generation is over: keep the answer, extract the code, verify it off the engine thread"""
    text, think_tokens, closed = answer_text(st)
    code = extract_code(text)
    gen = len(st["p1_ids"]) + len(st["p2_ids"]); st["gen_tokens"] += gen
    st["cur"] = {"turn": st["turn"], "mode": st["mode"] or "natural", "end_reason": reason, "think_tokens": int(think_tokens),
                 "think_closed": closed, "gen_tokens": gen, "answer_chars": len(text), "code_found": code is not None,
                 "sec": round(time.perf_counter() - st["t_turn"], 1)}
    st["answer"] = text
    try:   # v1.1 (F5): keep every turn's answer and the end of its thinking for diagnosis
        _ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
        _end = think_tokens if closed else len(_ids)
        _tail = tokenizer.decode(_ids[max(0, _end - 1500):_end], skip_special_tokens=False)
        with open(ANSWERS_DIR / f"{st['tid']}.txt", "a", encoding="utf-8") as f:
            f.write("\n".join([f"### TURN {st['turn']} end_reason={reason} mode={st['cur']['mode']} think={think_tokens}",
                               "### THINKING TAIL", _tail, "### ANSWER", text, ""]))
    except Exception as e:
        print("answer save failed:", st["tid"], repr(e)[:200])
    if st.get("kind") == "grid":            # R6 v1.1: direct-grid fallback turn - nothing to execute
        st["direct"] = parse_grids(text, len(CHALLENGES[st["tid"]]["test"]))
        st["turns"].append(dict(st["cur"], kind="grid", grids_parsed=sum(g is not None for g in st["direct"]), verified=False))
        return finalize(st, "unverified_grid")
    st["status"] = "verifying"
    PENDING[VERIFY_POOL.submit(verify_program, st["tid"], code)] = st["tid"]

def on_verified(tid, res):
    st = STATE[tid]
    res = dict(res, turn=st["turn"])
    st["programs"].append(res)
    turn = dict(st["cur"], train_correct=res["train_correct"], n_train=res["n_train"], cells_right=res["cells_right"],
                test_ok=res["test_ok"], verified=res["verified"], error=(res["error"] or "")[:300])
    st["turns"].append(turn)
    try:
        with open(ANSWERS_DIR / f"{tid}.txt", "a", encoding="utf-8") as f:
            f.write(f"### VERIFY turn {st['turn']}: train {res['train_correct']}/{res['n_train']} cells_right "
                    f"{res['cells_right']} test_ok {res['test_ok']} error {res['error']}\n\n")
    except Exception:
        pass
    now = time.perf_counter()
    if res["verified"]:
        return finalize(st, "verified")
    if STOP_REPAIRS[0] or now >= DRAIN_DEADLINE:
        return finalize(st, "unverified")
    if st["turn"] >= MAX_TURNS:
        return start_grid_turn(st, res)
    msgs = st["messages"] + [{"role": "assistant", "content": st["answer"] or "(no answer)"},
                             {"role": "user", "content": feedback_text(tid, res)}]
    ids = tokenizer.encode(render_conversation(msgs, REPAIR_EFFORT), add_special_tokens=False)
    cap = max(0, min(REPAIR_THINK_BUDGET, MAX_MODEL_LEN - len(ids) - ANSWER_MAX_TOKENS - CONTEXT_SAFETY))
    if cap < 1024:
        return start_grid_turn(st, res)
    st.update({"messages": msgs, "prompt_ids": ids, "cap": cap, "turn": st["turn"] + 1, "status": "ready"})
    READY.append(tid)

GRID_REQUEST = ("Skip the code now. Work out the final output grid for each test input directly from the rule. Reply with "
                "one ```grid block per test input, in order (Test INPUT 1 first), each grid as rows of space-separated digits, "
                "and nothing after the last block.")

def start_grid_turn(st, res):
    """R6 v1.1: no program reproduced every training pair -> one last turn asks for the grid(s) directly"""
    if not GRID_FALLBACK:
        return finalize(st, "unverified")
    tid = st["tid"]
    fb = feedback_text(tid, res).rsplit("\n", 1)[0] if res["code_found"] else "Your reply did not contain a usable solve()."
    msgs = st["messages"] + [{"role": "assistant", "content": st["answer"] or "(no answer)"},
                             {"role": "user", "content": fb + "\n\n" + GRID_REQUEST}]
    ids = tokenizer.encode(render_conversation(msgs, REPAIR_EFFORT), add_special_tokens=False)
    cap = max(0, min(GRID_THINK_BUDGET, MAX_MODEL_LEN - len(ids) - ANSWER_MAX_TOKENS - CONTEXT_SAFETY))
    if cap < 1024:
        return finalize(st, "unverified_no_room")
    st.update({"messages": msgs, "prompt_ids": ids, "cap": cap, "turn": st["turn"] + 1, "kind": "grid", "status": "ready"})
    READY.append(tid)

def finalize(st, status):
    tid = st["tid"]
    st["status"] = "done"; st["final_status"] = status; st["t_done"] = time.perf_counter()
    att = make_attempts(tid, st["programs"], st.get("direct"))
    FINAL[tid] = att
    sc = score_task(tid, att)
    vt = next((t["turn"] for t in st["turns"] if t["verified"]), None)
    rec = {"task_id": tid, "status": status, "n_turns": len(st["turns"]), "verified_turn": vt,
           "gen_tokens": st["gen_tokens"], "prompt_tokens": len(PROMPT_IDS[tid]), "sec": round(st["t_done"] - st["t_admit"], 1),
           "turns": st["turns"], "src": [(a["src_1"], a["src_2"]) for a in att], "score": sc}
    RECORDS.append(rec)
    with open(RECORDS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    _tr = " ".join(f"t{t.get('turn')}:{str(t.get('mode', '?'))[:6]}/{t.get('train_correct', '-')}of{t.get('n_train', '-')}"
                   for t in st["turns"])
    print(f"[{(time.perf_counter() - NOTEBOOK_T0) / 60:6.1f} min] {tid} {status:18s} gen {st['gen_tokens']:6d} | {_tr} | "
          f"src={att[0]['src_1']}" + (f" hits={sc['query_hits']}" if sc else ""), flush=True)
    write_submission()

def collect_verifications(block=False):
    deadline = VERIFY_DEADLINE
    while True:
        done = [f for f in list(PENDING) if f.done()]
        for fut in done:
            tid = PENDING.pop(fut)
            try:
                res = fut.result()
            except Exception as e:
                res = verify_program(tid, None); res["error"] = "verify crashed: " + repr(e)[:200]
            on_verified(tid, res)
        if not block or not PENDING or time.perf_counter() >= deadline:
            break
        cf.wait(list(PENDING), timeout=max(0.1, min(5.0, deadline - time.perf_counter())), return_when=cf.FIRST_COMPLETED)
    if block:   # verification cut by the deadline: keep what earlier turns produced
        for fut in list(PENDING):
            tid = PENDING.pop(fut); fut.cancel()
            st = STATE[tid]; st["turns"].append(dict(st["cur"], verified=False, error="verification cut by deadline"))
            finalize(st, "verify_cut")

def run_scheduler():
    global GEN_TOKENS_TOTAL
    queue = [t for t in UNITS if t in PROMPT_IDS]
    last_log = time.perf_counter(); drained = False
    while True:
        now = time.perf_counter()
        active = [s for s in STATE.values() if s.get("status") == "engine"]
        if now >= HARD_STOP:
            ids = [_req(s["tid"], s["turn"], s["phase"]) for s in active]
            if ids: engine.abort_request(ids)
            STOP_REPAIRS[0] = True
            for s in active:
                SCHED["aborted_at_hard_stop"] += 1; s["turns"].append({"turn": s["turn"], "mode": "hard_stop", "verified": False})
                finalize(s, "hard_stop")
            break
        if now >= DRAIN_DEADLINE and not drained:
            drained = True; STOP_REPAIRS[0] = True
            p1 = [s for s in active if s["phase"] == 1]
            if p1: engine.abort_request([_req(s["tid"], s["turn"], 1) for s in p1])
            for s in p1:
                SCHED["drained"] += 1; s["p1_finish"] = "drained"
                if not start_phase2(s, DRAIN_ANSWER_MAX_TOKENS): end_generation(s, "drain_no_room")
            while READY:                                   # prepared repair turns will not run any more
                finalize(STATE[READY.popleft()], "unverified_drain")
            active = [s for s in STATE.values() if s.get("status") == "engine"]
        collect_verifications()
        while len(active) < CONCURRENCY and not drained:   # repair turns first, then new tasks
            if READY:
                st = STATE[READY.popleft()]; submit_turn(st)
            elif queue and now < ADMIT_DEADLINE:
                admit(queue.pop(0))
            else:
                break
            active = [s for s in STATE.values() if s.get("status") == "engine"]
        SCHED["max_active"] = max(SCHED["max_active"], len(active))
        if not active:
            if PENDING:
                cf.wait(list(PENDING), timeout=0.5, return_when=cf.FIRST_COMPLETED); continue
            if READY and not drained:
                continue
            if not queue or now >= ADMIT_DEADLINE or drained:
                break
            continue
        outs = engine.step(); SCHED["steps"] += 1
        for o in outs:
            tid, turn, phase = _parse_req(o.request_id); st = STATE[tid]
            if st.get("status") != "engine" or st["turn"] != turn or st["phase"] != phase: continue
            comp = o.outputs[0] if o.outputs else None
            if comp is not None:
                new = list(comp.token_ids); GEN_TOKENS_TOTAL += len(new)
                (st["p1_ids"] if phase == 1 else st["p2_ids"]).extend(new)
            if o.finished:
                fr = comp.finish_reason if comp is not None else None
                if phase == 1:
                    st["p1_finish"] = fr
                    if THINK_CLOSE_ID in st["p1_ids"] and fr != "length":
                        end_generation(st, "natural")
                    elif not start_phase2(st, ANSWER_MAX_TOKENS):
                        end_generation(st, "no_room_for_answer")
                else:
                    st["p2_finish"] = fr; end_generation(st, st["mode"])
        if time.perf_counter() - last_log > LOG_EVERY_SEC:
            last_log = time.perf_counter(); el = last_log - SCHED["started"]
            print(f"  .. {el / 60:6.1f} min | engine {len(active)} | verifying {len(PENDING)} | ready {len(READY)} | "
                  f"queued {len(queue)} | done {len(FINAL)} | gen {GEN_TOKENS_TOTAL} tok | "
                  f"{GEN_TOKENS_TOTAL / max(el, 1e-6):.0f} tok/s", flush=True)
            METRICS_SNAPSHOTS.append({"t_min": round(el / 60, 1), "active": len(active), "metrics": metrics_snapshot()})
    STOP_REPAIRS[0] = True
    SCHED["not_admitted"] = len(queue)
    collect_verifications(block=True)
    for st in list(STATE.values()):                       # anything still waiting for a repair slot
        if st.get("status") in ("ready", "verifying", "engine"):
            finalize(st, "unfinished_at_end")
    SCHED["wall_sec"] = round(time.perf_counter() - SCHED["started"], 1)
    SCHED["gen_tokens"] = GEN_TOKENS_TOTAL
    SCHED["tok_per_sec"] = round(GEN_TOKENS_TOTAL / max(SCHED["wall_sec"], 1e-6), 1)

# v1.1 (F2): a runtime error in an ~11 h loop must not end the notebook with an exception (that scores zero even
# with a valid submission.json); keep every finished task, write the submission, report the error.
SCHED_ERROR = None
METRICS_SNAPSHOTS = []
try:
    if not ENGINE_READY:     # v1.3: engine never started - keep the identity submission, still write the report
        SCHED["skipped"] = "engine_not_ready"; SCHED["not_admitted"] = len(PROMPT_IDS)
    else:
        run_scheduler()
except BaseException:
    SCHED_ERROR = traceback.format_exc()
    print("SCHEDULER ERROR - keeping finished tasks:", flush=True); print(SCHED_ERROR[-3000:], flush=True)
    STOP_REPAIRS[0] = True
    try:
        _act = [_req(s["tid"], s["turn"], s["phase"]) for s in STATE.values() if s.get("status") == "engine"]
        if _act: engine.abort_request(_act)
    except BaseException:
        pass
finally:
    try:
        collect_verifications(block=True)
        for _st in list(STATE.values()):    # tasks with programs from earlier turns still produce attempts
            if _st.get("status") in ("ready", "verifying", "engine"):
                finalize(_st, "unfinished_after_error")
    except BaseException:
        print("final collection failed:", traceback.format_exc()[-1500:])
    SCHED.setdefault("wall_sec", round(time.perf_counter() - SCHED["started"], 1))
    SCHED.setdefault("gen_tokens", GEN_TOKENS_TOTAL)
    SCHED.setdefault("tok_per_sec", round(GEN_TOKENS_TOTAL / max(SCHED["wall_sec"], 1e-6), 1))
    SCHED["error"] = SCHED_ERROR[-4000:] if SCHED_ERROR else None
    SCHED["setup_to_scheduler_sec"] = round(SCHED["started"] - NOTEBOOK_T0, 1)      # v1.3: real setup time
    METRICS_SNAPSHOTS.append({"t_min": "final", "metrics": metrics_snapshot()})
    try:
        write_json_atomic(OUT / "vllm_metrics.json", METRICS_SNAPSHOTS)
    except Exception:
        pass
    write_submission()
print("scheduler done:", json.dumps({k: v for k, v in SCHED.items() if k != "error"}), "| error:", bool(SCHED_ERROR))
'''

C7 = r'''# ==================================================================== CELL 7 — report, projection, bundle
# v1.1 (F2): the report must never turn a finished run into a notebook exception
try:
    rows = RECORDS
    def _mean(xs): return round(sum(xs) / len(xs), 1) if xs else None
    _turns = [t for r in rows for t in r["turns"]]
    report = {"run_id": RUN_ID, "profile": PROFILE, "rerun": RERUN, "config": CONFIG, "engine": ENGINE_INFO, "scheduler": SCHED,
              "tasks_planned": len(TASK_IDS), "tasks_finished": len(FINAL), "skipped_long_prompt": SKIPPED,
              "final_status": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
              "verified_tasks": sum(r["verified_turn"] is not None for r in rows),
              "verified_at_turn": {str(k): sum(r["verified_turn"] == k for r in rows) for k in range(1, MAX_TURNS + 1)},
              "mean_turns": _mean([r["n_turns"] for r in rows]),
              "turn_modes": {m: sum(t.get("mode") == m for t in _turns) for m in ("natural", "continue", "forced_close")},
              "code_found_turns": sum(bool(t.get("code_found")) for t in _turns), "turns_total": len(_turns),
              "mean_gen_tokens_per_task": _mean([r["gen_tokens"] for r in rows]),
              "grid_fallback_turns": sum(t.get("kind") == "grid" for t in _turns),
              "first_attempt_source": {s: sum(r["src"][0][0] == s for r in rows) for s in ("verified_code", "unverified_code", "direct_grid", "test_input", "mirror")}}
    if SOLUTIONS is not None:
        # Kaggle metric per test query (either attempt exact) over ALL planned queries; unfinished tasks are misses.
        _hits = [h for r in rows if r["score"] for h in r["score"]["query_hits"]]
        _planned_q = sum(len(SOLUTIONS[t]) for t in TASK_IDS)
        report["eval_query_hits"] = int(sum(_hits)); report["eval_query_total"] = int(_planned_q)
        report["eval_score_pct"] = round(100 * sum(_hits) / max(_planned_q, 1), 2)
        report["eval_task_normalized_pct"] = round(100 * sum(r["score"]["score"] for r in rows if r["score"]) / max(len(TASK_IDS), 1), 2)
        report["eval_pass1_queries"] = sum(sum(r["score"]["pass1"]) for r in rows if r["score"])
        report["eval_hit_tasks"] = [r["task_id"] for r in rows if r["score"] and sum(r["score"]["query_hits"])]
        _by_src = {}
        for r in rows:
            if r["score"]:
                for (s1, s2), h, p1 in zip(r["src"], r["score"]["query_hits"], r["score"]["pass1"]):
                    if h: _by_src[s1 if p1 else s2] = _by_src.get(s1 if p1 else s2, 0) + 1
        report["eval_hits_by_source"] = _by_src
        report["verified_but_wrong_tasks"] = [r["task_id"] for r in rows if r["verified_turn"] and r["score"]
                                              and sum(r["score"]["pass1"]) < len(r["score"]["pass1"])]
    _gen = sorted(r["gen_tokens"] for r in rows)
    report["gen_tokens_p50_p90_max"] = [_gen[len(_gen) // 2] if _gen else None, _gen[min(len(_gen) - 1, int(0.9 * len(_gen)))] if _gen else None,
                                        _gen[-1] if _gen else None]
    try:   # v1.3: MTP acceptance from the final vLLM metrics snapshot
        _fin = METRICS_SNAPSHOTS[-1]["metrics"] if METRICS_SNAPSHOTS else []
        _val = lambda key: sum(float(m.get("value") or 0) for m in _fin if key in m.get("name", "") and "per_pos" not in m.get("name", ""))
        _drafted, _accepted = _val("spec_decode_num_draft_tokens"), _val("spec_decode_num_accepted_tokens")
        report["mtp_draft_tokens"], report["mtp_accepted_tokens"] = _drafted, _accepted
        report["mtp_acceptance_rate"] = round(_accepted / _drafted, 3) if _drafted else None
        report["num_preemptions"] = _val("num_preemptions")
    except Exception as _e:
        report["metrics_error"] = repr(_e)[:200]
    # a projection is only valid from a complete, error-free commit run (an incomplete run is biased to easy tasks)
    report["projection_valid"] = bool(ENGINE_READY and len(FINAL) == len(TASK_IDS) and not SCHED.get("error") and not SKIPPED
                                      and SCHED.get("not_admitted", 0) == 0 and SCHED.get("aborted_at_hard_stop", 0) == 0
                                      and SCHED.get("drained", 0) == 0 and not any(r["status"] == "verify_cut" for r in rows))
    report["projection_hidden_tasks"] = N_HIDDEN_TASKS
    report["projection_hidden_generation_hours"] = report["projection_hidden_end_to_end_hours"] = None
    if not report["projection_valid"]:
        report["projection_reason"] = "incomplete commit run, engine failure, scheduler error, skipped / unadmitted / drained / hard-stopped / verify-cut tasks"
    if report["projection_valid"] and SCHED.get("tok_per_sec") and report["mean_gen_tokens_per_task"]:
        _rate = SCHED["tok_per_sec"]
        _gen_h = N_HIDDEN_TASKS * report["mean_gen_tokens_per_task"] / _rate / 3600
        _setup_h = (SCHED.get("setup_to_scheduler_sec") or 0) / 3600 + 0.05
        report["projection_hidden_generation_hours"] = round(_gen_h, 2)
        report["projection_hidden_end_to_end_hours"] = round(_gen_h + _setup_h, 2)
        report["projection_note"] = ("overall rate incl. the tail of a small run (few active tasks at the end) - pessimistic "
                                     "for the full hidden set; up to %d concurrent of %d slots" % (SCHED["max_active"], CONCURRENCY))
    report["deadlines_min"] = {k: round((v - NOTEBOOK_T0) / 60, 1) for k, v in
                               {"admit": ADMIT_DEADLINE, "drain": DRAIN_DEADLINE, "hard_stop": HARD_STOP, "verify": VERIFY_DEADLINE}.items()}
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
              f"{report['eval_score_pct']}% | task-normalised {report['eval_task_normalized_pct']}% | train-verified tasks "
              f"{report['verified_tasks']}/{len(TASK_IDS)} at turns {report['verified_at_turn']} | throughput {SCHED['tok_per_sec']} tok/s | "
              f"projected hidden ({N_HIDDEN_TASKS} tasks) end-to-end hours {report.get('projection_hidden_end_to_end_hours')}")
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
