# --- NOTEBOOK SOURCE CELL 1 ---
# ==================================================================== CELL 0 — mode + fallback submission (stdlib only)
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

# R5: MODE CONTRACT. A non-rerun commit intentionally contains an identity safety floor.
MODE_CONTRACT = {"competition_rerun_env": os.environ.get("KAGGLE_IS_COMPETITION_RERUN"),
                 "rerun": bool(RERUN), "submission_file_semantics":
                 "MODEL_PREDICTIONS_WHEN_FINISHED" if RERUN else "IDENTITY_FLOOR_NOT_MODEL_OUTPUT"}
write_json_atomic(OUT / "mode_contract.json", MODE_CONTRACT)
write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))
print("SUBMISSION CONTRACT:", MODE_CONTRACT, flush=True)
print("MODE:", "COMPETITION RERUN (hidden test)" if RERUN else "COMMIT / INTERACTIVE (evaluation)",
      "| fallback submission.json:", len(test_challenges), "tasks | RUN_ID", RUN_ID)


# --- NOTEBOOK SOURCE CELL 2 ---
# ==================================================================== CELL 1 — configuration
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
# R5: Optional low-effort reviewer; NOT enabled for real hidden runs until benchmarked.
# It runs only AFTER first-pass coverage, and stops strictly before HARD_STOP.
AUDIT_ENABLED = os.environ.get("Q27X_ENABLE_AUDIT", "0") == "1"
AUDIT_MAX_QUERIES = max(0, int(os.environ.get("Q27X_AUDIT_MAX_QUERIES", "8")))
AUDIT_THINK_BUDGET = max(128, int(os.environ.get("Q27X_AUDIT_THINK_BUDGET", "4096")))
AUDIT_ANSWER_MAX = max(128, int(os.environ.get("Q27X_AUDIT_ANSWER_MAX", "4096")))
AUDIT_CONCURRENCY = max(1, int(os.environ.get("Q27X_AUDIT_CONCURRENCY", "2")))

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
          "TOTAL_BUDGET_SEC", "MIN_THINK_BUDGET", "AUDIT_ENABLED", "AUDIT_MAX_QUERIES", "AUDIT_THINK_BUDGET", "AUDIT_ANSWER_MAX", "AUDIT_CONCURRENCY"]}
CONFIG["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"]
CONFIG["profile_row"] = P
print(json.dumps(CONFIG, indent=1, default=str))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"


# --- NOTEBOOK SOURCE CELL 3 ---
# ==================================================================== CELL 2 — offline install (PIV/V3.1 parity block) + tokenizer
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


# --- NOTEBOOK SOURCE CELL 4 ---
# ==================================================================== CELL 3 — tasks + prompts (one prompt per test input)
if P["source"] == "eval":
    _ch = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
    _sol = _load_json(ARC_ROOT / "arc-agi_evaluation_solutions.json")
    TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))
    CHALLENGES = {t: _ch[t] for t in TASK_IDS}
    SOLUTIONS = {t: _sol[t] for t in TASK_IDS}
else:
    CHALLENGES = dict(test_challenges); TASK_IDS = sorted(CHALLENGES); SOLUTIONS = None

# v1.4: russcore's LB-10.14 free-form prompt, verbatim (one test input per prompt, rows of space-separated digits,
# grid-only fenced answer)
def render_grid(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)

def build_user_message(train, test_input):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(train):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i + 1, render_grid(p["input"]), i + 1, render_grid(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render_grid(test_input))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)

def render_prompt(train, test_input):
    msgs = [{"role": "user", "content": build_user_message(train, test_input)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True, reasoning_effort=REASONING_EFFORT)

# a unit = one test input of one task; request ids, seeds, records and the scheduler all work on units
UNITS, UNIT_TASK, UNIT_Q = [], {}, {}
for _tid in TASK_IDS:
    for _qi in range(len(CHALLENGES[_tid]["test"])):
        _uid = f"{_tid}#{_qi}"
        UNITS.append(_uid); UNIT_TASK[_uid] = _tid; UNIT_Q[_uid] = _qi
UNIT_INDEX = {u: i for i, u in enumerate(UNITS)}             # v1.1 (F13): seeds do not depend on timing
N_HIDDEN_UNITS = sum(len(t["test"]) for t in test_challenges.values())   # 259 on Kaggle; the projection is per query

PROMPTS, PROMPT_IDS, SKIPPED, P1_CAPS = {}, {}, {}, {}
for uid in UNITS:
    _task = CHALLENGES[UNIT_TASK[uid]]
    text = render_prompt(_task["train"], _task["test"][UNIT_Q[uid]]["input"])
    ids = tokenizer.encode(text, add_special_tokens=False)
    cap = phase1_cap(len(ids))
    if cap < min(MIN_THINK_BUDGET, THINK_BUDGET):  # v1.2: only prompts that leave < 4k thinking tokens are skipped
        SKIPPED[uid] = {"prompt_tokens": len(ids), "reason": "insufficient_context"}; continue
    PROMPTS[uid] = text; PROMPT_IDS[uid] = ids; P1_CAPS[uid] = cap
_pl = sorted(len(v) for v in PROMPT_IDS.values())
print(f"{PROFILE}: {len(TASK_IDS)} tasks, {len(UNITS)} queries (one prompt each; hidden test has {N_HIDDEN_UNITS}) | "
      f"prompt tokens median {_pl[len(_pl)//2] if _pl else 0}, max {_pl[-1] if _pl else 0} | reduced thinking budget: "
      f"{sum(c < THINK_BUDGET for c in P1_CAPS.values())} | skipped (too long): {len(SKIPPED)}")
_m0 = build_user_message(CHALLENGES[TASK_IDS[0]]["train"], CHALLENGES[TASK_IDS[0]]["test"][0]["input"])
assert "xhigh" not in _m0 and "FINAL INPUT:" in _m0 and _m0.endswith("Output nothing after the block.")
assert render_grid([[1, 0], [2, 3]]) == "1 0\n2 3"


# --- NOTEBOOK SOURCE CELL 5 ---
# ==================================================================== CELL 4 — grid parsing, attempts, scoring
def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
_SEP_ROW = re.compile(r"\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?")      # "1 0 2", "1,0,2", "[1, 0, 2]," (one digit per cell)
_JSON_ROW = re.compile(r"\[\s*\d(?:\s*,\s*\d)*\s*\]")

def _parse_row(s):
    s = s.strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):                                  # contiguous digits, one per cell ("102")
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
    if isinstance(obj, list) and obj and valid_grid(obj[-1]):     # a list of grids: the last one
        return obj[-1]
    return None

def _json_rows_grid(b):
    """v1.4: JSON with broken brackets (run 3: 6/12 blocks lacked one '['): rows are read one by one; a row followed by
    ']' ends a grid; the last grid wins"""
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
    """answer text after </think> -> grid or None: the last fenced block (any label) that holds a valid grid; else the
    last run of digit rows in the text (unfenced answer or a fence that was never closed)"""
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

def make_attempts(uid, grid):
    """attempt_1 = the model's grid; attempt_2 = the test input, or its mirror when the model returned the input"""
    inp = CHALLENGES[UNIT_TASK[uid]]["test"][UNIT_Q[uid]]["input"]
    cands = ([("model", grid)] if valid_grid(grid) else []) + [("test_input", inp), ("mirror", [list(reversed(r)) for r in inp])]
    uniq = []
    for src, g in cands:
        if all(g != u[1] for u in uniq):
            uniq.append((src, g))
    while len(uniq) < 2:                     # a palindromic input: mirror == input
        uniq.append(uniq[-1])
    return {"attempt_1": uniq[0][1], "attempt_2": uniq[1][1], "src_1": uniq[0][0], "src_2": uniq[1][0]}

def score_unit(uid, att):
    if SOLUTIONS is None:
        return None
    sol = SOLUTIONS[UNIT_TASK[uid]][UNIT_Q[uid]]
    return {"hit": int(att["attempt_1"] == sol or att["attempt_2"] == sol), "pass1": int(att["attempt_1"] == sol)}

# R5: Keep malformed final-grid evidence. Diagnostic only: never guess which cell to delete.
def raw_grid_diagnostics(answer_text):
    blocks = _FENCE_RE.findall(answer_text or "")
    if blocks:
        label, body = blocks[-1]
        source = "final_closed_fence"
    else:
        text = answer_text or ""
        body = text.rsplit("```", 1)[-1] if "```" in text else text
        label, source = "", "unfenced_or_open"
    raw_lines = [line.strip() for line in body.splitlines() if line.strip()]
    rows = [_parse_row(line) for line in raw_lines]
    lengths = [len(row) for row in rows if row is not None]
    counts = {w: lengths.count(w) for w in sorted(set(lengths))}
    return {"source": source, "fence_label": label.strip(), "parseable_rows": len(lengths),
            "row_widths": counts, "ragged": len(counts) > 1,
            "too_tall": len(lengths) > 30, "too_wide": bool(lengths and max(lengths) > 30),
            "unparsed_nonempty_lines": len(raw_lines) - len(lengths)}


def merge_second_candidate(uid, grid, source="model_review"):
    """Promote a *new* legal candidate without changing attempt_1 or a useful prior fallback."""
    if uid not in FINAL or not valid_grid(grid):
        return False
    prev = FINAL[uid]
    if grid in (prev["attempt_1"], prev["attempt_2"]):
        return False
    prev["attempt_2"] = grid
    prev["src_2"] = source
    return True

# ---- self-tests (CPU, every run)
_G = [[1, 0, 2], [3, 4, 5]]
_layouts = {
    "grid_fence":          "Done.\n```grid\n1 0 2\n3 4 5\n```",
    "no_label":            "```\n1 0 2\n3 4 5\n```",
    "text_contiguous":     "```text\n102\n345\n```",
    "comma_rows":          "```grid\n1,0,2\n3,4,5\n```",
    "header_in_fence":     "```grid\nOutput:\n1 0 2\n3 4 5\n```",
    "json_list":           "```json\n[[1, 0, 2], [3, 4, 5]]\n```",
    "json_object":         "```json\n{\"test_outputs\": [[[1, 0, 2], [3, 4, 5]]]}\n```",
    "json_missing_open":   "```json\n{\"test_outputs\": [[1, 0, 2], [3, 4, 5]]]}\n```",
    "last_fence_not_grid": "```grid\n1 0 2\n3 4 5\n```\n```python\nprint('x')\n```",
    "earlier_ragged":      "```grid\n9 9\n9\n```\nfixed:\n```grid\n1 0 2\n3 4 5\n```",
    "unfenced":            "The final grid is:\n1 0 2\n3 4 5\n",
    "unclosed_fence":      "```grid\n1 0 2\n3 4 5\n",
}
for _name, _txt in _layouts.items():
    assert parse_grid(_txt) == _G, (_name, parse_grid(_txt))
assert parse_grid("no grid here") is None and parse_grid("") is None and parse_grid(None) is None
assert parse_grid("```grid\n1 0\n3 4 5\n```") is None                  # ragged
assert parse_grid("```grid\n1 10 2\n```") is None                      # a two-digit cell is not a colour
CHALLENGES["__selftest__"] = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[6, 7]]}]}
UNIT_TASK["__selftest__#0"] = "__selftest__"; UNIT_Q["__selftest__#0"] = 0
_a = make_attempts("__selftest__#0", [[7, 6]])
assert (_a["attempt_1"], _a["src_1"], _a["attempt_2"], _a["src_2"]) == ([[7, 6]], "model", [[6, 7]], "test_input"), _a
_a = make_attempts("__selftest__#0", None)
assert (_a["attempt_1"], _a["src_1"], _a["attempt_2"], _a["src_2"]) == ([[6, 7]], "test_input", [[7, 6]], "mirror"), _a
_a = make_attempts("__selftest__#0", [[6, 7]])                         # model returned the input -> mirror second
assert (_a["src_1"], _a["src_2"], _a["attempt_2"]) == ("model", "mirror", [[7, 6]]), _a
del CHALLENGES["__selftest__"], UNIT_TASK["__selftest__#0"], UNIT_Q["__selftest__#0"]
print(f"cell 4 self-tests PASS ({len(_layouts)} answer layouts + ragged / two-digit / empty / attempts)")


# --- NOTEBOOK SOURCE CELL 6 ---
# ==================================================================== CELL 5 — engine (PIV/V3.1 C1 settings + long context + fp8 KV + MTP)
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
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
import vllm
try:   # R5: attach AFTER vLLM imports; same logger hierarchy as vLLM uses.
    _vllm_log_handler = logging.FileHandler(OUT / "vllm_frontend.log", mode="w", encoding="utf-8")
    _vllm_log_handler.setLevel(logging.INFO)
    _vllm_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("vllm").addHandler(_vllm_log_handler)
except Exception as _e:
    print("vLLM log file handler not installed:", repr(_e)[:200])
# R5 log probe: an empty log file was observed in RUN4.
logging.getLogger("vllm").info("Q27X_LOG_PROBE_FRONTEND")
try:
    _vllm_log_handler.flush()
except NameError:
    pass

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


# --- NOTEBOOK SOURCE CELL 7 ---
# ==================================================================== CELL 6 — continuous xhigh scheduler (vLLM step API), one unit per test input
RECORDS_PATH = OUT / "records.jsonl"
STATE = {}          # uid -> dict
FINAL = {}          # uid -> attempts for that test input
RECORDS = []
LOG_EVERY_SEC = 120.0
GEN_TOKENS_TOTAL = 0
SCHED = {"started": time.perf_counter(), "admitted": 0, "phase2_continue": 0, "phase2_forced": 0, "drained": 0,
         "aborted_at_hard_stop": 0, "not_admitted": 0, "max_active": 0, "steps": 0}

def write_submission():
    sub = identity_submission(test_challenges)
    if not RERUN:
        # R5: save real EVAL12 candidates separately. The committed submission remains identity floor.
        ev = {tid: [{"attempt_1": FINAL.get(f"{tid}#{qi}", {}).get("attempt_1", q["input"]),
                     "attempt_2": FINAL.get(f"{tid}#{qi}", {}).get("attempt_2", q["input"])}
                    for qi, q in enumerate(CHALLENGES[tid]["test"])] for tid in TASK_IDS}
        write_json_atomic(OUT / "eval_predictions.json", ev)
        write_json_atomic(SUBMISSION_PATH, sub)
        return
    for uid, att in FINAL.items():
        tid, qi = UNIT_TASK[uid], UNIT_Q[uid]
        if tid in sub and qi < len(sub[tid]):
            sub[tid][qi] = {"attempt_1": att["attempt_1"], "attempt_2": att["attempt_2"]}
    write_json_atomic(SUBMISSION_PATH, sub)

def _req(uid, phase):
    return f"{uid}|{phase}"

def admit(uid):
    st = {"uid": uid, "phase": 1, "prompt_ids": PROMPT_IDS[uid], "p1_ids": [], "p2_ids": [], "p2_prefix": [],
          "p1_finish": None, "p2_finish": None, "t_admit": time.perf_counter(), "t_done": None, "mode": None}
    STATE[uid] = st
    engine.add_request(_req(uid, 1), {"prompt_token_ids": list(st["prompt_ids"])},
                       make_params(P1_CAPS[uid], SEED + 10 * UNIT_INDEX[uid] + 1))
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
    engine.add_request(_req(st["uid"], 2), {"prompt_token_ids": prefix}, make_params(cap, SEED + 10 * UNIT_INDEX[st["uid"]] + 2))
    return True

def answer_text(st):
    ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
    if THINK_CLOSE_ID not in ids:
        return "", 0, False
    k = len(ids) - 1 - ids[::-1].index(THINK_CLOSE_ID)
    return tokenizer.decode(ids[k + 1:], skip_special_tokens=True), k, True

ANSWERS_DIR = OUT / "answers"; ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

def finish(st, reason):
    uid = st["uid"]
    st["phase"] = "done"; st["t_done"] = time.perf_counter(); st["end_reason"] = reason
    text, think_tokens, closed = answer_text(st)
    grid = parse_grid(text)
    st.update({"answer_chars": len(text), "think_tokens": int(think_tokens if closed else len(st["p1_ids"])),
               "think_closed": closed, "has_grid": grid is not None,
               "grid_diagnostic": raw_grid_diagnostics(text)})
    try:   # v1.1 (F5): keep the answer and the end of the thinking for diagnosis
        _ids = list(st["p1_ids"]) + list(st["p2_prefix"]) + list(st["p2_ids"])
        _end = think_tokens if closed else len(_ids)
        _tail = tokenizer.decode(_ids[max(0, _end - 1500):_end], skip_special_tokens=False)
        (ANSWERS_DIR / f"{UNIT_TASK[uid]}_q{UNIT_Q[uid]}.txt").write_text(
            "\n".join([f"### end_reason={reason} mode={st['mode']}", "### THINKING TAIL", _tail, "### ANSWER", text, ""]),
            encoding="utf-8")
    except Exception as e:
        print("answer save failed:", uid, repr(e)[:200])
    _record(uid, st, make_attempts(uid, grid))

def _record(uid, st, att):
    FINAL[uid] = att
    sc = score_unit(uid, att)
    rec = {"unit": uid, "task_id": UNIT_TASK[uid], "query": UNIT_Q[uid], "mode": st["mode"] or "natural",
           "end_reason": st["end_reason"], "prompt_tokens": len(st["prompt_ids"]), "p1_tokens": len(st["p1_ids"]),
           "p2_tokens": len(st["p2_ids"]), "think_tokens": st["think_tokens"], "think_closed": st["think_closed"],
           "p1_finish": st["p1_finish"], "p2_finish": st["p2_finish"], "sec": round(st["t_done"] - st["t_admit"], 1),
           "answer_chars": st["answer_chars"], "has_grid": st["has_grid"], "src": [att["src_1"], att["src_2"]],
           "score": sc, "p1_cap": P1_CAPS.get(uid), "grid_diagnostic": st.get("grid_diagnostic")}
    RECORDS.append(rec)
    with open(RECORDS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{(time.perf_counter() - NOTEBOOK_T0) / 60:6.1f} min] {uid:14s} {rec['mode']:12s} think {rec['think_tokens']:6d} "
          f"closed={rec['think_closed']} grid={rec['has_grid']} src={att['src_1']}" + (f" hit={sc['hit']}" if sc else ""),
          flush=True)
    write_submission()

def run_scheduler():
    global GEN_TOKENS_TOTAL
    queue = [u for u in UNITS if u in PROMPT_IDS]
    last_log = time.perf_counter(); drained = False
    while True:
        now = time.perf_counter()
        active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        if now >= HARD_STOP:
            ids = [_req(s["uid"], s["phase"]) for s in active]
            if ids: engine.abort_request(ids)
            for s in active:
                SCHED["aborted_at_hard_stop"] += 1; finish(s, "hard_stop")
            break
        if now >= DRAIN_DEADLINE and not drained:
            drained = True
            p1 = [s for s in active if s["phase"] == 1]
            if p1: engine.abort_request([_req(s["uid"], 1) for s in p1])
            for s in p1:
                SCHED["drained"] += 1; s["p1_finish"] = "drained"
                if not start_phase2(s, DRAIN_ANSWER_MAX_TOKENS): finish(s, "drain_no_room")
            active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        while queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        SCHED["max_active"] = max(SCHED["max_active"], len(active))
        if not active:
            if not queue or now >= ADMIT_DEADLINE or drained:
                break
            continue
        outs = engine.step(); SCHED["steps"] += 1
        for o in outs:
            uid, phase = o.request_id.rsplit("|", 1); phase = int(phase); st = STATE[uid]
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
        if time.perf_counter() - last_log > LOG_EVERY_SEC:
            last_log = time.perf_counter(); el = last_log - SCHED["started"]
            print(f"  .. {el / 60:6.1f} min | active {len(active)} | queued {len(queue)} | done {len(FINAL)} | "
                  f"gen {GEN_TOKENS_TOTAL} tok | {GEN_TOKENS_TOTAL / max(el, 1e-6):.0f} tok/s", flush=True)
            METRICS_SNAPSHOTS.append({"t_min": round(el / 60, 1), "active": len(active), "metrics": metrics_snapshot()})
    SCHED["not_admitted"] = len(queue)
    SCHED["wall_sec"] = round(time.perf_counter() - SCHED["started"], 1)
    SCHED["gen_tokens"] = GEN_TOKENS_TOTAL
    SCHED["tok_per_sec"] = round(GEN_TOKENS_TOTAL / max(SCHED["wall_sec"], 1e-6), 1)

# R5: truly distinct, optional second model attempt. Run only on remaining budget after coverage.
# This experiment has NO proven accuracy gain; it never replaces attempt_1.
AUDIT_STATS = {"enabled": AUDIT_ENABLED, "started": 0, "completed": 0,
               "valid_distinct": 0, "aborted_at_deadline": 0, "candidates_planned": 0}
AUDIT_RECORDS_PATH = OUT / "audit_records.jsonl"

def _review_prompt(uid):
    task = CHALLENGES[UNIT_TASK[uid]]
    test_input = task["test"][UNIT_Q[uid]]["input"]
    orig = FINAL[uid]
    text = build_user_message(task["train"], test_input)
    # Present the existing candidate as a tentative guess; ask for a corrected alternate.
    if orig["src_1"] == "model":
        text += ("\nThe earlier tentative output grid could contain an indexing or copying mistake. "
                 "Independently check it against EVERY example. Recompute the final grid "
                 "and return the corrected grid in one ```grid block. "
                 "Do not repeat an unverified guess.\nTENTATIVE GRID:\n```grid\n"
                 + render_grid(orig["attempt_1"]) + "\n```\n")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True, reasoning_effort="low")

def run_optional_audit():
    """Isolated second-pass queue; protects existing answers if review fails or times out."""
    if not AUDIT_ENABLED or not ENGINE_READY or AUDIT_MAX_QUERIES == 0:
        return
    # Model candidate requests first, then malformed-answer / fallback requests.
    queue = sorted((u for u in UNITS if u in FINAL),
                   key=lambda u: (FINAL[u]["src_1"] != "model", UNIT_INDEX[u]))[:AUDIT_MAX_QUERIES]
    AUDIT_STATS["candidates_planned"] = len(queue)
    state = {}
    limit = min(AUDIT_CONCURRENCY, CONCURRENCY)
    cutoff = HARD_STOP - 45  # leave safety window for output serialization

    def finish_audit(st, reason):
        uid = st["uid"]
        ids = st["p1_ids"] + st["p2_prefix"] + st["p2_ids"]
        closed = THINK_CLOSE_ID in ids
        if closed:
            k = len(ids) - 1 - ids[::-1].index(THINK_CLOSE_ID)
            text = tokenizer.decode(ids[k+1:], skip_special_tokens=True)
        else:
            text = ""
        grid = parse_grid(text)
        upgraded = merge_second_candidate(uid, grid)
        AUDIT_STATS["completed"] += 1
        AUDIT_STATS["valid_distinct"] += int(upgraded)
        rec = {"unit": uid, "mode": reason, "valid_grid": valid_grid(grid),
               "promoted": upgraded, "prompt_tokens": len(st["prompt_ids"]),
               "generation_tokens": len(st["p1_ids"]) + len(st["p2_ids"]),
               "diagnostic": raw_grid_diagnostics(text)}
        with open(AUDIT_RECORDS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        st["phase"] = "done"
        write_submission()

    while queue or any(s["phase"] in (3, 4) for s in state.values()):
        now = time.perf_counter()
        active = [s for s in state.values() if s["phase"] in (3, 4)]
        if now >= cutoff:
            if active:
                engine.abort_request([_req(s["uid"], s["phase"]) for s in active])
            AUDIT_STATS["aborted_at_deadline"] += len(active)
            break
        while queue and len(active) < limit and now < cutoff - 90:
            uid = queue.pop(0)
            prompt_ids = tokenizer.encode(_review_prompt(uid), add_special_tokens=False)
            cap = max(0, min(AUDIT_THINK_BUDGET, MAX_MODEL_LEN - len(prompt_ids) - AUDIT_ANSWER_MAX - CONTEXT_SAFETY))
            if cap < 128:
                continue
            st = {"uid": uid, "phase": 3, "prompt_ids": prompt_ids,
                  "p1_ids": [], "p2_ids": [], "p2_prefix": []}
            state[uid] = st
            engine.add_request(_req(uid, 3), {"prompt_token_ids": prompt_ids},
                               make_params(cap, SEED + 10 * UNIT_INDEX[uid] + 7))
            AUDIT_STATS["started"] += 1
            active.append(st)
        if not active:
            break
        for o in engine.step():
            uid, p = o.request_id.rsplit("|", 1)
            phase = int(p)
            st = state.get(uid)
            if st is None or st["phase"] != phase:
                continue
            comp = o.outputs[0] if o.outputs else None
            if comp is not None:
                (st["p1_ids"] if phase == 3 else st["p2_ids"]).extend(comp.token_ids)
            if not o.finished:
                continue
            if phase == 3:
                if THINK_CLOSE_ID in st["p1_ids"] and comp is not None and comp.finish_reason != "length":
                    finish_audit(st, "natural")
                    continue
                closed = THINK_CLOSE_ID in st["p1_ids"]
                if not closed:
                    while st["p1_ids"] and st["p1_ids"][-1] in EOS_IDS:
                        st["p1_ids"].pop()
                pre = list(st["prompt_ids"]) + list(st["p1_ids"]) + ([] if closed else list(STOP_IDS))
                answer_cap = min(AUDIT_ANSWER_MAX, MAX_MODEL_LEN - len(pre) - 8)
                if answer_cap < 64:
                    finish_audit(st, "no_room")
                    continue
                st["phase"] = 4
                st["p2_prefix"] = [] if closed else list(STOP_IDS)
                engine.add_request(_req(uid, 4), {"prompt_token_ids": pre},
                                   make_params(answer_cap, SEED + 10 * UNIT_INDEX[uid] + 8))
            else:
                finish_audit(st, "forced_or_continued")

# v1.1 (F2): a runtime error in an ~11 h loop must not end the notebook with an exception (that scores zero even
# with a valid submission.json); keep every finished query, write the submission, report the error.
SCHED_ERROR = None
METRICS_SNAPSHOTS = []
try:
    if not ENGINE_READY:     # v1.3: engine never started - keep the identity submission, still write the report
        SCHED["skipped"] = "engine_not_ready"; SCHED["not_admitted"] = len(PROMPT_IDS)
    else:
        run_scheduler()
        run_optional_audit()
        # R5: account for optional extra inference when summarizing elapsed time.
        SCHED["wall_sec_with_audit"] = round(time.perf_counter() - SCHED["started"], 1)
except BaseException:
    SCHED_ERROR = traceback.format_exc()
    print("SCHEDULER ERROR - keeping finished queries:", flush=True); print(SCHED_ERROR[-3000:], flush=True)
    try:
        _act = [_req(s["uid"], s["phase"]) for s in STATE.values() if s["phase"] in (1, 2)]
        if _act: engine.abort_request(_act)
    except BaseException:
        pass
finally:
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


# --- NOTEBOOK SOURCE CELL 8 ---
# ==================================================================== CELL 7 — report, projection, bundle
# v1.1 (F2): the report must never turn a finished run into a notebook exception
try:
    rows = RECORDS
    def _mean(xs): return round(sum(xs) / len(xs), 1) if xs else None
    report = {"run_id": RUN_ID, "profile": PROFILE, "rerun": RERUN, "config": CONFIG, "engine": ENGINE_INFO, "scheduler": SCHED,
              "tasks_planned": len(TASK_IDS), "queries_planned": len(UNITS), "queries_finished": len(FINAL),
              "tasks_finished": sum(all(f"{t}#{q}" in FINAL for q in range(len(CHALLENGES[t]["test"]))) for t in TASK_IDS),
              "skipped_long_prompt": SKIPPED,
              "modes": {m: sum(r["mode"] == m for r in rows) for m in ("natural", "continue", "forced_close")},
              "think_closed_naturally": sum(r["mode"] == "natural" for r in rows),
              "mean_think_tokens": _mean([r["think_tokens"] for r in rows]),
              "mean_total_gen_tokens": _mean([r["p1_tokens"] + r["p2_tokens"] for r in rows]),
              "with_grid": sum(bool(r["has_grid"]) for r in rows),
              "first_attempt_source": {s: sum(r["src"][0] == s for r in rows) for s in ("model", "test_input", "mirror")},
              "audit_stats": AUDIT_STATS, "submission_mode_contract": MODE_CONTRACT}
    if SOLUTIONS is not None:
        # Kaggle metric per test query (either attempt exact) over ALL planned queries; unfinished / skipped queries are
        # misses. The task-normalised figure (each task worth 1, split over its test inputs) is kept as a diagnostic.
        # R5: scoring uses merged final attempts (optional pass2), not stale first-pass RECORDS.
        _final_scored = [(uid, score_unit(uid, att)) for uid, att in FINAL.items()]
        _hits = [sc["hit"] for uid, sc in _final_scored if sc]
        report["eval_query_hits"] = int(sum(_hits)); report["eval_query_total"] = len(UNITS)
        report["eval_score_pct"] = round(100 * sum(_hits) / max(len(UNITS), 1), 2)
        _by = {}
        for uid, sc in _final_scored:
            if sc: _by.setdefault(UNIT_TASK[uid], []).append(sc["hit"])
        report["eval_task_normalized_pct"] = round(100 * sum(sum(v) / len(CHALLENGES[t]["test"]) for t, v in _by.items())
                                                   / max(len(TASK_IDS), 1), 2)
        report["eval_pass1_queries"] = sum(r["score"]["pass1"] for r in rows if r["score"])
        report["eval_hit_units"] = [uid for uid, sc in _final_scored if sc and sc["hit"]]
        report["eval_incremental_pass2_queries"] = sum(sc["hit"] and not sc["pass1"] for _, sc in _final_scored if sc)
    _live = sorted(r["prompt_tokens"] + r["p1_tokens"] + r["p2_tokens"] for r in rows)
    _pct = lambda q: _live[min(len(_live) - 1, int(q * len(_live)))] if _live else None
    report["live_tokens_p50_p75_p90_max"] = [_pct(0.5), _pct(0.75), _pct(0.9), _live[-1] if _live else None]
    _gen = sorted(r["p1_tokens"] + r["p2_tokens"] for r in rows)
    report["gen_tokens_p50_p90"] = [_gen[len(_gen) // 2] if _gen else None, _gen[min(len(_gen) - 1, int(0.9 * len(_gen)))] if _gen else None]
    try:   # v1.3: MTP acceptance from the final vLLM metrics snapshot
        _fin = METRICS_SNAPSHOTS[-1]["metrics"] if METRICS_SNAPSHOTS else []
        _val = lambda key: sum(float(m.get("value") or 0) for m in _fin if key in m.get("name", "") and "per_pos" not in m.get("name", ""))
        _drafted, _accepted = _val("spec_decode_num_draft_tokens"), _val("spec_decode_num_accepted_tokens")
        report["mtp_draft_tokens"], report["mtp_accepted_tokens"] = _drafted, _accepted
        report["mtp_acceptance_rate"] = round(_accepted / _drafted, 3) if _drafted else None
        report["num_preemptions"] = _val("num_preemptions")
    except Exception as _e:
        report["metrics_error"] = repr(_e)[:200]
    # v1.3: a projection is only valid from a complete, error-free commit run (an incomplete run is biased to easy queries)
    report["projection_valid"] = bool(ENGINE_READY and len(FINAL) == len(UNITS) and not SCHED.get("error") and not SKIPPED
                                      and SCHED.get("not_admitted", 0) == 0 and SCHED.get("aborted_at_hard_stop", 0) == 0
                                      and SCHED.get("drained", 0) == 0)
    report["projection_hidden_queries"] = N_HIDDEN_UNITS
    report["projection_hidden_generation_hours"] = report["projection_hidden_end_to_end_hours"] = None
    if not report["projection_valid"]:
        report["projection_reason"] = "incomplete commit run, engine failure, scheduler error, skipped / unadmitted / drained / hard-stopped queries"
    if report["projection_valid"] and SCHED.get("tok_per_sec") and report["mean_total_gen_tokens"]:
        _rate = SCHED["tok_per_sec"]
        _gen_h = N_HIDDEN_UNITS * report["mean_total_gen_tokens"] / _rate / 3600
        _setup_h = (SCHED.get("setup_to_scheduler_sec") or 0) / 3600 + 0.05   # measured setup + report tail
        report["projection_hidden_generation_hours"] = round(_gen_h, 2)
        report["projection_hidden_end_to_end_hours"] = round(_gen_h + _setup_h, 2)
        report["projection_note"] = ("measured with up to %d concurrent queries (%d slots configured); the hidden run's rate "
                                     "may differ (longer contexts, KV pressure, MTP acceptance) - this is not a guarantee"
                                     % (SCHED["max_active"], CONCURRENCY))
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
              f"{report['eval_score_pct']}% | task-normalised {report['eval_task_normalized_pct']}% | grids parsed "
              f"{report['with_grid']}/{report['queries_finished']} | throughput {SCHED['tok_per_sec']} tok/s | projected hidden-test "
              f"({N_HIDDEN_UNITS} queries) end-to-end hours {report.get('projection_hidden_end_to_end_hours')}")
except BaseException:
    print("REPORT ERROR (submission.json is already written):"); print(traceback.format_exc()[-3000:])
