"""E21 (user 2026-10-28) = Duck config B exactly as R3K2 (Experiment 15, sha c058d58b: 50K45K, 16 streams, Duck package
9e4b89285136, before3, weak commit) on vLLM OpenAI servers instead of SGLang, two halves in one notebook:
half A = vLLM 0.19.0 (the S1-60K engine as the E12-V Duck server), half B = vLLM 0.32.0 (the VMS arm), the same 120
evaluation tasks each, ~5.5 h per half; the hidden rerun runs half A only (all 240 tasks, 12 h).
  NB1: RedHat INT4 (W4A16) + MTP-3, 20 streams in both halves (fallback FP8 MTP-3, 16 streams)
  NB2: FP8 + MTP-3 + --mamba-ssm-cache-dtype float16, 16 streams in both halves (fallback: SSM dtype default)
Donor map: Experiment 21 README. Every reused cell or function is taken from its donor notebook text here and asserted.
usage: python build_e21.py            -> ../Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb, ..._NB2_FP8_SSM16_S16.ipynb"""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
EXP = NBDIR.parents[1]
DONORS = {
    "E15B": (EXP / "Experiment 15 - Duck 2x35k independent attempts (planned)" / "Experiment Notebooks" /
             "Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K2_DUCK_SGLANG.ipynb", "c058d58b"),
    "E12V": (EXP / "Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)" / "Experiment Notebooks" /
             "Q27-XHIGH-E12_DUCK_REPL_VLLM_SMOKE1.ipynb", "e85b69c2"),
    "E18": (EXP / "Experiment 18 - S1-60K on vLLM 0.32 (planned)" / "Experiment Notebooks" /
            "Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb", "116d2d41"),
}
VARIANTS = {"NB1": "Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb", "NB2": "Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16.ipynb",
            "NB1SMOKE": "Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20_SMOKE.ipynb",
            "NB2SMOKE": "Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16_SMOKE.ipynb"}
PUSHED: set = set()          # notebook sha256 of every pushed version (the builder never overwrites one)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


NB = {}
for k, (p, s) in DONORS.items():
    b = p.read_bytes()
    assert sha(b).startswith(s), (k, sha(b)[:8])
    NB[k] = [src(c) for c in json.loads(b.decode("utf-8"))["cells"]]
E15B, E12V, E18 = NB["E15B"], NB["E12V"], NB["E18"]
assert E15B[1] == E12V[1] == E18[1], "CELL 0 differs between donors"
assert E15B[6] == E12V[5] == E18[4], "the task-order cell differs between donors"


def rep(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


def segment(text, name):
    """The exact source of a top-level def/class/assignment `name` in a donor cell."""
    for node in ast.parse(text).body:
        n = getattr(node, "name", None)
        if n is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
        if n == name:
            return ast.get_source_segment(text, node)
    raise KeyError(name)


def module_bindings(text):
    """Every name a cell binds at module level (not inside its def/class bodies): the `global` list that makes the
    wrapped cell behave exactly like the cell run at module level."""
    names = []

    def add(n):
        if n not in names:
            names.append(n)

    def targets(t):
        if isinstance(t, ast.Name):
            add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                targets(e)
        elif isinstance(t, ast.Starred):
            targets(t.value)

    def walk(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                add(s.name)
                continue
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    targets(t)
            elif isinstance(s, (ast.AugAssign, ast.AnnAssign)):
                targets(s.target)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                targets(s.target)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for it in s.items:
                    if it.optional_vars is not None:
                        targets(it.optional_vars)
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                for a in s.names:
                    add((a.asname or a.name).split(".")[0])
            elif isinstance(s, ast.Global):
                for n in s.names:
                    add(n)
            for fld in ("body", "orelse", "finalbody"):
                if isinstance(getattr(s, fld, None), list):
                    walk(getattr(s, fld))
            for h in getattr(s, "handlers", []) or []:
                if h.name:
                    add(h.name)
                walk(h.body)
            for node in ast.walk(s) if not isinstance(s, (ast.FunctionDef, ast.ClassDef)) else []:
                if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                    add(node.target.id)
    walk(ast.parse(text).body)
    return names


def wrap(text, fname, doc):
    """The cell as a function run once per half: every module-level binding declared global (no multi-line string in
    the cell, asserted, so the indentation changes nothing)."""
    import io
    import tokenize
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):     # rev 2: any multi-line token (f-strings too)
        assert tok.start[0] == tok.end[0] or tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
                                                          tokenize.ENDMARKER), ("multi-line token", tok.type, tok.start)
    tree = ast.parse(text)
    for n in ast.walk(tree):          # rev 2 (VAL_state_B1 L9, B2 LOW-7): constructs the wrapper cannot keep
        assert not isinstance(n, (ast.Match, ast.Delete) + ((ast.TypeAlias,) if hasattr(ast, "TypeAlias") else ())), \
            ("construct the wrapper cannot keep", type(n).__name__, getattr(n, "lineno", None))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id not in ("exec", "eval", "locals", "vars", "dir"), ("scope-sensitive call", n.func.id, n.lineno)
    assert not (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
                and isinstance(tree.body[0].value.value, str)), "a module docstring would become a function docstring"
    names = module_bindings(text)
    body = "\n".join(("    " + l) if l.strip() else "" for l in text.splitlines())
    glob = "    global " + ", ".join(names) if names else ""
    out = f"def {fname}():\n    # {doc}\n{glob}\n{body}\n"
    import symtable                  # rev 2: every remaining local of the function must be a comprehension variable
    st = symtable.symtable(out, "<wrap>", "exec").get_children()[0]
    comp = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                for t in ast.walk(g.target):
                    if isinstance(t, ast.Name):
                        comp.add(t.id)
    bad = [x.get_name() for x in st.get_symbols() if x.is_local() and not x.is_global() and x.get_name() not in comp
           and not x.get_name().startswith(".")]   # ".format" etc.: compiler-internal names
    assert not bad, ("locals left in the wrapped cell", bad)
    return out


# ------------------------------------------------------------------ CELL 1: configuration (E15B CELL 1 + the E21 block)
def cell1(variant, profile="E21X2EVAL120"):
    c = E15B[2]
    c = rep(c, 'os.environ.get("Q27X_PROFILE", "E15BEVAL120R3I")', f'os.environ.get("Q27X_PROFILE", "{profile}")')
    c = rep(c, '    "E14SEVAL120G4B": {', '''    # E21 (user 2026-10-28): two halves (vLLM 0.19, then 0.32) on all 120 evaluation tasks in one 12-h commit; each half
    # gets a fixed solve window from its own start (E21_HALF_SOLVE_SEC); the hidden rerun = RERUN_ALL with half A only
    "E21X2EVAL120": {"source": "eval", "n_tasks": 120, "time_budget_sec": 12.0 * 3600 - 3 * 60, "admit_frac": 0.90,
                     "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
                     "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "solve_window_sec": 5.45 * 3600},   # rev 2b
    # rev 2d: the GPU smoke (both halves, real loads, probes, ~2 h): 12 evaluation tasks, a 20-min window per half
    "E21SMOKE": {"source": "eval", "n_tasks": 12, "time_budget_sec": 2.5 * 3600, "admit_frac": 0.90,   # rev 2f
                 "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
                 "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "solve_window_sec": 20 * 60},
    "E14SEVAL120G4B": {''')
    nb1 = variant == "NB1"
    spec = ('{"A": {"vllm": "0.19.0", "weights": "int4", "seqs": 20, "ssm": None,\n'
            '              "fallback": {"weights": "fp8", "seqs": 16, "ssm": None}},\n'
            '        "B": {"vllm": "0.32.0", "weights": "int4", "seqs": 20, "ssm": None,\n'
            '              "fallback": {"weights": "fp8", "seqs": 16, "ssm": None}}}') if nb1 else (
           '{"A": {"vllm": "0.19.0", "weights": "fp8", "seqs": 16, "ssm": "float16",\n'
           '              "fallback": {"ssm": None}},\n'
           '        "B": {"vllm": "0.32.0", "weights": "fp8", "seqs": 16, "ssm": "float16",\n'
           '              "fallback": {"ssm": None}}}')
    c += f'''

# ==================================================================== E21 (Experiment 21, user 2026-10-28)
# Duck config B (= R3K2, everything above unchanged) on vLLM OpenAI servers. Half A = vLLM 0.19.0 in this kernel (the
# S1-60K engine, launched as the E12-V Duck server); half B = vLLM 0.32.0 in /tmp/vllm-venv (the VMS installer and arm).
# Both halves solve the same 120 evaluation tasks with fresh Duck state; the hidden rerun runs half A only.
E21_VARIANT = "{variant}"
E21_HALF_SPECS = {{"{variant}": {spec}}}
E21_HALVES = ["A"] if RERUN else ["A", "B"]
ENGINE_BACKEND = "vllm"                       # CELL 1b stages the variant's weights; CELL 2 installs vllm==0.19.0 here
E21_STAGE_PATHS = [{"SGL_MODEL_PATH, MODEL_PATH" if nb1 else "MODEL_PATH"}]   # rev 2: NB1 stages its FP8 fallback too (VAL_engine_A2 H2)
E12_SOLVE_WINDOW_SEC = (float(os.environ.get("E21_HALF_SOLVE_SEC", P["solve_window_sec"]))
                        if P.get("solve_window_sec") else None)          # per half, from its own solve start
E21_HALF_SOLVE_SEC = E12_SOLVE_WINDOW_SEC or 0.0
E21_B_SETUP_RESERVE_SEC = float(os.environ.get("E21_B_SETUP_RESERVE_SEC", 30 * 60))   # rev 2b (C1 M3): switch + launch + preflight
E21_LAUNCH_TIMEOUT_SEC = 30 * 60
E21_LAUNCH_TIMEOUT_V032_SEC = 20 * 60      # rev 2b (C1 M3): 0.32 from staged weights was ready in ~550 s (E18)
# rev 2b (VAL_fidelity_E1 M2, E2 M2): Duck counts reasoning tokens with E12_TOKENIZER_JSON (the INT4 file, as R3K2);
# without the INT4 model attached it silently fell back to a chars/token ratio. The FP8 tokenizer has the same vocab
# and merges and encodes identically (README Finding 4, Kaggle CPU kit on the real files).
if not Path(os.environ.get("E12_TOKENIZER_JSON", "")).is_file() and (MODEL_PATH / "tokenizer.json").is_file():
    os.environ["E12_TOKENIZER_JSON"] = str(MODEL_PATH / "tokenizer.json")
E21_B_FOLLOWS_A = True                       # user option (ka): a half-A fallback config is repeated in half B
assert PROFILE in ("E21X2EVAL120", "E21SMOKE", "RERUN_ALL") or os.environ.get("E21_ANY_PROFILE") == "1", PROFILE   # rev 2 (E1 LOW-5)
assert RERUN or len(E21_HALVES) == 1 or (E12_SOLVE_WINDOW_SEC or 0) > 0, "two halves need a solve window"   # C1 L3
E21_VLLM_MAX_MODEL_LEN = 135168               # E12-V: Duck's 128 Ki context window + margin
E21_PORTS = {{"A": 1234, "B": 8011}}
E21_SERVED = {{"fp8": "Qwen/Qwen3.8-27B-FP8", "int4": "Qwen/Qwen3.8-27B-INT4"}}   # never "qwen" (Duck alias, E12-V run 1)
E18_VLLM_WHEELHOUSE = Path(os.environ.get("E18_VLLM_WHEELHOUSE", "/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"))
E18_SILENCE_SEC = float(os.environ.get("E18_SILENCE_SEC", 600))
E21_FAKE = os.environ.get("E21_FAKE_SERVER") if os.environ.get("E11_HARNESS") == "1" else None   # CPU harness only
E18_FAKE = E21_FAKE
E12_R6["E12_R6_CACHE_DIAG"] = 0               # the SGLang cache diagnostic (after solving) is not run on vLLM
os.environ["E12_R6_CACHE_DIAG"] = "0"
if E12_SOLVE_WINDOW_SEC and not RERUN:
    assert 2 * E12_SOLVE_WINDOW_SEC + E21_B_SETUP_RESERVE_SEC < HARD_STOP - NOTEBOOK_T0, "two windows must fit"
CONFIG["e21"] = {{"variant": E21_VARIANT, "halves": E21_HALVES, "specs": E21_HALF_SPECS[E21_VARIANT],
                 "half_solve_sec": E12_SOLVE_WINDOW_SEC, "b_setup_reserve_sec": E21_B_SETUP_RESERVE_SEC,
                 "max_model_len": E21_VLLM_MAX_MODEL_LEN, "served": E21_SERVED, "ports": E21_PORTS,
                 "stage": [str(p) for p in E21_STAGE_PATHS], "vllm032_wheelhouse": str(E18_VLLM_WHEELHOUSE),
                 "reasoning_history_key": "reasoning"}}
# rev 2 (VAL_fidelity_E2 LOW-1, E1 LOW-1): CONFIG was snapshot above before this block; record what really runs
CONFIG["e11"]["ENGINE_BACKEND"] = ENGINE_BACKEND
CONFIG["e12"]["E12_SOLVE_WINDOW_SEC"] = E12_SOLVE_WINDOW_SEC
CONFIG["e12"].setdefault("rev6", {{}})["E12_R6_CACHE_DIAG"] = 0
CONFIG["e12"]["E12_STREAMS"] = "per half (CONFIG['e21'] / e12_run.streams)"
print("E21:", json.dumps(CONFIG["e21"], default=str), flush=True)
'''
    return c


# ------------------------------------------------------------------ CELL 1b: E14 facts / staging (the variant's weights)
def cell1b():
    c = E15B[3]
    c = rep(c, 'E14L.read_probe(SGL_MODEL_PATH) if ENGINE_BACKEND == "sglang" else None',
            'E14L.read_probe(E21_STAGE_PATHS[0])   # E21: the weights this variant loads')
    c = rep(c, 'if ENGINE_BACKEND == "sglang" and E14L.loadpath_enabled():',
            'if ENGINE_BACKEND in ("sglang", "vllm") and E14L.loadpath_enabled():       # E21: vLLM halves too')
    c = rep(c, "E14L.Staging([SGL_MODEL_PATH, SGL_DRAFT_PATH], workers=24,", "E14L.Staging(E21_STAGE_PATHS, workers=24,")
    c = rep(c, '"submit_arm": None if E14_DO_MEASURE else "DFlash width 6, 16 streams: mrr 17, graphs 1..17, mamba 68, min-free-slots-delay 1 (Exp 15 Notes/02)",',
            '"submit_arm": "E21: vLLM halves, see CONFIG[\'e21\']",')
    return c


# ------------------------------------------------------------------ CELL 3: Duck package (+ the vLLM reasoning key)
def cell3():
    c = E15B[5]
    old = ('os.environ.update(E12A.duck_env(f"http://127.0.0.1:{int(os.environ.get(\'E11_PORT\', 30000))}/v1", "pending", '
           'SGL_MAX_RUNNING))\n')
    c = rep(c, old, old + '''# E21: vLLM 0.19 / 0.32 read only message["reasoning"] of an assistant history message (vllm/entrypoints/chat_utils.py
# 1512 in v0.19.0; vllm/renderers/chat_utils.py 2083 in v0.32.0); "reasoning_content" (DUCK_ENV, read by SGLang) would
# be dropped from the prompt. E12-V run 2 sent "reasoning" and its prompts grew by the generated tokens (6,882 + 12,288
# -> 19,215). Duck reads the key at call time (tool_agent._reasoning_history_keys).
os.environ["ARC3_REASONING_HISTORY_KEY"] = "reasoning"
''')
    anchor = "_E12_TA, _E12_DP, _e12_run_sandboxed, _ = E12A._import_duck()\n"
    c = rep(c, anchor, anchor + (HERE / "e21_cell3_patch.py").read_text(encoding="utf-8"))
    return c


# ------------------------------------------------------------------ CELL 5: the vLLM engine library
def engine_cell():
    e12v5, e18_5 = E12V[6], E18[6]
    parts = [
        ("E12-V CELL 5", segment(e12v5, "ensure_libcuda_linkable")),
        ("E12-V CELL 5", segment(e12v5, "_e12v_request_json")),
        ("E12-V CELL 5", segment(e12v5, "_E12V_PROM_RE")),
        ("E12-V CELL 5", segment(e12v5, "_E12V_ALIAS")),
        ("E12-V CELL 5", segment(e12v5, "metrics_snapshot")),
        ("E12-V CELL 5", segment(e12v5, "_E12V_STAT_RE")),
        ("E12-V CELL 5", segment(e12v5, "e11_decode_summary")),
        ("E12-V CELL 5", segment(e12v5, "e11_kill_server")),
        ("E18 R2 CELL 5", segment(e18_5, "e18_install")),
        ("E18 R2 CELL 5", segment(e18_5, "E18Server")),
        ("E18 R2 CELL 5", segment(e18_5, "_e18_kill_gpu_pids")),
        ("E18 R2 CELL 5 (= S1-60K CELL 5)", segment(e18_5, "_wait_gpu_release")),
    ]
    head = '''# ==================================================================== CELL 5 — E21: vLLM engine library (two halves)
# Donor functions, verbatim (build_e21.py extracts and asserts them):
#   E12-V CELL 5 (Q27-XHIGH-E12_DUCK_REPL_VLLM_SMOKE1 e85b69c2 = the vLLM SMOKE1 run 2 Duck server): ensure_libcuda_linkable,
#     _e12v_request_json, the Prometheus reader + vLLM->SGLang name aliases (metrics_snapshot), the stats-line summary
#     (e11_decode_summary), e11_kill_server;
#   E18 R2 CELL 5 (116d2d41): e18_install (the VMS vLLM 0.32 installer), E18Server (ready = /health + its own KV line,
#     stall on 600 s of log silence; kill = SIGTERM then always SIGKILL the group), _e18_kill_gpu_pids, _wait_gpu_release
#     (= S1-60K CELL 5).
# New (E21): the per-half flag sets (vLLM 0.19 = the E12-V command; vLLM 0.32 = the VMS arm), the capacity check, one
# launch with its smoke, the per-half begin / end and the watchdog relaunch.
import urllib.request as _urlreq, signal as _signal, statistics as _statistics
import shutil as _shutil
E11_HARNESS = os.environ.get("E11_HARNESS") == "1"            # local CPU harness only: fake server, no GPU
E18_URL = None
SGL_URL = None
ENGINE_READY = False; llm = None; engine = None; CONCURRENCY = 0; E11_SERVER = None; E12_VLLM_CONCURRENCY = 0
ENGINE_INFO = {"backend": None}
'''
    body = "\n\n\n".join(f"# ---- {who}, verbatim\n{s}" for who, s in parts)
    new = (HERE / "e21_engine_new.py").read_text(encoding="utf-8")
    return head + "\n" + body + "\n\n\n" + new


# ------------------------------------------------------------------ CELL 6: the Duck run (E15B CELL 6, vLLM probes)
CB_OLD = 'def _e12_branch_done_r6(rec):   # rev 6: a branch reports every pause and its finish; keep the latest per branch\n    with _E12_LOCK:\n        rows = E12_RECORDS.setdefault(rec["task_id"], [])\n        # rev 6g H7-1 (GPT rev 6f audits): callbacks can finish out of order; never let an older record replace a newer one\n        _old = next((r for r in rows if r.get("branch") == rec.get("branch")), None)\n        if _old is not None and int(rec.get("publish_seq") or 0) <= int(_old.get("publish_seq") or 0):\n            E12_RUN["stale_callbacks_dropped"] = E12_RUN.get("stale_callbacks_dropped", 0) + 1\n            return\n        new = rec.get("branch") not in {r.get("branch") for r in rows}\n        rows[:] = [r for r in rows if r.get("branch") != rec.get("branch")] + [rec]\n        E12_RUN["branches_started"] = E12_RUN.get("branches_started", 0) + int(new)\n        E12_RUN["branches_done"] = sum(1 for v in E12_RECORDS.values() for r in v if r.get("status") == "done")\n        E12_RUN["branches_accepted"] = sum(1 for v in E12_RECORDS.values() for r in v if r.get("accepted"))\n    if time.perf_counter() >= globals().get("E15B_SOLVE_END_PC", float("inf")):'
CB_NEW = 'def _e12_branch_done_r6(rec, _recs=E12_RECORDS, _run=E12_RUN, _lock=_E12_LOCK, _half=E21_HALF):\n    # rev 6: a branch reports every pause and its finish; keep the latest per branch.\n    # E21 rev 2c (GPT audits 1-3, CRITICAL, reproduced on this source): the callback is bound to ITS half\'s records, run\n    # state and lock (default arguments, evaluated when this half\'s CELL 6 defines it). A half-A worker that is still\n    # alive (a straggler past join grace) can never write into half B\'s records, and never writes half B\'s submission.\n    with _lock:\n        rows = _recs.setdefault(rec["task_id"], [])\n        # rev 6g H7-1 (GPT rev 6f audits): callbacks can finish out of order; never let an older record replace a newer one\n        _old = next((r for r in rows if r.get("branch") == rec.get("branch")), None)\n        if _old is not None and int(rec.get("publish_seq") or 0) <= int(_old.get("publish_seq") or 0):\n            _run["stale_callbacks_dropped"] = _run.get("stale_callbacks_dropped", 0) + 1\n            return\n        new = rec.get("branch") not in {r.get("branch") for r in rows}\n        rows[:] = [r for r in rows if r.get("branch") != rec.get("branch")] + [rec]\n        _run["branches_started"] = _run.get("branches_started", 0) + int(new)\n        _run["branches_done"] = sum(1 for v in _recs.values() for r in v if r.get("status") == "done")\n        _run["branches_accepted"] = sum(1 for v in _recs.values() for r in v if r.get("accepted"))\n    if _half != E21_HALF:\n        _run["late_cross_half_callbacks"] = _run.get("late_cross_half_callbacks", 0) + 1\n        return\n    if time.perf_counter() >= globals().get("E15B_SOLVE_END_PC", float("inf")):'
CB_OLD2 = '        E12_RUN["late_writes_skipped"] = E12_RUN.get("late_writes_skipped", 0) + 1     # E15B B1\n        return'
CB_NEW2 = '        _run["late_writes_skipped"] = _run.get("late_writes_skipped", 0) + 1     # E15B B1\n        return'
HB_OLD = 'def _e12_heartbeat(every=300):\n    while not _E12_STOP.wait(every):'
HB_NEW = "def _e12_heartbeat(every=300, _stop=_E12_STOP):     # E21 rev 2c: bound to this half's stop event (GPT audit 1)\n    while not _stop.wait(every):"
HB_START_OLD = '        threading.Thread(target=_e12_heartbeat, daemon=True).start()'
HB_START_NEW = '        threading.Thread(target=_e12_heartbeat, daemon=True, name="e12_heartbeat").start()'
SCHED_OLD = '        E12_RUN["rev6_scheduler"] = dict(E12A.SCHED_LAST)'
SCHED_NEW = '        E12_RUN["rev6_scheduler"] = dict(E12A.SCHED_LAST)\n        # E21 rev 2c (GPT audits 1-2): the deadline contract as evidence (planned solve end vs when the solver returned)\n        E12_RUN["solve_overrun_sec"] = round(max(0.0, time.perf_counter() - _e15b_solve_end), 1)\n        E12_RUN["planned_solve_end_sec"] = round(_e15b_solve_end - NOTEBOOK_T0, 1)\n        E12_RUN["solve_end_epoch"] = time.time() + (_e15b_solve_end - time.perf_counter())   # rev 2f: request t / t0 are epoch\n        E12_RUN["stragglers"] = (E12_RUN["rev6_scheduler"] or {}).get("stragglers")\n        if E12_RUN["solve_overrun_sec"] > 120:\n            E12_RUN["deadline_violation"] = True\n            print("E21 WARNING: the solver returned", E12_RUN["solve_overrun_sec"], "s after its deadline", flush=True)'


NEW_PROBE = '''def _e12_capacity_probe():   # E21: vLLM 0.19 / 0.32 gauges (the max over label rows); vLLM has no separate Mamba gauge
    try:
        rows = [r for r in e21_prom() if getattr(r, "value", None) is not None]
    except Exception:
        rows = None
    if not rows:
        # E21 rev 2c (GPT audit 1 HIGH-4): Duck's KV gate admits on a missing sample (fail-open). On vLLM a failed
        # /metrics read while requests run means "unknown", so report a full pool: admissions wait; with nothing
        # running Duck's idle override still admits (no deadlock); the watchdog handles a dead server.
        E21_STATE["probe_failures"] = E21_STATE.get("probe_failures", 0) + 1
        _t0 = E21_STATE.setdefault("probe_fail_since", time.monotonic())
        if time.monotonic() - _t0 > 120:     # rev 2d: a long outage -> Duck's own fail-open (counted), not 1 stream forever
            E21_STATE["probe_fail_open"] = E21_STATE.get("probe_fail_open", 0) + 1
            return {"kv_usage": None, "running": None, "queue": None, "mamba_usage": None, "probe_failed": True}
        return {"kv_usage": 1.0, "running": None, "queue": None, "mamba_usage": None, "probe_failed": True}
    E21_STATE.pop("probe_fail_since", None)
    pick = lambda name: [float(r.value) for r in rows if r.name == name]
    out = {}
    for key, name in (("kv_usage", "vllm:kv_cache_usage_perc"), ("running", "vllm:num_requests_running"),
                      ("queue", "vllm:num_requests_waiting")):
        v = pick(name)
        out[key] = max(v) if v else None
    out["mamba_usage"] = None
    return out'''
NEW_PROGRESS = '''def _e21_progress():
    # E21 rev 2h (GPT v3 HIGH-1, VAL2_dryrun F1): (requests running or waiting, prompt + generated tokens) from /metrics; None when
    # /metrics fails or lacks a field (then the stall rule never acts)
    try:
        _r = e21_prom()
    except Exception:
        return None
    _run = [x.value for x in _r if x.name in ("vllm:num_requests_running", "vllm:num_requests_waiting")]
    _tok = [x.value for x in _r if x.name in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")]
    return (sum(_run), sum(_tok)) if (_run and _tok) else None      # running + waiting on the server'''
NEW_GEN = '''def _e15b_gen_tokens_total():
    # E15B #32 on vLLM (E21): the server's own generated-token counter (vllm:generation_tokens_total, summed over rows)
    try:
        _v = [float(r.value) for r in e21_prom() if getattr(r, "name", "") == "vllm:generation_tokens_total"
              and getattr(r, "value", None) is not None]
        return sum(_v) if _v else None
    except Exception:
        return None'''
NEW_LOGSUM = (HERE / "e21_logsum.py").read_text(encoding="utf-8").rstrip("\n")
NEW_WATCHDOG = '''        def _e15b_watchdog(_wd_stop=_E12_STOP, _wd_run=E12_RUN, _wd_half=E21_HALF, _e15b_solve_end=_e15b_solve_end):
            # E21 rev 2f (VAL2_cells F1): this half's objects bound at def time (a module-level binding is global
            # after wrap() and half B would rebind it)
            # E21: the E15B watchdog on vLLM: a dead server (or /health failing for _WD_FAIL s while its log is silent)
            # is relaunched once with the same half's ready config (e21_relaunch, CELL 5); while it is down the
            # branches wait (arc2_duck.SERVER_DOWN) instead of burning their consecutive-failure budget
            _bad, _sz_last = None, {}
            _wd = _wd_run["watchdog"]
            _prog = {"v": None, "t": time.monotonic()}      # rev 2h: the last progress value and when it changed
            _wd.update(stall_sec=_WD_STALL, stalls=0, no_progress_sec_max=0)
            while not _wd_stop.wait(_WD_POLL):
                if E21_HALF != _wd_half:          # rev 2c: a half-A watchdog never acts in half B
                    return
                _srv = E11_SERVER
                _dead = _srv is None or not _srv.alive()
                _stall = False
                if not _dead:
                    _tk = _e15b_gen_tokens_total()      # B6: the last counter value of THIS server (it resets on relaunch)
                    if _tk is not None:
                        _wd["tok_last"] = _tk
                    # rev 2h (GPT v3 HIGH-1): /health 200 + a silent log while running requests make no progress = hung
                    _pr = _e21_progress()
                    if _pr is not None:
                        if not _pr[0] or _pr[1] != _prog["v"]:
                            _prog.update(v=_pr[1], t=time.monotonic())
                        else:
                            _np = time.monotonic() - _prog["t"]
                            _wd["no_progress_sec_max"] = max(_wd["no_progress_sec_max"], round(_np))
                            _stall = _np >= _WD_STALL
                if _stall:
                    _wd["stalls"] += 1
                    print("E15B watchdog: no progress for", round(time.monotonic() - _prog["t"]), "s with",
                          _pr[0], "requests running -> treated as dead", flush=True)
                elif not _dead:
                    try:     # B2: a growing server log = alive; /health only when it is silent
                        _sz = os.path.getsize(_srv.log_path)
                    except Exception:
                        _sz = None
                    if _sz is not None and _sz != _sz_last.get(id(_srv)):
                        _sz_last[id(_srv)] = _sz
                        _bad = None
                        continue
                    try:
                        _wd["health_probes"] += 1
                        _urlreq.urlopen(SGL_URL + "/health", timeout=10).read()
                        _bad = None
                        continue
                    except Exception:
                        _bad = _bad or time.monotonic()
                        _wd["health_fail_sec_max"] = max(_wd["health_fail_sec_max"],
                                                                         round(time.monotonic() - _bad))
                        if time.monotonic() - _bad < _WD_FAIL:
                            continue
                _left = _e15b_solve_end - time.perf_counter()
                _rec = {"t": round(time.perf_counter() - NOTEBOOK_T0), "dead": _dead, "left_sec": round(_left),
                        "stall": _stall}
                if len(_wd["relaunches"]) >= 2 or _left < _WD_MIN_LEFT:
                    _rec["action"] = "none (limit or < 20 min left)"
                    _wd["relaunches"].append(_rec)
                    print("E15B watchdog: server down, no relaunch:", json.dumps(_rec), flush=True)
                    return
                if E21_HALF != _wd_half:          # rev 2f: re-check right before acting
                    return
                E12A.SERVER_DOWN.set()
                print("E15B watchdog: server down -> relaunch", json.dumps(_rec), flush=True)
                try:
                    try:
                        _wd["old_logs"].append(str(_srv.log_path))
                    except Exception:
                        pass
                    if _wd.get("tok_last") is not None and _wd.get("tok_base") is not None:   # B6
                        _wd["tok_acc"] += max(0.0, _wd["tok_last"] - _wd["tok_base"])
                    _wd["tok_base"], _wd["tok_last"] = 0.0, None
                    _ok, _why = e21_relaunch()
                    _rec.update(ok=_ok, why=_why)
                except Exception as _e:
                    _rec.update(ok=False, why=repr(_e)[:300])
                finally:
                    E12A.SERVER_DOWN.clear()
                    _wd["relaunches"].append(_rec)
                    _bad = None
                    _prog.update(v=None, t=time.monotonic())
                print("E15B watchdog:", json.dumps(_rec), flush=True)
                if not _rec.get("ok") and len(_wd["relaunches"]) >= 2:    # B3: a failed relaunch retries once
                    return
        threading.Thread(target=_e15b_watchdog, daemon=True, name="e15b_watchdog").start()'''


def cell6():
    c = E15B[9]
    for _o, _n in ((CB_OLD, CB_NEW), (CB_OLD2, CB_NEW2), (HB_OLD, HB_NEW), (HB_START_OLD, HB_START_NEW),
                   (SCHED_OLD, SCHED_NEW)):
        c = rep(c, _o, _n)
    for name, new in (("_e12_capacity_probe", NEW_PROBE), ("_e15b_gen_tokens_total", NEW_PROGRESS + "\n\n\n" + NEW_GEN),
                      ("_e15b_server_log_summary", NEW_LOGSUM)):
        c = rep(c, segment(c, name), new)
    i0 = c.index("        def _e15b_watchdog():")
    end = '        threading.Thread(target=_e15b_watchdog, daemon=True, name="e15b_watchdog").start()'
    i1 = c.index(end) + len(end)
    assert c.count(end) == 1 and "e11_flags" in c[i0:i1] and "E11Server(" in c[i0:i1]
    c = c[:i0] + NEW_WATCHDOG + c[i1:]
    c = rep(c, '_WD_POLL = float(os.environ.get("E15B_WD_POLL_SEC", "10"))',
            '_WD_POLL = float(os.environ.get("E15B_WD_POLL_SEC", "10"))\n'
            '        _WD_STALL = float(os.environ.get("E15B_WD_STALL_SEC", "600"))     # E21 rev 2h: no-progress limit')
    c = rep(c, "r = json.loads(_urlreq.urlopen(req, timeout=180).read())",
            "r = json.loads(_urlreq.urlopen(req, timeout=60).read())     # E21 rev 2b (C1 M3): <= ~2 min per preflight")
    c = rep(c, 'E12_RUN["model_id"] = str(SGL_MODEL_PATH)',
            'E12_RUN["model_id"] = E21_SERVED[(ENGINE_INFO.get("cfg") or {}).get("weights", "fp8")]   # E21: the served name')
    c = rep(c, "        _end = HARD_STOP - E12_STOP_BEFORE_HARD_SEC\n",
            "        _end = min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC)     # E21: this half's latest end\n")
    c = rep(c, "_end = _e12_shifted_end(_end, _shift, HARD_STOP - E12_STOP_BEFORE_HARD_SEC)",
            "_end = _e12_shifted_end(_end, _shift, min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC))")
    # no SGLang server call is left outside _e12_no_spec / _e12_parity (the parity profile only; never called in E21)
    rest = c
    for name in ("_e12_no_spec", "_e12_parity"):
        rest = rest.replace(segment(c, name), "")
    for bad in ("e11_flags", "E11Server(", "sglang:", "E14_VENV", "e11_prom("):
        assert bad not in rest, bad
    assert "e21_relaunch()" in c and "e21_prom()" in c
    return c


# ------------------------------------------------------------------ CELL 7: report + per-half bundle (E15B CELL 7)
def cell7():
    c = E15B[10]
    c = rep(c, '_pack = WORKING / "Q27X_E12_DUCK_BUNDLE.zip"', '_pack = WORKING / f"Q27X_E21_{E21_VARIANT}_HALF{E21_HALF}_DUCK_BUNDLE.zip"')
    c = rep(c, '"cache_note": "aggregate over all requests; a missing prompt_tokens_details means 0 cached (SGLang 0.5.21)",',
            '"cache_note": "aggregate over all requests; a missing prompt_tokens_details means 0 cached (vLLM 0.19 / 0.32, as SGLang 0.5.21)",')
    return c


RUNNER = '''# ==================================================================== CELL 8 — E21: run the halves
# Commit: half A (vLLM 0.19) then half B (vLLM 0.32) on the same 120 evaluation tasks, each with fresh Duck state and its
# own folder q27x/half<H>/; the hidden rerun: half A only. Each step is isolated: a failed half never stops the next.
def _e21_cell6_of(H):
    E21_STATE["halves"].setdefault(H, {})["cell6_ran"] = True     # CELL 6 re-creates E12_RECORDS / E12_RUN first
    e21_cell6()


for _H in E21_HALVES:
    if _H != E21_HALVES[0] and E21_STATE.get("abort_next_half"):        # rev 2h (GPT v3 audit 2): fail closed
        E21_STATE["halves"].setdefault(_H, {}).setdefault("errors", []).append(
            {"step": "handoff", "error": "PREVIOUS_WATCHDOG_STILL_ACTIVE: " + str(E21_STATE["abort_next_half"])})
        print(f"E21 half {_H} NOT started:", E21_STATE["abort_next_half"], flush=True)
        continue
    for _step, _fn in (("begin", lambda: e21_begin_half(_H)), ("cell6", lambda: _e21_cell6_of(_H)), ("cell7", e21_cell7),
                       ("end", lambda: e21_end_half(_H))):
        try:
            _fn()
        except BaseException:
            E21_STATE["halves"].setdefault(_H, {}).setdefault("errors", []).append({"step": _step,
                                                                                   "traceback": traceback.format_exc()[-3000:]})
            print(f"E21 half {_H} {_step} error:", traceback.format_exc()[-2000:], flush=True)
            if _step == "begin":
                ENGINE_READY = False
'''

REPORT = '''# ==================================================================== CELL 9 — E21: the two-half report and bundle
# Per half: the full score (attempts 1 + 2, from its CELL 7 report), the score of attempt 1 alone (branch 0 only; the
# same for both halves whatever attempt 2 reached), engine config actually used, capacity, throughput, prefix cache,
# preemptions, timeline. Written to q27x/e21_report.json and printed as one table.
def _e21_score(recs):
    if SOLUTIONS is None:
        return None
    gen = E12A.build_submission({t: CHALLENGES[t] for t in TASK_IDS}, {t: list(recs.get(t, [])) for t in TASK_IDS})
    hits = total = 0; tn = 0.0
    for t in TASK_IDS:
        th = 0
        for qi, att in enumerate(gen[t]):
            g = SOLUTIONS[t][qi]; h = att["attempt_1"] == g or att["attempt_2"] == g
            hits += h; total += 1; th += h
        tn += th / max(1, len(gen[t]))
    return {"query_hits": hits, "queries": total, "task_norm_pct": round(100 * tn / max(1, len(TASK_IDS)), 2)}


def _e21_stage1(recs, run):
    """rev 2b (VAL_timing_C2 LOW-4): the attempt-1-only score compares halves only if stage 1 (attempt 1 of every task)
    finished in both: when the last attempt 1 ended, how many attempts 1 never started or were cut by the deadline."""
    b0 = [r for v in recs.values() for r in v if r.get("branch") == 0]
    ends = [max((q.get("t", 0) + (q.get("sec") or 0)) for q in (r.get("requests") or [])) for r in b0 if r.get("requests")]
    t0 = min((r.get("t0") or 0) for r in b0) if b0 else None
    last = max(ends) if ends else None
    dl = (run or {}).get("solve_end_epoch")          # rev 2f (VAL2_cells F2): the solve deadline, not the last end
    return {"attempt1_records": len(b0), "tasks": len(TASK_IDS),
            "attempt1_missing": len(TASK_IDS) - len({r.get("task_id") for r in b0}),
            "attempt1_cut_by_deadline": (sum(1 for x in ends if dl - x < 90) if (ends and dl) else None),
            "attempt1_all_done_h": round((last - t0) / 3600, 2) if (ends and t0) else None}


def _e21_pass1(recs):
    if SOLUTIONS is None:
        return None
    gen = E12A.build_submission({t: CHALLENGES[t] for t in TASK_IDS},
                                {t: [r for r in recs.get(t, []) if r.get("branch") == 0] for t in TASK_IDS})
    h = sum(int(a["attempt_1"] == SOLUTIONS[t][qi]) for t in TASK_IDS for qi, a in enumerate(gen[t]))
    return {"pass1_query_hits": h, "queries": sum(len(gen[t]) for t in TASK_IDS)}


E21_REPORT = {"run_id": RUN_ID, "profile": PROFILE, "rerun": RERUN, "variant": E21_VARIANT, "config": CONFIG.get("e21"),
              "halves": {}}
try:
    for _H in E21_HALVES:
        _st = E21_STATE["halves"].get(_H, {})
        _rec = _st.get("records") or {}
        _rep = {}
        try:
            _rep = json.loads((BASE_OUT / f"half{_H}" / "e12_run_report.json").read_text(encoding="utf-8"))
        except Exception as _e:
            _rep = {"error": repr(_e)[:200]}
        _eng = _st.get("engine") or {}
        _run = _st.get("e12_run") or {}
        _bs = _run.get("b_summary") or {}
        _sl = _bs.get("server_log") or {}
        E21_REPORT["halves"][_H] = {
            "engine_used": _eng.get("backend"), "config_used": _eng.get("used"), "spec": _eng.get("spec"),
            "attempts": [{k: a.get(k) for k in ("tag", "ready", "why", "ready_sec", "error")} for a in _eng.get("attempts", [])],
            "capacity": _eng.get("capacity"), "streams": _run.get("streams"), "workers": _run.get("rev6_workers"),
            "begin_sec": _eng.get("begin_sec"), "solve_start_sec": _run.get("started_sec"),
            "solve_finished_sec": _run.get("finished_sec"), "deadline_in_sec": _run.get("deadline_in_sec"),
            "end_sec": _st.get("end_sec"), "relaunches": _eng.get("relaunches"),
            "score_full": {k: _rep.get(k) for k in ("eval_query_hits", "eval_query_total", "eval_pass1_hits",
                                                    "eval_pass2_hits", "eval_task_norm_pct")},
            "score_attempt1_only": _e21_score({t: [r for r in v if r.get("branch") == 0] for t, v in _rec.items()}),
            "attempts_started": _bs.get("attempts"), "attempt2_started": _bs.get("attempt2_started"),
            "accepted": _bs.get("accepted"), "weak_accepts": _bs.get("weak_accepts"),
            "server_running_mean": _bs.get("server_running_mean"), "gen_tok_s_p50": _sl.get("gen_tok_s_p50"),
            "gen_tok_s_mean": _sl.get("gen_tok_s_mean"),
            # rev 2e (Kaggle CPU kit H3): the rev 2 log summary names it "..._rolling" (vLLM's last-1,000-request rate)
            "prefix_cache_hit_pct_last": _sl.get("prefix_cache_hit_pct_last_rolling"),
            "prefix_hit_pct_whole_run": (_eng.get("final_counters") or {}).get("prefix_hit_pct_whole_run"),
            "kv_usage_pct_max": _sl.get("kv_usage_pct_max"), "preempt_lines": _sl.get("preempt_lines"),
            "cached_pct_requests": (_rep.get("usage_all") or {}).get("cached_pct"),
            "server_gen_tokens": _bs.get("server_gen_tokens"), "counted_gen_tokens": _bs.get("counted_gen_tokens"),
            # rev 2: the solve window actually used, the prefill the server recomputed (vLLM align mode never reuses a
            # previous reply: VAL_api_D2 F1), Duck's gate, the server counters, the cut-call restorations, the probe
            "window_sec": _eng.get("window_sec"), "window_note": _eng.get("window_note"), "spec_note": _eng.get("spec_note"),
            "recomputed_prompt_tokens": sum(int(u.get("prompt") or 0) - int(u.get("cached") or 0)
                                            for v in _rec.values() for r in v for u in (r.get("usage") or [])),
            "prompt_tokens": sum(int(u.get("prompt") or 0) for v in _rec.values() for r in v for u in (r.get("usage") or [])),
            "gate": _st.get("gate"), "counters": _eng.get("final_counters"),
            "length_restored": _st.get("length_restored"),
            "stage1": _e21_stage1(_rec, _run),
            "score_attempt1_strict_pass1": _e21_pass1(_rec),
            "contract": next((a.get("contract") for a in reversed(_eng.get("attempts") or []) if a.get("contract")), None),
            "gpu_checks": _eng.get("gpu_checks"),
            "planned_end_sec": _eng.get("end_pc_in_sec"), "planned_solve_end_sec": _run.get("planned_solve_end_sec"),
            "solve_overrun_sec": _run.get("solve_overrun_sec"), "deadline_violation": _run.get("deadline_violation"),
            "stragglers": _run.get("stragglers"), "late_cross_half_callbacks": _run.get("late_cross_half_callbacks"),
            "lmo_in_log": next((a.get("lmo_in_log") for a in reversed(_eng.get("attempts") or []) if "lmo_in_log" in a), None),
            "errors": _st.get("errors"),
            # rev 2h (GPT v3 HIGH-2): solver error, complete scoring, the watchdog's stall evidence
            "solver_error": (_run.get("error") or None) and str(_run.get("error"))[-600:],
            "scoring_complete": (isinstance(_rep.get("eval_query_total"), int) and _rep["eval_query_total"]
                                 == sum(len((CHALLENGES.get(t) or {}).get("test") or []) for t in TASK_IDS) > 0),
            "watchdog_stalls": ((_run.get("watchdog") or {}).get("stalls")),
            "no_progress_sec_max": ((_run.get("watchdog") or {}).get("no_progress_sec_max"))}
    E21_REPORT["tool_calls_with_content"] = globals().get("E21_CALL_WITH_CONTENT")
    _ws = [h.get("deadline_in_sec") for h in E21_REPORT["halves"].values() if h.get("deadline_in_sec")]
    E21_REPORT["windows_sec"] = _ws
    # rev 2c (GPT audits 1/3), rev 2e (CPU kit H3: compare the configs that ran, not the tags; a relaunch that switched
    # config mid-half invalidates the comparison): valid only with equal windows (1 %), the same weights / seqs / SSM in
    # both halves (only the vLLM version differs) and no deadline violation
    _ran = {}
    for H in E21_REPORT["halves"]:
        _e = (E21_STATE["halves"].get(H, {}).get("engine") or {})
        _c = _e.get("cfg")
        _ran[H] = {"cfg": None if not _c else {k: _c.get(k) for k in ("weights", "seqs", "ssm")}, "tag": _e.get("used"),
                   "switched": bool(_e.get("relaunch_cfg_switched"))}
    _prim = {H: {k: v for k, v in E21_HALF_SPECS[E21_VARIANT][H].items() if k in ("weights", "seqs", "ssm")} for H in _ran}
    E21_REPORT["configs_ran"] = _ran
    if any(v["cfg"] is None for v in _ran.values()):
        E21_REPORT["config_status"] = "NOT_RUN"
    elif any(v["switched"] for v in _ran.values()):
        E21_REPORT["config_status"] = "SWITCHED_MIDHALF"
    elif len({json.dumps(v["cfg"], sort_keys=True) for v in _ran.values()}) > 1:
        E21_REPORT["config_status"] = "CONFOUNDED_FALLBACK"
    elif all(v["cfg"] == _prim[H] for H, v in _ran.items()):
        E21_REPORT["config_status"] = "PRIMARY_BOTH"
    else:
        E21_REPORT["config_status"] = "FALLBACK_BOTH"
    E21_REPORT["comparison_valid"] = (len(_ws) == 2 and max(_ws) - min(_ws) <= 0.01 * max(_ws)
                                      and E21_REPORT["config_status"] in ("PRIMARY_BOTH", "FALLBACK_BOTH")
                                      # rev 2e (CPU kit gateAB): the same client streams in both halves
                                      and len({h.get("streams") for h in E21_REPORT["halves"].values()}) == 1
                                      and not any(h.get("deadline_violation") for h in E21_REPORT["halves"].values())
                                      # rev 2h (GPT v3 HIGH-2): both halves ran cleanly and were fully scored
                                      and all(not h.get("errors") and not h.get("solver_error")
                                              and h.get("scoring_complete") is True and not h.get("watchdog_stalls")
                                              for h in E21_REPORT["halves"].values()))
    E21_REPORT["windows_equal"] = (max(_ws) - min(_ws) <= 300) if len(_ws) > 1 else None
    if E21_REPORT["windows_equal"] is False:
        print("E21 WARNING: the halves' solve windows differ by more than 5 min:", _ws, flush=True)
    write_json_atomic(BASE_OUT / "e21_report.json", E21_REPORT)
    print("E21 REPORT", E21_VARIANT, "| profile", PROFILE, "| rerun", RERUN, flush=True)
    for _H, _h in E21_REPORT["halves"].items():
        print(f"  half {_H}: {_h['engine_used']} ({_h['config_used']}) | streams {_h['streams']} | full",
              json.dumps(_h["score_full"]), "| attempt 1 only", json.dumps(_h["score_attempt1_only"]),
              "| attempt 2 started", _h["attempt2_started"], "| tok/s p50", _h["gen_tok_s_p50"],
              "| prefix hit", _h["prefix_cache_hit_pct_last"], "% | cached", _h["cached_pct_requests"],
              "% | preempt lines", _h["preempt_lines"], "| window", _h["window_sec"], "| recomputed",
              _h["recomputed_prompt_tokens"], "/", _h["prompt_tokens"], "| length restored", _h["length_restored"],
              "| errors", len(_h["errors"] or []), flush=True)
except BaseException:
    print("E21 report error:", traceback.format_exc()[-3000:], flush=True)
try:
    _pack = WORKING / f"Q27X_E21_{E21_VARIANT}_BUNDLE.zip"
    with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in [BASE_OUT / "e21_report.json", SUBMISSION_PATH] + sorted(BASE_OUT.glob("half*/e12_run_report.json")) \\
                + sorted(BASE_OUT.glob("half*/vllm_server_*.log")) + sorted(BASE_OUT.glob("half*/sglang_decode_summary.json")) \\
                + sorted((BASE_OUT / "e14").glob("*")) + [WORKING / "e14_state.json"]:
            if fp.exists() and fp.is_file():
                z.write(fp, arcname=fp.relative_to(WORKING).as_posix() if WORKING in fp.parents else fp.name)
    print("E21 bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
except BaseException:
    print("E21 bundle error:", traceback.format_exc()[-2000:], flush=True)
'''


def md(variant):
    v = {"NB1": "RedHat INT4 (W4A16) + MTP-3, **20 streams** (fallback: FP8 MTP-3, 16 streams)",
         "NB2": "FP8 + MTP-3 + `--mamba-ssm-cache-dtype float16`, **16 streams** (fallback: SSM dtype default = S1's engine)"}[variant]
    return (f"# E21 {variant} · Duck config B (= R3K2) on vLLM 0.19 + vLLM 0.32, two halves in one notebook\n\n"
            f"- **Weights / streams:** {v}\n"
            "- **Half A:** vLLM 0.19.0 (the S1-60K engine as the E12-V Duck server) · **Half B:** vLLM 0.32.0 (the VMS arm)\n"
            "- Both halves: the same 120 evaluation tasks, fresh Duck state, ~5.5 h each (commit). Hidden rerun: half A only, 12 h.\n"
            "- Duck: package 9e4b89285136, 50k + 45k attempts, before3, weak commit (R3K2 exactly); history reasoning key "
            "`reasoning` (vLLM reads only that key).\n"
            "- Report: `q27x/e21_report.json` (per half: full score, attempt-1-only score, engine, tok/s, prefix cache).\n")


def build(name):
    variant, profile = (name[:3], "E21SMOKE") if name.endswith("SMOKE") else (name, "E21X2EVAL120")
    return _build(variant, profile, VARIANTS[name])


def _build(variant, profile, out_name):
    e21_cell6 = wrap(cell6(), "e21_cell6", "E15B CELL 6 (R3K2) with the vLLM probes / watchdog (build_e21.py), run once per half")
    e21_cell7 = wrap(cell7(), "e21_cell7", "E15B CELL 7 (R3K2) with a per-half bundle name, run once per half")
    cells = [("markdown", md(variant)),
             ("code", E15B[1]),                       # CELL 0, = every donor
             ("code", cell1(variant, profile)),
             ("code", cell1b()),
             ("code", E15B[4]),                       # CELL 2 (vllm==0.19.0 into this kernel), verbatim
             ("code", cell3()),
             ("code", E15B[6]),                       # task order, = every donor
             ("code", engine_cell()),
             ("code", "# ==================================================================== CELL 6/7 — E21: the Duck run and its report as functions\n"
                      "# Both are E15B R3K2's cells (build_e21.py: the changed lines are listed in the Experiment 21 README),\n"
                      "# wrapped so each half runs them once (every module-level name is declared global: same behaviour).\n"
              + e21_cell6 + "\n\n" + e21_cell7),
             ("code", RUNNER),
             ("code", REPORT)]
    nb = json.loads(DONORS["E15B"][0].read_text(encoding="utf-8"))
    out_cells = []
    for kind, text in cells:
        if kind == "code":
            compile(text, f"<{variant}>", "exec")
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out_cells.append(cell)
    nb["cells"] = out_cells
    data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    assert len(data) < 400 * 1024, "Kaggle commit notebook size limit (keep < 400 kB)"
    out = NBDIR / out_name
    pushed = lambda f: f.exists() and any(sha(f.read_bytes()).startswith(x) for x in PUSHED)    # rev 2: 8-hex prefixes
    if pushed(out):
        out = out.with_name(out.stem + "_REBUILD.ipynb")
        assert not pushed(out), out
    out.write_bytes(data)
    print(out.name, sha(data)[:8], f"{len(data) / 1024:.0f} kB", "| cells", len(out_cells))
    return out


if __name__ == "__main__":
    for v in (sys.argv[1:] or ["NB1", "NB2", "NB1SMOKE", "NB2SMOKE"]):
        build(v)
