"""E31 Flash-Next DIRECT 30 (user 2026-11-01): a GPU measurement notebook = the E28 K100 EVAL120 6 h notebook (sha d3d132da, the
Merlin vLLM-0.32 klee100 3bpw-MTP boot: TP2 x PP2, FP8 KV + static QSA scales, MTP K3, Q-defer, I8, 24 seqs, pin 7,933,526,016)
with Duck replaced by the proven Qwen3.8-27B DIRECT method (v1.4.4 / E10 ORDER2) on 30 eval tasks. Derived from the E28 K100
NOTEBOOK FILE (read, sha-pinned, never written): every cell is copied byte for byte except the marked "E31" edits:

  D1 (CELL 1)  the editable line E31_CFG = {...} at the top (plain Python literal, outside every packed / sha-locked text: the
               user changes think / answer / window_min / workers / temperature without a rebuild); E31_BUDGET_MIN = boot 15 +
               window_min + reserve 25; the deadman = E31_BUDGET_MIN * 60 (HARD_STOP + 5 min, the K100Q rule; the built default
               12,600 s as a literal for the kits); NB5_DEFAULT_PROFILE / K100_PROFILES = E31DIRECT30 / RERUN_PROD
  D2 (CELL 1)  E31 defs (the 30 tasks in 3 groups, E31_LEVERS = E28_LEVERS without e28_sched: Duck stays the original,
               the E28 report explains it) + the profile row E31DIRECT30 (source eval, task_list = the 30, one phase
               Q_E31_FN_DIRECT30 with the fixed window E31_WINDOW_SEC = a cap) replacing E28EVAL120; RERUN_PROD = E28's row with
               E31_LEVERS (the hidden rerun = every test task through the same direct solver, the E15B writer)
  D3 (CELL 1)  ONE sha-pinned lzma + base85 literal with e28k100_lib.py (unchanged, E28 report / I8) + e31_direct.py (the
               direct solver) -> E28L, E31D; the unused E28 scheduler module is dropped (size); CONFIG["nb5"]["e31"]
  D4 (NEW CELL 7b, between CELL 6/7 and CELL 8) E12A.solve_all_r6 -> the direct solver (same signature / return / record
               contract), E12A.build_submission -> the direct records' own ORDER2 make_attempts (other tasks: Duck's)
  D5 (markdown) the E31 header
CELL 0 (E18 R2 verbatim), CELL 3, the E10 task cell, CELL 5 (engine + gates), CELL 6/7, CELL 8, CELL 9: unchanged.

usage: python build_e31_direct.py   -> ../Q27-XHIGH-E31_FN_DIRECT30.ipynb + e31_code_export.py + e31_build_facts.json + the GPU
       push folder D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e31-fn-direct30__esfersami50/ (nothing is pushed)
"""
import ast
import base64
import hashlib
import json
import lzma
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E28NB_DIR = Path(r"D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks")
E28B = E28NB_DIR / "Build"
E28_NB = E28NB_DIR / "Q27-XHIGH-E28_K100_EVAL120_6H.ipynb"
E28_FACTS = E28B / "e28k100_build_facts.json"
E28_EXPORT = E28B / "e28k100_code_export.py"
E28_SHA = "d3d132da55ca9f4901404f0b656664af8947a53196a1ea89ccb6fe1c809e04c1"
E28_MODULES_SHA = "d4165f6c859b32ed2f2438d92290b6ba1f7093136196497b94238d6c2836c241"
LIB = E28B / "e28k100_lib.py"
LIB_SHA = "ff923ef252dc3a85ef5e7f5de8af1b777a8ee3a76317f2eb5614220434c2511b"
DIRECT = HERE / "e31_direct.py"
OUT_NB = HERE.parent / "Q27-XHIGH-E31_FN_DIRECT30.ipynb"
EXPORT = HERE / "e31_code_export.py"
FACTS = HERE / "e31_build_facts.json"
SLUG = "q27-xhigh-e31-fn-direct30"
OWNER = "esfersami50"                          # the klee100 inputs are shared with sami's account (E28 K100 / K100Q ran there)
PUSH = Path(r"D:/ARC AGI 2/_scratch/push_e25") / f"{SLUG}__{OWNER}"
OWNER2 = "foysalemonshanto"
PUSH2 = Path(r"D:/ARC AGI 2/_scratch/push_e25") / f"{SLUG}__{OWNER2}"
DEFAULT_PROFILE = "E31DIRECT30"
PHASE = "Q_E31_FN_DIRECT30"
PUSHED: set = set()   # sha256 of every E31 notebook that ran on Kaggle (never overwrite one: rebuild -> *_REBUILD)
CFG = {"think": 104000, "answer": 8192, "window_min": 170, "workers": 24, "temperature": 0.6, "drain_min": 23}   # user 2026-11-01 (+ drain_min: audit E31-3)
CFG_LINE = "E31_CFG = {'think': 104000, 'answer': 8192, 'window_min': 170, 'workers': 24, 'temperature': 0.6, 'drain_min': 23}"
BOOT_MIN, RESERVE_MIN, HARD_MARGIN_MIN = 15, 20, 5            # audit E31-7: ~205 min total for a 170-min window
BUDGET_MIN = BOOT_MIN + CFG["window_min"] + RESERVE_MIN       # 205
DEADMAN_SEC = BUDGET_MIN * 60                                 # 12,300 = HARD_STOP (200 min) + 5 min
assert eval(CFG_LINE.split("=", 1)[1]) == CFG and BUDGET_MIN == 205 and DEADMAN_SEC == (BUDGET_MIN - HARD_MARGIN_MIN) * 60 + 5 * 60
# audit E31-7: the window math of nb5_phase (window = min(fixed, HARD_STOP - E12_STOP_BEFORE_HARD_SEC 180 - now - 60)) with the
# boot at its E28 K100 real time (window start +14.65 min) and with one failed boot try (+13 min, the E28 plan)
E28_WINDOW_START_MIN, FAILED_TRY_MIN = 879 / 60, 13


def window_room_min(start_min, budget_min=BUDGET_MIN):
    return (budget_min - HARD_MARGIN_MIN) - 180 / 60 - 60 / 60 - start_min


assert window_room_min(E28_WINDOW_START_MIN) >= CFG["window_min"] and window_room_min(BOOT_MIN) >= CFG["window_min"], window_room_min(BOOT_MIN)
WINDOW_PLAN = {f"start_{s}_min": round(min(CFG["window_min"], window_room_min(s)), 1) for s in (BOOT_MIN, round(E28_WINDOW_START_MIN, 2),
                                                                                                  BOOT_MIN + FAILED_TRY_MIN)}


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, ("E31 " + label, text.count(old))
    return text.replace(old, new)


def pack_lzma(b):
    """= build_e26_k100_quality.pack_lzma (klee100 LOAD's pack): lzma (preset 9 | extreme) + b85."""
    return base64.b85encode(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


def packed_lzma(text, label):
    """= build_e26_k100_quality.packed_lzma: the sha-checked lzma + b85 literal (copied verbatim)."""
    h = sha(text.encode("utf-8"))
    return (f'(lambda _t: _t if __import__("hashlib").sha256(_t.encode()).hexdigest() == "{h}" else '
            f'(_ for _ in ()).throw(RuntimeError("{label}: embedded source sha256 mismatch")))('
            f'__import__("lzma").decompress(__import__("base64").b85decode("{pack_lzma(text.encode("utf-8"))}")).decode("utf-8"))')


def unpack_literal(expr):
    """Evaluate one packed literal expression (a lambda over lzma + b85, sha-checked) -> its text."""
    return eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<packed>", "eval"))


# ------------------------------------------------------------------ the base: the E28 K100 notebook file
E28_RAW = E28_NB.read_bytes()
assert sha(E28_RAW) == E28_SHA, "the E28 K100 notebook on disk changed"
E28_JSON = json.loads(E28_RAW.decode("utf-8"))
assert (json.dumps(E28_JSON, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8") == E28_RAW, "E28 JSON form"
BASE = [("".join(c["source"]), c["cell_type"]) for c in E28_JSON["cells"]]
assert [k for _, k in BASE] == ["markdown"] + ["code"] * 8
BASE_CODE = [t for t, k in BASE if k == "code"]
assert BASE_CODE[0].startswith("# ==================================================================== CELL 0") and "CELL 6/7" in BASE_CODE[5].splitlines()[0]


def readable(code):
    """= the E28 kit's E28_READABLE: CELL 6/7 carries e21_cell7 packed; the readable cells put its text back."""
    c = code[5]
    i = c.index("# ---- E28 K100 (size: the 400-kB commit limit)")
    call = next(n for n in ast.walk(ast.parse(c)) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
    body = unpack_literal(ast.get_source_segment(c, call.args[0].args[0]))
    return code[:5] + [c[:i] + body] + code[6:]


assert "\n\n".join(readable(BASE_CODE)) == E28_EXPORT.read_bytes().decode("utf-8"), "the E28 readable cells != the E28 export"
LIB_TEXT = LIB.read_bytes().decode("utf-8")
assert sha(LIB_TEXT) == LIB_SHA and "\r" not in LIB_TEXT
_m_line = next(l for l in BASE_CODE[1].splitlines() if l.startswith("E28_SRC = json.loads("))
_E28_MODULES_TEXT = unpack_literal(_m_line[len("E28_SRC = json.loads("):-1])
assert sha(_E28_MODULES_TEXT) == E28_MODULES_SHA and json.loads(_E28_MODULES_TEXT)["e28k100_lib.py"] == LIB_TEXT
DIRECT_TEXT = DIRECT.read_bytes().decode("utf-8")
assert "\r" not in DIRECT_TEXT, "e31_direct.py must be LF"
compile(DIRECT_TEXT, "e31_direct.py", "exec")
sys.path.insert(0, str(HERE))
import e31_direct as E31D   # noqa: E402
assert E31D.CFG_DEFAULT == CFG and len(E31D.TASKS30) == 30 == len(set(E31D.TASKS30))
# audit E31-3: the drain reserve covers the largest live answer (max test inputs x answer at ~18 tok/s per stream) and the
# phases stay ordered inside the window (admission stop T - 12 min, drain T - drain_min, abort T - 1 min)
_EVCH = json.loads(Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
                        r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_challenges.json").read_text(encoding="utf-8"))
MAX_TESTS = max(len(_EVCH[t]["test"]) for t in E31D.TASKS30)
DRAIN_NEED_MIN = E31D.drain_need_min(MAX_TESTS, CFG["answer"])
assert MAX_TESTS == 3 and DRAIN_NEED_MIN <= CFG["drain_min"] < CFG["window_min"] - 1, (MAX_TESTS, DRAIN_NEED_MIN)
assert E31D.HARD_BEFORE_END_SEC < CFG["drain_min"] * 60 and E31D.ADMIT_STOP_BEFORE_END_SEC < CFG["window_min"] * 60
MODULES = {"e28k100_lib.py": LIB_TEXT, "e31_direct.py": DIRECT_TEXT}
MODULES_JSON = json.dumps(MODULES, sort_keys=True, separators=(",", ":"))
# the per-task reference table embedded in e31_direct.py == the Exp 28 results file (the 30 rows)
_REFMD = (E28NB_DIR.parent / "Experiment Results" / "EVAL120_K100_W4_Direct_Duck_per_task_2026-11-01.md").read_text(encoding="utf-8")
_MARK = {"\u2705": 1.0, "\u25d0": 0.5, "\u274c": 0.0}
_REF = {}
for _l in _REFMD.splitlines():
    _c = [x.strip() for x in _l.strip().strip("|").split("|")]
    if len(_c) == 6 and _c[0].isdigit():
        _REF[_c[1]] = [_MARK[x] for x in _c[2:]]
assert len(_REF) == 120 and {t: _REF[t] for t in E31D.TASKS30} == E31D.REF_TABLE, "REF_TABLE != the Exp 28 per-task file"


# ------------------------------------------------------------------ E31 edits
E31_TOP = f'''{CFG_LINE}
# ---- E31 (user 2026-11-01): the line above is the ONE place to change this run (no rebuild): think = the phase-1 cap
# (>= 60,000, the user floor), answer = phase-2 tokens per test input (x the task's test inputs), window_min = the solve-window
# CAP (the direct solver returns as soon as every task is done; never a wait), workers = client streams (<= the 24 server
# seqs), temperature (ORDER2: 0.6), drain_min = the forced-close reserve at the window end (covers 3 test inputs x answer at
# ~18 tok/s per stream; inside the window: no new task from T - 12 min, every request aborted at T - 1 min). The budget
# follows it: boot ~{BOOT_MIN} min + window_min + a {RESERVE_MIN}-min reserve (nb5_phase's solve end HARD_STOP - 3 min and its 1-min
# room, hard margin 5, a late boot up to ~11 min); the deadman = the budget (HARD_STOP + 5 min).
if os.environ.get("E11_HARNESS") == "1" and os.environ.get("E31_CFG_JSON"):   # local / CPU-kit harness only (short fake runs)
    E31_CFG = dict(E31_CFG, **json.loads(os.environ["E31_CFG_JSON"]))
E31_BUDGET_MIN = {BOOT_MIN} + int(E31_CFG["window_min"]) + {RESERVE_MIN}
'''

E31_DEFS = r'''# ---- E31 (user 2026-11-01, Exp 31): Flash-Next DIRECT on 30 eval tasks = the proven ORDER2 direct method (one prompt per
# TASK, xhigh thinking, forced close, parse_grid / make_attempts) on THIS boot; the solver is e31_direct.py (CELL 7b rebinds
# E12A.solve_all_r6). E31_LEVERS = E28_LEVERS without e28_sched (Duck's arc2_duck.py stays the original payload file).
E31_GROUPS = __GROUPS__
E31_TASKS = [t for g in ("main", "partial", "control") for t in E31_GROUPS[g]]
E31_WINDOW_SEC = int(E31_CFG["window_min"]) * 60                # the solve-window cap (the solver returns when every task is done)
E31_LEVERS = {k: v for k, v in E28_LEVERS.items() if k != "e28_sched"}
assert len(E31_TASKS) == len(set(E31_TASKS)) == 30 and "e28_sched" not in E31_LEVERS and E31_LEVERS["top_rung_seqs"] == 24
assert 1 <= int(E31_CFG["workers"]) <= E31_LEVERS["top_rung_seqs"], E31_CFG
assert E11_HARNESS or int(E31_CFG["think"]) >= 60000, ("user floor: thinking >= 60k (approval needed below)", E31_CFG)
assert 1 <= float(E31_CFG["drain_min"]) < int(E31_CFG["window_min"]), E31_CFG
'''

E31_PROFILE = r'''    # E31: the 30 eval tasks (task_list; ORDER2 inside the direct solver), budget E31_BUDGET_MIN (HARD_STOP = budget - 5 min,
    # the deadman at the budget), ONE phase with the fixed window E31_WINDOW_SEC (a cap; shortened, never skipped, on a late boot)
    "E31DIRECT30": dict(_NB5_T, time_budget_sec=E31_BUDGET_MIN * 60, hard_margin_sec=5 * 60, source="eval", task_list=E31_TASKS,
                        quality_ab=True, equal_windows=True, fixed_window_sec=E31_WINDOW_SEC,
                        nb5_phases=[["Q_E31_FN_DIRECT30", "all", None, "on", "A", "fp8"]], **E31_LEVERS),
}
PROFILE_TABLE["RERUN_PROD"] = dict(PROFILE_TABLE["RERUN_PROD"], **E31_LEVERS)   # E31: the hidden rerun = NB6's (every task, 12 h - 3 min, E18 ladder) + klee100 + the direct solver
'''


# D6 (size, the 400-kB commit limit, as E28's K2b): the comment-only lines of two Duck-configuration regions of CELL 1 are
# dropped (Duck is not run here); every code line stays byte for byte (asserted)
SIZE_CUTS = [("# SMOKE = real evaluation tasks", "# ---- E26 K100 (lead 2026-11-01): klee100 3bpw-MTP on NB5"),
             ("# ---- E12: Duck REPL agent", "E12_SOLVE_WINDOW_SEC = P.get")]


def drop_comment_lines(c, start, end):
    a, b = c.index(start), c.index(end)
    assert c.count(start) == 1 and c.count(end) == 1 and a < b
    seg = c[a:b].splitlines(keepends=True)
    keep = [l for l in seg if not l.lstrip().startswith("#")]
    assert [l for l in seg if not l.lstrip().startswith("#")] == keep and keep
    return c[:a] + "".join(keep) + c[b:], sum(len(l.encode()) for l in seg) - sum(len(l.encode()) for l in keep)


SIZE_CUT_BYTES = {}


def cell1(c):
    for _s, _e in SIZE_CUTS:
        c, SIZE_CUT_BYTES[_s] = drop_comment_lines(c, _s if _s != "# SMOKE = real evaluation tasks" else _s, _e)
    head = "# ==================================================================== CELL 1 — NB5 configuration (E25 FN-Duck)\n"
    assert c.startswith(head)
    c = head + E31_TOP + c[len(head):]
    c = once(c, "# ---- E26 K100 (user 2026-11-01): the commit's ABSOLUTE DEADMAN (E28 K100: HARD_STOP 410 min + 5 = 415 min), armed first",
             "# ---- E26 K100 (user 2026-11-01): the commit's ABSOLUTE DEADMAN (E31: HARD_STOP + 5 min = E31_BUDGET_MIN), armed first",
             "D1 deadman comment")
    c = once(c, "K100_DEADMAN_SEC = 24900   # E28 K100: HARD_STOP (410 min) + 5 min\n",
             f"K100_DEADMAN_SEC = {DEADMAN_SEC}   # E31: the built default (E31_CFG window_min {CFG['window_min']} -> budget {BUDGET_MIN} min = HARD_STOP {BUDGET_MIN - HARD_MARGIN_MIN} + 5)\n"
             "K100_DEADMAN_SEC = int(E31_BUDGET_MIN * 60)   # E31: follows E31_CFG['window_min'] (HARD_STOP + 5 min, the K100Q rule)\n",
             "D1 deadman")
    c = once(c, 'NB5_DEFAULT_PROFILE = "E28EVAL120"   # E28 K100\n', f'NB5_DEFAULT_PROFILE = "{DEFAULT_PROFILE}"   # E31\n', "D1 profile")
    c = once(c, 'K100_PROFILES = ("E28EVAL120", "RERUN_PROD")   # E28 K100\n', f'K100_PROFILES = ("{DEFAULT_PROFILE}", "RERUN_PROD")   # E31\n', "D1 guard")
    c = once(c, "PROFILE_TABLE = {\n", E31_DEFS.replace("__GROUPS__", json.dumps(E31D.GROUPS)) + "PROFILE_TABLE = {\n", "D2 defs")
    i = c.index("    # E28 K100: EVAL120 (all 120 eval tasks, plain order), 415-min budget, hard margin 5 (HARD_STOP 410 = 6 h 50 min, the\n")
    j_old = 'PROFILE_TABLE["RERUN_PROD"] = dict(PROFILE_TABLE["RERUN_PROD"], **E28_LEVERS)   # E28 K100: the hidden rerun = NB6\'s (every task, 12 h - 3 min, E18 ladder) + klee100 + E28\n'
    j = c.index(j_old) + len(j_old)
    old = c[i:j]
    assert old.count('"E28EVAL120": dict(') == 1 and old.count("}\n") == 1, old[:300]
    c = c[:i] + E31_PROFILE + c[j:]
    m_old = _m_line + "\n"
    c = once(c, "# ---- E28 K100: the new code, ONE sha-pinned lzma + base85 literal: e28k100_lib.py (I8 read-back / static check, the\n"
                "# projected-peak note, the run report) + e28_sched_patch.py (the A / B / C scheduler, Build/E28_SCHED_API.md)\n" + m_old,
             "# ---- E31: ONE sha-pinned lzma + base85 literal: e28k100_lib.py (E28's, unchanged: I8 read-back / static check, the run\n"
             "# report) + e31_direct.py (the ORDER2 direct solver, Exp 31 Build); E28's scheduler module is not embedded (e28_sched off)\n"
             "E28_SRC = json.loads(" + packed_lzma(MODULES_JSON, "e31_modules") + ")\n", "D3 modules")
    c = once(c, "import e28k100_lib as E28L\n", "import e28k100_lib as E28L\nimport e31_direct as E31D   # E31\n", "D3 import")
    c = once(c, '    E28S_ERROR = "interim build: e28_sched_patch.py not embedded"\n',
             '    E28S_ERROR = "E31: the A / B / C scheduler is not embedded (e28_sched off: Duck is not run)"\n', "D3 sched note")
    c = once(c, 'print("NB5:", json.dumps(CONFIG["nb5"], default=str), flush=True)\n',
             'CONFIG["nb5"]["e31"] = {"cfg": dict(E31_CFG), "budget_min": E31_BUDGET_MIN, "window_sec": E31_WINDOW_SEC, "tasks": E31_TASKS,\n'
             '                        "groups": E31_GROUPS, "deadman_sec": K100_DEADMAN_SEC, "lib": E31D.LIB_VERSION,\n'
             '                        "levers": E31_LEVERS}   # E31\n'
             'print("NB5:", json.dumps(CONFIG["nb5"], default=str), flush=True)\n', "D3 config")
    return c


CELL7B = r'''# ==================================================================== CELL 7b — E31: the DIRECT solver replaces Duck (E12A.solve_all_r6)
# User 2026-11-01 (Exp 31): e21_cell6 looks E12A.solve_all_r6 up at call time, so this rebind swaps Duck for the proven ORDER2
# direct method (e31_direct.py, sha-pinned in CELL 1's E28_SRC literal) with Duck's signature, return and record contract:
# e21_cell6 (window, preflight, watchdog, the E15B writer callback), e21_cell7 (score, bundle), the nb5 phase record and the
# E28 report run unchanged. E12A.build_submission -> e31_build_submission, DIRECT-only (audit E31-2): the direct records' own
# ORDER2 make_attempts (attempt_1 = the model grid, attempt_2 = the test input / its mirror), the identity for a task without a
# record; no Duck candidate / CPU rule. The solver returns as soon as every task is done (E31_CFG window_min is a cap). Full
# thinking of every request: phase dir duck/<task>_b0/ (transcript.txt + token_ids.json).
E31_RUN = {"calls": [], "tokenizer": None}


def e31_tok():
    """The checkpoint's own tokenizer + chat template (CELL 8's NB5_LOCAL_TOK: the pinned klee100 tokenizer / template hashes
    verified), ORDER2's render (enable_thinking True, reasoning_effort xhigh) and decode; EOS ids from generation_config.json."""
    if not NB5_LOCAL_TOK.get("verified"):
        raise RuntimeError("E31: the Flash-Next tokenizer / template are not the pinned ones: " + str(NB5_LOCAL_TOK.get("why")))
    _hf, _tp = NB5_LOCAL_TOK["hf"], NB5_LOCAL_TOK["template"]
    _gp = next((p_ for p_ in (CKPT / "generation_config.json", Path(str(globals().get("K100_DIR") or "/nonexistent")) / "generation_config.json")
                if p_.is_file()), None)
    _eos = (json.loads(_gp.read_text(encoding="utf-8")).get("eos_token_id") if _gp is not None else None) or []
    t_ = {"render": lambda m_: _hf.apply_chat_template(m_, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                      reasoning_effort="xhigh", chat_template=_tp),
          "encode": lambda s_: TOK.encode(s_, add_special_tokens=False).ids,
          "decode": lambda i_, k_: TOK.decode(list(i_), skip_special_tokens=bool(k_)),
          "eos_ids": _eos if isinstance(_eos, list) else [_eos]}
    E31_RUN["tokenizer"] = {"generation_config": str(_gp), "eos_ids": t_["eos_ids"],
                            "stop_ids_equal_spec": t_["encode"](E31D.STOP_THINKING_TEXT) == E31D.FN_STOP_IDS,
                            "think_close_ids": t_["encode"]("</think>")}
    return t_


def e31_solve_all_r6(tasks, order, *, out_dir, deadline, base_url, model_id, on_branch_done=None, workers=10,
                     capacity_probe=None, **_rest):
    """E12A.solve_all_r6's signature: Duck-only keyword arguments are recorded (ignored_args), never used."""
    _w = min(int(E31_CFG["workers"]), int(E12_RUN["streams"]))   # merge list 3: the engine's streams (DIRECT runs no sandbox:
    # the isolation-gated Duck `workers` argument is recorded, not used)
    # audit E31-6: the client talks to THIS notebook's booted server only (E28's SGL_URL, port NB5_PORT; never E18's 8011);
    # e21_relaunch keeps the port, so the URL holds across a relaunch; the E15B watchdog is the one lifecycle owner
    _srv_url = str(getattr(E11_SERVER, "url", "") or "").rstrip("/")
    assert SGL_URL and SGL_URL.rstrip("/") == _srv_url and SGL_URL.rstrip("/").endswith(f":{NB5_PORT}") \
        and str(base_url).rstrip("/") == SGL_URL.rstrip("/") + "/v1", ("E31: base_url != the booted server", SGL_URL, _srv_url, base_url, NB5_PORT)
    E31_RUN["calls"].append({"tasks": len(tasks), "workers_arg": int(workers), "workers": _w, "base_url": base_url,
                             "model_id": model_id, "deadline_in_sec": round(deadline - time.monotonic(), 1), "ignored": sorted(_rest)})
    return E31D.solve_all_direct(tasks, order, out_dir=out_dir, deadline=deadline, base_url=SGL_URL, model_id=model_id,
                                 on_branch_done=on_branch_done, workers=_w, capacity_probe=capacity_probe, cfg=E31_CFG,
                                 tok=e31_tok(), solutions=SOLUTIONS, report_dir=BASE_OUT, server_down=E12A.SERVER_DOWN,
                                 metrics=e21_counters, sched_last=E12A.SCHED_LAST, sampler_path=OUT / "nb5_sampler.jsonl",
                                 t0_pc=NOTEBOOK_T0, harness=E11_HARNESS, ignored=sorted(_rest))


if not hasattr(E12A, "_e31_orig_solve_all_r6"):      # once (a re-run of this cell never wraps twice)
    E12A._e31_orig_solve_all_r6, E12A._e31_orig_build_submission = E12A.solve_all_r6, E12A.build_submission


def e31_build_submission(tasks, records):
    """audit E31-2: DIRECT-only (no Duck candidate, no harvested program, no CPU train-exact rule): the direct record's ORDER2
    make_attempts per test input; a task without a direct record = the identity (ORDER2 write_submission)."""
    return E31D.build_submission_direct(tasks, records)


E12A.solve_all_r6, E12A.build_submission = e31_solve_all_r6, e31_build_submission
CONFIG["e31_solver"] = {"lib": E31D.LIB_VERSION, "solve_all_r6": "e31_solve_all_r6", "build_submission": "e31_build_submission"}
print("E31: E12A.solve_all_r6 -> the direct solver (", E31D.LIB_VERSION, ") | E31_CFG", json.dumps(E31_CFG), "| budget",
      E31_BUDGET_MIN, "min | deadman", K100_DEADMAN_SEC, "s", flush=True)
'''

MD0 = f"""# E31 Flash-Next DIRECT 30 · klee100 3bpw-MTP (E28 K100 boot) · the ORDER2 direct method · {DEFAULT_PROFILE}

- **Question:** the proven Qwen3.8-27B DIRECT method (v1.4.4 / E10 ORDER2: russcore's prompt, xhigh thinking, forced close,
  parse_grid / make_attempts) on Flash-Next, on 30 eval tasks (16 main = solved by another method but not K100, 8 partial,
  6 control), concurrency 24; compare per task vs K100 / W4 / Direct-27B / Duck (Exp 28 per-task table).
- **Engine = E28 K100, unchanged** (sha d3d132da): Merlin vLLM 0.32, klee100 3bpw-MTP, TP2 x PP2, FP8 KV + static QSA scales,
  MTP K3, Q-defer, I8, pin 7,933,526,016 B, 24 seqs, the relaunch watchdog, the commit deadman. Duck is not run (no e28_sched).
- **Direct:** one prompt per TASK (multi-test: labeled grid blocks), phase 1 <= `think`, forced close + phase 2 <= `answer` x tests,
  ORDER2 order and seeds, drain 8 min before the window cap; the solver returns as soon as every task is done.
- **Edit `E31_CFG` at the top of CELL 1** (think / answer / window_min / workers / temperature): no rebuild. Default budget
  {BUDGET_MIN} min, deadman {DEADMAN_SEC // 60} min (commit only). Dual mode (proven): rerun = every hidden task through the same
  solver, CELL 0 fallback, E15B writer. Report: `q27x/e31_direct_report.json`, `q27x/e31_direct_records.jsonl`, `q27x/nb5_report.json`.
"""


_E28_ROW_LINES = ['    "E28EVAL120": dict(_NB5_T, time_budget_sec=415 * 60, hard_margin_sec=5 * 60, source="eval", n_tasks=120,',
                  '                       quality_ab=True, equal_windows=True, fixed_window_sec=E28_WINDOW_SEC,',
                  '                       nb5_phases=[["Q_E28_K100_EVAL120", "all", None, "on", "A", "fp8"]], **E28_LEVERS),', '}']


def cells():
    code = list(BASE_CODE)
    code[1] = cell1(code[1])
    _nc = lambda s: [l for l in s.splitlines() if not l.lstrip().startswith("#")]
    _base_nc, _new_nc = _nc(BASE_CODE[1]), _nc(code[1])
    assert all(l in _new_nc or l.startswith(("K100_DEADMAN_SEC = ", "NB5_DEFAULT_PROFILE = ", "K100_PROFILES = ", "E28_SRC = json.loads(",
                                               "PROFILE_TABLE[\"RERUN_PROD\"] = dict(", "    E28S_ERROR = \"interim"))
               or "E28EVAL120" in l or l in _E28_ROW_LINES for l in _base_nc), "a CELL 1 code line was lost"
    for i, t in enumerate(code):
        if i != 1:
            assert t == BASE_CODE[i]
    out = [("markdown", MD0)] + [("code", t) for t in code[:6]] + [("code", CELL7B)] + [("code", t) for t in code[6:]]
    return out


def kernel_metadata():
    m = dict(json.loads(E28_FACTS.read_text(encoding="utf-8"))["kernel_metadata"])
    m.update(id=f"{OWNER}/{SLUG}", title="Q27 Xhigh E31 Fn Direct30", code_file=f"{SLUG}.ipynb")
    return m


def notebook_bytes(cl):
    nb = {"cells": [], "metadata": E28_JSON["metadata"], "nbformat": E28_JSON["nbformat"], "nbformat_minor": E28_JSON["nbformat_minor"]}
    for kind, text in cl:
        assert "\r" not in text
        if kind == "code":
            compile(text, "<e31>", "exec")
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    return (json.dumps(nb, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


# the E28 JSON writer reproduces the E28 notebook from its own cells (the same form is used here)
assert notebook_bytes([(k, t) for t, k in BASE]) == E28_RAW, "the E31 JSON writer does not reproduce the E28 notebook"


def build():
    cl = cells()
    data = notebook_bytes(cl)
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    code = [t for k, t in cl if k == "code"]
    assert code[1].splitlines()[1] == CFG_LINE, "the E31_CFG line must be the first code line of CELL 1"
    assert [l for l in code[1].splitlines() if l.startswith("E31_CFG = {")] == [CFG_LINE]
    out = OUT_NB
    if out.exists() and sha(out.read_bytes()) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
        assert not (out.exists() and sha(out.read_bytes()) in PUSHED), out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / f"{SLUG}.ipynb").write_bytes(data)
    (PUSH / "kernel-metadata.json").write_bytes((json.dumps(kernel_metadata(), indent=2) + "\n").encode("utf-8"))
    # a second push folder for foysalemonshanto (user 2026-11-02: esfersami50's weekly GPU quota is used up): the same notebook,
    # the same inputs (the Models / datasets are foysalemonshanto's own), only the kernel id differs
    PUSH2.mkdir(parents=True, exist_ok=True)
    (PUSH2 / f"{SLUG}.ipynb").write_bytes(data)
    (PUSH2 / "kernel-metadata.json").write_bytes((json.dumps(dict(kernel_metadata(), id=f"{OWNER2}/{SLUG}"), indent=2) + "\n").encode("utf-8"))
    export = "\n\n".join(readable(code))
    EXPORT.write_bytes(export.encode("utf-8"))
    f28 = json.loads(E28_FACTS.read_text(encoding="utf-8"))
    facts = dict(f28)
    facts.update({
        "notebook": str(out), "notebook_sha256": sha(data), "bytes": len(data), "cells": len(cl), "json_format": "compact",
        "export": str(EXPORT), "export_sha256": sha(export), "push": str(PUSH), "kernel_metadata": kernel_metadata(),
        "default_profile": DEFAULT_PROFILE, "phase": PHASE, "interim": False,
        "base_cell_sha256": [sha(c) for c in BASE_CODE],
        "modules": {n: sha(t) for n, t in MODULES.items()}, "modules_json_sha256": sha(MODULES_JSON),
        "module_paths": {"e28k100_lib.py": str(LIB), "e31_direct.py": str(DIRECT)},
        "window_sec": CFG["window_min"] * 60, "budget_min": BUDGET_MIN, "hard_margin_min": HARD_MARGIN_MIN, "deadman_sec": DEADMAN_SEC,
        "e31": {"cfg": CFG, "cfg_line": CFG_LINE, "boot_min": BOOT_MIN, "reserve_min": RESERVE_MIN, "tasks": E31D.TASKS30,
                "groups": E31D.GROUPS, "lib": E31D.LIB_VERSION, "direct_sha256": sha(DIRECT_TEXT), "cell7b_sha256": sha(CELL7B),
                "base_e28": {"notebook": str(E28_NB), "sha256": E28_SHA, "export_sha256": sha(E28_EXPORT.read_bytes())},
                "edits": ["D1 E31_CFG + budget + deadman (CELL 1)", "D2 E31 defs + E31DIRECT30 row + RERUN_PROD E31_LEVERS (CELL 1)",
                          "D3 modules literal e28k100_lib + e31_direct (CELL 1)", "D4 NEW CELL 7b (solver + build_submission rebind)",
                          "D5 markdown", "D6 size: comment-only lines of two Duck-config regions of CELL 1 dropped"],
                "size_cut_bytes": SIZE_CUT_BYTES,
                "timeline_min": {"boot": BOOT_MIN, "window_cap": CFG["window_min"], "reserve": RESERVE_MIN, "budget": BUDGET_MIN,
                                 "hard_stop": BUDGET_MIN - HARD_MARGIN_MIN, "solve_end_cap": BUDGET_MIN - HARD_MARGIN_MIN - 3,
                                 "deadman": DEADMAN_SEC / 60, "window_by_start": WINDOW_PLAN,
                                 "in_window_phases": {"admit_stop_before_end_min": 12, "drain_before_end_min": CFG["drain_min"],
                                                      "abort_before_end_min": 1, "drain_need_min_3_tests": round(DRAIN_NEED_MIN, 2)}}},
        "push_order": ["0. the tf-ref dataset version E28 K100 / K100Q read (k100q_payload_emb.json, k100q_report.py, nb4_ppack.py) "
                       "must be the mounted one (nothing new to push)", "1. the CPU kits (esfersami50)", "2. the GPU kernel (user go only)"]})
    facts.pop("hard_stop_plan", None)
    facts.pop("token_plan", None)
    FACTS.write_bytes((json.dumps(facts, indent=1, default=str) + "\n").encode("utf-8"))
    print("wrote", out.name, "sha256", sha(data), "|", len(data), "bytes |", len(cl), "cells | export", sha(export)[:12],
          "| direct", sha(DIRECT_TEXT)[:12], flush=True)
    return out


if __name__ == "__main__":
    build()
