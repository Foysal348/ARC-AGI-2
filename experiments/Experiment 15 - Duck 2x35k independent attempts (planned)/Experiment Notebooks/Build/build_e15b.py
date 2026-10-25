"""Experiment 15 notebook builder: E15B = config B (user 2026-10-24): 2 x 35k independent Duck attempts per task, attempt 1
on every task first, 16 real streams (Exp 15 Notes/02). Nothing of Experiments 12 / 14 is written or changed.

How it is built:
- build_e14.py (the E14S submission notebook, LB 22.78) is EXECUTED IN MEMORY up to the line where it would write its
  notebook; it in turn runs the E12 builder in memory. Its notebook dict and payloads are taken from that run.
- The Duck package = the E14S package with three files replaced from ../../src/duck_b/ (arc2_duck.py, the vendored
  tool_agent.py and python_tool_sandbox.py); it goes to a NEW payload .bin (name carries its sha) in this experiment's
  payload folder, next to the unchanged E14 modules .bin.
- B changes are applied to the cell strings with once() (each patch must match exactly once).
    python build_e15b.py                # E15B SUBMIT: commit = a GPU smoke at 16 streams; rerun = every hidden task
    python build_e15b.py --eval120g4b   # E15B EVAL120G4B: commit = all 120 eval tasks at the hidden per-task share
"""
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E15 = HERE.parents[1]
SRC_B = E15 / "src" / "duck_b"
E14_BUILDER = Path(r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Notebooks/"
                   r"Build/build_e14.py")
R3I = "--r3i" in sys.argv       # rev 3i (user 2026-10-25): R + stage 3, EVAL120 in a 6-h notebook
# rev 3j (user 2026-10-25): two notebooks, every task keeps its attempt 2. --r3j=1: attempt 1 -> attempt 2 (unsolved) ->
# attempt 3 (unsolved) -> attempt 2 (accepted) -> attempt 4+; --r3j=2: attempt 1 -> attempt 2 (unsolved) -> attempt 2
# (accepted) -> attempt 3+ (unsolved)
R3J = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--r3j=")), "")
assert R3J in ("", "1", "2"), R3J
EVAL120G4B = "--eval120g4b" in sys.argv or R3I or bool(R3J)
VARIANT = "E15B"
# rev 3f / 3g (user 2026-10-25): per-attempt budgets from --budgets=A1,A2 (default 45k + 30k; also 40k + 35k)
_BUD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--budgets=")), "45000,30000")
ATTEMPT_TOKENS, ATTEMPT_TOKENS_B1 = (int(x) for x in _BUD.split(","))
assert 0 < ATTEMPT_TOKENS_B1 <= ATTEMPT_TOKENS
BTAG = f"{ATTEMPT_TOKENS // 1000}K{ATTEMPT_TOKENS_B1 // 1000}K"
# the commit run (20 tasks, both attempts): window / budget in minutes per variant (Notes/08)
COMMIT_WINDOW_MIN, COMMIT_BUDGET_MIN = {"45K30K": (92, 115), "40K35K": (92, 115),   # rev 3h: Auditor 4 #3
                                         # 50K45K (user 2026-10-25): measured 45K30K solve 50 min; worst path = 50k a1
                                         # (~2,000 s at budget) + a1 of the last 4 + 45k a2 (~1,800 s) ~ 97 min
                                         "50K45K": (105, 128)}[BTAG]
REV = ("R3K" + R3J) if R3J else "R3I" if R3I else "R3H"     # rev 3k (2026-10-25): the R3J orders + all fixes + W1     # rev 3h: a new name, so the pushed rev 3g notebooks stay as run
NAME = (f"Q27-XHIGH-E15B_EVAL120_{BTAG}_S16_{REV}_DUCK_SGLANG.ipynb" if (R3I or R3J)
        else f"Q27-XHIGH-E15B_EVAL120G4B_{BTAG}_S16_{REV}_DUCK_SGLANG.ipynb" if EVAL120G4B
        else f"Q27-XHIGH-E15B_SUBMIT_{BTAG}_S16_{REV}_DUCK_SGLANG.ipynb")
DEFAULT_PROFILE = "E15BEVAL120R3I" if (R3I or R3J) else "E15BEVAL120G4B" if EVAL120G4B else "E15B"
# the payload dataset of the account that runs it (user 2026-10-24: Sami's account); the notebook finds the files by
# name + sha under /kaggle/input, the slug is only shown in the error message
PAYLOAD_SLUG = "esfersami50/q27-e15b-payload"
PAYLOAD_DIR = HERE / "payload"
PUSHED: set = {"0f0d6f675dfcd44a412da63b85ce591c96c32d32880cc2da647c49037bd0ebad",   # 2X35K SUBMIT, foysal
          "560ed8b886e5c0e6d8d0a03ef7c2e635001429edc2caf716e9f5363e1779b14f",   # 40K35K SUBMIT, foysal 2026-10-25
          "5d8069b99780561b580b7a96bf460ce16c5144c2e7dab2737e495b7a2fa7450f",
          "91b219bc147c72a0752966659b6687f898feea432b1e47ca6c52acd1ddc09211",
          "6baa9361c0e08666cca4835f677a6fb4dab7d45949a52cc75543a5384929ba25",   # R3K1 EVAL120, esfersami50 2026-10-25
          "c058d58be3cc7fa37e3a60e2986419b913ec25ef7a6456f1510700e9d9b3ecb0"}   # R3K2 EVAL120, hosen42 2026-10-25   # 50K45K R3H SUBMIT, esfersami50 2026-10-25   # 45K30K SUBMIT, esfersami50 2026-10-25   # shas of E15B notebooks run on Kaggle (memory: never rebuild a run notebook); 2026-10-25 SUBMIT commit on foysalemonshanto
NB_LIMIT = 400_000
STREAMS, SERVER_RUNNING, MAMBA = 16, 17, 68          # Notes/02 Rev 2c
# rev 3i: R (no attempt 2 after an accepted attempt 1) + stage 3 (attempts 3..8 on tasks with no accepted answer)
R3I_KEYS = (',\n        "E12_B_SKIP_B1_IF_ACCEPTED": 1, "E12_B_EXTRA_ATTEMPTS": 1, "E12_B_MAX_ATTEMPTS": 8'
            if R3I else
            ',\n        "E12_B_SKIP_B1_IF_ACCEPTED": 0, "E12_B_EXTRA_ATTEMPTS": 1, "E12_B_MAX_ATTEMPTS": 8, "E12_B_WEAK_COMMIT": 1, "E12_B_B1_ACC_MODE": ' + R3J
            if R3J else "")
RESERVE, WINDOW, TIMEOUT, MIN_ADMIT, KV_PER_REQUEST = 4096, 100000, 3600, 300, 50_000   # A5: min_admit 300; rev 3h: timeout 3,600 s (Auditor 4 #7)

# ------------------------------------------------------------------ 1. the E14S builder in memory
_src = E14_BUILDER.read_text(encoding="utf-8")
_cut = _src.rindex("\nout = HERE.parent / NAME") + 1      # the last one (build_e14 cuts the E12 builder with the same text)
_argv = sys.argv
sys.argv = ["build_e14.py"] + (["--eval120g4b"] if EVAL120G4B else [])
G = {"__file__": str(E14_BUILDER), "__name__": "e14_builder_in_memory"}
exec(compile(_src[:_cut], str(E14_BUILDER), "exec"), G)
sys.argv = _argv
assert G["VARIANT"] == "E14S" and not G["E16"] and not G["E15"] and not G["EFFORT"]
once = G["once"]
nb = G["nb"]
cells = ["".join(c["source"]) for c in nb["cells"]]
heads = [c.splitlines()[0] if c else "" for c in cells]
IDX = {k: next(i for i, h in enumerate(heads) if k in h) for k in
       ("CELL 1 ", "CELL 1b", "CELL 3 ", "E12: = E11-06 CELL 5", "CELL 6 ", "CELL 7 ", "CELL 7b")}

# ------------------------------------------------------------------ 2. the B Duck package (3 files replaced)
_old_zip = base64.b64decode(G["G"]["ZIP_B64"])
files = {}
with zipfile.ZipFile(io.BytesIO(_old_zip)) as z:
    for n in z.namelist():
        files[n] = z.read(n)
REPL = {"arc2_duck.py": SRC_B / "arc2_duck.py",
        "vendor/inference/agent/tool_agent.py": SRC_B / "tool_agent.py",
        "vendor/inference/agent/python_tool_sandbox.py": SRC_B / "python_tool_sandbox.py"}
OLD_SHA = {k: hashlib.sha256(files[k]).hexdigest() for k in REPL}
for k, p in REPL.items():
    files[k] = p.read_bytes()
NEW_SHA = {k: hashlib.sha256(files[k]).hexdigest() for k in REPL}
assert all(OLD_SHA[k] != NEW_SHA[k] for k in REPL), "a B file equals the E14S file"
assert OLD_SHA["arc2_duck.py"] == G["DUCK_SHA"]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for k in sorted(files):
        zi = zipfile.ZipInfo(k, date_time=(2026, 10, 24, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, files[k])
DUCK_BYTES = buf.getvalue()
DUCK_ZIP_SHA = hashlib.sha256(DUCK_BYTES).hexdigest()
DUCK_ZIP = f"q27x_e15b_duck_{DUCK_ZIP_SHA[:12]}.bin"
MOD_ZIP, MOD_ZIP_SHA, MOD_BYTES = G["PAYLOAD"]["q27x_e14_modules"]
B_SHA = NEW_SHA["arc2_duck.py"]

# ------------------------------------------------------------------ 3. CELL 3: the B package from the payload dataset
c = cells[IDX["CELL 3 "]]
c = once(c, 'q27x_payload("' + G["DUCK_ZIP"] + '", "' + G["DUCK_ZIP_SHA"] + '")',
         'q27x_payload("' + DUCK_ZIP + '", "' + DUCK_ZIP_SHA + '")', "B Duck zip")
for k in REPL:
    c = once(c, '"' + k + '": "' + OLD_SHA[k] + '"', '"' + k + '": "' + NEW_SHA[k] + '"', "manifest " + k)
c = once(c, "# LICENSE + NOTICE inside). The 20 files of inference/ are byte-identical to that commit (sha256 manifest below);",
         "# LICENSE + NOTICE inside). E15B: inference/agent/tool_agent.py (yield after the batch, ARC3_YIELD_AFTER_BATCH) and\n"
         "# python_tool_sandbox.py (sys shim / time / import messages, ARC2_SANDBOX_SYS_SHIM) carry two switched-off-by-default\n"
         "# patches (Exp 15 src/duck_b); the other 18 files are byte-identical to that commit (sha256 manifest below);",
         "package comment")
c = once(c, "    _st_task = test_challenges[sorted(test_challenges)[0]]",
         "    _st_task = E15B_SELFTEST_TASK      # E15B #17 (code audit M4): a fixed task, never the first hidden task",
         "fixed self-test task")
c = once(c, "_E12_MANIFEST = ", '''# E15B #17: the sandbox self-test task (identity on every pair): fixed, so an unusual first hidden task cannot fail it
E15B_SELFTEST_TASK = {"train": [{"input": [[1, 0], [0, 2]], "output": [[1, 0], [0, 2]]},
                                {"input": [[3, 3, 0]], "output": [[3, 3, 0]]}], "test": [{"input": [[4]]}]}
_E12_MANIFEST = ''', "self-test task literal")
cells[IDX["CELL 3 "]] = c

# ------------------------------------------------------------------ 4. CELL 1b: build record
c = cells[IDX["CELL 1b"]]
assert c.count(G["DUCK_SHA"]) == 2, c.count(G["DUCK_SHA"])
c = c.replace(G["DUCK_SHA"], B_SHA)
c = once(c, '"variant": "E14S"', '"variant": "' + VARIANT + '"', "variant")
c = once(c, '"''' + G["DUCK_ZIP"] + '''": "''' + G["DUCK_ZIP_SHA"] + '"', '"' + DUCK_ZIP + '": "' + DUCK_ZIP_SHA + '"',
         "payload record")
c = once(c, 'Q27X_PAYLOAD_SLUG = "' + G["PAYLOAD_SLUG"] + '"', 'Q27X_PAYLOAD_SLUG = "' + PAYLOAD_SLUG + '"', "slug")
# E15B #27 (code audit L7): the staging copy stops when the launch stops waiting for it (it used to run on to +45 min,
# reading the same files while SGLang loads them from the original paths)
c = once(c, "deadline=NOTEBOOK_T0 + 45 * 60, log=print, state=E14_STATE, dest_candidates=_e14_dest)",
         "deadline=max(NOTEBOOK_T0 + 10 * 60, min(NOTEBOOK_T0 + 45 * 60, SGL_GIVE_UP_AT - 20 * 60)), log=print, state=E14_STATE,\n"
         "                               dest_candidates=_e14_dest)   # E15B #27", "staging deadline")
c = once(c, '"submit_arm": None if E14_DO_MEASURE else "DFlash width 6 + CUDA graphs 1..13 (Notes/30)",',
         '"submit_arm": None if E14_DO_MEASURE else "DFlash width 6, 16 streams: mrr 17, graphs 1..17, mamba 68, '
         'min-free-slots-delay 1 (Exp 15 Notes/02)",', "submit arm label")
cells[IDX["CELL 1b"]] = c

# ------------------------------------------------------------------ 4b. CELL 2: tokenizer probe on a fixed task (#17)
_i2 = next(i for i, h in enumerate(heads) if "CELL 2" in h)
c = cells[_i2]
c = once(c, "        _t0 = test_challenges[sorted(test_challenges)[0]]",
         '        _t0 = {"train": [{"input": [[0, 1, 2], [3, 4, 5]], "output": [[5, 4, 3], [2, 1, 0]]}]}   # E15B #17: fixed',
         "tokenizer probe task")
cells[_i2] = c

# ------------------------------------------------------------------ 5. CELL 1: profiles, streams, B settings
c = cells[IDX["CELL 1 "]]
c = once(c, 'os.environ.get("Q27X_PROFILE", "' + G["DEFAULT_PROFILE"] + '")',
         'os.environ.get("Q27X_PROFILE", "' + DEFAULT_PROFILE + '")', "default profile")
c = once(c, '    "E14SEVAL120G4B": {', ('''    # E15B commit (user 2026-10-25, Exp 15 Notes/08): 20 eval tasks with BOTH attempts finishing = the 16 G4B tasks + the
    # first 4 by id of the tasks solved in 1-2 of the 3 EVAL120 G4B runs (2026-10-24). 40 attempts on 16 streams = 3 waves;
    # rev 3f / 3g: waves 16 x a1, then 4 x a1 + 12 x a2, then the last 4 x a2 (4 running = faster). 45k + 30k at 22 tok/s:
    # worst ~ 83 min -> window 92, budget 115; 40k + 35k: worst ~ 74 min -> window 85, budget 110 (READY took 12-13 min;
    # the solve deadline = hard stop - 180 s still clamps it)
    "E15B":      {"source": "eval", "task_list": ["136b0064", "898e7135", "135a2760", "b5ca7ac4", "4c3d4a41", "62593bfd",
                                                  "247ef758", "bf45cf4b", "f931b4a8", "65b59efc", "9385bd28", "cbebaa4b",
                                                  "2ba387bc", "1818057f", "45a5af55", "9aaea919",
                                                  "16de56c4", "1ae2feb7", "221dfab4", "269e22fb"],
                  "time_budget_sec": %d * 60, "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60,
                  "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60,
                  "solve_window_sec": %d * 60},
    # E15B EVAL120 gate 4b: every evaluation task at the hidden run's compute per task (as E14SEVAL120G4B)
    "E15BEVAL120G4B": {"source": "eval", "n_tasks": 120, "time_budget_sec": 6.5 * 3600, "admit_frac": 0.90,
                       "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
                       "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "hidden_equiv": True},''' + ('''
    # rev 3i (user 2026-10-25): all 120 eval tasks, exactly a 6-h notebook; the solve window = the hidden run's per-task
    # share (120 x (hidden solve end - start) / 240 ~ 5.8 h), clamped by this run's hard stop - 180 s (~5.7 h)
    "E15BEVAL120R3I": {"source": "eval", "n_tasks": 120, "time_budget_sec": 6.0 * 3600, "admit_frac": 0.90,
                       "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
                       "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "hidden_equiv": True},''' if (R3I or R3J) else "") + '''
    "E14SEVAL120G4B": {''') % (COMMIT_BUDGET_MIN, COMMIT_WINDOW_MIN), "E15B profiles")
c = once(c, "SGL_MAX_RUNNING = 12                # E11-04 W1",
         f"SGL_MAX_RUNNING = {STREAMS}                # E15B (Notes/02 Rev 2c): 16 real streams; was 12 (E11-04 W1", "streams")
c = once(c, "SGL_SERVER_MAX_RUNNING = 13         # rev 3c",
         f"SGL_SERVER_MAX_RUNNING = {SERVER_RUNNING}         # E15B: client 16 + 1 (Notes/30 nodelay crash: a 13th on buffers sized 12); rev 3c",
         "server running")
c = once(c, "SGL_KV_PER_REQUEST = 70_000         # concurrency <= KV // this",
         f"SGL_KV_PER_REQUEST = {KV_PER_REQUEST:_}         # E15B: the MEAN attempt context ~ 50k (attempt 1 at 45k ends at ~57-59k p90, max ~64k; attempt 2 less; Auditor 4); was 70k (KV // 70k = 12). concurrency <= KV // this",
         "kv per request")
c = once(c, 'E12_STREAMS = int(os.environ.get("E12_STREAMS", 12 if _E12_REV4 else 0))',
         f'E12_STREAMS = int(os.environ.get("E12_STREAMS", {STREAMS} if _E12_REV4 else 0))', "E12 streams")
c = once(c, 'E12_MAX_GEN_PER_BRANCH = int(os.environ.get("E12_MAX_GEN", 100000 if _E12_REV4 else 0)) or None',
         f'E12_MAX_GEN_PER_BRANCH = int(os.environ.get("E12_MAX_GEN", {ATTEMPT_TOKENS} if _E12_REV4 else 0)) or None   # E15B rev 3f: attempt 1 = {ATTEMPT_TOKENS:,} (attempt 2: E15B block)',
         "attempt budget")
c = once(c, '"E12_R6_FIRST_EXPOSURE": 28000, "E12_R6_B1_VIEW": "indexed", "E12_R6_MIN_ADMIT_SEC": 240,',
         f'"E12_R6_FIRST_EXPOSURE": 0, "E12_R6_B1_VIEW": "facts", "E12_R6_MIN_ADMIT_SEC": {MIN_ADMIT},   # E15B #3, #1, #49',
         "B scheduler settings")
c = once(c, '"E12_R6_FACTS_MAX_TOKENS": 800, "E12_R6_FAIR_SHARE": 1,',
         '"E12_R6_FACTS_MAX_TOKENS": 800, "E12_R6_FAIR_SHARE": 0,   # E15B #45: no chunks / pauses',
         "fair share off")
c = once(c, 'E12_DUCK_REQUEST_TIMEOUT = int(os.environ.get("E12_DUCK_REQUEST_TIMEOUT", 1800 if _E12_REV4 else 0)) or None',
         f'E12_DUCK_REQUEST_TIMEOUT = int(os.environ.get("E12_DUCK_REQUEST_TIMEOUT", {TIMEOUT} if _E12_REV4 else 0)) or None   # E15B #47',
         "request timeout")
c = once(c, "E12_SOLVE_WINDOW_SEC = P.get(\"solve_window_sec\")", f'''# E15B (user 2026-10-24, Exp 15 Notes/02; rev 3f 2026-10-25): config B = 2 independent attempts per task, attempt 1 =
# {ATTEMPT_TOKENS:,} and attempt 2 = {ATTEMPT_TOKENS_B1:,} generated tokens (attempt 1 on every
# task first, then attempt 2 with its own agent and the facts view); no pause / chunk / resume; request cap = remaining
# budget - {RESERVE:,} (a final commit turn); fixed {WINDOW:,}-token Duck window (no trim inside an attempt); 16 real streams
E15B = {{"E12_B_MODE": 1, "E12_B_ATTEMPT_TOKENS": E12_MAX_GEN_PER_BRANCH or {ATTEMPT_TOKENS}, "E12_B_RESERVE": {RESERVE},
        "E12_B_WINDOW": {WINDOW}, "E12_B_ATTEMPT_TOKENS_B1": {ATTEMPT_TOKENS_B1}{R3I_KEYS}}}
for _k, _v in E15B.items():
    E15B[_k] = int(os.environ.get(_k, _v))
    os.environ[_k] = str(E15B[_k])
assert E15B["E12_B_ATTEMPT_TOKENS"] == E12_MAX_GEN_PER_BRANCH, "the attempt budget and the session cap must agree"
assert 0 < E15B["E12_B_ATTEMPT_TOKENS_B1"] <= E15B["E12_B_ATTEMPT_TOKENS"], "attempt 2's budget (rev 3f)"
E12_SOLVE_WINDOW_SEC = P.get("solve_window_sec")''', "E15B block")
c = once(c, 'CONFIG["e12"] = {k: globals()[k] for k in ("E12_BRANCHES",', 'CONFIG["e15b"] = dict(E15B)\nCONFIG["e12"] = {k: globals()[k] for k in ("E12_BRANCHES",', "config e15b")
cells[IDX["CELL 1 "]] = c

# ------------------------------------------------------------------ 6. CELL 5: engine flags for 16 streams, launch fallback
c = cells[IDX["E12: = E11-06 CELL 5"]]
c = once(c, 'W1 = {"strategy": "extra_buffer_lazy", "mamba_cache": 56, "no_prefill_graph": True, "mem": "0.83"}',
         f'W1 = {{"strategy": "extra_buffer_lazy", "mamba_cache": {MAMBA}, "no_prefill_graph": True, "mem": "0.83"}}   # E15B: 16 x ~3 slots + radix',
         "mamba 68")
c = once(c, '''E14_SUBMIT_ARM = {"--cuda-graph-bs-decode": [str(b) for b in range(1, SGL_SERVER_MAX_RUNNING + 1)],
                  "--speculative-num-draft-tokens": ["6"]}''',
         '''E14_SUBMIT_ARM = {"--cuda-graph-bs-decode": [str(b) for b in range(1, SGL_SERVER_MAX_RUNNING + 1)],
                  "--speculative-num-draft-tokens": ["6"],
                  # E15B #36 (Exp 15 Notes/02): the DFlash min-free-slots delayer off ("<= 1 disables",
                  # managers/min_free_slots_delayer.py): with it the last stream waited while the others ran (EVAL120:
                  # 11.2 of 12 running, queue > 0 for 72 % of the time)
                  "--min-free-slots-delay": ["1"]}''', "delayer off")
c = once(c, '''    for k, v in E14_SUBMIT_ARM.items():
        out += [k] + list(v)
    return out''', '''    for k, v in E14_SUBMIT_ARM.items():
        out += [k] + list(v)
    if cfg.get("streams12"):     # E15B #16: the 12-stream fallback launch (graphs 1..13, mrr 13; the delayer stays off)
        out = e14_edit_flags_b(out, {"--max-running-requests": ["13"], "--cuda-graph-bs-decode": [str(b) for b in range(1, 14)]})
    assert "--enable-prefill-delayer" not in out
    return out


def e14_edit_flags_b(flags, set_):
    out, i = [], 0
    while i < len(flags):
        if flags[i] in set_:
            i += 1
            while i < len(flags) and not str(flags[i]).startswith("--"):
                i += 1
            continue
        out.append(flags[i])
        i += 1
    for k, v in set_.items():
        out += [k] + list(v)
    return out''', "fallback flags")
c = once(c, '''    if any(k in (tail or "").lower() for k in E11_OOM_MARKERS) and cfg["mem"] == "0.83" and "oom" not in used:
        return dict(cfg, mem="0.78"), "oom"''', '''    global SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING
    _oom = any(k in (tail or "").lower() for k in E11_OOM_MARKERS)
    if not cfg.get("streams12") and "streams12" not in used:   # E15B #16 (H2): ANY failed 16-stream launch
        # E15B #16: a 16-stream launch that ran out of memory or failed without a diagnosed cause relaunches at the
        # E14S 12-stream size (mamba 56, mrr 13, graphs 1..13; the delayer stays off); the capacity gate then sees 12
        SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING = 12, 13
        return dict(cfg, streams12=True, mamba_cache=56), "streams12"
    if _oom and cfg["mem"] == "0.83" and "oom" not in used:
        return dict(cfg, mem="0.78"), "oom"''', "streams12 fallback")
c = once(c, '''def e11_bring_up():
    global E11_SERVER''', '''def e11_bring_up():
    global E11_SERVER, SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING   # E15B: the 12-stream fallbacks''', "bring-up globals")
c = once(c, '''    caps["gate"], caps["concurrency"] = e11_capacity_gate(caps, cfg)
    E11_SGL["caps"] = caps''', '''    caps["gate"], caps["concurrency"] = e11_capacity_gate(caps, cfg)
    if not caps["gate"]["pass"] and not cfg.get("streams12") and SGL_GIVE_UP_AT - time.perf_counter() > 120:
        # E15B #16: the 16-stream server came up but failed the capacity gate (e.g. GPU headroom): one relaunch at the
        # E14S 12-stream size, the delayer still off
        E11_SGL["gate16"] = {"caps": dict(caps), "cfg": dict(cfg)}
        print("E15B: 16-stream capacity gate failed -> 12-stream relaunch:", json.dumps(caps["gate"]), flush=True)
        E11_SERVER.kill()
        _e11_gpu_free()
        SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING = 12, 13
        cfg = dict(cfg, streams12=True, mamba_cache=56)
        srv = E11Server(launch + e11_flags(cfg), env, "try_s12")
        srv.start()
        ok, why = srv.wait_ready(min(SGL_RELAUNCH_TIMEOUT_SEC, max(60.0, SGL_GIVE_UP_AT - time.perf_counter())))
        E11_SGL["attempts"].append({"try": "s12", "cfg": dict(cfg), "ready": ok, "why": why,
                                    "sec": round(time.perf_counter() - srv.t0, 1)})
        if not ok:
            srv.kill()
            raise E11Fail(f"12-stream relaunch after the 16-stream gate failed: {why}")
        E11_SERVER = srv
        E11_SGL["launch_flags"] = e11_flags(cfg)
        E14_VENV["cfg"] = dict(cfg)
        caps = e11_caps(E11_SERVER.log_text())
        caps["gate"], caps["concurrency"] = e11_capacity_gate(caps, cfg)
    E11_SGL["caps"] = caps''', "gate fallback")
c = once(c, '        return dict(cfg, strategy="extra_buffer", mamba_cache=70), "lazy"          # 5 slots per request -> 14 >= 12',
         '        return dict(cfg, strategy="extra_buffer", mamba_cache=5 * SGL_SERVER_MAX_RUNNING), "lazy"   # E15B: 5 slots x mrr',
         "lazy fallback")
c = once(c, '''    if cfg["no_prefill_graph"] and "graph" in t and "prefill" not in used:
        return dict(cfg, no_prefill_graph=False), "prefill"
    return None''', '''    if cfg["no_prefill_graph"] and "graph" in t and "prefill" not in used:
        return dict(cfg, no_prefill_graph=False), "prefill"
    if "w1" not in used:          # E15B #16 (H2): last resort, the plain W1 flags (width 8, the server's defaults)
        return dict(cfg, w1=True), "w1"
    return None''', "w1 fallback")
c = once(c, '''    f = _e11_flags_w1(cfg)
    if E14_DO_MEASURE:
        return f''', '''    f = _e11_flags_w1(cfg)
    if E14_DO_MEASURE or cfg.get("w1"):     # E15B #16: the W1 fallback launch keeps the plain W1 flags
        return f''', "w1 flags")
c = once(c, '        ENGINE_INFO.update(backend="sglang-0.5.21-int4-dflash2-w1", kv_tokens=_caps.get("kv_tokens"),',
         '        ENGINE_INFO.update(backend="sglang-0.5.21-int4-dflash2-" + ("w1" if (E14_VENV.get("cfg") or {}).get("w1") else "w6") '
         '+ f"-s{SGL_MAX_RUNNING}", kv_tokens=_caps.get("kv_tokens"),', "backend label")   # E15B #28
# E15B #24 (code audit M6): every install subprocess has a timeout (a hung pip would hang the notebook)
c = once(c, '        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)',
         '        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=300)', "venv timeout")
c = once(c, """                            "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts", "-q"],
                           capture_output=True, text=True)""", """                            "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts", "-q"],
                           capture_output=True, text=True, timeout=1800)""", "pip timeout")
c = once(c, """    site = Path(subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True).stdout.strip())""", """    site = Path(subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True, timeout=120).stdout.strip())""", "sysconfig timeout")
# E15B #33 (agent A §9): the retract counter matched the 'retraction_policy' server argument
c = once(c, '           "retract_lines": sum(1 for l in E11_SERVER.log_text().splitlines() if "retract" in l.lower()),',
         '           "retract_lines": sum(1 for l in E11_SERVER.log_text().splitlines() if "retract" in l.lower() '
         'and "retraction_policy" not in l and "server_args" not in l.lower()),', "retract counter")
cells[IDX["E12: = E11-06 CELL 5"]] = c

# ------------------------------------------------------------------ 7. CELL 6: fixed window, B summary, concurrency
c = cells[IDX["CELL 6 "]]
c = once(c, '''        E12_RUN["rev6_window"] = min(131072, int(E12_R6["E12_R6_KV_FRACTION"] * _kv / max(1, E12_RUN["streams"])))''',
         '''        # E15B #46 (Exp 15 Notes/02): a FIXED window, not KV / streams (16 streams -> ~54k would trim earlier than rev 6g);
        # an attempt's history stays <= ~50k, the KV admission gate (0.95) covers memory
        E12_RUN["rev6_window"] = min(131072, int(E15B["E12_B_WINDOW"]))''', "fixed window")
c = once(c, '''        if P.get("parity"):          # R6-X1 diagnostic profile: no solving (the fallback submission stays)''',
         '''        # E15B #15 (code audit H1): a server watchdog for the solve. A dead SGLang process (or /health failing for
        # 180 s) is relaunched with the same command (at most 2 times, only with >= 20 min left); while it is down the
        # branches wait (arc2_duck.SERVER_DOWN) instead of burning their consecutive-failure budget.
        E12_RUN["watchdog"] = {"relaunches": [], "health_fail_sec_max": 0, "old_logs": [], "tok_acc": 0.0, "tok_last": None,
                               "health_probes": 0}
        _e15b_solve_end = time.perf_counter() + max(0.0, _end - time.perf_counter())
        E15B_SOLVE_END_PC = _e15b_solve_end       # B1: per-branch submission writes stop here (the final write stays)
        _e15b_tok0 = _e15b_gen_tokens_total()
        E12_RUN["watchdog"]["tok_base"] = _e15b_tok0

        # A7 (Notes/07): the limits come from env only for the CPU harness's fault test; Kaggle uses the defaults
        _WD_POLL = float(os.environ.get("E15B_WD_POLL_SEC", "10"))
        _WD_FAIL = float(os.environ.get("E15B_WD_FAIL_SEC", "180"))
        _WD_MIN_LEFT = float(os.environ.get("E15B_WD_MIN_LEFT_SEC", str(20 * 60)))
        E12_RUN["watchdog"]["limits"] = {"poll_sec": _WD_POLL, "fail_sec": _WD_FAIL, "min_left_sec": _WD_MIN_LEFT}

        def _e15b_watchdog():
            global E11_SERVER, SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING
            _bad, _sz_last = None, {}
            _wd = E12_RUN["watchdog"]
            while not _E12_STOP.wait(_WD_POLL):
                _srv = E11_SERVER
                _dead = _srv is None or not _srv.alive()
                if not _dead:
                    _tk = _e15b_gen_tokens_total()      # B6: the last counter value of THIS server (it resets on relaunch)
                    if _tk is not None:
                        _wd["tok_last"] = _tk
                    try:     # B2 (auditor 2 F2 / F8): a growing server log = alive (a prefill storm logs batches, decode
                        _sz = os.path.getsize(_srv.log_path)     # logs every 40 steps); /health only when it is silent
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
                        E12_RUN["watchdog"]["health_fail_sec_max"] = max(E12_RUN["watchdog"]["health_fail_sec_max"],
                                                                         round(time.monotonic() - _bad))
                        if time.monotonic() - _bad < _WD_FAIL:
                            continue
                _left = _e15b_solve_end - time.perf_counter()
                _rec = {"t": round(time.perf_counter() - NOTEBOOK_T0), "dead": _dead, "left_sec": round(_left)}
                if len(E12_RUN["watchdog"]["relaunches"]) >= 2 or _left < _WD_MIN_LEFT:
                    _rec["action"] = "none (limit or < 20 min left)"
                    E12_RUN["watchdog"]["relaunches"].append(_rec)
                    print("E15B watchdog: server down, no relaunch:", json.dumps(_rec), flush=True)
                    return
                E12A.SERVER_DOWN.set()
                print("E15B watchdog: server down -> relaunch", json.dumps(_rec), flush=True)
                try:
                    try:
                        _tail = _srv.log_text()[-20000:].lower()
                    except Exception:
                        _tail = ""
                    try:
                        _srv.kill()
                    except Exception:
                        pass
                    _wd["old_logs"].append(str(_srv.log_path))
                    if _wd.get("tok_last") is not None and _wd.get("tok_base") is not None:   # B6
                        _wd["tok_acc"] += max(0.0, _wd["tok_last"] - _wd["tok_base"])
                    _wd["tok_base"], _wd["tok_last"] = 0.0, None
                    _e11_gpu_free()
                    # B3 (auditor 2 F3, auditor 3 R1): a 16-stream-specific crash (OOM / RuntimeError) or a second relaunch
                    # -> the 12-stream flags (the clients then queue up to 4 requests at the server)
                    _cfg0 = dict(E14_VENV.get("cfg") or {})
                    _cmd = list(_srv.cmd)
                    if (any(k in _tail for k in E11_OOM_MARKERS) or "runtimeerror" in _tail or _wd["relaunches"]) \\
                            and _cfg0 and not _cfg0.get("streams12") and not _cfg0.get("w1") and E14_VENV.get("launch"):
                        _cfg1 = dict(_cfg0, streams12=True, mamba_cache=56)
                        SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING = 12, 13
                        _cmd = list(E14_VENV["launch"]) + e11_flags(_cfg1)
                        E14_VENV["cfg"] = _cfg1
                        _rec["streams12"] = True
                    _new = E11Server(_cmd, _srv.env, f"relaunch{len(E12_RUN['watchdog']['relaunches'])}")
                    _new.start()
                    _ok, _why = _new.wait_ready(min(SGL_RELAUNCH_TIMEOUT_SEC, max(60.0, _left - _WD_MIN_LEFT / 2)))
                    _rec.update(ok=_ok, why=_why)
                    if _ok:
                        E11_SERVER = _new
                    else:
                        _new.kill()
                except Exception as _e:
                    _rec.update(ok=False, why=repr(_e)[:300])
                finally:
                    E12A.SERVER_DOWN.clear()
                    E12_RUN["watchdog"]["relaunches"].append(_rec)
                    _bad = None
                print("E15B watchdog:", json.dumps(_rec), flush=True)
                if not _rec.get("ok") and len(_wd["relaunches"]) >= 2:    # B3: a failed relaunch retries once
                    return
        threading.Thread(target=_e15b_watchdog, daemon=True, name="e15b_watchdog").start()
        if P.get("parity"):          # R6-X1 diagnostic profile: no solving (the fallback submission stays)''', "watchdog")
c = once(c, '''                          stage2_p90_cutoff=bool(E12_R6["E12_R6_P90_CUTOFF"]))''',
         '''                          stage2_p90_cutoff=bool(E12_R6["E12_R6_P90_CUTOFF"]),
                          join_grace_sec=90.0,       # E15B #54: room for the final harvest / partial test (30 s grace)
                          max_generated_tokens_b1=E15B["E12_B_ATTEMPT_TOKENS_B1"]''' + (''',
                          skip_b1_if_accepted=bool(E15B["E12_B_SKIP_B1_IF_ACCEPTED"]),     # rev 3i: R
                          extra_attempts=bool(E15B["E12_B_EXTRA_ATTEMPTS"]),               # rev 3i: stage 3
                          max_attempts=int(E15B["E12_B_MAX_ATTEMPTS"])''' if (R3I or R3J) else "") + (''',
                          # rev 3j: 1 = after the attempt 3s (notebook 1), 2 = before the attempt 3s (notebook 2)
                          b1_accepted_mode={1: "after3", 2: "before3"}[int(E15B["E12_B_B1_ACC_MODE"])],
                          weak_commit=bool(E15B["E12_B_WEAK_COMMIT"])''' if R3J else "") + ''')    # rev 3f: attempt 2's session cap''',
         "join grace")
c = once(c, '''def _e12_capacity_probe():''', '''def _e15b_gen_tokens_total():
    # E15B #32: the server's own generated-token counter, to count tokens the report misses. B6 (auditor 2 F6): max over
    # the TP ranks of each label set, summed over the other label rows (e.g. is_streaming)
    try:
        import re as _re
        _g = {}
        for r in e11_prom():
            if getattr(r, "name", "") == "sglang:generation_tokens_total" and getattr(r, "value", None) is not None:
                _k = _re.sub(r'tp_rank="[^"]*",?', "", str(getattr(r, "labels", "") or ""))
                _g[_k] = max(_g.get(_k, 0.0), float(r.value))
        return sum(_g.values()) if _g else None
    except Exception:
        return None


def _e15b_all_server_logs():
    # B6 (auditor 2 F6, auditor 3 L85 / L125): every server log of the solve (the crashed ones first), not only the last
    _t = []
    for _p in ((E12_RUN.get("watchdog") or {}).get("old_logs") or []):
        try:
            _t.append(Path(_p).read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    try:
        _t.append(E11_SERVER.log_text())
    except Exception:
        pass
    return "\\n".join(_t)


def _e15b_server_log_summary(text):
    # E15B #42 / #53: the engine block from the server log: running-level histogram of the decode lines, queue > 0
    # share, gen throughput, retract / preempt lines (server_args excluded), OOM / Traceback
    import re as _re
    _hist, _q, _tp, _n, _hq, _acc, _tpl = {}, 0, [], 0, {}, [], {}
    for _l in text.splitlines():
        if "Decode batch" not in _l:
            continue
        _m = _re.search(r"#running-req: (\\d+)", _l)
        _mq = _re.search(r"#queue-req: (\\d+)", _l)
        _mt = _re.search(r"gen throughput \\(token/s\\): ([\\d.]+)", _l)
        if _m:
            _n += 1
            _hist[int(_m.group(1))] = _hist.get(int(_m.group(1)), 0) + 1
            if _mq and int(_mq.group(1)) > 0:     # A1 (Notes/07): the running level while requests wait in the server
                _q += 1
                _hq[int(_m.group(1))] = _hq.get(int(_m.group(1)), 0) + 1
            if _mt:
                _tp.append(float(_mt.group(1)))
                _tpl.setdefault(int(_m.group(1)), []).append(float(_mt.group(1)))     # B6: tok/s per running level
            _ma = _re.search(r"accept len: ([\\d.]+)", _l)
            if _ma:
                _acc.append(float(_ma.group(1)))
    _bad = lambda k: sum(1 for _l in text.splitlines() if k in _l.lower() and "server_args" not in _l.lower()
                         and "retraction_policy" not in _l)
    _tp.sort()
    # A3 (Notes/07): the startup split from TP0's lines (weight loads, each CUDA graph capture, KV pool, log start -> ready)
    import datetime as _dt
    _st = {"load_sec": {}, "graph_capture": {}, "kv_tokens": None, "ready_span_sec": None}

    def _ts(_l):
        _m = _re.match(r"\\[(\\d{4}-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d)", _l)
        return _dt.datetime.strptime(_m.group(1), "%Y-%m-%d %H:%M:%S") if _m else None
    _t_first = _t_ready = None
    for _l in text.splitlines():
        _t = _ts(_l)
        if _t is not None and _t_first is None:
            _t_first = _t
        if "The server is fired up and ready to roll" in _l and _t is not None and _t_ready is None:
            _t_ready = _t
        if "TP0]" not in _l:
            continue
        _m = _re.search(r"Load weight end\\. elapsed=([\\d.]+) s, type=(\\w+)", _l)
        if _m:
            _st["load_sec"][_m.group(2)] = float(_m.group(1))
        _m = _re.search(r"Capture (.+?) CUDA graph end\\. elapsed=([\\d.]+) s, mem usage=([\\d.]+) GB, avail mem=([\\d.]+) GB", _l)
        if _m:
            _st["graph_capture"][_m.group(1)] = {"sec": float(_m.group(2)), "mem_gb": float(_m.group(3)),
                                                 "avail_gb_after": float(_m.group(4))}
        _m = _re.search(r"KV Cache is allocated\\. dtype: \\S+, #tokens: (\\d+)", _l)
        if _m and _st["kv_tokens"] is None:
            _st["kv_tokens"] = int(_m.group(1))
    if _t_first is not None and _t_ready is not None:
        _st["ready_span_sec"] = (_t_ready - _t_first).total_seconds()
    _st["graph_capture_sec_total"] = round(sum(v["sec"] for v in _st["graph_capture"].values()), 2)
    return {"decode_lines": _n, "running_hist": dict(sorted(_hist.items())),
            "running_hist_queue_gt0": dict(sorted(_hq.items())), "startup": _st,
            "accept_len_p50": sorted(_acc)[len(_acc) // 2] if _acc else None,
            "gen_tok_s_p50_by_running": {k: sorted(v)[len(v) // 2] for k, v in sorted(_tpl.items())},
            "running_mean_lines": round(sum(k * v for k, v in _hist.items()) / _n, 2) if _n else None,
            "queue_gt0_line_share": round(_q / _n, 3) if _n else None,
            "gen_tok_s_p50": _tp[len(_tp) // 2] if _tp else None,
            "retract_lines": _bad("retract"), "preempt_lines": _bad("preempt"), "oom_lines": _bad("out of memory"),
            "traceback_lines": sum(1 for _l in text.splitlines() if "Traceback" in _l)}


def _e12_capacity_probe():''', "report helpers")
c = once(c, '''        E12_RUN["rev6_scheduler"] = dict(E12A.SCHED_LAST)''', '''        E12_RUN["rev6_scheduler"] = dict(E12A.SCHED_LAST)
        try:      # E15B #29, #30, #39: per-attempt summary + real concurrency from the 30-s server gauges
            _att = [r for v in E12_RECORDS.values() for r in v]
            _ev = [r.get("b_events") or {} for r in _att]
            # rev 3f: each attempt's own budget (attempt 1 = E12_B_ATTEMPT_TOKENS, attempt 2 = E12_B_ATTEMPT_TOKENS_B1)
            _bud = lambda r: int(E15B["E12_B_ATTEMPT_TOKENS_B1"] if r.get("branch") ''' + (">=" if (R3I or R3J) else "==") + ''' 1 else E15B["E12_B_ATTEMPT_TOKENS"])
            _g = [x for x in (E12_RUN["rev6_scheduler"].get("gauges") or []) if isinstance(x, list) and len(x) >= 5
                  and isinstance(x[3], (int, float))]
            _run = [x[3] for x in _g]
            _q = [x[4] for x in _g if isinstance(x[4], (int, float))]
            # A1 (Notes/07): the 30-s samples in which every client stream was busy (workers_running >= streams)
            _S = E12_RUN.get("streams") or 0
            _full = [x for x in _g if len(x) >= 6 and isinstance(x[5], (int, float)) and _S and x[5] >= _S]
            E12_RUN["b_summary"] = {
                "attempts": len(_att), "attempt2_started": sum(1 for r in _att if r.get("branch") == 1),
                "accepted": sum(1 for r in _att if r.get("accepted")),
                "at_budget": sum(1 for r in _att if int(r.get("generated_tokens") or 0) >= _bud(r) - 64),
                # no host-accepted candidate at all (an accepted exact check counts too), but a transform was written
                "at_budget_with_progress": sum(1 for r in _att if int(r.get("generated_tokens") or 0)
                                               >= _bud(r) - 64
                                               and not any(x.get("accepted") for x in (r.get("submissions") or [])
                                                           + (r.get("checked") or []))
                                               and ((r.get("rev6_events") or {}).get("first_transform_tokens") is not None)),
                "commit_turns": sum(1 for e in _ev if e.get("commit_at_tokens") is not None),
                "commit_with_tool_call": sum(1 for e in _ev if e.get("commit_tool_call")),
                "submit_tool_routed": sum(int(e.get("submit_tool_routed") or 0) for e in _ev),
                "reasoning_harvest_ran": sum(1 for e in _ev if (e.get("reasoning_harvest") or {}).get("ran")),
                "reasoning_harvest_new_accepted": sum(int((e.get("reasoning_harvest") or {}).get("new_accepted") or 0) for e in _ev),
                "trimmed_requests": sum(1 for r in _att for q in (r.get("requests") or []) if q.get("trimmed")),
                "worker_errors": E12_RUN["rev6_scheduler"].get("worker_errors"),
                "setup_failed": E12_RUN["rev6_scheduler"].get("setup_failed"),     # rev 3h: Auditor 4 #4
                "cut_code_saved": sum(int(e.get("cut_code_saved") or 0) for e in _ev),     # rev 3h
                "cut_code_harvested": sum(1 for e in _ev if (e.get("reasoning_harvest") or {}).get("cut_code_blocks")),''' + ('''
                # rev 3i: R + stage 3
                "b1_skipped": E12_RUN["rev6_scheduler"].get("b1_skipped"),
                "extra_started": E12_RUN["rev6_scheduler"].get("extra_started"),
                "extra_by_attempt": E12_RUN["rev6_scheduler"].get("extra_by_attempt"),
                "extra_accepted": sum(1 for r in _att if int(r.get("branch") or 0) >= 2 and r.get("accepted")),
                "b1_deferred": E12_RUN["rev6_scheduler"].get("b1_deferred"),                 # rev 3j
                "b1_deferred_started": E12_RUN["rev6_scheduler"].get("b1_deferred_started"),
                "weak_accepts": E12_RUN["rev6_scheduler"].get("weak_accepts"),               # rev 3k W1
                "weak_only_tasks_continued": E12_RUN["rev6_scheduler"].get("weak_only_tasks_continued"),
                "stage2_errors": E12_RUN["rev6_scheduler"].get("stage2_errors"),''' if (R3I or R3J) else "") + '''
                "server_running_mean": round(sum(_run) / len(_run), 2) if _run else None,
                "server_queue_gt0_share": round(sum(1 for v in _q if v > 0) / len(_q), 3) if _q else None,
                "gauge_samples": len(_g), "streams": E12_RUN.get("streams"),
                "client_full_samples": len(_full),
                "server_running_mean_client_full": round(sum(x[3] for x in _full) / len(_full), 2) if _full else None,
                "server_at_streams_share_client_full": (round(sum(1 for x in _full if x[3] >= _S) / len(_full), 3)
                                                        if _full else None),
                "kv_stale_checks": E12_RUN["rev6_scheduler"].get("kv_stale_checks"),      # A4
                # #34: accepted submissions vs harvested exact checks; #31 roll-up; #35 the effort actually sent
                "accepted_submit": sum(int((r.get("b_candidate") or {}).get("submitted_accepted") or 0) > 0 for r in _att),
                "harvested_only": sum(1 for r in _att if int((r.get("b_candidate") or {}).get("harvested") or 0) > 0
                                      and not int((r.get("b_candidate") or {}).get("submitted_accepted") or 0)),
                "never_wrote_candidate": sum(1 for r in _att if not (r.get("b_candidate") or {}).get("candidate_source_written")),
                "wrote_never_checked": sum(1 for r in _att if (r.get("b_candidate") or {}).get("candidate_source_written")
                                           and not (r.get("b_candidate") or {}).get("check_executed")),
                "partial_test_runs": sum(1 for e in _ev if (e.get("partial_test") or {}).get("name")),
                "effective_effort": E12_REASONING_EFFORT or "xhigh (template default)",
                # #32: server-side generated tokens in the solve vs the attempts' count
                # B6: per server (the counter resets on a relaunch): the finished servers' deltas + the current one
                "server_gen_tokens": (lambda a, b: round((E12_RUN["watchdog"].get("tok_acc") or 0.0) + a - b)
                                      if a is not None and b is not None else None)(
                    _e15b_gen_tokens_total(), E12_RUN["watchdog"].get("tok_base")),
                "counted_gen_tokens": sum(int(r.get("generated_tokens") or 0) for r in _att),
                "server_down_waits": dict(E12A.SERVER_WAITS), "watchdog": E12_RUN.get("watchdog")}
            try:
                E12_RUN["b_summary"]["server_log"] = _e15b_server_log_summary(_e15b_all_server_logs())   # #42 / #53; B6
            except Exception as _e:
                E12_RUN["b_summary"]["server_log"] = {"error": repr(_e)[:200]}
            print("E15B summary:", json.dumps(E12_RUN["b_summary"]), flush=True)
            if _run and E12_RUN["b_summary"]["server_running_mean"] < (E12_RUN.get("streams") or 0) - 0.5:
                print("E15B WARNING: mean server running", E12_RUN["b_summary"]["server_running_mean"], "< streams - 0.5",
                      flush=True)
        except Exception as _e:
            E12_RUN["b_summary"] = {"error": repr(_e)[:300]}''', "B summary")
cells[IDX["CELL 6 "]] = c

c = cells[IDX["CELL 7 "]]
c = once(c, '''        report.update(eval_query_hits=hits, eval_query_total=total, eval_pass1_hits=pass1, eval_pass2_hits=pass2,
                      eval_task_norm_pct=round(100 * task_norm / max(1, len(TASK_IDS)), 2), eval_unit_records=recs_eval)''',
         '''        report.update(eval_query_hits=hits, eval_query_total=total, eval_pass1_hits=pass1, eval_pass2_hits=pass2,
                      eval_task_norm_pct=round(100 * task_norm / max(1, len(TASK_IDS)), 2), eval_unit_records=recs_eval)
        try:    # E15B A2 (Exp 15 Notes/07): what each attempt found. A task counts for an attempt when that attempt has a
            # right output for EVERY test input: "acc" = among its host-accepted candidates, "any" = also partial ones
            def _e15b_has(r, qi, gold, acc_only):
                return any(qi < len(s.get("test_outputs") or []) and s["test_outputs"][qi] == gold
                           for s in (r.get("submissions") or []) + (r.get("checked") or []) if s.get("accepted") or not acc_only)
            _B_TOK = int(E15B["E12_B_ATTEMPT_TOKENS"]) if "E15B" in globals() else 35000
            _pa = dict.fromkeys(("tasks", "b1_started", "hit", "b0_acc", "b1_acc", "both_acc", "b0_only_acc", "b1_only_acc",
                                 "b0_any", "b1_any", "b1_only_any", "hit_without_b0_any", "b0_budget_partial",
                                 "b0_budget_partial_b1_acc"), 0)
            _rows = []
            for t in TASK_IDS:
                _rs = {r.get("branch"): r for r in _recs.get(t, [])}
                _golds = SOLUTIONS[t]
                _f = lambda b, acc: b in _rs and all(_e15b_has(_rs[b], qi, g, acc) for qi, g in enumerate(_golds))
                a0, a1, n0, n1 = _f(0, True), _f(1, True), _f(0, False), _f(1, False)
                hit = all(att["attempt_1"] == g or att["attempt_2"] == g for att, g in zip(_gen[t], _golds))
                r0 = _rs.get(0) or {}
                bp = (int(r0.get("generated_tokens") or 0) >= _B_TOK - 64
                      and not any(x.get("accepted") for x in (r0.get("submissions") or []) + (r0.get("checked") or []))
                      and (r0.get("rev6_events") or {}).get("first_transform_tokens") is not None)
                for k, v in (("tasks", True), ("b1_started", 1 in _rs), ("hit", hit), ("b0_acc", a0), ("b1_acc", a1),
                             ("both_acc", a0 and a1), ("b0_only_acc", a0 and not a1), ("b1_only_acc", a1 and not a0),
                             ("b0_any", n0), ("b1_any", n1), ("b1_only_any", n1 and not n0),
                             ("hit_without_b0_any", hit and not n0), ("b0_budget_partial", bp),
                             ("b0_budget_partial_b1_acc", bp and a1)):
                    _pa[k] += int(bool(v))
                _rows.append({"task": t, "hit": hit, "b0_acc": a0, "b1_acc": a1, "b0_any": n0, "b1_any": n1,
                              "b0_budget_partial": bp, "b0_tokens": r0.get("generated_tokens"),
                              "b1_tokens": (_rs.get(1) or {}).get("generated_tokens")})
            report["eval_b_attempts"] = dict(_pa, rows=_rows)''' + ('''
            # rev 3i: what stage 3 added (a task hit whose only right accepted answer came from attempt 3+)
            _x = {"tasks_with_attempt3plus": 0, "attempt3plus_acc_tasks": 0, "hit_only_from_attempt3plus": 0}
            for t in TASK_IDS:
                _rs = {r.get("branch"): r for r in _recs.get(t, [])}
                _golds = SOLUTIONS[t]
                _ok = lambda b: b in _rs and all(_e15b_has(_rs[b], qi, g, True) for qi, g in enumerate(_golds))
                _hi = [b for b in _rs if (b or 0) >= 2]
                if _hi:
                    _x["tasks_with_attempt3plus"] += 1
                    if any(_ok(b) for b in _hi):
                        _x["attempt3plus_acc_tasks"] += 1
                        if not _ok(0) and not _ok(1):
                            _x["hit_only_from_attempt3plus"] += 1
            report["eval_b_attempts"].update(_x)
            print("E15B stage 3 (eval):", json.dumps(_x), flush=True)''' if (R3I or R3J) else "") + '''
            print("E15B attempts (eval):", json.dumps(_pa), flush=True)
        except Exception as _e:
            report["eval_b_attempts"] = {"error": repr(_e)[:300]}''', "A2 per-attempt eval")
cells[IDX["CELL 7 "]] = c


def _cell_with(t):
    _ix = [i for i, x in enumerate(cells) if t in x]
    assert len(_ix) == 1, (t[:60], _ix)
    return _ix[0]


# B1 (auditor 2 F1): after the solve end the final publishes of ~480 sessions would each rebuild + rewrite submission.json
# (0.27-0.56 s each in E15B: 2.2-4.5 min against the 12-h limit); the write after the solve (after _E12_STOP) stays
_i = _cell_with("def _e12_branch_done_r6(rec):")
cells[_i] = once(cells[_i], '''        E12_RUN["branches_accepted"] = sum(1 for v in E12_RECORDS.values() for r in v if r.get("accepted"))
    try:
        e12_write_submission()
    except Exception as _e:
        print("E12 submission write failed:", repr(_e)[:200], flush=True)''', '''        E12_RUN["branches_accepted"] = sum(1 for v in E12_RECORDS.values() for r in v if r.get("accepted"))
    if time.perf_counter() >= globals().get("E15B_SOLVE_END_PC", float("inf")):
        E12_RUN["late_writes_skipped"] = E12_RUN.get("late_writes_skipped", 0) + 1     # E15B B1
        return
    try:
        e12_write_submission()
    except Exception as _e:
        print("E12 submission write failed:", repr(_e)[:200], flush=True)''', "B1 no late writes")
# B4 (auditor 3 L111): a dead main process can leave its TP workers alive (180 s gpu-free wait + pkill per crash)
_i = _cell_with("class E11Server:")
cells[_i] = once(cells[_i], '''    def kill(self, wait=True):
        if self.alive():''', '''    def kill(self, wait=True):
        if self.proc is not None and not self.alive() and hasattr(os, "killpg"):    # E15B B4: orphaned TP workers
            try:
                os.killpg(self.proc.pid, getattr(_signal, "SIGKILL", _signal.SIGTERM))
            except Exception:
                pass
        if self.alive():''', "B4 killpg dead")
# B5 (auditor 2 F4): in the hidden run an engine failure = the identity file (~0): keep trying up to T0 + 4 h, 4 launches
_i = _cell_with("SGL_GIVE_UP_AT = min(NOTEBOOK_T0 + 75 * 60, ADMIT_DEADLINE - 5 * 60)")
cells[_i] = once(cells[_i], "SGL_GIVE_UP_AT = min(NOTEBOOK_T0 + 75 * 60, ADMIT_DEADLINE - 5 * 60)",
                 "SGL_GIVE_UP_AT = min(NOTEBOOK_T0 + (240 if RERUN else 75) * 60, ADMIT_DEADLINE - 5 * 60)   # E15B B5: RERUN 4 h",
                 "B5 give-up")
_i = _cell_with("E14_STAGING.join(timeout=max(0.0, SGL_GIVE_UP_AT - 20 * 60 - time.perf_counter()))")
cells[_i] = once(cells[_i], "E14_STAGING.join(timeout=max(0.0, SGL_GIVE_UP_AT - 20 * 60 - time.perf_counter()))",
                 "E14_STAGING.join(timeout=max(0.0, min(SGL_GIVE_UP_AT - 20 * 60, NOTEBOOK_T0 + 55 * 60) - time.perf_counter()))",
                 "B5 staging join bound")
_i = _cell_with("    for i in range(3):\n        left = SGL_GIVE_UP_AT - time.perf_counter()")
cells[_i] = once(cells[_i], "    for i in range(3):\n        left = SGL_GIVE_UP_AT - time.perf_counter()",
                 "    for i in range(4):      # E15B B5: 16 -> 12 -> diagnosed -> W1\n        left = SGL_GIVE_UP_AT - time.perf_counter()",
                 "B5 4 launches")
cells[_i] = once(cells[_i], 'raise E11Fail("3 launches failed")', 'raise E11Fail("4 launches failed")', "B5 message")

# ------------------------------------------------------------------ 8. the notebook
MD = (f"# E15B · config B: 2 independent Duck attempts per task ({ATTEMPT_TOKENS // 1000}k + {ATTEMPT_TOKENS_B1 // 1000}k tokens), 16 real streams (SGLang 0.5.21 + RedHat INT4 + DFlash2 W6)\n\n"
      "**Base (unchanged code path):** the E14S submission notebook (LB 22.78), built in memory by build_e14.py.\n\n"
      f"**E15B changes (Experiment 15 Notes/02, 08):** two independent attempts per task, {ATTEMPT_TOKENS:,} then {ATTEMPT_TOKENS_B1:,} generated tokens (attempt 1 "
      "on every task first, attempt 2 = a fresh agent with the facts view), no pause / chunk / resume, no 12k / 24k / 32k "
      "request caps (cap = remaining budget - 4,096; then one commit turn), a fixed 100k Duck window, the tool contract and "
      "the budget in the prompt, submit / check tool calls routed to python, auto-check of candidate / trans names, the "
      "sandbox's safe sys shim and time, the defined-functions inventory, code written only in the reasoning harvested at "
      "the end, a reply's remaining tool calls run before a yield (no turn rollback), a restarting scheduler worker. "
      "Engine: 16 streams (mrr 17, graphs 1..17, mamba 68), the DFlash min-free-slots delayer off, KV per request 50k, "
      "request timeout 3,000 s; a 12-stream fallback launch.\n\n"
      "**Commit:** " + ("all 120 evaluation tasks at the hidden run's compute per task (gate 4b)." if EVAL120G4B else
                        f"a GPU run of 20 eval tasks with both attempts finishing ({COMMIT_WINDOW_MIN}-min solve window, {COMMIT_BUDGET_MIN}-min budget), scored.") +
      " **Rerun:** every hidden task with the same solver.\n\n"
      "**Inputs:** the private dataset `" + PAYLOAD_SLUG + "` (files named by their sha256, verified before use) must be "
      "attached.\n")
nb["cells"][0]["source"] = MD.splitlines(keepends=True)
for i, s in enumerate(cells):
    if i > 0:
        nb["cells"][i]["source"] = s.splitlines(keepends=True)
for _i, _c in enumerate(nb["cells"]):
    if _c["cell_type"] == "code":
        compile("".join(_c["source"]), f"<e15b cell {_i}>", "exec")
        assert "\r" not in "".join(_c["source"]), f"CR line ending in cell {_i}"
_text = json.dumps(nb, indent=1, ensure_ascii=False)
assert len(_text.encode("utf-8")) < NB_LIMIT, f"notebook {len(_text.encode('utf-8'))} B >= {NB_LIMIT} B"
assert G["DUCK_ZIP"] not in _text and G["DUCK_SHA"] not in _text, "an E14S payload reference is left"
out = HERE.parent / NAME
if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() in PUSHED:
    out = out.with_name(out.stem + "_REBUILD.ipynb")
out.write_text(_text, encoding="utf-8")
PAYLOAD_DIR.mkdir(exist_ok=True)
for _name, _h, _data in ((DUCK_ZIP, DUCK_ZIP_SHA, DUCK_BYTES), (MOD_ZIP, MOD_ZIP_SHA, MOD_BYTES)):
    (PAYLOAD_DIR / _name).write_bytes(_data)
    assert hashlib.sha256((PAYLOAD_DIR / _name).read_bytes()).hexdigest() == _h
(PAYLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(
    {"title": "Q27 E15B payload", "id": PAYLOAD_SLUG, "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")
print(out.name, hashlib.sha256(out.read_bytes()).hexdigest(), round(out.stat().st_size / 1e3), "kB | arc2_duck", B_SHA[:12],
      "| duck zip", DUCK_ZIP, "| modules", MOD_ZIP)
print("replaced:", {k: (OLD_SHA[k][:8], NEW_SHA[k][:8]) for k in REPL})
