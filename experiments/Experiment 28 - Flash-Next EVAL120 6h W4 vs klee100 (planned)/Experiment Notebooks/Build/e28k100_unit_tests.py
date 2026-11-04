"""E28 K100 unit tests (quick, local + the Kaggle CPU units kit): the built notebook Q27-XHIGH-E28_K100_EVAL120_6H.ipynb, its
facts, e28k100_lib.py and the scheduler integration. No GPU, no server, no whole-notebook run (that is the e2e kit's job).

  A  provenance: facts == notebook (sha / bytes / < 400 kB / LF); CELL 0 == E18 R2 CELL 0; the packed e21_cell7 == K100Q's text;
     the packed modules == the Build files (shas); every changed hunk vs K100Q carries an E28 marker
  B  profiles: E28EVAL120 / RERUN_PROD rows (exec of CELL 1's profile block), the time plan (HARD_STOP 410 min, deadman 415),
     the T1 / B2 asserts, the guard
  C  the edits in place: MAX_LEN 262,144 (server + fake argv), the T1 capacity ceiling, workers rule, I8 ladder / read-back,
     scheduler apply in CELL 3, N1 / N2 / N3, plain order, the report block, the bundle
  D  N1 / N2 / N4 numbers (exec of the edited lines, MAX_LEN 131,072): T1 104k 120,384 / 36 blocks, B2 120k 131,072 / 38 blocks
  E  e28k100_lib: I8 read-back / failure reason / static check (the real klee100 config + the real Merlin source when present),
     capacity note, the report on K100Q's REAL GPU output (score == its run report)
  F  e28_sched_patch on the real arc2_duck.py (when present): apply / verify / revert on a temp copy, the text mode

usage: python e28k100_unit_tests.py   (env E28KT_NB / E28KT_FACTS / E28KT_DUCK / E28KT_K100Q_OUT / E28KT_SOL / E28KT_E18 /
       E28KT_KLEE_CFG / E28KT_MERLIN_SRC / E28KT_K100Q_NB override the default local paths; a missing optional input = SKIP)
Prints one PASS / FAIL / SKIP line per check, then "<n>/<m> PASS"; exit 1 on any FAIL.
"""
import ast
import difflib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RESULTS = []


def check(label, fn):
    try:
        ok = bool(fn())
        why = ""
    except Exception as e:
        ok, why = False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}"
    RESULTS.append((label, ok))
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <- " + why[:900]), flush=True)
    return ok


def skip(label, why):
    print(f"SKIP {label}: {why}", flush=True)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def p_env(name, default):
    return Path(os.environ.get(name) or default)


NB = p_env("E28KT_NB", HERE.parent / "Q27-XHIGH-E28_K100_EVAL120_6H.ipynb")
FACTS_P = p_env("E28KT_FACTS", HERE / "e28k100_build_facts.json")
LIB_P = HERE / "e28k100_lib.py"
SCHED_P = HERE / "e28_sched_patch.py"
DUCK = p_env("E28KT_DUCK", r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/duck_b/arc2_duck.py")
K100Q_OUT = p_env("E28KT_K100Q_OUT", r"D:/ARC AGI 2/_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x")
SOL = p_env("E28KT_SOL", r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
            r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json")
E18 = p_env("E28KT_E18", r"D:/ARC AGI 2/experiments/Experiment 18 - S1-60K on vLLM 0.32 (planned)/Experiment Notebooks/"
            r"Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb")
KLEE_CFG = p_env("E28KT_KLEE_CFG", r"D:/ARC AGI 2/_scratch/klee100_upload_fetch_2026-10-31")
MERLIN_SRC = p_env("E28KT_MERLIN_SRC", r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
K100Q_NB = p_env("E28KT_K100Q_NB", r"D:/ARC AGI 2/experiments/Experiment 26 - Flash-Next klee100 3bpw low-bit (planned)/"
                 r"Experiment Notebooks/Q27-XHIGH-E26_K100_QUALITY.ipynb")
NB5LIB_P = p_env("E28KT_NB5LIB", r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/"
                 r"Build/nb5_lib.py")

RAW = NB.read_bytes()
FACTS = json.loads(FACTS_P.read_text(encoding="utf-8"))
NBJ = json.loads(RAW.decode("utf-8"))
CODE = ["".join(c["source"]) for c in NBJ["cells"] if c["cell_type"] == "code"]
C0, C1, C3, C4, ENG, C67, C8, C9 = CODE
sys.path.insert(0, str(HERE))
import e28k100_lib as L   # noqa: E402


def packed_text(expr_src):
    """Evaluate one sha-checked packed literal expression (the KQ.packed_lzma form) -> the text."""
    return eval(compile(ast.Expression(ast.parse(expr_src, mode="eval").body), "<packed>", "eval"), {"__builtins__": __builtins__})


def cell7_text():
    tree = ast.parse(C67)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
    return packed_text(ast.get_source_segment(C67, call.args[0].args[0]))


def modules():
    line = next(l for l in C1.splitlines() if l.startswith("E28_SRC = json.loads("))
    expr = ast.parse(line).body[0].value.args[0]
    return json.loads(packed_text(ast.get_source_segment(line, expr)))


READ = C67[:C67.index("# ---- E28 K100 (size: the 400-kB commit limit)")] + cell7_text()
READABLE = [C0, C1, C3, C4, ENG, READ, C8, C9]

# ------------------------------------------------------------------ A provenance
check("A1 facts == notebook (sha, bytes), < 400,000 B, no CR, 9 cells, compact JSON",
      lambda: FACTS["notebook_sha256"] == sha(RAW) and FACTS["bytes"] == len(RAW) < 400_000 and b"\r" not in RAW
      and len(NBJ["cells"]) == 9 and FACTS["json_format"] == "compact")
if E18.is_file():
    check("A2 CELL 0 == E18 R2 cell 1 (proven dual-mode CELL 0, verbatim)",
          lambda: C0 == "".join(json.loads(E18.read_text(encoding="utf-8"))["cells"][1]["source"]).replace("\r\n", "\n"))
else:
    skip("A2 CELL 0 == E18 R2", f"{E18} missing")
check("A3 the packed e21_cell7 == the facts' sha (K100Q's text, unchanged)",
      lambda: sha(cell7_text()) == FACTS["e21_cell7_packed"]["sha256"] and cell7_text().startswith("def e21_cell7():\n"))
check("A4 the packed modules == the Build files (e28k100_lib.py, e28_sched_patch.py) and the facts",
      lambda: {n: sha(t) for n, t in modules().items()} == FACTS["modules"] == {n: sha((HERE / n).read_bytes()) for n in FACTS["modules"]}
      and set(FACTS["modules"]) == {"e28k100_lib.py", "e28_sched_patch.py"})
if K100Q_NB.is_file():
    def _a5():
        base = ["".join(c["source"]) for c in json.loads(K100Q_NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        assert sha(K100Q_NB.read_bytes()) == FACTS["base"]["k100q_sha256"]
        bad = []
        for i, (a, b) in enumerate(zip(base, READABLE)):
            hunk, has = [], False
            for ln in list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0))[2:] + ["@@"]:
                if ln.startswith("@@"):
                    if hunk and not has:
                        bad.append((i, hunk[:3]))
                    hunk, has = [], False
                elif ln.startswith("+"):
                    hunk.append(ln[:120])
                    has = has or "E28" in ln
                elif ln.startswith("-"):
                    hunk.append(ln[:120])
        assert not bad, bad[:6]
        return len(base) == len(READABLE) == 8
    check("A5 every changed hunk vs K100Q (4506a71f, readable cells) adds a line marked E28 (no unmarked change)", _a5)
else:
    skip("A5 diff vs K100Q", f"{K100Q_NB} missing")

# ------------------------------------------------------------------ B profiles + time plan
def cell1_ns(rerun=False, env=None):
    """CELL 1's profile block (from _NB5_T to the NB5HARNESS line) + the time block, executed with stubs."""
    i, j = C1.index("_NB5_T = {"), C1.index('if E11_HARNESS and os.environ.get("NB5H_TASKS"):')
    k, m = C1.index("# ---- time (E15B / E21 CELL 1 block, verbatim)"), C1.index("# ---- NB5 engine + workload parameters.")
    g = {"os": os, "json": json, "E11_HARNESS": False, "__builtins__": __builtins__}
    src = C1[i:j] + "\nPROFILE = 'RERUN_PROD' if RERUN else 'E28EVAL120'\nP = PROFILE_TABLE[PROFILE]\nNOTEBOOK_T0 = 1000.0\n" + C1[k:m]
    old = {k_: os.environ.get(k_) for k_ in ("Q27X_TIME_BUDGET",)}
    os.environ.pop("Q27X_TIME_BUDGET", None)
    try:
        exec(compile("RERUN = " + repr(rerun) + "\n" + src, "<cell1 profiles>", "exec"), g)
    finally:
        for k_, v in old.items():
            if v is not None:
                os.environ[k_] = v
    return g


G1, GR = cell1_ns(False), cell1_ns(True)
P, PR = G1["PROFILE_TABLE"]["E28EVAL120"], GR["PROFILE_TABLE"]["RERUN_PROD"]
LEV = {"nb5_gate": 1.0, "nb5_attempt": {"b0": 104000, "b1": 104000}, "admit_mode": "slots", "e28_workers_rule": True,
       "f_rule": "measured", "pin_ladder": True, "patch_60490": True, "lever_env": {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"},
       "no_pure_decode": True, "continue_after_failed_boot": True, "i8": True, "plain_order": True, "e28_sched": True,
       "seqs_rule": "k100", "caps16": False, "k100_caps": True, "k100": True, "top_rung_seqs": 24,
       "nb5_env": {"E12_B_SKIP_B1_IF_ACCEPTED": "0", "E12_B_B1_ACC_MODE": "0", "E12_B_EXTRA_ATTEMPTS": "0"}}
check("B1 E28EVAL120: eval, all 120 tasks (no task list / order), 415-min budget, hard margin 5 min, fixed 21,600-s window, one "
      "phase Q_E28_K100_EVAL120 (MTP on, variant A, FP8), quality_ab + equal_windows (the K100Q fixed-window mechanism), the E28 levers",
      lambda: P["source"] == "eval" and P["n_tasks"] == 120 and "task_list" not in P and "task_order" not in P
      and P["time_budget_sec"] == 415 * 60 and P["hard_margin_sec"] == 300 and P["fixed_window_sec"] == 21600
      and P["nb5_phases"] == [["Q_E28_K100_EVAL120", "all", None, "on", "A", "fp8"]] and P["quality_ab"] and P["equal_windows"]
      and all(P[k] == v for k, v in LEV.items()) and "LOCAL_ANALYZER_SEED" not in P["nb5_env"])
check("B2 RERUN_PROD: every hidden test task (source test, n_tasks None), 12 h - 3 min, the E18 ladder, phase EVAL, the same E28 "
      "levers + klee100 boot levers, no fixed window",
      lambda: PR["source"] == "test" and PR["n_tasks"] is None and PR["time_budget_sec"] == 12 * 3600 - 180 and PR["e18_ladder"] is True
      and PR["nb5_phases"] == [["EVAL", "all", None, "on", "A", "fp8"]] and all(PR[k] == v for k, v in LEV.items())
      and "fixed_window_sec" not in PR and not PR.get("quality_ab"))
check("B3 only two profiles pass the guard (E28EVAL120, RERUN_PROD); the commit default = E28EVAL120; no K100QUALITY",
      lambda: G1["K100_PROFILES"] == ("E28EVAL120", "RERUN_PROD") and 'NB5_DEFAULT_PROFILE = "E28EVAL120"' in C1
      and "K100QUALITY" not in G1["PROFILE_TABLE"])
check("B4 commit time plan: HARD_STOP = T0 + 410 min (6 h 50 min), the deadman K100_DEADMAN_SEC = 24,900 = HARD_STOP + 5 min, "
      "ADMIT < DRAIN < HARD_STOP < VERIFY < budget end",
      lambda: G1["HARD_STOP"] - 1000.0 == 410 * 60 and "K100_DEADMAN_SEC = 24900" in C1 and 24900 == 410 * 60 + 300
      and 1000.0 < G1["ADMIT_DEADLINE"] < G1["DRAIN_DEADLINE"] < G1["HARD_STOP"] < G1["VERIFY_DEADLINE"] < 1000.0 + 415 * 60)
check("B5 rerun time plan = NB6's: 12 h - 3 min budget, hard margin 2 min (no deadman in the rerun: 'if not RERUN:' arms it)",
      lambda: GR["TOTAL_BUDGET_SEC"] == 12 * 3600 - 180 and GR["HARD_STOP"] - 1000.0 == 12 * 3600 - 180 - 120
      and "if not RERUN:\n    K100_DEADMAN.update(" in C1)
check("B6 thinking (user 2026-11-01): T1 = 104,000 (K100Q's) for A / B1 / C, B2 = 120,000, B2's window = the normal 130,560; "
      "the K100Q asserts kept (NB5_CAP <= 130,560, max attempt <= 120,000) and B2 under the same 120k rule; NO 262k anywhere: "
      "E21_VLLM_MAX_MODEL_LEN stays 131,072, no E28_SERVER_MAX_LEN / E28_T1_CTX",
      lambda: "assert NB5_CAP <= 131072 - 512 and max(NB5_ATTEMPT.values()) <= 120000, NB5_ATTEMPT" in C1
      and G1["E28_T1"] == 104000 and G1["E28_B2"] == 120000 and G1["E28_B2_WINDOW"] == 130560 and G1["E28_B2_TIMEOUT"] == 9000
      and "assert E28_B2 <= 120000 and E28_B2 - 4096 <= E28_B2_WINDOW and E28_T1 == 104000" in C1
      and "E21_VLLM_MAX_MODEL_LEN = 131072\n" in C1 and not any(s in "".join(READABLE) for s in ("E28_SERVER_MAX_LEN", "E28_T1_CTX"))
      and G1["E28_I8_ENV"] == {"DRAFT_HEAD": "int8"})
check("B7 request timeouts: E12_DUCK_REQUEST_TIMEOUT default 7,200 s with the scheduler (T1), 3,600 s without; B2 = 9,000 s "
      "(E28_B2_REQUEST_TIMEOUT set in CELL 3 with the scheduler env, user 2026-11-01)",
      lambda: 'E12_DUCK_REQUEST_TIMEOUT = int(os.environ.get("E12_DUCK_REQUEST_TIMEOUT", (7200 if P.get("e28_sched") else 3600) if _E12_REV4 else 0))' in C1
      and '_e28env["E28_B2_REQUEST_TIMEOUT"] = str(E28_B2_TIMEOUT)' in C3 and C3.index('_e28env["E28_B2_REQUEST_TIMEOUT"]') < C3.index("os.environ.update(_e28env)"))

# ------------------------------------------------------------------ C the edits in place
check("C1 MAX_LEN stays K100Q's 131,072 (the server --max-model-len, real + fake argv: str(MAX_LEN))",
      lambda: "MAX_LEN = 131072\n" in ENG and ENG.count('"--max-model-len", str(MAX_LEN)') == 2)
check("C2 the capacity plan = K100Q's (MAX_LEN at every capacity_plan call); workers = the R2 rule with e28_workers_rule; the E28 "
      "capacity note (recorded)",
      lambda: ENG.count("NB5_TOOL_ALLOW, MAX_LEN,") == 3
      and 'if NB5_ADMIT_MODE == "slots" or P.get("e28_workers_rule"):' in ENG and 'plan["e28"] = E28L.capacity_note(' in ENG)
check("C10 workers 24 on the TOP rung only (user 2026-11-01): top_rung_seqs 24 in both profiles; nb5_boot launches 24 on the "
      "top rung (pin 7,933,526,016) and K100Q's table elsewhere; nb5_capacity gives 24 workers only to a 24-seq launch (lower rungs "
      "= the R2 min); the K100Q captures (max 84) cover 24 x 4 / 2 = 48 tokens per PP2 micro-batch",
      lambda: P["top_rung_seqs"] == PR["top_rung_seqs"] == 24 and "seqs_fixed" not in P and "workers_fixed" not in P
      and G1["E28_PP2_MICROBATCH_TOKENS"] == 48 and 48 in G1["K100_CAPS"] and max(G1["K100_CAPS"]) == 84
      and abs(G1["K100_PIN_STEPS_GIB"][0] - (7933526016 - 2479838700) / 2 ** 30) < 1e-12
      and 'if P.get("top_rung_seqs") and _seq and int(_seq) == int(P["top_rung_seqs"]):' in ENG
      and ENG.index('            _w = int(_seq)') < ENG.index('            plan["workers"] = _w')
      and 'if P.get("top_rung_seqs") and abs(float(tr["gib"]) - K100_PIN_STEPS_GIB[0]) < 1e-9:' in ENG
      and ENG.index('            seqs = int(P["top_rung_seqs"])') < ENG.index('        NB5_BOOT_EXTRA = base_extra + (["--max-num-seqs", str(seqs)] if seqs else [])')
      and '"captures_cover_pp2_microbatch": max(K100_CAPS) >= -(-_w * 4 // 2)' in ENG)


def _c11():
    """nb5_capacity's workers block (exec of the notebook's own nb5_capacity with a stub server): pool 479 + launch 24 -> 24;
    pool 414 + launch 18 -> 18 (R2 rule 18); pool 317 + launch 14 -> 14."""
    import importlib.util as _u
    s_ = _u.spec_from_file_location("nb5_lib_c11", NB5LIB_P)
    lib = _u.module_from_spec(s_)
    s_.loader.exec_module(lib)
    fn = next(n for n in ast.parse(ENG).body if isinstance(n, ast.FunctionDef) and n.name == "nb5_capacity")
    out = []
    for pool, seqs in ((479, 24), (414, 18), (317, 14)):
        x = 6.0 * 0 + pool / (32 + 6)
        log = (f"GPU KV cache size: {pool * 4096:,} tokens, Maximum concurrency for 131,072 tokens per request: {x:.2f}x\n"
               f"non-default args: {{'max_num_seqs': {seqs}, 'block_size': 4096}}")
        g = {"NB5L": lib, "NB5_ATTEMPT": {"b0": 104000, "b1": 104000}, "NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192, "MAX_LEN": 131072,
             "NB5_GATE": 1.0, "NB5_F_RULE": "measured", "NB5_ADMIT_MODE": "slots", "NB5_NB1_CAPACITY": {}, "NB5_POOL_PLACEHOLDER": {"fp8": pool, "bf16": 78},
             "NB5_F_PLACEHOLDER": 6, "P": {"top_rung_seqs": 24, "e28_workers_rule": True, "k100_caps": True}, "NB5_CAPS16": [1],
             "K100_CAPS": G1["K100_CAPS"], "nb5_metrics_text": lambda: "", "E28L": L, "E28_T1": 104000, "E28_B2": 120000}
        exec(compile(ast.Module([fn], []), "<nb5_capacity>", "exec"), g)

        class Srv:
            def log(self):
                return log
        plan = g["nb5_capacity"](Srv(), "fp8")
        out.append((plan["workers"], plan["r2_workers"]["rung_workers"], plan["caps"].get("max_num_seqs")))
    assert [o[0] for o in out] == [24, 18, 14], out
    return all(o[0] == o[1] for o in out)


if NB5LIB_P.is_file():
    check("C11 nb5_capacity (the notebook's own def, real nb5_lib): top rung (pool 479, launch 24) -> 24 workers; rung 2 (414, 18) -> 18; "
          "rung 3 (317, 14) -> 14 (K100Q's R2 values below the top rung); the rung's workers recorded", _c11)
check("C3 I8: DRAFT_HEAD into the boot levers while on, the try name _i8, the drop rule (same pin, continue) before the ladder "
      "break, the read-back after C-11, the static check before the first boot",
      lambda: 'NB5_BOOT_LEVERS = dict(levers, **(E28_I8_ENV if E28_I8["on"] else {}))' in ENG and '("_i8" if E28_I8["on"] else "")' in ENG
      and ENG.index("E28L.i8_failure_reason(") < ENG.index('        if not fail["failed"] or not P.get("pin_ladder")')
      and 'trec["i8_readback"] = ' in ENG and "def e28_i8_static():" in ENG and "E28_I8[\"static\"] = e28_i8_static()" in C8
      and ENG.index("E28L.i8_failure_reason(") < ENG.index("            continue\n        if not fail[\"failed\"]"))
check("C4 the scheduler in CELL 3: after the manifest assert, before `import arc2_duck`; commit NO_GO raise / rerun revert",
      lambda: C3.index('assert not _bad, ("E12 package hash mismatch", _bad)') < C3.index("E28S.apply_e28_sched(E12_DIR)")
      < C3.index("import arc2_duck as E12A") and 'raise RuntimeError("NO_GO_E28_SCHED: "' in C3 and "E28S.revert_e28_sched(E12_DIR)" in C3
      and "os.environ.update(_e28env)" in C3)
check("C5 N1 / N2 (E28 SCHED, verbatim: no 262k ceiling) in CELL 3; N3 + plain_order in CELL 6/7",
      lambda: "E28 SCHED N1" in C3 and "E28 SCHED N2" in C3 and "NB5_WINDOW) + 512" not in C3
      and "E28 SCHED N3" in READ and 'os.environ.get("E28_SCHED", "").strip() == "1" or P.get("plain_order")' in READ)
check("C6 nb5_qab_phase_record uses the phase's TASK_IDS when no task list; CELL 9 paired list = NB5_ALL_IDS",
      lambda: 'P.get("task_list") or list(TASK_IDS)' in ENG and 'P["task_list"]' not in "".join(READABLE[1:])
      and "(P.get(\"task_list\") or NB5_ALL_IDS)" in C9)
check("C7 CELL 9: the E28 report (e28_k100_report.json) replaces the K100Q quality report; in the bundle; no k100_quality_report",
      lambda: "E28L.build_e28_report(" in C9 and 'BASE_OUT / "e28_k100_report.json"' in C9 and "k100_quality_report" not in C9
      and "K100Q.build_quality_report" not in C9)
check("C8 kept K100Q pieces: Q-defer apply / gate / read-back, k100_setup, the boot gates, the K3 pins, the relaunch config, "
      "the deadman thread name, the fixed-window runner hook",
      lambda: all(s in ENG for s in ("def k100_qdefer_apply(", "def k100_qd_gate(", "def k100_qd_readback(", "def k100_setup(",
                                      "def k100_boot_gates(", "NB5_ACTIVE_BOOT.update(", "K100_PIN_STEPS_GIB"))
      and 'name="k100q_deadman"' in C1 and 'if P.get("fixed_window_sec"):' in C8 and "k100_setup()" in C8)
check("C9 the W4 / 27B reference tables are not embedded (K100Q_REF = K100Q_REF27 = None)",
      lambda: "K100Q_REF = K100Q_REF27 = None" in C1 and "nb5_fp8_reference: embedded" not in C1)

# ------------------------------------------------------------------ D N1 / N2 numbers
def _n1_n2():
    t = ast.parse(C3)
    fn = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "_nb5_peak_len")
    g = {"NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192, "NB5_WINDOW": 130560, "MAX_LEN": 131072,
         "NB5_BLOCKS": {"peak_b0": 120384, "peak_b1": 120384, "b0": 36, "b1": 36, "start": 8, "f": 6, "block": 4096},
         "NB5_ADMIT_MODE": "projected"}
    exec(compile(ast.Module([fn], []), "<n1>", "exec"), g)
    run = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "_nb5_session_run")
    stmts = [s for s in run.body if isinstance(s, ast.Assign) and any(getattr(x, "id", None) in ("_e28p", "need") for x in s.targets)]
    assert [x.targets[0].id for x in stmts] == ["_e28p", "need"], [ast.dump(x)[:80] for x in stmts]
    seg = "\n".join(ast.get_source_segment(C3, s) for s in stmts)
    g["NB5L"] = type("L", (), {"cdiv": staticmethod(L.cdiv)})

    class A:
        arc2_branch = 0
    out = {}
    for lab, plan, br in (("t1", {"budget": 104000, "window": None, "phase": "A"}, 0),
                          ("b2", {"budget": 120000, "window": 130560, "phase": "B2"}, 2), ("none", None, 1)):
        a = A()
        a.arc2_branch = br
        if plan is not None:
            a._e28_plan = plan
        s = type("S", (), {})()
        s.branch, s.e28_plan = br, plan
        ns = dict(g, self=s)
        exec(compile(seg, "<n2>", "exec"), ns)
        out[lab] = (g["_nb5_peak_len"](a), ns["need"])
    return out


_D = _n1_n2()
check("D1 N1 (MAX_LEN 131,072): a T1 104k attempt's peak 120,384, a B2 120k attempt's 131,072, no plan = NB5_BLOCKS' peak",
      lambda: (_D["t1"][0], _D["b2"][0], _D["none"][0]) == (120384, 131072, 120384))
check("D2 N2: T1 36 blocks, B2 38 blocks, no plan = NB5_BLOCKS b1 (36) (in slots = the C-7 above-pool check only)",
      lambda: (_D["t1"][1], _D["b2"][1], _D["none"][1]) == (36, 38, 36))
check("D3 the capacity note at pool 479 / F 6 / gate 1.0 (recorded): 13 T1 attempts, 12 / 11 / 10 with 1 / 2 / 3 B2 in flight",
      lambda: (lambda c: c["t1_blocks"] == 36 and c["b2_blocks"] == 38 and c["t1_streams"] == 13
               and c["t1_streams_with_k_b2"] == {"0": 13, "1": 12, "2": 11, "3": 10} and c["workers"] == 21)(
          L.capacity_note(479, 4096, 6, 1.0, 104000, 120000, 131072, 131072, 21)) and FACTS["capacity"]["t1_streams"] == 13)

if "def _e28_try_reserve(key, plan):" in C3:
    def _d4():
        """N4 (the scheduler's non-blocking fit check, backfill) reserves the SAME blocks as N2 + the E28 K100 ceiling, on a
        real nb5_lib.Reservation-shaped object: T1 36, B2 38; a key already held -> True; a need that does not fit -> False."""
        import threading as _th
        t = ast.parse(C3)
        fn = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "_e28_try_reserve")

        class Res:
            def __init__(self, cap):
                self.cap, self.used, self.held, self.f_extra, self.live_hold = cap, 0, {}, 0, None
                self.stats = {"idle_grants": 0, "grants": 0, "max_used": 0, "max_held": 0}
                self.cond = _th.Condition()
        g = {"NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192, "NB5_WINDOW": 130560, "MAX_LEN": 131072, "NB5_ADMIT_MODE": "projected",
             "NB5_BLOCKS": {"b0": 36, "b1": 36, "start": 8, "f": 6, "block": 4096}, "NB5L": type("L", (), {"cdiv": staticmethod(L.cdiv)})}
        exec(compile(ast.Module([fn], []), "<n4>", "exec"), g)
        g["NB5_RES"] = r = Res(479)
        ok1 = g["_e28_try_reserve"](("a", 0), {"budget": 104000, "window": None, "branch": 0})
        ok2 = g["_e28_try_reserve"](("b", 2), {"budget": 120000, "window": 130560, "branch": 2})
        held = dict(r.held)
        again = g["_e28_try_reserve"](("a", 0), {"budget": 104000, "window": None, "branch": 0})
        r.used = 479 - 10
        nofit = g["_e28_try_reserve"](("c", 1), {"budget": 104000, "window": None, "branch": 1})
        return ok1 and ok2 and held == {("a", 0): 36, ("b", 2): 38} and again and nofit is False
    check("D4 N4 (backfill fit check): T1 reserves 36 blocks, B2 38 (= N2), held key -> True, no room -> False", _d4)

    def _d5():
        """slots (the E28 admission): N4 takes ONE slot of the 21 for any attempt (T1 or B2), refuses when all 21 are held."""
        import threading as _th
        t = ast.parse(C3)
        fn = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "_e28_try_reserve")

        class Res:
            def __init__(self, cap):
                self.cap, self.used, self.held, self.f_extra, self.live_hold = cap, 0, {}, 0, None
                self.stats = {"idle_grants": 0, "grants": 0, "max_used": 0, "max_held": 0}
                self.cond = _th.Condition()
        g = {"NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192, "NB5_WINDOW": 130560, "MAX_LEN": 131072, "NB5_ADMIT_MODE": "slots",
             "NB5_BLOCKS": {"b0": 36, "b1": 36, "start": 8, "f": 6, "block": 4096}, "NB5L": type("L", (), {"cdiv": staticmethod(L.cdiv)})}
        exec(compile(ast.Module([fn], []), "<n4>", "exec"), g)
        g["NB5_RES"] = r = Res(21)
        a = g["_e28_try_reserve"](("a", 0), {"budget": 104000, "window": None, "branch": 0})
        b = g["_e28_try_reserve"](("b", 2), {"budget": 120000, "window": 130560, "branch": 2})
        held = dict(r.held)
        r.used = 21
        full = g["_e28_try_reserve"](("c", 1), {"budget": 104000, "window": None, "branch": 1})
        return a and b and held == {("a", 0): 1, ("b", 2): 1} and full is False
    check("D5 slots admission (E28): N4 = one slot per attempt (T1 and B2 alike), refused only when all 21 slots are held", _d5)
else:
    skip("D4 N4 fit check", "the scheduler module has no N4 edit")

# ------------------------------------------------------------------ E e28k100_lib
_ARGV8 = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","draft_lm_head_quant":"int8"}']
_ARGV0 = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}']
_NA8 = ("INFO non-default args: {'max_num_seqs': 21, 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 3, "
        "'draft_sample_method': 'probabilistic', 'draft_lm_head_quant': 'int8'}}")
_HEAD = "(Worker_PP1_TP0 pid=1) INFO Draft lm_head: INT8 weight-only copy of the 124160x2560 shard (305.1 MiB; the target keeps its torch.bfloat16 head)"
check("E1 i8_readback: ENGAGED (argv + log spec int8 + INT8 lines), SPEC_ONLY (no line), CONTRADICTION (BF16 spec), BF16_HEAD",
      lambda: L.i8_readback(_NA8 + "\n" + _HEAD + "\n" + _HEAD.replace("TP0", "TP1"), _ARGV8, "int8")["label"] == "I8_ENGAGED"
      and L.i8_readback(_NA8, _ARGV8, "int8")["label"] == "I8_SPEC_ONLY"
      and L.i8_readback("", _ARGV0, "int8")["label"] == "I8_CONTRADICTION"
      and L.i8_readback("", _ARGV0, None)["label"] == "BF16_HEAD"
      and L.i8_readback(_HEAD, _ARGV0, None)["label"] == "BF16_HEAD_CONTRADICTION"
      and L.i8_readback(_NA8 + "\n" + _HEAD, _ARGV8, "int8")["head_mib_total"] == 305.1)
check("E2 i8_failure_reason: Merlin's refusal / OOM / other, every one labelled I8_UNSUPPORTED_ON_K100",
      lambda: L.i8_failure_reason("SERVER_DIED", "ValueError: draft_lm_head_quant needs a plain unquantized ParallelLMHead")["kind"]
      == "DRAFT_LM_HEAD_QUANT_REFUSED" and L.i8_failure_reason("SERVER_DIED", "torch.OutOfMemoryError: CUDA out of memory")["kind"]
      == "OOM_WITH_INT8_HEAD" and L.i8_failure_reason("LAUNCH_FAILED", "")["kind"] == "BOOT_FAILED_WITH_INT8_HEAD:LAUNCH_FAILED"
      and L.i8_failure_reason("X", "")["label"] == "I8_UNSUPPORTED_ON_K100")
check("E3 i8_static_check (synthetic): tied embeddings / misaligned shard = UNSUPPORTED_STATIC; no Merlin source = INCONCLUSIVE",
      lambda: L.i8_static_check({"tie_word_embeddings": True, "vocab_size": 248320, "hidden_size": 2560},
                                {"block_name_to_quantize": "model.language_model.layers"})["verdict"] == "UNSUPPORTED_STATIC"
      and L.i8_static_check({"tie_word_embeddings": False, "vocab_size": 248320, "hidden_size": 2500},
                            {"block_name_to_quantize": "model.language_model.layers"})["verdict"] == "UNSUPPORTED_STATIC"
      and L.i8_static_check({"tie_word_embeddings": False, "vocab_size": 248320, "hidden_size": 2560},
                            {"block_name_to_quantize": "model.language_model.layers"})["verdict"] == "INCONCLUSIVE")
_MS = {rel: (MERLIN_SRC / rel) for rel in ("vllm/models/qwen4_exp/nvidia/mtp.py", "vllm/models/qwen4_exp/nvidia/model.py",
                                           "vllm/v1/worker/gpu/spec_decode/speculator.py")}
if (KLEE_CFG / "config.json").is_file() and all(p.is_file() for p in _MS.values()):
    check("E4 i8_static_check on the REAL klee100 config + quantization_config and the REAL Merlin v3.0.0 source: SUPPORTED_STATIC "
          "(head not tied, shard 124160 x 2560 Marlin-aligned, lm_head outside the quantized blocks, Qwen4ExpMTP opts in)",
          lambda: (lambda r: r["verdict"] == "SUPPORTED_STATIC" and r["shard"] == [124160, 2560] and all(r["checks"].values()))(
              L.i8_static_check(json.loads((KLEE_CFG / "config.json").read_text(encoding="utf-8")),
                                json.loads((KLEE_CFG / "quantization_config.json").read_text(encoding="utf-8")),
                                {rel: p.read_text(encoding="utf-8") for rel, p in _MS.items()})))
else:
    skip("E4 i8_static_check on the real klee100 config + Merlin source", f"{KLEE_CFG} / {MERLIN_SRC} missing")

K100Q_FIX = os.environ.get("E28KT_K100Q_FIXTURE")


def k100q_fixture(out_dir, sol_path):
    """K100Q's REAL GPU output reduced to what build_e28_report reads (the kit carries this fixture; the 400-kB kit limit):
    the phase / per-config / boot records, the run report's score keys, the generated attempts, the 60 branch records
    (outcome keys + accepted / train / test outputs of every candidate), the 30 tasks' eval solutions."""
    rep = json.loads((out_dir / "nb5_report.json").read_text(encoding="utf-8"))
    d = out_dir / "phase_Q_K100_FP8_K3"
    ph = {k: v for k, v in rep["phases"]["Q_K100_FP8_K3"].items() if k in ("status", "solver_error", "error", "counters",
                                                                              "solver_window", "phase_window", "steady_mixed", "b_summary")}
    pc = {k: v for k, v in rep["quality_ab"]["per_config"]["Q_K100_FP8_K3"].items() if k in ("window", "speed", "capacity_levers")}
    b = rep["boots"][-1]
    boot = {k: b.get(k) for k in ("tag", "status", "pin_used", "ready_sec", "t0_sec", "levers_dropped")}
    boot["tries"] = [{k: t_.get(k) for k in ("name", "status", "pin_gib", "kv_bytes", "max_num_seqs_launch", "levers", "ready_sec",
                                             "ladder", "k100_gates")} for t_ in b.get("tries") or []]
    for t_ in boot["tries"]:
        t_["k100_gates"] = {"fatal": (t_.get("k100_gates") or {}).get("fatal")}
    rr0 = json.loads((d / "e12_run_report.json").read_text(encoding="utf-8"))
    rr = {k: rr0.get(k) for k in ("eval_query_hits", "eval_query_total", "eval_pass1_hits", "eval_pass2_hits", "eval_task_norm_pct")}
    ga = json.loads((d / "e12_generated_attempts.json").read_text(encoding="utf-8"))
    recs = {}
    for f in sorted((d / "duck").glob("*/branch_result.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        k = {x: r.get(x) for x in ("task_id", "branch", "accepted", "b_budget", "b_events", "generated_tokens", "note", "nb5_verdict",
                                   "progress", "status", "t0", "t1") if x in r}
        for nm in ("submissions", "checked"):
            k[nm] = [{x: s.get(x) for x in ("accepted", "train_exact", "test_outputs", "t")} for s in r.get(nm) or [] if isinstance(s, dict)]
        recs.setdefault(r["task_id"], []).append(k)
    sol_all = json.loads(Path(sol_path).read_text(encoding="utf-8"))
    samp = [{"t": s.get("t"), "preempt": s.get("preempt")} for s in (json.loads(l) for l in
            (d / "nb5_sampler.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())]
    return {"phase": ph, "per_config": pc, "boot": boot, "run_report": rr, "generated": ga, "records": recs,
            "solutions": {t: sol_all[t] for t in sorted(recs)}, "sampler": samp}


_FX = (json.loads(Path(K100Q_FIX).read_text(encoding="utf-8")) if K100Q_FIX else
       (k100q_fixture(K100Q_OUT, SOL) if (K100Q_OUT / "nb5_report.json").is_file() and SOL.is_file() else None))
if _FX is not None:
    def _e5():
        recs, sol = _FX["records"], _FX["solutions"]
        tids = sorted(recs)
        t0 = min(float(r["t0"]) for v in recs.values() for r in v)
        out = L.build_e28_report(phase=_FX["phase"], per_config=_FX["per_config"], run_report=_FX["run_report"],
                                 generated_attempts=_FX["generated"], records=recs, solutions=sol, task_ids=tids, boot=_FX["boot"],
                                 i8={"want": False}, qdefer={}, sched=None,
                                 window={"start_epoch": t0, "end_epoch": t0 + _FX["per_config"]["window"]["actual_sec"]},
                                 budgets={"T1": 104000}, phase_of=None, rerun=False, max_model_len=131072,
                                 sampler=_FX.get("sampler"))
        g, r_ = out["score"]["generated"], out["score"]["run_report"]
        assert out["status"] == "VALID", (out["status"], out.get("errors"))
        assert out["score"]["agree"] is True and (g["eval_query_hits"], g["eval_query_total"]) == (31, 38), (g, r_)
        assert g["tasks_solved"] == 23 and len(out["attempts"]) == 60 and out["engine"]["preemptions"] == 0
        assert out["engine"]["workers"] == 21 and out["engine"]["pool_blocks"] == 479, out["engine"]
        assert out["engine"]["preemptions_sampler"] == 0 and out["attempts_with_preemption"] == 0, out["engine"]
        assert all(r["preempt_while_running"] == 0 for r in out["attempts"])
        md = L.markdown(out)
        assert md.count(chr(10) + "| ") >= 61 and "31/38" in md
        return True
    check("E5 build_e28_report on K100Q's REAL GPU output: VALID, score 31 / 38 queries == the run report, 23 / 30 solved, 60 attempt "
          "rows, preemptions 0 (counter + sampler, every attempt flag 0), workers 21, pool 479, the markdown table", _e5)
    check("E6 build_e28_report: rerun -> SKIPPED_RERUN; no phase -> NO_PHASE_RUN; a solver error -> INCOMPLETE_PHASE",
          lambda: L.build_e28_report(phase={}, per_config={}, run_report={}, generated_attempts={}, records={}, solutions={},
                                     task_ids=[], boot={}, i8={}, qdefer={}, rerun=True)["status"] == "SKIPPED_RERUN"
          and L.build_e28_report(phase={}, per_config={}, run_report={}, generated_attempts={}, records={}, solutions={},
                                 task_ids=[], boot={}, i8={}, qdefer={})["status"] == "NO_PHASE_RUN"
          and L.build_e28_report(phase={"status": "DONE", "solver_error": "x"}, per_config={}, run_report={"a": 1},
                                 generated_attempts={}, records={}, solutions={}, task_ids=[], boot={}, i8={}, qdefer={})["status"]
          == "INCOMPLETE_PHASE")
else:
    skip("E5 / E6 build_e28_report on K100Q's real output", f"{K100Q_OUT} / {SOL} / E28KT_K100Q_FIXTURE missing")
check("E8 build_e28_report reads arc2_duck.SCHED_LAST['e28'] (phase counts, classes, slots, per-attempt rows)",
      lambda: (lambda o: o["scheduler"]["present"] and o["scheduler"]["started"] == {"A": 2} and o["sched_attempts"] == [{"task": "x"}]
               and o["sched_slots"] == {"x": 1})(L.build_e28_report(phase={"status": "DONE"}, per_config={}, run_report={"a": 1},
                  generated_attempts={}, records={}, solutions={}, task_ids=[], boot={}, i8={}, qdefer={},
                  sched={"rec": {"applied": True}, "e28": {"started": {"A": 2}, "attempts": [{"task": "x"}], "slots": {"x": 1}}})))
check("E7 attempt_rows: phases from rec['e28'], B2 rows (120k budget; 119,500 tokens = above T1 104,000, BUDGET_END)",
      lambda: (lambda rows: [r["phase"] for r in rows] == ["A", "C", "B2"] and L.b2_rows(rows, 120000, 104000)[0]["reached_above_t1"] is True
               and rows[2]["outcome"] == "BUDGET_END")(
          L.attempt_rows({"t1": [{"branch": 0, "e28": {"phase": "A"}, "generated_tokens": 5, "accepted": True},
                                 {"branch": 1, "e28": {"phase": "C"}, "generated_tokens": 7, "accepted": True},
                                 {"branch": 2, "e28": {"phase": "B2"}, "generated_tokens": 119500, "b_budget": 120000}]},
                         {"t1": [[[1]]]}, {"T1": 104000, "B2": 120000}, phase_of=lambda r: (r.get("e28") or {}).get("phase"))))

check("E9 per-attempt preemption flag: the sampler counter increments inside [t0, t1] of an attempt (a recompute, not a failure); "
      "a counter reset (relaunch) is ignored; no sampler -> None",
      lambda: (lambda o: [r["preempt_while_running"] for r in o["attempts"]] == [2, 0] and o["attempts_with_preemption"] == 1
               and o["engine"]["preemptions_sampler"] == 2 and o["engine"]["preemption_intervals"] == 1)(
          L.build_e28_report(phase={"status": "DONE"}, per_config={}, run_report={"a": 1}, generated_attempts={}, solutions={},
                             records={"t": [{"branch": 0, "t0": 100.0, "t1": 200.0, "generated_tokens": 5},
                                            {"branch": 1, "t0": 300.0, "t1": 400.0, "generated_tokens": 5}]},
                             task_ids=[], boot={}, i8={}, qdefer={},
                             sampler=[{"t": 90.0, "preempt": 0}, {"t": 150.0, "preempt": 2}, {"t": 250.0, "preempt": 2},
                                      {"t": 350.0, "preempt": 0}, {"t": 380.0, "preempt": 0}]))
      and L.attempt_rows({"t": [{"branch": 0}]}, {}, {})[0].get("preempt_while_running") is None)

# ------------------------------------------------------------------ G the I8 activation gate + the same-pin BF16 retry (audit 2026-11-01)


def _eng_def(name):
    fn = next(n for n in ast.parse(ENG).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.Module([fn], [])


def _gate_ns(fake=None, head="int8"):
    g = {"FAKE": fake, "NB5_BOOT_LEVERS": {"DRAFT_HEAD": head} if head else {}, "json": json, "log": lambda *a: None}
    exec(compile(_eng_def("e28_i8_gate"), "<gate>", "exec"), g)
    return g


def _g1():
    eng_log = _NA8 + "\n" + _HEAD + "\n" + _HEAD.replace("TP0", "TP1")
    na_bf16 = _NA8.replace(", 'draft_lm_head_quant': 'int8'", "")
    cases = {"engaged": (eng_log, _ARGV8), "spec_only": (_NA8, _ARGV8), "bf16_build": (na_bf16, _ARGV8),
             "contradiction": (_NA8 + "\n" + _HEAD.replace("INT8", "INT4"), _ARGV8)}
    out = {}
    for k, (lg, av) in cases.items():
        g = _gate_ns()
        trec = {"i8_readback": L.i8_readback(lg, av, "int8")}
        out[k] = (g["e28_i8_gate"](trec), trec.get("status"), (trec.get("i8_failure") or {}).get("label"))
    fake = _gate_ns(fake="fake.py")
    bf = _gate_ns(head=None)
    t_fake = {"i8_readback": {"label": "FAKE_SERVER"}}
    t_bf = {"i8_readback": L.i8_readback(na_bf16, _ARGV0, None)}
    assert out["engaged"] == (False, None, None), out
    assert out["spec_only"] == (True, "I8_READBACK_UNVERIFIED", "I8_SPEC_ONLY"), out
    assert out["bf16_build"][0] is True and out["bf16_build"][1] == "I8_READBACK_UNVERIFIED", out
    assert out["contradiction"] == (True, "I8_READBACK_UNVERIFIED", "I8_CONTRADICTION"), out
    return fake["e28_i8_gate"](t_fake) is False and "status" not in t_fake and bf["e28_i8_gate"](t_bf) is False


check("G1 e28_i8_gate: INT8 build logs (I8_ENGAGED) -> continue; INT8 argv with no head-build line, a BF16 build (log spec without "
      "int8), an INT4 head (contradiction) -> I8_READBACK_UNVERIFIED (stop); the fake server and a BF16-head boot are never gated",
      _g1)
check("G2 the gate sits right after the read-back in nb5_boot_try (before the k100 gates / probes / phases) and returns (the server "
      "stops via with_servers)",
      lambda: ENG.index('trec["i8_readback"] = ') < ENG.index("        if e28_i8_gate(trec):") < ENG.index('trec["k100_gates"] = k100_boot_gates(s, mtp_on)')
      and "        if e28_i8_gate(trec):                                # E28 K100 I8 (audit 2026-11-01): INT8 requested -> I8_ENGAGED required\n            return {}" in ENG)
if NB5LIB_P.is_file():
    import importlib.util as _ilu5
    _s5 = _ilu5.spec_from_file_location("nb5_lib_kt", NB5LIB_P)
    NB5L_T = _ilu5.module_from_spec(_s5)
    _s5.loader.exec_module(NB5L_T)

    def _ladder(script, rerun=False):
        """nb5_boot (the notebook's own def) with a scripted nb5_boot_try: each try's status, the levers / argv it got."""
        g = {"json": json, "NB5L": NB5L_T, "NB5_KV_DTYPE": "fp8", "NB5_KV_BYTES": {"fp8": 2479838700}, "NB5_BOOTS": [],
             "NB5_EXTRA_ARGS": [], "NB5_MTP_OFF_ARGS": ["--speculative-config", "None"], "FAKE": None,
             "P": dict({"pin_ladder": True, "patch_60490": True, "k100": True, "i8": True, "top_rung_seqs": 24},
                       **({"e18_ladder": True} if rerun else {})),
             "NB5_PIN": {"start_gib": None, "levers": None, "patch_60490": {"ok": True}}, "K100_PIN_STEPS_GIB": G1["K100_PIN_STEPS_GIB"],
             "NB5_LEVER_ENV": {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, "NB5_SEQS_RULE": "k100", "K100_SEQS_TABLE": G1["K100_SEQS_TABLE"],
             "left": lambda: 1e9, "E28_I8": {"want": True, "on": True, "static": None, "dropped": None, "readback": None},
             "E28_I8_ENV": {"DRAFT_HEAD": "int8"}, "E28L": L, "log": lambda *a: None, "nb5_write_report": lambda: None, "ABORT": {},
             "NB5_READY_TIMEOUT": 1800, "NB5_BOOT_LEVERS": {}, "NB5_BOOT_EXTRA": [], "NB5_BOOT_KV": None}
        seen, it = [], iter(script)

        def nb5_boot_try(trec, name, kv_bytes, phases, mtp_on):
            st = next(it)
            av = list(g["NB5_BOOT_EXTRA"])
            seen.append({"name": name, "kv_bytes": kv_bytes, "levers": dict(g["NB5_BOOT_LEVERS"]),
                         "seqs": av[av.index("--max-num-seqs") + 1] if "--max-num-seqs" in av else None})
            if st == "UNVERIFIED":
                trec["i8_readback"] = L.i8_readback(_NA8, _ARGV8, "int8")
                gg = _gate_ns()
                gg["NB5_BOOT_LEVERS"] = dict(g["NB5_BOOT_LEVERS"])
                assert gg["e28_i8_gate"](trec) is True
            elif st == "OOM":
                trec.update(status="LAUNCH_FAILED", log_text_for_ladder="torch.OutOfMemoryError: CUDA out of memory")
            else:
                trec.update(status="DONE", phases_started=True)
            return trec
        g["nb5_boot_try"] = nb5_boot_try
        exec(compile(_eng_def("nb5_boot"), "<nb5_boot>", "exec"), g)
        b = g["nb5_boot"]("merlin_fp8_mtp", [["Q", "all", None, "on", "A", "fp8"]], True, "fp8")
        return b, seen, g

    def _g3():
        b, seen, g = _ladder(["UNVERIFIED", "DONE"])
        assert [s["kv_bytes"] for s in seen] == [7933526016, 7933526016] and [s["seqs"] for s in seen] == ["24", "24"], seen
        assert seen[0]["levers"].get("DRAFT_HEAD") == "int8" and "DRAFT_HEAD" not in seen[1]["levers"] and seen[0]["name"].endswith("_i8"), seen
        assert g["E28_I8"]["on"] is False and g["E28_I8"]["dropped"]["kind"] == "READBACK_UNVERIFIED", g["E28_I8"]
        assert g["E28_I8"]["dropped"]["readback"]["label"] == "I8_SPEC_ONLY" and b["status"] == "DONE"
        assert any(d["keys"] == ["DRAFT_HEAD"] and d["why"] == "I8_UNSUPPORTED_ON_K100: READBACK_UNVERIFIED" for d in b["levers_dropped"])
        assert b["pin_used"]["kv_bytes"] == 7933526016 and b["pin_used"]["max_num_seqs_launch"] == 24
        assert [t_.get("e28_i8") for t_ in b["tries"]] == [True, False]
        b2, seen2, g2 = _ladder(["UNVERIFIED", "DONE"], rerun=True)
        return [s["kv_bytes"] for s in seen2] == [7933526016] * 2 and b2["status"] == "DONE" and g2["E28_I8"]["dropped"]["kind"] == "READBACK_UNVERIFIED"
    check("G3 I8 read-back unverified -> the SAME pin (7,933,526,016 B) and --max-num-seqs 24 with the BF16 head; the BF16 retry boots "
          "and the eval starts with BF16 (status DONE, I8 off, I8_UNSUPPORTED_ON_K100 READBACK_UNVERIFIED recorded); commit and rerun "
          "(E18 ladder)", _g3)

    def _g4():
        b, seen, g = _ladder(["UNVERIFIED", "OOM", "DONE"])
        assert [s["kv_bytes"] for s in seen] == [7933526016, 7933526016, 6859784192], seen
        assert [s["seqs"] for s in seen] == ["24", "24", "18"] and ["DRAFT_HEAD" in s["levers"] for s in seen] == [True, False, False]
        b1, seen1, g1 = _ladder(["DONE"])
        return (b["status"] == "DONE" and len(seen1) == 1 and seen1[0]["levers"].get("DRAFT_HEAD") == "int8" and g1["E28_I8"]["on"] is True
                and g1["E28_I8"]["dropped"] is None)
    check("G4 the BF16 retry fails (OOM) -> the existing pin ladder (next rung 6,859,784,192 B with K100Q's 18 seqs, BF16); an engaged INT8 boot "
          "keeps I8 (one try)", _g4)
else:
    skip("G3 / G4 the nb5_boot ladder with the gate", f"{NB5LIB_P} missing")

# ------------------------------------------------------------------ F the scheduler module on the real arc2_duck.py
if SCHED_P.is_file() and DUCK.is_file():
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("e28_sched_patch_t", SCHED_P)
    S = _ilu.module_from_spec(_s)
    _s.loader.exec_module(S)

    def _f1():
        d = Path(tempfile.mkdtemp(prefix="e28kt_"))
        try:
            shutil.copyfile(DUCK, d / "arc2_duck.py")
            assert S.tree_state(d) == "ORIGINAL"
            r = S.apply_e28_sched(d)
            assert r["applied"] == "applied" and r["ok"] and S.verify_e28_sched(d)[0] and S.tree_state(d) == "FULLY_PATCHED"
            assert S.apply_e28_sched(d)["applied"] == "already_applied"
            v = S.revert_e28_sched(d)
            assert v["reverted"] == "reverted" and S.tree_state(d) == "ORIGINAL" and (d / "arc2_duck.py").read_bytes() == DUCK.read_bytes()
            (d / "arc2_duck.py").write_bytes(DUCK.read_bytes() + b"\n# junk\n")
            try:
                S.apply_e28_sched(d)
                return False
            except RuntimeError as e:
                return "E28_SCHED_PRE_SHA_MISMATCH" in str(e)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    check("F1 e28_sched_patch on a copy of the REAL arc2_duck.py (b9f67010): apply -> FULLY_PATCHED (post sha), again = no-op, "
          "revert -> the exact original bytes, an unknown file -> PRE_SHA_MISMATCH (nothing written)", _f1)
    check("F2 the embedded module == the Build file; server_env = T1 104,000 / B2 120,000 / window 130,560; its NOTEBOOK_EDITS' new "
          "text is in the notebook (N1 / N2 in CELL 3, N3 in CELL 6/7)",
          lambda: modules()["e28_sched_patch.py"] == SCHED_P.read_text(encoding="utf-8")
          and {k: S.server_env().get(k) for k in ("E28_SCHED", "E28_T1", "E28_B2_TOKENS", "E28_B2_WINDOW")}
          == {"E28_SCHED": "1", "E28_T1": "104000", "E28_B2_TOKENS": "120000", "E28_B2_WINDOW": "130560"}
          and S.NOTEBOOK_EDITS[0][2].splitlines()[0] in C3 and S.NOTEBOOK_EDITS[1][2].splitlines()[0] in C3
          and S.NOTEBOOK_EDITS[2][2].splitlines()[1] in READ)
else:
    skip("F1 / F2 e28_sched_patch on the real arc2_duck.py", f"{SCHED_P} / {DUCK} missing")

n_bad = sum(1 for _, ok in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_bad}/{len(RESULTS)} PASS", flush=True)
sys.exit(1 if n_bad else 0)
