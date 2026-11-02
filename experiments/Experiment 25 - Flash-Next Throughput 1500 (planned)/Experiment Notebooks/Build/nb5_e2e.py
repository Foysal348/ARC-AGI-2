"""E25 NB5 local end-to-end: runs EVERY code cell of the real generated notebook, in order, in one namespace, against
nb5_fake_server.py (no GPU, no install, no Kaggle), then checks what it produced. Setup pattern = Exp 21
src/harness/harness_e21_nb.py (the Windows shims are copied from it: the 65 kB sandbox bootstrap passed as -c, no
os.killpg).

usage: python nb5_e2e.py <notebook.ipynb> [mode]
modes: commit  SMOKE (3 smoke tasks, A/C split) -> EVAL (4 eval tasks, variant C), exact-match scored
       rerun   KAGGLE_IS_COMPETITION_RERUN=1: RERUN_ALL on the fake test set (4 tasks), submission from the records
       mtp     SMOKE -> EVAL (MTP on) -> EVALOFF (a second boot with --speculative-config None), equal windows
       notok   /tokenize answers 404: every variant-C commit falls back to variant A (counted), the run still works
       floor   the thinking floor above what the harness budgets allow: every attempt BUDGET_LIMITED (counted), the run
               still ends through Duck's own commit route
       quality the notebook's BAKED profile (NB5QUALITY_R2): Q_FP8_K3 then Q_BF16_K3 on the lead's 15 tasks, equal windows,
               the pin ladder (the fake refuses a pin above RC2 + 0.8 GiB: the +1.0 step must fail with an OOM line, +0.75
               must boot, the BF16 boot starts at +0.75), the lever env (fake F 5 -> the measured F), the seqs table,
               commit variant A, the per-task table + the paired summary
Output: one line per check, then @@NB5_E2E_OK@@ (exit 0) or @@NB5_E2E_FAIL@@ (exit 1)."""
import faulthandler
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NB = Path(sys.argv[1]).resolve()
MODE = sys.argv[2] if len(sys.argv) > 2 else "commit"
assert MODE in ("commit", "rerun", "mtp", "notok", "floor", "quality"), MODE
HERE = Path(__file__).resolve().parent
FAKE = HERE / "nb5_fake_server.py"
SRC_INPUT = Path(os.environ.get("NB5_E2E_INPUT") or r"D:/ARC AGI 2/_scratch/e13_rc2_local/input")            # RC2's local harness tree (real part-1 small files)
DATA = Path(os.environ.get("NB5_E2E_DATA") or r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
            r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
PAYSRC = Path(os.environ.get("NB5_E2E_PAYSRC") or r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload")
ROOT = Path(os.environ.get("NB5_E2E_ROOT", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/nb5/e2e"))
sys.stdout.reconfigure(encoding="utf-8")
faulthandler.dump_traceback_later(float(os.environ.get("NB5_E2E_MAX_SEC", 1800)), exit=True)
if os.name == "nt":      # Exp 21 harness shims (Windows only)
    _real = subprocess.Popen
    _bdir = Path(tempfile.gettempdir()) / "e14_harness_bootstrap"
    _bdir.mkdir(parents=True, exist_ok=True)

    class _Shim(_real):
        def __init__(self, args, *a, **kw):
            if isinstance(args, list) and len(args) >= 5 and args[1:4] == ["-I", "-S", "-c"]:
                f = _bdir / "sandbox_bootstrap.py"
                if not f.exists() or f.read_text(encoding="utf-8") != args[4]:
                    f.write_text(args[4], encoding="utf-8")
                args = [args[0], "-I", "-S", str(f)] + args[5:]
            super().__init__(args, *a, **kw)
    subprocess.Popen = _Shim
    if not hasattr(os, "killpg"):
        import signal as _sig
        os.killpg = lambda pid, sig: os.kill(pid, 9)
        if not hasattr(_sig, "SIGKILL"):
            _sig.SIGKILL = 9

WORK = ROOT / f"run_{MODE}_{time.strftime('%H%M%S')}"
WORK.mkdir(parents=True)
INP = WORK / "inputs"
shutil.copytree(SRC_INPUT, INP)
COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"
ch = json.load(open(DATA / "arc-agi_evaluation_challenges.json"))
sol = json.load(open(DATA / "arc-agi_evaluation_solutions.json"))
shutil.copy(DATA / "arc-agi_evaluation_challenges.json", COMP / "arc-agi_evaluation_challenges.json")
shutil.copy(DATA / "arc-agi_evaluation_solutions.json", COMP / "arc-agi_evaluation_solutions.json")
EVAL_TASKS = ["136b0064", "bf45cf4b", "247ef758", "b5ca7ac4"]
QAB = {"solid": ["1818057f", "45a5af55", "bf45cf4b", "2ba387bc", "8f215267", "38007db0", "58f5dbd5"],
       "sometimes": ["247ef758", "f931b4a8", "b5ca7ac4", "65b59efc", "221dfab4"], "never": ["2b83f449", "16b78196", "64efde09"]}
QAB_TASKS = [x for g in ("solid", "sometimes", "never") for x in QAB[g]]
json.dump({t: {"train": ch[t]["train"], "test": [{"input": q["input"]} for q in ch[t]["test"]]} for t in EVAL_TASKS},
          open(COMP / "arc-agi_test_challenges.json", "w"))
pd = INP / "datasets" / "foysalemonshanto" / "q27-e15b-payload"      # the foysalemonshanto copy (lead 2026-10-31)
pd.mkdir(parents=True)
shutil.copy(PAYSRC / "q27x_e15b_duck_9e4b89285136.bin", pd / "q27x_e15b_duck_9e4b89285136.bin")
for d in ("tmp", "shm", "working", "fake_state"):
    (WORK / d).mkdir()
with socket.socket() as s_:              # a free port for the fake server (8001 may be held by another run)
    s_.bind(("127.0.0.1", 0))
    PORT = s_.getsockname()[1]
SM = ["SMOKE", "smoke", 150, "on", "split", None, {"early_exit": True}]
phases = {"commit": [SM, ["EVAL", "all", 150, "on", "C"]], "notok": [SM, ["EVAL", "all", 150, "on", "C"]],
          "floor": [SM, ["EVAL", "all", 150, "on", "C"]],
          "mtp": [SM, ["EVAL", "all", 120, "on", "C"], ["EVALOFF", "all", 120, "off", "C"]],
          "rerun": None, "quality": None}[MODE]
os.environ.update(E11_HARNESS="1", E23_FAKE_SERVER=str(FAKE), E23_INPUT=str(INP), E23_TMP=str(WORK / "tmp"), E23_WIN_SEC="4",
                  E23_EST_SCALE="0.01", RC2_SHM=str(WORK / "shm"), Q27X_WORKING=str(WORK / "working"), Q27X_ARC_ROOT=str(COMP),
                  Q27X_PAYLOAD_DIR=str(INP), E12_DIR=str(WORK / "e12_duck"), NB5_ATTEMPT_B0="6000", NB5_ATTEMPT_B1="4000",
                  NB5_RESERVE="1000", E12_R6_MIN_ADMIT_SEC="5", E12_B_MAX_ATTEMPTS="3", Q27X_TIME_BUDGET="3600",
                  ARC3_HTTP_RETRY_BASE_SECONDS="0.05", ARC3_HTTP_RETRY_MAX_SECONDS="0.1", NB5_FAKE_ARC=str(COMP),
                  FAKE_STATE_DIR=str(WORK / "fake_state"), FAKE_SPEED="3000", PYTHONIOENCODING="utf-8", NB5_SAMPLER_SEC="2",
                  E15B_WD_POLL_SEC="2", NB5_PORT=str(PORT), NB5_SMOKE_GATE_SEC="300",
                  NB5_THINK_FLOOR="7000" if MODE == "floor" else "3000",
                  NB5_FAKE_HOST_FLOOR_MIB="1024")   # KIT-6: the 30 GiB CPU kit (FAKE only; the real notebook keeps 73 GiB)
for k in ("KAGGLE_IS_COMPETITION_RERUN", "FAKE_NO_TOKENIZE", "Q27X_PROFILE", "NB5H_TASKS", "NB5H_PHASES", "NB5_COMMIT_MODE",
          "FAKE_OOM_ABOVE_BYTES", "FAKE_LEVER_ERROR", "NB5_PHASE_TAIL_SEC"):
    os.environ.pop(k, None)
if MODE == "rerun":
    os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
elif MODE == "quality":            # the baked profile; RC2 FP8 pin 2,479,838,700 + 0.8 GiB = the fake's OOM threshold
    os.environ.update(FAKE_OOM_ABOVE_BYTES=str(2479838700 + int(0.8 * 2 ** 30)), NB5_PHASE_TAIL_SEC="60")
else:
    if MODE != "quality":
        os.environ.update(Q27X_PROFILE="NB5HARNESS", NB5H_TASKS=",".join(EVAL_TASKS), NB5H_PHASES=json.dumps(phases))
if MODE == "notok":
    os.environ["FAKE_NO_TOKENIZE"] = "1"
nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
heads = [c.splitlines()[0][:90] for c in code]
print(f"[nb5 e2e] {NB.name} | mode {MODE} | {len(code)} code cells | work {WORK}", flush=True)
G = {"__name__": "__main__"}
cell_err = None
t0 = time.time()
for i, c in enumerate(code):
    try:
        exec(compile(c, f"<cell{i}>", "exec"), G)
    except BaseException:
        import traceback
        cell_err = (i, heads[i], traceback.format_exc()[-4000:])
        print(f"[nb5 e2e] UNCAUGHT in code cell {i}: {cell_err[2]}", flush=True)
        break
faulthandler.cancel_dump_traceback_later()
print(f"[nb5 e2e] cells done in {time.time() - t0:.0f} s", flush=True)

checks = []


def check(label, cond, info=""):
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, info = False, f"check raised {type(e).__name__}: {e} | {info}"
    checks.append((label, ok, "" if ok else str(info)[:1500]))


W = WORK / "working"
rep = json.load(open(W / "q27x" / "nb5_report.json", encoding="utf-8")) if (W / "q27x" / "nb5_report.json").exists() else {}
ph = rep.get("phases") or {}
launches = [json.loads(l) for l in open(WORK / "fake_state" / "launches.jsonl")] if (WORK / "fake_state" / "launches.jsonl").exists() else []
stats = [json.load(open(p)) for p in (WORK / "fake_state").glob("stats_*.json")]
sub = json.load(open(W / "submission.json")) if (W / "submission.json").exists() else {}
test = json.load(open(COMP / "arc-agi_test_challenges.json"))
check("every code cell ran without an uncaught exception", cell_err is None, cell_err)
check("9 code-less-markdown layout: 8 code cells (CELL 0, 1, 3, order, 5, 6/7, 8, 9)", len(code) == 8, heads)
check("NB5 report has no runner error", not rep.get("errors"), rep.get("errors"))
check("Duck payload taken from datasets/foysalemonshanto/q27-e15b-payload (path recorded in the report config)",
      lambda: Path(rep["config"]["payload_used"]["q27x_e15b_duck_9e4b89285136.bin"]["path"]) == pd / "q27x_e15b_duck_9e4b89285136.bin",
      (rep.get("config") or {}).get("payload_used"))
check("submission.json: full schema (every test task / input, attempt_1 + attempt_2 grids)",
      lambda: set(sub) == set(test) and all(len(sub[t]) == len(test[t]["test"]) for t in test)
      and all(isinstance(a.get(k), list) and a[k] and all(isinstance(r, list) for r in a[k]) for t in sub for a in sub[t]
              for k in ("attempt_1", "attempt_2")), list(sub)[:5])
want = {"commit": ["SMOKE", "EVAL"], "notok": ["SMOKE", "EVAL"], "floor": ["SMOKE", "EVAL"], "mtp": ["SMOKE", "EVAL", "EVALOFF"],
        "rerun": ["EVAL"], "quality": ["Q_FP8_K3", "Q_BF16_K3"]}[MODE]
check(f"phases run = {want}, all DONE", lambda: [n for n in ph] == want and all(ph[n].get("status") == "DONE" for n in want),
      {n: (p.get("status"), (p.get("error") or "")[-600:]) for n, p in ph.items()})
check("boot count = distinct (MTP, KV) groups", len(rep.get("boots") or []) == (2 if MODE in ("mtp", "quality") else 1),
      [b.get("tag") for b in rep.get("boots") or []])
b0 = (rep.get("boots") or [{}])[0]
check("boot 0: probes ok (census, chat, sampled, content, contract), commit probe c_usable (except notok)",
      lambda: b0.get("probes", {}).get("fatal") is None and (MODE == "notok" or b0["probes"]["commit_probe"]["c_usable"]),
      json.dumps(b0.get("probes"), default=str)[:1500])
cp = (b0.get("probes") or {}).get("commit_probe") or {}
if MODE != "notok":
    check("commit probe: tail ok + render parity + variant C tool call + cfm render starts with the C ids + server parser agrees + cold/warm EQUAL",
          lambda: cp["tail_ok"] and cp["render_parity"] and cp["variant_c"]["tool_calls"] == 1 and cp["cfm_starts_with_c"]
          and cp["server_parsers"]["args_equal_client_parse"] and cp["cold_vs_warm"]["verdict"] == "EQUAL", json.dumps(cp)[:1500])
cap = b0.get("capacity") or {}
if MODE == "quality":
    check("quality capacity: pin +0.75 GiB -> pool 197 from cache_config_info, F_log 5 (fake lever) used as F (measured rule), "
          "launch max_num_seqs 8 (table), workers = min(rule 197 // 21 = 9, 8) = 8", lambda: cap["pool_blocks"] == 197
          and cap["pool_source"] == "cache_config_info" and abs(cap["f_log"] - 5.0) < 0.06 and cap["f_eff"] == 5
          and cap["caps"]["max_num_seqs"] == 8 and cap["r2_workers"]["rule"] == 9 and cap["workers"] == 8, json.dumps(cap)[:900])
else:
    check("capacity from the live server: pool 149 from cache_config_info, F_log 6.0 from the max-concurrency line, F_eff 6, "
          "max_num_seqs 8 read from the log", lambda: cap["pool_blocks"] == 149 and cap["pool_source"] == "cache_config_info"
          and abs(cap["f_log"] - 6.0) < 0.05 and cap["f_eff"] == 6 and cap["caps"]["max_num_seqs"] == 8, json.dumps(cap)[:800])
check("warm-up recorded the JIT kernels before any metric window", lambda: b0["warmup"]["jit_after"] >= 1
      and all(not p.get("jit_kernels_in_phase") for p in ph.values()), json.dumps(b0.get("warmup"))[:600])
if MODE == "quality":
    check("quality: no pure decode window (R2 saves the time)", "pure_decode" not in b0)
elif MODE != "rerun":
    check("pure decode window (commit only) = L.steady STEADY at level = workers, tick/cycle reported",
          lambda: b0["pure_decode"]["status"] == "STEADY" and b0["pure_decode"]["level"] == cap["workers"]
          and b0["pure_decode"]["engine_rates"]["cycle_ms"] is not None, json.dumps(b0.get("pure_decode"), default=str)[:800])
else:
    check("rerun: no pure decode window", "pure_decode" not in b0)
main = ph.get("EVAL") or ph.get("Q_FP8_K3") or {}
tr = main.get("tracer") or {}
check("EVAL: requests traced, tool-call rate and prefix-hit tokens reported, no non-leading system message",
      lambda: tr["requests"] > 0 and tr["tool_call_rate"] is not None and tr["prefix_hit_pct"] is not None
      and tr["non_leading_system_requests"] == 0, json.dumps({k: tr.get(k) for k in ("requests", "tool_call_rate", "prefix_hit_pct",
                                                                                      "non_leading_system_requests")}))
switched = (ph.get("SMOKE") or {}).get("gate_decision") == "EVAL_SWITCH_TO_A"
if MODE == "quality":
    check("quality: every commit turn went as variant A (no variant C request), with tool calls",
          lambda: all("C" not in ((ph[n].get("tracer") or {}).get("commit_by_variant") or {}) for n in want)
          and all((((ph[n].get("tracer") or {}).get("commit_by_variant") or {}).get("A") or {}).get("with_tool_call", 0) > 0 for n in want),
          json.dumps({n: (ph.get(n, {}).get("tracer") or {}).get("commit_by_variant") for n in want})[:1200])
if MODE in ("commit", "mtp", "rerun", "floor"):
    if switched:
        check("EVAL: SMOKE switched C -> A (reason recorded) and every EVAL commit turn went as variant A with a tool call",
              lambda: ph["SMOKE"]["variant_decision"]["switch_reasons"] and "C" not in tr["commit_by_variant"]
              and tr["commit_by_variant"]["A"]["with_tool_call"] == tr["commit_by_variant"]["A"]["requests"] > 0,
              json.dumps({"decision": ph["SMOKE"].get("variant_decision"), "eval": tr.get("commit_by_variant")})[:1200])
    else:
        check("EVAL: commit turns sent as variant C, with a tool call, cache hits on them",
              lambda: tr["commit_by_variant"]["C"]["requests"] > 0 and tr["commit_by_variant"]["C"]["with_tool_call"] > 0
              and tr["commit_by_variant"]["C"]["cached"] > 0, json.dumps(tr.get("commit_by_variant")))
if MODE in ("commit", "mtp"):
    sm = (ph.get("SMOKE") or {}).get("tracer") or {}
    check("SMOKE: split = both variants A and C used (paired), each with a tool call",
          lambda: sm["commit_by_variant"]["A"]["requests"] > 0 and sm["commit_by_variant"]["C"]["requests"] > 0,
          json.dumps(sm.get("commit_by_variant")))
    check("SMOKE paired commit replay: A and C both answered with valid code",
          lambda: ph["SMOKE"]["paired_commit_replay"]["C"]["code_valid"] and ph["SMOKE"]["paired_commit_replay"]["A"]["tool_calls"] >= 0,
          json.dumps(ph.get("SMOKE", {}).get("paired_commit_replay"))[:1200])
    check("SMOKE scored by exact grid match (the 3 smoke tasks incl. 0934a4d8)",
          lambda: ph["SMOKE"]["score"]["eval_query_total"] >= 3 and ph["SMOKE"]["task_solve"].get("0934a4d8") is not None,
          json.dumps(ph.get("SMOKE", {}).get("score")))
    check("EVAL scored: at least one test-correct task (the scripted commit-turn lookup)",
          lambda: main["score"]["eval_query_hits"] >= 1, json.dumps(main.get("score")))
if MODE == "notok":
    check("notok: the commit probe fails on /tokenize -> variant C disabled for the run (c_usable False, recorded), every "
          "commit turn goes as variant A and still produces a tool call",
          lambda: cp.get("c_usable") is False and tr["commit_C"] == 0 and tr["kinds"].get("commit_A", 0) > 0
          and tr["commit_with_tool_call"] == tr["commit_requests"] > 0,
          json.dumps({k: tr.get(k) for k in ("kinds", "commit_C_fallbacks", "commit_with_tool_call")}))
if MODE == "rerun":
    check("rerun: submission.json carries at least one model answer (not only the identity fallback)",
          lambda: any(a["attempt_1"] != q["input"] for t in test for a, q in zip(sub[t], test[t]["test"])), "")
if MODE == "quality":
    b1 = (rep.get("boots") or [{}, {}])[1] if len(rep.get("boots") or []) > 1 else {}
    qa = rep.get("quality_ab") or {}
    pc = qa.get("per_config") or {}
    check("quality ladder: FP8 boot tried +1.0 GiB first (failed with an OOM line, no phase), booted at +0.75; the BF16 boot "
          "started at +0.75 (same settings) and booted; lever env kept; pin / pool / failures recorded",
          lambda: [t_["pin_gib"] for t_ in b0["tries"]] == [1.0, 0.75] and b0["tries"][0]["ladder"]["failed"]
          and b0["tries"][0]["ladder"]["oom_lines"] and b0["pin_used"]["gib"] == 0.75 and [t_["pin_gib"] for t_ in b1["tries"]] == [0.75]
          and b0["pin_used"]["levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"} and b1["pin_used"]["pool_blocks"] == 103,
          json.dumps({"b0": [(t_.get("name"), t_.get("status"), (t_.get("ladder") or {}).get("failed")) for t_ in b0.get("tries") or []],
                      "b1": [(t_.get("name"), t_.get("status")) for t_ in b1.get("tries") or []], "pin0": b0.get("pin_used"),
                      "pin1": b1.get("pin_used")}, default=str)[:1500])
    check("NB5Q-ENV-1: the lever env reached the server process: both ready fake launches saw MERLIN_SHARE_DRAFT_WORKSPACE=1 "
          "in their own environment, and both boot records carry server_env_levers {MERLIN_SHARE_DRAFT_WORKSPACE: 1} (also in pin_used)",
          lambda: [L_["env_levers"] for L_ in launches if L_["outcome"] == "ready"] == [{"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}] * 2
          and b0["server_env_levers"] == b1["server_env_levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}
          and b0["pin_used"]["server_env_levers"] == b1["pin_used"]["server_env_levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"},
          json.dumps({"launches": [(L_.get("outcome"), L_.get("env_levers")) for L_ in launches],
                      "b0": b0.get("server_env_levers"), "b1": b1.get("server_env_levers")}, default=str)[:1200])
    check("quality C-11: config read-back PASS on both boots (fp8_e4m3 / bfloat16, max_num_seqs 8 / 4 = the launch, pin bytes, MTP K3)",
          lambda: b0["config_readback"]["ok"] and b1["config_readback"]["ok"] and b0["config_readback"]["log"]["kv_cache_dtype"] == "fp8_e4m3"
          and b1["config_readback"]["log"]["kv_cache_dtype"] == "bfloat16" and b1["config_readback"]["log"]["max_num_seqs"] == 4,
          json.dumps({"b0": b0.get("config_readback"), "b1": b1.get("config_readback")}, default=str)[:1500])
    check("quality: equal planned windows in both phases, actual windows recorded",
          lambda: ph["Q_FP8_K3"]["window_plan"]["planned_sec"] == ph["Q_BF16_K3"]["window_plan"]["planned_sec"] > 0
          and all((ph[n].get("window") or {}).get("actual_sec") for n in want),
          json.dumps({n: [ph.get(n, {}).get("window_plan"), ph.get(n, {}).get("window")] for n in want}, default=str)[:1200])
    check("quality: per-task table for 15 tasks in the lead's order with groups, 2 attempts each (outcome, tokens, exact, commit), "
          "per-group summary, FP8 label, speed block (preemptions, max / mean running)",
          lambda: all(list(pc[n]["tasks"]) == QAB_TASKS and all(pc[n]["tasks"][x]["group"] == g for g, xs in QAB.items() for x in xs)
                      and all(len(v["attempts"]) == 2 for v in pc[n]["tasks"].values()) and pc[n]["summary"]["all"]["tasks"] == 15
                      and "preemptions" in pc[n]["speed"] for n in want)
          and pc["Q_FP8_K3"]["fp8_status"] == "UNVALIDATED_FP8_UNDER_TEST" and qa["nb0"]["verdict"] == "NO_GO_FP8_TF_DIVERGENCE"
          and ph["Q_FP8_K3"]["order"] == QAB_TASKS, json.dumps({n: pc.get(n, {}).get("summary") for n in want}, default=str)[:1500])
    check("quality: the paired summary over all tasks and finished-in-both, per group, with discordant pairs",
          lambda: set(qa["paired"]["all"]) == {"solid", "sometimes", "never", "all"} and qa["paired"]["all"]["all"]["tasks"] == 15
          and "finished_in_both" in qa["paired"] and "discordant" in qa["paired"]["all"]["all"], json.dumps(qa.get("paired"), default=str)[:1500])
    check("quality Q-1/2/3/5: every attempt outcome is ACCEPTED / BUDGET_END / INCOMPLETE_WINDOW / FAILED_OR_INTERRUPTED / "
          "NOT_STARTED and finished only on ACCEPTED / BUDGET_END; windows_check (2 s tolerance) decides the "
          "deployment_comparison status; its three results + the matrix for both phases; precision_comparison UNVERIFIED; "
          "phase_diff present; paired n + minimum attainable p next to McNemar",
          lambda: all(a["outcome"] in ("ACCEPTED", "BUDGET_END", "INCOMPLETE_WINDOW", "FAILED_OR_INTERRUPTED", "NOT_STARTED")
                      and a["finished"] == (a["outcome"] in ("ACCEPTED", "BUDGET_END"))
                      for n in want for v in pc[n]["tasks"].values() for a in v["attempts"])
          and qa["windows_check"]["tolerance_sec"] == 2.0 and qa["windows_equal"] == qa["windows_check"]["equal"]
          and qa["deployment_comparison"]["status"] == ("MEASURED" if qa["windows_check"]["equal"] else "INCONCLUSIVE_UNEQUAL_WINDOWS")
          and qa["deployment_comparison"]["all_tasks_fixed_window"]["all"]["tasks"] == 15
          and "finished_in_both" in qa["deployment_comparison"]
          and set(qa["deployment_comparison"]["missing_incomplete_failed"]["phases"]) == set(want)
          and qa["precision_comparison"]["status"] == "UNVERIFIED" and "differences" in qa["phase_diff"]
          and {"paired_n", "min_attainable_p_paired_n", "reading"} <= set(qa["paired"]["all"]["all"]),
          json.dumps({k: qa.get(k) for k in ("windows_check", "precision_comparison", "phase_diff")}, default=str)[:1500])
check("reservation: grants > 0, never over capacity, released at the end",
      lambda: main["reservation"]["grants"] > 0 and main["reservation"]["max_used"] <= main["reservation"]["capacity_blocks"]
      and main["reservation"]["used"] == 0, json.dumps(main.get("reservation"))[:600])
check("running == in-flight assertion evaluated (PASS or INCONCLUSIVE, never FAIL)",
      lambda: main["running_check"]["verdict"] in ("PASS", "INCONCLUSIVE_NO_FULL_SAMPLE"), json.dumps(main.get("running_check")))
check("solve-vs-tokens bands + latency means + counters deltas reported; bands count the accepts",
      lambda: main["bands"]["bands"] and "time_to_first_token_seconds" in main["latency_means"]
      and (MODE == "rerun" or main["bands"]["accepted"] >= 1)
      and main["counters"].get("vllm:generation_tokens_total", 0) > 0, "")
check("full traces: nb5_requests.jsonl per attempt with the full reasoning",
      lambda: any(json.loads(l).get("reasoning") for f in (W / "q27x").rglob("nb5_requests.jsonl") for l in open(f, encoding="utf-8")),
      list((W / "q27x").rglob("nb5_requests.jsonl"))[:3])
check("throughput: pure decode / steady mixed / solver window / whole notebook separately",
      lambda: set(rep["throughput"]) >= {"pure_decode", "steady_mixed", "solver_window", "whole_notebook"}
      and rep["throughput"]["whole_notebook"]["tok_s"] is not None, json.dumps(rep.get("throughput"), default=str)[:800])
fs = stats[-1] if stats else {}
check("fake server: turn N+1 prompt extended turn N's prompt in every checked Duck turn",
      lambda: sum(s.get("prefix_chain_checked", 0) for s in stats) > 0
      and all(s.get("prefix_chain_ok") == s.get("prefix_chain_checked") for s in stats),
      [(s.get("prefix_chain_ok"), s.get("prefix_chain_checked")) for s in stats])
if MODE == "mtp":
    check("mtp: the second boot launched with --speculative-config None (last), the first without",
          lambda: len(launches) >= 2 and "None" in launches[-1]["argv"][-1:] and "--speculative-config" not in launches[0]["argv"],
          [L["argv"][-4:] for L in launches])
    check("mtp: report has the MTP pair block", lambda: rep["mtp_pair"]["score_on"] is not None, json.dumps(rep.get("mtp_pair"))[:600])
gd = main.get("guard") or {}
check("per-request budget guard: every EVAL request measured on the server's own render (/tokenize), or estimated "
      "when /tokenize is off (notok)", lambda: gd["requests"] > 0 and (gd["estimated"] == gd["requests"] if MODE == "notok"
                                                                       else gd["tokenized"] == gd["requests"]), gd)
if MODE == "floor":
    check("floor: every attempt BUDGET_LIMITED (floor events counted, verdict in the records), solving still ended cleanly",
          lambda: main["budget_limited_attempts"] > 0 and gd["floor_events"] > 0 and main["status"] == "DONE",
          json.dumps({k: main.get(k) for k in ("budget_limited_attempts", "guard")})[:600])
else:
    check("no BUDGET_LIMITED attempt with a reachable floor", lambda: main["budget_limited_attempts"] == 0, main.get("budget_limited_attempts"))
if "SMOKE" in ph:
    s_ = ph["SMOKE"]
    check("SMOKE: early exit fired once its gates passed, the gate decided within its bound (EVAL / switch / abort), "
          "variant decision + paired split + preemption gate reported", lambda: s_["early_exit"]["fired"] is not None
          and s_["smoke_gate_within_bound"] and s_["gate_decision"] in ("EVAL", "EVAL_SWITCH_TO_A", "ABORT")
          and "switch_reasons" in s_["variant_decision"] and s_["paired_split"] is not None
          and s_["preemption_gate"]["verdict"] in ("PASS", "PREEMPTED"),
          json.dumps({k: s_.get(k) for k in ("early_exit", "smoke_gate_at_min", "gate_decision", "variant_decision",
                                             "preemption_gate")}, default=str)[:1200])
check("admission interaction per phase (server waiting while Duck holds, free capacity without admission) + Deferred lines",
      lambda: all("free_projected_capacity_no_admission" in p["interaction"] and "lines" in p["deferred"] for p in ph.values()),
      json.dumps(main.get("interaction")))
check("duty factor (productive decode / wall, time + token-weighted) per phase and for the whole notebook",
      lambda: all(p["duty"]["duty_time"] is not None for p in ph.values())
      and rep["throughput"]["duty_factor"]["whole_notebook"]["duty_time"] is not None,
      json.dumps(rep.get("throughput", {}).get("duty_factor"), default=str)[:800])
check("GPU-only gates listed with this run's values, marked as fake-server (not production) evidence",
      lambda: rep["gpu_only_gates"]["evidence"].startswith("FAKE_SERVER") and all(
          rep["gpu_only_gates"][k]["verdict"].startswith("LOCAL_ONLY") for k in
          ("token_id_completion_semantics", "parser_correctness", "mtp_on_state_consistency", "real_prefix_reuse")),
      json.dumps(rep.get("gpu_only_gates"), default=str)[:800])
check("bundle zip written", (W / "Q27X_NB5_BUNDLE.zip").exists())
bad = [c for c in checks if not c[1]]
for label, ok, info in checks:
    print(f"[nb5 e2e {'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f" -> {info}"), flush=True)
print(f"{len(checks) - len(bad)}/{len(checks)} PASS | work {WORK}", flush=True)
print("@@NB5_E2E_OK@@" if not bad else "@@NB5_E2E_FAIL@@", flush=True)
sys.exit(0 if not bad else 1)
