"""CPU harness for E18 (S1-60K on vLLM 0.32, 20 streams). Runs the REAL notebook cells in order: CELL 0, CELL 1, the task
order, the Poetiq core, the NEW engine cell (vLLM 0.32 launch ladder + shim) against fake_vllm_e18_server.py, the S1 solver,
the S1 report and CELL 8b. CELL 2 (pip) is replaced by the real Qwen3.8 tokenizer; the sandbox verifier is the S1 harness's
MARK verifier. Behaviours per task as harness_s1.py (Experiment 10).
usage: python harness_e18_nb.py <notebook> <scenario>
scenarios: mixed | kvgate | hardstop | rerun | lmofail (rung 1 rejected -> the VMS flags) | allfail (no engine -> identity)
| death (rev 2: the server dies after 4 solve requests -> one relaunch inside the shim, the run completes)"""
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

NB, SCEN = sys.argv[1], sys.argv[2]
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("E18_DATA", r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 "
                                     r"(planned)/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data"))
TOKDIR = Path(os.environ.get("E18_TOKDIR", r"D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/tok"))
sys.stdout.reconfigure(encoding="utf-8")
if os.name == "nt" and not hasattr(os, "killpg"):
    import signal as _sig
    if not hasattr(_sig, "SIGKILL"):
        _sig.SIGKILL = 9
WORK = Path(tempfile.mkdtemp(prefix=f"e18_{SCEN}_"))
ARC = WORK / "arc"; ARC.mkdir()
ch = json.load(open(DATA / "arc-agi_evaluation_challenges.json", encoding="utf-8"))
sol = json.load(open(DATA / "arc-agi_evaluation_solutions.json", encoding="utf-8"))
shutil.copy(DATA / "arc-agi_evaluation_challenges.json", ARC / "arc-agi_evaluation_challenges.json")
shutil.copy(DATA / "arc-agi_evaluation_solutions.json", ARC / "arc-agi_evaluation_solutions.json")
multi = sorted(t for t in ch if len(ch[t]["test"]) >= 2)[:3]
single = sorted(t for t in ch if len(ch[t]["test"]) == 1)[:5]
PICK = multi + single
test_file = {t: {"train": ch[t]["train"], "test": [{"input": q["input"]} for q in ch[t]["test"]]} for t in PICK}
json.dump(test_file, open(ARC / "arc-agi_test_challenges.json", "w"))
BEH = {multi[0]: "partial_then_fail", multi[1]: "forced_then_pass", multi[2]: "fail_distinct",
       single[0]: "pass", single[1]: "illustrative_first", single[2]: "fail_same", single[3]: "pass", single[4]: "pass"}
if SCEN == "hardstop":
    BEH.update({single[0]: "never", single[3]: "never"})
with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); PORT = s.getsockname()[1]
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(TOKDIR), Q27X_SKIP_INSTALL="1",
                  E10_POETIQ_MAX_ITERATIONS="3", PYTHONIOENCODING="utf-8", E18_PORT=str(PORT),
                  E18_FAKE_SERVER=str(HERE / "fake_vllm_e18_server.py"), Q27X_GPU_WAIT_MAX="1")
os.environ.pop("E18_NO_FALLBACK", None)   # allfail: the fallback runs (vLLM 0.19 is not installed here: it must fail cleanly)
if SCEN == "rerun":
    os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "true"

nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert len(code) == 9 and "CELL 5 — engine: vLLM 0.32.0" in code[5] and "CELL 8b" in code[8], len(code)

# ---- profile check on a fresh namespace: the commit default is E18SMOKE (20 tasks, 45 min) with 20 seqs
_P = {"__name__": "__main__"}
_saved = os.environ.pop("Q27X_PROFILE", None)
_rr = os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
exec(compile(code[0], "<cell0>", "exec"), _P); exec(compile(code[1], "<cell1>", "exec"), _P)
if _rr is not None:
    os.environ["KAGGLE_IS_COMPETITION_RERUN"] = _rr
PROFILE_OK = (_P["PROFILE"] == "E18SMOKE" and len(_P["P"]["task_list"]) == 20 and _P["TOTAL_BUDGET_SEC"] == 2700
              and _P["MAX_NUM_SEQS"] == 20 and round(_P["HARD_STOP"] - _P["NOTEBOOK_T0"]) == 2520
              and round(_P["DRAIN_DEADLINE"] - _P["NOTEBOOK_T0"]) == 2160 and round(_P["ADMIT_DEADLINE"] - _P["NOTEBOOK_T0"]) == 2100)
PROFILE_INFO = (_P["PROFILE"], _P["MAX_NUM_SEQS"], _P["TOTAL_BUDGET_SEC"])

if SCEN != "rerun":
    os.environ["Q27X_PROFILE"] = "EVAL12"
G = {"__name__": "__main__"}
exec(compile(code[0], "<cell0>", "exec"), G)
exec(compile(code[1], "<cell1>", "exec"), G)
if SCEN != "rerun":
    G["P"] = dict(G["P"], task_list=PICK)

# ---- CELL 2 replacement: the real Qwen3.8 tokenizer and token constants
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(TOKDIR))
G["tokenizer"] = tok
G["THINK_CLOSE_ID"] = tok.encode("</think>", add_special_tokens=False)[0]
G["STOP_IDS"] = tok.encode(G["STOP_THINKING_TEXT"], add_special_tokens=False)
G["EOS_IDS"] = {248046, 248044}
exec(compile(code[3], "<cell3>", "exec"), G)
exec(compile(code[4], "<cell4>", "exec"), G)
G["E10_SANDBOX_SELFTEST"] = {"ok": True, "mode": "harness"}


def fake_verify(code_text, task, timeout_s=5.0):
    tid = next(t for t, v in G["CHALLENGES"].items() if v is task)
    truth = sol.get(tid) or [[[0]]] * len(task["test"])
    n_tr = len(task["train"])

    def res(passed, score, grids):
        tr = [{"success": passed, "soft_score": score, "feedback": "f", "error": None} for _ in range(n_tr)]
        tests = [{"grid": g, "error": None if g is not None else "boom"} for g in grids]
        usable = all(G["poetiq_valid_grid"](g) for g in grids)
        return {"train_results": tr, "test_results": tests, "passed": passed, "test_usable": usable,
                "verified_usable": passed and usable, "score": score, "feedback": "fb"}
    if "MARK:PASS_ALL" in code_text:
        return res(True, 1.0, [truth[i] for i in range(len(task["test"]))])
    if "MARK:PASS_PARTIAL" in code_text:
        return res(True, 1.0, [truth[i] if i < len(task["test"]) - 1 else None for i in range(len(task["test"]))])
    if "MARK:FAIL_ITER" in code_text:
        k = int(code_text.split("MARK:FAIL_ITER")[1].split()[0])
        return res(False, 0.5, [[[k % 10]] for _ in task["test"]])
    return res(False, 0.0, [[[9]] for _ in task["test"]])


G["poetiq_verify_code"] = fake_verify
FLOG = WORK / "fake_log.json"
CFG = {"tokenizer": str(TOKDIR / "tokenizer.json"), "challenges": str(ARC / "arc-agi_evaluation_challenges.json"),
       "tasks": PICK, "behaviour": BEH, "close_id": G["THINK_CLOSE_ID"], "scenario": SCEN, "log": str(FLOG),
       "marker": str(WORK / "fake_died")}
json.dump(CFG, open(WORK / "fake_cfg.json", "w"))
os.environ["E18_FAKE_CFG"] = str(WORK / "fake_cfg.json")
t0 = time.time()
exec(compile(code[5], "<cell5 engine>", "exec"), G)                 # the REAL engine cell
if SCEN == "kvgate":
    G["E10_KV_GATE_MIN_ACTIVE"] = 4
if SCEN == "hardstop":
    G["DRAIN_DEADLINE"] = time.perf_counter() + 4.0
    G["ADMIT_DEADLINE"] = G["DRAIN_DEADLINE"] - 0.5
    G["HARD_STOP"] = G["DRAIN_DEADLINE"] + 20.0
exec(compile(code[6], "<cell6 solver>", "exec"), G)
exec(compile(code[7], "<cell7 report>", "exec"), G)
exec(compile(code[8], "<cell8b>", "exec"), G)
WALL = time.time() - t0

fails = []


def check(cond, msg, info=""):
    print(("PASS " if cond else "FAIL ") + msg + ("" if cond else f"  -> {str(info)[:600]}"))
    if not cond:
        fails.append(msg)


EI, S, F = G["ENGINE_INFO"], G["E10_SCHED"], G["E10_FINAL"]
fl = json.load(open(FLOG, encoding="utf-8")) if FLOG.exists() else {"requests": [], "disconnects": 0, "max_running": 0}
reqs = [r for r in fl["requests"] if r.get("phase") in (1, 2)]
sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
check(PROFILE_OK, "CELL 1: commit default E18SMOKE (20 tasks, 45 min: admit 35, drain 36, hard stop 42 min), 20 seqs", PROFILE_INFO)
check(set(sub) == set(test_file) and all(len(sub[t]) == len(test_file[t]["test"]) for t in sub), "submission schema")
if SCEN == "allfail":
    check(not G["ENGINE_READY"] and G["CONCURRENCY"] == 0, "no engine -> ENGINE_READY False, concurrency 0", EI)
    check([a.get("name") for a in EI.get("e18_attempts", [])] == ["s1", "vms", "vms_triton"]
          and all(not a["ok"] for a in EI["e18_attempts"]), "the 3 vLLM 0.32 rungs tried and failed", EI.get("e18_attempts"))
    check("fallback vLLM 0.19 failed" in str(EI.get("fatal")), "the S1 vLLM 0.19 fallback ran and failed cleanly (no crash)", EI.get("fatal"))
    check(S.get("skipped") == "engine_unavailable", "solver skipped", S.get("skipped"))
    check(all(a["attempt_1"] == q["input"] for t in test_file for a, q in zip(sub[t], test_file[t]["test"])), "identity submission")
else:
    a = EI["attempts"][-1]
    want_rung = "vms" if SCEN == "lmofail" else "s1"
    check(G["ENGINE_READY"] and a["name"] == want_rung and a["ok"], f"engine ready on rung {want_rung}", EI["attempts"])
    if SCEN == "lmofail":
        check(EI["attempts"][0]["name"] == "s1" and not EI["attempts"][0]["ok"], "rung s1 failed first, then vms", EI["attempts"])
    f = a["flags"]
    def val(k):
        return f[f.index(k) + 1] if k in f else None
    check(val("--max-num-seqs") == "20" and val("--tensor-parallel-size") == "4" and val("--kv-cache-dtype") == "fp8_e4m3"
          and val("--max-model-len") == "106496" and val("--max-num-batched-tokens") == "4096"
          and val("--gpu-memory-utilization") == "0.9" and "--disable-custom-all-reduce" in f and "--enable-prefix-caching" in f
          and json.loads(val("--speculative-config")) == {"method": "mtp", "num_speculative_tokens": 3}
          and val("--attention-backend") == "FLASHINFER" and val("--seed") == str(G["SEED"]),
          "flags = S1's engine config with 20 seqs (fp8 KV, MTP 3, prefix caching, FlashInfer, 4096, 0.90, CAR off)", f)
    if SCEN != "lmofail":
        check("--language-model-only" in f and val("--load-format") == "safetensors" and "--trust-remote-code" not in f,
              "rung s1: language-model-only, safetensors, trust_remote_code off (S1's LLM() kwargs)", f)
    else:
        check("--language-model-only" not in f and "--trust-remote-code" in f, "rung vms: the exact VMS fp8_mtp3_s20 flag set", f)
    check(EI["kv_tokens"] == 1006517 and G["CONCURRENCY"] == 20 and EI["vllm"] == "0.32.0",
          "KV 1,006,517 tokens parsed; concurrency = min(20, KV // 43,000) = 20; version 0.32.0", (EI["kv_tokens"], G["CONCURRENCY"], EI.get("vllm")))
    check((a.get("smoke") or {}).get("tokens", 0) > 0, "engine smoke decoded tokens through the shim", a.get("smoke"))
    check(G["E10_SCHED_ERROR"] is None, "solver ended without an error", str(G["E10_SCHED_ERROR"])[-400:])
    sm = [r for r in reqs]
    check(sm and all(r["stream"] and r["return_token_ids"] and r["skip_special_tokens"] is False and r["model"] == "q27"
                     for r in sm), "every request: streaming, return_token_ids, skip_special_tokens False, model q27")
    check(all(r["temperature"] == G["POETIQ_SOLVER_TEMPERATURE"] and r["top_p"] == 0.95 and r["top_k"] == 20
              and isinstance(r["seed"], int) for r in sm), "sampling = S1's (temperature 1.0, top_p 0.95, top_k 20, per-request seed)",
          [(r["temperature"], r["top_p"], r["top_k"], r["seed"]) for r in sm[:3]])
    p2 = [r for r in sm if r["phase"] == 2]
    p2k = [r for r in p2 if r.get("prefix_ok") is not None]
    check(p2k and all(r["prefix_ok"] for r in p2k) and (SCEN == "death" or len(p2k) == len(p2)),
          f"phase 2 = the phase-1 prompt + its tokens ({len(p2k)} of {len(p2)} phase-2 requests checkable)")
    p1caps = [r["max_tokens"] for r in sm if r["phase"] == 1]
    check(p1caps and all(c == G["phase1_cap"](r["len"]) for c, r in zip(p1caps, [r for r in sm if r["phase"] == 1])),
          "phase-1 max_tokens = S1's phase1_cap(prompt length)")
    st = G["engine"].stats
    check(st["added"] == st["finished"] and (SCEN == "death" or not st["errors"]) and st["missing_token_ids"] == 0,
          "shim: every stream ended, no stream error (death: only the dead server's), integer token ids everywhere", st)
    check((WORK / "q27x" / "e18_engine_stats.json").exists() and not G["E18_SERVER"].alive(),
          "CELL 8b: engine stats written, server stopped")
    rep = json.load(open(WORK / "q27x" / "poetiq_run_report.json", encoding="utf-8"))
    check(rep["engine"]["engine"] == "vllm-0.32.0-server", "report carries the E18 engine info", rep["engine"].get("engine"))
    if SCEN == "mixed":
        m0 = multi[0]
        check(F[m0][0]["attempt_1"] == sol[m0][0], "F3: train-passing program with a crashing test input keeps query 0")
        check(G["E10_STATES"][(single[1], 0)]["status"] == "train_verified", "F1: the last transform block was executed")
        check(all(p["attempt_1"] != p["attempt_2"] for t in F for p in F[t]), "A2: attempt_2 never repeats attempt_1")
        check(F[multi[1]][0]["attempt_1"] == sol[multi[1]][0], "forced iteration 1 (length) then natural pass in iteration 2")
        check(G["E10_STATES"][(multi[2], 0)]["iteration"] == 3, "a failing task gets all 3 iterations")
        hits = sum(F[t][i]["attempt_1"] == sol[t][i] or F[t][i]["attempt_2"] == sol[t][i] for t in F for i in range(len(F[t])))
        check(rep.get("eval_query_hits") == hits and hits >= 5, f"eval scored ({hits} hits)", rep.get("eval_query_hits"))
    if SCEN == "kvgate":
        check(S["max_active"] == 4 and S["kv_gate_blocks"] > 0, "KV gate (server kv_cache_usage 0.97) held running at 4",
              (S["max_active"], S["kv_gate_blocks"]))
    if SCEN == "hardstop":
        nev = [single[0], single[3]]
        check(fl["disconnects"] >= 2, f"drain: the never-closing phase-1 streams were closed (server saw {fl['disconnects']} disconnects)")
        check(all(G["E10_STATES"][(t, 0)]["status"] == "train_verified" for t in nev),
              "drained units answered in phase 2 (forced close) and verified")
        check(st["by_reason"].get("abort", 0) >= 2, "shim recorded the aborted streams", st["by_reason"])
    if SCEN == "rerun":
        check(sub[single[0]][0]["attempt_1"] == sol[single[0]][0], "rerun: solved task written into submission.json")
    if SCEN == "death":
        rl = st.get("relaunches") or []
        check(len(rl) == 1 and str(rl[0].get("ready", "")).startswith("ready") and (rl[0].get("smoke") or {}).get("tokens"),
              "server death mid-run: relaunched once (ready + smoke) inside the shim", rl)
        check(not any(s_["status"] == "running" for s_ in G["E10_STATES"].values()), "every unit closed after the relaunch",
              {k: v["status"] for k, v in G["E10_STATES"].items()})
        check(G["E18_SERVER"].name.endswith("_relaunch"), "E18_SERVER is the relaunched server", G["E18_SERVER"].name)
print(f"\n[{SCEN}] {'ALL CHECKS PASS' if not fails else str(len(fails)) + ' failed'} in {WALL:.0f} s")
sys.exit(1 if fails else 0)
