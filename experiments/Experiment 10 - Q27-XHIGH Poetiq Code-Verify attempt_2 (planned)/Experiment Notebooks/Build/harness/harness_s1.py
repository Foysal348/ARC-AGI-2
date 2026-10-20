"""CPU harness for E10 S1 (Poetiq R5 + fixes). Runs the REAL notebook cells 0,1,3(tasks),4(poetiq core),6(scheduler)
with: the real Qwen3.8 tokenizer + chat template, real eval JSON, a fake vLLM engine (add_request/step/abort_request,
DELTA token ids, finish 'stop'|'length') and a fake sandbox verifier driven by a marker inside the generated code.
Cells 2 (pip install) and 5 (engine load) are replaced by stubs. usage: python harness_s1.py <notebook> <scenario>
scenarios: mixed | kvgate | hardstop | rerun | slowverify"""
import json, os, sys, time, types, tempfile, shutil, random
from pathlib import Path

NB, SCEN = sys.argv[1], sys.argv[2]
HERE = Path(__file__).resolve().parent
DATA = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
            r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
TOKDIR = HERE.parent / "e11_03_harness" / "tok"
WORK = Path(tempfile.mkdtemp(prefix=f"s1_{SCEN}_"))
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
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(TOKDIR), Q27X_SKIP_INSTALL="1",
                  E10_POETIQ_MAX_ITERATIONS="3", PYTHONIOENCODING="utf-8")
if SCEN == "rerun":
    os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "true"
else:
    os.environ["Q27X_PROFILE"] = "EVAL12"

nb = json.load(open(NB, encoding="utf-8"))
code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]   # 0 mode,1 config,2 install,3 tasks,4 core,5 engine,6 sched,7 report
G = {"__name__": "__main__"}
exec(compile(code[0], "<cell0>", "exec"), G)
exec(compile(code[1], "<cell1>", "exec"), G)
if SCEN != "rerun":
    G["P"] = dict(G["P"], task_list=PICK)        # small fixed eval subset: 3 multi-test + 5 single-test tasks

# ---- cell 2 replacement: tokenizer and token constants (real Qwen3.8 tokenizer)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(TOKDIR))
G["tokenizer"] = tok
G["THINK_CLOSE_ID"] = tok.encode("</think>", add_special_tokens=False)[0]
G["STOP_IDS"] = tok.encode(G["STOP_THINKING_TEXT"], add_special_tokens=False)
G["EOS_IDS"] = {248046, 248044}
exec(compile(code[3], "<cell3>", "exec"), G)
exec(compile(code[4], "<cell4>", "exec"), G)
G["E10_SANDBOX_SELFTEST"] = {"ok": True, "mode": "harness"}

# ---- fake sandbox verifier: behaviour from a marker in the code
VERIFY_SLEEP = {"hardstop": 2.5, "slowverify": 6.0}.get(SCEN, 0.0)
def fake_verify(code_text, task, timeout_s=5.0):
    time.sleep(VERIFY_SLEEP)
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
    if "MARK:PASS_PARTIAL" in code_text:          # train passes, the LAST test input crashes
        return res(True, 1.0, [truth[i] if i < len(task["test"]) - 1 else None for i in range(len(task["test"]))])
    if "MARK:FAIL_ITER" in code_text:            # wrong but valid grids that differ per iteration
        k = int(code_text.split("MARK:FAIL_ITER")[1].split()[0])
        return res(False, 0.5, [[[k % 10]] for _ in task["test"]])
    return res(False, 0.0, [[[9]] for _ in task["test"]])
G["poetiq_verify_code"] = fake_verify

# ---- task behaviours (by position in PICK)
BEH = {multi[0]: "partial_then_fail", multi[1]: "forced_then_pass", multi[2]: "fail_distinct",
       single[0]: "pass", single[1]: "illustrative_first", single[2]: "fail_same", single[3]: "pass", single[4]: "exc_once"}
def answer_code(tid, it):
    b = BEH.get(tid, "fail_same")
    if b == "pass" or b == "exc_once": return "```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```"
    if b == "partial_then_fail":
        return "```python\ndef transform(grid):\n    return grid  # MARK:PASS_PARTIAL\n```" if it == 0 else \
               f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```"
    if b == "forced_then_pass":
        return (f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```" if it == 0 else
                "```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```")
    if b == "fail_distinct": return f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```"
    if b == "illustrative_first":
        return ("For example:\n```python\nx = helper(grid)  # MARK:ZERO no transform here\n```\n"
                "First idea:\n```python\ndef transform(grid):\n    return grid  # MARK:ZERO\n```\n"
                "Final:\n```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```")
    return "```python\ndef transform(grid):\n    return grid  # MARK:ZERO\n```"

FILL = tok.encode(" thinking", add_special_tokens=False)
CLOSE = G["THINK_CLOSE_ID"]

class Out:
    def __init__(self, rid, ids, finished, reason, broken=False):
        self.request_id = rid; self.finished = finished; self._broken = broken
        self._c = types.SimpleNamespace(token_ids=ids, finish_reason=reason)
    @property
    def outputs(self):
        if self._broken: raise ValueError("simulated corrupt engine output")
        return [self._c]

class FakeEngine:
    def __init__(self):
        self.active = {}; self.admit_log = []; self.aborted = []; self.injected = set()
    def add_request(self, rid, prompt, params):
        tid, rest = rid.split("~e", 1); it = int(rest.split("~i")[1].split("|")[0]); phase = int(rid[-1])
        if phase == 1: self.admit_log.append((tid, it))
        b = BEH.get(tid, "fail_same")
        forced = (b == "forced_then_pass" and it == 0)
        cap = params.max_tokens
        if phase == 1 and forced:
            stream, reason = (FILL * (cap // len(FILL) + 1))[:cap], "length"
        elif phase == 1:
            stream = FILL * 30 + [CLOSE] + tok.encode("\n\n" + answer_code(tid, it), add_special_tokens=False); reason = "stop"
        else:
            stream = tok.encode(answer_code(tid, it), add_special_tokens=False); reason = "stop"
        self.active[rid] = {"stream": stream, "pos": 0, "reason": reason}
    def abort_request(self, ids):
        for i in ids: self.active.pop(i, None); self.aborted.append(i)
    def step(self):
        outs = []
        if SCEN == "mixed" and "badid" not in self.injected:
            self.injected.add("badid"); outs.append(Out("not-a-poetiq-id", [], True, "stop"))
        for rid, r in list(self.active.items()):
            chunk = r["stream"][r["pos"]:r["pos"] + 4000]; r["pos"] += len(chunk)
            done = r["pos"] >= len(r["stream"])
            tid = rid.split("~e")[0]
            broken = SCEN == "mixed" and BEH.get(tid) == "exc_once" and done and "exc" not in self.injected
            if broken: self.injected.add("exc")
            outs.append(Out(rid, chunk, done, r["reason"] if done else None, broken))
            if done: self.active.pop(rid, None)
        time.sleep(0.002)
        return outs

class FakeLLM:
    def get_metrics(self):
        n = len(G["E10_ACTIVE"]) if "E10_ACTIVE" in G else 0
        kv = 0.97 if (SCEN == "kvgate" and n >= 4) else 0.30
        return [types.SimpleNamespace(name="vllm:kv_cache_usage_perc", value=kv)]

engine = FakeEngine()
G.update(engine=engine, llm=FakeLLM(), ENGINE_READY=True, CONCURRENCY=16,
         SamplingParams=lambda **kw: types.SimpleNamespace(**kw), RequestOutputKind=types.SimpleNamespace(DELTA=1),
         metrics_snapshot=lambda: [])
if SCEN == "kvgate":
    G["E10_KV_GATE_MIN_ACTIVE"] = 4          # small fixture: gate binds above 4 running
    os.environ["E10_POETIQ_MAX_ITERATIONS"] = "3"
if SCEN == "slowverify":
    G["POETIQ_SANDBOX_TIMEOUT"] = 0.5     # harvest window 1 s; 8 verifications x 4 s on 4 workers are pending
if SCEN in ("hardstop", "slowverify"):
    G["HARD_STOP"] = time.perf_counter() + 1.5          # hard stop while the first verifications are still running
    G["DRAIN_DEADLINE"] = G["HARD_STOP"] - 0.7; G["ADMIT_DEADLINE"] = G["DRAIN_DEADLINE"] - 0.3
    G["TOTAL_BUDGET_SEC"] = (G["HARD_STOP"] - G["NOTEBOOK_T0"]) + 360
t0 = time.time()
exec(compile(code[6], "<cell6>", "exec"), G)
WALL = time.time() - t0
print(f"\nscheduler finished in {WALL:.1f}s")

# ---------------------------------------------------------------- checks
S = G["E10_SCHED"]; F = G["E10_FINAL"]; fails = []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)
check(G["E10_SCHED_ERROR"] is None, f"solver ended without an error ({str(G['E10_SCHED_ERROR'])[-200:]})")
first = [t for t, it in engine.admit_log if it == 0]
check(first == G.get("E10_ORDER", [])[:len(first)], f"iteration-1 admissions follow ORDER2: {first[:8]}")
o2 = [int(len(G["CHALLENGES"][t]["train"]) >= 3) + int(len(G["CHALLENGES"][t]["test"]) == 1) for t in first]
check(o2 == sorted(o2, reverse=True), f"ORDER2 scores non-increasing: {o2}")
if SCEN == "hardstop":
    print("hard stop: aborted requests", S.get("aborted"), "| verifications committed", S.get("sandbox_verifications"))
if SCEN == "mixed":
    check(len(S["unit_errors"]) >= 2, f"bad id and corrupt output isolated: {S['unit_errors']}")
    m0 = multi[0]
    check(F[m0][0]["attempt_1"] == sol[m0][0], "F3: train-passing program with a crashing test input keeps query 0")
    check(G["E10_STATES"][(single[1], 0)]["status"] == "train_verified", "F1: the last transform block was executed")
    check(all(p["attempt_1"] != p["attempt_2"] for t in F for p in F[t]), "A2: attempt_2 never repeats attempt_1")
    check(F[multi[1]][0]["attempt_1"] == sol[multi[1]][0], "forced iteration 1 then natural pass in iteration 2")
    check(G["E10_STATES"][(multi[2], 0)]["iteration"] == 3, "failing task gets all 3 iterations")
    check(G["E10_STATES"][(single[4], 0)]["status"] == "unit_error", "corrupt output closed only that unit")
    check(G["E10_STATES"][(single[0], 0)]["status"] == "train_verified", "other tasks unaffected by the unit error")
    it2 = [i for i, (t, it) in enumerate(engine.admit_log) if it == 1]
    it1_last = max(i for i, (t, it) in enumerate(engine.admit_log) if it == 0)
    check(not it2 or min(it2) > it1_last or len(first) < len(PICK), "every task got iteration 1 before any iteration 2")
RECS = [json.loads(l) for l in open(G["E10_RESULT_PATH"], encoding="utf-8")] if Path(G["E10_RESULT_PATH"]).exists() else []
if any("test_grids" in r for r in RECS):          # D66 per-iteration evidence (read back from the JSONL)
    ver = [r for r in RECS if r.get("score") is not None]
    check(bool(ver) and all(len(r["test_grids"]) == len(G["CHALLENGES"][r["task_id"]]["test"]) for r in ver),
          f"D66 log: one test grid per input in each of {len(ver)} verified iterations")
    if SCEN == "rerun":                        # hidden test: no truth is known, so nothing may be marked correct
        check(all(r.get("test_correct") is None for r in RECS), "D66 log: rerun has no test truth (test_correct None)")
    else:
        check(all(isinstance(r["test_correct"], list) and len(r["test_correct"]) == len(r["test_grids"]) for r in ver),
              "D66 log: test_correct per input")
        check(any(any(r["test_correct"]) for r in ver) and any(not all(r["test_correct"]) for r in ver),
              "D66 log: both correct and wrong iterations recorded")
    check(all(len(r["train_soft_scores"]) == len(G["CHALLENGES"][r["task_id"]]["train"]) for r in ver), "D66 log: train soft scores")
    check(all("log_error" not in r for r in RECS) and all("t_done_sec" in r for r in RECS), "D66 log: no logging errors")
    check(all("t_admit" in r and "p1_finish" in r for r in RECS), "D66 log: admit time and phase-1 finish")
    if SCEN != "rerun":
        _acp = Path(G["OUT"]) / "poetiq_attempt_curve.jsonl"
        AC = [json.loads(l) for l in open(_acp, encoding="utf-8")] if _acp.exists() else []
        _last = {}
        for _a in AC: _last[_a["task_id"]] = _a
        _final_hit = lambda t: [(F[t][i]["attempt_1"] == sol[t][i] or F[t][i]["attempt_2"] == sol[t][i]) for i in range(len(F[t]))]
        check(bool(AC) and all(_last[t]["hit"] == _final_hit(t) for t in _last),
              f"D66 curve: {len(AC)} entries; the last entry per task equals the final attempts")
if SCEN == "kvgate":
    check(S["max_active"] == 4, f"KV gate held running requests at the floor of 4 (max_active {S['max_active']})")
    check(S["kv_gate_blocks"] > 0, f"gate blocks counted: {S['kv_gate_blocks']}, last kv {S['kv_gate_last']}")
if SCEN == "hardstop":
    check(S.get("remaining_verifications") == 0, f"verifications pending at the hard stop were harvested ({S.get('remaining_verifications')})")
    check(S["sandbox_verifications"] >= 1, f"verifications committed: {S['sandbox_verifications']}")
if SCEN == "slowverify":
    check(WALL < 10.0, f"F2b: queued verifications cancelled; solver ended {WALL:.1f}s after start "
                       f"(hard stop 1.5 s; F2b ends near 6-7 s; waiting for all 8 x 6 s on 4 workers would end near 12 s)")
    check(S.get("remaining_verifications", 0) >= 1, f"unfinished verifications recorded: {S.get('remaining_verifications')}")
if SCEN == "rerun":
    sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
    check(set(sub) == set(test_file), "submission has every hidden task")
    check(all(len(sub[t]) == len(test_file[t]["test"]) for t in sub), "one attempt pair per test input")
    check(sub[single[0]][0]["attempt_1"] == sol[single[0]][0], "solved task written into submission.json")
    check(all(isinstance(p["attempt_1"], list) and isinstance(p["attempt_2"], list) for t in sub for p in sub[t]), "schema")
print(f"\n{len(fails)} failed checks" if fails else "\nALL CHECKS PASS")
sys.exit(1 if fails else 0)
