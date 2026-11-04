"""CPU harness for E11-05 (S49 on SGLang). Runs the REAL notebook cells 0-4 (config, install branch with
Q27X_SKIP_INSTALL, tokenizer + E11 tokenizer check, tasks, Poetiq core), the REAL SGLang engine cell (launch, caps,
smoke, shim) against a fake streaming SGLang server over HTTP, the REAL fallback cell (S49 engine code) with a fake
vllm package, the REAL scheduler and report cells. Only the sandbox verifier is faked (marker in the code).
usage: python harness_e11_05.py <notebook> <scenario>
scenarios: mixed | kvgate | hardstop | drain | rerun | slowverify | lazyfail | launchfail | abortbroken | serverdeath | lowcap | nocaplog"""
import json, os, shutil, socket, sys, time, urllib.request
from pathlib import Path

NB, SCEN = sys.argv[1], sys.argv[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "fake_vllm"))
import beh_e11_05 as B

ROOT = HERE / "work"; ROOT.mkdir(exist_ok=True)
WORK = ROOT / f"{SCEN}_{time.strftime('%H%M%S')}"
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir()
ch, sol, multi, single = B.load()
PICK = multi + single
BEH = B.beh_map(multi, single, SCEN)

# ---- fixtures: FP8-like tokenizer dir (S49 tokenizer files + fp8 config), INT4 dir (real RedHat tokenizer files),
# DFlash2 config, a wheelhouse marker, ARC data
FP8 = ROOT / "fp8_tok"
if not FP8.exists():
    FP8.mkdir()
    src = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
               r"E10_ONE_THINKING_TWO_ATTEMPTS_FOR_ASTRA/tokenizer")
    for f in src.iterdir():
        shutil.copy(f, FP8 / f.name)
    json.dump({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
               "quantization_config": {"quant_method": "fp8"}}, open(FP8 / "config.json", "w"))
DFL = ROOT / "dflash"; DFL.mkdir(exist_ok=True)
json.dump({"architectures": ["DFlash2DraftModel"]}, open(DFL / "config.json", "w"))
WH = ROOT / "wheelhouse"; WH.mkdir(exist_ok=True)
json.dump({"sglang": "0.5.21"}, open(WH / "BUILD_INFO.json", "w"))
(WH / "requirements.lock").write_text("", encoding="utf-8"); (WH / "SHA256SUMS").write_text("", encoding="utf-8")
ARC = WORK / "arc"; ARC.mkdir()
shutil.copy(B.DATA / "arc-agi_evaluation_challenges.json", ARC / "arc-agi_evaluation_challenges.json")
shutil.copy(B.DATA / "arc-agi_evaluation_solutions.json", ARC / "arc-agi_evaluation_solutions.json")
test_file = {t: {"train": ch[t]["train"], "test": [{"input": q["input"]} for q in ch[t]["test"]]} for t in PICK}
json.dump(test_file, open(ARC / "arc-agi_test_challenges.json", "w"))

with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); PORT = s.getsockname()[1]
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(FP8), Q27X_SKIP_INSTALL="1",
                  E10_POETIQ_MAX_ITERATIONS="3", PYTHONIOENCODING="utf-8", E11_HARNESS="1", E11_PORT=str(PORT),
                  E11_FAKE_SERVER=str(HERE / "fake_sgl_stream_server.py"), E11_INT4=str(B.TOKDIR_INT4), E11_DFLASH=str(DFL),
                  E11_SGL_WHEELHOUSE=str(WH), FAKE_SCEN=SCEN)
if SCEN == "rerun":
    os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "true"
else:
    os.environ["Q27X_PROFILE"] = "EVAL12"

nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
# 0 mode, 1 config, 2 install+tokenizer, 3 tasks, 4 poetiq core, 5 SGLang engine, 6 vLLM fallback, 7 scheduler, 8 report
assert len(code) == 9 and "CELL 5 — engine: SGLang" in code[5] and "CELL 5b" in code[6]
G = {"__name__": "__main__"}
for i in (0, 1):
    exec(compile(code[i], f"<cell{i}>", "exec"), G)
if SCEN != "rerun":
    G["P"] = dict(G["P"], task_list=PICK)
for i in (2, 3, 4):
    exec(compile(code[i], f"<cell{i}>", "exec"), G)
G["E10_SANDBOX_SELFTEST"] = {"ok": True, "mode": "harness"}

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
    if "MARK:PASS_PARTIAL" in code_text:
        return res(True, 1.0, [truth[i] if i < len(task["test"]) - 1 else None for i in range(len(task["test"]))])
    if "MARK:FAIL_ITER" in code_text:
        k = int(code_text.split("MARK:FAIL_ITER")[1].split()[0])
        return res(False, 0.5, [[[k % 10]] for _ in task["test"]])
    return res(False, 0.0, [[[9]] for _ in task["test"]])


G["poetiq_verify_code"] = fake_verify
t_eng = time.time()
exec(compile(code[5], "<cell5 sglang>", "exec"), G)
exec(compile(code[6], "<cell5b fallback>", "exec"), G)
ENG_SEC = time.time() - t_eng
ADMIT, ABORTS, EVENTS = [], [], []
if G["ENGINE_READY"]:
    _add, _abort = G["engine"].add_request, G["engine"].abort_request
    G["engine"].add_request = lambda rid, p, params: (ADMIT.append(rid), EVENTS.append(("add", rid)), _add(rid, p, params))[2]
    G["engine"].abort_request = lambda ids: (ABORTS.extend(ids), EVENTS.extend(("abort", i) for i in ids), _abort(ids))[2]
if SCEN == "kvgate":
    G["E10_KV_GATE_MIN_ACTIVE"] = 4
if SCEN == "slowverify":
    G["POETIQ_SANDBOX_TIMEOUT"] = 0.5
if SCEN in ("hardstop", "slowverify"):
    G["HARD_STOP"] = time.perf_counter() + 1.5
    G["DRAIN_DEADLINE"] = G["HARD_STOP"] - 0.7; G["ADMIT_DEADLINE"] = G["DRAIN_DEADLINE"] - 0.3
    G["TOTAL_BUDGET_SEC"] = (G["HARD_STOP"] - G["NOTEBOOK_T0"]) + 360
if SCEN == "drain":
    G["DRAIN_DEADLINE"] = time.perf_counter() + 3.0
    G["ADMIT_DEADLINE"] = G["DRAIN_DEADLINE"] - 0.5; G["HARD_STOP"] = G["DRAIN_DEADLINE"] + 8.0
    G["TOTAL_BUDGET_SEC"] = (G["HARD_STOP"] - G["NOTEBOOK_T0"]) + 360
t0 = time.time()
exec(compile(code[7], "<cell7 scheduler>", "exec"), G)
WALL = time.time() - t0
FAKE_LOG = {}
if G.get("E11_SERVER") is not None and G["E11_SERVER"].alive():
    try:
        FAKE_LOG = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/fake_log", timeout=30).read())
    except Exception as e:
        print("fake log read failed", e)
exec(compile(code[8], "<cell8 report>", "exec"), G)
print(f"\nengine cells {ENG_SEC:.1f}s | scheduler {WALL:.1f}s")

# ---------------------------------------------------------------- checks
S = G["E10_SCHED"]; F = G["E10_FINAL"]; E = G["ENGINE_INFO"]; SG = G["E11_SGL"]; fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


STOP = G["STOP_IDS"]
check(G["E11_TOKENIZER_CHECK"].get("ok") is True, f"E11 tokenizer check: {G['E11_TOKENIZER_CHECK']}")
fallback = SCEN in ("launchfail", "abortbroken", "nocaplog")
if fallback:
    check("fallback" in str(E.get("backend")), f"vLLM fallback engine used: {E.get('backend')}")
    check(SG.get("ok") is False and SG.get("error"), f"SGLang failure recorded: {str(SG.get('error'))[:160]}")
    if SCEN == "launchfail":
        check("no diagnosed fallback" in SG["error"] and len(SG["attempts"]) == 1, "permanent launch error: 1 launch, no retry")
    if SCEN == "abortbroken":
        check(SG["smoke"]["a"]["ok"] and not SG["smoke"]["b"]["ok"], f"smoke: stream ok, abort not honoured -> fail: {SG['smoke']}")
    if SCEN == "nocaplog":
        check("capacity gate failed" in SG["error"] and SG["caps"]["gate"]["mamba_cache_logged"] is False,
              f"missing mamba-cache evidence -> gate FAIL -> fallback: {SG['caps']['gate']}")
    check(G["E11_SERVER"] is None, "SGLang server killed before the fallback")
else:
    check(str(E.get("backend", "")).startswith("sglang"), f"SGLang engine used: {E.get('backend')}")
    check(SG["smoke"]["ok"], f"smoke ok: {SG['smoke']}")
    caps = SG["caps"]
    check(E["kv_gate"] == {"min_active": 4 if SCEN == "kvgate" else 10, "admit_max": 0.85} or SCEN == "kvgate",
          f"KV gate active on SGLang: {E['kv_gate']}")
    check(caps["kv_tokens"] == 881336 and caps["verify_graph_bs"][-1] == 12, f"caps parsed: {caps}")
    lf = SG["launch_flags"]
    check("--incremental-streaming-output" in lf and lf[lf.index("--stream-interval") + 1] == "1", "streaming flags in the launch (interval 1)")
    check(caps["gate"]["pass"], f"capacity gate PASS: {caps['gate']}")
    if SCEN == "lazyfail":
        check(SG["attempts"][0].get("retry_reason") == "lazy" and SG["attempts"][1]["ready"], "lazy rejected -> diagnosed relaunch")
        check(lf[lf.index("--mamba-radix-cache-strategy") + 1] == "extra_buffer" and lf[lf.index("--max-mamba-cache-size") + 1] == "70",
              "relaunch uses extra_buffer + mamba cache 70")
        check(caps["slots_per_request"] == 5 and caps["effective_running"] == 12, f"effective running 12 (70 // 5 = 14): {caps}")
    elif SCEN == "lowcap":
        check(caps["effective_running"] == 10 and G["CONCURRENCY"] == 10 and S["max_active"] <= 10,
              f"mamba cache 40 -> 10 effective -> concurrency 10 (max_active {S['max_active']})")
    else:
        check(len(SG["attempts"]) == 1 and lf[lf.index("--mamba-radix-cache-strategy") + 1] == "extra_buffer_lazy"
              and "--disable-prefill-cuda-graph" in lf and lf[lf.index("--mem-fraction-static") + 1] == "0.83",
              "first launch = E11-04 W1 flags")
        check(caps["effective_running"] == 12 and G["CONCURRENCY"] == 12, f"concurrency 12 (56 // 4 = 14): {caps['effective_running']}")
    st = G["engine"].stats
    print("shim stats:", json.dumps({k: v for k, v in st.items() if k != "errors"}), "| errors", st["errors"][:3])
    check(not G["E11_SERVER"].alive(), "server stopped at the end of the report cell")

if SCEN == "serverdeath":
    check("SGLang server" in str(G["E10_SCHED_ERROR"]), f"server death ends the solver with a clear error: {str(G['E10_SCHED_ERROR'])[-200:]}")
    sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
    check(set(sub) == set(test_file), "submission.json still written for every task")
    check(Path(G["OUT"] / "poetiq_run_report.json").exists(), "run report written")
    _parts = list((G["OUT"] / "poetiq_thinking").glob("*.partial.txt"))
    check(len(_parts) == S.get("partial_thinking_saved", -1) and len(_parts) > 0 and all(
          f.read_text(encoding="utf-8").startswith("# INCOMPLETE") for f in _parts),
          f"partial thinking of the running units saved: {len(_parts)} files")
else:
    check(G["E10_SCHED_ERROR"] is None, f"solver ended without an error ({str(G['E10_SCHED_ERROR'])[-300:]})")
    first = [B.parse_rid(r)[0] for r in ADMIT if r[-1] == "1" and B.parse_rid(r)[1] == 0]
    check(first == G["E10_ORDER"][:len(first)], f"iteration-1 admissions follow ORDER2 + cells: {first[:8]}")
    n_think = len(list((G["OUT"] / "poetiq_thinking").glob("*.txt")))
    check(n_think == S["iterations_finished"], f"full thinking saved per iteration: {n_think} files / {S['iterations_finished']} iterations")
    rep = json.load(open(G["OUT"] / "poetiq_run_report.json", encoding="utf-8"))
    check("e11" in rep and "error" not in rep["e11"], f"report has the E11 section: {str(rep.get('e11'))[:200]}")

if SCEN in ("mixed", "lazyfail", "launchfail", "abortbroken", "lowcap", "nocaplog"):
    check(F[multi[0]][0]["attempt_1"] == sol[multi[0]][0], "F3: train-passing program with a crashing test input keeps query 0")
    check(G["E10_STATES"][(single[1], 0)]["status"] == "train_verified", "F1: the last transform block was executed")
    check(all(p["attempt_1"] != p["attempt_2"] for t in F for p in F[t]), "A2: attempt_2 never repeats attempt_1")
    check(F[multi[1]][0]["attempt_1"] == sol[multi[1]][0], "forced iteration 1 then natural pass in iteration 2")
    check(G["E10_STATES"][(multi[2], 0)]["iteration"] == 3, "failing task gets all 3 iterations")
    check(all(G["E10_STATES"][(t, 0)]["status"] == "train_verified" for t in (single[0], single[3], single[4])), "pass tasks verified")
    it2 = [i for i, r in enumerate(ADMIT) if r[-1] == "1" and B.parse_rid(r)[1] == 1]
    it1_last = max(i for i, r in enumerate(ADMIT) if r[-1] == "1" and B.parse_rid(r)[1] == 0)
    check(not it2 or min(it2) > it1_last, "every task got iteration 1 before any iteration 2")
if SCEN in ("mixed", "lazyfail"):
    rid1, rid2 = f"{multi[1]}~e0~i0|1", f"{multi[1]}~e0~i0|2"
    a, b = FAKE_LOG.get(rid1), FAKE_LOG.get(rid2)
    check(a is not None and b is not None and b["input"] == a["input"] + a["sent"] + STOP,
          "forced close: phase-2 prompt = phase-1 prompt + every streamed delta + stop text (exact)")
    check(a is not None and len(a["sent"]) == a["max_new"] == G["THINK_BUDGET"], "phase-1 cap = 49,152 thinking tokens")
    nat = f"{single[0]}~e0~i0|1"
    check(FAKE_LOG.get(nat, {}).get("temperature") == 1.0 and FAKE_LOG.get(nat, {}).get("top_k") == 20,
          "Poetiq sampling sent (temperature 1.0, top_k 20)")
    st = G["engine"].stats
    check(st["by_reason"].get("stop", 0) > 0 and st["by_reason"].get("length", 0) > 0 and not st["errors"],
          f"finish reasons mapped: {st['by_reason']}")
    check(st["finished"] == st["added"], f"every request finished: {st['finished']}/{st['added']}")
    recs = [json.loads(l) for l in open(G["OUT"] / "sglang_requests.jsonl", encoding="utf-8")]
    check(any((r["cached_tokens"] or 0) > 0 for r in recs if "~i1|" in r["rid"]), "prefix-cache hits recorded for iteration 2")
    ds = json.load(open(G["OUT"] / "sglang_decode_summary.json", encoding="utf-8"))
    check(ds["n_lines"] > 0 and ds["by_context"], f"decode summary from the server log: {ds['n_lines']} lines")
if SCEN == "kvgate":
    check(S["max_active"] == 4, f"KV gate held running requests at the floor of 4 (max_active {S['max_active']})")
    check(S["kv_gate_blocks"] > 0 and S["kv_gate_last"] == 0.97, f"gate read SGLang full_token_usage: {S['kv_gate_blocks']}, {S['kv_gate_last']}")
if SCEN == "hardstop":
    check(S.get("remaining_verifications") == 0, f"verifications pending at the hard stop were harvested ({S.get('remaining_verifications')})")
    check(S["sandbox_verifications"] >= 1, f"verifications committed: {S['sandbox_verifications']}")
if SCEN == "slowverify":
    check(WALL < 10.0, f"F2b: queued verifications cancelled; solver ended {WALL:.1f}s after start")
    check(S.get("remaining_verifications", 0) >= 1, f"unfinished verifications recorded: {S.get('remaining_verifications')}")
if SCEN == "drain":
    for t in (single[0], single[3]):
        a, b = FAKE_LOG.get(f"{t}~e0~i0|1"), FAKE_LOG.get(f"{t}~e0~i0|2")
        ok = a is not None and b is not None and a["aborted"] and b["input"][:len(a["input"])] == a["input"] \
            and b["input"][-len(STOP):] == STOP
        k = len(b["input"]) - len(a["input"]) - len(STOP) if ok else -1
        check(ok and 0 < k <= len(a["sent"]) and b["input"][len(a["input"]):len(a["input"]) + k] == a["sent"][:k]
              and k < a["max_new"], f"{t}: drained at {k} of {len(a['sent']) if a else '?'} streamed tokens; phase 2 = prompt + "
                                    f"received deltas + stop text; phase 1 aborted on the server")
        check(G["E10_STATES"][(t, 0)]["status"] == "train_verified", f"{t}: drained answer verified")
        check(a is not None and b is not None and b["t_start"] >= a.get("t_finish", 1e18),
              f"{t}: phase 2 sent only after the aborted phase-1 stream ended (abort is asynchronous; 0.5 s delay)")
    st = G["engine"].stats
    check(st["abort_timeouts"] == 0 and st["abort_wait_sec_max"] >= 0.4, f"abort waited for the stream end: {st['abort_wait_sec_max']} s")
    drained_rids = [r for r in ABORTS if r.endswith("|1")]
    check(len(drained_rids) == 2, f"drain aborted exactly the 2 running phase-1 requests: {drained_rids}")
    i0 = EVENTS.index(("abort", drained_rids[0])) if drained_rids else len(EVENTS)
    check(bool(drained_rids) and not [r for k, r in EVENTS[i0:] if k == "add" and r.endswith("|1")],
          "no new phase-1 admission after the drain")
if SCEN == "rerun":
    check(E.get("drain_before_hard_stop_sec") == 720, f"SGLang drain 12 min before the hard stop: {E.get('drain_before_hard_stop_sec')}")
    sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
    check(set(sub) == set(test_file), "submission has every hidden task")
    check(all(len(sub[t]) == len(test_file[t]["test"]) for t in sub), "one attempt pair per test input")
    check(sub[single[0]][0]["attempt_1"] == sol[single[0]][0], "solved task written into submission.json")
print(f"\n{len(fails)} failed checks" if fails else "\nALL CHECKS PASS")
sys.stdout.flush(); sys.stderr.flush()
os._exit(1 if fails else 0)
