"""E28 W4 EVAL120 6 h Kaggle CPU kits (lead 2026-11-01): the same four kernels as NB6Q (units, e2e-rerun, e2e-commit, real) on
foysalemonshanto, the pinned Py3.12 image, no GPU, no internet, private. A wrapper, like NB6Q's kit over NB6's: the NB6Q kit
builder (Exp 25 build_e25_nb6q_cpu_kit.py, itself a wrapper over build_e25_nb6_cpu_kit.py; neither is written) is read as text,
the marked "E28 kit" edits below are applied to it and, through code injected into it, to the NB6 kit text it edits; it is then
executed with the E28 W4 notebook (+ its code export). Every NB6 / NB6Q kit check stays unless an E28 lever changes its expected
value (listed). Each kernel prints exactly one line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.

  K1  names: notebook / export / slug q27-e28-w4-kit / titles / tmp dir / report names; the push folder under test = the E28 W4 one
  K2  units: e28w4_unit_tests.py (T1-T7, the mounted Duck payload for T6; all PASS, the killpg deadman path runs on Linux)
      replaces NB6Q's nb6q_unit_tests S1 + S3; nb5_unit_tests on NB5 QUALITY (gated) and on this notebook (recorded) stay; the
      embedded-notebook sha check is NB6Q 499a4319 (the reference the review diff is computed against)
  K3  e2e (both): the profile E28EVAL120 (commit, with a fake short deadline Q27X_TIME_BUDGET, the baked 412 min checked in P) /
      RERUN_PROD (rerun); the fake boot = pin +0.75 GiB (fake pool 197, F 5), launch seqs 12, workers 12 (C12 rung); rerun-retry: the fake
      OOM above RC2 + 0.6 GiB -> 0.75 fails, 0.5 boots (fake pool 181); the commit solve set = all 120 eval tasks;
      E28_B2_ATTEMPT 9,000 (harness only). The kit-local fake server additionally: accepts --max-model-len 262144 (unused since the 131,072 decision), reports
      language_model_only, puts DRAFT_HEAD=int8 into its speculative config and prints NB4T's two 'Draft lm_head: INT8' line
      shapes (FAKE ONLY, no engine claim), records DRAFT_HEAD per launch
  K4  e2e E28 checks: scheduler APPLIED (post sha, env from the notebook's numbers), the run went through the A/B/C scheduler
      (SCHED_LAST e28: every task has its phase-A attempt, branches 0/1/2, phases A/B1/B2/C, every cap-cut A has its B2 re-run
      with the B2 budget and the 130,560 window), no ORDER2, every launch argv (131072, seqs 12 last, --language-model-only)
      and env (DRAFT_HEAD), the I8 / C12 / 131072 / text-only read-back + pool / F, slots admission (12 slots = workers, preemptions recorded; workers
      12), full saved responses (every attempt's nb5_requests.jsonl, rows = traced requests, reasoning + content keys), the
      e28 scheduler report, the request timeouts (7,200 s; B2 scaled), the deadman (commit: armed at HARD_STOP + 300 s and the
      run ended before it; rerun: not armed)
  K5  real: the boot plans start at the I8 pin (pin_tries(..., 0.75): 3 rungs) with seqs 12; argv checks: --max-model-len
      131072, --language-model-only, Merlin's --speculative-config carries draft_lm_head_quant int8, the plan records the build
      env; vLLM's own argparse: max_model_len 131072, language_model_only True, the spec's draft head int8; + the E28 section:
      SpeculativeConfig accepts draft_lm_head_quant 'int8' (the Merlin venv's own type hints), the checkpoint's max position
      embeddings (recorded), the scheduler patch on the REAL extracted payload (both namespaces APPLIED, verify -> revert ->
      ORIGINAL pre sha -> re-apply -> FULLY_PATCHED, the imported arc2_duck has the E28 hand-over)
  v2  (kits v1 on Kaggle, notebook 809cb2ad): V2-1 F_log tolerance at 262144, V2-2 no argparse.py in K (it shadowed the stdlib)
      + the fake-L4 fallback for vLLM's argparse, V2-3 strip every --port, V2-4 the commit profile in the submission check,
      V2-5 the inherited E28 env in the server recorded, not asserted absent (all KIT defects; the notebook unchanged)
Rules: nothing is pushed here; long tests only on the Kaggle CPU kits.
Run: python build_e28_w4_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e28-w4-kit-{units,e2e-rerun,e2e-commit,real}__foysalemonshanto/
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
QKIT = E25B / "build_e25_nb6q_cpu_kit.py"
NB = HERE.parent / "Q27-XHIGH-E28_W4_EVAL120_6H.ipynb"
EXP = HERE / "e28w4_code_export.py"
FACTS = json.loads((HERE / "e28w4_build_facts.json").read_text(encoding="utf-8"))
NB6Q_NB = E25B.parent / "Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb"
NB6Q_SHA = "499a431904fdaf47831ef14529b7b956c45d9e374c6c427b3b3e4698fa8a5c7d"
PUSHQ_META = Path(r"D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb6q-submit__foysalemonshanto/kernel-metadata.json")
DUCK_POST = FACTS["sched"]["patched_arc2_duck_sha256"]
DUCK_PRE = "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128"
assert hashlib.sha256(NB.read_bytes()).hexdigest() == FACTS["sha256"], "the notebook is not the one in the build facts (rebuild)"
assert hashlib.sha256(NB6Q_NB.read_bytes()).hexdigest() == NB6Q_SHA


def once(text, old, new, what, count=1):
    assert text.count(old) == count, (what, text.count(old))
    return text.replace(old, new)


def src(p):
    return Path(p).read_text(encoding="utf-8").replace("\r\n", "\n")


Q = src(QKIT)
# ---------------------------------------------------------------- K1 names (NB6Q kit text)
Q = once(Q, 'NBQ = HERE.parent / "Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb"', f'NBQ = Path(r"{NB}")   # E28 kit K1', "K1 nb")
Q = once(Q, 'EXPQ = HERE / "nb6q_code_export.py"', f'EXPQ = Path(r"{EXP}")   # E28 kit K1', "K1 export")
for o, n in (("q27-e25-nb6q-kit", "q27-e28-w4-kit"), ('"/tmp/nb6qkit"', '"/tmp/e28w4kit"'), ("nb6q_kit_{KIT}_report.json", "e28w4_kit_{KIT}_report.json"),
             ('"E25 NB6 CPU kit", "E25 NB6Q CPU kit"', '"E25 NB6 CPU kit", "E28 W4 CPU kit"')):
    Q = Q.replace(o, n)
Q = once(Q, 'f\'"title": "Q27 E25 NB6Q Kit {kit}"\'', 'f\'"title": "Q27 E28 W4 Kit {kit}"\'', "K1 titles")
Q = once(Q, '''PUSH6 = ROOT / "q27-xhigh-e25-nb6q-submit__foysalemonshanto"   # NB6Q kit K1: the folder under test''',
         '''PUSH6 = ROOT / "q27-xhigh-e28-w4-eval120-6h__foysalemonshanto"   # E28 kit K1: the folder under test''', "K1 push")
# ---------------------------------------------------------------- K2 units files (NB6Q kit's NS)
Q = once(Q, '''      "NB6Q_UNIT_FILES": {"nb6ref.ipynb": NB6_RAW.decode("utf-8"), "push6ref/kernel-metadata.json": PUSH6REF,
                          **{n: src(n) for n in ("nb6q_unit_tests.py", "nb6q_lib.py", "nb4_ppack.py")}},''',
         '''      "NB6Q_UNIT_FILES": E28_UNIT_FILES,   # E28 kit K2''', "K2 unit files")

# ---------------------------------------------------------------- the code injected into the NB6Q kit (edits of the NB6 kit text S + FAKE)
INJ = src(HERE / "e28w4_kit_inject.py")   # the injected code (a template; see its header)

E2E_E28 = r'''
# ---- E28 kit K4: the E28 levers + the A/B/C scheduler on the fake server
X = G.get("E28_SCHED_REC") or {}
_e28r = ((G.get("E12_RUN") or {}).get("rev6_scheduler") or {}).get("e28") or {}
_rows = [r_ for r_ in (_e28r.get("attempts") or []) if isinstance(r_, dict)]
_acls = _e28r.get("a_class") or {}
_bt = [t_ for b in boots for t_ in b.get("tries") or [] if (t_.get("e28_readback") or {}).get("after_warmup")]
_rb = (_bt[-1]["e28_readback"]["after_warmup"] if _bt else {})
_last = lambda a, f: (lambda i: a[i + 1] if 0 <= i < len(a) - 1 else None)(len(a) - 1 - a[::-1].index(f) if f in a else -1)
_DD = W / "q27x" / "phase_EVAL" / "duck"
_rq = {p.parent.name: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] for p in _DD.glob("*_b*/nb5_requests.jsonl")}
_rq_rows = [r_ for v in _rq.values() for r_ in v if r_.get("kind") != "no_request"]
REC["e28"] = {"sched_rec": {k: X.get(k) for k in ("state", "applied", "sha_after", "env", "error", "restored")},
              "phases": {p_: sum(1 for r_ in _rows if r_.get("phase") == p_) for p_ in ("A", "B1", "B2", "C")},
              "a_class": {t_: (v.get("kind") if isinstance(v, dict) else v) for t_, v in _acls.items()},
              "b2_rows": [r_ for r_ in _rows if r_.get("phase") == "B2"][:10], "readback": {k: _rb.get(k) for k in ("i8_label", "launch_knobs", "knobs_ok")},
              "launch_env": [L_.get("env_e28") for L_ in launches], "request_files": len(_rq), "request_rows": len(_rq_rows),
              "config": _e28r.get("config"), "deadman": {k: v for k, v in (G.get("E28_DEADMAN") or {}).items() if k != "thread"}}
check("E28: the scheduler patch APPLIED in CELL 3 (arc2_duck.py post sha __DUCKPOST__), the notebook env E28_SCHED=1 / T1 = NB5_ATTEMPT b0 "
      "(6000) / B2 = E28_B2_ATTEMPT (9000) / window 130560, recorded in nb5_report.json",
      lambda: X["state"] == "APPLIED" and X["sha_after"] == "__DUCKPOST__" and (rep.get("e28_sched") or {}).get("state") == "APPLIED"
      and X["env"] == {"E28_SCHED": "1", "E28_T1": "6000", "E28_B2_TOKENS": "9000", "E28_B2_WINDOW": "130560",
                       "E28_B2_REQUEST_TIMEOUT": "9000"}, REC["e28"]["sched_rec"])
check("E28: the solve ran the A/B/C scheduler: SCHED_LAST e28 enabled, every solve-set task has its phase-A attempt (branch 0), "
      "branches only 0 / 1 / 2, phases only A / B1 / B2 / C",
      lambda: _e28r.get("enabled") is True and {r_["task"] for r_ in _rows if r_.get("phase") == "A" and r_.get("branch") == 0} == set(SOLVE_IDS)
      and {r_.get("branch") for r_ in _rows} <= {0, 1, 2} and {r_.get("phase") for r_ in _rows} <= {"A", "B1", "B2", "C"},
      json.dumps(REC["e28"]["phases"]) + " " + json.dumps(sorted({(r_.get("phase"), r_.get("branch")) for r_ in _rows}, key=str)))
_b2t = sorted(t_ for t_, v in REC["e28"]["a_class"].items() if v == "B2")
check("E28: every task whose attempt 1 was cap-cut (A class B2) got its B2 re-run (branch 2) with the B2 budget 9000 and the Duck window "
      "130,560; B2 rows only for those tasks", lambda: all(any(r_.get("task") == t_ and r_.get("phase") == "B2" and r_.get("branch") == 2
      and r_.get("budget") == 9000 and r_.get("window_plan") == 130560 for r_ in _rows) for t_ in _b2t)
      and {r_.get("task") for r_ in _rows if r_.get("phase") == "B2"} <= set(_b2t), {"b2_tasks": _b2t, "b2_rows": REC["e28"]["b2_rows"]})
check("E28: phase C (attempt 2, branch 1) ran for >= 1 task, with the T1 budget (6000)", lambda: any(r_.get("phase") == "C" and r_.get("branch") == 1
      and r_.get("budget") == 6000 for r_ in _rows), REC["e28"]["phases"])
check("E28: no ORDER2: E12_ORDER == the plain TASK_IDS order", lambda: G["E12_ORDER"] == list(G["TASK_IDS"]), (G.get("E12_ORDER") or [])[:6])
_heads = [(L_.get("env_e28") or {}).get("DRAFT_HEAD") for L_ in launches]
_SEQ_OF_KV = {"3285145068": "12", "3016709612": "8", "2479838700": "6"}   # kit v4: C12 on the I8 pin only, NB6Q's table below
check("E28: every (fake) launch: --max-model-len 131072 (last), --max-num-seqs (last) 12 on the I8 pin 3,285,145,068 B, NB6Q's table "
      "below (0.5 -> 8, RC2 -> 6), --language-model-only; DRAFT_HEAD=int8 in the "
      "server env (FAKE: the build env) on every I8 launch" + ("; the same-pin BF16 retry has NO DRAFT_HEAD (kit v3)" if RETRY else "")
      + "; E28_SCHED as the server inherits it is recorded (kit v2 V2-5)",
      lambda: launches and all(_last(L_["argv"], "--max-model-len") == "131072" and _last(L_["argv"], "--max-num-seqs") == _SEQ_OF_KV.get(_last(L_["argv"], "--kv-cache-memory-bytes"))
                               and "--language-model-only" in L_["argv"] for L_ in launches)
      and (_heads[:-1] == ["int8"] * (len(_heads) - 1) and _heads[-1] is None if RETRY else _heads == ["int8"] * len(_heads)),
      REC["e28"]["launch_env"])
_bt0 = [t_ for t_ in b0.get("tries") or []]
if RETRY:
    check("E28 I8 gate (kit v3, the fake prints no INT8 build line): the 0.5 I8 try = I8_READBACK_UNVERIFIED (SPEC_ONLY), the server "
          "stopped; the SAME pin (3,016,709,612 B, seqs 8 = NB6Q's table: not the C12 rung) retried with the BF16 head (launch name _bf16head) booted; recorded in "
          "brec i8_fallback; the booted try's read-back = NOT_REQUESTED (BF16 head)",
          lambda: _bt0[1]["status"] == "I8_READBACK_UNVERIFIED" and (_bt0[1].get("i8_gate") or {}).get("label") == "SPEC_ONLY"
          and _bt0[1]["kv_bytes"] == _bt0[2]["kv_bytes"] == 3016709612 and _bt0[1]["max_num_seqs_launch"] == _bt0[2]["max_num_seqs_launch"] == 8
          and _bt0[0]["max_num_seqs_launch"] == 12 and (_bt0[0].get("e28_rung") or {}).get("c12") is True
          and (_bt0[2].get("e28_rung") or {}).get("c12") is False and (_bt0[2].get("e28_rung") or {}).get("workers") == 8
          and _bt0[2]["name"] == _bt0[1]["name"] + "_bf16head" and len(b0.get("i8_fallback") or []) == 1
          and _rb["i8_label"] == "NOT_REQUESTED" and all(_rb["knobs_ok"].values()) and _rb["capacity"]["pool_blocks"] == _cw[1],
          json.dumps({"tries": [{k: t_.get(k) for k in ("name", "status", "kv_bytes", "max_num_seqs_launch", "i8_gate")} for t_ in _bt0],
                      "fallback": b0.get("i8_fallback"), "rb": REC["e28"]["readback"]}, default=str)[:1500])
else:
    check("E28: the I8 gate passed (ENGAGED) on the booted try; read-back after the warm-up: I8 ENGAGED (the kit fake prints NB4T's "
          "line shapes), seqs 12, max-model-len 131072, text-only all ok; pool / F / workers recorded (nb5_report e28 capacity)",
          lambda: (_bt0[-1].get("i8_gate") or {}).get("verdict") == "ENGAGED" and _rb["i8_label"] == "ENGAGED"
          and all(_rb["knobs_ok"].values()) and _rb["capacity"]["pool_blocks"] == _cw[1]
          and _rb["capacity"]["workers"] == 12 and (rep.get("e28") or {}).get("capacity") and not b0.get("i8_fallback"),
          REC["e28"]["readback"])
check("E28: admission = slots (the reservation = slots = workers, never over it), workers " + ("8 (0.5 rung: NB6Q's slots rule, "
      "no C12)" if RETRY else "12 (the C12 rung: rule overridden)") + ", C-8 not hit; "
      "the run-level preemption record in the scheduler report",
      lambda: G["NB5_ADMIT_MODE"] == "slots" and main["reservation"]["capacity_blocks"] == _cw[4] and cap["capacity_blocks"] == _cw[1]
      and main["workers"] == _cw[4] and "preemptions" in (rep.get("e28_sched_report") or {})
      and (cap.get("e28_workers") or {}).get("workers") == _cw[4] and cap["e28_workers"]["c12_rung"] is (not RETRY)
      and cap["e28_workers"]["workers_fixed"] == (None if RETRY else 12), json.dumps({"res": main.get("reservation"), "cap": {k: cap.get(k) for k in ("capacity_blocks", "workers", "e28_workers")}})[:900])
check("E28: full saved responses: every attempt with tokens has its nb5_requests.jsonl; rows == the tracer's requests; every row carries "
      "the full 'reasoning' / 'content' / 'tool_calls' fields",
      lambda: _rq and {f"{r_['task']}_b{r_['branch']}" for r_ in _rows if int(r_.get("tokens") or 0) > 0} <= set(_rq)
      and len(_rq_rows) == tr["requests"] and all({"reasoning", "content", "tool_calls"} <= set(r_) for r_ in _rq_rows),
      {"files": len(_rq), "rows": len(_rq_rows), "tracer_requests": tr.get("requests")})
_sr = rep.get("e28_sched_report") or {}
check("E28: the scheduler report (CELL 9): state OK, rows = SCHED_LAST attempts, per-phase counts, e28_sched_report.json written",
      lambda: _sr.get("state") == "OK" and _sr.get("rows") == len(_rows) and (W / "q27x" / "e28_sched_report.json").is_file()
      and sum(v["attempts"] for v in _sr["phases"].values()) == len(_rows), {k: _sr.get(k) for k in ("state", "rows", "phases", "file")})
check("E28: request timeouts: the notebook env E12_DUCK_REQUEST_TIMEOUT 7200 and E28_B2_REQUEST_TIMEOUT 9000; the scheduler's request "
      "timeout 7200, B2 9000", lambda: G["E12_DUCK_REQUEST_TIMEOUT"] == 7200 and X["env"].get("E28_B2_REQUEST_TIMEOUT") == "9000"
      and float(_e28r["config"]["request_timeout"]) == 7200.0 and float(_e28r["config"]["b2_request_timeout"]) == 9000.0, _e28r.get("config"))
if RR:
    check("E28 deadman (rerun): NOT armed (RERUN_PROD keeps NB6's behaviour)", lambda: G["E28_DEADMAN"]["armed"] is False, REC["e28"]["deadman"])
else:
    check("E28 deadman (commit): armed at HARD_STOP + 300 s; the cells ended before it would fire",
          lambda: G["E28_DEADMAN"]["armed"] is True and abs(G["E28_DEADMAN"]["fire_pc"] - G["HARD_STOP"] - 300) < 1e-6
          and T_END < G["E28_DEADMAN"]["fire_pc"], REC["e28"]["deadman"])
'''

REAL_E28 = r'''
# ---------------------------------------------------------------- E28 kit K5: the scheduler on the REAL payload, the I8 spec type, the checkpoint length
E28R = {}
try:
    _M = GR.get("E28S") or G.get("E28S")
    _d = GR["E12_DIR"]
    E28R["sched"] = {"commit": (G.get("E28_SCHED_REC") or {}).get("state"), "rerun": (GR.get("E28_SCHED_REC") or {}).get("state"),
                     "commit_rec": {k: (G.get("E28_SCHED_REC") or {}).get(k) for k in ("state", "sha_after", "env", "error")},
                     "state_now": _M.tree_state(_d)}
    _v0 = _M.verify_e28_sched(_d)
    _rv = _M.revert_e28_sched(_d)
    _pre = hashlib.sha256((Path(_d) / "arc2_duck.py").read_bytes()).hexdigest()
    _ra = _M.apply_e28_sched(_d)
    _post = hashlib.sha256((Path(_d) / "arc2_duck.py").read_bytes()).hexdigest()
    E28R["sched"].update(verify_before=list(_v0), revert=_rv.get("reverted"), pre=_pre, reapply=_ra.get("applied"), post=_post,
                         roundtrip=bool(_v0[0]) and _pre == "__DUCKPRE__" and _post == "__DUCKPOST__" and _M.verify_e28_sched(_d)[0] is True,
                         handover=callable(getattr(GR.get("E12A"), "_e28_solve_all", None)) and callable(getattr(G.get("E12A"), "_e28_solve_all", None)))
except BaseException as e:
    E28R["sched_error"] = f"{type(e).__name__}: {e}"[:800]
    E28R.setdefault("sched", {})["tb"] = traceback.format_exc()[-1500:]
SPECQ = """
import json, typing
out = {}
try:
    from vllm.config import speculative as S
    h = typing.get_type_hints(S.SpeculativeConfig)
    t = h.get("draft_lm_head_quant")
    flat = set()
    for a in typing.get_args(t):
        flat |= set(typing.get_args(a) or (a,))
    out.update(field=str(t), values=sorted(str(x) for x in flat), ok="int8" in flat)
except BaseException as e:
    out.update(ok=False, error=f"{type(e).__name__}: {e}"[:600])
print("@JSON " + json.dumps(out, default=str))
"""
try:                          # plain CPU import first (VLLM_TARGET_DEVICE=cpu, as the argparse probe); else E27's fake-L4 platform
    _r = venv_py(SPECQ, [], name="e28_specq", extra_env=CPU_ENV)
    E28R["spec"] = dict(_r["json"] or {"ok": False, "tail": _r["tail"]}, how="cpu")
    if E28R["spec"].get("ok") is not True:
        _r2 = venv_py(E27_PLATFORM + SPECQ, [], name="e28_specq_l4", extra_env=e28_l4_env())
        E28R["spec_cpu_first"] = E28R["spec"]
        E28R["spec"] = dict(_r2["json"] or {"ok": False, "tail": _r2["tail"]}, how="fake_l4")
except BaseException as e:
    E28R["spec"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
try:
    _cf = json.loads((Path(G["CKPT"]) / "config.json").read_text(encoding="utf-8"))
    _tc = _cf.get("text_config") if isinstance(_cf.get("text_config"), dict) else {}
    _mp = _tc.get("max_position_embeddings") or _cf.get("max_position_embeddings")
    E28R["ckpt"] = {"max_position_embeddings": _mp, "rope_scaling": _tc.get("rope_scaling") or _cf.get("rope_scaling"),
                    "rope_theta": _tc.get("rope_theta") or _cf.get("rope_theta"), "ok": bool(_mp) and int(_mp) >= 262144}
except BaseException as e:
    E28R["ckpt"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
REP["checks"]["e28"] = {k: v for k, v in E28R.items()}; save()
log("e28 real", json.dumps(E28R, default=str)[:3000])
'''

E28_UNITS = r'''
PAY = find_payload(); REP["checks"]["payload"] = PAY; save()
log("payload", json.dumps(PAY)[:1200])
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    if not n.startswith("push"):
        (B / n).write_text(v, encoding="utf-8", newline="\n")
for d_ in ("push6", "push5", "push6ref"):
    (K / d_).mkdir()
    (K / d_ / "kernel-metadata.json").write_text(FILES[d_ + "/kernel-metadata.json"], encoding="utf-8", newline="\n")
(K / "push6" / FILES["push6/code_file"]).write_bytes((B / "nb6.ipynb").read_bytes())    # = the E28 W4 build (sha checked below)
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5", "NB6", "E12_", "E15B_", "Q27X_", "RC2_", "E23_", "FAKE_", "E28"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
U = {"sha_nb_written": hashlib.sha256((B / "nb6.ipynb").read_bytes()).hexdigest() == "__NBSHA__",
     "sha_nb6q_ref_written": hashlib.sha256((B / "nb6q.ipynb").read_bytes()).hexdigest() == "__NB6QSHA__"}
N6 = {}
try:                                       # E28 kit K2: e28w4_unit_tests.py T1-T7 (the mounted Duck payload for T6)
    r = sh("e28w4_units", [sys.executable, str(B / "e28w4_unit_tests.py")],
           {"E28T_NB": str(B / "nb6.ipynb"), "E28T_NB6Q": str(B / "nb6q.ipynb"), "E28T_FACTS": str(B / "e28w4_build_facts.json"),
            "E28T_DIFF": str(B / "e28w4_review_diff_vs_nb6q.diff"), "E28T_LIB": str(B / "e28w4_lib.py"),
            "E28T_SCHED": str(B / "e28_sched_patch.py"), "E28T_FIXTURE": str(B / "e28w4_nb4t_log_fixture.json"),
            "E28T_PAYLOAD": str(PAY["used"]), "E28T_PUSH": str(K / "push6"), "PYTHONIOENCODING": "utf-8"}, 3600, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+) / (\d+) PASS \((\d+) skipped\)", l) for l in lines if re.match(r"\d+ / \d+ PASS", l)), None)
    N6.update(rc=r["rc"], sec=r["sec"], fails=[l[:400] for l in lines if l.startswith("FAIL ")][:40],
              skips=[l[:300] for l in lines if l.startswith("SKIP ")], summary=m.group(0) if m else r["tail"][-800:], tail=r["tail"][-800:],
              all_pass=bool(m) and r["rc"] == 0 and m.group(1) == m.group(2) and m.group(3) == "0" and int(m.group(2)) >= 70
              and "@@E28W4_UNITS_OK@@" in r["_out"])
except BaseException as e:
    N6.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["e28w4_units"] = N6
REP["checks"]["units"] = U; save()
log("e28w4 units", json.dumps({k: v for k, v in N6.items() if k != "tb"})[:2500])
verdict({"payload_sha_ok_under_input": PAY["used"] is not None, "embedded_notebooks_sha_ok": U["sha_nb_written"] and U["sha_nb6q_ref_written"],
         "e28w4_units_T1_T7_all_pass_0_skipped": N6.get("all_pass") is True},
        recorded={"e28w4_units": N6.get("summary"), "fails": N6.get("fails"), "skips": N6.get("skips"), "payload_used": PAY.get("used")})
'''
_e27 = src(Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build/build_e27_nbdf_cpu_kit.py"))
_p0 = _e27.index('PLATFORM = r"""\n') + len('PLATFORM = r"""\n')
E27_PLATFORM = _e27[_p0:_e27.index('"""', _p0)]     # E25 klee100 kit / E27 NBDF kit: the fake-L4 platform for vllm config imports
assert "class FakeL4(VPC.NonNvmlCudaPlatform):" in E27_PLATFORM and "VP._current_platform = FakeL4()" in E27_PLATFORM
UNIT_FILES = {"nb6q.ipynb": NB6Q_NB.read_bytes().decode("utf-8"), "push6ref/kernel-metadata.json": PUSHQ_META.read_text(encoding="utf-8"),
              **{n: src(HERE / n) for n in ("e28w4_unit_tests.py", "e28w4_unpack.py", "e28w4_lib.py", "e28_sched_patch.py",
                                             "e28w4_build_facts.json", "e28w4_review_diff_vs_nb6q.diff", "e28w4_nb4t_log_fixture.json")}}
INJ = (INJ.replace("__E28UNITFILES__", repr(UNIT_FILES)).replace("__E27PLATFORMREPR__", repr(E27_PLATFORM)).replace("__E28UNITSBODY__", repr(E28_UNITS.replace("__NB6QSHA__", NB6Q_SHA))).replace("__E2EE28__", repr(E2E_E28.replace("__DUCKPOST__", DUCK_POST)))
       .replace("__REALE28__", repr(REAL_E28.replace("__DUCKPOST__", DUCK_POST).replace("__DUCKPRE__", DUCK_PRE)
                                       .replace("E27_PLATFORM + SPECQ", repr(E27_PLATFORM) + " + SPECQ")))
       .replace("__NB6QSHA__", NB6Q_SHA))
compile(INJ, "<e28 kit injection>", "exec")
Q = once(Q, "NB6_RAW = NB6_NB.read_bytes()\n", INJ + "\nNB6_RAW = NB6_NB.read_bytes()\n", "inject")
Q = once(Q, 'if __name__ == "__main__":\n    print("notebook under test", NS["NB_PATH"].name', 'if __name__ == "__nb6q_kit_main__":\n    print("notebook under test", NS["NB_PATH"].name',
         "no nb6q main")
QNS = {"__file__": str(QKIT), "__name__": "e28w4_kit_q"}
exec(compile(Q, str(QKIT) + " (E28 kit edits)", "exec"), QNS)
NS = QNS["NS"]
assert NS["NB_PATH"] == NB.resolve() and NS["EXPORT_OK"] is True and NS["PUSH6_OK"] is True
compile(NS["NB6_E2E"], "<e28 e2e>", "exec")

if __name__ == "__main__":
    print("notebook under test", NS["NB_PATH"].name, "sha256", NS["NB_SHA"], "| export", NS["EXPORT_PATH"].name, "equal", flush=True)
    for n in NS["KITS"]:
        NS["build"](n)
