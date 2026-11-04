"""E25 NB5 unit tests (fault injection + replay on REAL recorded RC2 data; never invented metric fixtures).

Real inputs (read-only): the RC2 GPU run output D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2/ (server_merlin_fp8.log,
server_merlin_bf16.log, rc2_results.json, traces/requests.jsonl, telemetry.jsonl), the Flash-Next tokenizer + chat
template (the RC2 local copy, sha-pinned), the commit-turn CPU test script + its recorded result, the Duck payload
(E15B 9e4b8928), the generated notebook.
usage: python nb5_unit_tests.py   (exit 0 = all PASS)
Kaggle CPU kit: NB5T_KIT=1 + NB5T_RC2 / NB5T_TOKDIR / NB5T_PAYLOAD / NB5T_NB / NB5T_CT_RESULT point at the kit's copies
(requests.jsonl may carry n_tokens instead of tokens); the checks that need the builder and its donor notebooks (verbatim
donors, packed-lab equality, literal scans, name collisions, PUSHED guard) are build-time checks: printed as SKIP, not run."""
import ast
import base64
import hashlib
import io
import json
import math
import os
import re
import runpy
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb5_lib as NB5L  # noqa: E402

KIT = os.environ.get("NB5T_KIT") == "1"          # Kaggle CPU kit mode (paths from env, build-time checks skipped)
RC2 = Path(os.environ.get("NB5T_RC2") or r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2")
TOKDIR = Path(os.environ.get("NB5T_TOKDIR") or r"D:/ARC AGI 2/_scratch/e13_rc2_local/tmp/fn_tok")
E25 = HERE.parents[1]
CT_SCRIPT = HERE / "commit_turn_prefix_test.py"
CT_RESULT = Path(os.environ.get("NB5T_CT_RESULT") or E25 / "Experiment Results" / "commit_turn_prefix_test_2026-10-31.json")
PAYLOAD = Path(os.environ.get("NB5T_PAYLOAD") or r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload/q27x_e15b_duck_9e4b89285136.bin")
NB = Path(os.environ.get("NB5T_NB") or HERE.parent / "Q27-XHIGH-E25_NB5_QUALITY.ipynb")   # R2: the notebook under test
# NB5Q-ENV-1: Merlin's own launcher (tools/merlin of ad42e37 = the e23-merlin-env payload tree; the kit ships these 3 files)
MERLIN_TOOLS = Path(os.environ.get("NB5T_MERLIN_TOOLS") or r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src/tools/merlin")
MERLIN_LAUNCH_SHA = {"launch.py": "87d1c5b75654fdec040735f1cf7f1d80c6a4656a6ffcd0248243c7eefe61cb46",
                     "registry.py": "d8adf99d9d032ebc3a46b01385c344d5f583bf3dc940f34b60bb3e7ecfa49020",
                     "profiles/flash-next.json": "ef0d63cf784db8c4a298d7f5c17e453bf983a74228e17b186138d5ed8abe4103"}
RESULTS = []
SKIPPED = []


def skip(name):
    SKIPPED.append(name)
    print("SKIP " + name + "  [kit mode: build-time check, needs the builder + its donor notebooks]", flush=True)


def check(name, cond, detail=""):
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, detail = False, f"raised {type(e).__name__}: {e} | {detail}"
    RESULTS.append((name, ok, str(detail)[:900]))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  [{str(detail)[:900]}]"), flush=True)


FP8_LOG = (RC2 / "logs" / "server_merlin_fp8.log").read_text(encoding="utf-8", errors="replace")
BF16_LOG = (RC2 / "logs" / "server_merlin_bf16.log").read_text(encoding="utf-8", errors="replace")
RES = json.loads((RC2 / "rc2_results.json").read_text(encoding="utf-8"))
REQS = [json.loads(l) for l in open(RC2 / "traces" / "requests.jsonl", encoding="utf-8")]
TEL = [json.loads(l) for l in open(RC2 / "telemetry.jsonl", encoding="utf-8")]
A3 = RES["sections"]["A3"]["merlin_fp8"]
WINDOWS = {**A3["steady"], **A3["census_long"]}
STAT_RE = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
                     r"Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: \d+ reqs)?(?:, KV fetch: [^%]*?completed waiting)?"
                     r"(?:, Preemptions: (\d+))?, GPU KV cache usage: ([\d.]+)%(?:, Prefix cache hit rate: ([\d.]+)%)?")


def rows_of(text):        # = E21's e21_stat_rows (the notebook's own copy is checked below on the same log)
    return [{"prompt_tps": float(g[0]), "gen_tps": float(g[1]), "running": int(g[2]), "waiting": int(g[3]),
             "preempt": int(g[4]) if g[4] else 0, "kv_pct": float(g[5]), "prefix_pct": float(g[6]) if g[6] else None}
            for g in (m.groups() for m in STAT_RE.finditer(text))]


FP8_ROWS = rows_of(FP8_LOG)

# ================================================================== 1. capacity from the REAL RC2 server logs
c8 = NB5L.server_caps(FP8_LOG)
check("caps FP8 (real log): max_num_seqs 8, block 4096, max_model_len 131072 (the LAST value), MTP 3, KV 513,940, conc 3.92",
      c8 == {**c8, "max_num_seqs": 8, "block_size": 4096, "max_model_len": 131072, "spec_tokens": 3, "kv_tokens": 513940,
             "conc_tokens": 131072, "conc_x": 3.92}, c8)
cb = NB5L.server_caps(BF16_LOG)
check("caps BF16 (real log): max_num_seqs 3, KV 269,042, conc 2.05", cb["max_num_seqs"] == 3 and cb["kv_tokens"] == 269042
      and cb["conc_x"] == 2.05, cb)
exact = [w["kv_usage_max"] for w in WINDOWS.values() if w.get("kv_usage_max")] + [w["kv_usage_mean"] for w in WINDOWS.values() if w.get("kv_usage_mean")]
check("pool from the real KV-usage gauge (rc2_results window values): 149 blocks (Merlin's 150-block pin minus the null block)",
      NB5L.pool_from_gauge([w["kv_usage_max"] for w in WINDOWS.values()]) == 149,
      NB5L.pool_from_gauge([w["kv_usage_max"] for w in WINDOWS.values()]))
check("pool from the real stats-line gauge (318 lines, 0.1 % rounding): 149", NB5L.pool_from_gauge(
    [r["kv_pct"] / 100 for r in FP8_ROWS], tol=0.004) == 149, NB5L.pool_from_gauge([r["kv_pct"] / 100 for r in FP8_ROWS], tol=0.004))
check("pool_from_metrics: cache_config_info num_gpu_blocks 150 -> usable 149; absent -> None",
      NB5L.pool_from_metrics('vllm:cache_config_info{block_size="4096",num_gpu_blocks="150"} 1.0\n')["usable"] == 149
      and NB5L.pool_from_metrics("vllm:num_requests_running 0\n") is None)
p8 = NB5L.capacity_plan(149, 4096, {"b0": 112000, "b1": 100000}, 8192, 8192, 131072, 6, 0.95, seqs=8,
                        conc_tokens=131072, conc_x=3.92)
check("capacity FP8 (real numbers): F_log = 149/3.92 - 32 = 6.01, F_eff 6, capacity 141 blocks, attempt 1 = 38 blocks -> 3 "
      "streams, attempt 2 = 35 -> 4, workers 4", abs(p8["f_log"] - 6.01) < 0.01 and p8["f_eff"] == 6 and p8["capacity_blocks"] == 141
      and p8["per_attempt"]["b0"]["blocks"] == 38 and p8["per_attempt"]["b0"]["streams"] == 3
      and p8["per_attempt"]["b1"]["streams"] == 4 and p8["workers"] == 4, json.dumps(p8)[:600])
pb = NB5L.capacity_plan(78, 4096, {"b0": 112000, "b1": 100000}, 8192, 8192, 131072, 6, 0.95, seqs=3, conc_tokens=131072, conc_x=2.05)
check("capacity BF16 (real numbers): F_log 6.05, capacity 74, 1 stream per long attempt", pb["f_eff"] == 6 and pb["capacity_blocks"] == 74
      and pb["per_attempt"]["b0"]["streams"] == 1, json.dumps(pb)[:400])
_p0 = NB5L.capacity_plan(30, 4096, {"b0": 104000, "b1": 104000}, 8192, 8192, 131072, 6, 1.0, seqs=8)
check("C-8: no attempt kind fits the pool (30 blocks < 36) -> workers 0, verdict NO_GO_INSUFFICIENT_KV (never rounded up to 1)",
      _p0["workers"] == 0 and _p0["verdict"] == "NO_GO_INSUFFICIENT_KV", _p0)
_pm = NB5L.capacity_plan(214, 4096, {"b0": 104000}, 8192, 8192, 131072, 6, 1.0, conc_tokens=131072, conc_x=round(214 / 37, 2),
                         f_rule="measured")
check("lever b: f_rule 'measured' uses F from the server's max-concurrency line (214 / 5.78 - 32 -> F 5) below the placeholder 6",
      _pm["f_eff"] == 5 and _pm["f_rule"] == "measured", (_pm["f_log"], _pm["f_eff"]))
check("R2 workers rule max(1, floor(pool / (ceil(65536 / 4096) + F))): FP8 149 -> 6, 214 -> 9; BF16 78 -> 3, 112 -> 5; "
      "a tiny pool -> 1", [NB5L.workers_rule(x, 4096, 6) for x in (149, 214, 78, 112, 10)] == [6, 9, 3, 5, 1])
check("R2 launch max_num_seqs table: FP8 6 / 8 / 8 / 9 and BF16 3 / 4 / 4 / 5 at +0 / 0.5 / 0.75 / 1.0 GiB = the rule on the "
      "expected pools (block 15.75 / 29.75 MiB per card, F 6)",
      [NB5L.seqs_for("fp8", g) for g in (0.0, 0.5, 0.75, 1.0)] == [6, 8, 8, 9]
      and [NB5L.seqs_for("bf16", g) for g in (0.0, 0.5, 0.75, 1.0)] == [3, 4, 4, 5]
      and all(NB5L.workers_rule(149 + int(g * 1024 / 15.75), 4096, 6) == NB5L.seqs_for("fp8", g) for g in (0.0, 0.5, 0.75, 1.0))
      and all(NB5L.workers_rule(78 + int(g * 1024 / 29.75), 4096, 6) == NB5L.seqs_for("bf16", g) for g in (0.0, 0.5, 0.75, 1.0)))
check("FULL captures cover workers x 4: CAPS16 (to 64) covers 9 and 16; RC2's shipped list (to 32) does not cover 9",
      NB5L.captures_cover([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64], 9) and NB5L.captures_cover([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64], 16)
      and not NB5L.captures_cover([1, 2, 3, 4, 6, 8, 12, 16, 24, 32], 9))
check("lever a: pin ladder order (NB1 descending): RC2 + 1.0 / 0.75 / 0.5 / 0 GiB; a start step cuts the top",
      [x["gib"] for x in NB5L.pin_tries(2479838700)] == [1.0, 0.75, 0.5, 0.0]
      and NB5L.pin_tries(2479838700)[0]["kv_bytes"] == 2479838700 + 2 ** 30
      and [x["gib"] for x in NB5L.pin_tries(2464415744, 0.75)] == [0.75, 0.5, 0.0] and NB5L.pin_tries(1, 0.0) == [{"gib": 0.0, "kv_bytes": 1}])
_bf = [NB5L.boot_failure("LAUNCH_FAILED", "x", {}), NB5L.boot_failure("DONE", "torch.OutOfMemoryError: CUDA out of memory", {}),
       NB5L.boot_failure("DONE", "torch.OutOfMemoryError: CUDA out of memory", {}, phases_started=True),
       NB5L.boot_failure("DONE", "ok", {}),
       NB5L.boot_failure("LAUNCH_FAILED", "RuntimeError: MERLIN_SHARE_DRAFT_WORKSPACE=1 bad", {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"})]
check("ladder step-down rule: launch failure -> down; OOM line before the phases -> down; OOM after a phase started -> no "
      "relaunch; clean -> stop; a lever named in an error line -> that lever is the suspect",
      [x["failed"] for x in _bf] == [True, True, False, False, True] and _bf[4]["lever_suspect"] == ["MERLIN_SHARE_DRAFT_WORKSPACE"])
_ew = NB5L.equal_windows(1000.0, 1000.0 + 150 * 60, 20 * 60, 300.0, 2)
_ew2 = NB5L.equal_windows(1000.0, 1000.0 + 140 * 60, 30 * 60, 300.0, 2)
check("equal windows: (left - later boot est max(1.25 x, +5 min) - tail) / 2 (150 min left, 20-min boot -> 60 min each); a "
      "slower first boot (30 min, 140 min left) shortens BOTH (each 46.25 min)", _ew["window_sec"] == 3600.0
      and _ew["later_boot_est_sec"] == 1500.0 and _ew2["window_sec"] == round((140 * 60 - 37.5 * 60 - 300) / 2, 1), (_ew, _ew2))
check("capacity: an NB1 F above the log value wins (F = max(param, F_log)); a placeholder below it never lowers it",
      NB5L.capacity_plan(149, 4096, {"b0": 112000}, 8192, 8192, 131072, 9, conc_tokens=131072, conc_x=3.92)["f_eff"] == 9
      and NB5L.capacity_plan(149, 4096, {"b0": 112000}, 8192, 8192, 131072, 2, conc_tokens=131072, conc_x=3.92)["f_eff"] == 6)

_a8 = RES["setup"]["merlin_plans"]["merlin_fp8"]["argv_tail"]
_a16 = RES["setup"]["merlin_plans"]["merlin_bf16"]["argv_tail"]
_rb = {"fp8": NB5L.config_readback(FP8_LOG, _a8, "fp8", True), "bf16": NB5L.config_readback(BF16_LOG, _a16, "bf16", True),
       "fp8_as_bf16": NB5L.config_readback(FP8_LOG, _a8, "bf16", True), "fp8_mtp_off": NB5L.config_readback(FP8_LOG, _a8, "fp8", False),
       "seqs_16_intended": NB5L.config_readback(FP8_LOG, _a8 + ["--max-num-seqs", "16"], "fp8", True),
       "pin_intended": NB5L.config_readback(FP8_LOG, _a8 + ["--kv-cache-memory-bytes", "3553580524"], "fp8", True)}
check("C-11 read-back on the REAL RC2 logs + argv: FP8 and BF16 boots PASS (fp8_e4m3 / bfloat16, max_num_seqs 8 / 3, pin bytes, "
      "MTP K3 in the non-default line + the engine line); the FP8 log read as BF16, as MTP-off, with seqs 16 or a +1 GiB pin "
      "intended -> INCONCLUSIVE_CONFIG_MISMATCH", _rb["fp8"]["ok"] and _rb["bf16"]["ok"] and _rb["bf16"]["log"]["max_num_seqs"] == 3
      and _rb["fp8"]["checks"].get("kv_cache_memory_bytes") is True
      and all(not _rb[k]["ok"] and _rb[k]["verdict"] == "INCONCLUSIVE_CONFIG_MISMATCH" for k in ("fp8_as_bf16", "fp8_mtp_off",
                                                                                                 "seqs_16_intended", "pin_intended")),
      {k: v["checks"] for k, v in _rb.items()})
check("C-11: MTP off read-back: '--speculative-config None' last + a log without a speculative config -> PASS",
      NB5L.config_readback("non-default args: {'kv_cache_dtype': 'fp8_e4m3', 'max_num_seqs': 8}\nspeculative_config=None",
                           ["--max-num-seqs", "8", "--speculative-config", "{\"method\":\"mtp\"}", "--speculative-config", "None"],
                           "fp8", False)["ok"])

# ================================================================== 2. controller replay on the REAL RC2 windows / series
F = 6
proj = {}
for k, w in WINDOWS.items():
    if w.get("status") != "STEADY":
        continue
    L_ = w["prompt_tokens"] + int(w["gen_tokens"] / w["level"]) + 2000        # prompt + window tokens + admission margin
    proj[k] = (w["level"] * (NB5L.cdiv(L_, 4096) + F), round(w["kv_usage_max"] * 149))
check("real RC2 STEADY windows: the live block rule (ceil(L/4096) + 6) never exceeds the measured used blocks "
      "(the rule is not an overcount)", all(p <= m for p, m in proj.values()), proj)
check("real RC2 retention: at 40k@1 the gauge held >= 20 blocks more than the live request (retained finished turns)",
      proj["40k@1"][1] - proj["40k@1"][0] >= 20, proj["40k@1"])
gate_raw = [r for r in FP8_ROWS if r["running"] > 0 and r["kv_pct"] >= 95.0]
check("real RC2 series: Duck's raw-gauge gate (>= 0.95) would have held admissions in lines where the server ran fine "
      "(0 preemptions in the stats lines and every STEADY window)", len(gate_raw) > 0 and sum(r["preempt"] for r in FP8_ROWS) == 0
      and all(not w.get("preemptions") for w in WINDOWS.values() if w.get("status") == "STEADY"), len(gate_raw))
# reservation on the real window levels (request = one admission unit, its real context)
dec = {}
for k in ("8k@8", "40k@5", "90k@4", "116k@3", "116k@4"):
    w = WINDOWS[k]
    res = NB5L.Reservation(141)
    blocks = NB5L.cdiv(w["prompt_tokens"] + int(w["gen_tokens"] / w["level"]), 4096) + F
    dec[k] = sum(res.acquire(i, blocks, deadline=0) for i in range(w["level"]))
check("reservation on the real levels: 8k x 8, 40k x 5, 90k x 4, 116k x 3 all admitted; the 4th 116k request waits "
      "(4 x 36 = 144 > 141; it ran at 145/149 in RC2: conservative by 1 block, preemption-free)",
      dec == {"8k@8": 8, "40k@5": 5, "90k@4": 4, "116k@3": 3, "116k@4": 3}, dec)
# replay over the REAL request timeline (requests.jsonl: end time t, duration sec, prompt_len, tokens)
ev = []
for i, r in enumerate(REQS):
    if r.get("sec") is None:
        continue
    b = NB5L.cdiv(r["prompt_len"] + (r["n_tokens"] if "n_tokens" in r else len(r["tokens"])), 4096) + F
    ev += [(r["t"] - r["sec"], 1, i, b), (r["t"], 0, i, b)]
ev.sort()
res, waiting, granted, mx, delayed = NB5L.Reservation(141), [], set(), 0, 0
for t_, kind, i, b in ev:
    if kind == 1:
        if res.acquire(i, b, deadline=0):
            granted.add(i)
        else:
            waiting.append((i, b))
            delayed += 1
    else:
        if i in granted:
            res.release(i)
        else:
            waiting = [(j, bb) for j, bb in waiting if j != i]
        for j, bb in list(waiting):
            if res.acquire(j, bb, deadline=0):
                granted.add(j)
                waiting.remove((j, bb))
    mx = max(mx, res.used)
check("reservation replay on the REAL RC2 request timeline (47 requests): never above 141 blocks, no request left waiting "
      "(no deadlock), the 116k@4 excess delayed", mx <= 141 and not waiting and delayed >= 1, (mx, len(waiting), delayed))
rc_rows = {k: NB5L.running_check([{"inflight": w["level"], "running": w["running_min"]}, {"inflight": w["level"], "running": w["running_max"]}],
                                 w["level"])["verdict"] for k, w in WINDOWS.items() if "running_min" in w}
check("running == in-flight on the real RC2 windows: STEADY windows PASS, 90k@3 (running_min 1 < 3) FAILS",
      all(v == "PASS" for k, v in rc_rows.items() if WINDOWS[k]["status"] == "STEADY") and rc_rows["90k@3"] == "FAIL_RUNNING_NE_INFLIGHT", rc_rows)
check("running_check: no full sample -> INCONCLUSIVE (never PASS by default)",
      NB5L.running_check([{"inflight": 1, "running": 1}], 3)["verdict"] == "INCONCLUSIVE_NO_FULL_SAMPLE")
sm = NB5L.steady_mixed(FP8_ROWS, 8)
check("steady mixed on the real stats lines at running 8: p50 within 25 % of RC2's 8k@8 window (261.8 tok/s)",
      sm["gen_tok_s_p50"] and abs(sm["gen_tok_s_p50"] - 261.8) / 261.8 < 0.25, sm)
check("JIT kernels from the real log (RC2 lines 441, 551-555): the 5 kernels",
      NB5L.jit_kernels(FP8_LOG) == ["_fill_num_accepted_kernel", "_causal_conv1d_update_kernel", "_gdn_recoverssm_verify_kernel",
                                    "_ple_conv_kernel", "_ple_conv_writeback_kernel"], NB5L.jit_kernels(FP8_LOG))
check("deferred lines (Merlin cost-aware admission) in the real log: 10", NB5L.deferred_lines(FP8_LOG)["lines"] == 10,
      NB5L.deferred_lines(FP8_LOG))
t_start, t_mid = TEL[0]["host"], next(x["host"] for x in TEL if x["t"] > 2000)
g1 = NB5L.ram_gate(lambda: t_start, 73 * 1024, wait_sec=0)
g2 = NB5L.ram_gate(lambda: t_mid, 73 * 1024, wait_sec=0)
g3 = NB5L.ram_gate(lambda: {}, 73 * 1024, wait_sec=0)
check("RAM gate on the real telemetry: start (avail 176,845 MiB) passes, the serving sample (avail 2,023 MiB) fails, "
      "unreadable cgroup -> None (recorded, not a pass)", g1["ok"] is True and g2["ok"] is False and g3["ok"] is None,
      (t_start.get("avail_mib"), t_mid.get("avail_mib")))
check("page-cache drop: unsupported platform reports supported False; a real file is fadvised where supported",
      (lambda d: d["supported"] is False if not hasattr(os, "posix_fadvise") else d["files"] >= 1)(NB5L.drop_page_cache([HERE / "nb5_lib.py"])))

# ================================================================== 3. Reservation fault injection
r = NB5L.Reservation(141)
check("C-7: nothing held + a need above capacity -> DENIED at once (KV_NEED_ABOVE_POOL counted, nothing held); "
      "above_pool() says so before any acquire", r.above_pool(500) and r.acquire("x", 500, deadline=time.monotonic() + 5) is False
      and r.stats["above_pool_denials"] == 1 and not r.held and r.used == 0 and r.events[-1]["denied"] == "KV_NEED_ABOVE_POOL")
check("C-7: nothing held + a need == capacity -> the idle grant stays", r.acquire("y", 141, deadline=0) and r.used == 141)
r.release("y")
r = NB5L.Reservation(100)
r.acquire("a", 60, 0)
t0 = time.monotonic()
check("reservation: no room -> waits and is denied at the deadline (not before)",
      r.acquire("b", 60, time.monotonic() + 0.3) is False and time.monotonic() - t0 >= 0.29 and r.stats["denied"] == 1)
got = {}
th = threading.Thread(target=lambda: got.update(ok=r.acquire("c", 60, time.monotonic() + 5)))
th.start()
time.sleep(0.2)
r.release("a")
th.join(3)
check("reservation: a release wakes a waiter, which is granted", got.get("ok") is True and r.used == 60)
r.set_live_hold("server waiting 2, running 1, in-flight 3")
check("reservation: a live hold blocks a new grant that would fit", r.acquire("d", 10, time.monotonic() + 0.2) is False)
r.set_live_hold(None)
check("reservation: the hold cleared -> granted", r.acquire("d", 10, time.monotonic() + 0.2) is True)
r.correct(1, "preemption")
check("reservation: a preemption correction adds 1 block to later admissions only", r.f_extra == 1 and r.used == 70
      and r.acquire("e", 10, 0) and r.held["e"] == 11)
r.update("e", 3)
check("reservation live mode: update() moves an attempt's blocks to its live size", r.held["e"] == 4 and r.used == 74)
conc = NB5L.Reservation(141)
peak = {"v": 0, "n": 0}
lk = threading.Lock()


def worker(i):
    if conc.acquire(i, 50, time.monotonic() + 10):
        with lk:
            peak["n"] += 1
            peak["v"] = max(peak["v"], peak["n"])
        time.sleep(0.05)
        with lk:
            peak["n"] -= 1
        conc.release(i)


ths = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
[t.start() for t in ths]
[t.join(15) for t in ths]
check("reservation under 8 threads (50 blocks each, capacity 141): never more than 2 at once, all served", peak["v"] == 2
      and conc.stats["grants"] == 8 and conc.used == 0, (peak, conc.snapshot()))

# ================================================================== 4. commit turn variant C on the REAL tokenizer + template
from tokenizers import Tokenizer  # noqa: E402
TOK = Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
check("Flash-Next tokenizer / template = the pinned RC2 files (sha 0997f410 / c3cf9e34)",
      hashlib.sha256((TOKDIR / "tokenizer.json").read_bytes()).hexdigest().startswith("0997f410")
      and hashlib.sha256((TOKDIR / "chat_template.jinja").read_bytes()).hexdigest().startswith("c3cf9e34"))
TA_, TC_ = NB5L.tails(lambda s: TOK.encode(s, add_special_tokens=False).ids)
check("tails: '<think>\\n' = [248068, 198]; '<think>\\n\\n</think>\\n\\n' = [248068, 271, 248069, 271]",
      TA_ == [248068, 198] and TC_ == [248068, 271, 248069, 271], (TA_, TC_))
_argv = sys.argv
sys.argv = [str(CT_SCRIPT), str(TOKDIR)]
_out = io.StringIO()
_so = sys.stdout
sys.stdout = _out
try:
    CT = runpy.run_path(str(CT_SCRIPT), run_name="ct")
finally:
    sys.stdout, sys.argv = _so, _argv
rec = json.loads(CT_RESULT.read_text(encoding="utf-8"))
check("commit-turn CPU test reproduces its recorded result exactly (rows, header shas, cached tokens)",
      CT["res"] == rec, (CT["res"]["rows"], rec["rows"]))
check("variant_c_ids(render A) == the CPU test's variant C ids (token-level tail replacement = full re-tokenization)",
      NB5L.variant_c_ids(CT["A"], TA_, TC_) == CT["C"])
check("variant C keeps the cache prefix (12,248 tokens = 98.3 %) and the header; variant B (enable_thinking False) keeps 3",
      CT["common"](CT["cached"], NB5L.variant_c_ids(CT["A"], TA_, TC_)) == 12248 and CT["common"](CT["cached"], CT["B"]) == 3)
check("variant_c_ids refuses a render that does not end with '<think>\\n' (-> the chat path)",
      NB5L.variant_c_ids(CT["B"], TA_, TC_) is None)
# turn N+1 extends turn N (addendum 1.2) on the real template with Duck-shaped history (vLLM preprocessing)
from transformers import AutoTokenizer  # noqa: E402
HF = AutoTokenizer.from_pretrained(str(TOKDIR))
TPL = (TOKDIR / "chat_template.jinja").read_text(encoding="utf-8")
TOOLS = CT["TOOLS"]


def vrender(msgs):
    mm = []
    for m in msgs:
        m2 = {k: v for k, v in m.items() if k != "reasoning"}
        if m["role"] == "assistant":
            m2["reasoning_content"] = m.get("reasoning") or ""
        mm.append(m2)
    return TOK.encode(HF.apply_chat_template(mm, tools=TOOLS, tokenize=False, add_generation_prompt=True, chat_template=TPL,
                                             enable_thinking=True, reasoning_effort="xhigh"), add_special_tokens=False).ids


hist = [{"role": "system", "content": "ARC task facts ..."}, {"role": "user", "content": "First turn."}]
ok_chain = True
prev = vrender(hist)
for k in range(4):
    hist += [{"role": "assistant", "content": "", "reasoning": f"Step {k}: rows and colours.\n\nSecond paragraph.",
              "tool_calls": [{"type": "function", "function": {"name": "run_python", "arguments": {"code": f"print({k})"}}}]},
             {"role": "tool", "content": f"{k}"}]
    if k % 2:
        hist.append({"role": "user", "content": f"Continue (turn {k})."})
    cur = vrender(hist)
    ok_chain = ok_chain and cur[:len(prev)] == prev
    prev = cur
check("turn N+1's rendered prompt starts with turn N's full prompt (real template, reasoning kept, new user openers)", ok_chain)
check("the Flash-Next template refuses a non-leading system message (vLLM would 400): Duck's only system message is the "
      "leading one (tool_agent.py 6082 / 6704)", "System message must be at the beginning" in TPL)

# ---- revision 3 / PLAN v3.6: the per-request budget guard on REAL renders of long tool histories
def duck_hist(n_turns, tool_chars):
    h = [{"role": "system", "content": "ARC task facts ..."}, {"role": "user", "content": "First turn."}]
    for k in range(n_turns):
        h += [{"role": "assistant", "content": "", "reasoning": f"Step {k}: rows and colours.",
               "tool_calls": [{"type": "function", "function": {"name": "run_python", "arguments": {"code": f"print({k})"}}}]},
              {"role": "tool", "content": " ".join(str((k * 7 + j) % 10) for j in range(tool_chars // 2))}]
    return h


p_short = len(vrender(duck_hist(1, 2000))) + 2
p_mid = len(vrender(duck_hist(20, 3300))) + 2
p_long = len(vrender(duck_hist(36, 3300))) + 2
g1 = NB5L.context_guard(p_short, 112000 - 4096, 0, 112000, 4096, False)
g2 = NB5L.context_guard(p_mid, 112000 - 20000 - 4096, 20000, 112000, 4096, False)
g3 = NB5L.context_guard(p_long, 112000 - 10000 - 4096, 10000, 112000, 4096, False)
g4 = NB5L.context_guard(p_long, 4000, 108000, 112000, 4096, True)
check("budget guard, first turn (real render, %d tok): no clamp, the full 107,904 cap, no floor event" % p_short,
      not g1["clamped"] and g1["cap"] == 107904 and not g1["floor_event"], g1)
check("budget guard, 20 long tool results (real render, %d tok): cap = 131072 - prompt - 2048 (clamped, counted), "
      "20k + safe >= 60k -> no floor event" % p_mid, g2["clamped"] and g2["cap"] == 131072 - p_mid - 2048
      and g2["safe_output_budget"] == g2["cap"] and not g2["floor_event"] and p_mid + g2["cap"] <= 131072, (p_mid, g2))
check("budget guard, 36 long tool results (real render, %d tok): generated 10k + safe < 60k -> BUDGET_LIMITED, budget "
      "shrinks to generated + cap + reserve (Duck's own commit route)" % p_long,
      g3["budget_limited"] and g3["new_budget"] == 10000 + g3["cap"] + 4096 and p_long + g3["cap"] <= 131072, (p_long, g3))
check("budget guard in the commit zone: clamp only, never a floor event", not g4["floor_event"] and g4["cap"] <= 4000, g4)
g5 = NB5L.context_guard(131000, 5000, 0, 112000, 4096, False)
check("A-13: no room left -> safe 0, cap 0 (no 1-token request), no_room -> CONTEXT_EXHAUSTED, BUDGET_LIMITED",
      g5["safe_output_budget"] == 0 and g5["cap"] == 0 and g5["no_room"] and g5["verdict"] == "CONTEXT_EXHAUSTED"
      and g5["budget_limited"], g5)
_gs = {n: NB5L.context_guard(n, 4000, 100000, 104000, 4096, True) for n in (129000, 130000, 131071, 131072)}
check("A-13 (audit 3 cases, commit zone): 129k -> cap 44, 130k / 131,071 / 131,072 -> no room (no request)",
      _gs[129000]["cap"] == 131072 - 129000 - 2048 and not _gs[129000]["no_room"]
      and all(_gs[n]["no_room"] and _gs[n]["cap"] == 0 for n in (130000, 131071, 131072)), {n: g["cap"] for n, g in _gs.items()})
_g9 = NB5L.context_guard(30000, 99904, 20000, 104000, 4096, False, max_len=120384)
check("C-9: the guard clamps to the attempt's projected peak (120,384 for 8k + 104k + 8k): cap = 120,384 - 30,000 - 2,048",
      _g9["cap"] == 120384 - 30000 - 2048 and _g9["clamped"] and 30000 + _g9["cap"] <= 120384, _g9)
tl = NB5L.stat_timeline(FP8_LOG)
w_ = tl[-1][0] - tl[0][0] + 10
df = NB5L.duty_factor(FP8_LOG, w_, 1_000_000, 261.8)
check("duty factor on the real RC2 FP8 log: 318 timed lines, 10-s interval, productive 3,170 s of 3,190 s (0.994), "
      "pure decode 0.919, token-weighted = tokens / (rate x wall)", len(tl) == 318 and df["interval_sec_median"] == 10.0
      and df["productive_decode_sec"] == 3170.0 and df["duty_time"] == 0.9937 and df["duty_pure_decode_time"] == 0.9185
      and df["duty_tokens"] == round(1_000_000 / (261.8 * w_), 4), df)
check("duty factor: no lines -> None (never a made-up number)", NB5L.duty_factor("", 100)["duty_time"] is None)
vo = NB5L.variant_outcomes({"tA": [{"branch": 0, "b_events": {"commit_at_tokens": 900},
                                    "submissions": [{"accepted": True, "test_outputs": [[[1]]]}]},
                                   {"branch": 1, "b_events": {"commit_at_tokens": None}}],
                            "tB": [{"branch": 0, "b_events": {"commit_at_tokens": 800},
                                    "checked": [{"accepted": False, "test_outputs": [["x"]]}]}]},
                           {"tA": [[[1]]], "tB": [[[2]]]}, "split")
check("paired split outcomes: per variant attempts with a commit turn, exact-match, non-grid candidates",
      vo["A"]["attempts"] + vo["C"]["attempts"] == 2 and vo["A"]["exact"] + vo["C"]["exact"] == 1
      and vo["A"]["invalid_grid_candidates"] + vo["C"]["invalid_grid_candidates"] == 1, vo)
rw = NB5L.Reservation(10)
rw.acquire("a", 10, 0)
th_ = threading.Thread(target=lambda: rw.acquire("b", 5, time.monotonic() + 1.0))
th_.start()
time.sleep(0.3)
w_now = rw.waiting
th_.join(3)
check("reservation counts the attempts waiting right now (the admission-interaction metric)", w_now == 1 and rw.waiting == 0)
check("future admission hook is a placeholder (returns None: projected decides)", NB5L.future_admission_blocks({}, {}) is None)

# ================================================================== 5. Duck's own parser on variant C replies (real package)
_dd = Path(tempfile.mkdtemp(prefix="nb5_duck_"))
zipfile.ZipFile(PAYLOAD).extractall(_dd)
sys.path[:0] = [str(_dd), str(_dd / "vendor")]
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
from inference.agent import tool_agent as TA  # noqa: E402
m, n, leak = NB5L.parse_c_reply(TA, "Committing.\n<tool_call>\n<function=python>\n<parameter=code>\ndef transform(g):\n    return g\nprint(check(transform))\n</parameter>\n</function>\n</tool_call>")
check("parse_c_reply: one python call, JSON arguments with the full code, content without markup, no think markup",
      n == 1 and json.loads(m["tool_calls"][0]["function"]["arguments"])["code"].startswith("def transform(g):")
      and m["content"] == "Committing." and not leak, m)
m2, n2, _ = NB5L.parse_c_reply(TA, "Committing.\n<tool_call>\n<function=python>\n<parameter=code>\ndef transform(g):\n    ret")
check("parse_c_reply: a call cut at max_tokens -> no call, the markup stays in content (Duck's R5-2 cut path)",
      n2 == 0 and "<tool_call>" in m2["content"])
_, _, leak3 = NB5L.parse_c_reply(TA, "oops </think> then text")
check("parse_c_reply: think markup in a variant-C reply is flagged", leak3 is True)


class _M:
    provider, model_id = "vllm", "q27"


class _Ag:
    _model = _M()
    _max_output_tokens = 4000
    _arc2_b = True
    _arc2_b_commit_sent = True

    def _harness_template_kwargs(self):
        return {"reasoning_effort": "xhigh"}


pl = NB5L.chat_payload(TA, _Ag(), [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                       {"tools": CT["TOOLS"], "max_tokens": 3000})
check("chat_payload = Duck's request: thinking ON, xhigh, the solver's own sampling (T / top_p / top_k), tools + auto",
      pl["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"} and pl["max_tokens"] == 3000
      and "temperature" in pl and "top_p" in pl and pl.get("top_k") and pl.get("tool_choice") == "auto", pl)
r_, info = NB5L.commit_c_call(TA, _Ag(), [{"role": "user", "content": "u"}], {"enable_thinking": False}, "http://127.0.0.1:9/v1",
                              TA_, TC_, 5)
check("commit_c_call never sends a thinking-off commit (enable_thinking False -> chat path)", r_ is None
      and "enable_thinking" in info["why"])
r_, info = NB5L.commit_c_call(TA, _Ag(), [{"role": "user", "content": "u"}], {}, "http://127.0.0.1:9/v1", TA_, TC_, 2)
check("commit_c_call: /tokenize unreachable -> None + reason (the chat path runs; never a crash)", r_ is None
      and "tokenize failed" in info["why"], info)
keys = [NB5L.commit_variant("split", type("A", (), {"arc2_task_id": f"t{i}", "arc2_branch": i % 2})()) for i in range(40)]
check("split variant: stable per (task, branch), both A and C occur", set(keys) == {"A", "C"} and
      NB5L.commit_variant("split", type("A", (), {"arc2_task_id": "t3", "arc2_branch": 1})()) == keys[3])

# ================================================================== 5b. the request dispatcher (A-13 / A-14 / C-9) executed for real
import types as _types  # noqa: E402
_nbc = ["".join(c["source"]) for c in json.loads(NB.read_bytes())["cells"] if c["cell_type"] == "code"]
_PM = "# ==================================================================== NB5 (E25): commit-turn variant C"
_patch = _nbc[2][_nbc[2].index(_PM):]


def _patch_ns(tok_fn, local=None, peak=None):
    calls = {"base": 0, "tok": 0, "max_tokens": None}
    L2 = _types.SimpleNamespace(**{k: getattr(NB5L, k) for k in dir(NB5L) if not k.startswith("__")})

    def _tok(url, payload, timeout=120):
        calls["tok"] += 1
        return tok_fn()
    L2.tokenize_chat = _tok
    L2.chat_payload = lambda TA, ag, m, kw: {"model": "q27", "messages": m, "chat_template_kwargs": {}}

    class _BS:
        def run(self, pause_at_tokens=None, tag=None):
            return "done"

    def _base(self, messages, **kw):
        calls["base"] += 1
        calls["max_tokens"] = kw.get("max_tokens")
        return _types.SimpleNamespace(message={"content": "x", "tool_calls": []}, finish_reason="stop", usage={})
    ns = {"_E21_CC_ORIG": _base, "E12A": _types.SimpleNamespace(BranchSession=_BS), "NB5L": L2,
          "_E12_TA": _types.SimpleNamespace(_ChatCompletionResult=lambda **k: _types.SimpleNamespace(**k)),
          "SGL_URL": "http://127.0.0.1:9", "MAX_LEN": 131072, "NB5_THINK_FLOOR": 60000, "NB5_CTX_MARGIN": 2048,
          "NB5_RESERVE": 4096, "NB5_ADMIT_MODE": "slots", "E12_R6": {"E12_R6_MIN_ADMIT_SEC": 5}, "time": time, "json": json,
          "print": lambda *a, **k: None}
    exec(compile(_patch, "<nb5 cell3 patch>", "exec"), ns)
    ns["NB5_BLOCKS"].update(peak_b0=peak, peak_b1=peak)
    if local:
        ns["NB5_LOCAL_TOK"].update(local)
    return ns, calls


class _AgQ:
    arc2_task_id, arc2_branch, generated_tokens = "tQ", 0, 50000
    _max_output_tokens, _arc2_b, _arc2_b_budget, _arc2_b_reserve, _arc2_b_commit_sent, _timeout = 99904, True, 104000, 4096, False, 60

    def _arc2_b_in_commit(self):
        return False

    def _estimate_request_input_tokens(self, m, tools=None):
        return 1234


_msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
_ns, _c = _patch_ns(lambda: list(range(131072)))
_ag = _AgQ()
_r = _ns["_nb5_cc"](_ag, _msgs)
_tr = _ns["NB5_TRACE"].summary()
check("A-13 (executed dispatcher): a 131,072-token prompt -> 0 calls to the server path, a synthetic 0-token reply, verdict "
      "CONTEXT_EXHAUSTED, Duck's own end route armed (budget = generated, commit flag), counted apart from the requests",
      _c["base"] == 0 and _c["tok"] == 1 and _r.served_by == "nb5_no_request" and _r.usage["completion_tokens"] == 0
      and _ag._nb5_verdict == "CONTEXT_EXHAUSTED" and _ag._arc2_b_budget == 50000 and _ag._arc2_b_commit_sent is True
      and _ns["NB5_GUARD"]["no_request"] == {"CONTEXT_EXHAUSTED": 1} and _tr["requests"] == 0
      and _tr["no_request"] == {"CONTEXT_EXHAUSTED": 1}, (_c, _tr.get("no_request")))


def _tok_fail():
    raise ConnectionError("tokenize down")


_ns, _c = _patch_ns(_tok_fail)
_ag = _AgQ()
_r = _ns["_nb5_cc"](_ag, _msgs)
_e = _ns["NB5_GUARD"]["estimates"][-1]
check("A-14: /tokenize fails 3 times (2 retries) and the local tokenizer is NOT verified -> no request, "
      "INCONCLUSIVE_CONTEXT_LENGTH, the estimate (Duck's 1,234) and every try recorded",
      _c["tok"] == 3 and _c["base"] == 0 and _ag._nb5_verdict == "INCONCLUSIVE_CONTEXT_LENGTH" and _e["duck_estimate"] == 1234
      and len(_e["tries"]) == 3 and _ns["NB5_GUARD"]["tokenize_retries"] == 2, (_c, _e))
_hf = _types.SimpleNamespace(apply_chat_template=lambda msgs, **k: "a" * 5000)
_ns, _c = _patch_ns(_tok_fail, local={"verified": True, "hf": _hf, "template": "", "encode": lambda s: list(range(len(s))),
                                      "default_kwargs": {}})
_ag = _AgQ()
_r = _ns["_nb5_cc"](_ag, _msgs)
check("A-14: /tokenize fails 3 times, the hash-verified local tokenizer counts 5,000 -> the request is sent (local count "
      "+ 2 + margin, recorded)", _c["base"] == 1 and _ns["NB5_GUARD"]["local_tokenizer"] == 1
      and _ns["NB5_GUARD"]["estimates"][-1]["local_tokens"] == 5000 and not getattr(_ag, "_nb5_verdict", None), _c)
_ns, _c = _patch_ns(lambda: list(range(30000)), peak=120384)
_ag = _AgQ()
_ns["_nb5_cc"](_ag, _msgs)
check("C-9 (executed dispatcher): a 30,000-token prompt with the attempt peak 120,384 -> max_tokens = 120,384 - 30,002 - 2,048 "
      "(the reservation is a true upper bound)", _c["base"] == 1 and _c["max_tokens"] == 120384 - 30002 - 2048, _c)
check("A-14 local render = the vLLM-preprocessed template render on the REAL Flash-Next tokenizer + template (Duck-shaped history "
      "with reasoning and tool calls)", NB5L.local_chat_ids({"messages": duck_hist(3, 500), "tools": TOOLS}, HF, TPL,
                                                            lambda s_: TOK.encode(s_, add_special_tokens=False).ids,
                                                            {"enable_thinking": True, "reasoning_effort": "xhigh"})
      == vrender(duck_hist(3, 500)))
_tn = NB5L.Tracer()
_tdn = Path(tempfile.mkdtemp(prefix="nb5_nr_"))
_tn.new_phase("T", _tdn)
_tn.note_no_request(type("A", (), {"arc2_task_id": "t1", "arc2_branch": 0, "generated_tokens": 7})(), "CONTEXT_EXHAUSTED", {"x": 1})
_sn = _tn.summary()
_rown = json.loads(open(_tdn / "t1_b0" / "nb5_requests.jsonl", encoding="utf-8").readline())
check("tracer: a not-sent request is a 'no_request' trace row + counter, never a request", _sn["requests"] == 0
      and _sn["no_request"] == {"CONTEXT_EXHAUSTED": 1} and _sn["by_attempt"]["t1|0"]["no_request"] == "CONTEXT_EXHAUSTED"
      and _rown["kind"] == "no_request")

# ================================================================== 5c. the paired quality report
_gold = {"tA": [[[1]]], "tB": [[[2]], [[3]]], "tC": [[[4]]]}
_recs = {"tA": [{"branch": 0, "t0": 100.0, "t1": 200.0, "generated_tokens": 9000, "accepted": True,
                 "submissions": [{"accepted": True, "test_outputs": [[[1]]]}], "b_events": {"commit_at_tokens": None}},
                {"branch": 1, "t0": 210.0, "t1": 1990.0, "generated_tokens": 40000, "submissions": []}],
         "tB": [{"branch": 0, "t0": 100.0, "t1": 900.0, "generated_tokens": 104000, "b_budget": 104000, "submissions": [],
                 "b_events": {"commit_at_tokens": 100000, "commit_tool_call": True}, "nb5_verdict": "BUDGET_LIMITED"}]}
_units = [{"task": "tA", "query": 0, "hit": True, "pass1": True, "pass2": False},
          {"task": "tB", "query": 0, "hit": True, "pass1": False, "pass2": True}, {"task": "tB", "query": 1, "hit": False},
          {"task": "tC", "query": 0, "hit": False}]
_qa = NB5L.qab_tasks(_recs, _gold, _units, {"tA": "solid", "tB": "sometimes", "tC": "never"}, ["tA", "tB", "tC"],
                     {"b0": 104000, "b1": 104000}, window_start_epoch=50.0, window_end_epoch=2000.0,
                     by_attempt={"tB|0": {"requests": 9, "commit_requests": 1, "commit_with_tool_call": 1}})
check("qab_tasks: per attempt ACCEPTED (exact), INCOMPLETE_WINDOW (running at the window end), BUDGET_END (+ commit with a call), "
      "NOT_STARTED; pass@2 solved = every query hit (tB 1/2 -> not solved); finished only when every attempt finished; the order",
      list(_qa) == ["tA", "tB", "tC"] and [a["outcome"] for a in _qa["tA"]["attempts"]] == ["ACCEPTED", "INCOMPLETE_WINDOW"]
      and _qa["tA"]["attempts"][0]["exact_accepted"] and _qa["tA"]["solved"] and not _qa["tA"]["finished"]
      and [a["outcome"] for a in _qa["tB"]["attempts"]] == ["BUDGET_END", "NOT_STARTED"] and not _qa["tB"]["solved"]
      and _qa["tB"]["query_hits"] == 1 and _qa["tB"]["attempts"][0]["commit"]["commit_with_tool_call"] == 1
      and _qa["tB"]["attempts"][0]["budget_limited"] and _qa["tC"]["attempts_started"] == 0
      and _qa["tA"]["attempts"][1]["end_offset_sec"] == 1940.0, json.dumps(_qa)[:1500])
_qs = NB5L.qab_summary(_qa)
check("qab_summary per group + all (tasks, solved, task-norm %, incomplete / not-started attempts)",
      _qs["all"]["tasks"] == 3 and _qs["all"]["solved"] == 1 and _qs["all"]["task_norm_pct"] == round(100 * 1.5 / 3, 2)
      and _qs["all"]["attempts_incomplete"] == 1 and _qs["all"]["attempts_not_started"] == 3 and _qs["solid"]["solved"] == 1, _qs)
_qb = json.loads(json.dumps(_qa))
_qb["tA"]["solved"], _qb["tC"]["solved"] = False, True
for _x in _qb.values():
    _x["finished"] = True
_qa2 = json.loads(json.dumps(_qa))
_qa2["tB"]["finished"] = _qa2["tC"]["finished"] = True
_pr = NB5L.qab_pair(_qa2, _qb)
check("qab_pair: per group + all, FP8-only / BF16-only discordant pairs (tA, tC) + exact McNemar p; finished-in-both counted apart",
      _pr["all"]["all"]["FP8_only"] == 1 and _pr["all"]["all"]["BF16_only"] == 1 and _pr["all"]["all"]["discordant"] == 2
      and _pr["all"]["all"]["discordant_tasks"] == ["tA", "tC"] and _pr["all"]["all"]["mcnemar_p"] == 1.0
      and _pr["finished_in_both_tasks"] == ["tB", "tC"] and _pr["finished_in_both"]["all"]["tasks"] == 2
      and NB5L.mcnemar_exact_p(0, 3) == 0.25 and NB5L.mcnemar_exact_p(0, 0) == 1.0, json.dumps(_pr)[:1200])
check("offline pair from one R2 report (both phases inside)", NB5L.qab_pair_reports(
    [{"quality_ab": {"per_config": {"Q_FP8_K3": {"tasks": _qa}, "Q_BF16_K3": {"tasks": _qb}}}}])["all"]["all"]["discordant"] == 2)
# ---- Q-1 (R2 audits 1 + 2): fail-closed attempt ends (Duck 9e4b8928 record shapes: status / note / accepted / b_events)
_WE = 2000.0
_mid = {"branch": 0, "status": "done", "t0": 100.0, "t1": 700.0, "generated_tokens": 30000, "b_budget": 104000, "note": "",
        "submissions": [], "checked": [], "b_events": {}}
_q1err = {"cuda": dict(_mid, note="RuntimeError: CUDA error: an illegal memory access was encountered"),
          "http": dict(_mid, note="HTTPError: HTTP Error 500: Internal Server Error"),
          "timeout": dict(_mid, note="ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=30000): Read timed out. (read timeout=900)"),
          "analyzer": dict(_mid, note="analyzer failed 11 times consecutively", failures=11),
          "server_verdict": dict(_mid, nb5_verdict="SERVER_ERROR"),
          "kv_need": dict(_mid, nb5_verdict="KV_NEED_ABOVE_POOL", note="nb5: KV_NEED_ABOVE_POOL (projected need above the whole pool)"),
          "ctx_unknown": dict(_mid, nb5_verdict="INCONCLUSIVE_CONTEXT_LENGTH"),
          "check_accepted_then_error": dict(_mid, checked=[{"accepted": True, "test_outputs": [[[1]]]}],
                                            note="ConnectionResetError: [Errno 104] Connection reset by peer"),
          "error_at_budget": dict(_mid, generated_tokens=103900, note="RuntimeError: CUDA error: device-side assert triggered")}
_q1e = {k: NB5L.attempt_end(r, 104000, _WE) for k, r in _q1err.items()}
check("Q-1: CUDA error / HTTP error / timeout / repeated analyzer failures / SERVER_ERROR / KV denial / no exact context count / "
      "a check()-accepted candidate then an error / an error at the budget -> FAILED_OR_INTERRUPTED (never finished, never "
      "ENDED_OTHER), classified", all(o == "FAILED_OR_INTERRUPTED" and o not in NB5L.FINISHED_OUTCOMES for o, _ in _q1e.values())
      and _q1e["cuda"][1] == "CUDA_ERROR" and _q1e["http"][1] == "HTTP_OR_SERVER_ERROR" and _q1e["timeout"][1] == "TIMEOUT"
      and _q1e["analyzer"][1] == "REPEATED_ANALYZER_FAILURES" and "HTTP_OR_SERVER_ERROR" in _q1e["server_verdict"][1]
      and "KV_NEED_ABOVE_POOL" in _q1e["kv_need"][1] and NB5L.attempt_outcome(_q1err["http"], 104000, _WE) == "FAILED_OR_INTERRUPTED",
      _q1e)
_q1ok = {"accepted": dict(_mid, accepted=True, submissions=[{"accepted": True, "weak": True, "test_outputs": [[[1]]]}]),
         "budget_spent": dict(_mid, generated_tokens=103500),
         "tiny_rest": dict(_mid, generated_tokens=103300, b_events={"tiny_request_skipped": 1}),
         "floor": dict(_mid, generated_tokens=80000, nb5_verdict="BUDGET_LIMITED"),
         "context_exhausted": dict(_mid, generated_tokens=90000, nb5_verdict="CONTEXT_EXHAUSTED"),
         "accepted_at_window_end": dict(_mid, accepted=True, t1=1995.0)}
_q1o = {k: NB5L.attempt_end(r, 104000, _WE) for k, r in _q1ok.items()}
check("Q-1: accepted submit (incl. a W1 weak commit-zone accept) -> ACCEPTED; budget spent / commit done + < 768 left / NB5 floor "
      "BUDGET_LIMITED / CONTEXT_EXHAUSTED (whitelisted budget end) -> BUDGET_END; both finished",
      [o for o, _ in _q1o.values()] == ["ACCEPTED", "BUDGET_END", "BUDGET_END", "BUDGET_END", "BUDGET_END", "ACCEPTED"]
      and all(o in NB5L.FINISHED_OUTCOMES for o, _ in _q1o.values()) and _q1o["context_exhausted"][1] == "CONTEXT_EXHAUSTED", _q1o)
_q1w = {"paused": dict(_mid, status="paused", t1=1200.0), "running_at_end": dict(_mid, t1=1990.0),
        "no_evidence": dict(_mid), "no_status_no_budget": {"branch": 0, "t0": 100.0, "t1": 500.0, "generated_tokens": 0}}
_q1x = {k: NB5L.attempt_end(r, 104000, _WE) for k, r in _q1w.items()}
check("Q-1: unfinished window (paused at the end / running at the window end) -> INCOMPLETE_WINDOW; ended mid-window with no "
      "note, no accept and no budget evidence -> FAILED_OR_INTERRUPTED (no_terminal_evidence)",
      _q1x["paused"] == ("INCOMPLETE_WINDOW", "paused_at_window_end") and _q1x["running_at_end"][0] == "INCOMPLETE_WINDOW"
      and _q1x["no_evidence"] == ("FAILED_OR_INTERRUPTED", "no_terminal_evidence")
      and _q1x["no_status_no_budget"][0] == "FAILED_OR_INTERRUPTED", _q1x)
_se = lambda b: dict(_mid, branch=b, nb5_verdict="SERVER_ERROR", note="HTTPError: HTTP Error 503: SERVER_ERROR")
_fin = lambda b: dict(_mid, branch=b, generated_tokens=104000)
_gold2 = {"tS": [[[1]]], "tF": [[[2]]]}
_un2 = [{"task": "tS", "query": 0, "hit": False}, {"task": "tF", "query": 0, "hit": False}]
_q1a = NB5L.qab_tasks({"tS": [_se(0), _se(1)], "tF": [_fin(0), _fin(1)]}, _gold2, _un2, {"tS": "solid", "tF": "never"},
                      ["tS", "tF"], {"b0": 104000, "b1": 104000}, 50.0, _WE)
_q1b = NB5L.qab_tasks({"tS": [_fin(0), _fin(1)], "tF": [_fin(0), _fin(1)]}, _gold2, _un2, {"tS": "solid", "tF": "never"},
                      ["tS", "tF"], {"b0": 104000, "b1": 104000}, 50.0, _WE)
_q1p = NB5L.qab_pair(_q1a, _q1b)
check("Q-1 (the audits' reproduction): a task with two SERVER_ERROR attempts -> both FAILED_OR_INTERRUPTED, task NOT finished, "
      "NOT in finished_in_both; the budget-end task is; qab_summary counts the failed attempts",
      [a["outcome"] for a in _q1a["tS"]["attempts"]] == ["FAILED_OR_INTERRUPTED"] * 2 and not _q1a["tS"]["finished"]
      and _q1a["tS"]["unfinished_outcomes"] == ["FAILED_OR_INTERRUPTED"] * 2 and _q1a["tF"]["finished"]
      and _q1p["finished_in_both_tasks"] == ["tF"] and _q1p["finished_in_both"]["all"]["tasks"] == 1
      and _q1p["all"]["all"]["tasks"] == 2 and NB5L.qab_summary(_q1a)["all"]["attempts_failed"] == 2
      and NB5L.qab_summary(_q1a)["all"]["outcomes"] == {"FAILED_OR_INTERRUPTED": 2, "BUDGET_END": 2}, json.dumps(_q1p)[:1200])
check("Q-5 qab_pair regression + power: the same discordant pairs / McNemar p as before, plus paired n, the minimum attainable p "
      "for that n and for that discordant count, and a reading that never says parity",
      _pr["all"]["all"]["discordant"] == 2 and _pr["all"]["all"]["mcnemar_p"] == 1.0 and _pr["all"]["all"]["paired_n"] == 3
      and _pr["all"]["all"]["min_attainable_p_paired_n"] == 0.25 and _pr["all"]["all"]["min_attainable_p_at_discordant"] == 0.5
      and _pr["all"]["all"]["reading"] == "NOT_SIGNIFICANT_NOT_PARITY_UNDERPOWERED"
      and _pr["finished_in_both_tasks"] == ["tB", "tC"] and "parity" in _pr["note"]
      and NB5L.qab_pair({f"t{i}": {"solved": True} for i in range(6)}, {f"t{i}": {"solved": False} for i in range(6)})
      ["all"]["all"]["reading"] == "DIFFERENCE_P_LT_0_05", json.dumps(_pr["all"]["all"]))
_pcw = lambda a, b, sa="DONE", sb="DONE", ua=False, ub=False: {
    "Q_FP8_K3": {"status": sa, "window": {"actual_sec": a}, "window_plan": {"planned_sec": 3800.0, "unequal": ua}},
    "Q_BF16_K3": {"status": sb, "window": {"actual_sec": b}, "window_plan": {"planned_sec": 3800.0, "unequal": ub}}}
_wm = NB5L.qab_windows_check({"Q_FP8_K3": _pcw(3800.0, 3800.0)["Q_FP8_K3"]})
check("Q-2: a missing phase (BF16 never ran) -> INCONCLUSIVE_UNEQUAL_WINDOWS (the old set-of-one rule said equal)",
      not _wm["equal"] and _wm["verdict"] == "INCONCLUSIVE_UNEQUAL_WINDOWS" and "Q_BF16_K3: phase missing" in _wm["reasons"], _wm)
_wu = {"delta_5s": NB5L.qab_windows_check(_pcw(3800.0, 3805.0)), "flag": NB5L.qab_windows_check(_pcw(3800.0, 3800.0, ub=True)),
       "not_done": NB5L.qab_windows_check(_pcw(3800.0, 3800.0, sb="ERROR")),
       "none": NB5L.qab_windows_check(_pcw(3800.0, None)), "nan": NB5L.qab_windows_check(_pcw(float("nan"), 3800.0))}
check("Q-2: unequal -> INCONCLUSIVE_UNEQUAL_WINDOWS: |delta| 5 s > 2 s, window_plan.unequal, a phase not DONE, a missing or NaN "
      "actual window", all(not v["equal"] and v["verdict"] == "INCONCLUSIVE_UNEQUAL_WINDOWS" for v in _wu.values())
      and _wu["delta_5s"]["delta_sec"] == 5.0, _wu)
_we = NB5L.qab_windows_check(_pcw(3800.0, 3801.5))
check("Q-2: equal = both DONE, numeric, |delta| 1.5 s <= the predeclared 2 s, neither unequal -> EQUAL (tolerance recorded)",
      _we["equal"] and _we["verdict"] == "EQUAL" and _we["tolerance_sec"] == 2.0 and _we["delta_sec"] == 1.5 and not _we["reasons"], _we)
_cap = lambda w, pool, pin, pre: {"capacity_levers": {"pin_gib": pin, "pool_blocks": pool, "workers": w, "f_eff": 6,
                                                      "max_num_seqs_launch": w, "lever_env": {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"},
                                                      "captures": [1, 2, 4, 8, 16, 64]},
                                  "speed": {"preemptions": pre, "speed_status": "PREEMPTED" if pre else "CLEAN"}}
_pcr = _pcw(3800.0, 3800.0)
_pcr["Q_FP8_K3"].update(_cap(9, 514, 1.0, 0), tasks=_q1a)
_pcr["Q_BF16_K3"].update(_cap(5, 269, 0.75, 2), tasks=_q1b)
_rp = NB5L.qab_report(_pcr)
_rpm = NB5L.qab_report({"Q_FP8_K3": _pcr["Q_FP8_K3"]})
_dc = _rp["deployment_comparison"]
check("Q-3 / Q-5 report: deployment_comparison MEASURED with three separate results (all tasks in the fixed window, finished in "
      "both, the missing / incomplete / failed matrix with outcome counts per phase); precision_comparison UNVERIFIED (workers "
      "9 vs 5, pool, pin); phase_diff lists pin / workers / seqs / pool / preemptions, captures + lever env the same; a missing "
      "phase -> INCONCLUSIVE_UNEQUAL_WINDOWS + phase_diff.missing_phase",
      _dc["status"] == "MEASURED" and _dc["label"] == "deployment_comparison" and _dc["all_tasks_fixed_window"]["all"]["tasks"] == 2
      and _dc["finished_in_both_tasks"] == ["tF"]
      and _dc["missing_incomplete_failed"]["phases"]["Q_FP8_K3"]["outcomes"] == {"FAILED_OR_INTERRUPTED": 2, "BUDGET_END": 2}
      and _dc["missing_incomplete_failed"]["phases"]["Q_FP8_K3"]["tasks_not_finished"] == {"tS": ["FAILED_OR_INTERRUPTED"] * 2}
      and _dc["missing_incomplete_failed"]["per_task"]["tS"]["Q_BF16_K3"] == ["BUDGET_END"] * 2
      and _rp["precision_comparison"]["status"] == "UNVERIFIED" and "workers 9 vs 5" in _rp["precision_comparison"]["reason"]
      and {"capacity_levers.pin_gib", "capacity_levers.workers", "capacity_levers.max_num_seqs_launch", "capacity_levers.pool_blocks",
           "speed.preemptions", "speed.speed_status"} <= set(_rp["phase_diff"]["differences"])
      and _rp["phase_diff"]["differences"]["capacity_levers.workers"] == {"FP8": 9, "BF16": 5}
      and {"capacity_levers.captures", "capacity_levers.lever_env", "capacity_levers.f_eff"} <= set(_rp["phase_diff"]["same"])
      and _rp["windows_equal"] is True and _rpm["deployment_comparison"]["status"] == "INCONCLUSIVE_UNEQUAL_WINDOWS"
      and _rpm["windows_equal"] is False and "skipped" in _rpm["paired"] and _rpm["phase_diff"]["missing_phase"] == ["Q_BF16_K3"]
      and _rpm["deployment_comparison"]["missing_incomplete_failed"]["per_task"]["tS"]["Q_BF16_K3"] == "MISSING",
      json.dumps({k: _rp[k] for k in ("windows_check", "precision_comparison", "phase_diff")}, default=str)[:1500])

# ================================================================== 6. tracer / bands / deltas / argv diff
td = Path(tempfile.mkdtemp(prefix="nb5_tr_"))
tr = NB5L.Tracer()
tr.new_phase("T", td)


class _R:
    def __init__(self, msg, fin, u):
        self.message, self.finish_reason, self.usage = msg, fin, u


ag = type("Ag", (), {"arc2_task_id": "t1", "arc2_branch": 0, "generated_tokens": 10})()
tr.record(ag, "chat", {"max_tokens": 9}, _R({"reasoning": "R" * 50, "content": None, "tool_calls": [{"function": {"name": "python", "arguments": "{}"}}]},
                                           "tool_calls", {"prompt_tokens": 5000, "completion_tokens": 900,
                                                          "prompt_tokens_details": {"cached_tokens": 0}}), 0, 1,
          messages=[{"role": "system"}, {"role": "user"}])
tr.record(ag, "chat", {}, _R({"reasoning": "R", "content": None, "tool_calls": []}, "length", {"prompt_tokens": 5900, "completion_tokens": 9,
                                                                                            "prompt_tokens_details": {"cached_tokens": 4992}}), 1, 2,
          messages=[{"role": "system"}, {"role": "user"}, {"role": "system"}])
tr.record(ag, "commit_C", {}, _R({"content": "x", "tool_calls": [{"function": {"name": "python", "arguments": "{}"}}]}, "tool_calls",
                                 {"prompt_tokens": 6000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 5888}}), 2, 3)
tr.record(ag, "commit_A", {}, _R({"content": "", "tool_calls": []}, "stop", {"prompt_tokens": 6100, "completion_tokens": 4000}), 3, 4)
s = tr.summary()
check("tracer: natural tool call / cut 'length' / natural stop classified; retained-session hit share; non-leading system counted",
      s["natural_tool_call"] == 2 and s["cut_length"] == 1 and s["open_think_other"] == 1 and s["later_with_cache_hit"] == 2
      and s["non_leading_system_requests"] == 1, s)
check("tracer: commit counters per variant (C: 1 with a call, cached 5,888; A: 1 without) and commit turns without a call",
      s["commit_by_variant"]["C"]["with_tool_call"] == 1 and s["commit_by_variant"]["C"]["cached"] == 5888
      and s["commit_by_variant"]["A"]["with_tool_call"] == 0 and s["commit_turns_without_tool_call"] == 0, s["commit_by_variant"])
lines = [json.loads(l) for l in open(td / "t1_b0" / "nb5_requests.jsonl", encoding="utf-8")]
check("tracer: one JSONL row per request with the FULL reasoning", len(lines) == 4 and lines[0]["reasoning"] == "R" * 50)
gold = {"t1": [[[1]]]}
recs = {"t1": [{"generated_tokens": 40000, "accepted": True, "t0": 100.0,
                "submissions": [{"accepted": True, "test_outputs": [[[1]]], "t": 160.0}]},
               {"generated_tokens": 100000, "accepted": False, "submissions": []}]}
b = NB5L.solve_bands(recs, gold)
check("solve bands: the accepted + correct attempt in <=45k, the budget attempt in <=120k; cumulative shares",
      b["bands"]["<=45k"]["correct"] == 1 and b["bands"]["<=120k"]["attempts"] == 1 and b["cumulative"]["<=30k"]["accepted_share"] == 0.0
      and b["cumulative"]["<=45k"]["correct_share"] == 1.0, b)
bw = NB5L.solve_bands({"t1": [{"generated_tokens": 5800, "accepted": False,
                                "submissions": [{"accepted": True, "weak": True, "test_outputs": [[[1]]]}]}]}, gold)
check("solve bands count a weak (commit-zone) accept without the attempt-level flag (E15B weak commit)",
      bw["bands"]["<=30k"]["accepted"] == 1 and bw["bands"]["<=30k"]["weak_accepted"] == 1 and bw["correct"] == 1, bw)
ts = NB5L.task_solve_times(recs, gold)
check("task solve time: correct, first accept 60 s after the first attempt started", ts["t1"]["correct"] and ts["t1"]["first_accept_sec"] == 60.0, ts)
cd = NB5L.counter_delta({"vllm:prefix_cache_queries_total": 100, "vllm:prefix_cache_hits_total": 10,
                         "vllm:spec_decode_num_accepted_tokens_total": 0, "vllm:spec_decode_num_drafts_total": 0},
                        {"vllm:prefix_cache_queries_total": 300, "vllm:prefix_cache_hits_total": 160,
                         "vllm:spec_decode_num_accepted_tokens_total": 220, "vllm:spec_decode_num_drafts_total": 100})
check("counter deltas (the timestamp-delta metric reset): prefix hit 75 %, acceptance length 3.2",
      cd["prefix_hit_pct"] == 75.0 and cd["mean_acceptance_length"] == 3.2, cd)
hd = NB5L.hist_delta({"vllm:time_to_first_token_seconds_sum": 10, "vllm:time_to_first_token_seconds_count": 2},
                     {"vllm:time_to_first_token_seconds_sum": 160, "vllm:time_to_first_token_seconds_count": 3})
check("latency histogram delta: one more request with TTFT 150 s -> mean 150", hd["time_to_first_token_seconds"] == {"count": 1, "mean_sec": 150.0})
check("engine rates: iteration histogram -> tick = step, cycle = 2 ticks (PP2); estimate -> the step is a cycle",
      NB5L.engine_rates({"step_ms": 40.0, "steps_source": "iteration_histogram"}) ["cycle_ms"] == 80.0
      and NB5L.engine_rates({"step_ms": 40.0, "steps_source": "estimate"})["tick_ms"] == 20.0)
tail8 = RES["setup"]["merlin_plans"]["merlin_fp8"]["argv_tail"]
tail16 = RES["setup"]["merlin_plans"]["merlin_bf16"]["argv_tail"]
d0 = NB5L.argv_diff(tail8, tail8)
d1 = NB5L.argv_diff(tail8, tail8 + ["--speculative-config", "None"])
d2 = NB5L.argv_diff(tail8, tail16)
check("argv diff vs the real RC2 FP8 launch: none for itself; MTP off = --speculative-config changed; BF16 = KV dtype, "
      "KV bytes, max-num-seqs changed", d0 == {"changed": {}, "added": {}, "removed": []}
      and list(d1["changed"]) == ["--speculative-config"] and {"--kv-cache-dtype", "--kv-cache-memory-bytes", "--max-num-seqs"} <= set(d2["changed"]),
      (d1, d2["changed"].keys()))

# ================================================================== 7. the generated notebook
nb_b = NB.read_bytes()
nb = json.loads(nb_b)
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
check("notebook < 400 kB (Kaggle commit limit 407-444 kB)", len(nb_b) < 400_000, len(nb_b))
check("every code cell parses as Python 3.12", all(ast.parse(c, feature_version=(3, 12)) for c in code))
check("no CR in the notebook (LF only)", b"\\r\\n" not in nb_b and b"\r" not in nb_b)
B = runpy.run_path(str(HERE / "build_e25_nb5.py"), run_name="nb5_builder") if not KIT else None
D = B["D"] if not KIT else None
if KIT:
    for _nm in ("verbatim: CELL 0 / task-order cell / E21 CELL 3", "verbatim: E21 donor functions", "verbatim: RC2 CELL 2 + step runner",
                "verbatim: E21 CELL 6/7"):
        skip(_nm)
else:
  check("verbatim: CELL 0, the task-order cell and E21's CELL 3 are byte-identical to the Kaggle-run E21 NB2 notebook",
      code[0] == D[1] and code[3] == D[6] and code[2].startswith(D[5]))
  check("verbatim: every E21 donor function is in the engine cell unchanged", all(s_ in code[4] for _, s_ in B["E21_DONORS"]))
  check("verbatim: RC2 CELL 2 (to the stock fallback, with the 8 marked NB5 edits) and the E23 step runner in the engine cell",
      B["RC2_ENGINE"] in code[4] and B["STEP_RUNNER"] in code[4] and B["RC2_CHECKS"] in code[4])
  check("verbatim: E21 CELL 6/7 with only the capacity-probe edit and the R2 fixed task order (3 added lines)",
        code[5] == B["CELL67"] and B["CELL67"].replace(
    '        out["kv_usage_server"] = out.get("kv_usage")      # NB5: recorded\n        out["kv_usage"] = nb5_gate_kv(out.get("kv_usage"))  # NB5: the live-block reservation fraction\n', "").replace(
    '    if P.get("task_order"):      # NB5 R2: the lead\'s fixed task order (the same in both paired phases)\n'
    '        E12_ORDER = [t for t in P["task_order"] if t in TASK_IDS] + [t for t in E12_ORDER if t not in P["task_order"]]\n', "") == D[8])


def unpack(cell, label):
    m_ = re.search(r'b64decode\("([A-Za-z0-9+/=]+)"\)', cell[cell.index(label):])
    import zlib
    return zlib.decompress(base64.b64decode(m_.group(1))).decode("utf-8")


if KIT:
    skip("packed e23_lab = the source file byte for byte (the notebook's own embedded sha256 check runs in the kits)")
else:
    lab_src = (Path(r"D:/ARC AGI 2/experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/src/e23_lab.py").read_text(encoding="utf-8")
               .replace("\r\n", "\n"))
    check("packed e23_lab = the source file byte for byte", unpack(code[4], "LAB_SRC = ") == lab_src)
check("packed nb5_lib = Build/nb5_lib.py byte for byte", unpack(code[1], "NB5_LIB_SRC = ") == (HERE / "nb5_lib.py").read_text(encoding="utf-8"))
if KIT:
    skip("no ARC-AGI-3 caps in the NB5 config / glue (literal scan of the builder parts)")
    skip("the commit dispatcher never turns thinking off (literal scan of the builder parts)")
else:
  check("no ARC-AGI-3 caps in the NB5 config / glue: no 12288 / 24576 / 32768 / 28000 / 28k pause literals",
        not re.search(r"\b(12288|24576|32768|28000)\b", B["CELL1"] + B["CELL3_PATCH"] + B["NB5_ENGINE"] + B["RUNNER"]))
  check("the commit dispatcher never turns thinking off ('enable_thinking': False only in the RC2 / E21 thinking-off probes)",
        "enable_thinking\": False" not in B["CELL3_PATCH"] and "enable_thinking=False" not in (HERE / "nb5_lib.py").read_text(encoding="utf-8"))
check("max-model-len 131072 in the engine; 11:50 hard stop (710 min) + E15B's 180-s branch stop",
      "MAX_LEN = 131072" in code[4] and '"time_budget_sec": 710 * 60' in code[1])
# CELL 0 + CELL 1 executed for real (commit defaults and the rerun profile)
arc = Path(tempfile.mkdtemp(prefix="nb5_arc_"))
json.dump({"t1": {"train": [], "test": [{"input": [[0]]}]}}, open(arc / "arc-agi_test_challenges.json", "w"))
work = Path(tempfile.mkdtemp(prefix="nb5_w_"))


def run_cells(env):
    old = dict(os.environ)
    for k in ("E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN", "Q27X_PROFILE", "NB5_ATTEMPT_B0", "NB5_ATTEMPT_B1", "E12_MAX_GEN",
              "NB5_NB0_FP8_GATE", "E12_B_EXTRA_ATTEMPTS", "LOCAL_ANALYZER_SEED", "NB5_ADMIT_MODE", "NB5_THINK_FLOOR",
              "NB5_RESERVE", "E12_B_RESERVE", "E12_R6_MIN_ADMIT_SEC", "E12_B_MAX_ATTEMPTS", "Q27X_TIME_BUDGET"):
        os.environ.pop(k, None)
    env = dict(env)
    if not env.pop("_BAKED", None) and "Q27X_PROFILE" not in env:
        env["Q27X_PROFILE"] = "NB5EVAL120"         # the production-default checks; the R2 notebook bakes NB5QUALITY_R2
    os.environ.update(Q27X_ARC_ROOT=str(arc), Q27X_WORKING=str(work), E23_TMP=str(work), **env)
    G = {"__name__": "__main__"}
    try:
        for c in code[:2]:
            exec(compile(c, "<cell>", "exec"), G)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return G


G = run_cells({})
# payload lookup order (lead 2026-10-31): the foysalemonshanto folder, then esfersami50's; sha256 either way; path recorded
_pr = Path(tempfile.mkdtemp(prefix="nb5_pay_"))
_pb, _pn = PAYLOAD.read_bytes(), PAYLOAD.name
_psha = hashlib.sha256(_pb).hexdigest()
_pdir = {o: _pr / "datasets" / o / "q27-e15b-payload" for o in ("foysalemonshanto", "esfersami50")}
for _d in _pdir.values():
    _d.mkdir(parents=True)
    (_d / _pn).write_bytes(_pb)


def _pay():
    _old = os.environ.get("Q27X_PAYLOAD_DIR")
    os.environ["Q27X_PAYLOAD_DIR"] = str(_pr)
    try:
        G["Q27X_PAYLOAD_USED"].clear()
        try:
            return G["q27x_payload"](_pn, _psha) == _pb, dict(G["Q27X_PAYLOAD_USED"].get(_pn) or {})
        except FileNotFoundError as e_:
            return None, str(e_)
    finally:
        os.environ.pop("Q27X_PAYLOAD_DIR", None)
        if _old is not None:
            os.environ["Q27X_PAYLOAD_DIR"] = _old


_ok1, _u1 = _pay()
check("payload: both copies present -> the foysalemonshanto folder is used (exact path recorded, nothing rejected); "
      "CONFIG['nb5']['payload_used'] is that record", _ok1 is True and _u1["path"] == str(_pdir["foysalemonshanto"] / _pn)
      and _u1["rejected_before"] == [] and G["CONFIG"]["nb5"]["payload_used"] is G["Q27X_PAYLOAD_USED"]
      and G["Q27X_PAYLOAD_SLUG"] == "foysalemonshanto/q27-e15b-payload", _u1)
(_pdir["foysalemonshanto"] / _pn).write_bytes(b"corrupt")
_ok2, _u2 = _pay()
check("payload: a wrong-sha foysalemonshanto copy is rejected (recorded) and esfersami50's copy is used",
      _ok2 is True and _u2["path"] == str(_pdir["esfersami50"] / _pn) and _u2["rejected_before"] == [str(_pdir["foysalemonshanto"] / _pn)], _u2)
(_pdir["esfersami50"] / _pn).write_bytes(b"corrupt")
_ok3, _u3 = _pay()
check("payload: no good copy under the two folders -> never a wrong-sha file (FileNotFoundError naming the foysalemonshanto "
      "dataset, or a real copy elsewhere under /kaggle/input)", (_ok3 is None and "foysalemonshanto/q27-e15b-payload" in _u3)
      or (_ok3 is True and _u3["path"].startswith("/kaggle/input")), _u3)
caps = [G["E12_R6"][k] for k in ("E12_R6_CAP_FIRST", "E12_R6_CAP_PROGRESS", "E12_R6_CAP_RESCUE", "E12_R6_CAP_FIRST_B1", "E12_R6_MIN_CHUNK")]
check("commit defaults: attempt 1 = 112,000, attempt 2 = 100,000 (>= 60k floor), every request cap = 107,904 (no 12k/24k/32k), "
      "first exposure 0 (no 28k pause), fair share 0 (no chunks), window 130,560, xhigh",
      G["E15B"]["E12_B_ATTEMPT_TOKENS"] == 112000 and G["E15B"]["E12_B_ATTEMPT_TOKENS_B1"] == 100000 and set(caps) == {107904}
      and G["E12_DUCK_MAX_OUTPUT"] == 107904 and G["E12_R6"]["E12_R6_FIRST_EXPOSURE"] == 0 and G["E12_R6"]["E12_R6_FAIR_SHARE"] == 0
      and G["E15B"]["E12_B_WINDOW"] == 130560 and G["E12_REASONING_EFFORT"] == "xhigh", (G["E15B"], caps))
check("SMOKE: 37-min solve window inside the 40-min gate, early exit on (production); admission projected, floor 60k",
      G["P"]["nb5_phases"][0][2] == 37 * 60 and G["P"]["nb5_phases"][0][6] == {"early_exit": True}
      and G["NB5_SMOKE_GATE_SEC"] == 40 * 60 and G["NB5_ADMIT_MODE"] == "projected" and G["NB5_THINK_FLOOR"] == 60000
      and G["NB5_CTX_MARGIN"] == 2048)
check("commit defaults: profile NB5EVAL120 = SMOKE (3 tasks incl. 0934a4d8, split) + EVAL (variant C); HARD_STOP 11:48, "
      "the epoch watchdog at 11:50", G["PROFILE"] == "NB5EVAL120" and [p[0] for p in G["P"]["nb5_phases"]] == ["SMOKE", "EVAL"]
      and "0934a4d8" in G["NB5_SMOKE_TASKS"] and abs((G["HARD_STOP"] - G["NOTEBOOK_T0"]) - 708 * 60) < 1
      and G["TOTAL_BUDGET_SEC"] == 710 * 60, (G["PROFILE"], G["HARD_STOP"] - G["NOTEBOOK_T0"]))
try:
    run_cells({"NB5_ATTEMPT_B1": "50000"})
    floor_ok = False
except AssertionError:
    floor_ok = True
check("the 60k thinking floor is enforced (attempt 2 = 50,000 is refused outside the local harness)", floor_ok)
G = run_cells({"KAGGLE_IS_COMPETITION_RERUN": "1"})
check("rerun: RERUN_ALL, source test, one EVAL phase with variant C (no SMOKE)", G["PROFILE"] == "RERUN_ALL" and G["P"]["source"] == "test"
      and G["P"]["nb5_phases"] == [["EVAL", "all", None, "on", "C"]])
check("KV dtype switch: fp8 (default, provisional) | bf16; the BF16 and FP8 RC2 pins both present", G["NB5_KV_DTYPE"] == "fp8"
      and set(G["NB5_KV_BYTES"]) == {"fp8", "bf16"} and G["NB5_POOL_PLACEHOLDER"] == {"fp8": 149, "bf16": 78})
try:
    run_cells({"Q27X_PROFILE": "NB5QUALITY"})
    q_refused = False
except RuntimeError as _e:
    q_refused = "NB0_FP8_GATE" in str(_e)
check("quality A/B is refused without NB0's FP8 gate PASS (off by default)", q_refused)
G = run_cells({"Q27X_PROFILE": "NB5QUALITY", "NB5_NB0_FP8_GATE": "PASS"})
qa = [NB5L.capacity_plan(pool, 4096, G["NB5_ATTEMPT"], G["NB5_PROMPT_ALLOW"], G["NB5_TOOL_ALLOW"], 131072, 6, G["NB5_GATE"],
                         seqs=8)["workers"] for pool in (149, 78)]
check("quality A/B (addendum 5): (FP8, K3) then (BF16, K3) on the 11 fixed tasks incl. 0934a4d8; streams from pool / F at the "
      "120k worst case = 4 (FP8 149) and 2 (BF16 78); attempts 104k (>= 60k); no stage 3; one fixed seed",
      [(p[0], p[3], p[5]) for p in G["P"]["nb5_phases"]] == [("Q_FP8_K3", "on", "fp8"), ("Q_BF16_K3", "on", "bf16")]
      and len(G["P"]["task_list"]) == 11 and "0934a4d8" in G["P"]["task_list"] and qa == [4, 2]
      and G["E15B"]["E12_B_EXTRA_ATTEMPTS"] == 0 and min(G["NB5_ATTEMPT"].values()) >= 60000, (qa, G["NB5_ATTEMPT"]))
G = run_cells({"Q27X_PROFILE": "NB5SMOKEDIAG"})
check("the long multi-task SMOKE is its own diagnostic profile (no early exit, no EVAL)",
      [(p[0], p[6]) for p in G["P"]["nb5_phases"]] == [("SMOKE", {"early_exit": False})])
G = run_cells({"Q27X_PROFILE": "NB5BUDGET"})
check("NB5BUDGET (off by default): EVAL30, 4 equal 160-min phases at 60k / 80k / 100k / 112k (all >= the floor), one boot",
      [p[6]["budget"] for p in G["P"]["nb5_phases"]] == [60000, 80000, 100000, 112000] and G["P"]["n_tasks"] == 30
      and len({p[2] for p in G["P"]["nb5_phases"]}) == 1 and len({(p[3], p[5]) for p in G["P"]["nb5_phases"]}) == 1)
G = run_cells({"NB5_ADMIT_MODE": "future"})
check("admission mode 'future' is accepted (placeholder = projected)", G["NB5_ADMIT_MODE"] == "future")
try:
    run_cells({"NB5_ADMIT_MODE": "bogus"})
    bad_mode = False
except AssertionError:
    bad_mode = True
check("an unknown admission mode is refused", bad_mode)
G = run_cells({"NB5_NB0_FP8_GATE": "FAIL"})
check("NB0 FP8 FAIL -> production KV dtype BF16", G["NB5_KV_DTYPE"] == "bf16")
G = run_cells({"_BAKED": 1})
_R2 = G["P"]
check("R2 notebook: baked profile NB5QUALITY_R2 = Q_FP8_K3 (fp8) then Q_BF16_K3 (bf16), MTP on, commit variant A, no SMOKE, "
      "windows None (equal windows set at run time), 180-min budget, 104k x 2 attempts, no stage 3, one fixed seed",
      G["PROFILE"] == "NB5QUALITY_R2" and [(x[0], x[2], x[3], x[4], x[5]) for x in _R2["nb5_phases"]]
      == [("Q_FP8_K3", None, "on", "A", "fp8"), ("Q_BF16_K3", None, "on", "A", "bf16")] and G["TOTAL_BUDGET_SEC"] == 180 * 60
      and G["NB5_ATTEMPT"] == {"b0": 104000, "b1": 104000} and G["E15B"]["E12_B_EXTRA_ATTEMPTS"] == 0
      and os.environ.get("LOCAL_ANALYZER_SEED") is None and _R2["nb5_env"]["LOCAL_ANALYZER_SEED"] == "20261005"
      and _R2["equal_windows"] and abs((G["HARD_STOP"] - G["NOTEBOOK_T0"]) - 178 * 60) < 1, (G["PROFILE"], _R2["nb5_phases"]))
check("R2: the lead's 15 tasks in order with group labels (7 solid / 5 sometimes / 3 never), task_order = that list",
      G["NB5_QAB_TASKS"] == ["1818057f", "45a5af55", "bf45cf4b", "2ba387bc", "8f215267", "38007db0", "58f5dbd5", "247ef758",
                             "f931b4a8", "b5ca7ac4", "65b59efc", "221dfab4", "2b83f449", "16b78196", "64efde09"]
      and _R2["task_list"] == _R2["task_order"] == G["NB5_QAB_TASKS"]
      and [len(G["NB5_QAB_GROUPS"][k]) for k in ("solid", "sometimes", "never")] == [7, 5, 3])
check("A-9: R2 never needs NB5_NB0_FP8_GATE (empty env runs); NB0's NO_GO_FP8_TF_DIVERGENCE baked + the FP8 label "
      "UNVALIDATED_FP8_UNDER_TEST recorded in CONFIG", G["CONFIG"]["nb5"]["quality_ab"]["nb0"]["verdict"] == "NO_GO_FP8_TF_DIVERGENCE"
      and G["CONFIG"]["nb5"]["quality_ab"]["nb0_gate_required"] is False and G["NB5_QAB_FP8_LABEL"] == "UNVALIDATED_FP8_UNDER_TEST")
check("R2 levers recorded: admission 'slots' (safety cap = workers), measured F, SEQS table, CAPS16 trailing compilation config "
      "(FULL covers 16 x 4), the pin ladder + #60490, lever env MERLIN_SHARE_DRAFT_WORKSPACE=1 only (no SSM dtype / logits change)",
      G["NB5_ADMIT_MODE"] == "slots" and G["NB5_F_RULE"] == "measured" and G["NB5_SEQS_RULE"] == "table"
      and G["NB5_EXTRA_ARGS"][-2] == "--compilation-config" and json.loads(G["NB5_EXTRA_ARGS"][-1])["cudagraph_capture_sizes"]
      == [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64] and _R2["pin_ladder"] and _R2["patch_60490"]
      and G["NB5_LEVER_ENV"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"} and "--mamba-ssm-cache-dtype" not in G["NB5_EXTRA_ARGS"])


def merlin_server_env(lever_env, base_env):
    """NB5Q-ENV-1: the notebook's own merlin_fn_server (cut from the engine cell) on Merlin's REAL launch.build() (the
    ad42e37 launcher, sha-pinned), FP8 / BF16 / FP8 + MTP-off as the kit's three boots; host_phase (disk checks of the
    installed tree) and the help check are the only stubs; no server starts. Returns {boot: (plan.env, server env, argv, record)}."""
    import importlib
    import types
    shas = {n: hashlib.sha256((MERLIN_TOOLS / n).read_bytes()).hexdigest() for n in MERLIN_LAUNCH_SHA}
    assert shas == MERLIN_LAUNCH_SHA, ("Merlin launcher files differ from ad42e37", shas)
    sys.path.insert(0, str(MERLIN_TOOLS))
    try:
        sys.modules.pop("launch", None)
        sys.modules.pop("registry", None)
        ML = importlib.import_module("launch")
    finally:
        sys.path.remove(str(MERLIN_TOOLS))
    plans = {}

    def build(*a, **k):
        p_ = ML.build(*a, **k)
        plans[a[3]["PORT"]] = dict(p_.env)
        return p_
    tree = ast.parse(code[4])
    want = {"merlin_fn_server", "MAX_LEN", "MERLIN_PROFILE", "XHIGH", "KV_FP8"}
    src = "\n".join(ast.get_source_segment(code[4], n_) for n_ in tree.body
                    if (isinstance(n_, ast.FunctionDef) and n_.name in want)
                    or (isinstance(n_, ast.Assign) and any(isinstance(x, (ast.Name, ast.Tuple)) and
                                                           ({x.id} if isinstance(x, ast.Name) else {e.id for e in x.elts}) & want
                                                           for x in n_.targets)))
    kv = {"MERLIN_VENV": "/tmp/merlin_off/venv", "MERLIN_TREE": "/tmp/merlin_off/merlin-engine",       # = the real kit's install
          "MERLIN_SHIM": "/tmp/merlin_off/libglibc238_shim.so", "MERLIN_CC": "/tmp/merlin_gcc13/bin/x86_64-conda-linux-gnu-gcc",
          "MERLIN_CXX": "/tmp/merlin_gcc13/bin/x86_64-conda-linux-gnu-g++"}

    class _Srv:
        def __init__(self, name, argv, env, port, logs):
            self.name, self.argv, self.env, self.port, self.proc = name, argv, env, port, None
    ns = {"json": json, "os": os, "sys": sys, "Path": Path, "FAKE": None, "MServer": _Srv, "LOGS": Path("/tmp/logs"),
          "TMP": Path("/tmp"), "SHIM": Path("/tmp/e23_libcuda"), "CKPT": Path("/tmp/fn_merlin"), "MERLIN_CC": {"mode": "gcc13"},
          "HOST_PHASE": {}, "HELP": {}, "R": {"setup": {}}, "save": lambda: None, "NB5L": NB5L,
          "L": types.SimpleNamespace(help_check=lambda *a: {"ran": False, "stub": True}),
          "merlin_install": lambda: {"kv": kv, "cuda_home": "/tmp/merlin_off/venv/cu13",
                                     "ML": types.SimpleNamespace(build=build, host_phase=lambda plan: None,
                                                                 distributed_ports=ML.distributed_ports)},
          "NB5_BASE_ENV": dict(base_env), "NB5_BOOT_LEVERS": dict(lever_env)}
    exec(compile(src, "<nb5 merlin_fn_server>", "exec"), ns)
    out = {}
    for name, kv_, port, extra in (("u_fp8", "fp8", 8001, []), ("u_bf16", "bf16", 8000, []),
                                   ("u_fp8_nomtp", "fp8", 8002, list(G["NB5_MTP_OFF_ARGS"]))):
        ns["NB5_BOOT_EXTRA"] = list(G["NB5_EXTRA_ARGS"]) + extra + ["--max-num-seqs", str(NB5L.seqs_for(kv_, 1.0))]
        s = ns["merlin_fn_server"](name, kv_, port, ns["KV_FP8"] + 2 ** 30)
        out[name] = (plans[str(port)], s.env, s.argv, ns["R"]["setup"]["merlin_plans"][name])
    return out


_me = merlin_server_env(G["NB5_LEVER_ENV"], G["NB5_BASE_ENV"])
check("NB5Q-ENV-1: Merlin's real launch.build() (ad42e37, sha-pinned) drops MERLIN_SHARE_DRAFT_WORKSPACE from plan.env; the "
      "notebook's merlin_fn_server puts it back AFTER build(): server env '1' on the FP8, BF16 and FP8 MTP-off boots, the final "
      "value recorded in merlin_plans[name].lever_env",
      all("MERLIN_SHARE_DRAFT_WORKSPACE" not in pe and se.get("MERLIN_SHARE_DRAFT_WORKSPACE") == "1"
          and rec["lever_env"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"} for pe, se, _a, rec in _me.values()) and len(_me) == 3,
      {n: (sorted(k for k in pe if k.startswith("MERLIN_")), se.get("MERLIN_SHARE_DRAFT_WORKSPACE"), rec.get("lever_env"))
       for n, (pe, se, _a, rec) in _me.items()})
_me2 = merlin_server_env({}, {"DRAFT_HEAD": "int8"})
check("NB5Q-ENV-1: a base knob still reaches build() (DRAFT_HEAD=int8 -> draft_lm_head_quant in the speculative config) and "
      "the dropped-lever try has no MERLIN_SHARE_DRAFT_WORKSPACE in the server env",
      all('"draft_lm_head_quant":"int8"' in " ".join(a_) and "MERLIN_SHARE_DRAFT_WORKSPACE" not in se and se.get("DRAFT_HEAD") == "int8"
          and rec["lever_env"] == {"DRAFT_HEAD": "int8"} for n, (pe, se, a_, rec) in _me2.items() if n != "u_fp8_nomtp")
      and "MERLIN_SHARE_DRAFT_WORKSPACE" not in _me2["u_fp8_nomtp"][1],
      {n: ([x for x in a_ if "draft_lm_head_quant" in x][:1], se.get("MERLIN_SHARE_DRAFT_WORKSPACE"), rec.get("lever_env"))
       for n, (pe, se, a_, rec) in _me2.items()})
_lg = _pr / "q27-e15b-payload"
_lg.mkdir()
(_lg / _pn).write_bytes(_pb)
G = run_cells({})
_okL, _uL = _pay()
check("KIT-5: payload only at the LEGACY mount /kaggle/input/q27-e15b-payload/ (datasets/ copies corrupt) -> found by the "
      "0-3 level search, sha-checked, path recorded", _okL is True and _uL["path"] == str(_lg / _pn), _uL)
_fsrc = next(ast.get_source_segment(code[4], n_) for n_ in ast.parse(code[4]).body
             if isinstance(n_, ast.FunctionDef) and n_.name == "nb5_host_floor")
_hf_ns = {"os": os, "NB5_RAM_NEED_MIB": 73 * 1024, "FAKE": None}
exec(_fsrc, _hf_ns)
os.environ["NB5_FAKE_HOST_FLOOR_MIB"] = "1024"
_f_real = _hf_ns["nb5_host_floor"]()
_hf_ns["FAKE"] = "/fake/server.py"
_f_fake = _hf_ns["nb5_host_floor"]()
os.environ.pop("NB5_FAKE_HOST_FLOOR_MIB")
check("KIT-6: NB5_FAKE_HOST_FLOOR_MIB is IGNORED without FAKE (73 GiB kept); FAKE mode -> 1024 MiB (the 30 GiB CPU kit passes)",
      _f_real[0] == 73 * 1024 and _f_fake[0] == 1024, (_f_real, _f_fake))
qt = NB5L.quality_table({"t1": [{"t0": 10.0, "t1": 70.0, "generated_tokens": 500,
                                 "submissions": [{"accepted": True, "test_outputs": [[[1, 2]]]}]}]}, {"t1": [[[1, 2]]]},
                        [["t1", 0, "chat", 10, 0, 5, "tool_calls", 1, 2.5, 1], ["t1", 0, "commit_C", 12, 10, 3, "tool_calls", 1, 0.4, 1]])
check("quality table: exact on every test input, parsable grid, complete (natural closed last turn), tokens, wall time, "
      "first-request latency, tool-call rate", qt["t1"] == {**qt["t1"], "exact_all_tests": True, "parsable_grid": True,
                                                             "complete": True, "output_tokens": 500, "wall_sec": 60.0,
                                                             "first_request_latency_sec": 2.5, "tool_call_rate": 1.0}, qt)
if KIT:
    skip("no module-level name clash between RC2-derived engine code and the Duck / E21 cells")
    skip("PUSHED sha guard: a rebuild of a pushed notebook goes to *_REBUILD.ipynb")
    skip("the six pre-R2 NB5 notebooks unchanged + frozen in PUSHED")
    skip("NB5-2: pyflakes undefined names over the R2 code cells (builder function)")
    check("NB5-2: tarfile imported and E11Server defined in the engine cell", "import tarfile" in code[4] and "def E11Server(" in code[4])
else:
    # module-level name collisions between the RC2-derived code and the Duck / E21 cells
    def bound(src):
        return set(E21_ := B["E21"]["module_bindings"](src))


    rc2_names = bound(B["RC2_ENGINE"]) | bound(B["RC2_LAB_LOAD"]) | bound(B["RC2_INPUTS"]) | bound(B["RC2_PROMPTS"]) | bound(B["STEP_RUNNER"]) | bound(B["RC2_CHECKS"])
    duck_names = bound(D[1]) | bound(D[5]) | bound(D[6]) | bound(code[1]) | bound(B["CELL67"]) | set(B["E21"]["module_bindings"](B["CELL67"]))
    # names the Duck cells READ (globals) that RC2 code also binds would silently change Duck behaviour
    def global_reads(src):
        """Names a cell reads from module scope (symtable: top-level references + implicit globals of every nested scope);
        function locals / comprehension variables are not module names."""
        import symtable
        out, todo = set(), [symtable.symtable(src, "<c>", "exec")]
        while todo:
            tb = todo.pop()
            for sy in tb.get_symbols():
                if (tb.get_type() == "module" and sy.is_referenced()) or (tb.get_type() != "module" and sy.is_global()):
                    out.add(sy.get_name())
            todo += tb.get_children()
        return out


    duck_reads = set().union(*(global_reads(c) for c in (D[5], B["CELL67"], D[1], code[1])))
    clash = sorted((rc2_names & (duck_names | duck_reads)) - {"_t", "_", "t", "k", "d", "mf", "json", "os", "re", "sys", "time", "Path",
                                                               "shutil", "subprocess", "hashlib", "threading", "math", "traceback",
                                                               "tarfile", "TMP", "INP", "e", "_e", "r", "i", "p", "s", "x", "g", "n",
                                                               # _n: a temp of Duck CELL 3's sandbox self-test (read only there,
                                                               # before RC2 CELL 3 runs) and RC2 CELL 3's file loop
                                                               "_n"})
    check("no module-level name clash between RC2-derived engine code and the Duck / E21 cells (PROFILE renamed MERLIN_PROFILE, "
          "OUT -> ENG_OUT, HARD_STOP -> HARD_STOP_EPOCH)", not clash, clash)
    # PUSHED guard: a pushed notebook is never overwritten (rebuild -> *_REBUILD)
    tmpd = Path(tempfile.mkdtemp(prefix="nb5_push_"))
    B["NBDIR"] = tmpd
    B["build"].__globals__["NBDIR"] = tmpd
    B["build"].__globals__["HERE"] = tmpd
    first = B["build"]("NB5EVAL120")
    B["build"].__globals__["PUSHED"].add(hashlib.sha256(first.read_bytes()).hexdigest())
    second = B["build"]("NB5EVAL120")
    check("PUSHED sha guard: a rebuild of a pushed notebook goes to *_REBUILD.ipynb", second.name.endswith("_REBUILD.ipynb")
          and first.exists(), (first.name, second.name))
    _six = {n: hashlib.sha256((HERE.parent / n).read_bytes()).hexdigest() for n in (
        "Q27-XHIGH-E25_NB5_FNDUCK.ipynb", "Q27-XHIGH-E25_NB5_FNDUCK_BUDGET.ipynb", "Q27-XHIGH-E25_NB5_FNDUCK_EVAL30.ipynb",
        "Q27-XHIGH-E25_NB5_FNDUCK_EVAL30MTP.ipynb", "Q27-XHIGH-E25_NB5_FNDUCK_QUALITY.ipynb", "Q27-XHIGH-E25_NB5_FNDUCK_SMOKEDIAG.ipynb")}
    check("the six pre-R2 NB5 notebooks are unchanged on disk and frozen in the builder's PUSHED set (a rebuild -> *_REBUILD)",
          set(_six.values()) <= set(B["PUSHED"]) and len(set(_six.values())) == 6, _six)
    _un = B["undefined_names_check"](B["cells"]("NB5QUALITY_R2"))
    check("NB5-2: tarfile imported and E11Server defined in the engine cell", "import tarfile" in code[4] and "def E11Server(" in code[4])
    if _un is None:
        skip("NB5-2: pyflakes undefined names over the R2 code cells (pyflakes not installed in this interpreter)")
    else:
        check("NB5-2: pyflakes finds no undefined name in the R2 notebook's code cells", _un is True)

n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS" + (f" ({len(SKIPPED)} build-time checks skipped in kit mode)" if SKIPPED else ""))
sys.exit(1 if n_fail else 0)
