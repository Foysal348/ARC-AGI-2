"""E25 NB4S quick unit tests (CPU, stdlib only; no GPU, no server, no local end-to-end).

  python nb4s_unit_tests.py [--b1-traces DIR] [--merlin-src DIR] [--nb1-results FILE]

T1  arm plan: B / S / QP / QD, one env key each, never-do list (env + argv) enforced, launch check on full envs
T2  env read-back parser (RPC) + /proc environ parser
T3  /metrics parsing (same dict as e23_lab.prom on the same text) + iteration-token histogram parity
T4  per-graph stall parser on synthetic traces; on the REAL NB1 R2 B1 traces (if found): node 172, stall ~21.5 ms,
    BALANCED 4+4, base_check labels B1 stalled; nb4_timeline.analyze + summarize_timeline on the same traces
T5  speed comparison, budget rules
T6  per-arm verdicts + the best arm (every label; rev 2: ENGAGED_FALLBACK, the latest Q read-back wins)
T7  nb4_ppack patch sequencing (PatchSeq) with stubs; the real apply / revert on a COPY of Merlin's three patched files
    (pp_utils.py, model_runner.py, gpu_worker.py; if found)
T8  CELL flow: the notebook's CELL 4 (step) + CELL 7 (the plan) from nb4s_code_export.py, executed with a stub boot
    runner (no server): order, patch apply / revert, budget skips, dependency skips, errors, the verdict always written
T9  notebook facts: embedded module shas == the Build files, the worker extension's nb4s_env, the export == the notebook
"""
import argparse
import base64
import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import traceback
import types
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb4s_lib as N  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    if not cond:
        print("FAIL", name, str(detail)[:600])


# ------------------------------------------------------------------ helpers
BASE_EXTRA = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}"]
BASE_ENV = {"VLLM_SERVER_DEV_MODE": "1"}
QENV = {"QP": {"NB4_PPACK": "1"}, "QD": {"NB4_PPDEFER": "1"}}
MERLIN_ENV = {"KV_BYTES": "3553580524", "NCCL_P2P_DISABLE": "1", "NCCL_VERSION": "2.25.1-1", "P2P": "off",
              "VLLM_PP_LAYER_PARTITION": "25,23", "VLLM_SERVER_DEV_MODE": "1", "VLLM_SKIP_P2P_CHECK": "1",
              "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}


def plan():
    return N.arm_plan(BASE_EXTRA, BASE_ENV, 3553580524, q_env=QENV)


def raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def kern(ts, dur, corr, stream=7, name="void k()", **a):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "tid": stream,
            "args": dict({"correlation": corr, "stream": stream}, **a)}


def synth_trace(rank, n_graphs=6, nodes=600, gap_us=21000.0, period=50000.0, gap_at=172, gen=(4, 4)):
    """A synthetic rank trace: n FULL graph replays of `nodes` kernels; stage-0 graphs stall `gap_us` before node
    `gap_at` and a pp_broadcast Int x8 kernel ends 6 us before the stall ends; scope names carry gen requests."""
    ev, t = [], 1000.0
    for g in range(n_graphs):
        corr = 1000 + g
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": t - 50, "dur": 20, "tid": 1, "args": {"correlation": corr}})
        n = gen[g % 2]
        ev.append({"ph": "X", "cat": "user_annotation", "name": f"execute_context_0(0)_generation_{n}({n * 4})", "ts": t - 60, "dur": 30, "tid": 1})
        cur = t
        for i in range(nodes):
            if i == gap_at and gap_us > 0:
                ev.append(kern(cur + gap_us - 30, 24, 9999, 31, "ncclDevKernel_Broadcast_RING_LL(x)", **{
                    "Collective name": "broadcast", "dtype": "Int", "In msg nelems": 8, "Process Group Description": "pp_broadcast"}))
                cur += gap_us
            ev.append(kern(cur, 10.0, corr, 2338 if i == gap_at else 7, "void k%d()" % i, **{"graph id": 5}))
            cur += 10.0
        t += period
    return {"traceEvents": ev, "distributedInfo": {"rank": rank}}


def write_traces(d, gap0=21000.0, gap1=0.0, gen=(4, 4), gap_at=172):
    d.mkdir(parents=True, exist_ok=True)
    for rank in range(4):
        pp, tp = rank // 2, rank % 2
        tr = synth_trace(rank, gap_us=gap0 if pp == 0 else gap1, gen=gen, gap_at=gap_at)
        with gzip.open(d / f"dp0_pp{pp}_tp{tp}_dcp0_ep{tp}_rank{rank}.1.pt.trace.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(tr, fh)
    return d


def win(tok, status="STEADY", parity="BALANCED", tick=48.0):
    return {"status": status, "tok_s": tok, "tick_ms": tick, "running_eq_level": True, "preemptions": 0,
            "parity": {"label": parity}}


def stall_rec(median, share, node=172, rel="ncclDevKernel_Broadcast_RING_LL broadcast Intx8 pp_broadcast", parity="BALANCED",
              s1_median=0.05):
    return {"stage0": {"graphs": 76, "stall_ms_median": median, "share_below_1ms": share, "node_index_mode": node,
                       "released_by_mode": rel},
            "stage1": {"graphs": 76, "stall_ms_median": s1_median, "sendrecv_ms_per_tick_mean": 8.2,
                       "pp_broadcast_send_ms_per_tick": 0.05},
            "parity": {"label": parity}}


def arm_rec(key, stall, tok=(255.0, 256.0), **kw):
    a = plan()[key]
    r = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS", "oracle_vs": "EXACT", "pin": 3553580524,
         "env_readback": {"ok": True, "why": "ok"}, "stall": stall,
         "steady": {"8k@8#1": win(tok[0]), "8k@8#2": win(tok[1]), "40k@8": win(tok[0] * 0.98)}}
    if a["q_label"]:
        r["q_readback"] = {"label": "ENGAGED", "detail": {}}
    r.update(kw)
    return r


# ------------------------------------------------------------------ T1
def t1():
    a = plan()
    check("T1 order", tuple(a) == ("B", "S", "QP", "QD"))
    check("T1 plan valid", N.check_arm_plan(a))
    check("T1 S env", a["S"]["env"].get("CUDA_DEVICE_MAX_CONNECTIONS") == "32" and a["S"]["change"] == {"CUDA_DEVICE_MAX_CONNECTIONS": "32"})
    check("T1 QP env no S", a["QP"]["env"].get("NB4_PPACK") == "1" and "CUDA_DEVICE_MAX_CONNECTIONS" not in a["QP"]["env"])
    check("T1 QD env", a["QD"]["env"].get("NB4_PPDEFER") == "1" and "NB4_PPACK" not in a["QD"]["env"])
    check("T1 B none", not any(k in a["B"]["env"] for k in N.WATCHED))
    check("T1 needs_patch", [k for k in a if a[k]["needs_patch"]] == ["QP", "QD"])
    check("T1 same flags + pin", all(x["extra"] == a["B"]["extra"] and x["kv_bytes"] == 3553580524 for x in a.values()))
    check("T1 expect", a["QP"]["expect"] == {"CUDA_DEVICE_MAX_CONNECTIONS": None, "NB4_PPACK": "1", "NB4_PPDEFER": None})
    check("T1 ports distinct", len({x["port"] for x in a.values()}) == 4)
    for label, mut in (("graph mixing 0", lambda p: p["S"]["env"].update(NCCL_GRAPH_MIXING_SUPPORT="0")),
                       ("graph register 1", lambda p: p["B"]["env"].update(NCCL_GRAPH_REGISTER="1")),
                       ("LL128", lambda p: p["QP"]["env"].update(NCCL_PROTO="LL128")),
                       ("P2P on", lambda p: p["QD"]["env"].update(P2P="on")),
                       ("NCCL_P2P_DISABLE 0", lambda p: p["S"]["env"].update(NCCL_P2P_DISABLE="0")),
                       ("launch order", lambda p: p["S"]["env"].update(NCCL_LAUNCH_ORDER_IMPLICIT="1")),
                       ("CTA tuning", lambda p: p["S"]["env"].update(NCCL_MIN_CTAS="4")),
                       ("shm memcpy", lambda p: p["S"]["env"].update(NCCL_SHM_USE_CUDA_MEMCPY="1")),
                       ("COUNTERS", lambda p: p["S"]["env"].update(COUNTERS="1")),
                       ("no async", lambda p: [x["extra"].append("--no-async-scheduling") for x in p.values()]),
                       ("prefix off", lambda p: [x["extra"].append("--no-enable-prefix-caching") for x in p.values()]),
                       ("ssm bf16", lambda p: [x["extra"].extend(["--mamba-ssm-cache-dtype", "bfloat16"]) for x in p.values()]),
                       ("QP carries S env", lambda p: p["QP"]["env"].update(CUDA_DEVICE_MAX_CONNECTIONS="32")),
                       ("two keys", lambda p: p["S"]["env"].update(VLLM_X="1")),
                       ("flags differ", lambda p: p["S"]["extra"].append("--foo")),
                       ("pin differs", lambda p: p["QD"].update(kv_bytes=1)),
                       ("B sets flag", lambda p: p["B"]["env"].update(CUDA_DEVICE_MAX_CONNECTIONS="8")),
                       ("ports", lambda p: p["S"].update(port=8011))):
        p = plan()
        mut(p)
        check(f"T1 refuses {label}", raises(lambda: N.check_arm_plan(p)))
    check("T1 never-do clean base", N.never_do_violations(MERLIN_ENV, BASE_EXTRA + ["--enable-prefix-caching"]) == [])
    check("T1 never-do last flag wins", N.never_do_violations({}, ["--mamba-ssm-cache-dtype", "bfloat16", "--mamba-ssm-cache-dtype", "auto"]) == [])
    for k in a:
        full = dict(MERLIN_ENV, **a[k]["change"])
        check(f"T1 launch_check {k} ok", N.launch_check(a[k], full, BASE_EXTRA) == [], N.launch_check(a[k], full, BASE_EXTRA))
    check("T1 launch_check S missing", N.launch_check(a["S"], MERLIN_ENV, BASE_EXTRA) != [])
    check("T1 launch_check B with flag", N.launch_check(a["B"], dict(MERLIN_ENV, CUDA_DEVICE_MAX_CONNECTIONS="32"), BASE_EXTRA) != [])
    check("T1 launch_check never-do", N.launch_check(a["B"], dict(MERLIN_ENV, NCCL_GRAPH_MIXING_SUPPORT="0"), BASE_EXTRA) != [])
    check("T1 launch_check QD with pack", N.launch_check(a["QD"], dict(MERLIN_ENV, NB4_PPDEFER="1", NB4_PPACK="1"), BASE_EXTRA) != [])


def t1_real_env(nb1_results):
    if not nb1_results or not Path(nb1_results).is_file():
        print("SKIP T1 real NB1 R2 env (no nb1_results.json)")
        return
    R = json.loads(Path(nb1_results).read_text(encoding="utf-8"))
    mp = R["setup"]["merlin_plans"]["nb1_b3_fp8_pin100"]
    a = plan()
    for k in a:
        env = dict(mp["env_keys"], **a[k]["change"])
        check(f"T1 real B3 env + argv, launch_check {k}", N.launch_check(a[k], env, mp["argv_tail"]) == [], N.launch_check(a[k], env, mp["argv_tail"]))


# ------------------------------------------------------------------ T2
def t2():
    exp = {"CUDA_DEVICE_MAX_CONNECTIONS": "32", "NB4_PPACK": None, "NB4_PPDEFER": None}
    ok = {"results": [{"rank": r, "pid": 100 + r, "env": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}} for r in range(4)]}
    v = N.parse_env_readback(ok, exp)
    check("T2 ok", v["ok"] and not v["definitive"], v)
    v = N.parse_env_readback({"results": ok["results"][:3]}, exp)
    check("T2 3 workers", not v["ok"] and not v["definitive"])
    bad = copy.deepcopy(ok)
    bad["results"][2]["env"] = {}
    v = N.parse_env_readback(bad, exp)
    check("T2 one rank missing value -> definitive", not v["ok"] and v["definitive"], v)
    dup = copy.deepcopy(ok)
    dup["results"][3]["rank"] = 0
    check("T2 duplicate rank", not N.parse_env_readback(dup, exp)["ok"])
    check("T2 rpc error", not N.parse_env_readback({"error": "x"}, exp)["ok"] and not N.parse_env_readback({"error": "x"}, exp)["definitive"])
    check("T2 garbage", not N.parse_env_readback(None, exp)["ok"] and not N.parse_env_readback({"results": "x"}, exp)["ok"])
    werr = copy.deepcopy(ok)
    werr["results"][1]["error"] = "boom"
    check("T2 worker error", not N.parse_env_readback(werr, exp)["ok"])
    b = {"CUDA_DEVICE_MAX_CONNECTIONS": None, "NB4_PPACK": None, "NB4_PPDEFER": None}
    check("T2 B absent ok", N.parse_env_readback({"results": [{"rank": r, "env": {}} for r in range(4)]}, b)["ok"])
    check("T2 B present fails", not N.parse_env_readback(ok, b)["ok"])
    q = {"CUDA_DEVICE_MAX_CONNECTIONS": None, "NB4_PPACK": "1", "NB4_PPDEFER": None}
    check("T2 QP ok", N.parse_env_readback({"results": [{"rank": r, "env": {"NB4_PPACK": "1"}} for r in range(4)]}, q)["ok"])
    pe = N.parse_proc_environ({1: b"A=1\0CUDA_DEVICE_MAX_CONNECTIONS=32\0", 2: None, 3: b"CUDA_DEVICE_MAX_CONNECTIONS=32\0X=y"}, {"CUDA_DEVICE_MAX_CONNECTIONS": "32"})
    check("T2 proc environ", pe["all_expected"] and pe["readable"] == 2 and pe["values"]["2"] == "UNREADABLE", pe)
    pe = N.parse_proc_environ({1: b"A=1\0"}, {"CUDA_DEVICE_MAX_CONNECTIONS": "32"})
    check("T2 proc environ missing", not pe["all_expected"])


# ------------------------------------------------------------------ T3
PROM_TXT = """# HELP x
vllm:generation_tokens_total{model_name="q27",engine="0"} 1000.0
vllm:num_requests_running{model_name="q27",engine="0"} 8.0
vllm:iteration_tokens_total_bucket{le="1.0",model_name="q27",engine="0"} 0.0
vllm:iteration_tokens_total_bucket{le="8.0",model_name="q27",engine="0"} 2.0
vllm:iteration_tokens_total_bucket{le="16.0",model_name="q27",engine="0"} 102.0
vllm:iteration_tokens_total_bucket{le="32.0",model_name="q27",engine="0"} 110.0
vllm:iteration_tokens_total_bucket{le="+Inf",model_name="q27",engine="0"} 115.0
vllm:iteration_tokens_total_count{model_name="q27",engine="0"} 115.0
vllm:kv_cache_usage_perc{model_name="q27",engine="0"} NaN
"""


def lab_prom_reference(txt):
    """e23_lab.prom's parsing loop, from the notebook's embedded LAB_SRC (fed the text instead of the URL)."""
    exp = (HERE / "nb4s_code_export.py").read_text(encoding="utf-8")
    line = next(l for l in exp.splitlines() if l.strip().startswith("LAB_SRC = "))
    src = json.loads(line.strip()[len("LAB_SRC = "):])
    ns = {"__name__": "lab_ref"}
    exec(compile(src, "e23_lab", "exec"), ns)

    class R_:
        def read(self):
            return txt.encode()
    ns["U"] = types.SimpleNamespace(urlopen=lambda *a, **k: R_())
    return ns["prom"](0)


def t3():
    agg, b = N.prom_parse(PROM_TXT)
    try:
        ref = lab_prom_reference(PROM_TXT)
        check("T3 prom_parse == e23_lab.prom", agg == ref, (agg, ref))
    except Exception as e:
        check("T3 prom_parse == e23_lab.prom", False, repr(e))
    check("T3 buckets", b[16.0] == 102.0 and b[float("inf")] == 115.0, b)
    h0 = {1.0: 0, 8.0: 0, 16.0: 100, 32.0: 100, float("inf"): 100}
    h1 = {1.0: 0, 8.0: 0, 16.0: 300, 32.0: 302, float("inf"): 302}
    p = N.parity_from_hist(N.hist_delta(h0, h1), 8)
    check("T3 balanced 4+4", p["label"] == "BALANCED", p)
    h1b = {1.0: 0, 8.0: 0, 16.0: 200, 32.0: 300, float("inf"): 300}     # 5/3: half the steps at 20 tokens
    check("T3 imbalanced 5+3", N.parity_from_hist(N.hist_delta(h0, h1b), 8)["label"] == "IMBALANCED")
    h1c = {1.0: 0, 8.0: 0, 16.0: 100, 32.0: 300, float("inf"): 300}     # 8/0: every step 32 tokens
    check("T3 imbalanced 8+0", N.parity_from_hist(N.hist_delta(h0, h1c), 8)["label"] == "IMBALANCED")
    coarse0, coarse1 = {64.0: 0, float("inf"): 0}, {64.0: 100, float("inf"): 100}
    check("T3 unresolved coarse", N.parity_from_hist(N.hist_delta(coarse0, coarse1), 8)["label"] == "UNRESOLVED")
    check("T3 unresolved none", N.parity_from_hist(None, 8)["label"] == "UNRESOLVED")
    check("T3 unresolved no steps", N.parity_from_hist(N.hist_delta(h0, h0), 8)["label"] == "UNRESOLVED")
    check("T3 parity scopes", N.parity_from_scopes({"0": {"4": 20}, "1": {"4": 20}})["label"] == "BALANCED"
          and N.parity_from_scopes({"0": {"8": 20}, "1": {"0": 20}})["label"] == "IMBALANCED"
          and N.parity_from_scopes({"0": {"5": 20}, "1": {"4": 20}})["label"] == "BALANCED")


# ------------------------------------------------------------------ T4
def t4(tmp, b1):
    d = write_traces(tmp / "syn_stalled")
    st = N.graph_stalls(d)
    s0 = st["stage0"]
    check("T4 synthetic stalled", s0["graphs"] == 8 and s0["node_index_mode"] == 172 and abs(s0["stall_ms_median"] - 21.0) < 0.1
          and s0["share_below_1ms"] == 0.0, s0)
    check("T4 synthetic release", "Broadcast_RING_LL broadcast Intx8 pp_broadcast" in (s0["released_by_mode"] or "")
          and s0["offset_to_pp_broadcast_end_us_median"] == 6.0, s0)
    check("T4 synthetic parity", st["parity"]["label"] == "BALANCED")
    check("T4 synthetic stage1 clean", st["stage1"]["share_below_1ms"] == 1.0, st["stage1"])
    st2 = N.graph_stalls(write_traces(tmp / "syn_fixed", gap0=0.0))
    check("T4 synthetic fixed", st2["stage0"]["share_below_1ms"] == 1.0 and st2["stage0"]["node_index_mode"] is None, st2["stage0"])
    st3 = N.graph_stalls(write_traces(tmp / "syn_8_0", gen=(8, 0)))
    check("T4 synthetic 8+0", st3["parity"]["label"] == "IMBALANCED")
    st4 = N.graph_stalls(write_traces(tmp / "syn_moved1", gap0=0.0, gap1=9000.0))
    check("T4 synthetic stage-1 stall", st4["stage1"]["stall_ms_median"] >= 8.9, st4["stage1"])
    check("T4 no trace", N.graph_stalls(tmp / "empty_dir_x" if (tmp / "empty_dir_x").mkdir() is None else None).get("error") == "NO_TRACE_FILE")
    if not b1 or not Path(b1).is_dir():
        print("SKIP T4 real B1 traces (not found)")
        return
    st = N.graph_stalls(b1)
    s0 = st["stage0"]
    check("T4 B1 node 172", s0["node_index_mode"] == 172, s0)
    check("T4 B1 stall ~21.5 ms", 20.0 <= s0["stall_ms_median"] <= 23.0 and s0["share_below_1ms"] == 0.0, s0)
    check("T4 B1 released by Int x8", "Broadcast_RING_LL broadcast Intx8 pp_broadcast" in (s0["released_by_mode"] or ""), s0)
    check("T4 B1 offset < 20 us", 0 <= s0["offset_to_pp_broadcast_end_us_median"] < 20, s0)
    check("T4 B1 parity 4+4", st["parity"]["label"] == "BALANCED" and st["parity"]["modal_requests"] == {"0": 4, "1": 4}, st["parity"])
    check("T4 B1 stage-1 SendRecv ~8 ms", 6.0 <= (st["stage1"]["sendrecv_ms_per_tick_mean"] or 0) <= 10.0, st["stage1"])
    check("T4 B1 stage-1 no stall", (st["stage1"]["stall_ms_median"] or 0) < 1.0, st["stage1"])
    streams = [x for r in st["by_rank"].values() if r["stage"] == 0 for x in r["stream_mode"]]
    check("T4 B1 stream 2338", all(s == "2338" for s, _ in streams), streams)
    B1 = arm_rec("B", st, env_readback={"ok": True})
    ok, lab, why = N.base_check(B1)
    check("T4 verdict code labels B1 STALLED (base_check OK)", ok and lab == "OK", (lab, why))
    v = N.arm_verdict(B1, arm_rec("S", st), "S")
    check("T4 B1 traces as both arms -> NO_EFFECT", v["verdict"] == "NO_EFFECT", v)
    check("T4 B1 as S is not STALL_FIXED", not v["verdict"].startswith("STALL_FIXED"))
    try:
        import nb4_timeline as TL
        rep = TL.strip_private(TL.analyze(b1))
        sm = N.summarize_timeline(rep)
        check("T4 timeline tick ~49.7", 48.0 <= (sm["tick_ms_mean"] or 0) <= 51.5, sm.get("tick_ms_mean"))
        check("T4 timeline stage-0 in-graph stall ~21.7", 20.0 <= (sm["stages"]["stage0"]["in_graph_stall"] or 0) <= 23.0, sm["stages"])
        check("T4 timeline share stall<1ms = 0", sm["tick_share_stall_below_1ms"] == 0.0, sm.get("tick_share_stall_below_1ms"))
        check("T4 timeline FULL", (sm["graph_modes_steady"] or {}).get("FULL", 0) > 0, sm.get("graph_modes_steady"))
        check("T4 timeline per-rank busy/idle", all("busy" in v_ and "idle" in v_ for v_ in sm["per_rank"].values()) and len(sm["per_rank"]) == 4)
        check("T4 timeline host fields", sm["stages"]["stage0"]["host_total"] is not None and sm["stages"]["stage0"]["nccl_peer_wait"] is not None)
        print("  timeline: parity", sm.get("parity_verdict"), "| dependency", sm.get("dependency_verdict"))
    except Exception:
        check("T4 timeline analyze on B1", False, traceback.format_exc())


# ------------------------------------------------------------------ T5
def t5():
    b = {"8k@8#1": win(250.0), "8k@8#2": win(255.0)}
    s = N.speed_compare(b, {"8k@8#1": win(300.0), "8k@8#2": win(305.0)})
    check("T5 pass", s["status"] == "PASS" and abs(s["gain"] - 0.198) < 0.01, s)
    s = N.speed_compare(b, {"8k@8#1": win(256.0), "8k@8#2": win(257.0)})
    check("T5 below 2 % fails", s["status"] == "FAIL", s)
    wide = {"8k@8#1": win(240.0), "8k@8#2": win(270.0)}                  # spread 11.8 %
    check("T5 spread gate", N.speed_compare(wide, {"8k@8#1": win(280.0), "8k@8#2": win(280.0)})["status"] == "FAIL")
    check("T5 B one window -> inconclusive", N.speed_compare({"8k@8#1": win(250.0)}, b)["status"] == "INCONCLUSIVE_B_SPREAD_UNKNOWN")
    s = N.speed_compare(b, {"8k@8#1": win(400.0, parity="IMBALANCED"), "8k@8#2": win(260.0)})
    check("T5 imbalanced excluded -> one X window is not enough", s["x_valid"] == ["8k@8#2"] and s["status"] == "INCONCLUSIVE_X_WINDOWS", s)
    # GPT audit 2026-11-01: B 100 / 101, X 160 / ERROR was PASS +59 % -> now INCONCLUSIVE_X_WINDOWS
    s = N.speed_compare({"8k@8#1": win(100.0), "8k@8#2": win(101.0)}, {"8k@8#1": win(160.0), "8k@8#2": win(0.0, status="ERROR")})
    check("T5 X one valid window (GPT case) -> INCONCLUSIVE_X_WINDOWS", s["status"] == "INCONCLUSIVE_X_WINDOWS" and "gain" not in s, s)
    s = N.speed_compare({"8k@8#1": win(100.0), "8k@8#2": win(101.0)}, {"8k@8#1": win(160.0), "8k@8#2": win(158.0)})
    check("T5 both X windows valid -> PASS", s["status"] == "PASS", s)
    check("T5 no X window", N.speed_compare(b, {"8k@8#1": win(300.0, status="INCONCLUSIVE_PREEMPTION")})["status"] == "INCONCLUSIVE_NO_X_WINDOW")
    w = win(300.0)
    w["preemptions"] = 1
    check("T5 preemption invalid", not N.window_ok(w)[0])
    w = win(300.0)
    w["running_eq_level"] = False
    check("T5 running != level invalid", not N.window_ok(w)[0])
    check("T5 side 40k", N.side_window({"40k@8": win(250.0)}, {"40k@8": win(275.0)})["gain"] == 0.1)
    a = plan()
    check("T5 fits", N.arm_fits(18 * 60 + 300, a["S"]) and not N.arm_fits(18 * 60 + 299, a["S"]))
    check("T5 B fits", N.arm_fits(27 * 60, a["B"]) and not N.arm_fits(26.9 * 60, a["B"]))
    check("T5 body cap", N.body_cap_sec(14, 3600) == 840 and N.body_cap_sec(14, 900) == 600 and N.body_cap_sec(14, 100) == 60)
    # the priority walk at 95 min: setup 7, B 21, every later arm 18.5 -> QD still starts; at 23 min per arm it does not
    for per, expect in ((18.5, ["S", "QP", "QD"]), (23.0, ["S", "QP"])):
        left, ran = (95 - 7 - 21) * 60.0, []
        for k in ("S", "QP", "QD"):
            if N.arm_fits(left, a[k]):
                ran.append(k)
                left -= per * 60
        check(f"T5 priority walk {per} min/arm", ran == expect, ran)


# ------------------------------------------------------------------ T6
def t6():
    stalled, fixed = stall_rec(21.5, 0.0), stall_rec(0.05, 1.0)
    B = arm_rec("B", stalled)
    V = lambda X, k="S": N.arm_verdict(B, X, k)["verdict"]
    check("T6 STALL_FIXED_WIN", V(arm_rec("S", fixed, tok=(300.0, 302.0))) == "STALL_FIXED_WIN")
    check("T6 STALL_FIXED_NO_SPEED", V(arm_rec("S", fixed, tok=(256.0, 257.0))) == "STALL_FIXED_NO_SPEED")
    check("T6 STALL_MOVED node", V(arm_rec("S", stall_rec(20.0, 0.0, node=300))) == "STALL_MOVED")
    check("T6 STALL_MOVED release", V(arm_rec("S", stall_rec(20.0, 0.0, rel="ncclDevKernel_SendRecv x"))) == "STALL_MOVED")
    check("T6 STALL_MOVED to stage 1 (QD risk)", V(arm_rec("QD", stall_rec(0.05, 1.0, s1_median=12.0), tok=(300.0, 300.0)), "QD") == "STALL_MOVED")
    check("T6 STALL_REDUCED", V(arm_rec("S", stall_rec(8.0, 0.2))) == "STALL_REDUCED")
    check("T6 NO_EFFECT", V(arm_rec("S", stall_rec(21.0, 0.0))) == "NO_EFFECT")
    check("T6 CORRECTNESS_FAIL", V(arm_rec("S", fixed, oracle_vs="FAIL_HIGH_MARGIN")) == "CORRECTNESS_FAIL")
    check("T6 CORRECTNESS_FAIL gate", V(arm_rec("QP", fixed, quality_gate="FAIL", oracle_vs="EXACT"), "QP") == "CORRECTNESS_FAIL")
    check("T6 NEAR_TIE ok", V(arm_rec("S", fixed, tok=(300.0, 302.0), oracle_vs="NEAR_TIE")) == "STALL_FIXED_WIN")
    check("T6 NOT_ENGAGED env", V(arm_rec("S", fixed, env_readback={"ok": False, "why": "x"})) == "NOT_ENGAGED")
    check("T6 NOT_ENGAGED status", V(arm_rec("S", None, status="NOT_ENGAGED", env_readback={"ok": False, "definitive": True}, oracle_vs=None,
                                              quality_gate=None, correctness="INCONCLUSIVE_NO_ORACLE")) == "NOT_ENGAGED")
    check("T6 Q NOT_ENGAGED readback", V(arm_rec("QP", fixed, q_readback={"label": "NOT_ENGAGED"}), "QP") == "NOT_ENGAGED")
    check("T6 Q NOT_INTEGRATED readback", V(arm_rec("QD", fixed, q_readback={"label": "NOT_INTEGRATED"}), "QD") == "NOT_INTEGRATED")
    check("T6 Q missing readback", V(arm_rec("QP", fixed, q_readback=None), "QP") == "NOT_ENGAGED")
    fb = {"label": "ENGAGED_FALLBACK", "detail": {"defer_per_rank": {"11": {"exec_share": 0.5}}}}
    check("T6 Q-defer ENGAGED_FALLBACK -> its own verdict (not engaged)",
          V(arm_rec("QD", fixed, tok=(300.0, 302.0), q_readback=fb), "QD") == "ENGAGED_FALLBACK")
    check("T6 latest read-back wins: mid ENGAGED, final FALLBACK -> ENGAGED_FALLBACK",
          V(arm_rec("QD", fixed, tok=(300.0, 302.0), q_readback_end={"label": "ENGAGED"}, q_readback_final=fb), "QD") == "ENGAGED_FALLBACK")
    check("T6 latest read-back wins: mid FALLBACK, end ENGAGED -> measured",
          V(arm_rec("QD", fixed, tok=(300.0, 302.0), q_readback=fb, q_readback_end={"label": "ENGAGED"}), "QD") == "STALL_FIXED_WIN")
    check("T6 final NOT_INTEGRATED wins over an ENGAGED end",
          V(arm_rec("QD", fixed, q_readback_end={"label": "ENGAGED"}, q_readback_final={"label": "NOT_INTEGRATED"}), "QD") == "NOT_INTEGRATED")
    v = N.verdict({"B": arm_rec("B", stalled), "S": arm_rec("S", stall_rec(21.0, 0.0)),
                   "QP": arm_rec("QP", stall_rec(8.0, 0.2)), "QD": arm_rec("QD", fixed, tok=(320.0, 322.0), q_readback_final=fb)})
    check("T6 ENGAGED_FALLBACK is never the best arm", v["stall_best_arm"] == "QP" and v["best_arm"] is None
          and v["arms"]["QD"]["verdict"] == "ENGAGED_FALLBACK" and not v["throughput_eligible"]["QD"]["ok"], v)
    check("T6 NOT_INTEGRATED skip", V({"status": "NOT_INTEGRATED", "why": "PATCH_PPACK_PRE_SHA_MISMATCH", "not_integrated": True}, "QP") == "NOT_INTEGRATED")
    check("T6 SKIPPED_BUDGET", V({"status": "SKIPPED_BUDGET", "why": "left"}, "QD") == "SKIPPED_BUDGET")
    check("T6 not run", V(None, "QD") == "SKIPPED_BUDGET")
    check("T6 SKIPPED_DEPENDENCY", V({"status": "SKIPPED_DEPENDENCY", "why": "B"}, "S") == "SKIPPED_DEPENDENCY")
    check("T6 pin mismatch", V(arm_rec("S", fixed, pin=1)) == "INCONCLUSIVE_PIN_MISMATCH")
    check("T6 imbalanced profile", V(arm_rec("S", stall_rec(0.05, 1.0, parity="IMBALANCED"))) == "INCONCLUSIVE_IMBALANCED")
    check("T6 inconclusive oracle", V(arm_rec("S", fixed, oracle_vs="INCONCLUSIVE_NO_ORACLE")) == "INCONCLUSIVE_ORACLE")
    nb = arm_rec("B", stall_rec(0.1, 1.0))
    check("T6 base no stall", N.arm_verdict(nb, arm_rec("S", fixed), "S")["verdict"] == "INCONCLUSIVE_BASE_NO_STALL")
    check("T6 base gate", N.arm_verdict(arm_rec("B", stalled, quality_gate="FAIL"), arm_rec("S", fixed), "S")["verdict"] == "INCONCLUSIVE_BASE_GATE")
    check("T6 no X trace", V(arm_rec("S", {"error": "NO_TRACE_FILE"})) == "INCONCLUSIVE_NO_TRACE")
    v = N.verdict({"B": B, "S": arm_rec("S", stall_rec(21.0, 0.0)), "QP": arm_rec("QP", fixed, tok=(290.0, 292.0)),
                   "QD": arm_rec("QD", fixed, tok=(305.0, 306.0))})
    check("T6 best arm = highest gain among WIN", v["best_arm"] == "QD" and v["best_verdict"] == "STALL_FIXED_WIN" and v["verdict"] == "QD:STALL_FIXED_WIN", v)
    v = N.verdict({"B": B, "S": arm_rec("S", fixed, tok=(256.0, 256.0)), "QP": arm_rec("QP", stall_rec(8.0, 0.2)),
                   "QD": {"status": "SKIPPED_BUDGET"}})
    check("T6 stall_best = NO_SPEED over REDUCED, no throughput winner", v["stall_best_arm"] == "S" and v["best_arm"] is None
          and v["verdict"] == "NO_THROUGHPUT_WINNER" and v["arms"]["QD"]["verdict"] == "SKIPPED_BUDGET", v)
    v = N.verdict({"B": B, "S": arm_rec("S", stall_rec(21.0, 0.0))})
    check("T6 none fixed", v["best_arm"] is None and v["verdict"] == "NO_THROUGHPUT_WINNER" and v["stall_best_verdict"] == "NO_ARM_FIXED"
          and v["arms"]["QP"]["verdict"] == "SKIPPED_BUDGET", v)
    # GPT audit 2026-11-01: S removes the stall but is 8 % slower, QP reduces it +12 %, QD reduces it +5 % -> best = QP
    red = stall_rec(8.0, 0.2)
    v = N.verdict({"B": B, "S": arm_rec("S", fixed, tok=(235.0, 236.0)), "QP": arm_rec("QP", red, tok=(286.0, 287.0)),
                   "QD": arm_rec("QD", red, tok=(268.0, 269.0))})
    check("T6 GPT case: throughput winner QP over a slower full fix",
          v["best_arm"] == "QP" and v["best_verdict"] == "STALL_REDUCED" and v["stall_best_arm"] == "S"
          and v["arms"]["S"]["verdict"] == "STALL_FIXED_NO_SPEED", v)
    v = N.verdict({"B": B, "S": arm_rec("S", fixed, tok=(235.0, 236.0)), "QP": arm_rec("QP", red, tok=(240.0, 241.0)),
                   "QD": arm_rec("QD", red, tok=(250.0, 250.0))})
    check("T6 every arm slower -> best_arm None", v["best_arm"] is None and v["verdict"] == "NO_THROUGHPUT_WINNER", v)
    qp = arm_rec("QP", red, tok=(286.0, 287.0))
    del qp["steady"]["40k@8"]
    v = N.verdict({"B": B, "S": arm_rec("S", stall_rec(21.0, 0.0)), "QP": qp, "QD": arm_rec("QD", red, tok=(268.0, 269.0))})
    check("T6 40k window missing -> not eligible, next winner QD", v["best_arm"] == "QD" and not v["throughput_eligible"]["QP"]["ok"], v)
    qd = arm_rec("QD", red, tok=(286.0, 287.0))
    qd["steady"]["40k@8"] = win(200.0)
    v = N.verdict({"B": B, "QD": qd})
    check("T6 40k regression -> not eligible", v["best_arm"] is None and "regression" in v["throughput_eligible"]["QD"]["why"], v)
    s = arm_rec("S", stall_rec(21.0, 0.0), tok=(286.0, 287.0))
    v = N.verdict({"B": B, "S": s})
    check("T6 NO_EFFECT stall but faster -> throughput winner", v["best_arm"] == "S" and v["best_verdict"] == "NO_EFFECT", v)
    v = N.verdict({"B": {"status": "NO_GO_QUALITY", "quality_gate": "FAIL"}})
    check("T6 base failed", v["verdict"] == "INCONCLUSIVE_BASE_GATE", v)
    check("T6 verdict JSON", json.loads(json.dumps(v)) == v)
    # GPT audit 2026-11-01: an abnormal final status never gets a positive verdict; B must be DONE and BALANCED
    for bad in ("BODY_ERROR", "LAUNCH_FAILED", "TIMEOUT", "ERROR"):
        check(f"T6 B status {bad} -> base not ok", not N.base_check(dict(arm_rec("B", stalled), status=bad))[0])
        vv = N.arm_verdict(B, dict(arm_rec("S", fixed, tok=(300.0, 302.0)), status=bad), "S")["verdict"]
        check(f"T6 arm status {bad} -> INCONCLUSIVE_ARM_FAILED", vv == "INCONCLUSIVE_ARM_FAILED", vv)
    vv = N.arm_verdict(B, dict(arm_rec("S", fixed, oracle_vs="FAIL_HIGH_MARGIN"), status="BODY_ERROR"), "S")["verdict"]
    check("T6 a real correctness failure still wins over the failed status", vv == "CORRECTNESS_FAIL", vv)
    bi = arm_rec("B", stall_rec(21.5, 0.0, parity="IMBALANCED"))
    check("T6 B imbalanced -> INCONCLUSIVE_BASE_IMBALANCED", N.base_check(bi)[1] == "INCONCLUSIVE_BASE_IMBALANCED", N.base_check(bi))


# ------------------------------------------------------------------ T7
def t7(tmp, merlin_src):
    a = plan()
    calls = []
    seq = N.PatchSeq(lambda: calls.append("apply") or {"ok": True}, lambda: calls.append("revert") or {"ok": True})
    res = [seq.before_arm(a[k])[0] for k in ("B", "S", "QP", "QD")]
    check("T7 seq all ok", res == [True] * 4 and calls == ["apply"], (res, calls))
    f = seq.finish()
    check("T7 reverted at finish", calls == ["apply", "revert"] and f["state"] == "reverted", (calls, f))
    check("T7 finish twice no-op", seq.finish()["state"] == "reverted" and calls == ["apply", "revert"])
    calls = []

    def bad_apply():
        calls.append("apply")
        raise RuntimeError("PATCH_PPACK_PRE_SHA_MISMATCH")
    seq = N.PatchSeq(bad_apply, lambda: calls.append("revert"))
    r_qp, r_qd = seq.before_arm(a["QP"]), seq.before_arm(a["QD"])
    check("T7 apply fails -> QP + QD not integrated", not r_qp[0] and not r_qd[0] and "NOT_INTEGRATED" in r_qp[1] and "NOT_INTEGRATED" in r_qd[1])
    check("T7 apply tried once, never reverted", calls == ["apply"] and seq.finish()["state"] == "failed", calls)
    calls = []
    seq = N.PatchSeq(lambda: calls.append("apply"), lambda: calls.append("revert"))
    seq.before_arm(a["QP"])
    check("T7 non-Q arm after Q -> revert first", seq.before_arm(a["S"]) == (True, "pristine") and calls == ["apply", "revert"])
    calls = []

    def bad_revert():
        raise RuntimeError("REVERT_PPACK_SHA_MISMATCH")
    seq = N.PatchSeq(lambda: calls.append("apply"), bad_revert)
    seq.before_arm(a["QD"])
    fin = seq.finish()
    check("T7 revert failure recorded", fin["state"] == "revert_failed" and "REVERT" in (fin["error"] or ""), fin)
    seq = N.PatchSeq(lambda: calls.append("apply"), lambda: None)
    check("T7 no Q arm -> no apply", seq.before_arm(a["B"])[0] and seq.before_arm(a["S"])[0] and seq.finish()["state"] == "pristine")
    # the real nb4_ppack apply / revert on a COPY of Merlin ad42e37's two files
    if not merlin_src or not Path(merlin_src).is_dir():
        print("SKIP T7 real patch (no merlin-src)")
        return
    try:
        import nb4_ppack as P
        tree = tmp / "tree"
        check("T7 real: the patch covers 3 files", set(P.PATCHES) == {P.PP_REL, P.MR_REL, P.GW_REL}, sorted(P.PATCHES))
        for rel in P.PATCHES:
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(merlin_src) / rel, tree / rel)
        seq = N.PatchSeq(lambda: P.apply_ppack_patch(tree, sys.executable), lambda: P.revert_ppack_patch(tree, sys.executable))
        ok_qp = seq.before_arm(a["QP"])
        st1 = P.verify_ppack_patch(tree)["state"]
        ok_qd = seq.before_arm(a["QD"])
        st2 = P.verify_ppack_patch(tree)["state"]
        fin = seq.finish()
        v = P.verify_ppack_patch(tree)
        check("T7 real apply before QP", ok_qp[0] and st1 == "applied", (ok_qp, st1))
        check("T7 real kept for QD", ok_qd == (True, "already applied") and st2 == "applied")
        check("T7 real revert -> pre-shas", fin["state"] == "reverted" and v["state"] == "not_applied"
              and all(v["files"][rel]["sha"] == p[0] for rel, p in P.PATCHES.items()), v)
        (tree / P.PP_REL).write_text((tree / P.PP_REL).read_text(encoding="utf-8") + "\n# foreign\n", encoding="utf-8")
        seq = N.PatchSeq(lambda: P.apply_ppack_patch(tree, sys.executable), lambda: P.revert_ppack_patch(tree, sys.executable))
        r = seq.before_arm(a["QP"])
        check("T7 real foreign tree refused (NOT_INTEGRATED)", not r[0] and "PRE_SHA_MISMATCH" in r[1], r)
        check("T7 real foreign tree untouched", P.verify_ppack_patch(tree)["files"][P.MR_REL]["sha"] == P.MR_PRE
              and P.verify_ppack_patch(tree)["files"][P.GW_REL]["sha"] == P.GW_PRE)
    except Exception:
        check("T7 real patch", False, traceback.format_exc())


# ------------------------------------------------------------------ T8 CELL flow with a stub boot runner
def export_cells():
    t = (HERE / "nb4s_code_export.py").read_text(encoding="utf-8")
    cells = t.split("\n\nif not RERUN:\n")
    return [cells[0]] + [textwrap.dedent(c) for c in cells[1:]]


class StubL:
    U = types.SimpleNamespace(urlopen=lambda *a, **k: None)

    @staticmethod
    def prom(port):
        return {}

    @staticmethod
    def gpu_mem():
        return []


def flow_ns(tmp, left_sec=95 * 60, sha_ok=True, sha_raises=False):
    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    ns = {"__name__": "nb4s_flow", "os": os, "sys": sys, "json": json, "math": __import__("math"), "time": __import__("time"),
          "re": re, "shutil": shutil, "subprocess": __import__("subprocess"), "hashlib": hashlib,
          "threading": __import__("threading"), "traceback": traceback, "Path": Path, "FAKE": None, "MODE": "FULL",
          "RERUN": False, "R": {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}, "errors": [], "order": []},
          "OUT": out, "TMP": tmp, "TRACES": tmp / "traces", "T_START": __import__("time").time(), "BUDGET": 95 * 60,
          "L": StubL, "KV_FP8": 2479838700, "EXT_FLAG": ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"],
          "PROFILER_CFG": "{}", "DIAG_ENV": {"VLLM_SERVER_DEV_MODE": "1"}, "NB1_EXTRA": {}, "NB1_ENV": {}, "NB1_KV": {},
          "ROWS": {"rc2_fp8": []}, "HELP": {}, "NVSMI_OK": False, "HOST_PHASE": {}, "PREFILL": {"tok_s": 2000.0, "measured": True},
          "ABORT": {"why": None}, "BOOT": {"deadline": None, "fired": None, "tag": "boot"}, "LEFT": [float(left_sec)]}
    ns["left"] = lambda: ns["LEFT"][0]
    ns["log"] = lambda *a: None
    ns["save"] = lambda: (out / "nb4s_results.json").write_text(json.dumps(ns["R"], default=str))
    ns["merlin_fn_server"] = lambda *a, **k: None
    ns["merlin_install"] = lambda: {"kv": {"MERLIN_TREE": str(tmp), "MERLIN_VENV": str(tmp)}}
    ns["apply_60490"] = lambda: {"ok": True}

    def sha_gate(t):
        if sha_raises:
            raise RuntimeError("sha thread died")
        return sha_ok, "ok" if sha_ok else "sha256 mismatch"
    ns["sha_gate"] = sha_gate
    (tmp / "traces").mkdir(exist_ok=True)
    cells = export_cells()
    os.environ["RC2_UNIT"] = "1"
    c4 = cells[3]
    exec(compile(c4, "<cell4>", "exec"), ns)          # RC2 / NB1 CELL 4: step(), with_servers(), keep_after_error() ...
    ns["ROWS"]["rc2_fp8"] = []                        # CELL 4 resets ROWS; NB1 CELL 6 would load RC2's recorded rows here
    exec(compile(cells[6], "<cell7>", "exec"), ns)    # NB4S CELL 7 (RC2_UNIT: definitions only, the main does not run)
    return ns


def runner_factory(ns, recs, minutes, order):
    def run(k):
        order.append(k)
        ns["LEFT"][0] -= minutes.get(k, 18) * 60
        r = recs.get(k)
        if isinstance(r, Exception):
            raise r
        return copy.deepcopy(r)
    return run


def t8(tmp):
    stalled, fixed = stall_rec(21.5, 0.0), stall_rec(0.05, 1.0)
    good = {"B": arm_rec("B", stalled), "S": arm_rec("S", stall_rec(21.0, 0.0)), "QP": arm_rec("QP", fixed, tok=(300.0, 301.0)),
            "QD": arm_rec("QD", fixed, tok=(290.0, 291.0))}
    # 1. everything runs
    try:
        ns = flow_ns(tmp / "f1")
        check("T8 plan valid in the cell", tuple(ns["ARMS"]) == ("B", "S", "QP", "QD") and ns["B3_PIN"] == 3553580524)
        check("T8 L.prom replaced", ns["L"].prom is ns["_prom_nb4s"])
        calls, order = [], []
        seq = ns["N"].PatchSeq(lambda: calls.append(("apply", list(order))) or {"ok": True}, lambda: calls.append(("revert", list(order))) or {"ok": True})
        ns["nb4s_main"](runner=runner_factory(ns, good, {"B": 21}, order), pseq=seq)
        v = json.loads((tmp / "f1/out/nb4s_verdict.json").read_text())
        check("T8 order B S QP QD", order == ["B", "S", "QP", "QD"], order)
        check("T8 patch applied right before QP, reverted after QD", calls == [("apply", ["B", "S"]), ("revert", ["B", "S", "QP", "QD"])], calls)
        check("T8 verdict best QP", v["best_arm"] == "QP" and v["arms"]["S"]["verdict"] == "NO_EFFECT", v.get("why"))
        check("T8 patch state in verdict", (v.get("patch") or {}).get("state") == "reverted")
        check("T8 step records DONE", all(ns["_arm_rec"](k)["status"] == "DONE" for k in ns["ARMS"]))
    except Exception:
        check("T8 flow 1", False, traceback.format_exc())
    # 2. B not clean
    try:
        ns = flow_ns(tmp / "f2")
        calls, order = [], []
        seq = ns["N"].PatchSeq(lambda: calls.append("apply"), lambda: calls.append("revert"))
        bad = dict(good, B=arm_rec("B", stalled, status="NO_GO_QUALITY", quality_gate="FAIL"))
        ns["nb4s_main"](runner=runner_factory(ns, bad, {}, order), pseq=seq)
        v = json.loads((tmp / "f2/out/nb4s_verdict.json").read_text())
        check("T8 B not clean -> only B ran, no patch", order == ["B"] and calls == [], (order, calls))
        check("T8 B not clean -> dependents skipped", all(ns["_arm_rec"](k)["status"] == "SKIPPED_DEPENDENCY" for k in ("S", "QP", "QD")))
        check("T8 B not clean verdict", v["verdict"] == "INCONCLUSIVE_BASE_GATE", v.get("verdict"))
    except Exception:
        check("T8 flow 2", False, traceback.format_exc())
    # 3. budget: 95 min, setup 7, B 21, S 30 -> QP needs 23 min, has 37 -> runs (30) -> QD has 7 -> skipped; patch reverted
    try:
        ns = flow_ns(tmp / "f3", left_sec=88 * 60)
        calls, order = [], []
        seq = ns["N"].PatchSeq(lambda: calls.append("apply"), lambda: calls.append("revert"))
        ns["nb4s_main"](runner=runner_factory(ns, good, {"B": 21, "S": 30, "QP": 30}, order), pseq=seq)
        check("T8 budget: QD skipped", order == ["B", "S", "QP"] and ns["_arm_rec"]("QD")["status"] == "SKIPPED_BUDGET", order)
        check("T8 budget: patch applied once, reverted", calls == ["apply", "revert"], calls)
        ns = flow_ns(tmp / "f3b", left_sec=88 * 60)
        calls, order = [], []
        seq = ns["N"].PatchSeq(lambda: calls.append("apply"), lambda: calls.append("revert"))
        ns["nb4s_main"](runner=runner_factory(ns, good, {"B": 21, "S": 45}, order), pseq=seq)
        check("T8 budget: no Q arm fits -> never patched", order == ["B", "S"] and calls == [] and
              ns["_arm_rec"]("QP")["status"] == ns["_arm_rec"]("QD")["status"] == "SKIPPED_BUDGET", (order, calls))
        v = json.loads((tmp / "f3b/out/nb4s_verdict.json").read_text())
        check("T8 budget verdict labels", v["arms"]["QP"]["verdict"] == "SKIPPED_BUDGET" and v["arms"]["QD"]["verdict"] == "SKIPPED_BUDGET")
    except Exception:
        check("T8 flow 3", False, traceback.format_exc())
    # 4. patch apply refused -> QP / QD NOT_INTEGRATED, no boot for them
    try:
        ns = flow_ns(tmp / "f4")
        order = []

        def refuse():
            raise RuntimeError("PATCH_PPACK_PRE_SHA_MISMATCH: foreign")
        seq = ns["N"].PatchSeq(refuse, lambda: None)
        ns["nb4s_main"](runner=runner_factory(ns, good, {}, order), pseq=seq)
        v = json.loads((tmp / "f4/out/nb4s_verdict.json").read_text())
        check("T8 apply refused -> no Q boot", order == ["B", "S"], order)
        check("T8 apply refused -> NOT_INTEGRATED", v["arms"]["QP"]["verdict"] == "NOT_INTEGRATED" and v["arms"]["QD"]["verdict"] == "NOT_INTEGRATED", v["arms"])
    except Exception:
        check("T8 flow 4", False, traceback.format_exc())
    # 5. S's runner raises -> S ERROR, QP / QD still run, verdict written
    try:
        ns = flow_ns(tmp / "f5")
        order = []
        seq = ns["N"].PatchSeq(lambda: None, lambda: None)
        ns["nb4s_main"](runner=runner_factory(ns, dict(good, S=RuntimeError("boom")), {}, order), pseq=seq)
        v = json.loads((tmp / "f5/out/nb4s_verdict.json").read_text())
        check("T8 S error recorded, plan continues", order == ["B", "S", "QP", "QD"] and ns["_arm_rec"]("S")["status"] == "ERROR", order)
        check("T8 S error verdict not a win", not v["arms"]["S"]["verdict"].startswith("STALL_FIXED"), v["arms"]["S"])
    except Exception:
        check("T8 flow 5", False, traceback.format_exc())
    # 6. an exception in the plan itself (sha gate) -> the finally still reverts and writes the verdict
    try:
        ns = flow_ns(tmp / "f6", sha_raises=True)
        seq = ns["N"].PatchSeq(lambda: None, lambda: None)
        try:
            ns["nb4s_main"](runner=runner_factory(ns, good, {}, []), pseq=seq)
            raised = False
        except RuntimeError:
            raised = True
        check("T8 plan exception propagates", raised)
        check("T8 verdict written anyway", (tmp / "f6/out/nb4s_verdict.json").is_file())
        ns = flow_ns(tmp / "f7", sha_ok=False)
        ns["nb4s_main"](runner=runner_factory(ns, good, {}, []), pseq=ns["N"].PatchSeq(lambda: None, lambda: None))
        check("T8 sha gate fail -> all SKIPPED_SHA_GATE", all(ns["_arm_rec"](k)["status"] == "SKIPPED_SHA_GATE" for k in ns["ARMS"]))
    except Exception:
        check("T8 flow 6", False, traceback.format_exc())
    # 7. the hist window helper and launch wrapper inside the cell
    try:
        ns = flow_ns(tmp / "f8")
        ns["_HIST"][1] = [(10.0, {16.0: 100.0, 32.0: 100.0}), (100.0, {16.0: 300.0, 32.0: 300.0})]
        check("T8 _hist_at", ns["_hist_at"](1, 10.2)[0] == 10.0 and ns["_hist_at"](1, 100.0)[0] == 100.0 and ns["_hist_at"](1, 50.0) is None)

        class Srv:
            env = dict(MERLIN_ENV, CUDA_DEVICE_MAX_CONNECTIONS="32")
            argv = ["vllm", "serve"] + BASE_EXTRA
        ns["_merlin_fn_server_nb1"] = lambda *a: Srv()
        ns["ARM_OF"]["nb4s_s"] = ns["ARMS"]["S"]
        check("T8 launch wrapper ok", ns["merlin_fn_server"]("nb4s_s", "fp8", 8021, 1) is not None)
        ns["ARM_OF"]["nb4s_b"] = ns["ARMS"]["B"]
        s_b = ns["merlin_fn_server"]("nb4s_b", "fp8", 8011, 1)
        check("T8 launch wrapper: an image-inherited watched key is removed for B (recorded)",
              "CUDA_DEVICE_MAX_CONNECTIONS" not in s_b.env and ns["NB4S"]["launch"]["nb4s_b"]["popped_unplanned_env"] == {"CUDA_DEVICE_MAX_CONNECTIONS": "32"})
        Srv.env = dict(MERLIN_ENV)
        check("T8 launch wrapper refuses S without the flag", raises(lambda: ns["merlin_fn_server"]("nb4s_s", "fp8", 8021, 1)))
        Srv.env = dict(MERLIN_ENV, NCCL_GRAPH_MIXING_SUPPORT="0")
        check("T8 launch wrapper refuses never-do", raises(lambda: ns["merlin_fn_server"]("nb4s_b", "fp8", 8011, 1)))
        log_ok = "\n".join(f'(Worker_PP{r // 2}_TP{r % 2} pid={100 + r}) NB4_PPQ ' + json.dumps({"event": "init", "pid": 100 + r, "ppack": True,
                                                                                               "ppdefer": False, "is_last_rank": r >= 2}) for r in range(4))
        log_ok += "\n" + "\n".join("NB4_PPQ " + json.dumps({"event": "stats", "pid": 100 + r, "key": "packed_send" if r >= 2 else "packed_recv",
                                                             "stats": {"packed_send": 5 if r >= 2 else 0, "packed_recv": 0 if r >= 2 else 5}})
                                     for r in range(4))
        srv = types.SimpleNamespace(log=lambda: log_ok)
        check("T8 q_readback ENGAGED", ns["q_readback"](srv, ns["ARMS"]["QP"])["label"] == "ENGAGED")
        srv3 = types.SimpleNamespace(log=lambda: "\n".join(log_ok.splitlines()[:3]))
        check("T8 q_readback 3 ranks NOT_INTEGRATED", ns["q_readback"](srv3, ns["ARMS"]["QP"])["label"] == "NOT_INTEGRATED")
        check("T8 q_readback B None", ns["q_readback"](srv, ns["ARMS"]["B"]) is None)

        def qd_log(exec_n, recv_n):
            ls = [f'(Worker_PP{r // 2}_TP{r % 2} pid={10 + r}) NB4_PPQ ' + json.dumps(
                {"event": "init", "ppack": False, "ppdefer": r < 2, "is_last_rank": r >= 2, "pid": 10 + r}) for r in range(4)]
            for r in (0, 1):
                s = {"defer_park": 1000, "flush_exec": exec_n, "flush_empty": 1000 - exec_n - recv_n, "flush_recv": recv_n,
                     "flush_consume": 0}
                ls += ["NB4_PPQ " + json.dumps({"event": "stats", "key": k, "pid": 10 + r, "stats": s}) for k in ("defer_park", "flush_exec")]
            return "\n".join(ls)
        check("T8 q_readback QD ENGAGED", ns["q_readback"](types.SimpleNamespace(log=lambda: qd_log(980, 5)), ns["ARMS"]["QD"])["label"] == "ENGAGED")
        r = ns["q_readback"](types.SimpleNamespace(log=lambda: qd_log(600, 300)), ns["ARMS"]["QD"])
        check("T8 q_readback QD ENGAGED_FALLBACK", r["label"] == "ENGAGED_FALLBACK", r)
        exp = (HERE / "nb4s_code_export.py").read_text(encoding="utf-8")
        check("T8 mid-arm: ENGAGED_FALLBACK does not stop the arm; the final read-back is taken after the server stopped",
              'not in ("ENGAGED", "ENGAGED_FALLBACK"):' in exp and 'o["q_readback_final"] = q_readback(srv, arm)' in exp)
    except Exception:
        check("T8 flow 7", False, traceback.format_exc())
    finally:
        os.environ.pop("RC2_UNIT", None)


# ------------------------------------------------------------------ T9
def t9():
    nbp = HERE.parent / "Q27-XHIGH-E25_NB4S_STALL.ipynb"
    exp = (HERE / "nb4s_code_export.py").read_text(encoding="utf-8")
    nb = json.loads(nbp.read_text(encoding="utf-8"))
    check("T9 export == notebook cells", "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code") == exp)
    check("T9 notebook < 400 kB", nbp.stat().st_size < 400_000, nbp.stat().st_size)
    check("T9 LF only", b"\r" not in nbp.read_bytes().replace(b"\\r", b""))
    sha = lambda p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()
    check("T9 lib sha embedded", f'NB4S_LIB_SHA = "{sha("nb4s_lib.py")}"' in exp)
    check("T9 timeline sha embedded (current file)", f'TL_SHA = "{sha("nb4_timeline.py")}"' in exp, sha("nb4_timeline.py"))
    check("T9 ppack sha embedded (current file)", f'PPACK_SHA = "{sha("nb4_ppack.py")}"' in exp)
    blobs = re.findall(r'zlib\.decompress\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)\)', exp)
    srcs = [zlib.decompress(base64.b64decode(b)).decode("utf-8") for b in blobs]
    check("T9 blobs == the files", len(srcs) == 2 and srcs[0].encode() == (HERE / "nb4_timeline.py").read_bytes()
          and srcs[1].encode() == (HERE / "nb4_ppack.py").read_bytes())
    line = next(l for l in exp.splitlines() if l.strip().startswith("NB1_EXT_SRC = "))
    ext = json.loads(line.strip()[len("NB1_EXT_SRC = "):])
    ns = {}
    exec(ext, ns)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
    try:
        r = ns["NB1WorkerExt"]().nb4s_env("CUDA_DEVICE_MAX_CONNECTIONS,NB4_PPACK")
    finally:
        os.environ.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
    check("T9 extension nb4s_env", r["env"] == {"CUDA_DEVICE_MAX_CONNECTIONS": "32"} and "pid" in r, r)
    check("T9 extension keeps NB1 methods", hasattr(ns["NB1WorkerExt"], "nb1_cache_info") and hasattr(ns["NB1WorkerExt"], "nb1_routing"))
    check("T9 budget 95 min", 'or 95 * 60)' in exp)
    check("T9 no S8 arm left", '"S8"' not in exp and "nb4s_s8" not in exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-traces", default=r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb1-capacity-r2/nb1/traces/profile_b1")
    ap.add_argument("--merlin-src", default=r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
    ap.add_argument("--nb1-results", default=r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb1-capacity-r2/nb1/nb1_results.json")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="nb4s_units_"))
    try:
        for name, fn in (("T1", t1), ("T1r", lambda: t1_real_env(a.nb1_results)), ("T2", t2), ("T3", t3),
                         ("T4", lambda: t4(tmp, a.b1_traces)), ("T5", t5), ("T6", t6), ("T7", lambda: t7(tmp, a.merlin_src)),
                         ("T8", lambda: t8(tmp)), ("T9", t9)):
            try:
                fn()
            except Exception:
                check(f"{name} no exception", False, traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"{n_ok}/{len(RESULTS)} PASS")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
