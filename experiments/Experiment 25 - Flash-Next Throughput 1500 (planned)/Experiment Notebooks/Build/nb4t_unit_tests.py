"""E25 NB4T quick unit tests (CPU, stdlib only; no GPU, no server, no local end-to-end).

  python nb4t_unit_tests.py [--merlin-src DIR] [--fixture FILE] [--nb1-results FILE]

T1  arm plan: Q / I8 / P26 / P27 / C12, exactly one change each, never-do list, the launch check (watched env incl. the
    partition, the --speculative-config draft head, the last --max-num-seqs, --language-model-only, the pin) on synthetic
    and on the REAL NB4S QD launch env / argv (fixture)
T2  env read-back parser (RPC) + /proc environ parser
T3  server-log config read-back on the REAL NB4S QD log lines (fixture) + synthetic I8 / P26 / C12 lines; capacity facts
    (pool blocks, F, per-stage bytes, unused pin) on the real QD record; the KV demand estimate (C12 40k@12 vs 215 blocks)
T4  parity: parity_of / parity_compare (same label, same split, PARITY_DIFFERS, UNRESOLVED) / stall_compare; synthetic traces
    through graph_stalls; histogram parity at 8 and 12 streams
T5  speed comparison with paired keys (Q 8k@8 vs C12 8k@12), window_ok without a parity gate, budget walk at 120 min
T6  per-arm verdicts (every label), base never rejected for parity, the throughput winner (C12 excluded), P27 dependency;
    the REAL NB4S records re-judged with the NB4T rule (B as base, QD as the arm: WIN instead of INCONCLUSIVE_BASE_IMBALANCED)
T7  PatchSeq (applied before Q, kept, reverted) + the real nb4_ppack apply / revert on a COPY of Merlin's three files
T8  CELL flow: the notebook's CELL 4 + CELL 7 from nb4t_code_export.py with a stub boot runner: order, patch, budget,
    Q not clean, P26 OOM -> P27 skipped, apply refused, errors, the verdict always written; the launch wrapper; prompts
T9  notebook facts: export == notebook, < 400 kB, LF, embedded module shas, CELL 0 = E18 R2 cell 1, NB4S cells 3-6 verbatim,
    the marked edits (budget 120, identity fallback, build knob before build())
T10 rerun: the notebook's code cells executed in a subprocess with KAGGLE_IS_COMPETITION_RERUN=1 on a fake ARC root ->
    only CELL 0's identity submission.json; CELL 0 alone in commit mode writes the same file
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
import subprocess
import sys
import tempfile
import textwrap
import traceback
import types
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb4t_lib as N  # noqa: E402

RESULTS = []
NB_NAME = "Q27-XHIGH-E25_NB4T_LEVERS.ipynb"


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    if not cond:
        print("FAIL", name, str(detail)[:700])


def raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


# ------------------------------------------------------------------ helpers
BASE_EXTRA = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}"]
BASE_ENV = {"VLLM_SERVER_DEV_MODE": "1"}
QENV = {"NB4_PPDEFER": "1"}
PIN = 3553580524
MERLIN_ENV = {"KV_BYTES": str(PIN), "NCCL_P2P_DISABLE": "1", "NCCL_VERSION": "2.25.1-1", "P2P": "off",
              "VLLM_PP_LAYER_PARTITION": "25,23", "VLLM_SERVER_DEV_MODE": "1", "VLLM_SKIP_P2P_CHECK": "1",
              "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}
SPEC_Q = '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
SPEC_I8 = '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","draft_lm_head_quant":"int8"}'


def plan():
    return N.arm_plan(BASE_EXTRA, BASE_ENV, PIN, q_env=QENV)


def merlin_argv(arm, spec=SPEC_Q, kv=PIN):
    """What Merlin's build() + NB1's extra produce: the profile args (spec, --max-num-seqs 8, the pin) then the arm's extra."""
    return (["vllm", "serve", "/tmp/fn_merlin", "--speculative-config", spec, "--max-num-seqs", "8", "--kv-cache-memory-bytes", str(kv),
             "--enable-prefix-caching"] + list(arm["extra"]))


def full_env(arm):
    return dict(MERLIN_ENV, **arm["env"])


def load_fixture(p):
    if not p or not Path(p).is_file():
        return None
    return json.loads(Path(p).read_text(encoding="utf-8"))


def kern(ts, dur, corr, stream=7, name="void k()", **a):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "tid": stream,
            "args": dict({"correlation": corr, "stream": stream}, **a)}


def synth_trace(rank, n_graphs=6, nodes=600, gap_us=0.0, period=50000.0, gap_at=172, gen=(4, 4)):
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
            ev.append(kern(cur, 10.0, corr, 7, "void k%d()" % i, **{"graph id": 5}))
            cur += 10.0
        t += period
    return {"traceEvents": ev, "distributedInfo": {"rank": rank}}


def write_traces(d, gap0=0.0, gap1=0.0, gen=(4, 4)):
    d.mkdir(parents=True, exist_ok=True)
    for rank in range(4):
        pp, tp = rank // 2, rank % 2
        with gzip.open(d / f"dp0_pp{pp}_tp{tp}_dcp0_ep{tp}_rank{rank}.1.pt.trace.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(synth_trace(rank, gap_us=gap0 if pp == 0 else gap1, gen=gen), fh)
    return d


def win(tok, level=8, status="STEADY", parity="IMBALANCED", tick=41.0, pre=0, run_eq=True):
    return {"status": status, "tok_s": tok, "tick_ms": tick, "running_eq_level": run_eq, "preemptions": pre, "level": level,
            "parity": {"label": parity}, "kv_usage_max": 0.6, "mean_acceptance_length": 3.1}


def stall_rec(median=0.035, share=1.0, parity="IMBALANCED", split=None, s1_median=0.05):
    split = split or {"0": 7, "1": 1}
    return {"stage0": {"graphs": 76, "stall_ms_median": median, "share_below_1ms": share, "node_index_mode": None, "released_by_mode": None},
            "stage1": {"graphs": 76, "stall_ms_median": s1_median, "sendrecv_ms_per_tick_mean": 3.5, "pp_broadcast_send_ms_per_tick": 0.05},
            "parity": {"label": parity, "modal_requests": split}}


def tl_rec(c0=21.6, c1=33.0, tick=44.1):
    return {"tick_ms_mean": tick, "stages": {"stage0": {"compute": c0}, "stage1": {"compute": c1}}, "parity_verdict": "UNBALANCED"}


def arm_rec(key, tok=(305.0, 312.0), tok40=297.0, stall=None, **kw):
    a = plan()[key]
    n = a["streams"]
    k = N.window_keys(n)
    r = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS", "oracle_vs": "EXACT", "pin": PIN,
         "env_readback": {"ok": True, "why": "ok"}, "config_readback": {"ok": True, "why": "ok", "log": {"block_bytes_per_stage": [14680064, 16515072]}},
         "q_readback_final": {"label": "ENGAGED", "detail": {}}, "stall": stall or stall_rec(), "timeline": tl_rec(),
         "steady": {k[0]: win(tok[0], n), k[1]: win(tok[1], n), k[2]: {"status": None, "trace_dir": "/x"}, k[3]: win(tok40, n)},
         "pool": {"kv_tokens": 738142, "max_concurrency": 5.63}, "workers": {"num_gpu_blocks": [215], "free_mib_per_rank": [639, 639, 191, 191]},
         "free_after_ready_min": {"min": [639, 639, 191, 191]}, "free_end_min": {"min": [621, 621, 173, 173]}}
    r.update(kw)
    return r


# ------------------------------------------------------------------ T1
def t1(fx):
    a = plan()
    check("T1 order", tuple(a) == ("Q", "I8", "P26", "P27", "C12"))
    check("T1 plan valid", N.check_arm_plan(a))
    check("T1 Q env = base + NB4_PPDEFER=1", a["Q"]["env"] == {"VLLM_SERVER_DEV_MODE": "1", "NB4_PPDEFER": "1"} and a["Q"]["build_env"] == {})
    check("T1 Q expect", a["Q"]["expect"] == {"CUDA_DEVICE_MAX_CONNECTIONS": None, "NB4_PPACK": None, "NB4_PPDEFER": "1",
                                              "VLLM_PP_LAYER_PARTITION": "25,23"}, a["Q"]["expect"])
    check("T1 I8 build knob only", a["I8"]["build_env"] == {"DRAFT_HEAD": "int8"} and a["I8"]["env"] == a["Q"]["env"]
          and a["I8"]["extra"] == a["Q"]["extra"] and a["I8"]["expect_cfg"]["draft_head"] == "int8")
    check("T1 P26 / P27 env only", a["P26"]["env"]["VLLM_PP_LAYER_PARTITION"] == "26,22" and a["P27"]["env"]["VLLM_PP_LAYER_PARTITION"] == "27,21"
          and a["P26"]["extra"] == a["Q"]["extra"] and a["P26"]["expect"]["VLLM_PP_LAYER_PARTITION"] == "26,22")
    check("T1 C12 trailing seqs 12 + 12 streams", a["C12"]["extra"] == a["Q"]["extra"] + ["--max-num-seqs", "12"]
          and a["C12"]["streams"] == 12 and a["C12"]["windows"] == ("8k@12#1", "8k@12#2", "profile_8k@12", "40k@12"))
    check("T1 every arm patched + Q-defer", all(x["needs_patch"] and x["q_label"] == "Q-defer" and x["env"]["NB4_PPDEFER"] == "1" for x in a.values()))
    check("T1 levers / measure-only", [k for k in a if a[k]["lever"]] == ["I8", "P26", "P27"] and [k for k in a if a[k]["measure_only"]] == ["C12"])
    check("T1 --language-model-only in every arm", all("--language-model-only" in x["extra"] for x in a.values()))
    for label, mut in (("graph mixing 0", lambda p: p["I8"]["env"].update(NCCL_GRAPH_MIXING_SUPPORT="0")),
                       ("LL128", lambda p: p["Q"]["env"].update(NCCL_PROTO="LL128")),
                       ("P2P on", lambda p: p["P26"]["env"].update(P2P="on")),
                       ("COUNTERS", lambda p: p["C12"]["env"].update(COUNTERS="1")),
                       ("no async", lambda p: [x["extra"].append("--no-async-scheduling") for x in p.values()]),
                       ("prefix off", lambda p: [x["extra"].append("--no-enable-prefix-caching") for x in p.values()]),
                       ("Q without defer", lambda p: [x["env"].pop("NB4_PPDEFER") for x in p.values()]),
                       ("Q with pack", lambda p: p["Q"]["env"].update(NB4_PPACK="1")),
                       ("Q sets partition", lambda p: p["Q"]["env"].update(VLLM_PP_LAYER_PARTITION="25,23")),
                       ("Q build knob", lambda p: p["Q"]["build_env"].update(DRAFT_HEAD="int8")),
                       ("I8 int4", lambda p: (p["I8"]["build_env"].update(DRAFT_HEAD="int4"), p["I8"]["change"].update(value="int4"))),
                       ("I8 also env", lambda p: p["I8"]["env"].update(CUDA_DEVICE_MAX_CONNECTIONS="32")),
                       ("P26 also flag", lambda p: p["P26"]["extra"].append("--foo")),
                       ("P26 bad sum", lambda p: (p["P26"]["env"].update(VLLM_PP_LAYER_PARTITION="26,23"),
                                                  p["P26"]["change"].update(value="26,23"), p["P26"]["expect"].update(VLLM_PP_LAYER_PARTITION="26,23"))),
                       ("C12 two flags", lambda p: p["C12"]["extra"].extend(["--block-size", "2048"])),
                       ("C12 env", lambda p: p["C12"]["env"].update(VLLM_X="1")),
                       ("pin differs", lambda p: p["P27"].update(kv_bytes=1)),
                       ("unpatched arm", lambda p: p["C12"].update(needs_patch=False)),
                       ("no text-only", lambda p: [x["extra"].remove("--language-model-only") for x in p.values()]),
                       ("ports", lambda p: p["I8"].update(port=8011))):
        p = plan()
        mut(p)
        check(f"T1 refuses {label}", raises(lambda: N.check_arm_plan(p)))
    check("T1 partition_ok", N.partition_ok("26,22") and N.partition_ok("27,21") and not N.partition_ok("1,47")
          and not N.partition_ok("26,23") and not N.partition_ok("x") and not N.partition_ok("48"))
    # launch check on the full final env / argv
    specs = {"Q": SPEC_Q, "I8": SPEC_I8, "P26": SPEC_Q, "P27": SPEC_Q, "C12": SPEC_Q}
    for k in a:
        bad = N.launch_check(a[k], full_env(a[k]), merlin_argv(a[k], specs[k]), PIN)
        check(f"T1 launch_check {k} ok", bad == [], bad)
    check("T1 launch I8 without the head refused", N.launch_check(a["I8"], full_env(a["I8"]), merlin_argv(a["I8"], SPEC_Q), PIN) != [])
    check("T1 launch Q with an int8 head refused", N.launch_check(a["Q"], full_env(a["Q"]), merlin_argv(a["Q"], SPEC_I8), PIN) != [])
    check("T1 launch P26 with Merlin's partition refused", N.launch_check(a["P26"], dict(full_env(a["P26"]), VLLM_PP_LAYER_PARTITION="25,23"),
                                                                     merlin_argv(a["P26"]), PIN) != [])
    check("T1 launch Q without a partition refused", N.launch_check(a["Q"], {k: v for k, v in full_env(a["Q"]).items() if k != "VLLM_PP_LAYER_PARTITION"},
                                                                    merlin_argv(a["Q"]), PIN) != [])
    check("T1 launch C12 seqs 8 refused", N.launch_check(a["C12"], full_env(a["C12"]), merlin_argv(a["Q"]), PIN) != [])
    check("T1 launch Q seqs 12 refused", N.launch_check(a["Q"], full_env(a["Q"]), merlin_argv(a["C12"]), PIN) != [])
    check("T1 launch pin mismatch refused", N.launch_check(a["Q"], full_env(a["Q"]), merlin_argv(a["Q"], kv=PIN - 268435456), PIN) != []
          and N.launch_check(a["Q"], full_env(a["Q"]), merlin_argv(a["Q"], kv=PIN - 268435456), PIN - 268435456) == [])
    check("T1 launch never-do refused", N.launch_check(a["Q"], dict(full_env(a["Q"]), NCCL_GRAPH_MIXING_SUPPORT="0"), merlin_argv(a["Q"]), PIN) != [])
    check("T1 launch defer missing refused", N.launch_check(a["I8"], dict(MERLIN_ENV), merlin_argv(a["I8"], SPEC_I8), PIN) != [])
    check("T1 launch unparsable spec refused", N.launch_check(a["Q"], full_env(a["Q"]), merlin_argv(a["Q"], "{bad"), PIN) != [])
    check("T1 spec_from_argv last wins", N.spec_from_argv(["--speculative-config", SPEC_I8, "--speculative-config", SPEC_Q]).get("draft_lm_head_quant") is None)
    check("T1 last_value", N.last_value(["--max-num-seqs", "8", "x", "--max-num-seqs", "12"], "--max-num-seqs") == "12"
          and N.last_value(["--max-num-seqs"], "--max-num-seqs") is None)
    if not fx:
        print("SKIP T1 real NB4S QD launch (no fixture)")
        return
    mp = fx["merlin_plans"]["nb4s_qd"]
    env = dict(mp["env_keys"], **(fx["launch_env"].get("nb4s_qd") or {}))
    argv = ["vllm", "serve", "/tmp/fn_merlin"] + mp["argv_tail"]
    check("T1 real NB4S QD launch == NB4T Q (env + argv)", N.launch_check(a["Q"], env, argv, PIN) == [], N.launch_check(a["Q"], env, argv, PIN))
    i8_argv = [SPEC_I8 if x == SPEC_Q else x for x in argv]
    check("T1 real argv: I8 needs Merlin's int8 spec", N.launch_check(a["I8"], env, i8_argv, PIN) == [] and N.launch_check(a["I8"], env, argv, PIN) != [])
    check("T1 real argv: C12 = Q argv + --max-num-seqs 12", N.launch_check(a["C12"], env, argv + ["--max-num-seqs", "12"], PIN) == [])
    check("T1 real env: P26 = Q env with 26,22", N.launch_check(a["P26"], dict(env, VLLM_PP_LAYER_PARTITION="26,22"), argv, PIN) == []
          and N.launch_check(a["P26"], env, argv, PIN) != [])


def t1_real_nb1(nb1_results):
    if not nb1_results or not Path(nb1_results).is_file():
        print("SKIP T1 real NB1 R2 B3 env (no nb1_results.json)")
        return
    R = json.loads(Path(nb1_results).read_text(encoding="utf-8"))
    mp = R["setup"]["merlin_plans"]["nb1_b3_fp8_pin100"]
    a = plan()
    argv = ["vllm", "serve"] + mp["argv_tail"]
    for k in ("Q", "P26", "P27"):
        env = dict(mp["env_keys"], **a[k]["env"])
        bad = N.never_do_violations(env, argv)
        check(f"T1 real NB1 B3 env + {k} change: no never-do item", bad == [], bad)


# ------------------------------------------------------------------ T2
def t2():
    exp = plan()["P26"]["expect"]
    rows = [{"rank": r, "pid": 100 + r, "env": {"NB4_PPDEFER": "1", "VLLM_PP_LAYER_PARTITION": "26,22"}} for r in range(4)]
    v = N.parse_env_readback({"results": rows}, exp)
    check("T2 P26 ok", v["ok"] and not v["definitive"], v)
    bad = copy.deepcopy(rows)
    bad[3]["env"]["VLLM_PP_LAYER_PARTITION"] = "25,23"
    v = N.parse_env_readback({"results": bad}, exp)
    check("T2 one rank on 25,23 -> definitive", not v["ok"] and v["definitive"], v)
    check("T2 3 workers", not N.parse_env_readback({"results": rows[:3]}, exp)["ok"])
    check("T2 rpc error not definitive", not N.parse_env_readback({"error": "x"}, exp)["definitive"])
    dup = copy.deepcopy(rows)
    dup[1]["rank"] = 0
    check("T2 duplicate rank", not N.parse_env_readback({"results": dup}, exp)["ok"])
    qexp = plan()["Q"]["expect"]
    check("T2 Q ok with Merlin's partition", N.parse_env_readback({"results": [{"rank": r, "env": {"NB4_PPDEFER": "1", "VLLM_PP_LAYER_PARTITION": "25,23"}}
                                                                             for r in range(4)]}, qexp)["ok"])
    check("T2 Q with pack fails", not N.parse_env_readback({"results": [{"rank": r, "env": {"NB4_PPDEFER": "1", "NB4_PPACK": "1",
                                                                                          "VLLM_PP_LAYER_PARTITION": "25,23"}} for r in range(4)]}, qexp)["ok"])
    pe = N.parse_proc_environ({1: b"NB4_PPDEFER=1\0VLLM_PP_LAYER_PARTITION=26,22\0", 2: None}, {"NB4_PPDEFER": "1", "VLLM_PP_LAYER_PARTITION": "26,22"})
    check("T2 proc environ", pe["all_expected"] and pe["readable"] == 1 and pe["values"]["2"] == "UNREADABLE", pe)


# ------------------------------------------------------------------ T3
def t3(fx):
    a = plan()
    if fx:
        log = fx["logs"]["QD"]
        cr = N.config_readback(a["Q"], log, PIN)
        check("T3 real QD log: Q config read-back OK", cr["ok"] and not cr["definitive"], cr)
        check("T3 real QD log: stage layers 0-25 / 25-48", cr["log"]["stage_layers"] == {"0": [(0, 25)], "1": [(25, 48)]}, cr["log"]["stage_layers"])
        check("T3 real QD log: block bytes", cr["log"]["block_bytes_per_stage"] == [14680064, 16515072], cr["log"]["block_bytes_per_stage"])
        check("T3 real QD log: spec K3 probabilistic, no head", cr["log"]["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 3,
                                                                                                  "draft_sample_method": "probabilistic"})
        for k, why in (("I8", "spec draft_lm_head_quant"), ("P26", "stage layers"), ("C12", "max_num_seqs")):
            c = N.config_readback(a[k], log, PIN)
            check(f"T3 real QD log under {k}'s plan -> definitive contradiction", not c["ok"] and c["definitive"] and why in c["why"], c["why"])
        check("T3 real QD log, wrong pin -> definitive", N.config_readback(a["Q"], log, PIN - 1)["definitive"])
        # the I8 / P26 / C12 logs the GPU should print (synthetic edits of the real QD lines)
        i8 = log.replace("'draft_sample_method': 'probabilistic'}", "'draft_sample_method': 'probabilistic', 'draft_lm_head_quant': 'int8'}")
        i8 += ("\n(Worker_PP1_TP0_EP0 pid=9550) INFO 11-01 10:00:00 [draft_lm_head_quant.py:118] Draft lm_head: INT8 weight-only copy of the "
               "124160x2560 shard (309.5 MiB; the target keeps its torch.bfloat16 head)\n(Worker_PP1_TP1_EP1 pid=9551) INFO 11-01 10:00:00 "
               "[draft_lm_head_quant.py:118] Draft lm_head: INT8 weight-only copy of the 124160x2560 shard (309.5 MiB; the target keeps its "
               "torch.bfloat16 head)")
        c = N.config_readback(a["I8"], i8, PIN)
        check("T3 I8 log -> OK (spec + 2 INT8 build lines)", c["ok"] and len(c["log"]["draft_head_lines"]) == 2
              and {h["worker"] for h in c["log"]["draft_head_lines"]} == {"PP1_TP0", "PP1_TP1"}, c)
        check("T3 I8 log under Q's plan -> definitive (head built on a BF16 arm)", N.config_readback(a["Q"], i8, PIN)["definitive"])
        c = N.config_readback(a["I8"], "\n".join(l for l in i8.splitlines() if "Draft lm_head" not in l), PIN)
        check("T3 I8 spec but no build line -> not ok, not definitive", not c["ok"] and not c["definitive"], c)
        p26 = log.replace("(pipeline layers 0-25)", "(pipeline layers 0-26)").replace("(pipeline layers 25-48)", "(pipeline layers 26-48)")
        check("T3 P26 log -> OK", N.config_readback(a["P26"], p26, PIN)["ok"])
        check("T3 P26 log under P27's plan -> definitive", N.config_readback(a["P27"], p26, PIN)["definitive"])
        c12 = log.replace("'max_num_seqs': 8", "'max_num_seqs': 12")
        check("T3 C12 log -> OK", N.config_readback(a["C12"], c12, PIN)["ok"])
        check("T3 no non-default line -> not ok, not definitive", (lambda c: not c["ok"] and not c["definitive"])(
            N.config_readback(a["Q"], "\n".join(l for l in log.splitlines() if "non-default args" not in l), PIN)))
        qd = fx["arms"]["QD"]
        cf = N.capacity_facts(dict(qd, config_readback={"log": {"block_bytes_per_stage": [14680064, 16515072]}}))
        check("T3 real QD capacity: 215 blocks, F 7, PP1 binds", cf["num_gpu_blocks"] == [215] and cf["F_blocks"] == 7
              and cf["binding_stage"] == 1 and cf["max_concurrency_131072"] == 5.63, cf)
        check("T3 real QD capacity: PP0 leaves ~378 MiB of its pin unused", 370 <= cf["pin_unused_mib_per_stage"][0] <= 390
              and cf["pin_unused_mib_per_stage"][1] < 5, cf.get("pin_unused_mib_per_stage"))
        check("T3 real QD free after ready", cf["free_after_ready_min_mib"] == [639, 639, 191, 191], cf["free_after_ready_min_mib"])
        w40 = qd["steady"]["40k@8"]
        check("T3 real QD 40k@8 KV usage max 0.967 with 8 running", w40["kv_usage_max"] > 0.96 and w40["running_eq_level"] is True)
    else:
        print("SKIP T3 real QD log (no fixture)")
    check("T3 kv_demand", N.kv_demand(8, 40000, 5500, 7) == 8 * (12 + 7) and N.kv_demand(12, 8192, 2000, 7) == 12 * (3 + 7))
    check("T3 C12 40k@12 predicted over the 215-block pool", N.capacity_predict(215, 7, 12, 40000, 6000)["label"] == "OVER_POOL")
    check("T3 C12 8k@12 fits", N.capacity_predict(215, 7, 12, 8192, 3000)["label"] == "FITS")
    check("T3 Q 40k@8 at the edge-or-fits", N.capacity_predict(215, 7, 8, 40000, 5500)["label"] in ("FITS", "EDGE"))
    check("T3 predict unknown", N.capacity_predict(None, 7, 12, 40000, 6000)["label"] == "UNKNOWN")


# ------------------------------------------------------------------ T4
def t4(tmp, fx):
    Q = arm_rec("Q")
    same = arm_rec("I8", stall=stall_rec(split={"0": 1, "1": 7}))
    check("T4 same label (7/1 vs 1/7) -> COMPARED", N.parity_compare(Q, same)[0] == "COMPARED")
    bal = arm_rec("P26", stall=stall_rec(parity="BALANCED", split={"0": 4, "1": 4}))
    check("T4 IMBALANCED 7/1 vs BALANCED 4/4 -> PARITY_DIFFERS", N.parity_compare(Q, bal)[0] == "PARITY_DIFFERS")
    q55 = arm_rec("Q", stall=stall_rec(parity="IMBALANCED", split={"0": 6, "1": 4}))
    x55 = arm_rec("C12", stall=stall_rec(parity="BALANCED", split={"0": 4, "1": 6}))
    check("T4 different labels but the same split -> COMPARED (same modal split)",
          N.parity_compare(q55, x55) == ("COMPARED", N.parity_compare(q55, x55)[1]) and N.parity_compare(q55, x55)[1]["rule"] == "same modal split")
    un = arm_rec("I8", stall={"stage0": {"graphs": 10}, "parity": {"label": "UNRESOLVED"}})
    check("T4 unresolved -> PARITY_UNRESOLVED", N.parity_compare(Q, un)[0] == "PARITY_UNRESOLVED")
    sc = N.stall_compare(Q, same)
    check("T4 stall_compare COMPARED + STILL_FIXED + compute deltas", sc["status"] == "COMPARED" and sc["class"] == "STALL_STILL_FIXED"
          and sc["compute_delta_ms"] == {"stage0_compute_ms": 0.0, "stage1_compute_ms": 0.0, "tick_ms_mean": 0.0}, sc)
    p26 = arm_rec("P26", timeline=tl_rec(c0=23.0, c1=31.5, tick=40.2))
    sc = N.stall_compare(Q, p26)
    check("T4 P26 compute moved to stage 0", sc["compute_delta_ms"]["stage1_compute_ms"] == -1.5 and sc["compute_delta_ms"]["stage0_compute_ms"] == 1.4, sc)
    check("T4 PARITY_DIFFERS -> no stall classes", "class" not in N.stall_compare(Q, bal) and N.stall_compare(Q, bal)["status"] == "PARITY_DIFFERS")
    ret = arm_rec("I8", stall=stall_rec(median=12.0, share=0.2))
    check("T4 stall returned", N.stall_compare(Q, ret)["class"] == "STALL_RETURNED")
    mv = arm_rec("I8", stall=stall_rec(s1_median=9.0))
    check("T4 stall moved to stage 1", N.stall_compare(Q, mv)["class"] == "STALL_MOVED_STAGE1")
    check("T4 no trace", N.stall_compare(Q, arm_rec("I8", stall={"error": "NO_TRACE_FILE"}))["status"] == "NO_TRACE")
    po = N.parity_of(Q)
    check("T4 parity_of records profile + windows", po["profile_label"] == "IMBALANCED" and po["modal_split"] == {"0": 7, "1": 1}
          and set(po["windows"]) == {"8k@8#1", "8k@8#2", "40k@8"}, po)
    # synthetic traces through the parser: a fixed stage 0 (Q), 6 + 6 at 12 streams
    st = N.graph_stalls(write_traces(tmp / "syn_q", gen=(7, 1)))
    check("T4 synthetic Q: no stall, 7 + 1 IMBALANCED", st["stage0"]["share_below_1ms"] == 1.0 and st["parity"]["label"] == "IMBALANCED"
          and st["parity"]["modal_requests"] == {"0": 7, "1": 1}, st["parity"])
    st12 = N.graph_stalls(write_traces(tmp / "syn_c12", gen=(6, 6)))
    check("T4 synthetic C12: 6 + 6 BALANCED", st12["parity"]["label"] == "BALANCED")
    st_r = N.graph_stalls(write_traces(tmp / "syn_ret", gap0=20000.0))
    check("T4 synthetic stall returned", (st_r["stage0"]["stall_ms_median"] or 0) > 19.0 and st_r["stage0"]["share_below_1ms"] == 0.0)
    # histogram parity at 8 and 12 streams
    h0 = {8.0: 0, 16.0: 0, 32.0: 0, 64.0: 0, float("inf"): 0}
    h12 = {8.0: 0, 16.0: 0, 32.0: 300, 64.0: 300, float("inf"): 300}
    check("T4 hist 12 streams: 24 tokens in (16, 32] BALANCED", N.parity_from_hist(N.hist_delta(h0, h12), 12)["label"] == "BALANCED")
    h8 = {8.0: 10, 16.0: 100, 32.0: 300, 64.0: 300, float("inf"): 300}
    check("T4 hist 8 streams mostly 28 tokens -> IMBALANCED", N.parity_from_hist(N.hist_delta(h0, h8), 8)["label"] == "IMBALANCED")
    if fx:
        B, QD = fx["arms"]["B"], fx["arms"]["QD"]
        st, det = N.parity_compare(B, QD)
        check("T4 real NB4S B (7/1) vs QD (7/1) -> COMPARED", st == "COMPARED" and det["q"]["split"] == {"0": 7, "1": 1}, det)
        st, det = N.parity_compare(QD, fx["arms"]["QP"])
        check("T4 real NB4S QD (7/1) vs QP (3/5): both IMBALANCED -> COMPARED (same label)", st == "COMPARED" and det["rule"] == "same parity label", det)


# ------------------------------------------------------------------ T5
def t5():
    q = {"8k@8#1": win(305.0), "8k@8#2": win(312.0), "40k@8": win(297.0)}
    x = {"8k@12#1": win(380.0, 12), "8k@12#2": win(384.0, 12), "40k@12": win(360.0, 12)}
    s = N.speed_compare(q, x, ("8k@8#1", "8k@8#2"), ("8k@12#1", "8k@12#2"))
    check("T5 paired keys PASS", s["status"] == "PASS" and abs(s["gain"] - 0.238) < 0.01, s)
    check("T5 window parity recorded, not gating", s["window_parity"] == {"q": ["IMBALANCED"] * 2, "x": ["IMBALANCED"] * 2})
    s = N.speed_compare(q, {"8k@8#1": win(305.0), "8k@8#2": win(306.0)}, ("8k@8#1", "8k@8#2"), ("8k@8#1", "8k@8#2"))
    check("T5 within spread -> NO_GAIN", s["status"] == "NO_GAIN", s)
    s = N.speed_compare(q, {"8k@8#1": win(280.0), "8k@8#2": win(281.0)}, ("8k@8#1", "8k@8#2"), ("8k@8#1", "8k@8#2"))
    check("T5 slower -> REGRESSION", s["status"] == "REGRESSION", s)
    s = N.speed_compare(q, {"8k@8#1": win(350.0), "8k@8#2": win(0.0, status="ERROR")}, ("8k@8#1", "8k@8#2"), ("8k@8#1", "8k@8#2"))
    check("T5 one X window -> INCONCLUSIVE_X_WINDOWS", s["status"] == "INCONCLUSIVE_X_WINDOWS" and "gain" not in s, s)
    s = N.speed_compare({"8k@8#1": win(305.0)}, x, ("8k@8#1", "8k@8#2"), ("8k@12#1", "8k@12#2"))
    check("T5 Q one window -> INCONCLUSIVE_Q_SPREAD_UNKNOWN", s["status"] == "INCONCLUSIVE_Q_SPREAD_UNKNOWN")
    check("T5 IMBALANCED window is valid", N.window_ok(win(300.0, parity="IMBALANCED"))[0] and N.window_ok(win(300.0, parity="UNRESOLVED"))[0])
    check("T5 preemption invalid", not N.window_ok(win(300.0, pre=1))[0])
    check("T5 running != level invalid", not N.window_ok(win(300.0, run_eq=False))[0])
    check("T5 capacity-limited invalid", not N.window_ok(win(300.0, status="CAPACITY_LIMITED"))[0])
    sw = N.side_window(q, x, "40k@8", "40k@12")
    check("T5 side 40k paired", sw["gain"] == round(360.0 / 297.0 - 1, 4) and sw["q_ok"] and sw["x_ok"], sw)
    check("T5 side 40k invalid X", not N.side_window(q, {"40k@12": win(360.0, 12, pre=3)}, "40k@8", "40k@12")["x_ok"])
    a = plan()
    check("T5 fits", N.arm_fits(20 * 60 + 300, a["I8"]) and not N.arm_fits(20 * 60 + 299, a["I8"]))
    check("T5 body cap", N.body_cap_sec(15, 3600) == 900 and N.body_cap_sec(15, 900) == 600 and N.body_cap_sec(15, 100) == 60)
    # the priority walk at 120 min: setup 5, Q 21 (first boot, like NB4S B), then per-arm actual minutes
    for per, expect in ((16.0, ["I8", "P26", "P27", "C12"]), (19.0, ["I8", "P26", "P27", "C12"]), (23.0, ["I8", "P26", "P27"])):
        left, ran = (120 - 5 - 21) * 60.0, []
        for k in ("I8", "P26", "P27", "C12"):
            if N.arm_fits(left, a[k]):
                ran.append(k)
                left -= per * 60
        check(f"T5 priority walk at {per} min/arm", ran == expect, ran)
    check("T5 hard stop 120", N.HARD_STOP_MIN == 120)


# ------------------------------------------------------------------ T6
def t6(fx):
    a = plan()
    Q = arm_rec("Q")
    V = lambda X, k: N.arm_verdict(Q, X, k, arms=a)
    check("T6 base OK although IMBALANCED", N.base_check(Q) == (True, "OK", N.base_check(Q)[2]))
    check("T6 base OK when BALANCED too", N.base_check(arm_rec("Q", stall=stall_rec(parity="BALANCED", split={"0": 4, "1": 4})))[0])
    check("T6 base OK without traces (parity / stall never gate the base)", N.base_check(arm_rec("Q", stall={"error": "NO_TRACE_FILE"}))[0])
    for lab, kw in (("INCONCLUSIVE_BASE_GATE", {"quality_gate": "FAIL"}), ("INCONCLUSIVE_BASE_FAILED", {"status": "BODY_ERROR"}),
                    ("INCONCLUSIVE_BASE_ENV", {"env_readback": {"ok": False, "why": "x"}}),
                    ("INCONCLUSIVE_BASE_CONFIG", {"config_readback": {"ok": False, "why": "x"}}),
                    ("INCONCLUSIVE_BASE_NOT_ENGAGED", {"q_readback_final": {"label": "ENGAGED_FALLBACK"}}),
                    ("INCONCLUSIVE_BASE_NOT_INTEGRATED", {"status": "NOT_INTEGRATED", "not_integrated": True})):
        check(f"T6 base {lab}", N.base_check(arm_rec("Q", **kw))[1] == lab, N.base_check(arm_rec("Q", **kw)))
    check("T6 base not run", N.base_check({})[1] == "INCONCLUSIVE_NOT_RUN" and N.base_check({"status": "SKIPPED_BUDGET"})[1] == "INCONCLUSIVE_NOT_RUN")
    check("T6 WIN", V(arm_rec("I8", tok=(340.0, 345.0), tok40=325.0), "I8")["verdict"] == "WIN")
    check("T6 NO_GAIN", V(arm_rec("I8", tok=(310.0, 312.0)), "I8")["verdict"] == "NO_GAIN")
    check("T6 REGRESSION", V(arm_rec("P26", tok=(280.0, 282.0)), "P26")["verdict"] == "REGRESSION")
    check("T6 REGRESSION_40K", V(arm_rec("P26", tok=(340.0, 345.0), tok40=280.0), "P26")["verdict"] == "REGRESSION_40K")
    x = arm_rec("P26", tok=(340.0, 345.0))
    x["steady"]["40k@8"] = win(330.0, pre=2)
    check("T6 INCONCLUSIVE_40K", V(x, "P26")["verdict"] == "INCONCLUSIVE_40K")
    x = arm_rec("I8", tok=(340.0, 345.0))
    x["steady"]["8k@8#2"] = win(0.0, status="ERROR")
    check("T6 INCONCLUSIVE_X_WINDOWS", V(x, "I8")["verdict"] == "INCONCLUSIVE_X_WINDOWS")
    check("T6 WIN with a different parity (speed not parity-gated), stall PARITY_DIFFERS",
          (lambda v: v["verdict"] == "WIN" and v["stall"]["status"] == "PARITY_DIFFERS")(
              V(arm_rec("I8", tok=(340.0, 345.0), tok40=325.0, stall=stall_rec(parity="BALANCED", split={"0": 4, "1": 4})), "I8")))
    check("T6 CORRECTNESS_FAIL", V(arm_rec("I8", oracle_vs="FAIL_HIGH_MARGIN"), "I8")["verdict"] == "CORRECTNESS_FAIL")
    check("T6 NEAR_TIE ok", V(arm_rec("I8", tok=(340.0, 345.0), tok40=325.0, oracle_vs="NEAR_TIE"), "I8")["verdict"] == "WIN")
    check("T6 INCONCLUSIVE_ORACLE", V(arm_rec("I8", oracle_vs="INCONCLUSIVE_NO_ORACLE"), "I8")["verdict"] == "INCONCLUSIVE_ORACLE")
    check("T6 NOT_ENGAGED env", V(arm_rec("P26", env_readback={"ok": False, "why": "x"}), "P26")["verdict"] == "NOT_ENGAGED")
    check("T6 NOT_ENGAGED config (I8 head missing)", (lambda v: v["verdict"] == "NOT_ENGAGED" and "config" in v["why"])(
        V(arm_rec("I8", config_readback={"ok": False, "why": "missing: draft_head_built"}), "I8")))
    check("T6 Q-defer FALLBACK on an arm", V(arm_rec("C12", q_readback_final={"label": "ENGAGED_FALLBACK"}), "C12")["verdict"] == "ENGAGED_FALLBACK")
    check("T6 Q-defer missing", V(dict(arm_rec("I8"), q_readback_final=None), "I8")["verdict"] == "NOT_ENGAGED")
    check("T6 latest read-back wins (end ENGAGED, final FALLBACK)",
          V(arm_rec("I8", q_readback_end={"label": "ENGAGED"}, q_readback_final={"label": "ENGAGED_FALLBACK"}), "I8")["verdict"] == "ENGAGED_FALLBACK")
    oom = {"status": "LAUNCH_FAILED", "tries": [{"launch_name": "nb4t_p26", "oom": True, "device_fault": False, "kv_bytes": PIN},
                                                {"launch_name": "nb4t_p26_t1", "oom": True, "device_fault": False, "kv_bytes": PIN - 268435456}]}
    check("T6 LAUNCH_OOM", V(oom, "P26")["verdict"] == "LAUNCH_OOM")
    check("T6 LAUNCH_FAILED (no OOM)", V({"status": "LAUNCH_FAILED", "tries": [{"oom": False}]}, "P26")["verdict"] == "LAUNCH_FAILED")
    check("T6 BODY_OOM -> INCONCLUSIVE_ARM_FAILED", V(dict(arm_rec("P26"), status="BODY_OOM_warmup", died_in="warmup"), "P26")["verdict"] == "INCONCLUSIVE_ARM_FAILED")
    for bad in ("BODY_ERROR", "SERVER_DIED_IN_8k@8#1", "DEVICE_FAULT_oracle", "ERROR"):
        check(f"T6 arm status {bad} -> INCONCLUSIVE_ARM_FAILED", V(dict(arm_rec("I8", tok=(340.0, 345.0)), status=bad), "I8")["verdict"] == "INCONCLUSIVE_ARM_FAILED")
    check("T6 correctness wins over a failed status", V(dict(arm_rec("I8", oracle_vs="FAIL_HIGH_MARGIN"), status="BODY_ERROR"), "I8")["verdict"] == "CORRECTNESS_FAIL")
    check("T6 SKIPPED_BUDGET", V({"status": "SKIPPED_BUDGET", "why": "left"}, "C12")["verdict"] == "SKIPPED_BUDGET")
    check("T6 not run", V(None, "C12")["verdict"] == "SKIPPED_BUDGET")
    check("T6 SKIPPED_P26_OOM", V({"status": "SKIPPED_P26_OOM", "why": "P26 OOM"}, "P27")["verdict"] == "SKIPPED_P26_OOM")
    check("T6 SKIPPED_DEPENDENCY", V({"status": "SKIPPED_DEPENDENCY", "why": "Q"}, "I8")["verdict"] == "SKIPPED_DEPENDENCY")
    check("T6 NOT_INTEGRATED", V({"status": "NOT_INTEGRATED", "not_integrated": True, "why": "pre-sha"}, "I8")["verdict"] == "NOT_INTEGRATED")
    v = V(arm_rec("I8", tok=(340.0, 345.0), tok40=325.0, pin=PIN - 268435456), "I8")
    check("T6 pin differs: recorded, verdict kept", v["verdict"] == "WIN" and v["pin"]["differs"] is True, v.get("pin"))
    check("T6 base gate label on a clean arm", N.arm_verdict(arm_rec("Q", quality_gate="FAIL"), arm_rec("I8"), "I8", arms=a)["verdict"]
          == "INCONCLUSIVE_BASE_GATE")
    check("T6 base label on a clean arm", N.arm_verdict(arm_rec("Q", config_readback={"ok": False, "why": "x"}), arm_rec("I8"), "I8", arms=a)["verdict"]
          == "INCONCLUSIVE_BASE_CONFIG")
    # the whole verdict: winner among levers; C12 excluded; parity of every arm
    recs = {"Q": Q, "I8": arm_rec("I8", tok=(330.0, 333.0), tok40=320.0), "P26": arm_rec("P26", tok=(350.0, 352.0), tok40=338.0),
            "P27": {"status": "SKIPPED_P26_OOM", "why": "x"}, "C12": arm_rec("C12", tok=(400.0, 405.0), tok40=380.0)}
    v = N.verdict(recs, a)
    check("T6 winner P26 (highest lever gain), C12 excluded", v["best_arm"] == "P26" and v["verdict"] == "P26:WIN"
          and v["arms"]["C12"]["verdict"] == "WIN" and not v["throughput_eligible"]["C12"]["ok"], v.get("why"))
    check("T6 C12 block", v["c12"]["verdict"] == "WIN" and abs(v["c12"]["gain_8k"] - (402.5 / 308.5 - 1)) < 1e-3
          and set(v["c12"]["windows"]["x"]) == {"8k@12#1", "8k@12#2", "40k@12"}, v["c12"])
    check("T6 parity of every arm recorded", set(v["parity"]) == {"Q", "I8", "P26", "P27", "C12"}, sorted(v["parity"]))
    check("T6 verdict JSON", json.loads(json.dumps(v, default=str)) is not None)
    v = N.verdict({"Q": Q, "C12": arm_rec("C12", tok=(400.0, 405.0), tok40=380.0)}, a)
    check("T6 only C12 faster -> NO_THROUGHPUT_WINNER", v["best_arm"] is None and v["verdict"] == "NO_THROUGHPUT_WINNER", v.get("why"))
    v = N.verdict({"Q": arm_rec("Q", status="LAUNCH_FAILED", quality_gate=None)}, a)
    check("T6 base failed -> base label", v["verdict"] == "INCONCLUSIVE_BASE_GATE", v["verdict"])
    # P27 dependency
    check("T6 P27 runs after a clean P26", N.p27_dependency(arm_rec("P26", tries=[{"oom": False}]))[0])
    check("T6 P27 runs when P26 never ran", N.p27_dependency(None)[0])
    check("T6 P27 skipped after a P26 launch OOM (even when the retry booted)",
          not N.p27_dependency(dict(arm_rec("P26"), tries=[{"oom": True, "device_fault": False}, {"oom": False}]))[0])
    check("T6 P27 skipped after a P26 BODY_OOM", not N.p27_dependency({"status": "BODY_OOM_warmup", "tries": [{"oom": False}]})[0])
    check("T6 a device fault is not an OOM", N.p27_dependency({"status": "LAUNCH_FAILED", "tries": [{"oom": True, "device_fault": True}]})[0])
    if not fx:
        print("SKIP T6 real NB4S re-judge (no fixture)")
        return
    # the NB4S fix on the real GPU records: B as the base (7/1), QD as the arm (7/1)
    import nb4s_lib as S
    B, QD = copy.deepcopy(fx["arms"]["B"]), copy.deepcopy(fx["arms"]["QD"])
    check("T6 real: the NB4S rule rejected the base", S.base_check(B)[1] == "INCONCLUSIVE_BASE_IMBALANCED")
    for r in (B, QD):
        r["config_readback"] = {"ok": True, "why": "ok (NB4S had no config read-back)"}
    B["q_readback_final"] = {"label": "ENGAGED"}               # B ran unpatched: only for the base check of this re-judge
    ok, lab, why = N.base_check(B)
    check("T6 real: NB4T accepts the same base", ok and lab == "OK", (lab, why))
    v = N.arm_verdict(B, QD, "QD", arms={"Q": {"streams": 8}, "QD": {"streams": 8}})
    check("T6 real: QD re-judged -> WIN +21 %", v["verdict"] == "WIN" and abs(v["speed"]["gain"] - 0.2107) < 0.002
          and abs(v["side_40k"]["gain"] - 0.198) < 0.002, {k: v.get(k) for k in ("verdict", "why")})
    check("T6 real: stall comparison made (same 7/1 split), stage-1 compute read", v["stall"]["status"] == "COMPARED"
          and v["stall"]["x_stage0"]["share_below_1ms"] == 1.0, v["stall"].get("status"))
    s = copy.deepcopy(fx["arms"]["S"])
    s["config_readback"] = {"ok": True}
    s["q_readback_final"] = {"label": "ENGAGED"}
    v = N.arm_verdict(B, s, "S", arms={"Q": {"streams": 8}, "S": {"streams": 8}})
    check("T6 real: S (6/2 vs 7/1, both IMBALANCED) speed +11.6 % PASS", v["verdict"] == "WIN" and abs(v["speed"]["gain"] - 0.116) < 0.003, v.get("why"))


# ------------------------------------------------------------------ T7
def t7(tmp, merlin_src):
    a = plan()
    calls = []
    seq = N.PatchSeq(lambda: calls.append("apply") or {"ok": True}, lambda: calls.append("revert") or {"ok": True})
    res = [seq.before_arm(a[k])[0] for k in a]
    check("T7 applied once before Q, kept for every arm", res == [True] * 5 and calls == ["apply"] and seq.log == [("apply", "Q")], (res, calls))
    f = seq.finish()
    check("T7 reverted at finish", calls == ["apply", "revert"] and f["state"] == "reverted", (calls, f))
    check("T7 finish twice no-op", seq.finish()["state"] == "reverted" and calls == ["apply", "revert"])
    calls = []

    def bad_apply():
        calls.append("apply")
        raise RuntimeError("PATCH_PPACK_PRE_SHA_MISMATCH")
    seq = N.PatchSeq(bad_apply, lambda: calls.append("revert"))
    r = [seq.before_arm(a[k]) for k in a]
    check("T7 apply fails -> every arm NOT_INTEGRATED, tried once", all(not x[0] and "NOT_INTEGRATED" in x[1] for x in r) and calls == ["apply"])
    check("T7 failed apply never reverted", seq.finish()["state"] == "failed")
    seq = N.PatchSeq(lambda: None, lambda: (_ for _ in ()).throw(RuntimeError("REVERT_PPACK_SHA_MISMATCH")))
    seq.before_arm(a["Q"])
    fin = seq.finish()
    check("T7 revert failure recorded", fin["state"] == "revert_failed" and "REVERT" in (fin["error"] or ""), fin)
    if not merlin_src or not Path(merlin_src).is_dir():
        print("SKIP T7 real patch (no merlin-src)")
        return
    try:
        import nb4_ppack as P
        tree = tmp / "tree"
        for rel in P.PATCHES:
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(merlin_src) / rel, tree / rel)
        check("T7 real: Merlin files at the pre-shas", P.verify_ppack_patch(tree)["state"] == "not_applied")
        seq = N.PatchSeq(lambda: P.apply_ppack_patch(tree, sys.executable), lambda: P.revert_ppack_patch(tree, sys.executable))
        ok_q = seq.before_arm(a["Q"])
        st1 = P.verify_ppack_patch(tree)["state"]
        later = [seq.before_arm(a[k]) for k in ("I8", "P26", "P27", "C12")]
        st2 = P.verify_ppack_patch(tree)["state"]
        fin = seq.finish()
        v = P.verify_ppack_patch(tree)
        check("T7 real apply before Q", ok_q[0] and st1 == "applied", (ok_q, st1))
        check("T7 real kept for I8 / P26 / P27 / C12", all(x == (True, "already applied") for x in later) and st2 == "applied")
        check("T7 real revert -> pre-shas", fin["state"] == "reverted" and v["state"] == "not_applied"
              and all(v["files"][rel]["sha"] == p[0] for rel, p in P.PATCHES.items()), v)
        check("T7 real Q-defer env", P.server_env("Q-defer") == {"NB4_PPDEFER": "1"})
        (tree / P.PP_REL).write_text((tree / P.PP_REL).read_text(encoding="utf-8") + "\n# foreign\n", encoding="utf-8")
        seq = N.PatchSeq(lambda: P.apply_ppack_patch(tree, sys.executable), lambda: P.revert_ppack_patch(tree, sys.executable))
        r = seq.before_arm(a["Q"])
        check("T7 real foreign tree refused (NOT_INTEGRATED)", not r[0] and "PRE_SHA_MISMATCH" in r[1], r)
        check("T7 real foreign tree untouched", P.verify_ppack_patch(tree)["files"][P.MR_REL]["sha"] == P.MR_PRE)
    except Exception:
        check("T7 real patch", False, traceback.format_exc())


# ------------------------------------------------------------------ T8 CELL flow with a stub boot runner
def export_cells():
    t = (HERE / "nb4t_code_export.py").read_text(encoding="utf-8")
    cells = t.split("\n\nif not RERUN:\n")
    return [cells[0]] + [textwrap.dedent(c) for c in cells[1:]]


class StubL:
    U = types.SimpleNamespace(urlopen=lambda *a, **k: None)
    LIVE = []

    @staticmethod
    def prom(port):
        return {}

    @staticmethod
    def gpu_mem():
        return []


def flow_ns(tmp, left_sec=120 * 60, sha_ok=True, sha_raises=False):
    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    ns = {"__name__": "nb4t_flow", "os": os, "sys": sys, "json": json, "math": __import__("math"), "time": __import__("time"),
          "re": re, "shutil": shutil, "subprocess": subprocess, "hashlib": hashlib,
          "threading": __import__("threading"), "traceback": traceback, "Path": Path, "FAKE": None, "MODE": "FULL",
          "RERUN": False, "R": {"sections": {"A0": {"merge": {"ok": True}}}, "setup": {}, "errors": [], "order": []},
          "OUT": out, "TMP": tmp, "TRACES": tmp / "traces", "T_START": __import__("time").time(), "BUDGET": 120 * 60,
          "L": StubL, "KV_FP8": 2479838700, "EXT_FLAG": ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"],
          "PROFILER_CFG": "{}", "DIAG_ENV": {"VLLM_SERVER_DEV_MODE": "1"}, "NB1_EXTRA": {}, "NB1_ENV": {}, "NB1_KV": {},
          "ROWS": {"rc2_fp8": []}, "HELP": {}, "NVSMI_OK": False, "HOST_PHASE": {}, "PREFILL": {"tok_s": 2000.0, "measured": True},
          "ABORT": {"why": None}, "BOOT": {"deadline": None, "fired": None, "tag": "boot"}, "LEFT": [float(left_sec)],
          "BANDS": {"2k": 2048, "8k": 8192, "40k": 40000}, "PP": 2, "WIN": 90.0}
    ns["PROMPTS"] = {"2k": [[0, v] for v in range(8)], "8k": [[1, v] for v in range(8)], "40k": [[2, v] for v in range(8)]}
    ns["prompt"] = lambda band, v: [band, v]
    ns["left"] = lambda: ns["LEFT"][0]
    ns["log"] = lambda *a: None
    ns["save"] = lambda: (out / "nb4t_results.json").write_text(json.dumps(ns["R"], default=str))
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
    exec(compile(cells[3], "<cell4>", "exec"), ns)    # NB4S / NB1 CELL 4: step(), with_servers(), keep_after_error() ...
    ns["ROWS"]["rc2_fp8"] = []
    ns["BANDS"].update({"2k": 2048, "8k": 8192, "40k": 40000})
    exec(compile(cells[6], "<cell7>", "exec"), ns)    # NB4T CELL 7 (RC2_UNIT: definitions only, the main does not run)
    return ns


def runner_factory(ns, recs, minutes, order):
    def run(k):
        order.append(k)
        ns["LEFT"][0] -= minutes.get(k, 16) * 60
        r = recs.get(k)
        if isinstance(r, Exception):
            raise r
        return copy.deepcopy(r)
    return run


def good_recs():
    return {"Q": arm_rec("Q"), "I8": arm_rec("I8", tok=(330.0, 333.0), tok40=320.0), "P26": arm_rec("P26", tok=(350.0, 352.0), tok40=338.0),
            "P27": arm_rec("P27", tok=(309.0, 310.0)), "C12": arm_rec("C12", tok=(400.0, 405.0), tok40=380.0)}


def t8(tmp):
    good = good_recs()
    try:
        ns = flow_ns(tmp / "f1")
        check("T8 plan valid in the cell", tuple(ns["ARMS"]) == ("Q", "I8", "P26", "P27", "C12") and ns["B3_PIN"] == 3553580524)
        check("T8 L.prom replaced", ns["L"].prom is ns["_prom_nb4t"])
        check("T8 Q-defer from nb4_ppack.server_env", ns["ARMS"]["Q"]["env"] == {"VLLM_SERVER_DEV_MODE": "1", "NB4_PPDEFER": "1"})
        calls, order = [], []
        seq = ns["N"].PatchSeq(lambda: calls.append(("apply", list(order))) or {"ok": True}, lambda: calls.append(("revert", list(order))) or {"ok": True})
        ns["nb4t_main"](runner=runner_factory(ns, good, {"Q": 21}, order), pseq=seq)
        v = json.loads((tmp / "f1/out/nb4t_verdict.json").read_text())
        check("T8 order Q I8 P26 P27 C12", order == ["Q", "I8", "P26", "P27", "C12"], order)
        check("T8 patch applied before Q, reverted after C12", calls == [("apply", []), ("revert", ["Q", "I8", "P26", "P27", "C12"])], calls)
        check("T8 verdict winner P26, C12 measured", v["best_arm"] == "P26" and v["arms"]["C12"]["verdict"] == "WIN"
              and v["arms"]["P27"]["verdict"] == "NO_GAIN", v.get("why"))
        check("T8 patch state in verdict", (v.get("patch") or {}).get("state") == "reverted")
        check("T8 step records DONE", all(ns["_arm_rec"](k)["status"] == "DONE" for k in ns["ARMS"]))
        check("T8 Q's pin carried", ns["PIN"]["source"] == "Q's final pin")
    except Exception:
        check("T8 flow 1", False, traceback.format_exc())
    try:   # Q not clean -> only Q ran, the patch reverted, dependents skipped
        ns = flow_ns(tmp / "f2")
        calls, order = [], []
        seq = ns["N"].PatchSeq(lambda: calls.append("apply"), lambda: calls.append("revert"))
        ns["nb4t_main"](runner=runner_factory(ns, dict(good, Q=arm_rec("Q", status="NO_GO_QUALITY", quality_gate="FAIL")), {}, order), pseq=seq)
        v = json.loads((tmp / "f2/out/nb4t_verdict.json").read_text())
        check("T8 Q not clean -> only Q", order == ["Q"] and calls == ["apply", "revert"], (order, calls))
        check("T8 Q not clean -> dependents skipped", all(ns["_arm_rec"](k)["status"] == "SKIPPED_DEPENDENCY" for k in ("I8", "P26", "P27", "C12")))
        check("T8 Q not clean verdict", v["verdict"] == "INCONCLUSIVE_BASE_GATE", v.get("verdict"))
        ns = flow_ns(tmp / "f2b")
        order = []
        ns["nb4t_main"](runner=runner_factory(ns, dict(good, Q=arm_rec("Q", q_readback_final={"label": "ENGAGED_FALLBACK"})), {}, order),
                        pseq=ns["N"].PatchSeq(lambda: None, lambda: None))
        check("T8 Q-defer not ENGAGED on Q -> no later arm", order == ["Q"] and ns["_arm_rec"]("I8")["status"] == "SKIPPED_DEPENDENCY")
    except Exception:
        check("T8 flow 2", False, traceback.format_exc())
    try:   # P26 launch OOM -> P27 skipped, C12 still runs
        ns = flow_ns(tmp / "f3")
        order = []
        p26 = {"status": "LAUNCH_FAILED", "tries": [{"launch_name": "nb4t_p26", "oom": True, "device_fault": False, "kv_bytes": PIN},
                                                    {"launch_name": "nb4t_p26_t1", "oom": True, "device_fault": False, "kv_bytes": PIN - 268435456}]}
        ns["nb4t_main"](runner=runner_factory(ns, dict(good, P26=p26), {}, order), pseq=ns["N"].PatchSeq(lambda: None, lambda: None))
        v = json.loads((tmp / "f3/out/nb4t_verdict.json").read_text())
        check("T8 P26 OOM -> P27 skipped, C12 runs", order == ["Q", "I8", "P26", "C12"] and ns["_arm_rec"]("P27")["status"] == "SKIPPED_P26_OOM", order)
        check("T8 P26 OOM verdicts", v["arms"]["P26"]["verdict"] == "LAUNCH_OOM" and v["arms"]["P27"]["verdict"] == "SKIPPED_P26_OOM"
              and v["best_arm"] == "I8", v.get("why"))
    except Exception:
        check("T8 flow 3", False, traceback.format_exc())
    try:   # budget: 120 min, Q 21, I8 30, P26 30 -> P27 needs 25, has 34 -> runs 30 -> C12 has 4 -> skipped
        ns = flow_ns(tmp / "f4", left_sec=115 * 60)
        order = []
        ns["nb4t_main"](runner=runner_factory(ns, good, {"Q": 21, "I8": 30, "P26": 30, "P27": 30}, order), pseq=ns["N"].PatchSeq(lambda: None, lambda: None))
        check("T8 budget: C12 skipped", order == ["Q", "I8", "P26", "P27"] and ns["_arm_rec"]("C12")["status"] == "SKIPPED_BUDGET", order)
        v = json.loads((tmp / "f4/out/nb4t_verdict.json").read_text())
        check("T8 budget verdict label", v["arms"]["C12"]["verdict"] == "SKIPPED_BUDGET")
    except Exception:
        check("T8 flow 4", False, traceback.format_exc())
    try:   # the patch refused -> no boot at all
        ns = flow_ns(tmp / "f5")
        order = []

        def refuse():
            raise RuntimeError("PATCH_PPACK_PRE_SHA_MISMATCH: foreign")
        ns["nb4t_main"](runner=runner_factory(ns, good, {}, order), pseq=ns["N"].PatchSeq(refuse, lambda: None))
        v = json.loads((tmp / "f5/out/nb4t_verdict.json").read_text())
        check("T8 apply refused -> no boot", order == [], order)
        check("T8 apply refused -> Q NOT_INTEGRATED, base label", ns["_arm_rec"]("Q")["status"] == "NOT_INTEGRATED"
              and v["base"]["label"] == "INCONCLUSIVE_BASE_NOT_INTEGRATED", v.get("base"))
    except Exception:
        check("T8 flow 5", False, traceback.format_exc())
    try:   # an arm runner raises -> that arm ERROR, the plan continues, the verdict written
        ns = flow_ns(tmp / "f6")
        order = []
        ns["nb4t_main"](runner=runner_factory(ns, dict(good, I8=RuntimeError("boom")), {}, order), pseq=ns["N"].PatchSeq(lambda: None, lambda: None))
        v = json.loads((tmp / "f6/out/nb4t_verdict.json").read_text())
        check("T8 I8 error recorded, plan continues", order == ["Q", "I8", "P26", "P27", "C12"] and ns["_arm_rec"]("I8")["status"] == "ERROR", order)
        check("T8 I8 error verdict", v["arms"]["I8"]["verdict"] == "INCONCLUSIVE_ARM_FAILED", v["arms"]["I8"])
        ns = flow_ns(tmp / "f7", sha_raises=True)
        try:
            ns["nb4t_main"](runner=runner_factory(ns, good, {}, []), pseq=ns["N"].PatchSeq(lambda: None, lambda: None))
            raised = False
        except RuntimeError:
            raised = True
        check("T8 plan exception propagates, verdict written", raised and (tmp / "f7/out/nb4t_verdict.json").is_file())
        ns = flow_ns(tmp / "f8", sha_ok=False)
        ns["nb4t_main"](runner=runner_factory(ns, good, {}, []), pseq=ns["N"].PatchSeq(lambda: None, lambda: None))
        check("T8 sha gate fail -> all SKIPPED_SHA_GATE", all(ns["_arm_rec"](k)["status"] == "SKIPPED_SHA_GATE" for k in ns["ARMS"]))
    except Exception:
        check("T8 flow 6", False, traceback.format_exc())
    try:   # the launch wrapper, the build knob registry, prompts, win_need
        ns = flow_ns(tmp / "f9")
        A = ns["ARMS"]

        class Srv:
            env = full_env(A["I8"])
            argv = merlin_argv(A["I8"], SPEC_I8)
        ns["_merlin_fn_server_nb1"] = lambda *a: Srv()
        ns["ARM_OF"]["nb4t_i8"] = A["I8"]
        ns["NB4T_BUILD_ENV"]["nb4t_i8"] = dict(A["I8"]["build_env"])
        s = ns["merlin_fn_server"]("nb4t_i8", "fp8", 8021, PIN)
        check("T8 launch wrapper I8 ok + recorded", s is not None and ns["NB4T"]["launch"]["nb4t_i8"]["spec"].get("draft_lm_head_quant") == "int8"
              and ns["NB4T"]["launch"]["nb4t_i8"]["build_env"] == {"DRAFT_HEAD": "int8"})
        Srv.argv = merlin_argv(A["I8"], SPEC_Q)
        check("T8 launch wrapper refuses I8 without the int8 spec", raises(lambda: ns["merlin_fn_server"]("nb4t_i8", "fp8", 8021, PIN)))
        Srv.env, Srv.argv = dict(full_env(A["P26"]), CUDA_DEVICE_MAX_CONNECTIONS="32"), merlin_argv(A["P26"])
        ns["ARM_OF"]["nb4t_p26"] = A["P26"]
        s = ns["merlin_fn_server"]("nb4t_p26", "fp8", 8031, PIN)
        check("T8 launch wrapper: an image-inherited watched key is removed (recorded)", "CUDA_DEVICE_MAX_CONNECTIONS" not in s.env
              and ns["NB4T"]["launch"]["nb4t_p26"]["popped_unplanned_env"] == {"CUDA_DEVICE_MAX_CONNECTIONS": "32"})
        Srv.env = full_env(A["Q"])
        check("T8 launch wrapper refuses P26 on 25,23", raises(lambda: ns["merlin_fn_server"]("nb4t_p26", "fp8", 8031, PIN)))
        Srv.env, Srv.argv = full_env(A["C12"]), merlin_argv(A["C12"])
        ns["ARM_OF"]["nb4t_c12"] = A["C12"]
        check("T8 launch wrapper C12 ok", ns["merlin_fn_server"]("nb4t_c12", "fp8", 8051, PIN) is not None)
        check("T8 launch wrapper refuses a pin mismatch", raises(lambda: ns["merlin_fn_server"]("nb4t_c12", "fp8", 8051, PIN - 1)))
        r = ns["ensure_prompts"](12)
        check("T8 ensure_prompts: 12 variants, 0-7 unchanged", r == {"8k": 12, "40k": 12} and ns["PROMPTS"]["8k"][:8] == [[1, v] for v in range(8)]
              and ns["PROMPTS"]["8k"][8:] == [[8192, v] for v in range(8, 12)] and len(ns["PROMPTS"]["2k"]) == 8)
        check("T8 win_need", ns["win_need"]("8k@12#1", 12) == 90.0 + 120 + 12 * 8192 / 2000.0 and ns["win_need"]("profile_8k@12", 12) == 400)
        check("T8 RPC keys", ns["RPC_KEYS"].split(",")[:4] == list(ns["N"].WATCHED))
        log_ok = "\n".join(f'(Worker_PP{r // 2}_TP{r % 2} pid={10 + r}) NB4_PPQ ' + json.dumps(
            {"event": "init", "ppack": False, "ppdefer": r < 2, "is_last_rank": r >= 2, "pid": 10 + r}) for r in range(4))
        for r in (0, 1):
            st = {"defer_park": 1000, "flush_exec": 980, "flush_empty": 15, "flush_recv": 5, "flush_consume": 0}
            log_ok += "\n" + "\n".join("NB4_PPQ " + json.dumps({"event": "stats", "key": k, "pid": 10 + r, "stats": st}) for k in ("defer_park", "flush_exec"))
        check("T8 q_readback Q-defer ENGAGED on every arm", all(ns["q_readback"](types.SimpleNamespace(log=lambda: log_ok), A[k])["label"] == "ENGAGED" for k in A))
    except Exception:
        check("T8 flow 7", False, traceback.format_exc())
    finally:
        os.environ.pop("RC2_UNIT", None)


# ------------------------------------------------------------------ T9
def t9():
    nbp = HERE.parent / NB_NAME
    exp = (HERE / "nb4t_code_export.py").read_text(encoding="utf-8")
    nb = json.loads(nbp.read_text(encoding="utf-8"))
    facts = json.loads((HERE / "nb4t_build_facts.json").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check("T9 export == notebook cells", "\n\n".join(code) == exp)
    check("T9 notebook sha == facts", hashlib.sha256(nbp.read_bytes()).hexdigest() == facts["sha256"])
    check("T9 notebook < 400 kB", nbp.stat().st_size < 400_000, nbp.stat().st_size)
    check("T9 LF only", b"\r" not in nbp.read_bytes().replace(b"\\r", b""))
    sha = lambda b: hashlib.sha256(b).hexdigest()
    check("T9 CELL 0 = E18 R2 cell 1 (sha pinned)", sha(code[0].encode()) == facts["e18_r2_cell0_sha256"] and code[0].startswith("# ==================================================================== CELL 0"))
    e18 = HERE.parent.parent.parent / "Experiment 18 - S1-60K on vLLM 0.32 (planned)" / "Experiment Notebooks" / "Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb"
    if e18.is_file():
        c1 = "".join(json.loads(e18.read_text(encoding="utf-8"))["cells"][1]["source"]).replace("\r\n", "\n")
        check("T9 CELL 0 == the E18 R2 notebook's cell 1 (file)", c1 == code[0])
    else:
        print("SKIP T9 E18 R2 notebook file (sha pin checked)")
    check("T9 NB4S CELL 3-6 verbatim", [sha(c.encode()) for c in code[3:7]] == facts["nb4s_code_cell_sha256"][2:6])
    nb4s = HERE.parent / "Q27-XHIGH-E25_NB4S_STALL.ipynb"
    if nb4s.is_file():
        s_code = ["".join(c["source"]) for c in json.loads(nb4s.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        check("T9 NB4S notebook is the GPU-run fa9f35ca", sha(nb4s.read_bytes()) == facts["nb4s_sha256"])
        check("T9 NB4S cells 3-6 == NB4T cells 3-6 (file)", s_code[2:6] == code[3:7])
    check("T9 lib sha embedded (current file)", f'NB4T_LIB_SHA = "{sha((HERE / "nb4t_lib.py").read_bytes())}"' in exp)
    check("T9 timeline sha embedded (current file)", f'TL_SHA = "{sha((HERE / "nb4_timeline.py").read_bytes())}"' in exp)
    check("T9 ppack sha embedded (current file)", f'PPACK_SHA = "{sha((HERE / "nb4_ppack.py").read_bytes())}"' in exp)
    blobs = re.findall(r'zlib\.decompress\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)\)', exp)
    srcs = [zlib.decompress(base64.b64decode(b)).decode("utf-8") for b in blobs]
    check("T9 blobs == the files", len(srcs) == 3 and srcs[0].encode() == (HERE / "nb4t_lib.py").read_bytes()
          and srcs[1].encode() == (HERE / "nb4_timeline.py").read_bytes() and srcs[2].encode() == (HERE / "nb4_ppack.py").read_bytes())
    check("T9 budget 120 min", 'or 120 * 60)     # E25 NB4T' in code[1])
    check("T9 no [[0]] placeholder anywhere", '"attempt_1": [[0]]' not in exp)
    check("T9 identity fallback writer", "write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))" in code[1])
    i_knob = code[2].find('base.update((globals().get("NB4T_BUILD_ENV")')
    i_build = code[2].find('plan = eng["ML"].build("flash-next"')
    check("T9 build knob before build()", 0 < i_knob < i_build, (i_knob, i_build))
    check("T9 NB4S worker extension nb4s_env kept", "def nb4s_env" in exp)
    check("T9 OUT nb4t", '"/kaggle/working/nb4t"' in code[1] and "nb4t_results.json" in code[1])
    check("T9 cells: md + 9 code", [c["cell_type"] for c in nb["cells"]] == ["markdown"] + ["code"] * 9)
    check("T9 rerun guard on cells 2-8", all(c.startswith("if not RERUN:\n") for c in code[2:]))


# ------------------------------------------------------------------ T10 rerun / CELL 0
RERUN_DRIVER = r'''
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
upto = int(sys.argv[2])
g = {"__name__": "__main__"}
for i, c in enumerate(code[:upto]):
    exec(compile(c, f"<cell {i}>", "exec"), g)
print("@@CELLS_RAN", upto)
'''


def t10(tmp):
    arc = tmp / "arc"
    arc.mkdir(parents=True, exist_ok=True)
    tasks = {"aaaa0001": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1, 2], [3, 4]]}]},
             "bbbb0002": {"train": [], "test": [{"input": [[5]]}, {"input": [[6, 7]]}]}}
    (arc / "arc-agi_test_challenges.json").write_text(json.dumps(tasks), encoding="utf-8")
    want = {"aaaa0001": [{"attempt_1": [[1, 2], [3, 4]], "attempt_2": [[1, 2], [3, 4]]}],
            "bbbb0002": [{"attempt_1": [[5]], "attempt_2": [[5]]}, {"attempt_1": [[6, 7]], "attempt_2": [[6, 7]]}]}
    (tmp / "driver.py").write_text(RERUN_DRIVER, encoding="utf-8")
    nbp = HERE.parent / NB_NAME
    for mode, upto in (("rerun", 99), ("commit_cell0", 1)):
        work = tmp / f"work_{mode}"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("RC2_", "E23_", "NB4_", "KAGGLE_"))}
        env.update(Q27X_ARC_ROOT=str(arc), Q27X_WORKING=str(work), E23_OUT=str(tmp / f"out_{mode}"), E23_TMP=str(tmp / f"tmp_{mode}"),
                   E23_INPUT=str(tmp / "no_input"))
        if mode == "rerun":
            env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
        r = subprocess.run([sys.executable, str(tmp / "driver.py"), str(nbp), str(upto)], capture_output=True, text=True, env=env, timeout=300)
        sub = work / "submission.json"
        got = json.loads(sub.read_text(encoding="utf-8")) if sub.is_file() else None
        check(f"T10 {mode}: ran", r.returncode == 0 and "@@CELLS_RAN" in r.stdout, (r.returncode, r.stdout[-500:], r.stderr[-1500:]))
        check(f"T10 {mode}: identity submission.json (E18 R2 format)", got == want, got)
        if mode == "rerun":
            check("T10 rerun: the NB4T rerun line, nothing else ran", "E25 NB4T speed levers: competition rerun" in r.stdout
                  and not (tmp / "out_rerun" / "nb4t_results.json").exists() and "RC2 | budget" not in r.stdout, r.stdout[-800:])
            check("T10 rerun: MODE line from CELL 0", "MODE: COMPETITION RERUN (hidden test)" in r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merlin-src", default=r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
    ap.add_argument("--fixture", default=str(HERE / "nb4t_fixture_nb4s_gpu_2026-11-01.json"))
    ap.add_argument("--nb1-results", default=r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb1-capacity-r2/nb1/nb1_results.json")
    a = ap.parse_args()
    fx = load_fixture(a.fixture)
    if fx is None:
        print("SKIP fixture (not found):", a.fixture)
    tmp = Path(tempfile.mkdtemp(prefix="nb4t_units_"))
    try:
        for name, fn in (("T1", lambda: t1(fx)), ("T1r", lambda: t1_real_nb1(a.nb1_results)), ("T2", t2), ("T3", lambda: t3(fx)),
                         ("T4", lambda: t4(tmp, fx)), ("T5", t5), ("T6", lambda: t6(fx)), ("T7", lambda: t7(tmp, a.merlin_src)),
                         ("T8", lambda: t8(tmp)), ("T9", t9), ("T10", lambda: t10(tmp))):
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
