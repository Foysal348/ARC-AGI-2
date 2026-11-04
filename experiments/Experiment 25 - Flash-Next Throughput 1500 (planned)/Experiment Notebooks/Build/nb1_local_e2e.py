"""E25 NB1 local end-to-end harness (fake-server mode, as RC2's local harness): runs the exported notebook code
(nb1_code_export.py) as one process per flow against a fake /kaggle/input tree, and checks the outcome.

Flows:
  main      nb1_fake_server.py: patch applied to a copy of Merlin ad42e37's qsa.py; B1 -> B2 -> B3 all CLEAN; scorecard;
            pool 149 and F 6 recovered from the cold-cache snapshots; 90k x 5; commit-graph A/B; PP2 trace; submission.
  e23fake   the E23 fake server (no dev endpoints, no logprob / usage): the run must finish (rc 0) with every
            unsupported measurement recorded as INCONCLUSIVE / error, never a crash; B3 skipped by the pin rule
            (no free-VRAM reading: fail closed).
  patchfail a tree whose qsa.py is not Merlin's: the patch step fails, no boot starts, the scorecard + submission exist.
  b1fail    B1's gate reference is corrupted (NB1_FAKE_BAD_REF): B1 NO_GO, B2 / B3 never start.
  rerun     KAGGLE_IS_COMPETITION_RERUN=1: only the fallback submission (120 tasks).

Usage: python nb1_local_e2e.py <fake input dir> <work dir> [flow ...]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INP, WORK = Path(sys.argv[1]), Path(sys.argv[2])
FLOWS = sys.argv[3:] or ["main", "e23fake", "patchfail", "b1fail", "rerun"]
MERLIN_QSA = Path(os.environ.get("NB1_TEST_MERLIN_QSA") or r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/models/qwen4_exp/nvidia/qsa.py")
E23_FAKE = Path(os.environ.get("NB1_TEST_E23_FAKE") or (r"D:/ARC AGI 2/experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/src/"
                                                        r"harness/fake_vllm_server.py"))
REL = Path("vllm/models/qwen4_exp/nvidia/qsa.py")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def tree(d, good=True):
    shutil.rmtree(d, ignore_errors=True)
    (d / REL).parent.mkdir(parents=True)
    data = MERLIN_QSA.read_bytes()
    (d / REL).write_bytes(data if good else data + b"\n# not Merlin's file\n")
    return d


def flow(case, extra, server):
    w = WORK / case
    shutil.rmtree(w, ignore_errors=True)
    for x in ("out", "tmp", "shm"):
        (w / x).mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in ("RC2_UNIT", "KAGGLE_IS_COMPETITION_RERUN")}
    env.update(E23_FAKE_SERVER=str(server), FAKE_SPEED="300", E23_OUT=str(w / "out"), E23_TMP=str(w / "tmp"), E23_INPUT=str(INP),
               RC2_SUBMISSION=str(w / "out/submission.json"), RC2_SHM=str(w / "shm"), RC2_BUDGET_SEC="5400", E23_WIN_SEC="5",
               E23_EST_SCALE="0.01", PYTHONIOENCODING="utf-8")
    env["NB1_FAKE_TREE"] = extra.get("NB1_FAKE_TREE") or str(tree(w / "tree"))
    env.update({k: v for k, v in extra.items() if k != "NB1_FAKE_TREE"})
    t = time.time()
    r = subprocess.run([sys.executable, str(HERE / "nb1_code_export.py")], capture_output=True, text=True, env=env, cwd=str(w),
                       timeout=3600, encoding="utf-8", errors="replace")
    (w / "run.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    res = {}
    try:
        res = json.loads((w / "out/nb1_results.json").read_text())
    except Exception:
        pass
    sc = None
    try:
        sc = json.loads((w / "out/lever_scorecard.json").read_text())
    except Exception:
        pass
    sub = None
    try:
        sub = json.loads((w / "out/submission.json").read_text())
    except Exception:
        pass
    print(f"--- flow {case}: rc {r.returncode}, {time.time() - t:.0f} s", flush=True)
    return r, res, sc, sub, w


def boot(res, sec, name):
    return ((res.get("sections") or {}).get(sec) or {}).get(name) or {}


for case in FLOWS:
    if case == "main":
        r, res, sc, sub, w = flow("main", {"NB1_FAKE_FREE_MIB": "1500"}, HERE / "nb1_fake_server.py")
        check("main: rc 0", r.returncode == 0, r.stderr[-600:])
        pt = boot(res, "A0", "patch_60490")
        check("main: #60490 applied, post sha + marker + py_compile", pt.get("status") == "DONE" and pt.get("ok") and pt.get("applied") == "applied"
              and pt.get("marker") and pt.get("sha_after") == "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b",
              json.dumps({k: pt.get(k) for k in ("status", "applied", "error")})[:300])
        check("main: worker extension written into the tree", (w / "tree/nb1_worker_ext.py").is_file())
        b3tries = (res.get("sections") or {}).get("B3") or {}
        b3name = next((n for n, v in b3tries.items() if v.get("status") == "DONE"), "nb1_b3_fp8_pin")
        for sec, name in (("B1", "nb1_b1_fp8"), ("B2", "nb1_b2_fp8"), ("B3", b3name)):
            b = boot(res, sec, name)
            check(f"main: {sec} DONE + CLEAN + quality PASS", b.get("status") == "DONE" and b.get("correctness") == "CLEAN"
                  and b.get("quality_gate") == "PASS", f"{b.get('status')} {b.get('correctness')} {b.get('quality_gate')} {str(b.get('error') or b.get('body_error'))[:200]}")
        b1, b2, b3 = boot(res, "B1", "nb1_b1_fp8"), boot(res, "B2", "nb1_b2_fp8"), boot(res, "B3", b3name)
        nbp = (res.get("nb1") or {})
        check("main: B3 pin ladder starts at the largest predicted-safe step (free 1500 MiB -> +0.75 GiB) and the pool grows",
              b3name == "nb1_b3_fp8_pin75" and (nbp.get("pin_rule") or {}).get("steps_gib") == [0.75, 0.5]
              and ((b3.get("kv") or {}).get("pool_blocks") or 0) > 149 and (b3.get("pin_check") or {}).get("ok") is True,
              json.dumps({"name": b3name, "rule": nbp.get("pin_rule"), "P": (b3.get("kv") or {}).get("pool_blocks")})[:300])
        kv = b1.get("kv") or {}
        check("main: B1 pool 149 + F 6 recovered from cold snapshots", kv.get("pool_blocks") == 149 and abs(((kv.get("fit") or {}).get("F") or 0) - 6) < 0.6,
              json.dumps({"P": kv.get("pool_blocks"), "fit": kv.get("fit")})[:300])
        seq8 = next((s for s in kv.get("sequences") or [] if s.get("band") == "8k"), {})
        check("main: B1 8k cold sequence 1/2/4/8 all resident, 0 prefix hits", [x.get("requests") for x in seq8.get("snapshots") or []] == [1, 2, 4, 8]
              and all(x.get("all_resident") and x.get("prefix_hits_delta") == 0 for x in seq8.get("snapshots") or []),
              json.dumps([{k: x.get(k) for k in ("requests", "used_blocks", "F", "all_resident", "prefix_hits_delta")} for x in seq8.get("snapshots") or []])[:400])
        check("main: B1 retained blocks after release measured, reset drops them", seq8.get("retained_after_release_blocks", 0) > 0
              and seq8.get("after_reset_blocks") == 0, f"{seq8.get('retained_after_release_blocks')} {seq8.get('after_reset_blocks')}")
        adm = b1.get("admission_90k@5") or {}
        check("main: B1 90k x 5 fresh admission all resident + a decode window (tick / cycle)", adm.get("status") == "ALL_RESIDENT" and adm.get("all_resident") is True
              and (adm.get("window") or {}).get("cycle_ms") and len(adm.get("ttft_per_request") or []) == 5,
              json.dumps({k: adm.get(k) for k in ("status", "all_resident", "window", "first_token")})[:300])
        st1 = b1.get("steady") or {}
        check("main: B1 ladder 8k@3 / 8k@8 / 40k@5 + highest, tick_ms + cycle_ms + TTFT recorded",
              all(k in st1 for k in ("8k@3", "8k@8", "40k@5")) and any(k.startswith("40k@") and k != "40k@5" for k in st1)
              and all(v.get("cycle_ms") == round(2 * v["tick_ms"], 1) and v.get("ttft_per_request") for k, v in st1.items()
                      if not k.startswith("_") and v.get("status") == "STEADY"), json.dumps({k: (v.get("status"), v.get("tick_ms"), v.get("cycle_ms")) for k, v in st1.items() if not k.startswith("_")})[:400])
        check("main: every STEADY window real running == level", all(v.get("running_eq_level") for bb in (b1, b2, b3) for k, v in (bb.get("steady") or {}).items()
                                                                    if not k.startswith("_") and v.get("status") == "STEADY"))
        st2 = b2.get("steady") or {}
        check("main: B2 seqs 16 read from the log and a window above 8 streams", (b2.get("caps") or {}).get("max_num_seqs") == 16
              and any(k.startswith("8k@") and int(k.split("@")[1]) > 8 for k in st2), str(sorted(k for k in st2 if not k.startswith("_"))))
        check("main: B2 C3 dtype from the worker extension", "bfloat16" in ((b2.get("workers") or {}).get("recurrent_tensor_dtypes") or [])
              and "float32" in ((b1.get("workers") or {}).get("recurrent_tensor_dtypes") or []))
        ab = b2.get("commit_graph_ab") or {}
        check("main: B2 commit-graph runtime A/B (eager vs graph windows + counters + greedy identity)",
              all((ab.get("windows") or {}).get(m, {}).get("status") == "STEADY" for m in ("eager", "staged", "graph"))
              and "replay_share" in ((ab.get("counters") or {}).get("graph") or {})
              and (ab.get("greedy_graph_vs_eager") or {}).get("status") == "EXACT" and "results" in (ab.get("stats_after") or {}),
              json.dumps({k: ab.get(k) for k in ("delta_tok_s", "greedy_graph_vs_eager")})[:300])
        mb = ((((b1.get("profile") or {}).get("pp2") or {}).get("ranks") or {}).get("0") or {}).get("microbatch_costs") or {}
        agg = mb.get("decode_only_by_verify_tokens") or {}
        check("main: B1 PP2 trace: MoE / GEMM / all-reduce / QSA / top-k ms per micro-batch vs verify tokens",
              agg and all("moe_marlin" in v["kernel_ms_mean"] and "nccl" in v["kernel_ms_mean"] and "qsa" in v["kernel_ms_mean"]
                          for v in agg.values()), json.dumps(agg)[:300])
        check("main: B1 PP2 trace: 4 ranks, stage overlap verdict, verify rows", len(((b1.get("profile") or {}).get("pp2") or {}).get("ranks") or {}) == 4
              and (((b1.get("profile") or {}).get("pp2") or {}).get("stage_overlap") or {}).get("verdict") in ("OVERLAPPED", "PARTIAL", "SERIAL"),
              json.dumps(((b1.get("profile") or {}).get("pp2") or {}).get("stage_overlap"))[:300])
        msr = ((res.get("nb1") or {}).get("max_safe_running_by_context") or {}).get("B1") or {}
        check("main: B1 max_safe_running_by_context 8k / 40k / 60k / 90k / 116k (P 149, F 6, seqs 8; 60k x 7 skipped for headroom)",
              {b: v.get("achieved") for b, v in (msr.get("bands") or {}).items()} == {"8k": 8, "40k": 8, "60k": 6, "90k": 5, "116k": 4}
              and msr["bands"]["60k"]["bound_at_measured_pool_F"] == 7 and msr.get("status") == "MEASURED" and msr.get("pool_blocks") == 149
              and msr.get("F") == 6 and msr.get("pool_check") == "MATCH", json.dumps(msr)[:400])
        st0 = (((b1.get("profile") or {}).get("pp2") or {}).get("stage_imbalance") or {})
        rt = ((b1.get("profile") or {}).get("routing_ids") or {})
        check("main: B1 routing-id capture started and stopped around the profiling window only; nb1_routing_ids.npz in the NB3 schema",
              "results" in (rt.get("start") or {}) and (rt.get("merged") or {}).get("written") is True
              and (w / "out/nb1_routing_ids.npz").is_file() and (rt.get("merged") or {}).get("ids_in_range") is True, json.dumps(rt.get("merged"))[:200])
        check("main: B1 critical path per stage kept apart from summed kernel time", "tick_critical_path_ms" in st0
              and all("summed_kernel_ms" in st0.get(s_, {}) for s_ in ("stage0", "stage1")), json.dumps(st0)[:200])
        check("main: B1 stage imbalance + graph-replay shapes per micro-batch", "heavy_stage" in st0
              and all("resolved_capture_size" in v for v in agg.values()), json.dumps(st0)[:200])
        check("main: B1 NCCL transport parsed (debug on in B1 only)", (b1.get("nccl") or {}).get("channel_via", {}).get("SHM/direct/direct")
              and not (b2.get("nccl") or {}).get("debug_on"))
        check("main: B1 prefix continuation + abort measured", (b1.get("prefix") or {}).get("8k", {}).get("status") == "MEASURED"
              and (b1.get("prefix") or {}).get("abort_8k", {}).get("status") == "MEASURED")
        check("main: B1 FP8 self-check 2k / 8k raw rows on disk", all((w / f"out/tf_raw/b1_{b}.json").is_file() for b in ("2k", "8k")))
        check("main: config diff vs RC2 FP8 shows --language-model-only (B1) and the SSM dtype (B2)",
              "--language-model-only" in ((b1.get("config_diff") or {}).get("flags") or {}) and "--mamba-ssm-cache-dtype" in ((b2.get("config_diff") or {}).get("flags") or {}))
        rows = (sc or {}).get("rows") or []
        check("main: scorecard has every lever row with the required fields",
              len(rows) == 10 and all(all(k in x for k in ("lever", "measured", "projected_run_tok_s_gain", "correctness", "confidence",
                                                             "adoption", "cost_gpu_h", "notes")) for x in rows), str([x.get("lever")[:24] for x in rows]))
        check("main: submission with 120 tasks", isinstance(sub, dict) and len(sub) == 120)
    elif case == "e23fake":
        r, res, sc, sub, w = flow("e23fake", {}, E23_FAKE)
        check("e23fake: rc 0 (no crash on missing dev endpoints / usage / prompt logprobs)", r.returncode == 0, r.stderr[-600:])
        b1 = boot(res, "B1", "nb1_b1_fp8")
        check("e23fake: B1 ran to the end (DONE)", b1.get("status") == "DONE", f"{b1.get('status')} {str(b1.get('body_error'))[:200]}")
        check("e23fake: unsupported measurements recorded, not crashed", "error" in (b1.get("workers") or {})
              and (b1.get("kv") or {}).get("pool_blocks") is None)
        rp = b1.get("max_safe_running_by_context") or {}
        check("e23fake: pool not measured -> residency INCONCLUSIVE_CAPACITY_UNKNOWN, no probe ran (no RC2 149 / 6 fallback)",
              rp.get("status") == "INCONCLUSIVE_CAPACITY_UNKNOWN" and rp.get("bands") == {} and rp.get("pool_blocks") is None, json.dumps(rp)[:300])
        b3 = boot(res, "B3", "nb1_b3_fp8_pin")
        check("e23fake: B3 skipped by the pin rule (free VRAM not measured: fail closed)", b3.get("status") == "SKIPPED_DEPENDENCY"
              and "pin rule" in str(b3.get("why")), str(b3)[:200])
        check("e23fake: scorecard written", isinstance(sc, dict) and len(sc.get("rows") or []) == 10)
    elif case == "patchfail":
        w0 = WORK / "patchfail_tree"
        r, res, sc, sub, w = flow("patchfail", {"NB1_FAKE_TREE": str(tree(w0, good=False))}, HERE / "nb1_fake_server.py")
        pt = boot(res, "A0", "patch_60490")
        check("patchfail: rc 0", r.returncode == 0, r.stderr[-400:])
        check("patchfail: patch step ERROR with PRE_SHA_MISMATCH", pt.get("status") == "ERROR" and "PRE_SHA_MISMATCH" in str(pt.get("error")), str(pt.get("error"))[:200])
        check("patchfail: no boot started (all SKIPPED_DEPENDENCY)", all(boot(res, s, n).get("status") == "SKIPPED_DEPENDENCY" for s, n in
                                                                      (("B1", "nb1_b1_fp8"), ("B2", "nb1_b2_fp8"), ("B3", "nb1_b3_fp8_pin"))))
        check("patchfail: no server log written", not list((w / "out/logs").glob("server_nb1_*.log")))
        check("patchfail: scorecard + submission exist", isinstance(sc, dict) and isinstance(sub, dict) and len(sub) == 120)
    elif case == "b1fail":
        r, res, sc, sub, w = flow("b1fail", {"NB1_FAKE_FREE_MIB": "1500", "FAKE_GREEDY_SALT": "b1fail"}, HERE / "nb1_fake_server.py")
        b1 = boot(res, "B1", "nb1_b1_fp8")
        check("b1fail: rc 0", r.returncode == 0, r.stderr[-400:])
        check("b1fail: B1 quality gate not PASS -> load tests skipped", b1.get("quality_gate") != "PASS" and "load_tests_skipped" in b1
              and "kv" not in b1 and "steady" not in b1, f"{b1.get('status')} {b1.get('quality_gate')}")
        check("b1fail: B2 / B3 never started", all(boot(res, s, n).get("status") == "SKIPPED_DEPENDENCY" for s, n in
                                                   (("B2", "nb1_b2_fp8"), ("B3", "nb1_b3_fp8_pin"))))
    elif case == "rerun":
        r, res, sc, sub, w = flow("rerun", {"KAGGLE_IS_COMPETITION_RERUN": "1"}, HERE / "nb1_fake_server.py")
        check("rerun: rc 0 and a 120-task fallback submission, no boot", r.returncode == 0 and isinstance(sub, dict) and len(sub) == 120
              and not list((w / "out/logs").glob("server_*.log")), r.stderr[-300:])

n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
(WORK / "e2e_results.json").write_text(json.dumps([{"test": n, "ok": ok, "detail": d} for n, ok, d in RESULTS], indent=1))
sys.exit(1 if n_fail else 0)
