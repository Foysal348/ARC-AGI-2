"""E25 NB-S2 unit tests (stdlib only; local Windows Python and the Py3.12 Kaggle CPU units kit).

  python nbs2_unit_tests.py [--libs DIR [DIR ...]] [--merlin-src DIR] [--body FILE] [--notebook FILE] [--facts FILE] [--export FILE]

Base: NBDF-K100 closed e2198285 (B1 seam 341d859b, B2 nbdf_lib b0147751), gated c20eba7f (kits_wave3 real-kit report).
The facts / export / body / switch checks need no notebook (the units kit ships the export, not the .ipynb).

Prints `FAIL <name>: <why>` per failure, `SKIP <name>: <why>` when an input is absent, and `<passed>/<total> PASS` last.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")


def _arg(name, default=None, many=False):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if not many:
        return sys.argv[i + 1]
    out = []
    for x in sys.argv[i + 1:]:
        if x.startswith("--"):
            break
        out.append(x)
    return out


LIBS = _arg("--libs", [str(HERE), str(E27B)], many=True)
for d in reversed(LIBS):
    sys.path.insert(0, d)
MERLIN = Path(_arg("--merlin-src", r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src"))
BODY = Path(_arg("--body", str(HERE / "nbs2_cell7_body.py")))
NBF = Path(_arg("--notebook", str(HERE.parent / "Q27-XHIGH-E25_NBS2_CAPACITY.ipynb")))
FACTS = Path(_arg("--facts", str(HERE / "nbs2_build_facts.json")))
EXPORT = Path(_arg("--export", str(HERE / "nbs2_code_export.py")))
NBDF_K100_CLOSED = "e21982855d6430616f277784f162cad27cac94f7bcd8488fed97f123645ba730"
NBDF_K100_GATED = "c20eba7f00b90b8d8e08308c48d57f09b404214039d6c4fa46c059fea92e2004"
SEAM_B1 = "341d859be5b2cc619247ff7b645304b9473d09f07995358ead2ee02be116c60b"
NBDF_LIB_B2 = "b014775162fb55401826fa8c498f7009b5d47ad15cc5cb365ab23e5ed37b1929"
SWITCH_DEFAULT = {"B": True, "IDX": True, "S28": True, "LP": True, "S32": True, "SPC": True}

import nb4t_lib as T  # noqa: E402
import nbdf_lib as N  # noqa: E402
import nbs2_lib as S2  # noqa: E402
try:
    import nb4_ppack as P  # noqa: E402
except Exception:
    P = None

RES = {"pass": 0, "total": 0}


def check(name, cond, why=""):
    RES["total"] += 1
    if cond:
        RES["pass"] += 1
    else:
        print(f"FAIL {name}: {why}"[:600], flush=True)


def skip(name, why):
    print(f"SKIP {name}: {why}", flush=True)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------ the NBDF-K100 D3s (nbdf_lib, configured as the K100 body)
K100_Q_CAPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96]
N.configure(SEQS=24, CONC_LADDER=(2, 8, 23), Q_CAPTURES=K100_Q_CAPS, Q_TAIL=["--compilation-config", N.compilation_json(K100_Q_CAPS)],
            WKEYS=("8k@16#1", "8k@16#2", "profile_8k@16", "40k@16"), PREFIX_RECORD_ONLY=True, SHORT_EXIT_ARMS=(),
            LONG_THR_FLOOR={"90k": 0.037, "116k": 0.054}, TF_MEAN_MARGIN=0.05, TF_RANK1_MARGIN=0.03, TF_REQUIRED=True)
Q_EXTRA = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}", "--max-num-seqs", "24"]
SEAM = {"DFLASH_SEAM": "1", "DFLASH_MASK_ROW_SHA": N.DRAFTER_PINS["mask_row"]["sha256"], "DFLASH_DRAFT_SW_BLOCK": "1024"}
QENV = P.server_env("Q-defer") if P else {T.ENV_DEFER: "1"}
ARMS0 = N.arm_plan(Q_EXTRA, {"VLLM_SERVER_DEV_MODE": "1", "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, 7933526016,
                   "/kaggle/input/models/foysalemonshanto/" + N.DRAFTER_REL, q_env=QENV, seam_envs={"D3s": dict(SEAM), "D4s": dict(SEAM)})
N.restrict_arms(("Q", "D3s", "D4s"))
N.check_arm_plan({k: ARMS0[k] for k in N.ORDER})
N.restrict_arms(("Q", "D3s"))
D3S = ARMS0["D3s"]
ARMS = S2.arm_plan(D3S, seam_env_k4=dict(SEAM), pin=7933526016, seqs=24)


def t_plan():
    check("plan_order", tuple(ARMS) == ("B",) + S2.LEVERS, list(ARMS))
    check("plan_valid", S2.check_arm_plan(ARMS, D3S) is True)
    b = ARMS["B"]
    check("B_is_D3s", all(b[f] == D3S[f] for f in ("env", "extra", "spec", "captures", "k", "method", "kv_bytes", "q_label", "expect")))
    check("B_no_draft_head", not b.get("draft_head") and "DRAFT_HEAD" not in b["env"] and b["spec"].get("draft_lm_head_quant") is None)
    check("B_captures_4_96", b["captures"] == [4 * i for i in range(1, 25)], b["captures"])
    check("B_seqs_24", T.last_value(b["extra"], "--max-num-seqs") == "24" and b["streams"] == 24)
    check("B_windows_std", b["windows_fixed"] == S2.STD_WINDOWS)
    i = ARMS["IDX"]
    check("IDX_argv", i["extra"][-2:] == ["--attention-config", '{"indexer_kv_dtype":"fp8"}'] and i["extra"][:-2] == b["extra"])
    check("IDX_no_dotted_flag", not any(str(x).startswith("--attention-config.") for x in i["extra"]))
    for k, s in (("S28", 28), ("S32", 32)):
        a = ARMS[k]
        check(f"{k}_seqs", T.last_value(a["extra"], "--max-num-seqs") == str(s) and a["streams"] == s and a["expect_cfg"]["max_num_seqs"] == s)
        check(f"{k}_captures", a["captures"] == [4 * x for x in range(1, s + 1)] and N.captures_from_argv(a["extra"]) == a["captures"])
        check(f"{k}_spec_same", T.spec_from_argv(a["extra"]) == b["spec"])
        check(f"{k}_windows", a["windows_fixed"][:4] == [["8k", s, "#1"], ["8k", s, "#2"], ["20k", s, ""], ["40k", s, ""]]
              and a["windows_fixed"][4:] == [["90k", 8, ""], ["116k", 6, ""]])
    check("DYN_not_in_nbs2", "DYN" not in ARMS and S2.DROPPED_LABEL["DYN"] == "MOVED_TO_NB_S1")
    lp = ARMS["LP"]
    check("LP_argv", lp["extra"][len(b["extra"]):] == ["--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "1024",
                                                       "--long-prefill-token-threshold-adaptive"])
    check("SPC_argv", ARMS["SPC"]["extra"][len(b["extra"]):] == ["--enable-mamba-shared-prefix-checkpoint"])
    check("ports_distinct", len({a["port"] for a in ARMS.values()}) == len(ARMS))
    check("pins_equal", {a["kv_bytes"] for a in ARMS.values()} == {7933526016})
    check("prefix_sibling_stall_sets", [k for k, a in ARMS.items() if a["s2"]["prefix_test"]] == ["B", "SPC"]
          and [k for k, a in ARMS.items() if a["s2"]["stall"]] == ["B", "LP"] and not any(a["s2"]["profile"] for a in ARMS.values()))
    # mutations: each must be refused
    for nm, mut in (("extra_flag", lambda a: a["IDX"]["extra"].append("--foo")),
                    ("env_change", lambda a: a["LP"]["env"].update(NCCL_ALGO="Ring")),
                    ("seqs_wrong", lambda a: a["S28"]["extra"].__setitem__(-1, "30")),
                    ("never_do", lambda a: a["SPC"]["extra"].append("--no-enable-prefix-caching")),
                    ("pin", lambda a: a["IDX"].update(kv_bytes=7665090560)),
                    ("order", lambda a: a.update({"X": a.pop("B")}))):
        a2 = copy.deepcopy(ARMS)
        mut(a2)
        check(f"plan_refuses_{nm}", raises(lambda: S2.check_arm_plan(a2, D3S), ValueError))
    check("base_must_be_d3s", raises(lambda: S2.arm_plan(ARMS0["Q"]), ValueError))


def t_launch_check():
    for k, a in ARMS.items():
        env = {w: v for w, v in a["expect"].items() if v is not None}
        argv = ["vllm", "serve", "/ckpt"] + list(a["extra"]) + ["--kv-cache-memory-bytes", str(a["kv_bytes"])]
        bad = N.launch_check(a, env, argv, a["kv_bytes"])
        check(f"launch_check_{k}", bad == [], bad)
        bad2 = N.launch_check(a, dict(env, DRAFT_HEAD="int8"), argv, a["kv_bytes"])
        check(f"launch_check_S4_{k}", any("S4" in x for x in bad2), bad2)


def t_dense_schedule():
    f = MERLIN / "vllm/v1/spec_decode/dynamic/utils.py"
    cases = [([[1, 8, 4], [9, 24, 3]], 24, 4), ([[1, 8, 4], [9, 32, 3]], 32, 4), ([[1, 16, 3], [32, 128, 2]], 64, 3), ([[1, 4, 7]], 10, 4)]
    for sch, m, k in cases:
        d = S2.dense_schedule(sch, m, k)
        check(f"dense_len_{m}_{k}", len(d) == m + 1 and d[0] == 0)
    check("dense_dyn", S2.dense_schedule(S2.DYN_SCHEDULE, 24, 4)[1:] == [4] * 8 + [3] * 16)
    for bad in ([], [[2, 8, 4]], [[1, 8, 4], [8, 24, 3]], [[1, 8]], [[3, 1, 2]]):
        check(f"dense_refuses_{bad}", raises(lambda b=bad: S2.dense_schedule(b, 24, 4), ValueError))
    if not f.is_file():
        skip("dense_vs_merlin", f"no {f}")
        return
    spec = importlib.util.spec_from_file_location("merlin_dyn_utils", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for sch, m, k in cases:
        check(f"dense_equals_merlin_{m}_{k}", S2.dense_schedule(sch, m, k) == mod.build_dynamic_sd_schedule_lookup(sch, m, k))


def t_captures():
    c = S2.dyn_captures(S2.DYN_SCHEDULE, 24, 4)
    check("dyn_caps_values", c == sorted(set([5 * b for b in range(1, 9)] + [4 * b for b in range(9, 25)])), c)
    check("dyn_cov_ok", S2.dyn_coverage(c, S2.DYN_SCHEDULE, 24, 4)["covers_all"])
    cov_b = S2.dyn_coverage([4 * b for b in range(1, 25)], S2.DYN_SCHEDULE, 24, 4)
    check("dyn_cov_round_up", cov_b["covers_all"], cov_b["uncovered_or_padded_b"])   # the V2 runner rounds 4k up to 5k (b <= 19)
    cov_s = S2.dyn_coverage([5, 10, 40], S2.DYN_SCHEDULE, 24, 4)
    check("dyn_cov_insufficient", not cov_s["covers_all"] and 3 in cov_s["uncovered_or_padded_b"], cov_s["uncovered_or_padded_b"])
    check("dflash_caps", S2.dflash_captures(32, 3)[-1] == 128 and len(S2.dflash_captures(32, 3)) == 32)


def t_readbacks():
    w = {"ranks": [{"rank": r, "kv_cache_specs": [{"type": "MLAAttentionSpec", "example": f"language_model.model.layers.3.self_attn.indexer.compressed_key_cache",
                                                   "dtype": "torch.float8_e4m3fn", "page_size_bytes": 10}]} for r in range(4)]}
    check("idx_engaged", S2.idx_readback(w)["engaged"] is True)
    w2 = copy.deepcopy(w)
    w2["ranks"][2]["kv_cache_specs"][0]["dtype"] = "torch.bfloat16"
    check("idx_not_engaged", S2.idx_readback(w2)["engaged"] is False)
    check("idx_none", S2.idx_readback({"ranks": [{"rank": 0, "kv_cache_specs": []}]})["engaged"] is None)
    full = "(Worker_PP0_TP0 pid=1) INFO Capturing CUDA graphs (FULL): 100%|####| 24/24 [00:03<00:00]\nnon-default args: {'compilation_config': {'cudagraph_mode': 'FULL_AND_PIECEWISE'}}"
    check("dyn_graph_pass", S2.dyn_graph_readback(full, [])["status"] == "PASS", S2.dyn_graph_readback(full, []))
    check("dyn_graph_override", S2.dyn_graph_readback(full + "\nWARNING " + S2._DYN_OVERRIDE + " from FULL_AND_PIECEWISE to PIECEWISE", [])["status"] == "FAIL")
    pw = "Capturing CUDA graphs (PIECEWISE): 100%|####| 24/24 [00:03<00:00]"
    check("dyn_graph_piecewise_only", S2.dyn_graph_readback(pw, [])["status"] == "FAIL")
    check("dyn_graph_none", S2.dyn_graph_readback("", [])["status"] == "INCONCLUSIVE")
    eng_pw = full + "\nINFO Initializing a V1 LLM engine (v0.32) with config: compilation_config={'level': 3, 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>}"
    check("dyn_graph_engine_piecewise", S2.dyn_graph_readback(eng_pw, [])["status"] == "FAIL", S2.dyn_graph_readback(eng_pw, []).get("logged_mode"))
    eng_ok = full + "\nINFO Initializing a V1 LLM engine (v0.32) with config: compilation_config={'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}"
    check("dyn_graph_engine_full", S2.dyn_graph_readback(eng_ok, [])["status"] == "PASS", S2.dyn_graph_readback(eng_ok, []).get("logged_mode"))
    check("k_eff", S2.k_eff({"accept_per_draft": 2.0, "accept_rate": 0.5}) == 4.0 and S2.k_eff({}) is None)
    rec = {"steady": {"8k@16#1": {"level": 16, "accept_per_draft": 1.8, "accept_rate": 0.6}, "90k@8": {"level": 8, "accept_per_draft": 2.4, "accept_rate": 0.6}}}
    t = S2.dyn_tiers(rec)
    check("dyn_tiers", t["8k@16#1"]["k_schedule"] == 3 and t["8k@16#1"]["engaged"] is True and t["90k@8"]["k_schedule"] == 4
          and t["90k@8"]["engaged"] is True, t)


METRICS = """# HELP x
vllm:kv_cache_usage_perc{engine="0",model_name="q27"} 0.5
vllm:num_requests_running{engine="0",model_name="q27"} 24
vllm:num_requests_waiting{engine="0",model_name="q27"} 2
vllm:num_preemptions_total{engine="0",model_name="q27"} 1
vllm:prompt_tokens_by_source_total{engine="0",model_name="q27",source="computed"} 1000
vllm:prompt_tokens_by_source_total{engine="0",model_name="q27",source="local_cache_hit"} 400
vllm:num_requests_waiting_by_reason{engine="0",model_name="q27",reason="capacity"} 2
"""


def t_ledger():
    m0 = S2.parse_metrics(METRICS)
    check("parse_metrics", m0["vllm:kv_cache_usage_perc"] == 0.5 and m0["by_source"]["computed"] == 1000 and m0["waiting_by_reason"]["capacity"] == 2, m0)
    m1 = copy.deepcopy(m0)
    m1["vllm:kv_cache_usage_perc"], m1["vllm:num_preemptions_total"] = 0.9, 3
    m1["by_source"]["computed"] = 5096
    s = S2.ledger_summary([(1.0, m0), (3.0, m1)], 0.5, 3.0, pool=540, level=24, band_tokens=20000, F=12, block=4096, gen_tokens=24000)
    check("ledger_numbers", s["kv_usage_max"] == 0.9 and s["preemptions"] == 2 and s["recompute_tokens"] == 4096
          and s["blocks_in_use_max"] == 486 and s["free_blocks_min"] == 54 and s["running_required_blocks_est"] == 24 * (6 + 12), s)
    check("ledger_reclaimable_not_readable", str(s["reclaimable_blocks"]).startswith("NOT_READABLE"))
    check("ledger_empty", S2.ledger_summary([], 0, 1) == {"samples": 0})


def t_probes():
    ser = {0: [(0.0, 0), (0.1, 4), (0.2, 8), (1.0, 12), (1.1, 16)], 1: [(0.0, 0), (0.15, 4), (1.05, 8)]}
    st = S2.stall_stats(ser, 0.5, 1.2)
    check("stall_baseline", st["baseline"]["gaps"] == 3 and st["baseline"]["max_ms"] == 150.0, st)
    check("stall_injection", st["injection"]["gaps"] == 3 and st["injection"]["max_ms"] == 900.0 and st["p95_ratio"] and st["p95_ratio"] > 1, st)
    check("sibling_hit", S2.sibling_label({"prefix_tokens": 6900, "hits": {"C": 6848}}, 1024)["label"] == "SPC_JUNCTION_HIT")
    check("sibling_block", S2.sibling_label({"prefix_tokens": 6900, "hits": {"C": 6144}}, 1024)["label"] == "BLOCK_BOUNDARY_ONLY")
    check("sibling_missing", S2.sibling_label({}, 1024)["label"] == "INCONCLUSIVE_SIBLING")


def _w(tok, level, al=3.0):
    return {"status": "STEADY", "level": level, "tok_s": tok, "running_min": level, "running_max": level, "running_eq_level": True,
            "prefill_done_before_window": True, "preemptions": 0, "mean_acceptance_length": al, "accept_per_draft": al - 1, "accept_rate": (al - 1) / 3}


def _rec(windows, **kw):
    r = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS", "long_gate": "PASS", "oracle_vs": "EXACT", "smoke_gate": "PASS",
         "q_readback_final": {"label": "ENGAGED", "detail": {"inits": 4, "expect": 4, "defer_per_rank": {}}},
         "env_readback": {"ok": True}, "config_readback": {"ok": True}, "dflash_readback": {"ok": True, "checks": {}, "contradictions": []},
         "steady": windows, "tf_gate": {"status": "PASS"}, "memory_gate_ready": {"ok": True}, "memory_gate_first_decode": {"ok": True}}
    r.update(kw)
    return r


B_WIN = {"8k@16#1": _w(500, 16), "8k@16#2": _w(505, 16), "20k@24": _w(560, 24), "40k@16": _w(440, 16), "90k@8": _w(300, 8), "116k@6": _w(250, 6)}


def t_verdict():
    B = _rec(dict(B_WIN))
    S = _rec({"8k@28#1": _w(560, 28), "8k@28#2": _w(565, 28), "20k@28": _w(620, 28), "40k@28": _w(470, 28), "90k@8": _w(301, 8), "116k@6": _w(251, 6)})
    v = S2.lever_verdict(B, S, "S28")
    check("capacity_win", v["verdict"] == "WIN" and v["primary"]["kind"] == "capacity", (v.get("verdict"), v.get("why")))
    v = S2.lever_verdict(B, dict(S, tf_gate={"status": "INCONCLUSIVE_TF"}), "S28")
    check("capacity_tf_pending", v["verdict"] == "INCONCLUSIVE_TF" and v.get("speed_verdict") == "WIN", (v.get("verdict"), v.get("why")))
    v = S2.lever_verdict(B, dict(S, tf_gate={"status": "FAIL_TF_OUTSIDE_Q_AA", "why": "x"}), "S28")
    check("tf_fail_real", v["verdict"] == "CORRECTNESS_FAIL_TF", v.get("verdict"))
    v = S2.lever_verdict(B, dict(S, status="NOT_INTEGRATED", not_integrated=True, early_stop="DYN_GRAPH_FALLBACK: x"), "DYN")
    check("real_failure_kept", v["verdict"] == "NOT_INTEGRATED", v.get("verdict"))
    S_reg = copy.deepcopy(S)
    S_reg["steady"]["116k@6"] = _w(200, 6)
    check("capacity_long_regression", S2.lever_verdict(B, S_reg, "S28")["verdict"] == "REGRESSION")
    D = _rec(dict(B_WIN, **{"90k@8": _w(340, 8, 3.3), "116k@6": _w(285, 6, 3.3)}))
    check("dyn_low_conc_win", S2.lever_verdict(B, D, "DYN")["verdict"] == "WIN")
    D2 = _rec(dict(B_WIN, **{"20k@24": _w(500, 24)}))
    check("dyn_k3_tier_regression", S2.lever_verdict(B, D2, "DYN")["verdict"] == "REGRESSION")
    Bs = dict(B, stall_probe={"stats": {"injection": {"p95_ms": 400.0}}})
    L_ = _rec(dict(B_WIN), stall_probe={"stats": {"injection": {"p95_ms": 250.0}}})
    check("lp_stall_win", S2.lever_verdict(Bs, L_, "LP")["verdict"] == "WIN")
    L2 = _rec(dict(B_WIN), stall_probe={"stats": {}})
    check("lp_stall_inconclusive", S2.lever_verdict(Bs, L2, "LP")["verdict"] == "INCONCLUSIVE_STALL")
    I_ = _rec(dict(B_WIN, **{"40k@16": _w(470, 16)}))
    check("idx_any_band_win", S2.lever_verdict(B, I_, "IDX")["verdict"] == "WIN")
    check("idx_no_gain", S2.lever_verdict(B, _rec(dict(B_WIN)), "IDX")["verdict"] == "NO_GAIN")
    allv = S2.verdict({"B": B, "S28": S, "DYN": D, "IDX": I_}, ARMS)
    check("verdict_dropped", allv["arms"]["K4"]["verdict"] == "N/A_DFLASH" and allv["arms"]["PP4"]["verdict"] == "N/A_DFLASH"
          and allv["arms"]["DYN"]["verdict"] == "MOVED_TO_NB_S1")
    check("verdict_skipped_not_run", allv["arms"]["SPC"]["verdict"].startswith("SKIPPED"), allv["arms"]["SPC"])
    check("verdict_winners", set(allv["winners"]) == {"S28", "IDX"} and allv["best_arm"] in allv["winners"], allv["winners"])
    check("verdict_base", allv["base"]["ok"] is True, allv["base"])


def _body_src():
    if not BODY.is_file():
        return None
    return BODY.read_text(encoding="utf-8")


def t_body_text():
    s = _body_src()
    if s is None:
        skip("body_text", f"no {BODY}")
        return
    for frag in ('is_q, is_d = arm["key"] == S2.BASE_KEY', "_s2f = s2_readback_failure(o, arm, t)", 'o["s2_records"] = s2_records(o, arm, t)',
                 "with S2Ledger(port) as _s2l:", 'r["kv_ledger"] = s2_window_ledger(o, r, band, n)', "    nbs2_main()",
                 '(arm.get("windows_fixed") or K100_CFG["windows"])', "if band in S2.CAPPED_BANDS:", 'seqs=int(o.get("streams") or N.SEQS))',
                 "if is_q and not is_d:", "BASE_LADDER_NONDET_RECORDED", 'S2.check_arm_plan(ARMS, NBDF_ARMS["D3s"])', 'NBS2_LIB_SRC = _nbs2_ds("nbs2_lib.py"',
                 '_nbs2_on = globals().get("NBS2_ON")', "NBS2_SWITCH = S2.switch_map(_nbs2_on, tuple(ARMS))",
                 'skip_arm(k, "switched off in CELL 1 (NBS2_ON)", "SKIPPED_SWITCH_OFF")', "if k not in RUN_KEYS:",
                 'NBS2["switch"] = S2.switch_record(_nbs2_on, tuple(ARMS))', 'o["s2_readback_error"]',
                 '"INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION"', "# records only: never voids the measured window",
                 "# records only: never fails the arm", "L.degenerate = N.hard_only(L.degenerate)"):
        check(f"body_has_{frag[:40]}", s.count(frag) == 1, f"count {s.count(frag)}")
    check("body_no_nbdf_main_call", "\n    nbdf_main()" not in s)
    check("body_no_switch_literal", "NBS2_ON =" not in s and "NBS2_ON = {" not in s, "the switch line belongs to CELL 1, not the payload body")
    for frag, lib_fn in (("short_gate_failed", "N.short_gate_hard("), ("quality_gate", "N.quality_gate_hard("),
                         ("oracle_gate", "N.oracle_gate_hard("), ("evidence_gate", "N.sampled_hard_fail(")):   # B2 hard-only rebinds (nbdf_lib b0147751)
        check(f"body_b2_hard_only_{frag}", f"\ndef {frag}(" in s and lib_fn in s, frag)
    check("body_compiles", compile(s, "body", "exec") is not None)


class _Seq:
    def __init__(self, fail_check=(), apply_ok=True):
        self.fail_check, self.apply_ok, self.state, self.ring, self.calls = set(fail_check), apply_ok, "pristine", True, []

    def apply(self):
        self.calls.append("apply")
        self.state = "applied" if self.apply_ok else "failed"
        return (True, "applied") if self.apply_ok else (False, "NOT_INTEGRATED: apply failed")

    def check(self, k):
        self.calls.append(("check", k))
        return (False, "SEAM_TREE_INVALID_MIXED") if k in self.fail_check else (True, "FULLY_PATCHED")

    def before_arm(self, a):
        self.calls.append(("before_arm", a["key"]))
        return (True, "ok")

    def finish(self):
        self.calls.append("finish")


def _main_ns(fail_arms=(), left_min=10_000, seam_ok=True, fail_check=(), drafter_ok=True, switch=None, merge_ok=True):
    s = _body_src()
    tree = ast.parse(s)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "nbs2_main")
    ran, skipped, fin = [], {}, []
    R = {"sections": {"A0": {"merge": {"ok": merge_ok}}}, "setup": {}}

    def step(sec, name, est, f):
        out = f() or {}
        rec = R["sections"].setdefault(sec, {}).setdefault(name, {"status": "NOT_RUN"})
        rec.update(out)
        if rec.get("status") in (None, "NOT_RUN"):
            rec["status"] = "DONE"
        return rec

    def runner(k):
        ran.append(k)
        return {"status": "LAUNCH_FAILED" if k in fail_arms else "DONE", "pin": 7933526016}

    def skip_arm(k, why, status="SKIPPED_DEPENDENCY", **kw):
        skipped[k] = status

    class _N:
        START_RESERVE_SEC = 300
        ring_status = staticmethod(lambda seq: (True, True, "ring"))
        arm_fits = staticmethod(lambda left, a: left >= a["est_min"] * 60 + 300)
        base_usable = staticmethod(N.base_usable)
        needs_nometa = staticmethod(N.needs_nometa)
        needs_isolation = staticmethod(N.needs_isolation)
    ns = {"ARMS": copy.deepcopy(ARMS), "R": R, "NBDF": {"budget": {}}, "NBS2": {}, "FAKE": True, "step": step, "run_arm": runner,
          "skip_arm": skip_arm, "apply_60490": lambda: {"ok": True}, "k100_setup": lambda: {"ok": True}, "merlin_install": lambda: {},
          "sha_gate": lambda t: (True, "ok"), "SEAM_API": {"ok": True}, "SEAM_LOAD": {}, "DRAFTER": {"ok": drafter_ok, "status": "OK"},
          "KV_KIT_GATE": {"passed": True}, "left": lambda: left_min * 60 - sum(ARMS[k]["est_min"] * 60 for k in ran), "log": lambda *a: None, "save": lambda: None,
          "_patch_rec": lambda q: {}, "_seam_rec": lambda q: {}, "_tree_py": lambda: (None, None), "P": None, "N": _N,
          "nbs2_finalize": lambda: fin.append(1), "json": json, "NBS2_SWITCH": S2.switch_map(switch, tuple(ARMS))}
    sseq = _Seq(fail_check=fail_check, apply_ok=seam_ok)
    pseq = _Seq()
    ns["PSEQ"], ns["SSEQ"] = pseq, sseq
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "nbs2_main", "exec"), ns)
    ns["nbs2_main"](runner=runner, pseq=pseq, sseq=sseq)
    _main_ns.last = ns
    return ran, skipped, fin, sseq, pseq


def t_main():
    if _body_src() is None:
        skip("main", f"no {BODY}")
        return
    ran, sk, fin, ss, ps = _main_ns()
    check("main_all_run", ran == list(ARMS) and not sk and fin == [1], (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(fail_arms=("B",))
    check("main_B_fail_levers_run", ran == list(ARMS) and fin == [1], (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(fail_arms=("IDX", "S28"))
    check("main_lever_fail_only_itself", ran == list(ARMS), ran)
    ran, sk, fin, ss, ps = _main_ns(fail_check=("IDX",))
    check("main_tree_fail_only_itself", ran == [k for k in ARMS if k != "IDX"] and sk == {"IDX": "TREE_INVALID_MIXED"}, (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(seam_ok=False)
    check("main_seam_fail_no_boot", ran == [] and set(sk) == set(ARMS) and fin == [1], (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(drafter_ok=False)
    check("main_drafter_fail_no_boot", ran == [] and set(sk.values()) == {"NOT_INTEGRATED"} and fin == [1], sk)
    ran, sk, fin, ss, ps = _main_ns(left_min=100)
    check("main_budget_walk", ran == ["B", "IDX"] and all(v == "SKIPPED_BUDGET" for v in sk.values()), (ran, sk))
    check("main_seam_first_ppack_before_arm_B", ps.calls[0] == ("before_arm", "B") and ss.calls[0] == "apply" and ss.calls[-1] == "finish"
          and ps.calls[-1] == "finish", (ps.calls[:3], ss.calls[:3]))


def t_switch_lib():
    keys = tuple(ARMS)
    check("switch_keys", S2.SWITCH_KEYS == keys == ("B", "IDX", "S28", "LP", "S32", "SPC") and S2.SWITCH_DEFAULT == SWITCH_DEFAULT, S2.SWITCH_KEYS)
    check("switch_absent_all_on", S2.switch_map(None, keys) == {k: True for k in keys})
    check("switch_default_all_on", S2.switch_map(dict(SWITCH_DEFAULT), keys) == {k: True for k in keys})
    check("switch_missing_key_off", S2.switch_map({"B": True, "S28": True}, keys) == {k: k in ("B", "S28") for k in keys})
    check("switch_empty_all_off", S2.switch_map({}, keys) == {k: False for k in keys})
    check("switch_invalid_all_on", S2.switch_map(["B"], keys) == {k: True for k in keys})
    r = S2.switch_record({"B": False, "IDX": True, "XYZ": True}, keys)
    check("switch_record", r["off"] == ["B", "S28", "LP", "S32", "SPC"] and r["unknown_keys"] == ["XYZ"] and r["absent"] is False
          and r["invalid_not_a_dict"] is False, r)
    check("switch_record_invalid", S2.switch_record("nope", keys)["invalid_not_a_dict"] is True)


def t_switch_main():
    if _body_src() is None:
        skip("switch_main", f"no {BODY}")
        return
    ran, sk, fin, ss, ps = _main_ns(switch=dict(SWITCH_DEFAULT))
    check("switch_all_on_all_launch", ran == list(ARMS) and not sk and fin == [1], (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(switch=None)
    check("switch_absent_all_launch", ran == list(ARMS) and not sk, (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(switch=dict(SWITCH_DEFAULT, B=False))
    check("switch_B_off_levers_launch", ran == list(S2.LEVERS) and sk == {"B": "SKIPPED_SWITCH_OFF"} and fin == [1], (ran, sk))
    check("switch_B_off_ppack_seam_still_applied", ss.calls[0] == "apply" and ps.calls[0] == ("before_arm", "B")
          and ("check", "B") not in ss.calls and ("before_arm", "B") not in ps.calls[1:], (ss.calls[:4], ps.calls[:4]))
    for k in S2.LEVERS:
        ran, sk, fin, ss, ps = _main_ns(switch=dict(SWITCH_DEFAULT, **{k: False}))
        check(f"switch_{k}_off_not_launched", k not in ran and ran == [x for x in ARMS if x != k] and sk == {k: "SKIPPED_SWITCH_OFF"}
              and ("check", k) not in ss.calls, (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(switch={"B": True, "LP": True})
    check("switch_missing_keys_off", ran == ["B", "LP"] and sk == {k: "SKIPPED_SWITCH_OFF" for k in ARMS if k not in ("B", "LP")}, (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(switch=dict(SWITCH_DEFAULT, S28=False), drafter_ok=False)
    check("switch_off_before_other_checks", ran == [] and sk["S28"] == "SKIPPED_SWITCH_OFF"
          and all(v == "NOT_INTEGRATED" for k, v in sk.items() if k != "S28"), sk)
    ran, sk, fin, ss, ps = _main_ns(switch=dict(SWITCH_DEFAULT, IDX=False), merge_ok=False)
    check("switch_off_kept_on_merge_fail", ran == [] and sk["IDX"] == "SKIPPED_SWITCH_OFF"
          and all(v == "SKIPPED_DEPENDENCY" for k, v in sk.items() if k != "IDX"), sk)
    ran, sk, fin, ss, ps = _main_ns(switch=dict(SWITCH_DEFAULT, B=False), left_min=100)
    check("switch_off_frees_budget", ran == ["IDX", "S28"] and sk.get("B") == "SKIPPED_SWITCH_OFF"
          and all(sk[k] == "SKIPPED_BUDGET" for k in ("LP", "S32", "SPC")), (ran, sk))
    ran, sk, fin, ss, ps = _main_ns(switch={k: False for k in ARMS})
    check("switch_all_off_no_setup", ran == [] and set(sk.values()) == {"SKIPPED_SWITCH_OFF"} and len(sk) == len(ARMS)
          and ss.calls == ["finish"] and ps.calls == ["finish"] and fin == [1], (sk, ss.calls, ps.calls))
    check("switch_run_keys_recorded", _main_ns.last["NBS2"].get("run_keys") == [], _main_ns.last["NBS2"])


def _export_cells(f):
    ex = EXPORT.read_bytes()
    cells, i = [], 0
    for n in f["cell_bytes"]:
        cells.append(ex[i:i + n].decode("utf-8"))
        i += n + 2
    return ex, cells


def t_facts():
    if not FACTS.is_file() or not EXPORT.is_file():
        skip("facts", f"no {FACTS} / {EXPORT}")
        return
    f = json.loads(FACTS.read_text(encoding="utf-8"))
    ex, cells = _export_cells(f)
    check("export_sha_facts", hashlib.sha256(ex).hexdigest() == f["export_sha256"])
    check("export_cells_join", b"\n\n".join(c.encode("utf-8") for c in cells) == ex and len(cells) == f["cells"] - 1, (len(cells), f["cells"]))
    check("cell0_e18r2_export", hashlib.sha256(cells[0].encode()).hexdigest() == f["e18_r2_cell0_sha256"] == "abe694be2a35b4dc00bfbf18fe2fdb0916bff15d74f502b5f5836db30cd46ec0")
    check("kv_gate_passed", f["kv_kit_gate"]["passed"] is True and f["kv_kit_gate"]["per_k_success"]["D3s"]["success"] is True)
    check("reproduced_nbdf_k100", f["nbdf_k100_reproduced_sha256"] == NBDF_K100_GATED, f["nbdf_k100_reproduced_sha256"])
    check("closed_nbdf_k100", f["nbdf_k100_closed_sha256"] == NBDF_K100_CLOSED, f["nbdf_k100_closed_sha256"])
    check("base_b1_seam", f["seam_sha256"] == SEAM_B1, f["seam_sha256"])
    check("base_b2_nbdf_lib", f["nbdf_lib_sha256"] == NBDF_LIB_B2 == hashlib.sha256(Path(N.__file__).read_bytes()).hexdigest(), f["nbdf_lib_sha256"])
    check("k100_report_path", f.get("k100_report") == "D:/ARC AGI 2/_scratch/kits_wave3/q27-e27-nbdf-k100-kit-real/nbdf_kit_real_report.json", f.get("k100_report"))
    if BODY.is_file():
        check("body_sha_facts", hashlib.sha256(BODY.read_bytes()).hexdigest() == f["body_file_sha256"] == f["payload_dataset"]["files"]["nbs2_cell7_body.py"])
    lib = Path(S2.__file__)
    check("lib_sha_facts", hashlib.sha256(lib.read_bytes()).hexdigest() == f["nbs2_lib_sha256"] == f["payload_dataset"]["files"]["nbs2_lib.py"])
    stub = cells[-2]
    check("stub_dataset_loader", "_nbs2_ds(\"nbs2_cell7_body.py\", NBDF_BODY_SHA)" in stub and f["body_file_sha256"] in stub)
    check("datasets", f["dataset_sources"] == ["foysalemonshanto/q27-e25-nb0-tf-ref", "foysalemonshanto/q27-e25-nbs2-payload"], f["dataset_sources"])
    check("models_drafter", "foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1" in f["model_sources"])
    check("e27_unchanged_by_build", f.get("e27_files_changed_by_this_build") == [], f.get("e27_files_changed_by_this_build"))
    # the CELL 1 switch line (a plain Python literal in the notebook, outside the sha-pinned payload)
    c1 = cells[1]
    lines = [l for l in ex.decode("utf-8").splitlines() if l.startswith("NBS2_ON =")]
    check("switch_line_once_in_export", len(lines) == 1, lines)
    c1_lines = c1.splitlines()
    sl = [i for i, l in enumerate(c1_lines) if l.startswith("NBS2_ON =")]
    check("switch_line_top_of_cell1", sl and all(l.startswith("#") for l in c1_lines[:sl[0]]) and sl[0] <= 6, sl)
    if sl:
        tree = ast.parse(c1_lines[sl[0]])
        node = tree.body[0] if tree.body else None
        val = ast.literal_eval(node.value) if isinstance(node, ast.Assign) else None
        check("switch_line_literal_default", val == SWITCH_DEFAULT and list(val) == list(ARMS), val)
        check("switch_line_repr", c1_lines[sl[0]] == "NBS2_ON = " + repr(SWITCH_DEFAULT) == f["switch"]["line"], c1_lines[sl[0]])
        check("switch_line_not_in_payload", "NBS2_ON =" not in (BODY.read_text(encoding="utf-8") if BODY.is_file() else "")
              and "NBS2_ON =" not in Path(S2.__file__).read_text(encoding="utf-8"))
        ns = {}
        exec(compile(c1_lines[sl[0]], "cell1_switch", "exec"), ns)
        check("switch_line_runs", S2.switch_map(ns.get("NBS2_ON"), tuple(ARMS)) == {k: True for k in ARMS})
    check("switch_facts", f.get("switch", {}).get("default") == SWITCH_DEFAULT and f["switch"]["in_payload"] is False and f["switch"]["cell"] == 1)
    check("cells_after_1_rerun_guarded", all(c.startswith("if not RERUN:\n") for c in cells[2:]))


def t_notebook():
    if not NBF.is_file() or not FACTS.is_file():
        skip("notebook", f"no {NBF} / {FACTS}")
        return
    data = NBF.read_bytes()
    f = json.loads(FACTS.read_text(encoding="utf-8"))
    check("nb_sha_facts", hashlib.sha256(data).hexdigest() == f["sha256"])
    check("nb_size", len(data) < 400_000, len(data))
    nb = json.loads(data)
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check("cell0_e18r2", hashlib.sha256(code[0].encode()).hexdigest() == f["e18_r2_cell0_sha256"] == "abe694be2a35b4dc00bfbf18fe2fdb0916bff15d74f502b5f5836db30cd46ec0")
    if EXPORT.is_file():
        check("nb_cells_equal_export", "\n\n".join(code).encode("utf-8") == EXPORT.read_bytes())
    check("nb_switch_line_cell1", code[1].count("NBS2_ON = " + repr(SWITCH_DEFAULT) + "\n") == 1)


for t in (t_plan, t_launch_check, t_dense_schedule, t_captures, t_readbacks, t_ledger, t_probes, t_verdict, t_body_text, t_main,
          t_switch_lib, t_switch_main, t_facts, t_notebook):
    try:
        t()
    except Exception as e:
        RES["total"] += 1
        print(f"FAIL {t.__name__} raised {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", flush=True)
print(f"{RES['pass']}/{RES['total']} PASS", flush=True)
sys.exit(0 if RES["pass"] == RES["total"] else 1)
