"""E25 NB2 fault-injection unit tests. Runs the exported notebook code (nb2_code_export.py) with NB2_UNIT=1 (definitions,
A0 merge, prompts; no boots) against a fake input tree, then checks the NB2 logic against the REAL Merlin 3.0.0
launch.build() (local source tree) and the REAL RC2 v1 FP8 server log.

Usage: python nb2_unit_tests.py <fake input dir> <work dir> [<merlin src dir>] [<rc2 gpu output dir>]
"""
import gzip
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INP, WORK = Path(sys.argv[1]), Path(sys.argv[2])
MSRC = Path(sys.argv[3] if len(sys.argv) > 3 else r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
RC2_OUT = Path(sys.argv[4] if len(sys.argv) > 4 else r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2")
shutil.rmtree(WORK, ignore_errors=True)
for d in ("out", "tmp", "shm"):
    (WORK / d).mkdir(parents=True)
os.environ.update(NB2_UNIT="1", E23_FAKE_SERVER=str(HERE / "nb2_fake_server.py"), E23_OUT=str(WORK / "out"),
                  E23_TMP=str(WORK / "tmp"), E23_INPUT=str(INP), NB2_SUBMISSION=str(WORK / "out/submission.json"),
                  NB2_SHM=str(WORK / "shm"), NB2_BUDGET_SEC="20000", NB2_MERLIN_SRC=str(MSRC))
for _k in ("KAGGLE_IS_COMPETITION_RERUN", "NB2_OPTIONAL", "NB2_PROFILE", "NB2_FAKE_HOST_FLOOR_MIB"):
    os.environ.pop(_k, None)


def load(tag):
    return runpy.run_path(str(HERE / "nb2_code_export.py"), run_name=tag)["nb2_arm"].__globals__


ns = load("nb2")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), str(detail)[:300]))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{str(detail)[:300]}]" if detail else ""))


L = ns["L"]
RC2_LOG = (RC2_OUT / "logs" / "server_merlin_fp8.log").read_text(encoding="utf-8", errors="replace")
SHIP = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
CAPS16 = SHIP + [40, 48, 64]          # shipped U capture_sizes_for(16, 3): the one list of A / B / C (A-7)

# 1. every boot's argv from the REAL Merlin launch.build(), and the exact diff vs RC2 FP8
cfgs = {}
for bk, seqs in (("A", None), ("B", None), ("C", None), ("K0", None), ("CTRL", None), ("B", 8)):
    cfg = ns["make_cfg"](ns["BOOTS"][bk], seqs)
    s = ns["merlin_fn_server"](cfg["name"], "fp8", cfg["port"], ns["KV_FP8"], cfg)
    cfgs[f"{bk}{seqs or ''}"] = cfg
fm = {k: ns["flag_map"](c["argv"]) for k, c in cfgs.items()}
spec = lambda k: json.loads(fm[k].get("--speculative-config") or "null")
caps = lambda k: json.loads(fm[k]["--compilation-config"])["cudagraph_capture_sizes"]
check("A (A-7): K3 + draft_lm_head_quant int8 from DRAFT_HEAD (Merlin's own spec string), seqs 16, captures CAPS16",
      spec("A") == {"method": "mtp", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic", "draft_lm_head_quant": "int8"}
      and fm["A"]["--max-num-seqs"] == "16" and caps("A") == CAPS16 and cfgs["A"]["seqs"] == 16, [spec("A"), caps("A")])
dA = cfgs["A"]["exact_diff"]
check("A exact diff vs RC2 FP8: --speculative-config / --max-num-seqs / --compilation-config changed, --profiler-config added, "
      "env DRAFT_HEAD + DEV_MODE", set(dA["flags_changed"]) == {"--speculative-config", "--max-num-seqs", "--compilation-config"}
      and set(dA["flags_added"]) == {"--profiler-config"} and not dA["flags_removed"]
      and set(dA["env_changed"]) == {"DRAFT_HEAD", "VLLM_SERVER_DEV_MODE"}, dA)
check("B: last --speculative-config K2 + int8 + probabilistic, last --max-num-seqs 16, captures cover 16 x 3 = 48 and keep RC2's",
      spec("B") == {"method": "mtp", "num_speculative_tokens": 2, "draft_sample_method": "probabilistic", "draft_lm_head_quant": "int8"}
      and fm["B"]["--max-num-seqs"] == "16" and 48 in caps("B") and set(SHIP) <= set(caps("B")), caps("B"))
check("B argv: trailing overrides come after the profile's flags (duplicates recorded)",
      set(cfgs["B"]["exact_diff"]["duplicate_flags_in_argv"]) >= {"--speculative-config", "--max-num-seqs", "--compilation-config"},
      cfgs["B"]["exact_diff"]["duplicate_flags_in_argv"])
check("C: K1 + int8, seqs 16, captures CAPS16 (the same list as A / B)",
      spec("C")["num_speculative_tokens"] == 1 and spec("C")["draft_lm_head_quant"] == "int8" and fm["C"]["--max-num-seqs"] == "16"
      and caps("C") == CAPS16, caps("C"))
check("A-7: A / B / C at seqs 16 with ONE capture list -> they differ only in K (comparison label K_ONLY)",
      all(cfgs[x]["seqs"] == 16 and cfgs[x]["captures"] == CAPS16 and cfgs[x]["head"] == "int8" for x in "ABC")
      and [cfgs[x]["k"] for x in "ABC"] == [3, 2, 1]
      and ns["comparison_label"](cfgs["B"], cfgs["A"])["label"] == "K_ONLY" and ns["comparison_label"](cfgs["C"], cfgs["A"])["label"] == "K_ONLY",
      {x: (cfgs[x]["seqs"], cfgs[x]["captures"]) for x in "ABC"})
check("comparison labels: B at seqs 8 vs A -> K_CHANGE_PLUS_GRAPH_CONFIG; A vs the RC2 control -> UNMATCHED_CONFIG (head + graph); "
      "CTRL vs control -> SAME_CONFIG; a head-only pair -> HEAD_ONLY",
      ns["comparison_label"](cfgs["B8"], cfgs["A"])["label"] == "K_CHANGE_PLUS_GRAPH_CONFIG"
      and ns["comparison_label"](cfgs["A"], ns["CONTROL_CFG"])["label"] == "UNMATCHED_CONFIG"
      and ns["comparison_label"](cfgs["CTRL"], ns["CONTROL_CFG"])["label"] == "SAME_CONFIG"
      and ns["comparison_label"](dict(cfgs["A"], head="bf16"), cfgs["A"])["label"] == "HEAD_ONLY",
      ns["comparison_label"](cfgs["A"], ns["CONTROL_CFG"]))
check("K0: no --speculative-config anywhere in the argv, seqs 16",
      "--speculative-config" not in cfgs["K0"]["argv"] and fm["K0"]["--max-num-seqs"] == "16"
      and "--speculative-config" in cfgs["K0"]["exact_diff"]["flags_removed"])
dC = cfgs["CTRL"]["exact_diff"]
check("CTRL = the exact RC2 FP8 config (empty diff, no profiler flag)",
      not dC["flags_changed"] and not dC["flags_added"] and not dC["flags_removed"] and not dC["env_changed"], dC)
check("B at seqs 8 (fallback): spec K2, no --max-num-seqs override, captures shipped U capture_sizes_for(8,2)",
      cfgs["B8"]["seqs"] == 8 and fm["B8"]["--max-num-seqs"] == "8" and spec("B8")["num_speculative_tokens"] == 2)
bh = ns["make_cfg"](ns["BOOTS"]["BH"])
check("BH (BF16-head fallback of B): K2 spec without draft_lm_head_quant, no DRAFT_HEAD env, seqs 16, CAPS16",
      bh["k"] == 2 and bh["head"] == "bf16" and "DRAFT_HEAD" not in bh["env"] and bh["seqs"] == 16 and bh["captures"] == CAPS16
      and "draft_lm_head_quant" not in (bh["spec"] or {}) and bh["fallback"] is True, bh["spec"])
# KIT-9: the CTRL exact diff with Merlin's real host_phase _toolchain PATH re-prepend (the Kaggle real-kit case)
_ML = ns["_local_ml"]()
_c13 = "/kit/venv/lib/python3.13/site-packages/nvidia/cu13"
_base = {"GPUS": "0,1,2,3", "PORT": "8015", "KV_BYTES": str(ns["KV_FP8"]), "P2P": "off", "VENV": str(WORK / "tmp/fake_venv"),
         "TREE": str(WORK / "tmp/fake_tree"), "CHAT_TEMPLATE": "", "WARM": "0", "CACHE_ROOT": str(WORK / "tmp/merlin-cache"),
         "SCALES": str(ns["CKPT"] / "qsa_kv_scales_262k.json"), "CUDA_HOME": _c13,
         "PATH": f"/kit/venv/bin:/tmp/merlin_gcc13/bin:{_c13}/bin:/usr/local/bin:/usr/bin"}
_cc = ns["make_cfg"](ns["BOOTS"]["CTRL"])
_x = ["--served-model-name", "q27", "--max-model-len", str(ns["MAX_LEN"]), "--port", "8015", "--default-chat-template-kwargs", ns["XHIGH"]]
_plan = _ML.build("flash-next", ns["PROFILE"], str(ns["CKPT"]), dict(_base), extra=list(_x), kv="fp8", checkpoint="flash-next", status="untested")
_ML._toolchain(_plan, False)                         # what host_phase() does to plan.env before the exact diff
_d = ns["nb2_exact_diff"](_ML, dict(_base), list(_x), "fp8", _cc, _plan.argv(), _plan.env)
check("KIT-9: CTRL exact diff after Merlin's _toolchain re-prepends CUDA_HOME/bin (already on PATH) -> empty; PATH recorded benign",
      _plan.env["PATH"].startswith(_c13 + "/bin:/kit/venv/bin") and not _d["flags_changed"] and not _d["flags_added"]
      and not _d["flags_removed"] and not _d["env_changed"] and _d["env_benign"]["PATH"]["duplicate_prepend"] == [_c13 + "/bin"], _d)
pdp = ns["path_dup_prepend"]
check("KIT-9: path_dup_prepend: a new directory or a reorder is a real change (None); a re-prepended entry is benign",
      pdp("/a:/b", "/b:/a:/b") == ["/b"] and pdp("/a:/b", "/n:/a:/b") is None and pdp("/a:/b", "/b:/a") is None
      and pdp("/a:/b", "/a:/b") is None and pdp(None, "/a") is None)
check("capture_sizes_for = Merlin's registry function when installed, local copy = same values",
      ns["capture_sizes_for"](16, 3)[0] == [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]
      and ns["capture_sizes_for"](16, 2)[0][-1] == 48 and ns["capture_sizes_for"](16, 0)[0][-1] == 16)

# 2. graph coverage = the 0.32 dispatcher rule; RC2's log shows the same FULL count
gp = ns["graph_plan"](SHIP, 3, 8)
lf = ns["log_facts_text"](RC2_LOG)
check("graph_plan K3 seqs 8 = FULL [4,8,12,16,24,32]; RC2 log FULL capture count 6",
      gp["full_decode_sizes"] == [4, 8, 12, 16, 24, 32] and ["FULL", 6] in lf["captures_done"], lf["captures_done"])
for c in ("B", "C", "K0"):
    g = ns["graph_plan"](cfgs[c]["captures"], cfgs[c]["k"], cfgs[c]["seqs"])
    check(f"graph_plan {c}: every request count 1..{cfgs[c]['seqs']} has a FULL decode graph", g["covers_all_request_counts"],
          g["full_decode_sizes"])
g = ns["graph_plan"](SHIP, 3, 16)
check("graph_plan K3 seqs 16 with RC2's list -> gap detected (needs 64)", not g["covers_all_request_counts"])

# 3. readback against the REAL RC2 FP8 log
rb_ctrl = ns["readback"](lf, cfgs["CTRL"])
rb_a = ns["readback"](lf, cfgs["A"])
check("readback: RC2 log vs CTRL config -> ok", rb_ctrl["ok"] is True, rb_ctrl)
check("readback: RC2 log vs boot A config -> NOT ok (head in spec / head built line)",
      rb_a["ok"] is False and rb_a["checks"]["head_in_spec"] is False and rb_a["checks"]["head_built_line"] is False, rb_a)
check("readback: RC2 log vs B (K2, seqs 16) -> NOT ok on k, seqs, captures",
      not any(ns["readback"](lf, cfgs["B"])["checks"][x] for x in ("k", "max_num_seqs", "capture_sizes")))
check("log facts (RC2): MTP KV default scale 1.0, single MTP layer reused, max_num_scheduled_tokens 1024, group sizes",
      lf["mtp_kv_default_scale_1"] and lf["single_mtp_layer_warning"] and lf["max_num_scheduled_tokens"] == 1024
      and lf["kv_cache_group_sizes"] == [4096, 4096, 4096, 8, 4096] and lf["mamba_page_pad_pct"] == 21.01,
      {k: lf[k] for k in ("mtp_kv_default_scale_1", "max_num_scheduled_tokens", "kv_cache_group_sizes")})
check("log facts (RC2): JIT kernels incl. _fill_num_accepted_kernel / _gdn_recoverssm_verify_kernel; draft modes",
      {"_fill_num_accepted_kernel", "_gdn_recoverssm_verify_kernel"} <= set(lf["jit_kernels"])
      and "prefill:FULL_AND_PIECEWISE,decode:FULL_DECODE_ONLY" in lf["draft_identity_modes"], lf["jit_kernels"])
check("log facts (RC2): per-worker model loading (873 s PP1_TP1) and initial free memory parsed",
      round(lf["model_loading"]["Worker_PP1_TP1_EP1"][1]) == 873 and lf["initial_free_gib"]["Worker_PP0_TP0_EP0"][0] == 21.67)
sl = ns["spec_log_parse"](RC2_LOG)
check("SpecDecoding log (RC2): 3 positions, decreasing, accepted/draft 1.8-2.4",
      len(sl["per_position"]) == 3 and sl["per_position"][0] > sl["per_position"][1] > sl["per_position"][2]
      and 1.8 < sl["accept_per_draft"] < 2.4, sl)

# 4. per-position metric (format of vllm/v1/spec_decode/metrics.py) and window_pos on snapshots
txt = ('# HELP x\nvllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="q27",position="0"} 80.0\n'
       'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="q27",position="1"} 60.0\n'
       'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",model_name="q27",position="1"} 4.0\n'
       'vllm:spec_decode_num_drafts_total{engine="0"} 100.0\n')
check("prom_pos_text: per position, summed over other labels", ns["prom_pos_text"](txt) == {0: 80.0, 1: 64.0})
ns["SNAP"]["by_port"][9] = [(10.0, {L.DRF: 100.0}, {0: 50.0, 1: 30.0}), (20.0, {L.DRF: 150.0}, {0: 90.0, 1: 55.0}),
                            (30.0, {L.DRF: 300.0}, {0: 210.0, 1: 140.0})]
wp = ns["window_pos"]({"port": 9, "t0": 10.5, "t1": 30.2}, 2)
check("window_pos: deltas between the m0 / m1 snapshots", wp["per_position"] == [0.8, 0.55] and wp["drafts"] == 200, wp)
check("window_pos: K0 -> N/A", ns["window_pos"]({"port": 9, "t0": 1, "t1": 2}, 0)["status"] == "N/A_NO_MTP")

# 5. F census math, capacity plan
probe = lambda F, B: {"probes": {"2_blocks": {"delta": 4 * (2 + F) / B}, "6_blocks": {"delta": 4 * (6 + F) / B}}}
r = ns["f_from_probes"](probe(7, 149))
check("F census: F 7 / pool 149 recovered", r["status"] == "MEASURED" and r["F_int"] == 7 and abs(r["pool_blocks"] - 149) < 0.5, r)
r = ns["f_from_probes"](probe(4, 200))
check("F census: F 4 / pool 200 recovered", r["F_int"] == 4 and abs(r["pool_blocks"] - 200) < 0.5, r)
check("F census: missing delta -> INCONCLUSIVE", ns["f_from_probes"]({"probes": {"2_blocks": {"delta": None}}})["status"] == "INCONCLUSIVE_CENSUS")
p8 = ns["plan_window"](8, 40000, 7, 149, secs=180)
p9 = ns["plan_window"](8, 40000, 9, 149, secs=180)
p16 = ns["plan_window"](16, 8192, 7, 149, secs=180)
p16b = ns["plan_window"](16, 8192, 5, 149, secs=240)
p12 = ns["plan_window"](12, 8192, 6, 149, secs=240)
check("capacity plan: 40k@8 F7 runs (<= 1.05 x pool), F9 skipped, 8k@16 F7 skipped, 8k@16 F5 runs, 8k@12 F6 runs",
      p8["run"] and not p9["run"] and not p16["run"] and p16b["run"] and p12["run"],
      [p8["blocks_needed"], p9["blocks_needed"], p16["blocks_needed"], p16b["blocks_needed"], p12["blocks_needed"]])

# 6. body gates: fail closed before any load
calls = []


class FakeS:
    port = 9

    def alive(self):
        return True

    def log(self):
        return ""

    def kill(self):
        calls.append("kill")


ORIG = {k: ns[k] for k in ("warmup", "f_census", "ladder", "decay_window", "trace_window", "cohort_replay", "log_facts", "oracle")}
ns["kv_census"] = lambda s: {}
ns["sampled_check"] = lambda s: {"ok": True, "garbage": None}
ns["L"].chat_probes = lambda p: {"ok": True}
for k in ("warmup", "f_census", "ladder", "decay_window", "trace_window", "cohort_replay"):
    ns[k] = (lambda k: lambda *a, **kw: calls.append(k) or ({"status": "x"} if k != "cohort_replay" else {"status": "MEASURED"}))(k)
ns["prefill_probe"] = lambda p: {"ok": True}
ns["PREFILL"]["measured"] = True
GOOD_FACTS = ns["log_facts_text"](RC2_LOG)


def run_body(cfg, facts=GOOD_FACTS, orc=None, sampled=True):
    calls.clear()
    ns["log_facts"] = lambda s: facts
    ns["sampled_check"] = lambda s: {"ok": sampled, "garbage": None}
    ns["oracle"] = lambda p, key=None, refs=(), prompts=None: dict(orc or {"errors": [None] * 4, "garbage": [None] * 4, "row_count": 4,
                                                                           "expected_rows": 4, "complete": True, "vs_rc2_fp8": {"status": "EXACT"}})
    ns["BOOT"].update(deadline=time.time() + 3600, t0=time.time())
    try:
        return ns["nb2_body"](cfg)([FakeS()])
    finally:
        ns["BOOT"].update(deadline=None)


ctrl = dict(cfgs["CTRL"], oracle_refs=("rc2_fp8",))
o = run_body(ctrl)
check("body CTRL happy path: warmup -> census -> ladder -> trace, no decay / no cohort (one window)",
      [c for c in calls] == ["warmup", "f_census", "ladder", "trace_window"], calls)
_fA = ns["log_facts_text"](RC2_LOG.replace("'max_num_seqs': 8", "'max_num_seqs': 16").replace(
    "'cudagraph_capture_sizes': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]", "'cudagraph_capture_sizes': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]")
    .replace("'num_speculative_tokens': 3, 'draft_sample_method': 'probabilistic'}",
             "'num_speculative_tokens': 3, 'draft_sample_method': 'probabilistic', 'draft_lm_head_quant': 'int8'}") +
    chr(10) + "(Worker_PP1_TP0_EP0 pid=1) INFO Draft lm_head: INT8 weight-only copy of the 124160x2560 shard (303.1 MiB; the target keeps its torch.bfloat16 head)")
_fB = dict(_fA, speculative_config=dict(_fA["speculative_config"], num_speculative_tokens=2),
           engine_spec=dict(_fA["engine_spec"] or {}, num_spec_tokens=2), spec_query_len=[3])
o = run_body(dict(cfgs["A"], oracle_refs=("rc2_fp8",)), facts=_fA)
check("A-7: body A (the long-window boot, its own seqs-16 / int8 log) -> warmup, census, cohort, ladder, cohort, DECAY window, trace",
      ns["readback"](_fA, cfgs["A"])["ok"] is True and calls == ["warmup", "f_census", "cohort_replay", "ladder", "cohort_replay",
                                                                  "decay_window", "trace_window"], [calls, ns["readback"](_fA, cfgs["A"])])
o = run_body(dict(cfgs["B"], oracle_refs=("rc2_fp8",)), facts=_fB)
check("A-7: body B -> no decay window (N/A_NOT_THE_LONG_WINDOW_BOOT); cohort start + after ladder kept",
      "decay_window" not in calls and (o.get("decay") or {}).get("status") == "N/A_NOT_THE_LONG_WINDOW_BOOT"
      and calls.count("cohort_replay") == 2 and ns["readback"](_fB, cfgs["B"])["ok"] is True,
      [calls, o.get("decay"), o.get("early_stop"), ns["readback"](_fB, cfgs["B"])])
check("A-7: the long window is boot A's only (BOOTS flags)", [k for k, b in ns["BOOTS"].items() if b.get("long_window")] == ["A"])
cA = dict(cfgs["A"], oracle_refs=("rc2_fp8",))
o = run_body(cA, facts=GOOD_FACTS)
check("body A with the RC2 (bf16-head) log -> CONFIG_READBACK_MISMATCH, no oracle / load", o.get("status") == "NO_GO_CONFIG_READBACK"
      and not calls, calls)
o = run_body(ctrl, orc={"errors": [None] * 4, "garbage": [None] * 4, "row_count": 4, "complete": True,
                        "vs_rc2_fp8": {"status": "FAIL_HIGH_MARGIN"}})
check("body: oracle FAIL_HIGH_MARGIN vs rc2_fp8 -> early stop, no warmup / census / ladder", "ORACLE_GATE" in str(o.get("early_stop"))
      and not calls, o.get("early_stop"))
o = run_body(ctrl, orc={"errors": [None] * 4, "garbage": [None] * 4, "row_count": 3, "complete": False,
                        "vs_rc2_fp8": {"status": "EXACT"}})
check("body: incomplete oracle -> early stop", "ORACLE_GATE" in str(o.get("early_stop")) and not calls, o.get("early_stop"))
o = run_body(ctrl, sampled=False)
check("body: sampled failure -> SHORT gate, no oracle / load", o.get("early_stop") == "SHORT_CORRECTNESS_GATE_FAILED" and not calls)
facts_b = ns["log_facts_text"](RC2_LOG.replace("'max_num_seqs': 8", "'max_num_seqs': 16").replace(
    "'num_speculative_tokens': 3, 'draft_sample_method': 'probabilistic'}",
    "'num_speculative_tokens': 3, 'draft_sample_method': 'probabilistic', 'draft_lm_head_quant': 'int8'}") +
    "\n(Worker_PP1_TP0_EP0 pid=1) INFO Draft lm_head: INT8 weight-only copy of the 124160x2560 shard (303.1 MiB; the target keeps its torch.bfloat16 head)")
rbA = ns["readback"](facts_b, dict(cfgs["A"], seqs=16, captures=SHIP))
check("readback: an int8-head + seqs-16 log matches a K3 / int8 / seqs-16 config (spec dict + 'Draft lm_head: INT8' line)",
      rbA["ok"] is True and rbA["checks"]["head_in_spec"] and rbA["checks"]["head_built_line"], rbA)
rbA2 = ns["readback"](facts_b, dict(cfgs["A"], seqs=16, captures=[1, 2, 4]))
check("readback: same log vs a different capture list -> capture_sizes False", rbA2["ok"] is False and not rbA2["checks"]["capture_sizes"])

# 6b. ladder: capacity plan, seqs cap, substitution of the highest admitted level, per-window fields
lv_calls = []


def fake_levels(port, band, lv, server=None, **kw):
    lv_calls.append((band, list(lv)))
    return {f"{band}@{n}": {"status": "STEADY", "level": n, "port": port, "t0": 1.0, "t1": 2.0, "running_min": n, "running_max": n,
                            "running_mean": n, "waiting_max": 0, "step_ms": 40.0, "tokens_per_step": n / 2 * 3.0,
                            "mean_acceptance_length": 3.0, "gen_tokens": 900 * n, "admission": {"tail": [[1, n, 0, n, n]]}} for n in lv}


ns["levels"] = fake_levels
ns["BOOT"].update(deadline=time.time() + 3600)
lc = dict(cfgs["B"])
o = {"f_census": {"status": "MEASURED", "F_int": 7, "pool_blocks": 149.0}, "log_facts": {"max_num_seqs": 16}}
SEQ0 = dict(ns["SEQS16"])
st = ORIG["ladder"](FakeS(), lc, o)
check("ladder (F 7, pool 149, seqs 16): 8k@16 predicted over the pool -> skipped, highest admitted level 14 run instead",
      st["8k@16"]["status"] == "SKIPPED_KV_PREDICTED" and st["8k@16"].get("substituted_level") == 14 and "8k@14" in st
      and ("8k", [8, 12, 14]) in lv_calls, [lv_calls, st["8k@16"].get("substituted_level")])
w = st["8k@12"]
check("ladder window fields: real running == level, all prefilled, tick_ms / cycle_ms, PP2 tokens/step check, live blocks",
      w["real_running_eq_level"] and w["all_prefilled_at_start"] and w["tick_ms"] == 40.0 and w["cycle_ms"] == 80.0
      and w["tokens_per_step_ratio"] == 1.0 and w["running_blocks_est"] == 12 * (3 + 7), w)
check("ladder: a window with > 8 running sets SEQS16 accepted", ns["SEQS16"]["accepted"] is True, ns["SEQS16"])
check("ladder: accepted output tokens per verify token = a / (1 + K) (B: K2, a 3.0 -> 1.0)",
      w["a_output_per_cycle"] == 3.0 and w["accepted_per_verify_token"] == 1.0, w)
gpB = ns["graph_plan"](cfgs["B"]["captures"], 2, 16)
mbs = {n: ns["microbatch_cover"](gpB, n, 2) for n in (5, 8, 12, 16)}
gpA16 = ns["graph_plan"](SHIP, 3, 32)
check("micro-batch coverage (addendum 5): K2 16 streams -> 8 x 3 = 24 tokens covered; K3 32 streams on RC2's list -> 64 not covered",
      mbs[16]["tokens"] == 24 and mbs[16]["covered"] and all(v["covered"] for v in mbs.values())
      and not ns["microbatch_cover"](gpA16, 32, 3)["covered"] and ns["microbatch_cover"](gpA16, 16, 3)["tokens"] == 32
      and ns["microbatch_cover"](gpA16, 16, 3)["covered"], {n: (v["tokens"], v["full_graph"]) for n, v in mbs.items()})
lv_calls.clear()
st = ORIG["ladder"](FakeS(), lc, {"f_census": {"status": "MEASURED", "F_int": 6, "pool_blocks": 149.0}, "log_facts": {"max_num_seqs": 8}})
check("ladder: server cap read back as 8 -> 8k@12 / 8k@16 SKIPPED_SEQS_CAP, never run", st["8k@12"]["status"] == "SKIPPED_SEQS_CAP"
      and st["8k@16"]["status"] == "SKIPPED_SEQS_CAP" and all(max(l) <= 8 for _, l in lv_calls), lv_calls)
rl = WORK / "req_test.jsonl"
NL = chr(10)
rl.write_text("".join(json.dumps({"port": 9, "prompt_len": 40000, "t": 100.0 + i, "ttft": 1.0}) + NL for i in range(5)) +
              "".join(json.dumps({"port": 9, "prompt_len": 40000, "t": 130.0 + i, "ttft": 9.0 + i}) + NL for i in range(8)) +
              json.dumps({"port": 9, "prompt_len": 8192, "t": 131.0, "ttft": 99.0}) + NL)
_rl0, ns["REQ_LOG"] = ns["REQ_LOG"], rl
tt = ns["ttft_summary"](9, 40000, 0, 200, 8)
check("ttft_summary: the window's own n rows (last n of the band, port), not the previous window's", tt == {"n": 8, "max": 16.0, "median": 12.5}, tt)
ns["REQ_LOG"] = _rl0
ns["SEQS16"].clear()
ns["SEQS16"].update(SEQ0)
ns["BOOT"].update(deadline=None)

check("reset_prefix_cache: HTTP 200 + success false (blocks held) is NOT ok; success true is ok",
      ns["reset_ok"](200, '{"success": false}') is False and ns["reset_ok"](200, '{"success": true}') is True
      and ns["reset_ok"](200, "") is False and ns["reset_ok"](404, '{"success": true}') is False)

# 7. main-flow dependencies (step / nb2_arm stubbed)
ns2 = load("nb2b")
ns2["sha_gate"] = lambda t: (True, "ok")
ns2["merlin_install"] = lambda: {"fake": True}
ns2["scorecard"] = lambda: None
order = []


HOOK = {}


def fake_arm(results):
    def arm(bk, seqs=None):
        order.append((bk, seqs))
        if bk in HOOK:
            HOOK[bk]()
        return dict(results.get((bk, seqs)) or results.get(bk) or {"status": "DONE", "correctness": "CLEAN"})
    return arm


def run_main(results, opt=("K0", "CTRL")):
    order.clear()
    ns2["R"]["sections"]["S2"] = {}
    ns2["SEQS16"].update(accepted=None, why="t")
    ns2["NB2_OPT"][:] = list(opt)
    ns2["nb2_arm"] = fake_arm(results)
    ns2["ROWS"]["A"] = [{"stale": True}]
    ns2["HARD_STOP"] = time.time() + 20000
    ns2["nb2_main"]()
    return list(order)


pl = ns2["nb2_plan"]()
check("A-7 default: NB2_OPTIONAL empty -> K0 / CTRL off; the plan = A / B / C at seqs 16, long window only in A",
      ns2["NB2_OPT"] == [] and [b["key"] for b in pl["boots"]] == ["A", "B", "C"] and [b["seqs"] for b in pl["boots"]] == [16, 16, 16]
      and [b["long_window"] for b in pl["boots"]] == [True, False, False], pl)
check("plan minutes: A 72 + B 45 + C 45 = 162 + setup 10 = 172 min (~2.9 h) inside the 265-min FULL cap; fallback conditional",
      pl["boot_min"] == 162 and pl["total_est_min"] == 172 and "BH / CH" in pl["conditional"]
      and "BUDGET = int(os.environ.get(\"NB2_BUDGET_SEC\") or (35 if SMOKE else 265) * 60)" in open(HERE / "nb2_code_export.py", encoding="utf-8").read(), pl)
o = run_main({"A": {"status": "HOST_RAM_GATE"}})
check("main: A fails for a non-head reason (HOST_RAM_GATE) -> B / C SKIPPED_DEPENDENCY, no fallback; A's rows dropped",
      o[:1] == [("A", 16)] and not any(b in ("B", "C", "BH", "CH") for b, _ in o)
      and ns2["R"]["sections"]["S2"]["B_skipped"]["status"] == "SKIPPED_DEPENDENCY" and "fallback" not in ns2["R"]["sections"]["S2"]
      and "A" not in ns2["ROWS"], o)
RB_HEAD = {"ok": False, "checks": {"non_default_args_line": True, "max_num_seqs": True, "capture_sizes": True, "k": True,
                                    "head_in_spec": False, "head_built_line": False, "probabilistic": True, "spec_query_len": True}}
o = run_main({"A": {"status": "NO_GO_CONFIG_READBACK", "correctness": "INCONCLUSIVE_NO_ORACLE", "config_readback": RB_HEAD}}, opt=())
fb = ns2["R"]["sections"]["S2"].get("fallback") or {}
check("fallback: A fails for an INT8-head reason (head not read back) -> BH (K2 bf16) then CH (K1 bf16) at seqs 16; B / C replaced",
      o == [("A", 16), ("BH", 16), ("CH", 16)] and fb.get("reason", "").startswith("INT8_HEAD_READBACK")
      and fb["boots"]["BH"]["decision"] == "RUN_BF16_HEAD" and fb["boots"]["CH"]["decision"] == "RUN_BF16_HEAD"
      and ns2["R"]["sections"]["S2"]["B_skipped"]["status"] == "REPLACED_BY_BF16_HEAD_FALLBACK", [o, fb])
o = run_main({"A": {"status": "NO_GO_CONFIG_READBACK", "config_readback": RB_HEAD},
              "BH": {"status": "NO_GO_ORACLE", "correctness": "NO_GO_ORACLE"}}, opt=())
fb = ns2["R"]["sections"]["S2"].get("fallback") or {}
check("fallback: BH not clean -> CH skipped (dependency)", o == [("A", 16), ("BH", 16)]
      and fb["boots"]["CH"]["decision"].startswith("SKIPPED_DEPENDENCY"), [o, fb])
HOOK["A"] = lambda: ns2.update(HARD_STOP=time.time() + 30 * 60)      # A used the budget up: 30 min left < BH 45 + 5
o = run_main({"A": {"status": "NO_GO_CONFIG_READBACK", "config_readback": RB_HEAD}}, opt=())
HOOK.clear()
fb = ns2["R"]["sections"]["S2"].get("fallback") or {}
check("fallback: no budget left (30 min < 45 + 5) -> SKIPPED_FALLBACK_NO_BUDGET recorded with the reason, no BH / CH boot",
      o == [("A", 16)] and fb["boots"]["BH"]["decision"] == "SKIPPED_FALLBACK_NO_BUDGET" and fb["boots"]["BH"]["left_min"] < 31
      and fb.get("reason", "").startswith("INT8_HEAD") and ns2["R"]["sections"]["S2"]["B_skipped"]["status"] == "SKIPPED_DEPENDENCY", [o, fb])
ihf = ns2["int8_head_failure"]
srv = lambda tail: {"servers": {"x": {"log_tail": tail}}}
check("int8_head_failure: head launch error / acceptance collapse / oracle fail with readback ok -> reason; OOM, generic launch, "
      "K readback, host gate, deadline -> None",
      str(ihf(dict(srv("ValueError: draft_lm_head_quant='int8' is not supported for this draft model"), status="LAUNCH_FAILED"))).startswith("INT8_HEAD_LAUNCH_FAILED")
      and str(ihf({"status": "NO_GO_ACCEPTANCE_COLLAPSE"})).startswith("INT8_HEAD_ACCEPTANCE_COLLAPSE")
      and str(ihf({"status": "NO_GO_ORACLE", "correctness": "NO_GO_ORACLE", "config_readback": {"ok": True}})).startswith("INT8_HEAD_OR_GRAPH_SUSPECT")
      and ihf({"status": "NO_GO_ORACLE", "correctness": "NO_GO_ORACLE"}) is None
      and ihf(dict(srv("torch.OutOfMemoryError: CUDA out of memory"), status="LAUNCH_FAILED")) is None
      and ihf(dict(srv("INFO Draft lm_head: INT8 weight-only copy" + NL + "RuntimeError: NCCL timeout"), status="SERVER_DIED")) is None
      and ihf({"status": "NO_GO_CONFIG_READBACK", "config_readback": {"ok": False, "checks": dict(RB_HEAD["checks"], k=False)}}) is None
      and ihf({"status": "HOST_RAM_GATE"}) is None and ihf({"status": "BOOT_DEADLINE"}) is None)
o = run_main({("B", 16): {"status": "LAUNCH_FAILED", "tries": [{"oom": False}]}})
check("main: B launch fails at seqs 16 (no OOM) -> B retried at 8, C and K0 at 8", o[:5] == [("A", 16), ("B", 16), ("B", 8), ("C", 8), ("K0", 8)], o)
o = run_main({("A", 16): {"status": "LAUNCH_FAILED", "tries": [{"oom": False}]}}, opt=())
check("main: A launch fails at seqs 16 (no OOM) -> A retried at 8, then B / C at 8", o == [("A", 16), ("A", 8), ("B", 8), ("C", 8)], o)
o = run_main({("B", 16): {"status": "LAUNCH_FAILED", "tries": [{"oom": True}, {"oom": False}]}})
check("main: B launch OOM at 16 -> no seqs-8 retry (RC2's KV retry already ran), C still at 16", ("B", 8) not in o and ("C", 16) in o, o)
o = run_main({}, opt=())
check("main: optional boots off -> A, B, C only (all at seqs 16)", o == [("A", 16), ("B", 16), ("C", 16)], o)
ns2["R"]["sections"]["A0"]["merge"] = {"ok": False}
o = run_main({})
check("main: merge failed -> no boot at all", o == [] and ns2["R"]["sections"]["S2"]["skipped"]["status"] == "SKIPPED_MERGE_FAILED")

# 7b. SMOKE profile (NB2_PROFILE=SMOKE, the 35-min default budget)
_bud = os.environ.pop("NB2_BUDGET_SEC")
os.environ["NB2_PROFILE"] = "SMOKE"
nsS = load("nb2s")
os.environ["NB2_BUDGET_SEC"] = _bud
os.environ.pop("NB2_PROFILE")
plS = nsS["nb2_plan"]()
check("SMOKE plan: profile SMOKE, 35-min budget, boot B only (K2 + int8 + seqs 16), 8k@8 + 8k@16 admission check, est 24 + setup 10 <= 35",
      nsS["SMOKE"] is True and nsS["BUDGET"] == 35 * 60 and [b["key"] for b in plS["boots"]] == ["B"]
      and plS["boots"][0]["seqs"] == 16 and plS["boots"][0]["head"] == "int8" and plS["boots"][0]["k"] == 2
      and plS["boots"][0]["ladder"] == ["8k@8", "8k@16 admission check"] and plS["total_est_min"] <= 35
      and nsS["BOOTS"]["B"]["est_min"] * 60 + 300 <= nsS["BUDGET"] - 4.5 * 60, plS)
nsS["sha_gate"] = lambda t: (True, "ok")
nsS["merlin_install"] = lambda: {"fake": True}
nsS["scorecard"] = lambda: None
SMOKE_OK = {"status": "DONE", "correctness": "CLEAN", "config_readback": {"ok": True}, "steady": {"8k@8": {"status": "STEADY"}},
            "admission_16": {"status": "ADMITTED_ALL"}, "trace": {"nb2": {"ranks": {"pp0_tp0": {}}}}}
for res, want in ((SMOKE_OK, "SMOKE_PASS"), (dict(SMOKE_OK, admission_16={"status": "NOT_ALL_ADMITTED"}), "SMOKE_FAIL")):
    order.clear()
    nsS["R"]["sections"]["S2"] = {}
    nsS["NB2_OPT"][:] = ["K0", "CTRL"]                 # ignored in SMOKE
    nsS["nb2_arm"] = fake_arm({"B": res})
    nsS["nb2_main"]()
    check(f"SMOKE main: boot B at 16 only (NB2_OPTIONAL ignored) -> {want}", order == [("B", 16)]
          and nsS["R"]["smoke"]["verdict"] == want, [order, nsS["R"]["smoke"]])
check("SMOKE verdict lists the failed check", nsS["R"]["smoke"]["failed"] == ["admission_16"], nsS["R"]["smoke"])
callsS = []
for k in ("warmup", "f_census", "ladder", "decay_window", "trace_window", "cohort_replay", "admission_check"):
    nsS[k] = (lambda k: lambda *a, **kw: callsS.append(k) or {"status": "x"})(k)
nsS["kv_census"] = lambda s: {}
nsS["sampled_check"] = lambda s: {"ok": True, "garbage": None}
nsS["L"].chat_probes = lambda p: {"ok": True}
_wi = nsS["L"].wait_idle
nsS["L"].wait_idle = lambda p, w=90: True
nsS["reset_prefix_cache"] = lambda p: {"ok": True}
nsS["PREFILL"]["measured"] = True
nsS["log_facts"] = lambda s: _fB
nsS["oracle"] = lambda p, key=None, refs=(), prompts=None: {"errors": [None] * 4, "garbage": [None] * 4, "row_count": 4,
                                                            "expected_rows": 4, "complete": True, "vs_rc2_fp8": {"status": "EXACT"}}
cS = dict(nsS["make_cfg"](nsS["BOOTS"]["B"]), oracle_refs=("rc2_fp8",))
nsS["BOOT"].update(deadline=time.time() + 3600, t0=time.time())
oS = nsS["nb2_body"](cS)([FakeS()])
nsS["BOOT"].update(deadline=None)
nsS["L"].wait_idle = _wi
check("SMOKE body: readback -> oracle -> 1 warm-up pass -> dev-mode reset -> 8k@8 ladder -> 8k@16 admission -> trace; no census / "
      "cohort / decay", callsS == ["warmup", "ladder", "admission_check", "trace_window"] and oS["decay"]["status"] == "N/A_SMOKE"
      and cS["ladder"] == [("8k", 8)] and cS["smoke"] is True and cS["captures"] == CAPS16 and oS["dev_mode_reset"]["ok"] is True,
      [callsS, oS.get("early_stop"), oS.get("config_readback")])


class FStream:
    def __init__(self, port, ids, max_new, **kw):
        self.t_first, self.err, self.max_new = None, None, max_new

    def start(self):
        self.t_first = 1.0

    def is_alive(self):
        return True

    def stop(self):
        pass

    def join(self, t=None):
        pass


_saved = (L.Stream, L.prom, L.wait_idle)
L.Stream, L.wait_idle = FStream, (lambda p, w=90: True)
L.prom = lambda p: {L.RUN: 16.0, L.WAIT: 0.0, L.PREEMPT: 0.0}
_s16 = dict(ns["SEQS16"])
ac = ns["admission_check"](FakeS(), cfgs["B"], {"log_facts": {"max_num_seqs": 16}}, n=16)
ac8 = ns["admission_check"](FakeS(), cfgs["B"], {"log_facts": {"max_num_seqs": 8}}, n=16)
L.prom = lambda p: {L.RUN: 8.0, L.WAIT: 8.0, L.PREEMPT: 0.0}
acn = ns["admission_check"](FakeS(), cfgs["B"], {"log_facts": {"max_num_seqs": 16}}, n=16, timeout=2)
L.Stream, L.prom, L.wait_idle = _saved
check("SMOKE admission check: 16 x 8k planned inside the pool (16 x (3 + F 6) = 144 blocks), all 16 running -> ADMITTED_ALL + SEQS16 "
      "accepted; server cap 8 -> SKIPPED_SEQS_CAP; 8 running + 8 waiting -> NOT_ALL_ADMITTED",
      ac["plan"]["run"] and ac["plan"]["blocks_needed"] == 144 and ac["status"] == "ADMITTED_ALL" and ac["running_max"] == 16
      and ns["SEQS16"]["accepted"] is True and ac8["status"] == "SKIPPED_SEQS_CAP" and acn["status"] == "NOT_ALL_ADMITTED"
      and acn["running_max"] == 8, [ac, acn.get("status")])
ns["SEQS16"].clear()
ns["SEQS16"].update(_s16)


# 7c. DIAG profiles (SMOKE_SEED2 / SMOKE_CTRL): plans, seed override, CTRL = exact RC2, window never stopped, per-stream table
def load_profile(prof, tag):
    bud = os.environ.pop("NB2_BUDGET_SEC")
    os.environ["NB2_PROFILE"] = prof
    try:
        return load(tag)
    finally:
        os.environ["NB2_BUDGET_SEC"] = bud
        os.environ.pop("NB2_PROFILE")


nsD = load_profile("SMOKE_SEED2", "nb2d")
nsC = load_profile("SMOKE_CTRL", "nb2ctrl")
plD, plC = nsD["nb2_plan"](), nsC["nb2_plan"]()
check("SMOKE_SEED2 plan: 35-min budget, DIAG, boot B only (K2 + int8 + seqs 16, CAPS16), 8k@8 + 8k@16 admission, window seed 4321",
      nsD["SMOKE"] is True and nsD["DIAG"] is True and nsD["BUDGET"] == 35 * 60 and [b["key"] for b in plD["boots"]] == ["B"]
      and (plD["boots"][0]["k"], plD["boots"][0]["head"], plD["boots"][0]["seqs"]) == (2, "int8", 16)
      and plD["boots"][0]["ladder"] == ["8k@8", "8k@16 admission check"] and plD["window_seed"] == 4321 and plD["total_est_min"] <= 35
      and nsD["make_cfg"](nsD["BOOTS"]["B"])["captures"] == CAPS16, plD)
check("SMOKE_CTRL plan: 35-min budget, DIAG, boot CTRL only (K3 + bf16 + seqs 8), 8k@8 only (no 16-stream admission), seed 1234",
      nsC["SMOKE"] is True and nsC["DIAG"] is True and nsC["BUDGET"] == 35 * 60 and [b["key"] for b in plC["boots"]] == ["CTRL"]
      and (plC["boots"][0]["k"], plC["boots"][0]["head"], plC["boots"][0]["seqs"]) == (3, "bf16", 8)
      and plC["boots"][0]["ladder"] == ["8k@8"] and plC["window_seed"] == 1234 and plC["total_est_min"] <= 35
      and nsC["BOOTS"]["CTRL"]["est_min"] * 60 + 300 <= nsC["BUDGET"] - 4.5 * 60, plC)
check("profiles: FULL / SMOKE are not DIAG and keep window seed 1234; an unknown profile is refused",
      ns["DIAG"] is False and nsS["DIAG"] is False and ns["WINDOW_SEED"] == 1234 and nsS["WINDOW_SEED"] == 1234
      and ns["NB2_PROFILES"] == ("FULL", "SMOKE", "SMOKE_SEED2", "SMOKE_CTRL") and "diag_prompt_check" not in ns["R"]
      and "if NB2_PROFILE not in NB2_PROFILES and not RERUN:" in open(HERE / "nb2_code_export.py", encoding="utf-8").read())
SEEN = {}


def seed_levels(tag):
    def lv(port, band, lv_, server=None, **kw):
        SEEN[tag] = dict(kw)
        return {f"{band}@{n}": {"status": "STEADY", "level": n} for n in lv_}
    return lv


import types
for tag, nx, bk, cap in (("FULL", ns, "B", 16), ("SMOKE", nsS, "B", 16), ("SEED2", nsD, "B", 16), ("CTRL", nsC, "CTRL", 8)):
    _lv0 = nx["levels"]
    nx["levels"] = seed_levels(tag)
    nx["BOOT"].update(deadline=time.time() + 3600)
    _lad = types.FunctionType(ORIG["ladder"].__code__, nx)      # the real ladder() on this profile's globals (FULL / SMOKE stubbed it)
    _lad(FakeS(), dict(nx["make_cfg"](nx["BOOTS"][bk]), ladder=[("8k", 8)]),
         {"f_census": {"status": "MEASURED", "F_int": 6, "pool_blocks": 149.0}, "log_facts": {"max_num_seqs": cap}})
    nx["BOOT"].update(deadline=None)
    nx["levels"] = _lv0
check("seed override only in SMOKE_SEED2: the 8k@8 window gets seed 4321 there, 1234 (= steady()'s default, the failed SMOKE) "
      "in FULL / SMOKE / SMOKE_CTRL; temperature 0.6 everywhere",
      {k: v.get("seed") for k, v in SEEN.items()} == {"FULL": 1234, "SMOKE": 1234, "SEED2": 4321, "CTRL": 1234}
      and all(v.get("temperature") == 0.6 for v in SEEN.values()), {k: v.get("seed") for k, v in SEEN.items()})
cC = nsC["make_cfg"](nsC["BOOTS"]["CTRL"])
nsC["merlin_fn_server"](cC["name"], "fp8", cC["port"], nsC["KV_FP8"], cC)
dCC = cC["exact_diff"]
check("SMOKE_CTRL boot = the exact RC2 FP8 config (real Merlin build(): empty diff, no env, no trailing flag, shipped captures) "
      "with the window seed 1234", not any(dCC.get(k) for k in ("flags_changed", "flags_added", "flags_removed", "env_changed"))
      and "error" not in dCC and cC["env"] == {} and cC["extra"] == [] and cC["captures"] == SHIP and cC["exact_rc2"] is True
      and cC["smoke"] is True and cC["diff_vs_rc2_fp8"] == ["none (exact RC2 FP8)"] and nsC["WINDOW_SEED"] == 1234, dCC)
fs = nsD["NB2_FAILED_SMOKE"]
check("failed SMOKE embedded: trace rows 37-44 = streams 0-7, seeds 1234-1241, stream 0 looped (LOW_VARIETY), the rest clean",
      [r["row"] for r in fs["rows"]] == list(range(37, 45)) and [r["stream"] for r in fs["rows"]] == list(range(8))
      and [r["seed"] for r in fs["rows"]] == list(range(1234, 1242)) and fs["rows"][0]["tail_verdict"] == "LOW_VARIETY"
      and not any(r["tail_verdict"] for r in fs["rows"][1:]) and fs["window"]["status"] == "FAIL_GARBAGE_UNDER_LOAD", fs["rows"][0])
pc = nsD["R"]["diag_prompt_check"]
_got = [nsD["_psha"](p) for p in nsD["PROMPTS"]["8k"][:8]]
check("DIAG prompt check: this run's 8k@8 prompt shas (stream order) vs the failed rows, recorded at load (local ARC copy "
      + ("matches" if pc["same_set_and_order"] else "differs") + ")", pc["this_run"] == _got
      and pc["failed_run_rows_37_44"] == [r["prompt_sha"] for r in fs["rows"]] and pc["same_set_and_order"] == (_got == pc["failed_run_rows_37_44"])
      and pc["mismatch_streams"] == [i for i in range(8) if _got[i] != pc["failed_run_rows_37_44"][i]], pc["mismatch_streams"])

# 7c-2: the LAB window (steady()) runs to its full length when a stream loops, keeps every stream, then fails the window


def _word(i):
    return "".join(chr(97 + (i // 26 ** k) % 26) for k in range(3))


class GStream:
    LOOP_SEED, MADE = 4324, []

    def __init__(self, port, ids, max_new, temperature=1.0, seed=None, **kw):
        self.seed, self.t0, self.t_first, self.err, self.t_stop = seed, None, None, None, None
        GStream.MADE.append(self)

    def start(self):
        self.t0 = self.t_first = time.perf_counter()

    def _n(self):
        return int(((self.t_stop or time.perf_counter()) - self.t0) * 400) + 1 if self.t0 else 0

    @property
    def tokens(self):
        return [7] * self._n()

    @property
    def text(self):
        n = self._n()
        return [" row?" * n if self.seed == GStream.LOOP_SEED else " ".join(_word(i + 999 * (self.seed or 0)) for i in range(n))]

    def is_alive(self):
        return self.t_stop is None

    def stop(self):
        self.t_stop = time.perf_counter()

    def join(self, t=None):
        pass


_T0 = time.perf_counter()


def _gprom(port):
    dt = time.perf_counter() - _T0
    return {L.RUN: 8.0, L.WAIT: 0.0, L.PROMPT: 65536.0, L.PREEMPT: 0.0, L.GEN: 3200.0 * dt, L.ITER: 300.0 * dt, L.ACC: 900.0 * dt,
            L.DRF: 1200.0 * dt, L.DTOK: 1200.0 * dt, L.KVU: 0.5}


_saved = (L.Stream, L.prom, L.wait_idle)
L.Stream, L.prom, L.wait_idle = GStream, _gprom, (lambda p, w=90: True)
try:
    gw = L.steady(9, [[1, 2, 3]] * 8, 8, secs=2.0, max_new=1000, temperature=0.6, seed=4321, sample=0.2, gpu_sample=False)
finally:
    L.Stream, L.prom, L.wait_idle = _saved
check("LAB steady(): one looping stream (seed 4324) -> the 2-s window still runs to its end (window_sec >= 2), every stream is "
      "stopped only after the window, seeds 4321-4328, then FAIL_GARBAGE_UNDER_LOAD with one garbage verdict",
      gw.get("status") == "FAIL_GARBAGE_UNDER_LOAD" and gw.get("window_sec", 0) >= 2.0 and len(GStream.MADE) == 8
      and all(g.t_stop is not None and g.t_stop >= gw["t1"] for g in GStream.MADE)
      and [g.seed for g in GStream.MADE] == list(range(4321, 4329)) and gw["garbage"]["garbage"] == ["LOW_VARIETY"]
      and gw["garbage"]["with_output"] == 8, {k: gw.get(k) for k in ("status", "window_sec", "garbage")})

# 7c-3: DIAG bodies: the garbage window never stops the boot (table, admission, trace still run)
TABLE_D, VERDICT_D, VERDICT_C = nsD["diag_stream_table"], nsD["diag_verdict"], nsC["diag_verdict"]
TABLE_C = nsC["diag_stream_table"]
callsD = []
LP_AT_LADDER = {}
GARB_W ={"8k@8": {"status": "FAIL_GARBAGE_UNDER_LOAD", "window_sec": 90.0, "garbage": {"garbage": ["LOW_VARIETY"]}}}
for nx, facts in ((nsD, _fB), (nsC, GOOD_FACTS)):
    nx["warmup"] = lambda *a, **kw: callsD.append("warmup") or {"clean": True}
    nx["ladder"] = (lambda nx_: lambda *a, **kw: callsD.append("ladder") or LP_AT_LADDER.update({nx_["NB2_PROFILE"]: nx_["DIAG_LP"]["port"]})
                    or json.loads(json.dumps(GARB_W)))(nx)
    nx["diag_stream_table"] = lambda *a, **kw: callsD.append(("diag_stream_table",) + a) or {"rows_found": 8, "loop_count": 1,
                                                                                              "looped_streams": [0], "streams": []}
    nx["kv_census"] = lambda s: {}
    nx["sampled_check"] = lambda s: {"ok": True, "garbage": None}
    nx["log_facts"] = (lambda f: lambda s: f)(facts)
    nx["oracle"] = lambda p, key=None, refs=(), prompts=None: {"errors": [None] * 4, "garbage": [None] * 4, "row_count": 4,
                                                               "expected_rows": 4, "complete": True, "vs_rc2_fp8": {"status": "EXACT"}}
    nx["PREFILL"]["measured"] = True
nsD["admission_check"] = lambda *a, **kw: callsD.append("admission_check") or {"status": "ADMITTED_ALL"}
nsD["trace_window"] = lambda *a, **kw: callsD.append("trace_window") or {"status": "x"}
nsD["reset_prefix_cache"] = lambda p: {"ok": True}
_chat0 = L.chat_probes
L.chat_probes = lambda p: {"ok": True}
L.wait_idle = lambda p, w=90: True
try:
    cD = dict(nsD["make_cfg"](nsD["BOOTS"]["B"]), oracle_refs=("rc2_fp8",))
    nsD["BOOT"].update(deadline=time.time() + 3600, t0=time.time())
    oD = nsD["nb2_body"](cD)([FakeS()])
    callsD_D = list(callsD)
    callsD.clear()
    cCb = dict(cC, oracle_refs=("rc2_fp8",))
    nsC["BOOT"].update(deadline=time.time() + 3600, t0=time.time())
    oC = nsC["nb2_body"](cCb)([FakeS()])
finally:
    L.chat_probes, L.wait_idle = _chat0, _saved[2]
    nsD["BOOT"].update(deadline=None)
    nsC["BOOT"].update(deadline=None)
check("SMOKE_SEED2 body: garbage 8k@8 window -> per-stream table (port, 8 streams, seed 4321, 8k, max_new) -> 8k@16 "
      "admission -> trace still run; dev-mode reset kept",
      [c if isinstance(c, str) else c[0] for c in callsD_D] == ["warmup", "ladder", "diag_stream_table", "admission_check", "trace_window"]
      and callsD_D[2][1:5] == (9, 8, 4321, 8192) and callsD_D[2][5] == nsD["MAX_NEW"]["8k"] and oD["diag_streams"]["loop_count"] == 1
      and oD["dev_mode_reset"]["ok"] is True and not oD.get("early_stop"), [callsD_D, oD.get("early_stop")])
check("SMOKE_CTRL body: garbage window -> table (seed 1234) still built; no dev-mode reset (exact RC2), 8k@16 admission SKIPPED_SEQS_CAP "
      "(seqs 8), trace SKIPPED (no profiler flag)",
      [c if isinstance(c, str) else c[0] for c in callsD] == ["warmup", "ladder", "diag_stream_table"] and callsD[2][3] == 1234
      and oC["dev_mode_reset"].get("status") == "N/A_EXACT_RC2_NO_DEV_MODE" and oC["admission_16"]["status"] == "SKIPPED_SEQS_CAP"
      and oC["trace"]["status"] == "SKIPPED_CTRL_HAS_NO_PROFILER_FLAG" and not oC.get("early_stop"), [callsD, oC.get("admission_16")])
check("N2-4 DIAG bodies: the window port is armed for logprobs 5 only during the 8k@8 ladder (disarmed after); the boot "
      "record names the request and its sampling-unchanged basis",
      LP_AT_LADDER == {"SMOKE_SEED2": callsD_D[2][1], "SMOKE_CTRL": callsD[2][1]} and nsD["DIAG_LP"]["port"] is None
      and nsC["DIAG_LP"]["port"] is None and oD["diag_logprobs"]["requested"] == 5 and oC["diag_logprobs"]["requested"] == 5
      and "raw" in oD["diag_logprobs"]["sampling_unchanged_basis"] and ns["DIAG_LP"]["n"] == 0 and nsS["DIAG_LP"]["n"] == 0,
      [LP_AT_LADDER, oD.get("diag_logprobs")])

# 7c-3b: N2-4 request bodies + the LAB SSE parser's top-logprob ids (a real L.Stream against a 3-event SSE stub server)
import http.server
import threading as _th


class _SSE(http.server.BaseHTTPRequestHandler):
    BODIES = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        _SSE.BODIES.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        ev = [{"choices": [{"text": " a b", "token_ids": [11, 12], "finish_reason": None,
                            "logprobs": {"top_logprobs": [{"token_id:11": -0.1, "token_id:7": -1.6, "token_id:8": -2.0},
                                                          {"token_id:12": -0.3, "token_id:9": -0.4}]}}]},
              {"choices": [{"text": " c", "token_ids": [13], "finish_reason": "length",
                            "logprobs": {"top_logprobs": [{"token_id:13": -0.01, "token_id:5": -5.0}]}}]}]
        for d in ev:
            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


_srv = http.server.HTTPServer(("127.0.0.1", 0), _SSE)
_th.Thread(target=_srv.serve_forever, daemon=True).start()
_sp = _srv.server_address[1]
try:
    nsD["DIAG_LP"]["port"] = _sp
    s_win = L.Stream(_sp, [1, 2, 3], 3, 0.6, 4321)                   # = steady()'s call (temperature 0.6, seed)
    s_oth = L.Stream(_sp + 1, [1, 2, 3], 3, 0.6, 4321)               # another port: untouched
    s_gr = L.Stream(_sp, [1, 2, 3], 3, temperature=0.0, logprobs=2)  # greedy oracle: keeps its own logprobs 2, no tops
    nsD["DIAG_LP"]["port"] = None
    s_off = L.Stream(_sp, [1, 2, 3], 3, 0.6, 4321)                   # disarmed: as the LAB
    s_win.run()
    s_off.run()
finally:
    nsD["DIAG_LP"]["port"] = None
    _srv.shutdown()
rw = s_win.row()
check("N2-4 window request: logprobs 5 + return_tokens_as_token_ids added only on the armed port at temperature > 0 (sampling "
      "keys unchanged: temperature 0.6, top_p 0.95, top_k 20, seed, ignore_eos); the LAB parser keeps every token's top entries "
      "(ids sorted by logprob) next to its margin; other ports / the greedy oracle / a disarmed stream unchanged",
      s_win.body.get("logprobs") == 5 and s_win.body.get("return_tokens_as_token_ids") is True
      and {k: s_win.body[k] for k in ("temperature", "top_p", "top_k", "seed", "ignore_eos")}
      == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 4321, "ignore_eos": True}
      and "logprobs" not in s_oth.body and s_oth.tops is None and s_gr.body.get("logprobs") == 2 and s_gr.tops is None
      and "logprobs" not in s_off.body and s_off.tops is None and "top_logprobs" not in s_off.row()
      and rw["tokens"] == [11, 12, 13] and rw["margins"] == [1.5, 0.1, 4.99] and not rw["error"]
      and rw["top_logprobs"] == [[["token_id:11", -0.1], ["token_id:7", -1.6], ["token_id:8", -2.0]],
                                 [["token_id:12", -0.3], ["token_id:9", -0.4]], [["token_id:13", -0.01], ["token_id:5", -5.0]]]
      and _SSE.BODIES[0].get("logprobs") == 5 and "logprobs" not in _SSE.BODIES[1],
      [s_win.body.get("logprobs"), rw.get("margins"), rw.get("top_logprobs")])

# 7c-4: per-stream table from traces/requests.jsonl (the real function, the real tokenizer)
TOKD = nsD["TOK"]
loop_txt = " row?" * 1500
ok_txt = " ".join(_word(i) for i in range(2500))
rl = WORK / "req_diag.jsonl"
mx = nsD["MAX_NEW"]["8k"]
rows_ = []
for i in range(8):
    txt = loop_txt if i == 3 else ok_txt
    rows_.append({"port": 8012, "prompt_len": 8192, "prompt_sha": nsD["_psha"](nsD["PROMPTS"]["8k"][i]), "t": 1000.0 + i, "text": txt,
                  "tokens": TOKD.encode(txt, add_special_tokens=False).ids, "finish_reason": None, "error": None,
                  "sampling": {"max_tokens": mx, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 4321 + i, "ignore_eos": True}})
rows_.append(dict(rows_[3], sampling=dict(rows_[3]["sampling"], temperature=1.0, seed=4324), t=1500.0, text=ok_txt))   # profiled_load
rows_.append(dict(rows_[2], t=10.0, text=loop_txt))                                    # an older request (outside the window)
rows_.append(dict(rows_[1], port=8015, text=loop_txt))                                 # another port
rows_.append(dict(rows_[0], sampling=dict(rows_[0]["sampling"], seed=4400), text=loop_txt))   # the admission check's seeds
rl.write_text("".join(json.dumps(r) + NL for r in rows_))
_rl0, nsD["REQ_LOG"] = nsD["REQ_LOG"], rl
try:
    tbl = TABLE_D(8012, 8, 4321, 8192, mx, 900.0, 2000.0)
finally:
    nsD["REQ_LOG"] = _rl0
st_ = tbl["streams"]
fsha = {r["stream"]: r["prompt_sha"] for r in fs["rows"]}
check("per-stream table: 8 rows (stream = seed - 4321) from requests.jsonl; profiled_load (T 1.0), older, other-port and "
      "admission-seed rows ignored; stream 3 looped: first garbage 2,000-token window at 0 + tail verdict LOW_VARIETY",
      tbl["rows_found"] == 8 and [x["stream"] for x in st_] == list(range(8)) and [x["seed"] for x in st_] == list(range(4321, 4329))
      and tbl["loop_count"] == 1 and tbl["looped_streams"] == [3] and st_[3]["first_garbage_window"]["start_token"] == 0
      and st_[3]["first_garbage_window"]["verdict"] and st_[3]["tail_verdict"] == "LOW_VARIETY"
      and all(x["first_garbage_window"] is None and x["tail_verdict"] is None for j, x in enumerate(st_) if j != 3)
      and all(x["n_tokens"] == len(rows_[j]["tokens"]) for j, x in enumerate(st_))
      and all(x["prompt_sha_matches_failed_run"] == (x["prompt_sha"] == fsha[x["stream"]]) for x in st_)
      and tbl["failed_run_looped_streams"] == [0] and tbl["loop_count_windowed"] == 1 and tbl["loop_count_tail_gate"] == 1,
      [(x["stream"], x.get("first_garbage_window"), x.get("tail_verdict")) for x in st_])
nsD["REQ_LOG"] = rl
try:
    tb6 = TABLE_D(8012, 8, 4321, 8192, mx, 1000.0, 1005.5)          # rows 6 / 7 written after the window bound
finally:
    nsD["REQ_LOG"] = _rl0
check("per-stream table: a stream without a row is listed as missing (rows_found 6 of 8)",
      tb6["rows_found"] == 6 and [x.get("missing") for x in tb6["streams"]][6:] == [True, True], tb6["rows_found"])

# 7c-4b: N2-1 / N2-2 fault injection through the REAL diag_stream_table() + diag_verdict() (written requests.jsonl rows)
ok300 = " ".join(_word(i) for i in range(300))
N_PREFIX = len(TOKD.encode(ok300, add_special_tokens=False).ids)
TBLN = [0]


def diag_rows(base, loop_at=None, tweak=None, extra=(), logprobs=False):
    rows = []
    for i in range(8):
        txt = (ok300 + loop_txt) if i == loop_at else ok_txt
        toks = TOKD.encode(txt, add_special_tokens=False).ids
        r = {"port": 8012, "prompt_len": 8192, "prompt_sha": nsD["_psha"](nsD["PROMPTS"]["8k"][i]), "t": 1000.0 + i, "text": txt,
             "tokens": toks, "finish_reason": None, "error": None, "margins": [],
             "sampling": {"max_tokens": mx, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": base + i, "ignore_eos": True}}
        if logprobs:
            r["sampling"].update(logprobs=5, return_tokens_as_token_ids=True)
            r["margins"] = [0.5] * len(toks)
            r["top_logprobs"] = [[[f"token_id:{t}", -0.1], ["token_id:7", -0.6]] for t in toks]
        rows.append(r)
    if tweak:
        tweak(rows)
    return rows + [e(rows) for e in extra]


def diag_table(fn, nsx, rows, seed):
    TBLN[0] += 1
    p = WORK / f"req_diag_{TBLN[0]}.jsonl"
    p.write_text("".join(json.dumps(r) + NL for r in rows))
    old, nsx["REQ_LOG"] = nsx["REQ_LOG"], p
    try:
        return fn(8012, 8, seed, 8192, mx, 900.0, 2000.0)
    finally:
        nsx["REQ_LOG"] = old


def diag_rec(cfg, sec=90.0, tbl_=None, **kw):
    return dict({"status": "NO_GO_GARBAGE_UNDER_LOAD", "correctness": "NO_GO_GARBAGE_UNDER_LOAD", "config_readback": {"ok": True},
                 "cfg": cfg, "steady": {"8k@8": {"status": "FAIL_GARBAGE_UNDER_LOAD", "window_sec": sec}},
                 "diag_streams": tbl_ if tbl_ is not None else tbl, "admission_16": {"status": "ADMITTED_ALL"}}, **kw)


def vD_of(rows, seed=4321):
    t = diag_table(TABLE_D, nsD, rows, seed)
    return t, VERDICT_D(diag_rec(cD, tbl_=t))


def _set(i, **kv):
    return lambda rows: rows[i].update(kv)


_t1, v1 = vD_of(diag_rows(4321, tweak=_set(2, prompt_sha="0" * 16)))
check("N2-1: one stream with a wrong prompt sha -> DIAG_INCOMPLETE (prompt_identity, stream 2 named); the rest of the table is fine",
      v1["verdict"] == "DIAG_INCOMPLETE" and v1["failed"] == ["prompt_identity"]
      and v1["failed_detail"]["prompt_identity"]["streams_sha_not_matching"] == [2] and _t1["status"] == "TABLE",
      [v1["failed"], v1.get("failed_detail")])
_t2, v2 = vD_of(diag_rows(4321, tweak=_set(5, error="stream error: {\"message\": \"server_error\"}")))
check("N2-2: one stream with a request error -> DIAG_INCOMPLETE (stream_integrity, the error listed per stream)",
      v2["verdict"] == "DIAG_INCOMPLETE" and v2["failed"] == ["stream_integrity"]
      and list(v2["failed_detail"]["stream_integrity"]["errors"]) == [5] and v2["streams"][5]["error"].startswith("stream error"),
      [v2["failed"], v2.get("failed_detail")])
_t3, v3 = vD_of(diag_rows(4321, tweak=_set(4, tokens=[], text="")))
check("N2-2: one stream with an empty token list -> DIAG_INCOMPLETE (stream_integrity, bad_n_tokens [4], n_tokens 0)",
      v3["verdict"] == "DIAG_INCOMPLETE" and v3["failed"] == ["stream_integrity"]
      and v3["failed_detail"]["stream_integrity"]["bad_n_tokens"] == [4] and v3["streams"][4]["n_tokens"] == 0,
      [v3["failed"], v3.get("failed_detail")])
_t4, v4 = vD_of(diag_rows(4321, tweak=lambda rows: rows[7]["sampling"].update(seed=4400)))
_t4b, v4b = vD_of(diag_rows(1234), seed=1234)            # SEED2 profile, table / rows at the failed run's seeds 1234-1241
check("N2-2 seed_coverage: a stream row with a seed outside 4321-4328 is not matched (stream 7 missing) -> DIAG_INCOMPLETE; "
      "8 complete rows at seeds 1234-1241 in the SEED2 profile (WINDOW_SEED 4321) -> DIAG_INCOMPLETE (stream_integrity only)",
      v4["verdict"] == "DIAG_INCOMPLETE" and v4["failed"] == ["all_8_streams_recorded", "prompt_identity", "stream_integrity"]
      and v4["failed_detail"]["stream_integrity"]["missing"] == [7]
      and v4b["verdict"] == "DIAG_INCOMPLETE" and v4b["failed"] == ["stream_integrity"] and _t4b["rows_found"] == 8
      and v4b["failed_detail"]["stream_integrity"]["table_seed"] == 1234, [v4["failed"], v4b["failed"], v4b.get("failed_detail")])
_t5, v5 = vD_of(diag_rows(4321, extra=(lambda rows: dict(rows[2], t=1100.0, text=loop_txt),)))
check("N2-2: two matching rows for one seed are never overwritten -> table AMBIGUOUS_STREAM_ROWS, duplicate_streams [2], "
      "stream 2 duplicate_rows 2 -> DIAG_INCOMPLETE (trace_status, no_duplicate_streams)",
      _t5["status"] == "AMBIGUOUS_STREAM_ROWS" and _t5["duplicate_streams"] == [2] and _t5["matching_rows"] == 9
      and _t5["streams"][2]["duplicate_rows"] == 2 and v5["verdict"] == "DIAG_INCOMPLETE"
      and v5["failed"] == ["trace_status", "no_duplicate_streams"], [_t5["status"], _t5.get("duplicate_streams"), v5["failed"]])
_wrong = (lambda rows: dict(rows[5], sampling=dict(rows[5]["sampling"], top_k=40), text=loop_txt, t=1200.0),
          lambda rows: dict(rows[6], sampling=dict(rows[6]["sampling"], top_p=0.9), text=loop_txt, t=1200.0),
          lambda rows: dict(rows[1], sampling=dict(rows[1]["sampling"], ignore_eos=False), text=loop_txt, t=1200.0))
_t6, v6 = vD_of(diag_rows(4321, extra=_wrong))
_t6b, v6b = vD_of(diag_rows(4321, tweak=lambda rows: rows[5]["sampling"].update(top_k=40)))
check("N2-2 sampling match: looping rows with top_k 40 / top_p 0.9 / ignore_eos False at window seeds are ignored (no duplicate, "
      "0 loops, DIAG_DONE); the only stream-5 row at top_k 40 -> stream 5 missing -> DIAG_INCOMPLETE",
      _t6["status"] == "TABLE" and _t6["duplicate_streams"] == [] and _t6["matching_rows"] == 8 and _t6["loop_count"] == 0
      and v6["verdict"] == "DIAG_DONE" and _t6["match"]["top_k"] == 20 and _t6["match"]["top_p"] == 0.95
      and _t6["match"]["ignore_eos"] is True and _t6b["streams"][5].get("missing") is True and v6b["verdict"] == "DIAG_INCOMPLETE",
      [_t6["status"], _t6["loop_count"], v6["failed"], v6b["failed"]])
_t7, v7 = vD_of(diag_rows(4321, loop_at=3, logprobs=True))
lp7 = (_t7["streams"][3].get("loop") or {})
on7 = lp7.get("onset") or {}
_ao = lp7.get("around_onset") or []
_cons = lambda v: v["interpretation"]["rows_consistent_with_this_result"]
_rows_txt = [x["reading"] for x in v7["interpretation"]["rows"]]
check("N2-1/2/4/5/6: 8 valid streams with one genuine loop -> DIAG_DONE, loop_count 1, stream 3 MODEL_REPETITION_OBSERVED / "
      "numeric cause UNDETERMINED, periodic onset at the loop start, 64 positions of top-logprob ids (int) + margins around it; "
      "interpretation: SEED2 loop -> rows 0 (CTRL clean: B GROUP) / 3 (both loop), every row 'suggests', none 'proves'",
      v7["verdict"] == "DIAG_DONE" and v7["failed"] == [] and v7["loop_count"] == 1 and v7["looped_streams"] == [3]
      and all(v7["checks"].values()) and lp7.get("label") == "MODEL_REPETITION_OBSERVED" and lp7.get("numeric_cause") == "UNDETERMINED"
      and lp7.get("onset_source") == "periodic_tail" and abs(on7.get("onset_token", -99) - N_PREFIX) <= 2 and on7.get("period") in (1, 2, 4)
      and len(_ao) == 64 and all(isinstance(e["top"][0][0], int) and e["top"][0][0] == e["token"] and e["margin"] == 0.5 for e in _ao)
      and v7["streams"][3]["logprobs"]["aligned"] is True and all("loop" not in x for j, x in enumerate(v7["streams"]) if j != 3)
      and _cons(v7) == [0, 3] and all("suggests" in t_ for t_ in _rows_txt) and not any("prove" in t_ for t_ in _rows_txt)
      and "GROUP" in _rows_txt[0] and v7["interpretation"]["this_result"] == {"profile": "SMOKE_SEED2", "side": "seed2", "outcome": "loop"},
      [v7["failed"], {k: on7.get(k) for k in ("onset_token", "period")}, N_PREFIX, _cons(v7)])
_t8, v8 = vD_of(diag_rows(4321))
_t8c = diag_table(TABLE_C, nsC, diag_rows(1234), 1234)
v8c = VERDICT_C(diag_rec(cC, tbl_=_t8c))
check("8 valid streams with zero loops -> DIAG_DONE, loop_count 0 (SEED2 clean -> rows 1 / 2; CTRL clean -> rows 0 / 2); an "
      "INCOMPLETE result applies no interpretation row",
      v8["verdict"] == "DIAG_DONE" and v8["loop_count"] == 0 and v8["looped_streams"] == [] and _cons(v8) == [1, 2]
      and v8c["verdict"] == "DIAG_DONE" and v8c["loop_count"] == 0 and _cons(v8c) == [0, 2]
      and _cons(v1) == [] and v1["interpretation"]["this_result"]["outcome"] is None, [v8["failed"], v8c["failed"], _cons(v8c)])
_lo = nsD["loop_onset"]
check("loop_onset: [5, 6, 7] + [1, 2] x 40 -> onset 3, period 2; a strictly increasing list -> None; a short tail -> None",
      (_lo([5, 6, 7] + [1, 2] * 40) or {}).get("onset_token") == 3 and _lo([5, 6, 7] + [1, 2] * 40)["period"] == 2
      and _lo(list(range(200))) is None and _lo(list(range(100)) + [1, 2] * 20) is None)

# 7c-5: DIAG verdicts + main flows
tblC = diag_table(TABLE_C, nsC, diag_rows(1234, loop_at=0), 1234)      # the failed run's looped stream 0, CTRL profile
vD = VERDICT_D(diag_rec(cD))
vD2 = VERDICT_D(diag_rec(cD, sec=40.0))
vD3 = VERDICT_D(diag_rec(cD, tbl_=tb6))
check("diag verdict (SEED2): full window + 8 streams -> DIAG_DONE with loop count 1 / stream [3] / per-stream rows / seed 4321 "
      "(the boot keeps NO_GO_GARBAGE_UNDER_LOAD); a short window or a missing stream -> DIAG_INCOMPLETE naming it; the failed "
      "run's row 37 labelled MODEL_REPETITION_OBSERVED / NUMERIC_CAUSE_UNDETERMINED (ids 2713 / 30, no margins kept)",
      vD["verdict"] == "DIAG_DONE" and vD["loop_count"] == 1 and vD["looped_streams"] == [3] and len(vD["streams"]) == 8
      and vD["seed"] == 4321 and vD["sampling"]["seeds"] == [4321, 4328] and vD["status"][0] == "NO_GO_GARBAGE_UNDER_LOAD"
      and vD["failed_smoke"]["looped_streams"] == [0] and "ctrl_exact_rc2_diff_empty" not in vD["checks"]
      and [(x["row"], x["label"], x["numeric_cause"], x["margins_kept"]) for x in vD["failed_smoke"]["looped_rows"]]
      == [(37, "MODEL_REPETITION_OBSERVED", "NUMERIC_CAUSE_UNDETERMINED", 0)]
      and sorted(x[0] for x in vD["failed_smoke"]["looped_rows"][0]["last_500_token_ids"][:2]) == [30, 2713]
      and vD2["verdict"] == "DIAG_INCOMPLETE" and vD2["failed"] == ["window_ran_full"]
      and vD3["verdict"] == "DIAG_INCOMPLETE" and vD3["failed"] == ["all_8_streams_recorded", "prompt_identity", "stream_integrity"]
      and vD3["failed_detail"]["stream_integrity"]["missing"] == [6, 7], [vD["checks"], vD2["failed"], vD3["failed"]])
vC = VERDICT_C(diag_rec(cC, tbl_=tblC))
vC2 = VERDICT_C(diag_rec(dict(cC, exact_diff=dict(dCC, flags_added={"--max-num-seqs": "16"})), tbl_=tblC))
vC3 = VERDICT_C(diag_rec(cC))                      # the SEED2 table (seeds 4321-4328) handed to the CTRL verdict
check("diag verdict (CTRL): empty exact diff -> DIAG_DONE (seed 1234, CTRL loop -> rows 1 / 3); a non-empty diff -> "
      "DIAG_INCOMPLETE (ctrl_exact_rc2_diff_empty); a table at seeds 4321-4328 -> DIAG_INCOMPLETE (stream_integrity)",
      vC["verdict"] == "DIAG_DONE" and vC["checks"]["ctrl_exact_rc2_diff_empty"] is True and vC["seed"] == 1234
      and vC["looped_streams"] == [0] and _cons(vC) == [1, 3]
      and vC2["verdict"] == "DIAG_INCOMPLETE" and vC2["failed"] == ["ctrl_exact_rc2_diff_empty"]
      and vC3["verdict"] == "DIAG_INCOMPLETE" and vC3["failed"] == ["stream_integrity"], [vC["checks"], vC2["failed"], vC3["failed"]])
for nx, want_order, rec in ((nsD, [("B", 16)], diag_rec(cD)), (nsC, [("CTRL", None)], diag_rec(cC, tbl_=tblC))):
    nx["sha_gate"] = lambda t: (True, "ok")
    nx["merlin_install"] = lambda: {"fake": True}
    nx["scorecard"] = lambda: None
    nx["HARD_STOP"] = time.time() + 35 * 60
    order.clear()
    nx["R"]["sections"]["S2"] = {}
    nx["NB2_OPT"][:] = ["K0", "CTRL"]                  # ignored in the DIAG profiles
    nx["nb2_arm"] = fake_arm({rec["cfg"]["key"]: rec})
    nx["nb2_main"]()
    check(f"{nx['NB2_PROFILE']} main: one boot {want_order} (NB2_OPTIONAL ignored) -> R['smoke'] = DIAG_DONE with the loop count",
          order == want_order and nx["R"]["smoke"]["verdict"] == "DIAG_DONE" and nx["R"]["smoke"]["loop_count"] == 1
          and nx["R"]["smoke"]["profile"] == nx["NB2_PROFILE"], [list(order), nx["R"]["smoke"].get("checks")])

# 8. host RAM gate + page cache + boot gate in merlin_boot
ns3 = load("nb2c")
ns3["boot_gate"] = lambda: {"ok": True, "gpus": []}
started = []
ns3["merlin_fn_server"] = lambda *a, **k: started.append(a)
ns3["FAKE"] = None
ns3["host_mem"] = lambda: {"avail_mib": 60 * 1024, "cgroup_max_mib": 177 * 1024}
_hg = ns3["host_gate"]
ns3["host_gate"] = lambda wait_sec=0: _hg(0)
o = ns3["merlin_boot"]("x", "fp8", 8011, 60, None, cfg=cfgs["A"])
check("host gate below 69+4 GiB -> HOST_RAM_GATE, no server built", o["status"] == "HOST_RAM_GATE" and not started, o["status"])
ns3["host_mem"] = lambda: {"avail_mib": 160 * 1024}
g = ns3["boot_gate_nb2"]("y")
check("host gate above the floor -> ok; page-cache drop saw every checkpoint file",
      g["ok"] is True and g["host_gate"]["ok"] is True and g["page_cache"]["files_seen"] >= 20, g["page_cache"])
check("page-cache drop records platform support (posix_fadvise) and files actually advised",
      g["page_cache"]["supported"] == hasattr(os, "posix_fadvise") and (g["page_cache"]["files"] > 0) == hasattr(os, "posix_fadvise"))
ns3["host_mem"] = lambda: {}
check("host gate without numbers -> ok None (recorded, not blocking)", _hg(0)["ok"] is None)
os.environ["NB2_FAKE_HOST_FLOOR_MIB"] = "1024"
ns3["host_mem"] = lambda: {"avail_mib": 28 * 1024, "cgroup_max_mib": 30 * 1024}      # the Kaggle CPU kit's 30 GiB cgroup
started.clear()
o = ns3["merlin_boot"]("x2", "fp8", 8011, 60, None, cfg=cfgs["A"])
check("KIT-1: NB2_FAKE_HOST_FLOOR_MIB is IGNORED without FAKE (real floor 73 GiB kept, HOST_RAM_GATE)",
      ns3["host_floor"]() == ((69 + 4) * 1024, "Merlin 69 GiB + 4 GiB margin") and o["status"] == "HOST_RAM_GATE" and not started,
      [ns3["host_floor"](), o["status"]])
ns3["FAKE"] = "fake_server.py"
g = _hg(0)
check("KIT-1: FAKE mode + NB2_FAKE_HOST_FLOOR_MIB=1024 -> floor 1024 MiB, a 30 GiB cgroup passes the gate",
      ns3["host_floor"]()[0] == 1024 and g["ok"] is True and g["floor_mib"] == 1024 and "fake" in g["floor_source"], g)
os.environ.pop("NB2_FAKE_HOST_FLOOR_MIB")
check("KIT-1: FAKE without the env -> the real floor", ns3["host_floor"]()[0] == (69 + 4) * 1024)
ns3["FAKE"] = None

# 8b. KIT-2: nb2_e2e_check.py on a result where a boot never reached DONE (no config_readback): FAIL lines, no KeyError
ew = WORK / "e2e_keyerr"
(ew / "out" / "traces").mkdir(parents=True)
(ew / "out" / "nb2_results.json").write_text(json.dumps({"sources": {str(i): "x" for i in range(8)}, "sections": {"S2": {
    "A_k3_int8_s16": {"status": "HOST_RAM_GATE", "boot_gate": {"ok": False}, "cfg": {"key": "A", "k": 3, "seqs": 16, "exact_rc2": False}},
    "B_skipped": {"status": "SKIPPED_DEPENDENCY"}}}}))
(ew / "out" / "lever_scorecard.json").write_text(json.dumps({"rows": []}))
(ew / "out" / "traces" / "requests.jsonl").write_text(json.dumps({"text": "t", "tokens": [1]}) + chr(10))
(ew / "out" / "submission.json").write_text(json.dumps({str(i): [] for i in range(120)}))
import subprocess as _sp
er = _sp.run([sys.executable, str(HERE / "nb2_e2e_check.py"), str(INP), str(ew), "--check-only"], capture_output=True, text=True,
             env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=120)
check("KIT-2: e2e check on a boot that stopped at HOST_RAM_GATE -> FAIL line naming the boot, exit 1, no traceback / KeyError",
      er.returncode == 1 and "Traceback" not in er.stderr and "KeyError" not in er.stdout + er.stderr
      and any(l.startswith("FAIL A_k3_int8_s16") and "HOST_RAM_GATE" in l for l in er.stdout.splitlines()), (er.stdout[-600:], er.stderr[-600:]))

# 9. per-boot deadline watchdog
calls.clear()
ns["BOOT"].update(deadline=time.time() - 125, t0=time.time() - 200, why=None)
t = ns["_deadline_killer"](FakeS(), "A")
time.sleep(2.5)
check("boot deadline: killer fires, server killed, reason recorded", "kill" in calls and "BOOT_DEADLINE A" in str(ns["BOOT"]["why"]),
      ns["BOOT"]["why"])
ns["BOOT"].update(deadline=None, why=None)

# 10. decay / cohort verdicts, long_quality wrap
W = lambda a, v=True: {"accept_per_draft": a, "valid": v}
check("decay verdict: stable / decay / collapse / too few",
      ns["decay_verdict"]([W(2.1), W(2.0)])["status"] == "STABLE" and ns["decay_verdict"]([W(2.1), W(1.6)])["status"] == "DECAY_SUSPECT"
      and ns["decay_verdict"]([W(2.1), W(0.2)])["status"] == "COLLAPSE" and ns["decay_verdict"]([W(2.1), W(1.0, False)])["status"].startswith("INCONCLUSIVE"))
C = lambda pp, t: {"status": "MEASURED", "per_position": pp, "t_boot_min": t}
cv = ns["cohort_verdict"]([C([0.825, 0.64, 0.495], 5), C([0.768, 0.56, 0.394], 18)])
check("cohort verdict: RC2-like drop (82.5->76.8 / 64->56 / 49.5->39.4) -> UPTIME_DECAY_SUSPECT", cv["status"] == "UPTIME_DECAY_SUSPECT"
      and cv["minutes_apart"] == 13, cv)
check("cohort verdict: identical replay -> STABLE", ns["cohort_verdict"]([C([0.8, 0.6], 1), C([0.79, 0.6], 12)])["status"] == "STABLE")
lq = ns["long_quality"]
base = {"tokens": 5000, "error": None, "window_garbage": None, "cut_by_cap": False}
check("long_quality: natural needs finish stop + think closed",
      lq("natural", dict(base, finish="stop", think_closed=True)) == "PASS_LONG"
      and lq("natural", dict(base, finish="length", think_closed=True)) == "INCOMPLETE_LONG"
      and lq("natural", dict(base, finish="stop", think_closed=False)) == "INCOMPLETE_LONG"
      and lq("forced60k", dict(base, reached_60k=True)) == "PASS_LONG")
check("long_gate maps the wrapped INCOMPLETE to INCONCLUSIVE_LONG_INCOMPLETE",
      ns["long_gate"]({"long": {"natural": dict(base, finish="length", think_closed=True), "forced60k": dict(base, reached_60k=True)}})
      == "INCONCLUSIVE_LONG_INCOMPLETE")

# 11. trace analysis on synthetic per-rank traces with known numbers
td = WORK / "trace"
td.mkdir()


def mk(pp, tp, k_off, base_ns):
    ev, c = [], 1
    off = base_ns / 1000.0
    for i in range(10):
        t = i * 40000.0 - off + 1e9
        ev.append({"ph": "X", "cat": "user_annotation", "name": "execute_context_0(0)_generation_4(16)", "ts": t, "dur": 30000, "tid": 1})
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": t + 10, "dur": 2, "tid": 1, "args": {"correlation": c}})
        ev.append({"ph": "X", "cat": "kernel", "name": "void marlin_moe_wna16::Marlin<1>()", "ts": t + k_off, "dur": 20000, "args": {"correlation": c}})
        c += 1
        if pp == 1:
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": t + 20, "dur": 2, "tid": 1, "args": {"correlation": c}})
            ev.append({"ph": "X", "cat": "kernel", "name": "_recoverssm_commit_fold_kernel", "ts": t + k_off + 20000, "dur": 1000,
                       "args": {"correlation": c}})
            c += 1
    with gzip.open(td / f"dp0_pp{pp}_tp{tp}_dcp0_ep{tp}_rank{2 * pp + tp}.1.pt.trace.json.gz", "wt") as fh:
        json.dump({"baseTimeNanoseconds": base_ns, "traceEvents": ev}, fh)


for pp, tp, ko, bn in ((0, 0, 0, 5_000_000), (0, 1, 0, 7_000_000), (1, 0, 15000, 9_000_000), (1, 1, 15000, 3_000_000)):
    mk(pp, tp, ko, bn)
tr = ns["nb2_trace"](td, 3)
pp2 = tr.get("pp2") or {}
check("trace: PP2 overlap from clock-aligned per-rank traces (stage busy 20 ms + commit 1 ms per 40 ms tick, 5 ms overlap)",
      pp2.get("both_busy_pct") is not None and abs(pp2["both_busy_pct"] - 100 * 5 * 10 / pp2["window_us"] * 1000) < 1.0, pp2)
check("trace: verify rows per microbatch = 4 requests x 4 tokens", tr["verify_rows"]["gen_reqs_hist"] == {4: 10}
      and tr["verify_rows"]["tokens_per_req"] == [4.0], tr["verify_rows"])
check("trace: target FULL on stage 0, commit EAGER detected", tr["graph_modes"]["target_stage0"]["mode"] == "FULL"
      and tr["graph_modes"]["recoverssm_commit"] == "EAGER", tr["graph_modes"])
check("trace categories: Marlin MoE and the RecoverSSM commit are separate", ns["nb2_cat"]("void marlin_moe_wna16::Marlin<1>()") == "marlin_moe"
      and ns["nb2_cat"]("_recoverssm_commit_fold_kernel") == "recoverssm_commit" and ns["nb2_cat"]("_gdn_recoverssm_verify_kernel") == "gdn"
      and ns["nb2_cat"]("persistent_topk_kernel") == "persistent_topk" and ns["nb2_cat"]("qsa_decode") == "qsa")
check("trace: empty dir -> NO_TRACE_FILE", ns["nb2_trace"](WORK / "out", 3)["status"] == "NO_TRACE_FILE")

# 12. RC2 reference rows: sha check, install, fake rule
rr = ns["R"]["rc2_refs"]
check("RC2 refs: not installed on the fake server by default", rr["installed"] is False and "rc2_fp8" not in ns["ROWS"], rr.get("why"))
os.environ["NB2_FAKE_RC2_REFS"] = "1"
rr = ns["install_rc2_refs"]()
check("RC2 refs: prompt sha check decides (local ARC copy " + ("matches" if rr["prompt_sha_match"] else "differs") + ")",
      rr["installed"] == rr["prompt_sha_match"] and (("rc2_fp8" in ns["ROWS"]) == rr["prompt_sha_match"]), rr["got"])
if rr["installed"]:
    ns["ROWS"].pop("rc2_fp8")
ns["RC2_REFS"]["rows"][0]["prompt_sha"] = "0" * 16
rr = ns["install_rc2_refs"]()
check("RC2 refs: a sha mismatch -> not installed (boot A becomes the self-reference)", rr["installed"] is False and "rc2_fp8" not in ns["ROWS"])
os.environ.pop("NB2_FAKE_RC2_REFS")
real_rows = ns["RC2_REFS"]["rows"]
same = L.compare(real_rows, [dict(r) for r in real_rows])
flip = [dict(r) for r in real_rows]
flip[0] = dict(flip[0], tokens=flip[0]["tokens"][:10] + [flip[0]["tokens"][10] + 1] + flip[0]["tokens"][11:])
check("compare() on the real RC2 rows: identical -> EXACT; a flipped token decided by RC2's own margin",
      same["status"] == "EXACT" and L.compare(real_rows, flip)["rows"][0]["first_diff"] == 10, L.compare(real_rows, flip)["rows"][0])

# 13. (the MTP-off reference / teacher-forced control moved to NB0: lead decision 2026-10-31)
check("NB2 has no teacher-forced control (NB0 owns it)", "tf_control" not in ns and "TF_BANDS" not in ns)

# 14. scorecard (PLAN 4.1) from synthetic boot records, the control from the embedded RC2 numbers
ctrl8 = ns["RC2_CONTROL"]["windows"]["8k@8"]
f = ns["steady_formula"](ctrl8)
check("steady formula on RC2 8k@8: a = output tokens per cycle (12.62 x 2 / 8), formula = measured 261.8",
      abs(f["a_output_tokens_per_cycle"] - 3.155) < 0.01 and abs(f["formula_tok_s"] - 261.8) < 1.0 and f["steady_tok_s"] == 261.8
      and f["tick_ms"] == 48.2 and f["cycle_ms"] == 96.4, f)


cfit = ns["fit_step_model"](ns["RC2_CONTROL"]["windows"])
check("step model on the embedded RC2 control windows ~ the RC2 analysis fit 25.60 + 5.79 r",
      cfit["status"] == "FITTED" and abs(cfit["floor_ms"] - 25.6) < 2.0 and abs(cfit["per_request_ms"] - 5.79) < 0.8, cfit)
_w = lambda r, st: {"status": "STEADY", "running_mean": 2 * r, "tokens_per_step": 3.0 * r, "mean_acceptance_length": 3.0, "step_ms": st}
sfit = ns["fit_step_model"]({"a": _w(1, 31.39), "b": _w(4, 48.76), "c": _w(2, 37.18)})
check("step model: exact line 25.60 + 5.79 r recovered", sfit["floor_ms"] == 25.6 and sfit["per_request_ms"] == 5.79, sfit)
check("step model: one distinct r -> INCONCLUSIVE", ns["fit_step_model"]({"a": _w(4, 48)})["status"] == "INCONCLUSIVE_FIT")


def win(tok_s, tps, step, run=8, acc=2.2):
    return {"status": "STEADY", "tok_s": tok_s, "tokens_per_step": tps, "step_ms": step, "running_mean": run, "running_min": run,
            "running_max": run, "accept_per_draft": acc, "mean_acceptance_length": 1 + acc, "per_pos": {"per_position": [0.8, 0.7, 0.6]}}


S2 = ns["R"]["sections"].setdefault("S2", {})
S2.clear()
S2["A_k3_int8_s8"] = {"status": "DONE", "correctness": "CLEAN", "cfg": cfgs["A"], "sec": 3000, "f_census": {"F": 7.0},
                      "steady": {"8k@8": win(280.0, 13.5, 48.2)}, "log_facts": lf,
                      "decay": {"windows": [{"valid": True, "a_output_per_cycle": 3.1}, {"valid": True, "a_output_per_cycle": 2.72}]}}
S2["B_k2_int8_s16"] = {"status": "NO_GO_ORACLE", "correctness": "NO_GO_ORACLE", "cfg": cfgs["B"], "sec": 900, "steady": {}}
sc = ns["scorecard"]()
rows = {r["lever"].split()[0] + (" " + r["lever"].split()[2] if r["lever"].startswith("D1") else ""): r for r in sc["rows"]}
d2 = sc["rows"][0]
check("scorecard D2 row: matched 8k@8, gain = 0.9 x (arm - control) from measured generation rates",
      d2["lever"].startswith("D2") and d2["correctness"] == "PASS" and d2["projected_run_tok_s_gain"]["gain_tok_s"] == round(0.9 * (280.0 - 261.8), 1),
      d2["projected_run_tok_s_gain"]["gain_tok_s"])
check("scorecard: B NO_GO -> correctness FAIL; D3 row present; adoption / cost / NB0 diagnostic note on every row",
      any(r["lever"].startswith("D1 MTP K2") and r["correctness"] == "FAIL" for r in sc["rows"])
      and any(r["lever"].startswith("D3") for r in sc["rows"])
      and all(r["adoption"] in ("config", "patch", "research") and "DIAGNOSTIC until NB0" in r["notes"] for r in sc["rows"]))
lt = d2["measured"]["long_trace"]
check("scorecard long-trace label: planning a 2.72, a series, projected long tok/s = steady x a_long / a_early",
      lt["planning_a_k3"] == 2.72 and lt["a_series"] == [3.1, 2.72] and lt["projected_long_tok_s"] == round(280.0 * 2.72 / (13.5 * 2 / 8), 1), lt)
check("source hashes embedded (builder, RC2 cells, lab + the NB2 DIAG top-logprob patch of it, control files, Merlin launch.py, "
      "failed SMOKE trace)", len(ns["R"]["sources"]) == 10 and all(len(v) == 64 for v in ns["R"]["sources"].values())
      and ns["R"]["sources"]["e23_lab.py (embedded LAB, LF)"] != ns["R"]["sources"]["e23_lab.py + NB2 DIAG top-logprob patch (embedded, LF)"]
      and "E25 NB2 DIAG: top-logprob ids per token" in ns["LAB_SRC"], list(ns["R"]["sources"]))
sc_file = json.loads((WORK / "out" / "lever_scorecard.json").read_text())
check("scorecard file written with the PLAN 4.1 fields", all(set(r) >= {"lever", "measured", "projected_run_tok_s_gain", "correctness",
                                                                        "confidence", "adoption", "cost_gpu_h", "notes"} for r in sc_file["rows"]))
S2.clear()

# 15. notebook size / cells / builder guards (the new DIAG notebooks = the current code; FULL / SMOKE are pushed, kept)
EXPORT_TXT = (HERE / "nb2_code_export.py").read_text(encoding="utf-8")
for prof in ("SMOKE_SEED2", "SMOKE_CTRL"):
    nbp = HERE.parent / f"Q27-XHIGH-E25_NB2_DECODE_{prof}.ipynb"
    data = nbp.read_bytes()
    codes = ["".join(c["source"]) for c in json.loads(data)["cells"] if c["cell_type"] == "code"]
    ok_compile = True
    for c in codes:
        try:
            compile(c, "<cell>", "exec")
        except SyntaxError:
            ok_compile = False
    check(f"{prof} notebook: < 400 kB, 9 code cells compile, LF only, code = nb2_code_export.py (profile FULL) except "
          f"NB2_DEFAULT_PROFILE = {prof}", len(data) < 400_000 and len(codes) == 9 and ok_compile and b"\r\n" not in data
          and EXPORT_TXT.count('NB2_DEFAULT_PROFILE = "FULL"') == 1
          and "\n\n".join(codes) == EXPORT_TXT.replace('NB2_DEFAULT_PROFILE = "FULL"', f'NB2_DEFAULT_PROFILE = "{prof}"'), len(data))
btxt = (HERE / "build_e25_nb2.py").read_text(encoding="utf-8")
PIN = {"Q27-XHIGH-E25_NB2_DECODE.ipynb": "e20350a952ad171441774fcfcb30ffd5b1093ea013ecceb5793c717b19ec845b",
       "Q27-XHIGH-E25_NB2_DECODE_SMOKE.ipynb": "ad8a28a5864aee83a1476ea591bce5a22aa50d9231d9195cb1f3fe8e95c87386"}
check("builder: FULL e20350a9 + SMOKE ad8a28a5 in PUSHED, never rebuilt (PUSHED_PROFILES), other pushed files -> _REBUILD",
      all(v in btxt for v in PIN.values()) and 'PUSHED_PROFILES = ("FULL", "SMOKE")' in btxt and "_REBUILD.ipynb" in btxt
      and "assert _after == PUSHED_SHA_BEFORE" in btxt)
import hashlib as _hl
_disk = {n: _hl.sha256((HERE.parent / n).read_bytes()).hexdigest() for n in PIN if (HERE.parent / n).exists()}
check("pushed FULL / SMOKE notebooks byte-identical to the run / audited files (where present; the CPU kit carries neither)",
      all(_disk[n] == PIN[n] for n in _disk), {n: v[:8] for n, v in _disk.items()})

n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
(WORK / "unit_results.json").write_text(json.dumps([{"test": n, "ok": ok, "detail": d} for n, ok, d in RESULTS], indent=1))
sys.exit(1 if n_fail else 0)
