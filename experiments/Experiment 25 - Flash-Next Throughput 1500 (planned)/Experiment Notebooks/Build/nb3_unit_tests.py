"""E25 NB3 local tests: pure-Python parts (T sets, D(T), message sizes, routing, projections, ranking, scorecard), the
CPU torch reference math, the once()-patched P5 bench planning, the fake benches, fault injection on the orchestration
(exported notebook code with NB3_UNIT=1), one full end-to-end FAKE run of the notebook code (E23_FAKE_SERVER), the
regressions of the three GPT audits (N3-1 .. N3-25, section 10) and the integration fault-injection run (N3-25, section 11).

Usage: python nb3_unit_tests.py <work dir>      (run build_e25_nb3.py first: it writes nb3_export/ + nb3_code_export.py)
Env overrides (CPU kit on Linux): NB3_TEST_P1 (real part-1 small files), NB3_TEST_PARTS, NB3_TEST_EVAL.
"""
import hashlib
import importlib.util
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE / "nb3_export"
WORK = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "_nb3_test_work")
P1 = Path(os.environ.get("NB3_TEST_P1") or r"D:/ARC AGI 2/_scratch/fn_ckpt_test/p1/out")            # real part-1 small files (index, config, tokenizer)
PARTS = json.loads(Path(os.environ.get("NB3_TEST_PARTS") or r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)/prep/merlin_fn_ckpt/kernels/parts.json").read_text())
EVAL = Path(os.environ.get("NB3_TEST_EVAL") or r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_challenges.json")
RESULTS = []
BASE_ENV = dict(os.environ)          # env_for() starts from this (os.environ is cleared between phases)
for _k in ("NB3_FAKE_FAULTS", "NB3_CPU_KIT"):
    BASE_ENV.pop(_k, None)
ST_OK = {"D5": "DONE", "D5_rest": "DONE", "D7": "DONE", "D4": "DONE"}


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def jcopy(x):
    return json.loads(json.dumps(x))


def run_moe_fake(out, extra=(), faults=""):
    env = {**BASE_ENV, "NB3_FAKE_FAULTS": faults}
    r = subprocess.run([sys.executable, str(EXP / "nb3_moe_bench.py"), "--fake", "--ckpt", str(WORK), "--out", str(out)] + list(extra),
                       capture_output=True, text=True, timeout=300, env=env)
    return r, (json.loads(Path(out).read_text()) if Path(out).is_file() else {})


shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
sys.path.insert(0, str(EXP))
C = load("nb3_common", EXP / "nb3_common.py")

# ------------------------------------------------------------------ 1. grid, D(T), sizes, call counts
ts = C.t_sets()
check("full T = seqs x (1+K)", ts["full"] == sorted({s * (1 + k) for s in C.SEQS for k in C.KS}), str(ts["full"]))
check("micro T at RC2 point = 16 (8 streams, K3, PP2)", C.micro_T(8, 3) == 16)
check("micro + draft counts in the union", {16, 4, 32, 64, 1, 9, 18}.issubset(set(ts["all"])), str(ts["all"]))
check("T union has 21 values", len(ts["all"]) == 21, str(len(ts["all"])))
d = {t: round(C.d_expected(t)) for t in (16, 32, 64)}
check("D(16)=139 D(32)=240 D(64)=367 (PLAN 1)", d == {16: 139, 32: 240, 64: 367}, str(d))
eb = C.expert_bytes()
check("one INT4 g128 expert ~2.53 MB", abs(eb - 2534400) < 1, str(eb))
check("AR payload T=16 = 81,920 B", C.ar_bytes(2560, 16) == 81920)
s1 = C.stage_layers(1)
check("stage 1 = 23 layers, 6 QSA, 17 GDN", s1 == {"layers": 23, "qsa": 6, "gdn": 17}, str(s1))
s0 = C.stage_layers(0)
check("stage 0 = 25 layers, 6 QSA", s0["layers"] == 25 and s0["qsa"] == 6, str(s0))
tc = C.tick_calls(8, 3)
check("tick calls RC2: 23 MoE @16 + 3 draft @4; 46 + 6 AR", (tc["moe_target"], tc["moe_draft"], tc["ar_target"], tc["ar_draft"], tc["r"]) == (23, 3, 46, 6, 4), str(tc))

# ------------------------------------------------------------------ 2. routing generators (synthetic)
ids = C.routing_ids(64, "independent", seed=1)
check("independent: 10 distinct experts per token", all(len(set(r)) == 10 for r in ids))
rq = C.routing_ids(16, "request", seed=1, group=4)
check("request: the 4 tokens of a request share their experts", all(rq[i] == rq[i - i % 4] for i in range(16)))
zp = C.routing_ids(64, "zipf", seed=1)
check("zipf: 10 distinct experts per token", all(len(set(r)) == 10 for r in zp))
du = sum(C.distinct(C.routing_ids(64, "independent", seed=s)) for s in range(5)) / 5
dz = sum(C.distinct(C.routing_ids(64, "zipf", seed=s)) for s in range(5)) / 5
check("zipf touches fewer distinct experts than uniform at T=64", dz < du, f"zipf {dz} uniform {du}")
check("uniform distinct close to D(64)=367", abs(du - C.d_expected(64)) < 25, str(du))
check("routing deterministic per seed", C.routing_ids(8, "zipf", seed=3) == C.routing_ids(8, "zipf", seed=3))
w = C.routing_weights(5, seed=2)
check("router weights positive and normalised", all(abs(sum(r) - 1) < 1e-9 and min(r) > 0 for r in w))
check("routing note says SYNTHETIC + NB1", "SYNTHETIC" in C.ROUTING_NOTE and "NB1" in C.ROUTING_NOTE)
for loc in ((0, 256), (256, 512)):
    adv = C.routing_ids(16, "adversarial", seed=1, local=loc)
    check(f"N3-3 adversarial routing {loc}: exactly 1 local expert per token from a set of 2, 10 distinct, odd rows repeat",
          all(len(set(r)) == 10 and sum(loc[0] <= e < loc[1] for e in r) == 1 and {e for e in r if loc[0] <= e < loc[1]} <= {loc[0], loc[0] + 1}
              for r in adv) and all(adv[i] == adv[i - 1] for i in range(1, 16, 2)), str(adv[:2]))

# ------------------------------------------------------------------ 3. P4 helpers cut in; CPU torch reference math
check("P4 stats/bench/metrics present in nb3_common", all(hasattr(C, f) for f in ("stats", "bench", "metrics")))
check("P4 stats median", C.stats([3, 1, 2])["median"] == 2)
try:
    import torch
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-8, 8, (64, 512), generator=g)
    sc = torch.rand(64, 4, generator=g) + 0.1
    pk = C.ct_pack_int4(q)
    check("ct pack: [N, K/8] int32", pk.shape == (64, 64) and pk.dtype == torch.int32)
    check("ct dequant(pack(q)) == q x scale", torch.equal(C.ct_dequant_int4(pk, sc, 128), q.float() * sc.repeat_interleave(128, 1)))
    # GPTQ layout equivalence (Merlin's _unpack_and_dequant_int4_gptq logic, transposed CT tensor), re-implemented here
    w_int32 = pk.T.contiguous().unsqueeze(0)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    nib = (w_int32.unsqueeze(-1) >> shifts) & 0xF
    Kp, N = w_int32.shape[1], w_int32.shape[2]
    wg = nib.permute(0, 1, 3, 2).reshape(1, Kp * 8, N).to(torch.int16) - 8
    gptq = (wg.float() * sc.T.unsqueeze(0).repeat_interleave(128, dim=1)).permute(0, 2, 1)[0]
    check("CT transpose == GPTQ unpack (Merlin int_wna16.py:1282-1339 math)", torch.equal(gptq, C.ct_dequant_int4(pk, sc, 128)))
    W = {e: (torch.randn(8, 32, generator=g), torch.randn(8, 32, generator=g), torch.randn(32, 8, generator=g)) for e in range(6)}
    x = torch.randn(4, 32, generator=g)
    tid = torch.tensor([[0, 3], [3, 5], [1, 4], [5, 2]])
    tw = torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    ref = C.moe_reference(x, tid, tw, lambda e: W[e], 0, 3)
    man = torch.zeros(4, 32)
    for t in range(4):
        for j in range(2):
            e = int(tid[t, j])
            if e < 3:
                man[t] += tw[t, j] * ((torch.nn.functional.silu(x[t] @ W[e][0].T) * (x[t] @ W[e][1].T)) @ W[e][2].T)
    check("moe_reference = per-token loop over LOCAL experts only", torch.allclose(ref, man, atol=1e-5))
    m = C.metrics(torch, ref.to(torch.bfloat16), ref)
    check("P4 metrics on a bf16 rounding: rel_fro < 1e-2, cos > 0.999", m["rel_fro"] < 1e-2 and m["cos"] > 0.999, str(m))
    mx = C.metrics_x(torch, ref.to(torch.bfloat16), ref)
    check("N3-3 metrics_x adds p99_abs + ref_max (finite, max_abs kept)", {"p99_abs", "ref_max", "finite", "max_abs", "max_abs_over_ref_max"} <= set(mx)
          and mx["p99_abs"] <= mx["max_abs"], str(mx))
except ImportError:
    check("torch available for the CPU reference tests", False, "pip torch")

# ------------------------------------------------------------------ 4. projection, candidates, ranking, scorecard
pj = C.projection(3.9)
mid = pj["by_step_ms"]["39.0"]
check("projection: 3.9 ms of a 39 ms tick = 10 % share", abs(mid["share_of_tick"] - 0.1) < 1e-9)
check("projection: steady 261.8 x 39/35.1", abs(mid["steady_tok_s"] - round(261.8 * 39 / 35.1, 1)) < 0.05, str(mid))
check("projection: run gain = 0.9 x steady gain", abs(mid["run_tok_s_gain"] - round(0.9 * (261.8 * 39 / 35.1 - 261.8), 2)) < 0.02)
check("projection labelled MICROBENCH_PROJECTED", "MICROBENCH_PROJECTED" in pj["label"])
check("projection: saved >= tick -> None (not a number)", C.projection(31.0)["by_step_ms"]["30.0"] is None)
fm, fg = C.fake_moe(), C.fake_gemm()
SIZES = [C.ar_bytes(2560, t) for t in C.t_sets()["all"]]
fa = {"0,1": C.fake_ar(SIZES, 1), "2,3": C.fake_ar(SIZES, 2)}
tables = {"d5": C.d5_table(fm), "d7": C.d7_table(fa, 2560, 2), "d4": C.d4_table(fg)}
check("d5 table: 21 T x 4 routing modes (uniform, zipf, hot, request)", len(tables["d5"]["baseline"]) == 84, str(len(tables["d5"]["baseline"])))
check("d5 table: adversarial reference rows at T16 / T4", sorted(x["T"] for x in tables["d5"]["adversarial"]) == [4, 16])
check("d5 table: hot rows have <= 16 distinct local experts at every T",
      all(x["distinct_local"] <= C.HOT_SET for x in tables["d5"]["baseline"] if x["mode"] == "hot"))
check("d5 table: split rows for C 8/16/24/32 with time, bytes, equivalence",
      [x["C"] for x in tables["d5"]["split"]] == [8, 16, 24, 32] and all(x["split_us"] and x["combined_us"] and x["bytes_split"]
                                                                         and x["equivalent"] is True for x in tables["d5"]["split"]))
check("d5 table: split bytes labelled DERIVED without ncu, ncu UNVERIFIED, no dram_gbps",
      all(x["bytes_source"].startswith("DERIVED") and x["ncu_status"] == C.NCU_UNVERIFIED and x["dram_gbps_split"] is None
          for x in tables["d5"]["split"]))
check("d5 table: replay 'real routing unavailable' recorded when no file", tables["d5"]["replay"]["available"] is False and
      "unavailable" in tables["d5"]["replay"]["why"])
check("d7 table rows at every size, worse pair taken", len(tables["d7"]["rows"]) == 21 and all(r["nccl_graph"] for r in tables["d7"]["rows"]))
check("N3-5 d7: no LL128 column, LL128 labelled UNTESTED (not run, NCCL docs corruption risk)",
      all("ll128_graph" not in r for r in tables["d7"]["rows"]) and tables["d7"]["ll128"] == C.LL128_LABEL
      and C.LL128_LABEL.startswith("UNTESTED (not run, NCCL docs corruption risk)"))
check("d7 numeric: p5 PASS, nccl PASS on both pairs x both ranks", all(v["p5"] == "PASS" and v["nccl"] == "PASS" for v in tables["d7"]["numeric"].values())
      and tables["d7"]["arms"] == {"nccl": "PASS", "p5": "PASS", "p5_int8": "PASS"}, json.dumps(tables["d7"]["arms"]))
check("d7 concurrency: per-size windows of both pairs overlap -> VERIFIED", tables["d7"]["concurrency"]["status"] == "VERIFIED",
      json.dumps({k: v for k, v in tables["d7"]["concurrency"].items() if k != "per_size_arm"}))
rk = C.rank_candidates(tables, status=ST_OK)
t0 = rk["targets"][0]
check("ranked at the RC2 point (T_micro 16)", t0["T_micro"] == 16 and len(t0["ranked"]) >= 3, str([c["id"] for c in t0["ranked"]]))
check("ranked: D5 cfg + D7 P5 + D4 plan winners, all PROMOTION_READY", {"D5_marlin_moe_cfg", "D7_p5_host_allreduce"} <= {c["id"] for c in t0["ranked"]}
      and any(c["id"].startswith("D4_") for c in t0["ranked"]) and all(c["eligibility"] == C.PROMOTION_READY for c in t0["ranked"]))
check("ranking sorted by ms saved per tick", all(a["saved_ms_per_tick"] >= b["saved_ms_per_tick"] for a, b in zip(t0["ranked"], t0["ranked"][1:])))
check("no LL128 candidate anywhere", not any("ll128" in c["id"] for tg in rk["targets"] for c in tg["ranked"] + tg["blocked"] + tg["diagnostic"]))
check("draft-head candidates never ranked (UNTESTED acceptance)", all(not c["id"].startswith("D2_") for c in t0["ranked"]) and
      any(c["id"].startswith("D2_") for c in t0["blocked"]))
check("every candidate has a family", all(c.get("family") for tg in rk["targets"] for c in tg["ranked"] + tg["blocked"] + tg["info"] + tg["diagnostic"]))
check("D5 MoE term info row (ms per tick)", any(c["id"] == "D5_moe_term_info" for c in t0["info"]))
check("16- / 32-stream targets: T_micro 32 / 64", [tg["T_micro"] for tg in rk["targets"][1:]] == [32, 64])
check("16- / 32-stream D5 candidates are gated by the D5 rest bench", all(c["bench"] == "D5_rest" for tg in rk["targets"][1:]
                                                                          for c in tg["ranked"] + tg["blocked"] + tg["diagnostic"] if c["lever"] == "D5"))
# fault injection: a failing reference blocks the D5 candidate; a P5 check failure blocks D7; score 0 unless PASS
bad = jcopy(fm)
bad["sweep"]["16"]["best"]["ref_ok"] = False
cb = C.candidates({**tables, "d5": C.d5_table(bad)})
check("D5 best rule with ref FAIL -> correctness FAIL", next(c for c in cb if c["id"] == "D5_marlin_moe_cfg")["correctness"] == "FAIL")
bad = jcopy(fm)
bad["sweep"]["16"]["best"]["equivalent_vs_base"] = False
cb = C.candidates({**tables, "d5": C.d5_table(bad)})
check("N3-3 D5 best rule not ULP-equivalent to the baseline -> FAIL", next(c for c in cb if c["id"] == "D5_marlin_moe_cfg")["correctness"] == "FAIL")
fa_bad = jcopy(fa)
fa_bad["2,3"]["groups"]["main_ab"]["rank0"]["checks"]["a_bitexact_vs_ordered_fp32_ref"] = False
tb = C.d7_table(fa_bad, 2560, 2)
cb = C.candidates({**tables, "d7": tb})
check("P5 bit-exact failure on one pair -> P5 candidate FAIL", next(c for c in cb if c["id"] == "D7_p5_host_allreduce")["correctness"] == "FAIL")
c8 = next(c for c in C.candidates(tables) if c["id"] == "D7_p5_int8_allreduce")
check("N3-11 INT8 P5 is PASS_BOUND_ONLY, eligibility DIAGNOSTIC_ONLY (never ranked)", c8["correctness"] == "PASS_BOUND_ONLY"
      and C.eligibility(c8, ST_OK) == C.DIAGNOSTIC_ONLY and not any(c["id"] == "D7_p5_int8_allreduce" for c in t0["ranked"])
      and any(c["id"] == "D7_p5_int8_allreduce" for c in t0["diagnostic"]))
rb = C.rank_candidates({**tables, "d7": tb}, status=ST_OK)
check("a FAIL candidate is listed as blocked, not ranked", any(c["id"] == "D7_p5_host_allreduce" for c in rb["targets"][0]["blocked"]) and
      not any(c["id"] == "D7_p5_host_allreduce" for c in rb["targets"][0]["ranked"]))
sc = C.scorecard(tables, ST_OK)
need = {"lever", "measured", "projected_run_tok_s_gain", "correctness", "confidence", "adoption", "cost_gpu_h", "notes", "score",
        "promotion", "bench_status", "microbench_saved_ms", "engine_confirmed_saved_ms", "additivity"}
check("scorecard: rows D5 / D7 / D4 / D2 / PP2 / C6-humming with every field", [r["lever"] for r in sc["rows"]] == ["D5", "D7", "D4", "D2", "PP2", "C6-humming"] and
      all(need <= set(r) for r in sc["rows"]), str([sorted(set(need) - set(r)) for r in sc["rows"]]))
check("scorecard: adoption classes (D5 config, D7 patch, D4 patch, D2 config)",
      [r["adoption"] for r in sc["rows"]] == ["config", "patch", "patch", "config", "research", "research"], str([r["adoption"] for r in sc["rows"]]))
check("scorecard: D5 / D7 / D4 PROMOTION_READY on all-valid data", [r["promotion"] for r in sc["rows"][:3]] == [C.PROMOTION_READY] * 3,
      str([r["promotion"] for r in sc["rows"]]))
r_pp2 = next(r for r in sc["rows"] if r["lever"] == "PP2")
check("scorecard PP2 split-read row: hypothesis, research, DIAGNOSTIC_ONLY, score 0, measured split vs combined by C",
      r_pp2["hypothesis"] is True and set(r_pp2["measured"]["split_vs_combined_by_C"]) == {"8", "16", "24", "32"}
      and r_pp2["correctness"] in ("PASS", "UNTESTED") and r_pp2["promotion"] == C.DIAGNOSTIC_ONLY and r_pp2["score"] == 0,
      json.dumps({k: r_pp2[k] for k in ("correctness", "promotion", "confidence", "adoption", "score")}))
check("PP2 split-read is a hypothesis: never ranked / blocked, listed under hypotheses",
      not any(c["id"] == "PP2_split_read" for c in t0["ranked"] + t0["blocked"]) and any(c["id"] == "PP2_split_read" for c in t0["hypotheses"]))
fm_bad = jcopy(fm)
fm_bad["split"]["8"]["equivalence"]["equivalent"] = False
r_bad = next(r for r in C.scorecard({**tables, "d5": C.d5_table(fm_bad)}, ST_OK)["rows"] if r["lever"] == "PP2")
check("PP2 with split != combined -> FAIL, score 0", r_bad["correctness"] == "FAIL" and r_bad["score"] == 0)
check("scorecard D5 row carries hot routing + replay status + adversarial + EP1 spot", {"hot_routing_by_T", "real_routing_replay",
                                                                                       "adversarial_reference", "ep1_spot"} <= set(sc["rows"][0]["measured"]))
check("scorecard: projection inputs shown + microbench label", all("inputs" in r["projected_run_tok_s_gain"] and "MICROBENCH" in r["projected_run_tok_s_gain"]["label"] for r in sc["rows"]))
check("scorecard: D2 UNTESTED -> score 0", next(r for r in sc["rows"] if r["lever"] == "D2")["score"] == 0)
check("scorecard: D5 notes carry the synthetic-routing note", "SYNTHETIC" in next(r for r in sc["rows"] if r["lever"] == "D5")["notes"])
empty = C.scorecard({}, {})
check("scorecard with no bench data: 6 rows UNTESTED, score 0, gain 0", len(empty["rows"]) == 6 and
      all(r["correctness"] == "UNTESTED" and r["score"] == 0 and (r["projected_run_tok_s_gain"]["run_tok_s_gain_headline"] or 0) == 0
          for r in empty["rows"]))
sc_bad = C.scorecard({**tables, "d7": tb}, ST_OK)
r7 = next(r for r in sc_bad["rows"] if r["lever"] == "D7")
check("scorecard D7 with P5 FAIL: never PASS on the failing arm", r7["best_candidate"] is None or r7["best_candidate"]["id"] != "D7_p5_host_allreduce")
for r in sc["rows"]:
    if r["promotion"] == C.PROMOTION_READY:
        g = r["projected_run_tok_s_gain"]["run_tok_s_gain_headline"] or 0
        want = round(g * C.CONF_WEIGHT[r["confidence"]] / C.ADOPTION_COST[r["adoption"]], 3)
        check(f"scorecard score formula {r['lever']}", abs(r["score"] - want) < 1e-6 and r["correctness"] == "PASS", f"{r['score']} vs {want}")

st8 = C.rc2_step_ms(8)
check("RC2 D-model step at 8 streams = 24.21 + 0.1758 D(16) ~ 48.6 ms (measured 48.2)", abs(st8 - (24.21 + 0.1758 * C.d_expected(16))) < 1e-9 and abs(st8 - 48.2) < 1.0, str(st8))
check("RC2 model steady at 8 streams ~ 251 tok/s (RC2 6.2)", abs(C.steady_from_step(8, st8) - 251) < 2, str(C.steady_from_step(8, st8)))
hp = C.projection(4.0, 8)["headline"]
check("headline projection uses the fitted step", hp["step_ms_fit"] == round(st8, 2) and hp["run_tok_s_gain"] > 0, str(hp))
check("headline at 32 streams uses its own fitted step", C.projection(4.0, 32)["headline"]["step_ms_fit"] == round(C.rc2_step_ms(32), 2))
fd = C.floor_decomposition(tables)
check("floor decomposition: dense + heads + all-reduce timed, remainder vs the 24.21 ms floor",
      fd["floor_timed_ms"] and abs(fd["floor_remainder_ms"] - (24.21 - fd["floor_timed_ms"])) < 1e-6 and fd["allreduce_nccl_ms"] and fd["draft_heads_bf16_ms"], json.dumps({k: fd[k] for k in ("dense_gemm_target_head_ms", "draft_heads_bf16_ms", "allreduce_nccl_ms", "floor_timed_ms", "floor_remainder_ms")}))
check("floor decomposition: MoE measured vs fitted D term + derived GB/s vs 173", fd["moe_measured_ms"] and fd["moe_vs_fitted_term"] and fd["moe_derived_gbps_vs_rc2_implied_173"], json.dumps({k: fd[k] for k in ("moe_measured_ms", "fitted_moe_term_ms", "moe_vs_fitted_term", "moe_derived_gbps_vs_rc2_implied_173")}))
check("N3-4 d5 rows carry derived_gbps vs 173 and vs 300, no bare 'gbps' column", all("derived_gbps_vs_rc2_implied_173" in x and "derived_gbps_vs_l4_peak_300" in x
                                                                                       and "gbps" not in x for x in tables["d5"]["baseline"])
      and all(not any(k == "gbps" or k.startswith("gbps_") for k in x) for x in tables["d5"]["split"]))
check("scorecard carries the RC2 cost model + floor decomposition", "rc2_cost_model" in sc and sc["floor_decomposition_rc2_point"]["T_micro"] == 16)
check("ranked candidates carry floor decompositions per target", set(rk["floor_decomposition"]) == {"8", "16", "32"})

# ------------------------------------------------------------------ 4b. extreme-hot routing, PP2 split / merge, equivalence, ncu, replay
hs = C.hot_set(seed=1)
for T in (4, 32, 128):
    h = C.routing_ids(T, "hot", seed=1)
    check(f"hot routing T={T}: 10 distinct per token, all from the fixed 16-set", all(len(set(r)) == 10 and set(r) <= set(hs) for r in h)
          and C.distinct(h) <= 16, str(C.distinct(h)))
check("hot set is the same for every T of a seed", set(e for r in C.routing_ids(128, "hot", seed=1) for e in r) <= set(hs)
      and C.hot_set(seed=1) == hs and len(hs) == 16)
check("hot routing touches far fewer experts than uniform at T=64", C.distinct(C.routing_ids(64, "hot", seed=1)) <= 16 < C.distinct(C.routing_ids(64, "independent", seed=1)))
cases = C.pp2_split_cases()
check("PP2 split cases: C 8/16/24/32 at K3 -> T 32/64/96/128, halves 16/32/48/64",
      [(c["C"], c["T"], c["T_half"]) for c in cases] == [(8, 32, 16), (16, 64, 32), (24, 96, 48), (32, 128, 64)])
check("split halves of 32 = [0,16) [16,32)", C.split_halves(32) == [(0, 16), (16, 32)])
check("half T = the RC2 micro-batch tokens (C/2)(1+K)", all(c["T_half"] == C.micro_T(c["C"], 3) for c in cases))
try:
    import torch
    g = torch.Generator().manual_seed(5)
    x = torch.randn(32, 8, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 512, (32, 10), generator=g, dtype=torch.int32)
    w = torch.rand(32, 10, generator=g)
    rowwise = lambda xx, ii, ww: (xx.float() * (ww.sum(1, keepdim=True) + ii[:, :1].float() / 512)).to(torch.bfloat16)  # noqa: E731
    comb = rowwise(x, ids, w)
    sm = C.split_merge(rowwise, x, ids, w, 32)
    eq = C.equivalence(sm, comb)
    check("split_merge of a row-wise op == combined, bitwise", eq["bitwise_equal"] and eq["equivalent"] and eq["max_abs"] == 0.0 and eq["max_ulp"] == 0, json.dumps(eq))
    batchdep = lambda xx, ii, ww: (xx.float() + xx.float().mean(0, keepdim=True)).to(torch.bfloat16)  # noqa: E731
    eq2 = C.equivalence(C.split_merge(batchdep, x, ids, w, 32), batchdep(x, ids, w))
    check("a batch-dependent op is caught (not equivalent)", not eq2["equivalent"] and not eq2["bitwise_equal"], json.dumps(eq2))
    one_ulp = (comb.float() * (1 + 2 ** -8)).to(torch.bfloat16)
    eq3 = C.equivalence(one_ulp, comb)
    check("a bf16-ulp difference: not bitwise, but equivalent within rounding", not eq3["bitwise_equal"] and eq3["equivalent"] and eq3["max_ulp"] <= 1, json.dumps(eq3))
    nanb = comb.clone()
    nanb[0, 0] = float("nan")
    check("a NaN is never equivalent", not C.equivalence(nanb, comb)["equivalent"])
    check("shape mismatch is never equivalent", not C.equivalence(comb[:16], comb[:16].repeat(1, 2))["equivalent"])
    # N3-3 / N3-18: a REAL elementwise BF16 ULP gate
    base = torch.linspace(0.5, 2.0, 256).reshape(16, 16).to(torch.bfloat16)
    for k_ulp, want in ((1, True), (2, True), (3, False), (4, False)):
        b2 = base.clone()
        b2.view(torch.int16)[5, 7] += k_ulp
        e = C.equivalence(b2, base)
        check(f"N3-3 ULP gate: one element {k_ulp} ULP off -> {'equivalent' if want else 'FAIL'}",
              e["equivalent"] is want and e["max_ulp"] == k_ulp and e["n_over_max_ulp"] == (0 if want else 1), json.dumps(e))
    neg = -base
    b3 = neg.clone()
    b3.view(torch.int16)[15, 15] -= 4               # the largest negative element (|v| ~ 2 > RMS), 4 ULP away
    check("N3-3 ULP gate: works on negative values (4 ULP off -> FAIL)", C.equivalence(b3, neg)["max_ulp"] == 4 and not C.equivalence(b3, neg)["equivalent"])
    near0 = base.clone()
    near0[0, 0] = 1e-6
    b4 = near0.clone()
    b4[0, 0] = 3e-6                                 # many ULPs apart, but far below the cancellation floor (2 ULP of the RMS)
    e4 = C.equivalence(b4, near0)
    check("N3-3 ULP gate: a near-zero cancellation element (huge ULP distance, |diff| << floor) stays equivalent",
          e4["max_ulp"] > 4 and e4["n_over_max_ulp"] == 0 and e4["equivalent"], json.dumps(e4))
    check("N3-3 equivalence reports finite / bitwise / max_abs / p99_abs / max_ulp", {"finite", "bitwise_equal", "max_abs", "p99_abs", "max_ulp", "rel_fro"} <= set(eq3))
    check("bf16_ulp(1.0) = 2^-7, bf16_ulp(3.0) = 2^-6", C.bf16_ulp(1.0) == 2 ** -7 and C.bf16_ulp(3.0) == 2 ** -6)
    check("EQUIV_TOL no longer claims '1-2 bf16 ulps' from a norm (real ULP check, max 2)", C.EQUIV_TOL["max_ulp"] == 2
          and "~1-2 bf16 ulps" not in (EXP / "nb3_common.py").read_text(encoding="utf-8"))
except ImportError:
    check("torch for the split tests", False)


def ncu_csv(kernels, read=None, write=None, drop_metric_for=None, bad_value_for=None):
    """An ncu --csv text with `kernels` kernel rows (read / write bytes per kernel)."""
    rows = []
    for i in range(kernels):
        rv = (read or [(i + 1) * 1000 for i in range(kernels)])[i]
        wv = (write or [i for i in range(kernels)])[i]
        rows.append(f'"{i}","1","k{i}","dram__bytes_read.sum","byte","{rv:,}"')
        if drop_metric_for != i:
            rows.append(f'"{i}","1","k{i}","dram__bytes_write.sum","byte","{"nan" if bad_value_for == i else wv}"')
    return "==PROF== Connected\n" + '"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n' + "\n".join(rows) + "\nNB3_MOE DONE\n"


rows = C.parse_ncu_csv(ncu_csv(6))
check("ncu CSV parser: 6 kernels, thousands separators handled", len(rows) == 6 and rows[5]["dram__bytes_read.sum"] == 6000.0, str(rows[:2]))
check("ncu CSV parser: no header -> no rows", C.parse_ncu_csv("==ERROR== ERR_NVGPUCTRPERM") == [])
# N3-1 / N3-20: per-arm aggregation, one child run per arm; unequal kernel counts never matter
for ks, kc in ((3, 3), (4, 5), (6, 4), (1, 1)):      # (1+2 vs 3), (2+2 vs 5), (3+3 vs 4), one kernel each
    sp_ = C.ncu_run_verdict(0, ncu_csv(ks, [100] * ks, [1] * ks))
    cb_ = C.ncu_run_verdict(0, ncu_csv(kc, [70] * kc, [2] * kc))
    both = C.ncu_split_combined(sp_, cb_)
    check(f"N3-1 ncu per arm: split {ks} kernels / combined {kc} kernels -> each arm summed over ALL its kernels",
          both["ok"] and both["status"] == "MEASURED" and both["split_dram__bytes_read.sum"] == 100.0 * ks
          and both["combined_dram__bytes_read.sum"] == 70.0 * kc and both["combined_dram__bytes_write.sum"] == 2.0 * kc
          and sp_["kernels"] == ks and cb_["kernels"] == kc, json.dumps({k: both[k] for k in ("ok", "split_dram__bytes_read.sum", "combined_dram__bytes_read.sum")}))
mm = C.ncu_run_verdict(0, ncu_csv(3, drop_metric_for=1))
check("N3-20 a kernel without dram__bytes_write.sum -> ok False MISSING_OR_BAD (never 0 bytes)", mm["ok"] is False
      and mm["status"] == C.NCU_UNVERIFIED and mm["why"] == "MISSING_OR_BAD_dram__bytes_write.sum" and mm["dram__bytes_write.sum"] is None, json.dumps(mm))
nn = C.ncu_run_verdict(0, ncu_csv(3, bad_value_for=0))
check("N3-20 a non-finite metric -> DRAM_BYTES_UNVERIFIED", nn["ok"] is False and nn["why"].startswith("MISSING_OR_BAD_"))
ng = C.ncu_arm_bytes([{"id": 0, "kernel": "k", "dram__bytes_read.sum": -5.0, "dram__bytes_write.sum": 1.0}])
check("N3-20 a negative metric -> DRAM_BYTES_UNVERIFIED", ng["ok"] is False and ng["why"] == "MISSING_OR_BAD_dram__bytes_read.sum")
pe = C.ncu_run_verdict(1, "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters")
check("N3-20 permission error ERR_NVGPUCTRPERM -> DRAM_BYTES_UNVERIFIED", pe["ok"] is False and pe["status"] == C.NCU_UNVERIFIED and "ERR_NVGPUCTRPERM" in pe["why"])
nz = C.ncu_run_verdict(3, ncu_csv(3))
check("N3-1 nonzero ncu exit with a full CSV -> DRAM_BYTES_UNVERIFIED (NCU_EXIT_3)", nz["ok"] is False and nz["why"] == "NCU_EXIT_3")
check("N3-20 no kernel rows -> DRAM_BYTES_UNVERIFIED", C.ncu_run_verdict(0, "==PROF== nothing")["why"] == "NO_KERNEL_ROWS")
half = C.ncu_split_combined(C.ncu_run_verdict(0, ncu_csv(3)), pe)
check("N3-1 one arm unverified -> the pair has no bytes at all (no mixed / zero-filled pair)", half["ok"] is False
      and half["split_dram__bytes_read.sum"] is None and half["combined_dram__bytes_read.sum"] is None and "ERR_NVGPUCTRPERM" in half["why"])
fm_n = jcopy(fm)
fm_n["split"]["8"]["ncu"] = C.ncu_split_combined(C.ncu_run_verdict(0, ncu_csv(3, [1e6] * 3)), C.ncu_run_verdict(0, ncu_csv(3, [5e5] * 3)))
fm_n["split"]["16"]["ncu"] = half
spn = {x["C"]: x for x in C.d5_table(fm_n)["split"]}
check("N3-4 measured dram_gbps only with MEASURED bytes, derived_gbps kept apart", spn[8]["ncu_status"] == "MEASURED" and spn[8]["dram_gbps_split"]
      and spn[8]["derived_gbps_split"] and spn[8]["dram_gbps_split"] != spn[8]["derived_gbps_split"] and spn[16]["dram_gbps_split"] is None
      and spn[16]["ncu_read_split"] is None and spn[16]["ncu_status"] == C.NCU_UNVERIFIED, json.dumps({k: spn[8][k] for k in ("dram_gbps_split", "derived_gbps_split", "ncu_read_split")}))
mb_src = (EXP / "nb3_moe_bench.py").read_text(encoding="utf-8")
check("N3-1 the ncu child profiles exactly ONE arm (--ncu-mode split|combined), parent runs both", '"--ncu-mode", mode' in mb_src
      and 'if a.ncu_mode == "split":' in mb_src and 'for mode in ("split", "combined"):' in mb_src and "ncu_split_bytes" not in mb_src)
# routing replay loader: npz and jsonl.gz, synthetic files in NB1's schema
import gzip
import numpy as np
recs_src = [{"layer": L, "stage": L // 25, "mb_size": m, "step": st, "ids": C.routing_ids(m, "zipf", seed=L * 10 + st)}
            for L in (3, 30) for st in range(4) for m in (8, 16)]
recs_src.append({"layer": 5, "stage": 0, "mb_size": 4, "step": 0, "ids": [[1] * 10, [600] + list(range(9)), list(range(10)), list(range(10, 20))]})
npz = WORK / "nb1_routing_ids.npz"
np.savez(npz, ids=np.array([row for r in recs_src for row in r["ids"]], dtype=np.int32),
         layer=np.array([r["layer"] for r in recs_src]), stage=np.array([r["stage"] for r in recs_src]),
         mb_size=np.array([r["mb_size"] for r in recs_src]), step=np.array([r["step"] for r in recs_src]))
jl = WORK / "nb1_routing_ids.jsonl.gz"
with gzip.open(jl, "wt") as fh:
    for r in recs_src:
        fh.write(json.dumps(r) + "\n")
for path in (npz, jl):
    recs, info = C.load_routing(path)
    check(f"routing loader {path.suffix}: 17 records, rows outside [0,512) dropped (1), the 16 good records intact",
          info["records"] == 17 and info["rows_dropped"] == 1 and recs[0]["ids"] == recs_src[0]["ids"] and info["mb_sizes"] == [3, 8, 16],
          json.dumps({k: info[k] for k in ("records", "rows_dropped", "mb_sizes", "layers")}))
check("npz and jsonl.gz loaders agree", C.load_routing(npz)[0] == C.load_routing(jl)[0])
sel = C.select_replay(C.load_routing(npz)[0], per_size=3)
check("replay selection: per micro-batch size, <= 3 records, spread over layers", set(sel) == {3, 8, 16} and all(len(v) <= 3 for v in sel.values())
      and len({r["layer"] for r in sel[8]}) == 2)
check("replay sizes: core = 8 / 16 (core T), rest = 3", C.replay_sizes(sel, "core") == [8, 16] and C.replay_sizes(sel, "rest") == [3])
stt = C.routing_stats(C.load_routing(npz)[0])
check("routing stats: distinct vs the uniform D(T) per size", [x["T"] for x in stt] == [3, 8, 16] and all(x["real_over_uniform"] for x in stt), json.dumps(stt))
r, jr = run_moe_fake(WORK / "moe_replay.json", ["--routing-file", str(npz)])
check("moe bench --fake with a routing file: replay available, sizes replayed (core 8/16 + rest 3)", r.returncode == 0 and jr["replay"]["available"]
      and sorted(b["T"] for b in jr["replay"]["by_size"]) == [3, 8, 16], (r.stdout + r.stderr)[-300:])
r, jn = run_moe_fake(WORK / "moe_noreplay.json", ["--routing-file", str(WORK / "absent.npz")])
check("moe bench with an absent routing file: 'real routing unavailable', synthetic kept", r.returncode == 0 and jn["replay"]["available"] is False
      and "unavailable" in jn["replay"]["why"] and "hot" in jn["baseline"]["16"])
check("moe bench: split section in the JSON (C 8/16/24/32)", set(jn["split"]) == {"8", "16", "24", "32"})


# ------------------------------------------------------------------ 5. the patched P5 bench (planning, no GPU)
os.environ.update(NB3_AR_SIZES=json.dumps([2560, 81920]), NB3_AR_DECODE="81920", NB3_AR_ITERS="[7, 2, 3, 2]")
B = load("nb3_bench_host_ar", EXP / "bench_host_ar.py")
check("P5 bench SIZES / DECODE / iters from the NB3 env", B.SIZES == [2560, 81920] and B.DECODE == 81920 and B.NB3_ITERS == [7, 2, 3, 2])
check("P5 bench has the 'n' NCCL numeric step", "n" in B.STEPS and B.STEPS["n"] is B.nb3_nccl_check)
src = (EXP / "bench_host_ar.py").read_text()
check("P5 iters patch applied in both latency steps", src.count("tuple(NB3_ITERS) if NB3_ITERS") == 2)
check("P5 NCCL debug subsys includes TUNING", '"INIT,ENV,GRAPH,TUNING"' in src)
check("N3-17 P5 records the per-size timed-region monotonic windows (nccl + host)", src.count("time.monotonic()") == 4
      and 'r["t_mono"] = {"nccl": [' in src and 'r.setdefault("t_mono", {})["host"]' in src)
calls = []
B.run_group = lambda a, gname, steps, extra, budget, outdir: calls.append((gname, steps, extra)) or {
    "status": "ok", "rcs": [0, 0], "sec": 1, "rank0": {"checks": {"x": True}}, "other_ranks": {}}
B.env_probe = lambda: {"nvidia_smi_L": {"rc": 0, "out": "GPU 0\nGPU 1\nGPU 2\nGPU 3"}}
so = WORK / "fake.so"
so.write_bytes(b"")


class A:
    so, out, world, steps, quick, timeout_s, kernel_timeout_ms = str(so), str(WORK / "p5plan.json"), 2, "a,b,c,d", True, 600, 1000


for arms, want in (("{}", [("main_ab", "n,a,b"), ("main_cd", "c,d")]),
                   ('{"armx": {"NCCL_X": "1"}}', [("main_ab", "n,a,b"), ("armx", "n,b_nccl"), ("main_cd", "c,d")])):
    os.environ["NB3_AR_NCCL_ARMS"] = arms
    calls.clear()
    try:
        B.parent(A)
    except SystemExit:
        pass
    check(f"P5 plan with arms {arms}: groups {want}", [(g, s) for g, s, _ in calls] == want, str(calls))
check("P5 plan: no NCCL CTA groups any more", not any(g.startswith("nccl_ctas") for g, _, _ in calls))
check("P5 shm sweep disabled by NB3_AR_NO_SHM_SWEEP", "os.environ.get(\"NB3_AR_NO_SHM_SWEEP\")" in src)

# ------------------------------------------------------------------ 6. fake benches as subprocesses (exit 0, schema)
for script, extra in (("nb3_moe_bench.py", ["--ckpt", str(WORK)]), ("nb3_gemm_bench.py", [])):
    out = WORK / (script + ".json")
    r = subprocess.run([sys.executable, str(EXP / script), "--fake", "--out", str(out)] + extra, capture_output=True, text=True, timeout=300,
                       env={**BASE_ENV, "NB3_FAKE_FAULTS": ""})
    j = json.loads(out.read_text()) if out.is_file() else {}
    check(f"{script} --fake exits 0 with status DONE", r.returncode == 0 and j.get("status") == "DONE", (r.stdout + r.stderr)[-300:])
    if script == "nb3_moe_bench.py":
        check("moe bench CPU self-test (pack round trip + reference + GPTQ ref)", (j.get("cpu_selftest") or {}).get("ok") is True, str(j.get("cpu_selftest"))[:200])
        check("moe bench records the synthetic-routing note", "SYNTHETIC" in (j.get("routing_note") or ""))
        check("moe bench all-phase: every unit done, EP1 spot recorded, checks incl. ep1_spot_reference / dequant / first_call_gate",
              not j["units_skipped"] and len(j["units_done"]) == len(C.d5_units("all")) and j["checks"].get("ep1_spot_reference") is True
              and j["checks"].get("first_call_gate") is True and j["checks"].get("dequant_crosscheck") is True, json.dumps(j["checks"]))
    else:
        check("gemm bench --fake carries draft heads + dispatch + graph status", j.get("draft_heads", {}).get("status") == "DONE"
              and all((r_.get("dispatch") or {}).get("ok") is True for s_ in j["shapes"].values() for r_ in s_["M"].values()))
r = subprocess.run([sys.executable, str(EXP / "nb3_common.py"), "--fake-ar", "--out", str(WORK / "ar.json"), "--sizes", "[2560, 5120]"],
                   capture_output=True, text=True, timeout=60, env={**BASE_ENV, "NB3_FAKE_FAULTS": ""})
check("nb3_common --fake-ar writes a P5-shaped JSON", r.returncode == 0 and "main_ab" in json.loads((WORK / "ar.json").read_text())["groups"])
r = subprocess.run([sys.executable, str(EXP / "nb3_moe_bench.py"), "--ckpt", str(WORK / "nope"), "--out", str(WORK / "moe_err.json")],
                   capture_output=True, text=True, timeout=300)
je = json.loads((WORK / "moe_err.json").read_text())
check("moe bench without vLLM / checkpoint -> exit 2, status ERROR, JSON written, no unit run", r.returncode == 2 and je["status"] == "ERROR"
      and not je["units_done"], je["errors"][:1])


# ------------------------------------------------------------------ 7. fake input tree (RC2 harness layout)
def fake_tree(inp):
    MD = inp / "models" / "foysalemonshanto"
    PD = {k: MD / f"qwen3-8-flash-next-w4a16-merlin-p{k}/transformers/default/1" for k in range(1, 6)}
    shutil.copytree(P1, PD[1])
    idx = json.loads((PD[1] / "model.safetensors.index.json").read_text())
    ple = set(PARTS["ple_from_overlay"])
    man1 = json.loads((PD[1] / "q27_fn_merlin_part1_manifest.json").read_text())
    shards = sorted(set(idx["weight_map"].values()) - ple - set(man1["files"]))

    def fake(path, n):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (n * 997).to_bytes(8, "little") * 16
        path.write_bytes(data)
        return [len(data), hashlib.sha256(data).hexdigest()]
    groups = [shards[i::5] for i in range(5)]
    for i, n in enumerate(groups[0]):
        man1["files"][n] = fake(PD[1] / n, i)
    (PD[1] / "q27_fn_merlin_part1_manifest.json").write_text(json.dumps(man1))
    for k in range(2, 6):
        man = {"part": k, "model_subdir": ".", "files": {n: fake(PD[k] / n, 100 * k + i) for i, n in enumerate(groups[k - 1])}}
        (PD[k] / f"q27_fn_merlin_part{k}_manifest.json").write_text(json.dumps(man))
    for i, n in enumerate(sorted(ple)):
        fake(MD / "qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1" / n, 9000 + i)
    mp = MD / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin"
    mp.parent.mkdir(parents=True)
    mp.write_bytes(b"")
    c = inp / "competitions" / "arc-prize-2026-arc-agi-2"
    c.mkdir(parents=True)
    shutil.copyfile(EVAL, c / "arc-agi_evaluation_challenges.json")
    shutil.copyfile(EVAL, c / "arc-agi_test_challenges.json")


INP = WORK / "input"
fake_tree(INP)
TEST_TASKS = json.loads((INP / "competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").read_text())


ROUTING_NPZ = WORK / "nb1_routing_ids.npz"


def env_for(tag, extra=None):
    for d in ("out", "tmp", "shm"):
        (WORK / tag / d).mkdir(parents=True, exist_ok=True)
    e = dict(BASE_ENV)
    e.update(E23_FAKE_SERVER=str(HERE / "fake_marker_not_used.py"), E23_OUT=str(WORK / tag / "out"), E23_TMP=str(WORK / tag / "tmp"),
             E23_INPUT=str(INP), RC2_SUBMISSION=str(WORK / tag / "out/submission.json"), RC2_SHM=str(WORK / tag / "shm"),
             NB3_BUDGET_SEC="3600")
    for k in ("KAGGLE_IS_COMPETITION_RERUN", "NB3_UNIT", "NB3_AR_NCCL_ARMS", "NB3_AR_SIZES", "NB3_AR_DECODE", "NB3_AR_ITERS",
              "NB3_ROUTING_FILE", "NB3_FAKE_FAULTS", "NB3_CPU_KIT", "NB3_D7_PAIRS_SEC", "NB3_D7_SOLO_SEC"):
        e.pop(k, None)
    if tag == "e2e":
        e["NB3_ROUTING_FILE"] = str(ROUTING_NPZ)
    e.update(extra or {})
    return e


def write_res(path, status="DONE", exit_code=0, checks=None):
    C.write_json(path, {"status": status, "exit_code": exit_code, "checks": {"x": True} if checks is None else checks})
    return path


# ------------------------------------------------------------------ 8. orchestration fault injection (NB3_UNIT=1)
os.environ.clear()
os.environ.update(env_for("unit"), NB3_UNIT="1")
ns = runpy.run_path(str(HERE / "nb3_code_export.py"), run_name="nb3")["finish"].__globals__
FAKE0, BOOT0 = ns["FAKE"], ns["boot_gate"]
check("unit import: merge ok on the fake tree", ns["R"]["sections"]["A0"]["merge"].get("ok") is True, str(ns["R"]["sections"]["A0"]["merge"])[:200])
check("unit import: no bench ran (NB3_UNIT)", not any(k in ns["R"]["sections"] for k in ("D5", "D7", "D4")))
shas = ns["R"]["setup"]["nb3_files"]
check("embedded files: 10 files (incl. nb3_humming.py), each sha256-checked, identical to Build/nb3_export, audit copy in OUT/nb3_files",
      len(shas) == 10 and "nb3_humming.py" in shas and all(hashlib.sha256((EXP / k).read_bytes()).hexdigest() == v for k, v in shas.items())
      and all((WORK / "unit/out/nb3_files" / k).is_file() for k in shas), str(sorted(shas)))
src_cells = (HERE / "nb3_code_export.py").read_text(encoding="utf-8")
check("embedded-file tamper check present (sha mismatch raises)", "NB3 embedded files fail their sha256" in src_cells)
check("unit import: no routing file -> 'real routing unavailable' recorded", ns["ROUTING_FILE"] is None and
      ns["R"]["setup"]["routing_file"]["status"] == "real routing unavailable")
check("AR sizes = hidden 2560 x T x 2 B", ns["AR"]["sizes"] == [2560 * t * 2 for t in ns["TS"]["all"]] and ns["AR"]["decode"] == 81920)
check("NCCL env for D7: PROTO / ALGO unset, DEBUG INFO", (lambda e: e["NCCL_PROTO"] is None and e["NCCL_ALGO"] is None and e["NCCL_DEBUG"] == "INFO" and e["NB3_AR_NCCL_ARMS"] == "{}")(ns["ar_extra"]()))
check("N3-5 AR env text says LL128 UNTESTED", "UNTESTED (not run, NCCL docs corruption risk)" in ns["AR"]["nccl_env"])
e = ns["bench_env"]("2,3", {"NCCL_PROTO": None, "X": 1})
check("bench_env: CUDA_VISIBLE_DEVICES per bench, None pops", e["CUDA_VISIBLE_DEVICES"] == "2,3" and "NCCL_PROTO" not in e and e["X"] == "1")
scr = WORK / "rc_script.py"
scr.write_text("import sys, time\nprint('hello', flush=True)\nif sys.argv[1] == 'sleep':\n    time.sleep(120)\nsys.exit(int(sys.argv[1]) if sys.argv[1] != 'sleep' else 0)\n")
ns["BENCH"]["py"] = sys.executable
# N3-22: the verified status (rc, timeout, GPU release, file status, exit_code, every check)
for rc, res_kw, want in (("0", {}, "DONE"), ("1", {"exit_code": 1, "checks": {"x": False}}, "CHECK_FAILED"), ("2", {"status": "ERROR", "exit_code": 2}, "BENCH_ERROR"),
                         ("0", {"status": "PARTIAL_BUDGET"}, "PARTIAL_BUDGET"), ("0", {"exit_code": 1}, "CHECK_FAILED"),
                         ("0", {"checks": {"x": True, "y": False}}, "CHECK_FAILED"), ("0", {"checks": {}}, "BENCH_ERROR"),
                         ("0", None, "BENCH_ERROR"), ("1", {"status": "DONE"}, "CHECK_FAILED")):
    jp = write_res(WORK / f"res_{rc}_{want}.json", **res_kw) if res_kw is not None else WORK / "none.json"
    r = ns["run_bench"](f"rc{rc}", [scr, rc], "0", 60)
    o = ns["finish"]("UNIT", r, jp, "0")
    check(f"N3-22 bench rc {rc}, file {res_kw if res_kw is not None else 'missing'} -> {want}", o["status"] == want, f"{o['status']} {o['status_reasons']}")
t = time.time()
r = ns["run_bench"]("sleeper", [scr, "sleep"], "0", 1)
check("a stuck bench is killed at its deadline (timeout flag, rc -9)", r["timeout"] and r["rc"] == -9 and time.time() - t < 100, f"{round(time.time() - t)} s")
check("killed bench reported TIMEOUT (even with a DONE file)", ns["finish"]("UNIT", r, write_res(WORK / "res_to.json"), "0")["status"] == "TIMEOUT")
check("no bench process left registered", not ns["BENCH"]["procs"])
vs = C.verified_bench_status
check("N3-22 verified status: GPU not released (file DONE) -> GPU_NOT_RELEASED; unknown release only allowed in the harness",
      vs({"rc": 0}, {"status": "DONE", "exit_code": 0, "checks": {"a": True}}, False)[0] == "GPU_NOT_RELEASED"
      and vs({"rc": 0}, {"status": "DONE", "exit_code": 0, "checks": {"a": True}}, None)[0] != "DONE"
      and vs({"rc": 0}, {"status": "DONE", "exit_code": 0, "checks": {"a": True}}, None, True)[0] == "DONE")
# N3-13: the global SHA gate before EVERY GPU bench; on failure no subprocess at all
started = []
start0 = ns["start_bench"]
ns["start_bench"] = lambda name, *a_, **k_: started.append(name) or start0(name, *a_, **k_)
ns["FAKE"] = None
ns["BENCH"]["plan_ok"] = True
ns["P5"]["so"] = "x.so"
for why in ("sha256 mismatch: ['model-00003-of-00017.safetensors']", "sha256 check not finished"):
    ns["ABORT"]["why"] = None
    ns["R"]["setup"]["sha_gate"] = {"ok": False, "why": why}
    outs = {n: ns[f]() for n, f in (("D5 core", "s_d5"), ("D7 build", "s_p5_build"), ("D7 pairs", "s_d7_pairs"), ("D7 solo", "s_d7_solo"), ("D4", "s_d4"))}
    outs["D5 rest"] = ns["s_d5"]("rest")
    check(f"N3-13 SHA gate '{why[:24]}': D5 core/rest, D7 build/pairs/solo, D4 all SKIPPED_SHA_GATE, no GPU subprocess, ABORT + evidence",
          all(v["status"] == "SKIPPED_SHA_GATE" for v in outs.values()) and not started and ns["ABORT"]["why"]
          and len(ns["R"]["setup"]["sha_gate_refusals"]) >= 6, json.dumps({k: v["status"] for k, v in outs.items()}))
ns["R"]["setup"]["sha_gate"] = {"ok": None}
check("N3-13 SHA gate never run -> refused too", ns["s_d4"]()["status"] == "SKIPPED_SHA_GATE" and not started)
ns["ABORT"]["why"] = None
ns["R"]["setup"]["sha_gate"] = {"ok": True}
ns["BENCH"]["plan_ok"] = False
check("N3-23 no verified bench env -> every bench SKIPPED_NO_BENCH_ENV", all(ns[f]()["status"] == "SKIPPED_NO_BENCH_ENV"
                                                                              for f in ("s_d5", "s_p5_build", "s_d7_pairs", "s_d7_solo", "s_d4")) and not started)
ns["BENCH"]["plan_ok"] = True
ns["P5"]["so"] = None
check("D7 pairs skipped without a probed P5 .so", ns["s_d7_pairs"]()["status"] == "SKIPPED_NO_P5")
check("D7 solo skipped without a probed P5 .so", ns["s_d7_solo"]()["status"] == "SKIPPED_NO_P5")
ns["boot_gate"] = lambda: {"ok": False, "gpus": [[0, 900, 23034]]}
check("D4 never starts on a non-empty GPU", ns["s_d4"]()["status"] == "GPU_NOT_EMPTY")
check("D5 never starts on a non-empty GPU", ns["s_d5"]()["status"] == "GPU_NOT_EMPTY" and not started)
ns["start_bench"] = start0
# N3-23: s_plan fails closed on the GPU run; EXPLORATORY_ENV only under NB3_CPU_KIT
venv = WORK / "venv"
(venv / "bin").mkdir(parents=True)
(venv / "bin" / "python").write_text("")
ns["merlin_install"] = lambda: {"kv": {"MERLIN_VENV": str(venv), "MERLIN_SHIM": "/shim.so"}, "cuda_home": "/cuda"}
good_srv = lambda *a_, **k_: types.SimpleNamespace(env={"NCCL_P2P_DISABLE": "1", "NCCL_PROTO": "LL128", "VLLM_PP_LAYER_PARTITION": "25,23"})  # noqa: E731


def raiser(exc):
    def f(*a_, **k_):
        raise exc
    return f


for label, srv in (("missing scales", raiser(FileNotFoundError("qsa_kv_scales_262k.json not found"))),
                   ("failed host_phase", raiser(RuntimeError("MERLIN_HOST_PHASE_FAILED: config.json lacks text_config.ple_embedding_dtype")))):
    ns["merlin_fn_server"] = srv
    os.environ.pop("NB3_CPU_KIT", None)
    try:
        ns["s_plan"]()
        raised = None
    except RuntimeError as ex:
        raised = str(ex)
    check(f"N3-23 s_plan GPU run, {label} -> RuntimeError BENCH_ENV_FAIL_CLOSED, no bench env", raised and "BENCH_ENV_FAIL_CLOSED" in raised
          and ns["BENCH"]["plan_ok"] is False and ns["s_d4"]()["status"] == "SKIPPED_NO_BENCH_ENV", str(raised)[:200])
    os.environ["NB3_CPU_KIT"] = "1"
    out = ns["s_plan"]()
    check(f"N3-23 s_plan CPU kit, {label} -> EXPLORATORY_ENV (labelled), bench env usable", out["plan_label"] == "EXPLORATORY_ENV"
          and out["plan"].startswith("EXPLORATORY_ENV") and ns["BENCH"]["plan_ok"] is True, out["plan"][:120])
    os.environ.pop("NB3_CPU_KIT", None)
ns["merlin_fn_server"] = good_srv
ns["merlin_install"] = lambda: {"kv": {"MERLIN_VENV": str(WORK / "no_such_venv"), "MERLIN_SHIM": "/shim.so"}, "cuda_home": "/cuda"}
try:
    ns["s_plan"]()
    raised = None
except RuntimeError as ex:
    raised = str(ex)
check("N3-23 s_plan GPU run, broken paths (venv python missing) -> fail closed", raised and "BENCH_ENV_FAIL_CLOSED" in raised and ns["BENCH"]["plan_ok"] is False)
ns["merlin_install"] = lambda: {"kv": {"MERLIN_VENV": str(venv), "MERLIN_SHIM": "/shim.so"}, "cuda_home": "/cuda"}
out = ns["s_plan"]()
check("N3-23 s_plan good plan -> RC2_FP8_PLAN, venv python, NCCL_PROTO / ALGO dropped", out["plan_label"] == "RC2_FP8_PLAN" and ns["BENCH"]["plan_ok"] is True
      and ns["BENCH"]["py"] == f"{venv}/bin/python" and "NCCL_PROTO" not in ns["BENCH"]["env"], json.dumps(out)[:200])
ns["merlin_fn_server"] = raiser(RuntimeError("MERLIN_HOST_PHASE_FAILED: x"))
rec = ns["step"]("B0", "bench_env_unit", 0, ns["s_plan"])
check("N3-23 through the step runner: bench_env ERROR, then the benches are skipped", rec["status"] == "ERROR" and "BENCH_ENV_FAIL_CLOSED" in rec["error"]
      and ns["s_d5"]()["status"] == "SKIPPED_NO_BENCH_ENV")
# N3-19 failure policy: a device fault line or leftover VRAM stops the notebook; UNSUPPORTED_SM89 lets the other arms run
ns["FAKE"], ns["boot_gate"] = FAKE0, BOOT0
ns["BENCH"].update(py=sys.executable, plan_ok=True)
ns["ABORT"]["why"] = None
fg_ = ns["fault_gate"](["NVRM: Xid (PCI:0000:00:04): 79, pid=1"], "unit")
check("Xid 79 in a bench log -> ABORT (reset class)", fg_ == "ABORT" and ns["ABORT"]["why"])
ns["ABORT"]["why"] = None
r = ns["run_bench"]("ima", [scr, "0"], "0", 60)
r["_log"] = "RuntimeError: CUDA error: an illegal memory access was encountered\n"
o = ns["finish"]("UNIT", r, write_res(WORK / "res_ima.json"), "0")
check("N3-19 illegal memory access in a bench log -> NO_GO_GPU_FAULT and the notebook stops (ABORT)", o["status"] == "NO_GO_GPU_FAULT"
      and "notebook stops" in (ns["ABORT"]["why"] or ""), str(ns["ABORT"]["why"]))
rec = ns["step"]("D4", "after_fault_unit", 0, lambda: {"status": "DONE"})
check("N3-19 after a device fault every later step is SKIPPED_ABORT", rec["status"] == "SKIPPED_ABORT")
ns["ABORT"]["why"] = None
wgr0 = ns["L"].wait_gpu_release
ns["L"].wait_gpu_release = lambda *a_, **k_: False
r = ns["run_bench"]("vram", [scr, "0"], "0", 60)
o = ns["finish"]("UNIT", r, write_res(WORK / "res_vram.json"), "0")
check("N3-19 leftover VRAM after a bench -> GPU_NOT_RELEASED and ABORT", o["status"] == "GPU_NOT_RELEASED" and ns["ABORT"]["why"], str(ns["ABORT"]["why"]))
ns["L"].wait_gpu_release = wgr0
ns["ABORT"]["why"] = None
ns["R"]["setup"]["sha_gate"] = {"ok": True}
ns["BENCH_STATUS"].clear()
ns["BENCH"]["env"]["NB3_FAKE_FAULTS"] = "d4_unsupported"
o4 = ns["s_d4"]()
ns["BENCH"]["env"]["NB3_FAKE_FAULTS"] = ""
o5 = ns["s_d5"]("rest")
scu = json.loads((WORK / "unit/out/lever_scorecard.json").read_text())
r4 = next(x for x in scu["rows"] if x["lever"] == "D4")
check("N3-18 missing sm_89 CuTe -> D4 UNSUPPORTED_SM89 (diagnostic only), no ABORT, the next arm (D5 rest) still runs DONE",
      o4["status"] == "UNSUPPORTED_SM89" and not ns["ABORT"]["why"] and o5["status"] == "DONE" and r4["promotion"] == C.DIAGNOSTIC_ONLY
      and r4["score"] == 0 and (o4.get("sm89_skinny") or {}).get("status") == "UNSUPPORTED_SM89", json.dumps([o4["status"], o5["status"], r4["promotion"]]))
probe_ok = {"info": "p5_host_ar v2 two-shot host-staged all-reduce; worlds 2/4/8; ...", "cover_check_world2_bad": []}
check("P5 probe text names worlds 2/4/8 (source facts recorded)", "worlds 2" in probe_ok["info"] and ns["P5_SOURCE_FACTS"]["world_check"].startswith("host_ar_binding.cpp:112"))
ns["nb3_report"]()
check("scorecard + tables written incrementally (even with no data)", (WORK / "unit/out/lever_scorecard.json").is_file() and (WORK / "unit/out/nb3_tables.json").is_file())
os.environ.pop("NB3_CPU_KIT", None)

# ------------------------------------------------------------------ 9. full FAKE end-to-end of the notebook code (all valid)
r = subprocess.run([sys.executable, str(HERE / "nb3_code_export.py")], env=env_for("e2e"), capture_output=True, text=True, timeout=1800)
out = WORK / "e2e" / "out"
(WORK / "e2e_stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
res = json.loads((out / "nb3_results.json").read_text()) if (out / "nb3_results.json").is_file() else {}
st = {f"{s}/{n}": v.get("status") for s, d in (res.get("sections") or {}).items() for n, v in d.items() if isinstance(v, dict) and "status" in v}
want = {"A0/merlin_install", "A0/sha_gate", "B0/bench_env", "D5/marlin_moe", "D7/p5_build", "D7/allreduce_pairs", "D7/allreduce_solo",
        "D4/gemm_plans_heads", "D5/marlin_moe_rest"}
check("e2e: rc 0", r.returncode == 0, r.stderr[-800:])
check("e2e: every step DONE", want <= set(st) and all(st[k] == "DONE" for k in want), json.dumps(st))
check("N3-19 e2e run order: SHA -> bench env -> D5 core -> D7 -> D4 -> D5 rest", [x for x in res.get("order", []) if x in want] ==
      ["A0/merlin_install", "A0/sha_gate", "B0/bench_env", "D5/marlin_moe", "D7/p5_build", "D7/allreduce_pairs", "D7/allreduce_solo",
       "D4/gemm_plans_heads", "D5/marlin_moe_rest"], str(res.get("order")))
check("e2e: verified bench statuses all DONE", res.get("bench_status") == {"D5": "DONE", "D7": "DONE", "D7_solo": "DONE", "D4": "DONE", "D5_rest": "DONE"},
      json.dumps(res.get("bench_status")))
sc = json.loads((out / "lever_scorecard.json").read_text()) if (out / "lever_scorecard.json").is_file() else {}
check("e2e: lever_scorecard.json rows D5 D7 D4 D2 PP2 C6-humming", [x["lever"] for x in sc.get("rows", [])] == ["D5", "D7", "D4", "D2", "PP2", "C6-humming"])
check("e2e: C6-humming measured in D5 core (fake), research row, DIAGNOSTIC_ONLY, score 0, 16 arm x T rows",
      next(x for x in sc["rows"] if x["lever"] == "C6-humming")["promotion"] == "DIAGNOSTIC_ONLY"
      and len(next(x for x in sc["rows"] if x["lever"] == "C6-humming")["measured"]["rows"]) == 16
      and next(x for x in sc["rows"] if x["lever"] == "C6-humming")["score"] == 0)
d5rec = res["sections"]["D5"]["marlin_moe"]
tb = json.loads((out / "nb3_tables.json").read_text()) if (out / "nb3_tables.json").is_file() else {}
check("e2e: routing file (NB3_ROUTING_FILE) found, passed to D5, replayed (core + rest sizes)", d5rec["routing_file"]["status"] == "attached"
      and d5rec["replay"]["available"] is True and sorted(x["T"] for x in tb["tables"]["d5"]["replay"]["rows"]) == [3, 8, 16],
      json.dumps(tb["tables"]["d5"]["replay"]["rows"])[:300])
check("e2e: D5 tables merged from core + rest (84 baseline rows, sweep 9 T, split 4 C, EP1 spot)", len(tb["tables"]["d5"]["baseline"]) == 84
      and len(tb["tables"]["d5"]["sweep"]) == 9 and len(tb["tables"]["d5"]["split"]) == 4 and (tb["tables"]["d5"]["ep1_spot"] or {}).get("ok") is True)
check("e2e: tables d5 / d7 / d4 + ranked NB4 candidates", {"d5", "d7", "d4"} <= set(tb.get("tables", {})) and
      tb["nb4_candidates"]["targets"][0]["ranked"])
check("N3-25 positive case: all-valid run still ranks D5 + D7 + D4 winners (nb4_top)", {"D5_marlin_moe_cfg", "D7_p5_host_allreduce"} <= {c["id"] for c in res.get("nb4_top", [])}
      and any(c["id"].startswith("D4_") for c in res.get("nb4_top", [])), json.dumps(res.get("nb4_top"))[:300])
check("e2e: D7 table merged pairs + solo (solo column filled), concurrency VERIFIED", all(x.get("solo_nccl_graph") for x in tb["tables"]["d7"]["rows"])
      and tb["tables"]["d7"]["concurrency"]["status"] == "VERIFIED")
check("e2e: fallback submission written", (out / "submission.json").is_file())
check("e2e: bench logs saved", all((out / "logs" / f"bench_{n}.log").is_file() for n in ("d5_moe_core", "d5_moe_rest", "d7_pair01", "d7_pair23", "d7_solo01", "d4_gemm")))
check("e2e: within budget, no abort", not res.get("errors") or all("abort" not in str(e).lower() for e in res["errors"]), str(res.get("errors"))[:300])
os.environ.clear()
os.environ.update(env_for("rerun"), KAGGLE_IS_COMPETITION_RERUN="1")
r = subprocess.run([sys.executable, str(HERE / "nb3_code_export.py")], env=dict(os.environ), capture_output=True, text=True, timeout=300)
check("rerun: rc 0, fallback submission only, no benches", r.returncode == 0 and (WORK / "rerun/out/submission.json").is_file()
      and not (WORK / "rerun/out/nb3_tables.json").is_file(), r.stderr[-300:])

# ------------------------------------------------------------------ 10. audit regressions (N3-1 .. N3-24; the N3-18 nine marked)
# N3-18 (1) PASS_BOUND_ONLY never ranked / scored
sc_all = C.scorecard(tables, ST_OK)
d7row = next(x for x in sc_all["rows"] if x["lever"] == "D7")
check("N3-18(1) PASS_BOUND_ONLY never ranked nor the scorecard's best (D7 best = native P5)", d7row["best_candidate"]["id"] == "D7_p5_host_allreduce"
      and all(c["eligibility"] != C.PROMOTION_READY for tg in rk["targets"] for c in tg["diagnostic"] if c["correctness"] == "PASS_BOUND_ONLY"))
only_int8 = jcopy(tables)
for row in only_int8["d7"]["rows"]:
    row["p5_graph"] = None
r7i = next(x for x in C.scorecard(only_int8, ST_OK)["rows"] if x["lever"] == "D7")
check("N3-11 D7 with only the INT8 arm (PASS_BOUND_ONLY) -> DIAGNOSTIC_ONLY, score 0, gain 0", r7i["promotion"] == C.DIAGNOSTIC_ONLY and r7i["score"] == 0
      and (r7i["projected_run_tok_s_gain"]["run_tok_s_gain_headline"] or 0) == 0 and r7i["diagnostic_best"]["id"] == "D7_p5_int8_allreduce",
      json.dumps({k: r7i[k] for k in ("promotion", "score", "correctness")}))
# N3-18 (2) / N3-2: a failed bench never ranked / scored (one PASS candidate then rc 1)
st_fail = C.verified_bench_status({"rc": 1}, {"status": "DONE", "exit_code": 1, "checks": {"reference_all_T": True, "split_equivalent": False}}, True)[0]
for fam, stv in (("D5", st_fail), ("D7", "TIMEOUT"), ("D4", "BENCH_ERROR")):
    stt_ = {**ST_OK, fam: stv}
    rk_f = C.rank_candidates(tables, status=stt_)
    rowf = next(x for x in C.scorecard(tables, stt_)["rows"] if x["lever"] == fam)
    check(f"N3-18(2) N3-2 {fam} bench {stv} with PASS candidates -> 0 ranked for that family, scorecard BLOCKED_BENCH_FAILURE, score 0, gain 0",
          not any(c["lever"] == fam for c in rk_f["targets"][0]["ranked"]) and any(c["lever"] == fam and c["eligibility"] == C.BLOCKED_BENCH_FAILURE
                                                                                   for c in rk_f["targets"][0]["blocked"])
          and rowf["promotion"] == C.BLOCKED_BENCH_FAILURE and rowf["score"] == 0 and rowf["projected_run_tok_s_gain"]["run_tok_s_gain_headline"] == 0
          and rowf["diagnostic_best"] and rowf["candidates_all"], json.dumps({k: rowf[k] for k in ("promotion", "score", "bench_status")}))
check("N3-2 no verified status at all -> nothing ranked", not C.rank_candidates(tables)["targets"][0]["ranked"])
# N3-18 (3) / N3-12: graph capture error -> no graph winner from eager numbers
tm = {"eager_us": {"median": 12.0}, "graph_error": "RuntimeError: capture failed"}
tt = C.timing(tm)
check("N3-12 timing(): graph_us None, eager kept apart, INCONCLUSIVE_CAPTURE_FAILED + graph_error", tt["graph_us"] is None and tt["eager_us"]["median"] == 12.0
      and tt["graph_status"] == "INCONCLUSIVE_CAPTURE_FAILED" and "capture failed" in tt["graph_error"])
r, jg = run_moe_fake(WORK / "moe_graph_err.json", faults="graph_error")
row16 = jg["baseline"]["16"]["independent"]
tg_ = C.d5_table(jg)
check("N3-18(3) D5 graph capture error at T16 -> graph_us null (eager kept), sweep has no base, no D5 candidate at the RC2 point",
      row16["graph_us"] is None and row16["eager_us"] and row16["graph_status"] == "INCONCLUSIVE_CAPTURE_FAILED"
      and jg["sweep"]["16"]["best"] is None and not any(c["id"] == "D5_marlin_moe_cfg" for c in C.candidates({**tables, "d5": tg_})),
      json.dumps({k: row16.get(k) for k in ("graph_us", "graph_status", "graph_error")}))
fg_g = jcopy(fg)
for s_ in fg_g["shapes"].values():
    for r_ in s_["M"].values():
        r_["best"] = None                        # what nb3_gemm_bench writes when no config has a graph number
        r_["configs"] = [{"source": "heuristic", "ok": True, "graph_us": None, "eager_us": {"median": 3.0}, "graph_status": "INCONCLUSIVE_CAPTURE_FAILED"}]
check("N3-12 D4 configs without a graph number -> no D4 winner", not any(c["lever"] == "D4" for c in C.candidates({**tables, "d4": C.d4_table(fg_g)}))
      and C.d4_table(fg_g)["rows"][0]["configs_graph_inconclusive"] == 1)
gb_src = (EXP / "nb3_gemm_bench.py").read_text(encoding="utf-8")
check("N3-12 no 'graph_us') or ... eager_us' substitution left in the benches", 'get("graph_us") or' not in gb_src and 'get("graph_us") or' not in mb_src)
# N3-18 (4) SHA mismatch / timeout -> no GPU subprocess: section 8 (N3-13 checks)
# N3-18 (5) / N3-14: one pair only / one rank only / one check missing -> INCONCLUSIVE
one = C.d7_table({"0,1": fa["0,1"]}, 2560, 2)
check("N3-18(5) N3-14 one pair only -> every arm INCONCLUSIVE_MISSING_PAIR, no row value, no D7 candidate",
      set(one["arms"].values()) == {"INCONCLUSIVE_MISSING_PAIR"} and all(x["nccl_graph"] is None for x in one["rows"])
      and not any(c["lever"] == "D7" for c in C.candidates({**tables, "d7": one})), json.dumps(one["arms"]))
empty_pair = C.d7_table({"0,1": fa["0,1"], "2,3": {}}, 2560, 2)
check("N3-14 a pair JSON missing (file absent) -> INCONCLUSIVE_MISSING_PAIR", set(empty_pair["arms"].values()) == {"INCONCLUSIVE_MISSING_PAIR"})
r1 = {"0,1": C.fake_ar(SIZES, 1, {"missing_rank1"}), "2,3": fa["2,3"]}
t_r1 = C.d7_table(r1, 2560, 2)
c_r1 = next(c for c in C.candidates({**tables, "d7": t_r1}) if c["id"] == "D7_p5_host_allreduce")
check("N3-14 rank 1 missing on one pair -> INCONCLUSIVE_MISSING_RANK, candidate BLOCKED_CORRECTNESS", t_r1["arms"]["p5"] == "INCONCLUSIVE_MISSING_RANK"
      and c_r1["correctness"] == "INCONCLUSIVE_MISSING_RANK" and C.eligibility(c_r1, ST_OK) == C.BLOCKED_CORRECTNESS)
mc = jcopy(fa)
del mc["2,3"]["groups"]["main_cd"]["other_ranks"]["1"]["checks"]["c_error_path_loud"]
t_mc = C.d7_table(mc, 2560, 2)
check("N3-14 one mandatory check absent on rank 1 -> INCONCLUSIVE_MISSING_CHECK (p5), the INT8 / NCCL arms unaffected",
      t_mc["arms"]["p5"] == "INCONCLUSIVE_MISSING_CHECK" and t_mc["arms"]["nccl"] == "PASS" and t_mc["arms"]["p5_int8"] == "PASS", json.dumps(t_mc["arms"]))
nf_ = jcopy(fa)
nf_["0,1"]["groups"]["main_ab"]["other_ranks"]["1"]["checks"]["n_nccl_within_rounding_bound"] = False
c_nf = next(c for c in C.candidates({**tables, "d7": C.d7_table(nf_, 2560, 2)}) if c["id"] == "D7_p5_host_allreduce")
check("N3-14 the NCCL reference failing on rank 1 -> the P5 candidate is not PASS", c_nf["correctness"] != "PASS" and C.eligibility(c_nf, ST_OK) != C.PROMOTION_READY)
# N3-17: per-size timed-region concurrency
shift = jcopy(fa)
for v in shift["2,3"]["groups"]["main_ab"]["rank0"]["b"]["latency"].values():
    v["t_mono"] = {k: [w[0] + 500.0, w[1] + 500.0] for k, w in v["t_mono"].items()}
t_sh = C.d7_table(shift, 2560, 2)
c_sh = next(c for c in C.candidates({**tables, "d7": t_sh}) if c["id"] == "D7_p5_host_allreduce")
check("N3-17 pairs whose timed regions never overlap -> CONCURRENCY_UNVERIFIED, P5 DIAGNOSTIC_ONLY (not ranked)", t_sh["concurrency"]["status"] == "CONCURRENCY_UNVERIFIED"
      and t_sh["concurrency"]["min_overlap"] == 0.0 and C.eligibility(c_sh, ST_OK) == C.DIAGNOSTIC_ONLY
      and not any(c["lever"] == "D7" for c in C.rank_candidates({**tables, "d7": t_sh}, status=ST_OK)["targets"][0]["ranked"]))
nots = jcopy(fa)
for v in nots["0,1"]["groups"]["main_ab"]["rank0"]["b"]["latency"].values():
    v.pop("t_mono")
check("N3-17 timestamps missing -> CONCURRENCY_UNVERIFIED (never inferred from the group window)", C.d7_table(nots, 2560, 2)["concurrency"]["status"] == "CONCURRENCY_UNVERIFIED")
# N3-18 (6) / N3-15: corrupted scale -> INT4_LAYOUT_OR_SCALE_MISMATCH before any sweep
r, jc = run_moe_fake(WORK / "moe_corrupt.json", faults="corrupt_scale")
check("N3-18(6) N3-15 corrupted scale -> exit 2, INT4_LAYOUT_OR_SCALE_MISMATCH, no unit / sweep ran", r.returncode == 2 and jc["status"] == "ERROR"
      and any("INT4_LAYOUT_OR_SCALE_MISMATCH" in e for e in jc["errors"]) and not jc["units_done"] and not jc["sweep"], json.dumps(jc["errors"])[:300])
check("N3-15 real path: the crosscheck raises before the Marlin prep / sweep (code order)", mb_src.index("dequant_gate(torch, C.ct_dequant_int4(pk, sc, G)")
      < mb_src.index("conv = process_marlin(") < mb_src.index("MARLIN_PREP_UNEXPECTED_OUTPUTS") < mb_src.index("run_units(res, save, units, do, left)\n    res[\"gpu_max"))
try:
    import torch
    MB = load("nb3_moe_bench_t", EXP / "nb3_moe_bench.py")
    a0 = torch.ones(2, 2)
    try:
        MB.dequant_gate(torch, a0, a0 * 1.5, "unit")
        dg = None
    except RuntimeError as ex:
        dg = str(ex)
    check("N3-15 dequant_gate raises INT4_LAYOUT_OR_SCALE_MISMATCH on any difference", dg and dg.startswith("INT4_LAYOUT_OR_SCALE_MISMATCH"))
    # N3-24: private-symbol preflight (signature + tiny INT4 parity)
    def _unpack_and_dequant_int4_gptq(w_int32, scale, qzeros, transpose_output, output_dtype=None):
        return MB.gptq_dequant_ref(torch, w_int32, scale)

    def _process_weights_marlin(layer, input_dtype, num_bits, pack_factor, group_size, w13_qweight, w2_qweight, w13_scales, w2_scales, w13_qzeros=None):
        return None

    def fused_marlin_moe(hidden_states, w1, w2, bias1, bias2, w1_scale, w2_scale, topk_weights, topk_ids, quant_type_id,
                         apply_router_weight_on_input=False, global_num_experts=-1, expert_map=None):
        return None
    good = {"_unpack_and_dequant_int4_gptq": _unpack_and_dequant_int4_gptq, "_process_weights_marlin": _process_weights_marlin,
            "fused_marlin_moe": fused_marlin_moe}
    pf = MB.preflight(torch, good)
    check("N3-24 preflight on Merlin-shaped symbols: ok, signatures recorded, tiny INT4 parity exact", pf["ok"] and pf["int4_parity"]["merlin_exact"]
          and set(pf["signatures"]) == set(good), json.dumps(pf)[:300])

    def swapped(scale, w_int32, qzeros, transpose_output, output_dtype=None):
        return MB.gptq_dequant_ref(torch, w_int32, scale)

    def nobias(w_int32, scale, qzeros, transpose_output, output_dtype=None):
        return MB.gptq_dequant_ref(torch, w_int32, scale) + 8.0 * 0.0 + 1.0

    def fused_old(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, quant_type_id):
        return None
    for label, patch, frag in (("positional order changed", {"_unpack_and_dequant_int4_gptq": swapped}, "leading params"),
                               ("dequant math changed", {"_unpack_and_dequant_int4_gptq": nobias}, "tiny INT4 parity failed"),
                               ("fused_marlin_moe keywords gone", {"fused_marlin_moe": fused_old}, "keywords missing")):
        pfb = MB.preflight(torch, {**good, **patch})
        check(f"N3-24 preflight catches: {label}", pfb["ok"] is False and any(frag in p for p in pfb["problems"]), json.dumps(pfb["problems"])[:200])
    check("N3-24 real path: preflight runs before the checkpoint is read (code order)",
          mb_src.index('res["preflight"] = preflight(torch') < mb_src.index("ck = Path(a.ckpt)") and "DEPENDENCY_PREFLIGHT_FAILED" in mb_src)
    check("REF_GATE (N3-3): finite + rel_fro + cos + max_abs_over_ref_max", set(MB.REF_GATE) == {"rel_fro", "cos", "max_abs_over_ref_max"}
          and MB.ref_ok({"finite": True, "rel_fro": 1e-3, "cos": 0.9999, "max_abs_over_ref_max": 0.2}) is False)
    # addendum 2: the draft-head op preflight (CPU: no vLLM -> UNTESTED path, not a D4 failure)
    GB = load("nb3_gemm_bench_t", EXP / "nb3_gemm_bench.py")
    dpf = GB.draft_head_preflight(torch)
    check("addendum 2: draft-head preflight on a box without the Marlin linear ops -> ok False, missing listed (marlin_gemm required, repack recorded)",
          dpf["ok"] is False and any("vllm" in m_ or "marlin_gemm" in m_ for m_ in dpf["missing"]) and "_C.gptq_marlin_repack" in dpf["ops"]
          and not any("gptq_marlin_repack" in m_ for m_ in dpf["missing"]), json.dumps(dpf)[:300])
    check("addendum 2: draft heads UNTESTED never add a failing D4 check (code path)", 'if res["draft_heads"].get("status") != "UNTESTED"' in gb_src
          and 'out["status"] = "UNTESTED"' in gb_src)
except ImportError:
    check("torch for the D5 gate / preflight tests", False)
# N3-18 (7) 4-ULP error -> FAIL: section 4b (N3-3 ULP gate)
# N3-18 (8) / N3-16 / N3-22: truncated D5 budget -> T16 persisted, PARTIAL_BUDGET, candidates not promoted
r, jb = run_moe_fake(WORK / "moe_budget.json", ["--phase", "core", "--budget-s", "64"])
check("N3-18(8) N3-16 truncated core budget: T16 baseline + sweep persisted first, PARTIAL_BUDGET, exit 0, humming (last) and adv skipped",
      r.returncode == 0 and jb["status"] == "PARTIAL_BUDGET" and jb["budget_stop"] is True and "independent" in jb["baseline"]["16"]
      and jb["sweep"]["16"]["best"] and ["humming", None] in jb["units_skipped"]
      and ["adv", 16] in jb["units_skipped"] and jb["units_skipped"][-1] == ["humming", None],
      json.dumps({k: jb[k] for k in ("status", "units_done", "units_skipped")})[:400])
vstat = C.verified_bench_status({"rc": r.returncode}, jb, True)[0]
rk_b = C.rank_candidates({**tables, "d5": C.d5_table(jb)}, status={**ST_OK, "D5": vstat})
cd5 = [c for c in rk_b["targets"][0]["diagnostic"] if c["id"] == "D5_marlin_moe_cfg"]
check("N3-22 fake clock: parent verified status PARTIAL_BUDGET -> D5 candidate PARTIAL_DIAGNOSTIC (kept, never ranked)", vstat == "PARTIAL_BUDGET"
      and cd5 and cd5[0]["eligibility"] == C.PARTIAL_DIAGNOSTIC and not any(c["lever"] == "D5" for c in rk_b["targets"][0]["ranked"]))
r, jb2 = run_moe_fake(WORK / "moe_budget2.json", ["--phase", "rest", "--budget-s", "62"])
check("N3-16 rest budget stop: T32 / T64 first, ncu never before the rest (fake: no ncu unit)", jb2["status"] == "PARTIAL_BUDGET"
      and jb2["units_done"][:3] == [["base", 32, "independent"], ["base", 32, "zipf"], ["sweep", 32]] if len(jb2["units_done"]) >= 3 else False,
      json.dumps(jb2["units_done"])[:200])
uc, ua = C.d5_units("core", replay=True), C.d5_units("rest", replay=True, ncu=True)
check("N3-16 priority (final audit): core = base16, replay, sweep16, T4 / T8, adversarial, C6-humming LAST; every core T in (16, 4, 8)",
      uc[:3] == [("base", 16, "independent"), ("replay", "core"), ("sweep", 16)]
      and uc[3:7] == [("base", 4, "independent"), ("sweep", 4), ("base", 8, "independent"), ("sweep", 8)] and uc[-1] == ("humming", None)
      and ("humming", None) not in ua
      and all(u[1] in (16, 4, 8) for u in uc if u[0] in ("base", "sweep", "adv")))
check("N3-16 priority: rest = T32 / T64 (+Zipf) first, hot / request next, ncu units LAST, EP1 spot before ncu",
      ua[:6] == [("base", 32, "independent"), ("base", 32, "zipf"), ("sweep", 32), ("base", 64, "independent"), ("base", 64, "zipf"), ("sweep", 64)]
      and all(u[0] == "ncu" for u in ua[-4:]) and ua.index(("ep1_spot", None)) < ua.index(("ncu", 8))
      and ua.index(("base", 16, "hot")) < ua.index(("base", 1, "independent")))
check("N3-8 EP rank 1 spot check in the rest plan (child at --ep-rank 1, T 4 / 16, 1 copy)", ("ep1_spot", None) in ua
      and '"--ep-rank", "1"' in mb_src and '"--spot", "4,16"' in mb_src)
# N3-7: first-call gate
check("N3-7 first-call gate T=1 and T=4 vs the reference before any unit (code order) + its check", "for T in (1, 4):" in mb_src
      and mb_src.index("FIRST_CALL_GATE_FAILED") < mb_src.index("run_units(res, save, units, do, left)\n    res[\"gpu_max")
      and jb["checks"].get("first_call_gate") is True)
# N3-18 (9) missing SM89 CuTe -> D4 diagnostics only (fake bench JSON level; orchestration in section 8)
r = subprocess.run([sys.executable, str(EXP / "nb3_gemm_bench.py"), "--fake", "--out", str(WORK / "gemm_unsup.json")], capture_output=True, text=True,
                   timeout=120, env={**BASE_ENV, "NB3_FAKE_FAULTS": "d4_unsupported"})
ju = json.loads((WORK / "gemm_unsup.json").read_text())
t4u = C.d4_table(ju)
check("N3-18(9) missing sm_89 CuTe -> exit 0, status UNSUPPORTED_SM89, cuBLAS kept, no D4 candidate, scorecard DIAGNOSTIC_ONLY",
      r.returncode == 0 and ju["status"] == "UNSUPPORTED_SM89" and all(x["cublas_us"] for x in t4u["rows"]) and all(x["best_us"] is None for x in t4u["rows"])
      and next(x for x in C.scorecard({**tables, "d4": t4u}, {**ST_OK, "D4": "UNSUPPORTED_SM89"})["rows"] if x["lever"] == "D4")["promotion"] == C.DIAGNOSTIC_ONLY)
# N3-21: D4 dispatch must be True
for disp, want_c in ((True, "PASS"), (None, "UNTESTED_DISPATCH"), (False, "FAIL_DISPATCH")):
    fgd = jcopy(fg)
    for s_ in fgd["shapes"].values():
        for r_ in s_["M"].values():
            if disp is None:
                r_.pop("dispatch", None)
            else:
                r_["dispatch"] = {"ok": disp}
    cs4 = [c for c in C.candidates({**tables, "d4": C.d4_table(fgd)}) if c["lever"] == "D4"]
    els = {C.eligibility(c, ST_OK) for c in cs4}
    check(f"N3-21 D4 dispatch {disp} -> correctness {want_c}" + (", promotable" if disp else ", never ranked (BLOCKED_CORRECTNESS)"),
          cs4 and all(c["correctness"] == want_c for c in cs4)
          and ((C.PROMOTION_READY in els and els <= {C.PROMOTION_READY, "NO_GAIN"}) if disp is True else els == {C.BLOCKED_CORRECTNESS}),
          str({c["correctness"] for c in cs4}) + " " + str(els))
# N3-6: the two saved-ms columns + the additivity note
check("N3-6 scorecard: microbench_saved_ms + engine_confirmed_saved_ms null + 'not additive under PP2 overlap'",
      all("microbench_saved_ms" in x and x["engine_confirmed_saved_ms"] is None and "not additive under PP2 overlap" in x["additivity"] for x in sc_all["rows"])
      and all(c["engine_confirmed_saved_ms"] is None and c["microbench_saved_ms"] == c["saved_ms_per_tick"] for c in C.candidates(tables)))
# N3-9: the NB4 adoption rule recorded
check("N3-9 NB4 adoption rule recorded in the scorecard (2x full-engine A/B, 40k / 90k rows, P5 1000+ replays + both-pair contention)",
      all(k in sc_all["adoption_rule"] for k in ("full-engine A/B", "40k / 90k", "1000+", "both-pair contention")))
check("promotion rule documented in the scorecard", "PASS_BOUND_ONLY is diagnostic" in sc_all["promotion_rule"])

# ------------------------------------------------------------------ 10b. C6-humming structural arms (addendum, research only)
HMm = load("nb3_humming_t", EXP / "nb3_humming.py")
gr = {(a_, r_): HMm.gate_reading(a_, r_) for a_, r_ in (("w3g128", 0.79), ("w3g128", 0.95), ("w2g128", 0.59), ("w2g128", 0.62), ("w2g128", 1.35), ("w4g128_ct", 1.02))}
check("C6 gate readings: W3 <= 0.8 PASS / 0.95 FAIL but capacity NET_POSITIVE_POSSIBLE; W2 <= 0.6 PASS / 0.62 FAIL; 1.35 TOO_SLOW; W4 = CONTROL",
      gr[("w3g128", 0.79)]["strict"] == "PASS" and gr[("w3g128", 0.95)] ["strict"] == "FAIL" and gr[("w3g128", 0.95)]["capacity"] == "NET_POSITIVE_POSSIBLE"
      and gr[("w2g128", 0.59)]["strict"] == "PASS" and gr[("w2g128", 0.62)]["strict"] == "FAIL" and gr[("w2g128", 1.35)]["capacity"] == "TOO_SLOW"
      and gr[("w4g128_ct", 1.02)]["strict"] == "CONTROL", json.dumps({f"{k[0]}@{k[1]}": v["strict"] + "/" + v["capacity"] for k, v in gr.items()}))
check("C6 arms: W4 control on the checkpoint's own INT4 (ct), W3 / W2 g128 + W3 g64 synthetic requant, T 16 / 8 / 32 / 64",
      [a_[0] for a_ in HMm.ARMS] == ["w4g128_ct", "w3g128", "w2g128", "w3g64"] and HMm.ARMS[0][3] == "ct"
      and all(a_[3] == "rtn" for a_ in HMm.ARMS[1:]) and HMm.T_LIST[0] == 16 and set(HMm.T_LIST) == {8, 16, 32, 64} and "SYNTHETIC REQUANT" in HMm.LABEL)
ht = C.humming_table(HMm.fake(0))
check("C6 table: 16 rows, graph and eager kept apart, ratios vs Marlin W4, own-weight reference, gate per row",
      len(ht["rows"]) == 16 and all(r_["graph_us"] and r_["eager_us"] and r_["graph_us"] != r_["eager_us"] and r_["ratio_vs_marlin_w4_graph"]
                                    and r_["gate"] and r_["ref_ok_own_weights"] is True for r_ in ht["rows"]))
r, jh = run_moe_fake(WORK / "moe_hum_unavail.json", ["--phase", "core"], faults="humming_unavailable")
check("C6 humming not importable -> UNAVAILABLE recorded, D5 core continues (exit 0, DONE, T4 / T8 measured), no humming check",
      r.returncode == 0 and jh["status"] == "DONE" and jh["humming"]["status"] == "UNAVAILABLE" and "4" in jh["baseline"] and "8" in jh["sweep"]
      and not any("humm" in k for k in jh["checks"]), json.dumps(jh["checks"]))
sch = C.scorecard({**tables, "d5": C.d5_table(jh)}, ST_OK)
check("C6 scorecard row UNAVAILABLE, score 0; never a NB4 candidate", next(x for x in sch["rows"] if x["lever"] == "C6-humming")["promotion"] == "UNAVAILABLE"
      and not any("humm" in c["id"].lower() for tg in rk["targets"] for c in tg["ranked"] + tg["blocked"] + tg["diagnostic"] + tg["hypotheses"]))
hum_src = (EXP / "nb3_humming.py").read_text(encoding="utf-8")
_mk = hum_src.split("    def make(", 1)[1].split("def run(", 1)[0]
check("C6 final audit: humming token alignment (moe_align) runs INSIDE the timed forward fn, like fused_marlin_moe; meta says so",
      "self.moe_align(" not in _mk.split("def fn():", 1)[0] and _mk.split("def fn():", 1)[1].count("self.moe_align(") == 2
      and "\"alignment_in_timed_forward\": True" in _mk and "HUMMING_INDEXED_EP_RESEARCH" in _mk)
check("C6 real path: Merlin INDEXED humming MoE (moe_align + w13 top_k + SiLU-and-mul + w2 top_k 1 + masked weighted sum), reference = own weights",
      all(k in hum_src for k in ("gemm_type\": \"indexed\"", "GemmType.INDEXED", "top_k=topk", "top_k=1", "torch.where(local", "dequantize_weight(",
                                  "arm.expert_fp32", "first_call_sec", "transform_sec")) and "def availability" in hum_src)
check("C6 placement: humming runs inside the core bench right after the T16 sweep, budget-bounded (reserve 120 s)", C.D5_RESERVE_S["humming"] == 120
      and "humming" in mb_src and "HM.run(torch, C, ctx" in (EXP / "nb3_moe_bench.py").read_text(encoding="utf-8"))

# ------------------------------------------------------------------ 11. N3-25 integration: every fault injected together
FAULTS = "d5_bench_error,graph_error,d7_timeout,missing_rank1,d4_check_failed,missing_dispatch"
t = time.time()
r = subprocess.run([sys.executable, str(HERE / "nb3_code_export.py")], capture_output=True, text=True, timeout=1800,
                   env=env_for("faults", {"NB3_FAKE_FAULTS": FAULTS, "NB3_D7_PAIRS_SEC": "20", "NB3_D7_SOLO_SEC": "20"}))
outf = WORK / "faults" / "out"
(WORK / "faults_stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
rf = json.loads((outf / "nb3_results.json").read_text()) if (outf / "nb3_results.json").is_file() else {}
scf = json.loads((outf / "lever_scorecard.json").read_text()) if (outf / "lever_scorecard.json").is_file() else {"rows": []}
tbf = json.loads((outf / "nb3_tables.json").read_text()) if (outf / "nb3_tables.json").is_file() else {}
promo = {x["lever"]: x["promotion"] for x in scf["rows"]}
check("N3-25 fault run: rc 0 (the notebook survives every bench failure)", r.returncode == 0, r.stderr[-600:])
check("N3-25 verified statuses: D5 BENCH_ERROR, D7 TIMEOUT, D4 CHECK_FAILED (D5 rest ran too: no device fault, no abort)",
      (rf.get("bench_status") or {}).get("D5") == "BENCH_ERROR" and rf["bench_status"].get("D7") == "TIMEOUT" and rf["bench_status"].get("D4") == "CHECK_FAILED"
      and rf["bench_status"].get("D5_rest") == "BENCH_ERROR", json.dumps(rf.get("bench_status")))
check("N3-25 nb4_top empty, nothing ranked at any target", rf.get("nb4_top") == [] and all(not tg["ranked"] for tg in tbf["nb4_candidates"]["targets"]),
      json.dumps(rf.get("nb4_top")))
check("N3-25 scorecard: D5 / D7 / D4 BLOCKED_BENCH_FAILURE, PP2 DIAGNOSTIC_ONLY", promo.get("D5") == promo.get("D7") == promo.get("D4") == C.BLOCKED_BENCH_FAILURE
      and promo.get("PP2") == C.DIAGNOSTIC_ONLY, json.dumps(promo))
check("N3-25 every score 0 and every projected gain 0", scf["rows"] and all(x["score"] == 0 and (x["projected_run_tok_s_gain"]["run_tok_s_gain_headline"] or 0) == 0
                                                                            and x["microbench_saved_ms"] == 0 for x in scf["rows"]))
check("N3-25 raw data kept: D5 rows (incl. the graph-error row as INCONCLUSIVE), D7 rows, D4 rows, diagnostic_best set",
      tbf["tables"]["d5"]["baseline"] and any(x["T"] == 16 and x["mode"] == "independent" and x["graph_status"] == "INCONCLUSIVE_CAPTURE_FAILED"
                                              for x in tbf["tables"]["d5"]["baseline"]) and tbf["tables"]["d7"]["rows"] and tbf["tables"]["d4"]["rows"]
      and next(x for x in scf["rows"] if x["lever"] == "D7")["diagnostic_best"])
check("N3-25 injected D7 rank-1 loss -> arms INCONCLUSIVE_MISSING_RANK; D4 dispatch missing -> UNTESTED_DISPATCH",
      set(tbf["tables"]["d7"]["arms"].values()) == {"INCONCLUSIVE_MISSING_RANK"} and
      all(c["correctness"] == "UNTESTED_DISPATCH" for tg in tbf["nb4_candidates"]["targets"] for c in tg["blocked"] if c["lever"] == "D4"),
      json.dumps(tbf["tables"]["d7"]["arms"]))
subf = json.loads((outf / "submission.json").read_text()) if (outf / "submission.json").is_file() else {}
check("N3-25 fallback submission valid: every test task, every test input, 2 attempts", set(subf) == set(TEST_TASKS) and all(
    len(subf[k]) == len(v["test"]) and all(set(a_) == {"attempt_1", "attempt_2"} for a_ in subf[k]) for k, v in TEST_TASKS.items()))
check("N3-25 fault run finished in reasonable time (two D7 timeouts)", time.time() - t < 600, f"{round(time.time() - t)} s")

nf = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\nNB3 unit tests: {len(RESULTS) - nf}/{len(RESULTS)} PASS" + (f", {nf} FAIL" if nf else ""))
sys.exit(1 if nf else 0)
