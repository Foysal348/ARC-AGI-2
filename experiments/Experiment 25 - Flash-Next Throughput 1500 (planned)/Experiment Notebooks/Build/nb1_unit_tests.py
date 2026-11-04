"""E25 NB1 fault-injection unit tests (pattern: Experiment 13 rc2_unit_tests.py). Runs the exported notebook code with
RC2_UNIT=1 (definitions + A0 + prompts, no boots) against a fake input tree, then injects each fault. Fixtures are REAL
recorded data wherever possible: the RC2 FP8 / BF16 server logs, RC2's window KV-usage readings, RC2's recorded oracle
prompt hashes, Merlin ad42e37's qsa.py.

Usage: python nb1_unit_tests.py <fake input dir> <work dir>
"""
import hashlib
import json
import os
import runpy
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INP, WORK = Path(sys.argv[1]), Path(sys.argv[2])
# fixture paths (env overrides: the Kaggle CPU kit carries the same files in its own directory)
RC2 = Path(os.environ.get("NB1_TEST_RC2") or r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2")
RC2_EXPORT = Path(os.environ.get("NB1_TEST_RC2_EXPORT") or (r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on "
                                                            r"4xL4 (planned)/Experiment Notebooks/Build/rc2_code_export.py"))
MERLIN_QSA = Path(os.environ.get("NB1_TEST_MERLIN_QSA") or r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/models/qwen4_exp/nvidia/qsa.py")
REL = Path("vllm/models/qwen4_exp/nvidia/qsa.py")
shutil.rmtree(WORK, ignore_errors=True)
for x in ("out", "tmp", "shm"):
    (WORK / x).mkdir(parents=True)
os.environ.update(RC2_UNIT="1", E23_FAKE_SERVER=str(HERE / "nb1_fake_server.py"), E23_OUT=str(WORK / "out"), E23_TMP=str(WORK / "tmp"),
                  E23_INPUT=str(INP), RC2_SUBMISSION=str(WORK / "out/submission.json"), RC2_SHM=str(WORK / "shm"), RC2_BUDGET_SEC="3600")
os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
EXPORT = HERE / "nb1_code_export.py"
ns = runpy.run_path(str(EXPORT), run_name="nb1")["nb1_main"].__globals__      # live module globals
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


L = ns["L"]
REAL_PIN_RULE = ns["pin_rule"]          # R2 audit: the notebook's own pin rule (section 8 replaces ns["pin_rule"])


def TF(F, samples=4, res=0.0):
    """R2 audit: a fit record with fit_alloc's fields (trusted when samples >= 3 and max residual <= 2 blocks)."""
    return {"F": F, "a": 1.0, "samples": samples, "max_abs_residual_blocks": res}
FP8_LOG = (RC2 / "logs/server_merlin_fp8.log").read_text(encoding="utf-8", errors="replace")
BF16_LOG = (RC2 / "logs/server_merlin_bf16.log").read_text(encoding="utf-8", errors="replace")

# ---- 0. reuse: the RC2 cells are RC2's Kaggle-run code; every changed hunk carries an "E25 NB1" marker
import difflib
rc2_txt = RC2_EXPORT.read_text(encoding="utf-8").replace("\r\n", "\n")
nb1_txt = EXPORT.read_text(encoding="utf-8")
nb1_cells = nb1_txt.split("\n\n# ==================================================================== CELL 6")[0]   # CELL1-5
groups = list(difflib.SequenceMatcher(None, rc2_txt.splitlines(), nb1_cells.splitlines(), autojunk=False).get_grouped_opcodes(0))
rc2_cut = rc2_txt.index('    if not RERUN and not os.environ.get("RC2_UNIT"):\n        m = R["sections"]["A0"]["merge"]')
unmarked = []
for g in groups:
    i1, i2, j1, j2 = g[0][1], g[-1][2], g[0][3], g[-1][4]
    new = nb1_cells.splitlines()[j1:j2]
    old_start = sum(len(l) + 1 for l in rc2_txt.splitlines()[:i1])
    if old_start >= rc2_cut:          # RC2's own run block and report cell: not reused (replaced by NB1's)
        continue
    if not any("E25 NB1" in l for l in new) and new != ["    save()", ""] and new:
        unmarked.append(new[:3])
check("reuse: every change to RC2's cells 1-5 is a marked 'E25 NB1' hunk", not unmarked, json.dumps(unmarked)[:400])

# ---- 1. the #60490 patch: strict applier + apply_60490 (good / already / wrong pre-sha / bad context)
src = MERLIN_QSA.read_bytes().decode("utf-8")
out = ns["apply_unified"](src, ns["P60490"])
check("patch: Merlin ad42e37 qsa.py -> the pinned post sha256", hashlib.sha256(out.encode()).hexdigest() == ns["P60490_POST"])
check("patch: the result contains the marker and the #60490 code", ns["P60490_MARKER"] in out and "get_current_vllm_config_or_none()" in out
      and "_TOPK_WORKSPACE_BYTES" in out)
try:
    ns["apply_unified"](src.replace("from vllm.config import VllmConfig\n", "from vllm.config import VllmConfig  # x\n"), ns["P60490"])
    bad = False
except ValueError as e:
    bad = "context mismatch" in str(e)
check("patch: a changed context line -> ValueError (no fuzz)", bad)
try:
    ns["apply_unified"](src, ns["P60490"].replace("@@ -782,6 +785,31 @@", "@@ -782,9 +785,31 @@"))
    bad = False
except ValueError as e:
    bad = "header says" in str(e)
check("patch: a wrong hunk length -> ValueError", bad)
tree = WORK / "tree"
(tree / REL).parent.mkdir(parents=True)
(tree / REL).write_bytes(MERLIN_QSA.read_bytes())
os.environ["NB1_FAKE_TREE"] = str(tree)
r1 = ns["apply_60490"]()
r2 = ns["apply_60490"]()
check("patch: apply_60490 applied, then already_applied (idempotent), worker extension written + compiled",
      r1["applied"] == "applied" and r2["applied"] == "already_applied" and r1["ok"] and (tree / "nb1_worker_ext.py").is_file()
      and r1["py_compile"]["rc"] == 0, json.dumps({k: r1.get(k) for k in ("applied", "py_compile")})[:200])
(tree / REL).write_bytes(MERLIN_QSA.read_bytes() + b"\n# drift\n")
try:
    ns["apply_60490"]()
    bad = False
except RuntimeError as e:
    bad = "PRE_SHA_MISMATCH" in str(e)
check("patch: a tree that is not Merlin ad42e37 -> PRE_SHA_MISMATCH (fail closed)", bad)
os.environ.pop("NB1_FAKE_TREE")
try:
    ns["apply_60490"]()
    bad = False
except RuntimeError as e:
    bad = "NO_TREE" in str(e)
check("patch: no tree -> raises", bad)

# ---- 2. the worker extension never raises
ext_ns = {}
exec(ns["NB1_EXT_SRC"], ext_ns)


class W(ext_ns["NB1WorkerExt"]):
    pass


info = W().nb1_cache_info()
check("worker ext: on a bare object every part reports its error, no exception", isinstance(info, dict)
      and "cache_tensors_error" in info and "kv_cache_specs_error" in info and json.dumps(info))


class Spec:
    block_size, page_size_bytes, page_size_padded = 4096, 16515072, 16515072


class Mod:
    kv_cache = [[]]


class W2(ext_ns["NB1WorkerExt"]):
    class vllm_config:
        class compilation_config:
            static_forward_context = {"a": Mod()}

    class cache_config:
        num_gpu_blocks, block_size, mamba_ssm_cache_dtype = 150, 4096, "auto"

    def get_kv_cache_spec(self):
        return {"layer.0": Spec()}


info = W2().nb1_cache_info()
check("worker ext: specs + cache config read, JSON-serialisable", info.get("num_gpu_blocks") == 150 and info["kv_cache_specs"][0]["page_size_bytes"] == 16515072
      and json.dumps(info), json.dumps(info)[:200])

rt = W().nb1_routing("start", "200")
check("worker ext: routing capture without vLLM importable -> error recorded, no exception", isinstance(rt, dict) and "error" in rt, json.dumps(rt)[:160])
rt = W().nb1_routing("stop", "")
check("worker ext: routing stop with nothing recorded -> counters, no file, no exception", rt.get("passes") == 0 and "path" not in rt, json.dumps(rt)[:160])

# critical path vs summed kernel time on a synthetic trace (two overlapping streams inside one micro-batch)
ev = [{"ph": "X", "cat": "user_annotation", "name": "execute_context_0(0)_generation_4(16)", "ts": 0, "dur": 30000, "tid": 1}]
for i, (ts, dur) in enumerate(((100, 10000), (5100, 10000), (20000, 5000))):
    ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 50 + i, "dur": 5, "tid": 1, "args": {"correlation": i}})
    ev.append({"ph": "X", "cat": "kernel", "name": "fused_marlin_moe_kernel", "ts": ts, "dur": dur, "tid": 7, "args": {"correlation": i}})
ns["CAPTURE_SIZES"] = [1, 2, 4, 8, 16, 32]
mb = ns["microbatch_costs"](ev, [(ev[0], 0, 4, 16)])
x = mb["examples"][0]
check("micro-batch costs: summed kernel 25 ms, busy union 20 ms, wall span 24.9 ms, capture size 16, MoE category",
      x["kernel_ms"] == {"moe_marlin": 25.0} and x["gpu_busy_union_ms"] == 20.0 and x["gpu_wall_ms"] == 24.9 and x["resolved_capture_size"] == 16
      and x["eager_kernel_launches"] == 3, json.dumps(x))

# routing-id file: the NB3 schema (addendum 9)
import numpy as np
parts = {}
for rank in (0, 1, 2):
    rows, offs, o_ = [], [], 0
    for ps in range(4):                       # 4 passes = 2 steps x 2 micro-batches
        for lay in ((3, 7) if rank < 2 else (27, 31)):
            n = 8
            rows.append((lay, ps, n, np.arange(n * 10, dtype=np.int16).reshape(n, 10) % 512))
    for r in rows:
        offs.append(o_)
        o_ += r[2]
    pth = WORK / f"part_rank{rank}.npz"
    np.savez_compressed(pth, ids=np.concatenate([r[3] for r in rows]), layer=np.array([r[0] for r in rows]),
                        pass_index=np.array([r[1] for r in rows]), mb_size=np.array([r[2] for r in rows]), offset=np.array(offs),
                        rank=np.array([rank]))
    parts[rank] = str(pth)
ns["ROUTING_FILE"] = WORK / "out/nb1_routing_ids.npz"
mr = ns["merge_routing"]({"results": [{"rank": r, "path": parts[r]} for r in (0, 1, 2)] + [{"rank": 3, "error": "x"}]})
z = np.load(WORK / "out/nb1_routing_ids.npz")
L_ = len(z["layer"])
check("routing file: exactly ids / layer / stage / mb_size / step / offset / mb_index; ids [total, 10] in [0, 512); TP rank 1 dropped",
      sorted(z.files) == sorted(["ids", "layer", "stage", "mb_size", "step", "offset", "mb_index"]) and z["ids"].shape == (int(z["mb_size"].sum()), 10)
      and all(len(z[k]) == L_ for k in ("layer", "stage", "mb_size", "step", "offset", "mb_index")) and L_ == 16
      and set(z["stage"].tolist()) == {0, 1} and z["ids"].min() >= 0 and z["ids"].max() < 512 and mr["written"], json.dumps(mr)[:200])
check("routing file: step = pass // 2, mb_index = pass % 2 (two PP2 micro-batches per step), offsets contiguous",
      sorted(set(zip(z["step"].tolist(), z["mb_index"].tolist()))) == [(0, 0), (0, 1), (1, 0), (1, 1)]
      and z["offset"].tolist() == [8 * i for i in range(16)])
check("routing file: nothing captured -> not written, the reason recorded", ns["merge_routing"]({"results": [{"rank": 0, "error": "no vllm"}]})["written"] is False)

# ---- 3. log parsers on RC2's real logs
caps = ns["server_caps"](FP8_LOG)
check("caps (RC2 FP8 log): seqs 8, block 4096, chunk 1024, captures to 32 cover 8 x 4", caps["max_num_seqs"] == 8 and caps["block_size"] == 4096
      and caps["max_num_batched_tokens"] == 1024 and caps["capture_sizes"][-1] == 32 and caps["captures_cover_seqs_x_1pk"] is True
      and caps["kv_cache_dtype"] == "fp8_e4m3" and caps["source"] == "log", json.dumps(caps)[:300])
pf = ns["pool_facts"](FP8_LOG)
check("pool (RC2 FP8 log): 513,940 tokens, 3.92x, padding 21.01 % -> minimum block 3385", pf["kv_tokens"] == 513940 and pf["max_concurrency"] == 3.92
      and pf["mamba_pad_pct"] == 21.01 and pf["min_block_for_mamba"] == 3385, json.dumps({k: pf[k] for k in ("kv_tokens", "max_concurrency", "mamba_pad_pct", "min_block_for_mamba")}))
check("pool (RC2 FP8 log): per-rank model load GiB / sec (rank 3 = 873 s)", pf["rank_load_gib"] == {0: 17.33, 1: 17.33, 2: 16.87, 3: 16.87}
      and round(pf["rank_load_sec"][3]) == 873 and pf["rank_graph_gib"] == {0: 0.33, 1: 0.33, 2: 0.43, 3: 0.43}, json.dumps(pf["rank_load_sec"]))
check("pool (RC2 FP8 log): MTP default-scale lines, static-scale lines, NFS prefetch / cgroup lines, shm leaks",
      len(pf["default_scale_lines"]) == 2 and len(pf["static_scale_lines"]) >= 4 and any("Skipping auto-prefetch" in l for l in pf["host_ram_lines"])
      and len(pf["shm_leak_lines"]) == 2)
check("pool (RC2 BF16 log): padding 142.03 %", ns["pool_facts"](BF16_LOG)["mamba_pad_pct"] == 142.03)
gf = ns["graph_facts"](FP8_LOG)
check("graphs (RC2 FP8 log): FULL 6 + PIECEWISE 10 captured, 5 JIT-during-inference kernels", ("FULL", 6) in gf["captured"] and ("PIECEWISE", 10) in gf["captured"]
      and gf["jit"] == ["_fill_num_accepted_kernel", "_causal_conv1d_update_kernel", "_gdn_recoverssm_verify_kernel", "_ple_conv_kernel",
                        "_ple_conv_writeback_kernel"], json.dumps(gf["jit"]))
nc = ns["nccl_transport"](FP8_LOG)
check("nccl (RC2 FP8 log): no debug lines, PYNCCL groups read", nc["debug_on"] is False and len(nc["pynccl_groups"]) >= 3)
fake_nccl = "\n".join(f"x [0] NCCL INFO Channel 0{i}/0 : 0[0] -> 1[1] via SHM/direct/direct" for i in range(2)) + "\nx NCCL INFO Using network Socket"
check("nccl: SHM channels and the network line parsed", ns["nccl_transport"](fake_nccl)["channel_via"] == {"SHM/direct/direct": 2})

# ---- 4. pool blocks from RC2's real usage readings; the cold-sequence fit
R = json.loads((RC2 / "rc2_results.json").read_text())
us = [v.get("kv_usage_max") for v in {**R["sections"]["A3"]["merlin_fp8"]["steady"], **R["sections"]["A3"]["merlin_fp8"]["census_long"]}.values()]
us += [0.7718120805369127, 0.30201342281879195]
check("pool blocks: RC2 FP8 usage readings -> 149", ns["pool_blocks_from_usage"](us) == 149, str(us[:4]))
check("pool blocks: BF16 multiples of 1/78 -> 78", ns["pool_blocks_from_usage"]([20 / 78, 45 / 78, 61 / 78]) == 78)
check("pool blocks: one reading only -> None (never guessed)", ns["pool_blocks_from_usage"]([0.5]) is None)


def seq(F, P=149, idle=0, band_len=8192, counts=(1, 2, 4, 8), retained=17, resident=True):
    snaps = []
    for c in counts:
        toks = [band_len + 300] * c
        used = idle + sum(-(-t // 4096) + F for t in toks)
        snaps.append({"requests": c, "usage": used / P, "running": c if resident else c - 1, "waiting": 0 if resident else 1, "tokens": toks})
    return {"status": "MEASURED", "idle_usage": [idle / P] * 3, "snapshots": snaps, "released_usage": (idle + retained) / P, "after_reset_usage": 0.0}


s1 = [seq(6), seq(6, band_len=40000, counts=(1, 2, 4))]
fit = ns["fit_alloc"](s1, 149, 4096)
check("F fit: synthetic F 6 at 8k and 40k -> a 1.0, F 6.0, residual 0", fit["F"] == 6.0 and fit["a"] == 1.0 and fit["max_abs_residual_blocks"] == 0, json.dumps(fit))
check("R2 audit N1-6: fit_alloc exposes a, samples, max residual and the residual per snapshot",
      fit["samples"] == 7 and fit["residuals_blocks"] == [0.0] * 7 and fit["singular"] is False and "a" in fit, json.dumps(fit))
fit2 = ns["fit_alloc"]([seq(6, idle=20)], 149, 4096)
check("F fit: retained idle blocks are subtracted (F stays 6)", fit2["F_at_a1"] == 6.0, json.dumps(fit2))
check("F fit: retained-after-release and after-reset blocks recorded", s1[0]["retained_after_release_blocks"] == 17 and s1[0]["after_reset_blocks"] == 0)
fit3 = ns["fit_alloc"]([seq(6, resident=False)], 149, 4096)
check("F fit: snapshots with a waiting request never enter the fit", fit3 is None)
ns["PREFILL"].update(tok_s=4400.0)
check("n_fit: P 149, F 6, 90-s window growth: 8k -> 14 (cap seqs 8 -> 8); 40k -> 7", ns["n_fit"](149, 6, 4096, "8k") == 14
      and ns["n_fit"](149, 6, 4096, "8k", seqs=8) == 8 and ns["n_fit"](149, 6, 4096, "40k") == 7)
check("resident_at 70k: P 149 F 6 -> 6 streams; F 7 -> 5", ns["resident_at"]({"pool_blocks": 149, "F": 6, "seqs": 16}) == 6
      and ns["resident_at"]({"pool_blocks": 149, "F": 7, "seqs": 16}) == 5)

# ---- 5. RC2's recorded oracle rows: identical prompts here (real tokenizer + template + eval file)
check("RC2 recorded oracle rows: prompt sha256 of ORACLE_P + LONG_P match RC2's (BF16 and FP8)",
      ns["NB1"]["rc2_oracle_prompts_match"] == {"bf16": True, "fp8": True}, json.dumps(ns["NB1"]["rc2_oracle_prompts_match"]))
check("RC2 rows are not used as references under the fake server", "rc2_fp8" not in ns["ROWS"])

# ---- 6. gates
qg = ns["quality_gate"]
base = {"sampled": {"ok": True}, "probes": {"ok": True}, "oracle": {"errors": [None] * 4, "garbage": [None] * 4, "complete": True,
                                                                    "vs_rc2_fp8": {"status": "EXACT"}}}
check("quality gate: EXACT complete -> PASS", qg(base, "rc2_fp8") == "PASS")
check("quality gate: FAIL_HIGH_MARGIN -> FAIL", qg(dict(base, oracle=dict(base["oracle"], vs_rc2_fp8={"status": "FAIL_HIGH_MARGIN"})), "rc2_fp8") == "FAIL")
check("quality gate: no reference -> INCONCLUSIVE", qg(dict(base, oracle=dict(base["oracle"], vs_rc2_fp8={"status": "INCONCLUSIVE_NO_ORACLE"})), "rc2_fp8") == "INCONCLUSIVE")
check("quality gate: sampled failure -> FAIL_SAMPLED", qg(dict(base, sampled={"ok": False}), "rc2_fp8") == "FAIL_SAMPLED")
bo = ns["boot_ok"]
check("boot_ok: needs DONE/TF_OOM + CLEAN + PASS", bo({"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS"})
      and not bo({"status": "PIN_FREE_BELOW_0P5_GIB", "correctness": "CLEAN", "quality_gate": "PASS"})
      and not bo({"status": "DONE", "correctness": "CLEAN", "quality_gate": "INCONCLUSIVE"}) and not bo(None))
TFO = {"status": "TF_OOM", "correctness": "CLEAN", "quality_gate": "PASS"}
check("R2 audit N1-7: TF_OOM -> boot_ok still True (dependents run) but headroom UNSAFE_HEADROOM (also after a post-processing "
      "error); a DONE boot has no flag; ensure_core records it", bo(TFO) and ns["headroom_flag"](TFO) == "UNSAFE_HEADROOM"
      and ns["headroom_flag"]({"status": "DONE", "status_before_postprocess": None}) is None
      and ns["headroom_flag"]({"status": "POSTPROCESS_ERROR", "status_before_postprocess": "TF_OOM"}) == "UNSAFE_HEADROOM"
      and ns["ensure_core"](dict(TFO))["headroom"] == "UNSAFE_HEADROOM" and "headroom" not in ns["ensure_core"](dict(TFO, status="DONE")))
co = ns["_corr"]
CB = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS"}
corr_cases = {"ERROR": co(dict(CB, status="ERROR")), "BENCH_ERROR": co(dict(CB, status="BENCH_ERROR")),
              "POSTPROCESS_ERROR": co(dict(CB, status="POSTPROCESS_ERROR")), "FAIL_X": co(dict(CB, status="FAIL_SAMPLED")),
              "NO_GO_QUALITY": co(dict(CB, status="NO_GO_QUALITY")), "TF_OOM": co(dict(CB, status="TF_OOM")),
              "SKIPPED": co(dict(CB, status="SKIPPED_BUDGET")), "PARTIAL": co(dict(CB, status="PARTIAL_HARD_STOP")),
              "LAUNCH_FAILED": co(dict(CB, status="LAUNCH_FAILED")), "NOT_RUN": co(dict(CB, status="NOT_RUN")),
              "DONE+postprocess_error": co(dict(CB, postprocess_error="KeyError: x")),
              "DONE+correctness BF16_DIFF": co(dict(CB, correctness="BF16_DIFF")), "DONE+gate INCONCLUSIVE": co(dict(CB, quality_gate="INCONCLUSIVE")),
              "DONE+gate FAIL": co(dict(CB, quality_gate="FAIL")), "DONE clean": co(CB), "None": co(None), "{}": co({})}
want_corr = {"ERROR": "FAIL", "BENCH_ERROR": "FAIL", "POSTPROCESS_ERROR": "FAIL", "FAIL_X": "FAIL", "NO_GO_QUALITY": "FAIL",
             "TF_OOM": "UNTESTED", "SKIPPED": "UNTESTED", "PARTIAL": "UNTESTED", "LAUNCH_FAILED": "UNTESTED", "NOT_RUN": "UNTESTED",
             "DONE+postprocess_error": "INCOMPLETE_DIAGNOSTIC", "DONE+correctness BF16_DIFF": "UNTESTED",
             "DONE+gate INCONCLUSIVE": "UNTESTED", "DONE+gate FAIL": "FAIL", "DONE clean": "DIAGNOSTIC_PASS", "None": "UNTESTED", "{}": "UNTESTED"}
check("R2 audit N1-1: _corr per status: ERROR / BENCH_ERROR / POSTPROCESS_ERROR / FAIL* / NO_GO* -> FAIL; not DONE -> UNTESTED; "
      "postprocess_error -> INCOMPLETE_DIAGNOSTIC; not CLEAN / gate not PASS -> UNTESTED; else DIAGNOSTIC_PASS (never PASS)",
      corr_cases == want_corr and "PASS" not in corr_cases.values(), json.dumps({k: v for k, v in corr_cases.items() if want_corr[k] != v}))
check("R2 audit N1-1: a lever check inside its boot's label: EXACT on a clean boot -> DIAGNOSTIC_PASS; EXACT on a POSTPROCESS_ERROR "
      "boot -> FAIL; EXACT on a DONE + postprocess_error boot -> INCOMPLETE_DIAGNOSTIC; FAIL_HIGH_MARGIN -> FAIL",
      ns["_corr_and"](CB, "EXACT") == "DIAGNOSTIC_PASS" and ns["_corr_and"](dict(CB, status="POSTPROCESS_ERROR"), "EXACT") == "FAIL"
      and ns["_corr_and"](dict(CB, postprocess_error="x"), "EXACT") == "INCOMPLETE_DIAGNOSTIC"
      and ns["_corr_and"](CB, "FAIL_HIGH_MARGIN") == "FAIL" and ns["_corr_and"](CB, None) == "UNTESTED")

# ---- 7. the B3 pin ladder
ns["FAKE"] = None
ns["R"]["sections"]["B1"] = {"nb1_b1_fp8": {"status": "DONE", "free_mib_per_card": [1607, 1607, 1647, 1633]}}
ns["R"]["sections"]["B2"] = {"nb1_b2_fp8": {"status": "DONE", "free_mib_per_card": [1650, 1650, 1700, 1690]}}
pr = ns["pin_rule"]()
check("pin rule: RC2's free MiB (min 1607) -> start +1.0 GiB, ladder 1.0 / 0.75 / 0.5", pr["ok"] and pr["steps_gib"] == [1.0, 0.75, 0.5], json.dumps(pr)[:200])
ns["R"]["sections"]["B2"]["nb1_b2_fp8"]["free_mib_per_card"] = [1400, 1500, 1500, 1500]
check("pin rule: 1400 MiB -> start +0.75", ns["pin_rule"]()["steps_gib"] == [0.75, 0.5])
ns["R"]["sections"]["B2"]["nb1_b2_fp8"]["free_mib_per_card"] = [900, 1500, 1500, 1500]
check("pin rule: 900 MiB -> no step (fail closed)", ns["pin_rule"]()["ok"] is False)
ns["R"]["sections"]["B1"]["nb1_b1_fp8"]["free_mib_per_card"] = None
ns["R"]["sections"]["B2"]["nb1_b2_fp8"]["free_mib_per_card"] = None
check("pin rule: free VRAM not measured -> fail closed", ns["pin_rule"]()["ok"] is False)
ns["R"]["sections"]["B1"]["nb1_b1_fp8"]["free_mib_per_card"] = [1607, 1607, 1647, 1633]
pr_b2miss = ns["pin_rule"]("B2")
pr_b1 = ns["pin_rule"]("B1")
ns["R"]["sections"]["B2"]["nb1_b2_fp8"]["free_mib_per_card"] = [1500, None, 1500, 1500]
pr_b2bad = ns["pin_rule"]("B2")
ns["R"]["sections"]["B2"]["nb1_b2_fp8"]["free_mib_per_card"] = [1650, 1650, 1700, 1690]
ns["R"]["sections"]["B1"]["nb1_b1_fp8"]["free_mib_per_card"] = None
pr_b2only = ns["pin_rule"]("B2")
check("R2 audit N1-3: pin rule on B2 with B2's free memory missing / invalid -> fail closed although B1's is valid (never from B1 "
      "alone); on B1 (B2 not clean) B1's own reading decides; on B2 with only B2's reading -> B2 decides",
      pr_b2miss["ok"] is False and "never decided from B1 alone" in pr_b2miss["why"] and pr_b2bad["ok"] is False
      and pr_b1["ok"] and pr_b1["free_sources"] == ["B1"] and pr_b1["steps_gib"] == [1.0, 0.75, 0.5]
      and pr_b2only["ok"] and pr_b2only["free_sources"] == ["B2"] and pr_b2only["min_free_mib_per_card"] == [1650, 1650, 1700, 1690],
      json.dumps([pr_b2miss["why"], pr_b2bad["why"], pr_b1.get("free_sources"), pr_b2only.get("free_sources")])[:400])
ns["R"]["sections"]["B1"]["nb1_b1_fp8"]["free_mib_per_card"] = [1607, 1607, 1647, 1633]
ns["NB1"]["pin_rule"] = ns["pin_rule"]()
calls = []
real_step, real_run = ns["step"], ns["run_boot"]


def fake_step(sec, name, est, fn):
    r = fn()
    ns["R"]["sections"].setdefault(sec, {})[name] = r
    return r


def fake_run(sec, body, need=()):
    calls.append(ns["BOOTS"]["B3"]["kv_bytes"])
    ok = len(calls) >= 2
    return {"status": "DONE" if ok else "PIN_FREE_BELOW_0P5_GIB", "correctness": "CLEAN", "quality_gate": "PASS",
            "kv_bytes": ns["BOOTS"]["B3"]["kv_bytes"], "pin_check": {"ok": ok}}


ns["step"], ns["run_boot"] = fake_step, fake_run
REAL_SCS = ns["scorecard_save"]
ns["scorecard_save"] = lambda: None
ns["pin_ladder"]()
check("pin ladder: +1.0 GiB fails the in-boot check -> +0.75 GiB boots clean -> stop (2 tries, descending)",
      calls == [ns["KV_FP8"] + 2 ** 30, ns["KV_FP8"] + int(0.75 * 2 ** 30)] and len(ns["NB1"]["pin_tries"]) == 2
      and ns["BOOTS"]["B3"]["name"] == "nb1_b3_fp8_pin75", json.dumps(ns["NB1"]["pin_tries"])[:300])
calls.clear()
ns["run_boot"] = lambda sec, body, need=(): calls.append(1) or {"status": "PIN_FREE_BELOW_0P5_GIB"}
ns["pin_ladder"]()
check("pin ladder: every step fails -> 3 tries, B3 not clean", len(calls) == 3 and not ns["boot_ok"](ns["_rec"]("B3"))
      and ns["NB1"]["recommended_pin"] is None)
calls.clear()


def tfoom_run(sec, body, need=()):
    calls.append(ns["BOOTS"]["B3"]["kv_bytes"])
    first = len(calls) == 1
    return {"status": "TF_OOM" if first else "DONE", "correctness": "CLEAN", "quality_gate": "PASS",
            "kv_bytes": ns["BOOTS"]["B3"]["kv_bytes"], "pin_check": {"ok": True}}


ns["run_boot"] = tfoom_run
ns["pin_ladder"]()
tr = ns["NB1"]["pin_tries"]
check("R2 audit N1-7: pin ladder: +1.0 GiB clean but TF_OOM -> UNSAFE_HEADROOM, never recommended, the ladder goes on; +0.75 "
      "clean -> the recommended pin (DIAGNOSTIC label)", len(calls) == 2 and tr[0]["headroom"] == "UNSAFE_HEADROOM"
      and tr[1]["headroom"] is None and ns["NB1"]["recommended_pin"]["gib"] == 0.75 and "DIAGNOSTIC" in ns["NB1"]["recommended_pin"]["label"]
      and tr[1]["correctness"] == "DIAGNOSTIC_PASS" and tr[0]["correctness"] == "UNTESTED", json.dumps(tr)[:400])
calls.clear()
ns["run_boot"] = lambda sec, body, need=(): calls.append(1) or {"status": "TF_OOM", "correctness": "CLEAN", "quality_gate": "PASS",
                                                                 "pin_check": {"ok": True}}
ns["pin_ladder"]()
check("R2 audit N1-7: pin ladder: every step TF_OOM -> 3 tries kept (measurements), no recommended pin",
      len(calls) == 3 and ns["NB1"]["recommended_pin"] is None and all(x["headroom"] == "UNSAFE_HEADROOM" for x in ns["NB1"]["pin_tries"]))
ns["step"], ns["run_boot"] = real_step, real_run
real_gm = L.gpu_mem
L.gpu_mem = lambda: [[0, 22600, 23034], [1, 22000, 23034], [2, 22000, 23034], [3, 22000, 23034]]
check("pin free check: a rank with 434 MiB free -> not ok", ns["pin_free_check"]()["ok"] is False)
L.gpu_mem = lambda: None
check("pin free check: no nvidia-smi on the GPU box -> fail closed", ns["pin_free_check"]()["ok"] is False)
L.gpu_mem = real_gm

# ---- 8. the chain (nb1_main) fails closed
def run_main(patch_ok=True, b1=None, b2=None, rows=True):
    seen = []
    ns["R"]["sections"] = {"A0": {"merge": {"ok": True}}}

    def st(sec, name, est, fn):
        seen.append(f"{sec}/{name}")
        if name == "merlin_install":
            r = {"status": "DONE"}
        elif name == "patch_60490":
            r = {"status": "DONE", "ok": True} if patch_ok else {"status": "ERROR", "error": "PATCH_60490_PRE_SHA_MISMATCH"}
        elif sec == "B1":
            r = b1
        elif sec == "B2":
            r = b2
        else:
            r = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS"}
        ns["R"]["sections"].setdefault(sec, {})[name] = r
        return r
    ns["step"] = st
    ns["pin_rule"] = lambda base="B2": {"ok": True, "steps_gib": [0.5], "why": "test", "base": base}
    ns["ROWS"].pop("rc2_fp8", None)
    if rows:
        ns["ROWS"]["rc2_fp8"] = [{}]
    out = ns["nb1_main"]()
    return seen, out


GOOD = {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS", "core_measurements_complete": True,
        "core_measurements": {"missing": []}}
saved_fake = ns["FAKE"]
ns["FAKE"] = None
seen, out = run_main(patch_ok=False)
check("chain: patch failure -> no boot starts, all SKIPPED_DEPENDENCY", seen == ["A0/merlin_install", "A0/patch_60490"]
      and set(out.values()) == {"SKIPPED_DEPENDENCY"}, json.dumps(out))
seen, out = run_main(rows=False)
check("chain: RC2 oracle rows unusable -> no boot starts", not any(s.startswith("B") for s in seen) and set(out.values()) == {"SKIPPED_DEPENDENCY"})
seen, out = run_main(b1={"status": "NO_GO_QUALITY", "correctness": "CLEAN", "quality_gate": "FAIL"})
check("chain: B1 gate FAIL -> B2 / B3 never start", [s for s in seen if s.startswith("B")] == ["B1/nb1_b1_fp8"]
      and out["B2"] == out["B3"] == "SKIPPED_DEPENDENCY", json.dumps(out))
seen, out = run_main(b1=GOOD, b2={"status": "LAUNCH_FAILED", "correctness": None})
check("chain: B2 launch failure -> the B3 pin ladder runs on B1's clean configuration (no B2 levers, no NCCL debug)",
      [s for s in seen if s.startswith("B")][:2] == ["B1/nb1_b1_fp8", "B2/nb1_b2_fp8"] and any(s.startswith("B3/") for s in seen)
      and ns["NB1"]["b3_base"]["base"] == "B1" and ns["BOOTS"]["B3"]["extra"] == ns["FP8_EXTRA"]
      and "NCCL_DEBUG" not in ns["BOOTS"]["B3"]["env"] and "VLLM_RECOVERSSM_COMMIT_GRAPH" not in ns["BOOTS"]["B3"]["env"], str(seen))
seen, out = run_main(b1=GOOD, b2=GOOD)
check("chain: B2 clean -> the B3 ladder runs on B2's configuration without the commit-graph env",
      ns["NB1"]["b3_base"]["base"] == "B2" and "--mamba-ssm-cache-dtype" in ns["BOOTS"]["B3"]["extra"]
      and "VLLM_RECOVERSSM_COMMIT_GRAPH" not in ns["BOOTS"]["B3"]["env"])
seen, out = run_main(b1=GOOD, b2=GOOD)
check("chain: B1 + B2 clean -> the B3 ladder runs", any(s.startswith("B3/nb1_b3_fp8_pin") for s in seen), str(seen))
seen, out = run_main(b1={"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS", "postprocess_error": "KeyError: x"}, b2=GOOD)
b2s = ns["_rec"]("B2")
check("R2 audit N1-3: chain: B1 clean (boot_ok) but core measurements missing -> B2 SKIPPED_DEPENDENCY (reason lists the missing "
      "parts); the B3 ladder may run on B1's configuration only through the pin rule on B1's own free memory",
      "B2/nb1_b2_fp8" not in seen and b2s.get("status") == "SKIPPED_DEPENDENCY" and "core measurements incomplete" in b2s.get("why", "")
      and "free_mib_per_card" in b2s.get("why", "") and ns["NB1"]["b3_base"]["base"] == "B1"
      and ns["_rec"]("B1").get("core_measurements_complete") is False and ns["_rec"]("B1").get("postprocess_ok") is False,
      json.dumps({"seen": seen, "b2": b2s})[:400])
ns["step"] = real_step
ns["FAKE"] = saved_fake

# ---- 9. windows: prefill must be done before timing; tick / cycle; uncached prefill
real_levels, real_kvu, real_acc, real_ctr, real_idle = ns["levels"], ns["kvu"], ns["accept_pos"], ns["_counters"], L.wait_idle
ns["kvu"] = lambda port: 0.1
ns["accept_pos"] = lambda port: None
L.wait_idle = lambda port, wait=90.0: True
seqc = iter([(1000.0, 0.0), (41000.0, 8000.0)])
ns["_counters"] = lambda port: next(seqc)
win = {"status": "STEADY", "running_min": 4, "running_max": 4, "step_ms": 37.5, "admission": {"sec": 10.0, "tail": [[10.0, 4, 0, 4, 4]]}}
ns["levels"] = lambda *a, **k: {"40k@4": dict(win)}
w1 = ns["window"](1, None, "40k", 4)
check("window: STEADY, tick 37.5 ms, cycle 75.0 ms, uncached prefill = (prompt - hits) / admission",
      w1["status"] == "STEADY" and w1["tick_ms"] == 37.5 and w1["cycle_ms"] == 75.0 and w1["prefill_tok_s_uncached"] == 3200.0
      and w1["retained_usage_before"] == 0.1, json.dumps({k: w1.get(k) for k in ("status", "tick_ms", "cycle_ms", "prefill_tok_s_uncached")}))
ns["_counters"] = lambda port: (None, None)
ns["levels"] = lambda *a, **k: {"40k@4": dict(win, admission={"sec": 10.0, "tail": [[10.0, 4, 0, 3, 3]]})}
check("window: a stream without its first token at window start -> INCONCLUSIVE_PREFILL_NOT_DONE",
      ns["window"](1, None, "40k", 4)["status"] == "INCONCLUSIVE_PREFILL_NOT_DONE")
n_calls = []
ns["levels"] = lambda *a, **k: n_calls.append(1) or {"40k@4": dict(win, status="INCONCLUSIVE_PREFILL_IN_WINDOW") if len(n_calls) == 1 else dict(win)}
w3 = ns["window"](1, None, "40k", 4)
check("window: prefill inside the window -> retried once", len(n_calls) == 2 and w3["status"] == "STEADY" and w3.get("retried_after"))
ns["levels"] = lambda *a, **k: {"40k@4": dict(win, running_min=3)}
check("window: running below the level -> not STEADY", ns["window"](1, None, "40k", 4)["status"] == "INCONCLUSIVE_RUNNING_NE_LEVEL")
ns["levels"], ns["kvu"], ns["accept_pos"], ns["_counters"], L.wait_idle = real_levels, real_kvu, real_acc, real_ctr, real_idle

# ---- 10. host memory gate, engine processes, boot gate composition, shm cleanup
real_hm, real_drop, real_sleep = ns["host_mem"], ns["drop_page_cache"], ns["time"].sleep
ns["drop_page_cache"] = lambda: {"available": True, "files": 1}
ns["time"].sleep = lambda s: None
ns["FAKE"] = None
ns["host_mem"] = lambda: {"avail_mib": 60000}
h = ns["host_ram_gate"]("unit")
check("host gate: avail 60,000 MiB < 73 GiB after 3 more drops -> fail closed", h["ok"] is False and len(h["drops"]) == 4 and "HOST_RAM_GATE" in h["why"])
ns["host_mem"] = lambda: {"avail_mib": 80000}
check("host gate: 80,000 MiB -> ok", ns["host_ram_gate"]("unit")["ok"] is True)
ns["host_mem"] = lambda: {}
check("host gate: no reading on the GPU box -> fail closed", ns["host_ram_gate"]("unit")["ok"] is False)
ns["FAKE"] = "fake"
ns["host_mem"] = lambda: {"avail_mib": 25000}
hf = ns["host_ram_gate"]("unit")
check("host gate: fake harness on a small CPU box -> recorded, not enforced, no retries", hf["ok"] is True and len(hf["drops"]) == 1
      and "not enforced" in hf["why"])
ns["FAKE"] = None
ns["time"].sleep = real_sleep
real_bg, real_cl, real_hg = ns["_boot_gate_rc2"], ns["clear_engine_procs"], ns["host_ram_gate"]
ns["_boot_gate_rc2"] = lambda: {"ok": False, "gpus": []}
check("boot gate: GPUs not released -> not ok, nothing else runs", ns["boot_gate"]()["ok"] is False and "host_ram" not in ns["boot_gate"]())
ns["_boot_gate_rc2"] = lambda: {"ok": True}
ns["clear_engine_procs"] = lambda: {"ok": False, "after_kill": [[123, "VLLM::Worker"]]}
g = ns["boot_gate"]()
check("boot gate: an engine process survives the kill -> ENGINE_PROCS_LEFT", g["ok"] is False and g["why"].startswith("ENGINE_PROCS_LEFT"))
ns["clear_engine_procs"] = lambda: {"ok": True}
ns["host_ram_gate"] = lambda tag: {"ok": False, "why": "HOST_RAM_GATE: x"}
check("boot gate: host RAM below the rule -> not ok", ns["boot_gate"]()["why"].startswith("HOST_RAM_GATE"))
ns["host_ram_gate"] = lambda tag: {"ok": True, "why": "ok"}
check("boot gate: all three hold -> ok", ns["boot_gate"]()["ok"] is True)
ns["_boot_gate_rc2"], ns["clear_engine_procs"], ns["host_ram_gate"], ns["host_mem"], ns["drop_page_cache"] = real_bg, real_cl, real_hg, real_hm, real_drop
shm = WORK / "shm"
(shm / "sem.loky-old").write_text("x")
before = ns["shm_names"]()
(shm / "sem.loky-4447-new").write_text("x")
(shm / "psm_9a642804").write_text("x")
(shm / "other_new").write_text("x")
res = ns["shm_cleanup"](before)
names = {p.name for p in shm.iterdir()}
check("shm: this boot's psm_ and sem.loky- entries removed, pre-existing and unknown kept",
      "sem.loky-old" in names and "other_new" in names and "sem.loky-4447-new" not in names and "psm_9a642804" not in names
      and res["removed_loky"] == ["sem.loky-4447-new"], str(sorted(names)))

# ---- 11. per-boot watchdog
killed = []


class S:
    def kill(self):
        killed.append(1)


body = ns["watched"](lambda ss: time.sleep(1.5) or {"ok": True}, -299.5, "unit")
o = body([S()])
check("watchdog: the body overruns cap + 300 s -> server killed, recorded", killed and "FIRED" in o.get("boot_watchdog", ""))
o = ns["watched"](lambda ss: {"ok": True}, 60, "unit")([S()])
check("watchdog: a body within its cap -> no kill", "boot_watchdog" not in o and len(killed) == 1)

# ---- 12. config diff vs RC2 FP8, long-row gate, lp_compare, commit counters
argv = ["/v/bin/vllm", "serve", "/ckpt"] + ns["RC2_FP8"]["argv_tail"] + ["--language-model-only", "--mamba-ssm-cache-dtype", "bfloat16"]
cd = ns["config_diff"](argv, dict(ns["RC2_FP8"]["env"], MERLIN_SHARE_DRAFT_WORKSPACE="1"), ns["KV_FP8"])
check("config diff: only the added flags and env differ from RC2 FP8", set(cd["flags"]) == {"--language-model-only", "--mamba-ssm-cache-dtype"}
      and set(cd["env"]) == {"MERLIN_SHARE_DRAFT_WORKSPACE"}, json.dumps(cd)[:300])
lq = ns["long_quality"]
row = {"tokens": 1000, "error": None, "window_garbage": None, "cut_by_cap": False, "finish": "length", "think_closed": True}
check("long gate: natural row with finish length -> INCOMPLETE_NATURAL_FINISH", lq("natural", row) == "INCOMPLETE_NATURAL_FINISH")
check("long gate: natural row with an open think -> INCOMPLETE_THINKING", lq("natural", dict(row, finish="stop", think_closed=False)) == "INCOMPLETE_THINKING")
o = {"long": {"natural": row, "forced60k": dict(row, reached_60k=True)}}
check("long gate: the record's verdict becomes INCONCLUSIVE_LONG_INCOMPLETE; forced rows labelled stability only",
      ns["long_gate"](o) == "INCONCLUSIVE_LONG_INCOMPLETE" and "stability" in o["long"]["forced60k"]["counts_as"])
a = {"ids": [1, 2, 3], "lps": [-0.1, -0.2, -0.3], "margins": [1.0, 2.0, 0.1]}
check("lp_compare: identical -> EXACT; low-margin split -> NEAR_TIE; high-margin split -> FAIL_HIGH_MARGIN",
      ns["lp_compare"](a, dict(a))["status"] == "EXACT" and ns["lp_compare"](a, dict(a, ids=[1, 2, 9]))["status"] == "NEAR_TIE"
      and ns["lp_compare"](a, dict(a, ids=[1, 9, 9]))["status"] == "FAIL_HIGH_MARGIN")
cc = ns["commit_counter_delta"]([{"eager_steps": 10, "staged_steps": 0, "replays": 0}] * 4,
                                [{"eager_steps": 10, "staged_steps": 30, "replays": 10, "fallbacks": {"pp_deferred": 30}}] * 4)
check("commit counters: per-arm deltas summed over ranks, replay share", cc["replays"] == 40 and cc["staged_steps"] == 120
      and cc["replay_share"] == 0.25 and cc["fallbacks_total_after"] == {"pp_deferred": 120}, json.dumps(cc))

# ---- 13. the lever scorecard: rows, fields, incremental (B1 only), attribution, combined labels, NB0 note
ns["BOOTS"]["B3"]["name"] = "nb1_b3_fp8_pin50"
ns["R"]["sections"] = {"A0": {"merge": {"ok": True}, "patch_60490": {"status": "DONE", "applied": "applied", "ok": True}},
                      "B1": {"nb1_b1_fp8": {"status": "DONE", "correctness": "CLEAN", "quality_gate": "PASS", "sec": 1800,
                                            "servers": {"nb1_b1_fp8": {"kv_tokens": 513940, "max_concurrency": 3.92,
                                                                       "gpu_mem_after_ready": [[0, 20500, 23034], [1, 20500, 23034], [2, 20800, 23034], [3, 21300, 23034]]}},
                                            "free_after_ready_mib": [2534, 2534, 2234, 1734], "free_mib_per_card": [1900, 1900, 1650, 1640],
                                            "pool": {"rank_load_gib": {0: 16.93, 1: 16.93, 2: 16.87, 3: 16.87}},
                                            "kv": {"pool_blocks": 149, "fit": {"F": 6.0}}, "caps": {"max_num_seqs": 8},
                                            "oracle": {"vs_rc2_fp8": {"status": "EXACT"}},
                                            "steady": {"_fit": {"8k": 8}, "_plan": ["8k@8"], "8k@8": {"status": "STEADY", "tok_s": 262.0, "mean_acceptance_length": 3.15, "tick_ms": 48.1, "cycle_ms": 96.2}}}}}
ns["SCORECARD"] = WORK / "out/lever_scorecard.json"
ns["scorecard_save"] = REAL_SCS
rows = ns["scorecard"]()
check("scorecard: 10 rows, every required field, written incrementally with only B1 present",
      len(rows) == 10 and all(all(k in r for k in ("lever", "measured", "projected_run_tok_s_gain", "correctness", "confidence", "adoption",
                                                    "cost_gpu_h", "notes", "gain_scope")) for r in rows), str([r["lever"][:20] for r in rows]))
d9 = rows[0]["measured"]["attribution"]
check("scorecard: C4 / D9 attribution per rank (C4 = 0.40 GiB weight drop on PP0 = 410 MiB; D9 = free change - C4)",
      d9["c4_weight_drop_mib_per_rank"] == [410, 410, 0, 0] and d9["d9_mib_per_rank"] == [403 - 410, 403 - 410, 63, 89 - 0]
      and d9["d9_expected_reserve_mib"] == 257.0, json.dumps(d9)[:300])
check("scorecard: the D9 row is a patch, C3 / share / seqs rows are labelled combined, every row carries the NB0 note",
      rows[0]["adoption"] == "patch" and all("combined" in r["gain_scope"] for r in rows if r["lever"].startswith(("C3", "MERLIN", "seqs")))
      and all("NB0" in r["notes"] for r in rows if r["lever"] != "W6 chunk 2048 (--max-num-batched-tokens)"))
check("scorecard: commit-graph projection is None without replays", next(r for r in rows if r["lever"].startswith("RecoverSSM"))["projected_run_tok_s_gain"] is None)
check("R2 audit N1-1: no scorecard row is labelled PASS; the D9 row on a clean B1 with EXACT -> DIAGNOSTIC_PASS",
      all(r["correctness"] != "PASS" for r in rows) and rows[0]["correctness"] == "DIAGNOSTIC_PASS"
      and next(r for r in rows if r["lever"].startswith("C4"))["correctness"] == "DIAGNOSTIC_PASS", json.dumps([r["correctness"] for r in rows]))
b1sc = ns["R"]["sections"]["B1"]["nb1_b1_fp8"]
cw0 = ns["capacity_winner"]()
check("R2 audit N1-2: capacity winner: B1 with no worker reading (pool unverified) is excluded (no winner)",
      cw0["winner"] is None and any("INCONCLUSIVE_CAPACITY_UNKNOWN" in w for w in cw0["excluded"]["B1"]), json.dumps(cw0)[:400])
b1sc.update(workers={"num_gpu_blocks": [150]}, kv={"pool_blocks": 149, "fit": TF(5.99)},
            pool={"rank_load_gib": {0: 16.93, 1: 16.93, 2: 16.87, 3: 16.87}, "kv_tokens": 513940, "max_concurrency": 3.92, "max_concurrency_tokens": 131072},
            census={"kv_tokens": [513940]}, max_safe_running_by_context={"status": "MEASURED"})
ns["ensure_core"](b1sc)
cw1 = ns["capacity_winner"]()
check("R2 audit: capacity winner: B1 certified (MATCH, fit 5.99 agrees with log 6), core complete, DIAGNOSTIC_PASS -> winner B1 at "
      "149 // (18 + 6) = 6 streams at 70k", b1sc["core_measurements_complete"] is True and cw1["winner"] == "B1"
      and cw1["resident_streams_at_70k"] == 6 and "DIAGNOSTIC" in cw1["label"], json.dumps(cw1)[:400])
b1sc["status"] = "TF_OOM"
ns["ensure_core"](b1sc)
cw2 = ns["capacity_winner"]()
rows2 = ns["scorecard"]()
c4 = next(r for r in rows2 if r["lever"].startswith("C4"))
check("R2 audit N1-7: B1 TF_OOM -> UNSAFE_HEADROOM: excluded from the capacity winner, its lever row NOT_RECOMMENDED, measurements kept",
      cw2["winner"] is None and any("UNSAFE_HEADROOM" in w for w in cw2["excluded"]["B1"]) and c4["adoption"].startswith("NOT_RECOMMENDED: UNSAFE_HEADROOM")
      and c4["measured"] is not None and b1sc["kv"]["pool_blocks"] == 149, json.dumps({"cw": cw2["excluded"], "adopt": c4["adoption"]})[:400])
b1sc["status"] = "DONE"
b1sc.pop("headroom", None)
ns["scorecard_save"]()
sc = json.loads((WORK / "out/lever_scorecard.json").read_text())
check("scorecard_save: the file exists with rows and the boot statuses", len(sc["rows"]) == 10 and sc["boots"]["B1"] == "DONE", json.dumps(sc["boots"]))
check("R2 audit: scorecard file carries boot labels, recommended_pin and capacity_winner", sc["boot_labels"]["B1"]["correctness"] == "DIAGNOSTIC_PASS"
      and "recommended_pin" in sc and sc["capacity_winner"]["winner"] == "B1", json.dumps({k: sc.get(k) for k in ("boot_labels", "capacity_winner")})[:400])
real_sc = ns["scorecard"]
ns["scorecard"] = lambda: 1 / 0
check("scorecard_save: an exception is recorded, never raised", ns["scorecard_save"]() is None and any("scorecard" in e for e in ns["R"]["errors"]))
ns["scorecard"] = real_sc

# ---- 14. alloc sequence fail-soft, prompts distinct per test
real_idle = L.wait_idle
L.wait_idle = lambda port, wait=90.0: False
check("alloc sequence: server not idle -> INCONCLUSIVE_NOT_IDLE", ns["alloc_sequence"](1, "8k", (1,))["status"] == "INCONCLUSIVE_NOT_IDLE")
L.wait_idle = real_idle
P = ns["PROMPTS"]
first_blocks = {tuple(P["8k"][v][:4096]) for v in ns["V_CENSUS"]["8k"]}
check("prompts: the 8 cold-census 8k prompts differ in their first 4,096-token block (no shared prefix)", len(first_blocks) == 8)
check("prompts: census / prefix / abort variants never used by an 8k steady window (0..23) or the 40k ladder (0..7)",
      min(ns["V_CENSUS"]["8k"]) >= 24 and ns["V_PREFIX"]["8k"] not in ns["V_CENSUS"]["8k"] and ns["V_ABORT"] not in ns["V_CENSUS"]["8k"]
      and min(ns["V_CENSUS"]["40k"]) >= 9 and ns["V_PREFIX"]["40k"] == 8)
check("residency bound: P 149, F 6 -> 8k 18, 40k 9, 60k 7, 90k 5, 116k 4 (cap seqs 8 -> 8 at 8k)",
      [ns["residency_bound"](149, 6, b) for b in ns["RESIDENCY_BANDS"]] == [18, 9, 7, 5, 4] and ns["residency_bound"](149, 6, "8k", seqs=8) == 8)
calls_ = []
real_adm = ns["admission_test"]
ns["admission_test"] = lambda port, s, band, n, max_new=0: calls_.append((band, n, max_new)) or {"status": "ALL_RESIDENT"}
ns["PREFILL"].update(tok_s=4400.0)
ns["FAKE"] = None
rp = ns["residency_probes"](1, None, {"kv": {"pool_blocks": 149, "fit": TF(6.0)}, "workers": {"num_gpu_blocks": [150]},
                                      "caps": {"max_num_seqs": 8}}, bands=("60k",))
check("residency probe: 60k x 7 (max_new 1440 < the staggered-admission growth) -> SKIPPED_HEADROOM, achieved at 6 with max_new 8192",
      rp["bands"]["60k"]["tries"][0]["status"] == "SKIPPED_HEADROOM" and calls_ == [("60k", 6, 8192)] and rp["bands"]["60k"]["achieved"] == 6
      and rp["bands"]["60k"]["bound_prompt_only"] == 7, json.dumps(rp["bands"]["60k"])[:300])

# C-6: the probes use this boot's own measured pool and F (kv.pool_blocks, ceil(kv.fit.F)); never RC2's 149 / 6
BLK = {b: -(-ns["BANDS"][b] // 4096) for b in ns["RESIDENCY_BANDS"]}
check("C-6 fixture: residency bands are 8k / 40k / 60k / 90k / 116k = 2 / 10 / 15 / 22 / 29 token blocks",
      list(ns["RESIDENCY_BANDS"]) == ["8k", "40k", "60k", "90k", "116k"] and BLK == {"8k": 2, "40k": 10, "60k": 15, "90k": 22, "116k": 29}, json.dumps(BLK))


def probe_c6(o, bands=tuple(ns["RESIDENCY_BANDS"])):
    calls_.clear()
    return ns["residency_probes"](1, None, o, bands=bands)


r5 = probe_c6({"kv": {"pool_blocks": 220, "fit": TF(5.0)}, "workers": {"num_gpu_blocks": [221]}, "caps": {}})
b5 = {b: v["bound_prompt_only"] for b, v in r5["bands"].items()}
c5 = list(calls_)
r8 = probe_c6({"kv": {"pool_blocks": 220, "fit": TF(8.0)}, "workers": {"num_gpu_blocks": [221]}, "caps": {}})
b8 = {b: v["bound_prompt_only"] for b, v in r8["bands"].items()}
c8 = list(calls_)
exp5 = {b: 220 // (BLK[b] + 5) for b in BLK}
exp8 = {b: 220 // (BLK[b] + 8) for b in BLK}
check("C-6: P 220 / F 5 -> bounds 8k 31, 40k 14, 60k 11, 90k 8, 116k 6 (this boot's pool and F, status MEASURED, F source measured)",
      b5 == exp5 == {"8k": 31, "40k": 14, "60k": 11, "90k": 8, "116k": 6} and r5["pool_blocks"] == 220 and r5["F"] == 5
      and r5["status"] == "MEASURED" and "measured in this boot" in r5["F_source"] and "RC2" not in r5["F_source"], json.dumps(b5))
check("C-6: P 220 / F 8 -> bounds 8k 22, 40k 12, 60k 9, 90k 7, 116k 5 (differ from F 5 in every band)",
      b8 == exp8 == {"8k": 22, "40k": 12, "60k": 9, "90k": 7, "116k": 5} and r8["F"] == 8 and all(b5[b] != b8[b] for b in BLK), json.dumps(b8))


def mn_ok(calls, P, F):
    return bool(calls) and all(m == min(8192, int((P // n - F) * 4096 - ns["BANDS"][b])) for b, n, m in calls)


check("C-6: the probe requests size max_new from the boot's F (F 5 and F 8 give different requests, each = (P // n - F) x 4096 - band)",
      mn_ok(c5, 220, 5) and mn_ok(c8, 220, 8) and c5 != c8, json.dumps({"F5": c5, "F8": c8})[:300])
rf = probe_c6({"kv": {"pool_blocks": 220, "fit": TF(7.3)}, "workers": {"num_gpu_blocks": [221]}, "caps": {}}, bands=("90k",))
check("C-6: fractional F 7.3 -> F 8 (rounded up), F_fit 7.3 kept; 6.0 stays 6", rf["F"] == 8 and rf["F_fit"] == 7.3
      and rf["bands"]["90k"]["bound_prompt_only"] == 7
      and ns["residency_capacity"]({"kv": {"pool_blocks": 149, "fit": TF(6.0)}, "workers": {"num_gpu_blocks": [150]}})["F"] == 6,
      json.dumps({k: rf[k] for k in ("F", "F_fit")}))
rp_ = probe_c6({"kv": {"pool_blocks": None, "fit": None, "why": "pool blocks not recoverable from the usage readings"},
                "workers": {"num_gpu_blocks": [150]}, "caps": {"max_num_seqs": 8}})
check("C-6: pool missing (worker reports 150 blocks) -> INCONCLUSIVE_CAPACITY_UNKNOWN, no probe requests, no 149 fallback",
      rp_["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and not calls_ and rp_["bands"] == {} and rp_["pool_blocks"] is None
      and rp_["F"] is None and rp_["F_source"] is None and any("pool_blocks" in w for w in rp_["why"]), json.dumps(rp_)[:300])
rF = probe_c6({"kv": {"pool_blocks": 220, "fit": None}, "workers": {"num_gpu_blocks": [221]}, "caps": {}})
rF2 = probe_c6({"kv": {"pool_blocks": 220, "fit": {"F": None}}, "caps": {}})
rF3 = probe_c6({"kv": {"pool_blocks": 220, "fit": {"F": float("nan")}}, "caps": {}})
rF4 = probe_c6({"kv": {"pool_blocks": 220, "fit": {"F": -1.5}}, "caps": {}})
check("C-6: F missing (fit None / F None / NaN / negative) -> INCONCLUSIVE_CAPACITY_UNKNOWN, no probe requests, no F 6 fallback",
      all(r["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and r["bands"] == {} and r["F"] is None for r in (rF, rF2, rF3, rF4)) and not calls_
      and any("kv.fit.F" in w for w in rF["why"]) and any("invalid" in w for w in rF3["why"]), json.dumps([rF["why"], rF3["why"], rF4["why"]]))
rn = probe_c6({"caps": {}})
check("C-6: no kv record at all (alloc off) -> INCONCLUSIVE_CAPACITY_UNKNOWN, no probe requests", rn["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and not calls_)
rm = probe_c6({"kv": {"pool_blocks": 220, "fit": TF(6.0)}, "workers": {"num_gpu_blocks": [150]}, "caps": {}}, bands=("90k",))
rm_calls = list(calls_)
rm2 = ns["residency_capacity"]({"kv": {"pool_blocks": 220, "fit": TF(6.0)}, "workers": {"num_gpu_blocks": [221, 230]}})
rok = ns["residency_capacity"]({"kv": {"pool_blocks": 220, "fit": TF(6.0)}, "workers": {"num_gpu_blocks": [221]}})
runk = probe_c6({"kv": {"pool_blocks": 220, "fit": TF(6.0)}, "workers": {"error": "<HTTPError 404>"}, "caps": {}}, bands=("90k",))
runk_calls = list(calls_)
check("R2 audit N1-2: worker num_gpu_blocks 150 vs gauge pool 220 -> MISMATCH -> INCONCLUSIVE_CAPACITY_UNKNOWN (POOL_MISMATCH), no "
      "probes; ranks 221 / 230 -> MISMATCH, not ok; 221 -> MATCH, certified; no worker reading -> UNKNOWN -> INCONCLUSIVE "
      "(POOL_UNVERIFIED), no probes",
      rm["capacity"]["pool_check"] == "MISMATCH" and rm["capacity"]["worker_num_gpu_blocks"] == [150] and rm["capacity"]["worker_usable_blocks"] == [149]
      and "220" in rm["capacity"]["pool_mismatch"] and rm["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and rm["bands"] == {} and not rm_calls
      and rm["capacity"]["reason_codes"] == ["POOL_MISMATCH"] and any("POOL_MISMATCH" in w for w in rm["why"]) and rm["capacity"]["F_resolved"] == 6
      and rm2["pool_check"] == "MISMATCH" and rm2["ok"] is False and rok["pool_check"] == "MATCH" and rok["ok"] and rok["status"] == "CERTIFIED"
      and "pool_mismatch" not in rok and runk["capacity"]["pool_check"] == "UNKNOWN" and runk["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN"
      and not runk_calls and runk["capacity"]["reason_codes"] == ["POOL_UNVERIFIED"] and "HTTPError" in runk["why"][0],
      json.dumps([rm["capacity"].get("why"), runk["why"]])[:400])
ns["admission_test"] = real_adm
lay = ns["block_layout"]({"ranks": [{"kv_cache_config": {"num_blocks": 150, "groups": [{"spec": "FullAttentionSpec", "layers": 6},
      {"spec": "MambaSpec", "layers": 13}, {"spec": "MambaSpec", "layers": 12}, {"spec": "QSARingSpec", "layers": 6}, {"spec": "MambaSpec", "layers": 12}]}}]})
check("block layout: 3 mamba groups -> F 6 = 2 per group (inferred, engine-derived F, no reclaim claim)", lay["mamba_groups"] == 3
      and lay["F_per_mamba_group_inferred"] == 2.0 and "not shown" in lay["note"])
check("profiler tools: availability recorded, DRAM bytes deferred with the reason", "dram_bytes" in ns["profiler_tools"]())
check("d9 reserve: seqs 8, K 3, chunk 1024, 131k, block 4096, compress 4 -> 257 MiB (256 MiB budget + 1 MiB top-k)", ns["d9_reserve_mib"]() == 257.0)

# ---- 11. R2: the R1 GPU crash (run_boot -> free_per_card on nb1_ladder's "_plan" list); post-processing keeps the record
FX = json.loads((HERE / "nb1_fixture_b1_gpu_2026-10-31.json").read_text(encoding="utf-8"))     # REAL B1 step record (R1 GPU run)
LW = FX["record"]["live_windows"]
check("R2 fixture: the real R1 B1 record = ERROR AttributeError 'list'.get in free_per_card, 4 STEADY live windows",
      FX["record"]["status"] == "ERROR" and "'list' object has no attribute 'get'" in FX["record"]["error"]
      and "free_per_card" in FX["record"]["tb"] and list(LW) == ["8k@3", "8k@8", "40k@5", "40k@8"]
      and all(w["status"] == "STEADY" for w in LW.values()), json.dumps({k: w["status"] for k, w in LW.items()}))
saved = {k: ns[k] for k in ("window", "due", "merlin_boot", "LoadMonitor", "free_per_card", "left", "sha_gate", "pin_rule", "FAKE")}
saved_abort = ns["ABORT"]["why"]
ns["ABORT"]["why"] = None
ns["FAKE"] = None
ns["left"] = lambda: 1e9
# the steady dict exactly as R1's nb1_ladder built it in B1 (seqs 8; the invalid fit F -5.32 gave _fit 8 / 8 -> 40k@8 added)
steady = {"_plan": list(LW), "_fit": {"8k": 8, "40k": 8}, "_seqs_cap": 8, **json.loads(json.dumps(LW))}
asked = []
ns["window"] = lambda port, s, band, n, **k: asked.append(f"{band}@{n}") or json.loads(json.dumps(LW.get(f"{band}@{n}") or {"status": "STEADY"}))
ns["due"] = lambda sec: True
B1_KV = {"pool_blocks": FX["max_safe_running_by_context_B1"]["pool_blocks"], "fit": {"F": -5.32}}
B1_POOL = ns["pool_facts"]("\n".join(FX["server_log"]["max_concurrency_lines"]))     # B1's REAL max-concurrency line
check("R2 pool_facts: B1's real line -> N 513,940 tokens, X 3.92, M 131,072 (M is new in R2)",
      B1_POOL["kv_tokens"] == 513940 and B1_POOL["max_concurrency"] == 3.92 and B1_POOL["max_concurrency_tokens"] == 131072, json.dumps(B1_POOL)[:200])
O_B1 = {"caps": {"max_num_seqs": 8, "block_size": 4096}, "kv": B1_KV, "pool": B1_POOL, "workers": {"num_gpu_blocks": [150]}}
st_new = ns["nb1_ladder"](8001, None, O_B1)
n40 = ns["n_fit"](149, 6, 4096, "40k", 0, 8, 8)
want_plan = ["8k@3", "8k@8", "40k@5"] + ([f"40k@{n40}"] if n40 > 5 else [])
asked_new = list(asked)
asked.clear()
st_none = ns["nb1_ladder"](8001, None, {"caps": {"max_num_seqs": 8, "block_size": 4096}, "kv": B1_KV})
check("R2 nb1_ladder: B1's invalid fit (-5.32) is never used: with B1's max-concurrency line F = 6 (log) sizes the extra level; "
      "without the line no extra level (R1 planned 40k@8 from F -5.32)",
      st_new["_F"] == 6 and st_new["_F_source"] == "server max-concurrency line (this boot)" and st_new["_plan"] == want_plan == asked_new
      and st_none["_F"] is None and st_none["_plan"] == ["8k@3", "8k@8", "40k@5"] and st_none["_fit"] == {"8k": None, "40k": None},
      json.dumps({"new": st_new["_plan"], "none": st_none["_plan"], "n40": n40}))


def r1_free_per_card(rec):                    # R1's loop (nb1_code_export R1 :1880-1883), for the reproduction only
    for w in list(((rec or {}).get("steady") or {}).values()) + [((rec or {}).get("profile") or {})]:
        for i, u in enumerate((w or {}).get("gpu_mem_peak_mib") or []):
            pass


ROWS_R = ns["RC2_FP8"]["gpu_mem_after_ready"]
ROWS_E = ns["RC2_FP8"]["gpu_mem_end"]
SRV = {"nb1_b1_fp8": {"gpu_mem_after_ready": ROWS_R, "gpu_mem_end": ROWS_E}}
try:
    r1_free_per_card({"servers": SRV, "steady": steady})
    r1err = None
except AttributeError as e:
    r1err = str(e)
peak = [max(w["gpu_mem_peak_mib"][i] for w in LW.values()) for i in range(4)]
want = [t - max(max(r[1] for r in ROWS_R + ROWS_E if r[0] == i), peak[i]) for i, _, t in ROWS_R]
fpc = saved["free_per_card"]({"servers": SRV, "steady": steady, "profile": {"status": "TRACED"}}, "nb1_b1_fp8")
check("R2 free_per_card: the real B1 steady shape -> R1's loop raises the GPU error; R2 returns total - max(used, every window peak)",
      r1err == "'list' object has no attribute 'get'" and fpc == want, json.dumps({"r1": r1err, "r2": fpc, "want": want}))
odd = {"servers": SRV, "steady": {"_plan": ["8k@3"], "_seqs_cap": 8, "_fit": {"8k": 8}, "8k@3": {"gpu_mem_peak_mib": [22000, None, "x", 1]},
                                  "8k@8": {"gpu_mem_peak_mib": None}, "bad": None}, "profile": ["not", "a", "dict"]}
f_odd = saved["free_per_card"](odd, "nb1_b1_fp8")
check("R2 free_per_card: list / int / None values, a None or non-numeric peak, a non-dict profile, steady not a dict -> no exception",
      f_odd[0] == ROWS_R[0][2] - 22000 and saved["free_per_card"]({"servers": SRV, "steady": ["x"], "profile": 3}, "nb1_b1_fp8") is not None
      and saved["free_per_card"]({"servers": {}, "steady": steady}, "nb1_b1_fp8") is None, json.dumps(f_odd))
check("R2 _conf: a non-dict window value is not counted (no exception)", ns["_conf"]({"windows": {"8k@3": {"status": "STEADY"}, "_plan": ["8k@3"]}})
      .startswith("LOW: 1 STEADY"))


class _Mon:
    def __init__(self, name):
        self.stop_ev, self.last = threading.Event(), {"t": 1}

    def start(self):
        pass


def clean_boot(status=None, **kw):
    return dict({"status": status, "correctness": "CLEAN", "quality_gate": "PASS", "servers": json.loads(json.dumps(SRV)),
                 "steady": json.loads(json.dumps(steady)), "kv": {"pool_blocks": 149, "fit": {"F": -5.32}, "sequences": [1, 2]},
                 "census": {"kv_tokens": 610304}, "oracle": {"vs_rc2_fp8": {"status": "EXACT"}},
                 "pool": dict(B1_POOL), "workers": {"num_gpu_blocks": [150]}, "max_safe_running_by_context": {"status": "MEASURED"}}, **kw)


ns["LoadMonitor"] = _Mon
ns["merlin_boot"] = lambda name, kv, port, timeout, body, need=(), **k: clean_boot()
ns["R"]["sections"] = {}
rb = ns["step"]("B1", "nb1_b1_fp8", 92, lambda: ns["run_boot"]("B1", lambda ss: {}, need=("rc2_fp8",)))
check("R2 run_boot regression (real B1 steady shape, real free_per_card): DONE, free MiB per card set, record complete, boot_ok",
      rb["status"] == "DONE" and rb["free_mib_per_card"] == want and "postprocess_error" not in rb and rb["kv"]["pool_blocks"] == 149
      and rb["steady"]["_plan"] == list(LW) and ns["boot_ok"](rb), json.dumps({k: rb.get(k) for k in ("status", "free_mib_per_card")}))
check("R2 audit N1-3: run_boot: a clean boot with every core measurement -> postprocess_ok True, core_measurements_complete True",
      rb["postprocess_ok"] is True and rb["core_measurements_complete"] is True and rb["core_measurements"]["missing"] == []
      and "headroom" not in rb, json.dumps(rb.get("core_measurements")))


def boom(*a, **k):
    raise RuntimeError("injected post-processing fault")


ns["free_per_card"] = boom
ns["R"]["sections"] = {}
rb = ns["step"]("B1", "nb1_b1_fp8", 92, lambda: ns["run_boot"]("B1", lambda ss: {}, need=("rc2_fp8",)))
check("R2 run_boot: a post-processing exception on a clean boot -> DONE + postprocess_error/tb, every measurement kept, boot_ok",
      rb["status"] == "DONE" and "injected post-processing fault" in rb["postprocess_error"] and "free_per_card" in rb["postprocess_tb"]
      and rb["kv"]["sequences"] == [1, 2] and rb["census"]["kv_tokens"] == 610304 and rb["steady"]["40k@8"] == LW["40k@8"]
      and rb["planned"] == ns["NB1"]["boot_plan"]["B1"] and "error" not in rb and ns["boot_ok"](rb),
      json.dumps({k: rb.get(k) for k in ("status", "postprocess_error")}))
check("R2 audit N1-3: run_boot: the post-processing fault left free MiB per card unmeasured -> postprocess_ok False, "
      "core_measurements_complete False (missing free_mib_per_card), measurements kept",
      rb["postprocess_ok"] is False and rb["core_measurements_complete"] is False and rb["core_measurements"]["missing"] == ["free_mib_per_card"],
      json.dumps(rb.get("core_measurements")))
ns["merlin_boot"] = lambda name, kv, port, timeout, body, need=(), **k: clean_boot("NO_GO_QUALITY", quality_gate="FAIL")
ns["R"]["sections"] = {}
rb = ns["step"]("B1", "nb1_b1_fp8", 92, lambda: ns["run_boot"]("B1", lambda ss: {}, need=("rc2_fp8",)))
check("R2 run_boot: a post-processing exception on a NOT clean boot -> POSTPROCESS_ERROR, status_before kept, measurements kept",
      rb["status"] == "POSTPROCESS_ERROR" and rb["status_before_postprocess"] == "NO_GO_QUALITY" and rb["kv"]["sequences"] == [1, 2]
      and not ns["boot_ok"](rb), json.dumps({k: rb.get(k) for k in ("status", "status_before_postprocess")}))


def raise_after_body():
    ns["LIVE"]["out"] = clean_boot()                       # what watched() / run_boot() hold once the body returned
    raise KeyError("after the body")


ns["R"]["sections"] = {}
rs = ns["step"]("B1", "nb1_b1_fp8", 92, raise_after_body)
rs_err = ns["step"]("A0", "unit_plain_error", 1, boom)
check("R2 step: an exception after the body returned keeps the body record (clean -> DONE + postprocess_error); "
      "an exception with no body record stays ERROR; LIVE cleared",
      rs["status"] == "DONE" and "after the body" in rs["postprocess_error"] and rs["kv"]["sequences"] == [1, 2] and ns["boot_ok"](rs)
      and rs_err["status"] == "ERROR" and "injected post-processing fault" in rs_err["error"] and "postprocess_error" not in rs_err
      and ns["LIVE"]["out"] is None and ns["LIVE"]["rec"] is None, json.dumps({"rs": rs.get("status"), "plain": rs_err.get("status")}))
wo = ns["watched"](lambda ss: {"m": 1}, 30, "unit")([])
check("R2 watched: the body's dict is published to LIVE['out'] when the body returns", ns["LIVE"]["out"] is wo and wo == {"m": 1})
ns["LIVE"]["out"] = None

# the chain: B1 clean with a post-processing fault -> B2 and the B3 pin ladder still run (real step / run_boot / nb1_main)
ns["merlin_boot"] = lambda name, kv, port, timeout, body, need=(), **k: clean_boot()
ns["sha_gate"] = lambda t: (True, "ok")
real_pin_rule = REAL_PIN_RULE
ns["pin_rule"] = lambda base="B2": {"ok": True, "steps_gib": [0.5], "why": "unit", "base": base}
ns["ROWS"]["rc2_fp8"] = [{}]
real_step_ = ns["step"]
chain = []


def st_chain(sec, name, est, fn):
    chain.append(f"{sec}/{name}")
    if sec == "A0":
        r = {"status": "DONE", "ok": True}
        ns["R"]["sections"].setdefault(sec, {})[name] = r
        return r
    return real_step_(sec, name, est, fn)


ns["step"] = st_chain
ns["free_per_card"] = saved["free_per_card"]
ns["free_after_ready"], real_far = boom, ns["free_after_ready"]
ns["R"]["sections"] = {"A0": {"merge": {"ok": True}}}
outc = ns["nb1_main"]()
b1c = ns["_rec"]("B1")
check("R2 chain: B1 clean + a post-processing fault after the core measurements -> B1 DONE (postprocess_error kept, postprocess_ok "
      "False, core complete), B2 and the B3 pin ladder run (not SKIPPED_DEPENDENCY)",
      b1c.get("status") == "DONE" and "postprocess_error" in b1c and b1c["postprocess_ok"] is False and b1c["core_measurements_complete"] is True
      and "B2/nb1_b2_fp8" in chain and any(c.startswith("B3/nb1_b3_fp8_pin") for c in chain)
      and "SKIPPED_DEPENDENCY" not in outc.values(), json.dumps({"chain": chain, "out": outc}))
ns["free_after_ready"] = real_far
ns["free_per_card"] = boom
ns["pin_rule"] = real_pin_rule
chain.clear()
ns["R"]["sections"] = {"A0": {"merge": {"ok": True}}}
outc = ns["nb1_main"]()
b1c, b2c, b3c = ns["_rec"]("B1"), ns["_rec"]("B2"), ns["_rec"]("B3")
check("R2 audit N1-3: chain: B1 clean but the post-processing fault left free MiB per card missing -> B2 SKIPPED (core incomplete); "
      "the real pin rule on B1 fails closed (no free memory of the boot it builds on) -> B3 SKIPPED; B1's measurements kept",
      b1c.get("status") == "DONE" and b1c["core_measurements_complete"] is False and "B2/nb1_b2_fp8" not in chain
      and b2c.get("status") == "SKIPPED_DEPENDENCY" and "core measurements incomplete" in b2c.get("why", "")
      and b3c.get("status") == "SKIPPED_DEPENDENCY" and "pin rule" in b3c.get("why", "") and "B1" in b3c.get("why", "")
      and not any(c.startswith("B3/") for c in chain) and b1c["kv"]["sequences"] == [1, 2], json.dumps({"chain": chain, "out": outc, "b3": b3c})[:500])
ns["step"] = real_step_
for k, v in saved.items():
    ns[k] = v
ns["ABORT"]["why"] = saved_abort

# ---- 12. R2: F validation + this boot's max-concurrency line; the negative B1 fit (stale idle gauge after the reset)
rc = ns["residency_capacity"]
c_b1 = rc(O_B1)
check("R2 F: B1 (fit -5.32, real line) -> F_fit_invalid -5.32, F_log 5.999 -> F 6 from 'server max-concurrency line (this boot)', ok",
      c_b1["ok"] and c_b1["F"] == 6 and c_b1["F_fit_invalid"] == -5.32 and c_b1["F_source"] == "server max-concurrency line (this boot)"
      and abs(c_b1["F_log"]["F_log"] - 5.999) < 0.002 and "-5.32" in c_b1["F_fallback_reason"], json.dumps(c_b1)[:400])
f_px = ns["f_from_log"]({"pool": {"kv_tokens": None, "max_concurrency": 3.92, "max_concurrency_tokens": 131072}, "caps": {}}, 149)
check("R2 F: printed X only (149 / 3.92 - 32 = 6.01) -> ceil(F_log - 0.05) = 6 (nb5_lib rule)", f_px["ok"] and f_px["F"] == 6, json.dumps(f_px))
ns["admission_test"] = lambda port, s, band, n, max_new=0: calls_.append((band, n, max_new)) or {"status": "ALL_RESIDENT"}
ns["PREFILL"].update(tok_s=4400.0)
r_b1 = probe_c6(dict(O_B1, caps={"max_num_seqs": 8}), bands=("40k", "90k"))
check("R2 probes: B1 fixture (fit -5.32) -> log F 6, status MEASURED, probe requests sent, bounds at P 149 / F 6 (40k 9, 90k 5)",
      r_b1["status"] == "MEASURED" and r_b1["F"] == 6 and r_b1["F_source"] == "server max-concurrency line (this boot)" and calls_
      and r_b1["bands"]["40k"]["bound_prompt_only"] == 9 and r_b1["bands"]["90k"]["bound_prompt_only"] == 5, json.dumps({"calls": calls_, "F": r_b1["F"]}))
r_ok = probe_c6(dict(O_B1, kv={"pool_blocks": 149, "fit": {"F": 4.2}}), bands=("40k",))
check("R2 audit N1-4 / N1-6: fit 4.2 without samples / residual (not trusted), ceil 5 within 1 of log F 6 -> the log F 6, fit_check "
      "NOT_TRUSTED, probes run", r_ok["status"] == "MEASURED" and r_ok["F"] == 6 and r_ok["F_source"] == "server max-concurrency line (this boot)"
      and r_ok["capacity"]["fit_check"] == "NOT_TRUSTED" and r_ok["capacity"]["fit_agrees"] is False and calls_, json.dumps(r_ok["capacity"])[:300])
AUD = {}
for nm_, fit_ in (("fit_invalid_-5.32", {"F": -5.32, "samples": 4, "max_abs_residual_blocks": 0.3}), ("fit_nan", {"F": float("nan")}),
                  ("fit_5.99", TF(5.99, 4, 0.5)), ("fit_0.1", TF(0.1, 4, 0.2)), ("fit_0.1_untrusted", {"F": 0.1, "samples": 1}),
                  ("fit_7.5", TF(7.5)), ("fit_6.4_resid_3", TF(6.4, 4, 3.0)), ("fit_none", None)):
    AUD[nm_] = probe_c6(dict(O_B1, kv={"pool_blocks": 149, "fit": fit_}), bands=("40k",)), list(calls_)
a_inv, a_nan, a_ok, a_c, a_cu, a_75, a_r3, a_none = (AUD[k] for k in AUD)
check("R2 audit: F_fit invalid (-5.32 / NaN) + log 6 -> F 6 from the line, probes run",
      all(x[0]["status"] == "MEASURED" and x[0]["F"] == 6 and x[0]["capacity"]["fit_check"] == "INVALID" and x[1] for x in (a_inv, a_nan)))
check("R2 audit: fit 5.99 (samples 4, residual 0.5) + log 6 -> CERTIFIED (PASS): F 6 = the log F, fit_check AGREES, fit a / residual "
      "recorded, probes run", a_ok[0]["status"] == "MEASURED" and a_ok[0]["F"] == 6 and a_ok[0]["capacity"]["status"] == "CERTIFIED"
      and a_ok[0]["capacity"]["fit_check"] == "AGREES" and a_ok[0]["capacity"]["fit_agrees"] is True and a_ok[0]["capacity"]["fit_a"] == 1.0
      and a_ok[0]["capacity"]["fit_max_abs_residual_blocks"] == 0.5 and a_ok[0]["capacity"]["fit_samples"] == 4 and a_ok[1],
      json.dumps(a_ok[0]["capacity"])[:300])
check("R2 audit: fit 0.1 + log 6 -> INCONCLUSIVE_CAPACITY_UNKNOWN with F_FIT_LOG_CONFLICT, F None, no probes (trusted or not); "
      "fit 7.5 (ceil 8) vs 6 -> conflict too", all(x[0]["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and x[0]["F"] is None and not x[1]
                                                 and x[0]["capacity"]["reason_codes"] == ["F_FIT_LOG_CONFLICT"]
                                                 and any("F_FIT_LOG_CONFLICT" in w for w in x[0]["why"]) for x in (a_c, a_cu, a_75))
      and a_c[0]["capacity"]["fit_log_diff_blocks"] == 5 and a_75[0]["capacity"]["fit_log_diff_blocks"] == 2, json.dumps(a_c[0]["why"]))
check("R2 audit: fit 6.4 with max residual 3 blocks (> 2) + log 6 -> the log F 6, fit NOT_TRUSTED (residual), probes run; "
      "no fit + log 6 -> F 6 (fit_check MISSING)", a_r3[0]["F"] == 6 and a_r3[0]["capacity"]["fit_check"] == "NOT_TRUSTED"
      and "residual" in a_r3[0]["capacity"]["fit_not_trusted_reason"] and a_r3[1] and a_none[0]["F"] == 6 and a_none[0]["capacity"]["fit_check"] == "MISSING")
NOLINE = dict(O_B1, pool={})
b_tr = probe_c6(dict(NOLINE, kv={"pool_blocks": 149, "fit": TF(5.2)}), bands=("40k",)), list(calls_)
b_un = probe_c6(dict(NOLINE, kv={"pool_blocks": 149, "fit": TF(5.2, samples=2)}), bands=("40k",)), list(calls_)
b_no = probe_c6(dict(NOLINE, kv={"pool_blocks": 149, "fit": None}), bands=("40k",)), list(calls_)
check("R2 audit: only a trusted fit (5.2, samples 4, no line) -> ceil 6, labelled fit-only; an untrusted fit alone (samples 2) -> "
      "INCONCLUSIVE, no probes; both F missing -> INCONCLUSIVE, no probes",
      b_tr[0]["F"] == 6 and b_tr[0]["capacity"]["fit_check"] == "FIT_ONLY_TRUSTED" and "fit only" in b_tr[0]["F_source"] and b_tr[1]
      and b_un[0]["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and not b_un[1] and any("not trusted" in w for w in b_un[0]["why"])
      and b_no[0]["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and not b_no[1] and b_no[0]["F"] is None, json.dumps([b_un[0]["why"], b_no[0]["why"]])[:400])
p149 = ns["residency_capacity"](dict(O_B1, kv={"pool_blocks": 149, "fit": TF(5.99)}, workers={"num_gpu_blocks": [150]}))
p298 = probe_c6(dict(O_B1, kv={"pool_blocks": 298, "fit": TF(5.99)}, workers={"num_gpu_blocks": [150]}), bands=("40k",)), list(calls_)
check("R2 audit: pool 149 / worker 150 -> MATCH (certified); pool 298 / worker 150 -> MISMATCH -> INCONCLUSIVE, no probes",
      p149["pool_check"] == "MATCH" and p149["ok"] and p298[0]["capacity"]["pool_check"] == "MISMATCH"
      and p298[0]["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and not p298[1], json.dumps(p298[0]["why"]))
ns["due"], saved_due = (lambda sec: True), ns["due"]
ns["window"], saved_win = (lambda port, s, band, n, **k: {"status": "STEADY"}), ns["window"]
lc = ns["nb1_ladder"](8001, None, dict(O_B1, kv={"pool_blocks": 149, "fit": TF(0.1)}))
lm = ns["nb1_ladder"](8001, None, dict(O_B1, workers={"num_gpu_blocks": [300]}))
la = ns["nb1_ladder"](8001, None, dict(O_B1, kv={"pool_blocks": 149, "fit": TF(5.99)}))
ns["due"], ns["window"] = saved_due, saved_win
check("R2 audit: nb1_ladder uses the same rule: fit/line conflict or pool mismatch -> F None, no capacity-sized level (fixed levels "
      "only); fit 5.99 + log 6 -> F 6", lc["_F"] is None and lc["_plan"] == ["8k@3", "8k@8", "40k@5"] and lc["_capacity_status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN"
      and lc["_F_check"] == "CONFLICT" and lm["_F"] is None and lm["_plan"] == ["8k@3", "8k@8", "40k@5"]
      and la["_F"] == 6 and la["_F_check"] == "AGREES", json.dumps({"c": lc["_plan"], "m": lm["_plan"], "a": la["_plan"]}))
bad_line = {"kv_tokens": 513940, "max_concurrency": 3.92, "max_concurrency_tokens": 131072 * 64}
r_n1 = probe_c6(dict(O_B1, pool={}), bands=("40k",))
r_n2 = probe_c6(dict(O_B1, kv={"pool_blocks": 149, "fit": {"F": 45.0}}, pool=bad_line), bands=("40k",))
r_n3 = probe_c6(dict(O_B1, kv={"pool_blocks": None, "fit": {"F": 6.0}}), bands=("40k",))
check("R2 probes: neither valid (fit -5.32 + no line; fit 45 > 40 + line F_log out of range; no pool) -> INCONCLUSIVE_CAPACITY_UNKNOWN, "
      "no probe requests", all(r["status"] == "INCONCLUSIVE_CAPACITY_UNKNOWN" and r["F"] is None and r["F_source"] is None and r["bands"] == {}
                               for r in (r_n1, r_n2, r_n3)) and not calls_ and r_n2["capacity"]["F_fit_invalid"] == 45.0
      and any("max-concurrency" in w for w in r_n1["why"]), json.dumps([r_n1["why"], r_n2["why"], r_n3["why"]])[:500])
ns["admission_test"] = real_adm
check("R2 reset_succeeded: {'success': true} -> True; false / HTTP error / garbage -> False",
      ns["reset_succeeded"]({"ok": True, "body": '{"success":true}'}) and not ns["reset_succeeded"]({"ok": True, "body": '{"success": false}'})
      and not ns["reset_succeeded"]({"ok": False, "error": "x"}) and not ns["reset_succeeded"]({"ok": True, "body": "<html"}))

# B1's REAL gauge readings (server log: the loggers lines print the same kv_cache_usage the gauge exposes)
import re as _re
rows_ = []
for l in FX["server_log"]["alloc_sequence_lines"]:
    m = _re.search(r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%", l)
    rows_.append(("reset_ok" if "Successfully reset prefix cache" in l else None) or (m and (int(m.group(1)), int(m.group(2)), float(m.group(3)) / 100)))
# 8k (1, 2, 4, 8): stale 45.6 % at Running 0 just before the successful reset; 4 resident at 20.8 %; 8 -> 7 running + 1 waiting (excluded)
# 40k (1, 2, 4): stale 26.8 % before its successful resets; 1 / 2 / 4 resident at 7.4 / 16.1 / 38.9 %
g = [r for r in rows_ if isinstance(r, tuple)]
assert [x[:2] for x in g] == [(0, 0), (4, 0), (7, 1), (0, 0), (1, 0), (2, 0), (4, 0), (4, 0), (0, 0)], g


def b1_seqs(reset):
    rb = {"ok": True, "body": '{"success": true}'} if reset else {"ok": True, "body": '{"success": false}'}
    t8, t40 = ns["BANDS"]["8k"] + 64, ns["BANDS"]["40k"] + 64
    s8 = {"status": "MEASURED", "reset_before": rb, "idle_usage": [g[0][2]] * 3, "snapshots": [
        {"requests": 4, "usage": g[1][2], "running": 4, "waiting": 0, "tokens": [t8] * 4},
        {"requests": 8, "usage": g[2][2], "running": 7, "waiting": 1, "tokens": [t8] * 8}]}
    s40 = {"status": "MEASURED", "reset_before": rb, "idle_usage": [g[3][2]] * 3, "snapshots": [
        {"requests": n, "usage": g[i][2], "running": n, "waiting": 0, "tokens": [t40] * n} for i, n in ((4, 1), (5, 2), (7, 4))]}
    return [s8, s40]


f_r1, f_r2 = ns["fit_alloc"](b1_seqs(False), 149, 4096), ns["fit_alloc"](b1_seqs(True), 149, 4096)
sq_r2 = b1_seqs(True)
ns["fit_alloc"](sq_r2, 149, 4096)
check("R2 fit_alloc: B1's real gauge readings: the stale idle (45.6 % / 26.8 % right after a successful reset) gives a negative F "
      "(R1 rule); idle 0 after the confirmed reset gives a valid F (0..40), idle_source + idle_gauge_blocks recorded",
      f_r1["F"] < 0 and f_r1["F_at_a1"] < 0 and 0 <= f_r2["F"] <= 40 and 0 <= f_r2["F_at_a1"] <= 40 and sq_r2[0]["idle_blocks"] == 0
      and sq_r2[0]["idle_gauge_blocks"] == round(0.456 * 149) and "reset_prefix_cache success" in sq_r2[0]["idle_source"],
      json.dumps({"R1_rule": {k: f_r1[k] for k in ("a", "F", "F_at_a1")}, "R2": {k: f_r2[k] for k in ("a", "F", "F_at_a1")}}))

o_r2 = dict(O_B1, kv={"pool_blocks": 149, "fit": f_r2})
c_r2 = ns["residency_capacity"](o_r2)
check("R2 audit: B1's real gauge readings refitted under R2 (idle 0 after the confirmed reset) + B1's real line + worker 150 -> no "
      "conflict with the log F 6: F 6, certified; fit a / residuals recorded",
      c_r2["ok"] and c_r2["F"] == 6 and c_r2["fit_check"] in ("AGREES", "NOT_TRUSTED") and c_r2["fit_a"] == f_r2["a"]
      and c_r2["fit_residuals_blocks"] == f_r2["residuals_blocks"],
      json.dumps({k: c_r2.get(k) for k in ("F_fit", "fit_a", "fit_samples", "fit_max_abs_residual_blocks", "fit_check", "F")}))

n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
(WORK / "unit_results.json").write_text(json.dumps([{"test": n, "ok": ok, "detail": d} for n, ok, d in RESULTS], indent=1))
sys.exit(1 if n_fail else 0)
