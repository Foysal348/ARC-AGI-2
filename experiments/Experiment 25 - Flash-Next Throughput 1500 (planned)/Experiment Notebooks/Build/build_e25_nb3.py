"""E25 NB3 (FN-S3a kernel bench, spec Experiment Review/NB_BUILD_SPEC_2026-10-31.md section 3): a GPU notebook with NO
model server on Kaggle 4xL4 for Qwen3.8-Flash-Next on Merlin (vLLM 0.32.0 fork). Three kernel families, each in its own
subprocess with a timeout (one crash never kills the others), global budget 75 min + hard stop:

  D5  family 1, Marlin MoE routed experts: real Flash-Next experts (one layer of the merged checkpoint, INC W4A16 as
      compressed-tensors pack-quantized, 512 experts top-10, EP2 rank of TP2 + EP), Merlin's fused_marlin_moe at
      T = seqs x (1+K) (+ the PP2 micro-batch / MTP draft counts), synthetic routing (uniform D(T), Zipf, extreme-hot,
      per-request, adversarial reference case), fp32 dequantised reference, derived weight-traffic GB/s, the PP2
      split-read test (two micro-batch calls vs one combined call on the same ids, ncu bytes per arm in separate runs),
      real NB1 routing ids replayed when attached, MERLIN_MARLIN_MOE_CFG sweep, EP rank 1 spot check; in two benches
      (core = T16 / T4 / T8 first, rest = everything else last; nb3_common.D5_PRIORITY);
  D7  family 3, world-2 collectives on pairs (0,1) and (2,3) concurrently, then (0,1) alone: NCCL with the RC2 env
      (NCCL_P2P_DISABLE=1, NCCL_PROTO / NCCL_ALGO unset, protocol choice recorded from NCCL_DEBUG) vs the E23 P5
      host-staged all-reduce (native + INT8) built from source in the Merlin venv; LL128: UNTESTED (not run, NCCL docs
      corruption risk: forcing NCCL_PROTO=LL128 on an unsupported platform can corrupt data); a D7 arm passes only with
      both pairs x both ranks x every mandatory check, and the pair concurrency is checked per size;
  D4  family 2, dense GEMM + vocab / draft heads: Merlin's sm_89 skinny-GEMM plan path (CuTe DSL) vs the cuBLAS
      fallback on the Flash-Next TP2 shapes, missing plans, the patched (8,9) dispatch path, and the MTP draft head at
      DRAFT_HEAD bf16 / int8 / int4.
Order (N3-19): SHA gate -> bench env (fail closed) -> D5 core -> D7 build / pairs / solo -> D4 -> D5 rest. Every GPU
bench re-checks the global SHA gate first (N3-13). Failure policy: UNSUPPORTED_SM89 / a failed bench is recorded and the
other arms run; an illegal memory access / Xid / any GPU fault line / GPUs still holding memory stops the notebook.
Promotion (N3-2 / N3-11 / N3-22): one rule (nb3_common.eligibility) for the NB4 ranking and the scorecard: verified
bench status DONE + correctness exactly PASS + not info / hypothesis + saved > 0; everything else is blocked / diagnostic
with its raw numbers kept, score 0.
Outputs: d5/d7/d4 JSON, nb3_tables.json (tables + ranked NB4 candidates), lever_scorecard.json (PLAN_1500 v3.2 4.1,
written after every bench), the fallback submission.

Reuse (verbatim, the RC2 way): the RC2 builder's own cell texts are rebuilt by executing ITS assignment statements
(ast, no file written) -> CELL1 (config / inputs / lab module, once()-patched), CELL2 (host facts, merge_ckpt, sha
gate, merlin_install, merlin_fn_server; the stock fallback dropped), the A0 merge of CELL3, CELL4 (step runner); E23's
fault gate (XID_RESET, gpu_health, fault_gate, _killpg); E23 prep P5 bench_host_ar.py (once()-patched for world 2 at
the Flash-Next sizes) + its sources; E23 prep P4 stats / bench / metrics (cut into nb3_common).

Run: python build_e25_nb3.py  ->  ../Q27-XHIGH-E25_NB3_KERNELS.ipynb  (+ nb3_export/ and nb3_code_export.py for tests)
"""
import ast
import hashlib
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
E25 = HERE.parents[1]
E13 = Path(r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)")
E23 = Path(r"D:/ARC AGI 2/experiments/Experiment 23 - Six levers vLLM 0.32 (planned)")
RD = lambda p: Path(p).read_text(encoding="utf-8").replace("\r\n", "\n")  # noqa: E731
RC2_BUILDER = RD(E13 / "Experiment Notebooks" / "Build" / "build_e13_rc2.py")
E23_BUILDER = RD(E23 / "Experiment Notebooks" / "Build" / "build_e23_notebook.py")
LAB = RD(E23 / "src" / "e23_lab.py")
PARTS = json.loads(RD(E13 / "prep" / "merlin_fn_ckpt" / "kernels" / "parts.json"))
P4_BENCH = RD(E23 / "prep" / "P4_sm89_fp8_gemm" / "bench_fp8_gemm.py")
P5_DIR = E23 / "prep" / "P5_host_allreduce"
P5_BENCH = RD(P5_DIR / "bench_host_ar.py")
P5_SRC = {n: RD(P5_DIR / "src" / n) for n in ("build_host_ar.py", "host_ar.cu", "host_ar_binding.cpp", "host_ar_check.h",
                                              "host_ar_layout.h")}
OUT_NB = HERE.parent / "Q27-XHIGH-E25_NB3_KERNELS.ipynb"
PUSHED = set()      # shas of NB3 versions that ran on Kaggle (never overwrite one; rebuilds go to *_REBUILD)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")


def cut(text, start, end, label):
    assert text.count(start) == 1, f"{label}: start x{text.count(start)}"
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j]


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: x{text.count(old)}"
    return text.replace(old, new)


def times(text, old, new, n, label):
    assert text.count(old) == n, f"{label}: x{text.count(old)} (want {n})"
    return text.replace(old, new)


# ------------------------------------------------------------------ the RC2 cell texts, built by RC2's own statements
_WANT = ("E23_CELL2", "E23_CELL4", "E23_CELL5", "GATES", "RUN_AND_SHIM", "MERLIN_INSTALL", "STEP_RUNNER", "PLE",
         "CELL1", "CELL2", "CELL3", "CELL4")
_ns = {"cut": cut, "once": once, "E23_BUILDER": E23_BUILDER, "LAB": LAB, "PARTS": PARTS, "json": json}
for _node in ast.parse(RC2_BUILDER).body:
    if isinstance(_node, ast.Assign) and len(_node.targets) == 1 and isinstance(_node.targets[0], ast.Name) \
            and _node.targets[0].id in _WANT:
        exec(compile(ast.Module(body=[_node], type_ignores=[]), "build_e13_rc2.py", "exec"), _ns)
RC2 = {k: _ns[k] for k in _WANT}
assert all(isinstance(RC2[k], str) for k in _WANT if k != "PLE")

# ---- CELL 1: RC2 CELL1 with the NB3 identity, budget, output names
C1 = RC2["CELL1"]
C1 = once(C1, 'R = {"what": "E13 RC2 Flash-Next on Merlin",',
          'R = {"what": "E25 NB3 kernel bench (D5 Marlin MoE, D7 world-2 all-reduce, D4 sm_89 GEMM plans + heads)",', "what")
C1 = once(C1, 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # two cold Merlin boots + ladder + 90k/116k census + long rows',
          'BUDGET = int(os.environ.get("NB3_BUDGET_SEC") or 75 * 60)     # NB3: install + sha + three benches, no server boot', "budget")
C1 = once(C1, 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/rc2"))', 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb3"))', "out")
C1 = once(C1, 'OUT / "rc2_results.json.tmp"', 'OUT / "nb3_results.json.tmp"', "save tmp")
C1 = once(C1, 'OUT / "rc2_results.json")', 'OUT / "nb3_results.json")', "save")
C1 = once(C1, 'print("E13 RC2 diagnostic: competition rerun -> fallback submission.json only")',
          'print("E25 NB3 kernel bench: competition rerun -> fallback submission.json only")', "rerun msg")
C1 = once(C1, 'if not FAKE and not os.environ.get("RC2_CPU_KIT") and len(',
          'if not FAKE and not (os.environ.get("RC2_CPU_KIT") or os.environ.get("NB3_CPU_KIT")) and len(', "cpu kit")
C1 = once(C1, 'log("RC2 | budget"', 'log("NB3 | budget"', "log")
CELL1 = C1

# ---- CELL 2: RC2 CELL2 up to the stock-vLLM fallback (NB3 never serves)
CELL2 = cut(RC2["CELL2"], "# ====", "\nSTOCK = None\n", "rc2 cell2") + "\nsave()\n"
# ---- CELL 3: RC2's A0 (host facts, merge + background sha256, host memory); no prompts / tokenizer in NB3
CELL3 = ("# ==================================================================== CELL 3 — A0: host facts, checkpoint merge (RC2)\n"
         + cut(RC2["CELL3"], 'R["sections"]["A0"] = {}', "TOKDIR = TMP", "rc2 A0"))
# ---- CELL 4: RC2 CELL4 verbatim (HELP + the E23 step runner)
CELL4 = RC2["CELL4"]
# ---- E23's fault gate (Xid classes, GPU health probe) + _killpg; the health probe runs in the bench python
FAULT = cut(RC2["E23_CELL5"], "XID_RESET = {", "\n\ndef taint_check", "e23 fault gate")
FAULT = once(FAULT, 'h = subprocess.run([STOCK["py"], "-c", HEALTH_CODE], capture_output=True, text=True, timeout=300, env=STOCK["env"])',
             'h = subprocess.run([BENCH["py"], "-c", HEALTH_CODE], capture_output=True, text=True, timeout=300, env=bench_env("0,1,2,3"))',
             "health python")
KILLPG = cut(RC2["E23_CELL5"], "def _killpg(p):", "\n\n\n# ---- #2", "e23 _killpg")

# ------------------------------------------------------------------ embedded bench files
P4_HELPERS = "\n\n\n".join([cut(P4_BENCH, "def stats(xs):", "\n\n\n# ---", "p4 stats"),
                            cut(P4_BENCH, "def bench(torch, fns, warmup, iters, use_graph):", "\n\n\n# ---", "p4 bench"),
                            cut(P4_BENCH, "def metrics(torch, out, ref):", "\n\n\ndef run_child", "p4 metrics")])
COMMON = once(RD(HERE / "nb3_common.py"), "# __P4_HELPERS__", P4_HELPERS, "p4 helpers")

# P5 bench: world 2 at the Flash-Next message sizes; the CTA sweep becomes named NCCL arms (none by default: the
# forced-LL128 arm was dropped); a numeric check of every group's NCCL result ('n'); group timestamps; no /dev/shm sweep
# by one parent while the other pair's parent is still creating handles (the notebook cleans up after both).
B = P5_BENCH
B = once(B, "SIZES = [65536, 245760, 491520, 983040, 1966080, 3932160]\nDECODE = 983040\n",
         'SIZES = json.loads(os.environ.get("NB3_AR_SIZES") or "[65536, 245760, 491520, 983040, 1966080, 3932160]")  # NB3 (E25): Flash-Next sizes\n'
         'DECODE = int(os.environ.get("NB3_AR_DECODE") or 983040)  # NB3: the RC2 micro-batch payload (block sweep, step c)\n'
         'NB3_ITERS = json.loads(os.environ.get("NB3_AR_ITERS") or "null")  # NB3: [iters, reps, calls, replays] for the latency steps\n',
         "p5 sizes")
B = times(B, "iters, reps, calls, replays = (50, 3, 20, 5) if c.a.quick else (200, 5, 50, 20)",
          "iters, reps, calls, replays = tuple(NB3_ITERS) if NB3_ITERS else ((50, 3, 20, 5) if c.a.quick else (200, 5, 50, 20))",
          2, "p5 iters")
B = once(B, '        plan.append(("main_ab", ",".join(s for s in ("a", "b") if s in want), {}, 900))',
         '        plan.append(("main_ab", ",".join(["n"] + [s for s in ("a", "b") if s in want]), {}, 900))  # NB3: + NCCL numeric check',
         "p5 main_ab n")
B = once(B, '''        for x in NCCL_CTAS:
            plan.append((f"nccl_ctas{x}", "b_nccl", {"NCCL_MIN_CTAS": x, "NCCL_MAX_CTAS": x}, 240))''',
         '''        for gname, genv in json.loads(os.environ.get("NB3_AR_NCCL_ARMS") or "{}").items():  # NB3: named NCCL arms (CTA sweep: slower, A7)
            plan.append((gname, "n,b_nccl", genv, 240))''', "p5 arms")
B = once(B, '    return {"status": status, "rcs": rcs, "sec": round(time.time() - t0, 1), "env": extra_env,',
         '    return {"status": status, "rcs": rcs, "sec": round(time.time() - t0, 1), "env": extra_env,\n'
         '            "t0": round(t0, 2), "t1": round(time.time(), 2),  # NB3: wall-clock window (pair overlap)',
         "p5 group times")
B = once(B, '        for f in Path("/dev/shm").glob("p5har_*"):  # names are unlinked after attach; this only catches crashes',
         '        for f in ([] if os.environ.get("NB3_AR_NO_SHM_SWEEP") else Path("/dev/shm").glob("p5har_*")):  # NB3: two parents run at once',
         "p5 shm sweep")
NB3_N = '''def nb3_nccl_check(c):
    """NB3 (E25): numeric check of THIS group's NCCL all-reduce (the group's own NCCL env) at the NB3 sizes: bf16 and
    fp32, randn + cancel patterns; identical on all ranks and |out - exact FP64 sum| <= world x ulp x sum|addends| per
    element; bit-equality with the ordered FP32 reference is recorded, not required."""
    torch, P = c.torch, c.part
    res = {"cases": []}
    ok_ident = ok_bound = True
    for dname, dt, ulp in (("bfloat16", torch.bfloat16, 2.0 ** -8), ("float32", torch.float32, 2.0 ** -23)):
        for nbytes in SIZES:
            for pat in ("randn", "cancel"):
                n = nbytes // torch.tensor([], dtype=dt).element_size()
                seed = 7000 + (nbytes % 997) + (13 if pat == "cancel" else 0)
                xs = [gen(torch, c.dev, seed, j, n, dt, pat) for j in range(c.world)]
                y = xs[c.rank].clone()
                c.nc.all_reduce(y, y)
                torch.cuda.synchronize()
                exact = torch.stack([x.double() for x in xs]).sum(0)
                add = torch.stack([x.double().abs() for x in xs]).sum(0)
                err = (y.double() - exact).abs()
                within = bool((err <= add * (ulp * c.world) + 1e-30).all().item())
                gl = [torch.empty_like(y) for _ in range(c.world)]
                c.dist.all_gather(gl, y)
                ident = all(bool(torch.equal(bits(torch, gl[0]), bits(torch, g))) for g in gl[1:])
                beq, nd = bit_equal(torch, y, ordered_ref(torch, xs, dt))
                res["cases"].append({"dtype": dname, "bytes": nbytes, "pattern": pat, "within_bound": within,
                                     "identical_all_ranks": ident, "bitexact_vs_ordered_ref": beq, "ref_diff_elems": nd,
                                     **err_stats(torch, y, exact, add)})
                ok_ident &= ident
                ok_bound &= within
    P["n"] = res
    P.check("n_nccl_identical_all_ranks", ok_ident)
    P.check("n_nccl_within_rounding_bound", ok_bound)


STEPS = {"a": step_a, "b": step_b, "b_nccl": step_b_nccl, "c": step_c, "d": step_d}'''
B = once(B, 'STEPS = {"a": step_a, "b": step_b, "b_nccl": step_b_nccl, "c": step_c, "d": step_d}',
         NB3_N.replace('STEPS = {"a": step_a, "b": step_b, "b_nccl": step_b_nccl, "c": step_c, "d": step_d}',
                       'STEPS = {"a": step_a, "b": step_b, "b_nccl": step_b_nccl, "c": step_c, "d": step_d, "n": nb3_nccl_check}'),
         "p5 n step")
B = once(B, "    env.setdefault(\"NCCL_DEBUG_SUBSYS\", \"INIT,ENV,GRAPH\")",
         "    env.setdefault(\"NCCL_DEBUG_SUBSYS\", \"INIT,ENV,GRAPH,TUNING\")  # NB3: + TUNING (the protocol NCCL picks)",
         "p5 nccl subsys")
B = once(B, '        r = {}\n        r["nccl_eager"] = collective_med(c, time_eager(c, lambda: c.nc.all_reduce(x, y), iters, reps))',
         '        r = {}\n        _tm0 = time.monotonic()  # NB3 (N3-17): timed region of this size (pair concurrency check)\n'
         '        r["nccl_eager"] = collective_med(c, time_eager(c, lambda: c.nc.all_reduce(x, y), iters, reps))', "p5 t_mono nccl start")
B = once(B, '        r["torch_nccl_eager"] = collective_med(c, time_eager(c, lambda: c.dist.all_reduce(x), iters, reps))\n        res[str(nbytes)] = r',
         '        r["torch_nccl_eager"] = collective_med(c, time_eager(c, lambda: c.dist.all_reduce(x), iters, reps))\n'
         '        r["t_mono"] = {"nccl": [round(_tm0, 4), round(time.monotonic(), 4)]}  # NB3 (N3-17)\n        res[str(nbytes)] = r',
         "p5 t_mono nccl end")
B = once(B, '        r = res["latency"][str(nbytes)]\n        r["host_eager"] = ',
         '        r = res["latency"][str(nbytes)]\n        _th0 = time.monotonic()  # NB3 (N3-17)\n        r["host_eager"] = ', "p5 t_mono host start")
B = once(B, '        r["host_status_ok"] = c.status_ok(f"b:{nbytes}")',
         '        r.setdefault("t_mono", {})["host"] = [round(_th0, 4), round(time.monotonic(), 4)]  # NB3 (N3-17)\n'
         '        r["host_status_ok"] = c.status_ok(f"b:{nbytes}")', "p5 t_mono host end")
P5_BENCH_NB3 = B

FILES = {"nb3_common.py": COMMON, "nb3_moe_bench.py": RD(HERE / "nb3_moe_bench.py"), "nb3_humming.py": RD(HERE / "nb3_humming.py"),
         "nb3_gemm_bench.py": RD(HERE / "nb3_gemm_bench.py"), "bench_host_ar.py": P5_BENCH_NB3}
FILES.update({f"p5_src/{n}": t for n, t in P5_SRC.items()})
for _n, _t in FILES.items():
    if _n.endswith(".py"):
        compile(_t, _n, "exec")

MD0 = """# E25 NB3: Flash-Next kernel bench on 4×L4 (Merlin, vLLM 0.32.0 fork), no model server

Screening notebook (PLAN_1500 v3.2 §4.1): three kernel families, each in its own subprocess with a timeout, global
budget 75 min + hard stop. All numbers are **microbenchmarks**; projections against RC2's decode tick (step_ms 30-48 ms,
8 streams, K3, PP2 micro-batch T = ceil(streams/2) × (1+K)) are labelled MICROBENCH_PROJECTED and never count as an
engine gain until NB4 confirms them.

- **D5** Marlin MoE (family 1): real Flash-Next routed experts (one layer, INT4 g128, EP2 rank = 256 local experts),
  Merlin's `fused_marlin_moe`, synthetic routing (uniform D(T), Zipf, extreme-hot 16 experts, per-request) + an
  adversarial reference case, preflight (signatures + tiny INT4 parity), dequant cross-check and a T=1 / T=4 first-call
  gate before any sweep, fp32 reference + elementwise BF16 ULP checks, derived GB/s, the PP2 split-read test (two T/2
  calls vs one T call, same ids, C 8/16/24/32 at K3; DRAM bytes by ncu with one child run per arm when the counters are
  allowed, else DRAM_BYTES_UNVERIFIED and derived bytes only), NB1's real routing ids replayed when attached,
  `MERLIN_MARLIN_MOE_CFG` sweep, EP rank 1 spot check. Two benches in priority order: core (T16, T4, T8) before D7,
  rest (everything else, ncu last) at the end.
- **C6-humming** (structural track, research, never ranked): right after the T16 Marlin block, Merlin's humming MoE
  path on the same experts: W4 (the checkpoint's own INT4, kernel-vs-kernel control), W3 / W2 group 128 and W3 group 64
  (SYNTHETIC round-to-nearest requant: speed and bytes only), T 8-64, each checked against its own quantized weights;
  graph / eager time, ratios vs Marlin W4 (gate W3 <= 0.8x, W2 <= 0.6x; capacity reading <= 1.3x), JIT compile time,
  CUDA-graph capture; `import humming` failing = UNAVAILABLE.
- **D7** world-2 all-reduce (family 3): NCCL with the RC2 env (P2P off, protocol left to NCCL and recorded) vs the
  E23 P5 host-staged all-reduce (native, INT8), pairs (0,1) and (2,3) concurrently, then (0,1) alone; numeric checks on
  both pairs x both ranks; per-size timed-region overlap of the two pairs (else CONCURRENCY_UNVERIFIED);
  LL128: UNTESTED (not run, NCCL docs corruption risk).
- **D4** dense GEMM + heads (family 2): sm_89 skinny-GEMM plan path vs cuBLAS on the TP2 shapes, the (8,9) dispatch
  path patched in, and the draft head at bf16 / int8 / int4.

Promotion: a candidate is ranked only when its bench's verified status is DONE, its correctness is exactly PASS and
it saves time; every other row stays in the tables as BLOCKED / DIAGNOSTIC with score 0 (no microbench saving is an
engine gain: `engine_confirmed_saved_ms` stays null until NB4; savings are not additive under PP2 overlap).

Outputs: `nb3_results.json`, `d5_marlin_moe.json` + `d5_marlin_moe_rest.json`, `d7_*.json`, `d4_gemm.json`,
`nb3_tables.json` (tables + ranked NB4 candidates), `lever_scorecard.json`. A competition rerun writes the fallback
submission only.
"""

CELL5 = r'''# ==================================================================== CELL 5 — the NB3 benches (new code; E23 fault gate verbatim)
import signal
NB3_DIR = TMP / "nb3"
NB3_DIR.mkdir(parents=True, exist_ok=True)
import base64, zlib
# the bench files (Build/nb3_export/ in the repo, plain text) travel zlib + base64 to keep the notebook < 400 kB; every
# file is sha256-checked against the builder's value and a plain copy is kept in OUT/nb3_files for the audit trail
NB3_FILES = json.loads(zlib.decompress(base64.b64decode(__NB3_FILES_Z__)).decode("utf-8"))
NB3_FILES_SHA = __NB3_FILES_SHA__
_bad = sorted(k for k, v in NB3_FILES.items() if hashlib.sha256(v.encode("utf-8")).hexdigest() != NB3_FILES_SHA.get(k))
if _bad or set(NB3_FILES) != set(NB3_FILES_SHA):
    raise RuntimeError(f"NB3 embedded files fail their sha256: {_bad}")
for _root in (NB3_DIR, OUT / "nb3_files"):
    for _rel, _txt in NB3_FILES.items():
        (_root / _rel).parent.mkdir(parents=True, exist_ok=True)
        (_root / _rel).write_text(_txt, encoding="utf-8", newline="\n")
R["setup"]["nb3_files"] = NB3_FILES_SHA
sys.path.insert(0, str(NB3_DIR))
import nb3_common as C
BENCH = {"py": sys.executable, "env": dict(os.environ), "plan": None, "plan_ok": False, "procs": set()}
TABLES = {}
BENCH_STATUS = {}            # VERIFIED bench statuses (C.verified_bench_status) by bench key: D5, D5_rest, D7, D7_solo, D4
D5R = {}                     # the two D5 result files (core, rest)
SEC_D5_CORE, SEC_D5_REST, SEC_D4, SEC_PAIRS, SEC_SOLO = (int(os.environ.get(k) or v) for k, v in
                                                         (("NB3_D5_CORE_SEC", 720), ("NB3_D5_REST_SEC", 540), ("NB3_D4_SEC", 900),
                                                          ("NB3_D7_PAIRS_SEC", 900), ("NB3_D7_SOLO_SEC", 360)))
FLOOR = (lambda v: min(v, 20)) if FAKE else (lambda v: v)     # local harness only: short floors for the timeout tests
# Real router top-k ids written by NB1 (PLAN v3.6): attach NB1's output as this dataset, or set NB3_ROUTING_FILE. Absent =
# "real routing unavailable" (recorded); D5 then keeps the synthetic uniform / Zipf / extreme-hot cases only.
ROUTING_DIR = INP / "datasets" / "foysalemonshanto" / "e25-nb1-routing-ids"
ROUTING_FILE = next((p for p in ([Path(os.environ["NB3_ROUTING_FILE"])] if os.environ.get("NB3_ROUTING_FILE") else []) +
                     [ROUTING_DIR / n for n in C.ROUTING_FILE_NAMES] if p.exists()), None)
R["setup"]["routing_file"] = {"path": str(ROUTING_FILE) if ROUTING_FILE else None, "looked_in": str(ROUTING_DIR),
                              "names": list(C.ROUTING_FILE_NAMES), "schema": C.ROUTING_SCHEMA,
                              "status": "attached" if ROUTING_FILE else "real routing unavailable"}
KITS_DIR = INP / "datasets" / "foysalemonshanto" / "q27-e23-sm89-kernels"     # optional: E23's built p5_host_ar.so (fallback only)
P5 = {"so": None, "source": None}
# P5 at world 2, from its source (E23 prep/P5_host_allreduce/src, embedded here): host_ar_binding.cpp:112 TORCH_CHECK
# world in {2, 4, 8}; host_ar.cu:576-587 / :616-621 kernels instantiated for 2 / 4 / 8 (launch_w<2>); tensors flat,
# contiguous CUDA f32 / bf16 / f16 with 16-byte aligned pointers (host_ar_binding.cpp:241-257), so a [T, 2560] bf16
# hidden-state block is a valid input. The runtime probe below re-checks the loaded .so (info + cover_check at world 2).
P5_SOURCE_FACTS = {"world_check": "host_ar_binding.cpp:112 world == 2 || 4 || 8", "kernels": "host_ar.cu:576-587,616-621 <2>/<4>/<8>",
                   "layout": "host_ar_binding.cpp:241-257 contiguous CUDA f32/bf16/f16, 16 B aligned, numel-based (flat)",
                   "e23_numbers": "world 4 only (do not transfer to world 2)"}
''' + FAULT + "\n\n" + KILLPG + r'''


def _kill_benches():
    ABORT.update(why="HARD_STOP reached: watchdog killed the benches")
    for p in list(BENCH["procs"]):
        _killpg(p)


L.start_watchdog(HARD_STOP, _kill_benches)


def bench_env(gpus, extra=None):
    e = dict(BENCH["env"])
    e.update(CUDA_VISIBLE_DEVICES=str(gpus), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    for k, v in (extra or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = str(v)
    return e


def start_bench(name, argv, gpus, extra=None):
    fh = open(LOGS / f"bench_{name}.log", "w")
    p = subprocess.Popen([BENCH["py"]] + [str(x) for x in argv], env=bench_env(gpus, extra), stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=str(NB3_DIR))
    BENCH["procs"].add(p)
    return {"name": name, "p": p, "fh": fh, "t0": time.time(), "gpus": gpus, "extra": extra or {}}


def wait_benches(jobs, timeout):
    """Every job under one deadline (its own timeout, never past the hard stop); a job past it is killed (process group)."""
    deadline = time.time() + max(30, min(timeout, left() - 60))
    out = {}
    for j in jobs:
        rc, to = None, False
        try:
            rc = j["p"].wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            _killpg(j["p"])
            try:
                j["p"].kill()                       # the leader too (and the only kill where killpg does not exist)
            except Exception:
                pass
            try:
                j["p"].wait(30)
            except Exception:
                pass
            rc, to = -9, True
        BENCH["procs"].discard(j["p"])
        j["fh"].close()
        txt = (LOGS / f"bench_{j['name']}.log").read_text(errors="replace")
        out[j["name"]] = {"rc": rc, "timeout": to, "sec": round(time.time() - j["t0"], 1), "gpus": j["gpus"],
                          "env_overrides": {k: (None if v is None else str(v)[:200]) for k, v in j["extra"].items()},
                          "log_tail": txt[-2500:], "_log": txt}
    return out


def run_bench(name, argv, gpus, timeout, extra=None):
    return wait_benches([start_bench(name, argv, gpus, extra)], timeout)[name]


def released(gpus):
    rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus, free_mib=150)
    if rel is None and NVSMI_OK:
        rel = False
    if rel is False:
        ABORT["why"] = f"GPUs {gpus} still hold memory after a bench"
    return rel


def finish(key, r, jpath, gpus, tkey=None, table=None, res=None):
    """Bench record: JSON result, fault gate on its log, GPU release, then the VERIFIED status (N3-22,
    C.verified_bench_status: rc, timeout, GPU release, file status, exit_code, every check); tables + the scorecard are
    rewritten after every bench (spec addendum). Failure policy (N3-19): any GPU fault line in the log (illegal memory
    access, Xid, ...) or GPUs still holding memory stops the notebook (ABORT); a failed / unsupported bench does not."""
    log_txt = r.pop("_log", "")
    if res is None:
        res = json.loads(Path(jpath).read_text()) if jpath and Path(jpath).is_file() else {}
    o = {"bench": {k: v for k, v in r.items()}, "result_file": str(jpath), "status_in_file": res.get("status"),
         "exit_code_in_file": res.get("exit_code"), "checks": res.get("checks"), "errors": (res.get("errors") or [])[:6],
         "host_after": host_mem(), "gpus_end": L.gpu_mem()}
    fg = fault_gate([log_txt], f"nb3_{key}")
    o["gpu_released"] = released(gpus)
    o["status"], o["status_reasons"] = C.verified_bench_status(r, res, o["gpu_released"], allow_unknown_release=bool(FAKE))
    if fg:
        o.update(status="NO_GO_GPU_FAULT", gpu_fault_class=fg)
        ABORT["why"] = ABORT["why"] or f"NB3 failure policy: GPU fault ({fg}) in the {key} bench log: the notebook stops"
    BENCH_STATUS[key] = o["status"]
    if tkey and table is not None:
        TABLES[tkey] = table
    nb3_report()
    return o


def nb3_report():
    st = dict(BENCH_STATUS)
    try:
        cand = C.rank_candidates(TABLES, status=st)
        sc = C.scorecard(TABLES, st)
        C.write_json(OUT / "nb3_tables.json", {"tables": TABLES, "ar": AR, "t_sets": C.t_sets(), "nb4_candidates": cand,
                                              "bench_status": st, "families": C.FAMILY, "routing_note": C.ROUTING_NOTE,
                                              "promotion_rule": C.eligibility.__doc__.strip(), "additivity": C.ADDITIVITY_NOTE,
                                              "label": "MICROBENCH_PROJECTED; never an engine gain until NB4 confirms"})
        C.write_json(OUT / "lever_scorecard.json", sc)
        R["nb4_top"] = [{"id": c["id"], "saved_ms_per_tick": c["saved_ms_per_tick"], "corr": c["correctness"],
                         "eligibility": c["eligibility"]} for c in (cand["targets"][0]["ranked"] if cand["targets"] else [])[:8]]
        R["bench_status"] = st
        R["scorecard"] = [{k: row.get(k) for k in ("lever", "bench_status", "promotion", "correctness", "confidence", "adoption",
                                                   "score", "microbench_saved_ms", "engine_confirmed_saved_ms")} for row in sc["rows"]]
        R["floor_rc2_point"] = {k: v for k, v in sc["floor_decomposition_rc2_point"].items() if k not in ("dense", "rc2_fit")}
    except Exception as e:
        R["errors"].append(f"nb3_report: {type(e).__name__}: {e}"[:400])
    save()


def require_sha(what):
    """N3-13: ONE global gate before EVERY GPU bench (D5 core / rest, D7 build / pairs / solo, D4): the checkpoint's
    full sha256 must have passed. On failure: no subprocess, ABORT set, the refusal recorded as evidence."""
    sg = R["setup"].get("sha_gate") or {}
    if sg.get("ok") is True:
        return None
    ev = {"what": what, "sha_gate": sg or "not run", "t": time.strftime("%H:%M:%S")}
    R["setup"].setdefault("sha_gate_refusals", []).append(ev)
    ABORT["why"] = ABORT["why"] or f"SHA_GATE_FAILED before {what} ({sg.get('why') or 'gate not run'}): no GPU bench runs"
    save()
    return {"status": "SKIPPED_SHA_GATE", "why": f"{what}: the checkpoint sha256 gate has not passed", "evidence": ev}


def require_env(what):
    """N3-23: a bench runs only in a verified bench env (the RC2 plan; EXPLORATORY_ENV only inside the CPU kit)."""
    if BENCH.get("plan_ok") is True:
        return None
    return {"status": "SKIPPED_NO_BENCH_ENV", "why": f"{what}: no verified bench env ({BENCH.get('plan')})"}


def gates(what):
    return require_sha(what) or require_env(what)


def _st(sec, name):
    return ((R["sections"].get(sec) or {}).get(name) or {}).get("status")


# ---------------------------------------------------------------- setup steps
def s_sha():
    ok, why = sha_gate(min(1500, max(60, left() - 900)))
    R["setup"]["sha_gate"] = {"ok": ok, "why": why}
    if not ok:                                   # N3-13: no GPU bench after a failed / unfinished sha256
        ABORT["why"] = ABORT["why"] or f"SHA_GATE_FAILED: {why}"
    return {"sha_ok": ok, "why": why, "page_cache": drop_ckpt_cache(), "status": "DONE" if ok else "SHA_GATE_FAILED"}


def drop_ckpt_cache():
    """Spec section 2 host-RAM rule: drop the page cache of every merged file's resolved Model-mount target after the
    sha pass (no server boots here, so Merlin's 69 GiB floor gate is recorded only)."""
    before, n, errs = host_mem(), 0, []
    if not hasattr(os, "posix_fadvise"):
        return {"skipped": "no posix_fadvise on this platform", "host": before}
    for p in sorted(CKPT.iterdir()) if CKPT.is_dir() else []:
        try:
            fd = os.open(str(p.resolve()), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
        except Exception as e:
            errs.append(f"{p.name}: {e}"[:120])
    after = host_mem()
    floor = (after.get("avail_mib") or 0) >= (69 + 4) * 1024 if after.get("avail_mib") is not None else None
    return {"files": n, "errors": errs[:5], "host_before": before, "host_after": after, "merlin_floor_69_plus_4_gib": floor}


def s_plan():
    """The exact RC2 FP8 launch env (Merlin launch.build + host_phase + RC2's overrides, via merlin_fn_server) without
    starting a server; every bench runs in the Merlin venv python with it (+ CUDA_VISIBLE_DEVICES per bench).
    N3-23: on the GPU run a plan failure (missing scales, failed host_phase, broken paths, ...) FAILS CLOSED: no bench
    env, every bench step SKIPPED_NO_BENCH_ENV. Only the CPU kit (NB3_CPU_KIT=1, no checkpoint attached) may continue
    with a minimal env, labelled EXPLORATORY_ENV."""
    BENCH["plan_ok"] = False
    if FAKE:
        BENCH.update(py=sys.executable, env=dict(os.environ), plan="FAKE (local harness)", plan_label="FAKE", plan_ok=True)
        return {"plan": BENCH["plan"], "plan_label": "FAKE"}
    kit = bool(os.environ.get("NB3_CPU_KIT"))
    eng = merlin_install()
    kvv = eng["kv"]
    hc = L.help_check
    L.help_check = lambda *a, **k: {"ran": False, "why": "NB3: no server, no --help check"}
    try:
        srv = merlin_fn_server("nb3_plan_fp8", "fp8", 8000, KV_FP8)
        env, src, label = dict(srv.env), "RC2 FP8 plan (launch.build + host_phase + RC2 overrides)", "RC2_FP8_PLAN"
    except Exception as e:
        if not kit:
            BENCH["plan"] = f"FAILED: {type(e).__name__}: {e}"[:400]
            R["setup"]["bench_env_error"] = BENCH["plan"]
            L.help_check = hc
            raise RuntimeError(f"BENCH_ENV_FAIL_CLOSED: the RC2 FP8 plan failed ({type(e).__name__}: {e}); no bench runs "
                               f"on a guessed env (N3-23)"[:700])
        label = "EXPLORATORY_ENV"
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "LD_PRELOAD")}
        env.update(LD_PRELOAD=kvv["MERLIN_SHIM"], CUDA_HOME=eng["cuda_home"], LIBRARY_PATH=f"{SHIM}:{env.get('LIBRARY_PATH', '')}",
                   PATH=f"{kvv['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{env.get('PATH', '')}", NCCL_P2P_DISABLE="1",
                   VLLM_SKIP_P2P_CHECK="1", CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TRITON_CACHE_DIR=f"{TMP}/merlin-triton")
        for k in ("NCCL_P2P_LEVEL", "NCCL_PROTO", "NCCL_ALGO", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"):
            env.pop(k, None)
        src = f"EXPLORATORY_ENV (NB3_CPU_KIT only; the RC2 plan failed: {type(e).__name__}: {e})"[:400]
    finally:
        L.help_check = hc
    for k in ("NCCL_PROTO", "NCCL_ALGO"):                 # addendum 2: NCCL picks its own protocol / algorithm
        env.pop(k, None)
    py = f"{kvv['MERLIN_VENV']}/bin/python"
    if not kit and not Path(py).is_file():
        BENCH["plan"] = f"FAILED: Merlin venv python missing: {py}"
        raise RuntimeError(f"BENCH_ENV_FAIL_CLOSED: the Merlin venv python does not exist ({py})")
    BENCH.update(py=py, env=env, plan=src, plan_label=label, plan_ok=True)
    return {"plan": src, "plan_label": label, "py": BENCH["py"],
            "env_keys": {k: env[k] for k in sorted(env) if k.startswith(("NCCL_", "VLLM_PP", "CUDA_", "LD_PRELOAD", "MERLIN_"))},
            "config_diff_vs_rc2_fp8": "none at plan level (no server); per-bench overrides recorded in each bench record"}


def ckpt_facts():
    try:
        tc = json.loads((CKPT / "config.json").read_text()).get("text_config") or {}
        return int(tc.get("hidden_size") or 2560), {"bfloat16": 2, "float16": 2, "float32": 4}.get(tc.get("dtype"), 2)
    except Exception:
        return 2560, 2


HIDDEN, DTB = ckpt_facts()
TS = C.t_sets()
AR = {"hidden": HIDDEN, "dtype_bytes": DTB, "T": TS["all"], "sizes": [C.ar_bytes(HIDDEN, t, DTB) for t in TS["all"]],
      "decode": C.ar_bytes(HIDDEN, C.micro_T(8, 3), DTB), "iters": [100, 5, 20, 10],
      "nccl_env": "RC2 plan: NCCL_P2P_DISABLE=1, NCCL_PROTO / NCCL_ALGO unset; LL128: " + C.LL128_LABEL}
R["nb3_grid"] = {"t_sets": TS, "ar_sizes": AR["sizes"], "rc2_point": C.RC2_STEADY, "step_ms": list(C.STEP_MS)}


# ---------------------------------------------------------------- D5 Marlin MoE (family 1)
def s_d5(phase="core"):
    """D5 in priority order (N3-16 / N3-19): core = T16 (uniform + replay + sweep), T4 / T8, the adversarial reference
    cases, run BEFORE D7 / D4; rest = everything else (T32 / T64 + Zipf, hot / request, the other T, split, replay, EP
    rank 1 spot, ncu last), run at the end. Each bench keeps every valid row on a budget stop (PARTIAL_BUDGET)."""
    key = "D5" if phase == "core" else "D5_rest"
    g = gates(f"D5 {phase}")
    if g:
        return g
    gate = boot_gate()
    if gate["ok"] is not True:
        return {"status": "GPU_NOT_EMPTY", "boot_gate": gate}
    out = OUT / ("d5_marlin_moe.json" if phase == "core" else "d5_marlin_moe_rest.json")
    out.unlink(missing_ok=True)
    budget = (max(120, min(SEC_D5_CORE, left() - 2400)) if phase == "core" else max(120, min(SEC_D5_REST, left() - 180)))
    argv = [NB3_DIR / "nb3_moe_bench.py", "--ckpt", CKPT, "--out", out, "--layer", os.environ.get("NB3_MOE_LAYER", "0"),
            "--budget-s", int(budget), "--phase", phase] + (["--fake"] if FAKE else []) + (["--routing-file", ROUTING_FILE] if ROUTING_FILE else [])
    r = run_bench(f"d5_moe_{phase}", argv, "0", budget + 180, {"MERLIN_MARLIN_MOE_CFG": None})
    res = json.loads(out.read_text()) if out.is_file() else {}
    D5R[phase] = res
    merged = C.merge_d5(D5R.get("core"), D5R.get("rest"))
    o = finish(key, r, out, "0", "d5", C.d5_table(merged), res=res)
    o.update(boot_gate=gate, family=C.FAMILY["D5"], routing_note=C.ROUTING_NOTE, phase=phase,
             units_done=len(res.get("units_done") or []), units_skipped=res.get("units_skipped"),
             model=res.get("model"), load=res.get("load"), dequant_crosscheck=res.get("dequant_crosscheck"),
             preflight=res.get("preflight"), first_call_gate=res.get("first_call_gate"), ep1_spot=res.get("ep1_spot"),
             humming=(res.get("humming") or {}).get("summary") or (res.get("humming") or {}).get("status"),
             imports=res.get("imports"), table=TABLES.get("d5"), ncu=res.get("ncu"),
             replay={k: v for k, v in (res.get("replay") or {}).items() if k != "by_size"},
             routing_file=R["setup"]["routing_file"])
    return o


# ---------------------------------------------------------------- D7 world-2 all-reduce (family 3)
PROBE = ("import sys, json, torch\ntorch.ops.load_library(sys.argv[1])\nops = torch.ops.p5_host_ar\nbad = []\n"
         "for nb in json.loads(sys.argv[2]):\n    for dt, es in ((0, 4), (1, 2), (2, 2)):\n        for mode in (0, 1):\n"
         "            n = nb // es\n            r = list(ops.cover_check(n, 2, dt, mode, 1 + n % 7))\n"
         "            if r[0] != 1:\n                bad.append([nb, dt, mode, r])\n"
         "print('P5_PROBE ' + json.dumps({'info': ops.info(), 'torch': torch.__version__, 'cover_check_world2_bad': bad[:10]}))\n")


def s_p5_build():
    """P5 .so built from the embedded E23 sources with the Merlin venv's torch (exact ABI; nvcc 13 from the venv,
    sm_89 only): first with Merlin's GCC 13 env, then the system compiler; E23's dataset .so only as a last resort.
    Then a probe: the library loads, info() and cover_check at world 2 for every NB3 size."""
    g = gates("D7 P5 build")
    if g:
        return g
    if FAKE:
        P5.update(so="FAKE", source="fake")
        return {"so": "FAKE", "status": "DONE"}
    src, outd = NB3_DIR / "p5_src", NB3_DIR / "p5"
    tries = []
    for mode in ("merlin_env", "system_cc"):
        extra = {} if mode == "merlin_env" else {"CC": None, "CXX": None, "CUDAHOSTCXX": None}
        r = run_bench(f"p5_build_{mode}", [src / "build_host_ar.py", "--out", outd, "--build-dir", TMP / f"nb3_p5_build_{mode}",
                                           "--src", src], "", 900, extra)
        txt = r.pop("_log", "")
        bj = next((json.loads(l[len("BUILD_JSON "):]) for l in txt.splitlines() if l.startswith("BUILD_JSON ")), {})
        tries.append({"mode": mode, "rc": r["rc"], "sec": r["sec"], "ok": bj.get("ok"), "error": (bj.get("error") or "")[-800:],
                      "so_sha256": bj.get("so_sha256"), "torch": bj.get("torch"), "spill_lines": (bj.get("spill_lines") or [])[:5]})
        if bj.get("ok") and bj.get("so"):
            P5.update(so=bj["so"], source=f"built here ({mode})")
            break
    if not P5["so"]:
        mf = sorted(KITS_DIR.rglob("q27_e23_kernels_manifest.json")) if KITS_DIR.is_dir() else []
        if mf:
            so = mf[0].parent / "p5_host_ar.so"
            want = (json.loads(mf[0].read_text()).get("p5_host_ar.so") or {}).get("sha256")
            if so.is_file() and want and hashlib.sha256(so.read_bytes()).hexdigest() == want:
                P5.update(so=str(so), source="E23 dataset .so (sha-checked; built for the stock vLLM 0.32 torch)")
    if not P5["so"]:
        return {"status": "BUILD_FAILED", "tries": tries}
    pr = run_bench("p5_probe", ["-c", PROBE, P5["so"], json.dumps(AR["sizes"])], "", 300)
    ptxt = pr.pop("_log", "")
    probe = next((json.loads(l[len("P5_PROBE "):]) for l in ptxt.splitlines() if l.startswith("P5_PROBE ")), None)
    ok = bool(probe) and not probe.get("cover_check_world2_bad") and "worlds 2" in str(probe.get("info"))
    if not ok:
        P5["so"] = None
    return {"status": "DONE" if ok else "PROBE_FAILED", "tries": tries, "so": P5["so"], "source": P5["source"], "probe": probe,
            "probe_rc": pr["rc"], "source_facts": P5_SOURCE_FACTS}


def ar_argv(out, steps, tmo):
    if FAKE:
        return [NB3_DIR / "nb3_common.py", "--fake-ar", "--out", out, "--sizes", json.dumps(AR["sizes"]), "--steps", steps]
    return [NB3_DIR / "bench_host_ar.py", "--so", P5["so"], "--out", out, "--world", "2", "--steps", steps, "--quick",
            "--timeout-s", int(tmo)]


def ar_extra():
    return {"NB3_AR_SIZES": json.dumps(AR["sizes"]), "NB3_AR_DECODE": AR["decode"], "NB3_AR_ITERS": json.dumps(AR["iters"]),
            "NB3_AR_NCCL_ARMS": "{}", "NB3_AR_NO_SHM_SWEEP": "1", "NCCL_PROTO": None, "NCCL_ALGO": None,
            "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,TUNING"}


def nccl_protocols(jpaths):
    """The protocol / algorithm NCCL itself chose (NCCL_DEBUG=INFO, TUNING) from the bench's per-rank NCCL logs."""
    seen, lines = {}, []
    for jp in jpaths:
        wd = Path(jp).parent / (Path(jp).stem + "_work")
        for f in sorted(wd.rglob("nccl.*.txt")) if wd.is_dir() else []:
            try:
                for l in f.read_text(errors="replace").splitlines():
                    if re.search(r"AllReduce|protocol|Protocol|LL128|\bLL\b|Simple|algorithm|Algo", l):
                        for p in ("LL128", "LL", "Simple"):
                            if re.search(r"\b%s\b" % p, l):
                                seen[p] = seen.get(p, 0) + 1
                        if len(lines) < 30:
                            lines.append(l.strip()[-240:])
            except Exception:
                pass
    return {"mentions": seen, "lines": lines}


def _shm_p5():
    try:
        return {p.name for p in Path("/dev/shm").glob("p5har_*")}
    except Exception:
        return set()


def ar_result(js):
    """The pair / solo bench JSONs -> one record for C.verified_bench_status: every file present with exit_code 0,
    the bench's own checks of every group and rank (prefixed by pair)."""
    ecs = [j.get("exit_code") if j else None for j in js.values()]
    present = bool(js) and all(bool(j) for j in js.values())
    return {"status": "DONE" if present and all(e == 0 for e in ecs) else ("MISSING_RESULT" if not present else "BENCH_EXIT_NONZERO"),
            "exit_code": max((e if isinstance(e, int) else 9) for e in ecs) if ecs else 9,
            "checks": {f"{p}.{k}": v for p, j in js.items() for k, v in C._ar_checks(j, "*").items()}}


def s_d7_pairs():
    g = gates("D7 pairs")
    if g:
        return g
    if not P5["so"]:
        return {"status": "SKIPPED_NO_P5", "why": "the P5 build / probe failed: no world-2 comparison without it"}
    gate = boot_gate()
    if gate["ok"] is not True:
        return {"status": "GPU_NOT_EMPTY", "boot_gate": gate}
    shm0 = _shm_p5()
    tmo = max(FLOOR(120), min(SEC_PAIRS, left() - 360))
    outs = {"0,1": OUT / "d7_pair01.json", "2,3": OUT / "d7_pair23.json"}
    for p in outs.values():
        p.unlink(missing_ok=True)
    jobs = [start_bench(f"d7_pair{k.replace(',', '')}", ar_argv(p, "a,b,c,d", tmo - 30), k, ar_extra()) for k, p in outs.items()]
    rs = wait_benches(jobs, tmo + FLOOR(90))
    logs = "\n".join(v.pop("_log", "") for v in rs.values())
    js = {k: (json.loads(p.read_text()) if p.is_file() else {}) for k, p in outs.items()}
    left_shm = sorted(_shm_p5() - shm0)
    for n in left_shm:
        try:
            (Path("/dev/shm") / n).unlink()
        except Exception:
            pass
    D7["pairs"] = js
    tab = C.d7_table(js, HIDDEN, DTB, solo=D7.get("solo"))
    tab["nccl_protocols"] = nccl_protocols(list(outs.values()))
    o = finish("D7", {"rc": max(v["rc"] if v["rc"] is not None else 9 for v in rs.values()),
                      "timeout": any(v["timeout"] for v in rs.values()), "jobs": rs, "_log": logs},
               outs["0,1"], "0,1,2,3", "d7", tab, res=ar_result(js))
    o.update(boot_gate=gate, family=C.FAMILY["D7"], p5_shm_left_removed=left_shm, numeric=tab["numeric"], arms=tab["arms"],
             nccl_protocols=tab["nccl_protocols"], nccl_transports=tab["nccl_transports"], concurrency=tab["concurrency"],
             group_overlap_frac_info=tab["group_overlap_frac_info"], ll128=C.LL128_LABEL,
             summaries={k: j.get("summary") for k, j in js.items()}, p5_source_facts=P5_SOURCE_FACTS)
    return o


def s_d7_solo():
    g = gates("D7 solo")
    if g:
        return g
    if not P5["so"]:
        return {"status": "SKIPPED_NO_P5"}
    gate = boot_gate()
    if gate["ok"] is not True:
        return {"status": "GPU_NOT_EMPTY", "boot_gate": gate}
    out = OUT / "d7_solo01.json"
    out.unlink(missing_ok=True)
    tmo = max(FLOOR(90), min(SEC_SOLO, left() - 300))
    r = run_bench("d7_solo01", ar_argv(out, "b", tmo - 30), "0,1", tmo + FLOOR(60), ar_extra())
    js = json.loads(out.read_text()) if out.is_file() else {}
    D7["solo"] = js
    tab = C.d7_table(D7.get("pairs") or {}, HIDDEN, DTB, solo=js)
    tab["nccl_protocols"] = nccl_protocols([OUT / "d7_pair01.json", OUT / "d7_pair23.json", out])
    o = finish("D7_solo", r, out, "0,1", "d7", tab, res=ar_result({"0,1": js}))
    o.update(boot_gate=gate, summary=js.get("summary"), note="solo = interference reference only; D7 promotion uses the pairs")
    return o


D7 = {}


# ---------------------------------------------------------------- D4 GEMM plans + heads (family 2)
def s_d4():
    g = gates("D4")
    if g:
        return g
    gate = boot_gate()
    if gate["ok"] is not True:
        return {"status": "GPU_NOT_EMPTY", "boot_gate": gate}
    out = OUT / "d4_gemm.json"
    out.unlink(missing_ok=True)
    budget = max(120, min(SEC_D4, left() - 300))
    argv = [NB3_DIR / "nb3_gemm_bench.py", "--ckpt", CKPT, "--out", out, "--budget-s", int(budget)] + (["--fake"] if FAKE else [])
    r = run_bench("d4_gemm", argv, "0", budget + 180)
    res = json.loads(out.read_text()) if out.is_file() else {}
    o = finish("D4", r, out, "0", "d4", C.d4_table(res) if res else None)
    o.update(boot_gate=gate, family=C.FAMILY["D4"], plans=res.get("plans"), header_shapes=res.get("header_shapes"),
             imports=res.get("imports"), draft_heads={k: v for k, v in (res.get("draft_heads") or {}).items() if k != "tb"},
             sm89_skinny=res.get("sm89_skinny"), table=TABLES.get("d4"))
    return o


if not RERUN and not os.environ.get("NB3_UNIT"):
    m = R["sections"]["A0"]["merge"]
    if m.get("ok") or FAKE:
        step("A0", "merlin_install", 6, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        step("A0", "sha_gate", 4, s_sha)
        step("B0", "bench_env", 2, s_plan)
        step("D5", "marlin_moe", 13, lambda: s_d5("core"))     # N3-19: SHA -> D5 core (T16/T4/T8) -> D7 -> D4 -> D5 rest
        step("D7", "p5_build", 3, s_p5_build)
        step("D7", "allreduce_pairs", 10, s_d7_pairs)
        step("D7", "allreduce_solo", 4, s_d7_solo)
        step("D4", "gemm_plans_heads", 12, s_d4)
        step("D5", "marlin_moe_rest", 8, lambda: s_d5("rest"))
    else:
        R["errors"].append("NB3: checkpoint merge failed; no Merlin install, no benches")
    R["host_phase"] = HOST_PHASE
    R["minutes"] = round((time.time() - T_START) / 60, 1)
    save()
    log("NB3 done", json.dumps({f"{s}/{n}": r.get("status") for s, st in R["sections"].items() if isinstance(st, dict)
                                for n, r in st.items() if isinstance(r, dict) and "status" in r}))
'''

CELL6 = r'''# ==================================================================== CELL 6 — report + fallback submission
if not RERUN and not os.environ.get("NB3_UNIT"):
    write_fallback_submission()
    nb3_report()
    R["summary"] = {f"{s}/{n}": {k: r.get(k) for k in ("status", "why", "checks", "errors", "sec", "family") if k in r}
                    for s, st in R["sections"].items() if isinstance(st, dict) for n, r in st.items()
                    if isinstance(r, dict) and "status" in r}
    save()
    print(json.dumps({"summary": R["summary"], "nb4_top": R.get("nb4_top"), "scorecard": R.get("scorecard")}, default=str)[:8000])
'''

import base64
import zlib
subs = {"__LAB__": json.dumps(LAB), "__PLE__": json.dumps(RC2["PLE"]),
        "__NB3_FILES_Z__": repr(base64.b64encode(zlib.compress(json.dumps(FILES).encode("utf-8"), 9)).decode("ascii")),
        "__NB3_FILES_SHA__": json.dumps({k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in FILES.items()})}
cells_src = []
for i, c in enumerate((CELL1, CELL2, CELL3, CELL4, CELL5, CELL6)):
    for k, v in subs.items():
        c = c.replace(k, v)
    if i:                                      # E23 / RC2: a competition rerun skips cells 2-6 without an exception
        c = "if not RERUN:\n" + textwrap.indent(c, "    ")
    compile(c, f"<nb3 cell {i + 1}>", "exec")
    cells_src.append(c)
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
               [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cells_src],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
data = json.dumps(nb, indent=1).encode("utf-8")
assert b"\r" not in data.replace(b"\\r", b""), "CR in notebook"
assert len(data) < 400_000, len(data)
OUT_NB.write_bytes(data)
(HERE / "nb3_code_export.py").write_bytes("\n\n".join(cells_src).encode("utf-8"))   # one file, for the local harness
EXP = HERE / "nb3_export"
for n, t in FILES.items():
    (EXP / n).parent.mkdir(parents=True, exist_ok=True)
    (EXP / n).write_bytes(t.encode("utf-8"))
print("wrote", OUT_NB.name, "sha256", hashlib.sha256(data).hexdigest(), "|", len(data), "bytes")
