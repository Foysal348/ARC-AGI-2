"""E23 rev 3d gate tests (four GPT audits, REV3C_GPT4_PRIORITY_FIX_LIST #11): injected faults against the REAL gate code.
The notebook functions are taken from the generated FULL notebook code by name (ast), executed with stubs, and driven
with the cases each audit listed. Usage: python gate_tests.py <nb_FULL.py> <e23_lab.py>. Prints `GATES {json}` and
exits 0 only when every case gives the expected result."""
import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

NB, LABP = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("e23_lab", LABP)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
src = Path(NB).read_text(encoding="utf-8")
tree = ast.parse(src)
WANT = {"oracle_gate", "gated", "nvfp4_gate", "graph_proof", "s2", "s4_thread", "_killpg", "_windows", "_num", "taint_check",
        "s1_pp4", "_done"}
code = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in WANT and node.name not in code:
        code[node.name] = ast.get_source_segment(src, node)
missing = sorted(WANT - set(code))
RES, BAD = {"missing_functions": missing}, []


def check(name, got, want):
    ok = got == want if not callable(want) else bool(want(got))
    RES[name] = {"got": got, "ok": ok}
    if not ok:
        BAD.append(name)


import re, subprocess, os, signal, hashlib, time   # noqa: E401  (names the extracted functions use)
G = {"CLEAN": ("EXACT", "NEAR_TIE"), "R": {"sections": {}, "setup": {}, "errors": [], "inputs": {}}, "L": L, "json": json,
     "re": re, "subprocess": subprocess, "os": os, "signal": signal, "Path": Path, "FAKE": None, "SM": False,
     "NVSMI_OK": True, "ABORT": {"why": None}, "time": time, "hashlib": hashlib}
for node in ast.walk(tree):            # rev 3f: module-level constants the fault gate uses
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
            and node.targets[0].id in ("XID", "XID_RESET", "HEALTH_CODE") and node.targets[0].id not in G:
        exec(ast.get_source_segment(src, node), G)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ("gpu_health", "fault_gate") and node.name not in code:
        code[node.name] = ast.get_source_segment(src, node)
for n in ("_windows", "_num", "oracle_gate", "gated", "nvfp4_gate", "graph_proof", "_killpg", "gpu_health", "fault_gate",
          "s2", "s4_thread", "taint_check"):
    if n in code:
        exec(code[n], G)

# ---- oracle_gate / gated (#3)
og = G["oracle_gate"]
check("gate_clean", og({"oracle": {"errors": [None], "garbage": [None], "vs_x": {"status": "NEAR_TIE"}}}, need=("x",)), None)
check("gate_fail_margin", og({"oracle": {"errors": [None], "garbage": [None], "vs_x": {"status": "FAIL_HIGH_MARGIN"}}}, need=("x",)),
      "NO_GO_ORACLE")
check("gate_request_error", og({"oracle": {"errors": ["timeout"], "garbage": [None], "vs_x": {"status": "EXACT"}}}, need=("x",)),
      "NO_GO_ORACLE_REQUEST_ERROR")
check("gate_garbage", og({"oracle": {"errors": [None], "garbage": ["LOW_VARIETY"]}}), "NO_GO_ORACLE_GARBAGE")
check("gate_missing_ref", og({"oracle": {"errors": [None], "garbage": [None]}}, need=("x",)), "INCONCLUSIVE_NO_ORACLE")
check("gate_inconclusive", og({"oracle": {"errors": [None], "garbage": [None], "vs_x": {"status": "INCONCLUSIVE"}}}, need=("x",)),
      "INCONCLUSIVE_ORACLE")
o = G["gated"]({"oracle": {"errors": [None], "garbage": [None], "vs_x": {"status": "FAIL_HIGH_MARGIN"}}}, need=("x",))
check("gated_nested_fail_not_done", (o.get("status"), o.get("correctness")), ("NO_GO_ORACLE", "NO_GO_ORACLE"))

# ---- compare_tf completeness (#1a)
full = {"positions": [[-0.1, 1]] * 256, "n": 256}
one = {"positions": [[-0.1, 1]] + [None] * 255, "n": 256}
check("tf_1_of_256_inconclusive", L.compare_tf(full, one)["status"], "INCONCLUSIVE_INCOMPLETE_TF")
check("tf_missing_ref", L.compare_tf(None, full)["status"], "INCONCLUSIVE")
check("tf_complete_measured", L.compare_tf(full, full)["status"], "MEASURED")
bad_tf = {"positions": [[-25.0, 9]] * 256, "n": 256}
r = L.compare_tf(full, bad_tf)
check("tf_bad_is_measurement_only", (r["status"], r["rank1_agreement"]), ("MEASURED", 0.0))


# ---- nvfp4_gate (rev 3e: BF16-KV reference; NVFP4 within 3x the FP8-KV drift, rank-1 within 0.02)
def band(r1, mean, p99):
    return {"status": "MEASURED", "n": 256, "rank1_agreement": r1, "mean_abs_dlogprob": mean, "p99_abs_dlogprob": p99,
            "max_abs_dlogprob": p99 * 1.5}


def sec5(tgt="DONE", orc="NEAR_TIE", ref="DONE", f8="DONE", nv="DONE", nv_tf=None, f8_tf=None):
    G["R"]["sections"]["5_nvfp4_kv"] = {
        "nvfp4_target_only": {"status": tgt, "oracle": {"vs_fp8_mtp0": {"status": orc}}},
        "tf_bf16kv_reference": {"status": ref},
        "tf_fp8kv_vs_bf16": {"status": f8, "teacher_forced": f8_tf or {"40k": band(0.97, 0.03, 0.4)}},
        "tf_nvfp4_vs_bf16": {"status": nv, "teacher_forced": nv_tf or {"40k": band(0.96, 0.07, 0.9)}}}
    return G["nvfp4_gate"]()


check("nv_within_3x_fp8_allowed", sec5()[0], True)
check("nv_tf_inconclusive_blocks", sec5(nv="INCONCLUSIVE_TF")[0], False)
check("nv_missing_oracle_blocks", sec5(orc="INCONCLUSIVE_NO_ORACLE")[0], False)
check("nv_fail_margin_blocks", sec5(orc="FAIL_HIGH_MARGIN")[0], False)
check("nv_target_not_done_blocks", sec5(tgt="NO_GO_ORACLE_REQUEST_ERROR")[0], False)
check("nv_no_fp8_calibration_blocks", sec5(f8="INCONCLUSIVE_TF")[0], False)
check("nv_no_bf16_reference_blocks", sec5(ref="INCONCLUSIVE_TF")[0], False)
check("nv_24nat_0pct_blocks", sec5(nv_tf={"40k": band(0.0, 24.95, 30.0)})[0], False)
check("nv_rank1_drop_blocks", sec5(nv_tf={"40k": band(0.93, 0.03, 0.4)})[0], False)
check("nv_p99_4x_blocks", sec5(nv_tf={"40k": band(0.97, 0.03, 1.7)})[0], False)
check("nv_exact_fp8_floors", sec5(f8_tf={"40k": band(1.0, 0.0, 0.0)}, nv_tf={"40k": band(0.985, 0.015, 0.15)})[0], True)
check("nv_none_metric_no_crash", sec5(nv_tf={"40k": {"status": "MEASURED", "n": 256, "rank1_agreement": None}})[0], False)

# ---- graph_proof + trace_summary (#6)
gp = G["graph_proof"]
check("gp_unsettled", gp({"profile": {"status": "INCONCLUSIVE_PREFILL_NOT_SETTLED"}})["status"], "INCONCLUSIVE_PREFILL_NOT_SETTLED")
check("gp_unproven", gp({"profile": {"trace": {"graph_verdict": "UNPROVEN"}}})["status"], "INCONCLUSIVE_GRAPH")
check("gp_downgrade", gp({"profile": {"trace": {"graph_verdict": "FULL"}},
                          "servers": {"a": {"graph_downgrade_lines": ["setting cudagraph_mode=PIECEWISE"]}}}, expect="FULL")["status"],
      "DOWNGRADED")
check("gp_downgrade_to_full_decode_only_not_counted", gp({"profile": {"trace": {"graph_verdict": "FULL"}},
      "servers": {"a": {"graph_downgrade_lines": ["Overriding cudagraph_mode to FULL_DECODE_ONLY"]}}}, expect="FULL")["status"],
      "PROVEN_FULL")
check("gp_downgrade_032_wording", gp({"profile": {"trace": {"graph_verdict": "FULL"}},
      "servers": {"a": {"graph_downgrade_lines": ["Overriding cudagraph_mode to PIECEWISE"]}}}, expect="FULL")["status"], "DOWNGRADED")
check("gp_g0_mismatch", gp({"profile": {"trace": {"graph_verdict": "FULL"}}}, expect="PIECEWISE")["status"], "MISMATCH_EXPECTED")
check("gp_full_proven", gp({"profile": {"trace": {"graph_verdict": "FULL"}}}, expect="FULL")["status"], "PROVEN_FULL")
tdir = Path(NB).parent / "gate_trace"
tdir.mkdir(exist_ok=True)
ev = [{"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 10, "dur": 2, "tid": 1, "args": {"correlation": 7}},
      {"ph": "X", "cat": "kernel", "name": "flashinfer_attn", "ts": 12, "dur": 5, "tid": 2, "args": {"correlation": 7}}]
(tdir / "x_rank0.1.pt.trace.json").write_text(json.dumps({"traceEvents": ev}))
check("trace_no_annotation_unproven", L.trace_summary(tdir)["graph_verdict"], "UNPROVEN")
ev2 = ev + [{"ph": "X", "cat": "user_annotation", "name": "gpu_model_runner: forward", "ts": 5, "dur": 20, "tid": 1},
            {"ph": "X", "cat": "gpu_user_annotation", "name": "gpu_model_runner: forward", "ts": 5, "dur": 20, "tid": 9},
            {"ph": "X", "cat": "gpu_user_annotation", "name": "gpu_model_runner: forward", "ts": 6, "dur": 20, "tid": 9}]
(tdir / "x_rank0.1.pt.trace.json").write_text(json.dumps({"traceEvents": ev2}))
check("trace_cpu_annotation_full", L.trace_summary(tdir)["graph_verdict"], "FULL")

# ---- s2 (P5) release + result gate (#7) and s4_thread (P4) synthetic-weight gate (#10): stub bench scripts
stub = Path(NB).parent / "gate_stub"
stub.mkdir(exist_ok=True)
G["OUT"] = stub
G["STOCK"] = {"py": sys.executable, "env": dict(os.environ)}
G["FP8"] = stub


def kit_with(name, payload):
    (stub / name).write_text("import json, sys\na = sys.argv\nout = a[a.index('--out') + 1]\n"
                             f"json.dump({payload!r}, open(out, 'w'))\n")
    return str(stub / name)


class _L:
    rel = True

    @staticmethod
    def wait_gpu_release(*a, **k):
        return _L.rel


G["L"] = _L
G["kit"] = lambda n: kit_with(n, {"summary": {"all_checks_pass": True}}) if n.endswith("host_ar.py") else str(stub / n)
_L.rel = False
G["ABORT"]["why"] = None
check("p5_release_false_fails", (G["s2"]()["status"], bool(G["ABORT"]["why"])), ("FAIL_GPU_RELEASE", True))
_L.rel, G["ABORT"]["why"] = None, None
check("p5_release_none_fails", G["s2"]()["status"], "FAIL_GPU_RELEASE")
_L.rel, G["ABORT"]["why"] = True, None
check("p5_clean_done", G["s2"]()["status"], "DONE")
G["kit"] = lambda n: kit_with(n, {"summary": {"all_checks_pass": False}}) if n.endswith("host_ar.py") else str(stub / n)
check("p5_checks_failed", G["s2"]()["status"], "CHECK_FAILED")
G["kit"] = lambda n: kit_with(n, {"all_ok": True, "shapes": {"gate_up_proj": {"weights_source": {"kind": "synthetic"}}}}) \
    if n.endswith("fp8_gemm.py") else str(stub / n)
G["P4_PROC"] = {"p": None}
r4 = G["s4_thread"]()
check("p4_synthetic_rejected", (r4["status"], r4.get("why")), ("CHECK_FAILED", "SYNTHETIC_WEIGHTS_NOT_ALLOWED"))
G["kit"] = lambda n: kit_with(n, {"all_ok": True, "shapes": {"gate_up_proj": {"weights_source": {"kind": "real"}}}}) \
    if n.endswith("fp8_gemm.py") else str(stub / n)
check("p4_real_done", G["s4_thread"]()["status"], "DONE")

# ---- rev 3e: garbage-under-load windows and failed probes are NO_GO (audits B-O1, C-F2)
check("gate_garbage_window", og({"oracle": {"errors": [None], "garbage": [None]}, "steady": {"40k@16": {"status": "FAIL_GARBAGE_UNDER_LOAD"}}}),
      "NO_GO_GARBAGE_UNDER_LOAD")
check("gate_garbage_replica", og({"oracle": {"errors": [None], "garbage": [None]},
                                  "40k": {"replicas": [{"status": "STEADY"}, {"status": "FAIL_GARBAGE_UNDER_LOAD"}]}}),
      "NO_GO_GARBAGE_UNDER_LOAD")
check("gate_probe_fail_real", og({"oracle": {"errors": [None], "garbage": [None]}, "probes": {"ok": False}}), "NO_GO_SHORT_PROBE")
G["FAKE"] = "fake"
check("gate_probe_fail_fake_ignored", og({"oracle": {"errors": [None], "garbage": [None]}, "probes": {"ok": False}}), None)
G["FAKE"] = None

# ---- rev 3f: per-forward graph verdict (GPT rev 3e audit: 2 FULL + 1 EAGER read FULL by the median)
def fw_trace(kinds):
    evs, ts = [], 0
    for i, k in enumerate(kinds):
        lo = ts
        evs.append({"ph": "X", "cat": "user_annotation", "name": "gpu_model_runner: forward", "ts": lo, "dur": 50, "tid": 1})
        if k == "FULL":
            evs.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": lo + 5, "dur": 1, "tid": 1, "args": {}})
        elif k == "PIECEWISE":
            for j in range(3):
                evs.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": lo + 5 + j, "dur": 1, "tid": 1, "args": {}})
            evs.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": lo + 20, "dur": 1, "tid": 1, "args": {}})
        else:
            for j in range(4):
                evs.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": lo + 5 + j, "dur": 1, "tid": 1, "args": {}})
        ts += 100
    (tdir / "x_rank0.1.pt.trace.json").write_text(json.dumps({"traceEvents": evs}))
    return L.trace_summary(tdir)["graph_verdict"]


check("trace_all_full", fw_trace(["FULL"] * 3), "FULL")
check("trace_2full_1eager_mixed", fw_trace(["FULL", "FULL", "EAGER"]), "MIXED")
check("trace_all_piecewise", fw_trace(["PIECEWISE"] * 3), "PIECEWISE")

# ---- rev 3f: Xid recovery classes + GPU health probe
G.update(FAKE=None, ABORT={"why": None}, STOCK={"py": sys.executable, "env": dict(os.environ)})
G["R"]["setup"] = {}
G["gpu_health"] = lambda: {"ok": True}
check("xid_reset_class_aborts", (G["fault_gate"](["NVRM: Xid (PCI:0000:00:04): 79, GPU has fallen off the bus"], "armX"),
                                 bool(G["ABORT"]["why"])), ("ABORT", True))
G["ABORT"]["why"] = None
check("app_fault_healthy_continues", (G["fault_gate"](["CUDA error: an illegal memory access was encountered"], "armY"),
                                      G["ABORT"]["why"]), ("CONTINUE", None))
G["gpu_health"] = lambda: {"ok": False}
check("app_fault_unhealthy_aborts", (G["fault_gate"](["CUDA error: an illegal memory access was encountered"], "armZ"),
                                     bool(G["ABORT"]["why"])), ("ABORT", True))
G["ABORT"]["why"] = None
check("no_fault_line_none", G["fault_gate"](["all good"], "armW"), None)
G["gpu_health"] = lambda: {"ok": True}

# ---- rev 3e: a GPU fault marks only its arm; later arms are labelled, not skipped; _done needs DONE + CLEAN
G["R"]["setup"] = {}
a1 = G["taint_check"]({"status": "DONE", "correctness": "CLEAN", "servers": {"pp4c": {"xid": True, "xid_lines": ["CUDA error: an illegal memory access was encountered"]}}})
a2 = G["taint_check"]({"status": "DONE", "correctness": "CLEAN", "servers": {"next": {"xid": False}}})
check("taint_arm_no_go", (a1["status"], a1["correctness"]), ("NO_GO_GPU_FAULT", "NO_GO_GPU_FAULT"))
check("taint_later_arm_labelled_not_blocked", (a2["status"], a2.get("after_gpu_fault")), ("DONE", ["pp4c"]))
exec(code.get("_done", "def _done(path):\n    raise RuntimeError('missing')"), G)
G["R"]["sections"] = {"s": {"ok": {"status": "DONE", "correctness": "CLEAN", "after_gpu_fault": ["pp4c"]},
                            "nogo": {"status": "NO_GO_ORACLE", "correctness": "NO_GO_ORACLE"},
                            "dirty": {"status": "DONE", "correctness": "NO_GO_GARBAGE_UNDER_LOAD"}}}
check("aa_picks_clean_after_fault", G["_done"]("s/ok"), True)
check("aa_skips_no_go", G["_done"]("s/nogo"), False)
check("aa_skips_unclean", G["_done"]("s/dirty"), False)

# ---- rev 3e: PP4 staged fail-fast, counted requests (stubbed servers; the real s1_pp4)
class _Srv:
    port, name = 1, "pp4"

    def alive(self):
        return True


CALLS = []


def mk_greedy(bad_at=None):
    def greedy(port, prompts, n_tok=0, timeout=0):
        CALLS.append(len(prompts[0]))
        tok = [1, 2, 3] if len(prompts[0]) != bad_at else []
        return [{"tokens": tok, "text": "alpha beta gamma", "error": None, "margins": [2.0] * 3}]
    return greedy


def run_pp4(probe_ok=True, bad_at=None, ref_clean=True):
    CALLS.clear()
    LV = []
    Lp = types.SimpleNamespace(chat_probes=lambda port: {"ok": probe_ok}, greedy=mk_greedy(bad_at), compare=L.compare,
                               degenerate=L.degenerate)
    ref = [{"tokens": [1, 2, 3], "text": "alpha beta gamma", "error": None, "margins": [2.0] * 3}] * 4
    P = [[0] * n for n in (2048, 2049, 8192, 40000)]
    G.update(L=Lp, INT4="int4", ROWS={"int4_mtp0": ref} if ref_clean else {}, ORACLE_P=P[:3], LONG_P=P[3:], NTOK=3, SM=True,
             left=lambda: 9e9, stock_flags=lambda *a, **k: [], stock_server=lambda *a, **k: _Srv(),
             with_servers=lambda ss, t, body: body([_Srv()]), levels=lambda *a, **k: LV.append(a[1]) or {"40k@8": {"status": "STEADY"}})
    exec(code["s1_pp4"], G)
    G["taint_check"] = lambda o: o
    out = G["s1_pp4"](True)()
    return out.get("status"), list(CALLS), LV


check("pp4_probe_fail_no_requests", run_pp4(probe_ok=False)[:2], ("NO_GO_SHORT_PROBE", []))
check("pp4_2k_fail_stops", run_pp4(bad_at=2048)[:2], ("NO_GO_2k", [2048]))
check("pp4_40k_fail_no_throughput", (lambda r: (r[0], r[2]))(run_pp4(bad_at=40000)), ("NO_GO_40k", []))
check("pp4_all_pass_throughput", (lambda r: (r[0], r[1], r[2]))(run_pp4()), (None, [2048, 8192, 40000], ["40k"]))
check("pp4_no_reference_skips", run_pp4(ref_clean=False)[0], "SKIPPED_NO_ORACLE")
G["L"] = L

# ---- rev 3e: EAGLE falls through on LAUNCH_FAILED / SERVER_DIED / NO_GO to the next variant (the real s7_eagle)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "s7_eagle":
        code["s7_eagle"] = ast.get_source_segment(src, node)
        break


def run_eagle(statuses):
    seq = list(statuses)
    names = []

    def ws(ss, t, body):
        names.append(ss[0])
        return {"status": seq.pop(0), "servers": {}}
    G.update(eagle_install=lambda: {"vllm": ["x"]}, EAGLE_DRAFT="d", FP8="f", ABORT={"why": None}, left=lambda: 9e9,
             stock_flags=lambda *a, **k: [], stock_server=lambda name, *a, **k: name, with_servers=ws, taint_check=lambda o: o)
    exec(code.get("s7_eagle", "def s7_eagle():\n    raise RuntimeError('missing')"), G)
    out = G["s7_eagle"]()
    return out.get("status"), out.get("variant"), names


check("eagle_server_died_falls_through", run_eagle(["SERVER_DIED", "DONE"]),
      ("DONE", "eagle4_fp8", ["eagle4_fp8_cutedsl", "eagle4_fp8"]))
check("eagle_no_go_falls_through", run_eagle(["LAUNCH_FAILED", "NO_GO_ORACLE", "DONE"])[1], "eagle4_fp8_noprefix")
check("eagle_all_fail_label", run_eagle(["LAUNCH_FAILED", "SERVER_DIED", "LAUNCH_FAILED"])[0], "NO_GO_ALL_EAGLE_VARIANTS")
check("eagle_cutedsl_ok_first", run_eagle([None])[1], "eagle4_fp8_cutedsl")

# ---- Merlin checkpoint full sha (#14): a good manifest passes, one changed byte fails
if "_merlin_sha" not in code:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_merlin_sha":
            code["_merlin_sha"] = ast.get_source_segment(src, node)
            break
mdir = Path(NB).parent / "gate_merlin"
mdir.mkdir(exist_ok=True)
(mdir / "a.safetensors").write_bytes(b"abc" * 1000)
(mdir / "q27_e23_p1_manifest.json").write_text(json.dumps({"files": {"a.safetensors": [3000, hashlib.sha256(b"abc" * 1000).hexdigest()]}}))
G.update(MERLIN=mdir, MERLIN_SHA={"th": None, "ok": None, "bad": []}, save=lambda: None)
G["R"]["inputs"] = {}
exec(code.get("_merlin_sha", "def _merlin_sha():\n    raise RuntimeError('missing')"), G)
G["_merlin_sha"]()
check("merlin_sha_good", G["MERLIN_SHA"]["ok"], True)
(mdir / "a.safetensors").write_bytes(b"abd" + b"abc" * 999)
G["_merlin_sha"]()
check("merlin_sha_changed_byte", (G["MERLIN_SHA"]["ok"], G["MERLIN_SHA"]["bad"]), (False, ["a.safetensors"]))

# ---- static checks on the generated code (#4, #5, #8, #9)
check("pp4_eager_first", src.index('"e_pp4_int4_eager"') < src.index('"f_pp4_int4_compiled"'), True)
check("pp4_vs_int4_mtp0", 'ref = ROWS["int4_mtp0"]' in src and 'refs=("int4_mtp3",)' not in src, True)
check("int4_ref_in_smoke_and_full", 'step("4_sm89_fp8_gemm", "int4_mtp0_target_only_ref", 12, s4_int4_ref)' in src
      and src.index('"int4_mtp0_target_only_ref", 12') < src.index('"int4_mtp3_s20_control", 35'), True)
check("gpu_preflight_raises", 'raise RuntimeError("GPU_PREFLIGHT_FAILED' in src, True)
check("kit_hash_raises", 'raise RuntimeError("KERNEL_KIT_HASH_FAILED' in src, True)
check("p4_weights_real", '"--weights", "real"' in src, True)
check("eagle_cutedsl_first", src.index('"eagle4_fp8_cutedsl"') < src.index('("eagle4_fp8", [], True)'), True)

RES["all_ok"] = not BAD and not missing
RES["failed"] = BAD
print("GATES " + json.dumps(RES, default=str))
sys.exit(0 if RES["all_ok"] else 1)
