"""E27 NBDF build-level gate regression test (GPT audit B1 / B3; local, quick): the REAL builder, outputs in a temp dir
(--test-out), with synthetic real-kit reports:
  1 no report               -> KV_KIT_GATE closed (gpu_ready false)
  2 a passing report        -> KV_KIT_GATE["passed"] True in the built CELL 7 body, gpu_ready true, D4 / D3 / D7 / D4b all in ARMS
  3 drafter_loaded_weights_ok False (kv + seam OK) -> closed
  4 a stale notebook sha    -> closed
Run: python nbdf_build_gate_test.py   (prints 'FAIL <name>' lines, then '<pass>/<total> PASS')"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
import nbdf_lib as N  # noqa: E402
MERLIN_SRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
SRC = ("tools/merlin/launch.py", "tools/merlin/registry.py", "vllm/config/speculative.py")
RES = {"pass": 0, "fail": 0}


def ok(name, cond, info=""):
    RES["pass" if cond else "fail"] += 1
    if not cond:
        print("FAIL", name, str(info)[:400], flush=True)


def build(out, report=None):
    args = [sys.executable, str(HERE / "build_e27_nbdf.py"), "--test-out", str(out)] + (["--kv-gate-report", str(report)] if report else [])
    r = subprocess.run(args, capture_output=True, text=True, timeout=900, cwd=str(HERE))
    facts = json.loads((Path(out) / "nbdf_build_facts.json").read_text(encoding="utf-8")) if r.returncode == 0 else {}
    return r, facts


def report(nb_sha, seam_sha, **over):
    run = lambda b, g, d: {"num_blocks": [b] * 4, "kv_group_count": g, "drafter_group_count": d, "groups": [], "attn_metadata_builders": g}
    pins = N.DRAFTER_PINS
    rep = {"notebook_sha256": nb_sha, "verdict": {"checks": {c: True for c in N.REAL_KIT_CHECKS}},
           "checks": {"R1": {"embed": {"sha256": seam_sha}, "drafter": {"pins_ok": True, "facts": {
               "files": {"model.safetensors": [pins["model.safetensors"]["size"], pins["model.safetensors"]["sha256"]],
                         "config.json": [pins["config.json"]["size"], pins["config.json"]["sha256"]]},
               "embed": {"sha256": pins["embed_tokens"]["sha256"]}, "mask_row": {"sha256": pins["mask_row"]["sha256"]}}}},
                      "R2": {"src_shas": {r: hashlib.sha256((MERLIN_SRC / r).read_bytes()).hexdigest() for r in SRC},
                             "merlin_commit": "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"},
                      "R4": {"plans": {"Q": {"argv": ["--tensor-parallel-size", "2", "--pipeline-parallel-size", "2"],
                                             "env": {"VLLM_PP_LAYER_PARTITION": "25,23"}}}},
                      "R9": {"json": {"runs": {"Q": run(215, 5, 0), "K3": run(190, 6, 1), "K4": run(71, 6, 1), "K7": run(71, 6, 1), "K4_swblock1024": run(205, 6, 1), "K3_swblock1024": run(205, 6, 1),
                                               "K4_fp8": run(195, 6, 1)}}},
                      "R12": {"json": {"ok": True}}}}
    for k, v in over.items():
        rep["verdict"]["checks"][k] = v
    return rep


with tempfile.TemporaryDirectory(prefix="nbdf_gate_") as td:
    td = Path(td)
    r, f0 = build(td / "closed")
    ok("1 closed build runs", r.returncode == 0, r.stderr[-600:])
    ok("1 no report -> closed", f0.get("kv_kit_gate", {}).get("passed") is False and f0.get("gpu_ready") is False)
    ok("1 every D arm in the plan", f0.get("arms_in_plan") == ["Q", "D4s", "D3s", "D4", "D4t", "D7", "D4b"], f0.get("arms_in_plan"))
    sha_c, seam = f0.get("closed_build_sha256"), f0.get("seam_sha256")
    ok("1 the closed build's sha is the notebook's", sha_c == f0.get("sha256"))
    rp = td / "rep_ok.json"
    rp.write_text(json.dumps(report(sha_c, seam)))
    r, f1 = build(td / "gated", rp)
    g = f1.get("kv_kit_gate") or {}
    body = (td / "gated" / "nbdf_cell7_body.py").read_text(encoding="utf-8") if (td / "gated" / "nbdf_cell7_body.py").is_file() else ""
    ok("2 a passing report opens the gate", r.returncode == 0 and g.get("passed") is True and f1.get("gpu_ready") is True, (r.stderr[-400:], g.get("why")))
    ok("2 the gate is embedded in the CELL 7 body", '"passed": true' in body and all(g.get("per_k", {}).get(k) for k in N.D_ARMS))
    ok("2 D4 / D3 / D7 / D4b all in ARMS", f1.get("arms_in_plan") == ["Q", "D4s", "D3s", "D4", "D4t", "D7", "D4b"])
    ok("2 provenance = the closed rebuild", (g.get("provenance") or {}).get("matches") == ["closed_rebuild"], g.get("provenance"))
    ok("2 the gated notebook differs only by the gate", f1.get("closed_build_sha256") == sha_c and f1.get("sha256") != sha_c)
    rp3 = td / "rep_dlw.json"
    rp3.write_text(json.dumps(report(sha_c, seam, drafter_loaded_weights_ok=False)))
    r, f3 = build(td / "dlw", rp3)
    ok("3 drafter_loaded_weights_ok False -> closed", r.returncode == 0 and (f3.get("kv_kit_gate") or {}).get("passed") is False)
    rp4 = td / "rep_stale.json"
    rp4.write_text(json.dumps(report("0" * 64, seam)))
    r, f4 = build(td / "stale", rp4)
    ok("4 a stale notebook sha -> closed", r.returncode == 0 and (f4.get("kv_kit_gate") or {}).get("passed") is False
       and any("STALE_NOTEBOOK_SHA" in w for w in (f4.get("kv_kit_gate") or {}).get("why") or []))
print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
sys.exit(1 if RES["fail"] else 0)
