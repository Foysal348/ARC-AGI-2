import json, os, subprocess, sys, shutil
NB = r"D:\ARC AGI 2\Research Base Experiment\FN-SMOKE-FLASHNEXT-4XL4.ipynb"
COMP = r"C:\Users\hp\AppData\Local\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\fake_input\competitions\arc-prize-2026-arc-agi-2"
GOOD_SMI = "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0 |"
L4x4 = "\n".join(["NVIDIA L4, 23034 MiB, 8.9"] * 4)
MEM = lambda total, avail: f"MemTotal: {total*1000000} kB\nMemAvailable: {avail*1000000} kB\n"
RUNNER = r'''
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8")); G = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code": exec(compile("".join(c["source"]), f"cell{i}", "exec"), G)
r = G["REPORT"]; s0 = r["stages"].get("0_environment", {})
print("RESULT", json.dumps({"stopped_at": r["stopped_at"], "stages": {k: v["ok"] for k, v in r["stages"].items()},
                            "checks": s0.get("checks"), "total_sec": r.get("total_sec")}))
'''
def run(name, **env):
    work = os.path.join("val", name); shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    e = dict(os.environ, FN_SMOKE_WORK=os.path.abspath(work), FN_SMOKE_COMP=COMP, PYTHONIOENCODING="utf-8", **env)
    out = subprocess.run([sys.executable, "-c", RUNNER, NB], env=e, capture_output=True, text=True, timeout=300).stdout
    res = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][-1][7:])
    verdict = [l for l in out.splitlines() if l.startswith("VERDICT:")]
    print(f"{name:28} stopped_at={res['stopped_at']!s:22} stages={res['stages']} | {verdict[-1] if verdict else ''} | t={res['total_sec']}")
    if res["checks"]: print(" " * 30, "checks:", {k: v for k, v in res["checks"].items() if not v} or "all true")
    return res
good = dict(FN_SMOKE_FAKE_SMI=GOOD_SMI, FN_SMOKE_FAKE_GPUS=L4x4, FN_SMOKE_FAKE_MEMINFO=MEM(190, 170), FN_SMOKE_FAKE_TMPFREE="100")
r = run("deadline_budget_30s", FN_SMOKE_BUDGET="30", **good); assert r["stopped_at"] == "global_deadline"
r = run("good_hw_missing_inputs", **good); assert r["stages"]["0_environment"] and r["stopped_at"] == "1_inputs"
r = run("two_L4", **{**good, "FN_SMOKE_FAKE_GPUS": "\n".join(["NVIDIA L4, 23034 MiB, 8.9"] * 2)}); assert r["stopped_at"] == "0_environment"
r = run("four_T4", **{**good, "FN_SMOKE_FAKE_GPUS": "\n".join(["Tesla T4, 15360 MiB, 7.5"] * 4)}); assert r["stopped_at"] == "0_environment"
r = run("driver_570_cuda12_8", **{**good, "FN_SMOKE_FAKE_SMI": "Driver Version: 570.133.20   CUDA Version: 12.8"}); assert r["stopped_at"] == "0_environment"
r = run("ram_available_80GB", **{**good, "FN_SMOKE_FAKE_MEMINFO": MEM(190, 80)}); assert r["stopped_at"] == "0_environment"
r = run("tmp_free_10GB", **{**good, "FN_SMOKE_FAKE_TMPFREE": "10"}); assert r["stopped_at"] == "0_environment"
# never claim success for stages that did not run: emulate a run where only stage 0 executed and nothing set stopped_at
nb = json.load(open(NB, encoding="utf-8")); G = {"__name__": "__main__"}
os.environ.update(FN_SMOKE_WORK=os.path.abspath("val/incomplete"), FN_SMOKE_COMP=COMP); os.makedirs("val/incomplete", exist_ok=True)
codes = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
exec(compile(codes[0], "cell1", "exec"), G)
G["stage"]("0_environment", True); G["stage"]("1_inputs", True)   # pretend; later stages never ran, stopped_at stays None
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(compile(codes[-1], "cell_final", "exec"), G)
assert G["REPORT"]["stopped_at"] == "incomplete_unknown" and "NOT SHOWN TO WORK" in buf.getvalue(), buf.getvalue()[-400:]
print("incomplete run -> stopped_at=incomplete_unknown, verdict NOT SHOWN TO WORK")
print("ALL VALIDATION TESTS PASS")
