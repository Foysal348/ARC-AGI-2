"""Local DRYRUN of the E12-K1 notebook: executes the REAL code cells with K1_DRYRUN=1 (fake tuner and fake engine
results, no GPU, no vLLM) and checks the orchestration: stages, config-file install/removal in a fake vLLM tree,
per-variant results, kernel-evidence parsing, report. usage: python run_dryrun.py <notebook>"""
import json, os, sys, tempfile, shutil
from pathlib import Path

NB = sys.argv[1]
WORK = Path(tempfile.mkdtemp(prefix="k1_dry_"))
FAKE_VLLM = WORK / "fake_vllm"; CFG = FAKE_VLLM / "model_executor" / "layers" / "quantization" / "utils" / "configs"
CFG.mkdir(parents=True)
(CFG / "N=1,K=1,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json").write_text("{}")   # must survive
DATA = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
            r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
os.environ.update(K1_DRYRUN="1", K1_WORKING=str(WORK), Q27X_ARC_ROOT=str(DATA), K1_VLLM_DIR=str(FAKE_VLLM),
                  K1_TUNE_SEC="30", PYTHONIOENCODING="utf-8")
nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
snap = {}
for i, src in enumerate(code):
    exec(compile(src, f"<cell{i + 1}>", "exec"), G)
    snap[i + 1] = sorted(p.name for p in CFG.glob("*.json"))
R = json.load(open(WORK / "k1" / "results.json", encoding="utf-8"))
fails = []
def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: fails.append(m)
check(set(R["stages"]) >= {"install", "tune", "bench"}, f"stages {sorted(R['stages'])}")
cf = R["stages"]["tune"]["config_files"]
check(len(cf) == 5 and all("device_name=NVIDIA_L4" in f for f in cf), f"5 L4 config files: {len(cf)}")
check(len(R["stages"]["tune"]["timings"]) == 5, "timings for all 5 shapes")
B = R["stages"]["bench"]
check(set(B) == {"default", "tuned", "marlin"} and all("workloads" in B[v] for v in B), f"3 variants ran: {sorted(B)}")
check(len(B["tuned"]["configs_in_vllm"]) == 5 and B["default"]["configs_in_vllm"] == [] and B["marlin"]["configs_in_vllm"] == [],
      "configs installed only for the tuned variant")
check(snap[len(code)] == ["N=1,K=1,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json"],
      "vLLM's own config files untouched and ours removed at the end")
check(B["marlin"].get("env_force_fp8_marlin") == "1" and B["default"].get("env_force_fp8_marlin") is None, "Marlin env only for marlin")
S = R["summary"]
check(S["tuned"]["speedup_vs_default"]["short"] == 1.15 and S["marlin"]["speedup_vs_default"]["long"] == 1.25, "speedups computed")
check(S["marlin"]["greedy_common_prefix_vs_default"] == [3, 3, 3, 3], "greedy agreement computed")
check((WORK / "k1" / "report.md").exists() and "E12-K1" in (WORK / "k1" / "report.md").read_text(encoding="utf-8"), "report written")
print(f"\n{len(fails)} failed checks" if fails else "\nALL CHECKS PASS")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if fails else 0)
