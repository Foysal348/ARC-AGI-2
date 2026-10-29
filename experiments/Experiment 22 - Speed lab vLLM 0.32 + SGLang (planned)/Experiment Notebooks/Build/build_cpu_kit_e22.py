"""E22 CPU kit: the Exp 14 engine-matrix CPU kit (build_cpu_kit_engine.py: EMS / VMS fake-server harnesses on a Kaggle CPU
kernel with the GPU notebooks' pinned image) pointed at the two E22 Speed Lab notebooks:
  e22_sglang = harness_ems_nb.py on Q27-XHIGH-E22_SPEEDLAB_SGLANG.ipynb with HARNESS_EMS_PART=SPEED
  e22_vllm   = harness_vms_nb.py on Q27-XHIGH-E22_SPEEDLAB_VLLM032.ipynb
Dataset foysalemonshanto/q27-e22-cpu-kit, kernel foysalemonshanto/q27-e22-cpu-harness (Build/cpu_kernel_e22/).
The harnesses' arm-specific checks were written for the EMS / VMS arm sets: their FAIL lines are read against the E22
plan (expected arm-name mismatches vs real errors), see Experiment Review."""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
E14B = Path(r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Notebooks/Build")
s = (E14B / "build_cpu_kit_engine.py").read_text(encoding="utf-8")


def rep(text, a, b, n=1):
    assert text.count(a) == n, (text.count(a), a[:100])
    return text.replace(a, b)


s = rep(s, 'KIT_SLUG = "foysalemonshanto/q27-engine-cpu-kit"', 'KIT_SLUG = "foysalemonshanto/q27-e22-cpu-kit"')
s = rep(s, 'KERNEL_SLUG = "foysalemonshanto/q27-engine-cpu-harness"', 'KERNEL_SLUG = "foysalemonshanto/q27-e22-cpu-harness"')
s = rep(s, 'OUT_DIR = HERE / "cpu_kit_engine"', 'OUT_DIR = Path(r"' + str(HERE) + r'") / "cpu_kit_e22"')
s = rep(s, '''RUNS = [("ems_all", "e14src/harness/harness_ems_nb.py", NB_SAFE, {"HARNESS_EMS_PART": "ALL"}),
        ("ems_safe", "e14src/harness/harness_ems_nb.py", NB_SAFE, {"HARNESS_EMS_PART": "SAFE"}),
        ("ems_risky", "e14src/harness/harness_ems_nb.py", NB_RISKY, {"HARNESS_EMS_PART": "RISKY"}),
        ("ems_mtpx", "e14src/harness/harness_ems_nb.py", NB_MTPX, {"HARNESS_EMS_PART": "MTPX"}),
        ("vms", "e14src/harness/harness_vms_nb.py", NB_VMS, {}),
        ("vms_bad_int4", "e14src/harness/harness_vms_nb.py", NB_VMS, {"VMS_BAD_INT4": "1"})]''',
        '''NB_E22S = "Q27-XHIGH-E22_SPEEDLAB_SGLANG.ipynb"
NB_E22V = "Q27-XHIGH-E22_SPEEDLAB_VLLM032.ipynb"
E22NB = Path(r"''' + str(HERE.parent) + '''")
RUNS = [("e22_sglang", "e14src/harness/harness_ems_nb.py", NB_E22S, {"HARNESS_EMS_PART": "SPEED"}),
        ("e22_vllm", "e14src/harness/harness_vms_nb.py", NB_E22V, {})]''')
s = rep(s, '''for nb in (NB_SAFE, NB_RISKY, NB_MTPX, NB_VMS):
    files["nb/" + nb] = (HERE.parent / nb).read_bytes()''', '''for nb in (NB_E22S, NB_E22V):
    files["nb/" + nb] = (E22NB / nb).read_bytes()''')
s = rep(s, 'KD = HERE / "cpu_kernel_engine"', 'KD = Path(r"' + str(HERE) + r'") / "cpu_kernel_e22"')
s = rep(s, '"id": KERNEL_SLUG, "title": "Q27 Engine CPU Harness"', '"id": KERNEL_SLUG, "title": "Q27 E22 CPU Harness"')
s = rep(s, '''for nb in (NB_SAFE, NB_RISKY, NB_MTPX, NB_VMS):
    print("notebook", nb, hashlib.sha256((HERE.parent / nb).read_bytes()).hexdigest()[:16])''', '''for nb in (NB_E22S, NB_E22V):
    print("notebook", nb, hashlib.sha256((E22NB / nb).read_bytes()).hexdigest()[:16])''')
exec(compile(s, str(E14B / "build_cpu_kit_engine.py"), "exec"),
     {"__file__": str(E14B / "build_cpu_kit_engine.py"), "__name__": "__main__"})
