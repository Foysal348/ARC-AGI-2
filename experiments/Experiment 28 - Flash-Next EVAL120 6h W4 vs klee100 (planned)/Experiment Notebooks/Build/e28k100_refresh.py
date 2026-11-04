"""E28 K100: one command after any input change (e.g. the final e28_sched_patch.py): rebuild the notebook, run the units locally,
rebuild the CPU kits, run the units again from the units kit's own embedded files, write the review diff vs K100Q and the flat
audit zip. Nothing is pushed. usage: python e28k100_refresh.py   (prints one summary line per step; exit 1 on any failure)
"""
import base64
import difflib
import hashlib
import json
import lzma
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
PUSH = Path(r"D:/ARC AGI 2/_scratch/push_e25")
K100Q_NB = Path(r"D:/ARC AGI 2/experiments/Experiment 26 - Flash-Next klee100 3bpw low-bit (planned)/Experiment Notebooks/"
                r"Q27-XHIGH-E26_K100_QUALITY.ipynb")
PAYLOAD = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/"
               r"payload/q27x_e15b_duck_9e4b89285136.bin")
KITS = ("units", "e2e-rerun", "e2e-commit", "real")
DATE = "2026-11-01"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, env=None, cwd=HERE):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(cwd),
                       env={**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"})
    return r.returncode, r.stdout + r.stderr


def main():
    ok = True
    rc, out = run(["build_e28_k100.py"])
    print("build:", out.strip().splitlines()[-1] if out.strip() else rc)
    ok &= rc == 0
    rc, out = run(["e28k100_unit_tests.py"])
    (HERE / f"e28k100_unit_tests_local_{DATE}.log").write_text(out, encoding="utf-8", newline="\n")
    print("units (local):", out.strip().splitlines()[-1], "| fails:", [l[:120] for l in out.splitlines() if l.startswith("FAIL")])
    ok &= rc == 0
    rc, out = run(["build_e28_k100_cpu_kit.py"])
    print("kits:", rc, *[l for l in out.splitlines() if l.startswith("kit ")], sep="\n  ")
    ok &= rc == 0
    nb = json.loads((PUSH / "q27-e28-k100-kit-units__esfersami50" / "q27-e28-k100-kit-units.ipynb").read_text(encoding="utf-8"))
    src = "".join(nb["cells"][0]["source"])
    files = json.loads(lzma.decompress(base64.b85decode(re.search(r'b85decode\("([^"]+)"\)', src).group(1))).decode("utf-8"))
    with tempfile.TemporaryDirectory(prefix="e28k100_ukit_") as d:
        d = Path(d)
        for n, v in files.items():
            (d / n).write_text(v, encoding="utf-8", newline="\n")
        with zipfile.ZipFile(PAYLOAD) as z:
            (d / "arc2_duck_original.py").write_bytes(z.read("arc2_duck.py"))
        env = {"E28KT_NB": "e26.ipynb", "E28KT_FACTS": "e28k100_build_facts.json", "E28KT_DUCK": "arc2_duck_original.py",
               "E28KT_K100Q_FIXTURE": "k100q_fixture.json", "E28KT_E18": "e18r2_cells01.ipynb", "E28KT_K100Q_NB": "k100q.ipynb",
               "E28KT_KLEE_CFG": "x", "E28KT_MERLIN_SRC": "x", "E28KT_K100Q_OUT": "x", "E28KT_SOL": "x"}
        rc, out = run([str(d / "e28k100_unit_tests.py")], env, cwd=d)
    print("units (from the units kit's files):", out.strip().splitlines()[-1], "| skips:", [l[:60] for l in out.splitlines() if l.startswith("SKIP")])
    ok &= rc == 0
    base = json.loads(K100Q_NB.read_text(encoding="utf-8"))
    old = "\n\n".join("".join(c["source"]) for c in base["cells"] if c["cell_type"] == "code")
    new = (HERE / "e28k100_code_export.py").read_text(encoding="utf-8")
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "K100Q 4506a71f (code cells)",
                                        "E28 K100 (readable code cells)", n=2))
    (HERE / "e28k100_review_diff_vs_k100q.diff").write_text(diff, encoding="utf-8", newline="\n")
    files = [(HERE / "e28k100_00_GPT_PROMPT.md", "00_GPT_PROMPT.md"), (EXP / "README.md", None),
             (EXP / "Experiment Notebooks/Q27-XHIGH-E28_K100_EVAL120_6H.ipynb", None)]
    files += [(HERE / n, None) for n in ("build_e28_k100.py", "build_e28_k100_cpu_kit.py", "e28k100_lib.py", "e28k100_unit_tests.py",
                                          f"e28k100_unit_tests_local_{DATE}.log", "e28k100_build_facts.json", "e28k100_kit_facts.json",
                                          "e28k100_code_export.py", "e28k100_review_diff_vs_k100q.diff", "e28k100_refresh.py",
                                          "e28_sched_patch.py", "E28_SCHED_API.md")]
    files += [(EXP / "Experiment Review/Agent Reports/E28_REPORT_scheduler_2026-11-01.md", None),
              (Path(r"D:/ARC AGI 2/experiments/Experiment 26 - Flash-Next klee100 3bpw low-bit (planned)/Experiment Results/"
                    r"E26_K100Q_GPU_v1_2026-11-01.md"), None),
              (Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/"
                    r"E25_NB4T_GPU_v1_2026-11-01.md"), None),
              (PUSH / "q27-xhigh-e28-k100-eval120-6h__esfersami50/kernel-metadata.json", "gpu_kernel-metadata.json")]
    for k in KITS:
        dd = PUSH / f"q27-e28-k100-kit-{k}__esfersami50"
        files += [(dd / f"q27-e28-k100-kit-{k}.ipynb", None), (dd / "kernel-metadata.json", f"kit-{k}_kernel-metadata.json")]
    zp = EXP / "Experiment Review/Audit Zips" / f"E28_K100_audit_FLAT_{DATE}.zip"
    zp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f, name in files:
            if f.is_file():
                z.write(f, name or f.name)
            else:
                print("  (not in the zip, missing):", f)
        names = z.namelist()
    assert len(names) == len(set(names)) and all("/" not in n for n in names)
    print("audit zip:", zp, zp.stat().st_size, "B,", len(names), "files, sha256", sha(zp))
    facts = json.loads((HERE / "e28k100_build_facts.json").read_text(encoding="utf-8"))
    print("notebook:", facts["notebook_sha256"], facts["bytes"], "B | scheduler", facts["sched"].get("sha256"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
