"""Execute the notebook's cells IN ORDER, exactly as Kaggle would (shared globals, %%writefile cells written to disk,
the launcher building its own cfg), with /kaggle/working and /kaggle/input mapped to a temp folder. No GPU here, so
the launcher takes the DRYRUN path (fake OpenAI server) -- this is the end-to-end test that would have caught run 2's
KeyError. Windows-only gaps (no os.killpg, no libc via find_library('c'), no _posixsubprocess) are listed and allowed.

usage: python run_notebook_dryrun_local.py <notebook> <competition data dir>
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

NB, DATA = sys.argv[1], Path(sys.argv[2])
root = Path(tempfile.mkdtemp(prefix="q27nb_"))
work = root / "kaggle" / "working"
inp = root / "kaggle" / "input"
comp = inp / "arc-prize-2026-arc-agi-2"
for d in (work, comp):
    d.mkdir(parents=True)
for f in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json", "arc-agi_test_challenges.json"):
    shutil.copy(DATA / f, comp / f)
W, I = work.as_posix(), inp.as_posix()


def rewrite(src):
    return src.replace("/kaggle/working", W).replace("/kaggle/input", I)


cells = json.load(open(NB, encoding="utf-8"))["cells"]
g = {"__name__": "__main__"}
log = []
t_all = time.time()
os.chdir(work)
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = rewrite("".join(c["source"]))
    first = src.splitlines()[0] if src else ""
    t = time.time()
    try:
        if first.startswith("%%writefile "):
            path = Path(first.split(None, 1)[1].strip())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(src.splitlines()[1:]) + "\n", encoding="utf-8")
            status = "written"
        else:
            exec(compile(src, f"<cell {i}>", "exec"), g)
            status = "ran"
    except SystemExit as e:
        status = f"SystemExit {e.code}"
    except BaseException:
        status = "EXCEPTION\n" + traceback.format_exc()[-3000:]
    log.append((i, first[:60], status, round(time.time() - t, 1)))
    print(f"--- cell {i} [{status.splitlines()[0]}] {time.time() - t:.1f}s :: {first[:60]}", flush=True)
    if status.startswith("EXCEPTION"):
        print(status, flush=True)

print(f"\nnotebook finished in {time.time() - t_all:.0f}s; working dir {W}")
rep = json.loads((work / "dryrun_report.json").read_text()) if (work / "dryrun_report.json").exists() else None
assert rep is not None, "no dryrun_report.json -- the DRYRUN path did not complete"
WINDOWS_ONLY = {"guard_ctypes_libc_open", "guard_ctypes_libc_fopen", "guard_fork_exec"} if os.name == "nt" else set()
bad = [k for k, v in rep["checks"].items() if not v["ok"] and k not in WINDOWS_ONLY]
print("platform-allowed failures:", sorted(k for k in WINDOWS_ONLY if not rep["checks"].get(k, {}).get("ok", True)))
assert not any(s.startswith("EXCEPTION") for _, _, s, _ in log), "a cell raised"
assert not bad, {k: rep["checks"][k] for k in bad}
ev = json.loads((work / "agent_eval" / "eval_report.json").read_text())
print("eval:", {k: ev.get(k) for k in ("official_score_pct", "agent_only_official_pct", "direct_only_official_pct")},
      ev["complementarity"].get("rescue_statuses"))
print("NOTEBOOK DRYRUN LOCAL PASS")
