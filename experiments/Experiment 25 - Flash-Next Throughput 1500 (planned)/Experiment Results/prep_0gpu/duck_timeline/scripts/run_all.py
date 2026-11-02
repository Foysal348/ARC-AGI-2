"""Re-run both K100Q offline analyses end to end (CPU only, numeric fields only).
Usage: python -I run_all.py [run_q27x_dir] [out_root]
Defaults: the saved K100Q output and the folder that holds this script.
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/ARC AGI 2/_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x")
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent
data, o1, o2 = ROOT / "data", ROOT / "out_analysis1", ROOT / "out_analysis2"
py = [sys.executable, "-I"]
for cmd in ([*py, str(HERE / "extract.py"), str(RUN), str(data)],
            [*py, str(HERE / "timeline.py"), str(data), str(o1)],
            [*py, str(HERE / "tool_to_server.py"), str(o1)],
            [*py, str(HERE / "reprefill.py"), str(data), str(o2), str(o1 / "analysis1_summary.json")]):
    print(">>", " ".join(cmd[2:3]), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (ROOT / f"log_{Path(cmd[2]).stem}.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        sys.exit(f"failed: {cmd[2]} rc {r.returncode}")
print("done:", ROOT)
