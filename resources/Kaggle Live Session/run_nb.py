"""Run the code cells of a local .ipynb, in order, inside the LIVE Kaggle kernel (see run_cell.py).

Every cell's full output is appended to --log; a short status line per cell goes to stdout.
Overrides are exact text replacements applied to the cell sources before running:
  --set "VALIDATION_MODE = False=>VALIDATION_MODE = True"

Usage (venv _scratch/jmcp_venv):
  python run_nb.py --nb X.ipynb --log out.log [--from 0] [--to 46] [--set "a=>b"]... [--timeout 7200]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from run_cell import make_client, show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="end", type=int, default=10**9)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=7200)
    a = ap.parse_args()
    cells = json.loads(Path(a.nb).read_text(encoding="utf-8"))["cells"]
    subs = [s.split("=>", 1) for s in a.set]
    used = {s[0]: 0 for s in subs}
    log = open(a.log, "a", encoding="utf-8")
    with make_client() as k:
        for i, c in enumerate(cells):
            if not (a.start <= i <= a.end) or c.get("cell_type") != "code":
                continue
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for old, new in subs:
                if old in src:
                    src = src.replace(old, new); used[old] += 1
            t0 = time.time()
            log.write(f"\n===== cell {i} start {time.strftime('%H:%M:%S')} =====\n"); log.flush()
            r = k.execute(src, timeout=a.timeout)
            show(r.get("outputs", []), out=log)
            st = r.get("status")
            line = f"[cell {i}] {st} in {time.time() - t0:.0f} s"
            log.write(line + "\n"); log.flush()
            print(line, flush=True)
            if st != "ok":
                print(f"stopped at cell {i}; see the log", flush=True)
                break
    missing = [o for o, n in used.items() if n == 0]
    if missing:
        print("WARNING: these --set texts were not found in any run cell:", missing)


if __name__ == "__main__":
    main()
