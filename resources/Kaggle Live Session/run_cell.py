"""Run code in a LIVE Kaggle interactive notebook session (e.g. 4xL4) and print its outputs.

The user opens the session in the browser and saves either the websocket *channels* URL
(DevTools > Network > Socket > .../proxy/api/kernels/<id>/channels?session_id=...) or the official
"Run > Kaggle Jupyter Server > VS Code compatible URL" (ends in /proxy) into
_scratch/kaggle_live/channels_url.txt. That URL is signed: it is never printed.
No Kaggle API token is used (token=None -> signed proxy auth only).

Usage (venv _scratch/jmcp_venv):
  python run_cell.py "print(1+1)"
  python run_cell.py --file some_cell.py [--timeout 3600]
"""
import argparse
import sys
from pathlib import Path

from code_sandboxes.sandboxes.kaggle.client import KaggleKernelClient

URL_FILE = Path(__file__).with_name("channels_url.txt")


def show(outputs, out=sys.stdout):
    for o in outputs:
        kind = o.get("output_type")
        if kind == "stream":
            out.write(o.get("text", ""))
        elif kind in ("execute_result", "display_data"):
            out.write(str(o.get("data", {}).get("text/plain", "")) + "\n")
        elif kind == "error":
            out.write(("\n".join(o.get("traceback", [])) or f"{o.get('ename')}: {o.get('evalue')}") + "\n")


def make_client():
    url = URL_FILE.read_text(encoding="utf-8-sig").strip()
    if url.rstrip("/").endswith("/proxy"):
        # Official VS Code compatible URL = the proxy base. Attach to the notebook's running kernel
        # (the most recently active one) instead of starting a new one.
        import json, urllib.request
        base = url.rstrip("/")
        kernels = json.load(urllib.request.urlopen(base + "/api/kernels", timeout=30))
        if not kernels:
            sys.exit("no running kernel behind this URL: run a cell in the browser first")
        kid = max(kernels, key=lambda x: x.get("last_activity") or "")["id"]
        return KaggleKernelClient(server_url=base, kernel_id=kid, token=None)
    return KaggleKernelClient.from_channels_url(url, token=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    code = Path(a.file).read_text(encoding="utf-8") if a.file else a.code
    if not code:
        sys.exit("give code or --file")
    with make_client() as k:
        r = k.execute(code, timeout=a.timeout)
    show(r.get("outputs", []))
    print(f"[status {r.get('status')} | execution_count {r.get('execution_count')}]")


if __name__ == "__main__":
    main()
