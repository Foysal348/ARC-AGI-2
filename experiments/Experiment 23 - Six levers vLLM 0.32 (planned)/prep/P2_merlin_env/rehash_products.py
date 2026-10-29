"""E23 P2: after glibc235_fix.py changed .so bytes, re-record <tree>/.build-products.sha256 for the SAME file list
(the format Merlin's install-env.sh writes: '<sha256>  <path relative to tree>'), so launch.py's product check and
install-env.sh --verify accept the patched files. Every other line keeps its hash; a changed line is reported.
usage: python rehash_products.py <merlin-tree>   -> prints one REHASH {...} JSON line"""
import hashlib
import json
import sys
from pathlib import Path

tree = Path(sys.argv[1])
sums = tree / ".build-products.sha256"
lines = sums.read_text(encoding="utf-8").splitlines()
out, changed = [], []
for ln in lines:
    digest, rel = ln.split("  ", 1)
    new = hashlib.sha256((tree / rel).read_bytes()).hexdigest()
    if new != digest:
        changed.append({"path": rel, "old": digest, "new": new})
    out.append(f"{new}  {rel}\n")
sums.write_text("".join(out), encoding="utf-8")
print("REHASH " + json.dumps({"files": len(lines), "changed": changed}))
