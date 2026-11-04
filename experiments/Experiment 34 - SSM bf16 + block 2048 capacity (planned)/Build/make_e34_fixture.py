"""E34: the unit-test fixture from the E25 NB-S1 full GPU run (B's exact boot path: D3s + EMBED_HOST, block 4096, fp32 SSM, SW
1024). Read only: D:/ARC AGI 2/_scratch/e25_nbs1_full_out/nbdf (nbdf_results.json + logs/server_nbs1_b_t1.log). Writes
e34_log_fixture.json next to this file: the worker records (nb1_cache_info, cache tensors dropped), the boot capacity facts,
the KV layout, the DFlash read-back checks and the boot log lines E34 parses. No generated text."""
import hashlib
import json
import re
from pathlib import Path

SRC = Path(r"D:/ARC AGI 2/_scratch/e25_nbs1_full_out/nbdf")
R = json.loads((SRC / "nbdf_results.json").read_text(encoding="utf-8"))
o = R["sections"]["Q"]["nbs1_b"]
log = (SRC / "logs" / "server_nbs1_b_t1.log").read_text(encoding="utf-8", errors="replace")
keep = re.compile(r"kv cache group sizes|Padding mamba page size|GPU KV cache size|Maximum concurrency|Setting attention block size|"
                  r"non-default args:|draft_sw_block")
lines = []
for ln in log.splitlines():
    if keep.search(ln) and ln not in lines:
        lines.append(ln[:4000])
workers = {"ranks": [{k: v for k, v in w.items() if k != "cache_tensors"} for w in o["workers"]["ranks"]],
           "num_gpu_blocks": o["workers"].get("num_gpu_blocks")}
fx = {"source": "E25 NB-S1 full GPU run 2026-11-01/02, section Q / nbs1_b (= B's boot path), " + str(SRC),
      "results_sha256": hashlib.sha256((SRC / "nbdf_results.json").read_bytes()).hexdigest(),
      "log_sha256": hashlib.sha256((SRC / "logs" / "server_nbs1_b_t1.log").read_bytes()).hexdigest(),
      "workers": workers, "capacity_after_ready": o.get("capacity_after_ready"), "kv_layout": o.get("kv_layout"),
      "dflash_checks": (o.get("dflash_readback") or {}).get("checks"), "log_lines": lines[:40]}
out = Path(__file__).resolve().with_name("e34_log_fixture.json")
out.write_bytes((json.dumps(fx, indent=1, default=str) + "\n").encode("utf-8"))
print(out, out.stat().st_size, len(lines))
