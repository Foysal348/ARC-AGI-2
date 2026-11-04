"""capacity_results.json -> markdown tables (capacity_table.md)."""
import json, sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("capacity_results.json"))
R = json.loads(p.read_text(encoding="utf-8"))
L = []
L.append("| Arm | groups | stride PP0 / PP1 (B) | pool (usable) | F adm | F steady | F no-retain | SW adm | mc@131k s24 |")
L.append("|---|---|---|---|---|---|---|---|---|")
for r in R:
    if "error" in r:
        L.append(f"| {r['arm']} | ERROR: {r['error'][-200:].replace(chr(10), ' ').replace('|', '/')} |||||||| ")
        continue
    sw = [g.get("sw_admission_blocks") for g in r["sched_groups"] if "sw_admission_blocks" in g]
    s = r["stride_per_rank"]
    L.append(f"| {r['arm']} | {r['kv_groups']} | {s[0]:,} / {s[2]:,} | {min(r['num_blocks'])} ({r['usable_blocks']}) | "
             f"{r['F_admission']} | {r['F_steady']} | {r['F_steady_P4_no_retain']} | {sw[0] if sw else '-'} | "
             f"{r['max_concurrency_131072']['24']:.2f} |")
L.append("")
for basis in ("F_admission", "F_steady", "F_steady_P4_no_retain"):
    L.append(f"Residents ({basis}) = min(seqs, (usable - ring) // (ceil(L / 4096) + F)); columns 40k / 64k / 90k / 116k")
    L.append("")
    L.append("| Arm | seqs 24 | seqs 28 | seqs 32 |")
    L.append("|---|---|---|---|")
    for r in R:
        if "error" in r:
            continue
        cells = []
        for s in ("24", "28", "32"):
            d = r["residents"][basis][s]
            cells.append(" / ".join(str(d[k]) for k in ("40000", "64000", "90000", "116000")))
        L.append(f"| {r['arm']} | " + " | ".join(cells) + " |")
    L.append("")
L.append("Real KVCacheManager holding (sim): fixed blocks per request after 200 decode steps; pool used per request")
L.append("")
L.append("| Arm | P=40k | P=64k | P=90k |")
L.append("|---|---|---|---|")
for r in R:
    sim = r.get("sim") or {}
    if "runs" not in sim:
        L.append(f"| {r['arm']} | sim error: {str(sim.get('error'))[:150]} | | |")
        continue
    cells = [f"fixed {run['fixed_per_request'][0]}, ctx {run['context_blocks']}, used/req {run['pool_used_per_request']:.2f}"
             for run in sim["runs"]]
    L.append(f"| {r['arm']} | " + " | ".join(cells) + " |")
L.append("")
for r in R:
    if r.get("pressure"):
        L.append(f"Pressure (#62752 port) {r['arm']}: " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'cfg'}
                                                                      if isinstance(v, dict) else v
                                                                      for k, v in r["pressure"].items()}))
        L.append("")
    if r.get("swmask_test"):
        L.append(f"SW mask test ({r['arm']}): " + json.dumps({k: v for k, v in r["swmask_test"].items() if k != "tb"}))
        L.append("")
out = Path(p).with_name("capacity_table.md")
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
