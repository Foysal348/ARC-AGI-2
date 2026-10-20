"""Token cost of writing test-output grids after the code answer (Claude, 2026-10-14).
usage (from the bundle root): python analysis/measure_grid_answer_tokens.py
needs: pip install tokenizers
Measured result (2026-10-14):
  tokens per cell 2.01; per test grid median 764, p90 1804, max 1804 (n=172);
  per task (all test grids) median 972, p90 2648, max 3608; 18/120 tasks > 2k tokens, 7 > 3k;
  R4 forced-close code answers: median 2581, max 6562 tokens (n=24);
  code + grids would exceed the 8,192 answer cap in 2/24 (88bcf3b4: 6562 + 2434).
"""
import json
from pathlib import Path
import tokenizers

ROOT = Path(__file__).resolve().parents[1]
tok = tokenizers.Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
sol = json.load(open(ROOT / "data" / "arc-agi_evaluation_solutions.json"))


def grid_tokens(g):  # the DIRECT answer format: a fenced block of rows of space-separated digits
    return len(tok.encode("```grid\n" + "\n".join(" ".join(map(str, r)) for r in g) + "\n```").ids)


per_grid = [(grid_tokens(g), len(g) * len(g[0])) for gs in sol.values() for g in gs]
per_task = {t: sum(grid_tokens(g) for g in gs) for t, gs in sol.items()}
print("tokens per cell:", round(sum(a for a, _ in per_grid) / sum(b for _, b in per_grid), 3))
v = sorted(a for a, _ in per_grid)
print(f"per test grid: median {v[len(v)//2]}, p90 {v[int(.9*len(v))]}, max {v[-1]} (n={len(v)})")
w = sorted(per_task.values())
print(f"per task: median {w[len(w)//2]}, p90 {w[int(.9*len(w))]}, max {w[-1]}; > 2k: {sum(x > 2000 for x in w)}/120, > 3k: {sum(x > 3000 for x in w)}")
its = [json.loads(l) for l in open(ROOT / "runs" / "poetiq_r4_eval12" / "poetiq_iterations.jsonl")]
forced = [i for i in its if i["mode"] == "forced_close"]
p2 = sorted(i["p2_tokens"] for i in forced)
print(f"R4 forced-close code answers: median {p2[len(p2)//2]}, max {p2[-1]} (n={len(p2)})")
over = [(i["task_id"], i["iteration"], i["p2_tokens"] + per_task[i["task_id"]]) for i in forced if i["p2_tokens"] + per_task[i["task_id"]] > 8192]
print("code + grids > 8,192:", len(over), "of", len(forced), over)
