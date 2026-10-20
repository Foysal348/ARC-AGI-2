"""H2 evidence: R3 prompt length (Qwen tokenizer + chat template) on all 120 evaluation tasks, with and without Poetiq feedback.
usage: python measure_feedback_prompt_tokens.py <R3 notebook> <model dir with tokenizer.json + chat_template.jinja> <arc-agi_evaluation_challenges.json>
Needs: tokenizers, jinja2, numpy. Executes only R3's cell 5 (pure function definitions)."""
import json, random, sys
from pathlib import Path
import tokenizers, jinja2

NB, MODEL, CH = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tok = tokenizers.Tokenizer.from_file(str(MODEL / "tokenizer.json"))
env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"]); env.filters["tojson"] = lambda v, **k: json.dumps(v, **k)
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m)); env.globals["strftime_now"] = lambda f: ""
tpl = env.from_string((MODEL / "chat_template.jinja").read_text(encoding="utf-8"))
ns = {}; exec("".join(json.load(open(NB, encoding="utf-8"))["cells"][5]["source"]), ns)
ch = json.load(open(CH))
# a realistic ~1.2k-token program (14 helpers), standing in for each of the 5 feedback solutions
CODE = "import numpy as np\n" + "\n".join(f"def helper_{k}(grid):\n    # find objects and apply rule step {k}\n    h, w = grid.shape\n    out = grid.copy()\n    for r in range(h):\n        for c in range(w):\n            if grid[r, c] == {k % 10}:\n                out[r, c] = (grid[r, c] + 1) % 10\n    return out\n" for k in range(14)) + "\ndef transform(grid):\n    return helper_0(grid)\n"
print("code tokens", len(tok.encode(CODE).ids))

def ntok(task, sols):
    msg = ns["poetiq_build_message"](task, 1, 0, sols, variant=1)
    txt = tpl.render(messages=[{"role": "user", "content": msg}], add_generation_prompt=True, enable_thinking=True, reasoning_effort="xhigh")
    return len(tok.encode(txt).ids)

def fb(task, frac, rng):     # R3's own same-shape feedback format, frac of cells wrong
    parts = []
    for i, ex in enumerate(task["train"]):
        t = ex["output"]; pred = [[(v + 1) % 10 if rng.random() < frac else v for v in row] for row in t]
        parts.append(f"Solves Example #{i+1} incorrectly. Wrong cells appear as prediction/correct:\n```\n{ns['poetiq_array_diff'](pred, t)}\n```\nOutput accuracy: 0.50")
    return "\n\n".join(parts)

res = {"iter1": [], "fb5_20pct": [], "fb5_100pct": []}
rng = random.Random(0)
for tid, task in ch.items():
    res["iter1"].append(ntok(task, []))
    for key, frac in (("fb5_20pct", .2), ("fb5_100pct", 1.0)):
        res[key].append(ntok(task, [{"code": CODE, "feedback": fb(task, frac, rng), "score": .1 * k} for k in range(5)]))
# phase1_cap = min(77824, 106496 - prompt - 8192 - 256): full 77,824 needs prompt <= 20,224; >= 49,152 needs prompt <= 48,896
for k, v in res.items():
    v.sort(); n = len(v)
    print(f"{k:12s} median {v[n//2]:6d} max {v[-1]:6d} | think<77,824: {sum(x>20224 for x in v):3d}/{n} | "
          f"think<49,152: {sum(x>48896 for x in v):3d} | skipped(<4,096): {sum(x>106496-8192-256-4096 for x in v)}")
