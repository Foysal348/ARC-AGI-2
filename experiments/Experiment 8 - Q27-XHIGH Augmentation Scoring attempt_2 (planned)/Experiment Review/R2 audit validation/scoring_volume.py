"""E8 R2 scoring volume on a challenge file: candidates per unit (GPT's own e8_build_candidates, attempt_1 = a
stand-in distinct grid) x 8 views, prompt tokens measured with the real Qwen tokenizer (GPT's own view builder)."""
import json, sys, re
import numpy as np
import tokenizers

NB, CH, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
cells = json.load(open(NB, encoding="utf-8"))["cells"]


class Tok:
    def __init__(self, d):
        self.t = tokenizers.Tokenizer.from_file(d + "/tokenizer.json")

    def encode(self, text, add_special_tokens=False):
        return self.t.encode(text, add_special_tokens=add_special_tokens).ids


ch = json.load(open(CH))
s5 = "".join(cells[5]["source"])
ns = {"np": np, "json": json, "re": re, "tokenizer": Tok(MODEL), "MAX_MODEL_LEN": 81920}
exec(s5[s5.index("def valid_grid"):s5.index("def make_attempts")], ns)
exec("".join(cells[9]["source"]), ns)
s10 = "".join(cells[10]["source"])
exec(s10[:s10.index("def e8_score_batch")], ns)
UT, UQ = {}, {}
for t in ch:
    for q in range(len(ch[t]["test"])):
        UT[f"{t}#{q}"] = t; UQ[f"{t}#{q}"] = q
ns.update(CHALLENGES=ch, UNIT_TASK=UT, UNIT_Q=UQ)
units = scored = reqs = toks = 0
per_req = []
for u in UT:
    inp = ch[UT[u]]["test"][UQ[u]]["input"]
    a1 = [[(v + 1) % 10 for v in r] for r in inp]          # a model grid distinct from input / mirror
    a2 = inp                                                # v1.4.4 make_attempts: attempt_2 = test input
    cands = ns["e8_build_candidates"](u, a1, a2, 4)
    units += 1
    if not any(c["grid"] not in (a1, a2) for c in cands):
        continue
    scored += 1
    v = ns["e8_make_views"](u, cands[0]["grid"], "donor")
    n = len(ns["e8_prepare_score_request"](u, cands[0], v[0], 0)["prompt_ids"])
    reqs += 8 * len(cands); toks += 8 * len(cands) * n; per_req.append(n)
per_req.sort()
print(f"units {units} | scored (has a 'novel' candidate) {scored} | requests {reqs} | prompt tokens total {toks/1e6:.1f} M | "
      f"per-request median {per_req[len(per_req)//2]} max {per_req[-1]}")
