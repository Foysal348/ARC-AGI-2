"""Reproduce the E8 R2 per-unit AssertionError with the real Qwen tokenizer (CPU only)."""
import json, sys, traceback, re
import numpy as np
import tokenizers

NB, FX, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
cells = json.load(open(NB, encoding="utf-8"))["cells"]


class Tok:
    def __init__(self, d):
        self.t = tokenizers.Tokenizer.from_file(d + "/tokenizer.json")

    def encode(self, text, add_special_tokens=False):
        return self.t.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.t.decode(list(ids), skip_special_tokens=skip_special_tokens)


ch = json.load(open(FX + "/arc-agi_evaluation_challenges.json"))
s5 = "".join(cells[5]["source"])
ns = {"np": np, "json": json, "re": re, "tokenizer": Tok(MODEL), "MAX_MODEL_LEN": 81920}
exec(s5[s5.index("def valid_grid"):s5.index("def make_attempts")], ns)
exec("".join(cells[9]["source"]), ns)
s10 = "".join(cells[10]["source"])
exec(s10[:s10.index("def e8_score_batch")], ns)
units = sys.argv[4:]
ns["CHALLENGES"] = ch
ns["UNIT_TASK"] = {u: u.split("#")[0] for u in units}
ns["UNIT_Q"] = {u: int(u.split("#")[1]) for u in units}
for u in units:
    inp = ch[u.split("#")[0]]["test"][int(u.split("#")[1])]["input"]
    cand = {"grid": [list(reversed(r)) for r in inp], "source": "legacy_mirror"}
    try:
        views = ns["e8_make_views"](u, cand["grid"], "donor")
        reqs = [ns["e8_prepare_score_request"](u, cand, v, i) for i, v in enumerate(views)]
        print(u, "OK", [len(r["prompt_ids"]) for r in reqs], "boundary_equal", [r["boundary_equal"] for r in reqs])
    except Exception:
        tb = traceback.format_exc().strip().splitlines()
        print(u, "FAIL", " | ".join(tb[-4:]))
