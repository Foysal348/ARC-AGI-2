"""E20 = S1-60K (LB 34.58; Experiment 10, sha ed530071) with ONE change: the hidden-rerun drain starts 15 min before the
hard stop instead of 8 (user 2026-10-28, after E19 cleared the engine and the drain loss was confirmed: D66 lost 14 of 15
drained answers because re-prefilling ~16 x 40-60k prefixes at ~1,450 tok/s takes about 7-8 of the 8 minutes).
Needed margin ~ sum(active prefixes) / 1,450 tok/s + the answer decode: about 16 x 43k / 1,450 = 474 s + ~3 min = ~11-12 min.
Only the RERUN_ALL row of CELL 1 changes; the builder asserts every other cell is byte-identical to S1-60K.
usage: python build_e20.py  -> ../Q27-XHIGH-E20_S1_60K_DRAIN15.ipynb"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
SRC = NBDIR.parents[1] / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)" / "Experiment Notebooks" / \
    "Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb"
SRC_SHA = "ed530071"
OUT = NBDIR / "Q27-XHIGH-E20_S1_60K_DRAIN15.ipynb"
PUSHED: set = set()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


assert sha(SRC).startswith(SRC_SHA), sha(SRC)
nb = json.load(open(SRC, encoding="utf-8"))
cells = [src(c) for c in nb["cells"]]
c1 = cells[2]
OLD = '''                  # v1.4.4: admit until the end (NVARC style); drain 8 min before the hard stop, admit 2 min before drain
                  "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60},'''
NEW = '''                  # v1.4.4: admit until the end (NVARC style); drain 8 min before the hard stop, admit 2 min before drain
                  # E20 (2026-10-28): drain 15 min before the hard stop (8 min lost 14 of 15 drained answers in D66)
                  "drain_margin_sec": 15 * 60, "admit_margin_sec": 2 * 60},'''
assert c1.count(OLD) == 1, c1.count(OLD)
i_rerun = c1.index('"RERUN_ALL"')
assert c1.index(OLD) > i_rerun, "the replaced row must be the RERUN_ALL row"
c1 = c1.replace(OLD, NEW)
MD = ("# E20 — S1-60K with a 15-minute final drain\n\nS1-60K (LB 34.58, sha `ed530071`) with ONE change: the hidden rerun "
      "(`RERUN_ALL`) drains 15 min before the hard stop instead of 8 (D66 lost 14 of 15 drained answers to the re-prefill; "
      "E19 found no engine fault on S1's vLLM 0.19 config). Everything else is byte-identical.\n\n---\n\n")
nb["cells"][0]["source"] = (MD + cells[0]).splitlines(keepends=True)
nb["cells"][2]["source"] = c1.splitlines(keepends=True)
new = [src(c) for c in nb["cells"]]
changed = [i for i in range(len(cells)) if new[i] != cells[i]]
assert changed == [0, 2], changed
for c in nb["cells"]:
    if c["cell_type"] == "code":
        compile(src(c), "<cell>", "exec")
out = OUT if not (OUT.exists() and sha(OUT) in PUSHED) else OUT.with_name(OUT.stem + "_REBUILD.ipynb")
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
out.write_bytes(data)
print(out.name, hashlib.sha256(data).hexdigest()[:8], f"{len(data) / 1024:.0f} kB", "| changed cells", changed)
