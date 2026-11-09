"""Part B: markdown table of train-pass-but-wrong attempts with manual classification.
Facts come from false_verif.json (grid sizes, wrong-cell counts, verifier overfit flags);
classification + evidence were assigned by reading cases/<build>_<task>_a<k>.txt (submitted source closure + wrong cells).
Classes: a = hard-coded / special-cased to train, b = rule underdetermined by train (simpler rule fits train, not test),
c = test input outside train distribution, d = output shape/size error, e = other.
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "false_verif.json")))

CLS = {
    ("W4", "247ef758", 1): ("b", "Same single wrong cell (6,12) gold 6 / pred 4 as both klee100 attempts: overlap priority between two pasted shapes is not fixed by the 3 train pairs (test 1 of 2 right)."),
    ("W4", "16b78196", 1): ("b", "Pred is identical to klee100 a1 (same 46 wrong cells): one object is put in the wrong wall cavity (rows 1-4 empty, rows 16-18 filled); 2 train pairs do not fix the cavity choice, and the program adds a 'score-0 fallback' path that only the test uses (transcript l.4055)."),
    ("K100", "247ef758", 1): ("b", "Test 1 of 2 right; one cell (6,12) gold 6 / pred 4, the overlap priority of pasted shapes."),
    ("K100", "247ef758", 2): ("b", "Independent re-derivation, same single wrong cell; placement sorted by colour id, `if cur==0 or cur==v` keeps the first writer."),
    ("K100", "f931b4a8", 2): ("c", "Test 1 is 4x4 (2x2 quadrants) and has a new colour 7; all train inputs are 8x8 or 16x16. The 'rows/cols/pos' fill-mode heuristic picks 'pos' and tiles a 2x2 cell, so 6 of 16 cells are wrong. Test 0 (8x8) is right."),
    ("K100", "b5ca7ac4", 2): ("a", "The verifier flagged `fixed_offset`: target columns are hard-coded `[0,5]` / `[C-5,C-10]`. The test background is 9 (train backgrounds were 0, 4 and 1), and 132 of 484 cells are wrong (frames land in the wrong column slot)."),
    ("K100", "2b83f449", 1): ("a", "The verifier flagged `fixed_offset`, and the program uses tuned per-row caps (`cap=2 if g<=3 else (1 if g<gb else 0)`) fitted to 2 train pairs. The test (19x17) is larger than any train input, and 8 cells swap 3/8."),
    ("K100", "16b78196", 1): ("b", "Pred is identical to W4 a1 (46 wrong cells, same positions): the cavity assignment is underdetermined by 2 train pairs."),
    ("K100", "3dc255db", 1): ("b", "3 cells: the ejected colour-2 cells go out of the wrong end (pred (5,9),(6,9); gold (9,12)). The exit tip is chosen by a heuristic sort `(dl,-md)`. The test also has colours 2 and 8, which never occur in train."),
    ("K100", "35ab12c3", 1): ("b", "`seg_cells` returns [] for any hull edge that is not 0/45/90 degrees. This fits the 3 train pairs, but the test (21x21, larger than train) needs such an edge, so 3 cells are missing (gold 3, pred 0)."),
    ("K100", "35ab12c3", 2): ("b", "Same restriction in `seg` (returns [] for non-0/45/90 slopes), and the same 3 missing cells as a1."),
    ("K100", "2d0172a1", 1): ("d", "Output size is wrong on both tests (pred 11x13 vs gold 9x13; 17x15 vs 14x11). Size comes from fitted arithmetic (`H=2n+1`, `_round_half_even(n*wA,hA)`). The test inputs (25x25) are larger than train (20 rows) and have new colours 6, 7 and 8."),
    ("K100", "800d221b", 1): ("b", "1 cell: the hole centre gets `gmax` = the most frequent pattern colour, with ties broken by colour id; gold differs (3 vs 2). The test (30x30) is larger than train."),
    ("K100", "800d221b", 2): ("b", "5 cells (gold 2, pred 3): the colour of an arm is chosen by an adjacency-tally tie-break that train does not fix."),
    ("K100", "a395ee82", 2): ("b", "18 cells swap 6 and 8 in one copied block. The A/B colour roles come from a heuristic (`Acand`: 2 components, else the most frequent colour), and the placement direction from an angle classifier, both fitted to 3 train pairs."),
}

out = ["| build | task | att | group | tokens | train pairs | train in sizes | test in sizes (gold out) | wrong cells / test | verifier flag | class | evidence |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    k = (r["build"], r["task"], r["attempt"])
    c, ev = CLS[k]
    wrong = []
    for d in r["diff"]:
        wrong.append("ok" if d.get("shape_ok") and d["cells_wrong"] == 0 else (f"{d['cells_wrong']}/{d['cells']}" if d.get("shape_ok") else f"shape {list(d['pred'])} vs {list(d['gold'])}"))
    sz = lambda L: ", ".join(f"{a}x{b}" for a, b in L)
    out.append(f"| {r['build']} | {r['task']} | a{r['attempt']} | {r['group']} | {r['tokens']:,} | {r['n_train']} | {sz(r['train_in'])} | {sz(r['test_in'])} ({sz(r['gold'])}) | {'; '.join(wrong)} | {','.join(r['overfit_flags']) or '-'} | {c} | {ev} |")
from collections import Counter
cnt = Counter(CLS[(r["build"], r["task"], r["attempt"])][0] for r in rows if r["build"] == "K100")
out.append("")
out.append(f"klee100 class counts (13): {dict(sorted(cnt.items()))}")
open(os.path.join(HERE, "part_b_table.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
