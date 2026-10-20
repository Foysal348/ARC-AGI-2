"""What kind of ARC-AGI-2 puzzle does Qwen3.8-27B solve? (Claude, 2026-10-14)

Label per evaluation task (120): solved by anyone = ARC Prize's verified Qwen3.8-27B run at any reasoning level
(xhigh / medium / low / none) OR any of our GPU runs in the ledger OR the archived PIV EVAL120 run (1818057f, 2ba387bc).
Features are computed from the training pairs only (interpretable transformation descriptors, not NVARC tiers).
usage: python capability_profile.py   (prints tables; writes capability_profile.csv next to this file)
"""
import csv, json
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FX = (HERE.parent.parent / "experiments" / "Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)" / "Experiment Notebooks"
      / "Build" / "Q27X_build" / "fixtures" / "fake_input" / "competitions" / "arc-prize-2026-arc-agi-2")
CH = json.load(open(FX / "arc-agi_evaluation_challenges.json"))
LED = {r["task"]: r for r in csv.DictReader(open(HERE / "ledger_all120.csv"))}
PIV_SOLVED = {"1818057f", "2ba387bc"}
LV = ("arcprize_xhigh", "arcprize_medium", "arcprize_low", "arcprize_none")


def comps(g, bg):
    """4-connected components of non-background cells (any colour mix)"""
    h, w = g.shape; seen = np.zeros_like(g, bool); n = 0
    for r in range(h):
        for c in range(w):
            if g[r, c] != bg and not seen[r, c]:
                n += 1; st = [(r, c)]; seen[r, c] = True
                while st:
                    y, x = st.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and g[yy, xx] != bg:
                            seen[yy, xx] = True; st.append((yy, xx))
    return n


def is_subgrid(small, big):
    H, W = big.shape; h, w = small.shape
    if h > H or w > W: return False
    return any(np.array_equal(big[r:r + h, c:c + w], small) for r in range(H - h + 1) for c in range(W - w + 1))


def features(t):
    tr = t["train"]; I = [np.array(p["input"]) for p in tr]; O = [np.array(p["output"]) for p in tr]
    same = all(i.shape == o.shape for i, o in zip(I, O))
    smaller = all(o.size < i.size for i, o in zip(I, O)); larger = all(o.size > i.size for i, o in zip(I, O))
    shape_rel = "same size" if same else ("output smaller" if smaller else ("output larger" if larger else "mixed sizes"))
    bg = Counter(np.concatenate([i.ravel() for i in I]).tolist()).most_common(1)[0][0]
    f = {"shape_rel": shape_rel, "n_train": len(tr), "n_test": len(t["test"])}
    f["in_cells"] = float(np.mean([i.size for i in I]))
    f["objects"] = float(np.mean([comps(i, bg) for i in I]))
    newc = [set(np.unique(o)) - set(np.unique(i)) for i, o in zip(I, O)]
    f["new_colours"] = any(newc)
    if same:
        ch = [float(np.mean(i != o)) for i, o in zip(I, O)]; f["changed_frac"] = float(np.mean(ch))
        add = sum(int(np.sum((i != o) & (i == bg))) for i, o in zip(I, O))
        rem = sum(int(np.sum((i != o) & (o == bg))) for i, o in zip(I, O))
        tot = sum(int(np.sum(i != o)) for i, o in zip(I, O)) or 1
        f["edit_kind"] = ("only draws on background" if add == tot else "only erases to background" if rem == tot
                          else "recolours existing cells" if add == 0 and rem == 0 else "mixed edits")
    else:
        f["changed_frac"] = None; f["edit_kind"] = None
    f["crop_of_input"] = (not same) and smaller and all(is_subgrid(o, i) for i, o in zip(I, O))
    f["tiles_input"] = (not same) and larger and all(
        o.shape[0] % i.shape[0] == 0 and o.shape[1] % i.shape[1] == 0 for i, o in zip(I, O))
    return f


rows = []
for tid, t in CH.items():
    L = LED[tid]
    solved = any(float(L[k] or 0) > 0 for k in LV) or int(L["our_hits"] or 0) > 0 or tid in PIV_SOLVED
    levels = sum(float(L[k] or 0) > 0 for k in LV)
    rows.append(dict(task=tid, solved=solved, levels=levels, xhigh=float(L["arcprize_xhigh"] or 0) > 0,
                     our_hits=int(L["our_hits"] or 0), our_attempts=int(L["our_attempts"] or 0), **features(t)))

N = len(rows); base = sum(r["solved"] for r in rows) / N
print(f"tasks {N}; solved by anyone {sum(r['solved'] for r in rows)} ({100 * base:.0f} %); ARC Prize xhigh {sum(r['xhigh'] for r in rows)}")


def table(title, keyf, order=None):
    groups = {}
    for r in rows:
        k = keyf(r)
        if k is None: continue
        groups.setdefault(k, []).append(r)
    print(f"\n## {title}")
    print("| group | tasks | solved by anyone | % | ARC Prize xhigh % |")
    print("|---|---|---|---|---|")
    for k in (order or sorted(groups, key=lambda k: -np.mean([g["solved"] for g in groups[k]]))):
        g = groups.get(k, [])
        if not g: continue
        s = sum(x["solved"] for x in g); xh = sum(x["xhigh"] for x in g)
        print(f"| {k} | {len(g)} | {s} | {100 * s / len(g):.0f} | {100 * xh / len(g):.0f} |")


table("Output size versus input", lambda r: r["shape_rel"])
table("Output is an exact crop of the input", lambda r: "crop of the input" if r["crop_of_input"] else ("smaller, not a crop" if r["shape_rel"] == "output smaller" else None))
table("Same-size tasks: kind of edit", lambda r: r["edit_kind"])
table("Same-size tasks: share of cells that change",
      lambda r: None if r["changed_frac"] is None else ("< 5 %" if r["changed_frac"] < .05 else "5-15 %" if r["changed_frac"] < .15 else "15-30 %" if r["changed_frac"] < .30 else ">= 30 %"),
      ["< 5 %", "5-15 %", "15-30 %", ">= 30 %"])
table("Output introduces a colour absent from the input", lambda r: "new colour" if r["new_colours"] else "no new colour")
table("Training examples", lambda r: "2" if r["n_train"] <= 2 else "3" if r["n_train"] == 3 else "4+", ["2", "3", "4+"])
table("Test inputs", lambda r: str(min(r["n_test"], 3)) if r["n_test"] < 3 else "3+", ["1", "2", "3+"])
table("Input grid size (cells)", lambda r: "< 150" if r["in_cells"] < 150 else "150-400" if r["in_cells"] < 400 else "400-700" if r["in_cells"] < 700 else ">= 700",
      ["< 150", "150-400", "400-700", ">= 700"])
table("Objects in the input (4-connected, non-background)", lambda r: "1-3" if r["objects"] <= 3 else "4-10" if r["objects"] <= 10 else "11-30" if r["objects"] <= 30 else "> 30",
      ["1-3", "4-10", "11-30", "> 30"])

with open(HERE / "capability_profile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nwrote capability_profile.csv")
