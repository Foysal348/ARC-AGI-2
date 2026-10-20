"""Derive <= 3 puzzle-ordering rules for Qwen3.8-27B and validate them honestly (Claude, 2026-10-14).

Rules use the TRAINING pairs only. Score = number of rules satisfied (0-3), higher = order earlier.
Validation:
  1. nested cross-validation: the WHOLE greedy rule search runs inside each training fold and is scored on the
     held-out fold (repeated 10-fold) -> honest estimate of how a searched rule set ranks unseen puzzles;
  2. label independence: rules searched on ARC Prize labels only, then scored on OUR runs' labels (separate runs);
  3. stability: how often each rule is chosen across folds.
usage: python derive_ordering_rules.py   (needs capability_profile.csv from capability_profile.py)
"""
import csv, json
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FX = (HERE.parent.parent / "experiments" / "Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)" / "Experiment Notebooks"
      / "Build" / "Q27X_build" / "fixtures" / "fake_input" / "competitions" / "arc-prize-2026-arc-agi-2")
CH = json.load(open(FX / "arc-agi_evaluation_challenges.json"))
PROF = {r["task"]: r for r in csv.DictReader(open(HERE / "capability_profile.csv"))}
LED = {r["task"]: r for r in csv.DictReader(open(HERE / "ledger_all120.csv"))}
LV = ("arcprize_xhigh", "arcprize_medium", "arcprize_low", "arcprize_none")
TASKS = sorted(CH)

# ---------------------------------------------------------------- features (training pairs only)
def feats(tid):
    t = CH[tid]; p = PROF[tid]; tags = set(LED[tid]["tags"].split("|"))
    I = [np.array(x["input"]) for x in t["train"]]; O = [np.array(x["output"]) for x in t["train"]]
    T = [np.array(x["input"]) for x in t["test"]]
    f = {
        "n_train": int(p["n_train"]), "n_test": int(p["n_test"]),
        "same_size": p["shape_rel"] == "same size", "out_smaller": p["shape_rel"] == "output smaller",
        "changed_frac": float(p["changed_frac"]) if p["changed_frac"] not in ("", "None") else -1.0,
        "new_colour": p["new_colours"] == "True", "in_cells": float(p["in_cells"]), "objects": float(p["objects"]),
        "n_colours": len(set(np.concatenate([g.ravel() for g in I + O]).tolist())),
        "test_bigger": max(g.size for g in T) > max(g.size for g in I),
        "train_shapes_vary": len({g.shape for g in I}) > 1,
        "out_shape_const": len({g.shape for g in O}) == 1,
        "out_symmetric": all(np.array_equal(o, o[:, ::-1]) or np.array_equal(o, o[::-1]) for o in O),
    }
    for tg in ("dynamic", "high_ratio", "many_colors", "large", "sparse_bg", "dense_non_bg", "rare", "easy_static"):
        f["tag_" + tg] = tg in tags
    return f

F = {t: feats(t) for t in TASKS}

def label_any(t):
    L = LED[t]; return any(float(L[k] or 0) > 0 for k in LV) or int(L["our_hits"] or 0) > 0 or t in ("1818057f", "2ba387bc")
Y_ANY = np.array([label_any(t) for t in TASKS], float)
Y_ARC = np.array([any(float(LED[t][k] or 0) > 0 for k in LV) for t in TASKS], float)

# ---------------------------------------------------------------- candidate binary rules
CANDS = []
for k in F[TASKS[0]]:
    vals = np.array([F[t][k] for t in TASKS], float)
    if set(np.unique(vals)) <= {0.0, 1.0}:
        CANDS.append((f"{k}", vals == 1)); CANDS.append((f"not {k}", vals == 0))
    else:
        for q in np.unique(np.percentile(vals[vals >= 0] if k == "changed_frac" else vals, [20, 33, 50, 67, 80])):
            CANDS.append((f"{k} >= {q:.3g}", vals >= q)); CANDS.append((f"{k} < {q:.3g}", vals < q))
    if k == "changed_frac":   # interval rules on same-size tasks
        for lo, hi in ((.05, .15), (.05, .30), (.10, .30), (.15, .30), (.15, .40)):
            CANDS.append((f"changed_frac in [{lo},{hi})", (vals >= lo) & (vals < hi)))
CANDS.append(("changed 15-30% or out_smaller", np.array([(.15 <= F[t]["changed_frac"] < .30) or F[t]["out_smaller"] for t in TASKS])))
NAMES = [c[0] for c in CANDS]; M = np.array([c[1] for c in CANDS], float)

def auc(s, y):
    s = np.asarray(s, float); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg): return np.nan
    return float(np.mean((pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :])))

def search(idx, y, k_max=3, min_gain=0.005):
    chosen = []; best = 0.5
    for _ in range(k_max):
        base = M[chosen].sum(0) if chosen else np.zeros(M.shape[1])
        scores = [(auc((base + M[j])[idx], y[idx]), j) for j in range(len(M)) if j not in chosen]
        a, j = max(scores)
        if a - best < min_gain: break
        chosen.append(j); best = a
    return chosen, best

def nested_cv(y, reps=20, folds=10, seed=0):
    rng = np.random.default_rng(seed); out = []; picks = Counter()
    for _ in range(reps):
        perm = rng.permutation(len(y)); held = np.zeros(len(y))
        for f in range(folds):
            te = perm[f::folds]; tr = np.setdiff1d(perm, te)
            ch, _ = search(tr, y); held[te] = M[ch].sum(0)[te] if ch else 0
            picks.update(NAMES[j] for j in ch)
        out.append(auc(held, y))
    return np.array(out), picks

# ---------------------------------------------------------------- 1. full-data search (the rules we would ship)
ALL = np.arange(len(TASKS))
ch, a_in = search(ALL, Y_ANY)
print("RULES (searched on all 120, label = solved by anyone):")
for j in ch: print("  +1 if", NAMES[j], f"(true for {int(M[j].sum())} tasks)")
S = M[ch].sum(0)
print(f"in-sample AUC {a_in:.3f}")
for k in range(len(ch) + 1):
    m = S == k
    if m.any(): print(f"  score {k}: {int(m.sum()):3d} tasks, solved {int(Y_ANY[m].sum()):3d} ({100 * Y_ANY[m].mean():.0f} %)")

# ---------------------------------------------------------------- 2. honest estimate: nested CV of the search procedure
cv, picks = nested_cv(Y_ANY)
print(f"\nNESTED 10-fold CV (search redone inside every fold, 20 repeats): AUC mean {cv.mean():.3f}, 5-95 % {np.percentile(cv, 5):.3f}-{np.percentile(cv, 95):.3f}")
print("rules chosen most often across the 200 fold searches:", picks.most_common(6))

# hand-picked score from the earlier analysis, for comparison (in-sample only: it was chosen by eye on all data)
HAND = (np.array([F[t]["n_train"] >= 3 for t in TASKS], float) + np.array([F[t]["n_test"] == 1 for t in TASKS], float)
        + np.array([(.15 <= F[t]["changed_frac"] < .30) or F[t]["out_smaller"] for t in TASKS], float))
print(f"hand-picked 3-rule score: in-sample AUC {auc(HAND, Y_ANY):.3f}")
print(f"NVARC fine (lower = easier): AUC {auc([-float(LED[t]['fine']) for t in TASKS], Y_ANY):.3f}; tier: {auc([-int(LED[t]['tier']) for t in TASKS], Y_ANY):.3f}")

# ---------------------------------------------------------------- 3. label independence: ARC Prize labels -> our runs
ours = np.array([int(LED[t]["our_attempts"] or 0) > 0 for t in TASKS]); y_ours = np.array([int(LED[t]["our_hits"] or 0) > 0 for t in TASKS], float)
ch_arc, a_arc = search(ALL, Y_ARC)
s_arc = M[ch_arc].sum(0)
print(f"\nrules searched on ARC Prize labels only: {[NAMES[j] for j in ch_arc]} (in-sample AUC {a_arc:.3f})")
idx = np.where(ours)[0]
print(f"  scored on OUR runs' labels ({len(idx)} tasks we ran, {int(y_ours[idx].sum())} solved by us): AUC {auc(s_arc[idx], y_ours[idx]):.3f}"
      f" | shipped rules on the same: {auc(S[idx], y_ours[idx]):.3f} | hand-picked: {auc(HAND[idx], y_ours[idx]):.3f}")
