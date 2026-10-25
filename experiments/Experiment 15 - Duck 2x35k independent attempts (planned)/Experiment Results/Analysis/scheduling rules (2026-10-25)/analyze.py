"""Validate the two E15B scheduling rules on our multi-attempt Duck runs (2026-10-25).
A. stage-2 (attempt 2) class order: value of a FRESH attempt given attempt 1's class.
B. stage-1 order (ORDER2 + cells): does it put solvable / cheap tasks first? Alternatives, leave-one-run-out."""
import itertools, collections, random, statistics as st
import numpy as np
from load_attempts import load, SOL, CH

rows = load()
R = collections.defaultdict(dict)
for r in rows:
    R[(r["run"], r["task"])][r["b"]] = r
credit = lambda q: sum(q) / len(q)
union = lambda *qs: [any(x) for x in zip(*qs)]
CLS = {0: "0 partial progress, no accept", 1: "1 nothing", 2: "2 one accepted answer", 3: "3 two+ distinct accepted"}


def gain(a, b):
    return credit(union(a["acc_q"], b["acc_q"])) - credit(a["acc_q"])


def table(pairs, title):
    print("\n### " + title)
    print("| class of attempt 1 | pairs | attempt 1 credit | fresh attempt credit | **gain** | fresh tokens | gain / 100k tok |")
    print("|---|---|---|---|---|---|---|")
    tot = collections.defaultdict(list)
    for a, b in pairs:
        tot[a["cls"]].append((a, b))
    for c in sorted(tot):
        ps = tot[c]
        g = st.mean(gain(a, b) for a, b in ps)
        tk = st.mean(b["tokens"] for a, b in ps)
        print(f"| {CLS[c]} | {len(ps)} | {st.mean(credit(a['acc_q']) for a, b in ps):.3f} | "
              f"{st.mean(credit(b['acc_q']) for a, b in ps):.3f} | **{g:.3f}** | {tk:,.0f} | {100000 * g / tk:.3f} |")
    for name, f in (("2a flagged", lambda a: a["cls"] == 2 and a["flagged"]),
                    ("2b not flagged", lambda a: a["cls"] == 2 and not a["flagged"]),
                    ("2/3 accepted answer wrong", lambda a: a["cls"] >= 2 and credit(a["acc_q"]) == 0),
                    ("2/3 accepted answer right", lambda a: a["cls"] >= 2 and credit(a["acc_q"]) == 1)):
        ps = [(a, b) for a, b in pairs if f(a)]
        if ps:
            g = st.mean(gain(a, b) for a, b in ps)
            print(f"| &nbsp;&nbsp;{name} | {len(ps)} | {st.mean(credit(a['acc_q']) for a, b in ps):.3f} | "
                  f"{st.mean(credit(b['acc_q']) for a, b in ps):.3f} | {g:.3f} | {st.mean(b['tokens'] for a, b in ps):,.0f} | |")


# ---------------------------------------------------------------- A. stage-2 class value
xr = [(R[("G4B_XHIGH_E16", t)][0], R[("G4B_XHIGH_noE16", t)][0]) for t in SOL
      if 0 in R.get(("G4B_XHIGH_E16", t), {}) and 0 in R.get(("G4B_XHIGH_noE16", t), {})]
xr += [(b, a) for a, b in xr]
table(xr, "A1. Cross-run: attempt 1 = b0 of one XHIGH EVAL120 run, fresh attempt = b0 of the other (120 tasks x 2 directions)")
wr = [(R[k][0], R[k][1]) for k in R if k[0].startswith("E15B") and 0 in R[k] and 1 in R[k]]
table(wr, "A2. Within run, E15B (b0 -> fresh b1, facts view; 45K30K + 40K35K, 20 tasks x 2)")
gr = [(R[k][0], R[k][1]) for k in R if k[0] in ("G4B_XHIGH_E16", "G4B_XHIGH_noE16") and 0 in R[k] and 1 in R[k]]
table(gr, "A3. Within run, G4B XHIGH (b0 -> stage-2 b1, indexed view; b1 ran on 161 of 240 task-runs)")
table(xr + wr, "A4. A1 + A2 pooled")

print("\n### A5. k-th independent xhigh attempt on tasks where all earlier attempts had no correct accepted answer")
print("Pool per task: G4B XHIGH E16 b0, G4B XHIGH noE16 b0, E15B 45K30K b0+b1, E15B 40K35K b0+b1; 300 random orders per task.")
pool = collections.defaultdict(list)
for (run, t), d in R.items():
    if run in ("G4B_XHIGH_E16", "G4B_XHIGH_noE16") and 0 in d:
        pool[t].append(d[0])
    elif run.startswith("E15B"):
        pool[t].extend(d.values())
rng = random.Random(0)
acc = collections.defaultdict(lambda: [0, 0.0, 0.0])
for t, att in pool.items():
    for _ in range(300):
        a = att[:]
        rng.shuffle(a)
        got = [False] * len(SOL[t])
        for k, x in enumerate(a, 1):
            if not any(got):
                nb = union(got, x["acc_q"])
                acc[k][0] += 1; acc[k][1] += credit(nb) - credit(got); acc[k][2] += x["tokens"]
            got = union(got, x["acc_q"])
print("| k | cases still at 0 | P(k-th adds credit) | mean tokens | gain / 100k tok | tasks with >= k attempts |")
print("|---|---|---|---|---|---|")
for k in sorted(acc):
    n, g, tk = acc[k]
    print(f"| {k} | {n} | {g / n:.3f} | {tk / n:,.0f} | {100000 * g / tk:.3f} | {sum(1 for t in pool if len(pool[t]) >= k)} |")


# ---------------------------------------------------------------- B. stage-1 order
def feats(t):
    c = CH[t]
    cells = lambda g: len(g) * len(g[0]) if g and g[0] else 0
    tr = c["train"]
    ins = [cells(p["input"]) for p in tr]
    outs = [cells(p["output"]) for p in tr]
    same = all(len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0]) for p in tr)
    return {"order2": int(len(tr) >= 3) + int(len(c["test"]) == 1), "n_train": len(tr), "n_test": len(c["test"]),
            "total_cells": sum(ins) + sum(outs) + sum(cells(p["input"]) for p in c["test"]),
            "mean_in": st.mean(ins), "out_smaller": int(all(o < i for i, o in zip(ins, outs))), "same_size": int(same),
            "n_colours": len({v for p in tr for g in (p["input"], p["output"]) for row in g for v in row})}


F = {t: feats(t) for t in SOL}


def auc(score, lab):
    pos = [s for s, l in zip(score, lab) if l]
    neg = [s for s, l in zip(score, lab) if not l]
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


RUNS3 = ("G4B_XHIGH_E16", "G4B_XHIGH_noE16", "G4B_LOW_E16")
T = sorted(SOL)
lab = {r: {t: credit(R[(r, t)][0]["acc_q"]) for t in T if 0 in R.get((r, t), {})} for r in RUNS3}
tok = {r: {t: R[(r, t)][0]["tokens"] for t in T if 0 in R.get((r, t), {})} for r in RUNS3}
o2rank = {t: i for i, t in enumerate(sorted(T, key=lambda t: (-F[t]["order2"], F[t]["total_cells"], t)))}
cand = {"ORDER2 + cells (current)": lambda t: -o2rank[t], "ORDER2 only": lambda t: F[t]["order2"],
        "fewer total cells": lambda t: -F[t]["total_cells"], "more train pairs": lambda t: F[t]["n_train"],
        "1 test input": lambda t: int(F[t]["n_test"] == 1), "output smaller": lambda t: F[t]["out_smaller"],
        "same size": lambda t: F[t]["same_size"], "fewer colours": lambda t: -F[t]["n_colours"],
        "smaller mean input": lambda t: -F[t]["mean_in"]}
print("\n### B1. AUC of each order key for 'attempt 1 got credit > 0' (0.5 = random)")
print("| key | " + " | ".join(RUNS3) + " | mean |")
print("|---" * (len(RUNS3) + 2) + "|")
for name, f in cand.items():
    a = [auc([f(t) for t in lab[r]], [lab[r][t] > 0 for t in lab[r]]) for r in RUNS3]
    print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in a) + f" | {np.mean(a):.3f} |")
print("| (positives per run) | " + " | ".join(str(sum(v > 0 for v in lab[r].values())) for r in RUNS3) + " | |")
print("| (AUC of tokens used: fewer tokens -> solved?) | " + " | ".join(
    f"{auc([-tok[r][t] for t in lab[r]], [lab[r][t] > 0 for t in lab[r]]):.3f}" for r in RUNS3) + " | |")


def curve(order, r, frac):
    budget = frac * sum(tok[r].values())
    used = got = 0.0
    for t in order:
        if t not in tok[r]:
            continue
        if used + tok[r][t] > budget:
            break
        used += tok[r][t]; got += lab[r][t]
    return got


print("\n### B2. Attempt-1 task credit collected when only 30 % / 60 % of the run's attempt-1 tokens fit, in the given order")
hdr = [f"{r} @{int(f * 100)}%" for r in RUNS3 for f in (0.3, 0.6)]
print("| key | " + " | ".join(hdr) + " |")
print("|---" * (len(hdr) + 1) + "|")
rng = random.Random(1)
for name, f in list(cand.items()) + [("random (mean of 500)", None)]:
    vals = []
    for r in RUNS3:
        for fr in (0.3, 0.6):
            if f is None:
                vals.append(st.mean(curve(rng.sample(T, len(T)), r, fr) for _ in range(500)))
            else:
                vals.append(curve(sorted(T, key=lambda t: (-f(t), t)), r, fr))
    print(f"| {name} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
print("| (total credit at 100 %) | " + " | ".join(f"{sum(lab[r].values()):.1f}" for r in RUNS3 for _ in (0, 1)) + " |")

print("\n### B3. Two-key orders chosen on 2 runs (mean AUC), tested on the held-out run")
keys = [k for k in cand if k != "ORDER2 + cells (current)"]
for held in RUNS3:
    trn = [r for r in RUNS3 if r != held]
    best = None
    for a, b in itertools.permutations(keys, 2):
        f = lambda t, a=a, b=b: (cand[a](t), cand[b](t))
        s = np.mean([auc([f(t) for t in lab[r]], [lab[r][t] > 0 for t in lab[r]]) for r in trn])
        if best is None or s > best[0]:
            best = (s, a, b)
    f = lambda t, a=best[1], b=best[2]: (cand[a](t), cand[b](t))
    test = auc([f(t) for t in lab[held]], [lab[held][t] > 0 for t in lab[held]])
    cur = auc([cand["ORDER2 + cells (current)"](t) for t in lab[held]], [lab[held][t] > 0 for t in lab[held]])
    print(f"| held-out {held} | {best[1]}, then {best[2]} (train {best[0]:.3f}) | test {test:.3f} | current {cur:.3f} |")


# ---------------------------------------------------------------- B4. external labels (not Duck): ARC Prize and E10 direct
import csv
from pathlib import Path
LED = {r["task"]: r for r in csv.DictReader(open(Path(r"D:\ARC AGI 2\resources\Puzzle Capability Ledger\ledger_all120.csv")))}
UNITS = list(csv.DictReader(open(Path(r"D:\ARC AGI 2\resources\Puzzle Capability Ledger\ledger_units.csv"))))
e10 = collections.defaultdict(list)
for u in UNITS:
    e10[u["task"]].append("E10 ORDER2 EVAL120 direct" in (u.get("solved_by") or ""))
ext = {"ARC Prize xhigh": {t: LED[t]["arcprize_xhigh"] == "1" for t in T},
       "ARC Prize low": {t: LED[t]["arcprize_low"] == "1" for t in T},
       "E10 direct 86k": {t: any(e10[t]) for t in T}}
cand["fewer colours, then ORDER2 (proposed)"] = lambda t: (-F[t]["n_colours"], F[t]["order2"])
cand["ORDER2, then fewer colours"] = lambda t: (F[t]["order2"], -F[t]["n_colours"])
print("\n### B4. The same keys on labels from other methods (not used to pick anything)")
print("| key | " + " | ".join(ext) + " |")
print("|---" * (len(ext) + 1) + "|")
for name, f in cand.items():
    print(f"| {name} | " + " | ".join(f"{auc([f(t) for t in T], [ext[e][t] for t in T]):.3f}" for e in ext) + " |")
print("| (positives) | " + " | ".join(str(sum(ext[e].values())) for e in ext) + " |")
print("\n### B5. B2 for the combined keys")
for name in ("ORDER2 + cells (current)", "ORDER2 only", "fewer colours, then ORDER2 (proposed)", "ORDER2, then fewer colours"):
    f = cand[name]
    vals = [curve(sorted(T, key=lambda t: (-np.array(f(t)) if isinstance(f(t), tuple) else -f(t)).tolist() + [t] if isinstance(f(t), tuple) else (-f(t), t)), r, fr)
            for r in RUNS3 for fr in (0.3, 0.6)]
    print(f"| {name} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

# ---------------------------------------------------------------- A6. simulate stage-2 orders on the pooled pairs
print("\n### A6. Stage-2 queue simulation on the A4 pairs: credit gained when only part of the attempt-2 tokens fit")
pairs = xr + wr
rule = {"current (0, 1, 2 flagged first, 3)": lambda a: (a["cls"], 0 if a["flagged"] else 1),
        "no-accept first (0 = 1), then 3, then 2": lambda a: ({0: 0, 1: 0, 3: 1, 2: 2}[a["cls"]],),
        "1, 0, 3, 2": lambda a: ({1: 0, 0: 1, 3: 2, 2: 3}[a["cls"]],),
        "no-accept first, then accepted (2 = 3)": lambda a: (0 if a["cls"] < 2 else 1,)}
tot_tok = sum(b["tokens"] for a, b in pairs)
print("| order | @25 % | @50 % | @75 % | @100 % |")
print("|---|---|---|---|---|")
rng = random.Random(2)
for name, key in rule.items():
    out = []
    for fr in (0.25, 0.5, 0.75, 1.0):
        sims = []
        for _ in range(300):        # random order inside a class (ties) = the ORDER2 index is unrelated to the class
            ps = pairs[:]
            rng.shuffle(ps)
            ps.sort(key=lambda p: key(p[0]))
            used = g = 0.0
            for a, b in ps:
                if used + b["tokens"] > fr * tot_tok:
                    break
                used += b["tokens"]; g += gain(a, b)
            sims.append(g)
        out.append(st.mean(sims))
    print(f"| {name} | " + " | ".join(f"{v:.1f}" for v in out) + " |")
print(f"(pairs {len(pairs)}; credit units = tasks; 100 % = every pair's fresh attempt)")
