"""EVAL20 selection with the EDA notebook's own method (ARC2_Advanced_EDA_v8.ipynb, cells 35 + 49):
puzzle_features() and the weighted 'mirror gap' vs TEST_TARGET_PCT are copied from the EDA; lower gap = closer to test.
Two candidates: (A) EVAL12 + 8 (keeps the recorded DIRECT run-7 comparison), (B) best fresh 20.
usage: python select_eval20.py <arc-agi_evaluation_challenges.json>"""
import itertools, json, random, sys
import numpy as np

def cells(g): return int(np.prod(np.shape(g)))

def puzzle_features(key, data):                       # EDA cell 35 (tags subset used by the mirror report)
    p = data[key]; train = p["train"]
    ins = [cells(x["input"]) for x in train]; outs = [cells(x["output"]) for x in train]
    ratios = [max(i, o) / max(min(i, o), 1) for i, o in zip(ins, outs)]
    colors = set()
    for ex in train:
        for name in ["input", "output"]:
            colors.update(np.unique(np.array(ex[name])).tolist())
    avg_in = float(np.mean(ins)); avg_ratio = float(np.mean(ratios)); max_ratio = float(max(ratios))
    n_colors = len(colors); n_train = len(train); n_q = len(p["test"])
    dynamic = any(np.shape(x["input"]) != np.shape(x["output"]) for x in train)
    rare = bool(6 in colors or 9 in colors)
    tier = 3 if dynamic and max_ratio > 30 else 2 if dynamic and max_ratio > 5 else 2 if avg_in > 400 else 1
    fine = avg_in / 900 * 0.30 + (avg_ratio - 1) / 10 * 0.25 + n_colors / 10 * 0.20 + (0.25 if dynamic else 0.0)
    if n_train >= 6 and tier < 3: fine *= 0.7
    tags = [f"tier{tier}"]
    if tier == 1 and not dynamic and max_ratio == 1 and n_colors <= 5: tags.append("easy_static")
    if tier == 1 and fine > 0.45: tags.append("hard_tier1")
    return dict(key=key, tier=tier, fine=fine, max_ratio=max_ratio, dynamic=dynamic, rare=rare,
                n_queries=n_q, tags=tags)

TEST_TARGET_PCT = {"tier1": 74.2, "tier2": 21.7, "tier3": 4.2, "easy_static": 62.1, "rare": 54.6,
                   "multi_query": 7.1, "dynamic": 29.6, "hard_tier1": 2.9, "high_ratio": 10.0, "extreme_ratio": 4.2}
WEIGHTS = {"tier1": 2.0, "tier2": 1.5, "tier3": 1.2, "easy_static": 2.0, "multi_query": 1.7, "dynamic": 1.2,
           "rare": 1.0, "hard_tier1": 1.0, "high_ratio": 1.0, "extreme_ratio": 1.0}
CATS = list(TEST_TARGET_PCT)

def flags(x):
    return {"tier1": x["tier"] == 1, "tier2": x["tier"] == 2, "tier3": x["tier"] == 3,
            "easy_static": "easy_static" in x["tags"], "rare": x["rare"], "multi_query": x["n_queries"] > 1,
            "dynamic": x["dynamic"], "hard_tier1": "hard_tier1" in x["tags"], "high_ratio": x["max_ratio"] > 5,
            "extreme_ratio": x["max_ratio"] > 30}

def gap(keys, F):
    n = len(keys)
    pct = {c: 100.0 * sum(flags(F[k])[c] for k in keys) / n for c in CATS}
    return sum(WEIGHTS[c] * abs(pct[c] - TEST_TARGET_PCT[c]) for c in CATS) / sum(WEIGHTS.values()), pct

def best(F, fixed, n, pool, iters=4000, seed=0):
    rng = random.Random(seed); cur = list(fixed)
    while len(cur) < n:                                  # greedy fill
        cur.append(min((k for k in pool if k not in cur), key=lambda k: gap(cur + [k], F)[0]))
    free = [i for i in range(len(cur)) if cur[i] not in fixed]
    g0 = gap(cur, F)[0]
    for _ in range(iters):                               # swap search
        i = rng.choice(free); k = rng.choice([k for k in pool if k not in cur])
        cand = cur[:i] + [k] + cur[i + 1:]
        g = gap(cand, F)[0]
        if g < g0: cur, g0 = cand, g
    return sorted(cur), g0

if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    F = {k: puzzle_features(k, data) for k in data}
    EVAL12 = ["136b0064", "20a9e565", "21897d95", "247ef758", "2b83f449", "4c3d4a41", "4c7dc4dd", "6e4f6532",
              "88bcf3b4", "898e7135", "8b7bacbf", "b6f77b65"]
    pool = sorted(data)
    g12, p12 = gap(EVAL12, F)
    A, gA = best(F, EVAL12, 20, pool)
    B, gB = best(F, [], 20, pool)
    gall, pall = gap(pool, F)
    def show(name, keys, g):
        _, p = gap(keys, F)
        print(f"{name:<26} n={len(keys):3} gap={g:5.2f}pp  " + " ".join(f"{c}={p[c]:.0f}" for c in CATS))
    print("targets (EDA TEST_TARGET_PCT):", TEST_TARGET_PCT)
    show("EVAL12", EVAL12, g12); show("all 120 eval", pool, gall)
    show("A: EVAL12 + 8", A, gA); show("B: fresh 20", B, gB)
    print("A added:", [k for k in A if k not in EVAL12])
    print("A =", json.dumps(A)); print("B =", json.dumps(B))
    print("A test outputs:", sum(len(data[k]["test"]) for k in A), "| B test outputs:", sum(len(data[k]["test"]) for k in B))
    for k in A:
        x = F[k]; print(f"  {k} tier{x['tier']} fine={x['fine']:.3f} dyn={x['dynamic']} rare={x['rare']} q={x['n_queries']} "
                        f"{'easy_static' if 'easy_static' in x['tags'] else ''} {'EVAL12' if k in EVAL12 else 'NEW'}")
