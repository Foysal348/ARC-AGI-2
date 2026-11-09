"""Part B helper: per train-pass-but-wrong attempt, grid facts and prediction-vs-gold diff.
Reads attempts.json (from extract_attempts.py), branch_result.json of each attempt and the gold file.
Writes false_verif.json and prints a table.
"""
import json, os
from collections import Counter

ROOT = r"D:\ARC AGI 2"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, r"experiments\Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)\Experiment Review\E10_ONE_THINKING_TWO_ATTEMPTS_FOR_ASTRA\data")
chal = json.load(open(os.path.join(DATA, "arc-agi_evaluation_challenges.json")))
sol = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))
att = json.load(open(os.path.join(HERE, "attempts.json")))


def shp(g):
    return (len(g), len(g[0])) if g and isinstance(g, list) and isinstance(g[0], list) else None


def cols(g):
    return sorted({v for r in g for v in r})


def diff(p, g):
    if shp(p) != shp(g):
        return {"shape_ok": False, "pred": shp(p), "gold": shp(g)}
    n = sum(1 for i in range(len(g)) for j in range(len(g[0])) if p[i][j] != g[i][j])
    ch = Counter((g[i][j], p[i][j]) for i in range(len(g)) for j in range(len(g[0])) if p[i][j] != g[i][j])
    return {"shape_ok": True, "cells_wrong": n, "cells": len(g) * len(g[0]),
            "top_gold_to_pred": [[a, b, c] for (a, b), c in ch.most_common(4)]}


rows = []
for x in att:
    if x["outcome"] != "ACCEPTED":
        continue
    ok = x["test_ok_vs_gold"]
    if ok and all(ok):
        continue
    c = chal[x["task"]]
    br = json.load(open(os.path.join(ROOT, x["branch_dir"], "branch_result.json")))
    sub = [s for s in br["submissions"] if s.get("accepted")][-1]
    tr_cols = sorted({v for p in c["train"] for g in (p["input"], p["output"]) for r in g for v in r})
    te_cols = sorted({v for p in c["test"] for r in p["input"] for v in r})
    d = [diff(sub["test_outputs"][i] if i < len(sub["test_outputs"]) else None, g) for i, g in enumerate(sol[x["task"]])]
    rows.append({
        "build": x["build"], "task": x["task"], "attempt": x["attempt"], "group": x["group"], "tokens": x["tokens"],
        "sub_name": x["sub_name"], "src_chars": sub.get("source_chars"), "overfit_flags": sub.get("overfit_flags"),
        "n_train": len(c["train"]), "n_test": len(c["test"]),
        "train_in": [list(shp(p["input"])) for p in c["train"]], "train_out": [list(shp(p["output"])) for p in c["train"]],
        "test_in": [list(shp(p["input"])) for p in c["test"]], "gold": [list(shp(g)) for g in sol[x["task"]]],
        "test_ok": ok, "diff": d,
        "new_test_colours": [v for v in te_cols if v not in tr_cols],
        "test_in_larger_than_all_train": [shp(p["input"])[0] * shp(p["input"])[1] > max(shp(q["input"])[0] * shp(q["input"])[1] for q in c["train"]) for p in c["test"]],
        "branch_dir": x["branch_dir"],
    })

json.dump(rows, open(os.path.join(HERE, "false_verif.json"), "w"), indent=1)
for r in rows:
    print(r["build"], r["task"], "a%d" % r["attempt"], r["group"], "tok", r["tokens"], "train", r["n_train"], r["train_in"], "->", r["train_out"],
          "| test", r["test_in"], "gold", r["gold"], "| ok", r["test_ok"], "| diff", r["diff"], "| newcol", r["new_test_colours"],
          "| bigger", r["test_in_larger_than_all_train"], "| flags", r["overfit_flags"], r["sub_name"], r["src_chars"])
