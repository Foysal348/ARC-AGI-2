"""Score every NVARC run in this paper folder against the public evaluation solutions.

usage: python -B scripts/score_all.py            (run from the paper folder)
writes data/run_scores.csv, data/union_units.json and data/pool_units.json, prints the unions.

Sources scanned:
  - every submission*.json under "NVARC family (34 notebooks, downloaded 2026-10-19)/", "kaggle versions/" and
    "kaggle versions arc-2026d-with-codex/";
  - the two user-provided notebooks in "provided notebooks/" (no submission.json saved: the solved-unit set of
    run_dual_selection is read from cell 8's printed output, the third printed set);
  - candidate pools: "GT Present cases" table of EXPERIMENT 1 in any *.log run log (Kaggle JSON event log or text).
Scoring: a test input (unit) is right if attempt_1 or attempt_2 equals the solution (pass@2);
task-norm = sum over right units of 1 / (number of test inputs of that task), out of 120.
Duplicate submissions (same sha256) are counted once.
"""
import csv, glob, hashlib, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = json.load(open(os.path.join(ROOT, "data", "arc-agi_evaluation_solutions.json")))


def tn(units):
    return sum(1 / len(SOL[t]) for t, _ in units)


def score_submission(path):
    sub = json.load(open(path))
    return {(t, i) for t, outs in SOL.items() if t in sub for i, g in enumerate(outs)
            if i < len(sub[t]) and g in (sub[t][i].get("attempt_1"), sub[t][i].get("attempt_2"))}


def notebook_dual_set(path):
    nb = json.load(open(path, encoding="utf-8"))
    out = "".join("".join(x.get("text", "")) for x in nb["cells"][8].get("outputs", []))
    sets = [set(re.findall(r"'([0-9a-f]{8})_(\d+)'", m.group(1))) for m in re.finditer(r"\{('[0-9a-f]{8}_\d+'[^}]*)\}", out)]
    return {(t, int(i)) for t, i in sets[2]}


def log_text(path):
    raw = open(path, encoding="utf-8").read()
    try:
        return "".join(e.get("data", "") for e in json.loads(raw) if e.get("stream_name") == "stdout")
    except ValueError:
        return raw


def pool_from_log(path):
    t = log_text(path)
    s, e = t.find("EXPERIMENT 1:"), t.find("EXPERIMENT 2:")
    if s < 0:
        return None
    seg = t[s:e]
    seg = seg[seg.find("GT Present cases"):]
    return {(m.group(1), int(m.group(2))) for m in re.finditer(r"^\s*([0-9a-f]{8})_(\d+)\s+\d", seg, re.M)}


rows, seen, union, pools = [], {}, set(), {}
pats = ["NVARC family*/**/*submission*.json", "kaggle versions/**/*submission*.json", "kaggle versions arc-2026d-with-codex/**/*submission*.json"]
for pat in pats:
    for f in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
        sha = hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
        rel = os.path.relpath(f, ROOT).replace("\\", "/")
        if sha in seen:
            rows.append([rel, sha, "", "", f"duplicate of {seen[sha]}"]); continue
        seen[sha] = rel
        u = score_submission(f); union |= u
        rows.append([rel, sha, len(u), round(tn(u), 2), ""])
for f in sorted(glob.glob(os.path.join(ROOT, "provided notebooks", "*.ipynb"))):
    u = notebook_dual_set(f); union |= u
    rows.append([os.path.relpath(f, ROOT).replace("\\", "/"), "", len(u), round(tn(u), 2), "from printed run_dual_selection set"])
pool_seen = set()
for f in sorted(glob.glob(os.path.join(ROOT, "**", "*.log"), recursive=True)):
    p = pool_from_log(f)
    key = tuple(sorted(p)) if p else None
    if p and key not in pool_seen:
        pool_seen.add(key)
        pools[os.path.relpath(f, ROOT).replace("\\", "/")] = sorted(p)

with open(os.path.join(ROOT, "data", "run_scores.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["source", "sha256_8", "units_right_of_172", "task_norm_of_120", "note"]); w.writerows(rows)
json.dump(sorted(union), open(os.path.join(ROOT, "data", "union_units.json"), "w"))
json.dump(pools, open(os.path.join(ROOT, "data", "pool_units.json"), "w"), indent=1)

pool_union = set(tuple(x) for p in pools.values() for x in p)
full = lambda S: sum(all((t, i) in S for i in range(len(SOL[t]))) for t in SOL)
print(f"runs scored (unique): {sum(1 for r in rows if not r[4].startswith('duplicate'))}")
print(f"SUBMISSION UNION: {len(union)}/172 units, task-norm {tn(union):.2f} = {100*tn(union)/120:.2f} %, tasks fully right {full(union)}")
for k, p in pools.items():
    print(f"pool {k}: {len(p)} units, task-norm {tn(set(map(tuple, p))):.2f}")
allu = union | pool_union
print(f"POOL UNION: {len(pool_union)} units, task-norm {tn(pool_union):.2f}")
print(f"ALL EVIDENCE UNION: {len(allu)} units, task-norm {tn(allu):.2f} = {100*tn(allu)/120:.2f} %")
