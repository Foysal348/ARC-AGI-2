"""Offline check of facts v2 relations on the 120 evaluation tasks (no GPU):
1. coverage: how many tasks get each relation (holding on every train pair);
2. precision: does the relation also hold on the TEST pairs (eval solutions) -> a trustworthy hint, or misleading;
3. Duck link: for tasks with relations, what the R3K Duck attempts did (solved / tokens to the first correct accept);
4. prompt size of the relations block (Qwen tokenizer when available, else chars / 3.2).
Writes eval_relations_out.md next to this file."""
import csv, json, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arc2_relations as R  # noqa: E402

DATA = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
CH = json.load(open(DATA / "arc-agi_evaluation_challenges.json", encoding="utf-8"))
SOL = json.load(open(DATA / "arc-agi_evaluation_solutions.json", encoding="utf-8"))
PT = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Results/Analysis/scheduling rules (2026-10-25)/r3k_per_task.csv")
DUCK = {r["task"]: r for r in csv.DictReader(open(PT, encoding="utf-8"))}
try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures/tok_int4")
    ntok = lambda s: len(TOK.encode(s, add_special_tokens=False))
except Exception:
    ntok = lambda s: int(len(s) / 3.2)


def holds_on_test(rel, a, b):
    k, p = rel["kind"], rel["params"]
    if k == "dihedral":
        return R.DIHEDRAL[p["op"]](a) == b
    if k == "upscale_blocks":
        v = R.d_upscale_blocks(a, b)
        return bool(v) and v["fr"] == p["fr"] and v["fc"] == p["fc"]
    if k == "tiling":
        v = R.d_tiling(a, b)
        return bool(v) and v["tiles"] == p["tiles"]
    if k == "scale":
        v = R.d_scale(a, b)
        return bool(v) and (v["dir"], v["fr"], v["fc"]) == (p["dir"], p["fr"], p["fc"])
    if k == "crop":
        v = R.d_crop(a, b)
        if not v:
            return False
        if "component_rank" in p:
            return v.get("component_rank") == p["component_rank"] and v.get("conn") == p["conn"]
        if p.get("all_nonmodal_bbox"):
            return bool(v.get("all_nonmodal_bbox"))
        return True
    if k == "colormap":
        return R.shape(a) == R.shape(b) and [[p["map"].get(x, x) for x in row] for row in a] == b
    if k in ("paint_only", "erase_only", "change_scope"):
        v = R.d_change_scope(a, b)
        if not v:
            return False
        if k == "paint_only":
            return v["only_modal_painted"]
        if k == "erase_only":
            return v["only_to_modal_erased"]
        return set(v["from"]) <= set(p["from"]) and set(v["to"]) <= set(p["to"])
    if k.startswith("objects_"):
        v = R.d_objects(a, b)
        if not v:
            return False
        if k == "objects_move":
            return v["moved"] > 0 and not v["added"] and not v["removed"]
        if k == "objects_recolor":
            return v["recolored"] > 0 and not v["moved"] and not v["added"] and not v["removed"]
        return v["removed"] > 0 and not v["added"] and not v["moved"] and not v["recolored"]
    if k == "out_symmetry":
        s = R.d_out_symmetry(a, b)
        return bool(s) and set(p["sym"]) <= set(s["symmetries"])
    if k == "out_shape_const":
        return R.shape(b) == tuple(p["shape"])
    return None


def direct_answer(rel, a):
    """relations that determine the output outright"""
    k, p = rel["kind"], rel["params"]
    if k == "dihedral":
        return R.DIHEDRAL[p["op"]](a)
    if k == "upscale_blocks":
        return [[a[r // p["fr"]][c // p["fc"]] for c in range(len(a[0]) * p["fc"])] for r in range(len(a) * p["fr"])]
    if k == "tiling":
        rows = []
        for trow in p["tiles"]:
            tiles = [R.DIHEDRAL[op](a) for op in trow]
            for i in range(len(a)):
                rows.append([v for t in tiles for v in t[i]])
        return rows
    if k == "colormap":
        return [[p["map"].get(x, x) for x in row] for row in a]
    if k == "crop" and "component_rank" in p:
        comps = R.components(a, R.modal(a), p["conn"])
        if len(comps) > p["component_rank"]:
            r0, c0, r1, c1 = comps[p["component_rank"]]["bbox"]
            return R.sub(a, r0, c0, r1 - r0 + 1, c1 - c0 + 1)
    return None


rows, kinds = [], {}
for tid in sorted(CH):
    t = CH[tid]
    rels = R.relations(t)
    tests = [(q["input"], SOL[tid][i]) for i, q in enumerate(t["test"])]
    d = DUCK.get(tid, {})
    solved = [d.get(f"R3K{n}_hit", "0/1").split("/")[0] != "0" for n in (1, 2)]
    acc_tok = [int(d[f"R3K{n}_a1_correct_at_tokens"]) for n in (1, 2) if d.get(f"R3K{n}_a1_correct_at_tokens")]
    acc_tok += [int(d[f"R3K{n}_a2_correct_at_tokens"]) for n in (1, 2) if d.get(f"R3K{n}_a2_correct_at_tokens")]
    direct = None
    for rel in rels:
        ok = [holds_on_test(rel, a, b) for a, b in tests]
        rel["test_ok"] = all(ok)
        kinds.setdefault(rel["kind"], []).append((tid, rel["test_ok"], any(solved)))
        da = [direct_answer(rel, a) for a, _ in tests]
        if all(x is not None for x in da):
            direct = all(x == b for x, (_, b) in zip(da, tests))
    rows.append(dict(task=tid, rels=rels, solved_runs=sum(solved), acc_tok=acc_tok, direct=direct,
                     tokens=ntok(R.relations_text(t)) if rels else 0))

out = ["# Facts v2 relations on the 120 evaluation tasks (offline)\n"]
n_with = sum(1 for r in rows if r["rels"])
out.append(f"- tasks with >= 1 relation: {n_with}/120; relations block tokens: median "
           f"{st.median([r['tokens'] for r in rows if r['rels']]) if n_with else 0}, max {max(r['tokens'] for r in rows)}")
dirs = [r for r in rows if r["direct"] is not None]
out.append(f"- tasks whose relation determines the answer outright: {len(dirs)}; correct on every test input: "
           f"{sum(r['direct'] for r in dirs)} {[r['task'] for r in dirs if r['direct']]}; Duck R3K solved them in "
           f"{sum(r['solved_runs'] for r in dirs if r['direct'])}/{2 * sum(r['direct'] for r in dirs)} runs")
out.append("\n| relation | tasks | holds on the test pairs (precision) | of those, Duck solved in >= 1 R3K run |")
out.append("|---|---|---|---|")
for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
    out.append(f"| {k} | {len(v)} | {sum(x[1] for x in v)}/{len(v)} ({100 * sum(x[1] for x in v) / len(v):.0f} %) | "
               f"{sum(x[2] for x in v)}/{len(v)} |")
# Duck link: solved tasks' tokens with vs without a relation; unsolved tasks with a precise relation
w = [t for r in rows if r["rels"] for t in r["acc_tok"]]
wo = [t for r in rows if not r["rels"] for t in r["acc_tok"]]
out.append(f"\n- Duck correct accepts (R3K, both runs): tasks WITH a relation: {len(w)}, median tokens "
           f"{st.median(w) / 1e3 if w else 0:.1f}k; WITHOUT: {len(wo)}, median {st.median(wo) / 1e3 if wo else 0:.1f}k")
uns = [r for r in rows if r["rels"] and r["solved_runs"] == 0 and all(x["test_ok"] for x in r["rels"])]
out.append(f"- tasks Duck never solved (0/2 runs) that have only precise relations (all hold on the test): {len(uns)}")
part = [r for r in rows if r["rels"] and r["solved_runs"] < 2 and all(x["test_ok"] for x in r["rels"])]
out.append(f"- tasks solved in < 2 runs that have only precise relations: {len(part)}")
misl = [r for r in rows if any(not x["test_ok"] for x in r["rels"])]
out.append(f"- tasks where some relation does NOT hold on the test (would mislead): {len(misl)} "
           f"{[(r['task'], [x['kind'] for x in r['rels'] if not x['test_ok']]) for r in misl]}")
out.append("\n## Per task (tasks with relations)\n\n| task | Duck runs solved | Duck correct-accept tokens | relations (test ok?) |")
out.append("|---|---|---|---|")
for r in rows:
    if r["rels"]:
        out.append(f"| {r['task']} | {r['solved_runs']}/2 | {[round(t / 1e3, 1) for t in r['acc_tok']]} | "
                   + "; ".join(f"{x['kind']}{'' if x['test_ok'] else ' (TEST FAILS)'}" for x in r["rels"]) + " |")
(HERE / "eval_relations_out.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out[:40]))
ex = next(r for r in rows if len(r["rels"]) >= 2)
print("\nexample block for", ex["task"], ":\n" + R.relations_text(CH[ex["task"]]))
