"""Part B helper: for every train-pass-but-wrong attempt (false_verif.json) write cases/<build>_<task>_a<k>.txt with
 - the submitted function's source and the transitive closure of the helper functions it calls
   (last definition of each top-level def found in the python tool calls of transcript.txt),
 - wrong cells (row, col, gold, pred) per test,
 - the model's final note lines that mention the submission.
"""
import ast, json, os, re

ROOT = r"D:\ARC AGI 2"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, r"experiments\Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)\Experiment Review\E10_ONE_THINKING_TWO_ATTEMPTS_FOR_ASTRA\data")
sol = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))
rows = json.load(open(os.path.join(HERE, "false_verif.json")))
os.makedirs(os.path.join(HERE, "cases"), exist_ok=True)

CALL_RE = re.compile(r"<parameter=code>\n(.*?)\n</parameter>", re.S)

for r in rows:
    bdir = os.path.join(ROOT, r["branch_dir"])
    tr = open(os.path.join(bdir, "transcript.txt"), encoding="utf-8", errors="replace").read()
    br = json.load(open(os.path.join(bdir, "branch_result.json")))
    sub = [s for s in br["submissions"] if s.get("accepted")][-1]
    defs = {}
    for code in CALL_RE.findall(tr):
        try:
            mod = ast.parse(code)
        except SyntaxError:
            continue
        for node in mod.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defs[node.name] = ast.get_source_segment(code, node)
            elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                seg = ast.get_source_segment(code, node)
                if seg and len(seg) < 600:
                    for t in node.targets:
                        defs[t.id] = seg
    need, seen = [r["sub_name"]], []
    while need:
        n = need.pop()
        if n in seen or n not in defs:
            continue
        seen.append(n)
        try:
            names = {x.id for x in ast.walk(ast.parse(defs[n])) if isinstance(x, ast.Name)}
        except SyntaxError:
            names = set()
        need.extend(sorted(names & set(defs)))
    lines = [f"# {r['build']} {r['task']} attempt {r['attempt']} sub={r['sub_name']} tokens={r['tokens']} src_chars={r['src_chars']}"]
    lines.append(f"# train {r['n_train']} in {r['train_in']} out {r['train_out']} | test in {r['test_in']} gold {r['gold']} | new test colours {r['new_test_colours']}")
    for i, g in enumerate(sol[r["task"]]):
        p = sub["test_outputs"][i] if i < len(sub["test_outputs"]) else None
        if p is None or len(p) != len(g) or len(p[0]) != len(g[0]):
            lines.append(f"# test {i}: shape pred {None if p is None else (len(p), len(p[0]))} gold {(len(g), len(g[0]))}")
            continue
        w = [(a, b, g[a][b], p[a][b]) for a in range(len(g)) for b in range(len(g[0])) if g[a][b] != p[a][b]]
        lines.append(f"# test {i}: {len(w)} wrong cells (r,c,gold,pred): {w[:40]}")
    lines.append("\n# ---- source closure ----")
    for n in reversed(seen):
        lines.append(defs[n])
        lines.append("")
    open(os.path.join(HERE, "cases", f"{r['build']}_{r['task']}_a{r['attempt']}.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print(r["build"], r["task"], r["attempt"], "funcs", len(seen), "chars", sum(len(defs[n]) for n in seen))
