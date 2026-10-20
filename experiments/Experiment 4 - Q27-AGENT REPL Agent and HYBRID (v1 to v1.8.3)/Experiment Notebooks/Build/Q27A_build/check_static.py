"""Static gate before any push: every code cell / %%writefile module of the notebook must parse, and pyflakes must
report nothing but the known-harmless source-notebook leftover (repl.py `pre`). JSON data cells must load.
usage: python check_static.py <notebook>"""
import ast
import json
import pathlib
import subprocess
import sys
import tempfile

nb = json.load(open(sys.argv[1], encoding="utf-8"))
d = pathlib.Path(tempfile.mkdtemp(prefix="static_"))
errors = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    first = s.splitlines()[0]
    body = "\n".join(s.splitlines()[1:]) if first.startswith("%%writefile") else s
    if first.endswith(".json"):
        try:
            json.loads(body)
        except Exception as e:
            errors.append(f"cell {i} json: {e}")
        continue
    name = first.split("/")[-1] if first.startswith("%%writefile") else f"cell_{i:02d}.py"
    (d / name).write_text(body + "\n", encoding="utf-8")
    try:
        ast.parse(body)
    except SyntaxError as e:
        errors.append(f"{name}:{e.lineno}: {e.msg}")
r = subprocess.run([sys.executable, "-m", "pyflakes", str(d)], capture_output=True, text=True)
for line in r.stdout.splitlines():
    line = line.replace(str(d) + "\\", "").replace(str(d) + "/", "")
    if line.startswith("repl.py:") and "redefinition of unused 'pre'" in line:
        continue                                   # source-notebook leftover (preexec_fn helper, unused since v1.4)
    errors.append("pyflakes " + line)
print("\n".join(errors) if errors else "static gate: clean")
print("STATIC GATE PASS" if not errors else "STATIC GATE FAIL")
sys.exit(1 if errors else 0)
