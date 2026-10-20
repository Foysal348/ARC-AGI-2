"""E10 Poetiq R4 EVAL22 variant (Claude, 2026-10-14, user request: "22 tasks"). Deterministic.

Source: Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4T.ipynb (R4 + thinking saved, sha 1f8fd03e...), code cells 5 and 7 untouched.
Only change: cell 2 gets profile EVAL22 = the 12 EVAL12 tasks (TASK_SAMPLE_SEED 20261005, unchanged, so the DIRECT
comparison with E9 stays valid) + 10 RULE20 tasks (random.Random(20261014).sample(RULE20, 10)); EVAL22 becomes the
commit-run default. 22 tasks / 32 test inputs, 3 h budget, admit/drain 0.80/0.90 like EVAL12. Hidden rerun unchanged.
usage: python build_e10_r4_eval22.py
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4T.ipynb"   # R4 + thinking saved
SRC_SHA = "1f8fd03ec0b3ec5eb7733aec1a78a7e31516bbaad97b836ffd553dd4b82ac502"
OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R4T_EVAL22.ipynb"

EVAL12 = ["136b0064", "20a9e565", "21897d95", "247ef758", "2b83f449", "4c3d4a41", "4c7dc4dd", "6e4f6532", "88bcf3b4",
          "898e7135", "8b7bacbf", "b6f77b65"]
RULE20_10 = ["7b0280bc", "7b3084d4", "981571dc", "a395ee82", "a6f40cea", "b5ca7ac4", "c4d067a0", "da515329", "dbff022c", "dfadab01"]
EVAL22 = sorted(EVAL12 + RULE20_10)
assert len(set(EVAL22)) == 22


def once(text, old, new, where):
    if text.count(old) != 1:
        sys.exit(f"replacement not unique ({text.count(old)}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "R4 source changed"
nb = json.loads(raw)
c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
t = src(2)
t = once(t, '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n',
         '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n'
         '    # E10 R4 EVAL22: the 12 EVAL12 tasks + 10 RULE20 tasks (>= 3 train, 1 test; seed 20261014) = 22 tasks / 32 inputs\n'
         f'    "EVAL22":    {{"source": "eval", "task_list": {json.dumps(EVAL22)}, "time_budget_sec": 3.0 * 3600,\n'
         '                  "admit_frac": 0.80, "drain_frac": 0.90},\n', "profile row")
t = once(t, 'os.environ.get("Q27X_PROFILE", "EVAL12")', 'os.environ.get("Q27X_PROFILE", "EVAL22")', "default profile")
c[2]["source"] = t.splitlines(keepends=True)
c[0]["source"] = (src(0) + "\n\n**EVAL22 variant (2026-10-14):** commit run = 22 evaluation tasks (the 12 EVAL12 tasks + 10 RULE20 "
                  "tasks), 32 test inputs, 3 h. R4T code cells unchanged (full thinking saved to q27x/poetiq_thinking/). `Q27X_PROFILE=EVAL12` restores the R4 default.\n"
                  ).splitlines(keepends=True)
for cell in c:
    if cell["cell_type"] == "code":
        cell["outputs"] = []; cell["execution_count"] = None
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
OUT.write_bytes(data)
print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest()[:16])
