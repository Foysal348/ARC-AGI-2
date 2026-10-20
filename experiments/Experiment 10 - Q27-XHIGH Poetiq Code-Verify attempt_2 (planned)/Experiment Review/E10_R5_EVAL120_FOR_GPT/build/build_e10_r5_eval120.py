"""E10 Poetiq R5 + EVAL120 full evaluation run (Claude, 2026-10-14, user request). Deterministic.

Source: Acknowledgement/E10_POETIQ_R4T_SAFE_SANDBOX_R5.ipynb (GPT R5, sha 68b0d28f...) = R4 + full-thinking logging +
non-raising sandbox self-test + clone3->ENOSYS (Kaggle CPU probe PASS). Everything is kept as in R5:
77,824 thinking + 8,192 answer, MAX_MODEL_LEN 106,496, 1 expert, 10 iterations, temperature 1.0, prompts, voting.
Only change (cells 0 and 2): profile EVAL120 = all 120 evaluation tasks / 172 test inputs on the hidden-run clock
(budget 12 h - 10 min; admit until 694 min, drain 696, hard stop 704 - the RERUN_ALL margins), made the commit default.
Hidden rerun (RERUN_ALL) unchanged.
usage: python build_e10_r5_eval120.py
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "Acknowledgement" / "E10_POETIQ_R4T_SAFE_SANDBOX_R5.ipynb"
SRC_SHA = "68b0d28f758476336182a5df8a522964982ed5329241f5540ff2ce9a4cd7cfca"
OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R5_EVAL120.ipynb"


def once(text, old, new, where):
    if text.count(old) != 1:
        sys.exit(f"replacement not unique ({text.count(old)}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "R5 source changed"
nb = json.loads(raw)
c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
t = src(2)
t = once(t, '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n',
         '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n'
         '    # R5 EVAL120: full evaluation run (120 tasks / 172 inputs) on the hidden-run clock (RERUN_ALL margins)\n'
         '    "EVAL120":   {"source": "eval", "n_tasks": 120, "time_budget_sec": 12.0 * 3600 - 10 * 60,\n'
         '                  "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 6 * 60, "verify_margin_sec": 3 * 60,\n'
         '                  "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60},\n', "profile row")
t = once(t, 'os.environ.get("Q27X_PROFILE", "EVAL12")', 'os.environ.get("Q27X_PROFILE", "EVAL120")', "default profile")
c[2]["source"] = t.splitlines(keepends=True)
c[0]["source"] = (src(0) + "\n\n**R5 + EVAL120 (2026-10-14):** commit run = all 120 evaluation tasks / 172 test inputs on the "
                  "hidden-run clock (admit until 11:34, drain 11:36, hard stop 11:44). Everything else is R5 unchanged "
                  "(77,824 + 8,192, full thinking saved to q27x/poetiq_thinking/). `Q27X_PROFILE=EVAL12` gives the R5 default.\n"
                  ).splitlines(keepends=True)
for cell in c:
    if cell["cell_type"] == "code":
        cell["outputs"] = []; cell["execution_count"] = None
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
OUT.write_bytes(data)
print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())
