"""Experiment 10 BASE builder (Claude, 2026-10-14). Deterministic; every replacement must hit exactly once.

E10 base = Experiment 9 WITHOUT the tail resample (E9 EVAL12: resample used 41 % of the tokens for 0 gain).
Source: v1.4.4 DIRECT (LB 26.94), source/Q27-XHIGH-v1_4_4_LB26.94.ipynb. NO puzzle ordering (original v1.4.4 order).
Changes (all taken from build_e9.py, E9 run 4/22 on EVAL12):
  1. 86k budget: THINK_BUDGET 49,152 -> 77,824 (answer 8,192 unchanged), MAX_MODEL_LEN 81,920 -> 106,496,
     EXPECTED_LIVE_TOKENS_PER_SEQ 30,000 -> 43,000, phase1_cap asserts for the new values.
  2. Eval profile PROXY56 in the table (commit-run default stays EVAL12); fixed task-list support in cell 4.
  3. Records carry t_admit_min / t_done_min.
Not included: E9 tail resample (CELL7_RESAMPLE, tail admission loop, resample report fields). The Poetiq
code-verify attempt_2 will be added on top of this base by a separate builder step.
Variant --rule20 (2026-10-14, user request): a SEPARATE notebook Q27-XHIGH-v1_4_4_E10_BASE_86K_RULE20.ipynb whose default
commit profile is RULE20 = 20 evaluation tasks satisfying both ordering rules (>= 3 training examples AND 1 test input),
sampled with random.Random(20261014) from the 52 such tasks (resources/Puzzle Capability Ledger/derive_ordering_rules.py).
The base notebook itself is unchanged (Poetiq R3 builds assert its sha).
usage: python build_e10_base.py [--rule20]
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "source" / "Q27-XHIGH-v1_4_4_LB26.94.ipynb"
SRC_SHA = "959959806170960c"
OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb"

PROXY48 = ["0934a4d8", "20a9e565", "4c7dc4dd", "89565ca0", "136b0064", "291dc1e1", "16b78196", "271d71e2",
           "28a6681f", "2b83f449", "2c181942", "31f7f899", "35ab12c3", "3dc255db", "409aa875", "4c416de3",
           "64efde09", "6e453dd6", "7491f3cf", "7b0280bc", "7b80bb43", "7c66cb00", "1818057f", "332f06d7",
           "53fb4810", "7666fa5d", "da515329", "8f3a5a89", "db695cfb", "97d7923e", "de809cff", "c7f57c3e",
           "cb2d8a2c", "58490d8a", "dbff022c", "5961cc34", "80a900e0", "269e22fb", "8b9c3697", "e376de54",
           "b99e7126", "c4d067a0", "135a2760", "4a21e3da", "71e489b6", "800d221b", "dd6b8c4b", "36a08778"]
EVAL12_EXTRA = ["21897d95", "247ef758", "4c3d4a41", "6e4f6532", "88bcf3b4", "898e7135", "8b7bacbf", "b6f77b65"]
PROXY56 = PROXY48 + EVAL12_EXTRA
assert len(set(PROXY56)) == 56
RULE20 = ["332f06d7", "3dc255db", "6ffbe589", "7491f3cf", "7b0280bc", "7b3084d4", "8f215267", "8f3a5a89", "981571dc", "9bbf930d",
          "a395ee82", "a47bf94d", "a6f40cea", "b5ca7ac4", "bf45cf4b", "c4d067a0", "da515329", "dbff022c", "dfadab01", "eee78d87"]
assert len(set(RULE20)) == 20
VARIANT_RULE20 = "--rule20" in sys.argv
if VARIANT_RULE20:
    OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E10_BASE_86K_RULE20.ipynb"

BANNER = ("**Experiment 10 BASE (86k):** v1.4.4 DIRECT (LB 26.94), original order (no puzzle ordering). "
          "Thinking budget **77,824** + answer 8,192 (MAX_MODEL_LEN 106,496). No tail resample. "
          "Commit run = EVAL12 (12 tasks / 22 test inputs, 3 h); profile PROXY56 (56 tasks / 72 inputs, 6 h) is available. "
          "Hidden rerun deadlines unchanged.")
if VARIANT_RULE20:
    BANNER = BANNER.replace("Commit run = EVAL12 (12 tasks / 22 test inputs, 3 h)",
                            "Commit run = **RULE20** (20 evaluation tasks with >= 3 training examples and 1 test input, 3 h; "
                            "Q27X_PROFILE=EVAL12 restores the old default)")
    assert "RULE20" in BANNER


def once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        sys.exit(f"replacement not unique ({n}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def build():
    raw = SRC.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == SRC_SHA, "base changed"
    nb = json.loads(raw)
    c = nb["cells"]
    set_src(c[0], BANNER + "\n\n" + src_of(c[0]))
    # ---- cell 2: budgets, context, profile
    t = src_of(c[2])
    t = once(t, 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 49152))',
             'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 77824))   # E9/E10: 86k total (77,824 + 8,192); was 49,152',
             "think budget")
    t = once(t, 'MAX_MODEL_LEN = int(os.environ.get("Q27X_MAX_MODEL_LEN", 81920))',
             'MAX_MODEL_LEN = int(os.environ.get("Q27X_MAX_MODEL_LEN", 106496))   # E9/E10: prompt (<= ~20k) + 77,824 + 8,192 fits; was 81,920',
             "max model len")
    t = once(t, 'if (MAX_MODEL_LEN, THINK_BUDGET, ANSWER_MAX_TOKENS) == (81920, 49152, 8192):\n',
             'if (MAX_MODEL_LEN, THINK_BUDGET, ANSWER_MAX_TOKENS) == (106496, 77824, 8192):   # E9/E10\n'
             '    assert phase1_cap(12800) == 77824 and phase1_cap(20000) == 77824 and phase1_cap(30000) == 68048\n'
             'if (MAX_MODEL_LEN, THINK_BUDGET, ANSWER_MAX_TOKENS) == (81920, 49152, 8192):\n', "asserts")
    t = once(t, 'EXPECTED_LIVE_TOKENS_PER_SEQ = 30000 ',
             'EXPECTED_LIVE_TOKENS_PER_SEQ = 43000 ', "expected live")
    t = once(t, '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n',
             '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n'
             '    # E9/E10: PROXY48 (EDA LB-mirror) + the 8 EVAL12 tasks not in it = 56 tasks / 72 test inputs\n'
             f'    "PROXY56":   {{"source": "eval", "task_list": {json.dumps(PROXY56)}, "time_budget_sec": 6.0 * 3600,\n'
             '                  "admit_frac": 0.80, "drain_frac": 0.90},\n', "profile table")
    if VARIANT_RULE20:
        t = once(t, '    # E9/E10: PROXY48 (EDA LB-mirror)',
                 '    # E10 RULE20: 20 eval tasks with >= 3 training examples AND 1 test input (seed 20261014 from the 52 such tasks)\n'
                 f'    "RULE20":    {{"source": "eval", "task_list": {json.dumps(RULE20)}, "time_budget_sec": 3.0 * 3600,\n'
                 '                  "admit_frac": 0.80, "drain_frac": 0.90},\n'
                 '    # E9/E10: PROXY48 (EDA LB-mirror)', "rule20 profile")
        t = once(t, 'os.environ.get("Q27X_PROFILE", "EVAL12")', 'os.environ.get("Q27X_PROFILE", "RULE20")', "rule20 default")
    set_src(c[2], t)
    # ---- cell 4: fixed task list
    t = src_of(c[4])
    t = once(t, '    TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))\n',
             '    if P.get("task_list"):   # E9/E10: fixed list (every id must exist in the evaluation file)\n'
             '        assert all(t in _ch for t in P["task_list"]), "task list id missing from evaluation file"\n'
             '        TASK_IDS = sorted(P["task_list"])\n'
             '    else:\n'
             '        TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))\n', "task list")
    set_src(c[4], t)
    # ---- cell 7: record times only (no resample)
    t = src_of(c[7])
    t = once(t, '"score": sc, "p1_cap": P1_CAPS.get(uid)}',
             '"score": sc, "p1_cap": P1_CAPS.get(uid),\n'
             '           "t_admit_min": round((st["t_admit"] - NOTEBOOK_T0) / 60, 2),\n'
             '           "t_done_min": round((st["t_done"] - NOTEBOOK_T0) / 60, 2)}', "record times")
    set_src(c[7], t)
    for cell in c:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    OUT.write_bytes(data)
    print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest()[:16])


if __name__ == "__main__":
    build()
