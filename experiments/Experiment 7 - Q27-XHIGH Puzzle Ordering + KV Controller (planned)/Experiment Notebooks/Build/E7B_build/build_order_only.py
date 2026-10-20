"""Experiment 7, ORDERING ONLY (user request 2026-10-13). Claude.

Base: v1.4.4 DIRECT (LB 26.94), tests/fixtures/Q27-XHIGH-v1_4_4.ipynb. Changes, nothing else:
  1. cell 4: exact Codex/NVARC get_puzzle_priority (copied from the donor) + units admitted easy-first
     (commit AND hidden rerun).
  2. cell 2: PROXY48 evaluation profile (default for the commit run); cell 4: fixed task-list support.
  3. cell 7: the queue uses the Codex order; records carry tier, admit rank, admit/finish minutes.
No KV controller, engine cap 16, prompt / sampling / budgets / deadlines identical to v1.4.4.
usage: python build_order_only.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "tests" / "fixtures" / "Q27-XHIGH-v1_4_4.ipynb"
BASE_SHA = "959959806170960c"
DONOR = HERE / "tests" / "fixtures" / "codex_donor.ipynb"
OUT = HERE.parent.parent / "Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb"
sys.path.insert(0, str(HERE))
from build_e7b import PROXY48, once, src_of, set_src  # noqa: E402

BANNER = ("**Experiment 7 ORDERING ONLY:** v1.4.4 (LB 26.94) + exact Codex/NVARC puzzle ordering (easy tier first), "
          "in the commit run AND the hidden rerun. Commit run = PROXY48 (48 tasks / 56 outputs, 3 h budget). "
          "No KV controller; everything else is v1.4.4.")


def build():
    raw = BASE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == BASE_SHA, "base changed"
    nb = json.loads(raw)
    c = nb["cells"]
    d = "".join(json.loads(DONOR.read_text(encoding="utf-8"))["cells"][6]["source"])
    fn = d[d.index("def get_puzzle_priority"):d.index("\ndef local_worker")].strip() + "\n"
    set_src(c[0], BANNER + "\n\n" + src_of(c[0]))
    # cell 2: PROXY48 profile, default for the commit run
    t = src_of(c[2])
    t = once(t, 'os.environ.get("Q27X_PROFILE", "EVAL12")', 'os.environ.get("Q27X_PROFILE", "PROXY48")', "profile")
    t = once(t, '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n',
             '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n'
             '    # E7: EDA LB-mirror proxy (MAIN_DEBUG_48), 48 tasks / 56 test outputs\n'
             f'    "PROXY48":   {{"source": "eval", "task_list": {json.dumps(PROXY48)}, "time_budget_sec": 3.0 * 3600,\n'
             '                  "admit_frac": 0.80, "drain_frac": 0.90},\n', "profile table")
    set_src(c[2], t)
    # cell 4: task list + Codex ordering
    t = src_of(c[4])
    t = once(t, '    TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))\n',
             '    if P.get("task_list"):   # E7: fixed proxy list (every id must exist in the evaluation file)\n'
             '        assert all(t in _ch for t in P["task_list"]), "PROXY48 task missing from evaluation file"\n'
             '        TASK_IDS = sorted(P["task_list"])\n'
             '    else:\n'
             '        TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))\n', "task list")
    t = once(t, '    CHALLENGES = dict(test_challenges); TASK_IDS = sorted(CHALLENGES); SOLUTIONS = None\n',
             '    CHALLENGES = dict(test_challenges); TASK_IDS = sorted(CHALLENGES); SOLUTIONS = None\n\n'
             '# E7: EXACT Codex/NVARC get_puzzle_priority (verbatim); ties keep the challenge-file order\n'
             'import numpy as np\n' + fn +
             '\n_ORDER_SRC = _ch if P["source"] == "eval" else test_challenges\n'
             'CODEX_ORDERED_TASKS = sorted((tid for tid in _ORDER_SRC if tid in CHALLENGES),\n'
             '                             key=lambda tid: get_puzzle_priority(tid, _ORDER_SRC))\n'
             'CODEX_PRIORITY_TIERS = {tid: int(get_puzzle_priority(tid, _ORDER_SRC)[0]) for tid in CODEX_ORDERED_TASKS}\n',
             "ordering fn")
    t = once(t, 'N_HIDDEN_UNITS = sum(len(t["test"]) for t in test_challenges.values())',
             'ORDERED_UNITS = [f"{tid}#{qi}" for tid in CODEX_ORDERED_TASKS for qi in range(len(CHALLENGES[tid]["test"]))]\n'
             'assert len(ORDERED_UNITS) == len(UNITS) and set(ORDERED_UNITS) == set(UNITS)\n'
             'ORDER_ADMIT_RANK = {u: i for i, u in enumerate(ORDERED_UNITS)}\n'
             'print("E7 Codex ordering: tiers", {k: sum(v == k for v in CODEX_PRIORITY_TIERS.values()) for k in (1, 2, 3)},\n'
             '      "first", CODEX_ORDERED_TASKS[:5], "last", CODEX_ORDERED_TASKS[-5:])\n'
             'N_HIDDEN_UNITS = sum(len(t["test"]) for t in test_challenges.values())', "ordered units")
    set_src(c[4], t)
    # cell 7: queue in Codex order + record fields
    t = src_of(c[7])
    t = once(t, '    queue = [u for u in UNITS if u in PROMPT_IDS]', '    queue = [u for u in ORDERED_UNITS if u in PROMPT_IDS]   # E7',
             "queue")
    t = once(t, '"score": sc, "p1_cap": P1_CAPS.get(uid)}',
             '"score": sc, "p1_cap": P1_CAPS.get(uid),\n'
             '           "e7_admit_rank": ORDER_ADMIT_RANK[uid], "e7_priority_tier": CODEX_PRIORITY_TIERS[UNIT_TASK[uid]],\n'
             '           "t_admit_min": round((st["t_admit"] - NOTEBOOK_T0) / 60, 2),\n'
             '           "t_done_min": round((st["t_done"] - NOTEBOOK_T0) / 60, 2)}', "record")
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
