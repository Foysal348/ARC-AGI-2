"""E10 BASE 86K + Qwen-specific 2-rule puzzle ordering (Claude, 2026-10-14). Deterministic; every replacement hits once.

Source: Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb (sha 530cb119..., = v1.4.4 DIRECT LB 26.94 + 86k, no resample, original order).
One change: the scheduler queue follows a Qwen3.8-27B-specific order instead of sorted task ids.
  order score per task (training pairs / task layout only, no model call):
      +1 if the task has >= 3 training examples
      +1 if the task has exactly 1 test input
  higher score first; ties keep the original sorted-id order; units of a task stay together (query 0, 1, ...).
Evidence: resources/Puzzle Capability Ledger/derive_ordering_rules.py (120 evaluation tasks; nested-CV AUC 0.57 vs
NVARC tier/fine 0.51/0.53; solve rate by score 2/1/0 = 58/43/20 %). The order applies in every mode (commit + hidden rerun).
Records carry order_score and admit_rank. Budgets, prompt, sampling, engine and deadlines are unchanged.
Commit-run profile MIX20 (user request 2026-10-14; replaces the earlier EVAL20F, EVAL12 dropped as over-sampled):
  FAST8   = the 8 evaluation tasks that closed their thinking naturally with the FEWEST tokens in E8 R5 (PROXY48, 49k):
            e376de54 17.6k, b99e7126 18.7k, 31f7f899 31.0k, dbff022c 35.3k, 5961cc34 37.0k, c4d067a0 39.5k,
            3dc255db 39.7k, 7491f3cf 41.8k (6 of 8 solved there);
  FRESH12 = 12 tasks never run by us, sampled with random.Random(20261014) by stratum: 4 ARC-Prize-xhigh-solved with order
            score 2 + 4 xhigh-solved with score <= 1 + 2 solved only at another level + 1 never-solved score 2 + 1 never-solved
            score <= 1.
  20 tasks / 23 test inputs, 3 h.
Default commit-run profile EVAL120 (user request 2026-10-14, "full run"): all 120 evaluation tasks / 172 test inputs with
the hidden-run clock (11 h 50 min budget; admit until 11:34, drain 11:36, hard stop 11:44 - the RERUN_ALL margins).
MIX20 stays available via Q27X_PROFILE=MIX20.
usage: python build_e10_order2.py
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb"
SRC_SHA = "530cb119e302899426a2e3f24c5e100279e7167aef77525b6cb93b90a7fc10c0"
OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb"

BANNER_ADD = ("\n\n**ORDER2 variant (2026-10-14):** the queue follows a Qwen3.8-27B-specific order in every mode (commit and hidden "
              "rerun): score = [>= 3 training examples] + [1 test input], higher first, ties in the original sorted order. "
              "Everything else is the E10 BASE 86K notebook.")


def once(text, old, new, where):
    if text.count(old) != 1:
        sys.exit(f"replacement not unique ({text.count(old)}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "E10 base changed"
nb = json.loads(raw)
c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
FAST8 = ["e376de54", "b99e7126", "31f7f899", "dbff022c", "5961cc34", "c4d067a0", "3dc255db", "7491f3cf"]
FRESH12 = ["db0c5428", "b5ca7ac4", "58f5dbd5", "bf45cf4b", "edb79dae", "80a900e0", "d35bdbdc", "8e5c0c38",
           "58490d8a", "b10624e5", "8b9c3697", "8698868d"]
MIX20 = sorted(FAST8 + FRESH12)
assert len(set(MIX20)) == 20
BANNER_ADD += (" Commit run = **EVAL120** (full run): all 120 evaluation tasks / 172 test inputs, scored, with the hidden-run clock "
               "(admit until 11:34, drain 11:36, hard stop 11:44). Other profiles: `Q27X_PROFILE=MIX20` (8 tasks that closed "
               "fastest in E8 + 12 never-run tasks; 20 tasks / 23 inputs, 3 h) or `EVAL12`.")
c[0]["source"] = (src(0) + BANNER_ADD + "\n").splitlines(keepends=True)

# ---- cell 2: MIX20 profile, default for the commit run
t = src(2)
t = once(t, '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n',
         '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n'
         '    # ORDER2 MIX20: 8 fastest natural closes in E8 R5 + 12 never-run tasks (stratified, seed 20261014) = 20 tasks / 23 inputs\n'
         f'    "MIX20":     {{"source": "eval", "task_list": {json.dumps(MIX20)}, "time_budget_sec": 3.0 * 3600,\n'
         '                  "admit_frac": 0.80, "drain_frac": 0.90},\n'
         '    # ORDER2 EVAL120: full evaluation run (120 tasks / 172 inputs) on the hidden-run clock (RERUN_ALL margins)\n'
         '    "EVAL120":   {"source": "eval", "n_tasks": 120, "time_budget_sec": 12.0 * 3600 - 10 * 60,\n'
         '                  "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 6 * 60, "verify_margin_sec": 3 * 60,\n'
         '                  "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60},\n', "profile row")
t = once(t, 'os.environ.get("Q27X_PROFILE", "EVAL12")', 'os.environ.get("Q27X_PROFILE", "EVAL120")', "default profile")
c[2]["source"] = t.splitlines(keepends=True)

# ---- cell 4: order score + ordered units (after UNITS / UNIT_INDEX are built)
t = src(4)
t = once(t, 'N_HIDDEN_UNITS = sum(len(t["test"]) for t in test_challenges.values())',
         '# ORDER2: Qwen3.8-27B-specific order (resources/Puzzle Capability Ledger/derive_ordering_rules.py)\n'
         'def order_score(task):\n'
         '    return int(len(task["train"]) >= 3) + int(len(task["test"]) == 1)\n'
         'ORDER_SCORE = {tid: order_score(CHALLENGES[tid]) for tid in TASK_IDS}\n'
         'ORDERED_TASKS = sorted(TASK_IDS, key=lambda tid: (-ORDER_SCORE[tid], TASK_IDS.index(tid)))\n'
         'ORDERED_UNITS = [f"{tid}#{qi}" for tid in ORDERED_TASKS for qi in range(len(CHALLENGES[tid]["test"]))]\n'
         'assert len(ORDERED_UNITS) == len(UNITS) and set(ORDERED_UNITS) == set(UNITS)\n'
         'ORDER_ADMIT_RANK = {u: i for i, u in enumerate(ORDERED_UNITS)}\n'
         'print("ORDER2: tasks by score 2/1/0 =", [sum(v == k for v in ORDER_SCORE.values()) for k in (2, 1, 0)],\n'
         '      "| first units:", ORDERED_UNITS[:5])\n'
         'N_HIDDEN_UNITS = sum(len(t["test"]) for t in test_challenges.values())', "ordered units")
assert 'assert order_score({"train": [0, 0, 0], "test": [0]}) == 2' not in t
t += ('\nassert order_score({"train": [0, 0, 0], "test": [0]}) == 2 and order_score({"train": [0, 0], "test": [0, 0]}) == 0'
      ' and order_score({"train": [0, 0, 0, 0], "test": [0, 0]}) == 1\n')
c[4]["source"] = t.splitlines(keepends=True)

# ---- cell 7: queue in ORDER2 order; records carry order fields
t = src(7)
t = once(t, '    queue = [u for u in UNITS if u in PROMPT_IDS]', '    queue = [u for u in ORDERED_UNITS if u in PROMPT_IDS]   # ORDER2', "queue")
t = once(t, '"t_done_min": round((st["t_done"] - NOTEBOOK_T0) / 60, 2)}',
         '"t_done_min": round((st["t_done"] - NOTEBOOK_T0) / 60, 2),\n'
         '           "order_score": ORDER_SCORE[UNIT_TASK[uid]], "admit_rank": ORDER_ADMIT_RANK[uid]}', "record fields")
c[7]["source"] = t.splitlines(keepends=True)

for cell in c:
    if cell["cell_type"] == "code":
        cell["outputs"] = []; cell["execution_count"] = None
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
OUT.write_bytes(data)
print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest()[:16])
