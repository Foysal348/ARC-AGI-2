"""Build PIV-Q27-EVO12: V3.1 infrastructure (config, data, tokenizer, guard, verifier, helper library, model load)
+ a new evolution cell over verified programs + scoring/complementarity cell. 12 frozen random evaluation tasks."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if (ROOT / "notebooks" / "PIV-Q27-V3_1-EVAL120-ALLFIX.ipynb").exists():      # review-bundle layout
    SRC = ROOT / "notebooks" / "PIV-Q27-V3_1-EVAL120-ALLFIX.ipynb"
    DST = ROOT / "notebooks" / "PIV-Q27-EVO12-EVOLUTION.ipynb"
    NV = ROOT / "posthoc" / "nvarc_eval120_frozen.json"
else:                                                                        # project layout
    D = Path(r"D:\ARC AGI 2\Research Base Experiment")
    SRC = D / "PIV-Q27-V3_1-EVAL120-ALLFIX.ipynb"
    DST = D / "PIV-Q27-EVO12-EVOLUTION.ipynb"
    NV = Path(r"D:\ARC AGI 2\Research Experiment result\EVO24_build\nvarc_eval120_frozen.json")
# First 12 of random.Random(20261004).sample(sorted(eval ids), 24), sorted: 9 NVARC misses, 3 NVARC full solves, 17 queries.
# 12 tasks keep the run short (round 0 = 48 calls, repair rounds <= 24 calls each).
SUBSET = ["16de56c4", "2d0172a1", "332f06d7", "4c3d4a41", "4e34c42c", "6e453dd6",
          "88bcf3b4", "a6f40cea", "b5ca7ac4", "c4d067a0", "c7f57c3e", "dbff022c"]

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]


def cell_with(text):
    hits = [c for c in cells if c["cell_type"] == "code" and text in "".join(c["source"])]
    assert len(hits) == 1, (text[:60], len(hits))
    return hits[0]


def edit(cell, old, new):
    src = "".join(cell["source"])
    assert src.count(old) == 1, (old[:80], src.count(old))
    cell["source"] = src.replace(old, new).splitlines(keepends=True)


c_cfg = cell_with("REPAIR_SEED_OFFSET = 3_000_000")
c_tasks = cell_with("def choose_smoke8_from_proxy():")
edit(c_cfg, 'EXPERIMENT_ID = "PIV-Q27-V3.1-EVAL120-ALLFIX"', 'EXPERIMENT_ID = "PIV-Q27-EVO12"')
edit(c_cfg, '"bundle_name": "PIV_Q27_V3_EVAL120_BUNDLE.zip",', '"bundle_name": "PIV_Q27_EVO12_BUNDLE.zip",')
edit(c_cfg, 'ACTIVE_TASK_LIMIT = int(PROFILE["active_tasks"])\n',
     'ACTIVE_TASK_LIMIT = int(PROFILE["active_tasks"])\n'
     '# EVO12: first 12 of random.Random(20261004).sample(sorted(eval ids), 24). Empty list = all 120.\n'
     'EVAL_SUBSET_TASK_IDS = ' + json.dumps(SUBSET) + '\n'
     'EVO_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))   # scoring rerun after Submit: hidden test set\n'
     'if EVO_RERUN:\n'
     '    EVAL_SUBSET_TASK_IDS = []\n'
     'print("EVO MODE:", "COMPETITION RERUN (hidden test -> submission.json)" if EVO_RERUN else "EVALUATION (12 frozen eval tasks, scored)")\n'
     'if TASK_SET == "EVAL120" and EVAL_SUBSET_TASK_IDS:\n'
     '    assert len(set(EVAL_SUBSET_TASK_IDS)) == len(EVAL_SUBSET_TASK_IDS), "duplicate subset task"\n'
     '    ACTIVE_TASK_LIMIT = len(EVAL_SUBSET_TASK_IDS)\n')
edit(c_tasks, 'elif TASK_SET == "EVAL120": ACTIVE_TASK_IDS = sorted(eval_challenges)',
     'elif TASK_SET == "EVAL120": ACTIVE_TASK_IDS = (list(EVAL_SUBSET_TASK_IDS) if EVAL_SUBSET_TASK_IDS else sorted(eval_challenges))')
edit(c_tasks, 'if TASK_SET == "PROXY48": ACTIVE_TASK_IDS = list(PROXY48_TASK_IDS)',
     '# EVO dual mode: in the scoring rerun every downstream cell addresses the HIDDEN test tasks through eval_challenges\n'
     '# (no answers). The public evaluation data stays available as EVAL_CHALLENGES_PUBLIC.\n'
     'EVAL_CHALLENGES_PUBLIC = eval_challenges\n'
     'if EVO_RERUN:\n'
     '    eval_challenges = test_challenges\n'
     '    eval_solutions = {}\n'
     '    ACTIVE_TASK_LIMIT = len(test_challenges)\n'
     'if TASK_SET == "PROXY48": ACTIVE_TASK_IDS = list(PROXY48_TASK_IDS)')
edit(c_tasks, 'if TASK_SET == "EVAL120":\n    if not ACTIVE_TASK_SET.issubset(set(eval_solutions)):',
     'if EVO_RERUN:\n    print("RERUN:", len(ACTIVE_TASK_IDS), "hidden test tasks |", sum(len(eval_challenges[t]["test"]) for t in ACTIVE_TASK_IDS), "queries")\n'
     'elif TASK_SET == "EVAL120":\n    if not ACTIVE_TASK_SET.issubset(set(eval_solutions)):')

# keep cells up to and including the model-load cell; drop the old agent loop, repair and scoring cells
i_load = cells.index(cell_with("CELL 7 — CPU-SAFE PREFLIGHT OR REAL vLLM"))
cells[:] = cells[:i_load + 1]

nvj = json.loads(NV.read_text(encoding="utf-8"))
nv_small = {"source_notebook": nvj["source_notebook"], "source_sha256": nvj["source_sha256"], "rule": nvj["rule"],
            "per_task_score": {t: sum(v) / len(v) for t, v in nvj["per_task_queries_solved"].items()}}
evo_src = (HERE / "cell_evo.py").read_text(encoding="utf-8")
rep_src = (HERE / "cell_report.py").read_text(encoding="utf-8").replace("__NVARC_JSON__", json.dumps(nv_small))
for src in (evo_src, rep_src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src.splitlines(keepends=True)})
cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## Run sequence — PIV-Q27-EVO12\n\n",
    "Cells 1-7: config, data, tokenizer/template, guard, fresh verifier, helper library, model load (unchanged from V3.1). "
    "Cell 8: evolution (4 candidates/task -> verify -> top-2 distinct parents -> 2 repair generations). "
    "Cell 9: pass@2 scoring, complementarity with the frozen NVARC map, verdict, bundle.\n"]})

head = cells[0]
hs = "".join(head["source"])
head["source"] = ("# PIV-Q27-EVO12 — evolution over verified programs, 12 random evaluation tasks\n\n"
                  "Same model (Qwen3.8-27B-FP8, 4×L4) and the same tested infrastructure as V3.1; the single long REPL "
                  "conversation is replaced by an evolution loop (Imbue-style): per task 4 independent candidates "
                  "(rule in plain words -> `solution(grid)`), fresh verification on every training pair, then up to two 2-slot "
                  "follow-up generations subject to the 32,768-token hard cap: the 2 best "
                  "distinct candidates repaired for 2 generations with an exact error map, ≤32k generated tokens per "
                  "task, pass@2 from train-exact programs. Tasks: first 12 of the seed-20261004 sample (9 NVARC misses, 3 NVARC full solves).\n\n"
                  "Pre-registered verdict (`evo_verdict`): STRONG GO if ≥3 solved and ≥2 of them NVARC misses; "
                  "PROMISING if ≥1 NVARC-miss solve; otherwise WEAK / REDESIGN. "
                  "A deadline stop gives INCOMPLETE / NO DECISION; a verifier infrastructure failure gives INVALID. "
                  "Scoring is canonical per-query pass@2 (two distinct outputs per test query from train-exact programs).\n\n---\n\n"
                  "<details><summary>V3.1 notes (infrastructure provenance)</summary>\n\n" + hs + "\n\n</details>\n"
                  ).splitlines(keepends=True)
nb["metadata"]["piv_q27_evo12"] = {"derived_from": SRC.name, "derived_from_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
                                   "subset": SUBSET, "subset_seed": 20261004}
DST.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))   # LF only: OS-independent SHA
print("wrote", DST, hashlib.sha256(DST.read_bytes()).hexdigest())
