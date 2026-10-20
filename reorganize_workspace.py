"""Reorganize D:\\ARC AGI 2 into  experiments\\Experiment N - Name\\{Experiment Notebooks, Experiment Review,
Experiment Results}  +  resources\\ . Files are MOVED (never deleted); every move is logged to
reorganization_manifest_2026-10-12.csv (old -> new) so it can be reversed. Unmatched items go to _Unsorted.
usage: python reorganize_workspace.py [--execute]      (default: dry run, prints the plan)"""
import csv
import fnmatch
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\ARC AGI 2")
EXECUTE = "--execute" in sys.argv
X = "experiments"
E = {
    1: "Experiment 1 - Q1 Direct Baseline (PROXY48)",
    2: "Experiment 2 - PIV-P0 Stateful REPL Agent",
    3: "Experiment 3 - ACT Reasoning-Activation Ablation",
    4: "Experiment 4 - TP Throughput Probe",
    5: "Experiment 5 - C1 Terminal-Commit and Repair Solver",
    6: "Experiment 6 - PIV-SUB Hidden-Test Submission (LB 1.67)",
    7: "Experiment 7 - V3 EVAL120 Baseline and S12 Subset",
    8: "Experiment 8 - EVO Program Evolution",
    9: "Experiment 9 - Flash-Next Feasibility (FN)",
    10: "Experiment 10 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)",
    11: "Experiment 11 - R6 Code Loop",
    12: "Experiment 12 - Q27-AGENT REPL Agent and HYBRID (v1 to v1.8.3)",
    13: "Experiment 13 - Q27-AGENT v1.9 Long-Think Synthesis (planned)",
}
NB, RV, RS = "Experiment Notebooks", "Experiment Review", "Experiment Results"
RUNS = RS + "/Kaggle Runs"
BUILD = NB + "/Build"
R = "resources"


# The current phase (from 2026-10-05: LB era on the Qwen3.8-27B-FP8 + vLLM 0.19 stack, plus Flash-Next) gets full
# experiment folders, renumbered; everything earlier goes into ONE archive folder, grouped by family.
NEW = {10: "Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)",
       11: "Experiment 2 - R6 Code Loop",
       9: "Experiment 3 - Flash-Next Feasibility (FN)",
       12: "Experiment 4 - Q27-AGENT REPL Agent and HYBRID (v1 to v1.8.3)",
       13: "Experiment 5 - Q27-AGENT v1.9 Long-Think Synthesis (planned)"}
ARCH = "archive/Earlier Qwen3.8 experiments (2026-09-13 to 2026-10-04)"


def ex(n, sub):
    if n in NEW:
        return f"{X}/{NEW[n]}/{sub}"
    return f"{ARCH}/{E[n].split(' - ', 1)[1]}/{sub}"


# (source folder, pattern, destination folder) -- first match wins; patterns match the item name
RULES = [
    # ---- Qwen 3.8 27B Experiment (Q1)
    ("Qwen 3.8 27B Experiment", "Q1-DIR-*.ipynb", ex(1, NB)),
    ("Qwen 3.8 27B Experiment", "Q1_DIR_*BUNDLE*.zip", ex(1, RS)),
    ("Qwen 3.8 27B Experiment", "experiment-1-*", ex(1, RUNS)),
    # ---- Research Base Experiment (source notebooks)
    ("Research Base Experiment", "PIV-Q27-P0-*", ex(2, NB)),
    ("Research Base Experiment", "PIV-Q27-ACT-*", ex(3, NB)),
    ("Research Base Experiment", "TP-Q27-*", ex(4, NB)),
    ("Research Base Experiment", "PIV-Q27-C1-*", ex(5, NB)),
    ("Research Base Experiment", "PIV-Q27-SUB-*", ex(6, NB)),
    ("Research Base Experiment", "PIV-Q27-V3*", ex(7, NB)),
    ("Research Base Experiment", "PIV-Q27-EVAL120-*", ex(7, NB)),
    ("Research Base Experiment", "PIV-Q27-EVO*", ex(8, NB)),
    ("Research Base Experiment", "FN-*", ex(9, NB)),
    ("Research Base Experiment", "Q27-XHIGH-*", ex(10, NB)),
    ("Research Base Experiment", "q27-xhigh*", ex(10, NB)),
    ("Research Base Experiment", "Q27-R6-*", ex(11, NB)),
    ("Research Base Experiment", "Q27-AGENT-*", ex(12, NB)),
    ("Research Base Experiment", "ARC_AGI2_Agentic_Solver_Portability_Notes*", f"{R}/Analysis and Journals"),
    # ---- Research Experiment result
    ("Research Experiment result", "ARC3_top10_kv", f"{R}/ARC-AGI-3 related"),
    ("Research Experiment result", "P1_SPARSE_RUN_EVIDENCE*", f"{R}/ARC-AGI-3 related"),
    ("Research Experiment result", "notebook6e705f697b*", f"{R}/ARC-AGI-3 related"),
    ("Research Experiment result", "CANONICAL_BASELINE_EVAL120_*", ex(7, RS)),
    ("Research Experiment result", "EVO12_run", ex(8, RUNS)),
    ("Research Experiment result", "EVO24_build", ex(8, BUILD)),
    ("Research Experiment result", "EXPERIMENT_PLAN_2026-10-04.md", ex(8, "")),
    ("Research Experiment result", "FN_*REVIEW*.zip", ex(9, RV)),
    ("Research Experiment result", "FN_SMOKE_BUNDLE.zip", ex(9, RS)),
    ("Research Experiment result", "FN_*build*", ex(9, BUILD)),
    ("Research Experiment result", "FN_*run*", ex(9, RUNS)),
    ("Research Experiment result", "fn-smoke-*", ex(9, RUNS)),
    ("Research Experiment result", "GPT_DEEP_RESEARCH_PROMPT_*", f"{R}/Planning and Prompts"),
    ("Research Experiment result", "LB_compare", f"{R}/Reference Notebooks"),
    ("Research Experiment result", "russcore_*", f"{R}/Reference Notebooks"),
    ("Research Experiment result", "PIV-Q27-P0-*", ex(2, RUNS)),
    ("Research Experiment result", "PIV_Q27_P0_*", ex(2, RS)),
    ("Research Experiment result", "PIV_Q27_ACT_*", ex(3, RS)),
    ("Research Experiment result", "TP_Q27_*", ex(4, RS)),
    ("Research Experiment result", "PIV_Q27_C1_*AUDIT*", ex(5, RV)),
    ("Research Experiment result", "PIV_Q27_C1_*", ex(5, RS)),
    ("Research Experiment result", "harness_c1_*", ex(5, BUILD)),
    ("Research Experiment result", "PIV_Q27_KAGGLE_RUNS_REVIEW_*", ex(6, RV)),
    ("Research Experiment result", "PIV_Q27_REVIEW_BUNDLE_*", ex(6, RV)),
    ("Research Experiment result", "PIV_Q27_SUB_*", ex(6, RV)),
    ("Research Experiment result", "PIV_Q27_V31_S12_RESULT_*", ex(7, RS)),
    ("Research Experiment result", "PIV_Q27_V3*REVIEW*", ex(7, RV)),
    ("Research Experiment result", "PIV_Q27_EVO12_BUNDLE.zip", ex(8, RS)),
    ("Research Experiment result", "PIV_Q27_EVO24_REVIEW*", ex(8, RV)),
    ("Research Experiment result", "piv-q27-evo12-*", ex(8, RUNS)),
    ("Research Experiment result", "Q27X_run5_R6", ex(11, RUNS)),
    ("Research Experiment result", "q27-r6-code-loop*", ex(11, RUNS)),
    ("Research Experiment result", "Q27X_build", ex(10, BUILD)),
    ("Research Experiment result", "Q27X_run*", ex(10, RUNS)),
    ("Research Experiment result", "Q27X_BUNDLE*", ex(10, RS)),
    ("Research Experiment result", "Q27X_*REVIEW*", ex(10, RV)),
    ("Research Experiment result", "Q27X_v1_*", ex(10, RV)),
    ("Research Experiment result", "q27-xhigh-*", ex(10, RUNS)),
    ("Research Experiment result", "fork_q27_check", ex(10, RS)),
    ("Research Experiment result", "Q27A_build", ex(12, BUILD)),
    ("Research Experiment result", "Q27A_run1_esfer", ex(12, RUNS)),
    ("Research Experiment result", "Q27_AGENT_v1_9_PLAN_README.md", ex(13, "")),
    ("Research Experiment result", "Q27_AGENT_v1_9_*", ex(13, RV)),
    ("Research Experiment result", "Q27_AGENT_*", ex(12, RV)),
    ("Research Experiment result", "Q27_DIRECT_vs_AGENT_*", ex(12, RV)),
    # Kaggle-downloaded executed notebooks (identified by their first cell)
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 (10).ipynb", ex(2, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 (11).ipynb", ex(2, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 (12).ipynb", ex(3, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 (13).ipynb", ex(3, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 10_1.ipynb", ex(3, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 11_1.ipynb", ex(3, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8_V9.ipynb", ex(3, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 12_1.ipynb", ex(4, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 Throughput.ipynb", ex(4, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 (14).ipynb", ex(5, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 (15).ipynb", ex(5, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 14_1.ipynb", ex(5, RUNS)),
    ("Research Experiment result", "experiment-1-arc-agi-2-with-qwen-3-8 12_2.ipynb", ex(5, RUNS)),
    # ---- other top-level folders
    ("Analysis Resource", "*", f"{R}/Analysis and Journals"),
    ("Research Paper", "*", f"{R}/Research Papers"),
    ("Document", "ARC_REPL_Strategy.pdf", f"{R}/Research Papers"),
    ("Document", "*.ipynb", f"{R}/Reference Notebooks"),
    ("review", "Review Folder", ex(10, RV + "/R5 safe-fix review")),
    ("review", "Validate output", ex(10, RS + "/R5 validate output")),
]
SOURCES = ["Qwen 3.8 27B Experiment", "Research Base Experiment", "Research Experiment result", "Analysis Resource",
           "Research Paper", "Document", "review"]

plan, unmatched = [], []
for src in SOURCES:
    sdir = ROOT / src
    if not sdir.exists():
        continue
    for item in sorted(sdir.iterdir()):
        dest = None
        for s, pat, d in RULES:
            if s == src and fnmatch.fnmatch(item.name, pat):
                dest = d
                break
        if dest is None:
            unmatched.append(item)
            dest = f"_Unsorted/{src}"
        plan.append((item, ROOT / dest / item.name))

for old, new in plan:
    print(f"{str(old.relative_to(ROOT)):95s} -> {new.relative_to(ROOT)}")
print(f"\n{len(plan)} items; unmatched -> _Unsorted: {[str(u.relative_to(ROOT)) for u in unmatched]}")
clash = [n for _, n in plan if n.exists()]
assert not clash, f"destination exists: {clash[:5]}"
dests = [str(n).lower() for _, n in plan]
assert len(dests) == len(set(dests)), "two items map to the same destination"

if EXECUTE:
    for n in NEW:                                  # the standard skeleton for every current experiment
        for sub in (NB, RV, RS):
            (ROOT / X / NEW[n] / sub).mkdir(parents=True, exist_ok=True)
    man = ROOT / "reorganization_manifest_2026-10-12.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_path", "new_path"])
        for old, new in plan:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
            w.writerow([str(old), str(new)])
    for src in SOURCES:                            # remove only folders that are now EMPTY
        sdir = ROOT / src
        if sdir.exists() and not any(sdir.iterdir()):
            sdir.rmdir()
    print("moved; manifest:", man)
