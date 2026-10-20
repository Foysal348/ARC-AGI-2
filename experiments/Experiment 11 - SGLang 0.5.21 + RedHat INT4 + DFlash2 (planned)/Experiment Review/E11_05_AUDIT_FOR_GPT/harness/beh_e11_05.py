"""Shared fixture for the E11-05 harness: the task pick, per-task behaviours and the answer code (= harness_s1)."""
import json
from pathlib import Path

DATA = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
            r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
TOKDIR_INT4 = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/tok")


def load():
    ch = json.load(open(DATA / "arc-agi_evaluation_challenges.json", encoding="utf-8"))
    sol = json.load(open(DATA / "arc-agi_evaluation_solutions.json", encoding="utf-8"))
    multi = sorted(t for t in ch if len(ch[t]["test"]) >= 2)[:3]
    single = sorted(t for t in ch if len(ch[t]["test"]) == 1)[:5]
    return ch, sol, multi, single


def beh_map(multi, single, scen):
    b = {multi[0]: "partial_then_fail", multi[1]: "forced_then_pass", multi[2]: "fail_distinct",
         single[0]: "pass", single[1]: "illustrative_first", single[2]: "fail_same", single[3]: "pass", single[4]: "pass"}
    if scen == "drain":                      # phase 1 never closes and is still streaming at the drain
        b[single[0]] = "slow_pass"; b[single[3]] = "slow_pass"
    return b


def answer_code(b, it):
    if b in ("pass", "slow_pass"):
        return "```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```"
    if b == "partial_then_fail":
        return "```python\ndef transform(grid):\n    return grid  # MARK:PASS_PARTIAL\n```" if it == 0 else \
               f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```"
    if b == "forced_then_pass":
        return (f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```" if it == 0 else
                "```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```")
    if b == "fail_distinct":
        return f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```"
    if b == "illustrative_first":
        return ("For example:\n```python\nx = helper(grid)  # MARK:ZERO no transform here\n```\n"
                "First idea:\n```python\ndef transform(grid):\n    return grid  # MARK:ZERO\n```\n"
                "Final:\n```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```")
    return "```python\ndef transform(grid):\n    return grid  # MARK:ZERO\n```"


def parse_rid(rid):
    tid, rest = rid.split("~e", 1)
    it = int(rest.split("~i")[1].split("|")[0])
    return tid, it, int(rid[-1])
