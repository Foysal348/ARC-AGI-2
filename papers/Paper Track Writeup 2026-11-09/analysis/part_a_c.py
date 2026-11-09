"""Parts A and C (NB5 W4 FP8 vs K100Q klee100) from attempts.json. Prints JSON to part_a_c.json."""
import json, math, os, statistics
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
att = json.load(open(os.path.join(HERE, "attempts.json")))
NB5 = ['1818057f', '45a5af55', 'bf45cf4b', '2ba387bc', '8f215267', '38007db0', '58f5dbd5', '247ef758', 'f931b4a8',
       'b5ca7ac4', '65b59efc', '221dfab4', '2b83f449', '16b78196', '64efde09']


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def right(x):
    return bool(x["test_ok_vs_gold"]) and all(x["test_ok_vs_gold"])


def tally(rows):
    sub = [x for x in rows if x["outcome"] == "ACCEPTED"]
    r = sum(right(x) for x in sub)
    partial = sum(1 for x in sub if not right(x) and any(x["test_ok_vs_gold"] or []))
    q_hits = sum(sum(x["test_ok_vs_gold"]) for x in sub)
    q_tot = sum(len(x["test_ok_vs_gold"]) for x in sub)
    return {"attempts": len(rows), "train_pass_submitted": len(sub), "right_all_tests": r, "wrong": len(sub) - r,
            "wrong_partial_some_tests_right": partial, "precision": round(r / len(sub), 4) if sub else None,
            "wilson95": wilson(r, len(sub)), "query_level_right": q_hits, "query_level_total": q_tot,
            "query_precision": round(q_hits / q_tot, 4), "query_wilson95": wilson(q_hits, q_tot),
            "run_flag_exact_accepted": sum(1 for x in sub if x["exact_accepted"]),
            "not_submitted": len(rows) - len(sub),
            "not_submitted_outcomes": sorted((x["task"], x["attempt"], x["outcome"], x["tokens"]) for x in rows if x["outcome"] != "ACCEPTED")}


res = {"A": {}, "C": {}}
for scope, filt in (("NB5_15_tasks", lambda x: x["task"] in NB5), ("all_available", lambda x: True)):
    w = tally([x for x in att if x["build"] == "W4" and filt(x)])
    k = tally([x for x in att if x["build"] == "K100" and filt(x)])
    tab = [[w["wrong"], w["right_all_tests"]], [k["wrong"], k["right_all_tests"]]]
    qt = [[w["query_level_total"] - w["query_level_right"], w["query_level_right"]],
          [k["query_level_total"] - k["query_level_right"], k["query_level_right"]]]
    res["A"][scope] = {"W4": w, "K100": k,
                       "fisher_wrong_rate_two_sided_p": round(fisher_exact(tab)[1], 4),
                       "fisher_wrong_rate_one_sided_p_K100_greater": round(fisher_exact(tab, alternative="less")[1], 4),
                       "fisher_query_level_two_sided_p": round(fisher_exact(qt)[1], 4), "table_wrong_right": tab}
    # sensitivity: run's own exact_accepted flag (counts W4 247ef758 a1 = right)
    wf = w["run_flag_exact_accepted"]; kf = k["run_flag_exact_accepted"]
    tabf = [[w["train_pass_submitted"] - wf, wf], [k["train_pass_submitted"] - kf, kf]]
    res["A"][scope]["run_flag_variant"] = {"W4_right": wf, "K100_right": kf, "table": tabf,
                                           "W4_wilson": wilson(wf, w["train_pass_submitted"]), "K100_wilson": wilson(kf, k["train_pass_submitted"]),
                                           "fisher_two_sided_p": round(fisher_exact(tabf)[1], 4),
                                           "fisher_one_sided_p": round(fisher_exact(tabf, alternative="less")[1], 4)}

# Part C
for b in ("W4", "K100"):
    for scope, filt in (("NB5_15_tasks", lambda x: x["task"] in NB5), ("all_available", lambda x: True)):
        A = [x for x in att if x["build"] == b and filt(x)]
        by = {(x["task"], x["attempt"]): x for x in A}
        a1_passed = [t for (t, a), x in by.items() if a == 1 and x["outcome"] == "ACCEPTED"]
        a1_right = [t for t in a1_passed if right(by[(t, 1)])]
        a2_after_pass = [by[(t, 2)] for t in a1_passed if (t, 2) in by]
        rescued = [x["task"] for x in a2_after_pass if right(x) and not right(by[(x["task"], 1)])]
        a2_after_right = [x for x in a2_after_pass if right(by[(x["task"], 1)])]
        tot = sum(x["tokens"] or 0 for x in A)
        never = [x for x in A if not x["accepted"] and not x["train_pass_seen"]]
        groups = {}
        for x in A:
            if x["attempt"] == 1 and x["outcome"] == "ACCEPTED" and right(x):
                groups.setdefault(x["group"], []).append(x["tokens"])
        tp_wrong = [x for x in A if x["outcome"] == "ACCEPTED" and not right(x)]
        res["C"][f"{b}_{scope}"] = {
            "attempts": len(A), "total_attempt_tokens": tot,
            "a1_train_pass_tasks": len(a1_passed), "a1_right_tasks": len(a1_right),
            "a2_on_tasks_a1_passed_train": {"n": len(a2_after_pass), "tokens": sum(x["tokens"] or 0 for x in a2_after_pass),
                                             "share_of_total": round(sum(x["tokens"] or 0 for x in a2_after_pass) / tot, 4),
                                             "a2_right": sum(right(x) for x in a2_after_pass), "tasks_rescued_by_a2": rescued},
            "a2_on_tasks_a1_right": {"n": len(a2_after_right), "tokens": sum(x["tokens"] or 0 for x in a2_after_right),
                                     "share_of_total": round(sum(x["tokens"] or 0 for x in a2_after_right) / tot, 4)},
            "never_passed_train": {"n": len(never), "tokens": sum(x["tokens"] or 0 for x in never),
                                   "share_of_total": round(sum(x["tokens"] or 0 for x in never) / tot, 4),
                                   "list": [(x["task"], x["attempt"], x["outcome"], x["tokens"]) for x in never]},
            "train_pass_wrong_submitted": {"n": len(tp_wrong), "tokens": sum(x["tokens"] for x in tp_wrong),
                                           "share_of_total": round(sum(x["tokens"] for x in tp_wrong) / tot, 4)},
            "median_tokens_right_a1_by_group": {g: {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)} for g, v in groups.items()},
        }

json.dump(res, open(os.path.join(HERE, "part_a_c.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
