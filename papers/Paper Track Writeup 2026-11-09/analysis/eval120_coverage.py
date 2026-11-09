"""Part C (EVAL120 6-h runs): coverage, solve-vs-tokens/time curves, queue positions, token waste.

Read-only sources (relative to D:\\ARC AGI 2\\_scratch):
  W4  : e28_w4_out/q27-xhigh-e28-w4-eval120-6h/Q27X_E21_NB5_HALFEVAL_DUCK_BUNDLE.zip
          (duck/<task>_b<k>/branch_result.json, e12_run_report.json, e12_generated_attempts.json)
        e28_w4_out/q27-xhigh-e28-w4-eval120-6h/q27x/e28_sched_report.json, q27x/nb5_report.json
  K100: e28_k100_out/q27-xhigh-e28-k100-eval120-6h/Q27X_E21_NB5_HALFQ_E28_K100_EVAL120_DUCK_BUNDLE.zip
        e28_k100_out/q27-xhigh-e28-k100-eval120-6h/q27x/e28_k100_report.json, q27x/nb5_report.json
  gold: experiments/Experiment 10 .../E10_ONE_THINKING_TWO_ATTEMPTS_FOR_ASTRA/data/arc-agi_evaluation_solutions.json

Definitions
  window origin   = min(t0) over all branches (the first attempt starts at sched start_sec 0.0).
  attempt answer  = last accepted submission of the branch; if none, the last host-checked candidate that is
                    exact on all train pairs (what the answer-slot code calls an "r5" harvest); time = its 't'.
  attempt correct = answer exact on ALL test inputs vs gold.  query hit = per test input.
  solved-by-time  = tasks with >= 1 correct attempt answer whose time <= T (pass@any over attempts so far; the
                    final 2-slot submission is scored separately from submission.json).
  cumulative tokens = sum of per-request completion tokens ('usage' list of every branch) with usage t <= T.
  never passed train = branch with no accepted submission and no train-exact checked candidate.
"""
import json, os, zipfile, re
from collections import defaultdict

ROOT = r"D:\ARC AGI 2"
SCR = os.path.join(ROOT, "_scratch")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(ROOT, r"experiments\Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)\Experiment Review\E10_ONE_THINKING_TWO_ATTEMPTS_FOR_ASTRA\data\arc-agi_evaluation_solutions.json")
sol = json.load(open(GOLD))

RUNS = {
    "W4": {"zip": r"e28_w4_out\q27-xhigh-e28-w4-eval120-6h\Q27X_E21_NB5_HALFEVAL_DUCK_BUNDLE.zip",
           "sched": r"e28_w4_out\q27-xhigh-e28-w4-eval120-6h\q27x\e28_sched_report.json",
           "nb5": r"e28_w4_out\q27-xhigh-e28-w4-eval120-6h\q27x\nb5_report.json"},
    "K100": {"zip": r"e28_k100_out\q27-xhigh-e28-k100-eval120-6h\Q27X_E21_NB5_HALFQ_E28_K100_EVAL120_DUCK_BUNDLE.zip",
             "sched": r"e28_k100_out\q27-xhigh-e28-k100-eval120-6h\q27x\e28_k100_report.json",
             "nb5": r"e28_k100_out\q27-xhigh-e28-k100-eval120-6h\q27x\nb5_report.json"},
}


def exact(outs, gold):
    return [i < len(outs or []) and outs[i] == g for i, g in enumerate(gold)]


def analyse(name, cfg):
    z = zipfile.ZipFile(os.path.join(SCR, cfg["zip"]))
    names = z.namelist()
    rr = json.loads(z.read("e12_run_report.json"))
    sub = json.loads(z.read("e12_generated_attempts.json"))  # eval-task answer slots (submission.json is the 240-id private placeholder)
    sched = json.load(open(os.path.join(SCR, cfg["sched"])))
    order = rr["order"]
    pos = {t: i + 1 for i, t in enumerate(order)}
    # scheduler rows (phase, kind, end_reason)
    if name == "W4":
        srows = sched["e28"]["attempts"]
        ssum = sched["summary"]
    else:
        srows = sched["sched_attempts"]
        ssum = None
    srow = {(r["task"], r["branch"]): r for r in srows}

    br = {}
    for n in names:
        m = re.match(r"duck/([0-9a-f]{8})_b(\d+)/branch_result.json$", n)
        if m:
            br[(m.group(1), int(m.group(2)))] = json.loads(z.read(n))
    t_origin = min(b["t0"] for b in br.values())

    usage = []  # (t_rel, tokens)
    att = []
    for (task, k), b in sorted(br.items()):
        for u in b.get("usage", []):
            usage.append((u["t"] - t_origin, u["completion"]))
        phase = None
        for r in b.get("runs", []):
            phase = r.get("e28_phase", phase)
        sr = srow.get((task, k), {})
        phase = phase or sr.get("phase")
        accs = [s for s in b.get("submissions", []) if s.get("accepted")]
        trainpass = [s for s in (b.get("checked") or []) + (b.get("submissions") or [])
                     if s.get("train_exact") and all(s["train_exact"])]
        if accs:
            ans, src = accs[-1], "submission"
        elif trainpass:
            ans, src = max(trainpass, key=lambda s: s["t"]), "train_pass_checked"
        else:
            ans, src = None, None
        ok = exact(ans["test_outputs"], sol[task]) if ans else None
        last_u = max((u["t"] for u in b.get("usage", [])), default=b["t0"]) - t_origin
        first_tp = min((s["t"] for s in trainpass), default=None)
        att.append({
            "task": task, "branch": k, "phase": phase, "queue_pos": pos[task],
            "kind": sr.get("kind"), "end_reason": sr.get("end_reason"),
            "tokens": b.get("generated_tokens"), "start_rel": round(b["t0"] - t_origin, 1),
            "last_usage_rel": round(last_u, 1),
            "accepted": bool(accs), "train_pass": bool(trainpass),
            "first_train_pass_rel": round(first_tp - t_origin, 1) if first_tp else None,
            "first_train_pass_tokens": min((s["tokens"] for s in trainpass), default=None),
            "answer_src": src, "answer_rel": round(ans["t"] - t_origin, 1) if ans else None,
            "answer_tokens": ans.get("tokens") if ans else None,
            "test_ok": ok, "correct": bool(ok) and all(ok),
        })
    usage.sort()

    # final submission score vs gold
    fin_hits = fin_q = 0
    fin_solved = 0
    for t, g in sol.items():
        if t not in pos:
            continue
        preds = sub.get(t, [])
        hits = 0
        for i, gg in enumerate(g):
            p = preds[i] if i < len(preds) else {}
            if p.get("attempt_1") == gg or p.get("attempt_2") == gg:
                hits += 1
        fin_hits += hits
        fin_q += len(g)
        fin_solved += hits == len(g)
    final = {"tasks": len(order), "query_hits": fin_hits, "queries": fin_q, "tasks_all_queries": fin_solved,
             "task_norm_pct": round(100 * sum(
                 sum(1 for i, gg in enumerate(sol[t]) if i < len(sub.get(t, [])) and (sub[t][i].get("attempt_1") == gg or sub[t][i].get("attempt_2") == gg)) / len(sol[t])
                 for t in order) / len(order), 2)}

    def cum_tokens(T):
        return sum(c for t, c in usage if t <= T)

    def solved_by(T):
        best = defaultdict(lambda: None)
        tasks = set()
        qh = {}
        for a in att:
            if a["answer_rel"] is not None and a["answer_rel"] <= T and a["test_ok"]:
                if a["correct"]:
                    tasks.add(a["task"])
                cur = qh.get(a["task"], [False] * len(a["test_ok"]))
                qh[a["task"]] = [x or y for x, y in zip(cur, a["test_ok"])]
        return len(tasks), sum(sum(v) for v in qh.values())

    def started_by(T):
        return len({a["task"] for a in att if a["start_rel"] <= T and a["phase"] == "A"})

    horizon = max(max(a["last_usage_rel"] for a in att), 21600)
    time_rows = []
    for T in list(range(1800, 21601, 1800)):
        s, q = solved_by(T)
        time_rows.append({"t_sec": T, "t_h": round(T / 3600, 2), "cum_tokens": cum_tokens(T), "tasks_solved": s,
                          "query_hits": q, "tasks_started_A": started_by(T)})
    total_tok = cum_tokens(horizon + 1)
    tok_rows = []
    step = 500000
    for X in range(step, total_tok + step, step):
        X = min(X, total_tok)
        T = next((t for t, _ in [(u[0], 0) for u in usage] if cum_tokens(t) >= X), horizon)
        # faster: walk usage
        tok_rows.append({"cum_tokens": X})
    # walk usage once for token checkpoints
    tok_rows = []
    acc_tok, k = 0, 1
    for t, c in usage:
        acc_tok += c
        while acc_tok >= k * step:
            s, q = solved_by(t)
            tok_rows.append({"cum_tokens": k * step, "t_sec": round(t, 0), "tasks_solved": s, "query_hits": q})
            k += 1
    s, q = solved_by(horizon + 1)
    tok_rows.append({"cum_tokens": acc_tok, "t_sec": round(usage[-1][0], 0), "tasks_solved": s, "query_hits": q, "note": "end"})

    # coverage / queue
    startedA = sorted({a["task"] for a in att if a["phase"] == "A"}, key=pos.get)
    notstartedA = [t for t in order if t not in startedA]
    unfinished = [a for a in att if a["end_reason"] in ("running_at_window_end",) or a["kind"] == "TIME"]
    by_phase = defaultdict(lambda: {"started": 0, "tokens": 0, "correct": 0, "accepted": 0, "train_pass": 0})
    for a in att:
        p = by_phase[a["phase"]]
        p["started"] += 1
        p["tokens"] += a["tokens"] or 0
        p["correct"] += a["correct"]
        p["accepted"] += a["accepted"]
        p["train_pass"] += a["train_pass"]

    # tokens on a later attempt (branch >= 1) of tasks whose branch-0 attempt passed train
    a0 = {a["task"]: a for a in att if a["branch"] == 0}
    later_after_pass = [a for a in att if a["branch"] >= 1 and a0.get(a["task"], {}).get("train_pass")]
    later_after_correct = [a for a in later_after_pass if a0[a["task"]]["correct"]]
    rescued = sorted({a["task"] for a in att if a["branch"] >= 1 and a["correct"] and not a0.get(a["task"], {}).get("correct")})
    never_tp = [a for a in att if not a["train_pass"]]
    tp_wrong = [a for a in att if a["train_pass"] and a["answer_src"] and not a["correct"]]
    total_counted = sum(a["tokens"] or 0 for a in att)

    out = {
        "run": name, "sources": cfg, "window_origin_epoch": t_origin,
        "window": {"last_usage_rel_sec": round(max(a["last_usage_rel"] for a in att), 1)},
        "final_submission_vs_gold": final,
        "tasks_in_order": len(order),
        "phase_A_started": len(startedA), "phase_A_not_started": len(notstartedA),
        "phase_A_not_started_queue_positions": [pos[t] for t in notstartedA],
        "phase_A_last_started_queue_pos": max(pos[t] for t in startedA),
        "unfinished_at_window_end": [{"task": a["task"], "branch": a["branch"], "phase": a["phase"], "queue_pos": a["queue_pos"],
                                      "tokens": a["tokens"], "train_pass": a["train_pass"], "correct": a["correct"]} for a in unfinished],
        "by_phase": dict(by_phase),
        "counted_tokens_all_attempts": total_counted,
        "later_attempts_after_train_pass_a1": {"n": len(later_after_pass), "tokens": sum(a["tokens"] or 0 for a in later_after_pass),
                                               "correct": sum(a["correct"] for a in later_after_pass),
                                               "of_which_a1_correct_n": len(later_after_correct),
                                               "of_which_a1_correct_tokens": sum(a["tokens"] or 0 for a in later_after_correct)},
        "later_attempt_rescues": rescued,
        "never_passed_train": {"n": len(never_tp), "tokens": sum(a["tokens"] or 0 for a in never_tp),
                               "by_phase": {p: sum(1 for a in never_tp if a["phase"] == p) for p in sorted({a["phase"] for a in never_tp})}},
        "train_pass_but_wrong": {"n": len(tp_wrong), "tokens": sum(a["tokens"] or 0 for a in tp_wrong)},
        "solved_vs_time": time_rows, "solved_vs_tokens": tok_rows,
        "attempts": att,
    }
    if ssum:
        out["sched_summary"] = {k: ssum[k] for k in ("started", "done", "queued", "left", "a_class_counts", "phases", "preemptions")}
    return out


res = {n: analyse(n, c) for n, c in RUNS.items()}
for n, c in RUNS.items():
    nb5 = json.load(open(os.path.join(SCR, c["nb5"])))
    ph = list(nb5["phases"].values())[0]
    res[n]["window"].update({"phase_window": ph.get("phase_window"), "solver_window": ph.get("solver_window"),
                             "window_plan_sec": (ph.get("window_plan") or {}).get("planned_sec"),
                             "running_check": ph.get("running_check"), "preemptions": (ph.get("preemption_gate") or {}).get("preemptions"),
                             "score_report": ph.get("score")})
json.dump(res, open(os.path.join(HERE, "eval120_coverage.json"), "w"), indent=1)
for n, r in res.items():
    print("==", n)
    for k, v in r.items():
        if k in ("attempts", "sources"):
            continue
        print(k, json.dumps(v)[:1500])
