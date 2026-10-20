"""Commit-run evaluation: the REPL agent on a fixed set of evaluation tasks, scored against the public solutions.

    python run_eval.py '<json config>'   (normally called by submit_main once the server is up)

Writes /kaggle/working/agent_eval/attempts.jsonl (one line per finished attempt, with transcript) and
eval_report.json (score, per task, tokens, turns, statuses). Never touches /kaggle/working/submission.json.
"""
import glob
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import ChatClient                      # noqa: E402
from repl_agent import solve_repl, AGENT_OPTS, breaker, ABORT   # noqa: E402
from agent import to_submission, score             # noqa: E402
from rescue import run_rescue, hybrid_submission, direct_submission, hits, load_direct_ref   # noqa: E402  (v1.6/1.8)


def find(name, root="/kaggle/input"):
    found = glob.glob(f"{root}/**/{name}", recursive=True)
    if not found:
        raise FileNotFoundError(name)
    return found[0]


def main(cfg):
    t0 = time.time()
    root = cfg.get("data_dir", "/kaggle/input")
    ch = json.load(open(find("arc-agi_evaluation_challenges.json", root)))
    sol = json.load(open(find("arc-agi_evaluation_solutions.json", root)))
    ids = [t for t in cfg["eval_task_ids"] if t in ch]
    tasks = {t: ch[t] for t in ids}
    out = Path(cfg.get("eval_out", "/kaggle/working/agent_eval"))
    out.mkdir(parents=True, exist_ok=True)
    client = ChatClient(cfg["base_url"], cfg["model"], max_tokens=cfg.get("max_tokens", 16384),
                        temperature=cfg.get("temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                        top_k=cfg.get("top_k", 20), thinking=cfg.get("thinking", True),
                        reasoning_effort=cfg.get("reasoning_effort"),
                        thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=cfg.get("template_kwargs"))
    lock = threading.Lock()
    jsonl = open(out / "attempts.jsonl", "w", encoding="utf-8")     # v1.1: fresh log per run
    progress = {"attempts_done": 0, "verified_tasks": set(), "gen_tokens": 0, "statuses": Counter()}

    def on_done(rec):
        breaker(rec)                                   # v1.6: stop early if the first attempts all die on HTTP errors
        with lock:
            jsonl.write(json.dumps(rec, default=str) + "\n")
            jsonl.flush()
            # v1.1: a progress snapshot after every attempt survives a deadline cut or a killed process
            progress["attempts_done"] += 1
            progress["gen_tokens"] += rec.get("gen_tokens", 0)
            progress["statuses"][rec.get("status")] += 1
            if rec.get("status") == "verified":
                progress["verified_tasks"].add(rec["task_id"])
            (out / "progress.json.tmp").write_text(json.dumps(
                {"attempts_done": progress["attempts_done"], "verified_tasks": sorted(progress["verified_tasks"]),
                 "gen_tokens": progress["gen_tokens"], "statuses": dict(progress["statuses"]),
                 "elapsed_s": round(time.time() - t0)}, indent=1))
            os.replace(out / "progress.json.tmp", out / "progress.json")     # v1.2: atomic

    # v1.6: the agent gets agent_frac of the time; the rest is for DIRECT rescue answers.
    # v1.8: eval_direct "ref" (default) merges with the LB-18.75 pipeline's recorded EVAL12 answers instead of
    # re-generating them (~1.1M tokens, ~70 GPU-min) -> the agent keeps the whole eval time; "live" re-runs DIRECT
    direct_mode = cfg.get("eval_direct", "ref") if cfg.get("rescue") else "off"
    agent_deadline = (t0 + (cfg["deadline_epoch"] - t0) * float(cfg.get("agent_frac", 0.6))
                      if direct_mode == "live" else cfg["deadline_epoch"])
    print(f"eval: {len(tasks)} tasks, agent deadline in {(agent_deadline - time.time()) / 60:.0f} min, "
          f"final deadline in {(cfg['deadline_epoch'] - time.time()) / 60:.0f} min", flush=True)
    answers, report = solve_repl(
        tasks, client, attempts=cfg.get("attempts", 2), concurrency=cfg.get("concurrency", 16),
        deadline=agent_deadline, stop_after=1, max_turns=cfg.get("max_turns", 40),
        max_gen_tokens=cfg.get("eval_cap", 60000), context_budget=cfg.get("context_budget", 28000),
        nudge_every=cfg.get("nudge_every", 0), turn_max_tokens=cfg.get("turn_max_tokens"),
        in_python=cfg.get("in_python", False), log=lambda m: print(m, flush=True), on_done=on_done,
        **{k: cfg[k] for k in AGENT_OPTS if k in cfg and k not in ("turn_max_tokens", "in_python")})
    jsonl.close()
    # v1.6 HYBRID: DIRECT answers for tasks the agent did not verify (rescue_all: every task, to measure overlap)
    atts0 = report["attempts"]
    verified0 = {a["task_id"] for a in atts0 if a["status"] == "verified"}
    rescue_recs, rescue_client = {}, None
    rescue_t0 = rescue_t1 = time.time()
    ref_path = Path(__file__).with_name("direct_ref_eval12.json")
    ref_run4 = {}
    if direct_mode == "ref":
        ref = load_direct_ref(ref_path, cfg.get("eval_direct_run", "run7"))
        rescue_recs = {t: ref[t] for t in ids if t in ref and (cfg.get("rescue_all") or t not in verified0)}
        ref_run4 = {t: r for t, r in load_direct_ref(ref_path, "run4").items() if t in ids}
        print(f"DIRECT from the LB-18.75 run records ({cfg.get('eval_direct_run', 'run7')}): {len(rescue_recs)} tasks",
              flush=True)
    elif direct_mode == "live":
        ids_r = ids if cfg.get("rescue_all") else [t for t in ids if t not in verified0]
        print(f"rescue (direct answers): {len(ids_r)} tasks, {(cfg['deadline_epoch'] - time.time()) / 60:.0f} min left",
              flush=True)
        rjsonl = open(out / "rescue_attempts.jsonl", "w", encoding="utf-8")
        done_r = {}

        def on_rescue(tid, rec):                       # v1.7: durable record + deployed snapshot after every answer
            rjsonl.write(json.dumps(dict(rec, task_id=tid), default=str) + "\n")
            rjsonl.flush()
            done_r[tid] = rec
            snap = hybrid_submission(answers, {t: r for t, r in done_r.items() if t not in verified0}, tasks)
            (out / "eval_submission.partial.json.tmp").write_text(json.dumps(snap))
            os.replace(out / "eval_submission.partial.json.tmp", out / "eval_submission.partial.json")

        rescue_t0 = time.time()
        rescue_recs, rescue_client = run_rescue(ids_r, tasks, cfg, cfg["deadline_epoch"],
                                                log=lambda m: print(m, flush=True), on_result=on_rescue)
        rescue_t1 = time.time()
        rjsonl.close()
    sol_ids = {t: sol[t] for t in ids}
    agent_sub = to_submission(answers, tasks)
    direct_sub = direct_submission(rescue_recs, tasks)
    # v1.7 (GPT final audit): the DEPLOYED policy rescues only tasks the agent did not verify -- that is the score that
    # predicts the hidden run; the all-tasks-rescued merge is a counterfactual reported beside it
    deployed_rescue = {t: r for t, r in rescue_recs.items() if t not in verified0}
    sub = hybrid_submission(answers, deployed_rescue, tasks)
    all_sub = hybrid_submission(answers, rescue_recs, tasks)
    h_agent, h_direct, h_hybrid, h_all = (hits(x, sol_ids, ids) for x in (agent_sub, direct_sub, sub, all_sub))
    rescued_q = [(t, j) for t in rescue_recs for j in range(len(sol[t]))]
    r_status = Counter(str(u.get("status")).split(":")[0] for r in rescue_recs.values()
                       for u in (r.get("units") or {}).values())             # v1.8: per test input
    complementarity = {
        "queries": len(h_agent),
        "agent_correct": sum(h_agent.values()), "direct_correct": sum(h_direct.values()),
        "hybrid_correct": sum(h_hybrid.values()), "hybrid_all_rescued_correct": sum(h_all.values()),
        "agent_only": sum(1 for k in h_agent if h_agent[k] and not h_direct[k]),
        "direct_only": sum(1 for k in h_agent if h_direct[k] and not h_agent[k]),
        "both": sum(1 for k in h_agent if h_agent[k] and h_direct[k]),
        "oracle_union_hits": sum(1 for k in h_agent if h_agent[k] or h_direct[k]),   # retrospective bound, not a policy
        "hybrid_lost_vs_agent": sum(1 for k in h_agent if h_agent[k] and not h_hybrid[k]),
        "hybrid_lost_vs_direct": sum(1 for k in h_agent if h_direct[k] and not h_hybrid[k]),
        "all_rescued_lost_vs_agent": sum(1 for k in h_agent if h_agent[k] and not h_all[k]),
        "all_rescued_lost_vs_direct": sum(1 for k in h_agent if h_direct[k] and not h_all[k]),
        "direct_correct_on_unverified_queries": sum(h_direct[k] for k in h_direct if k[0] not in verified0),
        "direct_correct_on_rescued_queries": sum(h_direct[k] for k in rescued_q),
        "rescued_queries": len(rescued_q),
        "rescue_statuses": dict(r_status),
        "rescue_gen_tokens": sum(r.get("gen_tokens", 0) for r in rescue_recs.values()),
        "rescue_budget_dropped": bool(getattr(rescue_client, "budget_dropped", False)),
        "rescue_smoke_ok": cfg.get("rescue_smoke_ok"),
        "direct_source": {"ref": {"run7": "recorded: Q27-XHIGH v1.4.4 (LB 26.94), run 7",
                                  "run4": "recorded: Q27-XHIGH v1.4 (LB 18.75), run 4"}.get(
                                      str(cfg.get("eval_direct_run", "run7")), str(cfg.get("eval_direct_run"))),
                          "live": "generated in this run", "off": "none"}[direct_mode],
        # v1.7: the three-way comparison is only complete if DIRECT really answered (or tried) every task
        "three_way_complete": bool(direct_mode != "off" and cfg.get("rescue_all")
                                   and (direct_mode == "ref" or cfg.get("rescue_smoke_ok", True))
                                   and len(rescue_recs) == len(ids)
                                   and not any(s in r_status for s in ("skipped", "crashed", "error"))),
    }
    # v1.8.2 (GPT audit): the decision metric for the hidden-run split -- net exact-output gain of the deployed HYBRID
    # over DIRECT alone, minus the DIRECT hits the agent's tokens would have displaced (DIRECT tokens per query and
    # hit rate from the same reference)
    _units = [u for r in rescue_recs.values() for u in (r.get("units") or {}).values()]
    _dtok = [int(u.get("think_tokens") or u.get("gen_tokens") or 0) for u in _units]
    _dtok_q = (sum(_dtok) / len(_dtok)) if _dtok and sum(_dtok) else None
    _agent_tok = sum(a["gen_tokens"] for a in report["attempts"])
    _q_n = max(len(h_agent), 1)
    _disp_q = (_agent_tok / _dtok_q) if _dtok_q else None
    _disp_hits = (_disp_q * complementarity["direct_correct"] / _q_n) if _disp_q is not None else None
    complementarity["net_gain"] = {
        "hybrid_minus_direct_hits": complementarity["hybrid_correct"] - complementarity["direct_correct"],
        "agent_tokens": _agent_tok, "direct_tokens_per_query": round(_dtok_q) if _dtok_q else None,
        "displaced_direct_queries_equiv": round(_disp_q, 2) if _disp_q is not None else None,
        "expected_displaced_direct_hits": round(_disp_hits, 3) if _disp_hits is not None else None,
        "net_after_displacement": (round(complementarity["hybrid_correct"] - complementarity["direct_correct"]
                                         - _disp_hits, 3) if _disp_hits is not None else None)}
    if ref_run4:                                       # v1.8: the same merge against the other recorded DIRECT run
        h4 = hits(direct_submission(ref_run4, tasks), sol_ids, ids)
        h4h = hits(hybrid_submission(answers, {t: r for t, r in ref_run4.items() if t not in verified0}, tasks),
                   sol_ids, ids)
        complementarity["reference_run4"] = {"direct_correct": sum(h4.values()), "hybrid_correct": sum(h4h.values()),
                                             "agent_only_vs_run4": sum(1 for k in h_agent if h_agent[k] and not h4[k])}
    total, pct, per = score(sub, sol_ids)
    _, _, agent_per = score(agent_sub, sol_ids)
    q_tot = len(h_hybrid)
    q_hit = sum(h_hybrid.values())
    atts = report["attempts"]
    agent_gen = sum(a["gen_tokens"] for a in atts)
    agent_wall = float(report.get("wall_s") or (rescue_t0 - t0))
    rescue_gen = complementarity["rescue_gen_tokens"]
    rescue_wall = max(0.0, rescue_t1 - rescue_t0) if direct_mode == "live" else 0.0
    if direct_mode != "live":
        rescue_gen = 0                                 # reference tokens were spent in runs 4 / 7, not here
    wall = time.time() - t0
    official_pct = 100.0 * q_hit / q_tot if q_tot else 0.0
    verified_tasks = sorted({a["task_id"] for a in atts if a["status"] == "verified"})
    scout_cap = (cfg.get("rerun_attempt_caps") or cfg.get("attempt_caps") or [0])[0]
    # v1.2: generated code runs with the real filesystem, where the evaluation SOLUTIONS file exists. Flag every task
    # whose executed code touched files/paths so a score cannot silently come from reading answers.
    # v1.3: wider net (audit R3 read a file via builtins) -- this is DETECTION for manual review, not a sandbox
    leak_re = re.compile(r"solution|/kaggle/|open\s*\(|listdir|scandir|glob|pathlib|Path\s*\(|walk\s*\(|builtins|"
                         r"__import__|importlib|\bio\b|\bos\b|\bsys\b|subprocess|shutil|socket|eval\s*\(|exec\s*\(|"
                         r"compile\s*\(|getattr\s*\(|globals\s*\(|locals\s*\(|vars\s*\(|ctypes|cffi|mmap", re.I)
    suspicious = sorted({a["task_id"] for a in atts for x in a.get("transcript", [])
                         if isinstance(x, dict) and isinstance(x.get("code"), str) and leak_re.search(x["code"])})
    tdump = {id(a): json.dumps(a.get("transcript", []), default=str) for a in atts}
    summary = {
        # v1.7 (GPT final audit): from the AGENT-only submission -- a DIRECT rescue must not hide a wrong verified program
        "verified_but_wrong_tasks": [t for t in verified_tasks if agent_per.get(t, 0) < 1],   # train-fit != test-correct
        "verified_but_wrong_queries": [f"{t}#{j}" for t in verified_tasks for j in range(len(sol[t]))
                                       if not h_agent[(t, j)]],
        "file_access_suspicious_tasks": suspicious,
        # v1.7: what the guard actually stopped, and children that refused to run because the guard failed
        "isolation_denied_tasks": sorted({a["task_id"] for a in atts if "[sandbox] access denied" in tdump[id(a)]
                                          or "[sandbox] not allowed" in tdump[id(a)]}),
        "isolation_guard_failed_tasks": sorted({a["task_id"] for a in atts if "[sandbox] guard failed" in tdump[id(a)]}),
        "thinking_budget_dropped_at_runtime": bool(getattr(client, "budget_dropped", False)),   # v1.3
        # v1.1: Kaggle's metric = correct test outputs / all test outputs (micro); the task average is kept beside it
        # v1.7: official = the DEPLOYED hybrid (rescue only where the agent did not verify)
        "official_score_pct": round(official_pct, 2), "task_macro_score_pct": round(pct, 2),
        "hybrid_all_rescued_pct": round(100.0 * complementarity["hybrid_all_rescued_correct"] / max(q_tot, 1), 2),
        "agent_only_official_pct": round(100.0 * complementarity["agent_correct"] / max(q_tot, 1), 2),
        "direct_only_official_pct": round(100.0 * complementarity["direct_correct"] / max(q_tot, 1), 2),
        "complementarity": complementarity,
        "circuit_breaker": dict(ABORT),
        "agent_deadline_epoch": agent_deadline,
        "score_tasks": round(total, 3), "score_pct": round(pct, 2), "n_tasks": len(ids),
        "query_hits": q_hit, "queries": q_tot, "per_task": per, "per_task_agent_only": agent_per,
        "tasks_with_verified_program": len(verified_tasks),
        # v1.7: how many verifications the hidden run's smaller scout cap would have kept (counterfactual)
        "verified_first_attempt_within_rerun_scout_cap": {"cap": scout_cap, "tasks": sorted(
            {a["task_id"] for a in atts if a["status"] == "verified" and a["idx"] == 0
             and a["gen_tokens"] <= scout_cap})},
        "status_counts": dict(Counter(a["status"] for a in atts)),
        "notes": dict(Counter(a["note"] for a in atts if a["note"])),
        # v1.7 (GPT final audit): throughput per stage (the old figure divided agent tokens by agent + rescue time)
        "agent_generated_tokens": agent_gen, "agent_wall_s": round(agent_wall, 1),
        "agent_gen_tok_per_s": round(agent_gen / max(agent_wall, 1e-6), 1),
        "rescue_generated_tokens": rescue_gen, "rescue_wall_s": round(rescue_wall, 1),
        "rescue_gen_tok_per_s": round(rescue_gen / max(rescue_wall, 1e-6), 1),
        "combined_gen_tok_per_s": round((agent_gen + rescue_gen) / max(wall, 1e-6), 1),
        "gen_tokens_total": agent_gen + rescue_gen, "gen_tokens_mean_per_attempt": round(agent_gen / max(len(atts), 1)),
        "turns_mean": round(sum(a["turns"] for a in atts) / max(len(atts), 1), 1),
        "cuts_total": sum(a["cuts"] for a in atts), "submits_total": sum(a["submits"] for a in atts),
        "python_calls_total": sum(a["python_calls"] for a in atts),
        "wall_s": round(wall),
        "config": {k: v for k, v in cfg.items() if k not in ("base_url",)},
        "per_attempt": [{k: a[k] for k in ("task_id", "idx", "status", "note", "turns", "python_calls", "submits",
                                           "cuts", "forced", "gen_tokens", "prompt_tokens", "wall", "best_ok",
                                           "n_train", "effort_steps", "evictions")} for a in atts]}
    (out / "eval_report.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "eval_submission.json").write_text(json.dumps(sub))                     # the deployed policy
    (out / "eval_all_rescued_submission.json").write_text(json.dumps(all_sub))     # counterfactual
    print(f"\nscores (official = deployed hybrid): hybrid {official_pct:.1f}% | all-rescued "
          f"{summary['hybrid_all_rescued_pct']}% | agent {summary['agent_only_official_pct']}% | direct "
          f"{summary['direct_only_official_pct']}% | {complementarity}", flush=True)
    print(f"\nEVAL OFFICIAL {official_pct:.1f}% = queries {q_hit}/{q_tot} | task average {total:.2f} / {len(ids)} "
          f"({pct:.1f}%) | verified tasks {summary['tasks_with_verified_program']} "
          f"(wrong on test: {summary['verified_but_wrong_tasks']}) | agent {agent_gen} tok "
          f"({summary['agent_gen_tok_per_s']} tok/s) | rescue {rescue_gen} tok ({summary['rescue_gen_tok_per_s']} tok/s) "
          f"| wall {wall / 60:.0f} min | {summary['status_counts']}", flush=True)
    print("per task:", {t: round(v, 2) for t, v in per.items()}, flush=True)

if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
