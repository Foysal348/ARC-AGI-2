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
from repl_agent import solve_repl, AGENT_OPTS      # noqa: E402
from agent import to_submission, score             # noqa: E402


def find(name, root="/kaggle/input"):
    hits = glob.glob(f"{root}/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


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

    print(f"eval: {len(tasks)} tasks, deadline in {(cfg['deadline_epoch'] - time.time()) / 60:.0f} min", flush=True)
    answers, report = solve_repl(
        tasks, client, attempts=cfg.get("attempts", 2), concurrency=cfg.get("concurrency", 16),
        deadline=cfg["deadline_epoch"], stop_after=1, max_turns=cfg.get("max_turns", 40),
        max_gen_tokens=cfg.get("eval_cap", 60000), context_budget=cfg.get("context_budget", 28000),
        nudge_every=cfg.get("nudge_every", 0), turn_max_tokens=cfg.get("turn_max_tokens"),
        in_python=cfg.get("in_python", False), log=lambda m: print(m, flush=True), on_done=on_done,
        **{k: cfg[k] for k in AGENT_OPTS if k in cfg and k not in ("turn_max_tokens", "in_python")})
    jsonl.close()
    sub = to_submission(answers, tasks)
    total, pct, per = score(sub, {t: sol[t] for t in ids})
    q_hit = q_tot = 0
    for t in ids:
        for j, s in enumerate(sol[t]):
            q_tot += 1
            q_hit += int(s in (sub[t][j]["attempt_1"], sub[t][j]["attempt_2"]))
    atts = report["attempts"]
    gen = sum(a["gen_tokens"] for a in atts)
    wall = time.time() - t0
    official_pct = 100.0 * q_hit / q_tot if q_tot else 0.0
    verified_tasks = sorted({a["task_id"] for a in atts if a["status"] == "verified"})
    # v1.2: generated code runs with the real filesystem, where the evaluation SOLUTIONS file exists. Flag every task
    # whose executed code touched files/paths so a score cannot silently come from reading answers.
    # v1.3: wider net (audit R3 read a file via builtins) -- this is DETECTION for manual review, not a sandbox
    leak_re = re.compile(r"solution|/kaggle/|open\s*\(|listdir|scandir|glob|pathlib|Path\s*\(|walk\s*\(|builtins|"
                         r"__import__|importlib|\bio\b|\bos\b|\bsys\b|subprocess|shutil|socket|eval\s*\(|exec\s*\(|"
                         r"compile\s*\(|getattr\s*\(|globals\s*\(|locals\s*\(|vars\s*\(", re.I)
    suspicious = sorted({a["task_id"] for a in atts for x in a.get("transcript", [])
                         if isinstance(x, dict) and isinstance(x.get("code"), str) and leak_re.search(x["code"])})
    summary = {
        "verified_but_wrong_tasks": [t for t in verified_tasks if per.get(t, 0) < 1],   # train-fit != test-correct
        "file_access_suspicious_tasks": suspicious,
        "thinking_budget_dropped_at_runtime": bool(getattr(client, "budget_dropped", False)),   # v1.3
        # v1.1: Kaggle's metric = correct test outputs / all test outputs (micro); the task average is kept beside it
        "official_score_pct": round(official_pct, 2), "task_macro_score_pct": round(pct, 2),
        "score_tasks": round(total, 3), "score_pct": round(pct, 2), "n_tasks": len(ids),
        "query_hits": q_hit, "queries": q_tot, "per_task": per,
        "tasks_with_verified_program": sum(1 for t in ids if any(a["status"] == "verified" and a["task_id"] == t
                                                                  for a in atts)),
        "status_counts": dict(Counter(a["status"] for a in atts)),
        "notes": dict(Counter(a["note"] for a in atts if a["note"])),
        "gen_tokens_total": gen, "gen_tokens_mean_per_attempt": round(gen / max(len(atts), 1)),
        "turns_mean": round(sum(a["turns"] for a in atts) / max(len(atts), 1), 1),
        "cuts_total": sum(a["cuts"] for a in atts), "submits_total": sum(a["submits"] for a in atts),
        "python_calls_total": sum(a["python_calls"] for a in atts),
        "wall_s": round(wall), "gen_tok_per_s": round(gen / max(wall, 1e-6), 1),
        "config": {k: v for k, v in cfg.items() if k not in ("base_url",)},
        "per_attempt": [{k: a[k] for k in ("task_id", "idx", "status", "note", "turns", "python_calls", "submits",
                                           "cuts", "forced", "gen_tokens", "prompt_tokens", "wall", "best_ok",
                                           "n_train", "effort_steps", "evictions")} for a in atts]}
    (out / "eval_report.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "eval_submission.json").write_text(json.dumps(sub))
    print(f"\nEVAL OFFICIAL {official_pct:.1f}% = queries {q_hit}/{q_tot} | task average {total:.2f} / {len(ids)} "
          f"({pct:.1f}%) | "
          f"verified tasks {summary['tasks_with_verified_program']} | tokens {gen} ({summary['gen_tok_per_s']} tok/s) | "
          f"wall {wall / 60:.0f} min | {summary['status_counts']}", flush=True)
    print("per task:", {t: round(v, 2) for t, v in per.items()}, flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
