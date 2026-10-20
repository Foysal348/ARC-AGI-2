"""Competition run: the REPL agent on the hidden test set, writing submission.json.

    python run_submit.py '<json config>'

Safety first, because a crashed rerun wastes a day's submission:
  * the notebook writes a valid FALLBACK submission (each test input copied into
    both attempts) before anything heavy runs, via write_fallback();
  * as every attempt finishes, the submission is rewritten atomically with the
    best answers so far;
  * the agent stops at a hard deadline well inside the 12 h limit.

Budget: the per-attempt token cap is the estimated token supply (throughput x
time) divided by the number of hidden tasks, clamped -- the task count is only
known at run time (the public placeholder has 240).
"""
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import ChatClient                      # noqa: E402
from repl_agent import solve_repl, AGENT_OPTS, breaker   # noqa: E402

SUB = "/kaggle/working/submission.json"


def find(name, root="/kaggle/input"):
    hits = glob.glob(f"{root}/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def write_atomic(obj, path=SUB):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def write_fallback(challenges, path=SUB):
    """A valid submission with no model at all: every attempt is the test input."""
    write_atomic({t: [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in task["test"]]
                  for t, task in challenges.items()}, path)


def ranked_answers(records, task):
    """Per test input: candidate grids best first. Verified programs vote; then the
    unverified programs by train examples reproduced. Mirrors agent.aggregate."""
    out = []
    ntr = len(task["train"])
    for j in range(len(task["test"])):
        score = {}
        grids = {}
        for r in records:
            for p in r["programs"]:
                g = p["test"][j] if j < len(p["test"]) else None
                if g is None:
                    continue
                k = json.dumps(g)
                grids[k] = g
                verified = p["n_ok"] == ntr
                s = score.get(k, (0, 0, 0))
                score[k] = (s[0] + verified, max(s[1], p["n_ok"]), s[2] + 1)
        out.append([grids[k] for k in sorted(score, key=lambda k: score[k], reverse=True)])
    return out


def direct(cfg, tasks, client, deadline, sub, sub_path, t0):
    """Direct answers (run_direct.answer): every task's sample 0 before any sample 1, each
    task in `tasks` order. attempt_1 = the first answer; attempt_2 = the next DIFFERENT one."""
    from run_direct import answer
    lock = threading.Lock()
    got = {}
    jobs = [(t, k) for k in range(cfg.get("samples", 2)) for t in tasks]
    log = open(cfg.get("log_path", "/kaggle/working/direct_attempts.jsonl"), "a")
    count = {}

    def work(job):
        t, k = job
        if deadline - time.time() < cfg.get("min_start_s", 600):
            rec = {"status": "skipped: deadline", "grids": None, "gen_tokens": 0}
        else:
            ts = time.time()
            rec = answer(client, tasks[t], 1000 * k + 7, deadline, max_tokens=cfg.get("max_tokens"),
                         rescue_tokens=cfg.get("rescue_tokens", 8192))
            rec["wall"] = round(time.time() - ts, 1)
        with lock:
            if rec["grids"]:
                got.setdefault(t, []).append(rec["grids"])
                for j, x in enumerate(tasks[t]["test"]):
                    cands = []
                    for grids in got[t]:
                        if grids[j] not in cands:
                            cands.append(grids[j])
                    sub[t][j] = {"attempt_1": cands[0],
                                 "attempt_2": cands[1] if len(cands) > 1 else x["input"]}
                write_atomic(sub, sub_path)
            st = rec["status"].split(":")[0]
            count[st] = count.get(st, 0) + 1
            log.write(json.dumps(dict({k2: v for k2, v in rec.items() if k2 != "grids"},
                                      task_id=t, sample=k)) + "\n")
            log.flush()
            print(f"[{time.time() - t0:6.0f}s] {t}#{k} {rec['status'][:40]:12s} "
                  f"tok {rec.get('gen_tokens', 0):6d} finish {rec.get('finish')} | {count}", flush=True)

    with ThreadPoolExecutor(max_workers=cfg.get("concurrency", 10)) as ex:
        list(ex.map(work, jobs))
    log.close()
    print(f"done: {len(got)}/{len(tasks)} tasks answered, {count}, {time.time() - t0:.0f}s", flush=True)


def main(cfg):
    t0 = time.time()
    rerun = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
    sub_path = cfg.get("sub_path", SUB)
    ch = json.load(open(find(cfg.get("challenges", "arc-agi_test_challenges.json"),
                             cfg.get("data_dir", "/kaggle/input"))))
    if not rerun and not cfg.get("full_interactive"):
        # Interactive run (should not happen -- versions are Quick Saved): stay tiny.
        ch = {t: ch[t] for t in sorted(ch)[:cfg.get("interactive_tasks", 3)]}
    if not os.path.exists(sub_path):
        write_fallback(ch, sub_path)
    n = len(ch)
    deadline = cfg["deadline_epoch"]
    avail = max(0.0, deadline - time.time())
    supply = cfg.get("tok_per_s", 140) * avail
    cap = int(min(cfg.get("max_cap", 60000), max(cfg.get("min_cap", 15000), supply / max(n, 1) * cfg.get("cap_factor", 1.0))))   # v1.1: was x1.3
    print(f"rerun={rerun} tasks={n} available {avail / 3600:.2f} h, est supply {supply / 1e6:.1f}M "
          f"tokens -> per-attempt cap {cap}", flush=True)

    # Cheapest prompts first: if the deadline bites, the long tasks are the ones lost.
    order = sorted(ch, key=lambda t: len(json.dumps(ch[t])))
    tasks = {t: ch[t] for t in order}
    client = ChatClient(cfg["base_url"], cfg["model"], max_tokens=cfg.get("max_tokens", 16384),
                        temperature=cfg.get("temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                        top_k=cfg.get("top_k", 20), thinking=cfg.get("thinking", True),
                        reasoning_effort=cfg.get("reasoning_effort"),
                        thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=cfg.get("template_kwargs"))

    sub = json.load(open(sub_path))
    if cfg.get("mode") == "direct":
        return direct(cfg, tasks, client, deadline, sub, sub_path, t0)

    from rescue import run_rescue, merge, direct_grid      # v1.6
    lock = threading.Lock()
    done = {}
    direct_recs = {}

    def tagged(tid):
        """ranked_answers with the verified flag (rescue.merge puts the direct answer after verified grids only)"""
        task, recs = tasks[tid], done.get(tid, [])
        ntr = len(task["train"])
        out = []
        for j, cands in enumerate(ranked_answers(recs, task)):
            ver = {json.dumps(p["test"][j]) for r in recs for p in r["programs"]
                   if p["n_ok"] == ntr and j < len(p["test"]) and p["test"][j] is not None}
            out.append([(g, "verified" if json.dumps(g) in ver else "unverified") for g in cands])
        return out

    def refresh(tid):                                  # caller holds `lock`
        task = tasks[tid]
        for j, ranked in enumerate(tagged(tid)):
            sub[tid][j] = merge(ranked, direct_grid(direct_recs.get(tid), j), task["test"][j]["input"])
        write_atomic(sub, sub_path)

    def on_done(rec):
        breaker(rec)                                   # v1.6: stop early if the first attempts all die on HTTP errors
        with lock:
            done.setdefault(rec["task_id"], []).append(rec)
            refresh(rec["task_id"])

    rescue_log = cfg.get("rescue_log", "/kaggle/working/rescue_attempts.jsonl")

    def log_rescue(tid, rec):                          # v1.8.2: one line per finished test input, flushed
        try:
            with open(rescue_log, "a") as f:
                f.write(json.dumps({"task_id": tid, "status": rec.get("status"), "gen_tokens": rec.get("gen_tokens"),
                                    "answered": sum(1 for g in rec.get("grids") or [] if g is not None)},
                                   default=str) + "\n")
        except Exception as e:
            print("rescue log failed:", e, flush=True)

    if cfg.get("agent_disabled"):                      # v1.8.2: the agent failed its end-to-end probe
        print("AGENT disabled by the e2e gate -> DIRECT rescue on every task for the whole time", flush=True)

        def on_result0(tid, rec):                      # run_rescue calls this under its own lock
            with lock:
                direct_recs[tid] = rec
                refresh(tid)
                log_rescue(tid, rec)

        recs0, _ = run_rescue(list(tasks), tasks, cfg, deadline, log=lambda m: print(m, flush=True),
                              on_result=on_result0)
        print(f"direct-only done: {sum(1 for r in recs0.values() for g in r['grids'] if g is not None)} test inputs "
              f"answered, {time.time() - t0:.0f}s", flush=True)
        return

    agent_deadline = (t0 + (deadline - t0) * float(cfg.get("agent_frac", 0.5))
                      if cfg.get("rescue") else deadline)
    if cfg.get("rerun_attempt_caps"):                  # v1.7: 240 tasks need a smaller scout than EVAL12 (GPT audit)
        cfg["attempt_caps"] = list(cfg["rerun_attempt_caps"])
    print(f"per-attempt caps {cfg.get('attempt_caps')}", flush=True)
    print(f"agent until {(agent_deadline - time.time()) / 3600:.2f} h from now, then direct rescue until "
          f"{(deadline - time.time()) / 3600:.2f} h", flush=True)
    _, report = solve_repl(
        tasks, client, attempts=cfg.get("attempts", 1), concurrency=cfg.get("concurrency", 24),
        deadline=agent_deadline, stop_after=1, max_turns=cfg.get("max_turns", 40), max_gen_tokens=cap,
        context_budget=cfg.get("context_budget", 24000), nudge_every=cfg.get("nudge_every", 0),
        turn_max_tokens=cfg.get("turn_max_tokens"), in_python=cfg.get("in_python", False),
        log=lambda m: print(m, flush=True), on_done=on_done,
        **{k: cfg[k] for k in AGENT_OPTS if k in cfg and k not in ("turn_max_tokens", "in_python")})

    ver = sum(1 for a in report["attempts"] if a["status"] == "verified")
    print(f"agent done: {ver}/{n} tasks verified, {sum(a['gen_tokens'] for a in report['attempts'])} "
          f"tokens, {time.time() - t0:.0f}s", flush=True)
    if cfg.get("rescue"):
        verified = {a["task_id"] for a in report["attempts"] if a["status"] == "verified"}
        # v1.7: tasks the agent never finished an attempt on come first (nothing at all to submit), then the
        # attempted-but-unverified ones; cheapest prompts first inside each group
        todo = ([t for t in tasks if t not in verified and t not in done]
                + [t for t in tasks if t not in verified and t in done])

        def on_result(tid, rec):                       # run_rescue calls this under its own lock
            with lock:
                direct_recs[tid] = rec
                refresh(tid)
                log_rescue(tid, rec)

        print(f"rescue (direct answers): {len(todo)} unverified tasks, {(deadline - time.time()) / 3600:.2f} h left",
              flush=True)
        recs, _ = run_rescue(todo, tasks, cfg, deadline, log=lambda m: print(m, flush=True), on_result=on_result)
        n_q = sum(len(r["grids"]) for r in recs.values())
        n_g = sum(1 for r in recs.values() for g in r["grids"] if g is not None)
        print(f"rescue done: {n_g}/{n_q} test inputs answered ({len(todo)} tasks), {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
