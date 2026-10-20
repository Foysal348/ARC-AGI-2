"""v1.6 HYBRID: a DIRECT answer (the LB-18.75 method: reason, then write the grid; run_direct.answer) for every task
the agent did not verify, so no task ends with only the test input. Also the shared merge / scoring helpers.

Candidate order per test input: verified program outputs (voted) -> the direct answer -> unverified program outputs
(most train pairs right first) -> the test input. The first two distinct grids become attempt_1 / attempt_2.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from client import ChatClient
from run_direct import answer


def direct_client(cfg):
    think = int(cfg.get("rescue_think", 16384))
    use_think = bool(cfg.get("thinking", True))
    return ChatClient(cfg["base_url"], cfg["model"], max_tokens=think + int(cfg.get("rescue_answer", 4096)),
                      temperature=cfg.get("rescue_temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                      top_k=cfg.get("top_k", 20), thinking=use_think,
                      thinking_token_budget=(think if use_think and cfg.get("thinking_token_budget") else None),
                      template_kwargs={"reasoning_effort": cfg.get("rescue_effort", "xhigh")})


def run_rescue(task_ids, tasks, cfg, deadline, log=print, on_result=None):
    """direct answers for task_ids, concurrently; returns ({tid: record}, client). record['grids'] = list per test."""
    client = direct_client(cfg)
    out, lock = {}, threading.Lock()

    def work(tid):
        if deadline - time.time() < float(cfg.get("rescue_min_start_s", 300)):
            rec = {"status": "skipped: deadline", "grids": None, "gen_tokens": 0}
        else:
            t = time.time()
            try:
                rec = answer(client, tasks[tid], 7, deadline, max_tokens=client.max_tokens,
                             rescue_tokens=int(cfg.get("rescue_commit_tokens", 4096)))
            except Exception as e:                       # one task must never sink the stage
                rec = {"status": f"crashed: {type(e).__name__}: {e}"[:200], "grids": None, "gen_tokens": 0}
            rec["wall"] = round(time.time() - t, 1)
        with lock:
            out[tid] = rec
            if on_result:
                try:
                    on_result(tid, rec)
                except Exception as e:
                    log(f"[rescue] on_result failed for {tid}: {e}")
        log(f"[rescue] {tid} {str(rec.get('status'))[:40]} tok {rec.get('gen_tokens', 0)}")

    with ThreadPoolExecutor(max_workers=int(cfg.get("concurrency", 16))) as ex:
        list(ex.map(work, list(task_ids)))
    return out, client


def direct_grid(rec, j):
    grids = (rec or {}).get("grids")
    return grids[j] if isinstance(grids, list) and j < len(grids) else None


def merge(agent_ranked, direct, inp):
    """agent_ranked: [(grid, source)] best first, source 'verified ...' or 'unverified ...'; direct: grid or None"""
    ver = [g for g, s in agent_ranked if str(s).startswith("verified")]
    unv = [g for g, s in agent_ranked if not str(s).startswith("verified")]
    out = []
    for g in ver + ([direct] if direct is not None else []) + unv + [inp]:
        if g is not None and g not in out:
            out.append(g)
        if len(out) == 2:
            break
    while len(out) < 2:
        out.append(inp)
    return {"attempt_1": out[0], "attempt_2": out[1]}


def hybrid_submission(answers, rescue_recs, tasks):
    return {t: [merge(answers.get(t, [[]] * len(task["test"]))[j], direct_grid(rescue_recs.get(t), j), x["input"])
                for j, x in enumerate(task["test"])] for t, task in tasks.items()}


def direct_submission(rescue_recs, tasks):
    sub = {}
    for t, task in tasks.items():
        rows = []
        for j, x in enumerate(task["test"]):
            g = direct_grid(rescue_recs.get(t), j)
            rows.append({"attempt_1": g if g is not None else x["input"], "attempt_2": x["input"]})
        sub[t] = rows
    return sub


def hits(sub, sol, ids):
    """per-query correctness (Kaggle metric: either attempt exact)"""
    return {(t, j): int(s in (sub[t][j]["attempt_1"], sub[t][j]["attempt_2"]))
            for t in ids for j, s in enumerate(sol[t])}
