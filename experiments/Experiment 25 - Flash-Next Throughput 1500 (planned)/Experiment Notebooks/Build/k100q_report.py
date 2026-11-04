"""k100q_report.py -- the klee100 QUALITY report: phase Q_K100_FP8_K3 (klee100 low-bit Flash-Next, FP8 KV) against
NB5 QUALITY's FP8 phase Q_FP8_K3 (W4 Flash-Next, FP8 KV; label "w4") and, as a second reference, NB5's BF16 phase.

stdlib only. The klee100 per-config record is computed EXACTLY the way NB5 computes quality_ab.per_config:

* the attempt / task / group / pair functions in the VERBATIM block below are byte-identical copies of nb5_lib.py
  (NO_REQUEST_VERDICTS ... qab_pair); k100q_report_tests.py checks them against nb5_lib.py when it is next to this file;
* the per-config shape mirrors build_e25_nb5.py nb5_qab_phase_record (window, capacity_levers, speed, qab tasks +
  summary) and the R2 per_config block (kv, status, fp8_status, window, window_plan, capacity_levers, speed, order,
  score, tasks, summary);
* the pass@2 score from e12_generated_attempts.json mirrors the E12 run-report scoring (nb5_code_export.py, the
  "if SOLUTIONS is not None:" block): pass1 = attempt_1 == gold, pass2 = attempt_2 == gold, hit = either.

Entry points
    build_quality_report(run_report, generated_attempts, engine, solutions, ref=None, *, records=None, by_attempt=None,
                         phase_dir=None, order=None, group_of=None, budgets=None) -> dict
    markdown_table(report) -> str
    load_phase_records(phase_dir) -> (records, by_attempt, errors)
    load_reference(ref=None) -> dict

Inputs
    run_report          the phase's e12_run_report.json (dict or path). May be empty / None.
    generated_attempts  the phase's e12_generated_attempts.json ({task: [{"attempt_1": grid, "attempt_2": grid}, ...]}).
    engine              engine facts of the phase (dict). Canonical keys (aliases in brackets):
                        pool_blocks [pool], f_eff [F, f], workers, max_running, max_inflight, mean_running_busy,
                        preemptions, solver_tok_s [tok_s, window_tok_s], phase_tok_s, steady_p50
                        [steady_gen_tok_s_p50], pin_gib [pin], kv_bytes, ladder_rung [rung], max_num_seqs_launch,
                        capacity_blocks, pool_source, f_log, f_source, lever_env, captures, ladder_tries,
                        levers_dropped, config_readback, r2_workers, kv (default "fp8"), status, fp8_status,
                        window (NB5 window dict), window_plan, window_start_epoch, window_end_epoch, budgets
                        ({"b0": int, "b1": int}), sampler_rows (nb5_sampler rows: max_running / max_inflight /
                        mean_running_busy are then computed the NB5 way), capacity_levers / speed (full NB5 dicts,
                        used as the base when given), steady_mixed (NB5 steady_mixed dict), records / phase_dir /
                        by_attempt (same as the keyword arguments).
    solutions           {task: [gold grid per test query]} for the 15 tasks.
    ref                 nb5_fp8_reference.json (dict or path; None = the file next to this module), or a whole
                        nb5_report.json.

Records: NB5 computes the per-attempt fields from the in-memory E12 records (E12_RECORDS = the Duck branch results,
the same dicts as duck/<task>_b<k>/branch_result.json) and the tracer's by_attempt counts (nb5_requests.jsonl). Pass
records=E12_RECORDS (+ by_attempt=NB5_TRACE.summary()["by_attempt"]) inside the notebook, or phase_dir=<phase dir> to
load both from disk. Without either, the records are rebuilt from run_report["branches"] (DEGRADED: no t0 / t1, no
candidate grids -> exact_accepted None, window cuts not detectable); notes.records_source says which path was used.
"""
import json
import math
import statistics
from pathlib import Path

K100_PHASE = "Q_K100_FP8_K3"
REF_PHASE = "Q_FP8_K3"
REF_BF16_PHASE = "Q_BF16_K3"
LABELS = ("k100", "w4")
GROUP_ORDER = ("solid", "sometimes", "never")
DEFAULT_BUDGET = 104000
ATTEMPTS_PER_TASK = 2
REF_FILE_NAME = "nb5_fp8_reference.json"
MARK_OK = "\u2713"
MARK_BAD = "\u2717"


# ==================================================================== BEGIN VERBATIM nb5_lib.py (do not edit)
NO_REQUEST_VERDICTS = ("CONTEXT_EXHAUSTED", "INCONCLUSIVE_CONTEXT_LENGTH", "KV_NEED_ABOVE_POOL")


def _exact_all(cands, gold, accepted_only):
    return bool(gold) and any((c.get("accepted") or not accepted_only) and len(c.get("test_outputs") or []) == len(gold)
                              and all(a == b for a, b in zip(c["test_outputs"], gold)) for c in cands or [])


# Q-1 (R2 audits 1 + 2, fail-closed): an attempt is FINISHED only on positive terminal evidence of a normal end. Duck's
# normal ends (arc2_duck 9e4b8928, BranchSession.run / done_now) are exactly: (a) an accepted submit (record "accepted" =
# arc2_accepted set; this covers the W1 weak commit-zone accept and every solved early end: Duck ends the attempt on it);
# (b) the token budget (generated >= the session cap / b_budget - 1,024, or the commit turn done with < 768 tokens left =
# b_events.tiny_request_skipped, or NB5's floor event = BUDGET_LIMITED). Whitelisted as (b): NB5 CONTEXT_EXHAUSTED (the
# attempt's token room under the context ceiling is spent; it ends through the same budget route, no request sent).
# Everything else is NOT finished: a non-empty note (Duck's failure path: "analyzer failed N times consecutively",
# "<Exception>: ..." incl. CUDA / HTTP / timeout errors, "analyzer returned no result", setup failures, NB5 reservation
# denials), any other NB5 verdict (INCONCLUSIVE_CONTEXT_LENGTH, KV_NEED_ABOVE_POOL, SERVER_ERROR, ...), no terminal
# evidence -> FAILED_OR_INTERRUPTED; a paused attempt or one running at the window end -> INCOMPLETE_WINDOW; never
# started -> NOT_STARTED. A candidate accepted only by check() (no submit) is not terminal: the attempt went on.
FINISHED_OUTCOMES = ("ACCEPTED", "BUDGET_END")
BUDGET_END_VERDICTS = ("BUDGET_LIMITED", "CONTEXT_EXHAUSTED")


def fail_kind(text):
    """The failure class of a Duck note / NB5 verdict (report matrix only; every class is FAILED_OR_INTERRUPTED)."""
    t = str(text or "").lower()
    if "cuda" in t or "illegal memory" in t or "nccl" in t or "out of memory" in t:
        return "CUDA_ERROR"
    if "timeout" in t or "timed out" in t:
        return "TIMEOUT"
    if any(s in t for s in ("http", "connection", "urlerror", "server_error", "server error", "status code", "remote end")):
        return "HTTP_OR_SERVER_ERROR"
    if "analyzer failed" in t:
        return "REPEATED_ANALYZER_FAILURES"
    if "reservation" in t or "kv_need" in t:
        return "KV_RESERVATION_DENIED"
    if "no result" in t:
        return "ANALYZER_NO_RESULT"
    return "ERROR"


def attempt_end(r, budget, window_end_epoch=None, margin_sec=90.0):
    """(outcome, end_reason) of one attempt record (Q-1 rule above): ACCEPTED | BUDGET_END | INCOMPLETE_WINDOW |
    FAILED_OR_INTERRUPTED."""
    v = r.get("nb5_verdict")
    note = str(r.get("note") or "").strip()
    if r.get("accepted"):
        return "ACCEPTED", "accepted_submission"
    if v and v not in BUDGET_END_VERDICTS:
        return "FAILED_OR_INTERRUPTED", f"verdict {v}: " + fail_kind(f"{v} {note}")
    if note:
        return "FAILED_OR_INTERRUPTED", fail_kind(note)
    if v in BUDGET_END_VERDICTS:
        return "BUDGET_END", v
    tok, bud = int(r.get("generated_tokens") or 0), int(r.get("b_budget") or budget or 0)
    if (r.get("b_events") or {}).get("tiny_request_skipped"):
        return "BUDGET_END", "commit_done_tiny_rest"
    if bud and tok >= bud - 1024:
        return "BUDGET_END", "budget_spent"
    if r.get("status") == "paused":
        return "INCOMPLETE_WINDOW", "paused_at_window_end"
    if window_end_epoch and r.get("t1") and float(r["t1"]) >= float(window_end_epoch) - margin_sec:
        return "INCOMPLETE_WINDOW", "running_at_window_end"
    return "FAILED_OR_INTERRUPTED", "no_terminal_evidence"


def attempt_outcome(r, budget, window_end_epoch=None, margin_sec=90.0):
    return attempt_end(r, budget, window_end_epoch, margin_sec)[0]


def qab_tasks(records, solutions, unit_records, group_of, order, budgets, window_start_epoch=None, window_end_epoch=None,
              by_attempt=None, attempts_per_task=2):
    """Per task of one config (lead 2026-10-31): group, queries, pass@2 query hits from the scored submission
    (eval_unit_records), solved = every query hit; per attempt: outcome, finished, tokens, wall time, end offset in the
    window, accepted, exact (accepted / any candidate) on every test input, budget-limited, commit outcomes."""
    out, ur = {}, {}
    for u in unit_records or []:
        ur.setdefault(u.get("task"), []).append(u)
    for tid in order:
        rows = sorted((records or {}).get(tid) or [], key=lambda r: int(r.get("branch") or 0))
        gold = (solutions or {}).get(tid)
        att = []
        for r in rows:
            b = int(r.get("branch") or 0)
            cands = (r.get("submissions") or []) + (r.get("checked") or [])
            be = r.get("b_events") or {}
            ba = (by_attempt or {}).get(f"{tid}|{b}") or {}
            oc, why = attempt_end(r, (budgets or {}).get("b0" if b == 0 else "b1"), window_end_epoch)
            t0, t1 = r.get("t0"), r.get("t1")
            att.append({"attempt": b + 1, "outcome": oc, "finished": oc in FINISHED_OUTCOMES, "end_reason": why,
                        "status": r.get("status"), "failures": int(r.get("failures") or 0),
                        "near_window_end": bool(window_end_epoch and t1 and float(t1) >= float(window_end_epoch) - 90.0),
                        "tokens": int(r.get("generated_tokens") or 0),
                        "wall_sec": round(float(t1) - float(t0), 1) if t0 and t1 else None,
                        "end_offset_sec": round(float(t1) - float(window_start_epoch), 1) if t1 and window_start_epoch else None,
                        "accepted": any(c.get("accepted") for c in cands) or bool(r.get("accepted")),
                        "exact_accepted": _exact_all(cands, gold, True) if gold else None,
                        "exact_any_candidate": _exact_all(cands, gold, False) if gold else None,
                        "budget_limited": r.get("nb5_verdict") in ("BUDGET_LIMITED",) + NO_REQUEST_VERDICTS,
                        "nb5_verdict": r.get("nb5_verdict"),
                        "commit": {"commit_at_tokens": be.get("commit_at_tokens"), "commit_via": be.get("commit_via"),
                                   "commit_tool_call": be.get("commit_tool_call"), "commit_requests": ba.get("commit_requests"),
                                   "commit_with_tool_call": ba.get("commit_with_tool_call")},
                        "requests": ba.get("requests"), "no_request": ba.get("no_request"), "note": str(r.get("note") or "")[:200]})
        seen = {a["attempt"] for a in att}
        for k in range(1, attempts_per_task + 1):
            if k not in seen:
                att.append({"attempt": k, "outcome": "NOT_STARTED", "finished": False, "end_reason": "no_record", "tokens": 0})
        att.sort(key=lambda a: a["attempt"])
        q = ur.get(tid) or []
        nq = len((solutions or {}).get(tid) or []) or len(q)
        hits = sum(1 for u in q if u.get("hit"))
        t0s = [float(r["t0"]) for r in rows if r.get("t0")]
        t1s = [float(r["t1"]) for r in rows if r.get("t1")]
        out[tid] = {"group": (group_of or {}).get(tid), "queries": nq, "query_hits": hits,
                    "pass1_hits": sum(1 for u in q if u.get("pass1")), "pass2_hits": sum(1 for u in q if u.get("pass2")),
                    "solved": bool(nq) and hits == nq, "task_score": round(hits / nq, 3) if nq else None,
                    "attempts_started": sum(1 for a in att if a["outcome"] != "NOT_STARTED"),
                    "finished": all(a["finished"] for a in att),
                    "unfinished_outcomes": [a["outcome"] for a in att if not a["finished"]],
                    "tokens": sum(a.get("tokens") or 0 for a in att),
                    "wall_sec": round(max(t1s) - min(t0s), 1) if t0s and t1s else None, "attempts": att}
    return out


def qab_summary(tasks):
    """Per group (+ 'all') of one config: tasks, solved, task-normalized score, query hits, finished tasks / attempts,
    tokens, budget-limited, incomplete and not-started attempts."""
    groups = {}
    for v in (tasks or {}).values():
        for g in (v.get("group") or "?", "all"):
            s = groups.setdefault(g, {"tasks": 0, "solved": 0, "task_score_sum": 0.0, "query_hits": 0, "queries": 0,
                                      "finished_tasks": 0, "attempts_finished": 0, "attempts_incomplete": 0,
                                      "attempts_not_started": 0, "attempts_failed": 0, "outcomes": {}, "budget_limited": 0,
                                      "tokens": 0})
            s["tasks"] += 1
            s["solved"] += int(bool(v.get("solved")))
            s["task_score_sum"] += float(v.get("task_score") or 0)
            s["query_hits"] += int(v.get("query_hits") or 0)
            s["queries"] += int(v.get("queries") or 0)
            s["finished_tasks"] += int(bool(v.get("finished")))
            s["tokens"] += int(v.get("tokens") or 0)
            for a in v.get("attempts") or []:
                s["attempts_finished"] += int(bool(a.get("finished")) and a.get("outcome") != "NOT_STARTED")
                s["attempts_incomplete"] += int(a.get("outcome") == "INCOMPLETE_WINDOW")
                s["attempts_not_started"] += int(a.get("outcome") == "NOT_STARTED")
                s["attempts_failed"] += int(a.get("outcome") == "FAILED_OR_INTERRUPTED")
                s["outcomes"][str(a.get("outcome"))] = s["outcomes"].get(str(a.get("outcome")), 0) + 1
                s["budget_limited"] += int(bool(a.get("budget_limited")))
    for s in groups.values():
        s["task_norm_pct"] = round(100.0 * s.pop("task_score_sum") / s["tasks"], 2) if s["tasks"] else None
    return groups


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p (binomial on the discordant pairs)."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n), 4)


def qab_pair(a_tasks, b_tasks, labels=("FP8", "BF16")):
    """The paired summary (per group + all): solved in A / B / both / A only / B only / neither, the discordant pairs,
    the exact McNemar p, query hits; once over ALL tasks and once over the tasks FINISHED IN BOTH configs (every attempt
    of the task ended inside its window in both)."""
    la, lb = labels

    def block(tids):
        g = {}
        for t in tids:
            x, y = a_tasks.get(t) or {}, b_tasks.get(t) or {}
            for grp in (x.get("group") or y.get("group") or "?", "all"):
                s = g.setdefault(grp, {"tasks": 0, f"solved_{la}": 0, f"solved_{lb}": 0, "both": 0, f"{la}_only": 0,
                                       f"{lb}_only": 0, "neither": 0, f"query_hits_{la}": 0, f"query_hits_{lb}": 0,
                                       "queries": 0, "discordant_tasks": []})
                sa, sb = bool(x.get("solved")), bool(y.get("solved"))
                s["tasks"] += 1
                s[f"solved_{la}"] += int(sa)
                s[f"solved_{lb}"] += int(sb)
                s["both"] += int(sa and sb)
                s[f"{la}_only"] += int(sa and not sb)
                s[f"{lb}_only"] += int(sb and not sa)
                s["neither"] += int(not sa and not sb)
                s[f"query_hits_{la}"] += int(x.get("query_hits") or 0)
                s[f"query_hits_{lb}"] += int(y.get("query_hits") or 0)
                s["queries"] += int(x.get("queries") or y.get("queries") or 0)
                if sa != sb:
                    s["discordant_tasks"].append(t)
        for s in g.values():
            s["discordant"] = s[f"{la}_only"] + s[f"{lb}_only"]
            s["mcnemar_p"] = mcnemar_exact_p(s[f"{la}_only"], s[f"{lb}_only"])
            # Q-5: the power context next to McNemar: paired n, the smallest p this n / this discordant count can reach
            s["paired_n"] = s["tasks"]
            s["min_attainable_p_paired_n"] = round(min(1.0, 2.0 / 2.0 ** s["tasks"]), 6)
            s["min_attainable_p_at_discordant"] = round(min(1.0, 2.0 / 2.0 ** s["discordant"]), 6)
            s["reading"] = ("DIFFERENCE_P_LT_0_05" if s["mcnemar_p"] < 0.05 else
                            "NOT_SIGNIFICANT_NOT_PARITY" + ("_UNDERPOWERED" if s["min_attainable_p_at_discordant"] >= 0.05 else ""))
        return g
    tids = list(a_tasks) + [t for t in b_tasks if t not in a_tasks]
    fin = [t for t in tids if (a_tasks.get(t) or {}).get("finished") and (b_tasks.get(t) or {}).get("finished")]
    return {"labels": [la, lb], "all": block(tids), "finished_in_both": block(fin), "finished_in_both_tasks": fin,
            "note": "solved = every test query right in that config's pass@2 submission; finished = every attempt of the task "
                    "ended normally (ACCEPTED / BUDGET_END; never FAILED_OR_INTERRUPTED / INCOMPLETE_WINDOW / NOT_STARTED); "
                    "a non-significant McNemar p is never evidence of parity"}
# ==================================================================== END VERBATIM nb5_lib.py


# ------------------------------------------------------------------ small helpers
def _d(x):
    return x if isinstance(x, dict) else {}


def _l(x):
    return x if isinstance(x, list) else []


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_obj(x):
    """dict / list as is; a str / Path is read as a JSON file; anything else -> None."""
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, (str, Path)) and str(x):
        return _read_json(x)
    return None


def _median(v):
    return round(float(statistics.median(v)), 1) if v else None


def _mean(v):
    return round(float(sum(v)) / len(v), 1) if v else None


def _dv(a, b, nd=2):
    a, b = _num(a), _num(b)
    return {LABELS[0]: a, LABELS[1]: b, "delta": round(a - b, nd) if (a is not None and b is not None) else None,
            "ratio": round(a / b, 4) if (a is not None and b not in (None, 0)) else None}


_ENGINE_ALIASES = {"pool": "pool_blocks", "F": "f_eff", "f": "f_eff", "pin": "pin_gib", "rung": "ladder_rung",
                   "steady_gen_tok_s_p50": "steady_p50", "tok_s": "solver_tok_s", "window_tok_s": "solver_tok_s"}


def _norm_engine(engine):
    e = dict(_d(engine))
    for a, k in _ENGINE_ALIASES.items():
        if a in e and e.get(k) is None:
            e[k] = e[a]
    return e


# ------------------------------------------------------------------ inputs
def load_reference(ref=None):
    """The NB5 reference: nb5_fp8_reference.json (dict / path / None = next to this module) or a whole nb5_report.json
    (converted to the same shape). Always returns a dict (empty when nothing loads)."""
    obj = _as_obj(ref)
    if obj is None and ref is None:
        try:
            here = Path(__file__).resolve().parent
        except NameError:
            here = Path.cwd()
        for p in (here / REF_FILE_NAME, Path.cwd() / REF_FILE_NAME):
            if p.exists():
                obj = _read_json(p)
                break
    obj = _d(obj)
    if "quality_ab" in obj and "configs" not in obj:
        qa, ph = _d(obj.get("quality_ab")), _d(obj.get("phases"))
        cfg = {}
        for n, p in _d(qa.get("per_config")).items():
            c = dict(_d(p))
            c["steady_mixed"] = _d(ph.get(n)).get("steady_mixed")
            c["solver_window"] = _d(ph.get(n)).get("solver_window")
            cfg[n] = c
        obj = {"source": {"nb5_report": {"run_id": obj.get("run_id"), "note": "converted from a passed nb5_report"}},
               "primary": REF_PHASE, "secondary": REF_BF16_PHASE, "label": LABELS[1], "groups": qa.get("groups"),
               "fp8_label": qa.get("fp8_label"), "configs": cfg}
    return obj


def load_phase_records(phase_dir):
    """(records, by_attempt, errors) of one phase directory: records = {task: [branch_result.json dicts]} (= NB5's
    E12_RECORDS), by_attempt = the Tracer by_attempt counts rebuilt from every nb5_requests.jsonl (requests, errors,
    completion, commit_requests, commit_with_tool_call, no_request), exactly the keys nb5_lib.Tracer keeps."""
    records, by, errors = {}, {}, []
    try:
        base = Path(phase_dir)
    except TypeError:
        return records, by, ["phase_dir not a path"]
    duck = base / "duck" if (base / "duck").is_dir() else base
    if not duck.is_dir():
        return records, by, [f"no directory {duck}"]
    for f in sorted(duck.glob("*/branch_result.json")):
        r = _read_json(f)
        if not isinstance(r, dict):
            errors.append(f"unreadable {f.parent.name}/branch_result.json")
            continue
        tid = r.get("task_id") or r.get("task")
        if not tid:
            errors.append(f"no task id in {f.parent.name}/branch_result.json")
            continue
        records.setdefault(str(tid), []).append(r)
    for f in sorted(duck.glob("*/nb5_requests.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            errors.append(f"unreadable {f.parent.name}/nb5_requests.jsonl")
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            ba = by.setdefault(f"{row.get('task')}|{row.get('branch')}",
                               {"requests": 0, "errors": 0, "completion": 0, "commit_requests": 0,
                                "commit_with_tool_call": 0, "no_request": None})
            if row.get("kind") == "no_request":
                ba["no_request"] = row.get("verdict")
                continue
            err = row.get("error") is not None
            ba["requests"] += 1
            ba["errors"] += int(err)
            ba["completion"] += 0 if err else int(row.get("completion") or 0)
            if str(row.get("kind") or "").startswith("commit"):
                ba["commit_requests"] += 1
                ba["commit_with_tool_call"] += int(bool(row.get("tool_calls")))
    return records, by, errors


def records_from_run_report(run_report):
    """DEGRADED records from run_report["branches"] (no t0 / t1, no candidate grids)."""
    out = {}
    for b in _l(_d(run_report).get("branches")):
        if not isinstance(b, dict) or not b.get("task"):
            continue
        out.setdefault(str(b["task"]), []).append(
            {"task_id": b["task"], "branch": b.get("branch"), "accepted": b.get("accepted"), "status": b.get("status"),
             "note": b.get("note"), "failures": b.get("failures"), "generated_tokens": b.get("generated_tokens"),
             "nb5_verdict": b.get("nb5_verdict"), "b_budget": b.get("b_budget"), "_degraded_sec": b.get("sec")})
    return out


def _clean_records(records):
    """Only dict rows with an integer-like branch; non-numeric t0 / t1 / token / budget fields become None (shallow copies,
    the caller's records are never changed), so a malformed record cannot crash qab_tasks."""
    out = {}
    for t, rows in _d(records).items():
        rows = [r for r in (rows if isinstance(rows, list) else [rows]) if isinstance(r, dict)]
        good = []
        for r in rows:
            try:
                int(r.get("branch") or 0)
            except (TypeError, ValueError):
                continue
            bad = [k for k in ("t0", "t1", "generated_tokens", "b_budget", "failures")
                   if r.get(k) is not None and _num(r.get(k)) is None]
            for k in ("submissions", "checked"):
                if r.get(k) is not None and not isinstance(r.get(k), list):
                    bad.append(k)
            if bad:
                r = dict(r)
                for k in bad:
                    r[k] = None
            good.append(r)
        if good:
            out[str(t)] = good
    return out


def units_cross_check(run_units, gen_units):
    """(task, query) -> (hit, pass1, pass2) of the run report's units vs the canonical recomputed ones: MISMATCH on any
    difference, a duplicate (task, query) key, or a query present on one side only."""
    def key(u):
        return str(_d(u).get("task")), int(_d(u).get("query") or 0)

    def val(u):
        return bool(u.get("hit")), bool(u.get("pass1")), bool(u.get("pass2"))
    rk, dup = {}, []
    for u in _l(run_units):
        k = key(u)
        if k in rk:
            dup.append(list(k))
        rk[k] = val(_d(u))
    gk = {key(u): val(_d(u)) for u in _l(gen_units)}
    differ = [list(k) for k in gk if k in rk and rk[k] != gk[k]]
    miss = [list(k) for k in gk if k not in rk]
    extra = [list(k) for k in rk if k not in gk]
    bad = differ or miss or extra or dup
    return {"status": "MISMATCH" if bad else "MATCH", "units": len(gk), "differ": differ[:20], "missing_in_run_report": miss[:20],
            "extra_in_run_report": extra[:20], "duplicates": dup[:20]}


def score_from_generated(generated_attempts, solutions, task_ids, records=None, errors=None):
    """(score, unit_records) from the pass@2 submission, the E12 run-report way (nb5_code_export.py "if SOLUTIONS is
    not None:" block); the CANONICAL per-query units of the report (audit F). The gold sets the query count (a missing
    generated query = a miss); a task without gold is not scored. Validity problems go to errors (a list): MISSING_GOLD:<t>,
    MISSING_GENERATED:<t>, QUERY_COUNT_MISMATCH:<t>:generated=<n>,gold=<m>, INVALID_ATTEMPTS:<t>:<q>. Returns (None, [])
    without solutions."""
    sol, gen, recs = _d(solutions), _d(generated_attempts), _d(records)
    errs = errors if isinstance(errors, list) else []
    if not sol:
        return None, []
    hits = pass1 = pass2 = total = 0
    task_norm, recs_eval = 0.0, []
    errs.extend(f"MISSING_GOLD:{t}" for t in task_ids if not _l(sol.get(t)))
    task_ids = [t for t in task_ids if _l(sol.get(t))]   # a task without gold is not scored (E12: SOLUTIONS = the scored tasks)
    for t in task_ids:
        golds = _l(sol.get(t))
        g_t = _l(gen.get(t))
        if t not in gen:
            errs.append(f"MISSING_GENERATED:{t}")
        elif len(g_t) != len(golds):
            errs.append(f"QUERY_COUNT_MISMATCH:{t}:generated={len(g_t)},gold={len(golds)}")
        errs.extend(f"INVALID_ATTEMPTS:{t}:{qi}" for qi, x in enumerate(g_t)
                    if not (isinstance(x, dict) and "attempt_1" in x and "attempt_2" in x))
        nq = len(golds) if golds else len(g_t)        # audit 2026-11-01: gold is authoritative; a missing query = a miss
        th = 0
        for qi in range(nq):
            att = g_t[qi] if qi < len(g_t) and isinstance(g_t[qi], dict) else {}
            gold = golds[qi] if qi < len(golds) else None
            a = gold is not None and att.get("attempt_1") == gold
            b = gold is not None and att.get("attempt_2") == gold
            hits += a or b
            pass1 += a
            pass2 += b
            total += 1
            th += a or b
            acc = [s for r in _l(recs.get(t)) if isinstance(r, dict) for s in _l(r.get("submissions"))
                   if isinstance(s, dict) and s.get("accepted")]
            harv = [s for r in _l(recs.get(t)) if isinstance(r, dict) for s in _l(r.get("checked"))
                    if isinstance(s, dict) and s.get("accepted")]
            recs_eval.append({"task": t, "query": qi, "hit": bool(a or b), "pass1": bool(a), "pass2": bool(b),
                              "accepted_programs": len(acc),
                              "accepted_programs_correct": sum(1 for s in acc if qi < len(s.get("test_outputs") or [])
                                                               and s["test_outputs"][qi] == gold),
                              "harvested_programs": len(harv),
                              "harvested_programs_correct": sum(1 for s in harv if qi < len(s.get("test_outputs") or [])
                                                                and s["test_outputs"][qi] == gold)})
        task_norm += th / max(1, nq)
    score = {"eval_query_hits": hits, "eval_query_total": total, "eval_pass1_hits": pass1, "eval_pass2_hits": pass2,
             "eval_task_norm_pct": round(100 * task_norm / max(1, len(task_ids)), 2)}
    return score, recs_eval


SCORE_KEYS = ("eval_query_hits", "eval_query_total", "eval_pass1_hits", "eval_pass2_hits", "eval_task_norm_pct")


def _window(run_report, e):
    """NB5's window dict (start_sec, end_sec, actual_sec, planned_sec, solver_returned_sec) + the window epochs."""
    run = _d(_d(run_report).get("e12_run"))
    w = dict(_d(e.get("window")))
    if not w:
        ws = _d(run.get("window_shift")).get("window_start_sec")
        ws = ws if ws is not None else run.get("started_sec")
        end = _num(run.get("planned_solve_end_sec"))
        w = {"start_sec": ws, "end_sec": round(end) if end is not None else None,
             "actual_sec": round(end - float(ws), 1) if (end is not None and _num(ws) is not None) else None,
             "planned_sec": _d(e.get("window_plan")).get("planned_sec"), "solver_returned_sec": run.get("finished_sec")}
    we = _num(e.get("window_end_epoch"))
    we = we if we is not None else _num(run.get("solve_end_epoch"))
    ws_ep = _num(e.get("window_start_epoch"))
    if ws_ep is None and we is not None and _num(w.get("actual_sec")) is not None:
        ws_ep = we - float(w["actual_sec"])
    return w, ws_ep, we


CAP_KEYS = ("pool_blocks", "pool_source", "capacity_blocks", "f_eff", "f_log", "f_source", "pin_gib", "kv_bytes",
            "lever_env", "max_num_seqs_launch", "workers", "r2_workers", "captures", "ladder_tries", "levers_dropped",
            "config_readback")


def _capacity(e, run_report):
    """NB5 capacity_levers (same keys, same order) + ladder_rung; engine facts first, the run report's engine block as
    the fallback."""
    base = _d(e.get("capacity_levers"))
    rep_eng = _d(_d(run_report).get("engine"))
    cap = _d(rep_eng.get("capacity"))
    out = {}
    for k in CAP_KEYS:
        v = base.get(k)
        out[k] = v if v is not None else e.get(k)
    for k in ("pool_blocks", "capacity_blocks", "f_eff", "f_log"):
        if out.get(k) is None:
            out[k] = cap.get(k)
    if out.get("workers") is None:
        w = cap.get("workers") or rep_eng.get("concurrency") or _d(_d(run_report).get("e12_run")).get("rev6_workers")
        out["workers"] = int(w) if _num(w) is not None else None
    out["ladder_rung"] = base.get("ladder_rung") if base.get("ladder_rung") is not None else e.get("ladder_rung")
    return out


def _speed(e):
    """NB5 speed (same keys). sampler_rows -> max_running / max_inflight / mean_running_busy the NB5 way."""
    base = _d(e.get("speed"))
    get = lambda k: base.get(k) if base.get(k) is not None else e.get(k)
    rows = [r for r in _l(e.get("sampler_rows")) if isinstance(r, dict)]
    mx_run, mx_inf, mean_busy = get("max_running"), get("max_inflight"), get("mean_running_busy")
    if rows:
        busy = [r for r in rows if isinstance(r.get("running"), int) and (r.get("inflight") or 0) > 0]
        mx_run = max([r["running"] for r in rows if isinstance(r.get("running"), int)] or [0])
        mx_inf = max([int(r.get("inflight") or 0) for r in rows] or [0])
        mean_busy = round(sum(r["running"] for r in busy) / len(busy), 2) if busy else None
    pre = get("preemptions")
    pre = int(pre) if _num(pre) is not None else None
    return {"max_running": mx_run, "max_inflight": mx_inf, "mean_running_busy": mean_busy, "preemptions": pre,
            "speed_status": ("PREEMPTED" if pre else "CLEAN") if pre is not None else "UNKNOWN",
            "accuracy_affected_by_preemption": False, "solver_tok_s": get("solver_tok_s"),
            "phase_tok_s": get("phase_tok_s") if get("phase_tok_s") is not None else get("solver_tok_s")}


def _order(order, ref_cfg, run_report, solutions):
    for cand in (order, _d(ref_cfg).get("order"), _d(run_report).get("order"),
                 _d(_d(_d(run_report).get("configuration")).get("profile_row")).get("task_list"),
                 list(_d(solutions))):
        if isinstance(cand, (list, tuple)) and cand:
            return [str(t) for t in cand]
    return []


def token_groups(tasks, groups=GROUP_ORDER):
    """Generated tokens per group (the NB5 doc table): attempt-1 median / mean / max, attempt-2 median / mean, the
    per-task mean (both attempts) and the total; only started attempts / tasks count."""
    names = list(groups) + sorted({str(v.get("group")) for v in _d(tasks).values() if v.get("group") not in groups}
                                  - {"None"}) + ["all"]
    out = {}
    for g in names:
        vs = [v for v in _d(tasks).values() if g == "all" or v.get("group") == g]
        att = lambda k: [int(a.get("tokens") or 0) for v in vs for a in _l(v.get("attempts"))
                         if a.get("attempt") == k and a.get("outcome") != "NOT_STARTED"]
        a1, a2 = att(1), att(2)
        tt = [int(v.get("tokens") or 0) for v in vs if (v.get("attempts_started") or 0) > 0]
        out[g] = {"tasks": len(vs), "tasks_with_tokens": len(tt), "a1_n": len(a1), "a1_median": _median(a1),
                  "a1_mean": _mean(a1), "a1_max": max(a1) if a1 else None, "a2_n": len(a2), "a2_median": _median(a2),
                  "a2_mean": _mean(a2), "task_mean": _mean(tt), "tokens_total": sum(tt)}
    return out


def _attempt_view(v, k):
    for a in _l(_d(v).get("attempts")):
        if a.get("attempt") == k:
            return {"tokens": int(a.get("tokens") or 0), "outcome": a.get("outcome"), "end_reason": a.get("end_reason"),
                    "accepted": a.get("accepted"), "exact_accepted": a.get("exact_accepted"),
                    "wall_sec": a.get("wall_sec")}
    return {"tokens": 0, "outcome": "NOT_STARTED", "end_reason": "no_record", "accepted": None, "exact_accepted": None,
            "wall_sec": None}


def _side_view(v):
    v = _d(v)
    return {"present": bool(v), "solved": bool(v.get("solved")), "query_hits": v.get("query_hits"),
            "queries": v.get("queries"), "pass1_hits": v.get("pass1_hits"), "pass2_hits": v.get("pass2_hits"),
            "tokens": v.get("tokens"), "finished": v.get("finished"), "attempts_started": v.get("attempts_started"),
            "a1": _attempt_view(v, 1), "a2": _attempt_view(v, 2)}


def _label(sa, sb):
    return "both" if (sa and sb) else (f"{LABELS[0]}_only" if sa else (f"{LABELS[1]}_only" if sb else "neither"))


def paired_report(k_tasks, w_tasks, order):
    """Per task: group, klee100 vs W4 solved, attempt 1 / 2 tokens + outcomes on both sides, the label (both /
    k100_only / w4_only / neither); counts + the exact McNemar p via nb5_lib.qab_pair (labels k100 / w4)."""
    k_tasks, w_tasks = _d(k_tasks), _d(w_tasks)
    tids = list(order) + [t for t in list(k_tasks) + list(w_tasks) if t not in order]
    tids = list(dict.fromkeys(tids))
    rows = {}
    for t in tids:
        x, y = _d(k_tasks.get(t)), _d(w_tasks.get(t))
        rows[t] = {"group": x.get("group") or y.get("group"), "label": _label(bool(x.get("solved")), bool(y.get("solved"))),
                   LABELS[0]: _side_view(x), LABELS[1]: _side_view(y)}
    kk = {t: k_tasks.get(t) or {"group": rows[t]["group"]} for t in tids}
    ww = {t: w_tasks.get(t) or {"group": rows[t]["group"]} for t in tids}
    qp = qab_pair(kk, ww, labels=LABELS)
    a = _d(_d(qp.get("all")).get("all"))
    counts = {k: a.get(k) for k in ("tasks", f"solved_{LABELS[0]}", f"solved_{LABELS[1]}", "both", f"{LABELS[0]}_only",
                                    f"{LABELS[1]}_only", "neither", "discordant", "discordant_tasks", "mcnemar_p",
                                    "paired_n", "min_attainable_p_paired_n", "min_attainable_p_at_discordant", "reading")}
    labels = {}
    for t, r in rows.items():
        labels.setdefault(r["label"], []).append(t)
    return {"labels": list(LABELS), "tasks": rows, "label_tasks": labels, "counts": counts, "by_group": qp.get("all"),
            "finished_in_both": qp.get("finished_in_both"), "finished_in_both_tasks": qp.get("finished_in_both_tasks"),
            "note": qp.get("note")}


def _summary_all(rec):
    return _d(_d(_d(rec).get("summary")).get("all"))


def delta_report(k_rec, w_rec, groups=GROUP_ORDER):
    """klee100 minus W4: solved, queries, pass1 / pass2, task-norm, tokens per group, window tok/s, steady p50, pool
    blocks, workers (+ F, max_running, preemptions)."""
    k_rec, w_rec = _d(k_rec), _d(w_rec)
    ks, ws = _summary_all(k_rec), _summary_all(w_rec)
    ksc, wsc = _d(k_rec.get("score")), _d(w_rec.get("score"))
    kt, wt = token_groups(k_rec.get("tasks"), groups), token_groups(w_rec.get("tasks"), groups)
    tok = {}
    for g in list(dict.fromkeys(list(kt) + list(wt))):
        a, b = _d(kt.get(g)), _d(wt.get(g))
        tok[g] = {s: _dv(a.get(s), b.get(s), 1) for s in ("a1_median", "a1_mean", "a1_max", "a2_median", "task_mean",
                                                           "tokens_total")}
    by_group = {}
    for g in list(dict.fromkeys([*groups, *_d(k_rec.get("summary")), *_d(w_rec.get("summary"))])):
        a, b = _d(_d(k_rec.get("summary")).get(g)), _d(_d(w_rec.get("summary")).get(g))
        if a or b:
            by_group[g] = {"solved": _dv(a.get("solved"), b.get("solved")), "tasks": _dv(a.get("tasks"), b.get("tasks")),
                           "query_hits": _dv(a.get("query_hits"), b.get("query_hits")),
                           "task_norm_pct": _dv(a.get("task_norm_pct"), b.get("task_norm_pct"))}
    kc, wc = _d(k_rec.get("capacity_levers")), _d(w_rec.get("capacity_levers"))
    kp, wp = _d(k_rec.get("speed")), _d(w_rec.get("speed"))
    return {"solved": _dv(ks.get("solved"), ws.get("solved")), "tasks": _dv(ks.get("tasks"), ws.get("tasks")),
            "queries": _dv(ksc.get("eval_query_hits", ks.get("query_hits")), wsc.get("eval_query_hits", ws.get("query_hits"))),
            "queries_total": _dv(ksc.get("eval_query_total", ks.get("queries")), wsc.get("eval_query_total", ws.get("queries"))),
            "pass1": _dv(ksc.get("eval_pass1_hits"), wsc.get("eval_pass1_hits")),
            "pass2": _dv(ksc.get("eval_pass2_hits"), wsc.get("eval_pass2_hits")),
            "task_norm_pct": _dv(ksc.get("eval_task_norm_pct", ks.get("task_norm_pct")),
                                 wsc.get("eval_task_norm_pct", ws.get("task_norm_pct"))),
            "tokens_total": _dv(ks.get("tokens"), ws.get("tokens")),
            "tokens_by_group": tok, "tokens_by_group_k100": kt, "tokens_by_group_w4": wt, "by_group": by_group,
            "window_tok_s": _dv(kp.get("solver_tok_s"), wp.get("solver_tok_s")),
            "steady_p50": _dv(_d(k_rec.get("steady_mixed")).get("gen_tok_s_p50"),
                              _d(w_rec.get("steady_mixed")).get("gen_tok_s_p50")),
            "pool_blocks": _dv(kc.get("pool_blocks"), wc.get("pool_blocks")), "workers": _dv(kc.get("workers"), wc.get("workers")),
            "f_eff": _dv(kc.get("f_eff"), wc.get("f_eff")), "max_running": _dv(kp.get("max_running"), wp.get("max_running")),
            "preemptions": _dv(kp.get("preemptions"), wp.get("preemptions"))}


# ------------------------------------------------------------------ E26 30-task extension (2026-11-01)
NEW_GROUPS = ("hard_long", "sometimes_new")         # the 15 tasks after NB5's 15 (no W4 run exists for them)
REF27_FILE_NAME = "k100q_ref27b.json"
REF27_LABEL = "r3k2"


def load_ref27(ref27=None):
    """The 27B Duck R3K2 (E15B EVAL120, c058d58b) + ARC Prize Qwen3.8-27B xhigh reference: dict / path / None = the file
    next to this module. Always a dict ({} when nothing loads)."""
    if isinstance(ref27, dict):
        return ref27
    p = Path(ref27) if ref27 else Path(__file__).with_name(REF27_FILE_NAME)
    return _d(_read_json(p))


def scope_to_reference(k_rec, w_rec):
    """The klee100 record restricted to the tasks the W4 reference ran (NB5's 15): tasks, summary and score recomputed
    from those tasks only, so paired / delta never compare a 30-task side against a 15-task side and never count a task
    W4 did not run as 'W4 unsolved'. Returns (scoped record, the reference task ids)."""
    k_rec, w_tasks = _d(k_rec), _d(_d(w_rec).get("tasks"))
    order = [str(t) for t in _l(_d(w_rec).get("order")) if str(t) in w_tasks] or list(w_tasks)
    order += [t for t in w_tasks if t not in order]
    kt = {t: v for t, v in _d(k_rec.get("tasks")).items() if t in w_tasks}
    sc = None
    if kt and not k_rec.get("score_mismatch"):
        q = sum(int(v.get("queries") or 0) for v in kt.values())
        sc = {"eval_query_hits": sum(int(v.get("query_hits") or 0) for v in kt.values()), "eval_query_total": q,
              "eval_pass1_hits": sum(int(v.get("pass1_hits") or 0) for v in kt.values()),
              "eval_pass2_hits": sum(int(v.get("pass2_hits") or 0) for v in kt.values()),
              "eval_task_norm_pct": round(100.0 * sum(float(v.get("task_score") or 0) for v in kt.values()) / len(kt), 2)}
    out = dict(k_rec, tasks=kt, summary=qab_summary(kt) if kt else {}, score=sc,
               order=[t for t in order if t in kt or t in w_tasks], scope="reference_tasks")
    return out, order


# ------------------------------------------------------------------ audit E (2026-11-01): matched-window pairing
MATCHED_CUTOFF_SEC = 4050.0          # NB5 QUALITY Q_FP8_K3's window (W4 planned 4,049.8 s, actual 4,049.5 s)
SAME_WINDOW_TOL_SEC = 60.0


def offsets_from_finish(finish):
    """{"task|attempt": seconds after the window start} from the notebook's k100_finish block (the end of each attempt's
    last request; the record's own t1 is written at the session close = the window end, so it is never used here)."""
    out = {}
    for t, v in _d(_d(finish).get("per_task")).items():
        for a in _l(_d(v).get("attempts")):
            if _num(_d(a).get("end_offset_sec")) is not None:
                out[f"{t}|{int(a.get('attempt') or 1)}"] = float(a["end_offset_sec"])
    return out


def offsets_from_phase_dir(phase_dir, tasks, window_start_epoch):
    """The same offsets read from duck/<task>_b<k>/nb5_requests.jsonl (t1 of the last request - the window start epoch)."""
    out = {}
    if not phase_dir or _num(window_start_epoch) is None:
        return out
    for t, v in _d(tasks).items():
        for a in _l(_d(v).get("attempts")):
            k = int(a.get("attempt") or 1)
            f = Path(phase_dir) / "duck" / f"{t}_b{k - 1}" / "nb5_requests.jsonl"
            if not f.is_file():
                continue
            last = None
            for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    x = json.loads(ln)
                except Exception:
                    continue
                if isinstance(x, dict) and _num(x.get("t1")) is not None:
                    last = max(last or 0.0, float(x["t1"]))
            if last is not None:
                out[f"{t}|{k}"] = round(last - float(window_start_epoch), 1)
    return out


def accept_times(records, solutions, window_start_epoch):
    """Audit J: per attempt ("task|attempt"), the acceptance time of the EARLIEST accepted candidate (submissions + checked)
    that is exact on every test query (the exact_accepted rule), in seconds after the window start. Duck (arc2_duck.py
    sha b9f67010 = the notebook's CELL 3 file, Exp 15 src/duck_b): _verify_submission records t = time.time() when the
    verification starts (:2505) and verify_sec = its duration (:2554, rev 6), then sets accepted (:2562); a submit verdict
    goes to submissions (:2206), a check() verdict to checked (:2408 / :2444 / :2489). accept = t + verify_sec. No t or no
    verify_sec -> accept_sec None (verify_sec alone is kept for REQUEST_END_PLUS_VERIFY)."""
    out, ws, sol = {}, _num(window_start_epoch), _d(solutions)
    for tid, rs in _d(records).items():
        gold = _l(sol.get(tid))
        if not gold:
            continue
        for r in _l(rs):
            if not isinstance(r, dict):
                continue
            try:
                k = int(r.get("branch") or 0) + 1
            except (TypeError, ValueError):
                continue
            best, vs = None, []
            for c in _l(r.get("submissions")) + _l(r.get("checked")):
                if not (isinstance(c, dict) and _exact_all([c], gold, True)):
                    continue
                tt, v = _num(c.get("t")), _num(c.get("verify_sec"))
                if v is not None:
                    vs.append(v)
                if tt is not None and v is not None and ws is not None:
                    best = round(tt + v - ws, 1) if best is None else min(best, round(tt + v - ws, 1))
            if vs or best is not None:
                out[f"{tid}|{k}"] = {"accept_sec": best, "verify_sec": max(vs) if vs else None}
    return out


def window_basis(k_window, w_window, tol=SAME_WINDOW_TOL_SEC):
    """PAIRED_TASK_OUTCOME_SAME_WINDOW when both actual windows are known and within tol seconds, else
    PAIRED_TASK_OUTCOME_DIFFERENT_WINDOWS (unknown -> PAIRED_TASK_OUTCOME_WINDOWS_UNKNOWN)."""
    a = _num(_d(k_window).get("actual_sec"))
    b = _num(_d(w_window).get("actual_sec"))
    if a is None or b is None:
        return "PAIRED_TASK_OUTCOME_WINDOWS_UNKNOWN"
    return "PAIRED_TASK_OUTCOME_SAME_WINDOW" if abs(a - b) <= tol else "PAIRED_TASK_OUTCOME_DIFFERENT_WINDOWS"


WINDOW_TOL_SEC = 2.0


def matched_cutoff_report(k_tasks, w_tasks, order, offsets, cutoff=MATCHED_CUTOFF_SEC, w_window=None, k_window=None,
                          accept=None):
    """NB5's tasks, the outcome at a matched cutoff: solved = an attempt whose accepted answer is exact on every test query
    (the same rule on both sides); a klee100 attempt counts only if it ended (its last request) <= cutoff s after the
    klee100 window start. W4 ran a cutoff-long window, so all its attempts count. A klee100 exact attempt without timing
    that would decide the task -> the task is undecided and the whole block INCONCLUSIVE_NO_TIMING (never assumed)."""
    k_tasks, w_tasks, offsets, accept = _d(k_tasks), _d(w_tasks), _d(offsets), _d(accept)
    rows, undecided, bases = {}, [], set()
    for t in [t for t in order if t in w_tasks]:
        ex = [a for a in _l(_d(k_tasks.get(t)).get("attempts")) if a.get("exact_accepted") is True]
        offs, basis = {}, {}
        for a in ex:                      # audit J: the exact accept's own time, else the last request end + verify_sec
            k = int(a["attempt"])
            at = _d(accept.get(f"{t}|{k}"))
            o = _num(offsets.get(f"{t}|{k}"))
            if _num(at.get("accept_sec")) is not None:
                offs[k], basis[k] = float(at["accept_sec"]), "ACCEPT_TIME"
            elif o is not None and _num(at.get("verify_sec")) is not None:
                offs[k], basis[k] = round(o + float(at["verify_sec"]), 1), "REQUEST_END_PLUS_VERIFY"
            else:
                offs[k], basis[k] = None, None
        bases |= {b for b in basis.values() if b}
        inside = sorted(k for k, o in offs.items() if o is not None and o <= cutoff)
        untimed = sorted(k for k, o in offs.items() if o is None)
        sk = True if inside else (None if untimed else False)
        sw = any(a.get("exact_accepted") is True for a in _l(_d(w_tasks.get(t)).get("attempts")))
        sf = bool(ex)
        if sk is None:
            undecided.append(t)
        rows[t] = {"group": _d(k_tasks.get(t)).get("group") or _d(w_tasks.get(t)).get("group"),
                   "k100_exact_attempt_time_sec": {str(k): o for k, o in offs.items()},
                   "k100_exact_attempt_time_basis": {str(k): b for k, b in basis.items()}, "k100_full_window": sf,
                   "k100_at_cutoff": sk, "w4": sw, "label_full_window": _label(sf, sw),
                   "label_at_cutoff": _label(bool(sk), sw) if sk is not None else None}

    def counts(key):
        n = {"tasks": len(rows), "both": 0, f"{LABELS[0]}_only": 0, f"{LABELS[1]}_only": 0, "neither": 0}
        for r in rows.values():
            n[r[key]] += 1
        n["discordant"] = n[f"{LABELS[0]}_only"] + n[f"{LABELS[1]}_only"]
        n["mcnemar_p"] = mcnemar_exact_p(n[f"{LABELS[0]}_only"], n[f"{LABELS[1]}_only"])
        n[f"solved_{LABELS[0]}"] = n["both"] + n[f"{LABELS[0]}_only"]
        n[f"solved_{LABELS[1]}"] = n["both"] + n[f"{LABELS[1]}_only"]
        return n
    no_timing = not offsets and not any(_num(_d(v).get("accept_sec")) is not None for v in accept.values())
    status = (("INCONCLUSIVE_NO_TIMING" if no_timing else "INCONCLUSIVE_ACCEPT_TIME") if undecided
              else ("OK" if rows else "NO_REFERENCE_TASKS"))
    ka = _num(_d(k_window).get("actual_sec"))
    if k_window is not None and (ka is None or ka < cutoff - WINDOW_TOL_SEC):     # audit H: the window never reached the cutoff
        return {"status": "INCONCLUSIVE_SHORT_WINDOW", "actual_sec": ka, "required_sec": cutoff, "tolerance_sec": WINDOW_TOL_SEC,
                "note": "the klee100 window ended before the matched cutoff: no matched comparison (the full-window pairing is "
                        "in 'paired'; its same-rule counts below)", "counts_at_cutoff": None,
                "counts_full_window_same_rule": counts("label_full_window") if rows else None}
    return {"status": status, "cutoff_sec": cutoff, "w4_window_actual_sec": _d(w_window).get("actual_sec"),
            "k100_window_actual_sec": ka,
            "rule": "solved = an attempt with an exact accepted answer (every query); a klee100 attempt counts only if that "
                    f"accept happened <= {cutoff:g} s after the klee100 window start; W4's window was that long",
            "time_basis": "ACCEPT_TIME" if bases == {"ACCEPT_TIME"} else ("REQUEST_END_PLUS_VERIFY" if bases == {
                "REQUEST_END_PLUS_VERIFY"} else ("MIXED" if bases else None)),
            "time_basis_note": "ACCEPT_TIME = Duck's verification start t + verify_sec of the exact accepted candidate "
                               "(arc2_duck.py :2505 / :2554 / :2562); REQUEST_END_PLUS_VERIFY = the attempt's last request end "
                               "+ verify_sec",
            "tasks": rows, "undecided_tasks": undecided, "timed_attempts": len(offsets),
            "counts_at_cutoff": counts("label_at_cutoff") if status == "OK" else None,
            "counts_full_window_same_rule": counts("label_full_window") if rows else None}


def attempt1_report(k_tasks, w_tasks, order, offsets=None):
    """Attempt 1 only, klee100 vs W4, on the reference tasks (both attempt 1s run to completion): per task tokens /
    outcome / exact accepted on both sides and the label on attempt-1 exactness; counts + the exact McNemar p."""
    k_tasks, w_tasks = _d(k_tasks), _d(w_tasks)
    rows, n = {}, {"tasks": 0, "both": 0, f"{LABELS[0]}_only": 0, f"{LABELS[1]}_only": 0, "neither": 0,
                   f"{LABELS[0]}_a1_tokens": 0, f"{LABELS[1]}_a1_tokens": 0, f"{LABELS[0]}_a1_finished": 0,
                   f"{LABELS[1]}_a1_finished": 0}
    for t in [t for t in order if t in w_tasks]:
        a, b = _attempt_view(k_tasks.get(t), 1), _attempt_view(w_tasks.get(t), 1)
        a["finish_offset_sec"] = _d(offsets).get(f"{t}|1")       # audit E: the last request's end after the window start
        b["finish_offset_sec"] = None                            # W4: request timings not in the reference (window 4,050 s)
        ea, eb = a.get("exact_accepted") is True, b.get("exact_accepted") is True
        lab = _label(ea, eb)
        rows[t] = {"group": _d(k_tasks.get(t)).get("group") or _d(w_tasks.get(t)).get("group"), "label": lab,
                   LABELS[0]: a, LABELS[1]: b}
        n["tasks"] += 1
        n[lab] += 1
        for side, x in ((LABELS[0], a), (LABELS[1], b)):
            n[f"{side}_a1_tokens"] += int(x.get("tokens") or 0)
            n[f"{side}_a1_finished"] += int(x.get("outcome") in FINISHED_OUTCOMES)
    n["discordant"] = n[f"{LABELS[0]}_only"] + n[f"{LABELS[1]}_only"]
    n["mcnemar_p"] = mcnemar_exact_p(n[f"{LABELS[0]}_only"], n[f"{LABELS[1]}_only"])
    return {"rule": "attempt 1 exact accepted (every test query right), klee100 vs W4, NB5's tasks only", "tasks": rows,
            "counts": n}


def ref27_report(k_tasks, ref27, order):
    """Every task of the run vs the 27B Duck R3K2 run (its pass@2 hits, per-branch generated tokens / accepted; its
    attempts were 50k / 45k) and ARC Prize Qwen3.8-27B xhigh (pass on the ledger); per group: tasks, solved by klee100 /
    by R3K2 / by ARC Prize xhigh; klee100 vs R3K2 paired (McNemar) over the tasks R3K2 ran."""
    k_tasks, r = _d(k_tasks), _d(_d(ref27).get("tasks"))
    tids = list(dict.fromkeys([t for t in order] + [t for t in k_tasks if t not in order]))
    rows, groups, kk, yy = {}, {}, {}, {}
    for t in tids:
        x, y = _d(k_tasks.get(t)), _d(r.get(t))
        g = x.get("group") or y.get("group")
        rows[t] = {"group": g, LABELS[0]: _side_view(x),
                   REF27_LABEL: {"present": bool(y), "solved": y.get("solved"), "query_hits": y.get("query_hits"),
                                 "queries": y.get("queries"), "branches": y.get("branches")},
                   "arcprize_xhigh": y.get("arcprize_xhigh"),
                   "label_vs_r3k2": _label(bool(x.get("solved")), bool(y.get("solved"))).replace(LABELS[1], REF27_LABEL) if y else None}
        for gg in (str(g), "all"):
            s = groups.setdefault(gg, {"tasks": 0, f"solved_{LABELS[0]}": 0, f"solved_{REF27_LABEL}": 0, "solved_arcprize_xhigh": 0,
                                       f"query_hits_{LABELS[0]}": 0, "queries": 0})
            s["tasks"] += 1
            s[f"solved_{LABELS[0]}"] += int(bool(x.get("solved")))
            s[f"solved_{REF27_LABEL}"] += int(bool(y.get("solved")))
            s["solved_arcprize_xhigh"] += int(y.get("arcprize_xhigh") == 1)
            s[f"query_hits_{LABELS[0]}"] += int(x.get("query_hits") or 0)
            s["queries"] += int(x.get("queries") or y.get("queries") or 0)
        if y:
            kk[t] = x or {"group": g}
            yy[t] = {"group": g, "solved": bool(y.get("solved")), "query_hits": y.get("query_hits"), "queries": y.get("queries"),
                     "finished": True}
    qp = qab_pair(kk, yy, labels=(LABELS[0], REF27_LABEL)) if yy else {}
    return {"source": _d(ref27).get("source"), "tasks": rows, "by_group": groups,
            "paired_vs_r3k2": _d(qp.get("all")), "missing_in_ref27": [t for t in tids if t not in r]}


def run_summary(k_rec, order):
    """The klee100 run over ALL its tasks (30): per group (NB5's three + the new ones) solved / task-norm / query hits /
    finished / tokens (qab_summary), the score from the run report, attempt-1 / -2 token groups."""
    k_rec = _d(k_rec)
    gs = list(GROUP_ORDER) + [g for g in NEW_GROUPS]
    return {"tasks": len(_d(k_rec.get("tasks"))), "order_tasks": len(order), "score": k_rec.get("score"),
            "summary": k_rec.get("summary"), "tokens_by_group": token_groups(k_rec.get("tasks"), gs)}


def _attempt_notes(side, tasks):
    acc_wrong, cut, not_started, failed = [], [], [], []
    for t, v in _d(tasks).items():
        for a in _l(_d(v).get("attempts")):
            row = {"side": side, "task": t, "group": v.get("group"), "attempt": a.get("attempt"),
                   "tokens": int(a.get("tokens") or 0), "end_reason": a.get("end_reason")}
            if a.get("accepted") and a.get("exact_accepted") is False:
                acc_wrong.append(row)
            if a.get("outcome") == "INCOMPLETE_WINDOW":
                cut.append(row)
            if a.get("outcome") == "NOT_STARTED":
                not_started.append(row)
            if a.get("outcome") == "FAILED_OR_INTERRUPTED":
                failed.append(dict(row, note=a.get("note")))
    return acc_wrong, cut, not_started, failed


# ------------------------------------------------------------------ the report
def build_k100_record(run_report, generated_attempts, engine, solutions, ref=None, *, records=None, by_attempt=None,
                      phase_dir=None, order=None, group_of=None, budgets=None, _errors=None):
    """The klee100 per-config record in NB5's quality_ab.per_config shape (+ steady_mixed, solver_window, inputs)."""
    errors = _errors if isinstance(_errors, list) else []
    rr = _d(_as_obj(run_report))
    e = _norm_engine(engine)
    refd = load_reference(ref) if not (isinstance(ref, dict) and "configs" in ref) else ref
    ref_cfg = _d(_d(refd.get("configs")).get(refd.get("primary") or REF_PHASE))
    sol = _d(_as_obj(solutions))
    gen = _d(_as_obj(generated_attempts))
    records = records if records is not None else e.get("records")
    by_attempt = by_attempt if by_attempt is not None else e.get("by_attempt")
    phase_dir = phase_dir if phase_dir is not None else e.get("phase_dir")
    src = "records"
    if records is None and phase_dir:
        records, by2, errs = load_phase_records(phase_dir)
        errors.extend(errs)
        by_attempt = by_attempt if by_attempt is not None else by2
        src = "phase_dir"
    if records is None or (not records and rr.get("branches")):
        records, src = records_from_run_report(rr), "run_report_branches_DEGRADED"
    records = _clean_records(records)
    if not records:
        src = src + "_EMPTY"
    gof = dict(group_of) if isinstance(group_of, dict) else {t: g for g, ts in _d(refd.get("groups")).items()
                                                             for t in _l(ts)}
    tids = _order(order, ref_cfg, rr, sol)
    bud = budgets if isinstance(budgets, dict) else e.get("budgets")
    if not isinstance(bud, dict):
        pa = _d(_d(_d(rr.get("engine")).get("capacity")).get("per_attempt"))
        bud = {"b0": _d(pa.get("b0")).get("budget") or DEFAULT_BUDGET, "b1": _d(pa.get("b1")).get("budget") or DEFAULT_BUDGET}
    win, ws_ep, we_ep = _window(rr, e)
    # score + unit records: the run report's (NB5: p["score"] / rep["eval_unit_records"]), else from the submission
    g_score, g_units = score_from_generated(gen, sol, tids, records, errors) if gen or sol else (None, [])
    r_score = {k: rr.get(k) for k in SCORE_KEYS} if rr.get("eval_query_total") is not None else None
    r_units = rr.get("eval_unit_records") if isinstance(rr.get("eval_unit_records"), list) and rr.get("eval_unit_records") else None
    # audit F: the units recomputed from gold + generated attempts are canonical; the run report's are cross-checked
    units = g_units if g_score is not None else (r_units or [])
    unit_check = units_cross_check(r_units, g_units) if (g_score is not None and r_units) else {
        "status": "NOT_CHECKED", "why": "no gold" if g_score is None else "no run-report units"}
    if unit_check.get("status") == "MISMATCH":
        errors.append("UNIT_RECORD_MISMATCH: " + json.dumps({k: unit_check.get(k) for k in ("differ", "missing_in_run_report",
                                                                                          "extra_in_run_report", "duplicates")})[:600])
    agree = (r_score == g_score) if (r_score is not None and g_score is not None) else None
    score_sources = ("both" if agree is not None else "run_report_only" if r_score is not None
                     else "generated_only" if g_score is not None else "none")
    score = r_score if r_score is not None else g_score
    if agree is False:                       # audit 2026-11-01: never pick one silently; the record shows no number
        errors.append("SCORE_SOURCE_MISMATCH: run report " + json.dumps(r_score) + " vs generated attempts " + json.dumps(g_score))
        score = None
    score_check = {"run_report": r_score, "generated_attempts": g_score, "agree": agree, "score_sources": score_sources,
                   "units_source": "generated_attempts" if units is g_units else "run_report", "unit_check": unit_check}
    try:
        tasks = qab_tasks(records, sol or None, units, gof, tids, bud, ws_ep, we_ep, _d(by_attempt),
                          attempts_per_task=ATTEMPTS_PER_TASK)
    except Exception as ex:                  # never crash: an empty table, the error in notes
        errors.append(f"qab_tasks: {type(ex).__name__}: {ex}"[:400])
        tasks = {}
    if src.startswith("run_report_branches"):
        sec = {(t, int(r.get("branch") or 0)): r.get("_degraded_sec") for t, rs in records.items() for r in rs}
        for t, v in tasks.items():
            for a in v.get("attempts") or []:
                if (t, a["attempt"] - 1) in sec:
                    a["exact_accepted"] = a["exact_any_candidate"] = None
                    a["wall_sec"] = sec[(t, a["attempt"] - 1)]
    try:
        summary = qab_summary(tasks)
    except Exception as ex:
        errors.append(f"qab_summary: {type(ex).__name__}: {ex}"[:400])
        summary = {}
    run = _d(rr.get("e12_run"))
    status = e.get("status") or ("NO_RUN_REPORT" if not rr else ("SOLVER_ERROR" if run.get("error") else "DONE"))
    kv = e.get("kv") or "fp8"
    steady = dict(_d(e.get("steady_mixed")))
    if e.get("steady_p50") is not None:
        steady["gen_tok_s_p50"] = e.get("steady_p50")
    speed = _speed(e)
    rec = {"kv": kv, "status": status,
           "fp8_status": e.get("fp8_status") or (refd.get("fp8_label") if kv == "fp8" else "BF16_REFERENCE"),
           "window": win, "window_plan": e.get("window_plan"), "capacity_levers": _capacity(e, rr), "speed": speed,
           "order": rr.get("order") or tids, "score": score, "tasks": tasks, "summary": summary,
           "steady_mixed": steady or None,
           "solver_window": {"tok_s": speed.get("solver_tok_s")}}
    if agree is False:
        rec["score_mismatch"] = True             # audit 2026-11-01: the reference-scoped score is withheld too
    try:
        acc_t = accept_times(records, sol, ws_ep)
    except Exception as ex:
        errors.append(f"accept_times: {type(ex).__name__}: {ex}"[:400])
        acc_t = {}
    inputs = {"accept_times": acc_t,
              "records_source": src, "records_tasks": len(records), "records_attempts": sum(len(v) for v in records.values()),
              "by_attempt_keys": len(_d(by_attempt)), "order": tids, "budgets": bud,
              "window_start_epoch": ws_ep, "window_end_epoch": we_ep, "score_check": score_check,
              "solutions_tasks": len(sol), "generated_tasks": len(gen), "run_report_keys": sorted(rr)[:40],
              "tasks_outside_order": sorted(set(records) - set(tids))}
    return rec, inputs, refd


def build_quality_report(run_report, generated_attempts, engine, solutions, ref=None, *, records=None, by_attempt=None,
                         phase_dir=None, order=None, group_of=None, budgets=None, ref27=None, finish=None,
                         matched_cutoff_sec=MATCHED_CUTOFF_SEC):
    """The klee100 QUALITY report. Keys: phase, ref_phase, k100 (NB5 per-config shape, ALL tasks of the run), ref
    ({w4_fp8, w4_bf16}), paired (per task + counts + McNemar), paired_vs_bf16 (counts only), delta, attempt1_vs_w4
    (paired / delta / attempt 1: the reference tasks only = NB5's 15), run_summary (all tasks by group), vs_27b (the
    27B R3K2 + ARC Prize xhigh columns), notes, inputs, errors. Never raises."""
    errors = []
    try:
        k_rec, inputs, refd = build_k100_record(run_report, generated_attempts, engine, solutions, ref, records=records,
                                                by_attempt=by_attempt, phase_dir=phase_dir, order=order,
                                                group_of=group_of, budgets=budgets, _errors=errors)
    except Exception as ex:
        errors.append(f"build_k100_record: {type(ex).__name__}: {ex}"[:400])
        k_rec, inputs, refd = {"tasks": {}, "summary": {}}, {}, {}
    cfg = _d(refd.get("configs"))
    w_rec = _d(cfg.get(refd.get("primary") or REF_PHASE))
    b_rec = _d(cfg.get(refd.get("secondary") or REF_BF16_PHASE))
    groups = GROUP_ORDER
    order_ = list(inputs.get("order") or k_rec.get("order") or [])
    out = {"phase": K100_PHASE, "ref_phase": refd.get("primary") or REF_PHASE, "labels": list(LABELS), "k100": k_rec,
           "ref": {"w4_fp8": w_rec or None, "w4_bf16": b_rec or None, "source": refd.get("source"),
                   "groups": refd.get("groups")}}
    # scope (E26 30-task run): paired / paired_vs_bf16 / delta / attempt 1 compare ONLY the tasks the W4 reference ran
    # (NB5's 15); a task W4 never ran is never counted as "W4 unsolved". Without a reference the full record is used.
    if _d(w_rec.get("tasks")):
        k_ref, ref_order = scope_to_reference(k_rec, w_rec)
    else:
        k_ref, ref_order = k_rec, order_
    out["scope"] = {"paired_delta_attempt1_tasks": len(ref_order), "run_tasks": len(_d(k_rec.get("tasks"))),
                    "tasks_not_in_reference": [t for t in order_ if t not in ref_order]}
    try:
        out["paired"] = paired_report(k_ref.get("tasks"), w_rec.get("tasks"), ref_order)
    except Exception as ex:
        errors.append(f"paired: {type(ex).__name__}: {ex}"[:400])
        out["paired"] = {}
    try:
        _bt = _d(b_rec.get("tasks"))
        pb = qab_pair({t: v for t, v in _d(k_ref.get("tasks")).items() if t in _bt or not _bt},
                      {t: (_bt.get(t) or {"group": v.get("group")}) for t, v in _d(k_ref.get("tasks")).items() if t in _bt or not _bt},
                      labels=(LABELS[0], "bf16")) if b_rec else {"skipped": "no BF16 reference"}
        out["paired_vs_bf16"] = {k: pb.get(k) for k in ("labels", "all", "finished_in_both_tasks", "skipped") if k in pb}
    except Exception as ex:
        errors.append(f"paired_vs_bf16: {type(ex).__name__}: {ex}"[:400])
        out["paired_vs_bf16"] = {}
    try:
        out["delta"] = delta_report(k_ref, w_rec, groups)
    except Exception as ex:
        errors.append(f"delta: {type(ex).__name__}: {ex}"[:400])
        out["delta"] = {}
    _offs, _osrc = offsets_from_finish(finish), "k100_finish"
    if not _offs:
        _offs = offsets_from_phase_dir(phase_dir if phase_dir is not None else _d(engine).get("phase_dir"), k_ref.get("tasks"),
                                       inputs.get("window_start_epoch"))
        _osrc = "phase_dir_requests" if _offs else "none"
    try:
        out["paired"]["window_basis"] = window_basis(k_rec.get("window"), w_rec.get("window"))
        out["paired_at_matched_4050s"] = dict(matched_cutoff_report(k_ref.get("tasks"), w_rec.get("tasks"), ref_order, _offs,
                                                                    float(matched_cutoff_sec), w_rec.get("window"),
                                                                    _d(k_rec.get("window")), _d(inputs.get("accept_times"))),
                                              offsets_source=_osrc)
    except Exception as ex:
        errors.append(f"paired_at_matched_4050s: {type(ex).__name__}: {ex}"[:400])
        out["paired_at_matched_4050s"] = {"status": "ERROR"}
    try:
        out["attempt1_vs_w4"] = attempt1_report(k_ref.get("tasks"), w_rec.get("tasks"), ref_order, _offs)
    except Exception as ex:
        errors.append(f"attempt1_vs_w4: {type(ex).__name__}: {ex}"[:400])
        out["attempt1_vs_w4"] = {}
    try:
        out["run_summary"] = run_summary(k_rec, order_)
    except Exception as ex:
        errors.append(f"run_summary: {type(ex).__name__}: {ex}"[:400])
        out["run_summary"] = {}
    try:
        r27 = load_ref27(ref27) if ref27 is not False else {}
        out["vs_27b"] = ref27_report(k_rec.get("tasks"), r27, order_) if r27 else {"skipped": "no 27B / ARC Prize reference"}
    except Exception as ex:
        errors.append(f"vs_27b: {type(ex).__name__}: {ex}"[:400])
        out["vs_27b"] = {}
    notes = {"records_source": inputs.get("records_source"), "degraded": str(inputs.get("records_source", "")).startswith(
        "run_report_branches"), "reference_present": bool(w_rec), "reference_sha256": _d(_d(refd.get("source")).get(
            "nb5_report")).get("sha256")}
    try:
        acc_wrong, cut, ns, failed = [], [], [], []
        for side, tk in ((LABELS[0], k_rec.get("tasks")), (LABELS[1], w_rec.get("tasks"))):
            a, c, n, f = _attempt_notes(side, tk)
            acc_wrong += a
            cut += c
            ns += n
            failed += f
        ktasks = _d(k_rec.get("tasks"))
        notes.update(accepted_not_correct=acc_wrong, window_cut=cut, not_started=ns, failed=failed,
                     missing_tasks=[t for t in order_ if not (_d(ktasks.get(t)).get("attempts_started") or 0)],
                     tasks_outside_order=inputs.get("tasks_outside_order") or [],
                     score_check=inputs.get("score_check"))
        lines = []
        for r in acc_wrong:
            lines.append(f"accepted != correct: {r['side']} {r['task']} ({r['group']}) attempt {r['attempt']}, "
                         f"{r['tokens']:,} tokens")
        for r in cut:
            lines.append(f"cut by the window: {r['side']} {r['task']} ({r['group']}) attempt {r['attempt']}, "
                         f"{r['tokens']:,} tokens ({r['end_reason']})")
        for r in failed:
            lines.append(f"failed / interrupted: {r['side']} {r['task']} attempt {r['attempt']} ({r['end_reason']})")
        if notes["missing_tasks"]:
            lines.append(f"{LABELS[0]} tasks with no started attempt: {', '.join(notes['missing_tasks'])}")
        if notes["degraded"]:
            lines.append("DEGRADED records (run_report branches only): exact_accepted unknown, window cuts not detectable")
        sc = _d(notes.get("score_check"))
        if sc.get("agree") is False:
            lines.append("score mismatch: run report vs generated attempts " + json.dumps(
                {"run_report": sc.get("run_report"), "generated_attempts": sc.get("generated_attempts")}))
        if not w_rec:
            lines.append("NB5 FP8 reference missing: paired / delta show the klee100 side only")
        notes["lines"] = lines
    except Exception as ex:
        errors.append(f"notes: {type(ex).__name__}: {ex}"[:400])
    out["notes"] = notes
    out["inputs"] = inputs
    out["errors"] = errors
    return out


# ------------------------------------------------------------------ the notebook's CELL 9 block (audits 2026-11-01)
FP8_BASE_PIN_BYTES = 2479838700


def finish_table(tasks, window, phase_dir=None, window_start_epoch=None, task_solve=None, task_list=None):
    """E26 K100: per task finished / solved and the FINISH TIME in the solve window = the end of the attempt's last request
    (t1 of the last line of duck/<task>_b<k>/nb5_requests.jsonl - the window start; the record's own t1 is written when the
    session closes, at the window end, so it is kept only as record_end_offset_sec), the first accept (NB5 task_solve) and
    the attempt outcomes / tokens; all_finished = every task of the list finished (ACCEPTED / BUDGET_END on both attempts)."""
    per = {}
    for t, v in (tasks or {}).items():
        at, rows = v.get("attempts") or [], []
        for a in at:
            last, n = None, 0
            f = Path(phase_dir) / "duck" / f"{t}_b{int(a.get('attempt') or 1) - 1}" / "nb5_requests.jsonl" if phase_dir else None
            if f is not None and f.is_file():
                for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        x = json.loads(ln)
                    except Exception:
                        continue
                    n += 1
                    if isinstance(x.get("t1"), (int, float)):
                        last = max(last or 0.0, float(x["t1"]))
            rows.append({"attempt": a.get("attempt"), "outcome": a.get("outcome"), "tokens": a.get("tokens"), "requests": n,
                         "end_offset_sec": round(last - float(window_start_epoch), 1) if last and window_start_epoch else None,
                         "record_end_offset_sec": a.get("end_offset_sec")})
        ends = [r["end_offset_sec"] for r in rows if r["end_offset_sec"] is not None]
        per[t] = {"finished": bool(v.get("finished")), "solved": bool(v.get("solved")), "finish_offset_sec": max(ends) if ends else None,
                  "first_accept_sec": ((task_solve or {}).get(t) or {}).get("first_accept_sec"), "attempts": rows}
    want = list(task_list or per)
    fo = [x["finish_offset_sec"] for x in per.values() if x["finish_offset_sec"] is not None]
    return {"per_task": per, "tasks": len(per), "tasks_expected": len(want), "tasks_finished": sum(1 for x in per.values() if x["finished"]),
            "all_finished": bool(per) and all(per.get(t, {}).get("finished") for t in want),
            "last_finish_offset_sec": max(fo) if fo else None, "window": window,
            "time_basis": "end of each attempt's last request (nb5_requests.jsonl t1) minus the window start epoch"}


def engine_facts(per_config, phase, boot, sampler_rows=None):
    """The engine facts build_quality_report takes, from NB5's per-config record, the phase record and the boot record
    (the ladder rung = the index of the try that booted)."""
    pc, ph, kb = _d(per_config), _d(phase), _d(boot)
    cl, sp, tr = _d(pc.get("capacity_levers")), _d(pc.get("speed")), _l(kb.get("tries"))
    return dict(pc, pool_blocks=cl.get("pool_blocks"), f_eff=cl.get("f_eff"), workers=cl.get("workers"),
                max_running=sp.get("max_running"), preemptions=sp.get("preemptions"), solver_tok_s=sp.get("solver_tok_s"),
                phase_tok_s=sp.get("phase_tok_s"), steady_p50=_d(ph.get("steady_mixed")).get("gen_tok_s_p50"),
                pin_gib=cl.get("pin_gib"), kv_bytes=cl.get("kv_bytes"), max_num_seqs_launch=cl.get("max_num_seqs_launch"),
                ladder_rung=(len(tr) - 1) if tr else None, steady_mixed=ph.get("steady_mixed"),
                solver_window=ph.get("solver_window"), phase_window=ph.get("phase_window"), counters=ph.get("counters"),
                running_check=ph.get("running_check"), b_summary=ph.get("b_summary"), sampler_rows=sampler_rows,
                tool_call_rate=_d(ph.get("tracer")).get("tool_call_rate"))


def notebook_status(phase, quality, run_report_present=True, missing_gold=()):
    """The one final status of k100_quality_report.json (after SKIPPED_RERUN / NO_PHASE_RUN, set by the cell):
    INCOMPLETE_PHASE (not DONE, a solver error or an error) > MISSING_RUN_REPORT > INVALID_MISSING_GOLD (a task of the list
    without solutions) > INCONCLUSIVE_SCORE_MISMATCH (run report vs generated attempts) > INVALID_QUALITY_REPORT (any other
    report error) > VALID. The report itself is kept whenever it was built."""
    ph, q = _d(phase), _d(quality)
    if ph.get("status") != "DONE" or ph.get("solver_error") or ph.get("error"):
        return {"status": "INCOMPLETE_PHASE", "phase_error": str(ph.get("solver_error") or ph.get("error") or ph.get("status"))[-2000:]}
    if not run_report_present:
        return {"status": "MISSING_RUN_REPORT"}
    errs = [str(x) for x in _l(q.get("errors"))]
    if list(missing_gold or []) or any(x.startswith("MISSING_GOLD:") for x in errs):
        return {"status": "INVALID_MISSING_GOLD", "missing_gold": sorted(set(list(missing_gold or []) + [
            x.split(":", 1)[1] for x in errs if x.startswith("MISSING_GOLD:")]))}
    if _d(_d(q.get("inputs")).get("score_check")).get("agree") is False:
        return {"status": "INCONCLUSIVE_SCORE_MISMATCH"}
    if errs:
        return {"status": "INVALID_QUALITY_REPORT"}
    return {"status": "VALID"}


def review_flags(profile, nb5_pin, phase, per_config):
    """Loud notes: #60490 not verified (the ladder keeps only Merlin's FP8 pin, ~6 workers) and a shortened window
    (window_plan.unequal, or the actual window below the planned fixed window)."""
    P, ph, pc = _d(profile), _d(phase), _d(per_config)
    out = []
    if P.get("patch_60490") and not _d(_d(nb5_pin).get("patch_60490")).get("ok"):
        out.append(f"#60490 NOT VERIFIED: the pin ladder kept only Merlin's FP8 pin ({FP8_BASE_PIN_BYTES:,} B, ~6 workers): speed / "
                   "width are NOT the klee100 capacity reading")
    wp, w = _d(ph.get("window_plan")), _d(pc.get("window"))
    plan, act = float(P.get("fixed_window_sec") or 0), _num(w.get("actual_sec"))
    if wp.get("unequal") or (act is not None and act < plan - 2):
        out.append(f"WINDOW SHORTENED: planned {P.get('fixed_window_sec')} s, actual {w.get('actual_sec')} s (short by "
                   f"{wp.get('short_by_sec') or round(plan - float(act or 0), 1)} s: late boot / ladder rungs)")
    return out


def engine_summary(phase, per_config, finish, quality, qdefer=None):
    """The CELL 9 summary: preemptions, fresh-prefill tokens (server prompt - prefix-cache hit tokens over the phase:
    every uncached prefill, replays after a preemption included), generated tokens, workers, tasks finished per hour of
    the window, the full-window pairing basis, the matched 4,050-s cutoff, and Q-defer (applied, read-back label, exec
    share) with the comparison label: W4 (NB5) ran WITHOUT Q-defer; Q-defer changes timing, not outputs."""
    c, pc, fn, q = _d(_d(phase).get("counters")), _d(per_config), _d(finish), _d(quality)
    pt, ph = _num(c.get("vllm:prompt_tokens_total")), _num(c.get("vllm:prefix_cache_hits_total"))
    wa = _num(_d(pc.get("window")).get("actual_sec"))
    mq = _d(q.get("paired_at_matched_4050s"))
    return {"preemptions": c.get("vllm:num_preemptions_total"), "prompt_tokens": pt, "prefix_hit_tokens": ph,
            "fresh_prefill_tokens": int(pt - ph) if pt is not None and ph is not None else None,
            "fresh_prefill_note": "server prompt tokens - prefix-cache hit tokens over the phase (replays after a preemption included)",
            "generated_tokens": c.get("vllm:generation_tokens_total"), "workers": _d(pc.get("capacity_levers")).get("workers"),
            "speed_status": _d(pc.get("speed")).get("speed_status"), "tasks_finished": fn.get("tasks_finished"),
            "window_actual_sec": wa,
            "tasks_finished_per_hour": round(float(fn["tasks_finished"]) * 3600.0 / wa, 2)
            if wa and fn.get("tasks_finished") is not None else None,
            "paired_window_basis": _d(q.get("paired")).get("window_basis"), "matched_4050s_status": mq.get("status"),
            "matched_4050s_counts": mq.get("counts_at_cutoff"),
            "qdefer": {k: _d(qdefer).get(k) for k in ("applied", "status", "readback_label", "exec_share_min")},
            "comparison": comparison_label(qdefer),
            "note_lines": [comparison_label(qdefer), "Q-defer: applied {applied}, status {status}, read-back {readback_label}, "
                           "exec share {exec_share_min}".format(**{k: _d(qdefer).get(k) for k in (
                               "applied", "status", "readback_label", "exec_share_min")})]}


def comparison_label(qdefer):
    """klee100 + Q-defer vs W4 when the patch was applied (NB5's W4 reference ran without it), else klee100 vs W4."""
    if _d(qdefer).get("applied"):
        return ("COMPARISON: klee100 + Q-defer vs W4 (NB5 QUALITY ran WITHOUT Q-defer): speed / width / window outcomes mix "
                "both changes; the attempt-1 accuracy comparison stays valid (Q-defer changes timing, not outputs)")
    return "COMPARISON: klee100 vs W4 (Q-defer not applied in this run, as in NB5 QUALITY)"


# ------------------------------------------------------------------ markdown
def _fmt_tok(n):
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "?"


def _attempt_cell(a):
    a = _d(a)
    oc = a.get("outcome")
    if not a or oc == "NOT_STARTED":
        return "not started"
    tok = _fmt_tok(a.get("tokens"))
    if a.get("exact_accepted") is True:
        return f"{tok} {MARK_OK}"
    if oc == "INCOMPLETE_WINDOW":
        return f"{tok} incomplete"
    if oc == "FAILED_OR_INTERRUPTED":
        return f"{tok} failed"
    if a.get("exact_accepted") is None and a.get("accepted"):
        return f"{tok} accepted"
    if a.get("accepted"):
        return f"{tok} {MARK_BAD} (accepted)"
    return f"{tok} {MARK_BAD}"


def _task_cell(s):
    s = _d(s)
    if not s.get("present"):
        return "missing"
    q, h = s.get("queries"), s.get("query_hits")
    qtxt = f" {h}/{q}" if q is not None and h is not None else ""
    return (MARK_OK if s.get("solved") else MARK_BAD) + qtxt


def markdown_table(report):
    """task | group | K100 a1 | K100 a2 | K100 | W4 a1 | W4 a2 | W4; attempts = generated tokens + mark (exact accepted
    = check, otherwise cross / incomplete / failed / not started), task cells = solved mark + query hits; a totals row."""
    rep = _d(report)
    pr = _d(rep.get("paired"))
    rows = _d(pr.get("tasks"))
    k, w = LABELS
    out = ["| task | group | K100 a1 | K100 a2 | K100 | W4 a1 | W4 a2 | W4 |", "|---|---|---|---|---|---|---|---|"]
    tot = {k: [0, 0, 0, 0, 0], w: [0, 0, 0, 0, 0]}          # a1 tokens, a2 tokens, solved, query hits, queries
    for t, r in rows.items():
        r = _d(r)
        cells = [str(t), str(r.get("group") or "?")]
        for side in (k, w):
            s = _d(r.get(side))
            cells += [_attempt_cell(s.get("a1")), _attempt_cell(s.get("a2")), _task_cell(s)]
            tt = tot[side]
            tt[0] += int(_d(s.get("a1")).get("tokens") or 0)
            tt[1] += int(_d(s.get("a2")).get("tokens") or 0)
            tt[2] += int(bool(s.get("solved")))
            tt[3] += int(s.get("query_hits") or 0)
            tt[4] += int(s.get("queries") or 0)
        out.append("| " + " | ".join(cells) + " |")
    n = len(rows)
    cells = ["**total**", ""]
    for side in (k, w):
        tt = tot[side]
        cells += [_fmt_tok(tt[0]), _fmt_tok(tt[1]), f"{tt[2]}/{n} tasks, {tt[3]}/{tt[4]} q"]
    out.append("| " + " | ".join(cells) + " |")
    c = _d(pr.get("counts"))
    if c:
        out.append("")
        out.append(f"Paired: both {c.get('both')}, k100_only {c.get(k + '_only')}, w4_only {c.get(w + '_only')}, "
                   f"neither {c.get('neither')}; discordant {c.get('discordant')}, McNemar exact p {c.get('mcnemar_p')} "
                   f"({c.get('reading')}).")
    return "\n".join(out) + "\n"


def markdown_extra(report):
    """E26 30-task tables: attempt 1 only, klee100 vs W4 on NB5's tasks; then every task vs the 27B Duck R3K2 run and
    ARC Prize Qwen3.8-27B xhigh, with per-group solved counts."""
    rep = _d(report)
    k, w = LABELS
    out = []
    a1 = _d(rep.get("attempt1_vs_w4"))
    if _d(a1.get("tasks")):
        out += ["Attempt 1 only (NB5's tasks; both attempt 1s run to completion):", "",
                "| task | group | K100 a1 | K100 a1 end (s) | W4 a1 | label |", "|---|---|---|---|---|---|"]
        for t, r in _d(a1.get("tasks")).items():
            fo = _d(r.get(k)).get("finish_offset_sec")
            out.append(f"| {t} | {r.get('group')} | {_attempt_cell(r.get(k))} | {'?' if fo is None else f'{fo:,.0f}'} | "
                       f"{_attempt_cell(r.get(w))} | {r.get('label')} |")
        n = _d(a1.get("counts"))
        out.append(f"| **total** | | {_fmt_tok(n.get(k + '_a1_tokens'))} | | {_fmt_tok(n.get(w + '_a1_tokens'))} | both {n.get('both')}, "
                   f"k100_only {n.get(k + '_only')}, w4_only {n.get(w + '_only')}, neither {n.get('neither')}; p {n.get('mcnemar_p')} |")
    mc = _d(rep.get("paired_at_matched_4050s"))
    if mc:
        c = _d(mc.get("counts_at_cutoff"))
        out += ["", f"Full-window pairing: {_d(rep.get('paired')).get('window_basis')}. Matched cutoff {mc.get('cutoff_sec')} s "
                    f"(W4 window {mc.get('w4_window_actual_sec')} s): {mc.get('status')}"
                + (f"; both {c.get('both')}, k100_only {c.get(k + '_only')}, w4_only {c.get(w + '_only')}, neither {c.get('neither')}, "
                   f"McNemar p {c.get('mcnemar_p')}" if c else f"; undecided {mc.get('undecided_tasks')}") + "."]
    v27 = _d(rep.get("vs_27b"))
    if _d(v27.get("tasks")):
        out += ["", "All tasks vs 27B Duck R3K2 (50k / 45k attempts) and ARC Prize Qwen3.8-27B xhigh:", "",
                "| task | group | K100 a1 | K100 a2 | K100 | R3K2 b0 | R3K2 b1 | R3K2 | ARC Prize xhigh |", "|---|---|---|---|---|---|---|---|---|"]
        for t, r in _d(v27.get("tasks")).items():
            s, y = _d(r.get(k)), _d(r.get(REF27_LABEL))
            br = {int(b.get("branch") or 0): b for b in _l(y.get("branches"))}
            bc = [(f"{_fmt_tok(br[i].get('generated_tokens'))}{' acc' if br[i].get('accepted') else ''}" if i in br else "-") for i in (0, 1)]
            yt = ((MARK_OK if y.get("solved") else MARK_BAD) + f" {y.get('query_hits')}/{y.get('queries')}") if y.get("present") else "missing"
            ax = {1: MARK_OK, 0: MARK_BAD}.get(r.get("arcprize_xhigh"), "?")
            out.append(f"| {t} | {r.get('group')} | {_attempt_cell(s.get('a1'))} | {_attempt_cell(s.get('a2'))} | {_task_cell(s)} | "
                       f"{bc[0]} | {bc[1]} | {yt} | {ax} |")
        for g, s in _d(v27.get("by_group")).items():
            out.append(f"| **{g}** | | | | {s.get('solved_' + k)}/{s.get('tasks')} | | | {s.get('solved_' + REF27_LABEL)}/{s.get('tasks')} | "
                       f"{s.get('solved_arcprize_xhigh')}/{s.get('tasks')} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":     # offline: python k100q_report.py <phase_dir> <solutions.json> [<engine.json>] [--out r.json]
    import sys as _sys
    _a = _sys.argv[1:]
    if len(_a) >= 2:
        _out = _a[_a.index("--out") + 1] if "--out" in _a else None
        _pos = [x for x in _a if x not in ("--out", _out)]
        _pd = Path(_pos[0])
        _rep = build_quality_report(_pd / "e12_run_report.json", _pd / "e12_generated_attempts.json",
                                    _read_json(_pos[2]) if len(_pos) > 2 else {}, _pos[1], None, phase_dir=_pd)
        print(markdown_table(_rep))
        print("\n".join(_d(_rep.get("notes")).get("lines") or []))
        if _out:
            Path(_out).write_text(json.dumps(_rep, indent=1, ensure_ascii=True), encoding="utf-8", newline="\n")
