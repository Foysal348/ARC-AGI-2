#!/usr/bin/env python3
"""Compare Kaggle evaluation runs of the ARC-AGI-2 Duck REPL agent (E12 rev 6g / E14S / E16), stdlib only.

Usage:
    python compare_eval_runs.py <runA_dir> [<runB_dir> ...] [--labels A,B,...] [--out cmp.json]
                                [--solutions arc-agi_evaluation_solutions.json] [--hidden-tasks 240]
                                [--hidden-window-sec N] [--no-transcripts]

A run dir is a folder of downloaded Kaggle outputs. Files are looked up in the folder tree AND inside every *.zip in it
(Q27X_E12_DUCK_BUNDLE.zip, Q27X_E14A_BUNDLE.zip), preferring q27x/ > run root > bundle/ > other dirs > zip members:
    q27x/e12_run_report.json         score, branches[], rev6 metrics, e12_run (g4b window), usage
    q27x/duck/<task>_b<k>/branch_result.json + transcript.txt
    q27x/e12_generated_attempts.json  eval predictions (re-scored when --solutions is given / found)
    e14_results.json, e14_state.json  build (E16 overlay flag, duck sha), terminal phase
    logs/sglang_server_*.log          SGLang decode/prefill lines (the log that overlaps the solve window is used)
Everything missing is reported as "n/a"; nothing is written except the --out JSON. Markdown goes to stdout.

Definitions (all from the data, nothing assumed):
  started   a branch with rev6 requests_started > 0 or generated_tokens > 0 (a request really sent)
  end class A accepted | C token cap (E12_MAX_GEN_PER_BRANCH) | E error note / failures | P paused, never resumed
            (left waiting for stage 2 at the deadline) | D deadline cut (status done, none of the above; the Duck
            only stops a non-paused branch on accept, cap, deadline or error) | M = D with the in-flight request cut
            (transcript ends in request_timeout: that request's tokens are NOT in generated_tokens)
  finished  A, C or E (the branch ended by itself before the deadline)
  exposure  time of the branch-0 first request (requests[0].t, else t0) minus the window start (min t0 of all branches)
  censored medians: naive = report method (no event -> the branch's generated tokens); KM = Kaplan-Meier median
  LB proxy  task-norm over reached tasks x coverage (estimate; see the projection section)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import statistics
import sys
import zipfile
from pathlib import Path

NA = "n/a"
DEFAULT_SOLUTIONS = [
    Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
         r"E10_POETIQ_R5_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json"),
]
HIDDEN_SOLVE_END_DEFAULT = 42720.0      # build_e12_duck.py gate 4b: RERUN_ALL time budget - margins (H6-16 unit test)


# ----------------------------------------------------------------------------------------------------------- helpers
def g(d, *path, default=None):
    """Nested get that tolerates missing keys, None and wrong types."""
    cur = d
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list) and isinstance(p, int) and -len(cur) <= p < len(cur):
            cur = cur[p]
        else:
            return default
        if cur is None:
            return default
    return cur


def num(x, default=None):
    try:
        if x is None or isinstance(x, bool):
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def pct(xs, q):
    """Nearest-rank-interpolated percentile (q in 0..100)."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def km_median(pairs):
    """Kaplan-Meier median of (time, observed) pairs. Returns (median or None, censored_max)."""
    pairs = [(t, bool(e)) for t, e in pairs if t is not None]
    if not pairs:
        return None, None
    pairs.sort(key=lambda p: (p[0], not p[1]))
    n, s, i = len(pairs), 1.0, 0
    while i < len(pairs):
        t = pairs[i][0]
        d = sum(1 for tt, e in pairs[i:] if tt == t and e)
        c = sum(1 for tt, e in pairs[i:] if tt == t and not e)
        if d:
            s *= 1.0 - d / n
            if s <= 0.5:
                return t, None
        n -= d + c
        i += d + c
    return None, max(t for t, _ in pairs)


def fmt(v, nd=1, unit=""):
    if v is None:
        return NA
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}{unit}"
    if isinstance(v, float):
        if v.is_integer() and abs(v) >= 100:
            return f"{int(v):,}{unit}"
        return f"{v:,.{nd}f}{unit}"
    return f"{v}{unit}"


def fk(v):
    """Tokens in k."""
    return NA if v is None else f"{v / 1000:.1f}k"


def frac(a, b):
    if a is None or b in (None, 0):
        return NA
    return f"{a}/{b} ({100.0 * a / b:.1f} %)"


def ts_utc(epoch):
    if epoch is None:
        return NA
    return _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------------------------------- file locator
class RunFiles:
    """Index of a run folder: plain files + members of every zip in it (read lazily, never extracted)."""

    PRIORITY = ("q27x/", "", "bundle/")

    def __init__(self, root: Path):
        self.root = root
        self.entries = []              # (relpath posix, kind, ref, size)
        self.notes = []
        if not root.is_dir():
            self.notes.append(f"run dir not found: {root}")
            return
        for p in root.rglob("*"):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                self.entries.append((rel, "file", p, p.stat().st_size))
                if p.suffix.lower() == ".zip":
                    try:
                        with zipfile.ZipFile(p) as z:
                            for zi in z.infolist():
                                if not zi.is_dir():
                                    self.entries.append((zi.filename, "zip", (p, zi.filename), zi.file_size))
                    except (zipfile.BadZipFile, OSError) as exc:
                        self.notes.append(f"bad zip {rel}: {exc}")

    def _rank(self, e):
        rel, kind, _, _ = e
        if kind == "zip":
            return 10
        for i, pre in enumerate(self.PRIORITY):
            if pre == "" and "/" not in rel:
                return i
            if pre and rel.startswith(pre) and rel.count("/") == pre.count("/"):
                return i
        return 5 + rel.count("/")

    def find(self, name: str, largest=False):
        c = [e for e in self.entries if e[0].rsplit("/", 1)[-1] == name and e[0].rsplit("/", 1)[-1] != ""]
        if not c:
            return None
        if largest:
            return max(c, key=lambda e: (e[3], -self._rank(e)))
        return min(c, key=self._rank)

    def matching(self, regex: str):
        rx = re.compile(regex)
        return [e for e in self.entries if rx.search(e[0])]

    @staticmethod
    def read_bytes(e):
        if e is None:
            return None
        _, kind, ref, _ = e
        try:
            if kind == "file":
                return ref.read_bytes()
            with zipfile.ZipFile(ref[0]) as z:
                return z.read(ref[1])
        except (OSError, KeyError, zipfile.BadZipFile):
            return None

    def read_text(self, e):
        b = self.read_bytes(e)
        return None if b is None else b.decode("utf-8", errors="replace")

    def read_json(self, e):
        t = self.read_text(e)
        if t is None:
            return None
        try:
            return json.loads(t)
        except ValueError:
            self.notes.append(f"invalid JSON: {e[0]}")
            return None

    @staticmethod
    def where(e):
        if e is None:
            return NA
        return e[0] if e[1] == "file" else f"{e[2][0].name}:{e[0]}"


# --------------------------------------------------------------------------------------------------- server log parse
LOG_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: TP(\d+))?\]")
DECODE_RX = re.compile(r"Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+)"
                       r"(?:, mamba num: (\d+), mamba usage: ([\d.]+))?.*?(?:accept len: ([\d.]+))?"
                       r"(?:, accept rate: ([\d.]+))?.*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
PREFILL_RX = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#running-req: (\d+)"
                        r", #queue-req: (\d+)")
ACCEPT_RX = re.compile(r"accept len: ([\d.]+)")
EXCLUDE_RX = re.compile(r"avoid CUDA OOM|Error in sitecustomize|server_args=|No module named 'wrapt'")
ISSUE_PATTERNS = {
    "oom": re.compile(r"out of memory|OutOfMemory|\bOOM\b", re.I),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "retract": re.compile(r"\bretract", re.I),
    "cuda_error": re.compile(r"CUDA error|cudaError|NCCL error|illegal memory access", re.I),
    "watchdog_or_crash": re.compile(r"watchdog|Segmentation fault|core dumped|Killed|SIGKILL|scheduler.*(died|exception)", re.I),
    "exception": re.compile(r"\b[A-Z][A-Za-z]+(Error|Exception):"),
}


def _epoch(s):
    try:
        return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc).timestamp()
    except ValueError:
        return None


def parse_log(text: str):
    decode, prefill, issues = [], [], {k: [] for k in ISSUE_PATTERNS}
    last_kind = None
    for ln_no, line in enumerate(text.splitlines(), 1):
        m = LOG_TS.match(line)
        ts = _epoch(m.group(1)) if m else None
        tp = m.group(2) if m else None
        if "Decode batch" in line and tp in (None, "0"):
            d = DECODE_RX.search(line)
            if d:
                al = ACCEPT_RX.search(line)
                decode.append({"t": ts, "running": int(d.group(1)), "full_tokens": int(d.group(2)),
                               "kv_usage": float(d.group(3)),
                               "mamba_usage": float(d.group(5)) if d.group(5) else None,
                               "accept_len": float(al.group(1)) if al else None,
                               "gen_tok_s": float(d.group(8)), "queue": int(d.group(9)),
                               "after_prefill": last_kind == "prefill"})
                last_kind = "decode"
                continue
        if "Prefill batch" in line and tp in (None, "0"):
            p = PREFILL_RX.search(line)
            if p:
                prefill.append({"t": ts, "new_seq": int(p.group(1)), "new_tokens": int(p.group(2)),
                                "cached_tokens": int(p.group(3)), "running": int(p.group(4)), "queue": int(p.group(5))})
                last_kind = "prefill"
                continue
        if EXCLUDE_RX.search(line):
            continue
        for k, rx in ISSUE_PATTERNS.items():          # a line can count in several categories (OOM + retract)
            if rx.search(line):
                issues[k].append({"line": ln_no, "t": ts, "text": line.strip()[:260]})
    return {"decode": decode, "prefill": prefill, "issues": issues}


def in_window(t, w):
    return t is not None and w is not None and w[0] - 5 <= t <= w[1] + 5


def engine_summary(parsed, window, streams):
    dec = [d for d in parsed["decode"] if window is None or in_window(d["t"], window)]
    pre = [p for p in parsed["prefill"] if window is None or in_window(p["t"], window)]
    out = {"decode_lines": len(dec), "prefill_lines": len(pre)}
    if not dec:
        return out
    hist = {}
    for d in dec:
        hist[d["running"]] = hist.get(d["running"], 0) + 1
    out["running_hist"] = dict(sorted(hist.items()))
    mx = max(hist)
    out["running_max"] = mx
    out["accept_len_p50"] = median([d["accept_len"] for d in dec])
    out["gen_tok_s_p50_all"] = median([d["gen_tok_s"] for d in dec])

    def at(level, clean=None):
        xs = [d for d in dec if d["running"] == level and (clean is None or d["after_prefill"] != clean)]
        return median([d["gen_tok_s"] for d in xs]), len(xs)

    full = streams if streams else mx
    out["full_concurrency_level"] = full
    out["gen_tok_s_p50_full"], out["lines_full"] = at(full)
    out["gen_tok_s_p50_full_clean"], _ = at(full, clean=True)
    near = [d for d in dec if d["running"] >= full - 1]
    out["gen_tok_s_p50_full_minus1_or_more"] = median([d["gen_tok_s"] for d in near])
    out["lines_full_minus1_or_more"] = len(near)
    out["accept_len_p50_full_minus1_or_more"] = median([d["accept_len"] for d in near])
    out["gen_tok_s_p50_at_max_running"], out["lines_at_max_running"] = at(mx)
    out["share_lines_after_prefill"] = round(sum(1 for d in dec if d["after_prefill"]) / len(dec), 3)
    out["queue_max"] = max(d["queue"] for d in dec)
    out["kv_usage_max"] = max(d["kv_usage"] for d in dec)
    mus = [d["mamba_usage"] for d in dec if d["mamba_usage"] is not None]
    out["mamba_usage_max"] = max(mus) if mus else None
    # tokens generated per the log: throughput x interval since the previous decode line (TP0, 1 s timestamps)
    est, prev = 0.0, None
    for d in dec:
        if prev is not None and d["t"] is not None and prev["t"] is not None and 0 <= d["t"] - prev["t"] <= 120:
            est += d["gen_tok_s"] * (d["t"] - prev["t"])
        prev = d
    out["log_generated_tokens_est"] = int(est)
    if pre:
        new = sum(p["new_tokens"] for p in pre)
        cached = sum(p["cached_tokens"] for p in pre)
        out["prefill_new_tokens"] = new
        out["prefill_cached_tokens"] = cached
        out["prefill_cached_pct_log"] = round(100.0 * cached / max(1, new + cached), 1)
    t0 = min(d["t"] for d in dec if d["t"] is not None)
    t1 = max(d["t"] for d in dec if d["t"] is not None)
    out["decode_span_sec"] = t1 - t0
    return out


# ------------------------------------------------------------------------------------------------------ transcripts
SECTION_RX = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|ANALYZER STATUS|MODEL RESPONSE META|THINKING|ASSISTANT|"
                        r"TOOL CALL: [\w.-]+|TOOL RESULT: [\w.-]+)\]\s*$")
CHECK_RX = re.compile(r"\b(check|check_many|submit)\s*\(")
TRANSFORM_RX = re.compile(r"^\s*def\s+(transform\w*|solve\w*)\s*\(", re.M)


def parse_transcript(text: str):
    if text is None:
        return None
    sections, cur, buf = [], None, []
    for line in text.splitlines():
        m = SECTION_RX.match(line)
        if m:
            if cur is not None:
                sections.append((cur, "\n".join(buf)))
            cur, buf = m.group(1), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        sections.append((cur, "\n".join(buf)))
    calls = [b for s, b in sections if s.startswith("TOOL CALL")]
    statuses = [b.strip().splitlines()[0] if b.strip() else "" for s, b in sections if s == "ANALYZER STATUS"]
    last = statuses[-1] if statuses else ""
    return {"tool_calls": len(calls), "check_calls": sum(1 for b in calls if CHECK_RX.search(b)),
            "transform_defs": sum(1 for b in calls if TRANSFORM_RX.search(b)),
            "last_status": last[:200],
            "ends_in_timeout": bool(re.search(r"request_timeout|Read timed out", last)),
            "thinking_blocks": sum(1 for s, _ in sections if s == "THINKING")}


# ------------------------------------------------------------------------------------------------------ one run
def analyse_run(run_dir: Path, label: str, solutions, use_transcripts=True, hidden_tasks=240, hidden_window=None):
    F = RunFiles(run_dir)
    R = {"label": label, "dir": str(run_dir), "warnings": list(F.notes), "files": {}}
    W = R["warnings"]
    rep_e = F.find("e12_run_report.json")
    rep = F.read_json(rep_e) or {}
    R["files"]["e12_run_report.json"] = F.where(rep_e)
    if not rep:
        W.append("e12_run_report.json missing or unreadable: score/coverage from branch files only")
    res_e, st_e = F.find("e14_results.json"), F.find("e14_state.json")
    res, st = F.read_json(res_e) or {}, F.read_json(st_e) or {}
    R["files"]["e14_results.json"], R["files"]["e14_state.json"] = F.where(res_e), F.where(st_e)
    cfg = g(rep, "configuration", default={})
    e12cfg = g(cfg, "e12", default={})
    r6cfg = g(e12cfg, "rev6", default={})
    run = g(rep, "e12_run", default={})
    g4b = g(run, "g4b") or g(rep, "rev6", "g4b")

    # ---------------------------------------------------------------- meta
    effort = (g(e12cfg, "E12_REASONING_EFFORT") or "").strip() or g(cfg, "REASONING_EFFORT")
    meta = {"run_id": g(rep, "run_id"), "profile": g(rep, "profile"), "rerun": g(rep, "rerun"),
            "effort": effort, "think_budget": g(cfg, "THINK_BUDGET"),
            "build_variant": g(res, "build", "variant"), "duck_sha": (g(res, "build", "arc2_duck_sha")
                                                                    or g(cfg, "e14", "arc2_duck_sha") or "")[:12] or None,
            "e16_overlay": (bool(g(res, "build", "e16_adapter_overlay")) if res else None), "engine": g(rep, "engine", "backend"),
            "submit_arm": g(cfg, "e14", "submit_arm"), "streams": g(run, "streams"), "workers": g(run, "rev6_workers"),
            "kv_tokens": g(rep, "engine", "kv_tokens"), "engine_load_sec": g(rep, "engine", "load_sec"),
            "engine_fatal": g(rep, "engine", "fatal"), "e12_error": g(run, "error"),
            "state_terminal": g(st, "terminal_reason"), "state_current": g(st, "current"), "state_t_end": g(st, "t_end"),
            "g4b": g4b, "deadline_in_sec": g(run, "deadline_in_sec"), "started_sec": g(run, "started_sec"),
            "finished_sec": g(run, "finished_sec"), "window_shift": g(run, "window_shift"),
            "first_exposure_tokens": g(r6cfg, "E12_R6_FIRST_EXPOSURE"), "nudge_tokens_cfg": g(r6cfg, "E12_R6_NUDGE_TOKENS"),
            "max_gen_per_branch": g(e12cfg, "E12_MAX_GEN_PER_BRANCH"), "branches_cfg": g(e12cfg, "E12_BRANCHES")}
    R["meta"] = meta
    if meta["e12_error"]:
        W.append("e12_run.error is set: " + str(meta["e12_error"])[-300:])
    if meta["engine_fatal"]:
        W.append(f"engine.fatal: {meta['engine_fatal']}")
    if st and meta["state_terminal"] not in (None, "done"):
        W.append(f"e14_state terminal_reason = {meta['state_terminal']} (current {meta['state_current']})")

    # ---------------------------------------------------------------- task list
    task_list = g(cfg, "profile_row", "task_list") or g(rep, "order") or []
    recs = g(rep, "eval_unit_records", default=[]) or []
    if not task_list and recs:
        task_list = sorted({r.get("task") for r in recs if r.get("task")})
    branch_recs = {}
    for b in g(rep, "branches", default=[]) or []:
        if isinstance(b, dict) and b.get("task") is not None:
            branch_recs[(b["task"], int(b.get("branch") or 0))] = {"report": b}
    # branch files (dir preferred over zip by RunFiles priority)
    files = {}
    for e in F.matching(r"(^|/)duck/([0-9a-f]{8})_b(\d+)/(branch_result\.json|transcript\.txt)$"):
        m = re.search(r"duck/([0-9a-f]{8})_b(\d+)/([\w.]+)$", e[0])
        key = (m.group(1), int(m.group(2)), m.group(3))
        if key not in files or F._rank(e) < F._rank(files[key]):
            files[key] = e
    for (t, b, name), e in files.items():
        branch_recs.setdefault((t, b), {})
        branch_recs[(t, b)][name] = e
    missing_br = [f"{t}_b{b}" for (t, b), v in branch_recs.items() if "report" in v and "branch_result.json" not in v]
    if missing_br:
        W.append(f"{len(missing_br)} report branches without branch_result.json: {', '.join(sorted(missing_br)[:10])}")
    if not task_list:
        task_list = sorted({t for t, _ in branch_recs})
    tasks_planned = g(rep, "tasks_planned")
    if tasks_planned is not None and tasks_planned != len(task_list):
        W.append(f"tasks_planned {tasks_planned} != task list {len(task_list)}")
    R["n_tasks"] = len(task_list)

    # ---------------------------------------------------------------- score
    sc = {"query_hits": g(rep, "eval_query_hits"), "query_total": g(rep, "eval_query_total"),
          "pass1_hits": g(rep, "eval_pass1_hits"), "pass2_hits": g(rep, "eval_pass2_hits"),
          "task_norm_pct_report": g(rep, "eval_task_norm_pct"), "queries_planned": g(rep, "queries_planned")}
    per_task_hits = {}
    for r in recs:
        t = r.get("task")
        per_task_hits.setdefault(t, []).append(bool(r.get("hit")))
    if recs:
        sc["records"] = len(recs)
        sc["hits_from_records"] = sum(1 for r in recs if r.get("hit"))
        sc["attempt2_only_hits"] = sum(1 for r in recs if r.get("hit") and not r.get("pass1"))
        sc["attempt1_only_hits"] = sum(1 for r in recs if r.get("pass1") and not r.get("pass2"))
        tn = sum(sum(v) / len(v) for v in per_task_hits.values()) / max(1, len(task_list) or len(per_task_hits))
        sc["task_norm_pct"] = round(100 * tn, 2)
        sc["accepted_correct_queries"] = sum(1 for r in recs if (r.get("accepted_programs_correct") or 0) > 0)
        sc["hit_without_accepted_program"] = sum(1 for r in recs if r.get("hit") and not r.get("accepted_programs"))
        if sc["task_norm_pct_report"] is not None and abs(sc["task_norm_pct_report"] - sc["task_norm_pct"]) > 0.02:
            W.append(f"task-norm in report {sc['task_norm_pct_report']} != recomputed {sc['task_norm_pct']}")
        if sc["query_hits"] is not None and sc["query_hits"] != sc["hits_from_records"]:
            W.append(f"eval_query_hits {sc['query_hits']} != records {sc['hits_from_records']}")
        if sc["queries_planned"] is not None and sc["query_total"] is not None and sc["queries_planned"] != sc["query_total"]:
            W.append(f"queries_planned {sc['queries_planned']} != eval_query_total {sc['query_total']}")
    else:
        sc["task_norm_pct"] = sc["task_norm_pct_report"]
        W.append("no eval_unit_records (score n/a unless --solutions re-scoring works)")
    # independent re-score of the eval predictions
    ga_e = F.find("e12_generated_attempts.json")
    ga = F.read_json(ga_e)
    R["files"]["e12_generated_attempts.json"] = F.where(ga_e)
    if solutions and isinstance(ga, dict):
        rh, rt, mism, rtn = 0, 0, [], {}
        for t in task_list:
            qs = ga.get(t)
            gold = solutions.get(t)
            if not isinstance(qs, list) or not isinstance(gold, list):
                continue
            th = []
            for qi, gq in enumerate(gold):
                att = qs[qi] if qi < len(qs) and isinstance(qs[qi], dict) else {}
                hit = att.get("attempt_1") == gq or att.get("attempt_2") == gq
                th.append(hit)
                rh += hit
                rt += 1
                rec = next((r for r in recs if r.get("task") == t and r.get("query") == qi), None)
                if rec is not None and bool(rec.get("hit")) != hit:
                    mism.append(f"{t}#{qi}")
            rtn[t] = th
        sc["rescored_hits"], sc["rescored_total"] = rh, rt
        sc["rescored_task_norm_pct"] = round(100 * sum(sum(v) / len(v) for v in rtn.values()) / max(1, len(task_list)), 2) if rtn else None
        if mism:
            W.append(f"re-score vs eval_unit_records mismatch on {len(mism)} queries: {', '.join(mism[:10])}")
        if not recs and rtn:
            per_task_hits = rtn
            sc["task_norm_pct"] = sc["rescored_task_norm_pct"]
            sc["query_hits"], sc["query_total"] = rh, rt
    R["per_task_hits"] = per_task_hits
    sc["tasks_any_hit"] = sorted(t for t, v in per_task_hits.items() if any(v))
    sc["tasks_full"] = sorted(t for t, v in per_task_hits.items() if v and all(v))
    sc["tasks_partial"] = sorted(t for t, v in per_task_hits.items() if any(v) and not all(v))
    R["score"] = sc

    # ---------------------------------------------------------------- branches
    cap = num(meta["max_gen_per_branch"])
    branches = []
    for (t, b), v in sorted(branch_recs.items()):
        br = F.read_json(v.get("branch_result.json")) if "branch_result.json" in v else None
        rp = v.get("report") or {}
        src = rp or br or {}
        ev = (rp.get("rev6_events") if rp else None) or (br or {}).get("rev6_events") or {}
        gen = src.get("generated_tokens")
        if gen is None and br:
            gen = br.get("generated_tokens")
        gen = int(gen or 0)
        status = src.get("status") or (br or {}).get("status")
        note = src.get("note") or ""
        accepted = bool(src.get("accepted"))          # arc2_accepted: the branch stopped on an accepted submit
        verified = accepted or any(c.get("accepted") for c in ((br or {}).get("checked") or [])
                                   if isinstance(c, dict))   # + a harvested check() program that verified
        reqs = (br or {}).get("requests") or []
        t0 = num((br or {}).get("t0"))
        first_req_t = num(reqs[0].get("t")) if reqs and isinstance(reqs[0], dict) else None
        req_ends = [num(r.get("t")) + num(r.get("sec"), 0.0) for r in reqs if isinstance(r, dict) and num(r.get("t"))]
        tr = None
        if use_transcripts and "transcript.txt" in v:
            tr = parse_transcript(F.read_text(v["transcript.txt"]))
        rs = ev.get("requests_started")
        started = bool((rs or 0) > 0 or gen > 0 or (rs is None and (reqs or (tr and tr["tool_calls"]))))
        if not started:
            cls = "-"
        elif accepted:
            cls = "A"
        elif cap and gen >= cap:
            cls = "C"
        elif note:                      # analyzer failures / exception / setup failure (run_branch sets a note)
            cls = "E"
        elif status == "paused":
            cls = "P"
        else:
            cls = "M" if (tr and tr["ends_in_timeout"]) else "D"
        usage = rp.get("usage") if isinstance(rp.get("usage"), dict) else None
        runs = src.get("runs") or []
        branches.append({
            "task": t, "branch": b, "status": status, "class": cls, "started": started, "generated": gen,
            "accepted": accepted, "verified": verified, "note": note, "t0": t0, "first_req_t": first_req_t,
            "last_req_end": max(req_ends) if req_ends else None, "requests_started": rs,
            "stage2_class": src.get("stage2_class"), "view": src.get("view"),
            "runs": [{"stage": r.get("stage"), "pause_at": r.get("pause_at"), "tokens": r.get("tokens"),
                      "sec": r.get("sec"), "status": r.get("status")} for r in runs if isinstance(r, dict)],
            "first_check": ev.get("first_check_tokens"), "first_transform": ev.get("first_transform_tokens"),
            "first_tool": ev.get("first_tool_tokens"),
            "progress_runs": g(src, "progress", "runs") or g(br, "progress", "runs"),
            "checks_signalled": g(src, "rev5_events", "checks_signalled"),
            "nudge_tokens": ev.get("provisional_nudge_tokens"), "nudge_via": ev.get("provisional_nudge_via"),
            "nudge_reason": ev.get("e16_nudge_reason"), "nudge_seen_tokens": ev.get("provisional_nudge_seen_tokens"),
            "transform_after_nudge": ev.get("first_transform_after_nudge"),
            "check_after_nudge": ev.get("first_check_after_nudge"),
            "e16_present": "e16_inspection_calls" in ev, "e16_inspection_calls": ev.get("e16_inspection_calls"),
            "e16_raw": {k: ev.get(k) for k in ("e16_raw_content_kept", "e16_raw_content_same",
                                               "e16_raw_content_mismatch", "e16_raw_content_markup") if k in ev},
            "raw_reasoning": (br or {}).get("raw_reasoning"),
            "tool_calls": ev.get("tool_calls"), "length_cut_no_call": ev.get("length_cut_no_call"),
            "infra_errors": ev.get("infra_errors"), "cached_pct": (usage or {}).get("cached_pct"),
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": (usage or {}).get("cached_tokens"),
            "transcript": tr})
    R["branches"] = branches

    # ---------------------------------------------------------------- coverage
    by_task = {t: [x for x in branches if x["task"] == t] for t in task_list}
    extra = sorted({x["task"] for x in branches} - set(task_list))
    if extra:
        W.append(f"branches for tasks outside the task list: {', '.join(extra[:10])}")
    b0 = {t: next((x for x in v if x["branch"] == 0), None) for t, v in by_task.items()}
    b1 = {t: next((x for x in v if x["branch"] == 1), None) for t, v in by_task.items()}
    fin = ("A", "C", "E")
    cov = {"tasks": len(task_list),
           "b0_started": sum(1 for x in b0.values() if x and x["started"]),
           "b0_finished": sum(1 for x in b0.values() if x and x["class"] in fin),
           "b1_started": sum(1 for x in b1.values() if x and x["started"]),
           "b1_finished": sum(1 for x in b1.values() if x and x["class"] in fin),
           "b1_started_zero_tokens": sum(1 for x in b1.values() if x and x["started"] and x["generated"] == 0),
           "branches_started": sum(1 for x in branches if x["started"]),
           "branches_started_report": g(run, "branches_started"), "branches_done_report": g(run, "branches_done"),
           "branches_accepted_report": g(run, "branches_accepted"),
           "tasks_exposed_report": g(rep, "rev6", "metrics", "tasks_exposed"),
           "tasks_with_response_report": g(rep, "rev6", "metrics", "tasks_with_response"),
           "tasks_with_request_sent_report": g(rep, "rev6", "scheduler", "tasks_with_request_sent")}
    cov["never_reached"] = sorted(t for t in task_list if not (b0[t] and b0[t]["started"]))
    cls_count = {}
    for x in branches:
        if x["started"]:
            k = f"b{x['branch']}:{x['class']}"
            cls_count[k] = cls_count.get(k, 0) + 1
    cov["end_classes"] = dict(sorted(cls_count.items()))
    cov["cut_by_deadline"] = sum(1 for x in branches if x["started"] and x["class"] in ("P", "D", "M"))
    cov["cut_mid_request"] = sum(1 for x in branches if x["class"] == "M")
    cov["paused_unresumed"] = sum(1 for x in branches if x["class"] == "P")
    fe = num(meta["first_exposure_tokens"])
    cov["b0_reached_first_exposure"] = (sum(1 for x in b0.values() if x and (x["class"] in fin or
                                                                               (fe and x["generated"] >= fe)))
                                        if fe else None)
    t0s = [x["t0"] for x in branches if x["t0"]]
    wstart = min(t0s) if t0s else None
    ends = [x["last_req_end"] for x in branches if x["last_req_end"]]
    planned = num(g(g4b, "window_sec")) or num(meta["deadline_in_sec"])
    cov["window_start_utc"] = ts_utc(wstart)
    cov["window_planned_sec"] = planned
    cov["window_observed_sec"] = round(max(ends) - wstart, 1) if (ends and wstart) else None
    deadline = (wstart + num(meta["deadline_in_sec"])) if (wstart and num(meta["deadline_in_sec"])) else None
    cov["deadline_utc"] = ts_utc(deadline)
    expo = {}
    for t, x in b0.items():
        if x and x["started"]:
            tt = x["first_req_t"] or x["t0"]
            if tt and wstart:
                expo[t] = round(tt - wstart, 1)
    cov["exposure_last_task_sec"] = max(expo.values()) if expo else None
    cov["exposure_last_task"] = max(expo, key=expo.get) if expo else None
    cov["exposure_p50_sec"] = pct(list(expo.values()), 50)
    cov["exposure_p90_sec"] = pct(list(expo.values()), 90)
    if expo and planned:
        cov["exposure_last_task_frac_of_window"] = round(cov["exposure_last_task_sec"] / planned, 3)
    tok_task = {t: sum(x["generated"] for x in v) for t, v in by_task.items()}
    reached_tok = [tok_task[t] for t in task_list if t not in cov["never_reached"]]
    cov["tokens_per_task_p10"], cov["tokens_per_task_p50"], cov["tokens_per_task_p90"] = (
        pct(reached_tok, 10), pct(reached_tok, 50), pct(reached_tok, 90))
    cov["tokens_per_task_mean"] = (sum(reached_tok) / len(reached_tok)) if reached_tok else None
    cov["total_generated"] = sum(x["generated"] for x in branches)
    cov["completion_tokens_report"] = g(rep, "usage_all", "completion_tokens")
    if cov["completion_tokens_report"] is not None and cov["completion_tokens_report"] != cov["total_generated"]:
        W.append(f"usage_all.completion_tokens {cov['completion_tokens_report']} != sum of branch generated "
                 f"{cov['total_generated']}")
    win = cov["window_observed_sec"] or planned
    cov["eff_tok_s_window"] = round(cov["total_generated"] / win, 1) if win else None
    cov["eff_tok_s_planned"] = round(cov["total_generated"] / planned, 1) if planned else None
    # in-flight seconds lost at the deadline (M branches): deadline - last completed request end (or t0)
    lost = []
    for x in branches:
        if x["class"] == "M" and deadline:
            st0 = x["last_req_end"] or x["t0"]
            if st0:
                lost.append(max(0.0, deadline - st0))
    cov["inflight_cut_sec_sum"] = round(sum(lost), 1) if lost else 0.0
    if cov["branches_started_report"] is not None and cov["branches_started_report"] != cov["branches_started"]:
        W.append(f"e12_run.branches_started {cov['branches_started_report']} != started branches found "
                 f"{cov['branches_started']} (definition: request sent)")
    if cov["b1_started_zero_tokens"]:
        W.append(f"{cov['b1_started_zero_tokens']} branch-1 runs were started but count 0 generated tokens (their only "
                 f"request was cut at the deadline: in-flight tokens are not counted)")
    lowtok = [f"{x['task']}_b{x['branch']}" for x in branches if x["started"] and 0 < x["generated"] < 1000 and x["class"] in ("M", "D")]
    if lowtok:
        W.append(f"{len(lowtok)} deadline-cut branches with < 1k counted tokens (in-flight request lost): {', '.join(lowtok[:12])}")
    R["coverage"] = cov
    R["tokens_per_task"] = tok_task
    R["exposure"] = expo

    # ---------------------------------------------------------------- behaviour
    sb = [x for x in branches if x["started"]]
    beh = {"branches_started": len(sb)}

    def checked(x):
        return x["first_check"] is not None or (x["progress_runs"] or 0) > 0 or (x["checks_signalled"] or 0) > 0

    beh["checked_branches"] = sum(1 for x in sb if checked(x))
    tr_b = [x for x in sb if x["transcript"]]
    beh["transcripts"] = len(tr_b)
    beh["checked_branches_transcript"] = sum(1 for x in tr_b if x["transcript"]["check_calls"] > 0)
    beh["transform_def_branches_transcript"] = sum(1 for x in tr_b if x["transcript"]["transform_defs"] > 0)
    beh["tool_calls_p50"] = median([x["tool_calls"] for x in sb])
    for key in ("first_transform", "first_check"):
        for bn, sel in (("all", sb), ("b0", [x for x in sb if x["branch"] == 0]), ("b1", [x for x in sb if x["branch"] == 1])):
            obs = [x[key] for x in sel if x[key] is not None]
            cens = [x[key] if x[key] is not None else x["generated"] for x in sel]
            km, cmax = km_median([(x[key] if x[key] is not None else x["generated"], x[key] is not None) for x in sel])
            beh[f"{key}_{bn}"] = {"n": len(sel), "observed_n": len(obs), "median_observed": median(obs),
                                  "median_censored_naive": median(cens), "median_km": km,
                                  "km_not_reached_max": cmax, "within_20k": sum(1 for v in obs if v <= 20000)}
    rep_m = g(rep, "rev6", "metrics", default={})
    for k_rep, k_me in (("first_check_tokens_median_censored", ("first_check_all", "median_censored_naive")),
                        ("first_transform_tokens_median_censored", ("first_transform_all", "median_censored_naive"))):
        a, b_ = rep_m.get(k_rep), beh[k_me[0]][k_me[1]]
        if a is not None and b_ is not None and abs(float(a) - float(b_)) > 0.6:
            W.append(f"rev6.metrics.{k_rep} {a} != recomputed {b_}")
    nudge_cfg = num(meta["nudge_tokens_cfg"]) or 24000
    nud = [x for x in sb if x["nudge_tokens"] is not None]

    def reason(x):
        if x["nudge_reason"] == "inspection":
            return "e16_early"
        if x["nudge_reason"] == "existing_24k":
            return "24k"
        return "24k" if x["nudge_tokens"] >= nudge_cfg else "early(no reason field)"

    solved_tasks = set(sc["tasks_any_hit"])
    nb = {}
    for x in nud:
        r_ = reason(x)
        d = nb.setdefault(r_, {"branches": 0, "tokens": [], "seen": 0, "via": {}, "transform_after": 0,
                               "check_after": 0, "accepted": 0, "verified": 0, "task_solved": 0,
                               "tokens_to_transform": []})
        d["branches"] += 1
        d["tokens"].append(x["nudge_tokens"])
        d["seen"] += x["nudge_seen_tokens"] is not None
        d["via"][x["nudge_via"] or NA] = d["via"].get(x["nudge_via"] or NA, 0) + 1
        ta = x["transform_after_nudge"]
        if ta is None and x["first_transform"] is not None and x["first_transform"] >= x["nudge_tokens"]:
            ta = x["first_transform"]
        if ta is not None:
            d["transform_after"] += 1
            d["tokens_to_transform"].append(ta - x["nudge_tokens"])
        d["check_after"] += x["check_after_nudge"] is not None
        d["accepted"] += x["accepted"]
        d["verified"] += x["verified"]
        d["task_solved"] += x["task"] in solved_tasks
    for d in nb.values():
        d["tokens_p50"] = median(d.pop("tokens"))
        d["tokens_to_transform_p50"] = median(d.pop("tokens_to_transform"))
    beh["nudged_branches"] = len(nud)
    beh["nudges"] = nb
    beh["verified_branches"] = sum(1 for x in sb if x["verified"])
    rn = g(rep_m, "nudge", "branches")
    if rn is not None and rn != len(nud):
        W.append(f"rev6.metrics.nudge.branches {rn} != nudged branches found {len(nud)}")
    rc = rep_m.get("checked_branches")
    if rc is not None and rc != beh["checked_branches"]:
        W.append(f"rev6.metrics.checked_branches {rc} != recomputed {beh['checked_branches']}")
    if beh["checked_branches"] != beh["checked_branches_transcript"] and beh["transcripts"] == len(sb):
        W.append(f"check() callers: events {beh['checked_branches']} vs transcripts {beh['checked_branches_transcript']}")
    beh["nudge_eligible_not_nudged"] = sum(1 for x in sb if x["nudge_tokens"] is None and x["first_transform"] is None
                                           and x["generated"] >= nudge_cfg and not x["progress_runs"])
    e16p = [x for x in sb if x["e16_present"]]
    beh["e16_fields_present_branches"] = len(e16p)
    if e16p:
        ic = [x["e16_inspection_calls"] or 0 for x in e16p]
        beh["e16_inspection_calls_hist"] = {str(k): ic.count(k) for k in sorted(set(ic))}
        beh["e16_raw_content"] = {k: sum(int(x["e16_raw"].get(k) or 0) for x in e16p)
                                  for k in ("e16_raw_content_kept", "e16_raw_content_same",
                                            "e16_raw_content_mismatch", "e16_raw_content_markup")}
    rr = [x["raw_reasoning"] for x in sb if isinstance(x["raw_reasoning"], dict)]
    if rr:
        beh["raw_reasoning"] = {k: sum(int(d.get(k) or 0) for d in rr) for k in ("kept", "same", "mismatch", "markup")}
    beh["length_cut_no_call"] = sum(int(x["length_cut_no_call"] or 0) for x in sb)
    beh["infra_errors"] = sum(int(x["infra_errors"] or 0) for x in sb)
    if meta["e16_overlay"] and not e16p:
        W.append("e14_results says E16 overlay but no branch has e16_* fields")
    if not meta["e16_overlay"] and any(x["nudge_reason"] == "inspection" for x in sb):
        W.append("E16 'inspection' nudges found in a run without the E16 overlay flag")
    if e16p and meta["e16_overlay"] and not nb.get("e16_early") and len(sb) >= 10:
        W.append("E16 run without a single early (inspection) nudge")
    R["behaviour"] = beh

    # ---------------------------------------------------------------- engine
    eng = {"report_decode_summary": g(rep, "e11", "decode_summary"),
           "report_server_log": g(rep, "e11", "server_log"),
           "cached_pct_report": g(rep, "usage_all", "cached_pct"),
           "cached_pct_branch_p50": median([x["cached_pct"] for x in sb]),
           "uncached_prompt_tokens_report": g(rep_m, "uncached_prompt_tokens"),
           "accept_len_median_report": g(rep_m, "accept_len_median"),
           "real_running_max_report": g(rep_m, "real_running_max")}
    logs = {}
    for e in F.matching(r"sglang_server_[^/]*\.log$"):
        name = e[0].rsplit("/", 1)[-1]
        if name not in logs or e[3] > logs[name][3]:
            logs[name] = e
    win_w = (wstart, max(ends) if ends else (deadline or wstart)) if wstart else None
    parsed_logs, best, best_n = {}, None, -1
    for name, e in sorted(logs.items()):
        txt = F.read_text(e)
        if txt is None:
            continue
        p = parse_log(txt)
        parsed_logs[name] = (e, p)
        n_in = sum(1 for d in p["decode"] if in_window(d["t"], win_w)) if win_w else len(p["decode"])
        if n_in > best_n:
            best, best_n = name, n_in
    eng["logs_found"] = sorted(logs)
    if best is not None:
        e, p = parsed_logs[best]
        eng["log_used"] = F.where(e)
        use_window = win_w if best_n > 0 else None
        if win_w and best_n <= 0:
            W.append(f"no decode line of {best} falls in the solve window: whole log used")
        eng["log_window_filtered"] = use_window is not None
        eng.update(engine_summary(p, use_window, meta["streams"]))
        rsl = (eng["report_server_log"] or "").rsplit("/", 1)[-1]
        if rsl and rsl != best:
            W.append(f"report e11.server_log is {rsl} but the solve window overlaps {best}")
        if eng.get("log_generated_tokens_est") and cov["total_generated"]:
            eng["counted_vs_log_tokens"] = round(cov["total_generated"] / eng["log_generated_tokens_est"], 3)
            if eng["counted_vs_log_tokens"] < 0.8:
                W.append(f"branch generated tokens are {100 * eng['counted_vs_log_tokens']:.0f} % of the log's decode "
                         f"estimate ({cov['total_generated']:,} vs ~{eng['log_generated_tokens_est']:,}): "
                         f"tokens of deadline-cut requests are not counted")
        newp = eng.get("prefill_new_tokens")
        if newp is not None:
            gen_ref = eng.get("log_generated_tokens_est") or cov["total_generated"]
            eng["prefill_share_tokens"] = round(newp / max(1, newp + gen_ref), 3)
    else:
        W.append("no sglang_server_*.log found")
    issues = {}
    for name, (e, p) in parsed_logs.items():
        for k, lst in p["issues"].items():
            if not lst:
                continue
            d = issues.setdefault(k, {"count": 0, "in_window": 0, "samples": []})
            d["count"] += len(lst)
            d["in_window"] += sum(1 for it in lst if in_window(it["t"], win_w))
            for it in lst[: max(0, 4 - len(d["samples"]))]:
                d["samples"].append(f"{name}:{it['line']}: {it['text']}")
    # kernel / notebook logs (not sglang)
    eng["kernel_logs_scanned"] = 0
    for e in F.entries:
        nm = e[0].rsplit("/", 1)[-1]
        if e[1] == "file" and "/" not in e[0] and (nm.endswith(".log") or nm == "kernel_log.txt") \
                and not nm.startswith("sglang_server") and "heartbeat" not in nm:
            txt = F.read_text(e) or ""
            eng["kernel_logs_scanned"] += 1
            tb = [ln for ln in txt.splitlines() if "Traceback (most recent call last)" in ln and not EXCLUDE_RX.search(ln)]
            if tb:
                d = issues.setdefault("kernel_log_traceback", {"count": 0, "in_window": None, "samples": []})
                d["count"] += len(tb)
                d["samples"].append(f"{nm}: {len(tb)} Traceback line(s)")
    eng["issues"] = issues
    ds = eng["report_decode_summary"] or {}
    if (ds.get("retract_lines") or 0) > 0 and not issues.get("retract"):
        W.append(f"report e11.decode_summary.retract_lines = {ds.get('retract_lines')} but the log has no real retract "
                 f"line (the counter matches 'retraction_policy' in server_args)")
    R["engine"] = eng

    # ---------------------------------------------------------------- projection
    proj = {"hidden_tasks": hidden_tasks}
    reached = len(task_list) - len(cov["never_reached"])
    tn_all = sc.get("task_norm_pct")
    if tn_all is not None and reached:
        hit_reached = sum(sum(v) / len(v) for t, v in per_task_hits.items() if t not in cov["never_reached"] and v)
        proj["task_norm_reached_pct"] = round(100 * hit_reached / reached, 2)
    cov_exact = reached / len(task_list) if task_list else None
    proj["coverage_started"] = round(cov_exact, 3) if task_list else None
    proj["coverage_meaningful"] = (round(cov["b0_reached_first_exposure"] / len(task_list), 3)
                                   if task_list and cov["b0_reached_first_exposure"] is not None else None)
    hw = hidden_window
    if hw is None:
        if g4b and num(g4b.get("hidden_solve_end_sec")) and num(g4b.get("solve_start_sec")):
            hw = num(g4b["hidden_solve_end_sec"]) - num(g4b["solve_start_sec"])
        elif num(meta["started_sec"]):
            hw = HIDDEN_SOLVE_END_DEFAULT - num(meta["started_sec"])
    proj["hidden_window_sec"] = hw
    proj["hidden_sec_per_task"] = round(hw / hidden_tasks, 1) if hw else None
    proj["eval_sec_per_task"] = round(planned / len(task_list), 1) if planned and task_list else None
    proj["g4b"] = bool(g4b)
    if g4b:
        proj["coverage_hidden_est"] = cov_exact
        proj["method"] = ("gate 4b: the eval run had the hidden run's seconds per task, so its coverage is the hidden "
                          "estimate; LB proxy = task-norm(reached) x coverage (= the all-task task-norm)")
    elif cov["eff_tok_s_window"] and cov["tokens_per_task_mean"] and hw:
        need = hidden_tasks * cov["tokens_per_task_mean"]
        proj["coverage_hidden_est"] = round(min(1.0, cov["eff_tok_s_window"] * hw / need), 3)
        proj["method"] = ("not gate 4b: hidden coverage = min(1, eff tok/s x hidden window / (240 x mean tokens per "
                          "reached task)); the eval run's tokens per task were shaped by its own window (biased)")
    else:
        proj["coverage_hidden_est"] = None
        proj["method"] = NA
    if proj.get("task_norm_reached_pct") is not None and proj["coverage_hidden_est"] is not None:
        proj["lb_proxy_pct"] = round(proj["task_norm_reached_pct"] * proj["coverage_hidden_est"], 2)
        proj["coverage_hidden_est"] = round(proj["coverage_hidden_est"], 3)
    else:
        proj["lb_proxy_pct"] = None
    R["projection"] = proj
    return R


# ------------------------------------------------------------------------------------------------------ comparison
def compare(runs):
    labels = [r["label"] for r in runs]
    tasks = []
    for r in runs:
        for t in r["per_task_hits"]:
            if t not in tasks:
                tasks.append(t)
        for x in r["branches"]:
            if x["task"] not in tasks:
                tasks.append(x["task"])
    tasks.sort()
    C = {"labels": labels, "tasks": len(tasks)}
    # union by query (OR across runs); the query count per task = max over runs
    u_hits, u_total, u_tn = 0, 0, 0.0
    for t in tasks:
        qn = max((len(r["per_task_hits"].get(t, [])) for r in runs), default=0)
        if qn == 0:
            continue
        hits = [any((r["per_task_hits"].get(t) or [False] * qn)[q] if q < len(r["per_task_hits"].get(t) or []) else False
                    for r in runs) for q in range(qn)]
        u_hits += sum(hits)
        u_total += qn
        u_tn += sum(hits) / qn
    denom = max((r["n_tasks"] for r in runs), default=0) or len(tasks)
    C["union_query_hits"], C["union_query_total"] = u_hits, u_total
    C["union_task_norm_pct"] = round(100 * u_tn / max(1, denom), 2)
    sets = {r["label"]: set(r["score"]["tasks_any_hit"]) for r in runs}
    C["union_tasks_any_hit"] = sorted(set().union(*sets.values())) if sets else []
    C["intersection_tasks_any_hit"] = sorted(set.intersection(*sets.values())) if sets else []
    pw = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = sets[labels[i]], sets[labels[j]]
            pw.append({"a": labels[i], "b": labels[j], "both": len(a & b), "only_a": sorted(a - b),
                       "only_b": sorted(b - a), "union": len(a | b),
                       "jaccard": round(len(a & b) / len(a | b), 3) if (a | b) else None})
    C["pairwise"] = pw
    C["only_in"] = {lab: sorted(s - set().union(*(sets[o] for o in labels if o != lab))) for lab, s in sets.items()}
    return C, tasks


# ------------------------------------------------------------------------------------------------------ markdown
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render(runs, C, tasks):
    L = []
    P = L.append
    labels = [r["label"] for r in runs]
    P("# Duck eval run comparison")
    P("")
    P("Generated " + _dt.datetime.now().strftime("%Y-%m-%d %H:%M") + " by compare_eval_runs.py. "
      "'n/a' = field or file missing. Estimates are labelled.")
    P("")
    for r in runs:
        P(f"- **{r['label']}**: `{r['dir']}`")
    P("")

    def row(name, fn):
        vals = []
        for r in runs:
            try:
                vals.append(fn(r))
            except Exception:              # one bad field must never break the table
                vals.append(NA)
        return [name] + vals

    m = lambda r: r["meta"]
    s = lambda r: r["score"]
    c = lambda r: r["coverage"]
    b = lambda r: r["behaviour"]
    e = lambda r: r["engine"]
    pj = lambda r: r["projection"]

    P("## 1. Run setup")
    P(md_table(["", *labels], [
        row("profile / run id", lambda r: f"{m(r)['profile'] or NA} / {m(r)['run_id'] or NA}"),
        row("build variant", lambda r: fmt(m(r)["build_variant"])),
        row("reasoning effort", lambda r: fmt(m(r)["effort"])),
        row("E16 overlay", lambda r: fmt(m(r)["e16_overlay"])),
        row("duck sha", lambda r: fmt(m(r)["duck_sha"])),
        row("engine", lambda r: fmt(m(r)["engine"])),
        row("streams / workers", lambda r: f"{fmt(m(r)['streams'])} / {fmt(m(r)['workers'])}"),
        row("KV tokens", lambda r: fmt(m(r)["kv_tokens"])),
        row("engine load s", lambda r: fmt(m(r)["engine_load_sec"])),
        row("g4b sec/task", lambda r: fmt(g(m(r)["g4b"], "sec_per_task"))),
        row("g4b window s", lambda r: fmt(g(m(r)["g4b"], "window_sec"))),
        row("solve start s (notebook)", lambda r: fmt(m(r)["started_sec"])),
        row("deadline in s", lambda r: fmt(m(r)["deadline_in_sec"])),
        row("e14_state terminal", lambda r: f"{fmt(m(r)['state_terminal'])} ({fmt(m(r)['state_current'])})"),
        row("e12_run.error", lambda r: "none" if not m(r)["e12_error"] else "SET"),
    ]))
    P("")
    P("## 2. Score")
    P(md_table(["", *labels], [
        row("query hits", lambda r: (f"{s(r)['query_hits']}/{s(r)['query_total']}"
                                     if s(r)["query_hits"] is not None else NA)),
        row("pass1 (attempt_1 hits)", lambda r: fmt(s(r)["pass1_hits"])),
        row("pass2 (attempt_2 hits)", lambda r: fmt(s(r)["pass2_hits"])),
        row("attempt_2-only hits (contribution)", lambda r: fmt(s(r).get("attempt2_only_hits"))),
        row("task-norm % (report)", lambda r: fmt(s(r)["task_norm_pct_report"], 2)),
        row("task-norm % (recomputed)", lambda r: fmt(s(r).get("task_norm_pct"), 2)),
        row("re-scored hits (solutions)", lambda r: (f"{s(r)['rescored_hits']}/{s(r)['rescored_total']}"
                                                     if s(r).get("rescored_hits") is not None else NA)),
        row("tasks with >= 1 hit", lambda r: fmt(len(s(r)["tasks_any_hit"]))),
        row("tasks fully solved", lambda r: fmt(len(s(r)["tasks_full"]))),
        row("hit queries w/o accepted program", lambda r: fmt(s(r).get("hit_without_accepted_program"))),
    ]))
    P("")
    for r in runs:
        P(f"- **{r['label']}** solved ({len(r['score']['tasks_any_hit'])}): "
          + (", ".join(t + ("*" if t in r["score"]["tasks_partial"] else "") for t in r["score"]["tasks_any_hit"]) or "none")
          + ("  (* = partial)" if r["score"]["tasks_partial"] else ""))
    P("")
    P("## 3. Coverage")
    P(md_table(["", *labels], [
        row("tasks", lambda r: fmt(c(r)["tasks"])),
        row("b0 started", lambda r: frac(c(r)["b0_started"], c(r)["tasks"])),
        row("b0 finished (A/C/E)", lambda r: frac(c(r)["b0_finished"], c(r)["tasks"])),
        row("b0 got first exposure budget", lambda r: frac(c(r)["b0_reached_first_exposure"], c(r)["tasks"])),
        row("b1 started", lambda r: frac(c(r)["b1_started"], c(r)["tasks"])),
        row("b1 started, 0 tokens counted", lambda r: fmt(c(r)["b1_started_zero_tokens"])),
        row("b1 finished (A/C/E)", lambda r: frac(c(r)["b1_finished"], c(r)["tasks"])),
        row("tasks never reached", lambda r: fmt(len(c(r)["never_reached"]))),
        row("branches started (found / report)", lambda r: f"{c(r)['branches_started']} / {fmt(c(r)['branches_started_report'])}"),
        row("tasks with response (report)", lambda r: fmt(c(r)["tasks_with_response_report"])),
        row("end classes", lambda r: ", ".join(f"{k}={v}" for k, v in c(r)["end_classes"].items()) or NA),
        row("branches cut by deadline (P+D+M)", lambda r: fmt(c(r)["cut_by_deadline"])),
        row(".. paused never resumed (P)", lambda r: fmt(c(r)["paused_unresumed"])),
        row(".. in-flight request cut (M)", lambda r: fmt(c(r)["cut_mid_request"])),
        row("in-flight seconds cut (M)", lambda r: fmt(c(r)["inflight_cut_sec_sum"])),
        row("window start (UTC)", lambda r: c(r)["window_start_utc"]),
        row("window planned / observed s", lambda r: f"{fmt(c(r)['window_planned_sec'])} / {fmt(c(r)['window_observed_sec'])}"),
        row("first exposure of last task s", lambda r: f"{fmt(c(r)['exposure_last_task_sec'])} ({fmt(c(r)['exposure_last_task'])})"),
        row(".. as share of window", lambda r: fmt(c(r).get("exposure_last_task_frac_of_window"), 3)),
        row("exposure p50 / p90 s", lambda r: f"{fmt(c(r)['exposure_p50_sec'])} / {fmt(c(r)['exposure_p90_sec'])}"),
        row("tokens/task p10/p50/p90", lambda r: f"{fk(c(r)['tokens_per_task_p10'])} / {fk(c(r)['tokens_per_task_p50'])} / "
                                                 f"{fk(c(r)['tokens_per_task_p90'])}"),
        row("total generated (counted)", lambda r: fmt(c(r)["total_generated"])),
        row("log-estimated generated (window)", lambda r: fmt(e(r).get("log_generated_tokens_est"))),
        row("counted / log tokens", lambda r: fmt(e(r).get("counted_vs_log_tokens"), 3)),
        row("eff tok/s (counted / observed window)", lambda r: fmt(c(r)["eff_tok_s_window"])),
        row("eff tok/s (counted / planned window)", lambda r: fmt(c(r)["eff_tok_s_planned"])),
    ]))
    P("")
    P("End classes: A accepted (stopped on an accepted submit), C token cap, E error, P paused never resumed, D deadline cut, M deadline cut mid-request "
      "(transcript ends in request_timeout; its tokens are not counted).")
    P("")
    for r in runs:
        nr = r["coverage"]["never_reached"]
        P(f"- **{r['label']}** never reached ({len(nr)}): {', '.join(nr) if nr else 'none'}")
    P("")
    P("## 4. Behaviour")

    def med_cell(k):
        def f(r):
            d = b(r)[k]
            km = fmt(d["median_km"]) if d["median_km"] is not None else (f">{fmt(d['km_not_reached_max'])}"
                                                                          if d["km_not_reached_max"] is not None else NA)
            return (f"obs {fmt(d['median_observed'])} (n={d['observed_n']}/{d['n']}) · naive-cens "
                    f"{fmt(d['median_censored_naive'])} · KM {km} · <=20k {d['within_20k']}")
        return f

    def nud_cell(key):
        def f(r):
            d = b(r)["nudges"].get(key)
            if not d:
                return "0"
            return (f"{d['branches']} (tok p50 {fmt(d['tokens_p50'])}, seen {d['seen']}, via "
                    f"{'/'.join(f'{k}:{v}' for k, v in d['via'].items())}) -> transform {d['transform_after']} "
                    f"(+{fmt(d['tokens_to_transform_p50'])} tok p50), check {d['check_after']}, accepted "
                    f"{d['accepted']}, verified {d['verified']}, task solved {d['task_solved']}")
        return f

    P(md_table(["", *labels], [
        row("branches started", lambda r: fmt(b(r)["branches_started"])),
        row("ever called check() (events)", lambda r: frac(b(r)["checked_branches"], b(r)["branches_started"])),
        row("ever called check() (transcripts)", lambda r: frac(b(r)["checked_branches_transcript"], b(r)["transcripts"])),
        row("defined transform (transcripts)", lambda r: frac(b(r)["transform_def_branches_transcript"], b(r)["transcripts"])),
        row("tool calls / branch p50", lambda r: fmt(b(r)["tool_calls_p50"])),
        row("branches with a verified program", lambda r: frac(b(r)["verified_branches"], b(r)["branches_started"])),
        row("first transform tokens (all)", med_cell("first_transform_all")),
        row("first transform tokens (b0)", med_cell("first_transform_b0")),
        row("first transform tokens (b1)", med_cell("first_transform_b1")),
        row("first check tokens (all)", med_cell("first_check_all")),
        row("first check tokens (b0)", med_cell("first_check_b0")),
        row("nudged branches", lambda r: fmt(b(r)["nudged_branches"])),
        row("nudge 24k", nud_cell("24k")),
        row("nudge E16 early", nud_cell("e16_early")),
        row("nudge early (no reason field)", nud_cell("early(no reason field)")),
        row(">= 24k, no transform, no nudge", lambda r: fmt(b(r)["nudge_eligible_not_nudged"])),
        row("e16 inspection calls hist", lambda r: json.dumps(b(r).get("e16_inspection_calls_hist")) if b(r).get("e16_inspection_calls_hist") else NA),
        row("e16 raw content kept/same/mism/markup", lambda r: "/".join(str(v) for v in b(r)["e16_raw_content"].values()) if b(r).get("e16_raw_content") else NA),
        row("raw reasoning kept/same/mism/markup", lambda r: "/".join(str(v) for v in b(r)["raw_reasoning"].values()) if b(r).get("raw_reasoning") else NA),
        row("length cut, no call", lambda r: fmt(b(r)["length_cut_no_call"])),
        row("infra errors", lambda r: fmt(b(r)["infra_errors"])),
    ]))
    P("")
    P("## 5. Engine (server log lines inside the solve window)")
    P(md_table(["", *labels], [
        row("log used", lambda r: f"`{e(r).get('log_used', NA)}`"),
        row("decode / prefill lines", lambda r: f"{fmt(e(r).get('decode_lines'))} / {fmt(e(r).get('prefill_lines'))}"),
        row("running-req histogram", lambda r: " ".join(f"{k}:{v}" for k, v in (e(r).get("running_hist") or {}).items()) or NA),
        row("accept len p50 (all)", lambda r: fmt(e(r).get("accept_len_p50"), 2)),
        row("accept len p50 (running >= full-1)", lambda r: fmt(e(r).get("accept_len_p50_full_minus1_or_more"), 2)),
        row("gen tok/s p50 at full (= streams)", lambda r: f"{fmt(e(r).get('gen_tok_s_p50_full'))} (n={fmt(e(r).get('lines_full'))}, "
                                                           f"level {fmt(e(r).get('full_concurrency_level'))})"),
        row("gen tok/s p50 at full, clean", lambda r: fmt(e(r).get("gen_tok_s_p50_full_clean"))),
        row("gen tok/s p50 at running >= full-1", lambda r: f"{fmt(e(r).get('gen_tok_s_p50_full_minus1_or_more'))} "
                                                            f"(n={fmt(e(r).get('lines_full_minus1_or_more'))})"),
        row("gen tok/s p50 (all lines)", lambda r: fmt(e(r).get("gen_tok_s_p50_all"))),
        row("decode lines right after a prefill", lambda r: fmt(e(r).get("share_lines_after_prefill"), 3)),
        row("prefill share (uncached prefill / (prefill + gen))", lambda r: fmt(e(r).get("prefill_share_tokens"), 3)),
        row("prefill uncached / cached tokens (log)", lambda r: f"{fmt(e(r).get('prefill_new_tokens'))} / {fmt(e(r).get('prefill_cached_tokens'))}"),
        row("cached % (log prefill)", lambda r: fmt(e(r).get("prefill_cached_pct_log"))),
        row("cached % (report usage_all)", lambda r: fmt(e(r).get("cached_pct_report"))),
        row("cached % branch p50", lambda r: fmt(e(r).get("cached_pct_branch_p50"))),
        row("queue max / KV usage max / mamba max", lambda r: f"{fmt(e(r).get('queue_max'))} / {fmt(e(r).get('kv_usage_max'), 2)} / "
                                                              f"{fmt(e(r).get('mamba_usage_max'), 2)}"),
        row("OOM lines (all logs / in window)", lambda r: _iss(e(r), "oom")),
        row("Traceback lines", lambda r: _iss(e(r), "traceback")),
        row("Retract lines", lambda r: _iss(e(r), "retract")),
        row("CUDA/NCCL error lines", lambda r: _iss(e(r), "cuda_error")),
        row("other *Error:/*Exception: lines", lambda r: _iss(e(r), "exception")),
        row("kernel log Tracebacks", lambda r: (fmt(g(e(r)["issues"], "kernel_log_traceback", "count", default=0))
                                                 if e(r).get("kernel_logs_scanned") else NA)),
    ]))
    P("")
    for r in runs:
        smp = list(dict.fromkeys(x for d in r["engine"]["issues"].values() for x in d["samples"]))   # a line can be in 2 kinds
        if smp:
            P(f"<details><summary>{r['label']}: log issue samples ({len(smp)})</summary>")
            P("")
            for x in smp:
                P(f"    {x}")
            P("")
            P("</details>")
            P("")
    P("## 6. Per-task table")
    P("Cell = query hits/queries · b0 class/k tokens · b1 class/k tokens. '-' = not started, blank = no branch.")
    P("")
    rows = []
    for t in tasks:
        cells = [t]
        for r in runs:
            h = r["per_task_hits"].get(t)
            hs = f"{sum(h)}/{len(h)}" if h else "?"
            parts = []
            for bn in (0, 1):
                x = next((x for x in r["branches"] if x["task"] == t and x["branch"] == bn), None)
                parts.append("" if x is None else f"{x['class']}/{x['generated'] / 1000:.0f}k")
            mark = "**" if h and any(h) else ""
            cells.append(f"{mark}{hs}{mark} · {parts[0] or ' '} · {parts[1] or ' '}")
        who = [r["label"] for r in runs if any(r["per_task_hits"].get(t) or [])]
        cells.append(", ".join(who) if who else "")
        rows.append(cells)
    P(md_table(["task", *labels, "solved by"], rows))
    P("")
    P("## 7. Union and complementarity")
    P(f"- union (query OR across runs): {C['union_query_hits']}/{C['union_query_total']} queries, task-norm "
      f"{fmt(C['union_task_norm_pct'], 2)} %")
    P(f"- tasks with a hit in any run ({len(C['union_tasks_any_hit'])}): {', '.join(C['union_tasks_any_hit']) or 'none'}")
    P(f"- tasks with a hit in every run ({len(C['intersection_tasks_any_hit'])}): "
      f"{', '.join(C['intersection_tasks_any_hit']) or 'none'}")
    for lab, only in C["only_in"].items():
        P(f"- only {lab} ({len(only)}): {', '.join(only) or 'none'}")
    if C["pairwise"]:
        P("")
        P(md_table(["pair", "both", "only A", "only B", "union", "Jaccard"],
                   [[f"{p['a']} vs {p['b']}", p["both"], f"{len(p['only_a'])}: {', '.join(p['only_a']) or '-'}",
                     f"{len(p['only_b'])}: {', '.join(p['only_b']) or '-'}", p["union"], fmt(p["jaccard"], 3)]
                    for p in C["pairwise"]]))
    P("")
    P("## 8. Projection to the hidden test (ESTIMATE)")
    P(md_table(["", *labels], [
        row("gate 4b window", lambda r: fmt(pj(r)["g4b"])),
        row("eval s/task (planned window / tasks)", lambda r: fmt(pj(r)["eval_sec_per_task"])),
        row("hidden s/task (window / 240)", lambda r: fmt(pj(r)["hidden_sec_per_task"])),
        row("task-norm over reached tasks %", lambda r: fmt(pj(r).get("task_norm_reached_pct"), 2)),
        row("coverage (b0 started)", lambda r: fmt(pj(r)["coverage_started"], 3)),
        row("coverage (b0 got first-exposure budget)", lambda r: fmt(pj(r)["coverage_meaningful"], 3)),
        row("hidden coverage estimate", lambda r: fmt(pj(r)["coverage_hidden_est"], 3)),
        row("LB proxy % (estimate)", lambda r: fmt(pj(r)["lb_proxy_pct"], 2)),
    ]))
    P("")
    for r in runs:
        P(f"- {r['label']}: {r['projection']['method']}")
    P("- LB is scored on 120 of the 240 hidden tasks (noise sd ~1.5 points); past LB/eval ratios were coverage-driven. "
      "The proxy is an estimate, not a prediction.")
    P("")
    P("## 9. Data checks (anything that looks wrong)")
    any_w = False
    for r in runs:
        if r["warnings"]:
            any_w = True
            P(f"**{r['label']}**")
            for w in r["warnings"]:
                P(f"- {w}")
            P("")
    if not any_w:
        P("- none")
    return "\n".join(L)


def _iss(eng, k):
    d = (eng.get("issues") or {}).get(k)
    if not d:
        return "0"
    return f"{d['count']} / {d['in_window'] if d['in_window'] is not None else NA}"


# ------------------------------------------------------------------------------------------------------ main
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run folders")
    ap.add_argument("--labels", default="", help="comma-separated labels (default: folder names)")
    ap.add_argument("--out", default="", help="JSON output path")
    ap.add_argument("--solutions", default="", help="arc-agi_evaluation_solutions.json (default: auto-locate)")
    ap.add_argument("--hidden-tasks", type=int, default=240)
    ap.add_argument("--hidden-window-sec", type=float, default=None,
                    help="hidden solve window (default: g4b hidden_solve_end - solve_start, else 42720 - start)")
    ap.add_argument("--no-transcripts", action="store_true", help="skip transcript parsing (faster)")
    a = ap.parse_args(argv)
    dirs = [Path(p) for p in a.runs]
    labels = [s.strip() for s in a.labels.split(",")] if a.labels else []
    if labels and len(labels) != len(dirs):
        ap.error(f"{len(labels)} labels for {len(dirs)} runs")
    labels = labels or [d.name for d in dirs]
    sol = None
    cands = [Path(a.solutions)] if a.solutions else DEFAULT_SOLUTIONS
    for p in cands:
        if p.is_file():
            try:
                sol = json.loads(p.read_text(encoding="utf-8"))
                break
            except (OSError, ValueError):
                pass
    runs = []
    for d, lab in zip(dirs, labels):
        try:
            runs.append(analyse_run(d, lab, sol, use_transcripts=not a.no_transcripts, hidden_tasks=a.hidden_tasks,
                                    hidden_window=a.hidden_window_sec))
        except Exception as exc:          # a broken run must not hide the others
            import traceback
            print(f"[{lab}] analysis failed: {exc}\n{traceback.format_exc()}", file=sys.stderr)
    if not runs:
        print("no run could be analysed", file=sys.stderr)
        return 2
    C, tasks = compare(runs)
    md = render(runs, C, tasks)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    print(md)
    if a.out:
        out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
               "solutions": str(next((p for p in cands if p.is_file()), "")) if sol else None,
               "comparison": C, "runs": []}
        for r in runs:
            rr = dict(r)
            rr["branches"] = [{k: v for k, v in x.items()} for x in r["branches"]]
            out["runs"].append(rr)
        Path(a.out).write_text(json.dumps(_jsonable(out), indent=1, default=str), encoding="utf-8")
        print(f"\n(JSON written to {a.out})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
