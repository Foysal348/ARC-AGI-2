"""HYBRID rescue: the LB-18.75 DIRECT method for every test input whose task the agent did not verify.

v1.8 (run-record check, 2026-10-12): v1.6/v1.7 asked one 32k-token answer per TASK with another prompt. The EVAL12
records of our DIRECT pipeline (run 4 = v1.4, LB 18.75; run 7 = v1.4.4, LB 26.94; same prompt/parser) show every correct answer needed 26-37k thinking tokens (one more came from
a forced close at 49k), so 32k would have lost 3 of its 5 hits. Now exactly as Q27-XHIGH v1.4 (LB 18.75): one request
per TEST INPUT, its prompt and grid parser copied verbatim, xhigh, temperature 0.6 / top_p 0.95 / top_k 20, thinking
capped at 49,152 tokens and up to 8,192 answer tokens. Difference: the server forces </think> at the budget (v1.4
appended Qwen's stop sentence); 1 of 37 forced closes was correct there, the natural closes carry the method.
A reply without a parseable grid gets ONE no-thinking commit call with the tail of its notes.

Candidate order per test input (merge): the top verified program grid -> the direct grid -> further verified grids ->
unverified grids -> the test input; the first two distinct grids are the attempts. v1.8: only ONE verified grid ranks
above the direct answer (R6 code loop: 3 of 4 train-verified programs were wrong on test).
"""
import copy
import json
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from client import ChatClient


# ---------------------------------------------------------------- verbatim from Q27-XHIGH v1.4 (LB 18.75), cells 4-5
def render_grid(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


def build_user_message(train, test_input):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(train):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i + 1, render_grid(p["input"]), i + 1, render_grid(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render_grid(test_input))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)


def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
_SEP_ROW = re.compile(r"\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?")
_JSON_ROW = re.compile(r"\[\s*\d(?:\s*,\s*\d)*\s*\]")


def _parse_row(s):
    s = s.strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):                                  # contiguous digits, one per cell ("102")
        return [int(c) for c in s]
    if _SEP_ROW.fullmatch(s):
        return [int(c) for c in re.findall(r"\d", s)]
    return None


def _first_row_run(lines):
    rows = []
    for line in lines:
        r = _parse_row(line)
        if r is None:
            if rows:
                break
            continue
        rows.append(r)
    return rows if valid_grid(rows) else None


def _grid_from_obj(obj):
    if isinstance(obj, dict):
        for k in ("output", "grid", "test_output", "answer", "test_outputs"):
            if k in obj:
                return _grid_from_obj(obj[k])
        return None
    if valid_grid(obj):
        return obj
    if isinstance(obj, list) and obj and valid_grid(obj[-1]):     # a list of grids: the last one
        return obj[-1]
    return None


def _json_rows_grid(b):
    grids, cur = [], []
    for m in _JSON_ROW.finditer(b):
        cur.append(json.loads(m.group(0)))
        if b[m.end():].lstrip().startswith("]"):
            grids.append(cur); cur = []
    if cur:
        grids.append(cur)
    for g in reversed(grids):
        if valid_grid(g):
            return g
    return None


def _grid_from_block(body):
    b = body.strip()
    if b[:1] in ("[", "{"):
        try:
            g = _grid_from_obj(json.loads(b))
        except Exception:
            g = None
        if g is None:
            g = _json_rows_grid(b)
        if g is not None:
            return g
    return _first_row_run(b.splitlines())


def parse_grid(text):
    """answer text after </think> -> grid or None: the last fenced block (any label) that holds a valid grid; else the
    last run of digit rows in the text (unfenced answer or a fence that was never closed)"""
    for _lab, body in reversed(_FENCE_RE.findall(text or "")):
        g = _grid_from_block(body)
        if g is not None:
            return g
    runs, cur = [], []
    for line in (text or "").splitlines():
        r = _parse_row(line)
        if r is None:
            if cur:
                runs.append(cur); cur = []
        else:
            cur.append(r)
    if cur:
        runs.append(cur)
    for run in reversed(runs):
        if valid_grid(run):
            return run
    return None
# ---------------------------------------------------------------- end of the verbatim block


COMMIT = """{prompt}

## Your notes so far (they were cut off)
{notes}

Stop analysing now. Apply the rule you were converging on and give ONLY the final output grid as rows of space-separated digits inside a fenced block:
```grid
<rows>
```"""


def direct_client(cfg):
    """v1.8: base_url / model default to the local server (v1.7 crashed: the smoke passed a cfg without base_url)"""
    think = int(cfg.get("rescue_think", 49152))
    use_think = bool(cfg.get("rescue_thinking", True))   # v1.8.2: NOT the agent's cfg["thinking"] fallback
    return ChatClient(cfg.get("base_url", "http://127.0.0.1:1234/v1"), cfg.get("model", "qwen"),
                      max_tokens=think + int(cfg.get("rescue_answer", 8192)),
                      temperature=cfg.get("rescue_temperature", 0.6), top_p=0.95, top_k=20, thinking=use_think,
                      thinking_token_budget=(think if use_think else None),   # v1.8.2: not the agent's budget
                      template_kwargs={"reasoning_effort": cfg.get("rescue_effort", "xhigh")})


def time_budget(think_max, finish, rate, answer_reserve_tokens=4096, overhead_s=120.0, min_think=1024):
    """v1.8.3 (GPT review): the thinking budget that still ends before `finish`, or None when not even `min_think`
    fits (then no new long request starts). remaining = finish - now - overhead (prefill + safety); token capacity =
    remaining x per-stream tok/s (a conservative estimate); the answer keeps answer_reserve_tokens of it.
    Unlike v1.4.4's global drain (forced close at 11:36 from actual progress) this is decided at admission, so the
    speed estimate must be conservative -- see run_rescue."""
    remaining = finish - time.time() - overhead_s
    if remaining <= 0 or rate <= 0:
        return None
    think = int(remaining * rate) - int(answer_reserve_tokens)
    if think < min_think:
        return None
    return int(min(think_max, think))


def direct_answer(client, task, j, seed, deadline, commit_tokens=8192, notes_chars=16000, think_budget=None):
    """one LB-26.94-method request for test input j; returns a record with "grid" (or None).
    deadline = the time by which the answer must be back; think_budget (v1.8.3) caps the thinking for this request."""
    prompt = build_user_message(task["train"], task["test"][j]["input"])

    def timeout():
        return max(60.0, min(4 * 3600, deadline - time.time() + 60))

    _tb = getattr(client, "thinking_token_budget", None)
    if think_budget is not None and _tb and think_budget < _tb:
        answer_room = client.max_tokens - client.thinking_token_budget
        client = copy.copy(client)                      # per-request budget; the shared client stays unchanged
        client.thinking_token_budget = int(think_budget)
        client.max_tokens = int(think_budget) + answer_room
    mt = client.max_tokens
    for _ in range(3):
        r = client.chat([{"role": "user", "content": prompt}], seed=seed, max_tokens=mt, timeout=timeout())
        if not (r.get("context_overflow") and mt > 16384):
            break
        mt = int(mt * 0.8)                     # a very long prompt: shrink the reply, never give up
    if "error" in r:
        return {"status": "error: " + str(r["error"])[:200], "grid": None, "gen_tokens": 0}
    rec = {"finish": r.get("finish_reason"), "gen_tokens": int(r.get("completion_tokens") or 0), "max_tokens": mt,
           "think_budget": getattr(client, "thinking_token_budget", None),
           "reasoning_chars": len(r.get("reasoning") or ""), "answer_tail": (r.get("content") or "")[-1200:],
           "reasoning_tail": (r.get("reasoning") or "")[-2000:]}
    grid = parse_grid(r.get("content") or "")
    if grid is not None:
        return dict(rec, status="answered", grid=grid)
    notes = ((r.get("reasoning") or "") + "\n" + (r.get("content") or "")).strip()
    if notes and deadline - time.time() > 120:
        r2 = client.chat([{"role": "user", "content": COMMIT.format(prompt=prompt, notes=notes[-notes_chars:])}],
                         seed=seed + 1, max_tokens=commit_tokens, timeout=timeout(),
                         template_kwargs={"enable_thinking": False, "reasoning_effort": "low"})
        if "error" in r2:
            rec["commit_error"] = str(r2["error"])[:200]
        else:
            rec["gen_tokens"] += int(r2.get("completion_tokens") or 0)
            rec["commit_finish"] = r2.get("finish_reason")
            grid = parse_grid(r2.get("content") or "") or parse_grid(r2.get("reasoning") or "")
            if grid is not None:
                return dict(rec, status="committed", grid=grid)
    return dict(rec, status="no grid", grid=None)


def run_rescue(task_ids, tasks, cfg, deadline, log=print, on_result=None):
    """direct answers for every test input of task_ids (in that order), concurrently.
    Returns ({tid: {"grids": [grid or None per test input], "units": {j: rec}, "gen_tokens", "status"}}, client).

    v1.8.3 timing (GPT review): new requests may start until `deadline` (no 5-min cut-off any more) while a request's
    thinking budget still fits before finish = deadline + rescue_finish_extra_s (default 480 s: 11:43 in the hidden run,
    v1.4.4's hard stop was 11:44); the per-stream speed is the 25th percentile of the last 32 finished requests
    (tokens / wall, prefill included), 15 tok/s until 4 have finished (run 7 measured ~17); no new request starts while
    the server's KV usage is >= rescue_kv_throttle (from the metrics sampler). Each record keeps its plan and whether
    it came back after `finish` (late) so the forecast error is visible."""
    client = direct_client(cfg)
    out, lock = {}, threading.Lock()
    finish = deadline + float(cfg.get("rescue_finish_extra_s", 480))
    rates = []
    rate0 = float(cfg.get("rescue_stream_tok_s", 15.0))
    throttle = float(cfg.get("rescue_kv_throttle", 0.95))
    for tid in task_ids:
        out[tid] = {"grids": [None] * len(tasks[tid]["test"]), "units": {}, "gen_tokens": 0, "status": "pending"}
    units = [(tid, j) for tid in task_ids for j in range(len(tasks[tid]["test"]))]

    def current_rate():
        with lock:
            r = sorted(rates[-32:])
        return r[len(r) // 4] if len(r) >= 4 else rate0

    def work(unit):
        tid, j = unit
        budget = None
        while time.time() < deadline:
            budget = time_budget(int(cfg.get("rescue_think", 49152)), finish, current_rate(),
                                 int(cfg.get("rescue_answer_reserve_tokens", 4096)),
                                 float(cfg.get("rescue_overhead_s", 120)), int(cfg.get("rescue_min_think", 4096)))
            kv = SAMPLER.latest_kv() if SAMPLER else None
            if budget is None or kv is None or kv < throttle:
                break
            time.sleep(float(cfg.get("rescue_throttle_sleep_s", 15)))   # KV pressure: let running requests finish
        if budget is None or time.time() >= deadline:
            rec = {"status": "skipped: deadline", "grid": None, "gen_tokens": 0}
        else:
            t = time.time()
            try:
                rec = direct_answer(client, tasks[tid], j, 7 + 1000 * j, finish,
                                    commit_tokens=int(cfg.get("rescue_commit_tokens", 8192)), think_budget=budget)
            except Exception as e:                       # one unit must never sink the stage
                rec = {"status": f"crashed: {type(e).__name__}: {e}"[:200], "grid": None, "gen_tokens": 0}
            rec["wall"] = round(time.time() - t, 1)
            rec.update(planned_think_budget=budget, started=round(t), finish_target=round(finish),
                       late=time.time() > finish)
            if rec.get("gen_tokens") and rec["wall"] > 0:
                with lock:
                    rates.append(rec["gen_tokens"] / rec["wall"])
        with lock:
            tr = out[tid]
            tr["units"][j] = rec
            tr["grids"][j] = rec.get("grid")
            tr["gen_tokens"] += int(rec.get("gen_tokens") or 0)
            sts = [str(u.get("status")).split(":")[0] for u in tr["units"].values()]
            tr["status"] = sts[0] if len(set(sts)) == 1 else "mixed"
            if on_result:
                try:
                    on_result(tid, tr)
                except Exception as e:
                    log(f"[rescue] on_result failed for {tid}: {e}")
        log(f"[rescue] {tid}#{j} {str(rec.get('status'))[:30]} tok {rec.get('gen_tokens', 0)} "
            f"budget {rec.get('planned_think_budget')} finish {rec.get('finish')} late {rec.get('late')}")

    with ThreadPoolExecutor(max_workers=int(cfg.get("rescue_concurrency", cfg.get("concurrency", 16)))) as ex:
        list(ex.map(work, units))
    return out, client


def server_metrics(base_url="http://127.0.0.1:1234/v1"):
    """v1.8.3: vLLM's own counters (KV size / usage, preemptions, MTP acceptance) -- v1.4.4 reported these"""
    try:
        text = urllib.request.urlopen(base_url.rsplit("/v1", 1)[0] + "/metrics", timeout=10).read().decode()
    except Exception as e:
        return {"error": repr(e)[:200]}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, val = line.rsplit(" ", 1)
        base = name.split("{")[0]
        if base in ("vllm:num_preemptions_total", "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
                    "vllm:num_requests_running", "vllm:num_requests_waiting",
                    "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:generation_tokens_total", "vllm:prompt_tokens_total"):
            try:
                out[base.split(":", 1)[1]] = out.get(base.split(":", 1)[1], 0.0) + float(val)
            except ValueError:
                pass
        if base == "vllm:cache_config_info":
            for k in ("num_gpu_blocks", "block_size"):
                m = re.search(k + r'="(\d+)"', name)
                if m:
                    out[k] = int(m.group(1))
    if out.get("num_gpu_blocks") and out.get("block_size"):
        out["kv_tokens"] = out["num_gpu_blocks"] * out["block_size"]
    if out.get("spec_decode_num_draft_tokens_total"):
        out["mtp_acceptance"] = round(out.get("spec_decode_num_accepted_tokens_total", 0.0)
                                      / out["spec_decode_num_draft_tokens_total"], 3)
    return out


class MetricsSampler:
    """v1.8.3 (GPT review): /metrics every `every` seconds in a background thread -- KV usage peak / p95, running and
    waiting requests, and counter DELTAS (preemptions, MTP accepted / draft, generated tokens) that survive a server
    restart (a counter that goes down starts a new segment). Never fatal: failures are counted and reported."""

    COUNTERS = ("num_preemptions_total", "spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total",
                "generation_tokens_total")

    def __init__(self, base_url="http://127.0.0.1:1234/v1", every=30.0):
        self.base_url, self.every = base_url, every
        self.kv, self.running, self.waiting, self.fails, self.samples = [], [], [], 0, 0
        self.delta = {k: 0.0 for k in self.COUNTERS}
        self.last = {}
        self._stop = threading.Event()
        self._t = None

    def sample(self):
        m = server_metrics(self.base_url)
        if "error" in m:
            self.fails += 1
            return
        self.samples += 1
        kv = m.get("kv_cache_usage_perc", m.get("gpu_cache_usage_perc"))
        if kv is not None:
            self.kv.append(float(kv))
        for k, dst in (("num_requests_running", self.running), ("num_requests_waiting", self.waiting)):
            if k in m:
                dst.append(float(m[k]))
        for k in self.COUNTERS:
            if k in m:
                v = float(m[k])
                prev = self.last.get(k)
                if prev is not None:
                    self.delta[k] += (v - prev) if v >= prev else v     # restart -> the new counter counts from 0
                self.last[k] = v

    def latest_kv(self):
        return self.kv[-1] if self.kv else None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:
                self.fails += 1
            self._stop.wait(self.every)

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=15)
        try:
            self.sample()                                # final counters
        except Exception:
            self.fails += 1
        return self.summary()

    def summary(self):
        if not self.samples:
            return {"metrics_unavailable": True, "failures": self.fails}
        kv = sorted(self.kv)
        acc, drf = self.delta["spec_decode_num_accepted_tokens_total"], self.delta["spec_decode_num_draft_tokens_total"]
        return {"samples": self.samples, "failures": self.fails,
                "kv_usage_peak": kv[-1] if kv else None, "kv_usage_p95": kv[int(0.95 * (len(kv) - 1))] if kv else None,
                "running_peak": max(self.running) if self.running else None,
                "waiting_peak": max(self.waiting) if self.waiting else None,
                "preemptions": self.delta["num_preemptions_total"],
                "generated_tokens": self.delta["generation_tokens_total"],
                "mtp_acceptance": round(acc / drf, 3) if drf else None}


SAMPLER = None              # set by run_eval / run_submit while their stage runs (used for the KV-pressure throttle)


def direct_grid(rec, j):
    grids = (rec or {}).get("grids")
    return grids[j] if isinstance(grids, list) and j < len(grids) else None


def merge(agent_ranked, direct, inp):
    """agent_ranked: [(grid, source)] best first, source 'verified ...' or 'unverified ...'; direct: grid or None"""
    ver = [g for g, s in agent_ranked if str(s).startswith("verified")]
    unv = [g for g, s in agent_ranked if not str(s).startswith("verified")]
    order = (ver[:1] + [direct] + ver[1:] + unv) if direct is not None else (ver + unv)
    out = []
    for g in order + [inp]:
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


def load_direct_ref(path, run="run7"):
    """v1.8: the LB-18.75 pipeline's own answers on EVAL12 (runs 4 and 7, parsed with the verbatim parser) in the
    rescue-record shape, so the commit run can merge the agent with DIRECT exactly without spending GPU on DIRECT"""
    ref = json.load(open(path))
    return {t: {"grids": [q.get("grid") for q in qs], "gen_tokens": sum(int(q.get("think_tokens") or 0) for q in qs),
                "status": "reference", "units": {j: q for j, q in enumerate(qs)}}
            for t, qs in ref[run].items()}
