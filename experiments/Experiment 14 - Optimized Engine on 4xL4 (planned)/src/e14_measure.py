"""E14-A measurement library (2026-10-23; plan: Notes/13). Stdlib only; embedded in the E14 notebook as text.

Everything the notebook owns (server URL, server log, tokenizer, Duck prompt builders, deadlines) comes in through `Ctx`,
so the module runs unchanged against the real SGLang 0.5.21 server on Kaggle and against the CPU fake server.

Rules taken from project memory and the GPT reviews (Notes/08-10):
- a window counts only when the server REALLY runs the requested number of requests (from /metrics, max over TP ranks);
- decode step time = running x accept_len / gen_tok_s per server decode line (= window wall / 40 forwards, Notes/02),
  lines right after a prefill batch excluded;
- greedy parity is reported with the first divergence position and the top-1/top-2 margin there, against an A/A
  control; no p-value is used;
- every generated text is saved (thinking-trace rule).
"""
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request
from pathlib import Path


# --------------------------------------------------------------------------------------------- context + http
class Ctx:
    def __init__(self, url, log_text, out_dir, *, model_id="default", deadline=None, log=print, now=time.perf_counter,
                 encode=None, chat_ids=None, normalize=None):
        """url: server base (http://127.0.0.1:30000); log_text(): the CURRENT server log text; deadline: absolute
        perf_counter value no phase may pass; encode(text) -> ids; chat_ids(messages, tools=None, think=True) -> ids of
        the rendered prompt (generation prompt included); normalize(text) = Duck's _normalize_message_content."""
        self.url, self.log_text, self.out = url.rstrip("/"), log_text, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.model_id, self.deadline, self.log, self.now = model_id, deadline, log, now
        self.encode, self.chat_ids, self.normalize = encode, chat_ids, normalize
        # an unknown header is ignored by SGLang; the CPU fake server uses it to keep probe traffic out of its Duck flow
        self.headers = {"Content-Type": "application/json", "X-E14-Probe": "1"}
        self.scale = 1.0          # safety margins x scale (the CPU harness runs everything faster)

    def m(self, sec):
        return float(sec) * self.scale

    def left(self):
        return float("inf") if self.deadline is None else self.deadline - self.now()

    def save(self, name, obj):
        p = self.out / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, default=str), encoding="utf-8")
        os.replace(tmp, p)
        return p

    def append(self, name, obj):
        with open(self.out / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")


CLEANUP_SEC = 20.0        # one shared bound for aborts + joins at the end of a phase step (GPT audits 17-19)


def post_json(ctx, path, body, timeout=600, cleanup=False):
    # every request ends by the phase deadline (E12 PARITY: urlopen(timeout=900) x 24 probes could hold 6 h); past the
    # deadline nothing new is sent (audits 17 HIGH-3 / 18 E14-4: the old 5 s floor still sent requests); cleanup calls
    # (/abort_request) get their own short bound instead, so outstanding server work can still be stopped
    if cleanup:
        timeout = min(float(timeout), 10.0)
    else:
        left = ctx.left()
        if left <= 0:
            raise TimeoutError(f"E14 phase deadline passed before {path}")
        timeout = min(float(timeout), left)
    req = urllib.request.Request(ctx.url + path, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers=dict(getattr(ctx, "headers", None) or {"Content-Type": "application/json"}))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except ValueError:                # /start_profile answers plain text "Start profiling.\n" (http_server.py:1164-1172)
        return {"text": raw.decode("utf-8", "replace")[:500]}


_PROM_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def prom(ctx, timeout=10):
    """{metric name: max over label sets} (max over TP ranks; never sglang:token_usage = max of pools)."""
    out = {}
    txt = urllib.request.urlopen(ctx.url + "/metrics", timeout=timeout).read().decode("utf-8", "replace")
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_RE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = max(v, out.get(m.group(1), v))
    return out


class Sampler(threading.Thread):
    """Samples running / queue / KV / mamba every `every` s until stop()."""

    def __init__(self, ctx, every=0.5):
        super().__init__(daemon=True)
        # never `self._stop`: Python <= 3.12 Thread.join() calls its own self._stop() (E14A SLIM run1 crashed on it)
        self.ctx, self.every, self.rows, self._halt = ctx, every, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                p = prom(self.ctx, timeout=5)
                self.rows.append({"t": round(self.ctx.now(), 2), "running": p.get("sglang:num_running_reqs"),
                                  "queue": p.get("sglang:num_queue_reqs"), "kv": p.get("sglang:full_token_usage"),
                                  "mamba": p.get("sglang:mamba_usage"),
                                  "retracted": p.get("sglang:num_retracted_requests_total")})
            except Exception as e:
                self.rows.append({"t": round(self.ctx.now(), 2), "error": repr(e)[:120]})
            self._halt.wait(self.every)

    def stop(self):
        self._halt.set()
        self.join(10)
        return self.rows


def running_share(rows, level):
    ok = [r for r in rows if r.get("running") is not None]
    return round(sum(1 for r in ok if r["running"] >= level) / len(ok), 3) if ok else None


# --------------------------------------------------------------------------------------------- server log parsing
_TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_KV_RE = re.compile(r"([A-Za-z#][A-Za-z#_ ()/\-]*?):\s*(-?\d+(?:\.\d+)?)")


def decode_lines(text, start=0):
    """TP0 'Decode batch' lines from offset `start` of the log text, each with its fields, the timestamp and whether a
    'Prefill batch' line came right before it (such windows include prefill time)."""
    rows, after_prefill = [], False
    for line in text[start:].splitlines():
        if "TP0]" not in line:
            continue
        if "Prefill batch" in line:
            after_prefill = True
            continue
        if "Decode batch" not in line:
            continue
        f = {k.strip(): float(v) for k, v in _KV_RE.findall(line.split("Decode batch", 1)[1])}
        m = _TS_RE.match(line)
        rows.append({"running": f.get("#running-req"), "full_token": f.get("#full token"), "queue": f.get("#queue-req"),
                     "gen_tps": f.get("gen throughput (token/s)"), "accept": f.get("accept len"),
                     "mamba": f.get("mamba usage"), "kv": f.get("full token usage"), "after_prefill": after_prefill,
                     "ts": m.group(1) if m else None})
        after_prefill = False
    return rows


def _q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))] if v else None


def window_stats(rows, level, spec=True):
    """Median decode step (ms), tok/s, accept len and context per request over the lines at `level` running."""
    use = [r for r in rows if r["running"] == level and not r["after_prefill"] and r["gen_tps"]]
    step = []
    for r in use:
        acc = r["accept"] if (spec and r["accept"]) else 1.0
        step.append(1000.0 * r["running"] * acc / r["gen_tps"])
    med = lambda v: round(statistics.median(v), 2) if v else None
    return {"lines": len(use), "lines_all": len(rows), "step_ms_p50": med(step), "step_ms_p10": _q(step, .1),
            "step_ms_p90": _q(step, .9), "tok_s_p50": med([r["gen_tps"] for r in use]),
            "accept_p50": med([r["accept"] for r in use if r["accept"]]),
            "ctx_per_req_p50": med([r["full_token"] / r["running"] for r in use if r["full_token"]]),
            "mamba_p50": med([r["mamba"] for r in use if r["mamba"] is not None]),
            "running_hist": {str(int(k)): sum(1 for r in rows if r["running"] == k)
                             for k in sorted({r["running"] for r in rows if r["running"] is not None})}}


# --------------------------------------------------------------------------------------------- generation traffic
class Gen(threading.Thread):
    """One non-streaming /generate request with an rid (abortable with /abort_request)."""

    def __init__(self, ctx, rid, input_ids, max_new, temperature=1.0, top_p=0.95, top_k=20, logprobs=False,
                 timeout=1800):
        super().__init__(daemon=True)
        self.ctx, self.rid, self.ids, self.max_new = ctx, rid, list(input_ids), int(max_new)
        self.sp = {"max_new_tokens": self.max_new, "temperature": temperature, "top_p": top_p, "top_k": top_k}
        self.logprobs, self.timeout, self.result, self.error = logprobs, timeout, None, None
        self.t0 = self.t1 = None

    def run(self):
        self.t0 = self.ctx.now()
        body = {"rid": self.rid, "input_ids": self.ids, "sampling_params": self.sp}
        if self.logprobs:
            body.update(return_logprob=True, top_logprobs_num=2)
        try:
            r = post_json(self.ctx, "/generate", body, timeout=self.timeout)
            self.result = r[0] if isinstance(r, list) else r
        except Exception as e:
            self.error = repr(e)[:300]
        self.t1 = self.ctx.now()

    def output_ids(self):
        r = self.result or {}
        if r.get("output_ids"):
            return list(r["output_ids"])
        # (logprob, token_id, text) tuples (tokenizer_manager.py L2916-2920) when the ids are not in the reply
        return [x[1] for x in (r.get("meta_info") or {}).get("output_token_logprobs") or []
                if isinstance(x, (list, tuple)) and len(x) > 1]

    def meta(self):
        return (self.result or {}).get("meta_info") or {}


def join_all(threads, budget):
    """Join every thread against ONE shared deadline (not budget per thread: 14 x join(60) was 14 min). Returns the
    number still alive."""
    end = time.monotonic() + max(0.0, float(budget))
    for t in threads:
        if t.ident is not None:          # never started (skipped at the deadline): nothing to join
            t.join(max(0.0, end - time.monotonic()))
    return sum(1 for t in threads if t.is_alive())


def _abort_one(ctx, rid, budget):
    try:
        post_json(ctx, "/abort_request", {"rid": rid}, timeout=budget, cleanup=True)
    except Exception:
        pass


def abort(ctx, rids, budget=8.0):
    """/abort_request for every rid at once, bounded by one shared `budget` (seconds)."""
    ths = [threading.Thread(target=_abort_one, args=(ctx, r, budget), daemon=True) for r in rids]
    for t in ths:
        t.start()
    join_all(ths, budget)


def server_idle(ctx, wait=5.0):
    """True once /metrics shows 0 running and 0 queued (False if that never happens within `wait` s)."""
    end = time.monotonic() + max(0.5, float(wait))
    while True:
        try:
            p = prom(ctx, timeout=3)
            if not (p.get("sglang:num_running_reqs") or 0) and not (p.get("sglang:num_queue_reqs") or 0):
                return True
        except Exception:
            pass
        if time.monotonic() >= end:
            return False
        time.sleep(0.25)


def flush_cache(ctx, wait=10.0):
    """/flush_cache on an idle server (0.5.21 http_server.py:987: plain text, refused while requests run). Returns
    {"ok", "idle", "reply"}; never raises (a failed flush only means the next pass is not cold)."""
    idle = server_idle(ctx, wait=wait)
    try:
        txt = (post_json(ctx, "/flush_cache", {}, timeout=30, cleanup=True) or {}).get("text") or ""
    except Exception as e:
        txt = "error: " + repr(e)[:200]
    return {"ok": txt.strip().startswith("Cache flushed"), "idle": idle, "reply": txt.strip()[:200]}


def cleanup(ctx, gens, budget=None):
    """End of a measurement step: abort what is alive, join everything, check the server is idle - all inside ONE
    bound. server_idle False = requests still run on the server: the caller must not measure the next thing on this
    server (the notebook relaunches it; audit 19 P0-B, audit 18 E14-5 carry-over)."""
    budget = CLEANUP_SEC if budget is None else float(budget)
    t0 = time.monotonic()
    live = [g for g in gens if g.is_alive()]
    if live:
        abort(ctx, [g.rid for g in live], budget=budget * 0.4)
    alive = join_all(gens, max(0.0, budget * 0.75 - (time.monotonic() - t0)))
    idle = server_idle(ctx, wait=max(0.5, budget - (time.monotonic() - t0)))
    return {"sec": round(time.monotonic() - t0, 2), "aborted": len(live), "alive_threads": alive, "server_idle": idle}


def wait_running(ctx, level, timeout=300.0, need=3):
    """True once /metrics shows >= level running in `need` consecutive samples (1 s apart)."""
    t_end, hits = ctx.now() + min(timeout, ctx.left()), 0
    while ctx.now() < t_end:
        try:
            hits = hits + 1 if (prom(ctx).get("sglang:num_running_reqs") or 0) >= level else 0
        except Exception:
            hits = 0
        if hits >= need:
            return True
        time.sleep(1.0)
    return False


def wait_level(ctx, level, timeout=300.0, plateau_sec=45.0, need=3):
    """The running level to measure at: `level` once /metrics shows it `need` times in a row; otherwise the plateau
    level if running stays at one value >= level - 1 for plateau_sec (SGLang 0.5.21's DFlash min-free-slots delayer
    holds W1 at 11 of 12: allocatable 2 < threshold 3; Notes/14). None if neither happens before the timeout."""
    t_end, hits, last, since = ctx.now() + min(timeout, ctx.left()), 0, None, ctx.now()
    while ctx.now() < t_end:
        try:
            r = prom(ctx).get("sglang:num_running_reqs") or 0
        except Exception:
            r = None
        hits = hits + 1 if (r is not None and r >= level) else 0
        if hits >= need:
            return int(level)
        if r != last:
            last, since = r, ctx.now()
        elif r is not None and r >= level - 1 and ctx.now() - since >= plateau_sec:
            return int(r)
        time.sleep(1.0)
    return None


def steady_window(ctx, prompts, *, level=12, secs=75.0, max_new=4096, tag="w", spec=True, during=None, warm=True):
    """Start `level` requests, wait until the server really runs them all, then measure `secs` seconds of decode
    lines; abort the rest. `during(ctx)` runs inside the window (the profiler). warm: one 1-token request first, so a
    shared prefix (the 40k band's reference block) is in the radix cache before the streams arrive together (otherwise
    each of them prefills the whole prompt). Returns the window stats."""
    if ctx.left() < secs + ctx.m(120):
        return {"skipped": "deadline", "left_sec": round(ctx.left())}
    if warm and prompts:
        w = Gen(ctx, f"e14-{tag}-warm-{random.randrange(10 ** 9)}", prompts[0], 1)
        w.start()
        w.join(min(600.0, max(ctx.m(30), ctx.left() - secs - ctx.m(60))))
    else:
        w = None
    gens = [Gen(ctx, f"e14-{tag}-{i}-{random.randrange(10 ** 9)}", prompts[i % len(prompts)], max_new)
            for i in range(level)]
    for g in gens:
        g.start()
    got = wait_level(ctx, level, timeout=min(600.0, ctx.left() - secs - ctx.m(30)), plateau_sec=ctx.m(45))
    out = {"tag": tag, "level": level, "real_running_reached": got == level, "measured_level": got}
    if got is None:
        out["cleanup"] = cleanup(ctx, gens + ([w] if w else []))
        out["error"] = "the server never ran `level` (or a stable level - 1) requests: window invalid"
        return out
    level = got                       # the window is measured at the level the server really runs
    off = len(ctx.log_text())
    smp = Sampler(ctx)
    smp.start()
    t0 = ctx.now()
    if during is not None:
        try:
            out["during"] = during(ctx)
        except Exception as e:
            out["during"] = {"error": repr(e)[:300]}
    while ctx.now() - t0 < secs and any(g.is_alive() for g in gens):
        time.sleep(0.5)
    out["window_sec"] = round(ctx.now() - t0, 1)
    if during is not None:      # the timed loop starts only after during() returned: this window is a diagnostic
        out["diagnostic_only"] = "profiler window: its tok/s includes the profiler wait; never a speed reference"
    rows = smp.stop()
    out["cleanup"] = cleanup(ctx, gens + ([w] if w else []))
    lines = decode_lines(ctx.log_text(), off)
    out.update(window_stats(lines, level, spec=spec))
    out["real_running_share"] = running_share(rows, level)
    out["metrics_samples"] = len(rows)
    out["errors"] = [g.error for g in gens if g.error][:3]
    ctx.append("e14_windows.jsonl", out)
    return out


# --------------------------------------------------------------------------------------------- profiler
def profile_steps(ctx, steps=20, wait_sec=240.0, summarize=None):
    """POST /start_profile (num_steps auto-stops it), wait for the trace files, summarize them in-process."""
    d = ctx.out / f"profile_{int(time.time())}"
    d.mkdir(parents=True, exist_ok=True)
    t0 = ctx.now()
    try:
        # with_stack / record_shapes default to true in 0.5.21 (tokenizer_control_mixin.py:403-410): off keeps traces small
        post_json(ctx, "/start_profile", {"output_dir": str(d), "num_steps": steps, "activities": ["CPU", "GPU"],
                                          "with_stack": False, "record_shapes": False}, timeout=60)
    except Exception as e:
        return {"error": "start_profile: " + repr(e)[:200]}
    files, stable = [], 0
    while ctx.now() - t0 < min(wait_sec, ctx.left() - ctx.m(20)):
        cur = sorted(p for p in d.rglob("*") if p.is_file() and (".json" in p.name or ".pt.trace" in p.name))
        sizes = [p.stat().st_size for p in cur]
        stable = stable + 1 if cur and [p.stat().st_size for p in files] == sizes and len(cur) == len(files) else 0
        files = cur
        if stable >= 3:
            break
        time.sleep(3)
    out = {"dir": str(d), "files": [p.name for p in files], "bytes": sum(p.stat().st_size for p in files),
           "sec": round(ctx.now() - t0, 1)}
    if summarize is not None and files:
        try:
            out["summary"] = summarize(str(d), steps)
        except Exception as e:
            out["summary"] = {"error": repr(e)[:300]}
    return out


# --------------------------------------------------------------------------------------------- E14-P token replay
def chat(ctx, body, timeout=1200):
    return post_json(ctx, "/v1/chat/completions", body, timeout=timeout)


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def replay_conversation(ctx, system, user, tools, *, order=("raw", "normalized"), max_tokens=2048, tag="c"):
    """Request 1 (thinking on, Duck-shaped), then the follow-up twice: once with the server's raw reasoning in history,
    once with Duck's normalized reasoning. Exact token ids (return_token_ids) give the first divergent id."""
    base = {"model": ctx.model_id, "tools": tools, "tool_choice": "auto", "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": True}, "return_token_ids": True}
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    r1 = chat(ctx, dict(base, messages=msgs, max_tokens=max_tokens))
    c1 = r1["choices"][0]
    m1 = c1["message"]
    raw = m1.get("reasoning_content") or ""
    p1, o1 = list(c1.get("prompt_token_ids") or []), list(c1.get("response_token_ids") or [])
    calls = m1.get("tool_calls") or []
    out = {"tag": tag, "prompt1": len(p1), "output1": len(o1), "finish1": c1.get("finish_reason"),
           "reasoning_chars": len(raw), "blank_lines_in_reasoning": raw.count("\n\n"), "tool_calls": len(calls),
           "usage1": r1.get("usage")}
    ctx.append("e14_replay_texts.jsonl", {"tag": tag, "reasoning": raw, "content": m1.get("content"), "tool_calls": calls})
    for kind in order:
        reason = raw if kind == "raw" else (ctx.normalize(raw) if ctx.normalize else raw)
        am = {"role": "assistant", "content": m1.get("content") or "", "reasoning_content": reason}
        nxt = []
        if calls:
            am["tool_calls"] = calls
            nxt = [{"role": "tool", "tool_call_id": calls[0].get("id"), "content": "stdout: (2, 3)\nok"}]
        else:
            nxt = [{"role": "user", "content": "Continue."}]
        r2 = chat(ctx, dict(base, messages=msgs + [am] + nxt, max_tokens=1, temperature=0.0))
        c2 = r2["choices"][0]
        p2 = list(c2.get("prompt_token_ids") or [])
        u2 = r2.get("usage") or {}
        det = u2.get("prompt_tokens_details") if isinstance(u2.get("prompt_tokens_details"), dict) else {}
        exp = p1 + o1
        k = first_divergence(exp, p2)
        out[kind] = {"prompt2": len(p2), "cached": int(det.get("cached_tokens") or 0),
                     "first_divergence": k, "divergence_in_reply_at": (k - len(p1)) if k >= len(p1) else None,
                     "divergence_before_reply": k < len(p1), "expected_len": len(exp),
                     "full_reply_text_match": k >= len(exp) - 1}
    return out


def replay_check(ctx, convs):
    """convs: [(system, user, tools)] -> list of per-conversation results; alternates the order raw/normalized."""
    res = []
    for i, (system, user, tools) in enumerate(convs):
        if ctx.left() < ctx.m(240):
            res.append({"skipped": "deadline"})
            break
        try:
            res.append(replay_conversation(ctx, system, user, tools, tag=f"c{i}",
                                           order=("raw", "normalized") if i % 2 == 0 else ("normalized", "raw")))
        except Exception as e:
            res.append({"error": repr(e)[:300]})
    ok = [r for r in res if "raw" in r]
    summ = {"conversations": len(ok)}
    for kind in ("raw", "normalized"):
        summ[kind] = {"cached_p50": statistics.median([r[kind]["cached"] for r in ok]) if ok else None,
                      "full_reply_reused": sum(1 for r in ok if r[kind]["cached"] >= r["prompt1"] + r["output1"] - 64),
                      "cached_at_prefill_end": sum(1 for r in ok if abs(r[kind]["cached"] - (r["prompt1"] // 64) * 64) <= 64),
                      "text_match": sum(1 for r in ok if r[kind]["full_reply_text_match"])}
    return {"rows": res, "summary": summ}


# --------------------------------------------------------------------------------------------- parity probes
def parity_pass(ctx, prompts, *, max_new=512, concurrent=False, tag="p"):
    """Greedy outputs (ids + top-2 logprobs) for every prompt, one at a time or all at once."""
    gens = [Gen(ctx, f"e14-{tag}-{i}-{random.randrange(10 ** 9)}", p, max_new, temperature=0.0, top_p=1.0, top_k=1,
                logprobs=True) for i, p in enumerate(prompts)]
    if concurrent:
        for g in gens:
            g.start()
        join_all(gens, max(0.0, ctx.left()))
    else:
        for g in gens:
            if ctx.left() < ctx.m(60):
                g.error = "skipped: phase deadline"      # kept in place: rows stay index-aligned with the prompts
                continue
            g.start()
            join_all([g], max(0.0, ctx.left()))
    cl = cleanup(ctx, gens)                   # the phase deadline passed: stop them so the next phase starts clean
    rows = []
    for g in gens:
        meta = g.meta()
        top = meta.get("output_top_logprobs") or []
        margins = []
        for t in top:
            vals = sorted((x[0] for x in (t or []) if isinstance(x, (list, tuple)) and x and x[0] is not None), reverse=True)
            margins.append(round(vals[0] - vals[1], 4) if len(vals) >= 2 else None)
        rows.append({"ids": g.output_ids(), "margins": margins, "error": g.error,
                     "accept": meta.get("spec_accept_length"), "sec": round((g.t1 or 0) - (g.t0 or 0), 1)})
    ctx.append("e14_parity_outputs.jsonl", {"tag": tag, "rows": rows, "cleanup": cl})
    if not cl["server_idle"] and rows:
        rows[0]["server_not_idle"] = True      # surfaces to the phase record (dirty server -> relaunch)
    return rows


def _margin(row, k):
    m = row.get("margins") or []
    return m[k] if k < len(m) else None


def compare_passes(a, b, name, n=None):
    """Per prompt, index-aligned (a missing row is an error, never skipped): identical?, first divergence, both runs'
    top-1/top-2 margins there. n: compare only the first n tokens (a 256-token gate against a 512-token reference).
    A large-margin flip = a divergence where either run was confident (margin > 1 nat). A divergence without both
    margins is counted as missing evidence, never as "no flip" (GPT audits 17-19)."""
    expected = max(len(a), len(b))
    per = []
    for i in range(expected):
        ra = a[i] if i < len(a) else {"error": "missing row"}
        rb = b[i] if i < len(b) else {"error": "missing row"}
        ia, ib = list(ra.get("ids") or []), list(rb.get("ids") or [])
        if n:
            ia, ib = ia[:n], ib[:n]
        if ra.get("error") or rb.get("error") or not ia or not ib:
            per.append({"error": ra.get("error") or rb.get("error") or "empty"})
            continue
        if ia == ib:
            per.append({"identical": True, "first_divergence": None})
            continue
        k = first_divergence(ia, ib)
        ma, mb = _margin(ra, k), _margin(rb, k)
        per.append({"identical": False, "first_divergence": k, "margin_a": ma, "margin_b": mb,
                    "missing_margin": ma is None or mb is None,
                    "large_margin_flip": ma is not None and mb is not None and max(ma, mb) > 1.0})
    ok = [p for p in per if "identical" in p]
    div = [p for p in ok if not p["identical"]]
    return {"name": name, "expected": expected, "compared": len(ok), "errors": expected - len(ok),
            "identical": sum(1 for p in ok if p["identical"]),
            "divergence_p50": statistics.median([p["first_divergence"] for p in div]) if div else None,
            "large_margin_flips": sum(1 for p in div if p["large_margin_flip"]),
            "missing_margins": sum(1 for p in div if p["missing_margin"]), "rows": per}


# --------------------------------------------------------------------------------------------- latch test (#44401)
def latch_test(ctx, short_prompts, long_prompts, *, decodes=11, arrivals=3, secs=60.0, max_new=6000, server_cap=13,
               level=12):
    """`decodes` long decodes; once they run, `arrivals` long prompts arrive (chunked prefill). Samples running/queue.
    Primary metrics are about the CLIENT level (12 streams), not the server cap (audits 17 §5 / 19 P0-C):
    share_running_ge_level and share_stuck_below_queued (= level - 1 running with a queue: the 11 + 1 pathology).
    share_at_target (>= min(offered, cap)) stays as a diagnostic."""
    if ctx.left() < secs + ctx.m(240):
        return {"skipped": "deadline"}
    gens = [Gen(ctx, f"e14-latch-d{i}-{random.randrange(10 ** 9)}", short_prompts[i % len(short_prompts)], max_new)
            for i in range(decodes)]
    for g in gens:
        g.start()
    real = wait_running(ctx, decodes, timeout=ctx.m(300))
    smp = Sampler(ctx, every=0.5)
    smp.start()
    t_arr = ctx.now()
    arr = [Gen(ctx, f"e14-latch-a{i}-{random.randrange(10 ** 9)}", long_prompts[i % len(long_prompts)], 512)
           for i in range(arrivals)]
    for g in arr:
        g.start()
        time.sleep(2.0)
    while ctx.now() - t_arr < secs and ctx.left() > 0:
        time.sleep(0.5)
    rows = smp.stop()
    cl = cleanup(ctx, gens + arr)
    offered = decodes + arrivals
    ok = [r for r in rows if r.get("running") is not None]
    want = min(offered, server_cap)
    n = len(ok)
    share = (lambda f: round(sum(1 for r in ok if f(r)) / n, 3) if n else None)
    _q0 = next((i for i, r in enumerate(ok) if r["t"] > t_arr and (r.get("queue") or 0) > 0), None)
    drain = None if _q0 is None else next((round(r["t"] - t_arr, 1) for r in ok[_q0:] if not (r.get("queue") or 0)), None)
    retr = [r["retracted"] for r in ok if r.get("retracted") is not None]
    return {"decodes_reached": real, "offered": offered, "server_cap": server_cap, "level": level, "samples": n,
            "running_max": max((r["running"] for r in ok), default=None),
            "share_running_ge_level": share(lambda r: r["running"] >= level),
            "share_stuck_below_queued": share(lambda r: r["running"] == level - 1 and (r.get("queue") or 0) >= 1),
            "queue_drain_sec": drain, "retracted_delta": (max(retr) - min(retr)) if retr else None,
            "share_at_target": share(lambda r: r["running"] >= want),
            "share_stuck_one_below": share(lambda r: r["running"] == want - 1 and (r.get("queue") or 0) > 0),
            "running_hist": {str(int(k)): sum(1 for r in ok if r["running"] == k) for k in sorted({r["running"] for r in ok})},
            "cleanup": cl}


# --------------------------------------------------------------------------------------------- agent-like replay (E14-M)
def agent_replay(ctx, sessions, *, turns=6, admit=12, max_tokens=768, tool_text="", tag="ar"):
    """`sessions`: [(system, user)]. Each session is a real multi-turn conversation (raw reasoning kept, a ~tool result
    appended each turn); at most `admit` sessions run at once. Classifies every follow-up request like Notes/11:
    reply reused / lost at prefill end / deeper miss, from the server's own usage numbers."""
    sem = threading.Semaphore(admit)
    rows, lock = [], threading.Lock()

    def run_session(si, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prev = None
        with sem:
            for t in range(turns):
                if ctx.left() < ctx.m(90):
                    return
                body = {"model": ctx.model_id, "messages": msgs, "max_tokens": max_tokens, "temperature": 1.0,
                        "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"preserve_thinking": True,
                                                                             "enable_thinking": True}}
                try:
                    r = chat(ctx, body)
                except Exception as e:
                    with lock:
                        rows.append({"session": si, "turn": t, "error": repr(e)[:200]})
                    return
                u = r.get("usage") or {}
                det = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
                cur = {"session": si, "turn": t, "prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                       "cached": int(det.get("cached_tokens") or 0)}
                if prev is not None and cur["prompt"]:
                    a = prev["prompt"] + prev["completion"]
                    full, pre = (a // 64) * 64, (prev["prompt"] // 64) * 64
                    c = cur["cached"]
                    cur["cls"] = ("reply_reused" if c >= full - 64 else "lost_at_prefill_end" if abs(c - pre) <= 64
                                  else "deeper_miss" if c < pre - 64 else "partial_reply")
                with lock:
                    rows.append(cur)
                m = r["choices"][0]["message"]
                msgs = msgs + [{"role": "assistant", "content": m.get("content") or "",
                                "reasoning_content": m.get("reasoning_content") or ""},
                               {"role": "user", "content": tool_text + f"\n(turn {t + 1}) Continue the analysis."}]
                prev = cur

    ths = [threading.Thread(target=run_session, args=(i, s, u), daemon=True) for i, (s, u) in enumerate(sessions)]
    t0 = ctx.now()
    for th in ths:
        th.start()
    # every chat request already ends by the deadline (post_json); one shared join bound + an idle check after it
    alive = join_all(ths, max(0.0, ctx.left()) + CLEANUP_SEC)
    idle = server_idle(ctx, wait=5.0)
    follow = [r for r in rows if r.get("cls")]
    cnt = {k: sum(1 for r in follow if r["cls"] == k) for k in ("reply_reused", "lost_at_prefill_end", "deeper_miss",
                                                                  "partial_reply")}
    ctx.append("e14_agent_replay.jsonl", {"tag": tag, "rows": rows})
    return {"tag": tag, "sessions": len(sessions), "turns": turns, "admit": admit, "requests": len(rows),
            "followups": len(follow), "classes": cnt, "errors": sum(1 for r in rows if r.get("error")),
            "sec": round(ctx.now() - t0, 1),
            "prompt_tokens": sum(int(r.get("prompt") or 0) for r in rows),
            "uncached_tokens": sum(int(r.get("prompt") or 0) - int(r.get("cached") or 0) for r in rows),
            "cleanup": {"alive_threads": alive, "server_idle": idle}}


# --------------------------------------------------------------------------------------------- decisions
def adopt_speed(base, arm, min_gain=0.0, tol=0.02, occupancy=False, min_lines=8, ctx_tol=0.25, per_step=False):
    """Arm speed rule: tok/s >= base x (1 + min_gain) (minus tol for 'no loss'), only between VALID comparable windows
    (audits 17 §6.1 / 19 HIGH-D): both windows real (enough decode lines, no error, the server idle afterwards), the
    same measured running level unless occupancy=True (an arm whose point is 11 -> 12 running, labelled as such), and
    a per-request context within ctx_tol. Otherwise INCONCLUSIVE = no adoption."""
    def bad(w, nm):
        if not w or not w.get("tok_s_p50"):
            return f"{nm}: missing window"
        if w.get("error") or w.get("skipped"):
            return f"{nm}: {w.get('error') or w.get('skipped')}"
        if (w.get("lines") or 0) < min_lines:
            return f"{nm}: {w.get('lines')} decode lines < {min_lines}"
        if (w.get("cleanup") or {}).get("server_idle") is False:
            return f"{nm}: server not idle after the window"
        return None
    why = bad(base, "base") or bad(arm, "arm")
    if why:
        return {"adopt": False, "verdict": "INCONCLUSIVE", "why": why}
    lb, la = base.get("measured_level"), arm.get("measured_level")
    if not occupancy and lb != la:
        return {"adopt": False, "verdict": "INCONCLUSIVE", "why": f"running level differs: {lb} vs {la}"}
    cb, ca = base.get("ctx_per_req_p50"), arm.get("ctx_per_req_p50")
    if cb and ca and abs(ca / cb - 1.0) > ctx_tol:
        return {"adopt": False, "verdict": "INCONCLUSIVE", "why": f"context differs: {cb:.0f} vs {ca:.0f} per request"}
    b, a = base["tok_s_p50"], arm["tok_s_p50"]
    raw = a / b
    ratio = raw
    ab, aa = base.get("accept_p50"), arm.get("accept_p50")
    if per_step and ab and aa:
        # tok/s = running x accept / step: DFlash2 accept length swings per server boot (#38009), so arms that do not
        # touch the drafter are compared on decoded steps per second (tok/s / accept), not on tok/s
        ratio = (a / aa) / (b / ab)
    ok = ratio >= (1.0 + min_gain - (tol if min_gain == 0 else 0.0))
    return {"adopt": ok, "verdict": "PASS" if ok else "FAIL", "ratio": round(ratio, 3), "raw_tok_s_ratio": round(raw, 3),
            "basis": "steps/s (accept-normalised)" if (per_step and ab and aa) else "tok/s", "base": b, "arm": a,
            "accept": [ab, aa],
            "levels": [lb, la], "label": ("occupancy gain %s -> %s running" % (lb, la)) if lb != la else "same level"}


def parity_ok(cmp_arm, cmp_aa, allowance=1, identity="if_known"):
    """Correctness rule, fail closed (audits 17 HIGH-1, 18 E14-6, 19 HIGH-D):
    - INCOMPLETE unless every planned prompt was compared in BOTH the arm and the A/A control (same count), with no
      error and both margins at every divergence;
    - the arm's large-margin flips may not exceed the A/A control's (an A/A flip is numeric variation, not corruption);
    - identity: "require" = whole-sequence identity must be usable (A/A > 0 identical, else INCONCLUSIVE) and the arm
      may trail it by at most `allowance`; "if_known" = the same check only when the A/A has any identical sequence;
      "ignore" = flips decide (arms that change the arithmetic or the batching: width, NCCL, nodelay at 12 at once)."""
    if not cmp_arm or not cmp_aa:
        return {"ok": False, "verdict": "INCOMPLETE", "why": "not compared"}
    for nm, c in (("arm", cmp_arm), ("control", cmp_aa)):
        if not c.get("expected") or c.get("compared") != c.get("expected"):
            return {"ok": False, "verdict": "INCOMPLETE", "why": f"{nm}: {c.get('compared')} of {c.get('expected')} compared"}
        if c.get("missing_margins"):
            return {"ok": False, "verdict": "INCOMPLETE", "why": f"{nm}: no margins at {c['missing_margins']} divergences"}
    if cmp_arm["expected"] != cmp_aa["expected"]:
        return {"ok": False, "verdict": "INCOMPLETE", "why": f"sample counts differ: {cmp_arm['expected']} vs {cmp_aa['expected']}"}
    known = cmp_aa["identical"] > 0
    if identity == "require" and not known:
        return {"ok": False, "verdict": "INCONCLUSIVE", "why": "the A/A control has 0 identical sequences"}
    flips_ok = cmp_arm["large_margin_flips"] <= cmp_aa["large_margin_flips"]
    use_id = identity == "require" or (identity == "if_known" and known)
    ident_ok = (cmp_arm["identical"] >= cmp_aa["identical"] - allowance) if use_id else True
    ok = flips_ok and ident_ok
    return {"ok": ok, "verdict": "PASS" if ok else "FAIL", "arm_identical": cmp_arm["identical"],
            "aa_identical": cmp_aa["identical"], "large_margin_flips": cmp_arm["large_margin_flips"],
            "aa_large_margin_flips": cmp_aa["large_margin_flips"], "compared": cmp_arm["compared"],
            "identity": "checked" if use_id else "not used (flips decide)"}


def gate_reference(serial_a, serial_b, conc_a, conc_b):
    """The baseline rows a later arm gate is compared with (serial = one at a time, conc = all at once; each twice)."""
    return {"serial_a": serial_a, "serial_b": serial_b, "conc_a": conc_a, "conc_b": conc_b}


def arm_gate(ctx, prompts, ref, *, n_serial=6, n_tok=256, tag="gate", identity=True):
    """Short correctness gate run on a candidate arm BEFORE it is adopted (audits 17-19): greedy on the same prompt
    indices as the baseline reference, serial (bs 1) and all at once, compared on the first n_tok tokens against the
    reference and judged against the reference's own A/A. identity=True: bs-1 output must match the baseline
    (scheduling / graph arms); False: flips decide (arms that change the arithmetic: width, NCCL transport,
    target-only). Missing reference, incomplete samples or a dirty server = no adoption."""
    if not ref or not ref.get("serial_a") or not ref.get("conc_a"):
        return {"ok": False, "verdict": "INCOMPLETE", "why": "no baseline reference"}
    if ctx.left() < ctx.m(120):
        return {"ok": False, "verdict": "INCOMPLETE", "why": "deadline"}
    s = parity_pass(ctx, prompts[:n_serial], max_new=n_tok, tag=tag + "_serial")
    c = parity_pass(ctx, prompts[:len(ref["conc_a"])], max_new=n_tok, concurrent=True, tag=tag + "_conc")
    ps = parity_ok(compare_passes(ref["serial_a"][:n_serial], s, tag + ": serial vs baseline", n=n_tok),
                   compare_passes(ref["serial_a"][:n_serial], ref["serial_b"][:n_serial], "baseline serial A/A", n=n_tok),
                   allowance=1, identity="require" if identity else "ignore")
    pc = parity_ok(compare_passes(ref["conc_a"], c, tag + ": all-at-once vs baseline", n=n_tok),
                   compare_passes(ref["conc_a"], ref["conc_b"], "baseline all-at-once A/A", n=n_tok),
                   allowance=2, identity="ignore")
    dirty = any(r.get("server_not_idle") for r in s[:1] + c[:1])
    ok = ps["ok"] and pc["ok"] and not dirty
    return {"ok": ok, "verdict": "PASS" if ok else ("FAIL" if "FAIL" in (ps["verdict"], pc["verdict"]) else "INCOMPLETE"),
            "serial": ps, "all_at_once": pc, "server_not_idle": dirty}


def latch_gain(ref, arm, min_gain=0.10):
    """Occupancy gain at the same offered load (audits 17 §5, 18 Q2, 19 P0-C): the >= level share rises by min_gain
    and the stuck-at-level-1-with-a-queue share does not rise, or the stuck share falls by min_gain and the >= level
    share does not fall; retractions may not rise. Equal numbers = no gain (no adoption by noise)."""
    k1, k2 = "share_running_ge_level", "share_stuck_below_queued"
    if not ref or not arm or ref.get(k1) is None or arm.get(k1) is None:
        return {"gain": False, "why": "missing latch result"}
    for nm, l in (("reference", ref), ("arm", arm)):
        if not l.get("decodes_reached") or not l.get("samples") or (l.get("cleanup") or {}).get("server_idle") is False:
            return {"gain": False, "why": f"invalid {nm} latch (decodes not reached, no samples or a dirty server)"}
    ge_r, ge_a, st_r, st_a = ref[k1], arm[k1], ref.get(k2) or 0.0, arm.get(k2) or 0.0
    up = ge_a - ge_r >= min_gain and st_a <= st_r
    down = st_r - st_a >= min_gain and ge_a >= ge_r
    retr = (arm.get("retracted_delta") or 0) > (ref.get("retracted_delta") or 0)
    return {"gain": bool((up or down) and not retr), "ge_level": [ge_r, ge_a], "stuck_below": [st_r, st_a],
            "retracted_more": retr}


ERROR_SIGNS = ("runs out of memory", "Not enough space", "out of memory", "Traceback")


def log_errors(signs):
    """Server-log sign counts (substring diagnostic, not a structured classifier) that block an adoption."""
    return {k: v for k, v in (signs or {}).items() if k in ERROR_SIGNS and v}


def phase_budget(now, hard_stop, cap, reserve=0.0, margin=480.0, min_useful=120.0):
    """Seconds a phase may run: min(cap, hard_stop - now - margin - reserve), or None when less than min_useful is
    left. reserve = time kept for later work (the 25-min Duck run while A1 measures; audits 17 HIGH-2, 18 P0, 19 HIGH-E)."""
    left = float(hard_stop) - float(now) - float(margin) - float(reserve)
    return None if left < float(min_useful) else min(float(cap), left)


def launch_allowed(now, hard_stop, reserve=0.0, relaunch_cost=400.0, floor=900.0, margin=480.0, min_useful=120.0,
                   force=False):
    """A server relaunch is started only if it cannot eat the reserve: hard_stop - now >= reserve + a conservative
    relaunch cost (cold graph capture ~400 s, not the warm median) + margin + one useful phase; never inside the last
    `floor` s. force=True (the Duck's own fresh launch, restoring the kept arm) only keeps the floor."""
    left = float(hard_stop) - float(now)
    if force:
        return left >= float(floor)
    return left >= max(float(floor), float(reserve) + float(relaunch_cost) + float(margin) + float(min_useful))
