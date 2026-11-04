"""E25 NB5 FN-Duck library (stdlib + requests only; Python 3.12). New code of the NB5 notebook, kept apart from the
reused (verbatim) RC2 / E21 / E15B code so it can be unit-tested on its own. The builder embeds this file into the
notebook (written to TMP/nb5_lib.py and imported as NB5L).

Contents
- capacity: server caps from the Merlin / vLLM startup log, the usable KV pool (cache_config_info, the usage-gauge
  quantization cross-check, or the NB1 parameter), the fixed blocks per request F (log value vs NB1 parameter), the
  per-attempt block rule ceil(P / B) + F and the stream count;
- Reservation: the live-block admission gate (C2): an attempt is admitted only while the projected peak blocks of
  every admitted attempt fit floor(G x pool); running work is never touched; with nothing admitted a grant is
  immediate (no deadlock) for any need <= capacity, a need above capacity is denied (C-7); no step-down logic (W1);
- host RAM: page-cache drop (posix_fadvise DONTNEED) and Merlin's own floor rule (max - (current - inactive_file));
- commit turn variant C (W3): the server's own thinking-on render (/tokenize, same messages / tools / template
  kwargs as the chat request), the trailing '<think>\\n' replaced by '<think>\\n\\n</think>\\n\\n', sent as token ids
  to /v1/completions; the reply parsed with Duck's own markup parser into a chat-shaped result;
- Tracer: every request (full reasoning, content, tool calls, usage) in a per-branch JSONL + aggregates (prefix-hit
  tokens, recomputed tokens, per-turn prefill, tool-call rate, commit turns without a tool call, in-flight count);
- reports: running == in-flight assertion, solve-vs-tokens bands, steady-mixed / solver-window / whole-notebook
  averages, phase counter deltas, argv diff vs RC2 FP8."""
import json
import math
import os
import re
import threading
import time
from pathlib import Path

# ------------------------------------------------------------------ capacity
_ND_RE = re.compile(r"non-default args: (\{.*\})\s*$", re.M)
_KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_MC_RE = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_LCM_RE = re.compile(r"kv lcm block sizes (\d+)")


def _int_after(text, key):
    m = re.search(r"'" + re.escape(key) + r"': (\d+)", text or "")
    return int(m.group(1)) if m else None


def server_caps(log_text):
    """What the server itself says it runs with (memory rule: read the caps from the log, never assume them):
    max_num_seqs / block_size / max_model_len from the LAST 'non-default args' line, the speculative config, the KV
    line and the max-concurrency line. Missing fields are None."""
    t = log_text or ""
    nd = _ND_RE.findall(t)
    last = nd[-1] if nd else ""
    out = {"max_num_seqs": _int_after(last, "max_num_seqs"), "block_size": _int_after(last, "block_size"),
           "max_model_len": _int_after(last, "max_model_len"), "non_default_found": bool(nd)}
    lcm = _LCM_RE.findall(t)
    if out["block_size"] is None and lcm:
        out["block_size"] = int(lcm[-1])
    m = re.search(r"'speculative_config': (\{[^}]*\}|None)", last)
    out["speculative_config"] = m.group(1) if m else None
    k = re.search(r"'num_speculative_tokens': (\d+)", last)
    out["spec_tokens"] = int(k.group(1)) if k else 0
    kv = _KV_RE.findall(t)
    mc = _MC_RE.findall(t)
    out["kv_tokens"] = int(kv[-1].replace(",", "")) if kv else None
    out["conc_tokens"] = int(mc[-1][0].replace(",", "")) if mc else None
    out["conc_x"] = float(mc[-1][1]) if mc else None
    return out


_CCI_RE = re.compile(r'^vllm:cache_config_info\{([^}]*)\}\s+\S+', re.M)


def pool_from_metrics(metrics_text):
    """num_gpu_blocks from vLLM's cache_config_info gauge labels; the usable pool is that minus the null block
    (RC2: the usage gauge moves in steps of 1/149 with Merlin's 150-block pin)."""
    m = _CCI_RE.search(metrics_text or "")
    if not m:
        return None
    lab = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    try:
        n = int(lab.get("num_gpu_blocks"))
    except (TypeError, ValueError):
        return None
    return {"num_gpu_blocks": n, "usable": n - 1, "block_size": int(lab["block_size"]) if str(lab.get("block_size", "")).isdigit() else None}


def pool_from_gauge(samples, lo=20, hi=4000, tol=0.002):
    """The smallest N in [lo, hi] for which every usage sample x is a multiple of 1/N (to within tol blocks): the pool
    size the KV-usage gauge counts in. Needs >= 3 distinct non-zero samples, else None (a cross-check only)."""
    xs = sorted({round(float(x), 9) for x in samples if x is not None and 0 < float(x) <= 1})
    if len(xs) < 3:
        return None
    for n in range(lo, hi + 1):
        if all(abs(x * n - round(x * n)) <= tol * max(1.0, x * n) for x in xs):
            return n
    return None


def cdiv(a, b):
    return -(-int(a) // int(b))


def capacity_plan(pool_blocks, block, budgets, prompt_allow, tool_allow, max_len, f_param, gate=0.95, seqs=None,
                  conc_tokens=None, conc_x=None, f_extra=0, f_rule="max"):
    """The pre-run capacity check (memory rule: tokens needed vs available, resident contexts vs KV).
    F_log = pool / X - cdiv(M, B) from the server's own max-concurrency line; F_eff = max(F_param, ceil(F_log - 0.05)).
    Per attempt kind b: projected peak P_b = min(max_len, prompt_allow + budget_b + tool_allow) tokens ->
    cdiv(P_b, B) + F_eff blocks. C = floor(gate x pool) admits; streams_b = C // blocks_b; workers = min(seqs,
    max_b streams_b) (the reservation decides the real admissions).
    C-8 (audit 3): workers may be 0 (no attempt kind fits the pool): verdict NO_GO_INSUFFICIENT_KV, the caller runs no
    solver (the fallback submission stays); never rounded up to 1.
    f_rule 'measured' (lever b, NB5QUALITY_R2): F_eff = ceil(F_log - 0.05) from the server's own line when it is there
    (a smaller F from a non-numeric lever is used), f_param only when the line is missing."""
    f_log = None
    if conc_tokens and conc_x:
        f_log = pool_blocks / float(conc_x) - cdiv(conc_tokens, block)
    if f_rule == "measured" and f_log is not None:
        f_eff = max(1, int(math.ceil(f_log - 0.05))) + int(f_extra)
    else:
        f_eff = max(int(f_param), int(math.ceil(f_log - 0.05)) if f_log is not None else 0) + int(f_extra)
    cap = int(math.floor(float(gate) * pool_blocks))
    per = {}
    for k, b in budgets.items():
        p = min(int(max_len), int(prompt_allow) + int(b) + int(tool_allow))
        nb = cdiv(p, block) + f_eff
        per[str(k)] = {"budget": int(b), "projected_peak_tokens": p, "blocks": nb, "streams": cap // nb}
    streams = max((v["streams"] for v in per.values()), default=0)
    workers = min(int(seqs), streams) if seqs else streams
    fit = {str(L): cap // (cdiv(min(L, max_len), block) + f_eff) for L in (40000, 60000, 70000, 90000, 120000, max_len)}
    return {"pool_blocks": int(pool_blocks), "block_size": int(block), "gate": float(gate), "capacity_blocks": cap,
            "f_param": int(f_param), "f_log": None if f_log is None else round(f_log, 3), "f_eff": f_eff, "f_rule": f_rule,
            "f_extra": int(f_extra), "per_attempt": per, "streams": streams, "seqs": seqs, "workers": max(0, int(workers)),
            "verdict": "OK" if workers >= 1 else "NO_GO_INSUFFICIENT_KV",
            "streams_fit_by_context": fit, "prompt_allow": int(prompt_allow), "tool_allow": int(tool_allow),
            "max_len": int(max_len)}


class Reservation:
    """C2 live-block admission: acquire(key, blocks, deadline) blocks until used + blocks <= capacity (or nothing is
    held: never a deadlock), or returns False at the monotonic deadline. release(key) frees and wakes the waiters.
    Running work is never touched; there is no step-down: the only correction is a larger F for FUTURE admissions after
    a real preemption (correct()).
    C-7 (audit 3): a need above the whole capacity is denied at once (KV_NEED_ABOVE_POOL, counted; the caller finalizes
    the attempt without a request); the idle grant (nothing held) applies only to needs <= capacity."""

    def __init__(self, capacity_blocks, clock=time.monotonic):
        self.cap = int(capacity_blocks)
        self.clock = clock
        self.used = 0
        self.held = {}
        self.f_extra = 0
        self.live_hold = None          # a reason string while the SERVER shows it cannot admit (waiting > 0, running < in-flight)
        self.waiting = 0               # attempts blocked in acquire() right now
        self.cond = threading.Condition()
        self.stats = {"grants": 0, "waits": 0, "wait_sec": 0.0, "denied": 0, "max_used": 0, "max_held": 0,
                      "over_cap_grants": 0, "idle_grants": 0, "releases": 0, "corrections": 0, "live_hold_waits": 0,
                      "live_holds": 0, "above_pool_denials": 0}
        self.events = []

    def above_pool(self, blocks):
        """C-7: this need (with the current F correction) can never fit the capacity."""
        with self.cond:
            return int(blocks) + int(self.f_extra) > self.cap

    def acquire(self, key, blocks, deadline):
        need = int(blocks) + int(self.f_extra)
        with self.cond:
            t0, waited = self.clock(), False
            while True:
                if key in self.held:
                    return True
                if need > self.cap:                 # C-7: never granted, not even with nothing held
                    self.stats["above_pool_denials"] += 1
                    self.stats["denied"] += 1
                    if len(self.events) < 200:
                        self.events.append({"t": round(time.time(), 1), "denied": "KV_NEED_ABOVE_POOL", "need": need,
                                            "capacity": self.cap, "key": str(key)[:80]})
                    return False
                if self.held and self.live_hold:
                    if not waited:
                        self.stats["live_hold_waits"] += 1
                elif not self.held or self.used + need <= self.cap:
                    if not self.held:
                        self.stats["idle_grants"] += 1
                    self.held[key] = need
                    self.used += need
                    self.stats["grants"] += 1
                    self.stats["max_used"] = max(self.stats["max_used"], self.used)
                    self.stats["max_held"] = max(self.stats["max_held"], len(self.held))
                    if waited:
                        self.stats["wait_sec"] += self.clock() - t0
                    return True
                now = self.clock()
                if now >= deadline:
                    self.stats["denied"] += 1
                    if waited:
                        self.stats["wait_sec"] += now - t0
                    return False
                if not waited:
                    waited = True
                    self.stats["waits"] += 1
                self.waiting += 1
                try:
                    self.cond.wait(timeout=min(5.0, max(0.02, deadline - now)))
                finally:
                    self.waiting -= 1

    def update(self, key, blocks):
        """'live' admission mode (addendum 3): an admitted attempt's blocks follow its LIVE context
        (ceil(L / B) + F after each of its requests) instead of its projected peak; retained blocks of finished turns
        are not counted (Merlin reclaims them). Smaller reservations -> more admissions, guarded by the live hold and
        the preemption correction."""
        with self.cond:
            if key in self.held:
                n = int(blocks) + int(self.f_extra)
                self.used += n - self.held[key]
                self.held[key] = n
                self.stats["max_used"] = max(self.stats["max_used"], self.used)
                self.cond.notify_all()

    def release(self, key):
        with self.cond:
            n = self.held.pop(key, None)
            if n is not None:
                self.used -= n
                self.stats["releases"] += 1
            self.cond.notify_all()

    def set_live_hold(self, why):
        """Live observation (addendum 2): while the server itself shows it cannot admit (requests waiting although the
        client has no more in flight than are running), no new attempt is granted (idle grants still happen). Cleared
        with why=None. Never touches running work."""
        with self.cond:
            if why and not self.live_hold:
                self.stats["live_holds"] += 1
                if len(self.events) < 200:
                    self.events.append({"t": round(time.time(), 1), "live_hold": str(why)[:200]})
            self.live_hold = why or None
            self.cond.notify_all()

    def frac(self):
        with self.cond:
            return self.used / float(self.cap) if self.cap > 0 else 1.0

    def correct(self, extra_blocks, why):
        with self.cond:
            self.f_extra += int(extra_blocks)
            self.stats["corrections"] += 1
            if len(self.events) < 200:
                self.events.append({"t": round(time.time(), 1), "f_extra": self.f_extra, "why": str(why)[:200]})

    def snapshot(self):
        with self.cond:
            return {"capacity_blocks": self.cap, "used": self.used, "held": len(self.held), "f_extra": self.f_extra,
                    "live_hold": self.live_hold, "waiting_now": self.waiting,
                    **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.stats.items()},
                    "events": list(self.events)}


# ------------------------------------------------------------------ host RAM (spec section 2)
def drop_page_cache(roots, resolve=True):
    """posix_fadvise(DONTNEED) on every file under `roots` (and, with resolve, each symlink's target: the merged
    checkpoint is symlinks into the Model mounts). Linux only; elsewhere {'supported': False}."""
    out = {"supported": hasattr(os, "posix_fadvise"), "files": 0, "bytes": 0, "errors": 0, "sec": None}
    if not out["supported"]:
        return out
    t0, seen = time.time(), set()
    for r in roots:
        rp = Path(r)
        if not rp.exists():
            continue
        it = [rp] if rp.is_file() else rp.rglob("*")
        for p in it:
            try:
                q = p.resolve() if resolve else p
                if not q.is_file() or str(q) in seen:
                    continue
                seen.add(str(q))
                fd = os.open(str(q), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                out["files"] += 1
                out["bytes"] += q.stat().st_size
            except Exception:
                out["errors"] += 1
    out["sec"] = round(time.time() - t0, 1)
    return out


def ram_gate(host_mem_fn, need_mib, drop_fn=None, wait_sec=120.0, poll=5.0):
    """Merlin's own floor rule on the cgroup: avail = max - (current - inactive_file) >= need_mib (69 GiB + 4 GiB).
    First the page-cache drop, then up to wait_sec for the kernel to reclaim. ok True / False; None when the cgroup
    numbers are unreadable (recorded, never a pass by default: the caller decides)."""
    rec = {"need_mib": int(need_mib), "before": host_mem_fn()}
    if drop_fn is not None:
        rec["drop"] = drop_fn()
    t_end = time.time() + float(wait_sec)
    while True:
        h = host_mem_fn()
        a = h.get("avail_mib") if isinstance(h, dict) else None
        if a is None:
            rec.update(after=h, ok=None, why="cgroup memory numbers unavailable")
            return rec
        if a >= need_mib:
            rec.update(after=h, ok=True)
            return rec
        if time.time() >= t_end:
            rec.update(after=h, ok=False, why=f"avail {a} MiB < {need_mib} MiB after the drop + {wait_sec:.0f} s")
            return rec
        time.sleep(poll)


# ------------------------------------------------------------------ commit turn variant C (W3)
THINK_OPEN = "<think>\n"
THINK_CLOSED = "<think>\n\n</think>\n\n"


def tails(tok_encode):
    """(TAIL_A, TAIL_C) token ids: the thinking-on generation-prompt end and its hand-closed replacement."""
    return list(tok_encode(THINK_OPEN)), list(tok_encode(THINK_CLOSED))


def variant_c_ids(ids, tail_a, tail_c):
    """The thinking-on render with the trailing '<think>\\n' replaced by '<think>\\n\\n</think>\\n\\n' (token level);
    None when the render does not end with tail_a (then the commit request falls back to the normal chat path)."""
    ids = list(ids)
    if len(ids) < len(tail_a) or ids[-len(tail_a):] != list(tail_a):
        return None
    return ids[:-len(tail_a)] + list(tail_c)


def commit_variant(mode, agent):
    """'C' / 'A' for one commit request. mode 'C' | 'A' | 'split' (the paired SMOKE comparison: a stable hash of
    (task, branch) picks the variant, so each attempt uses one variant for all its commit requests)."""
    if mode in ("C", "A"):
        return mode
    import hashlib
    key = f"{getattr(agent, 'arc2_task_id', '')}_{getattr(agent, 'arc2_branch', '')}"
    return "C" if hashlib.sha256(key.encode()).digest()[0] % 2 == 0 else "A"


def is_commit(agent):
    """The Duck B-mode commit turn has been sent (E15B #48): every later request of the attempt is a commit request."""
    return bool(getattr(agent, "_arc2_b", False) and getattr(agent, "_arc2_b_commit_sent", False))


def chat_payload(TA, agent, messages, kw):
    """The exact chat payload Duck's ToolAgent._chat_completion would send (tool_agent.py: build_chat_payload, the
    template-kwargs merge, enable_thinking override, control keys stripped); provider prefs are OpenRouter-only."""
    tools = kw.get("tools")
    max_tokens = kw.get("max_tokens")
    enable_thinking = kw.get("enable_thinking")
    payload = TA.build_chat_payload(
        provider=agent._model.provider, model=agent._model.model_id, messages=messages,
        max_tokens=agent._max_output_tokens if max_tokens is None else max_tokens,
        temperature=TA._LOCAL_ANALYZER_TEMPERATURE, top_p=TA._LOCAL_ANALYZER_TOP_P, top_k=TA._LOCAL_ANALYZER_TOP_K,
        thinking=(bool(TA._LOCAL_ANALYZER_ENABLE_THINKING) if enable_thinking is None else enable_thinking),
        tools=tools, tool_choice=TA._request_tool_choice(tools), seed=TA._LOCAL_ANALYZER_SEED)
    tk = dict(payload.get("chat_template_kwargs") or {})
    tk.update(agent._harness_template_kwargs())
    if enable_thinking is not None:
        tk["enable_thinking"] = bool(enable_thinking)
    if tk:
        payload["chat_template_kwargs"] = tk
    payload["messages"] = TA._strip_control_keys(TA._apply_summary_visibility(
        payload.get("messages") or [], evicted=getattr(agent, "_has_evicted", False)))
    return payload


def _root(base_url):
    b = str(base_url).rstrip("/")
    return b[:-3] if b.endswith("/v1") else b


def _post(url, body, timeout):
    import requests
    r = requests.post(url, json=body, timeout=(10.0, float(timeout)))
    if r.status_code >= 400:
        raise requests.RequestException(f"{r.status_code} Error | {url} | response: {r.text.strip()[:600]}")
    return r.json()


def tokenize_chat(base_url, payload, timeout=120.0):
    """The SERVER's render of the chat payload (vLLM /tokenize: the same chat preprocessing and default template
    kwargs as /v1/chat/completions)."""
    body = {"model": payload.get("model"), "messages": payload.get("messages") or [], "add_generation_prompt": True}
    if payload.get("tools"):
        body["tools"] = payload["tools"]
    if payload.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = payload["chat_template_kwargs"]
    out = _post(_root(base_url) + "/tokenize", body, timeout)
    return list(out.get("tokens") or [])


def parse_c_reply(TA, text, n=0):
    """A raw completion after a closed think block -> a chat-shaped assistant message with Duck's own markup parser
    (the qwen3_coder XML the template asks for)."""
    t = text or ""
    calls = TA._recover_tool_calls_from_markup(t)
    content = TA._strip_tool_call_markup(t) if calls else t.strip()
    msg = {"role": "assistant", "content": content, "reasoning": None}
    if calls:
        msg["tool_calls"] = [{"id": f"chatcmpl-tool-nb5c-{n}-{i}", "type": "function", "function": c["function"]}
                             for i, c in enumerate(calls)]
    return msg, len(calls), ("</think>" in t or "<think>" in t)


def commit_c_call(TA, agent, messages, kw, base_url, tail_a, tail_c, timeout, n=0, ids=None):
    """One commit request as variant C. Returns (result, info); result None = use the normal chat path (info['why']).
    Transport errors raise requests.RequestException (Duck's retryable-failure path), as the chat path does."""
    payload = chat_payload(TA, agent, messages, kw)
    info = {"why": None}
    if (payload.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
        info["why"] = "enable_thinking False requested"
        return None, info
    try:
        if ids is None:
            ids = tokenize_chat(base_url, payload, timeout=min(120.0, max(5.0, float(timeout))))
    except Exception as e:
        info["why"] = f"tokenize failed: {type(e).__name__}: {e}"[:300]
        return None, info
    ids_c = variant_c_ids(ids, tail_a, tail_c)
    if ids_c is None:
        info.update(why="render does not end with '<think>\\n'", tail=ids[-6:])
        return None, info
    body = {"model": payload.get("model"), "prompt": ids_c, "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"), "top_p": payload.get("top_p"), "stream": False}
    for k in ("top_k", "seed"):
        if payload.get(k) is not None:
            body[k] = payload[k]
    t0 = time.time()
    out = TA._post_with_retries(lambda: __import__("requests").post(_root(base_url) + "/v1/completions", json=body,
                                                                    timeout=(10.0, float(timeout))),
                                retries=int(os.environ.get("ARC3_HTTP_RETRIES", "3") or 3),
                                base_seconds=float(os.environ.get("ARC3_HTTP_RETRY_BASE_SECONDS", "5") or 5),
                                max_seconds=float(os.environ.get("ARC3_HTTP_RETRY_MAX_SECONDS", "5") or 5))
    if out.status_code >= 400:
        import requests
        raise requests.RequestException(f"{out.status_code} Error | variant C | response: {out.text.strip()[:600]}")
    d = out.json()
    ch = (d.get("choices") or [{}])[0]
    msg, ncalls, leak = parse_c_reply(TA, ch.get("text") or "", n)
    fin = "tool_calls" if ncalls else str(ch.get("finish_reason") or "")
    info.update(prompt_ids=len(ids_c), render_ids=len(ids), calls=ncalls, think_markup_in_reply=leak,
                sec=round(time.time() - t0, 2), finish=ch.get("finish_reason"))
    return TA._ChatCompletionResult(message=msg, finish_reason=fin, usage=d.get("usage"), served_by=""), info


# ------------------------------------------------------------------ request tracer
def _calls_code_valid(calls):
    import ast as _ast
    try:
        for c in calls or []:
            a = (c.get("function") or {}).get("arguments")
            a = json.loads(a) if isinstance(a, str) else (a or {})
            if not isinstance(a, dict) or not str(a.get("code") or "").strip():
                return False
            _ast.parse(a["code"])
        return bool(calls)
    except Exception:
        return False


def variant_outcomes(records, solutions, mode, cls_of_agent=None):
    """The paired split (PLAN v3.6): per commit variant, the attempts that reached their commit turn, how many of them
    have a host-accepted candidate exactly right on every test input, and how many hold a non-grid candidate output."""
    out = {"A": {"attempts": 0, "exact": 0, "invalid_grid_candidates": 0}, "C": {"attempts": 0, "exact": 0, "invalid_grid_candidates": 0}}
    for tid, rows in (records or {}).items():
        gold = (solutions or {}).get(tid)
        for r in rows or []:
            if (r.get("b_events") or {}).get("commit_at_tokens") is None:
                continue
            ag = type("K", (), {"arc2_task_id": tid, "arc2_branch": r.get("branch")})()
            v = commit_variant(mode, ag)
            o = out[v]
            o["attempts"] += 1
            cands = (r.get("submissions") or []) + (r.get("checked") or [])
            if gold is not None and any(c.get("accepted") and len(c.get("test_outputs") or []) == len(gold)
                                        and all(a == b for a, b in zip(c["test_outputs"], gold)) for c in cands):
                o["exact"] += 1
            o["invalid_grid_candidates"] += sum(1 for c in cands if c.get("test_outputs") is not None
                                                and not all(_valid_grid(g) for g in c["test_outputs"]))
    for o in out.values():
        o["exact_rate"] = round(o["exact"] / o["attempts"], 3) if o["attempts"] else None
    return out


class Tracer:
    """Every request of every attempt: aggregates + a per-branch JSONL with the FULL reasoning / content / tool calls
    (memory rule: save the full thinking of every request)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.out_dir = None
        self.phase = None
        self.inflight = 0
        self.max_inflight = 0
        self.agg = {}
        self.new_phase(None, None)

    def new_phase(self, name, out_dir):
        with self.lock:
            self.phase, self.out_dir = name, (Path(out_dir) if out_dir else None)
            self.agg = {"requests": 0, "errors": 0, "prompt_tokens": 0, "cached_tokens": 0, "cached_reported": 0,
                        "completion_tokens": 0, "with_tool_call": 0, "finish": {}, "kinds": {}, "uncached_per_request": [],
                        "first_request_uncached": [], "later_request_uncached": [], "cached_block_floor_tokens": 0,
                        "commit_requests": 0, "commit_with_tool_call": 0, "commit_C": 0, "commit_C_with_tool_call": 0,
                        "commit_by_variant": {}, "non_leading_system_requests": 0,
                        "natural_stop": 0, "natural_tool_call": 0, "cut_length": 0, "open_think_other": 0,
                        "later_requests": 0, "later_with_cache_hit": 0, "later_cached_tokens": 0,
                        "first_requests": 0, "first_with_cache_hit": 0, "latency_sec": [],
                        "commit_C_fallbacks": {}, "commit_C_think_markup": 0, "commit_first_no_call": 0,
                        "commit_attempts": 0, "attempt_keys": 0, "trace_write_errors": 0,
                        "by_attempt": {}, "no_request": {}}
            self._seen_attempt = set()
            self._commit_first = {}
            self.commit_samples = {}       # variant -> the first commit request (messages / tools / kwargs) for the replay

    def enter(self):
        with self.lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)

    def exit(self):
        with self.lock:
            self.inflight = max(0, self.inflight - 1)

    def keep_commit_sample(self, variant, payload, agent):
        with self.lock:
            if variant not in self.commit_samples and isinstance(payload, dict):
                self.commit_samples[variant] = {"task": getattr(agent, "arc2_task_id", None),
                                                "branch": getattr(agent, "arc2_branch", None),
                                                "payload": json.loads(json.dumps(payload, default=str))}

    def record(self, agent, kind, kw, result, t0, t1, error=None, info=None, block=4096, messages=None):
        msg = getattr(result, "message", None) if result is not None else None
        msg = msg if isinstance(msg, dict) else {}
        u = getattr(result, "usage", None) if result is not None else None
        u = u if isinstance(u, dict) else {}
        det = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
        p, c, o = int(u.get("prompt_tokens") or 0), int(det.get("cached_tokens") or 0), int(u.get("completion_tokens") or 0)
        calls = msg.get("tool_calls") or []
        fin = str(getattr(result, "finish_reason", "") or "") if result is not None else ""
        key = (getattr(agent, "arc2_task_id", None), getattr(agent, "arc2_branch", None))
        commit = kind.startswith("commit")
        with self.lock:
            a = self.agg
            first = key not in self._seen_attempt
            self._seen_attempt.add(key)
            a["attempt_keys"] = len(self._seen_attempt)
            a["requests"] += 1
            a["kinds"][kind] = a["kinds"].get(kind, 0) + 1
            ba = a["by_attempt"].setdefault(f"{key[0]}|{key[1]}", {"requests": 0, "errors": 0, "completion": 0,
                                                                  "commit_requests": 0, "commit_with_tool_call": 0,
                                                                  "no_request": None})
            ba["requests"] += 1
            ba["errors"] += int(error is not None or result is None)
            ba["completion"] += o if (error is None and result is not None) else 0
            if commit:
                ba["commit_requests"] += 1
                ba["commit_with_tool_call"] += int(bool(calls))
            if error is not None or result is None:
                a["errors"] += 1
            else:
                a["prompt_tokens"] += p
                a["cached_tokens"] += c
                a["cached_reported"] += int("cached_tokens" in det)
                a["completion_tokens"] += o
                a["cached_block_floor_tokens"] += (c // block) * block
                a["with_tool_call"] += int(bool(calls))
                a["finish"][fin] = a["finish"].get(fin, 0) + 1
                # addendum 2: a turn is finished only with its think block closed (content or a parsed call after
                # </think>) and a natural finish; 'length' (incl. E21's restored cut calls) is a cut
                closed = bool(str(msg.get("content") or "").strip() or calls)
                if fin == "stop" and closed:
                    a["natural_stop"] += 1
                elif fin == "tool_calls" and closed:
                    a["natural_tool_call"] += 1
                elif fin == "length":
                    a["cut_length"] += 1
                else:
                    a["open_think_other"] += 1
                if len(a["uncached_per_request"]) < 20000:
                    a["uncached_per_request"].append(p - c)
                    a.setdefault("turns", []).append([key[0], key[1], kind, p, c, o, fin, int(bool(calls)),
                                                      round(t1 - t0, 2), int(fin in ("stop", "tool_calls") and closed)])
                (a["first_request_uncached"] if first else a["later_request_uncached"]).append(p - c)
                # retained-session hits (addendum 3): a later turn of an attempt that found its earlier prefix cached
                if first:
                    a["first_requests"] += 1
                    a["first_with_cache_hit"] += int(c > 0)
                else:
                    a["later_requests"] += 1
                    a["later_with_cache_hit"] += int(c > 0)
                    a["later_cached_tokens"] += c
                if len(a["latency_sec"]) < 20000:
                    a["latency_sec"].append(round(t1 - t0, 2))
            if messages and any(isinstance(m, dict) and str(m.get("role", "")).strip() == "system" for m in messages[1:]):
                a["non_leading_system_requests"] += 1      # RC2 log 389 (Merlin merges inline system messages)
            if commit:
                v = "C" if kind == "commit_C" else "A"
                cv = a["commit_by_variant"].setdefault(v, {"requests": 0, "with_tool_call": 0, "prompt": 0, "cached": 0,
                                                           "completion": 0, "errors": 0, "code_valid": 0, "corrupt": 0})
                ok_code = _calls_code_valid(calls)
                cv["code_valid"] += int(ok_code)
                # no-corruption (PLAN v3.6): a call whose code does not parse, or think markup in a closed-think reply
                cv["corrupt"] += int(bool(calls) and not ok_code) + int(bool((info or {}).get("think_markup_in_reply")))
                cv["requests"] += 1
                cv["with_tool_call"] += int(bool(calls))
                cv["prompt"] += p
                cv["cached"] += c
                cv["completion"] += o
                cv["errors"] += int(error is not None or result is None)
                a["commit_requests"] += 1
                a["commit_with_tool_call"] += int(bool(calls))
                if key not in self._commit_first:
                    self._commit_first[key] = bool(calls)
                    a["commit_attempts"] += 1
                    a["commit_first_no_call"] += int(not calls)
                if kind == "commit_C":
                    a["commit_C"] += 1
                    a["commit_C_with_tool_call"] += int(bool(calls))
                    a["commit_C_think_markup"] += int(bool((info or {}).get("think_markup_in_reply")))
                elif kind.startswith("commit_A_fallback"):
                    w = (info or {}).get("why") or "?"
                    a["commit_C_fallbacks"][w[:60]] = a["commit_C_fallbacks"].get(w[:60], 0) + 1
            d = self.out_dir
        if d is None or key[0] is None:
            return
        row = {"t0": round(t0, 3), "t1": round(t1, 3), "phase": self.phase, "task": key[0], "branch": key[1], "kind": kind,
               "max_tokens": kw.get("max_tokens"), "prompt": p, "cached": c, "completion": o, "finish": fin,
               "error": None if error is None else str(error)[:600], "info": info,
               "generated_tokens_after": getattr(agent, "generated_tokens", None),
               "reasoning": msg.get("reasoning") if msg.get("reasoning") is not None else msg.get("reasoning_content"),
               "content": msg.get("content"),
               "tool_calls": [{"name": (x.get("function") or {}).get("name"), "arguments": (x.get("function") or {}).get("arguments")}
                              for x in calls if isinstance(x, dict)]}
        try:
            f = d / f"{key[0]}_b{key[1]}" / "nb5_requests.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            with self.lock, open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception:
            with self.lock:
                self.agg["trace_write_errors"] += 1

    def note_no_request(self, agent, verdict, info=None):
        """A-13 / A-14 / C-7: a request that was NOT sent (no room, no exact prompt length, KV need above the
        pool). Counted apart from the requests (never in the request metrics) and
        written to the attempt's trace as a 'no_request' row."""
        key = (getattr(agent, "arc2_task_id", None), getattr(agent, "arc2_branch", None))
        with self.lock:
            a = self.agg
            a["no_request"][verdict] = a["no_request"].get(verdict, 0) + 1
            ba = a["by_attempt"].setdefault(f"{key[0]}|{key[1]}", {"requests": 0, "errors": 0, "completion": 0,
                                                                  "commit_requests": 0, "commit_with_tool_call": 0,
                                                                  "no_request": None})
            ba["no_request"] = verdict
            d = self.out_dir
        if d is None or key[0] is None:
            return
        try:
            f = d / f"{key[0]}_b{key[1]}" / "nb5_requests.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            with self.lock, open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t0": round(time.time(), 3), "phase": self.phase, "task": key[0], "branch": key[1],
                                     "kind": "no_request", "verdict": verdict, "info": info,
                                     "generated_tokens_after": getattr(agent, "generated_tokens", None)}, default=str) + "\n")
        except Exception:
            with self.lock:
                self.agg["trace_write_errors"] += 1

    def summary(self):
        with self.lock:
            a = json.loads(json.dumps(self.agg))
            infl, mx = self.inflight, self.max_inflight
        q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else None
        p = a["prompt_tokens"]
        a.update(prefix_hit_pct=round(100.0 * a["cached_tokens"] / p, 2) if p else None,
                 recomputed_prompt_tokens=p - a["cached_tokens"],
                 prefix_hit_pct_at_block_floor=round(100.0 * a["cached_block_floor_tokens"] / p, 2) if p else None,
                 tool_call_rate=round(a["with_tool_call"] / max(1, a["requests"] - a["errors"]), 4),
                 commit_tool_call_rate=round(a["commit_with_tool_call"] / a["commit_requests"], 4) if a["commit_requests"] else None,
                 commit_C_tool_call_rate=round(a["commit_C_with_tool_call"] / a["commit_C"], 4) if a["commit_C"] else None,
                 commit_turns_without_tool_call=a["commit_first_no_call"],
                 per_turn_prefill_p50=q(a["uncached_per_request"], .5), per_turn_prefill_p90=q(a["uncached_per_request"], .9),
                 per_turn_prefill_max=max(a["uncached_per_request"]) if a["uncached_per_request"] else None,
                 later_turn_prefill_p50=q(a["later_request_uncached"], .5),
                 retained_session_hit_share=round(a["later_with_cache_hit"] / a["later_requests"], 4) if a["later_requests"] else None,
                 latency_sec_p50=q(a["latency_sec"], .5), latency_sec_p90=q(a["latency_sec"], .9),
                 latency_sec_max=max(a["latency_sec"]) if a["latency_sec"] else None,
                 first_turn_prefill_p50=q(a["first_request_uncached"], .5),
                 inflight_now=infl, max_inflight=mx)
        for v in a["commit_by_variant"].values():
            v["cached_pct"] = round(100.0 * v["cached"] / v["prompt"], 2) if v["prompt"] else None
            v["tool_call_rate"] = round(v["with_tool_call"] / v["requests"], 4) if v["requests"] else None
            v["completion_mean"] = round(v["completion"] / v["requests"], 1) if v["requests"] else None
        for k in ("uncached_per_request", "first_request_uncached", "later_request_uncached", "turns", "latency_sec"):
            if k in a:
                a[k] = a[k][:3000]
        a["turn_fields"] = ["task", "branch", "kind", "prompt", "cached", "completion", "finish", "tool_call", "latency_sec",
                            "natural_closed"]
        return a


# ------------------------------------------------------------------ reports
def running_check(samples, workers, min_share=0.9):
    """Memory rule 'real running == level': in every sample where the client had `workers` requests in flight, the
    server must report the same number running. PASS when >= min_share of those samples agree; INCONCLUSIVE without
    any such sample. samples: [{'inflight', 'running', 'waiting'}]."""
    full = [s for s in samples if s.get("inflight") is not None and s.get("running") is not None
            and int(s["inflight"]) >= int(workers) > 0]
    ok = [s for s in full if int(s["running"]) == int(s["inflight"])]
    lt = [s for s in full if int(s["running"]) < int(s["inflight"])]
    gt = [s for s in full if int(s["running"]) > int(s["inflight"])]
    anyin = [s for s in samples if (s.get("inflight") or 0) > 0 and s.get("running") is not None]
    share = len(ok) / len(full) if full else None
    return {"workers": int(workers), "samples": len(samples), "full_samples": len(full), "agree": len(ok),
            "server_below_client": len(lt), "server_above_client": len(gt), "agree_share": None if share is None else round(share, 3),
            "busy_samples": len(anyin),
            "busy_agree_share": round(sum(1 for s in anyin if int(s["running"]) == int(s["inflight"])) / len(anyin), 3) if anyin else None,
            "verdict": "INCONCLUSIVE_NO_FULL_SAMPLE" if not full else ("PASS" if share >= min_share else "FAIL_RUNNING_NE_INFLIGHT")}


BAND_EDGES = (30000, 45000, 60000, 90000, 120000)


def _accepted_correct(rec, gold):
    if gold is None:
        return None
    for s in (rec.get("submissions") or []) + (rec.get("checked") or []):
        to = s.get("test_outputs") or []
        if s.get("accepted") and len(to) == len(gold) and all(a == b for a, b in zip(to, gold)):
            return True
    return False


def solve_bands(records, solutions=None, edges=BAND_EDGES):
    """Solve-vs-tokens: each attempt's generated tokens when it ended (an accepted attempt ends at its accept), by band;
    accepted / host-accepted-and-test-correct counts per band and the cumulative share of all accepts (and correct
    accepts) reached by each edge."""
    names = [f"<={e // 1000}k" for e in edges] + [f">{edges[-1] // 1000}k"]
    bands = {n: {"attempts": 0, "accepted": 0, "correct": 0, "weak_accepted": 0} for n in names}
    acc_tok, cor_tok = [], []
    for tid, rows in (records or {}).items():
        gold = (solutions or {}).get(tid) if solutions is not None else None
        for r in rows or []:
            g = int(r.get("generated_tokens") or 0)
            i = next((k for k, e in enumerate(edges) if g <= e), len(edges))
            b = bands[names[i]]
            b["attempts"] += 1
            cands = (r.get("submissions") or []) + (r.get("checked") or [])
            # E15B weak commit (rev 3k W1): a commit-zone accept is a host-accepted candidate (weak) without the
            # attempt-level 'accepted' flag; both count as accepted here (weak ones also counted apart)
            if r.get("accepted") or any(v.get("accepted") for v in cands):
                b["accepted"] += 1
                acc_tok.append(g)
                if any(v.get("accepted") and v.get("weak") for v in (r.get("submissions") or []) + (r.get("checked") or [])):
                    b["weak_accepted"] += 1
                c = _accepted_correct(r, gold)
                if c:
                    b["correct"] += 1
                    cor_tok.append(g)
    cum = {f"<={e // 1000}k": {"accepted_share": round(sum(1 for x in acc_tok if x <= e) / len(acc_tok), 3) if acc_tok else None,
                               "correct_share": round(sum(1 for x in cor_tok if x <= e) / len(cor_tok), 3) if cor_tok else None}
           for e in edges}
    return {"edges": list(edges), "bands": bands, "cumulative": cum, "accepted": len(acc_tok), "correct": len(cor_tok),
            "scored": solutions is not None}


def task_solve_times(records, solutions=None):
    """Per task (addendum 4, the paired quality mode): solved (any host-accepted candidate exactly right on every test
    input), first accept time after the task's first attempt started, attempts, tokens; plus completeness (attempts
    that ended by accept or at budget vs cut by the deadline)."""
    out = {}
    for tid, rows in (records or {}).items():
        rows = rows or []
        t0 = min([float(r.get("t0")) for r in rows if r.get("t0")] or [0.0]) or None
        gold = (solutions or {}).get(tid) if solutions is not None else None
        acc_t = []
        ok = False
        for r in rows:
            for v in (r.get("submissions") or []) + (r.get("checked") or []):
                if v.get("accepted"):
                    if v.get("t"):
                        acc_t.append(float(v["t"]))
                    to = v.get("test_outputs") or []
                    if gold is not None and len(to) == len(gold) and all(a == b for a, b in zip(to, gold)):
                        ok = True
        out[tid] = {"attempts": len(rows), "accepted": any(r.get("accepted") for r in rows),
                    "correct": ok if gold is not None else None,
                    "first_accept_sec": round(min(acc_t) - t0, 1) if acc_t and t0 else None,
                    "tokens": sum(int(r.get("generated_tokens") or 0) for r in rows),
                    "complete_attempts": sum(1 for r in rows if r.get("status") == "done" and not str(r.get("note") or "").startswith("nb5:"))}
    return out


def _valid_grid(g):
    return (isinstance(g, list) and g and all(isinstance(r, list) and r and all(isinstance(v, int) and 0 <= v <= 9 for v in r)
                                               for r in g) and len({len(r) for r in g}) == 1)


def quality_table(records, solutions, turns, ttft_mean=None):
    """Addendum 5, one row per task of a quality config: exact grid match on every test input (any host-accepted
    candidate), answer completeness (a candidate with a parsable grid for every test input + every turn finished
    naturally with a closed think block), output tokens, wall time (first attempt start -> last attempt end), first
    request latency (non-streaming: TTFT is a phase mean from the server histogram), tool-call rate."""
    out = {}
    by = {}
    for t in turns or []:
        by.setdefault(t[0], []).append(t)
    for tid, rows in (records or {}).items():
        rows = rows or []
        gold = (solutions or {}).get(tid)
        cands = [v for r in rows for v in (r.get("submissions") or []) + (r.get("checked") or [])]
        exact = any(v.get("accepted") and gold is not None and len(v.get("test_outputs") or []) == len(gold)
                    and all(a == b for a, b in zip(v["test_outputs"], gold)) for v in cands)
        parsable = any(len(v.get("test_outputs") or []) == len((solutions or {}).get(tid) or v.get("test_outputs") or [1])
                       and all(_valid_grid(g) for g in v.get("test_outputs") or [None]) for v in cands)
        tt = by.get(tid, [])
        t0s = [float(r["t0"]) for r in rows if r.get("t0")]
        t1s = [float(r["t1"]) for r in rows if r.get("t1")]
        out[tid] = {"exact_all_tests": bool(exact) if gold is not None else None, "parsable_grid": bool(parsable),
                    "turns": len(tt), "natural_closed_turns": sum(1 for t in tt if len(t) > 9 and t[9]),
                    "complete": bool(parsable) and bool(tt) and all(len(t) > 9 and t[9] for t in tt[-1:]),
                    "output_tokens": sum(int(r.get("generated_tokens") or 0) for r in rows),
                    "wall_sec": round(max(t1s) - min(t0s), 1) if t0s and t1s else None,
                    "first_request_latency_sec": tt[0][8] if tt and len(tt[0]) > 8 else None,
                    "tool_call_rate": round(sum(t[7] for t in tt if len(t) > 7) / len(tt), 3) if tt else None,
                    "attempts": len(rows), "ttft_phase_mean_sec": ttft_mean}
    return out


def engine_rates(steady, pp=2):
    """tick_ms and cycle_ms separately (addendum 4): with the iteration histogram one vLLM step is one pipeline tick and
    a request advances once per pp ticks (PP2: cycle = 2 ticks); the lab's fallback estimate (drafts / running) is
    already a per-request cycle."""
    if not isinstance(steady, dict) or not steady.get("step_ms"):
        return {"tick_ms": None, "cycle_ms": None, "pp": pp}
    if steady.get("steps_source") == "iteration_histogram":
        return {"tick_ms": steady["step_ms"], "cycle_ms": round(pp * steady["step_ms"], 1), "pp": pp, "basis": "iteration_histogram"}
    return {"tick_ms": round(steady["step_ms"] / pp, 1), "cycle_ms": steady["step_ms"], "pp": pp, "basis": "estimate"}


def steady_mixed(rows, workers):
    """Generation tok/s of the server's 10-s stats lines while every slot ran (running == workers): the steady mixed
    rate (prefill of later turns included), separate from the pure decode window and the run averages."""
    at = [r["gen_tps"] for r in rows if r.get("running") == int(workers)]
    busy = [r for r in rows if r.get("running", 0) > 0 or r.get("waiting", 0) > 0]
    srt = sorted(at)
    return {"workers": int(workers), "lines": len(rows), "busy_lines": len(busy), "lines_at_workers": len(at),
            "share_at_workers": round(len(at) / len(busy), 3) if busy else None,
            "gen_tok_s_p50": srt[len(srt) // 2] if srt else None,
            "gen_tok_s_mean": round(sum(srt) / len(srt), 1) if srt else None,
            "prompt_tok_s_mean_at_workers": round(sum(r["prompt_tps"] for r in rows if r.get("running") == int(workers)) / len(at), 1) if at else None}


def rate(tokens, sec):
    return round(float(tokens) / float(sec), 1) if tokens is not None and sec and sec > 0 else None


def counter_delta(c0, c1):
    """Per-phase deltas of the server's cumulative counters (the timestamp-delta metric reset)."""
    out = {}
    for k, v in (c1 or {}).items():
        if isinstance(v, (int, float)) and isinstance((c0 or {}).get(k), (int, float)):
            out[k] = v - c0[k]
    q, h = out.get("vllm:prefix_cache_queries_total"), out.get("vllm:prefix_cache_hits_total")
    out["prefix_hit_pct"] = round(100.0 * h / q, 2) if q else None
    a, d = out.get("vllm:spec_decode_num_accepted_tokens_total"), out.get("vllm:spec_decode_num_drafts_total")
    out["mean_acceptance_length"] = round(1 + a / d, 3) if a is not None and d else None
    return out


THINK_FLOOR = 60000                 # user rule: thinking >= 60k (never below without approval)


def context_guard(prompt_len, requested, generated, budget, reserve, in_commit, max_len=131072, floor=THINK_FLOOR, margin=2048):
    """Every request (revision 3): prompt_tokens + max_new <= max_len with the ACTUAL prompt length (the server's own
    render); the output cap is the smaller of Duck's request cap and the room left. Thinking floor: outside the commit
    zone, an attempt whose generated tokens + this request's cap cannot reach the floor any more (long tool histories
    ate the window) ends its thinking path the way Duck ends an exhausted budget: the attempt budget shrinks to
    generated + cap + reserve, so this is its last thinking request and the next turn is Duck's own commit turn.
    Nothing is silent: every clamp and floor event is returned for counting.
    PLAN v3.6: safe_output_budget = max(0, min(attempt cap, max_len - actual prompt - 2048)); a floor event marks the
    attempt BUDGET_LIMITED (its verdict in the record and the report).
    A-13 (audits 1 / 3): cap = safe (no 1-token floor). safe == 0 -> no_room: the caller sends NO request and finalizes
    the attempt as CONTEXT_EXHAUSTED (commit turn included). C-9: max_len is the attempt's own projected peak
    min(131072, P_b), so its block reservation is a true upper bound."""
    room = max(0, int(max_len) - int(prompt_len) - int(margin))
    safe = max(0, min(int(requested), room))
    cap = safe
    out = {"prompt": int(prompt_len), "requested": int(requested), "room": room, "safe_output_budget": safe, "cap": cap,
           "max_len": int(max_len), "clamped": cap < int(requested), "floor_event": False, "budget_limited": False,
           "new_budget": None, "no_room": safe <= 0, "verdict": None}
    if safe <= 0:
        out.update(budget_limited=True, verdict="CONTEXT_EXHAUSTED")
        return out
    if not in_commit and int(generated) + safe < int(floor):
        out.update(floor_event=True, budget_limited=True, new_budget=min(int(budget), int(generated) + cap + int(reserve)))
    return out


def future_admission_blocks(attempt, telemetry):
    """PLACEHOLDER (revision 2): a later 'future-aware' admission mode fed by NB1's memory telemetry. Off: returns None,
    so the projected mode decides; no behaviour change now."""
    return None


_TS_STAT_RE = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d)[^\n]*?Avg prompt throughput: ([\d.]+) tokens/s, "
                         r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs")


def stat_timeline(text):
    """(t_sec, prompt_tok_s, gen_tok_s, running) of the server's periodic stats lines (MM-DD HH:MM:SS clock)."""
    import datetime as _dt
    out = []
    for m in _TS_STAT_RE.finditer(text or ""):
        t = _dt.datetime.strptime("2000-" + m.group(1), "%Y-%m-%d %H:%M:%S")
        out.append(((t - _dt.datetime(2000, 1, 1)).total_seconds(), float(m.group(2)), float(m.group(3)), int(m.group(4))))
    return out


def duty_factor(log_text, wall_sec, gen_tokens=None, decode_tok_s=None):
    """Productive duty factor (revision 1): the server's decode time / wall time. Each stats line covers the interval since
    the previous one (capped at 3 x the median interval: idle gaps print no line); productive = generated tokens > 0 in
    it, pure decode = also no prompt tokens. Token-weighted = generated tokens / (the steady pure-decode rate x wall)."""
    rows = stat_timeline(log_text)
    if not rows or not wall_sec:
        return {"lines": len(rows), "duty_time": None, "duty_tokens": None, "wall_sec": wall_sec}
    gaps = sorted(b[0] - a[0] for a, b in zip(rows, rows[1:]) if b[0] > a[0])
    med = gaps[len(gaps) // 2] if gaps else 10.0
    prod = pure = 0.0
    prev = None
    for r in rows:
        dt = med if prev is None else min(max(0.0, r[0] - prev), 3 * med)
        prev = r[0]
        if r[2] > 0:
            prod += dt
            if r[1] == 0:
                pure += dt
    out = {"lines": len(rows), "interval_sec_median": med, "productive_decode_sec": round(prod, 1),
           "pure_decode_sec": round(pure, 1), "wall_sec": round(float(wall_sec), 1),
           "duty_time": round(min(1.0, prod / float(wall_sec)), 4), "duty_pure_decode_time": round(min(1.0, pure / float(wall_sec)), 4),
           "gen_tokens": gen_tokens, "steady_decode_tok_s": decode_tok_s,
           "duty_tokens": (round(float(gen_tokens) / (float(decode_tok_s) * float(wall_sec)), 4)
                           if gen_tokens and decode_tok_s and wall_sec else None)}
    return out


_JIT_RE = re.compile(r"JIT compilation during inference: (\w+)")


def jit_kernels(log_text):
    """Merlin's jit_monitor warnings ('Triton kernel JIT compilation during inference: <kernel>'; RC2 FP8 log lines 441,
    551-555): the kernels compiled while serving. A measurement window must start after the last one."""
    return _JIT_RE.findall(log_text or "")


_HIST = ("vllm:time_to_first_token_seconds", "vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds",
         "vllm:e2e_request_latency_seconds", "vllm:request_decode_time_seconds")


def hist_sums(rows):
    """_sum / _count of vLLM's latency histograms (summed over label rows) from parsed /metrics rows (.name / .value)."""
    out = {}
    for r in rows or []:
        n = getattr(r, "name", "")
        for h in _HIST:
            if n in (h + "_sum", h + "_count"):
                out[n] = out.get(n, 0.0) + float(r.value)
    return out


def hist_delta(h0, h1):
    """Per-phase means (seconds) of TTFT / queue / prefill / e2e / decode from two hist_sums snapshots (addendum 3:
    Merlin's cost-aware admission deferred a cold request 143.7 s TTFT in RC2)."""
    out = {}
    for h in _HIST:
        s_ = (h1 or {}).get(h + "_sum", 0.0) - (h0 or {}).get(h + "_sum", 0.0)
        c_ = (h1 or {}).get(h + "_count", 0.0) - (h0 or {}).get(h + "_count", 0.0)
        out[h.split(":", 1)[1]] = {"count": int(c_), "mean_sec": round(s_ / c_, 2) if c_ > 0 else None}
    return out


_DEF_RE = re.compile(r"Deferred: (\d+) reqs")


def deferred_lines(log_text):
    """Merlin's cost-aware admission defers ('Deferred: N reqs' in the stats lines): lines and the max N."""
    v = [int(x) for x in _DEF_RE.findall(log_text or "")]
    return {"lines": len(v), "max": max(v) if v else 0}


def argv_diff(ref_tail, tail):
    """Flags whose LAST value differs from the RC2 FP8 launch (extra args come last and win), plus added / removed."""
    def last(argv):
        d, i = {}, 0
        while i < len(argv):
            a = str(argv[i])
            if a.startswith("--"):
                v = argv[i + 1] if i + 1 < len(argv) and not str(argv[i + 1]).startswith("--") else True
                d[a] = v
                i += 2 if v is not True else 1
            else:
                i += 1
        return d
    a, b = last(list(ref_tail or [])), last(list(tail or []))
    return {"changed": {k: [a[k], b[k]] for k in a if k in b and a[k] != b[k] and k != "--port"},
            "added": {k: b[k] for k in b if k not in a}, "removed": sorted(k for k in a if k not in b)}


# ------------------------------------------------------------------ C-11: config read-back from the server log
KV_DTYPE_NAME = {"fp8": "fp8_e4m3", "bf16": "bfloat16"}


def argv_last(argv):
    """{flag: LAST value} of an argv (argparse: the last occurrence wins); a flag without a value -> True."""
    d, i, argv = {}, 0, list(argv or [])
    while i < len(argv):
        a = str(argv[i])
        if a.startswith("--"):
            v = argv[i + 1] if i + 1 < len(argv) and not str(argv[i + 1]).startswith("--") else True
            d[a] = v
            i += 2 if v is not True else 1
        else:
            i += 1
    return d


def log_config(log_text):
    """What the server says it runs with (NB2 log_facts pattern): from the LAST 'non-default args' line kv_cache_dtype,
    max_num_seqs, kv_cache_memory_bytes, speculative_config (dict / None / absent); the engine's SpeculativeConfig(...)
    line (method, num_spec_tokens) or 'speculative_config=None'."""
    import ast
    t = log_text or ""
    nd = [ln for ln in t.splitlines() if "non-default args:" in ln]
    last = nd[-1] if nd else ""
    f = {"non_default_found": bool(nd)}
    m = re.search(r"'kv_cache_dtype': '([^']*)'", last)
    f["kv_cache_dtype"] = m.group(1) if m else None
    f["max_num_seqs"] = _int_after(last, "max_num_seqs")
    f["kv_cache_memory_bytes"] = _int_after(last, "kv_cache_memory_bytes")
    m = re.search(r"'speculative_config': (\{[^{}]*\}|None)", last)
    f["speculative_config_found"] = bool(m)
    try:
        f["speculative_config"] = (None if m.group(1) == "None" else ast.literal_eval(m.group(1))) if m else None
    except Exception:
        f["speculative_config"] = "UNPARSED " + m.group(1)[:200]
    es = re.findall(r"SpeculativeConfig\(method='(\w+)', model='[^']*', num_spec_tokens=(\d+)\)", t)
    f["engine_spec"] = {"method": es[-1][0], "num_spec_tokens": int(es[-1][1])} if es else None
    f["engine_spec_none"] = "speculative_config=None" in t
    return f


def config_readback(log_text, argv, kv, mtp_on, k=3):
    """C-11 (audit 3): the server log must show the intended KV dtype, max_num_seqs (= the argv's last value), the KV
    pin bytes (when the argv sets them) and the speculative config (MTP K on / off; every key of the argv's JSON).
    ok False -> the boot's phases are INCONCLUSIVE_CONFIG_MISMATCH (paired results must provably differ only in KV)."""
    f, a = log_config(log_text), argv_last(argv)
    want_kv = KV_DTYPE_NAME[kv]
    chk = {"non_default_args_line": f["non_default_found"],
           "kv_cache_dtype": f["kv_cache_dtype"] == want_kv and a.get("--kv-cache-dtype", want_kv) == want_kv,
           "max_num_seqs": a.get("--max-num-seqs") not in (None, True) and f["max_num_seqs"] == int(a["--max-num-seqs"])}
    if a.get("--kv-cache-memory-bytes") not in (None, True):
        chk["kv_cache_memory_bytes"] = f["kv_cache_memory_bytes"] == int(a["--kv-cache-memory-bytes"])
    sc, es = f["speculative_config"], f["engine_spec"]
    sa = a.get("--speculative-config")
    if mtp_on:
        want = {"method": "mtp", "num_speculative_tokens": int(k)}
        if sa not in (None, True, "None"):
            try:
                want.update(json.loads(sa))
            except Exception:
                chk["argv_speculative_json"] = False
        chk["speculative_config"] = (isinstance(sc, dict) and sc.get("num_speculative_tokens") == int(k)
                                     and all(sc.get(x) == y for x, y in want.items()))
        chk["engine_spec"] = es is None or (es["method"] == "mtp" and es["num_spec_tokens"] == int(k))
    else:
        chk["speculative_config"] = sc is None and sa in ("None", None)
        chk["engine_spec"] = es is None
    ok = all(chk.values())
    return {"ok": ok, "verdict": "PASS" if ok else "INCONCLUSIVE_CONFIG_MISMATCH", "checks": chk, "log": f,
            "intended": {"kv_cache_dtype": want_kv, "max_num_seqs": a.get("--max-num-seqs"), "mtp_k": int(k) if mtp_on else 0,
                         "kv_cache_memory_bytes": a.get("--kv-cache-memory-bytes"), "speculative_config_argv": sa}}


# ------------------------------------------------------------------ A-14: the local render (only with verified hashes)
def vllm_messages(msgs):
    """vLLM 0.32 chat preprocessing as the template sees it: 'reasoning' -> reasoning_content, tool-call arguments JSON ->
    dict, None content -> '' (the same rules as the NB5 fake server and the unit-test render)."""
    out = []
    for m in msgs or []:
        m2 = {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
        if m2.get("content") is None:
            m2["content"] = ""
        if m.get("role") == "assistant":
            m2["reasoning_content"] = m.get("reasoning") or m.get("reasoning_content") or ""
            tcs = []
            for tc in m.get("tool_calls") or []:
                fn = dict(tc.get("function") or {})
                x = fn.get("arguments")
                if isinstance(x, str):
                    try:
                        fn["arguments"] = json.loads(x) if x.strip() else {}
                    except Exception:
                        fn["arguments"] = {}
                tcs.append(dict(tc, function=fn))
            if tcs:
                m2["tool_calls"] = tcs
        out.append(m2)
    return out


def local_chat_ids(payload, hf, template, encode, default_kwargs=None):
    """The chat payload rendered with the LOCAL Flash-Next tokenizer + chat template (A-14 fallback after /tokenize failed
    3 times; the caller uses it only when both file hashes were verified at load)."""
    kw = dict(default_kwargs or {})
    kw.update(payload.get("chat_template_kwargs") or {})
    text = hf.apply_chat_template(vllm_messages(payload.get("messages")), tools=payload.get("tools") or None, tokenize=False,
                                  add_generation_prompt=True, chat_template=template, **kw)
    return list(encode(text))


# ------------------------------------------------------------------ R2 concurrency rule (lead 2026-10-31, simplified)
WORKER_CTX_TOKENS = 65536


def workers_rule(pool_blocks, block, f_eff, ctx_tokens=WORKER_CTX_TOKENS):
    """R2 (user decision): workers = max_num_seqs = max(1, floor(pool / (ceil(65536 / B) + F))) with THIS boot's pool and
    F (f_log from the max-concurrency line); the engine schedules, preemptions are allowed (counted: speed only)."""
    return max(1, int(pool_blocks) // (cdiv(int(ctx_tokens), int(block)) + int(f_eff)))


# the launch --max-num-seqs per pin step (no relaunch in 3 h): the rule on the pool expected at each step (RC2 per-card
# block 15.75 MiB FP8 / 29.75 MiB BF16; F 6): FP8 149 -> 6, ~181 / ~197 -> 8, ~214 -> 9; BF16 78 -> 3, ~95 / ~103 -> 4, ~112 -> 5
SEQS_TABLE = {"fp8": {1.0: 9, 0.75: 8, 0.5: 8, 0.0: 6}, "bf16": {1.0: 5, 0.75: 4, 0.5: 4, 0.0: 3}}


def seqs_for(kv, pin_gib):
    return int(SEQS_TABLE[kv][round(float(pin_gib), 2)])


def captures_cover(caps, seqs, k=3):
    """FULL uniform-decode graphs must cover seqs x (1 + K) tokens (NB2 CAPS16 rule)."""
    return max(caps or [0]) >= int(seqs) * (int(k) + 1)


# ------------------------------------------------------------------ lever a: KV pin ladder (NB1's descending rule)
PIN_STEPS_GIB = (1.0, 0.75, 0.5, 0.0)        # + GiB per card over the RC2 pin; 0.0 = the RC2 pin itself
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory")


def pin_tries(base_bytes, start_gib=None, steps=PIN_STEPS_GIB):
    """The descending ladder from start_gib (default: the top step): [{gib, kv_bytes}] (NB1 pin_ladder order)."""
    s = [g for g in steps if start_gib is None or g <= float(start_gib) + 1e-9]
    return [{"gib": g, "kv_bytes": int(base_bytes) + int(round(g * 2 ** 30))} for g in s]


def boot_failure(status, log_text, lever_env=None, phases_started=False):
    """Whether a boot try failed in a way the ladder steps down on: a launch failure (not ready / died before the
    phases) or an OOM line before any phase started. lever_suspect = the lever env keys named in an error line (that
    lever is dropped for the next try)."""
    t = log_text or ""
    oom = [ln[:240] for ln in t.splitlines() if OOM_RE.search(ln)][:6]
    bad = str(status or "") in ("LAUNCH_FAILED", "SERVER_DIED") or bool(oom)
    err = [ln for ln in t.splitlines() if re.search(r"Error|Traceback|error:", ln)]
    sus = sorted(k for k in (lever_env or {}) if any(k in ln for ln in err))
    return {"failed": bool(bad and not phases_started), "status": status, "oom_lines": oom, "lever_suspect": sus}


def equal_windows(now_pc, solve_end_pc, boot_overhead_sec, tail_sec=300.0, phases=2):
    """Equal solve windows for the paired phases, fixed when the FIRST phase starts: what is left to the solve end minus
    the later boots (each estimated as max(1.25 x this boot's measured overhead, + 5 min)) and the phase tails, split
    equally. A slow first boot shortens every window equally (it is measured before the split)."""
    boot2 = max(1.25 * float(boot_overhead_sec), float(boot_overhead_sec) + 300.0)
    avail = float(solve_end_pc) - float(now_pc) - (phases - 1) * (boot2 + float(tail_sec))
    w = max(0.0, avail / phases)
    return {"window_sec": round(w, 1), "boot_overhead_sec": round(float(boot_overhead_sec), 1),
            "later_boot_est_sec": round(boot2, 1), "tail_sec": float(tail_sec), "avail_sec": round(avail, 1), "phases": phases}


# ------------------------------------------------------------------ the paired FP8 / BF16 quality report
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


QAB_PHASES = ("Q_FP8_K3", "Q_BF16_K3")
QAB_WINDOW_TOL_SEC = 2.0          # Q-2: predeclared tolerance on the actual solve windows (window starts are rounded to 1 s)
QAB_DIFF_FIELDS = (("capacity_levers", "pin_gib"), ("capacity_levers", "kv_bytes"), ("capacity_levers", "pool_blocks"),
                   ("capacity_levers", "pool_source"), ("capacity_levers", "capacity_blocks"), ("capacity_levers", "f_eff"),
                   ("capacity_levers", "f_log"), ("capacity_levers", "f_source"), ("capacity_levers", "workers"),
                   ("capacity_levers", "max_num_seqs_launch"), ("capacity_levers", "lever_env"),
                   ("capacity_levers", "levers_dropped"), ("capacity_levers", "captures"), ("capacity_levers", "config_readback"),
                   ("capacity_levers", "ladder_tries"), ("speed", "preemptions"), ("speed", "speed_status"),
                   ("speed", "max_running"), ("speed", "mean_running_busy"), ("window", "actual_sec"),
                   ("window_plan", "planned_sec"), ("window_plan", "unequal"), ("status",), ("order",))


def qab_windows_check(per_config, phases=QAB_PHASES, tol_sec=QAB_WINDOW_TOL_SEC):
    """Q-2: EQUAL only when both phases are present and DONE, both actual windows are numbers, |delta| <= tol_sec and
    neither window_plan is marked unequal; anything else -> INCONCLUSIVE_UNEQUAL_WINDOWS (reasons listed)."""
    why, act = [], {}
    for n in phases:
        p = (per_config or {}).get(n)
        if not p:
            why.append(f"{n}: phase missing")
            continue
        if p.get("status") != "DONE":
            why.append(f"{n}: status {p.get('status')}")
        a = (p.get("window") or {}).get("actual_sec")
        if isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a):
            act[n] = float(a)
        else:
            why.append(f"{n}: actual_sec not numeric ({a!r})")
        if (p.get("window_plan") or {}).get("unequal"):
            why.append(f"{n}: window_plan.unequal")
    d = round(abs(act[phases[0]] - act[phases[1]]), 1) if len(act) == 2 else None
    if d is not None and d > tol_sec:
        why.append(f"|delta| {d} s > {tol_sec} s")
    return {"equal": not why, "verdict": "EQUAL" if not why else "INCONCLUSIVE_UNEQUAL_WINDOWS", "tolerance_sec": tol_sec,
            "actual_sec": act, "delta_sec": d, "reasons": why}


def qab_outcome_matrix(per_config, phases=QAB_PHASES):
    """Q-5: the missing / incomplete / failed matrix: per phase the outcome and end-reason counts, the tasks missing from
    the phase, the tasks not finished (with their attempt outcomes); per task the attempt outcomes in each phase."""
    tids = []
    for n in phases:
        tids += [t for t in ((per_config or {}).get(n) or {}).get("tasks") or {} if t not in tids]
    out = {"phases": {}, "per_task": {t: {} for t in tids}}
    for n in phases:
        p = (per_config or {}).get(n)
        tk = (p or {}).get("tasks") or {}
        ph = {"present": bool(p), "status": (p or {}).get("status"), "tasks": len(tk),
              "tasks_missing": [t for t in tids if t not in tk], "outcomes": {}, "end_reasons": {},
              "tasks_finished": sum(1 for v in tk.values() if v.get("finished")),
              "tasks_not_finished": {t: v.get("unfinished_outcomes") for t, v in tk.items() if not v.get("finished")}}
        for t in tids:
            at = (tk.get(t) or {}).get("attempts")
            out["per_task"][t][n] = [a.get("outcome") for a in at] if at else "MISSING"
            for a in at or []:
                ph["outcomes"][str(a.get("outcome"))] = ph["outcomes"].get(str(a.get("outcome")), 0) + 1
                ph["end_reasons"][str(a.get("end_reason"))] = ph["end_reasons"].get(str(a.get("end_reason")), 0) + 1
        out["phases"][n] = ph
    return out


def qab_phase_diff(per_config, phases=QAB_PHASES, labels=("FP8", "BF16")):
    """Q-3: every effective non-dtype difference between the two phases (the KV dtype itself is the treatment)."""
    a, b = ((per_config or {}).get(n) or {} for n in phases)

    def get(p, path):
        for k in path:
            p = p.get(k) if isinstance(p, dict) else None
        return p
    diff, same = {}, []
    for path in QAB_DIFF_FIELDS:
        x, y = get(a, path), get(b, path)
        if x == y:
            same.append(".".join(path))
        else:
            diff[".".join(path)] = {labels[0]: x, labels[1]: y}
    return {"phases": list(phases), "labels": list(labels), "excluded": ["kv dtype (the treatment)"],
            "missing_phase": [n for n in phases if not (per_config or {}).get(n)], "differences": diff,
            "n_differences": len(diff), "same": same}


def qab_report(per_config, phases=QAB_PHASES, labels=("FP8", "BF16"), tol_sec=QAB_WINDOW_TOL_SEC):
    """Q-2 / Q-3 / Q-5: the R2 report blocks. deployment_comparison = the same wall-clock window, each dtype in its own
    production config (three separate results: all tasks in the fixed window, finished in both, the missing / incomplete /
    failed matrix); precision_comparison = UNVERIFIED (the phases differ in more than the dtype: see phase_diff)."""
    pc = per_config or {}
    wc = qab_windows_check(pc, phases, tol_sec)
    fa, fb = (pc.get(n) or {} for n in phases)
    paired = (qab_pair(fa["tasks"], fb["tasks"], labels) if (fa.get("tasks") and fb.get("tasks")) else
              {"skipped": "both phases are needed (offline: python nb5_lib.py qab-pair nb5_report.json)"})
    mx, dif = qab_outcome_matrix(pc, phases), qab_phase_diff(pc, phases, labels)
    st = ("INCONCLUSIVE_UNEQUAL_WINDOWS" if not wc["equal"] else ("INCOMPLETE_PHASES" if "skipped" in paired else "MEASURED"))
    dep = {"label": "deployment_comparison", "definition": "same wall-clock solve window, each KV dtype in its production "
           "config (its own pin step, pool, F, workers); a fixed-wall-time deployment A/B, not a pure precision A/B",
           "status": st, "windows": wc, "all_tasks_fixed_window": paired.get("all"),
           "finished_in_both": paired.get("finished_in_both"), "finished_in_both_tasks": paired.get("finished_in_both_tasks"),
           "missing_incomplete_failed": mx, "parity_rule": "never claim parity from a non-significant result"}
    w = (dif["differences"].get("capacity_levers.workers") or {})
    prec = {"label": "precision_comparison", "status": "UNVERIFIED",
            "reason": (f"the phases differ in more than the KV dtype: workers {w.get(labels[0])} vs {w.get(labels[1])}, "
                       "the KV pool and possibly the pin step (phase_diff lists every effective difference); a pure "
                       "precision answer needs both phases at the same workers"),
            "differing_fields": sorted(dif["differences"])}
    return {"windows_check": wc, "windows_equal": wc["equal"], "paired": paired, "deployment_comparison": dep,
            "precision_comparison": prec, "phase_diff": dif}


def qab_table_text(tasks, label):
    """A fixed-width per-task table for the notebook log."""
    lines = [f"{label}: task      group      q  hits solved  attempts (outcome/tokens/exact)"]
    for t, v in (tasks or {}).items():
        at = "  ".join(f"a{a['attempt']}:{str(a['outcome'])[:10]}/{a.get('tokens', 0)}/{'Y' if a.get('exact_accepted') else 'n'}"
                       for a in v.get("attempts") or [])
        lines.append(f"{label}: {t}  {str(v.get('group'))[:9]:9}  {v.get('queries')}  {v.get('query_hits'):4}  "
                     f"{'YES' if v.get('solved') else 'no ':3}     {at}")
    return "\n".join(lines)


def qab_pair_reports(reports):
    """Offline: the paired summary from one NB5QUALITY_R2 report (both phases inside) or two reports."""
    tasks = {}
    for rep in reports:
        for n, p in ((rep.get("quality_ab") or {}).get("per_config") or {}).items():
            tasks[n] = p.get("tasks") or {}
    names = sorted(tasks, key=lambda n: (0 if "FP8" in n else 1, n))
    if len(names) < 2:
        return {"error": f"need two configs, found {names}"}
    return qab_pair(tasks[names[0]], tasks[names[1]], labels=("FP8" if "FP8" in names[0] else names[0],
                                                               "BF16" if "BF16" in names[1] else names[1]))


if __name__ == "__main__":       # offline: python nb5_lib.py qab-pair <nb5_report.json> [<second report>] [--out pair.json]
    import sys as _sys
    _a = _sys.argv[1:]
    if _a and _a[0] == "qab-pair":
        _out = _a[_a.index("--out") + 1] if "--out" in _a else None
        _files = [x for x in _a[1:] if x not in ("--out", _out)]
        _res = qab_pair_reports([json.loads(Path(x).read_text(encoding="utf-8")) for x in _files])
        print(json.dumps(_res, indent=1))
        if _out:
            Path(_out).write_text(json.dumps(_res, indent=1), encoding="utf-8")
