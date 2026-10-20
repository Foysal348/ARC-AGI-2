"""Experiment 11 Stage 3: E11-04 capacity + real-concurrency validation (Claude, 2026-10-17, user: "E11-04 এর অগ্রগতি").
Deterministic builder. It reuses the E11-03 cells that ran clean on Kaggle (install, preflight, tokenization, server
helpers) verbatim from build_e11_03_gpu_validation.py and patches only what E11-03 run 1 + the two GPT audits showed:

  * E11-03's speed curve was a CAP: SGLang caps max_running_requests by the mamba state cache (5 slots/request with
    extra_buffer + overlap; kv_cache_configurator.py:2232-2330). C ran 6, B 10, D 13 whatever we asked.
  * Auto mem_fraction_static 0.62 left 4.4 GB/GPU free; the 2.6 GB prefill CUDA graph; KV only 387k (need >= 900k).
  * Real concurrency must be asserted from the decode log, startup caps read from the server log (server_info lies).

Arms (all INT4 + DFlash2 block 8, unquant drafter, TP4, max running 12, prefill CUDA graph OFF, mem 0.83 -> 0.78 ladder):
  W1  extra_buffer_lazy (4 slots/req), mamba cache 56, SSM float16 (E11-03 D, the only arm that really ran 12)
  W2  no_buffer (3 slots/req) + ReplaySSM spec-verify (no per-draft SSM snapshots), SSM float32 (accuracy reference)
  W3  W1 with SSM bfloat16 (the dtype SGLang's own warning suggests)
Stages: startup caps + gates, greedy, sweep 1/4/8/12 with real-concurrency check, PROD (closed loop: 12 always in
flight, real ARC prompts, up to 60,000 thinking tokens, T 1.0, refill on finish, tok/s from the generation counter),
marker contamination test, prefix-cache continuation test, 8-puzzle sanity solve.
usage: python build_e11_04_capacity_validation.py
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent
NAME = "E11_04_sglang_capacity_validation_on_KAGGLE.ipynb"
SRC = HERE / "build_e11_03_gpu_validation.py"
_ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("\ncells = [md(M0)")[0], _ns)       # E11-03 cell texts; writes nothing
md, code = _ns["md"], _ns["code"]


def once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"{what}: expected 1 match, found {n}"
    return text.replace(old, new)


M0 = """
# E11-04 · SGLang 0.5.21 + RedHat Qwen3.8-27B INT4 + DFlash2 · capacity + real-concurrency validation on 4x L4

**Settings:** Accelerator **GPU L4 x4** · Internet **OFF** · no Secrets.
**Attach (same as E11-03):** dataset `foysalemonshanto/sglang-0-5-21-cu13-wheelhouse`, models
`foysalemonshanto/qwen3-8-27b-int4-redhat` (w4a16-g128) and `foysalemonshanto/qwen3-8-27b-dflash2` (bf16),
competition `arc-prize-2026-arc-agi-2`. **Not a submission.**

E11-03 run 1 found that the flat speed curve was a **cap**, not the engine: SGLang limited the running requests through
the mamba (SSM) state cache (C ran 6, B 10, D 13), left 4.4 GB/GPU unused and gave only 387k KV tokens
(production needs about 920k for 12 x 60k). This run answers, at a **real** 12 concurrent requests:
1. Do the memory flags give **12 real running + >= 900k KV**? (read from the server log, asserted)
2. How fast is INT4 + DFlash2 on a **production-like** load (12 always in flight, ARC prompts, up to 60k thinking)?
   Target: >= 332 tok/s (1.3 x vLLM FP8's long-context 255).
3. Which SSM state handling works: **W1** fp16 + lazy buffer · **W2** fp32 + ReplaySSM · **W3** bf16 + lazy buffer.
4. Correctness: marker contamination test, prefix-cache continuation, 8-puzzle sanity solve, FP8 KV scale warning.

Expected about 2.5 h; no new stage starts after 4.5 h. Everything is written to `/kaggle/working/e11_04/`.
"""

# ---------------------------------------------------------------- cell 0
C0 = _ns["C0"]
C0 = once(C0, "CELL 0 — configuration, inputs, results file", "CELL 0 — configuration, inputs, results file (E11-04)", "c0 title")
C0 = once(C0, '"/kaggle/working/e11_03"', '"/kaggle/working/e11_04"', "out dir")
C0 = once(C0, 'os.environ.get("E11_TIME_BUDGET", 5.0 * 3600)', 'os.environ.get("E11_TIME_BUDGET", 4.5 * 3600)', "time budget")
C0 = once(C0, "MAX_RUNNING = 32\n", "MAX_RUNNING = 12                # E11-04: production slots; SGLang must REALLY run 12\n", "max running")
C0 = once(C0, "SWEEP_LEVELS = [1, 4, 8, 12, 16, 24, 32]   # rev 3: 12 added (D66/L66: 12 slots suit long-thinking Poetiq)",
          "SWEEP_LEVELS = [1, 4, 8, 12]   # E11-04: up to the production 12 (E11-03 measured beyond)", "sweep levels")
C0 = once(C0, "SANITY_TIMEOUT_SEC = 60 * 60\n",
          "SANITY_TIMEOUT_SEC = 60 * 60\n"
          "# ---- E11-04\n"
          "KV_MIN = 900_000                # gate: >= 12 x 75k (S1-60K peaked at 68 % of 1.35M = ~920k on vLLM)\n"
          "PROD_N_UNITS = 48               # distinct ARC prompts cycled through the closed-loop load\n"
          "PROD_CONCURRENCY, PROD_MAX_NEW = 12, 60000      # = S1-60K: 12 slots, 60,000 thinking\n"
          "PROD_SEC = {\"W1\": 35 * 60, \"W2\": 25 * 60, \"W3\": 15 * 60}   # closed-loop window per arm\n"
          "PROD_SEC_GATE_FAIL = 10 * 60    # a failed capacity gate still measures, but shorter\n"
          "if os.environ.get(\"E11_PROD_SEC\"):          # harness only\n"
          "    PROD_SEC = {k: float(os.environ[\"E11_PROD_SEC\"]) for k in PROD_SEC}\n"
          "    PROD_SEC_GATE_FAIL = float(os.environ[\"E11_PROD_SEC\"]) / 2\n"
          "TARGET_TOK_S = 332              # 1.3 x vLLM FP8 + MTP-3 long-context ~255 tok/s\n"
          "MARK_N = 8                      # contamination companions, each told to write its own codename\n", "e11-04 constants")
C0 = once(C0, 'RESULTS = {"run_id": RUN_ID, "inputs": {k: str(v) for k, v in INPUTS.items()}, "env": {}, "preflight": None,\n'
              '           "configs": {}, "best_spec": None, "sanity": None, "notes": []}',
          'RESULTS = {"run_id": RUN_ID, "experiment": "E11-04", "inputs": {k: str(v) for k, v in INPUTS.items()}, "env": {},\n'
          '           "preflight": None, "configs": {}, "best_spec": None, "sanity": None, "notes": []}\n'
          '# FP8 KV scales: does the INT4 checkpoint carry per-layer k/v scales? (E11-03 log: "no scaling factors provided")\n'
          '_idx = INT4 / "model.safetensors.index.json"\n'
          'try:\n'
          '    _wm = _json(_idx).get("weight_map", {}) if _idx.is_file() else {}\n'
          '    RESULTS["kv_scale_tensors_in_checkpoint"] = sum(1 for k in _wm if k.endswith(("k_scale", "v_scale")))\n'
          'except Exception as _e:\n'
          '    RESULTS["kv_scale_tensors_in_checkpoint"] = f"error {_e!r}"[:200]\n'
          'RESULTS["kv_cache_scheme"] = ((_json(INT4 / "config.json").get("quantization_config") or {}).get("kv_cache_scheme"))',
          "results + kv scale probe")

# ---------------------------------------------------------------- cell 1 + 2 verbatim
C1, C2 = _ns["C1"], _ns["C2"]

# ---------------------------------------------------------------- cell 3: more prompt units + marked companions
C3 = _ns["C3"]
C3 = once(C3, "SWEEP_UNITS = [f\"{t}#0\" for t in random.Random(SEED).sample(_pool, max(SWEEP_LEVELS))]",
          "SWEEP_UNITS = [f\"{t}#0\" for t in random.Random(SEED).sample(_pool, max(max(SWEEP_LEVELS), PROD_N_UNITS))]\n"
          "MARK_CODES = {u: f\"ZQX-{701 + i}\" for i, u in enumerate(SWEEP_UNITS[GREEDY_N:GREEDY_N + MARK_N])}   # E11-04",
          "sweep units")
C3 = once(C3, '    units[uid] = tok.encode(text, add_special_tokens=False)\n',
          '    units[uid] = tok.encode(text, add_special_tokens=False)\n'
          '    if uid in cfg.get("marks", {}):          # E11-04 contamination companion: same task, own codename\n'
          '        code = cfg["marks"][uid]\n'
          '        mm = [{"role": "user", "content": "Begin your reasoning by writing the codename " + code + " exactly once. "\n'
          '               "Never write any other codename.\\n\\n" + build_user_message(task["train"], task["test"][int(qi)]["input"])}]\n'
          '        mt = tok.apply_chat_template(mm, tokenize=False, add_generation_prompt=True, enable_thinking=True,\n'
          '                                     reasoning_effort="xhigh")\n'
          '        units["M:" + uid] = tok.encode(mt, add_special_tokens=False)\n', "marked units")
C3 = once(C3, '"units": SWEEP_UNITS + SANITY_UNITS, "stop_text": STOP_THINKING_TEXT, "out": str(_to)}),',
          '"units": SWEEP_UNITS + SANITY_UNITS, "stop_text": STOP_THINKING_TEXT, "out": str(_to),\n'
          '                           "marks": MARK_CODES}),', "tokenize cfg marks")

# ---------------------------------------------------------------- cell 4: helpers + caps + real concurrency
C4 = _ns["C4"]
C4 = once(C4, 'LADDER = [[], ["--attention-backend", "triton"]]       # the 2nd launch only if the 1st fails',
          '# E11-04: memory ladder. The 2nd launch (only if the 1st fails, e.g. OOM) leaves 0.05 more for activations.\n'
          'LADDER = [["--mem-fraction-static", "0.83"], ["--mem-fraction-static", "0.78"]]', "ladder")
C4 = once(C4, '"max_mamba_cache_size", "internal_states"]',
          '"max_mamba_cache_size", "internal_states", "disable_prefill_cuda_graph", "enable_linear_replayssm_spec"]',
          "server keys")
C4 = once(C4, '''    st["decode_log"] = summarize_decode(rows, level=c)
    if tag == "long":''', '''    st["decode_log"] = summarize_decode(rows, level=c)
    st["real_concurrency"] = real_concurrency(st, c)          # E11-04: fail-closed (E11-03 C ran 6 when asked 12)
    if tag == "long":''', "load_level real concurrency")
C4 = once(C4, '''          f"accept {st['accept_len_mean']} | wall {st['wall_s']}s | ok {st['ok']}/{c}")
    return st''', '''          f"accept {st['accept_len_mean']} | wall {st['wall_s']}s | ok {st['ok']}/{c} | real running max "
          f"{st['real_concurrency']['running_max']} ({'PASS' if st['real_concurrency']['pass'] else 'FAIL'})")
    return st''', "load_level stamp")
C4 = once(C4, "def run_batch(ids_list, max_new, sampling, seeds, stop_ids=None):", r'''def real_concurrency(st, level, min_samples=2):
    """PASS only if the decode log shows the requested number of requests running together (E11-03's lesson)."""
    dl = st.get("decode_log") or {}
    rm, n_at = int(dl.get("running_max") or 0), int(dl.get("n_lines_at_level") or 0)
    return {"requested": level, "running_max": rm, "samples_at_level": n_at,
            "pass": rm >= level and n_at >= (1 if level == 1 else min_samples) and st.get("ok") == level}


def _all_ints(s):
    return [int(x) for x in re.findall(r"\d+", s)]


def server_caps(text):
    """Effective limits from the server log (not /get_server_info: it still reports the requested 32 / 12)."""
    caps = {}
    m = re.findall(r"max_running_requests is capped to (\d+) by the mamba state cache \(max_mamba_cache_size=(\d+), (\d+) state", text)
    caps["capped_to"] = min(int(a) for a, _, _ in m) if m else None
    caps["state_slots_per_request"] = int(m[0][2]) if m else None
    tp0 = "\n".join(l for l in text.splitlines() if "TP0]" in l) or text   # each TP rank logs the same lines
    kv = re.findall(r"KV Cache is allocated\. dtype: ([\w.]+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB", tp0)
    caps["kv_tokens"] = int(kv[0][1]) if kv else None                 # first TP0 line = target (second = DFlash2 draft)
    caps["kv_gb_target"] = round(float(kv[0][2]) + float(kv[0][3]), 2) if kv else None
    caps["kv_gb_draft"] = round(float(kv[1][2]) + float(kv[1][3]), 2) if len(kv) > 1 else None
    mc = re.findall(r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+)(.*)", text)
    caps["mamba_cache_size"] = int(mc[0][0]) if mc else None
    caps["mamba_cache_line"] = mc[0][1][:240] if mc else None
    vb = re.findall(r"Capture target verify CUDA graph begin\..*?bs=\[([\d, ]+)\]", text)
    db = re.findall(r"Capture target decode CUDA graph begin\..*?bs=\[([\d, ]+)\]", text)
    caps["verify_graph_bs"] = _all_ints(vb[0]) if vb else None
    caps["decode_graph_bs"] = _all_ints(db[0]) if db else None
    caps["prefill_graph_captured"] = "Capture target prefill CUDA graph" in text
    am = re.findall(r"avail mem=([\d.]+) GB", text)
    caps["avail_mem_gb_last"] = float(am[-1]) if am else None
    caps["fp8_kv_scale_warning"] = "no scaling factors provided" in text
    caps["replayssm_lines"] = [l[:200] for l in text.splitlines() if "replayssm" in l.lower()][:4]
    caps["effective_running"] = min(MAX_RUNNING, caps["capped_to"]) if caps["capped_to"] else MAX_RUNNING
    return caps


def capacity_gate(caps, spec=True):
    g = {"running_12": caps.get("effective_running", 0) >= PROD_CONCURRENCY,
         "kv_min": (caps.get("kv_tokens") or 0) >= KV_MIN,
         "verify_graph_12": (not spec) or (PROD_CONCURRENCY in (caps.get("verify_graph_bs") or []))}
    g["pass"] = all(g.values())
    return g


def metric(name):
    """Sum of a Prometheus counter/gauge over its label sets (e.g. sglang:generation_tokens_total)."""
    try:
        txt = str(http("/metrics", timeout=120))
    except Exception:
        return None
    vals = [float(x) for x in re.findall(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([-\d.e+]+)\s*$", txt, re.M)]
    return sum(vals) if vals else None


def run_batch(ids_list, max_new, sampling, seeds, stop_ids=None):''', "caps helpers")
# caps + gate right after a successful launch
C4 = once(C4, '''        try:
            rec["startup"] = startup_lines(srv.log_text())''', '''        try:
            rec["startup"] = startup_lines(srv.log_text())
            rec["caps"] = server_caps(srv.log_text())                              # E11-04
            rec["gate"] = capacity_gate(rec["caps"], spec=bool(spec_flags))
            stamp(f"{name} caps: effective running {rec['caps']['effective_running']} (capped_to {rec['caps']['capped_to']}) | "
                  f"KV {rec['caps']['kv_tokens']} | verify graph bs {rec['caps']['verify_graph_bs']} | free "
                  f"{rec['caps']['avail_mem_gb_last']} GB | gate {'PASS' if rec['gate']['pass'] else 'FAIL ' + str(rec['gate'])}")
            save_results()''', "caps after launch")
C4 = once(C4, "def run_config(name, spec_flags, ssm_dtype=\"float32\", stages=(\"greedy\", \"sweep\", \"long\")):",
          "def run_config(name, spec_flags, ssm_dtype=\"float32\", stages=(\"greedy\", \"sweep\", \"long\")):\n"
          "    # E11-04: the memory ladder (LADDER) supplies --mem-fraction-static; nothing else in the flags sets it", "run_config doc")

# ---------------------------------------------------------------- cell 5: the arms' stages (prod, marker contamination, contcache)
C5 = r'''
# ==================================================================== CELL 5 — E11-04 stages: PROD closed loop, marker contamination, contcache
T10 = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}           # Poetiq solver temperature
CONT_THINK = 6144


def prod_cap(u):
    return max(4096, min(PROD_MAX_NEW, CONTEXT_LEN - len(PROMPT[u]) - ANSWER_MAX_TOKENS - CONTEXT_SAFETY))


def x_prod(srv, name):
    """Closed loop: PROD_CONCURRENCY requests always in flight (refill on finish), real ARC prompts, thinking up to
    60,000 tokens, stop at </think>. Throughput = generation-token counter delta / window (no tail effects)."""
    arm = name.split("_")[0]
    gate = (RESULTS["configs"].get(name) or {}).get("gate") or {}
    seconds = PROD_SEC.get(arm, 15 * 60) if gate.get("pass") else min(PROD_SEC.get(arm, 15 * 60), PROD_SEC_GATE_FAIL)
    seconds = min(seconds, max(60, time_left() - 25 * 60))
    off = srv.log_size()
    g0, t0 = metric("sglang:generation_tokens_total"), time.time()
    done, k = [], 0
    ex = cf.ThreadPoolExecutor(max_workers=PROD_CONCURRENCY)
    futs = {}

    def submit():
        nonlocal k
        u = SWEEP_UNITS[k % len(SWEEP_UNITS)]
        futs[ex.submit(generate, PROMPT[u], prod_cap(u), T10, SEED + 7000 + k, THINK_CLOSE)] = (u, time.time())
        k += 1

    for _ in range(PROD_CONCURRENCY):
        submit()
    g1 = t1 = None
    while futs:
        if t1 is not None and time.time() - t1 > 10 * 60:           # abort_all did not drain the tail: stop waiting
            RESULTS["notes"].append(f"{name}: {len(futs)} PROD requests still running 10 min after the window")
            break
        fin, _ = cf.wait(list(futs), timeout=20, return_when=cf.FIRST_COMPLETED)
        if g1 is None and time.time() - t0 >= seconds:
            g1, t1 = metric("sglang:generation_tokens_total"), time.time()       # window closes here
            try:
                http("/abort_request", {"abort_all": True}, timeout=120)          # stop the in-flight tail
            except Exception as e:
                RESULTS["notes"].append(f"{name} abort_all failed: {e!r}"[:200])
        for f in fin:
            u, ts = futs.pop(f)
            try:
                r = f.result()
                m = r.get("meta_info", {})
                done.append({"unit": u, "prompt": int(m.get("prompt_tokens", 0)), "gen": int(m.get("completion_tokens", 0)),
                             "finish": str((m.get("finish_reason") or {}).get("type")), "accept": m.get("spec_accept_length"),
                             "sec": round(time.time() - ts, 1), "in_window": g1 is None})
            except Exception as e:
                done.append({"unit": u, "error": f"{type(e).__name__}: {str(e)[:200]}", "in_window": g1 is None})
            if g1 is None:
                submit()
    ex.shutdown(wait=False, cancel_futures=True)
    if g1 is None:
        g1, t1 = metric("sglang:generation_tokens_total"), time.time()
    text = srv.log_text(off)
    rows = decode_rows(text)
    (OUT / f"decode_rows_{name}_prod.json").write_text(json.dumps(rows), encoding="utf-8")
    full = [r.get("#full token", r.get("#token", 0)) for r in rows]
    run_counts = [r.get("#running-req", 0) for r in rows]
    st = {"window_s": round(t1 - t0, 1), "gen_tokens": (g1 - g0) if (g0 is not None and g1 is not None) else None,
          "requests_started": k, "finished_in_window": sum(1 for d in done if d.get("in_window") and "error" not in d),
          "natural_close_in_window": sum(1 for d in done if d.get("in_window") and d.get("finish") == "stop"),
          "errors": [d["error"] for d in done if "error" in d and d.get("in_window")][:3],
          "decode_log": summarize_decode(rows, level=PROD_CONCURRENCY),
          "share_lines_at_12": round(sum(1 for c in run_counts if c == PROD_CONCURRENCY) / max(1, len(run_counts)), 3),
          "kv_tokens_in_use_max": max(full) if full else None,
          "context_max": max((d["prompt"] + d["gen"] for d in done if "gen" in d), default=None),
          "retract_lines": sum(1 for l in text.splitlines() if "retract" in l.lower()),
          "gate_pass": gate.get("pass"), "requests": done}
    st["tok_s"] = round(st["gen_tokens"] / st["window_s"], 1) if st["gen_tokens"] and st["window_s"] else None
    st["real_concurrency"] = {"requested": PROD_CONCURRENCY, "running_max": st["decode_log"].get("running_max"),
                              "share_at_level": st["share_lines_at_12"],
                              "pass": (st["decode_log"].get("running_max") or 0) >= PROD_CONCURRENCY and st["share_lines_at_12"] >= 0.5}
    st["meets_target"] = bool(st["tok_s"] and st["tok_s"] >= TARGET_TOK_S)
    stamp(f"{name} PROD {st['window_s']}s: {st['tok_s']} tok/s | real 12 share {st['share_lines_at_12']} | KV in use max "
          f"{st['kv_tokens_in_use_max']} | context max {st['context_max']} | retract lines {st['retract_lines']} | "
          f"finished {st['finished_in_window']} (natural {st['natural_close_in_window']})")
    return st


def x_mark(srv, name):
    """Contamination: 4 greedy prompts solo (twice: repeatability) vs inside a batch with MARK_N companions that each write
    their own codename. A codename in a greedy output, or a companion writing another's codename = contamination."""
    solo_a = {u: generate(PROMPT[u], GREEDY_TOKENS, {"temperature": 0.0}) for u in GREEDY_UNITS}
    solo_b = {u: generate(PROMPT[u], GREEDY_TOKENS, {"temperature": 0.0}) for u in GREEDY_UNITS}
    comp = list(MARK_CODES)
    ids = [PROMPT[u] for u in GREEDY_UNITS] + [PROMPT["M:" + u] for u in comp]
    sps = [{"temperature": 0.0}] * len(GREEDY_UNITS) + [T10] * len(comp)
    out = [None] * len(ids)
    with cf.ThreadPoolExecutor(max_workers=len(ids)) as ex:
        futs = {ex.submit(generate, x, GREEDY_TOKENS, sps[i], SEED + 950 + i): i for i, x in enumerate(ids)}
        for f in cf.as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception as e:
                out[futs[f]] = {"error": repr(e)[:200]}
    ids_of = lambda r: list((r or {}).get("output_ids", []))
    rep = prefix_match({u: ids_of(solo_a[u]) for u in GREEDY_UNITS}, {u: ids_of(solo_b[u]) for u in GREEDY_UNITS})
    bat = prefix_match({u: ids_of(solo_a[u]) for u in GREEDY_UNITS}, {u: ids_of(out[i]) for i, u in enumerate(GREEDY_UNITS)})
    codes = list(MARK_CODES.values())
    leaks = []
    for i, u in enumerate(GREEDY_UNITS):
        t = str((out[i] or {}).get("text", ""))
        leaks += [{"greedy": u, "code": c} for c in codes if c in t]
    own, cross = 0, []
    for j, u in enumerate(comp):
        t = str((out[len(GREEDY_UNITS) + j] or {}).get("text", ""))
        own += MARK_CODES[u] in t
        cross += [{"companion": u, "code": c} for c in codes if c != MARK_CODES[u] and c in t]
    res = {"solo_repeat": rep, "solo_vs_batch": bat, "greedy_leaks": leaks, "companion_cross": cross,
           "companions_wrote_own_code": own, "companions": len(comp),
           "contamination": bool(leaks or cross),
           "note": "solo_repeat < identical = nondeterminism even alone; leaks/cross = contamination (blocker)"}
    return res


def x_contcache(srv, name):
    """Forced-close continuation must hit the prefix cache (Poetiq phase 2 + feedback iterations rely on it)."""
    units = GREEDY_UNITS
    r1, _ = run_batch([PROMPT[u] for u in units], CONT_THINK, T10, [SEED + 7 + i for i in range(len(units))])
    rows = []
    for u, r in zip(units, r1):
        if not r or "error" in r:
            rows.append({"unit": u, "error": (r or {}).get("error")}); continue
        cont = PROMPT[u] + list(r.get("output_ids", [])) + STOP_IDS
        warm = generate(cont, 64, T10, SEED + 11)
        cold_ids = cont[:1] + [(cont[1] + 1) % 240000] + cont[2:]
        cold = generate(cold_ids, 64, T10, SEED + 11)
        mw, mc = warm.get("meta_info", {}), cold.get("meta_info", {})
        rows.append({"unit": u, "prompt_tokens": mw.get("prompt_tokens"), "cached_warm": mw.get("cached_tokens"),
                     "cached_cold": mc.get("cached_tokens"), "wall_warm_s": round(warm["_wall"], 2), "wall_cold_s": round(cold["_wall"], 2)})
    ok = [r for r in rows if r.get("prompt_tokens")]
    ratio = [r["cached_warm"] / r["prompt_tokens"] for r in ok if r.get("cached_warm") is not None]
    sp = round(sum(r["wall_cold_s"] for r in ok) / max(1e-9, sum(r["wall_warm_s"] for r in ok)), 2) if ok else None
    nonempty = all((x or {}).get("output_ids") for x in r1)
    return {"rows": rows, "cache_ratio_min": round(min(ratio), 3) if ratio else None, "speedup_warm_vs_cold": sp,
            "phase1_nonempty": nonempty, "pass": bool(ratio) and min(ratio) >= 0.9 and (sp or 0) >= 1.5 and nonempty}


EXTRA_STAGES.update(prod=x_prod, mark=x_mark, contcache=x_contcache)
print("E11-04 stages ready:", list(EXTRA_STAGES))
'''

# ---------------------------------------------------------------- cell 6: the three arms
C6 = r'''
# ==================================================================== CELL 6 — arms W1 / W2 / W3
DF = SPEC["C_int4_dflash2"]                                            # DFLASH, unquant drafter, 8 draft tokens
NO_PREFILL_GRAPH = ["--disable-prefill-cuda-graph"]                     # frees ~2.6 GB/GPU (E11-03 D log)
LAZY = ["--mamba-radix-cache-strategy", "extra_buffer_lazy", "--max-mamba-cache-size", "56"]   # 4 slots/req -> 14 >= 12
# ReplaySSM spec-verify: raw-input window + fp32 replay instead of per-draft state snapshots (memory_pool.py:758-790).
# no_buffer, not lazy: the ReplaySSM ring is not reset on the extra_buffer donation path (attention_hook.py:315-320).
REPLAY = ["--mamba-radix-cache-strategy", "no_buffer", "--max-mamba-cache-size", "48", "--enable-linear-replayssm-spec"]
ARMS = [
    ("W1_lazy_fp16", DF + NO_PREFILL_GRAPH + LAZY, "float16", ("greedy", "sweep", "prod", "mark", "contcache", "sanity")),
    ("W2_replay_fp32", DF + NO_PREFILL_GRAPH + REPLAY, "float32", ("greedy", "sweep", "prod", "contcache", "sanity")),
    ("W3_lazy_bf16", DF + NO_PREFILL_GRAPH + LAZY, "bfloat16", ("greedy", "sweep", "prod")),
]
if not PREFLIGHT_OK:
    RESULTS["notes"].append("preflight failed: server stages skipped")
else:
    for _name, _flags, _ssm, _stages in ARMS:
        if time_left() < 30 * 60:
            RESULTS["notes"].append(f"{_name} skipped: time budget")
            continue
        _st = list(_stages)
        if "sanity" in _st and _name != "W1_lazy_fp16" and time_left() < 75 * 60:
            _st.remove("sanity")                                         # W1's sanity is the must-have one
        run_config(_name, _flags, ssm_dtype=_ssm, stages=tuple(_st))
        _r = RESULTS["configs"].get(_name, {})
        if _r.get("sanity"):
            RESULTS.setdefault("sanity_by_arm", {})[_name] = _r["sanity"]
    _ref = (RESULTS["configs"].get("W1_lazy_fp16") or {}).get("greedy_ids")
    for _name in ("W2_replay_fp32", "W3_lazy_bf16"):
        _got = (RESULTS["configs"].get(_name) or {}).get("greedy_ids")
        if _ref and _got:
            RESULTS["configs"][_name]["greedy_vs_W1"] = prefix_match(_ref, _got)


def _arm_ok(rec):
    p = (rec or {}).get("prod") or {}
    return bool(rec and rec.get("ok") and (rec.get("gate") or {}).get("pass") and (p.get("real_concurrency") or {}).get("pass"))


_c = [((r.get("prod") or {}).get("tok_s") or 0, n) for n, r in RESULTS["configs"].items() if _arm_ok(r)]
RESULTS["best_spec"] = max(_c)[1] if _c and max(_c)[0] > 0 else None
save_results()
stamp(f"best arm (prod tok/s, capacity gate + real 12 passed): {RESULTS['best_spec']} {sorted(_c, reverse=True)}")
'''

# ---------------------------------------------------------------- cell 7: report + GO/NO-GO
C7 = r'''
# ==================================================================== CELL 7 — report + GO / NO-GO
def _f(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


L = [f"# E11-04 capacity + real-concurrency validation · run {RUN_ID}", ""]
_pf = RESULTS.get("preflight") or {}
L += [f"Driver {_pf.get('driver')} (CUDA {_pf.get('driver_cuda')}) · preflight {'PASS' if PREFLIGHT_OK else 'FAIL'} · "
      f"checkpoint k/v scale tensors {RESULTS.get('kv_scale_tensors_in_checkpoint')} · kv_cache_scheme "
      f"{json.dumps(RESULTS.get('kv_cache_scheme'))[:160]}", ""]
L += ["## Startup caps (from the server log)", "",
      "| arm | launch | effective running | capped_to | state slots/req | KV tokens | KV GB target+draft | mamba cache | "
      "verify graph bs | prefill graph | free GB | FP8 scale warning | gate |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n, r in RESULTS["configs"].items():
    c, g = r.get("caps") or {}, r.get("gate") or {}
    att = r.get("attempts") or []
    L.append(f"| {n} | {[(a['extra_flags'][-1] if a['extra_flags'] else '', a['why']) for a in att]} | {c.get('effective_running')} | "
             f"{c.get('capped_to')} | {c.get('state_slots_per_request')} | {c.get('kv_tokens')} | {c.get('kv_gb_target')}+{c.get('kv_gb_draft')} | "
             f"{c.get('mamba_cache_size')} | {c.get('verify_graph_bs')} | {c.get('prefill_graph_captured')} | {c.get('avail_mem_gb_last')} | "
             f"{c.get('fp8_kv_scale_warning')} | {'PASS' if g.get('pass') else 'FAIL ' + json.dumps(g)} |")
L += ["", "## Sweep (short prompts, 4,096 tokens) with real concurrency", "",
      "| arm | requests | total tok/s | log gen tok/s | accept | real running max | real 12? |", "|---|---|---|---|---|---|---|"]
for n, r in RESULTS["configs"].items():
    for c, st in (r.get("sweep") or {}).items():
        if "skipped" in st:
            continue
        rc = st.get("real_concurrency") or {}
        L.append(f"| {n} | {c} | {_f(st.get('agg_tok_s'))} | {_f((st.get('decode_log') or {}).get('gen_tok_s_median'))} | "
                 f"{_f(st.get('accept_len_mean'), 2)} | {rc.get('running_max')} | {'PASS' if rc.get('pass') else 'FAIL'} |")
L += ["", f"## PROD closed loop ({PROD_CONCURRENCY} in flight, ARC prompts, up to {PROD_MAX_NEW} thinking, T 1.0) · target >= {TARGET_TOK_S} tok/s", "",
      "| arm | window s | gen tokens | **tok/s** | share of log lines at 12 | accept | KV in use max | context max | retract lines | finished (natural) | meets target |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for n, r in RESULTS["configs"].items():
    p = r.get("prod")
    if not p or "window_s" not in p:
        if p:
            L.append(f"| {n} | {json.dumps(p)[:200]} |||||||||| ")
        continue
    L.append(f"| {n} | {p['window_s']} | {p['gen_tokens']} | **{_f(p['tok_s'])}** | {p['share_lines_at_12']} | "
             f"{_f(p['decode_log'].get('accept_len_median'), 2)} | {p['kv_tokens_in_use_max']} | {p['context_max']} | {p['retract_lines']} | "
             f"{p['finished_in_window']} ({p['natural_close_in_window']}) | {p['meets_target']} |")
L += ["", "## Correctness", ""]
for n, r in RESULTS["configs"].items():
    if r.get("mark"):
        m = r["mark"]
        L.append(f"- **{n} marker contamination:** contamination {m.get('contamination')} · greedy leaks {m.get('greedy_leaks')} · "
                 f"companion cross {m.get('companion_cross')} · companions wrote own code {m.get('companions_wrote_own_code')}/{m.get('companions')} · "
                 f"solo repeat identical {m['solo_repeat']['identical']}/{len(GREEDY_UNITS)} (mean prefix {m['solo_repeat']['mean_prefix']}) · "
                 f"solo vs batch identical {m['solo_vs_batch']['identical']}/{len(GREEDY_UNITS)} (mean prefix {m['solo_vs_batch']['mean_prefix']})")
    if r.get("contcache"):
        cc = r["contcache"]
        L.append(f"- **{n} contcache:** pass {cc.get('pass')} · min cached ratio {cc.get('cache_ratio_min')} · warm vs cold {cc.get('speedup_warm_vs_cold')}x")
    if r.get("greedy_vs_W1"):
        L.append(f"- **{n} greedy vs W1:** {r['greedy_vs_W1']['identical']}/{len(GREEDY_UNITS)} identical, mean prefix {r['greedy_vs_W1']['mean_prefix']}")
    if r.get("error"):
        L.append(f"- **{n} error:** {r['error']}")
for n, s in (RESULTS.get("sanity_by_arm") or {}).items():
    L += ["", f"### Sanity {n}: {s['hits']}/{s['n']} solved · FP8 expectation {s['fp8_expected_hits']}", "",
          "| puzzle | hit | mode | think tokens | FP8 think | accept | sec |", "|---|---|---|---|---|---|---|"]
    for row in s["rows"]:
        L.append(f"| {row['uid']} | {row.get('hit')} | {row.get('mode', row.get('error'))} | {row.get('think_tokens')} | "
                 f"{row.get('fp8_think')} | {_f(row.get('accept_len'), 2)} | {row.get('sec')} |")
# GO / NO-GO per arm
L += ["", "## GO / NO-GO (production candidate = all PASS)", "",
      "| arm | capacity gate | real 12 in PROD | PROD >= target | no retraction | no contamination | sanity >= 6/8 | contcache |",
      "|---|---|---|---|---|---|---|---|"]
_w1mark = ((RESULTS["configs"].get("W1_lazy_fp16") or {}).get("mark") or {})
for n, r in RESULTS["configs"].items():
    p, s = r.get("prod") or {}, (RESULTS.get("sanity_by_arm") or {}).get(n)
    m = r.get("mark") or _w1mark
    cells = [(r.get("gate") or {}).get("pass"), (p.get("real_concurrency") or {}).get("pass"), p.get("meets_target"),
             (p.get("retract_lines") == 0) if p else None, (not m.get("contamination")) if m else None,
             (s["hits"] >= 6) if s else None, (r.get("contcache") or {}).get("pass")]
    L.append(f"| {n} | " + " | ".join("-" if v is None else ("PASS" if v else "**FAIL**") for v in cells) + " |")
L += ["", f"Total time {(time.time() - T0) / 60:.0f} min · best arm: {RESULTS.get('best_spec')} · notes: {RESULTS.get('notes')}"]
REPORT = "\n".join(L)
(OUT / "report.md").write_text(REPORT, encoding="utf-8")
RESULTS["total_min"] = round((time.time() - T0) / 60, 1)
save_results()
print(REPORT)
'''

cells = [md(M0), code(C0), code(C1), code(C2), code(C3), code(C4), code(C5), code(C6), code(C7)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
(OUTDIR / NAME).write_bytes(data)
print("BUILT", NAME, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())
