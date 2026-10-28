# ---- E21: the new code of the engine cell (the donor functions above are inserted verbatim by build_e21.py).
# rev 2 (2026-10-28): the fixes of the 10 validation reports (Experiment Review/VAL_*.md; list in the README "rev 2").

# VAL_api_D2 F2 / VAL_engine_A1 L2: vLLM 0.19 inserts ", Preemptions: N" and 0.32 also ", Deferred: N reqs" / "KV fetch"
# and drops the prefix-cache field until the first prefix query; the E12-V regex (above) missed exactly those lines.
# Replaces the E12-V _E12V_STAT_RE; groups: 1 prompt tok/s, 2 gen tok/s, 3 running, 4 waiting, 5 preemptions (or None),
# 6 KV %, 7 prefix-cache hit % (or None).
_E12V_STAT_RE = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
                           r"Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: \d+ reqs)?(?:, KV fetch: [^%]*?completed waiting)?"
                           r"(?:, Preemptions: (\d+))?, GPU KV cache usage: ([\d.]+)%(?:, Prefix cache hit rate: ([\d.]+)%)?")


def e21_stat_rows(text):
    """vLLM's 10-s stats lines as dicts (both versions, every optional field)."""
    out = []
    for m in _E12V_STAT_RE.finditer(text):
        g = m.groups()
        out.append({"prompt_tps": float(g[0]), "gen_tps": float(g[1]), "running": int(g[2]), "waiting": int(g[3]),
                    "preempt": int(g[4]) if g[4] is not None else 0, "kv_pct": float(g[5]),
                    "prefix_pct": float(g[6]) if g[6] is not None else None})
    return out


def e11_decode_summary():
    """E12-V's stats-line summary on the rev 2 regex (preemption / deferred lines included) + the counter snapshot."""
    if E11_SERVER is None:
        return None
    txt = E11_SERVER.log_text()
    rows = e21_stat_rows(txt)
    med = lambda v: round(_statistics.median(v), 1) if v else None
    acc = re.findall(r"Mean acceptance length: ([\d.]+)", txt)
    pre = [r["prefix_pct"] for r in rows if r["prefix_pct"] is not None]
    out = {"n_lines": len(rows),
           "running_hist": {str(k): sum(1 for r in rows if r["running"] == k) for k in sorted({r["running"] for r in rows})},
           "gen_tok_s_median_at_concurrency": med([r["gen_tps"] for r in rows if r["running"] >= E12_VLLM_CONCURRENCY]),
           "gen_tok_s_median_all_busy": med([r["gen_tps"] for r in rows if r["running"] > 0]),
           "waiting_max": max((r["waiting"] for r in rows), default=None),
           "kv_usage_pct_max": max((r["kv_pct"] for r in rows), default=None),
           "prefix_cache_hit_pct_last_rolling": pre[-1] if pre else None,
           "preemptions_in_stats_lines": sum(r["preempt"] for r in rows),
           "spec_mean_acceptance_median": med([float(a) for a in acc]),
           "accept_len_median": med([float(a) for a in acc]),      # the key CELL 7 reads
           "counters": ENGINE_INFO.get("final_counters") or e21_counters()}
    try:
        write_json_atomic(OUT / "sglang_decode_summary.json", out)   # bundled under this name by CELL 7
    except Exception:
        pass
    return out


_E21_COUNTERS = ("vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                 "vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:request_success_total",
                 "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total")


def e21_counters():
    """VAL_engine_A2 M3 / VAL_api_D2 F6-F7: the server's own counters (preemptions, whole-run prefix hits, tokens), summed
    over label rows. Taken before every kill (a dead server loses them)."""
    try:
        rows = e21_prom()
    except Exception as e:
        return {"error": repr(e)[:200]}
    out = {}
    for r in rows:
        if r.name in _E21_COUNTERS:
            out[r.name] = out.get(r.name, 0.0) + float(r.value)
    q, h = out.get("vllm:prefix_cache_queries_total"), out.get("vllm:prefix_cache_hits_total")
    out["prefix_hit_pct_whole_run"] = round(100 * h / q, 2) if q else None
    return out


def e21_snapshot_counters():
    if E11_SERVER is not None and E11_SERVER.alive():
        c = e21_counters()
        if "error" not in c:
            ENGINE_INFO["final_counters"] = c


def e11_kill_server(wait=True):
    """E21 harness finding: the E12-V e11_kill_server calls kill(wait=...), but the server here is an E18Server whose
    kill() takes `fast` (CELL 7's final call raised TypeError, silently). rev 2: the counters are snapshot first."""
    e21_snapshot_counters()
    if E11_SERVER is not None:
        E11_SERVER.kill(fast=not wait)


def e21_duck_serving_flags(served, port):
    """E12-V CELL 5 (the vLLM SMOKE1 run 2 server): Duck's start_vllm_server flags, verbatim, plus the prompt-tokens
    details (cached_tokens in usage) and the 135,168-token context (Duck window 128 Ki + margin). rev 2: no uvicorn
    access log (VAL_api_D2 F4: our own /metrics polls kept the log growing, so the watchdog never reached /health)."""
    return ["--served-model-name", served, "--host", "127.0.0.1", "--port", str(port),
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--generation-config", "vllm",
            "--enable-prefix-caching", "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}),
            "--reasoning-parser", "qwen3", "--max-model-len", str(E21_VLLM_MAX_MODEL_LEN), "--enable-prompt-tokens-details",
            "--disable-uvicorn-access-log"]


def e21_reject_broken_int4(cfg):
    """rev 2i (GPT INT4 audits 1+2): RedHat INT4 on vLLM 0.19.0 generates garbage from the first token (GPU smoke run 1,
    2026-10-29; the same checkpoint is coherent on 0.32). Never launch that pair."""
    if cfg.get("vllm") == "0.19.0" and cfg.get("weights") == "int4":
        raise RuntimeError("E21_BLOCKER_INT4_V019: RedHat INT4 generates garbage on vLLM 0.19.0 (GPU smoke run 1)")


E21_CONTENT_PROBES = [("Reply with exactly one number: 7 * 6 =", "42", "number"),          # audit 1
                      ("Repeat exactly, nothing else: ARC-GRID-17", "ARC-GRID-17", "copy"),   # audit 2
                      ("Complete the sequence with the next number only: 2, 4, 6, 8,", "10", "number")]   # audit 2
E21_NUMBER_WORDS = {"4": "four", "10": "ten", "42": "forty-two"}


def e21_garbage(gen, limit=80):
    """rev 2i: empty, too long for the question, or < 90 % ASCII (the 0.19 INT4 multilingual token soup)."""
    g = (gen or "").strip()
    return not g or len(g) > limit or sum(ch.isascii() for ch in g) < 0.9 * len(g)


def e21_content_ok(gen, want, kind, limit=80):
    """rev 2i (GPT INT4 audits 1+2): a clean reply that holds the answer (number: the last number in it, or its word)."""
    g = (gen or "").strip()
    if e21_garbage(g, limit):
        return False
    if kind == "number":
        return re.findall(r"-?\d+", g)[-1:] == [want] or E21_NUMBER_WORDS.get(want, "\0") in g.lower()
    return want in g


def e21_content_probes(served):
    """rev 2i: three more greedy, thinking-off probes after the 2 + 2 smoke. A garbage reply (or one cut at the limit) is
    fatal in every mode; a clean but wrong answer is fatal in commit / smoke and a recorded warning in the hidden rerun,
    so a formatting surprise can never leave the rerun without an engine."""
    res = []
    for prompt, want, kind in E21_CONTENT_PROBES:
        r = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload={
            "model": served, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
        c = r["choices"][0]
        gen = (c["message"].get("content") or "").strip()
        res.append({"want": want, "generated": gen[:120], "finish": c.get("finish_reason"),
                    "clean": not e21_garbage(gen) and c.get("finish_reason") != "length",
                    "ok": e21_content_ok(gen, want, kind) and c.get("finish_reason") != "length"})
    if not all(x["clean"] for x in res):
        raise RuntimeError(f"content probes: garbage reply {json.dumps(res, ensure_ascii=True)[:600]}")
    if not all(x["ok"] for x in res):
        if not RERUN:
            raise RuntimeError(f"content probes: wrong answer {json.dumps(res, ensure_ascii=True)[:600]}")
        print("E21 WARNING (rerun, not fatal): content probes wrong answer", json.dumps(res, ensure_ascii=True)[:300], flush=True)
    return res


def e21_model_path(weights):
    return E21_STAGED.get(weights) or (SGL_MODEL_PATH if weights == "int4" else MODEL_PATH)


def e21_flags_v019(cfg, port, backend):
    """vLLM 0.19 = the E12-V CELL 5 command: Duck's flags + the S1-60K engine (TP4, fp8_e4m3 KV, MTP-3, FlashInfer,
    4096 batched tokens, 0.90 GPU memory, language-model-only, custom all-reduce off, seed). E21 changes: the model path
    (FP8 in both variants since rev 2i), --max-num-seqs per half, --mamba-ssm-cache-dtype for NB2 (vLLM 0.19 CacheConfig.mamba_ssm_cache_dtype)."""
    f = ["--model", str(e21_model_path(cfg["weights"]))] + e21_duck_serving_flags(E21_SERVED[cfg["weights"]], port)
    f += ["--tensor-parallel-size", "4", "--kv-cache-dtype", KV_CACHE_DTYPE, "--speculative-config", json.dumps(SPECULATIVE_CONFIG),
          "--attention-backend", backend, "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
          "--max-num-seqs", str(cfg["seqs"]), "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
          "--language-model-only", "--disable-custom-all-reduce", "--seed", str(SEED)]
    if cfg.get("ssm"):
        f += ["--mamba-ssm-cache-dtype", cfg["ssm"]]
    return f


def e21_flags_v032(cfg, port, backend):
    """vLLM 0.32 = the VMS CELL 6 vms_flags() arm (fp8_mtp3_ssm_fp16 for NB2, int4_mtp3_s20 for NB1, the reference
    fp8_mtp3 as the fallback): TP4, fp8_e4m3 KV, max-num-seqs, 4096 batched, 0.90, custom all-reduce off, seed,
    FlashInfer, --trust-remote-code, prefix caching, MTP-3 (+ --mamba-ssm-cache-dtype); E21: Duck's serving flags and the
    135,168 context (VMS: 106,496). rev 2 (VAL_fidelity_E2 M1, VAL_engine_A1 H2): + --language-model-only, as half A and
    as the E18 R2 rung s1 that became ready on 0.32 (KV 1,110,415 with it vs 1,006,517 without, same FP8 / 20 seqs)."""
    f = [str(e21_model_path(cfg["weights"])), "--tensor-parallel-size", "4", "--kv-cache-dtype", KV_CACHE_DTYPE,
         "--max-num-seqs", str(cfg["seqs"]), "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
         "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION), "--disable-custom-all-reduce", "--seed", str(SEED),
         "--attention-backend", backend, "--trust-remote-code",
         "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3}), "--language-model-only"]
    f += e21_duck_serving_flags(E21_SERVED[cfg["weights"]], port)
    if cfg.get("ssm"):
        f += ["--mamba-ssm-cache-dtype", cfg["ssm"]]
    return f


def e21_prom():
    """The server's /metrics as rows with .name / .labels / .value (the e11_prom() shape CELL 6 reads)."""
    from types import SimpleNamespace
    txt = _urlreq.urlopen(SGL_URL + "/metrics", timeout=10).read().decode("utf-8", "replace")
    rows = []
    for line in txt.splitlines():
        m = _E12V_PROM_RE.match(line) if line and not line.startswith("#") else None
        if not m:
            continue
        try:
            rows.append(SimpleNamespace(name=m.group(1), labels=m.group(2) or "", value=float(m.group(3))))
        except ValueError:
            pass
    return rows


E21_SIZING_TOKENS = 57000        # R3K2 per-attempt peak context p50 (p90 65k, max 79k, mean ~50k)
E21_MIN_STREAMS = 8              # = R3K2 SGL_MIN_RUNNING: fewer streams -> this config is refused (the fallback runs)


def e21_capacity(log, seqs):
    """Pre-run capacity check, fail-closed (rev 2: VAL_engine_A1 H1, VAL_engine_A2 M1, VAL_api_D2 F3, VAL_fidelity_E1 M3).
    vLLM's "Maximum concurrency for M tokens per request: X" = num_blocks / (cdiv(M, B) + F): each request holds
    cdiv(L, B) attention blocks plus F fixed Mamba blocks (align mode: 3 groups x (2 + spec tokens) = 15 with MTP-3,
    kv_cache_interface.py; checked on the E12-V log: 8.46 x (85 + 15) = 846 blocks, and 846 // 4 x 1,600 = 337,600 = its
    'GPU KV cache size' line). Streams = min(seqs, num_blocks // (cdiv(57k, B) + F)); a missing line, or < 8 streams,
    refuses the config."""
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    mc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", log)
    bs = re.findall(r"Setting attention block size to (\d+) tokens", log)
    if not mc or not bs:
        raise RuntimeError(f"capacity: vLLM log lines missing (max concurrency {bool(mc)}, block size {bool(bs)})")
    B, M, X = int(bs[-1]), int(mc[-1][0].replace(",", "")), float(mc[-1][1])
    F = 3 * (2 + int((SPECULATIVE_CONFIG or {}).get("num_speculative_tokens", 0)))
    cd = lambda a: -(-int(a) // B)
    nb = int(round(X * (cd(M) + F)))
    need = lambda L, n: n * (cd(L) + F)
    out = {"kv_tokens_logged": int(kv[-1].replace(",", "")) if kv else None, "max_concurrency_logged": {"tokens": M, "x": X},
           "block_size": B, "fixed_blocks_per_request": F, "num_blocks": nb,
           "effective_kv_tokens": nb * B,
           "util_mean_50k": round(need(50000, seqs) / nb, 3), "util_p50_peak_57k": round(need(57000, seqs) / nb, 3),
           "util_p90_peak_65k": round(need(65000, seqs) / nb, 3), "util_max_79k": round(need(79000, seqs) / nb, 3),
           "streams_fit": {str(L): nb // (cd(L) + F) for L in (50000, 57000, 65000, 79000)}}
    out["streams"] = min(int(seqs), nb // (cd(E21_SIZING_TOKENS) + F))
    out["fits_p50_peak"] = out["streams"] >= int(seqs)
    if out["streams"] < E21_MIN_STREAMS:
        raise RuntimeError(f"capacity: only {out['streams']} streams fit ({json.dumps(out)})")
    return out


# rev 2d (GPT audit 3): Duck's own tool definition = tool_agent.ToolAgent._tools() with arc2's description
# (arc2_duck.py 1572), so the probe exercises the exact schema the Duck sends
E21_PROBE_TOOLS = [{"type": "function", "function": {"name": "python", "description": E12A.ARC2_TOOL_DESCRIPTION,
                    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": (
                        "Python code to run. The snippet is ephemeral and is not saved across tool calls.")}},
                        "required": ["code"]}}}]


def e21_contract_probe(served):
    """rev 2 (VAL_api_D1 F2, VAL_engine_A1 M1): the Duck contract on this server, before a 5.5-h half. Turn 1: tools +
    thinking (reasoning_effort low) asking for one python call. Hard failures: an HTTP error, tool-call arguments that are
    not a JSON object with a string 'code', think / tool-call markup leaking into content, a degenerate reply. If a call
    came back, turn 2 sends it as history (reasoning under 'reasoning', tool result) and the prompt must grow by about
    turn 1's prompt + completion (a dropped reasoning shows up as a shortfall, the E12-V check). No call within the
    budget (thinking ran long) is inconclusive, not a failure."""
    out = {"ok": False}
    kw = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"}
    msgs = [{"role": "user", "content": "Use the python tool exactly once to run print(6 * 7). Keep your thinking very short."}]
    r1 = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload={
        "model": served, "messages": msgs, "tools": E21_PROBE_TOOLS, "tool_choice": "auto", "temperature": 0.0,
        "max_tokens": 3072, "chat_template_kwargs": kw}, timeout=600)
    m, u = r1["choices"][0]["message"], r1.get("usage") or {}
    reas, cont, calls = m.get("reasoning") or m.get("reasoning_content") or "", m.get("content") or "", m.get("tool_calls") or []
    out.update(finish=r1["choices"][0].get("finish_reason"), reasoning_chars=len(reas), content=cont[:200],
               tool_calls=len(calls), prompt=u.get("prompt_tokens"), completion=u.get("completion_tokens"))
    bad = [k for k in ("<think>", "</think>", "<tool_call>", "<function=") if k in cont]
    if bad:
        raise RuntimeError(f"contract: markup in content {bad}: {cont[:300]!r}")
    txt = (reas + cont).strip()
    if len(txt) >= 40 and len(set(txt.replace(" ", "").replace("\n", ""))) <= 2:
        raise RuntimeError(f"contract: degenerate reply {txt[:80]!r}")
    for c in calls:
        a = json.loads(c["function"]["arguments"])
        if not (isinstance(a, dict) and isinstance(a.get("code"), str) and a["code"].strip()):
            raise RuntimeError(f"contract: tool-call arguments {c['function']['arguments'][:200]!r}")
    if not calls:
        out.update(turn2="inconclusive: no tool call in turn 1")
        e21_soft(out, "cache", lambda: e21_cache_probe(served))
        out["ok"] = True
        return out
    hist = msgs + [{"role": "assistant", "reasoning": reas, "tool_calls": calls},
                   {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": "42"}]
    r2 = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload={
        "model": served, "messages": hist, "tools": E21_PROBE_TOOLS, "tool_choice": "auto", "temperature": 0.0,
        "max_tokens": 512, "chat_template_kwargs": kw}, timeout=600)
    p2 = (r2.get("usage") or {}).get("prompt_tokens")
    short = (int(out["prompt"]) + int(out["completion"])) - int(p2)
    out.update(turn2_prompt=p2, turn2_shortfall=short)
    if reas and short > max(128, len(reas) // 6):
        e21_soft(out, "history", lambda: (_ for _ in ()).throw(
            RuntimeError(f"contract: history prompt {p2} short by {short} tokens (reasoning dropped?)")))
    e21_soft(out, "cache", lambda: e21_cache_probe(served))
    out["ok"] = True
    return out


def e21_soft(out, key, fn):
    """rev 2f: a check that is fatal in commit / smoke (falls back) but only a recorded warning in the hidden rerun."""
    try:
        out[key] = fn()
    except Exception as e:
        if not RERUN:
            raise
        out.setdefault("warnings", []).append(f"{key}: {e}"[:600])
        print("E21 WARNING (rerun, not fatal):", key, str(e)[:300], flush=True)


def e21_cache_probe(served):
    """rev 2b (VAL_api_D2 F5): the align-mode prefix-cache path that serves 67-94 % of every later Duck prompt. A shared
    ~5,000-token prefix with a marker per request: the cold request, then the same request again (it must hit the cache,
    >= 1 block) at temperature 0, thinking off; both replies must contain the request's own marker and be equal."""
    import random as _rnd
    _r7 = _rnd.Random(7)                       # rev 2f (VAL2_config_smoke F6): one RNG (the old code repeated one digit)
    rows = " ".join(str(_r7.randrange(10)) for _ in range(6000))
    res = []
    for i in range(2):
        mark = f"QZ{_rnd.randrange(10 ** 6, 10 ** 7)}K{i}"
        msgs = [{"role": "user", "content": "Grid digits for context: " + rows + f"\n\nReply with exactly this code: {mark}"}]
        body = {"model": served, "messages": msgs, "temperature": 0.0, "max_tokens": 96,
                "chat_template_kwargs": {"enable_thinking": False}}
        r1 = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload=body, timeout=300)
        r2 = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload=body, timeout=300)
        g1 = r1["choices"][0]["message"].get("content") or ""
        g2 = r2["choices"][0]["message"].get("content") or ""
        u2 = r2.get("usage") or {}
        c2 = ((u2.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        res.append({"prompt": u2.get("prompt_tokens"), "cached_repeat": c2, "own_marker": mark in g1 and mark in g2,
                    "equal": g1 == g2})
    if not all(x["own_marker"] and x["equal"] and x["cached_repeat"] >= 1024 for x in res):
        raise RuntimeError(f"cache probe failed: {json.dumps(res)}")
    return res


def e21_v032_install_bg():
    """rev 2b (VAL_timing_C1 M3, VAL_fidelity_E2 LOW-7): build the vLLM 0.32 venv on CPU while half A solves, so half B's
    setup is only its launch. The result (or the error) is what e21_launch would have produced."""
    def _run():
        try:
            E21_STATE["v032_install"] = e18_install()
        except Exception as e:
            E21_STATE["v032_install"] = {"error": f"{type(e).__name__}: {e}"[:600]}
    if E21_STATE.get("v032_install") is None and E21_STATE.get("v032_thread") is None:
        E21_STATE["v032_thread"] = threading.Thread(target=_run, daemon=True, name="e21_v032_install")
        E21_STATE["v032_thread"].start()


def e21_launch(H, cfg, tag, timeout=None):
    """One server launch for half H with config cfg; returns (server, info). Raises on failure (its own server killed).
    rev 2: the strict smoke + the contract probe + the fail-closed capacity check are part of 'ready'; a relaunch passes
    its own timeout; the failure path never touches a server or the GPUs of another half (VAL_state_B2 L1)."""
    global E18_URL, E11_SERVER
    port = E21_PORTS[H]
    E18_URL = f"http://127.0.0.1:{port}"
    backend = "FLASHINFER" if CUDA_LINK.get("probe") == "PASS" else "TRITON_ATTN"
    info = {"tag": tag, "cfg": dict(cfg), "attention_backend": backend, "port": port}
    try:
        e21_reject_broken_int4(cfg)                  # rev 2i
    except RuntimeError as _e:
        raise RuntimeError(str(_e), info)
    if cfg["vllm"] == "0.19.0":
        if not E11_HARNESS:
            assert pkg_version("vllm") == "0.19.0", pkg_version("vllm")
        launch = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] if not E21_FAKE else [sys.executable, E21_FAKE]
        env = os.environ.copy()
        env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1",
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "PYTHONUNBUFFERED": "1"})   # E12-V: Duck's vllm_env + S1
        flags = e21_flags_v019(cfg, port, backend)
    else:
        _th = E21_STATE.get("v032_thread")
        if _th is not None and _th.is_alive():
            _th.join(timeout=2400)
        if isinstance(E21_STATE.get("v032_install"), dict) and E21_STATE["v032_install"].get("error"):
            raise RuntimeError("vLLM 0.32 install failed earlier: " + E21_STATE["v032_install"]["error"], info)   # A1 M4
        if E21_STATE.get("v032_install") is None:
            try:
                E21_STATE["v032_install"] = e18_install()        # the VMS installer (E18 R2), once per notebook
            except Exception as e:
                E21_STATE["v032_install"] = {"error": f"{type(e).__name__}: {e}"[:600]}
                raise RuntimeError("vLLM 0.32 install failed: " + E21_STATE["v032_install"]["error"], info)
        inst = E21_STATE["v032_install"]
        launch, env = list(inst["launch"]), dict(inst["env"])
        env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
        env.update(TRITON_CACHE_DIR="/tmp/e21-v032-triton", FLASHINFER_WORKSPACE_BASE="/tmp/e21-v032-flashinfer")
        flags = e21_flags_v032(cfg, port, backend)
    srv = E18Server(f"half{H}_{tag}", launch + flags, env)
    info.update(flags=flags, log=str(srv.log_path))
    if timeout is None:
        _cap = E21_LAUNCH_TIMEOUT_SEC if cfg["vllm"] == "0.19.0" else E21_LAUNCH_TIMEOUT_V032_SEC
        timeout = max(60.0, min(_cap, E21_HALF_END_PC - time.perf_counter() - 30 * 60))
    try:
        srv.start()
        E11_SERVER = srv
        print(f"E21 half {H} launch {tag}: vllm {cfg['vllm']}", " ".join(str(x) for x in launch[-2:]), " ".join(flags),
              "| ready timeout", round(timeout), flush=True)
        ok, why = srv.wait_ready(timeout)
        info.update(ready=ok, why=why, ready_sec=round(time.perf_counter() - srv.t0, 1))
        if not ok:
            raise RuntimeError(f"half {H} {tag}: not ready ({why})")
        models = _e12v_request_json(SGL_URL + "/v1/models", timeout=30)
        info["models"] = [m.get("id") for m in models.get("data", [])]
        info["capacity"] = e21_capacity(srv.log_text(), cfg["seqs"])
        # rev 2c (GPT audit 3): the engine really runs text-only (the non-default-args line of both versions)
        info["lmo_in_log"] = "'language_model_only': True" in srv.log_text()
        # the E12-V smoke, strict (rev 2: VAL_engine_A2 H1, VAL_timing_C2 M4: "!!!!" output must not pass)
        r = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload={
            "model": E21_SERVED[cfg["weights"]], "messages": [{"role": "user", "content": "Answer in one short sentence: what is 2 + 2?"}],
            "temperature": 0.0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
        gen = (r["choices"][0]["message"].get("content") or "").strip()
        usage = r.get("usage") or {}
        info["smoke"] = {"generated": gen[:200], "usage": usage, "finish": r["choices"][0].get("finish_reason"),
                         "cache_details_present": isinstance(usage.get("prompt_tokens_details"), dict),
                         # rev 2i (GPT INT4 audit 1, section 6): the answer, not "a 4 anywhere"; clean, mostly ASCII
                         "ok": e21_content_ok(gen, "4", "number", limit=160)
                         and not (len(gen) >= 8 and len(set(gen.replace(" ", ""))) <= 2)   # rev 2g (VAL2_engine F10)
                         and isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0
                         and isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] > 0}
        if not info["smoke"]["ok"]:
            raise RuntimeError(f"half {H} {tag}: smoke failed {json.dumps(info['smoke'])[:400]}")
        info["content_probes"] = e21_content_probes(E21_SERVED[cfg["weights"]])   # rev 2i
        info["contract"] = e21_contract_probe(E21_SERVED[cfg["weights"]])
        print(f"E21 half {H} {tag}: {why} | capacity", json.dumps(info["capacity"]), "| smoke", json.dumps(info["smoke"]),
              "| contract", json.dumps(info["contract"]), flush=True)
    except Exception as e:
        info["log_tail"] = srv.log_text()[-3000:]
        srv.kill()
        if E11_SERVER is srv:
            E11_SERVER = None
        if E21_HALF == H:            # a straggler relaunch of a finished half never kills the next half's server
            info["gpu_pids_killed"] = _e18_kill_gpu_pids()
            info["gpu_release"] = _wait_gpu_release()
        raise RuntimeError(f"{type(e).__name__}: {e}"[:900], info)
    return srv, info


E21_RELAUNCH_MIN_SEC = 12 * 60   # vLLM needs ~7-10 min to be ready: a relaunch with less is pointless


def e21_relaunch():
    """The CELL 6 watchdog's relaunch: the same half, the config that was ready. rev 2 (VAL_state_B1 M1, VAL_engine_A1 M6,
    VAL_fidelity_E1 LOW-3): the ready timeout follows the solve end (min(30 min, end - now - 5 min)), as the E15B donor's
    min(20 min, left - 10 min) did; none with less than 12 min."""
    H, cfg = E21_HALF, ENGINE_INFO.get("cfg")
    if len(ENGINE_INFO.get("relaunches") or []) >= 1:          # the same config already crashed once: try the other one
        _alt = [c for _t, c in e21_cfgs(ENGINE_INFO.get("spec") or E21_HALF_SPECS[E21_VARIANT][H]) if c != cfg
                and c.get("weights") == (cfg or {}).get("weights")]   # rev 2g (VAL2_engine F1): same served name
        if _alt:
            cfg = _alt[0]
            ENGINE_INFO["relaunch_cfg_switched"] = dict(cfg)
    t = min(E21_LAUNCH_TIMEOUT_SEC, globals().get("E15B_SOLVE_END_PC", E21_HALF_END_PC) - time.perf_counter() - 5 * 60)
    if t < E21_RELAUNCH_MIN_SEC:
        ENGINE_INFO.setdefault("relaunches", []).append({"skipped": f"only {round(t)} s for a relaunch"})
        return False, "too little time for a relaunch"
    e21_snapshot_counters()
    E21_STATE["relaunch_inflight"] = H           # rev 2k: e21_end_half cancels / aborts only while this is set
    try:
        try:
            if E11_SERVER is not None:
                E11_SERVER.kill()
        except Exception:
            pass
        _e18_kill_gpu_pids()
        _wait_gpu_release()
        if E21_HALF != H or E21_STATE.get("ending") == H:      # rev 2h (GPT v3 audit 2): the half ended meanwhile
            ENGINE_INFO.setdefault("relaunches", []).append({"skipped": "the half ended during the relaunch"})
            return False, "the half ended during the relaunch"
        try:
            srv, info = e21_launch(H, cfg, f"relaunch{len(ENGINE_INFO.setdefault('relaunches', []))}", timeout=t)
            ENGINE_INFO["relaunches"].append({k: info.get(k) for k in ("tag", "ready", "why", "ready_sec", "capacity", "cfg")})
            ENGINE_INFO["cfg"] = cfg
            return True, info.get("why")
        except Exception as e:
            ENGINE_INFO.setdefault("relaunches", []).append({"error": repr(e)[:600]})
            return False, repr(e)[:300]
    finally:
        if E21_STATE.get("relaunch_inflight") == H:
            E21_STATE["relaunch_inflight"] = None


def e21_cfgs(spec):
    """[("primary", cfg), ("fallback", cfg)] of a half spec (the fallback overrides some keys of the primary)."""
    base = {k: v for k, v in spec.items() if k != "fallback"}
    return [("primary", dict(base)), ("fallback", dict(base, **spec["fallback"]))]


def e21_gpus_free(tag):
    """VAL_engine_A1 M3 / VAL_state_B2 L6: never launch on GPUs a previous server still holds (the E18 donor stopped)."""
    if E21_FAKE and os.environ.get("E21_FAKE_GPU_HELD") == "1":            # harness only: GPUs never released
        ENGINE_INFO.setdefault("gpu_checks", []).append({"released": False, "tag": tag, "fake_held": True})
        return False
    rel = _wait_gpu_release()
    if not rel.get("released") and not E21_FAKE:
        _e18_kill_gpu_pids()
        rel = _wait_gpu_release()
    rel["tag"] = tag
    ENGINE_INFO.setdefault("gpu_checks", []).append(rel)
    return bool(rel.get("released")) or bool(E21_FAKE)


def e21_begin_half(H):
    """Half H: its own output folder, a fresh Duck scheduler state, its vLLM server (primary config, else the fallback)."""
    global OUT, E21_HALF, E21_HALF_T0, E21_HALF_END_PC, SGL_URL, ENGINE_READY, CONCURRENCY, ENGINE_INFO, E12_STREAMS
    global E12_VLLM_CONCURRENCY, E15B_SOLVE_END_PC, engine, llm, E12_SOLVE_WINDOW_SEC, E12_RECORDS, E12_RUN
    # rev 2 (VAL_state_B2 LOW-2): the per-half state is reset first, so a later raise can never leave half A's record
    ENGINE_READY, CONCURRENCY, engine, llm = False, 0, None, None
    ENGINE_INFO = {"half": H, "variant": E21_VARIANT, "attempts": [], "backend": None, "kv_tokens": None,
                   "concurrency": None, "fatal": None, "load_sec": None, "cuda_link": CUDA_LINK}
    E12_RECORDS, E12_RUN = {}, {}               # VAL_state_B1 L3 (CELL 6 re-creates both first)
    E21_HALF, E21_HALF_T0 = H, time.perf_counter()
    OUT = BASE_OUT / f"half{H}"
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(E21_HALF_SPECS[E21_VARIANT][H]))
    _a = (E21_STATE["halves"].get("A") or {}).get("engine") or {}
    if H != "A" and E21_B_FOLLOWS_A and _a.get("used") == "fallback":
        # user decision (2026-10-28, option ka): a half-A fallback is repeated in half B, so the two halves differ only in
        # the vLLM version
        spec = dict(spec, **spec["fallback"])
        ENGINE_INFO["spec_note"] = "half A ran its fallback config; half B uses the same config (E21_B_FOLLOWS_A)"
    ENGINE_INFO["spec"] = spec
    _stop = HARD_STOP - E12_STOP_BEFORE_HARD_SEC
    # half A of a two-half commit stops early enough for half B's setup + window; otherwise the notebook's own stop
    E21_HALF_END_PC = _stop if (H != "A" or len(E21_HALVES) == 1) else _stop - E21_HALF_SOLVE_SEC - E21_B_SETUP_RESERVE_SEC
    # rev 2 (VAL timing C1 M2 / C2 M1, state B1 M2, fidelity E1 M1 / E2, engine A1 M7, A2 M4): equal windows. Half B gets
    # half A's achieved window when that was shorter (A's setup ate into it), so 0.19 and 0.32 get the same seconds.
    if E12_SOLVE_WINDOW_SEC and H != "A":
        _wa = ((E21_STATE["halves"].get("A") or {}).get("e12_run") or {}).get("deadline_in_sec")
        if _wa and _a.get("used") and _wa < E21_HALF_SOLVE_SEC:
            E12_SOLVE_WINDOW_SEC = float(_wa)
            ENGINE_INFO["window_note"] = f"half B window = half A's achieved {round(_wa)} s"
    ENGINE_INFO.update(window_sec=E12_SOLVE_WINDOW_SEC, end_pc_in_sec=round(E21_HALF_END_PC - NOTEBOOK_T0),
                       begin_sec=round(E21_HALF_T0 - NOTEBOOK_T0))
    SGL_URL = f"http://127.0.0.1:{E21_PORTS[H]}"
    E15B_SOLVE_END_PC = float("inf")
    E12A.SERVER_DOWN.clear(); E12A.SERVER_WAITS.update(waits=0, sec=0.0); E12A.SERVER_LAST.clear(); E12A.SCHED_LAST.clear()
    # rev 2 (VAL_state_B1 H1, B2 HIGH-1, timing C2 HIGH-1, fidelity E1 HIGH-1 / E2 HIGH-1): Duck's priority gate is a
    # process-wide singleton built once with ARC3_MAX_ACTIVE_STREAMS and a min() clock; a fresh gate per half (half A's
    # stragglers keep their own reference to the old one)
    with _E12_TA._PRIORITY_GATE_LOCK:
        _E12_TA._PRIORITY_GATE = None
    _E12_TA._GATE_STATS.update(enqueued=0, blocked=0, blocked_ms=0, handover_ms=0)
    os.environ["ARC3_REASONING_HISTORY_KEY"] = "reasoning"     # vLLM reads only message["reasoning"] (see CELL 3)
    os.environ["LOCAL_ANALYZER_BASE_URL"] = os.environ["OPENAI_BASE_URL"] = SGL_URL + "/v1"
    E21_STATE.pop("probe_fail_since", None)      # rev 2f (VAL2_cells F3): the outage clock is per half
    E21_STATE["halves"][H] = {"engine": ENGINE_INFO}
    print(f"E21 half {H}: {json.dumps(spec)} | window {E12_SOLVE_WINDOW_SEC} | solve must end by {ENGINE_INFO['end_pc_in_sec']} s",
          flush=True)
    if not E10_SANDBOX_SELFTEST.get("ok", False):
        ENGINE_INFO["fatal"] = "sandbox self-test failed"
        return
    _t = time.perf_counter()
    for tag, cfg in e21_cfgs(spec):
        if not e21_gpus_free(f"before {tag}"):
            ENGINE_INFO["fatal"] = "GPU memory still held before the launch"
            break
        try:
            srv, info = e21_launch(H, cfg, tag)
            ENGINE_INFO["attempts"].append(dict(info))
            ENGINE_INFO.update(cfg=cfg, used=tag, backend=f"vllm-{cfg['vllm']}-{cfg['weights']}-mtp{(SPECULATIVE_CONFIG or {}).get('num_speculative_tokens', 0)}-s{cfg['seqs']}"
                               + (f"-ssm-{cfg['ssm']}" if cfg.get("ssm") else "") + f"-{info['attention_backend']}",
                               kv_tokens=info["capacity"].get("effective_kv_tokens"), capacity=info["capacity"],
                               max_model_len=E21_VLLM_MAX_MODEL_LEN)
            ENGINE_READY = True
            CONCURRENCY = int(info["capacity"]["streams"])
            break
        except Exception as e:
            _info = e.args[1] if len(e.args) > 1 and isinstance(e.args[1], dict) else {}
            ENGINE_INFO["attempts"].append(dict(_info, error=repr(e.args[0] if e.args else e)[:900]))
            print(f"E21 half {H} {tag} FAILED:", repr(e.args[0] if e.args else e)[:900], flush=True)
            if time.perf_counter() > E21_HALF_END_PC - 30 * 60:
                break
    ENGINE_INFO["load_sec"] = round(time.perf_counter() - _t, 1)
    ENGINE_INFO["concurrency"] = int(CONCURRENCY)
    E12_STREAMS = int(CONCURRENCY)
    E12_VLLM_CONCURRENCY = int(CONCURRENCY)      # read by e11_decode_summary (E12-V)
    if H == "A" and "B" in E21_HALVES and any(c.get("vllm") == "0.32.0" for c in E21_HALF_SPECS[E21_VARIANT].values()):
        e21_v032_install_bg()
    if not ENGINE_READY:
        ENGINE_INFO["fatal"] = ENGINE_INFO.get("fatal") or "no vLLM config became ready"
    print(f"E21 half {H} engine:", json.dumps(ENGINE_INFO, default=str)[:3000], flush=True)


def e21_scheduler_threads():
    """Duck scheduler threads still alive (solve_all_r6's worker / probe_loop closures)."""
    n = 0
    for th in threading.enumerate():
        q = getattr(getattr(th, "_target", None), "__qualname__", "") or ""
        if th.is_alive() and q.startswith("solve_all_r6.<locals>."):
            n += 1
    return n


def e21_end_half(H):
    """Stop half H's server and free the GPUs for the next half; keep its records for the E21 report. rev 2: the kill
    always runs (finally: VAL_engine_A1 M5); the last half kills fast and does not wait for the GPUs (nothing follows;
    VAL timing C1 M1 / C2 M2: the hidden rerun's tail); half A's watchdog thread is joined first (VAL_state_B2 L1)."""
    global E11_SERVER
    st = E21_STATE["halves"].setdefault(H, {})
    last = H == E21_HALVES[-1]
    E21_STATE["ending"] = H              # rev 2h: a watchdog relaunch in flight sees this and never launches
    try:
        for th in threading.enumerate():
            if th.name in ("e15b_watchdog", "e12_heartbeat") and th is not threading.current_thread():
                th.join(timeout=30)
                # rev 2k (CPU kit wdhold): a watchdog only blocked in a probe is harmless (it cannot act in half B:
                # the rev 2c half check, the rev 2h relaunch re-check); cancel / abort only a relaunch in flight
                _inflight = th.is_alive() and th.name == "e15b_watchdog" and E21_STATE.get("relaunch_inflight") == H
                if th.is_alive() and th.name == "e15b_watchdog" and not _inflight:
                    st["watchdog_blocked_not_relaunching"] = True
                if _inflight:
                    # rev 2h (GPT v3 audit 2): a relaunch is in flight; kill its server (half B has not begun, so
                    # every server / GPU process is this half's), then join again, bounded
                    st["watchdog_relaunch_cancelled"] = True
                    try:
                        if E11_SERVER is not None:
                            E11_SERVER.kill(fast=True)
                    except Exception:
                        pass
                    _e18_kill_gpu_pids()
                    th.join(timeout=180)
                st[th.name + "_alive_after_join"] = th.is_alive()
                if _inflight and th.is_alive() and not last and E21_STATE.get("relaunch_inflight") == H:
                    E21_STATE["abort_next_half"] = f"half {H} watchdog relaunch still in flight after the handoff joins"
        _recs, _run = globals().get("E12_RECORDS") or {}, globals().get("E12_RUN") or {}
        if st.get("cell6_ran"):
            with (globals().get("_E12_LOCK") or threading.Lock()):
                st["records"] = {t: list(v) for t, v in _recs.items() if v}
            st["e12_run"] = dict(_run)
        _lr = int(globals().get("E21_LENGTH_RESTORED", {}).get("n", 0))
        st["length_restored"] = _lr - sum(int(v.get("length_restored") or 0) for k, v in E21_STATE["halves"].items() if k != H)
        st["gate"] = {"stats": dict(_E12_TA._GATE_STATS), "slots_env": os.environ.get("ARC3_MAX_ACTIVE_STREAMS"),
                      "free_now": getattr(_E12_TA._PRIORITY_GATE, "_free", None)}
        e21_snapshot_counters()
        st["engine"] = ENGINE_INFO
        st["decode_summary"] = e11_decode_summary() if E11_SERVER is not None else None
        _acc = (st["decode_summary"] or {}).get("accept_len_median")
        if _acc is not None and _acc < 1.5:          # an INT4 MTP-head mismatch shows up here first (VAL_engine_A1 M2)
            st["mtp_acceptance_warning"] = _acc
            print(f"E21 WARNING half {H}: MTP mean acceptance {_acc} < 1.5", flush=True)
    except Exception as e:
        st["end_error"] = repr(e)[:300]
    finally:
        try:
            if E11_SERVER is not None:
                E11_SERVER.kill(fast=last)
        except Exception:
            pass
        E11_SERVER = None
        if not last:
            st["gpu_pids_killed"] = _e18_kill_gpu_pids()
            st["gpu_release"] = _wait_gpu_release()
        if not last:     # rev 2d (GPT audit 2): a clean handoff, bounded; counted, never assumed
            _w0 = time.perf_counter()
            while e21_scheduler_threads() and time.perf_counter() - _w0 < 120:
                time.sleep(2)
            st["handoff_scheduler_threads_left"] = e21_scheduler_threads()
            st["handoff_wait_sec"] = round(time.perf_counter() - _w0, 1)
        st["end_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
        print(f"E21 half {H} ended at {st['end_sec']} s | GPU release", json.dumps(st.get("gpu_release")),
              "| scheduler threads left", st.get("handoff_scheduler_threads_left"), flush=True)


BASE_OUT = OUT
E21_STATE = {"halves": {}, "v032_install": None}
E21_STAGED = {}
if E14_STAGING is not None:          # CELL 1b started the copy of this variant's weights (fail open: original paths)
    E14_STATE.enter("STAGING_JOIN")
    E14_STAGING.join(timeout=max(0.0, min(SGL_GIVE_UP_AT - 20 * 60, NOTEBOOK_T0 + 55 * 60) - time.perf_counter()))
    if E14_STAGING.result is None:
        E14_RESULTS["staging"] = {"still_running": True, "note": "launch from the original paths"}
    else:
        _paths, E14_RESULTS["staging"] = E14_STAGING.result
        for _w, _p in (("int4", SGL_MODEL_PATH), ("fp8", MODEL_PATH)):
            if _paths.get(str(_p)) and _paths[str(_p)] != str(_p):
                E21_STAGED[_w] = Path(_paths[str(_p)])
    E14_RESULTS["launch_paths"] = {k: str(v) for k, v in E21_STAGED.items()}
    e14_save()
    print("E14 staging:", json.dumps(E14_RESULTS.get("staging"))[:1200], "| staged", E14_RESULTS["launch_paths"], flush=True)
os.environ.setdefault("MAX_JOBS", "4")              # bound FlashInfer JIT build parallelism (S1-60K)
CUDA_LINK = ensure_libcuda_linkable() if not E21_FAKE else {"probe": "PASS", "fake": True}
CONFIG["cuda_link"] = CUDA_LINK
print("CUDA/FlashInfer linker preflight:", json.dumps(CUDA_LINK), flush=True)
# every launch command this notebook can make (primary + fallback of each half, both attention backends), for the record
# and for the Kaggle CPU kit, which parses each one with the real vLLM 0.19 / 0.32 argument parser
E21_FLAGS = {f"{_H}_{_tag}_{_be}": {"vllm": _cfg["vllm"], "flags": (e21_flags_v019 if _cfg["vllm"] == "0.19.0" else e21_flags_v032)(
                 _cfg, E21_PORTS[_H], _be)}
             for _H, _sp in E21_HALF_SPECS[E21_VARIANT].items() for _tag, _cfg in e21_cfgs(_sp)
             for _be in ("FLASHINFER", "TRITON_ATTN")}
CONFIG["e21"]["flags"] = E21_FLAGS
E14_STATE.enter("LAUNCH", deadline=HARD_STOP)
