"""Experiment 11 Stage 3: E11-04 capacity + real-concurrency validation (Claude, 2026-10-17; rev 2 2026-10-18 after two
GPT audits, every point verified against the SGLang 0.5.21 source first; user "ok"). Deterministic builder. It reuses the
E11-03 cells that ran clean on Kaggle (install, preflight, tokenization, server helpers) verbatim from
build_e11_03_gpu_validation.py and patches only what E11-03 run 1 + the audits showed:

  * E11-03's speed curve was a CAP: SGLang caps max_running_requests by the mamba state cache (5 slots/request with
    extra_buffer + overlap, 4 with extra_buffer_lazy; kv_cache_configurator.py:2232-2330).
  * Auto mem_fraction_static 0.62 left 4.4 GB/GPU free; the 2.6 GB prefill CUDA graph; KV only 387k (need >= 900k).
rev 2 (verified audit findings):
  * ReplaySSM spec + DFLASH requires a KDA model (kv_cache_configurator.py:1125-1136) -> the ReplaySSM arm is removed.
  * sglang:generation_tokens_total only counts FINISHED requests (tokenizer_manager.py:3085 -> metrics_collector.py:1797)
    -> PROD throughput = sum of completion_tokens of every request (finished + aborted partial; abort_all returns a normal
    response, scheduler.py:5569-5584, tokenizer_manager.py:1790-1829) / window; the counter is kept as a lower bound.
  * Capacity gate is fail-closed (slots per request from the flags, unknown -> FAIL); PROD log frozen at the window end;
    long-context proof from the decode log; memory ladder retries only on OOM; marker / contcache fail-closed; no arm
    inherits another arm's result; capacity winner and production candidate are separate; stage order: sweep -> PROD
    -> marker -> contcache (+ long-prefix answer continuation) -> sanity.
  * "no scaling factors provided" is printed whenever --quantization-param-path is unset (load_model_utils.py:118-136);
    the checkpoint's k/v scale VALUES are read and reported; runtime use stays unverified.

Arms (all INT4 + DFlash2 block 8, unquant drafter, TP4, max running 12, prefill CUDA graph OFF, extra_buffer_lazy with
mamba cache 56 = 14 requests, mem 0.83 -> 0.78 only on OOM):
  W1  SSM float16 (E11-03 D, the only arm that really ran 12)       full: greedy, sweep, PROD 45 min, marker, cache, sanity
  W2  SSM bfloat16 (the dtype SGLang's own warning suggests)        full: greedy, sweep, PROD 30 min, marker, cache, sanity
  W3  SSM float32 (numerical reference; its KV will be smaller)     short: greedy + sweep
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
# E11-04 (rev 2) · SGLang 0.5.21 + RedHat Qwen3.8-27B INT4 + DFlash2 · capacity + real-concurrency validation on 4x L4

**Settings:** Accelerator **GPU L4 x4** · Internet **OFF** · no Secrets.
**Attach (same as E11-03):** dataset `foysalemonshanto/sglang-0-5-21-cu13-wheelhouse`, models
`foysalemonshanto/qwen3-8-27b-int4-redhat` (w4a16-g128) and `foysalemonshanto/qwen3-8-27b-dflash2` (bf16),
competition `arc-prize-2026-arc-agi-2`. **Not a submission.**

E11-03 run 1 found that the flat speed curve was a **cap**, not the engine: SGLang limited the running requests through
the mamba (SSM) state cache (C ran 6, B 10, D 13), left 4.4 GB/GPU unused and gave only 387k KV tokens
(production needs about 920k for 12 x 60k). This run answers, at a **real** 12 concurrent requests:
1. Do the memory flags give **12 real running + >= 900k KV**? (from the server log, fail-closed gate)
2. How fast is INT4 + DFlash2 on a **production-like** load (12 always in flight, ARC prompts, up to 60k thinking), and
   at what context lengths? Target >= 332 tok/s (1.3 x vLLM FP8's long-context 255) **with long contexts proven**.
3. SSM state precision: **W1** fp16 · **W2** bf16 (both full) · **W3** fp32 (short numerical reference).
4. Correctness: marker contamination test, prefix-cache continuation (short + long-prefix answer), 8-puzzle sanity.

Expected about 2.7 h; no new stage starts after 4.5 h. Everything is written to `/kaggle/working/e11_04/`.
"""

# ---------------------------------------------------------------- cell 0
C0 = _ns["C0"]
C0 = once(C0, "CELL 0 — configuration, inputs, results file", "CELL 0 — configuration, inputs, results file (E11-04 rev 2)", "c0 title")
C0 = once(C0, '"/kaggle/working/e11_03"', '"/kaggle/working/e11_04"', "out dir")
C0 = once(C0, 'os.environ.get("E11_TIME_BUDGET", 5.0 * 3600)', 'os.environ.get("E11_TIME_BUDGET", 4.5 * 3600)', "time budget")
C0 = once(C0, "MAX_RUNNING = 32\n", "MAX_RUNNING = 12                # E11-04: production slots; SGLang must REALLY run 12\n", "max running")
C0 = once(C0, "SWEEP_LEVELS = [1, 4, 8, 12, 16, 24, 32]   # rev 3: 12 added (D66/L66: 12 slots suit long-thinking Poetiq)",
          "SWEEP_LEVELS = [1, 4, 8, 12]   # E11-04: up to the production 12 (E11-03 measured beyond)", "sweep levels")
C0 = once(C0, "SANITY_TIMEOUT_SEC = 60 * 60\n",
          "SANITY_TIMEOUT_SEC = 60 * 60\n"
          "# ---- E11-04\n"
          "KV_MIN = 900_000                # gate: >= 12 x 75k (S1-60K peaked at 68 % of 1.35M = ~920k on vLLM)\n"
          "MIN_FREE_GB = 0.5               # gate: free memory left after all captures (activations of chunked prefill)\n"
          "PROD_N_UNITS = 48               # distinct ARC prompts cycled through the closed-loop load\n"
          "PROD_CONCURRENCY, PROD_MAX_NEW = 12, 60000      # = S1-60K: 12 slots, 60,000 thinking\n"
          "PROD_SEC = {\"W1\": 45 * 60, \"W2\": 35 * 60, \"W3\": 0}   # closed-loop window per arm (W3: no PROD)\n"
          "PROD_SEC_GATE_FAIL = 10 * 60    # a failed capacity gate still measures, but shorter\n"
          "PROD_DRAIN_SEC = 10 * 60        # after abort_all: give up waiting after this\n"
          "LONG_CTX = 40000                # long-context proof: decode lines at 12 running with mean context >= this\n"
          "LONG_CTX_MIN_LINES = 10\n"
          "if os.environ.get(\"E11_PROD_SEC\"):          # harness only\n"
          "    PROD_SEC = {k: (float(os.environ[\"E11_PROD_SEC\"]) if v else 0) for k, v in PROD_SEC.items()}\n"
          "    PROD_SEC_GATE_FAIL = float(os.environ[\"E11_PROD_SEC\"]) / 2\n"
          "TARGET_TOK_S = 332              # 1.3 x vLLM FP8 + MTP-3 long-context ~255 tok/s\n"
          "MARK_N = 8                      # contamination companions, each told to write its own codename\n"
          "LONG_CONT_N = 2                 # long-prefix answer continuation: the 2 longest PROD outputs ...\n"
          "LONG_CONT_MIN = int(os.environ.get(\"E11_LONG_CONT_MIN\", 20000))   # ... of at least this many tokens\n", "e11-04 constants")
C0 = once(C0, 'RESULTS = {"run_id": RUN_ID, "inputs": {k: str(v) for k, v in INPUTS.items()}, "env": {}, "preflight": None,\n'
              '           "configs": {}, "best_spec": None, "sanity": None, "notes": []}',
          'RESULTS = {"run_id": RUN_ID, "experiment": "E11-04 rev 2", "inputs": {k: str(v) for k, v in INPUTS.items()}, "env": {},\n'
          '           "preflight": None, "configs": {}, "best_spec": None, "best_capacity_arm": None, "sanity": None, "notes": []}\n'
          '# FP8 KV scales: names in the INT4 checkpoint index (values are read in cell 3, inside the venv)\n'
          '_idx = INT4 / "model.safetensors.index.json"\n'
          'try:\n'
          '    _wm = _json(_idx).get("weight_map", {}) if _idx.is_file() else {}\n'
          '    RESULTS["kv_scale_tensors_in_checkpoint"] = sum(1 for k in _wm if k.endswith(("k_scale", "v_scale")))\n'
          'except Exception as _e:\n'
          '    RESULTS["kv_scale_tensors_in_checkpoint"] = f"error {_e!r}"[:200]\n'
          'RESULTS["kv_cache_scheme"] = ((_json(INT4 / "config.json").get("quantization_config") or {}).get("kv_cache_scheme"))',
          "results + kv scale names")

# ---------------------------------------------------------------- cell 1 + 2 verbatim
C1, C2 = _ns["C1"], _ns["C2"]

# ---------------------------------------------------------------- cell 3: more prompt units + marked companions + scale values
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
C3 = once(C3, 'print(json.dumps(RESULTS["prompts"], indent=1)[:1500])', r'''print(json.dumps(RESULTS["prompts"], indent=1)[:1500])

# rev 2: the checkpoint's FP8 k/v scale VALUES (the server's "no scaling factors provided" warning is printed whenever
# --quantization-param-path is unset, load_model_utils.py:118-136, so it says nothing about these). Runtime use: unverified.
SCALE_SRC = r"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); out = {}
try:
    wm = json.load(open(d / "model.safetensors.index.json", encoding="utf-8")).get("weight_map", {})
    names = sorted(k for k in wm if k.endswith(("k_scale", "v_scale")))
    out["n_tensors"] = len(names); out["sample_names"] = names[:4]
    from safetensors import safe_open
    vals, by = [], {}
    for k in names:
        by.setdefault(wm[k], []).append(k)
    for f, ks in by.items():
        with safe_open(str(d / f), framework="pt") as h:
            for k in ks:
                vals += [float(v) for v in h.get_tensor(k).float().flatten().tolist()]
    if vals:
        out.update(n_values=len(vals), min=min(vals), max=max(vals), n_exactly_1=sum(1 for v in vals if v == 1.0),
                   n_nonpositive=sum(1 for v in vals if not v > 0))
except Exception as e:
    out["error"] = repr(e)[:300]
print(json.dumps(out))
"""
_ss = (OUT if HARNESS else Path("/tmp")) / "e11_kv_scales.py"
_ss.write_text(SCALE_SRC, encoding="utf-8")
try:
    _r = subprocess.run([os.environ.get("E11_TOKENIZER_PYTHON", str(VPY)), str(_ss), str(INT4)], env=ENV,
                        capture_output=True, text=True, timeout=900)
    RESULTS["kv_scales_checkpoint"] = json.loads(_r.stdout.strip().splitlines()[-1]) if _r.stdout.strip() else {"error": _r.stderr[-300:]}
except Exception as _e:
    RESULTS["kv_scales_checkpoint"] = {"error": repr(_e)[:300]}
RESULTS["kv_scales_checkpoint"]["runtime_use"] = "unverified (server warning is not evidence either way)"
save_results()
print("checkpoint k/v scales:", RESULTS["kv_scales_checkpoint"])''', "scale values probe")

# ---------------------------------------------------------------- cell 4: helpers + caps + real concurrency
C4 = _ns["C4"]
C4 = once(C4, 'LADDER = [[], ["--attention-backend", "triton"]]       # the 2nd launch only if the 1st fails',
          '# E11-04: memory ladder. The 2nd launch only if the 1st failed with an OOM (rev 2: anything else is permanent).\n'
          'LADDER = [["--mem-fraction-static", "0.83"], ["--mem-fraction-static", "0.78"]]\n'
          'OOM_MARKERS = ("out of memory", "outofmemory", "cuda error: out of memory", "not enough memory",\n'
          '               "insufficient memory", "cudaerrormemoryallocation")', "ladder")
C4 = once(C4, '"max_mamba_cache_size", "internal_states"]',
          '"max_mamba_cache_size", "internal_states", "disable_prefill_cuda_graph", "mamba_radix_cache_strategy"]',
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


def slots_per_request(flags):
    """kv_cache_configurator.py:2232-2261 with the overlap scheduler on: extra_buffer 5, extra_buffer_lazy 4, no_buffer 3."""
    if "--disable-radix-cache" in flags:
        return 1
    s = flags[flags.index("--mamba-radix-cache-strategy") + 1] if "--mamba-radix-cache-strategy" in flags else "extra_buffer"
    return {"extra_buffer_lazy": 4, "no_buffer": 3}.get(s, 5)


def server_caps(text):
    """Effective limits from the server log (not /get_server_info: it still reports the requested 32 / 12)."""
    caps = {}
    m = re.findall(r"max_running_requests is capped to (\d+) by the mamba state cache \(max_mamba_cache_size=(\d+), (\d+) state", text)
    caps["capped_to"] = min(int(a) for a, _, _ in m) if m else None
    caps["state_slots_per_request_logged"] = int(m[0][2]) if m else None
    tp0 = "\n".join(l for l in text.splitlines() if "TP0]" in l) or text   # each TP rank logs the same lines
    kv = re.findall(r"KV Cache is allocated\. dtype: ([\w.]+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB", tp0)
    caps["kv_tokens"] = int(kv[0][1]) if kv else None                 # first TP0 line = target (second = DFlash2 draft)
    caps["kv_gb_target"] = round(float(kv[0][2]) + float(kv[0][3]), 2) if kv else None
    caps["kv_gb_draft"] = round(float(kv[1][2]) + float(kv[1][3]), 2) if len(kv) > 1 else None
    mc = re.findall(r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+)(.*)", tp0)
    caps["mamba_cache_size"] = int(mc[0][0]) if mc else None
    caps["mamba_cache_line"] = mc[0][1][:240] if mc else None
    vb = re.findall(r"Capture target verify CUDA graph begin\..*?bs=\[([\d, ]+)\]", text)
    db = re.findall(r"Capture target decode CUDA graph begin\..*?bs=\[([\d, ]+)\]", text)
    caps["verify_graph_bs"] = _all_ints(vb[0]) if vb else None
    caps["decode_graph_bs"] = _all_ints(db[0]) if db else None
    caps["prefill_graph_captured"] = "Capture target prefill CUDA graph" in text
    am = re.findall(r"avail mem=([\d.]+) GB", tp0)
    caps["avail_mem_gb_last"] = float(am[-1]) if am else None
    caps["fp8_kv_scale_warning"] = "no scaling factors provided" in text
    return caps


def capacity_gate(caps, slots, spec=True):
    """rev 2, fail-closed: effective running = mamba cache // slots per request (and the logged cap); unknown -> FAIL."""
    mc = caps.get("mamba_cache_size")
    inferred = min(MAX_RUNNING, mc // slots) if isinstance(mc, int) and slots else None
    eff = min(inferred, caps["capped_to"]) if (inferred is not None and caps.get("capped_to")) else inferred
    caps["effective_running"] = eff
    free = caps.get("avail_mem_gb_last")
    g = {"running_12": eff is not None and eff >= PROD_CONCURRENCY,
         "kv_min": (caps.get("kv_tokens") or 0) >= KV_MIN,
         "verify_graph_12": (not spec) or (PROD_CONCURRENCY in (caps.get("verify_graph_bs") or [])),
         "no_prefill_graph": caps.get("prefill_graph_captured") is False,
         "headroom": free is not None and free >= MIN_FREE_GB}
    g["pass"] = all(g.values())
    return g


def is_oom(text):
    t = (text or "").lower()
    return any(k in t for k in OOM_MARKERS)


def metric(name):
    """Sum of a Prometheus counter/gauge over its label sets. NOTE: generation_tokens_total counts FINISHED requests only."""
    try:
        txt = str(http("/metrics", timeout=120))
    except Exception:
        return None
    vals = [float(x) for x in re.findall(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([-\d.e+]+)\s*$", txt, re.M)]
    return sum(vals) if vals else None


def run_batch(ids_list, max_new, sampling, seeds, stop_ids=None):''', "caps helpers")
C4 = once(C4, '''        try:
            rec["startup"] = startup_lines(srv.log_text())''', '''        try:
            rec["startup"] = startup_lines(srv.log_text())
            rec["caps"] = server_caps(srv.log_text())                              # E11-04 rev 2: fail-closed gate
            rec["slots_per_request"] = slots_per_request(spec_flags)
            rec["gate"] = capacity_gate(rec["caps"], rec["slots_per_request"], spec=bool(spec_flags))
            stamp(f"{name} caps: effective running {rec['caps']['effective_running']} (mamba cache {rec['caps']['mamba_cache_size']} / "
                  f"{rec['slots_per_request']} slots, logged cap {rec['caps']['capped_to']}) | KV {rec['caps']['kv_tokens']} | verify "
                  f"graph bs {rec['caps']['verify_graph_bs']} | free {rec['caps']['avail_mem_gb_last']} GB | gate "
                  f"{'PASS' if rec['gate']['pass'] else 'FAIL ' + str(rec['gate'])}")
            save_results()''', "caps after launch")
C4 = once(C4, '''            stamp(f"{name} launch {i} FAILED ({why}); log tail:\\n{att['log_tail'][-1500:]}")
            save_results()
            continue''', '''            stamp(f"{name} launch {i} FAILED ({why}); log tail:\\n{att['log_tail'][-1500:]}")
            save_results()
            if not is_oom(att["log_tail"]):                     # rev 2: only a memory failure earns the 0.78 retry
                att["no_retry"] = "not an out-of-memory failure"
                break
            continue''', "ladder only on OOM")
C4 = once(C4, "def run_config(name, spec_flags, ssm_dtype=\"float32\", stages=(\"greedy\", \"sweep\", \"long\")):",
          "def run_config(name, spec_flags, ssm_dtype=\"float32\", stages=(\"greedy\", \"sweep\", \"long\")):\n"
          "    # E11-04: the memory ladder (LADDER) supplies --mem-fraction-static; nothing else in the flags sets it", "run_config doc")
# rev 2b (self-audit): every arm shares the lazy + no-prefill-graph flags, so one rejected flag would sink the whole run.
# A failed launch now gets ONE diagnosed fallback per cause: OOM -> mem 0.78; lazy/extra_buffer rejected -> extra_buffer +
# mamba cache 70 (5 slots -> 14); prefill-graph complaint -> keep the prefill graph. Max 3 launches per arm.
C4 = once(C4, '''    for i, extra in enumerate(LADDER):
        if time_left() < 25 * 60:
            rec["skipped"] = "time budget"
            break
        srv = Server(name, base_flags(ssm_dtype) + spec_flags + extra)
        srv.start(f"try{i}")''', '''    plan, i = [(list(spec_flags), list(LADDER[0]))], 0
    while i < len(plan):
        if time_left() < 25 * 60:
            rec["skipped"] = "time budget"
            break
        cur_flags, extra = plan[i]
        srv = Server(name, base_flags(ssm_dtype) + cur_flags + extra)
        srv.start(f"try{i}")''', "launch plan loop")
C4 = once(C4, '''            if not is_oom(att["log_tail"]):                     # rev 2: only a memory failure earns the 0.78 retry
                att["no_retry"] = "not an out-of-memory failure"
                break
            continue''', '''            nxt = next_launch(cur_flags, extra, att["log_tail"], [a.get("retry_reason") for a in rec["attempts"]])
            if nxt is None or len(plan) >= 3:
                att["no_retry"] = "no diagnosed fallback" if nxt is None else "3 launches used"
                break
            att["retry_reason"] = nxt[2]
            plan.append((nxt[0], nxt[1]))
            i += 1
            continue''', "diagnosed fallback")
C4 = once(C4, '''            rec["slots_per_request"] = slots_per_request(spec_flags)
            rec["gate"] = capacity_gate(rec["caps"], rec["slots_per_request"], spec=bool(spec_flags))''',
          '''            rec["launch_flags"] = cur_flags + extra
            rec["slots_per_request"] = slots_per_request(cur_flags)
            rec["gate"] = capacity_gate(rec["caps"], rec["slots_per_request"], spec=bool(spec_flags))''', "slots from the launched flags")
C4 = once(C4, "def is_oom(text):", r'''def next_launch(flags, extra, tail, used):
    """rev 2b: one diagnosed fallback per cause. Returns (flags, extra, reason) or None. Lazy / prefill causes are read
    from the error lines only (normal startup lines also say "Mamba Cache ...")."""
    t = "\n".join(l for l in (tail or "").splitlines()
                  if any(k in l.lower() for k in ("error", "exception", "assert", "unsupported", "not supported")))[-4000:].lower()
    if is_oom(tail) and extra == LADDER[0] and "oom" not in used:
        return list(flags), list(LADDER[1]), "oom"
    if "extra_buffer_lazy" in flags and ("lazy" in t or "extra_buffer" in t or "mamba" in t) and "lazy" not in used:
        f = list(flags)
        f[f.index("--mamba-radix-cache-strategy") + 1] = "extra_buffer"
        if "--max-mamba-cache-size" in f:
            f[f.index("--max-mamba-cache-size") + 1] = "70"          # 5 slots per request -> 14 >= 12
        return f, list(extra), "lazy"
    if "--disable-prefill-cuda-graph" in flags and "graph" in t and "prefill" not in used:
        return [x for x in flags if x != "--disable-prefill-cuda-graph"], list(extra), "prefill"
    return None


def is_oom(text):''', "next_launch")

# ---------------------------------------------------------------- cell 5: E11-04 stages
C5 = r'''
# ==================================================================== CELL 5 — E11-04 stages: PROD closed loop, marker, contcache, sanity
T10 = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}           # Poetiq solver temperature
CONT_THINK = 6144
LONG_SAMPLES = {}                                                # arm -> [(unit, output_ids)] longest PROD outputs


def prod_cap(u):
    return max(4096, min(PROD_MAX_NEW, CONTEXT_LEN - len(PROMPT[u]) - ANSWER_MAX_TOKENS - CONTEXT_SAFETY))


def _tail_ok(name):
    p = (RESULTS["configs"].get(name) or {}).get("prod") or {}
    return p.get("abort_drained") is not False                  # stages after a stuck PROD tail are skipped


def context_bins(rows, level):
    """Decode rate by mean context per running request (#full token / #running-req) at the given running level."""
    bins = {"<10k": [], "10-20k": [], "20-40k": [], ">=40k": []}
    for r in rows:
        n, ft, g = r.get("#running-req"), r.get("#full token", r.get("#token")), r.get("gen throughput (token/s)")
        if n != level or not ft or g is None:
            continue
        ctx = ft / n
        k = "<10k" if ctx < 10000 else "10-20k" if ctx < 20000 else "20-40k" if ctx < 40000 else ">=40k"
        bins[k].append(g)
    return {k: {"lines": len(v), "gen_tok_s_median": round(statistics.median(v), 1) if v else None} for k, v in bins.items()}


def x_prod(srv, name):
    """Closed loop: PROD_CONCURRENCY requests always in flight (refill on finish), real ARC prompts, thinking up to
    60,000 tokens, stop at </think>. rev 2: tok/s = completion tokens of EVERY request (finished in the window + the
    partial outputs abort_all returns) / window; the decode log is frozen at the window end, before the abort."""
    arm = name.split("_")[0]
    gate = (RESULTS["configs"].get(name) or {}).get("gate") or {}
    base = PROD_SEC.get(arm, 15 * 60)
    seconds = base if gate.get("pass") else min(base, PROD_SEC_GATE_FAIL)
    seconds = min(seconds, max(60, time_left() - 25 * 60))
    off = srv.log_size()
    g0, t0 = metric("sglang:generation_tokens_total"), time.time()
    done, k, futs = [], 0, {}
    ex = cf.ThreadPoolExecutor(max_workers=PROD_CONCURRENCY)

    def submit():
        nonlocal k
        u = SWEEP_UNITS[k % len(SWEEP_UNITS)]
        futs[ex.submit(generate, PROMPT[u], prod_cap(u), T10, SEED + 7000 + k, THINK_CLOSE)] = (u, time.time())
        k += 1

    for _ in range(PROD_CONCURRENCY):
        submit()
    g1 = t1 = frozen = None
    keep = []                                                    # (completion tokens, unit, output_ids) of the longest
    while futs:
        if t1 is not None and time.time() - t1 > PROD_DRAIN_SEC:
            RESULTS["notes"].append(f"{name}: {len(futs)} PROD requests still running {PROD_DRAIN_SEC}s after the abort")
            break
        fin, _ = cf.wait(list(futs), timeout=20, return_when=cf.FIRST_COMPLETED)
        if t1 is None and time.time() - t0 >= seconds:
            g1, t1 = metric("sglang:generation_tokens_total"), time.time()     # window closes here
            frozen = srv.log_text(off)                                          # rev 2: freeze BEFORE abort / drain
            try:
                http("/abort_request", {"abort_all": True}, timeout=120)
            except Exception as e:
                RESULTS["notes"].append(f"{name} abort_all failed: {e!r}"[:200])
        for f in fin:
            u, ts = futs.pop(f)
            try:
                r = f.result()
                m = r.get("meta_info", {})
                ct = int(m.get("completion_tokens", 0))
                fr = str((m.get("finish_reason") or {}).get("type"))
                done.append({"unit": u, "prompt": int(m.get("prompt_tokens", 0)), "gen": ct, "finish": fr,
                             "accept": m.get("spec_accept_length"), "sec": round(time.time() - ts, 1), "in_window": t1 is None})
                ids = r.get("output_ids") or []
                if ct >= LONG_CONT_MIN and ids:                     # rev 2b: prefer the latest (still in the prefix cache):
                    keep.append((t1 is not None, ct, u, list(ids)))  # responses after the window first, then the longest
                    keep = sorted(keep, key=lambda z: (not z[0], -z[1]))[:LONG_CONT_N]
            except Exception as e:
                done.append({"unit": u, "error": f"{type(e).__name__}: {str(e)[:200]}", "in_window": t1 is None})
            if t1 is None:
                submit()
    ex.shutdown(wait=False, cancel_futures=True)
    if t1 is None:
        g1, t1, frozen = metric("sglang:generation_tokens_total"), time.time(), srv.log_text(off)
    LONG_SAMPLES[name] = [(u, ids) for _, _, u, ids in keep]
    rows = decode_rows(frozen or "")
    (OUT / f"decode_rows_{name}_prod.json").write_text(json.dumps(rows), encoding="utf-8")
    full = [r.get("#full token", r.get("#token", 0)) for r in rows]
    run_counts = [r.get("#running-req", 0) for r in rows]
    window = t1 - t0
    toks = sum(d.get("gen", 0) for d in done)                   # finished in window + aborted partial outputs
    at12 = [r.get("gen throughput (token/s)") for r in rows if r.get("#running-req") == PROD_CONCURRENCY
            and r.get("gen throughput (token/s)") is not None]
    bins = context_bins(rows, PROD_CONCURRENCY)
    st = {"window_s": round(window, 1), "requests_started": k, "responses": len(done),
          "tokens_all_responses": toks, "tok_s": round(toks / window, 1) if window > 0 else None,
          "tok_s_counter_lower_bound": round((g1 - g0) / window, 1) if (g0 is not None and g1 is not None and window > 0) else None,
          "decode_rate_at_12_median": round(statistics.median(at12), 1) if at12 else None,
          "decode_rate_at_12_lines": len(at12), "context_bins_at_12": bins,
          "long_context_lines_at_12": bins[">=40k"]["lines"],
          "long_context_verified": bins[">=40k"]["lines"] >= LONG_CTX_MIN_LINES,
          "finished_in_window": sum(1 for d in done if d.get("in_window") and "error" not in d),
          "natural_close_in_window": sum(1 for d in done if d.get("in_window") and d.get("finish") == "stop"),
          "aborted_responses": sum(1 for d in done if d.get("finish") == "abort"),
          "errors": [d["error"] for d in done if "error" in d][:3], "n_errors": sum(1 for d in done if "error" in d),
          "abort_drained": len(futs) == 0 and len(done) == k,
          "decode_log": summarize_decode(rows, level=PROD_CONCURRENCY),
          "share_lines_at_12": round(sum(1 for c in run_counts if c == PROD_CONCURRENCY) / max(1, len(run_counts)), 3),
          "kv_tokens_in_use_max": max(full) if full else None,
          "context_max": max((d["prompt"] + d["gen"] for d in done if "gen" in d), default=None),
          "retract_lines": sum(1 for l in (frozen or "").splitlines() if "retract" in l.lower()),
          "gate_pass": gate.get("pass"), "long_samples": [(u, len(ids)) for u, ids in LONG_SAMPLES[name]],
          "requests": done}
    st["real_concurrency"] = {"requested": PROD_CONCURRENCY, "running_max": st["decode_log"].get("running_max"),
                              "share_at_level": st["share_lines_at_12"],
                              "pass": (st["decode_log"].get("running_max") or 0) >= PROD_CONCURRENCY and st["share_lines_at_12"] >= 0.5}
    st["meets_target"] = bool(st["tok_s"] and st["tok_s"] >= TARGET_TOK_S)
    stamp(f"{name} PROD {st['window_s']}s: {st['tok_s']} tok/s (all responses; counter lower bound {st['tok_s_counter_lower_bound']}; "
          f"log rate at 12 {st['decode_rate_at_12_median']}) | real 12 share {st['share_lines_at_12']} | long-context lines "
          f"{st['long_context_lines_at_12']} ({'PROVEN' if st['long_context_verified'] else 'NOT PROVEN'}) | context max "
          f"{st['context_max']} | retract {st['retract_lines']} | drained {st['abort_drained']} | errors {st['n_errors']}")
    return st


def x_mark(srv, name):
    """Contamination: 4 greedy prompts solo (twice: repeatability) vs inside a batch with MARK_N companions that each write
    their own codename. rev 2: PASS needs every companion to write its OWN code, no errors, and no code anywhere else."""
    if not _tail_ok(name):
        return {"skipped": "PROD tail not drained"}
    solo_a = {u: generate(PROMPT[u], GREEDY_TOKENS, {"temperature": 0.0}) for u in GREEDY_UNITS}
    solo_b = {u: generate(PROMPT[u], GREEDY_TOKENS, {"temperature": 0.0}) for u in GREEDY_UNITS}
    comp = list(MARK_CODES)
    ids = [PROMPT[u] for u in GREEDY_UNITS] + [PROMPT["M:" + u] for u in comp]
    sps = [{"temperature": 0.0}] * len(GREEDY_UNITS) + [T10] * len(comp)
    out = [None] * len(ids)
    with cf.ThreadPoolExecutor(max_workers=len(ids)) as ex:
        futs = {ex.submit(generate, x, GREEDY_TOKENS if i < len(GREEDY_UNITS) else 2 * GREEDY_TOKENS, sps[i], SEED + 950 + i): i
                for i, x in enumerate(ids)}                         # rev 2b: companions get 1,024 tokens to write their code
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
    errors = sum(1 for r in out if not r or "error" in r)
    valid = own == len(comp) and errors == 0 and all(ids_of(out[i]) for i in range(len(GREEDY_UNITS)))
    contamination = bool(leaks or cross)
    return {"solo_repeat": rep, "solo_vs_batch": bat, "greedy_leaks": leaks, "companion_cross": cross,
            "companions_wrote_own_code": own, "companions": len(comp), "errors": errors,
            "contamination": contamination, "valid": valid, "pass": valid and not contamination,
            "note": "solo_repeat < identical = nondeterminism even alone; leaks/cross = contamination (blocker); "
                    "valid=False (a companion did not write its own code, or errors) = test not meaningful = FAIL"}


def x_contcache(srv, name):
    """Prefix cache for Poetiq: (a) short forced-close continuation after 6,144 thinking tokens (warm vs cold);
    (b) rev 2: long-prefix answer continuation from the longest PROD outputs (prompt + >= 20k thinking + stop text)."""
    if not _tail_ok(name):
        return {"skipped": "PROD tail not drained"}
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
    all_valid = len(ok) == len(units) and len(ratio) == len(units) and all(r.get("cached_cold") is not None for r in ok)
    short_pass = all_valid and nonempty and min(ratio) >= 0.9 and (sp or 0) >= 1.5
    longrows = []
    for u, ids in LONG_SAMPLES.get(name, []):
        while ids and ids[-1] in EOS_IDS:
            ids = ids[:-1]
        cont = PROMPT[u] + ids + ([] if THINK_CLOSE[0] in ids else STOP_IDS)
        try:
            a = generate(cont, 1024, SAMPLING, SEED + 13)
            m = a.get("meta_info", {})
            longrows.append({"unit": u, "prefix_tokens": len(cont), "cached": m.get("cached_tokens"),
                             "cached_ratio": round((m.get("cached_tokens") or 0) / max(1, len(cont)), 3),
                             "wall_s": round(a["_wall"], 2), "answer_tokens": m.get("completion_tokens"),
                             "answer_nonempty": bool(str(a.get("text", "")).strip()), "grid_parsed": parse_grid(a.get("text", "")) is not None})
        except Exception as e:
            longrows.append({"unit": u, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    long_pass = (all("error" not in r and r["cached_ratio"] >= 0.9 and r["answer_nonempty"] for r in longrows)
                 if longrows else None)
    return {"rows": rows, "cache_ratio_min": round(min(ratio), 3) if ratio else None, "speedup_warm_vs_cold": sp,
            "phase1_nonempty": nonempty, "all_valid": all_valid, "short_pass": short_pass,
            "long_prefix_rows": longrows, "long_prefix_pass": long_pass,
            "pass": bool(short_pass and long_pass is not False)}


def x_sanity(srv, name):
    if not _tail_ok(name):
        return {"skipped": "PROD tail not drained"}
    return sanity(srv, name)


EXTRA_STAGES.update(prod=x_prod, mark=x_mark, contcache=x_contcache, sanity_x=x_sanity)     # insertion order = run order
print("E11-04 stages ready:", list(EXTRA_STAGES))
'''

# ---------------------------------------------------------------- cell 6: the three arms + selection
C6 = r'''
# ==================================================================== CELL 6 — arms W1 / W2 / W3 + selection
DF = SPEC["C_int4_dflash2"]                                            # DFLASH, unquant drafter, 8 draft tokens
NO_PREFILL_GRAPH = ["--disable-prefill-cuda-graph"]                     # frees ~2.6 GB/GPU (E11-03 D log)
LAZY = ["--mamba-radix-cache-strategy", "extra_buffer_lazy", "--max-mamba-cache-size", "56"]   # 4 slots/req -> 14 >= 12
FULL = ("greedy", "sweep", "prod", "mark", "contcache", "sanity_x")
ARMS = [
    ("W1_lazy_fp16", DF + NO_PREFILL_GRAPH + LAZY, "float16", FULL),
    ("W2_lazy_bf16", DF + NO_PREFILL_GRAPH + LAZY, "bfloat16", FULL),
    ("W3_lazy_fp32", DF + NO_PREFILL_GRAPH + LAZY, "float32", ("greedy", "sweep")),    # numerical reference, short
]
if not PREFLIGHT_OK:
    RESULTS["notes"].append("preflight failed: server stages skipped")
else:
    for _name, _flags, _ssm, _stages in ARMS:
        if time_left() < 30 * 60:
            RESULTS["notes"].append(f"{_name} skipped: time budget")
            continue
        _st = list(_stages)
        if "sanity_x" in _st and _name != "W1_lazy_fp16" and time_left() < 100 * 60:
            _st.remove("sanity_x")                                       # W1's sanity is the must-have one
        run_config(_name, _flags, ssm_dtype=_ssm, stages=tuple(_st))
        _r = RESULTS["configs"].get(_name, {})
        if isinstance(_r.get("sanity_x"), dict) and "hits" in _r["sanity_x"]:
            RESULTS.setdefault("sanity_by_arm", {})[_name] = _r["sanity_x"]
    _ref = (RESULTS["configs"].get("W1_lazy_fp16") or {}).get("greedy_ids")
    for _name in ("W2_lazy_bf16", "W3_lazy_fp32"):
        _got = (RESULTS["configs"].get(_name) or {}).get("greedy_ids")
        if _ref and _got:
            RESULTS["configs"][_name]["greedy_vs_W1"] = prefix_match(_ref, _got)


def capacity_candidate(r):
    """Engine qualifies on capacity + speed: gate, real 12, long contexts proven, target speed, clean tail."""
    p = (r or {}).get("prod") or {}
    return bool(r and r.get("ok") and (r.get("gate") or {}).get("pass") and (p.get("real_concurrency") or {}).get("pass")
                and p.get("long_context_verified") and p.get("meets_target") and p.get("retract_lines") == 0
                and p.get("abort_drained") and not p.get("n_errors"))


def production_candidate(r, name):
    """+ its OWN correctness results (never inherited): marker, prefix cache, sanity >= 6/8."""
    s = (RESULTS.get("sanity_by_arm") or {}).get(name) or {}
    return bool(capacity_candidate(r) and ((r.get("mark") or {}).get("pass") is True)
                and ((r.get("contcache") or {}).get("pass") is True) and s.get("n") == 8 and s.get("hits", 0) >= 6)


_cap = [((r.get("prod") or {}).get("tok_s") or 0, n) for n, r in RESULTS["configs"].items() if capacity_candidate(r)]
_prd = [((r.get("prod") or {}).get("tok_s") or 0, n) for n, r in RESULTS["configs"].items() if production_candidate(r, n)]
RESULTS["best_capacity_arm"] = max(_cap)[1] if _cap else None
RESULTS["best_spec"] = max(_prd)[1] if _prd else None
save_results()
stamp(f"capacity winner: {RESULTS['best_capacity_arm']} {sorted(_cap, reverse=True)} | production candidate: {RESULTS['best_spec']}")
'''

# ---------------------------------------------------------------- cell 7: report + GO/NO-GO
C7 = r'''
# ==================================================================== CELL 7 — report + GO / NO-GO
def _f(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _pf(v):
    return "NOT TESTED" if v is None else ("PASS" if v else "**FAIL**")


L = [f"# E11-04 (rev 2) capacity + real-concurrency validation · run {RUN_ID}", ""]
_pf0 = RESULTS.get("preflight") or {}
L += [f"Driver {_pf0.get('driver')} (CUDA {_pf0.get('driver_cuda')}) · preflight {'PASS' if PREFLIGHT_OK else 'FAIL'}", "",
      f"- FP8 KV: checkpoint k/v scale tensors {RESULTS.get('kv_scale_tensors_in_checkpoint')} · values "
      f"{json.dumps(RESULTS.get('kv_scales_checkpoint'))[:300]} · kv_cache_scheme {json.dumps(RESULTS.get('kv_cache_scheme'))[:160]}", ""]
L += ["## Startup caps (server log; fail-closed gate)", "",
      "| arm | launches | effective running (mamba cache / slots) | logged cap | KV tokens | KV GB target+draft | verify graph bs | "
      "prefill graph | free GB | FP8 scale warning | gate |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for n, r in RESULTS["configs"].items():
    c, g = r.get("caps") or {}, r.get("gate") or {}
    att = r.get("attempts") or []
    L.append(f"| {n} | {[(a['extra_flags'][-1] if a['extra_flags'] else '', a['why'], a.get('retry_reason') or a.get('no_retry', '')) for a in att]} | "
             f"{c.get('effective_running')} ({c.get('mamba_cache_size')} / {r.get('slots_per_request')}) | {c.get('capped_to')} | "
             f"{c.get('kv_tokens')} | {c.get('kv_gb_target')}+{c.get('kv_gb_draft')} | {c.get('verify_graph_bs')} | "
             f"{c.get('prefill_graph_captured')} | {c.get('avail_mem_gb_last')} | {c.get('fp8_kv_scale_warning')} | "
             f"{'PASS' if g.get('pass') else 'FAIL ' + json.dumps(g)} |")
L += ["", "## Sweep (short prompts, 4,096 tokens) with real concurrency", "",
      "| arm | requests | total tok/s | log gen tok/s | accept | real running max | real level? |", "|---|---|---|---|---|---|---|"]
for n, r in RESULTS["configs"].items():
    for c, st in (r.get("sweep") or {}).items():
        if "skipped" in st:
            continue
        rc = st.get("real_concurrency") or {}
        L.append(f"| {n} | {c} | {_f(st.get('agg_tok_s'))} | {_f((st.get('decode_log') or {}).get('gen_tok_s_median'))} | "
                 f"{_f(st.get('accept_len_mean'), 2)} | {rc.get('running_max')} | {'PASS' if rc.get('pass') else '**FAIL**'} |")
L += ["", f"## PROD closed loop ({PROD_CONCURRENCY} in flight, ARC prompts, up to {PROD_MAX_NEW} thinking, T 1.0) · target >= {TARGET_TOK_S} tok/s", "",
      "| arm | window s | **tok/s (all responses)** | counter lower bound | log rate at 12 | real 12 share | long-ctx lines (>=40k at 12) | "
      "context max | retract | drained | errors | meets target |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n, r in RESULTS["configs"].items():
    p = r.get("prod")
    if not p or "window_s" not in p:
        continue
    L.append(f"| {n} | {p['window_s']} | **{_f(p['tok_s'])}** | {_f(p['tok_s_counter_lower_bound'])} | {_f(p['decode_rate_at_12_median'])} | "
             f"{p['share_lines_at_12']} | {p['long_context_lines_at_12']} ({'PROVEN' if p['long_context_verified'] else 'NOT PROVEN'}) | "
             f"{p['context_max']} | {p['retract_lines']} | {p['abort_drained']} | {p['n_errors']} | {p['meets_target']} |")
L += ["", "Decode rate at 12 running by mean context per request (server log):", "",
      "| arm | <10k | 10-20k | 20-40k | >=40k |", "|---|---|---|---|---|"]
for n, r in RESULTS["configs"].items():
    b = (r.get("prod") or {}).get("context_bins_at_12")
    if b:
        L.append(f"| {n} | " + " | ".join(f"{_f(b[k]['gen_tok_s_median'])} ({b[k]['lines']} lines)" for k in ("<10k", "10-20k", "20-40k", ">=40k")) + " |")
L += ["", "## Correctness (each arm's own results only)", ""]
for n, r in RESULTS["configs"].items():
    m = r.get("mark")
    if isinstance(m, dict) and "solo_repeat" in m:
        L.append(f"- **{n} marker:** pass {m.get('pass')} (valid {m.get('valid')}, contamination {m.get('contamination')}) · own codes "
                 f"{m.get('companions_wrote_own_code')}/{m.get('companions')} · leaks {m.get('greedy_leaks')} · cross {m.get('companion_cross')} · "
                 f"solo repeat identical {m['solo_repeat']['identical']}/{len(GREEDY_UNITS)} · solo vs batch mean prefix {m['solo_vs_batch']['mean_prefix']}")
    cc = r.get("contcache")
    if isinstance(cc, dict) and "rows" in cc:
        L.append(f"- **{n} contcache:** pass {cc.get('pass')} · short {cc.get('short_pass')} (min ratio {cc.get('cache_ratio_min')}, warm vs cold "
                 f"{cc.get('speedup_warm_vs_cold')}x) · long-prefix answer {cc.get('long_prefix_pass')} {cc.get('long_prefix_rows')}")
    if r.get("greedy_vs_W1"):
        L.append(f"- **{n} greedy vs W1:** {r['greedy_vs_W1']['identical']}/{len(GREEDY_UNITS)} identical, mean prefix {r['greedy_vs_W1']['mean_prefix']}")
    for key in ("mark", "contcache", "sanity_x"):
        if isinstance(r.get(key), dict) and "skipped" in r[key]:
            L.append(f"- **{n} {key}:** skipped ({r[key]['skipped']})")
    if r.get("error"):
        L.append(f"- **{n} error:** {r['error']}")
for n, s in (RESULTS.get("sanity_by_arm") or {}).items():
    L += ["", f"### Sanity {n}: {s['hits']}/{s['n']} solved · FP8 expectation {s['fp8_expected_hits']}", "",
          "| puzzle | hit | mode | think tokens | FP8 think | accept | sec |", "|---|---|---|---|---|---|---|"]
    for row in s["rows"]:
        L.append(f"| {row['uid']} | {row.get('hit')} | {row.get('mode', row.get('error'))} | {row.get('think_tokens')} | "
                 f"{row.get('fp8_think')} | {_f(row.get('accept_len'), 2)} | {row.get('sec')} |")
L += ["", "## GO / NO-GO (production candidate = every column PASS; NOT TESTED never counts as PASS)", "",
      "| arm | capacity gate | real 12 in PROD | long context proven | PROD >= target | no retraction | tail drained | "
      "no contamination | prefix cache | sanity >= 6/8 | capacity winner? | production candidate? |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n, r in RESULTS["configs"].items():
    p, s = r.get("prod") or {}, (RESULTS.get("sanity_by_arm") or {}).get(n)
    m, cc = r.get("mark"), r.get("contcache")
    cells = [(r.get("gate") or {}).get("pass"), (p.get("real_concurrency") or {}).get("pass") if p else None,
             p.get("long_context_verified") if p else None, p.get("meets_target") if p else None,
             (p.get("retract_lines") == 0) if p else None, p.get("abort_drained") if p else None,
             m.get("pass") if isinstance(m, dict) and "pass" in m else None,
             cc.get("pass") if isinstance(cc, dict) and "pass" in cc else None,
             (s["hits"] >= 6) if s else None]
    L.append(f"| {n} | " + " | ".join(_pf(v) for v in cells) + f" | {'YES' if n == RESULTS.get('best_capacity_arm') else '-'} | "
             f"{'YES' if n == RESULTS.get('best_spec') else '-'} |")
L += ["", f"Total time {(time.time() - T0) / 60:.0f} min · capacity winner: {RESULTS.get('best_capacity_arm')} · production candidate: "
      f"{RESULTS.get('best_spec')} · notes: {RESULTS.get('notes')}",
      "", "Limits: FP8 KV scale runtime use is unverified; no paired vLLM FP8 run on this same workload (planned E11-05)."]
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
