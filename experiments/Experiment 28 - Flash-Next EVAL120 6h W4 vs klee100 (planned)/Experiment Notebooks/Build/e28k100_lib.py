"""e28k100_lib.py: the NEW code of the E28 klee100 notebook (Q27-XHIGH-E28_K100_EVAL120_6H.ipynb), embedded in the notebook as a
sha-pinned lzma + base85 literal. Pure functions only (no notebook globals), unit-tested by e28k100_unit_tests.py.

  I8   DRAFT_HEAD=int8 on the klee100 MTP head (user 2026-11-01; measured on W4 only: NB4T arm I8, +11.4 % at 40k, oracle EXACT).
       The env and the read-back are NB4T's (build_e25_nb4t.py / nb4t_lib.py): DRAFT_HEAD is an input of Merlin's build()
       (tools/merlin/launch.py:319-322 -> '"draft_lm_head_quant":"int8"' in --speculative-config); the read-back is the argv spec,
       the non-default-args spec and the 'Draft lm_head: INT8 weight-only copy ...' lines (nb4t_lib _HEAD_RX / _SPEC_RX, verbatim).
       i8_static_check: the conditions of Merlin's speculator._maybe_quantize_draft_lm_head on THIS checkpoint (CPU, before any
       boot); a definitive miss drops I8 before the first boot (I8_UNSUPPORTED_ON_K100), anything else tries it; a boot that fails
       with the INT8 head is retried at the SAME pin with the BF16 head (user: keep the proven 7.93 GB pin and 21 workers).
  CAP  per-attempt projected-peak blocks (min(K100Q's max-model-len 131,072, 8,192 + budget + 8,192): T1 104k 36 blocks, B2 120k 38
       at F 6): recorded (the notebook admits by 'slots', as K100Q; the engine schedules KV).
  PRE  preemptions: the server counter over the phase + a per-attempt flag (the 15-s sampler's counter increments inside the
       attempt's run: a preemption happened while it ran; vLLM does not name the preempted request). A preemption is a
       recompute, not a failure.
  REP  the E28 run report: score (run report + recomputed from the pass@2 submission), tok/s, preemptions, prefix hit, the
       per-attempt table, the B2 (200k) attempts with their context reached, I8 / Q-defer / boot facts, a markdown table.
"""
import json
import math
import re

LIB_VERSION = "e28k100_lib 1.0 (2026-11-01)"
I8_KEY, I8_VALUE = "DRAFT_HEAD", "int8"
I8_ENV = {I8_KEY: I8_VALUE}
I8_DROP = "I8_UNSUPPORTED_ON_K100"
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory")          # = nb5_lib.OOM_RE
FINISHED_OUTCOMES = ("ACCEPTED", "BUDGET_END")
BUDGET_END_VERDICTS = ("BUDGET_LIMITED", "CONTEXT_EXHAUSTED")


def cdiv(a, b):
    return -(-int(a) // int(b))


def _d(x):
    return x if isinstance(x, dict) else {}


def _l(x):
    return x if isinstance(x, list) else []


# ------------------------------------------------------------------ I8: argv / log read-back (nb4t_lib, verbatim pieces)
def last_value(argv, flag):
    """The value after the LAST occurrence of `flag` in argv (vLLM's argparse: the last one wins), or None. (nb4t_lib)"""
    a = [str(x) for x in (argv or ())]
    idx = [i for i, x in enumerate(a) if x == flag]
    return a[idx[-1] + 1] if idx and idx[-1] + 1 < len(a) else None


def spec_from_argv(argv):
    """The effective --speculative-config of an argv (last occurrence) as a dict; {"error": ...} when unparsable. (nb4t_lib)"""
    v = last_value(argv, "--speculative-config")
    if v is None:
        return None
    try:
        d = json.loads(v)
        return d if isinstance(d, dict) else {"error": "not a dict"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:120]}


_WTAG = re.compile(r"\(Worker_PP(\d+)_TP(\d+)")                                                    # nb4t_lib
_HEAD_RX = re.compile(r"Draft lm_head: (INT8|INT4) weight-only copy of the (\d+)x(\d+) shard \(([\d.]+) MiB")   # nb4t_lib
_SPEC_RX = re.compile(r"'speculative_config': (\{[^{}]*\})")                                       # nb4t_lib


def _literal(s):
    import ast
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def i8_readback(log_text, argv, want):
    """NB4T's I8 read-back (nb4t_lib.launch_check / config_readback, the draft-head part) on one boot: the argv spec's
    draft_lm_head_quant, the non-default-args spec, the 'Draft lm_head:' build lines. want = "int8" (I8 on) or None (BF16 head).
    label: I8_ENGAGED (int8 everywhere + INT8 lines) | I8_SPEC_ONLY (spec int8, no build line seen) | I8_CONTRADICTION (a
    value against the plan) | BF16_HEAD (want None, no quantized head) | BF16_HEAD_CONTRADICTION. Recorded only, never fatal."""
    t = log_text or ""
    lines = t.splitlines()
    sp = spec_from_argv(argv)
    na = next((l for l in lines if "non-default args:" in l), "")
    m = _SPEC_RX.search(na)
    lsp = _literal(m.group(1)) if m else None
    heads = []
    for l in lines:
        h = _HEAD_RX.search(l)
        if h:
            w = _WTAG.search(l)
            heads.append({"mode": h.group(1), "n": int(h.group(2)), "k": int(h.group(3)), "mib": float(h.group(4)),
                          "worker": f"PP{w.group(1)}_TP{w.group(2)}" if w else None})
    modes = sorted({h["mode"] for h in heads})
    any_head = any("Draft lm_head:" in l for l in lines)
    a_q = (sp or {}).get("draft_lm_head_quant") if isinstance(sp, dict) else None
    l_q = (lsp or {}).get("draft_lm_head_quant") if isinstance(lsp, dict) else None
    checks = {"argv_spec": a_q == want, "log_spec": (l_q == want) if lsp is not None else None,
              "head_built": (modes == [str(want).upper()]) if want else (not any_head)}
    contra = []
    if a_q != want:
        contra.append(f"argv draft_lm_head_quant {a_q!r} != {want!r}")
    if lsp is not None and l_q != want:
        contra.append(f"log draft_lm_head_quant {l_q!r} != {want!r}")
    if want and any_head and modes != [str(want).upper()]:
        contra.append(f"draft head built {modes}, expected {str(want).upper()}")
    if not want and any_head:
        contra.append(f"a quantized draft head was built ({modes}) on a BF16-head boot")
    if want:
        label = ("I8_CONTRADICTION" if contra else "I8_ENGAGED" if checks["head_built"] and checks["log_spec"] is not False
                 else "I8_SPEC_ONLY")
    else:
        label = "BF16_HEAD_CONTRADICTION" if contra else "BF16_HEAD"
    return {"label": label, "want": want, "checks": checks, "contradictions": contra, "argv_spec": sp, "log_spec": lsp,
            "draft_head_lines": heads[:8], "head_mib_total": round(sum(h["mib"] for h in heads), 1) if heads else None}


def i8_failure_reason(status, log_text):
    """Why a boot with the INT8 draft head failed (the I8_UNSUPPORTED_ON_K100 lever-drop record): Merlin's refusal
    (speculator._maybe_quantize_draft_lm_head / QuantizedDraftLMHead raise a ValueError naming draft_lm_head_quant), an OOM
    (the INT8 copy adds memory on the PP1 cards: NB4T +~310 MiB per card on W4), or any other failure of that boot."""
    t = log_text or ""
    ref = [ln[:300] for ln in t.splitlines() if "draft_lm_head_quant" in ln and re.search(r"Error|error|raise|not supported|needs", ln)]
    oom = [ln[:240] for ln in t.splitlines() if OOM_RE.search(ln)][:4]
    kind = ("DRAFT_LM_HEAD_QUANT_REFUSED" if ref else "OOM_WITH_INT8_HEAD" if oom else f"BOOT_FAILED_WITH_INT8_HEAD:{status}")
    return {"label": I8_DROP, "kind": kind, "status": status, "refusal_lines": ref[:4], "oom_lines": oom}


def i8_static_check(config, quant_config, merlin_src=None, tp=2):
    """The conditions Merlin's speculator._maybe_quantize_draft_lm_head (vllm/v1/worker/gpu/spec_decode/speculator.py:547-603)
    and QuantizedDraftLMHead (vllm/model_executor/layers/draft_lm_head_quant.py:75-90) put on the draft head, read on THIS
    checkpoint before any boot. config / quant_config = the merged model dir's config.json / quantization_config.json (dicts);
    merlin_src = {relative path: source text} of the installed tree (optional). verdict: SUPPORTED_STATIC | UNSUPPORTED_STATIC
    (a definitive miss: I8 is dropped before the first boot) | INCONCLUSIVE (tried on the GPU; a failed boot drops it)."""
    c, q = _d(config), _d(quant_config)
    tc = _d(c.get("text_config")) or c
    out, bad, unk = {"checks": {}}, [], []
    tie = tc.get("tie_word_embeddings", c.get("tie_word_embeddings"))
    out["checks"]["lm_head_not_tied"] = tie is False
    if tie is True:
        bad.append("tie_word_embeddings: the head is the embedding (not a plain ParallelLMHead)")
    elif tie is None:
        unk.append("tie_word_embeddings unknown")
    V, H = tc.get("vocab_size"), tc.get("hidden_size")
    if isinstance(V, int) and isinstance(H, int):
        vp = cdiv(V, 64) * 64                                    # ParallelLMHead pads to DEFAULT_VOCAB_PADDING_SIZE = 64
        shard = vp // tp
        out["shard"] = [shard, H]
        out["checks"]["marlin_aligned"] = shard % 64 == 0 and H % 128 == 0      # draft_lm_head_quant.py:84 (_GROUP_SIZE 128)
        if not out["checks"]["marlin_aligned"]:
            bad.append(f"shard {shard}x{H} not Marlin-aligned (vocab shard % 64, hidden % 128)")
    else:
        unk.append("vocab_size / hidden_size unknown")
    blk = q.get("block_name_to_quantize")
    blks = [blk] if isinstance(blk, str) else (blk if isinstance(blk, list) else [])
    extra = q.get("extra_config")
    if isinstance(extra, str):
        try:
            import ast
            extra = ast.literal_eval(extra)
        except Exception:
            extra = None
    keys = list(extra) if isinstance(extra, dict) else []
    head_keys = [k for k in keys if k == "lm_head" or k.endswith(".lm_head") or "lm_head" in k]
    out["checks"]["lm_head_outside_quantized_blocks"] = bool(blks) and not any("lm_head".startswith(b) or b.endswith("lm_head") for b in blks)
    out["lm_head_extra_config_keys"] = head_keys[:4]
    if head_keys:
        unk.append(f"extra_config names the lm_head ({head_keys[:2]}): the target head is built without quant_config anyway")
    src = merlin_src or {}
    if src:
        mtp = src.get("vllm/models/qwen4_exp/nvidia/mtp.py") or ""
        mdl = src.get("vllm/models/qwen4_exp/nvidia/model.py") or ""
        spc = src.get("vllm/v1/worker/gpu/spec_decode/speculator.py") or ""
        out["checks"]["mtp_supports_draft_lm_head_quant"] = "supports_draft_lm_head_quant = True" in mtp
        out["checks"]["target_head_plain_parallel_lm_head"] = bool(re.search(r"self\.lm_head = ParallelLMHead\(\s*config\.vocab_size,\s*"
                                                                            r"config\.hidden_size,\s*prefix=", mdl))
        out["checks"]["speculator_has_quantize_hook"] = "_maybe_quantize_draft_lm_head" in spc
        for k_ in ("mtp_supports_draft_lm_head_quant", "speculator_has_quantize_hook"):
            if not out["checks"][k_]:
                bad.append(f"Merlin tree: {k_} False")
        if not out["checks"]["target_head_plain_parallel_lm_head"]:
            unk.append("target lm_head construction not recognised in model.py")
    else:
        unk.append("Merlin source not read")
    out["verdict"] = "UNSUPPORTED_STATIC" if bad else ("INCONCLUSIVE" if unk else "SUPPORTED_STATIC")
    out["why"] = bad or unk or ["every static condition holds"]
    return out


# ------------------------------------------------------------------ CAP: per-attempt projected peak (projected-peak admission)
def attempt_peak(budget, ctx_ceiling, prompt_allow=8192, tool_allow=8192):
    """C-9 for one attempt: min(context ceiling, prompt_allow + budget + tool_allow) tokens (nb5_lib.capacity_plan's rule)."""
    return int(min(int(ctx_ceiling), int(prompt_allow) + int(budget) + int(tool_allow)))


def attempt_blocks(budget, ctx_ceiling, block, f_eff, prompt_allow=8192, tool_allow=8192):
    return cdiv(attempt_peak(budget, ctx_ceiling, prompt_allow, tool_allow), block) + int(f_eff)


def capacity_note(pool_blocks, block, f_eff, gate, t1, b2, t1_ctx, b2_ctx, workers):
    """Recorded only (the notebook admits by slots): what a projected-peak gate WOULD admit on this boot, T1 attempts alone
    and with k B2 attempts in flight. Under slots vLLM can preempt (a recompute; counted)."""
    cap = int(math.floor(float(gate) * int(pool_blocks)))
    bt, bb = attempt_blocks(t1, t1_ctx, block, f_eff), attempt_blocks(b2, b2_ctx, block, f_eff)
    mix = {str(k): max(0, (cap - k * bb) // bt) for k in range(0, 4) if cap - k * bb >= 0}
    return {"capacity_blocks": cap, "t1_blocks": bt, "b2_blocks": bb, "t1_streams": cap // bt, "b2_alone_streams": cap // bb,
            "t1_streams_with_k_b2": mix, "workers": int(workers),
            "note": "recorded only: what a projected-peak gate would admit. The notebook admits by slots (the R2 rule, 21 "
                    "workers); vLLM schedules KV and CAN preempt (a recompute): preemptions are counted per phase and flagged per "
                    "attempt. (A projected gate would wait instead when the projected "
                    "peaks of the attempts in flight would exceed the pool.)"}


# ------------------------------------------------------------------ REP: the E28 run report
def fail_kind(text):
    """= k100q_report.fail_kind (the report's failure classes)."""
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
        return "NO_RESULT"
    return "OTHER"


def attempt_end(r, budget, window_end_epoch=None, margin_sec=90.0):
    """= k100q_report.attempt_end: (outcome, end_reason) ACCEPTED | BUDGET_END | INCOMPLETE_WINDOW | FAILED_OR_INTERRUPTED."""
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
    if _d(r.get("b_events")).get("tiny_request_skipped"):
        return "BUDGET_END", "commit_done_tiny_rest"
    if bud and tok >= bud - 1024:
        return "BUDGET_END", "budget_spent"
    if r.get("status") == "paused":
        return "INCOMPLETE_WINDOW", "paused_at_window_end"
    if window_end_epoch and r.get("t1") and float(r["t1"]) >= float(window_end_epoch) - margin_sec:
        return "INCOMPLETE_WINDOW", "running_at_window_end"
    return "FAILED_OR_INTERRUPTED", "no_terminal_evidence"


def _train(r):
    b = _d(_d(r.get("progress")).get("best"))
    k, n = b.get("k"), b.get("n")
    return {"k": k, "n": n, "acc": b.get("acc"), "full": bool(isinstance(k, int) and isinstance(n, int) and n and k == n)}


def _exact(r, gold):
    """Every test output of an ACCEPTED submission equals the gold (None without gold)."""
    if not gold:
        return None
    for s in _l(r.get("submissions")):
        if isinstance(s, dict) and s.get("accepted") and len(_l(s.get("test_outputs"))) == len(gold) \
                and all(a == b for a, b in zip(s["test_outputs"], gold)):
            return True
    return False


def attempt_rows(records, solutions, budgets, window_end_epoch=None, window_start_epoch=None, phase_of=None):
    """One row per attempt record: task, branch, E28 phase (phase_of(record) -> A | B1 | B2 | C | None), budget, generated
    tokens, outcome / end reason, best train match k / n, accepted, exact (accepted submission == gold), t0 / t1 offsets."""
    rows = []
    sol = _d(solutions)
    for t, rs in sorted(_d(records).items()):
        for r in sorted(_l(rs), key=lambda x: (int(x.get("branch") or 0), float(x.get("t0") or 0))):
            if not isinstance(r, dict):
                continue
            ph = None
            try:
                ph = phase_of(r) if phase_of else r.get("e28_phase")
            except Exception:
                ph = None
            b = int(r.get("b_budget") or _d(budgets).get(ph) or _d(budgets).get("T1") or 0)
            oc, why = attempt_end(r, b, window_end_epoch)
            tr = _train(r)
            off = (lambda x: round(float(x) - float(window_start_epoch), 1) if x and window_start_epoch else None)
            rows.append({"task": t, "branch": r.get("branch"), "phase": ph, "budget": b, "tokens": int(r.get("generated_tokens") or 0),
                         "outcome": oc, "end_reason": why, "verdict": r.get("nb5_verdict"), "train_k": tr["k"], "train_n": tr["n"],
                         "train_full": tr["full"], "accepted": bool(r.get("accepted")), "exact": _exact(r, sol.get(t)),
                         "start_sec": off(r.get("t0")), "end_sec": off(r.get("t1")), "note": str(r.get("note") or "")[:160]})
    return rows


def score_from_generated(generated_attempts, solutions, task_ids):
    """The pass@2 score of the written submission (= the E12 run report's rule: gold sets the query count)."""
    sol, gen = _d(solutions), _d(generated_attempts)
    hits = pass1 = pass2 = total = 0
    tn = 0.0
    tids = [t for t in task_ids if _l(sol.get(t))]
    per = {}
    for t in tids:
        golds, g_t = _l(sol.get(t)), _l(gen.get(t))
        th = 0
        for qi, gold in enumerate(golds):
            att = g_t[qi] if qi < len(g_t) and isinstance(g_t[qi], dict) else {}
            a, b = att.get("attempt_1") == gold, att.get("attempt_2") == gold
            hits += a or b
            pass1 += a
            pass2 += b
            total += 1
            th += a or b
        tn += th / max(1, len(golds))
        per[t] = {"queries": len(golds), "hits": th, "solved": th == len(golds)}
    return {"eval_query_hits": hits, "eval_query_total": total, "eval_pass1_hits": pass1, "eval_pass2_hits": pass2,
            "eval_task_norm_pct": round(100 * tn / max(1, len(tids)), 2), "tasks_scored": len(tids),
            "tasks_solved": sum(1 for v in per.values() if v["solved"])}, per


def preempt_events(sampler_rows):
    """[(t_prev, t, delta)] of the sampler's cumulative vllm:num_preemptions_total (a counter reset = a relaunch: ignored)."""
    ev, last = [], None
    for s in sorted([r for r in _l(sampler_rows) if isinstance(r, dict) and isinstance(r.get("preempt"), (int, float))
                     and isinstance(r.get("t"), (int, float))], key=lambda r: r["t"]):
        if last is not None and s["preempt"] > last["preempt"]:
            ev.append((float(last["t"]), float(s["t"]), int(s["preempt"] - last["preempt"])))
        last = s
    return ev


def flag_preemptions(rows, records, events):
    """Per attempt row: preempt_while_running = the preemptions counted in sampler intervals that overlap [t0, t1] of the
    attempt (an upper bound of its own: vLLM does not name the preempted request); None without sampler data."""
    tt = {}
    for t_, rs in _d(records).items():
        for r in _l(rs):
            if isinstance(r, dict):
                tt[(t_, r.get("branch"))] = (r.get("t0"), r.get("t1"))
    for row in rows:
        a, b = tt.get((row["task"], row["branch"]), (None, None))
        if events is None or not a or not b:
            row["preempt_while_running"] = None
            continue
        row["preempt_while_running"] = sum(d for t0_, t1_, d in events if t1_ >= float(a) and t0_ <= float(b))
    return rows


def b2_rows(rows, b2_budget, t1_budget=None):
    """The B2 (200k) attempts: context reached (= generated tokens), verdict, outcome (200k-context correctness record)."""
    return [{k: r.get(k) for k in ("task", "branch", "tokens", "outcome", "end_reason", "verdict", "train_k", "train_n", "exact",
                                   "preempt_while_running")}
            | {"budget": b2_budget, "reached_above_t1": bool(t1_budget) and r.get("tokens", 0) > int(t1_budget)}
            for r in rows if r.get("phase") == "B2"]


def build_e28_report(*, phase, per_config, run_report, generated_attempts, records, solutions, task_ids, boot, i8, qdefer,
                     sched=None, window=None, budgets=None, phase_of=None, rerun=False, max_model_len=None, sampler=None):
    """The E28 report (never raises: errors go to report["errors"]). status: SKIPPED_RERUN | NO_PHASE_RUN | INCOMPLETE_PHASE |
    MISSING_RUN_REPORT | INCONCLUSIVE_SCORE_MISMATCH | VALID."""
    errs = []
    ph, pc, rr = _d(phase), _d(per_config), _d(run_report)
    out = {"lib": LIB_VERSION, "max_model_len": max_model_len, "budgets": budgets, "i8": i8, "qdefer": qdefer, "errors": errs}
    sc = _d(sched)
    e28 = _d(sc.get("e28"))
    out["scheduler"] = {"rec": sc.get("rec"), "present": bool(e28)}
    if e28:      # arc2_duck.SCHED_LAST["e28"] (E28 SCHED: phase counts, timeline, slots, per-attempt rows); E15B's b_summary
        # counts b2 as "attempt 3+" with the b1 budget, so the phase view comes from here and from the records' own b_budget
        out["scheduler"].update({k: e28.get(k) for k in ("enabled", "config", "warnings", "started", "done", "queued", "left",
                                                          "first_start_sec", "last_start_sec", "queue_empty_sec", "order_holds",
                                                          "a_class", "b_class", "setup_failed", "ignored_args",
                                                          "backfill_starts", "forced_admits", "fit_checks", "fit_fails",
                                                          "fit_errors", "admission_checks", "idle")})
        out["sched_attempts"] = _l(e28.get("attempts"))
        out["sched_slots"] = e28.get("slots")
    try:
        b = _d(boot)
        out["boot"] = {k: b.get(k) for k in ("tag", "status", "pin_used", "ready_sec", "t0_sec", "levers_dropped")}
        out["boot_tries"] = [{k: t_.get(k) for k in ("name", "status", "pin_gib", "kv_bytes", "max_num_seqs_launch", "levers", "ready_sec")}
                             | {"ladder_failed": _d(t_.get("ladder")).get("failed"), "i8": t_.get("e28_i8"),
                                "i8_readback": _d(t_.get("i8_readback")).get("label"),
                                "k100_gate_fatal": _d(t_.get("k100_gates")).get("fatal")} for t_ in _l(b.get("tries"))]
    except Exception as e:
        errs.append(f"boot: {type(e).__name__}: {e}"[:300])
    if rerun:
        out["status"] = "SKIPPED_RERUN"
        return out
    if not ph:
        out["status"] = "NO_PHASE_RUN"
        return out
    w = dict(_d(pc.get("window")), **_d(window))
    ws, we = w.get("start_epoch"), w.get("end_epoch")
    try:
        c, sp, cl = _d(ph.get("counters")), _d(pc.get("speed")), _d(pc.get("capacity_levers"))
        pt, phit = c.get("vllm:prompt_tokens_total"), c.get("vllm:prefix_cache_hits_total")
        out["engine"] = {"solver_tok_s": _d(ph.get("solver_window")).get("tok_s"), "phase_tok_s": _d(ph.get("phase_window")).get("tok_s"),
                         "steady_mixed_p50": _d(ph.get("steady_mixed")).get("gen_tok_s_p50"),
                         "server_running_mean": _d(ph.get("b_summary")).get("server_running_mean"),
                         "max_running": sp.get("max_running"), "preemptions": c.get("vllm:num_preemptions_total"),
                         "prefix_hit_pct": c.get("prefix_hit_pct"), "prompt_tokens": pt, "prefix_hit_tokens": phit,
                         "fresh_prefill_tokens": int(pt - phit) if isinstance(pt, (int, float)) and isinstance(phit, (int, float)) else None,
                         "generated_tokens": c.get("vllm:generation_tokens_total"), "pool_blocks": cl.get("pool_blocks"),
                         "f_eff": cl.get("f_eff"), "workers": cl.get("workers"), "pin_gib": cl.get("pin_gib"), "kv_bytes": cl.get("kv_bytes"),
                         "window": w, "actual_window_sec": w.get("actual_sec"), "planned_window_sec": w.get("planned_sec"),
                         "compare_note": "compare runs on actual_window_sec (a late boot shortens the planned 21,600 s window)"}
    except Exception as e:
        errs.append(f"engine: {type(e).__name__}: {e}"[:300])
    try:
        rows = attempt_rows(records, solutions, budgets, we, ws, phase_of)
        ev = preempt_events(sampler) if sampler is not None else None
        flag_preemptions(rows, records, ev)
        out.setdefault("engine", {}).update(preemption_intervals=len(ev) if ev is not None else None,
                                            preemptions_sampler=sum(d for _, _, d in ev) if ev is not None else None,
                                            preemption_note="a preemption = a recompute (speed only), not a failure")
        out["attempts"] = rows
        out["attempt_summary"] = {"attempts": len(rows), "by_phase": {}, "by_outcome": {}}
        for r in rows:
            k = str(r.get("phase"))
            s = out["attempt_summary"]["by_phase"].setdefault(k, {"n": 0, "accepted": 0, "exact": 0, "train_full": 0, "tokens": 0})
            s["n"] += 1
            s["accepted"] += int(r["accepted"])
            s["exact"] += int(r["exact"] is True)
            s["train_full"] += int(r["train_full"])
            s["tokens"] += r["tokens"]
            out["attempt_summary"]["by_outcome"][r["outcome"]] = out["attempt_summary"]["by_outcome"].get(r["outcome"], 0) + 1
        out["attempts_with_preemption"] = sum(1 for r in rows if r.get("preempt_while_running"))
        out["accepted_not_exact"] = [{k: r[k] for k in ("task", "branch", "phase", "tokens")} for r in rows
                                     if r["accepted"] and r["exact"] is False]
        out["b2"] = b2_rows(rows, _d(budgets).get("B2"), _d(budgets).get("T1"))
    except Exception as e:
        errs.append(f"attempts: {type(e).__name__}: {e}"[:300])
    try:
        g, per = score_from_generated(generated_attempts, solutions, list(task_ids or []))
        r_score = {k: rr.get(k) for k in ("eval_query_hits", "eval_query_total", "eval_pass1_hits", "eval_pass2_hits",
                                          "eval_task_norm_pct")} if rr.get("eval_query_total") is not None else None
        agree = None if r_score is None else all(r_score[k] == g[k] for k in r_score)
        out["score"] = {"run_report": r_score, "generated": g, "agree": agree}
        out["per_task"] = per
        out["missing_gold"] = [t for t in (task_ids or []) if not _l(_d(solutions).get(t))]
    except Exception as e:
        errs.append(f"score: {type(e).__name__}: {e}"[:300])
    if ph.get("status") != "DONE" or ph.get("solver_error") or ph.get("error"):
        out["status"] = "INCOMPLETE_PHASE"
        out["phase_error"] = str(ph.get("solver_error") or ph.get("error") or ph.get("status"))[-2000:]
    elif not rr:
        out["status"] = "MISSING_RUN_REPORT"
    elif _d(out.get("score")).get("agree") is False:
        out["status"] = "INCONCLUSIVE_SCORE_MISMATCH"
    elif errs:
        out["status"] = "INVALID_REPORT"
    else:
        out["status"] = "VALID"
    return out


def markdown(report):
    """The per-attempt table (task | phase | branch | tokens | outcome | train | exact) + the summary lines."""
    rep = _d(report)
    e, s = _d(rep.get("engine")), _d(_d(rep.get("score")).get("generated"))
    out = [f"E28 K100 EVAL120: status {rep.get('status')} | score {s.get('eval_query_hits')}/{s.get('eval_query_total')} queries "
           f"(pass1 {s.get('eval_pass1_hits')}, pass2 {s.get('eval_pass2_hits')}), task-norm {s.get('eval_task_norm_pct')} %, "
           f"solved {s.get('tasks_solved')}/{s.get('tasks_scored')}",
           f"actual window {e.get('actual_window_sec')} s (planned {e.get('planned_window_sec')}; compare runs on the actual) | "
           f"solver tok/s {e.get('solver_tok_s')} | steady p50 {e.get('steady_mixed_p50')} | running mean {e.get('server_running_mean')} "
           f"| max running {e.get('max_running')} | preemptions {e.get('preemptions')} | prefix hit {e.get('prefix_hit_pct')} % | "
           f"workers {e.get('workers')} | pool {e.get('pool_blocks')} | F {e.get('f_eff')} | attempts with a preemption while "
           f"running {rep.get('attempts_with_preemption')} (a recompute, not a failure)",
           f"I8: {json.dumps({k: _d(rep.get('i8')).get(k) for k in ('want', 'on', 'static', 'dropped', 'readback')}, default=str)[:600]}",
           "", "| task | phase | b | tokens | outcome | train | accepted | exact |", "|---|---|---|---|---|---|---|---|"]
    for r in _l(rep.get("attempts")):
        tr = f"{r.get('train_k')}/{r.get('train_n')}" if r.get("train_n") else "-"
        ex = {True: "✓", False: "✗", None: "?"}[r.get("exact")]
        out.append(f"| {r['task']} | {r.get('phase')} | {r.get('branch')} | {r['tokens']:,} | {r['outcome']} | {tr} | "
                   f"{'yes' if r['accepted'] else 'no'} | {ex} |")
    return "\n".join(out) + "\n"
