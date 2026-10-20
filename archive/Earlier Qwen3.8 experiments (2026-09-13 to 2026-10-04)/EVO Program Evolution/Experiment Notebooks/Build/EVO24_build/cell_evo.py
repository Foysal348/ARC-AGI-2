# =====================================================================================
# CELL 8 — EVO: EVOLUTION OVER VERIFIED PROGRAMS
#   rule -> solution() -> fresh verification on every training pair -> error map -> mutate
# (Imbue-style code evolution, PiVOTools/Agentica-style verification; replaces the single long REPL chat)
# =====================================================================================
import concurrent.futures as _evo_futures
import zlib as _evo_zlib

EVO_CONFIG = {
    "organisms": 4,               # independent initial candidates per task
    "parents": 2,                 # distinct parents carried into each repair generation
    "repair_generations": 2,
    "reasoning_effort": "medium",
    "think_budget": 3072,         # phase 1: thinking, stops at </think> or this many tokens
    "answer_max_tokens": 2048,    # phase 2: rule + code
    "task_token_cap": 32768,      # aggregate generated tokens per task (all calls)
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "max_grid_cells_shown": 400,  # predicted grids shown in repair feedback only up to this size
    "verify_workers": 8,
}
EVO_CALL_MAX = EVO_CONFIG["think_budget"] + EVO_CONFIG["answer_max_tokens"]
EVO_SYSTEM = ("ARC-EVO v1. You are an expert at solving ARC-AGI abstraction-and-reasoning puzzles by writing short, "
              "correct Python programs.")
EVO_LENSES = [
    "",
    "Hint for this attempt: think in terms of objects (connected components): their shapes, colors, sizes, positions, "
    "and how each one moves, changes or interacts.",
    "Hint for this attempt: think in terms of geometry and layout: rows, columns, symmetry, rotation or reflection, "
    "tiling, cropping, scaling, lines and borders.",
    "Hint for this attempt: think in terms of colors and counts: which colors map to which, what is counted, and what "
    "is drawn where because of it.",
]
EVO_FORMAT = (
    "Answer in exactly this format:\n"
    "## Rule\n<the transformation in a few plain sentences>\n"
    "## Code\n```python\ndef solution(grid):\n    ...\n```\n\n"
    "Code rules: `grid` is a list of lists of ints 0-9; return a NEW list of lists of ints 0-9. The standard library "
    "and numpy may be used. These helpers are preloaded and may be called inside solution (do not redefine them): "
    "`components(grid, color=None, diag=False, background=None)` -> list of dicts {color, cells, bbox=(r0,c0,r1,c1), size}; "
    "`bbox(grid, ignore=None)` -> (r0,c0,r1,c1) or None; `crop(grid, r0, c0, r1, c1)` (inclusive); `rot90(grid)` "
    "(clockwise); `flip_h(grid)`; `flip_v(grid)`; `transpose(grid)`; `color_counts(grid)` -> dict; "
    "`background_color(grid)` (most common colour: a heuristic, pass background= explicitly when unsure); "
    "`grid_shape(grid)` -> (rows, cols); `diff_cells(a, b)`. No printing, no files, no network. Everything else the "
    "program uses must be defined in the code block. It must work for every training input and the test input(s)."
)

def _evo_grid_text(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)

def evo_initial_prompt(task, organism):
    lens = EVO_LENSES[organism % len(EVO_LENSES)]
    return (puzzle_to_source_text(task_without_test_solutions(task)) + "\n\n"
            "Find the transformation that maps every training input to its output; the same transformation must "
            "produce the test output(s).\n" + (lens + "\n" if lens else "") + "\n" + EVO_FORMAT)

def _evo_failure_grids(verifier, task, limit_pairs=2):
    out = []
    by_index = {int(x.get("index", i)): x for i, x in enumerate(list((verifier or {}).get("train") or []))}
    for i, ex in enumerate(task["train"]):
        e = by_index.get(i) or {}
        pred = e.get("pred")
        if e.get("exact") or pred is None:
            continue
        exp = ex["output"]
        if len(pred) * len(pred[0]) > EVO_CONFIG["max_grid_cells_shown"]:
            continue
        same = len(pred) == len(exp) and all(len(r) == len(exp[0]) for r in pred)
        if same:
            rows = [" ".join((str(p) if p == q else f"{p}*") for p, q in zip(pr, er)) for pr, er in zip(pred, exp)]
            out.append(f"Your output for training example {i} (cells marked * are wrong):\n" + "\n".join(rows))
        else:
            out.append(f"Your output for training example {i} (wrong size {len(pred)}x{len(pred[0])}, expected "
                       f"{len(exp)}x{len(exp[0])}):\n" + _evo_grid_text(pred))
        if len(out) >= limit_pairs:
            break
    return "\n\n".join(out)

def evo_repair_prompt(task, parent, with_grids=True):
    report = _v3_report_text(parent["verifier"], task) if parent.get("verifier") else "The previous code could not be run."
    grids = _evo_failure_grids(parent.get("verifier"), task) if with_grids else ""
    return (puzzle_to_source_text(task_without_test_solutions(task)) + "\n\n"
            "A previous attempt at this puzzle:\n## Rule\n" + (parent.get("rule") or "(no rule given)") + "\n## Code\n"
            "```python\n" + strip_library(parent.get("code") or "") + "\n```\n\n"
            "Result of running this code on every training example in a fresh process:\n" + report +
            ("\n\n" + grids if grids else "") + "\n\n"
            "Improve it. Fix the code if it does not implement the rule correctly; if the rule itself is wrong, replace "
            "it with a better rule that explains ALL training examples.\n\n" + EVO_FORMAT)

def render_evo_chat(user_text):
    msgs = [{"role": "system", "content": EVO_SYSTEM}, {"role": "user", "content": user_text}]
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True,
              reasoning_effort=EVO_CONFIG["reasoning_effort"])
    return tokenizer.apply_chat_template(msgs, **kw)

_EVO_THINK_END_ID = tokenizer.encode("</think>", add_special_tokens=False)
assert len(_EVO_THINK_END_ID) == 1, _EVO_THINK_END_ID
_EVO_THINK_END_ID = _EVO_THINK_END_ID[0]
_EVO_FORCE_IDS = list(tokenizer.encode(STOP_THINKING_TEXT, add_special_tokens=False))
_EVO_AFTER_THINK_IDS = list(tokenizer.encode("\n\n", add_special_tokens=False))
_EVO_RULE_RE = re.compile(r"##\s*Rule\s*\n(.*?)(?:\n##\s*Code|```)", re.S | re.I)
_EVO_EOS_IDS = {e[0] for e in (tokenizer.encode(x, add_special_tokens=False) for x in ("<|im_end|>", "<|endoftext|>")) if len(e) == 1}
assert _EVO_EOS_IDS, "EOS tokens must encode to single ids"

def evo_alternative_prompt(task, exact):
    """Second pass@2 slot: a candidate already fits every training pair; ask for a genuinely different reading."""
    return (puzzle_to_source_text(task_without_test_solutions(task)) + "\n\n"
            "This program reproduces EVERY training example exactly:\n## Rule\n" + (exact.get("rule") or "(no rule given)") +
            "\n## Code\n```python\n" + strip_library(exact.get("code") or "") + "\n```\n\n"
            "Its output for the test input(s) is not known to be correct. Find a genuinely different transformation that "
            "ALSO fits every training example but could produce a different test output (a different reading of the "
            "examples). Do not merely rewrite the same idea.\n\n" + EVO_FORMAT)

def evo_fresh_prompt(task, k):
    """Backfill when a task has fewer usable parents than slots: a fresh attempt with a rotated lens."""
    return (evo_initial_prompt(task, k) + "\n\nEarlier attempts at this puzzle did not produce a usable program. Try a "
            "substantially different hypothesis.")

def _evo_seed(tid, round_no, slot):
    return int(_evo_zlib.crc32(f"{SEED}|{tid}|{round_no}|{slot}".encode()) & 0x7FFFFFFF)

def _evo_sp(max_tokens, seed, stop_ids=None):
    kw = dict(temperature=EVO_CONFIG["temperature"], top_p=EVO_CONFIG["top_p"], top_k=EVO_CONFIG["top_k"],
              min_p=EVO_CONFIG["min_p"], max_tokens=int(max_tokens), seed=int(seed), skip_special_tokens=False)
    if stop_ids:
        kw["stop_token_ids"] = list(stop_ids)
    return SamplingParams(**kw)

def _evo_close_phase1(ids, text, stop_reason):
    """Continuation ids after phase 1 and whether thinking closed naturally. Only the </think> token (in the ids,
    as stop_reason, or in the text) counts as a natural close; EOS or the length limit inside thinking is a forced
    close with Qwen's documented stop-thinking text."""
    ids = list(ids)
    if _EVO_THINK_END_ID in ids:
        return ids[:ids.index(_EVO_THINK_END_ID) + 1] + _EVO_AFTER_THINK_IDS, True
    if stop_reason == _EVO_THINK_END_ID or "</think>" in (text or ""):
        return ids + [_EVO_THINK_END_ID] + _EVO_AFTER_THINK_IDS, True      # stopped on </think>, token not echoed
    while ids and ids[-1] in _EVO_EOS_IDS:
        ids = ids[:-1]
    return ids + _EVO_FORCE_IDS, False

EVO_ENGINE_FAILURES = []

def _evo_safe_generate(prompts, params, phase):
    """vLLM generate that never raises: an exception or a missing/short output list is recorded and returns None,
    so the run stops as INVALID / ENGINE FAILURE with its report and bundle instead of crashing the notebook."""
    try:
        out = arc_llm.generate(prompts, params, use_tqdm=False)
    except Exception as exc:
        EVO_ENGINE_FAILURES.append({"phase": phase, "type": type(exc).__name__, "error": str(exc)[-2000:],
                                    "requests": len(prompts)})
        return None
    out = list(out or [])
    if len(out) != len(prompts) or any(not getattr(x, "outputs", None) for x in out):
        EVO_ENGINE_FAILURES.append({"phase": phase, "type": "output_cardinality", "expected": len(prompts),
                                    "received": len(out)})
        return None
    return out

def evo_generate(jobs):
    """Two-phase generation for a batch of jobs. Phase 1: think until </think> or think_budget. Phase 2: token-exact
    continuation (forced close when needed), then the answer. Fills job['answer'], ['tokens'], ['think_closed'].
    Returns False on an engine failure (recorded in EVO_ENGINE_FAILURES)."""
    if not jobs:
        return True
    p1 = _evo_safe_generate([{"prompt_token_ids": j["prompt_ids"]} for j in jobs],
                            [_evo_sp(EVO_CONFIG["think_budget"], j["seed"], [_EVO_THINK_END_ID]) for j in jobs], "phase1")
    if p1 is None:
        return False
    cont = []
    for j, r in zip(jobs, p1):
        o = r.outputs[0]
        ids, closed = _evo_close_phase1(o.token_ids, o.text, getattr(o, "stop_reason", None))
        j["think_tokens"] = len(o.token_ids); j["think_closed"] = bool(closed); j["think_finish"] = o.finish_reason
        cont.append(j["prompt_ids"] + ids)
    p2 = _evo_safe_generate([{"prompt_token_ids": c} for c in cont],
                            [_evo_sp(EVO_CONFIG["answer_max_tokens"], j["seed"] + 1) for j in jobs], "phase2")
    if p2 is None:
        return False
    for j, r in zip(jobs, p2):
        o = r.outputs[0]
        j["answer"] = o.text or ""
        j["answer_tokens"] = len(o.token_ids); j["answer_finish"] = o.finish_reason
        j["tokens"] = int(j["think_tokens"] + j["answer_tokens"])
    return True

def _evo_parse(answer):
    code = extract_last_solution_code(answer)
    m = _EVO_RULE_RE.search(answer or "")
    rule = (m.group(1).strip() if m else "")[:1500]
    return rule, code

def _evo_train_signature(verifier):
    return json.dumps([(x.get("exact"), x.get("pred")) for x in list((verifier or {}).get("train") or [])])[:20000]

def evo_identity_wrong_frac(task):
    """Wrong-cell fraction of the identity program (return the input), counted like _v3_train_quality: a wrong-shaped
    pair counts every expected cell wrong. The reference every near-miss claim is measured against (S12 lesson)."""
    wrong = total = 0
    for ex in task["train"]:
        a, b = ex["input"], ex["output"]
        cells = len(b) * len(b[0]); total += cells
        if len(a) == len(b) and all(len(r) == len(b[0]) for r in a):
            wrong += sum(x != y for ra, rb in zip(a, b) for x, y in zip(ra, rb))
        else:
            wrong += cells
    return round(wrong / max(total, 1), 6)

EVO_INFRA_FAILURES = []
EVO_ROUND0_MIN_TOK_PER_SEC = 60.0   # assumed aggregate throughput before anything is measured (baseline 65 tok/s)
EVO_CHUNK_CALLS = 128               # a round is generated in chunks of this many calls; deadline checked before each
EVO_SUBMISSION_PATH = Path("/kaggle/working/submission.json")

def _evo_valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and len(r) == len(g[0]) for r in g)
            and 1 <= len(g[0]) <= 30 and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))

def evo_build_submission(task_ids):
    """Competition format: {task_id: [{"attempt_1": grid, "attempt_2": grid} per test query]}. Attempts come from
    evo_attempts_by_query; a query without an attempt gets the test input as fallback."""
    sub = {}
    for tid in task_ids:
        task = eval_challenges[tid]
        n = len(task["test"])
        att = evo_attempts_by_query(EVO_STATE[tid]["cands"], n) if tid in EVO_STATE else [[] for _ in range(n)]
        rows = []
        for i in range(n):
            fb = [list(r) for r in task["test"][i]["input"]]
            a = [g for g in (att[i] if i < len(att) else []) if _evo_valid_grid(g)]
            rows.append({"attempt_1": a[0] if a else fb, "attempt_2": a[1] if len(a) > 1 else fb})
        sub[tid] = rows
    return sub

def evo_write_submission(task_ids):
    sub = evo_build_submission(task_ids)
    assert set(sub) == set(task_ids)
    assert all(_evo_valid_grid(x["attempt_1"]) and _evo_valid_grid(x["attempt_2"]) for rows in sub.values() for x in rows)
    EVO_SUBMISSION_PATH.write_text(json.dumps(sub), encoding="utf-8")
    return sub

def evo_submission_task_ids():
    # rerun: EVERY hidden test task (also tasks dropped by the context gate); evaluation: the frozen eval subset
    return sorted(test_challenges) if EVO_RERUN else list(EVO_TASKS)

def _evo_verify(cand, task):
    try:
        return _evo_verify_inner(cand, task)
    except Exception as exc:
        try:
            ident = evo_identity_wrong_frac(task); qual = list(_v3_train_quality(None, task))
        except Exception:
            ident = 1.0; qual = list(_v3_train_quality(None, {"train": []}))
        cand["verifier"] = None; cand["quality"] = qual; cand["infra_failure"] = True
        cand["wrong_frac"] = 1.0; cand["identity_wrong_frac"] = ident
        cand["gain_vs_identity"] = round(ident - 1.0, 6)
        EVO_INFRA_FAILURES.append({"cand_id": cand.get("cand_id"), "errors": [f"verify exception {type(exc).__name__}: {exc}"[:500]]})
        return cand

def _evo_verify_inner(cand, task):
    ident = evo_identity_wrong_frac(task)
    if not cand.get("code"):
        cand["verifier"] = None
        cand["quality"] = list(_v3_train_quality(None, task))
    else:
        v = fresh_verify_solution(cand["code"], task, VERIFIER_TIMEOUT_SEC)
        cand["verifier"] = v
        cand["quality"] = list(_v3_train_quality(v, task))
        cand["solved_train"] = bool(v.get("all_train_exact") and v.get("test_execution_ok"))
        if v.get("infrastructure_failure"):
            cand["infra_failure"] = True
            EVO_INFRA_FAILURES.append({"cand_id": cand.get("cand_id"), "errors": v.get("infrastructure_errors")})
    cand["wrong_frac"] = round(-float(cand["quality"][6]), 6)
    cand["identity_wrong_frac"] = ident
    cand["gain_vs_identity"] = round(ident - cand["wrong_frac"], 6)
    return cand

def _evo_rank_key(c):
    """(train-exact and test runs, train-exact, exact pairs, right-shaped pairs, gain over the identity program,
    test runs, pairs without exception, -wrong-cell fraction, shorter code)."""
    q = c["quality"]
    return (q[0], q[1], q[2], q[3], c.get("gain_vs_identity", 0.0), q[4], q[5], q[6], -len(c.get("code") or ""))

def _evo_select_parents(cands, k, exclude=()):
    out, seen = [], set()
    for c in sorted(cands, key=_evo_rank_key, reverse=True):
        if not c.get("code") or c.get("cand_id") in exclude:
            continue
        sig = _evo_train_signature(c.get("verifier"))
        if sig in seen:
            continue
        seen.add(sig); out.append(c)
        if len(out) >= k:
            break
    return out

def _evo_exact_cands(cands):
    """Train-exact candidates, best first (ranking uses training evidence only, never test labels)."""
    return sorted([c for c in cands if (c.get("verifier") or {}).get("all_train_exact")], key=_evo_rank_key, reverse=True)

def evo_attempts_by_query(cands, n_queries):
    """Canonical pass@2, per test query: for each query, up to two distinct non-None outputs taken from train-exact
    candidates in rank order. A program that fails on one query still contributes its valid outputs on the others."""
    out = [[] for _ in range(int(n_queries))]
    for c in _evo_exact_cands(cands):
        outs = list((c.get("verifier") or {}).get("test_outputs") or [])
        for i in range(int(n_queries)):
            g = outs[i] if i < len(outs) else None
            if g is not None and len(out[i]) < 2 and g not in out[i]:
                out[i].append(g)
    return out

def evo_task_complete(cands, n_queries):
    """Search for a task stops only when every test query has two distinct attempts."""
    return all(len(a) >= 2 for a in evo_attempts_by_query(cands, n_queries))

def evo_verdict(solved, evo_only, stop_reason, infra_failures):
    """Pre-registered verdict. A run that did not execute as planned never yields a model verdict."""
    if infra_failures:
        return "INVALID / VERIFIER INFRASTRUCTURE FAILURE"
    if stop_reason == "no_model":
        return "INVALID / NO MODEL"
    if stop_reason == "engine_error":
        return "INVALID / ENGINE FAILURE"
    if stop_reason == "evo_exception":
        return "INVALID / EVO EXCEPTION"
    if stop_reason == "deadline":
        return "INCOMPLETE / NO DECISION"
    if stop_reason == "interface_code_extraction":
        return "WEAK / REDESIGN - INTERFACE FAILURE"
    if solved >= 3 and evo_only >= 2:          # 12-task run (9 NVARC misses): pre-registered before the run
        return "STRONG GO"
    if evo_only >= 1:
        return "PROMISING"
    return "WEAK / REDESIGN"

EVO_RECORDS_PATH = WORK_ROOT / "evo_records.jsonl"
EVO_TASKS = list(ACTIVE_TASK_IDS)
EVO_STATE = {t: {"cands": [], "tokens": 0, "calls": 0, "skipped_calls": 0, "backfills": 0, "alternatives": 0,
                 "repair_parents_nonpositive_gain": 0, "exact_partial_repairs": 0}
             for t in EVO_TASKS}
EVO_ROUNDS = []

def _evo_log(rec):
    with open(EVO_RECORDS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

def evo_run_round(round_no, plan):
    """plan: list of (tid, slot, kind, prompt_text, parent_or_None). Generates, verifies and records one round.
    The per-task token cap is a hard ceiling: every admitted call reserves its maximum size."""
    t0 = time.perf_counter()
    jobs, reserved = [], {}
    for tid, slot, kind, text, parent in plan:
        st = EVO_STATE[tid]
        if st["tokens"] + reserved.get(tid, 0) + EVO_CALL_MAX > EVO_CONFIG["task_token_cap"]:
            st["skipped_calls"] += 1
            continue
        rendered = render_evo_chat(text)
        ids = list(tokenizer.encode(rendered, add_special_tokens=False))
        if len(ids) + EVO_CALL_MAX + CONTEXT_SAFETY_TOKENS > MAX_MODEL_LEN and kind == "repair":
            rendered = render_evo_chat(evo_repair_prompt(eval_challenges[tid], parent, with_grids=False))
            ids = list(tokenizer.encode(rendered, add_special_tokens=False))
        if len(ids) + EVO_CALL_MAX + CONTEXT_SAFETY_TOKENS > MAX_MODEL_LEN:
            st["skipped_calls"] += 1
            _evo_log({"run_id": RUN_ID, "task_id": tid, "round": round_no, "slot": slot, "skipped": "context"})
            continue
        reserved[tid] = reserved.get(tid, 0) + EVO_CALL_MAX
        jobs.append({"task_id": tid, "slot": slot, "kind": kind, "round": round_no, "prompt_ids": ids,
                     "prompt_tokens": len(ids), "seed": _evo_seed(tid, round_no, slot),
                     "parent_id": (parent or {}).get("cand_id")})
    gen_t0 = time.perf_counter()
    if not evo_generate(jobs):
        for j in jobs:
            _evo_log({"run_id": RUN_ID, "task_id": j["task_id"], "round": round_no, "slot": j["slot"],
                      "kind": j["kind"], "engine_failure": True})
        info = {"round": round_no, "calls": len(jobs), "engine_failure": EVO_ENGINE_FAILURES[-1],
                "gen_sec": round(time.perf_counter() - gen_t0, 1), "round_sec": round(time.perf_counter() - t0, 1)}
        EVO_ROUNDS.append(info)
        print(f"[EVO] round {round_no}: ENGINE FAILURE", json.dumps(info, default=str))
        return info
    gen_sec = time.perf_counter() - gen_t0
    cands = []
    for j in jobs:
        rule, code = _evo_parse(j["answer"])
        cands.append({"cand_id": f"{j['task_id']}:r{round_no}:s{j['slot']}", "task_id": j["task_id"], "round": round_no,
                      "slot": j["slot"], "kind": j["kind"], "parent_id": j["parent_id"], "rule": rule, "code": code,
                      "tokens": j["tokens"], "think_tokens": j["think_tokens"], "think_closed": j["think_closed"],
                      "think_finish": j["think_finish"], "answer_tokens": j["answer_tokens"],
                      "answer_finish": j["answer_finish"], "prompt_tokens": j["prompt_tokens"],
                      "answer_tail": j["answer"][-600:]})
    ver_t0 = time.perf_counter()
    with _evo_futures.ThreadPoolExecutor(max_workers=EVO_CONFIG["verify_workers"]) as ex:
        cands = list(ex.map(lambda c: _evo_verify(c, eval_challenges[c["task_id"]]), cands))
    ver_sec = time.perf_counter() - ver_t0
    for c in cands:
        st = EVO_STATE[c["task_id"]]
        st["cands"].append(c); st["tokens"] += int(c["tokens"]); st["calls"] += 1
        v = c.get("verifier") or {}
        _evo_log({"run_id": RUN_ID, **{k: c[k] for k in c if k not in ("verifier",)},
                  "verifier_status": v.get("verifier_status"), "all_train_exact": bool(v.get("all_train_exact")),
                  "test_execution_ok": bool(v.get("test_execution_ok")), "train_exact_pairs": _v25_train_exact_count(v),
                  "infrastructure_failure": bool(v.get("infrastructure_failure")),
                  "test_outputs": v.get("test_outputs") if v.get("all_train_exact") else None})
    gen_tokens = sum(int(j["tokens"]) for j in jobs)
    info = {"round": round_no, "calls": len(jobs), "kinds": dict(collections.Counter(j["kind"] for j in jobs)),
            "gen_sec": round(gen_sec, 1), "verify_sec": round(ver_sec, 1),
            "round_sec": round(time.perf_counter() - t0, 1), "generated_tokens": gen_tokens,
            "tok_per_sec": round(gen_tokens / max(gen_sec, 1e-6), 1),
            "code_extracted": sum(1 for c in cands if c.get("code")),
            "think_closed": sum(1 for c in cands if c.get("think_closed")),
            "train_exact": sum(1 for c in cands if (c.get("verifier") or {}).get("all_train_exact")),
            "solved_train": sum(1 for c in cands if c.get("solved_train")),
            "infra_failures_total": len(EVO_INFRA_FAILURES),
            "tasks_all_queries_attempted": sum(1 for t in EVO_TASKS if all(evo_attempts_by_query(EVO_STATE[t]["cands"], len(eval_challenges[t]["test"])))),
            "tasks_complete_2_per_query": sum(1 for t in EVO_TASKS if evo_task_complete(EVO_STATE[t]["cands"], len(eval_challenges[t]["test"])))}
    EVO_ROUNDS.append(info)
    print(f"[EVO] round {round_no}:", json.dumps(info))
    return info

def evo_run_round_chunked(round_no, plan):
    """Run one round in chunks of EVO_CHUNK_CALLS. Before each chunk, project its duration from the throughput
    measured so far (1.25x margin; before any measurement: maximum call size at EVO_ROUND0_MIN_TOK_PER_SEC) and stop
    cleanly if it would cross the deadline. Unfinished tasks keep fallback answers in submission.json."""
    agg = {"round": round_no, "calls": 0, "code_extracted": 0, "deadline_stop": False, "engine_failure": False, "chunks": 0}
    for k in range(0, len(plan), EVO_CHUNK_CALLS):
        chunk = plan[k:k + EVO_CHUNK_CALLS]
        done = [i for i in EVO_ROUNDS if "generated_tokens" in i and i.get("calls")]
        if done:
            rate = sum(i["generated_tokens"] for i in done) / max(sum(i["gen_sec"] for i in done), 1e-6)
            per_call = sum(i["generated_tokens"] for i in done) / max(sum(i["calls"] for i in done), 1)
            verify = sum(i["verify_sec"] for i in done) / max(sum(i["calls"] for i in done), 1)
            est = 1.25 * len(chunk) * (per_call / max(rate, 1e-6) + verify)
        else:
            est = len(chunk) * EVO_CALL_MAX / EVO_ROUND0_MIN_TOK_PER_SEC
        if time.perf_counter() + est > deadline:   # deadline is on the perf_counter clock (cell 7)
            print(f"[EVO] STOP before round {round_no} chunk {k // EVO_CHUNK_CALLS + 1}: projected {est:.0f}s would cross the deadline.")
            agg["deadline_stop"] = True
            return agg
        info = evo_run_round(round_no, chunk)
        agg["chunks"] += 1; agg["calls"] += int(info.get("calls") or 0); agg["code_extracted"] += int(info.get("code_extracted") or 0)
        if info.get("engine_failure"):
            agg["engine_failure"] = True
            return agg
        try:
            evo_write_submission(evo_submission_task_ids())   # keep submission.json current after every chunk
        except Exception as exc:
            print("[EVO] submission refresh failed (kept previous file):", type(exc).__name__, exc)
    return agg

def evo_plan_generation(g):
    """Two slots per unfinished task. A task is finished only with two distinct pass@2 outputs. With one: slot 0
    asks for a genuinely different program, slot 1 repairs the best other candidate. Slots without a usable parent
    are backfilled with fresh attempts, so a task whose first answers had no code is not abandoned."""
    plan = []
    k = EVO_CONFIG["parents"]
    for t in EVO_TASKS:
        st = EVO_STATE[t]; task = eval_challenges[t]
        good = _evo_exact_cands(st["cands"])
        if evo_task_complete(st["cands"], len(task["test"])):
            continue
        slots = []
        if good:
            slots.append(("alternative", evo_alternative_prompt(task, good[0]), good[0]))
            st["alternatives"] += 1
            partial = [c for c in good if not (c.get("verifier") or {}).get("test_execution_ok")]
            if partial:   # train-exact but fails on a test input: repair it with the TEST INPUT FAILURE report
                slots.append(("repair", evo_repair_prompt(task, partial[0]), partial[0]))
                st["exact_partial_repairs"] += 1
                parents = []
            else:
                parents = _evo_select_parents(st["cands"], k - 1, exclude={c["cand_id"] for c in good})
        else:
            parents = _evo_select_parents(st["cands"], k)
        slots += [("repair", evo_repair_prompt(task, p), p) for p in parents]
        st["repair_parents_nonpositive_gain"] += sum(1 for p in parents if p.get("gain_vs_identity", 0.0) <= 0 and p["quality"][2] == 0)
        while len(slots) < k:
            slots.append(("fresh", evo_fresh_prompt(task, g * k + len(slots)), None))
            st["backfills"] += 1
        plan += [(t, s, kind, text, par) for s, (kind, text, par) in enumerate(slots)]
    return plan

def run_evo():
    EVO_RECORDS_PATH.write_text("", encoding="utf-8")
    loop_t0 = time.perf_counter()
    plan = [(t, s, "initial", evo_initial_prompt(eval_challenges[t], s), None)
            for t in EVO_TASKS for s in range(EVO_CONFIG["organisms"])]
    r0 = evo_run_round_chunked(0, plan)
    if r0["deadline_stop"]:
        return {"stopped": "deadline", "loop_sec": time.perf_counter() - loop_t0}
    if EVO_ENGINE_FAILURES:
        print("[EVO] STOP: engine failure; the run is invalid as a measurement.", EVO_ENGINE_FAILURES[-1])
        return {"stopped": "engine_error", "loop_sec": time.perf_counter() - loop_t0}
    if EVO_INFRA_FAILURES:
        print("[EVO] STOP: verifier infrastructure failure; the run is invalid as a measurement.", EVO_INFRA_FAILURES[:3])
        return {"stopped": "verifier_infrastructure", "loop_sec": time.perf_counter() - loop_t0}
    if r0["calls"] and r0["code_extracted"] < 0.25 * r0["calls"]:
        print("[EVO] STOP: fewer than 25 % of round-0 answers contained a solution(); interface problem, no repairs.")
        return {"stopped": "interface_code_extraction", "loop_sec": time.perf_counter() - loop_t0}
    for g in range(1, EVO_CONFIG["repair_generations"] + 1):
        plan = evo_plan_generation(g)
        if not plan:
            break
        rg = evo_run_round_chunked(g, plan)
        if rg["deadline_stop"]:
            return {"stopped": "deadline", "loop_sec": time.perf_counter() - loop_t0}
        if EVO_ENGINE_FAILURES:
            print("[EVO] STOP: engine failure; the run is invalid as a measurement.", EVO_ENGINE_FAILURES[-1])
            return {"stopped": "engine_error", "loop_sec": time.perf_counter() - loop_t0}
        if EVO_INFRA_FAILURES:
            print("[EVO] STOP: verifier infrastructure failure; the run is invalid as a measurement.", EVO_INFRA_FAILURES[:3])
            return {"stopped": "verifier_infrastructure", "loop_sec": time.perf_counter() - loop_t0}
    return {"stopped": None, "loop_sec": time.perf_counter() - loop_t0}

def run_evo_selftests():
    _t = {"train": [{"input": [[1, 0], [0, 0]], "output": [[1]]}, {"input": [[0, 0], [0, 7]], "output": [[7]]}],
          "test": [{"input": [[0, 3], [0, 0]]}]}
    ans = "## Rule\nCrop to the single coloured cell.\n## Code\n```python\ndef solution(grid):\n    return crop(grid, *bbox(grid, ignore=0))\n```"
    rule, code = _evo_parse(ans)
    assert rule.startswith("Crop") and code and "def solution" in code, (rule, code)
    c = _evo_verify({"code": code, "cand_id": "c"}, _t)
    assert c["solved_train"] and c["verifier"]["test_outputs"] == [[[3]]] and not c.get("infra_failure"), c
    bad = _evo_verify({"code": "def solution(grid):\n    return [row[:] for row in grid]", "cand_id": "b"}, _t)
    assert not bad.get("solved_train") and _evo_rank_key(c) > _evo_rank_key(bad)
    none = _evo_verify({"code": None, "cand_id": "n"}, _t)
    assert _evo_rank_key(bad) >= _evo_rank_key(none)
    rp = evo_repair_prompt(_t, {**bad, "rule": "copy"})
    assert "wrong size 2x2, expected 1x1" in rp and "## Rule\ncopy" in rp and "ARC-EVO" not in rp, rp[-800:]
    ap = evo_alternative_prompt(_t, {**c, "rule": "crop"})
    assert "genuinely different transformation" in ap and "crop(grid" in ap
    assert "substantially different hypothesis" in evo_fresh_prompt(_t, 5)
    assert len(_evo_select_parents([c, bad, none, dict(bad)], 2)) == 2
    assert _evo_select_parents([c, bad], 1, exclude={"c"})[0]["cand_id"] == "b"
    assert evo_attempts_by_query([c, dict(c)], 1) == [[[[3]]]]     # identical outputs are not used twice
    c2 = _evo_verify({"code": "def solution(grid):\n    return [[max(max(r) for r in grid)]]", "cand_id": "c2"}, _t)
    assert evo_attempts_by_query([c, c2], 1) == [[[[3]]]] and not evo_task_complete([c, c2], 1)
    # review R2: per-query pass@2. A:(A0,X) > B:(B0,X) > C:(B0,Y); truth (B0,Y) is reachable only per query.
    _q = [True, True, 3, 3, True, 3, -0.0]
    _mk = lambda cid, outs, code: {"cand_id": cid, "code": code, "quality": list(_q), "gain_vs_identity": 0.5,
                                   "verifier": {"all_train_exact": True, "test_outputs": outs}}
    A0, B0, X, Y = [[1]], [[2]], [[3]], [[4]]
    _abc = [_mk("A", [A0, X], "a"), _mk("B", [B0, X], "bb"), _mk("C", [B0, Y], "ccc")]
    _att = evo_attempts_by_query(_abc, 2)
    assert _att == [[A0, B0], [X, Y]], _att
    assert all(any(g == gt for g in _att[i]) for i, gt in enumerate([B0, Y]))     # strict under canonical pass@2
    assert not evo_task_complete(_abc[:2], 2) and evo_task_complete(_abc, 2)
    _part = [_mk("P", [A0, None], "p"), _mk("Q", [None, Y], "qq")]
    assert evo_attempts_by_query(_part, 2) == [[A0], [Y]]                          # partial outputs are kept
    # identity-relative ranking (S12 lesson): a candidate worse than returning the input has negative gain
    _ti = {"train": [{"input": [[1, 1, 1, 1], [1, 1, 1, 1]], "output": [[1, 1, 1, 1], [1, 1, 1, 2]]}], "test": [{"input": [[1]]}]}
    assert abs(evo_identity_wrong_frac(_ti) - 0.125) < 1e-9
    worse = _evo_verify({"code": "def solution(grid):\n    return [[1, 1, 1, 1], [1, 1, 2, 3]]", "cand_id": "w"}, _ti)
    near = _evo_verify({"code": "def solution(grid):\n    return [[1, 1, 1, 1], [1, 1, 1, 3]]", "cand_id": "m"}, _ti)
    assert worse["gain_vs_identity"] < 0 and near["gain_vs_identity"] == 0.0 and _evo_rank_key(near) > _evo_rank_key(worse)
    # phase-1 close: </think> in ids / as stop_reason / EOS inside thinking / length
    i1, k1 = _evo_close_phase1([5, _EVO_THINK_END_ID, 9], "", None)
    assert k1 and i1 == [5, _EVO_THINK_END_ID] + _EVO_AFTER_THINK_IDS
    i2, k2 = _evo_close_phase1([5], "x", _EVO_THINK_END_ID)
    assert k2 and i2 == [5, _EVO_THINK_END_ID] + _EVO_AFTER_THINK_IDS
    _eos = sorted(_EVO_EOS_IDS)[0]
    i3, k3 = _evo_close_phase1([5, _eos], "x", None)
    assert not k3 and i3 == [5] + _EVO_FORCE_IDS
    i4, k4 = _evo_close_phase1([5, 6], "x", None)
    assert not k4 and i4 == [5, 6] + _EVO_FORCE_IDS
    # verdicts: an incomplete or invalid run never yields a model verdict
    assert evo_verdict(9, 9, "deadline", []) == "INCOMPLETE / NO DECISION"
    assert evo_verdict(9, 9, None, [{"x": 1}]).startswith("INVALID")
    assert evo_verdict(3, 2, None, []) == "STRONG GO" and evo_verdict(3, 0, None, []) == "WEAK / REDESIGN"
    assert evo_verdict(1, 1, None, []) == "PROMISING" and evo_verdict(5, 1, None, []) == "PROMISING"
    assert _evo_valid_grid([[0, 9]]) and not _evo_valid_grid([[10]]) and not _evo_valid_grid([[1], [1, 2]]) and not _evo_valid_grid([[True]])
    assert evo_verdict(9, 9, "engine_error", []) == "INVALID / ENGINE FAILURE"
    assert evo_verdict(9, 9, "evo_exception", []) == "INVALID / EVO EXCEPTION"
    _bv = _evo_verify({"code": "def solution(grid):\n    return grid", "cand_id": "boom"}, None)   # task=None raises inside
    assert _bv.get("infra_failure") and EVO_INFRA_FAILURES, _bv
    _pr = evo_repair_prompt(_t, {"rule": "r", "code": "def solution(grid):\n    return [[1]]",
                                 "verifier": {"all_train_exact": True, "test_execution_ok": False, "train": [],
                                              "test_error": "KeyError: 9"}})
    assert "TEST INPUT FAILURE" in _pr, _pr[-600:]
    r = render_evo_chat("x")
    assert "<tools>" not in r and r.rstrip().endswith("<think>"), r[-200:]
    EVO_INFRA_FAILURES.clear(); EVO_ENGINE_FAILURES.clear()
    print("EVO self-tests PASS")
    return True

import collections
EVO_SELFTEST_OK = run_evo_selftests()
# a valid submission.json exists before any generation (fallback = test inputs), then is refreshed after every chunk
EVO_FALLBACK_SUBMISSION = evo_write_submission(evo_submission_task_ids())
print("fallback submission.json written:", len(EVO_FALLBACK_SUBMISSION), "tasks")
if USE_MODEL:
    try:
        EVO_RESULT = run_evo()
    except Exception as _evo_exc:
        import traceback as _evo_tb
        EVO_RESULT = {"stopped": "evo_exception", "error": f"{type(_evo_exc).__name__}: {_evo_exc}"[:2000],
                      "traceback": _evo_tb.format_exc()[-4000:]}
        print("[EVO] UNEXPECTED EXCEPTION; run marked INVALID, report and bundle still written:", EVO_RESULT["error"])
else:
    EVO_RESULT = {"stopped": "no_model"}
print("Cell 8 (EVO) complete:", json.dumps(EVO_RESULT, default=str))
