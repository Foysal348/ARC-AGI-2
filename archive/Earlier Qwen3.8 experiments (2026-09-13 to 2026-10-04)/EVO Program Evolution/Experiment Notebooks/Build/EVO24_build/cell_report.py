# =====================================================================================
# CELL 9 — EVO SCORING: pass@2 strict score, complementarity with NVARC and old PIV, go/no-go, bundle
# =====================================================================================
import csv as _evo_csv, zipfile

# Frozen post-hoc references (never used by the solver).
NVARC_EVAL120_FROZEN = __NVARC_JSON__
PIV_OLD_STRICT = {"1818057f", "2ba387bc"}          # canonical baseline, Strict Solved
PIV_OLD_TRAIN_VERIFIED = {"1818057f", "2ba387bc", "7b5033c1"}
EVO_GO_RULES = {"strong_go": "solved>=3 and evo_only>=2 (12 tasks, 9 NVARC misses)", "promising": "evo_only>=1",
                "weak": "otherwise", "no_decision": "deadline stop -> INCOMPLETE; verifier infrastructure failure or no model -> INVALID"}

def evo_score():
    rows = []
    nv_score = NVARC_EVAL120_FROZEN["per_task_score"]
    for t in EVO_TASKS:
        st = EVO_STATE[t]
        gt = eval_solutions.get(t) or []
        attempts = evo_attempts_by_query(st["cands"], len(eval_challenges[t]["test"]))   # canonical per-query pass@2
        q = [any(a == g for a in attempts[i]) if i < len(attempts) else False for i, g in enumerate(gt)]
        strict = bool(q) and all(q)
        cands = st["cands"]
        best = max(cands, key=_evo_rank_key) if cands else None
        nvs = nv_score.get(t)
        rows.append({
            "task_id": t, "nvarc_task_score": nvs, "nvarc_miss": nvs == 0.0, "nvarc_partial": nvs is not None and 0.0 < nvs < 1.0,
            "piv_old_strict": t in PIV_OLD_STRICT, "evo_strict": strict, "evo_query_frac": (sum(q) / len(q)) if q else 0.0,
            "evo_train_exact_candidates": sum(1 for c in cands if (c.get("verifier") or {}).get("all_train_exact")),
            "evo_attempts": min((len(a) for a in attempts), default=0), "attempts_per_query": [len(a) for a in attempts],
            "repair_parents_nonpositive_gain": st["repair_parents_nonpositive_gain"], "calls": st["calls"], "tokens": st["tokens"], "skipped_calls": st["skipped_calls"],
            "code_extracted": sum(1 for c in cands if c.get("code")),
            "think_closed_naturally": sum(1 for c in cands if c.get("think_closed")),
            "best_train_exact_pairs": (best["quality"][2] if best else 0), "n_train": len(eval_challenges[t]["train"]),
            "identity_wrong_frac": evo_identity_wrong_frac(eval_challenges[t]),
            "best_wrong_frac": (best["wrong_frac"] if best else None),
            "best_gain_vs_identity": (best["gain_vs_identity"] if best else None),
            "alternatives": st["alternatives"], "backfills": st["backfills"],
            "first_train_exact_round": min([c["round"] for c in cands if (c.get("verifier") or {}).get("all_train_exact")], default=None),
        })
    n = len(rows)
    solved = sum(r["evo_strict"] for r in rows)
    evo_only = sum(1 for r in rows if r["evo_strict"] and r["nvarc_miss"])
    both = sum(1 for r in rows if r["evo_strict"] and (r["nvarc_task_score"] or 0) == 1.0)
    nvarc_only = sum(1 for r in rows if not r["evo_strict"] and (r["nvarc_task_score"] or 0) == 1.0)
    evo_on_partial = sum(1 for r in rows if r["evo_strict"] and r["nvarc_partial"])
    stop_reason = (EVO_RESULT or {}).get("stopped")
    verdict = evo_verdict(solved, evo_only, stop_reason, EVO_INFRA_FAILURES)
    calls = sum(r["calls"] for r in rows)
    summary = {
        "tasks": n, "planned_tasks": len(PLANNED_TASK_IDS), "queries": sum(len(eval_solutions.get(r["task_id"]) or []) for r in rows),
        "evo_strict_solved": solved, "evo_arc_score_pct": round(100 * sum(r["evo_query_frac"] for r in rows) / max(n, 1), 2),
        "EVOLUTION_ONLY_(nvarc_miss)": evo_only, "BOTH": both, "NVARC_ONLY": nvarc_only,
        "NVARC_PARTIAL_tasks": sum(r["nvarc_partial"] for r in rows), "EVO_SOLVES_NVARC_PARTIAL": evo_on_partial,
        "stop_reason": stop_reason, "verifier_infrastructure_failures": len(EVO_INFRA_FAILURES),
        "tasks_complete_2_per_query": sum(1 for r in rows if r["evo_attempts"] >= 2),
        "tasks_with_positive_gain_vs_identity": sum(1 for r in rows if (r["best_gain_vs_identity"] or 0) > 0),
        "nvarc_misses_in_subset": sum(r["nvarc_miss"] for r in rows),
        "piv_old_strict_in_subset": sum(r["piv_old_strict"] for r in rows),
        "tasks_with_train_exact_candidate": sum(1 for r in rows if r["evo_train_exact_candidates"] > 0),
        "train_exact_but_wrong_test": sum(1 for r in rows if r["evo_train_exact_candidates"] > 0 and not r["evo_strict"]),
        "repair_parents_nonpositive_gain_total": sum(r["repair_parents_nonpositive_gain"] for r in rows),
        "tasks_with_8_calls": sum(1 for r in rows if r["calls"] >= 8), "tasks_with_7_calls": sum(1 for r in rows if r["calls"] == 7),
        "tasks_with_6_or_fewer_calls": sum(1 for r in rows if r["calls"] <= 6),
        "budget_skipped_calls": sum(r["skipped_calls"] for r in rows),
        "exact_partial_repairs": sum(EVO_STATE[r["task_id"]]["exact_partial_repairs"] for r in rows),
        "engine_failures": EVO_ENGINE_FAILURES,
        "calls": calls, "code_extraction_rate": round(sum(r["code_extracted"] for r in rows) / max(calls, 1), 3),
        "natural_think_close_rate": round(sum(r["think_closed_naturally"] for r in rows) / max(calls, 1), 3),
        "tokens_total": sum(r["tokens"] for r in rows), "tokens_per_task_mean": round(sum(r["tokens"] for r in rows) / max(n, 1)),
        "rounds": EVO_ROUNDS, "evo_result": EVO_RESULT, "config": EVO_CONFIG,
        "verdict_rule": EVO_GO_RULES, "verdict": verdict,
        "solved_tasks": [r["task_id"] for r in rows if r["evo_strict"]],
        "evolution_only_tasks": [r["task_id"] for r in rows if r["evo_strict"] and r["nvarc_miss"]],
    }
    (WORK_ROOT / "evo_report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8")
    with open(WORK_ROOT / "evo_task_table.csv", "w", newline="", encoding="utf-8") as f:
        w = _evo_csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("=" * 116)
    print("EVO REPORT:", json.dumps({k: v for k, v in summary.items() if k not in ("rounds", "config", "evo_result")}, indent=2))
    print("ROUNDS:", json.dumps(EVO_ROUNDS, indent=1))
    for r in rows:
        print(f"  {r['task_id']} nvarc={r['nvarc_task_score']} evo_strict={r['evo_strict']} exact_cands={r['evo_train_exact_candidates']} "
              f"best_pairs={r['best_train_exact_pairs']}/{r['n_train']} wrong={r['best_wrong_frac']} identity={r['identity_wrong_frac']} "
              f"gain={r['best_gain_vs_identity']} attempts={r['evo_attempts']} calls={r['calls']} tok={r['tokens']}")
    print("VERDICT:", verdict)
    return summary

def evo_rerun_summary():
    n_att = {t: evo_attempts_by_query(EVO_STATE[t]["cands"], len(eval_challenges[t]["test"])) for t in EVO_TASKS}
    summary = {"mode": "competition_rerun", "hidden_tasks": len(test_challenges), "tasks_run": len(EVO_TASKS),
               "tasks_with_any_attempt": sum(1 for a in n_att.values() if any(a)),
               "tasks_all_queries_attempted": sum(1 for a in n_att.values() if a and all(a)),
               "calls": sum(EVO_STATE[t]["calls"] for t in EVO_TASKS),
               "tokens_total": sum(EVO_STATE[t]["tokens"] for t in EVO_TASKS),
               "stop_reason": (EVO_RESULT or {}).get("stopped"), "rounds": EVO_ROUNDS,
               "engine_failures": EVO_ENGINE_FAILURES, "verifier_infrastructure_failures": len(EVO_INFRA_FAILURES)}
    (WORK_ROOT / "evo_report.json").write_text(json.dumps({"summary": summary}, indent=2, default=str), encoding="utf-8")
    print("EVO RERUN SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "rounds"}, indent=2, default=str))
    return summary

try:
    EVO_SUMMARY = (evo_rerun_summary() if EVO_RERUN else evo_score()) if EVO_TASKS else {}
except Exception as _rep_exc:   # the bundle below must still be written
    import traceback as _rep_tb
    EVO_SUMMARY = {"scoring_exception": f"{type(_rep_exc).__name__}: {_rep_exc}", "traceback": _rep_tb.format_exc()[-4000:],
                   "evo_result": EVO_RESULT, "engine_failures": EVO_ENGINE_FAILURES}
    (WORK_ROOT / "evo_report.json").write_text(json.dumps({"summary": EVO_SUMMARY}, indent=2, default=str), encoding="utf-8")
    print("VERDICT: INVALID / SCORING EXCEPTION", EVO_SUMMARY["scoring_exception"])

# final submission.json (both modes); on any failure the last valid file (fallback or per-chunk refresh) stays
try:
    EVO_FINAL_SUBMISSION = evo_write_submission(evo_submission_task_ids())
    print("submission.json written:", len(EVO_FINAL_SUBMISSION), "tasks |", EVO_SUBMISSION_PATH)
except Exception as _sub_exc:
    print("FINAL submission write failed, previous submission.json kept:", type(_sub_exc).__name__, _sub_exc)
RUN_META = {"experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "run_profile": RUN_PROFILE, "tasks": EVO_TASKS,
            "engine_config": ENGINE_CONFIG if USE_MODEL else None, "model_fingerprint": MODEL_ARTIFACT_FINGERPRINT_SHA256,
            "wall_sec_total": round(time.perf_counter() - notebook_start, 1)}
(WORK_ROOT / "evo_run_metadata.json").write_text(json.dumps(RUN_META, indent=2, default=str), encoding="utf-8")
ZIP_PATH = Path("/kaggle/working")/BUNDLE_NAME
if ZIP_PATH.exists(): ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(WORK_ROOT.rglob("*")):
        if p.is_file() and p.stat().st_size < 200 * 1024**2:
            zf.write(p, p.relative_to(WORK_ROOT.parent).as_posix())
print("=" * 116); print(EXPERIMENT_ID, "PIPELINE COMPLETE | bundle:", ZIP_PATH, "| MB:", round(ZIP_PATH.stat().st_size / 1024**2, 2))
