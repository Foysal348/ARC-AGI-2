# v1.9.1 DRYCHECK: only after a DRYRUN (CPU, fake server). Checks the files the real path wrote and the launched cfg,
# and writes /kaggle/working/dryrun_report.json. Fake-server results say NOTHING about Qwen accuracy.
import glob, json, sys
from pathlib import Path

if Path("/kaggle/working/dryrun_flag").exists() and Path("/kaggle/working/dryrun_flag").read_text().strip() == "1":
    W = Path("/kaggle/working")
    CFG = globals().get("cfg") or {}   # the launch cell's cfg (same kernel)
    checks = {}

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"[:300]
        checks[name] = {"ok": bool(ok), "detail": str(detail)[:300]}

    def load(p):
        return json.loads((W / p).read_text())

    def ok_grid(g):
        return (isinstance(g, list) and 1 <= len(g) <= 30
                and all(isinstance(r, list) and 1 <= len(r) <= 30 and len(r) == len(g[0]) for r in g)
                and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))

    srv = load("agent_server_report.json") if (W / "agent_server_report.json").exists() else {}
    ev = load("agent_eval/eval_report.json") if (W / "agent_eval/eval_report.json").exists() else {}
    check("server_report_no_error", lambda: (bool(srv) and not srv.get("error"), srv.get("error")))
    check("budget_contract_pass", lambda: (srv.get("budget_contract_pass") is True, srv.get("chosen")))

    def _contract():
        rc = srv["rungs"][-1]["budget_smoke"]["request_contract"]
        return (rc["thinking_token_budget"] == 49152 and rc["max_tokens_full"] == 57344
                and rc["effort"] == "xhigh" and rc["tool_choice"] is None), rc
    check("request_contract_49152_57344_xhigh_no_tools", _contract)
    check("eval_report_v191", lambda: (ev.get("version") == "v1.9.1", ev.get("version")))
    check("official_is_micro", lambda: (
        ev["official_score_pct"] == round(100.0 * ev["query_hits"] / max(ev["test_outputs"], 1), 2)
        and ev["agent_only_official_pct"] == round(100.0 * ev["agent_query_hits"] / max(ev["test_outputs"], 1), 2),
        {k: ev.get(k) for k in ("official_score_pct", "query_hits", "test_outputs", "task_macro_score_pct")}))

    def _attempts():
        n = len((W / "agent_eval/attempts.jsonl").read_text().splitlines())
        return n == ev["tasks"] == len(CFG["eval_task_ids"]), f"{n} records / {ev['tasks']} tasks"
    check("one_attempt_record_per_task", _attempts)
    KNOWN = {"verified_usable", "verified_test_error", "train_mismatch", "no_program", "nonportable",
             "unsafe_program", "context_too_large", "deadline", "request_error", "request_timeout",
             "budget_rejected", "unexpected_tool_call", "malformed_response", "verification_error", "crashed"}
    check("statuses_known", lambda: (set(ev["status_counts"]) <= KNOWN, ev["status_counts"]))
    check("think_accounting_recorded", lambda: (
        all(v.get("think_source") in ("usage", "derived") for v in ev["think"].values()), ev["think"]))

    def _eval_sub():
        sub = load("agent_eval/eval_submission.json")
        bad = [t for t in ev["per_task"] for r in sub.get(t, [None])
               if not (isinstance(r, dict) and ok_grid(r.get("attempt_1")) and ok_grid(r.get("attempt_2")))]
        return not bad and set(sub) == set(ev["per_task"]), f"bad rows {bad[:5]}"
    check("eval_submission_schema", _eval_sub)

    def _submission():
        ch = json.load(open(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)[0]))
        sub = load("submission.json")
        return set(sub) == set(ch), f"{len(sub)} tasks / {len(ch)} hidden"
    check("submission_covers_every_hidden_task", _submission)
    check("server_metrics_written", lambda: ((W / "agent_eval/server_metrics.json").exists(), ""))
    sys.path.insert(0, "/kaggle/working/agent")
    from check_config import validate

    def _cfg_ok():
        return validate(dict(CFG)), "launched cfg passes check_config"
    check("launched_cfg_passes_gate", _cfg_ok)

    def _neg(key, value):
        def f():
            try:
                validate(dict(CFG, **{key: value}))
            except ValueError as e:
                return True, str(e)
            return False, f"{key}={value} was accepted"
        return f
    for _k, _v in (("rescue_min_think", 4096), ("synth_think", 6144), ("repair_turns", 1), ("force_submit_after", 1)):
        check(f"gate_rejects_{_k}_{_v}", _neg(_k, _v))
    check("eval12_default", lambda: (len(CFG["eval_task_ids"]) == 12 or bool(__import__("os").environ.get("Q27A_EVAL_TASKS")),
                                     len(CFG["eval_task_ids"])))
    n_ok = sum(v["ok"] for v in checks.values())
    (W / "dryrun_report.json").write_text(json.dumps({"version": "v1.9.1", "passed": n_ok, "total": len(checks),
                                                      "checks": checks}, indent=1))
    for k, v in checks.items():
        print(("PASS " if v["ok"] else "FAIL ") + k, "" if v["ok"] else v["detail"])
    print(f"DRYCHECK v1.9.1: {n_ok}/{len(checks)} passed", flush=True)
else:
    print("DRYCHECK skipped (not a DRYRUN)")
