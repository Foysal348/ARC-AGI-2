"""Item-by-item check of Exp 15 Notes/02 (items 1-55) against the BUILT E15B notebook and the B Duck package.
Read-only. Each item: the evidence that must be present (exact text in the notebook / package files), or the reason it
is not a code item. Usage: python verify_items.py [notebook]"""
import base64
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

E15 = Path(__file__).resolve().parents[1]
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else E15 / "Experiment Notebooks" / "Q27-XHIGH-E15B_SUBMIT_45K30K_S16_R3H_DUCK_SGLANG.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
# rev 3g: the per-attempt budgets and the commit window / budget of this variant (from its name)
A1, A2, CW, CB = ((40000, 35000, 92, 115) if "_40K35K_" in NB.name else (50000, 45000, 105, 128) if "_50K45K_" in NB.name
                  else (45000, 30000, 92, 115))
SRC = "\n".join("".join(c["source"]) for c in nb["cells"])
m = re.search(r'q27x_payload\("(q27x_e15b_duck_[0-9a-f]{12}\.bin)", "([0-9a-f]{64})"\)', SRC)
PAY = E15 / "Experiment Notebooks" / "Build" / "payload" / m.group(1)
data = PAY.read_bytes()
assert hashlib.sha256(data).hexdigest() == m.group(2), "payload sha"
Z = zipfile.ZipFile(io.BytesIO(data))
DUCK = Z.read("arc2_duck.py").decode("utf-8")
TA = Z.read("vendor/inference/agent/tool_agent.py").decode("utf-8")
SB = Z.read("vendor/inference/agent/python_tool_sandbox.py").decode("utf-8")
man = json.loads(re.search(r"_E12_MANIFEST = (\{.*?\})\n", SRC, re.S).group(1))
bad_man = [k for k in man if hashlib.sha256(Z.read(k)).hexdigest() != man[k]]
WHERE = {"nb": SRC, "duck": DUCK, "tool_agent": TA, "sandbox": SB}

ITEMS = [
    (1, "two independent attempts, attempt 2 = facts view", [("nb", '"E12_R6_B1_VIEW": "facts"'), ("nb", 'E12_BRANCHES", 2)')]),
    (2, "attempt 1 on every task first (stage 2 waits for stage-1 starts)", [("duck", 'prio[0] == 1 and state["stage1_left"] > 0')]),
    (3, "per-attempt budget (rev 3f: 45k / 30k), no pause / chunk / resume", [("nb", f'"E12_MAX_GEN", {A1}'), ("nb", '"E12_R6_FIRST_EXPOSURE": 0'),
                                                     ("nb", 'assert E15B["E12_B_ATTEMPT_TOKENS"] == E12_MAX_GEN_PER_BRANCH')]),
    (4, "no 12k/24k/32k caps: cap = remaining - reserve", [("duck", "cap = (left - self._arc2_b_reserve) if left > self._arc2_b_reserve + 1024 else max(256, left)")]),
    (5, "no trim inside an attempt: reserve = cap, fixed window", [("duck", "self._reply_reserve_tokens = cap"),
                                                                 ("nb", 'E12_RUN["rev6_window"] = min(131072, int(E15B["E12_B_WINDOW"]))')]),
    (6, "admission (superseded by #49)", [("nb", '"E12_R6_MIN_ADMIT_SEC": 300')]),
    (7, "a1/a2 assembly + partial-program test run (attempt_2 tier)", [("duck", "def arc2_b_partial_test(self)"),
                                                                     ("duck", 'agent.arc2_b_events["partial_test"] = agent.arc2_b_partial_test()'),
                                                                     ("duck", "for s in others + partial:")]),
    (8, "end-of-attempt harvest (autocheck + check harvest + #50)", [("duck", "def _arc2_autocheck"), ("duck", "def arc2_b_reasoning_harvest")]),
    (9, "xhigh only (no effort override)", [("nb", 'E12_REASONING_EFFORT = os.environ.get("E12_REASONING_EFFORT", "")')]),
    (10, "turn rollback fix (yield after the batch)", [("tool_agent", "if _yield_after_batch_enabled():"),
                                                       ("duck", 'env["ARC3_YIELD_AFTER_BATCH"] = "1"')]),
    (11, "submit / check called as tools -> python", [("duck", 'name not in ("submit", "check", "check_many")'),
                                                     ("duck", 'self.arc2_b_events["submit_tool_routed"] += 1')]),
    (12, "tool contract in the prompt", [("duck", "are Python functions \"\n     \"inside it, not tools")]),
    (13, "autocheck names candidate / trans", [("duck", r'_R6_AUTOCHECK_RE_B = re.compile(r"^(transform|solve|solution|predict|candidate)\w*$|^trans(\d*|_\w*)$|^\w+_(alt\w*|v\d+)$")')]),
    (14, "sandbox: sys shim, time, import messages", [("sandbox", "class _SysShim:"), ("sandbox", 'SAFE_MODULES.add("time")'),
                                                     ("sandbox", "is not a module: it is already defined"),
                                                     ("duck", 'env["ARC2_SANDBOX_SYS_SHIM"] = "1"')]),
    (15, "server watchdog + branches wait while down", [("nb", "def _e15b_watchdog():"), ("nb", "E12A.SERVER_DOWN.set()"),
                                                       ("duck", "if SERVER_DOWN.is_set() and getattr(agent, \"_arc2_b\", False):")]),
    (16, "launch fallback 16 -> 12 -> W1; gate fallback", [("nb", 'return dict(cfg, streams12=True, mamba_cache=56), "streams12"'),
                                                          ("nb", 'return dict(cfg, w1=True), "w1"'), ("nb", 'srv = E11Server(launch + e11_flags(cfg), env, "try_s12")')]),
    (17, "self-test + tokenizer probe on fixed tasks", [("nb", "    _st_task = E15B_SELFTEST_TASK"), ("nb", "# E15B #17: fixed")]),
    (18, "worker restarts on an error; running count released", [("duck", "return _worker_body()"), ("duck", 'log.warning("%s b%d: publish failed: %s", tid, b, exc)')]),
    (19, "attempt-2 order by class (progress -> nothing -> verified)", [("duck", "return (1, rounds, cls, flag, idx[tid], sub)")]),
    (20, "CPU rule library after every program candidate", [("duck", "def rule_library_outputs(task: dict)"),
                                                            ("duck", "rules = rule_library_outputs(task) if _b_mode() else []")]),
    (21, "defined-functions inventory", [("duck", 'payload["defined_functions"] = sorted(self._kept_functions or {})[:40]')]),
    (22, "bounded search guidance", [("duck", "Bound any search or backtracking")]),
    (23, "no overfit_flags selection (selection code has no overfit_flags)", [("duck", "def select_attempts_r5")]),
    (24, "pip / venv / sysconfig timeouts", [("nb", "check=True, timeout=300"), ("nb", "capture_output=True, text=True, timeout=1800"),
                                            ("nb", "check=True, timeout=120).stdout.strip()")]),
    (25, "keep payload .bin in the dataset (process)", []),
    (26, "submission.json written per branch on purpose (crash safety)", []),
    (27, "staging copy stops at the launch join (>= 10 min floor)", [("nb", "deadline=max(NOTEBOOK_T0 + 10 * 60, min(NOTEBOOK_T0 + 45 * 60, SGL_GIVE_UP_AT - 20 * 60))")]),
    (28, "labels (variant, backend, arm)", [("nb", '"variant": "E15B"'), ("nb", 'f"-s{SGL_MAX_RUNNING}"'), ("nb", "16 streams: mrr 17")]),
    (29, "per-attempt record", [("duck", 'st["b_events"] = a.arc2_b_events'), ("duck", 'st["b_budget"] = a._arc2_b_budget')]),
    (30, "B-risk metric", [("nb", '"at_budget_with_progress"')]),
    (31, "candidate fields", [("duck", "def arc2_b_candidate_fields(self)"), ("duck", 'st["b_candidate"] = a.arc2_b_candidate_fields()')]),
    (32, "server-side generated tokens", [("nb", '"server_gen_tokens"'), ("nb", "def _e15b_gen_tokens_total():")]),
    (33, "retract counter fixed", [("nb", '"retraction_policy" not in l and "server_args" not in l.lower()),')]),
    (34, "accepted vs harvested", [("nb", '"accepted_submit"'), ("nb", '"harvested_only"')]),
    (35, "effective effort", [("nb", '"effective_effort"')]),
    (36, "delayer off", [("nb", '"--min-free-slots-delay": ["1"]')]),
    (37, "16 streams: mrr 17, graphs 1..17, mamba 68", [("nb", "SGL_MAX_RUNNING = 16"), ("nb", "SGL_SERVER_MAX_RUNNING = 17"),
                                                      ("nb", '"mamba_cache": 68'),
                                                      ("nb", 'E14_SUBMIT_ARM = {"--cuda-graph-bs-decode": [str(b) for b in range(1, SGL_SERVER_MAX_RUNNING + 1)]')]),
    (38, "gates allow 16 (KV per request 50k, streams 16)", [("nb", "SGL_KV_PER_REQUEST = 50_000"), ("nb", '"E12_STREAMS", 16')]),
    (39, "real-concurrency check", [("nb", '"server_running_mean"'), ("nb", "E15B WARNING: mean server running")]),
    (40, "KV fit (measured in the smoke)", []),
    (41, "READY time with graphs 1..17 (measured in the smoke)", []),
    (42, "engine block from the server log", [("nb", "def _e15b_server_log_summary(text):"), ("nb", '"running_hist"')]),
    (43, "harness + CPU kit (harness built; CPU kit = Kaggle push)", []),
    (44, "commit profile E15B (user 2026-10-25: 20 eval tasks, both attempts; 115 min, 92-min window)",
     [("nb", '"E15B":      {"source": "eval"'), ("nb", f'"time_budget_sec": {CB} * 60'), ("nb", f'"solve_window_sec": {CW} * 60'),
      ("nb", '"16de56c4", "1ae2feb7", "221dfab4", "269e22fb"]')]),
    (45, "fair share off", [("nb", '"E12_R6_FAIR_SHARE": 0')]),
    (46, "fixed window", [("nb", 'int(E15B["E12_B_WINDOW"])')]),
    (47, "request timeout (rev 3h: 3600 s)", [("nb", '"E12_DUCK_REQUEST_TIMEOUT", 3600')]),
    (48, "commit turn + budget in prompt / opener", [("duck", 'R6_TEXT_B = {'), ("duck", 'state, text = "commit", self._arc2_b_commit_text("opener")'),
                                                    ("duck", 'payload["harness_note"] = self._arc2_b_commit_text("tool")')]),
    (49, "min_admit 300 (Notes/07 A5)", [("nb", '"E12_R6_MIN_ADMIT_SEC": 300')]),
    (50, "reasoning-code harvest", [("duck", "def b_harvest_source"), ("duck", "agent.arc2_b_reasoning_harvest(self.state_path)")]),
    (51, "yield 2,048 kept on purpose (harmless after #10)", []),
    (52, "no new scheduler (rank by class)", [("duck", "cls = stage2_class(task_s, strong_only=weak_commit)")]),   # rev 3k W1 form
    (53, "retract / preempt = 0 check", [("nb", '"retract_lines": _bad("retract"), "preempt_lines": _bad("preempt")')]),
    (54, "deadline harvest grace + join grace", [("duck", '_env_int("E12_B_FINAL_GRACE_SEC", 30)'), ("nb", "join_grace_sec=90.0"),
                                                ("duck", 'getattr(self, "_arc2_verify_grace", 0.0))')]),
    (55, "< 400 kB; traces saved; facts cap (measured: 1,000 training tasks max 800 tokens)", [("nb", "E12_RAW_REASONING_HISTORY")]),
]
NOTE = {25: "process (not code)", 26: "kept on purpose", 40: "GPU smoke", 41: "GPU smoke", 43: "harness = src/harness/harness_e15b_nb.py; CPU kit needs a Kaggle push",
        51: "kept on purpose"}
# Notes/07: the adopt list from the two AI proposals (A7 = the harness fault scenarios, checked by the harness runs)
ADOPT = [
    ("A1", "running histogram with queue > 0 + server running while every client stream is busy",
     [("nb", '"running_hist_queue_gt0": dict(sorted(_hq.items()))'), ("nb", '"server_running_mean_client_full"'),
      ("nb", '"server_at_streams_share_client_full"')]),
    ("A2", "per-attempt correctness in the eval score", [("nb", 'report["eval_b_attempts"] = dict(_pa, rows=_rows)'),
                                                       ("nb", '"b1_only_acc"'), ("nb", '"b0_budget_partial_b1_acc"')]),
    ("A3", "startup split from the server log", [("nb", '"graph_capture_sec_total"'), ("nb", "Load weight end"),
                                               ("nb", '"ready_span_sec"')]),
    ("A4", "KV stale-sample count", [("duck", 'state["kv_stale_checks"] += 1'), ("nb", '"kv_stale_checks"')]),
    ("A5", "min_admit 300", [("nb", '"E12_R6_MIN_ADMIT_SEC": 300')]),
    ("A6", "rule library inside its own try", [("duck", "# E15B A6 (Exp 15 Notes/07): a rule error must never cost")]),
    ("A7", "harness fault injection (server kill -> relaunch; slow healthy -> no relaunch)", []),
    # Exp 15 Notes/agent audits (2026-10-25): auditor 1 (Duck), 2 (notebook / engine), 3 (regression)
    ("1F1", "partial test before the harvest; harvest restores retained functions", [
        ("duck", "kept0 = dict(self._kept_functions or {})"), ("duck", "# rev 3k ORD1: the harvest FIRST")]),   # rev 3k: order reversed, F1 safety via P1
    ("1F2", "commit text after a rejected submit; harvest is not a commit", [
        ("duck", "Auditor 1 F2: a rejected submit inside the commit zone"), ("duck", 'not getattr(self, "_arc2_b_harvesting", False)')]),
    ("1F3", "sandbox timeout clamped to the deadline + grace", [("duck", "_tmo = max(1, min(self._python_timeout, int(_left) - 1))")]),
    ("3R9", "harvest imports inside try", [("duck", "Auditor 3 R9: a blocked import")]),
    ("3L41", "unknown tool with code -> python", [("duck", "Auditor 3 L41")]),
    ("2F1", "no submission rewrite after the solve end; rule cache", [("nb", 'E12_RUN["late_writes_skipped"]'),
                                                                      ("nb", "E15B_SOLVE_END_PC = _e15b_solve_end"),
                                                                      ("duck", "_RULE_CACHE: dict = {}")]),
    ("2F2", "watchdog liveness = log growth, /health only when silent", [("nb", "_sz = os.path.getsize(_srv.log_path)")]),
    ("2F3", "relaunch with 12-stream flags after OOM / RuntimeError or on the 2nd; failed relaunch retries", [
        ("nb", '_rec["streams12"] = True'), ("nb", 'if not _rec.get("ok") and len(_wd["relaunches"]) >= 2:')]),
    ("2F4", "RERUN give-up T0 + 4 h, 4 launches, staging join bound", [
        ("nb", "(240 if RERUN else 75) * 60"), ("nb", "for i in range(4):"), ("nb", "NOTEBOOK_T0 + 55 * 60) - time.perf_counter()")]),
    ("2F6", "engine block over every server log; per-server tokens; accept len; tok/s by level", [
        ("nb", "def _e15b_all_server_logs():"), ("nb", '"accept_len_p50"'), ("nb", '"gen_tok_s_p50_by_running"'),
        ("nb", '_wd["tok_acc"] += max(0.0')]),
    ("3L111", "kill() also kills the group of a dead main process", [("nb", "# E15B B4: orphaned TP workers")]),
    ("3R4", "commit window (rev 3h: 92 min; 50K45K: 105 min)", [("nb", f'"solve_window_sec": {CW} * 60')]),
    ("GPT", "GPT final audit blocker: session setup before the slot is taken; failed jobs never requeued", [
        ("duck", "GPT final audit (2026-10-25): the session first"), ("duck", "(tid, 1) not in failed")]),
    ("3J", "rev 3j: attempt 2 of accepted tasks deferred before (R3J2) / after (R3J1) the attempt 3s", [
        ("duck", 'b1_accepted_mode: str = ""'), ("duck", 'if b1_accepted_mode in ("before3", "after3") and cls >= 2:'),
        ("duck", "if prio[0] >= 1.5 and state[\"stage1_left\"] > 0:")]
     + ([("nb", '"E12_B_SKIP_B1_IF_ACCEPTED": 0, "E12_B_EXTRA_ATTEMPTS": 1, "E12_B_MAX_ATTEMPTS": 8, '),
         ("nb", '"E12_B_B1_ACC_MODE": ' + ("1" if ("_R3J1_" in NB.name or "_R3K1_" in NB.name) else "2")),
         ("nb", 'b1_accepted_mode={1: "after3", 2: "before3"}[int(E15B["E12_B_B1_ACC_MODE"])]'),
         ("nb", '"E15BEVAL120R3I": {"source": "eval", "n_tasks": 120, "time_budget_sec": 6.0 * 3600'),
         ("nb", '"b1_deferred": E12_RUN["rev6_scheduler"].get("b1_deferred")')]
        if any(x in NB.name for x in ("_R3J1_", "_R3J2_", "_R3K1_", "_R3K2_")) else [])),
    ("3K", "rev 3k: gate on attempt-1 starts, stage tags, stage2 errors, harvest bound, cut-code prefix, partial best source, "
           "tie-break, harvest-first, no tiny request, W1 weak commit accepts", [
        ("duck", 'if prio[0] >= 1.5 and state["stage1_left"] > 0:'),
        ("duck", 'tag = {"stage": (prio[0] + 1) if prio[0] <= 1 else (2 if b == 1 else 3)}'),
        ("duck", 'state["stage2_errors"] = state.get("stage2_errors", 0) + 1'),
        ("duck", "# rev 3k H1 (GPT audit 1 F1): bound by whole definitions"),
        ("duck", "self._arc2_b_cut_code.append(_cc[:24000])"),
        ("duck", "self._arc2_best_kept = dict(self._kept_functions or {})"),
        ("duck", "-min(int(s.get(\"branch\") or 0) for s in kv[1]),"),
        ("duck", "# rev 3k ORD1: the harvest FIRST"),
        ("duck", "agent._arc2_b_left() < 768"),
        ("duck", 'v["weak"] = True'), ("duck", "stage2_class(task_s, strong_only=weak_commit)")]
     + ([("nb", '"E12_B_WEAK_COMMIT": 1'), ("nb", 'weak_commit=bool(E15B["E12_B_WEAK_COMMIT"])'),
         ("nb", '"weak_accepts": E12_RUN["rev6_scheduler"].get("weak_accepts")')]
        if ("_R3K1_" in NB.name or "_R3K2_" in NB.name) else [])),
    ("3I", "rev 3i: R (no attempt 2 after an accepted attempt 1) + stage 3 (attempts 3..8 on unsolved tasks)", [
        ("duck", "skip_b1_if_accepted: bool = False"), ("duck", 'if b1_accepted_mode == "skip" and cls >= 2:'),   # rev 3j form
        ("duck", "push((2, nb, idx[tid], 0), tid, nb, \"start1\")"),
        ("duck", "if prio[0] >= 1.5 and state[\"stage1_left\"] > 0:")]
     + ([("nb", '"E12_B_SKIP_B1_IF_ACCEPTED": 1, "E12_B_EXTRA_ATTEMPTS": 1, "E12_B_MAX_ATTEMPTS": 8'),
         ("nb", 'skip_b1_if_accepted=bool(E15B["E12_B_SKIP_B1_IF_ACCEPTED"])'),
         ("nb", '"E15BEVAL120R3I": {"source": "eval", "n_tasks": 120, "time_budget_sec": 6.0 * 3600'),
         ("nb", 'os.environ.get("Q27X_PROFILE", "E15BEVAL120R3I")'),
         ("nb", '"extra_started": E12_RUN["rev6_scheduler"].get("extra_started")')] if "_R3I_" in NB.name else [])),
    ("3H2", "rev 3h: nudge scaled to the budget; fair-share cap per session; setup_failed / cut code in b_summary", [
        ("duck", "int(0.69 * self._arc2_b_budget)"), ("duck", "cap = s.max_generated_tokens or max_generated_tokens or 0"),
        ("nb", '"setup_failed": E12_RUN["rev6_scheduler"].get("setup_failed")'), ("nb", '"cut_code_saved": sum(')]),
    ("3H", "rev 3h: code of a tool call cut at the output limit kept and harvested (sandbox + host verifier)", [
        ("duck", "def b_cut_call_code(raw) -> str:"), ("duck", "def b_complete_prefix(code: str"),
        ("duck", 'self.arc2_b_events["cut_code_saved"] = len(self._arc2_b_cut_code)'), ("duck", 'out["cut_code_blocks"]')]),
    ("3F", "rev 3f: per-attempt budgets (agent, prompt, session cap, report)", [
        ("duck", "def b_attempt_budget(branch: int) -> int:"), ("duck", "max_generated_tokens_b1 if (b >= 1"),
        ("nb", f'"E12_B_ATTEMPT_TOKENS_B1": {A2}'), ("nb", 'max_generated_tokens_b1=E15B["E12_B_ATTEMPT_TOKENS_B1"]'),
        ("nb", f'E12_MAX_GEN", {A1} if _E12_REV4'), ("nb", "_bud = lambda r:")]),
]
ok_all = True
print(f"notebook {NB.name}  sha {hashlib.sha256(NB.read_bytes()).hexdigest()[:16]}  {NB.stat().st_size / 1e3:.0f} kB  "
      f"| package {PAY.name}  manifest mismatches {bad_man}")
for n, what, ev in ITEMS:
    miss = [f"{w}: {s[:70]}" for w, s in ev if s not in WHERE[w]]
    status = "NOTE" if not ev else ("OK  " if not miss else "MISS")
    ok_all &= not miss
    print(f"{n:3d} {status} {what}" + (f"  [{NOTE[n]}]" if n in NOTE else "") + ("" if not miss else f"  -> missing {miss}"))
for n, what, ev in ADOPT:
    miss = [f"{w}: {x[:70]}" for w, x in ev if x not in WHERE[w]]
    ok_all &= not miss
    print(f"{n:>3} {'NOTE' if not ev else ('OK  ' if not miss else 'MISS')} {what}"
          + ("  [harness_e15b_nb.py --fault kill / --fault slow]" if not ev else "") + ("" if not miss else f"  -> missing {miss}"))
print("size < 400 kB:", NB.stat().st_size < 400_000)
print("ALL CODE ITEMS PRESENT" if ok_all and not bad_man else "SOMETHING MISSING")
