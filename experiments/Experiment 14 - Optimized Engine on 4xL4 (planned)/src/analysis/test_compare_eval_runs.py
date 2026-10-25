"""Unit tests for compare_eval_runs.py (stdlib unittest; synthetic fixtures in a temp dir).

    python -I test_compare_eval_runs.py
Covers what the two real test folders do not: a gate-4b window, E16 counters / nudge reasons, a never-reached task,
a zip-only layout, log issue filtering (Triton 'avoid CUDA OOM' / sitecustomize / server_args excluded), and a folder
with nothing in it.
"""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_eval_runs as C  # noqa: E402

T0 = 1792800000.0       # 2026-10-24 06:40:00 UTC
STAMP = lambda dt: __import__("datetime").datetime.fromtimestamp(T0 + dt, __import__("datetime").timezone.utc) \
    .strftime("%Y-%m-%d %H:%M:%S")


def ev(**kw):
    base = {"first_tool_tokens": 80, "first_check_tokens": None, "first_transform_tokens": None,
            "provisional_nudge_tokens": None, "provisional_nudge_via": None, "provisional_nudge_seen_tokens": None,
            "first_transform_after_nudge": None, "first_check_after_nudge": None, "requests_started": 3,
            "tool_calls": 4, "length_cut_no_call": 0, "infra_errors": 0,
            "e16_inspection_calls": 0, "e16_nudge_reason": None, "e16_raw_content_kept": 2,
            "e16_raw_content_same": 1, "e16_raw_content_mismatch": 0, "e16_raw_content_markup": 0}
    base.update(kw)
    return base


def branch(task, b, gen, status, accepted=False, runs=None, **evkw):
    return {"task": task, "branch": b, "accepted": accepted, "turns": 2, "failures": 0, "note": "",
            "generated_tokens": gen, "status": status, "runs": runs or [{"stage": 1, "pause_at": 28000, "tokens": gen,
                                                                         "sec": 100.0, "status": status}],
            "usage": {"requests": 3, "prompt_tokens": 1000, "cached_tokens": 800, "cached_pct": 80.0},
            "rev5_events": {"checks_signalled": 1 if evkw.get("first_check_tokens") else 0},
            "rev6_events": ev(**evkw)}


TRANSCRIPT_OK = """--- analysis_step=1 | action=1 | 06:40:00 | tool-agent ---
[SYSTEM PROMPT]
write `transform(grid)` and run `check(transform)`
[THINKING]
hmm
[TOOL CALL: python]
def transform(grid):
    return grid
print(check(transform))
[TOOL RESULT: python]
ok
[ANALYZER STATUS]
completed
"""
TRANSCRIPT_CUT = """--- analysis_step=1 | action=1 | 06:40:00 | tool-agent ---
[SYSTEM PROMPT]
write `transform(grid)` and run `check(transform)`
[TOOL CALL: python]
print(train[0])
[TOOL RESULT: python]
[[1]]
[ANALYZER STATUS]
request_timeout (yielding, history preserved): HTTPConnectionPool(host='127.0.0.1', port=30000): Read timed out.
"""


def server_log():
    lines = ["Error in sitecustomize; set PYTHONVERBOSE for traceback:",
             "ModuleNotFoundError: No module named 'wrapt'",
             f"[{STAMP(-50)}] server_args={{'retraction_policy': 'length'}}"]
    for i in range(0, 300, 4):
        run = 12 if i % 8 else 11
        lines.append(f"[{STAMP(i)} TP0] Decode batch, #running-req: {run}, #full token: 1000, full token usage: 0.10, "
                     f"mamba num: 30, mamba usage: 0.50, accept len: 4.{i % 5}0, accept rate: 0.60, cuda graph: True, "
                     f"gen throughput (token/s): {400 + i % 3}.00, #queue-req: 0")
        if i % 40 == 0:
            lines.append(f"[{STAMP(i + 1)} TP0] Prefill batch, #new-seq: 1, #new-token: 500, #cached-token: 1500, full "
                         f"token usage: 0.10, mamba usage: 0.50, #running-req: 11, #queue-req: 0, #pending-token: 0, "
                         f"cuda graph: False, input throughput (token/s): 100.00")
    lines.append(f"[{STAMP(30)} TP0] Triton kernel 'x' device-loaded after serving started. Pre-load it during engine "
                 f"init to avoid CUDA OOM.")
    lines.append(f"[{STAMP(31)} TP1] Decode batch, #running-req: 1, #full token: 1, full token usage: 0.0, accept len: "
                 f"9.0, gen throughput (token/s): 1.0, #queue-req: 0")         # TP1 lines are ignored
    lines.append(f"[{STAMP(200)} TP0] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.5")
    lines.append(f"[{STAMP(5000)} TP0] Scheduler hit an exception: Traceback (most recent call last):")
    return "\n".join(lines) + "\n"


def make_g4b_run(root: Path, e16=True, zip_only=False):
    tasks = ["aaaa0001", "bbbb0002", "cccc0003"]
    br = [
        # solved: accepted after an E16 early nudge at 12.5k
        branch("aaaa0001", 0, 20000, "done", accepted=True, first_check_tokens=15000, first_transform_tokens=15000,
               provisional_nudge_tokens=12500, provisional_nudge_via="tool", provisional_nudge_seen_tokens=12600,
               first_transform_after_nudge=15000, first_check_after_nudge=15000, e16_inspection_calls=2,
               e16_nudge_reason="inspection"),
        branch("aaaa0001", 1, 5000, "done", first_check_tokens=4000, first_transform_tokens=4000),
        # 24k nudge, no transform, paused never resumed
        branch("bbbb0002", 0, 30000, "paused", provisional_nudge_tokens=24100, provisional_nudge_via="opener",
               e16_inspection_calls=1, e16_nudge_reason="existing_24k"),
        # b1 started, request cut at the deadline, 0 tokens counted
        branch("bbbb0002", 1, 0, "done", requests_started=1, first_tool_tokens=None),
    ]
    if not e16:
        for b in br:
            for k in [k for k in b["rev6_events"] if k.startswith("e16_")]:
                del b["rev6_events"][k]
    recs = [{"task": "aaaa0001", "query": 0, "hit": True, "pass1": True, "pass2": False, "accepted_programs": 1},
            {"task": "aaaa0001", "query": 1, "hit": True, "pass1": False, "pass2": True, "accepted_programs": 1},
            {"task": "bbbb0002", "query": 0, "hit": False, "pass1": False, "pass2": False},
            {"task": "cccc0003", "query": 0, "hit": False, "pass1": False, "pass2": False}]
    rep = {"run_id": "TEST", "profile": "E14SEVAL120G4B", "configuration": {
        "REASONING_EFFORT": "xhigh", "profile_row": {"task_list": tasks},
        "e12": {"E12_REASONING_EFFORT": "", "E12_MAX_GEN_PER_BRANCH": 100000,
                "rev6": {"E12_R6_FIRST_EXPOSURE": 28000, "E12_R6_NUDGE_TOKENS": 24000}}},
        "engine": {"backend": "sglang-test", "kv_tokens": 900000},
        "e12_run": {"streams": 12, "rev6_workers": 12, "deadline_in_sec": 534, "started_sec": 1500,
                    "branches_started": 4, "error": None,
                    "g4b": {"hidden_solve_end_sec": 42720, "solve_start_sec": 1500, "hidden_tasks": 240,
                            "sec_per_task": 171.8, "tasks": 3, "window_sec": 515}},
        "branches": br, "usage_all": {"completion_tokens": 55000, "cached_pct": 80.0},
        "tasks_planned": 3, "queries_planned": 4, "eval_query_hits": 2, "eval_query_total": 4, "eval_pass1_hits": 1,
        "eval_pass2_hits": 1, "eval_task_norm_pct": 33.33, "eval_unit_records": recs,
        "rev6": {"metrics": {"nudge": {"branches": 2}, "checked_branches": 2}},
        "e11": {"server_log": "/kaggle/working/q27x/sglang_server_try0.log", "decode_summary": {"retract_lines": 1}}}
    results = {"build": {"variant": "E14S", "arc2_duck_sha": "f" * 64, **({"e16_adapter_overlay": True} if e16 else {})}}
    files = {"q27x/e12_run_report.json": json.dumps(rep), "e14_results.json": json.dumps(results),
             "e14_state.json": json.dumps({"terminal_reason": "done", "current": "REPORT"})}
    starts = {("aaaa0001", 0): 0.5, ("aaaa0001", 1): 200.0, ("bbbb0002", 0): 0.7, ("bbbb0002", 1): 300.0}
    for b in br:
        t, k = b["task"], b["branch"]
        st = starts[(t, k)]
        reqs = [] if b["generated_tokens"] == 0 else [{"t": T0 + st, "sec": 50.0}, {"t": T0 + st + 60, "sec": 40.0}]
        res = {"task_id": t, "branch": k, "t0": T0 + st, "t1": T0 + 520, "generated_tokens": b["generated_tokens"],
               "status": b["status"], "accepted": b["accepted"], "requests": reqs, "checked": [],
               "progress": {"runs": 1 if b["rev6_events"]["first_check_tokens"] else 0},
               "raw_reasoning": {"kept": 1, "same": 0, "mismatch": 0, "markup": 0}}
        files[f"q27x/duck/{t}_b{k}/branch_result.json"] = json.dumps(res)
        cut = (t, k) in (("bbbb0002", 1),)
        files[f"q27x/duck/{t}_b{k}/transcript.txt"] = TRANSCRIPT_CUT if cut else (
            TRANSCRIPT_OK if b["rev6_events"]["first_check_tokens"] else TRANSCRIPT_CUT.replace("request_timeout", "ok"))
    log = server_log()
    root.mkdir(parents=True, exist_ok=True)
    if zip_only:
        with zipfile.ZipFile(root / "Q27X_E12_DUCK_BUNDLE.zip", "w") as z:
            for k, v in files.items():
                z.writestr(k.replace("q27x/", ""), v)
            z.writestr("sglang_server_try0.log", log)
    else:
        for k, v in files.items():
            p = root / k
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(v, encoding="utf-8")
        with zipfile.ZipFile(root / "Q27X_E14A_BUNDLE.zip", "w") as z:
            z.writestr("logs/sglang_server_try0.log", log)
            z.writestr("logs/sglang_server_other.log", f"[{STAMP(-9000)} TP0] Decode batch, #running-req: 3, #full token: 1, "
                                                       f"full token usage: 0.0, accept len: 2.0, gen throughput "
                                                       f"(token/s): 9.0, #queue-req: 0\n")
    return root


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.a = make_g4b_run(base / "A")
        cls.b = make_g4b_run(base / "B", e16=False, zip_only=True)
        cls.empty = base / "EMPTY"
        cls.empty.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_a(self):
        return C.analyse_run(self.a, "A", None)

    def test_score(self):
        s = self.run_a()["score"]
        self.assertEqual((s["query_hits"], s["query_total"], s["attempt2_only_hits"]), (2, 4, 1))
        self.assertAlmostEqual(s["task_norm_pct"], 33.33, places=2)
        self.assertEqual(s["tasks_any_hit"], ["aaaa0001"])

    def test_coverage_and_classes(self):
        r = self.run_a()
        c = r["coverage"]
        self.assertEqual(c["never_reached"], ["cccc0003"])
        self.assertEqual((c["b0_started"], c["b1_started"], c["b1_started_zero_tokens"]), (2, 2, 1))
        cls = {(x["task"], x["branch"]): x["class"] for x in r["branches"]}
        self.assertEqual(cls[("aaaa0001", 0)], "A")
        self.assertEqual(cls[("bbbb0002", 0)], "P")
        self.assertEqual(cls[("bbbb0002", 1)], "M")
        self.assertEqual(cls[("aaaa0001", 1)], "D")
        self.assertEqual(c["exposure_last_task"], "bbbb0002")
        self.assertAlmostEqual(c["exposure_last_task_sec"], 0.2, places=3)
        self.assertEqual(c["total_generated"], 55000)
        self.assertEqual(c["window_planned_sec"], 515)

    def test_behaviour_e16(self):
        b = self.run_a()["behaviour"]
        self.assertEqual(b["nudged_branches"], 2)
        self.assertEqual(b["nudges"]["e16_early"]["branches"], 1)
        self.assertEqual(b["nudges"]["e16_early"]["transform_after"], 1)
        self.assertEqual(b["nudges"]["e16_early"]["task_solved"], 1)
        self.assertEqual(b["nudges"]["24k"]["branches"], 1)
        self.assertEqual(b["e16_inspection_calls_hist"], {"0": 2, "1": 1, "2": 1})
        self.assertEqual(b["checked_branches"], 2)
        self.assertEqual(b["checked_branches_transcript"], 2)
        self.assertEqual(b["first_check_all"]["median_observed"], 9500.0)

    def test_engine_log(self):
        e = self.run_a()["engine"]
        self.assertTrue(e["log_used"].endswith("sglang_server_try0.log"))
        self.assertEqual(set(e["running_hist"]), {11, 12})       # TP1 line ignored, other log not chosen
        self.assertEqual(e["gen_tok_s_p50_full"], 401.0)
        self.assertEqual(e["issues"]["retract"]["count"], 1)     # real retract only, not server_args
        self.assertEqual(e["issues"]["oom"]["count"], 1)         # 'Decode out of memory', not the Triton warning
        self.assertEqual(e["issues"]["traceback"]["in_window"], 0)
        self.assertEqual(e["prefill_cached_pct_log"], 75.0)

    def test_km(self):
        self.assertEqual(C.km_median([(10, True), (20, True), (30, False)]), (20, None))
        self.assertEqual(C.km_median([(10, False), (20, False)]), (None, 20))
        self.assertEqual(C.km_median([]), (None, None))

    def test_projection_g4b(self):
        p = self.run_a()["projection"]
        self.assertTrue(p["g4b"])
        self.assertAlmostEqual(p["coverage_started"], 0.667, places=3)
        self.assertAlmostEqual(p["task_norm_reached_pct"], 50.0)
        self.assertAlmostEqual(p["lb_proxy_pct"], 33.33, places=2)
        self.assertEqual(p["hidden_window_sec"], 41220)

    def test_zip_only_and_compare(self):
        rb = C.analyse_run(self.b, "B", None)
        self.assertEqual(rb["score"]["query_hits"], 2)
        self.assertEqual(rb["behaviour"]["e16_fields_present_branches"], 0)
        self.assertEqual(rb["meta"]["e16_overlay"], False)     # e14_results.json in the zip, no overlay key
        comp, tasks = C.compare([self.run_a(), rb])
        self.assertEqual(comp["union_query_hits"], 2)
        self.assertEqual(comp["pairwise"][0]["both"], 1)

    def test_empty_and_cli(self):
        r = C.analyse_run(self.empty, "E", None)
        self.assertEqual(r["coverage"]["tasks"], 0)
        out = Path(self.tmp.name) / "o.json"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = C.main([str(self.a), str(self.b), str(self.empty), "--labels", "A,B,E", "--out", str(out),
                         "--solutions", str(Path(self.tmp.name) / "none.json")])
        self.assertEqual(rc, 0)
        md = buf.getvalue()
        for sec in ("## 2. Score", "## 3. Coverage", "## 5. Engine", "## 7. Union", "## 8. Projection", "## 9. Data"):
            self.assertIn(sec, md)
        js = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(js["runs"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
