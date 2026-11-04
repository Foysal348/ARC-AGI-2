"""Quick local unit tests for k100q_report.py (stdlib only): python k100q_report_tests.py

1. Verbatim: the copied nb5_lib.py functions / constants are identical to nb5_lib.py (when it sits next to this file).
2. Real data (skipped when the NB5 output is not on this machine): NB5's own FP8 phase fed through
   build_quality_report must reproduce NB5's per-config record exactly (13/15, 18/20, pass1 17 / pass2 1, per-task
   tokens, outcomes, exactness, every paired label both / neither); the BF16 phase reproduces NB5's BF16 record and the
   NB5 FP8 vs BF16 discordant pairs; the reference file equals nb5_report.json; the degraded path (run report only).
3. Edge cases on a synthetic run: a missing task, an attempt not started, an empty run report, all-None inputs,
   malformed records, klee100 solving a W4-unsolved task (McNemar 1.0 / 0.5), accepted != correct, window cuts.
4. File hygiene: the three files are LF-only; the module and the tests are ASCII; the table's only non-ASCII
   characters are the check / cross marks.

Data paths: K100Q_NB5_OUT (default D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x) and
K100Q_SOLUTIONS (default: arc_eval_15tasks_solutions.json inside the NB5 output audit zip).
"""
import copy
import inspect
import json
import os
import sys
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k100q_report as K  # noqa: E402

try:
    import nb5_lib as NB5L  # noqa: E402
except Exception:            # the module must stand alone; the verbatim check needs nb5_lib
    NB5L = None

NB5_OUT = Path(os.environ.get("K100Q_NB5_OUT", "D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x"))
AUDIT_ZIP = HERE.parent.parent / "Experiment Review" / "Audit Zips" / "E25_NB5_QUALITY_output_FLAT_2026-11-01.zip"
REF_FILE = HERE / K.REF_FILE_NAME
OK, BAD = chr(0x2713), chr(0x2717)
VERBATIM_FUNCS = ("_exact_all", "fail_kind", "attempt_end", "attempt_outcome", "qab_tasks", "qab_summary",
                  "mcnemar_exact_p", "qab_pair")
VERBATIM_CONSTS = ("NO_REQUEST_VERDICTS", "FINISHED_OUTCOMES", "BUDGET_END_VERDICTS")


def _solutions():
    p = os.environ.get("K100Q_SOLUTIONS")
    if p and Path(p).exists():
        return json.loads(Path(p).read_text(encoding="utf-8"))
    if AUDIT_ZIP.exists():
        with zipfile.ZipFile(AUDIT_ZIP) as z:
            return json.loads(z.read("arc_eval_15tasks_solutions.json"))
    return None


SOL = _solutions()
HAVE_REAL = bool(SOL) and (NB5_OUT / "nb5_report.json").exists() and (NB5_OUT / "phase_Q_FP8_K3" / "duck").is_dir()
NB5_REPORT = json.loads((NB5_OUT / "nb5_report.json").read_text(encoding="utf-8")) if HAVE_REAL else None


def _jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _engine_full(phase):
    """The engine facts NB5 recorded for this phase (capacity_levers / speed / window / window_plan / steady_mixed)."""
    pc = NB5_REPORT["quality_ab"]["per_config"][phase]
    return {"capacity_levers": pc["capacity_levers"], "speed": pc["speed"], "window": pc["window"],
            "window_plan": pc["window_plan"], "steady_mixed": NB5_REPORT["phases"][phase]["steady_mixed"],
            "kv": pc["kv"], "fp8_status": pc["fp8_status"]}


def _engine_short(phase):
    """Only the facts the klee100 notebook is asked to pass (pool / F / workers / max_running / preemptions / solver
    tok/s / steady p50 / pin / kv_bytes / ladder rung), plus the sampler rows; window from the run report."""
    pc = NB5_REPORT["quality_ab"]["per_config"][phase]
    cl, sp = pc["capacity_levers"], pc["speed"]
    return {"pool": cl["pool_blocks"], "F": cl["f_eff"], "workers": cl["workers"], "max_running": sp["max_running"],
            "preemptions": sp["preemptions"], "solver_tok_s": sp["solver_tok_s"],
            "steady_p50": NB5_REPORT["phases"][phase]["steady_mixed"]["gen_tok_s_p50"], "pin": cl["pin_gib"],
            "kv_bytes": cl["kv_bytes"], "ladder_rung": "merlin_fp8_mtp_pin100",
            "sampler_rows": _jsonl(NB5_OUT / f"phase_{phase}" / "nb5_sampler.jsonl")}


def _phase_inputs(phase):
    d = NB5_OUT / f"phase_{phase}"
    rr = json.loads((d / "e12_run_report.json").read_text(encoding="utf-8"))
    ga = json.loads((d / "e12_generated_attempts.json").read_text(encoding="utf-8"))
    return d, rr, ga


def _strip_offsets(tasks):
    t = copy.deepcopy(tasks)
    for v in t.values():
        for a in v.get("attempts") or []:
            a.pop("end_offset_sec", None)
    return t


# ------------------------------------------------------------------ synthetic run
SYN_GROUPS = {"solid": ["tA"], "sometimes": ["tB"], "never": ["tC"]}
SYN_SOL = {"tA": [[[1]]], "tB": [[[2]], [[3]]], "tC": [[[4]]]}
SYN_ORDER = ["tA", "tB", "tC"]


def _rec(task, branch, t0, t1, tokens, accepted, outs=None, note="", status="done"):
    subs = [{"name": "transform_final", "accepted": bool(accepted), "test_outputs": outs}] if outs is not None else []
    return {"task_id": task, "branch": branch, "t0": t0, "t1": t1, "generated_tokens": tokens, "accepted": accepted,
            "status": status, "note": note, "submissions": subs, "checked": [], "b_events": {}, "b_budget": 104000}


def _syn_run(records):
    """(run_report, generated_attempts) for synthetic records: window 100 s -> 1100 s of the notebook, epochs
    1000 -> 2000; the generated attempts are the first two accepted outputs (else [[0]])."""
    gen = {}
    for t in SYN_ORDER:
        rows = sorted(records.get(t) or [], key=lambda r: r["branch"])
        outs = [s["test_outputs"] for r in rows for s in r["submissions"] if s.get("accepted")]
        q = []
        for qi in range(len(SYN_SOL[t])):
            a1 = outs[0][qi] if len(outs) > 0 else [[0]]
            a2 = outs[1][qi] if len(outs) > 1 else [[0]]
            q.append({"attempt_1": a1, "attempt_2": a2})
        if rows:
            gen[t] = q
    rr = {"order": SYN_ORDER, "e12_run": {"window_shift": {"window_start_sec": 100}, "started_sec": 99,
                                          "planned_solve_end_sec": 1100.0, "solve_end_epoch": 2000.0, "finished_sec": 1101},
          "branches": [{"task": r["task_id"], "branch": r["branch"], "accepted": r["accepted"], "status": r["status"],
                        "note": r["note"], "failures": 0, "generated_tokens": r["generated_tokens"],
                        "sec": round(r["t1"] - r["t0"], 1)} for v in records.values() for r in v]}
    return rr, gen


def _syn_k100_records():
    return {"tA": [_rec("tA", 0, 1000.0, 1100.0, 5000, True, [[[1]]]), _rec("tA", 1, 1100.0, 1300.0, 6000, True, [[[1]]])],
            "tB": [_rec("tB", 0, 1000.0, 1500.0, 30000, True, [[[9]], [[3]]]),          # accepted, query 0 wrong
                   _rec("tB", 1, 1500.0, 1800.0, 103500, False)],                       # budget spent
            "tC": [_rec("tC", 0, 1000.0, 1995.0, 50000, False)]}                        # cut by the window, b1 never ran


def _syn_w4_records():
    return {"tA": [_rec("tA", 0, 1000.0, 1100.0, 4000, True, [[[1]]]), _rec("tA", 1, 1100.0, 1200.0, 4500, True, [[[1]]])],
            "tB": [_rec("tB", 0, 1000.0, 1400.0, 25000, True, [[[2]], [[3]]]), _rec("tB", 1, 1400.0, 1700.0, 20000, True, [[[2]], [[3]]])],
            "tC": [_rec("tC", 0, 1000.0, 1700.0, 103900, False), _rec("tC", 1, 1700.0, 1990.0, 9000, False)]}


def _syn_engine(pool=300, workers=12, tok_s=250.0, p50=300.0):
    return {"pool": pool, "F": 5, "workers": workers, "max_running": workers, "preemptions": 0, "solver_tok_s": tok_s,
            "steady_p50": p50, "pin": 1.0, "kv_bytes": 123, "ladder_rung": "rung0"}


def _syn_ref():
    """A W4 reference built by the same code from the synthetic W4 run (shape = nb5_fp8_reference.json)."""
    w = _syn_w4_records()
    rr, ga = _syn_run(w)
    rec, _inputs, _r = K.build_k100_record(rr, ga, _syn_engine(214, 9, 172.0, 225.0), SYN_SOL,
                                           {"configs": {}, "groups": SYN_GROUPS}, records=w, order=SYN_ORDER)
    return {"primary": "Q_FP8_K3", "secondary": "Q_BF16_K3", "groups": SYN_GROUPS, "fp8_label": "X",
            "source": {"nb5_report": {"sha256": "synthetic"}}, "configs": {"Q_FP8_K3": rec}}


def _syn_report(records, ref=None, **kw):
    rr, ga = _syn_run(records)
    return K.build_quality_report(rr, ga, _syn_engine(), SYN_SOL, ref if ref is not None else _syn_ref(),
                                  records=records, **kw)


# ==================================================================== tests
class VerbatimTests(unittest.TestCase):
    @unittest.skipIf(NB5L is None, "nb5_lib.py not next to this file")
    def test_functions_identical_to_nb5_lib(self):
        for n in VERBATIM_FUNCS:
            a = inspect.getsource(getattr(K, n)).replace("\r\n", "\n")
            b = inspect.getsource(getattr(NB5L, n)).replace("\r\n", "\n")
            self.assertEqual(a, b, n)
        for n in VERBATIM_CONSTS:
            self.assertEqual(getattr(K, n), getattr(NB5L, n), n)

    @unittest.skipIf(NB5L is None, "nb5_lib.py not next to this file")
    def test_mcnemar_same_values(self):
        for b, c in ((0, 0), (1, 0), (2, 0), (1, 1), (3, 1), (5, 0), (6, 2)):
            self.assertEqual(K.mcnemar_exact_p(b, c), NB5L.mcnemar_exact_p(b, c))
        self.assertEqual(K.mcnemar_exact_p(1, 0), 1.0)
        self.assertEqual(K.mcnemar_exact_p(2, 0), 0.5)


@unittest.skipUnless(HAVE_REAL, "NB5 QUALITY output / solutions not on this machine")
class RealDataTests(unittest.TestCase):
    def _run(self, phase, engine, **kw):
        d, rr, ga = _phase_inputs(phase)
        kw.setdefault("phase_dir", d)
        return K.build_quality_report(rr, ga, engine, SOL, None, **kw)

    def test_fp8_reproduces_nb5_exactly(self):
        pc = NB5_REPORT["quality_ab"]["per_config"]["Q_FP8_K3"]
        r = self._run("Q_FP8_K3", _engine_full("Q_FP8_K3"))
        self.assertEqual(r["errors"], [])
        k = r["k100"]
        self.assertEqual(r["notes"]["records_source"], "phase_dir")
        self.assertEqual(k["tasks"], pc["tasks"])                      # every per-task / per-attempt field
        self.assertEqual(k["summary"], pc["summary"])
        self.assertEqual(k["score"], pc["score"])
        for key in ("kv", "status", "fp8_status", "window", "window_plan", "order"):
            self.assertEqual(k[key], pc[key], key)
        self.assertEqual({x: k["capacity_levers"][x] for x in pc["capacity_levers"]}, pc["capacity_levers"])
        self.assertEqual(k["speed"], pc["speed"])
        s = k["summary"]["all"]
        self.assertEqual((s["solved"], s["tasks"], s["query_hits"], s["queries"]), (13, 15, 18, 20))
        self.assertEqual((k["score"]["eval_pass1_hits"], k["score"]["eval_pass2_hits"]), (17, 1))
        self.assertEqual(k["score"]["eval_task_norm_pct"], 86.67)
        self.assertEqual(k["steady_mixed"]["gen_tok_s_p50"], 225.0)
        for t, v in pc["tasks"].items():                                 # per-task tokens (both attempts)
            row = r["paired"]["tasks"][t]
            self.assertEqual(row["k100"]["a1"]["tokens"], v["attempts"][0]["tokens"])
            self.assertEqual(row["k100"]["a2"]["tokens"], v["attempts"][1]["tokens"])
            self.assertEqual(row["k100"]["tokens"], v["tokens"])
            self.assertIn(row["label"], ("both", "neither"))
        c = r["paired"]["counts"]
        self.assertEqual((c["both"], c["k100_only"], c["w4_only"], c["neither"], c["discordant"], c["mcnemar_p"]),
                         (13, 0, 0, 2, 0, 1.0))
        self.assertEqual(sorted(r["paired"]["label_tasks"]["neither"]), ["16b78196", "2b83f449"])
        dl = r["delta"]
        for key in ("solved", "queries", "pass1", "pass2", "task_norm_pct", "tokens_total", "window_tok_s", "steady_p50",
                    "pool_blocks", "workers"):
            self.assertEqual(dl[key]["delta"], 0, key)
        self.assertEqual((dl["window_tok_s"]["k100"], dl["pool_blocks"]["k100"], dl["workers"]["k100"]), (172.0, 214, 9))
        for g, row in dl["tokens_by_group"].items():
            for st, v in row.items():
                self.assertEqual(v["delta"], 0, (g, st))
        tg = dl["tokens_by_group_k100"]                                # the NB5 doc table (E25_GPU_NB5_QUALITY)
        self.assertEqual((tg["solid"]["a1_median"], round(tg["solid"]["task_mean"])), (6344, 14327))
        self.assertEqual((tg["sometimes"]["a1_median"], round(tg["sometimes"]["task_mean"])), (32679, 50089))
        self.assertEqual((tg["never"]["a1_median"], tg["never"]["a1_max"], round(tg["never"]["task_mean"])),
                         (90477, 103761, 106724))
        n = r["notes"]
        self.assertEqual([(x["side"], x["task"], x["attempt"]) for x in n["accepted_not_correct"]],
                         [("k100", "16b78196", 1), ("w4", "16b78196", 1)])
        self.assertEqual(sorted((x["task"], x["attempt"]) for x in n["window_cut"] if x["side"] == "k100"),
                         [("16b78196", 2), ("2b83f449", 2), ("64efde09", 2)])
        self.assertEqual(n["missing_tasks"], [])
        self.assertTrue(n["score_check"]["agree"])
        md = K.markdown_table(r)
        self.assertEqual(sum(1 for x in md.splitlines() if x.startswith("|")), 1 + 1 + 15 + 1)   # header, rule, 15, total
        self.assertIn(f"| 16b78196 | never | 86,756 {BAD} (accepted) | 26,279 incomplete | {BAD} 0/1 |", md)
        self.assertIn(f"| 1818057f | solid | 4,261 {OK} | 4,661 {OK} | {OK} 1/1 |", md)

    def test_fp8_short_engine_and_sampler_rows(self):
        pc = NB5_REPORT["quality_ab"]["per_config"]["Q_FP8_K3"]
        r = self._run("Q_FP8_K3", _engine_short("Q_FP8_K3"))
        k = r["k100"]
        self.assertEqual(k["tasks"], pc["tasks"])                      # window reconstructed from the run report
        self.assertEqual(k["summary"], pc["summary"])
        w = {x: pc["window"][x] for x in ("start_sec", "end_sec", "actual_sec", "solver_returned_sec")}
        self.assertEqual({x: k["window"][x] for x in w}, w)
        sp = k["speed"]
        self.assertEqual((sp["max_running"], sp["max_inflight"], sp["mean_running_busy"], sp["preemptions"],
                          sp["speed_status"], sp["solver_tok_s"]), (9, 9, 6.14, 0, "CLEAN", 172.0))
        cl = k["capacity_levers"]
        self.assertEqual((cl["pool_blocks"], cl["f_eff"], cl["workers"], cl["pin_gib"], cl["kv_bytes"], cl["ladder_rung"]),
                         (214, 6, 9, 1.0, 3553580524, "merlin_fp8_mtp_pin100"))
        self.assertEqual(r["delta"]["steady_p50"]["delta"], 0.0)

    def test_fp8_in_memory_records(self):
        pc = NB5_REPORT["quality_ab"]["per_config"]["Q_FP8_K3"]
        d, rr, ga = _phase_inputs("Q_FP8_K3")
        recs, by, errs = K.load_phase_records(d)
        self.assertEqual(errs, [])
        self.assertEqual((len(recs), sum(len(v) for v in recs.values()), len(by)), (15, 30, 30))
        r = K.build_quality_report(rr, ga, _engine_full("Q_FP8_K3"), SOL, None, records=recs, by_attempt=by)
        self.assertEqual(r["k100"]["tasks"], pc["tasks"])
        self.assertEqual(r["notes"]["records_source"], "records")

    def test_bf16_reproduces_nb5_and_nb5_pairs(self):
        pc = NB5_REPORT["quality_ab"]["per_config"]["Q_BF16_K3"]
        r = self._run("Q_BF16_K3", _engine_full("Q_BF16_K3"))
        k = r["k100"]
        self.assertEqual(_strip_offsets(k["tasks"]), _strip_offsets(pc["tasks"]))
        for t, v in pc["tasks"].items():                                 # window-start epoch is reconstructed: <= 0.1 s
            for a, b in zip(v["attempts"], k["tasks"][t]["attempts"]):
                if a.get("end_offset_sec") is not None:
                    self.assertLessEqual(abs(a["end_offset_sec"] - b["end_offset_sec"]), 0.11)
        self.assertEqual(k["summary"], pc["summary"])
        self.assertEqual(k["score"], pc["score"])
        c = r["paired"]["counts"]                                        # BF16 as "k100" vs FP8 as "w4" = NB5's pairs
        self.assertEqual((c["both"], c["k100_only"], c["w4_only"], c["neither"], c["discordant"], c["mcnemar_p"]),
                         (12, 1, 1, 1, 2, 1.0))
        lt = r["paired"]["label_tasks"]
        self.assertEqual((lt["k100_only"], lt["w4_only"], lt["neither"]), (["16b78196"], ["247ef758"], ["2b83f449"]))
        self.assertEqual(r["delta"]["workers"]["k100"], 5)
        self.assertEqual(r["delta"]["pool_blocks"]["k100"], 112)

    def test_degraded_run_report_only(self):
        pc = NB5_REPORT["quality_ab"]["per_config"]["Q_FP8_K3"]
        d, rr, ga = _phase_inputs("Q_FP8_K3")
        r = K.build_quality_report(rr, ga, _engine_short("Q_FP8_K3"), SOL, None)
        self.assertEqual(r["errors"], [])
        self.assertTrue(r["notes"]["degraded"])
        k = r["k100"]
        for t, v in pc["tasks"].items():
            self.assertEqual(k["tasks"][t]["solved"], v["solved"], t)
            self.assertEqual(k["tasks"][t]["tokens"], v["tokens"], t)
            self.assertEqual([a["tokens"] for a in k["tasks"][t]["attempts"]], [a["tokens"] for a in v["attempts"]])
            self.assertTrue(all(a["exact_accepted"] is None for a in k["tasks"][t]["attempts"]))
        self.assertEqual(k["score"], pc["score"])
        self.assertTrue(any("DEGRADED" in x for x in r["notes"]["lines"]))
        K.markdown_table(r)

    def test_reference_file_equals_nb5_report(self):
        import hashlib
        ref = json.loads(REF_FILE.read_text(encoding="utf-8"))
        raw = (NB5_OUT / "nb5_report.json").read_bytes()
        self.assertEqual(ref["source"]["nb5_report"]["sha256"], hashlib.sha256(raw).hexdigest())
        qa = NB5_REPORT["quality_ab"]
        self.assertEqual(ref["groups"], qa["groups"])
        for ph in ("Q_FP8_K3", "Q_BF16_K3"):
            c = dict(ref["configs"][ph])
            self.assertEqual(c.pop("steady_mixed"), NB5_REPORT["phases"][ph]["steady_mixed"])
            self.assertEqual(c.pop("solver_window"), NB5_REPORT["phases"][ph]["solver_window"])
            self.assertEqual(c, qa["per_config"][ph])
        self.assertLess(REF_FILE.stat().st_size, 60000)
        txt = REF_FILE.read_text(encoding="utf-8")
        for bad in ('"reasoning"', '"content"', '"messages"', '"raw_reasoning"', '"usage"'):
            self.assertNotIn(bad, txt)

    def test_reference_from_whole_nb5_report(self):
        d, rr, ga = _phase_inputs("Q_FP8_K3")
        r = K.build_quality_report(rr, ga, _engine_full("Q_FP8_K3"), SOL, NB5_REPORT, phase_dir=d)
        self.assertEqual(r["paired"]["counts"]["both"], 13)
        self.assertEqual(r["ref"]["w4_fp8"]["summary"], NB5_REPORT["quality_ab"]["per_config"]["Q_FP8_K3"]["summary"])

    def test_score_from_generated_matches_run_report(self):
        for ph in ("Q_FP8_K3", "Q_BF16_K3"):
            d, rr, ga = _phase_inputs(ph)
            recs, _by, _e = K.load_phase_records(d)
            score, units = K.score_from_generated(ga, SOL, rr["order"], recs)
            self.assertEqual(score, {k: rr[k] for k in K.SCORE_KEYS}, ph)
            key = lambda u: (u["task"], u["query"])
            self.assertEqual(sorted(units, key=key), sorted(rr["eval_unit_records"], key=key), ph)

    def test_real_k100_solves_w4_unsolved(self):
        d, rr, ga = _phase_inputs("Q_FP8_K3")
        recs, by, _e = K.load_phase_records(d)
        recs = copy.deepcopy(recs)
        ga, rr = copy.deepcopy(ga), copy.deepcopy(rr)
        gold = SOL["2b83f449"]
        b0 = [x for x in recs["2b83f449"] if x["branch"] == 0][0]
        b0["accepted"] = True
        b0["submissions"] = [{"name": "transform_final", "accepted": True, "test_outputs": gold}]
        ga["2b83f449"] = [{"attempt_1": g, "attempt_2": g} for g in gold]
        for k_ in K.SCORE_KEYS + ("eval_unit_records",):                 # force the score from the edited submission
            rr.pop(k_, None)
        r = K.build_quality_report(rr, ga, _engine_full("Q_FP8_K3"), SOL, None, records=recs, by_attempt=by)
        c = r["paired"]["counts"]
        self.assertEqual((c["both"], c["k100_only"], c["w4_only"], c["neither"], c["mcnemar_p"]), (13, 1, 0, 1, 1.0))
        self.assertEqual(r["paired"]["tasks"]["2b83f449"]["label"], "k100_only")
        self.assertEqual(r["delta"]["solved"]["delta"], 1)
        self.assertEqual(r["k100"]["tasks"]["2b83f449"]["attempts"][0]["outcome"], "ACCEPTED")


class EdgeTests(unittest.TestCase):
    def test_synthetic_basic(self):
        r = _syn_report(_syn_k100_records())
        self.assertEqual(r["errors"], [])
        k = r["k100"]["tasks"]
        self.assertEqual([a["outcome"] for a in k["tA"]["attempts"]], ["ACCEPTED", "ACCEPTED"])
        self.assertEqual([a["outcome"] for a in k["tB"]["attempts"]], ["ACCEPTED", "BUDGET_END"])
        self.assertEqual([a["outcome"] for a in k["tC"]["attempts"]], ["INCOMPLETE_WINDOW", "NOT_STARTED"])
        self.assertEqual((k["tB"]["query_hits"], k["tB"]["queries"], k["tB"]["solved"]), (1, 2, False))
        self.assertIs(k["tB"]["attempts"][0]["exact_accepted"], False)
        self.assertEqual(r["paired"]["tasks"]["tA"]["label"], "both")
        self.assertEqual(r["paired"]["tasks"]["tB"]["label"], "w4_only")
        self.assertEqual(r["paired"]["tasks"]["tC"]["label"], "neither")
        self.assertEqual(r["paired"]["counts"]["mcnemar_p"], 1.0)
        n = r["notes"]
        self.assertEqual([(x["side"], x["task"], x["attempt"]) for x in n["accepted_not_correct"]], [("k100", "tB", 1)])
        self.assertEqual(sorted((x["side"], x["task"], x["attempt"]) for x in n["window_cut"]),
                         [("k100", "tC", 1), ("w4", "tC", 2)])
        self.assertEqual(r["delta"]["solved"], {"k100": 1, "w4": 2, "delta": -1, "ratio": 0.5})
        self.assertEqual(r["delta"]["pool_blocks"]["delta"], 300 - 214)
        self.assertEqual(r["delta"]["tokens_by_group"]["never"]["a1_median"]["k100"], 50000.0)
        md = K.markdown_table(r)
        self.assertIn("| tC | never | 50,000 incomplete | not started |", md)
        self.assertIn(f"30,000 {BAD} (accepted) | 103,500 {BAD} | {BAD} 1/2", md)

    def test_missing_task(self):
        recs = _syn_k100_records()
        del recs["tA"]
        r = _syn_report(recs)
        self.assertEqual(r["errors"], ["MISSING_GENERATED:tA"])                 # audit F validity check
        v = r["k100"]["tasks"]["tA"]
        self.assertEqual([a["outcome"] for a in v["attempts"]], ["NOT_STARTED", "NOT_STARTED"])
        self.assertFalse(v["solved"])
        self.assertEqual(r["notes"]["missing_tasks"], ["tA"])
        self.assertEqual(r["paired"]["tasks"]["tA"]["label"], "w4_only")
        self.assertIn("| tA | solid | not started | not started |", K.markdown_table(r))
        self.assertEqual(r["k100"]["summary"]["all"]["attempts_not_started"], 3)

    def test_attempt_not_started(self):
        recs = _syn_k100_records()
        recs["tA"] = [x for x in recs["tA"] if x["branch"] == 0]
        r = _syn_report(recs)
        a = r["k100"]["tasks"]["tA"]["attempts"]
        self.assertEqual([x["outcome"] for x in a], ["ACCEPTED", "NOT_STARTED"])
        self.assertTrue(r["k100"]["tasks"]["tA"]["solved"])
        self.assertEqual(r["paired"]["tasks"]["tA"]["k100"]["a2"]["tokens"], 0)
        self.assertIn(f"| tA | solid | 5,000 {OK} | not started | {OK} 1/1 |", K.markdown_table(r))

    def test_empty_run_report(self):
        ref = _syn_ref()
        r = K.build_quality_report({}, {}, {}, SYN_SOL, ref)
        self.assertEqual(r["errors"], ["MISSING_GENERATED:tA", "MISSING_GENERATED:tB", "MISSING_GENERATED:tC"])   # audit F
        k = r["k100"]
        self.assertEqual(k["status"], "NO_RUN_REPORT")
        self.assertEqual(k["summary"]["all"]["solved"], 0)
        self.assertTrue(all(a["outcome"] == "NOT_STARTED" for v in k["tasks"].values() for a in v["attempts"]))
        self.assertEqual(r["notes"]["missing_tasks"], SYN_ORDER)
        self.assertEqual(r["paired"]["counts"]["w4_only"], 2)
        self.assertIn("not started", K.markdown_table(r))

    def test_all_none_inputs(self):
        r = K.build_quality_report(None, None, None, None, None)      # loads nb5_fp8_reference.json next to the module
        self.assertIsInstance(r, dict)
        for key in ("k100", "ref", "paired", "delta", "notes"):
            self.assertIn(key, r)
        K.markdown_table(r)
        K.markdown_table({})
        K.markdown_table(None)
        r2 = K.build_quality_report(None, None, None, None, {"configs": {}})
        self.assertIn("NB5 FP8 reference missing", " ".join(r2["notes"]["lines"]))

    def test_malformed_inputs(self):
        recs = _syn_k100_records()
        rr, ga = _syn_run(recs)
        recs["tA"].append("garbage")
        recs["tA"][0] = dict(recs["tA"][0], t0="abc", generated_tokens="x")
        recs["tB"][0] = dict(recs["tB"][0], submissions="not a list")
        recs["zz"] = [{"branch": "bad"}]
        ga["tB"] = [{"attempt_1": [[2]]}]                              # attempt_2 missing, one query missing
        r = K.build_quality_report(rr, ga, {"preemptions": "n/a", "sampler_rows": [1, None, {"running": "x"}]},
                                   {"tA": [[[1]]], "tB": [[[2]], [[3]]], "tC": "bad"}, _syn_ref(), records=recs)
        self.assertEqual(r["errors"], ["MISSING_GOLD:tC", "QUERY_COUNT_MISMATCH:tB:generated=1,gold=2", "INVALID_ATTEMPTS:tB:0"])
        self.assertEqual(r["k100"]["speed"]["speed_status"], "UNKNOWN")
        K.markdown_table(r)
        json.dumps(r)                                                   # the report is JSON-serialisable

    def test_k100_solves_w4_unsolved(self):
        recs = _syn_k100_records()
        recs["tC"] = [_rec("tC", 0, 1000.0, 1600.0, 70000, True, [[[4]]])]
        r = _syn_report(recs)
        self.assertEqual(r["paired"]["tasks"]["tC"]["label"], "k100_only")
        c = r["paired"]["counts"]
        self.assertEqual((c["k100_only"], c["w4_only"], c["discordant"], c["mcnemar_p"]), (1, 1, 2, 1.0))
        recs["tB"] = [_rec("tB", 0, 1000.0, 1500.0, 30000, True, [[[2]], [[3]]])]
        r = _syn_report(recs)
        c = r["paired"]["counts"]
        self.assertEqual((c["both"], c["k100_only"], c["w4_only"], c["neither"]), (2, 1, 0, 0))
        self.assertEqual(c["mcnemar_p"], 1.0)
        ref = _syn_ref()                                                # W4 unsolved on tB and tC -> 2 discordant
        ref["configs"]["Q_FP8_K3"]["tasks"]["tB"]["solved"] = False
        r = _syn_report(recs, ref=ref)
        c = r["paired"]["counts"]
        self.assertEqual((c["k100_only"], c["w4_only"], c["discordant"], c["mcnemar_p"]), (2, 0, 2, 0.5))
        self.assertEqual(sorted(r["paired"]["label_tasks"]["k100_only"]), ["tB", "tC"])

    def test_degraded_synthetic(self):
        rr, ga = _syn_run(_syn_k100_records())
        r = K.build_quality_report(rr, ga, _syn_engine(), SYN_SOL, _syn_ref())
        self.assertTrue(r["notes"]["degraded"])
        self.assertEqual(r["k100"]["tasks"]["tA"]["attempts"][0]["wall_sec"], 100.0)
        self.assertIn("accepted", K.markdown_table(r))


class HygieneTests(unittest.TestCase):
    def test_lf_and_ascii(self):
        for name in ("k100q_report.py", "k100q_report_tests.py", K.REF_FILE_NAME):
            b = (HERE / name).read_bytes()
            self.assertNotIn(b"\r", b, name)
            self.assertTrue(all(x < 128 for x in b), name)

    def test_table_only_marks_non_ascii(self):
        r = _syn_report(_syn_k100_records())
        md = K.markdown_table(r)
        self.assertEqual({ch for ch in md if ord(ch) > 127}, {OK, BAD})

    def test_output_keys(self):
        r = _syn_report(_syn_k100_records())
        self.assertEqual(sorted(r), sorted(["phase", "ref_phase", "labels", "k100", "ref", "paired", "paired_vs_bf16",
                                            "delta", "notes", "inputs", "errors",
                                            # E26 30-task extension (2026-11-01): scope, attempt 1, all-task summary, 27B columns
                                            "scope", "attempt1_vs_w4", "run_summary", "vs_27b",
                                            "paired_at_matched_4050s"]))   # audit E (2026-11-01)
        self.assertEqual(r["phase"], "Q_K100_FP8_K3")
        for key in ("kv", "status", "fp8_status", "window", "window_plan", "capacity_levers", "speed", "order", "score",
                    "tasks", "summary"):
            self.assertIn(key, r["k100"])



# ==================================================================== E26 30-task extension (2026-11-01)
SYN_NEW = ["tD", "tE"]                                     # tasks the W4 reference never ran (groups hard_long / sometimes_new)
SYN_GROUP_OF = {"tA": "solid", "tB": "sometimes", "tC": "never", "tD": "hard_long", "tE": "sometimes_new"}
SYN_SOL30 = dict(SYN_SOL, tD=[[[7]]], tE=[[[8]]])
SYN_ORDER30 = SYN_ORDER + SYN_NEW


def _syn_k100_30():
    r = _syn_k100_records()
    r["tD"] = [_rec("tD", 0, 1000.0, 1600.0, 90000, True, [[[7]]]), _rec("tD", 1, 1600.0, 1900.0, 40000, False)]   # solved
    r["tE"] = [_rec("tE", 0, 1000.0, 1200.0, 20000, True, [[[1]]])]                                                 # wrong
    return r


def _syn_run30(records):
    rr, gen = _syn_run({t: v for t, v in records.items() if t in SYN_ORDER})
    rr["order"] = list(SYN_ORDER30)
    for t in SYN_NEW:
        rows = sorted(records.get(t) or [], key=lambda r: r["branch"])
        outs = [s["test_outputs"] for r in rows for s in r["submissions"] if s.get("accepted")]
        if rows:
            gen[t] = [{"attempt_1": outs[0][0] if outs else [[0]], "attempt_2": outs[1][0] if len(outs) > 1 else [[0]]}]
        rr["branches"] += [{"task": r["task_id"], "branch": r["branch"], "accepted": r["accepted"], "status": r["status"],
                            "note": r["note"], "failures": 0, "generated_tokens": r["generated_tokens"],
                            "sec": round(r["t1"] - r["t0"], 1)} for r in rows]
    return rr, gen


SYN_REF27 = {"source": {"r3k2_note": "synthetic"}, "tasks": {
    "tA": {"group": "solid", "queries": 1, "query_hits": 1, "solved": True,
           "branches": [{"branch": 0, "generated_tokens": 3000, "accepted": True}], "arcprize_xhigh": 1},
    "tB": {"group": "sometimes", "queries": 2, "query_hits": 1, "solved": False, "branches": [], "arcprize_xhigh": 1},
    "tC": {"group": "never", "queries": 1, "query_hits": 0, "solved": False, "branches": [], "arcprize_xhigh": 0},
    "tD": {"group": "hard_long", "queries": 1, "query_hits": 0, "solved": False,
           "branches": [{"branch": 0, "generated_tokens": 50000, "accepted": False},
                        {"branch": 1, "generated_tokens": 45000, "accepted": False}], "arcprize_xhigh": 1},
    "tE": {"group": "sometimes_new", "queries": 1, "query_hits": 1, "solved": True, "branches": [], "arcprize_xhigh": 0}}}


class Extension30Tests(unittest.TestCase):
    def _r(self, **kw):
        recs = _syn_k100_30()
        rr, ga = _syn_run30(recs)
        return K.build_quality_report(rr, ga, _syn_engine(), SYN_SOL30, _syn_ref(), records=recs, ref27=SYN_REF27, **kw)

    def test_failure_mode_1_no_order_drops_the_new_tasks(self):
        """Review blocker, mode 1 (reproduced): without order= the reference's order (its 15) wins, the new tasks fall out."""
        ref_cfg = _syn_ref()["configs"]["Q_FP8_K3"]
        self.assertEqual(len(K._order(None, ref_cfg, {"order": SYN_ORDER30}, {})), 3)
        r = self._r()
        self.assertEqual(len(r["k100"]["tasks"]), 3)
        self.assertEqual(sorted(r["inputs"]["tasks_outside_order"]), ["tD", "tE"])

    def test_fixed_call_covers_all_tasks(self):
        r = self._r(order=SYN_ORDER30, group_of=SYN_GROUP_OF)
        self.assertEqual(r["errors"], [])
        k = r["k100"]
        self.assertEqual(list(k["tasks"]), SYN_ORDER30)
        self.assertEqual(k["summary"]["all"]["tasks"], 5)
        self.assertEqual((k["summary"]["hard_long"]["solved"], k["summary"]["sometimes_new"]["solved"]), (1, 0))
        self.assertEqual(r["inputs"]["tasks_outside_order"], [])
        self.assertEqual(r["run_summary"]["tasks"], 5)
        self.assertIn("hard_long", r["run_summary"]["tokens_by_group"])

    def test_failure_mode_2_paired_and_delta_only_on_reference_tasks(self):
        """Review blocker, mode 2: with every task in order, a task W4 never ran must not count as 'W4 unsolved'."""
        r = self._r(order=SYN_ORDER30, group_of=SYN_GROUP_OF)
        self.assertEqual(r["scope"]["paired_delta_attempt1_tasks"], 3)
        self.assertEqual(r["scope"]["tasks_not_in_reference"], ["tD", "tE"])
        self.assertEqual(list(r["paired"]["tasks"]), SYN_ORDER)
        c = r["paired"]["counts"]
        self.assertEqual(c["tasks"], 3)
        self.assertNotIn("tD", c["discordant_tasks"] or [])
        base = _syn_report(_syn_k100_records())                 # the same 3 tasks without the new ones
        self.assertEqual(r["paired"]["counts"], base["paired"]["counts"])
        for key in ("solved", "tasks", "tokens_total", "queries", "queries_total", "pass1", "pass2", "task_norm_pct"):
            self.assertEqual(r["delta"][key], base["delta"][key], key)
        self.assertEqual(r["paired_vs_bf16"], base["paired_vs_bf16"])

    def test_attempt1_table(self):
        r = self._r(order=SYN_ORDER30, group_of=SYN_GROUP_OF)
        a = r["attempt1_vs_w4"]
        self.assertEqual(list(a["tasks"]), SYN_ORDER)              # NB5's tasks only
        self.assertEqual(a["tasks"]["tA"]["label"], "both")       # attempt 1 exact on both sides
        self.assertEqual(a["tasks"]["tB"]["label"], "w4_only")    # k100 a1 accepted a wrong query 0; W4 a1 right
        self.assertEqual(a["tasks"]["tC"]["label"], "neither")
        n = a["counts"]
        self.assertEqual((n["both"], n["k100_only"], n["w4_only"], n["neither"], n["discordant"]), (1, 0, 1, 1, 1))
        self.assertEqual((n["k100_a1_tokens"], n["w4_a1_tokens"]), (5000 + 30000 + 50000, 4000 + 25000 + 103900))
        md = K.markdown_extra(r)
        self.assertIn("| tB | sometimes |", md)
        self.assertIn("Attempt 1 only", md)

    def test_vs_27b_columns(self):
        r = self._r(order=SYN_ORDER30, group_of=SYN_GROUP_OF)
        v = r["vs_27b"]
        self.assertEqual(list(v["tasks"]), SYN_ORDER30)
        d = v["tasks"]["tD"]
        self.assertTrue(d["k100"]["solved"])
        self.assertFalse(d["r3k2"]["solved"])
        self.assertEqual(d["arcprize_xhigh"], 1)
        self.assertEqual(d["label_vs_r3k2"], "k100_only")
        self.assertEqual(v["tasks"]["tE"]["label_vs_r3k2"], "r3k2_only")
        g = v["by_group"]
        self.assertEqual((g["hard_long"]["solved_k100"], g["hard_long"]["solved_r3k2"], g["hard_long"]["solved_arcprize_xhigh"]),
                         (1, 0, 1))
        self.assertEqual(g["all"]["tasks"], 5)
        self.assertEqual(v["paired_vs_r3k2"]["all"]["discordant"], 2)          # tD k100 only, tE r3k2 only (tA both)
        md = K.markdown_extra(r)
        self.assertIn("| tD | hard_long |", md)
        self.assertIn("50,000 | 45,000 |", md)

    def test_real_ref27_file(self):
        p = HERE / K.REF27_FILE_NAME
        if not p.exists():
            self.skipTest("k100q_ref27b.json not built")
        d = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(len(d["tasks"]), 30)
        self.assertEqual(list(d["groups"]), ["solid", "sometimes", "never", "hard_long", "sometimes_new"])
        x = d["tasks"]["3dc255db"]
        self.assertEqual(([b["generated_tokens"] for b in x["branches"]], x["solved"], x["arcprize_xhigh"]), ([50000, 44783], False, 1))
        self.assertTrue(d["tasks"]["a395ee82"]["solved"])
        self.assertEqual((d["tasks"]["20270e3b"]["queries"], d["tasks"]["20270e3b"]["query_hits"]), (2, 1))
        self.assertTrue(all(not d["tasks"][t]["solved"] and d["tasks"][t]["arcprize_xhigh"] == 1 for t in d["groups"]["hard_long"]))
        b = p.read_bytes()
        self.assertNotIn(b"\r", b)
        self.assertTrue(all(c < 128 for c in b))

    @unittest.skipUnless(HAVE_REAL, "NB5 output / solutions not on this machine")
    def test_real_nb5_phase_with_extension(self):
        d, rr, ga = _phase_inputs("Q_FP8_K3")
        r = K.build_quality_report(rr, ga, _engine_full("Q_FP8_K3"), SOL, None, phase_dir=d,
                                   order=list(NB5_REPORT["quality_ab"]["per_config"]["Q_FP8_K3"]["order"]))
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["paired"]["counts"]["both"], 13)
        a = r["attempt1_vs_w4"]["counts"]
        self.assertEqual((a["tasks"], a["k100_only"], a["w4_only"]), (15, 0, 0))
        self.assertEqual(a["k100_a1_tokens"], a["w4_a1_tokens"])
        if (HERE / K.REF27_FILE_NAME).exists():
            v = r["vs_27b"]
            self.assertEqual(len(v["tasks"]), 15)
            self.assertEqual(v["missing_in_ref27"], [])

# ==================================================================== audit fixes A / C (2026-11-01)
class AuditFixTests(unittest.TestCase):
    def test_fix_c_missing_query_counts_as_miss(self):
        sc, units = K.score_from_generated({"taskA": [{"attempt_1": [[1]], "attempt_2": [[0]]}]}, {"taskA": [[[1]], [[2]]]}, ["taskA"])
        self.assertEqual((sc["eval_query_hits"], sc["eval_query_total"], sc["eval_task_norm_pct"]), (1, 2, 50.0))
        self.assertEqual([u["hit"] for u in units], [True, False])

    def test_fix_c_task_missing_from_submission(self):
        sc, _u = K.score_from_generated({}, {"taskA": [[[1]], [[2]]]}, ["taskA"])
        self.assertEqual((sc["eval_query_hits"], sc["eval_query_total"], sc["eval_task_norm_pct"]), (0, 2, 0.0))

    def test_tasks_without_gold_are_not_in_the_task_norm(self):
        """Surfaced by fix A on NB5's real data with a 30-task order and 15 solutions: 86.67 vs 43.33 was a silent mismatch."""
        sc, _u = K.score_from_generated({"taskA": [{"attempt_1": [[1]], "attempt_2": [[0]]}]}, {"taskA": [[[1]]]}, ["taskA", "taskB"])
        self.assertEqual((sc["eval_query_hits"], sc["eval_query_total"], sc["eval_task_norm_pct"]), (1, 1, 100.0))

    def test_fix_a_score_source_mismatch(self):
        recs = _syn_k100_records()
        rr, ga = _syn_run(recs)
        g, _u = K.score_from_generated(ga, SYN_SOL, SYN_ORDER, recs)
        rr.update(g, eval_query_hits=999)                     # the run report disagrees with the submission
        r = K.build_quality_report(rr, ga, _syn_engine(), SYN_SOL, _syn_ref(), records=recs)
        self.assertTrue(any(e.startswith("SCORE_SOURCE_MISMATCH") for e in r["errors"]), r["errors"])
        self.assertIsNone(r["k100"]["score"])
        sc = r["inputs"]["score_check"]
        self.assertIs(sc["agree"], False)
        self.assertEqual(sc["score_sources"], "both")
        self.assertEqual(r["delta"]["pass1"]["k100"], None)                # the scoped score is withheld too

    def test_fix_a_agreeing_sources_keep_the_score(self):
        recs = _syn_k100_records()
        rr, ga = _syn_run(recs)
        g, _u = K.score_from_generated(ga, SYN_SOL, SYN_ORDER, recs)
        rr.update(g)
        r = K.build_quality_report(rr, ga, _syn_engine(), SYN_SOL, _syn_ref(), records=recs)
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["k100"]["score"], g)
        self.assertEqual(r["inputs"]["score_check"]["score_sources"], "both")

    def test_fix_a_single_source_labels(self):
        recs = _syn_k100_records()
        rr, ga = _syn_run(recs)                               # _syn_run has no score keys: generated only
        r = K.build_quality_report(rr, ga, _syn_engine(), SYN_SOL, _syn_ref(), records=recs)
        self.assertEqual(r["inputs"]["score_check"]["score_sources"], "generated_only")
        self.assertIsNotNone(r["k100"]["score"])
        g, _u = K.score_from_generated(ga, SYN_SOL, SYN_ORDER, recs)
        rr.update(g)
        r2 = K.build_quality_report(rr, {}, _syn_engine(), {}, _syn_ref(), records=recs)
        self.assertEqual(r2["inputs"]["score_check"]["score_sources"], "run_report_only")
        self.assertEqual(r2["k100"]["score"], g)

# ==================================================================== audit E (2026-11-01): matched 4,050-s cutoff
def _fin(offs):
    per = {}
    for key, o in offs.items():
        tk, k = key.split("|")
        per.setdefault(tk, {"attempts": []})["attempts"].append({"attempt": int(k), "end_offset_sec": o})
    return {"per_task": per}


class MatchedCutoffTests(unittest.TestCase):
    def _r(self, offs, actual=6600.0, verify=0.5, accept=None):
        """The synthetic run with a klee100 window of `actual` s (None = no window fact; audit H), these request-end offsets,
        verify_sec on every submission (None = not recorded) and accept = {"task|attempt": Duck's verification start t} with
        the window start epoch 0 (audit J: accept = t + verify_sec)."""
        recs = _syn_k100_records()
        rr, ga = _syn_run(recs)
        for tid, rs in recs.items():
            for r in rs:
                for s in r["submissions"]:
                    if verify is not None:
                        s["verify_sec"] = verify
                    if accept and f"{tid}|{r['branch'] + 1}" in accept:
                        s["t"] = accept[f"{tid}|{r['branch'] + 1}"]
        eng = dict(_syn_engine(), window={"start_sec": 100, "actual_sec": actual} if actual is not None else {"start_sec": 100},
                   window_start_epoch=0.0)
        return K.build_quality_report(rr, ga, eng, SYN_SOL, _syn_ref(), records=recs, finish=_fin(offs) if offs is not None else None)

    def test_j_accept_after_the_cutoff_is_outside(self):
        """Request end 4,049 s, accept (t 4,054.5 + verify 0.5) 4,055 s -> outside the matched cutoff."""
        m = self._r({"tA|1": 4049.0, "tA|2": 4049.0}, accept={"tA|1": 4054.5, "tA|2": 4054.5})["paired_at_matched_4050s"]
        ta = m["tasks"]["tA"]
        self.assertEqual((m["status"], m["time_basis"], ta["k100_exact_attempt_time_sec"]["1"], ta["k100_at_cutoff"]),
                         ("OK", "ACCEPT_TIME", 4055.0, False))
        self.assertEqual(ta["label_full_window"], "both")                          # the full window is unchanged

    def test_j_accept_before_the_cutoff_is_inside(self):
        m = self._r({"tA|1": 3990.0}, accept={"tA|1": 3999.5})["paired_at_matched_4050s"]
        self.assertEqual((m["tasks"]["tA"]["k100_exact_attempt_time_sec"]["1"], m["tasks"]["tA"]["k100_at_cutoff"],
                          m["tasks"]["tA"]["k100_exact_attempt_time_basis"]["1"]), (4000.0, True, "ACCEPT_TIME"))

    def test_j_no_accept_and_no_verify_is_inconclusive(self):
        m = self._r({"tA|1": 3000.0, "tA|2": 3100.0}, verify=None)["paired_at_matched_4050s"]
        self.assertEqual((m["status"], m["undecided_tasks"], m["counts_at_cutoff"]), ("INCONCLUSIVE_ACCEPT_TIME", ["tA"], None))

    def test_j_request_end_plus_verify_basis(self):
        m = self._r({"tA|1": 4049.0, "tA|2": 4049.8}, verify=0.5)["paired_at_matched_4050s"]
        ta = m["tasks"]["tA"]
        self.assertEqual((m["time_basis"], ta["k100_exact_attempt_time_sec"], ta["k100_at_cutoff"]),
                         ("REQUEST_END_PLUS_VERIFY", {"1": 4049.5, "2": 4050.3}, True))

    def test_short_window_3600s_is_inconclusive(self):
        m = self._r({"tA|1": 3000.0, "tA|2": 3100.0}, actual=3600.0)["paired_at_matched_4050s"]
        self.assertEqual((m["status"], m["actual_sec"], m["required_sec"], m["counts_at_cutoff"]),
                         ("INCONCLUSIVE_SHORT_WINDOW", 3600.0, 4050.0, None))
        self.assertEqual(m["counts_full_window_same_rule"]["both"], 1)              # the full-window results stay
        self.assertEqual(self._r({"tA|1": 3000.0}, actual=None)["paired_at_matched_4050s"]["status"], "INCONCLUSIVE_SHORT_WINDOW")
        self.assertEqual(self._r({"tA|1": 3000.0}, actual=4047.0)["paired_at_matched_4050s"]["status"], "INCONCLUSIVE_SHORT_WINDOW")

    def test_window_4050_and_6600_give_the_normal_analysis(self):
        for act in (4048.5, 4050.0, 6600.0):                                       # 4,050 - 2 s tolerance is enough
            m = self._r({"tA|1": 3000.0, "tA|2": 4100.0}, actual=act)["paired_at_matched_4050s"]
            self.assertEqual((m["status"], m["k100_window_actual_sec"], m["tasks"]["tA"]["label_at_cutoff"]), ("OK", act, "both"), act)
        self.assertEqual(self._r(None, actual=6600.0)["paired_at_matched_4050s"]["status"], "INCONCLUSIVE_NO_TIMING")

    def test_attempt_at_4100s_counts_only_in_the_full_window(self):
        m = self._r({"tA|1": 4100.0, "tA|2": 4100.0})["paired_at_matched_4050s"]
        self.assertEqual(m["status"], "OK")
        self.assertEqual(m["offsets_source"], "k100_finish")
        ta = m["tasks"]["tA"]
        self.assertEqual((ta["k100_full_window"], ta["k100_at_cutoff"], ta["w4"]), (True, False, True))
        self.assertEqual((ta["label_full_window"], ta["label_at_cutoff"]), ("both", "w4_only"))
        self.assertEqual(m["counts_full_window_same_rule"]["both"], 1)
        self.assertEqual((m["counts_at_cutoff"]["both"], m["counts_at_cutoff"]["w4_only"]), (0, 2))

    def test_attempt_at_3000s_counts_in_both(self):
        m = self._r({"tA|1": 3000.0, "tA|2": 4100.0})["paired_at_matched_4050s"]
        ta = m["tasks"]["tA"]
        self.assertEqual((ta["k100_full_window"], ta["k100_at_cutoff"], ta["label_at_cutoff"]), (True, True, "both"))
        self.assertEqual(m["counts_at_cutoff"], m["counts_full_window_same_rule"])

    def test_no_timing_is_inconclusive(self):
        m = self._r(None)["paired_at_matched_4050s"]
        self.assertEqual((m["status"], m["undecided_tasks"], m["counts_at_cutoff"], m["offsets_source"]),
                         ("INCONCLUSIVE_NO_TIMING", ["tA"], None, "none"))
        m2 = self._r({"tA|2": 3500.0})["paired_at_matched_4050s"]     # the untimed attempt 1 does not matter: 2 is inside
        self.assertEqual((m2["status"], m2["tasks"]["tA"]["k100_at_cutoff"]), ("OK", True))
        m3 = self._r({"tA|2": 4500.0})["paired_at_matched_4050s"]     # 2 outside, 1 untimed: never assumed
        self.assertEqual((m3["status"], m3["undecided_tasks"]), ("INCONCLUSIVE_ACCEPT_TIME", ["tA"]))   # timing exists, not for 1

    def test_window_basis_label(self):
        self.assertEqual(K.window_basis({"actual_sec": 6600.0}, {"actual_sec": 4049.5}), "PAIRED_TASK_OUTCOME_DIFFERENT_WINDOWS")
        self.assertEqual(K.window_basis({"actual_sec": 4050.0}, {"actual_sec": 4049.5}), "PAIRED_TASK_OUTCOME_SAME_WINDOW")
        self.assertEqual(K.window_basis({}, {"actual_sec": 4049.5}), "PAIRED_TASK_OUTCOME_WINDOWS_UNKNOWN")
        r = self._r({"tA|1": 3000.0})
        self.assertIn(r["paired"]["window_basis"], ("PAIRED_TASK_OUTCOME_SAME_WINDOW", "PAIRED_TASK_OUTCOME_DIFFERENT_WINDOWS",
                                                    "PAIRED_TASK_OUTCOME_WINDOWS_UNKNOWN"))

    def test_attempt1_table_shows_the_finish_offset(self):
        r = self._r({"tA|1": 3000.0, "tB|1": 3990.5})
        a = r["attempt1_vs_w4"]["tasks"]
        self.assertEqual((a["tA"]["k100"]["finish_offset_sec"], a["tB"]["k100"]["finish_offset_sec"], a["tC"]["k100"]["finish_offset_sec"]),
                         (3000.0, 3990.5, None))
        md = K.markdown_extra(r)
        self.assertIn("| tA | solid |", md)
        self.assertIn("| 3,000 |", md)
        self.assertIn("Matched cutoff 4050.0 s", md)

# ==================================================================== the notebook block's functions (moved for the 400-kB cap)
class NotebookBlockTests(unittest.TestCase):
    def test_status_order(self):
        ok = {"inputs": {"score_check": {"agree": True}}, "errors": []}
        self.assertEqual(K.notebook_status({"status": "DONE"}, ok), {"status": "VALID"})
        self.assertEqual(K.notebook_status({"status": "PHASE_FAILED", "solver_error": "boom"}, ok)["status"], "INCOMPLETE_PHASE")
        self.assertEqual(K.notebook_status({"status": "DONE", "solver_error": "boom"}, ok)["phase_error"], "boom")
        self.assertEqual(K.notebook_status({"status": "DONE", "error": "x"}, ok)["status"], "INCOMPLETE_PHASE")
        self.assertEqual(K.notebook_status({"status": "PHASE_FAILED"}, None, run_report_present=False)["status"], "INCOMPLETE_PHASE")
        self.assertEqual(K.notebook_status({"status": "DONE"}, None, run_report_present=False)["status"], "MISSING_RUN_REPORT")
        self.assertEqual(K.notebook_status({"status": "DONE"}, ok, missing_gold=["t9"]),
                         {"status": "INVALID_MISSING_GOLD", "missing_gold": ["t9"]})
        self.assertEqual(K.notebook_status({"status": "DONE"}, {"inputs": {"score_check": {"agree": False}}, "errors": ["x"]})["status"],
                         "INCONCLUSIVE_SCORE_MISMATCH")
        self.assertEqual(K.notebook_status({"status": "DONE"}, {"errors": ["delta: x"]})["status"], "INVALID_QUALITY_REPORT")
        self.assertEqual(K.notebook_status({}, {})["status"], "INCOMPLETE_PHASE")

    def test_flags(self):
        P = {"patch_60490": True, "fixed_window_sec": 6600}
        self.assertEqual(K.review_flags(P, {"patch_60490": {"ok": True}}, {}, {"window": {"actual_sec": 6599.5}}), [])
        f = K.review_flags(P, {"patch_60490": {"ok": False}}, {"window_plan": {"unequal": True, "short_by_sec": 900.0}},
                           {"window": {"actual_sec": 5700.0}})
        self.assertTrue(f[0].startswith("#60490 NOT VERIFIED") and "2,479,838,700 B" in f[0])
        self.assertIn("short by 900.0 s", f[1])

    def test_engine_summary(self):
        s = K.engine_summary({"counters": {"vllm:num_preemptions_total": 3, "vllm:prompt_tokens_total": 1000.0,
                                           "vllm:prefix_cache_hits_total": 700.0, "vllm:generation_tokens_total": 50.0}},
                             {"window": {"actual_sec": 7200.0}, "capacity_levers": {"workers": 21}}, {"tasks_finished": 20},
                             {"paired": {"window_basis": "PAIRED_TASK_OUTCOME_DIFFERENT_WINDOWS"},
                              "paired_at_matched_4050s": {"status": "OK", "counts_at_cutoff": {"both": 1}}})
        self.assertEqual((s["preemptions"], s["fresh_prefill_tokens"], s["tasks_finished_per_hour"], s["workers"]), (3, 300, 10.0, 21))
        self.assertEqual((s["paired_window_basis"], s["matched_4050s_status"]), ("PAIRED_TASK_OUTCOME_DIFFERENT_WINDOWS", "OK"))
        self.assertIsNone(K.engine_summary({}, {}, {}, {})["tasks_finished_per_hour"])

# ==================================================================== audit F (2026-11-01): adversarial, one defect per fixture
_EMPTY_REF = {"configs": {}, "groups": {}}


def _adv(gen, sol, order, rr=None):
    return K.build_quality_report(rr or {}, gen, {}, sol, _EMPTY_REF, records={}, order=order, ref27=False)


def _hit(x):
    return {"attempt_1": x, "attempt_2": [[0]]}


class AdversarialTests(unittest.TestCase):
    def test_two_gold_one_generated(self):
        r = _adv({"taskA": [_hit([[1]])]}, {"taskA": [[[1]], [[2]]]}, ["taskA"])
        self.assertIn("QUERY_COUNT_MISMATCH:taskA:generated=1,gold=2", r["errors"])
        s = r["k100"]["score"]
        self.assertEqual((s["eval_query_hits"], s["eval_query_total"], s["eval_task_norm_pct"]), (1, 2, 50.0))
        self.assertEqual(K.notebook_status({"status": "DONE"}, r)["status"], "INVALID_QUALITY_REPORT")

    def test_headline_2_of_4_units_4_of_4(self):
        sol = {"tX": [[[1]], [[2]]], "tY": [[[3]], [[4]]]}
        gen = {"tX": [_hit([[1]]), _hit([[9]])], "tY": [_hit([[3]]), _hit([[9]])]}       # 2 / 4 right
        g, _u = K.score_from_generated(gen, sol, ["tX", "tY"])
        rr = dict(g, eval_unit_records=[{"task": t, "query": q, "hit": True, "pass1": True, "pass2": False}
                                        for t in ("tX", "tY") for q in (0, 1)])        # the run report claims 4 / 4
        r = _adv(gen, sol, ["tX", "tY"], rr)
        self.assertTrue(r["inputs"]["score_check"]["agree"])                             # the headline agrees
        self.assertTrue(any(e.startswith("UNIT_RECORD_MISMATCH") for e in r["errors"]), r["errors"])
        self.assertEqual(sum(v["query_hits"] for v in r["k100"]["tasks"].values()), 2)  # the canonical units, not 4
        self.assertEqual(r["inputs"]["score_check"]["units_source"], "generated_attempts")
        self.assertEqual(K.notebook_status({"status": "DONE"}, r)["status"], "INVALID_QUALITY_REPORT")

    def test_duplicate_unit_key(self):
        sol = {"tX": [[[1]]]}
        gen = {"tX": [_hit([[1]])]}
        g, u = K.score_from_generated(gen, sol, ["tX"])
        rr = dict(g, eval_unit_records=[dict(x) for x in u] + [dict(x) for x in u])
        r = _adv(gen, sol, ["tX"], rr)
        self.assertEqual(r["inputs"]["score_check"]["unit_check"]["duplicates"], [["tX", 0]])
        self.assertTrue(any(e.startswith("UNIT_RECORD_MISMATCH") for e in r["errors"]))

    def test_run_report_99_generated_2_of_4(self):
        sol = {"tX": [[[1]], [[2]]], "tY": [[[3]], [[4]]]}
        gen = {"tX": [_hit([[1]]), _hit([[9]])], "tY": [_hit([[3]]), _hit([[9]])]}
        rr = {"eval_query_hits": 99, "eval_query_total": 99, "eval_pass1_hits": 99, "eval_pass2_hits": 0, "eval_task_norm_pct": 100.0}
        r = _adv(gen, sol, ["tX", "tY"], rr)
        self.assertTrue(any(e.startswith("SCORE_SOURCE_MISMATCH") for e in r["errors"]))
        self.assertIsNone(r["k100"]["score"])
        self.assertEqual(K.notebook_status({"status": "DONE"}, r)["status"], "INCONCLUSIVE_SCORE_MISMATCH")

    def test_missing_gold_for_one_task(self):
        r = _adv({"tX": [_hit([[1]])], "tY": [_hit([[3]])]}, {"tX": [[[1]]]}, ["tX", "tY"])
        self.assertIn("MISSING_GOLD:tY", r["errors"])
        self.assertEqual(K.notebook_status({"status": "DONE"}, r), {"status": "INVALID_MISSING_GOLD", "missing_gold": ["tY"]})

    def test_phase_failed_with_partial_json(self):
        r = _adv({"tX": [_hit([[1]])]}, {"tX": [[[1]]], "tY": [[[3]]]}, ["tX", "tY"])     # tY never written: partial
        st = K.notebook_status({"status": "PHASE_FAILED", "error": "RuntimeError: server died"}, r)
        self.assertEqual(st["status"], "INCOMPLETE_PHASE")
        self.assertIn("server died", st["phase_error"])
        self.assertIn("tX", r["k100"]["tasks"])                                        # the report is kept for forensics

    def test_done_consistent_is_valid(self):
        sol = {"tX": [[[1]], [[2]]], "tY": [[[3]]]}
        gen = {"tX": [_hit([[1]]), _hit([[2]])], "tY": [_hit([[9]])]}
        g, u = K.score_from_generated(gen, sol, ["tX", "tY"])
        r = _adv(gen, sol, ["tX", "tY"], dict(g, eval_unit_records=u))
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["inputs"]["score_check"]["unit_check"]["status"], "MATCH")
        self.assertEqual(K.notebook_status({"status": "DONE"}, r), {"status": "VALID"})

if __name__ == "__main__":
    unittest.main(verbosity=2)
