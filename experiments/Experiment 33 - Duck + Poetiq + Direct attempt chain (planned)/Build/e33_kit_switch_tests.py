"""E33 kit tests (quick; the Kaggle CPU units kit runs them on Py3.12): the CELL 1 switch E33_CHAIN of the BUILT notebook and its
parser e33_lib.check_chain (the payload-dataset bytes), design section 5: "Switches (plain CELL 1 literals): each method on / off
(Poetiq C+D, Direct E, attempt 2), the Poetiq budgets, the Direct config. A bad value turns only that item off."
Plus the admission rule e33_lib.can_start (design section 3: Poetiq / Direct start only when their thinking can reach the
60,000 floor before the drain point; Duck stages never gated) and items_state (the [LIVE start] / report labels).

usage: python -m unittest -v e33_kit_switch_tests      env E33KT_NB = the notebook (.ipynb); e33_lib.py next to this file
"""
import ast
import copy
import json
import os
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e33_lib as L   # noqa: E402

NB = Path(os.environ.get("E33KT_NB") or (HERE.parent / "Experiment Notebooks" / "Q27-XHIGH-E33_CHAIN.ipynb"))
CODE = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
C1 = CODE[1]
LINE = [ln for ln in C1.splitlines() if ln.startswith("E33_CHAIN = {")]
SPEC = {"on": True, "poetiq": True, "direct": True, "attempt2": True, "poetiq_think": 60000, "poetiq_answer": 8192,
        "direct_think": 104000, "direct_answer": 8192, "temperature": 1.0}   # design sections 2 / 5 (drain_min: the builder's)


def literal():
    ln = LINE[0]
    expr = ln[len("E33_CHAIN = "):]
    node = ast.parse(expr, mode="eval").body          # a trailing comment is not part of the expression
    return ast.literal_eval(node)


def parse_stmt():
    """The notebook's own CELL 1 statement that runs check_chain on the literal (the only E33_CHAIN = E33L.check_chain(...) line)."""
    ls = [ln.strip() for ln in C1.splitlines() if ln.strip().startswith("E33_CHAIN = ") and "check_chain(" in ln]
    return ls


class TestNotebookSwitch(unittest.TestCase):
    def test_plain_literal_once_in_cell1(self):
        self.assertEqual(len(LINE), 1, LINE)
        self.assertEqual(sum(ln.startswith("E33_CHAIN = {") for c in CODE for ln in c.splitlines()), 1)
        lit = literal()
        self.assertIsInstance(lit, dict)
        self.assertEqual(sorted(lit), sorted(L.CHAIN_KEYS))
        self.assertEqual({k: lit[k] for k in SPEC}, SPEC)
        self.assertTrue(isinstance(lit["drain_min"], int) and lit["drain_min"] > 0)

    def test_notebook_parse_statement_eq_check_chain(self):
        ls = parse_stmt()
        self.assertEqual(len(ls), 1, ls)
        for harness in (False, True):
            g = {"E33L": L, "os": os, "json": json, "E33_CHAIN": literal(), "E11_HARNESS": harness}
            old = os.environ.pop("E33_CHAIN_JSON", None)
            try:
                exec(compile(ls[0], "<cell1 E33_CHAIN>", "exec"), g)
            finally:
                if old is not None:
                    os.environ["E33_CHAIN_JSON"] = old
            want = L.check_chain(literal())
            self.assertEqual(g["E33_CHAIN"], want)
        self.assertTrue(want["on"] and want["poetiq"] and want["direct"] and want["attempt2"] and not want["invalid"], want)
        self.assertEqual((want["poetiq_think"], want["poetiq_answer"], want["direct_think"], want["direct_answer"], want["temperature"]),
                         (60000, 8192, 104000, 8192, 1.0))

    def test_harness_override_only_with_e11_harness(self):
        ls = parse_stmt()[0]
        os.environ["E33_CHAIN_JSON"] = json.dumps({"poetiq_think": 400, "direct_think": 400})
        try:
            g = {"E33L": L, "os": os, "json": json, "E33_CHAIN": literal(), "E11_HARNESS": False}
            exec(compile(ls, "<cell1>", "exec"), g)
            self.assertFalse(g["E33_CHAIN"]["harness_override"])
            self.assertEqual(g["E33_CHAIN"]["poetiq_think"], 60000)
            if "E11_HARNESS" in ls or "E11_HARNESS" in C1:
                os.environ["E11_HARNESS"] = "1"
                g = {"E33L": L, "os": os, "json": json, "E33_CHAIN": literal(), "E11_HARNESS": True}
                exec(compile(ls, "<cell1>", "exec"), g)
                self.assertTrue(g["E33_CHAIN"]["harness_override"])
                self.assertEqual(g["E33_CHAIN"]["poetiq_think"], 400)
        finally:
            os.environ.pop("E33_CHAIN_JSON", None)
            os.environ.pop("E11_HARNESS", None)


def chk(**kw):
    c = dict(literal(), **kw)
    return L.check_chain(c)


class TestBadValueOnlyThatItem(unittest.TestCase):
    def assertOnly(self, r, off, inv_key):
        on = {"poetiq", "direct", "attempt2"} - set(off)
        self.assertTrue(r["on"], r)
        for k in on:
            self.assertTrue(r[k], (k, r))
        for k in off:
            self.assertFalse(r[k], (k, r))
        self.assertIn(inv_key, r["invalid"])

    def test_poetiq_items(self):
        for kw in ({"poetiq": "yes"}, {"poetiq": 1}, {"poetiq_think": 59999}, {"poetiq_think": "60000"}, {"poetiq_think": True},
                   {"poetiq_answer": 0}, {"poetiq_answer": None}, {"poetiq_think": 130000}):
            r = chk(**kw)
            self.assertOnly(r, ["poetiq"], "poetiq")
            self.assertEqual((r["direct_think"], r["direct_answer"]), (104000, 8192), kw)

    def test_direct_items(self):
        for kw in ({"direct": "on"}, {"direct_think": 50000}, {"direct_think": 1.5}, {"direct_answer": -1}, {"direct_think": 125000}):
            r = chk(**kw)
            self.assertOnly(r, ["direct"], "direct")
            self.assertEqual((r["poetiq_think"], r["poetiq_answer"]), (60000, 8192), kw)

    def test_attempt2_item(self):
        for v in ("true", 1, None):
            r = chk(attempt2=v)
            self.assertOnly(r, ["attempt2"], "attempt2")

    def test_shared_items_turn_off_the_solvers_that_use_them(self):
        for kw in ({"temperature": "1.0"}, {"temperature": 3.0}, {"drain_min": 0}, {"drain_min": 2.5}):
            r = chk(**kw)
            self.assertTrue(r["on"] and r["attempt2"], r)
            self.assertFalse(r["poetiq"] or r["direct"], (kw, r))

    def test_explicit_off_is_not_invalid(self):
        r = chk(poetiq=False)
        self.assertFalse(r["poetiq"])
        self.assertNotIn("poetiq", r["invalid"])
        self.assertTrue(r["direct"] and r["attempt2"])
        r = chk(direct=False, attempt2=False)
        self.assertTrue(r["poetiq"] and not r["direct"] and not r["attempt2"] and not r["invalid"], r)

    def test_chain_level(self):
        for c in ("on", None, [], 5):
            self.assertFalse(L.check_chain(c)["on"])
        for v in ("True", 1, None):
            r = chk(on=v)
            self.assertFalse(r["on"] or r["poetiq"] or r["direct"] or r["attempt2"], r)
            self.assertIn("chain", r["invalid"])
        r = chk(on=False)
        self.assertFalse(r["on"] or r["poetiq"] or r["direct"] or r["attempt2"])
        r = chk(extra_key=3)
        self.assertTrue(r["on"] and r["poetiq"] and r["direct"] and r["attempt2"])
        self.assertEqual(r["ignored"], ["extra_key"])

    def test_never_raises(self):
        for c in ({"on": True, "poetiq": object()}, {"on": True}, {"on": True, "temperature": float("nan")}):
            r = L.check_chain(c)
            self.assertIsInstance(r, dict)
        self.assertIn("chain", L.check_chain(literal(), harness_json="{bad json")["invalid"])

    def test_given_is_not_mutated(self):
        c = literal()
        c0 = copy.deepcopy(c)
        L.check_chain(c, harness_json=json.dumps({"poetiq_think": 400}))
        self.assertEqual(c, c0)


class TestAdmission(unittest.TestCase):
    def test_floor_before_drain(self):
        cfg = L.check_chain(literal())
        dm = cfg["drain_min"] * 60.0
        need = L.START_MARGIN_SEC + 60000 / 18.0
        now = 1000.0
        for st in ("C", "D", "E"):
            ok, why, n = L.can_start(now, now + dm + need + 1.0, cfg, st, 18.0)
            self.assertEqual((ok, why), (True, "ok"), st)
            self.assertAlmostEqual(n, round(need, 1), places=1)
            ok, why, _ = L.can_start(now, now + dm + need - 1.0, cfg, st, 18.0)
            self.assertEqual((ok, why), (False, "late"), st)
        for st in ("A", "B1", "B2"):
            self.assertEqual(L.can_start(now, now + 1.0, cfg, st, 18.0)[:2], (True, "duck"))

    def test_floor_is_60k_even_for_the_direct_104k_budget(self):
        cfg = L.check_chain(literal())
        self.assertEqual(L.floor_tokens(cfg, "direct"), 60000)
        self.assertEqual(L.floor_tokens(cfg, "poetiq"), 60000)

    def test_off_items_never_start(self):
        now = 0.0
        cfg = L.check_chain(dict(literal(), poetiq=False))
        self.assertEqual(L.can_start(now, 1e9, cfg, "C", 18.0)[:2], (False, "off"))
        self.assertEqual(L.can_start(now, 1e9, cfg, "D", 18.0)[:2], (False, "off"))
        self.assertTrue(L.can_start(now, 1e9, cfg, "E", 18.0)[0])
        cfg = L.check_chain(dict(literal(), on=False))
        for st in ("C", "D", "E"):
            self.assertFalse(L.can_start(now, 1e9, cfg, st, 18.0)[0])

    def test_rate(self):
        self.assertEqual(L.per_stream_rate(None, None), 18.0)
        self.assertEqual(L.per_stream_rate(450.0, 24), 18.0)
        self.assertEqual(L.per_stream_rate(0.0, 3), 18.0)

    def test_items_state(self):
        self.assertEqual(L.items_state(L.check_chain(literal())), {"chain": "ON", "poetiq": "ON", "direct": "ON", "attempt2": "ON"})
        s = L.items_state(L.check_chain(dict(literal(), poetiq_think=10)))
        self.assertTrue(s["poetiq"].startswith("OFF (invalid") and s["direct"] == "ON")
        s = L.items_state(L.check_chain(dict(literal(), on=False)))
        self.assertTrue(all(v == "OFF (chain off)" for k, v in s.items() if k != "chain"), s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
