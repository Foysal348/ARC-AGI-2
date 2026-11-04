"""E31 Flash-Next DIRECT 30 unit tests (local + the Kaggle CPU units kit, Py3.12). Every check prints PASS / FAIL / SKIP; the
last line is "<n>/<m> PASS". The direct solver under test is the e31_direct.py UNPACKED FROM THE NOTEBOOK's CELL 1 literal (what
the GPU run executes), not the Build file.
env: E31T_NB (notebook), E31T_FACTS (e31_build_facts.json), E31T_TOKDIR (the klee100 tokenizer dir: tokenizer.json, tokenizer_config.json,
     chat_template.jinja, generation_config.json; sha-pinned), E31T_EVAL (dir with arc-agi_evaluation_challenges.json + _solutions.json),
     E31T_DUCK (optional: the payload's original arc2_duck.py), E31T_E10_NB (optional: the E10 ORDER2 notebook, provenance of the
     ORDER2 fixture), E31T_EXPORT (optional: Build/e31_code_export.py), E31T_MERLIN_SRC (optional: a Merlin source tree)
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
RESULTS = []


def check(name, cond, info=""):
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, info = False, f"raised {type(e).__name__}: {e} | {info} | {traceback.format_exc()[-800:]}"
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" -> {str(info)[:1500]}"), flush=True)
    return ok


def skip(name, why):
    print(f"SKIP {name}: {why}", flush=True)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


HERE = Path(__file__).resolve().parent
NB = Path(os.environ.get("E31T_NB") or HERE.parent / "Q27-XHIGH-E31_FN_DIRECT30.ipynb")
FACTS = json.loads(Path(os.environ.get("E31T_FACTS") or HERE / "e31_build_facts.json").read_text(encoding="utf-8"))
TOKDIR = Path(os.environ.get("E31T_TOKDIR") or r"D:/ARC AGI 2/_scratch/e13_rev6/harness/fixtures/tok_int4")
EVAL = Path(os.environ.get("E31T_EVAL") or r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
DUCK = os.environ.get("E31T_DUCK") or r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/duck_b/arc2_duck.py"
E10_NB = os.environ.get("E31T_E10_NB") or r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb"
EXPORT = os.environ.get("E31T_EXPORT") or str(HERE / "e31_code_export.py")
MERLIN_SRC = os.environ.get("E31T_MERLIN_SRC") or r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src"
TOK_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"   # the klee100 tokenizer (E28 CELL 5 TOK_SHA)
TPL_SHA = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"

# ---------------------------------------------------------------- the notebook under test
NB_RAW = NB.read_bytes()
NBJ = json.loads(NB_RAW.decode("utf-8"))
CELLS = ["".join(c["source"]) for c in NBJ["cells"]]
CODE = ["".join(c["source"]) for c in NBJ["cells"] if c["cell_type"] == "code"]


def unpack_literal(expr):
    return eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<packed>", "eval"))


def readable(code):
    c = code[5]
    i = c.index("# ---- E28 K100 (size: the 400-kB commit limit)")
    call = next(n for n in ast.walk(ast.parse(c)) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
    return code[:5] + [c[:i] + unpack_literal(ast.get_source_segment(c, call.args[0].args[0]))] + code[6:]


READ = readable(CODE)
_ml = next(l for l in CODE[1].splitlines() if l.startswith("E28_SRC = json.loads("))
MODS = json.loads(unpack_literal(_ml[len("E28_SRC = json.loads("):-1]))
TMP = Path(tempfile.mkdtemp(prefix="e31ut_"))
for n_, s_ in MODS.items():
    (TMP / n_).write_text(s_, encoding="utf-8")
sys.path.insert(0, str(TMP))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E31D = load("e31_direct", TMP / "e31_direct.py")
E28L = load("e28k100_lib", TMP / "e28k100_lib.py")

# ---------------------------------------------------------------- T1 provenance / structure
check("T1 notebook sha256 + bytes == the build facts; < 400 kB; no CR", sha(NB_RAW) == FACTS["notebook_sha256"]
      and len(NB_RAW) == FACTS["bytes"] < 400_000 and b"\r" not in NB_RAW.replace(b"\\r", b""))
check("T1 layout: markdown + 9 code cells (CELL 0, 1, 3, E10 order, 5, 6/7, 7b, 8, 9)",
      [c["cell_type"] for c in NBJ["cells"]] == ["markdown"] + ["code"] * 9 and CODE[6].startswith(
          "# ==================================================================== CELL 7b — E31") and "CELL 8" in CODE[7].splitlines()[0]
      and "CELL 9" in CODE[8].splitlines()[0])
check("T1 the packed modules == the facts (e28k100_lib unchanged = E28's, e31_direct = the Build file)",
      {n: sha(s) for n, s in MODS.items()} == FACTS["modules"] and FACTS["modules"]["e28k100_lib.py"] ==
      "ff923ef252dc3a85ef5e7f5de8af1b777a8ee3a76317f2eb5614220434c2511b" and set(MODS) == {"e28k100_lib.py", "e31_direct.py"})
check("T1 the readable code cells (e21_cell7 unpacked) == the export sha in the facts", sha("\n\n".join(READ)) == FACTS["export_sha256"])
if Path(EXPORT).is_file():
    check("T1 the export file == the readable cells", Path(EXPORT).read_bytes().decode("utf-8") == "\n\n".join(READ))
_bs = FACTS.get("base_cell_sha256") or []
check("T1 every cell except CELL 1 and the new CELL 7b == the E28 K100 cell byte for byte (base cell shas)",
      len(_bs) == 8 and [sha(CODE[i]) for i in (0, 2, 3, 4, 5, 7, 8)] == [_bs[i] for i in (0, 2, 3, 4, 5, 6, 7)] and sha(CODE[1]) != _bs[1])

# ---------------------------------------------------------------- T2 the editable E31_CFG line
_c1 = CODE[1].splitlines()
check("T2 the E31_CFG line is the first code line of CELL 1, a plain literal, == the facts / the module default",
      _c1[1] == FACTS["e31"]["cfg_line"] and ast.literal_eval(_c1[1].split("=", 1)[1].strip()) == FACTS["e31"]["cfg"] == E31D.CFG_DEFAULT
      and [l for l in CODE[1].splitlines() if l.startswith("E31_CFG = {")] == [_c1[1]])
check("T2 the E31_CFG line is outside every packed / sha-locked text and present in the export (readable cells)",
      _c1[1] in "\n\n".join(READ) and all(_c1[1] not in s for s in MODS.values()) and "drain_min" in _c1[1])
_cfg_end = next(i for i, l in enumerate(_c1) if l.startswith("E31_BUDGET_MIN = ")) + 1
_g = {"os": os, "json": json}
exec(compile("\n".join(_c1[1:_cfg_end]), "<cfg>", "exec"), _g)
check("T2 the budget follows E31_CFG: 15 + window_min + 20 = 205 min; the deadman literal 12,300 s + the derived line",
      _g["E31_BUDGET_MIN"] == 205 and "K100_DEADMAN_SEC = 12300   # E31" in CODE[1]
      and "K100_DEADMAN_SEC = int(E31_BUDGET_MIN * 60)" in CODE[1] and FACTS["deadman_sec"] == 12300 and FACTS["budget_min"] == 205)
_g2 = {"os": os, "json": json}
os.environ.update(E11_HARNESS="1", E31_CFG_JSON=json.dumps({"window_min": 50, "think": 6000}))
exec(compile("\n".join(_c1[1:_cfg_end]), "<cfg>", "exec"), _g2)
os.environ.pop("E11_HARNESS"); os.environ.pop("E31_CFG_JSON")
check("T2 harness-only override: E31_CFG_JSON merges into E31_CFG under E11_HARNESS=1 (budget 85 min)",
      _g2["E31_CFG"]["window_min"] == 50 and _g2["E31_CFG"]["think"] == 6000 and _g2["E31_BUDGET_MIN"] == 85)
_row = CODE[1][CODE[1].index('    "E31DIRECT30": dict('):]
check("T2 profile row E31DIRECT30: task_list = the 30, phase Q_E31_FN_DIRECT30, fixed window, E31_LEVERS (no e28_sched); default + guard",
      re.search(r'nb5_phases=\[\["(\w+)"', _row).group(1) == "Q_E31_FN_DIRECT30" and "task_list=E31_TASKS" in _row.split("\n")[0]
      and "fixed_window_sec=E31_WINDOW_SEC" in _row and '**E31_LEVERS)' in _row and 'NB5_DEFAULT_PROFILE = "E31DIRECT30"' in CODE[1]
      and 'K100_PROFILES = ("E31DIRECT30", "RERUN_PROD")' in CODE[1] and '"E28EVAL120"' not in CODE[1]
      and 'PROFILE_TABLE["RERUN_PROD"] = dict(PROFILE_TABLE["RERUN_PROD"], **E31_LEVERS)' in CODE[1]
      and 'E31_LEVERS = {k: v for k, v in E28_LEVERS.items() if k != "e28_sched"}' in CODE[1])
_tl = FACTS["e31"]["timeline_min"]
check("T2 window math (nb5_phase: min(fixed, HARD_STOP - 3 - now - 1)): a boot at 15 / 14.65 min keeps the full 170-min window",
      _tl["window_by_start"]["start_15_min"] == 170 and _tl["window_by_start"]["start_14.65_min"] == 170 and _tl["hard_stop"] == 200)

# ---------------------------------------------------------------- T3 ORDER2 fixture (E10 nb[4] / nb[5] verbatim) vs the module
ORDER2_SRC = r'''
def render_grid(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)

def build_user_message(train, test_input):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(train):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i + 1, render_grid(p["input"]), i + 1, render_grid(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render_grid(test_input))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)
'''
O2 = {}
exec(ORDER2_SRC, O2)
if Path(E10_NB).is_file():
    _e10 = ["".join(c["source"]) for c in json.loads(Path(E10_NB).read_text(encoding="utf-8"))["cells"]]
    check("T3 the ORDER2 fixture == the E10 ORDER2 notebook nb[4] text (render_grid + build_user_message)",
          ORDER2_SRC.strip().replace("\n\n", "\n") in _e10[4].replace("\n\n", "\n"))
    _src5 = _e10[5]
    _parse = _src5[_src5.index("def valid_grid(g):"):_src5.index("def score_unit(uid, att):")]
    _mod = MODS["e31_direct.py"]
    _fns = ["valid_grid", "_parse_row", "_first_row_run", "_grid_from_obj", "_json_rows_grid", "_grid_from_block", "parse_grid"]
    _seg = lambda s, f: (lambda t: t[:t.index("\n\n")] if "\n\n" in t else t)(s[s.index(f"def {f}("):])
    check("T3 the module's ORDER2 parser functions == E10 nb[5] verbatim (valid_grid .. parse_grid; regexes)",
          all(_seg(_parse, f).rstrip() == _seg(_mod, f).rstrip() for f in _fns)
          and all(l in _mod for l in _parse.splitlines() if l.startswith(("_FENCE_RE", "_SEP_ROW", "_JSON_ROW"))))
    _ma = _src5[_src5.index("def make_attempts(uid, grid):"):_src5.index("def score_unit(uid, att):")]
    _body = lambda s: [l.strip() for l in s[s.index("    cands = "):s.index('"src_2": uniq[1][0]}')].splitlines()]
    _mma = _mod[_mod.index("def make_attempts(test_input, grid):"):]
    check("T3 make_attempts body == ORDER2 from 'cands =' to the return (only the test input argument differs)",
          _body(_ma) == _body(_mma), (_body(_ma), _body(_mma)))
else:
    skip("T3 ORDER2 fixture vs the E10 notebook", "E10 notebook not attached (the fixture is embedded verbatim)")
_mf = lambda f: ast.get_source_segment(MODS["e31_direct.py"], next(n for n in ast.parse(MODS["e31_direct.py"]).body
                                                                 if isinstance(n, ast.FunctionDef) and n.name == f))
check("T3 the module's render_grid / build_user_message source == the ORDER2 fixture byte for byte",
      _mf("render_grid") + "\n" in ORDER2_SRC and _mf("build_user_message") in ORDER2_SRC)
CH = json.loads((EVAL / "arc-agi_evaluation_challenges.json").read_text(encoding="utf-8"))
SOL = json.loads((EVAL / "arc-agi_evaluation_solutions.json").read_text(encoding="utf-8"))
_single = [t for t in E31D.TASKS30 if len(CH[t]["test"]) == 1]
check(f"T3 prompt bytes: every single-test task of the 30 ({len(_single)}) -> task_message == ORDER2 build_user_message",
      _single and all(E31D.task_message(CH[t]) == O2["build_user_message"](CH[t]["train"], CH[t]["test"][0]["input"]) for t in _single))
_m0 = E31D.task_message(CH[_single[0]])
check("T3 ORDER2 self-test on the message (no 'xhigh', 'FINAL INPUT:', ends 'Output nothing after the block.')",
      "xhigh" not in _m0 and "FINAL INPUT:" in _m0 and _m0.endswith("Output nothing after the block."))
_mt = {"train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}], "test": [{"input": [[5, 0]]}, {"input": [[2], [3]]}]}
_want = ("You are solving an abstract reasoning puzzle. Each example has an INPUT grid and its OUTPUT grid (digits 0-9 are colors; "
         "0 is usually background). Infer the ONE rule that maps every input to its output, then apply it to the FINAL INPUT.\n\n"
         "Example 1 INPUT:\n1 0\n0 1\nExample 1 OUTPUT:\n0 1\n1 0\n\nFINAL INPUT 1:\n5 0\n\nFINAL INPUT 2:\n2\n3\n\n"
         "Reason step by step however you like. Then give ONLY the final output grids, one per FINAL INPUT in order, each as rows "
         "of space-separated digits inside its own labeled fenced block:\n```grid 1\n<rows>\n```\n```grid 2\n<rows>\n```\n"
         "Output nothing after the last block.")
check("T3 multi-test prompt bytes (2 tests) == the expected text exactly", E31D.task_message(_mt) == _want, repr(E31D.task_message(_mt)))
_b6 = E31D.task_message(CH["b6f77b65"])
_o2b6 = O2["build_user_message"](CH["b6f77b65"]["train"], CH["b6f77b65"]["test"][0]["input"])
check("T3 multi-test prompt (b6f77b65, 3 tests): the ORDER2 intro + examples byte-identical, FINAL INPUT 1..3, labels grid 1..3",
      _b6.split("FINAL INPUT 1:")[0] == _o2b6.split("FINAL INPUT:")[0] and all(f"FINAL INPUT {k}:\n" in _b6 for k in (1, 2, 3))
      and all(f"```grid {k}\n<rows>\n```" in _b6 for k in (1, 2, 3)) and "FINAL INPUT 4" not in _b6 and _b6.endswith("Output nothing after the last block."))
check("T3 ORDER2 order over the 30 tasks (score [>=3 train] + [1 test], ties sorted-id) and seeds SEED + 10 * index + phase",
      E31D.order2({t: CH[t] for t in E31D.TASKS30}) == ['0934a4d8', '136b0064', '1818057f', '35ab12c3', '409aa875', '4c416de3', '5545f144',
                                                       '7491f3cf', '800d221b', '89565ca0', 'a47bf94d', 'a6f40cea', '135a2760', '13e47133',
                                                       '142ca369', '16b78196', '247ef758', '2d0172a1', '36a08778', '581f7754', '71e489b6',
                                                       '7b5033c1', '88e364bc', '9385bd28', 'a32d8b75', 'b6f77b65', 'f931b4a8', '446ef5d2',
                                                       '4c3d4a41', 'cbebaa4b']
      and E31D.task_seed(0, 1) == 20261006 and E31D.task_seed(3, 2) == 20261037)
check("T3 the 30 tasks = 3 groups (16 / 8 / 6), 46 test inputs; the reference table has the 30 rows",
      [len(E31D.GROUPS[g]) for g in ("main", "partial", "control")] == [16, 8, 6] and sum(len(CH[t]["test"]) for t in E31D.TASKS30) == 46
      and set(E31D.REF_TABLE) == set(E31D.TASKS30) and E31D.REF_TABLE["7b5033c1"] == [0.0, 0.0, 1.0, 0.0])

# ---------------------------------------------------------------- T4 the real tokenizer + template
TOK = HF = None
try:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    assert sha((TOKDIR / "tokenizer.json").read_bytes()) == TOK_SHA and sha((TOKDIR / "chat_template.jinja").read_bytes()) == TPL_SHA
    TOK = Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
    HF = AutoTokenizer.from_pretrained(str(TOKDIR))
    TPL = (TOKDIR / "chat_template.jinja").read_text()
except Exception as e:
    print("tokenizer not loaded:", repr(e)[:300])
if TOK is not None:
    GC = json.loads((TOKDIR / "generation_config.json").read_text(encoding="utf-8"))
    TOKD = {"render": lambda m: HF.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                       reasoning_effort="xhigh", chat_template=TPL),
            "encode": lambda s: TOK.encode(s, add_special_tokens=False).ids,
            "decode": lambda i, k: TOK.decode(list(i), skip_special_tokens=bool(k)), "eos_ids": GC["eos_token_id"]}
    _tiny = O2["build_user_message"]([{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}, {"input": [[2, 2]], "output": [[3, 3]]}], [[5, 0, 5]])
    _spec = ("<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
             "<|im_start|>user\n" + _tiny + "<|im_end|>\n<|im_start|>assistant\n<think>\n")
    check("T4 render (the klee100 tokenizer, sha 06b95093 / template c3cf9e34, xhigh) == the spec's exact rendered text",
          TOKD["render"]([{"role": "user", "content": _tiny}]) == _spec)
    _sid = TOKD["encode"](E31D.STOP_THINKING_TEXT)
    check("T4 STOP_IDS: 25 ids == the spec list, </think> = 248069 at index 23, last 271; </think> one token; EOS [248046, 248044]",
          _sid == E31D.FN_STOP_IDS and len(_sid) == 25 and _sid[23] == 248069 and _sid[-1] == 271
          and TOKD["encode"]("</think>") == [248069] and GC["eos_token_id"] == [248046, 248044] == list(E31D.FN_EOS_IDS))
    _ids = TOKD["encode"](_spec)
    check("T4 the rendered prompt ids end with <think>\\n (248068, 198) and decode back to the text", _ids[-2:] == [248068, 198]
          and TOKD["decode"](_ids, False) == _spec)
    _pl = {t: len(TOKD["encode"](TOKD["render"]([{"role": "user", "content": E31D.task_message(CH[t])}]))) for t in E31D.TASKS30}
    _caps = {t: E31D.phase1_cap(n, 104000, 8192 * len(CH[t]["test"])) for t, n in _pl.items()}
    print("prompt tokens of the 30:", json.dumps(_pl), "| phase-1 caps (answer x tests):", json.dumps(_caps))
    check("T4 every one of the 30 prompts keeps a phase-1 cap >= the 60k floor (max prompt %d tokens)" % max(_pl.values()),
          all(c >= 60000 for c in _caps.values()))
    check("T4 phase-1 cap reserves answer x N tests: b6f77b65 (3 tests, prompt 4,006) -> 102,234; every single-test task keeps "
          "min(104,000, 131,072 - prompt - 8,192 - 256)",
          _pl["b6f77b65"] == 4006 and _caps["b6f77b65"] == 102234
          and all(_caps[t] == E31D.phase1_cap(_pl[t], 104000, 8192) for t in E31D.TASKS30 if len(CH[t]["test"]) == 1), _caps)
else:
    skip("T4 tokenizer", f"the pinned klee100 tokenizer is not at {TOKDIR}")
check("T4 phase1_cap = min(think, 131072 - prompt - answer - 256): 1,000 -> 104,000; 20,000 -> 102,624; 100,000 -> 22,624; 130,000 -> 0",
      [E31D.phase1_cap(p, 104000, 8192) for p in (1000, 20000, 100000, 130000)] == [104000, 102624, 22624, 0])

# ---------------------------------------------------------------- T5 parsers
_G = [[1, 0, 2], [3, 4, 5]]
_layouts = {"grid_fence": "Done.\n```grid\n1 0 2\n3 4 5\n```", "no_label": "```\n1 0 2\n3 4 5\n```",
            "text_contiguous": "```text\n102\n345\n```", "comma_rows": "```grid\n1,0,2\n3,4,5\n```",
            "header_in_fence": "```grid\nOutput:\n1 0 2\n3 4 5\n```", "json_list": "```json\n[[1, 0, 2], [3, 4, 5]]\n```",
            "json_object": "```json\n{\"test_outputs\": [[[1, 0, 2], [3, 4, 5]]]}\n```",
            "json_missing_open": "```json\n{\"test_outputs\": [[1, 0, 2], [3, 4, 5]]]}\n```",
            "last_fence_not_grid": "```grid\n1 0 2\n3 4 5\n```\n```python\nprint('x')\n```",
            "earlier_ragged": "```grid\n9 9\n9\n```\nfixed:\n```grid\n1 0 2\n3 4 5\n```",
            "unfenced": "The final grid is:\n1 0 2\n3 4 5\n", "unclosed_fence": "```grid\n1 0 2\n3 4 5\n"}
check("T5 ORDER2 parse_grid self-tests: 12 layouts + empty / None / ragged / two-digit",
      all(E31D.parse_grid(t) == _G for t in _layouts.values()) and E31D.parse_grid("no grid here") is None and E31D.parse_grid("") is None
      and E31D.parse_grid(None) is None and E31D.parse_grid("```grid\n1 0\n3 4 5\n```") is None and E31D.parse_grid("```grid\n1 10 2\n```") is None)
check("T5 parse_grids n=1 = ORDER2 parse_grid for every layout (source parse_grid / fallback)",
      all(E31D.parse_grids(t, 1) == ([_G], ["parse_grid"], [None]) for t in _layouts.values())
      and E31D.parse_grids("nothing", 1) == ([None], ["fallback"], ["no_grid_parsed"]) and E31D.parse_grids("", 1)[2] == ["no_answer_text"])
A, B_, C = [[1, 1]], [[2, 2]], [[3, 3]]
blk = lambda lab, g: f"```{lab}\n" + "\n".join(" ".join(map(str, r)) for r in g) + "\n```\n"
PG = E31D.parse_grids
_cases = {
    "2 tests labeled in order": (blk("grid 1", A) + blk("grid 2", B_), 2, [A, B_], ["label", "label"]),
    "2 tests labels reversed": (blk("grid 2", B_) + blk("grid 1", A), 2, [A, B_], ["label", "label"]),
    "3 tests labeled": (blk("grid 1", A) + blk("grid 2", B_) + blk("grid 3", C), 3, [A, B_, C], ["label"] * 3),
    "3 tests reordered 3-1-2": (blk("grid 3", C) + blk("grid 1", A) + blk("grid 2", B_), 3, [A, B_, C], ["label"] * 3),
    "repeated label: the last valid wins": (blk("grid 1", C) + blk("grid 2", B_) + blk("grid 1", A), 2, [A, B_], ["label", "label"]),
    "repeated label, the last malformed -> the earlier valid": (blk("grid 1", A) + blk("grid 2", B_) + "```grid 1\n1 1\n2\n```\n", 2, [A, B_], ["label", "label"]),
    "missing first label (an unlabeled block exists: never positional)": (blk("grid", C) + blk("grid 2", B_), 2, [None, B_], ["fallback", "label"]),
    "missing last label": (blk("grid 1", A) + blk("grid", C), 2, [A, None], ["label", "fallback"]),
    "3 tests, label 2 missing": (blk("grid 1", A) + blk("grid 3", C), 3, [A, None, C], ["label", "fallback", "label"]),
    "extra reasoning fence (python) + labels": ("```python\nprint(1)\n```\n" + blk("grid 1", A) + blk("grid 2", B_), 2, [A, B_], ["label", "label"]),
    "truncated final block (unclosed)": (blk("grid 1", A) + "```grid 2\n2 2\n", 2, [A, None], ["label", "fallback"]),
    "duplicate grids on two labels": (blk("grid 1", A) + blk("grid 2", A), 2, [A, A], ["label", "label"]),
    "malformed grid under label 2": (blk("grid 1", A) + "```grid 2\n2 2\n2\n```\n", 2, [A, None], ["label", "fallback"]),
    "label out of range only (grid 5): labels exist -> no positional": (blk("grid 5", A) + blk("grid", B_), 2, [None, None], ["fallback", "fallback"]),
    "no labels, exactly N valid blocks -> positional": (blk("grid", A) + blk("", B_), 2, [A, B_], ["positional", "positional"]),
    "no labels, N+1 valid blocks -> no guess": (blk("grid", A) + blk("grid", B_) + blk("grid", C), 2, [None, None], ["fallback", "fallback"]),
    "no labels, one block for 2 tests -> no guess": (blk("grid", A), 2, [None, None], ["fallback", "fallback"]),
    "no labels, example fence that is not a grid + N grids": ("```python\nx = 1\n```\n" + blk("", A) + blk("", B_), 2, [A, B_], ["positional", "positional"]),
}
for _n, (_t, _k, _wg, _ws) in _cases.items():
    _r = PG(_t, _k)
    check(f"T5 multi parser: {_n}", _r[0] == _wg and _r[1] == _ws and all((w is None) == (g is not None) for g, w in zip(_r[0], _r[2])), _r)
check("T5 multi parser reasons: label_missing / label_block_invalid / no_labels_<k>_valid_blocks_not_<n> / no_answer_text",
      PG(blk("grid 1", A), 2)[2] == [None, "label_missing"] and PG(blk("grid 1", A) + "```grid 2\n1\n2 2\n```\n", 2)[2] == [None, "label_block_invalid"]
      and PG(blk("", A), 2)[2] == ["no_labels_1_valid_blocks_not_2"] * 2 and PG("", 3)[2] == ["no_answer_text"] * 3)
_inp = [[6, 7]]
check("T5 make_attempts (ORDER2): model grid -> (model, test input); None -> (test input, mirror); grid == input -> (model, mirror)",
      E31D.make_attempts(_inp, [[7, 6]])["attempt_2"] == _inp and E31D.make_attempts(_inp, None)["attempt_1"] == _inp
      and E31D.make_attempts(_inp, None)["attempt_2"] == [[7, 6]] and E31D.make_attempts(_inp, [[6, 7]])["src_2"] == "mirror"
      and E31D.make_attempts([[5, 5]], None)["attempt_2"] == [[5, 5]])

# ---------------------------------------------------------------- T6 the DIRECT-only adapter, CELL 7b, CELL 7's scorer, the E15B writer
def drec(tid, grids, seq=1):
    task = CH[tid]
    atts = [E31D.make_attempts(q["input"], g) for q, g in zip(task["test"], grids)]
    return {"task_id": tid, "branch": 0, "status": "done", "publish_seq": seq, "accepted": False, "generated_tokens": 1000,
            "t0": time.time() - 5, "t1": time.time(), "note": "", "nb5_verdict": None,
            "submissions": [{"test_outputs": list(grids), "accepted": False, "train_exact": [], "t": time.time(), "branch": 0}],
            "e31_attempts": [{"attempt_1": a["attempt_1"], "attempt_2": a["attempt_2"]} for a in atts], "e31": {"mode": "natural"}}


_t1, _t2, _t3 = "7b5033c1", "142ca369", "b6f77b65"
_gold2, _gold3 = SOL[_t2], SOL[_t3]
_recs = {_t1: [drec(_t1, [SOL[_t1][0]])], _t2: [drec(_t2, [_gold2[0], None])],
         _t3: [drec(_t3, [_gold3[0], CH[_t3]["test"][1]["input"], [[9, 9]]])]}
_tasks = {t: CH[t] for t in (_t1, _t2, _t3, "135a2760")}
_sub = E31D.build_submission_direct(_tasks, _recs)
check("T6 adapter: 1-test model grid; 2-test partial (grid + no grid -> input / mirror); 3-test mixed (grid, grid == input -> mirror, wrong); "
      "a task without a record -> the identity",
      _sub[_t1] == [{"attempt_1": SOL[_t1][0], "attempt_2": CH[_t1]["test"][0]["input"]}]
      and _sub[_t2][0]["attempt_1"] == _gold2[0] and _sub[_t2][1] == {"attempt_1": CH[_t2]["test"][1]["input"],
                                                                     "attempt_2": [list(reversed(r)) for r in CH[_t2]["test"][1]["input"]]}
      and _sub[_t3][1] == {"attempt_1": CH[_t3]["test"][1]["input"], "attempt_2": [list(reversed(r)) for r in CH[_t3]["test"][1]["input"]]}
      and _sub[_t3][2]["attempt_1"] == [[9, 9]] and _sub["135a2760"] == [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in CH["135a2760"]["test"]])
_c7 = READ[5]
_a = _c7.index("        if SOLUTIONS is not None:\n            hits = pass1 = pass2 = total = 0")
_b = _c7.index("            try:    # E15B A2", _a)
SCORER = textwrap.dedent(_c7[_a:_b])


def cell7_score(gen, recs, tids):
    g = {"SOLUTIONS": {t: SOL[t] for t in tids}, "TASK_IDS": list(tids), "_gen": gen, "_recs": recs, "report": {}}
    exec(compile(SCORER, "<CELL 7 scorer (the notebook's text)>", "exec"), g)
    return g["report"]


_r7 = cell7_score(_sub, _recs, [_t1, _t2, _t3])
check("T6 the notebook's own CELL 7 scorer over the adapter: 1 + 1 + 1 hits of 6 queries (pass1 3), task-norm (1 + 0.5 + 1/3) / 3",
      _r7["eval_query_hits"] == 3 and _r7["eval_query_total"] == 6 and _r7["eval_pass1_hits"] == 3
      and abs(_r7["eval_task_norm_pct"] - round(100 * (1 + 0.5 + 1 / 3) / 3, 2)) < 1e-9, _r7)
_g28, _per = E28L.score_from_generated(_sub, {t: SOL[t] for t in (_t1, _t2, _t3)}, [_t1, _t2, _t3])
check("T6 the E28 report's score_from_generated agrees with CELL 7 on the same attempts",
      all(_g28[k] == _r7[k] for k in ("eval_query_hits", "eval_query_total", "eval_pass1_hits", "eval_pass2_hits", "eval_task_norm_pct")))
_rows = E28L.attempt_rows(_recs, {t: SOL[t] for t in (_t1, _t2, _t3)}, {"T1": 104000})
check("T6 the E28 report's per-attempt rows accept the direct records (one row per task, branch 0)",
      len(_rows) == 3 and all(r["branch"] == 0 for r in _rows))
# DIRECT wrong + a CPU train-exact rule right -> scored WRONG (Duck's selector would have added the rule output)
_rt = {"train": [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}, {"input": [[5, 6, 7]], "output": [[7, 6, 5]]}],
       "test": [{"input": [[8, 9]]}]}
_rgold = [[9, 8]]
_rrec = {"task_id": "rule0001", "branch": 0, "status": "done", "publish_seq": 1, "accepted": False,
         "submissions": [{"test_outputs": [[[0, 0]]], "accepted": False, "train_exact": [], "t": 1.0, "branch": 0}],
         "e31_attempts": [{"attempt_1": [[0, 0]], "attempt_2": [[8, 9]]}]}
_ds = E31D.build_submission_direct({"rule0001": _rt}, {"rule0001": [_rrec]})
CH["rule0001"], SOL["rule0001"] = _rt, [_rgold]
_r7r = cell7_score(_ds, {"rule0001": [_rrec]}, ["rule0001"])
check("T6 regression: DIRECT wrong + a CPU flip_h rule right -> the E31 adapter scores it WRONG (attempts = model grid, test input)",
      _ds["rule0001"] == [{"attempt_1": [[0, 0]], "attempt_2": [[8, 9]]}] and _r7r["eval_query_hits"] == 0)
if Path(DUCK).is_file() and sha(Path(DUCK).read_bytes()) == "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128":
    try:
        _src = Path(DUCK).read_text(encoding="utf-8")
        _dm = types.ModuleType("arc2_duck_sel")
        _sel_src = _src[_src.index("def _rl_rot90(g):"):_src.index("def build_submission(")]
        _hdr = ("from __future__ import annotations\nfrom typing import Any, Callable\n"   # = arc2_duck.py's own header (Py3.12)
                "import json, os, logging\nlog = logging.getLogger('x')\n" +
                _src[_src.index("def valid_grid("):_src.index("\n\n\n", _src.index("def valid_grid("))] + "\n\n" +
                "def _b_mode():\n    return os.environ.get('E12_B_MODE', '') == '1'\n\n"
                "def _env_int(n, d=0):\n    return int(os.environ.get(n, d) or d)\n\n")
        exec(compile(_hdr + _sel_src, "<arc2_duck selection (payload text)>", "exec"), _dm.__dict__)
        os.environ["E12_B_MODE"] = "1"
        _duck_rows = _dm.select_attempts_r5(_rt, [_rrec])
        os.environ.pop("E12_B_MODE")
        check("T6 regression (contrast): Duck's real select_attempts_r5 under E12_B_MODE=1 puts the CPU rule's right answer in "
              "attempt_2 for the same record -> that is why E31 never uses it", _duck_rows[0]["attempt_2"] == _rgold
              and cell7_score({"rule0001": _duck_rows}, {"rule0001": [_rrec]}, ["rule0001"])["eval_query_hits"] == 1, _duck_rows)
    except Exception as e:
        check("T6 Duck select_attempts_r5 contrast", False, repr(e))
else:
    skip("T6 Duck select_attempts_r5 contrast", "the payload arc2_duck.py (sha b9f67010) is not available")
del CH["rule0001"], SOL["rule0001"]

# CELL 7b itself (the notebook's text) with stubs: the rebind, the DIRECT-only adapter, the base_url assertion
_e12a = types.SimpleNamespace(solve_all_r6=lambda *a, **k: "duck", build_submission=lambda t, r: "duck_sel",
                              SERVER_DOWN=threading.Event(), SCHED_LAST={})
_calls = []
_ns = {"E12A": _e12a, "E31D": types.SimpleNamespace(LIB_VERSION="x", STOP_THINKING_TEXT=E31D.STOP_THINKING_TEXT, FN_STOP_IDS=E31D.FN_STOP_IDS,
                                                    solve_all_direct=lambda *a, **k: _calls.append(k) or {"ok": 1},
                                                    build_submission_direct=E31D.build_submission_direct),
       "json": json, "time": time, "Path": Path, "CONFIG": {}, "E31_CFG": dict(E31D.CFG_DEFAULT), "E31_BUDGET_MIN": 205,
       "K100_DEADMAN_SEC": 12300, "SGL_URL": "http://127.0.0.1:8001", "NB5_PORT": 8001,
       "E11_SERVER": types.SimpleNamespace(url="http://127.0.0.1:8001"), "NB5_LOCAL_TOK": {"verified": True, "hf": None, "template": None},
       "CKPT": TOKDIR, "TOK": TOK, "SOLUTIONS": None, "BASE_OUT": TMP, "OUT": TMP, "e21_counters": lambda: {}, "NOTEBOOK_T0": 0.0,
       "E11_HARNESS": True, "print": lambda *a, **k: None, "E12_RUN": {"streams": 24}}
exec(compile(CODE[6], "<CELL 7b>", "exec"), _ns)
check("T6 CELL 7b rebinds E12A.solve_all_r6 / build_submission (originals kept once); build_submission is DIRECT-only",
      _e12a.solve_all_r6 is _ns["e31_solve_all_r6"] and _e12a.build_submission is _ns["e31_build_submission"]
      and _e12a.build_submission(_tasks, _recs) == _sub and _e12a._e31_orig_build_submission({}, {}) == "duck_sel")
if TOK is not None:
    _r = _e12a.solve_all_r6({_t1: CH[_t1]}, [_t1], out_dir=TMP, deadline=time.monotonic() + 100, base_url="http://127.0.0.1:8001/v1",
                            model_id="q27", workers=6, branches=2, first_exposure_tokens=None)
    _w24 = _calls[-1]["workers"]
    _ns["E12_RUN"] = {"streams": 16}
    _e12a.solve_all_r6({_t1: CH[_t1]}, [_t1], out_dir=TMP, deadline=time.monotonic() + 100, base_url="http://127.0.0.1:8001/v1",
                       model_id="q27", workers=24)
    _w16 = _calls[-1]["workers"]
    _ns["E12_RUN"] = {"streams": 24}
    check("T6 CELL 7b workers = min(E31_CFG workers, E12_RUN streams): the isolation-gated Duck `workers` (6) is not used -> 24; "
          "streams 16 -> 16", _w24 == 24 and _w16 == 16 and _ns["E31_RUN"]["calls"][0]["workers_arg"] == 6, (_w24, _w16))
    check("T6 CELL 7b passes SGL_URL (:8001), workers min(24, E31_CFG), E12A.SERVER_DOWN / SCHED_LAST, BASE_OUT, the Duck-only args recorded",
          _r == {"ok": 1} and _calls[0]["base_url"] == "http://127.0.0.1:8001" and _calls[0]["workers"] == 24
          and _calls[0]["server_down"] is _e12a.SERVER_DOWN and _calls[0]["sched_last"] is _e12a.SCHED_LAST
          and _calls[0]["ignored"] == ["branches", "first_exposure_tokens"] and _calls[0]["tok"]["eos_ids"] == [248046, 248044])
    _ns["SGL_URL"], _ns["E11_SERVER"] = "http://127.0.0.1:8011", types.SimpleNamespace(url="http://127.0.0.1:8011")
    try:
        _e12a.solve_all_r6({_t1: CH[_t1]}, [_t1], out_dir=TMP, deadline=time.monotonic() + 100, base_url="http://127.0.0.1:8011/v1",
                           model_id="q27", workers=24)
        _bad = False
    except AssertionError:
        _bad = True
    check("T6 CELL 7b refuses a URL that is not the booted NB5_PORT server (E18's 8011)", _bad)
# the E15B writer (e21_cell6's e12_write_submission, the notebook's text) in the RERUN overlay with the adapter
_w0 = _c7.index("    def e12_write_submission():\n")
_w1 = _c7.index("\n\n\n", _w0)
_wsrc = textwrap.dedent(_c7[_w0:_w1])
_sp = TMP / "submission.json"
_test = {t: {"train": CH[t]["train"], "test": [{"input": q["input"]} for q in CH[t]["test"]]} for t in (_t2, _t3, "135a2760")}


def _wj(path, obj):
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


_wg = {"RERUN": True, "E12_RECORDS": {_t2: _recs[_t2], _t3: _recs[_t3], "135a2760": []}, "CHALLENGES": _test, "test_challenges": _test,
       "identity_submission": lambda ch: {t: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in v["test"]] for t, v in ch.items()},
       "_E12_WRITE_LOCK": threading.Lock(), "_E12_LOCK": threading.Lock(), "_e12_write_json_atomic": _wj, "SUBMISSION_PATH": _sp,
       "E12A": _e12a}
exec(compile(_wsrc, "<e12_write_submission (E15B writer)>", "exec"), _wg)
_wg["e12_write_submission"]()
_ws = json.loads(_sp.read_text(encoding="utf-8"))
check("T6 the E15B writer (rerun overlay, the notebook's text) + the adapter: direct rows for the direct tasks, identity otherwise",
      _ws[_t2] == _sub[_t2] and _ws[_t3] == _sub[_t3] and _ws["135a2760"] == _wg["identity_submission"]({"x": _test["135a2760"]})["x"])

# ---------------------------------------------------------------- T7 the solver against an in-process fake streaming server
class Fake:
    """A token-id streaming /v1/completions server (vLLM 0.32 chunk shape: choices[{token_ids, finish_reason}]), scripted per task
    from the prompt ids; /health; every request body recorded; no prefix cache (a continuation can only be token-exact)."""

    def __init__(self, scripts, prompts, tok):
        self.scripts, self.prompts, self.tok = scripts, prompts, tok
        self.bodies, self.healthy, self.lock, self.p1_count, self.closed = [], True, threading.Lock(), {}, 0
        self.filler = tok["encode"]("Let me think about the rule step by step. ")
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                code = 200 if fake.healthy else 503
                self.send_response(code); self.send_header("Content-Length", "0"); self.send_header("Connection", "close")
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with fake.lock:
                    fake.bodies.append(dict(body, _port=self.server.server_address[1]))
                fake.serve(self, body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def task_of(self, ids):
        for t, p in self.prompts.items():
            if ids[:len(p)] == p:
                return t, len(ids) - len(p)
        return None, 0

    def serve(self, h, body):
        ids = list(body["prompt"])
        tid, extra = self.task_of(ids)
        sc = dict(self.scripts.get(tid) or {"kind": "natural"})
        mt = int(body["max_tokens"])
        p1 = (int(body["seed"]) - E31D.SEED) % 10 == 1           # phase 1 (also a token-exact phase-1 resend) by its seed
        if p1:
            with self.lock:
                self.p1_count[tid] = self.p1_count.get(tid, 0) + 1
                n_p1 = self.p1_count[tid]
        ans = self.tok["encode"](sc.get("answer", "\n\n```grid\n1 1\n```"))
        if p1:                                           # phase 1
            k = sc["kind"]
            if k == "natural":
                out, fin = self.filler * sc.get("think_rep", 3) + [248069] + ans + [248046], "stop"
            elif k == "cut_answer":                      # closes thinking, the answer is cut by max_tokens (continue)
                out, fin = (self.filler * 2 + [248069] + ans + [248046])[:mt], "length"
            elif k in ("never_close", "slow"):
                out, fin = None, "length"
            elif k == "error_once":                      # k streamed tokens, then the connection breaks (a kill)
                if n_p1 == 1:
                    return self.stream(h, (self.filler * 3)[:sc.get("k", 24)], None, broken=True)
                out, fin = self.filler * 2 + [248069] + ans + [248046], "stop"
            elif k == "http500_once":                    # an HTTP 500 with 0 tokens
                if n_p1 == 1:
                    b_ = json.dumps({"object": "error", "message": "fake 500"}).encode()
                    h.send_response(500); h.send_header("Content-Type", "application/json"); h.send_header("Content-Length", str(len(b_)))
                    h.send_header("Connection", "close"); h.end_headers(); h.wfile.write(b_); return
                out, fin = self.filler * 2 + [248069] + ans + [248046], "stop"
            elif k == "error_always":
                return self.stream(h, self.filler[:4], None, broken=True)
            elif k == "error_unhealthy":                 # a kill, and /health stays down until heal_at (after the drain)
                if n_p1 == 1:
                    self.healthy = False
                    threading.Timer(sc["heal_after"], lambda: setattr(self, "healthy", True)).start()
                    return self.stream(h, self.filler * 2, None, broken=True)
                out, fin = self.filler * 2 + [248069] + ans + [248046], "stop"
            elif k == "eos_before_close":
                out, fin = self.filler * 2 + [248046], "stop"
            else:
                raise ValueError(k)
            if out is None:
                return self.stream(h, None, "length", endless=mt, delay=sc.get("delay", 0.0))
            if len(out) > mt:
                out, fin = out[:mt], "length"
            return self.stream(h, out, fin, delay=sc.get("delay", 0.0))
        tail = ids[-25:] == E31D.FN_STOP_IDS
        if sc.get("p2_slow"):
            return self.stream(h, None, "length", endless=mt, delay=0.02)
        if sc.get("p2_error_always"):
            return self.stream(h, ans[:2], None, broken=True)
        out = (ans + [248046]) if tail else (self.tok["encode"]("```grid\n1 1\n```") + [248046])
        return self.stream(h, out[:mt], "stop" if len(out) <= mt else "length")

    def stream(self, h, out, fin, endless=0, delay=0.0, broken=False):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream"); h.send_header("Transfer-Encoding", "chunked")
        h.send_header("Connection", "close"); h.end_headers(); h.close_connection = True

        def send(d):
            b = b"data: " + json.dumps(d).encode() + b"\n\n"
            h.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n"); h.wfile.flush()
        try:
            if endless:
                sent = 0
                while sent < endless:
                    n = min(8, endless - sent); sent += n
                    send({"choices": [{"index": 0, "text": "", "token_ids": (self.filler * 3)[:n],
                                       "finish_reason": "length" if sent >= endless else None}]})
                    time.sleep(delay or 0.01)
            else:
                for i in range(0, len(out), 8):
                    last = i + 8 >= len(out)
                    send({"choices": [{"index": 0, "text": "", "token_ids": out[i:i + 8],
                                       "finish_reason": (fin if last and not broken else None)}]})
                    if delay:
                        time.sleep(delay)
                if broken:
                    h.wfile.flush(); h.connection.shutdown(2); return
            b = b"data: [DONE]\n\n"
            h.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n0\r\n\r\n"); h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            with self.lock:
                self.closed += 1


def run_solver(tasks, scripts, deadline_sec, cfg=None, **kw):
    prompts = {t: TOKD["encode"](TOKD["render"]([{"role": "user", "content": E31D.task_message(tasks[t])}])) for t in tasks}
    fake = Fake(scripts, prompts, TOKD)
    d = Path(tempfile.mkdtemp(prefix="e31run_", dir=TMP))
    done = []
    sl = {}
    t0 = time.monotonic()
    res = E31D.solve_all_direct(tasks, sorted(tasks), out_dir=d / "duck", deadline=time.monotonic() + deadline_sec, base_url=fake.url,
                                model_id="q27", on_branch_done=done.append, workers=kw.pop("workers", 24),
                                cfg=dict({"think": 600, "answer": 64, "drain_min": 0.1}, **(cfg or {})), tok=TOKD,
                                solutions={t: SOL[t] for t in tasks if t in SOL}, report_dir=d, sched_last=sl, harness=True,
                                log=lambda *a, **k: None, **kw)
    return {"res": res, "fake": fake, "done": done, "sl": sl, "dir": d, "sec": time.monotonic() - t0, "prompts": prompts,
            "rep": json.loads((d / "e31_direct_report.json").read_text(encoding="utf-8")),
            "rows": [json.loads(l) for l in (d / "e31_direct_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}


if TOK is not None:
    _old = (E31D.ADMIT_STOP_BEFORE_END_SEC, E31D.HARD_BEFORE_END_SEC)
    E31D.ADMIT_STOP_BEFORE_END_SEC, E31D.HARD_BEFORE_END_SEC = 2.0, 1.0      # short test windows (the GPU run: 12 min / 1 min)
    try:
        _ts = {t: CH[t] for t in ("7b5033c1", "a47bf94d", "142ca369", "b6f77b65", "1818057f", "136b0064")}
        _gold = lambda t: "\n\n" + "".join(blk(f"grid {k + 1}" if len(SOL[t]) > 1 else "grid", g) for k, g in enumerate(SOL[t]))
        _sc = {"7b5033c1": {"kind": "natural", "answer": _gold("7b5033c1")},
               "a47bf94d": {"kind": "never_close", "answer": _gold("a47bf94d")},
               "142ca369": {"kind": "cut_answer", "answer": _gold("142ca369")},
               "b6f77b65": {"kind": "natural", "think_rep": 1, "answer": "\n\n" + blk("grid 2", SOL["b6f77b65"][1]) + blk("grid 1", SOL["b6f77b65"][0])},
               "1818057f": {"kind": "eos_before_close", "answer": _gold("1818057f")},
               "136b0064": {"kind": "error_once", "k": 24, "answer": _gold("136b0064")}}
        R = run_solver(_ts, _sc, 120.0, cfg={"think": 600, "answer": 2048})
        _st = R["sl"]["e31"]
        check("T7 early return: every task done -> the solver returned long before the window end (no wait), no live stream",
              R["sec"] < 60 and _st["returned_early"] is True and _st["ended_before_deadline_sec"] > 50 and _st["live_streams_at_return"] == 0
              and len(R["done"]) == 6, {"sec": R["sec"], "sched": {k: _st.get(k) for k in ("returned_early", "ended_before_deadline_sec")}})
        _bm = {b["prompt"][0]: b for b in R["fake"].bodies}
        _p1 = [b for b in R["fake"].bodies if any(b["prompt"] == p for p in R["prompts"].values())]
        check("T7 every request: /v1/completions on THIS server's port, prompt = token ids, every sampling field explicit "
              "(T 0.6, top_p 0.95, top_k 20, min_p 0, penalties neutral, seed, stream, return_token_ids, skip_special_tokens False)",
              all(b["_port"] == R["fake"].port and b["temperature"] == 0.6 and b["top_p"] == 0.95 and b["top_k"] == 20 and b["min_p"] == 0.0
                  and b["presence_penalty"] == 0.0 and b["frequency_penalty"] == 0.0 and b["repetition_penalty"] == 1.0
                  and b["stream"] is True and b["return_token_ids"] is True and b["skip_special_tokens"] is False and b["ignore_eos"] is False
                  and isinstance(b["seed"], int) and all(isinstance(x, int) for x in b["prompt"]) for b in R["fake"].bodies))
        _idx = {t: i for i, t in enumerate(sorted(_ts))}
        check("T7 seeds per task: phase 1 = SEED + 10 * sorted index + 1 (phase 2: + 2); phase-1 max_tokens = the cap",
              all(b["seed"] == E31D.task_seed(_idx[t], 1) and b["max_tokens"] == E31D.phase1_cap(len(R["prompts"][t]), 600, 2048 * len(_ts[t]["test"]))
                  for t in _ts for b in _p1 if b["prompt"] == R["prompts"][t]))
        _recs7 = {r["task_id"]: r for r in R["done"]}
        _e = lambda t: _recs7[t]["e31"]
        _tid = json.loads((R["dir"] / "duck" / "a47bf94d_b0" / "token_ids.json").read_text())
        _p2b = [b for b in R["fake"].bodies if b["prompt"][:len(R["prompts"]["a47bf94d"])] == R["prompts"]["a47bf94d"]
                and len(b["prompt"]) > len(R["prompts"]["a47bf94d"])]
        check("T7 never-closing thinking -> phase 1 cut at the cap (length) -> forced close: phase-2 prompt == prompt ids + phase-1 ids "
              "+ the 25 STOP_IDS exactly (ids only, no re-tokenization), seed + 2, answer cap = answer x tests; the grid parsed and scored",
              _e("a47bf94d")["mode"] == "forced_close" and _e("a47bf94d")["p1_finish"] == "length" and len(_p2b) == 1
              and _p2b[0]["prompt"] == R["prompts"]["a47bf94d"] + _tid["p1"] + E31D.FN_STOP_IDS and _tid["p2_prefix"] == E31D.FN_STOP_IDS
              and _p2b[0]["seed"] == E31D.task_seed(_idx["a47bf94d"], 2) and _p2b[0]["max_tokens"] == 2048 * len(SOL["a47bf94d"])
              and _e("a47bf94d")["score"] == [{"hit": 1, "pass1": 1}] * len(SOL["a47bf94d"]), _e("a47bf94d"))
        check("T7 natural close in phase 1 (stop after </think>) -> mode natural, one request, the gold grid hit",
              _e("7b5033c1")["mode"] == "natural" and _e("7b5033c1")["end_reason"] == "natural" and _e("7b5033c1")["p2_cap"] is None
              and _e("7b5033c1")["score"] == [{"hit": 1, "pass1": 1}])
        _p2c = [b for b in R["fake"].bodies if b["prompt"][:len(R["prompts"]["142ca369"])] == R["prompts"]["142ca369"]
                and len(b["prompt"]) > len(R["prompts"]["142ca369"])]
        _tc = json.loads((R["dir"] / "duck" / "142ca369_b0" / "token_ids.json").read_text())
        check("T7 thinking closed but the answer cut (length) -> 'continue': phase-2 prompt == prompt + phase-1 ids (no stop text)",
              _e("142ca369")["mode"] == "continue" and len(_p2c) == 1 and _p2c[0]["prompt"] == R["prompts"]["142ca369"] + _tc["p1"]
              and _tc["p2_prefix"] == [])
        _te = json.loads((R["dir"] / "duck" / "1818057f_b0" / "token_ids.json").read_text())
        check("T7 an EOS before </think> is stripped before the forced close (ORDER2 F8): phase-1 ids end without 248046, + STOP_IDS",
              _e("1818057f")["mode"] == "forced_close" and _te["p1"][-1] not in (248046, 248044) and _te["p2_prefix"] == E31D.FN_STOP_IDS)
        check("T7 3-test task, the answer labels 'grid 2' then 'grid 1' and omits 3 -> label mapping (query 0 / 1 by label, query 2 "
              "fallback; never positional); per-test rows written",
              _e("b6f77b65")["mapping_source"] == ["label", "label", "fallback"] and _e("b6f77b65")["score"][:2] == [{"hit": 1, "pass1": 1}] * 2
              and _e("b6f77b65")["fallback_reason"] == [None, None, "label_missing"] and _recs7["b6f77b65"]["e31_queries"][0]["query_index"] == 0
              and sum(1 for r in R["rows"] if r["task_id"] == "b6f77b65") == 3, _e("b6f77b65"))
        _pk = R["prompts"]["136b0064"]
        _b1 = [b for b in R["fake"].bodies if b["prompt"][:len(_pk)] == _pk and (b["seed"] - E31D.SEED) % 10 == 1]
        _k24 = (R["fake"].filler * 3)[:24]
        check("T7 a kill after k = 24 streamed tokens before the drain: never a finish, never a forced close -> phase 1 RESENT "
              "token-exact (prompt + the 24 kept ids), cap = the phase-1 cap - 24, the same seed; then it closed naturally",
              _st["request_errors"] == 1 and _st["error_retries"] == 1 and _st.get("phase1_resent") == 1 and len(_b1) == 2
              and _b1[0]["prompt"] == _pk and _b1[1]["prompt"] == _pk + _k24 and _b1[1]["max_tokens"] == _b1[0]["max_tokens"] - 24
              and _b1[1]["seed"] == _b1[0]["seed"] and _e("136b0064")["mode"] == "natural" and _e("136b0064")["end_reason"] == "natural"
              and _e("136b0064")["p2_cap"] is None and len(_e("136b0064")["request_errors"]) == 1
              and json.loads((R["dir"] / "duck" / "136b0064_b0" / "token_ids.json").read_text())["p1"][:24] == _k24,
              {"bodies": [(len(b["prompt"]) - len(_pk), b["max_tokens"], b["seed"]) for b in _b1], "e31": _e("136b0064")})
        check("T7 E12A.SCHED_LAST populated (Duck's keys: gauges, stragglers, workers, tasks_initialized, the e31 section)",
              all(k in R["sl"] for k in ("gauges", "stragglers", "workers", "tasks_initialized", "tasks_with_request_sent", "e31"))
              and R["sl"]["tasks_initialized"] == 6)
        check("T7 one completed branch record per task through on_branch_done (E28 contract: branch 0, status done, submissions"
              "[{test_outputs: one per test}], accepted False, t0 / t1 epoch, publish_seq)",
              sorted(r["task_id"] for r in R["done"]) == sorted(_ts) and all(r["branch"] == 0 and r["status"] == "done" and r["accepted"] is False
                  and len(r["submissions"][0]["test_outputs"]) == len(_ts[r["task_id"]]["test"]) and r["t1"] >= r["t0"] > 1e9 for r in R["done"]))
        _rp = R["rep"]
        check("T7 report: groups with query_hits / fully_solved_tasks / task_norm separately, multi vs single, coverage 60k / 78k / 104k, "
              "speed (solve wall, aggregate tok/s, per-request tok/s), the reference comparison, the E28 Duck reference",
              all(k in _rp["groups"]["all"] for k in ("query_hits", "fully_solved_tasks", "task_norm_pct")) and "multi_test" in _rp["groups"]
              and "single_test" in _rp["groups"] and set(_rp["think_coverage"]) == {"60000", "78000", "104000"}
              and _rp["speed"]["solve_wall_sec"] > 0 and _rp["speed"]["aggregate_decode_tok_s"] > 0 and _rp["speed"]["per_request_tok_s_p50"]
              and set(_rp["comparison_vs_reference"]) == {"K100", "W4", "Direct27B", "Duck"} and _rp["e28_duck_reference"]["solver_tok_s"] == 419.5)
        check("T7 the full thinking of every request saved (transcript.txt + token_ids.json per task)",
              all((R["dir"] / "duck" / f"{t}_b0" / "transcript.txt").is_file() and (R["dir"] / "duck" / f"{t}_b0" / "token_ids.json").is_file() for t in _ts))
        # the drain: 24 slow never-closing phase-1 streams until drain_at = T - drain_min, then 24 forced-close requests
        _t24 = {t: CH[t] for t in sorted(CH)[:24]}
        R2 = run_solver(_t24, {t: {"kind": "slow", "delay": 0.01} for t in _t24}, 9.0, cfg={"think": 100000, "drain_min": 0.1})
        _s2 = R2["sl"]["e31"]
        _p2s = [b for b in R2["fake"].bodies if b["prompt"][-25:] == E31D.FN_STOP_IDS]
        _okpref = all(any(b["prompt"] == R2["prompts"][t] + json.loads((R2["dir"] / "duck" / f"{t}_b0" / "token_ids.json").read_text())["p1"]
                          + E31D.FN_STOP_IDS for b in _p2s) for t in _t24)
        check("T7 drain: 24 simultaneous phase-1 streams running at drain_at -> all 24 aborted (ids kept) -> 24 forced-close phase-2 "
              "requests (prefix token-exact), every answer parsed, finished before the hard abort, no orphan stream",
              _s2["max_active"] == 24 and _s2["drained"] == 24 and _s2["phase2_forced"] == 24 and len(_p2s) == 24 and _okpref
              and all(r["e31"]["p1_finish"] == "drained" and r["e31"]["mode"] == "forced_close" for r in R2["done"]) and len(R2["done"]) == 24
              and _s2["aborted_at_hard_stop"] == 0 and _s2["live_streams_at_return"] == 0 and R2["sec"] < 9.0,
              {k: _s2.get(k) for k in ("max_active", "drained", "phase2_forced", "aborted_at_hard_stop", "live_streams_at_return")})
        check("T7 admission stops at T - admit_stop: no phase-1 request starts after it (24 workers, 24 tasks: all admitted at once)",
              _s2["admitted"] == 24 and _s2["not_admitted"] == 0)
        # the hard abort at T - 1 min: phase 2 that never finishes is aborted and its partial text parsed (ORDER2 hard stop)
        _t1h = {t: CH[t] for t in ("7b5033c1",)}
        R3 = run_solver(_t1h, {"7b5033c1": {"kind": "slow", "p2_slow": True}}, 5.0, cfg={"think": 100000, "drain_min": 0.05, "answer": 100000})
        _s3 = R3["sl"]["e31"]
        check("T7 hard abort: a phase 2 still running at T - hard is aborted, the task ends 'hard_stop' with its partial answer, the solver "
              "returns before T", _s3["aborted_at_hard_stop"] == 1 and R3["done"][0]["e31"]["end_reason"] == "hard_stop" and R3["sec"] < 5.0
              and _s3["live_streams_at_return"] == 0)
        # admission stop: 2 workers, 4 slow tasks, the window closes before the queue empties -> not admitted (identity)
        _t4 = {t: CH[t] for t in sorted(CH)[:4]}
        R4 = run_solver(_t4, {t: {"kind": "slow", "delay": 0.01} for t in _t4}, 4.0, cfg={"think": 100000, "drain_min": 0.04}, workers=2)
        _s4 = R4["sl"]["e31"]
        _sub4 = E31D.build_submission_direct(_t4, R4["res"])
        check("T7 tasks not admitted before the admission stop get no record and the identity in the submission (ORDER2)",
              _s4["not_admitted"] == 2 and len(R4["done"]) == 2 and all(_sub4[t] == [{"attempt_1": q["input"], "attempt_2": q["input"]}
                                                                         for q in _t4[t]["test"]] for t in _s4["not_admitted_tasks"]))
        # an unrecovered error: phase 2 always breaks -> after max_error_retries the task ends 'error' (never a success)
        R5 = run_solver({"7b5033c1": CH["7b5033c1"]}, {"7b5033c1": {"kind": "never_close", "p2_error_always": True}}, 30.0)
        check("T7 an unrecovered request error ends the task as end_reason 'error' after the bounded retries (never counted a finish)",
              R5["done"][0]["e31"]["end_reason"] == "error" and R5["sl"]["e31"]["error_finishes"] == 1 and R5["sl"]["e31"]["request_errors"] == 3)
        # merge list 1: phase-1 request errors -> resend before the drain, forced close only at / after it, bounded retries
        _t1 = {"7b5033c1": CH["7b5033c1"]}
        _p7 = lambda R_: [b for b in R_["fake"].bodies if (b["seed"] - E31D.SEED) % 10 == 1]
        _q7 = lambda R_: [b for b in R_["fake"].bodies if (b["seed"] - E31D.SEED) % 10 == 2]
        R6 = run_solver(_t1, {"7b5033c1": {"kind": "http500_once", "answer": _gold("7b5033c1")}}, 30.0)
        check("T7 an HTTP 500 with 0 tokens before the drain -> phase 1 resent with the same prompt / cap / seed (no forced close, no "
              "phase 2), then natural; the error counted, never a finish",
              len(_p7(R6)) == 2 and _p7(R6)[1]["prompt"] == _p7(R6)[0]["prompt"] and _p7(R6)[1]["max_tokens"] == _p7(R6)[0]["max_tokens"]
              and _p7(R6)[1]["seed"] == _p7(R6)[0]["seed"] and not _q7(R6) and R6["done"][0]["e31"]["mode"] == "natural"
              and R6["sl"]["e31"]["request_errors"] == 1 and R6["sl"]["e31"].get("phase1_resent") == 1
              and R6["done"][0]["e31"]["score"] == [{"hit": 1, "pass1": 1}], R6["done"][0]["e31"])
        R7 = run_solver(_t1, {"7b5033c1": {"kind": "error_unhealthy", "heal_after": 4.0, "answer": _gold("7b5033c1")}}, 7.0,
                        cfg={"think": 100000, "drain_min": 0.05})
        _f7 = [b for b in _q7(R7) if b["prompt"][-25:] == E31D.FN_STOP_IDS]
        check("T7 a phase-1 error whose server comes back only AFTER the drain (T - drain_min) -> the forced close (STOP_IDS), not a resend",
              R7["done"][0]["e31"]["mode"] == "forced_close" and R7["done"][0]["e31"]["p1_finish"] == "error" and len(_p7(R7)) == 1
              and len(_f7) == 1 and not R7["sl"]["e31"].get("phase1_resent") and R7["sl"]["e31"]["request_errors"] == 1,
              {"e31": R7["done"][0]["e31"], "sched": {k: R7["sl"]["e31"].get(k) for k in ("request_errors", "phase1_resent", "drain_pending")}})
        R8 = run_solver(_t1, {"7b5033c1": {"kind": "error_always"}}, 30.0)
        check("T7 phase 1 always breaks -> resent twice (the bounded retries), then end_reason 'error'; never a forced close or a phase 2",
              R8["done"][0]["e31"]["end_reason"] == "error" and len(_p7(R8)) == 3 and not _q7(R8) and R8["sl"]["e31"].get("phase1_resent") == 2
              and R8["sl"]["e31"]["error_finishes"] == 1, R8["done"][0]["e31"])
        # merge list 4: fail closed before any request when the stop ids / </think> id differ from the spec
        _badtok = dict(TOKD, encode=lambda s: TOKD["encode"](s.replace("directly now", "directly")) if "Considering" in s else TOKD["encode"](s))
        _fk = Fake({}, {}, TOKD)
        _dd = Path(tempfile.mkdtemp(prefix="e31fc_", dir=TMP))
        try:
            E31D.solve_all_direct(_t1, ["7b5033c1"], out_dir=_dd / "duck", deadline=time.monotonic() + 30, base_url=_fk.url, model_id="q27",
                                  cfg={"think": 600, "answer": 64}, tok=_badtok, report_dir=_dd, harness=True, log=lambda *a, **k: None)
            _fc = None
        except RuntimeError as e:
            _fc = str(e)
        check("T7 fail closed: STOP_IDS != the 25-id spec -> RuntimeError FAIL_CLOSED_TOKEN_IDS before any request (the fake saw "
              "nothing), the reason written to e31_direct_report.json",
              _fc and "FAIL_CLOSED_TOKEN_IDS" in _fc and not _fk.bodies
              and json.loads((_dd / "e31_direct_report.json").read_text())["status"] == "FAIL_CLOSED_TOKEN_IDS", _fc)
        # the server down (watchdog relaunching): no admission, no continuation until /health is back
        _ev = threading.Event(); _ev.set()
        _tdn = {"7b5033c1": CH["7b5033c1"]}
        _box = {}

        def _run_down():
            _box["R"] = run_solver(_tdn, {}, 30.0, server_down=_ev)
        _th = threading.Thread(target=_run_down); _th.start(); time.sleep(1.5)
        _ev.clear(); _th.join(30)
        check("T7 while E12A.SERVER_DOWN is set the solver admits nothing; after it clears the task runs normally",
              _box["R"]["sl"]["e31"]["server_down_polls"] > 0 and _box["R"]["done"][0]["e31"]["end_reason"] == "natural")
    finally:
        E31D.ADMIT_STOP_BEFORE_END_SEC, E31D.HARD_BEFORE_END_SEC = _old
else:
    skip("T7 solver vs the fake server", "no tokenizer")
check("T7 the GPU-run phase constants: no admission from T - 12 min, abort at T - 1 min; drain_min 23 >= 3 x 8,192 / 18 tok/s / 60",
      E31D.ADMIT_STOP_BEFORE_END_SEC == 720 and E31D.HARD_BEFORE_END_SEC == 60 and E31D.drain_need_min(3, 8192) <= 23 <= E31D.CFG_DEFAULT["drain_min"])

# ---------------------------------------------------------------- T8 the request fields vs the Merlin vLLM 0.32 source (optional)
_proto = Path(MERLIN_SRC) / "vllm/entrypoints/openai/completion/protocol.py"
_serv = Path(MERLIN_SRC) / "vllm/entrypoints/openai/completion/serving.py"
if _proto.is_file() and _serv.is_file():
    _pt, _sv = _proto.read_text(encoding="utf-8"), _serv.read_text(encoding="utf-8")
    _cls = _pt[_pt.index("class CompletionRequest(OpenAIBaseModel):"):]
    _body = E31D.request_body("q27", [1, 2], 10, 0.6, 1)
    check("T8 every request field is a CompletionRequest field of the Merlin vLLM 0.32 source; streamed chunks carry token_ids when "
          "return_token_ids (serving.py); skip_special_tokens / top_k / min_p / seed / penalties reach SamplingParams",
          all(re.search(rf"^\s+{k}: ", _cls, re.M) for k in _body) and "token_ids=(" in _sv and "if request.return_token_ids" in _sv
          and all(f"{k}=self.{k}" in _cls for k in ("skip_special_tokens", "ignore_eos", "seed")) and '"top_k": 0' in _cls and '"min_p": 0.0' in _cls,
          [k for k in _body if not re.search(rf"^\s+{k}: ", _cls, re.M)])
else:
    skip("T8 Merlin source field check", f"no Merlin source at {MERLIN_SRC} (the real kit checks the INSTALLED tree)")

n_bad = sum(1 for _, ok in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_bad}/{len(RESULTS)} PASS", flush=True)
sys.exit(1 if n_bad else 0)
