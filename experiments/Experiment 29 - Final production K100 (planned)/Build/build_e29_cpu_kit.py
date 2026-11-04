"""E29 FINAL Kaggle CPU kits (2026-11-01, the kit-must-mirror rule): independent CPU kernels on foysalemonshanto (the klee100 /
Merlin / tf-ref / e15b-payload inputs, the drafter Model and the new q27-e29-final-payload dataset are owned there), the notebooks'
pinned Py3.12 image, no GPU, no internet, private. Each kernel is fail-soft, writes its JSON report and prints exactly one line
`@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`. NOTHING is pushed here.

Built as a wrapper over the E28 K100 kit chain (build_e28_k100_cpu_kit.py, which wraps build_e26_k100q_cpu_kit.py; both
imported by exec of their source with exact-once retargeting edits, never written): the notebook under test is the E29 variant
(Build/<variant>_build_facts.json = E28's facts format + the e29 key); only the marked E29 edits below change the E28 kits.

  units-<v>    (v = default / dflash-d3s / mtp-v64) E28's units kit retargeted to the E29 notebook: e28k100_unit_tests.py with
               exact-once E29 edits (readable cells = BOTH packed literals unpacked (e21_cell6 E29, e21_cell7 E28), the five packed
               modules == the files, every hunk vs K100Q marked E28 / E29, every hunk vs the E28 K100 notebook marked E29, the C3 /
               C4 / G2 anchors at their E29 text, the nb5_boot ladder tests G3 / G4 with the E29 defs and no E29 lever on = E28),
               + e29_kit_lib_tests.py (e29_lib checks, the notebook's own nb5_boot with df / idx / vocab on, scripted tries)
               + E28's tf-ref / payload / Duck payload checks
  sched        e29_sched_tests.py (the scheduler S-1 module, the same file in all three notebooks) on the REAL Duck payload of
               the mounted q27-e15b-payload, the REAL K100Q / W4 branch records (bundled), the competition's eval challenges,
               nb5_lib.py; e28_sched_patch.py from E29_TEST_E28_DIR; run with sys.executable (never -I)
  real-<v>     (v = dflash-d3s / mtp-v64) E28's real kit on the E29 notebook (cells 0-4 in the commit + rerun namespace on the real
               mounts: the modules literal, inputs / merge / K6 / prompts, merlin_install, #60490, sha gate, Q-defer apply / revert,
               the scheduler (E29 S-1 applied / verify / revert / re-apply on the payload's arc2_duck.py), k100_setup (overlay +
               scales sidecar), I8 static + the INT8 quantizer probe, every E28 boot plan through merlin_fn_server + vLLM argparse,
               the MM probe, CELL 0 vs E18 R2, the submissions) + E29: e29_setup() on the real mounts (b: drafter_facts on the
               drafter Model, the seam from the new dataset via q27x_payload, SeamSeq apply + verify + the ppack still applied;
               c: the 64k vocab file found + sha), then the notebook's OWN nb5_boot with a scripted nb5_boot_try that calls the real
               merlin_fn_server (NB5_BOOT_EXTRA / NB5_BOOT_LEVERS exactly as nb5_boot builds them, e29_env / e29_extra, the D
               tries, the lever drops, the MTP fallback try): K6, launch.build(), host_phase(), the HELP check (KL.cli_flags_check,
               the notebook's FAIL_CLI_FLAGS gate) per try, the argv / env recorded; vLLM's argparse (VLLM_TARGET_DEVICE=cpu) on
               every E29 argv (indexer_kv_dtype fp8, dflash K3 spec, captures); b: NBDF's SpeculativeConfig preflight on the spec;
               c: Merlin's load_draft_vocab_ids(path, 248320, tp_size=2) with VLLM_DRAFT_VOCAB_SHA256 (+ a wrong sha refused);
               then the seam reverted (exact Merlin files, ppack still applied). The report carries a step-coverage table:
               every pre-launch step of the notebook, in notebook order -> the kit check(s) -> PASS / FAIL.
Rules: no Kaggle API calls, no git, no local whole-notebook / kit runs (user rule: long tests on Kaggle CPU kits only).
Run: python build_e29_cpu_kit.py [--local-units]  ->  D:/ARC AGI 2/_scratch/push_e25/q27-e29-kit-<name>__foysalemonshanto/
     --local-units: also runs the two quick unit files (retargeted E28 unit tests, e29_kit_lib_tests.py) locally per variant
"""
import ast
import base64
import glob
import hashlib
import json
import lzma
import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E28B = Path(r"D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build")
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
E28_KIT = E28B / "build_e28_k100_cpu_kit.py"
OWNER = "foysalemonshanto"
DS_ID = "foysalemonshanto/q27-e29-final-payload"
DRAFTER_MODEL = "foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1"
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
LOCAL = HERE / "_kit_local"                # local unit runs + the LF copies of the variant facts (see facts_lf)
VARIANTS = {"default": "E29_DEFAULT", "dflash-d3s": "E29_DFLASH_D3S", "mtp-v64": "E29_MTP_V64"}


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, what, count=1):
    assert text.count(old) == count, (what, text.count(old))
    return text.replace(old, new)


def lf(p):
    b = Path(p).read_bytes()
    assert b"\r" not in b, f"CR bytes in {p}"
    return b.decode("utf-8")


E28_KIT_SRC = lf(E28_KIT)
E28_KIT_SHA = sha(E28_KIT_SRC)
def facts_lf(name):
    """build_e29_final.py writes its facts JSON with write_text (CRLF on this Windows box); the E26 / E28 kit chain reads the facts
    with lf_text (no CR allowed): an LF copy (same JSON) under Build/_kit_local/facts/ is the chain's input."""
    src = (HERE / name).read_bytes()
    d = LOCAL / "facts"
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    data = src.replace(bytes([13, 10]), bytes([10]))
    assert 13 not in data and json.loads(data.decode("utf-8")) == json.loads(src.decode("utf-8"))
    if not out.is_file() or out.read_bytes() != data:
        out.write_bytes(data)
    return out


BUILD_FACTS = json.loads(lf(facts_lf("e29_build_facts.json")))
NBDF_FACTS = json.loads(lf(E27B / "nbdf_k100_build_facts.json"))
_ndcfg = json.loads(re.search(r'"K100": \{"boot": .*?"cfg": (\{.*?"prefill_admit_floor_tok_s": \d+\})', lf(E27B / "nbdf_k100_cell7_body.py")).group(1))
NBDF_K100_CFG = {"nbdf_q_pin": _ndcfg["q_pin"], "nbdf_d_retry": _ndcfg["d_retry"], "nbdf_q_captures": _ndcfg["q_captures"]}

# ------------------------------------------------------------------ E29 edits on the E28 kit builder source
E29_READABLE_SRC = r'''
E29_ENGINE_HEAD = "\n\n\n# ---- E29 FINAL rev 2 (size: the 400-kB commit limit): the E29 engine block"


def E28_READABLE(code):
    """E29: CELL 6/7 carries e21_cell6 (E29, the 400-kB limit) and e21_cell7 (E28) as sha-pinned packed literals; the
    readable cells (= the build's code export) put both texts back."""
    def _unpack(seg):
        call = next(n for n in ast.walk(ast.parse(seg)) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
        expr = ast.get_source_segment(seg, call.args[0].args[0])
        return eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<e29 packed>", "eval"))
    e = code[4]                                  # E29 rev 2: the DFlash notebook packs its E29 engine block too
    while E29_ENGINE_HEAD in e:                  # rev 2d: every packed block (the E29 engine block + the E21 donor block)
        h = e.index(E29_ENGINE_HEAD)
        x = e.index("exec(compile(", h)
        z = e.index("\n", x) + 1
        e = e[:h] + _unpack(e[x:z]) + e[z:]
    code = code[:4] + [e] + code[5:]
    c = code[5]
    j = c.index("# ---- E28 K100 (size: the 400-kB commit limit)")
    head = c[:j]
    if "# ---- E29 FINAL (size: the 400-kB commit limit)" in head:
        i6 = head.index("# ---- E29 FINAL (size: the 400-kB commit limit)")
        head = head[:i6] + _unpack(head[i6:])
    return code[:5] + [head + _unpack(c[j:])] + code[6:]
'''


def e28_kit_src(variant, slug):
    s = E28_KIT_SRC
    s = once(s, 'E28_FACTS = HERE / "e28k100_build_facts.json"\n',
             f'E28_FACTS = Path(r"{facts_lf(variant.lower() + "_build_facts.json")}")   # E29 kit: the LF copy\n', "facts")
    s = once(s, 'SLUG, OWNER = "q27-e28-k100-kit", "esfersami50"\n', f'SLUG, OWNER = "{slug}", "{OWNER}"   # E29 kit\n', "slug")
    a = s.index("_E28_READABLE_SRC = r'''")
    b = s.index("'''\n", a + 30) + 4
    s = s[:a] + "_E28_READABLE_SRC = " + repr(E29_READABLE_SRC) + "   # E29 kit: both packed literals\n" + s[b:]
    s = once(s, 'assert {n: sha((HERE / n).read_bytes()) for n in MODS} == MODS, "the Build modules != the ones packed in the notebook (rebuild)"\n',
             'assert {n: sha(E29_MOD_TEXT(n)) for n in MODS} == MODS, "the Build modules != the ones packed in the notebook (rebuild)"   # E29 kit\n',
             "modules check")
    return s


def packed_modules(nb_txt):
    c1 = ["".join(c["source"]) for c in json.loads(nb_txt)["cells"] if c["cell_type"] == "code"][1]
    line = next(l for l in c1.splitlines() if l.startswith("E28_SRC = json.loads("))
    expr = ast.get_source_segment(line, ast.parse(line).body[0].value.args[0])
    return json.loads(eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<packed>", "eval")))


def load_chain(variant, slug):
    facts = json.loads(lf(facts_lf(variant.lower() + "_build_facts.json")))
    mods_nb = packed_modules(lf(facts["notebook"]))

    def mod_text(n):
        p = (E28B / n) if n.startswith("e28") else (HERE / n)
        if n == "e29_lib.py" and facts["e29"]["cfg"]["drafter"] == "mtp":   # rev 2: DEFAULT / MTP_V64 keep the audited 1.0
            p = HERE / "e29_lib_v1_0.py"
        if n == "e29_nbdf.py":                  # built by build_e29_final.py (nbdf_lib pieces): the notebook's literal, sha = facts
            return mods_nb[n]
        t = lf(p)
        assert t == mods_nb[n], (n, "Build file != the notebook's packed module")
        return t
    ns = {"__file__": str(E28_KIT), "__name__": "e29kit_e28", "E29_MOD_TEXT": mod_text}
    exec(compile(e28_kit_src(variant, slug), str(E28_KIT), "exec"), ns)
    assert ns["FACTS"]["e29"]["variant"] == variant and ns["NB_SHA"] == facts["notebook_sha256"]
    return ns, facts, mods_nb, mod_text


# ------------------------------------------------------------------ the retargeted E28 unit test file
UT_SRC = lf(E28B / "e28k100_unit_tests.py")
ENG_UNPACK_SRC = r'''


def _e29_unpack_engine(c):          # E29 kit rev 2: = build_e29_final.unpack_engine (the packed E29 engine block back in place)
    head = "\n\n\n# ---- E29 FINAL rev 2 (size: the 400-kB commit limit): the E29 engine block"
    while head in c:                # rev 2d: every packed block
        h = c.index(head)
        x = c.index("exec(compile(", h)
        z = c.index("\n", x) + 1
        call = next(n for n in ast.walk(ast.parse(c[x:z])) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
        expr = ast.get_source_segment(c[x:z], call.args[0].args[0])
        c = c[:h] + eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<e29 packed>", "eval")) + c[z:]
    return c


'''


def unit_test_text():
    t = UT_SRC
    t = once(t, '    g = {"os": os, "json": json, "E11_HARNESS": False, "__builtins__": __builtins__}\n',
             '    g = {"os": os, "json": json, "E11_HARNESS": False, "__builtins__": __builtins__}\n'
             '    _sm = next((l for l in C1.splitlines() if l.startswith("E29_SMOKE = ")), None)   # E29 kit rev 2e: the smoke row reads it\n'
             '    if _sm:\n        g["E29_SMOKE"] = ast.literal_eval(_sm[len("E29_SMOKE = "):].split("   #")[0])\n', "smoke ns")
    t = once(t, """      lambda: G1["K100_PROFILES"] == ("E28EVAL120", "RERUN_PROD") and 'NB5_DEFAULT_PROFILE = "E28EVAL120"' in C1""",
             """      lambda: G1["K100_PROFILES"] in (("E28EVAL120", "RERUN_PROD"), ("E28EVAL120", "RERUN_PROD", "E29SMOKE"))   # E29 kit rev 2e
      and 'NB5_DEFAULT_PROFILE = "E28EVAL120"' in C1 and ('E29_PROFILE = "E28EVAL120"' in C1 or "E29_PROFILE" not in C1)""", "B3 smoke")
    t = once(t, "C0, C1, C3, C4, ENG, C67, C8, C9 = CODE\n",
             "C0, C1, C3, C4, ENG, C67, C8, C9 = CODE\n" + ENG_UNPACK_SRC +
             "ENG = _e29_unpack_engine(ENG)   # E29 kit rev 2: the DFlash notebook packs its E29 engine block\n", "ENG unpack")
    t = once(t, '"""E28 K100 unit tests (quick, local + the Kaggle CPU units kit)',
             '"""E29 kit: E28 K100 unit tests RETARGETED to the E29 FINAL notebook (build_e29_cpu_kit.py, exact-once E29 edits marked\n'
             '"# E29 kit"); the original: E28 K100 unit tests (quick, local + the Kaggle CPU units kit)', "doc")
    t = once(t, '''def cell7_text():
    tree = ast.parse(C67)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
    return packed_text(ast.get_source_segment(C67, call.args[0].args[0]))
''', '''def _packed_of(label):          # E29 kit: CELL 6/7 has two packed literals; pick by its compile() label
    calls = [n for n in ast.walk(ast.parse(C67)) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec"
             and isinstance(n.args[0], ast.Call) and label in str(getattr(n.args[0].args[1], "value", ""))]
    assert len(calls) == 1, (label, len(calls))
    return packed_text(ast.get_source_segment(C67, calls[0].args[0].args[0]))


def cell7_text():
    return _packed_of("e21_cell7")


def cell6_text():               # E29 kit
    return _packed_of("e21_cell6")
''', "cell7_text")
    t = once(t, 'READ = C67[:C67.index("# ---- E28 K100 (size: the 400-kB commit limit)")] + cell7_text()\n',
             '_H6 = C67[:C67.index("# ---- E28 K100 (size: the 400-kB commit limit)")]   # E29 kit: e21_cell6 unpacked too\n'
             'READ = _H6[:_H6.index("# ---- E29 FINAL (size: the 400-kB commit limit)")] + cell6_text() + cell7_text()\n', "READ")
    t = once(t, '      and set(FACTS["modules"]) == {"e28k100_lib.py", "e28_sched_patch.py"})',
             '      and set(FACTS["modules"]) == {"e28k100_lib.py", "e28_sched_patch.py", "e29_lib.py", "e29_sched_patch.py", "e29_nbdf.py"}'
             '   # E29 kit\n      and sha(cell6_text()) == FACTS["e21_cell6_packed"]["sha256"] and cell6_text().startswith("def e21_cell6():\\n"))',
             "A4 modules")
    t = once(t, '                    has = has or "E28" in ln\n', '                    has = has or "E28" in ln or "E29" in ln   # E29 kit\n', "A5 marker")
    t = once(t, '    check("A5 every changed hunk vs K100Q (4506a71f, readable cells) adds a line marked E28 (no unmarked change)", _a5)\n',
             '    check("A5 every changed hunk vs K100Q (4506a71f, readable cells) adds a line marked E28 or E29 (no unmarked change)", _a5)   # E29 kit\n',
             "A5 label")
    t = once(t, '    skip("A5 diff vs K100Q", f"{K100Q_NB} missing")\n',
             '''    skip("A5 diff vs K100Q", f"{K100Q_NB} missing")
E28NB =Path(os.environ.get("E29KT_E28_NB") or r"D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Q27-XHIGH-E28_K100_EVAL120_6H.ipynb")
if E28NB.is_file():               # E29 kit: the E29 edits vs the E28 K100 notebook (the base this build asserts), every hunk marked E29
    def _a6():
        raw = E28NB.read_bytes()
        assert sha(raw) == FACTS["e29"]["e28_notebook_sha256"], "not the E28 K100 notebook E29 was built on"
        c = ["".join(x["source"]) for x in json.loads(raw.decode("utf-8"))["cells"] if x["cell_type"] == "code"]
        tr = ast.parse(c[5])
        call = next(n for n in ast.walk(tr) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
        base = c[:5] + [c[5][:c[5].index("# ---- E28 K100 (size: the 400-kB commit limit)")]
                        + packed_text(ast.get_source_segment(c[5], call.args[0].args[0]))] + c[6:]
        bad, n_h, lit = [], 0, []
        for i, (a, b) in enumerate(zip(base, READABLE)):
            hunk, has = [], False
            for ln in list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0))[2:] + ["@@"]:
                if ln.startswith("@@"):
                    if hunk:
                        n_h += 1
                        if not has and all(h_[1:].startswith("E28_SRC = json.loads(") for h_ in hunk):
                            lit.append(i)    # the modules literal line (E1: it now carries the E29 modules; A4 checks its shas)
                        elif not has:
                            bad.append((i, hunk[:3]))
                    hunk, has = [], False
                elif ln.startswith("+"):
                    hunk.append(ln[:120])
                    has = has or "E29" in ln.upper()   # E29 code (e29_* names count: the E5 e18-cap line)
                elif ln.startswith("-"):
                    hunk.append(ln[:120])
        assert not bad, bad[:6]
        print("     A6 hunks vs E28:", n_h, "| unmarked = the E28_SRC modules literal line only, cells", lit, flush=True)
        return len(base) == len(READABLE) == 8 and n_h > 0 and lit in ([1], [])   # E29 kit rev 2b: [] = every hunk marked
    check("A6 every changed hunk vs the E28 K100 notebook (the facts' e28_notebook_sha256, readable cells) adds a line marked E29 (the one unmarked hunk allowed: the E28_SRC modules literal line in CELL 1, shas checked by A4)", _a6)
else:
    skip("A6 diff vs E28 K100", f"{E28NB} missing")
''', "A6")
    t = once(t, """      lambda: 'NB5_BOOT_LEVERS = dict(levers, **(E28_I8_ENV if E28_I8["on"] else {}))' in ENG and '("_i8" if E28_I8["on"] else "")' in ENG""",
             """      lambda: 'NB5_BOOT_LEVERS = dict(levers, **(E28_I8_ENV if E28_I8["on"] and not _e29d else {}))' in ENG   # E29 kit
      and '("_i8" if E28_I8["on"] and not _e29d else "")' in ENG""", "C3")
    t = once(t, """      lambda: C3.index('assert not _bad, ("E12 package hash mismatch", _bad)') < C3.index("E28S.apply_e28_sched(E12_DIR)")""",
             """      lambda: C3.index('assert not _bad, ("E12 package hash mismatch", _bad)') < C3.index("_ap(E12_DIR)")   # E29 kit: E29 then E28""", "C4")
    t = once(t, """      lambda: ENG.index('trec["i8_readback"] = ') < ENG.index("        if e28_i8_gate(trec):") < ENG.index('trec["k100_gates"] = k100_boot_gates(s, mtp_on)')""",
             """      lambda: ENG.index('trec["i8_readback"] = ') < ENG.index("        if e28_i8_gate(trec):") < ENG.index('trec["k100_gates"] = k100_boot_gates(s, mtp_on and not E29_RT.get("df_try"))')   # E29 kit""",
             "G2")
    t = once(t, '''        g["nb5_boot_try"] = nb5_boot_try
        exec(compile(_eng_def("nb5_boot"), "<nb5_boot>", "exec"), g)
''', '''        g["nb5_boot_try"] = nb5_boot_try
        exec(compile(_eng_def("nb5_boot"), "<nb5_boot>", "exec"), g)
        import time as _t29, e29_lib as _E29L     # E29 kit: the E29 defs nb5_boot calls, no E29 lever on (= the E28 ladder)
        for _n29 in ("e29_env", "e29_extra", "e29_ntries", "e29_drop"):
            exec(compile(_eng_def(_n29), "<" + _n29 + ">", "exec"), g)
        g.update(E29L=_E29L, E29_CFG=_E29L.check_cfg(FACTS["e29"]["cfg"]), DS=None, SSEQ=None, time=_t29, NOTEBOOK_T0=_t29.perf_counter(),
                 E29_RT={"on": [], "dropped": [], "tries": [], "df": None, "vocab": None, "errors": [], "df_try": False, "df_live": False})
''', "ladder e29 defs")
    assert "\r" not in t
    compile(t, "<e29 retargeted e28 unit tests>", "exec")
    return t


UT29 = unit_test_text()
UT29_CHECKS = len(re.findall(r'^\s*check\("', UT29, re.M)) - 1           # every check but E4 (it runs in the real kit)
LIBT = lf(HERE / "e29_kit_lib_tests.py")
LIBT_CHECKS = len(re.findall(r'^check\("', LIBT, re.M))
LIBT_CHECKS_REV2 = len(re.findall(r'^\s*check\("', LIBT, re.M))      # rev 2: + the R checks (the DFlash notebook only)
DS_DIR = ROOT / "ds_q27-e29-final-payload"                            # the payload dataset folder (local runs: E29LT_PAYLOAD_DIR)


def ds_sums():
    """rev 2b: SHA256SUMS over every data file of the payload dataset folder (the builder's files + the kits' fixture)."""
    fs = sorted(f for f in DS_DIR.iterdir() if f.is_file() and f.name not in ("SHA256SUMS", "dataset-metadata.json"))
    (DS_DIR / "SHA256SUMS").write_bytes("".join(f"{sha(f.read_bytes())}  {f.name}\n" for f in fs).encode("utf-8"))

# ------------------------------------------------------------------ units body (E28's, + the E29 lib tests)
UNITS_E29_RUN = r'''
L29 = {}
try:
    (B / "e29_kit_facts.json").write_text(FILES["e29_kit_facts.json"], encoding="utf-8")
    r = sh("e29_kit_lib_tests", [sys.executable, str(B / "e29_kit_lib_tests.py")],
           {"E29LT_NB": str(B / "e26.ipynb"), "E29LT_FACTS": str(B / "e28k100_build_facts.json"), "E29LT_KF": str(B / "e29_kit_facts.json"),
            "E29LT_NB5LIB": str(B / "nb5_lib.py"), "PYTHONIOENCODING": "utf-8"}, 900, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    L29.update(rc=r["rc"], sec=r["sec"], summary=m.group(0) if m else None, fails=[l[:900] for l in lines if l.startswith("FAIL ")][:40],
               tail=r["tail"][-1500:])
    L29["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not L29["fails"] and int(m.group(2)) >= KF["e29_lib_checks_min"]
except BaseException as e:
    L29.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["e29_lib_tests"] = L29
REP["checks"]["units"] = U; save()
log("e29 kit lib tests", json.dumps({k: v for k, v in L29.items() if k != "tb"})[:3000])
'''


UNITS_FIXTURE_STAGE = r'''
for _sn, _ss in (KF.get("staged") or {}).items():          # E29 kit rev 2b / 2d: test inputs from the mounted E29 payload dataset
    if _sn in FILES:
        continue
    _hit = [f for f in sorted(Path("/kaggle/input").glob("**/" + _sn)) if sha_file(f) == _ss]
    U.setdefault("staged_from_dataset", {})[_sn] = str(_hit[0]) if _hit else None
    if _hit:
        (B / _sn).write_bytes(_hit[0].read_bytes())
'''


def units_body(e28_units):
    e28_units = once(e28_units, '''    (B / n).write_text(v, encoding="utf-8", newline="\\n")
# the payload's REAL arc2_duck.py''', '''    (B / n).write_text(v, encoding="utf-8", newline="\\n")
''' + UNITS_FIXTURE_STAGE + '''# the payload's REAL arc2_duck.py''', "fixture stage")
    u = once(e28_units, 'r = sh("e28k100_unit_tests", [sys.executable, str(B / "e28k100_unit_tests.py")],',
             'r = sh("e29_e28k100_unit_tests", [sys.executable, str(B / "e29_e28k100_unit_tests.py")],   # E29 kit: retargeted', "ut name")
    u = once(u, '"E28KT_K100Q_OUT": str(B / "no_k100q_out_here"),', '"E28KT_K100Q_OUT": str(B / "no_k100q_out_here"), "E29KT_E28_NB": str(B / "e28k100.ipynb"),',
             "e28 nb env")
    u = once(u, 'log("e28 unit tests", json.dumps({k: v for k, v in T.items() if k != "tb"})[:3000])\n',
             'log("e28 unit tests", json.dumps({k: v for k, v in T.items() if k != "tb"})[:3000])\n' + UNITS_E29_RUN, "e29 run")
    u = once(u, '"e28_unit_tests_all_pass": T.get("all_pass") is True, "e28_unit_tests_only_E4_skipped": T.get("skips_other") == []},',
             '"e28_unit_tests_retargeted_e29_all_pass": T.get("all_pass") is True, "e28_unit_tests_only_E4_skipped": T.get("skips_other") == [],\n'
             '         "e29_kit_lib_tests_all_pass": L29.get("all_pass") is True},', "verdict")
    u = once(u, 'recorded={"summary": T.get("summary"),', 'recorded={"e29_lib": {k: L29.get(k) for k in ("summary", "fails")}, "summary": T.get("summary"),',
             "recorded")
    return u


# ------------------------------------------------------------------ sched kit body
SCHED_BODY = r'''
U = {}
PAY = find_payload()
U["payload"] = {k: PAY.get(k) for k in ("used",)}
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    (B / n).parent.mkdir(parents=True, exist_ok=True)
    (B / n).write_text(v, encoding="utf-8", newline="\n")
U["payload_sha_ok"] = bool(PAY.get("used")) and sha_file(PAY["used"]) == KF["duck_payload_sha"]
U["eval"] = str(COMPD / "arc-agi_evaluation_challenges.json")
U["eval_present"] = (COMPD / "arc-agi_evaluation_challenges.json").is_file()
U["fixtures"] = {d: len(list((B / d).glob("*/duck/*/branch_result.json"))) for d in ("k100", "w4")}
for k_ in [k_ for k_ in os.environ if k_.startswith(("E28_", "E29_", "E12_", "ARC3_"))]:
    os.environ.pop(k_, None)
T = {}
try:
    # never `python -I`: the test file imports e29_sched_patch from its own dir (sys.path[0]) and e28_sched_patch from E29_TEST_E28_DIR
    r = sh("e29_sched_tests", [sys.executable, str(B / "e29_sched_tests.py"), "-v"],
           {"E29_TEST_E28_DIR": str(B / "e28dir"), "E28_TEST_PAYLOAD": str(PAY.get("used")), "E28_TEST_K100": str(B / "k100"),
            "E28_TEST_W4": str(B / "w4"), "E28_TEST_EVAL": U["eval"], "E28_TEST_EXPORTS": str(B / "exports"), "E28_TEST_TMP": str(K),
            "PYTHONIOENCODING": "utf-8"}, 3000, cwd=B)
    m = re.search(r"E29 SCHED tests: (\d+) run, (\d+) failures, (\d+) errors, (\d+) skipped", r["_out"])
    T.update(rc=r["rc"], sec=r["sec"], summary=m.group(0) if m else None, tail=r["tail"][-3000:],
             fails=[l[:400] for l in r["_out"].splitlines() if l.endswith(("FAIL", "ERROR")) or l.startswith(("FAIL:", "ERROR:"))][:40])
    T["all_pass"] = bool(m) and r["rc"] == 0 and m.group(2) == m.group(3) == "0" and int(m.group(1)) >= KF["sched_tests_min"]
    T["no_skips"] = bool(m) and m.group(4) == "0"
except BaseException as e:
    T.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["sched_tests"] = T
REP["checks"]["sched"] = U; save()
log("e29 sched tests", json.dumps({k: v for k, v in T.items() if k != "tb"})[:4000])
verdict({"duck_payload_mounted_sha_9e4b8928": U["payload_sha_ok"] is True, "competition_eval_challenges_present": U["eval_present"],
         "k100q_w4_branch_records_bundled": U["fixtures"]["k100"] == KF["n_fix"]["k100"] and U["fixtures"]["w4"] == KF["n_fix"]["w4"],
         "e29_sched_tests_all_pass": T.get("all_pass") is True, "e29_sched_tests_no_skips": T.get("no_skips") is True},
        recorded={"summary": T.get("summary"), "fails": T.get("fails"), "payload": U["payload"]})
'''

# ------------------------------------------------------------------ real kit: the E29 section
PLATFORM = E28_LIKE = None
_nbdf_kit = lf(E27B / "build_e27_nbdf_cpu_kit.py")
PLATFORM = _nbdf_kit[_nbdf_kit.index('PLATFORM = r"""') + len('PLATFORM = r"""'):]
PLATFORM = PLATFORM[:PLATFORM.index('\n"""\n') + 1]
SPECPROBE = _nbdf_kit[_nbdf_kit.index('SPECPROBE = r"""') + len('SPECPROBE = r"""'):]
SPECPROBE = SPECPROBE[:SPECPROBE.index('\n"""\n') + 1]
assert "class FakeL4(VPC.NonNvmlCudaPlatform):" in PLATFORM and "SpeculativeConfig" in SPECPROBE and 'out["platform"] = PLAT' in SPECPROBE
VOCABPROBE = r'''
import json, os, sys
out = {"platform": PLAT}
try:
    import vllm.models.qwen4_exp.nvidia.draft_head as DH
    out["env_names"] = {"file": getattr(DH, "ENV_DRAFT_VOCAB_FILE", None), "sha": getattr(DH, "ENV_DRAFT_VOCAB_SHA256", None)}
    os.environ[out["env_names"]["sha"] or "VLLM_DRAFT_VOCAB_SHA256"] = sys.argv[2]
    ids = DH.load_draft_vocab_ids(sys.argv[1], 248320, tp_size=2)
    out.update(ok=True, n=int(ids.numel()), dtype=str(ids.dtype), first=int(ids[0]), last=int(ids[-1]),
               sorted_unique=bool((ids[1:] > ids[:-1]).all()), contiguous=bool(ids.is_contiguous()))
    os.environ[out["env_names"]["sha"] or "VLLM_DRAFT_VOCAB_SHA256"] = "0" * 64
    try:
        DH.load_draft_vocab_ids(sys.argv[1], 248320, tp_size=2)
        out["wrong_sha_refused"] = False
    except ValueError as e:
        out["wrong_sha_refused"] = "sha256 mismatch" in str(e)
except BaseException as e:
    import traceback
    out.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1200:])
print("@JSON " + json.dumps(out, default=str))
'''
E29_ARGPARSE_TAIL = r'''
        _ac = getattr(ea, "attention_config", None)
        r["indexer_kv_dtype"] = (_ac.get("indexer_kv_dtype") if isinstance(_ac, dict) else getattr(_ac, "indexer_kv_dtype", None))
        r["attention_config"] = str(_ac)[:400]
        r["ok"] = True
'''

E29_REAL = r'''
# ---------------------------------------------------------------- E29 kit: e29_setup() + every boot try of the notebook's OWN nb5_boot
E29K = {"variant": KF["e29_variant"], "cfg": KF["e29_cfg"]}
E29V = {}
try:
    E29K["modules_loaded"] = {n: n in (G.get("E28_SRC") or {}) for n in ("e29_lib.py", "e29_sched_patch.py", "e29_nbdf.py")}
    E29K["cfg_nb"] = G.get("E29_CFG")
    E29K["sched_error_nb"] = G.get("E29S_ERROR")
    # kit rev 2c (kit-must-mirror): CELL 8 sets these right before e29_setup() (NB5_PIN #60490 = this kit's real apply; I8 = the
    # profile's i8) -> the notebook's E28 ladder / I8 restore see what the run sees
    if G["NB5_PIN"].get("patch_60490") is None:
        G["NB5_PIN"]["patch_60490"] = (R1.get("patch_60490") if isinstance(globals().get("R1"), dict) else None) or None
    G["E28_I8"].update(want=bool(G["P"].get("i8")), on=bool(G["P"].get("i8")))
    E29K["runner_mirror"] = {"patch_60490_ok": (G["NB5_PIN"].get("patch_60490") or {}).get("ok"), "i8_want": G["E28_I8"]["want"]}
    G["e29_setup"]()
    _rt = G["E29_RT"]
    E29K["setup"] = json.loads(json.dumps({k: _rt.get(k) for k in ("on", "dropped", "df", "vocab", "idx", "supply", "seam", "drafter", "errors")}, default=str))
    E29K["i8_after_setup"] = {k: G["E28_I8"].get(k) for k in ("want", "on", "suspended", "dropped")}
except BaseException as e:
    E29K.update(setup_error=f"{type(e).__name__}: {e}"[:800], setup_tb=traceback.format_exc()[-2000:])
REP["checks"]["e29"] = E29K; save()
log("E29 setup", json.dumps({k: E29K.get(k) for k in ("setup", "i8_after_setup", "setup_error")}, default=str)[:4000])
_su = E29K.get("setup") or {}
def _in_ds(p):                                 # the E29 payload dataset mount (datasets/<owner>/<slug> or the legacy /<slug>)
    return bool(p) and "q27-e29-final-payload" in Path(p).resolve().parts
E29V["e29_modules_loaded_cfg_eq_facts_no_sched_error"] = (all((E29K.get("modules_loaded") or {}).values()) and E29K.get("cfg_nb") == KF["e29_cfg"]
                                                          and E29K.get("sched_error_nb") is None)
E29V["e29_setup_levers_on_as_cfg_nothing_dropped"] = _su.get("on") == KF["e29_on_want"] and not _su.get("dropped") and not _su.get("errors")
if KF["e29_df"]:
    try:
        _df, _dr = _su.get("df") or {}, _su.get("drafter") or {}
        DSM = G["DS"]
        _tree = Path(G["MERLIN_ENG"]["kv"]["MERLIN_TREE"])
        _seam_used = (G.get("Q27X_PAYLOAD_USED") or {}).get("dflash_seam.py") or {}
        E29K["seam_file"] = {"used": _seam_used, "module": getattr(DSM, "__file__", None),
                             "module_sha": sha_file(DSM.__file__) if DSM is not None else None}
        E29K["seam_tree_state"] = DSM.tree_state(_tree, ring=G["SSEQ"].ring)
        E29K["ppack_state_after_seam"] = G["PPK"].verify_ppack_patch(_tree)["state"]
        E29V.update({
            "e29_drafter_facts_real_mount_pins_embed_ok": _dr.get("ok") is True and _dr.get("pins_ok") is True
                and (_dr.get("check") or {}).get("embed_in_checkpoint") is True and "bf16-embed" in str(_dr.get("dir")),
            "e29_seam_from_new_dataset_sha_pinned": _in_ds(_seam_used.get("path"))
                and _seam_used.get("sha256") == KF["seam_sha"] and E29K["seam_file"]["module_sha"] == KF["seam_sha"],
            "e29_seam_grd_e2_guards_applied_post_sha": G["E29L"].grd_check(DSM.verify_dflash_patch(_tree, ring=G["SSEQ"].ring))
                and (_su.get("seam") or {}).get("grd") is True,
            "e29_seamseq_applied_verified_fully_patched": G["SSEQ"].state == "applied" and E29K["seam_tree_state"] == "FULLY_PATCHED"
                and str((_su.get("seam") or {}).get("apply", "")).startswith("applied"),
            "e29_ppack_still_applied_after_seam": E29K["ppack_state_after_seam"] == "applied" and (_su.get("seam") or {}).get("ppack_after") == "applied",
            "e29_df_spec_dflash_k3_auto_flash_attn_drafter_dir": _df.get("ok") is True and (_df.get("spec") or {}).get("method") == "dflash"
                and (_df.get("spec") or {}).get("num_speculative_tokens") == 3 and (_df.get("spec") or {}).get("model") == _dr.get("dir")
                and (_df.get("spec") or {}).get("kv_cache_dtype") == "auto" and (_df.get("spec") or {}).get("attention_backend") == "FLASH_ATTN",
            "e29_df_captures_4_to_96_mask_row_sw_block_1024": _df.get("captures") == KF["df_captures"] and _df.get("sw_block") == 1024
                and _df.get("mask_row_sha") == KF["mask_row_sha"],
            "e29_i8_suspended_on_dflash": E29K["i8_after_setup"].get("on") is False and E29K["i8_after_setup"].get("suspended") == "N/A_DFLASH"})
    except BaseException as e:
        E29K.update(df_check_error=f"{type(e).__name__}: {e}"[:800], df_check_tb=traceback.format_exc()[-1500:])
        E29V["e29_df_checks_ran"] = False
if KF["e29_vocab"]:
    _vo = _su.get("vocab") or {}
    E29V["e29_vocab_64k_file_found_in_new_dataset_sha"] = (_in_ds(_vo.get("file"))
                                                           and Path(_vo["file"]).name == KF["vocab_file"][0] and _vo.get("sha256") == KF["vocab_file"][1]
                                                           and sha_file(_vo["file"]) == KF["vocab_file"][1])
if KF.get("e29_direct"):                    # E29 kit rev 2: DIRECT (the CELL 1 switch on): CELL 3's diff + e29_setup's hook
    _dr = G.get("E29_DIRECT_RT") or {}
    _sc, _sr = G.get("E29D_SCHED_REC") or {}, GR.get("E29D_SCHED_REC") or {}
    E29K["direct"] = {"commit_sched": {k: _sc.get(k) for k in ("want", "ok", "path", "error", "after", "sha_after")},
                      "rerun_sched": {k: _sr.get(k) for k in ("want", "ok", "path", "error", "after", "sha_after")},
                      "state": _dr.get("state"), "lib": _dr.get("lib"), "error": _dr.get("error"),
                      "hook": getattr(G.get("E12A"), "E29D_RUN", None) is not None, "env": os.environ.get("E29_DIRECT"),
                      "switch": G.get("E29_DIRECT2"), "files": G.get("E29D_FILES"),
                      "used": {n: (G.get("Q27X_PAYLOAD_USED") or {}).get(n) for n in ("e29_direct.py", "e29d_sched_patch.py")}}
    E29V["e29_direct_sched_applied_commit_rerun"] = (_sc.get("ok") is True and _sr.get("ok") is True and _sc.get("sha_after") == KF["direct_post"]
                                                     and all(_in_ds((u or {}).get("path")) for u in E29K["direct"]["used"].values()))
    E29V["e29_direct_setup_on_hook_installed"] = (_dr.get("state") == "ON" and E29K["direct"]["hook"] is True
                                                  and (G.get("E29_DIRECT2") or {}).get("on") is True and str(_dr.get("lib")).startswith("e29_direct 1.0"))
REP["checks"]["e29"] = E29K; save()

# the notebook's own nb5_boot (commit namespace) with a scripted nb5_boot_try: the REAL merlin_fn_server per try (= nb5_boot_try's
# launch-prep part incl. its E29 except branch, asserted verbatim at kit build); then a scripted outcome: an OOM while an E29 lever
# is on (the notebook's drop rule walks every lever path down to the plain E28 config), DONE without one. No server is started.
KITTRY = []
os.environ["VLLM_TARGET_DEVICE"] = "cpu"       # kit only (as the E28 plans): the HELP subprocess inherits it via the server env
_try0 = G["nb5_boot_try"]
def _kit_try(trec, name, kv_bytes, phases, mtp_on):
    rec = {"name": name, "kv_bytes": kv_bytes, "boot_kv": G["NB5_BOOT_KV"], "extra": list(G["NB5_BOOT_EXTRA"]), "levers": dict(G["NB5_BOOT_LEVERS"]),
           "df_try": bool(G["E29_RT"].get("df_try")), "e29_on": list(trec.get("e29_on") or []), "e29_df": trec.get("e29_df"),
           "e28_i8": trec.get("e28_i8"), "seqs": trec.get("max_num_seqs_launch")}
    KITTRY.append(rec)
    try:
        srv = G["merlin_fn_server"](name, G["NB5_BOOT_KV"], int(G.get("NB5_PORT") or 8000), kv_bytes)
    except Exception as _e29e:
        rec.update(prep_raised=f"{type(_e29e).__name__}: {_e29e}"[:800], help=(G.get("HELP") or {}).get(name),
                   cli=((G["K1"].get("cli_flags") or {}).get(name)), k6=((G["K1"].get("k6_before_launch") or {}).get(name)),
                   host_phase=(G.get("HOST_PHASE") or {}).get(name))
        if not trec.get("e29_on"):
            rec["raised_without_e29_lever"] = True      # the notebook would raise out of nb5_boot here (the E28 rule)
            trec.update(status="KIT_E28_PREP_RAISED", log_text_for_ladder="")
            ABORT_KIT.append(name)
            return trec
        trec.update(status="E29_LAUNCH_PREP_FAILED", e29_error=f"{type(_e29e).__name__}: {_e29e}"[:800],
                    log_text_for_ladder=f"{type(_e29e).__name__}: {_e29e}")
        return trec
    rec.update(argv=list(srv.argv), server_started=srv.proc is not None, help=(G.get("HELP") or {}).get(name),
               cli=((G["K1"].get("cli_flags") or {}).get(name)), k6=((G["K1"].get("k6_before_launch") or {}).get(name)),
               host_phase=(G.get("HOST_PHASE") or {}).get(name),
               env={k: srv.env.get(k) for k in sorted(srv.env) if k.startswith(("DFLASH_", "VLLM_DRAFT_VOCAB", "VLLM_GDN", "NB4_", "VLLM_QSA_KV_SCALES",
                                                                                 "VLLM_SPARSE_INDEXER", "VLLM_E2_QSA"))
                    or k in ("DRAFT_HEAD", "MERLIN_SHARE_DRAFT_WORKSPACE", "VLLM_SERVER_DEV_MODE", "VLLM_TARGET_DEVICE")})
    if trec.get("e29_on"):
        trec.update(status="SERVER_DIED", log_text_for_ladder="torch.OutOfMemoryError: CUDA out of memory (E29 kit script)")
    else:
        trec.update(status="DONE", phases_started=True)
    return trec
ABORT_KIT = []
BR = {}
try:
    G["nb5_boot_try"] = _kit_try
    BR = G["nb5_boot"]("kit_e29", [list(G["P"]["nb5_phases"][0])], True, "fp8")
except BaseException as e:
    E29K.update(ladder_error=f"{type(e).__name__}: {e}"[:800], ladder_tb=traceback.format_exc()[-2000:])
finally:
    G["nb5_boot_try"] = _try0
E29K["ladder"] = {"status": BR.get("status"), "tries": [{k: t_.get(k) for k in ("name", "status", "kv_bytes", "max_num_seqs_launch", "e28_i8", "e29_on", "e29_df")}
                                                        for t_ in BR.get("tries") or []],
                  "dropped": json.loads(json.dumps(G["E29_RT"].get("dropped"), default=str)), "i8": {k: G["E28_I8"].get(k) for k in ("on", "dropped")}}
E29K["kit_tries"] = KITTRY
REP["checks"]["e29"] = E29K; save()
for t_ in KITTRY:
    log("E29 try", t_["name"], json.dumps({k: t_.get(k) for k in ("kv_bytes", "seqs", "df_try", "e29_on", "e28_i8", "prep_raised", "cli", "host_phase")}, default=str)[:1500])

def _last(a, f):
    a = list(a or [])
    return a[len(a) - 1 - a[::-1].index(f) + 1] if f in a else None
_DT = [t_ for t_ in KITTRY if t_["df_try"]]
_MT = [t_ for t_ in KITTRY if not t_["df_try"]]
_BUILT = [t_ for t_ in KITTRY if t_.get("argv")]
_IDX = " ".join(KF["idx_args"])
def _ladder_ok():
    bad = []
    if BR.get("status") != "DONE": bad.append(f"boot status {BR.get('status')}")
    if ABORT_KIT: bad.append(f"E28 prep raised {ABORT_KIT}")
    if KF["e29_df"]:
        if [t_["kv_bytes"] for t_ in _DT][:1] != [KF["d_pins"][0]] or sorted(set(t_["kv_bytes"] for t_ in _DT)) != sorted(set(KF["d_pins"])):
            bad.append(f"D pins {[t_['kv_bytes'] for t_ in _DT]}")
        for t_ in _DT:
            lv = t_["levers"]
            if t_["extra"][-4:] != (G["E29_RT"]["df"] or {}).get("tail"): bad.append(f"{t_['name']} tail")
            if _last(t_["extra"], "--max-num-seqs") != "24": bad.append(f"{t_['name']} seqs")
            if lv.get(G["DS"].ENV_SEAM) != "1" or lv.get(G["DS"].ENV_SW_BLOCK) != "1024" or lv.get(G["DS"].ENV_MASK_SHA) != KF["mask_row_sha"]:
                bad.append(f"{t_['name']} seam env {lv}")
            if "DRAFT_HEAD" in lv: bad.append(f"{t_['name']} DRAFT_HEAD on a D try")
            if KF.get("e29_embed_host") and lv.get(G["DS"].ENV_EMBED_HOST) != "1": bad.append(f"{t_['name']} no EMBED_HOST (rev 2: from the first D try)")
            if KF.get("e29_embed_host") and t_.get("argv") and (t_.get("env") or {}).get(G["DS"].ENV_EMBED_HOST) != "1":
                bad.append(f"{t_['name']} EMBED_HOST not on the server env")
            if t_.get("argv"):
                sp = json.loads(_last(t_["argv"], "--speculative-config") or "{}")
                cc = json.loads(_last(t_["argv"], "--compilation-config") or "{}")
                if sp.get("method") != "dflash" or sp.get("num_speculative_tokens") != 3: bad.append(f"{t_['name']} spec {sp}")
                if cc.get("cudagraph_capture_sizes") != KF["df_captures"]: bad.append(f"{t_['name']} captures {cc.get('cudagraph_capture_sizes')}")
                if (t_.get("env") or {}).get(G["DS"].ENV_SEAM) != "1": bad.append(f"{t_['name']} seam env not on the server env")
        _q = KF.get("qsa_env") or {}
        for t_ in _DT:                              # kit rev 2c: the QSA workspace on every D try's FINAL server env (after build())
            if t_.get("argv") and any((t_.get("env") or {}).get(k) != v for k, v in _q.items()): bad.append(f"{t_['name']} QSA env {t_.get('env')}")
        for t_ in _MT:                              # ... and Merlin's own 256 / 256 on the MTP tries (the proven config)
            if t_.get("argv") and any((t_.get("env") or {}).get(k) != "256" for k in _q): bad.append(f"{t_['name']} QSA env on MTP {t_.get('env')}")
        if _MT and str(_MT[0].get("seqs")) != "24": bad.append(f"MTP fallback seqs {_MT[0].get('seqs')} (want 24, the top rung)")
        if not _MT or _MT[0]["kv_bytes"] != KF["pins"][0] or _MT[0]["levers"].get("DRAFT_HEAD") != "int8" or any(k.startswith("DFLASH_") for k in _MT[0]["levers"]):
            bad.append(f"MTP fallback try {(_MT or [{}])[0].get('name')} {(_MT or [{}])[0].get('levers')}")
        for t_ in _MT:
            if t_.get("argv") and json.loads(_last(t_["argv"], "--speculative-config") or "{}").get("method") != "mtp": bad.append(f"{t_['name']} not mtp")
    if KF["e29_vocab"]:
        _vt = [t_ for t_ in KITTRY if "vocab" in t_["e29_on"]]
        if not _vt: bad.append("no vocab try")
        for t_ in _vt:
            if t_["levers"].get("VLLM_DRAFT_VOCAB_FILE") != (_su.get("vocab") or {}).get("file") or t_["levers"].get("VLLM_DRAFT_VOCAB_SHA256") != KF["vocab_file"][1]:
                bad.append(f"{t_['name']} vocab env {t_['levers']}")
            if t_.get("argv") and (t_.get("env") or {}).get("VLLM_DRAFT_VOCAB_FILE") != (_su.get("vocab") or {}).get("file"): bad.append(f"{t_['name']} vocab not on the server env")
    for t_ in KITTRY:
        if ("idx" in t_["e29_on"]) != (_IDX in " ".join(t_["extra"])): bad.append(f"{t_['name']} idx argv vs e29_on")
        if t_.get("argv") and t_.get("server_started"): bad.append(f"{t_['name']} server started")
    last = KITTRY[-1] if KITTRY else {}
    if last.get("e29_on") or last.get("df_try") or last.get("kv_bytes") != KF["pins"][0]: bad.append(f"last try not the plain E28 top rung {last.get('name')}")
    return bad
E29K["ladder_bad"] = _ladder_ok() if KITTRY else ["no try"]
E29V["e29_nb5_boot_every_try_planned_d_ladder_drops_mtp_fallback"] = not E29K["ladder_bad"]
E29V["e29_k6_host_phase_ok_every_built_try"] = bool(_BUILT) and all((t_.get("k6") or {}).get("ok") is True and t_.get("host_phase") == "ok" for t_ in _BUILT)
# the notebook's HELP gate (KL.cli_flags_check on vLLM's own `serve --help=all`, FAIL_CLI_FLAGS): every try with an E29 lever must pass it
E29K["help_gate"] = {t_["name"]: {"cli": t_.get("cli"), "prep_raised": t_.get("prep_raised")} for t_ in KITTRY}
E29V["e29_help_gate_cli_flags_ok_every_try"] = bool(KITTRY) and all((t_.get("cli") or {}).get("ok") is True and not t_.get("prep_raised") for t_ in KITTRY)
if KF["e29_idx"]:
    E29V["e29_indexer_flag_reached_a_built_argv"] = any(_IDX in " ".join(t_.get("argv") or []) for t_ in KITTRY)
REP["checks"]["e29"] = E29K; save()

# vLLM's own argparse (Merlin venv, VLLM_TARGET_DEVICE=cpu) on every built E29 argv
R29 = {}
try:
    argvs = {t_["name"]: t_["argv"] for t_ in _BUILT}
    (K / "e29_argvs.json").write_text(json.dumps(argvs))
    r = venv_py(E29_ARGPARSE, [str(K / "e29_argvs.json")], name="e29_argparse", extra_env=CPU_ENV)
    R29["parsed"] = r["json"] or {"error": r["tail"]}
    fb = {}
    for t_ in _BUILT:
        x = (R29["parsed"] or {}).get(t_["name"]) or {}
        bad = []
        if x.get("ok") is not True: bad.append("parse " + str(x.get("error"))[:200])
        if x.get("unknown"): bad.append(f"unknown {x.get('unknown')}")
        if x.get("from_cli_args_error"): bad.append("from_cli_args " + str(x.get("from_cli_args_error"))[:200])
        if x.get("max_num_seqs") != 24: bad.append(f"max_num_seqs {x.get('max_num_seqs')}")
        if x.get("kv_cache_memory_bytes") != t_["kv_bytes"]: bad.append(f"kv {x.get('kv_cache_memory_bytes')}")
        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")
        if ("idx" in t_["e29_on"]) and x.get("indexer_kv_dtype") != "fp8": bad.append(f"indexer_kv_dtype {x.get('indexer_kv_dtype')}")
        if ("idx" not in t_["e29_on"]) and x.get("indexer_kv_dtype") not in (None, "auto"): bad.append(f"indexer_kv_dtype w/o idx {x.get('indexer_kv_dtype')}")
        sp = x.get("speculative_config")
        if t_["df_try"]:
            if not (isinstance(sp, dict) and sp.get("method") == "dflash" and sp.get("num_speculative_tokens") == 3): bad.append(f"spec {str(sp)[:120]}")
            if (x.get("capture_sizes") or []) != KF["df_captures"]: bad.append(f"captures {x.get('capture_sizes')}")
        elif not (isinstance(sp, dict) and sp.get("method") == "mtp" and sp.get("num_speculative_tokens") == 3): bad.append(f"spec {str(sp)[:120]}")
        fb[t_["name"]] = bad
    R29["field_check"] = fb
    R29["ok"] = bool(fb) and all(not v for v in fb.values())
except BaseException as e:
    R29.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
REP["checks"]["e29_argparse"] = R29; save()
log("E29 argparse", json.dumps(R29.get("field_check"), default=str)[:3000])
E29V["e29_vllm_argparse_every_built_e29_argv"] = R29.get("ok") is True
if KF["e29_df"]:
    try:
        _sp = (G["E29_RT"].get("df") or {}).get("spec") or (_su.get("df") or {}).get("spec")
        r = venv_py(E29_PLATFORM + E29_SPECPROBE, [json.dumps({"D3s": _sp})], name="e29_spec_preflight", extra_env=dict(CPU_ENV, PYTHONUNBUFFERED="1"))
        E29K["spec_preflight"] = r["json"] or {"error": r["tail"]}
        E29V["e29_dflash_spec_preflight_speculativeconfig_fields_literals"] = (E29K["spec_preflight"] or {}).get("ok") is True
    except BaseException as e:
        E29K["spec_preflight"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
        E29V["e29_dflash_spec_preflight_speculativeconfig_fields_literals"] = False
if KF["e29_vocab"]:
    try:
        r = venv_py(E29_PLATFORM + E29_VOCABPROBE, [str((_su.get("vocab") or {}).get("file")), KF["vocab_file"][1]], name="e29_vocab_probe",
                    extra_env=dict(CPU_ENV, PYTHONUNBUFFERED="1"))
        E29K["vocab_probe"] = r["json"] or {"error": r["tail"]}
        _vp = E29K["vocab_probe"] or {}
        E29V["e29_merlin_load_draft_vocab_ids_64k_tp2_sha_env"] = (_vp.get("ok") is True and _vp.get("n") == 65536 and _vp.get("sorted_unique") is True
                                                                  and _vp.get("wrong_sha_refused") is True
                                                                  and _vp.get("env_names") == {"file": "VLLM_DRAFT_VOCAB_FILE", "sha": "VLLM_DRAFT_VOCAB_SHA256"})
    except BaseException as e:
        E29K["vocab_probe"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
        E29V["e29_merlin_load_draft_vocab_ids_64k_tp2_sha_env"] = False
os.environ.pop("VLLM_TARGET_DEVICE", None)
# the seam back to the exact Merlin files (kit cleanup = the notebook's revert path), ppack untouched
if KF["e29_df"]:
    try:
        _fin = G["SSEQ"].finish()
        _tree = Path(G["MERLIN_ENG"]["kv"]["MERLIN_TREE"])
        E29K["seam_revert"] = {"finish": _fin, "tree_state": G["DS"].tree_state(_tree, ring=G["SSEQ"].ring),
                               "ppack_state": G["PPK"].verify_ppack_patch(_tree)["state"]}
        E29V["e29_seam_revert_exact_merlin_files_ppack_kept"] = (_fin.get("state") == "reverted" and E29K["seam_revert"]["tree_state"] == "ORIGINAL"
                                                                and E29K["seam_revert"]["ppack_state"] == "applied")
    except BaseException as e:
        E29K["seam_revert"] = {"error": f"{type(e).__name__}: {e}"[:600]}
        E29V["e29_seam_revert_exact_merlin_files_ppack_kept"] = False
E29K["launch_argv"] = {t_["name"]: t_.get("argv") for t_ in _BUILT}
REP["checks"]["e29"] = E29K; save()
log("E29 verdict items", json.dumps(E29V))

# the step-coverage table (every pre-launch step of the notebook, notebook order -> kit check(s) -> result)
E29_STEPS = json.loads(r"""__STEPS__""")
_verdict0 = verdict
def verdict(ok, recorded=None):
    rows = []
    for st, keys in E29_STEPS:
        ks = [k for k in keys if k in ok]
        res = "NOT_COVERED" if not ks else ("PASS" if all(ok[k] for k in ks) else "FAIL: " + ", ".join(k for k in ks if not ok[k]))
        rows.append({"step": st, "checks": ks, "result": res})
    REP["step_coverage"] = rows
    print("STEP COVERAGE (notebook order):", flush=True)
    for r_ in rows:
        print(f"  {r_['step'][:90]:<90} | {r_['result'][:120]}", flush=True)
    return _verdict0(dict(ok, step_coverage_every_step_covered=all(r_["result"] != "NOT_COVERED" for r_ in rows)), recorded)
'''

# the pre-launch steps of the notebook in its order (CELL 0 .. CELL 8 up to the first launch) -> the real kit's verdict keys
STEPS_COMMON = [
    ("CELL 0: mode + CELL 0 fallback submission (E18 R2 verbatim), commit + rerun namespace",
     ["cell0_same_test_file_and_fallback_as_e18_r2_real_mount", "submission_commit_strict_full_schema_identity",
      "submission_rerun_strict_full_schema_identity", "profile_commit", "profile_rerun_RERUN_PROD_source_test_all_tasks"]),
    ("CELL 1: modules literal (E28_SRC: e28k100_lib, e28_sched_patch, e29_lib, e29_sched_patch, e29_nbdf) + E29_CFG + profiles",
     ["cells_before_6_7_ran_commit_and_rerun", "e29_modules_loaded_cfg_eq_facts_no_sched_error"]),
    ("CELL 1: tf-ref dataset pins (k100q payload, nb4_ppack.py, k100q_report.py) + the K100Q payload blobs",
     ["tf_ref_k100q_version_mounted", "tf_ref_every_staged_file_sha_ok", "payload_loaded_from_dataset_files", "payload_sha_ok_under_input"]),
    ("CELL 3: Duck payload + manifest + sandbox self-test", ["duck_cell_loaded", "sandbox_selftest_ok_incl_memory_limit"]),
    ("CELL 3: scheduler apply E29 S-1 (verify; failure -> revert -> E28) + server env, revert / re-apply on the payload's arc2_duck.py",
     ["e28_sched_applied_commit_rerun_revert_reapply"]),
    ("CELL 5 (runner order) nb5_rc2_inputs: exact Model / dataset mounts", ["inputs_found"]),
    ("nb5_rc2_prompts: klee100 merge + fixkeys overlay == v2 + model dir + K6; prompt hashes (klee100 tokenizer)",
     ["klee100_merge_overlay_eq_v2_model_dir_k6", "prompt_hashes_ok_klee100_tokenizer"]),
    ("merlin_install (offline tar, venv, tree)", ["merlin_install_ok"]),
    ("nb5_apply_60490 (#60490 applied + verified)", ["patch_60490_applied_verified"]),
    ("sha_gate: K7 / K3 shard digests", ["k7_k3_digests_all_ok"]),
    ("k100_setup: worker extension, Q-defer nb4_ppack apply (+ revert / re-apply / injected-failure restore), scales sidecar",
     ["k100_setup_ran_on_real_tree", "worker_ext_installed_compiled_resolves_in_tree", "scales_sidecar_sha_ok_tree_dataset_model_dir",
      "qdefer_lib_loaded_from_dataset_sha_ok", "k100_setup_qdefer_applied_post_shas_env", "qdefer_revert_exact_pre_shas",
      "qdefer_reapplied_after_revert", "qdefer_rerun_namespace_already_applied_env"]),
    ("e28_i8_static (SUPPORTED_STATIC on the merged klee100 dir + installed tree) + Merlin's INT8 quantizer on real lm_head rows",
     ["e28_i8_static_supported_real_klee100_installed_tree", "e28_i8_merlin_int8_quantizer_on_real_lm_head_rows"]),
]
STEPS_TAIL = [
    ("nb5_boot E28 ladder plans (both profiles, every K3 rung, I8 first rung, 24 / table seqs) -> merlin_fn_server: K6, build(), "
     "host_phase(), HELP, argv / env; vLLM argparse",
     ["commit_rungs_equal_facts", "rerun_rungs_e18_first_3", "plans_every_rung_both_profiles_ok", "host_phase_ok_all_plans",
      "help_cpu_every_flag_present_all_plans", "argv_env_pin_seqs_caps_fp8_mtp_lmo_ext_scales_devmode_levers_all_plans",
      "vllm_argparse_fields_all_plans", "no_server_started", "e28_first_rung_both_profiles_built_with_int8_head"]),
    ("nb5_boot with the E29 levers (the notebook's own def): every try (D ladder / lever drops / MTP fallback) with e29_env / "
     "e29_extra -> merlin_fn_server (K6, build(), host_phase())",
     ["e29_nb5_boot_every_try_planned_d_ladder_drops_mtp_fallback", "e29_k6_host_phase_ok_every_built_try"]),
    ("the HELP gate per E29 try (KL.cli_flags_check: FAIL_CLI_FLAGS before any launch)",
     ["e29_help_gate_cli_flags_ok_every_try", "e29_indexer_flag_reached_a_built_argv"]),
    ("vLLM's argparse on every E29 launch argv (the argv recorded in the report)", ["e29_vllm_argparse_every_built_e29_argv"]),
    ("worker-side: the MM method-map probe on real Merlin objects", ["method_map_probe_real_merlin_objects_pass", "method_map_src_anchors_ok"]),
]
STEPS_DF = [("e29_setup DFlash: drafter_facts on the drafter Model mount, the seam from the E29 dataset (q27x_payload), SeamSeq "
             "apply + verify, ppack still applied, the spec / captures / seam env, I8 suspended",
             ["e29_setup_levers_on_as_cfg_nothing_dropped", "e29_drafter_facts_real_mount_pins_embed_ok", "e29_seam_from_new_dataset_sha_pinned",
              "e29_seam_grd_e2_guards_applied_post_sha",
              "e29_seamseq_applied_verified_fully_patched", "e29_ppack_still_applied_after_seam", "e29_df_spec_dflash_k3_auto_flash_attn_drafter_dir",
              "e29_df_captures_4_to_96_mask_row_sw_block_1024", "e29_i8_suspended_on_dflash"])]
STEPS_DIRECT = [("CELL 3: the DIRECT diff (e29d_sched_patch.py from the E29 dataset) applied + verified, commit + rerun; e29_setup: "
                 "e29_direct.py from the E29 dataset + the E12A.E29D_RUN hook (the CELL 1 switch on)",
                 ["e29_direct_sched_applied_commit_rerun", "e29_direct_setup_on_hook_installed"])]
STEPS_DF_TAIL = [("the dflash spec: SpeculativeConfig preflight (fields, Literal values, backend, draft TP)",
                  ["e29_dflash_spec_preflight_speculativeconfig_fields_literals"]),
                 ("kit cleanup: the seam reverted to the exact Merlin files, ppack kept", ["e29_seam_revert_exact_merlin_files_ppack_kept"])]
STEPS_V = [("e29_setup vocab: the 64k draft vocab file found (q27x_payload) + sha; indexer lever on",
            ["e29_setup_levers_on_as_cfg_nothing_dropped", "e29_vocab_64k_file_found_in_new_dataset_sha"])]
STEPS_V_TAIL = [("Merlin's load_draft_vocab_ids(path, 248320, tp_size=2) with VLLM_DRAFT_VOCAB_SHA256 (+ a wrong sha refused)",
                 ["e29_merlin_load_draft_vocab_ids_64k_tp2_sha_env"])]


def steps_for(variant):
    df = variant == "E29_DFLASH_D3S"
    return (STEPS_COMMON + (STEPS_DF + STEPS_DIRECT if df else STEPS_V) + STEPS_TAIL[:4] + (STEPS_DF_TAIL[:1] if df else STEPS_V_TAIL) + STEPS_TAIL[4:]
            + (STEPS_DF_TAIL[1:] if df else []))


DIRECT_REVERT_SRC = r"""    _DS29 = G.get("E29DS") if KF.get("e29_direct") else None   # E29 kit rev 2: the DIRECT post -> the E29 post first
    E28R["direct_state"] = _DS29.tree_state(E12D) if _DS29 is not None else None
    if E28R["direct_state"] == "DIRECT_PATCHED":
        E28R["direct_revert"] = {k: v for k, v in _DS29.revert_direct(E12D).items() if k != "text"}
        E28R["state"] = S_.tree_state(E12D)   # = the E29 post file under the DIRECT diff
"""
DIRECT_REAPPLY_SRC = r"""    if _DS29 is not None:                   # E29 kit rev 2: the DIRECT diff again (= the state the run uses)
        E28R["direct_reapply"] = {k: v for k, v in _DS29.apply_direct(E12D).items() if k != "text"}
"""


def real_body(ns, variant, facts):
    real = ns["REAL"]
    eng = ["".join(c["source"]) for c in json.loads(ns["NB_TXT"])["cells"] if c["cell_type"] == "code"][4]
    # the scripted try mirrors nb5_boot_try's launch-prep block: assert the notebook's text is the one mirrored
    assert ('    try:\n        srv = merlin_fn_server(name, NB5_BOOT_KV, NB5_PORT, kv_bytes)\n    except Exception as _e29e:' in eng
            and 'trec.update(status="E29_LAUNCH_PREP_FAILED", e29_error=f"{type(_e29e).__name__}: {_e29e}"[:800],\n'
                '                    log_text_for_ladder=f"{type(_e29e).__name__}: {_e29e}")\n        return trec\n' in eng)
    # E28 real: the scheduler applied is E29 S-1 (supply) -> its module / post sha / revert / re-apply
    real = once(real, '    S_ = G["E28S"]\n', '    S_ = G["E29S"] if (G.get("E28_SCHED_REC") or {}).get("path") == "E29" else G["E28S"]   # E29 kit\n', "S_")
    real = once(real, "    rv = S_.revert_e28_sched(E12D)\n", DIRECT_REVERT_SRC +
                "    rv = (getattr(S_, 'revert_e29_sched', None) or S_.revert_e28_sched)(E12D)   # E29 kit\n", "revert")
    real = once(real, "    ap = S_.apply_e28_sched(E12D)\n",
                "    ap = (getattr(S_, 'apply_e29_sched', None) or S_.apply_e28_sched)(E12D)   # E29 kit\n" + DIRECT_REAPPLY_SRC, "apply")
    real = once(real, '                        and E28R["sched_commit"]["env"]["E28_B2_REQUEST_TIMEOUT"] == "9000")\n',
                '                        and E28R["sched_commit"]["env"]["E28_B2_REQUEST_TIMEOUT"] == "9000"\n'
                '                        and (G.get("E28_SCHED_REC") or {}).get("path") == KF["sched_path"]   # E29 kit: S-1 (supply) applied\n'
                '                        and E28R["sched_commit"]["env"].get("E29_SUPPLY") == ("1" if KF["sched_path"] == "E29" else None))\n', "sched path")
    probes = ("\nE29_ARGPARSE = json.loads(" + repr(json.dumps(E29_ARGPARSE)) + ")\nE29_PLATFORM = json.loads(" + repr(json.dumps(PLATFORM))
              + ")\nE29_SPECPROBE = json.loads(" + repr(json.dumps(SPECPROBE)) + ")\nE29_VOCABPROBE = json.loads(" + repr(json.dumps(VOCABPROBE)) + ")\n")
    sec = probes + E29_REAL.replace("__STEPS__", json.dumps(steps_for(variant)))
    v = real.index("sc, sr = SUB.get(\"commit\") or {}, SUB.get(\"rerun\") or {}\nverdict({")
    real = real[:v] + sec + real[v:]
    real = once(real, "    **QDV, **E28V,\n}, recorded={", "    **QDV, **E28V, **E29V,\n}, recorded={\"e29\": {k: v for k, v in E29K.items() if not k.endswith(\"tb\") and k != \"launch_argv\"}, ",
                "verdict e29")
    return real


# E29_ARGPARSE = the klee100 kit's ARGPARSE probe + the attention config read
def _mk_argparse(ns):
    a = ns["NS"]["ARGPARSE"]
    a = a[a.index('r"""') + 4:a.rindex('"""')] if a.lstrip().startswith("ARGPARSE") else a
    return once(a, '        r["ok"] = True\n', E29_ARGPARSE_TAIL, "argparse tail")


# ------------------------------------------------------------------ build
def kit_meta(ns, slug, title, models, datasets):
    return {"id": f"{OWNER}/{slug}", "title": title, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": list(datasets), "kernel_sources": [], "competition_sources": list(ns["NS"]["COMPS"]),
            "model_sources": list(models), "docker_image": ns["NS"]["IMAGE"], "machine_shape": "None"}


def build_kit(ns, name, body, files, title, models, datasets, kf):
    NS = ns["NS"]
    assert all("\r" not in v for v in files.values()), name
    blob = base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    shas = {n: sha(v) for n, v in files.items()}
    code = NS["HEAD"] + body
    code = once(code, '"/tmp/e28k100kit"', '"/tmp/e29kit"', "tmp")
    code = once(code, 'f"e28k100_kit_{KIT}_report.json"', 'f"e29_kit_{KIT}_report.json"', "report")
    code = code.replace("E28 K100 CPU kit", "E29 FINAL CPU kit")
    if "e26.ipynb" not in files:   # the sched kit bundles no notebook: the head's notebook-sha assert applies only when it is bundled
        code = once(code, 'assert hashlib.sha256(FILES["e26.ipynb"].encode("utf-8")).hexdigest() ==',
                    'assert "e26.ipynb" not in FILES or hashlib.sha256(FILES["e26.ipynb"].encode("utf-8")).hexdigest() ==', "nb sha optional")
    for k, v in (("__SHAS__", json.dumps(shas)), ("__NBSHA__", ns["NB_SHA"]), ("__NAME__", name), ("__KF__", json.dumps(kf)), ("__BLOB__", blob)):
        code = code.replace(k, v)
    for ph in ("__SHAS__", "__NBSHA__", "__KF__", "__BLOB__", "__MODE__", "__STEPS__"):
        assert ph not in code, (name, ph)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e29-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert b"\r" not in data.replace(b"\\r", b"")
    assert len(data) < 400_000, (name, len(data))
    out = ROOT / f"{slug}__{OWNER}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = (json.dumps(kit_meta(ns, slug, title, models, datasets), indent=2) + "\n").encode("utf-8")
    (out / "kernel-metadata.json").write_bytes(meta)
    assert b"\r" not in (out / f"{slug}.ipynb").read_bytes() and b"\r" not in (out / "kernel-metadata.json").read_bytes()
    print("kit", out.name, "|", len(data), "bytes | sha256", sha(data), flush=True)
    return {"folder": str(out), "sha256": sha(data), "bytes": len(data), "datasets": datasets, "models": models}


def variant_kf(ns, facts, variant):
    kf = dict(ns["KF"])
    cfg = facts["e29"]["cfg"]
    supply = bool(cfg["attempts_supply"])
    sch = facts["e29"]["sched_sha"]
    kf.update(sched_post=sch["post"] if supply else kf["sched_post"], sched_path="E29" if supply else "E28",
              e29_variant=variant, e29_cfg=cfg, e29_df=cfg["drafter"] != "mtp", e29_vocab=bool(cfg["draft_vocab"]), e29_idx=bool(cfg["indexer_fp8"]),
              e29_on_want=(["df"] if cfg["drafter"] != "mtp" else []) + (["idx"] if cfg["indexer_fp8"] else []) + (["vocab"] if cfg["draft_vocab"] else []),
              seam_sha=facts["e29"]["seam"][1], vocab_file=facts["e29"]["vocab_files"].get(cfg["draft_vocab"] or "64k"),
              idx_args=["--attention-config", '{"indexer_kv_dtype":"fp8"}'], d_pins=[NBDF_K100_CFG["nbdf_q_pin"]] + [p for p, _ in NBDF_K100_CFG["nbdf_d_retry"]],
              df_captures=[4 * i for i in range(1, 25)], mask_row_sha=NBDF_FACTS.get("mask_row_sha256") or MASK_ROW_SHA,
              e29_lib_checks_min=(LIBT_CHECKS_REV2 if cfg["drafter"] != "mtp" else LIBT_CHECKS), unit_checks_min=UT29_CHECKS, pins=facts["pins"],
              e29_embed_host=bool(cfg["embed_host"]), e29_direct=bool((facts["e29"].get("direct") or {}).get("files")),
              direct_post=(facts["e29"].get("direct") or {}).get("duck_post_sha256"),
              qsa_env=(facts["e29"].get("qsa_env") or {}))
    return kf


_lib_src = lf(HERE / "e29_lib.py")
_dp = {}           # the drafter pins: the e29_nbdf literal of the DFLASH notebook (nbdf_lib verbatim)
exec(compile(packed_modules(lf(BUILD_FACTS["variants"]["E29_DFLASH_D3S"]["notebook"]))["e29_nbdf.py"], "e29_nbdf.py", "exec"), _dp)
MASK_ROW_SHA = _dp["DRAFTER_PINS"]["mask_row"]["sha256"]


def main(local_units=False, only=None):
    out = {}
    import importlib.util as _u
    _s = _u.spec_from_file_location("e29_lib_kitb", HERE / "e29_lib.py")
    E29L = _u.module_from_spec(_s)
    _s.loader.exec_module(E29L)
    assert E29L.IDX_ARGS == ["--attention-config", '{"indexer_kv_dtype":"fp8"}']   # the JSON form (help gate: the dotted key is not listed)
    caps = None
    for short, variant in VARIANTS.items():
        if only and short not in only:          # rev 2: rebuild only the asked variants' kits (the others stay as built)
            continue
        ns, facts, mods_nb, mod_text = load_chain(variant, f"q27-e29-kit-{short}")
        NS = ns["NS"]
        kf = variant_kf(ns, facts, variant)
        if caps is None:
            nbdf_ns = {}
            exec(compile(mods_nb["e29_nbdf.py"], "e29_nbdf.py", "exec"), nbdf_ns)
            caps = nbdf_ns["capture_sizes"](24, 3)
            assert caps == kf["df_captures"], caps
            assert nbdf_ns["DRAFTER_PINS"]["mask_row"]["sha256"] == MASK_ROW_SHA
        kit_facts = {"variants_cfg": BUILD_FACTS["variants_cfg"], **NBDF_K100_CFG}
        ufiles = dict(ns["UNIT_FILES"])
        ufiles.pop("e28k100_unit_tests.py")
        ufiles.update({"e29_e28k100_unit_tests.py": UT29, "e29_kit_lib_tests.py": LIBT, "e29_kit_facts.json": json.dumps(kit_facts),
                       "e29_lib.py": mod_text("e29_lib.py"), "e29_sched_patch.py": mod_text("e29_sched_patch.py"),
                       "e29_nbdf.py": mod_text("e29_nbdf.py"), "e28k100.ipynb": lf(E28B.parent / "Q27-XHIGH-E28_K100_EVAL120_6H.ipynb")})
        assert sha(ufiles["e28k100.ipynb"]) == facts["e29"]["e28_notebook_sha256"]
        if facts["e29"]["cfg"]["drafter"] != "mtp":      # rev 2b: the K100Q fixture moves to the E29 payload dataset (kit size)
            kf["staged"] = {}
            for _sn in ("k100q_fixture.json", "k100q.ipynb"):   # rev 2d: + the K100Q notebook (A5's diff base; kit size)
                _fx = ufiles.pop(_sn).encode("utf-8")
                (DS_DIR / _sn).write_bytes(_fx)
                kf["staged"][_sn] = sha(_fx)
            ds_sums()
        datasets = list(NS["DATASETS"])
        models_real = list(NS["MODELS"])
        if local_units:
            d = LOCAL / short
            d.mkdir(parents=True, exist_ok=True)
            for n, v in ufiles.items():
                (d / n).parent.mkdir(parents=True, exist_ok=True)
                (d / n).write_bytes(v.encode("utf-8"))
            env = dict(os.environ, E28KT_NB=facts["notebook"], E28KT_FACTS=str(facts_lf(variant.lower() + "_build_facts.json")),
                       E28KT_NB5LIB=str(d / "nb5_lib.py"), E29KT_E28_NB=str(d / "e28k100.ipynb"), E28KT_K100Q_FIXTURE=str(d / "k100q_fixture.json") if "k100q_fixture.json" in ufiles else str(DS_DIR / "k100q_fixture.json"),
                       E29LT_NB=facts["notebook"], E29LT_FACTS=str(facts_lf(variant.lower() + "_build_facts.json")),
                       E29LT_KF=str(d / "e29_kit_facts.json"), E29LT_NB5LIB=str(d / "nb5_lib.py"), PYTHONIOENCODING="utf-8",
                       E29LT_PAYLOAD_DIR=str(DS_DIR))
            for f in ("e29_e28k100_unit_tests.py", "e29_kit_lib_tests.py"):
                r = subprocess.run([sys.executable, str(d / f)], capture_output=True, text=True, env=env, cwd=str(d), timeout=600)
                lines = r.stdout.splitlines()
                print(f"LOCAL {short} {f}: rc {r.returncode} |", next((l for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None),
                      "|", [l[:400] for l in lines if l.startswith(("FAIL", "SKIP"))], (r.stderr[-800:] if r.returncode else ""), flush=True)
                out.setdefault("local", {})[f"{short}:{f}"] = {"rc": r.returncode, "summary": next((l for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None),
                                                               "fails": [l[:400] for l in lines if l.startswith("FAIL")],
                                                               "skips": [l[:300] for l in lines if l.startswith("SKIP")]}
        out[f"units-{short}"] = build_kit(ns, f"units-{short}", units_body(ns["UNITS"]), ufiles, f"Q27 E29 Kit Units {short}", [], datasets, kf)
        if variant != "E29_DEFAULT":
            global E29_ARGPARSE
            E29_ARGPARSE = _mk_argparse(ns)
            out[f"real-{short}"] = build_kit(ns, f"real-{short}", real_body(ns, variant, facts), dict(ns["REAL_FILES"]),
                                             f"Q27 E29 Kit Real {short}", models_real, datasets, kf)
        if variant == "E29_DEFAULT":
            sfiles = {"e29_sched_tests.py": lf(HERE / "e29_sched_tests.py"), "e29_sched_patch.py": mod_text("e29_sched_patch.py"),
                      "e28dir/e28_sched_patch.py": mod_text("e28_sched_patch.py"), "exports/nb5_lib.py": lf(E25B / "nb5_lib.py")}
            nfix = {}
            for lab, ph in (("k100", Path(r"D:/ARC AGI 2/_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3")),
                            ("w4", Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_FP8_K3"))):
                fs = sorted(glob.glob(str(ph / "duck" / "*" / "branch_result.json")))
                nfix[lab] = len(fs)
                for f in fs:
                    sfiles[f"{lab}/{ph.name}/duck/{Path(f).parent.name}/branch_result.json"] = lf(f)
            skf = dict(kf, duck_payload_sha="9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566", n_fix=nfix,
                       sched_tests_min=len(re.findall(r"^    def test_", sfiles["e29_sched_tests.py"], re.M)))
            for v2 in ("E29_DFLASH_D3S", "E29_MTP_V64"):        # the scheduler module is the same file in all three notebooks
                assert json.loads(lf(facts_lf(v2.lower() + "_build_facts.json")))["modules"]["e29_sched_patch.py"] == facts["modules"]["e29_sched_patch.py"]
            out["sched"] = build_kit(ns, "sched", SCHED_BODY, sfiles, "Q27 E29 Kit Sched", [], datasets, skf)
        if variant == "E29_DFLASH_D3S" and (facts["e29"].get("direct") or {}).get("files"):
            dfiles = {"e29d_direct_tests.py": lf(HERE / "e29d_direct_tests.py"), "e29_sched_tests.py": lf(HERE / "e29_sched_tests.py"),
                      "e29_sched_patch.py": mod_text("e29_sched_patch.py"), "e28dir/e28_sched_patch.py": mod_text("e28_sched_patch.py")}
            dkf = dict(kf, duck_payload_sha="9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566",
                       direct_files=facts["e29"]["direct"]["files"],
                       direct_tests_min=len(re.findall(r"^    def test_", dfiles["e29d_direct_tests.py"], re.M)))
            out["direct"] = build_kit(ns, "direct", DIRECT_BODY, dfiles, "Q27 E29 Kit Direct", [], datasets, dkf)
    return out


DIRECT_BODY = r"""
U = {}
PAY = find_payload()
U["payload"] = {k: PAY.get(k) for k in ("used",)}
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    (B / n).parent.mkdir(parents=True, exist_ok=True)
    (B / n).write_text(v, encoding="utf-8", newline="\n")
U["payload_sha_ok"] = bool(PAY.get("used")) and sha_file(PAY["used"]) == KF["duck_payload_sha"]
U["direct_files"] = {}
for k_, (n_, s_) in KF["direct_files"].items():          # e29_direct.py / e29d_sched_patch.py from the mounted E29 payload dataset
    hit = [f for f in sorted(Path("/kaggle/input").glob("**/" + n_)) if sha_file(f) == s_]
    U["direct_files"][n_] = str(hit[0]) if hit else None
    if hit:
        (B / n_).write_bytes(hit[0].read_bytes())
for k_ in [k_ for k_ in os.environ if k_.startswith(("E28_", "E29_", "E12_", "ARC3_"))]:
    os.environ.pop(k_, None)
T = {}
try:
    r = sh("e29d_direct_tests", [sys.executable, str(B / "e29d_direct_tests.py"), "-v"],
           {"E29_TEST_E28_DIR": str(B / "e28dir"), "E28_TEST_PAYLOAD": str(PAY.get("used")), "E28_TEST_TMP": str(K),
            "PYTHONIOENCODING": "utf-8"}, 3000, cwd=B)
    m = re.search(r"E29 DIRECT tests: (\d+) run, (\d+) failures, (\d+) errors, (\d+) skipped", r["_out"])
    T.update(rc=r["rc"], sec=r["sec"], summary=m.group(0) if m else None, tail=r["tail"][-3000:],
             fails=[l[:400] for l in r["_out"].splitlines() if l.endswith(("FAIL", "ERROR")) or l.startswith(("FAIL:", "ERROR:"))][:40])
    T["all_pass"] = bool(m) and r["rc"] == 0 and m.group(2) == m.group(3) == "0" and int(m.group(1)) >= KF["direct_tests_min"]
    T["no_skips"] = bool(m) and m.group(4) == "0"
except BaseException as e:
    T.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["direct_tests"] = T
REP["checks"]["direct"] = U; save()
log("e29 direct tests", json.dumps({k: v for k, v in T.items() if k != "tb"})[:4000])
verdict({"duck_payload_mounted_sha_9e4b8928": U["payload_sha_ok"] is True,
         "e29_direct_files_from_the_e29_dataset_by_sha": all(U["direct_files"].values()),
         "e29d_direct_tests_all_pass": T.get("all_pass") is True, "e29d_direct_tests_no_skips": T.get("no_skips") is True},
        recorded={"summary": T.get("summary"), "fails": T.get("fails"), "payload": U["payload"], "direct_files": U["direct_files"]})
"""


if __name__ == "__main__":
    _only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    res = main(local_units="--local-units" in sys.argv, only=_only)
    rec = {"builder_sha256": sha((HERE / "build_e29_cpu_kit.py").read_bytes()), "e28_kit_sha256": E28_KIT_SHA,
           "notebooks": {v: BUILD_FACTS["variants"][v]["sha256"] for v in VARIANTS.values()}, "kits": res}
    _kf = HERE / "e29_kit_facts.json"                  # rev 2: --only keeps the other kits' records
    _old = json.loads(_kf.read_text(encoding="utf-8")) if (_only and _kf.is_file()) else {}
    rec["kits"] = dict(_old.get("kits") or {}, **res)
    _kf.write_bytes((json.dumps(rec, indent=1, default=str) + "\n").encode("utf-8"))
    print(json.dumps({k: (v.get("bytes"), v.get("sha256", "")[:16]) for k, v in res.items() if k != "local"}, indent=1))
