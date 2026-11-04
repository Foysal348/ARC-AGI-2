"""E32 quick unit tests (local, seconds; no server, no GPU, no Kaggle). usage: python e32_tests.py  (run after build_e32_levers.py)

  e32_checks.run (P / A / O / D / R / S / K, the same checks the Kaggle CPU units kit runs) on the built notebook, the E29
  production notebook, Build/e32_lib.py + Build/e32_nbx.py, the E29 kit-local nb5_lib.py; the E28 / E29 modules from the E32
  notebook's own CELL 1 literal. Plus the builder-only checks:
  I1  the notebook on disk = this build; CELL 0 = E18 R2; the E29 payload pins = the E29 Build files = the E29 payload folder
  I2  every changed line is an E32 edit (exact edit texts); 7 E29 lines replaced; the SMOKE copy = ONE line
  I3  the push folders (prod / smoke / submit) + the E32 payload folder
  I4  the E29 production files byte-identical
Prints PASS / FAIL per check, then "<n>/<m> PASS"; exit 1 on any FAIL.
"""
import difflib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_e32_levers as B   # noqa: E402  (module import: reproduces E29 byte for byte, applies the E32 edits in memory)
import e32_kit_tests as KT     # noqa: E402

E29B = B.E29B
NB_PATH = B.ROOT / "Experiment Notebooks" / f"{B.NB_STEM}.ipynb"
SM_PATH = B.ROOT / "Experiment Notebooks" / f"{B.NB_STEM}_SMOKE.ipynb"
E29_NB = B.E29_PROTECTED["notebook"][0]
NB5LIB = B.E29_BUILD / "_kit_local" / "dflash-d3s" / "nb5_lib.py"     # = the K100Q payload's nb5_lib (sha 18efcbc1), read only
check = KT.check
assert (HERE / "e32_lib.py").read_text(encoding="utf-8") == B.LIB_TEXT, "Build/e32_lib.py is stale: run build_e32_levers.py"
X = KT.context(NB_PATH, E29_NB, HERE / "e32_lib.py", HERE / "e32_nbx.py", NB5LIB)
assert X["E29L"].LIB_VERSION == B.E29L.LIB_VERSION and X["E32_FILES"] == B.E32_FILES
KT.C.run(X, check)
NB_BYTES = NB_PATH.read_bytes()
readable = KT.C.all_cells


def _i1():
    assert NB_BYTES == B.assemble(list(B.CELLS))[0], "the notebook on disk != this build"
    assert B.CELLS[1][1] == E29B.KQ.E18R2_CELL0
    for k, (n, s) in E29B.DIRECT_FILES.items():
        assert B.sha((B.E29_BUILD / n).read_bytes()) == s == B.sha((E29B.DS_DIR / n).read_bytes())
    return B.sha(NB5LIB.read_bytes()) == "18efcbc1dde9c590a14ac1937b45dec563ff76ed4d92b16632d273dec2c88500"


check("I1 the notebook on disk = this build; CELL 0 = E18 R2's (dual mode + fallback submission); the E29D_FILES pins = the E29 "
      "Build files = the E29 payload folder; nb5_lib = the K100Q payload's (18efcbc1)", _i1)


def _i2():
    d = B.readable(B.CELLS).splitlines()
    e = B.readable(B.E29_CELLS).splitlines()
    add = [l[1:] for l in difflib.unified_diff(e, d, n=0, lineterm="") if l.startswith("+") and not l.startswith("+++")]
    rem = [l[1:] for l in difflib.unified_diff(e, d, n=0, lineterm="") if l.startswith("-") and not l.startswith("---")]
    news = set(B.X4_BLOCK.splitlines()) | set(B.X2_SWITCH.splitlines()) | set(B.X9[1].splitlines()) | set(B.X10[1].splitlines())
    olds = set(B.X9[0].splitlines()) | set(B.X10[0].splitlines())
    for _, a, b in B.ENGINE_EDITS:
        news |= set(b.splitlines())
        olds |= set(a.splitlines())
    frag_n = [x for x in news if x.strip()] + [B.X3_NEW]
    frag_o = [x for x in olds if x.strip()] + [B.X3_OLD]
    bad_add = [l for l in add if l.strip() and l not in news and not any(f in l for f in frag_n)]
    bad_rem = [l for l in rem if l.strip() and l not in olds and not any(f in l for f in frag_o)]
    assert not bad_add and not bad_rem, (bad_add[:8], bad_rem[:8])
    assert len(rem) == 7, rem                                 # 7 E29 lines replaced (each by its E32-marked line)
    sm = SM_PATH.read_bytes()
    s1, s0 = readable(sm), readable(NB_BYTES)
    dl = [(i, a, b) for i, (x, y) in enumerate(zip(s0, s1)) for a, b in zip(x.splitlines(), y.splitlines()) if a != b]
    assert len(dl) == 1 and 'E29_PROFILE = "E29SMOKE"' in dl[0][2] and len(sm) < 400_000, dl
    return True


check("I2 every changed line is one of the marked E32 edits (X1-X10 exact texts); 7 E29 lines replaced (X3 smoke budget, X4 extra, X5 "
      "capacity, X7 seqs / top / name, X8 relaunch); the SMOKE copy differs by ONE line (E29_PROFILE = \"E29SMOKE\")", _i2)


def _i3():
    out = {}
    e29km = json.loads((B.PUSH_ROOT / "q27-xhigh-e29-dflash-d3s__foysalemonshanto" / "kernel-metadata.json").read_text(encoding="utf-8"))
    for suf in ("", "-smoke", "-submit"):
        p = B.PUSH_ROOT / f"{B.SLUG}{suf}__{B.OWNER}"
        km = json.loads((p / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert km["id"] == f"{B.OWNER}/{B.SLUG}{suf}" and km["enable_gpu"] and km["machine_shape"] == e29km["machine_shape"]
        assert km["docker_image"] == e29km["docker_image"] and km["competition_sources"] == e29km["competition_sources"]
        assert km["dataset_sources"] == e29km["dataset_sources"] + [B.DS_ID] and km["model_sources"] == e29km["model_sources"]
        assert km["is_private"] is True and km["enable_internet"] is False
        out[suf] = B.sha((p / km["code_file"]).read_bytes())
    sm = B.sha(SM_PATH.read_bytes())
    ds = {f.name: B.sha(f.read_bytes()) for f in B.DS_DIR.iterdir()}
    sums = (B.DS_DIR / "SHA256SUMS").read_text(encoding="utf-8")
    return (out[""] == B.sha(NB_BYTES) and out["-smoke"] == out["-submit"] == sm and ds["e32_lib.py"] == B.E32_FILES["lib"][1]
            and ds["e32_nbx.py"] == B.E32_FILES["nbx"][1] and all(f"{v}  {k}" in sums for k, v in ds.items() if k.endswith(".py")))


check("I3 push folders (prod / smoke / submit = the smoke bytes, owner foysalemonshanto): the E29 datasets + models unchanged in order, "
      "+ the E32 payload dataset; the payload folder holds e32_lib.py / e32_nbx.py at the notebook's pinned shas (SHA256SUMS)", _i3)


def _i4():
    st = B.protected_state()
    return st == B.PROTECTED_BEFORE and st["notebook"] == B.E29_PROTECTED["notebook"][1] and st["smoke"] == B.E29_PROTECTED["smoke"][1]


check("I4 the E29 production files are byte-identical (the two notebooks at their pinned shas, the prod / submit push folders and the "
      "E29 payload folder file by file)", _i4)

n_ok = sum(1 for _, ok in KT.RESULTS if ok)
print(f"{n_ok}/{len(KT.RESULTS)} PASS", flush=True)
sys.exit(0 if n_ok == len(KT.RESULTS) else 1)
