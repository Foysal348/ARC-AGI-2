"""E32 units (the Kaggle CPU kit runs this on the notebooks' pinned Py3.12 image; also run locally): e32_checks.run on the E32
notebook + the E29_DFLASH_D3S notebook + e32_lib.py / e32_nbx.py (the payload) + nb5_lib.py. The E28 / E29 modules
(e29_lib, e28k100_lib, e29_nbdf) are the ones of the E32 notebook's own CELL 1 literal (= what the notebook imports).

usage: python e32_kit_tests.py   env E32KT_NB, E32KT_E29NB, E32KT_LIB, E32KT_NBX, E32KT_NB5LIB (paths); E32KT_TMP (module dir)
Prints PASS / FAIL per check, then "<n>/<m> PASS"; exit 1 on any FAIL.
"""
import importlib.util
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e32_checks as C   # noqa: E402

RESULTS = []


def check(label, fn):
    try:
        ok, why = bool(fn()), ""
    except Exception as e:
        ok, why = False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-900:]}"
    RESULTS.append((label, ok))
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <- " + why[:1500]), flush=True)
    return ok


def load(name, path):
    s = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def context(nb, e29nb, lib, nbx, nb5lib, tmp=None):
    e32_nb, e29_nb = Path(nb).read_bytes(), Path(e29nb).read_bytes()
    mods = C.modules_literal(C.code_cells(e32_nb)[1])
    d = Path(tmp or tempfile.mkdtemp(prefix="e32kt_"))
    d.mkdir(parents=True, exist_ok=True)
    for n in ("e29_lib.py", "e28k100_lib.py"):
        (d / n).write_bytes(mods[n].encode("utf-8"))
    E29L = load("e29_lib", d / "e29_lib.py")
    E28L = load("e28k100_lib", d / "e28k100_lib.py")
    NB5L = load("nb5_lib", nb5lib)
    E32L = load("e32_lib", lib)
    return C.ctx(e32_nb, e29_nb, E29L, E28L, NB5L, E32L, Path(nbx).read_bytes().decode("utf-8"))


def main():
    e = os.environ
    X = context(e["E32KT_NB"], e["E32KT_E29NB"], e["E32KT_LIB"], e["E32KT_NBX"], e["E32KT_NB5LIB"], e.get("E32KT_TMP"))
    print("python", sys.version.split()[0], "| e32_lib", X["E32L"].LIB_VERSION, "| e29_lib", X["E29L"].LIB_VERSION, flush=True)
    C.run(X, check)
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print(f"{n_ok}/{len(RESULTS)} PASS", flush=True)
    return n_ok == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
