"""E33 kit: runs agent P's e33_poetiq_tests.py UNCHANGED with its build-source paths pointed at the kit's copies (no other change).

e33_poetiq_tests.py reads the E18 R2 notebook (the reference Poetiq core, cell 5) and e29_direct.py through e33_poetiq_build.py's
module constants E18_NB / E29_DIRECT (local D:/ paths) and load_sources' default arguments. On the Kaggle CPU kit those paths do
not exist (the reference / rebuild tests would SKIP); this runner imports e33_poetiq_build first, points both at the files next
to it (sha-pinned by the builder: E18 R2 116d2d41, e29_direct 8b42802c), then runs the test file as __main__ (its own
unittest.main). usage: python e33_kit_run_poetiq_tests.py [-v]
"""
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e33_poetiq_build as PB   # noqa: E402

PB.E18_NB = HERE / "e18r2.ipynb"
PB.E29_DIRECT = HERE / "e29_direct.py"
PB.load_sources.__defaults__ = (PB.E18_NB, PB.E29_DIRECT)
assert PB.E18_NB.is_file() and PB.E29_DIRECT.is_file(), (PB.E18_NB, PB.E29_DIRECT)
sys.argv = [str(HERE / "e33_poetiq_tests.py")] + sys.argv[1:]
runpy.run_path(str(HERE / "e33_poetiq_tests.py"), run_name="__main__")
