"""E27 NBDF-K100 CPU kits (units + real, foysalemonshanto: the drafter Model is foysal-private): the NBDF kit builder
(build_e27_nbdf_cpu_kit.py) run with NBDF_VARIANT=k100. Differences (marked "E27 K100" there):
  units  + nbdf_k100_unit_tests.py on the K100 body (method map / scales WITHOUT MTP on the real k100load_lib, the provisional
         arm set, nbdf_main, the K3-ladder step, the memory gate as a flag), k100load_lib.py and the QSA scales sidecar bundled
  real   the K100 notebook's export (CELL 1 loads the lab and CELL 2b the K100 payload from the tf-ref dataset, attached);
         R9 get_kv_cache_configs on the klee100 K3_ONLY v2 GPU worker record: Q must calibrate to 480 blocks (479 usable) at
         7,933,526,016 B, then K3 / K4 SW-1024 ~540 blocks / pages / F / groups at the same pin (research A's 10-item list); R12 drafter load; the seam apply / verify /
         revert; the argv of the arm set; CELL 0 identity; the provenance the gated rebuild needs (as NBDF)
Run: python build_e27_nbdf_k100_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e27-nbdf-k100-kit-{units,real}__foysalemonshanto/
"""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    os.environ["NBDF_VARIANT"] = "k100"
    sys.argv = [str(HERE / "build_e27_nbdf_cpu_kit.py")] + sys.argv[1:]
    runpy.run_path(str(HERE / "build_e27_nbdf_cpu_kit.py"), run_name="__main__")
