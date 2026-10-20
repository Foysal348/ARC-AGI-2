"""FN-SMOKE v3: Intel W4A16 main shards (dfranzen, already on Kaggle) + albucino FP8 PLE overlay (foysalemonshanto)
merged into one model folder; every shard checked (size + safetensors header SHA256 vs the pinned HF commit);
host-RAM gate lowered for the FP8 PLE table. Launch profile and all other stages unchanged from v2."""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE.parent / "franzen" / "make_fn_smoke.py"          # v2 builder (reviewed)
OUT = HERE / "make_fn_smoke_v3.py"
s = SRC.read_text(encoding="utf-8")
EXPECT = json.loads((HERE / "smoke_expect.json").read_text(encoding="utf-8"))


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('DST = Path(r"D:\\ARC AGI 2\\Research Base Experiment\\FN-SMOKE-FLASHNEXT-4XL4.ipynb")',
    'DST = Path(r"D:\\ARC AGI 2\\Research Base Experiment\\FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb")')
rep("C_HEAD = r'''# FN-SMOKE — Qwen3.8-Flash-Next (Intel W4A16 AutoRound) on Kaggle 4×L4 for ARC-AGI-2\n",
    "C_HEAD = r'''# FN-SMOKE v3 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2\n")
rep("1. inputs — model, wheelhouse, competition data present\n",
    "1. inputs — merge Intel main shards + FP8 PLE overlay into one folder; verify every shard (size + header SHA256)\n")
anchor = "only the tracked server process group is killed.\n"
rep(anchor, anchor + (
    "\nv3: model = the 15 main W4A16 shards of `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (byte-identical to "
    "Intel's HF commit 4c67bf6) + the FP8 PLE overlay `foysalemonshanto/qwen38-flash-next-fp8ple-overlay` (albucino "
    "c6af9fe: 10 FP8 PLE shards with scale, config with `ple_embedding_dtype: float8_e4m3fn`, index). Pennyroyal builds "
    "the pinned host PLE table directly in FP8 (~51 GB in total instead of ~102 GB BF16), so the available-RAM gate is "
    "70 GB. Load reads ~128 GB from disk instead of 181 GB. Launch profile unchanged from v2.\n"))

rep('MIN_RAM_AVAILABLE_GB = 110          # PLE pinned host table ~95.4 GB (TP4-sharded, 23.8 GB/rank) + headroom',
    'MIN_RAM_AVAILABLE_GB = 70           # FP8 PLE pinned host table ~51.2 GB (TP4-sharded, ~12.8 GB/rank) + headroom')
rep('''MODEL_DIR = _first(["/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
                    "/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
                    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/*/*/*"])''',
    '''INTEL_DIR = os.environ.get("FN_SMOKE_INTEL") or _first([
    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
    "/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/*/*/*"])
OVERLAY_DIR = os.environ.get("FN_SMOKE_OVERLAY") or _first([
    "/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1",
    "/kaggle/input/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1",
    "/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/*/*/*"])
MODEL_DIR = os.environ.get("FN_SMOKE_MODEL_DST", "/tmp/fn_model")   # merged folder, built in stage 1 (symlinks)''')
rep('REPORT = {"experiment": "FN-SMOKE-FLASHNEXT-4XL4", "rerun": RERUN, "stages": {}, "stopped_at": None,\n'
    '          "paths": {"competition": COMP_DIR, "wheelhouse": WHEELHOUSE_DIR, "model": MODEL_DIR}}',
    'REPORT = {"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4", "rerun": RERUN, "stages": {}, "stopped_at": None,\n'
    '          "paths": {"competition": COMP_DIR, "wheelhouse": WHEELHOUSE_DIR, "intel": INTEL_DIR, "overlay": OVERLAY_DIR,\n'
    '                    "model": MODEL_DIR}}')

# ---------------------------------------------------------------- stage 1: merge + verify
i = s.index("C_INPUTS = r'''")
j = s.index("'''\n", i + len("C_INPUTS = r'''")) + 4
NEW_INPUTS = (HERE / "stage1_v3.txt").read_text(encoding="utf-8")
NEW_INPUTS = NEW_INPUTS.replace("__EXPECT__", json.dumps(EXPECT, separators=(",", ":")))
s = s[:i] + NEW_INPUTS + s[j:]
OUT.write_text(s, encoding="utf-8")
print("wrote", OUT)
