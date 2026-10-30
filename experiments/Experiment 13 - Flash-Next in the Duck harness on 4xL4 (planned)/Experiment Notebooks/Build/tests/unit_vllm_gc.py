"""E13 RC1: unit test of the REAL vLLM gc hook decoded from the notebook: a fake vllm.model_executor.model_loader.utils with
a MoE quant method whose repack leaves a garbage cycle holding the old buffer; the hook must wrap it, collect, and log."""
import base64, json, os, re, subprocess, sys, zlib
from pathlib import Path
NB = sys.argv[1]
W = Path(__file__).resolve().parent / "uvgc"
src = "".join("".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
m = re.search(r'E13_VLLM_HOOK_SRC = __import__\("zlib"\)\.decompress\(_b64\.b64decode\("([A-Za-z0-9+/=]+)"\)\)', src)
assert m, "vLLM hook not found"
(W / "hook").mkdir(parents=True, exist_ok=True)
(W / "hook" / "sitecustomize.py").write_text(zlib.decompress(base64.b64decode(m.group(1))).decode(), encoding="utf-8")
pk = W / "fakepkg" / "vllm" / "model_executor" / "model_loader"
pk.mkdir(parents=True, exist_ok=True)
for d in (W / "fakepkg" / "vllm", W / "fakepkg" / "vllm" / "model_executor", pk):
    (d / "__init__.py").write_text("")
(pk / "utils.py").write_text('''
class AutoGPTQMoEMethod:
    def process_weights_after_loading(self, layer):
        layer.repacked = True
class UnquantizedLinearMethod:
    def process_weights_after_loading(self, layer):
        layer.lin = True
def process_weights_after_loading(model, model_config, target_device):
    for name, module in model.named_modules():
        qm = getattr(module, "quant_method", None)
        if qm is not None:
            qm.process_weights_after_loading(module)
''')
(W / "run.py").write_text('''
import torch
from torch import nn
from vllm.model_executor.model_loader.utils import process_weights_after_loading, AutoGPTQMoEMethod, UnquantizedLinearMethod
class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.a = nn.Linear(2, 2); s.a.quant_method = UnquantizedLinearMethod()
        s.e = nn.Linear(2, 2); s.e.quant_method = AutoGPTQMoEMethod()
        s.e2 = nn.Linear(2, 2); s.e2.quant_method = s.e.quant_method      # one quant method shared by two layers
m = M()
process_weights_after_loading(m, None, "cpu")
print("OK", m.e.repacked, m.e2.repacked, m.a.lin, "instance_patched", "process_weights_after_loading" in vars(m.e.quant_method))
''')
tdir = W / "trace"
for f in tdir.glob("*") if tdir.exists() else []:
    f.unlink()
env = {**os.environ, "E13_VLLM_TRACE_DIR": str(tdir), "PYTHONPATH": os.pathsep.join([str(W / "hook"), str(W / "fakepkg")])}
p = subprocess.run([sys.executable, str(W / "run.py")], capture_output=True, text=True, env=env, timeout=120)
recs = [json.loads(l) for f in tdir.glob("*.jsonl") for l in open(f, encoding="utf-8")]
ev = [r["ev"] for r in recs]
vg = [r for r in recs if r["ev"] == "vgc"]
ok = (p.returncode == 0 and "OK True True True instance_patched False" in p.stdout and ev.count("vpatched") == 1
      and [r for r in recs if r["ev"] == "vpwal_end"][0]["moe_modules"] == 2
      and [r["module"] for r in vg] == ["e", "e2"]
      and [r for r in recs if r["ev"] == "vpwal_start"][0]["wrapped_classes"] == ["AutoGPTQMoEMethod"])
print("UNIT_VLLM_GC", json.dumps({"ok": ok, "events": ev, "stdout": p.stdout[-200:], "stderr": p.stderr[-400:]}))
sys.exit(0 if ok else 1)
