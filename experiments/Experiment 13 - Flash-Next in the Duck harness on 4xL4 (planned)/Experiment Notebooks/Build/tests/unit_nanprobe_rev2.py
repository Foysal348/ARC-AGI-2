"""E13 NANPROBE rev 2: unit test of the REAL decoded sitecustomize (taken from the built notebook) on CPU.
A fake sglang.srt.model_loader.loader runs postprocess_weights; a toy model shaped like SGLang's calls is driven through
model.forward(...) (never __call__), with keyword-called layers, a NaN made in plain function code between modules, an int32
indexer output, a dataclass logits output, an fp8 output and a constant collapse. usage: python unit_rev2.py <notebook>"""
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

NB = sys.argv[1]
HERE = Path(__file__).resolve().parent
W = HERE / "u2"
nb = json.load(open(NB, encoding="utf-8"))
src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
m = re.search(r'E13_MEMTRACE_SRC = _b64\.b64decode\("([A-Za-z0-9+/=]+)"\)', src)
assert m, "memtrace source not found in the notebook"
hook = base64.b64decode(m.group(1)).decode()
(W / "hook").mkdir(parents=True, exist_ok=True)
(W / "hook" / "sitecustomize.py").write_text(hook, encoding="utf-8")
pk = W / "fakepkg" / "sglang" / "srt" / "model_loader"
pk.mkdir(parents=True, exist_ok=True)
for p in (W / "fakepkg" / "sglang", W / "fakepkg" / "sglang" / "srt", pk):
    (p / "__init__.py").write_text("")
(pk / "loader.py").write_text('''
class _QM:
    def process_weights_after_loading(self, module):
        module.done = True

def _modules_with_quant_method(model):
    for m in model.modules():
        if hasattr(m, "weight") and m is not model:
            yield m, _QM()

class DefaultModelLoader:
    @staticmethod
    def load_weights_only(model, weights, target_device):
        return model
    @staticmethod
    def postprocess_weights(model, target_device):
        for module, qm in _modules_with_quant_method(model):
            qm.process_weights_after_loading(module)
''')
RUN = r'''
import dataclasses, enum, json, os, sys, torch
from torch import nn
from sglang.srt.model_loader import loader as LD
SCEN = sys.argv[1]
class Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
@dataclasses.dataclass
class FB:
    forward_mode: Mode
    batch_size: int
    seq_lens_sum: int
@dataclasses.dataclass
class LogitsOut:
    next_token_logits: torch.Tensor
class Indexer(nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 4), -1 if SCEN == "indexer" else 0, dtype=torch.int32)
class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.o_proj = nn.Linear(8, 8); s.indexer = Indexer()
    def forward(s, x):
        s.indexer(x)
        return s.o_proj(x)
class Layer(nn.Module):
    def __init__(s, i):
        super().__init__(); s.i = i; s.self_attn = Attn(); s.mlp = nn.Linear(8, 8)
    def forward(s, hidden_states=None):
        h = s.self_attn(hidden_states)
        if SCEN == "between" and s.i == 2:
            h = h * float("nan")                      # plain function code between modules
        out = s.mlp(h)
        if SCEN == "second" and s.i == 1:
            return out, out * float("nan")            # NaN only in the 2nd output tensor
        if SCEN == "collapse" and s.i == 3:
            return torch.zeros_like(out)
        return out
class Inner(nn.Module):
    def __init__(s):
        super().__init__(); s.embed = nn.Linear(8, 8); s.layers = nn.ModuleList([Layer(i) for i in range(5)])
    def forward(s, input_ids=None, forward_batch=None):
        x = s.embed(input_ids)
        for l in s.layers:
            x = l(hidden_states=x)                    # keyword-only call (SGLang style)
            x = x[0] if isinstance(x, tuple) else x
        return x
class Fp8Out(nn.Module):
    def forward(self, x):
        return x.to(torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else x
class Top(nn.Module):
    def __init__(s):
        super().__init__(); s.model = Inner(); s.lm = nn.Linear(8, 16); s.q8 = Fp8Out()
    def forward(s, input_ids, positions, forward_batch):
        h = s.model(input_ids=input_ids, forward_batch=forward_batch)
        s.q8(h)
        lg = s.lm(h)
        if SCEN == "logits":
            lg = lg * float("nan")
        return LogitsOut(next_token_logits=lg)
m = Top()
LD.DefaultModelLoader.load_weights_only(m, None, "cpu")
LD.DefaultModelLoader.postprocess_weights(m, "cpu")
for i in range(int(os.environ.get("CALLS", "4"))):
    fb = FB(Mode.EXTEND if i == 0 else Mode.DECODE, 2, 10 + i)
    m.forward(torch.randn(2, 8), torch.arange(2), fb)   # direct .forward, as eager_runner.py does
print("FORWARD_IS_CLASS", "forward" not in m.__dict__)
'''
(W / "run_u2.py").write_text(RUN, encoding="utf-8")
RES, BAD = {}, []


def run(scen, nfw=3, calls=4):
    tdir = W / f"trace_{scen}"
    for f in tdir.glob("*.jsonl") if tdir.exists() else []:
        f.unlink()
    env = {**os.environ, "E13_MEMTRACE_DIR": str(tdir), "E13_NANPROBE": "1", "E13_NAN_FORWARDS": str(nfw), "CALLS": str(calls),
           "PYTHONPATH": os.pathsep.join([str(W / "hook"), str(W / "fakepkg")])}
    p = subprocess.run([sys.executable, str(W / "run_u2.py"), scen], capture_output=True, text=True, env=env, timeout=300)
    recs = [json.loads(l) for f in tdir.glob("*.jsonl") for l in open(f, encoding="utf-8")]
    return p, recs


def check(name, cond, info=""):
    RES[name] = bool(cond)
    if not cond:
        BAD.append(name)
        print("FAIL", name, str(info)[:600])


p, r = run("clean")
nf = [x for x in r if x["ev"] == "nan_forward"]
check("rc0_and_class_forward_restored", p.returncode == 0 and "FORWARD_IS_CLASS True" in p.stdout, p.stdout + p.stderr)
check("direct_forward_logged_3_then_removed", len(nf) == 3 and any(x["ev"] == "nan_hooks_removed" for x in r), [x["ev"] for x in r])
check("forward_mode_batch_recorded", nf and nf[0].get("forward_mode") == "EXTEND" and nf[1].get("forward_mode") == "DECODE"
      and nf[0].get("batch_size") == 2 and nf[0].get("seq_lens_sum") == 10, nf[:2])
check("logits_stats_present", nf and nf[0]["logits"].get("present") and "argmax" in nf[0]["logits"] and "top_gap" in nf[0]["logits"], nf[:1])
check("layer_stats_every_forward", all(len(x.get("layers") or {}) == 5 for x in nf), [x.get("layers") for x in nf])
check("fp8_output_no_crash_no_error", not [x for x in r if x["ev"] == "hook_error"], [x for x in r if x["ev"] == "hook_error"])
check("clean_no_nonfinite", all(not x["first_nonfinite"] for x in nf), nf)

p, r = run("between")
nf = [x for x in r if x["ev"] == "nan_forward"]
f0 = (nf[0]["first_nonfinite"] or [{}])[0] if nf else {}
check("between_blamed_as_propagated", f0.get("module") == "model.layers.2.mlp" and f0.get("inputs_bad") is True
      and f0.get("prev_clean_module") == "model.layers.2.self_attn", f0)

p, r = run("second")
nf = [x for x in r if x["ev"] == "nan_forward"]
f0 = (nf[0]["first_nonfinite"] or [{}])[0] if nf else {}
check("second_output_tensor_caught", f0.get("module") == "model.layers.1" and f0.get("tensor_idx") == 1, nf[:1])

p, r = run("logits")
nf = [x for x in r if x["ev"] == "nan_forward"]
check("logits_only_nan_caught", nf and nf[0]["logits"].get("nonfinite") is True and not nf[0]["first_nonfinite"], nf[:1])

p, r = run("collapse")
nf = [x for x in r if x["ev"] == "nan_forward"]
check("collapse_constant_caught", nf and any(c["module"] == "model.layers.3" and c["value"] == 0.0 for c in nf[0]["const"]), nf[:1])

p, r = run("indexer")
nf = [x for x in r if x["ev"] == "nan_forward"]
ix = (nf[0].get("indexer") or [{}])[0] if nf else {}
check("indexer_int_stats", ix.get("dtype") == "int32" and ix.get("min") == -1 and ix.get("neg1") == 8, ix)

p, r = run("clean", nfw=2, calls=5)
nf = [x for x in r if x["ev"] == "nan_forward"]
check("window_limit_respected", len(nf) == 2 and p.returncode == 0, len(nf))

print("UNIT_REV2 " + json.dumps({"ok": not BAD, "failed": BAD, "cases": len(RES)}))
sys.exit(0 if not BAD else 1)
