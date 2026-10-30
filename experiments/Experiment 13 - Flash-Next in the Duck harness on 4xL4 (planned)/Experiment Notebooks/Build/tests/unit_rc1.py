"""E13 RC1: unit test of the REAL decoded sitecustomize (mp_memtrace + mp_nanprobe + mp_rc1, taken from the built notebook) on
CPU. A fake sglang loader runs postprocess_weights; a toy model with FusedMoE-like experts modules (hidden_states +
NamedTuple top-k, w13_scales / w13_qweight / expert_map) is driven through model.forward(...). Fake qwen4_exp (PLE layer)
and auto_round modules check the PLE mode switch and the layer-2 non-Marlin switch. usage: python unit_rc1.py <notebook>"""
import base64
import json
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path

NB = sys.argv[1]
HERE = Path(__file__).resolve().parent
W = HERE / "urc1"
nb = json.load(open(NB, encoding="utf-8"))
src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
m = re.search(r'E13_MEMTRACE_SRC = __import__\("zlib"\)\.decompress\(_b64\.b64decode\("([A-Za-z0-9+/=]+)"\)\)', src)
assert m, "zlib memtrace source not found in the notebook"
hook = zlib.decompress(base64.b64decode(m.group(1))).decode()
(W / "hook").mkdir(parents=True, exist_ok=True)
(W / "hook" / "sitecustomize.py").write_text(hook, encoding="utf-8")
base = W / "fakepkg" / "sglang" / "srt"
for d in (base / "model_loader", base / "models", base / "layers" / "quantization"):
    d.mkdir(parents=True, exist_ok=True)
for d in (W / "fakepkg" / "sglang", base, base / "model_loader", base / "models", base / "layers", base / "layers" / "quantization"):
    (d / "__init__.py").write_text("")
(base / "model_loader" / "loader.py").write_text('''
class GPTQMarlinMoEMethod:
    def process_weights_after_loading(self, module):
        module.done = True

class _QM(GPTQMarlinMoEMethod):
    pass

def _modules_with_quant_method(model):
    for n, m in model.named_modules():
        if n.endswith("mlp.experts"):
            m.quant_method = _QM()
            yield m, m.quant_method

class DefaultModelLoader:
    @staticmethod
    def load_weights_only(model, weights, target_device):
        return model
    @staticmethod
    def postprocess_weights(model, target_device):
        for module, qm in _modules_with_quant_method(model):
            qm.process_weights_after_loading(module)
''')
(base / "models" / "qwen4_exp.py").write_text('''
CALLS = {"orig": 0}
class _Stream:
    def __init__(self):
        self.syncs = 0
    def synchronize(self):
        self.syncs += 1
class Qwen4ExpPLELayer:
    def __init__(self):
        self._prefetch_stream = _Stream()
    def start_prefetch(self, batch, forward_batch):
        CALLS["orig"] += 1
        return "prefetched"
''')
(base / "layers" / "quantization" / "auto_round.py").write_text('''
class AutoRoundConfig:
    def get_quant_method(self, layer, prefix):
        return ("marlin", prefix)
    def apply_gptq_quant_layer(self, layer, prefix, backend="auto"):
        return ("wna16", prefix, backend)
''')
RUN = r'''
import dataclasses, enum, json, os, sys, typing, torch
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
class TopK(typing.NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor
class FusedMoE(nn.Module):
    def __init__(s, i):
        super().__init__(); s.i = i; s.hidden_size = 8; s.num_experts = 16; s.num_local_experts = 4; s.moe_ep_rank = 0
        s.w13_scales = nn.Parameter(torch.rand(4, 2, 16).to(torch.bfloat16), requires_grad=False)
        s.w13_qweight = nn.Parameter(torch.randint(-2**31, 2**31 - 1, (4, 1, 16), dtype=torch.int32), requires_grad=False)
        s.register_buffer("expert_map_cpu", torch.tensor([0, 1, 2, 3] + [-1] * 12, dtype=torch.int32))
        import types as _t
        s.dispatcher = _t.SimpleNamespace(local_expert_mapping=torch.tensor([-1] * 8 + [0, 1, 2, 3] + [-1] * 4, dtype=torch.int32))
        s.reduce_results = False
        w2 = torch.rand(4, 16, 3)
        w2[2, 5, 1] = float("nan")
        s.w2_scales = nn.Parameter(w2.transpose(1, 2), requires_grad=False)      # non-contiguous 3D [4, 3, 16]
        if SCEN == "badscale" and i == 2:
            with torch.no_grad():
                s.w13_scales[3, 1, 5] = float("nan")
    def forward(s, hidden_states, topk_output):
        y = hidden_states * topk_output.topk_weights.sum(-1, keepdim=True)
        if s.i == 2 and SCEN in ("badscale", "clean_nan"):
            y = y * float("nan")
        if SCEN == "inplace" and s.i in (1, 2):              # SGLang inplace=True: the output overwrites the input
            with torch.no_grad():
                hidden_states.copy_(y * (float("nan") if s.i == 2 else 3.0))
            return hidden_states
        if s.i == 2 and SCEN == "mutate":
            with torch.no_grad():
                s.w13_qweight.add_(1)
        return y
class Mlp(nn.Module):
    def __init__(s, i):
        super().__init__(); s.gate = nn.Linear(8, 16); s.experts = FusedMoE(i)
    def forward(s, x):
        lg = s.gate(x)
        w, ids = torch.topk(torch.softmax(lg, -1), 4)
        return s.experts(hidden_states=x, topk_output=TopK(w, ids.to(torch.int32), lg))
class Layer(nn.Module):
    def __init__(s, i):
        super().__init__(); s.mlp = Mlp(i); s.proj = nn.Linear(8, 8)
    def forward(s, hidden_states=None):
        return s.mlp(s.proj(hidden_states))
class Inner(nn.Module):
    def __init__(s):
        super().__init__(); s.embed = nn.Linear(8, 8); s.layers = nn.ModuleList([Layer(i) for i in range(6)])
    def forward(s, input_ids=None, forward_batch=None):
        x = s.embed(input_ids)
        for l in s.layers:
            x = l(hidden_states=x)
        return x
class Top(nn.Module):
    def __init__(s):
        super().__init__(); s.model = Inner(); s.lm = nn.Linear(8, 16)
    def forward(s, input_ids, positions, forward_batch):
        return LogitsOut(next_token_logits=s.lm(s.model(input_ids=input_ids, forward_batch=forward_batch)))
torch.manual_seed(0)
m = Top()
LD.DefaultModelLoader.load_weights_only(m, None, "cpu")
LD.DefaultModelLoader.postprocess_weights(m, "cpu")
for i in range(3):
    m.forward(torch.randn(10, 8), torch.arange(10), FB(Mode.EXTEND if i == 0 else Mode.DECODE, 1, 10 + i))
from sglang.srt.models import qwen4_exp as Q
P = Q.Qwen4ExpPLELayer()
res = {}
flag = os.environ["E13_PLE_FLAG"]
for mode in ("orig", "sync", "skip"):
    if mode == "orig":
        if os.path.exists(flag):
            os.remove(flag)
    else:
        open(flag, "w").write(mode)
    before = Q.CALLS["orig"]
    r = P.start_prefetch(None, None)
    res[mode] = {"ret": r, "orig_called": Q.CALLS["orig"] - before, "syncs": P._prefetch_stream.syncs}
from sglang.srt.layers.quantization import auto_round as AR
c = AR.AutoRoundConfig()
res["ar_l2"] = list(c.get_quant_method(None, "model.layers.2.mlp.experts"))
res["ar_l3"] = list(c.get_quant_method(None, "model.layers.3.mlp.experts"))
print("RES " + json.dumps(res))
'''
(W / "run_rc1.py").write_text(RUN, encoding="utf-8")
RES, BAD = {}, []


def run(scen, extra=None):
    tdir = W / f"trace_{scen}"
    (W / f"ple_{scen}").unlink(missing_ok=True)             # every run starts in PLE mode orig
    if tdir.exists():
        for f in tdir.iterdir():
            f.unlink()
    env = {**os.environ, "E13_MEMTRACE_DIR": str(tdir), "E13_NANPROBE": "1", "E13_NAN_FORWARDS": "240",
           "E13_NAN_ONLY": r"(^|\.)layers\.[0-3](\.|$)", "E13_RC1_SCAN": "1", "E13_RC1_LAYERS": "0,1,2,3",
           "E13_RC1_FIXTURE": "1", "E13_RC1_FIX_LAYERS": "1,2", "E13_RC1_FIX_FORWARDS": "2", "E13_RC1_FIX_TOKENS": "6",
           "E13_PLE_FLAG": str(W / f"ple_{scen}"), "E13_L2_NO_MARLIN": "1",
           "PYTHONPATH": os.pathsep.join([str(W / "hook"), str(W / "fakepkg")]), **(extra or {})}
    p = subprocess.run([sys.executable, str(W / "run_rc1.py"), scen], capture_output=True, text=True, env=env, timeout=300)
    recs = [json.loads(l) for f in tdir.glob("*.jsonl") for l in open(f, encoding="utf-8")]
    res = next((json.loads(l[4:]) for l in p.stdout.splitlines() if l.startswith("RES ")), {})
    return p, recs, res, tdir


def check(name, cond, info=""):
    RES[name] = bool(cond)
    if not cond:
        BAD.append(name)
        print("FAIL", name, str(info)[:700])


p, r, res, tdir = run("badscale")
check("rc0", p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:])
ev = [x["ev"] for x in r]
check("no hook errors / patch errors", not [x for x in r if x["ev"] in ("hook_error", "rc1_scan_error", "rc1_patch_error",
                                                                          "nan_hooks_error", "patch_error")],
      [x for x in r if "error" in x["ev"]])
sc = {x["when"]: x["layers"] for x in r if x["ev"] == "rc1_scan"}
check("two scans (after postprocess, after forward 0), layers 0-3", set(sc) == {"after_postprocess", "after_forward_0"}
      and all(set(v) == {"0", "1", "2", "3"} for v in sc.values()), sc.keys())
t2 = ((sc.get("after_postprocess") or {}).get("2") or {}).get("tensors") or {}
check("layer-2 scale NaN found with its local expert (row 3)", (t2.get("w13_scales") or {}).get("nonfinite") == 1
      and (t2.get("w13_scales") or {}).get("bad_rows") == [3], t2.get("w13_scales"))
t1 = ((sc.get("after_postprocess") or {}).get("1") or {}).get("tensors") or {}
check("int checksum + expert map scanned", isinstance((t1.get("w13_qweight") or {}).get("wsum"), int)
      and (t1.get("expert_map_cpu") or {}).get("min") == -1 and (t1.get("w13_scales") or {}).get("nonfinite") == 0, t1)
s0 = (sc.get("after_postprocess") or {}).get("0") or {}
check("runtime source sha of the quant method", len(((s0.get("src") or {}).get("quant_method") or {}).get("sha256", "")) == 64
      and s0.get("quant_method") == "_QM", s0.get("src"))
nf = [x for x in r if x["ev"] == "nan_forward"]
check("3 forwards logged; layers 0-3 only (E13_NAN_ONLY)", len(nf) == 3 and set(nf[0]["layers"]) == {f"model.layers.{i}" for i in range(4)},
      [list(x["layers"]) for x in nf])
f0 = (nf[0]["first_nonfinite"] or [{}])[0] if nf else {}
check("first NaN = layer-2 experts with finite inputs", f0.get("module") == "model.layers.2.mlp.experts" and f0.get("inputs_bad") is False, f0)
mi = (nf[0].get("moe_in") or {}).get("model.layers.2.mlp.experts") if nf else None
check("MoE inputs incl. NamedTuple top-k: hidden bf/fp + int32 ids with min/max/stride",
      mi and any(d["dtype"] == "int32" and "min" in d and "stride" in d for d in mi) and any(d["dtype"] == "float32" for d in mi), mi)
check("ple_mode on every forward", all(x.get("ple_mode") == "orig" for x in nf), [x.get("ple_mode") for x in nf])
fx = sorted(f.name for f in tdir.glob("fixture_*.pt"))
check("fixtures: L1/L2 x forwards 0-1", len(fx) == 4 and any("L1_fw0" in n for n in fx) and any("L2_fw1" in n for n in fx), fx)
if fx:
    import torch
    d = torch.load(tdir / [n for n in fx if "L2_fw0" in n][0], weights_only=False)
    check("fixture content: 6 tokens hidden, int ids [6,4], weights, NaN output, expert map, meta",
          list(d["hidden"].shape) == [6, 8] and any(t is not None and list(t.shape) == [6, 4] for t in d["ints"])
          and any(t is not None and list(t.shape) == [6, 4] for t in d["floats"]) and bool(torch.isnan(d["out"]).all())
          and d.get("expert_map") is not None and d["meta"].get("num_local_experts") == 4, {k: (v if not hasattr(v, "shape") else list(v.shape)) for k, v in d.items()})
check("PLE switch: orig calls SGLang, sync calls + synchronizes, skip returns None without prefetch",
      res.get("orig", {}).get("orig_called") == 1 and res.get("orig", {}).get("syncs") == 0
      and res.get("sync", {}).get("orig_called") == 1 and res.get("sync", {}).get("syncs") == 1
      and res.get("skip", {}).get("orig_called") == 0 and res.get("skip", {}).get("ret") is None, res)
check("L2 non-Marlin switch: layer 2 -> apply_gptq_quant_layer(backend=gptq), layer 3 untouched",
      res.get("ar_l2") == ["wna16", "model.layers.2.mlp.experts", "gptq"] and res.get("ar_l3") == ["marlin", "model.layers.3.mlp.experts"], res)
check("events: rc1_ple_patched + rc1_ar_patched + rc1_l2_no_marlin + ple_mode", {"rc1_ple_patched", "rc1_ar_patched", "rc1_l2_no_marlin",
                                                                                "ple_mode"} <= set(ev), sorted(set(ev)))

t2b = ((sc.get("after_postprocess") or {}).get("2") or {}).get("tensors") or {}
w2 = t2b.get("w2_scales") or {}
check("non-contiguous 3D scale scanned row by row: 1 NaN in expert row 2, contig False",
      w2.get("nonfinite") == 1 and w2.get("bad_rows") == [2] and w2.get("contig") is False and w2.get("shape") == [4, 3, 16], w2)
dm = t2b.get("dispatcher.local_expert_mapping") or {}
check("scan includes the dispatcher's EP map", dm.get("min") == -1 and dm.get("max") == 3 and dm.get("shape") == [16], dm)
if fx:
    import torch
    d0 = torch.load(tdir / [n for n in fx if "L2_fw0" in n][0], weights_only=False)
    check("fixture expert map = the dispatcher map (rank 2 block 8..11), reduce_results recorded, hidden from pre hook",
          d0["expert_map"].tolist() == [-1] * 8 + [0, 1, 2, 3] + [-1] * 4 and d0["meta"].get("reduce_results") is False
          and d0.get("hidden_from") == "pre_hook", (d0.get("expert_map"), d0.get("meta"), d0.get("hidden_from")))

p, r, res, tdir = run("inplace", {"E13_SYNC_RE": r"layers\.2\.mlp\.experts$"})
nf = [x for x in r if x["ev"] == "nan_forward"]
f0 = (nf[0]["first_nonfinite"] or [{}])[0] if nf else {}
check("in-place MoE: input finiteness checked before the overwrite (inputs_bad False, origin layer 2)",
      p.returncode == 0 and f0.get("module") == "model.layers.2.mlp.experts" and f0.get("inputs_bad") is False, f0)
import torch
fxs = {f.name: torch.load(f, weights_only=False) for f in tdir.glob("fixture_L1_fw0_*.pt")}
d1 = next(iter(fxs.values()), None)
check("in-place MoE: the fixture keeps the true input (output = 3 x input x sum(w), not equal to the stored hidden)",
      d1 is not None and d1["hidden_from"] == "pre_hook" and not torch.allclose(d1["hidden"].float(), d1["out"].float()), d1 and d1["hidden_from"])
check("E13_SYNC_RE on CPU: the failed sync is logged, never a hook error",
      any(x["ev"] == "sync_error" for x in r) and not [x for x in r if x["ev"] == "hook_error"], [x["ev"] for x in r if "error" in x["ev"]][:5])

p, r, res, tdir = run("mutate", {"E13_L2_NO_MARLIN": "0"})
sc = {x["when"]: x["layers"] for x in r if x["ev"] == "rc1_scan"}
a = ((((sc.get("after_postprocess") or {}).get("2") or {}).get("tensors") or {}).get("w13_qweight") or {})
b = ((((sc.get("after_forward_0") or {}).get("2") or {}).get("tensors") or {}).get("w13_qweight") or {})
check("a weight change during forward 0 shows in the checksums", p.returncode == 0 and a.get("sum") is not None and a.get("sum") != b.get("sum"), (a, b))
check("E13_L2_NO_MARLIN=0: auto_round not patched", res.get("ar_l2") == ["marlin", "model.layers.2.mlp.experts"]
      and "rc1_ar_patched" not in [x["ev"] for x in r], res)

print("UNIT_RC1 " + json.dumps({"ok": not BAD, "failed": BAD, "cases": len(RES)}))
sys.exit(0 if not BAD else 1)
