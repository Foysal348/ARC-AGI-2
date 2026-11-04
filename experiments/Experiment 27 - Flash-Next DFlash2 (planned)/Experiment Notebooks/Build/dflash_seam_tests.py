"""CPU tests for dflash_seam.py (E27 DFlash2 seam on Merlin v3.0.0). Quick, no GPU, no vllm install needed: the REAL
original and patched Merlin functions / classes are extracted by AST from copies of the Merlin source and run against
stubbed vllm imports and small fakes.

  python dflash_seam_tests.py [--merlin-src D:/ARC AGI 2/_scratch/e23/p2/merlin-src] [--nb4-dir <E25 Build dir>]

T1  patch mechanics: the 6 source files are at Merlin's pre-sha; apply -> post-sha + marker + LF + py_compile, idempotent;
    revert -> exact pre-sha; refuses a modified copy and a mixed tree without writing; no-fuzz applier; ring=False /
    draft_block=False subsets; every hunk carries "E27"; header shas = constants; ast.parse(feature_version=3.12).
T2  interfaces (AST): Qwen4ExpModel(nn.Module, EagleModelMixin) + supports_aux_hidden_states_over_pp = True + the E27
    methods; SupportsEagle3 on Qwen4ExpForCausalLM AND Qwen4ExpForConditionalGeneration; imports; the original has
    none of it. MRO mimic: every class of the real base graph (interfaces.py / interfaces_base.py / qwen3_vl.py /
    qwen3_5.py / model.py) rebuilt from its AST base list: original and patched hierarchies build (no MRO TypeError),
    SupportsEagle3 is in the MRO of both served classes, Qwen3ForCausalLM -> DFlash*ForCausalLM has has_own_embed_tokens.
T3  forward meta test (the REAL Qwen4ExpModel methods, tiny config hc 4 x H 8, 48 layers, fake HC layers with the
    delayed-combine protocol, the REAL EagleModelMixin / SupportsEagle3 / eagle3_utils):
    a env-off identity: original vs patched (no taps set), PP1 / PP2 stage 0 / stage 1, HC-SP (2-rank thread sim),
      deepstack on / off: outputs bitwise equal, op traces equal;
    b PP1 taps via the real set_eagle3_aux_hidden_state_layers (dflash target_layer_ids [5,15,25,35,45] -> (6,16,26,36,46)):
      (hidden, 5 x [T, 8]) bf16, each tap bitwise = mean over hc of the independently computed completed layer output;
      hidden unchanged; CausalLM- and ConditionalGeneration-shaped outers;
    c PP2 25/23: stage 0 returns hidden + aux_hidden_states_0/1 (taps 6, 16); the real reserve_aux_intermediate_tensor_slots
      reserves the same keys [B, 8] on stage 1; slot base 2, upstream total 2; the runner copy; stage 1 returns REMOTE
      FIRST [6, 16, 26, 36, 46] bitwise = PP1 (a permuted order is detected); verify_supports_aux_hidden_states_over_pp;
    d HC-SP: taps gathered to the full token axis, = reference; one extra gather per tap;
    e boundary ids (0, 1, 25, 26, 48) over PP2; f deepstack + taps; g tap modes pre_combine / norm_input (+ refusals);
    h fail loud: no DFLASH_SEAM, out-of-range / empty ids, tampered tap count, missing relay key.
T4  config.py allowlist: original vs patched over methods, env off identical (type + message), env on admits dflash only.
T5  qsa_cache ring: env off = original (capacity or the same AssertionError) for K 0..12 x 15 block sizes; env on: same
    capacity whenever the original passes, else the brute-force widening or ValueError; K 5/6/7 @ 4096 -> 16.
T6  qwen3_dflash load_weights: env off identical; repacked drafter (embed_tokens.weight) loaded as model.embed_tokens.weight
    with has_own_embed_tokens True and _e27_embed_loaded True; loaded-weights check (missing tensor raises, opt-out);
    sidecar fallback (file / index dir / plain dir / key / mtp exclusion / ambiguous / shape / dtype / zero / NaN / both
    / PP1 skip); dflash_seam.embed_row_sha.
T7  load_dflash_model: the REAL maybe_share_target_embed: Merlin raises under PP2 without an embedding (negative
    control, original and patched); repacked: own + shared head, the replaced draft lm_head released; PP1: shared,
    the replaced draft embed released; zero / NaN / not-loaded embed, missing head, quantized linear raise; the
    silent early-return path (a drafter class without has_own_embed_tokens) raises only patched; mask row sha
    (logged, asserted with DFLASH_MASK_ROW_SHA, other TP shard); DFLASH_DRAFT_EMBED_HOST guards; env off identical.
T8  attention.py drafter SW block: the REAL get_kv_cache_spec: env off identical; DFLASH_DRAFT_SW_BLOCK=1024 on draft
    layers only; a non-divisor raises.
T9  readback_label / parse_log / server_env on synthetic lines.
T10 composition with nb4_ppack (E25 Q-pack / Q-defer): disjoint files, both apply orders give the same file shas,
    each reverts independently, the runner's aux relay lines survive NB4's patch; combined_tree_state.
T11 dflash/speculator.py in-graph draft metadata (vLLM #63587 / PR #64059 port): AST placement (class default, the
    configure call at the end of set_attn after super().set_attn, the elif refresh right after the skipped
    _build_uniform_attn_metadata and before run_fullgraph, the original has no else); the REAL
    _e27_configure_ingraph_meta (env off: untouched, no log; DFLASH_SEAM=1: refresh = in-graph flag; native off;
    DFLASH_NO_INGRAPH_META=1 forces the eager rebuild); the REAL elif body + the REAL
    AttentionGroup.map_to_kernel_block_table with manager block 4096 / kernel block 64 (and a packed stride, and a
    group without kernel blocks): the replay-time kernel table expands the CURRENT manager ids (3 -> 192..255), the
    unrefreshed table keeps the capture-time ids (the #63587 negative control).
T12 qwen3_dflash2.py grouped conv A/B (DFLASH_CONV_IMPL=torch; vLLM #58250 / #55960, SGLang #41736): AST (the flag
    branch first, the rest of _grouped_conv unchanged; the fused kernel masks its tap loads); the REAL
    _e27_conv_torch_enabled env gating + log; the REAL torch path vs Merlin's torch fallback (bitwise on finite input,
    Flash-Next shapes 160 groups x 16, taps 2, blocks 4 / 5 / 8) and vs the #55960 fp32 loop reference; NaN in one
    request's rows: Merlin's fallback leaks it into the next request, the E27 path does not; the flag routes around
    the fused op.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Literal, Protocol, cast, overload, runtime_checkable

import torch
from torch import nn

try:
    from typing import TypeIs
except ImportError:  # Python 3.12
    from typing_extensions import TypeIs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflash_seam as D  # noqa: E402

DEFAULT_SRC = "D:/ARC AGI 2/_scratch/e23/p2/merlin-src"
DEFAULT_NB4 = "D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build"
RESULTS = []
E27_ENVS = D.ALL_ENVS


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL {name}: {detail}"[:2500], flush=True)
    return bool(cond)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clear_env():
    for k in E27_ENVS:
        os.environ.pop(k, None)


@contextmanager
def env(**kw):
    clear_env()
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        clear_env()


def raises(fn, exc=Exception):
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return e
    return None


def outcome(fn):
    """(None, result) or (exception type name, message)."""
    try:
        return (None, fn())
    except Exception as e:  # noqa: BLE001
        return (type(e).__name__, str(e))


# ------------------------------------------------------------------------------------------------ AST helpers
def find_class(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def find_top(tree, name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return n
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            t = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if isinstance(t, ast.Name) and t.id == name:
                return n
    return None


def method(tree, cls, name):
    c = find_class(tree, cls)
    for f in c.body:
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return f
    return None


def run_nodes(nodes, ns, tag, strip_decorators=False):
    nodes = [copy.deepcopy(n) for n in nodes]
    if strip_decorators:
        for n in nodes:
            if isinstance(n, ast.FunctionDef):
                n.decorator_list = []
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, f"<{tag}>", "exec"), ns)  # noqa: S102
    return ns


def make_tree(src, dst, rels):
    for rel in rels:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(src) / rel, dst / rel)


# ------------------------------------------------------------------------------------------------ stubs
H = SimpleNamespace(pp=None, sp=None, trace=[], lock=threading.Lock(), draft=None)


def pp_group(rank, world):
    return SimpleNamespace(rank_in_group=rank, world_size=world, is_first_rank=rank == 0,
                           is_last_rank=rank == world - 1, last_rank=world - 1)


class FakeIT:
    """IntermediateTensors: .tensors, [key], [key] = v, len()."""

    def __init__(self, tensors):
        self.tensors = dict(tensors)

    def __getitem__(self, k):
        return self.tensors[k]

    def __setitem__(self, k, v):
        self.tensors[k] = v

    def __len__(self):
        return len(self.tensors)


class FakePPMissing(nn.Identity):
    pass


class FakeLoRA(nn.Module):
    pass


class _Logger:
    def info(self, *a, **k):
        pass

    warning = warning_once = info_once = debug = exception = info


def install_stubs(iface):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    for n in ("vllm", "vllm.model_executor", "vllm.model_executor.models", "vllm.lora", "vllm.lora.layers",
              "vllm.compilation", "vllm.utils", "vllm.v1", "vllm.v1.attention"):
        mod(n)
    mod("vllm.distributed", get_pp_group=lambda: H.pp)
    mod("vllm.distributed.parallel_state", get_pp_group=lambda: H.pp, model_parallel_is_initialized=lambda: True)
    mod("vllm.config", SpeculativeConfig=object, VllmConfig=object)
    mod("vllm.logger", init_logger=lambda name: _Logger())
    mod("vllm.model_executor.models.interfaces", SupportsEagle3=iface["SupportsEagle3"],
        supports_eagle3=iface["supports_eagle3"], EagleModelMixin=iface["EagleModelMixin"])
    mod("vllm.model_executor.models.utils", PPMissingLayer=FakePPMissing)
    mod("vllm.model_executor.models.qwen3_dflash", dflash_has_any_non_causal=lambda cfg: False)
    mod("vllm.lora.layers.base", BaseLayerWithLoRA=FakeLoRA)

    @contextmanager
    def set_model_tag(tag):
        yield
    mod("vllm.compilation.backends", set_model_tag=set_model_tag)
    mod("vllm.platforms", current_platform=SimpleNamespace(is_cuda=lambda: True))
    mod("vllm.utils.torch_utils", get_accelerator_view_from_cpu_tensor=lambda t: t)
    mod("vllm.v1.attention.backend", MultipleOf=MultipleOf)


class MultipleOf:
    def __init__(self, base):
        self.base = base


def load_interfaces(src):
    """The REAL SupportsEagleBase / EagleModelMixin / SupportsEagle3 / supports_eagle3 of interfaces.py."""
    tree = ast.parse((Path(src) / "vllm/model_executor/models/interfaces.py").read_text(encoding="utf-8"))
    nodes = [find_class(tree, n) for n in ("SupportsEagleBase", "EagleModelMixin", "SupportsEagle3")]
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "supports_eagle3"
           and not any(getattr(d, "id", None) == "overload" for d in n.decorator_list)]
    ns = {"Protocol": Protocol, "runtime_checkable": runtime_checkable, "ClassVar": ClassVar, "Literal": Literal,
          "TypeIs": TypeIs, "overload": overload, "bisect_right": bisect_right, "torch": torch,
          "IntermediateTensors": FakeIT, "__name__": "e27_iface"}
    return run_nodes(nodes + fns[-1:], ns, "interfaces")


def load_eagle3_utils(src):
    text = (Path(src) / "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py").read_text(encoding="utf-8")
    ns = {"__name__": "e27_eagle3_utils"}
    exec(compile(text, "eagle3_utils", "exec"), ns)  # noqa: S102
    return ns


# ------------------------------------------------------------------------------------------------ T1 / T2
def t1_patch(src, work):
    rels = list(D.PATCHES)
    for rel, (pre, *_r) in D.PATCHES.items():
        check(f"T1 source {rel} is Merlin pre-sha", sha(Path(src) / rel) == pre, sha(Path(src) / rel))
    tree = work / "tree_apply"
    make_tree(src, tree, rels)
    rec = D.apply_dflash_patch(tree)
    check("T1 apply ok", rec.get("ok") and rec["applied"] == "applied", json.dumps(rec)[:400])
    check("T1 py_compile rc 0", rec["py_compile"]["rc"] == 0, rec["py_compile"])
    patched = {}
    for rel, (pre, post, marker, patch) in D.PATCHES.items():
        data = (tree / rel).read_bytes()
        patched[rel] = data
        check(f"T1 post-sha {rel}", sha(tree / rel) == post)
        check(f"T1 marker {rel}", marker.encode() in data)
        check(f"T1 LF only {rel}", b"\r" not in data)
        check(f"T1 header shas {rel}", f"pre-patch sha256  {pre}" in patch and f"post-patch sha256 {post}" in patch)
        hunks = patch.split("\n@@ ")[1:]
        check(f"T1 every hunk carries E27 {rel}", hunks and all("E27" in h for h in hunks), len(hunks))
        try:
            ast.parse(data.decode("utf-8"), feature_version=(3, 12))
            ok = True
        except SyntaxError as e:
            ok = str(e)
        check(f"T1 ast.parse feature_version 3.12 {rel}", ok is True, ok)
    check("T1 dflash_seam.py LF only", b"\r" not in (HERE / "dflash_seam.py").read_bytes())
    check("T1 re-apply = already_applied", D.apply_dflash_patch(tree)["applied"] == "already_applied")
    rv = D.revert_dflash_patch(tree)
    check("T1 revert ok", rv.get("ok") and rv["reverted"] == "reverted")
    for rel, (pre, *_r) in D.PATCHES.items():
        check(f"T1 revert pre-sha {rel}", sha(tree / rel) == pre)
    check("T1 revert again = not_applied", D.revert_dflash_patch(tree)["reverted"] == "not_applied")
    # a modified copy
    bad = work / "tree_bad"
    make_tree(src, bad, rels)
    f = bad / D.MDL_REL
    f.write_bytes(f.read_bytes().replace(b"        last_layer = None\n", b"        last_layer = None \n", 1))
    before = {rel: sha(bad / rel) for rel in rels}
    e = raises(lambda: D.apply_dflash_patch(bad), RuntimeError)
    check("T1 refuses a modified copy", e is not None and "PRE_SHA_MISMATCH" in str(e), str(e)[:200])
    check("T1 refusal wrote nothing", before == {rel: sha(bad / rel) for rel in rels})
    e = raises(lambda: D.revert_dflash_patch(bad), RuntimeError)
    check("T1 revert refuses a modified copy", e is not None and "SHA_MISMATCH" in str(e), str(e)[:200])
    # mixed tree
    mixed = work / "tree_mixed"
    make_tree(src, mixed, rels)
    (mixed / D.CFG_REL).write_bytes(patched[D.CFG_REL])
    check("T1 verify says mixed", D.verify_dflash_patch(mixed)["state"] == "mixed", D.verify_dflash_patch(mixed)["state"])
    e = raises(lambda: D.apply_dflash_patch(mixed), RuntimeError)
    check("T1 refuses a mixed tree", e is not None and "PRE_SHA_MISMATCH" in str(e), str(e)[:200])
    # no fuzz
    text = (Path(src) / D.QSA_REL).read_text(encoding="utf-8").replace(
        "        return QSAStateBackend\n", "        return QSAStateBackend  # x\n", 1)
    check("T1 applier has no fuzz", raises(lambda: D.apply_unified(text, D.QSA_PATCH), ValueError) is not None)
    # subsets
    sub = work / "tree_sub"
    make_tree(src, sub, rels)
    r = D.apply_dflash_patch(sub, ring=False, draft_block=False)
    check("T1 core-only apply", r.get("ok") and sha(sub / D.QSA_REL) == D.QSA_PRE and sha(sub / D.ATT_REL) == D.ATT_PRE)
    check("T1 core-only tree is 'mixed' for the full set", D.verify_dflash_patch(sub)["state"] == "mixed")
    check("T1 full apply refuses a core-only tree",
          raises(lambda: D.apply_dflash_patch(sub), RuntimeError) is not None)
    r = D.revert_dflash_patch(sub, ring=False, draft_block=False)
    check("T1 core-only revert", r.get("ok") and all(sha(sub / rel) == D.PATCHES[rel][0] for rel in rels))
    # tree_state
    ts = work / "tree_state"
    make_tree(src, ts, rels)
    check("T1 tree_state ORIGINAL", D.tree_state(ts) == "ORIGINAL")
    D.apply_dflash_patch(ts)
    check("T1 tree_state FULLY_PATCHED", D.tree_state(ts) == "FULLY_PATCHED")
    (ts / D.QSA_REL).write_bytes((Path(src) / D.QSA_REL).read_bytes())
    check("T1 tree_state INVALID_MIXED (one file reverted)", D.tree_state(ts) == "INVALID_MIXED")
    check("T1 tree_state core-only view still FULLY_PATCHED", D.tree_state(ts, ring=False) == "FULLY_PATCHED")
    (ts / D.MDL_REL).write_bytes(b"# foreign\n")
    check("T1 tree_state INVALID_MIXED (a foreign file)", D.tree_state(ts, ring=False) == "INVALID_MIXED")
    return patched


def t2_ast(src, patched):
    orig = {rel: (Path(src) / rel).read_text(encoding="utf-8") for rel in D.PATCHES}
    pt = ast.parse(patched[D.MDL_REL].decode("utf-8"))
    ot = ast.parse(orig[D.MDL_REL])

    def bases(tree, cls):
        return [ast.unparse(b) for b in find_class(tree, cls).bases]
    check("T2 Qwen4ExpModel bases (nn.Module, EagleModelMixin)", bases(pt, "Qwen4ExpModel") == ["nn.Module", "EagleModelMixin"],
          bases(pt, "Qwen4ExpModel"))
    check("T2 original Qwen4ExpModel bases (nn.Module,)", bases(ot, "Qwen4ExpModel") == ["nn.Module"])
    body = find_class(pt, "Qwen4ExpModel").body
    flag = [n for n in body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "supports_aux_hidden_states_over_pp"]
    check("T2 supports_aux_hidden_states_over_pp = True", len(flag) == 1 and ast.unparse(flag[0].value) == "True")
    meths = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    want = {"_set_aux_hidden_state_layers", "_e27_dflash_tap", "_e27_check_aux", "_e27_next_hc"}
    check("T2 Qwen4ExpModel E27 methods", want <= meths, sorted(meths))
    check("T2 original has none of them", not want & {n.name for n in find_class(ot, "Qwen4ExpModel").body
                                                     if isinstance(n, ast.FunctionDef)})
    for cls in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        b = bases(pt, cls)
        check(f"T2 {cls} mounts SupportsEagle3 (last base)", b[-1] == "SupportsEagle3" and b.count("SupportsEagle3") == 1, b)
        check(f"T2 {cls} other bases unchanged", b[:-1] == bases(ot, cls), (b, bases(ot, cls)))
        check(f"T2 {cls} keeps the protocol's default setter",
              not {"set_aux_hidden_state_layers", "get_eagle3_default_aux_hidden_state_layers"}
              & {n.name for n in find_class(pt, cls).body if isinstance(n, ast.FunctionDef)})
    imp = [n for n in pt.body if isinstance(n, ast.ImportFrom) and n.module == "vllm.model_executor.models.interfaces"]
    names = {a.name for n in imp for a in n.names}
    check("T2 imports EagleModelMixin + SupportsEagle3", {"EagleModelMixin", "SupportsEagle3"} <= names, sorted(names))
    check("T2 imports grouped_gemma_rmsnorm", any(isinstance(n, ast.ImportFrom) and n.module == "ops.hc" and n.level == 1
                                                  for n in pt.body))
    dt = ast.parse(patched[D.DFL_REL].decode("utf-8"))
    c = find_class(dt, "DFlashQwen3ForCausalLM")
    hv = [n for n in c.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "has_own_embed_tokens"]
    check("T2 DFlashQwen3ForCausalLM.has_own_embed_tokens = False (explicit)", len(hv) == 1 and ast.unparse(hv[0].value) == "False")
    # top-level names: patched = original + E27 additions only
    for rel in D.PATCHES:
        o = {n.name for n in ast.parse(orig[rel]).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        p = {n.name for n in ast.parse(patched[rel].decode("utf-8")).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        extra = p - o
        check(f"T2 {rel}: no top-level def removed", o <= p, sorted(o - p))
        check(f"T2 {rel}: new top-level defs are E27 / qsa_ring helpers",
              all(n.startswith(("_e27", "qsa_ring")) for n in extra), sorted(extra))
    hct = ast.parse((Path(src) / "vllm/models/qwen4_exp/nvidia/ops/hc.py").read_text(encoding="utf-8"))
    hcf = find_top(hct, "_hc_combine")
    body = ast.unparse(hcf)
    check("T2 Merlin _hc_combine is pure: writes a fresh out = residual.new_empty(...) and returns it",
          "out = residual.new_empty(residual.shape)" in body and body.rstrip().endswith("return out")
          and not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.endswith("_")
                      and not n.func.attr.startswith("_") for n in ast.walk(hcf)), "")
    tap = method(pt, "Qwen4ExpModel", "_e27_dflash_tap")
    inplace = [ast.unparse(n) for n in ast.walk(tap) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr.endswith("_") and not n.func.attr.startswith("_")]
    check("T2 _e27_dflash_tap calls no in-place tensor op", not inplace, inplace)
    mro_mimic(src, patched)


def mro_mimic(src, patched):
    """Rebuild the real class graph from AST base lists and create the classes (C3 MRO is checked by type creation)."""
    srcs = {
        "iface": (Path(src) / "vllm/model_executor/models/interfaces.py").read_text(encoding="utf-8"),
        "base": (Path(src) / "vllm/model_executor/models/interfaces_base.py").read_text(encoding="utf-8"),
        "vl": (Path(src) / "vllm/model_executor/models/qwen3_vl.py").read_text(encoding="utf-8"),
        "q35": (Path(src) / "vllm/model_executor/models/qwen3_5.py").read_text(encoding="utf-8"),
        "q3": (Path(src) / "vllm/model_executor/models/qwen3.py").read_text(encoding="utf-8"),
    }
    common = {}
    for key in ("iface", "base", "vl", "q35", "q3"):
        for n in ast.parse(srcs[key]).body:
            if isinstance(n, ast.ClassDef):
                common.setdefault(n.name, [b for b in n.bases])

    def build(defs):
        made = {"Protocol": Protocol, "object": object}
        stack = set()

        def get(name):
            if name in made:
                return made[name]
            if name in ("nn.Module", "torch.nn.Module"):
                return nn.Module
            if name not in defs or name in stack:
                made[name] = type(name, (), {})
                return made[name]
            stack.add(name)
            bs = []
            for b in defs[name]:
                s = ast.unparse(b.value if isinstance(b, ast.Subscript) else b)
                if s == "Generic":
                    continue
                bs.append(get(s))
            stack.discard(name)
            made[name] = types.new_class(name, tuple(bs) or (object,))
            return made[name]
        return get

    mdl = {"orig": (Path(src) / D.MDL_REL).read_text(encoding="utf-8"), "patched": patched[D.MDL_REL].decode("utf-8")}
    dfl = {"orig": (Path(src) / D.DFL_REL).read_text(encoding="utf-8"), "patched": patched[D.DFL_REL].decode("utf-8")}
    for v in ("orig", "patched"):
        defs = dict(common)
        for n in ast.parse(mdl[v]).body:
            if isinstance(n, ast.ClassDef):
                defs[n.name] = list(n.bases)
        for n in ast.parse(dfl[v]).body:
            if isinstance(n, ast.ClassDef):
                defs[n.name] = list(n.bases)
        get = build(defs)
        try:
            cl = get("Qwen4ExpForCausalLM")
            cg = get("Qwen4ExpForConditionalGeneration")
            dr = get("DFlashQwen3ForCausalLM")
            ok = True
        except TypeError as e:
            ok = str(e)
        check(f"T2 MRO mimic {v}: the class graph builds", ok is True, ok)
        if ok is not True:
            continue
        se3 = get("SupportsEagle3")
        check(f"T2 MRO mimic {v}: SupportsEagle3 in the ConditionalGeneration MRO", se3 in cg.__mro__)
        check(f"T2 MRO mimic {v}: SupportsEagle3 in the CausalLM MRO == patched",
              (se3 in cl.__mro__) == (v == "patched"))
        check(f"T2 MRO mimic {v}: SupportsEagleBase in the DFlash drafter MRO (has_own_embed_tokens exists)",
              get("SupportsEagleBase") in dr.__mro__)


# ------------------------------------------------------------------------------------------------ T3 forward
HC, HD, NL, VOC = 4, 8, 48, 300
EPS = 1e-6


def combine_ref(res, block, inj, hc):
    n, d = res.shape
    r = res.float().view(n, hc, d // hc)
    b = block.float()[:, None, :]
    if inj is not None:
        b = b * (2.0 * torch.sigmoid(inj.float() / hc))[:, :, None]
    return (r + b).reshape(n, d).to(res.dtype)


def gemma_norm_ref(x, w, eps, hc):
    n, d = x.shape
    xf = x.float().view(n, hc, d // hc)
    r = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (r * (1.0 + w.float().view(hc, d // hc))).reshape(n, d).to(x.dtype)


def tr(*rec):
    with H.lock:
        H.trace.append(rec)


class FakeHC:
    def __init__(self, gen, tag):
        self.tag, self.hc_count, self.hidden_size = tag, HC, HD
        self.config = SimpleNamespace(rms_norm_eps=EPS)
        self.hc_norm = SimpleNamespace(weight=(torch.randn(HC * HD, generator=gen) * 0.1).bfloat16())
        self.g = torch.randn(HC, HD, generator=gen)

    def mix(self, hs):
        n = hs.shape[0]
        xn = gemma_norm_ref(hs, self.hc_norm.weight, EPS, HC)
        x3 = xn.float().view(n, HC, HD)
        return hs, (x3 * self.g).sum(1).to(hs.dtype), x3.mean(-1).to(hs.dtype)

    def combine(self, hs, block, inj):
        tr("combine", self.tag)
        return combine_ref(hs, block, inj, HC)

    def combine_and_mix(self, hs, block, inj):
        tr("combine_and_mix", self.tag)
        return self.mix(combine_ref(hs, block, inj, HC))


class FakeMixer(FakeHC):
    def combine_and_mix(self, hs, block, inj):
        tr("final", self.tag)
        multi = combine_ref(hs, block, inj, HC) if block is not None else hs
        return multi, self.mix(multi)[1], None


class FakeLayer:
    def __init__(self, idx):
        g = torch.Generator().manual_seed(1000 + idx)
        self.idx = idx
        self.attn_hyper_connection = FakeHC(g, f"a{idx}")
        self.mlp_hyper_connection = FakeHC(g, f"m{idx}")
        self.wa, self.ba = torch.randn(HD, generator=g), torch.randn(HD, generator=g) * 0.1
        self.wm, self.bm = torch.randn(HD, generator=g), torch.randn(HD, generator=g) * 0.1
        self.ple = None
        self.use_sp = False

    def __call__(self, hidden_states, prev_block_output, prev_injection, positions, *, input_ids, query_start_loc,
                 ngram_context):
        tr("layer", self.idx)
        if prev_block_output is not None:
            hs, bi, inj = self.attn_hyper_connection.combine_and_mix(hidden_states, prev_block_output, prev_injection)
        else:
            hs, bi, inj = self.attn_hyper_connection.mix(hidden_states)
        t = positions.shape[-1]
        if self.use_sp:
            bi = H.sp.gather(bi)[:t]
        out = torch.tanh(bi.float() * self.wa + self.ba).to(bi.dtype)
        if self.use_sp:
            out = H.sp.shard(out)
        hs, bi, inj = self.mlp_hyper_connection.combine_and_mix(hs, out, inj)
        if self.use_sp:
            bi = H.sp.gather(bi)[:t]
        mo = torch.tanh(bi.float() * self.wm + self.bm).to(bi.dtype)
        if self.use_sp:
            mo = H.sp.shard(mo)
        return hs, mo, inj


class SPSim:
    def __init__(self, tp):
        self.tp, self.bar, self.buf, self.tl = tp, threading.Barrier(tp), [None] * tp, threading.local()

    def rank(self):
        return getattr(self.tl, "rank", 0)

    def shard(self, x):
        per = -(-x.shape[0] // self.tp)
        pad = per * self.tp - x.shape[0]
        if pad:
            x = torch.cat([x, x.new_zeros((pad,) + tuple(x.shape[1:]))], 0)
        r = self.rank()
        tr("shard", r)
        return x[r * per:(r + 1) * per].contiguous()

    def gather(self, x):
        r = self.rank()
        tr("gather", r)
        self.buf[r] = x
        self.bar.wait(timeout=20)
        out = torch.cat(list(self.buf), 0)
        self.bar.wait(timeout=20)
        return out


class World:
    """Shared fake weights, inputs and the reference."""

    def __init__(self, t=7, seed=5):
        g = torch.Generator().manual_seed(seed)
        self.layers = [FakeLayer(i) for i in range(NL)]
        self.mixer = FakeMixer(torch.Generator().manual_seed(77), "final")
        self.table = (torch.randn(VOC, HD, generator=g)).bfloat16()
        self.t = t
        self.ids = torch.randint(0, VOC, (t,), generator=g)
        self.positions = torch.arange(t)
        self.ds = FakeIT({f"deepstack_input_embeds_{i}": (torch.randn(t, HD, generator=g) * 0.5).bfloat16()
                          for i in range(2)})

    def set_sp(self, on):
        for layer in self.layers:
            layer.use_sp = on

    def reference(self, ds=False):
        self.set_sp(False)
        hs = self.table[self.ids].repeat(1, HC)
        done, pre = {0: hs}, {0: hs}
        for i, layer in enumerate(self.layers):
            h1, mo, inj = layer(hs, None, None, self.positions, input_ids=self.ids, query_start_loc=None,
                                ngram_context=None)
            hs = combine_ref(h1, mo, inj, HC)
            pre[i + 1] = h1
            if ds and i < len(self.ds):
                e = self.ds[f"deepstack_input_embeds_{i}"]
                hs = hs + e.unsqueeze(-2).expand(self.t, HC, HD).flatten(-2)
                pre[i + 1] = hs
            done[i + 1] = hs
        sample = self.mixer.mix(hs)[1]
        H.trace.clear()
        return done, pre, sample

    def tap(self, ref, tap_id, mode="combine"):
        done, pre, _ = ref
        base = pre[tap_id] if mode == "pre_combine" else done[tap_id]
        if mode == "norm_input":
            hc = self.layers[tap_id].attn_hyper_connection if tap_id < NL else self.mixer
            base = gemma_norm_ref(base, hc.hc_norm.weight, EPS, HC)
        return base.unflatten(-1, (HC, HD)).mean(dim=-2)


def model_ns(iface):
    def sp_shard(x):
        return H.sp.shard(x)

    def sp_all_gather(x):
        return H.sp.gather(x)

    def grouped(x, w, eps, hc):
        tr("norm")
        return gemma_norm_ref(x, w, eps, hc)
    return {"torch": torch, "nn": nn, "os": os, "islice": __import__("itertools").islice,
            "get_pp_group": lambda: H.pp, "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
            "is_forward_context_available": lambda: False, "get_forward_context": None, "sp_padding_mask": None,
            "sp_shard": sp_shard, "sp_all_gather": sp_all_gather, "IntermediateTensors": FakeIT,
            "EagleModelMixin": iface["EagleModelMixin"], "grouped_gemma_rmsnorm": grouped, "Iterable": Iterable,
            "__name__": "e27_model"}


def build_model_cls(text, iface, tag):
    tree = ast.parse(text)
    cls = find_class(tree, "Qwen4ExpModel")
    body = []
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name not in ("__init__", "load_weights"):
            body.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            t = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if isinstance(t, ast.Name) and t.id in ("supports_aux_hidden_states_over_pp", "_e27_local_aux_ids",
                                                    "_e27_tap_mode"):
                body.append(n)
    bases = [ast.Name(id="EagleModelMixin", ctx=ast.Load())] if any(
        ast.unparse(b) == "EagleModelMixin" for b in cls.bases) else []
    new = ast.ClassDef(name="Qwen4ExpModelUT", bases=bases, keywords=[], body=copy.deepcopy(body), decorator_list=[],
                       type_params=[])
    ns = run_nodes([new], model_ns(iface), tag)
    return ns["Qwen4ExpModelUT"]


def new_model(cls, w, start, end, last, sp):
    m = cls.__new__(cls)
    m.config = SimpleNamespace(hc_count=HC, hidden_size=HD, num_hidden_layers=NL)
    m.use_hc_sequence_parallel = sp
    m.use_sequence_parallel = False
    m.start_layer, m.end_layer = start, end
    m.layers = w.layers
    m.embed_tokens = (lambda ids: w.table[ids]) if start == 0 else None
    m.hyper_connection_mixer = w.mixer if last else None
    m._mtp_hidden_buffer = None
    m.make_empty_intermediate_tensors = lambda batch_size, dtype, device: FakeIT(
        {"hidden_states": torch.zeros(batch_size, HC * HD, dtype=dtype)})
    return m


def fwd(m, w, it=None, ds=False):
    return m.forward(w.ids if m.start_layer == 0 else None, w.positions, it, None, None, None, w.ds if ds else None)


def run(m_or_pair, w, pp, it=None, ds=False, sp=False):
    """One stage. sp: run two TP-rank instances on threads (HC-SP); returns the rank-0 output and checks both agree."""
    H.pp = pp
    H.trace.clear()
    w.set_sp(sp)
    if not sp:
        return fwd(m_or_pair, w, it, ds), list(H.trace)
    H.sp = SPSim(2)
    outs, errs = [None, None], []

    def go(r):
        try:
            H.sp.tl.rank = r
            outs[r] = fwd(m_or_pair[r], w, it, ds)
        except Exception:  # noqa: BLE001
            errs.append(traceback.format_exc())
    th = [threading.Thread(target=go, args=(r,)) for r in (0, 1)]
    for x in th:
        x.start()
    for x in th:
        x.join(60)
    if errs:
        raise RuntimeError(errs[0])
    check("T3 HC-SP: both TP ranks return the same output", same(outs[0], outs[1]), "")
    return outs[0], list(H.trace)


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, FakeIT):
        return set(a.tensors) == set(b.tensors) and all(same(a.tensors[k], b.tensors[k]) for k in a.tensors)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def trace_key(t, sp):
    return Counter(t) if sp else t


class Outer:
    pass


def make_outers(iface):
    SE3 = iface["SupportsEagle3"]

    class FakeCausalLM(SE3):           # = patched Qwen4ExpForCausalLM (no get_language_model)
        def __init__(self, inner):
            self.model = inner
            self.make_empty_intermediate_tensors = inner.make_empty_intermediate_tensors

    class FakeCG(SE3):                 # = Qwen4ExpForConditionalGeneration (get_language_model -> Qwen4ExpForCausalLM)
        def __init__(self, inner):
            self.language_model = FakeCausalLM(inner)
            self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        def get_language_model(self):
            return self.language_model
    return FakeCausalLM, FakeCG


def spec_config(ids=D.DFLASH_TARGET_LAYER_IDS):
    return SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(
        dflash_config={"target_layer_ids": list(ids)})))


def set_taps(e3, outer, ids=None, aux=None):
    buf = io.StringIO()
    with redirect_stdout(buf):
        if aux is not None:
            outer.set_aux_hidden_state_layers(tuple(aux))
            e3["reserve_aux_intermediate_tensor_slots"](outer)
        else:
            e3["set_eagle3_aux_hidden_state_layers"](outer, spec_config(ids or D.DFLASH_TARGET_LAYER_IDS))
    return D.parse_log(buf.getvalue())


def t3_forward(src, patched, iface, e3):
    otext = (Path(src) / D.MDL_REL).read_text(encoding="utf-8")
    ptext = patched[D.MDL_REL].decode("utf-8")
    Orig = build_model_cls(otext, iface, "model_orig")
    Pat = build_model_cls(ptext, iface, "model_patched")
    FakeCausalLM, FakeCG = make_outers(iface)
    w = World()
    PP1, S0, S1 = pp_group(0, 1), pp_group(0, 2), pp_group(1, 2)
    BOUND = 25

    # a env-off identity
    for ds in (False, True):
        for sp in (False, True):
            ref = w.reference(ds)
            for tag, start, end, pp in (("PP1", 0, NL, PP1), ("PP2s0", 0, BOUND, S0), ("PP2s1", BOUND, NL, S1)):
                if sp and tag != "PP1":
                    continue      # HC-SP with PP > 1 is refused by config.py
                it = None
                if start > 0:
                    it = FakeIT({"hidden_states": ref[0][BOUND].clone()})
                mk = (lambda C: [new_model(C, w, start, end, end == NL, True) for _ in range(2)]) if sp else \
                    (lambda C: new_model(C, w, start, end, end == NL, False))
                oo, ot = run(mk(Orig), w, pp, it, ds, sp)
                po, pt_ = run(mk(Pat), w, pp, it, ds, sp)
                name = f"{tag} ds={ds} sp={sp}"
                check(f"T3a env-off output bitwise == original {name}", same(oo, po), "")
                check(f"T3a env-off op trace == original {name}", trace_key(ot, sp) == trace_key(pt_, sp),
                      f"{len(ot)} vs {len(pt_)}")
                if tag == "PP1":
                    check(f"T3a reference == original model {name}", same(oo, ref[2]), "")
                if tag == "PP2s0":
                    check(f"T3a env-off stage 0 keys {name}", isinstance(po, FakeIT) and set(po.tensors) == {"hidden_states"})
    # b PP1 taps via the real eagle3 helper
    ref = w.reference(False)
    with env(DFLASH_SEAM=1):
        for kind, Outer_ in (("CausalLM", FakeCausalLM), ("CG", FakeCG)):
            H.pp = PP1
            inner = new_model(Pat, w, 0, NL, True, False)
            outer = Outer_(inner)
            check(f"T3b supports_eagle3({kind})", iface["supports_eagle3"](outer))
            ev = set_taps(e3, outer)
            check(f"T3b {kind}: ids from the dflash config = {D.AUX_IDS}", inner.aux_hidden_state_layers == D.AUX_IDS,
                  inner.aux_hidden_state_layers)
            check(f"T3b {kind}: aux_layers log line", len(ev) == 1 and ev[0]["event"] == "aux_layers"
                  and tuple(ev[0]["local"]) == D.AUX_IDS and ev[0]["tap_mode"] == "combine", ev)
            out, trc = run(inner, w, PP1)
            ok = isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], list) and len(out[1]) == 5
            check(f"T3b {kind}: returns (hidden, 5 taps)", ok, type(out))
            if ok:
                check(f"T3b {kind}: hidden unchanged by the taps", same(out[0], ref[2]))
                check(f"T3b {kind}: taps [T, H] bf16", all(t.shape == (w.t, HD) and t.dtype == torch.bfloat16 for t in out[1]))
                check(f"T3b {kind}: taps bitwise = mean over hc of the completed outputs",
                      all(same(t, w.tap(ref, i)) for t, i in zip(out[1], D.AUX_IDS)),
                      [float((t.float() - w.tap(ref, i).float()).abs().max()) for t, i in zip(out[1], D.AUX_IDS)])
                check(f"T3b {kind}: the tap is NOT the un-averaged / pre-combine state",
                      not same(out[1][0], w.tap(ref, 6, "pre_combine")))
                ncomb = sum(1 for r in trc if r[0] == "combine")
                check(f"T3b {kind}: one extra pure combine per tap", ncomb == 5, ncomb)
            pp1_out = out
    # c PP2 25/23
    with env(DFLASH_SEAM=1):
        H.pp = S0
        m0 = new_model(Pat, w, 0, BOUND, False, False)
        o0 = FakeCausalLM(m0)
        ev0 = set_taps(e3, o0)
        H.pp = S1
        m1 = new_model(Pat, w, BOUND, NL, True, False)
        o1 = FakeCausalLM(m1)
        ev1 = set_taps(e3, o1)
        check("T3c stage 0 local taps (6, 16), slot base 0", m0._e27_local_aux_ids == (6, 16) and m0._aux_slot_base_cached == 0,
              (m0._e27_local_aux_ids, m0._aux_slot_base_cached))
        check("T3c stage 1 local taps (26, 36, 46), slot base 2 = bisect_right(ids, 25), upstream total 2",
              m1._e27_local_aux_ids == (26, 36, 46) and m1._aux_slot_base_cached == 2 == bisect_right(D.AUX_IDS, 25)
              and m1._aux_upstream_total_cached == 2,
              (m1._e27_local_aux_ids, m1._aux_slot_base_cached, m1._aux_upstream_total_cached))
        check("T3c log lines (slot_base / upstream_total)", ev0[0]["slot_base"] == 0 and ev1[0]["slot_base"] == 2
              and ev1[0]["upstream_total"] == 2)
        # ported from vLLM PR #61180 test_qwen3_5_declares_and_configures_aux_hidden_state_pp: the outer's setter with
        # unsorted / duplicate ids routes through EagleModelMixin._set_aux_hidden_state_layers (sorted unique ids +
        # the PP slot metadata), on both outer shapes
        for kind, Outer_ in (("CausalLM", FakeCausalLM), ("CG", FakeCG)):
            H.pp = S1
            mx = new_model(Pat, w, BOUND, NL, True, False)
            set_taps(e3, Outer_(mx), aux=(46, 6, 26, 16, 36, 6))
            check(f"T3c PR61180 {kind}: unsorted / duplicate ids -> sorted unique, slot base 2, upstream total 2",
                  mx.aux_hidden_state_layers == D.AUX_IDS and mx._aux_slot_base_cached == 2
                  and mx._aux_upstream_total_cached == 2 and mx._e27_local_aux_ids == (26, 36, 46),
                  (mx.aux_hidden_state_layers, mx._aux_slot_base_cached, mx._aux_upstream_total_cached))
        H.pp = S1
        e3["verify_supports_aux_hidden_states_over_pp"](o1, "dflash")
        check("T3c verify_supports_aux_hidden_states_over_pp passes (patched)", True)
        check("T3c relay keys for PP2 = ()", e3["aux_hidden_state_relay_keys"](o1) == ())
        out0, _ = run(m0, w, S0)
        check("T3c stage 0 IntermediateTensors keys", isinstance(out0, FakeIT)
              and set(out0.tensors) == {"hidden_states", "aux_hidden_states_0", "aux_hidden_states_1"},
              sorted(getattr(out0, "tensors", {})))
        check("T3c stage 0 aux values = taps 6, 16", same(out0["aux_hidden_states_0"], w.tap(ref, 6))
              and same(out0["aux_hidden_states_1"], w.tap(ref, 16)))
        check("T3c stage 0 hidden = the materialized state after layer 24", same(out0["hidden_states"], ref[0][BOUND]))
        buf = o1.make_empty_intermediate_tensors(batch_size=16, dtype=torch.bfloat16, device="cpu")
        check("T3c stage 1 reserved keys == stage 0 sent keys", set(buf.tensors) == set(out0.tensors), sorted(buf.tensors))
        check("T3c reserved aux slots [B, 2560-like H]", all(buf[k].shape == (16, HD) for k in buf.tensors if k.startswith("aux")))
        n = w.t
        recv = FakeIT({k: v[:n].copy_(out0.tensors[k][:n]) for k, v in buf.tensors.items()})   # model_runner.py:2449-2455
        out1, _ = run(m1, w, S1, recv)
        ok = isinstance(out1, tuple) and len(out1[1]) == 5
        check("T3c stage 1 returns (hidden, 5 taps)", ok)
        if ok:
            check("T3c stage 1 order REMOTE FIRST: taps bitwise = [6, 16, 26, 36, 46]",
                  all(same(t, w.tap(ref, i)) for t, i in zip(out1[1], D.AUX_IDS)))
            check("T3c PP2 taps == PP1 taps, hidden == PP1 hidden", same(out1[1], pp1_out[1]) and same(out1[0], pp1_out[0]))
            perm = out1[1][2:] + out1[1][:2]
            check("T3c a permuted order would be detected (taps pairwise distinct)", not same(perm, pp1_out[1])
                  and len({t.float().sum().item() for t in out1[1]}) == 5)
        # missing relay key: Merlin's collect raises
        bad = FakeIT({k: v for k, v in recv.tensors.items() if k != "aux_hidden_states_1"})
        e = raises(lambda: run(m1, w, S1, bad))
        check("T3h missing relay key raises RuntimeError 'Missing aux_hidden_states_1' (PR #61180 case)",
              isinstance(e, RuntimeError) and "Missing aux_hidden_states_1" in str(e), repr(e))
    # verify fails for an inner without the flag (the original seam / roozich)
    class NoFlag(iface["EagleModelMixin"]):
        pass
    o_nf = FakeCausalLM(new_model(Pat, w, BOUND, NL, True, False))
    o_nf.model = NoFlag()
    check("T3c verify raises without supports_aux_hidden_states_over_pp",
          raises(lambda: e3["verify_supports_aux_hidden_states_over_pp"](o_nf, "dflash"), ValueError) is not None)
    # d HC-SP PP1 with taps
    with env(DFLASH_SEAM=1):
        H.pp = PP1
        pair = [new_model(Pat, w, 0, NL, True, True) for _ in range(2)]
        for m in pair:
            set_taps(e3, FakeCausalLM(m))
        out, trc = run(pair, w, PP1, sp=True)
        check("T3d HC-SP taps = reference (full token axis)", isinstance(out, tuple)
              and all(same(t, w.tap(ref, i)) for t, i in zip(out[1], D.AUX_IDS)), "")
        offpair = [new_model(Pat, w, 0, NL, True, True) for _ in range(2)]
        _, trc0 = run(offpair, w, PP1, sp=True)
        g1 = sum(1 for r in trc if r[0] == "gather")
        g0 = sum(1 for r in trc0 if r[0] == "gather")
        check("T3d HC-SP: one extra gather per tap per rank", g1 - g0 == 2 * 5, (g1, g0))
    # e boundary ids over PP2, f deepstack
    for ds in (False, True):
        refd = w.reference(ds)
        ids = (0, 1, 2, 25, 26, 48)
        with env(DFLASH_SEAM=1):
            H.pp = PP1
            mp = new_model(Pat, w, 0, NL, True, False)
            set_taps(e3, FakeCausalLM(mp), aux=ids)
            outp, _ = run(mp, w, PP1, ds=ds)
            H.pp = S0
            m0 = new_model(Pat, w, 0, BOUND, False, False)
            set_taps(e3, FakeCausalLM(m0), aux=ids)
            H.pp = S1
            m1 = new_model(Pat, w, BOUND, NL, True, False)
            o1 = FakeCausalLM(m1)
            set_taps(e3, o1, aux=ids)
            check(f"T3e ds={ds} stage locals (0, 1, 2, 25) / (26, 48)", m0._e27_local_aux_ids == (0, 1, 2, 25)
                  and m1._e27_local_aux_ids == (26, 48) and m1._aux_slot_base_cached == 4)
            out0, _ = run(m0, w, S0, ds=ds)
            buf = o1.make_empty_intermediate_tensors(batch_size=16, dtype=torch.bfloat16, device="cpu")
            recv = FakeIT({k: v[:w.t].copy_(out0.tensors[k][:w.t]) for k, v in buf.tensors.items()})
            out1, _ = run(m1, w, S1, recv, ds=ds)
            check(f"T3e/f ds={ds} PP1 taps = reference (id 0 = embedding, 48 = after the last layer)",
                  all(same(t, w.tap(refd, i)) for t, i in zip(outp[1], ids)))
            check(f"T3e/f ds={ds} PP2 taps == PP1 taps", same(out1[1], outp[1]))
            check(f"T3e id 0 tap == the embedding rows ds={ds}", same(outp[1][0], w.table[w.ids]))
    # g tap modes
    ref = w.reference(False)
    for mode in ("pre_combine", "norm_input"):
        with env(DFLASH_SEAM=1, DFLASH_TAP_MODE=mode):
            H.pp = PP1
            m = new_model(Pat, w, 0, NL, True, False)
            ev = set_taps(e3, FakeCausalLM(m), aux=(6, 16, 26, 36, 46, 48))
            out, _ = run(m, w, PP1)
            check(f"T3g tap mode {mode} = its reference", ev[0]["tap_mode"] == mode
                  and all(same(t, w.tap(ref, i, mode)) for t, i in zip(out[1], (6, 16, 26, 36, 46, 48))))
            check(f"T3g tap mode {mode} differs from combine", not same(out[1][0], w.tap(ref, 6)))
    with env(DFLASH_SEAM=1, DFLASH_TAP_MODE="norm_input"):
        H.pp = S0
        m0 = new_model(Pat, w, 0, BOUND, False, False)
        e = raises(lambda: set_taps(e3, FakeCausalLM(m0), aux=(6, 25)), ValueError)
        check("T3g norm_input refuses a tap whose next layer is on the next PP stage", e is not None and "not on this PP stage" in str(e), e)
        H.pp = PP1
        m = new_model(Pat, w, 0, NL, True, False)
        w.layers[16].ple = object()
        try:
            e = raises(lambda: set_taps(e3, FakeCausalLM(m), aux=(16,)), ValueError)
        finally:
            w.layers[16].ple = None
        check("T3g norm_input refuses a PLE next layer", e is not None and "PLE" in str(e), e)
    with env(DFLASH_SEAM=1, DFLASH_TAP_MODE="bogus"):
        H.pp = PP1
        e = raises(lambda: set_taps(e3, FakeCausalLM(new_model(Pat, w, 0, NL, True, False)), aux=(6,)), ValueError)
        check("T3g an unknown tap mode raises", e is not None and "DFLASH_TAP_MODE" in str(e), e)
    # h fail loud
    H.pp = PP1
    clear_env()
    e = raises(lambda: set_taps(e3, FakeCausalLM(new_model(Pat, w, 0, NL, True, False))), RuntimeError)
    check("T3h setter without DFLASH_SEAM=1 raises", e is not None and "DFLASH_SEAM=1" in str(e), e)
    with env(DFLASH_SEAM=1):
        for bad_ids, label in (((49,), "out of range"), ((), "empty"), ((-1, 6), "negative")):
            e = raises(lambda: set_taps(e3, FakeCausalLM(new_model(Pat, w, 0, NL, True, False)), aux=bad_ids), ValueError)
            check(f"T3h {label} tap ids raise", e is not None, e)
        m = new_model(Pat, w, 0, NL, True, False)
        set_taps(e3, FakeCausalLM(m))
        m._e27_local_aux_ids = (6, 16, 26, 36)
        e = raises(lambda: run(m, w, PP1), RuntimeError)
        check("T3h a tap-count mismatch raises DFLASH_AUX_COUNT_MISMATCH", e is not None and "DFLASH_AUX_COUNT_MISMATCH" in str(e), e)
        H.pp = S1
        m1 = new_model(Pat, w, BOUND, NL, True, False)
        set_taps(e3, FakeCausalLM(m1))
        m1._aux_upstream_total_cached = 1        # a short relay
        it = FakeIT({"hidden_states": ref[0][BOUND], "aux_hidden_states_0": w.tap(ref, 6)})
        e = raises(lambda: run(m1, w, S1, it), RuntimeError)
        check("T3h a short upstream list raises DFLASH_AUX_COUNT_MISMATCH", e is not None
              and "DFLASH_AUX_COUNT_MISMATCH" in str(e) and "upstream" in str(e), e)
        H.pp = S1
        m1 = new_model(Pat, w, BOUND, NL, True, False)
        set_taps(e3, FakeCausalLM(m1))
        it = FakeIT({"hidden_states": ref[0][BOUND], "aux_hidden_states_0": w.tap(ref, 6),
                     "aux_hidden_states_1": torch.zeros(w.t, HD + 1, dtype=torch.bfloat16)})
        e = raises(lambda: run(m1, w, S1, it), RuntimeError)
        check("T3h a remote tap of the wrong width raises DFLASH_AUX_WIDTH_MISMATCH", e is not None
              and "DFLASH_AUX_WIDTH_MISMATCH" in str(e), e)
    t3_inputs_and_numeric(Pat, w, iface)


def t3_inputs_and_numeric(Pat, w, iface):
    """T3i: the REAL _e27_dflash_tap on the real Flash-Next shapes (hc 4, H 2560): inputs unmutated, finite, and equal to
    an independent per-stream reference of combine(...).unflatten(-1, (4, 2560)).mean(-2); HC-SP via 2 threads."""
    hc, hd, t = 4, 2560, 37
    g = torch.Generator().manual_seed(9)
    res = (torch.randn(t, hc * hd, generator=g)).bfloat16()
    blk = (torch.randn(t, hd, generator=g)).bfloat16()
    inj = (torch.randn(t, hc, generator=g)).bfloat16()

    def reference(r, b, i):          # per stream k: res[:, k*H:(k+1)*H] + b * 2 sigmoid(i[:, k] / hc), then mean over k
        acc = []
        for k in range(hc):
            s = r[:, k * hd:(k + 1) * hd].float() + b.float() * (2 * torch.sigmoid(i[:, k:k + 1].float() / hc)
                                                                if i is not None else 1.0)
            acc.append(s.to(torch.bfloat16).float())
        return torch.stack(acc, 0).mean(0)

    class Mhc:
        def combine(self, a, b, c):
            return combine_ref(a, b, c, hc)
    layer = SimpleNamespace(mlp_hyper_connection=Mhc())
    m = Pat.__new__(Pat)
    m.config = SimpleNamespace(hc_count=hc, hidden_size=hd, num_hidden_layers=NL)
    m.use_hc_sequence_parallel = False
    m._e27_tap_mode = "combine"
    snap = (res.clone(), blk.clone(), inj.clone())
    for label, b_, i_ in (("with injection", blk, inj), ("unit injection", blk, None)):
        tap = m._e27_dflash_tap(layer, res, b_, i_, t, 6)
        diff = float((tap.float() - reference(res, b_, i_)).abs().max())
        check(f"T3i real shapes {label}: tap [T, 2560] bf16 finite, = per-stream reference (max |d| {diff:.2e})",
              tap.shape == (t, hd) and tap.dtype == torch.bfloat16 and bool(torch.isfinite(tap).all()) and diff <= 2e-2, diff)
    check("T3i the tap leaves hidden_states / block_output / injection unmutated",
          torch.equal(res, snap[0]) and torch.equal(blk, snap[1]) and torch.equal(inj, snap[2]))
    # HC-SP: each TP rank holds a row shard, the tap is gathered to the full token axis
    H.sp = SPSim(2)
    m.use_hc_sequence_parallel = True
    per = -(-t // 2)
    outs, errs = [None, None], []

    def go(r):
        try:
            H.sp.tl.rank = r
            pad = lambda x: torch.cat([x, x.new_zeros((per * 2 - t,) + tuple(x.shape[1:]))], 0)  # noqa: E731
            outs[r] = m._e27_dflash_tap(layer, pad(res)[r * per:(r + 1) * per], pad(blk)[r * per:(r + 1) * per],
                                        pad(inj)[r * per:(r + 1) * per], t, 6)
        except Exception:  # noqa: BLE001
            errs.append(traceback.format_exc())
    th = [threading.Thread(target=go, args=(r,)) for r in (0, 1)]
    for x in th:
        x.start()
    for x in th:
        x.join(60)
    full = m.__class__._e27_dflash_tap
    m.use_hc_sequence_parallel = False
    want = full(m, layer, res, blk, inj, t, 6)
    check("T3i HC-SP: both ranks' gathered taps = the unsharded tap (full token axis)", not errs and outs[0] is not None
          and same(outs[0], want) and same(outs[1], want), errs[:1])
    # the pure combine in Merlin's kernel wrapper allocates its output (ops/hc.py _hc_combine)
    return None


# ------------------------------------------------------------------------------------------------ T4 config.py
def t4_config(src, patched):
    texts = {"orig": (Path(src) / D.CFG_REL).read_text(encoding="utf-8"), "patched": patched[D.CFG_REL].decode("utf-8")}

    class Base:
        @staticmethod
        def verify_and_update_config(vc):
            return None
    fns = {}
    for k, t in texts.items():
        node = method(ast.parse(t), "Qwen4ExpForConditionalGenerationConfig", "verify_and_update_config")
        ns = {"Qwen3_5ForConditionalGenerationConfig": Base, "logger": _Logger(), "_strip_qwen4_exp_mrope": lambda mc: None,
              "__name__": f"cfg_{k}"}
        fns[k] = run_nodes([node], ns, f"cfg_{k}", strip_decorators=True)["verify_and_update_config"]

    def vc(method_):
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hc_count=4, ple_layer_ids=[], indexer_n_heads=None),
                                         multimodal_config=None),
            parallel_config=SimpleNamespace(tensor_parallel_size=2, enable_hc_sp=False, use_sequence_parallel_moe=False,
                                            pipeline_parallel_size=2, enable_dbo=False, ubatch_size=1),
            speculative_config=None if method_ is None else SimpleNamespace(method=method_))
    methods = (None, "mtp", "ngram", "ngram_gpu", "dflash", "eagle3", "dspark", "eagle")
    with env():
        for m in methods:
            o, p = outcome(lambda: fns["orig"](vc(m))), outcome(lambda: fns["patched"](vc(m)))
            check(f"T4 env off: {m} identical", o == p, (o, p))
        check("T4 env off: dflash refused", outcome(lambda: fns["patched"](vc("dflash")))[0] == "NotImplementedError")
    with env(DFLASH_SEAM=1):
        for m in methods:
            o, p = outcome(lambda: fns["orig"](vc(m))), outcome(lambda: fns["patched"](vc(m)))
            if m == "dflash":
                check("T4 DFLASH_SEAM=1 admits dflash", p == (None, None) and o[0] == "NotImplementedError", (o, p))
            elif o[0] is None:
                check(f"T4 DFLASH_SEAM=1: {m} still accepted", p == o, (o, p))
            else:
                check(f"T4 DFLASH_SEAM=1: {m} still refused (message + suffix)", p[0] == o[0] and p[1].startswith(o[1])
                      and "DFLASH_SEAM=1" in p[1], (o, p))


# ------------------------------------------------------------------------------------------------ T5 qsa ring
def t5_ring(src, patched):
    texts = {"orig": (Path(src) / D.QSA_REL).read_text(encoding="utf-8"), "patched": patched[D.QSA_REL].decode("utf-8")}
    fns = {}
    for k, t in texts.items():
        tree = ast.parse(t)
        nodes = [method(tree, "QSAKeyStateCache", "get_kv_cache_spec")]
        if k == "patched":
            nodes = [find_top(tree, n) for n in ("QSA_RING_MAX_WIDENING", "qsa_ring_widen_enabled",
                                                  "qsa_ring_capacity")] + nodes
        ns = {"cdiv": lambda a, b: -(-a // b), "CircularBufferSpec": lambda **kw: kw, "KVCacheSpec": object,
              "VllmConfig": object, "__name__": f"qsa_{k}"}
        got = run_nodes(nodes, ns, f"qsa_{k}")
        fns[k] = got["get_kv_cache_spec"]
        if k == "patched":
            fns["qrc"] = got["qsa_ring_capacity"]

    def call(fn, ratio, k, block):
        self = SimpleNamespace(compress_ratio=ratio, cache_config=SimpleNamespace(block_size=block), head_size=128,
                               dtype=torch.bfloat16)
        buf = io.StringIO()
        with redirect_stdout(buf):
            r = outcome(lambda: fn(self, SimpleNamespace(num_speculative_tokens=k))["block_size"])
        return r, D.parse_log(buf.getvalue())
    fns = {k: v for k, v in fns.items()}
    blocks = (12, 16, 24, 32, 48, 64, 96, 128, 256, 848, 1024, 1616, 2048, 4096, 8192)
    n_same = n_wide = n_err = 0
    with env():
        for k in range(13):
            for b in blocks:
                o, p = call(fns["orig"], 4, k, b)[0], call(fns["patched"], 4, k, b)[0]
                n_same += o == p
                if o != p:
                    check(f"T5 env off K{k} block {b} identical", False, (o, p))
    check("T5 env off: 195 cases identical (capacity or the same AssertionError)", n_same == 13 * len(blocks), n_same)
    with env(DFLASH_QSA_RING_WIDEN=1):
        for k in range(13):
            for b in blocks:
                o = call(fns["orig"], 4, k, b)[0]
                p, ev = call(fns["patched"], 4, k, b)
                if o[0] is None:
                    check(f"T5 widen K{k} b{b}: unchanged when the original passes", p == o and not ev, (o, p))
                    continue
                minimal = 4 * -(-(4 + k) // 4)
                cands = [c for c in range(minimal + 4, min(b, 2 * minimal) + 1, 4) if b % c == 0]
                if cands:
                    n_wide += 1
                    c = cands[0]
                    check(f"T5 widen K{k} b{b} -> {c}", p == (None, c) and c >= 4 + k and b % c == 0 and len(ev) == 1
                          and ev[0]["capacity"] == c, (p, ev))
                else:
                    n_err += 1
                    check(f"T5 widen K{k} b{b}: ValueError", p[0] == "ValueError", p)
        for k in (3, 4):
            check(f"T5 K{k} @ 4096 -> 8 (no widening)", call(fns["patched"], 4, k, 4096)[0] == (None, 8))
        for k in (5, 6, 7, 8):
            check(f"T5 K{k} @ 4096 -> 16", call(fns["patched"], 4, k, 4096)[0] == (None, 16))
    with env():
        check("T5 env off: K7 @ 4096 asserts (as Merlin)", call(fns["patched"], 4, 7, 4096)[0][0] == "AssertionError")
    qrc = fns["qrc"]
    for k, want in ((3, 8), (4, 8), (5, 16), (7, 16)):
        buf = io.StringIO()
        with redirect_stdout(buf):
            got = qrc(4, k, 4096)
        orig = 4 * -(-(4 + k) // 4)
        check(f"T5 qsa_ring_capacity(4, K{k}, 4096) = {want}" + (" (= the original formula)" if k < 5 else ""),
              got == want and (k >= 5 or got == orig) and bool(D.parse_log(buf.getvalue())) == (k >= 5), got)
    e = raises(lambda: qrc(4, 7, 4124), ValueError)        # 4124 = 4 x 1031: no multiple of 4 in 12..24 divides it
    check("T5 qsa_ring_capacity: informative ValueError for an impossible size", e is not None
          and "num_speculative_tokens=7" in str(e) and "4124" in str(e), e)
    check("T5 coverage: widened and refused cases both exercised", n_wide > 5 and n_err > 0, (n_wide, n_err))


# ------------------------------------------------------------------------------------------------ T6 qwen3_dflash load
V6, H6 = 64, 8


class Loader:
    last = None

    def __init__(self, module):
        self.module = module
        Loader.last = self

    def load_weights(self, weights, mapper=None):
        self.items = [(n, w) for n, w in weights]
        self.mapper = dict(mapper.orig_to_new_substr)
        params = dict(self.module.named_parameters())
        loaded = set()
        for n, w in self.items:
            if any(s in n for s, v in self.mapper.items() if v is None):
                continue
            if n in params:
                params[n].data.copy_(w)
                loaded.add(n)
        return loaded


def t6_load(src, patched, work):
    texts = {"orig": (Path(src) / D.DFL_REL).read_text(encoding="utf-8"), "patched": patched[D.DFL_REL].decode("utf-8")}
    utree = ast.parse((Path(src) / "vllm/model_executor/models/utils.py").read_text(encoding="utf-8"))
    pew = run_nodes([find_top(utree, "process_eagle_weight")], {"supports_any_eagle": lambda m: True, "nn": nn,
                                                               "__name__": "pew"}, "pew")["process_eagle_weight"]
    fns = {}
    for k, t in texts.items():
        tree = ast.parse(t)
        nodes = [method(tree, "DFlashQwen3ForCausalLM", "load_weights")]
        if k == "patched":
            nodes = [find_top(tree, n) for n in ("_E27_EMBED_FLOAT_DTYPES", "_e27_log", "_e27_mark_embed_loaded",
                                                  "_e27_check_loaded", "_e27_with_embed_sidecar",
                                                  "_e27_embed_sidecar_iter", "_e27_read_embed_sidecar")] + nodes
        ns = {"torch": torch, "nn": nn, "Iterable": Iterable, "AutoWeightsLoader": Loader,
              "WeightsMapper": lambda orig_to_new_substr: SimpleNamespace(orig_to_new_substr=dict(orig_to_new_substr)),
              "process_eagle_weight": pew, "__name__": f"dfl_{k}"}
        fns[k] = run_nodes(nodes, ns, f"dfl_{k}")
    E = (torch.randn(V6, H6) * 0.3).bfloat16()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_w = nn.Parameter(torch.zeros(4, H6), requires_grad=False)
            self.embed_tokens = nn.Module()
            self.embed_tokens.weight = nn.Parameter(torch.zeros(V6, H6, dtype=torch.bfloat16), requires_grad=False)
            self.fc = nn.Module()
            self.fc.weight = nn.Parameter(torch.zeros(H6, 5 * H6), requires_grad=False)
            self.use_aux_hidden_state = True
            self.has_separate_mask_embedding = False
            self.mask_token_id = 7

        def _build_fused_kv_buffers(self):
            pass

    class Draft(nn.Module):
        has_own_embed_tokens = False
        has_own_lm_head = False

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=V6, hidden_size=H6)
            self.model = Inner()
            self.lm_head = nn.Module()
            self.lm_head.weight = nn.Parameter(torch.zeros(V6, H6), requires_grad=False)

        def _read_mask_embedding(self):
            return None

    def ckpt(embed=False, drop_layer=False):
        ws = [("layer_w", torch.ones(4, H6)), ("fc.weight", torch.ones(H6, 5 * H6))]
        if drop_layer:
            ws = ws[1:]
        if embed:
            ws.append(("embed_tokens.weight", E.clone()))
        return ws

    def load(k, ws, pp_world=2):
        H.pp = pp_group(1, pp_world) if pp_world > 1 else pp_group(0, 1)
        d = Draft()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r = outcome(lambda: fns[k]["load_weights"](d, iter(ws)))
        L = Loader.last
        rec = {"out": r, "names": [n for n, _ in L.items] if L else None, "mapper": L.mapper if L else None,
               "has_own": d.has_own_embed_tokens, "loaded_flag": getattr(d, "_e27_embed_loaded", None),
               "embed": d.model.embed_tokens.weight.detach().clone(), "log": D.parse_log(buf.getvalue())}
        Loader.last = None
        return rec

    def same_rec(a, b):
        return (a["out"][0] == b["out"][0] and a["names"] == b["names"] and a["mapper"] == b["mapper"]
                and a["has_own"] == b["has_own"] and torch.equal(a["embed"], b["embed"]))
    # files
    from safetensors.torch import save_file
    sd = work / "side"
    sd.mkdir()
    other = torch.randn(3, 3)
    save_file({"model.language_model.embed_tokens.weight": E, "model.language_model.layers.0.w": other,
               "mtp.embed_tokens.weight": torch.zeros(V6, H6, dtype=torch.bfloat16)}, str(sd / "single.safetensors"))
    (sd / "idx").mkdir()
    save_file({"model.language_model.embed_tokens.weight": E}, str(sd / "idx" / "model-00002-of-00002.safetensors"))
    save_file({"model.language_model.layers.0.w": other}, str(sd / "idx" / "model-00001-of-00002.safetensors"))
    (sd / "idx" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.language_model.embed_tokens.weight": "model-00002-of-00002.safetensors",
        "model.language_model.layers.0.w": "model-00001-of-00002.safetensors"}}))
    (sd / "plain").mkdir()
    save_file({"a.embed_tokens.weight": E}, str(sd / "plain" / "x.safetensors"))
    save_file({"b.w": other}, str(sd / "plain" / "y.safetensors"))
    save_file({"model.embed_tokens.weight": E, "model.language_model.embed_tokens.weight": E.clone()},
              str(sd / "ambig.safetensors"))
    save_file({"model.embed_tokens.weight": E[:10]}, str(sd / "shape.safetensors"))
    save_file({"model.embed_tokens.weight": torch.ones(V6, H6, dtype=torch.int32)}, str(sd / "int.safetensors"))
    save_file({"model.embed_tokens.weight": torch.zeros(V6, H6, dtype=torch.bfloat16)}, str(sd / "zero.safetensors"))
    nan = E.clone()
    nan[0, 0] = float("nan")
    save_file({"model.embed_tokens.weight": nan}, str(sd / "nan.safetensors"))
    # env off: identical, even with every env but DFLASH_SEAM set
    for embed in (False, True):
        for extra in ({}, {"DFLASH_EMBED_SIDECAR": str(sd / "single.safetensors")}):
            with env(**extra):
                a, b = load("orig", ckpt(embed)), load("patched", ckpt(embed))
                check(f"T6 env off identical embed={embed} extra={sorted(extra)}", same_rec(a, b) and b["loaded_flag"] is None
                      and not b["log"], (a["names"], b["names"], b["log"]))
    with env(DFLASH_SEAM=1):
        a, b = load("orig", ckpt(True)), load("patched", ckpt(True))
        check("T6 repacked drafter: loaded as model.embed_tokens.weight, kept (not mapped away)",
              b["out"][0] is None and "model.embed_tokens.weight" in b["names"] and "embed_tokens" not in b["mapper"]
              and torch.equal(b["embed"], E), (b["out"], b["mapper"]))
        check("T6 repacked drafter: has_own_embed_tokens True, _e27_embed_loaded True", b["has_own"] is True and b["loaded_flag"] is True)
        check("T6 repacked drafter: same loader input as the original", same_rec(a, b))
        lc = [e for e in b["log"] if e["event"] == "draft_load_check"]
        check("T6 load check logged (expected == loaded)", len(lc) == 1 and lc[0]["expected"] == lc[0]["loaded"] == 3, lc)
        b = load("patched", ckpt(False))
        check("T6 no embed: mapped away, _e27_embed_loaded False, load check passes", b["out"][0] is None
              and b["mapper"].get("embed_tokens", 1) is None and b["loaded_flag"] is False, b)
        b = load("patched", ckpt(True, drop_layer=True))
        check("T6 a missing drafter tensor raises (strict check)", b["out"][0] == "RuntimeError" and "layer_w" in b["out"][1], b["out"])
        a = load("orig", ckpt(True, drop_layer=True))
        check("T6 negative control: the original loads silently with a missing tensor", a["out"][0] is None)
    with env(DFLASH_SEAM=1, DFLASH_LOAD_CHECK=0):
        b = load("patched", ckpt(True, drop_layer=True))
        check("T6 DFLASH_LOAD_CHECK=0 skips the check", b["out"][0] is None and any(e.get("skipped") for e in b["log"]))
    # sidecar fallback
    cases = [("single file", str(sd / "single.safetensors"), {}, True),
             ("index dir", str(sd / "idx"), {}, True),
             ("plain dir", str(sd / "plain"), {}, True),
             ("explicit key", str(sd / "ambig.safetensors"), {"DFLASH_EMBED_KEY": "model.embed_tokens.weight"}, True)]
    for label, path, extra, _ in cases:
        with env(DFLASH_SEAM=1, DFLASH_EMBED_SIDECAR=path, **extra):
            b = load("patched", ckpt(False))
            ev = [e for e in b["log"] if e["event"] == "embed_sidecar"]
            check(f"T6 sidecar {label}: loaded, own, flag True", b["out"][0] is None and torch.equal(b["embed"], E)
                  and b["has_own"] is True and b["loaded_flag"] is True and b["names"][-1] == "model.embed_tokens.weight"
                  and len(ev) == 1 and ev[0]["shape"] == [V6, H6], (b["out"], ev))
    bad = [("ambiguous keys", str(sd / "ambig.safetensors"), {}, "KeyError"),
           ("missing key", str(sd / "single.safetensors"), {"DFLASH_EMBED_KEY": "nope"}, "KeyError"),
           ("wrong shape", str(sd / "shape.safetensors"), {}, "ValueError"),
           ("int dtype", str(sd / "int.safetensors"), {}, "TypeError"),
           ("all zero", str(sd / "zero.safetensors"), {}, "ValueError"),
           ("NaN", str(sd / "nan.safetensors"), {}, "ValueError"),
           ("missing path", str(sd / "nope.safetensors"), {}, "FileNotFoundError")]
    for label, path, extra, exc in bad:
        with env(DFLASH_SEAM=1, DFLASH_EMBED_SIDECAR=path, **extra):
            b = load("patched", ckpt(False))
            check(f"T6 sidecar {label} raises {exc}", b["out"][0] == exc, b["out"])
    with env(DFLASH_SEAM=1, DFLASH_EMBED_SIDECAR=str(sd / "single.safetensors")):
        b = load("patched", ckpt(True))
        check("T6 sidecar + a checkpoint that ships embed_tokens raises", b["out"][0] == "ValueError", b["out"])
        a, b = load("orig", ckpt(False), pp_world=1), load("patched", ckpt(False), pp_world=1)
        check("T6 sidecar at PP1: skipped (logged), loader input = original", same_rec(a, b)
              and any(e["event"] == "embed_sidecar_skipped" for e in b["log"]), b["log"])
    # embed_row_sha
    r = 5
    want = hashlib.sha256(E[r].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    for label, path in (("file", sd / "single.safetensors"), ("index dir", sd / "idx"), ("plain dir", sd / "plain")):
        got = D.embed_row_sha(path, row=r)
        check(f"T6 embed_row_sha {label}", got[0] == want and got[3] == "torch.bfloat16", got)
    check("T6 embed_row_sha ambiguous raises", raises(lambda: D.embed_row_sha(sd / "ambig.safetensors", row=r), KeyError) is not None)


# ------------------------------------------------------------------------------------------------ T7 load_dflash_model
V7, H7 = 40, 6


class UnquantizedEmbeddingMethod:
    pass


class UnquantizedLinearMethod:
    pass


class Fp8LinearMethod:
    pass


class Emb(nn.Module):
    def __init__(self, w, start=0, end=None):
        super().__init__()
        self.weight = nn.Parameter(w.clone(), requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()
        self.shard_indices = SimpleNamespace(org_vocab_start_index=start,
                                             org_vocab_end_index=w.shape[0] + start if end is None else end)


class QKVParallelLinear(nn.Module):
    def __init__(self, qm):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, H7), requires_grad=False)
        self.quant_method = qm


def t7_finish(src, patched):
    texts = {"orig": (Path(src) / D.DFU_REL).read_text(encoding="utf-8"), "patched": patched[D.DFU_REL].decode("utf-8")}
    etree = ast.parse((Path(src) / "vllm/v1/worker/gpu/spec_decode/eagle/utils.py").read_text(encoding="utf-8"))
    eag = run_nodes([find_top(etree, n) for n in ("_should_share", "get_target_lm_head", "maybe_share_target_embed")],
                    {"torch": torch, "nn": nn, "get_pp_group": lambda: H.pp, "PPMissingLayer": FakePPMissing,
                     "BaseLayerWithLoRA": FakeLoRA, "__name__": "eagle_utils"}, "eagle_utils")
    fns = {}
    for k, t in texts.items():
        tree = ast.parse(t)
        nodes = [find_top(tree, "load_dflash_model")]
        if k == "patched":
            nodes = [find_top(tree, "_e27_release_module"), find_top(tree, "_e27_finish_draft")] + nodes

        def rep(obj, **kw):
            d = dict(vars(obj))
            d.update(kw)
            return SimpleNamespace(**d)
        ns = {"torch": torch, "nn": nn, "VllmConfig": object, "replace": rep, "get_model": lambda **kw: H.draft,
              "get_draft_load_config": lambda vc: vc.load_config, "get_pp_safe_draft_load_config": lambda c: c,
              "_should_share": eag["_should_share"], "get_target_lm_head": eag["get_target_lm_head"],
              "maybe_share_target_embed": eag["maybe_share_target_embed"], "__name__": f"dfu_{k}"}
        fns[k] = run_nodes(nodes, ns, f"dfu_{k}")["load_dflash_model"]
    vcfg = SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=None),
                                                              attention_backend=None, kv_cache_dtype=None),
                           attention_config=SimpleNamespace(use_non_causal=False, backend=None),
                           cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"), load_config=SimpleNamespace())
    TW = (torch.randn(V7, H7) * 0.2).bfloat16()

    class Inner(nn.Module):
        def __init__(self, w, qm, start=0, end=None):
            super().__init__()
            self.embed_tokens = Emb(w, start, end)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].self_attn = nn.Module()
            self.layers[0].self_attn.qkv_proj = QKVParallelLinear(qm)
            self.mask_token_id = 30

    def make_draft(w=None, own=False, loaded=None, qm=None, no_attr=False, start=0, end=None, class_own=False):
        attrs = {} if no_attr else {"has_own_embed_tokens": class_own, "has_own_lm_head": False}

        def init(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(has_own_lm_head=False)
            self.model = Inner(TW if w is None else w, qm or UnquantizedLinearMethod(), start, end)
            self.lm_head = nn.Module()
            self.lm_head.weight = nn.Parameter(torch.ones(V7, H7), requires_grad=False)
            if own:
                self.has_own_embed_tokens = True
            if loaded is not None:
                self._e27_embed_loaded = loaded
        cls = type("DFlash2Qwen3ForCausalLM", (nn.Module,), dict(attrs, __init__=init))
        return cls()

    def make_target(pp1):
        t = nn.Module()
        t.model = nn.Module()
        t.model.embed_tokens = Emb(TW) if pp1 else FakePPMissing()
        t.lm_head = nn.Module()
        t.lm_head.weight = nn.Parameter(torch.full((V7, H7), 2.0), requires_grad=False)
        return t

    def go(k, draft, target, pp1=False, **e):
        H.pp = pp_group(0, 1) if pp1 else pp_group(1, 2)
        H.draft = draft
        old_head = draft.lm_head
        old_embed = draft.model.embed_tokens
        buf = io.StringIO()
        with env(**e), redirect_stdout(buf):
            r = outcome(lambda: fns[k](target, vcfg))
        return {"out": r, "draft": draft, "log": D.parse_log(buf.getvalue()), "old_head": old_head,
                "old_embed": old_embed}

    def res_key(x):
        d = x["draft"]
        return (x["out"][0], x["out"][1] if x["out"][0] else None, d.model.embed_tokens is x["old_embed"],
                getattr(d, "lm_head", None) is x["old_head"])
    # S1 PP2 without an embedding: Merlin raises (both); env off identical
    for e_ in ({}, {"DFLASH_SEAM": 1}):
        a, b = go("orig", make_draft(), make_target(False), **e_), go("patched", make_draft(), make_target(False), **e_)
        check(f"T7 S1 PP2 no embedding: Merlin raises 'needs the target input embedding' (orig + patched, env {e_})",
              a["out"][0] == b["out"][0] == "RuntimeError" and "needs the target input embedding" in b["out"][1], (a["out"], b["out"]))
    # S2 repacked (own + loaded)
    tgt = make_target(False)
    b = go("patched", make_draft(own=True, loaded=True), tgt, DFLASH_SEAM=1)
    ev = [x for x in b["log"] if x["event"] == "embed"]
    check("T7 S2 repacked PP2: ok, embed own, head shared", b["out"][0] is None and len(ev) == 1
          and ev[0]["embed_source"] == "own" and ev[0]["head_source"] == "shared_target"
          and b["draft"].lm_head is tgt.lm_head, (b["out"], ev))
    check("T7 S2 the replaced draft lm_head is released (B5)", len(b["old_head"]._parameters) == 0
          and ev and ev[0]["released_bytes"]["lm_head"] == V7 * H7 * 4 and ev[0]["released_bytes"]["embed_tokens"] == 0, ev)
    a = go("orig", make_draft(own=True, loaded=True), make_target(False))
    check("T7 S2 negative control: the original keeps the orphaned head's parameters", len(a["old_head"]._parameters) == 1)
    a, b2 = go("orig", make_draft(own=True, loaded=True), make_target(False)), go("patched", make_draft(own=True, loaded=True), make_target(False))
    check("T7 S2 env off identical (no release, no log)", res_key(a) == res_key(b2) and not b2["log"]
          and len(b2["old_head"]._parameters) == 1)
    # mask row
    row = TW[30].contiguous()
    want = hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).hexdigest()
    mr = ev[0]["mask_row"] if ev else {}
    check("T7 mask row 30: sha + norm logged on the rank holding it", mr.get("on_this_rank") and mr.get("sha256") == want, mr)
    b = go("patched", make_draft(own=True, loaded=True), make_target(False), DFLASH_SEAM=1, DFLASH_MASK_ROW_SHA=want)
    check("T7 DFLASH_MASK_ROW_SHA match: ok", b["out"][0] is None, b["out"])
    b = go("patched", make_draft(own=True, loaded=True), make_target(False), DFLASH_SEAM=1, DFLASH_MASK_ROW_SHA="00" * 32)
    check("T7 DFLASH_MASK_ROW_SHA mismatch raises", b["out"][0] == "RuntimeError" and "mask_token_id" in b["out"][1], b["out"])
    shard = TW[:20]
    b = go("patched", make_draft(w=shard, own=True, loaded=True, start=0, end=20), make_target(False), DFLASH_SEAM=1,
           DFLASH_MASK_ROW_SHA="00" * 32)
    ev = [x for x in b["log"] if x["event"] == "embed"]
    check("T7 mask row on the other TP shard: not on this rank, no assert", b["out"][0] is None and ev
          and ev[0]["mask_row"]["on_this_rank"] is False, (b["out"], ev))
    b = go("patched", make_draft(w=TW[20:], own=True, loaded=True, start=20, end=40), make_target(False), DFLASH_SEAM=1)
    ev = [x for x in b["log"] if x["event"] == "embed"]
    check("T7 mask row on shard [20, 40): local index 10", ev and ev[0]["mask_row"].get("sha256") == want, ev)
    # S3 / S4 zero / NaN, not loaded, quantized
    for label, w_ in (("all-zero", torch.zeros(V7, H7, dtype=torch.bfloat16)), ("NaN", TW.clone().index_fill_(0, torch.tensor([0]), float("nan")))):
        b = go("patched", make_draft(w=w_, own=True, loaded=True), make_target(False), DFLASH_SEAM=1)
        a = go("orig", make_draft(w=w_, own=True, loaded=True), make_target(False))
        check(f"T7 {label} embed raises (patched), original silent", b["out"][0] == "RuntimeError" and a["out"][0] is None, (b["out"], a["out"]))
    b = go("patched", make_draft(own=True, loaded=False, class_own=True), make_target(False), DFLASH_SEAM=1)
    check("T7 has_own True but _e27_embed_loaded False raises", b["out"][0] == "RuntimeError" and "neither shared" in b["out"][1], b["out"])
    b = go("patched", make_draft(own=True, loaded=True, qm=Fp8LinearMethod()), make_target(False), DFLASH_SEAM=1)
    check("T7 a quantized drafter linear raises (UnquantizedLinearMethod required)", b["out"][0] == "RuntimeError"
          and "UnquantizedLinearMethod" in b["out"][1], b["out"])
    # S5 PP1 shared
    tgt = make_target(True)
    b = go("patched", make_draft(), tgt, pp1=True, DFLASH_SEAM=1)
    ev = [x for x in b["log"] if x["event"] == "embed"]
    check("T7 S5 PP1: embed + head shared from the target", b["out"][0] is None and ev
          and ev[0]["embed_source"] == "shared_target" and b["draft"].model.embed_tokens is tgt.model.embed_tokens, (b["out"], ev))
    check("T7 S5 PP1: the replaced draft embed_tokens and lm_head released", ev
          and ev[0]["released_bytes"]["embed_tokens"] == V7 * H7 * 2 and ev[0]["released_bytes"]["lm_head"] == V7 * H7 * 4
          and len(b["old_embed"]._parameters) == 0 and len(b["old_head"]._parameters) == 0, ev)
    check("T7 S5 PP1: the shared target modules are untouched", len(tgt.model.embed_tokens._parameters) == 1
          and len(tgt.lm_head._parameters) == 1)
    # S6 the silent early return: a drafter class without has_own_embed_tokens
    a = go("orig", make_draft(no_attr=True), make_target(False))
    b = go("patched", make_draft(no_attr=True), make_target(False), DFLASH_SEAM=1)
    check("T7 S6 no-attribute drafter under PP2: the original returns silently (garbage), patched raises",
          a["out"][0] is None and b["out"][0] == "RuntimeError" and "uninitialised" in b["out"][1], (a["out"], b["out"]))
    # S7 head missing: no target head
    tgt = make_target(False)
    del tgt.lm_head
    b = go("patched", make_draft(own=True, loaded=True), tgt, DFLASH_SEAM=1)
    check("T7 S7 lm_head neither shared nor own raises", b["out"][0] == "RuntimeError" and "lm_head" in b["out"][1], b["out"])
    # host embed guards (CPU: the UVA path itself needs CUDA, kit / GPU)
    b = go("patched", make_draft(own=True, loaded=True), make_target(False), DFLASH_SEAM=1, DFLASH_DRAFT_EMBED_HOST=1)
    check("T7 DFLASH_DRAFT_EMBED_HOST on a CPU weight raises (needs CUDA)", b["out"][0] == "RuntimeError"
          and "DFLASH_DRAFT_EMBED_HOST" in b["out"][1], b["out"])
    b = go("patched", make_draft(), make_target(True), pp1=True, DFLASH_SEAM=1, DFLASH_DRAFT_EMBED_HOST=1)
    ev = [x for x in b["log"] if x["event"] == "embed"]
    check("T7 DFLASH_DRAFT_EMBED_HOST leaves a shared (target) embedding alone", b["out"][0] is None and ev and ev[0]["host"] is False)
    # env off identical across scenarios
    for label, mk in (("PP2 own", lambda: (make_draft(own=True, loaded=True), make_target(False), False)),
                      ("PP1", lambda: (make_draft(), make_target(True), True)),
                      ("zero", lambda: (make_draft(w=torch.zeros(V7, H7, dtype=torch.bfloat16), own=True), make_target(False), False)),
                      ("no-attr", lambda: (make_draft(no_attr=True), make_target(False), False))):
        d1, t1, p1 = mk()
        d2, t2, p2 = mk()
        a, b = go("orig", d1, t1, pp1=p1), go("patched", d2, t2, pp1=p2, DFLASH_DRAFT_EMBED_HOST=1, DFLASH_MASK_ROW_SHA="0")
        check(f"T7 env off identical: {label}", res_key(a) == res_key(b) and not b["log"], (res_key(a), res_key(b)))


# ------------------------------------------------------------------------------------------------ T8 attention.py
def t8_attention(src, patched):
    texts = {"orig": (Path(src) / D.ATT_REL).read_text(encoding="utf-8"), "patched": patched[D.ATT_REL].decode("utf-8")}

    class SWSpec:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        @property
        def real_page_size_bytes(self):
            return self.block_size * 2 * self.num_kv_heads * self.head_size * 2
    AT = SimpleNamespace(ENCODER_ONLY="eo", ENCODER="e", DECODER="d")
    fns = {}
    for k, t in texts.items():
        tree = ast.parse(t)
        nodes = [find_top(tree, "_largest_kernel_block_within"), method(tree, "Attention", "get_kv_cache_spec")]
        if k == "patched":
            nodes = [find_top(tree, "_E27_DRAFT_SW_LOGGED"), find_top(tree, "_e27_draft_sw_block")] + nodes
        ns = {"AttentionType": AT, "get_kv_quant_mode": lambda d: None, "SlidingWindowSpec": SWSpec,
              "FullAttentionSpec": SWSpec, "KVCacheSpec": object, "VllmConfig": object, "nn": nn,
              "__name__": f"att_{k}"}
        fns[k] = run_nodes(nodes, ns, f"att_{k}")["get_kv_cache_spec"]
    backend = SimpleNamespace(is_mla=lambda: False, customize_spec=lambda s: s,
                              get_supported_kernel_block_sizes=lambda spec: [16, 32, 64, MultipleOf(16)])

    def layer(draft, sw=2048):
        return SimpleNamespace(attn_type="d", kv_cache_dtype="auto", sliding_window=sw, attn_backend=backend,
                               num_kv_heads=4, head_size=128, head_size_v=128, kv_cache_torch_dtype=torch.bfloat16,
                               is_draft_layer=draft, layer_name=f"drafter.{draft}.{sw}")
    vc = SimpleNamespace(cache_config=SimpleNamespace(block_size=4096, skip_page_size_padded=None))

    def call(k, lay):
        buf = io.StringIO()
        with redirect_stdout(buf):
            r = outcome(lambda: vars(fns[k](lay, vc)))
        return r, D.parse_log(buf.getvalue())
    for e_ in ({}, {"DFLASH_SEAM": 1}):
        with env(**e_):
            for draft in (False, True):
                for sw in (2048, None):
                    a, b = call("orig", layer(draft, sw))[0], call("patched", layer(draft, sw))[0]
                    check(f"T8 env off identical draft={draft} sw={sw} {sorted(e_)}", a == b, (a, b))
    with env(DFLASH_DRAFT_SW_BLOCK=1024):
        a = call("orig", layer(True))[0]
        b, ev = call("patched", layer(True))
        check("T8 DFLASH_DRAFT_SW_BLOCK=1024: drafter SW block 4096 -> 1024", a[1]["block_size"] == 4096
              and b[1]["block_size"] == 1024 and ev and ev[0]["draft_block"] == 1024, (a, b, ev))
        check("T8 a target SW layer keeps its block", call("patched", layer(False))[0] == call("orig", layer(False))[0])
        check("T8 a full-attention drafter layer keeps its block", call("patched", layer(True, None))[0] == call("orig", layer(True, None))[0])
    with env(DFLASH_DRAFT_SW_BLOCK=1000):
        r, _ = call("patched", layer(True))
        check("T8 a non-divisor raises", r[0] == "ValueError", r)


# ------------------------------------------------------------------------------------------------ T9 readback
def t9_readback():
    def line(**kw):
        return "(Worker pid=1) E27_DFLASH " + json.dumps(kw)
    aux = []
    for pid, (s, e, loc, base, up) in zip((11, 12, 21, 22), ((0, 25, [6, 16], 0, 0),) * 2 + ((25, 48, [26, 36, 46], 2, 2),) * 2):
        aux.append(line(event="aux_layers", pid=pid, ids=list(D.AUX_IDS), local=loc, start_layer=s, end_layer=e,
                        slot_base=base, upstream_total=up, tap_mode="combine"))
    emb = [line(event="embed", pid=21, embed_source="own", head_source="shared_target", host=False,
                mask_row={"id": 248077, "on_this_rank": False}),
           line(event="embed", pid=22, embed_source="own", head_source="shared_target", host=False,
                mask_row={"id": 248077, "on_this_rank": True, "sha256": "ab" * 32})]
    chk = [line(event="draft_load_check", pid=p, expected=81, loaded=81) for p in (21, 22)]
    chk += [line(event="ingraph_meta", pid=p, supports_decode_metadata_update=True, forced_off=False,
                 kernel_map_refresh=True) for p in (21, 22)]
    good = aux + emb + chk
    lab, d = D.readback_label("\n".join(good))
    check("T9 ENGAGED", lab == "ENGAGED", d)
    lab, d = D.readback_label("\n".join(good), mask_row_sha="ab" * 32)
    check("T9 ENGAGED with the mask-row sha", lab == "ENGAGED", d)
    lab, d = D.readback_label("\n".join(good), mask_row_sha="cd" * 32)
    check("T9 mask-row sha mismatch -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(aux[:3] + emb + chk))
    check("T9 a missing worker -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(good), tap_mode="norm_input")
    check("T9 wrong tap mode -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(good), sidecar=True)
    check("T9 sidecar expected but absent -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(good), embed_host=True)
    check("T9 host expected but off -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(aux + emb))
    check("T9 no load-check lines -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(aux + emb + chk[:2]))
    check("T9 no ingraph_meta lines -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    stale = chk[:2] + [line(event="ingraph_meta", pid=p, supports_decode_metadata_update=True, forced_off=False,
                            kernel_map_refresh=False) for p in (21, 22)]
    lab, d = D.readback_label("\n".join(aux + emb + stale))
    check("T9 in-graph update without the kernel-map refresh (#63587) -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    eager = chk[:2] + [line(event="ingraph_meta", pid=p, supports_decode_metadata_update=False, forced_off=True,
                            kernel_map_refresh=False) for p in (21, 22)]
    lab, d = D.readback_label("\n".join(aux + emb + eager), no_ingraph_meta=True)
    check("T9 DFLASH_NO_INGRAPH_META arm -> ENGAGED", lab == "ENGAGED", d)
    lab, d = D.readback_label("\n".join(aux + emb + eager))
    check("T9 forced_off lines on a normal arm -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(good), no_ingraph_meta=True)
    check("T9 no_ingraph_meta requested, in-graph still on -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(good), ring_k=7)
    check("T9 ring expected but absent -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    lab, d = D.readback_label("\n".join(good + [line(event="qsa_ring", num_speculative_tokens=7, capacity=16)]), ring_k=7)
    check("T9 ring K7 -> ENGAGED", lab == "ENGAGED", d)
    bad_local = [x.replace("[26, 36, 46]", "[26, 36]") for x in aux]
    lab, d = D.readback_label("\n".join(bad_local + emb + chk))
    check("T9 a missing local tap -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", d)
    check("T9 server_env", D.server_env() == {"DFLASH_SEAM": "1"} and D.server_env(ring=True, tap_mode="pre_combine",
          mask_row_sha="AB", embed_host=True, draft_sw_block=1024) == {
        "DFLASH_SEAM": "1", "DFLASH_QSA_RING_WIDEN": "1", "DFLASH_TAP_MODE": "pre_combine", "DFLASH_MASK_ROW_SHA": "ab",
        "DFLASH_DRAFT_EMBED_HOST": "1", "DFLASH_DRAFT_SW_BLOCK": "1024"}, D.server_env(ring=True))
    check("T9 server_env no_ingraph_meta", D.server_env(no_ingraph_meta=True) == {"DFLASH_SEAM": "1",
                                                                                "DFLASH_NO_INGRAPH_META": "1"})
    check("T9 server_env conv_impl", D.server_env(conv_impl="torch") == {"DFLASH_SEAM": "1", "DFLASH_CONV_IMPL": "torch"}
          and raises(lambda: D.server_env(conv_impl="x"), ValueError) is not None)
    fused = [line(event="conv_impl", pid=p, impl="fused") for p in (21, 22)]
    torch_l = [line(event="conv_impl", pid=p, impl="torch") for p in (21, 22)]
    check("T9 conv_impl fused lines on the default arm -> ENGAGED", D.readback_label("\n".join(good + fused))[0] == "ENGAGED")
    check("T9 conv_impl torch arm -> ENGAGED", D.readback_label("\n".join(good + torch_l), conv_impl="torch")[0] == "ENGAGED")
    check("T9 conv_impl torch requested, fused lines -> NOT_INTEGRATED",
          D.readback_label("\n".join(good + fused), conv_impl="torch")[0] == "NOT_INTEGRATED")
    check("T9 conv_impl torch requested, no lines -> NOT_INTEGRATED",
          D.readback_label("\n".join(good), conv_impl="torch")[0] == "NOT_INTEGRATED")
    check("T9 server_env refuses an unknown tap mode", raises(lambda: D.server_env(tap_mode="x"), ValueError) is not None)


# ------------------------------------------------------------------------------------------------ T10 nb4 composition
def t10_compose(src, work, nb4_dir):
    spec = importlib.util.spec_from_file_location("nb4_ppack_e27t", str(Path(nb4_dir) / "nb4_ppack.py"))
    nb4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb4)
    check("T10 nb4 files == dflash_seam.NB4_FILES", set(nb4.PATCHES) == set(D.NB4_FILES), sorted(nb4.PATCHES))
    check("T10 disjoint file sets", not set(nb4.PATCHES) & set(D.PATCHES))
    rels = list(nb4.PATCHES) + list(D.PATCHES)
    finals = {}
    for order in ("nb4_first", "e27_first"):
        tree = work / f"compose_{order}"
        make_tree(src, tree, rels)
        steps = [nb4.apply_ppack_patch, D.apply_dflash_patch]
        if order == "e27_first":
            steps.reverse()
        for s in steps:
            r = s(tree)
            check(f"T10 {order}: {s.__name__} ok", r.get("ok"), r.get("after"))
        check(f"T10 {order}: both verify applied", nb4.verify_ppack_patch(tree)["state"] == "applied"
              and D.verify_dflash_patch(tree)["state"] == "applied")
        finals[order] = {rel: sha(tree / rel) for rel in rels}
        mr = (tree / nb4.MR_REL).read_text(encoding="utf-8")
        check(f"T10 {order}: the runner's aux relay / setter lines survive NB4",
              "return self.pp_handler.relay_aux_hidden_states(" in mr
              and "set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)" in mr
              and "self.pp_handler.configure_aux_hidden_state_relay(self.model)" in mr)
        D.revert_dflash_patch(tree)
        check(f"T10 {order}: E27 reverted, NB4 still applied", D.verify_dflash_patch(tree)["state"] == "not_applied"
              and nb4.verify_ppack_patch(tree)["state"] == "applied")
        D.apply_dflash_patch(tree)
        nb4.revert_ppack_patch(tree)
        check(f"T10 {order}: NB4 reverted, E27 still applied", nb4.verify_ppack_patch(tree)["state"] == "not_applied"
              and D.verify_dflash_patch(tree)["state"] == "applied")
        D.revert_dflash_patch(tree)
        check(f"T10 {order}: all {len(rels)} files back to Merlin", all(sha(tree / rel) == (nb4.PATCHES.get(rel) or D.PATCHES[rel])[0]
                                                            for rel in rels))
    check(f"T10 both orders give the same {len(rels)} file shas", finals["nb4_first"] == finals["e27_first"])
    tree = work / "compose_states"
    make_tree(src, tree, rels)
    st = [D.combined_tree_state(tree, nb4)]
    D.apply_dflash_patch(tree)
    st.append(D.combined_tree_state(tree, nb4))
    nb4.apply_ppack_patch(tree)
    st.append(D.combined_tree_state(tree, nb4))
    (tree / nb4.MR_REL).write_bytes((Path(src) / nb4.MR_REL).read_bytes())
    st.append(D.combined_tree_state(tree, nb4))
    check("T10 combined_tree_state: ORIGINAL/ORIGINAL, FULLY/ORIGINAL, FULLY/FULLY valid; NB4 mixed -> invalid",
          [(s["e27"], s["nb4"], s["valid"]) for s in st] == [
              ("ORIGINAL", "ORIGINAL", True), ("FULLY_PATCHED", "ORIGINAL", True),
              ("FULLY_PATCHED", "FULLY_PATCHED", True), ("FULLY_PATCHED", "INVALID_MIXED", False)], st)
    check("T10 combined state = each module's post-sha", all(
        finals["nb4_first"][rel] == (nb4.PATCHES.get(rel) or D.PATCHES[rel])[1] for rel in rels))


# ------------------------------------------------------------------------------------------------ T11 speculator
def t11_ingraph(src, patched):
    otext = (Path(src) / D.SPC_REL).read_text(encoding="utf-8")
    ptext = patched[D.SPC_REL].decode("utf-8")
    ot, pt = ast.parse(otext), ast.parse(ptext)
    pc = find_class(pt, "DFlashSpeculator")
    dflt = [n for n in pc.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_e27_kernel_map_refresh"]
    check("T11 class default _e27_kernel_map_refresh = False", len(dflt) == 1 and ast.unparse(dflt[0].value) == "False")
    sa = method(pt, "DFlashSpeculator", "set_attn")
    calls = [ast.unparse(s) for s in sa.body if isinstance(s, ast.Expr)]
    check("T11 set_attn: super().set_attn(...) first, self._e27_configure_ingraph_meta() last",
          calls and calls[0].startswith("super().set_attn(") and calls[-1] == "self._e27_configure_ingraph_meta()"
          and ast.unparse(sa.body[-1]) == "self._e27_configure_ingraph_meta()", calls)

    def branch_if(tree):
        fn = method(tree, "DFlashSpeculator", "propose")
        for n in ast.walk(fn):
            if isinstance(n, ast.If) and ast.unparse(n.test) == "not (is_full_cudagraph and self.supports_decode_metadata_update)":
                return fn, n
        return fn, None
    ofn, oif = branch_if(ot)
    pfn, pif = branch_if(pt)
    check("T11 original: the skipped build has no else (the #63587 bug)", oif is not None and not oif.orelse)
    ok = (pif is not None and len(pif.orelse) == 1 and isinstance(pif.orelse[0], ast.If)
          and ast.unparse(pif.orelse[0].test) == "self._e27_kernel_map_refresh" and not pif.orelse[0].orelse)
    check("T11 patched: elif self._e27_kernel_map_refresh right after the skipped build", ok,
          ast.unparse(pif) if pif else None)
    rf = [n.lineno for n in ast.walk(pfn) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("run_fullgraph")]
    check("T11 the refresh runs before run_fullgraph", ok and rf and pif.orelse[0].lineno < min(rf), rf)
    body = ast.unparse(pif.orelse[0]) if ok else ""
    check("T11 the refresh maps self.block_tables.input_block_tables[group.kv_cache_group_id] for every group",
          "for groups in self.attn_groups" in body and "for group in groups" in body
          and "group.map_to_kernel_block_table(self.block_tables.input_block_tables[group.kv_cache_group_id])" in body)
    o_other = [ast.unparse(s) for s in ofn.body]
    p_other = [ast.unparse(s) for s in pfn.body]
    check("T11 propose: every other statement unchanged", len(o_other) == len(p_other) and all(
        a == b for a, b in zip(o_other, p_other) if "_build_uniform_attn_metadata" not in a), "")

    # the REAL configure helper
    cfg = run_nodes([method(pt, "DFlashSpeculator", "_e27_configure_ingraph_meta")], {"__name__": "spc"}, "spc_cfg")[
        "_e27_configure_ingraph_meta"]

    class Spec:
        _speculator_name = "DFlash2"
        _e27_kernel_map_refresh = False

        def __init__(self, native):
            self.supports_decode_metadata_update = native
            self.attn_groups = [[], [SimpleNamespace(kv_cache_group_id=1, kv_cache_spec=SimpleNamespace(block_size=4096),
                                                     kernel_block_size=64)]]

    def conf(native, **e):
        s = Spec(native)
        buf = io.StringIO()
        with env(**e), redirect_stdout(buf):
            cfg(s)
        return s, D.parse_log(buf.getvalue())
    for e_ in ({}, {"DFLASH_NO_INGRAPH_META": 1}):
        s, ev = conf(True, **e_)
        check(f"T11 env off ({sorted(e_)}): flags untouched, no log", s.supports_decode_metadata_update is True
              and "_e27_kernel_map_refresh" not in vars(s) and not ev)
    s, ev = conf(True, DFLASH_SEAM=1)
    check("T11 DFLASH_SEAM=1, in-graph on: kernel_map_refresh True, one ingraph_meta line", s._e27_kernel_map_refresh is True
          and s.supports_decode_metadata_update is True and len(ev) == 1 and ev[0]["kernel_map_refresh"] is True
          and ev[0]["groups"] == [{"gid": 1, "manager_block": 4096, "kernel_block": 64}], ev)
    s, ev = conf(False, DFLASH_SEAM=1)
    check("T11 DFLASH_SEAM=1, backend without in-graph update: no refresh needed", s._e27_kernel_map_refresh is False
          and ev[0]["native_supports_decode_metadata_update"] is False)
    s, ev = conf(True, DFLASH_SEAM=1, DFLASH_NO_INGRAPH_META=1)
    check("T11 DFLASH_NO_INGRAPH_META=1: in-graph update forced off (eager rebuild every step)",
          s.supports_decode_metadata_update is False and s._e27_kernel_map_refresh is False and ev[0]["forced_off"] is True
          and ev[0]["native_supports_decode_metadata_update"] is True, ev)

    # the REAL elif body + the REAL AttentionGroup.map_to_kernel_block_table
    utree = ast.parse((Path(src) / "vllm/v1/worker/utils.py").read_text(encoding="utf-8"))
    mk = run_nodes([method(utree, "AttentionGroup", "map_to_kernel_block_table")], {"torch": torch, "__name__": "ag"},
                   "ag")["map_to_kernel_block_table"]
    fn = ast.FunctionDef(name="refresh", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], vararg=None,
                                                            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                         body=copy.deepcopy(pif.orelse[0].body), decorator_list=[], returns=None, type_params=[])
    refresh = run_nodes([fn], {"__name__": "refresh"}, "refresh")["refresh"]

    class Group:
        map_to_kernel_block_table = mk

        def __init__(self, gid, kernel, stride=0, rows=4, cols=3):
            self.kv_cache_group_id = gid
            self.kv_cache_spec = SimpleNamespace(block_size=4096)
            self.kernel_block_size = kernel
            self.kernel_block_stride = stride
            if kernel is None:
                self.kernel_block_table = None
                self.kernel_block_offsets = None
            else:
                self.kernel_block_table = torch.full((1, rows, cols * (4096 // kernel)), -1, dtype=torch.int32)
                self.kernel_block_offsets = torch.arange(4096 // kernel, dtype=torch.int32)
    rows, cols = 4, 3
    tables = [torch.zeros(rows, cols, dtype=torch.int32) for _ in range(3)]
    groups = [Group(0, None), Group(1, 64), Group(2, 64, stride=128)]
    spec = SimpleNamespace(attn_groups=[[groups[0]], [groups[1], groups[2]]],
                           block_tables=SimpleNamespace(input_block_tables=tables))
    for t in tables:                     # capture time: the eager build maps manager ids 1, 2
        t[0, :2] = torch.tensor([1, 2], dtype=torch.int32)
    for g in groups:
        g.map_to_kernel_block_table(tables[g.kv_cache_group_id])
    captured = groups[1].kernel_block_table.clone()
    for t in tables:                     # replay time: the request now owns manager blocks 3, 5
        t[0, :2] = torch.tensor([3, 5], dtype=torch.int32)
    ar = torch.arange(64, dtype=torch.int32)
    check("T11 negative control: without the refresh the kernel table still holds the capture-time ids 1, 2 (stale KV)",
          torch.equal(groups[1].kernel_block_table[0, 0, :64], 1 * 64 + ar)
          and torch.equal(groups[1].kernel_block_table[0, 0, 64:128], 2 * 64 + ar))
    refresh(spec)
    kt = groups[1].kernel_block_table[0, 0]
    check("T11 refresh: manager id 3 -> kernel ids 192..255, id 5 -> 320..383 (block 4096 / kernel 64)",
          torch.equal(kt[:64], 3 * 64 + ar) and torch.equal(kt[64:128], 5 * 64 + ar), kt[:4].tolist())
    check("T11 refresh: the other rows and the trailing column follow the current table (id 0 -> 0..63)",
          torch.equal(groups[1].kernel_block_table[0, 1, :64], ar) and torch.equal(kt[128:192], ar)
          and not torch.equal(groups[1].kernel_block_table, captured))
    kt2 = groups[2].kernel_block_table[0, 0]
    check("T11 refresh with a packed kernel stride 128: id 3 -> 384..447", torch.equal(kt2[:64], 3 * 128 + ar), kt2[:4].tolist())
    check("T11 a group without kernel blocks is left alone", groups[0].kernel_block_table is None)


# ------------------------------------------------------------------------------------------------ T12 grouped conv
def t12_conv(src, patched):
    import torch.nn.functional as F
    otext = (Path(src) / D.DF2_REL).read_text(encoding="utf-8")
    ptext = patched[D.DF2_REL].decode("utf-8")
    ot, pt = ast.parse(otext), ast.parse(ptext)
    og, pg = find_top(ot, "_grouped_conv"), find_top(pt, "_grouped_conv")
    first = pg.body[0]
    check("T12 _grouped_conv: `if _E27_CONV_TORCH:` first, returning _e27_grouped_conv_torch(...)",
          isinstance(first, ast.If) and ast.unparse(first.test) == "_E27_CONV_TORCH"
          and ast.unparse(first.body[-1]).startswith("return _e27_grouped_conv_torch(") and not first.orelse)
    check("T12 _grouped_conv: the rest of the body is Merlin's", [ast.unparse(s) for s in pg.body[1:]]
          == [ast.unparse(s) for s in og.body] and ast.unparse(pg.args) == ast.unparse(og.args))
    kern = ast.unparse(find_top(ot, "_dflash2_grouped_conv_kernel"))
    check("T12 Merlin's fused kernel loads the tap rows under tap_mask (SGLang #41736 does not apply to it)",
          "x_ptr + (row - tap) * x_stride_row + channels, mask=tap_mask, other=0.0" in kern, "")
    # env gating
    en = run_nodes([find_top(pt, "_e27_conv_torch_enabled")], {"__name__": "df2e"}, "df2e")["_e27_conv_torch_enabled"]

    def gate(**e):
        buf = io.StringIO()
        with env(**e), redirect_stdout(buf):
            r = outcome(en)
        return r, D.parse_log(buf.getvalue())
    for e_, want, logged in (({}, False, None), ({"DFLASH_CONV_IMPL": "torch"}, False, None),
                             ({"DFLASH_SEAM": 1}, False, "fused"), ({"DFLASH_SEAM": 1, "DFLASH_CONV_IMPL": "fused"}, False, "fused"),
                             ({"DFLASH_SEAM": 1, "DFLASH_CONV_IMPL": "torch"}, True, "torch")):
        r, ev = gate(**e_)
        check(f"T12 gate {sorted(e_)} -> {want}, log {logged}", r == (None, want)
              and ([x.get("impl") for x in ev] == ([logged] if logged else [])), (r, ev))
    r, _ = gate(DFLASH_SEAM=1, DFLASH_CONV_IMPL="bogus")
    check("T12 gate: an unknown DFLASH_CONV_IMPL raises", r[0] == "ValueError", r)
    # numerics: the REAL E27 torch path vs Merlin's torch fallback and the #55960 reference
    calls = []

    def fused(*a):
        calls.append(a)
        raise AssertionError("the fused op must not run here")
    base_ns = {"torch": torch, "F": F, "dflash2_grouped_conv": fused, "__name__": "df2"}
    e27 = run_nodes([find_top(pt, "_e27_grouped_conv_torch")], dict(base_ns), "df2t")["_e27_grouped_conv_torch"]
    orig = run_nodes([og], dict(base_ns), "df2o")["_grouped_conv"]
    pat_on = run_nodes([find_top(pt, "_e27_grouped_conv_torch"), pg], dict(base_ns, _E27_CONV_TORCH=True), "df2p1")["_grouped_conv"]
    pat_off = run_nodes([find_top(pt, "_e27_grouped_conv_torch"), pg], dict(base_ns, _E27_CONV_TORCH=False), "df2p0")["_grouped_conv"]
    groups, gsz, taps = 160, 16, 2
    for blk in (4, 5, 8):
        g = torch.Generator().manual_seed(blk)
        batch = 6
        rows = batch * blk
        hidden = torch.randn(rows, groups * gsz, generator=g).bfloat16()
        base = torch.randn(taps, groups * gsz, generator=g).bfloat16()
        delta = torch.randn(rows, taps, groups, generator=g).bfloat16()
        a = orig(hidden, delta, base, blk, groups, gsz, taps)
        b = e27(hidden, delta, base, blk, groups, gsz, taps)
        check(f"T12 block {blk}: E27 torch path == Merlin's torch fallback (finite input, bitwise)", torch.equal(a, b)
              and a.dtype == b.dtype and a.shape == b.shape)
        hb = hidden.float().view(batch, blk, groups, gsz)
        bb = base.float().view(taps, groups, gsz)
        db = delta.float().view(batch, blk, taps, groups)
        ref = torch.zeros_like(hb)
        for pos in range(blk):
            for tap in range(min(taps, pos + 1)):
                ref[:, pos] += (bb[tap] + db[:, pos, tap, :, None]) * hb[:, pos - tap]
        diff = float((b.float() - ref.flatten(0, 1).flatten(-2)).abs().max())
        check(f"T12 block {blk}: = the #55960 fp32 reference within bf16 tolerance (max |d| {diff:.3f})",
              torch.allclose(b.float(), ref.flatten(0, 1).flatten(-2), rtol=5e-2, atol=5e-2), diff)
        check(f"T12 block {blk}: the flag routes to the torch path, never the fused op", torch.equal(
            pat_on(hidden, delta, base, blk, groups, gsz, taps), b) and torch.equal(
            pat_off(hidden, delta, base, blk, groups, gsz, taps), a) and not calls)
    # NaN / Inf in request 0 must not reach request 1 (SGLang #41736)
    blk = 4
    hidden = torch.randn(3 * blk, groups * gsz).bfloat16()
    hidden[:blk] = float("nan")
    hidden[blk - 1, :5] = float("inf")
    base = torch.randn(taps, groups * gsz).bfloat16()
    delta = torch.randn(3 * blk, taps, groups).bfloat16()
    a = orig(hidden, delta, base, blk, groups, gsz, taps)
    b = e27(hidden, delta, base, blk, groups, gsz, taps)
    check("T12 NaN in request 0: Merlin's torch fallback leaks it into request 1 row 0 (negative control)",
          bool(torch.isnan(a[blk]).any()))
    check("T12 NaN in request 0: the E27 path keeps requests 1 and 2 finite, request 0 stays NaN",
          bool(torch.isfinite(b[blk:]).all()) and bool(torch.isnan(b[:blk]).all()))
    clean = hidden.clone()
    clean[:blk] = 0
    check("T12 NaN in request 0: requests 1-2 equal the run without the NaN rows",
          torch.equal(b[blk:], e27(clean, delta, base, blk, groups, gsz, taps)[blk:]))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merlin-src", default=DEFAULT_SRC)
    ap.add_argument("--nb4-dir", default=DEFAULT_NB4)
    a = ap.parse_args()
    t0 = time.time()
    clear_env()
    work = Path(tempfile.mkdtemp(prefix="e27seam_"))
    sections = {}
    try:
        patched = t1_patch(a.merlin_src, work)
        iface = load_interfaces(a.merlin_src)
        install_stubs(iface)
        e3 = load_eagle3_utils(a.merlin_src)
        for name, fn in (("T2", lambda: t2_ast(a.merlin_src, patched)),
                         ("T3", lambda: t3_forward(a.merlin_src, patched, iface, e3)),
                         ("T4", lambda: t4_config(a.merlin_src, patched)),
                         ("T5", lambda: t5_ring(a.merlin_src, patched)),
                         ("T6", lambda: t6_load(a.merlin_src, patched, work)),
                         ("T7", lambda: t7_finish(a.merlin_src, patched)),
                         ("T8", lambda: t8_attention(a.merlin_src, patched)),
                         ("T9", t9_readback),
                         ("T10", lambda: t10_compose(a.merlin_src, work, a.nb4_dir)),
                         ("T11", lambda: t11_ingraph(a.merlin_src, patched)),
                         ("T12", lambda: t12_conv(a.merlin_src, patched))):
            n0 = len(RESULTS)
            try:
                fn()
            except Exception:  # noqa: BLE001
                check(f"{name} no exception", False, traceback.format_exc())
            finally:
                clear_env()
            sections[name] = len(RESULTS) - n0
    except Exception:  # noqa: BLE001
        check("no exception", False, traceback.format_exc())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    n_ok = sum(ok for _, ok, _ in RESULTS)
    by = Counter(n.split()[0] for n, _, _ in RESULTS)
    print(json.dumps({"checks": len(RESULTS), "pass": n_ok, "fail": len(RESULTS) - n_ok, "by_section": dict(by),
                      "secs": round(time.time() - t0, 1),
                      "shas": {rel: {"pre": p[0], "post": p[1]} for rel, p in D.PATCHES.items()}}, indent=1))
    print("ALL PASS" if n_ok == len(RESULTS) else "SOME FAILED")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
