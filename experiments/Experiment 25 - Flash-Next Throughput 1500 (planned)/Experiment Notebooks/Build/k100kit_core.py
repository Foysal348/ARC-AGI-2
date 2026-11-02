"""E25 klee100 CPU kit, core (pure Python + numpy; runs in the notebook kernel and locally).

Decides fail-closed whether klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP can be loaded correctly by Merlin v3.0.0
(vLLM 0.32 fork) in TP2xPP2+EP 25/23 with a config-only "fixkeys" overlay and Merlin's FP8 PLE grafted in.

Inputs (all embedded in the kit, sha256-checked; the builder derived them from downloaded HF files, never executed):
  tables/klee100.json, tables/merlin.json   compact, lossless tensor tables (name, dtype, shape, shard) of both
                                            checkpoints, rebuilt here and verified against the canonical sha256
  hf/klee100/config.json (+ build_info, reports)   klee100's own files (config sha fb17316d...)
  hf/merlin/config.json                     Merlin W4A16 config (HF cec53e9f; compared with the mounted p1 copy)
  ple/                                      262 BF16 PLE rows of klee100 + the same rows of Merlin's FP8 table (HF
                                            range fetch), the global FP8 scale, both checkpoints' PLE hash buffers
  facts/merlin_src_facts.json               Merlin source facts used by the static rules, with the sha256 of every
                                            source file they come from (the real mode re-hashes the installed tree)
  facts/author_patch.json                   what the klee100 author's vLLM patch changes (read, never executed)

Outputs: manifests M1 (module map), M2 (quantization map), M3 (weight coverage), the fixkeys overlay files and their
diff, and the gate results A-D. Every gate is fail-closed: a missing input, an unexpected or unmapped item, or a
fallback to BF16 / unquantized / a default quantization is NOT OK.
"""
import collections
import copy
import hashlib
import json
import math
import re
import struct
from pathlib import Path

NL, HID, MOE_I, NE = 48, 2560, 640, 512
GiB, MiB = 2 ** 30, 2 ** 20
DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
            "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "U32": 4}
PLE_PFX = "model.language_model.layers.1.ple.ple_embedding."
KLEE_BF16_PLE_SHARD = "model-00001-of-00011.safetensors"
RX_CHARS = set(r"*+?^$()[]{}|\\")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jdump(o):
    return json.dumps(o, indent=1, default=str)


# ===================================================================================== tensor tables
def expand_table(enc):
    """Lossless decode of the builder's compact table -> {name: (dtype, shape tuple, shard)}."""
    shards = enc["shards"]
    out = {}
    for tpl, dt, shp, f, idx in enc["groups"]:
        shp, sh = tuple(shp), shards[f]
        if idx is None:
            out[tpl] = (dt, shp, sh)
        elif idx and isinstance(idx[0][0], int):            # 1-index runs [[a, b], ...]
            for a, b in idx:
                for x in range(a, b + 1):
                    out[tpl.format(x)] = (dt, shp, sh)
        else:                                                # 2-index: [[runs_a, runs_b], ...]
            for runs_a, runs_b in idx:
                for a0, a1 in runs_a:
                    for a in range(a0, a1 + 1):
                        for b0, b1 in runs_b:
                            for b in range(b0, b1 + 1):
                                out[tpl.format(a, b)] = (dt, shp, sh)
    return out


def canon_sha(tab):
    h = hashlib.sha256()
    for n in sorted(tab):
        dt, shp, sh = tab[n]
        h.update(f"{n}\t{dt}\t{list(shp)}\t{sh}\n".encode())
    return h.hexdigest()


def nbytes(dt, shp):
    return math.prod(shp) * DT_BYTES[dt]


# ===================================================================================== Merlin loader semantics
def layer_of(mod):
    m = re.search(r"\.layers\.(\d+)\.", mod + ".")
    return int(m.group(1)) if m else None


STACK = [  # (substring in the HF key, fused module suffix, shard id) -- Merlin hf_to_vllm / packed / QSA loader
    (".self_attn.indexer.index_qk_proj.", ".self_attn.qkv_proj.", "idx"),
    (".self_attn.q_proj.", ".self_attn.qkv_proj.", "q"), (".self_attn.k_proj.", ".self_attn.qkv_proj.", "k"),
    (".self_attn.v_proj.", ".self_attn.qkv_proj.", "v"),
    (".linear_attn.in_proj_qkv.", ".linear_attn.in_proj_qkvz.", "qkv"), (".linear_attn.in_proj_z.", ".linear_attn.in_proj_qkvz.", "z"),
    (".linear_attn.in_proj_b.", ".linear_attn.in_proj_ba.", "b"), (".linear_attn.in_proj_a.", ".linear_attn.in_proj_ba.", "a"),
    (".mlp.shared_expert.gate_proj.", ".mlp.shared_expert.gate_up_proj.", 0), (".mlp.shared_expert.up_proj.", ".mlp.shared_expert.gate_up_proj.", 1),
    ("_hyper_connection.input_mix_weight_down.", "_hyper_connection.input_mix_weight_down_block_inject.", 0),
    ("_hyper_connection.block_inject_weight.", "_hyper_connection.input_mix_weight_down_block_inject.", 1),
    (".ple.key_proj.", ".ple.kv_proj.", 0), (".ple.value_proj.", ".ple.kv_proj.", 1),
]


def map_key(name, lm_only=False):
    """One checkpoint key -> where Merlin loads it. Returns a dict (side, module, param, shard, ...) or
    {"dropped": reason}. side: target (Qwen4ExpForConditionalGeneration) or draft (Qwen4ExpMTP)."""
    r = {"key": name}
    if name.startswith("mtp."):
        side, rest = "draft", name[4:]
        mod_prefix = "mtp."
        rest = re.sub(r"^layers\.(\d+)\.", lambda m: f"layers.{NL + int(m.group(1))}.", rest)   # prefix layers.48
    elif name.startswith("model.visual."):
        if lm_only:
            return {"key": name, "dropped": "visual (--language-model-only)"}
        side, mod_prefix, rest = "target", "visual.", name[len("model.visual."):]
    elif name.startswith("lm_head."):
        side, mod_prefix, rest = "target", "language_model.lm_head.", name[len("lm_head."):]
    elif name.startswith("model.language_model."):
        side, mod_prefix, rest = "target", "language_model.model.", name[len("model.language_model."):]
    else:
        return {"key": name, "dropped": "no Merlin module (unknown prefix)"}
    if "hyper_connection_mixer.block_inject_weight" in name:
        return {"key": name, "dropped": "skip_substrs (final mixer has no block inject)"}
    full = "." + rest
    shard = None
    if ".ple.ple_embedding.ngram_embedding.shard_" in full:
        m = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", full)
        r.update(side=side, module=mod_prefix + rest.split(".ngram_embedding.")[0] + ".ngram_embedding", param="weight",
                 shard=f"rows[{m.group(1)}]" if m else None, cls="PLE_TABLE")
        return r
    m = re.search(r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)$", full)
    if m:
        e, proj, suf = int(m.group(1)), m.group(2), m.group(3)
        mod = mod_prefix + full[1:full.index(".mlp.experts.")] + ".mlp.experts"
        r.update(side=side, module=mod, param=("w13_" if proj != "down_proj" else "w2_") + suf,
                 shard={"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[proj], expert=e, proj=proj, suffix=suf,
                 cls="RoutedExperts")
        return r
    for sub, fused, sid in STACK:
        if sub in full:
            if "hyper_connection_mixer" in full:     # final mixers: no merged down+inject linear
                break
            full = full.replace(sub, fused, 1)
            shard = sid
            break
    mod, _, param = (mod_prefix + full[1:]).rpartition(".")
    r.update(side=side, module=mod, param=param, shard=shard)
    return r


MOD_RULES = [  # (regex on the vLLM module name, class, quant_config passed, TP rule); sources in facts/merlin_src_facts
    (r"\.mlp\.experts$", "RoutedExperts", True, "ep"),
    (r"\.mlp\.gate$", "GateLinear", False, "rep"),
    (r"\.mlp\.shared_expert_gate$", "ReplicatedLinear", False, "rep"),
    (r"\.mlp\.shared_expert\.gate_up_proj$", "MergedColumnParallelLinear", True, "col"),
    (r"\.mlp\.shared_expert\.down_proj$", "RowParallelLinear", True, "row"),
    (r"\.self_attn\.qkv_proj$", "Qwen4ExpQSAQKVIndexerLinear", True, "qkvidx"),
    (r"\.self_attn\.o_proj$", "RowParallelLinear", True, "row"),
    (r"\.self_attn\.(q_norm|k_norm)$", "GemmaRMSNorm", False, "rep"),
    (r"\.self_attn\.indexer\.(q_layernorm|k_layernorm)$", "IndexerNorm", False, "rep"),
    (r"\.linear_attn\.in_proj_qkvz$", "MergedColumnParallelLinear", True, "col"),
    (r"\.linear_attn\.in_proj_ba$", "MergedColumnParallelLinear", True, "ba"),
    (r"\.linear_attn\.conv1d$", "ColumnParallelLinear", False, "col"),
    (r"\.linear_attn\.out_proj$", "RowParallelLinear", True, "row"),
    (r"\.linear_attn\.norm$", "RMSNormGated", False, "rep"),
    (r"\.linear_attn$", "QwenGatedDeltaNetAttention", False, "col"),
    (r"_hyper_connection\.input_mix_weight_down_block_inject$", "MergedColumnParallelLinear(disable_tp)", False, "rep"),
    (r"hyper_connection(_mixer)?\.input_mix_weight_(up|down)$", "ReplicatedLinear", False, "rep"),
    (r"hyper_connection(_mixer)?\.hc_norm$", "GroupedGemmaRMSNorm", False, "rep"),
    (r"\.ple\.kv_proj$", "MergedColumnParallelLinear(disable_tp)", True, "rep"),
    (r"\.ple\.ple_embedding\.ngram_embedding$", "Qwen4ExpPLEEmbedding", "ple", "host"),
    (r"\.ple\.ple_embedding$", "Qwen4ExpNGramEmbedding(buffers)", False, "rep"),
    (r"\.ple\.(norm_conv|norm_key|norm_query)$", "Qwen4ExpPLEGroupedNorm", False, "rep"),
    (r"\.ple\.conv1d$", "nn.Conv1d", False, "rep"),
    (r"^language_model\.model\.embed_tokens$", "PinnedHostVocabParallelEmbedding (VLLM_HOST_EMBED_TABLE=1)", False, "host_embed"),
    (r"^language_model\.lm_head$", "ParallelLMHead", False, "vocab"),
    (r"^mtp\.fc_(embedding|hidden)$", "ColumnParallelLinear", True, "col"),
    (r"^mtp\.pre_fc_norm_(embedding|hidden)$", "GemmaRMSNorm", False, "rep"),
    (r"^visual\..*\.(qkv|proj|linear_fc1|linear_fc2)$", "VisionLinear", True, "col"),
    (r"^visual\.", "VisionOther", False, "col"),
]


def classify(mod):
    for rx, cls, q, tp in MOD_RULES:
        if re.search(rx, mod):
            return cls, q, tp
    return None, None, None


def stage_of(side, mod):
    if side == "draft":
        return "PP1"
    if mod.startswith("visual.") or mod == "language_model.model.embed_tokens":
        return "PP0"
    L = layer_of(mod)
    if L is not None:
        return "PP0" if L < 25 else "PP1"
    return "PP1"         # lm_head, final norm / HC mixer


def build_modules(tab, lm_only=False):
    """Manifest-1 rows from a tensor table: {(side, module): row}; plus dropped keys."""
    mods, dropped = {}, []
    for name in sorted(tab):
        r = map_key(name, lm_only)
        if "dropped" in r:
            dropped.append(r)
            continue
        k = (r["side"], r["module"])
        row = mods.get(k)
        if row is None:
            cls, q, tp = classify(r["module"])
            if r.get("cls") == "PLE_TABLE":
                cls, q, tp = "Qwen4ExpPLEEmbedding", "ple", "host"
            row = mods[k] = {"side": r["side"], "module": r["module"], "class": cls, "quant_config_passed": q, "tp": tp,
                             "stage": stage_of(r["side"], r["module"]), "layer": layer_of(r["module"]),
                             "params": collections.Counter(), "subs": set(), "keys": [], "bytes": 0}
        row["params"][r["param"]] += 1
        if "expert" in r:                                # K4: exact expert-id list per (projection, suffix)
            row.setdefault("expert_ids", collections.defaultdict(list))[f"{r['proj']}.{r['suffix']}"].append(r["expert"])
        if r.get("shard") is not None and not str(r.get("shard")).startswith("rows["):
            row["subs"].add(str(r["shard"]))
        row["keys"].append(name)
        dt, shp, _ = tab[name]
        row["bytes"] += nbytes(dt, shp)
    return mods, dropped


def key_pattern(keys):
    pats = collections.Counter(re.sub(r"(?<=\.)\d+(?=\.|$)", "{n}", k) for k in keys)
    return dict(sorted(pats.items()))


# ===================================================================================== static INC resolver
class StaticINC:
    """Pure-Python mirror of Merlin's INCConfig.get_quant_method + INCConfigParser._resolve_raw (CUDA branch,
    inc/config_parser.py, inc/inc.py, schemes/inc_wna16_scheme.py, inc_wna16_linear.py). Labelled STATIC."""

    def __init__(self, q, prefix_map, packed):
        self.bits, self.sym = q["bits"], q.get("sym", True)
        self.group = q["group_size"]
        self.packing = q.get("packing_format") or "auto_round:auto_gptq"
        self.data_type = q.get("data_type", "int")
        bn = q.get("block_name_to_quantize") or q.get("to_quant_block_names")
        bn = bn.split(",") if isinstance(bn, str) else (list(bn) if bn else None)
        ec = q.get("extra_config") or {}

        def mp(k):
            for a, b in prefix_map:
                if k.startswith(a):
                    return b + k[len(a):]
            return k
        self.extra = {mp(k): v for k, v in ec.items()} if prefix_map else dict(ec)
        self.blocks = [mp(b) for b in bn] if (bn and prefix_map) else bn
        self.packed = packed
        self.order = {k: i for i, k in enumerate(self.extra)}
        self.regex = []
        for k, v in self.extra.items():
            if isinstance(k, str) and any(c in RX_CHARS for c in k):
                try:
                    self.regex.append((re.compile(k), v))
                except re.error:
                    pass

    def get_config(self, name, quantized=True):
        dflt = (self.bits if quantized else 16, self.group if quantized else -1, self.sym if quantized else True)
        if not self.extra:
            return dflt
        c = self.extra.get(name)
        if c is None:
            for rx, cc in self.regex:
                if rx.search(name):
                    c = cc
                    break
        if c is None:
            return dflt
        g = c.get("group_size", dflt[1])
        return (c.get("bits", dflt[0]), tuple(g) if isinstance(g, list) else g, c.get("sym", dflt[2]))

    def explicit(self, name):
        return name in self.extra or any(rx.search(name) for rx, _ in self.regex)

    def resolve(self, name, cls):
        if self.extra and name in self.extra:
            return self.get_config(name), "exact"
        if self.extra:
            parts = name.split(".")
            cands = [".".join(parts[i:]) for i in range(1, len(parts))]
            hits = [c for c in cands if c in self.extra]
            if hits:
                k = min(hits, key=lambda c: self.order[c])
                return self.get_config(k), f"suffix:{k}"
        quantized = cls != "ParallelLMHead"
        if self.blocks:
            quantized = any(name.startswith(b) for b in self.blocks)
        if self.extra and "fusedmoe" in cls.lower():
            cfgs = {self.get_config(k, quantized) for k in self.extra if k.startswith(name)}
            if len(cfgs) == 1:
                return cfgs.pop(), "fusedmoe_subkeys"
            if cfgs:
                return ("ERROR_INCONSISTENT_MOE",), "fusedmoe_subkeys"
        if self.extra:
            for fk, subs in self.packed.items():
                if fk in name and name.count(fk) == 1:
                    sub_names = [name.replace(fk, s) for s in subs]
                    if not any(self.explicit(n) for n in sub_names):
                        continue
                    cfgs = [self.get_config(n, quantized) for n in sub_names]
                    if len(set(cfgs)) == 1:
                        return cfgs[0], f"packed:{fk}"
                    return ("ERROR_INCONSISTENT_FUSED", sorted(set(map(str, cfgs)))), f"packed:{fk}"
        return self.get_config(name, quantized), ("default" if quantized else "not_in_block_name_to_quantize")

    def method(self, name, cls):
        """-> (method label, (bits, group, sym), how)."""
        moe = cls == "RoutedExperts"
        for k in (name, "model." + name):
            c = self.extra.get(k)
            if c is not None and c.get("bits", 16) >= 16:
                return ("UnquantizedFusedMoEMethod" if moe else "UnquantizedLinearMethod"), (16, -1, True), "exact>=16"
        r, how = self.resolve(name, cls)
        if isinstance(r[0], str):
            return "ERROR:" + r[0], r, how
        bits, group, sym = r
        if bits >= 16:
            return ("UnquantizedFusedMoEMethod" if moe else "UnquantizedLinearMethod"), r, how
        if self.data_type != "int":
            return f"UNSUPPORTED_DATA_TYPE:{self.data_type}", r, how
        gptq = "gptq" in self.packing
        if bits in (2, 3, 5, 6, 7):
            if not (gptq or "awq" in self.packing):
                return "NotImplementedError(humming needs gptq/awq)", r, how
            return ("HummingMoEMethod" if moe else "HummingLinearMethod"), r, how
        if bits in (4, 8) and sym:
            return ("AutoGPTQMoEMethod(Marlin)" if moe else "INCLinearMethod(AutoGPTQLinearMethod/Marlin)"), r, how
        return ("MoeWNA16Method" if moe else "INCLinearMethod(GPTQLinearMethod)"), r, how


def ple_method(cfg, qc_target_static, prefix):
    tc = cfg.get("text_config") or {}
    if tc.get("ple_embedding_dtype") == "float8_e4m3fn":
        return "Qwen4ExpPLEFp8EmbeddingMethod", "text_config.ple_embedding_dtype"
    if qc_target_static is None:
        return "Qwen4ExpPLEUnquantizedEmbeddingMethod", "no quant config"
    r, how = qc_target_static.resolve(prefix, "NoneType")
    if not isinstance(r[0], str) and r[0] >= 16:
        return "Qwen4ExpPLEUnquantizedEmbeddingMethod", how
    return "NotImplementedError(INC quantized PLE prefix -> not Fp8Config)", how


GPU_ONLY_HUMMING = "GPU_ONLY_HUMMING_OBJECT"
EXPECTED_MOE = [("target", f"language_model.model.layers.{L}.mlp.experts") for L in range(NL)] + \
               [("draft", f"mtp.layers.{NL}.mlp.experts")]         # 48 target layers + the MTP bank (K2 / K4)
MOE_PROJ = ("gate_proj", "up_proj", "down_proj")


def method_family(m):
    m = m or ""
    if m.startswith("ERROR"):
        return "error"
    if m.startswith(GPU_ONLY_HUMMING):
        # the object's constructor failed on cuInit inside humming frames (no GPU): the resolver chose the humming
        # path; the object itself is GPU-only (see the construction status of the row)
        return "humming"
    if "Humming" in m:
        return "humming"
    if "Marlin" in m or "AutoGPTQ" in m:
        return "marlin"
    if "Unquantized" in m:
        return "bf16"
    if "Fp8Embedding" in m:
        return "fp8_ple"
    return "other:" + m


# ===================================================================================== checkpoint-side truth
def arch_dims(tc):
    """(N, K) of every linear sub-weight from the text config (independent of both checkpoints)."""
    H = tc["hidden_size"]
    nh, hd, kvh = tc["num_attention_heads"], tc["head_dim"], tc["num_key_value_heads"]
    kd = tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
    vd = tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
    idx_n = (tc["indexer_n_heads"] + tc["indexer_kv_heads"]) * tc["indexer_head_dim"]
    hc, lr = tc["hc_count"], tc["hc_lowrank"]
    I, Ish, nexp = tc["moe_intermediate_size"], tc["shared_expert_intermediate_size"], tc["num_experts"]
    return [(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj)$", I, H), (r"\.mlp\.experts\.\d+\.down_proj$", H, I),
            (r"\.self_attn\.q_proj$", nh * hd * 2, H), (r"\.self_attn\.(k_proj|v_proj)$", kvh * hd, H),
            (r"\.self_attn\.indexer\.index_qk_proj$", idx_n, H), (r"\.self_attn\.o_proj$", H, nh * hd),
            (r"\.linear_attn\.in_proj_qkv$", 2 * kd + vd, H), (r"\.linear_attn\.in_proj_z$", vd, H),
            (r"\.linear_attn\.(in_proj_a|in_proj_b)$", tc["linear_num_value_heads"], H), (r"\.linear_attn\.out_proj$", H, vd),
            (r"\.mlp\.shared_expert\.(gate_proj|up_proj)$", Ish, H), (r"\.mlp\.shared_expert\.down_proj$", H, Ish),
            (r"\.mlp\.gate$", nexp, H), (r"\.mlp\.shared_expert_gate$", 1, H),
            (r"\.ple\.key_proj$", hc * H, tc["ple_embed_dim"]), (r"\.ple\.value_proj$", H, tc["ple_embed_dim"]),
            (r"^mtp\.fc_(embedding|hidden)$", H, H), (r"input_mix_weight_down$", lr, hc * H), (r"input_mix_weight_up$", hc * H, lr),
            (r"block_inject_weight$", hc, hc * H), (r"(embed_tokens|^lm_head)$", tc["vocab_size"], H)]


def nk_of(dims, sub):
    for rx, n, k in dims:
        if re.search(rx, sub):
            return n, k
    return None


def ckpt_formats(tab, dims, merlin_bits=None):
    """Per HF sub-module (key minus suffix): format, bits, group, shape checks. GPTQ (qweight [K*b/32, N], qzeros
    [K/g, N*b/32], scales [K/g, N]), compressed-tensors (weight_packed [N, K*b/32], weight_scale [N, K/g | 1],
    weight_shape [2]) or dense ([N, K])."""
    by = collections.defaultdict(dict)
    for n, (dt, shp, _) in tab.items():
        base, _, suf = n.rpartition(".")
        by[base][suf] = (dt, shp)
    out, bad = {}, []
    for base, s in by.items():
        nk = nk_of(dims, base)
        rec = {"suffixes": sorted(s)}
        if "qweight" in s:
            if nk is None or not {"qweight", "qzeros", "scales"} <= set(s) or set(s) - {"qweight", "qzeros", "scales", "g_idx"}:
                rec["fmt"] = "GPTQ?"
                rec["error"] = f"suffix set {sorted(s)} / dims {nk}"
                bad.append(base)
            else:
                N, K = nk
                qw, qz, sc = s["qweight"][1], s["qzeros"][1], s["scales"][1]
                b = qw[0] * 32 // K if qw[0] * 32 % K == 0 else None
                g = K // sc[0] if sc[0] and K % sc[0] == 0 else None
                ok = (b is not None and g is not None and qw[1] == N and sc[1] == N and list(qz) == [K // g, N * b // 32]
                      and s["qweight"][0] == "I32" and s["qzeros"][0] == "I32" and s["scales"][0] in ("F16", "BF16"))
                rec.update(fmt="GPTQ", bits=b, group=g, N=N, K=K, packed_axis="qweight dim0 (K), qzeros dim1 (N)",
                           scale_dtype=s["scales"][0], has_qzeros=True, has_g_idx="g_idx" in s, ok=ok)
                if not ok:
                    bad.append(base)
        elif "weight_packed" in s:
            N, K = nk if nk else (None, None)
            wp, ws = s["weight_packed"][1], s.get("weight_scale", (None, [None, None]))[1]
            b = wp[1] * 32 // K if K and wp[1] * 32 % K == 0 else None
            g = (K // ws[1]) if (K and ws and ws[1] and K % ws[1] == 0 and ws[1] > 1) else ("channel" if ws and ws[1] == 1 else None)
            ok = bool(nk and wp[0] == N and b and {"weight_packed", "weight_scale", "weight_shape"} <= set(s))
            rec.update(fmt="CT", bits=b, group=g, N=N, K=K, packed_axis="weight_packed dim1 (K)", ok=ok)
            if not ok:
                bad.append(base)
        elif "weight" in s:
            dt, shp = s["weight"]
            rec.update(fmt="DENSE", dtype=dt, shape=list(shp), bits=8 * DT_BYTES[dt])
            if nk is not None and len(shp) == 2 and list(shp) != [nk[0], nk[1]]:
                rec.update(ok=False, error=f"shape {shp} != arch {list(nk)}")
                bad.append(base)
            else:
                rec["ok"] = True
        else:
            rec.update(fmt="OTHER", ok=True)
        out[base] = rec
    return out, bad


# ===================================================================================== the kit
class Kit:
    def __init__(self, emb, log=print):
        self.emb = Path(emb)
        self.log = log
        self.R = {"inputs": {}, "manifests": {}, "gates": {}, "artifacts": {}, "notes": []}

    # ------------------------------------------------------------------ inputs
    def load(self):
        e = self.emb
        self.facts = json.loads((e / "facts" / "merlin_src_facts.json").read_text())
        self.patch = json.loads((e / "facts" / "author_patch.json").read_text())
        self.kcfg_raw = (e / "hf" / "klee100" / "config.json").read_bytes()
        self.kcfg = json.loads(self.kcfg_raw)
        self.mcfg = json.loads((e / "hf" / "merlin" / "config.json").read_text())
        self.build_info = json.loads((e / "hf" / "klee100" / "build_info.json").read_text())
        self.serving = json.loads((e / "hf" / "klee100" / "serving_report.json").read_text())
        self.validation = json.loads((e / "hf" / "klee100" / "validation_report.json").read_text())
        self.tabs, ok = {}, {}
        for who in ("klee100", "merlin"):
            enc = json.loads((e / "tables" / f"{who}.json").read_text())
            t = expand_table(enc)
            got = canon_sha(t)
            ok[who] = {"tensors": len(t), "canon_sha256": got, "want": enc["canon_sha256"], "ok": got == enc["canon_sha256"]
                       and len(t) == enc["n_tensors"], "rev": enc["rev"], "repo": enc["repo"],
                       "index_metadata": enc.get("index_metadata"), "header_sha256": enc.get("header_sha256")}
            self.tabs[who] = t
            self.__dict__[f"enc_{who}"] = enc
        self.R["inputs"]["tables"] = ok
        self.R["inputs"]["klee100_config_sha256"] = sha(self.kcfg_raw)
        self.ple = json.loads((e / "ple" / "ple_rows_manifest.json").read_text())
        self.ple_bf16 = (e / "ple" / "klee100_ple_rows_bf16.bin").read_bytes()
        self.R["inputs"]["ple_rows"] = {"n": self.ple["n_rows"], "bf16_ok": sha(self.ple_bf16) == self.ple["bf16_bin_sha256"]}
        return all(v["ok"] for v in ok.values()) and all(self.R["inputs"]["ple_rows"].values())

    # ------------------------------------------------------------------ fixkeys overlay (config only)
    def fixkeys(self, cfg=None):
        """klee100 config + (1) one layer-level extra_config key per routed-expert layer spelled as Merlin's
        RoutedExperts module name (language_model.model.layers.N.mlp.experts, bits/group from the checkpoint's own
        per-expert entries, which must agree), (2) every checkpoint MTP entry mtp.layers.0.* also written as
        mtp.layers.48.* (the runtime draft prefix; the author's patch does this remap in code) plus the layer key
        mtp.layers.48.mlp.experts, (3) block_name_to_quantize += mtp.layers, (4) text_config.ple_embedding_dtype =
        float8_e4m3fn (Merlin FP8 PLE graft; launch.py:1219 rule). Nothing else changes."""
        c = copy.deepcopy(cfg or self.kcfg)
        q = c["quantization_config"]
        ec = q["extra_config"]
        added, problems = {}, []
        for L in range(NL):
            vals = {json.dumps(v, sort_keys=True) for k, v in ec.items() if k.startswith(f"model.language_model.layers.{L}.mlp.experts.")}
            if len(vals) > 1:
                problems.append(f"layer {L}: {len(vals)} distinct per-expert configs")
                continue
            # no per-expert entries -> the layer uses the top-level default (bits 2 / group_size 64); written explicitly
            v = json.loads(vals.pop()) if vals else {"bits": q["bits"], "group_size": q["group_size"]}
            added[f"language_model.model.layers.{L}.mlp.experts"] = {k: v[k] for k in ("bits", "group_size") if k in v}
        mtp_new = {}
        for k, v in ec.items():
            if k.startswith("mtp.layers.0."):
                mtp_new[k.replace("mtp.layers.0.", f"mtp.layers.{NL}.", 1)] = v
        vals = {json.dumps(v, sort_keys=True) for k, v in ec.items() if k.startswith("mtp.layers.0.mlp.experts.")}
        if len(vals) == 1:
            v = json.loads(vals.pop())
            mtp_new[f"mtp.layers.{NL}.mlp.experts"] = {k: v[k] for k in ("bits", "group_size", "sym") if k in v}
        else:
            problems.append(f"mtp experts: {len(vals)} distinct per-expert configs")
        for k in list(added) + list(mtp_new):
            if k in ec:
                problems.append(f"key already present: {k}")
        ec.update(added)
        ec.update(mtp_new)
        bn_old = q.get("block_name_to_quantize")
        bn = bn_old.split(",") if isinstance(bn_old, str) else list(bn_old or [])
        if "mtp.layers" not in bn:
            bn.append("mtp.layers")
        q["block_name_to_quantize"] = ",".join(bn)
        c["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
        diff = {"extra_config_added_layer_keys": added, "extra_config_added_mtp_keys_count": len(mtp_new),
                "extra_config_added_mtp_keys_nonexpert": {k: v for k, v in mtp_new.items() if ".mlp.experts." not in k},
                "extra_config_added_mtp_expert_keys": f"{sum('.mlp.experts.' in k for k in mtp_new)} keys "
                                                      f"mtp.layers.48.mlp.experts.{{0..511}}.{{gate,up,down}}_proj (copies of mtp.layers.0.*)",
                "block_name_to_quantize": [bn_old, q["block_name_to_quantize"]],
                "text_config.ple_embedding_dtype": [self.kcfg["text_config"].get("ple_embedding_dtype"), "float8_e4m3fn"],
                "unchanged": "every other key of config.json (incl. the original extra_config entries, layer_types, "
                             "quant_method auto-round, packing_format auto_round:auto_gptq, bits 2 / group_size 64 defaults)",
                "problems": problems}
        return c, diff

    def write_overlay(self, out_dir, cfg_fix, diff, index_map):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        b = (json.dumps(cfg_fix, indent=2, ensure_ascii=False) + "\n").encode()
        (out / "config.json").write_bytes(b)
        qb = (json.dumps(cfg_fix["quantization_config"], indent=2, ensure_ascii=False) + "\n").encode()
        (out / "quantization_config.json").write_bytes(qb)
        ib = (json.dumps(index_map, indent=2) + "\n").encode()
        (out / "model.safetensors.index.json").write_bytes(ib)
        (out / "FIXKEYS_DIFF.json").write_text(jdump(diff))
        rec = {n: {"bytes": len(x), "sha256": sha(x)} for n, x in (("config.json", b), ("quantization_config.json", qb),
                                                                   ("model.safetensors.index.json", ib))}
        rec["FIXKEYS_DIFF.json"] = {"sha256": sha((out / "FIXKEYS_DIFF.json").read_bytes())}
        self.R["artifacts"]["fixkeys_overlay"] = {"dir": str(out), "files": rec}
        return rec

    # ------------------------------------------------------------------ grafted index
    def graft(self):
        """klee100 weight_map with the 128 BF16 PLE shards replaced by Merlin's FP8 shards + the global scale."""
        mt, kt = self.tabs["merlin"], self.tabs["klee100"]
        wm = {n: v[2] for n, v in kt.items()}
        removed = [n for n in wm if n.startswith(PLE_PFX + "ngram_embedding.")]
        for n in removed:
            del wm[n]
        added = {n: v[2] for n, v in mt.items() if n.startswith(PLE_PFX + "ngram_embedding.")}
        wm.update(added)
        tab = {n: v for n, v in kt.items() if n not in set(removed)}
        tab.update({n: mt[n] for n in added})
        meta = dict(self.enc_klee100.get("index_metadata") or {})
        meta["total_size"] = sum(nbytes(dt, shp) for dt, shp, _ in tab.values())
        meta["e25_fixkeys_graft"] = "128 BF16 PLE shards of klee100 replaced by Merlin's FP8 PLE shards (model-plefp8-0000N) + weight_scale"
        idx = {"metadata": meta, "weight_map": dict(sorted(wm.items()))}
        files = sorted(set(wm.values()))
        chk = {"removed_bf16_ple_tensors": len(removed), "added_fp8_ple_tensors": len(added),
               "klee100_bf16_ple_shard_referenced": KLEE_BF16_PLE_SHARD in files,
               "ple_tensors_all_in_plefp8_files": all(wm[n].startswith("model-plefp8-") for n in added),
               "ple_dtypes": dict(collections.Counter(mt[n][0] for n in added)),
               "files": files, "n_tensors": len(wm)}
        chk["ok"] = (len(removed) == 128 and len(added) == 129 and not chk["klee100_bf16_ple_shard_referenced"]
                     and chk["ple_tensors_all_in_plefp8_files"] and chk["ple_dtypes"] == {"F8_E4M3": 128, "BF16": 1})
        return idx, tab, chk

    # ------------------------------------------------------------------ Manifest 1 + Manifest 3
    @staticmethod
    def expert_id_coverage(mods_k):
        """K4: per expected MoE module (48 target layers + the MTP bank) and per (projection, suffix) of its format
        family, the expert ids must be exactly {0..NE-1}, each once. Counts alone pass a renamed id (0 -> 512) or a
        duplicate that replaces another id; this check does not."""
        fam = {"GPTQ": ("qweight", "qzeros", "scales"), "CT": ("weight_packed", "weight_scale", "weight_shape"),
               "DENSE": ("weight",)}
        errs, per = [], {}
        want = set(range(NE))
        for k in EXPECTED_MOE:
            r = mods_k.get(k)
            if r is None or r["class"] != "RoutedExperts":
                errs.append(f"{k[1]}: MoE module missing")
                per[k[1]] = {"ok": False, "why": "missing"}
        for k, r in sorted(mods_k.items()):
            if r["class"] != "RoutedExperts":
                continue
            ids = r.get("expert_ids") or {}
            sufs = {s.split(".", 1)[1] for s in ids}
            fmt = next((f for f, ss in fam.items() if sufs and sufs <= set(ss)), None)
            rec = {"fmt": fmt, "pairs": {}}
            if fmt is None:
                errs.append(f"{k[1]}: suffix set {sorted(sufs)} is no known format family")
            for proj in MOE_PROJ:
                for s in fam.get(fmt) or ():
                    got = ids.get(f"{proj}.{s}", [])
                    c = collections.Counter(got)
                    dup = sorted(i for i, n in c.items() if n > 1)
                    miss = sorted(want - set(c))
                    extra = sorted(set(c) - want)
                    ok = not dup and not miss and not extra and len(got) == NE
                    rec["pairs"][f"{proj}.{s}"] = {"n": len(got), "ok": ok, **({"duplicate_ids": dup[:8], "missing_ids": miss[:8],
                                                                                 "extra_ids": extra[:8]} if not ok else {})}
                    if not ok:
                        errs.append(f"{k[1]}: {proj}.{s} ids n={len(got)} duplicate {dup[:4]} missing {miss[:4]} extra {extra[:4]}")
            rec["ok"] = fmt is not None and all(v["ok"] for v in rec["pairs"].values())
            per[k[1]] = rec
        return {"expected_modules": len(EXPECTED_MOE), "checked_modules": sum(1 for k in EXPECTED_MOE if k in mods_k),
                "pairs_checked": sum(len(v.get("pairs", {})) for v in per.values()),
                "errors": errs, "ok": not errs,
                "per_module_ok": {m: v["ok"] for m, v in per.items()},
                "bad_modules": {m: v for m, v in per.items() if not v["ok"]}}

    def _merlin_side(self, lm_only):
        """Merlin (RC2-proven) module set + formats; computed once per lm_only (it never changes)."""
        cache = self.__dict__.setdefault("_mcache", {})
        if lm_only not in cache:
            mods_m, drop_m = build_modules(self.tabs["merlin"], lm_only)
            fm, bad_m = ckpt_formats(self.tabs["merlin"], arch_dims(self.mcfg["text_config"]))
            cache[lm_only] = (mods_m, drop_m, fm, bad_m)
        return cache[lm_only]

    def manifests_13(self, tab_k, lm_only=False, label="klee100_fixkeys_graft"):
        mods_k, drop_k = build_modules(tab_k, lm_only)
        mods_m, drop_m, fm, bad_m = self._merlin_side(lm_only)
        dims = arch_dims(self.kcfg["text_config"])
        fk, bad_k = ckpt_formats(tab_k, dims)
        # module sets: klee100 vs the RC2-proven Merlin set (same model class)
        ks, ms = set(mods_k), set(mods_m)
        unexpected_mod = sorted(ks - ms)
        missing_mod = sorted(ms - ks)
        unclassified = sorted(k for k, r in mods_k.items() if r["class"] is None)
        # per-module param coverage: the suffix family of each module must be a known format family
        fam = {"GPTQ": {"qweight", "qzeros", "scales"}, "CT": {"weight_packed", "weight_scale", "weight_shape"}, "DENSE": {"weight"}}
        unmapped, missing = [], []
        for k, r in mods_k.items():
            if r["class"] == "RoutedExperts":
                sufs = {p.split("_", 1)[1] for p in r["params"]}
                per = collections.Counter(r["params"])
                for p in r["params"]:
                    if p.split("_", 1)[1] not in fam["GPTQ"] and p.split("_", 1)[1] not in fam["CT"] | fam["DENSE"]:
                        unmapped.append(f"{k[1]}:{p}")
                fmt = "GPTQ" if sufs <= fam["GPTQ"] else ("CT" if sufs <= fam["CT"] else "DENSE" if sufs <= fam["DENSE"] else "?")
                if fmt == "?":
                    unmapped.append(f"{k[1]}: mixed suffixes {sorted(sufs)}")
                for s in fam.get(fmt, ()):
                    if per.get("w13_" + s, 0) != 2 * NE or per.get("w2_" + s, 0) != NE:
                        missing.append(f"{k[1]}: w13_{s} {per.get('w13_' + s, 0)}/{2 * NE}, w2_{s} {per.get('w2_' + s, 0)}/{NE}")
            elif r["class"] == "Qwen4ExpPLEEmbedding":
                if r["params"].get("weight") != 128 or r["params"].get("weight_scale", 0) not in (0, 1):
                    missing.append(f"{k[1]}: PLE shards {r['params'].get('weight')}/128")
        sub_bad = list(bad_k)          # sub-weights whose shapes / suffixes do not fit a known format at the arch dims
        # format of the PLE table after the graft
        ple_scale = any(n.endswith("ngram_embedding.weight_scale") for n in tab_k)
        m1 = []
        for k in sorted(mods_k):
            r = mods_k[k]
            mref = mods_m.get(k)
            m1.append({"side": r["side"], "module": r["module"], "class": r["class"], "quant_config_passed": r["quant_config_passed"],
                       "tp": r["tp"], "stage": r["stage"], "layer": r["layer"], "params": dict(r["params"]),
                       "fused_shards": sorted(r["subs"]), "n_ckpt_keys": len(r["keys"]), "ckpt_key_patterns": key_pattern(r["keys"]),
                       "ckpt_keys_sha256": sha("\n".join(sorted(r["keys"])).encode()), "bytes": r["bytes"],
                       "in_merlin_rc2_set": mref is not None, "merlin_params": dict(mref["params"]) if mref else None})
        m3 = {"label": label, "language_model_only": lm_only, "provided_keys": len(tab_k),
              "loaded_keys": sum(len(r["keys"]) for r in mods_k.values()),
              "dropped_keys": collections.Counter(d["dropped"] for d in drop_k),
              "merlin_dropped_keys": collections.Counter(d["dropped"] for d in drop_m),
              "modules": len(mods_k), "merlin_modules": len(mods_m),
              "unexpected_modules": [list(x) for x in unexpected_mod], "missing_modules": [list(x) for x in missing_mod],
              "unclassified_modules": [list(x) for x in unclassified], "unmapped_params": unmapped, "missing_params": missing,
              "bad_shapes": sub_bad[:50], "bad_shapes_n": len(sub_bad), "merlin_bad_shapes_n": len(bad_m),
              "ple_weight_scale_present": ple_scale}
        dropped_bad = [d["key"] for d in drop_k if d["dropped"].startswith("no Merlin module")]
        m3["unknown_prefix_keys"] = dropped_bad[:20]
        ids = self.expert_id_coverage(mods_k)
        missing_moe = [e for e in ids["errors"] if e.endswith("MoE module missing")]
        missing += missing_moe
        m3["missing_params"] = missing
        m3["expert_id_coverage"] = ids
        m3["unexpected"] = len(unexpected_mod) + len(unmapped) + len(dropped_bad) + len(unclassified)
        m3["missing"] = len(missing_mod) + len(missing)
        m3["verdict"] = ("NO_GO_UNMAPPED_WEIGHT" if (unexpected_mod or unmapped or dropped_bad or unclassified)
                         else "NO_GO_MISSING_WEIGHT" if (missing_mod or missing)
                         else "NO_GO_EXPERT_ID_SET" if not ids["ok"]
                         else "NO_GO_BAD_SHAPE" if sub_bad
                         else "OK")
        self._mods_k, self._fk, self._fm = mods_k, fk, fm
        return m1, m3, mods_k, fk

    # ------------------------------------------------------------------ intended scheme (checkpoint truth)
    def intended(self, mods_k, fk):
        """Per quant-config module: intended (bits, group) from the checkpoint tensors (all sub-weights must agree)."""
        out = {}
        for (side, mod), r in mods_k.items():
            if r["class"] in ("Qwen4ExpPLEEmbedding",) or r["quant_config_passed"] is not True:
                continue
            subs = sorted({key.rpartition(".")[0] for key in r["keys"]})
            fmts = {(fk[s].get("fmt"), fk[s].get("bits"), fk[s].get("group")) for s in subs if s in fk}
            if len(fmts) == 1:
                f, b, g = fmts.pop()
                out[(side, mod)] = {"fmt": f, "bits": b if f != "DENSE" else 16, "group": g if f != "DENSE" else -1,
                                    "dense_dtype": fk[subs[0]].get("dtype") if f == "DENSE" else None, "uniform": True}
            else:
                out[(side, mod)] = {"fmt": "MIXED", "variants": sorted(map(str, fmts)), "uniform": False}
        return out

    def stated_layout_check(self, intended):
        """klee100 build_info: W3 g128 layers 0-11 & 36-47, W2 g64 12-35, MTP experts W4 g128, MTP QSA + dense W8 g128."""
        bad = {}
        for L in range(NL):
            want = (3, 128) if (L <= 11 or L >= 36) else (2, 64)
            got = intended.get(("target", f"language_model.model.layers.{L}.mlp.experts"), {})
            if (got.get("bits"), got.get("group")) != want:
                bad[str(L)] = [got.get("bits"), got.get("group"), list(want)]
        got = intended.get(("draft", f"mtp.layers.{NL}.mlp.experts"), {})
        if (got.get("bits"), got.get("group")) != (4, 128):
            bad["mtp_experts"] = [got.get("bits"), got.get("group")]
        for m in (f"mtp.layers.{NL}.self_attn.qkv_proj", f"mtp.layers.{NL}.self_attn.o_proj", "mtp.fc_embedding", "mtp.fc_hidden"):
            got = intended.get(("draft", m), {})
            if (got.get("bits"), got.get("group")) != (8, 128):
                bad[m] = [got.get("bits"), got.get("group")]
        return {"ok": not bad, "bad": bad, "build_info_allocation": self.build_info.get("backbone_routed_expert_allocation"),
                "mtp_experts": self.build_info.get("mtp_experts"), "mtp_qsa_and_dense": self.build_info.get("mtp_qsa_and_dense")}

    # ------------------------------------------------------------------ Manifest 2 (static) + Gate A rows
    def static_quant(self, cfg, mods_k):
        F = self.facts
        tq = StaticINC(cfg["quantization_config"], [tuple(x) for x in F["cg_rename_prefixes"]], F["packed_target"])
        dq = StaticINC(cfg["quantization_config"], None, F["packed_draft"])
        rows = {}
        for (side, mod), r in mods_k.items():
            if r["quant_config_passed"] is True:
                qq = tq if side == "target" else dq
                m, res, how = qq.method(mod, r["class"])
                rows[(side, mod)] = {"method": m, "resolved": list(res) if not isinstance(res[0], str) else list(map(str, res)),
                                     "how": how}
            elif r["quant_config_passed"] == "ple":
                m, how = ple_method(cfg, tq, mod)
                rows[(side, mod)] = {"method": m, "resolved": None, "how": how}
            else:
                rows[(side, mod)] = {"method": "NO_QUANT_CONFIG (source: module built with quant_config=None)", "resolved": [16, -1, True],
                                     "how": "source"}
        return rows

    def gate_a_rows(self, quant_rows, intended, mods_k, label):
        """Compare resolved vs intended for every quant-config module; collect unexpected quantized modules."""
        rows, bad, unexpected_q = [], [], []
        for (side, mod), r in sorted(mods_k.items()):
            q = quant_rows.get((side, mod))
            if q is None:
                bad.append(f"{mod}: no resolution")
                continue
            it = intended.get((side, mod))
            fam = method_family(q["method"])
            row = {"side": side, "module": mod, "class": r["class"], "method": q["method"], "family": fam,
                   "resolved": q.get("resolved"), "how": q.get("how"), "intended": it,
                   "construction": q.get("construction", "NOT_ATTEMPTED (static)")}
            if r["quant_config_passed"] == "ple":
                row["ok"] = fam == "fp8_ple"
            elif r["quant_config_passed"] is not True:
                row["ok"] = True
            elif it is None or not it.get("uniform"):
                row["ok"] = False
                row["why"] = "no uniform checkpoint format"
            else:
                res = q.get("resolved") or [None, None]
                if it["bits"] >= 16:
                    row["ok"] = fam == "bf16"
                    if fam != "bf16":
                        unexpected_q.append(mod)
                else:
                    want_fam = "humming" if it["bits"] in (2, 3, 5, 6, 7) else "marlin"
                    row["ok"] = (res[0] == it["bits"] and res[1] == it["group"] and fam == want_fam
                                 and (len(res) < 3 or res[2] is True))
                    row["want_family"] = want_fam
            # resolution = bits / group / family (the row's ok so far); construction is a separate K1 status
            row["resolution_ok"] = bool(row["ok"])
            if str(row["construction"]).startswith("ERROR"):
                row["ok"] = False
            if not row["ok"]:
                bad.append(f"{mod}: {q['method']} {q.get('resolved')} vs ckpt {it} | construction {row['construction']}"[:600])
            rows.append(row)
        cons = collections.Counter(str(r["construction"]).split(":")[0].split(" ")[0] for r in rows)
        return {"label": label, "rows": rows, "bad": bad, "n_bad": len(bad), "unexpected_quantized_modules": unexpected_q,
                "construction_counts": dict(cons),
                "gpu_only_construction": [r["module"] for r in rows if str(r["construction"]).startswith(GPU_ONLY_HUMMING)],
                "construction_errors": [r["module"] for r in rows if str(r["construction"]).startswith("ERROR")],
                "moe_target_resolution_pass": sum(1 for r in rows if r["class"] == "RoutedExperts" and r["side"] == "target"
                                                  and r["resolution_ok"]),
                "ok": not bad and not unexpected_q}

    # ------------------------------------------------------------------ Gate B: PLE provenance + numerics
    @staticmethod
    def e4m3_lut():
        import numpy as np
        v = []
        for b in range(256):
            s = -1.0 if b & 0x80 else 1.0
            e, m = (b >> 3) & 0xF, b & 7
            if e == 15 and m == 7:
                v.append(float("nan"))
            elif e == 0:
                v.append(s * (m / 8) * 2.0 ** -6)
            else:
                v.append(s * (1 + m / 8) * 2.0 ** (e - 7))
        return np.array(v, dtype=np.float32)

    @staticmethod
    def to_bf16(x):
        import numpy as np
        u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
        r = (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16)
        return r.view(np.float32)

    def ple_numeric(self, fp8_rows, scale_bf16_hex, label, permute=False):
        """Merlin's dequant (qwen4_exp/common/ngram_embedding.py Qwen4ExpPLEFp8EmbeddingMethod.dequantize:
        embeddings.to(bf16) * weight_scale.to(bf16)) vs klee100's BF16 rows."""
        import numpy as np
        n, C = self.ple["n_rows"], self.ple["cols"]
        bf = (np.frombuffer(self.ple_bf16, dtype="<u2").reshape(n, C).astype(np.uint32) << 16).view(np.float32)
        f8 = np.frombuffer(fp8_rows, dtype=np.uint8).reshape(n, C)
        if permute:
            f8 = f8[np.roll(np.arange(n), 7)]
        lut = self.e4m3_lut()
        scale = struct.unpack("<f", b"\x00\x00" + bytes.fromhex(scale_bf16_hex))[0]
        deq = self.to_bf16(self.to_bf16(lut[f8]) * np.float32(scale))
        nb, nd = np.linalg.norm(bf, axis=1), np.linalg.norm(deq, axis=1)
        tiny = scale * 2.0 ** -10                    # below half the smallest e4m3 subnormal x scale -> FP8 must be 0
        degenerate = np.abs(bf).max(1) < tiny
        cos = np.where(degenerate, np.nan, (bf * deq).sum(1) / (nb * nd + 1e-30))
        mx = np.abs(deq - bf).max(1)
        normal = np.abs(bf) >= scale * 2.0 ** -6
        rel = np.abs(deq - bf)[normal] / np.abs(bf)[normal]
        shift = np.roll(np.arange(n), 1)
        cs = (bf * deq[shift]).sum(1) / (nb * nd[shift] + 1e-30)
        order = np.argsort(lut[:127])
        vals = lut[:127][order]
        q = (bf / np.float32(scale)).astype(np.float32)
        a = np.abs(q).ravel()
        i = np.clip(np.searchsorted(vals, a), 1, len(vals) - 1)
        pick = np.where(a - vals[i - 1] <= vals[i] - a, i - 1, i)
        code = (order[pick].astype(np.uint8) | (np.signbit(q.ravel()).astype(np.uint8) << 7)).reshape(n, C)
        rank = np.zeros(256, np.int64)
        rank[order] = np.arange(127)
        rank[128:255] = rank[0:127]
        sg = lambda x: np.where(x & 0x80, -1, 1)
        step = rank[code] * sg(code) - rank[f8] * sg(f8)
        step = np.where(((code & 0x7F) == 0) & ((f8 & 0x7F) == 0), 0, step)
        nd_rows = ~degenerate
        out = {"label": label, "permuted_negative_test": permute, "rows": int(n), "cols": int(C), "scale": scale,
               "degenerate_rows": [dict(self.ple["rows"][j], bf16_absmax=float(np.abs(bf[j]).max()),
                                        fp8_nonzero=int((f8[j] & 0x7F).astype(bool).sum())) for j in np.where(degenerate)[0]],
               "cos_min": float(np.nanmin(cos)), "cos_median": float(np.nanmedian(cos)),
               "max_abs_err": float(mx[nd_rows].max()), "normal_range_rel_err_max": float(rel.max()) if rel.size else None,
               "normal_range_rel_err_p99": float(np.quantile(rel, 0.99)) if rel.size else None,
               "e4m3_half_ulp_rel": 2.0 ** -4,
               "shifted_row_control_abs_cos_max": float(np.abs(cs[nd_rows]).max()),
               "rtn_code_equal_frac": float((code == f8)[nd_rows].mean()),
               "code_step_hist": {int(k): int(v) for k, v in zip(*np.unique(step[nd_rows], return_counts=True))},
               "degenerate_rows_fp8_zero": bool(all((f8[j] & 0x7F).max() == 0 for j in np.where(degenerate)[0]))}
        out["ok"] = bool(out["cos_min"] >= 0.999 and out["normal_range_rel_err_max"] is not None
                         and out["normal_range_rel_err_max"] <= 2.0 ** -4 * 1.02 and out["shifted_row_control_abs_cos_max"] < 0.6
                         and set(out["code_step_hist"]) <= {-1, 0, 1} and out["degenerate_rows_fp8_zero"]
                         and len(out["degenerate_rows"]) <= 3)
        if out["ok"]:
            out["production"] = ("FP8 E4M3 with one global BF16 scale of the SAME table: every non-degenerate row cos >= 0.999, "
                                 "element error within the e4m3 half-ulp; codes equal round-to-nearest(BF16 / scale) for "
                                 f"{out['rtn_code_equal_frac']:.1%} of elements and differ by exactly one code step otherwise "
                                 "(symmetric), i.e. quantized from a higher-precision copy of the same base table, not from "
                                 "klee100's BF16 bytes")
        return out

    def ple_metadata(self, overlay_headers, overlay_files):
        """Overlay (mounted) metadata vs Merlin's index vs klee100's BF16 table."""
        mt, kt = self.tabs["merlin"], self.tabs["klee100"]
        k_t = {n: kt[n] for n in kt if n.startswith(PLE_PFX + "ngram_embedding.")}
        m_t = {n: mt[n] for n in mt if n.startswith(PLE_PFX + "ngram_embedding.")}
        rows_k = sum(v[1][0] for n, v in k_t.items())
        rows_m = sum(v[1][0] for n, v in m_t.items() if n.endswith(".weight"))
        tc_k, tc_m = self.kcfg["text_config"], self.mcfg["text_config"]
        layout_keys = ("ngram_vocab_size_base", "heads_per_ngram", "ngram_size", "split_ngram_parts", "ple_embed_dim",
                       "make_ngram_vocab_size_divisible_by", "ple_layer_ids", "vocab_size", "eos_token_id")
        lay = {k: [tc_k.get(k), tc_m.get(k)] for k in layout_keys}
        bufs = self.ple["buffers"]
        buf_eq = {b: bufs.get(f"klee100:{b}", {}).get("values") == bufs.get(f"merlin:{b}", {}).get("values")
                  for b in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")}
        ov = {"files_present": overlay_files, "headers": overlay_headers}
        if overlay_headers:
            ovt = {}
            for f, h in overlay_headers.items():
                for n, v in h.items():
                    if n != "__metadata__":
                        ovt[n] = (v["dtype"], tuple(v["shape"]), f)
            ov["tensors"] = len(ovt)
            ov["equal_to_merlin_index"] = {n: ovt.get(n) == m_t[n] for n in m_t} if ovt else None
            ov["all_equal_to_merlin_index"] = bool(ovt) and all(ovt.get(n) == m_t[n] for n in m_t) and len(ovt) == len(m_t)
        rec = {"klee100": {"tensors": len(k_t), "dtypes": dict(collections.Counter(v[0] for v in k_t.values())),
                           "shape": sorted({tuple(v[1]) for v in k_t.values()}), "rows_total": rows_k,
                           "bytes": sum(nbytes(v[0], v[1]) for v in k_t.values()), "shard_file": sorted({v[2] for v in k_t.values()}),
                           "base_revision": self.build_info.get("base_revision"), "base_model": self.build_info.get("base_model")},
               "merlin_fp8": {"tensors": len(m_t), "dtypes": dict(collections.Counter(v[0] for v in m_t.values())),
                              "shape": sorted({tuple(v[1]) for v in m_t.values() if len(v[1]) == 2}), "rows_total": rows_m,
                              "bytes": sum(nbytes(v[0], v[1]) for v in m_t.values()), "files": sorted({v[2] for v in m_t.values()}),
                              "weight_scale": self.ple["merlin"]["weight_scale"],
                              "provenance": self.facts["fp8_ple_provenance"]},
               "layout_config_equal": {k: v[0] == v[1] for k, v in lay.items()}, "layout_config": lay,
               "hash_buffers_equal": buf_eq, "overlay": ov}
        rec["ok"] = (len(k_t) == 128 and set(rec["klee100"]["dtypes"]) == {"BF16"} and len(m_t) == 129
                     and rec["merlin_fp8"]["dtypes"] == {"F8_E4M3": 128, "BF16": 1} and rows_k == rows_m
                     and rec["klee100"]["shape"] == rec["merlin_fp8"]["shape"] and all(rec["layout_config_equal"].values())
                     and all(buf_eq.values()))
        return rec

    # ------------------------------------------------------------------ Gate C: MTP
    def gate_c(self, mods_k, a_rows, cfg_fix):
        kt = self.tabs["klee100"]
        mtp_keys = [n for n in kt if n.startswith("mtp.")]
        layers = sorted({int(m.group(1)) for n in mtp_keys for m in [re.match(r"mtp\.layers\.(\d+)\.", n)] if m})
        remap = {n: map_key(n) for n in mtp_keys}
        draft_mods = sorted({r["module"] for r in remap.values() if "module" in r})
        elems = sum(math.prod(kt[n][1]) for n in mtp_keys)
        logical = 0
        for n in mtp_keys:
            dt, shp, _ = kt[n]
            if n.endswith(".qweight"):
                base = n.rpartition(".")[0]
                rec = self._fk.get(base, {})
                logical += rec.get("N", 0) * rec.get("K", 0)
            elif n.endswith((".qzeros", ".scales")):
                continue
            else:
                logical += math.prod(shp)
        rowsd = [r for r in a_rows["rows"] if r["side"] == "draft"]
        need = {f"mtp.layers.{NL}.mlp.experts": (4, 128, "marlin"), f"mtp.layers.{NL}.self_attn.qkv_proj": (8, 128, "marlin"),
                f"mtp.layers.{NL}.self_attn.o_proj": (8, 128, "marlin"), "mtp.fc_embedding": (8, 128, "marlin"),
                "mtp.fc_hidden": (8, 128, "marlin")}
        got = {r["module"]: r for r in rowsd}
        res_ok = {m: bool(got.get(m) and got[m]["ok"] and (got[m].get("resolved") or [None])[0] == b
                          and got[m]["family"] == fam) for m, (b, g, fam) in need.items()}
        tc = cfg_fix["text_config"]
        rec = {"mtp_tensors": len(mtp_keys), "build_info_mtp_tensor_count": self.validation.get("mtp_tensor_count"),
               "checkpoint_mtp_layers": layers, "runtime_prefix": f"mtp.layers.{NL}", "mtp_num_hidden_layers": tc.get("mtp_num_hidden_layers"),
               "remap_rule": "checkpoint mtp.layers.0.X -> param path model.layers.0.X (ModuleList index 0) under the "
                             "module prefix mtp.layers.48 (mtp.py _remap_mtp_weight_name + Qwen4ExpMultiTokenPredictor "
                             "prefix layers.{num_hidden_layers + idx}); quantization lookups use the prefix, so the config "
                             "keys must say mtp.layers.48 (the author patches _make_draft_vllm_config to remap extra_config)",
               "draft_modules": len(draft_mods), "remap_samples": {n: remap[n].get("module") for n in sorted(mtp_keys)[:4]},
               "dropped_mtp_keys": [n for n, r in remap.items() if "dropped" in r][:10],
               "stored_elements": elems, "logical_parameters": logical,
               "embed_tokens": "no mtp.embed_tokens key (mtp_use_dedicated_embeddings=%s): the draft loads "
                               "model.language_model.embed_tokens.weight into its own embed_tokens; under PP2 it is NOT "
                               "shared (llm_base_proposer._maybe_share_embeddings shares only when PP world size is 1), "
                               "so PP1 holds a second copy; both copies are pinned host tables (VLLM_HOST_EMBED_TABLE=1, Merlin "
                               "launch.py:385; mtp.py:239-249 PinnedHostVocabParallelEmbedding), not GPU memory"
                               % tc.get("mtp_use_dedicated_embeddings"),
               "lm_head": "no mtp head key; Qwen4ExpMTP builds ParallelLMHead(lm_head) and the proposer replaces it with the "
                          "target lm_head (MTP branch of _maybe_share_lm_head) -> shared, no extra memory",
               "resolved": {m: {"ok": v, "row": {k: got.get(m, {}).get(k) for k in ("method", "resolved", "how")}} for m, v in res_ok.items()},
               "draft_rows_bad": [r["module"] for r in rowsd if not r["ok"]],
               "mtp_runtime_active": {"author_serving_report": {k: self.serving.get(k) for k in
                                                                ("server", "mtp_runtime_active", "mtp_runtime_parameter_count",
                                                                 "mtp_checkpoint_tensor_count")},
                                      "meaning": "the author's HF smoke ran the Transformers fallback with MTP inactive; the "
                                                 "author's vLLM serve.sh passes --speculative-config mtp K=1; Merlin activates "
                                                 "the draft only through --speculative-config (RC2: method mtp, K3), so MTP "
                                                 "weights are loaded and used only when that flag is set; nothing in the "
                                                 "checkpoint config enables or disables it"}}
        rec["ok"] = (len(mtp_keys) == (self.validation.get("mtp_tensor_count") or -1) and layers == [0]
                     and not rec["dropped_mtp_keys"] and all(res_ok.values()) and not rec["draft_rows_bad"])
        return rec

    # ------------------------------------------------------------------ Gate D: memory (ESTIMATE)
    def per_rank(self, tab, lm_only, quant_cfg_is_inc):
        mods, _ = build_modules(tab, lm_only)
        out = {"PP0": 0.0, "PP1": 0.0}
        host, host_embed = 0, 0
        unclassified = []
        for (side, mod), r in mods.items():
            tp = r["tp"]
            if tp is None:
                unclassified.append(mod)
                continue
            if tp == "host":
                host += r["bytes"]
                continue
            if tp == "host_embed":                       # Merlin launch.py:385 VLLM_HOST_EMBED_TABLE=1: pinned host table
                host_embed += r["bytes"] * (2 if mod == "language_model.model.embed_tokens" else 1)   # + the draft's own copy
                continue
            b = 0.0
            for key in r["keys"]:
                dt, shp, _ = tab[key]
                nb = nbytes(dt, shp)
                if tp in ("ep", "col", "row", "vocab"):
                    nb /= 2
                elif tp == "qkvidx":
                    nb = nb if ".indexer.index_qk_proj." in key else nb / 2
                elif tp == "ba":
                    nb = nb if quant_cfg_is_inc else nb / 2         # maybe_disable_tp: INC/GPTQ/AWQ -> replicated
                b += nb
            out[r["stage"]] += b
        return out, (host, host_embed), unclassified

    def gate_d(self, tab_k, profile):
        F = self.facts["rc2_calibration"]
        BLOCK = F["block_bytes"]
        res = {"label": "ESTIMATE (header bytes x sharding rules, calibrated on RC2's measured weights; never measured)",
               "layout": {"tp": 2, "pp": 2, "ep": True, "partition": self.facts["pp_partition"],
                          "source": self.facts["pp_partition_source"]},
               "rc2": F, "merlin_profile_static": profile, "variants": {}}
        for lm_only in (False, True):
            m_pc, m_host, m_unc = self.per_rank(self.tabs["merlin"], lm_only, False)
            k_pc, k_host, k_unc = self.per_rank(tab_k, lm_only, True)
            tag = "language_model_only" if lm_only else "with_vision"
            freed = {s: m_pc[s] - k_pc[s] for s in m_pc}
            est = {s: F["weights_gib"][s] * GiB - freed[s] for s in m_pc}
            blocks = F["blocks"] + min(math.floor(freed[s] / BLOCK) for s in m_pc)
            blocks_c5 = F["blocks"] + min(math.floor((freed[s] + (F["min_free_mib"][s] - 512) * MiB) / BLOCK) for s in m_pc)
            pin_rule = {}
            if profile and profile.get("overhead_gib") is not None:
                vram = profile.get("card_vram_gib", 22.03)
                pin_rule = {s: round(vram - est[s] / GiB - profile["overhead_gib"], 3) for s in est}
            streams = {str(L): {"blocks_per_request": math.ceil(L / 4096) + F["F"],
                                "resident": blocks // (math.ceil(L / 4096) + F["F"]),
                                "resident_with_c5": blocks_c5 // (math.ceil(L / 4096) + F["F"])} for L in (68000, 90000, 120000)}
            res["variants"][tag] = {
                "merlin_computed_gib": {s: round(m_pc[s] / GiB, 3) for s in m_pc},
                "merlin_calibration_delta_gib": {s: round(F["weights_gib"][s] - m_pc[s] / GiB, 3) for s in m_pc},
                "klee100_computed_gib": {s: round(k_pc[s] / GiB, 3) for s in k_pc},
                "klee100_est_gpu_weights_gib": {s: round(est[s] / GiB, 3) for s in est},
                "freed_vs_merlin_gib": {s: round(freed[s] / GiB, 3) for s in freed},
                "ple_host_bytes": {"merlin": m_host[0], "klee100_graft": k_host[0]},
                "host_embed_bytes_target_plus_draft_copy": {"merlin": m_host[1], "klee100": k_host[1],
                                                            "note": "VLLM_HOST_EMBED_TABLE=1 (launch.py:385): embed tables in pinned "
                                                                    "host RAM; the draft's copy is separate under PP2"},
                "kv_blocks_4096_est": blocks, "kv_blocks_4096_est_with_c5_pin": blocks_c5,
                "merlin_profile_pin_rule_gib": pin_rule, "resident_streams_est": streams,
                "unclassified": {"merlin": m_unc, "klee100": k_unc}}
        v = res["variants"]["with_vision"]
        cal = v["merlin_calibration_delta_gib"]
        res["ok"] = (not v["unclassified"]["merlin"] and not v["unclassified"]["klee100"]
                     and all(abs(x) <= 0.6 for x in cal.values()) and v["kv_blocks_4096_est"] >= F["blocks"]
                     and v["ple_host_bytes"]["klee100_graft"] == v["ple_host_bytes"]["merlin"])
        return res

    # ------------------------------------------------------------------ negative tests
    def _m3_isolated(self, tab, label):
        """manifests_13 on a modified table without touching the state the real run keeps (_mods_k / _fk / _fm)."""
        keep = (self.__dict__.get("_mods_k"), self.__dict__.get("_fk"), self.__dict__.get("_fm"))
        try:
            return self.manifests_13(tab, label=label)[1]
        finally:
            self._mods_k, self._fk, self._fm = keep

    def negative_rename_key(self, tab):
        t = dict(tab)
        victim = "model.language_model.layers.7.mlp.experts.3.gate_proj.qweight"
        t[victim.replace(".qweight", ".qweight_renamed")] = t.pop(victim)
        m3 = self._m3_isolated(t, "NEGATIVE_renamed_" + victim)
        return {"renamed": victim, "verdict": m3["verdict"], "unexpected": m3["unexpected"], "missing": m3["missing"],
                "unmapped_params": m3["unmapped_params"][:3], "missing_params": m3["missing_params"][:3],
                "ok": m3["verdict"] == "NO_GO_UNMAPPED_WEIGHT"}

    def negative_expert_ids(self, tab):
        """K4 regression tests on the real table: each mutation must fail M3 with the stated verdict and evidence."""
        P = "model.language_model.layers."
        cases = {}

        def run(name, t, want_verdict, want_text):
            m3 = self._m3_isolated(t, "NEGATIVE_" + name)
            errs = m3["expert_id_coverage"]["errors"] + m3["missing_params"]
            hit = any(want_text in e for e in errs)
            cases[name] = {"verdict": m3["verdict"], "want_verdict": want_verdict, "evidence_found": hit,
                           "expert_id_errors": m3["expert_id_coverage"]["errors"][:4], "missing_params": m3["missing_params"][:4],
                           "ok": m3["verdict"] == want_verdict and hit}
        # 1. expert 0 renamed to 512 in every tensor of one layer (tensor counts unchanged)
        t = dict(tab)
        for k in [k for k in t if k.startswith(P + "3.mlp.experts.0.")]:
            t[k.replace(".experts.0.", ".experts.512.", 1)] = t.pop(k)
        run("expert_0_renamed_512", t, "NO_GO_EXPERT_ID_SET", "layers.3.mlp.experts: gate_proj.qweight ids n=512")
        # 2. one expert's qzeros missing (layer 20, expert 100, down_proj)
        t = dict(tab)
        t.pop(P + "20.mlp.experts.100.down_proj.qzeros")
        run("missing_qzeros", t, "NO_GO_MISSING_WEIGHT", "layers.20.mlp.experts: down_proj.qzeros ids n=511")
        # 3. the MTP expert bank missing
        t = {k: v for k, v in tab.items() if not k.startswith("mtp.layers.0.mlp.experts.")}
        run("missing_mtp_bank", t, "NO_GO_MISSING_WEIGHT", f"mtp.layers.{NL}.mlp.experts: MoE module missing")
        # 4. duplicate id: expert 9 of layer 30 renamed to '07' (parses as id 7) -> 7 twice, 9 missing, counts unchanged
        t = dict(tab)
        for k in [k for k in t if k.startswith(P + "30.mlp.experts.9.")]:
            t[k.replace(".experts.9.", ".experts.07.", 1)] = t.pop(k)
        run("duplicate_expert_id", t, "NO_GO_EXPERT_ID_SET", "layers.30.mlp.experts: gate_proj.qweight ids n=512 duplicate [7]")
        return {"cases": cases, "ok": all(c["ok"] for c in cases.values()) and len(cases) == 4}

    def negative_config_key(self, cfg_fix, mods_k, intended):
        c = copy.deepcopy(cfg_fix)
        ec = c["quantization_config"]["extra_config"]
        k = "language_model.model.layers.5.mlp.experts"
        ec["language_model.model.layer.5.mlp.experts"] = ec.pop(k)
        rows = self.static_quant(c, mods_k)
        ga = self.gate_a_rows(rows, intended, mods_k, "NEGATIVE_config_key_renamed")
        return {"renamed_config_key": k, "gate_a_ok": ga["ok"], "bad": ga["bad"][:3], "ok": (not ga["ok"]) and
                any("layers.5.mlp.experts" in b for b in ga["bad"]), "cfg": c}
