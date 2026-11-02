"""E25 klee100 low-bit CPU kit builder (structural track S1: can Merlin v3.0.0 load klee100 3bpw-MTP correctly?).

One Kaggle CPU kernel (pinned Py3.12 image, no GPU, no internet). It decides fail-closed whether
klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP can be loaded correctly by Merlin v3.0.0 in TP2xPP2+EP (25/23) with a
config-only "fixkeys" overlay plus Merlin's FP8 PLE grafted in, and prints exactly one line
`@@KIT klee100 ALL OK@@` or `@@KIT klee100 NOT OK: <items>@@`.

  Manifest 1  module map: every Merlin module (48 decoder layers, MTP layer 48, embeddings / heads / PLE / vision)
              under its Merlin name, its class, whether it is built with a quant config, its TP / PP placement and
              the checkpoint keys it loads (patterns + sha256 of the exact list).
  Manifest 2  quantization map: per module the method Merlin resolves under the fixkeys overlay. REAL mode =
              Merlin's own from_config / configure_quant_config / get_quant_method / RoutedExperts on meta with a
              CUDA (8, 9) platform patch (k100kit_real.py in the Merlin venv, libcuda stub on LD_LIBRARY_PATH);
              STATIC mode = pure-Python mirror of the same rules (k100kit_core.StaticINC), labelled, fallback only.
  Manifest 3  weight coverage: expected (the RC2-proven Merlin module / key set of the same model class) vs provided
              (klee100 + FP8 PLE graft): modules, params, shapes vs architecture dims, dtypes, packed axes, group sizes,
              scales / zero-points; plus the real-data probe of layers 0 (W3) and 12 (W2) from the private dataset; plus
              a negative test (one renamed quantization key -> NO_GO_UNMAPPED_WEIGHT).
  Gate A      quantization: all 48 MoE layers + MTP + every quant-config linear resolve (REAL mode) to the intended
              bits / group / method family (humming W3 g128 0-11 & 36-47, humming W2 g64 12-35, Marlin W4 g128 MTP
              experts, Marlin W8 g128 main projections, BF16 elsewhere); unexpected quantized modules = 0; a renamed
              config key must break it (negative test).
  Gate B      PLE provenance: overlay metadata vs Merlin's index vs klee100's BF16 table (shapes, rows, layout config,
              hash buffers, base revision); numeric FP8-dequant vs BF16 on 262 sampled rows read from the MOUNTED
              overlay (cosine, max abs / rel error, shifted-row control, permutation negative test); the grafted index
              references no klee100 BF16 PLE shard; Merlin's PLE method resolves FP8 and its guard accepts F8_E4M3.
  Gate C      MTP: tensors present, remap mtp.layers.0 -> mtp.layers.48, embed / lm_head handling, parameter count,
              W4 experts + W8 dense resolved, mtp_runtime_active handling recorded.
  Gate D      memory: per-rank inventory for TP2xPP2+EP 25/23 from the real headers, calibrated on RC2, KV blocks at
              4096-token blocks and resident streams. ESTIMATE, never measured.
Artifacts: the fixkeys overlay (config.json, quantization_config.json, grafted model.safetensors.index.json,
FIXKEYS_DIFF.json) under /kaggle/working/fixkeys_overlay, the three manifests, k100kit_report.json,
klee100_shard_digests.json.

v2 (2026-10-31, audit items K1-K5, K7 + the v1 CPU limits; PENDING_BATCH_FIXES.md):
  K1  a RoutedExperts row with build_error / stub build / stub_error / error -> ERROR:MOE_CONSTRUCTION_NOT_PROVEN;
      cuInit inside humming frames (both the real build and the stub) -> GPU_ONLY_HUMMING_OBJECT: resolution judged
      on bits / group / family, construction reported as GPU-only (never OK); linear rows need a passing create_weights
      representative of their kind (side, class, format, bits, group), PLE rows the FP8 method + guard rule.
  K2  real parameters for EVERY expected MoE module (48 + MTP bank; one meta module per layer, released), statuses
      M3_STATIC_OK / M3_SAMPLED_REAL_OK / M3_COMPLETE_REAL_OK.
  K3  streaming sha256 of every mounted FP8 PLE shard vs PLE_EXPECTED.
  K4  exact expert-id sets per (projection, suffix) for the 48 target layers + the MTP bank, with 4 negative tests.
  K5  ALL_OK = CPU_COMPATIBILITY_PASS (eligible for a GPU load attempt), never "all tensors loaded".
  K7  HF LFS digest manifest (HEAD x-linked-etag + tree API) for all 13 klee100 shards and the 10 FP8 PLE shards.
  CPU: the TP / PP groups are initialised inside set_current_vllm_config(VllmConfig()) (v1 asserted there).

Inputs (data only, never executed): D:/ARC AGI 2/_scratch/hf_lowbit_configs/{klee100,halt95}/... (configs, index,
every shard's safetensors header), D:/ARC AGI 2/_scratch/klee100_ple_rows_2026-10-31/ (262 PLE rows by HTTP Range,
both checkpoints), D:/ARC AGI 2/_scratch/klee100_author_patch_2026-10-31/ (the author's patch, read only), the Merlin
v3.0.0 source tree D:/ARC AGI 2/_scratch/e23/p2/merlin-src/ (rules + file sha256).

Run:  python build_e25_klee100_cpu_kit.py            -> push folder + local quick check (core only, seconds)
"""
import base64
import collections
import hashlib
import json
import lzma
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCR = Path(r"D:/ARC AGI 2/_scratch")
HF = SCR / "hf_lowbit_configs"
KDIR = HF / "klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
MDIR = HF / "halt95/Qwen3.8-Flash-Next-W4A16-Merlin"
PLE_DIR = SCR / "klee100_ple_rows_2026-10-31"
PATCH_DIR = SCR / "klee100_author_patch_2026-10-31"
MSRC = SCR / "e23/p2/merlin-src"
ROOT = SCR / "push_e25"
SLUG = "q27-e25-klee100-cpu-kit"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODELS = ["foysalemonshanto/e23-merlin-env/Other/v1/1",
          "foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/Transformers/default/1",
          "foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1"]
DATASETS = ["foysalemonshanto/q27-e25-klee100-layers-0-12"]
K_REV = "600715b31bed3833fb4a51d9ea5d684ea2c7f32c"       # klee100 main at fetch time (2026-10-31); headers re-verified
M_REV = "cec53e9f90e5995b81691604ec0b32191debf797"       # Merlin W4A16 revision our Kaggle p1-p5 Models mirror
SRC_FILES = [
    "vllm/model_executor/layers/quantization/inc/config_parser.py", "vllm/model_executor/layers/quantization/inc/inc.py",
    "vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py",
    "vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py",
    "vllm/model_executor/layers/quantization/humming.py",
    "vllm/models/qwen4_exp/nvidia/model.py", "vllm/models/qwen4_exp/nvidia/mtp.py",
    "vllm/models/qwen4_exp/nvidia/ngram_embedding.py", "vllm/models/qwen4_exp/nvidia/ple_format_guard.py",
    "vllm/models/qwen4_exp/nvidia/ple_layer.py", "vllm/models/qwen4_exp/nvidia/qsa.py",
    "vllm/models/qwen4_exp/nvidia/hyperconnection.py", "vllm/models/qwen4_exp/nvidia/indexer_qsa.py",
    "vllm/models/qwen4_exp/common/ngram_embedding.py",
    "vllm/model_executor/models/qwen3_5.py", "vllm/model_executor/models/qwen3_vl.py",
    "vllm/model_executor/models/qwen3_next.py", "vllm/model_executor/models/qwen2_moe.py",
    "vllm/model_executor/models/utils.py", "vllm/model_executor/model_loader/utils.py",
    "vllm/model_executor/layers/fused_moe/routed_experts.py",
    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", "vllm/v1/spec_decode/llm_base_proposer.py",
    "vllm/distributed/utils.py", "tools/merlin/profiles/flash-next.json", "tools/merlin/launch.py"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pack(b):
    return base64.b85encode(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


def norm_sha(p):
    return sha(Path(p).read_bytes().replace(b"\r\n", b"\n"))


# ------------------------------------------------------------------------------------------- data derivation
def encode_table(d, repo, rev):
    man = json.loads((d / "_fetch_manifest.json").read_text())
    H, hsha = {}, {}
    for f, meta in man["headers"].items():
        b = (d / "_headers" / (f + ".header.json")).read_bytes()
        assert sha(b) == meta["sha256"], f
        hsha[f] = meta["sha256"]
        for k, v in json.loads(b).items():
            if k != "__metadata__":
                H[k] = (v["dtype"], tuple(v["shape"]), f)
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    assert set(wm) == set(H) and all(wm[k] == H[k][2] for k in H), repo
    shards = sorted({v[2] for v in H.values()})
    groups = collections.defaultdict(list)
    for k, (dt, shp, f) in H.items():
        nums = tuple(int(x) for x in re.findall(r"(?<=\.)\d+(?=\.|$)", k))
        tpl = re.sub(r"(?<=\.)\d+(?=\.|$)", "{}", k)
        groups[(tpl, dt, shp, shards.index(f))].append(nums)

    def runs(v):
        v = sorted(v)
        out, s, p = [], v[0], v[0]
        for x in v[1:]:
            if x == p + 1:
                p = x
                continue
            out.append([s, p])
            s = p = x
        out.append([s, p])
        return out
    enc = []
    for (tpl, dt, shp, f), lst in sorted(groups.items()):
        ar = len(lst[0])
        assert all(len(x) == ar for x in lst) and ar <= 2, (tpl, ar)
        if ar == 0:
            enc.append([tpl, dt, list(shp), f, None])
        elif ar == 1:
            enc.append([tpl, dt, list(shp), f, runs([x[0] for x in lst])])
        else:
            by = collections.defaultdict(list)
            for a, b in lst:
                by[a].append(b)
            inv = collections.defaultdict(list)
            for a, bs in by.items():
                inv[json.dumps(runs(bs))].append(a)
            enc.append([tpl, dt, list(shp), f, [[runs(as_), json.loads(r)] for r, as_ in inv.items()]])
    h = hashlib.sha256()
    for n in sorted(H):
        dt, shp, f = H[n]
        h.update(f"{n}\t{dt}\t{list(shp)}\t{f}\n".encode())
    return {"repo": repo, "rev": rev, "n_tensors": len(H), "canon_sha256": h.hexdigest(), "shards": shards, "groups": enc,
            "index_metadata": idx.get("metadata"), "index_sha256": man["files"]["model.safetensors.index.json"]["sha256"],
            "header_sha256": hsha}


def merlin_facts():
    prof = json.loads((MSRC / "tools/merlin/profiles/flash-next.json").read_text())
    p = next(x for x in prof["profiles"] if x.get("id") == "fn-quad-24g-sm89")
    srv, cap = p["serve"], p.get("capacity", {})
    kb = srv.get("kv_alt", {}).get("bfloat16", {}).get("capacity", {})
    m = re.search(r"([\d.]+) GiB overhead", kb.get("basis", ""))
    for f in SRC_FILES:
        assert (MSRC / f).is_file(), f
    # spot-check the source facts the static rules encode (fail the build if the tree differs)
    src = lambda f: (MSRC / f).read_text(encoding="utf-8")
    assert '"fusedmoe" in layer.__class__.__name__.lower()' in src(SRC_FILES[0])
    assert 'CUDA_HUMMING_SUPPORTED_BITS = {2, 3, 5, 6, 7}' in src(SRC_FILES[2])
    assert 'if embedding_dtype == "float8_e4m3fn":' in src("vllm/models/qwen4_exp/common/ngram_embedding.py")
    assert '"qkv_proj": ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"]' in src("vllm/models/qwen4_exp/nvidia/model.py")
    assert 'prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}"' in src("vllm/models/qwen4_exp/nvidia/mtp.py")
    assert 'quant_config=None,\n            prefix=f"{prefix}.shared_expert_gate"' in src("vllm/model_executor/models/qwen3_next.py")
    assert "if get_pp_group().world_size == 1:" in src("vllm/v1/spec_decode/llm_base_proposer.py")
    assert 'ple_embedding_dtype") == "float8_e4m3fn"' in src("tools/merlin/launch.py")
    return {
        "merlin_version": "v3.0.0 (E23 merlin_offline_tar source tree)",
        "source_files_sha256_lf": {f: norm_sha(MSRC / f) for f in SRC_FILES},
        "cg_rename_prefixes": [["model.visual.", "visual."], ["lm_head.", "language_model.lm_head."],
                               ["model.language_model.", "language_model.model."]],
        "cg_rename_source": "qwen3_vl.py:1987-1993 (Qwen3VLForConditionalGeneration.hf_to_vllm_mapper) | qwen3_5.py:525-528 "
                            "(mtp.: None, removed by get_rename_mapper) -> configure_quant_config (model_loader/utils.py:361-381)",
        "packed_target": {"qkv_proj": ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"], "gate_up_proj": ["gate_proj", "up_proj"],
                          "qkv": ["qkv"], "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"],
                          "kv_proj": ["key_proj", "value_proj"],
                          "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight", "_input_mix_padding"]},
        "packed_target_source": "qwen3_vl.py:1967-1978 | qwen3_5.py:530-533 | nvidia/model.py:1171-1180",
        "packed_draft": {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"],
                         "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"],
                         "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight", "_input_mix_padding"]},
        "packed_draft_source": "nvidia/mtp.py:498-508 (Qwen4ExpMTP has no hf_to_vllm_mapper; _make_draft_vllm_config "
                               "mtp.py:160-200 remaps ignored_layers / exclude_modules / quantized_layers to layers.48 but NOT "
                               "INC extra_config / block_name_to_quantize)",
        "quant_config_none_modules": "GateLinear mlp.gate (qwen3_next.py SparseMoeBlock), shared_expert_gate (quant_config=None), "
                                     "all HyperConnection linears (hyperconnection.py:100-134), GDN conv1d, embed_tokens, "
                                     "lm_head (ParallelLMHead without quant_config), norms",
        "pp_partition": p["layout"]["partition"], "pp_partition_source": "tools/merlin/profiles/flash-next.json fn-quad-24g-sm89 layout",
        "profile_static": {"id": p["id"], "kv_cache_memory_bytes": srv.get("kv_cache_memory"), "max_num_seqs": srv.get("max_num_seqs"),
                           "capture_sizes": srv.get("capture_sizes"), "pin_gib": cap.get("pin_gib"),
                           "bytes_per_block_mib": cap.get("bytes_per_block_mib"),
                           "overhead_gib": float(m.group(1)) if m else None,
                           "overhead_source": "kv_alt.bfloat16.capacity.basis: 'Pin = card VRAM - binding-stage weights - 2.75 GiB overhead'",
                           "card_vram_gib": 22.49, "card_vram_note": "L4 23034 MiB (RC2 initial free 21.67 GiB)",
                           "pin_rule": "preflight.py:920-930 kv_bytes_for_vram: pin = kv_cache_memory - (24 GiB class - card GiB); "
                                       "L4: 4.1e9 - 1.51 GiB = 2,479,838,700 B = RC2's pin (150 blocks). The rule ignores the "
                                       "weights, so klee100's freed bytes become KV only with an explicit larger "
                                       "--kv-cache-memory-bytes (the block estimate below assumes that)",
                           "graphs_measured_rc2_gib": [0.22, 0.43]},
        "rc2_calibration": {"weights_gib": {"PP0": 17.33, "PP1": 16.87}, "min_free_mib": {"PP0": 1607, "PP1": 1633}, "blocks": 149,
                            "block_bytes": 2479838700 / 150, "F": 6,
                            "source": "E13 RC2 GPU v1 (Model loading took ...; --kv-cache-memory-bytes 2479838700 = 150 blocks, "
                                      "149 usable); humming kit H4 calibrated Merlin within -0.44 / +0.18 GiB"},
        "fp8_ple_provenance": {"kaggle_model": "foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
                               "hf_source": "albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE@c6af9fee7ad2cb8ef44b534a99b8c1b81fb0f6da",
                               "same_lfs_oids_as": f"halt95/Qwen3.8-Flash-Next-W4A16-Merlin@{M_REV} model-plefp8-0000{{0..9}}",
                               "origin_claim": "the official RadixArk FP8 n-gram/PLE table + weight_scale (E3 FN_FP8PLE build notes)",
                               "base_revision_recorded": None,
                               "note": "the shards' safetensors __metadata__ is empty; the kit scans the overlay's README / "
                                       "config for a revision string and records what it finds"},
        "ple_expected_shards": json.loads(re.search(r'PLE_EXPECTED = json.loads\(r"""(.*?)"""\)',
                                                    (HERE / "nb3_code_export.py").read_text(encoding="utf-8")).group(1)),
    }


DIGEST_CACHE = SCR / "klee100_shard_digests_2026-10-31" / "shard_digests.json"


def _hf_head(repo, rev, f):
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    op = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{rev}/{f}", method="HEAD")
    try:
        r = op.open(req, timeout=60)
        h = r.headers
    except urllib.error.HTTPError as e:          # the 302 to the CDN carries the LFS headers
        h = e.headers
    return {"head_sha256": (h.get("X-Linked-ETag") or "").strip('"'), "head_size": int(h.get("X-Linked-Size") or -1),
            "head_repo_commit": h.get("X-Repo-Commit")}


def _hf_tree(repo, rev):
    import urllib.request
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true", timeout=60) as r:
        return {x["path"]: x for x in json.loads(r.read()) if x.get("type") == "file"}


def shard_digests(refetch=False):
    """K7: trusted full-shard digests from HF LFS, two independent HF answers per file (HEAD x-linked-etag /
    x-linked-size at the pinned revision, and the tree API lfs.oid / size), cached so the build is reproducible."""
    if DIGEST_CACHE.is_file() and not refetch:
        return json.loads(DIGEST_CACHE.read_text())
    import datetime
    out = {"source": "huggingface.co LFS: HEAD /resolve/<rev>/<file> (X-Linked-ETag = sha256, X-Linked-Size) + "
                     "/api/models/<repo>/tree/<rev> (lfs.oid, size); both must agree",
           "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    kman = json.loads((KDIR / "_fetch_manifest.json").read_text())
    for tag, repo, rev, pick in (("klee100", "klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP", K_REV,
                                  lambda p: p.endswith(".safetensors") or p in ("model.safetensors.index.json", "tokenizer.json")),
                                 ("merlin_fp8_ple", "halt95/Qwen3.8-Flash-Next-W4A16-Merlin", M_REV,
                                  lambda p: p.startswith("model-plefp8-"))):
        tree = _hf_tree(repo, rev)
        rec = {"repo": repo, "revision": rev, "shards": {}, "other_lfs_files": {}}
        for p in sorted(x for x in tree if pick(x)):
            lfs = tree[p].get("lfs") or {}
            v = {"tree_sha256": lfs.get("oid"), "tree_size": lfs.get("size", tree[p].get("size")), **_hf_head(repo, rev, p)}
            assert v["head_sha256"] == v["tree_sha256"] and v["head_size"] == v["tree_size"], (repo, p, v)
            assert v["head_repo_commit"] == rev, (repo, p, v["head_repo_commit"])
            v.update(sha256=v["head_sha256"], size=v["head_size"])
            if tag == "klee100" and p in kman["headers"]:
                assert kman["headers"][p]["file_size"] == v["size"], (p, kman["headers"][p], v)
            (rec["shards"] if p.endswith(".safetensors") else rec["other_lfs_files"])[p] = v
        out[tag] = rec
    DIGEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DIGEST_CACHE.write_text(json.dumps(out, indent=1))
    return out


def author_patch():
    p = (PATCH_DIR / "patches/qwen38-ple-ssd.patch").read_text(encoding="utf-8")
    hunks = {}
    for f in ("vllm/model_executor/layers/quantization/inc/config_parser.py", "vllm/models/qwen4_exp/nvidia/mtp.py",
              "vllm/models/qwen4_exp/nvidia/ngram_embedding.py", "vllm/model_executor/model_loader/default_loader.py"):
        i = p.index(f"diff --git a/{f} ")
        j = p.find("\ndiff --git ", i + 10)
        hunks[f] = p[i:j if j > 0 else None][:4000]
    return {"repo": "https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx", "commit": "c7fe762c415c9f53fabccdadc25be949eb92052f",
            "patch_sha256": sha((PATCH_DIR / "patches/qwen38-ple-ssd.patch").read_bytes()),
            "vllm_base": "a5a30471ff2bb7f0824f2da10e358af98d304472", "hunks": hunks,
            "items": [
                {"what": "config_parser: per-expert extra_config overrides apply to RoutedExperts (isinstance check added to the "
                         "'fusedmoe' branch)", "config_expressible": True,
                 "how": "layer-level exact key language_model.model.layers.N.mlp.experts (exact match is checked first in "
                        "_resolve_raw, so the dead 'fusedmoe' branch is never needed)"},
                {"what": "mtp._make_draft_vllm_config: remap extra_config keys mtp.layers.0.* -> mtp.layers.48.*",
                 "config_expressible": True, "how": "write the mtp.layers.48.* keys (and mtp.layers.48.mlp.experts) into extra_config"},
                {"what": "ngram_embedding: INC quant config -> unquantized (BF16) PLE method", "config_expressible": True,
                 "how": "not needed: text_config.ple_embedding_dtype = float8_e4m3fn selects the FP8 method first (Merlin FP8 PLE graft)"},
                {"what": "ple_ssd.py + ple_ssd_io.c + default_loader + model_state: metadata-only BF16 PLE loading, async SSD row "
                         "reads, prompt read-ahead", "config_expressible": False,
                 "how": "not needed in Merlin: the grafted FP8 table is served by Merlin's own pinned-host PLE pull transport"},
                {"what": "breakable_cudagraph / cudagraph_utils: host breaks inside full CUDA graphs for the SSD reads",
                 "config_expressible": False, "how": "not needed (no SSD reads in Merlin's PLE path)"}]}


def build_emb():
    emb = {}
    rec = {}

    def add(rel, b, source=None):
        emb[rel] = [sha(b), pack(b)]
        rec[rel] = {"bytes": len(b), "sha256": sha(b), "packed_chars": len(emb[rel][1]), **({"source": source} if source else {})}
    kman = json.loads((KDIR / "_fetch_manifest.json").read_text())
    for f in ("config.json", "build_info.json", "validation_report.json", "quality_report.json", "serving_report.json"):
        b = (KDIR / f).read_bytes()
        assert sha(b) == kman["files"][f]["sha256"], f
        add(f"hf/klee100/{f}", b, f"https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP/resolve/{K_REV}/{f}")
    mman = json.loads((MDIR / "_fetch_manifest.json").read_text())
    b = (MDIR / "config.json").read_bytes()
    assert sha(b) == mman["files"]["config.json"]["sha256"]
    add("hf/merlin/config.json", b, f"https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin/resolve/{M_REV}/config.json")
    for who, d, repo, rev in (("klee100", KDIR, "klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP", K_REV),
                              ("merlin", MDIR, "halt95/Qwen3.8-Flash-Next-W4A16-Merlin", M_REV)):
        t = encode_table(d, repo, rev)
        add(f"tables/{who}.json", json.dumps(t, separators=(",", ":")).encode(), f"derived from {repo} index + all shard headers")
    pm = json.loads((PLE_DIR / "ple_rows_manifest.json").read_text())
    assert all(v["same_as_earlier_fetch"] for w in ("klee100", "merlin") for v in pm[w]["headers"].values())
    # trimmed manifest (the FP8 rows themselves are NOT embedded: the kit reads them from the mounted overlay and checks
    # them against the sha256 of the HF halt95@cec53e9f copy fetched here; the local check reads the fetched file)
    slim = {k: pm[k] for k in ("klee100", "merlin", "buffers", "bf16_bin_sha256", "fp8_bin_sha256", "n_rows", "cols")}
    slim["rows"] = [{"shard": r["shard"], "row": r["row"]} for r in pm["rows"]]
    slim["fetch"] = "HTTP Range 2026-10-31 (scratchpad fetch_ple_rows.py): klee100@600715b3 model-00001-of-00011 BF16 rows, "                     "halt95@cec53e9f model-plefp8-0000N FP8 rows; seed 20261031, 2 random rows per shard + first/last of shards 0/63/127"
    add("ple/ple_rows_manifest.json", json.dumps(slim, indent=1).encode(), "derived from the fetch manifest")
    add("ple/klee100_ple_rows_bf16.bin", (PLE_DIR / "klee100_ple_rows_bf16.bin").read_bytes(), "HTTP Range fetch 2026-10-31")
    add("facts/merlin_src_facts.json", json.dumps(merlin_facts(), indent=1).encode(), "Merlin v3.0.0 source (read)")
    add("facts/author_patch.json", json.dumps(author_patch(), indent=1).encode(), "author patch (read, never executed)")
    dg = shard_digests("--refetch-digests" in sys.argv)
    ple_exp = merlin_facts()["ple_expected_shards"]
    assert {f: [v["size"], v["sha256"]] for f, v in dg["merlin_fp8_ple"]["shards"].items()} == ple_exp, "HF digests != PLE_EXPECTED"
    assert len(dg["klee100"]["shards"]) == 13, sorted(dg["klee100"]["shards"])
    add("facts/shard_digests.json", json.dumps(dg, indent=1).encode(), f"HF LFS digests (K7), cache {DIGEST_CACHE}")
    return emb, rec


# ------------------------------------------------------------------------------------------- notebook
HEAD = r'''# E25 klee100 low-bit CPU kit (pinned image, no GPU): see the builder docstring for M1-M3, gates A-D and the rules
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, lzma, math, glob, ctypes, struct
from pathlib import Path
T0 = time.time()
KIT = "klee100"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")
K = Path(os.environ.get("KIT_TMP", "/tmp/k100kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": "E25 klee100 CPU kit", "builder_sha256": "__BSHA__", "python": sys.version, "checks": {}, "failed": []}
def save():
    (W / "k100kit_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def unpack(s):
    return lzma.decompress(base64.b85decode(s))
EMB_BLOBS = __EMB__
EMBD = K / "emb"
emb = {}
for rel, (want, blob) in EMB_BLOBS.items():
    try:
        raw = unpack(blob)
        (EMBD / rel).parent.mkdir(parents=True, exist_ok=True)
        (EMBD / rel).write_bytes(raw)
        emb[rel] = {"bytes": len(raw), "sha256_ok": hashlib.sha256(raw).hexdigest() == want}
    except Exception as e:
        emb[rel] = {"error": f"{type(e).__name__}: {e}"[:300], "sha256_ok": False}
CODE = {"nb3_code_export.py": __NB3__, "k100kit_core.py": __CORE__, "k100kit_real.py": __REAL__, "k100kit_main.py": __MAIN__}
CODE_SHA = __CODESHA__
for n, s in CODE.items():
    raw = unpack(s)
    (K / n).write_bytes(raw)
    emb["code/" + n] = {"bytes": len(raw), "sha256_ok": hashlib.sha256(raw).hexdigest() == CODE_SHA[n]}
REP["embedded"] = {"files": emb, "all_ok": all(v.get("sha256_ok") for v in emb.values())}
log("embedded", json.dumps({k: v.get("sha256_ok") for k, v in emb.items()}))
save()
sys.path.insert(0, str(K))
runpy.run_path(str(K / "k100kit_main.py"), init_globals={"K": K, "W": W, "EMBD": EMBD, "INPUT_ROOT": INPUT_ROOT, "REP": REP,
                                                         "save": save, "log": log, "T0": T0, "KIT": KIT}, run_name="k100main")
'''


def build():
    emb, rec = build_emb()
    code = {n: (HERE / n).read_bytes().replace(b"\r\n", b"\n") for n in ("nb3_code_export.py", "k100kit_core.py", "k100kit_real.py",
                                                                       "k100kit_main.py")}
    for n, b in code.items():
        if n.endswith(".py"):
            compile(b, n, "exec")
    src = HEAD
    bsha = sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
    for k, v in (("__EMB__", json.dumps(emb)), ("__NB3__", json.dumps(pack(code["nb3_code_export.py"]))),
                 ("__CORE__", json.dumps(pack(code["k100kit_core.py"]))), ("__REAL__", json.dumps(pack(code["k100kit_real.py"]))),
                 ("__MAIN__", json.dumps(pack(code["k100kit_main.py"]))),
                 ("__CODESHA__", json.dumps({n: sha(b) for n, b in code.items()})), ("__BSHA__", bsha)):
        src = src.replace(k, v)
    compile(src, "<klee100 kit>", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": "k100kit", "metadata": {}, "outputs": [],
                     "source": src.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, len(data)
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{SLUG}__foysalemonshanto"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SLUG}.ipynb").write_bytes(data)
    meta = {"id": f"foysalemonshanto/{SLUG}", "title": "Q27 E25 Klee100 CPU Kit", "code_file": f"{SLUG}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": DATASETS, "kernel_sources": [],
            "competition_sources": ["arc-prize-2026-arc-agi-2"],      # NB3's import needs the ARC eval file (merlin_install path)
            "model_sources": MODELS, "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    for n, b in code.items():
        rec["code/" + n] = {"bytes": len(b), "sha256": sha(b)}
    (out / "EMBEDDED_FILES.json").write_text(json.dumps(rec, indent=1))
    print("kit", out, "|", len(data), "bytes | sha256", sha(data), flush=True)
    return out, src, rec, data


def local_check_fake():
    """Plumbing test of k100kit_real.py + the REAL merge path with a fake vllm (scratchpad make_fakevllm.py). Says nothing
    about Merlin's behaviour; it only catches glue-code bugs before the Kaggle run."""
    import os
    base = Path(r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/k100kit_fake")
    sp = Path(r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad")
    out, src, rec, data = build()
    base.mkdir(parents=True, exist_ok=True)
    (base / "kit.py").write_text(src, encoding="utf-8")
    env = dict(os.environ, KIT_INPUT=str(base / "input"), KIT_TMP=str(sp / "k100kit_local" / "tmp"), KIT_WORK=str(base / "work"),
               K100_LOCAL="1", K100_LOCAL_FP8_ROWS=str(PLE_DIR / "merlin_ple_rows_fp8.bin"), K100_LOCAL_FAKE_VLLM=str(sp / "fakevllm"),
               K100_LOCAL_TREE=str(MSRC), K100_LOCAL_LAYERS=str(SCR / "ds_klee100_layers"))
    (base / "input").mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(base / "kit.py")], capture_output=True, text=True, env=env, timeout=900)
    (base / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print("FAKE rc", r.returncode, "|", next((ln for ln in r.stdout.splitlines() if ln.startswith("@@KIT")), None), flush=True)
    print(r.stdout[-5000:])
    if r.returncode:
        print(r.stderr[-4000:])


def local_check():
    """Quick local check (seconds): the core on the embedded data, the FP8 rows of the HF copy as the overlay stand-in,
    no Merlin venv (Gate A static only -> expected NOT OK locally, by design)."""
    base = Path(r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/k100kit_local")
    out, src, rec, data = build()
    base.mkdir(parents=True, exist_ok=True)
    (base / "kit.py").write_text(src, encoding="utf-8")
    env = dict(**__import__("os").environ, KIT_INPUT=str(base / "input"), KIT_TMP=str(base / "tmp"), KIT_WORK=str(base / "work"),
               K100_LOCAL="1", K100_LOCAL_FP8_ROWS=str(PLE_DIR / "merlin_ple_rows_fp8.bin"),
               K100_LOCAL_LAYERS=str(SCR / "ds_klee100_layers"))
    (base / "input").mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(base / "kit.py")], capture_output=True, text=True, env=env, timeout=900)
    (base / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print("LOCAL rc", r.returncode, "|", next((ln for ln in r.stdout.splitlines() if ln.startswith("@@KIT")), None), flush=True)
    print(r.stdout[-6000:])
    if r.returncode:
        print(r.stderr[-4000:])


if __name__ == "__main__":
    if "--no-check" in sys.argv:
        build()
    elif "--fake-real" in sys.argv:
        local_check_fake()
    else:
        local_check()
