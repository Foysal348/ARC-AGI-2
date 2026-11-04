"""E25 klee100 CPU kit, orchestration (runs inside the notebook with K, W, EMBD, INPUT_ROOT, REP, save, log, T0, KIT)."""
import collections
import ctypes
import glob
import hashlib
import json
import os
import re
import runpy
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import k100kit_core as C

LOCAL = bool(os.environ.get("K100_LOCAL"))
CHK = REP["checks"]
kit = C.Kit(EMBD, log)


def guard(name, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        CHK.setdefault("errors", {})[name] = {"error": f"{type(e).__name__}: {e}"[:800], "tb": traceback.format_exc()[-2500:]}
        log("ERROR", name, f"{type(e).__name__}: {e}"[:300])
        return None


def jw(name, obj):
    (W / name).write_text(json.dumps(obj, indent=1, default=str))


# ===================================================================== 1. inputs, fixkeys overlay, graft
inputs_ok = bool(guard("load", kit.load)) and REP["embedded"]["all_ok"]
CHK["inputs"] = dict(kit.R["inputs"], embedded_all_ok=REP["embedded"]["all_ok"], ok=inputs_ok)
log("inputs", json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in CHK["inputs"].get("tables", {}).items()}),
    CHK["inputs"].get("ple_rows"))
save()
cfg_fix, diff = kit.fixkeys()
idx_g, tab_g, graft_chk = kit.graft()
overlay_files = kit.write_overlay(W / "fixkeys_overlay", cfg_fix, diff, idx_g)
(W / "fixkeys_overlay" / "config.original.sha256").write_text(C.sha(kit.kcfg_raw) + "  config.json (klee100 as published)\n")
CHK["fixkeys"] = {"diff": diff, "files": overlay_files, "graft": graft_chk,
                  "ok": not diff["problems"] and graft_chk["ok"] and len(diff["extra_config_added_layer_keys"]) == C.NL}
log("fixkeys", json.dumps({"layer_keys": len(diff["extra_config_added_layer_keys"]), "mtp_keys": diff["extra_config_added_mtp_keys_count"],
                           "problems": diff["problems"], "graft_ok": graft_chk["ok"]}))
save()

# ===================================================================== 2. Manifest 1 + 3 (checkpoint side), intended scheme
m1, m3, mods_k, fk = kit.manifests_13(tab_g)
m1_lm, m3_lm, _, _ = kit.manifests_13(tab_g, lm_only=True, label="klee100_fixkeys_graft --language-model-only")
intended = kit.intended(mods_k, fk)
layout = kit.stated_layout_check(intended)
neg_key = kit.negative_rename_key(tab_g)
neg_ids = kit.negative_expert_ids(tab_g)            # K4 regression tests: 0->512, missing qzeros, missing bank, duplicate id
CHK["M3_static"] = {k: v for k, v in m3.items()}
CHK["M3_static_lm_only"] = {k: m3_lm[k] for k in ("verdict", "unexpected", "missing", "modules", "dropped_keys")}
CHK["stated_layout"] = layout
CHK["negative_renamed_tensor_key"] = neg_key
CHK["negative_expert_ids"] = neg_ids
log("M3", m3["verdict"], "unexpected", m3["unexpected"], "missing", m3["missing"], "| expert ids", m3["expert_id_coverage"]["ok"],
    m3["expert_id_coverage"]["checked_modules"], "/", m3["expert_id_coverage"]["expected_modules"], "modules,",
    m3["expert_id_coverage"]["pairs_checked"], "pairs | layout", layout["ok"], "| neg", neg_key["verdict"],
    "| neg ids", {k: v["verdict"] for k, v in neg_ids["cases"].items()}, neg_ids["ok"])
save()


# ===================================================================== 2b. K7 trusted full-shard digest manifest (HF LFS)
def digest_manifest():
    """The builder fetched, for every klee100 shard (11 model incl. the unused BF16 PLE shard 00001 + 2 MTP) and every
    Merlin FP8 PLE shard, the LFS sha256 + size from HF twice (HEAD x-linked-etag / x-linked-size and the tree API
    lfs.oid / size) at the pinned revisions. Here: internal consistency with the tensor tables and PLE_EXPECTED.
    The future GPU notebook streams the sha of every mounted shard against this manifest before boot."""
    D = json.loads((EMBD / "facts" / "shard_digests.json").read_text())
    out = {"source": D.get("source"), "fetched_utc": D.get("fetched_utc"), "repos": {}}
    probs = []
    enc = kit.enc_klee100
    kt = kit.tabs["klee100"]
    by_shard = collections.Counter()
    for n, (dt, shp, f) in kt.items():
        by_shard[f] += C.nbytes(dt, shp)
    kd = D["klee100"]
    if kd.get("revision") != enc["rev"]:
        probs.append(f"klee100 revision {kd.get('revision')} != table rev {enc['rev']}")
    if set(kd["shards"]) != set(enc["shards"]):
        probs.append(f"klee100 shard set {sorted(set(kd['shards']) ^ set(enc['shards']))} differs from the tensor table")
    for f, v in kd["shards"].items():
        if not (re.fullmatch(r"[0-9a-f]{64}", v.get("sha256") or "") and v.get("head_sha256") == v.get("tree_sha256") == v["sha256"]
                and v.get("head_size") == v.get("tree_size") == v["size"]):
            probs.append(f"klee100 {f}: HEAD vs tree disagree or malformed")
        hdr = (v["size"] - by_shard.get(f, 0))
        v["header_plus_padding_bytes"] = hdr
        if not 8 <= hdr <= 64 * MiB:
            probs.append(f"klee100 {f}: size {v['size']} vs tensor bytes {by_shard.get(f, 0)}")
    mp = D["merlin_fp8_ple"]
    exp = kit.facts["ple_expected_shards"]
    for f, (size, s) in exp.items():
        v = mp["shards"].get(f) or {}
        if v.get("sha256") != s or v.get("size") != size:
            probs.append(f"merlin {f}: HF digest {v.get('sha256')} / {v.get('size')} != PLE_EXPECTED {s} / {size}")
    out["repos"] = {"klee100": kd, "merlin_fp8_ple": mp}
    out["unused_by_the_graft"] = [f for f in kd["shards"] if f == C.KLEE_BF16_PLE_SHARD]
    out["problems"] = probs
    out["ok"] = not probs and len(kd["shards"]) == 13 and len(mp["shards"]) == len(exp) == 10
    out["use"] = ("GPU notebook: stream sha256 of every resolved shard (klee100 10 model + 2 MTP, Merlin FP8 PLE 10) against "
                  "this manifest before launch; any mismatch or missing shard -> no boot (K6 / K7)")
    jw("klee100_shard_digests.json", out)
    return out


MiB = 2 ** 20
K7 = guard("digest_manifest", digest_manifest) or {"ok": False, "why": "digest manifest check crashed (see errors)"}
CHK["K7_shard_digest_manifest"] = {k: v for k, v in K7.items() if k != "repos"} | {"file": "klee100_shard_digests.json"}
log("K7 digests", K7.get("ok"), K7.get("problems"))
save()

# ===================================================================== 3. Manifest 2 static (fallback, labelled)
st_fix = kit.static_quant(cfg_fix, mods_k)
st_asis = kit.static_quant(kit.kcfg, mods_k)
ga_st_fix = kit.gate_a_rows(st_fix, intended, mods_k, "STATIC_CUDA_DISPATCH klee100_fixkeys")
ga_st_asis = kit.gate_a_rows(st_asis, intended, mods_k, "STATIC_CUDA_DISPATCH klee100 as published")
neg_cfg = kit.negative_config_key(cfg_fix, mods_k, intended)
neg_cfg_path = K / "negative_cfgkey_config.json"
neg_cfg_path.write_text(json.dumps(neg_cfg.pop("cfg")))
CHK["A_static"] = {"fixkeys": {k: ga_st_fix[k] for k in ("label", "n_bad", "bad", "unexpected_quantized_modules", "ok")},
                   "as_published": {k: ga_st_asis[k] for k in ("label", "n_bad", "unexpected_quantized_modules", "ok")},
                   "as_published_bad_sample": ga_st_asis["bad"][:12], "negative_config_key": neg_cfg}
log("A static", "fixkeys ok", ga_st_fix["ok"], ga_st_fix["n_bad"], "| as published bad", ga_st_asis["n_bad"], "| neg", neg_cfg["ok"])
save()


# ===================================================================== 4. mounts
def find_dirs(fname, max_depth=8):
    hits = []
    root = INPUT_ROOT.rstrip("/")
    for r, dirs, files in os.walk(root):
        if r[len(root):].count(os.sep) >= max_depth:
            dirs[:] = []
        if fname in files:
            hits.append(Path(r))
    return hits


MNT = {}
if not LOCAL:
    MNT["p1"] = [str(p) for p in find_dirs("config.json") if "merlin-p1" in str(p)]
    MNT["overlay"] = [str(p) for p in find_dirs("model-plefp8-00000.safetensors")]
    MNT["layers"] = [str(p) for p in find_dirs("klee100_layer00_experts.safetensors")]
CHK["mounts"] = MNT
P1 = Path(MNT["p1"][0]) if MNT.get("p1") else None
OVL = Path(MNT["overlay"][0]) if MNT.get("overlay") else None
LAY = Path(MNT["layers"][0]) if MNT.get("layers") else None
if LOCAL and os.environ.get("K100_LOCAL_LAYERS"):
    LAY = Path(os.environ["K100_LOCAL_LAYERS"])       # local quick check only: the same files as the private dataset
p1 = {"dir": str(P1)}
if P1:
    b = (P1 / "config.json").read_bytes()
    p1["config_sha256"] = C.sha(b)
    p1["equal_to_hf_cec53e9f"] = json.loads(b) == kit.mcfg
    p1["ple_embedding_dtype"] = json.loads(b).get("text_config", {}).get("ple_embedding_dtype")
CHK["p1_control"] = p1
log("mounts", json.dumps(MNT)[:800])
save()


# ===================================================================== 5. real-data probe of layers 0 (W3 g128) and 12 (W2 g64)
def st_header(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return n, json.loads(fh.read(n))


def unpack_bits(words, bits, count):
    """LSB-first bitstream of little-endian int32 words (AutoGPTQ packing, 3-bit fields span words) -> values."""
    import numpy as np
    by = np.ascontiguousarray(words).view(np.uint8)
    b = np.unpackbits(by.reshape(by.shape[0], -1), axis=1, bitorder="little")[:, : count * bits]
    return (b.reshape(b.shape[0], count, bits) * (1 << np.arange(bits, dtype=np.uint32))).sum(-1)


def probe_layers():
    import numpy as np
    out = {"dir": str(LAY)}
    man = json.loads((LAY / "manifest.json").read_text())
    kt = kit.tabs["klee100"]
    for L, info in sorted(man["layers"].items(), key=lambda kv: int(kv[0])):
        f = LAY / info["file"]
        rec = {"file": info["file"], "bytes": f.stat().st_size}
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        rec["sha256_ok"] = h.hexdigest() == info["sha256"] == info.get("sha256")
        n, hdr = st_header(f)
        names = {k for k in hdr if k != "__metadata__"}
        want = {k for k in kt if k.startswith(f"model.language_model.layers.{L}.mlp.experts.")}
        rec["tensors"] = len(names)
        rec["names_equal_ckpt"] = names == want
        rec["dtype_shape_equal_ckpt"] = all(hdr[k]["dtype"] == kt[k][0] and tuple(hdr[k]["shape"]) == tuple(kt[k][1]) for k in names & want)
        base = 8 + n
        mm = np.memmap(f, dtype=np.uint8, mode="r")

        def arr(name, dt):
            a, b2 = hdr[name]["data_offsets"]
            return mm[base + a: base + b2].view(dt).reshape(hdr[name]["shape"])
        it = intended.get(("target", f"language_model.model.layers.{L}.mlp.experts"), {})
        bits, g = it.get("bits"), it.get("group")
        zvals, sc_min, sc_max, sc_bad, lv = collections.Counter(), float("inf"), -float("inf"), 0, collections.Counter()
        sc_neg, sc_n, bf_rel, bf_changed = 0, 0, 0.0, 0
        deq_std = []
        for e in range(C.NE):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                pfx = f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}."
                qz = arr(pfx + "qzeros", np.int32)
                N = qz.shape[1] * 32 // bits
                z = unpack_bits(qz, bits, N)
                zvals.update(np.unique(z).tolist())
                s = arr(pfx + "scales", np.float16).astype(np.float32)
                sc_bad += int((~np.isfinite(s)).sum() + (s == 0).sum())
                sc_neg += int((s < 0).sum())
                sc_n += s.size
                sc_min, sc_max = min(sc_min, float(s.min())), max(sc_max, float(s.max()))
                sb = C.Kit.to_bf16(s)                          # Merlin creates scale params in the model dtype (BF16)
                nz = s != 0
                bf_rel = max(bf_rel, float((np.abs(sb - s)[nz] / np.abs(s)[nz]).max()) if nz.any() else 0.0)
                bf_changed += int((sb != s).sum())
                if e < 8:
                    qw = arr(pfx + "qweight", np.int32)
                    Kk = qw.shape[0] * 32 // bits
                    q = unpack_bits(qw.T, bits, Kk)            # [N, K]
                    lv.update(np.unique(q, return_counts=True)[0].tolist())
                    zc = list(zvals)[0] if len(zvals) == 1 else 0
                    zp = zc + 1 if zc == (1 << (bits - 1)) - 1 else zc
                    w = (q.astype(np.float32) - zp) * np.repeat(s.T, g, axis=1)[:, :Kk]
                    deq_std.append(float(w.std()))
        rec.update(bits=bits, group=g, qzeros_values=sorted(zvals), sym_constant=len(zvals) == 1,
                   qzeros_convention=("gptq_v1 (zero - 1 = 2^(b-1) - 1)" if sorted(zvals) == [(1 << (bits - 1)) - 1] else
                                      "gptq_v2 (zero = 2^(b-1))" if sorted(zvals) == [1 << (bits - 1)] else "ASYM_OR_UNKNOWN"),
                   scales_min=sc_min, scales_max=sc_max, scales_nonfinite_or_zero=sc_bad,
                   scales_negative_fraction=round(sc_neg / max(1, sc_n), 4),
                   scales_note="AutoRound full-range symmetric: signed scales (max-magnitude value / 2^(b-1)); the "
                               "dequant (q - 2^(b-1)) * s is linear in s, so the sign is carried by the scale",
                   scales_f16_to_bf16_rel_err_max=round(bf_rel, 6), scales_changed_by_bf16_cast_fraction=round(bf_changed / max(1, sc_n), 4),
                   qweight_levels_used=sorted(lv), all_levels_used=sorted(lv) == list(range(1 << bits)),
                   dequant_std_first8=[round(x, 6) for x in deq_std[:6]])
        rec["ok"] = bool(rec["sha256_ok"] and rec["names_equal_ckpt"] and rec["dtype_shape_equal_ckpt"] and rec["sym_constant"]
                         and rec["qzeros_convention"] != "ASYM_OR_UNKNOWN" and sc_bad == 0 and rec["all_levels_used"])
        out[L] = rec
        log("probe layer", L, json.dumps({k: rec[k] for k in ("ok", "bits", "group", "qzeros_values", "qzeros_convention", "all_levels_used")}))
    out["ok"] = all(v.get("ok") for k, v in out.items() if k.isdigit()) and sum(k.isdigit() for k in out) == 2
    return out


CHK["M3_layer_probe"] = (guard("probe_layers", probe_layers) if LAY else {"ok": False, "why": "klee100 layers dataset not mounted"}) \
    or {"ok": False, "why": "probe failed (see errors)"}
save()


# ===================================================================== 6. Gate B: PLE provenance + numerics from the MOUNTED overlay
def sha256_stream(p, chunk=1 << 24):
    h = hashlib.sha256()
    t = time.time()
    n = 0
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n, round(time.time() - t, 1)


def overlay_shas(exp):
    """K3: streaming sha256 of every mounted FP8 PLE shard vs the expected sha (4 threads; hashlib releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    t = time.time()
    todo = [f for f in exp if (OVL / f).is_file()]
    res = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for f, r in zip(todo, ex.map(lambda f: sha256_stream(OVL / f), todo)):
            res[f] = {"sha256": r[0], "bytes_hashed": r[1], "sec": r[2], "sha256_ok": r[0] == exp[f][1] and r[1] == exp[f][0]}
            log("K3 sha", f, "OK" if res[f]["sha256_ok"] else "MISMATCH", r[2], "s")
    return res, round(time.time() - t, 1)


def overlay_read():
    heads, files = {}, {}
    exp = kit.facts["ple_expected_shards"]
    for fname, (size, _sha) in exp.items():
        p = OVL / fname
        files[fname] = {"present": p.is_file(), "size_ok": p.is_file() and p.stat().st_size == size}
        if p.is_file():
            n, h = st_header(p)
            heads[fname] = h
            files[fname]["header_len"] = n
    shas, sec = overlay_shas(exp)
    for fname, r in shas.items():
        files[fname].update(r)
    SHA_K3.update(n_expected=len(exp), n_hashed=len(shas), wall_sec=sec,
                  all_ok=len(shas) == len(exp) and all(r["sha256_ok"] for r in shas.values()),
                  source="nb3_code_export.py PLE_EXPECTED (= HF LFS sha of halt95@cec53e9f model-plefp8-0000N; "
                         "cross-checked against the HF digest manifest in K7)")
    rows = bytearray()
    loc = {}
    for fname, h in heads.items():
        for name, v in h.items():
            if name != "__metadata__":
                loc[name] = (fname, v)
    for r in kit.ple["rows"]:
        name = C.PLE_PFX + f"ngram_embedding.shard_{r['shard']}.weight"
        fname, v = loc[name]
        off = 8 + files[fname]["header_len"] + v["data_offsets"][0] + r["row"] * v["shape"][1]
        with open(OVL / fname, "rb") as fh:
            fh.seek(off)
            rows += fh.read(v["shape"][1])
    sname = C.PLE_PFX + "ngram_embedding.weight_scale"
    fname, v = loc[sname]
    with open(OVL / fname, "rb") as fh:
        fh.seek(8 + files[fname]["header_len"] + v["data_offsets"][0])
        scale_hex = fh.read(2).hex()
    prov = {}
    for side in ("README.md", "config.json", "SHA256SUMS"):
        p = OVL / side
        if p.is_file():
            t = p.read_text(errors="replace")
            prov[side] = {"bytes": len(t), "sha256": C.sha(p.read_bytes()),
                          "revision_like": sorted(set(re.findall(r"\b[0-9a-f]{40}\b", t)))[:10],
                          "mentions": [ln.strip()[:200] for ln in t.splitlines() if re.search(r"(?i)radixark|fp8|revision|base_model|source", ln)][:12]}
    return heads, files, bytes(rows), scale_hex, prov


SHA_K3 = {}


def gate_b():
    G = {}
    if OVL:
        heads, files, rows, scale_hex, prov = overlay_read()
        G["overlay_files"] = files
        G["overlay_provenance_scan"] = prov
        G["K3_full_shard_sha256"] = dict(SHA_K3)
        src = "MOUNTED_OVERLAY"
    elif LOCAL and os.environ.get("K100_LOCAL_FP8_ROWS"):
        heads, files, rows, scale_hex = None, None, Path(os.environ["K100_LOCAL_FP8_ROWS"]).read_bytes(), kit.ple["merlin"]["weight_scale"]["raw_hex"]
        src = "LOCAL_HF_ROWS_NOT_MOUNT"
    else:
        return {"ok": False, "why": "FP8 PLE overlay not mounted"}
    G["rows_source"] = src
    G["rows_equal_hf_halt95_cec53e9f"] = C.sha(rows) == kit.ple["fp8_bin_sha256"]
    G["scale_equal_hf"] = scale_hex == kit.ple["merlin"]["weight_scale"]["raw_hex"]
    G["metadata"] = kit.ple_metadata(heads, files)
    G["numeric"] = kit.ple_numeric(rows, scale_hex, src)
    G["numeric_negative_permuted"] = kit.ple_numeric(rows, scale_hex, src + " PERMUTED", permute=True)
    G["graft_index"] = graft_chk
    G["config_dtype"] = cfg_fix["text_config"].get("ple_embedding_dtype")
    G["launcher_rule_launch_py_1219"] = "PASS" if G["config_dtype"] == "float8_e4m3fn" else "FAIL"
    meta_ok = G["metadata"]["ok"] and (src != "MOUNTED_OVERLAY" or (G["metadata"]["overlay"].get("all_equal_to_merlin_index")
                                                                     and all(v["size_ok"] for v in files.values())))
    G["ok"] = bool(src == "MOUNTED_OVERLAY" and meta_ok and SHA_K3.get("all_ok") is True
                   and G["numeric"]["ok"] and not G["numeric_negative_permuted"]["ok"]
                   and graft_chk["ok"] and G["rows_equal_hf_halt95_cec53e9f"] and G["scale_equal_hf"]
                   and G["launcher_rule_launch_py_1219"] == "PASS")
    return G


GB = guard("gate_b", gate_b) or {"ok": False, "why": "gate B crashed"}
log("B", json.dumps({"ok": GB.get("ok"), "src": GB.get("rows_source"), "cos_min": (GB.get("numeric") or {}).get("cos_min"),
                     "rel_max": (GB.get("numeric") or {}).get("normal_range_rel_err_max"),
                     "neg_permuted_ok": (GB.get("numeric_negative_permuted") or {}).get("ok"),
                     "meta_ok": (GB.get("metadata") or {}).get("ok"),
                     "K3_full_sha": {k: (GB.get("K3_full_shard_sha256") or {}).get(k) for k in ("all_ok", "n_hashed", "wall_sec")}}))
CHK["B"] = GB
save()

# ===================================================================== 7. Merlin venv (NB3 offline install) + REAL mode
MER, PY, ENV, TREE = {}, None, None, None
if not LOCAL:
    try:
        os.environ.update(NB3_UNIT="1", NB3_CPU_KIT="1", E23_OUT=str(K / "nb3_out"), E23_TMP=str(K / "nb3_tmp"), E23_INPUT=INPUT_ROOT,
                          RC2_SUBMISSION=str(K / "nb3_out/submission.json"), RC2_SHM=str(K / "nb3_shm"))
        os.environ.pop("E23_FAKE_SERVER", None)
        for d in ("nb3_out", "nb3_tmp", "nb3_shm"):
            (K / d).mkdir(exist_ok=True)
        G = runpy.run_path(str(K / "nb3_code_export.py"), run_name="nb3kit")["finish"].__globals__
        MER["inputs"] = G["R"]["inputs"]
        G["MERLIN_OK"] = True
        t = time.time()
        eng = G["merlin_install"]()
        MER["install"] = {"ok": True, "sec": round(time.time() - t), "kv": dict(eng["kv"]), "cuda_home": eng["cuda_home"]}
        pl = G["s_plan"]()
        PY, ENV, TREE = G["BENCH"]["py"], G["bench_env"](""), eng["kv"].get("MERLIN_TREE")
        log("Merlin install", json.dumps(MER["install"])[:400])
    except Exception as e:
        MER.setdefault("install", {"ok": False})
        MER["fatal"] = f"{type(e).__name__}: {e}"[:1000]
        MER["tb"] = traceback.format_exc()[-2500:]
        log("Merlin install FATAL", MER["fatal"])
if LOCAL and os.environ.get("K100_LOCAL_FAKE_VLLM"):   # local plumbing test only: fake vllm API, labelled, never on Kaggle
    PY, TREE = sys.executable, os.environ.get("K100_LOCAL_TREE")
    ENV = dict(os.environ, PYTHONPATH=os.environ["K100_LOCAL_FAKE_VLLM"])
    MER["install"] = {"ok": False, "local_fake_vllm": True}
CHK["merlin"] = MER
# libcuda.so.1 stub for the REAL subprocess (KIT-10: vllm._C_stable_libtorch needs it at import)
STUB = {}
try:
    ctypes.CDLL("libcuda.so.1")
    STUB["libcuda_so_1"] = "loadable"
except OSError as e:
    STUB["libcuda_so_1"] = f"not loadable: {e}"[:200]
cands = []
for root in [(ENV or {}).get("CUDA_HOME", ""), (MER.get("install") or {}).get("cuda_home", ""), "/usr/local/cuda",
             "/usr/local/cuda/targets/x86_64-linux"]:
    if root:
        cands += glob.glob(f"{root}/lib*/stubs/libcuda.so") + glob.glob(f"{root}/**/stubs/libcuda.so", recursive=True)
if cands:
    d = K / "cudastub"
    d.mkdir(exist_ok=True)
    if not (d / "libcuda.so.1").exists():
        os.symlink(cands[0], d / "libcuda.so.1")
    STUB.update(stub=cands[0], dir=str(d))
CHK["libcuda"] = STUB
save()

def cw_spec(side, cls, mod, order=None):
    """create_weights args [side, class, module, K, N parts] derived from the checkpoint sub-weights of `mod`
    (fused shard order from the loader map: `order` lists the shard ids, None = a single sub-weight)."""
    r = mods_k[(side, mod)]
    subs = {}
    for key in r["keys"]:
        base = key.rpartition(".")[0]
        subs[C.map_key(key).get("shard")] = base
    bases = [subs[s] for s in (order or [None])]
    nk = []
    for b in bases:
        f = fk[b]
        nk.append((f["N"], f["K"]) if f.get("fmt") == "GPTQ" else tuple(f["shape"]))
    assert len({k for _, k in nk}) == 1, (mod, nk)
    return [side, cls, mod, nk[0][1], [n for n, _ in nk]]


# one create_weights representative per quantized-linear KIND (side, class, format, bits, group): K1 construction rule
CW_REPS = {
    "qsa_qkv_indexer_W8": ["target", "Qwen4ExpQSAQKVIndexerLinear", "language_model.model.layers.3.self_attn.qkv_proj", 2560, [12288, 512, 512, 640]],
    "qsa_o_W8": ["target", "RowParallelLinear", "language_model.model.layers.3.self_attn.o_proj", 6144, [2560]],
    "gdn_in_proj_qkvz_W8": ["target", "MergedColumnParallelLinear", "language_model.model.layers.0.linear_attn.in_proj_qkvz", 2560, [2048, 2048, 6144, 6144]],
    "gdn_out_proj_W8": ["target", "RowParallelLinear", "language_model.model.layers.0.linear_attn.out_proj", 6144, [2560]],
    "mtp_fc_embedding_W8": ["draft", "ColumnParallelLinear", "mtp.fc_embedding", 2560, [2560]],
    "mtp_qkv_W8": ["draft", "Qwen4ExpQSAQKVIndexerLinear", f"mtp.layers.{C.NL}.self_attn.qkv_proj", 2560, [12288, 512, 512, 640]],
    "shared_expert_gate_up_BF16": ["target", "MergedColumnParallelLinear", "language_model.model.layers.0.mlp.shared_expert.gate_up_proj", 2560, [640, 640]]}
for _tag, _a in {"mtp_o_W8": ("draft", "RowParallelLinear", f"mtp.layers.{C.NL}.self_attn.o_proj", None),
                 "mtp_shared_expert_down_BF16": ("draft", "RowParallelLinear", f"mtp.layers.{C.NL}.mlp.shared_expert.down_proj", None),
                 "mtp_shared_expert_gate_up_BF16": ("draft", "MergedColumnParallelLinear", f"mtp.layers.{C.NL}.mlp.shared_expert.gate_up_proj", [0, 1]),
                 "shared_expert_down_BF16": ("target", "RowParallelLinear", "language_model.model.layers.0.mlp.shared_expert.down_proj", None),
                 "gdn_in_proj_ba_BF16": ("target", "MergedColumnParallelLinear", "language_model.model.layers.0.linear_attn.in_proj_ba", ["b", "a"]),
                 "ple_kv_proj_BF16": ("target", "MergedColumnParallelLinear(disable_tp)", "language_model.model.layers.1.ple.kv_proj", [0, 1]),
                 "visual_attn_proj_BF16": ("target", "VisionLinear", "visual.blocks.0.attn.proj", None)}.items():
    try:
        CW_REPS[_tag] = cw_spec(*_a)
    except Exception as e:
        CHK.setdefault("errors", {})[f"cw_spec {_tag}"] = {"error": f"{type(e).__name__}: {e}"[:400]}


def lin_kind(side, mod):
    r = mods_k.get((side, mod)) or {}
    it = intended.get((side, mod)) or {}
    return (side, r.get("class"), it.get("fmt"), it.get("bits"), it.get("group"))


REAL, R0 = {}, {}
if PY:
    cfg_dir = W / "fixkeys_overlay"
    asis_p, fix_p, mer_p = K / "cfg_asis.json", cfg_dir / "config.json", K / "cfg_merlin.json"
    asis_p.write_bytes(kit.kcfg_raw)
    mer_p.write_text(json.dumps(kit.mcfg))
    real_in = {"configs": {"klee100_fixkeys": str(fix_p), "klee100_as_published": str(asis_p), "negative_cfgkey": str(neg_cfg_path),
                           "merlin_control": str(mer_p)},
               "modules": [[s, m, r["class"], r["quant_config_passed"]] for (s, m), r in sorted(mods_k.items())
                           if r["quant_config_passed"] in (True, "ple")],
               "create_weights": CW_REPS,
               "src_sha": kit.facts["source_files_sha256_lf"],
               "tree_roots": [TREE, str(Path(TREE).parent) if TREE else None],
               "autoconfig_dirs": {"klee100_fixkeys": str(cfg_dir), **({"merlin_p1": str(P1)} if P1 else {})},
               "r0_dir": str(K / "r0_ckpt")}
    r0d = K / "r0_ckpt"                     # fixkeys config + Merlin p1's small tokenizer / processor files (no weights)
    r0d.mkdir(exist_ok=True)
    for n in ("config.json", "quantization_config.json", "model.safetensors.index.json"):
        shutil.copyfile(cfg_dir / n, r0d / n)
    if P1:
        for f in P1.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors") and not (r0d / f.name).exists() and f.stat().st_size < 64 * 2 ** 20:
                shutil.copyfile(f, r0d / f.name)
    (K / "real_in.json").write_text(json.dumps(real_in))
    renv = dict(ENV or os.environ)
    renv.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
    if STUB.get("dir"):
        renv["LD_LIBRARY_PATH"] = STUB["dir"] + ":" + renv.get("LD_LIBRARY_PATH", "")
    for mode, outp, tmo in (("real", K / "real.json", 2400), ("r0", K / "r0.json", 900)):
        t = time.time()
        try:
            r = subprocess.run([PY, str(K / "k100kit_real.py"), str(K / "real_in.json"), str(outp), mode], capture_output=True,
                               text=True, timeout=tmo, env=renv, cwd=str(K))
            (W / f"kit_{mode}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        res = json.loads(outp.read_text()) if outp.is_file() else {"error": f"no output (rc {rc})"}
        res["rc"], res["wall_sec"] = rc, round(time.time() - t)
        (REAL if mode == "real" else R0).update(res)
        log("REAL", mode, "rc", rc, json.dumps(res.get("steps", {}).get("platform")), "ran_all", res.get("ran_all"))
        save()
jw("k100kit_real_raw.json", REAL)
CHK["real_meta"] = {"steps": REAL.get("steps"), "rc": REAL.get("rc"), "ran_all": REAL.get("ran_all"),
                    "mp_kernels": REAL.get("mp_linear_kernel_choice_sm89_w8g128_tp2"), "autoconfig": REAL.get("autoconfig"),
                    "r0": dict({k: (R0.get("r0") or {}).get(k) for k in ("engine_config", "error", "target_ok")}, rc=R0.get("rc"),
                               n_quant_modules=len((R0.get("r0") or {}).get("target_modules_with_quant_method") or {}),
                               label="best-effort full meta build, not gated") if R0 else None}


# ===================================================================== 8. merge REAL -> Manifest 2, Gate A, coverage
REALIN_CW = CW_REPS
BENIGN_LIN = {"g_idx", "g_idx_sort_indices"}
BENIGN_MOE = {"g_idx", "g_idx_sort_indices", "bias"}
LAYER_OWN = {"bias"}     # linear bias: registered by the layer constructor (LinearBase subclasses, bias=True), not by
                         # quant_method.create_weights; the stub has has_bias=False, so it is not part of this check
BENIGN_NOTE = {"g_idx": "GPTQ g_idx: desc_act=False (AutoGPTQConfig(desc_act=False) in inc_wna16_linear.py / "
                        "inc_wna16_scheme.py), so it is never read",
               "bias": "MoE w13_bias / w2_bias are created but not in the checkpoint: auto_gptq.py:621-628 registers them as "
                       "None when no expert bias was loaded (routed_experts.py:156, 785 _loaded_expert_biases)"}


def moe_construction(x):
    """K1 construction status of one REAL RoutedExperts row (resolution is judged separately by Gate A)."""
    be, se = x.get("build_error"), x.get("stub_error")
    if x.get("error"):
        return "ERROR:MOE_CONSTRUCTION_NOT_PROVEN (error: %s)" % str(x["error"])[:160]
    if not be and not se and str(x.get("built", "")).startswith("RoutedExperts(meta") and x.get("method") not in (None, "None"):
        return "CONSTRUCTED_ON_META"
    if be and se and x.get("build_cuda_init") and x.get("stub_cuda_init") and x.get("build_humming_frame") and x.get("stub_humming_frame"):
        return C.GPU_ONLY_HUMMING + " (cuInit failed inside humming frames: no GPU in the CPU container)"
    if be and x.get("stub_built"):
        return "ERROR:MOE_CONSTRUCTION_NOT_PROVEN (real build failed, stub only: %s)" % str(be)[:160]
    return "ERROR:MOE_CONSTRUCTION_NOT_PROVEN (build: %s | stub: %s)" % (str(be)[:120], str(se)[:120])


def ple_construction(x):
    if x.get("error"):
        return "ERROR:PLE_METHOD_NOT_PROVEN (%s)" % str(x["error"])[:160]
    if (x.get("method") == "Qwen4ExpPLEFp8EmbeddingMethod" and x.get("guard_F8_E4M3") == "ACCEPTED"
            and str(x.get("guard_BF16", "")).startswith("REFUSED")):
        return "PLE_METHOD_FROM_QUANT_CONFIG_OK (FP8 method; guard accepts F8_E4M3, refuses BF16)"
    return "ERROR:PLE_METHOD_NOT_PROVEN (method %s, guard F8 %s, BF16 %s)" % (x.get("method"), x.get("guard_F8_E4M3"),
                                                                             str(x.get("guard_BF16"))[:60])


def real_rows(cname, lin_reps=None):
    """REAL rows -> labelled rows with a construction status (K1). lin_reps: {kind: (tag, ok)} from the create_weights
    representatives (klee100_fixkeys only); other configs keep RESOLVED_ON_STUB (they are resolution checks)."""
    rr = ((REAL.get("configs") or {}).get(cname) or {}).get("rows") or {}
    rows = {}
    for (side, mod), r in mods_k.items():
        if r["quant_config_passed"] is True or r["quant_config_passed"] == "ple":
            x = rr.get(f"{side}|{mod}")
            if x is None:
                continue
            lab = x.get("method", "MISSING") + (f"({x['inner']})" if x.get("inner") else "")
            if r["class"] == "RoutedExperts":
                cons = moe_construction(x)
                if cons.startswith(C.GPU_ONLY_HUMMING):
                    lab = f"{C.GPU_ONLY_HUMMING}({x.get('build_humming_object') or 'class not identified; humming frames'})"
                elif cons.startswith("ERROR"):
                    lab = cons
            elif r["quant_config_passed"] == "ple":
                cons = ple_construction(x)
                if x.get("error"):
                    lab = "ERROR:" + str(x["error"])[:200]
            else:
                if x.get("error"):
                    lab = "ERROR:LINEAR_RESOLUTION_FAILED " + str(x["error"])[:200]
                    cons = "ERROR:LINEAR_RESOLUTION_FAILED"
                elif lin_reps is None:
                    cons = "RESOLVED_ON_STUB (create_weights runs for klee100_fixkeys only)"
                else:
                    rep = lin_reps.get(lin_kind(side, mod))
                    cons = ("ERROR:LINEAR_CONSTRUCTION_NOT_PROVEN (no create_weights representative for kind %s)" % (lin_kind(side, mod),)
                            if rep is None else
                            f"CREATE_WEIGHTS_OK (kind representative {rep[0]})" if rep[1] else
                            f"ERROR:LINEAR_CONSTRUCTION_NOT_PROVEN (kind representative {rep[0]} failed)")
            be = str(x.get("wna16_moe_backend") or x.get("experts_cls") or "")
            if "AutoGPTQMoEMethod" in lab and be and "arlin" not in be.lower() and "MARLIN" not in be:
                lab += f"[backend {be} NOT MARLIN]"
            rows[(side, mod)] = {"method": lab, "resolved": x.get("resolved"), "how": "REAL", "construction": cons, "raw": x}
        else:
            rows[(side, mod)] = {"method": "NO_QUANT_CONFIG (source)", "resolved": [16, -1, True], "how": "source",
                                 "construction": "NOT_QUANTIZED (source)"}
    return rows


def moe_param_shapes(r, created, ep=2):
    """Created RoutedExperts params (EP2: NE/2 local experts) vs the checkpoint per-expert shapes: w13 = gate|up
    concatenated on N, w2 = down; GPTQ suffixes qweight [K*b/32, N], qzeros [K/g, N*b/32], scales [K/g, N]."""
    kt = kit.tabs["klee100"]
    per = collections.defaultdict(set)
    for key in r["keys"]:
        m = re.search(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.(\w+)$", key)
        if m:
            per[(m.group(1), m.group(2))].add(tuple(kt[key][1]))
    out, ok = {}, True
    for (proj, suf) in [(p, s) for p in ("gate_proj", "down_proj") for s in ("qweight", "qzeros", "scales")]:
        if (proj, suf) not in per:
            continue
        name = ("w13_" if proj == "gate_proj" else "w2_") + suf
        shp = per[(proj, suf)] | (per.get(("up_proj", suf), set()) if proj == "gate_proj" else set())
        if len(shp) != 1 or name not in created:
            out[name] = {"ok": False, "ckpt_shapes": sorted(shp), "created": created.get(name)}
            ok = False
            continue
        a, b = shp.pop()
        want = [C.NE // ep, a, 2 * b if proj == "gate_proj" else b]
        got = created[name][0]
        out[name] = {"want": want, "created": got, "ok": list(got) == want}
        ok = ok and list(got) == want
    return out, ok


def real_param_coverage():
    """K2: real parameter coverage over EVERY expected MoE module (48 target layers + the MTP bank) and the linear
    create_weights representatives. Per MoE module: REAL_PARAMS_OK / REAL_PARAMS_BAD / GPU_ONLY_NOT_CONSTRUCTED /
    ERROR_NOT_CONSTRUCTED / MISSING_REAL_ROW / MISSING_IN_CHECKPOINT. Parameter definitions only (meta device,
    no-op weight_loader): no tensor is loaded."""
    rr = ((REAL.get("configs") or {}).get("klee100_fixkeys") or {})
    out = {}
    for (side, mod) in C.EXPECTED_MOE:
        r = mods_k.get((side, mod))
        x = (rr.get("rows") or {}).get(f"{side}|{mod}")
        if r is None:
            out[mod] = {"status": "MISSING_IN_CHECKPOINT"}
            continue
        if x is None:
            out[mod] = {"status": "MISSING_REAL_ROW"}
            continue
        if "params" not in x:
            cons = moe_construction(x)
            out[mod] = {"status": "GPU_ONLY_NOT_CONSTRUCTED" if cons.startswith(C.GPU_ONLY_HUMMING) else "ERROR_NOT_CONSTRUCTED",
                        "construction": cons}
            continue
        created = set(x["params"])
        fed = set(r["params"])
        shapes, shp_ok = moe_param_shapes(r, x["params"])
        rec = {"ckpt_params": sorted(fed), "not_created_for_ckpt": sorted(fed - created), "created_not_fed": sorted(created - fed),
               "shapes": shapes, "construction": moe_construction(x)}
        ok = (not rec["not_created_for_ckpt"] and all(p.split("_", 1)[1] in BENIGN_MOE for p in rec["created_not_fed"])
              and shp_ok and rec["construction"] == "CONSTRUCTED_ON_META")
        rec["status"] = "REAL_PARAMS_OK" if ok else "REAL_PARAMS_BAD"
        out[mod] = rec
    lin = {}
    for tag, spec in REALIN_CW.items():
        cw = (rr.get("create_weights") or {}).get(tag)
        side, mod = spec[0], spec[2]
        r = mods_k.get((side, mod))
        if cw is None:
            lin[tag] = {"ok": False, "why": "no create_weights record", "kind": lin_kind(side, mod)}
            continue
        fed_all = set(r["params"]) if r else set()
        fed = fed_all - LAYER_OWN                    # bias belongs to the LinearBase constructor, not to create_weights
        created = set((cw.get("params") or {}))
        rec = {"created": cw.get("params"), "error": cw.get("error"), "ckpt_params": sorted(fed_all), "kind": lin_kind(side, mod),
               "layer_own_params_not_checked_here": sorted(fed_all & LAYER_OWN),
               "not_created_for_ckpt": sorted(fed - created), "created_not_fed": sorted(created - fed), "kernel": cw.get("kernel")}
        if cw.get("error") and "parallel group is not initialized" in cw["error"]:
            rec["cpu_limit"] = "CPU_HARNESS_TP_GROUP_NOT_INITIALISED (steps.distributed: %s)" % \
                               json.dumps((REAL.get("steps") or {}).get("distributed"), default=str)[:200]
        shp_ok = True
        if r and cw.get("params"):
            sub = collections.defaultdict(dict)
            for key in r["keys"]:
                base, _, suf = key.rpartition(".")
                sub[suf][base] = kit.tabs["klee100"][key][1]
            for suf, by in sub.items():
                p = (cw["params"] or {}).get(suf)
                if p is None:
                    continue
                shapes = list(by.values())
                if suf in ("qweight", "scales", "qzeros", "weight"):
                    ax = 0 if suf == "weight" else 1
                    tot = sum(s[ax] for s in shapes)
                    other = {s[1 - ax] for s in shapes}
                    want = [tot, other.pop()] if ax == 0 and len(other) == 1 else ([other.pop(), tot] if len(other) == 1 else None)
                    rec.setdefault("shape_check", {})[suf] = {"param": p[0], "ckpt_concat": want, "ok": want == p[0]}
                    shp_ok = shp_ok and want == p[0]
        rec["ok"] = bool(r) and bool(cw.get("params")) and not rec["not_created_for_ckpt"] and \
            all(p in BENIGN_LIN for p in rec["created_not_fed"]) and shp_ok
        lin[tag] = rec
    st = collections.Counter(v["status"] for v in out.values())
    kinds_all = {lin_kind(s, m) for (s, m), r in mods_k.items() if r["quant_config_passed"] is True and r["class"] != "RoutedExperts"}
    kinds_rep = {}
    for tag, rec in lin.items():
        k = rec.get("kind")
        if k not in kinds_rep or (rec["ok"] and not kinds_rep[k][1]):
            kinds_rep[k] = (tag, bool(rec["ok"]))
    lin_ok = bool(lin) and all(v["ok"] for v in lin.values()) and kinds_all <= set(kinds_rep)
    bad = [m for m, v in out.items() if v["status"] not in ("REAL_PARAMS_OK", "GPU_ONLY_NOT_CONSTRUCTED")] + \
          [t for t, v in lin.items() if not v["ok"]]
    return {"moe": out, "moe_status_counts": dict(st), "expected_moe_modules": len(C.EXPECTED_MOE),
            "real_checked_ok": st.get("REAL_PARAMS_OK", 0), "gpu_only_not_constructed": st.get("GPU_ONLY_NOT_CONSTRUCTED", 0),
            "linear_create_weights": lin, "linear_kinds": len(kinds_all),
            "linear_kinds_without_representative": sorted(map(str, kinds_all - set(kinds_rep))),
            "linear_reps_by_kind": kinds_rep, "linear_ok": lin_ok, "bad": bad,
            "ok": not bad and lin_ok and len(out) == len(C.EXPECTED_MOE) and st.get("REAL_PARAMS_OK", 0) >= 1,
            "benign_unfed": {"linear": sorted(BENIGN_LIN), "moe": sorted(BENIGN_MOE)}, "benign_note": BENIGN_NOTE,
            "label": "parameter definitions on the meta device with a no-op weight_loader; no tensor is loaded"}


GA = {"mode": "REAL_PATCHED_PLATFORM" if REAL.get("configs") else "STATIC_ONLY"}
if REAL.get("configs"):
    pc = real_param_coverage()
    rf = real_rows("klee100_fixkeys", pc["linear_reps_by_kind"])
    pc["linear_reps_by_kind"] = {str(k): v for k, v in pc["linear_reps_by_kind"].items()}     # JSON-safe after use
    ra = real_rows("klee100_as_published")
    rn = real_rows("negative_cfgkey")
    ga_fix = kit.gate_a_rows(rf, intended, mods_k, "REAL klee100_fixkeys")
    ga_asis = kit.gate_a_rows(ra, intended, mods_k, "REAL klee100 as published")
    ga_neg = kit.gate_a_rows(rn, intended, mods_k, "REAL negative (renamed layer-5 key)")
    n_q = sum(1 for (s, m), r in mods_k.items() if r["quant_config_passed"] in (True, "ple"))
    covered = len([1 for k in rf if mods_k[k]["quant_config_passed"] in (True, "ple")])
    mc = real_rows("merlin_control")
    moe_ctl = {m: r["method"] for (s, m), r in mc.items() if mods_k[(s, m)]["class"] == "RoutedExperts" and s == "target"}
    ctl_ok = len(moe_ctl) == C.NL and all(("WNA16" in v or "Marlin" in v or "MARLIN" in v) and "Unquantized" not in v and
                                          "ERROR" not in v for v in moe_ctl.values())
    src = (REAL.get("steps") or {}).get("source_identity") or {}
    moe_rows = [r for r in ga_fix["rows"] if r["class"] == "RoutedExperts"]
    res_pass_48 = ga_fix["moe_target_resolution_pass"] == C.NL
    res_fail = [r["module"] for r in ga_fix["rows"] if not r["resolution_ok"]]
    cons_err = ga_fix["construction_errors"]
    gpu_only = ga_fix["gpu_only_construction"]
    GA.update(fixkeys={k: ga_fix[k] for k in ("label", "n_bad", "bad", "unexpected_quantized_modules", "ok", "construction_counts",
                                              "gpu_only_construction", "construction_errors", "moe_target_resolution_pass")},
              as_published={k: ga_asis[k] for k in ("label", "n_bad", "unexpected_quantized_modules", "ok", "construction_counts")},
              as_published_bad_sample=ga_asis["bad"][:12],
              negative_config_key={"ok": (not ga_neg["ok"]) and any("layers.5.mlp.experts" in b for b in ga_neg["bad"]),
                                   "bad": ga_neg["bad"][:3]},
              coverage={"quant_modules": n_q, "real_resolved": covered},
              resolution={"target_moe_layers_pass": ga_fix["moe_target_resolution_pass"], "of": C.NL,
                          "modules_failing_resolution": res_fail[:20],
                          "mtp_bank": next((f"{r['construction']} | resolution_ok {r['resolution_ok']}" for r in moe_rows
                                            if r["side"] == "draft"), None)},
              linear_construction={"ok": pc["linear_ok"], "kinds": pc["linear_kinds"],
                                   "without_representative": pc["linear_kinds_without_representative"],
                                   "reps": pc["linear_reps_by_kind"]},
              merlin_control={"ok": ctl_ok, "moe_methods": sorted(set(moe_ctl.values()))},
              source_identity={k: src.get(k) for k in ("all_identical", "vllm_files_identical")},
              source_files=src.get("files"), param_coverage={k: v for k, v in pc.items() if k not in ("moe", "linear_create_weights")},
              static_agrees_with_real={"fixkeys_bad_static": ga_st_fix["n_bad"], "fixkeys_bad_real": ga_fix["n_bad"]})
    # ALL_OK rule: resolution PASS for all 48 target MoE layers (and every other quant module), no construction ERROR
    # (K1), every linear kind constructed by a create_weights representative. GPU-only humming objects do not fail the
    # gate but are reported as GPU-only (status), never as constructed / OK.
    GA["ok"] = bool(ga_fix["ok"] and res_pass_48 and not res_fail and not cons_err and pc["linear_ok"] and covered == n_q
                    and GA["negative_config_key"]["ok"] and ctl_ok and src.get("vllm_files_identical")
                    and (REAL.get("steps") or {}).get("platform", {}).get("ok"))
    if GA["ok"]:
        GA["status"] = (f"RESOLUTION_PASS_CONSTRUCTION_GPU_ONLY ({len(gpu_only)} humming MoE objects need a GPU to construct; "
                        "all other quant modules constructed on meta)" if gpu_only else "RESOLUTION_AND_META_CONSTRUCTION_PASS")
    else:
        # an ERROR construction label also fails that row's resolution (no method object -> no family), so the
        # construction verdict is named first; FAIL_RESOLUTION = a bits / group / family mismatch on a constructed row
        GA["status"] = ("FAIL_CONSTRUCTION" if (cons_err or not pc["linear_ok"]) else
                        "FAIL_RESOLUTION" if (res_fail or not res_pass_48) else "FAIL_OTHER (coverage / control / source / platform)")
    GA["as_published_fails_as_expected"] = not ga_asis["ok"]
    M2 = {"label": "REAL_PATCHED_PLATFORM (Merlin from_config / configure_quant_config / get_quant_method, CUDA (8, 9) patch)",
          "rows": ga_fix["rows"], "as_published_rows_bad": ga_asis["bad"], "static_fallback_rows": ga_st_fix["rows"]}
else:
    pc = {"ok": False, "why": "REAL mode did not run"}
    GA.update(ok=False, status="STATIC_ONLY", why="REAL mode unavailable: Gate A needs Merlin's own resolution (static is fallback only)",
              static=CHK["A_static"])
    M2 = {"label": "STATIC_CUDA_DISPATCH (fallback only; Gate A NOT OK without REAL)", "rows": ga_st_fix["rows"],
          "as_published_rows_bad": ga_st_asis["bad"]}
CHK["A"] = GA
log("A", GA["mode"], "ok", GA["ok"], GA.get("status"), json.dumps({k: GA.get(k) for k in ("coverage", "resolution", "merlin_control",
                                                                                         "source_identity")}, default=str)[:700])
save()

# real PLE method / guard joins Gate B
if REAL.get("configs"):
    pr = ((REAL["configs"].get("klee100_fixkeys") or {}).get("rows") or {})
    ple_row = next((v for k, v in pr.items() if k.endswith(".ple.ple_embedding.ngram_embedding")), {})
    pa = ((REAL["configs"].get("klee100_as_published") or {}).get("rows") or {})
    ple_asis = next((v for k, v in pa.items() if k.endswith(".ple.ple_embedding.ngram_embedding")), {})
    GB["real_ple_method"] = {"fixkeys": ple_row, "as_published": ple_asis}
    GB["real_ple_construction"] = ple_construction(ple_row) if ple_row else "ERROR:PLE_METHOD_NOT_PROVEN (no REAL PLE row)"
    GB["real_ple_ok"] = GB["real_ple_construction"].startswith("PLE_METHOD_FROM_QUANT_CONFIG_OK")
    GB["ok"] = bool(GB.get("ok") and GB["real_ple_ok"])
else:
    GB["real_ple_ok"] = False
    GB["ok"] = False
CHK["B"] = GB

# ===================================================================== 9. Gate C (MTP) + Gate D (memory)
a_for_c = ga_fix if REAL.get("configs") else ga_st_fix         # the same rows as Gate A (construction status included)
GC = guard("gate_c", kit.gate_c, mods_k, a_for_c, cfg_fix) or {"ok": False}
GC["mode"] = "REAL" if REAL.get("configs") else "STATIC_ONLY"
GC["ok"] = bool(GC.get("ok") and REAL.get("configs"))
ac = (REAL.get("autoconfig") or {})
if ac:
    GC["autoconfig_parity"] = ac
CHK["C"] = GC
prof = dict(kit.facts["profile_static"])
GD = guard("gate_d", kit.gate_d, tab_g, prof) or {"ok": False}
CHK["D"] = GD
log("C", GC.get("ok"), "| D", GD.get("ok"), json.dumps({t: {k: v.get(k) for k in ("klee100_est_gpu_weights_gib", "kv_blocks_4096_est",
                                                                           "kv_blocks_4096_est_with_c5_pin", "merlin_calibration_delta_gib")}
                                                       for t, v in (GD.get("variants") or {}).items()}))
save()

# ===================================================================== 10. manifests + verdict
for row in m1:
    q = (M2["rows"] and next((r for r in M2["rows"] if r["module"] == row["module"] and r["side"] == row["side"]), None))
    row["resolved_method"] = q["method"] if q else None
jw("manifest1_module_map.json", {"what": "Merlin v3.0.0 module map for klee100 + fixkeys + FP8 PLE graft (target "
                                 "Qwen4ExpForConditionalGeneration, draft Qwen4ExpMTP)", "n_modules": len(m1), "modules": m1,
                                 "language_model_only_variant": {k: m3_lm[k] for k in ("modules", "dropped_keys", "verdict")}})
jw("manifest2_quant_map.json", M2)
M1_ok = bool(not m3["unclassified_modules"] and not m3["unexpected_modules"] and not m3["missing_modules"]
             and GA.get("ok") and not (GA.get("fixkeys") or {}).get("unexpected_quantized_modules"))
# K2 status: STATIC (checkpoint side, every key, exact expert ids) vs SAMPLED / COMPLETE real parameter coverage
m3_static_ok = bool(m3["verdict"] == "OK" and neg_key["ok"] and neg_ids["ok"] and CHK["M3_layer_probe"].get("ok"))
if not m3_static_ok:
    M3_STATUS = "M3_FAIL_STATIC"
elif not REAL.get("configs"):
    M3_STATUS = "M3_STATIC_OK"
elif not pc.get("ok"):
    M3_STATUS = "M3_FAIL_REAL" if pc.get("bad") or not pc.get("linear_ok") else "M3_STATIC_OK"
elif pc["real_checked_ok"] == len(C.EXPECTED_MOE):
    M3_STATUS = "M3_COMPLETE_REAL_OK"
else:
    M3_STATUS = "M3_SAMPLED_REAL_OK"
M3_INFO = {"status": M3_STATUS, "static_ok": m3_static_ok,
           "real_moe_params_ok": pc.get("real_checked_ok"), "gpu_only_not_constructed": pc.get("gpu_only_not_constructed"),
           "expected_moe_modules": len(C.EXPECTED_MOE), "linear_ok": pc.get("linear_ok"),
           "meaning": {"M3_STATIC_OK": "checkpoint side only: every key mapped, every expected module, exact expert-id sets",
                       "M3_SAMPLED_REAL_OK": "static OK + Merlin's real parameters checked for SOME MoE modules (the rest are "
                                             "GPU-only humming objects, listed), every linear kind checked",
                       "M3_COMPLETE_REAL_OK": "static OK + Merlin's real parameters checked for all 49 MoE modules and every "
                                              "linear kind"}}
CHK["M3"] = M3_INFO
jw("manifest3_weight_coverage.json", {"status": M3_INFO, "static": m3, "real_param_coverage": pc, "layer_probe": CHK["M3_layer_probe"],
                                      "negative_renamed_tensor_key": neg_key, "negative_expert_ids": neg_ids,
                                      "checkpoint_formats": {k: v for k, v in sorted(fk.items()) if ".mlp.experts." not in k
                                                             or re.search(r"\.experts\.(0|511)\.", k)}})
M3_ok = M3_STATUS in ("M3_SAMPLED_REAL_OK", "M3_COMPLETE_REAL_OK")
ok = {"INPUTS_EMBEDDED_SHA": bool(inputs_ok), "FIXKEYS_OVERLAY": bool(CHK["fixkeys"]["ok"]),
      "M1_MODULE_MAP": M1_ok, "M3_WEIGHT_COVERAGE": M3_ok, "GATE_A_QUANT_REAL": bool(GA.get("ok")),
      "GATE_B_PLE": bool(GB.get("ok")), "GATE_C_MTP": bool(GC.get("ok")), "GATE_D_MEMORY_ESTIMATE": bool(GD.get("ok")),
      "K7_SHARD_DIGEST_MANIFEST": bool(K7.get("ok"))}
ALL = all(ok.values())
REP["verdict"] = {
    "checks": ok, "ALL_OK": ALL,
    "label": "CPU_COMPATIBILITY_PASS" if ALL else "CPU_COMPATIBILITY_FAIL",
    "meaning": ("CPU_COMPATIBILITY_PASS = eligible for a GPU load attempt. It is NOT 'all tensors loaded': Merlin's resolver "
                "and create_weights ran on the meta device with a no-op weight_loader (parameter definitions only); no "
                "tensor byte was loaded into a module, no kernel ran, and the humming MoE objects listed as GPU-only were "
                "not constructed. Physical loading + accuracy = GPU_LOAD_AND_ACCURACY_PASS, decided only by the GPU notebook "
                "(boot, every indexed shard resolved and sha-verified against klee100_shard_digests.json, short oracle)."),
    "gpu_load_and_accuracy": "NOT_TESTED (needs the GPU notebook)",
    "minutes": round((time.time() - T0) / 60, 1),
    "summary": {"m3": m3["verdict"], "m3_status": M3_STATUS, "neg_tensor_key": neg_key["verdict"],
                "neg_expert_ids": {k: v["verdict"] for k, v in neg_ids["cases"].items()}, "A_mode": GA["mode"],
                "A_status": GA.get("status"),
                "A_target_moe_resolution_pass": (GA.get("resolution") or {}).get("target_moe_layers_pass"),
                "A_gpu_only_construction": len((GA.get("fixkeys") or {}).get("gpu_only_construction") or []),
                "A_construction_errors": len((GA.get("fixkeys") or {}).get("construction_errors") or []),
                "A_fixkeys_bad": (GA.get("fixkeys") or {}).get("n_bad"),
                "A_as_published_bad": (GA.get("as_published") or CHK["A_static"]["as_published"]).get("n_bad"),
                "B_cos_min": (GB.get("numeric") or {}).get("cos_min"),
                "B_K3_full_sha": (GB.get("K3_full_shard_sha256") or {}).get("all_ok"),
                "D_blocks": ((GD.get("variants") or {}).get("with_vision") or {}).get("kv_blocks_4096_est"),
                "K7_digests": K7.get("ok")}}
save()
bad = [k for k, v in ok.items() if not v]
print("VERDICT", json.dumps(REP["verdict"], default=str)[:6000], flush=True)
print("LABEL", REP["verdict"]["label"], "(eligible for a GPU load attempt; not a load or accuracy test)" if ALL else "", flush=True)
print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
