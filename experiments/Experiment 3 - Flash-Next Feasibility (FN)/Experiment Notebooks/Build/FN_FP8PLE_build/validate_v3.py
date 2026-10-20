"""Local validation of FN-SMOKE v3 (real notebook code): v2 gate tests with the new RAM limit + stage-1 merge tests on
small fake checkpoints (the real EXPECT values are checked separately against the HF manifest)."""
import contextlib, hashlib, io, json, os, shutil, struct, subprocess, sys
from pathlib import Path

NB = r"D:\ARC AGI 2\Research Base Experiment\FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb"
HERE = Path(__file__).parent
COMP = str(HERE.parent / "fake_input" / "competitions" / "arc-prize-2026-arc-agi-2")
GOOD_SMI = "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0 |"
L4x4 = "\n".join(["NVIDIA L4, 23034 MiB, 8.9"] * 4)
MEM = lambda total, avail: f"MemTotal: {total*1000000} kB\nMemAvailable: {avail*1000000} kB\n"
CODES = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
RUNNER = r'''
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8")); G = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code": exec(compile("".join(c["source"]), f"cell{i}", "exec"), G)
r = G["REPORT"]; s0 = r["stages"].get("0_environment", {}); s1 = r["stages"].get("1_inputs", {})
print("RESULT", json.dumps({"stopped_at": r["stopped_at"], "stages": {k: v["ok"] for k, v in r["stages"].items()},
                            "checks": s0.get("checks"), "s1": s1}, default=str))
'''


def run(name, **env):
    work = HERE / "val3" / name; shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    e = dict(os.environ, FN_SMOKE_WORK=str(work), FN_SMOKE_COMP=COMP, PYTHONIOENCODING="utf-8",
             FN_SMOKE_MODEL_DST=str(work / "fn_model"), FN_SMOKE_LINK="copy", **env)
    out = subprocess.run([sys.executable, "-c", RUNNER, NB], env=e, capture_output=True, text=True, timeout=300).stdout
    res = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][-1][7:])
    print(f"{name:30} stopped_at={res['stopped_at']!s:18} stages={res['stages']}")
    return res


# ---------------------------------------------------------------- fake checkpoints
def st(path, tensors, pad=64):
    hdr = json.dumps({k: {"dtype": "U8", "shape": [pad], "data_offsets": [i * pad, (i + 1) * pad]} for i, k in enumerate(tensors)}).encode()
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0" * (pad * len(tensors)))


def fake(root, ple_fp8=True):
    intel, ov = root / "intel", root / "overlay"
    shutil.rmtree(root, ignore_errors=True); intel.mkdir(parents=True); ov.mkdir()
    st(intel / "model-00001-of-00017.safetensors", ["a.w"]); st(intel / "model-00003-of-00017.safetensors", ["b.w"])
    st(intel / "model-ple-00001-of-00022.safetensors", ["ple.shard_0.weight"]); st(intel / "model_extra_tensors.safetensors", ["mtp.w"])
    for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "ple_reshard_manifest.json"):
        (intel / n).write_text("{}")
    base = {"model_type": "qwen4_exp", "quantization_config": {"quant_method": "auto-round", "bits": 4, "group_size": 128},
            "text_config": {"ple_embed_dim": 2560}}
    (intel / "config.json").write_text(json.dumps(base))
    (intel / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a.w": "model-00001-of-00017.safetensors",
                                                                                   "ple.shard_0.weight": "model-ple-00001-of-00022.safetensors"}}))
    cfg = json.loads(json.dumps(base))
    if ple_fp8: cfg["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
    (ov / "config.json").write_text(json.dumps(cfg))
    st(ov / "model-plefp8-00000.safetensors", ["ple.shard_0.weight", "ple.weight_scale"])
    (ov / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a.w": "model-00001-of-00017.safetensors", "b.w": "model-00003-of-00017.safetensors",
        "ple.shard_0.weight": "model-plefp8-00000.safetensors", "ple.weight_scale": "model-plefp8-00000.safetensors"}}))
    (ov / "LICENSE").write_text("Qwen Community License 1.0")
    return intel, ov


def expect_of(intel, ov):
    shards = {}
    for d in (intel, ov):
        for p in d.glob("*.safetensors"):
            if p.name.startswith(("model-ple-", "model_extra")): continue
            b = p.read_bytes(); n = struct.unpack("<Q", b[:8])[0]
            shards[p.name] = [len(b), hashlib.sha256(b[:8 + n]).hexdigest(), n]
    files = {n: [len((ov / n).read_bytes()), hashlib.sha256((ov / n).read_bytes()).hexdigest()]
             for n in ("config.json", "model.safetensors.index.json")}
    return {"shards": shards, "files": files}


G = {"__name__": "__main__"}
os.environ.update(FN_SMOKE_WORK=str(HERE / "val3" / "unit"), FN_SMOKE_COMP=COMP, FN_SMOKE_LINK="copy")
(HERE / "val3" / "unit").mkdir(parents=True, exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    for k in (0, 2):
        exec(compile(CODES[k], f"cell{k}", "exec"), G)   # cell 2 defines build_model_dir; go() is False (stage 0 not run)
build = G["build_model_dir"]
R = HERE / "val3" / "merge"


def case(name, mutate=None, ple_fp8=True, expect_mut=None):
    intel, ov = fake(R / name, ple_fp8)
    exp = expect_of(intel, ov)
    if mutate: mutate(intel, ov)
    if expect_mut: expect_mut(exp)
    ok, info = build(str(intel), str(ov), str(R / name / "dest"), exp)
    print(f"merge:{name:24} ok={ok!s:5} problems={info.get('problems')} intel={info.get('from_intel')} overlay={info.get('from_overlay')}")
    return ok, info


ok, info = case("good")
assert ok and info["from_intel"] == 2 and info["from_overlay"] == 1 and info["ple_embedding_dtype"] == "float8_e4m3fn"
dest = R / "good" / "dest"
assert sorted(p.name for p in dest.glob("*.safetensors")) == ["model-00001-of-00017.safetensors", "model-00003-of-00017.safetensors",
                                                              "model-plefp8-00000.safetensors"], "BF16 PLE / MTP must not be linked"
assert json.loads((dest / "config.json").read_text())["text_config"]["ple_embedding_dtype"] == "float8_e4m3fn"
assert (dest / "tokenizer.json").exists() and not (dest / "ple_reshard_manifest.json").exists()
ok, info = case("overlay_shard_missing", lambda i, o: (o / "model-plefp8-00000.safetensors").unlink()); assert not ok
ok, info = case("intel_header_changed", lambda i, o: st(i / "model-00003-of-00017.safetensors", ["b.x"])); assert not ok
ok, info = case("intel_size_changed", lambda i, o: (i / "model-00001-of-00017.safetensors").open("ab").write(b"x")); assert not ok
ok, info = case("intel_shard_missing", lambda i, o: (i / "model-00003-of-00017.safetensors").unlink()); assert not ok
ok, info = case("config_tampered", lambda i, o: (o / "config.json").write_text("{}")); assert not ok and "differs" in info["problems"][0]
ok, info = case("config_bf16_ple", ple_fp8=False); assert not ok and any("ple_embedding_dtype" in p for p in info["problems"])
ok, info = case("unexpected_weight", expect_mut=lambda e: e["shards"].pop("model-00003-of-00017.safetensors")); assert not ok

# real EXPECT embedded in the notebook == HF manifest (sizes / header hashes / small-file hashes)
man = json.loads((HERE / "manifest.json").read_text())
E = G["EXPECT"]
assert len(E["shards"]) == 25 and all(E["shards"][n] == [v["size"], v["header_sha256"], v["header_len"]]
                                      for n, v in man["files"].items() if n.endswith(".safetensors"))
assert E["files"]["config.json"][1] == man["files"]["config.json"]["sha256"]
print("embedded EXPECT matches the HF manifest (25 shards + config + index)")
up = json.load(open(r"D:\ARC AGI 2\Research Base Experiment\FN-PLEFP8-UPLOAD.ipynb", encoding="utf-8"))
upsrc = "".join(up["cells"][1]["source"])
assert all(n in upsrc for n in E["files"]) and all(n in upsrc for n in E["shards"] if "plefp8" in n)
print("every overlay file the smoke notebook needs is in the uploader's EXPECTED_FILES")

# ---------------------------------------------------------------- whole-notebook runs (stage gates, wiring)
good = dict(FN_SMOKE_FAKE_SMI=GOOD_SMI, FN_SMOKE_FAKE_GPUS=L4x4, FN_SMOKE_FAKE_MEMINFO=MEM(190, 170), FN_SMOKE_FAKE_TMPFREE="100")
r = run("deadline_budget_30s", FN_SMOKE_BUDGET="30", **good); assert r["stopped_at"] == "global_deadline"
r = run("ram_available_75GB_passes", **{**good, "FN_SMOKE_FAKE_MEMINFO": MEM(100, 75)})
assert r["stages"]["0_environment"] and r["stopped_at"] == "1_inputs"
r = run("ram_available_60GB_fails", **{**good, "FN_SMOKE_FAKE_MEMINFO": MEM(100, 60)}); assert r["stopped_at"] == "0_environment"
r = run("two_L4", **{**good, "FN_SMOKE_FAKE_GPUS": "\n".join(["NVIDIA L4, 23034 MiB, 8.9"] * 2)}); assert r["stopped_at"] == "0_environment"
r = run("driver_570", **{**good, "FN_SMOKE_FAKE_SMI": "Driver Version: 570.133.20   CUDA Version: 12.8"}); assert r["stopped_at"] == "0_environment"
intel, ov = fake(HERE / "val3" / "nb_fake")
r = run("fake_model_vs_real_expect", FN_SMOKE_INTEL=str(intel), FN_SMOKE_OVERLAY=str(ov), **good)
assert r["stopped_at"] == "1_inputs" and r["s1"]["missing_inputs"] == ["wheelhouse"], r["s1"]   # wheelhouse absent locally
print("ALL V3 VALIDATION TESTS PASS")
