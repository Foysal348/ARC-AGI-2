"""Validation of FN-SMOKE v3 using the real notebook code, runnable from an extracted bundle:
    python build/validate_v3.py
Merge tests on small fake checkpoints, the embedded EXPECT vs manifest, hardware gates, stage-3 and launch-RAM checks
with a fake subprocess, memlock handling with a fake `resource` module, and the incomplete-run verdict."""
import contextlib, hashlib, io, json, os, shutil, struct, subprocess, sys, tempfile, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb"
UP = ROOT / "notebooks" / "FN-PLEFP8-UPLOAD.ipynb"
COMP = str(ROOT / "tests" / "fake_comp")
TMP = Path(tempfile.mkdtemp(prefix="fnv3_"))
CODES = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
GOOD_SMI = "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0 |"
L4x4 = "\n".join(["NVIDIA L4, 23034 MiB, 8.9"] * 4)
MEM = lambda total, avail: f"MemTotal: {total*1000000} kB\nMemAvailable: {avail*1000000} kB\n"
os.environ.update(FN_SMOKE_COMP=COMP, FN_SMOKE_LINK="copy", PYTHONIOENCODING="utf-8")


def fresh(name, **env):
    """exec cell 1 (config) in a new namespace; returns G"""
    work = TMP / name; work.mkdir(parents=True, exist_ok=True)
    os.environ["FN_SMOKE_WORK"] = str(work); os.environ["FN_SMOKE_MODEL_DST"] = str(work / "fn_model")
    for k, v in env.items(): os.environ[k] = v
    G = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(CODES[0], "cell1", "exec"), G)
    for k in env: os.environ.pop(k, None)
    return G


def ex(G, k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(CODES[k], f"cell{k+1}", "exec"), G)
    return buf.getvalue()


# ---------------------------------------------------------------- fake checkpoints
def st(path, tensors, pad=64, fill=b"\0"):
    hdr = json.dumps({k: {"dtype": "U8", "shape": [pad], "data_offsets": [i * pad, (i + 1) * pad]} for i, k in enumerate(tensors)}).encode()
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr + fill * (pad * len(tensors)))


def fake(root, ple_fp8=True):
    intel, ov = root / "intel", root / "overlay"
    shutil.rmtree(root, ignore_errors=True); intel.mkdir(parents=True); ov.mkdir()
    st(intel / "model-00001-of-00017.safetensors", ["a.w"]); st(intel / "model-00003-of-00017.safetensors", ["b.w"])
    st(intel / "model-ple-00001-of-00022.safetensors", ["ple.shard_0.weight"]); st(intel / "model_extra_tensors.safetensors", ["mtp.w"])
    for n in ("tokenizer.json", "quantization_config.json", "config.json.autoround.bak", "processor_config.json", "ple_reshard_manifest.json"):
        (intel / n).write_text('{"group_size": 64}')
    base = {"model_type": "qwen4_exp", "quantization_config": {"quant_method": "auto-round", "bits": 4, "group_size": 128},
            "text_config": {"ple_embed_dim": 2560}}
    (intel / "config.json").write_text(json.dumps(base))
    cfg = json.loads(json.dumps(base))
    if ple_fp8: cfg["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
    (ov / "config.json").write_text(json.dumps(cfg))
    st(ov / "model-plefp8-00000.safetensors", ["ple.shard_0.weight", "ple.weight_scale"])
    (ov / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a.w": "model-00001-of-00017.safetensors", "b.w": "model-00003-of-00017.safetensors",
        "ple.shard_0.weight": "model-plefp8-00000.safetensors", "ple.weight_scale": "model-plefp8-00000.safetensors"}}))
    (ov / "tokenizer.json").write_text('{"albucino": true}'); (ov / "LICENSE").write_text("Qwen Community License 1.0")
    return intel, ov


def expect_of(intel, ov):
    shards = {}
    for d in (intel, ov):
        for p in d.glob("*.safetensors"):
            if p.name.startswith(("model-ple-", "model_extra")): continue
            b = p.read_bytes(); n = struct.unpack("<Q", b[:8])[0]
            shards[p.name] = [len(b), hashlib.sha256(b).hexdigest(), hashlib.sha256(b[:8 + n]).hexdigest(), n]
    files = {n: [len((ov / n).read_bytes()), hashlib.sha256((ov / n).read_bytes()).hexdigest()]
             for n in ("config.json", "model.safetensors.index.json", "tokenizer.json", "LICENSE")}
    return {"shards": shards, "files": files}


G = fresh("unit")
ex(G, 2)    # defines build_model_dir (stage 1 itself fails: no inputs locally)
build = G["build_model_dir"]


def case(name, mutate=None, ple_fp8=True, expect_mut=None, full_hash=True):
    intel, ov = fake(TMP / "merge" / name, ple_fp8)
    exp = expect_of(intel, ov)
    if mutate: mutate(intel, ov)
    if expect_mut: expect_mut(exp)
    ok, info = build(str(intel), str(ov), str(TMP / "merge" / name / "dest"), exp, full_hash=full_hash)
    print(f"merge:{name:30} ok={ok!s:5} problems={info.get('problems')}")
    return ok, info, TMP / "merge" / name / "dest"


ok, info, dest = case("good")
assert ok and info["from_intel"] == 2 and info["from_overlay"] == 1 and info["full_hash"] and info["ple_embedding_dtype"] == "float8_e4m3fn"
assert sorted(p.name for p in dest.glob("*.safetensors")) == ["model-00001-of-00017.safetensors", "model-00003-of-00017.safetensors",
                                                              "model-plefp8-00000.safetensors"], "BF16 PLE / MTP must not be linked"
assert sorted(p.name for p in dest.iterdir() if not p.name.endswith(".safetensors")) == ["LICENSE", "config.json", "model.safetensors.index.json", "tokenizer.json"]
assert json.loads((dest / "tokenizer.json").read_text()) == {"albucino": True}, "overlay must be authoritative for small files"
print("   -> Intel's quantization_config.json / config.json.autoround.bak / processor_config.json are NOT linked")


def flip_after_header(i, o):
    p = i / "model-00003-of-00017.safetensors"; b = bytearray(p.read_bytes()); b[-1] ^= 0xFF; p.write_bytes(bytes(b))


ok, _, _ = case("corrupt_after_header_fullhash", flip_after_header); assert not ok
ok, _, _ = case("corrupt_after_header_nohash", flip_after_header, full_hash=False); assert ok   # shows why full hash matters
ok, _, _ = case("overlay_shard_missing", lambda i, o: (o / "model-plefp8-00000.safetensors").unlink()); assert not ok
ok, _, _ = case("overlay_tokenizer_missing", lambda i, o: (o / "tokenizer.json").unlink()); assert not ok
ok, _, _ = case("intel_header_changed", lambda i, o: st(i / "model-00003-of-00017.safetensors", ["b.x"])); assert not ok
ok, _, _ = case("intel_size_changed", lambda i, o: (i / "model-00001-of-00017.safetensors").open("ab").write(b"x")); assert not ok
ok, _, _ = case("intel_shard_missing", lambda i, o: (i / "model-00003-of-00017.safetensors").unlink()); assert not ok
ok, info, _ = case("config_tampered", lambda i, o: (o / "config.json").write_text("{}")); assert not ok and "differs" in info["problems"][0]
ok, info, _ = case("config_bf16_ple", ple_fp8=False); assert not ok and any("ple_embedding_dtype" in p for p in info["problems"])
ok, _, _ = case("unexpected_weight", expect_mut=lambda e: e["shards"].pop("model-00003-of-00017.safetensors")); assert not ok

# ---------------------------------------------------------------- embedded EXPECT vs manifest vs uploader
man = json.loads((ROOT / "build" / "manifest.json").read_text())
E = G["EXPECT"]
assert len(E["shards"]) == 25 and all(E["shards"][n] == [v["size"], v["sha256"], v["header_sha256"], v["header_len"]]
                                      for n, v in man["files"].items() if n.endswith(".safetensors"))
overlay = {n for n, v in man["files"].items() if v["role"] == "overlay"}
assert set(E["files"]) | {n for n in E["shards"] if "plefp8" in n} == overlay
U = {}
exec("".join(json.load(open(UP, encoding="utf-8"))["cells"][1]["source"]).replace("import requests", "requests = None"), U)
assert set(U["EXPECTED_FILES"]) == overlay and all(U["EXPECTED_FILES"][n][2] == man["files"][n]["sha256"] for n in overlay)
print(f"embedded EXPECT == manifest (25 shards, {len(E['files'])} small files); uploader uploads exactly the {len(overlay)} overlay files")

# ---------------------------------------------------------------- whole-notebook runs
RUNNER = r'''
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8")); G = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code": exec(compile("".join(c["source"]), f"cell{i}", "exec"), G)
r = G["REPORT"]
print("RESULT", json.dumps({"stopped_at": r["stopped_at"], "stages": {k: v["ok"] for k, v in r["stages"].items()},
                            "s0": r["stages"].get("0_environment", {}), "s1": r["stages"].get("1_inputs", {})}, default=str))
'''


def run(name, **env):
    work = TMP / "nb" / name; work.mkdir(parents=True)
    e = dict(os.environ, FN_SMOKE_WORK=str(work), FN_SMOKE_MODEL_DST=str(work / "fn_model"), **env)
    out = subprocess.run([sys.executable, "-c", RUNNER, str(NB)], env=e, capture_output=True, text=True, timeout=300).stdout
    res = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][-1][7:])
    print(f"nb:{name:30} stopped_at={res['stopped_at']!s:18} stages={res['stages']}")
    return res


good = dict(FN_SMOKE_FAKE_SMI=GOOD_SMI, FN_SMOKE_FAKE_GPUS=L4x4, FN_SMOKE_FAKE_MEMINFO=MEM(190, 170), FN_SMOKE_FAKE_TMPFREE="100")
assert run("deadline_budget_30s", FN_SMOKE_BUDGET="30", **good)["stopped_at"] == "global_deadline"
r = run("ram_80GB_passes", **{**good, "FN_SMOKE_FAKE_MEMINFO": MEM(100, 80)}); assert r["stages"]["0_environment"] and r["stopped_at"] == "1_inputs"
assert "memlock" in r["s0"]
assert run("ram_70GB_fails", **{**good, "FN_SMOKE_FAKE_MEMINFO": MEM(100, 70)})["stopped_at"] == "0_environment"
assert run("ram_70GB_override_passes", FN_MIN_RAM_GB="70", **{**good, "FN_SMOKE_FAKE_MEMINFO": MEM(100, 70)})["stages"]["0_environment"]
assert run("two_L4", **{**good, "FN_SMOKE_FAKE_GPUS": "\n".join(["NVIDIA L4, 23034 MiB, 8.9"] * 2)})["stopped_at"] == "0_environment"
assert run("driver_570", **{**good, "FN_SMOKE_FAKE_SMI": "Driver Version: 570.133.20   CUDA Version: 12.8"})["stopped_at"] == "0_environment"
intel, ov = fake(TMP / "nb_fake")
r = run("fake_model_vs_real_expect", FN_SMOKE_INTEL=str(intel), FN_SMOKE_OVERLAY=str(ov), **good)
assert r["stopped_at"] == "1_inputs" and r["s1"]["missing_inputs"] == ["wheelhouse"], r["s1"]

# ---------------------------------------------------------------- stage 3 with a fake subprocess
def stage3(name, help_rc=0, probe_rc=0, raise_exc=False, drop_flag=None):
    G = fresh("s3_" + name)
    G.update(PREFIX=str(TMP / "pfx"), VENV=str(TMP / "pfx" / "venv"), PYTHON="python", SGLANG="sglang")
    need = ["--tensor-parallel-size", "--ep-size", "--quantization", "--ple-offload-embedding",
            "--mamba-backend", "--linear-attn-decode-backend", "--linear-attn-prefill-backend", "--disable-cuda-graph",
            "--max-mamba-cache-size", "--moe-runner-backend", "--mamba-radix-cache-strategy",
            "--mamba-track-interval", "--disable-radix-cache", "--attention-backend"]
    def run_(cmd, **kw):
        if raise_exc: raise OSError("boom")
        if cmd[0] == "sglang":
            return types.SimpleNamespace(returncode=help_rc, stdout=" ".join(f for f in need if f != drop_flag), stderr="")
        return types.SimpleNamespace(returncode=probe_rc, stdout="matmul ok", stderr="")
    G["subprocess"] = types.SimpleNamespace(run=run_, Popen=None)
    ex(G, 4)
    st3 = G["REPORT"]["stages"]
    print(f"stage3:{name:22}", {k: v["ok"] for k, v in st3.items()})
    return st3


assert stage3("good")["3_preflight"]["ok"]
assert not stage3("help_exit_1", help_rc=1)["3_preflight"]["ok"]
assert not stage3("probe_fails", probe_rc=1)["3_preflight"]["ok"]
assert not stage3("flag_missing", drop_flag="--ep-size")["3_preflight"]["ok"]
s3 = stage3("exception", raise_exc=True); assert "3_preflight" in s3 and not s3["3_preflight"]["ok"] and "3_torch_cuda" not in s3

# ---------------------------------------------------------------- stage 4: RAM re-check right before launch
G = fresh("launch_ram")
G.update(SGLANG="sglang", SERVE_ENV={}, PYTHON="python")
os.environ["FN_SMOKE_FAKE_MEMINFO"] = MEM(100, 60)
G["subprocess"] = types.SimpleNamespace(Popen=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")), run=None)
ex(G, 1)    # stage 0 defines mem_available_gb (and fails on the fake 60 GB itself); reset so launch sees go() == True
G["REPORT"]["stages"].clear(); G["REPORT"]["stopped_at"] = None
ex(G, 5)
os.environ.pop("FN_SMOKE_FAKE_MEMINFO")
s4 = G["REPORT"]["stages"]["4_launch"]; assert not s4["ok"] and s4["ram_available_gb"] < 75, s4
print("launch: MemAvailable 60 GB right before launch -> 4_launch FAIL, server not started")

# ---------------------------------------------------------------- memlock with a fake resource module
def fake_resource(soft, hard, can_raise=True):
    m = types.ModuleType("resource"); m.RLIMIT_MEMLOCK = 8; m.RLIM_INFINITY = -1; state = {"v": (soft, hard)}
    m.getrlimit = lambda k: state["v"]
    def setr(k, v):
        if not can_raise: raise ValueError("not allowed")
        state["v"] = v
    m.setrlimit = setr
    return m


for soft, hard, can, want in [(65536, -1, True, "unlimited"), (65536, 65536, True, 65536), (65536, -1, False, 65536)]:
    sys.modules["resource"] = fake_resource(soft, hard, can)
    info = G["memlock_info"]()
    print("memlock:", info); assert info["soft_now"] == want and "error" not in info
sys.modules.pop("resource")

# ---------------------------------------------------------------- v3.2 launch arguments (semantic, from the notebook)
G = fresh("args32"); G.update(SGLANG="sglang", PYTHON="python")
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(CODES[5].split("if go():")[0], "launch_defs", "exec"), G)


def opt(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def mamba_slots(args):
    # kv_cache_configurator._calculate_mamba_ratio: 1 if radix cache disabled; base 3 (+2 extra_buffer with overlap)
    if "--disable-radix-cache" in args:
        return 1
    if opt(args, "--mamba-radix-cache-strategy", "no_buffer") == "extra_buffer":
        return 3 + (1 if "--disable-overlap-schedule" in args else 2)
    return 3


def serves_one_request(args):
    return int(opt(args, "--max-mamba-cache-size")) // mamba_slots(args) >= int(opt(args, "--max-running-requests"))


base = G["base_args"]()
assert "--language-model-only" not in base and "--max-mamba-cache-size" not in base and opt(base, "--page-size") == "64"
assert float(opt(base, "--mem-fraction-static")) == 0.90 and opt(base, "--context-length") == "4096"
assert [n for n, _ in G["ATTEMPTS"]] == ["tp4_ep4_noradix_triton_attn", "tp4_ep4_extrabuf_triton_attn"]
for name, extra in G["ATTEMPTS"]:
    a = base + extra
    assert serves_one_request(a), name
    assert opt(a, "--attention-backend") == "triton" and opt(a, "--moe-runner-backend") == "auto", name   # AutoRound W4 MoE: auto only (gptq_kernels.py:360)
    if "--disable-radix-cache" not in a:   # extra_buffer rules (server_args.py _validate_mamba_extra_buffer)
        assert opt(a, "--mamba-radix-cache-strategy") == "extra_buffer" and int(opt(a, "--mamba-track-interval")) % 64 == 0
    print(f"args32: {name:30} mamba slots/req={mamba_slots(a)} cache={opt(a, '--max-mamba-cache-size')} -> serves 1 request")
# regression: the v3.1 combination must be rejected by the same rule
v31 = base + ["--mamba-radix-cache-strategy", "extra_buffer", "--max-mamba-cache-size", "4"]
assert not serves_one_request(v31)
assert serves_one_request(base + ["--mamba-radix-cache-strategy", "extra_buffer", "--max-mamba-cache-size", "5"])
assert serves_one_request(base + ["--disable-radix-cache", "--max-mamba-cache-size", "1"])
print("args32: regression extra_buffer+cache4 -> 0 requests (rejected); +cache5 -> OK; no-radix+cache1 -> OK")

# ---------------------------------------------------------------- v3.3 guard: AutoRound needs MoE runner auto
G = fresh("moe_guard"); G.update(SGLANG="sglang", PYTHON="python", SERVE_ENV={}, mem_available_gb=lambda: 190.0)
G["subprocess"] = types.SimpleNamespace(Popen=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")), run=None, STDOUT=-2)
src = CODES[5].replace('"--moe-runner-backend", "auto", "--max-mamba-cache-size", "2"', '"--moe-runner-backend", "triton", "--max-mamba-cache-size", "2"')
assert src != CODES[5]
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "launch_guard", "exec"), G)
s4 = G["REPORT"]["stages"]["4_launch"]; assert not s4["ok"] and s4["attempts_rejected"] == ["tp4_ep4_noradix_triton_attn"]
print("guard: AutoRound + moe triton -> 4_launch FAIL before any server start")

# ---------------------------------------------------------------- v3.2 pinned-memory probe gates the launch
def pin_case(name, pin_rc, pin_out):
    G = fresh("pin_" + name)
    G.update(PREFIX=str(TMP / "pfx"), VENV=str(TMP / "pfx" / "venv"), PYTHON="python", SGLANG="sglang")
    need_all = " ".join(f"--{w}" for w in ["tensor-parallel-size", "ep-size", "quantization", "ple-offload-embedding",
        "mamba-backend", "linear-attn-decode-backend", "linear-attn-prefill-backend", "disable-cuda-graph",
        "max-mamba-cache-size", "moe-runner-backend", "mamba-radix-cache-strategy", "mamba-track-interval",
        "disable-radix-cache", "attention-backend"])
    calls = {"popen": 0}
    def run_(cmd, **kw):
        if cmd[0] == "sglang":
            return types.SimpleNamespace(returncode=0, stdout=need_all, stderr="")
        if "cudaHostRegister" in cmd[-1]:
            return types.SimpleNamespace(returncode=pin_rc, stdout=pin_out, stderr="")
        return types.SimpleNamespace(returncode=0, stdout="matmul ok", stderr="")
    def popen(*a, **k):
        calls["popen"] += 1; raise RuntimeError("stop after launch was attempted")
    G["subprocess"] = types.SimpleNamespace(run=run_, Popen=popen, STDOUT=-2)
    G["SERVE_ENV"] = {}
    G["mem_available_gb"] = lambda: 190.0    # defined by the stage-0 cell, which this unit test does not run
    os.environ["FN_SMOKE_FAKE_MEMINFO"] = MEM(200, 190)
    ex(G, 4)
    try:
        ex(G, 5)
    except RuntimeError:
        pass
    os.environ.pop("FN_SMOKE_FAKE_MEMINFO")
    st = {k: v["ok"] for k, v in G["REPORT"]["stages"].items()}
    print(f"pin:{name:12} stages={st} server_popen_calls={calls['popen']}")
    return st, calls["popen"]


st, n = pin_case("fails", 1, "cudaHostRegister 64 MiB failed: 1"); assert st.get("3_pinned_memory") is False and n == 0
st, n = pin_case("ok", 0, "pinned 64 MiB OK pinned 1024 MiB OK PIN_PROBE_OK"); assert st.get("3_pinned_memory") and n >= 1

# ---------------------------------------------------------------- never claim success for stages that did not run
G = fresh("incomplete")
G["stage"]("0_environment", True); G["stage"]("1_inputs", True)
out = ex(G, len(CODES) - 1)
assert G["REPORT"]["stopped_at"] == "incomplete_unknown" and "NOT SHOWN TO WORK" in out
print("incomplete run -> stopped_at=incomplete_unknown, verdict NOT SHOWN TO WORK")
shutil.rmtree(TMP, ignore_errors=True)
print("ALL V3 VALIDATION TESTS PASS")
