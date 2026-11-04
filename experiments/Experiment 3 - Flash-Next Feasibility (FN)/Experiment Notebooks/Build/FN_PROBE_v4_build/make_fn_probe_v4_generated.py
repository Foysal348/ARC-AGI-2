"""Build FN-SMOKE: can Qwen3.8-Flash-Next (Intel W4A16 AutoRound) be served by Pennyroyal/SGLang on Kaggle 4xL4 for
ARC-AGI-2, and how does it do on a few evaluation tasks? Every stage is recorded; the first failing stage stops the run
and the report says exactly where. Dual mode: in the scoring rerun it only writes a valid fallback submission."""
import hashlib
import json
from pathlib import Path

DST = Path(r"D:\ARC AGI 2\Research Base Experiment\FN-PROBE-FLASHNEXT-v4.ipynb")

C_HEAD = r'''# FN-PROBE v4 — Qwen3.8-Flash-Next on Kaggle 4×L4: fix the decode crash, measure throughput

**Why:** our Qwen3.8-27B pipeline (Q27-XHIGH v1.4) is limited by tokens: 18/22 queries are cut at 49k thinking tokens
and score 0, while the 4 that finish are right or 1 cell off. 4×L4 gives the 27B ~373 tok/s with 16 running. A faster
model buys more tokens per query. This probe answers one question: how many tokens/s does Flash-Next give on 4×L4?

**v4 changes (from the v3.3 log, checked in the Pennyroyal SGLang source):**
* v3.3 loaded the model and started the server, then crashed on the first decode in `flash_attn/cute` (FA4, Hopper /
  Blackwell only): Qwen4-Exp's full-attention layers use `qwen_sparse_attn_backend.py`, whose decode fallback
  (`_forward_paged_attention`) calls `_resolve_flash_attn_varlen_func()` = classic FA2 if importable, else FA4 CuTe.
  `--attention-backend triton` does not reach it. Stage `3_fa2_shim` installs a PyTorch stand-in for FA2's
  `flash_attn_varlen_func` (`q27_fa2_varlen_shim.py`, tested against a naive reference on CPU and on the GPU in the
  run) and makes the resolver return it on GPUs below sm90.
* KV room: v3.3 had 15,424 KV tokens at mem-fraction 0.90 (weights 19.06 GB of 22 GB). v4: 0.96, 16K context,
  up to 8 running requests (fallback attempt: 0.93, 4 requests).
* Stage 5: greedy sanity (the model must count 1..20). Stage 6: aggregate decode throughput at concurrency 1/4/8
  (ARC evaluation prompts, thinking on, fixed decode length). Optional stage 7: relaunch WITH CUDA graphs and measure
  again. The verdict compares with the 27B's 373 tok/s.

---
(v3.3 header kept for reference)

# FN-SMOKE v3.3 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2

Load-only feasibility probe (`LOAD_ONLY = True`) of the Pennyroyal SGLang fork from dfranzen's ARC-AGI-3 Milestone-2
notebook on 4×L4 (Ada, sm89, 24 GB each). Absolute 50-min deadline; every blocking call is clamped to it.

Model folder (built in stage 1 from symlinks):
* the 15 main W4A16 shards of `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (byte-identical to
  Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@4c67bf6);
* everything else from the overlay `foysalemonshanto/qwen38-flash-next-fp8ple-overlay` (albucino@c6af9fe): 10 FP8 PLE
  shards (RadixArk FP8 table + scale), config with `ple_embedding_dtype: float8_e4m3fn`, index, tokenizer files.
Pennyroyal allocates the pinned host PLE table directly in FP8: 51.2 GB in total (12.8 GB per TP rank).

Stages (each recorded in `fn_smoke_report.json`; the first failure stops the run):
0. environment gates: 4 GPUs, all L4, all sm89, driver ≥ 580 / CUDA 13, MemAvailable ≥ 75 GB, /tmp ≥ 25 GB;
   RLIMIT_MEMLOCK logged and its soft limit raised to the hard limit (informational: the PLE is pinned with
   mmap + cudaHostRegister, not mlock)
1. inputs: merged model folder; every overlay file SHA256-checked; every shard checked by size, safetensors-header
   SHA256 and full SHA256 against the pinned Hugging Face commits
2. install Pennyroyal offline (uv venv)
3. preflight: torch/sglang/sgl_kernel/flashinfer/triton import on all 4 GPUs (8,9); `sglang serve --help` must exit 0
   and list every flag used; then `3_pinned_memory`: cudaHostRegister of 64 MiB and 1 GiB (Pennyroyal's call and flags),
   because RLIMIT_MEMLOCK is 8 MiB on Kaggle — failure stops the run before the model load
4. launch, re-checking MemAvailable right before it: TP4 + EP4, Triton GDN, BF16 KV, 4K context, 1 request,
   mem-fraction-static 0.90, no CUDA graphs, no DP-attention, no speculative decoding, triton attention, MoE runner auto (Marlin; the only runner AutoRound W4 MoE accepts).
   Attempt 1: radix cache disabled (mamba cache 2 slots; 1 needed per request). Attempt 2: mamba radix `extra_buffer`
   with track interval 64, as dfranzen (mamba cache 6 slots; 5 needed per request)
5. one 64-token generation

v3.1 (after the first 4×L4 run): stages 0–3 passed (full SHA256 of all 25 shards OK, 193 s); both launches stopped in
argument resolution because `--language-model-only` is not supported for Qwen4Exp. Removed; radix settings now
follow dfranzen's launcher.

v3.2 (review R3, checked against the Pennyroyal source): mamba cache sized per attempt (v3.1's 4 slots would have
given 0 requests with extra_buffer, after the load); real pinned-memory probe; simplest attempt first; more static
GPU memory, 4K context.

v3.3 (second 4×L4 run): stages 0-3 passed and cudaHostRegister pinned 1 GiB despite the 8 MiB memlock limit. The load
stopped at `gptq_kernels.py:360` (`assert get_moe_runner_backend().is_auto()`): AutoRound W4 MoE accepts only the
`auto` runner (Marlin). Both attempts now use `auto`.

Commit / interactive run: stages 0–5. Competition rerun (`KAGGLE_IS_COMPETITION_RERUN`): only a valid fallback
`submission.json` (this probe is not a solver).
'''

C_CFG = r'''# ==================================================================== CELL 1 — config, mode, report, fallback submission
import os, sys, json, time, glob, shutil, socket, zipfile, subprocess, re, shlex, threading, traceback, hashlib
from pathlib import Path

T0 = time.time()
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
WORK = Path(os.environ.get("FN_SMOKE_WORK", "/kaggle/working")); WORK.mkdir(parents=True, exist_ok=True)   # override only for local tests
OUT = WORK / "fn_smoke"; OUT.mkdir(parents=True, exist_ok=True)
LOAD_ONLY = False                   # v4: the throughput probe replaces the ARC test
SMOKE_BUDGET_SEC = float(os.environ.get("FN_SMOKE_BUDGET", 80 * 60))   # whole-probe wall budget
DEADLINE = T0 + SMOKE_BUDGET_SEC    # absolute; every blocking call is clamped to it
SAFETY_SEC = 60                     # left for report + bundle after the last blocking call
SERVER_ATTEMPT_TIMEOUT = 18 * 60    # per launch attempt (also clamped to the deadline)
MIN_RAM_AVAILABLE_GB = float(os.environ.get("FN_MIN_RAM_GB", 75))   # FP8 PLE pinned 51.2 GB (12.8 GB/rank) + ~24 GB headroom
MIN_TMP_FREE_GB = 25                # Pennyroyal venv + JIT caches
PORT = 8001

def _first(patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None

COMP_DIR = os.environ.get("FN_SMOKE_COMP") or _first(["/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2"])
WHEELHOUSE_DIR = _first(["/kaggle/input/datasets/dfranzen/pennyroyal-v253", "/kaggle/input/pennyroyal-v253"])
INTEL_DIR = os.environ.get("FN_SMOKE_INTEL") or _first([
    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
    "/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/*/*/*"])
OVERLAY_DIR = os.environ.get("FN_SMOKE_OVERLAY") or _first([
    "/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1",
    "/kaggle/input/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1",
    "/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/*/*/*"])
MODEL_DIR = os.environ.get("FN_SMOKE_MODEL_DST", "/tmp/fn_model")   # merged folder, built in stage 1 (symlinks)

REPORT = {"experiment": "FN-PROBE-FLASHNEXT-v4", "rerun": RERUN, "stages": {}, "stopped_at": None,
          "paths": {"competition": COMP_DIR, "wheelhouse": WHEELHOUSE_DIR, "intel": INTEL_DIR, "overlay": OVERLAY_DIR,
                    "model": MODEL_DIR}}

def stage(name, ok, **info):
    REPORT["stages"][name] = {"ok": bool(ok), "t_sec": round(time.time() - T0, 1), **info}
    if not ok and REPORT["stopped_at"] is None:
        REPORT["stopped_at"] = name
    (OUT / "fn_smoke_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info, default=str)[:1500], flush=True)

def remaining():
    return max(0.0, DEADLINE - time.time())

def clamp(t):
    """timeout for a blocking call: never past DEADLINE - SAFETY_SEC"""
    return max(1.0, min(float(t), remaining() - SAFETY_SEC))

def go():
    if REPORT["stopped_at"] is not None:
        return False
    if remaining() <= SAFETY_SEC:
        stage("global_deadline", False, reason="smoke wall-time budget exhausted", budget_sec=SMOKE_BUDGET_SEC)
        return False
    return True

def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=clamp(timeout))
        return (r.stdout + r.stderr)[-20000:]
    except Exception as e:
        return repr(e)

# valid submission.json first (test inputs as answers) — both modes
def write_fallback_submission():
    test = json.load(open(Path(COMP_DIR) / "arc-agi_test_challenges.json"))
    sub = {t: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in v["test"]] for t, v in test.items()}
    (WORK / "submission.json").write_text(json.dumps(sub))
    return len(sub)

n_sub = write_fallback_submission() if COMP_DIR else 0
print("MODE:", "COMPETITION RERUN (fallback submission only)" if RERUN else "SMOKE PROBE", "| fallback submission tasks:", n_sub)
if RERUN:
    REPORT["stopped_at"] = "rerun_mode_fallback_only"
'''

C_ENV = r'''# ==================================================================== CELL 2 — stage 0: environment gates
def stage0_check(smi, gpu_csv, meminfo_text, tmp_free_gb):
    m_drv = re.search(r"Driver Version:\s*([\d.]+)", smi or ""); m_cuda = re.search(r"CUDA Version:\s*([\d.]+)", smi or "")
    driver = m_drv.group(1) if m_drv else None; driver_cuda = m_cuda.group(1) if m_cuda else None
    gpus = [g.strip() for g in (gpu_csv or "").strip().splitlines() if g.strip() and "," in g]
    names = [g.split(",")[0].strip() for g in gpus]; ccs = [g.split(",")[-1].strip() for g in gpus]
    mem = {}
    for line in (meminfo_text or "").splitlines():
        k, _, rest = line.partition(":"); p = rest.split()
        if p and p[0].isdigit(): mem[k] = int(p[0]) * 1024
    ram_total = round(mem.get("MemTotal", 0) / 1e9, 1); ram_avail = round(mem.get("MemAvailable", 0) / 1e9, 1)
    checks = {
        "four_gpus": len(gpus) == 4,
        "all_l4": bool(names) and all("L4" in n for n in names),
        "all_sm89": bool(ccs) and all(c == "8.9" for c in ccs),
        "driver_r580_plus": driver is not None and int(driver.split(".")[0]) >= 580,
        "driver_cuda13_plus": driver_cuda is not None and float(driver_cuda.split(".")[0]) >= 13,
        "ram_available_enough": ram_avail >= MIN_RAM_AVAILABLE_GB,
        "tmp_free_enough": tmp_free_gb >= MIN_TMP_FREE_GB,
    }
    info = {"driver": driver, "driver_max_cuda": driver_cuda, "gpus": gpus, "ram_total_gb": ram_total,
            "ram_available_gb": ram_avail, "tmp_free_gb": round(tmp_free_gb, 1), "checks": checks}
    return all(checks.values()), info

def memlock_info():
    # informational: Pennyroyal pins the PLE with mmap + cudaHostRegister (no mlock); raise soft to hard anyway,
    # the server inherits the limit from this process
    try:
        import resource
        fmt = lambda v: "unlimited" if v == resource.RLIM_INFINITY else v
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        raised = False
        if soft != hard:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard)); raised = True
            except (ValueError, OSError):
                pass
        soft2, hard2 = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        return {"soft_before": fmt(soft), "hard": fmt(hard2), "soft_now": fmt(soft2), "raised": raised}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

def mem_available_gb():
    try:
        txt = os.environ.get("FN_SMOKE_FAKE_MEMINFO") or open("/proc/meminfo").read()
    except OSError:
        return 0.0
    for line in txt.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    return 0.0

if go():
    smi = os.environ.get("FN_SMOKE_FAKE_SMI") or sh("nvidia-smi")
    gpu_csv = os.environ.get("FN_SMOKE_FAKE_GPUS") or sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader")
    try:
        meminfo = os.environ.get("FN_SMOKE_FAKE_MEMINFO") or open("/proc/meminfo").read()
    except OSError:
        meminfo = ""
    tmp_free = float(os.environ.get("FN_SMOKE_FAKE_TMPFREE") or shutil.disk_usage("/tmp" if os.path.isdir("/tmp") else ".").free / 1e9)
    ok0, info0 = stage0_check(smi, gpu_csv, meminfo, tmp_free)
    stage("0_environment", ok0, **info0, python=sys.version.split()[0], cpu_count=os.cpu_count(),
          memlock=memlock_info(), disk_working=sh("df -h /kaggle/working | tail -1").strip())
'''

C_INPUTS = r'''# ==================================================================== CELL 3 — stage 1: inputs, merged model folder
# expected values from the pinned Hugging Face commits (albucino c6af9fe / Intel 4c67bf6):
#   shards: name -> [bytes, full sha256, sha256 of the safetensors header (8-byte length + JSON), header length]
#   files:  every small overlay file -> [bytes, sha256]
EXPECT = {"shards":{"model-00001-of-00017.safetensors":[2899311784,"70734bddc4db7476d4ba03a86f808290e88488d39bf513c639ad3957768692de","c8a3adef4e064142c043ac15eb5684bbca6aa8d382ac92f13e060160add2ca16",1257480],"model-00003-of-00017.safetensors":[5370548192,"0ecb3e52cad98e4c3c042432f3db2f12ca6c41ea6844a964f29e66f673fb5be5","e402fea51350325e2ef36543d7229647f0bffee8d2e46dd7984fa801f207a037",2270104],"model-00004-of-00017.safetensors":[5370910416,"2865e0756b9a3cc0e045eeec349d3e495d1630476cc5355ea0da6a272092d9b0","b5911cc51e1f0cdd92997c2b9e9fbbc777c624120ba8fd6963b291def150a7b1",2296968],"model-00005-of-00017.safetensors":[5370140704,"5f0b9642174a7bfbfa57c6fda7917effeb3afb23c2377cfc5f8f9e29222dd048","ca9dbcd24ecdd9ccd299dc492e17e904fcb74dae30d31787699080719dc6bf55",2381464],"model-00006-of-00017.safetensors":[5370205184,"0d06de9de63a1cdabd39a6c11ea1b6431b24be17c4a51fe4db236ea77c33d0b9","d32809b6820acdb3dcafe2c0038f17648099478bd1bd671fc0c467934132be7c",2314936],"model-00007-of-00017.safetensors":[5370927288,"a632e777190fac1ee19b8706925862af7b4f3ce44e58b40f01a79c0ab7697e96","e22724f5fd969bd22c5b2d7d1efee8d4d2e50fbbd1c6419cd4febad13a10e702",2313840],"model-00008-of-00017.safetensors":[5344445352,"cfc3279b2981d2b6c319b6be03d33091c0dea1b94d194705450b4cece4df6a7d","1966e17ed55878d5d9ad2be412f909542d5ee0225b4803cc740926eb2c544194",2358624],"model-00009-of-00017.safetensors":[5370355648,"f3f8390dc32f3a2ea3980fa1a44cc37d02394c40388bd4688b16779431791327","e549fa1e0fefe530c0f79e31c5e136d59d575a5cd5275874c256da3a03919cda",2328888],"model-00010-of-00017.safetensors":[5370927288,"48f9d03732bc89e177b946da67702e3eeec55d185a4c795dfebfecf380014d42","5e5f041eabe2b3522d330fbe9937543449f562411b217b69a59a79cabcaff7dd",2313840],"model-00011-of-00017.safetensors":[5324173744,"8592f8599a883e1a57f4010b3cfa7516e8f1c5777d21a62f9b0e20e06e56840c","adcb6b830d6bfcad671dc54a339648930115fe7239de51aeb85b6c3fd0676d9b",2341992],"model-00012-of-00017.safetensors":[5370438520,"6d58f6a6ac680e6cde7e061025a55cac650ec32f00a7ae1fb6aa95839971ba40","046d7f0d43aa2395f24cc9cd027ce27d505c99ba31a33581094bfb90437d7a2b",2329392],"model-00013-of-00017.safetensors":[5370927288,"901b9e3f509df9a051d99ad5a9815339b68b4e78006805175e7c1f75423883c0","f5d8d767720ff4c4280df86cc784aa22f32a59107dd0654e57d5df43eb15bb5e",2313840],"model-00014-of-00017.safetensors":[5342689856,"a77beb75499e376d0de54232d510f533eaf730149c4d9f69e5b327850b3da66b","5d9ac041730163503ba64f60d7b8ef3a3826f92323c5f7571f8ef9c5d1123efa",2342648],"model-00015-of-00017.safetensors":[5050474736,"f457d062094bda4c6bd486fe373ae56d04f2b205ef083e95cfdee9400da86d2c","bf736d33acb319bc64450042013d713b68c58480af478ade491b7338a458ed3a",1304840],"model-00017-of-00017.safetensors":[1284526608,"fdbe0ca6b1bfab1dece276c48efadbc7d24ebb4c2619497c19db47b400f31782","69bbcc7eabd49c8aafc24e718d92db11a5d76cee27cdcb385227b795ccc9f673",520],"model-plefp8-00000.safetensors":[5200027199,"dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd","435bdcd0c1900a90eb1e591ba41638447ffb35ea85abdda593f4ecb30c8c983d",2231],"model-plefp8-00001.safetensors":[5200027202,"899eaa0716e28594468a1389ee58cb065c23907f1270de3831f5ecb0a4f82d56","915ee1fc1d6f084916b77ef4e81549e4559a0a82c9932822ab521e919a5c2375",2234],"model-plefp8-00002.safetensors":[5200027198,"06fd8a11abf0419a669f89397b8d70dd6ff42d401e6b2a037c65e49704faaf71","c35a43c2cf337cb67cf8cf7723567ffe01904b4bf09ac24c6fb5f9fd1475a600",2230],"model-plefp8-00003.safetensors":[5200027189,"c6aaa1fc08e84eced3c8151ac8679ed943888eba0fcef2556963693430f95bd9","4d204e7713fbb70101bcdf48daa1013caa0abf8649156f8c5ff73bfb97f3a78a",2221],"model-plefp8-00004.safetensors":[5200027190,"d94e97c96d3ea09208614da016960f3f4b429f47a044c898a51b493f42f74ba2","d968afd72f5e8790732bb7ab81748a64afbeaa1bbf91fd095d7a326d2e381185",2222],"model-plefp8-00005.safetensors":[5200027190,"03d5d4792e14a4ab55bae50bb459624b01786f818bbcfcaed1a5d0235af484c6","09298a74d7b5027e9011fe276ee9160b20633a5f4fbb7b1e44a34d3650efafd5",2222],"model-plefp8-00006.safetensors":[5200027190,"586cccbc12383021bc9bc02f206d9b19fbd1672373ed9a5d91e3b0ce34c2418f","47c64778d21384aa38488419c17008be79f3aa8414bc6b7f73ba264b6075f695",2222],"model-plefp8-00007.safetensors":[5200027190,"c4a23bcc10f3cde6b633e82c282a9a518e3a64d433628283e1bc592c94cf3d6c","96237d882cac517da8fc2192da53cf1816b57295d9ada1c53d7d06bee4154c33",2222],"model-plefp8-00008.safetensors":[5200027190,"2dc8098c0d020bff277c9cf499a6b908e17836b70a5f949dfa24793371c9a87e","01af649c81cb57c52cdcecb20d2cae509843d087f3620e0d29a5324482e9c4d1",2222],"model-plefp8-00009.safetensors":[4400023163,"61de98b89bb79f386795787d7a76827a26f1e292c26edbb0a1b613da455f5a9c","433b6ba953bea80d2562f694dc825623045f329499098a49efc0ad92a15b0acb",2033]},"files":{"LICENSE":[3235,"a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"],"README.md":[15547,"3c1ecf16c50cdca78d208dedc5a70cc974e154a83e2e5ea62f6df072cdd37497"],"SHA256SUMS":[9453,"e8a0a6ebd68053b35d28f08e6164354351867c9c9fcf9ada81fee5a557d61b80"],"chat_template.jinja":[8952,"c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"],"config.json":[266471,"2925a0d0989384e0ef7973a14c97ba9c8dfc82c4ca3684d2e83cf4f57db6c172"],"generation_config.json":[214,"6ef70d6670f24f03a9041724c7663ef26bc1e678d10f79c93fe11c192ba14b7d"],"model.safetensors.index.json":[23657880,"eb08a649acf89aa2d0c5c1858704cba8e86cdf2b93dec8cf9a89024292a3cc5a"],"preprocessor_config.json":[443,"3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2"],"tokenizer.json":[19989325,"06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"],"tokenizer_config.json":[1165,"792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"],"video_preprocessor_config.json":[385,"7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"]}}
FULL_HASH = os.environ.get("FN_SMOKE_FULL_HASH", "1") == "1"   # full SHA256 of all 25 shards (also warms the page cache)

def _link(src, dst):
    if os.environ.get("FN_SMOKE_LINK") == "copy":   # local tests only (no symlink privilege on Windows)
        shutil.copyfile(src, dst)
    else:
        os.symlink(src, dst)

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()

def build_model_dir(intel_dir, overlay_dir, dest, expect, full_hash=True, hash_timeout=1800):
    # The overlay is authoritative for every file except weights: config, index, tokenizer, chat template, generation /
    # preprocessor configs (all SHA256-checked). From Intel's mount only the weight files named in the overlay index and
    # absent from the overlay are linked: the 15 main shards. Intel's BF16 PLE, MTP file, quantization_config.json and
    # other metadata are never linked.
    problems = []
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True); dest.mkdir(parents=True)
    for name, (size, sha) in expect["files"].items():
        p = Path(overlay_dir, name)
        if not p.is_file():
            problems.append(f"overlay missing {name}")
        elif p.stat().st_size != size or _sha256(p) != sha:
            problems.append(f"overlay {name} differs from albucino c6af9fe")
    if problems:
        return False, {"problems": problems}
    for name in expect["files"]:
        _link(str(Path(overlay_dir, name)), str(dest / name))
    index = json.loads((dest / "model.safetensors.index.json").read_text())
    weights = sorted(set(index["weight_map"].values()))
    sources = {}
    for name in weights:
        in_overlay = Path(overlay_dir, name).is_file()
        src = Path(overlay_dir if in_overlay else intel_dir, name)
        if not src.is_file():
            problems.append(f"missing weight file {name}")
            continue
        sources[name] = src
        _link(str(src), str(dest / name))
        exp = expect["shards"].get(name)
        if exp is None:
            problems.append(f"unexpected weight file {name}")
            continue
        size, _, head_sha, head_len = exp
        if src.stat().st_size != size:
            problems.append(f"{name}: {src.stat().st_size} bytes, expected {size}")
            continue
        with open(src, "rb") as f:
            head = f.read(8 + head_len)
        if hashlib.sha256(head).hexdigest() != head_sha:
            problems.append(f"{name}: safetensors header differs from the pinned commit")
    if set(weights) != set(expect["shards"]):
        problems.append(f"index names {len(weights)} weight files, expected {len(expect['shards'])}")
    hash_info = {"full_hash": bool(full_hash)}
    if full_hash and not problems:
        import concurrent.futures as cf
        t = time.time()
        ex = cf.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
        futs = {ex.submit(_sha256, src): name for name, src in sources.items()}
        try:
            for fut in cf.as_completed(futs, timeout=hash_timeout):
                name = futs[fut]
                if fut.result() != expect["shards"][name][1]:
                    problems.append(f"{name}: full SHA256 differs from the pinned commit")
        except cf.TimeoutError:
            problems.append(f"full SHA256 did not finish within {round(hash_timeout)} s")
        ex.shutdown(wait=False, cancel_futures=True)
        gb = sum(src.stat().st_size for src in sources.values()) / 1e9
        hash_info.update(hash_sec=round(time.time() - t, 1), hashed_gb=round(gb, 1),
                         hash_gb_per_sec=round(gb / max(time.time() - t, 1e-6), 2))
    cfg = json.loads((dest / "config.json").read_text())
    ple_dtype = cfg.get("text_config", {}).get("ple_embedding_dtype")
    if ple_dtype != "float8_e4m3fn":
        problems.append(f"config ple_embedding_dtype={ple_dtype!r}")
    q = cfg.get("quantization_config", {})
    info = {"weight_files": len(weights),
            "from_intel": sum(src.parent == Path(intel_dir) for src in sources.values()),
            "from_overlay": sum(src.parent == Path(overlay_dir) for src in sources.values()),
            "weights_gb": round(sum(src.stat().st_size for src in sources.values()) / 1e9, 1),
            "ple_embedding_dtype": ple_dtype, "model_type": cfg.get("model_type"),
            "quant": {k: q.get(k) for k in ("quant_method", "bits", "group_size")},
            "linked_other": sorted(p.name for p in dest.iterdir() if not p.name.endswith(".safetensors")),
            **hash_info, "problems": problems}
    return not problems, info

if go():
    ok = all([COMP_DIR, WHEELHOUSE_DIR, INTEL_DIR, OVERLAY_DIR])
    info = {"missing_inputs": [k for k, v in {"competition": COMP_DIR, "wheelhouse": WHEELHOUSE_DIR,
                                              "intel": INTEL_DIR, "overlay": OVERLAY_DIR}.items() if not v]}
    if ok:
        try:
            t = time.time()
            ok, merged = build_model_dir(INTEL_DIR, OVERLAY_DIR, MODEL_DIR, EXPECT, full_hash=FULL_HASH,
                                         hash_timeout=clamp(30 * 60))
            info.update(merged, merge_sec=round(time.time() - t, 1))
        except Exception as e:
            ok = False; info["error"] = f"{type(e).__name__}: {e}"[:2000]
    if WHEELHOUSE_DIR:
        info["sglang_wheels"] = [p.name for p in Path(WHEELHOUSE_DIR).rglob("sglang-*.whl")]
    stage("1_inputs", ok, **info)
'''

C_INSTALL = r'''# ==================================================================== CELL 4 — stage 2: install Pennyroyal (offline)
PREFIX = "/tmp/sgl-intel"; VENV = f"{PREFIX}/venv"; PYTHON = f"{VENV}/bin/python"; SGLANG = f"{VENV}/bin/sglang"
if go():
    t = time.time()
    try:
        WHEELS = next(p for p in Path(WHEELHOUSE_DIR).rglob("wheels") if p.is_dir())
        LOCK = next(Path(WHEELHOUSE_DIR).rglob("requirements.lock"))
        wheel = sorted(WHEELS.glob("sglang-*.whl"))[0]
        install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1"); install_env.pop("PYTHONPATH", None)
        uv_whl = sorted(WHEELS.glob("uv-*.whl"))[0]
        uv_bin = Path(PREFIX) / "bin/uv"; uv_bin.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(uv_whl) as a:
            member = [n for n in a.namelist() if Path(n).name == "uv" and not n.endswith("/")][0]
            uv_bin.write_bytes(a.read(member))
        uv_bin.chmod(0o755)
        def run(cmd, env=install_env, timeout=1800):
            r = subprocess.run(list(map(str, cmd)), text=True, capture_output=True, env=env, timeout=clamp(timeout))
            if r.returncode:
                raise RuntimeError(f"{shlex.join(map(str, cmd))} -> {r.returncode}\n{(r.stdout + r.stderr)[-4000:]}")
            return r
        run([uv_bin, "venv", "--python", sys.executable, VENV])
        run([uv_bin, "pip", "install", "--python", PYTHON, "--no-index", "--find-links", WHEELS, "-r", LOCK])
        run([uv_bin, "pip", "install", "--python", PYTHON, "--no-index", "--find-links", WHEELS, "--reinstall", "--no-deps", wheel])
        stage("2_install", True, sec=round(time.time() - t, 1), sglang_wheel=wheel.name)
    except Exception as e:
        stage("2_install", False, sec=round(time.time() - t, 1), error=f"{type(e).__name__}: {e}"[:4000])
'''

C_CUDA = r'''# ==================================================================== CELL 5 — stage 3: torch + CUDA toolkit in the venv

# ---- v4: FA2 varlen stand-in (see the header)
SHIM_SRC = '"""Stand-in for flash_attn (FA2) `flash_attn_varlen_func` on Ada GPUs (L4, sm89).\n\nSGLang\'s Qwen4-Exp sparse-attention decode fallback (qwen_sparse_attn_backend._forward_paged_attention) calls a dense\nvarlen kernel on the packed top-k K/V of every request: classic FA2 if importable, else flash-attn-4\'s CuTe interface,\nwhich only builds on Hopper/Blackwell (FN-SMOKE v3.3 crashed there with an MLIRError on L4). This module computes the\nsame thing in PyTorch.\n\nSemantics follow FA2: q [total_q, Hq, D], k/v [total_k, Hkv, D(v)], cu_seqlens_* int32 prefix sums, GQA maps query head\nh to KV head h // (Hq // Hkv), `causal` aligns the query block to the END of its keys. Rows with no key give zeros.\nReturns [total_q, Hq, Dv] in q\'s dtype. The decode branch (one query per sequence) does no host synchronisation.\n"""\nimport torch\n\n\ndef flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,\n                           softmax_scale=None, causal=False, **kwargs):\n    total_q, hq, d = q.shape\n    hkv, dv = k.shape[1], v.shape[-1]\n    g = hq // hkv\n    b = cu_seqlens_q.numel() - 1\n    scale = float(softmax_scale) if softmax_scale is not None else d ** -0.5\n    if total_q == b:                     # decode: one query row per sequence -> the query sees all of its keys\n        lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.long)\n        maxk = max(int(max_seqlen_k), 1)\n        ar = torch.arange(maxk, device=q.device)\n        valid = ar[None, :] < lens[:, None]                                   # [b, maxk]\n        idx = cu_seqlens_k[:-1].to(torch.long)[:, None] + ar[None, :]\n        idx = torch.where(valid, idx, torch.zeros_like(idx)).clamp(max=max(k.shape[0] - 1, 0))\n        kk = k[idx].float()                                                   # [b, maxk, hkv, d]\n        vv = v[idx].float()\n        qf = q.float().reshape(b, hkv, g, d)\n        s = torch.einsum("bhgd,bkhd->bhgk", qf, kk) * scale\n        s = s.masked_fill(~valid[:, None, None, :], float("-inf"))\n        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)\n        return torch.einsum("bhgk,bkhd->bhgd", p, vv).reshape(b, hq, dv).to(q.dtype)\n    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()                     # general varlen (prefill-shaped)\n    outs = []\n    for i in range(b):\n        qi = q[cq[i]:cq[i + 1]].float()\n        ki = k[ck[i]:ck[i + 1]].float().repeat_interleave(g, dim=1)\n        vi = v[ck[i]:ck[i + 1]].float().repeat_interleave(g, dim=1)\n        lq, lk = qi.shape[0], ki.shape[0]\n        s = torch.einsum("qhd,khd->hqk", qi, ki) * scale\n        if causal:\n            keep = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(diagonal=lk - lq)\n            s = s.masked_fill(~keep[None], float("-inf"))\n        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)\n        outs.append(torch.einsum("hqk,khd->qhd", p, vi))\n    return torch.cat(outs).to(q.dtype) if outs else q.new_zeros(0, hq, dv)\n'
SHIM_MARK = 'Q27 FN-PROBE v4: Ada'
SHIM_OLD = '    try:\n        from flash_attn import flash_attn_varlen_func\n\n        return flash_attn_varlen_func\n    except ImportError:\n        pass\n'
SHIM_NEW = ('    try:   # Q27 FN-PROBE v4: Ada (sm89) has no FA2 in this venv and FA4 CuTe does not build for it\n'
            '        import torch as _q27_torch\n'
            '        if _q27_torch.cuda.is_available() and _q27_torch.cuda.get_device_capability()[0] < 9:\n'
            '            from q27_fa2_varlen_shim import flash_attn_varlen_func\n\n'
            '            return flash_attn_varlen_func\n'
            '    except ImportError:\n'
            '        pass\n')
SHIM_GPU_TEST = 'import torch, math\nfrom q27_fa2_varlen_shim import flash_attn_varlen_func as f\ntorch.manual_seed(0); dev = "cuda"\nlens = [7, 0, 33, 2, 2048]; ck = [0]\nfor n in lens: ck.append(ck[-1] + n)\nb = len(lens); hq, hkv, d = 8, 2, 128\nq = torch.randn(b, hq, d, device=dev, dtype=torch.bfloat16)\nk = torch.randn(ck[-1], hkv, d, device=dev, dtype=torch.bfloat16); v = torch.randn(ck[-1], hkv, d, device=dev, dtype=torch.bfloat16)\no = f(q, k, v, torch.arange(b + 1, dtype=torch.int32, device=dev), torch.tensor(ck, dtype=torch.int32, device=dev), 1, max(lens),\n      softmax_scale=d ** -0.5, causal=True)\nerr = 0.0\nfor i, n in enumerate(lens):\n    if n == 0:\n        assert o[i].abs().max().item() == 0.0; continue\n    kk = k[ck[i]:ck[i + 1]].float().repeat_interleave(hq // hkv, 1); vv = v[ck[i]:ck[i + 1]].float().repeat_interleave(hq // hkv, 1)\n    s = torch.einsum("hd,khd->hk", q[i].float(), kk) / math.sqrt(d)\n    err = max(err, (o[i].float() - torch.einsum("hk,khd->hd", s.softmax(-1), vv)).abs().max().item())\nassert err < 3e-2, err\nprint("SHIM_GPU_OK", round(err, 5), torch.cuda.get_device_name(0))\n'
PIN_PROBE = """
import mmap, time, torch
torch.zeros(1, device="cuda")                       # create the CUDA context on GPU 0
FLAGS = 0x01 | 0x02                                 # cudaHostRegisterPortable | Mapped, as in qwen4_exp._allocate_pinned_table
for mib in (64, 1024):
    n = mib << 20
    buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    raw = torch.frombuffer(buf, dtype=torch.uint8)
    t = time.time()
    err = torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), n, FLAGS)
    if int(err) != 0:
        raise SystemExit(f"cudaHostRegister {mib} MiB failed: {err}")
    gpu = torch.empty(n, dtype=torch.uint8, device="cuda"); gpu.copy_(raw, non_blocking=True); torch.cuda.synchronize()
    err = torch.cuda.cudart().cudaHostUnregister(raw.data_ptr())
    if int(err) != 0:
        raise SystemExit(f"cudaHostUnregister {mib} MiB failed: {err}")
    del raw, gpu; buf.close()
    print(f"pinned {mib} MiB OK in {time.time() - t:.2f}s")
print("PIN_PROBE_OK")
"""
if go():
    try:
        cu13 = sorted(Path(VENV).glob("lib/python*/site-packages/nvidia/cu13"))
        CUDA_HOME = str(cu13[0]) if cu13 else None
        env = dict(os.environ); env.pop("PYTHONPATH", None)
        cache = f"{PREFIX}/cache"; Path(cache).mkdir(parents=True, exist_ok=True)
        if CUDA_HOME:
            libdir = Path(CUDA_HOME) / "lib"; lib64 = Path(CUDA_HOME) / "lib64"
            if not lib64.exists() and not lib64.is_symlink(): lib64.symlink_to("lib")
            for so in sorted(libdir.glob("*.so.*")):
                link = libdir / re.sub(r"\.so\..*$", ".so", so.name)
                if not link.exists() and not link.is_symlink(): link.symlink_to(so.name)
            if not (libdir / "libcuda.so").exists():
                cands = []
                for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
                    cands += sorted(p for p in Path(d).glob("libcuda.so*") if p.is_file())
                if cands: (libdir / "libcuda.so").symlink_to(cands[0].resolve())
        CXX = next((c for c in ("g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
        env.update({"PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "TORCH_CUDA_ARCH_LIST": "8.9",            # L4 (Ada); dfranzen used 12.0 (RTX PRO 6000)
                    "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "0", "SGLANG_SM120_ONLINE_MXFP8": "0",
                    "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
                    "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
                    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "TOKENIZERS_PARALLELISM": "false",
                    "OMP_NUM_THREADS": "8", "MAX_JOBS": "8",
                    "XDG_CACHE_HOME": cache, "TRITON_CACHE_DIR": f"{cache}/triton", "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
                    "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit"})
        if CUDA_HOME:
            env.update({"CUDA_HOME": CUDA_HOME, "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
                        "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:" + env.get("PATH", ""),
                        "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{CUDA_HOME}/lib:" + env.get("LD_LIBRARY_PATH", "")})
        if CXX: env.update({"CC": CXX.replace("g++", "gcc"), "CXX": CXX, "CUDAHOSTCXX": CXX})
        SERVE_ENV = env
        probe = ("import torch, sglang, sgl_kernel, flashinfer, triton;"
                 "print('sglang', sglang.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__);"
                 "assert torch.cuda.is_available() and torch.cuda.device_count() == 4, torch.cuda.device_count();"
                 "caps=[torch.cuda.get_device_capability(i) for i in range(4)]; print('caps', caps);"
                 "assert all(c == (8, 9) for c in caps), caps;"
                 "x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); print('matmul ok', tuple(y.shape))")
        r = subprocess.run([PYTHON, "-c", probe], text=True, capture_output=True, env=env, timeout=clamp(600))
        need = ["--tensor-parallel-size", "--ep-size", "--quantization", "--ple-offload-embedding",
                "--mamba-backend", "--linear-attn-decode-backend", "--linear-attn-prefill-backend", "--disable-cuda-graph",
                "--max-mamba-cache-size", "--moe-runner-backend", "--mamba-radix-cache-strategy",
                "--mamba-track-interval", "--disable-radix-cache", "--attention-backend"]
        h = subprocess.run([SGLANG, "serve", "--help"], text=True, capture_output=True, env=env, timeout=clamp(300))
        help_text = h.stdout + h.stderr
        missing = [f for f in need if f not in help_text]
        stage("3_preflight", r.returncode == 0 and h.returncode == 0 and not missing, cuda_home=CUDA_HOME, cxx=CXX,
              probe_output=(r.stdout + r.stderr)[-3000:], missing_flags=missing, help_exit=h.returncode)
        if REPORT["stopped_at"] is None:
            # RLIMIT_MEMLOCK is 8 MiB on Kaggle: prove that cudaHostRegister (Pennyroyal's exact call and flags) can pin
            # far more than that before the 125 GB load. The PLE needs ~12.8 GB per rank.
            pin = subprocess.run([PYTHON, "-c", PIN_PROBE], text=True, capture_output=True, env=env, timeout=clamp(600))
            stage("3_pinned_memory", pin.returncode == 0 and "PIN_PROBE_OK" in pin.stdout,
                  memlock=REPORT["stages"].get("0_environment", {}).get("memlock"),
                  output=(pin.stdout + pin.stderr)[-2500:])
    except Exception as e:
        stage("3_preflight", False, error=f"{type(e).__name__}: {e}"[:3000])
    # ---- v4: stage 3_fa2_shim - PyTorch stand-in for FA2 flash_attn_varlen_func in the Qwen4-Exp QSA decode fallback
    if REPORT["stopped_at"] is None:
        try:
            purelib = subprocess.run([PYTHON, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True,
                                     capture_output=True, env=env, timeout=clamp(120)).stdout.strip()
            (Path(purelib) / "q27_fa2_varlen_shim.py").write_text(SHIM_SRC)
            target = Path(purelib) / "sglang" / "srt" / "layers" / "attention" / "qwen_sparse_attn_backend.py"
            text = target.read_text()
            if SHIM_MARK not in text:
                assert text.count(SHIM_OLD) == 1, "patch anchor not found exactly once"
                target.write_text(text.replace(SHIM_OLD, SHIM_NEW + SHIM_OLD))
                for pyc in target.parent.glob("__pycache__/qwen_sparse_attn_backend*.pyc"):
                    pyc.unlink()
            t1 = subprocess.run([PYTHON, "-c", SHIM_GPU_TEST], text=True, capture_output=True, env=env, timeout=clamp(600))
            t2 = subprocess.run([PYTHON, "-c", "from sglang.srt.layers.attention.qwen_sparse_attn_backend import "
                                 "_resolve_flash_attn_varlen_func as r; print('RESOLVED', r().__module__)"],
                                text=True, capture_output=True, env=env, timeout=clamp(600))
            ok = ("SHIM_GPU_OK" in t1.stdout) and ("RESOLVED q27_fa2_varlen_shim" in t2.stdout)
            stage("3_fa2_shim", ok, target=str(target), gpu_test=(t1.stdout + t1.stderr)[-1500:],
                  resolver=(t2.stdout + t2.stderr)[-1500:])
        except Exception as e:
            stage("3_fa2_shim", False, error=f"{type(e).__name__}: {e}"[:2000])
'''

C_LAUNCH = r'''# ==================================================================== CELL 6 — stage 4: launch the server (ordered attempts)
LOG = OUT / "serve.log"
SERVER = None
def base_args():
    # minimal first-boot profile (review): add one variable at a time later (context, KV dtype, concurrency, ...)
    return [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",
            "--served-model-name", "flashnext", "--host", "127.0.0.1", "--port", str(PORT),
            "--dtype", "bfloat16", "--quantization", "auto-round", "--kv-cache-dtype", "auto",
            "--mem-fraction-static", "0.96", "--context-length", "16384", "--page-size", "64",
            "--max-running-requests", "8", "--chunked-prefill-size", "2048", "--max-prefill-tokens", "4096",
            "--mamba-ssm-dtype", "bfloat16",
            "--ple-offload-embedding", "--trust-remote-code",
            "--reasoning-parser", "qwen3", "--watchdog-timeout", "1800", "--disable-cuda-graph",
            "--tensor-parallel-size", "4", "--ep-size", "4",
            "--mamba-backend", "triton", "--linear-attn-decode-backend", "triton", "--linear-attn-prefill-backend", "triton"]
# Not tried: FlashInfer GDN (gdn_flashinfer.py requires SM90+); DP-attention (the PLE would be replicated on every rank,
# 4 x 51 GB FP8 pinned host RAM); plain TP4 (experts split to 160 columns, not a multiple of the INT4 group 128).
# v3.1: compressed QSA pins page_size=64, and the default mamba radix strategy (no_buffer) requires page_size 1 ->
# use dfranzen's extra_buffer (needs linear_attn_backend triton, the default) or disable the radix cache.
# v3.2: mamba state slots per request (kv_cache_configurator.py:1919-1947): 1 with the radix cache disabled, 5 with
# extra_buffer (3 + 2, overlap schedule on); max requests = max_mamba_cache_size // slots, 0 raises after the load.
MAMBA_SLOTS_PER_REQ = {"no_radix": 1, "extra_buffer": 5}
# v3.3: AutoRound W4 MoE = GPTQ-Marlin MoE method, which asserts the runner is `auto` (gptq_kernels.py:360).
ATTEMPTS = [   # v4: radix cache off -> 1 mamba slot per request; 8 requests + margin
    ("noradix_eager_mf096_r8", ["--disable-radix-cache", "--attention-backend", "triton", "--moe-runner-backend", "auto",
                                "--max-mamba-cache-size", "10"]),
    ("noradix_eager_mf093_r4", ["--disable-radix-cache", "--attention-backend", "triton", "--moe-runner-backend", "auto",
                                "--max-mamba-cache-size", "6", "--mem-fraction-static", "0.93", "--max-running-requests", "4"]),
]
def kill_server():
    global SERVER
    if SERVER is not None and SERVER.poll() is None:
        try: os.killpg(SERVER.pid, 15); SERVER.wait(timeout=60)
        except Exception:
            try: os.killpg(SERVER.pid, 9)
            except Exception: pass
    SERVER = None   # only the tracked process group is killed; nothing else on the machine

if go():
    ram_now = mem_available_gb()
    if ram_now < MIN_RAM_AVAILABLE_GB:
        stage("4_launch", False, reason="MemAvailable below the gate right before launch",
              ram_available_gb=round(ram_now, 1), min_gb=MIN_RAM_AVAILABLE_GB)
if go():
    # AutoRound W4 MoE = GPTQ-Marlin MoE method: only the `auto` runner is accepted (gptq_kernels.py:360) -> reject a
    # wrong runner here instead of ~70 s into a load
    _b = base_args(); _q = _b[_b.index("--quantization") + 1] if "--quantization" in _b else None
    _bad = [n for n, ex in ATTEMPTS if _q == "auto-round"
            and (ex[ex.index("--moe-runner-backend") + 1] if "--moe-runner-backend" in ex else "auto") != "auto"]
    if _bad:
        stage("4_launch", False, reason="AutoRound W4 MoE accepts only --moe-runner-backend auto", attempts_rejected=_bad)
if go():
    results = []
    for name, extra in ATTEMPTS:
        if remaining() < 10 * 60 + SAFETY_SEC:   # a load needs >= ~10 min; never start one that cannot finish
            results.append({"attempt": name, "skipped": "smoke budget", "remaining_sec": round(remaining())}); break
        args = base_args() + extra
        off = LOG.stat().st_size if LOG.exists() else 0
        with open(LOG, "ab", buffering=0) as lf:
            lf.write(f"\n\n===== ATTEMPT {name}: {shlex.join(map(str, args))}\n".encode())
            SERVER = subprocess.Popen(list(map(str, args)), env=SERVE_ENV, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        t = time.time(); ready = False
        attempt_limit = clamp(SERVER_ATTEMPT_TIMEOUT)
        while time.time() - t < attempt_limit:
            if SERVER.poll() is not None: break
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200: ready = True; break
            except Exception: pass
            time.sleep(5)
        tail = LOG.read_bytes()[off:][-6000:].decode(errors="replace")
        results.append({"attempt": name, "ready": ready, "sec": round(time.time() - t, 1),
                        "exit_code": SERVER.poll(), "log_tail": tail})
        print(f"[launch] {name}: ready={ready} after {time.time() - t:.0f}s exit={SERVER.poll()}", flush=True)
        if ready:
            LAUNCHED = name; LAUNCHED_EXTRA = extra; break
        kill_server()
    ok = any(r.get("ready") for r in results)
    stage("4_launch", ok, attempts=results, gpu_mem=sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))
    # key lines for diagnosis (MoE runner / Marlin, weight loading, memory, errors), success or not
    try:
        KEY = re.compile(r"(?i)marlin|moe.*(runner|backend)|intermediate|load weight|weight load|memory|KV Cache|"
                         r"max_total_num_tokens|mamba|ple|pinned|error|exception|assert|ready to roll|fired up")
        key_lines = [l for l in LOG.read_text(errors="replace").splitlines() if KEY.search(l)]
        print("----- serve.log key lines (last 80) -----"); print("\n".join(key_lines[-80:]))
    except Exception as e:
        print("key-line scan failed:", e)
    if not ok:   # the stage line is truncated; show every attempt's log tail in full
        for r in results:
            print(); print(f"----- {r.get('attempt')} (exit {r.get('exit_code')}, {r.get('sec')} s) -----")
            print(r.get("log_tail", r.get("skipped", "")))
'''

C_ARC = r'''# ==================================================================== CELL 7 — stages 5-7: sanity, throughput (eager), throughput (CUDA graphs)
import urllib.request, concurrent.futures as cf
BASELINE_TOK_S = 373.0          # Qwen3.8-27B-FP8 (vLLM 0.19, fp8 KV, MTP3), 16 running, Q27-XHIGH v1.4 run 4 (2026-10-06)
DECODE_TOKENS = int(os.environ.get("FN_PROBE_DECODE", 1536))   # forced decode length per request (ignore_eos)
LEVELS = [1, 4, 8]

def _grid(g): return "\n".join(" ".join(str(v) for v in r) for r in g)
def arc_prompt(task):
    parts = ["You are solving an abstract reasoning puzzle. Infer the rule that maps every INPUT grid to its OUTPUT grid, "
             "then give the output grid for the FINAL INPUT.\n"]
    for i, ex in enumerate(task["train"]):
        parts.append(f"Example {i + 1} INPUT:\n{_grid(ex['input'])}\nExample {i + 1} OUTPUT:\n{_grid(ex['output'])}\n")
    parts.append(f"FINAL INPUT:\n{_grid(task['test'][0]['input'])}\n")
    return "\n".join(parts)

def chat(messages, max_tokens, thinking, temperature=0.0, ignore_eos=False, timeout=1800):
    body = {"model": "flashnext", "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking}}
    if ignore_eos:
        body["ignore_eos"] = True
    t = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=clamp(timeout)).read())
    return resp, time.time() - t

def sanity():
    resp, sec = chat([{"role": "user", "content": "Reply with the numbers 1 to 20 separated by single spaces, nothing else."}],
                     96, False)
    text = resp["choices"][0]["message"].get("content") or ""
    return re.findall(r"\d+", text)[:20] == [str(i) for i in range(1, 21)], text[:300], sec

def measure(level):
    msgs = [[{"role": "user", "content": arc_prompt(EVAL[t])}] for t in TASKS[:level]]
    t = time.time()
    with cf.ThreadPoolExecutor(max_workers=level) as ex:
        res = list(ex.map(lambda m: chat(m, DECODE_TOKENS, True, temperature=0.6, ignore_eos=True), msgs))
    wall = time.time() - t
    toks = [(r[0].get("usage") or {}).get("completion_tokens") or 0 for r in res]
    return {"concurrency": level, "wall_sec": round(wall, 1), "completion_tokens": toks,
            "prompt_tokens": [(r[0].get("usage") or {}).get("prompt_tokens") for r in res],
            "aggregate_tok_s": round(sum(toks) / max(wall, 1e-6), 1),
            "per_request_tok_s": [round(x / max(r[1], 1e-6), 1) for x, r in zip(toks, res)]}

def run_levels(tag):
    rows = []
    for lv in LEVELS:
        if remaining() < 6 * 60 + SAFETY_SEC:
            rows.append({"concurrency": lv, "skipped": "budget"}); break
        try:
            rows.append(measure(lv))
        except Exception as e:
            rows.append({"concurrency": lv, "error": f"{type(e).__name__}: {e}"[:600]}); break
        print(f"[{tag}] {rows[-1]}", flush=True)
    return rows

def kv_lines():
    try:
        return [l[-300:] for l in LOG.read_text(errors="replace").splitlines()
                if re.search(r"max_total_num_tokens|KV Cache is allocated|Mamba Cache is allocated|Capture cuda graph|cuda graph", l)][-12:]
    except Exception as e:
        return [repr(e)]

def launch_once(args, name):
    """same procedure as stage 4 for one extra launch; returns (ready, info)"""
    global SERVER
    off = LOG.stat().st_size if LOG.exists() else 0
    with open(LOG, "ab", buffering=0) as lf:
        lf.write(f"\n\n===== ATTEMPT {name}: {shlex.join(map(str, args))}\n".encode())
        SERVER = subprocess.Popen(list(map(str, args)), env=SERVE_ENV, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    t = time.time(); ready = False
    while time.time() - t < clamp(SERVER_ATTEMPT_TIMEOUT):
        if SERVER.poll() is not None: break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200: ready = True; break
        except Exception: pass
        time.sleep(5)
    return ready, {"attempt": name, "ready": ready, "sec": round(time.time() - t, 1), "exit_code": SERVER.poll(),
                   "log_tail": LOG.read_bytes()[off:][-4000:].decode(errors="replace")}

if go() and not RERUN:
    EVAL = json.load(open(Path(COMP_DIR) / "arc-agi_evaluation_challenges.json"))
    TASKS = sorted(EVAL)[:8]
    try:
        _ok, _text, _sec = sanity()
        stage("5_sanity", _ok, text=_text, sec=round(_sec, 1))
    except Exception as e:
        stage("5_sanity", False, error=f"{type(e).__name__}: {e}"[:2000], log_tail=LOG.read_bytes()[-4000:].decode(errors="replace"))
if go() and not RERUN:
    _rows = run_levels("eager")
    _best = max([r.get("aggregate_tok_s", 0) for r in _rows] or [0])
    stage("6_throughput_eager", any("aggregate_tok_s" in r for r in _rows), rows=_rows, best_aggregate_tok_s=_best,
          decode_tokens=DECODE_TOKENS, kv=kv_lines())
# stage 7 (optional, never fatal): CUDA graphs on - only after eager worked and with time for a reload + probe
REPORT["graphs"] = {"ran": False}
if go() and not RERUN and remaining() > 25 * 60:
    kill_server()
    _args = [a for a in base_args() if a != "--disable-cuda-graph"] + list(LAUNCHED_EXTRA) + ["--cuda-graph-max-bs", "8"]
    _ready, _info = launch_once(_args, "graphs_on")
    REPORT["graphs"] = {"ran": True, "launch": {k: v for k, v in _info.items() if k != "log_tail"}}
    if _ready:
        try:
            _ok, _text, _ = sanity()
        except Exception as e:
            _ok, _text = False, repr(e)[:300]
        _rows = run_levels("graphs") if _ok else []
        REPORT["graphs"].update({"sanity_ok": _ok, "sanity_text": _text, "rows": _rows, "kv": kv_lines(),
                                 "best_aggregate_tok_s": max([r.get("aggregate_tok_s", 0) for r in _rows] or [0])})
    else:
        REPORT["graphs"]["log_tail"] = _info["log_tail"]
    (OUT / "fn_smoke_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    print("graphs:", json.dumps({k: v for k, v in REPORT["graphs"].items() if k != "log_tail"}, default=str)[:3000], flush=True)
'''

C_REPORT = r'''# ==================================================================== CELL 8 — report, cleanup, bundle (always runs)
try: kill_server()
except Exception: pass
REPORT["total_sec"] = round(time.time() - T0, 1)
(OUT / "fn_smoke_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
print("=" * 100)
_expected = ["0_environment", "1_inputs", "2_install", "3_preflight", "3_pinned_memory", "3_fa2_shim", "4_launch",
             "5_sanity", "6_throughput_eager"]
_all_ok = REPORT["stopped_at"] is None and all(REPORT["stages"].get(k, {}).get("ok") for k in _expected)
if not RERUN and REPORT["stopped_at"] is None and not _all_ok:
    REPORT["stopped_at"] = "incomplete_unknown"   # never claim success for stages that did not run
    (OUT / "fn_smoke_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
print("STOPPED AT:", REPORT["stopped_at"] or "none (all stages passed)")
print("VERDICT:", "LOADS AND GENERATES ON 4xL4" if _all_ok else ("RERUN FALLBACK" if RERUN else "NOT SHOWN TO WORK - see stopped_at"))
try:
    _e = REPORT["stages"].get("6_throughput_eager", {}).get("best_aggregate_tok_s")
    _g = REPORT.get("graphs", {}).get("best_aggregate_tok_s")
    _best = max([x for x in (_e, _g) if x] or [0])
    print(f"THROUGHPUT: eager best {_e} tok/s | CUDA graphs best {_g} tok/s | 27B baseline {BASELINE_TOK_S} tok/s ->",
          "FASTER than the 27B" if _best > BASELINE_TOK_S else "NOT faster than the 27B")
except Exception as _x:
    print("throughput summary failed:", _x)
for k, v in REPORT["stages"].items():
    print(f"  {k}: {'PASS' if v['ok'] else 'FAIL'} at {v['t_sec']}s")
if "5_arc_test" in REPORT["stages"] and REPORT["stages"]["5_arc_test"]["ok"]:
    s = REPORT["stages"]["5_arc_test"]
    print("ARC test:", s["solved"], "/ 4 solved |", s["train_exact_tasks"], "train-exact |", s["aggregate_tok_per_sec"], "tok/s aggregate")
    for r in s["rows"]:
        print("  ", {k: r.get(k) for k in ("task", "test_correct", "train_exact", "code", "completion_tokens", "tok_per_sec", "finish", "error")})
ZIP = WORK / "FN_PROBE_BUNDLE.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.stat().st_size < 200 * 1024**2: z.write(p, p.relative_to(WORK).as_posix())
    if (WORK / "submission.json").exists(): z.write(WORK / "submission.json", "submission.json")
print("bundle:", ZIP, round(ZIP.stat().st_size / 1e6, 2), "MB | total", REPORT["total_sec"], "s")
'''

cells = [{"cell_type": "markdown", "metadata": {}, "source": C_HEAD.splitlines(keepends=True)}]
for src in (C_CFG, C_ENV, C_INPUTS, C_INSTALL, C_CUDA, C_LAUNCH, C_ARC, C_REPORT):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
DST.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print("wrote", DST, hashlib.sha256(DST.read_bytes()).hexdigest())
