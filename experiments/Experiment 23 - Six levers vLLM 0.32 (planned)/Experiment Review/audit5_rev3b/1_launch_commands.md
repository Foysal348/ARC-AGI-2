# E23 rev 3b audit 5, part 1: server launch commands

Scope: every server command the notebook builds (`stock_flags` / `stock_server`, `merlin_server`, the EAGLE step), checked for:
- (a) flag existence;
- (b) value compatibility;
- (c) memory fit at max-model-len 131072;
- (d) env isolation.

Files audited:
- builder `Experiment Notebooks/Build/build_e23_notebook.py` (working-tree sha256 39d5d58b0f2b; last commit 13c219c). Line numbers below are builder lines.
- `src/e23_lab.py`

Sources used:
- **Stock vLLM v0.32.0:** the local source refs in `D:/ARC AGI 2/_scratch/vllm_whl/src032/` (flattened file names). The local wheel `vllm_whl/vllm-0.32.0-...whl` is truncated at 256 MiB and is not a valid zip. `vllm/config/profiler.py` came from GitHub (raw `v0.32.0`).
- **Merlin:** `_scratch/e23/gpt_audit_rev3/merlin_ref/launch.py` and `qwen38-27b.json`, plus the Merlin 3.0.0 checkout `_scratch/e23/p2/merlin-src` (tag ad42e37).
- **FlashInfer:** the 0.7.0.post1 wheel in `_scratch/fi_check/`.
- **EAGLE fork:** the P6a report `_scratch/e23/p6a3/p6a_report.json`.

Model facts come from the FP8 checkpoint config: vocab 248,320, hidden 5120, 4 KV heads, head_dim 256, 16 full-attention layers + 1 MTP layer.

---

## Findings (still wrong or risky in rev 3b)

### F1. HIGH: the teacher-forced request (`prompt_logprobs`) will OOM the FP8-KV and NVFP4-KV target-only servers

**Where:**
- builder 638-639: `s6(0)` → `tforced(port, key="fp8_mtp0")`
- builder 722-723: `s5(0)` → `tforced(port, ref="fp8_mtp0")`
- `e23_lab.py` 484-489: `prompt_logprobs: 1` on 40k and 90k prompts
- server config: `stock_flags` builder 300-305, with `--max-num-batched-tokens 4096` and `--gpu-memory-utilization 0.90`

**Failure scenario.**
1. vLLM V1 computes prompt logprobs for every scheduled prompt chunk: `logits = self.model.compute_logits(prompt_hidden_states)`, then `self.sampler.compute_logprobs(logits)`. Evidence: `src032/vllm_v1_worker_gpu_model_runner.py:5374-5388`.
2. One chunk of the 40k prompt is 4,096 tokens. The vocab-parallel logits are gathered to full vocab on every TP rank:
   - logits: 4096 × 248,320 × bf16 = 2.03 GB;
   - fp32 log-softmax: 4.07 GB;
   - total: about 6.1 GB per GPU at once.
3. With `--gpu-memory-utilization 0.90`, the free headroom at runtime is about 10 % of 22.49 GiB, so at most ~2.2 GiB. The startup profile run sizes the KV pool from a sampler pass over `max_num_seqs` tokens, not 4,096 prompt-logit rows. Evidence: the same file; the profile never calls `_get_prompt_logprobs_dict`.
4. Result: CUDA OOM in the EngineCore, so the server dies inside the body.
   - In `s6(0)` (the `mtp0_target_only` step), this happens right after the oracle. All of section 6's MTP-OFF steady windows (40k / 90k / 120k) are lost. Those windows are the baseline row of section 6.
   - In `s5(0)`, the NVFP4 correctness gate gets `INCONCLUSIVE`, and the 90k / 120k residency census never runs.
   - The outcome is the same in SMOKE_B and FULL.

**Fix.** Do not run `prompt_logprobs` on the measurement servers. Add one short TF-only launch per KV dtype with the same reduced chunk on both sides, so the comparison stays like-for-like:

```python
def s5_tf(kv, key=None, ref=None):          # FP8-KV ref, then NVFP4-KV
    srv = stock_server(f"tf_{kv}", stock_flags(FP8, 8200, spec=0, kv=kv, seqs=2, batched=1024, mem=0.80), 8200)
    return with_servers([srv], 1800, lambda ss: {"teacher_forced": tforced(ss[0].port, key=key, ref=ref)})
```

At 1024 × 248,320 × 6 B ≈ 1.5 GB, this fits in the ~4.5 GiB headroom at 0.80. Then remove the two `tforced` calls from the `s6(0)` and `s5(0)` bodies.

The minimum alternative is to move both `tforced` calls to the last line of their bodies, so an OOM costs only the TF rows. In that case, record `SERVER_DIED` there as `TF_OOM`.

### F2. HIGH: Merlin serve uses the system GCC 11, but Merlin's FlashInfer compiles all its kernels at first start and Merlin requires GCC ≥ 13 for that

**Where:** builder 241-248. `merlin_server` pops `CC` / `CXX` / `CUDAHOSTCXX`, and its comment reads "system GCC at serve time (GCC 13 was only the install rule)".

**Evidence.**
- `merlin-src/README.md:229`: "Build tools | Git and GCC/G++ 13 or newer: FlashInfer compiles kernels on first start."
- `release/install-env.sh:90-98`: GCC ≥ 13 and C++20 are required.
- `release/requirements-0.32.lock:6`: "Exclude flashinfer-jit-cache: the tile must use the patched JIT source." The install also aborts if a jit-cache is present (`install-env.sh:140-144`). So every FlashInfer attention and sampling kernel for sm_89, head_dim 256 and FP8 KV is JIT-compiled from Merlin's patched headers (`prefill.cuh`, `scheduler.cuh`) at the first TP1 launch.

**Why E18 is not counter-evidence.** The stock wheelhouse ships `flashinfer-jit-cache==0.7.0.post1+cu130` (`_scratch/vllm_whl/vllm_0_32_0_requirements.lock`). Stock vLLM therefore never JIT-compiled those kernels with GCC 11.

**Why the CPU kits do not cover it.** P2 verified only `import` and `install-env.sh --verify`. No JIT compile was ever run.

**Failure scenario.**
1. The first `merlin_tp1` launch fails in the FlashInfer nvcc/g++ build, giving `LAUNCH_FAILED`.
2. The TP1 fallback recompiles the same sources and fails the same way.
3. Every Merlin step is lost: #1 a-d, the Merlin half of #4b, and the A/A Merlin repeats. That is the entire purpose of SMOKE_A.

There is a second, related risk. First-start JIT, torch.compile and graph capture all fall inside the 2,400 s launch timeout. If the GPU VM has few cores, this window is tight.

**Fix (before the GPU run).** Run a CPU-kit JIT probe. nvcc needs no GPU.
1. In the Merlin venv, with `FLASHINFER_CUDA_ARCH_LIST=8.9`, build the sm_89 BatchPrefill / BatchDecode modules for head_dim 256, fp8_e4m3 KV and bf16 Q.
2. Build them once with the system g++ 11.
3. Build them once with `CC/CXX/CUDAHOSTCXX=$MERLIN_CC/$MERLIN_CXX` and `LD_LIBRARY_PATH=/tmp/merlin_gcc13/lib:$LD_LIBRARY_PATH`. The conda libstdc++ must be the one loaded at runtime; otherwise the `.so` built by GCC 13 fails with `GLIBCXX_3.4.3x not found`.
4. Keep whichever compiler works. Ship the resulting `FLASHINFER_WORKSPACE_BASE` cache in the payload, so the GPU run skips the JIT entirely.

**Fix (in the notebook).** On `LAUNCH_FAILED`, if the log tail matches `ninja: build stopped|nvcc.*error|error: .*\.cuh`, relaunch once with the GCC 13 env above before falling back to the profile defaults.

### F3. MEDIUM: the TP1 memory fallback only covers launch failure, not a runtime OOM in the 90k / 120k prefill

**Where:**
- builder 572-581: the fallback runs only when `status == "LAUNCH_FAILED"`;
- builder 594: `s1_rep` reuses `TP1_CFG`;
- builder 831: the A/A loop repeats `s1_rep(4)`.

**Evidence.** The 27b-tp1-sm89-24g profile evidence (`qwen38-27b.json`) says: "262K prefill ran out of memory at max_num_batched_tokens 1024 (GPU 0 at 23.50 of 23.52 GiB); the row now ships 512. Not re-run at 512." The L4 pin (8,055,029,760 B) keeps that thin margin: it is the 4090 pin minus the 1.5 GiB VRAM difference. Rev 3 then raises seqs from 2 to 4 and the capture sizes from [1,2,5,10] to [1,2,5,10,15,20] without a measurement. That adds graph-pool and spec/sampler buffers.

**Failure scenario.**
1. An OOM during the 90k ×2 or 116k prefill kills the server mid-body.
2. `with_servers` returns `SERVER_DIED`, so no fallback runs.
3. `s1_rep(2)` and `s1_rep(4)` then start 2 and 4 replicas with the same seqs-4 config and die the same way. In FULL, this costs about 65 min plus each A/A repeat.

**Fix.**
- In `s1_tp1`, take the fallback when `o.get("status") in ("LAUNCH_FAILED", "SERVER_DIED")` or when any `servers[*].log_tail` matches `out of memory|CUDA error`.
- In `s1_rep`, use the fallback config whenever the single-replica run did not end with `alive_after_body=True`.

### F4. MEDIUM: the EAGLE arm turns prefix caching off, which confounds its correctness oracle with the known E19 failure mode

**Where:** builder 781: `stock_flags(FP8, 8300, ..., prefix=False, spec_json=spec)`.

**Evidence.**
- With prefix caching off, a hybrid model is forced to `mamba_cache_mode='none'` (`src032/vllm_model_executor_models_config.py:702-708`).
- E19 measured that spec decode with prefix OFF produces 4-12 high-margin flips on this model, while every prefix-ON arm passed.
- The EAGLE oracle compares against `fp8_kvauto_mtp0` and `fp8_mtp0`, which are both prefix ON (`stock_flags` default).

**Failure scenario.** EAGLE shows `FAIL_HIGH_MARGIN` (or a lower acceptance rate). The report cannot tell an EAGLE / tree-verify defect from the prefix-off mamba rollback bug, so the feasibility verdict is uninterpretable.

**Fix.**
1. Launch EAGLE first with `prefix=True`.
2. Fall back to `prefix=False` only on `LAUNCH_FAILED`.
3. When the arm ran with prefix off, label its oracle result `CONFOUNDED_PREFIX_OFF_E19` instead of PASS or FAIL.

Related, LOW: P6a parsed the EAGLE command only at `--max-model-len 106496` (p6a_report flagcheck). The notebook launches at 131072, which was never parsed, and the drafter's max positions were never recorded (P6b keeps only architectures, hidden, layers and vocab). If the drafter caps lower, vLLM caps the draft length (`src032/vllm__config__speculative.py:1556-1560, 1633-1672`): no crash, but long prompts get no drafting. Record `draft max_model_len` from the server log in `facts()`.

### F5. LOW: `CUDA_DEVICE_ORDER` is not set in the stock env while P4 runs on "GPU 3" next to Merlin TP1 on GPU 0

**Where:**
- builder 158-161: the STOCK env;
- builder 534: `--gpu 3`;
- builder 803-804: P4 runs concurrently with `s1_tp1`.

**Evidence.**
- Merlin pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` (`launch.py:523-524`).
- `bench_fp8_gemm.py:542-543` sets only `CUDA_VISIBLE_DEVICES=3`, under CUDA's default FASTEST_FIRST order. With identical devices, CUDA leaves the order unspecified.
- The GPU-release gate and `nvidia-smi` use PCI order.

**Failure scenario (unlikely on identical L4s, but costly).** P4 lands on Merlin's card, which already has the thinnest margin in the plan. Merlin TP1 OOMs, and P4's numbers come from a shared GPU.

**Fix.** Add `CUDA_DEVICE_ORDER="PCI_BUS_ID"` to the `env.update(...)` at builder 159.

### F6. LOW: the Merlin TP4 capture list goes to 120 (3× the profile's 40), and its fallback is not comparable

**Where:**
- builder 505: `CAP_TP4`;
- builder 618-622: the fallback with `seqs=24` and the profile captures.

**Evidence.** The 27b-tp4-sm89-24g profile was validated with captures up to 40 (`qwen38-27b.json`, release bench card). In the fallback with K=4, any decode batch above 40 tokens (9 or more running streams) has no captured graph and runs eager.

**Failure scenario.** A capture-OOM triggers the fallback. The fallback's 16- and 24-stream windows are then eager, but they are reported as "Merlin TP4" against stock vLLM's graphed rows.

**Fix.**
- Use `seqs=8` in the fallback (8 × 5 = 40, all graphed), or
- tag the fallback rows `config="profile captures <=40: >8 streams eager"`.

### F7. LOW: PP4 runs on the V1 model runner, so async scheduling is forced off

**Where:** builder 750-754: the STOCK env has no `VLLM_USE_V2_MODEL_RUNNER`.

**Evidence.** `src032/vllm_config_vllm.py:1634-1642`: "Async scheduling is disabled because the V1 model runner does not support it with pipeline parallelism". By contrast, `max_concurrent_batches` is `pp+1` only on V2 (lines 578-587).

**Effect.** The PP4 rows measure synchronous PP with pipeline bubbles, so they understate PP. The launch does not fail.

**Fix.** Note this in the PP4 row. Optionally, add a PP4 compiled arm with `env_extra={"VLLM_USE_V2_MODEL_RUNNER": "1"}` if time remains.

### F8. LOW: the profiled and unprofiled arms that are compared differ in launch config and env

**Where:**
- builder 631: `mtp3_eq_G0_piecewise` launches with `--profiler-config` plus `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` (builder 322-323);
- builder 813 and 830: `mtp1` and the A/A `mtp3_control_rep*` launch without them.

**Effect.** The per-step `record_function` scopes are cheap when inactive, but the A/A repeat is then not the same command as the arm it repeats.

**Fix.** Pass the same `profile_dir` (or a dummy one) to the reps, or drop the profiler from the steady-window server and profile in a separate short launch.

### F9. LOW: the SMOKE_B PP4 oracle has no reference, and the INT4 input is optional

**Where:** builder 758 uses `refs=("int4_mtp3",)`, but `s4_int4` runs only in FULL (builder 822).

**Effect.**
- In SMOKE_B, `vs_int4_mtp3` is always `INCONCLUSIVE_NO_ORACLE`, so PP4 smoke is gated only by the garbage check and the probes.
- If the INT4 model is not attached (builder 99-100: `must=False`), PP4 launches `None` as its model path and fails, with no explicit SKIPPED status.

**Fix.**
- In SMOKE, run one INT4 TP4 `spec=0` short oracle before PP4, or compare PP4 against `fp8_mtp0` as a cross-weight sanity check (garbage only).
- Return `SKIPPED_REQUIRED_ARTIFACT` when `INT4 is None`.

---

## Verified correct

### (a) Flags exist and are spelled right

**Stock vLLM v0.32.0** (`src032/vllm__engine__arg_utils.py`):
- `--language-model-only`: 1460
- `--profiler-config`: 1870
- `--attention-backend`: dataclass field 791
- `--kv-cache-memory-bytes`: 1363
- `--enable-prefix-caching` / `--no-enable-prefix-caching` (BooleanOptional): 1372
- `--mamba-cache-mode`: 1405
- `--disable-custom-all-reduce`: 1336
- `--tensor-parallel-size`, `--pipeline-parallel-size`, `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, `--seed`, `--served-model-name`, `--trust-remote-code`, `--speculative-config`, `--compilation-config`, `--enforce-eager`: all present
- `--host` / `--port`: from the OpenAI cli_args, same as the E18 command that ran

**Backend names** (`src032/vllm_v1_attention_backends_registry.py`): `TRITON_ATTN` (48), `FLASHINFER` (71) and `CUSTOM` (158, plugin-registered) are valid. `PIECEWISE`, `FULL_AND_PIECEWISE` and `FULL_DECODE_ONLY` are CUDAGraphMode names.

**KV dtype:** `nvfp4` is in `CacheDType` (`src032/vllm__config__cache.py:57`).

**ProfilerConfig at v0.32.0 (GitHub):** `profiler`, `torch_profiler_dir`, `torch_profiler_with_stack`, `ignore_frontend`, `delay_iterations` and `max_iterations` all exist. `VLLM_CUSTOM_SCOPES_FOR_PROFILING` exists (`src032/vllm_envs.py:287`). The scope name `gpu_model_runner: forward` (`gpu_model_runner.py:4201`) matches what `trace_summary` looks for.

**Merlin:**
- Repeated regular flags (`--max-model-len`, `--port`, `--served-model-name`, `--max-num-seqs`, `--compilation-config`) only log a "duplicate" warning. argparse's last value wins (`merlin-src/vllm/utils/argparse_utils.py:404-458`), so the extra args override the profile.
- The override `--compilation-config` JSON reproduces all three keys of the profile's JSON (`launch.py:599-601`, `B27_INDUCTOR` at line 62), so nothing of the profile is dropped.

**EAGLE fork:** P6a v3 parsed 4 of 4 commands with the fork's own parser (`eagle_harness`, `--attention-backend CUSTOM`), and `glibc_over_2_35` is empty.

### (b) Values are compatible

- **Prefix caching is never off with MTP.**
  - All MTP arms (mtp1, mtp3 / G0-G3, nvfp4_mtp3, int4_mtp3, A/A) use the `stock_flags` default `prefix=True`.
  - The Merlin profiles always emit `--enable-prefix-caching --mamba-cache-mode align` (`launch.py:590`).
  - For the hybrid model, stock switches to `align` automatically (`src032/vllm_model_executor_models_config.py:685-693`).
- **PP4 runs without MTP** (`spec=0`, builder 754). Qwen3_5 implements `SupportsPP` (`src032/vllm_model_executor_models_qwen3_5.py:305`).
- **FlashInfer on sm_89:**
  - Graph support is `UNIFORM_SINGLE_TOKEN_DECODE` (no trtllm-gen on sm_89; `src032/vllm_v1_attention_backends_flashinfer.py:1036-1080`), so G0 pinned to `PIECEWISE` is the honest control.
  - The #60532 hunk makes sm<90 return `UNIFORM_BATCH`.
  - `patch -p1 --dry-run` of the flashinfer-only diff on the v0.32.0 file succeeds, 13/13 hunks. The hunks apply with offsets of -1 to -111 lines: the diff's index blob `bb9e1b0e` is not the v0.32.0 blob `f6c3e262`, so "clean" means "applies with offsets".
  - FlashInfer 0.7.0.post1 `BatchDecodeWithPagedKVCacheWrapper.plan` documents `q_len_per_req > 1` on fa2.
- **G1 / G2:** TRITON_ATTN is `AttentionCGSupport.ALWAYS` and GDN is `UNIFORM_BATCH` (Merlin-tree `triton_attn.py:92`, `gdn_attn.py:98`; same upstream code), so FULL_AND_PIECEWISE and FULL_DECODE_ONLY with MTP-3 are valid.
- **NVFP4 KV on sm_89:**
  - It takes the fa2 path, since family 80 covers 8.9 (`flashinfer.py:109-110, 530-533`; `src032/vllm_platforms_interface.py:588`).
  - head_dim 256 is allowed; only 64 and sinks are rejected (`flashinfer.py:548-560, 834`).
  - The MTP draft layer shares the same KV spec, so NVFP4 + MTP3 is a legal combination.
- **Merlin:**
  - The capture lists are multiples of 1+K (K=4): TP1 4 × 5 = 20, TP4 24 × 5 = 120.
  - TP1 is `spec_k_fixed`, with RecoverSSM, V2 runner and replayssm set by the launcher (`launch.py:552-557`).
  - `P2P=off` sets `NCCL_P2P_DISABLE=1` and `VLLM_SKIP_P2P_CHECK=1` on TP4 (`launch.py:534-538`). TP1 has no P2P switch.

### (c) Memory at max-model-len 131072

- **Merlin TP4:** the KV pin is 16,301,686,784 B per GPU, about 1.83 M tokens. That holds 24 × 40k, 16 × 90k and 12 × 120k.
- **Merlin TP1:** the KV pin is about 225k tokens. It holds 4 × 40k, 2 × 90k and 1 × 120k; 2 × 120k shows up as CAPACITY_LIMITED, which is measured, not a failure. The runtime OOM risk is F3.
- **Stock FP8 TP4 at 0.90:**
  - FP8 KV costs 17 layers × 1 head × 256 × 2 B = 8.5 KiB per token per GPU.
  - That gives about 1.4 M tokens (estimate): 20 × 90k and 20 × 120k come out CAPACITY_LIMITED and are recorded.
  - NVFP4 needs about 0.56× the bytes per token, so it holds about 2.5 M tokens.
- **BF16-KV / EAGLE:** 16 KiB per token per GPU, which still holds one full 131072-token session.
- **The 120k band:** 116,000 prompt + 14,560 new + 512 margin = 131,072 exactly, so no request exceeds max-model-len.

### (d) Env isolation

Every server gets a fresh dict, and `os.environ` is never mutated.
- **STOCK:** removes `PYTHONPATH`, `CUDA_PATH` and `LD_PRELOAD`.
- **Merlin:** starts from `os.environ`, not from STOCK, so stock's `CUDA_HOME` does not leak in. It sets its own `CUDA_HOME` (venv cu13), the glibc shim and per-port Triton caches. `MASTER_PORT` / `VLLM_PORT` come from Merlin's `distributed_ports` and are disjoint per replica: 8100→29600/40000, 8101→29601/40100.
- **EAGLE:** a copy of STOCK with its own `PATH`, `VLLM_CACHE_ROOT` and V2 runner.
- **`VLLM_CUSTOM_SCOPES_FOR_PROFILING`** is set only on the server copy for profiled arms.
- **The #60532 revert** uses copy2 plus a sha check. The restored file has a different size, so any stale `.pyc` is invalidated, and the patch cannot leak into the NVFP4 / EAGLE / PP4 arms.
