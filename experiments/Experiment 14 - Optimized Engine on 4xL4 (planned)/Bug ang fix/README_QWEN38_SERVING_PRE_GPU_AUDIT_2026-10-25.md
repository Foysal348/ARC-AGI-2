# README — Qwen3.8-27B / ARC-AGI-2: Pre-GPU Serving Audit

**Audit snapshot:** 2026-10-25 (Asia/Dhaka)  
**Audience:** Claude / engineer reviewing Kaggle serving experiments  
**Format:** Bengali analysis; flags, error messages, versions and source URLs retained verbatim.  
**URL policy:** Sources are written as COMPLETE, RAW `https://...` strings; no opaque ChatGPT citations, shortlinks or `[title](url)` links.

> IMPORTANT: This file consolidates the GPT serving audit discussion and spot-checks key upstream issues. It is NOT an independent Kaggle GPU validation. Source reports about other GPUs/checkpoints are risk evidence, NOT proof that this exact L4 + RedHat checkpoint reproduces the failure. Never infer a fix is included in a tag solely because a PR is merged; inspect the release/tag or runtime source.

## 0. Executive verdict and priority

1. **Target-only first** as a reference accuracy control. `GO` for limited preflight, NOT guaranteed model quality.
2. **NO-GO until corrected:** 20/24-stream SGLang with unchanged Mamba state pool of 68; `flashinfer`/`cutedsl` GDN as a purported drop-in SM89 replacement; vLLM compressed-tensors INT4 + BF16 MTP head until loader contract inspected; obsolete `VLLM_TEST_FORCE_FP8_MARLIN=1` experiment.
3. **HIGH accuracy gating:** DFlash2 thinking-mode output divergence; DFlash2 concurrency state corruption; W4A16 polluted CUDA compute cache; MTP state/prefix cache; SSM dtype drift; FP8/INT4 batch-dependent numerics.
4. **Worth measuring after correctness:** SGLang DFlash W8/W6/W4, MTP 3/1/4 vs 2/1/3, 2 × TP2, and vLLM FP8 kernel selection. No speedup assumption is verified on this exact hardware.
5. `SGLang #41872` and `#44401` were backported locally, per the user. The user's observed 11/12 admission problem persists; these PRs are NOT evidence it has been solved in the custom build. SGLang `v0.5.21` tag predates those PRs.

Status key: `GO-TEST` = fit for a gated experiment; `CONDITIONAL` = do not deploy before relevant gate; `NO-GO` = specified blocking condition remains; `UNKNOWN` = no exact-stack evidence found.
Evidence key: `UPSTREAM` = actual code/issue/PR; `USER-OBSERVED` = reported by user; `CALCULATED` = arithmetic using cited model-specific source; `UNVERIFIED` = not reproduced on current Kaggle L4.

## 1. EXACT environment to preserve

- Kaggle offline notebook at run time; 4 × NVIDIA L4 24GB, Ada `sm_89`, PCIe PHB, no NVLink; NCCL currently SHM; four-GPU custom all-reduce disabled. Driver CUDA 13.0, Python 3.12, glibc 2.35.
- Qwen3.8-27B, hybrid Gated DeltaNet + full attention, native MTP head; generations 60k–90k; context as high as 155648 tokens; 12–24 streams; temperature 1.0, `top_p=0.95`, `top_k=20`.
- Checkpoints: `RedHatAI/Qwen3.8-27B-INT4` (W4A16 group128 compressed-tensors Marlin, AWQ smoothing, BF16 MTP), Qwen-layout repacked FP8 block128 with MTP, and `z-lab/Qwen3.8-27B-DFlash2` BF16 drafter.
- SGLang 0.5.21; PyTorch 2.13.0 cu130; FlashInfer 0.6.18 (+cu130 JIT), sglang-kernel 0.4.7; transformers 5.12.1.
- vLLM 0.32.0; PyTorch 2.13.0 cu130; FlashInfer 0.7.0.post1 (+cu130 JIT), transformers 5.19.0; compressed-tensors 0.17.0; xgrammar 0.2.8; Triton 3.7.1.
- Production reference SGLang flags:

```text
--tp-size 4
--kv-cache-dtype fp8_e4m3
--context-length 155648
--chunked-prefill-size 4096
--mamba-ssm-dtype float16
--mamba-radix-cache-strategy extra_buffer_lazy
--max-mamba-cache-size 68
--mem-fraction-static 0.83
--disable-prefill-cuda-graph
--max-running-requests 17
--cuda-graph-bs-decode [batch graph sizes 1 through 17 in current launch configuration]
--min-free-slots-delay 1
--speculative-algorithm DFLASH
--speculative-draft-model-path <DFlash2>
--speculative-draft-model-quantization unquant
--speculative-num-draft-tokens 6
--incremental-streaming-output
--stream-interval 1
--schedule-policy lpm
--enable-cache-report
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
--disable-flashinfer-autotune
```

`[batch graph sizes ...]` above is a description, NOT a literal CLI value. Derive valid CLI format from installed `--help` and local launcher source before running.

## 2. Silent-accuracy risks — test BEFORE performance

| ID | Failure mechanism | Pinned-version relevance | How to detect | Mandatory mitigation | Source |
|---|---|---|---|---|---|
| ACC-1 | Compressed-tensors W4A16 wrong PPL from persistent CUDA driver compute cache; report SGLang 0.5.19–0.5.21 | **SGLang 0.5.21 reported**, other checkpoint; not proven RedHat | Same fixed input-ID set, cross-engine PPL/logprob; clean vs persisted cache, server restart | Isolate effective `SGLANG_CACHE_DIR` and driver cache, see §6.1 | E01 |
| ACC-2 | DFlash2 greedy result differs from target-only when thinking is enabled, first divergence reported at token 30 | Reported Qwen3.8 family; exact L4 stack unproven | Greedy token IDs target-only vs DFlash; thinking on/off; repeat across restarts | Fail-closed for mismatches; consider target-only or verified MTP | E02 |
| ACC-3 | DFlash2 concurrent request state corruption | Public issue, exact pinned hardware unproven | Same long prefix and continuation at 1, 8, 12, 17, 20 running + churn | Quarantine DFlash mode until verified; inspect state/scheduler commit paths | E03 |
| ACC-4 | FlashInfer autotune can change DFlash greedy outputs between restarts | Public issue, exact stack unproven | Cold-start repeats and kernel choice logs | Retain `--disable-flashinfer-autotune`; not a complete DFlash correctness fix | E04 |
| ACC-5 | Hybrid recurrent-state speculative vs non-speculative slot desync, stale MTP scratch/prefix checkpoint | Several vLLM fixes merged before 0.32.0; verify exact wheel code | Draft unavailable/zero accepted/full accepted, prefix cold/warm and growing prompts | Use v0.32.0 fixed path; keep prefix-cache correctness tests | E05 E06 E07 |
| ACC-6 | Qwen3.8 score changes with batch size (reported FP8 and BF16) | Reported upstream; not proof all batch-shape changes are bugs | Fixed prompt at batch 1/8/12/16/24; logprobs and chosen token | Near-tie warning; fp32 lm-head control; do not use exact greedy match as universal guarantee of stochastic equivalence | E08 |
| ACC-7 | Custom thinking-budget logits processor not applied under speculative decode | SGLang reports, exact combination needs check | Explicit token-budget boundary assertions; validate processor invocation on every speculative step | Do not rely on processor for strict stop policy until tested; harness enforcement fallback | E09 E10 |
| ACC-8 | MTP head dense BF16 but vLLM compressed-tensors loader expects quantized packed names (`fc.weight_packed`) | vLLM 0.30.0 source report, 0.32.0 resolution not established | Inspect safetensors names + `quantization_config.ignore`; attempt boot | Build MTP predictor unquantized when checkpoint explicitly stores dense BF16 MTP; otherwise turn MTP off | E11 |

**Validation nuance:** `temperature=0` tests greedy path, while production uses `temperature=1.0, top_p=0.95, top_k=20`. Equal greedy IDs do NOT certify equal sampling distribution. On stochastic settings compare teacher-forced logprobs, top-k probabilities, sampling distribution for selected fixed prefix states, and downstream ARC task solve rate. Pure byte-for-byte equality under different kernels/batch shapes can be overly strict; always inspect first divergent step and logit margin. For explicitly deterministic same-stack restarts, substantial unexpected variation is itself a failure signal.

## 3. SGLang 0.5.21: exact configuration-by-configuration disposition

| Arm | Issue (type) | Affects our version? / fix status | Severity / verdict | Exact flag, patch or workaround | Confidence / evidence |
|---|---|---|---|---|---|
| **S1 target-only** | INT4 GEMM/cache silent wrong output, GDN precision and long context memory | SGLang W4A16 dirty driver cache was reported in 0.5.21; RedHat-specific repro not found | HIGH QA; **GO-TEST** | Remove speculative flags completely; maintain Triton GDN, isolated cache; PPL/logprob/long-context control | High issue / medium checkpoint E01 E08 E12 |
| **S2 INT4 MTP steps=3 topk=1 draft=4** | BF16 MTP on quantized target, separate drafter path ambiguity, draft redundant embeddings, verify state; memory/startup | MTP present in v0.5.21 cookbook, checkpoint-specific load NOT verified; redundant-copy memory bug v0.5.21 report | HIGH **CONDITIONAL** | Follow pinned Qwen3.8 3/1/4 recipe; attempt same-checkpoint MTP without `--speculative-draft-model-path` if recipe supports; inspect actual worker loaded model; FP32 state control; patch/rebase EAGLE memory init only after code inspection | High recipe / medium integration E12 E13 |
| **S3 MTP steps=2 draft=3; mem .86; ReplaySSM fp16/fp32** | ReplaySSM ring vs verification state drift; unbounded higher KV pool causing OOM | ReplaySSM feature in pinned branch; exact Qwen3.8+INT4+L4 equivalence not shown | HIGH **CONDITIONAL** | Compare `--enable-linear-replayssm-spec` off/on at `--mamba-ssm-dtype float32` then `float16`; boot warning and logit consistency; `--mem-fraction-static 0.83` control then .86 | Medium E12 E14 |
| **S4 Competitor FP8 flags and identical INT4 flags** | `extra_buffer` 5 state slots/request, float32 SSM pool explosion, FP8/INT4 kernel non-equivalence, 2048 prefill latency tradeoff | 0.5.21 calculator confirms slot counts; LB 65.97 causality/universal benefit not independently proven | HIGH memory **CONDITIONAL** | FP8 and INT4 as separate arms; set mamba slots = mrr×5 and prove fit; compare `--chunked-prefill-size 2048` vs 4096, `--linear-attn-backend triton`; do not infer score from competitor flags | High capacity, low LB attribution E12 E15 |
| **S5 DFlash2, mrr21 and mrr25** | 68 Mamba slots cannot sustain 21/25 requests with `extra_buffer_lazy`; graph batch ceiling may induce tensor size crash | **Yes capacity mismatch**; reported 12/13 tensor crash is USER-OBSERVED, upstream exact fix not established | **NO-GO until corrected** | `--max-mamba-cache-size 84` (mrr21) or `100` (mrr25), CUDA graph max batch >= mrr via installed CLI; prove memory, admit all streams, churn test; 25×fp32 SSM essentially memory infeasible on L4 | High arithmetic E12 E16 E03 |
| **S6 Best MTP @20 streams** | Scratch/ring pool/graph limits, startup redundant MTP weights; plateau at 10 possible | State capacity explicitly constrained in 0.5.21; throughput unknown | HIGH **CONDITIONAL** | For mrr21 (20 desired streams plus headroom), `--max-mamba-cache-size 84` for lazy; max graph >=21; measure actual running not queued; issue E13 startup cleanup | High capacity / low throughput E12 E13 |
| **S7 DFlash W4, W8 (plus W6)** | Checkpoint trains with `block_size=8`; override does not guarantee same acceptance/logits; speedup hypothesis unproven | `dflash_worker_v2.py` v0.5.21 has size handling; exact L4 W4 gain not found | HIGH QA **CONDITIONAL** | Baseline W8; controlled W6/W4; record accepted tokens/step, ms/step, output tok/s, kernel/memory + same-prompt logprob/greedy; reject if correctness fails | High code / low gain E17 E18 |
| **S8 GDN backend flashinfer or cutedsl on sm_89** | Official GDN backend matrix has hardware/phase limitations; FlashInfer GDN prefill fast path validated for newer SM than Ada; CuTe DSL does not cover all phases | Published matrix does NOT demonstrate complete L4 GDN route; no successful identical stack reference found | **NO-GO as drop-in replacement** | Keep `--linear-attn-backend triton`; only validate new backend if every prefill/decode/verify kernel supported or an explicitly logged safe fallback; never force unsupported path | Medium-high E15 E12 |
| **S9 2 independent TP2 servers (0,1 and 2,3)** | Twice target/drafter weight allocation per GPU shard relative TP4, two independent state pools; ports/process/NCCL/shared memory isolation, P2P topology pairing | TP4 custom allreduce unavailable on >2 PCIe GPU observed; TP2 fit/speed unknown | HIGH **CONDITIONAL**, high-priority performance experiment | Distinct `CUDA_VISIBLE_DEVICES`, `--tp-size 2`, `--port`, `--nccl-port`, log directories and CUDA caches; size lazy Mamba slots >=32 for 8 streams per replica; measure actual NCCL and peak mem | High topology / medium feasibility E19 E20 |

**S4 exact competitor flags supplied by user** (run as separate FP8 and INT4 arms; don't blindly assume benefit):

```text
--kv-cache-dtype fp8_e4m3
--attention-backend flashinfer
--linear-attn-backend triton
--chunked-prefill-size 2048
--max-prefill-tokens 4096
--mamba-radix-cache-strategy extra_buffer
--mamba-full-memory-ratio 0.5
--mamba-ssm-dtype float32
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

### SGLang explicit Mamba state capacity calculator

Pinned Qwen3.8-27B cookbook states `--max-mamba-cache-size = max_running_requests × S` (`S=4` for `extra_buffer_lazy`, `S=5` for `extra_buffer`). `D` speculative verify intermediates are separately budgeted, DO NOT add `D` again. `--enable-linear-replayssm-spec` changes verify intermediate accounting but does not erase normal concurrent-state requirements.

| `--max-running-requests` | `extra_buffer_lazy` min slots | `extra_buffer` min slots |
|---|---:|---:|
| 8 (per TP2 server) | 32 | 40 |
| 17 | 68 | 85 |
| 21 | 84 | 105 |
| 25 | 100 | 125 |

From official model-specific cookbook, one state slot is roughly 153.9 MB at fp32 and 78.4 MB at bf16. **100×153.9MB ≈15.39GB** of fp32 SSM state alone, not counting target/drafter weights (~6.11 GB/GPU historical TP4 observation), KV, graph workspace, activation and verifier. Thus 25 streams + fp32 SSM on 24GB L4 is a severe preflight NO-GO unless a materially different memory mode/config is proven. This is arithmetic, not measured allocation.

Source raw URL:

```text
https://github.com/sgl-project/sglang/blob/v0.5.21/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
```

## 4. vLLM 0.32.0: configuration-by-configuration disposition

| Arm | Issue | Affects 0.32.0? / fix status | Severity / verdict | Exact mitigation / test | Confidence / evidence |
|---|---|---|---|---|---|
| **V1 FP8 block128 MTP3, 16 seq, FP8 KV, prefix cache, 106k–155k** | Mamba/prefix-checkpoint stale state; prefix hits can save inaccurate state; fp8 KV does not halve SSM | Major related fixes merged before 0.32.0 (#56531 #59759 #60797), but local packaged code/load still verify | HIGH **CONDITIONAL** | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`; `--kv-cache-dtype fp8`; forced cold/warm prefix reuse, partial rejection and varying long contexts | High fix history / medium current integration E05 E06 E07 E21 |
| **V2 MTP speculative tokens 2/3/4** | Wrong Mamba state slot at mixed spec/non-spec steps, verify scratch/graph memory, throughput can regress | Specific #56531 merged 2026-10-05, should be in tag; no unconditional proof at K4/L4 | HIGH **CONDITIONAL** | Separate K2/K3/K4 tests; accepted length and rejected-draft conditions; logit control and memory fit | High mechanism / medium application E05 |
| **V3 INT4 compressed-tensors + MTP** | Dense BF16 MTP shard mismatches quantized Linear expected packed `fc` keys -> deterministic load crash | #58807 reports v0.30.0. Fix in 0.32.0 NOT independently confirmed; RedHat layout requires inspect | **NO-GO until static check passes** | Inspect `mtp.*` keys and quant `ignore`; patch full MTP layer construction to `quant_config=None` only if MTP is actually dense, or run INT4 target-only | High bug / medium exact checkpoint E11 |
| **V4 `--max-num-seqs` 16/20/24** | GPU memory/preemption and true active concurrency; massive output histories and prefix state | Config memory tradeoff applies; exact L4 fit not independently verified | HIGH **CONDITIONAL** | 16→20→24 increasing load; log physical HBM, KV capacity, preemptions, admission, no OOM | High generic / unknown thresholds E22 |
| **V5 `--data-parallel-size 2 --tensor-parallel-size 2`** | `--max-num-seqs` is per DP rank, not global; each DP rank has independent KV cache; router balancing may change cache reuse | **Yes, v0.32.0 official docs explicitly confirm** | HIGH capacity **CONDITIONAL** | For 16 streams GLOBAL, set `--max-num-seqs 8` PER rank (then verify scheduling); compare total 2×TP2 vs TP4 and note HTTP routing skew | High E23 |
| **V6 SM89 block-FP8 default/tuned/Marlin** | `VLLM_TEST_FORCE_FP8_MARLIN` removed; requested backend may fall back if unsupported on L4; default kernel cannot be assumed | **Yes, removal PR #52182 merged Aug 18, before v0.32.0** | **NO-GO for old env arm** | Stop setting env; use `--linear-backend auto|triton|marlin` (cutlass only if available), verify CLI and ACTUAL selected kernel in logs; gate logprob and PPL | High E24 E25 |

**Important vLLM MTP static inspection**:

```python
import json
from pathlib import Path

model = Path('/path/to/RedHatAI-Qwen3.8-27B-INT4')
config = json.loads((model/'config.json').read_text())
quant = config.get('quantization_config', {})
print('format', quant.get('format'))
print('ignore', quant.get('ignore'))
index = model/'model.safetensors.index.json'
if index.exists():
    wm = json.loads(index.read_text())['weight_map']
    mtp = [key for key in wm if key.startswith('mtp.')]
    packed = [key for key in mtp if any(tag in key for tag in
              ('weight_packed', 'weight_scale', 'weight_zero_point'))]
    dense = [key for key in mtp if key.endswith('.weight')]
    print('mtp', len(mtp), 'packed-related', len(packed), 'dense', len(dense))
    print('sample', mtp[:20])
else:
    print('No shard index. Read the safetensors header(s), not the full weights.')
```

If MTP names are dense `.weight`, packed names absent, and target's quantization config doesn't exempt MTP, flag loader mismatch and do NOT burn GPU launching the MTP arm. This check is a diagnostic, not proof of a bug if `ignore` semantics / module wrappers differ.

## 5. Exact incidents already observed by the team — evidence boundaries

| Incident | Classification | Next action |
|---|---|---|
| `min-free-slots` admission remains 11/12 under churn; with delayer off server hit 13 running and `The size of tensor a (12) must match the size of tensor b (13)` in `dflash_worker_v2.py` | USER-OBSERVED, specific runtime stack | Graph maximum must cover mrr, then 11/12/13 transition soak. PR #41872 and #44401 alone did not solve observed condition. Capture the specific failing call/shape and compare current backported source to upstream. |
| `--enable-int8-mamba-checkpoint` caused OOM | USER-OBSERVED; source suggests extra independent checkpoint HBM accounting | Leave disabled. If retesting, constrain `--int8-mamba-ckpt-size`, compare allocation & free HBM, run without speculation first. |
| DFlash greedy differing on server restarts | USER-OBSERVED plus upstream issue family | Retain `--disable-flashinfer-autotune`; clean cache; fixed prompt logit/restart comparison. Do not attribute all divergence to one source without trace. |
| Thinking-budget logits processor fails under speculation | USER-OBSERVED and upstream reports | Use external/harness hard stop; unit-test strict reasoning budget under speculation before trusting any internal processor. |
| MTP (fp32 state/auto mem fraction) cap at 10 running due to Mamba state cache | USER-OBSERVED + pinned capacity rule | Explicit mamba slots, confirm startup `max_running_requests`; compute hbm fit. |
| TP4 no optimized SGLang custom all-reduce on 4 PCIe-only GPUs | USER LOG + upstream source | NCCL SHM vs P2P profiling at 8KB–1MB; don't turn 64 MB throughput result into small-collective latency. |

### 5.1 SGLang redundant MTP weights: candidate patch, not blind drop-in

Source #42510 proposes calling the existing idempotent draft-sharing routine before target memory-pool profiling. Before backport, inspect the **exact installed** `python/sglang/srt/managers/scheduler.py` for whether `_release_draft_redundant_weights()` exists and how it references the inner EAGLE worker. The source issue's patch should be reviewed fully; a one-line early call alone may not be sufficient if the routine itself is absent in your source.

Intended ordering is approximately:

```python
# Scheduler.init_target_memory_pool() -- SCHEMATIC, not an apply-ready diff.
self._release_draft_redundant_weights()  # must resolve the correct draft inner worker
self.tp_worker.model_runner.account_preloaded_weights(...)
self.tp_worker.alloc_memory_pool()
```

Validation: no double frees, same correct weight pointers/sharing semantics, startup peak HBM before/after, KV allocation size, generated token IDs and early/late call idempotence. Stop if any path diverges.

Raw source:

```text
https://github.com/sgl-project/sglang/issues/42510
```

### 5.2 Driver cache workaround — IMPORTANT correction

SGLang `srt/environ.py` may set `CUDA_CACHE_PATH=$SGLANG_CACHE_DIR/nv`, overriding an external CUDA cache setting. Thus setting only `CUDA_CACHE_PATH` is NOT a guaranteed fix. Use a truly clean isolated `SGLANG_CACHE_DIR` and verify the effective worker environment:

```bash
export SGLANG_CACHE_DIR="/tmp/sglang_l4_audit_clean_cache"
mkdir -p "$SGLANG_CACHE_DIR"
# Also isolate CUDA cache if version does not override it:
export CUDA_CACHE_PATH="$SGLANG_CACHE_DIR/nv"
# Start SGLang after these exports, NOT before.
# In the actual server worker environment, confirm effective CUDA_CACHE_PATH.
```

Compare a warm run and a genuinely clean run on identical model bytes and logging. Do not delete a mounted shared cache while another server is using it. A clean cache run is diagnostic, NOT a guarantee all W4A16 kernels are correct.

Raw source:

```text
https://github.com/sgl-project/sglang/issues/42917
```

## 6. Hybrid GDN / long context / Ada / PCIe risk matrix

| Area | Evidence | What NOT to assume | Required probe |
|---|---|---|---|
| Gated DeltaNet speculation | DFlash Qwen3.8 divergence and vLLM #56531 recurrent-state correction | `acceptance rate high` implies answer correctness | Greedy, forced reject, partial accept, long repeated prefix |
| Mamba + prefix cache | vLLM #59759/#60797 fixed stale blocks/conv checkpoint inputs; SGLang cache slots separate | FP8 KV fixes recurrent SSM memory or checkpoint correctness | Cold/warm same-prefix replay across chunk boundaries |
| ReplaySSM | Pinned cookbook notes state-drift warnings with non-fp32 | `float16` equals `float32` at 90k reasoning | FP32/FP16 vs no ReplaySSM branch-sensitive prompts |
| Long context >64k | Model's advertised max context does not certify available KV memory or full 155k concurrency | 24 requests each at 155k fit 4×24GB | Progressive prompt length 32k/64k/96k/155k, track real tokens/HBM/preemptions |
| SM89 GDN backend | SGLang backend matrix/hardware gate; FlashInfer GDN prefill fast path favors newer SM | `--linear-attn-backend flashinfer` automatically fast on L4 | Backend startup capability, actual chosen kernels, equivalence and fallback |
| SM89 FP8 block128 | Native FP8 hardware present, but vLLM backend selection/fallback differs by layer | Default is necessarily Triton, or forced CUTLASS works | Report actual `Selected <kernel>` logs per layer |
| W4A16 Marlin | #42917 driver-cache PPL discrepancy; compressed-tensors layout affects drafter | Successful startup certifies quality | PPL/logprobs vs other trusted engine, cache clean |
| NCCL PCIe PHB | TP4 optimized custom AR unavailable; actual NCCL transport is topology-dependent | TP2 necessarily 2x faster; `NCCL_P2P_DISABLE=1` always optimal | `nvidia-smi topo`, NCCL debug, 8KB–1MB microbenchmark |
| 2 independent servers | Distinct process groups, caches, GPU visibility and communication ports required | Different `--port` alone isolates all NCCL traffic | bind GPUs, ports, shm, NCCL unique IDs, simultaneous boot, kill/restart one |

## 7. Two server topology: safe experiment template (fill paths locally)

First run:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p p
```

Select actual fastest pairs rather than assuming GPU `(0,1)` and `(2,3)` are optimal. For example, **ONLY IF topology supports the pairs**:

```bash
# Server A: use independent log/caches, port and NCCL port.
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_CACHE_DIR=/tmp/sglang_replica_A \
python -m sglang.launch_server \
  --model-path /path/to/target \
  --tp-size 2 \
  --port 30000 \
  --nccl-port 29500 \
  --max-running-requests 8 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size 32 \
  [OTHER TESTED MODEL/DFLASH OR MTP FLAGS]

# Server B: launch separately, with a different GPU pair, ports and cache.
CUDA_VISIBLE_DEVICES=2,3 \
SGLANG_CACHE_DIR=/tmp/sglang_replica_B \
python -m sglang.launch_server \
  --model-path /path/to/target \
  --tp-size 2 \
  --port 30001 \
  --nccl-port 29501 \
  --max-running-requests 8 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size 32 \
  [OTHER TESTED MODEL/DFLASH OR MTP FLAGS]
```

Square-bracket `[...]` placeholders are NOT literal CLI arguments. Before using the template, carry over target quantization, KV dtype, context length, graph sizes, speculative mode, drafter/model flags, tool/reasoning parser and all other validated flags. Ensure Kaggle permits two subprocess servers. Monitor shared `/dev/shm`, CPU and disk, not only GPU HBM. Test both servers at the same time and during abrupt termination of either replica.

For vLLM `DP=2 TP=2`: `--max-num-seqs` is per rank. To benchmark 16 requests TOTAL, start with `--max-num-seqs 8`, not 16; confirm capacity and routing in practice.

## 8. Required validation gates and scorecard

### Stage A — CPU/static, before booking GPU time

- `pip show sglang vllm torch flashinfer sglang-kernel transformers compressed-tensors triton` in exact prebuilt/offline wheelhouse; archive `pip freeze` plus wheel SHA256.
- Read local `--help` to confirm every flag, particularly speculative draft path, graph batch-size form, `--linear-backend`, ReplaySSM and extra checkpoint flags.
- Verify SHA256 / config / safetensors key index for target, FP8, INT4 and DFlash2. Inspect MTP BF16 vs packed mismatch.
- Inspect applied/backported PR #41872/#44401 in the real SGLang installation; not just the repository checkout.
- Confirm compiled torch/FlashInfer/sglang-kernel ABI and CUDA driver constraints; no runtime downloads, JIT cache warmup must work offline.
- Compute per-replica weight + GDN pool + KV + graph budget; cannot assume `mem-fraction-static` is free unallocated memory.

### Stage B — correctness GPU smoke (before benchmark)

- Fixed 20–50 prompts; diverse ARC-like, code, long reasoning, leading to few-token and long outputs.
- Test target-only, MTP and DFlash greedy `temperature=0` at concurrency 1 and >=8; restart each server 3 times. Capture first divergent token and corresponding logprob margin.
- Test `temperature=1.0, top_p=0.95, top_k=20`; compare logit distributions at fixed-prefix states and task accuracy on small gold subset. Do NOT require random samples to produce identical text without controlled RNG.
- Thinking enabled/disabled; reasoning-parser on/off control; strict custom budget checks if used.
- Prefix cache miss/hit at 2k/4k/8k chunk boundaries; multiple progressively appended prompts; force draft rejection/zero acceptance cases.
- At each config measure actual `#running-req`, not only assigned queue length; request churn at 11→12→13 and 16→17→20→24 boundaries.
- Longer context probes: 32k, 64k, 96k, up to target context conditional on HBM; compare repeatability after 30k–90k generated tokens when time allows.
- Stop immediately for unexplained wrong-answer semantic changes, NaNs, crashes, hangs, silent request drops, CUDA graph overrun or cross-request contamination.

### Stage C — performance only after Stage B PASS

Capture at least:

```text
engine_version/git_sha/patch_manifest
model_checkpoint_sha256
GPU_arch/CUDA/driver/NCCL_version
args_and_effective_env
engine_startup_sec
steady_state_output_tok_per_sec
end_to_end_output_tok_per_sec
median/p95_decode_step_ms
prefill_latency_ms
accepted_tokens_per_verification
per_step_running_requests/queued_requests
collective_gpu_critical_path_ms
peak_gpu_memory_gb
kv_cache_capacity_and_utilization
mamba_pool_capacity_and_utilization
cache_hit_rate
request_errors
first_divergent_token / logprob comparison
ARC task outcomes / budget consumed
```

Use `nvidia-smi topo -m`, `nvidia-smi topo -p2p p`, and if available `nccl-tests` all_reduce_perf at 8KB–1MB. A 64MB bandwidth result does NOT give per-step small-collective latency. Profiling must count GPU critical-path collectives rather than summing overlapping kernel durations across all devices.

**Decision rule:** If any arm fails output/correctness gate, reject it regardless of throughput. If two arms both pass, rank by real ARC solves per wall-clock GPU time, then full workload throughput, not tokens/s alone.

## 9. Specific research follow-ups / no-evidence statements

1. **Upstream fix for exact SGLang 13-running `dflash_worker_v2.py` tensor 12-vs-13 crash:** not independently established by this audit. Graph max and slot sizing mitigate identifiable boundary conditions; inspect exact stack and maintain regression test.
2. **#41872/#44401 solving 11-of-12 admission:** not supported by USER-OBSERVED run after backport. More code-level instrumentation needed.
3. **W4/W6/W8 DFlash throughput superiority on 4×L4 PCIe:** no exact-hardware benchmark established. W4 > W8 is a hypothesis.
4. **2×TP2 vs TP4 performance multiplier:** not proven; only topology and communication-path reasons justify the test.
5. **Competitor LB 65.97 flags transfer to RedHat INT4/FP8 checkpoints:** unproven. Public score cannot be attributed solely to serving flags.
6. **vLLM 0.32.0 solving issue #58807 for this checkpoint:** not confirmed. Check issue, merged patch and installed code, then static keys/boot.
7. **Qwen3.5/3.6/3.8 GDN in all combinations of DFlash, MTP, FP8 KV, ReplaySSM, CUDA graphs:** no single public source proves universal correctness. Separate evidence-backed risks from specific local repros.
8. **Best FP8 block128 GEMM kernel/tuning on sm_89:** cannot pick winner from docs or synthetic benchmarks alone; actual selected kernel and model accuracy must be tested.
9. **Alternative serving arms to research after validation:** `--disable-radix-cache` S=1 (may trade away prefix reuse, never presume faster); 2048 chunked prefill under mixed request load; individual TP2 replicas; a validated MTP without DFlash; `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` only if the code version and concurrency/accuracy validation support it. All remain experiments, not proven improvements.

## 10. RAW URL source index — copy/paste directly into Claude or GitHub

Every entry below is a literal full URL. `E##` identifiers in audit tables refer here. URLs are intentionally NOT formatted as Markdown hyperlinks.

**SGLang source, release, issues, PRs**

```text
E01  https://github.com/sgl-project/sglang/issues/42917
E02  https://github.com/sgl-project/sglang/issues/38009
E03  https://github.com/sgl-project/sglang/issues/36548
E04  https://github.com/sgl-project/sglang/issues/39597
E08  https://github.com/sgl-project/sglang/issues/43956
E09  https://github.com/sgl-project/sglang/issues/26330
E10  https://github.com/sgl-project/sglang/issues/25536
E12  https://github.com/sgl-project/sglang/blob/v0.5.21/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx
E13  https://github.com/sgl-project/sglang/issues/42510
E14  https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/eagle_worker_v2.py
E15  https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx
E16  https://github.com/sgl-project/sglang/issues/33483
E17  https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/speculative/dflash_worker_v2.py
E18  https://github.com/sgl-project/sglang/issues/36127
E19  https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py
S01  https://github.com/sgl-project/sglang/pull/41872
S02  https://github.com/sgl-project/sglang/pull/44401
S03  https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py
S04  https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/arg_groups/speculative_hook.py
S05  https://github.com/sgl-project/sglang/releases/tag/v0.5.21
```

**vLLM issues, fixes, release, docs**

```text
E05  https://github.com/vllm-project/vllm/pull/56531
E06  https://github.com/vllm-project/vllm/pull/59759
E07  https://github.com/vllm-project/vllm/pull/60797
E11  https://github.com/vllm-project/vllm/issues/58807
E21  https://github.com/vllm-project/vllm/issues/53670
E22  https://github.com/vllm-project/vllm/issues/55196
E23  https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/
E24  https://github.com/vllm-project/vllm/pull/52182
E25  https://docs.vllm.ai/en/v0.32.0/cli/serve/
V01  https://github.com/vllm-project/vllm/releases/tag/v0.32.0
V02  https://github.com/vllm-project/vllm/issues/60174
V03  https://github.com/vllm-project/vllm/pull/55601
V04  https://github.com/vllm-project/vllm/blob/v0.32.0/docs/features/quantization/llm_compressor/fp8.md
V05  https://github.com/vllm-project/vllm/issues/35288
V06  https://github.com/vllm-project/vllm/issues/62304
V07  https://github.com/vllm-project/vllm/pull/52966
```

**Hardware, drafter, NCCL, profiling and Kaggle**

```text
E20  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
H01  https://github.com/NVIDIA/nccl-tests
H02  https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
H03  https://www.nvidia.com/en-in/data-center/l4/
H04  https://developer.nvidia.com/blog/co-designing-ai-models-using-speculative-decoding-for-faster-llm-inference
H05  https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx
H06  https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
H07  https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
H08  https://inco.ai/blog/dflash2/
H09  https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators
```

**Source quality caveat:** Several URLs are version-tagged source snapshots (strong evidence for version), others point to `main` docs (which may have changed after the pinned release), and GitHub issues contain user reports (which may be corrected or resolved later). Model repo naming (`z-lab` vs `incoai`) must be matched against the EXACT local drafter revision; do not silently substitute weights. A URL existing or being cited does not establish the issue reproduces on the present local stack.

## 11. Instructions to Claude (paste this section with file)

You are reviewing a PRE-GPU audit, not executing GPU tests yet. Use EXACT versions, 4×L4 PCIe, and model/checkpoint revisions in §1. Start with silent accuracy, crashes and feasibility; assign GO / CONDITIONAL / NO-GO for each S1–S9 and V1–V6. For any proposed correction, provide a code-level patch or exact flag and a reproducible validation test. Open raw `https://` references from §10; verify whether a fix is in tagged source, not just merged on `main`. If a PR is unverified or an issue is for another model/hardware, say **NOT VERIFIED ON OUR STACK**. Do not spend scarce Kaggle GPU time on a configuration that cannot pass static checks. Do not call a throughput gain proven unless logs reproduce it. Produce findings in Bengali with raw URLs and exact error strings preserved. Do not optimize away long-context ARC task accuracy.

**Final status as of this README:** Accuracy-qualified TARGET-ONLY smoke first; S5 as originally sized, S8 drop-in SM89 alternatives, V3 loader mismatch pending static inspection, and V6 old environment-variable test are NOT approved. All other arms are controlled experiments subject to validation.
