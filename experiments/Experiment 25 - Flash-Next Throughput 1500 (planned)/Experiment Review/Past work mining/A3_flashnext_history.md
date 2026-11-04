# A3: Flash-Next history mining (E3, E13, Merlin, stock vLLM 0.32, ARC-AGI-3 FN track)

Date: 2026-10-31. Read-only mining for Experiment 25 (target: ≥ 1500 tok/s aggregate decode, Qwen3.8-Flash-Next, Kaggle 4×L4).

**Labels:**
- [M] measured, with the source;
- [C] a claim by a named third party, not checked by us;
- [I] my inference or arithmetic;
- UNVERIFIED: not yet confirmed by a source or a run.

**Path legend:**
- **[E3]** `D:\ARC AGI 2\experiments\Experiment 3 - Flash-Next Feasibility (FN)`.
- **[E13]** `D:\ARC AGI 2\experiments\Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)`.
- **[MS]** `D:\ARC AGI 2\_scratch\e23\p2\merlin-src` (Merlin v3.0.0, tag merlin-v3.0.0).
- **[FNP]** `[MS]\tools\merlin\profiles\flash-next.json`. It is byte-identical to `_scratch\fn_public\merlin_flash-next.json`.
- **[V32]** `D:\ARC AGI 2\_scratch\vllm032_qwen4exp_src` (stock vLLM 0.32.0).
- **[FNPUB]** `D:\ARC AGI 2\_scratch\fn_public`.
- **[RN900]** `D:\ARC AGI 2\research_notes\৪xL4 এ ৯০০ টোকেন গতির উপায়`.
- **[A3Z]** `D:\ARC AGI 2\resources\ARC-AGI-3 related\P1_SPARSE_RUN_EVIDENCE.zip`; **[A3Z1]** its copy `P1_SPARSE_RUN_EVIDENCE (1).zip`.
- **[FZ]** `D:\ARC AGI 2\_scratch\ext\kaggle_dfranzen_versions` (Franzen milestone-2 notebook versions, with outputs).

**Status note:** the E13 RC2 GPU run (`foysalemonshanto/q27-xhigh-e13-flashnext-rc2` v1, notebook 12eaa96c) has no local output yet: `_scratch\e13_rc2_gpu_v1` does not exist. So **no Merlin Flash-Next number on L4 exists yet**. Every Merlin L4 figure below is an estimate.

**Model facts used below** (Merlin card; [RN900]\engines_flashnext_duck.md:95-112):
- 125B main parameters with about 6B active. The project context says "A5B"; the card says 6B.
- 48 layers in a 3 GDN : 1 QSA pattern. 12 QSA layers with 2 KV heads and head_dim 256.
- 512 experts, 10 routed per token, intermediate size 640.
- A 51B PLE n-gram table. Vocabulary 248,320.
- GDN `mamba_ssm_dtype` float32. One MTP layer.

---

## 1. Flash-Next numbers table

### 1a. Our own Flash-Next runs on Kaggle 4×L4 (sm_89)

| Source | Hardware | Config | context@streams | tok/s | Memory | Evidence |
|---|---|---|---|---|---|---|
| E3 FN-SMOKE v3 (2026-10-05) | 4×L4 | SGLang 0.5.19 Pennyroyal, TP4/EP4, Intel W4A16 + FP8 PLE overlay (124.8 GB, 25 files) | – | – (launch exit rc 1 after 15 s, both arms: MoE runner) | sha256 192.5 s at 0.65 GB/s | [E3]\Experiment Results\Kaggle Runs\FN_SMOKE_v3_run\fn_smoke\fn_smoke_report.json `stages.1_inputs`, `stages.4_launch.attempts` |
| E3 v3.3 (2026-10-05) | 4×L4 | same; ctx 4096, running 1, chunk 1024, bf16 KV | 4k@1 | – (exit -9 at 930 s / 428 s) | weights 19.06 GB/rank, 2.55 GB free after load; KV 19,968 then 15,424 tokens; 2.44-2.51 GB free after pool; load 402 s, then 258-281 s | [E3]\...\FN_v33_run\fn_smoke\serve.log:97-137, 1285-1327 |
| E3 FN-PROBE v4 (2026-10-10) | 4×L4 | same | – | – (2 launches not ready at 1,080 s each) | hash 124.8 GB at 0.29 GB/s (428 s); 19,249 MiB used at timeout | [E3]\...\FN_PROBE_v4_run1\fn_smoke\fn_smoke_report.json `stages.1_inputs.hash_gb_per_sec`, `stages.4_launch` |
| E3 v4.2.1 (2026-10-10) | 4×L4 | SGLang 0.5.19, TP4/EP4, eager, mem 0.96, running 8, chunk 2048, ctx 16384, bf16 KV, FA2 shim | 64-token@1 | 5.95 gen tok/s (eager, output "!!!!" garbage) | load 344.8 s with prefetch; 19.06 GB/rank; KV bf16 118,080 tokens (0.68+0.68 GB/rank); 0.97 GB free after pool, 0.66 GiB after warmup; ready 862 s | [E3]\...\FN_PROBE_v421_run1\fn-probe-flashnext-4xl4.log:63-79; fn_smoke\serve.log:576-577; fn_smoke_report.json `stages.4_launch.attempts[0].sec`=862.2, `stages.5_sanity` |
| E13 SMOKE1 rev 6b (2026-10-26) | 4×L4 | SGLang 0.5.21, TP4+EP4, Franzen Intel package (BF16 PLE) | – | – (CUDA OOM in `gptq_marlin_moe_repack`) | 20.86 GiB allocated, 154.56 MiB free on TP0; 38 shards in 9:00; host need 122.4 GB of 155.1 GB cgroup free | [E13]\Experiment Results\E13_SMOKE1_rev6b_GPU_run1_2026-10-26.md:14-16 |
| E13 LOADPROBE bf16ple (2026-10-27) | 4×L4 | SGLang 0.5.21, TP4+EP4, fp8 KV (scale 1.0), mem 0.93, decode graphs | 1 stream | 40.8 tok/s with graphs (garbage output, so not meaningful) | 19.10 GB used, 2.51 GB free after load; KV fp8 94,528 tokens; 1.05 GB free after graphs; ready 1,132 s (load 582 s + graph capture 420 s TileLang JIT); host 23.84 GiB PLE per rank (BF16) | [E13]\Experiment Results\E13_LOADPROBE_GPU_run1_2026-10-27.md:12, 33, 48 |
| E13 LOADPROBE fp8ple | 4×L4 | same, FP8 PLE overlay | – | – (OOM on TP1 in the repack of layers.8) | +0.30 GiB per MoE layer not freed (20.26 → 20.86 GiB); host need 71.2 GB; 11.92 GiB PLE per rank | same file:13, 17, 24-33 |
| E13 NANPROBE rev2 kv_lean (2026-10-30) | 4×L4 | SGLang 0.5.21, mem 0.95, mamba 20 slots, graphs on | 4×20k prompts | – (still "!!!!") | KV fp8 **161,984** tokens (0.92 GB/rank); 0.62 GB free after graphs; 4×20k prefill in 37.6 s; max running seen 2 | [E13]\Experiment Results\E13_NANPROBE_rev2_GPU_run1_2026-10-30.md:49-59 |
| E13 NANPROBE nan_eager | 4×L4 | eager with hooks | – | first eager prefill 428 s (includes TileLang JIT); about 22 s per later forward (hooks on) | – | same file:47 |
| E13 RC1 vllm_fn (2026-10-31) | 4×L4 | **stock vLLM 0.32.0**, TP4+EP, eager, BF16 KV, util 0.93, max_model_len 8192, seqs 4, `--language-model-only`, engram cpu_offload | 8k | **first correct FN output** (count + "4"); the 4×256 stream test died (HTTP 500, FlashInfer sampler JIT `-lcuda`) | Model loading 18.43 GiB/rank, 610 s; loading weights 564 s (NFS, 168.75 GiB); Available KV 0.57 GiB → **30,208 tokens** (3.69× at 8k); init engine 96 s; ready 825 s; install 99 s | [E13]\Experiment Results\E13_RC1_GPU_run1_2026-10-31.md:12-29; `_scratch\e13_rc1_v1\q27x\vllm_server_e13_vllm_fn.log`:7, 48, 64, 86, 110-112, 127-128, 143 |
| E13 RC2 CPU kit v1 (2026-10-31) | Kaggle CPU | Merlin v3.0.0 offline install, `build()` + `host_phase()` for BF16 and FP8 | – | – | sha256 of 124 GB merged in 275 s; Merlin install 394 s (CPU) | [E13]\Experiment Results\E13_RC2_CPU_kit_v1_2026-10-31.md:9-18 |
| E13 RC2 GPU (running) | 4×L4 | Merlin fn-quad-24g-sm89, KV_BYTES FP8 2,479,838,700 / BF16 2,464,415,744, P2P off, WARM 0 | ladder 40k@1/2/4/5, 8k@8, census 90k/116k@3/4 | **pending** | – | [E13]\Experiment Notebooks\Build\build_e13_rc2.py:391, 401-414; [E13]\Experiment Review\RC2_BUILD_2026-10-31.md:58-66 |

### 1b. Merlin's estimates for L4 (not measured)

| Source | Hardware | Config | context@streams | tok/s | Memory | Evidence |
|---|---|---|---|---|---|---|
| Merlin preflight + matrix | 4×L4 (22.49 GiB as reported) | fn-quad-24g-sm89, FP8 KV, seqs 8 | ~1×262K / ~3×131K | none claimed ("untested") | pin lowered to 2.31 GiB (2,479,838,700 B = 150 blocks × 16,532,258); ~560K pool | [MS]\docs\merlin\hardware-matrix.md:28; [MS]\tools\merlin\preflight.py:178, 920-930, 1274-1279 |
| Our derivation from the Merlin block rule | 4×L4 | FP8 149 usable blocks; per request ceil(L/4096)+7 blocks | 70k → 5 streams (6 marginal); 100k → 4; 131k → 3 | – | FP8 pool 557,992; BF16 79 blocks = 292,103 | [E13]\Experiment Review\MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:198-219 |
| Synthesis projection [I] | 4×L4 | scaled from 3090 by 0.43-0.56 | 5-6 long streams | GUESS 200-320 aggregate; single stream 72-100; 8 streams at moderate context 300-400 | – | same file:221-230 |

### 1c. Merlin's own published Flash-Next numbers (other hardware)

| Source | Hardware | Config | context@streams | tok/s | Memory | Evidence |
|---|---|---|---|---|---|---|
| Merlin 3.0.0 bench card 2026-10-20 | 4×3090, 220 W, PCIe Gen4, **P2P on** | TP2×PP2+EP, MTP K3 probabilistic, FP8 KV, BF16 draft head, effort low | single stream, 4k / 32k / 130k / 260k | 167.7 / 167.6 / 170.3 / 187.4 (thinking on); ITL 18.37-18.94 ms; 3.03-3.44 tokens/event; thinking off 121.3-131.0 | pool 924,993; boot 231 s with a carried cache | [MS]\models\flash-next\benchmarks\2026-10-20\BENCH-CARD.md:3-8, 24-33 |
| same | same | same | 262k@1 | 183.6 on / 149.0 off | cold TTFT 44.9 s | same:37-40 |
| same | same | same | prefill | 5,159-6,081 tok/s (4k-262k); 5,660 at 10k; 6,051 at 101,875 | – | same:47-65 |
| same | same | same, thinking off | 16k agents @2/4/8 | **221.52 / 378.17 / 638.16** total (overlap windows 4.55-5.97 s, "read loosely") | – | same:104-112 |
| same | same | same | 3×262K and 6×131K resident | – | peak pool use 100 %, 0 preemptions | same:67-74 |
| 0.32 summary 2026-10-15 | 4×3090 P2P | pre-release 0.32 build | **6×126,976 prompt + 512 out each** | **574.5 aggregate**; 10.78 ms/tok per agent; cold prefill 5,423; 0 preemptions | pool 924,993 | [MS]\models\flash-next\benchmarks\2026-10-15\BENCH-CARD.md:12-20 |
| ENGINE-CHANGELOG | 4×3090 | shipped profile vs greedy draft vs greedy + local argmax | 8-way aggregate | 483 / 462 / 459 | – | [MS]\docs\merlin\ENGINE-CHANGELOG.md:119 |
| ENGINE-CHANGELOG | 4×3090 | effort xhigh vs medium | MTP tokens drafted per step | prose **2.29 vs 2.66**; code 2.35 vs 3.06 (14-25 % fewer at xhigh) | – | same:9-15 |
| CHANGELOG (DRAFT_HEAD A/B, 2026-10-17) | 4×3090 | DRAFT_HEAD int8 vs bf16 | 32k / 131k / 258k | **−6.5 to −7.2 % ms per output token**, same acceptance | – | [MS]\models\flash-next\CHANGELOG.md:10 |
| v3.0.0 (vLLM 0.30) card 2026-10-09, historical | 4×3090 | release gate | single stream | 186-206 on / 134-138 off; median ITL 16.4-17.0 ms; 3.02-3.39 tokens/event | – | [MS]\models\flash-next\benchmarks\2026-10-09\BENCH-CARD.md:21-30 |
| same | same | v2.5.1 node trace at 131k | per step | stage 0 7.0 ms + stage 1 11.0 ms + 2.0 ms neither computes; stage 1: dense GEMM 3.9, BF16 vocab head 2.25 (5 calls), Marlin MoE 1.3, all-reduce 0.6, QSA 0.66-0.89, GDN 0.25-0.35 ms | – | same:39-42 |
| same | same | concurrent cold 16k + 512 decode | @2/4/8 | 266 / 442 / **659** (v2.5.1: 204 / 371 / 598) | – | same:52-53, 63-64 |
| same | same | start-up | – | 268 s empty cache; 158-172 s warm | – | same:87 |
| v2.5.1 notes, historical | 4×3090 | 8 concurrent | – | 533.7 vs 531.5 (v2.2.0) | pool 806,792 → 924,993 | [MS]\models\flash-next\CHANGELOG.md:152, 158, 192 |
| v2.5.1 no-MTP, historical | 4×3090 | `--use-replayssm` without spec | single stream | ~82 tok/s vs 158-174 with MTP | pool 1,057,314 (MTP costs ~12.5 % of the pool) | same:223-227 |
| v1 TP4 vs v2 TP2×PP2, historical | 4×3090 | TP4+EP vs TP2×PP2+EP, MTP K3 | single stream 4k-261k | TP4 163-172; TP2×PP2 162-176 | **pool 342,912 (TP4, pin 2.9e9) vs 806,792 (TP2×PP2, pin 4.1e9)**; seqs 2 vs 8 | [MS]\models\flash-next\docs\history.md:161-172, 301 |
| history: how v2 got there | 4×3090 | PLE pull transport, pinned host embeddings | – | pull +4-10 % decode vs push; pipeline "serial (stage 0 ~9.4 ms then waits; stage 1 waits ~7.8 ms then ~13.3 ms)" | host embeddings freed 608 MiB/rank → pin 3.8e9 → 4.1e9 | same:289-293 |
| fn-quad-3090-nop2p evidence (v2.2.0) | 4×3090, **no P2P** | TP2×PP2 | 54,713 prompt | 127-141 decode incl. TTFT (vs 168-184 with P2P, −20 to −25 %); ~4,000 prefill | – | [FNP]:237, 242 |
| fn-quad-3090-nop2p-pp4 | 4×3090, no P2P | TP1×PP4, partition 12/13/13/10, seqs 32, captures to 128 | – | boot 227 s; smoke passed | **pool 506,083 at pin 2.5e9**; the 12/12/12/12 split OOMed on stage 3 | [FNP]:249-336, 462-463 |
| fn-sm80-64g-x4-pp4 (tested) | 4×CMP 170HX 64 GB, Gen2 x16, no P2P | TP1×PP4, BF16 KV, chunk 2048, MTP k3 | c1; 16k@4 | c1 142-157; 16k c4 122.2; prefill ~12,000-13,120; **MTP k=4: +4-6 % c1, −24 % at 16k c4** | pool 4,638,492; binding stage (PP3 + MTP) 19.06 GiB | [FNP]:2254, 2268-2272 |
| Community in [FNP] | 7×3090 + 1 Ti, TP2×PP4 (v2.2.0) | MTP K3, seqs 8 | 8-way | 268-312 (325-346 with greedy draft + local argmax) | KV pin 9e9 | [FNP]:453 |
| Community in [FNP] (Jon-Nielsen) | 8×3090, 180 W, no P2P (BAR1) | TP4×PP2+EP, MTP-4, BF16 KV, seqs 32, older stack | 1 / 4 streams | 131.0 / ~239; his 4-card TP2×PP2: 105.6 single | pool 1,317,519 | [FNP]:454, 582 |
| Merlin HF card (FN-specific claims) | 4×3090 | – | – | 638 for 8 agents, 184 at 262k, 6,051 prefill | 115.3 GiB checkpoint; FP8 PLE 47.7 GiB in host RAM; main shards 66.2 GiB; MTP experts 1.2 GiB | [FNPUB]\merlin_fn_card.md:29, 46-48, 67-77 |

### 1d. Other community Flash-Next numbers

| Source | Hardware | Config | context@streams | tok/s | Memory | Evidence |
|---|---|---|---|---|---|---|
| todi recipe v1.1-v1.4 (vLLM nightly eed1f3d0 / 29468dde) | 4×3090, 200 W, P2P via patched driver | TP2×PP2+EP, FP8 KV, seqs 3, util 0.94 | c1 | MTP n=2: 131.8 prose / 160.5 code (acceptance 70 %); n=3: 131.9 / 171.5 (acceptance 53 %); no spec 83.8 / 84.1 | pool MTP2 526,909; no spec 1,017,569; < 1 GiB free per GPU under MTP; host ~85 GiB (FP8 PLE) / ~149 GiB (BF16 PLE) | [FNPUB]\todi\Readme.md:14-18, 106-117, 130 |
| todi P2P note | same | – | – | P2P: 80 MB all-reduce 26 → 5 ms; 100k prefill 1.9×; **decode changed very little** | – | same:142-144 |
| chenxin 3×4090 48 GB (vLLM 0.30 + ~15 patches) [C] | 3×4090 48 GB | TP3+EP3, NVFP4 experts via Marlin, FP8 KV, MTP2 | @1/4/8/16 | 124.3 / 387.4 / 617.6 / **635.3**; at 16 streams K0 559.0, K1 636.4, **K2 644.5**, K3 541.1 | ~28 GiB per card; "24GB cards: OOM at load" | [RN900]\engines_flashnext_duck.md:63-64, 196 |
| tonyd2wild [C] | 4×3090 | vLLM TP4+EP W4A16 | single; 6 agents | 109.5-145.8; 252.6 aggregate | – | same:197 |
| SGLang cookbook [C] | 1×RTX PRO 6000 | NVFP4 | 16 req with MTP; 64 without | 613; 861 | – | same:34 |
| vLLM recipe [C] | MI355X | MXFP4, TEP8 | c8 / c32 | 665 / 1,831 | – | same:22 |
| Strata (llama.cpp-derived) [C] | 1×4090 | UD-Q4_K_XL | 1-8 clients | 64.6-86 | – | same:198 |
| Franzen M2 (Kaggle) [M] | 1×RTX PRO 6000 | SGLang Pennyroyal 0.5.19 TP1, Intel W4A16 + BF16 PLE, NEXTN 3 + FR-Spec, fp8 KV, 10 streams | 10 game streams, 139,264 ctx | **588-617 "generated tokens/sec (job wallclock)"** over the run; P90 aggregate decode 836.6; peak 946 | GPU 91.1-93.6 / 95.6 GB; host RAM 106-119 / 176.9 GB; ready 478 s / 562 s (precache 185.40 GB in 235 / 321 s) | [FZ]\v1_run354090904.ipynb cell 20 output ("595.33", "617.25"), cell 12 output ("READY after 478s"); [FZ]\v3_run354920985.ipynb cell 20 output ("582.06"), cell 12 output ("READY after 562s"), cell 18 output; [RN900]\engines_flashnext_duck.md:258-259 |
| M2 winners [C] | RTX PRO 6000 | – | peak | dfranzen 946, lordhansolo 1,135, sirikilohit 1,159; ready 531 / 544 / 615 s | – | [RN900]\engines_flashnext_duck.md:259 |
| Our ARC-AGI-3 P1_SPARSE run A [M] | 1×RTX PRO 6000 | **vLLM** NVFP4 TP1, ctx 32,768, KV pin 5 GiB auto dtype, seqs 8, MTP 3, **prefix caching off**, chunk 8192, captures 32, async | ~3.2 resident at 32k (`kv_cache_max_concurrency` 3.21) | 1,981,372 gen tokens / 7,978 s ≈ **248 tok/s**; MTP acceptance 1,269,545 / 2,137,962 = 59.4 % (≈ 2.78 tokens/step) | 20 preemptions; prompt 27,598,439 vs gen 1,981,372 (13.9:1); ready ≈ 455 s | [A3Z] p1_runtime_health.json `vllm_metrics_final.interesting_lines`, `json_artifacts.vllm-server-identity.json.json.argv`; p1_run_timing.json `run_cell_plus_teardown_s` |
| Our ARC-AGI-3 P1_SPARSE run B [M] | same | same | same | 1,937,026 / 7,966 s ≈ **243 tok/s**; acceptance 59.9 % (≈ 2.80) | 77 preemptions; prompt 28,044,262 (14.5:1); ready ≈ 535 s | [A3Z1] same keys |
| Darren (Kaggle 750546) [C] | 4×L4, model not stated | "lots of optimizations" | high concurrency, ARC-AGI-2 | peak 900-980; **710 average over 6 h, 2 passes** (tail starved); ~250 under heavy prefill; "pure decode ~1100" | – | [RN900]\engines_flashnext_duck.md:300-307 |
| TheDivine Aurora [C] | not L4 for evals | own harness | – | 250-400 tps | – | same:311 |
| Adarsh [C] | 4×L4, Franzen Duck setup | 27B vs FN | – | "almost breakeven in speed and accuracy on L4x4" | – | same:202 |

Reference point, not Flash-Next: Merlin 27B TP4 MTP4 on Kaggle 4×L4 gave 421.5 tok/s at 40k@8 and 329.0 at 120k@8, with accept length 4.227 and 0 preemptions ([E13]\Experiment Review\MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:222).

---

## 2. Merlin Flash-Next knobs table

How Merlin applies them:
- `launch._build_flash_next` reads these from the caller's environment.
- It pops every `MERLIN_*` variable, plus FN_UNSET and FN_RESET_EXTRA, then sets its own values ([MS]\tools\merlin\launch.py:377-386).
- `Plan.argv()` = prefix + `vllm serve ckpt` + profile args + **extra args (last)** (launch.py:110-111). So a trailing flag in `extra` overrides the profile flag.

### 2a. Launcher knobs (env read by launch.py)

| Knob | Default | Values | Documented effect | Evidence path:line |
|---|---|---|---|---|
| `KV_BYTES` | profile `serve.kv_cache_memory` = 4,100,000,000 for fn-quad-24g-sm89 (NOT lowered automatically by `build()`) | whole bytes | `--kv-cache-memory-bytes` per rank. Preflight prints the L4 value 2,479,838,700 (FP8, 150 blocks); "adjust in 0.25 GiB steps (lower it if the boot OOMs)" | launch.py:303-305, 456; preflight.py:920-930, 1229-1263, 1274-1279; [FNP]:943 |
| `CHUNK` | 1024 (`max_num_batched_tokens`) | integer | `--max-num-batched-tokens`. On CMP PP4, chunk 2048 gave +7-12 % prefill and +15 % at 16k c4, headroom unchanged | launch.py:333, 454; [FNP]:2272 |
| `BLOCK_SIZE` | 4096 (base) | integer | `--block-size`. 4096 = four 1,024-token chunks per block. The computed 3,392 block was 6-12 % slower prefill under RecoverSSM; 4096 costs 0.5 % of the pool | launch.py:334, 474; [MS]\models\flash-next\CHANGELOG.md:209 |
| `LPT` | unset (no flag) | integer | `--long-prefill-token-threshold` when set | launch.py:335, 463-464 |
| `DRAFT_HEAD` | `bf16` (`R.DRAFT_HEAD_DEFAULT`) | bf16, int8, int4 | Adds `"draft_lm_head_quant":"int8"` to the MTP spec JSON. int8: −6.5 to −7.2 % ms/token at the same acceptance (3090), planned as the next default | launch.py:319-322, 441; registry.py:405; [MS]\models\flash-next\CHANGELOG.md:10 |
| `REDUCED_DRAFT_VOCAB` / `DRAFT_VOCAB` | 0 | 0, 1 (1 needs a pinned table that does not ship) | Reduced draft vocabulary; unavailable in 3.0.0 | launch.py:323-329, 426-439; reference.md:198, 217-219 |
| `WARM` | profile `warm`: **0** on sm_89 (1 on the sm_86 rows) | 0, 1 | 1 sets `TRITON_CACHE_DIR=$CACHE_ROOT/triton` + `--kernel-config` (persist Triton autotune, QSA warmup bounds, reuse GDN prefill warmup). The registry enforces warm 0 off sm_86: "persisted Triton autotune is sm_86 only". On sm_89 "the first boot autotunes the GDN kernels live … can pick a slow configuration" | launch.py:54, 330-332, 405-408; registry.py:195; [FNP]:920, 940; hardware-matrix.md:216 |
| `P2P` | profile default **off** for sm_89 | on, off | off: `NCCL_P2P_DISABLE=1`, `VLLM_SKIP_P2P_CHECK=1`; unset NCCL_P2P_LEVEL/PROTO/ALGO and `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`; `--disable-custom-all-reduce`. on: `NCCL_P2P_LEVEL=SYS`, `NCCL_PROTO=Simple`, FlashInfer PCIe IPC all-reduce for decode | launch.py:349-371, 449-450; [FNP]:891-918; ENGINE-CHANGELOG.md:90-93 |
| `AUTO_TOPO` | 0 | 0, 1, strict | Runs `topo_select.py` (GPU order, P2P/NCCL vars). Applies only to four-card P2P-on rows, so it is ignored at P2P=off | launch.py:337-344, 359-362; reference.md:110-128 |
| `PLE_HOME` | 0 | GPU index | `VLLM_PLE_OFFLOAD_HOME_DEVICE`: the PLE offload process's home device | launch.py:384; [MS]\vllm\v1\ple_offload\worker.py:179-189 |
| `COUNTERS` | 0 | 0, 1 | 1 sets `MERLIN_TOPK_EXACT_COUNTERS_LOG=30` and `VLLM_QSA_KV_CLIP_COUNT=300`. The clip counter disables the fused QSA writer, so it does not test the production path | launch.py:421-423; synthesis:81 |
| `SCALES` | checkpoint `qsa_kv_scales.json` if present, else recipe `scales/qsa_kv_scales_262k.json` | path | `VLLM_QSA_KV_SCALES` (FP8 only). Without it, FP8 KV runs at scale 1.0 | launch.py:395-403; reference.md:215-217 |
| `CHAT_TEMPLATE` | froggeric v22.5 (shared with 27B) | path; empty = the checkpoint's own | `--chat-template` | launch.py:53, 388-394 |
| reasoning effort (no env knob) | **medium** (`--default-chat-template-kwargs`) | trailing `--default-chat-template-kwargs` in extra overrides it | xhigh drafts fewer tokens per step (2.29 vs 2.66) | launch.py:470-472; registry.py:409; ENGINE-CHANGELOG.md:9-15 |
| `GUARD` / `VLLM_E2_GUARD_MODE` | warn | raise, warn (off refused) | Bounds-guard reader; costs ~0.3 ms per decode step on TP2×PP2 | launch.py:273-289; CHANGELOG.md:204; reference.md:134-136 |
| `VLLM_E2_GUARD_INTERVAL` | 0.05 s | > 0 | Guard reader poll period | launch.py:282-289; [MS]\vllm\merlin\envs.py:596 |
| `IMG_CAP_PX` | 8,388,608 | ≥ 65,536 | `--mm-processor-kwargs` longest_edge. Image traffic is not used by ARC | launch.py:245-251, 458 |
| `MODEL_NAME`, `HOST`, `PORT` | "flash-next-v2 flash-next flash-mtp flash-next-mtp", 127.0.0.1, 8000 | – | served names and endpoint (RC2 adds `--served-model-name q27`) | launch.py:440-445; [FNP]:14, 20 |
| `CUDA_VISIBLE_DEVICES` / `GPUS` | 0,1,2,3 | – | GPU order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`); TP pairs (0,1)/(2,3) | launch.py:364-366; [FNP]:919 |
| `TREE`, `VENV`, `CACHE_ROOT` | repo, repo/venv, repo/.vllm-cache | paths | engine tree, venv, `VLLM_CACHE_ROOT` | launch.py:306-316, 404 |

### 2b. Fixed by the launcher (no env knob; override only with trailing extra args)

| Setting | Value | Evidence |
|---|---|---|
| MTP | `{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}` (K fixed by `R.MTP_K=3`; no `SPEC_K` for Flash-Next — `SPEC_K` exists only for the 27B) | registry.py:49; launch.py:441, 452, 503-508 |
| `--max-num-seqs` | 8 FP8 / 3 BF16 variant. Profile note: "capped at the 131K-session fit and at 8 (the only concurrency the engine is gated at)" | [FNP]:853, 871, 942 |
| CUDA graph | FULL_AND_PIECEWISE, captures [1,2,3,4,6,8,12,16,24,32]. **Capture sizes are counted in tokens = seqs × (K+1)** (`capture_sizes_for`, and "extend to 128 decode tokens (32 x (K+1))") | launch.py:459-460; registry.py:83-85; [FNP]:461 |
| Layout | TP2×PP2, `--enable-expert-parallel`, `VLLM_PP_LAYER_PARTITION=25,23` ("capacity-optimal: 26/22 → 857,585, 24/24 → 739,100") | launch.py:373-376, 445-448; [FNP]:843-851, 137 |
| KV dtype | fp8_e4m3 + scales (BF16 variant via `kv="bf16"`) | launch.py:456; registry.py:272-294 |
| Caching | `--enable-prefix-caching --mamba-cache-mode align --prefix-match-unit 64`, FGPC off, `--use-replayssm` (RecoverSSM) | launch.py:461, 474-475; reference.md:201 |
| Memory | `--gpu-memory-utilization 0.96` (overridden by the KV pin); `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`; `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256`; `VLLM_E2_QSA_LOGITS_GRANULE_MB=256`; PYTORCH_CUDA_ALLOC_CONF removed | launch.py:55-61, 347, 417-420, 455 |
| Host embeddings / PLE | `VLLM_HOST_EMBED_TABLE=1`, `VLLM_E2_PLE_FAST_IDS=1`, `VLLM_E2_PLE_EARLY_REQUEST=1` (0.11-0.16 ms per output token faster), `VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800` (RC2 raises it to 3600 after build) | launch.py:383-386, 415-416; CHANGELOG.md:67 |
| Runner / kernels | `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `MERLIN_FULL_K/QSA_SORT/TIE_DET/TIE_RECENT/TOPK_SORTED_EMIT=1`, `--no-enable-flashinfer-autotune` | launch.py:386, 411-414, 466 |
| Multimodal | `--limit-mm-per-prompt {"image":42,"video":0}` (the vision tower stays loaded; `--language-model-only` is not passed) | launch.py:457; PLAN_1500.md:39 |
| Sampling | `--override-generation-config` T 1.0 / top_p 0.95 / top_k 20 (every request must carry its own sampling) | launch.py:51, 468 |

### 2c. Engine env knobs in `vllm/merlin/envs.py` (runtime; NOTE: launch.py pops every `MERLIN_*` variable, so these must be re-exported after `build()`)

| Knob | Default | Documented effect | Evidence |
|---|---|---|---|
| `MERLIN_MARLIN_MOE_CFG` | off | Per-shape override of the Marlin MoE `block_size_m` and thread config (`m=…,bm=…,g=…,ack=bitwise`), from already-compiled kernels. With EP, `m` = ceil(M·E/global_experts). Needs an `ack` from `benchmarks/kernels/bench_marlin_moe_cfg.py` | envs.py:674-676; [MS]\vllm\merlin\marlin_moe_cfg.py:1-60 |
| `VLLM_GDN_DECODE_FASTPATH` | off | Pure spec-decode FULL-graph batches stage GDN metadata with slices instead of CPU-mask indexing + H2D copies | envs.py:550-556 |
| `VLLM_GDN_SHARED_ALIGN_INDEX` | off | Align-mode GDN gather index computed once per batch | envs.py:543-549 |
| `VLLM_TARGET_SEQ_LENS_ASYNC` | off | Verify build without the seq_lens device sync | envs.py:524-538 |
| `VLLM_DRAFT_BOUNDED_PLAN` | off (and dropped by the Flash-Next launcher) | Draft decode planned from CPU bounds without a sync | envs.py:509-523; launch.py:123 |
| `VLLM_GDN_BA_AUX_STREAM` | off (dropped by the launcher) | GDN in_proj_ba on an aux stream | envs.py:651-654; launch.py:123 |
| `VLLM_DRAFT_SEQ_LENS_NOSYNC` | on | Draft step ≥ 2 uses CPU seq_lens + 1 | envs.py:496-504 |
| `VLLM_FLASHINFER_PLAN_CACHE`, `VLLM_GDN_BA_SPLITK` | on | sm_86 27B-series knobs (FA2 plan cache; GDN BA split-K at M ≤ 16) | envs.py:482-495 |
| `VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE` | off | RecoverSSM commit order | envs.py:539-542 |
| `VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK`, `VLLM_SPEC_STEP_KEYED_RNG` | off | Draft truncation / keyed RNG | envs.py:577-584 |
| `VLLM_E2_SCHED_MISS_DEFER_S` | 120 s | Cost-aware admission: parks a cold miss behind pinned waiters (prefix-cache thrash fix) | envs.py:677-680; [MS]\vllm\v1\core\cache_thrash.py:1-28; ENGINE-CHANGELOG.md:124 (Flash-Next follow-up reuse 74.7 % at 8×131K and 82.1 % at 12×80K oversubscription) |
| `VLLM_E2_SCHED_COHERENT_YIELD` | on | Session-coherent yields | envs.py:681-683; cache_thrash.py:11-18 |
| `MERLIN_SHARE_DRAFT_WORKSPACE`, `VLLM_MERLIN_VISION_ROTARY_INPLACE` | off (on for the 27B single-card rows) | Memory savings (MTP draft shares the target workspace; in-place vision rotary) | envs.py:562-569; ENGINE-CHANGELOG.md:17-23 |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | 600 (launcher sets 1800) | PLE load / worker registration timeout | envs.py:573-576 |
| `VLLM_PLE_PROGRESS_DEADLINE_S` | 300 | PLE progress watchdog (in FN_UNSET) | envs.py:605-607; launch.py:59 |
| `VLLM_E2_MEMSTATS(_DIR)` | 0 | Memory stats (in FN_UNSET) | envs.py:600-601 |
| `use_local_argmax_reduction` + greedy draft (CLI/spec) | off | +10-20 % 8-way on 8×3090 TP2×PP4; no gain on 4×3090 (483 shipped vs 459) | ENGINE-CHANGELOG.md:119; [FNP]:467 |

---

## 3. GOOD (what worked)

1. **Weight load on Kaggle 4×L4.**
   - The Intel W4A16 non-PLE weights load at 18.4-19.1 GB per rank under TP4/EP4: SGLang 19.06 GB, vLLM 18.43 GiB.
   - `--weight-loader-prefetch-checkpoints` brought load to 345 s for 124.8 GB ([E3]\README.md:15; FN_PROBE_v421_run1 log:63).
2. **The FP8 PLE overlay works.** 10 shards, 51.2 GB, sha-verified on Kaggle. It cuts host PLE from 23.84 to 11.92 GiB per rank in SGLang ([E13] LOADPROBE.md:33). Merlin uses one shared 47.7 GiB copy ([FNPUB]\merlin_fn_card.md:68).
3. **Host RAM fits.**
   - SGLang pinned about 95 GiB of BF16 PLE without failure (NANPROBE.md:36).
   - The Kaggle cgroup is 185.8 GB and MemTotal 197.9 GB (synthesis:138-142).
4. **The OOM was understood.** The SGLang repack OOM is a delayed free (+0.30 GiB per MoE layer). It is not a version or PLE effect (LOADPROBE.md:20-36).
5. **The NaN was localised.** It starts in `model.layers.2.mlp.experts` (GPTQMarlinMoE) with finite inputs on all ranks, so it is in the SGLang EP4 Marlin MoE path (NANPROBE.md:22-27; RC1.md:31-53).
6. **First correct Flash-Next output on Kaggle 4×L4: stock vLLM 0.32.0 + Intel W4A16**, TP4+EP, eager, PLE cpu_offload (RC1.md:10-20). The gc hook freed 0 bytes there; vLLM has no delayed-free problem.
7. **The SGLang KV knob responded.** mem 0.93 → 0.95 raised the pool from 94,528 to 161,984 fp8 tokens (+71 %). 4×20k prefill completed in 37.6 s (NANPROBE.md:49-58).
8. **The Merlin env is proven on Kaggle.**
   - On Kaggle GPU (E23 27B): glibc 2.35 shim, CUDA 13, GCC 13, Python 3.13.
   - On the Kaggle CPU kit: `host_phase()` passes for BF16 and FP8 Flash-Next plans; 124 GB sha256 in 275 s (RC2_CPU_kit.md:9-18).
9. **Merlin TP2×PP2+EP is the capacity layout.** Measured on 3090: pool 806,792-924,993 vs 342,912 at TP4, because TP4 duplicates the 2 KV heads (history.md:164; [FNP]:458).
10. **Merlin's reference workloads are close to ours.**
    - Six 127k sessions: 574.5 tok/s, 0 preemptions.
    - Six 131K + three 262K resident with 0 preemptions; 8 agents 638 tok/s.
    - These are long-context multi-stream results on 3090 (BENCH 2026-10-15, 2026-10-20).
11. **Franzen's Duck + Flash-Next on one PRO 6000** sustained 588-617 tok/s job-wallclock with 10 streams. Admission was capped at the server limit to keep prefixes resident, with 93.43 % prefix reuse ([FZ] v1 cell 20; [RN900]\engines_flashnext_duck.md:258, 266).

## 4. BAD (failures and root causes)

| Failure | Root cause (status) | Evidence |
|---|---|---|
| E3 v3: launch rc 1 after 15 s (both arms) | AutoRound MoE needs runner `auto`; TP4 Marlin N=160 invalid (640/4 not a multiple of 128) → EP4 required | [E3]\README.md:12; [E13]\VALIDATION.md:26 (row 16) |
| E3 v3.3: rc −9 at 930 s / 428 s | killed at small ctx/KV (4k, KV 15-20k); host/GPU limit (not diagnosed further) | FN_v33_run fn_smoke_report.json `stages.4_launch` |
| E3 v4: 2 launches not ready in 1,080 s | slow /kaggle/input (hash 0.29 GB/s); no prefetch | FN_PROBE_v4_run1 report `stages.1_inputs.hash_gb_per_sec`; [E3]\README.md:13 |
| E3 v4.2.1, E13 LOADPROBE / NANPROBE: "!!!!" token 0 | SGLang EP4 GPTQ-Marlin MoE outputs wrong from layer 1 and NaN at layer 2. PLE (orig/sync/skip) and checkpoint scales are ruled out. The exact defect (repack layout / EP launch / runner wiring) is not pinned down | RC1.md:31-65; NANPROBE.md:15-43 |
| E13 SMOKE1 + LOADPROBE fp8ple OOM in `gptq_marlin_moe_repack` | old Marlin buffer not freed (reference cycle freed only by GC; UNVERIFIED which object) | LOADPROBE.md:20-36 |
| SGLang SSM dtype mismatch | SGLang logs `mamba_ssm_dtype=bfloat16`, config says float32 | NANPROBE.md:46 |
| TileLang "Data race detected: `Logits(bx, position)`" in the QSA indexer during graph capture | candidate only; not the measured NaN site | LOADPROBE.md:47 |
| RC1 4-stream test HTTP 500, engine dead | FlashInfer 0.7.0.post1 sampling JIT link `/usr/bin/ld: cannot find -lcuda` (environment) | RC1.md:22-25 |
| RC1 stock vLLM KV only 30,208 tokens | TP4 duplicates KV heads; BF16 KV; 0.57 GiB available for KV | RC1.md:27-29; vllm log:127-128 |
| RC1 arm 3 skipped | host-RAM guard counted page cache (cgroup free 8.9 GB with MemAvailable 55 GB); vLLM saw "Available RAM 0.24 GiB" and skipped its prefetch | RC1.md:57-59; RC2_REVIEW_PAST_EXPERIMENTS.md:63-66 |
| RC1 `expandable_segments … OOM … free: 3211264` warnings during load; 1 leaked shared_memory object | thin margin; PLE /dev/shm unlinked only on a graceful teardown | vllm log:121-127; RC2_REVIEW:56-78 |
| Stock vLLM 0.32 refuses PP>1 with PLE | `model_state.py` raises "N-gram PLE embedding currently requires pipeline_parallel_size=1" (Merlin carries the #54709 fix) | [V32]\vllm\models\qwen4_exp\nvidia\model_state.py:35-41; synthesis:100 |
| Community: PP + MTP + prefix caching → 12.5 % constant-token loops at ≥ 3 concurrent | mamba spec block tables indexed by batch row (#55506); no crash, nothing logged | [FNPUB]\todi\Readme.md:315-326 |
| Community: `--no-async-scheduling` deadlocks PP (gloo timeout after 30 min, 36/36 fail) | PP stage 2 blocks in `irecv_tensor_dict` | same:328-332 |
| Community: vLLM Marlin with EP multiplied top-k weights inside the kernel → garbage | chenxin patch 02k (upstream status unknown) | [RN900]\engines_flashnext_duck.md:64, 180 |
| Community: v0.32 single prefill crossing 262,144 → illegal memory access (#62383, open); indexer workspace not profiled → first long prefill OOM (#60489) | upstream bugs | same:25-26 |
| ARC-AGI-3 FN track (our vLLM NVFP4, PRO 6000): only ~243-248 tok/s with 20/77 preemptions | KV pin 5 GiB = 3.2 concurrency at 32k; prefix caching off → prompt:gen 14:1 re-prefill | [A3Z]/[A3Z1] p1_runtime_health.json metrics |
| Merlin on L4: three things are silent if left out | `build()` keeps the 4.1e9 pin unless `KV_BYTES` is exported; the server default is medium effort; Merlin `MERLIN_*` env is popped | synthesis:38-41; launch.py:303, 377, 470-472 |

## 5. What the ARC-AGI-3 FN track did that transfers (engine, concurrency, harness)

**Our ARC-AGI-3 Duck + Flash-Next NVFP4 runs** (vLLM `0.1.dev20073+g8e685d198`, TP1, RTX PRO 6000):
- Argv: `--kv-cache-memory-bytes 5368709120 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 --async-scheduling --max-cudagraph-capture-size 32 --no-enable-prefix-caching --speculative-config {"method":"mtp","num_speculative_tokens":3}` ([A3Z] p1_runtime_health.json `json_artifacts.vllm-server-identity.json.json.argv`).
- Transfers as lessons:
  - (a) A KV pin far below the stream count gives ~3 real running streams and preemptions (20/77) → ~245 tok/s on a 1.8 TB/s card.
  - (b) Prefix caching OFF made prompt tokens 14× the generated tokens. Every Duck turn re-prefilled.
  - (c) MTP K3 acceptance under a real agent load was 59-60 % per draft token (≈ 2.8 tokens/step).
- Merlin keeps prefix caching on and must, because MTP + prefix-off failed correctness in E19 (memory `experiment-19-vllm-correctness.md`).

**Franzen's M2 solution** ([FZ] cell 12 src lines 12-35; [RN900]\engines_flashnext_duck.md:225-277). The parts that transfer:
- **Admission = residency.** Admit exactly as many streams as the server holds (10 = `--max-running-requests`). A stream keeps its slot across tool calls and resumptions and requeues only on context eviction.
  - On 4×L4 Merlin this means a client-side cap at the block-rule fit (5 at 70k on FP8). Otherwise re-admission thrash applies, as the RC2 plan already sets (synthesis:180).
- **Prefix-aware scheduling.** SGLang `lpm` + page 64 + a final-prefill Mamba checkpoint. On Merlin the equivalents are prefix caching align + `--prefix-match-unit 64` + `VLLM_E2_SCHED_MISS_DEFER_S` / coherent yield (launch.py:461, 475; cache_thrash.py:11-24).
- **Soft startup.** Release the harness while the server loads; precache the checkpoint in parallel (185 GB in 235-321 s on PRO 6000).
- **A priority scheduler.** All tasks start; priority by expected gain. A tail-concurrency lever: Darren's 710 average vs 980 peak is "tail starved" ([RN900]\engines_flashnext_duck.md:304).
- **Calibrated token accounting** for admission.

**What does not transfer:**
- TP1 on 1.8 TB/s with ~1M KV tokens and no all-reduce.
- NEXTN + FR-Spec on SGLang (wrong outputs on L4 in our runs).
- The ARC-AGI-3 caps (12,288 output, 2,048-token yield, 57k trim), by project rule (memory `arc-agi2-needs-long-thinking.md`).

---

## 6. Direct implications for reaching 1500 on 4×L4

**Current measured anchors:**
- Flash-Next on L4: correctness only (RC1). No speed number exists yet.
- Merlin 3090 multi-stream: 574.5-659 tok/s at 6-8 streams.
- 3090 to L4 ratio: 0.43-0.56 (synthesis:224).

So reaching 1500 on L4 needs every term below to move. Each term, what it is today, and its lever:

**1. Resident streams (KV pool).**
- FP8 pin 2.31 GiB = 149 usable blocks. Each request costs ceil(L/4096) + 7 blocks.
- At 70k that is 5 streams; at 8 streams the average context must stay ≤ ~45-47k (synthesis:208-215; RC2_REVIEW:101-103).
- 1500 at 5 streams would need 300 tok/s per stream: about 1.6× Merlin's 3090 single stream on a card with ~1/3 the bandwidth. **Raising residency is therefore lever #1.**
- What can raise it:
  - **Pin +0.25 / +0.5 GiB:** about +16 blocks each (synthesis:218).
  - **Free VRAM.** Load text-only, since Merlin passes `--limit-mm-per-prompt image 42` and not `--language-model-only` (launch.py:457; PLAN_1500.md:39). Lower the 2.75 GiB overhead (graphs, the 256 MiB indexer logits, sampler).
  - **Shrink the fixed 7 blocks per request** (≈ 115 MB/card each; 56 of 149 blocks at 8 streams). [I] UNVERIFIED: these 7 are most likely recurrent GDN state slots. MTP costs 12.5 % of the Merlin pool (1,057,314 → 924,993, CHANGELOG.md:223-227) and 48 % in todi's stack (1,017,569 → 526,909). So the SSM state dtype and K are capacity levers.
  - **Admission by live blocks, not worst-case length.** vLLM allocates blocks on demand. With a growing-thinking workload, the mean live context is about half the final length. A preemption-free client admission controller can hold more streams than the worst-case fit.
- **DP is not an option for W4A16.** One TP2 replica needs ≥ (66.2 + 1.2)/2 ≈ 33.7 GiB per card against 22.03 GiB CUDA-visible ([FNPUB]\merlin_fn_card.md:67-69; vLLM log:121 total 23,659,151,360 B). [I]
- **PP4×TP1 leaves almost no KV.** Binding stage 19.06 GiB + 2.75 GiB overhead ≈ 21.8 GiB ([FNP]:2254). [I]

**2. Bytes read per step (L4: 300 GB/s per card).** [I] Roofline arithmetic, an estimate to be replaced by RC2 numbers.
- Per card under TP2×PP2+EP: about 14-15 GiB of routed experts plus about 3 GiB dense. The routed experts are ≈ 58 GiB in total (48 × 512 × 3 × 2560 × 640 int4 + g128 scales); the binding stage is 17.44 GiB ([FNP]:1401).
- The share of a rank's experts touched per step is f = 1 − (1 − 10/512)^T, with T = streams × (1+K) verify tokens.
  - 8 streams, K3 (T=32): f = 0.47 → ≈ 10 GiB/card/stage → ≈ 35 ms per stage.
  - 16 streams (T=64): f = 0.72 → ≈ 13.4 GiB → ≈ 48 ms.
- So at Duck-scale concurrency **Flash-Next reads more bytes per card per step than the 27B INT4 under TP4**: about 4.9 GB/card, all weights, every step. "6B active" pays off only at low concurrency ([RN900]\engines_flashnext_duck.md:146 agrees).
- Implied ceilings, with L = accepted tokens per step:

  | Streams | Serial PP2 (step = 2 stages + ~2 ms, BENCH 2026-10-09:39-42) | Two microbatches in flight (perfect overlap) |
  |---|---|---|
  | 8 | ≈ 255-335 tok/s (L = 2.3-3.0) | – |
  | 16 | ≈ 375-490 | ≈ 525-685 |
  | 24 | – | ≈ 650-850 |
  | 32 | – | ≈ 770-1,000 |

- Long-context KV, indexer and GDN-state traffic come on top and lower these.
- **For 1500 the step must therefore carry more tokens per byte read.** That means:
  - PP microbatch overlap (open question 1);
  - higher accepted length per step;
  - fewer expert bytes (a sub-4-bit or expert-pruned checkpoint: REAP-288 GGUF, EXL3 3.05 bpw exist, [RN900]\engines_flashnext_duck.md:123; none ships for Merlin);
  - or a layout without the PP serial bubble that keeps the pool (TP4 loses 57 % of the pool: 342,912 vs 806,792).

**3. MTP depth vs concurrency.**
- Each extra draft token adds verify tokens, so it touches more experts.
- Evidence:
  - chenxin at 16 streams: K2 644.5 > K1 636.4 > K0 559.0 > K3 541.1;
  - Merlin CMP: k=4 −24 % at c4 while +4-6 % at c1 ([FNP]:2269);
  - todi: n=3 acceptance 53 % vs n=2 70 % ([FNPUB]\todi\Readme.md:108-110).
- At xhigh, Merlin's acceptance is 2.29 tokens/step (prose).
- **Expect K2 (or K1) to beat K3 above ~8 streams.**
- K is not a Merlin knob for Flash-Next. It needs a trailing `--speculative-config` in extra (argv order launch.py:110-111) that restates `draft_sample_method` and `draft_lm_head_quant`. The capture list also needs extending if seqs×(K+1) > 32 (registry.py:83-85).

**4. DRAFT_HEAD=int8:** −6.5 to −7.2 % ms/token, the only Merlin-measured decode lever (3090, single stream; CHANGELOG.md:10). The 5 vocab-head calls per step take 2.25 ms of the stage-1 11 ms on 3090 (BENCH 2026-10-09:41), so the gain may be larger on L4 at batch.

**5. Graph mode and live autotune on sm_89.**
- FULL graphs about double decode vs PIECEWISE in todi's stack ([FNPUB]\todi\Readme.md:164). On Merlin 27B, FULL_AND_PIECEWISE was downgraded to PIECEWISE under spec decode (RC2_REVIEW:154-158). The RC2 log must show which mode Flash-Next got.
- With WARM=0 on sm_89, GDN Triton kernels autotune live and "can pick a slow configuration" (hardware-matrix.md:216).
- QSA Triton has tuning tables only for sm_90/sm_120 ([V32]\vllm\models\qwen4_exp\nvidia\ops\qsa.py:19-28).
- `cooperative_topk` needs sm_90; sm_89 uses `persistent_topk` ([V32]\…\ops\qsa_indexer.py:486-495).
- → Kernel tuning for sm_89 is open headroom: Triton configs, and `MERLIN_MARLIN_MOE_CFG` per-shape Marlin MoE (re-export after build; envs.py:674-676).

**6. Interconnect.**
- P2P on L4 is UNRESOLVED: E14 showed `can_access_peer` True and NCCL "via P2P/IPC" (synthesis:90).
- todi: P2P barely changes decode but gives 1.9× prefill.
- Merlin v2.2.0 no-P2P: −20 to −25 % decode incl. TTFT ([FNP]:237).
- E22: P2P IPC all-reduce 0.68× vs NCCL SHM on Kaggle (RC2_REVIEW:143).
- → P2P is a measured arm (A7), not a default.

**7. Prefill and the Duck workload.**
- Merlin 3090 prefill is 5.2-6.1k tok/s; L4 is unmeasured (RC2 A1 measures 40k TTFT).
- Darren reports throughput dropping to ~250 under heavy prefill.
- Our ARC-3 track shows what re-prefill costs (14:1).
- → Keep prefix caching on, cap admission at residency, and use chunk 2048 for prefill (+7-12 % on CMP).

**8. Time budget.**
- 1500 × 41,400 s ≈ 62.1M tokens (PLAN_1500.md:17).
- Cold boot GUESS 15-25 min (synthesis:162). Our measured Flash-Next readies: 825 s stock vLLM, 862-1,132 s SGLang.
- Every boot minute costs ~90k tokens at 1500 tok/s. One boot per run, and a compile-cache tarball, matter.

**9. Host RAM.**
- Merlin: 47.7 GiB pinned FP8 PLE + ~4.2 GiB pinned embeddings; 69 GiB floor; 96 GB qualified (launch.py:67-70).
- The Kaggle cgroup is 185.8 GB. The vLLM-visible "available" was only 92-99 GiB in E23 (synthesis:143).
- → One Merlin server per container. Never stage weights in /dev/shm.

**10. Coverage arithmetic (not throughput).**
- Swift1.5-Flash-Next (31-58 % fewer thinking tokens on BF16 benchmarks; profile `fn-swift1.5-quad-24g-sm89` exists; licence §5 risk) lowers the tokens per task rather than raising tok/s (synthesis:184, 306-323; [FNP]:2546).

## 7. Open questions

1. Does Merlin's PP2 + MTP run two microbatches in flight (stage overlap), or is decode stage-serial at every concurrency as in the single-stream trace (history.md:293)? This decides the ceiling in §6.2.
2. What is the real L4 pool at KV_BYTES 2,479,838,700 / 2,464,415,744, and the minimum free memory per stage? CUDA reports 22.03 GiB against preflight's 22.49 GiB (23,034 MiB); does the 2.75 GiB overhead hold on L4?
3. What are the 7 fixed blocks per request (GDN state slots × (1+K)? conv state? MTP)? How do they change with K = 1/2 and with the SSM state dtype? Is `--mamba-ssm-cache-dtype` accepted by Merlin, and is long-reasoning output still correct?
4. Does native E4M3 QSA KV give BF16-level logprobs on sm_89? Does the MTP draft QSA layer run at the default scale 1.0 (synthesis:75-83)?
5. Real aggregate decode on L4 at 1/2/4/5/8 streams with xhigh 60k+ thinking: accepted tokens per step, step time per stage, CUDA graph mode actually used (FULL vs PIECEWISE), and the effect of live autotune (WARM=0).
6. Does a trailing `--speculative-config` / `--compilation-config` override in extra args replace the profile's (last wins in argparse) without breaking Merlin's guards? Does seqs > 8 boot, given "8 is the only concurrency the engine is gated at" ([FNP]:942)?
7. What do `MERLIN_MARLIN_MOE_CFG` rules, `VLLM_GDN_DECODE_FASTPATH`, `VLLM_TARGET_SEQ_LENS_ASYNC` and `DRAFT_HEAD=int8` give on sm_89 at 8-16 streams? Are they bitwise identical?
8. Is P2P real on Kaggle L4 (E14 `can_access_peer` True), and does Merlin P2P=on help decode or only prefill?
9. How does Merlin's admission policy (`VLLM_E2_SCHED_MISS_DEFER_S` 120 s) interact with a client-side residency cap under a Duck multi-turn load on a 149-block pool?
10. Is there a sub-4-bit or expert-pruned Flash-Next checkpoint loadable by Merlin/vLLM (EXL3 and GGUF-only today) that cuts expert bytes per step and frees KV?
11. Darren's 980 peak / 710 average on 4×L4: which model and layout? This is the only ≥ 700 tok/s claim on this hardware.
12. Cold vs warm boot time for Merlin Flash-Next on Kaggle: /kaggle/input MB/s for 124 GB, torch.compile for 2 stages, live autotune. Can a cache tarball make the production boot warm?
