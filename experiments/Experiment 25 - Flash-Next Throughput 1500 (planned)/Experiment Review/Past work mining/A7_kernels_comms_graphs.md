# A7: kernels, communication, CUDA graphs and profiling (past-work mining for E25)

Date: 2026-10-31. Read-only mining by agent A7. Scope:
- E12 FP8 kernel tuning;
- the E14 NCCL/P2P probe and engine findings;
- E22 ENGINE_FINDINGS;
- E23 P4 / P5 / G3 / PP4 and the SMOKE profiler traces;
- the vLLM 0.32 and Merlin v3.0.0 `qwen4_exp` sources, for their sm_89 fallbacks.

Nothing was run or pushed.

**Path legend:**
- **[E14R]** `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/Kaggle Runs/E14A SLIM R2 run1 (hosen42, 2026-10-23)/bundle/e14/`
- **[E23S]** `experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/Experiment Results/smoke_rev3g/`
- **[E22F]** `experiments/Experiment 22 - Speed lab vLLM 0.32 + SGLang (planned)/ENGINE_FINDINGS_vLLM032_SGLang.md`
- **[E14F]** `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/README_FINDINGS.md`
- **[VMS]** `_scratch/ems_out/vms/q27x/vms/` (VMS server logs, 2026-10-26)
- **[ST]** `_scratch/vllm032_qwen4exp_src/vllm/` (stock vLLM 0.32.0 qwen4_exp extract)
- **[MS]** `_scratch/e23/p2/merlin-src/` (Merlin v3.0.0 checkout; `vllm/` is its engine tree)
- **[FNP]** `_scratch/fn_public/merlin_flash-next.json` (= Merlin `tools/merlin/profiles/flash-next.json`)
- **[CFG]** `_scratch/hf_intel_fn_cfg/config.json` (Flash-Next text config)

**Tags:**
- MEASURED: on the Kaggle 4×L4, unless the hardware column says otherwise.
- SOURCE: read in code.
- DERIVED: arithmetic from measured or config values.
- GUESS: an unmeasured estimate.

**Hardware facts used throughout** (MEASURED, `[E14R]nccl_bench.json` topology; `[E23S]SMOKE_A/p5_bench.json` env):
- 4× L4 (sm_89, 58 SMs, 22.03 GiB usable);
- PCIe link: gen max 3, x16, all pairs PHB, one NUMA node;
- driver 580.178.04, NCCL 2.29.7, CUDA 13.0;
- 48 vCPU Xeon @ 2.2 GHz;
- /dev/shm 87.5 GB.

**Flash-Next shapes used in §5** (SOURCE, `[CFG]`):
- hidden 2560, 48 layers: 36 GDN + 12 QSA, `full_attention_interval` 4;
- MoE: 512 experts, top-10, `moe_intermediate_size` 640, shared expert 640;
- GDN: 16 K heads + 48 V heads × 128, `mamba_ssm_dtype` float32;
- QSA: 24 heads, 2 KV heads, head_dim 256, indexer budget 2048, compress ratio 4, indexer 4 heads × 128 with 1 KV head;
- hyper-connections: `hc_count` 4, `hc_lowrank` 320;
- vocab 248,320;
- PLE: `ple_layer_ids [2]`, n-gram size 3, 8 heads per n-gram.

---

## 1. Measured kernel / communication numbers

### 1.1 Collectives and interconnect

| Item | Value | Hardware / config | Evidence |
|---|---|---|---|
| NCCL all-reduce, 983,040 B, inside a CUDA graph (140 calls/graph) | **343.88 µs/call** (p90 344.2) | 4×L4, world 4, NCCL SHM, 2 channels, ring | `[E14R]nccl_bench.json` `arms.baseline.graph_983040.per_call_us`; `summary.best_arm` = baseline |
| NCCL all-reduce, eager, world 4 | 8 KiB **90.1 µs**; 80 KiB **107.0**; 256 KiB 183.9; 983,040 B 382.0; 2 MiB 721.9; 64 MiB 20,538 µs (busbw 4.90 GB/s) | same | `[E14R]nccl_bench.json` `arms.baseline.eager.*.median_us`, `busbw_GBps` |
| NCCL all-gather, 11.9 MB per rank | **8.07 ms** eager / 8.02 ms graph (busbw 4.43-4.46 GB/s) | same | `arms.baseline.allgather_11_9MB` |
| NCCL P2P/IPC transport (`NCCL_P2P_LEVEL=SYS`) | 983,040 B graph **1,880.5 µs (5.47× slower)**; 64 MiB busbw 0.42 GB/s; all-gather 80.5 ms (10×). `can_access_peer` True on all pairs, `nvidia-smi topo -p2p r` OK | same | `arms.p2p_sys.*`; `topology.p2p_matrix` |
| `NCCL_SHM_USE_CUDA_MEMCPY=1` | eager 983 KB 502.8 µs (slower); **graph capture timed out (240 s, rc −9)** | same | `arms.shm_memcpy.status` = timeout |
| NCCL all-reduce (P5 bench, own ctypes NCCL), graph | 65,536 B **49.7 µs**; 245,760 B 134.3; 491,520 B 193.5; 983,040 B **336.2**; 1.97 MB 634.2; 3.93 MB 1,235 µs | world 4 | `[E23S]SMOKE_A/p5_bench.json` `groups.main_ab.rank0.progress` ("b: … nccl_graph") |
| NCCL CTA knobs (MIN/MAX_CTAS 4/8/16/32), 983 KB graph | 351.8 / 424.1 / 428.8 / 423.4 µs, **all slower than default 336.2** | world 4 | `p5_bench.json` `summary.nccl_tuned_983040_us` |
| Host-staged two-shot all-reduce (P5, native fp32-accumulate), graph | 65,536 B **34.9 µs (1.42× faster than NCCL)**; 245,760 B **99.9 (1.34×)**; 491,520 B 188.2 (1.03×); 983,040 B 375.5 / best 361.7 at 12 blocks (**0.93×**); 1.97 MB 727.7; 3.93 MB 1,416 | world 4, pinned host buffer, 512 threads | `p5_bench.json` `progress` "b: host …", `summary.host_graph_best_us_983040`, `speedup_vs_best_nccl` 0.93 |
| Host-staged INT8 payload (absmax / 32 groups), graph | 65,536 B **27.8 µs**; 245,760 B 67.6; 491,520 B 122.7; 983,040 B **237.7 (1.41× vs NCCL)**; 1.97 MB 452.6; 3.93 MB 876.6 | world 4 | `p5_bench.json` `groups.main_cd.rank0.progress` "d: latency …"; `summary.int8_graph_us_983040` |
| P5 correctness | bit-exact vs ordered fp32 ref, identical on all ranks, graph replays bit-exact, loud error path. The INT8 "CHECK_FAILED" is a float `== 1.0` artefact (worst_err_over_bound 0.959 / 0.941 < 1) | world 4 | `p5_bench.json` `summary.checks`; `[E23S]SMOKE_REV3G_RESULTS_2026-10-30.md:56-60` |
| FlashInfer 0.7 PCIe-IPC all-reduce in an engine | vLLM FP8 MTP3: **0.68×** (297 / 278 / 206 vs 434 / 399 / 281 tok/s); SGLang DFlash W6: **0.39×**; SGLang MTP: 0.42× | 27B TP4 | `[E22F]:16,40`; `EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md:24,27,51,66` |
| PCIe-IPC init cost | JIT `pcie_ipc_comm` build plus autotune of 19 profiles, 3 min 26 s | 27B TP4 | `[VMS]vllm_server_fp8_mtp3_pcie.log:283-305` |
| All-reduce backend actually used by vLLM 0.32 | `['PYNCCL']` only. SymmMem: "Device capability 8.9 not supported". FlashInfer AR: "not supported for world_size=4". Custom AR disabled. The same holds for the `ep:0` group | 27B TP4; Flash-Next RC1 TP4+EP | `[E23S]SMOKE_B/logs/server_fp8_mtp3.log:43-48`; `_scratch/e13_rc1_v1/q27x/vllm_server_e13_vllm_fn.log:26-35` |
| Communication share of a decode step, SGLang | all-reduce 50.9 ms (140 calls) + all-gather 8.1 ms of 116 ms = **51 %** | 27B INT4 DFlash2 TP4, 20-step profile | `[E14F]:26-41`; `[E22F]:46` |
| NCCL share of kernel time, vLLM | **35.7 %** (G0 PIECEWISE) and **35.6 %** (G3 FULL_AND_PIECEWISE). The all-reduce kernel `AllReduce_Sum_bf16_RING_LL` is 27.9 % and the all-gather `AllGather_RING_LL` 7.8 % of the summed kernel time | 27B FP8 MTP3 TP4, 40k@16 | `[E23S]SMOKE_B/e23_results.json` `sections.6_mtp_depth.mtp3_eq_G0_piecewise.profile.trace.{by_category_pct,top_kernels_us}` (941,160 + 264,692 of 3,379,115 µs); `sections.3_full_graph.G3_flashinfer_pr60532.profile.trace` |
| PP hand-off + host launch (non-compute time per step) | **2.0 ms/step**. Stage 0 7.0 ms, stage 1 11.0 ms; the stages run serially at 1 stream | **4×3090 with P2P**, Flash-Next v2.5.1 TP2×PP2, single stream, 131K | `[MS]models/flash-next/benchmarks/2026-10-09/BENCH-CARD.md:39-42` |
| Merlin custom all-reduce, Flash-Next stage 1 | 0.6 ms/step | 4×3090 with P2P, single stream | same, `:41-42` |

### 1.2 GEMM / MoE / attention / GDN kernels

| Item | Value | Hardware / config | Evidence |
|---|---|---|---|
| Marlin W8A16 (FP8 weight-only), `gate_up_proj` 8704×5120 | graph median M=1 **176.1 µs**, M=16 177.5, M=80 249.3; weight stream **253 / 251 / 179 GB/s** | 1× L4, real 27B FP8 weights | `[E23S]SMOKE_A/p4_bench.json` `shapes.gate_up_proj.timing.{1,16,80}.marlin_w8a16` |
| CUTLASS block-FP8 W8A8 (PR 61224 .so) + act quant, same shape | M=1 193.3, M=16 196.9, **M=80 247.2 µs** | same | same, `cutlass_w8a8_quant_gemm` |
| BF16 cuBLAS, same shape | M=1 343.6 µs (259 GB/s), M=80 403.3 | same | same, `bf16_cublas` |
| `down_proj` 5120×4352: Marlin vs CUTLASS+quant | M=1 89.7 vs 98.7; M=16 90.3 vs 102.4; **M=80 130.3 vs 111.1** | same | `shapes.down_proj.timing` |
| MLP time per forward (gate_up + down only) | M=1: Marlin 17,011 vs CUTLASS 18,690 µs (**Marlin 9 % faster**); M=16: 17,136 vs 19,153 (11 %); **M=80: CUTLASS 6 % faster** (22,932 vs 24,294); BF16 33,101-39,030 | same | `p4_bench.json` `summary_per_m`; `SMOKE_REV3G_RESULTS:52-55` |
| Activation quant alone | 3.2-5.4 µs in a graph (57-61 µs eager) | same | `p4_bench.json` `quant_only` |
| Kernel kinds selected in the real engine | vLLM FP8 → `MarlinFP8ScaledMMLinearKernel` (W8A16). INT4 → `MarlinLinearKernel`. Flash-Next MoE → `'MARLIN' WNA16 MoE backend` / `MarlinExperts`. CUTLASS block-FP8 is SM90+ only | 27B and Flash-Next | `[E22F]:53-54`; `[E14F]:123-127`; RC1 log `:41,88` |
| Triton W8A8 (`--linear-backend triton`) vs Marlin | 438 / 398 / 283 vs 434 / 399 / 281 tok/s (equal) | 27B FP8 MTP3 TP4 16 streams | `EMS_SAFE_RISKY_VMS…md:50` |
| SGLang step composition | Marlin INT4 GEMM 25.3 ms (22 %), other GEMM 14.5, GDN (Triton) 8.0, attention 2.7 of 116 ms | 27B TP4 DFlash2 | `[E14F]:26-37` |
| SGLang step model | step ≈ 26 ms + 0.94 ms per verify row. Weight reading is only ~16-22 ms of a 107-125 ms step | 27B TP4 | `[E22F]:50` |
| vLLM kernel-time split | NCCL 35.7 %, Marlin 27.0 %, attention (FlashInfer `BatchPrefillWithPagedKVCacheKernel` used for verify) 19.8 %, other GEMM 9.0 %, sampling 0.7 %, other 7.6 % | 27B FP8 MTP3, 40k@16 | `e23_results.json` (as above) |
| GDN share, corrected | The trace classifier puts GDN at 0.2 %. Its top kernel `gdn_decode_post_conv_mtp_kernel<float,3>` is 206,180 µs of 3,379,115 = **6.1 %** (G3: 220,521 / 3,627,841 = 6.1 %), filed under "other". Classifier bug | same | `top_kernels_us` in both traces |
| GDN spec-verify kernel path | fused CUDA by default. With `--mamba-ssm-cache-dtype float16` it falls back to Triton ("requires … BF16 or FP32 recurrent state") | 27B vLLM 0.32 | `[VMS]vllm_server_fp8_mtp3_ssm_fp16.log:61-62`; `[E22F]:55` |
| Flash-Next per-kernel split, stage 1 | dense GEMMs **3.9 ms**; BF16 vocab head **2.25 ms (5 calls/step)**; Marlin MoE **1.3 ms**; custom AR 0.6; QSA attention 0.66-0.89; GDN 0.25-0.35. Depth 4K→258K adds only ~0.6 ms | **4×3090**, P2P, v2.5.1, single stream | `BENCH-CARD.md 2026-10-09:39-42` |
| Flash-Next 3090 throughput anchors | single stream 186-206 tok/s (v3.0.0 gate). 8 agents 598-638. Six 126,976-token sessions **574.5 tok/s = 10.78 ms/token per agent**. No-MTP single stream ~82 tok/s | 4×3090 P2P | `[MS]models/flash-next/CHANGELOG.md:12-13,44,227`; BENCH-CARD 2026-10-09:47 |
| DRAFT_HEAD int8 | **6.5-7.2 % lower ms/output token** than BF16, same acceptance | 4×3090 | `CHANGELOG.md:10` |
| Merlin guard reader (`VLLM_E2_GUARD_MODE=warn`) | ~**0.3 ms per decode step** on TP2×PP2 (`off` skips it) | 4×3090 | `CHANGELOG.md:204` |
| PLE fast-ids + early-request switches | 0.11-0.16 ms per output token saved | 4×3090 | `CHANGELOG.md:67` |
| QSA indexer top-k timings we hold (SGLang `fast_topk` / FlashInfer ragged) | 0.9-6.4 ms "per call", 4-10 rows, 300-34,816 columns, K=512. **Not kernel time**: the loop includes a per-repeat CPU correctness check | 1× L4 | `_scratch/e13_rc1_v1/q27x/e13_topk.json`; `mp_topk.py:88-107` |

### 1.3 CUDA graphs, compile and JIT

| Item | Value | Config | Evidence |
|---|---|---|---|
| FULL_AND_PIECEWISE downgrade | "FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE" | every vLLM 0.32 27B MTP arm, including Merlin TP4 | `[VMS]vllm_server_fp8_mtp3_pcie.log:226-232`; `[E23S]SMOKE_A/logs/server_merlin_tp4.log:206-209` |
| Merlin 27B drafter graph mode | `role=draft cudagraph_mode=prefill:PIECEWISE,decode:NONE` (draft decode eager) | Merlin TP4 27B | `server_merlin_tp4.log:211-214` |
| RecoverSSM commit graph | "[recoverssm-commit] mode=graph: no graph for this step (n=24, 0 captured); running staged-eager" | Merlin 27B (TP4: n=24; TP1: n=4) | `server_merlin_tp4.log:255-258`; `server_merlin_tp1.log:114` |
| PR 60532 backport (G3, FlashInfer FULL graphs with multi-token verify) | **448.9 vs 435.7 tok/s (+3.0 %)** at 40k@16. Graph launches 1,809 → 116, kernel launches 7,155 → 1,566, eager kernel share 25.5 % → 5.2 %. Attention and GDN go from eager to graph-launched. NCCL share unchanged (35.6 %). Graph proof INCONCLUSIVE (0 forward annotations; the fix is to match `execute_context_*` scopes). No A/A repeat | 27B FP8 MTP3 TP4 | `SMOKE_REV3G_RESULTS:16-17,37-48`; `REV3G_SMOKE_GPT2_PRIORITY_FIX_LIST_2026-10-30.md:14` |
| Run-to-run spread for scale | SGLang reference 572-583 / 521-544 tok/s across two runs (±1-2 %) | 27B | `EMS_SAFE_RISKY_VMS…md:18-20` |
| PIECEWISE vs eager | ≈ 1.01× on L4 (vLLM #59548, upstream claim, not ours) | — | `[E22F]:56` |
| SGLang CUDA graphs 1..13 vs default | 500.1 vs 505.0 tok/s (neutral) | 27B DFlash2 | `[E14F]:23` |
| torch.compile | backbone 52-55 s per boot; eagle head ~2 s | 27B TP4 | `server_fp8_mtp3.log:167-182`; `server_merlin_tp4.log:144-180` |
| FlashInfer JIT, cold | ~150 s (prefill kernel 05:28:53, sampling module 05:29:41, done ≈05:31:23) | Merlin TP1 27B | `server_merlin_tp1.log:112-114` |
| FlashInfer sampling JIT failure | `ld: cannot find -lcuda`, engine died. Fixed in RC2 by a `LIBRARY_PATH` shim | Flash-Next RC1, stock vLLM | `E13_RC1_GPU_run1_2026-10-31.md:22-25`; `build_e13_rc2.py:426,453,466` |
| PP4 | stock vLLM 0.32 + RedHat INT4, eager: 2k EXACT, **8k `CUDA error: an illegal memory access`** (NCCL watchdog, rank 0, no Xid; #63505 class). Compiled PP4 skipped | 27B | `SMOKE_REV3G_RESULTS:62-66` |
| TileLang QSA indexer (SGLang Flash-Next) | "Data race detected: `Logits(bx, position)` …" during decode-graph capture; capture took 420 s | SGLang 0.5.21 | `E13_LOADPROBE_GPU_run1_2026-10-27.md:12,47` |

### 1.4 E12 FP8 kernel tuning (2026-10-16)

- **Never run on the GPU.** The folder has no Experiment Results.
- **Its premise is obsolete on vLLM 0.32.** In vLLM 0.19, block-FP8 used the Triton w8a8 kernel with the default config (BLOCK 64/128/128, 4 warps, 2 stages) for 5 shapes. In 0.32 the automatic choice on sm_89 is Marlin W8A16 (`[E14F]:123-127`). `VLLM_TEST_FORCE_FP8_MARLIN` is gone (vLLM #52182, `[E14F]:155`).
- **What remains useful:** the method. Tune a production Triton kernel's config table per shape on L4, gate each config on correctness against the default, and ship the JSON. The same method applies directly to the Flash-Next QSA table (§4).

---

## 2. GOOD

1. **NCCL SHM baseline is the best collective path on this VM for ≥ 0.5 MB, and the environment is fully characterised.**
   - Every alternative tried is slower or broken: P2P 5.5×; CE memcpy slower and hangs in capture; CTA knobs; PCIe-IPC 0.39-0.68× in engines.
   - So the comm path for Flash-Next starts from the known-best option and the profile default `P2P=off` (`[FNP]:903-915`).
2. **Host-staged all-reduce (P5) is correct, graph-capturable and *faster than NCCL at small messages*.**
   - It gains 1.42× at 64 KB and 1.34× at 240 KB (world 4). The INT8 variant is 1.8-2.0× at those sizes.
   - The verdict "0.93×, not a path" is only true at 983 KB, the 27B TP4 decode size.
   - Flash-Next TP2 messages are much smaller: about 80-160 KB (§5). The kernel already supports worlds 2/4/8 (`p5_bench.json` rank0 `info`).
   - This makes P5 a live candidate for Flash-Next, unlike for 27B.
3. **The L4 weight-streaming ceiling is known.** Marlin W8A16 and cuBLAS BF16 reach 250-259 GB/s at M ≤ 16, 84-86 % of the 300 GB/s peak (P4).
   - Marlin stays flat from M=1 to 16. That gives a reliable bytes→ms conversion for the §5 model.
4. **Marlin W8A16 / W4A16 is the right decode kernel at M ≤ 16** (9-11 % faster than CUTLASS W8A8 + quant). CUTLASS only wins at M=80 (6 %).
   - Flash-Next per-stage micro-batches are 8-32 tokens, so the PR 61224 Ada CUTLASS path is a prefill-only lever. It is not needed for decode.
5. **Flash-Next on Merlin avoids the FlashInfer graph downgrade, by construction** (SOURCE).
   - The QSA backend declares `AttentionCGSupport.UNIFORM_BATCH` (`[MS]vllm/models/qwen4_exp/nvidia/qsa.py:320`, `common/qsa_cache.py:652`), as do GDN (`[MS]vllm/v1/attention/backends/gdn_attn.py:98`) and GDN-RecoverSSM (`gdn_recoverssm.py:208`).
   - Merlin always passes `FULL_AND_PIECEWISE` (`[MS]tools/merlin/launch.py:459-460`). So the 27B downgrade (FlashInfer UNIFORM_SINGLE_TOKEN_DECODE) should not apply to Flash-Next.
   - To be confirmed by the RC2 graph-mode log lines (`build_e13_rc2.py:733`).
6. **PP2 interleaves two micro-batches.** In V2 + PP + async, a request is decoded only every `pp_size` steps (`[MS]vllm/v1/core/sched/scheduler.py:792-796`), and `max_concurrent_batches` = pp_size + 1 (`[MS]vllm/config/vllm.py:607-617`).
   - With N running streams, each stage processes about N/2 requests per tick while the other stage works on the other half.
   - The 3090 single-stream "stages run serially" profile is therefore not the multi-stream picture.
7. **Flash-Next communication volume is small compared with 27B TP4.**
   - The all-reduce payload is the 2,560-wide block output: 5,120 B per token in bf16 (`[MS]…/nvidia/model.py:324-335,369-410`). That is half of 27B's 10,240.
   - It runs at world 2, not 4.
   - EP inside TP2 at dp=1 uses no all-to-all, only the TP all-reduce (`[MS]vllm/model_executor/layers/fused_moe/config.py:1026-1029`). So there are 2 all-reduces per layer.
8. **E14's measurement method is reusable.** The NCCL bench (`e14_nccl_bench.py`, graph-captured 140-call replay, per-rank medians, NCCL transport parse) and the trace summariser can be pointed at TP2 / 80-160 KB with no new code beyond arguments.
9. **The capture-size semantics are now settled from source (PLAN_1500 §5 open question 3).** See §6 Q1: the sizes count tokens, and with K3 they are rounded to multiples of 4.

## 3. BAD

1. **P2P "works" but is unusable for bandwidth.** NCCL P2P/IPC gives 0.42-0.78 GB/s busbw.
   - The E13 synthesis lists P2P availability as UNRESOLVED (`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:85-96`), but the E14 bench resolves it: functional, 5.5-10× slower.
   - Merlin's interconnect mode `on` exports `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` + `NCCL_P2P_LEVEL=SYS` + `NCCL_PROTO=Simple` (`[FNP]:891-900`). Those are exactly the two paths measured slower: PCIe-IPC 0.68× in vLLM, P2P/IPC 5.5×.
   - The RC2 optional arm A7 (P2P on) is therefore expected to be negative. Keep the default `off`.
2. **sm_89 is outside every arch-tuned table in `qwen4_exp`** (§4): no QSA tuning table, no low-latency skinny-GEMM plans, no fused HC down+SiLU, no cooperative top-k, and no persisted Triton autotune for GDN (WARM=0).
   - Every Flash-Next kernel on L4 runs a default or generic path.
3. **FlashInfer + spec decode forces PIECEWISE on 27B.** PIECEWISE ≈ eager.
   - The Merlin drafter ran decode with **no graph** (`decode:NONE`), and the RecoverSSM commit ran **staged-eager**, with 0 graphs captured. If the same holds for Flash-Next, three MTP draft passes per step run eager, each with QSA + MoE + HC + draft head.
   - This is unverified for Flash-Next.
4. **FULL graphs gave only +3 % (one run, no A/A).** At 27B TP4 the step is comm- and bandwidth-bound, not launch-bound. Graph work is not the 1500 lever on its own.
5. **The comm share does not move with graphs.** NCCL stays at 35.6-35.7 % (vLLM) to 51 % (SGLang) of the 27B step whatever the graph mode.
6. **PP4 crashes on stock vLLM 0.32** (illegal memory access at 8k).
   - Merlin carries the PP fixes #54709, #54793/#63617 and #54795. No reference to #63505 is in the Merlin docs or tree (grep empty).
   - So PP2 long-prefill stability is still an RC2 gate.
7. **The trace classifier has two blind spots.**
   - Forward annotations: 0 found, because it matches the wrong scope names.
   - GDN kernels are filed under "other" (6.1 % real vs 0.2 % reported).
   - Any Flash-Next trace summary needs the fixed classifier: `execute_context_*` scopes, plus `gdn_*`, `qsa_*`, `marlin_moe` and `persistent_topk` patterns.
8. **We have no valid top-k kernel timing.** The E13 top-k numbers include a CPU check per repeat.
9. **No TP2 (world 2) NCCL or P5 measurement exists on Kaggle L4.** Every number above is world 4.
10. **FlashInfer JIT is a boot-time and environment hazard:** ~150 s cold, and the `-lcuda` link failure in RC1. Merlin turns FlashInfer autotune off (`launch.py:465`). The sampler still JITs.

---

## 4. sm_89 fallback inventory for `qwen4_exp` (vLLM 0.32 stock and Merlin v3.0.0)

| # | Component | Gate / fallback (path:line) | What sm_89 gets | Can a tuned config be added? |
|---|---|---|---|---|
| F1 | **QSA sparse paged GQA attention** (Triton split-K) | Stock: `[ST]models/qwen4_exp/nvidia/ops/qsa.py:19-28` (`_is_sm120`, `_is_sm90`), `:546-590` (`_select_config`; the default table is tuned on GB300). Merlin: `[MS]vllm/models/qwen4_exp/common/qsa_config.py:5-8` (`QSA_TUNED_CAPABILITIES = (90, 120)`), `:83-134` (`select_qsa_sparse_config`), `[MS]…/nvidia/ops/qsa.py:39-51` (`qsa_tuning_capability()` → None on 86/89), used at `:709,:822` | The GB300 default table, keyed on base_programs = rows × kv_heads. At TP2, kv heads per rank = 1, so a 16-32-token micro-batch gives bp 16-32 → (BLOCK_N 32, splits 64 or 16, warps 4 or 1). It ignores `is_fp8` | **Yes, data-only.** Add 89 to `QSA_TUNED_CAPABILITIES` and a `_select_sm89_config(base_programs, use_prefill_config, is_fp8)` branch. Warmup enumerates through the same selector (`qsa_config.py:137-175`), so warmup and dispatch stay consistent. The tuning sweep goes over (block_n ∈ {32,64,128}, splits, warps) per bp bucket, FP8 and BF16, on the L4's 58 SMs. Method: the E12-K1 tuner pattern (correctness-gated). Expected gain: within QSA's ~2-5 ms/tick estimate (§5); not a top lever |
| F2 | **QSA E4M3 KV read** | Merlin `…/nvidia/ops/qsa.py:25-36` (`qsa_e4m3_uses_native_cast()` true at ≥ 89); `…/nvidia/qsa.py:229-239` (E4M3 admitted only on 86 or ≥ 89) | Upstream native fp8e4nv cast (vllm#55557). Not a fallback, but the path is untested on any Ada card (`[FNP]:830-945` status "untested") | n/a. This is a correctness gate (RC2 A3) |
| F3 | **QSA indexer top-k** | Stock `[ST]…/nvidia/ops/qsa_indexer.py:477-496`; Merlin `[MS]…/nvidia/ops/qsa_indexer.py:628-648`: `cooperative_topk` only if rows ≤ 64, stride%4, **`has_device_capability(90)`**, not sm_120 | `torch.ops._C.persistent_topk` (stock) / `merlin_persistent_topk` (Merlin), block_topk = 2048/4 = 512 of L/4 compressed blocks | **Not a config.** The gate is a capability check on a C++ op. Making the cooperative kernel run on sm_89 would need a code port (it likely relies on Hopper cluster or cooperative features; UNVERIFIED). First measure persistent_topk kernel time in the RC2 trace. Merlin's `kernel_sorted_emit()` skips the Python sort on this path (`:650-655`) |
| F4 | **Low-latency skinny GEMM (CuTe DSL)** for unquantized BF16 linears | Stock and Merlin (identical files) `…/nvidia/low_latency_gemm.py:294-300`: plans only for (10,3), (10,0), (9,0), (12,1). `:347-357`: no plan → `torch.nn.functional.linear` (cuBLAS). `:376-385`: `enable_…` returns early with no plans. Called from `model.py:931`, `mtp.py:607` | cuBLAS for the QSA fused QKV/gate, QSA o_proj, shared-expert gate/up/down, router (512×2560), HC up (10240×320) and LM head (124,160×2560 at TP2). The INT8 GDN in/out projections are Marlin W8A16 and outside this table | **Yes, data-only:** add `(8, 9): QWEN4_EXP_SM89_GEMM_PLANS`. Seed it from the **SM121 (DGX Spark GB10) table, which already lists the TP2 local shapes** (`:171-292`: (6656,2560) QSA QKV TP2, (2560,3072) out-proj TP2, (8192,2560) GDN QKVZ TP2, (124160,2560) LM head TP2, (48,2560), (640,2560), (512,2560), (10240,320)). GB10's ~273 GB/s is close to the L4's 300 GB/s. Then retune per M ∈ capture sizes. Preconditions: `shape_dynamic_skinny_gemm.is_available()` only checks the `cutlass` import (`[MS]vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py:50-62`); PDL is off on sm_89 (`:113-115`). Whether the CuTe DSL kernel compiles for sm_89 is UNVERIFIED (it is a dot-product kernel with no wgmma). A potentially large lever: dense GEMMs are 3.9 ms of an 11 ms stage on 3090 |
| F5 | **Fused HC down-projection + SiLU** | Merlin `…/nvidia/hyperconnection.py:112-116` (`_use_hc_down_silu` needs `has_device_capability(90)`); `…/nvidia/ops/cute_dsl/hc_down_silu.py:212` (warmup gate). The MMA variant imports `cutlass.pipeline.sm90` and `setmaxregister` (`_hc_down_silu_mma.py:19,313,376`). The FMA variant uses PDL `griddepcontrol` (`_hc_down_silu_fma.py:254,334`). Stock: `[ST]…/nvidia/hyperconnection.py:112-116` | Unfused `MergedColumnParallelLinear` (10240 → 320+4+12 pad) via cuBLAS, plus a separate SiLU. 2 HC modules per layer, **replicated weights** (`disable_tp=True`) | **Code change, not a config.** Port the FMA variant without PDL for sm_89 (M ≤ 48 = `MAX_FUSED_M`, `hc_down_silu.py:27`), or cover the unfused GEMM with an F4 plan entry for (336, 10240). Related: HC weights (~27 MB per layer, DERIVED) are read in full on both TP ranks |
| F6 | **GDN prefill / spec-verify Triton autotune** | `[FNP]:940` ("WARM defaults to 0 because the persisted Triton autotune is sm_86 only"); `[MS]docs/merlin/hardware-matrix.md:216` (synthesis R17) | Live autotune on the first boot. It can pick slow configs, and winners can differ by cache (CHANGELOG:64-66: the Ampere fixed-config fix applies to sm_86) | **Yes, cache-level.** After the first L4 boot, keep the Triton cache (shared `TRITON_CACHE_DIR`, not per port) and ship it as a warm-cache dataset. Better: pin one fixed config per FLA kernel for sm_89, as Merlin did for Ampere. Needs a determinism check |
| F7 | **GDN spec verify, fused CUDA kernel** | vLLM 0.32: fused CUDA only with BF16/FP32 state (`[VMS]…ssm_fp16.log:61`). Merlin RecoverSSM: "speculative verify runs the Triton GDN decode kernel and replays the state each step" (`[MS]models/flash-next/CHANGELOG.md:207`) | Triton GDN decode + RecoverSSM replay; commit graph possibly staged-eager (§1.3) | Config-level only for the state dtype (`--mamba-ssm-cache-dtype`, PLAN L5). The commit graph capture is a code question (`recoverssm_commit_graph.py`) |
| F8 | **All-reduce backends** | SymmMem: "Device capability 8.9 not supported"; FlashInfer AR: not supported at world 4 (and 2, per `[E22F]:57`); custom AR disabled; QuickReduce / AITER are ROCm (logs §1.1) | PYNCCL over SHM (P2P off) | No vLLM config. The lever is the P5 host-staged kernel (worlds 2/4/8, graph-capturable) integrated as a custom all-reduce for the tp group |
| F9 | **Block-FP8 CUTLASS linear** | `kernels/linear/__init__.py:478-489`, `cutlass_scaled_mm_entry.cu:161-174` (SM90+) (`[E14F]:124`) | Marlin W8A16 | Not relevant to Flash-Next decode (no FP8 linears; the PLE FP8 table is a gather). PR 61224 helps only at M ≥ 80 (P4) |
| F10 | **Marlin MoE (W4A16 experts)** | Merlin `csrc/moe/marlin_moe_wna16/ops.cu:41-65` (`MERLIN_FULL_K`, PR #46639 CUDA half). No sglang#47188 one-block-per-SM rule (`ops.cu:262-277`, synthesis R5) | The generic Marlin MoE heuristic tile choice. No sm_89 tuning exists anywhere | **Unknown, worth a microbench.** This is the single biggest decode cost (§5). A P4-style bench of Merlin's Marlin MoE on real Flash-Next experts at 1-2 tokens per expert (M_per_expert) and 40-250 active experts per layer, on one L4, would give the real GB/s and show whether tile, split or one-block-per-SM changes pay |
| F11 | **Graph capture sizes under MTP** | `[MS]vllm/v1/worker/gpu/cudagraph_utils.py:268-347` (MRV2: sizes = num_tokens; uniform-decode graphs rounded to multiples of `decode_query_len`, capped at `max_num_reqs × decode_query_len` and the max capture size); V1 runner `config/compilation.py:1503-1515,1543-1571`; `gpu_model_runner.py:877` (`uniform_decode_query_len = 1 + num_spec_tokens`) | Profile capture `[1,2,3,4,6,8,12,16,24,32]` (`[FNP]:846-858`) → with K3, uniform graphs at 4/8/12/16/24/32 tokens = 1/2/3/4/6/8 requests. Varlen decode graphs are also captured for token counts ≤ max_decode_tokens | Config: extend `cudagraph_capture_sizes` (multiples of 4) when raising `--max-num-seqs`. Under PP2 a micro-batch carries about N/2 requests, so seqs 16 still fits a 32-token capture *if* the split is even (§6 Q2) |

---

## 5. Estimated decode-step cost for Flash-Next on TP2×PP2+EP (Merlin, 4×L4, MTP K3)

### 5.1 Model of a step

**Layout** (`[FNP]:843-850`):
- stage 0 = GPUs 0-1, 25 layers, plus the PLE layer and the host embedding;
- stage 1 = GPUs 2-3, 23 layers, plus the final HC mixer, the LM head, 3 MTP draft passes, sampling and the logits gather.

**Tick and cycle:**
- With N running streams, each stage processes a micro-batch of about N/2 requests, i.e. T = 4·N/2 = 2N verify tokens at K3 (GOOD 6).
- One "tick" = max(stage 0, stage 1) + hand-off.
- A request advances once per 2 ticks, so t_cycle = 2 × tick, and aggregate tok/s = N × a / t_cycle (PLAN_1500 §1).

**Bandwidth:** about 250 GB/s per L4 for weight streaming (P4 measured for Marlin and cuBLAS at M ≤ 16). The MoE kernel's own efficiency is unmeasured (F10).

**Expert bytes** (DERIVED):
- One expert (gate_up 2×640×2560 + down 640×2560 = 4.92 M params, INT4 g128 + bf16 scales) ≈ **2.53 MB**, so 1.30 GB per layer. Each GPU holds 256 experts of its stage's layers (EP2).
- Distinct experts touched per layer by T tokens, with independent routing (upper bound): D(T) = 512·(1 − (1 − 10/512)^T). That gives D(4)=39, D(8)=75, D(16)=139, D(24)=193, D(32)=240, D(64)=367, D(128)=471.
- Lower bound: all K+1 tokens of a request route alike, so D_low = D(requests).
- Cross-check against the 3090 single stream (4 tokens): 1.3 ms of Marlin MoE on stage 1 implies 0.29-1.13 GB at 220-870 GB/s. The true locality is between the two bounds.

### 5.2 Cost table per tick, N = 8 streams (micro-batch ≈ 4 requests, T = 16)

| Component | Stage | Bytes / count per GPU per tick | Est. ms per tick | Basis |
|---|---|---|---|---|
| **MoE expert streaming** (Marlin W4A16, 10 of 512) | both | 24 layers × D/2 × 2.53 MB = **1.18 GB (D_low = 39) to 4.21 GB (D = 139)** | **4.7-21** | DERIVED; P4 bandwidth |
| **Dense GEMMs** (INT8 GDN in/out Marlin W8A16 + BF16 QSA QKV/o + shared expert + router + **replicated HC ~27 MB/layer**) | both | ≈ 1.65 GB (GDN 0.52 + QSA 0.31 + shared 0.12 + router 0.06 + HC 0.64) | **6.6-12** | Lower: bytes at 250 GB/s. Upper: the 3090 measured 3.9 ms × 3.1 (bandwidth ratio; cuBLAS at small M on a card with no skinny-GEMM plans, F4) |
| **Vocab heads** (target LM head + 3 draft heads, TP2 shard 124,160×2560) | 1 | BF16: 4 × 636 MB = 2.54 GB; `DRAFT_HEAD=int8`: 1.59 GB; int4: 1.11 GB | **10.2 / 6.4 / 4.5** | DERIVED. The 3090 card shows 2.25 ms for 5 calls, so v2.5.1 probably used a reduced or quantized draft vocabulary; 3.0.0 ships the full-vocab BF16 draft head (CHANGELOG:10) |
| **TP2 all-reduce** | both | 48 calls (24 layers × 2) + ~6 MTP calls on stage 1, each 16 × 5,120 B = **81,920 B** | **1.5-3.8** | World-4 graph: 49.7 µs @ 64 KB, 134 µs @ 240 KB; world 2 GUESS 0.5-0.75× of that, i.e. 30-70 µs per call |
| **Logits all-gather** (vocab-parallel) | 1 | target 16 × 124,160 × 2 B = 4.0 MB/rank + 3 drafts × 1.0 MB | **2-3** | 4.4 GB/s busbw (E14 all-gather) |
| **PP2 send/recv + host launch/hand-off** | between | hidden × hc_count = 10,240 bf16 = **20,480 B/token** → 328 KB per micro-batch (`[MS]…/nvidia/model.py:626-629,779-787`), plus sampled tokens back | **1-3** | 3090 P2P measured 2.0 ms/step of non-compute; SHM adds transfer latency (GUESS) |
| **QSA** (6 layers per stage: indexer score over L/4, persistent top-k 512 blocks, sparse 2048-token gather, FP8 KV) | both | gather ≈ 1 MB per query token per layer per rank (2048 × 1 KV head × 256 × 2 × 1 B); indexer 2-4.5 MB per request per layer at 70k | **2-5** | 3090: 0.66-0.89 ms × 3.1, growing with T; top-k unmeasured (F3) |
| **GDN recurrent + RecoverSSM** (18 layers per stage) | both | state 48×128×128×4 B / 2 = **1.57 MB per request per layer**, read + write → ~0.23 GB for 4 requests | **1-3** | 3090 0.25-0.35 ms × 3.1 plus state traffic; commit may be staged-eager |
| **PLE gather** (1 layer, stage 0) | 0 | (3−1)×8 = 16 heads × 160 B FP8 = **2,560 B/token** from pinned host + 5,120 B/token host embedding → ~120 KB per tick | **0.1-0.5** | DERIVED; cross-process `cuStreamWaitValue32` latency unmeasured (R15) |
| **Guard reader** | both | — | **0.3** | CHANGELOG:204 (`VLLM_E2_GUARD_MODE=off` removes it) |

**Results:**
- **Stage 1 tick (N=8):** ≈ 30-62 ms (midpoint ~42), so t_cycle ≈ 60-124 ms.
- **Stage 0** ≈ 17-45 ms. Stage 1 carries the heads, so stage 0 idles 5-15 ms per tick unless the 25/23 partition is retuned for L4.
- With a = 2.3-3.0 (Merlin xhigh prose 2.29, `[MS]docs/merlin/ENGINE-CHANGELOG.md:13-14` as cited in `MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:133`; Jon-Nielsen ~3.0 with FP8 KV), the **aggregate is ~150-400 tok/s at 8 streams**. This agrees with the synthesis's independent projection of 200-320 (`MERLIN_FN_L4_SYNTHESIS:226-230`). It is the baseline the RC2 ladder will replace with a measurement.

**N = 16** (micro-batch 8 requests, T = 32):
- MoE grows to 2.27-7.28 GB = **9-29 ms**;
- all-reduces become 163,840 B (2-4.5 ms);
- the rest is roughly flat;
- tick ≈ 36-75 ms, so 16 × 3 / (2 × tick) ≈ **320-670 tok/s**.

**The 1500 budget in these terms:**
- PLAN_1500 rows: C 8 / a 3 → t_cycle 16 ms (tick 8 ms); C 12 / a 4 → 32 ms (tick 16 ms).
- Both need the per-tick total cut by ~3-5× against the midpoint above, with the MoE term as the largest single piece.

**MoE bytes per verify token is the quantity to drive down:**
- It is 48 × D(T)/T × 2.53 MB: **1.21 GB at T=1, 0.91 GB at T=32, 0.70 GB at T=64, 0.45 GB at T=128** (independent routing).
- Across the 4 GPUs the aggregate weight bandwidth is ~1 TB/s, so the MoE-only ceiling is ≈ 1 TB/s × (a/(K+1)) / bytes-per-verify-token.
- Bigger micro-batches, higher a/(K+1), real routing locality and fewer bits per expert all raise that ceiling.
- **PP2 also has a structural cost:** two half-batches touch more distinct experts than one full batch. At N=16: 2 × D(32) = 480 vs D(64) = 367 expert reads per layer pair, i.e. +31 %.

### 5.3 Biggest optimisation targets (ranked by estimated ms per tick and leverage)

1. **MoE expert-weight streaming** (4.7-29 ms/tick; it scales with distinct experts).
   - (a) Measure the real D(T) per layer from routing logs in RC2, and Marlin MoE GB/s on L4 (F10 microbench).
   - (b) Raise tokens per expert read: more streams per micro-batch (KV levers L1/L5/L6/L9), and higher acceptance per verify token (MTP depth vs a, PLAN L3).
   - (c) Tune Marlin MoE tiles for 1-2 tokens per expert on sm_89.
   - (d) Re-weigh PP2 against layouts that put one larger batch through each expert. The cost is the +31 % above versus TP4's comm and its KV duplication (RC1: TP4 BF16 pool only 30,208 tokens).
   - (e) Lower-bit or reduced-expert checkpoints (PLAN §5 Q3).
2. **Dense BF16/INT8 GEMMs including the replicated HC** (6.6-12 ms/tick).
   - Add an sm_89 entry to `QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY`, seeded from the SM121 TP2 table (F4).
   - Port the fused HC down+SiLU (F5).
   - Consider TP-sharding the HC weights (~0.64 GB per stage-GPU per tick read twice).
3. **Vocab heads on stage 1** (4.5-10 ms/tick).
   - `DRAFT_HEAD=int8` (measured −6.5-7.2 % on 3090) or int4.
   - Then retune the 25/23 PP partition for L4 so stage 0 stops idling.
   - Sharded argmax/Gumbel sampling removes the logits all-gather (2-3 ms; `[E22F]:74`).
4. **Collectives** (all-reduce 1.5-4.5 + gather 2-3 + hand-off 1-3 ms).
   - Measure NCCL world-2 at 80-160 KB first.
   - Integrate P5 host-staged AR (1.3-1.4× at these sizes on world 4; INT8 1.8-2× behind a logprob gate) as the tp-group all-reduce.
   - Keep P2P off.
5. **Graph coverage on the draft and commit paths.**
   - Confirm FULL decode graphs at the actual micro-batch sizes, and that the drafter decode is graphed (27B Merlin showed `decode:NONE`).
   - Get the RecoverSSM commit graph captured (27B showed staged-eager).
   - Extend capture sizes in multiples of 4 when seqs go above 8.
6. **QSA** (2-5 ms): the sm_89 QSA table (F1); measure persistent top-k (F3).
7. **GDN / RecoverSSM** (1-3 ms): warm Triton cache / fixed configs (F6); state dtype (F7).
8. **Host overhead** (0.4-0.8 ms): `VLLM_E2_GUARD_MODE=off` (0.3 ms), and the PLE fast-ids switches (already on by default in 3.0.0).

---

## 6. Open questions

1. **Answered from source: capture sizes count tokens.**
   - MRV2 `cudagraph_utils.py:268-347` builds descriptors by `num_tokens`. Uniform-decode graphs are rounded to multiples of `decode_query_len` = 1 + K, and limited by `max_num_reqs × decode_query_len` and the max capture size.
   - V1 does the same rounding (`compilation.py:1503-1515,1543-1571`).
   - With K3 and capture ≤ 32, the uniform graphs cover 1, 2, 3, 4, 6 and 8 requests per forward.
   - To verify in the RC2 log: the captured list and cudagraph metrics.
2. Under PP2 + async, are micro-batches really ~N/2 each and stable (`scheduler.py:792-796`)? Or does admission or a finish skew them so one micro-batch exceeds the 32-token capture? Log per-step `num_reqs` per virtual engine.
3. Does Flash-Next on Merlin keep FULL decode graphs for the **target** (expected from the UNIFORM_BATCH backends), the **drafter** and the **RecoverSSM commit** on sm_89? Grep RC2 for `cudagraph_mode`, `role=draft` and `recoverssm-commit`.
4. What are NCCL world-2 all-reduce latencies at 40/80/160/320 KB under graphs on Kaggle L4, and do the two TP pairs' simultaneous SHM traffic interfere? Run the E14 bench with `world=2` on GPUs {0,1} and {2,3} at the same time. What is P5 at world 2?
5. What are the real per-layer distinct experts D(T) at T = 4-32 for ARC xhigh reasoning (routing locality between consecutive and draft tokens)? And Merlin Marlin MoE GB/s on L4? (F10 microbench on one GPU with real experts.)
6. What do the RC2 traces show as the per-stage tick composition on L4? The fixed trace classifier is needed (scopes, GDN/QSA/MoE patterns, `persistent_topk`). Is stage 1 the binding stage, by how much, and what partition balances it?
7. Does the CuTe DSL skinny GEMM (and an FMA `hc_down_silu` without PDL) compile and win on sm_89? Start from the SM121 TP2 plans.
8. How long does `persistent_topk` take (QSA indexer, block_topk 512) per micro-batch at 40k / 70k / 120k context? Does a tuned sm_89 QSA table move QSA time measurably?
9. Does the PLE cross-process transport add per-step latency on stage 0 in the Kaggle container (cuStreamWaitValue32 over host-mapped memory)? It needs a per-step timestamp in the trace.
10. Is the PP2 hand-off on L4 SHM close to the 3090's 2.0 ms, or larger? It needs a trace with both stages on one timeline.
11. Does the FlashInfer sampler JIT (with the `-lcuda` shim) add first-request latency, and should a non-FlashInfer sampler be used to remove the JIT and its environment risk?
12. Does the persisted Triton autotune from a first L4 boot reproduce the same GDN configs on later boots (determinism, as fixed for Ampere in CHANGELOG:64-66)?
