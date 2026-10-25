# Local validation of GPT notes 01, 03 and 04 on the 4x L4 serving bottleneck (logs and SGLang 0.5.21 source only)

Scope and conventions:
- Only local files were used. No GPU, no Kaggle, no web.
- **Verdicts:** TRUE / FALSE / PARTLY TRUE / UNVERIFIABLE (cannot be checked from local files).
- **"Measured"** means a number printed in a log. **"Derived"** means computed by me from logged numbers; every derived number names its formula.

**Short names for paths** (all files read completely; for decode lines, every line was parsed by script):
- **[E11-06 log]** `D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs/E11-06 P1-60K EVAL14 run1 (esfersami50, 2026-10-19)/q27x/sglang_server_try0.log` (1,831 lines; 1,299 TP0 decode lines).
- **[E11-05 log]** `.../Kaggle Runs/E11-05 EVAL30 run1 (esfersami50, 2026-10-18)/q27x/sglang_server_try0.log` (3,425 lines; 2,509 TP0 decode lines).
- **[E11-05c log]** `.../Kaggle Runs/E11-05 S1COMMIT4 run1 (foysal, 2026-10-18)/q27x/sglang_server_try0.log` (634 lines; 367 decode lines).
- **[E12 log]** `D:/ARC AGI 2/_scratch/rev6f_bundle_verify/E12_REV6F_IMPLEMENTATION_AUDIT_FOR_GPT_2026-10-22/runs/server_logs_12_streams/E12_EVAL30_rev1_sglang_server.log` (4,920 lines; 1,209 decode lines).
- **[G4B log]** `D:/ARC AGI 2/_scratch/g4b_bundle_verify/runs/rev6d_EVAL12R6G4B/sglang_server_try0.log` (1,191 lines; 446 decode lines).
- **[E11-03 dir]** `.../Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/` (`report.md`, `results.json`, `preflight_stdout.log`, `server_*_try0.log`).
- **[E11-04 dir]** `.../Kaggle Runs/E11-04 rev2b run1 (esfersami50, 2026-10-18)/e11_04/`.
- **[SRC]** `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/srt/` (SGLang 0.5.21 python source from `sglang-0.5.21-cp310-...whl`).
- **[CFG-T]** target config: `D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures/tok_int4/config.json`. Its md5 c55819bc… is identical to the Kaggle-input mirror `D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/run_badsha/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1/config.json`.
- **[CFG-D]** drafter config: `D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/run_badsha/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1/config.json`. Its `dflash_config`, layer types, `is_causal` and sliding window are identical to `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_config.json`.
- **Scripts:** `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/decode_stats.py` and `.../scratchpad/overall.py`.

Two source facts underpin many verdicts below. Both are from [SRC] `managers/scheduler_components/metrics_reporter.py`.

1. **What "accept len" and "gen throughput" mean.**
   - A decode log line is printed every `decode_log_interval` = 40 decode forwards; server_args show `'decode_log_interval': 40` in every log.
   - `gap_latency` is the wall time since the previous decode log (L898-899).
   - `last_gen_throughput = num_generated_tokens / gap_latency` (L900).
   - `accept len = spec_num_accept_tokens / spec_num_forward_ct` (L929).
   - `spec_num_forward_ct += bs` per forward (L501).
   - `num_accept_tokens` is the same `num_generated_tokens` (`batch_result_processor.py` L952-958).
   - Therefore, with a constant batch size B over the 40-forward window: **running × accept / gen_tok_s = gap_latency / 40 exactly**. The "inferred step time" is the window's wall time per decode forward, and it includes any prefill, scheduler or idle time inside the window.
2. **What "accept rate" counts.** `draft_per_round = speculative_num_draft_tokens − 1` = 7 (L932), and `accept_rate = correct_drafts / (7 × Σbs)`. So `accept len = 1 + 7 × accept rate`. The accept length includes the bonus/anchor token, and its maximum is 8 at width 8.

---

## Note 01: server configuration (E11-05 and E11-06 server_args)

### Takeaway
All six configuration claims are TRUE in both logs. The same args also appear in the E11-05 commit run, E12 EVAL30 rev 1 and G4B rev 6d.

### Cited Findings

| Claim (note 01 §1) | Verdict | Evidence (identical keys in [E11-05 log] L8, [E11-06 log] L8, [E11-05c log] L8) |
|---|---|---|
| TP=4 | TRUE | `'tp_size': 4` |
| Speculative algorithm DFLASH | TRUE | `'speculative_algorithm': 'DFLASH'` |
| Verification width 8 | TRUE | `'speculative_num_draft_tokens': 8`, `'speculative_dflash_block_size': None`; the runtime log line says `block_size=8` ([E11-06 log] L99, [E11-05 log] L100) |
| Steps 1 | TRUE | `'speculative_num_steps': 1` |
| Drafter Qwen3.8-27B-DFlash2 BF16 | TRUE | `'speculative_draft_model_path': '/kaggle/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1'`, `'speculative_draft_model_quantization': None`; load line `type=DFlash2DraftModel` ([E11-06 log] L92-98) |
| KV cache FP8 E4M3 | TRUE | `'kv_cache_dtype': 'fp8_e4m3'`; `KV Cache is allocated. dtype: torch.float8_e4m3fn` ([E11-06 log] L110-125) |

Other settings in the same server_args line that matter for the bottleneck:
- Communication: `'disable_custom_all_reduce': False`, `'enable_mscclpp': False`, `'enable_torch_symm_mem': False`, `'enable_symm_mem': False`, `'enable_nccl_nvls': False`, `'enable_flashinfer_allreduce_fusion': False`, `'enable_p2p_check': False`, `'boundary_reduction': 'rs+rsv'`, `'speculative_boundary_reduction': 'ar'`.
- Mamba state: `'mamba_ssm_dtype': 'float16'`.
- Capacity: `'max_running_requests'` is 13 in E11-06 and 12 in both E11-05 runs; `'mem_fraction_static': 0.83`; `'context_length': 155648`.
- `'cuda_graph_max_bs_decode': 12` with bs list `[1..8,10,12]`.
- Both target and drafter use `'attention_backend': 'flashinfer'`; `'linear_attn_backend': 'triton'`.
- The NCCL version is logged as `sglang is using nccl==2.29.7` ([E11-06 log] L32).
- **Line counts.**
  - GPT's counts (3,470 / 1,876) are exactly 45 higher than the newline counts (3,425 / 1,831). Each file contains 45 `\r` characters (tqdm progress bars), so GPT split on `\r` as well. Derived.
  - The files in the rev 6f ZIP extract are byte-identical to the local Kaggle-run copies (sizes 759,380 and 408,964 B; Python byte compare `True`).

### Inferences
- The production engine in E11-05, E11-06, E12 and G4B is the same: TP4, DFLASH width 8, FP8 KV, INT4 target and BF16 drafter. Cross-run step-time differences therefore come from workload (context, batch) and not from configuration.

### Gaps
- None for this question.

---

## Note 01 / 04: custom all-reduce messages in the logs

### Takeaway
TRUE. Every run logs both messages on all four TP ranks: E11-05 EVAL30, E11-05 commit, E11-06, E12 EVAL30 rev 1 and G4B rev 6d. The source shows the rule is a hard `world_size > 2 and not full_nvlink` check. It also shows the fallback is NCCL, with no other change.

### Cited Findings
- **[E11-06 log] L33-40**, verbatim:
  - `[2026-10-18 19:08:57 TP3] CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs. To silence this warning, specify disable_custom_all_reduce=True explicitly.` (and L34 TP2, L35 TP0, L36 TP1)
  - `[2026-10-18 19:08:57 TP0] CustomAllreduce is disabled because it's not supported on more than two PCIe-only GPUs. To silence this warning, specify disable_custom_all_reduce=True explicitly.` (and L38 TP3, L39 TP1, L40 TP2)
- **Same 8 lines in the other logs:**
  - [E11-05 log] L34-41 (06:55:41)
  - [E11-05c log] L33-40 (06:55:57)
  - [E12 log] L33-40 (2026-10-19 10:51:29)
  - [G4B log] L34-41 (2026-10-21 11:39:12)
  - So note 04's claim that the message appears in E11-05, E11-06 and E12 is TRUE.
- **A related line:** `multimem all-gather disabled (no multicast for world_size=4)` (E11-06 L144-147, E11-05 L145-148, G4B L146-148).
  - The source is [SRC] `distributed/device_communicators/triton_symm_mem_ag.py` L543-550.
  - It concerns the target LM-head logits all-gather (`layers/logits_processor.py` L482-488), which falls back to NCCL `tensor_model_parallel_all_gather` (triton_symm_mem_ag.py L519-521).
- **The deciding source**, [SRC] `distributed/device_communicators/custom_all_reduce_utils.py` L487-498:
  `full_nvlink = is_full_nvlink(physical_device_ids, world_size)` … `if world_size > 2 and not full_nvlink: logger.warning(f"{cls_name} is disabled because it's not supported on more than two PCIe-only GPUs. ..."); return`.
  - The P2P test (`can_p2p`, L504-510) runs only after this, so it is never reached at TP4.
  - A second gate in `custom_all_reduce.py` L271-274: `if self.world_size == 2 or self.full_nvlink: return inp_size <= self.max_size` / `return False`.
  - V2 is tried first (`custom_all_reduce.py` L358-369; `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` default True, `environ.py` L1347). `can_use_custom_all_reduce_v2` ends with `return full_nvlink is True` (`custom_all_reduce_v2.py` L497).
- **The fallback path:** pynccl inside CUDA graphs and torch.distributed (NCCL) in eager mode.
  - `parallel_state.py` L1068-1080: `if pynccl_comm is not None and not pynccl_comm.disabled: pynccl_comm.all_reduce(input_) ... else: torch.distributed.all_reduce(...)`.
  - `graph_capture` enables pynccl (L672-686).
- **NCCL transport (P2P vs SHM):** UNVERIFIABLE. No log contains `NCCL INFO` lines (grep over all E11 logs: none), and SGLang sets no `NCCL_P2P_*` or `NCCL_SHM_*` variables. It sets only `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` unless overridden (`entrypoints/engine.py` L1801-1816).

### Inferences
- GPT's statements here are all correct:
  - "The optimized custom AR path is unused at TP4."
  - "NCCL still works."
  - "The transport and its time share are unknown."
- The warning proves the code path. It does not prove time cost.

### Gaps
- Which NCCL transport was used, and the per-collective latency at about 1 MB. This needs `NCCL_DEBUG=INFO` and a size sweep (see the NCCL question below).

---

## Note 01: acceptance means by running count (E11-06 at 8 and 12, E11-05 at 12)

### Takeaway
The plain arithmetic means are exactly as GPT reported: 4.4628 (n=408), 4.3708 (n=278) and 3.9130 (n=114).
- GPT's caveat that these are "not the exact per-step aggregate" is only PARTLY TRUE. Per the source, at a fixed running count each record covers 40 forwards × B request-steps, so the plain mean equals the exact request-step-weighted mean, assuming B is constant inside each window.
- Token-weighted means are about 0.03-0.05 higher. Time-weighted means are within ±0.02.

### Cited Findings
Recomputed from every TP0 decode line with `decode_stats.py`. Two weightings are reported:
- **Model weights:** G_i = 40·B·a_i tokens, and gap_i = 40·B·a_i/g_i seconds.
- **Timestamp weights:** dt between consecutive decode lines, 1 s resolution.

| Log | B | n | arithmetic mean (GPT) | median | time-weighted (model / timestamp) | token-weighted (model / timestamp) | Verdict |
|---|---|---|---|---|---|---|---|
| E11-06 | 8 | 408 | **4.4628** (4.46) | 4.470 | 4.4685 / 4.4717 | 4.5102 / 4.5134 | TRUE |
| E11-06 | 12 | 278 | **4.3708** (4.37) | 4.340 | 4.3495 / 4.3677 | 4.4038 / 4.4048 | TRUE |
| E11-05 EVAL30 | 12 | 114 | **3.9130** (3.91) | 3.900 | 3.8888 / 3.8930 | 3.9417 / 3.9446 | TRUE |

Context for the same runs:
- **E11-06 at other running counts:**
  - B=9: n=400, mean 4.8127
  - B=11: n=158, mean 4.4710
  - B=10: n=36, mean 5.2075
  - Record counts by B: {1:1, 2:6, 3:2, 4:2, 5:1, 7:7, 8:408, 9:400, 10:36, 11:158, 12:278}.
- **E11-05 EVAL30** is dominated by B=11: n=2,371, mean 3.8473, median 3.820.
- **Whole-run means:** E11-05 3.8665, E11-06 4.5834, E12 EVAL30 rev 1 5.0993.
- **Cross-check of the accept-len definition:** across all records of all three logs, max |accept rate − (accept len − 1)/7| = 0.0057, which is rounding. This confirms the source definition.

### Inferences
- The "about 4.3" figure is representative only of E11-06 at 11-12 running. Across production runs the mean ranges from 3.87 to 5.10, which agrees with note 04.

### Gaps
- Per-position acceptance (the accepted-prefix histogram) is not in any log. Logs give means only. This is the input a width-4/6 prediction needs.

---

## Note 01 / 03 / 04: inferred step time at 12 running, by context bin

### Takeaway
GPT's medians and 80 % range are TRUE: E11-06 124.6 ms, p10-p90 116.2-132.5; E11-05 128.4 ms. Three findings GPT could not check:
- **The formula equals window wall time divided by 40.**
- **Context explains the 115 vs 125 ms gap.** At ≤10k context per request, every run shows about 115 ms:
  - E11-03 D: 115.5 ms
  - E11-06: 115.3 ms
  - E12: 115.4 ms
- **The 18 ms context cost matches a KV-read model.** From 0-10k to 50-60k context the step grows by about 18 ms, close to the 16.4 ms KV-read lower bound.

### Cited Findings
**Medians and spread at 12 running** (derived, t = B·a/g):

| Log | n | median | mean | p10-p90 | p5-p95 | windows with no prefill line and g>50 |
|---|---|---|---|---|---|---|
| E11-06 | 278 | **124.6** | 135.2 | **116.2-132.5** | 115.0-133.3 | n=273, median 124.4, p10-p90 116.1-131.9 |
| E11-05 EVAL30 | 114 | **128.4** | 144.9 | 125.2-143.1 | 124.2-221.3 | n=92, median 127.7, p10-p90 124.6-131.5 |

**By context bin.** The bin is `#full token` / B. `#full token` is the KV held by running requests: capacity − (available + evictable), per `pool_stats_observer.py` L271.

| Context per request | E11-06 B=12: n, median ms, mean acc | E11-05 B=12 | E11-05 B=11 | E12 EVAL30 B=12 (g>50, no prefill) |
|---|---|---|---|---|
| 0-10k | 28, **115.3**, 4.880 | – | 36, 113.2, 4.412 | 27, **115.4** |
| 10-20k | 55, 118.3, 4.544 | – | 145, 116.8, 3.970 | 18, 117.4 |
| 20-30k | 58, 122.5, 4.299 | 15, 124.3, 3.934 | 499, 120.8, 3.883 | – |
| 30-40k | 63, 126.8, 4.167 | 67, 127.6, 3.806 | 1,138, 124.5, 3.837 | 10, 128.2 |
| 40-50k | 58, 130.9, 4.342 | 32, 131.4, 4.127 | 553, 127.2, 3.768 | 20, 132.1 |
| 50-60k | 16, **133.3**, 4.049 | – | – | 44, 135.7 |
| 60-80k | – | – | – | 73, 140.2 |

**The "115.5 ms" figure** (Claude note 00; note 03 calls it a "context-normalized regression"):
- It is the raw median of E11-03 run D (INT4 + DFlash2, SSM fp16) at 12 running: n=67, all records in the 0-10k bin. The p10 is 114.8 ([E11-03 dir] `server_D_C_int4_dflash2_ssm_fp16_try0.log`).
- So note 03's interpretation is FALSE. It is a raw short-context median, and it equals E11-06's own 0-10k bin (115.3 ms).
- The other Claude-note anchors are also raw short-context medians: A (no spec) 12 running at 0-10k = 37.9 ms (n=236); 1 running = 21.9 ms; 22 running = 50.0 ms; B (MTP) 10 running at 0-10k = 79.9 ms; 1 running = 40.4 ms. D at 1 running = 39.9 ms.

**Prefill pollution.** Windows that contain at least one prefill log line have much larger inferred times:
- E11-06 B=12: n=5, median 323 ms
- E11-05 B=12: n=22, median 145.8 ms

This is consistent with source fact 1: gap_latency includes interleaved chunked-prefill time.

**E12 EVAL30 rev 1 at 12 running** (note 04 claims 137.9 ms). The median depends on the filter:
- all 280 records: 139.4
- g>50: 138.4 (n=248)
- g>100: **137.9** (n=233)
- prefill-free windows: 135.7 (n=192)

So the verdict is PARTLY TRUE (filter-dependent). E12's median context at 12 running is 658,066 tokens, about 54.8k per request, which explains the higher value. 87 E12 windows have g ≤ 50 (tool-call and prefill gaps).

### Inferences
- **At equal context, E11-06 and E12 run at the same about 115 ms per step** at 12 × 8 verified tokens. Differences between runs are context (KV read) plus prefill interleaving, not engine regressions.
- **The context slope fits a KV-read model.**
  - E11-06: 115.3 → 133.3 ms over about 50k tokens.
  - Target-only A: 37.9 → 44.4 ms over 0-10k → 20-30k.
  - A's growth (+6.5 ms for about 20k) matches 12 × 20k × 8,192 B = 1.97 GB per GPU ÷ 300 GB/s = 6.6 ms. Derived; the 300 GB/s is NVIDIA's L4 spec, not measured locally.

### Gaps
- None of these numbers is a CUDA-event or kernel time. A profiler trace is still needed to split the about 115 ms.

---

## Note 01 / 04: startup timelines

### Takeaway
GPT's three numbers are reproducible, but they use **different end points**:
- E11-05 7:39 and E11-06 7:30 run to the `Engine startup timings` / Uvicorn line.
- G4B 19:09 runs to `The server is fired up and ready to roll!`.

Measured to the same end point ("fired up"), the totals are E11-05 8:00, E11-06 7:52, G4B 19:09 and E12 8:05. The "CUDA graphs dominate E11; weight load dominates the G4B extra" statement is TRUE.

### Cited Findings
All rows are derived from log timestamps (1 s resolution) and the logged `elapsed=` values.

| Run | server_args → Uvicorn/timings | server_args → "fired up" | Weight load (target + drafter, logged) | CUDA graphs (target verify + draft, logged) | Other (to "fired up") |
|---|---|---|---|---|---|
| E11-05 EVAL30 | 06:55:13 → 07:02:52 = **459 s (7:39)** | → 07:03:13 = 480 s (8:00) | 12.87 + 0.55 = 13.42 s | 373.40 + 9.85 = 383.25 s | 83.3 s |
| E11-06 | 19:08:37 → 19:16:07 = **450 s (7:30)** | → 19:16:29 = 472 s (7:52) | 12.79 + 0.56 = 13.35 s | 372.53 + 9.73 = 382.26 s | 76.4 s |
| G4B rev 6d | 11:38:44 → 11:57:31 = 1,127 s (18:47) | → 11:57:53 = **1,149 s (19:09)** | **669.26** + 0.56 = 669.82 s | 384.08 + 9.92 = 394.00 s | 85.2 s |
| E12 EVAL30 rev 1 | 10:51:09 → 10:58:52 = 463 s | → 10:59:14 = 485 s | 13.49 + 0.55 = 14.04 s | 384.53 + 9.96 = 394.49 s | 76.5 s |

Sources:
- [E11-05 log] L8, L42, L57, L82, L86, L99, L142, L153, L168, L172, L185, L232
- [E11-06 log] L8, L41, L54, L75, L83, L98, L139, L149, L167, L171, L184, L230
- [G4B log] L8, L42, L56, L76, L87, L99, L138, L153, L171, L181, L185, L231
- [E12 log] L8, L41, L56, L81, L89, L97, L143, L150, L170, L180, L184, L231

The logged `Engine startup timings` lines agree:
- E11-06: `load_weight=13.48, kv_cache_allocation=4.08, scheduler_e2e=438.91, cuda_graph={... target_verify=372.55, ... draft_decode=9.74 ...}` (L184).
- G4B: `load_weight=669.96 ... scheduler_e2e=1115.52 ... target_verify=384.08` (L185).

What "other" contains (E11-06):
- server_args → Init parallel begin: 11 s
- Init parallel: 8.9 s; E11-05 and G4B: 16.5-16.9 s
- KV and mamba allocation: 4.1 s
- Draft-graph end → `DFLASH sampling pre-warm`: 22 s (19:15:44 → 19:16:06)
- Uvicorn → "fired up" warm-up: 22 s, including about 7 × 1.8 s of `chunk_gated_delta_rule_fwd_kkt_solve_kernel` Triton compiles after serving started (L201-225)

**Where the graph time goes.** The 372 s target-verify capture is almost entirely the **first** batch size: the tqdm line shows `bs=12 … 1/10 [06:02<…, 362.28s/it]`, after which the remaining 9 sizes take about 10 s ([E11-06 log] L148). This looks like one-time JIT/compile on the first capture. Inference; the cause is not in the log.

**G4B weight load.** 669 s vs about 13 s in the other runs: same model path and same per-GPU 4.62 GB ([G4B log] L76-82). So the storage-stall reading is consistent with the log, though the cause itself is not logged.

### Inferences
- Each cold start of a TP4 width-8 server costs about 7.5-8 min on a normal storage day. About 6 min of that is the first target-verify graph capture.
- A width-4 or TP2 arm would recapture with different shapes. The same order of cost should be budgeted per arm unless the compile cache carries over (not tested).

### Gaps
- Why the first capture takes about 362 s (Triton autotune, inductor, or GDN kernels) is not in the logs.
- The note-04 claim that L4x4 notebooks use double quota cannot be checked locally (Kaggle page).

---

## Note 01 / 03 / 04: per-GPU memory at TP4

### Takeaway
TRUE: INT4 target 4.62 GB/GPU, drafter 1.49 GB/GPU, target-verify graph 0.68 GB, draft graph 0.35 GB.
- The full per-GPU ledger is below.
- GPT's TP2 figure (12.22 GB combined weights by doubling) is PARTLY TRUE. It is an upper-bound style estimate: parts of the drafter are replicated per rank and would not double. That is an inference from the source and config, not measured.

### Cited Findings
**[E11-06 log] memory ledger, identical on all 4 ranks:**

| Step | Line | Value |
|---|---|---|
| Start | L53-56 | `Load weight begin. avail mem=21.72 GB` |
| INT4 target | L75-81 | `Load weight end. elapsed=12.79 s, type=Qwen3_5ForConditionalGeneration, quant=compressed-tensors, avail mem=17.10 GB, mem usage=4.62 GB` |
| DFlash2 drafter | L92-98 | `type=DFlash2DraftModel, avail mem=15.58 GB, mem usage=1.49 GB` |
| Multimodal reserve | L102-105 | `Reserving 0.10 GB of the KV budget for post-sizing multimodal allocations` |
| Mamba / GDN state | L106-109 | `max_mamba_cache_size: 56, conv_state size: 0.04GB, ssm_state size: 1.00GB intermediate_ssm_state_cache size: 1.97GB intermediate_conv_window_cache size: 0.03GB`, total 3.04 GB |
| Target KV | L110-116 | `#tokens: 866744, K size: 3.31 GB, V size: 3.31 GB`, total 6.62 GB; `Memory pool end. avail mem=5.89 GB` |
| Drafter KV | L117-125 | `#tokens: 866744, K size: 1.03 GB, V size: 1.03 GB`, total 2.06 GB; `avail mem=3.82 GB` |
| Target-verify graphs | L149-152 | `elapsed=372.53 s, mem usage=0.68 GB, avail mem=2.61 GB` |
| Draft graphs | L167-170 | `elapsed=9.73 s, mem usage=0.35 GB, avail mem=2.26 GB` |
| Sampling pre-warm | L171 | `peak=293771264 B headroom=389126144 B` |
| Final | L172 | `max_total_num_tokens=866744 … max_running_requests=13 … available_gpu_mem=1.88 GB` |

**[E11-05 log]** (max_running 12) differs only where expected:
- intermediate_ssm 1.83 GB (L107-110)
- KV #tokens 881,336 with K/V 3.36 GB and drafter K/V 1.05 GB (L111-126)
- graph avail 2.57 / 2.22 GB (L150-171)
- final 1.84 GB (L173)

**The memory numbers match the configs** (derived):
- **Target KV:** 16 full-attention layers × (4 KV heads / 4 ranks = 1) × head_dim 256 × 1 B (FP8) = 4,096 B per token for K, and the same for V.
  - 866,744 × 4,096 B = 3.55e9 B = 3.31 GiB. ✔
  - Per GPU that is 8,192 B/token; across 4 ranks 32,768 B/token.
  - [CFG-T]: `num_hidden_layers 64`, `full_attention_interval 4`; full layers at indices 3, 7, …, 63 (16); `num_key_value_heads 4`, `head_dim 256`.
- **Drafter KV:** 5 layers × (8 KV heads / 4 = 2) × 128 × 1 B = 1,280 B → 1.03 GiB. ✔
  - [CFG-D]: 5 layers, `num_key_value_heads 8`, `head_dim 128`.
  - Log L101: `DFLASH fused KV materialization enabled. n_layers=5, num_kv_heads=2, head_dim=128`.
- **SSM state:** 48 GDN layers × (48 value heads / 4 = 12) × 128 × 128 × 2 B (fp16) = 18.9 MB per slot.
  - × 56 slots = 1.06e9 B ≈ 1.00 GB. ✔
  - The intermediate (verify) cache = 13 requests × 8 tokens × 18.9 MB = 1.96 GB. ✔ E11-05: 12 × 8 → 1.81 GB vs the logged 1.83.
- **Drafter parts that are not TP-sharded** ([SRC] `models/dflash.py`):
  - `self.fc = nn.Linear(num_context_features*hidden, hidden)` (L700-707): 5 × 5120 × 5120 × 2 B = 262 MB.
  - `CandidateSelector` predecessor/successor codebooks `[vocab, r]`, "replicated on every TP rank" (L1077-1120): 2 × 248,320 × 256 × 2 B ≈ 254 MB, assuming BF16.
- **Some target parts are unquantized BF16:** [CFG-T] `quantization_config.ignore` lists every `model.language_model.layers.N.linear_attn` (all 48 GDN layers, with `in_proj_a`, `in_proj_b` and `norm`) plus `lm_head`.

### Inferences
- **TP2 per-GPU weights (estimate):**
  - Target: about 2 × 4.62 = 9.24 GB, if all target parts are sharded (not verified per tensor).
  - Drafter: about 2 × (1.49 − about 0.52 replicated) + 0.52 ≈ 2.5 GB, rather than 2.98 GB.
  - Total: about 11.7 GB, not 12.22 GB.
- **TP2 per-GPU state (estimate):** KV per token doubles to 16,384 B (target) + 2,560 B (drafter). The SSM state per slot doubles to 37.7 MB.
- So a TP2 replica holds roughly half as many tokens and slots per GPU. All of this is derived and needs a startup test.
- Claude note 00's "about 9.3 GB INT4 TP2 weights" matches the target-only doubling.

### Gaps
- Exact TP2 numbers need a real TP2 boot. This includes whether the embedding and lm_head shard as assumed, and the graph and activation sizes at TP2.

---

## Note 01 / 03 / 04: the 64 MB NCCL all-reduce measurement

### Takeaway
The 20.9 ms comes from the E11-03 preflight step `nccl_allreduce_tp4`. It measured **one size only**: 16 Mi fp32 elements = 64 MiB, with `torch.distributed` (ProcessGroupNCCL), 10 iterations. E11-04 repeated it (20.76-20.83 ms). **No smaller sizes were ever measured.**
- GPT note 01's statement that the raw output was not in its ZIP may be true of the ZIP, but the raw output and the code exist locally.

### Cited Findings
**Code:** `D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Notebooks/Build/build_e11_03_gpu_validation.py`, `_ar_worker` and `t_nccl`, about L324-357:
- `dist.init_process_group("nccl", ...)`
- correctness all-reduce of `torch.full((16 * 1024 * 1024,), rank+1)`
- then `y = torch.ones(16 * 1024 * 1024)` (fp32, 67,108,864 B)
- `for _ in range(10): dist.all_reduce(y)` with `torch.cuda.synchronize()` around a `time.time()` timer
- result key `ms_per_64MB`

The same function is in `build_e11_03_gpu_validation.rev1_backup.py`, `E11_03_sglang_int4_gpu_validation_on_KAGGLE.ipynb` and `E11_04_sglang_capacity_validation_on_KAGGLE.ipynb`.

**Raw output:**
- E11-03 (run 20261017T050445Z): `[{"rank": 0, "ok": true, "ms_per_64MB": 20.89}, {"rank": 1, ... 20.89}, {"rank": 2, ... 20.85}, {"rank": 3, ... 20.86}]`, step `sec 19.5`.
  - Sources: [E11-03 dir] `results.json` L88-100 (`/preflight/steps/nccl_allreduce_tp4`); `report.md` L12; `preflight_stdout.log` L4; Kaggle log `e11-03-sglang-int4-gpu-validation.log` L51.
- E11-04: `20.79, 20.76, 20.80, 20.83` ms, step `sec 11.8` ([E11-04 dir] `results.json` about L96; `e11-04-sglang-capacity-validation.log` L51).
- Same preflights, BF16 4096³ torch GEMM: **75.9 TFLOPS** (E11-03) and **78.8 TFLOPS** (E11-04); `bf16_gemm_tflops` in `preflight_stdout.log`.

**Derived bandwidth:**
- algbw = 67,108,864 B / 20.89 ms = **3.21 GB/s** (2.99 GiB/s)
- busbw = algbw × 2(n−1)/n = × 1.5 = 4.82 GB/s
- The notes' "3.06-3.1 GB/s" corresponds to 64e6 B / 20.9 ms (decimal MB). It is a rounding difference, not an error.

### Inferences
- The measurement is a large-message, eager, fp32 torch.distributed number. Production collectives are about 0.98 MB BF16, pynccl inside CUDA graphs, run 140 times per step.
- So GPT's warning not to extrapolate from 64 MB to small messages is correct (TRUE).
- **An upper bound from our own logs** (derived; assumes the 300 GB/s spec):
  - Target-only A at 1 running takes 21.9 ms per step.
  - That step includes 130 collectives (see the collective count below) and a read of about 4 GB of weights per GPU (about 13 ms at 300 GB/s).
  - That leaves about 9 ms or less for everything else, so small-message (10 KB) NCCL latency here is at most about 70 µs per collective.
  - This bounds the 10 KB regime only, not the 1 MB regime used at 12 × 8.

### Gaps
- No 8 KB-1 MB latency curve exists. No TP2-pair number exists. No in-graph pynccl number exists. All three need the GPU (note 01 §8 A2 / note 04 Gate A).

---

## Note 01 / 03 / 04: collectives per decode step for Qwen3.8-27B in SGLang 0.5.21

### Takeaway
From the source, one decode step with 12 requests × width 8 issues:
- **one target verify forward:** 129 all-reduces + 1 logits all-gather;
- **one draft forward:** 11 all-reduces + 2 small all-gathers;
- **accept:** 1-2 small broadcasts.

That is **140 all-reduces of 983,040 B each per step**, plus about 11.9 MB per rank of logits all-gather. There is no all-reduce + RMSNorm fusion on sm_89 with these flags, and `boundary_reduction 'rs+rsv'` changes nothing in this pure-TP layout.
- The notes' "64 × 2 = 128" is a close underestimate.
- Note 03/04's caveat that fused or deferred paths might change the count: the reductions are deferred (moved into the next layer's prepare stage), but not removed or fused. The count stays at 2 per layer.

### Cited Findings
Source: [SRC]. Read via a delegated full-file source read and spot-checked by me at qwen3_5.py L452-462 and L1238-1247, dflash.py L231-246 and L436-449, and custom_all_reduce_utils.py L485-510.

| Collective | Per target verify | Per draft step | Message size (per-rank send) | Source |
|---|---|---|---|---|
| AR: embedding (VocabParallelEmbedding) | 1 | 1 (eager, target embedding reused for the draft block) | 96 × 5120 × 2 = 983,040 B | `layers/vocab_parallel_embedding.py` L577-590; `speculative/dflash_worker_v2.py` L2492 |
| AR: GDN `out_proj` (`RowParallelLinear(..., reduce_results=False)`), completed in the FFN prepare stage | 48 | – | 983,040 B | `models/qwen3_5.py` L452-462; `layer_boundary` prepare.py L99-103 → ops.py L162 |
| AR: full-attention `o_proj` (`reduce_results=False`) | 16 | – | 983,040 B | `models/qwen3_5.py` L1238-1247 |
| AR: MLP `down_proj`, layers 0-62, deferred to the next layer's attention prepare | 63 | – | 983,040 B | `qwen2_moe.py` L194-203; layer_boundary exit.py L177-243, prepare.py L268-293 |
| AR: MLP layer 63 (inline) | 1 | – | 983,040 B | `layers/linear.py` L1671-1690 |
| AG: target logits (vocab-sharded lm_head; multimem disabled → NCCL) | 1 | – | 96 × 62,080 × 2 = 11,919,360 B in, 47,677,440 B out | `layers/logits_processor.py` L633-652, L913-925; `triton_symm_mem_ag.py` L519-521 |
| AR: drafter `o_proj` + `down_proj` × 5 layers (drafter IS TP-sharded: `QKVParallelLinear`, `RowParallelLinear` with default `reduce_results=True`, `MergedColumnParallelLinear`) | – | 10 | 983,040 B | `models/dflash.py` L231-246, L436-449 |
| AG: drafter candidate top-K values (fp32) and ids (int64) | – | 2 | 84 × 16 × 4 = 5,376 B and 84 × 16 × 8 = 10,752 B | `models/dflash.py` L105-152, L1228-1237; K = `selector_top_k 16` from [CFG-D] |
| Broadcast: accept (`SpecTpSync`, default `SGLANG_SPEC_TP_SYNC="all"`) | – | 1 greedy / 2 sampling | 768 B / 48 + 96 B | `speculative/dflash_worker_v2.py` L2135-2165; `environ.py` L1393 |
| Reduce-scatter | 0 | 0 | – | – |

Notes on the table:
- **Selector K.** The delegated read reported selector K as unverified. I resolved it from [CFG-D]: `selector_rank 256, selector_top_k 16`.
- **Collective-free drafter parts** (from the same source read): `fc`, the context-KV materialization from target layers [5, 19, 33, 47, 61], and the selector walk.
- **Fusion is unavailable.** FlashInfer all-reduce fusion auto-enables only on `is_sm90 or is_sm100` (`arg_groups/overrides.py` L931-941; residual/add_norm.py L142-184), and L4 is sm_89.
- **Graph buckets.** All target collectives run inside the CUDA graph at bucket 12 (captured bs list includes 12, so there are no padded rows). The draft embedding all-reduce and the broadcasts run eagerly.
- **Model dimensions** (TRUE): [CFG-T] `hidden_size 5120`, `num_hidden_layers 64`, `intermediate_size 17408`, 24 heads, 4 KV heads, `head_dim 256`, `vocab_size 248320`. GDN: 16 key heads, 48 value heads, head dim 128, `linear_conv_kernel_dim 4`.

### Inferences
- **Naive bandwidth model at 140 all-reduces:** 140 × 983,040 B / 3.06 GB/s = 45.0 ms. With note 00's 128: 41.1 ms, so the arithmetic in notes 01, 03 and 04 is TRUE.
- It remains a hypothesis. The per-message latency at about 1 MB under pynccl-in-graph is unmeasured, and the logits all-gather adds traffic that no note counted.
- **What the step-time data adds:**
  - The +8.4 ms/request marginal cost from B=1 (39.9 ms) to B=12 (115.5 ms) in E11-03 D includes both a growing collective payload and growing GEMM work.
  - The logs cannot separate them. Only a trace can.

### Gaps
- Scheduler-level CPU (gloo) object broadcasts were not counted.
- Actual per-collective time and overlap need an nsys or torch-profiler trace.

---

## Source: custom all-reduce at TP2, and alternatives in 0.5.21 for sm_89

### Takeaway
- **TP4 on PCIe:** both custom all-reduce versions are hard-disabled. Only a code patch would change this.
- **TP2:** V1 `CustomAllreduce` turns on automatically for messages up to 8 MiB. It assumes P2P works and does not test it, because `enable_p2p_check=False` monkey-patches the check to `True`. V2 stays off because it needs full NVLink.
- **The only non-NCCL option at TP4 on PCIe** is FlashInfer PCIe-IPC all-reduce, through env `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`. Its availability in flashinfer 0.6.18 on sm_89 is unverified.
- **Ruled out by the code on sm_89 PCIe:** torch symmetric memory, quick all-reduce, NVLS, multimem and FlashInfer fusion. MSCCL++ is TP4-only and its package is not installed.

### Cited Findings
**V1 at TP2:**
- `custom_all_reduce.py` L41 allows world sizes `[2,4,6,8]`, and `_MAX_CAR_SIZE = 8192*1024` (L42).
- At world_size 2 the NVLink rule is skipped. `should_custom_ar` returns `inp_size <= self.max_size` (L271-273).
- The buffers are IPC (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`, L147-171), which in practice needs P2P.
- With `enable_p2p_check=False` (`arg_groups/fields/parallel.py` L236-239), `bootstrap.py` L121-122 calls `monkey_patch_p2p_access_check()`. That replaces `gpu_p2p_access_check` with `lambda *a, **k: True` (`utils/common.py` L2400-2408).
- `--enable-p2p-check` runs the real IPC test (`custom_all_reduce_utils.py` L149-231, L304-318).
- CUDA graphs are supported (`register_graph_buffers`, L181-194, L234-258).

**V2:** `can_use_custom_all_reduce_v2` → `return full_nvlink is True` (`custom_all_reduce_v2.py` L482-497). sm_89 gets the default size table with no multicast (`configs/custom_all_reduce_v2.py` L293-300).

**PCIe-IPC (FlashInfer):**
- `pcie_ipc_ar.py` L2-6, docstring: "targets hosts where every peer transfer crosses the CPU root complex -- no NVLink, no multicast".
- `_SUPPORTED_WORLD_SIZES = (2, 4, 8)` (L77).
- Enabled by `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE = EnvBool(False)` (`environ.py` L1354), with `SGLANG_PCIE_IPC_MAX_NUMEL` (L1355).
- BF16 only, `numel <= max_numel` (L387-404).
- It needs `from flashinfer.comm import PcieIpcAllReduceWorkspace`, otherwise it warns and stays on NCCL (L224-231).
- It is dispatched right after custom AR (`parallel_state.py` L969-978) and built before graph capture (`base_runner.py` L256, L337-352).
- On success it logs `FlashInfer PCIe-IPC all-reduce enabled (world=...)` (L241-245).
- The pinned flashinfer is 0.6.18 (`D:/ARC AGI 2/_scratch/session_1c6db7aa/sglwh/lock_cuda130.txt` L42; also preflight versions).

**Ruled out or limited:**
- **Torch symm-mem** (`--enable-torch-symm-mem`): max-size table only for capability majors 9 and 10 (`all_reduce_utils.py` L3-16); sm_89 logs "Device capability 8 not supported" (`torch_symm_mem.py` L80-89); it also needs `multicast_ptr != 0` (L116-123).
- **Quick AR:** ROCm only (`custom_all_reduce_ops.py` L18 `IS_QUICK_AR_AVAILABLE = _is_hip`; `quick_all_reduce.py` L27-38).
- **MSCCL++** (`--enable-mscclpp`): world sizes (4, 8, 16, 32, 64) (`pymscclpp.py` L324, L1075-1083), so not TP2. It has non-NVLS algorithms `default_allreduce_packet` (≤2 MB) and `rsag_zero_copy` (L381-401). It needs the `mscclpp` package, which is absent from the local lock files.
- **NCCL symmetric memory** (`--enable-symm-mem`): pynccl with `graphUsageMode=1` (`pynccl.py` L113-120). It forces `NCCL_CUMEM_ENABLE=1` and `NCCL_NVLS_ENABLE=1` (`engine.py` L1801-1810). Its benefit on PCIe is unverified.
- **NVLS** (`--enable-nccl-nvls`): only sets `NCCL_NVLS_ENABLE=1`, and needs NVSwitch.
- **FlashInfer fusion:** auto-enables only on sm90/sm100.
- **Adaptive speculation exists but excludes DFLASH:** `adaptive_unsupported_reason` returns `"speculative_algorithm=… (only EAGLE/EAGLE3 are supported)"` (`speculative/adaptive_spec_params.py` L59-63). The server_args show `'speculative_adaptive': False`.
- **Ragged/variable-width verify** (`SGLANG_RAGGED_VERIFY_MODE`, default `"static"`, `environ.py` L1383; `speculative/ragged_verify.py`) is wired into DSpark components. `dflash_worker_v2.py` has no `ragged_verify` reference (grep). So it is not a DFLASH option in 0.5.21, which is consistent with note 01 saying dynamic verification (#36127) is not available.

### Inferences
- **The TP4-vs-2×TP2 comparison should include a third arm:** TP4 with `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`. It is a single env var and needs no topology change. Startup will show whether flashinfer 0.6.18 has the workspace class.
- **At TP2, run with `--enable-p2p-check` once.** If P2P is actually blocked in the Kaggle VM, the default unchecked V1 path could fail.

### Gaps
- Whether Kaggle L4 VMs allow GPU P2P; whether flashinfer 0.6.18 ships `PcieIpcAllReduceWorkspace` with sm_89 kernels. There is no local FlashInfer source.

---

## Source: what happens when `--speculative-num-draft-tokens` differs from block_size 8

### Takeaway
It is a pure runtime override; no weights are retrained or reshaped.
- **What changes:**
  - the worker's block width;
  - the draft block layout (anchor/bonus token + width−1 `<|MASK|>` tokens);
  - the conv position modulus;
  - the verify graph width.
- **Why width 4 is not a truncation of width 8:** the DFlash2 drafter attends **bidirectionally** (non-causal) within the block. At width 4 the first positions no longer see the 4 extra mask positions, so their hidden states, and therefore the candidates, differ.
- The grouped conv and the selector walk are causal/forward and would truncate cleanly. The attention does not.
- So GPT's statement that the change is "not equivalent to dropping the last four predictions" is TRUE, and this is the mechanism.

### Cited Findings
**Worker** (`speculative/dflash_worker_v2.py` L476-493):
```python
self.block_size = int(get_spec().speculative_num_draft_tokens)
model_block_size = draft_config.block_size
...
if model_block_size is not None and int(model_block_size) != int(self.block_size):
    logger.warning("DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.", ...)
self.draft_model.set_block_size(self.block_size)
self.speculative_num_draft_tokens = int(self.block_size)
```

**set_block_size** (`models/dflash.py` L739-750) only stores the integer:
```python
self.block_size = int(block_size)
for layer in self.layers:
    for conv in (layer.attention_conv, layer.mlp_conv):
        if conv is not None:
            conv.block_size = self.block_size
```
The docstring says: "The convolutions are built from the checkpoint's block_size, which --speculative-num-draft-tokens may override; the layout they index depends on it." No parameter is touched.

**Grouped conv** (`models/dflash.py` L465-477) uses the new width only as the in-block position modulus. Note 03's claim is TRUE:
```python
position = torch.arange(hidden_states.shape[0], ...)
if block_size & (block_size - 1) == 0: position = position & (block_size - 1)
else: position = position % block_size
for tap in range(1, taps):
    shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
    out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
```
With `conv_kernel_size 2` ([CFG-D]) it looks back one token only, which is causal.

**Block layout at draft time** (`dflash_worker_v2.py` L2466-2467, eager path L2447-2448): `block_ids.fill_(int(self._mask_token_id)); block_ids[:, 0].copy_(draft_input.bonus_tokens)`, then `noise_embedding = embed_module(block_ids)` (L2492).
- Width w means 1 anchor + (w−1) masks, at positions prefix_len .. prefix_len+w−1.
- The log confirms `mask_token=<|MASK|>, mask_token_id=248070 ... noise_embed_scale=1.0` ([E11-06 log] L100).

**Non-causal attention.** [CFG-D] has `"is_causal": false`, `"layer_types": ["sliding_attention" ×5]` and `"sliding_window": 2048`.
- `_get_dflash_attention_type` returns `AttentionType.DECODER if is_causal else AttentionType.ENCODER_ONLY` (`models/dflash.py` L155-161); sliding layers use window `2048−1` (L181-185; `dflash_utils.py` L418-435).
- FlashInfer then runs `causal = False` for `ENCODER_ONLY` (`layers/attention/flashinfer_backend.py` L1402-1407). Its comment says: "For cases like DFLASH draft (ENCODER_ONLY / non-causal)" (L1127).

**Selector:** `CandidateSelector.sample_path` walks forward slot by slot (`models/dflash.py` L1137-1196; Triton `selector_walk_triton`), so it is causal along positions.

**Target side:** the verify CUDA graph is captured with `num_tokens_per_req = speculative_num_draft_tokens` (log `Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=8`, [E11-06 log] L137-143). A width change recaptures the verify graph and changes the per-step collective payload (96 → 48 rows at width 4).

**The checkpoint was trained and benchmarked at width 8** (TRUE): [CFG-D] `dflash_config: block_size 8, conv_group_size 16, conv_kernel_size 2, mask_token_id 248070, selector_rank 256, selector_top_k 16, target_layer_ids [5, 19, 33, 47, 61]`. The local `dflash2_README.md` L16-35 and the Evaluation section say "Speculation block size: 8 (7 draft tokens per verification step)".

**Width-4 maximum:** at width 4 the maximum `accept len` is 4 (3 drafts + 1 bonus), by the source definition `draft_per_round = num_draft_tokens − 1`.
- Note 01 ("theoretical maximum 4") is TRUE.
- Note 04 ("with the bonus-token convention, more than 4 output tokens per step are possible at width 4") is **FALSE**.
- Width 8 logs never exceed 8: the maximum observed accept len is 7.90 in E11-03 C at B=1.

### Inferences
- Width 4 or 6 is a valid runtime setting, but its acceptance cannot be predicted from width-8 logs. Both the drafter's context (bidirectional over fewer masks) and the selector lattice length change.
- A measured accepted-prefix histogram at each width is needed, as notes 03 and 04 say.
- **Drafter KV read is cheap at any width:** with `sliding_window 2048` the drafter attention reads only the last 2,047 context tokens. That is about 12 × 2,048 × 2,560 B ≈ 63 MB per GPU per step, so the drafter adds almost no context cost.

### Gaps
- How much acceptance drops at width 4 or 6 for this checkpoint is unknown without a GPU run.

---

## Note 01: remaining checkable claims

### Takeaway
The arithmetic in note 01 is correct. The geometric-acceptance model reproduces exactly.

### Cited Findings
- **Marginal costs** (note 01 §4.1, from note 00). Derived from the raw medians above; TRUE.
  - Target-only: (37.9 − 21.9)/11 = 1.45 ms and (50.0 − 21.9)/21 = 1.34 ms.
  - MTP: (79.9 − 40.4)/9 = 4.39 ms, i.e. 1.10 ms per verified token at about 4 tokens.
  - DFlash2: (115.5 − 39.9)/11 = 6.87 ms, i.e. 0.86 ms per verified token.
- **Throughput model:** T8 = 12 × 4.3 / 0.1155 = 446.8 tok/s (note 01 "≈447", TRUE). With 0.115 s: 448.7 (note 04 "≈449", TRUE).
- **Geometric model:** solving 1+α+…+α⁷ = 4.3 gives α = 0.8109 (note 01 "≈0.811", TRUE). Then L4 = 3.002, L6 = 3.785, L3 = 2.469; T4 at 70 ms = 514.6 tok/s (note 01 "≈515", TRUE).
- **Required L4** for 550 / 650 tok/s at 70 ms: 3.208 / 3.792 (notes 01, 03, 04 "3.21-3.79", TRUE).
- **The width-4 table** in notes 03 and 04: 2.5 → 428.6, 3.0 → 514.3, 3.3 → 565.7, 3.5 → 600.0, 3.8 → 651.4, 4.0 → 685.7. TRUE.
- **Message-size arithmetic:** 12 × 8 × 10,240 = 983,040 B; × 128 = 125.8 MB; / 3.06 GB/s = 41.1 ms (notes 01, 03, 04, TRUE).
  - Correction from the source: there are 140 all-reduces per step, which gives 45.0 ms.
- **Correctness evidence the notes did not have:** our own E11-03 run already showed greedy divergence. This is consistent with note 01 §7's concern (issues #36548 and #38009 are UNVERIFIABLE locally), but our data does not separate batch nondeterminism from DFlash-specific bugs.
  - DFlash2 vs target-only greedy: 1/4 identical, mean common prefix 155.0 of 512 tokens.
  - MTP vs target-only greedy: 0/4 identical.
  - Greedy alone vs inside a 16-batch: 0/4 identical, mean common prefix 15.5 of 512.
  - Source: [E11-03 dir] `report.md` "greedy_vs_A" lines and the "contam" extra test.
- **L4 121 TFLOPS dense BF16:** UNVERIFIABLE locally. Locally measured torch BF16 GEMM is 75.9 / 78.8 TFLOPS (preflights above).
- **Public issues** #36548, #38009, #37475, #39087, #28180, #36127: UNVERIFIABLE locally.
  - #37475 (missing `target_layer_ids`): our checkpoint config has `target_layer_ids [5, 19, 33, 47, 61]`, so that risk does not apply.
  - #39087 (quantized drafter): our drafter is BF16, `speculative_draft_model_quantization None`.

### Inferences
- The 46 % figure (8 verified rows → 4.3 output tokens) and note 04's 52.9 % draft rejection ((7 − 3.3)/7) describe different denominators. Both are arithmetically right.

### Gaps
- The hardware specs (300 GB/s, 121 TFLOPS, PCIe Gen4 rates) and the Kaggle quota policy need non-local sources.

---

## Note 03: claim-by-claim

### Takeaway
All log-derived numbers in note 03 reproduce. Its one interpretive claim, that 115.5 ms is a context-normalized regression, is FALSE. Its source claim about the grouped conv is TRUE.

### Cited Findings

| Note 03 claim | Verdict | Evidence |
|---|---|---|
| E11-05 B=11: n=2,370, median 124.2 ms, accept median 3.82 | TRUE | 2,371 records; 2,370 after excluding the one g ≤ 50 window ([E11-05 log] L309, 17.36 tok/s); median 124.2 either way; accept median 3.820 |
| E11-05 B=12: n=114, 128.4 ms, accept 3.90 | TRUE | n=114, median 128.4, accept median 3.900 (with g>50: n=113, same medians) |
| E11-06 B=11: n=158, 124.4 ms, accept 4.47 | TRUE | n=158, 124.4, median 4.475 (the mean of 4.47 and 4.48) |
| E11-06 B=12: n=277, 124.6 ms, accept 4.34 | TRUE | 278 records; 277 after excluding the g=20.73 window ([E11-06 log] L311); median 124.6, accept 4.340 |
| Median total context at 12 running: E11-05 ≈449k, E11-06 ≈357k | TRUE | E11-05 449,175 (all 114) / 448,853 (g>50); E11-06 355,966 (all 278) / **357,034** (n=277) |
| 115.5 ms "may be a context-normalized regression estimate, not a raw median" | **FALSE** | Raw median of E11-03 D at 12 running (n=67, all at 0-10k context); E11-06's 0-10k bin at 12 running = 115.3 ms (n=28) |
| 64 layers, 16 full-attention | TRUE | [CFG-T] `num_hidden_layers 64`, `full_attention_interval 4`, layer_types full at 3, 7, …, 63 |
| FP8 KV ≈ 32.8 KB/token total; 12 × 50k ≈ 4.9 GB/GPU; ≈16.4 ms at 300 GB/s | TRUE (arithmetic); 300 GB/s UNVERIFIABLE locally | 16 × 4 heads × 256 × 2 (K, V) × 1 B = 32,768 B; per GPU 8,192 B × 600,000 = 4.915e9 B; matches logged `K size: 3.31 GB` for 866,744 tokens. Drafter KV (2,560 B/token/GPU) is windowed at 2,048 tokens, so the drafter context read is about 63 MB |
| Bus-bandwidth factor 2(4−1)/4 = 1.5 | TRUE (arithmetic) | Applied: 64 MiB / 20.89 ms → busbw 4.82 GB/s |
| 64 MB / 20.9 ms ≈ 3 GB/s algorithm bandwidth | TRUE | 3.21 GB/s (decimal bytes, 64 MiB payload) |
| Per-GPU INT4 4.62 GB, drafter 1.49 GB, total ≈6.11 GB | TRUE | [E11-06 log] L75-81, L92-98 |
| TP2 ≈12.2 GB weights per GPU | PARTLY TRUE | Naive doubling; replicated drafter parts (about 0.52 GB) don't double, so ≈11.7 GB (derived estimate) |
| PCIe Gen4 x16 = 32 GB/s each way (64 GB/s aggregate) | UNVERIFIABLE locally | Not in any local file |
| dflash.py grouped-conv position indexing uses the new width after set_block_size() | TRUE | `models/dflash.py` L739-750 sets `conv.block_size`; `_grouped_conv` L470-473 uses `position & (block_size-1)` / `% block_size` |
| A width change is not a free optimization (conv block handling changes) | TRUE, but the main reason is different | The bigger effect is non-causal intra-block attention (`is_causal false` → ENCODER_ONLY → `causal = False`), see the block-size section |
| Collectives: "128 is a good starting estimate, but LayerCommunicator / fused / deferred paths make the exact count uncertain" | PARTLY TRUE | This build has no LayerCommunicator in the Qwen3.5 path; it uses `layer_boundary` stages. Reductions are deferred but not fused or removed: 129 AR + 1 AG per verify and 11 AR + 2 AG per draft step = 140 AR/step |
| DFlash2 article: mean acceptance ≈4.80 | TRUE (local mirror of the model card) | `sgl_src/dflash2_README.md` acceptance table (H200, block 8): GSM8K 5.46, MATH-500 5.28, HumanEval 4.39, MBPP 4.79, MT-Bench 4.10; mean 4.80 |
| "Our ARC traces ~4.3" | PARTLY TRUE | 4.37 at B=12 in E11-06, but whole-run means are 3.87 (E11-05), 4.58 (E11-06), 5.10 (E12) |
| Width-8 mean alone cannot determine width-4 acceptance | TRUE | The logs contain only means; no per-position histogram is logged (metrics_reporter.py L929-937) |
| "~5,300 raw lines" in the two E11 logs | TRUE | 3,425 + 1,831 = 5,256 newline lines (5,346 if `\r` counts) |
| HF discussion (positions 4-7 ≈ 0.1; 4-token 60-75 %), SpecKV, Learning-to-Draft, the tonyd2wild TP2/TP4 repo, NVIDIA blog | UNVERIFIABLE locally | – |

### Inferences
- Note 03's main practical point is supported by the logs: compare arms at matched context, because context alone moves the step from about 115 to about 133 ms at B=12.

### Gaps
- PCIe link width and generation on the Kaggle host (`nvidia-smi -q` was never logged).

---

## Note 04: claim-by-claim

### Takeaway
Note 04's log numbers and the accept-len correction are TRUE. The E12 step median is filter-dependent. One claim is FALSE: at width 4, more than 4 tokens per step is impossible.

### Cited Findings

| Note 04 claim | Verdict | Evidence |
|---|---|---|
| CustomAllReduceV2 warning present in the E11-05, E11-06 and E12 EVAL30 logs | TRUE | [E11-05 log] L34-37, [E11-06 log] L33-36, [E12 log] L33-36 (plus the CustomAllreduce lines after them); also [E11-05c log] and [G4B log] |
| The TP2 restriction does not apply; the optimized path is possible with P2P | TRUE for V1, FALSE for V2 | V1 enabled at world_size 2 (`custom_all_reduce.py` L271-273); V2 requires `full_nvlink is True` (`custom_all_reduce_v2.py` L497). P2P is assumed, not tested, by default |
| "P2P absent / host-RAM staging" not proven | TRUE | No NCCL_DEBUG output in any log; the SGLang P2P test is never reached at TP4 |
| E11-05: 2,509 records, mean accept 3.87, 12-stream median 128.4 ms | TRUE | 2,509; 3.8665; 128.4 |
| E11-06: 1,299 records, 4.58, 124.6 ms | TRUE | 1,299; 4.5834; 124.6 |
| E12 EVAL30 rev 1: 1,209 records, 5.10, 137.9 ms | TRUE / TRUE / PARTLY TRUE | 1,209; 5.0993; 12-running median 139.4 (all 280), 138.4 (g>50), **137.9 (g>100, n=233)**, 135.7 (prefill-free) |
| E11-06 graph memory: target 0.68 GB, draft 0.35 GB; target capture 372.5 s | TRUE | [E11-06 log] L149-152 (`elapsed=372.53-372.55 s, mem usage=0.68 GB`), L167-170 (`mem usage=0.35 GB`) |
| accept len = output tokens per verify step including the bonus token | TRUE | `metrics_reporter.py` L929 (`spec_num_accept_tokens / spec_num_forward_ct`), with `num_accept_tokens=num_generated_tokens` (`batch_result_processor.py` L952-958); per-request mean, since forward_ct += bs (L501) |
| accept rate counts accepted drafts, denominator num_draft_tokens − 1 = 7 | TRUE | L931-937: `draft_per_round = get_spec().speculative_num_draft_tokens - 1`; `spec_accept_rate = num_correct_drafts / total_draft_tokens` |
| Cross-check of logged pairs | TRUE | 4.07 ↔ 0.44: (4.07 − 1)/7 = 0.4386; 5.90 ↔ 0.70: 4.90/7 = 0.700 ([E11-06 log] first two decode lines); max deviation over all 5,017 records in three logs = 0.0057 |
| 4.3 accept len → 3.3 correct drafts → 47.1 % draft acceptance | TRUE | 3.3/7 = 0.4714 |
| At width 4, more than 4 output tokens per step are possible (bonus convention) | **FALSE** | The block contains the anchor plus w−1 drafts (`dflash_worker_v2.py` L2466-2467); maximum accept len = w = 4 |
| FP8 KV: 12 × 50,000 × 16 × (4/4) × 256 × 2 × 1 ≈ 4.92 GB/GPU | TRUE | = 4,915,200,000 B; per-token per-GPU 8,192 B confirmed by logged K size |
| 300 GB/s → ≈16.4 ms | TRUE (arithmetic); spec UNVERIFIABLE locally | 4.9152e9 / 3e11 = 16.4 ms; observed context slope ≈18 ms (115.3 → 133.3 ms, E11-06 B=12) |
| TFLOPS: 2 × 27B × 96 / (4 × 0.115) ≈ 11.3 TFLOPS/GPU | TRUE (arithmetic) | = 11.27 TFLOPS. Context: measured torch BF16 GEMM on these L4s = 75.9 / 78.8 TFLOPS (E11-03 / E11-04 preflight). The 48 GDN layers and lm_head are BF16 (unquantized), not INT4 ([CFG-T] ignore list), so "INT4 GEMM" understates part of the work |
| Mean acceptance varies 3.87-5.10 across runs | TRUE | See above |
| 128 AR/step "plausible estimate; actual count not profiled" | TRUE (source count is 140) | See the collective-count section |
| 2×TP2 KV/context capacity likely lower than TP4 | TRUE (derived) | Per-GPU KV per token doubles (8,192 → 16,384 B target + 1,280 → 2,560 B drafter); SSM per slot doubles |
| Repeated cold starts are expensive (372.5 s graph capture) | TRUE | Startup table above; first-bucket capture ≈362 s |
| PCIe ≈64 GB/s bidirectional, ≈31.5 GB/s one way; L4 24 GB / 300 GB/s / 121 TFLOPS | UNVERIFIABLE locally | The logs show `avail mem=21.72 GB` at start ([E11-06 log] L53-56) |
| "Top Kaggle teams did the same optimization": no evidence | UNVERIFIABLE | – |

### Inferences
- The accept-len correction matters only for interpreting "waste". All throughput arithmetic in notes 01, 03 and 04 already uses accept len as output tokens per request-step, which is correct.

### Gaps
- None beyond the hardware-spec and web items listed.

---

## Facts the GPT notes could not check (summary)

### Takeaway
Eleven local facts change or sharpen the picture:
1. **Step-time formula.** The "step time" proxy is exactly window wall time divided by 40 forwards, prefill included.
2. **The 115 ms is a raw short-context median.** All runs show about 115 ms at B=12 below 10k context, so the 115-vs-125 ms gap is context.
3. **Collective count.** It is 140 all-reduces per step, about 0.98 MB each, plus a 12 MB logits all-gather. Nothing is fused.
4. **PCIe-IPC all-reduce.** SGLang 0.5.21 ships a FlashInfer PCIe-IPC all-reduce for exactly this no-NVLink case, at TP4 and TP2, behind `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`.
5. **Custom AR at TP2.** It is V1 only, and P2P is assumed without being tested.
6. **Width 4 changes predictions.** The drafter's intra-block attention is non-causal, so width 4 is not a truncation.
7. **Drafter KV read is cheap.** Drafter attention is sliding-window 2,048.
8. **Some weights are BF16.** The GDN layers and lm_head are unquantized BF16.
9. **NCCL measurement scope.** The NCCL figure is a single 64 MiB eager fp32 measurement, repeated twice (20.9 and 20.8 ms). No small sizes were measured.
10. **Startup cost.** About 362 s of every cold start is the first target-verify graph bucket.
11. **Greedy divergence.** Our E11-03 run already showed greedy divergence: batch vs solo 0/4 identical, DFlash2 vs target 1/4 identical.

### Cited Findings
- Each fact is sourced in the sections above:
  - metrics_reporter.py L898-937, L501
  - E11-03 D, E11-06 and E12 context bins
  - collective table
  - `pcie_ipc_ar.py` L2-6, L77; `environ.py` L1354
  - `custom_all_reduce.py` L271-273; `utils/common.py` L2400-2408
  - `models/dflash.py` L155-185; `flashinfer_backend.py` L1402-1407; [CFG-D]
  - [CFG-T] ignore list
  - `build_e11_03_gpu_validation.py` `_ar_worker`
  - [E11-06 log] L148
  - [E11-03 dir] `report.md`

### Inferences
- **The cheapest first GPU measurement that discriminates the hypotheses:**
  1. `NCCL_DEBUG=INFO` plus an 8 KB-1 MB all-reduce sweep (TP4 and both TP2 pairs);
  2. a TP4 width-8 baseline at matched context;
  3. the same with `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`;
  4. width 6/4 with accepted-prefix histograms.
- If PCIe-IPC loads and cuts the step noticeably, it captures part of the TP2 benefit without two servers. This is an inference, not measured.

### Gaps
- Per-kernel time split of the about 115 ms step (GEMM vs GDN vs NCCL vs drafter): needs a profiler trace on GPU.
- NCCL transport, P2P availability, and flashinfer PCIe-IPC availability on Kaggle L4: needs a GPU run.
- Acceptance at widths 4/6: needs a GPU run.
