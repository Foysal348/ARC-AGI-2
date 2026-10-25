# DFlash2 verify width and correctness for Qwen3.8-27B on SGLang 0.5.21+ (4x L4, 12 concurrent, ~60k thinking)

Labels used below: **[measured]** = a number someone actually measured (ours or a third party's), **[claimed]** = an author's or vendor's statement, **[code]** = read directly from the local SGLang v0.5.21 source, **[inference]** = my own reasoning or calculation.
Local source root: `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/srt/` (abbreviated `SRC/` below).
Our own measured runs: `D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/` (abbreviated `E11R/`).

## Q1. Width override: what happens when `--speculative-num-draft-tokens` is not the checkpoint `block_size` (8)?

### Takeaway
Setting a width of 4 or 6 is a supported code path. SGLang logs a warning, resizes the verify block, and passes the new size to the drafter's convolutions. It is **not** a plain truncation, though. The drafter is non-causal (bidirectional within the block), so with fewer mask slots it computes the early positions differently. The DFlash paper's train/test mismatch ablation suggests that running a block smaller than the trained one costs only about 2–4 % τ compared with a model trained at that size. Running a larger block than trained costs much more. Neither the DFlash authors nor inco.ai publish a recommendation for a smaller-than-trained block on Qwen3.8-27B, but the DFlash paper does recommend smaller blocks when verification is compute-bound (large batches).

### Cited Findings
- **[code]** When `speculative_num_draft_tokens` is set, the worker uses it as `self.block_size`. If it differs from `dflash_config.block_size`, it only logs `"DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s."` and then calls `self.draft_model.set_block_size(self.block_size)`. — `SRC/speculative/dflash_worker_v2.py` lines 476–493
- **[code]** `set_block_size()` sets `self.block_size` on the model and on every layer's `attention_conv` / `mlp_conv`. Per its docstring: "The convolutions are built from the checkpoint's block_size, which --speculative-num-draft-tokens may override; the layout they index depends on it, so the resolved value has to reach them." — `SRC/models/dflash.py` lines 739–750
- **[code]** The two-tap grouped conv indexes `position % block_size` (or `& (block_size-1)` for powers of two) and masks taps with `position >= tap`, so the taps reset at every block boundary for any width. — `SRC/models/dflash.py` lines 465–477
- **[code]** The draft block is built as `[bonus_token, mask, mask, …]` of length `block_size`: `block_ids.fill_(mask_token_id); block_ids[:,0] = bonus_tokens`, with positions `prefix_len + 0..block_size-1`. — `SRC/speculative/dflash_worker_v2.py` lines 2425–2470
- **[code]** The checkpoint config has `"is_causal": false`, and `_get_dflash_attention_type` maps that to `AttentionType.ENCODER_ONLY`, i.e. bidirectional attention inside the block. — `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_config.json`; `SRC/models/dflash.py` lines 155–161
- **[code]** Checkpoint facts: `block_size 8`, `selector_top_k 16`, `conv_kernel_size 2`, `target_layer_ids [5,19,33,47,61]`, 5 sliding-window layers (window 2048), mask_token_id 248070. — `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_config.json`
- **[code]** The verify kernel treats `candidates[:,0]` as the current token and compares drafts `1..block_size-1` with the target argmax. Width W therefore means W−1 draft tokens plus one bonus token, and the maximum accepted length is W. — `SRC/speculative/dflash_utils.py` lines 875–925
- **[code]** The per-request KV reserve for each decode step is `2 * block_size`, so a narrower width reserves less. — `SRC/speculative/dflash_info_v2.py` lines 141–147
- **[claimed]** DFlash training: "the draft model is trained to predict the next block_size−1 tokens in parallel." Tokens within a block "attend bidirectionally", and attention across blocks is disallowed. — [DFlash paper v2, §4.2](https://arxiv.org/html/2602.06036v2)
- **[measured, paper Table 8]** Train/test block-size mismatch (8-layer draft), shown as Math500 / HumanEval / MT-Bench τ: train16-test16 6.33 / 5.29 / 3.50; **train16-test8 5.09 / 4.44 / 3.18**; train8-test16 5.02 / 4.28 / 3.09; train8-test8 5.21 / 4.61 / 3.29. The authors say a block-16 model tested at 8 is "close to" a model trained and tested at 8. The reverse direction does not hold. — [DFlash paper v2, §5.5.4 / Table 8](https://arxiv.org/html/2602.06036v2)
- **[claimed]** "large blocks can increase verification cost under compute-bound settings (e.g., large batch sizes); reducing the block size in such cases can therefore yield better overall speedup." Dynamic block-size scheduling is left as future work. — [DFlash paper v2, §5.5.4](https://arxiv.org/html/2602.06036v2)
- **[claimed]** The DFlash2 blog and model card use block 8 (7 drafts) for Qwen3.8-27B. They report no smaller-block results and make no block-size recommendation for high concurrency. oMLX uses a runtime block size of 5. — [inco.ai DFlash2 blog](https://inco.ai/blog/dflash2/); local mirror of the card `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_README.md`
- **[claimed]** z-lab README (MLX only): "For quantized targets or drafts, use `block_size <= 5`: MLX's current quantized matmul kernel becomes less efficient at larger verify widths." — [z-lab/dflash README](https://github.com/z-lab/dflash)
- **[measured, third party]** A draft budget of 2 crashes an SGLang DFlash server at CUDA-graph capture, with an error that misleadingly blames FP4. Budgets 4, 6 and 8 work. — [model-serving-minefield #105](https://github.com/Blackwellboy/model-serving-minefield/issues/105) (from a search summary; not opened in full)
- **[claimed]** SGLang never validates the draft layout (1+N vs anchor-first), and an anchor-first checkpoint is silently shifted by one position. The width flag is honoured with only a warning. The fix PR #40203 is open and unmerged. — [sglang #40144](https://github.com/sgl-project/sglang/issues/40144)
- **[measured, ours]** Our checkpoint is evidently read with the correct layout: DFlash2 at W=8 reaches accept length 4.2–4.95 on L4 (an anchor-shift bug gives about 1.06). — `E11R/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md`

### Inferences
- **[inference]** Because the mask block is bidirectional, at W=4 the slots at positions 1–3 attend only to `[bonus, m1, m2, m3]` plus context, not to `m4..m7`. Their top-16 candidates, and the selector's path through them, can therefore differ from the first 3 positions at W=8. In expectation the difference should be small: paper Table 8 shows a 2–4 % τ loss for "trained large, run smaller", relative to a model trained at the smaller size.
- **[inference]** Width 4 or 6 is valid and supported. Width 3 should work (it was not tested in the third-party report). Width 2 is reported to crash. Expect the mismatch warning in the logs; it is benign.

### Gaps
- No direct measurement exists anywhere of a Qwen3.8-27B DFlash2 checkpoint (trained at b8) run at b4 or b6 in SGLang, so the per-position change caused by the shorter mask block is unquantified.
- The inco.ai blog does not state the training block size. It is inferred to be 8 from the config.

## Q2. Per-position acceptance and expected acceptance length at widths 3, 4, 6 and 8

### Takeaway
No vendor publishes per-position acceptance for Qwen3.8-27B DFlash2. The "positions 4–7 usually 0.1 or less" line is an anecdote from two community users on unspecified workloads, and it **cannot** hold for our workload: we measured a median accept length of 4.55 at W=8, and if positions 4–7 were each ≤ 0.1 the mean could not exceed 4.4. A geometric model fitted to our measurement (α ≈ 0.83) gives L3 ≈ 2.52, L4 ≈ 3.09, L6 ≈ 3.95, L8 = 4.55. GPT's α = 0.811 model gives L4 3.00, L6 3.79, L8 4.30. Both are plausible central estimates. If DFlash's heavy "whole block accepted" tail also holds for DFlash2, truncation hurts more than the geometric model says: the lower bound is L4 ≈ 2.52 and L6 ≈ 3.54.

### Cited Findings
- **[claimed / anecdote]** incoai discussion #9, "Shall I reduce the num of spec tokens?" The original poster wrote: "Positions 4-7 usually get around 0.1 or less." A second user wrote: "Positions 4–7 usually stay around 0.1 or lower… Reducing the number of speculative tokens to 3 makes dflash perform similarly to MTP." A third user runs 4 and gets 60–75 % with coding. No maintainer replied, and no engine or concurrency is stated. — [incoai/Qwen3.8-27B-DFlash2 discussion #9](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/9)
- **[measured, third party, llama.cpp, 1 stream, RTX 3090, about 176k context]** n_max 4: 56.96 tok/s, acceptance 72.4 %, mean length 3.90. n_max 5: 57.22 tok/s, 68.38 %, 4.42. n_max 7: 54.94 tok/s, 64.45 %, 5.51. "n_max=4-5 was the sweet spot". — [z-lab/Qwen3.8-27B-DFlash2 discussion #9](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/9)
- **[measured, vendor, Qwen3-4B DFlash, GSM8K]** Recall@1 by position 0–6: 85.4, 80.3, 79.4, 78.3, 77.5, 75.9, 72.9 %. DFlash2 on Qwen3.5-4B MATH-500 holds 85.0–86.5 % from position 2 to 14. The two-tap conv "recovers most of what ten extra layers buy." Qwen3.8-27B mean acceptance (blog Table 4): DFlash2 4.80, MTP 4.28, DSpark 3.62. — [inco.ai DFlash2 blog](https://inco.ai/blog/dflash2/)
- **[measured, vendor, H200, b8, xhigh, 4096 max tokens]** Acceptance length for DFlash2 / MTP / DSpark: GSM8K 5.46 / 5.02 / 4.36; MATH-500 5.28 / 4.72 / 3.92; HumanEval 4.39 / 3.91 / 3.30; MBPP 4.79 / 3.99 / 3.51; MT-Bench 4.10 / 3.74 / 3.01. — local card mirror `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_README.md`
- **[measured, paper]** On Math500 the block-8 DFlash model "frequently fully accepts entire blocks (35.7%)". This is a heavy tail, i.e. acceptance at successive positions is correlated. — [DFlash paper v2, §5.5.4](https://arxiv.org/html/2602.06036v2)
- **[measured, ours, 4xL4 TP4, INT4 W4A16, fp8 KV, W=8]** Accept length 4.31 / 4.51 / 4.85 / 4.62 / 4.78 at c = 1 / 4 / 8 / 12 / 16 (fp16 SSM). Long runs of 12 × 24,576 tokens: 4.20. ARC sanity puzzles: 3.77–5.50 per puzzle. — `E11R/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md`
- **[measured, ours, production ARC solver, 11 running]** "Accept length median 4.55". — `E11R/E11-05 run1 analysis (2026-10-18).md`
- **[code, ours]** SGLang logs `accept len` and `accept rate`, where accept rate = (accept_len − 1)/(W − 1) (e.g. 3.33 → 0.33 at W=8). It logs no per-position histogram. — `E11R/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/server_D_C_int4_dflash2_ssm_fp16_try0.log`

### Inferences
- **[inference]** Expected acceptance length L_W = 1 + Σ_{k=1}^{W−1} s_k, where s_k is the probability that the first k drafts are all accepted. These assume truncation keeps the same predictions (see Q1), and were computed with a short script:

| Model | L3 | L4 | L6 | L8 |
|---|---|---|---|---|
| GPT geometric α=0.811 | 2.47 | 3.00 | 3.79 | 4.30 |
| Geometric fitted to our measured 4.55 (α=0.8295) | 2.52 | 3.09 | 3.95 | 4.55 |
| inco Table 1 per-position recall treated as conditional (Qwen3-4B) | 2.54 | 3.08 | 3.84 | 4.27 |
| Lower bound: flat survival (all-or-nothing blocks), L8=4.55 | 2.01 | 2.52 | 3.54 | 4.55 |
| "Positions 4–7 ≤ 0.1" anecdote | ≈L8 − ≤0.4 | — | — | ≤ 4.4 (≈3.3–3.5 realistic) |

- **[inference]** Which model is more realistic? For our workload, the geometric fit to our own measurement (α ≈ 0.83) is the best central estimate. GPT's α = 0.811 is close but underestimates our L8 (4.30 vs 4.55). The "0.1" anecdote is inconsistent with our measurement and describes low-acceptance (likely agentic/coding) workloads. Correlation between positions (DFlash's 35.7 % full-block acceptance) pushes reality toward the flat lower bound, so a narrower width loses somewhat more than geometric predicts. A realistic L4 is about 2.8–3.1 and L6 about 3.7–3.95.
- **[inference]** Throughput model calibrated on our own c=12 measurements: AR 289.4 tok/s, i.e. a 41.5 ms step for 12 tokens; DFlash W8 403.7 tok/s at L = 4.62, i.e. about 137 ms per step. With step(W) = a + 12·b·W + d·W/8 and drafter cost d ∈ [10, 40] ms, predicted total tok/s at c=12 for W = 3 / 4 / 6 / 8:
  - geometric fit: 418–434 / 435–446 / 427–431 / 399
  - flat lower bound: 334–347 / 355–364 / 382–385 / 399

  So W=4 ranges from about +10 % (geometric) to about −10 % (fully correlated tail), and W=6 from about +7 % to about −4 %. The linear-in-W cost assumption favours narrow widths. If the verify forward at 48 tokens per step is memory-floor-bound on L4 (W4A16 Marlin), the narrow-width gain shrinks. **The sign must be measured, not assumed.**

### Gaps
- There is no per-position acceptance data for Qwen3.8-27B DFlash2 from any official source, and SGLang 0.5.21 does not log it. A real survival curve would need a patch or log parsing of per-step `accept_length` distributions (e.g. `spec_trace_dir` or request `meta_info`; not verified).
- There is no measured L4/L6 for this checkpoint on our ARC prompts.

## Q3. Dynamic or adaptive verification width (SGLang #36127, `--speculative-adaptive`)

### Takeaway
Nothing in SGLang 0.5.21 or 0.5.22 adapts the DFlash/DFlash2 verify width to batch size or confidence. #36127 / PR #36136 (DFlash2 confidence-based dynamic verify) is open and unmerged. `--speculative-adaptive` is EAGLE/EAGLE3-only by code, and its own defaults shrink drafts sharply at batch ≥ 8. DSpark in 0.5.21 has a ragged-verify planner, but it needs an offline-profiled SPS cost table, without which it verifies everything. v0.5.22 adds opt-in *block verification* for DFlash, which is a no-op for greedy and gives about +1.8 % at c8.

### Cited Findings
- **[claimed]** #36127 "[Feature] Dynamic verification for DFlash2": open, no comments, opened 2026-08-24. It proposes DSpark-style cross-request verify budgets driven by per-token survival estimates. Its only linked PR is #36136 (open). — [sglang #36127](https://github.com/sgl-project/sglang/issues/36127)
- **[measured, author, Qwen3-4B, H20, TP1, greedy]** PR #36136 (`DFLASH_CONFIDENCE` mode) is open, not merged, with CI failing and no review. Throughput change vs fixed width: c1 −3.6 %; c4 −7 to −8 %; c8 −1.7 to −6.3 %; **c12 −0.2 / +4.6 / +10.8 %; c16 +0.3 / +0.7 / +12.8 %**; c32 +4.9 / +9.6 / +20.2 % (GSM8K / HumanEval / MT-Bench). Accuracy is unchanged. The cost table must be re-profiled for each hardware setup. — [sglang PR #36136](https://github.com/sgl-project/sglang/pull/36136)
- **[code]** `adaptive_unsupported_reason()` returns unsupported unless `speculative_algorithm in ("EAGLE","EAGLE3")`, so DFLASH is excluded. Default adaptive candidate steps by batch size: bs1 `[1,3,5,7]`, **bs8 `[0,1,3]`**, bs32 `[0,1]`, bs64 `[0]`. — `SRC/speculative/adaptive_spec_params.py` lines 26–63
- **[measured]** #23330: `--speculative-adaptive` with MTP on hybrid GDN Qwen3.5 crashed with `'RadixLinearAttention' object has no attribute 'is_cross_attention'` because the MTP draft mutates a shared config. The fix PR #23331 has unknown merge state. The issue was closed by the inactivity bot on 2026-06-22. — [sglang #23330](https://github.com/sgl-project/sglang/issues/23330)
- **[code]** DSpark in 0.5.21 has `--speculative-dspark-sps-table-path`: "Omit for an uninitialized flat constant-SPS table: the budget degenerates to verify-all (zero throughput gain by itself)". It also has an optional STS confidence calibration ("losslessness is unaffected either way"). — `SRC/arg_groups/fields/spec.py` lines 77–100; `SRC/speculative/dspark_components/dspark_planner.py`, `dspark_sps.py`
- **[measured, PR author]** v0.5.22 #43012 "Support block verification for the DFlash family" (merged 2026-10-08): opt-in `--speculative-use-block-verification`, applied only to sampled paths. For greedy DFlash drafts it logs that the flag has no effect. On Qwen3.8-27B DFlash2, AIME25, GB300, c8: accept 4.009 → 4.142 (+3.3 %), tok/s +1.76 % pooled, per seed +2.75 / +2.67 / −0.29 %. — [sglang PR #43012](https://github.com/sgl-project/sglang/pull/43012); [v0.5.22 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)

### Inferences
- **[inference]** SGLang's own EAGLE adaptive defaults (≤3 steps at bs8, ≤1 at bs32) and PR #36136's gains appearing only at c ≥ 12 both support the DFlash paper's statement that verification becomes compute-bound at batch sizes like ours, so narrower verification helps there. Neither is DFlash2-27B-on-L4 evidence.
- **[inference]** For us the only available "dynamic width" is a static width chosen per run. Block verification (v0.5.22) is worth a small gain for sampled (temperature 1.0) decoding, but it requires upgrading to 0.5.22.

### Gaps
- #36136 has no 27B or L4 results, and confidence-head checkpoints for Qwen3.8-27B DFlash2 are not published (it can fall back to selector-lattice scores).

## Q4. Correctness issues: status, details, affected versions, fixes

### Takeaway
As of 2026-10-23, every DFlash2 correctness issue listed in the task is **still open with no merged fix**. #28180 is the exception: closed, fixed in PRs. The most serious one, #36548 (answer-level corruption under concurrency on Qwen3.8-27B), traces to GDN TARGET_VERIFY vs DECODE numeric drift (#35150). Its partial fix (#36014) is unmerged. Width 4 cut the wrong-answer rate about 3× (111 → 37 of 304) but did not remove it. v0.5.22 (2026-10-19, the newest release) contains adjacent hardening (NaN isolation, mamba-state boundary, compact-verify graph tokens) but none of these issues' fixes.

### Cited Findings
- **#36548 "[Bug] DFlash2 state corruption with concurrent requests"**. Open, opened 2026-08-26, no linked fix.
  - **[measured, gabrielbryk, DGX Spark, NVFP4 target, fp8 KV, bf16 mamba, 8 draft tokens, max 8 running, greedy ordering prompt]** DFlash8 serial 0/100 wrong. DFlash8 at c8 7/100 wrong, and **111/304 (36.5 %)** on the expanded set. No-spec at c8 **0/304**.
  - Each single ablation (fp32 SSM, `--disable-decode-cuda-graph`, **draft depth 4**, the #36014 beta hunk) brought it to **37/304**, McNemar p ≤ 8.3e-12. None fixed it.
  - At bs1, width-one TARGET_VERIFY with all drafts rejected diverges from plain decode in 19/32 cases. The first differing value is the selected-step SSM transition output.
  - A dense BF16 target lm_head brought it to 1/304, with drift still present: "useful in practice but not a fix."
  - Source: [sglang #36548](https://github.com/sgl-project/sglang/issues/36548)
- **#35150 "Qwen3.8 DSpark forced-reject is not lossless: accumulated GDN state drift vs Base decode"**. Open.
  - **[measured]** With gamma=1 and drafts forced to reject, the first divergence is at token 13. Beta alignment moves it to 23, fp32 SSM to 38, BF16 instead of FP8 KV to 39. The drift accumulates over verify/commit rounds.
  - Source: [sglang #35150](https://github.com/sgl-project/sglang/issues/35150)
- **PR #36014 "[Fix] Align GDN target-verify beta semantics with packed decode"**. Open, not merged, blocked at CI.
  - Packed decode rounds `sigmoid(b)` to BF16; target-verify kept it in FP32. With the fix the unit test is bitwise equal.
  - It "does not claim to resolve all Qwen3.8 or speculative-decode numerical divergence".
  - Source: [sglang PR #36014](https://github.com/sgl-project/sglang/pull/36014)
- **#38009 "[DFlash2] Greedy output diverges from target-only Qwen3.8-27B when thinking is enabled"**. Open, opened 2026-09-04, no linked PR.
  - **[measured]** Setup: 4×3090 TP4, BF16 target and draft, block 8, BF16 KV, CUDA graphs, overlap and radix all disabled. Thinking on: first difference at token 30. Thinking off: all 64 tokens match. Target-only A/A: identical.
  - A commenter on GB10 with NVFP4 and FP8 KV found that output is byte-identical within one process, but diverges across container restarts. The divergence correlates with accept length, and block size 8 vs 16 changes the output.
  - Source: [sglang #38009](https://github.com/sgl-project/sglang/issues/38009)
- **z-lab/dflash #159** (MLX, no SGLang involved). Open, no maintainer reply.
  - **[measured]** Greedy DFlash2 diverges from AR at token 30, at both block 5 and block 8.
  - The reporter asks whether the card's "greedy output matches the target model exactly" or the MLX README's allowance of divergence is the intended contract.
  - Source: [z-lab/dflash #159](https://github.com/z-lab/dflash/issues/159)
- **#37475 "DFlash target_layer_ids fallback guesses wrong tap layers; count collision fails silently"**. Open, fix PR #37476 open.
  - If `target_layer_ids` is missing, the fallback formula for Qwen3.8-27B-DFlash2 gives `[1,16,31,46,61]` instead of the trained `[5,19,33,47,61]`. The counts match, so it loads silently and acceptance collapses.
  - It is dead code when the key is present. **Our config sets the key** (`[5,19,33,47,61]`), so we are not affected.
  - Sources: [sglang #37475](https://github.com/sgl-project/sglang/issues/37475); `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_config.json`
- **#39087 "Quantized DFlash2 draft silently yields ~0% acceptance"**. Open, opened 2026-09-11 on v0.5.19, PR #39254 open.
  - **[measured]** W4A16 drafter: accept 1.03 at 38 tok/s. BF16 drafter: 3.71 at 171 tok/s.
  - **When `--speculative-draft-model-quantization` is unset, it defaults to the target's quantization, which also reproduces ~0 % acceptance with a BF16 drafter.** The `unquant` flag avoids this.
  - **[measured, ours]** Our accept length of 4.2–4.95 shows we are not affected.
  - Sources: [sglang #39087](https://github.com/sgl-project/sglang/issues/39087); `E11R/.../e11_03/report.md`
- **#28180 "[DFlash] Infinite loop when using repetition_penalty"**. Closed as completed on 2026-08-17 (also auto-labelled inactive).
  - Cause: `prepare_for_decode` skipped `cumulate_output_tokens`, and DFlash applied only additive penalties, never scaling ones.
  - Fixes: PRs #28200 / #28535 (Spec V2), plus fork commit dcd0071 for additive penalties. The release that contains them is not identified.
  - Source: [sglang #28180](https://github.com/sgl-project/sglang/issues/28180)
- **#43514 "DFlash + Qwen3.8-27B: accept rate drops to 0 during serving"**. Open, opened 2026-10-10, affects v0.5.19 and v0.5.20.
  - Trigger: grammar-constrained requests (`tool_choice: required`) plus multi-chunk prefill admissions with `--enable-mixed-chunk`, at `--speculative-num-draft-tokens 6`.
  - Symptoms: the whole batch emits token 0 `"!"`, accept rate 0.00, accept length 1.00, with no error.
  - Related PR #43518 "Clear stale grammar masks before DFlash and DSpark verification" is open and unmerged.
  - Sources: [sglang #43514](https://github.com/sgl-project/sglang/issues/43514); [sglang PR #43518](https://github.com/sgl-project/sglang/pull/43518)
- **Newer releases**
  - v0.5.22 (19 Oct) is the latest; nothing newer exists. It does not mention 36548, 38009, 37475, 37476, 39087, 39254, 28180, 43514, 43518, 36136, 35150, 23330 or 36014.
  - DFlash-adjacent fixes in it:
    - #41736 keeps grouped-conv taps inside each request's block "so NaN/Inf cannot leak across requests"
    - #41737 keeps the DFlash2 greedy selector walk in range for all-NaN rows
    - #42709 fixes wrong tokens from compact-verify CUDA graphs
    - #43029 fixes hybrid models resuming "from wrong mamba states when speculative decoding and structured output were used together"
    - #42651 sizes the draft KV pool after capture
    - #43012 adds block verification
  - Caveat: the release notes were truncated after about 100k characters.
  - Sources: [v0.5.22 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.22); [releases list](https://github.com/sgl-project/sglang/releases)
- **[measured, ours, 0.5.21, INT4, fp8 KV, W=8]**
  - Greedy DFlash2 vs target-only: 1/4 identical, mean common prefix 155 of 512.
  - Greedy MTP vs target-only: 0/4 identical, prefix 152.
  - fp16 vs fp32 SSM: 1/4, prefix 166.
  - **"contam" (greedy DFlash2 alone vs inside a 16-batch): 0/4 identical, mean common prefix 15.5 of 512.**
  - There was **no target-only batch-invariance control**.
  - Source: `E11R/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md`

### Inferences
- **[inference]** Our stack has the #36548 ingredients: Qwen3.8 GDN hybrid, DFlash2 W8, concurrency, fp8 KV, a reduced-precision SSM state (fp16, where #36548 used bf16) and BF16 beta. The c16 "contam" common prefix of 15.5 tokens fits batch-composition sensitivity. Without a target-only control we cannot tell whether that is excess over normal AR batch non-invariance.
- **[inference]** #36548 matters at the answer level (36.5 % wrong on an easy greedy probe, vs 0 % without spec), not just at the token level. It is the main correctness risk for a 60k-token reasoning workload, where drift has many steps to compound. Its ablations suggest width 4 and fp32 SSM each cut the error about 3×. The dense lm_head result suggests the risk rises when the target's lm_head is quantized. **[measured, local config]** The RedHat W4A16 `quantization_config.ignore` list contains `lm_head`, `re:^mtp.*` and every `language_model.layers.N.linear_attn` module (with its norm, `in_proj_a` and `in_proj_b`). So our target's lm_head and GDN layers are unquantized BF16, the same condition that brought #36548 down to 1/304 (`D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures/tok_int4/config.json`). The #36548 rate is therefore probably lower on our stack than on NVFP4, but it is not proven to be zero: the drift was still there in that arm.
- **[inference]** #43514 is relevant only if we use grammar/tool-constrained requests together with `--enable-mixed-chunk`. Our E11-03 server ran with `enable_mixed_chunk: False`. #28180 is relevant only if we set repetition_penalty. #37475 and #39087 do not affect our config.

### Gaps
- No maintainer has stated whether DFlash2 + GDN is expected to be bit-exact against target-only decode.
- The effect of width 4 on #36548 has been measured on only one setup (GB10, NVFP4).

## Q5. Correctness gate design (preflight)

### Takeaway
Exact token equality against target-only cannot serve as a pass/fail test for 60k-token thinking on this stack. Divergence by token 15–155 is routine (ours, #38009, z-lab #159), and AR itself is not batch-invariant. Practitioners use (a) greedy equality on short prompts as an algorithm check, plus (b) statistical answer-level equivalence against a target-only arm under the same concurrency. #36548's protocol is a good template: one easy greedy probe with a single correct answer, repeated hundreds of times at c8, with McNemar's test. The gate should compare against A/A baselines, not against zero.

### Cited Findings
- **[claimed]** vLLM: decoding is "theoretically lossless up to the precision limits of hardware numerics". It is validated by a rejection-sampler convergence test and a "Greedy sampling equality" test. vLLM "does not currently guarantee stable token log probabilities", and "Changes in batch size may cause variations in logprobs and output probabilities". — [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- **[claimed]** SGLang deterministic mode (`--enable-deterministic-inference`, batch-invariant ops) covers FlashInfer, FA3 and Triton backends. Its test `python3 -m sglang.test.test_deterministic --test-mode single|prefix|radix_cache` should show "Unique samples: 1". Nondeterminism comes mainly from batch size changing reduction order. The page says nothing about speculative decoding or hybrid models. — [SGLang deterministic inference docs](https://docs.sglang.io/advanced_features/deterministic_inference.html); [LMSYS deterministic blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- **[measured]** #36548's protocol: a greedy four-item ordering prompt; serial vs c8; no-spec vs DFlash; 100–304 trials per arm; wrong-answer counts; McNemar significance. Wrong answers looked like normal completions: the reasoning states the constraints correctly but the final answer is wrong. — [sglang #36548](https://github.com/sgl-project/sglang/issues/36548)
- **[measured]** #38009's protocol: compare `logprobs.content[*].token` (token IDs, not text), with a target-only A/A repeat and a thinking-off control. A commenter adds: run A/A across fresh server starts for each profile before the A/B, and log accept length per request from `meta_info`. — [sglang #38009](https://github.com/sgl-project/sglang/issues/38009)
- **[measured]** First-divergence positions reported: our DFlash2 and MTP vs target-only ~152–155 tokens (mean); DFlash alone vs in a batch 15.5; #38009 token 30; z-lab #159 token 30; #35150 tokens 13–39 depending on dtype. — `E11R/.../e11_03/report.md`; [#38009](https://github.com/sgl-project/sglang/issues/38009); [z-lab #159](https://github.com/z-lab/dflash/issues/159); [#35150](https://github.com/sgl-project/sglang/issues/35150)
- **[measured]** Silent-failure signatures to monitor:
  - accept rate 0.00 / accept length 1.00 with token-0 runs ([#43514](https://github.com/sgl-project/sglang/issues/43514))
  - accept length ≈ 1.03 from a quantized or mis-quantized drafter ([#39087](https://github.com/sgl-project/sglang/issues/39087))
  - accept length ≈ 1.06 from a layout shift ([#40144](https://github.com/sgl-project/sglang/issues/40144))
  - `accept rate: 1.00` with never-ending output from repetition_penalty ([#28180](https://github.com/sgl-project/sglang/issues/28180))

### Inferences (proposed protocol, [inference])

All arms use the same build and flags as production: INT4, fp8 KV, fp16 SSM, TP4, CUDA graphs, radix cache, and the same `max_running_requests`. Arms:
- **T**: target-only, no speculation
- **D8**: DFlash2 W=8
- **D4**: DFlash2 W=4
- optionally **D6**

Steps:
1. **Health (about 2 min per arm).** Smoke-generate, then parse `accept len` lines.
   - Fail if any 40-step window shows accept rate 0.00.
   - Fail if the mean accept length is below 3.5 at W8, below 2.5 at W4, or below 3.2 at W6.
   - Fail if any output has a run of more than 50 identical token IDs.
   - Confirm the expected `DFLASH block size mismatch` warning appears when W ≠ 8.
2. **Short exact-match algorithm check (thinking OFF, greedy, 20 prompts × 256 tokens, serial).**
   - Record token IDs with `logprobs` and `top_logprobs=2`.
   - Compare D vs T; compare T vs T across a server restart (A/A).
   - At each first-divergence point, compute the T top-1 vs top-2 logprob margin. A margin under about 0.1 nat counts as a numeric near-tie (normal). **A divergence where T's margin is over 1 nat counts as a bug flag.**
   - Pass if the D-vs-T exact-match rate is within the A/A rate and there are no large-margin flips.
3. **Answer-level probe (the #36548 template).** Use 2–3 short greedy prompts with a single verifiable answer (e.g. an ordering or small grid transform, thinking on, ≤ 2k tokens).
   - Run 150 trials per arm at c=12, with the batch filled by real ARC requests of realistic length, plus 50 serial trials.
   - T at c12 must be essentially 0 wrong.
   - **Fail if D's wrong rate exceeds 2 % or is McNemar-significant vs T (p < 0.01)**, measured against #36548's 0/304 vs 111/304.
   - This is the step most likely to reveal #36548 on our stack, and to show whether W4 helps.
4. **Workload equivalence (32 fixed ARC eval prompts, thinking on, about 60k cap, c=12, sampling as in production with a fixed `sampling_seed`).**
   - Compare the solve rate and the train-pass rate of D vs T.
   - Pass if the difference is within binomial noise (with n=32, about ±2 tasks at 1 sd).
   - Log the first-divergence index distribution for information only. Token equality over 60k tokens is not expected.
   - This step is about 1 GPU-hour per arm at 12 concurrency. A 20-prompt subset can be used if time binds (an inference from our measured ~330–400 tok/s).
5. **Batch-invariance control.** Run T alone vs T inside a 16-batch (the "contam" test with an AR arm) so the existing DFlash contam result (prefix 15.5) can be judged against AR's own non-invariance.

What counts as normal numeric noise vs a bug:
- **Normal:** token-level divergence at a random position, at a near-tie, at a rate similar to the A/A and AR batch-variance rates, with final answers statistically equal to T.
- **Bug:**
  - an answer-level error rate significantly above T (as in #36548)
  - large-margin flips
  - divergence that grows with concurrency in D but not in T
  - any silent-failure accept-length signature

### Gaps
- There is no published tolerance for "acceptable" divergence on GDN hybrids; #35150 asks for one and is unanswered.
- It is unverified whether `--enable-deterministic-inference` works with DFLASH and mamba on 0.5.21, so it cannot be relied on to make the gate exact.

## Q6. Alternatives (DSpark, MTP/NEXTN, DFlash2) at high concurrency; best width at batch 8–16

### Takeaway
On Qwen3.8-27B, DFlash2 beats MTP and DSpark on acceptance and throughput at every published concurrency. At c32 on an H200, MTP falls to 0.77–1.04× of AR while DFlash2 holds 1.01–1.45×. On our 4xL4, MTP 3/1/4 was 0.91× AR at c12, while DFlash2 W8 was 1.06× (fp32 SSM) and 1.39× (fp16 SSM). No source measures DFlash2 width at batch 8–16 for a 27B model. The indirect evidence, all pointing toward 4–6, is: the DFlash paper's compute-bound advice, SGLang's EAGLE adaptive defaults, #36136 gains at c ≥ 12, and llama.cpp sweet spots of n_max 4–5. The decision should come from an on-hardware A/B, with W=4 also reducing #36548 errors.

### Cited Findings
- **[measured, vendor, H200, b8 / 7 drafts for all methods]** Throughput vs AR, as MTP / DSpark / DFlash2:
  - c1: 1.96–2.59× / 2.00–2.69× / 2.67–3.43×
  - c8: 1.74–2.19× / 1.67–2.23× / 2.27–2.85×
  - **c32: 0.77–1.04× / 0.74–1.13× / 1.01–1.45×**

  Source: `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_README.md` (mirror of the [incoai card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2))
- **[measured, ours, 4xL4 TP4 INT4]** Total tok/s at c12:
  - AR: 289.4
  - MTP (3 steps, 4 draft tokens): 263.6 (0.91×), accept 3.11
  - DFlash2 W8, fp32 SSM: 307.3 (1.06×), accept 4.70
  - DFlash2 W8, fp16 SSM: 403.7 (1.39×), accept 4.62

  At c16 the same arms give 0.91× / 0.82× / 1.10×. In long runs (12 × 24,576 tokens): AR 282.4, MTP 262.3, DFlash2 fp32 258.3.

  Source: `E11R/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md`
- **[measured, ours]** Production ARC solver: SGLang INT4 + DFlash2 reached 328.9 tok/s at 11 running, vs vLLM FP8 + MTP-3 at 280.9 (16 running) and 254–256 (12–16 running). — `E11R/E11-05 run1 analysis (2026-10-18).md`
- **[code]** DSpark's verify window is `gamma + 1` (default gamma 7, so 8 slots), and `--speculative-num-draft-tokens` must equal gamma + 1. DSpark has an SPS ragged-verify planner, which needs an offline table. — `SRC/arg_groups/speculative_hook.py` lines 700–751; `SRC/speculative/dspark_components/dspark_config.py` (`DEFAULT_DSPARK_GAMMA = 7`)
- **[measured, vendor]** Qwen3.8-27B mean acceptance: DFlash2 4.80, MTP 4.28, DSpark 3.62. — [inco.ai blog](https://inco.ai/blog/dflash2/)
- **[measured, paper, B200]** DFlash speedup falls with concurrency, e.g. Qwen3-8B Math500 from 5.1× at c1 to 2.8× at c32; LLaMA-3.1-8B GSM8K from 2.4× to 1.6×. — [DFlash paper v2, Tables 3 and 5](https://arxiv.org/html/2602.06036v2)
- **[claimed]** LMSYS: the DFlash drafter beats MTP "in all of the settings we tested", concurrency 1–32. "Draft token/block counts selected for maximum throughput" (block 16 for DFlash). No per-width high-concurrency table. — [LMSYS DFlash + Spec V2 blog](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- **[measured, third party, 1 stream]** llama.cpp: n_max 4–5 beat 7. Community users run 4, and one says 3 "performs similarly to MTP". — [z-lab discussion #9](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/9); [incoai discussion #9](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/9)
- **[measured, author, Qwen3-4B]** Dynamic, narrower-on-average verification gains only from c12 upward (−0.2 % to +10.8 % at c12, up to +20 % at c32). — [sglang PR #36136](https://github.com/sgl-project/sglang/pull/36136)

### Inferences
- **[inference]** MTP/NEXTN is not a better alternative on our hardware: it was measured below AR at c12–16. DSpark has lower acceptance (3.62 vs 4.80) and its dynamic planner needs SPS profiling, so it is not an easy win either. DFlash2 stays the engine.
- **[inference]** Best width at c12 on 4xL4: the evidence favours trying **W=6 and W=4 against W=8**.
  - The model predicts W4 between −10 % and +12 % and W6 between −4 % and +7 % (Q2 table), so the expected value is mildly positive, but unproven.
  - W4 has a measured side benefit: about 3× fewer #36548 wrong answers in that report's setup.
  - W3 is predicted to be no better than W4 (L3 ≈ 2.5) and adds risk. W2 crashes.
  - Suggested A/B: identical recorded ARC prompt mix at a real 12 running (assert running == 12 from the logs), about 10–15 min steady-state per width. Record total tok/s and accept length by context band, plus the Q5 step-3 answer probe at W8 vs W4.
- **[inference]** A narrower width also shrinks the per-request KV reserve (2 × W) and the verify workspace, freeing a little memory toward the 12-slot cap.

### Gaps
- There is no published or measured DFlash2 throughput at W ≠ 8 for any 27B target at batch 8–16, on any GPU.
- There is no L4-specific data on how the W4A16 Marlin verify GEMM cost scales between 48 and 96 tokens (12 × W), which decides the sign of the W4 change.
