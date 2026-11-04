# andy2709 / arc-flash-next-w4a16-inducer: notes (read in full, 2026-11-03)

The source is a script kernel (1,470 lines), last run 2026-10-31.
- Copy: `andy2709_arc-flash-next-w4a16-inducer.py`.
- The harness (codex_lite.py, kaggle_runner.py, ...) is in the public dataset `andy2709/arc-flash-next-inducer-runtime`. Not read yet.

## Engine
- **Server:** SGLang 0.5.20, heavily patched (flash_next_serving.py). Model W4A16 AutoRound (dfranzen), REAP pruned to 480/512 experts, GPTQ INT8 dense layers.
- **Layout:** TP4 + EP4, with decode context parallel DCP2 (each KV head sharded over a pair: ≈1.8× tokens).
- **Memory:** FP8 KV; KV cap 331,776 rows a rank (663k tokens a pair); mem-fraction 0.92.
- **Concurrency:** MAX_RUNNING_REQUESTS = 10, i.e. only 10 running. Context 139,264; page 64; prefill chunk 512.
- **Spec decoding:** NEXTN MTP, 3 steps / 4 draft tokens.
  - The draft is refit to the pruned target: AL 2.807 → 2.992.
  - A hot-token map restricts the draft vocabulary; the draft shares the target's vocab.
- **Kernels and patches, with their measured gains:**
  - host all-reduce: neutral on Kaggle;
  - fused MoE down + all-reduce: -4.9 to -6.4 %;
  - HC mix CUDA: -4.6 / -5.1 %;
  - HC local up INT8: decode -4/-5 %, prefill -18 %;
  - top-32 logits all-gather: -3.6 %;
  - breakable prefill CUDA graph: -6.5 %;
  - CUDA QSA indexer decode kernel: 2.46 ms per step saved at 10 running;
  - host embed + PLE offload; mamba radix cache extra_buffer_lazy (4 slots a request).
- **Boot:** a pre-built compile cache (a separate cache kernel) is restored, and weights are staged from NFS-mounted Models.
- **Speed:** decode step ≈ 67-74 ms at 8-10 running with AL ≈ 3, so ≈ 400 tok/s. That is about our E28 level (419.5).

## Solver (codex_lite)
- **Settings:** a codex-style `exec_command` sandbox tool; reasoning_effort **medium**; temperature 0.7, top-p 0.95, top-k 20.
- **Token caps:** 65,536 per attempt, and since v27 a 12,288 output cap per request.
- **Logit bias:** -1 on "Hmm" and "Alternatively".
- **Images:** since v28 the grids are also shown as images (the vision tower is loaded).
- **Admission:** adaptive (grows while KV use ≤ 60 %). A running attempt averages ~37.5k tokens (p90 peak 92k).
- **Second attempts:** `--secondary-unverified-first`. Coverage comes first; second attempts go first to tasks whose first attempt did not verify, and carry that attempt's work.
- **Reported local public-120 scores:** 79.55 over 4 seeds, 81.11 (phi1, v31). The comments also cite "s2234 76.81 and 77.36"; these equal LB rows 11 (tascj) and 9.

## Leaderboard (top 20 at close, API)
- 1 Tufa Labs 88.06 (the Duck harness authors; above 85 %, so the Bonus Prize applies).
- 2 Yi-Chia Chen 87.22; 3 nvbanana 83.89; ... 15 okiek 75.69 (= writeup 755321); 19 Daniel Franzen 74.72.

## Against ours (E28 K100)
- **Speed is similar** (≈ 400 vs 419.5 tok/s), yet their public-eval score is ≈ 80 vs our 65.0 % task-norm.
- **The gap is token efficiency and the harness, not throughput:**
  - medium effort and a 65k cap (ours: xhigh, 104k / 120k, 60k thinking floor);
  - a 12k per-request cap and anti-hedging logit bias;
  - images;
  - second attempts on unverified tasks first.
- They run 10 streams with long contexts in 663k KV tokens. We run 24 streams.
