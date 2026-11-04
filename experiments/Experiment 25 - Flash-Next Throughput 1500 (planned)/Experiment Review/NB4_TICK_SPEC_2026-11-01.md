# NB4 TICK: where does the stage-0 gap come from, and FULL graph / commit graph / P5 A/B (spec, 2026-11-01)

User request (2026-11-01): build NB4.
1. Use a host timeline to find where the ~23 ms per tick of stage-0 wait comes from.
2. Run separate A/B arms for FULL graph, commit graph and P5.

## Evidence that motivates it (measured)

**NB1 R2 B1 trace** (`_scratch/e25_gpu_out/q27-xhigh-e25-nb1-capacity-r2/nb1/traces/profile_b1/*.pt.trace.json.gz`, 4 ranks):

| Stage | Critical path per micro-batch | GPU-busy union | Summed kernels |
|---|---|---|---|
| stage 0 | 50.6 ms | 27.2 ms | 29.5 ms |
| stage 1 | 31.6 ms | 23.3 ms | 25.8 ms |

- 16 verify tokens per micro-batch.
- The light stage is idle ≈ 27 ms per tick.
- Kernel time by category: NCCL 63.5 % (top kernel `ncclDevKernel_Broadcast_RING_LL`), Marlin MoE 16.5 %, other GEMM 9.0 %.
- Graph verdict PIECEWISE.

**NB2 SEED2 trace** (40 forwards): FULL 0, PIECEWISE 20, EAGER 20.

**NB1 R2 B2** (bundled change, oracle FAIL): RecoverSSM commit graph captured only n=16 at warm-up; the other steps ran staged-eager.

**NB3:** P5 host-staged world-2 all-reduce is 1.39-1.68× faster than NCCL for 5-123 KB and bit-exact vs an ordered fp32 reference. Microbenchmark only: concurrency is unverified and there is no engine integration.

**E23 (27B):** FULL graph G3 +2 % at 16 streams; Triton FULL −7 %; A/A noise ±1 %.

## Base configuration (every arm = base + exactly one change)

The W4 Merlin checkpoint at the NB1 R2 B3 settings, the config NB5 QUALITY is running now:
- FP8 E4M3 QSA KV, pin 3,553,580,524 B (214 blocks);
- MTP K3, `--language-model-only`, #60490 applied;
- seqs = NB5's 9; captures as in NB5 (CAPS16 list if seqs > 8).

BF16 is not used: NB5 FP8 solved 13/15 tasks.

## Arms (one boot each, same seeds and prompts)

| Arm | Change | Notes |
|---|---|---|
| T0 | Base + full host timeline | See the timeline section below |
| T0b | Base again, no profiler | A/A repeat, so the noise floor is measured in this session |
| G | FULL graph coverage | The exact change is decided by the source research: why decode ran 0 FULL forwards (captured sizes vs PP micro-batch token counts (C/2)·(1+K), the draft model's graph mode, uniform-batch rules) |
| C | `VLLM_RECOVERSSM_COMMIT_GRAPH=graph` with commit capture sizes that cover the real n | Nothing else changed (B2 bundled it with SSM bf16) |
| P | P5 world-2 host-staged all-reduce inside the engine's TP all-reduce | Fail closed: if it cannot be wired in, the arm is `P5_NOT_INTEGRATED` and is skipped, not faked |

### Timeline in T0 (once, then the profiler is off)
- torch profiler on all 4 workers with CPU + CUDA activities and `record_shapes` off.
- The engine-core / scheduler process timing: per-step scheduler time, the time between steps, and the async PP micro-batch bookkeeping, from vLLM's own step timing logs or an env-gated timer.
- The gap analysis splits each stage-0 tick into:
  - GPU busy;
  - NCCL broadcast / recv wait (time spent inside NCCL kernels while the peer stage is still computing);
  - host launch gap (GPU idle while the host is preparing or launching);
  - scheduler / IPC wait (GPU idle with no host activity in the worker, waiting for the next step from the engine core).
- The analysis module is validated offline on the real NB1 R2 B1 traces before GPU.

### Per arm
- Boot and read-back (the change must be visible: graph-mode lines, commit-graph capture lines, P5 init lines).
- Greedy oracle vs base, and TF 2k / 8k / 40k vs base: correctness gate EXACT or NEAR_TIE per the NB0 / NB1 rules.
- 8k@8 and 40k@8 steady windows (90 s): tok/s, tick ms, accept length per position, running == level, 0 preemptions.
- A short profiled window (graph verdict, per-stage critical path vs busy) on T0, G, C and P.

### Gates and labels
- An arm is a speed winner only when:
  - its correctness gate passes;
  - its tok/s beats base by more than the measured A/A spread (and > 2 % absolute);
  - its trace shows the intended mechanism (e.g. G: FULL forwards > 0).
- Labels: `WIN / NO_GAIN / SLOWER / CORRECTNESS_FAIL / NOT_INTEGRATED / SKIPPED_BUDGET`.
- Speed numbers are engine end-to-end; microbenchmark numbers are never counted (PLAN §0).

## Budget
- Hard stop 150 min.
- One boot ≈ 7-10 min ready + 10-12 min measuring.
- Order: T0 → G → C → P → T0b. T0b goes last so drift over the session is visible. If time is short, T0b moves to after C.

## Rules (standing)
- Dual-mode CELL 0 (rerun = fallback submission only).
- Every request saved in full.
- Fail-fast on wasted-GPU risks.
- Blocker = wrong raw data / mid-run crash / wasted GPU.
- Notebook < 400 kB; payloads in sha-checked datasets.
- CPU kit + flat audit zip in `Experiment Review/Audit Zips/` before any GPU push.
- The GPU push needs the user's go.
