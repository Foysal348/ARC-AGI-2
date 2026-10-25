# 25 — New speed experiment (E15) design, from the 4-day speed review (Claude, 2026-10-24)

**Inputs:**
- `Notes/Speed review 2026-10-24/`, four parts from four agents. Every source file was read in full:
  - part A1: 4xL4 research report, overheads, profiling, NCCL;
  - part A2: agent throughput, width, versions, startup, token variants, TP2;
  - part B: the fastest-serving research;
  - part C: all measured runs plus the 12 conversation-insight files (250 numbers, 73 levers, 59 contradictions).
- GPT notes 01, 03, 04, 08, 09; agent notes 14, 15; the 10-23 Kaggle measurements (Notes/22); the discussions (Notes/24).

## 1. What the review settles

| Fact | Evidence |
|---|---|
| Best measured engine: W1 = TP4, INT4 + DFlash2 width 8, fp16 SSM, extra_buffer_lazy, mamba 56, mem 0.83 | E11-04: 416.5 tok/s at a real 12 running (458 / 413 / 388 by context band); E11-03 D 404 tok/s vs no-spec 289 (1.39×) |
| Communication ≈ 56 ms of the 115-133 ms step (42-49 %) | 10-23 NCCL bench: 345 µs × 140 + 8.0 ms all-gather |
| P2P / IPC paths are dead on this box | P2P 5.5× slower, PHB topology. This refutes every P2P lever in the docs (expected 40 → 8-12 ms) |
| 2×TP2 **with DFlash2** is nearly ruled out | ≈170k KV per replica (needs 700-920k in total); 2-3 long streams per replica; estimate 0.7-1.0× of TP4 |
| ReplaySSM is blocked **only for DFlash** in the 0.5.21 source | overheads note: saves 6-7 ms/step and 1.8-2 GB |
| MTP vs DFlash2 was never compared at a real 12 running with long context | E11-03: MTP capped at 10; 297 vs 356 tok/s at 8 running |
| Step time still has 11-28 ms unexplained | Needs the profiler (R2 A1-2) |
| Agent-side losses are as large as the engine levers | re-prefill (fix in R2), KV thrash (EVAL30, 2.1×), prefill 13-55 % of the window |

**Correction:**
- The SGLang flag set "MTP/EAGLE 4 drafts, chunked 2048, ssm fp32" was posted on 10-05 for NguyenThanhNhan's 23.33 pure-reasoning submissions.
- The config of his later 65.97 notebook is unknown. Notes/24 §A attributed the flags to the wrong submission.

## 2. What R2 (running now on hosen42) will already answer

R2 runs nodelay, #44401, graphs13, width 6/4, target-only, int8 Mamba, the profiler split and a Duck run with raw reasoning. **E15 does not repeat these.** E15 takes R2's numbers as its baseline and tests only what R2 cannot: the **MTP family** and the **layouts that MTP unlocks**.

## 3. E15 arms (engine only, no Duck; one notebook, about 75 min wall, about 2.5 quota-hours)

| Arm | Layout | Spec | What it tests | Adopt if |
|---|---|---|---|---|
| E15-0 | TP4 | DFlash2-8 (R2's adopted flags) | Reference in the same session (accept swings per boot) | — |
| E15-M1 | TP4 | **MTP** (NEXTN), steps 3, draft 4 | DFlash2 weight freed (~1.8 GB/GPU) → KV; verify payload halved → communication | Higher committed tok/s at 12 **and** KV ≥ the reference, gate PASS |
| E15-M2 | TP4 | MTP + **ReplaySSM** (only if the source/flag accepts it with MTP) | The 6-7 ms/step and 1.8-2 GB from the docs | M1 gate PASS + gain over M1 |
| E15-M3 | TP4 | Best MTP, **14 and 16 really running** (mrr 15/17, mamba sized) | Whether the freed memory buys real concurrency | Aggregate tok/s up, no retract / OOM |
| E15-T1 | **2×TP2** (GPUs 0-1, 2-3) | MTP | Two-GPU all-reduce with half the payload; KV per replica measured | Combined tok/s ≥ 1.2× the best TP4 at the same total streams (6+6 and 8+8), 40k band included |
| E15-N | NCCL bench only (no server) | — | `NCCL_PROTO=LL` / `LL128` / `Simple`, `NCCL_ALGO=Ring` / `Tree`, `NCCL_MIN_NCHANNELS` at 983 KB inside a graph | ≥ 20 % per-call gain → an env arm on the best engine |

**Every arm:**
- 8k and 40k windows at the measured real level;
- accept length plus `accept rate`;
- KV tokens and mamba slots read from the server log;
- retract / OOM signs;
- the correctness gate against the same-session A/A (large-margin flips decide).

**Order:** E15-N (5 min) → E15-0 → M1 → (M2) → M3 → T1. Budget: phases skip, never squeeze, with a hard stop.

## 4. Deferred, named so they are not lost

| Lever | Why later |
|---|---|
| Vocab-parallel logits sampling (−8 ms all-gather, ≈ +9-11 %, estimate) | Code patch + exactness test |
| GDN verify-kernel launch config | Only if R2's profiler shows GDN ≥ 15 ms |
| Mamba end-of-prefill checkpoint protection (≤ +14 %) | Needs R2's agent-replay classes first |
| Compile-cache restore across sessions (≈ 360-430 s per cold start) | Needs the C8 known-answer check; R2 saves the tar |
| Wheelhouse as a Kaggle Model (NFS) instead of a dataset (GCSFUSE) | Small; bundle with the next submission build |

## 5. Gates before the GPU (rule: Kaggle CPU harness)

1. Builder variant E15. The notebook stays < 400 kB; payloads go in the dataset.
2. Kaggle CPU harness passes (fake server: MTP flags, two servers on two ports).
3. The pre-run sheet: account, quota, wall time, stop time, sha. Push only on the user's go.
