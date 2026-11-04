# Speed levers for Flash-Next 1500 tok/s: measured, running, missing (document gap analysis, 2026-11-01)

The agent read all of Experiment 25 (PLAN_1500 v3.6, README, every file in Experiment Results and Experiment Review including Past work mining A1-A7, GPT research inputs 00-06 with their verifications, the NB4 research notes, the PR #53948 comparison, the tick spec, the NB4 audit prompt, PENDING_BATCH_FIXES, NB_BUILD_SPEC and STRUCTURAL_TRACK_FEASIBILITY), Exp 26 K3_ONLY v1/v2, and Exp 13 RC2 plus the FN 1500 research prompt.

"(est.)" marks the agent's own arithmetic on numbers from the files. Every other number is quoted.

## File keys
| Key | File |
|---|---|
| PLAN | Exp25 `PLAN_1500.md` (v3.6) |
| R-NB1 | `E25_GPU_NB1_R2_2026-10-31.md` |
| R-W1A / R-W1B | `E25_GPU_wave1_NB0_NB1` / `E25_GPU_wave1_NB3_NB2SMOKE` |
| R-NB5 / R-NB6 | `E25_GPU_NB5_QUALITY` / `E25_NB6_COMMIT_v1` |
| R-K100 / R-KIT3 | `E25_klee100_cpu_kit_v1` / `E25_NB3_CPU_kits_v1` |
| CTJ | `commit_turn_prefix_test_2026-10-31.json` |
| E26v1 / E26v2 | Exp26 `E26_K3ONLY_GPU_v1` / `v2` |
| RC2 / RP | Exp13 `E13_RC2_GPU_v1` / `FN_1500_RESEARCH_PROMPT` |
| A1-A7 | `Past work mining/A*.md` |
| G1-G3, V1-V3, AU1-AU3 | GPT research inputs and their verifications |
| ST | `STRUCTURAL_TRACK_FEASIBILITY_2026-10-31.md` |
| NR-G / NR-F / NR-C | NB4 research: graphs / folders / conversations |
| PRC / TS / AP | NB4 PR #53948 comparison / tick spec / NB4 audit prompt |
| PBF / SPEC | `PENDING_BATCH_FIXES.md` / `NB_BUILD_SPEC_2026-10-31.md` |

## 1. Top 5 missing levers (ranked by expected gain × confidence)
A lever counts as missing when it is not measured, not running, and not in the next steps, but there is evidence of a real gain.

Levers 1, 3 and 5 depend on NB4S. Once the stage-0 stall is removed, stage 1 sets the tick:
- stage-1 compute is 34.56 ms vs 23.54 ms on stage 0 (R-NB1:94, 108-119);
- the ideal overlapped tick is 34.6 ms (R-NB1:107);
- NR-F puts stage 1 at 41-42 ms vs 29 ms (NR-F:113-115).

**None of the running arms (S, QP, QD) touch stage 1.**

| # | Lever | Evidence | Expected gain | Cost / risk | Status |
|---|---|---|---|---|---|
| 1 | **INT8 draft LM head at K3** (`DRAFT_HEAD=int8`, K kept at 3) | Merlin on 4×3090: −6.5 to −7.2 % ms per token at the same acceptance (A3:72, 126, 307; RP:19; V1:53). On our NB2, the −7 % of K2+INT8 came only from losing the third position; per-position acceptance was 0.865/0.723 with INT8 vs 0.867/0.716/0.586 on the K3 control (R-W1B:61-63). One LM-head shard call costs 2.5 ms on L4 (R-W1B:32). The 3 draft graphs cost 12.4 ms per tick on stage 1 (NR-F:102). Vocab heads per tick: BF16 10.2 / int8 6.4 / int4 4.5 ms (A7:90, 219) | About −3.8 ms on the stage-1 tick, so **+8-11 % once stage 1 binds** (est.). About 0 while the stall binds | Config, 1 boot. Needs the GPU-side `_C.gptq_marlin_repack` preflight (R-KIT3:17-18) | Never measured alone; wrongly filed as a negative (NR-F:451; R-W1B:55-56). Only an optional NB4 arm (AP:75) |
| 2 | **More resident streams on klee100**: past 21, up to the resident bound of 24-29 | 16 to 24 streams at 20k: 460.4 to 562.0 tok/s (+22 %), tick 55.8 to 68.7 ms (E26v2:87-89). Resident bound 29/21/17/13 at 40k/64k/90k/120k (E26v2:79). 20k@32 and 40k@24 were skipped as KV-bound (E26v2:92), but KV max was only 0.84 at 40k@16, while live demand was about 0.53 (est.). Finished sessions stay retained, so the usage gauge is not live demand (RC2:98-105; PLAN:605) | 21 to 29 streams at ≤ 40k: about +15 % (est.); 16 to 24: +22 % (measured) | Low if the skip or gate is the problem; medium if live-block admission (PLAN C2) is needed. Risk: collapse when admitting beyond residency, so a preemption-counter gate is needed | Partly in the next steps (QUALITY at 21 workers). The 24-32 points and a running-block KV gate are not covered. PBF C-10 conflicts with PLAN C2 |
| 3 | **Retune the PP layer partition** (`VLLM_PP_LAYER_PARTITION`, now 25/23; move layers toward stage 0) | Stage 1 is heavier: 34.56 vs 23.54 ms (R-NB1:94), or 41-42 vs 29 ms (NR-F:113). The PP1 cards are also tighter: free MiB 1629/1629/939/925 on W4 (R-NB1:14) and 2795/2791/1189/1189 on klee100 (E26v2:81). One W4 layer is about 0.63 GB per stage-0 card (NR-G:299). Upstream retune: +5-11 % (PRC:202). PLAN froze 25/23 only "before the NB1 timeline" (PLAN:191) | Upper bound +19 %; moving about 4 layers about +10 % (est.); after the stall fix | Config, 1-2 boots. It changes the per-card pool | Planned only (D6, PLAN:338); in no arm or notebook |
| 4 | **Stage-1 post-target path**: sharded sampling (no full-vocab logits AllGather) plus graphing the eager post-target kernels | Stage 1 per tick: AllGather 2.9 ms, eager LM head 2.6 ms (R-NB1:113-116). 102 post-target eager kernels, 4.2 ms (NR-F:101). A7 estimates 2-3 ms (A7:221, 264) | About 3-5 ms of a 35-41 ms tick: **+8-12 %** (est.) | Source patch, medium-high; needs a bitwise or greedy check | Planned only (D6); graphing mentioned only (NR-F:411-413) |
| 5 | **Humming sm_89 tile tuning plus a W2 g64 benchmark** | klee100 runs all 48 MoE layers on humming (E26v1:21; E26v2:66). On the same W4 weights humming is 1.06/1.20/1.07/1.07 × Marlin at T 8/16/32/64, with T16 its worst (R-W1B:15-24). W2 g64 (layers 12-35) was never measured (R-W1B:24; PBF:73). MoE is about 18.5 ms per stage tick at T16 (R-W1B:13) | **+3-8 %** (est.) | Heuristics table data plus about 30 GPU-min of benchmarking (ST:326-333) | Mentioned only |

## 2. Status of every lever named in the documents (condensed)
- **PP and the stall:**
  - Running: S, QP, QD.
  - QP+QD combined: mentioned only.
  - Rejected: a PR #53948 port (a no-op under `--profiler-config`); #47686 (QP is its port); `NCCL_LAUNCH_ORDER_IMPLICIT`; pipeline depth 3.
  - Mentioned as fallbacks: a host-polled or gloo receive; stream priorities; a separate communicator; a parity-balancing scheduler.
  - Partition retune: planned.
  - Async scheduling: on (turning it off deadlocks PP).
- **Speculative decoding:**
  - MTP K3 probabilistic is the base: acceptance 3.04-3.18 in windows, real Duck p50 2.88, klee100 3.18-3.22.
  - K2: measured −7 % twice.
  - K1/K0: built, not run.
  - K4: planned, then dropped.
  - INT8 head at K3: never measured alone. INT4 head: hurts acceptance.
  - Draft-vocab shortlist: not available in 3.0.0.
  - Local argmax reduction plus a greedy draft: planned (D8).
  - DFlash2 / EAGLE: blocked by the speculator allowlist; a 1-2 week port.
  - Adaptive verification: rejected.
  - MTP-layer scale calibration: dropped.
- **CUDA graphs:**
  - Decode is already FULL; the "PIECEWISE" label was a parser artefact.
  - Capture coverage of (C/2)(1+K): not verified for klee100 at 24+ streams.
  - RecoverSSM commit graph: ≤ 1 % expected.
  - PR #60532: does not apply.
  - Graphing the eager pre- and post-graph work: mentioned.
- **Communication:**
  - SHM with P2P off is the default and the best.
  - P5 host all-reduce at world 2: microbenchmark 1.39-1.68×, in-engine 0-2 %, not in NB4S. P5 INT8: microbenchmark only.
  - LL128: dropped for corruption risk, although PLAN still lists +6.2 %.
  - `NCCL_CUMEM_ENABLE=0` / `NTHREADS=512`: untested.
  - The expandable_segments / MoE aux-stream / deterministic-align audit (D11): never done.
- **Kernels:**
  - Marlin MoE tuning: no gain. sm_89 skinny-GEMM plans: unsupported.
  - HC fused down+SiLU and TP-sharding the HC weights: planned or mentioned.
  - QSA Triton table: low value (0.7 % of kernel time).
  - GDN autotune / warm cache: planned.
  - Humming tuning: mentioned.
  - The QSA/PLE fusions are already in Merlin.
  - Guard off: refused by the launcher.
- **KV and capacity:**

  | Item | Result |
  |---|---|
  | W4 pin +1.0 GiB | adopted (149 to 214 blocks) |
  | klee100 7.39 GiB | 479 blocks |
  | Text-only | PP0 only, no pool gain |
  | #60490 | safety only |
  | SSM bf16 | 0 capacity plus an oracle fail |
  | `MERLIN_SHARE_DRAFT_WORKSPACE` | 0 B, and it never reached the NB5 server (NB5Q-ENV-1) |
  | FP8 KV | ARC 13/15 = BF16 13/15 |
  | F | 6 |
  | Live-block admission (C2) | planned; conflicts with PBF C-10 |
  | Mamba page padding | 21 % |
  | Block 2048 | later |
  | Haberstroh W2 g128 | estimate only, no MTP |
  | Own 3/4-bit checkpoint | high effort |
  | TP4 / DP / PP4 | rejected |
- **Scheduler, prefill and prefix cache:**
  - A trailing `--max-num-seqs` above 8 is accepted.
  - The 24-32 stream points were skipped.
  - CHUNK 2048 / long-prefill threshold: deferred. A 1,024-token chunk step costs about 233 ms vs 30-48 ms for decode; +7-15 % on another rig.
  - Prefix reuse: 93-98.5 %. Aborted requests keep their cache on Merlin.
  - Commit-turn variant C: built.
  - Cost-aware retention stall at 90k@3.
  - "KV gate on running blocks": stated as a rule, implementation not evidenced.
- **Workload (Duck):**
  - Duty factor: 172/225 = 0.76 vs the ≥ 0.882 gate; only 0.347 of log lines had all workers running (15 tasks, tail-dominated).
  - Stage-3 extra attempts (W5): off in NB6. The E9 filler gave +11.6 % whole-run.
  - Skip attempt 2 after an accepted attempt 1: saves 25-35 % of tokens, but accepted answers are wrong 26-31 % of the time.
  - Power-cap telemetry: planned.
- **Boot and IO:**
  - Page-cache drop: measured (load 239-254 s vs 873 s).
  - Kaggle Models: done.
  - Hash during prefetch (B2), compile cache Model (B1) and pre-built venv (B4): planned, never built.

## 3. Missing levers beyond the top 5
| # | Lever | Evidence | Gain (est.) | Cost / risk | Status |
|---|---|---|---|---|---|
| M6 | Boot / setup time (B1 cache Model, B2 sha during the prefetch, B4 venv) | klee100 ready 519 s; sha 191 s + install 170 s; one boot minute ≈ 90k tokens at 1500 | 5-8 min ≈ +0.7-1.2 % | Low-medium | Planned, never built |
| M7 | Tail filler / attempt policy in production | NB6 has stage-3 off; coverage binds at 37-80 %; E9 filler +11.6 % | Up to about +10 % whole-run when the queue empties | Config; must copy E15B SUBMIT's proven settings | Planned (W5), not active |
| M8 | Profile klee100 at 16-24 streams | All NB4 / NB4S windows are W4 at levels 8-9; klee100's tick is 55.8-68.7 ms | Sizes every post-stall lever | One short profiled window | Not planned |
| M9 | Capture coverage at 24-32 streams (48 / 64 tokens per micro-batch) | A 9-request micro-batch ran eager with CAPS16; under PP2, 32 captured tokens cover 16 streams | 0 if covered; unknown if 562@24 ran eager | Read-back only | Not checked |
| M10 | Prefill isolation: CHUNK 2048 / long-prefill threshold | 1,024-chunk step about 233 ms vs 30-48 ms for decode; +7-15 % on the CMP rig | Unknown; more relevant at 21+ workers | 1-2 boots | Deferred |
| M11 | Micro-batch parity logging and control | 8/0 vs 4/4 cycles differ | A confound for every A/B | Scheduler patch | Mentioned |
| M12 | QP+QD as one arm | Upstream calls them complementary | — | Already allowed by `server_env` | Mentioned |
| M13 | K4 | Position-3 acceptance 0.586; 27B MTP4 acceptance 3.57-3.76 | ± unknown | 1 boot plus captures | Dropped |
| M14 | Merlin env audit: expandable_segments / deterministic MoE align / aux-stream | +13-37 %, +37 % and −20 % on jon-nielsen's 4×3090 kit | Unknown on Merlin | Log read only | D11, never done |
| M15 | P5 world-2 all-reduce in the engine | about 0.9 ms per stage tick | 0-2 % | Patch | Not in NB4S |
| M16 | PP2 double-read redesign / single-batch layout | MoE-only 1,095 vs 1,706 at C 32 | Structural | High | Design study |
| M17 | DFlash2 port | +18-24 % on GB300 | Unmeasured at 60-120k | 1-2 weeks | Blocked |
| M18 | 120k band | Every speed window so far is ≤ 60k; the user rule says measure the 120k band | Validates the gains | — | Not measured |

## 4. Never-do list (PLAN:373-389, plus later additions)
| Never | Reason |
|---|---|
| P2P on; PCIe-IPC all-reduce; NCCL CTA tuning; `NCCL_SHM_USE_CUDA_MEMCPY` | Measured slower; memcpy hangs in capture |
| KV offload to host for live streams | A decoding stream needs its KV on the GPU; an sm_89 fork wedged at 0.3-0.9 tok/s |
| Prefix caching off while MTP is on | E19 state flips; 40k fell to 3 tok/s |
| `--no-async-scheduling` with PP | Deadlocks |
| Admitting beyond residency; time-slicing; pause/resume | E12 EVAL30; the Franzen collapse; B13 |
| Abort-and-resubmit drains | D66 lost 14/15 units; E18 lost 16/18 |
| Synthetic prompts for K or acceptance decisions | 3.57 synthetic vs 2.69 real acceptance |
| A microbenchmark gain counted as an engine gain | [A1] |
| `COUNTERS=1` in production | Disables the fused QSA writer |
| `--enable-hc-sp`; FlashInfer PCIe IPC | PP = 1 only / needs P2P |
| FlashInfer sampler without the `-lcuda` shim | RC1 died on its first sampled request |
| SGLang EP4 for Flash-Next; NVFP4 KV on the QSA path | Wrong outputs / not supported |
| Early-stop of hopeless thinking | AUC 0.63; −18 % train passes |
| Later rejections | `NCCL_GRAPH_MIXING_SUPPORT=0`, `NCCL_GRAPH_REGISTER=1`, `NCCL_LAUNCH_ORDER_IMPLICIT`, forced LL128, guard off |

## 5. Contradictions, and claims later corrected
| # | Earlier claim | Correction |
|---|---|---|
| 1 | "Graph verdict PIECEWISE" (R-NB1:49); SEED2 FULL 0; "Flash-Next decode has 0 FULL forwards" (PLAN:523) | Decode is one FULL graph; the analyzer mislabelled it (NR-G:85-91; NR-F:31-37). PLAN:523 and R-NB1:49 are stale |
| 2 | "The binding stage is stage 0; the light stage idles about 27 ms" (R-NB1:68-69) | Stage 1 is heavier once drafting is counted; its receive wait is 7-9 ms (R-NB1:108-119; NR-F:99-115) |
| 3 | Tick after the fix 34.6 ms, "+43 %" (R-NB1:107) | 40-42 ms, +19-25 % (NR-G:127, 296; NR-F:114) |
| 4 | Stall 20.96 ms, 36/36, kernel 172 (R-NB1:95-101) | 21.2 ms mean, 39/40 graphs, node #173 (NR-G:97, 417); 21.70 ms median (NR-F:70). "21.75 ms" appears in none of the files; NR-F gives a 49.83 ms tick |
| 5 | "NCCL is 63.5 % of kernel time" read as a cost | Receive-side spin on 128-B messages, an accounting artefact (NR-G:105-111) |
| 6 | RecoverSSM: "only n = 16 captured" (R-NB1:45) | 9 keys on PP1, none on PP0; the oracle ran eager, so the graph is not the cause of B2's fail (NR-G:229-236) |
| 7 | "K2 / INT8 head is not a speed lever" (R-W1B:55-56; NR-F:451) | The loss is only the missing third position; INT8 does not lower acceptance (R-W1B:61-63). K3 + INT8 was never tested |
| 8 | F = 7; "56 of 149 blocks"; "5 streams at 70k" (PLAN:140, 228-229, 321) | F = 6 (RC2; PLAN:182; V2:11) |
| 9 | "seqs 8 = the only concurrency the engine is gated at" (PLAN:217) | A trailing `--max-num-seqs` is accepted; klee100 ran 16 and 24 |
| 10 | klee100: 463/534 blocks; "BF16 about 244, not the 16-32 stream path" (PLAN:507-511); 267.8 at 5.27 GB | Measured 479 FP8 blocks at 7.39 GiB; resident bound 29 at 40k. PLAN:507-511 is superseded |
| 11 | FP8 KV NO_GO by teacher forcing | ARC 13/15 in both; NB6 uses FP8 |
| 12 | "Static scales on only 6 of 12 QSA layers" (PLAN:532-533) | Withdrawn: all 12 are covered; only MTP is at 1.0 |
| 13 | LL128 +6.2 % listed as a lever (PLAN:251, 339) | Dropped as UNTESTED for corruption risk (PLAN:609). PLAN is internally inconsistent |
| 14 | CTA knobs "untested" (A5) | Measured slower at 983 KB, world 4 (A7:59); world-2 small messages untested |
| 15 | Host all-reduce "dropped, 0.93×" (A5:134) | 1.39-1.68× at world 2 for 5-123 KB (R-W1B:27-30) |
| 16 | PR #60532 a P1 lever, +36-47 % | Not applicable: QSA / GDN are already FULL |
| 17 | Commit-turn cache miss with thinking off (A4:120) | Corrected in PLAN v3.1; NB5 tool-call rate 0.996 |
| 18 | SSM fp16/bf16 +10 % KV | Flash-Next: 0 capacity plus an oracle fail |
| 19 | Text-only +10.3 % KV | Flash-Next: PP0 only, no pool gain |
| 20 | "An aborted request loses its cache" | Merlin keeps it (95.5 %) |
| 21 | Acceptance planned at 2.3-2.8 / 2.72 | Windows 3.04-3.22 (synthetic early reasoning); real Duck p50 2.88 |
| 22 | Stage 0 idles 5-15 ms; PP hand-off 1-3 ms | Measured stall 21 ms; SendRecv 0.05 ms; all-reduce 3.0-4.2 ms per stage |
| 23 | 8k@8 "step 90.2 ms" | That is a cycle (2 ticks), not a tick |
| 24 | "NB4 needs nsys" | nsys is not on the image; the torch trace suffices |
| 25 | 4-8 CPU cores and 14 GB /dev/shm in the research prompt | 46-48 vCPU and 86 GB /dev/shm (RC2:52, 243) |
| 26 | PLAN C2 live-block admission ranked second | PBF C-10 refuses live admission in production |
| 27 | P2P on L4 "UNRESOLVED" | 5.5× slower, measured (A7:56) |
| 28 | "25/23 is capacity-optimal" (PLAN:215) | The direction of 26/22 is ambiguous; it needs an L4 measurement |
| 29 | NB5 README about 250 tok/s FP8 | Measured window 172, steady 225 |
| 30 | GPT throughput formulas | Lack the PP2 cycle factor of 2 |
| 31 | "32 capture tokens are enough" | Under PP2, 32 covers only 16 streams |

## 6. Rule checks
1. No lever proposes thinking below 60k.
2. Items that need care:
   - PBF C-9's `min(131072, P_b)` context clamp needs approval.
   - The W4 deadline clamp can give late requests below 60k; this is proven E15B behaviour.
   - "Unequal first-pass budgets" are acceptable only at ≥ 60k.
3. Skipping attempt 2 is scheduling, not a thinking cut; do it only when time binds.
4. Several PBF items add own rerun / eval gates, which conflicts with the 2026-11-01 proven-methods rule.
5. The never-do list is respected by all running work.

## 7. Bottom line
- **Running:** the stall arms S, QP and QD. The ceiling, if they work, is about +22-25 % on the W4 8k@8 base.
- **Missing, if the stall clears:** stage 1 then binds the tick. Three cheap levers act on it, and none is in any arm:
  - INT8 draft head at K3;
  - PP partition retune;
  - sharded sampling plus graphing the post-target work.
- **Missing on klee100 (2.0-2.5× faster):**
  - streams 24-29, with a KV gate on running blocks;
  - humming tuning and a W2 g64 benchmark.
- **Missing workload items:** the production tail filler, boot caching, and a 120k-band check of every winner.
- **Caveat:** a 15-30 task QUALITY run is tail-dominated (NB5 had only 34.7 % of its lines at full workers). Report steady speed at full workers separately from the window average.
