# Speed levers for Flash-Next 1500 tok/s: what is measured, what is running, and what is still missing (2026-11-01)

I read every assigned document in full, with seven parallel readers each covering a fixed group of files.
- **Read myself, line by line:** `PLAN_1500.md`, `README.md`, every file in `Experiment 25/Experiment Results/`, and both E26 K3_ONLY results.
- **Read by the readers:** all other `file:line` citations (A1-A7, GPT inputs 00-06 with their verifications, the NB4 research notes, PR #53948 comparison, tick spec, NB4 audit prompt, PENDING_BATCH_FIXES, NB_BUILD_SPEC, STRUCTURAL_TRACK_FEASIBILITY, E13 RC2, E13 research prompt). I did not re-open those files to recheck each line number.
- No edits, no git, no push. The `.zip` files were not opened.
- "(est.)" marks my own arithmetic on numbers from the files. Every other number is quoted.

**File keys used below**

| Key | File |
|---|---|
| PLAN | Exp25 `PLAN_1500.md` (v3.6) |
| R-NB1 | Exp25 Results `E25_GPU_NB1_R2_2026-10-31.md` |
| R-W1A | Exp25 Results `E25_GPU_wave1_NB0_NB1_2026-10-31.md` |
| R-W1B | Exp25 Results `E25_GPU_wave1_NB3_NB2SMOKE_2026-10-31.md` |
| R-NB5 | Exp25 Results `E25_GPU_NB5_QUALITY_2026-11-01.md` |
| R-NB6 | Exp25 Results `E25_NB6_COMMIT_v1_2026-11-01.md` |
| R-K100 | Exp25 Results `E25_klee100_cpu_kit_v1_2026-10-31.md` |
| R-KIT3 | Exp25 Results `E25_NB3_CPU_kits_v1_2026-10-31.md` |
| CTJ | Exp25 Results `commit_turn_prefix_test_2026-10-31.json` |
| E26v1 / E26v2 | Exp26 Results `E26_K3ONLY_GPU_v1` / `v2_2026-11-01.md` |
| RC2 | Exp13 Results `E13_RC2_GPU_v1_2026-10-31.md` |
| RP | Exp13 Review `FN_1500_RESEARCH_PROMPT_2026-10-31.md` |
| A1-A7 | Exp25 Review `Past work mining/A*.md` |
| G1 / V1 | `GPT research inputs/01_GPT_REPORT_1` / `01_VERIFICATION_by_agent` |
| AU1 / AU2 / AU3 (+V) | `02_RC2_AUDIT_GPT` / `03_RC2_AUDIT_AI2` / `04_RC2_AUDIT_AI3`, and their verifications |
| G2 / V2 | `05_GPT_REPORT_2` / `05_..._VERIFICATION` |
| G3 / V3 | `06_GPT_REPORT_3` / `06_..._VERIFICATION_AND_ROOFLINE` |
| ST | `STRUCTURAL_TRACK_FEASIBILITY_2026-10-31.md` |
| NR-G / NR-F / NR-C | `NB4_RESEARCH_graphs` / `_folders` / `_conversations` |
| PRC / TS / AP | `NB4_PR53948_COMPARISON` / `NB4_TICK_SPEC` / `Audit Zips/00_AUDIT_PROMPT_NB4.md` |
| PBF / SPEC | `PENDING_BATCH_FIXES.md` / `NB_BUILD_SPEC_2026-10-31.md` |

---

## 1. Top 5 missing levers (ranked by expected gain × confidence)

A lever counts as missing here when it is not measured, not running, and not in the next steps, but there is evidence of a real gain.

The levers ranked 1, 3 and 5 depend on the NB4S result. Once the stage-0 stall is removed, stage 1 sets the tick:
- stage-1 compute is 34.56 ms per tick vs 23.54 ms on stage 0 (R-NB1:94, 108-119);
- the ideal overlapped tick is 34.6 ms (R-NB1:107);
- NR-F puts stage 1 at 41-42 ms vs 29 ms (NR-F:113-115).

**None of the running arms (S, QP, QD) touch stage 1** (NR-F:113-115, 436-440).

| Rank | Lever | Evidence | Expected gain | Cost / risk | Status today |
|---|---|---|---|---|---|
| 1 | **INT8 draft LM head at K3** (`DRAFT_HEAD=int8`, K kept at 3) | Merlin on 4×3090 Flash-Next: −6.5 to −7.2 % ms per token at the same acceptance (A3:72, 126, 307; RP:19; V1:53). Our NB2: the −7 % of K2+INT8 comes only from losing the third draft position. Per-position acceptance is 0.865 / 0.723 with INT8 vs 0.867 / 0.716 / 0.586 on the K3 control, so "**the INT8 head does not lower acceptance**" (R-W1B:61-63). One LM-head shard call costs 2,492-2,562 µs on L4 (R-W1B:32). The 3 draft graphs cost 12.4 ms per tick on stage 1 (NR-F:102; R-NB1:113 gives 6.8 + 3.75 ms). Vocab heads per tick: BF16 10.2 / int8 6.4 / int4 4.5 ms (A7:90, 219; PLAN:109). | About −3.8 ms on the stage-1 tick, so **+8-11 % tok/s once stage 1 binds** (est.). About 0 while the stage-0 stall binds, because the drafts are off that chain (NR-F:110-111). | Config only, 1 boot. Needs the GPU-side `_C.gptq_marlin_repack` preflight (R-KIT3:17-18). The int8 head was in the SMOKE loop boot, but the loop did not reproduce (R-W1B:47-56). | **Never measured alone.** It is wrongly filed as a negative in NR-F:451 and R-W1B:55-56 (see the contradictions). It is not in NB4S. It is only an "optional, budget-dependent" NB4 arm (AP:75). |
| 2 | **More resident streams on klee100: past 21, up to the resident bound of 24-29** | E26v2:87-89: going from 16 to 24 streams at 20k gives 460.4 → 562.0 tok/s (+22 %), with tick 55.8 → 68.7 ms. The resident bound is 29 / 21 / 17 / 13 streams at 40k / 64k / 90k / 120k (E26v2:79). 20k@32 and 40k@24 were skipped "as KV-bound" (E26v2:92), yet KV max was only 0.84 at 40k@16, where the live demand is 16 × (10 + 6) = 256 / 479 = 0.53 (est.). The RC2 finding explains the gap: up to 16 finished sessions stay retained, so `kv_cache_usage_perc` is not live demand (RC2:98-105; PLAN:605, "gate on running blocks"). E23 on the 27B: streams were the lever at every length (PLAN:505-506). Live context is about half the final length (A3:271). Real NB5 per-task means are 14k / 50k / 107k (R-NB5:37-39). | 21 → 29 streams at ≤ 40k: about +15 % (est., using the measured tick slope of about 1.6 ms per stream). 16 → 24: +22 % (measured at 20k). | Low cost if the skip or gate is the problem. Medium cost if live-block admission (PLAN C2) is needed. Risk: the Franzen collapse when admitting beyond residency (PLAN:235; A5:78), so a preemption counter gate is needed (PLAN:200). | Only partly in the next steps (the ~21-worker QUALITY run, E26v2:117). Not covered: the 24-32 points, which check skipped them, and a Duck KV gate that reads running blocks, not usage. PBF C-10 refuses live admission in production (PBF:215-218), which conflicts with PLAN C2. |
| 3 | **Retune the PP layer partition** (`VLLM_PP_LAYER_PARTITION`, now 25/23; move layers toward stage 0) | Compute: stage 1 is heavier, 34.56 vs 23.54 ms (R-NB1:94) or 41-42 vs 29 ms (NR-F:113). Memory: the PP1 cards are also tighter, with free MiB 1629 / 1629 / 939 / 925 on W4 B3 (R-NB1:14) and 2795 / 2791 / 1189 / 1189 on klee100 (E26v2:81). Each W4 layer adds about 0.63 GB per stage-0 card (NR-G:299). Upstream partition retune gave +5-11 % (PRC:202). PLAN froze 25/23 only "before the NB1 timeline" (PLAN:191), and that timeline now exists. | Ideal balance 29.05 ms vs 34.6 ms: an upper bound of +19 %. Moving about 4 layers: about +10 % (est.). Applies after the stall fix. | Config only, 1-2 boots. It changes the per-card KV layers and pool; PLAN:215's capacity-optimal note does not say whether 25/23 or 26/22 holds more on L4. | **Planned only** (D6, PLAN:338). Not in any arm or notebook; "silently absent from the spec" (NR-C:423-427, 499; NR-F:440; NR-G:298-299). |
| 4 | **Stage-1 post-target path: sharded sampling (no full-vocab logits AllGather) plus graphing the eager post-target kernels** | Stage 1 per tick: AllGather 2.9 ms and eager LM head 2.6 ms (R-NB1:113-116). There are 102 post-target eager kernels, 4.2 ms of kernel time, including the 1.44 ms AllGather (NR-F:101). A7 K27 estimates 2-3 ms per tick (A7:221, 264). | About 3-5 ms off a 35-41 ms stage-1 tick: **+8-12 %** (est.), after the stall fix. | Source patch. Sharded top-k / top-p is harder than sharded argmax. Medium-high cost; needs a bitwise or greedy check. | **Planned only** (D6 per-shard sampling, PLAN:338). Graphing the eager post-target kernels is mentioned only (NR-F:411-413). Neither is in any arm. |
| 5 | **Humming sm_89 tile tuning, plus a benchmark of W2 g64** (klee100's own kernels) | klee100 now runs all 48 MoE layers on humming (E26v1:21; E26v2:66). On the same W4 weights humming is 1.06 / 1.20 / 1.07 / 1.07 × Marlin at T 8 / 16 / 32 / 64. T16 is its worst point, called "a tuning target" (R-W1B:15-24). **W2 g64, used by klee100's layers 12-35, was never measured** (R-W1B:24; PBF:73). MoE is about 18.5 ms per stage tick at T16 W4 (R-W1B:13). | Removing humming's 7-20 % overhead: about 7-17 % off the MoE term, so **+3-8 % tok/s** (est.). | Sm89Heuristics table data plus about 30 GPU-min of benchmarking (ST:326-333). Low numeric risk if the reference check passes. | Mentioned only (R-W1B:24; NR-C:429). Not planned anywhere. |

Two more items come next, not top 5, because their gain is smaller or less certain:
- Boot / setup time (section 3, M6).
- The Duck attempt policy that fills the tail (section 3, M7).

---

## 2. Every lever named in the documents, with its status

Status words: **MEASURED**, **RUNNING**, **BUILT-NOT-RUN**, **PLANNED**, **MENTIONED**, **NEVER-DO**, **REJECTED** (rejected by evidence, outside the never-do list).

### 2a. PP pipeline and the stage-0 stall

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| `CUDA_DEVICE_MAX_CONNECTIONS=32` (arm S) | RUNNING | Tests the hardware work-queue hypothesis. Expected about +22-25 % if the stall goes (NR-G:127, 296). Not guaranteed: the graph forks about 75-96 streams | NR-G:57, 69-79; NR-F:395-404; R-NB1:105 |
| Packed PP broadcast (arm QP) | RUNNING | One int64 broadcast instead of 3. Upstream #47686: about 31 % of stage-0 kernel time spinning; +5.6 % tok/s at concurrency 256 | PRC:128-143, 230-234 |
| Deferred PP receive (arm QD) | RUNNING | Posted after the graph(T+1) launch. Upstream #53948: +38.5 % on H100 PP4, no spec decoding, no PP2 validation | PRC:37-59, 198-215 |
| QP + QD combined | MENTIONED | Upstream calls them complementary; `server_env` allows it; no arm | PRC:237-238 |
| NB4S fixes F1 (shutdown drain), F2 (flush at sync RPCs), F3 (sender draft snapshot on the main stream), R1 (strict read-back) | MENTIONED (recommended) | F2 prevents a deadlock that loses an arm. F3 is a silent wrong-draft bug. **Whether NB4S applied them is not stated** | PRC:76-91, 160-193 |
| Port of PR #53948 | REJECTED | For PP2 its post time equals QD. Its profiler gate turns it off under `--profiler-config`, so a port would be a silent no-op | PRC:9-13, 145-158 |
| PR #47686 | REJECTED (cannot apply) | QP is in effect its port | PRC:128-143 |
| Host-polled or gloo receive; host-mapped flag | MENTIONED | Fallbacks if S and QD fail | NR-F:405-406 |
| Stream priorities; separate broadcast communicator; host-thread receive | MENTIONED | Fallback tests | AP:190 |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | REJECTED | Cannot reorder captured graph nodes | NR-G:58; NR-F:407 |
| Pipeline depth 3 | REJECTED | Would not hide the stall (the wait is on T−1) | NR-G:44-51 |
| Parity-balancing scheduler / micro-batch split control | MENTIONED | NB2 SEED2 ran 8/0: 77.6 ms per 8-request cycle vs 99.6 ms at 4/4 (confounded with K2, the INT8 head and seqs 16) | NR-G:128-131, 300; NR-F:119-122, 428-432 |
| PP partition retune | PLANNED | Top-5 #3 | PLAN:191, 338 |
| Async scheduling | MEASURED (on) | Already on; turning it off deadlocks PP | NR-G:45; PLAN:380 |

### 2b. Speculative decoding

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| MTP K3, probabilistic draft | MEASURED (base) | Accept 3.04-3.18 in windows (R-NB1:19-22); real Duck p50 2.88 (R-NB6:25); klee100 3.18-3.22 (E26v2:87-89). Probabilistic beats greedy draft by +4.5 % on 3090 | A3:70 |
| K2 | MEASURED | −7 %, twice: 240.9 vs 259.1 and 242.6 vs 261.8 | R-W1B:43-45, 54-56, 61-63 |
| K1 / K0 | BUILT-NOT-RUN | NB2 FULL A/B/C is deprioritised. klee100 K0 / K2 boot plans are built (R-K100:60). E23 27B: MTP0 222.5 vs MTP3 359.1 | SPEC:81-87; PLAN:520 |
| K4 | PLANNED, then dropped | "One K4 arm in NB4" (PLAN:521); optional NB4 arm (AP:75); not in NB4S. Position-3 acceptance 0.586 | PLAN:344, 521; NR-C:187, 402-405 |
| INT8 draft head at K3 | not measured alone | Top-5 #1 | R-W1B:61-63 |
| INT4 draft head | MENTIONED (negative) | Cuts traffic but hurts acceptance | G3:93, 128 |
| Draft-vocab shortlist / FR-Spec | MENTIONED | `REDUCED_DRAFT_VOCAB` is not available in 3.0.0 | G3:147; A3:127; V1:63 |
| `use_local_argmax_reduction` + greedy draft | PLANNED (D8) | +10-20 % on 8×3090, 0 on 4×3090 | PLAN:340; A3:84, 179 |
| DFlash2 / EAGLE drafter | MENTIONED (blocked) | Merlin speculator allowlist is {mtp, ngram, ngram_gpu}. Port (S3) is 1-2 weeks; acceptance +0.18 to +0.42 | ST:208-241, 356-370 |
| Adaptive verification | REJECTED | Incompatible with the GDN backends | ST:223 |
| MTP-layer QSA FP8 KV scale calibration | PLANNED, then dropped | The MTP layer runs at the default scale 1.0. The "6 of 12 layers" lead was withdrawn | PLAN:532-534; E26v1:24; NR-C:406-409, 493 |

### 2c. CUDA graphs

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| FULL decode graphs | MEASURED (already FULL) | One FULL graph per step: stage 0 1,037 nodes, stage 1 951, draft 72 / 65 / 65. The "PIECEWISE" label is a parser artefact | NR-G:85-95; NR-F:31-37, 247-249 |
| Capture list covering (C/2)(1+K) (NB4 arm G) | BUILT-NOT-RUN (likely SKIPPED_BUDGET); not in NB4S | With CAPS16 at seqs 9, a 9-request micro-batch runs fully eager; 5 and 7 are padded. The `captures_cover` helper bug is flagged. **klee100 at 24 streams needs 48 tokens; the coverage there is not stated** | NR-G:132, 178-208; AP:72, 90; V3:9 |
| RecoverSSM commit graph (arm C) | MEASURED partly (B2); BUILT as a runtime A/B | Expected ≤ 1 %: stage-0 commits are always eager, and only stage 1 (0.87 ms) can graph | R-NB1:45; NR-G:212-245; NR-F:418-423 |
| PR #60532 | REJECTED | Does not apply: QSA and GDN are already UNIFORM_BATCH FULL | V2:13; V3:10 |
| Graphing the eager pre-graph (82 launches, ≈ 1.9 ms) and post-target (4.2 ms) work | MENTIONED | Part of top-5 #4 | NR-F:75-82, 101, 411-413 |
| Breakable graphs | kept OFF | — | NR-G:208 |

### 2d. Communication

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| NCCL SHM, P2P off | MEASURED (default) | Best on this VM | A7:53-55; PLAN:244-248 |
| P5 host-staged all-reduce at world 2 (arm P) | MEASURED (microbenchmark) / BUILT-NOT-RUN in the engine | 1.39-1.68× at 5-123 KB, status PASS_BOUND_ONLY; in-engine estimate 0-2 % (about 0.9 ms per stage tick). Not in NB4S | R-W1B:27-30; NR-G:305-347; PLAN:453 |
| P5 INT8 all-reduce | MEASURED (microbenchmark) | 1.41-2×; never run in-engine; needs a logprob gate | A7:61-62 |
| `NCCL_PROTO=LL128` | dropped (UNTESTED) | NCCL docs warn of data corruption. PLAN still lists "+6.2 %" in §2.4 and D7 | PLAN:251, 339, 609; PBF:32; AU2V:14 |
| P2P on, PCIe-IPC / FlashInfer all-reduce, CTA knobs, `NCCL_SHM_USE_CUDA_MEMCPY` | NEVER-DO | All measured slower; memcpy hangs in capture | PLAN:377 |
| `NCCL_CUMEM_ENABLE=0`, `NCCL_NTHREADS=512` | MENTIONED (untested) | No numbers | PLAN:251; NR-C:165 |
| No `expandable_segments`; MoE aux-stream off | PLANNED (D11: "check what Merlin enables") | +13-37 % / −20 % on jon-nielsen's 4×3090 kit. No result anywhere. The aux-stream env is dropped by the launcher | PLAN:251, 343; V1:72, 105; NR-F:327 |
| `VLLM_DETERMINISTIC_MOE_ALIGN=0` equivalent | MENTIONED | 4 streams 199 → 273 (+37 %) on the jon-nielsen fork; "check for a Merlin equivalent" | V1:72, 105 |
| `NCCL_GRAPH_MIXING_SUPPORT=0`, `NCCL_GRAPH_REGISTER=1` | REJECTED | Unsafe or deadlock risk | NR-F:448-449 |
| FlashInfer PCIe-IPC at world 2 | MENTIONED | Availability unverified | NR-G:347 |

### 2e. Kernels

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| Marlin MoE config tuning (D5) | MEASURED: no gain | Best config equals the base within 0.4-1.1 %; 204-252 GB/s | R-W1B:10-13 |
| sm_89 skinny-GEMM plans (D4) | MEASURED: UNSUPPORTED_SM89 | CuTe candidates fail to compile | R-W1B:7, 32-34 |
| Fused HC down+SiLU FMA port; TP-sharding the replicated HC weights | PLANNED / MENTIONED | No numbers | PLAN:262, 336; A7:182, 259-260 |
| QSA sm_89 Triton table | PLANNED (D10) | QSA is only 0.7 % of kernel time, so low value | PLAN:342; R-NB1:59 |
| GDN autotune / warm Triton cache | PLANNED (D10) | 5 JIT kernels during inference | PLAN:265; AU1V:20 |
| Humming tile tuning; W2 g64 benchmark | MENTIONED / not measured | Top-5 #5 | R-W1B:15-25 |
| QSA / PLE fusions (#59533, #57097, #54513, #54517, #58114) | already in Merlin | #59533 end to end only −0.4 to +5.1 % | V1:24-28 |
| Guard off (`VLLM_E2_GUARD_MODE=off`) | REJECTED | The launcher refuses it (0.3 ms per step) | NR-F:328-331; PLAN:340 |
| `TARGET_SEQ_LENS_ASYNC`, `GDN_DECODE_FASTPATH`, `GDN_SHARED_ALIGN_INDEX` | PLANNED (D8) | `TARGET_SEQ_LENS_ASYNC` is skipped under PP. The host is 62-117 ms ahead, so these are low value | PLAN:340; NR-G:61; NR-F:319, 327 |

### 2f. KV, capacity and checkpoints

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| W4 KV pin +1.0 GiB | MEASURED, ADOPTED | 149 → 214 blocks (+44 %); tick unchanged | R-NB1:14, 76-78 |
| klee100 pin 7.39 GiB | MEASURED | 479 blocks, 1,652,197 tokens. Free MiB 2795 / 2791 / 1189 / 1189, so PP1 is near its limit | E26v2:64, 75-81 |
| Text-only (C4) | MEASURED, kept | Frees 430-451 MiB on PP0 only; no pool gain | R-NB1:79 |
| #60490 (D9) | MEASURED, kept | Safety only, no speed change | R-NB1:80 |
| SSM bf16 (C3) | MEASURED, parked | 0 capacity (padded to the same 2 MiB page) plus an oracle fail | R-NB1:41-46, 81 |
| `MERLIN_SHARE_DRAFT_WORKSPACE` | MEASURED: 0 B | It also never reached the NB5 server (bug NB5Q-ENV-1) | R-NB1:44; PBF:131-132 |
| FP8 QSA KV | MEASURED | 1.91× pool. Teacher-forced NO_GO at 40k and 90k. ARC 13/15 = BF16 13/15. Used in NB6 | R-W1A:14-32; R-NB5:7-32; R-NB6:9 |
| FP8 byte-decoder path | MEASURED | Not better; divergence is not the native cast | R-W1A:16-25 |
| C1: shrink F | MEASURED (F = 6) | No lever found. F for K0/K1/K2 is unmeasured | R-NB1:12-14; E26v2:78 |
| C2: live-block / projected-peak admission | PLANNED | PBF C-10 refuses live admission in production | PLAN:322; PBF:215-218 |
| Mamba page padding (21 % on FP8) | MENTIONED | — | RC2:78-79, 121 |
| Block size 2048 | PLANNED (later) | — | PLAN:345 |
| klee100 3bpw | MEASURED | 2.0-2.5× W4 speed. Teacher-forced divergence SEVERE. The ARC QUALITY run is being built | E26v2:84-117 |
| Haberstroh W2 g128 (no MTP) | MENTIONED (estimate only) | 632 / 702 blocks (est.). Target-only; MTP0 was −38 % vs MTP3 on the 27B (PLAN:520) | PBF:72, 83, 251; ST:151 |
| S2: our own Merlin-format 3/4-bit checkpoint | MENTIONED | +200-300 blocks (est.); high effort | ST:335-354 |
| TP4+EP, DP 2×TP2, PP4 | REJECTED | −57 % pool; infeasible; crash | PLAN:231-232; ST:376-377 |
| Single-batch expert read (avoid the PP2 double read) | MENTIONED (design study) | MoE-only at C 32: 1,095 vs 1,706 | V3:30-31, 56-61 |
| EXL3, llama.cpp, NInfer, Mirai, GLQ, pruned models | MENTIONED (alternatives) | Not shortlisted | ST:372-383; PLAN:391-394 |

### 2g. Scheduler, prefill and prefix cache

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| seqs above 8 | MEASURED | klee100 at 16 and 24 streams. A trailing `--max-num-seqs` is accepted | E26v2:84-92; PBF:79 |
| Stream points 20k@32, 40k@24, 100k@16 | skipped "KV-bound" | Top-5 #2 | E26v2:92 |
| CHUNK 2048 / `--long-prefill-token-threshold` (W6) | PLANNED, deferred | A 1,024-token chunk step costs ≈ 233 ms vs 30-48 ms for a decode step. Chunk 2048 gave +7-15 % on the CMP rig | SPEC:74; PLAN:356; RC2:148-149; A3:123, 327 |
| Prefix reuse, two-turn and after an abort | MEASURED | 93.0 / 98.5 / 95.5 %. An aborted request keeps its cache on Merlin | R-NB1:27-34 |
| Whole-run Duck prefix hit (W2) | partly measured | NB6: 49 % on 3 tasks and 233 s. Not reported for NB5 | R-NB6:25 |
| Commit-turn variant C | BUILT (NB5) | CPU test: 98.3 % reuse vs 0 %. NB5 QUALITY ran variant A, with tool-call rate 0.996 | CTJ; PLAN:278-290; R-NB5:4, 12 |
| Cost-aware admission and retention | MEASURED (RC2 stall) | 90k@3: TTFT 143.7 s; window 48.7 tok/s. No test of a change | RC2:98-105, 455, 488 |
| KV gate on running blocks, not usage | stated as a rule | Implementation not evidenced | PLAN:605 |

### 2h. Workload (Duck)

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| Workers = resident capacity | MEASURED | NB5: 9 workers; klee100 rule gives 21 | R-NB5:7; E26v2:82 |
| Duty factor (window vs steady) | MEASURED, below the gate | 172 / 225 = 0.76 vs the ≥ 0.882 gate. Only **0.347 of log lines had all workers running**; time duty 0.999. On 15 tasks the run is tail-dominated | R-NB5:14-17; PLAN:198 |
| Stage-3 extra attempts as tail filler (W5) | PLANNED; **off in NB6** (`E12_B_EXTRA_ATTEMPTS` 0) | E9 tail filler: whole-run +11.6 %. A third attempt on an unsolved task is worth 0.12-0.16 tasks | R-NB6:17; PLAN:355; A4:93, 97 |
| Skip attempt 2 after an accepted attempt 1 | MENTIONED; off in NB6 | Saves 25-35 % of tokens on NB5. But accepted answers are wrong 26-31 % of the time ("only when time binds") | R-NB5:42-43; A4:213; R-NB6:17 |
| ORDER2 ordering, deadline clamp, admit until 11:50, side-work isolation, real running == workers | BUILT (E15B / Duck) | — | A4:92-100; PLAN:351-357 |
| Power-cap telemetry (`nvidia-smi dmon`) | PLANNED (S1) | No result reported | PLAN:403; A5:186, 203 |
| Swift1.5 | MENTIONED (coverage, not tok/s) | See the rule checks | PLAN:360 |

### 2i. Boot and IO

| Lever | Status | Result / reason | Evidence |
|---|---|---|---|
| Page-cache drop before each boot | MEASURED | Load 239-254 s vs 873 s; works on NFS | R-W1A:40; R-KIT3:13-14 |
| Kaggle Models, not datasets | done | — | PLAN:300 |
| B2: hash during the prefetch | PLANNED | NB1 sha still 191 s; klee100 digests 287 s on the CPU kit | PLAN:367; R-NB1:6; R-K100:42 |
| B1: compile/JIT cache Model | PLANNED | 3-6 min per boot. Never worked across sessions | PLAN:309, 366 |
| B4: pre-built venv | PLANNED (UNVERIFIED) | — | PLAN:369 |
| Sampler warm-up inside the boot; weight prefetch; one boot per run | in RC2 / rules | — | PLAN:370-371 |

---

## 3. Missing levers beyond the top 5

| # | Lever | Evidence | Gain (est.) | Cost / risk | Status |
|---|---|---|---|---|---|
| M6 | **Boot / setup time**: B1 cache Model + B2 sha as prefetch + B4 venv | klee100 ready in 519.1 s (E26v2:64); sha 191 s + install 170 s (R-NB1:6). One boot minute ≈ 90k tokens at 1500 (PLAN:85). | 5-8 min saved ≈ +0.7-1.2 % of the run average | Low-medium; the earlier tar attempt failed (PLAN:309) | Planned, never built |
| M7 | **Tail filler / attempt policy in production** (stage-3 extra attempts; skip attempt 2 only when time binds) | NB6 has both off (R-NB6:17). Coverage binds at 37-80 % (R-NB5:45-52). E9 filler +11.6 % (A4:93). | Up to about +10 % whole-run when the queue empties (est.) | Config. Must copy E15B SUBMIT's proven settings (memory rule) | Planned (W5), not active |
| M8 | **Profile klee100 at 16-24 streams** (the stall, stage balance and capture coverage at klee100's own shape) | All NB4 / NB4S windows are W4 at 8k / 40k, levels 8-9 (TS:61; AP:81). klee100's tick is 55.8-68.7 ms. | Sizes every post-stall lever | One short profiled window | Not planned |
| M9 | **Capture coverage at 24-32 streams** (48 / 64 tokens per micro-batch) | A 9-request micro-batch ran eager with CAPS16 (NR-G:184-208). Under PP2 32 covers only 16 streams (V3:9). | 0 if covered. Unknown if 562@24 ran eager | Read-back only | Not checked for klee100 |
| M10 | **Prefill isolation: CHUNK 2048 / LPT** | 1,024-token chunk ≈ 233 ms vs 30-48 ms decode (RC2:148-149). CMP: +7-15 % (A3:123, 327). A 27B prefill burst cut generation to 52 tok/s (A5:95-96). | Unknown; more relevant at 21+ workers | 1-2 boots | Deferred |
| M11 | **Parity / micro-batch split logging and control** | 8/0 vs 4/4 (NR-F:119-122) | Confound for every A/B; possibly large | Scheduler patch | Mentioned |
| M12 | **QP + QD as one arm** | Upstream "complementary" (PRC:237-238) | — | Already allowed in `server_env` | Mentioned |
| M13 | **K4 at K3's stage-1 cost** | Position-3 acceptance 0.586; 27B MTP4 acceptance 3.57-3.76 (PLAN:520-521) | ± unknown | 1 boot plus captures | Dropped from NB4S |
| M14 | **expandable_segments / DET-align / aux-stream audit of Merlin's env** | +13-37 %, +37 %, −20 % on jon-nielsen's kit (V1:72, 105) | Unknown on Merlin | Log read only | D11, never done |
| M15 | **P5 world-2 all-reduce in the engine** | 0.9 ms per stage tick (NR-G:341-345) | 0-2 % | Patch; strict adoption rule | Not in NB4S |
| M16 | **PP2 double-read redesign / single-batch layout** | 1,095 vs 1,706 at C 32 (V3:59) | Structural | High | Design study only |
| M17 | **DFlash2 port** | GB300 +18-24 % (ST:212, 241) | Unmeasured at 60-120k | 1-2 weeks | Blocked |
| M18 | **120k band** | Every speed window is ≤ 60k (E26v2:87-92). The user rule says measure the 120k band. | Validates the gains | — | Not measured |

---

## 4. Never-do list and its reasons (PLAN:373-389, plus later additions)

| Never | Reason |
|---|---|
| P2P on, PCIe-IPC all-reduce, NCCL CTA tuning, `NCCL_SHM_USE_CUDA_MEMCPY` | All measured slower; memcpy hangs in capture (A7) |
| KV offload to host for live streams | A decoding stream needs its KV on the GPU; an sm_89 fork wedged at 0.3-0.9 tok/s (A5) |
| Prefix caching off while MTP is on | E19 state flips; 40k fell to 3 tok/s |
| `--no-async-scheduling` with PP | Deadlocks |
| Admitting beyond residency; time-slicing; pause/resume | E12 EVAL30; Franzen collapse; B13 |
| Abort-and-resubmit drains | D66 lost 14/15 units, E18 16/18 |
| Synthetic prompts for K or acceptance decisions | 3.57 synthetic vs 2.69 real |
| A microbenchmark gain counted as an engine gain | [A1] |
| `COUNTERS=1` in production | Disables the fused QSA writer |
| `--enable-hc-sp`, FlashInfer PCIe IPC | PP = 1 only / needs P2P |
| FlashInfer sampler without the `-lcuda` shim | RC1 died on the first sampled request |
| SGLang EP4 for Flash-Next; NVFP4 KV on the QSA path | Wrong outputs / not supported (bf16 or fp8 only) |
| Early-stop of hopeless thinking | AUC 0.63, −18 % train passes |
| **Later rejections, not in PLAN's table** | `NCCL_GRAPH_MIXING_SUPPORT=0`, `NCCL_GRAPH_REGISTER=1`, `NCCL_LAUNCH_ORDER_IMPLICIT` (NR-F:448-449; NR-G:58); forced LL128 (corruption risk; PLAN:609); guard off (refused by the launcher) |

---

## 5. Contradictions, and claims later corrected

| # | Earlier claim | Correction / other side |
|---|---|---|
| 1 | "Graph verdict PIECEWISE" (R-NB1:49); SEED2 "FULL 0 / PIECEWISE 20 / EAGER 20" (R-W1B:67); "Flash-Next decode has 0 FULL forwards" (PLAN:523); TS:20-21 | Decode is one FULL graph; the analyzer mislabels; the EAGER scopes are empty 0-token steps (NR-G:85-91; NR-F:31-37). **PLAN:523 and R-NB1:49 are stale.** |
| 2 | "The binding stage is stage 0, not stage 1 with the heads; the light stage is idle about 27 ms" (R-NB1:68-69; NR-C:37-44) | Same file R-NB1:108-119, plus NR-F:99-115: stage 1 is heavier once drafting is counted. Its receive wait is 7-9 ms, not 27 ms. |
| 3 | Tick after the stall fix: 34.6 ms, "+43 % tok/s" (R-NB1:107; AP:189) | 40-41 ms, +22-25 % (NR-G:127, 296); 41-42 ms, +19-21 % (NR-F:114). The accounting differs (compute 34.56 vs busy 40.1). |
| 4 | Stall size and count: 20.96 ms of NCCL peer wait, 36/36 ticks, kernel 172 (R-NB1:95, 100-101) | 21.2 ms mean, 39/40 graphs, node #173 (NR-G:97, 417); 21.70 ms median (NR-F:70). **The 21.75 ms in your brief appears in none of the files.** The 49.66 ms tick is in R-NB1:90; NR-F gives 49.83. |
| 5 | "NCCL is 63.5 % of kernel time" read as a cost (R-NB1:54, 71) | An accounting artefact: receive-side spin on 128-B messages (NR-G:105-111). R-W1B:68-69 already called it an upper bound. |
| 6 | RecoverSSM commit graph: "only n = 16 captured" (R-NB1:45) | 9 keys were captured on PP1 and none on PP0. The oracle ran eager, so the commit graph is not the cause of B2's FAIL (NR-G:229-236; NR-F:283-288). |
| 7 | "K2 / INT8 head is not a speed lever" (R-W1B:55-56; NR-F:451) | The same file says the loss is only the missing third position and the INT8 head does not lower acceptance (R-W1B:61-63; NR-C:410-414). **K3 + INT8 was never tested** (top-5 #1). |
| 8 | F = 7 and "56 of 149 blocks (38 %)"; "5 streams at 70k" (PLAN:228-229, 321, 140; RP:20; G2:185; G3:167) | F = 6 (RC2; PLAN:182; V2:11). PLAN §2.2 and C1 are stale (PBF C-12, line 223). |
| 9 | "seqs 8 = the only concurrency the engine is gated at" (PLAN:217) | A trailing `--max-num-seqs` is accepted (PBF:79); klee100 ran 16 and 24 (E26v2). |
| 10 | klee100: 463 / 534 blocks (R-K100:12); "BF16 KV ≈ 244 blocks … klee100 alone is not the 16-32-stream path" (PLAN:507-511); 267.8 blocks at a 5.27 GB pin (E26v1:27) | Measured 479 FP8 blocks at a 7.39 GiB pin; resident bound 29 at 40k (E26v2:76-79). **PLAN:507-511 is superseded.** ST:147 predicted 481. |
| 11 | FP8 KV NO_GO by teacher forcing (R-W1A:19) | ARC 13/15 in both FP8 and BF16 (R-NB5:9). NB6 uses FP8 (R-NB6:9). Accepted only by the paired ARC route, as PLAN:610 allows. |
| 12 | "Static scales exist on only 6 of 12 QSA layers" (PLAN:532-533) | Withdrawn: all 12 are covered; only the MTP layer is at 1.0 (NR-C:409, 493; E26v1:23-24) |
| 13 | LL128 +6.2 % listed as a lever (PLAN:251, 339 D7) | Dropped as UNTESTED for corruption risk (PLAN:609; PBF:32; AU2V:14). PLAN is internally inconsistent. |
| 14 | CTA knobs: "untested" (A5:159, 186) | Measured slower at 983 KB, world 4 (A7:59). World-2 small messages remain untested. |
| 15 | Host-staged all-reduce "dropped, 0.93×" (A5:134) | 1.39-1.68× at world 2 for 5-123 KB (R-W1B:27-30; A7:125-129) |
| 16 | PR #60532 is a P1 lever, +36-47 % (G1, G2:140, G3:142); "does not apply: TileLang" (A5:134) | Not applicable because QSA and GDN are already FULL (V2:13; V3:10; A7:134-137) |
| 17 | Commit-turn cache miss with thinking off (A4:120) | Corrected in PLAN v3.1 (PLAN:278-290). NB5's tool-call rate of 0.996 (R-NB5:12) suggests the 25 % no-tool-call problem (27B data) is small on Flash-Next. |
| 18 | C3 SSM fp16/bf16 gives +10 % KV (27B; PLAN:233, 323) | On Flash-Next: 0 capacity plus an oracle fail (R-NB1:43) |
| 19 | C4 text-only +10.3 % KV (PLAN:324) | Flash-Next: PP0 only, no pool gain (R-NB1:79) |
| 20 | "An aborted request loses its cache" (PLAN:272, 471) | Merlin keeps it (95.5 %; R-NB1:34) |
| 21 | Acceptance planned at 2.3-2.8 / 2.72 (PLAN:91, 159) | Windows show 3.04-3.22 (synthetic early reasoning). Real Duck p50 is 2.88 (R-NB6:25). V3 uses 3.05, ST uses 2.72. |
| 22 | Per-tick estimate: stage 0 idles 5-15 ms, PP hand-off 1-3 ms, TP all-reduce 1.5-3.8 ms (PLAN:103-118) | Measured stall 21 ms; SendRecv 0.05 ms; all-reduce 3.0-4.2 ms per stage (NR-F:85-88, 99) |
| 23 | 8k@8 "280.0 tok/s, step 90.2" (R-W1A:42) | 90.2 ms is a cycle (2 ticks), not a tick (NR-F:152) |
| 24 | "NB4 needs nsys" (R-W1B:70) | nsys is not on the image; the torch trace is enough (NR-F:352-355) |
| 25 | 4-8 CPU cores, 14 GB /dev/shm (RP:12; NR-C:352) | 46-48 vCPU and 86 GB /dev/shm (RC2:52, 243). The wrong figures went into an external research prompt. |
| 26 | PLAN C2 (live-block admission) is the second-ranked capacity lever | PBF C-10 refuses live admission in production (PBF:215-218) |
| 27 | P2P on L4 "UNRESOLVED" (A3:317) | 5.5× slower, measured (A7:56) |
| 28 | "Partition 25/23 capacity-optimal: 26/22 = 857,585, 24/24 = 739,100" (PLAN:215) | The direction of the 26/22 effect is ambiguous and needs L4 measurement |
| 29 | NB5 README estimate ≈ 250 tok/s FP8 (PBF:140) | Measured window 172, steady 225 (R-NB5:14-15) |
| 30 | GPT throughput formulas | Lack the PP2 cycle factor of 2 (V2:12; V3:8) |
| 31 | "32 capture tokens are enough" (G2:142; G3:189) | Under PP2, 32 covers only 16 streams (V3:9) |

---

## 6. Rule checks (60k floor, no token cuts, long thinking, never-do list)

1. **No lever in any file proposes thinking below 60k.**
2. **Items that need care:**
   - PBF C-9 clamps `max_len_eff = min(131072, P_b)`. That is a context cut above the floor, so it needs your approval (PBF:217).
   - The W4 deadline clamp's `min_admit_sec` can give late requests `max_tokens` below 60k. This is the proven E15B method (A4:100, 195).
   - G2:356 suggests "unequal first-pass budgets". That is acceptable only if no first pass goes below 60k.
   - Swift1.5 shortens reasoning by model choice (PLAN:360).
3. **Skipping attempt 2 after an accepted attempt 1** is attempt scheduling, not a thinking cut. A4:213 says do it only when time binds.
4. **Own gates vs the "proven methods only" rule:** several NB5 fix items add our own rerun or eval gates (PBF A-8, A-15, B-1/B-2/B-4, C-8, C-10). This conflicts with the 2026-11-01 rule to copy proven submission methods exactly.
5. **The never-do list is respected** by all running work. No file proposes P2P, prefix-off with MTP, or async-off.

---

## 7. Bottom line

- **Running:** the stall arms S, QP and QD. If they work, the documented ceiling is about +22-25 % on the W4 8k@8 base.
- **Missing, if the stall clears:** three levers act on stage 1, which then binds the tick. All three are cheap or config-level, and none is in any arm or next step:
  - INT8 draft head at K3;
  - PP partition retune;
  - sharded sampling plus graphing the post-target work.
- **Missing on klee100 (measured as 2.0-2.5× faster):**
  - streams past 21, up to 24-29, with a KV gate on running blocks;
  - humming tile tuning and a W2 g64 benchmark.
- **Missing workload items:**
  - the production tail and attempt policy (filler is off in NB6);
  - boot caching;
  - a 120k-band check of every speed winner.
- **Measurement caveat:** a 15-30-task QUALITY run at 21 workers will be dominated by the tail. On 15 tasks NB5 spent only 34.7 % of its log lines at full workers. Report steady speed at full workers separately from the window average.
