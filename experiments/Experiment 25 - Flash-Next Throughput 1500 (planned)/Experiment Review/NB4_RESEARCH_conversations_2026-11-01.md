# NB4 TICK: research from the conversation transcripts (2026-10-30 to 2026-11-01)

Read-only research for `NB4_TICK_SPEC_2026-11-01.md`.

**Sources read**
- `MAIN` = `C:/Users/hp/.claude/projects/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f.jsonl`. Its entries from 2026-10-29T18:00Z (2026-10-30 00:00 local, UTC+6) to 2026-10-31T19:04Z were streamed:
  - 269 human messages, including the GPT reports the user pasted and the PLAN_1500 text the user selected in the IDE;
  - 577 assistant text messages;
  - targeted tool results (the NB1 R2 trace summary).
- Subagent transcripts in `.../1c6db7aa-.../subagents/`. Their final reports were checked for tick facts:
  - `agent-a60fb6e525034e51e` (E13 RC2 GPU analysis);
  - `agent-a5ae433b82da4d859` (NB1 builder);
  - `agent-aa8ba632bd68f6706` (NB2 builder);
  - the NB4 builder agents started at 19:00Z were still running and are not used.
- No other top-level transcript in the project folder was modified after 2026-10-30.

**How to read this file**
- Timestamps are UTC as written in the transcript. Add 6 h for local time.
- "user" means the user's own words.
- "GPT (pasted by user)" means an external report that the user pasted into the chat.
- "assistant" means the main Claude session's conclusion.
- No credentials appear in this file. The token-file path rule is cited only as a rule.

---

## 1. Measured facts relevant to the tick

### 1.1 The stage-0 gap itself (NB1 R2, boot B1, torch profiler, 4 ranks)

- **F1. Per-stage numbers** from `nb1_results.json` `sections.B1` (MAIN, tool result 2026-10-31T17:13:58Z):

  | Stage | Critical path per micro-batch | GPU-busy union | Summed kernels | `scope_ms` | Verify tokens |
  |---|---|---|---|---|---|
  | stage 0 | 50.645 ms | 27.242 ms | 29.515 ms | 13.045 | 16 |
  | stage 1 | 31.551 ms | 23.295 ms | 25.824 ms | 9.337 | 16 |

  - The JSON also records `binding_stage = stage0`, `heavy_stage = stage0`, `tick_critical_path_ms = 50.645` and `light_stage_idle_ms_per_tick_est = 27.35`.
  - Its note says: "critical path = the GPU wall span of a micro-batch on the binding stage (timeline); summed_kernel_ms adds overlapping streams; PP0 and PP1 run concurrently".
  - Category shares in the same JSON: gemm_marlin 3.3, gemm_other 9.0, sampling 0.2, other 4.2.
  - The top kernel by total time is `ncclDevKernel_Broadcast_RING_LL`, at 1,713,554 µs total.
- **F2. The assistant's reading** (MAIN 2026-10-31T17:15:02Z, msg "NB1 R2 … শেষ হয়েছে"):
  - About 23 ms per tick, the GPU on stage 0 is waiting.
  - The 63.5 % of kernel time inside NCCL Broadcast is mostly that wait.
  - Stage 1 also waits about 27 ms per tick.
  - So the tick is bound by synchronisation, host scheduling and launch gaps, not by MoE bandwidth.
  - A host-side timeline is needed to tell CPU scheduler, broadcast and launch apart.
  - The memory note written at 17:14:36Z calls it "~23 ms/tick wait (NCCL broadcast spin) = biggest lever".
- **F3. B1 and B3 have the same tick.** Pin +1.0 GiB adds capacity only (MAIN 17:15:02Z).
  - B3 has 214 blocks / 738,142 tokens, against 149 / 513,940 in B1.
  - 8k@8: 261.0 (B1) vs 254.6 (B3).
  - 40k peak: 227.9 @7 (B1) vs 249.5 @8 (B3).
  - 90k@5: B1 had 1 preemption at KV 0.97; B3 had none, at KV 0.68 and 187.8 tok/s.
- **F4. NB1 R2 B2 bundled five changes and failed the oracle** (MAIN 17:15:02Z).
  - The five changes: SSM bf16, seqs 16, captures to 64, shared draft workspace, RecoverSSM commit graph.
  - One of 4 oracle rows diverged at token 31 (margin 0.75). The others were near ties (0.125 / 0.125 / 0.0).
  - The commit graph was captured only for n = 16; the other steps ran staged-eager. Its speed was not measured.
  - The draft-workspace share logged "0 B" (no effect).
  - SSM bf16 gives 0 capacity, because the state drops from 1.73 to 0.95 MB but is padded to the same 2 MiB page.
- **F5. Whether the B1 traces contain CPU activity is unknown.**
  - The assistant asked NB4 agent C to check it, and to state the profiler settings NB4 needs if they do not (MAIN tool_use 2026-10-31T18:59:48Z).

### 1.2 Graph modes

- **F6. NB2 SEED2 profiler, 40 forwards** (MAIN 2026-10-31T16:34:22Z):
  - FULL 0, PIECEWISE 20, EAGER 20, eager kernel share 0.083.
  - The trace does not say which forwards ran eager.
  - The eager helper kernels seen were `prepare_commit_plan`, `compact_conv_state` and `recoverssm_align`, attributed to the RecoverSSM commit graph being off.
  - Kernel shares: NCCL 59.5 % (Broadcast_RING_LL 905 ms), Marlin 22.2 %, other GEMM 8.2 %, other 5.2 %, GDN 3.9 %.
  - The assistant's warning: NCCL kernel time includes waiting for the other stage, so 59.5 % is only an upper bound on communication cost.
- **F7. FULL graphs are captured; the problem is that decode never uses them.**
  - The RC2 FP8 server log captured 6 FULL and 10 PIECEWISE target graphs (BF16: 3 + 7). Sources:
    - GPT RC2 audit, pasted 2026-10-31T07:44:59Z: "`server_merlin_fp8.log:384–417`-এ FULL capture 6/6 এবং PIECEWISE capture 10/10";
    - the RC2 analysis agent `agent-a60fb6e5…` final report.
  - Target mode is FULL_AND_PIECEWISE. The draft runs prefill FULL_AND_PIECEWISE and decode FULL_DECODE_ONLY.
  - FP8 capture grid [1,2,3,4,6,8,12,16,24,32]. The BF16 grid was missing sizes ([1,2,3,4,6,8,12]).
  - So arm G is a dispatch question (why replay never picks FULL), not a capture question.
- **F8. Capture sizes count tokens = seqs × (1+K)**, from `cudagraph_utils.py:268-347` and `compilation.py:1543-1571` (A1/A7 mining, MAIN 06:13Z and 06:21Z).
  - Uniform-decode graphs come in multiples of 4 at K3.
  - Captures to 32 cover 1/2/3/4/6/8 requests per forward.
  - Under PP2 each micro-batch holds about C/2 requests, so 32 covers 16 streams, 48 covers 24 and 64 covers 32 (MAIN 08:05:52Z).
  - The scheduler alternates two half-batches (`scheduler.py:792-796`).
- **F9. The QSA, GDN and GDN-RecoverSSM backends declare UNIFORM_BATCH full graphs.**
  - So the 27B FlashInfer downgrade (FULL to PIECEWISE under spec decode) should not apply.
  - But on the 27B, Merlin's draft ran `decode:NONE` and the RecoverSSM commit logged "0 captured" (PLAN §2.3, quoted in the user's IDE selection at 2026-10-31T18:03Z).
  - PIECEWISE breaks the graph at every QSA layer, because QSA is a split op.
- **F10. FULL-graph gains measured on other setups:**
  - E23 FULL (27B, 16 streams): G3 (FlashInfer + PR 60532, PROVEN_FULL) 364.7 vs G0 A/A mean 357 (+2 %). Triton FULL G1 330.7 and G2 334.9 (−7 %). (MAIN 12:23Z, 16:47Z.)
  - E23 SMOKE: G3 +3 %. Launch count fell a lot, but the NCCL share did not change (A7, MAIN 06:21Z).
  - Community recipe (todi, 3090): FULL_AND_PIECEWISE gives decode "about 2× PIECEWISE" (MAIN 00:47:52Z; A5).
  - PR 60532 author: +36-47 % at batch 1 (PLAN §2.3).
  - The assistant: PR 60532 does not apply to Flash-Next, because it is a FlashInfer-attention fix and Flash-Next QSA/GDN already ran FULL in RC2 (6/6) (MAIN 07:59:36Z).
  - vLLM #57759 (4×L20, sm_89 PCIe): one FULL capture failure under heavy host load; the workaround was PIECEWISE (PLAN §2.3).

### 1.3 RecoverSSM commit

- **F11. Mode and coverage** (MAIN 07:43:16Z and 07:59:36Z; RC2 agent report):
  - `VLLM_RECOVERSSM_COMMIT_GRAPH` is unset, so the commit runs eager by default, and no commit-capture line appears.
  - Merlin has three modes: eager, staged and graph. They can be switched at runtime and give bit-identical results.
  - Graph mode removes 5 Triton launches per GDN group per step.
  - **PP-deferred commits always run eager.** The assistant wrote that NB1 should first count the share of steps that can use the graph.
  - RecoverSSM uses spec_query_len 4 and align mode, with 6-7 layers per stage (GPT RC2 audit 07:44:59Z).

### 1.4 PP2 behaviour and the tick model

- **F12. PP2 overlap appears to hold under MTP K3**, counting requests per step: from 2 streams on, every step carries exactly C/2 requests. Examples: 4 streams 6.03 = 2 × 3.017; 8 streams 12.62 = 4 × 3.154 (MAIN 08:32:40Z; RC2 agent).
  - 2 streams give 1.53× one stream.
- **F13. Step fits on the 7 RC2 steady windows:**
  - step_ms = 25.60 + 5.79·r (R² 0.988), or 24.18/24.21 + 0.1759·D(4r), where r = requests per micro-batch and D = distinct experts touched.
  - Leave-one-out error: mean 1.5 %, max 4.0 %. The intercept stays at 23.2-25.0 ms.
  - Valid only for C = 2-8 (MAIN 08:38:19Z).
  - The 24 ms intercept is FITTED and "may contain stage idle and hand-off, not only compute" (PLAN v3.5 §1.1).
- **F14. Tick vs cycle:** one tick = one micro-batch, in which C/2 requests advance; a request advances once every 2 ticks.
  - At 8 streams, cycle = 2 × 48.2 = 96.4 ms (MAIN 07:47:09Z).
  - Every notebook must write `tick_ms` and `cycle_ms` separately.
  - PLAN v3.5 asks for four rates: tick, request cycle, completed-micro-batch rate and wall clock.
- **F15. RC2 stage-imbalance estimate:** 1 stream 44.1 ms (serial) vs 2 streams 30.3 ms (overlapped). That suggests a heavy stage of about 30 ms and a light stage idling about 14-16.5 ms per step (MAIN 08:32:40Z; RC2 agent).
  - At the time, the heavy stage was assumed to be stage 1, which carries the LM head and 3 MTP draft passes. **The NB1 trace contradicts this: stage 0 is the heavy one** (see §4, C10).
- **F16. Per-tick cost estimate (A7, 8 streams, T = 16; estimates, not measurements):**

  | Part | ms per tick |
  |---|---|
  | MoE | 4.7-21 |
  | Dense GEMM | 6.6-12 |
  | Vocab heads, BF16 | 10.2 (int8 6.4, int4 4.5) |
  | TP2 all-reduce | 1.5-3.8 |
  | Logits gather | 2-3 |
  | PP hand-off | 1-3 |
  | QSA | 2-5 |
  | GDN + RecoverSSM | 1-3 |
  | PLE | < 0.5 |
  | Guard reader | 0.3 |

  A7 also estimated "stage 0 idles 5-15 ms/tick" (PLAN §1, user IDE selection 18:03Z).
- **F17. PP2 + async interleaves two micro-batches:** `max_concurrent_batches = pp + 1` (`vllm/config/vllm.py:607-617`) (A7; PLAN §1).
  - GPT (pasted 08:23:14Z): "vLLM source-এ PP-এর max_concurrent_batches ও async overlap logic রয়েছে". The capture arithmetic holds "যদি scheduler সত্যিই সমান দুই microbatch ব্যবহার করে" (if the scheduler really uses two equal micro-batches).
- **F18. PP hand-off warnings** (RC2 log lines 296-303, MAIN 07:35:30Z):
  - a non-writable buffer warning;
  - NCCL creates the per-pair communicators later ("PP send/recv uses lazily created 2-rank communicators", RC2 agent). Their cost is unknown.
  - All NCCL groups (tp, pp, ep) use pynccl 2.29.7.
- **F19. `max_num_scheduled_tokens = 1024`** triggers a "suboptimal performance" warning (GPT RC2 audits, pasted 07:33:37Z and 07:44:59Z). The profile uses chunk 1024. NB1's builder says the chunk cannot be changed at runtime (MAIN 09:55:49Z).

### 1.5 MoE, dense and LM head (NB3, 15-min kernel bench)

- **F20. D5 Marlin MoE (W4), T16:** 771 µs per layer × 24 MoE layers per stage ≈ 18.5 ms per tick, out of a whole tick of 42.7-48.2 ms (about 40 %) (MAIN 12:23:03Z).
  - Bandwidth reached 204-252 GB/s, against the L4 peak of 300.
  - Marlin config tuning gives 0.4-1.1 % (not a lever).
- **F21. D4 dense GEMM (cuBLAS BF16):**
  - The LM head (TP2 shard) takes 2.5 ms per call, and the MTP draft head reads it on every draft step.
  - QKV 137 µs, o_proj 65 µs.
  - The alternative kernel did not compile on sm_89 (`UNSUPPORTED_SM89`).
- **F22. D7 world-2 P5 vs NCCL**, measured on both TP2 pairs:
  - 5-123 KB: P5 1.39-1.68× faster (T16: NCCL 49.0 µs vs P5 30.7 µs).
  - 246-328 KB: 1.25×. 0.5-0.66 MB: 1.06×.
  - Numerically bit-exact, but labelled `PASS_BOUND_ONLY`, because the two pairs overlapped only 0.43 (< 0.5) when run at the same time. So the speed under PP2 concurrency is unproven.
  - Estimated gain: about 48 all-reduces × about 18 µs ≈ **0.9 ms/tick, about 2 %** (MAIN 12:23:03Z).
- **F23. P5 at world 4 (E23 FULL, 983 KB):** 0.94× NCCL (slower); the INT8 variant was 1.43×.
  - The E23 `BENCH_ERROR` was a port collision (`EADDRINUSE`) in one sub-step; all correctness checks passed (MAIN 12:23:03Z).
  - Flash-Next TP2 messages are 80-160 KB (5,120 B per token, 2 per layer; no all-to-all with EP at dp = 1), so P5 applies only on the world-2 path (A7; PLAN §6.6).
- **F24. Earlier comms facts:**
  - 27B: NCCL is 36 % of the step on vLLM and 51 % on SGLang.
  - Plain NCCL over SHM with P2P off is the best L4 path: 343.9 µs per 983 KB in a graph, 49.7 µs at 64 KB.
  - P2P is 5.5× slower; PCIe-IPC 0.39-0.68×; CTA 4/8/16/32 slower; `NCCL_SHM_USE_CUDA_MEMCPY` slower and hangs in capture (A7, MAIN 06:21Z).
  - A5 instead reports NCCL SHM running on only 2 channels, at about 35 % of the host path's potential, and calls `NCCL_MIN_CTAS`/`MAX_CTAS` 4-8 untested (MAIN 06:16:21Z). See §4, C14.
  - jon-nielsen 4×3090 no-P2P kit: `NCCL_PROTO=LL128` +6.2 %; no `expandable_segments` +13-37 % multi-stream; fused_moe aux-stream overlap −20 % (PLAN §2.4).
  - Also seen: `NCCL_CUMEM_ENABLE=0` (todi), `NCCL_NTHREADS=512` untested.

### 1.6 MTP and acceptance

- **F25. NB2 SEED2 vs CTRL, 8k@8** (MAIN 16:34:22Z):

  | Run | Config | tok/s | Tick | Tokens per step | Acceptance a | Per-position acceptance |
  |---|---|---|---|---|---|---|
  | SEED2 | K2 + INT8 head, seqs 16 | 240.9 | 43.0 ms | 10.35 | 2.587 | 0.865 / 0.723 |
  | CTRL | RC2 K3, BF16 head, seqs 8 | 259.1 | 48.9 ms | 12.68 | 3.17 | 0.867 / 0.716 / 0.586 |

  - K2 makes the tick 12 % shorter but carries 18 % fewer tokens per step, a net −7 %. The same result came twice (the first SMOKE: 242.6 vs 261.8).
  - **The INT8 head does not lower acceptance at positions 1-2.**
  - Other numbers:
    - ttft median 9.06 s vs 6.92 s;
    - warm-up 54.4 s vs 34.6 s;
    - graph memory 0.42-0.52 vs 0.33-0.43 GiB;
    - model load 327-349 s vs 415-433 s per worker.
- **F26. Long-run acceptance decays:** 2.72 on a natural long xhigh run, falling from 3.18 to 2.69 over 33 min. The 90 s windows at the start of an answer read 3.0-3.25 (MAIN 08:22:33Z).
  - 27B: synthetic prompts gave 3.57, real Poetiq reasoning 2.69 (A1).
  - RC2 per-position acceptance: 0.834 / 0.660 / 0.516.
  - NB5 QUALITY FP8 (real Duck) median acceptance: 2.83 (MAIN 18:30:17Z).
- **F27. MTP depth on the 27B, 40k@16:** MTP0 222.5, MTP1 319.2, MTP3 359.1 (+61 %). Merlin MTP4 acceptance 3.57-3.76 at 8-24 streams. Flash-Next K4 is untested (MAIN 16:47:34Z).

### 1.7 Noise, boot and capacity facts used by the NB4 budget and gates

- **F28. A/A noise:**
  - E23 stock MTP3 40k@16: 353-361 (±1 %).
  - TP1×4: 503.8 / 506.8 / 506.2 (±0.6 %).
  - The assistant's rule: differences < 2 % are noise in every NB4 arm (MAIN 16:47:34Z).
- **F29. Boot timing:**
  - Merlin first boot 390 s, warm 135-141 s (E23).
  - Flash-Next worker load 327-433 s (NB2).
  - One RC2 FP8 worker took 873 s under page-cache pressure, against about 125 s for the others; a page-cache drop before each boot fixes this (MAIN 07:09:23Z).
  - An MTP boot needs 423-777 s to become ready (reviewer finding on the klee100 notebook, MAIN 18:25:42Z).
  - NB1 R2 took 75.8 min for 3 boots.
- **F30. Base capacity (NB5 QUALITY FP8 = the NB4 base, MAIN 18:30:17Z and 17:41:50Z):**
  - pool 214 blocks, F = 6, 9 workers (seqs 9);
  - p50 tok/s by running streams: 9 → 224.9, 8 → 214.2, 6 → 184.3, 3 → 118.4;
  - whole-window average about 172, because of the tail;
  - prefix hit 80.9-83.4 %;
  - KV peaked at **99.1 %** with 0 preemptions;
  - accuracy 13/15 tasks, 18/20 test inputs.
- **F31. KV edge = stall.** On E23, 90k@20 (KV 0.999) and 120k@16 (0.997) each gave 1 preemption plus a stalled stream; every arm at ≤ 0.97 was clean. NB2 SEED2 reached 96.6 % at 16 streams (MAIN 16:47:34Z).
- **F32. Merlin keeps up to 16 finished sessions**, so `kv_cache_usage_perc` is not live demand; gates should use running blocks (PLAN changelog v3.3, user IDE selection 18:03Z).
- **F33. Same seed ≠ same text under batching.** The failed run's row 37 and CTRL's seed 1234 had identical prompts, yet only the first 4 tokens matched, because neighbouring requests change the numerics and the sampling path (MAIN 16:34:22Z).

### 1.8 The 1500 arithmetic

- **F34. Target:** 1500 tok/s is the whole-run average (user, 2026-10-31T06:18:49Z, below), which needs a steady rate of about 1,580-1,900.
  - Steady = (C/2) · a / t_tick.
  - Required tick for a steady 1,700 at a = 2.72: C 8 → 6.4 ms, C 16 → 12.8 ms, C 24 → 19.2 ms, C 32 → 25.6 ms (PLAN v3.5).
  - The assistant's summary (MAIN 18:56:27Z): we are at about 225 tok/s (9 streams, tick about 48 ms), which is 15 % of 1500. A 1500 run needs for example 24 streams and a tick of about 22 ms. If the 23 ms gap could be removed, the tick could roughly halve.

---

## 2. User decisions and rules that NB4 must follow

Each rule quotes the Bengali (`MAIN`, UTC timestamp), then gives an English gloss. Quotes are verbatim.

### 2.1 GPU, push and approval

- **R1.** 2026-10-29T22:42:08Z: "আমার অনুমতি ছাড়া জিপিইউ কমিট করবি না।"
  - Gloss: never make a GPU commit without my permission.
- **R2.** 2026-10-31T05:07:00Z: "পুশ পরে করিও,আমি অনুমতি দিলে।"
  - Gloss: push later, only when I allow it.
- **R3.** 2026-10-31T04:29:51Z: "সিপিইউ চালাও আগে"
  - Gloss: run the CPU kit first.
- **R4.** 2026-10-31T04:50:49Z: "আমাদের এখন সময় কম।সিপিইউ কিট চালালে ৪-৫ টা চালাতে পারবি একসাথে যাতে সময় কম লাগে।"
  - Gloss: time is short; run 4-5 CPU kits in parallel.
- **R5.** 2026-10-31T09:49:37Z: "যেগুলো শেষ হবে,সেগুলো ক্যাগল সিপিইউতে পাঠাবে,আর আমাকে অডিটের জন্য জিপ করেও দিবে।"
  - Gloss: every finished notebook goes to a Kaggle CPU kit and also to an audit zip for me.
- **R6.** 2026-10-31T14:12:38Z: "ফ্লাট জিপ দিও,এগুলো জিপিটি ওপেন করতে পারে না।"
  - Gloss: give flat zips; GPT cannot open nested ones.
- **R7.** 2026-10-31T17:08:04Z: "তুমি রুলস অমান্য করো,আমাদের নিয়ম ছিল একটা ফোল্টারে রাখা,তাও স্পেসিফিক,তুমি এখানে কেন রাখো?"
  - Gloss: you break the rule; zips go in one specific folder (the experiment's `Experiment Review/Audit Zips/`), never `_scratch`.
- **R8.** 2026-10-31T11:38:08Z: "২ টা জিপিইউ ক্যাগল একবারে দেয় কমিট এর জন্য।আর আরো একটা নেওয়া যায় ইন্টারএক্টিভ er jonno … total 3"
  - Gloss: Kaggle allows 2 GPU commits at once per account, plus 1 interactive session.
- **R9.** 2026-10-31T03:13:16Z: "…নোটবুক থেকে মডেল হিসেবে রাখতে ১৫ গুণ দ্রুত লোড হয়।"
  - Gloss: payloads kept as Kaggle Models load about 15× faster than notebook outputs.
- **R10.** 2026-10-31T15:46:57Z: "ভুল ভাল হিসাব দিলি,তুই আগেও অনেক হিসাব করেছিস।"
  - Gloss: your quota accounting was wrong.
  - The assistant recorded the fix: use the web quota value, and remember that 4×L4 bills 2× (MAIN 15:51:07Z).
  - NB4's 150-min hard stop costs about 5 h of quota.
- **R11.** 2026-10-30T04:35:55Z: never print the contents of the token-file folder. A security rule; no content is cited here.

### 2.2 Audits, fixes and loops

- **R12.** 2026-10-30T02:07:00Z: "আগেই কিছু ফিক্স করবে না।শুধু প্রায়োরিটি ফিক্স লিস্ট বানাবে।তুমি গুরুত্বপূর্ণ কিছুই বাদ দিও না,ডিপলি দেখবে."
  - Gloss: do not fix anything early; first make a priority fix list; leave out nothing important; look deeply.
- **R13.** 2026-10-30T11:32:30Z: "২ টা এজেন্ট দিয়ে তুমিও অডিট করাও।এরপর ফিক্স লিস্ট বানাও,এরপর সব একসাথে পরিবর্তন করবে।"
  - Gloss: also audit with 2 agents, then make the fix list, then change everything at once.
- **R14.** 2026-10-30T02:35:31Z: "২ টা এজেন্ট কে দিয়ে স্বাধীন ভাবে দেখে নাও ,কোনো ফিক্স বাদ গেছে কি না,২ টা এজেন্ট দিয়ে দেখিও নতুন কোনো বাগ ডুকলো কি না"
  - Gloss: 2 independent agents check that no fix was missed, and 2 check that no new bug came in.
- **R15.** 2026-10-31T08:21:27Z: "আগেই কিছু পরিবর্তন করিও না।আগে ভ্যালিডেট করো"
  - Gloss: change nothing before validating the claims.
- **R16.** 2026-10-31T09:10:56Z: "ফিক্স এর কিছু প্রমাণ ফেলে ফিক্স লিস্টে তুলবে।"
  - Gloss: a fix goes on the list only with evidence.
- **R17.** 2026-10-31T10:47:06Z: "…আমাদের মারাত্বক ব্লকার,বাগ না থাকলে দ্রুত জিপিইউ রানে যেতে হবে,এভাবে লুপে পড়ে যাচ্ছি,"
  - Gloss: without a fatal blocker or bug, go to the GPU run fast; we are falling into loops.
- **R18.** 2026-10-31T16:13:22Z: "আমরা কি আবার লুপে পড়ে গেলাম? আর অপ্রয়োজনীয় এক্সপেরিমেন্ট করতেছি?"
  - Gloss: are we looping again, running unnecessary experiments?
- **R19.** 2026-10-31T18:00:55Z: "তুমি আরো ডিপলি এনালাইসিস করো,বাগ পাইলে সেগুলোও ঠিক করবে,অডিট গুলো আবার দেখে নাও"
  - Gloss: analyse more deeply, fix any bugs you find, re-read the audits.
- **R20.** 2026-10-30T11:47:13Z: "তুই আরো ডিপ্লি এনালাইসিস কর,তুই অনেক কিছুই বাদ দিস"
  - Gloss: analyse more deeply; you leave out a lot.
- **R21.** 2026-10-31T18:50:33Z: "ফিক্স এজেন্ট ৫ টা নিও।আলাদা আলাদা ভাগ করে দিলেই তো হতো"
  - Gloss: use 5 fix agents and split the work between them. The assistant implemented this as one agent per file.
- **R22.** 2026-10-31T07:02:10Z and 05:14:22Z: "একসাথে শুরু করো,৫-১০ টা এজেন্ট লাগলে নিও" and "…দ্রুত পুশ করো,প্রয়োজনে ৩-৫ টা এজেন্ট নিও"
  - Gloss: work in parallel with several agents.

### 2.3 Notebook design

- **R23.** 2026-10-31T13:36:50Z: "admission part দিয়ে তুই জটিলতা বাড়াচ্ছিস,এটা সিম্পল রাখা যেত।"
  - Gloss: you add complexity with the admission part; keep it simple.
- **R24.** 2026-10-31T13:21:59Z: "[…] নোটবুক একটাই বানা,দুইটা BF16:  FP8: কাজ করবে … রানটাইম ৩ ঘন্টা,"
  - An insult is elided.
  - Gloss: build one notebook with both phases, not two notebooks.
  - Pattern for NB4: keep the arms in one notebook.
- **R25.** 2026-10-31T18:42:19Z: "চতুর্থ একটা বুট, K3 = klee100 + FP8 KV + MTP K3। এটাকে প্রথম বুট করো"
  - Gloss: make the most important boot the first boot.
  - Pattern: put the arm that matters most first, so it is never lost to the budget.
- **R26.** 2026-10-31T13:16:36Z: "২ টাই করো,আর হার্ড টাইম লিমিট ৯০ মিনিট করে দিবে।"
  - Gloss: hard time limits are set per notebook.
  - The NB4 spec has 150 min. The user did not set NB4's limit in these transcripts.
- **R27.** Smoke rule, as restated by the assistant (MAIN 09:10:05Z) after the user's question at 09:05:27Z ("এর আগে কি একটা স্মোক দিয়ে নিবে।জিপিইউ যাতে কম খরচ হয়?"):
  - A separate smoke is needed only for runs of 3-4 h or more. Shorter notebooks rely on fail-fast gates.
  - NB4 at 150 min therefore needs no separate smoke.
- **R28.** 2026-10-30T10:26:05Z: "আমি ফুল ই চালাবো।এভাবে সময় নষ্ট হচ্ছে"
  - Gloss: I'll run the full notebook; smoke steps waste time.

### 2.4 Direction, scope and the target

- **R29.** 2026-10-31T05:36:32Z: "…আমি FLASH NEXT এর জন্য যা যা লাগবে তাই করবো ২৭ B আপাতত অফ রাখ"
  - Gloss: do only what Flash-Next needs; keep the 27B off for now.
- **R30.** 2026-10-31T06:18:49Z, the user's own line: "তাই আমার সুপারিশ: লক্ষ্য হোক পুরো run-এর গড় ১৫০০, যার জন্য steady লাগবে ~১,৬০০-১,৯০০।"
  - Gloss: the target is a whole-run average of 1500, which needs a steady rate of about 1,600-1,900.
- **R31.** 2026-10-31T07:23:21Z: the user quotes GPT's "একটি বড় strategic correction: ১৫০০ tok/s-এর জন্য সব ১৫টি lever পরীক্ষা করা বাধ্যতামূলক নয়। … কেবল সবচেয়ে high-upside ২–৩টি পরিবর্তনে GPU budget খরচ করাই ভালো।" and adds "বড় strategic correction এটাই বাস্তবায়ন করবে।"
  - Gloss: implement this. After the screening notebooks, spend GPU only on the 2-3 highest-upside changes. This became PLAN §4.1.
- **R32.** 2026-10-31T16:42:15Z: "আমরা ১২ ঘন্টার যে একটা বড় রান সামি ভাইয়ের আইডিতে দিছিলাম।সেটা থেকে আমরা কিছুই নিচ্ছি না কেন?"
  - Gloss: use the E23 FULL results. This produced the ±1 % noise floor, FULL +2 % / −7 %, the K4 arm and P5 world-2 only.
- **R33.** 2026-10-31T16:26:24Z: "১৫০০+ এর সেই এক্সপেরিমেন্ট গুলো কই গেলো?" and 18:55:51Z "আমাদের ১৫০০+ এর অগ্রগতি কতদূর?"
  - Gloss: keep the 1500 levers moving.
- **R34.** 2026-10-31T18:58:33Z, the NB4 request itself: "NB4, tick-এর notebook: stage-0-এর ২৩ ms ফাঁক কোথা থেকে আসছে, সেটা host timeline দিয়ে বের করা; FULL graph, commit graph আর P5-এর আলাদা আলাদা A/B। … এটার নোটবুক বিল্ড করা শুরু করো"
  - Gloss: build NB4: find where the 23 ms stage-0 gap comes from using a host timeline, and run separate A/Bs for FULL graph, commit graph and P5.
- **R35.** 2026-10-31T19:02:25Z: "একটা এজেন্টকে আমাদের গত দুইদিনের কথোপকথন পড়াও,tick-এর notebook এর জন্য গুরুত্বপূর্ণ কথোপকথন বের করতে বলিও,… হতে পারে গুরুত্বপূর্ণ কিছু থাকবে যা আমরা বাদ দিচ্ছি"
  - Gloss: this research; find what we may be missing.
- **R36.** 2026-10-30T08:41:27Z, on Merlin's default `reasoning_effort` "medium": "xhigh কেন দিস নি?"
  - Gloss: why didn't you use xhigh? Every NB4 request must force xhigh.

### 2.5 Standing PLAN §0 rules the user endorsed (as quoted in the user's PLAN selection, 18:03Z)

- **R37.** Short speed stage of about 10-15 min; audit with 2 agents, then a Kaggle CPU kit; push only after the user's go; few arms per notebook; speed measured in short runs.
- **R38.** Never substitute configured capacity for observed running streams. Always report the steady peak and the run average. Save the full thinking traces.
- **R39.** Every speed winner needs a paired ARC accuracy run (same tasks) before adoption.
- **R40.** Evidence rules:
  - a partial TF is never CLOSE (TF_INCOMPLETE / TF_PARTIAL_BUDGET);
  - no concurrency decision from an unknown or mismatched KV pool or an unmeasured F;
  - MTP acceptance is certified only against an MTP-off reference of the same checkpoint, with per-position acceptance always recorded;
  - a missing memory reading means the base pin, labelled UNVERIFIED_HEADROOM.
- **R41.** Engine gates: fail-fast after the short checks, non-finite logprob gate, real running == level, boot gate < 150 MiB, graceful kill + /dev/shm cleanup, every request saved.

---

## 3. Ideas, levers and warnings from the conversations that the NB4 spec does not cover

Ordered by how directly each one bears on the stage-0 gap or on an NB4 arm.

### 3.1 The gap analysis (T0)

- **M1. Serial-stage hypothesis (analyst's arithmetic from F1, not stated in the conversations; flagged for checking).**
  - Stage-0 GPU-busy 27.2 ms + stage-1 GPU-busy 23.3 ms = 50.5 ms, almost exactly the stage-0 critical path of 50.6 ms.
  - The 23 ms stage-0 gap is also about the stage-1 busy time.
  - These numbers are what you would see if, at C = 8 with 16 verify tokens, the two stages ran mostly in series per tick, with stage 0 waiting in the NCCL Broadcast for stage 1's sampled and draft tokens. They do not look like the overlap inferred from the C/2-per-tick request counts (F12).
  - T0 should measure this directly: the overlap fraction of stage-0 and stage-1 busy intervals on one wall clock, per tick.
  - This needs cross-rank clock alignment of the 4 profiler traces, which the spec does not mention.
  - If confirmed, the lever is the PP return path (sampled-token broadcast, scheduler round trip), not kernel speed. FULL graph and P5 would then attack only the busy parts.
- **M2. The heavy stage is stage 0, not stage 1.**
  - RC2/A7 assumed stage 1 is heavy (LM head + 3 MTP draft passes) and stage 0 idles (F15, F16).
  - NB1 R2 gives `heavy_stage = stage0`, with 27.2 vs 23.3 ms busy.
  - T0 should state per stage what runs there: the PLE pull, embeddings, LM head, draft passes, sampler, and where the RecoverSSM commit and the broadcast originate. The spec does not list per-stage op placement.
- **M3. The PLE host pull is a gap category the spec lacks.**
  - Flash-Next keeps the PLE table (about 51 GB FP8) in pinned host RAM, and Merlin has its own PLE pull transport (`vllm/v1/ple_offload/worker.py`) (GPT/PLAN text pasted 06:08:36Z).
  - The assistant flagged "PLE CPU thread: the PLE lookup may be CPU-bound; record and vary `OMP_NUM_THREADS`" (MAIN 05:32:17Z).
  - The A7 estimate of < 0.5 ms is not a measurement.
  - The spec splits the gap into GPU busy / NCCL wait / host launch / scheduler-IPC. An "H2D / PLE copy wait" slice should be added, along with the host thread count.
- **M4. Host CPU contention.**
  - Kaggle has 4-8 CPU cores (prompt facts, MAIN 05:42:30Z). They serve 4 workers, the engine core, the API server, PLE threads and, in arm P, the P5 host path.
  - The timeline should record per-process CPU usage. P5's host-staged copies could widen the very host-launch gap T0 measures (see M13).
- **M5. Known trace-classifier bugs (A7, MAIN 06:21:16Z).**
  - The E23 trace classifier showed GDN at 0.2 % when it was really 6.1 %, and no forward annotation matched.
  - The NB4 analysis module must not reuse that classifier unfixed. NB1's own summary code is the comparable one.
- **M6. The wait must be split from the transfer.**
  - The assistant wrote that before claiming a broadcast gain, an nsys timeline should separate waiting from the actual transfer (MAIN 16:34:22Z).
  - NB1's builder found that nsys and ncu cannot be attached to the running server (MAIN 09:55:49Z).
  - So with torch-profiler-only data, the "NCCL wait" split must come from cross-rank alignment (M1), not from kernel durations.
- **M7. Lazy PP communicators** and the non-writable-buffer warning (F18) can produce first-use stalls. The profiled window must begin after warm-up traffic, and the analysis should drop the first ticks.

### 3.2 Arm G (FULL graph)

- **M8. FULL graphs are already captured (6 FULL / 10 PIECEWISE, F7).**
  - The research question is why replay dispatch never picks them. Candidates raised in the conversations:
    - non-uniform micro-batches;
    - mixed prefill/decode ticks under the `max_num_scheduled_tokens = 1024` chunk;
    - the draft's FULL_DECODE_ONLY path;
    - PP intermediate stages.
  - Which 20 of the 40 forwards ran EAGER is unknown. If they are the draft forwards, arm G is really a draft-graph arm.
- **M9. FULL graphs can be slower** (Triton FULL −7 %), and the 27B FlashInfer gain was +2-3 %.
  - The only large claims are external: todi 2× on 3090, PR 60532 +36-47 % at batch 1.
- **M10. #57759: FULL capture can fail under heavy host load** on sm_89 PCIe. A PIECEWISE fallback with a clear label should be ready.
- **M11. Capture lists must cover (C/2)(1+K) per micro-batch.**
  - The base runs seqs 9, so micro-batches of 5 and 4 requests = 20 and 16 verify tokens. Both must be FULL-captured sizes, while the windows run 8 streams (4 + 4 = 16).
- **M12. FULL graphs change kernels, so greedy EXACT may break legitimately** (F33). The NEAR_TIE rule (spec) is needed. Acceptance per position must not drop.

### 3.3 Arm C (RecoverSSM commit graph)

- **M12a. PP-deferred commits always run eager**, whatever `VLLM_RECOVERSSM_COMMIT_GRAPH` is set to (F11).
  - The spec's C arm does not measure the share of commits eligible for the graph under PP2. NB1 was meant to count it first.
  - Without that count, a NO_GAIN result cannot tell "graph mode does not help" from "graph mode almost never applies".
  - C should report:
    - commits per tick split into graph / staged / eager / PP-deferred;
    - the capture sizes actually used. B2 captured only n = 16.

### 3.4 Arm P (P5)

- **M13. P5 under concurrency is unproven.** Two-pair overlap was 0.43 (< 0.5), so the result is `PASS_BOUND_ONLY` (F22).
  - In the engine both TP pairs (stage 0: GPUs 0-1, stage 1: GPUs 2-3) all-reduce at the same time over shared host memory and CPU.
  - P's timeline should show whether host-staged all-reduces lengthen host gaps elsewhere.
- **M14. Ports.** The E23 P5 `BENCH_ERROR` was a port collision (`EADDRINUSE`, F23). The engine integration needs unique ports or rendezvous per pair and per boot. A port-reuse fix also appears in the klee100 fix list (MAIN 18:25:42Z).
- **M15. Use the path the engine actually takes.** All groups use pynccl, and Merlin's no-P2P profile sets `disable_custom_all_reduce = true` and unsets `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_P2P_LEVEL` and `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`. Sources:
  - GPT Merlin deep research, pasted 01:03:09Z / 01:05:06Z, quoting `tools/merlin/profiles/flash-next.json`;
  - A1: on TP2, custom all-reduce may enable itself, so check in the log that `--disable-custom-all-reduce` really takes effect (MAIN 06:13Z).
  - P5 must hook the all-reduce actually used at decode, inside captured PIECEWISE graphs. A7 calls P5 "graph-capturable", but the PLAN §4.1-7 rule asks for 1000+ graph replays before integration.
- **M16. P5 belongs on world-2 TP groups only**, never PP, EP or world-4 (PLAN §6.6).

### 3.5 Levers the conversations assigned to NB4 that the spec drops

- **M17. One K4 arm.** "K4 এখনও চালানো হয়নি, আর তৃতীয় পজিশন 0.586 হারে গৃহীত হয়। তাই NB4-এ একটা K4 রান থাকবে" (MAIN 16:47:34Z; PLAN v3.2 §6.3).
  - Gloss: K4 has not been run yet and the third position is accepted at 0.586, so NB4 will have one K4 run.
  - K changes require a trailing `--speculative-config`, because the last argument wins and no env exists (MAIN 09:00:48Z).
  - The captures must grow to (C/2)·5.
- **M18. The QSA FP8 scale-calibration arm.** "scale calibration-এর একটা রান NB4-এ যাবে" (MAIN 16:47:34Z); PLAN §6.8: "a scale-calibration arm belongs in NB4".
  - Gloss: one scale-calibration run goes into NB4.
  - GPT (pasted 17:56:33Z) ranked "FP8 QSA scale calibration / alternative numerically valid KV compression" as **priority 2** for future GPU effort, ahead of P5, the RecoverSSM graph and deeper MTP ("Later").
  - Update at 19:00:52Z: the scale file is identical to NB0's and covers all 12 QSA layers (each stage takes its own 6). The "6 of 12" lead was a misreading. The open item is only the MTP layer's KV scale, which defaults to 1.0.
- **M19. K3 + `DRAFT_HEAD=int8` is untested.**
  - NB2 FULL, which had boot A = K3 + INT8, was dropped. SEED2 shows that INT8 does not lower acceptance at positions 1-2 (F25); the −7 % came from K2.
  - Merlin's CHANGELOG claims −6.5-7.2 % ms/token at the same acceptance (3090). The LM head is 2.5 ms per call (F21), and A7 estimates the vocab heads at about 10 ms per tick.
  - This is a config-only lever. Check the log line "INT8 weight-only copy" (MAIN 09:00:48Z).
  - It is not in any planned notebook now.
- **M20. D8, Merlin runtime switches** (PLAN §3 D8, A3):
  - `VLLM_TARGET_SEQ_LENS_ASYNC`, which looks like a host-sync lever;
  - `VLLM_E2_GUARD_MODE=off` (−0.3 ms/step; the "guard reader 0.3 ms" term);
  - `VLLM_GDN_DECODE_FASTPATH`;
  - `VLLM_GDN_SHARED_ALIGN_INDEX`;
  - `use_local_argmax_reduction` + greedy draft (+10-20 % on 8×3090, 0 on 4×3090).
  - Each needs a bitwise or greedy check.
  - These host and sync switches bear directly on the "host launch gap" and "scheduler wait" slices. T0 should at least record which of them the profile already sets.
- **M21. D6, PP2 balance and per-shard sampling.**
  - Retune 25/23 via `VLLM_PP_LAYER_PARTITION`, or sample per shard so the full vocab is not gathered (logits gather 2-3 ms, est.).
  - The assistant listed "stage balance" as a candidate fix at 18:56:27Z.
  - PLAN: no 25/23 change until the NB1 timeline exists (it now does). GPT (pasted 12:08:04Z): "Measure per-stage execution, idle and synchronization before changing PP partition".
  - 25/23 is the capacity optimum, so this is a capacity trade-off (A3).
- **M22. D11:** check whether Merlin enables the MoE aux-stream and `expandable_segments` (jon-nielsen kit: −20 % / +13-37 %).
- **M23. Humming T16 tile tuning** was listed for NB4 at 16:26:49Z. It is irrelevant on the W4 base and belongs to the klee100 track.

### 3.6 Measurement hygiene

- **M24. Profiler flags.** NB2 ran `--profiler-config` and `VLLM_SERVER_DEV_MODE=1` on every boot except CTRL, and "whether they affect speed is unproven" (MAIN 09:00:48Z).
  - The spec's T0b has no profiler. If G, C and P carry the profiler flags for their short profiled window, their speed windows differ from T0b in flags.
  - Either keep the flag set identical across all arms and T0b, with the profiler inactive in the speed windows, or measure the flag effect.
- **M25. `COUNTERS=1` disables the fused QSA writer** (never-do list). No Merlin counter mode may be used for timing.
- **M26. `--no-async-scheduling` with PP deadlocks** (never-do). Async off is not an available diagnostic arm, so the scheduler share must come from timestamps.
- **M27. Real xhigh ARC prompts only.**
  - Synthetic prompts inflate acceptance (3.57 vs 2.69).
  - Merlin's default `reasoning_effort` is medium (R36).
  - Acceptance decays over an answer (F26), so every arm's windows must sit at the same answer position. The RC2 `prompt(band, 5)` was the shared basis for NB0, NB2 and RC2 (MAIN 09:00:48Z).
  - Measure after prefill has settled, using prefill counters (A1 rule 10).
- **M28. Four rates per window** (PLAN v3.5): tick, cycle (= 2 ticks), completed-micro-batch rate and wall clock. The spec lists only tok/s and tick ms. The metric name for per-position acceptance is `vllm:spec_decode_num_accepted_tokens_per_pos_total`.
- **M29. KV and pool gates.**
  - Use running blocks, not `kv_cache_usage_perc` (F32).
  - The worker pool and the gauge pool must match, or capacity is not certified. GPT (pasted 14:29:00Z) found an earlier mismatch: gauge 149 vs worker 220/219.
  - The admission guard stays below 0.95 (F31).
- **M30. Env after `build()`.**
  - Merlin's `launch.build()` wipes the `MERLIN_*` env, and `MERLIN_MARLIN_MOE_CFG` must be re-exported after `build()` (MAIN 06:25:39Z, 06:30:18Z).
  - Every NB4 env change must be applied after `build()` and read back from the worker. The spec's read-back requirement covers this, but the trap should be named.
- **M31. Free memory comes from the smaller of two sources.**
  - nvidia-smi showed 734-990 MiB more free memory than the worker's own reading in NB1 R2 (MAIN 18:25:42Z).
  - Any capture-list growth in arms G and C uses graph memory (0.33-0.52 GiB per rank), and the tightest card had only 191 MiB free at pin +1.0 GiB.
  - **Risk:** larger capture lists on the 214-block pin can OOM at boot. A fallback pin ladder is needed (base pin as the last step).
- **M32. Budget realism.**
  - MTP boots take 423-777 s to become ready (F29), and a page-cache-pressure outlier reached 873 s per worker. With 5 boots in 150 min the margin is thin.
  - Each step should check the remaining time before it starts and end as `SKIPPED_BUDGET`, as NB0-NB3 do (MAIN 11:31:22Z).
- **M33. Same-session A/A spread.** E23's ±0.6-1 % floor came from long repeats. One T0b repeat gives a single difference, not a spread.
  - P5's expected gain is about 2 % (F22), and the 27B FULL gain was +2 %. Both sit at the spec's "> 2 % absolute" threshold, so NO_GAIN is likely by construction unless windows are longer or repeated.

### 3.7 Structural items raised (not NB4 scope, but the T0 timeline can feed them)

- **M34. The PP2 split-batch re-reads experts:** 2·D(2C) vs D(4C) per cycle, an estimated +31 % at C 16. The L2 cache (48 MB) cannot hold the touched experts (about 300 MB per layer per card). A one-batch layout is a design study (MAIN 08:22:33Z, 08:32:40Z).
- **M35. Two-micro-batch "separate vs combined" MoE test (NB3 add-on).** Its result was not reported in the transcripts.
- **M36. Interpreting the 24 ms intercept.** GPT (pasted 12:08:04Z): "24ms fitted overhead-এর সবটাই kernel code দিয়ে কমানো যাবে—এমন ধরে source patch শুরু করা উচিত নয়। এর মধ্যে pipeline idle, synchronization ও overlap-যোগ্য latency থাকতে পারে।" Also: "যোগফলকে wall-clock critical path ধরে double-count করবেন না।"
  - Gloss: do not start a source patch assuming the whole fitted 24 ms can be removed by kernel code; it may contain pipeline idle, synchronisation and latency that overlap can hide. Do not double-count the sum as the wall-clock critical path.
  - NB3's saved-ms numbers are not additive under PP2 overlap; `engine_confirmed_saved_ms` is filled only by NB4.

---

## 4. Contradictions between the spec and the conversations

- **C1. Cumulative bundles vs independent arms.**
  - PLAN §4.1 rule 3: "Patch winners go to NB4, one per bundle (NB4-A = the top patch alone, NB4-B = the second, on top of A; rollback on any regression)". GPT (pasted 06:39:47Z) said the same: "NB4-তে মাত্র দুটি independently testable bundles রাখো".
  - Spec: every arm = base + exactly one change, with three patch arms.
  - The user's 18:58Z request says "আলাদা আলাদা A/B" (separate A/Bs), which matches the spec.
  - Under the user-intent rule the spec stands. **PLAN §4.1 should be updated** so the two do not disagree.
- **C2. The ≥ 5 % selection rule vs the expected gains.**
  - PLAN §4.1 rule 3 picks only levers with a projected gain ≥ 5 % of the run average each.
  - The conversations' own estimates are P5 ≈ 0.9 ms/tick ≈ 2 % (F22) and FULL +2 % (27B).
  - The commit graph has no estimate. Only the 23 ms gap itself (about 45 % of the tick) clears the bar.
  - The user's explicit request overrides rule 3 for NB4. The report should say that P and G were run on the user's instruction, as diagnostics.
- **C3. P5 adoption evidence.**
  - PLAN §4.1 rule 7: a P5 change "is adopted only after a full-engine A/B (2 repeats) plus the 40k / 90k correctness rows; P5 additionally needs 1000+ graph replays and both-pair contention evidence before integration", and `PASS_BOUND_ONLY` rows score 0.
  - GPT (pasted 12:08:04Z): "Do not select P5 for NB4 while its benchmark status is BENCH_ERROR".
    - That error was later explained as a port collision. NB3's D7 stays `PASS_BOUND_ONLY`.
  - Spec: one boot, TF 2k/8k/40k with no 90k row, no repeat, no replay count, no contention evidence.
  - Consequence: under PLAN rules, arm P can at most be `DIAGNOSTIC` / "WIN pending adoption evidence", not an adoptable WIN.
- **C4. Paired ARC accuracy.**
  - PLAN §0 and GPT (pasted 12:08:04Z, 14:29:00Z: "DIAGNOSTIC_PASS production PASS নয়। Paired ARC accuracy ও numerical gate পাসের পরে promotion হবে") require a paired ARC accuracy run before any speed winner is adopted.
  - The spec's `WIN` label has no such step. It should read "WIN (speed); adoption needs paired accuracy".
- **C5. The K4 arm.** PLAN v3.2 §6.3 and the assistant (16:47:34Z) put one K4 arm in NB4; the spec has none (M17).
- **C6. The scale-calibration arm.** PLAN §6.8 and the assistant (16:47:34Z) put a QSA FP8 scale-calibration arm in NB4; the spec has none (M18). The motivating "6 of 12 layers" lead was withdrawn at 19:00:52Z, leaving only the MTP layer's scale.
- **C7. The original NB4 scope.**
  - The original NB4 (MAIN 06:33:40Z, PLAN §4 S3) covered:
    - D4 GEMM plans: dropped legitimately (`UNSUPPORTED_SM89`);
    - D5 MoE tuning: dropped legitimately (0 %);
    - LL128: dropped legitimately (never-do; corruption risk per NVIDIA docs);
    - D6 PP balance / sharded sampling: **silently absent**;
    - D8 Merlin switches: **silently absent**.
  - D6 and D8 are the levers closest to a host and sync gap (M20, M21).
- **C8. Torch profiler vs nsys.**
  - The assistant said nsys is needed to separate wait from transfer (16:34:22Z). NB1's builder said nsys/ncu cannot attach to the running server (09:55:49Z).
  - The spec chooses the torch profiler, consistent with the constraint, but its "NCCL wait" category then depends on cross-rank alignment that the spec does not specify (M1, M6).
- **C9. Base streams vs window levels.** The base has seqs 9 (NB5) and CAPS16, but the windows run at 8 (8k@8, 40k@8).
  - The NB1 R2 evidence came from B1 (seqs 8, 149 blocks), and B1 = B3 in tick (F3), so the baseline is comparable.
  - The arms should still log the micro-batch sizes and captures actually dispatched (M11).
- **C10. Which stage is "light".**
  - The spec says "the light stage is idle ≈ 27 ms per tick". That is stage 1 (`light_stage_idle_ms_per_tick_est` = 27.35 = 50.645 − 23.295).
  - The conversations' earlier model said the light stage is stage 0 and stage 1 carries the heads (F15, F16).
  - The NB1 data says stage 0 is both heavy and binding, yet it has the 23 ms gap. This contradicts the old stage-imbalance story and points to M1 (serial stages / return-path wait).
- **C11. Order of priorities.**
  - GPT (pasted 17:56:33Z) ranked the next GPU effort as:
    1. klee100 load;
    2. FP8 QSA scale calibration;
    3. humming W2-G64 tuning, then LM-head/PP critical-path attribution;
    4. later: P5, the RecoverSSM graph, deeper MTP.
  - The spec makes P5 and the commit graph the main arms now.
  - The user's 18:58Z instruction overrides this. T0 (the attribution) matches GPT's priority 3.
- **C12. Precision base.**
  - The spec: "BF16 is not used: NB5 FP8 solved 13/15 tasks".
  - At spec time NB5's BF16 phase was still running. The assistant labelled the comparison `precision_comparison: UNVERIFIED` (9 vs 5 streams, different pools; MAIN 16:16:27Z). NB0's TF verdict on FP8 was NO_GO.
  - This does not affect the speed A/Bs, which compare FP8 with FP8, but the spec's sentence overstates the evidence.
- **C13. NB5 KV peak vs the 0.95 rule.** NB5 FP8 (the NB4 base) peaked at 99.1 % KV with 0 preemptions (F30), above the standing 0.95 admission gate (R41, F31). NB4 windows at 40k@8 on 214 blocks stay far below it, but the base itself does not follow the rule in production.
- **C14. NCCL CTAs (inside the conversations, not the spec).** A7 says CTA 4/8/16/32 is slower and puts CTA tuning on the never-do list. A5 says `NCCL_MIN_CTAS`/`MAX_CTAS` 4-8 is untested and SHM uses only 2 channels (MAIN 06:16:21Z vs 06:21:16Z). NB4 should not add CTA arms without settling this.
