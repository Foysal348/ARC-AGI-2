## Where speed is lost: Flash-Next on Merlin v3.0.0 (TP2×PP2+EP, MTP K3, FP8 QSA KV), measured on 4×L4

This is read-only work: no project files were edited, nothing was pushed and git was not run. My analysis scripts are in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\`:
- `nb5_timeline.py`
- `nb5_branches.py`
- `nb5_prefill.py`
- `nb5_spec.py`
- intermediate files: `stats_*.json`, `tl_*.json`

**Short answer: yes, a lot is still being missed.** In order of gain:
1. The PP stage-0 stall.
2. Too few streams for the KV pool we actually use.
3. The previous turn's generated tokens are prefilled again on every turn.
4. Boot time.
5. FP8 lowers MTP acceptance.
6. PP0/PP1 memory imbalance.

The window loss in NB5 (225 → 172 tok/s) is almost entirely occupancy, not engine speed. Most of that occupancy loss comes from running only 15 tasks.

---

### a. NB5 FP8 duty factor (9 workers, 4,050 s window)

Sources:
- `phase_Q_FP8_K3/duck/*/nb5_requests.jsonl` and `branch_result.json`
- the server log stats lines (`server_merlin_fp8_mtp_pin100.log`)
- `e12_run_report.json` → `rev6_scheduler`

**Slot-seconds: 4,050 s × 9 = 36,450 slot-s, built from per-request t0/t1 at 1-s resolution**

| bucket | slot-s | share |
|---|---|---|
| request in flight, decoding (24,867 in flight − 273 req × 1.72 s mean TTFT) | 24,397 | 66.9 % |
| request in flight, waiting for prefill (TTFT) | ~470 | 1.3 % |
| host/tool gaps between turns (243 gaps: mean 0.45 s, p50 0.31, p90 0.88, max 3.2) | 109 (+2 s head gaps) | 0.3 % |
| KV admission gate (`kv_hold_sec`, 17 holds) | 35.5 | 0.1 % |
| branch handoffs | ~30 | 0.1 % |
| **idle while work is pending: a b1 attempt waits for its own b0 to finish** (`b1_deferred` 14; task heap empty at 1,263 s; live branches fall 9 → 3 from 1,594 s; every b1 starts 1–2 s after its b0 ends) | ~9,420 | **25.8 %** |
| idle with nothing left to start (after 3,720 s) | 1,980 | 5.4 |

**Tokens (from the stats lines)**
- Ideal: 9 running with no prefill. The regression intercept is 239.4 tok/s, and no-prompt lines at 9 running have p50 244. That gives 969.6k tokens.
- Prefill interference while 9 are running: mean 220.2 vs 239.4 → **−8.0 %** (77.8k tokens).
- Running < 9: 892k → 697k actual → **−195k tokens**, which is 20 % of ideal and the whole 225 → 172 gap.
- Server running histogram: 9 for 140 lines, 3 for 133 lines. Gen p50 at 3 running = 118 tok/s.

**BF16 phase for comparison (5 workers):** in flight 94.8 %, host gap 0.5 %, idle-pending only 92 slot-s, nothing-left 860 slot-s (4.2 %).
- The b1-after-b0 idle only bites when the queue runs dry.
- On 240 hidden tasks it becomes an end-of-run tail effect. Its size there is not measured.

**Prefill between turns**
- Per request: prompt mean 24.6k tokens (6,725,089 / 273), cached 83.4 %.
- Uncached per request: mean 4,084, p50 2,570, max 89,830. Total uncached 1,115,041.
- Measured prefill rate at the 1024 budget: 3,269–4,032 tok/s uncached (klee100 sweep), 4,487 (NB1 40k).
- That is 250–340 s of engine time, 6–8 % of the window. This matches the regression slope of −5.73 tok/s per 1k uncached tokens in 10 s, at 9 running.
- Big re-prefills stall the other requests' decode. A 89,830-token re-prefill cut decode at 3 running to 10.5–11.8 tok/s for ~20 s. Four such stats lines, at 3,169 / 3,179 / 3,379 / 3,619 s.

### b. MTP K3 acceptance

Source: SpecDecoding lines; context is the time-interpolated mean of in-flight requests.

| | AL | pos1 | pos2 | pos3 | conditional p2/p1, p3/p2 |
|---|---|---|---|---|---|
| NB5 FP8 (Duck) | 2.833 | 0.795 | 0.599 | 0.439 | 0.753, 0.734 |
| NB5 BF16 (Duck) | 2.957 | 0.804 | 0.642 | 0.511 | 0.798, 0.797 |
| klee100 sweep (synthetic 20k) | 3.21 | 0.864 | 0.731 | 0.607 | |

- **By context, FP8:** AL 2.94 (0–10k), 2.85 (10–20k), 2.83 (20–30k), 2.81 (30–40k), 2.78 (40–50k), 2.90 (50–60k), 2.88 (60–80k), 2.77 (80–140k). There is no trend, so context-adaptive K has no case.
- **Synthetic sweeps overstate Duck tok/s by ~12 %** through acceptance alone.

**Choice of K**
- Tokens per verify step:
  - K2: 2.394 (−15.5 %)
  - K3: 2.833
  - K4: ~3.155, extrapolating p4 = 0.439 × 0.733 (+11.4 %)
- Step-time model from NB5: tick 35.4 ms at 3 streams, 45.9 at 6, 52.2 at 9.
  - Computed as gen / AL / streams.
  - One extra stream costs ~2.1 ms per tick at 9 streams.
  - The extra MTP pass is ~3.5 ms (NB1 profile: drafts 6.8 + 3.75 ms per tick).
- Estimates: K4 −1 % to +2 %; K2 −7 % to +3 %. The two cost models disagree in sign for K2.
- **Verdict:** keep K3. Gain is ≤ ±3 % and not resolvable from this data.
- K4 also needs capture sizes in multiples of 5. Revisit only after the PP stall is fixed.

### c. Boot time

**NB6, from the notebook log and server log (fresh boot)**
- Notebook start → install: 69 s.
- A0 install: 220 s. Extract 44 + install 175 run in parallel with sha256 219 s.
  - Inside install: "Prepared 222 packages 53.8 s", vllm build 17 s, resolve 17.5 s.
- Server, launch → ready: 812 s:
  - pre-load (arch resolve ×2 ~29 s, engine spawn, worker init): 74 s
  - **weight load: 421.4 s** (PP1 448.6 s)
  - PLE worker ready: 43 s
  - torch.compile: 29 s
  - KV + QSA warm-up: 26 s
  - **FlashInfer "Building JIT module sampling": 108 s** (21:05:00 → 21:06:48)
  - RecoverSSM + graph capture: 23 s
  - API up: 13 s
- Post-ready probes + warm-up (74.9 s) + census: 119 s.
- **First Duck request at 1,210 s = 2.8 % of a 12-h run (43,200 s).**

**NB5 and klee100**
- NB5 FP8: phase start 1,019 s. The weight load was 299.3 s.
- klee100: ready_sec 519 s.
  - sha 178.7 s (101.8 GB at 0.57 GB/s), install 178.4 s.
  - Weights 166.6 s; the FlashInfer JIT again took 107 s.
  - Its worker-extension injection serialised the 4 workers at ~9 s each (27 s).

**Across all 13 boots in the logs**
- First boots: init engine 186–206 s. Reboots in the same session: 36–55 s. There is no FlashInfer JIT line on reboot, so ~140 s per fresh boot is JIT and warm-up work that a cache would remove.
- First-boot W4 weight loads: 233, 258, 299, 315, 320, 345, 407, 416, 421 s.
  - The loader reads at 0.27–0.41 GiB/s, while the sha256 pass reads the same files at 0.56–0.75 GB/s.
  - The host gate drops 123.8 GB of model page cache between the two.
  - Log: "NFS … Skipping auto-prefetch".

**What can be saved**
- FlashInfer JIT cache in the payload: −108 s.
- Triton / compile caches: ~−30 s.
- Sequential prefetch or `fastsafetensors` (already in the venv): −100 to 200 s.
- Prebuilt venv: ~−100 s.
- Total about −340 to 440 s → **+0.8 to 1.0 % of 12-h tokens.** Cost and risk are low.

### d. klee100 sweep

Source: `k100_results.json` → `sections.K3.k100_K3_fp8_mtp3_pin739_r0`.

**Tok/s by level**

| window | tok/s | tick | AL | prefill uncached |
|---|---|---|---|---|
| 20k@16 | 460.4 | 55.8 ms | 3.214 | 3,536 tok/s |
| 20k@24 | 562.0 | 68.7 ms | 3.215 | 3,269 tok/s |
| 40k@16 | 444.7 | 57.2 ms | 3.179 | |
| 60k@16 admission | 431.1 | 57.8 ms | | ALL_RESIDENT, kv 0.70 |

- Per stream: 28.8 tok/s at 16 streams, 23.4 at 24.
- Tick grows 12.9 ms per 8 streams (0.8 ms per verify token).

**What limits the higher levels: only the KV fit rule, not speed**
- The rule is need = n × (ceil(ctx/4096) + F=6) against 0.95 × 479 = 455 blocks.
  - 20k@32: 512 blocks needed
  - 40k@24: 552
  - 60k@16: 480
  - 100k@16: 720
- The rule uses the end-of-window context (37,035 / 67,321 / 97,607). The real admission at 60k@16 was fully resident.

**Free memory after ready**
- nvidia-smi: 2,795 / 2,791 / 1,189 / 1,189 MiB.
- Worker view: 2,325 / 2,321 / 719 / 719 MiB.
- Peaks during the sweep were the same (+4 MiB), so PP1 does not grow under load.

**Higher pin:** about +500 MiB on PP1 → +30 blocks (+6 %). That alone does not make 32@20k fit.

**Fewer captures**
- Captured: 13 PIECEWISE + 9 FULL. Graph memory 0.45 GiB on PP0, **0.60 GiB on PP1**.
- Trimming PIECEWISE sizes or using FULL_DECODE_ONLY could return about 0.2–0.3 GiB → +12 to 18 blocks (+3 to 4 %). This is an estimate; the split between modes is not logged.

**Smaller `max_num_seqs`:** no lever. It changes graph memory by ~0.17 GiB at most.

**What 32 streams need**
- At 20k: +60 blocks ≈ +1.0 GB per PP1 card. Options:
  - pin + graph trim + F reduction
  - PP re-partition (see item 6)
- At the measured Duck mix (14.6 blocks per stream): 32 streams need 468 blocks (492 at 0.95), so they roughly fit 479.
- Projection: tick ~81.6 ms → ~630 tok/s at synthetic AL, ~555 at Duck AL. That is +12 % over 24 streams, with diminishing returns.

### e. Per-request generation rate vs running

Source: FP8 stats lines.

**Per-request rate by running count**

| running | per-request tok/s | aggregate p50 (no-prompt lines) |
|---|---|---|
| 3 | 39.5 | 120 |
| 4 | 36.8 | |
| 5 | 33.8 | |
| 6 | 30.7 | 185 |
| 7 | 28.3 | |
| 8 | 26.8 | |
| 9 | 25.0 | 244 |

- BF16: 45.1 at 3 running → 33.8 at 5.

**Health of the scheduler**
- Preemptions: 0 (counter and lines).
- Waiting: max 1, in one line.
- Variation at 9 running: CV 0.063.
- No scheduler stall. The only slow lines are the four big re-prefills listed in (a).

### f. Other things in the logs

- **Duplicate CLI flags:** "Found duplicate keys --port, --max-model-len, --compilation-config, --max-num-seqs, --default-chat-template-kwargs, --served-model-name". The last value wins and the readback confirms 131072 / 9 / xhigh / 13 sizes. No speed effect; it is a hazard only.
- **Capture sizes vs batches:** with seqs 9 only ≤ 36 tokens are kept: PIECEWISE 10 sizes `[1..32]`, FULL 6 sizes `[4,8,12,16,24,32]`.
  - PP2 splits 9 requests into micro-batches of 4 and 5 (16 and 20 tokens) → 20 pads to 24. This is an inference; the split is not logged.
  - A 9-request micro-batch (36 tokens) would have no graph.
  - Adding sizes 20 and 36 is worth an estimated ≤ 2 %.
  - At klee100 levels 16 / 24 / 32 the micro-batches are 32 / 48 / 64 tokens, which match exactly.
- **Recompiles and eager runs:**
  - 6 Triton JIT kernels compiled during inference, all in warm-up probes and none in the solve windows: `_fill_num_accepted`, `_causal_conv1d_update`, `_gdn_recoverssm_verify`, `_ple_conv`, `_ple_conv_writeback`, `_post_update`.
  - No other runtime warnings in either phase.
  - Mixed batches over 64 tokens (every 1024-token prefill chunk) run without a CUDA graph (compile range 1..1024).
- **Other warnings:**
  - `max_num_scheduled_tokens` 1024 warning: costs 8 % at 9 streams (see a).
  - "num_speculative_tokens > 1 … lower acceptance": confirmed by the conditional acceptance decay above.
  - The MTP layer's FP8 KV uses "k_scale=1 v_scale=1 (DEFAULT 1.0 -- no static calibration loaded)".
  - Cgroup preflight shows only 0–8 MiB available at boot. This is an OOM risk, not a speed loss.
  - `MERLIN_SHARE_DRAFT_WORKSPACE`: "0 B" — the setting has no effect.
  - "Padding mamba page size by 21.01 %" (min block 3,385).
  - NCCL_PROTO is not set in the NB5 env.
- **Config differences:** NB5/NB6 run without `--language-model-only` (MM warm-up 23.8 s; NB1 measured 430–451 MiB freed on PP0), although klee100 uses it.

---

## Findings ranked by gain

**1. PP stage-0 in-graph stall**
- **Measured:** NB1 R2 B1 trace (`E25_GPU_NB1_R2_2026-10-31.md`, re-analysed with `nb4_timeline.py`).
  - Tick is 49.66 ms. Stage 0 waits 20.96 ms on NCCL per tick, in 36 of 36 ticks.
  - The wait sits after kernel 172, until the PP `Broadcast Int x8` of the sampled tokens.
  - Each micro-batch steps every ~99 ms against a ~50 ms chain. The ideal overlapped tick is 34.6 ms.
- **Gain:** up to −30 % tick, **+43 % tok/s**.
- **Cost / risk:** engine change in Merlin `_pp_broadcast_prev_sampled_token_ids` / `_pp_receive_…` (around `gpu_model_runner.py:4577-4600`). The first cheap test is `CUDA_DEVICE_MAX_CONNECTIONS`. Medium-high risk; the cause (a false dependency) is unverified.

**2. Too few streams for the pool**
- **Measured:** workers = 9 comes from a rule that assumes 22 blocks per stream: floor(214 / (16 + 6)).
  - Measured Duck mix, time-weighted: context mean 33.2k, p90 78.8k → **14.6 blocks per stream on average**, 26 at p90.
  - Mean fit: 214 × 0.95 / 14.6 = 13.9 streams.
- **Gain:** 12 streams → projected tick ~57 ms, **~298 tok/s (+22 %)**. 16 streams +40–47 % if the pool allows.
- **Cost / risk:** a policy change. Risk is preemption at p90 peaks.
  - Server KV usage reads 87–95 % even at 3 running, because finished-turn prefixes are retained (up to 16 turns).
  - So the retention policy competes with extra streams. A KV-aware admission gate already exists.

**3. The previous turn's output is not reused as cached prefix**
- **Measured:**
  - 239 of 243 later turns have cached = previous prompt (within 128 tokens), and 0 turns go beyond it.
  - ~650k of the 1,115k uncached tokens (58 %) are previous-turn generated tokens prefilled again. BF16: 574k of 926k.
  - Log: "RecoverSSM + pipeline parallel: Mamba prefix checkpoints are published for prompt positions only".
  - Decomposition of later-turn uncached: truly new 270k, previous context not reused 763k, first requests 82k.
- **Gain:** **+4.6 %** at full occupancy (8 % × 0.58). It also removes the ~20 s decode collapses and cuts TTFT.
- **Cost / risk:** a Merlin scheduler/RecoverSSM change to publish the checkpoint at the end of generation under PP. Medium-high risk. Whether the re-rendered history tokenizes identically to the generated ids is unverified.

**4. FP8 KV lowers MTP acceptance**
- **Measured:** AL 2.833 vs 2.957 (−4.2 %). pos3 0.439 vs 0.511, consistently across context bins. The MTP layer's KV scale is the default 1.0.
- **Gain:** up to **+4.4 %**.
- **Cost / risk:** low. Add the `mtp.layers.48` K/V scales to `qsa_kv_scales_262k.json`, or run the draft layer's KV in BF16. Part of the gap may come from the target model in FP8 (NB0 teacher-forced divergence).

**5. PP memory imbalance and layer partition**
- **Measured:**
  - Worker-free memory, PP0 vs PP1: klee100 2,325 vs 719 MiB; W4 (NB1 B3) 639 vs 191–205 MiB.
  - Block bytes: 14.68 MB on PP0 vs 16.52 MB on PP1. The pool is the minimum over ranks, so PP0 uses only 214 × 14.68 MB = 3.14 GB of its 3.55 GB pin (klee100: 7.03 of 7.93 GB).
  - NB5/NB6 also lack `--language-model-only` (430–451 MiB on PP0).
- **Gain:** re-partition `VLLM_PP_LAYER_PARTITION` 25,23 → 26,22 or 27,21, plus text-only → est **+10–19 % pool on W4** (+8–12 % on klee100), which means more streams (item 2).
  - It also balances compute: stage-1 compute is 34.6 vs 23.5 ms on stage 0, and stage 1 binds once the stall is gone.
- **Cost / risk:** env + flag change, then a capacity census and oracle check. Low-medium risk.

**6. Tail when the queue is empty (b1 waits for its b0)**
- **Measured:** 25.8 % of slot-s idle with work pending and 5.4 % with none left in FP8; −195k tokens in the window.
- **Gain:** on 240 tasks this is only the end tail; the size there is not measured. My rough projection is ~3 % if the tail is ~2,800 s at half occupancy.
- **Cost / risk:** a Duck scheduler policy change: start b1 alongside b0 when nothing else is queued, and start long tasks first. Low risk, but b1's "facts" view may depend on b0 finishing first.

**7. Boot**
- **Measured:** 1,210 s = 2.8 % of 12 h. FlashInfer JIT 108 s, weight load 299–421 s, install 175–178 s.
- **Gain:** **+0.8 to 1.0 %**.
- **Cost / risk:** low. Ship JIT/compile caches and a prebuilt venv; prefetch the weights sequentially.

**8. Prefill chunk 1024**
- **Measured:** prefill costs 8 % of throughput at 9 streams.
- **Gain:** a larger budget is estimated at 1–3 %; not measured.
- **Cost / risk:** activation memory on PP1 (only 191–719 MiB free), so OOM risk on the first long prefill.

**9. Capture sizes and graph memory**
- **Gain:** est ≤ 2 % from the 20 → 24 padding, and +3–4 % pool from trimming graphs.
- **Cost / risk:** config only, low risk. Removing PIECEWISE graphs makes small mixed batches run eager.

**10. MTP K**
- **Gain:** K4 −1 to +2 %, K2 −7 to +3 %.
- **Decision:** keep K3. Test K4 only after the stall fix, with capture sizes in multiples of 5.

**11. Mamba page padding 21 %**
- **Measured:** F = 6 pages per request, which is 25 % of the pool at 9 streams.
- **Gain:** a block size of 3,392 would make those pages ~17 % cheaper → est +4–7 % pool.
- **Cost / risk:** high. The QSA kernels and ring layout may require a 4096 block. Unmeasured.

**12. NCCL_PROTO=LL128**
- **Measured:** absent from the NB5 env. +6.2 % only from the 4×3090 analog in PLAN_1500; it is not measured on our runs.
- **Cost / risk:** an env change, low risk.

**Not losses:** host/tool gaps (0.3 %), the KV gate (0.1 %), preemption (0), queueing (none), runtime recompiles (none) and the duplicate flags (resolved correctly). P2P is reported available by the driver but is deliberately off; PLAN_1500 measured it slower.