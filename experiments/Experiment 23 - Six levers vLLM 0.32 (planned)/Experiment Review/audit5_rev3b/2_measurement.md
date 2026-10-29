# E23 rev 3b — measurement audit (agent 2: steady / oracle / teacher-forced / profiler / statistics)

Scope: `src/e23_lab.py` (rev 3) and its use in `Experiment Notebooks/Build/build_e23_notebook.py` (rev 3b).
I did not edit any project file.

**Sources used**
- vLLM v0.32.0 from GitHub (raw files): `v1/metrics/loggers.py`, `v1/engine/async_llm.py`, `v1/spec_decode/metrics.py`, `config/scheduler.py`, `v1/core/kv_cache_manager.py`, `v1/core/sched/scheduler.py`, `v1/worker/gpu_model_runner.py`, `v1/worker/gpu_worker.py`, `profiler/wrapper.py`, `config/profiler.py`, `entrypoints/serve/profile/api_router.py`, `entrypoints/openai/completion/{protocol,serving}.py`, `distributed/utils.py`, `v1/utils.py`.
- The local `_scratch/vllm_whl/*.whl` is truncated (exactly 256 MiB, not a zip). The Merlin tree in `_scratch/e23/p2/merlin-src` is a 0.32 fork, so I used it only for cross-checks.
- Real data:
  - the E14A real torch trace: `Experiment 14/.../E14A SLIM R2 run1/bundle/e14/profile_1792778154/*-TP-0.trace.json.gz` (SGLang, same Kineto format);
  - 295 real Qwen3.8-27B ARC thinking traces (E10 `poetiq_thinking`);
  - the E8 R5 post-mortem (prompt_logprobs OOM);
  - the E21 KV-pool notes;
  - the Merlin profile JSON.
- Simulation and repro scripts, all in the session scratchpad: `tsim.py` (trace verdict), `gsim.py` and `gsim2.py` (garbage detector), `adm.py` (admission timeline), `budget.py` (FULL step time).

---

## BLOCKER 1 — the graph verdict counts GPU-side annotation copies, so FULL and PIECEWISE both come out as EAGER or MIXED

**Where:** `e23_lab.py:743-749` (with the verdict at `:764-768`).

**What the code does:**
- `fw` = every `ph == "X"` event whose name contains both "forward" and "gpu_model_runner".
- Kineto writes every `record_function` range twice:
  - `cat: user_annotation` on the CPU thread;
  - `cat: gpu_user_annotation` with `pid 0, tid = <CUDA stream>`, usually several per CPU range.
- In the real E14 trace there is one `step[VERIFY bs=12]` range. It has 20 `user_annotation` events and 2,520 `gpu_user_annotation` events.
- The GPU copies have no `cuda_runtime` events on their tid, so each one gives `[0, 0]`. `fw[:400]` is then mostly GPU copies, and the median is 0.
- So `fw_g = 0` gives **"EAGER"**, or 0.5 gives **"MIXED"**. This happens for a FULL-graph run and for a PIECEWISE run alike.

**Evidence (reproduced):**
- I took the real E14 trace and renamed `step[VERIFY …]` to `gpu_model_runner: forward`. This range is a captured graph: 1 `cudaGraphLaunch` per step, 0 kernel launches.
- I saved it as `dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json.gz` and ran `L.trace_summary`.

| Filter | forwards | median graph launches / kernel launches per forward | verdict |
|---|---|---|---|
| as written (CPU + GPU copies) | 400 | 0.0 / 0.0 | `EAGER` |
| GPU copies removed | 20 | 1.0 / 0.0 | `FULL` (correct) |

**Consequence:**
- Section 3's deliverable (G0 PIECEWISE vs G1 / G2 / G3 FULL, "with a graph verdict") would report EAGER for every arm.
- That number would be wrong and would look plausible.

**Fix:**
```python
fw = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation"
             and e.get("name") == "gpu_model_runner: forward"), key=lambda e: e.get("ts", 0))
```
- Also keep the rank-0 trace: copy the gz to `OUT/` before `shutil.rmtree(pdir)` (build `:646`, `:699`). This verdict code has never run on a real 0.32 trace, because the kit's fake server writes none (`NO_TRACE_FILE` in kit v1).

---

## BLOCKER 2 — `teacher_forced()` at 40k / 90k will OOM the engine and kill the reference server before its steady levels run

**Where:**
- `e23_lab.py:484-504`;
- called at build `:638-639` (`s6(0)`, before `levels`) and at `:722-723` (`s5(0)`, before `levels`).

**Mechanism (0.32 `gpu_model_runner.py:5306-5413`):**
- Prompt logprobs are computed per scheduled prefill chunk.
- In `compute_logits` the bf16 logits of the chunk are all-gathered to every TP rank:
  - with `--max-num-batched-tokens 4096` and vocab 248,320 (local `p1/hf_small/config.json`), that is 4096 × 248,320 × 2 B = **2.03 GB**.
- `compute_logprobs` (`log_softmax`, fp32) adds **4.07 GB**. The rank mask in `gather_logprobs` adds about 1 GB.
- So the peak is about 7 GB per GPU, transient.
- The memory profile does not account for it: `profile_run` / `_dummy_sampler_run` only cover `max_num_seqs` logit rows.
- With `--gpu-memory-utilization 0.90` the slack on an L4 is at most about 2 GB.

**Our own evidence:**
- E8 R5 (`Experiment 8/Experiment Review/E8_R5_GPU_RUN/CLAUDE_ANALYSIS.md` §2): **one** request with 1,665 prompt tokens and `prompt_logprobs=1`.
- All 4 TP workers raised `torch.OutOfMemoryError` in `LogitsProcessor._gather_logits` ("Tried to allocate 790 MiB … 349–417 MiB free").
- The result was `EngineDeadError`.
- Here every chunk would be 2.5× that size, and there are 10 chunks (40k) and 22 chunks (90k).

**Failure scenario:**
- SMOKE B and FULL `6_mtp_depth/mtp0_target_only`: probes OK, then oracle OK, then `tforced` OOMs and EngineCore dies.
- Every following `steady()` returns `INCONCLUSIVE_NOT_IDLE_AT_START` or errors, and the step ends `SERVER_DIED`.
- The section-6 MTP-OFF reference windows are lost (the oracle rows `fp8_mtp0` survive).
- The same happens again in `5_nvfp4_kv/nvfp4_target_only`: NVFP4 90k / 120k residency and tok/s are lost.

**Fix:** run teacher-forced on its own short launch, with the same weights and KV dtype, and with a small chunk:
- `stock_flags(FP8, 8200, spec=0, kv=<fp8_e4m3|nvfp4>, seqs=2, batched=256, mem=0.85)`;
- 256 × 248,320 × ~7 B ≈ 0.45 GB transient;
- 130k prompt tokens at 256 per step ≈ 1-2 min.

Or at minimum move `tforced(...)` to the **end** of both bodies, so an OOM costs only the TF row. Both reference and candidate must use the same `batched` value: chunking changes the numerics.

---

## HIGH 3 — FULL cannot fit its plan: the steps are prefill-bound far beyond `est_min`, and the A/A repeats never run (arithmetic)

**Where:**
- build `:801-835` (est_min values and the A/A guard `left() < (70+50)*60`);
- build `:487` (per-level guard `left() < WIN + 300`).

**A/A is unreachable even when every estimate holds:**
- The FULL estimates add up to 620 min: A = 30+75+30+35+60; B = 50+35+60+35+35+40+55+45+35.
- At the A/A loop, `left()` = 690 − 620 = 70 min, which is below the 120 needed. So `break` on i = 1.
- The A/A repeats run only if earlier steps crash fast. The FULL report therefore has **no noise floor** for any comparison.

**The estimates are low** (`budget.py` scratch). Model used:
- serial chunked prefill at R tok/s;
- startup 490 s;
- 120-s windows plus settle and teardown;
- KV fit from the E21 pool (845 blocks × 1,600 tok, mamba overhead 6 / 9 / 15 blocks per request for MTP 0 / 1 / 3; NVFP4 ×1.8 tokens per block);
- no prefix-cache reuse between levels. Reuse helps the 40k ascending levels; it barely helps 90k / 120k, because new prompts evict the old ones.

| B-block step (est) | R = 1k | R = 2k | R = 4k |
|---|---|---|---|
| 6 mtp0 (50) | 158 | 98 | 67 |
| 6 mtp3 G0 + profile (60) | 164 | 101 | 70 |
| 5 nvfp4 mtp0 (55) | 254 | 146 | 93 |
| 5 nvfp4 mtp3 (45) | 232 | 135 | 86 |
| B block total (390) | 1,147 | 702 | 480 |

**Consequence:**
- Even at R = 4k, A (230) + B (480) is more than 690 min.
- In run order, `5_nvfp4_kv/nvfp4_mtp3`, `4_sm89_fp8_gemm/int4_mtp3_s20_control` and both PP4 arms end `SKIPPED_BUDGET`, or the hard stop cuts them.
- The per-level guard lets a level start with 7 min left although its admission alone can take 20-40 min (116k@20 at R = 1k: 2,378 s). That level is then killed by the watchdog.

**Fix:**
1. `levels()`: skip unless `left() > n*plen/700 + WIN + 600`.
2. Also in `levels()`: stop a band after its first `CAPACITY_LIMITED` / `INCONCLUSIVE_NO_SETTLE` with `best_running < n`. Record `SKIPPED_KV_BOUND (fits=best_running)` for its larger levels. The census is already in hand, and every larger level only repeats the same prefill.
   - FP8 TP4 is KV-bound at about 90k@12, 90k@20 and 120k@12. NVFP4 is KV-bound at about 90k@24 and 120k@20 (and at 120k@16 with MTP3). 40k@20 with MTP3 is borderline (about 19 fit).
3. Re-derive est_min from the table above.
4. Schedule **one** explicit A/A (the `mtp3_eq_G0` control relaunched) before section 5, instead of the unreachable loop.

---

## MEDIUM 4 — a preemption before the window makes a KV-bound window `INCONCLUSIVE_STALLED_STREAM` within 6 s

**Where:**
- `e23_lab.py:571-578` (settle);
- `e23_lab.py:607-613`, `:637-638` (stall check).

**Mechanism:**
- Stock 0.32 admits a new request only if its **whole** prompt fits (`scheduler_reserve_full_isl = True`, `config/scheduler.py:190`; `kv_cache_manager.py:522`). So KV-bound levels plateau at `run < level`, `wait > 0`. Good.
- The running streams keep growing, though. Initial headroom is uniform between 0 and one request (about 75 blocks at 120k), and 9 streams × 25 tok/s cross a 1,600-token block about every 7 s.
- So the first preemption comes within about 0 to 630 s.
- The victim is `running[-1]`. It goes back to waiting and is not re-admitted: its full ISL is one block short. This gives a new, stable plateau.
- The settle rule accepts that plateau 30 s later, because it checks preemptions only between consecutive samples.
- The victim still has `t_first` set and is alive. In the window, `stalls[i]` counts it, and after 3 samples the window breaks as `INCONCLUSIVE_STALLED_STREAM`.

**Scenario:**
- FP8 TP4 120k@12: 9 running; preemption at t = 200 s of decode; settle; window. The window stops at 6 s.
- About a third of KV-bound windows end this way: preemption before or inside the settle + window span of about 160 s, out of the 0-630 s range.
- The rest end as `INCONCLUSIVE_PREEMPTION` or `CAPACITY_LIMITED`.

**Fix:**
- At window start, record `members = [i for i, (a, b) in enumerate(zip(prev_tok, now_tok)) if b > a]`, the streams gaining in the last settle sample.
- Count stalls only for `members`.
- Store `preemptions_before_window = m0[PREEMPT] - p0[PREEMPT]` in `r`.
- Require the set-constancy checks against `len(members)`.

---

## MEDIUM 5 — `degenerate()` LOW_VARIETY flags real ARC thinking (spaced grids), and `FAIL_GARBAGE_UNDER_LOAD` overrides STEADY

**Where:**
- `e23_lab.py:430-432` (rule);
- `:663-667` (applied to every stream's last 4,000 chars under load).

**Evidence (`gsim.py`):**
- 295 real Qwen3.8-27B ARC thinking traces (E10 R5 EVAL120 + L66).
- The detector run on the 4,000-char tail at cuts from 3k to 20k chars, matching what a stream has after admission plus window.
- **21 / 5,288 windows flagged (0.40 %), all LOW_VARIETY**. The flags are grid transcriptions like `4: 8 8 8 8 2 2 2 2 …`, where every digit is a `\w+` word, so unique words are under 8 %.
- Per steady window, P(at least one flagged stream) = 3.1 % at 8 streams, 6.2 % at 16 and 9.1 % at 24.
- Over the about 60 windows of FULL, expect 2-4 healthy windows turned into `FAIL_GARBAGE_UNDER_LOAD`.
- `_tp1_level` and CELL6 then treat those levels as failed.

**Fix:** exclude digit-only words from the variety test:
```python
words = [w for w in re.findall(r"\w+", t) if not w.isdigit()]
```
- With that change: 0 / 5,288 false positives.
- It still flags `"the the …"`, `"I think I think …"` and `"abc def ghi …"` loops. `!!!!` is still caught by REPEATED_CHAR.

---

## MEDIUM 6 — the TP1 (Merlin, one L4) admission cap assumes 700 tok/s prefill, a TP4 figure

**Where:**
- `e23_lab.py:528`: `max(420, 120 + level*plen/700)`;
- build `:563-569` (`_tp1_body`), `:600-604` (`steady_multi`), `:819` (`profiled_load`).

**Facts:**
- Merlin `27b-tp1-sm89-24g` runs `max_num_batched_tokens = 512` (`tools/merlin/profiles/qwen38-27b.json`; 1024 OOMed on a 4090).
- It runs on one 72 W L4. TP1 prefill is about a quarter of the TP4 1-4k figure: roughly 250-1,000 tok/s.

**`adm.py`:**

| TP1 level | cap | t_adm @ 250 tok/s | t_adm @ 400 tok/s | t_adm @ 700 tok/s |
|---|---|---|---|---|
| 40k@4 | 420 | 648 (fail) | 408 | 237 |
| 90k@2 | 420 | 729 (fail) | 459 (fail) | 266 |
| 116k@1 | 420 | 470 (fail) | 296 | 172 |
| 116k@2 | 451 | 940 (fail) | 592 (fail) | 344 |

**Consequences:**
- `INCONCLUSIVE_NO_SETTLE` after the cap.
- `_tp1_level` then falls back to the defaults (40k → 2, 90k → 1), so the replica runs use too few streams.
- On the TP1 pool (Merlin pin about 225-270k tokens), 116k@2 and 40k@4 are also KV-bound.

**Fix:**
- Make the cap progress-based: extend the deadline by `plen/150` s each time `first` increases, with a hard ceiling of `level*plen/150 + 300`.
- Or pass `prefill_tok_s=200` in every Merlin TP1 call: `levels(..., prefill_tok_s=200)`, `steady_multi(..., prefill_tok_s=200)`, `profiled_load(..., prefill_tok_s=200)`.
- TP4 caps are fine (see verified 1).

---

## MEDIUM 7 — `compare()` turns a candidate *error* into `FAIL_HIGH_MARGIN` (a corruption verdict)

**Where:** `e23_lab.py:455-457`.

**Mechanism:**
- If the candidate row has `error` (`GREEDY_TIMEOUT`, a connection reset, or a dead server) and empty text, then `degenerate("") == "EMPTY"`.
- The oracle is clean, so the status is overwritten to `FAIL_HIGH_MARGIN`.

**Scenario:**
- An NVFP4 server whose 40k greedy row times out (prefill plus 128 tokens within `min(900, left/6)`), or crashes, is declared `NO_GO_CORRUPT` (build `:724-726`), and `nvfp4_mtp3` is skipped as corrupt.

**Fix:**
```python
if g and not b.get("error") and b.get("tokens") and degenerate(a.get("text")) is None:
    st = "FAIL_HIGH_MARGIN"
```

---

## LOW 8 — the margin uses oracle top-1 minus top-2, not top-1 minus the candidate's token

**Where:** `e23_lab.py:366-367`, `:452-454`; greedy `logprobs=2` (`:403`).

**Mechanism:**
- If the candidate picks the oracle's rank-5 token at a position where top-1 minus top-2 is 0.3, the result is `NEAR_TIE`. The real gap may be well over 0.5.
- The test is biased toward passing.

**Fix:**
- Request `logprobs=5` and keep the oracle's per-position `{token_id: lp}`.
- Margin = `lp_top1 − lp(cand token)`; if the candidate token is absent from the top 5, the result is `FAIL_HIGH_MARGIN`.
- Keep 0.5 as the threshold.

## LOW 9 — the `teacher_forced()` fallback silently uses another token's logprob

**Where:** `e23_lab.py:502`, `next(iter(d.values()))`.

- vLLM always includes the actual prompt token, so this branch only hides a key mismatch.
- **Fix:** without the fallback, return `None` for that position.

## LOW 10 — PP4 `step_ms` / `tokens_per_step` are per output batch, not per step

**Where:** `e23_lab.py:623`, `:631-633`.

- With `--pipeline-parallel-size 4`, the iteration histogram counts each micro-batch's output. Steps come out about 4× higher, step_ms about ¼, and tokens_per_step = running / 4.
- tok/s is fine.
- **Fix:** label them `per_output_batch` when pp > 1, or skip step_ms for PP.

## LOW 11 — `_tp1_level` picks the largest *requested* level, also from `CAPACITY_LIMITED` windows

**Where:** build `:584-588`.

- With the fallback profile (seqs 2), 40k@4 gives `CAPACITY_LIMITED` at 2 running (a seqs limit, labelled as KV), and the replicas then run 4 streams for 2 slots.
- **Fix:** use `max(v["running_max"])` over the accepted windows.

## LOW 12 — the FULL verdict needs `fw_k == 0` exactly; the CUPTI fallback counts sampler / drafter kernels

**Where:** `e23_lab.py:767-773`.

- A single small eager kernel inside the forward range (a padding fill) turns FULL into MIXED.
- **Fix:** FULL when `fw_g == 1 and fw_k <= 2`, and report the numbers (already done).
- In the fallback, use attention-category eager share only.

## LOW 13 — category mapping

**Where:** `e23_lab.py:718-719`.

- `cumsum` catches torch's top-p `cumsum` / cub scan kernels, and `chunk_` catches aten `chunk_cat_cuda_kernel`. Both are booked as GDN.
- **Fix:** use `chunk_gated`, `chunk_fwd` and `chunk_scaled_dot`, and drop `cumsum` (fla's cumsum kernels are `chunk_local_cumsum*`, which `chunk_` variants still match).

## LOW 14 — the admission `prefill_tok_s` includes prefix-cache hits

**Where:** `e23_lab.py:583-584`.

- `vllm:prompt_tokens_total` counts the full prompt at first token, cached or not. Ascending levels reuse prompts 0..n_prev−1, so the figure is inflated.
- **Fix:** label it `admitted_prompt_tok_s`, or subtract `vllm:prompt_tokens_cached_total`.

---

## Statistics (focus 4)

- **Window length: adequate.**
  - A 120-s FULL window holds at least 8 × 20 × 120 ≈ 19k emitted tokens, at about 0.25 s counter granularity. Counting error is under 0.5 %.
  - A 30-s SMOKE window is fine for a smoke test.
  - The dominant variance is between launches (graph capture / autotune, clocks) and content-driven MTP acceptance at temperature 1.0. A longer window does not reduce either.
- **A/A: none will run** (HIGH 3). Without it, differences of 3-8 % (G0 vs G3, K1 vs K3, Merlin vs stock) cannot be told from noise.
- **Cheap within-run check:** compute two half-window tok/s values from `mids` (already sampled; currently discarded) and store their ratio as a drift flag.
- **Comparing graph modes:** use `step_ms` at equal `running` together with `mean_acceptance_length`, not tok/s alone. Acceptance differs between arms because the sampled text diverges.
- Optional: sample `clocks.sm` / `power.draw` with `gpu_mem()`. L4 is 72 W-limited.

---

## Verified correct (rev 3b)

1. **TP4 admission cap holds.** `adm.py` models serial 4096-token chunks with interleaved decode. At R = 1k / 2k / 4k, 90k@12 needs 1,107 / 567 / 297 s against a cap of 1,663. 116k@12 / 16 / 20 needs 1,427 / 1,902 / 2,378 s at R = 1k, against caps of 2,109 / 2,771 / 3,434. 40k@24 needs 984 s against 1,491.
2. **`max_new` is never exhausted.**
   - During admission each step carries one prefill chunk, so the early streams decode only about acc × (Σ prompt / 4096) ≈ 850-1,300 tokens.
   - At window end, the first stream has about 3.8k (90k@12) to 4.6k (116k@20) tokens, against `max_new` 40,560 / 14,560.
   - The 120k-band window context is about 117-120.6k, so the band label is right.
3. **Settle cannot fire early on TP4.**
   - A prefilling request sits in `running` and emits nothing, so `gaining >= run` fails until every prefill is done.
   - `vllm:prompt_tokens_total` moves only at a request's first token (`stats.py` `is_prefilling` path), and a re-prefill after preemption is not counted (`scheduler.py:1194`). That is why the guard rests on `gaining` / `first` and not on the counter. The design is right.
4. **Never-settle risk is low on TP4.** Full-ISL admission (`scheduler_reserve_full_isl = True`) rules out chunk-admission thrash. KV-bound plateaus are reached; MEDIUM 4 covers what happens afterwards.
5. **STEADY only at `min(runs) == max(runs) == level`.** RUNNING_SET_CHANGED and UNSTABLE are correct, and `wait_idle` at entry keeps the previous level's streams out.
6. **Metric names in 0.32:**
   - `vllm:generation_tokens`, `prompt_tokens`, `num_preemptions` (+`_total`);
   - `num_requests_running` / `waiting` (`waiting_by_reason` is a separate name, not summed);
   - `kv_cache_usage_perc`;
   - `spec_decode_num_{drafts,draft_tokens,accepted_tokens}`;
   - `iteration_tokens_total` (histogram, so `_count` exists).
   The `prom()` regex and its label summing are correct.
7. **Throughput metric is right.**
   - `generation_tokens` adds `len(new_token_ids)` per output, which is accepted + bonus tokens. That is the user-visible decode rate, the right metric.
   - `mean_acceptance_length = 1 + acc/drafts` matches vLLM's own definition (`spec_decode/metrics.py:114`), and `accept_rate = acc/draft_tokens`.
8. **Steps from the iteration histogram.** One observation is made per `EngineCoreOutputs` message that has outputs (`async_llm.py:850-853`, `loggers.py:1273`), which is one per engine step in steady decode. So `step_ms = dt/steps` and `tokens_per_step = running × accept_len` are sound (PP excepted, LOW 10). The fake runs only exercised the `estimate` path.
9. **Streamed logprobs.**
   - With `return_tokens_as_token_ids`, the keys are `token_id:N`.
   - Greedy with `logprobs=2` gives the dict {sampled = top-1, top-2}: the duplicate is removed, filter `num_output_top_logprobs >= i`.
   - So `vals[0] − vals[1]` is the oracle's top-1/top-2 margin, and margins stay index-aligned with `token_ids`, including multi-token MTP chunks.
10. **First-divergence logic.** The comparison is at the first differing index with identical prefixes, using the oracle's margin at that index; FAIL at 0.5 nat is a sensible threshold for same-weight numeric noise. The `INCONCLUSIVE` handling for missing data is correct.
11. **`prompt_logprobs` layout.** It sits at `choices[0].prompt_logprobs`, a `list[dict[int, Logprob] | None]` with `pl[0] = None`; JSON keys are `str(id)`. Indexing and `rank` are correct (the OOM is BLOCKER 2).
12. **Profiler API in 0.32.**
    - `POST /start_profile` accepts a JSON body `{delay_iterations, max_iterations}` (`StartProfileRequest`, strict ints); `/stop_profile` exists. The router is attached only when `--profiler-config` sets a profiler.
    - `max_iterations` auto-stop goes through `stop()`, so a later stop is ignored safely; the export runs inside the stop RPC.
    - Trace file: `dp0_pp0_tp0_dcp0_ep0_rank0.<ns>.pt.trace.json.gz` (`get_worker_rank_suffix`, `use_gzip` default True). The glob and the `rank0\.` regex both match.
    - `"gpu_model_runner: forward"` is emitted with `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`, and the MTP drafter runs in a separate `"gpu_model_runner: draft"` scope. So the per-forward count isolates target verification, as intended.
13. **Kernel-to-launch attribution.** Attribution through the CUPTI `correlation` id works on a real trace: 29,020 kernels map to `cudaGraphLaunch` (graph id ≠ 0), and 2,781 eager kernels map to `cudaLaunchKernel` / `cuLaunchKernelEx` / `cudaLaunchKernelExC`. All three names match `is_kl`.
14. **The per-forward method itself is sound once fixed.** With BLOCKER 1's fix it gives `[1, 0]`, which is FULL, on the real graph-captured verify. PIECEWISE gives graph launches > 1 per forward, because attention and the GDN ops are splitting ops.
