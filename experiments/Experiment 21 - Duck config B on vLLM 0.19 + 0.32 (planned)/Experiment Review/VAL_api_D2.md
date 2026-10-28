# E21 validation D2: what the run cell and the scheduler read from the vLLM server (besides chat completions)

Reviewer: independent second reviewer (read-only), 2026-10-28.

Scope: the server-side reads of the built notebooks `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb` and `..._NB2_FP8_SSM16_S16.ipynb`. CELL 5 (engine) and CELL 6/7 (cell index 7 and 8) are identical in both notebooks. The items:
- `e21_prom`, `_e12_capacity_probe`, `_e15b_gen_tokens_total`;
- `_E12V_STAT_RE` / `e11_decode_summary` / `_e15b_server_log_summary`;
- `metrics_snapshot` and the heartbeat;
- `E18Server.wait_ready`, `/v1/models`, the watchdog's `/health` path;
- `_e12_r6_preflight`;
- the KV gate in `arc2_duck.solve_all_r6` (`D:/ARC AGI 2/_scratch/donor_map/duck_9e4b/arc2_duck.py`).

It also covers the prefix-cache behaviour of Duck turns on vLLM (E12-V run 2).

## Sources used

- **vLLM 0.19.0:**
  - `vllm/v1/metrics/loggers.py`, `vllm/v1/metrics/stats.py`, `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_manager.py`, `vllm/entrypoints/openai/api_server.py`, `vllm/entrypoints/launcher.py` and `vllm/entrypoints/serve/instrumentator/health.py`: fetched from the v0.19.0 tag. Copies are in the session scratchpad `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/v019/`.
  - Local copies: `D:/ARC AGI 2/_scratch/vllm_src/v0.19.0/` (core_scheduler, single_type_kv_cache_manager, cache, envs, speculative, multiproc_executor) and `D:/ARC AGI 2/_scratch/ext/vllm_0_19_0_src/` (chat serving, engine serving, cli_args, qwen3 reasoning parser).
- **vLLM 0.32.0:**
  - `D:/ARC AGI 2/_scratch/vllm_whl/src032/vllm_v1_metrics_loggers.py`;
  - `D:/ARC AGI 2/_scratch/vllm_src/v0.32.0/` (core_scheduler, cache, speculative, kv_cache_coordinator, mamba_utils).
- **Real logs:**
  - vLLM 0.19: `D:/ARC AGI 2/experiments/Experiment 12 - .../E12-V vLLM SMOKE1 run2 (esfersami50, 2026-10-19)/q27x/vllm_openai_server.log`, plus the notebook log `q27-xhigh-e12-duck-repl-vllm-smoke1.log`;
  - vLLM 0.32: `D:/ARC AGI 2/_scratch/e18_gpu_v1/q27x/vllm_server_s1.log` and `poetiq_vllm_metrics.json` (a real 0.32 `/metrics` capture).
- **Request data:**
  - E12-V run 2 `q27x/duck/2ba387bc_b0|b1/branch_result.json` and `requests.jsonl`;
  - for projection: R3K2 `D:/ARC AGI 2/experiments/Experiment 15 - .../E15B EVAL120 50K45K R3K2 commit1 (hosen42, 2026-10-25, c058d58b)/q27x/duck/*/branch_result.json` (198 branches, 2,303 requests).
- **Model config:** `D:/ARC AGI 2/_scratch/e13_rev6/harness/fixtures/tok_int4/config.json`. It has 64 layers, 16 of them full attention (every 4th), 48 GDN value heads, 16 key heads, head dim 128, `mamba_ssm_dtype` float32, and 1 MTP layer.

---

## Findings

### F1 — HIGH — prefix caching on vLLM never reuses the previous turn's generated tokens

**Location:** server behaviour, vLLM 0.19 and 0.32, mamba cache mode `align`. Both logs say: "Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled". The 0.19 log adds "...support for Mamba layers is experimental". Both logs show "Setting attention block size to 1600 tokens".

This affects every Duck turn, and with it the throughput of both halves of both notebooks.

**Evidence (E12-V run 2, all 10 follow-up requests of both branches):**
- The hit of request n+1 is exactly `floor(prompt_n / 1600) * 1600 - 1600`. It holds 10/10, with no exception.

| branch / req | prompt | completion of previous | cached | rule | recomputed = prev completion + prev-prompt tail + new |
|---|---|---|---|---|---|
| b0 req1 (turn 2) | 19,215 | 12,288 | 4,800 | 4,800 | 14,415 = 12,288 + 2,082 + 45 |
| b0 req2 | 26,342 | 6,069 | 17,600 | 17,600 | 8,742 = 6,069 + 1,615 + 1,058 |
| b0 req3 | 27,725 | 945 | 24,000 | 24,000 | 3,725 = 945 + 2,342 + 438 |
| b0 req4 | 30,138 | 2,253 | 25,600 | 25,600 | 4,538 = 2,253 + 2,125 + 160 |
| b0 req5 | 30,527 | 298 | 27,200 | 27,200 | 3,327 = 298 + 2,938 + 91 |
| b0 req6 | 30,684 | 123 | 28,800 | 28,800 | 1,884 = 123 + 1,727 + 34 |
| b1 req1 | 19,965 | 12,273 | 4,800 | 4,800 | 15,165 = 12,273 + 2,082 + 810 |
| b1 req2 | 20,872 | 831 | 17,600 | 17,600 | 3,272 = 831 + 2,365 + 76 |
| b1 req3 | 21,099 | 142 | 19,200 | 19,200 | 1,899 = 142 + 1,672 + 85 |
| b1 req4 | 21,280 | 151 | 19,200 | 19,200 | 2,080 = 151 + 1,899 + 30 |

Every cached value is a multiple of 1,600. Whole smoke: 261,611 prompt tokens, 192,000 cached (73.4 %), 69,611 recomputed. The server's own prompt-throughput lines add up to about 67.7k computed tokens, which agrees.

**Why turn 2 hit only 4,800 of 19,215 tokens.** It is not the reasoning rendering. The turn-2 prompt is exactly turn-1 prompt + turn-1 completion + 45 tokens (6,882 + 12,288 + 45 = 19,215). So the full reasoning was rendered: `reasoning` key + `preserve_thinking` worked, and the prompt prefix was intact. The cause is three server mechanisms:
1. **No mamba checkpoints for decoded tokens.**
   - In align mode the SSM state is saved only at prefill chunk ends aligned to the block. `_mamba_block_aligned_split` runs only while `num_computed_tokens < max(num_prompt_tokens, num_tokens - 1)`: 0.19 `core_scheduler.py` 298-345; 0.32 says "Past the prompt the manager writes nothing".
   - During decode, `MambaManager.remove_skipped_blocks` frees the older state block every step (0.19 `single_type_kv_cache_manager.py` 763-860).
   - A hybrid hit needs both the attention blocks and a mamba state at the same boundary. So nothing inside the previous turn's 12,288 generated tokens can be a hit.
2. **The MTP (eagle) block drop.** `use_eagle()` is true for `mtp` (0.19 `speculative.py` 857; 0.32 1944). The last cacheable prefill position is `num_tokens - num_tokens % 1600 - 1600`. For the 6,882-token turn-1 prompt that is 4,800.
3. **The partial-block floor.** The last 0-1,599 tokens of the previous prompt.

**Projection onto R3K2's real request pattern** (198 branches, 2,303 requests, median 11 requests per branch, 8.26 M generated, 59.4 M prompt):
- **SGLang:** 93.4 % cached, 3.91 M recomputed.
- **vLLM rule** (first request of each branch given SGLang's hit, so optimistic): 74.7 % cached, 15.04 M recomputed. That is 3.85× SGLang, +11.1 M prefill tokens, about 1.35 per generated token.
- **Breakdown over the 2,105 follow-up requests:**
  - previous completions re-prefilled: 7.76 M;
  - MTP block drop: 3.36 M;
  - partial-block floor: 1.72 M;
  - new tokens, which SGLang also computes: 1.85 M.
- **Prefill rate on 4×L4:**
  - the 0.32 FP8 E18 log peaks at 3,006-3,502 tok/s with near-zero decode;
  - the E12-V 0.19 prefill-heavy 10-s windows show 1,441-1,516 tok/s (mixed with decode).
- **Cost in the window:** the extra 11.1 M tokens cost about 0.9-1.0 h at 3,000-3,500 tok/s, and about 2 h at 1,500 tok/s, of a 5.5-h half window. Prefill chunks share steps with decode, so the real loss is somewhat lower than this exclusive-time estimate.
- **Decode rates:** E18 0.32 FP8 decodes about 376 tok/s at 18 or more running with no prefill. R3K2 SGLang at 16 streams ran 441 tok/s p50.
- **Net:** this predicts noticeably fewer generated tokens per half than R3K2, independent of answer quality.

**Scenario:** both halves (0.19 and 0.32). Each of about 2,000 follow-up turns re-prefills the previous turn's whole output plus 1.6-3.2k tokens. A lower score than R3K2 would then partly be this coverage loss, not "vLLM per-request quality". README Finding 7 records the hit numbers, but it attributes them to "1,600-token blocks". The decode region and the MTP drop dominate.

**Fix / what to do (none is a verified config lever):**
- **Report it.** Per half, record recomputed tokens = sum(prompt - cached) from the per-request `usage` already in the branch records. Record the predicted share of these components, and judge the halves also on the attempt-1-only score that CELL 9 already computes.
- **NB2 (`--mamba-ssm-cache-dtype float16`) should change the block size.** This is a source-based prediction, to be verified from the "Setting attention block size" line:
  - FP32 SSM page per GDN layer per GPU = 12 heads × 128 × 128 × 4 B = 786,432 B;
  - conv = 2,560 × 3 × 4 B = 30,720 B;
  - total 817,152 B. Padded to 1,600 × 512 B = 819,200 B, which matches the logged "Padding mamba page size by 0.25%".
  - With an FP16 SSM the page is about 423,936 B, so the attention block size becomes about 832 tokens.
  - That halves the MTP drop and the floor (about 2.5 M fewer recomputed tokens on the R3K2 pattern) and halves the per-request mamba overhead (F3).
  - So the NB1 vs NB2 difference is not only "SSM precision + streams": the cache granularity differs too. Record the block size per half (see F3).
- **vLLM 0.32 only:** `SpeculativeConfig.disable_eagle_block_drop` ("experimental option for measuring the acceptance-rate impact", 0.32 `speculative.py` 452) would save the 3.36 M MTP-drop share. It is untested, and its effect on MTP acceptance is unknown. It is a measured arm, not a default.
- **What does not help:**
  - 0.32 "partial hash hits" (`kv_cache_coordinator.py` 680-700) can shrink the floor share at best;
  - the decode-region share (7.76 M) is structural in align mode in both versions;
  - 0.19's `mamba_cache_mode "all"` exists in the Literal, but vLLM picked `align` for Qwen3_5 "by default", so it is very likely unsupported for GDN. A GPU test would be needed;
  - turning `preserve_thinking` off is not a fix: it would violate the long-thinking rule and change the prompt mid-history.

### F2 — MEDIUM — the stats-line regex silently drops the lines with preemptions or deferred requests

**Location:** CELL 5 `_E12V_STAT_RE`, used by `e11_decode_summary` (CELL 5) and `_e15b_server_log_summary` (CELL 6). Both feed `b_summary.server_log`, `sglang_decode_summary.json` and the E21 report fields `gen_tok_s_p50`, `gen_tok_s_mean`, `prefix_cache_hit_pct_last`, `kv_usage_pct_max` and `running_hist`.

**Evidence:**
- **vLLM 0.19** `LoggingStatLogger.log` (loggers.py 213-262) builds "Avg prompt throughput: …, Avg generation throughput: …, Running: %d reqs, Waiting: %d reqs" and then `, Preemptions: %d` when the interval had a preemption. Only after that comes "GPU KV cache usage: %.1f%%, Prefix cache hit rate: %.1f%%".
- **vLLM 0.32** (loggers.py 276-338) inserts `, Deferred: %d reqs` (when `num_skipped_waiting_reqs > 0`), `, KV fetch: …` (KV connector only) and `, Preemptions: %d` before "GPU KV cache usage". It prints "Prefix cache hit rate" only when `not prefix_caching_metrics.empty`.
- The regex requires "Waiting: (\d+) reqs, GPU KV cache usage: …, Prefix cache hit rate: …" back to back, so any such line does not match. Tested on synthetic 0.19 and 0.32 lines with Preemptions or Deferred: no match.
- The regex matches 46/46 real 0.19 lines and 164/164 real 0.32 lines. Neither real log had a preemption, so the bug is latent. The harness fake server (`src/harness/fake_vllm_duck_server.py` 342-344) never emits these variants.

**Scenario:** exactly the high-pressure 10-s intervals, where KV is about 100 % and preemptions happen, disappear from the summaries.
- `kv_usage_pct_max` under-reports the peak;
- `gen_tok_s_p50/mean` are biased (recompute intervals removed);
- `running_hist` and `queue_gt0_line_share` lose the saturated lines.

`preempt_lines` still counts them, because it is a substring count. The half comparison on tok/s and KV peak would be wrong in the very case it matters (NB1 with 20 streams; see F3).

**Fix:**
```python
_E12V_STAT_RE = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: \d+ reqs)?(?:, KV fetch: [^%]*?completed waiting)?"
    r"(?:, Preemptions: (\d+))?, GPU KV cache usage: ([\d.]+)%(?:, Prefix cache hit rate: ([\d.]+)%)?")
```
- It matches 46/46 (0.19) and 164/164 (0.32) real lines and all synthetic variants.
- It now has 7 groups (preemptions = index 4, KV = 5, prefix = 6, the optional ones may be None). The two consumers that do `tuple(float(x) for x in m.groups())` and index `r[4]`/`r[5]` must map None and use the new indices.
- `_E12V_STAT_RE` is a verbatim E12-V donor, so `build_e21.py`'s verbatim assertion has to allow this change.

### F3 — MEDIUM — the pre-run capacity check ignores the fixed mamba / MTP blocks of every running request

**Location:** CELL 5 `e21_capacity` (from `e21_engine_new.py` 62-77): `effective = factor × max_len`, `streams = min(seqs, effective // 50,000)`, `fits_p50_peak = effective >= seqs × 57,000`.

**Evidence:**
- **Pool size (E12-V 0.19 FP8, TP4, 0.90 GPU memory, block 1,600):** the real heartbeat gauge `token_usage` (= `vllm:kv_cache_usage_perc`) was 0.05798816568047338, which is exactly 49/845. So the pool has 845 usable blocks. Consistent: "GPU KV cache size: 337,600 tokens" = (846 // 4 groups) × 1,600.
- **Per-request footprint:** at that sample, 2 requests at about 21.9k and 21.3k context hold 26 distinct attention blocks (2 shared prefix blocks) plus 23-24 others. At 07:34:39, 2 requests at 6,882 prompt tokens use 3.8 % = 32 blocks: 8 attention + 24.
- So each running request holds about 12 extra blocks. That is 3 mamba groups × (1 running-state + 3 speculative blocks), as in `MambaManager.allocate_new_blocks` (align mode allocates `1 + num_speculative_blocks` per group). It equals 19,200 token-equivalents per stream at block 1,600.
- vLLM's own factor implies about 15: 845 / 8.46 = 99.9 blocks per request at 135,168 = 85 attention + about 15.
- In 0.32, "GPU KV cache size: 1,110,415 tokens, Maximum concurrency for 106,496 tokens per request: 10.43x" is the same group-aware figure (10.43 × 106,496 = 1,110,753). Same issue.

**Scenario (blocks per request = ceil(L/1600) + 12..15):**
- **FP8 0.19, 845 blocks** (NB1 fallback = NB2 fallback = S1 engine):

  | context | streams that fit |
  |---|---|
  | p50 peak 57k | 16.5-17.6 |
  | p90 65k | 15.1-15.9 |
  | mean 50k | 18-19 |

  The check prints `fits_p50_peak: True` with an "effective 1.143 M ≥ 0.912 M" margin. The truth is that 16 streams sit at the pool edge whenever several attempts are at p90 length at once.
- **NB1 INT4, 20 streams:** using the README's +3.5 GB per GPU (about +267 blocks of 13.1 MB, so about 1,110 blocks; an estimate), 20 streams take 79-85 % at the mean, 86-92 % at p50 and 95-101 % at p90.
- **Effect:** the 0.95 KV gate then holds admissions often, and vLLM preempts (recompute) when the pool is exhausted. Preemptions are invisible in the summaries (F2).
- **Logged numbers:** the printed "capacity" is systematically optimistic, by about 19-24k tokens per stream at block 1,600 (about half that with the FP16 SSM of NB2).

**Fix:**
1. Parse `Setting attention block size to (\d+) tokens` (B; logged by both versions) into the capacity record.
2. Compute `streams_fit(L) = factor × (ceil(max_len/B) + m) / (ceil(L/B) + m)` with `m = 3 × (1 + num_speculative_tokens) = 12`. Print it for L = 50k / 57k / 65k / 79k.
3. Keep `streams = max(8, min(seqs, floor(streams_fit(57k))))` if the plan stays "fit p50".
4. Record B, so F1's granularity is visible per half.

### F4 — MEDIUM — the watchdog's "growing log = alive" check is defeated by our own /metrics polling

**Location:** CELL 6 `_e15b_watchdog`:
- `_tk = _e15b_gen_tokens_total()` (a GET /metrics);
- then `os.path.getsize(_srv.log_path)`: if the size changed, `continue`;
- `/health` is probed only when the log is silent.

The capacity probe (`kv_poll_sec` 5) GETs /metrics too.

**Evidence:**
- uvicorn's access log is on in both versions. The real logs contain lines like `INFO: 127.0.0.1:… - "GET /metrics HTTP/1.1" 200 OK`: 15 in the E18 0.32 log, 1 in E12-V 0.19 (its single heartbeat).
- Every watchdog poll is preceded by our own /metrics GET, so the log grows every 10 s by construction for as long as the API process answers HTTP. The `/health` branch (and `_WD_FAIL` 180 s) is unreachable then.

**What still works (verified):**
- process death → `alive()` false → relaunch;
- vLLM's API exits on engine death: `launcher.watchdog_loop` → `terminate_if_errored` with `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` default 0;
- a stuck worker becomes engine death after `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` = 300 (0.19 `envs.py` 186, `multiproc_executor.py` 304/316);
- a blocked API event loop makes /metrics time out, so no access line, so the log goes silent and /health is probed.

**Scenario:** any failure where the API keeps answering /metrics but no tokens are produced and the process does not exit. One example: an engine stall that is not an `execute_model` RPC timeout. It is never relaunched, and the rest of the half (hours) is lost. Low probability, high cost.

**Fix:** either works.
- Add `--disable-uvicorn-access-log`. It is in 0.19 `cli_args.py` 226; the Kaggle CPU kit's real argument parser should confirm it for 0.32. The log then grows only with engine stats and warnings.
- Or base liveness on progress: a relaunch when `vllm:num_requests_running > 0` but `vllm:generation_tokens_total` has not increased for N × 60 s, with a /health probe in between.

### F5 — MEDIUM — the isolation preflight does not exercise vLLM's prefix-cache path

**Location:** CELL 6 `_e12_r6_preflight(n)`: n concurrent requests "Reply with exactly this code and nothing else: ZQ…K<i>", about 30 prompt tokens, `enable_thinking False`, max_tokens 24, plus 2 repeated pairs. Its result sets the worker count (`workers_for_isolation`: PASS = all streams, otherwise 6).

**Evidence:**
- vLLM's prefix cache works only in full blocks (1,600 tokens; about 832 with NB2's FP16 SSM, see F1). A hybrid hit also needs a mamba checkpoint.
- A 30-token prompt never creates or hits a cached block. So the test checks batch isolation only. The SGLang version was aimed at radix-cache contamination (#36548), which is token-granular.
- The vLLM 0.19 log itself warns: "Prefix caching in Mamba cache 'align' mode is … experimental".
- Every Duck turn after the first gets 67-94 % of its prompt from that path (F1). A wrong reused SSM state would corrupt outputs silently, and a PASS here says nothing about it.
- Weak positive evidence: E12-V run 2 (align caching on, 2 branches) accepted 2/2 with correct test output. E19 passed the S1 vLLM 0.32 engine configuration with prefix caching + MTP, on single-turn Poetiq requests.

**What does work (verified):**
- chat_template_kwargs merge `default | request` (0.19 `engine_serving.py` 798-807), so `{"preserve_thinking": true, "enable_thinking": false}`;
- the qwen3 parser returns everything as content when thinking is disabled (0.19 `qwen3_reasoning_parser.py`);
- the E12-V cold smoke with the same kwargs returned content "2 + 2 equals 4.".

So markers in `message.content` are valid.

**Fix:** add a cache-path step, about 3 extra requests per stream, small in time:
1. Build one shared prefix of ≥ 3 blocks (≥ 5,000 tokens of digit rows, like `_e12_cache_diag`'s `_blob`).
2. Give each request its own marker after the prefix and send them concurrently.
3. Repeat each one.
4. Require: own marker in every reply, `cached_tokens ≥ 1 block` on the repeats, and cold reply == cached reply at T 0 with thinking off (or equal first 16 tokens).

### F6 — LOW — the preemption count is the number of 10-s intervals with a preemption

**Location:** `_e15b_server_log_summary` `preempt_lines`, `e11_decode_summary` `preemption_lines`, E21 report `preempt_lines`.

**Evidence:** the vLLM v1 scheduler does not log individual preemptions (no logger call in 0.19 or 0.32 `core_scheduler.py`). The only text is ", Preemptions: N" inside a stats line. The counter `vllm:num_preemptions_total` exists: 0.19 loggers.py 583, and the real 0.32 capture.

**Fix:** read `vllm:num_preemptions_total` at solve start and end, exactly like `_e15b_gen_tokens_total` (per server, with the relaunch accumulation). Report that as the count; F2's fixed regex then also gives N per interval.

### F7 — LOW — `prefix_cache_hit_pct_last` is a rolling hit rate of the last 1,000 requests

**Location:** the summaries and the E21 report `prefix_cache_hit_pct_last`.

**Evidence:**
- `CachingMetrics(max_recent_requests=1000)` (0.19 `stats.py` 35-75; 0.32 uses the same class).
- R3K2 had 2,303 requests, so "last" covers the last 43 %.
- vLLM counts queries per scheduling of a request, so preempted and rescheduled requests count again.

**Fix:**
- Label the field as rolling.
- Add the whole-solve delta of `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` (both are in the real 0.32 capture).
- Keep the per-request `cached_pct_requests` (from usage) as the main number. It is already in the report, and it is engine-agnostic.

### F8 — LOW — `startup.kv_tokens` means different things in the two halves

**Location:** `_e15b_server_log_summary` `startup.kv_tokens` = the first "GPU KV cache size" value.

**Evidence:**
- 0.19 logs (num_blocks // groups) × block = 337,600 for an FP8 pool that is really 845 blocks × 1,600.
- 0.32 logs a group-aware capacity, 1,110,415, equal to factor × max_len.

**Fix:** label it per version, or report `ENGINE_INFO.capacity.effective_kv_tokens`, which is consistent across versions.

### F9 — LOW — the heartbeat shows only running, queue and KV usage on vLLM

**Location:** CELL 6 `_e12_heartbeat` keep filter ("num_running_reqs", "num_queue_reqs", "gen_throughput", "token_usage", "cache_hit_rate").

**Evidence:**
- `metrics_snapshot` aliasing works: the real E12-V 0.19 heartbeat printed `{"num_running_reqs": 2.0, "num_queue_reqs": 0.0, "token_usage": 0.05798816568047338}`.
- vLLM has no gen-throughput or hit-rate gauge, so those two keys never appear.
- The snapshot does include `vllm:generation_tokens_total` and the `prefix_cache_*` counters, but the keep filter drops them.
- The `{name: value}` dict collapses label rows. vLLM has one engine row each (labels `{engine="0",model_name=<served>}`, as in the real 0.32 capture), so nothing is lost.

**Fix:** add "generation_tokens_total", "prefix_cache_hits", "prefix_cache_queries" and "num_preemptions" to the keep filter. This is cosmetic.

### F10 — LOW — CELL 7 `_r6e_concurrency` parses SGLang decode lines

**Location:** `e21_cell7` → `_r6e_concurrency` (looks for `TP0]`, "Prefill batch", "Decode batch", "gen throughput (token/s)").

**Evidence:** vLLM logs have none of these, so the result is always `{}`.

**Fix:** none needed; label it "SGLang only", or drop it from the vLLM report.

### F11 — LOW — `e11_kill_server(wait=False)` at the end of `e21_cell7` raises a TypeError, which is swallowed

**Location:** CELL 7 tail `e11_kill_server(wait=False)`, which calls `E11_SERVER.kill(wait=wait)`. `E18Server.kill(self, fast=False)` has no `wait`.

**Evidence:** the signature mismatch. `except Exception: pass` hides it. The server is then killed by `e21_end_half`, so the effect today is nil. The dead `_e12_parity` has the same mismatch, plus `E11Server`, which is undefined in E21.

**Fix:** `E11_SERVER.kill(fast=not wait)` in `e11_kill_server`, or remove the call.

### F12 — LOW — the preflight records the DFlash2 drafter config

**Location:** `_e12_r6_preflight` reads `SGL_DRAFT_PATH/config.json` into `drafter_target_layer_config`.

**Evidence:** E21 runs MTP-3 from the main checkpoint; DFlash2 is not loaded. The record is misleading in the report.

**Fix:** skip it on vLLM, or record `SPECULATIVE_CONFIG`.

### F13 — LOW — the model-id fallback would make every Duck request fail

**Location:** CELL 6: if `/v1/models` fails, `E12_RUN["model_id"] = str(SGL_MODEL_PATH)`.

**Evidence:** vLLM serves only `--served-model-name` (`Qwen/Qwen3.8-27B-INT4` or `-FP8`; the real E12-V `/v1/models` returned exactly that). A request with another model name gets a 404 "model does not exist". In NB2 / fallbacks, `SGL_MODEL_PATH` is not even the loaded model.

**Probability:** very low, because `/v1/models` already answered in `e21_launch` seconds earlier.

**Fix:** fall back to `E21_SERVED[ENGINE_INFO["cfg"]["weights"]]`.

### F14 — LOW — the watchdog thread is not joined at the end of a half

**Location:** CELL 6: the `e15b_watchdog` daemon thread exits only at its next `_E12_STOP.wait`. `e21_relaunch` (kill → `_e18_kill_gpu_pids` → `_wait_gpu_release` ≤ 180 s → `e21_launch`) can be in flight when half A ends.

**Scenario:** if a relaunch were still running after `e21_end_half(A)`:
- its `_e18_kill_gpu_pids()` would kill half B's server while it loads;
- or its `e21_launch` would start a second server.

The watchdog relaunches only with ≥ 20 min left, and a relaunch is bounded:
- `wait_ready` timeout = max(60 s, min(30 min, half end − now − 30 min)), which is 60 s whenever < 31 min are left;
- plus a 300-s smoke timeout and a 180-s GPU wait.

So in practice it finishes before the window ends. Improbable today.

A side effect of the same timeout formula: with 20-31 min left, a relaunch can never become ready. vLLM needed 412-513 s from first log line to startup complete in the real logs. So those relaunches (and the B3 retry) are doomed. That is harmless: the server is dead anyway.

**Fix:** at the end of `e21_cell6`, join the watchdog thread (timeout = its worst case). Or pass a half token to `e21_relaunch` and refuse to act when `E21_HALF` has changed.

---

## Verified (no finding)

1. **Metric names, units and label rows.**
   - Names, both versions: `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`, `vllm:num_requests_waiting` (gauges), and `vllm:generation_tokens` (counter, exported as `vllm:generation_tokens_total` plus `_created`).
   - Sources: 0.19 loggers.py 437/447/483/636 (set at 1040-1046/1126); 0.32 loggers.py 527-626/798 (set at 1147-1165/1270).
   - Units: KV usage is a fraction 0..1. The doc says "1 means 100 percent usage", and the value is `block_pool.get_usage()` = 1 − free/total blocks, where cached-but-unreferenced (evictable) blocks count as free (0.19 block_pool.py 487-498). So the gate sees blocks held by running requests, mamba state blocks included.
   - Labels: `["model_name", "engine"]`, one row per engine.
   - Real 0.32 capture: `vllm:kv_cache_usage_perc{engine="0",model_name="q27"} 0.3887…`, `vllm:generation_tokens_total{…} 454276.0`. In 0.32, `num_requests_waiting` includes deferred requests (0.32 1150-1154).
   - `_e12_capacity_probe`'s max over rows is correct for one engine.
   - `_e15b_gen_tokens_total`'s exact-name match excludes `_created` and `_bucket`.
   - `_E12V_PROM_RE` parses values like `1.7e+09` and labels with `/` and `.`.
2. **KV gate (`solve_all_r6`).**
   - `_finite01` accepts the 0..1 fraction and rejects NaN or out-of-range values.
   - `mamba_usage = None` is used only in telemetry: `SERVER_LAST`, the `gauges` column, the per-request `mamba` field. `kv_blocked` reads only `kv`.
   - `probe_once` stamps `kv_t` on every attempt, failures included. A failed fetch gives `kv=None`, so the gate fails open, `kv_missing` is counted, and a warning appears after 30 s.
   - A sample is stale only if the probe thread itself blocks > `kv_stale_sec` (30 s). `e21_prom` uses `urlopen(timeout=10)`, so `kv_stale_checks` should stay about 0.
   - With nothing running, a hold is overridden (`kv_idle_overrides`), so there is no deadlock.
   - vLLM gauges are updated every engine step. After the last request of a step finishes, the stats are made after its blocks are freed, so an idle server does not show a stale high value.
3. **Stats-line format.**
   - The prefix "Engine 000: " is not anchored, so it is fine.
   - Matches: 46/46 real 0.19 lines and 164/164 real 0.32 lines.
   - Idle intervals are logged at debug level (not printed) in both versions, so only working lines exist.
   - Exceptions: F2.
4. **Capacity lines.**
   - `GPU KV cache size: ([\d,]+) tokens` and `Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x` match the 0.19 two-line format and the 0.32 one-line format ("GPU KV cache size: 1,110,415 tokens, Maximum concurrency for 106,496 tokens per request: 10.43x").
   - Effective KV = factor × max_len is consistent across versions. Its meaning is limited: F3.
5. **/health, readiness and /v1/models.**
   - The socket is bound early (`setup_server` → `create_server_socket`, bind only), but uvicorn listens only after the engine is built. So /health is refused during load and returns 200 only when the engine is up (`health.py`: 503 on `EngineDeadError`).
   - `E18Server.wait_ready` (/health + its own "GPU KV cache size" line) is sound for both versions.
   - Startup log silence: at most 60 s in both real logs (the shm_broadcast heartbeat) vs the 600-s stall rule. Startup spans: 412 s (0.19) and 513 s (0.32).
   - `/v1/models` returns `{"data":[{"id": <served name>}]}` (real E12-V).
   - After the server died: connection refused. The watchdog catches it through `alive()`. The API exits on engine death (launcher watchdog, KEEP_ALIVE 0).
6. **Prompt-token details.**
   - vLLM 0.19 sets `prompt_tokens_details` only when `enable_prompt_tokens_details and num_cached_tokens` (chat serving 1214/1601), so it is null on a 0-hit request. The real E12-V cold smoke shows `"prompt_tokens_details": null`.
   - Duck (`arc2_duck.py` 2054-2058) and CELL 7 treat null as 0 cached.
7. **Reasoning in history.**
   - `ARC3_REASONING_HISTORY_KEY=reasoning` is read at call time (`tool_agent._reasoning_history_keys`). `DUCK_ENV` is applied once in CELL 3, before the override; nothing re-applies it.
   - E12-V run 2: prompt growth = previous completion + the new message (+45, +1,058, +438, +160, +91, +34 for b0). The full prior thinking is in the prompt, so prefix misses are not caused by dropped or re-rendered reasoning (F1).
8. **Watchdog token accounting (B6) on vLLM.**
   - The counter resets per server. `tok_base` is read after the preflight.
   - A relaunch adds `tok_last − tok_base` to `tok_acc`, then the base becomes 0.
   - The 96-token smoke of a relaunched server is counted, which is negligible.
9. **Ports and logs per launch.** Each launch writes its own log `vllm_server_half<H>_<tag>.log`, and `[-1]` capacity regex hits are per file. Half A and half B use different ports (1234 / 8011), and half A's process group is killed and the GPUs released before half B starts.

## Summary for the caller

No BLOCKER in this scope. The run cell and scheduler read vLLM's real metric names with the right units (fraction 0..1, one engine row). The KV gate handles the missing Mamba gauge and stale or missing samples fail-open as designed. /health, /v1/models and the readiness logic match vLLM 0.19 and 0.32.

The issues:
- **HIGH (F1):** structural. On vLLM's hybrid align-mode prefix cache, each Duck turn re-prefills the previous turn's entire output plus 1.6-3.2k tokens. The cache rule `floor(prev_prompt/1600)·1600 − 1600` holds 10/10 on E12-V run 2. On R3K2's request pattern that is about 74.7 % cached vs SGLang's 93.4 %: +11.1 M prefill tokens, roughly 1-2 h of a 5.5-h half window.
- **MEDIUM:**
  - F2: the stats regex loses the preemption and deferred lines;
  - F3: the capacity check ignores about 12 fixed blocks (19.2k tokens) per stream;
  - F4: our own /metrics GETs keep the server log growing, so the watchdog's /health path is dead;
  - F5: the isolation preflight never touches the prefix cache.
- **LOW:** F6-F14, reporting accuracy and robustness.
