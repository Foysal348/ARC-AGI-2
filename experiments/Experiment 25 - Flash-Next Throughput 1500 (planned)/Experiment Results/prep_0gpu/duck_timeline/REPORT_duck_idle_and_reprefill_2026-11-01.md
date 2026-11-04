## K100Q offline analyses: Duck idle timeline (2-1) and re-prefill breakdown (2-2), 2026-11-01

**The REPORT .md file was not written.** The harness blocks subagents from writing report files, so `REPORT_duck_idle_and_reprefill_2026-11-01.md` does not exist yet. The full text is below; please save it into the folder.

Everything else is in `D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/prep_0gpu/duck_timeline/`:
- `scripts/`: `extract.py`, `timeline.py`, `tool_to_server.py`, `reprefill.py`, `schema.py`, `run_all.py`;
- `data/`, `out_analysis1/`, `out_analysis2/` and `log_*.txt`.

`python -I scripts/run_all.py` re-runs everything end to end; I re-ran it from that folder and the numbers reproduce. Scratch copies are in `D:/ARC AGI 2/_scratch/prep_timeline/`. No existing files were changed, and there was no GPU, Kaggle, git or push. No response, thinking or prompt text was read or printed: text fields entered the scripts only as character counts. No PNG plots: matplotlib is not installed here. The plot series are in `out_analysis1/timeline_10s.csv` and `heap_and_active_1s.csv`.

---

# K100Q offline analyses: Duck idle timeline (2-1) and re-prefill breakdown (2-2), 2026-11-01

- **Run:** `esfersami50/q27-xhigh-e26-k100-quality`. Merlin (vLLM 0.32 fork), klee100, TP2 × PP2, MTP K3, 21 workers, 30 tasks × 2 attempts, window 6,600.4 s.
- **Inputs** (in `_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/`):
  - `duck/*/nb5_requests.jsonl` (742 per-request records);
  - `duck/*/branch_result.json` (60 attempts);
  - `nb5_sampler.jsonl` (every 15 s);
  - the server log (10 s metric lines);
  - `e12_run_report.json` (scheduler gauges);
  - `nb5_report.json` / `k100_quality_report.json` (Prometheus counters and latency means).
- **Method:** CPU only, numeric fields only.

## Headline

1. **The 385 vs 414 gap has no tool, client or GPU cause.**
   - 414 is the median of the same 10 s series whose mean is 385: 413.55 over all 670 busy log lines, 415.1 over the 660 lines inside the window.
   - The real level with all 21 workers busy is **457.5 tok/s** (curve fitted to rate vs concurrency). The other measures are 459.9 (stable-21 mean), 453.7 (mean of all 327 lines at 21) and 471.8 (their median).
   - Of the 72 tok/s between 385 and 457.5, **67.0 tok/s (93 %) is work exhaustion.** From 3,628 s on, no attempt was left to start. The heap was empty, all 30 first attempts had started, and each second attempt is queued only when its first attempt ends.
   - The rest is small: tools + turn assembly 0.4 tok/s, client in-request overhead (tokenize guard + HTTP/API) 1.0, server queue 0.2, setup/tail 0.1, admission while work was queued 0.1, residual 3.3.
   - Before the work ran out (0-3,628 s, 55 % of the window) the run averaged **452.6 tok/s**, with mean running 20.87 and 90.1 % of lines at 21.

2. **A ready-queue / async client would recover about 1.8 tok/s (≈ 0.5 %).**
   - Upper bound 3.7 tok/s (1.0 %). Item 2-1's expected 0-7 % comes down to **≤ 1 %**.
   - Tools are very short: the gap from one response to the next request is p50 0.29 s, mean 0.45 s. Requests last 147 s on average.
   - The real lever for this loss is attempt supply (Tier S-1), not the client.

3. **The 58 % re-prefill claim is too low. The measured share is 72.7 % of server fresh-prefill tokens** (2,430,527 of 3,343,259).
   - It is 74.6 % of the Duck records' own uncached tokens and 78.5 % of later-turn uncached tokens.
   - The 58 % came from a different run: 650k of 1,115k in the E25 speed-data report. It was then applied to K100Q as 1.94 M. The measured figure is 2.43 M (+25 %).
   - The mechanism is confirmed in all 677 later turns. The prefix hit always ends 64-127 tokens before the end of the previous prompt and never reaches into the generated tokens.

4. **A perfect generated-tail checkpoint would save about 2.43 M prefill tokens.**
   - That is 2.51 M with the 4 requests cut at the window end, and 10.1 % of all prompt tokens.
   - In engine time it is worth **6.0-8.2 % of the run**:
     - 540 s at the boot-probe prefill rate (8.2 % of the window);
     - 181k generated tokens at the measured in-run marginal prefill cost (7.1 %, +27.4 tok/s on 385);
     - 6.0 % with the within-concurrency slope.
   - This is above the priority list's 2-6 %. It is a gross upper bound: the cost of the checkpoint itself is not counted.

---

## Analysis 1: occupancy / idle timeline

### What each timestamp means (harness source)

- **Harness:** `Experiment 15 .../src/duck_b/arc2_duck.py`, sha256 b9f67010, which matches the notebook manifest. The NB5 wrappers are in notebook CELL 3.
- **`nb5_requests.jsonl` `t0` / `t1`** (written by `_nb5_cc`):
  - `t0` is taken *before* the guard. The guard calls the server's `/tokenize` on the full rendered prompt (`how: tokenize` on all 742 requests).
  - `t1` is taken after Duck's `_chat_completion` returns.
  - So [t0, t1] = guard tokenize + HTTP + API-server render + engine queue + prefill + decode + response.
- **`NB5_TRACE.inflight`** (in the sampler) counts requests *after* the guard (`enter()`).
- **`branch_result.requests[]`** (`t`, `sec`, `idle_sec`) wraps the same call one level up. Its offset from `t0` averages −0.0001 s and `sec` differs by +0.007 s on average, so these layers cost nothing.
- **The gap t1(k) → t0(k+1)** contains:
  - Duck's response handling;
  - the python tool (a fresh sandbox subprocess per call) and host verification on submit;
  - message trim / token estimate, two JSON deep copies, the `prompt.log` snapshot write and the arc2 request-begin JSON dump.
  - There are **no tool start/end timestamps**, so tool time and assembly time are split with a fit (see below).
- **Attempt start / end:**
  - start = record `t0`, set when the agent is created inside `BranchSession.run`;
  - end = `t0 + runs[0].sec`; every attempt had exactly one run.
  - The scheduler's `durations` sum is 109,232.5 s against 109,213.2 s of run time, so publishing costs 19.3 s in total.
- **Clocks:** client epoch and server-log UTC agree. The best offset is 0 s; the mean |running difference| against the sampler is 0.15 over 440 samples.

### Worker-second budget (21 × 6,600.4 = 138,608.4 worker-s)

| State | worker-s | share |
|---|---|---|
| In request (client t0..t1) | 108,834.1 | 78.52 % |
| └ engine running (prefill + decode), estimated | ≈ 108,586 | 78.34 % |
| └ guard `/tokenize` (Little's law: sampler requests − inflight) | 112.5 (0.152 s / request) | 0.08 % |
| └ HTTP / API layer (inflight − running − waiting) | 135.3 (0.183 s / request) | 0.10 % |
| └ engine queue (Prometheus mean 0.17 s × 741) | ≈ 126 | 0.09 % |
| Gap: tool + turn assembly | 306.9 | 0.22 % |
| Setup (agent created → first request) | 10.4 | 0.01 % |
| Tail (last response → attempt end: harvest / partial test / verify) | 57.1 | 0.04 % |
| **Idle: no attempt assigned** | **29,399.9** | **21.21 %** |
| └ while work was queued (KV-gauge holds / turnover) | 92.1 | 0.07 % |
| └ heap empty before the last attempt start (5,622.4 s): attempt 2 waits for its attempt 1 | 13,964.6 | 10.07 % |
| └ after the last attempt start (pure drain) | 15,350.5 | 11.07 % |

- **Heap and running:** the heap first emptied at 2,952 s. Running stayed at 21 until 3,628 s, then fell below 21 for good.
- **Admission:** 34 KV-gauge holds, 62.7 s in total. NB5 reservation: 60 grants, 0 waits, 0 s.
- **Sampler means:** client requests 16.509, inflight after guard 16.497, server running 16.474, server waiting 0.002. In 9 of 439 samples, inflight was above running + waiting.
- **Cross-check of client overhead:** successful client request time is 108,638.0 s over 737 requests, against a Prometheus e2e sum of 108,554.5 s over 762 requests (25 of them non-Duck). That is consistent with about 0.3 s of client overhead per request.
- **Per worker** (`out_analysis1/per_worker.csv`; greedy feasible assignment, since threads are interchangeable):

| Per worker | Total | Range |
|---|---|---|
| Attempts | 60 | 1-5 |
| Requests | 741 | 18-50 |
| GPU time in server | 108,460.6 s | 4,353-6,544 s |
| Client in-request | 247.7 s | 6.0-16.7 s |
| Engine queue | 126.1 s | 3.1-8.5 s |
| Tool + assembly | 306.9 s | 6.3-52.0 s |
| Setup + tail | 67.4 s | 1.5-4.8 s |
| Idle | 29,399.9 s | 19.1-2,212 s (almost all after each worker's last attempt) |

### Engine side (660 in-window 10 s log lines)

- **Occupancy:** running < 21 in 50.45 % of lines, = 21 in 49.55 %; mean running 16.44.
- **Zero-running intervals: none.** The minimum running count is 4 in both the log and the sampler.
- **Running histogram:** 4:5, 5:68, 6:19, 7:27, 8:15, 9:4, 10:13, 11:34, 12:8, 13:12, 14:7, 15:4, 16:7, 17:11, 18:2, 19:24, 20:73, 21:327.
- **Ramp:** 21 was first reached on the line at 17.6 s; the first line showed 11.
- **Before 3,628 s:** 33 isolated dips to 19-20, each one 10-30 s line. These are request or attempt turnovers caught at the sampling moment.
- **From 3,628 s to the end:** one continuous stretch below 21, falling to 4.
- **Split at the last attempt start (5,622 s):**
  - before: 41.8 % of lines below 21, mean running 18.39, mean active attempts 18.51, 416.2 tok/s;
  - after: mean running 5.27, 207.9 tok/s.
- **Split at 3,628 s:**
  - before: 452.6 tok/s, running 20.87, 90.1 % of lines at 21, prompt 573.9 tok/s;
  - after: 303.1 tok/s, running 11.03.
- **Server waiting** was above 0 on only 3 lines (1, 1 and 10).

### Rate versus concurrency (stable 10 s lines)

- **Mean gen tok/s at n running:** 5: 197.6, 6: 240.6, 7: 245.8, 8: 247.4, 9: 287.5, 10: 310.6, 11: 321.4, 12: 359.0, 13: 335.8, 14: 383.7, 16: 386.6, 17: 338.6, 19: 435.5, 20: 431.1, 21: 459.9.
- **Pure-decode lines only:** 21: 515.6 (74 lines), 20: 456.6, 19: 470.6.
- **Fit:** R(n) = 99.17 + 22.11 n − 0.2405 n², so R(21) = 457.5. The marginal gain from 20 to 21 streams is 12.25 tok/s; the average per stream at 21 is 21.8 tok/s.
- **Prefill stall at 21 running:** gen = 500.4 − 0.0745 × prompt_tps (n = 294).
  - Each fresh prefill token costs 0.0745 generated tokens, which is the same as a prefill rate of about 6,717 tok/s. The within-n slope is −0.0623.
  - Over the run: 3.34 M fresh tokens × 0.0745 = 249k generated tokens, i.e. **37.8 tok/s** (208k with the within-n slope).
  - This prefill load is already inside the 457.5 reference, so it is not part of the 385 → 457.5 gap.

### 385 → 457.5 decomposition (lost generated tokens versus R(21) × window)

Method: for each 10 s line, the occupancy loss R(21) − R(n̄) is split in proportion to the missing slots by cause. Causes come from the exact client intervals and the reconstructed heap. The residual is R(n̄) − actual. The parts sum to the total within 183 tokens.

| Cause | missing slot-s | lost tokens | tok/s | % of 385 |
|---|---|---|---|---|
| Work exhausted: heap empty, attempt 2 waits for attempt 1 (before the last start) | 13,898.1 | 199,993 | 30.3 | 7.9 % |
| Work exhausted: end drain (after the last start) | 15,360.9 | 242,485 | 36.7 | 9.5 % |
| Client in request (guard tokenize + HTTP/API; 10 s method) | 516.6 | 6,788 | 1.0 | 0.27 % |
| Tool + assembly gap | 306.9 | 2,527 | 0.4 | 0.10 % |
| Engine queue (server waiting) | 120.0 | 1,137 | 0.2 | 0.04 % |
| Idle while work was queued (KV holds / turnover) | 100.2 | 721 | 0.1 | 0.03 % |
| Tail (finalize) | 56.3 | 528 | 0.1 | 0.02 % |
| Setup | 13.3 | 136 | 0.0 | 0.00 % |
| Residual (prefill load / context variation around R(n)) | — | 21,962 | 3.3 | 0.86 % |
| **Total** | | **476,460** | **72.2** | |

- **Waterfall:**
  1. pure decode at 21 ≈ 500-516 tok/s;
  2. minus the prefill load (≈ 40 at the mean prompt rate of the 21-running lines) → 457.5;
  3. minus work exhaustion 67.0;
  4. minus client / tool / queue / setup / tail / admission 1.8;
  5. minus residual 3.3 → **385.3**.
- The 29 tok/s between 385 and 414 is the left tail of the series after 3,628 s, i.e. the same work-exhaustion loss.

### Tool durations and the tool → server path

- **Tool + assembly gap** (677 gaps between successful requests; 681 including the gaps before the 4 cut requests):
  - p50 0.29 s, p90 0.84 s, p99 1.91 s, mean 0.45 s, max 24.6 s;
  - 6 gaps were over 2 s, 42 s in total;
  - after request 0: p50 0.37 s, max 0.44 s.
- **Turn assembly** (scales with context; from a p90-trimmed fit of gap vs serialized characters: 0.349 s per MB, intercept 0.285 s):
  - per turn: mean 0.032 s, p50 0.024 s, p90 0.073 s, p99 0.109 s;
  - serialized context: p50 0.068 MB, max 0.326 MB;
  - lower envelope: the 10th percentile of the gap rises from 0.19 s at 0.01 MB to 0.31 s at 0.26 MB.
- **Tool execution** (gap − assembly): p50 0.24 s, p90 0.82 s, p99 1.83 s, mean 0.42 s, max 24.5 s.
- **Tool result → next request reaches the API server:**
  - central estimate (assembly + guard tokenize 0.152 s + half of the 0.183 s HTTP/API layer): p50 0.27 s, p90 0.32 s, p99 0.35 s, mean 0.28 s;
  - lower bound (guard only): 0.15 s mean;
  - upper bound (previous response → arrival, if the tool took 0 s): p50 0.63 s, p90 1.17 s, p99 2.24 s.
- **After arrival (Prometheus means over 762 requests):** engine queue 0.17 s, TTFT 2.01 s, prefill phase 1.68 s, decode 140.48 s, e2e 142.46 s.
- **Other client-side timings:**
  - attempt setup: mean 0.17 s (p90 0.68 s);
  - tail: mean 0.95 s (p90 1.34 s);
  - client request duration: p50 44.3 s, p90 427 s, p99 1,286 s, mean 146.9 s, max 1,941 s.

### What a ready-queue design recovers

1. **Prepare the next request during decode / async client.**
   - This can only hide the work that does not depend on the tool result:
     - the guard `/tokenize` (0.152 s per request);
     - the HTTP/API render (0.183 s per request; the API server re-renders and re-tokenizes the full prompt, about 35k tokens on average, every turn);
     - assembly (0.032 s per turn).
   - That is 248 slot-s by Little's law (517 slot-s by the 10 s method) → **0.5-1.0 tok/s (0.1-0.3 %)**.
2. **A ready-for-GPU queue separate from waiting-for-tool** (a few more attempts than GPU slots).
   - On top of (1) it fills the tool gaps (306.9 slot-s, 0.4 tok/s), setup/tail (69.6, 0.1), admission/turnover (100, 0.1) and server waiting (120, 0.2).
   - **Total ≈ 1.8 tok/s ≈ 0.5 % of 385.** Upper bound: all 1,113 such slot-s filled at the average 21.8 tok/s per stream gives 3.7 tok/s (1.0 %).
3. **Neither recovers the 67 tok/s (17.4 %) lost to work exhaustion.**
   - That loss comes from this run's design: 30 tasks × 2 attempts in a fixed window, attempt 2 queued only when its attempt 1 ends, stage 3 off.
   - It needs more startable work while time remains: S-1 extra attempts / attempt-2 re-runs, or a larger task set.
   - In a 12 h run with 120-240 tasks only the final drain would remain. Here it lasted from 3,628 s to 6,600 s, and its length is set by the longest attempts.

---

## Analysis 2: re-prefill breakdown

### Definitions (per attempt, turn k ≥ 1)

P = prompt tokens, C = cached tokens (`usage.prompt_tokens_details.cached_tokens`), G = completion tokens. The uncached part U_k = P_k − C_k splits **exactly** into:
- **(c)** old-prompt tail not hit = P_{k−1} − C_k;
- **(a)** previous turn's generated tokens = G_{k−1};
- **(b)** new content = P_k − P_{k−1} − G_{k−1}: tool result, user/nudge text and template wrappers.

First turns are a separate class **(d)**.

### Checks (all 737 successful requests)

- **Hit never reaches the generated tokens:** C_k ≤ P_{k−1} in 677 / 677 later turns.
- **Hit always ends just before the previous prompt's end:** P_{k−1} − C_k is between 64 and 127 in 677 / 677 turns (mean 95.6), i.e. one to two prefix units of 64 short.
  - 0 eviction-like misses (> 127), 0 trims, prefix_ok in every turn.
  - All cached values are multiples of 64.
- **Reasoning is kept in history:** (b) is at least 41 in every turn and never negative, so the previous turn's reasoning is fully present in the next prompt (raw-reasoning history is on).
- **No cross-attempt prefix sharing:** all 60 first turns had cached = 0.
  - Likely causes, not verified here: the per-attempt isolation marker in the opener (rev6 preflight isolation PASS, 21 / 21), and/or Mamba checkpoints existing only at each request's own aligned prompt end.
- **Mechanism confirmed:** the token counts match the earlier claim that Mamba/GDN checkpoints are published only at prompt positions under RecoverSSM + PP. The server log shows `prefix_match_unit 64`, `mamba_cache_mode align`, `use_replayssm True`, `block_size 4096`, `max_num_batched_tokens 1024`, and "GDN RecoverSSM path taken … align=True".

### Totals

| Component | tokens | share of records' uncached (3,259,726) | share of later-turn uncached (3,095,304) | share of server fresh (3,343,259) |
|---|---|---|---|---|
| (a) previous turn's generated tokens | **2,430,527** | **74.56 %** | **78.52 %** | **72.70 %** |
| (b) new tool result / user content + wrappers | 600,064 | 18.41 % | 19.39 % | 17.95 % |
| (c) prefix-unit alignment tail (64-127 per turn) | 64,713 | 1.99 % | 2.09 % | 1.94 % |
| (d) first-turn prompts (60 × mean 2,740) | 164,422 | 5.04 % | — | 4.92 % |
| Server fresh outside the Duck records | 83,533 | — | — | 2.50 % |

- **Client records:** 737 requests, prompt 23,866,318, cached 20,606,592 (86.34 %), uncached 3,259,726, completion 2,537,367.
- **Server phase counters:** prompt 24,149,339, hits 20,806,080 (86.16 %), fresh 3,343,259, generated 2,543,542. The E26 summary's 85.4 % / 24.54 M is the whole-log figure, including the boot probes.
- **The 83.5k server-only fresh tokens:** the 4 requests cut at the window end are estimated at 84.1k, from their previous turn plus the median (b). Preflight and parity requests account for the rest.

### Per-turn distributions (677 later turns)

| | p10 | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| U (uncached) | 854 | 2,242 | 10,745 | 32,341 | 42,502 | 4,572 |
| (a) previous generated | 146 | 1,105 | 9,591 | 31,662 | 41,486 | 3,590 |
| (b) new content | 222 | 617 | 1,940 | 3,842 | 6,933 | 886 |
| (c) alignment tail | 71 | 95 | 120 | 127 | 127 | 95.6 |
| (a) / U | 0.1 | 0.6 | 0.9 | 1.0 | 1.0 | 0.60 |
| prompt P | 5,643 | 27,405 | 77,116 | 112,558 | 116,431 | 35,010 |

- **First-turn U:** p50 2,724, range 2,570-2,872.
- **(a) share by turn index:**
  - turn 1: 24.2 % (60 turns; the first tool turn has a short generation);
  - turns 2-4: 72.2 % (180);
  - turns 5-9: 84.3 % (244);
  - turns 10-19: 85.1 % (171);
  - turns 20+: 75.3 % (22).
- **(a) share by size of the previous generation:**
  - < 1k: 29.2 % (323 turns, 137k tokens);
  - 1k-4k: 70.6 % (181, 389k);
  - 4k-16k: 88.7 % (141, 1.11 M);
  - ≥ 16k: 96.3 % (32, 796k). These 32 turns hold 33 % of all (a) tokens.

### What a perfect generated-tail checkpoint would save

With a checkpoint at the end of generation, turn k's hit would end at align(P_{k−1} + G_{k−1}) instead of align(P_{k−1}). The 64-127 alignment tail moves to the new end, so each turn saves about G_{k−1} (bounds G ± 63).

| Basis | prefill tokens saved | of server fresh | of all prompt | engine s at probe 4,504.6 tok/s | % of window | gen tokens recovered (0.0745 per prefill token) | % of generated | tok/s gain |
|---|---|---|---|---|---|---|---|---|
| Records (677 turns) | 2,430,527 | 72.7 % | 10.1 % | 539.6 | 8.17 % | 181,074 | 7.12 % | +27.4 |
| + 4 cut requests | 2,511,799 | 75.1 % | 10.4 % | 557.6 | 8.45 % | 187,129 | 7.36 % | +28.4 |
| Low bound (G − 63) | 2,388,128 | 71.4 % | 9.9 % | 530.2 | 8.03 % | 177,916 | 6.99 % | +27.0 |
| High bound (G + 63) | 2,473,178 | 74.0 % | 10.2 % | 549.0 | 8.32 % | 184,252 | 7.24 % | +27.9 |

- **Within-concurrency slope (0.0623):** the records' saving is 151,422 tokens = 5.95 %.
- **Prometheus prefill-phase time** in proportion to these tokens is 931 s. It overlaps decode, so it shows up as TTFT, not as exclusive engine time.
- **TTFT added per turn by the re-prefill** (at the probe rate): mean 0.8 s, p50 0.2 s, p90 2.1 s, p99 7.0 s, max 9.2 s.
- **Step limit:** with `max_num_batched_tokens 1024` and 21 × 4 decode tokens per step, at most about 940 prefill tokens fit in one step, so a p90 tail of 9.6k tokens takes at least 11 steps.

## Caveats and missing fields

- **Fields that do not exist in the saved data:**
  - tool start/end timestamps (the tool vs assembly split is a fit);
  - per-request guard `/tokenize` duration (a Little's-law mean is used);
  - per-request server arrival and first-token timestamps (uvicorn access lines carry no timestamps; Prometheus gives only means);
  - per-message token counts of the re-rendered assistant turn, so (a) uses `usage.completion_tokens`.
- **Error in (a):** the template re-render of the same reasoning + tool call can differ by a few tokens, and template wrappers land in (b), which is at least 41 in every turn. The error is at most tens of tokens against a mean (a) of 3,590. Best bounds for (a): 2.39-2.47 M given the 64-unit alignment.
- **Sampling granularity:** 10 s log lines (instantaneous running count, averaged throughput) and a 15 s sampler.
- **Drain vs context length:** R(n) at low n also reflects the long contexts of the tail streams, so part of the drain loss is context length, not only stream count.
- **Prefill-cost slope:** it is a regression on 294 lines at 21 running and may be confounded with turn boundaries and context length. The boot probe (4,504.6 tok/s, idle engine, 40k-token prompt) gives the other end, hence the 6.0-8.2 % range.
- **The savings are gross.** Publishing the Mamba/GDN state at finish time needs one more state page per request between turns; under align mode a state page is block-sized, and the KV pool peaked at 99.4 %. The four gates also still apply: token-ID identity, checkpoint position, KV + recurrent-state equivalence, and the PP0/PP1 fence.
- **Work-exhaustion idle is specific to this run's design** (30 tasks, 2 attempts, 6,600 s window, b1_accepted_mode before3, extra attempts off). It is not a throughput limit of the harness.