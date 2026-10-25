# Duck agent effective throughput on Kaggle 4x L4: where the wall time goes, and which serving/cache levers recover it (research notes, 2026-10-23)

Labels used throughout: **[M]** = measured from our own logs (script + output path given); **[D]** = derived arithmetic from measured numbers (method given); **[C]** = claimed by an external source (not re-measured by us); **[S]** = read from the local SGLang 0.5.21 source.

Abbreviations for local paths:
- `RES` = `D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Results/`
- `RUNS` = `RES/Kaggle Runs/`
- `G4B` = `D:/ARC AGI 2/_scratch/g4b_bundle_verify/runs/rev6d_EVAL12R6G4B/`
- `SGL` = `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/srt/` (SGLang 0.5.21 source; `_version.py` = 0.5.21)
- `SP` = `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/agent_tput/` (my read-only analysis scripts and their saved outputs)

Scripts written for this note (all read-only, run with `python -I`):
- `SP/wall_time_breakdown.py` → `SP/wall_time_breakdown_output.txt` / `.json` (time accounting from server log + branch usage)
- `SP/prefix_reuse_detail.py` → `SP/prefix_reuse_detail_output.txt` (per-request hit classification)
- `SP/reuse_offsets.py` → `SP/reuse_offsets_output.txt` (where the hit stops relative to the previous prompt end)
- `SP/miss_vs_prevlen.py` → `SP/miss_vs_prevlen_output.txt` (miss rate vs previous reply length / gap)

All five runs used the identical server config [M, from `server_args` line of each `sglang_server_try0.log`]: SGLang 0.5.21, TP4, INT4 W4A16 + DFLASH (8 draft tokens), `max_running_requests` 13, `chunked_prefill_size` 4096, `schedule_policy` lpm, `page_size` 1, `radix_eviction_policy` lru, `mamba_radix_cache_strategy` extra_buffer_lazy, `mamba_track_interval` 256, `max_mamba_cache_size` 56, `mamba_ssm_dtype` float16, `enable_mixed_chunk` False, `prefill_decode_interval` 0, `enable_hierarchical_cache` False, `enable_cache_report` True, `decode_log_interval` 40. KV pool 866,744 tokens; per GPU: target KV K+V 6.62 GB, draft KV 2.06 GB, mamba ssm 1.00 GB + intermediate ssm 1.97 GB ([M] `G4B/sglang_server_try0.log`, "KV Cache is allocated" / "Mamba Cache is allocated" lines). EVAL6, EVAL6R5 and EVAL12R5_LOW ran 6 client streams; EVAL30 and G4B ran 12.

---

## 1. Our own data: where does agent wall time go?

### Takeaway
On 4x L4 the engine's **prefill runs at only ~1,500 tok/s and stalls all decoding while it runs**, so every uncached prompt token costs about a quarter of a generated token's engine time. In E12 EVAL30, re-prefill after eviction took **55 % of the solve window** (7.6-7.7 ks of 14.0 ks), which is the whole "half of decode rate" gap. In the well-behaved runs (EVAL6, EVAL6R5, EVAL12R5_LOW, G4B rev 6d) prefill still takes **13-20 % of the window**, and the largest part of it is **re-prefilling the previous reply on every turn**: in 72-91 % of later requests the cache hit stops at the previous prompt's end. The rest of the G4B loss is under-occupancy: the probe bug held 6 running for 28 % of the time, and the "11 running + 1 queued" state. Whole-engine idle time (nothing in flight while tools run) is negligible (17-38 s per run).

### Cited Findings

**Engine mechanics (why prefill time is lost decode time)**
- With no prefill-decode interleaving configured, the scheduler always runs a pending prefill batch before decode ("Run prefill first if possible"; decode runs only when `new_batch is None`). `prefill_decode_interval` defaults to 0 (disabled) and `enable_mixed_chunk` is False in our runs, so a long re-prefill (4,096-token chunks back to back) stalls decode for all other streams [S] — `SGL/managers/scheduler.py` lines ~3723-3753 (`get_next_batch_to_run` tail); `SGL/arg_groups/fields/schedule.py` `prefill_decode_interval` help text.
- The `input throughput` field in a Prefill log line is tokens since the last prefill log divided by the gap since that log; the Decode line's `gen throughput` is tokens since the last decode log divided by its gap (so it includes any prefill stall in between). Decode lines are logged every 40 decode iterations [S] — `SGL/managers/scheduler_components/metrics_reporter.py` lines ~688-700, ~898-900; `SGL/arg_groups/fields/observability.py` `decode_log_interval` = 40.

**Prefill throughput on this engine [M]** (`SP/wall_time_breakdown_output.txt`, median `input throughput` of ≥3,500-token prefill lines that directly follow another prefill line, i.e. inside a chunked burst):

| Run | Burst prefill rate (tok/s), depth < 16k | Samples |
|---|---|---|
| EVAL30 | 1,474 (16-48k: 1,501; 48-80k: 1,375) | 2,418 |
| EVAL6 | 1,548 | 126 |
| EVAL6R5 | 1,534 | 157 |
| EVAL12R5_LOW | 1,544 | 275 |
| G4B rev 6d | 1,578 | 91 |

**Time accounting over the solve window [M/D]** (`SP/wall_time_breakdown_output.txt`). Window = first branch `t0` to last branch `t1` from `branch_result.json`. "Prefill stall" was computed two independent ways: (a) the excess of each decode-log interval that contains prefill lines over the median clean (no-prefill) interval at the same running count and context bin; (b) logged prefill tokens ÷ burst prefill rate. They agree within 1-5 %.

| Run | Streams | Window (s) | Generated | Effective tok/s | Prefill stall (a) / (b) (s) | Prefill % of window | Decode time (s) | Decode-only rate (gen / decode time) [D] | Uncached prompt / generated |
|---|---|---|---|---|---|---|---|---|---|
| EVAL30 | 12 | 14,046 | 2,357,584 | 167.8 | 7,735 / 7,617 | **55.1 %** | 6,273 | 376 | **4.07** |
| EVAL6 | 6 | 3,744 | 1,217,904 | 325.3 | 499 / 504 | 13.3 % | 3,228 | 377 | 0.64 |
| EVAL6R5 | 6 | 3,464 | 846,080 | 244.2 | 645 / 633 | 18.6 % | 2,799 | 302 | 1.15 |
| EVAL12R5_LOW | 6 | 6,570 | 1,743,889 | 265.4 | 1,342 / 1,317 | 20.4 % | 5,207 | 335 | 1.17 |
| G4B rev 6d | 12 | 2,050 | 572,035 | 279.0 | 406 / 425 | 19.8 % | 1,622 | 353 | 1.17 |

- Note on EVAL30's headline: the earlier analysis computed ~187 tok/s as 2.36 M / 12.6 ks (`RES/E12 SGLang EVAL30 run1 analysis (2026-10-19).md`); my window (branch t0 → t1, 14,046 s) includes the 679 s deadline overrun documented in the fix list T8 section, hence 167.8.
- Logged prefill new tokens equal the usage-based uncached tokens exactly in EVAL6/R5/LOW/G4B (e.g. G4B 670,442 vs 669,980) — every prefill batch is logged. EVAL30 logged 11.23 M vs 9.60 M uncached in usage: the extra ~1.6 M are requests with no usage record (timed out / retracted at the deadline) [M].
- Decode tokens implied by decode lines (40 × running × accept len) match usage completions within 0.1-0.2 % except EVAL30 (+13 %, the same timed-out requests) [M].

**Decode rate by running count (clean intervals without prefill) [M]** (`SP/wall_time_breakdown_output.txt`):
- EVAL30 (mostly ≥40k context): 10 running 395, 11 running 422, 12 running 462 tok/s.
- G4B (20-40k context): 6 → 327, 8 → 357, 10 → 360, 11 → 397, 12 → 436 tok/s.
- 6-stream runs: 6 running 328-421 tok/s; 1 running 82-191 tok/s.
- The existing decode summary for EVAL30 gives 419.6 tok/s at max running and ≥60k context; G4B 370.9 at 11 running / 20-40k (`RUNS/E12 SGLang EVAL30 run1 (hosen42, 2026-10-19)/q27x/sglang_decode_summary.json`; `G4B/sglang_decode_summary.json`).

**Occupancy and queue states (time-weighted over decode-log intervals) [M]** (`SP/wall_time_breakdown_output.txt`):

| Run | Time < 12 running | < 11 | < 10 | < 6 | Queue share of time (0 / 1 / 2 / 3) | 11 running with queue ≥ 1 |
|---|---|---|---|---|---|---|
| EVAL30 | 67.8 % | 16.3 % | 1.6 % | 0.5 % | 34.3 / 54.0 / 11.4 / 0.3 % | 50.0 % |
| G4B rev 6d | 82.8 % | 47.1 % | 45.6 % | 11.8 % | 74.8 / 25.2 % | 25.2 % |
| EVAL6 / R5 / LOW (6 streams) | — | — | — | 20.5 / 24.6 / 24.4 % | queue 0 always | — |

- G4B: 27.7 % of the time at exactly 6 running — the false 12 → 10 → 8 → 6 probe step-down (events at t = 240/600/780 s at KV 0.13/0.26/0.31) documented in `RES/E12 rev 6d G4B GPU run analysis (2026-10-21).md` Finding 1; replaced in rev 6e by G4-1 (fixed 12 workers, KV gate 0.95) per `Experiment Review/E12 next fix list (running, 2026-10-20).md` lines 1473, 1546.
- "11 running + 1 queued" is a steady state at 12 in flight (EVAL30 50 % of time; G4B 25 % = all of the 12-in-flight time before the step-down). Cost at the same context: 11 vs 12 running = 422 vs 462 (EVAL30) and 397 vs 436 (G4B), i.e. ≈ −9 % [M/D]. Cause not established (the fix list calls it "the known 11 of 12 behaviour"; server max-running is 13 as a workaround).

**Idle and tails [M]**
- No long idle gaps (no interval > 60 s with ≤ 1 running and no prefill) in any run; time before the first / after the last decode line is 17-38 s per run (`T_long_idle_gaps_s` = 0, `no_decode_before_first_or_after_last_s`). So time with nothing in flight while tools execute is ≤ ~1 % of the window; per-stream tool time is hidden by the other streams. (Per-request client idle seconds are not in these runs' records; rev 6e G4-15 adds them — fix list line 1486.)
- Tail (last 10 % of the window): mean running 1.2 (EVAL6), 1.29 (EVAL6R5), 3.16 (LOW), 3.2 (G4B, stage 2 under the deadline), 10.15 (EVAL30, deadline-cut). In the 6-stream runs 7-8 % of the window ran at exactly 1 stream (82-191 tok/s) [M].

**Cache hit rate per request index (mean cached fraction; index 0 = branch's first request) [M]** (`SP/wall_time_breakdown_output.txt`):
- EVAL30: idx0 0.11, idx1 0.24, idx2 0.76, idx3 0.81, idx4 0.65, idx5 0.70, idx6 0.87, idx7 0.60, idx8 0.69, idx9 0.67, idx10+ 0.58 (n=185). Hit rate falls with depth into the run (KV thrash).
- G4B: 0.32, 0.57, 0.80, 0.67, 0.72, 0.86, 0.61, 0.53, 0.73, 0.78, 10+ 0.84.
- EVAL12R5_LOW: 0.32, 0.76, 0.77, 0.72, 0.74, 0.72, 0.80, 0.88, 0.89, 0.91, 10+ 0.91.
- EVAL6R5: 0.40, 0.20, 0.84, 0.84, 0.93, 0.91, 0.97, 0.97, 0.77, 0.80, 10+ 0.89.
- Token-weighted cache: EVAL30 59.6 %, EVAL6 78.8 %, EVAL6R5 84.3 %, LOW 88.7 %, G4B 75.0 % [M].

**Where the uncached tokens come from [M]** (`SP/prefix_reuse_detail_output.txt`, `SP/wall_time_breakdown_output.txt`). Classes for later requests: `full_reply` = hit ≥ previous prompt + 90 % of the previous reply; `prefill_end` = hit ≥ 95 % of the previous prompt but not into the reply; `partial` = some hit below 95 % of the previous prompt; `zero` = no hit (`cached` null; null = exactly 0 because SGLang omits `prompt_tokens_details` when 0 — fix list line 1395).

| Run | full_reply | prefill_end | partial | zero | Uncached in later <50 %-cached requests | Excess over unavoidable in hit requests (≈ re-prefilled previous replies) | First requests |
|---|---|---|---|---|---|---|---|
| EVAL30 (423 later) | 10.4 % | 59.1 % | 12.1 % | 18.4 % | **8,394,803** | 1,064,762 | 165,227 |
| EVAL6 (84) | 4.8 % | 84.5 % | 7.1 % | 3.6 % | 531,830 | 508,217 | 45,417 |
| EVAL6R5 (115) | 4.3 % | 91.3 % | 2.6 % | 1.7 % | 555,478 | 734,018 | 42,000 |
| EVAL12R5_LOW (425) | 17.4 % | 72.5 % | 7.5 % | 2.6 % | 1,076,913 | 1,035,817 | 86,628 |
| G4B rev 6d (164) | 6.7 % | 75.6 % | 9.1 % | 8.5 % | 386,168 | 245,462 | 34,049 |

(The two middle columns overlap for requests between 50 % and 95 %; they are not additive.)
- In the `prefill_end` class the hit stops a median **15-32 tokens before** the previous prompt's end (p10 −53 to −60 in EVAL30/G4B/LOW); 58-72 % of them fall within [−256, 0]. Most hits are multiples of 64 (e.g. G4B: 141 of 150 non-zero hits % 64 == 0; LOW 413 of 414), consistent with Mamba checkpoints on a 64-token grid — so the hit is the end-of-prefill checkpoint, and **the previous reply (reasoning + tool call) is prefilled again** [M] (`SP/reuse_offsets_output.txt`, `SP/prefix_reuse_detail_output.txt`).
- Median "reply reuse fraction" (cached − previous prompt) / previous completion is −0.003 to −0.009 in every run; p90 0.76-0.96 [M].
- When the reply *is* reused (`full_reply`), the hit ends 10-20 tokens short of the reply end and is not 64-aligned (e.g. G4B (prev completion, hit beyond prev prompt): (247, 234), (2008, 1880), (5585, 5457)), i.e. a decode-end Mamba state does get inserted and can be hit; so the mechanism works when the re-rendered history matches the generated tokens [M] (`SP/reuse_offsets_output.txt`).
- The chat template re-renders an assistant turn as `'<|im_start|>assistant\n<think>\n' + reasoning_content|trim + '\n</think>\n\n' + content` and rebuilds tool calls from parsed arguments (`<tool_call>\n<function=…>\n<parameter=…>\n…\n</parameter>\n</function>\n</tool_call>`); the generation prompt ends with `<|im_start|>assistant\n<think>\n` [M/S] (`D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/int4_chat_template.jinja`). Any whitespace trim or non-canonical sampled tokenization makes the re-tokenized history diverge from the generated token ids early in the reply.
- Daniel Franzen observed the same class of issue on ARC-AGI-3: "due to re-tokenization, the tokenized conversation might not exactly match the generated token sequence from before", which is why his patch keeps the end-of-prefill checkpoint [C] (`D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md` §2 "Retaining only useful Mamba checkpoints").

**KV / Mamba usage when evictions happen [M]**
- EVAL30: decode-line KV usage p50/p90/max = 0.77 / 0.95 / 1.00; mamba usage 0.61 / 0.66 / 0.71. Cold large prefills (≥3,500 new tokens, < 1,000 cached; 2,506 of 2,652 such lines) happened at KV p10/p50/p90 = 0.58 / 0.84 / 0.92 (the prefill line's own reading), mamba 0.54 / 0.59 / 0.66. KV at the end of `zero`-class requests: p50 0.83 / p90 0.94 (`SP/wall_time_breakdown_output.txt`, `SP/prefix_reuse_detail_output.txt`). Cause per the run analysis: 30 time-sliced branches × 50-100k context > the 866,744-token pool (`RES/E12 SGLang EVAL30 run1 analysis (2026-10-19).md` §2).
- The other runs never approached KV pressure: KV max 0.43-0.47 (6-stream runs), 0.44 (G4B). Yet G4B still had 14 zero-hit and 15 partial later requests (317k uncached tokens = 47 % of G4B's uncached) at KV p50 0.28 / 0.10 and mamba p50 0.61 (max 0.68) [M].
- "mamba usage" = slots held by running requests / 56 (EVAL30 line: `mamba num: 36, mamba usage: 0.64` at 12 running) — it does **not** count radix-cached checkpoints, so ≈ 20 of the 56 slots are all that is left for cached checkpoints of paused branches and branch points at 12 running [M/D] (`RUNS/E12 SGLang EVAL30 run1 (hosen42, 2026-10-19)/q27x/sglang_server_try0.log` line 260; arithmetic 56 − 36).
- Misses (hit < 95 % of previous prompt) are more frequent after a long previous reply: EVAL6 1/40 (prev reply < 2k) vs 7/26 (≥ 10k); LOW 15/260 vs 13/42; G4B 12/114 (< 2k) vs 14/39 (2-10k); EVAL30 30/192 vs 55/129 [M] (`SP/miss_vs_prevlen_output.txt`). Confounded with trims (long replies cause history trims), so this is correlation only.

### Inferences
- Cost model [D]: with ~1,500 tok/s prefill and ~400 tok/s decode at 11-12 running, one uncached prompt token costs ≈ 0.27 generated-token-equivalents of engine time. EVAL30's 4.07 uncached tokens per generated token therefore doubled the time per generated token; that alone explains "effective ≈ half of decode rate".
- On Franzen's RTX PRO 6000 build prefill was 10,553 tok/s (p90) vs 836.6 decode (ratio 0.08) [C, `dafr/WRITEUP.md` §2 table]; ours is 1,500 vs ~400 (ratio ~0.27-0.3). **Prefix reuse is worth ~3-4x more on 4x L4 than on his hardware.**
- After rev 6's admission rule, KV-driven eviction should not recur at 12 workers (G4B KV max 0.44); the remaining uncached work in G4B splits roughly into: re-prefilled previous replies ~37 % (245k of 670k), zero/partial hits at low KV ~47 % (317k), first requests ~5 % (34k), unavoidable tool-result deltas ~11 % (~74k = 319k hit-class uncached − 245k excess) [D].
- The 11-of-12 state and the probe step-down are occupancy problems, not cache problems; the probe is already removed in rev 6e.

### Gaps
- Exact cause of the reply re-prefill (template trim / re-tokenization vs a missing decode checkpoint) is not proven: our runs did not record token ids (`return_input_ids`/`return_output_ids` were False). A cheap test: one request with `return_output_ids`, then the next request's `input_ids` (via `/v1/chat/completions` with `return_input_ids` or offline tokenization of the same messages) → first divergent token index.
- Cause of zero/partial hits at low KV in G4B (mamba checkpoint LRU eviction vs trims vs SGLang #22935 split) is not separable from rev 6d data; rev 6e's G4-15 per-request telemetry (prefix kept, trimmed flag, idle seconds) and H7-13 classification are designed to separate them (fix list lines 1486, 2051).
- Per-request start times are not logged (only end time `t`), so per-stream tool-execution time could not be measured directly.
- The "11 running + queue 1" root cause is unknown.

---

## 2. SGLang HiCache (host-memory KV tier): does it exist in 0.5.21, does it support our hybrid GDN model, and what does it buy?

### Takeaway
HiCache exists in 0.5.21 and **is wired for FULL+MAMBA hybrids** (a host Mamba pool plus a draft-KV sidecar for non-MTP drafters such as DFlash). But the external record for hybrid GDN host tiers in Sept-Oct 2026 is a run of **correctness and capacity bugs** (wrong answers from host-tier hits on Flash-Next; zero host hits on Qwen3.8-27B once more than ~3 long prefixes are evicted, due to the small fixed host Mamba pool), and after rev 6's admission rule our device KV is not under pressure (KV max 0.44 in G4B). HiCache is therefore a **low-priority, high-risk** lever for us; it only pays if we deliberately keep more sessions than device KV holds (e.g. many paused stage-2 branches).

### Cited Findings
- **Flags present in 0.5.21 [S]:** `enable_hierarchical_cache` (default False), `hicache_host_memory_mode` (`cache` | `buffer_only`), `hicache_ratio` (default resolved to 2.0 in cache mode, 1.2 in buffer_only), `hicache_size` (GB, overrides ratio), `hicache_host_memory_fraction` (default 0.8 of available host memory after a 10 GiB reserve, used only when neither ratio nor size is set), `hicache_write_policy` (`write_back` | `write_through` (default) | `write_through_selective`), `hicache_io_backend` (`direct` | `kernel` (default) | `kernel_ascend`), `hicache_mem_layout`, `hicache_storage_backend` — `SGL/arg_groups/fields/memory.py` lines ~99-140; `SGL/arg_groups/hicache_hook.py` `handle_hicache_ratio_default`. Our server_args show `hicache_ratio: 2.0` resolved but HiCache off [M].
- **Hybrid support [S]:** `create_unified_radix_cache` builds the tree with components FULL + MAMBA for hybrid-SSM models and calls `cache.init_hicache(...)` when HiCache is on (`SGL/mem_cache/registry.py` ~151-205). `_MambaStrategy` matches `HybridLinearKVPool` with {FULL, MAMBA} and builds a "KV + MAMBA" host stack with host Mamba eviction (`SGL/mem_cache/hybrid_cache/hybrid_pool_assembler.py` lines 1785-1839). `MambaPoolHost` sizes the host Mamba pool as `device_capacity × host_to_device_ratio` or from `hicache_size` (`SGL/mem_cache/pool_host/mamba.py` lines 74-120). Our server log confirms the tree in use is `UnifiedRadixCache`, components (FULL, MAMBA), `RustUnifiedTreeCore`, `hicache_attached=False` [M] (`G4B/sglang_server_try0.log` 11:57:30).
- **Constraints found in 0.5.21 [S]:** `buffer_only` host mode rejects MAMBA trees ("supports only FULL/SWA unified trees") (`SGL/mem_cache/unified_radix_cache.py` lines 404-421); `--enable-int8-mamba-checkpoint` is rejected with HiCache ("host-offload path is not int8-aware") (`SGL/arg_groups/mamba_hook.py` lines 74-89); HiCache is mutually exclusive with `--disable-radix-cache`, LMCache and the external linker (`SGL/arg_groups/kv_cache_hook.py` ~393-440; `hicache_hook.py` ~27-40); HiCache disables the tc_piecewise prefill CUDA graph (`SGL/arg_groups/cuda_graph_hook.py` ~234-237; we run without a prefill graph anyway: launch cfg `no_prefill_graph: true` in `G4B/e12_run_report.json` → `engine.sglang.attempts[0].cfg`). Retraction backup to host is refused for Mamba models (`registry.py` ~158-160; `unified_radix_cache.py` `supports_retraction_backup` returns False for mamba).
- **Draft model with HiCache [S]:** non-MTP drafters get `HiCacheDraftMode.SIDECAR` (the draft KV pool is backed up as a sidecar) (`SGL/speculative/base_spec_worker.py` lines 253-293; `hybrid_pool_assembler.py` `build_hicache_draft_sidecars`). No hicache-specific Mamba ratio flag exists in 0.5.21 (grep for `hicache_mamba` finds nothing) [S].
- **External measured gains (not hybrid) [C]:** LMSYS HiCache launch post: up to 6× throughput and up to 80 % TTFT reduction; Novita AI coding agent (Qwen3-Coder-480B, 3FS backend): cache hit 40 % → 80 %, TTFT −56 %, throughput 2×; benchmarks on 8×H20/H800 with page size 64 and hicache-ratio 2; the post does not mention hybrid/Mamba models — [LMSYS blog 2025-09-10](https://lmsys.org/blog/2025-09-10-sglang-hicache). Best-practices doc recommends `--hicache-ratio 2 --hicache-size 0`, `page_first` with `kernel` io; host pool must be larger than the GPU pool under write_through; no hybrid/spec statements — [SGLang HiCache best practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices).
- **Hybrid GDN host-tier bugs [C]:**
  - [#39830](https://github.com/sgl-project/sglang/issues/39830) (open, Sep 16 2026, main 0.5.20.dev728, Qwen3.8-Flash-Next NVFP4, extra_buffer, track interval 64, mamba cache 48): host-tier hits return fluent but wrong answers, 0/10 correct vs 10/10 cold/device; root causes include Mamba side states (PLE) and the QSA compressed-K index not carried to host; fixes only partial/fork.
  - [#44735](https://github.com/sgl-project/sglang/issues/44735) (closed Oct 16 2026): v0.5.20 host-tier hits wrong on Flash-Next (0/2, 0/3); fixed in nightly `ed8c3139` via #39862 (Oct 4), #43596 (Oct 11), #40223 (Oct 12, "preserves all MTP KV and recurrent state through the host tier"). Not tested: non-SM120 GPUs, no_buffer.
  - [#45400](https://github.com/sgl-project/sglang/issues/45400) (open, Oct 20 2026, v0.5.22, **Qwen3.8-27B-NVFP4**, EAGLE, extra_buffer): host KV pages stay resident but re-asks get `cached_tokens: 0` once the evicted working set exceeds ~3 long (89k) prefixes; host Mamba pool 56 slots vs 702 Mamba backups; plain-LRU host Mamba eviction drops interior states; a proposed "evict pass-through checkpoints first" change restored 89k-token hits (25 s → 0.58-0.92 s). Answers stayed correct there.
  - [#36935](https://github.com/sgl-project/sglang/issues/36935) (open, 0.5.17, Qwen3.8-27B-FP8 with `--enable-hierarchical-cache --hicache-size 150`): enabling HiCache lowered the derived device Mamba pool from 165 to 147 slots (EAGLE: 70).
- **club-3090 discussion #1288 [C]:** 4× RTX 3090 (power-capped 220 W), 94 GiB RAM, SGLang 0.5.19, Qwen3.8-27B W8A8 + DFlash2 (8 draft tokens), TP4, `max_running_requests` 16, HiCache ratio 1.5 (L2 host tier). Reported prefill 4,000-4,300 tok/s at 10k, 3,029 at 90k; c=16 decode ~1,811 tok/s; a 12-turn agentic run's TTFT rose 93 ms → 2,446 ms at 35k tokens, which the author attributes to DeltaNet/SSM state not being prefix-cached. **No HiCache-off baseline and no hit rate are reported** — [club-3090 #1288](https://github.com/noonghunna/club-3090/issues/1288) (redirects to the discussion).
- **Host RAM fit [D]:** device cache pools are 6.62 + 2.06 GB KV and 1.04 GB Mamba state per GPU × 4 GPUs ≈ 34.7 GB KV + 4.2 GB Mamba (from G4B log). At `hicache_ratio` 2 the host tier would need ≈ 69 GB KV + ≈ 8 GB Mamba, within the 202.6 GB Kaggle host RAM (pinned memory; minus whatever weight loading / the sandbox use).
- **Multi-turn papers [C]:** CachedAttention (ATC'24): host/disk KV tiers with layer-wise preloading, TTFT −87 %, prefill throughput up to 7.8× for multi-turn — [arXiv 2403.19708](https://arxiv.org/pdf/2403.19708). Pensieve (EuroSys'25): GPU+CPU multi-tier conversation cache, 1.14-3.0× vLLM/TensorRT-LLM throughput — [arXiv 2312.05516](https://arxiv.org/html/2312.05516v2). Both are dense-attention systems.

### Inferences
- For our workload after rev 6 (12 fixed workers ≈ 12 resident sessions, KV gate 0.95, G4B KV max 0.44), device KV is not the binding store; the remaining misses are the reply re-prefill (a token-identity problem HiCache cannot fix) and Mamba-checkpoint availability (which HiCache's small, LRU host Mamba pool reproduces on the host side, per #45400).
- HiCache would become relevant only if the scheduler is changed to keep more live sessions than device KV holds (e.g. 24 paused stage-2 branches). In G4B, stage-2 resumes re-prefilled p50 4.4k / p90 24.7k tokens (`G4B/e12_run_report.json` rev6.metrics `stage2_resume_uncached_p50/p90`); at 1,500 tok/s that is ~3 s / ~16 s each [D] — small at 6 resumes per run, larger at 240-task scale.
- Any HiCache trial on our stack would need a correctness gate (host-hit vs cold answers at T=0, as in #39830/#44735) plus a host-hit telemetry check, and it cannot coexist with int8 Mamba checkpoints.

### Gaps
- Whether 0.5.21 includes the Oct 4-12 host-tier fixes (#39862, #43596, #40223) is not established; the local `pool_host/mamba.py` has no explicit side-state handling (grep), which only matters for PLE models (Qwen3.8-27B has none) — unverified for DFlash-sidecar correctness.
- No measured HiCache gain exists for a Qwen3.8-27B hybrid multi-turn agent on any hardware; club-3090 #1288 gives no off/on comparison.
- PCIe host↔GPU bandwidth on Kaggle L4 (load-back cost per token) was not found.

---

## 3. Mamba radix cache for hybrids: strategies, track interval, page size, LPM; what Franzen and rellik13 did

### Takeaway
For a hybrid model a prefix hit needs **both** the KV and a Mamba checkpoint at the exact match depth; KV residency alone does not guarantee a hit. Our strategy (`extra_buffer_lazy`, track interval 256, 56 Mamba slots, chunk 4096) gives end-of-prefill hits reliably but loses the previous reply most of the time, and only ~20 of 56 slots are free for cached checkpoints at 12 running. Franzen's two serving changes target exactly this: keep the useful checkpoints (final prefill checkpoint + one cold-start branch) and **refresh a checkpoint's LRU position when its long decode lock is released**; his harness admits only as many games as stay resident and trims history in big blocks. He reached 93.43 % token-weighted reuse. rellik13's lesson is the same in capacity terms: KV sat at 91-97 % and requests queued for cache space; spending freed KV (fp8) on history and fewer streams took him from 14.5 to 22.5.

### Cited Findings
- **0.5.21 strategy options [S]:** `--mamba-radix-cache-strategy` ∈ {auto, no_buffer, extra_buffer, extra_buffer_lazy}; `--mamba-track-interval` default 256 ("interval to track the mamba state during decode"); `--mamba-max-states-per-path` (-1 unlimited; caps interior states per root-to-tail path); `--enable-int8-mamba-checkpoint` ("~2x cached-prefix capacity at fixed memory"), `--int8-mamba-ckpt-size` (default 2× active pool); `--max-mamba-cache-size`; `--mamba-full-memory-ratio` (fallback 0.9) — `SGL/arg_groups/fields/exec_.py` ~340-415, `fields/schedule.py` ~205-220.
- **Validation rules [S]:** extra_buffer requires `mamba_track_interval ≥ speculative_num_draft_tokens` and `mamba_track_interval % page_size == 0`, and warns if `chunked_prefill_size < mamba_cache_chunk_size` (checkpoints may be skipped); extra_buffer_lazy is unsupported under PD disaggregation but wired for DFlash verify; no_buffer requires page_size 1 and `--disable-overlap-schedule` — `SGL/arg_groups/mamba_hook.py` lines 92-153.
- **Lazy mode semantics [S]:** each running request normally holds one ping-pong slot; at a track boundary a temporary second slot is allocated with **"No evict-retry: a transient slot is not worth evicting a cached checkpoint for; on failure tracking degrades in place"** (`SGL/managers/schedule_batch.py` `mamba_lazy_prealloc_at_boundary` ~3464-3493 and `mamba_lazy_spec_prepare` ~3495-3530). Decode checkpoints land on `lcm(mamba_checkpoint_grid(page), mamba_track_interval)` (`SGL/runtime_context.py` `mamba_track_grid` ~1919-1924).
- **General design [C]:** MambaRadixCache returns the deepest node whose Mamba value is not None; states are inserted after chunked prefill or decode; KV and Mamba states keep separate LRU lists and Mamba states can be evicted from any node; extra_buffer supports branching-position states at extra memory — [PyTorch blog "Hybrid Models Meet SGLang"](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/) (Dec 2025, v0.5.5). The LMSYS unified radix cache lets FULL/SWA/MAMBA components vote on the deepest accepted prefix boundary [C, secondary] — [AlphaSignal summary](https://alphasignal.ai/news/lmsys-rebuilds-sglang-s-cache-to-finally-support-hybrid-ai-models).
- **Known hybrid prefix-cache bugs [C]:** [#22935](https://github.com/sgl-project/sglang/issues/22935) (open; reproduces through UnifiedRadixCache): the `input_len − 1` lookup splits a node leaving the parent without a Mamba value → 0-hit; workaround extra_buffer (fixes identical prompts and short suffixes, not long divergent suffixes); PR #38625 under review (Qwen3.8-27B fp8 4,096-token prompts: cached 0 → 4,032, latency 1.49 → 0.35 s). [#36935](https://github.com/sgl-project/sglang/issues/36935) (open): interior checkpoints only at chunk boundaries; branch reuse collapses when kv_pool_tokens / chunked_prefill_size > max_mamba_cache_size (cps 2048: 0-token branch hits; bf16 SSM doubling slots 165 → 325 restored 32,768-token hits); `--mamba-track-interval` "is decode-side only" and did not help. Ours: 866,744 / 4,096 = 211.6 vs 56 slots (fix list line 1359), though our sampled mamba usage never exceeded 0.71 (fix list line 1397; usage excludes cached checkpoints, see §1).
- **Our measured strategy behaviour [M]:** end-of-prefill hits on a 64-token grid; reply reuse median ≈ 0; misses rise after long replies (§1 tables; `SP/reuse_offsets_output.txt`, `SP/miss_vs_prevlen_output.txt`).
- **Franzen (ARC-AGI-3 milestone 2, LB 27.89) [C]** — `D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md` (read in full) and `dafr/serving/build_bundle_pennyroyal.sh` header + patch 4 (lines 1-20, 290-320, 320-600):
  - KV-resident admission: "limited admission to ten game streams, matching the server's request limit, to keep their cached prefixes resident"; a game "keeps its admission slot across tool calls"; context eviction (trim) is "the natural handover point" for re-queueing (§2, §3 "Preserve prefix reuse while scheduling").
  - Checkpoint retention patch (v2.2): keep only the final prefill checkpoint plus one cold-start branch (end of system prompt) instead of intermediate prefill checkpoints, **and refresh the end-of-prefill checkpoint to MRU when its final device lock drops** (a prefill checkpoint "can stay locked for a long decode… make its age reflect release"); env `SGLANG_MAMBA_PREFILL_FINAL_ONLY` / `SGLANG_MAMBA_REFRESH_ON_UNLOCK`. The sparse-prefill part is gated `not params.enable_mamba_extra_buffer_lazy` (extra_buffer only); **the refresh-on-unlock part is "independent of sparse-prefill, including for extra_buffer_lazy"**. Patch states "validated on source/control-flow tests only; GPU integration testing is pending" (for v2.2 on Pennyroyal d00d88e).
  - Trim hysteresis: harness context 128 Ki, 12 Ki output reserve, trim target ≈ 57 Ki retained (`drain 59,392`), so the prefix stays reusable for many requests.
  - Measured (earlier demo run, slightly different config): **93.43 % of prompt tokens reused** over 5,132 requests; long-prompt (> 62k) cache-miss rate 0.03 % (1/3,248); prefill time fraction 15.25 %; p90 effective prefill 10,552.6 tok/s; p90 decode 836.6 tok/s; harness-side generation 609.0 tok/s (§2 table). Our P1 ARC-3 notes measured 590-617 tok/s for his build (fix list line 402).
  - Serving settings: 1× RTX PRO 6000, 10 streams, KV fp8, Mamba cache 60 slots, chunked prefill 8,192, mem fraction 0.96 (§1 table).
- **Milestone-2 comparison (Tong Hui Kang) [C]** — `D:/ARC AGI 2/_scratch/webresearch/d_744792.md` (read in full): dfranzen 10 streams (all 110 games live, KV 1,011,264 tokens, trims 118k → ~59k); lordhansolo vLLM with "align-state retention reusing Mamba state across turns" and "prompt-tail state caching", 14 streams, KV 1,417,100, trim ~126k → 81.5k; sirikilohit SGLang with **48 GB hierarchical KV + Mamba host cache**, 16 streams, 28 games live, KV 1,004,288, trim 57k → 45k (LB 22.53, the lowest of the three).
- **rellik13 [C]** (`d_744792.md` lines 93-101): "I thought the decode throughput was the bottleneck, so I kept the concurrency high. The logs showed me that the kv pool sat somewhere between 91-97% full and requests queued for cache space, so the memory was the actual limit… the fp8 kv was implemented yesterday and i spent the free memory on history (the trims moved from 37k->27k to 57k->45k). I guess this alone took the score from 14.5 to 22.5."
- **Page size / LPM [S/C]:** we run `page_size` 1 and `lpm` already (server_args). The HiCache benchmarks and most hybrid deployments in the issues use page size 64 (LMSYS blog; #39830; #44735). Our hits already land on a 64-grid (§1), so page 64 would cost ≤ 63 tokens per hit on the KV side; its benefit would be HiCache I/O efficiency, not hit rate [D]. SGLang #31954 (LPM can starve cold requests) is listed in our fix list as not verified (line 558).

### Inferences
- Our dominant hybrid-specific loss is not "too few checkpoints for branches" (#36935's regime — our pattern is append-only) but (a) the reply not being token-identical on the next turn and (b) probable LRU loss of the end-of-prefill checkpoint after long decodes (Franzen's exact mechanism; consistent with our miss-vs-reply-length correlation, not proven).
- `mamba_track_interval` 256 → 64 would only help the subset of `prefill_end`/partial requests whose divergence lies deep inside the reply (19-35 % of prefix-hit requests show a hit > 0 tokens into the reply, `SP/reuse_offsets_output.txt` "share > 0"); it costs more transient slot allocations, which in lazy mode fail silently when no slot is free. Low expected value until the divergence point is measured.
- Raising `max_mamba_cache_size` (56 → e.g. 96) or `--enable-int8-mamba-checkpoint` would enlarge the ~20-slot evictable checkpoint pool at 12 running. Each fp16 slot costs ≈ 1.00 GB / 56 ≈ 18 MB per GPU (ssm state, G4B log) and KV costs ≈ 8.68 GB / 866,744 ≈ 10 KB per token per GPU, so +40 slots ≈ −72k KV tokens [D] — affordable under the rev 6 per-stream window of 72,228 (`G4B/e12_run_report.json` `rev6_window`) only if the window is re-derived.

### Gaps
- Franzen's patch is written for Pennyroyal (d00d88e), not upstream 0.5.21; whether `reset_node_mru` and the lock-release hook exist with the same names in 0.5.21's Rust tree core was not checked.
- No measurement of how often the end-of-prefill checkpoint is evicted in our runs (no metric in 0.5.21 logs; #36935 requests one; a Modal-fork PR #38018 adds checkpoint-miss metrics per #36935 thread).

---

## 4. Eviction and admission policy: which fits agent workloads, and can SGLang 0.5.21 pin or prioritize a session's prefix?

### Takeaway
SGLang 0.5.21 has several policy knobs that are directly aimed at multi-turn agents — `priority` eviction, `tlru` (tail-optimized LRU for conversations), `slru`/`lfu`, `--enable-session-radix-cache` (session references: unreferenced entries evicted first), and **streaming sessions** (session KV + Mamba state held in a slot outside the radix tree, "invisible to eviction", append-only). None pins a prefix forever, but streaming sessions come closest and also remove the re-tokenization problem because turns are appended at the token level. The evidence that matters most, though, is that **admission control (sessions ≤ what KV holds) is the first-order lever** (EVAL30 vs G4B), and that is already implemented in rev 6.

### Cited Findings
- **Eviction policies [S/C]:** `--radix-eviction-policy` ∈ lru (default) | lfu | slru | priority | tlru; `--radix-eviction-policy-config` JSON; slru `protected_threshold` (default 2); tlru `threshold` and `next_prompt_estimate` (tokens; threshold must exceed the estimate; requires UnifiedRadixCache) — `SGL/arg_groups/fields/memory.py` lines 25-60; `SGL/arg_groups/kv_cache_hook.py` ~385-391; `SGL/mem_cache/registry.py` ~268-279. Docs: "A policy only scores; it cannot pin KV in memory"; `priority` evicts the prefix of the lowest-priority request (node keeps the max priority of requests reaching it; without priority scheduling every node is 0 → LRU); `tlru` protects only the part of a conversation's history its next prefill needs to stay within a TTFT budget, evicting the rest of the tail first — [SGLang radix eviction policy docs](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy) (tlru cites arXiv 2510.15152 in the flag help).
- **Session-aware cache [S]:** `--enable-session-radix-cache`: "Track per-session references on UnifiedRadixCache KV: eviction consumes unreferenced entries before referenced ones, and closing a session only dereferences its KV" (`memory.py` ~87-90; `registry.py` ~260-266 requires UnifiedRadixCache).
- **Streaming sessions [S/C]:** `--enable-streaming-session` wraps the tree in `StreamingSession`; `SessionSlot` holds the request's KV record between turns and the slot's Mamba pool entries (`session_held_mamba_slots`, `_free_slot_mamba`), so hybrid models are supported in 0.5.21 (`SGL/session/streaming_session.py` lines 1-80, 478-509, 620-621; `registry.py` ~281-288). Session params: `id`, `rid`, `offset`, `replace`, `drop_previous_output`; streaming sessions reject replace / offset / drop_previous_output ("append-only") (`SGL/managers/io_struct.py` `SessionParams` ~135-156; `SGL/session/session_controller.py` ~204-234). `session_params` is accepted on both `/generate` and the OpenAI chat request (`SGL/entrypoints/openai/protocol.py` ~380, ~937). NVIDIA Dynamo's agent page: "Session KV is invisible to eviction"; turn 1 can reuse the radix prefix, turns 2+ restore KV from the slot; idle timeout default 300 s resets on each request, so a long tool call can get the session reaped; requires SGLang ≥ 0.5.11; metrics `sglang:num_streaming_sessions`, `sglang:streaming_session_held_tokens` — [Dynamo "SGLang for Agentic Workloads"](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/backends/sg-lang/agents-on-sg-lang).
- **Priority scheduling [S/C]:** `--enable-priority-scheduling`, `--schedule-low-priority-values-first`, `--priority-scheduling-preemption-threshold`, `--retraction-policy length|priority` exist (`SGL/arg_groups/fields/schedule.py` ~82-132); ours are default (off). Dynamo also lists `speculative_prefill` (a `max_tokens=1` warm-up after a response) and sticky session routing (multi-worker only) [C, same Dynamo page].
- **Cache report [M/S]:** `--enable-cache-report` is on; `cached_tokens` is omitted when 0 (`usage_processor.py:13-15` per fix list line 1395), so null = 0.
- **Agent scheduling research [C]:** Continuum (ICLR 2026) keeps a tool-calling request's KV in GPU for a TTL predicted from tool-call duration distributions; SWE-Bench tool time mean 925 ms, BFCL 1,923 ms; delay −1.12× to −3.66×, throughput +1.10× to +3.22× over vLLM baselines — [arXiv 2511.02230](https://arxiv.org/html/2511.02230v2). Marconi (MLSys'25): for hybrid models, admit only SSM states with high reuse likelihood and use FLOP-aware eviction; up to 34.4× higher token hit rate vs prior prefix caching — [arXiv 2411.19379](https://arxiv.org/pdf/2411.19379).
- **Our admission evidence [M]:** EVAL30 (all 60 branches live, 30 time-sliced at 50-100k context) → 144 of 423 later requests < 50 % cached, 8.39 M re-prefill tokens, KV p90 0.95; G4B (12 fixed workers) → KV max 0.44 (§1). The rev 6e KV gate replay on 7 recorded logs holds new work only in EVAL30 (219 samples, 18.2 min) — fix list line 1473.

### Inferences
- With ≤ 12 sessions on an 866k pool and per-stream window 72k, plain LRU is adequate for **KV**; the policy that matters is for **Mamba checkpoints** (≈ 20 evictable slots), where "refresh on unlock" (Franzen) or Marconi-style selective admission address the right failure. `priority`/`tlru` act on KV leaves and would not by themselves protect a paused branch's Mamba checkpoint (not verified whether tlru scores Mamba components).
- Streaming sessions are the only 0.5.21 feature that both (1) keeps a session's KV + Mamba state out of eviction and (2) makes the next turn token-identical (server appends new input to the stored prompt + output ids). Caveats for us: append-only (a history trim must close and reopen the session), the 300 s idle reaper (our G4B stage-2 paused branches idle much longer), and the slot holds KV for every paused session, so it only fits with admission ≤ KV.

### Gaps
- No measured result of streaming sessions or `--enable-session-radix-cache` on a hybrid GDN model with DFlash was found.
- Whether the chat-completions path with `session_params` re-renders the template for the delta (and thus re-introduces the trim problem) was not traced in the source.

---

## 5. Agent-side design levers (no prompt or solver-logic change)

### Takeaway
Rev 6 already implements the two strongest harness levers (fixed 12 workers ≈ resident sessions, and no new task/branch while full-attention KV ≥ 0.95). What remains on the agent side is (1) making each turn's prompt token-identical to "previous prompt + generated tokens" (avoid template re-rendering differences), (2) keeping trims rare and large (hysteresis), and (3) reducing pause-resume churn of stage-2 branches whose checkpoints/prefixes age out.

### Cited Findings
- **Already in rev 6/6e [M]:** fixed 12 workers; KV admission gate on `sglang:full_token_usage` ≥ `E12_R6_KV_ADMIT` 0.95, poll 5 s, fail-open on missing metric (G4-1, fix list line 1473); resident history 10 × 79k ≤ 0.9 × KV planning rule and per-stream window (R6-S5, lines 793, 931, 1065; G4B report `rev6_window` 72,228); prefix invariants — deterministic facts/system prompt, state only appended at the tail, same tools/template kwargs every request, fake-server check "84 ok, 0 breaks" (R6-D2, lines 792, 1104); per-request telemetry of prefix kept / trimmed / idle seconds / KV / mamba (G4-15, line 1486) and zero-hit classification (H7-13, line 2051); the reasoning is preserved in history (`preserve_thinking: True` in server `default_chat_template_kwargs`, [M] server_args).
- **Trim behaviour [M]:** rev 4 EVAL6 had ~1 trim per branch (`RES/E12 SGLang EVAL6 run1 analysis (2026-10-20).md`); rev 5 ≤ 1-2 per branch; the 49k output reserve forced trims at ~82k context (`RES/E12 rev 5 runs analysis (2026-10-21).md`); G4B had 1 reasoning-only trim (fix list line 1446).
- **Franzen's harness levers [C]:** admission = server request limit; slot kept across tool calls; trim hysteresis (drop to ~57 Ki at once); calibrated token estimate; re-queue only at a trim (`dafr/WRITEUP.md` §2-3). Our token estimate is already calibrated (rev 5 B3: estimate vs server prompt tokens median +0.4 %, `RES/E12 rev 5 runs analysis (2026-10-21).md`).
- **Code-first lessons relevant to cache [M]:** never force a named `tool_choice` (HTTP 500 in Exp 4); the facts block lives in the system prompt (never trimmed) (`C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/code-first-duck-lessons.md`).

### Inferences
- **Token-identical continuation** is the largest remaining agent-side lever by our numbers (the reply re-prefill is 37-76 % of uncached tokens in the non-thrash runs; §1). Options, in increasing invasiveness: (a) verify with one offline test where the re-rendered assistant turn diverges from the generated ids (likely the `reasoning_content|trim`, the `'\n</think>\n\n'` join, or the rebuilt tool-call XML); (b) send the assistant turn back exactly as generated (e.g. as raw content including `<think>` … `</tool_call>` rather than parsed fields) if the template allows it without changing what the model sees; (c) token-level continuation via SGLang sessions (`/generate` with `input_ids` deltas or `session_params`), which changes the client transport but not prompts or solver logic.
- **Stage-2 resumes:** every pause/resume of a branch while other branches run risks its Mamba checkpoint being LRU-evicted (only ~20 evictable slots). Keeping a branch's slot until it is finished or trimmed (Franzen's rule) instead of time-slicing branches reduces this; the G4-17 minimum chunk (12,288) makes stage-2 rounds shorter and more frequent, which the fix list already flags as "more re-prefill" (line 1458).
- The 11-of-12 queue state wastes ≈ 9 % of decode throughput at 12 in flight; if it is caused by per-request Mamba slot reservation (4 slots per request, fix list line 1170) rather than KV, one more server slot or a larger Mamba pool may resolve it — unverified.

### Gaps
- The exact divergence point of the re-rendered history is unmeasured (see §1 gaps).
- Duck's history-trim rule interaction with streaming-session append-only semantics was not designed or tested.

---

## 6. Sizing the prize: recoverable effective throughput, levers ranked by evidence

### Takeaway
Measured decode-only rates are 353-376 tok/s at our running levels and 397-462 at 11-12 running. Re-prefill and under-occupancy are what separate effective throughput from that. **EVAL30-style workload: 168 (or the earlier 187) → ~330 tok/s by removing eviction re-prefill alone, ~370 by also removing reply re-prefill, ~410 with 12 running.** **G4B rev 6d: 279 → ~303 (probe fix) → ~325 (12 running) → ~357 (+ no reply re-prefill) → ~408 (+ no zero/partial misses).** The ceiling is the decode rate itself (~400-460 tok/s at 11-12 running on 4x L4), i.e. a 1.4-1.5× gain over G4B and ~2.2-2.4× over EVAL30.

### Cited Findings (all [D] from the measured inputs in §1; formula: time = generated / decode rate + uncached / 1,500-1,578 tok/s + edge time; effective = generated / time)

**G4B rev 6d** (generated 572,035; uncached 669,980; prefill rate 1,577; edges 22 s; measured 279.0 tok/s):

| Scenario | Decode rate | Uncached | Time (s) | Effective tok/s |
|---|---|---|---|---|
| Measured | 352.7 (gen / decode time) | 669,980 | 2,050 | 279 |
| S1: probe fixed, steady 11 running | 397 | 669,980 | 1,888 | ~303 |
| S2: 12 running (11-of-12 solved) | 436 | 669,980 | 1,759 | ~325 |
| S3: S2 + reply re-prefill removed (−245,462) | 436 | ~424,500 | 1,604 | ~357 |
| S4: S3 + zero/partial misses removed (−316,865) | 436 | ~107,700 | 1,402 | ~408 |
| Franzen-level 93.4 % reuse on G4B's 2.68 M prompt tokens, 12 running | 436 | ~176,900 | 1,446 | ~396 |

**EVAL30** (generated 2,357,584; uncached 9,597,852; prefill 1,475; edges 38 s; measured 167.8 over 14,046 s, 187 over 12.6 ks in the original analysis):

| Scenario | Decode rate | Uncached | Time (s) | Effective tok/s |
|---|---|---|---|---|
| Measured | 375.8 | 9,597,852 | 14,046 | 168 |
| E1: no eviction re-prefill (−8,394,803; what rev 6 admission targets) | 375.8 | ~1,203,000 | ~7,126 | ~331 |
| E2: E1 + reply re-prefill removed (−1,064,762) | 375.8 | ~138,000 | ~6,405 | ~368 |
| E3: E2 at 12 running, ≥ 60k context (419.6) | 419.6 | ~138,000 | ~5,750 | ~410 |

**Levers ranked by strength of evidence:**

| Rank | Lever | Evidence | Size (derived) | Status |
|---|---|---|---|---|
| 1 | Admit no more live sessions than KV holds (fixed workers + KV gate) | [M] EVAL30 vs G4B; KV-gate replay on 7 logs; rellik13 [C]; Franzen [C] | EVAL30: 168 → ~331 (+97 %) | Done in rev 6 / 6e |
| 2 | Keep occupancy at 11-12 running (probe removed) | [M] 13.3 % loss at the same context bin (fix list line 1465); 27.7 % of G4B at 6 running | G4B: 279 → ~303 (+9 %) | Done in rev 6e (G4-1) |
| 3 | Token-identical next-turn prompt (no reply re-prefill) | [M] 72-91 % of later requests stop at prefill end; reply reuse median ≈ 0; 37-76 % of uncached tokens; [C] Franzen notes re-tokenization | G4B S2 → S3: +10 %; EVAL6R5-type runs: ~76 % of prefill time | Not built; cause to confirm offline |
| 4 | Protect end-of-prefill Mamba checkpoints (MRU on unlock; larger / int8 checkpoint pool) | [M] G4B zero/partial misses at KV ≤ 0.44, correlated with long previous replies; [C] Franzen patch, #36935, #45400, Marconi | G4B S3 → S4: up to +14 % | Not built; needs a server patch or flag test |
| 5 | Solve the 11-running + 1-queued state | [M] 25-50 % of time; 11 vs 12 = −9 % | +9 % at 12 in flight | Cause unknown |
| 6 | Streaming sessions (KV + Mamba held per session, token append) | [S] supported for Mamba in 0.5.21; [C] Dynamo docs; no measured result for hybrids | Could cover levers 3+4 together | Untested; append-only and 300 s reaper caveats |
| 7 | HiCache host tier | [S] FULL+MAMBA supported; [C] open correctness / capacity bugs on hybrid GDN (#39830, #45400; #44735 fixed post-0.5.20); no hybrid gain measured | ≈ 0 while KV max ≤ 0.44 | Not recommended now |
| 8 | Prefill throughput itself (1.5k tok/s; chunk size, prefill CUDA graph) | [M] rate measured; [C] club-3090 4x3090 W8A8 reached 4.0-4.3k tok/s at 10k | Scales all re-prefill cost | Not researched here (engine scope) |

### Inferences
- At hidden scale (240 tasks, ~11.3 h window, 12 workers) the G4B-type profile is the relevant one. Going from 279 to ~360-400 tok/s is +29-43 % more generated tokens in the same window, before any solver change. The 240 × 60k = 14.4 M token target needs ≈ 370 tok/s over the 11.46 h window (fix list line 1403); levers 2 + 3 (or 2 + 5) reach that range on G4B's profile [D].
- These scenarios assume the removed prefill time converts fully into decode time at the stated rate; they ignore the context growth that higher reuse would allow and any acceptance-length changes.

### Gaps
- Scenario rates S2/E3 assume the clean 12-running rate is sustained; the 12-running clean samples are few in G4B (29 lines, early phase; `RES/E12 rev 6d G4B GPU run analysis (2026-10-21).md` Finding 1 correction).
- No run has yet measured rev 6e/6g (KV gate, no probe) on GPU; the rev 6g G4B log folder `RUNS/E12 SGLang EVAL12R6G4B rev6g run1 (esfersami50, 2026-10-22)/kernel_log.txt` contains only `[]`.
</content>
</invoke>
