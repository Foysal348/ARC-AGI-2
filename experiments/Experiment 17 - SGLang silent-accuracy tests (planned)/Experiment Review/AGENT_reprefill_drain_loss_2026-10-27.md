# Re-prefill, drain and coverage losses in the S1-family Poetiq runs (2026-10-27)

This is a read-only CPU analysis of saved Kaggle runs. No project files were changed.
Scripts and raw outputs are in `D:\ARC AGI 2\_scratch\e17_bughunt\agent_drain\`:
`analyze.py`, `windows.py`, `match.py`, `w3h_eos.py` and `final_numbers.py`, plus
`analysis_full.json`, `phase2_hit_miss.json`, `final_numbers.json` and `w3h_eos_rows.json`.

Runs analysed. All use vLLM 0.19, MTP 3, fp8 KV, prefix caching, mamba cache mode `align` and a 1,600-token block.

| run | effort / think cap | slots | budget | records |
|---|---|---|---|---|
| R5 EVAL120 | xhigh / 77,824 | 16 | 42,600 s | 172 |
| D66 HARD66 | xhigh / 77,824 | 16 | 43,020 s | 151 |
| L66 HARD66 | xhigh / 122,880 | 12 | 43,020 s | 123 |
| W3H EVAL120 | low / 60,000 | 16 | 10,800 s | 125 |
| E18 smoke (vLLM 0.32 server) | xhigh / 60,000 | 20 | 2,700 s | 20 |

## Method and how much to trust each number

- **Phase-2 prefix length.** It is computed as prompt + p1 + 25 stop tokens, or prompt + p1 for `continue`. As a check, the sum of phase-1 prompts plus phase-2 prefixes matches vLLM's `prompt_tokens` counter to within 21 tokens in every run. The 21 tokens are the engine smoke request. **This number is exact.**
- **Phase-2 cache-hit tokens** come from `prompt_tokens_by_source{local_cache_hit}`. Phase-1 prompts never hit the cache in any run: 0 hit tokens in every window that held only phase-1 admissions. In L66 those windows included 578k tokens of iteration-2+ prompts. So all cache hits belong to phase-2 prefixes. **This number is exact.**
- **Hit granularity.** A hit is always (floor(L/1600) - 1) x 1600 tokens, or zero. This was verified exactly in the drain windows: L66 hit 657,600 tokens on 12 prefixes and computed 29,466; W3H hit 320,000 on 13. In R5, 69 of 71 hit amounts decompose exactly into whole-prefix hits, so I saw no partial hits.
- **Unavoidable tail** is the tokens that must be recomputed even on a full hit: L - hitlen, about 1,600 to 3,200 per prefix. **Avoidable re-prefill** = computed phase-2 tokens - tail.
- **Prefill throughput** was measured from the drain windows, where decode is almost stopped:
  - R5: 166,738 tokens in 120 s and 165,699 in 123 s, so 1,350 to 1,390 tok/s.
  - D66: 1,530 to 1,700 tok/s.
  - vLLM 0.32 smoke: 474,613 computed tokens in about 330 s, so about 1,440 tok/s. Its log shows 2,300 to 2,600 in the bursts, alternating with 0.
  - Independent check: a regression of generation per window on computed prefill per window. Each prefill token costs 0.14 to 0.28 generated tokens, which at the run's generation rate is 1,410 to 1,770 tok/s.

  **I use 1,450 tok/s. All times below are estimates.**
- **R5 timing.** R5 records have no `t_admit`. I rebuilt done times from the Kaggle log commit lines, with offset spread 0.49 s, and used admit = done - `sec`. R5 also has no `p1_finish`. I classed its 15 units with p1 < cap that were admitted before the drain and finished after it as "drained (est)". This fits the 16 units active at the drain and the hit amount in the drain window.
- **Per-request hit or miss.** I matched window hit totals to subsets of nearby phase-2 events. These counts are estimates. The token totals above them are exact.
- R5 has no per-unit `test_correct`. For R5 I report train-pass and score only.

## 1. How phase 1 ended

| run | natural close | cap (`length`) | cap with `</think>` (`continue`) | EOS, no `</think>` | drained | drained, then hard stop |
|---|---|---|---|---|---|---|
| R5 (172) | 64 (37.2%) | 89 (51.7%) | 4 (2.3%) | 0 | 15 (8.7%, est) | 0 |
| D66 (151) | 34 (22.5%) | 101 (66.9%) | 1 (0.7%) | 0 | 1 (0.7%) | 14 (9.3%) |
| L66 (123) | 100 (81.3%) | 10 (8.1%) | 1 (0.8%) | 0 | 12 (9.8%) | 0 |
| W3H (125) | 19 (15.2%) | 0 | 0 | **93 (74.4%)** | 13 (10.4%) | 0 |
| E18 0.32 smoke (20) | 2 | 0 | 0 | 0 | 2 | 16 |

Phase-2 requests (forced close or `continue`) and their prefix lengths:

| run | phase-2 requests | cap prefixes, mean / median / max | drained prefixes, mean / median | total prefix tokens |
|---|---|---|---|---|
| R5 | 108 (104 forced + 4 continue) | 85,712 / 85,194 / 96,291 (n=89) | 46,884 / 47,429 (n=15) | 8,669,425 |
| D66 | 117 (116 forced + 1 continue) | 87,592 / 86,511 / 98,073 (n=101) | 54,513 / 57,537 (14 cut), 22,920 (1) | 9,722,769 |
| L66 | 23 (22 forced + 1 continue) | 130,186 / 129,942 / 134,578 (n=10) | 57,256 / 52,459 (n=12) | 2,120,408 |
| W3H | 106 (all forced) | EOS units: 38,432 / 39,057 / 67,077 (n=93) | 27,051 / 27,496 (n=13) | 3,925,852 |
| E18 smoke | 18 | - | about 29k | 524,192 |

Prefix-cache hit share for phase-2 prefills:

| run | hit tokens | hit fraction | computed phase-2 tokens | of which unavoidable tail | avoidable (full misses) | estimated full misses |
|---|---|---|---|---|---|---|
| R5 | 5,852,800 | **67.5%** | 2,816,646 | 261,425 | 2,555,221 | about 30 to 37 of 108 (matched: 69 hits / 37 misses / 2 unresolved) |
| D66 | 2,806,400 | **28.9%** | 6,896,354 | 276,369 | 6,619,985 | about 76 to 80 of 117 (matched: 31 / 76 / 10) |
| L66 | 2,065,600 | 97.4% (= 100% of cacheable) | 54,829 | 54,808 | 21 | 0 |
| W3H | 3,659,200 | 93.2% (= 100% of cacheable) | 266,673 | 265,052 | 1,621 | 0 (about 1) |
| E18 0.32 smoke | 49,600 | **9.5%** | 474,613 | about 46k | about 429k | about 16 of 18 |

Hits are all-or-nothing per prefix. A prefix either reuses everything except its last block, or nothing at all.

Misses go with KV pressure and with waves of phase-2 events at the same time:

| run | KV usage median, hits | KV usage median, misses | nearby phase-2 events (±200 s), hits | same, misses | preemptions |
|---|---|---|---|---|---|
| R5 | 0.789 | 0.807 (7 misses > 0.9, vs 1 hit) | 1.7 | 6.0 | 9 |
| D66 | 0.834 | 0.882 (25 misses > 0.9, vs 0 hits) | 1.9 | 3.0 | 66 |

The two runs that hit everything ran at lower KV pressure: L66 KV median 0.68 with 12 slots, W3H KV median 0.53. Both had 0 preemptions.

**The vLLM 0.32 smoke breaks the pure-pressure story.** Right after the drain abort, KV usage fell to 3.5%, and still about 16 of the 18 re-submitted prefixes missed. In vLLM 0.19, by contrast, 25 of 25 aborted-then-resubmitted prefixes hit in L66 and W3H.

What I read from this (hypothesis, not proven from source code):
- In align mode, a hit depends on one specific cached block per request, the mamba-state block. Once that block is evicted, the whole prefix misses.
- One missed prefix then holds the whole 4,096-token batch budget for about 60 s. Meanwhile the other re-queued prefixes wait, and their freed blocks get evicted. That is the cascade you see in the R5, D66 and smoke drains: the first 2 to 5 requests hit, then everything misses.
- The 0.32 server path needs its own check. On this evidence it loses the cache of aborted requests even with a nearly empty pool.

## 2. GPU time spent re-prefilling phase 2 (estimate at 1,450 tok/s)

| run | computed phase-2 re-prefill | avoidable part | if the cache never hit | phase-1 prompt prefill, for scale |
|---|---|---|---|---|
| R5 | 1,943 s = 32 min = **4.6% of 42,600 s** | 1,762 s = 29 min = **4.1%** | 5,979 s = 14.0% | 942 s |
| D66 | 4,756 s = 79 min = **11.1%** | 4,566 s = 76 min = **10.6%** | 6,705 s = 15.6% | 1,020 s |
| L66 | 38 s = 0.09% | 0 | 1,462 s = 3.4% | 817 s |
| W3H | 184 s = 1.7% | 1 s | 2,707 s = 25.1% | 572 s |
| E18 0.32 smoke | about 330 s = **12% of 2,700 s**, almost all at the drain | about 300 s | - | - |

This time is real lost generation, because prefill and decode run one after the other:
- In the drain windows, generation drops from about 280 to 300 tok/s down to 5 to 17 tok/s.
- The regression slope says each prefill second costs about 0.19 x 1,450 ≈ 276 generated tokens, which is roughly the whole generation rate.

What the avoidable time would have bought (estimate):
- R5: about 1,762 s x 281 tok/s ≈ 0.50M tokens, about 6 more 80k iterations.
- D66: about 4,566 s x 256 tok/s ≈ 1.17M tokens, about 14 iterations.

## 3. Final drain and hard stop

**R5.** Drain at t = 41,760; hard stop at 42,240; drain window 480 s. There were 16 active units: 15 in phase 1 were drained, and one (88e364bc) was already in phase 2 from the cap.
- The drain window shows 361,600 hit tokens (about 4 prefixes), then 431,360 computed tokens over the next about 360 s. Generation was 10 to 17 tok/s in two full windows.
- All 16 still produced code. The last one finished at 42,190, 50 s before the hard stop. **None were cut.**
- 4 drained units passed train: 9bbf930d, b10624e5, a251c730, a25697e4. a251c730 is a final eval hit. The other three are not.
- The drained tasks d8e07eb2, cb2d8a2c, 8b7bacbf and a6f40cea have final query hits. Their attempts came from non-passing candidates, and without per-unit test data I cannot attribute those hits to the drained iteration.
- The 15 drained units had been thinking for 205 to 4,056 s (median 1,932 s) and had produced 522k thinking tokens before the forced close.
- 7 units were admitted in the last 30 minutes before the drain. The last admission was at t = 41,555, which left 205 s of thinking.

**D66. Here the drain lost answers.** Drain at 42,420; hard stop at 42,900. There were 15 active units, all in iteration 3.
- 1 completed: 3a25b0d8, score 0, wrong.
- **14 were cut by the hard stop with no code.** Their p2 lengths were 737 to 1,527 tokens against an 8,192 cap.
- Re-prefill: 156,800 hit tokens (2 prefixes), then 609k computed tokens over the next about 360 s. At the last snapshot (42,853, 47 s before the stop), 15 were running and 0 waiting. So decoding only began in the last about 60 to 100 s.
- Tasks that lost their iteration-3 answer: 2d0172a1, 21897d95, 581f7754, 7b80bb43, 8b7bacbf, a32d8b75, b10624e5, abc82100, a25697e4, 88bcf3b4, de809cff, b6f77b65, 4c7dc4dd, 4e34c42c.
  - 581f7754 and 8b7bacbf still have one query hit from earlier candidates.
  - 4c7dc4dd and 4e34c42c had no valid program at all (best_score -1).
  - Several had near-miss best scores: 7b80bb43 0.992, 8b7bacbf 0.993, b10624e5 0.992, de809cff 0.984.
- Expected answers lost (estimate): between 0 and about 1.3 queries.
  - The low end uses the drained-unit train-pass rate in L66 and W3H: 0/25.
  - The high end uses R5's 4/15 train-pass rate times D66's 35% test precision.
  - The certain loss is 14 iterations of compute: 552k thinking tokens with no output.

**L66.** There were 12 active units, all drained, all full cache hits. The whole re-prefill fit in one window: 657,600 hit and 29,466 computed.
- All 12 produced code before the hard stop, and none were cut.
- 0 passed train. 135a2760 was test-correct without passing train, and it is a final hit.

**W3H.** 15 were active: 13 drained plus 2 EOS units already in phase 2. All hit the cache, all completed, and none were cut. 0 passed train, and there were 0 hits from them.

**E18 vLLM 0.32 smoke.** Drain margin was 360 s. 18 units were aborted at the drain.
- Re-prefill took about 330 s (06:38:42 to 06:44:12), with generation at 0.5 to 27 tok/s.
- **16 of 18 were cut by the hard stop with 601 to 1,091 p2 tokens and no code.** 2 completed with 438 and 912 tokens, both score 0.
- This is the D66 failure in an even stronger form.

## 4. Coverage

| run | tasks | got iteration 1 | iteration 2+ | iteration 3+ | first train-verify at iterations 1 / 2 / 3 | unverified tasks left in queue |
|---|---|---|---|---|---|---|
| R5 | 120 | 120 (none missed) | 52 | 0 | 55 / 16 / - | 49 (13 had only 1 iteration, 36 had 2) |
| D66 | 66 | 66 | 52 | 33 | 14 / 11 / 8 | 33 (8 at 2 iterations, 25 at 3) |
| L66 | 66 | 66 | 41 | 16 | 25 / 18 / - | 23 (7 at 2, 16 at 3) |
| W3H | 120 | 120 | 5 | 0 | 16 / - / - | 105 (100 had only 1 iteration) |

Eval scores for reference:
- R5: 78/172 queries, task-normalized 58.33/120 = 48.6%.
- D66: 28/103, 18.83/66 = 28.5%.
- L66: 34/103, 21.5/66 = 32.6%.
- W3H: 17/172, 13.17/120 = 11.0%.

Every task got iteration 1 in every run. The coverage loss is all in iteration 2 and later:
- R5 ended with 47 queued units; 13 unverified tasks never got a second iteration.
- In D66, 8 tasks were first solved at iteration 3. That shows later iterations pay, and that D66's lost ~76 min would have bought real iterations.

## 5. Answer quality: natural close vs forced close

| run / class | units | train-pass | mean train score | test all-correct | test correct given train-pass |
|---|---|---|---|---|---|
| R5 natural | 64 | 48 (75%) | 0.940 | n/a | n/a |
| R5 cap forced | 89 | 17 (19%) | 0.649 | n/a | n/a |
| R5 cap continue | 4 | 2 | 0.968 | n/a | n/a |
| R5 drained | 15 | 4 (27%) | 0.834 | n/a | n/a |
| D66 natural | 34 | 20 (59%) | 0.915 | 9 (26%) | 7/20 (35%) |
| D66 cap forced | 101 (100 with code) | 12 (12%) | 0.631 | 8 (8%) | 7/12 (58%) |
| L66 natural | 100 | 42 (42%) | 0.849 | 18 (18%) | 14/42 (33%) |
| L66 cap forced | 10 | 1 (10%) | 0.501 | 1 | 1/1 |
| L66 drained | 12 | 0 | 0.608 | 1 (135a2760) | - |
| W3H natural | 19 | 11 (58%) | 0.799 | 7 (37%) | 7/11 |
| W3H EOS-no-close | 93 (90 with code) | 5 (5.4%) | 0.735 | 4 (4.4%) | 1/5 |
| W3H drained | 13 | 0 | 0.658 | 0 | - |

Forced close cuts the yield to between a third and a sixth of natural close (train-pass 10 to 27% vs 42 to 75%).

When a forced unit does pass train, its test precision is not lower: D66 58% vs 35%, L66 1/1. W3H's EOS units are the exception at 1/5.

This is confounded by difficulty. A natural close is more likely on the easier tasks.

## 6. The W3H anomaly: 93 units hit EOS mid-thinking without `</think>`

- **All 93 have `p1_finish = stop`**: a real EOS token, not the length cap. xhigh never shows this, with the same engine, MTP and prefix caching: 0 of 446 phase-1 requests across R5, D66 and L66.
- **The text is cut mid-content, unlike a natural ending.**
  - 0 of 93 end with a newline; 11 end on sentence punctuation; 15 end inside a grid row. Examples: "Row 3: 9 9 9 8 8 8 9 3 9 9 8 8 8 8 8 9", "that doesn'", "The 3-shape's conc".
  - All 19 natural closes end with a newline, and 17 of them on punctuation.
- **It does not line up with block boundaries.** The absolute position (prompt + p1) mod 1600 is roughly uniform. 5 of 93 lie within 32 tokens of a 1,600 boundary, against 3.7 expected by chance; 14 lie within 100, against 11.6 expected.
- **It is not linked to prefix-cache reuse.**
  - The unit that ended this way never ran on a cache-restored prefix itself, because phase-1 prompts got 0 hit tokens in all runs.
  - The first EOS in the run (4c416de3, at about 1,178 s) came before any cache hit anywhere in the run. The first hit (19,200 tokens, in window 1141-1261) is exactly that unit's own phase-2 prefix, (floor(21,241/1600) - 1) x 1600.
  - Per-window EOS counts do not correlate with cache hits in the previous window (r = -0.10). The same-window r = 0.68 is the EOS causing its own phase-2 hit.
- **MTP cannot be measured per request.** The spec-decode counters are global only. MTP was also on in all the xhigh runs, which show 0 such EOS events.
- **Hazard grows with thinking length:**

  | thinking length | units still thinking | EOS | natural close |
  |---|---|---|---|
  | 0-10k | 125 | 0 | 5 |
  | 10-20k | 117 | 11 (9%) | 2 |
  | 20-30k | 99 | 28 (28%) | 8 |
  | 30-40k | 61 | 38 (62%) | 3 |
  | 40-50k | 17 | 13 (76%) | 0 |
  | 50-60k | 4 | 3 | 1 |

  The remaining units in each band were drained.

  The shortest EOS came at 13,357 tokens; the median at 31.7k.
- **Verdict.** It looks like model behaviour at low effort when thinking runs far past what low effort normally uses, a budget-style abrupt stop. It does not look like a cache or MTP fault.
- **Caveat.** The popped EOS id was not saved, and MTP cannot be split out per request. To settle it, log the finishing token id and the stop position next time.

## Key takeaways

1. **Re-prefill loss depends on KV pressure.**
   - At 16 slots with 77.8k thinking, phase-2 prefix reuse fails all-or-nothing in about 30% (R5) to about 68% (D66) of forced closes. That cost about 29 min (4.1%) and about 76 min (10.6%) of the 12 h runs (estimate).
   - At 12 slots with 123k thinking (L66), or with short contexts (W3H), reuse was perfect and the loss was about 0.
2. **The drain margin is too short whenever the re-prefill misses.**
   - Re-prefilling 15 to 18 prefixes at about 1,450 tok/s takes 330 to 420 s of a 360 to 480 s margin.
   - D66 lost 14 of 15 drain units, and the vLLM 0.32 smoke lost 16 of 18, with no code.
   - R5 survived with 50 s spare.
   - A safe margin is about sum(prompt + p1 of active units)/1,450 s + about 300 s of decode. For 16 units of about 50k each, that is about 850 s.
3. **The vLLM 0.32 server path missed the cache on about 16 of 18 aborted prefixes even with 96% of KV free.** Check this before relying on drain re-prefill on 0.32.
4. **Coverage.** All tasks got iteration 1 in every run. The loss is all in iterations 2 and later: R5 had 49 unverified tasks left in the queue and W3H had 100.
5. **The W3H EOS problem is low-effort model behaviour, not a cache or MTP fault.** The evidence is circumstantial, and the EOS token id was not logged.
