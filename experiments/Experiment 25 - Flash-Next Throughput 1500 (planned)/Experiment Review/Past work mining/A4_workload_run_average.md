# A4: workload, harness and scheduler throughput over the whole run (past-work mining for Exp 25)

**Date:** 2026-10-31. **Author:** Claude (A4 agent).
**Mode:** read-only, except this file. Nothing was pushed or run on Kaggle.

**Question.** For each past run, what was the whole-run average tok/s and how did it compare with the steady (full-concurrency) tok/s? What lowered the average: tail starvation, KV thrash, re-prefill, controller mistakes or drain losses? Which of our controllers and schedulers can be reused for a Flash-Next Duck run with 5-8 resident streams at 70-120k context?

**How the numbers were made**
- vLLM 0.19 runs: I recomputed the whole-span average, the average while running == max, utilisation, tail length and tail tok/s from every `vllm_metrics.json` / `poetiq_vllm_metrics.json` timeline. The script is `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\a4\tl.py`; it reads `vllm:generation_tokens`, `vllm:num_requests_running`, `vllm:kv_cache_usage_perc`, `vllm:num_preemptions` and the prefix counters.
- SGLang Duck runs: `sglang_decode_summary.json` (p50 gen tok/s at max running) plus `e12_run_report.json` (`b_summary.server_gen_tokens`, `g4b.window_sec`).
- vLLM Duck runs (E21): `e21_report.json` (`halves.*.server_gen_tokens`, `solve_start_sec`, `solve_finished_sec`, `gen_tok_s_p50`, `prefix_hit_pct_whole_run`, `recomputed_prompt_tokens`).
- **"Steady"** = tok/s while the server ran at its maximum running count. It still contains prefill stalls inside those intervals, so it is not the clean decode rate. Where a clean decode rate was measured separately, I give it too.
- **"Whole-run avg"** = generated tokens ÷ the solve window (first admit to last finish, or to the deadline).

---

## 1. Run-average table

Abbreviations: tn = task-normalised eval score; util = time-weighted mean running ÷ max running.

### 1a. Single-request methods (DIRECT / Poetiq), vLLM 0.19 in-process, FP8, MTP-3, fp8 KV (1,353,600 tokens)

| Experiment / run | Streams (max running) | Steady tok/s (at max running) | Whole-run avg tok/s | Avg ÷ steady | Util | Tail after the last full-concurrency moment | Tasks covered | LB / LB÷eval | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| E1 run 3 v1.3.5, EVAL12, prefix OFF | 12 | 298 (report: 307 "excluding stalls") | 247.8 | 0.83 | 95 % | 14 min at 137 tok/s | 12/12 | — | `Exp1/.../Q27X_run3/bundle/q27x/run_report.json` scheduler.tok_per_sec; README: forced close re-prefills ~52k, 13.8 % of wall time |
| E1 run 4 v1.4, EVAL12, prefix OFF | 16 | 344 (p50 366) | 248.9 | 0.72 | 73 % | 32 min at 137 tok/s | 22/22 | **LB 18.75** | `Q27X_run4/bundle/q27x/run_report.json`, `vllm_metrics.json` |
| E1 run 6 v1.4.1, prefix ON | 16 | 366 | 285.3 | 0.78 | 75 % | 26 min at 191 tok/s | 22/22 | — | `Q27X_run6_v141/q27x/run_report.json` (prefix_cache_hit_rate 0.786) |
| E1 run 7 v1.4.4 (admit to 11:34) | 16 | 357 | 274.6 | 0.77 | 75 % | 28 min at 172 tok/s | 22/22 | **LB 26.94**; LB÷eval at 49k ≈ 1.06-1.10 | `Q27X_run7_v144/q27x/run_report.json`; `resources/Analysis and Journals/Qwen3.8-27B LB and run deep analysis (2026-10-16).md` §3 |
| E6 dual v2, EVAL12 | 16 | 371 | 288.5 | 0.78 | 75 % | 26 min at 186 tok/s | 22/22 | — | `Exp6/.../GPU EVAL12 dual v2/q27x/run_report.json` |
| E7 ORDERING_ONLY (v1.4.4 + ORDER2) | 16 | no commit-run metrics in the folder | — | — | — | — | — | **LB 27.92** | `Exp7/README.md` status; LB deep analysis §1 |
| E8 R5 PROXY48 (scorer) | 16 | 355 | 355.3 | 1.00 | 100 % | none (56 units > 16 slots); **crashed** at the first scorer request (logits all-gather OOM), 15 in-flight units lost | 41/56 finished | not submitted | `Exp8/.../GPU PROXY48 E8 R5/q27x/run_report.json`; memory `arc-agi2-qwen27b-project.md` 2026-10-13 |
| E9 EVAL12 86k + tail resample | 16 | 356 | 306.6 | 0.86 | 88 % | 84 min at **282 tok/s** (the resamples kept the tail full) | 22/22 + 14 resamples | **LB 22.92**; LB÷eval **0.62** | `Exp9/.../GPU EVAL12 E9/q27x/run_report.json` (resample 14/14, gain 0) |
| E10 ORDER2 EVAL120 86k DIRECT | 16 | 346.5 | 333.3 | 0.96 | 95.8 % | 42 min at 200 tok/s (7.2 active); KV max 0.873, 0 preemptions | 172/172, all finished by 534 min | (its hidden twin E9: 0.62) | `Exp10/.../GPU EVAL120 ORDER2 86k DIRECT/q27x/run_report.json`; `Exp10/.../Analysis/GPU EVAL120 ORDER2 86k analysis (2026-10-14).md` §5 |
| E10 Poetiq R4 EVAL12 | 12 | 293.5 | 225.2 | 0.77 | 76.6 % | 108 min at 208 tok/s | 12 tasks | — | `poetiq_vllm_metrics.json` (R4) |
| E10 Poetiq R5 EVAL120, T 1.0 | 16 | 284 (p50 295) | 281.5 | 0.99 | 99.6 % | none; KV max **0.995**, **9 preemptions**, phase-2 prefix hit 67.5 % | 120/120 iteration 1; 52 iteration 2; 47 queued at the end | **LB 21.39**; LB÷eval **0.44** | R5 `poetiq_vllm_metrics.json`; `Exp17/.../AGENT_reprefill_drain_loss_2026-10-27.md` §1-4 |
| E10 W3H low effort, 3 h | 16 | 407 | 404.0 | 0.99 | 99.6 % | 6 min; KV max 0.64 | 120 iteration 1 | — | W3H `poetiq_vllm_metrics.json` |
| E10 D66 hard 77.8k | 16 | 264 (p50 286) | 256.0 | 0.97 | 99.2 % | KV max **0.999**, **66 preemptions**, phase-2 hit **28.9 %**, ~76 min avoidable re-prefill; the drain lost 14/15 units | 66 tasks | — | D66 metrics; Exp17 drain audit §2-3 |
| E10 L66 hard 122.9k | 12 | 254 | 254.3 | 1.00 | 100 % | 0 preemptions, phase-2 hit 97.4 % | 66 tasks | — | L66 metrics |
| S1-60K family (60k Poetiq + ORDER2 + KV gate 0.95) | 16 | (≈ R5 engine) | ~256 (memory) | — | — | hidden run: ~205-225 iteration-1 requests for 240 tasks | ~85-90 % hidden coverage (estimate) | **LB 34.58 / 31.94 / 31.81 (55k) / 32.08 (49k)**; E20 drain15 31.53 → true level ~32 | `Exp20/.../E20_LB_analysis_2026-10-29.md`; `Exp17/.../AGENT_S1_rerun_path_audit_2026-10-27.md` F1-F2 |
| E18 S1 on the vLLM 0.32 server, smoke, 20 streams | 20 | 382 (16.6-34.6 min) | 284 / 279.2 (scheduler) | 0.74 | ~100 % until the drain | drain re-prefill at 2.4-3.5k tok/s; prefix hit **7.7 %**; 16/18 drained units cut with no code | 20 tasks (smoke) | never submitted | `Exp18/.../E18_SMOKE_GPU_v1_2026-10-27.md`; `GPU_commit_R2_2026-10-29/e18_engine_stats.json` |

### 1b. Agent methods (multi-turn)

| Experiment / run | Engine | Streams | Steady tok/s | Whole-run avg tok/s | Avg ÷ steady | Main loss | Tasks covered | LB / LB÷eval | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| E4 Q27-AGENT v1.8.3, EVAL12 | vLLM 0.19 server | 16 (12 running peak) | — | 256.5 | — | prompt 2.78M vs generated 0.70M (short turns re-send history); KV peak 0.31 | 12 tasks, 0/22 | — | `Exp4/.../run4_v1_8_3.../eval_report.json` agent_gen_tok_per_s, server_metrics |
| E5 v1.9.1 long-think, EVAL12 | vLLM 0.19 server | 12 | — | 281 (712,767 / 2,537.6 s) | — | 12 tasks on 12 slots, no queue | 12 | — | `Exp5/.../GPU EVAL12 v1.9.1/agent_eval/eval_report.json`, `server_metrics.json` |
| **E12 Duck rev 1 EVAL30** | SGLang 0.5.21 INT4 + DFlash2 | 12 | 431 at max; clean decode 376-462 | **168-187** | **0.39** | **KV thrash:** 30 time-sliced branches × 48-98k context > the 866,744 pool; 8.39M re-prefill tokens; prefill **55 %** of the window; 144/423 later requests < 50 % cached; branch 1 never started in 29/30 tasks | 30 tasks, 7/48 | — | `Exp12/Experiment Results/E12 SGLang EVAL30 run1 analysis (2026-10-19).md`; `research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/agent_effective_throughput.md` §1, §6 |
| E12 rev 4 EVAL6 | same | 6 | decode-only 377 | 325 | 0.86 | prefill 13 %; tail: mean running 1.2 in the last 10 %; 7-8 % of the window at 1 stream (82-191 tok/s) | 6 tasks, 3/6 | — | same note §1 |
| E12 EVAL6R5 xhigh | same | 6 | decode-only 302 | 244 | 0.81 | prefill 18.6 %; uncached prompt 1.15 × generated | 6 tasks, 4/6 | — | `Exp12/.../E12 rev 5 runs analysis (2026-10-21).md` |
| E12 EVAL12R5_LOW | same | 6 | decode-only 335 | 265 | 0.79 | prefill 20.4 % | 12 tasks, 7/12 | **hidden rerun LB 17.50** (coverage ~31 %) | same; memory `arc-agi2-qwen27b-project.md` 2026-10-22 |
| E12 rev 6d G4B | same | 12 workers | 436 at 12 running | **279** | 0.64 | **false capacity step-down 12→10→8→6** (queue = 1 is SGLang's normal state); 27.7 % of the time at 6 running; uncached 670k > generated 572k | 12/12 exposed, 2/19 | — | `Exp12/.../E12 rev 6d G4B GPU run analysis (2026-10-21).md` Finding 1 |
| E14S Duck rev 6g, EVAL120 (other agent's scope, for the ratio) | SGLang width 6 | 12 | — | — | — | — | 120 | **LB 22.78** vs EVAL120 XH 25.8 % → LB÷eval ≈ 0.88 | memory `arc-agi2-needs-long-thinking.md` |
| E15 45K30K commit, EVAL20 | SGLang E15B (delayer off, mrr 17) | 16 | 451 at 16 | **386** (1,155,991 / 2,998 s) | 0.86 | short-run end tail (mean running 14.17 overall vs 15.6 while the client was full); cache 94 % | 20 tasks, 13/20 | — | `Exp15/Notes/11 ...md` engine table |
| E15 40K35K commit | same | 16 | 469 | 388 (1,184,773 / 3,054 s) | 0.83 | same | 20, 7/20 | submitted (57609813, LB not in the files) | Notes/11 |
| E15 50K45K R3H commit | same | 16 | 452 | ~378 (20 × 69.8k / 3,696 s) | 0.84 | same | 20, 12.5/20 | — | Notes/11 |
| **E15 R3K1 EVAL120 6 h** | same | 16 | 442 (p50 at 16) | **395** (8.30M / 21,010 s) | **0.89** | time-bound (queue left 114, no tail); running mean 15.65; cached 92.8 %; 54.2M prompt tokens for 8.3M generated | 120 attempt 1 + 79 attempt 2 | — | `Exp15/Experiment Results/Kaggle Runs/E15B EVAL120 50K45K R3K1.../q27x/e12_run_report.json` b_summary, g4b.window_sec; `scheduling rules (2026-10-25)/r3k_analysis_out.md` |
| **E15 R3K2 EVAL120 6 h** | same | 16 | 442 | **395** (8.27M / 20,945 s) | **0.89** | same; cached 93.4 % | 120 + 78 | 65/172, tn 39.03 % | same, R3K2 folder |
| E15P Poetiq S60 on the E15B engine, EVAL120 6 h | same | 15 | (15 running for 3,743/3,790 samples) | 389 (7.83M / 20,127 s) | — | 99/120 iteration-1 requests force-closed at 60k; only 18 reached iteration 2 | 120 iteration 1 | 64/172, tn 39.31 % | `Exp15/Notes/20 ...md` |
| **E21 NB1 half A** (vLLM 0.19 FP8, SSM fp16) | vLLM 0.19 server | 18 | 329 (p50 at 18) | **311** (6,108,284 / 19,626 s) | 0.95 | prefix hit 88 %; recomputed 5.07M | 120 attempt 1 (done at 4.77 h) + 34 attempt 2 | 54/172, tn 34.31 % | `_scratch/e21_sami_logs/out_q27-xhigh-e21-duckb-vllm-nb1-int4-s20/q27x/e21_report.json` halves.A |
| **E21 NB1 half B** (vLLM 0.32 INT4) | vLLM 0.32 server | 20 | 252 | **231** (4,530,102 / 19,623 s) | 0.92 | prefix hit **59.6 %**; recomputed **13.2M** (2.6 × half A); waiting up to 17 | 114/120 attempt 1 (4 cut by the deadline), **0 attempt 2** | 43/172, tn 28.19 % | same file, halves.B |
| E21 NB2 half A (vLLM 0.19 FP8, SSM fp16) | vLLM 0.19 | 16 | 327 | **312** (6,119,115 / 19,626 s) | 0.95 | prefix hit 87.8 %; recomputed 4.52M | 120 + 31 attempt 2 | 52/172, tn 33.33 % | `_scratch/e21_sami_logs/out_...nb2.../Q27X_E21_NB2_BUNDLE.zip:q27x/e21_report.json` |
| E21 NB2 half B (vLLM 0.32 FP8, SSM fp16) | vLLM 0.32 | 16 | 278 | **247** (4,848,135 / 19,622 s) | 0.89 | prefix hit 68.5 %; recomputed 9.82M | 118/120 attempt 1 (2 cut), 4 attempt 2 | 53/172, tn 33.19 % | same zip |

**Reference point outside our runs:** one public team reports 980 peak and 710 average on 4×L4, a ratio of 0.72 (Kaggle thread 750546, cited in `PLAN_1500.md` §1).

### 1c. Reading the table

The average ÷ steady ratio has three regimes:

1. **Time-bound long runs with a full queue: 0.89-0.99.**
   - Examples: E10 ORDER2 0.96, R5 0.99, D66 0.97, R3K 0.89, E21 0.89-0.95.
   - The queue never empties, so the tail is short or absent.
   - The remaining gap is prefill and recompute inside the "steady" intervals.
   - But "steady" is itself depressed by re-prefill. R5's 284 at 16 running is 19 % below clean DIRECT at 16 (351, LB deep analysis §5). E12 EVAL30's 168 effective is 39 % of its own steady and 45 % of its clean decode.
2. **Small subsets with more units than slots, but not many more: 0.72-0.86.**
   - Examples: the EVAL12 runs (22 units on 16 slots) 0.72-0.78; the E15 EVAL20 commits 0.83-0.86.
   - The tail runs at 137-208 tok/s for 14-108 min.
   - This is the "tail starved of concurrency" pattern of the 980/710 team.
3. **Controller or cache failure: 0.39-0.74.**
   - E12 EVAL30 thrash 0.39; G4B false step-down 0.64; E18 0.32 drain re-prefill 0.74.

---

## 2. GOOD: what raised the run average or coverage

| # | Lever | Measured effect | Evidence |
|---|---|---|---|
| G1 | **Prefix caching on**, so a forced close no longer re-prefills ~50k | run 4 → run 6: whole-run 248.9 → 285.3 tok/s (+14.6 %). LB v1.4 18.75 → v1.4.4 26.94 (+8.2, the only real LB jump we have; bundled with the later admit) | run 3/4/6 `run_report.json`; LB deep analysis §2-3 |
| G2 | **Admit until the end of the clock** (11:34; S1: 11:45) | part of the +8.2 above; "queries finished in 12 h = score" | LB deep analysis §2 |
| G3 | **Shorter per-request cap for coverage** (Poetiq 77.8k → 60k) | R5 LB 21.39 → S1-60K 34.58 (later draws 31.5-32.1, so the true level is ~32). Simulated ~208/240 hidden tasks reach iteration 1 at 60k vs ~177 at 77.8k | memory 2026-10-16/17; E20 LB analysis §1 |
| G4 | **ORDER2 easy-first ordering** | +4-7 vs random under a 50-70 % budget; replay +3 points at a 4-6 h clock; 0 when every unit finishes; ORDER2 beats 100 % of 1,000 random orders at 120 min | `Exp10/.../GPU EVAL120 ORDER2 86k analysis` §8; `Poetiq R5 with ORDER2 - counterfactual replay (2026-10-15).md` §2 |
| G5 | **Keep the tail full with more work** | E9: the resamples kept the tail at 282 tok/s for 84 min, against 172-191 tok/s tails in the other EVAL12 runs; the whole-run average was 306.6 vs 274.6 (run 7). **But** the resamples gave 0 hits on EVAL12-hard units: filling the tail raises tok/s, and its value depends on what fills it | `Exp9/.../GPU EVAL12 E9 result (2026-10-13).md` |
| G6 | **Fixed workers = what KV holds, plus one KV gate** (no new task while KV ≥ 0.95; rev 6e G4-1) | Derived: EVAL30 168 → ~331 tok/s without eviction re-prefill. G4B KV max 0.44; the R3K runs had 0 KV holds, 0 retractions and running mean 15.65/16 | agent_effective_throughput.md §6 rank 1; R3K `e12_run_report.json` rev6_scheduler.kv_holds 0 |
| G7 | **Real 16 streams on SGLang** (delayer off, server max-running 17 for 16 client streams: the 11-of-12 workaround) | R3K: 16 running for 3,109-3,173 of 3,444-3,462 decode lines; 442 tok/s p50 at 16 vs 11-12 in E12 | `Exp15/.../R3K*/q27x/sglang_decode_summary.json` running_hist |
| G8 | **Stage-1-first scheduling** (attempt 1 on every task before any attempt 2) | Every task exposed: G4B 12/12; R3K 120/120 attempt 1; E21 half A attempt 1 done at 4.77-4.84 h of 5.45 h | Notes/12; E21 reports stage1.attempt1_all_done_h |
| G9 | **Independent second attempts** (complementary: D66∪L66 50 vs 28/34; R3K1∪R3K2 80 vs 50/65 queries) | Notes/12 A5: a 3rd attempt on an unsolved task is worth 0.12-0.16 tasks per attempt, 2-3 × an attempt 2 on an already-accepted task (0.055). This makes extra attempts the natural tail filler | `Exp15/Notes/12 ...md` §A4-A5; r3k_analysis_out.md overlap |
| G10 | **Low KV pressure keeps the prefix cache whole** | L66 (12 slots, KV median 0.68) and W3H (KV median 0.53): phase-2 reuse 97-100 % of cacheable; R5/D66 at 16 slots, KV 0.99: 67.5 % / 28.9 % | Exp17 drain audit §1 |
| G11 | **SSM state fp16 on vLLM 0.19** (block 832 instead of 1,600) | 19 streams fit at 57k; E21 halves A at 88 % prefix hit, 0 preemptions | memory `experiment-21-duckb-vllm.md`; E21 capacity blocks |
| G12 | **Deadline-clamped requests instead of an abort drain** (Duck R6-S6: every request clamped to the deadline, no item starts with < min_admit_sec left) | E21 halves: solve overrun 0.8-1.2 s, 0 stragglers; R3K: verify_skipped_deadline 0 | E21 report `solve_overrun_sec`, `stragglers`; `arc2_duck.py:3220-3250` docstring |
| G13 | **Weight prefetch** | weight load 64 s → 4.8 s; engine start 600 → 532 s (more solve time) | memory `d66-l66-hard-task-findings.md` lesson 7 |

---

## 3. BAD: what lowered the run average

| # | Loss | Measured size | Evidence |
|---|---|---|---|
| B1 | **Tail starvation on small subsets** | EVAL12 tails of 26-32 min at 137-191 tok/s: whole-run 72-78 % of steady. R4 Poetiq: 108 min at 208. E15 EVAL20: 0.83-0.86. 6-stream Duck: mean running 1.2-1.3 in the last 10 %, 7-8 % of the window at 1 stream | §1 tables; agent_effective_throughput.md §1 "Idle and tails" |
| B2 | **Tail after the last admit on full runs** | ORDER2: 44-min tail at 7.2 active (≈ 200 tok/s), 34 min with running < 12. S1 hidden: ~20-35 tasks never admitted (identity) | ORDER2 analysis §5; AGENT_S1 audit F2 |
| B3 | **KV thrash from over-admitting resident agent contexts** | E12 EVAL30: 168-187 tok/s vs 376-462 decode; 8.39M re-prefill tokens; prefill 55 % of the window. R5/D66 (single-request but KV 0.995-0.999): 9 / 66 preemptions; D66 phase-2 reuse 28.9 % and ~76 min avoidable re-prefill (10.6 % of 12 h) | E12 EVAL30 analysis §2; Exp17 drain audit §2 |
| B4 | **Re-prefill of the previous reply on every Duck turn** (the hit stops at the end of the previous prompt) | SGLang: 72-91 % of later requests; 37-76 % of uncached tokens; prefill 13-20 % of the window even without thrash. On vLLM 0.32 align: Duck prefix hit 59.6-68.5 % vs 87.8-88 % on 0.19 with the same harness, so recomputed tokens 9.8-13.2M vs 4.5-5.1M per 5.45-h half, and generated tokens −21 % (FP8 16 streams 4.85M vs 6.12M) | agent_effective_throughput.md §1, §3; E21 reports |
| B5 | **Abort-and-resubmit drains** (force-close = new request = re-prefill prompt + thinking) | D66: 15 drained, 14 cut at the hard stop with no code; E18 0.32: 16/18 cut, and 0.32 missed the cache on ~16/18 aborted prefixes with 96 % of KV free. Drain re-prefill runs at ~1,450 tok/s while generation falls to 5-17 tok/s | Exp17 drain audit §3; E18 smoke "Finding" |
| B6 | **The controller stepped down on a normal signal** (rev 6d probe treated queue = 1 as pressure) | 12 → 6 within 13 min; 27.7 % of G4B at 6 running; −13 % tok/s at the same context bin. The test fixture used queue = 0, which I had made up; the real logs showed queue = 1 in 2,615/2,816 lines | G4B analysis Finding 1; memory `verify-real-concurrency-in-benchmarks.md` |
| B7 | **The server ran fewer requests than requested** (SGLang 11-of-12: a chunked-prefill request counted twice in admission) | 11 vs 12 running ≈ −9 %; 25-50 % of the time in EVAL30/G4B | agent_effective_throughput.md §1 occupancy table; memory 2026-10-18 rev 3c |
| B8 | **No per-branch cap / unfair start** | E12 EVAL30: branch 1 never started in 29/30 tasks; branch 0 ran 61-124k with 0 submits | E12 EVAL30 analysis §1 |
| B9 | **Fixed lock-step waves** (PIV v1.3) | 29 % of wall time wasted; LB 1.67 | LB deep analysis §8-9 |
| B10 | **Side workload OOM killed the engine** (E8 scorer `prompt_logprobs` all-gather needed 790 MiB with ~0.4 GB free) | 15 in-flight units (445k tokens) lost at minute ~124 | memory 2026-10-13; LB deep analysis §8 item 8 |
| B11 | **More streams at long context bought nothing on 27B** | D66 16 slots 256 tok/s with 66 preemptions vs L66 12 slots 254; E21 NB1 A 18 streams 6.108M generated vs NB2 A 16 streams 6.119M (equal tok/s, equal score) | §1 tables; `d66-l66-hard-task-findings.md` lesson 6 |
| B12 | **Commit turns with `enable_thinking: false` change the system header**, so the cache misses (~45-55k prefill per commit turn, ~290 per hidden run) | not built; open F2 | Notes/11 F2 |
| B13 | **Time-sliced pause/resume of branches** (the 28k pause + resumes) | 58-66 % of all prefill in the G4B EVAL120 runs; pause-resume is also negative for accuracy | `Exp15/README.md` "Why"; memory `coverage-levers-early-stop-ordering.md` |
| B14 | **Early-stop of hopeless thinking does not work** | best text signal AUC 0.63, saves 3-4 % of tokens, loses 18 % of train passes out of sample | `resources/Analysis and Journals/Coverage levers - hopeless thinking and task order (2026-10-17).md` |

---

## 4. Controller and scheduler designs we already have (code paths)

| Design | What it does | Code path | Ran on GPU? | Reuse for Flash-Next |
|---|---|---|---|---|
| **Duck rev 6 / E15B scheduler `solve_all_r6`** | Stage 1 = attempt 1 of every task in ORDER2 + cells order. Stage 2 by `stage2_class` (0 partial, 1 nothing, 2 one accept, 3 two+), round robin by rounds. Optional stage 3 (`extra_attempts`, `max_attempts`) and `b1_accepted_mode` before3/after3. Fixed workers. KV gate (`kv_admit_limit` 0.95, poll 5 s, stale 30 s, fail-open counted). `min_admit_sec`. Every request deadline-clamped (R6-S6). Fair share off in B mode | `experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/duck_b/arc2_duck.py:3220` (docstring 3233-3250; gate 3422-3428; min_admit 3454; `stage2_cutoff_hit` 3189; `stage2_class` 3197; `share_tokens` 3302-3315; `rank` 3291-3300) | yes: E15 commits, R3K1/R3K2, E21 (all four halves) | **Yes, first choice.** Change `workers` to the Merlin resident capacity, point the gate at vLLM `kv_cache_usage_perc`, keep stage 3 as the tail filler |
| **vLLM pre-run capacity check `e21_capacity`** | Parses "Maximum concurrency for M tokens: X" and the block size; num_blocks = X × (cdiv(M,B) + F) with F = 3 × (2 + k) Mamba blocks; streams = min(seqs, num_blocks // (cdiv(57k,B) + F)); refuses < 8 streams | `experiments/Experiment 21 - Duck config B on vLLM 0.19 + 0.32 (planned)/Experiment Notebooks/Build/e21_engine_new.py:202-235` (`E21_SIZING_TOKENS` 57000, `E21_MIN_STREAMS` 8) | yes (E21 halves; it matched real KV) | **Yes**, with F and the block size taken from Merlin's FN log (block 4096; F unknown for FN's GDN groups) and the floor lowered from 8 |
| **E21 stall watchdog** | No token progress for 600 s while requests run → relaunch; cross-half relaunch race guard | `.../Experiment 21 .../Build/build_e21.py:339-366` (`_e21_progress`) | yes (0 stalls) | Yes |
| **E21 content probes + contract probe** | Garbage / wrong-answer probes ("!!!!" would fail); a turn-2 prompt-growth check catches dropped reasoning | `e21_engine_new.py:142-145` and `e21_contract_probe` (~244-300) | yes | Yes: FN produced "!!!!" in E13 |
| **S1 KV admission gate** (single-request) | Admit while KV ≤ 0.95 once ≥ 10 are active; fail-open on a missing metric | `experiments/Experiment 10 - .../Experiment Notebooks/Build/build_e10_s1.py:245-270` (`e10_kv_usage`, `e10_kv_gate_open`, admission while-loop) | yes (S1 LB runs) | Pattern only |
| **E15P gate variant** (fail-closed on a missing metric; max(KV, mamba)) | Same as S1, but holds new admissions when the metric is missing | `experiments/Experiment 15 - .../Experiment Notebooks/Build/build_e15p.py:300-345` | yes (E15P, 0 blocks) | Pattern only |
| **E7 KV-projection controller** (shelved) | Predicted usage = a·n + b·Σlive, self-calibrated; admits while the projected peak (all active units at expected final length, p90 of finished units) < 0.90; preemption → cooldown + tighter target; 2 preemptions / 30 min → extras off | `experiments/Experiment 7 - .../Experiment Notebooks/Build/E7B_build/build_e7b.py:26-62, 199-264` (constants, breaker, ceiling); tests `tests/test_e7_v3.py` | **never** (shelved 2026-10-13) | Idea only: the projected-peak rule is the right shape for 70-120k reasoning that grows; must be replayed on real FN metric series first |
| **E18 vLLM server shim** | In-process `add_request/step/abort_request/get_metrics` API over the OpenAI server; abort closes the stream | `experiments/Experiment 18 - .../Experiment Notebooks/Build/e18_engine_cell.py:213-420` | yes | Only if a non-Duck (Poetiq/DIRECT) FN arm is built |
| **ORDER2 ordering** | NVARC `get_puzzle_priority` (tier, fine), stable | `resources/Puzzle Ordering (NVARC 32.22 ref)/puzzle_order.py`; injection `Exp7/.../E7B_build/build_order_only.py:35` | yes (E7, S1, Duck) | Yes, unchanged |
| **Replay and simulation tools** | Time-sliced replay at 240-task scale, budget grid, skip policy | `Exp15/Experiment Results/Analysis/scheduling rules (2026-10-25)/time_slice_sim.py`, `budget_grid.py`, `replay_skip_policy.py`, `r3k_analysis.py` | offline | **Yes:** re-run with FN per-stream rates before any 12-h run |
| KV-gate replay on recorded logs | Replays the gate rule on server logs | `_scratch/e12_rev6_results/.../analysis/kv_queue_scan.py` (cited in the G4B analysis) | offline | Yes: replay any FN controller on real FN logs (memory rule) |
| Drain and re-prefill analysis | Phase-2 hit and miss, drain timing | `_scratch/e17_bughunt/agent_drain/` (`analyze.py`, `windows.py`, `final_numbers.py`) | offline | Yes, for any FN forced-close path |
| Wall-time breakdown (prefill stall, occupancy, reuse classes) | | `research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/` SP scripts cited in `agent_effective_throughput.md` §1 | offline | Yes: the standard post-run accounting |
| Timeline analyzer (this report) | Whole-run vs steady, util, tail | `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\a4\tl.py` | offline | Yes, on FN `/metrics` snapshots |

**Merlin's own FN serving flags that matter for the workload** (read in `_scratch/rc2_final_bundle/merlin/launch.py`, `_build_flash_next` 292-480):
- `--enable-prefix-caching --mamba-cache-mode align --enable-prompt-tokens-details` (461);
- `--block-size` default 4096 (334);
- `--use-replayssm --prefix-match-unit 64` (474-475);
- an optional `--long-prefill-token-threshold` (463-464);
- MTP `draft_sample_method: probabilistic` (444);
- the Froggeric chat template, "which keeps a coding agent's per-turn notes in place … [otherwise] breaks the prefix cache" (388-393);
- Mamba checkpoint retention "always on" (retired knob, 116).

These are exactly the knobs behind B4/B5. Whether replay-SSM plus 64-token match units fixes the vLLM 0.32 align reply re-prefill is **UNVERIFIED** (see §6).

---

## 5. Transfer to a Flash-Next Duck run (5-8 resident streams at 70-120k)

### 5.1 The budget

- `PLAN_1500.md` §1: 1500 tok/s × 41,400 s ≈ 62.1M tokens.
- Our best whole-run ÷ steady ratios on time-bound Duck runs are 0.89 (SGLang, 93 % cache) and 0.89-0.95 (vLLM).
- **To average 1500, the steady rate must be about 1,580-1,690 tok/s at full residency.** That assumes reuse as good as SGLang's 93 % or vLLM 0.19's 88 %.
- At vLLM-0.32-level reuse (60-68 %), Duck lost 21 % of generated tokens against 0.19 at the same streams. That would need about 1,900 steady.

### 5.2 Residency, not stream count, is the first lever

With 5-8 resident contexts, every non-decoding stream costs 12-20 % of the batch, against 6 % at 16 streams.

- **Fix workers = resident capacity**, computed with the `e21_capacity` block rule on Merlin's own startup log. PLAN §2 gives FP8 = 150 blocks ≈ 5 streams at 70k.
- **Never time-slice more agent sessions than KV holds.** EVAL30 (B3) halved the effective rate on 27B with a 870k pool. FN has 558k, so the same mistake costs more.
- **Keep one KV gate** (no new attempt while usage ≥ 0.95) and **no step-down logic** (B6).
- **Assert real running == workers from the server's own `num_requests_running`** (B7). E21 at 18/20 streams still showed waiting up to 10-17.

### 5.3 Fixed Mamba blocks per request may dominate

- 27B on vLLM: F = 3 × (2 + k) = 15 blocks with MTP-3 (`e21_engine_new.py:209`).
- At block 4096, a 70k context is 18 attention blocks. If FN's F is ~12, PLAN's 150 blocks give 150 / 30 = 5 streams: about 40 % of each stream's KV would be state, not context.
- That makes L3 (k), L5 (SSM dtype) and L9 (block size) **residency levers**, not only speed levers. Every one changes how many 70-120k sessions fit. Measure F from the RC2 log ("Maximum concurrency" line).

### 5.4 Prefix reuse decides the multi-turn cost

- On L4, one uncached prompt token costs ≈ 0.27 generated-token-equivalents (27B: prefill ~1,450-1,578 tok/s vs decode ~400). FN's prefill rate on 4×L4 is unmeasured.
- Duck replays the whole history each turn, so at 70-120k contexts every turn's previous reply (often 20-60k) is at stake.
- **Required smoke metrics** (all exist in E21 `e21_report.json`):
  - `prefix_hit_pct_whole_run`, `recomputed_prompt_tokens`;
  - the reply-reuse class (`full_reply` / `prefill_end` / `partial` / `zero`, agent_effective_throughput.md §1);
  - burst prefill tok/s.
- Targets: ≥ 88 % whole-run hit (vLLM 0.19 level), ideally ≥ 93 % (SGLang / Franzen).
- Merlin's `--use-replayssm`, `--prefix-match-unit 64` and "checkpoint retention always on" are the candidate fix. The Froggeric template is part of it.

### 5.5 No abort-and-resubmit anywhere in the solve path

- Use deadline-clamped `max_tokens` per request (R6-S6) and `min_admit_sec`, as in E15B and E21: overrun ≤ 1.2 s, 0 stragglers.
- On a 0.32 base, aborted requests lost their cache even with free KV (B5). A Poetiq-style drain on FN is risky until a drain smoke shows its phase-2 hit rate.

### 5.6 Keep the queue non-empty to the deadline

- At 1500 tok/s the run has ~62M tokens. 240 tasks × 2 attempts × ~120k ≈ 57.6M, so supply and demand are close. A tail **will** appear near the end unless something fills it.
- Use stage 3 (`extra_attempts`): an attempt 3 on tasks with no accepted answer. When time binds, it ranks above an attempt 2 of a class-2 task (Notes/12 §D, "best order").
- This raises both tok/s (G5) and score (G9). E9 is the warning: filler on hopeless units adds tok/s, not hits.

### 5.7 Tail length scales with the per-stream rate

- With few streams, the final wave lasts about one attempt: L / r_stream. At 1500 / 8 ≈ 190 tok/s per stream, a 120k attempt runs ~10.5 min. At 5 streams and 300 tok/s per stream, ~6.7 min.
- So the residual tail costs ~1-2 % of 11.5 h, provided §5.6 holds. Without §5.6 it could be hours: R4 had 108 min, ORDER2 44 min.

### 5.8 Order and attempt policy carry over unchanged

- ORDER2 + cells for stage 1: insurance, +3 points at a tight clock.
- Stage 2 order 1, 0, 3, 2 without "flagged first".
- Do not skip attempt 2 on accepted tasks unless time binds: accepted answers are wrong 26-31 % of the time (Notes/12 §C).

### 5.9 Budget per attempt

- On 27B Duck the plateau was 55-60k per attempt, with no solve after 55,126 tokens (Notes/12 §E).
- FN at 70-120k per attempt (`thinking-120k-full-run-12h.md`) needs its own band measurement: the share of solves within 30/45/60/90/120k.
- The time model is linear in tokens per attempt, so a longer attempt directly lengthens the final wave (§5.7).

### 5.10 Isolate side workloads

- E8 lost 15 units to a scorer OOM (B10). No side GPU work (scorers, probes with large `prompt_logprobs`) may share the engine during the solve.
- Probes go before the solve, as E21 does.

### 5.11 Concurrency sweep

- More streams did not help 27B at long context (B11). FN is MoE: fewer weight bytes per token, and the KV read per stream is the binding cost.
- The stream-scaling curve at 70-120k must be measured on FN, not assumed.
- The measurement must assert real running == level (memory rule).

---

## 6. Open questions

1. **Does Merlin's `--use-replayssm` + `--prefix-match-unit 64` + always-on checkpoint retention keep the previous reply cached across Duck turns with MTP K3 on?**
   - Our only 0.32 Duck data (stock vLLM 0.32, align) shows 59.6-68.5 %.
   - Upstream #50551 (decode checkpoints) rejects hidden-state spec decode (`kv_capacity_and_harness.md` §3).
   - Needs a multi-turn smoke with `prefix_hit_pct_whole_run` and reuse classes.
2. **What are FN's fixed Mamba blocks per request (F), and how do they change with MTP k and the SSM dtype?** That decides residency (§5.3).
3. **What is FN's prefill rate on 4×L4** (TP2×PP2+EP, chunk 1024)? It sets the cost of every uncached token and of any drain.
4. **Does chunk 1024 interleave prefill with decode well enough** that a 60k re-prefill does not stall the other 4-7 streams? On 27B vLLM a re-prefill ran at ~1,450 tok/s and generation fell to 5-17 tok/s in drains. L2 / `--long-prefill-token-threshold` is the knob.
5. **Is the stock vLLM 0.32 "aborted request loses its cache even with free KV" behaviour present in Merlin?** It decides whether any abort path (drain, timeout retry, relaunch) is safe (Exp17 drain audit takeaway 3).
6. **Run average under PP2.** With PP2 and only 5-8 sequences, micro-batches are tiny. Does the steady rate fall faster than linearly when 1-2 streams are in prefill or tool time? We have no PP data at agent concurrency: the E23 PP4 smoke crashed at 8k (memory `experiment-23-six-levers.md`).
7. **FN solve-vs-tokens curve** for 70-120k attempts: the share of correct solves per band. This sets the attempt cap that maximises score per run-hour (27B: plateau 55-60k). It is not a throughput question, but it sets the tail (§5.7).
8. **Hidden-run LB÷eval for Duck.** We have one ratio (E14S ≈ 0.88). The E15 40K35K (57609813) and E21 NB1 (57751618) LB results are not in the files I read, so the coverage ratio of the best Duck configs on the hidden run is unknown.
9. **The source of the 980 peak / 710 average gap** (thread 750546): tail, prefill or admission? The ratio 0.72 matches our EVAL12-type tails (0.72-0.78) and the E18 drain case (0.74). That suggests the team's harness lets the queue empty or drains with re-prefill. Unverified.
10. **Correction for a later reader.** The E18 note says the 7.7 % align-mode hit "is shared with S1" on 0.19. But in-process 0.19 DIRECT/Poetiq runs show phase-2 hits of 67.5-100 % (ORDER2 counters 0.834; L66/W3H 25/25 aborted-then-resubmitted prefixes hit). So the near-zero reuse is a property of the 0.32 server path, not of align mode on 0.19 (Exp17 drain audit §1 "The vLLM 0.32 smoke breaks the pure-pressure story"). For a 0.32 fork like Merlin, treat it as a risk to measure, not a given.
