# Insights from part_07 (conversation 2026-10-21 15:58 to 16:10)

## Scope and shape of this part

- I read all 3,155 lines, start to end.
- **Real dialogue: only 4 user messages** (2026-10-21 15:58, 16:06, 16:06, 16:09) and 5 assistant messages (16:00 to 16:10).
- **Most of the text is two IDE selections** of the doc `experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Review/E12 next fix list (running, 2026-10-20).md`:
  - lines 1–1413 attached at 15:58 (convo lines 3–1418);
  - lines 1–1468 attached at 16:06 (convo lines 1482–2952).
- The second selection repeats the first one. Its only new part is the section "Re-read of both G4B audits" (convo L2895–2949), which the assistant added at 16:00–16:02:
  - the fix-by-fix verdict table;
  - G4-14..G4-18;
  - corrections to its own numbers.
- The doc snapshot carries the full E12 history from rev 5 to rev 6d plus the G4B GPU run. Everything quoted below from the doc is marked "(doc)". Its dates are those the doc records (all 2026-10-21 unless noted).
- **Cut `[result]` lines:** these are only code reads near the end (L3040–3146: the `arc2_duck.py` grep `[+5803]`, the BranchSession read `[+26295]`, the resume texts `[+41777]`, the agent class `[+52520]`, builder reads `[+1968]`, `[+5884]`, `[+23924]`, `[+2997]`, `[+11067]`, `[+4869]`).
  - They were the start of the "take all" implementation.
  - Their cut parts are not needed for any finding below. I do not guess their content.
  - One cut matters a little: L3066–3067 shows `ARC2_RESUME_TRIMMED = ("Nothing has been …`. So the trimmed-resume text still starts with "Nothing has been…". The rest is cut, so I cannot tell whether it is the wrong "Nothing has been submitted yet" text (see 5.13).
- The part ends at 16:10 while the assistant is still reading code. **No implementation of the G4 bundle happens inside this part.**

---

## 1. User instructions, rules, preferences and decisions

### Messages in this part

| Date/time | Verbatim | English meaning | Type |
|---|---|---|---|
| 2026-10-21 15:58 | "রিভিউ gulo abar dekhe nio....I think you miss something" | "Look at the reviews again… I think you missed something" | Complaint / correction (correct: the assistant had left out the fix-by-fix verdicts) |
| 2026-10-21 16:06 | "কোন পরিবর্তন আমাদের কি ফিক্স করবে?" | "Which change will fix what for us?" | Asks for a problem → fix → confidence map |
| 2026-10-21 16:09 | "সবগুলোই nio...update" | "Take all of them… update" | **Decision: take every G4 item.** This covers G4-1..G4-18, including G4-17, which the assistant had said needs a separate decision |

**Ambiguities in the 16:09 decision (unresolved in this part):**
- **G4-17 option.** The user said "all", but did not pick between the two options: (a) lower the fair-share minimum chunk to about 12,288, or (b) allocate by remaining candidates (L1451, L1476, L3006, L3032).
- **Items marked "later" or "not now".** "All" literally includes:
  - G4-11 and G4-12, which the assistant had put under "পরে" (later), L1477;
  - G4-9 "Not now: `--disable-radix-cache`, mamba-strategy switch, kernel patches" (L2856).
- The assistant answered "সবগুলো নিচ্ছি" ("taking all", 16:09). The part does not show how it treated G4-9, G4-11 or G4-12.

### Standing rules and decisions recorded in the doc snapshot (still binding context)

| Recorded date | Quote (doc) | Meaning |
|---|---|---|
| 2026-10-20 | "Read every GPT report in full and validate every claim. Add the validated items here. Change no code until the user says so." | Standing review rule. 15:58 is a case of it being violated (the assistant did not record everything) |
| 2026-10-21 | "Those were different experiments. That something did not work there does not mean it will not work here." | Past negatives from other setups are context, never verdicts |
| 2026-10-21 | "Gate 4b: permission granted. Take 12 streams. Permission for the rest too" | Approves 12 workers, the G4B time-share profile, and the other rev 6 approvals: the R6-Q1 caps 12,288/24,576/32,768 and the R6-Q2 first exposure 28k |
| 2026-10-21 | "OK, do 1 to 5" | Rev 6d = audit items 1–5 only. PARITY fixes (6) and the self-contained bundle (7) are not taken |
| 2026-10-21 | "run EVAL12R6G4B on the Foysal account" | Explicit GPU go, for that notebook only |
| 2026-10-21 | User decision after G4B: replace the probe with "no new task while KV usage ≥ 0.95" | Becomes G4-1 (P0) |
| 2026-10-21 | "check every fix was taken, update, validate, then zip" | Origin of rev 6b |
| 2026-10-21 | "re-read every GPT and agent recommendation, something important may be missing" | Origin of the rev 6 gap check. **Second time the same complaint pattern appears** |
| 2026-10-21 | "the same notebook, change only the xhigh agent" (chose low) | Origin of the low-effort variant |
| 2026-10-21 | "take them, validate again" (R5-1..R5-5) | — |
| 2026-10-21 | B1–B8 + B9(a), temperature 1.0 | Rev 5 bundle |
| 2026-10-19 (per settings doc) | "12,288 = approved exception to the 60k floor" | Valid for the rev 1 / EVAL30 milestone port only. Rev 6 needed a new decision, which came with "Permission for the rest too" |
| (doc, F7) | "the user's rule is that Duck runs on every task (full coverage)" | Full coverage is a hard goal |
| (doc) | User's "bundle changes, big jumps" rule vs GPT's "isolated, measured steps" | The doc states that the order of testing is the user's decision |

---

## 2. Measured facts and results (exact numbers)

### G4B GPU run (EVAL12R6G4B, sha `1bdadd82…`, kernel `foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b` v1)

- **Before the run:**
  - GPU quota 21.88 of 30 h used on the Foysal account.
  - The check-only passed (account, 3 models, 2 datasets, sha).
  - Expected about 50 min (13 min load + about 35 min solve + report).
- **Result:** COMPLETE in **3,528 s**. **2/19, task-norm 16.67 %** (bf45cf4b, b5ca7ac4 solved).
- **Volume:**
  - 12/12 tasks exposed;
  - **47.7k tokens per task**;
  - 572,035 tokens, 180 requests;
  - 75.0 % cached;
  - 16 branches: 2 done, 14 paused;
  - 0 stragglers; stage-1 left 0.
- **Load: 1,197 s** ("slow cold start").
- **Capacity probe bug:** stepped 12 → 6 on queue = 1 at KV 13.3 / 26.0 / 31.5 %. Queue = 1 is the steady state with 12 in flight (11 running + 1 queued).
- **Throughput (assistant Bash at 16:00, L1428–1431):**
  - 11 running @ 20–40k context: **370.9 tok/s**;
  - 6 running @ 20–40k: **321.4 tok/s**, a **13.3 % loss**;
  - 11 running @ <10k: 406.3 tok/s (20.9 % vs 6).
- **Fair share:**
  - shares [7,544, 7,266, 7,332, 6,044, 5,510, 3,758] vs min chunk 24,576;
  - stage-2 runs: 136b0064 b1 29,137; 247ef758 b0 25,859; 247ef758 b1 25,399; 9385bd28 b1 31,142; f931b4a8 b0 25,240; f931b4a8 b1 37,882.
- **12-running lines:**
  - only 29 decode lines, all in 11:59–12:12;
  - they alternate 42–250 and 382–497 tok/s. This fits admission prefill interfering with decode, not a ceiling.
  - Earlier reading "194–258 vs 370–406 at 11" was withdrawn.
- **Facts view:**
  - first call after 46 tokens;
  - 7/12 b0 never wrote a transform (29–59k tokens spent, L2964);
  - p90 25.6k characters per request; max 65k characters in one request (L2964).
- **Print-guard leak:** `print(r, row)` rows bypassed `_GRID_ROW_RE`. 173 rows leaked in 3 branches; 432 rows were hidden.
- **First check:**
  - observed median 29,384.5 (6/16 checked); b0 26,228;
  - the reported 33,690.5 was a censored proxy.
- **Cache:**
  - aggregate 75.0 % vs rev 5 aggregate 84.3 % (xhigh) / 88.7 % (low), a −9.3 / −13.7 point deficit;
  - the README had used request-level 85.4 / 91.0.
  - 20/180 requests (254,890 prompt tokens; 14 later ones) had `cached` = null. In SGLang 0.5.21 null means exactly 0 (`usage_processor.py:13-15` omits `prompt_tokens_details` when cached = 0, `--enable-cache-report` on).
  - Null share: G4B 11 % vs rev 5 xhigh 6/127, low 23/449 (5 %).
  - 15/180 later requests had < 5 % cache while KV ≤ 0.44; trims explain only 2 branches.
- **Mamba:**
  - usage max 0.70–0.71 in all 7 logs;
  - kv_pool / chunked_prefill = 866,744 / 4,096 = **211.6 vs mamba cache 56** (the #36935 regime; downgraded later).
- **FP8 KV:** "no scaling factors, defaulting to 1.0" appears 8 times; the same warnings exist in rev 4 EVAL6 and both rev 5 runs (not new).
- **b1:**
  - 21.6 % of generated tokens (123,560 / 572,035), 0 verified candidates;
  - all 4 b1 first requests cut at 12,288 before any call (first call at 12.4–18.0k);
  - the 32,768 rescue was never used.
- **Mechanism counters:**
  - helpers: `inspect_pair` 7, `components` 3, others 0; `check_many` 0;
  - auto-check 2 calls; verify 2 verified, 1 duplicate skipped, 0.86 s;
  - estimate ratio 0.998;
  - resumes: first 16, continue 45, length 5, partial 5, verified 1; `nudges_replaced` 0;
  - soft-pause overshoot p50 5.1k, max 15.8k;
  - deadline: 91 clamps, 0 stragglers, finished 9 s early;
  - 7 no-call cuts; 1 reasoning-only trim;
  - isolation 12/12; control parity 2/2; target-only parity never run.
- **Late partial:** 247ef758_b0 reached 1/3 pairs, cell accuracy 0.771, 5 partial resumes, no solve.
- **Accepted candidates double-count (M1-A):** 4 events = 2 programs. `fixed_offset` 2 = both from b5ca7ac4_b0 (check + submit).
- **Facts sizes:** 510–798 tokens in the 16 branches.
- **Solved branches** wrote `transform` only at call 15 (bf45cf4b_b0) and call 18 (b5ca7ac4_b0). The 7 no-transform branches had 8–14 calls.
- **Capacity arithmetic:** 240 × 60k = 14.4M tokens = **10.81 h at 370 tok/s vs an 11.46 h window**.
  - At 6 streams for about 11 h: about 54k vs 63k tokens per task (≈ 8k less, L2990).
- **G4B window formula:** hidden solve end 43,020 − 120 − 180 = 42,720 s; 12 × (42,720 − start) / 240 ≈ 175 s per task ≈ 35 min.

### Earlier facts carried in the doc (for reference)

- **Rev 5 results:**
  - EVAL6R5 xhigh (sha `f4fed1ac…`): 4/6 tasks, 5/8 inputs;
  - EVAL12R5_LOW (sha `26fff41e…`): 7/12 tasks, 10/19, submitted as ref 57456562 (pending; no LB in this part).
- **Rev 5 low throughput:** 1.744M tokens in 6,570 s at 6 streams ≈ 265 tok/s.
- **Rev 4 / rev 5 replays:** rev 4 EVAL6 replay 7/8; rev 5 xhigh 5/8; low 10/19 (oracle = final, so selection is not the lever).
- **Selective b1 replay:** xhigh 5/8 at 498,253 vs 846,080 tokens (−41.1 %); low 9/19 at 1,094,372 vs 10/19 at 1,743,889 (−37.2 %).
- **First check (rev 5):** median 55.1k (xhigh) / 53.7k (low); 37 % / 20 % of pre-check tokens come after the first code-intent phrase; misreads in 12/12 and 23/24 branches.
- **Request concentration (rev 5):**
  - xhigh: 28 of 127 requests ≥ 10k carry 80 % of tokens;
  - median completion per request 1,115 / 1,090; median prompt 51.9k / 41.6k;
  - uncached ≈ 888k / 1.577M.
- **Late corrections:** 135a2760_b1 71,306 → 97,972; 898e7135_b0 56,868 → 73,523; 898e7135_b1 77,859 → 79,662.
- **EVAL30 T8 overrun:** 679 s past the solve deadline, not 1,412 s.
  - The per-turn read timeout ran 419.6 s and 469.7 s past it.
  - At 14,247 s: `num_running_reqs` 11, `num_queue_reqs` 8, `gen_throughput` 0.0.
- **EVAL30 request timeouts:** 40 in 21 of 31 branches (28 full 900 s + 12 deadline-bounded, ≈ 31,000 stream-seconds).
- **Decode / prefill / spec numbers:**
  - decode at 40–60k context about 25 % slower (323 vs 429 tok/s);
  - DFlash acceptance 4.43 → 3.77 over context;
  - EVAL30 re-prefilled 8.4M tokens;
  - cold prefill about 1,400–1,650 tok/s (≈ 30–35 s per 50k resume);
  - spec vs plain shared only 152–166 of 512 tokens under batching (E11-03);
  - an unchanged repeat scored 0.1875 vs a 0.50 threshold.
- **Cache history:**
  - stable order raised cache 11–18 % → 87 % (E11-06);
  - an abort without waiting gives 0 % vs 52–99.7 %;
  - no tools gives 15 % vs 99 % shared prefix;
  - a template conditional gives 75 → 45 %.
- **KV windows:**
  - per-stream window 1.0 × 881,336 / 12 = 73,444;
  - history 48.4k at the 24,576 cap and 60.6k at 12,288;
  - 0.9 → 66.1k / 41.0k; 0.95 → 69,772 / 44,684.
- **Facts block:** p50 699 / p90 779 / max 798 after the rev 6d levels (earlier p50 1,237 / p90 1,752 unbounded; 736 / 837 / 1,396 in rev 6b).
- **Quota (hosen42):** 30.79 of 60 h used (rev 5), 31.29 while EVAL6R5 ran.
- **Tests:** smoke_rev6 104/104 ×3; rev5 56/56; adapter 41/41; harness EVAL12R6 45/45, G4B 46/46, PARITY 19/19.
- **Rev 5 verification cost:** 0.75–2.71 s per verification, ≤ 3 per branch.

---

## 3. Failures, mistakes and root causes

### Assistant errors in this part

1. **Missed the fix-by-fix verdict table** (user 15:58: "I think you miss something").
   - The assistant admits at 16:00/16:02: "আপনি GPT-কে আসলে যেটা জিজ্ঞেস করেছিলেন … সেই fix-by-fix রায় আমি fix list-এ তুলিনি", i.e. "what you actually asked GPT for (the fix-by-fix verdicts), I did not put into the fix list".
   - Root cause: it validated GPT's claims, but did not record the answer to the user's actual question.
   - Six proposals were also missing (G4-14..G4-18 and the G4-4 paging addendum).
2. **Wrong throughput-loss number.** It wrote "15–20 %"; the right figure is **13.3 %** (370.9 → 321.4 at 20–40k).
   - Root cause: it mixed the < 10k bin (406.3) with the 20–40k bin.
   - GPT's "13–16 %" was right.
3. **#36935 overclaimed** ("Confirmed, and stronger than GPT says"), then downgraded: mamba usage never reached full (max 0.70). The cause stays open.
4. **"12 running is slower" claim withdrawn.** Root cause: 29 early lines mixed with admission prefill.
5. **Cache denominators.** The README used request-level cache % (85.4 / 91.0) instead of the aggregate (84.3 / 88.7).
6. **Metric bugs:**
   - `accepted_candidates` counted check + submit of one program twice (`build_e12_duck.py:1037`);
   - the first-check median was a censored proxy (33,690.5 vs observed 29,384.5).

### Run/engineering failures (G4B and rev 6 builds, from the doc)

- **Capacity probe stepped 12 → 6 on queue = 1 at KV 13–31 %.**
  - Root cause: a queue > 0 for 3 probes was treated as overload, but the server's steady state with 12 in flight is 11 running + 1 queued (the 11-of-12 behaviour).
  - Cost: 13.3 % throughput.
- **Print-guard leak.** Indexed rows like `0 [0, 0, 5, …]` did not match `_GRID_ROW_RE`.
  - The proposed regex catches 173/173 rows, but misses `0: [..]`, `row 3: [..]` and tuples. Even the widened one misses tuples.
- **Fair share was inert.** `max(24,576, share)` with every share 3.8–7.5k means the minimum chunk always wins. At 240 tasks it gets worse.
- **b1 first request always cut at 12,288.** Its first call comes at 12.4–18k, so one extra request per b1.
- **Prefix-cache regression** (75 % vs 84–89 %; 20/180 zero-hit requests): cause unknown.
  - Hypotheses: mamba radix split #22935, or the #36935 ratio.
  - **Note:** the rev 6b fake-server prefix invariant test passed "84 ok, 0 breaks". So the fake server could not see whatever broke the prefix on the real server.
- **Probe metric pick.** It took KV by substring. Only the last fallback `pick("token_usage")` could return max(full, swa, mamba). The real value used was `sglang:full_token_usage` (named `e11:kv_cache_usage`).
- **Earlier build defects (doc, all fixed):**
  - regex catastrophic backtracking hung the smoke test > 11 min (overlapping separators);
  - wrong opener after a rejected submit;
  - caps metric counted trims, not requests;
  - CRLF introduced into `arc2_duck.py`;
  - two workers could drive the same agent (24 concurrent entries in the negative control);
  - fair-share starvation (one task took 792k);
  - published records shared nested lists with live sessions (fixed with a deep copy);
  - isolation preflight failed open (`isolation_ok = not leaks`, `None` kept 12 workers);
  - final join had no shared bound (12 × 120 s = 24 min worst case);
  - p90 index `int(0.9 × (n − 1))` was not nearest-rank;
  - PARITY `urlopen(req, timeout=900)` with no deadline (24 probes, up to 6 h).
- **Earlier assistant errors corrected in the doc:**
  - T8 overrun 1,412 s → 679 s (relative vs absolute subtraction);
  - "4" EVAL30 timeouts → 40;
  - "temperature 0.6 Qwen-like" was wrong (official thinking setting: 1.0);
  - "the newest block always stays" was wrong;
  - "metrics_snapshot drops `num_retracted_requests_total`" was wrong (that was the vLLM-variant filter);
  - "G4B predicts LB/eval" overclaimed (it is a screening proxy);
  - "conditional branch 1" was really "prioritised by evidence class";
  - "No failed branch got even one train pair exact" was wrong.

---

## 4. Decisions made and reasons; options rejected

- **16:09: all G4 items taken.** Before that, at 16:02 and 16:07, the assistant grouped them:
  - **mandatory:** G4-1;
  - **take with it:** G4-2, G4-3, G4-4, G4-5, G4-6, G4-10, G4-16;
  - **measurement only:** G4-7, G4-8, G4-14, G4-15, G4-18;
  - **user decision:** G4-17;
  - **later:** G4-11, G4-12.
- **Confidence ranking (16:07):**
  - **certain:** G4-1 (13.3 % speed; in 7 real logs the rule would only have held in the KV-thrash run E12 EVAL30), G4-3 (173/173), G4-4, G4-5 (small gain);
  - **certain for the cut only:** G4-2. It will not fix b1's zero yield, because reasoning is kept after a cut;
  - **medium:** G4-6 (token-triggered, not call-count);
  - **low:** G4-10 (prompt-only rules failed twice: Exp 4 run 4, E11-06);
  - **low–medium:** G4-16.
- **Rejected or modified:**
  - **GPT's P0-2 call-count nudge** ("after 2–3 read-only calls") was rejected: solved branches wrote `transform` at calls 15 and 18, so a call-count trigger fires on them too. It became the token-triggered G4-6, with the threshold taken from a replay.
  - **"Do not drop to 11 running"** was agreed (no change to 12 workers).
  - **G4-9 "not now":** `--disable-radix-cache`, a mamba-strategy switch, kernel patches. A larger `--max-mamba-cache-size` or `--chunked-prefill-size` only after the G4-8 measurement.
  - **Use the exact metric name, with no fallback to `token_usage`;** fail open with a warning if the metric is missing.
  - **G4-10 is a prompt rule, not a hard "within first 2–3 interactions" rule.**
  - **KV fraction: keep 1.0 for G4B.** Audit 2 preferred 0.95 for the hidden run; this was left to G4B evidence.
- **Not in the bundle (16:07):** overfit (only 2 candidates, so it cannot be measured) and attempt_2 diversity.
- **Earlier doc decisions (context):**
  - p90 last-admission cutoff built but **off by default** (it would idle 12 workers for about 15 min; per-request clamps already prevent overruns);
  - target-only parity kept out of production ("never alongside production");
  - the 0.9 KV fraction rejected for 1.0 (41.0k history < rev 5 median prompts 41.6–51.9k; D66 ran 12 slots, KV peak 90 %, 0 preemptions);
  - the tokenizer fallback is recorded, not asserted (an assert would kill the solver);
  - R5-6 not taken (0 of 405 calls);
  - the engine's mamba 56 and CUDA-graph list 1–8, 10, 12 unchanged ("numbers show no need").

---

## 5. Open items

1. **Implementation of all G4 items (rev 6e).** Started at 16:09–16:10 (code reads only). Not done in this part. No build, sha or tests yet.
2. **G4-17 option not chosen** (chunk 12,288 vs candidate-based allocation). Needs the 240-task simulator; a smaller chunk means more re-prefill.
3. **How G4-9 / G4-11 / G4-12 were handled under "all"** is not shown.
4. **G4-6 threshold** must come from a replay on bf45cf4b_b0 / b5ca7ac4_b0 so solved branches are not nudged. Not computed in this part.
5. **Tuple rows in the guard** (`0 (0, 0, 5, 2)`) are missed even by the widened regex. The doc says "plus a tuple case".
6. **Cache regression cause unknown.** G4-8 (controlled replay) and G4-15 (per-request instrumentation) are diagnostics, not fixes.
7. **Unexplained:** `gen_throughput` 0.0 with 11 running + 8 queued in EVAL30 (likely KV thrash).
8. **PARITY fixes not taken** (C6-1 deadline, C6-3 tool name, Q2 long probes, A3-Q2 multi-turn continuation probe). PARITYR6 stays NO-GO. **Target-only parity has never been run.**
9. **Self-contained test bundle** not built (`smoke_adapter.py`, fixtures, relative paths).
10. **SGLang issues never verified by the assistant** ("C"): #38009, #36548, #37475, #31954, #39087, #43514, z-lab/dflash #166, HF discussion #173.
11. **Hidden-run defaults still open:**
    - KV fraction (0.95 vs 1.0) is undecided;
    - EVAL12R6 (unconstrained) to run after G4B; EVAL30R6 builder variant not built;
    - rev 6 is "not yet qualified as a 240-task submission".
12. **Pending submission** ref 57456562 (EVAL12R5_LOW): no LB result in this part.
13. The cut `ARC2_RESUME_TRIMMED = ("Nothing has been …"` (L3066): it is not visible whether the trimmed resume still carries the wrong "Nothing has been submitted yet" phrase. Worth checking in rev 6g.
14. **Overfit and second-candidate diversity:** no fix in the bundle.

---

## 6. Insights a fresh reader might miss

- **Same complaint twice.** "Something important may be missing" (gap check) and "I think you miss something" (15:58). The assistant tends to validate claims instead of answering the user's actual question. Always record the verdict to the question asked.
- **"An earlier Python call ≠ an earlier useful hypothesis".** The facts view moved the first call to 46 tokens, yet 7/12 b0 branches never wrote a transform. Tokens moved from hand-reading into redundant `arc_overview()` calls (14/16 first calls repeated the system-prompt facts).
- **queue = 1 is normal at 12 in flight on this engine** (11 running + 1 queued). Any queue-based overload rule misfires. This is the same 11-of-12 effect that E14 attributes to the DFlash min-free-slots delayer.
- **`token_usage` in SGLang 0.5.21 is the max over the full / swa / mamba pools.** Mamba sat at 0.62–0.70, so a `token_usage` gate would read ~0.7 at idle KV. Only `full_token_usage` is the KV pool.
- **Null `cached` = 0 hits on SGLang 0.5.21.** GPT read nulls as "unknown"; the code shows they are real zeros. Per-request averages hide this, so use aggregate sums.
- **Two different 12-running readings:**
  - "12 running slower" came from mixing admission-prefill windows;
  - throughput must be compared in the same context bin (<10k 406 vs 20–40k 371 at 11 running).
- **Fake-server tests passed the prefix invariant (84 ok, 0 breaks), but the real run lost cache.** The fake server cannot catch real-template or real-normalization changes. The later E14 finding (Duck normalization deleting blank lines → previous reply re-prefilled) fits this gap.
- **The fair-share code exists and is tested, but it is a no-op at real scale,** because the 24,576 minimum chunk dominates.
- **G4-2 does not fix b1's zero yield.** After a cut the reasoning is kept, so the loss was only one extra request. b1's 21.6 % of tokens with 0 candidates is a representation/hypothesis problem (G4-11/12).
- **G4B is a time-share screening, not an LB predictor** (12 hand-selected tasks; stage 2 starts much earlier than with 240). Yet the user had approved it as gate 4b, and it is the only run at hidden-equivalent compute (about 47.7k tokens per task).
- **Penalties are inactive under DFlash2 in 0.5.21** (EAGLE accumulates penalty tokens, DFlash never calls `cumulate_penalty_output_tokens`). Any penalty arm needs a 2-line SGLang patch.
- **Per-turn vs per-request timeouts (T8).** A timeout computed once per turn lets a late request run up to one analyzer timeout (1,800 s) past the deadline. Only a per-request clamp is safe.
- **Reversals inside the doc:**
  - the T8 overrun 1,412 s became 679 s;
  - #36935 "stronger" became "unproven";
  - "metrics filter drops retract" became "the probe just did not use it";
  - KV 0.9 became 1.0;
  - "conditional" b1 became "prioritised".
- **The quota for G4B was small** (about 59 min wall).
- **Load time 1,197 s** on 2026-10-21 is the GCSFUSE slowdown that MEMORY says threatens the 30-min launch timeout.

---

## 7. Relevance to the E14-A notebook (concrete checks / fixes)

1. **Baseline windows "at 12 running" must assert the real running count.**
   - Read `#running-req` from the decode log; do not trust `num_queue_reqs == 0` or the configured `max_running`.
   - Evidence: with 12 in flight, the E12 server sat at 11 running + 1 queued, and the queue-based probe read that as overload (12 → 6, KV 0.13–0.31).
   - For the `--min-free-slots-delay 1` arm, the pass criterion should be "running == 12 for the window", with queue = 0 as a secondary check.
   - Do not port the E12 queue-based capacity probe into the E14 25-min Duck run. If the rev 6g solver still has it, confirm it is replaced by the G4-1 KV rule (exact `sglang:full_token_usage`, no `token_usage` fallback).
2. **Throughput comparison method.**
   - Bin tok/s by context length (the E12 numbers differ 406 vs 371 between <10k and 20–40k at the same concurrency).
   - Exclude or flag windows that contain admission prefill. The 12-running lines alternated 42–250 and 382–497 tok/s, which made a false "12 is slower" finding.
   - Set a minimum sample size per window (29 lines was too few to judge).
   - Compare arms in the same context bin, or the arm effect will be confounded (13.3 % vs "20 %" error).
3. **Cache metrics in every phase:**
   - treat a missing `prompt_tokens_details` as **0 cached** (SGLang 0.5.21 omits it when cached = 0 with `--enable-cache-report`);
   - report the aggregate Σcached / Σprompt, not the mean of per-request %s (85.4 vs 84.3 error);
   - report the count of zero-hit requests (baseline: G4B 11 %, rev 5 5 %).
4. **Token-level replay and latch test should copy G4-8's design:** warm → exact replay → other prefixes → replay, recording cached tokens, TTFT, hit rate, KV and mamba usage.
   - This is the only way to separate "prefix changed" from "cache state lost".
   - Include the #36935 ratio in the record: kv_pool / chunked_prefill (866,744 / 4,096 = 211.6) vs `max_mamba_cache_size` (56).
   - For the int8 mamba checkpoint arm, log mamba pool usage (it was 0.70–0.71 max) and the zero-hit share, since the mamba radix split (#22935) is a live hypothesis for 0-hits.
5. **The Duck-history raw-reasoning fix must be verified against the real server tokenization, not a fake server.**
   - E12's fake-server prefix invariant passed 84/84 while the real run lost about 10–14 cache points.
   - E14 check: in the real Duck run, request n+1's prompt token prefix == request n's prompt + completion tokens (except after a trim or cut), measured via cached tokens per request (G4-15 fields: prefix-match length, cached, trimmed flag, idle time since the branch's last request).
6. **Greedy parity (A/A, concurrency 12, target-only):**
   - **(a) Run the same-config A/A control first.** Under batching even unchanged configs diverge (spec vs plain shared only 152–166 / 512 tokens; an unchanged repeat scored 0.1875 vs a 0.50 threshold). Judge target-only vs DFlash against the A/A divergence, with task-level outputs (tool call parsed, tool **name + arguments**, finish), not raw token equality.
   - **(b) Every parity request needs a deadline clamp.** E12 PARITY used `urlopen(timeout=900)` with no deadline (24 probes, up to 6 h, the C6-1 blocker), and the relaunch wait had a `max(60, …)` floor that can pass the deadline.
   - **(c) 2,048-token probes rarely reach a tool call** (rev 6 first calls came at 12.4–18k; rev 5 p50 33.6k). Parity probes of 2k only test thinking text.
   - **(d) The target-only relaunch must be its own bounded launch with the four `--speculative-*` flags removed.** Budget its load time: G4B load was 1,197 s, and a second cold launch can cost about 20 min of the 200 min budget. It must go through the same stall watchdog.
7. **CUDA-graph arm 1..13.** E12's list was 1–8, 10, 12, so 9 and 11 pad up, and 13 (the max-running 13 workaround for 11-of-12) had no graph. Verify in the server log that bs 11, 12 and 13 are captured in the arm, and measure capture memory (the reason they were left out).
8. **The 25-minute Duck run must keep the rev 6d deadline safety:**
   - per-request read-timeout clamp to the absolute deadline (T8: per-turn timeouts overran by 336 s, up to 1,800 s possible);
   - verify budget clamp;
   - one absolute join cutoff deadline + 60 s;
   - deep-copied published records;
   - three-state isolation verdict (12 workers only on PASS).
   - Check that these are on in the rev 6g code path used by E14-A.
9. **Stall-based launch watchdog: do not treat known-benign log lines as progress failures or errors.** FP8 KV "no scaling factors, defaulting to 1.0" appears in every run (8×). A runtime stall shows up differently: `gen_throughput 0.0` with 11 running and 8 queued (EVAL30, KV thrash). The watchdog covers launch; runtime thrash needs its own signal (`gen_throughput` + running + queue + KV).
10. **Phase timing and quota estimate.** G4B: 3,528 s total with 1,197 s load. The E14A 200-min budget should assume a load of about 20 min unless the /dev/shm copy proves faster; record load seconds in the phase record. The written wall-time/quota estimate before the run should cite this.
11. **Do not add penalty-based arms.** Presence/frequency/repetition penalties are inactive under DFlash2 in 0.5.21 (#28180, code-read), so such an arm would measure nothing.
12. **Decision metric for the whole E14 effort:** 240 × 60k = 14.4M tokens = 10.81 h at 370 tok/s vs an 11.46 h window. E14's baseline at real 12 running in the 20–40k bin must beat **370.9 tok/s at 11 running**; that is the number to compare against.
    - The 6-stream fallback gives 321.4; the < 10k bin 406.3.

**Concrete notebook fixes found (summary):**
- assert real running == 12 in the baseline windows;
- KV gating only on `sglang:full_token_usage`;
- null cached = 0 and aggregate cache %;
- an absolute deadline on every parity probe and on the target-only relaunch;
- A/A-calibrated parity on tool name + args;
- context-binned throughput that excludes prefill windows;
- a real-server prefix-match check for the raw-reasoning history fix;
- confirm CUDA-graph capture of bs 11/12/13.
