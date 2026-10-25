# Insights from part_06 (conversation 2026-10-21, 12:57 to 15:33)

## Source coverage note

- Input: `part_06.txt`, 4,478 lines. I read it all with the Read tool. Lines 1-1600 and 1601-2650 were read line by line. Lines 4005-4478 were also read line by line.
- Lines 2626-4011 are a second IDE paste of the fix list `E12 next fix list (running, 2026-10-20).md` (1,382 lines). I compared it with a byte-level `diff` against the first paste (lines 4-1338). The only differences:
  - the header status line;
  - the G4B run-log row, which is longer;
  - a new section "GPT audit of the G4B results ... G4-1..G4-9".
  
  All three differences are covered below. The rest of that paste is identical text, already covered from the first paste.
- Timestamps: the chat headers (`### USER 2026-10-21 12:57`) are UTC, not local time.
  - The G4B report run_id is `20261021T113337Z`, and its server log lines start at 11:57.
  - A robocopy at chat time 14:22 printed "Ended: Wednesday, October 21, 2026 8:24:53 PM", so local time = UTC+6.
- Cut `[result]` lines whose missing part may matter (content not guessed):
  - L1538-1546: old-log queue scan, cut after "+1717" and "+4774". The visible rows are E12 rev5 low, E12 rev5 xhigh, G4B, E12 EVAL30 and one 22-running log. The full per-run table appears later in the assistant's message at 13:01.
  - L1610: E11-06 KV distribution, cut at ">= …[+159]". The assistant's table at 13:11 gives E11-06 KV p50/p90/max 0.40/0.50/0.69, ≥0.95 = 0, mamba 0.66.
  - L1700-1704 and L1846-1891: per-branch metric tables, cut. Only the first rows are visible.
  - L2034 `_knobs.txt`, cut after `E12_R6_FAIR_…`. The rest of the knob list is not visible.
  - L2483-2503: probe and builder code, cut.
  - L2509-2515: `validate_gpt_g4b_out.txt`, cut after "M1-C". The assistant's summary gives its numbers.
  - L4415-4426: the 29 decode lines at exactly 12 running. Only the first 9 are visible; the rest is cut.
  - L1966-1980: the `facts_size` diff. It first ran with the wrong tokenizer set (p50 757 / max 1,005, 32 over 800). The rerun at L1982 gives the correct numbers.

---

## 1. User instructions, rules, preferences and decisions

| When | Verbatim (Bengali / mixed) | English meaning / effect |
|---|---|---|
| 2026-10-21 12:57 | "এত এত রিভিউ এর পরও এত বড় বাগ আসলো কিভাবে?" | **Complaint:** "After so many reviews, how did such a big bug get in?" It refers to the G4B capacity probe falsely stepping 12 → 6. The assistant admitted the fault and wrote a new process rule into memory: replay every runtime controller on REAL recorded metric series; take "healthy" fixture values from real logs; test the cooldown; give GPT real server-log excerpts. |
| 2026-10-21 13:10 | "এই probe stream 12 → 10 → 8 → 6 এটা এত জটিল করার কোনো মানেই ছিলো না।শুধু ৯৫% এর বেশি ক্যাশ থাকলে নতুন কোনো টাস্ক ডুকবে না,এটা সিম্পল একটা শর্ত দিলেই হতো।" | **Decision plus a complaint about complexity:** "There was no point making this 12→10→8→6 probe so complex. A simple condition was enough: if the cache is above 95 %, no new task enters." This became G4-1: 12 workers fixed; no new task/branch while full-attention KV usage is ≥ 0.95; fail-open if the metric is missing. It fits the standing "keep code simple" preference. |
| 2026-10-21 13:16 | "আমাদের কোড আর আউটপুট সহ রিডমি দিও ,আমি জিপিটি দিয়ে ভ্যালিডেট করাই,আর সমস্যা গুলো আইডেন্টিফাই করাই,আর ওর থেকে সমাধান চাই,আর আমাদের প্রতিটি ফিক্স এর কোনটা কাজ করেছে,কোনটা করেনি,আর কেন করে নি,কি করলে কাজ করবে সেগুলোও জানতে চাইবো।জিপ করো।প্রয়োজনীয় সব কিছু দিও জিপে" | "Give a README with our code and outputs. I'll have GPT validate it, identify the problems and propose solutions. For each fix I want to know: which worked, which did not, why not, and what would make it work. Zip it, with everything needed." The result was `E12_REV6D_G4B_RESULTS_FOR_GPT_2026-10-21.zip` (15.5 MB, 370 files). |
| 2026-10-21 14:21 | "D drive use koro" | "Use the D drive." The assistant moved the HF, pip and uv caches to `D:\DevCache` and set the user env vars TEMP/TMP to `D:\DevCache\Temp`, plus PIP_CACHE_DIR, HF_HOME and UV_CACHE_DIR. Revert script: `D:\DevCache\REVERT_env.ps1`. Memory `use-d-drive-for-scratch.md` was updated. |
| 2026-10-21 15:10 and 15:28 | (two pasted GPT audits, "E12 Rev6d — G4B GPU Results Deep Audit" and "... Independent Audit") | The standing rule applies: validate every claim and change no code until the user says so. |
| Earlier, from the compaction summary (L1356-1363, L1485-1501) | "হামিদ ভাইয়ের আইডিতে একটা কমিট করো।" then (interrupted) "এর আগে বলিও কোনটা কত সময় লাগবে?" | "Make a commit on Hamid bhai's ID", then "Before that, tell me how long each one will take." The rule: a time estimate before any run. |
| Earlier (summary) | "EVAL12R6G4B এটা ফয়সাল একাউন্টে চালাও" | "Run EVAL12R6G4B on the Foysal account." This was an explicit go, and the run was done. |
| Earlier (summary) | "রান শেষ আউটপুট বিশ্লেষণ করো" | "The run is finished; analyze the output." |
| Earlier (summary) | "OK ১ থেকে ৫ করে bolio" | "OK, do items 1 to 5 and tell me." This built rev 6d. Items 6 (PARITY fixes) and 7 (a self-contained test bundle) were **not** taken. |
| Standing (summary L1490-1501) | "অনুমতি ছাড়া কমিট করবি না" | "Don't commit without permission." Also listed there: never push/run/submit on Kaggle without an explicit go; never read or echo tokens or credentials; validate every claim; bundled evidence-based changes; "Change no code until the user says so"; D: drive for scratch; chat in Bengali; 60k floor (gate-4b approved); pre-run capacity check; never rebuild Kaggle-run notebooks; past negatives are not verdicts; trust the user's observations. |
| Quoted in the pasted fix list (2026-10-20) | "Read every GPT report in full and validate every claim. Add the validated items here. Change no code until the user says so." | A standing rule for GPT audits. |
| Quoted in the fix list (2026-10-21) | "Those were different experiments. That something did not work there does not mean it will not work here." | A past negative from a different setup is context only, never a verdict. |
| Quoted in the fix list (rev 6c, 2026-10-21) | "Gate 4b: permission granted. Take 12 streams. Permission for the rest too" | Approved the gate-4b time-share profile, 12 streams, and the other proposals (R6-Q1 caps 12,288/24,576/32,768 and the 28k first exposure). |
| Quoted in the fix list | "the same notebook, change only the xhigh agent" (chose low); "take them, validate again" (R5-1..R5-5); "check every fix was taken, update, validate, then zip"; "re-read every GPT and agent recommendation, something important may be missing"; "one bundled run, evidence-based changes only"; the user approved evidence-based output cuts (49,152 cap, 2026-10-21) | Earlier decisions recorded in the doc. The settings doc also records "12,288 = approved exception to the 60k floor, user 2026-10-19", for the rev 1 / EVAL30 run only. |
| Status at end of part (15:33) | (no user reply yet) | The user still has to choose which of G4-1..G4-13 to build. The assistant recommends G4-1, 2, 3, 4/5, 6+10, 7 and 8, and deferring G4-11 and G4-12. |

---

## 2. Measured facts and results

### EVAL12R6G4B v1 (Foysal account, kernel `foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b`, notebook sha `1bdadd8266e758ee48c0947d3197d0aa804f5d751d28277a79e947e7e73980a5`, adapter sha `7c60454c…`)

**Push and run**
- Check-only before the push: GPU quota used 21.88 of 30 h.
- Status COMPLETE in **3,528 s**.
- SGLang "ready in **1197 s**", a slow cold start (rev 5 took about 700 s). Solve start 1,463 s.
- Gate-4b window **2,063 s**, i.e. **171.9 s per task** (formula: 12 × (42,720 − solve start) / 240).

**Score**
- **2/19 inputs (10.53 %), task-norm 16.67 %**: bf45cf4b (b0, 19.4k tokens) and b5ca7ac4 (b0, 27.4k tokens). Both were submitted and host-accepted.

**Coverage and branches**
- 12/12 tasks exposed, stage1_left 0, stragglers 0. The run ended 9 s before the deadline.
- 16 branches (12 b0 + 4 b1): 14 paused, 2 done, heap_left 22.

**Tokens and requests**
- **572,035** generated tokens = 47.7k per task. Average **277 tok/s**.
- **180 requests**, 175 tool calls. Caps used: 12,288 × 20 and 24,576 × 160. The 32,768 rescue cap was never used.

**Prefix cache**
- Cached 2,008,877 / 2,678,857 = **75.0 %**. Uncached prompt **669,980** tokens, more than the 572,035 generated.
- 15 of 180 later requests were < 5 % cached, in 9 branches; only 2 branches had a trim event.
- 20 of 180 requests had `cached = None` (254,890 prompt tokens).

**Facts block**
- p50 627 / p90 763 / max 798 tokens. GPT 2: 510-798 across the 16 branches, 0 limit violations.
- Offline on 120 eval tasks with the real tokenizer: full facts p50 1,237 / p90 1,752 / max 2,892; bounded (shipped) p50 699 / p90 779 / p99 798 / max 798; raw grid text p50 4,881 / p90 8,950 / max 16,191. Bounded was smaller on 120/120, and identical under 3 hash seeds.

**Preflight and engine**
- Isolation PASS 12/12. `target_layer_ids` [5, 19, 33, 47, 61]. Same-server parity control 2/2.
- real_running_max 12, accept_len 4.2.

**Capacity probe events (the bug)**
- t=240 → 10, t=600 → 8, t=780 → 6. Each time queue = 1, KV usage 0.133 / 0.260 / 0.315, retracted = None.
- Server log: 188 decode lines with `#queue-req: 1`, 101 of them at running 11. Mamba usage ≤ 0.70.

**Throughput**
- 11 running ≈ **370-406 tok/s**; context-binned median at 20-40k context **370.9 tok/s**.
- 6 running ≈ **321 tok/s** (321.4).
- At exactly 12 running: 29 decode lines at 194-258 tok/s. **Retracted later:** all 29 come from the first 13 min, alternating 42-250 and 382-497 tok/s, a prefill effect (the visible lines include 139.74, 167.76, 466.1, 497.19).
- GPT 2: throughput loss from the false step-down is about 13-16 %. The assistant estimated about 15-20 % slower in the second half.

**Branch behaviour**
- First tool call median **46 tokens**, but it is mostly just `arc_overview()`.
- **7/12 b0 branches never wrote a transform**, although each got about 29-59k tokens.
- 10/16 branches never checked. Observed first-check median **29,384.5** (6/16 checked), b0 only 26,228. The reported **33,690.5** is a censored proxy.
- Print guard: 432 rows suppressed, but **173 indexed rows leaked** (`print(r, row)` form) in 3 branches; 138 `show()` rows.
- No-tool length cuts: 7. All 4 b1 first requests were cut at 12,288 before any call; the first call came at 12.4-18.0k tokens.
- b1 used **123,560 tokens (21.6 %)** and produced 0 verified candidates.
- The solved branches defined `transform` only at call **15** (bf45cf4b_b0) and call **18** (b5ca7ac4_b0). The 7 no-transform branches had 8-14 calls each.
- Reasoning per request: up to 65k characters in one request (136b0064_b0 max 64,870); p90 25.6k characters per request.
- Stage-1 runs took 1,120-1,510 s each (30-44k tokens). Overshoot p50 5.1k (5,107), max 15.8k (15,784).
- Fair share late in the run was 3.8-7.5k, but the minimum chunk is 24,576.
- 247ef758_b0 reached 1/3 pairs, cell accuracy 0.771. It checked incomplete transforms at transcript lines 1930, 2175 and 2876.

**Other counters**
- 91 deadline clamps.
- Verification median 0.86 s; 2 verified, 1 duplicate skipped, 0 cap hits.
- Estimate/actual prompt-token ratio median **0.998** (B3 tokenizer estimate works).
- `nudges_replaced` = 0; `cut_tool_calls_dropped` = 0.
- Resumes: first 16, continue 45, length 5, partial 5, verified 1.
- Per-stream window 72,228 (GPT 2), which is 866,744 / 12.

**Token-matched comparison**
- Within the G4B branch-0 token budget, rev 5 b0 would have reached a correct candidate on xhigh 1/6 and low 3/12 tasks; G4B solved 2.

### KV / queue scans of earlier logs (13:01, 13:11)

| Run | Decode lines | queue ≥ 1 | KV p50 / p90 / max | KV ≥ 0.95 lines | mamba max |
|---|---|---|---|---|---|
| E11-05 EVAL30 (esfersami50, 2026-10-18) | 2,816 | 2,615, mostly at running 11 | 0.44 / 0.54 / 0.63 (GPT: peak 0.65) | 0 | 0.70 |
| E12 EVAL30 (hosen42, 2026-10-19; the KV-thrash run) | 4,153 | 3,127 | 0.84 / 0.93 / 0.99 | **167** | 0.71 |
| E11-06 P1-60K | 1,438 | 128, 107 of them at running 11 | 0.40 / 0.50 / 0.69 | 0 | 0.66 |
| E12 rev 5 low | – | – | 0.28 / 0.36 / 0.47 | 0 | 0.36 |
| G4B | 710 | 194 | 0.20 / 0.35 / 0.44 | 0 | 0.70 |

Other scan results:
- One log at 22 running (10-17) had queue ≥ 1 on 230 lines.
- Retract lines: G4B 1, E12 EVAL30 5, E11-05 1.

### Server flags read from the G4B log (15:10-15:18)

- `--chunked-prefill-size 4096`
- `--max-mamba-cache-size 56`
- `--mamba-radix-cache-strategy extra_buffer_lazy`
- `--max-running-requests 13`
- `--mem-fraction-static 0.83`
- `--schedule-policy lpm`

KV pool / chunk = 866,744 / 4,096 = **211.6**, versus a mamba cache of **56**. This is the SGLang #36935 regime.

### SGLang 0.5.21 source facts verified

- `metrics_collector.py:88-92`:
  - `token_usage` = max(full, swa, mamba) ("FIXME: misleadingly named");
  - `full_token_usage` = the full-attention KV pool.
- The E11 snapshot exposes `full_token_usage` as `e11:kv_cache_usage(sglang:full_token_usage)` (`build_e11_05_s49_sglang.py:668-677`). The builder picks `pick("kv_cache_usage") or pick("full_token_usage") or pick("token_usage")` (`build_e12_duck.py:812`).
- `usage_processor.py:13-15`: `PromptTokensDetails(cached_tokens=count) if count > 0 else None`. With `--enable-cache-report`, a **null cached field means exactly 0**.
- Null-cached requests: G4B 20/180 (11 %), rev 5 xhigh 6/127, rev 5 low 23/449 (about 5 %).
- The FP8 KV "scaling factor 1.0" warning appears 8 times in rev4 EVAL6, rev5 xhigh and rev5 low as well. It is not new.

### SGLang issues fetched (15:10)

- **#36935** "[Prefix Cache / Hybrid] Branch reuse silently collapses under LRU pressure when (kv_pool_tokens / chunked_prefill_size) > max_mamba_cache_size". Open, opened 2026-08-29, seen on v0.5.17-cu130, "persists in the latest version".
- **#22935** "[Bug] Mamba radix cache can turn fresh prefill prefix hits into 0-hit after split". Open, opened 2026-04-16, linked PR #38625 (under review), tested on v0.5.9.

### Rev 5 cache denominators

- Aggregate `usage_all.cached_pct`: xhigh **84.3 %**, low **88.7 %**.
- The README had used request-level 85.4 % / 91.0 % (wrong denominator).
- G4B deficit on the same denominator: −9.3 / −13.7 points.

### Bundle and test results (13:17-13:49)

- Smoke suites run from inside the bundle copy: adapter 41/41, rev5 56/56, rev6 104/104.
- Harness: g4b 46/46 (144 s), r6 45/45 (69 s). Replays matched the recorded results.
- Zip: 15.5 MB, 370 files, 16 G4B transcripts. The only remaining absolute paths are in `build_e12_duck.py` and `history/build_e12_duck_rev6c.py` (the builder, not tests).

### Disk (14:22-14:28)

- C: free went from 0.17 GB to **8.85 GB**.
- Moved: HF cache 6.75 GB (including the aicrowd whestbench dataset, 5.95 GB), pip 1.49 GB, uv 0.28 GB.
- Kept on C:: codex-runtimes 1.35 GB, puppeteer 0.69 GB, Claude temp.
- D: had 73.5 GB free.

### Pasted fix-list facts (context; dated in the doc)

**Rev 5 runs**
- EVAL6R5 xhigh (`f4fed1ac`): 4/6 tasks, 5/8 inputs.
- EVAL12R5_LOW (`26fff41e`): 7/12 tasks, 10/19 inputs. Submitted as ref **57456562** (LB pending).
- Rev 5 low: 1.744M tokens in 6,570 s at 6 streams, about 265 tok/s.
- Rev 5 median completion per request 1,115 / 1,090 tokens. Rev 5 median prompt 51.9k / 41.6k.

**E12 EVAL30 (earlier)**
- 40 request timeouts (not 4) in 21 of 31 branches.
- Re-prefilled 8.4M tokens.
- At 14,247 s: 11 running, 8 queued, gen_throughput 0.0.
- Overrun **679 s** (not 1,412 s).

**Engine and KV**
- Cold prefill about 1,400-1,650 tok/s (E11).
- Decode at 40-60k context about 25 % slower (323 vs 429 tok/s).
- DFlash acceptance 4.43 → 3.77 over context.
- D66: 12 slots at context 155,648, KV peak 90 %, 0 preemptions.
- W1 KV assumed 881,336. Window 73,444 per stream at KV fraction 1.0.
- Mamba slots: 4 per request × 12 = 48 ≤ 56.
- CUDA-graph list 1-8, 10, 12.

**Parity history**
- Under batching, spec vs plain shared only 152-166 of 512 tokens (E11-03).
- An unchanged repeat scored 0.1875 vs a 0.50 threshold (Deep Audit).

**Prefix-cache levers (E11-06 etc.)**
- Stable order raised cache from 11-18 % to 87 %.
- An abort without waiting: 0 % vs 52-99.7 % with waiting.
- No tools list: 15 % vs 99 %.
- A template conditional: 75 → 45 %.

**Coverage arithmetic**
- 240 × 60k = 14.4M tokens = 10.81 h at 370 tok/s, versus about 11.46 h available after a 1,463 s solve start.

---

## 3. Failures, mistakes and root causes

| When | Failure | Root cause | Who found it |
|---|---|---|---|
| G4B run (11:33-12:32 UTC); explained 12:59-13:01 | **Capacity probe stepped 12 → 10 → 8 → 6 within about 13 min at KV 13-31 %** | The rule "queue > 0 in 3 consecutive probes = pressure" is wrong for SGLang at 12 in flight: "11 running + 1 queued" is the steady state (the known 11-of-12 behaviour, seen in E11-05, E12 EVAL30, E11-06 and G4B). Second flaw: no cooldown. "allowed" only affects new admissions, so the probe re-triggered before its step took effect. | Assistant, in the G4B analysis |
| Same (13:01) | Why tests and reviews missed it | **All tests used a fake server whose "healthy" state had queue = 0, a number the assistant assumed, not one from a real log.** The "healthy keeps 12" test and the harness "no step-down" check only verified that assumption. The GPT audits saw code and test logs, not real server metric series, so they asked for *more* triggers (Q1: retraction, KV ≥ 0.98), stacking layers on a wrong base. The assistant had even made a bundle about this behaviour for GPT (`E11_05_ADMISSION_11_OF_12_FOR_GPT`) and never replayed the rule on our own logs. **The same mistake happened a second time**: E11-03 (2026-10-17) had a fake server that did not model the real cap. That lesson was applied only to "how many the server runs", not to "which signal the controller decides on". | Assistant's own admission |
| 13:49 (correction 1) | "show() crop loop" overstated | A crop loop happened in only 2 branches. The real pattern: one or two long thinks per branch, and 7/12 b0 branches wrote no transform. | Assistant (full recount) |
| 13:49 (correction 2) | Print-guard gap | `_GRID_ROW_RE` does not match index-prefixed rows (`print(r, row)` → `0 [0, 0, 5, …]`), so 173 rows leaked in 3 branches. GPT's regex catches 173/173 but misses `0: [..]`, `row 3: [..]` and tuples. A widened regex catches all but tuples (tuples not seen in the run). | Assistant, confirmed by both GPT audits |
| 15:18 | README cache comparison wrong | It used request-level rev 5 cache (85.4 / 91.0) against G4B's aggregate 75.0. The correct aggregate is 84.3 / 88.7. | GPT audit 1 (M1-C) |
| 15:10-15:18 | Report counting bugs | `accepted_candidates` = 4 counts check + submit of the same program twice (`_r6_metrics_more`, `build_e12_duck.py:1037-1044`); the real number of distinct programs/predictions is 2. `overfit_flags.fixed_offset` = 2 is both from b5ca7ac4_b0. `first_check_tokens_median_censored` 33,690.5 includes no-check branches at their totals. | GPT audits (M1-A, M1-B) |
| 15:18, retracted 15:33 | "12 running is slower (194-258 vs 370-406)" | All 29 lines at exactly 12 running were in the first 13 min (admission/prefill transient). Not a 12-stream limit. Withdrawn. | GPT audit 2; assistant retracted |
| 15:18, weakened 15:33 | "#36935 ratio 211.6 vs 56 explains the cache misses, stronger than GPT says" | Mamba usage never exceeded 0.70, so the slots were never full. The explanation is weak; the cause is unproven until G4-8. | GPT audit 2; assistant weakened its claim |
| 15:29-15:33 | GPT 2's claim "14/15 low-cache requests have a null cached field, so 0-hit is not confirmed" | **GPT was wrong.** SGLang omits the details when cached == 0, so null = 0 and the 0-hits are real (G4B 11 % vs rev 5 about 5 %). | Assistant |
| 15:18 | GPT 1's "nudge after 2-3 read-only calls" | Wrong threshold: the solved branches wrote `transform` at calls 15 and 18, and failures had 8-14 calls, so a call count does not separate them. A token-based trigger is needed, replayed on bf45cf4b_b0 / b5ca7ac4_b0. | Assistant |
| 13:27-13:45 | Tool breakage | C: was full (0 GB), and tools broke twice while writing. A Bash heredoc EOF error, a SyntaxError in an inline script (`'\'`) and an IndexError in an inline script were fixed and rerun. A bundle assert (`facts_size.py` replacement count 3) was fixed. | Assistant |
| Pasted doc, rev 6 build | Regex hang (critical) | Overlapping separators `\s*,?\s+\d+ \| \s*,\s*\d+` backtracked exponentially, so the smoke hung for more than 11 min. Fixed with disjoint separators and a skip for rows over 600 characters. | Assistant, while testing |
| Pasted doc | Wrong opener after a rejected submit; caps metric counted trims; CRLF corruption of `arc2_duck.py`; two workers driving the same agent (24 concurrent entries in the negative control); fair-share starvation (one task took 792k); shared nested lists in published records mutated by stuck workers (now deep-copied); isolation markers that were substrings of each other (now `ZQ{i:02d}X`); timing-flaky Windows scheduler tests | Each fixed and tested in rev 6 / 6b / 6d | Assistant |
| Pasted doc | Earlier wrong claims | "No failed branch got a train pair exact" (wrong); "the newest block always stays" (wrong for tool-less replies); "temperature 0.6 is Qwen-like" (wrong; the official thinking setting is T 1.0, top_p 0.95, top_k 20); "EVAL30 request timeouts: 4" (really 40); "1,412 s overrun" (really 679 s: the relative deadline 13,367 was subtracted from the absolute 14,779); "the metrics filter drops retraction" (that was the vLLM filter; the SGLang `_E11_SNAP_KEYS` keeps it); "G4B predicts the LB/eval ratio" (it is only a screening proxy); "conditional branch 1" (it is prioritised by evidence class). | Various |
| Pasted doc (T8) | EVAL30 deadline overrun mechanism | `request_timeout_seconds = min(analyzer_timeout, remaining)` was computed once per turn (`arc2_duck.py:835-838`) and reused for every request in the turn. A request that starts just before the deadline therefore carries the turn-start timeout (e.g. 419.6 s read timeout at 14,436 s). The maximum overrun is about one analyzer timeout (1,800 s in rev 5). | Assistant |
| Pasted doc | PARITYR6 deadline bypass | `urlopen(req, timeout=900)` with no deadline (`build_e12_duck.py:852`); 24 probes could hold the session up to 6 h. The relaunch wait had a `max(60, …)` floor. Not fixed (items 6-8 not taken), so PARITYR6 stays NO-GO. | GPT audit C6-1 |
| Pasted doc | Isolation preflight failed open | `isolation_ok = not leaks` gave PASS on 0/12 own markers; an exception gave `None`, which kept 12 workers. Fixed in rev 6d with three states. | GPT audit C6-2 |
| Pasted doc | Unbounded final join | `t.join(timeout=max(0, deadline − now) + 120)` per thread meant up to 24 min. Fixed with one `join_until = deadline + 60`. | GPT audit C6-4 |

---

## 4. Decisions and reasons; rejected options

**Capacity control** (decided 13:10-13:11, user)
- Decision: replace the whole probe with "KV ≥ 0.95 → no new admission; workers fixed at 12; fail-open on a missing metric, with a report entry". The step-down, queue, retraction and streak code is to be removed.
- Evidence: replayed on 5 real logs. It would never hold in the 4 normal runs, and would hold only in E12 EVAL30, the real KV-thrash run (167 lines ≥ 0.95).
- Mamba is not gated: max 0.71 in all logs.
- Rejected: the assistant's own proposed fix (queue ≥ 2, or queue ≥ 1 with KV/mamba ≥ 0.85, plus a cooldown), as too complex and built on the wrong signal.
- Both GPT audits endorse KV-only, with:
  - a 5 s poll, which is an engineering value, not benchmarked;
  - the exact metric name `sglang:full_token_usage` (max over TP ranks); no fallback to `token_usage`;
  - a bounded retry, then fail-open with a warning and a counter;
  - running requests are never cancelled.

**Representation and nudges**
- Keep the facts view, but stop `arc_overview()` repeating the system facts (G4-5) and allow a bounded, indexed whole-grid `show()` (≤ 30×30, token-checked, G4-4).
- Nudge: **token-triggered**, not call-count-triggered (G4-6). GPT's candidate-first prompt (G4-10) is acceptable only as advice combined with G4-6, because prompt-only rules failed before.

**Branch 1**
- b1 first-request cap 24,576; b0 stays at 12,288 (G4-2). This does not reduce the thinking budget.
- Not removed permanently (both GPTs).
- G4-11 (compact indexed b1 view) and G4-12 (refine a partial b0 before starting b1) are **deferred**: big design changes with weak evidence. G4-12 needs the 240-task simulator.

**Engine**
- Keep 12 workers, not 11. The 12-running slowdown was not established.
- Not now: `--disable-radix-cache`, a mamba-strategy switch, kernel patches (G4-9). Changing `--max-mamba-cache-size` / `--chunked-prefill-size` only after the G4-8 measurement.
- FP8 KV scaling warning: not a blocker (present in every earlier run).

**Next run**
- GO / NO-GO: unmodified EVAL12R6 is NO-GO (both audits).
- The consolidated corrected run is a CONDITIONAL GO: G4-1 and its regressions must pass, as one bundled GPU run, not per-change runs.

**Disk**
- Caches were moved, not deleted.
- Other projects' Claude scratchpad folders on C: were not touched; the user decides.

**Earlier decisions in the pasted doc**
- KV fraction 1.0 instead of 0.9 for 12 streams: 0.9 gives 41.0k history at the 24,576 cap, below the rev 5 median prompts.
- The p90 last-admission cutoff is built but off by default: deadline clamps already exist, and it would idle 12 workers for about 15 min.
- Target-only parity needs a separate launch and is never run alongside production.
- Mamba 56 and the CUDA-graph list are unchanged (enough for 12).
- Rejected in rev 6: rolling summaries, stripping reasoning, forced `tool_choice` (Exp 4 HTTP 500; SGLang #43514), REFRAIN and text early exits, penalties under DFlash (#28180: inactive in 0.5.21), NVARC as attempt_2 filler, images, the CPU rule library, `ARC3_PREFER_TOOL_CALLS`.

---

## 5. Open items at the end of this part (15:33)

1. **No user decision yet** on G4-1..G4-13. The assistant asked "কোনগুলো বানাব বলুন" ("tell me which to build"). Nothing was built and nothing pushed.
2. G4-1 tests are promised but not done:
   - a replay of the 7 recorded logs;
   - KV 0.94 → 0.96 → 0.97 → 0.92 admission sequence;
   - None / NaN / exception → fail-open;
   - deadline while held;
   - the 240-task sim.
3. G4-8, the prefix-cache diagnostic: not run. The root cause of the 15 cold later requests (G4B 0-hit rate 11 % vs 5 %) is **unexplained**. The mamba #22935 / #36935 hypothesis is unproven for 0.5.21 + `extra_buffer_lazy`.
4. Target-only DFlash parity: never run. PARITYR6 is NO-GO (C6-1 deadline, C6-3 tool name, long probes and a tool-result continuation probe not built).
5. Submission **57456562** (EVAL12R5_LOW): LB pending.
6. SGLang #38009, #36548, #37475, #31954, #39087 and #43514: not verified by the assistant.
7. EVAL30 `gen_throughput` 0 with 11 running at 14,247 s: not explained (probably KV thrash).
8. Regex: the tuple form of indexed rows is still not covered.
9. A cold start of 1,197 s vs about 700 s before: noted, not addressed in E12.
10. Self-contained test bundle (item 7): partly addressed (tests run from the zip), but the builder still has absolute paths.
11. C: drive: other projects' scratch folders (`d--Soikat…`, `d--Machine-Learning-…ARC-AGI-3…`) are left for the user to decide. VS Code / Claude Code must be restarted for the new TEMP to take effect.
12. Whether facts view is a net win: unproven (GPT 1 and 2).
13. The report contains legacy DIRECT config fields that do not describe the Duck run (see section 6). Not raised as an item.

---

## 6. Insights a fresh reader might miss

- **The core lesson of this part:** a controller tested only against a fake "healthy" server tests the author's assumption, not reality. GPT audits that read only code amplify that error (the Q1 audit asked for more triggers). This is the second occurrence after E11-03. The memory `verify-real-concurrency-in-benchmarks.md` was updated to say "replay controllers on REAL recorded metrics".
- **"11 running + 1 queued" is SGLang's normal state at 12 in flight with `--max-running-requests 13`.** It shows up in every 12-stream log (E11-05 2,615/2,816 lines, E12 EVAL30 3,127/4,153). Any signal based on `num_queue_reqs` is meaningless as KV pressure. GPT 2: "waiting queue and memory pools are measured separately".
- **`token_usage` is not KV usage** in SGLang 0.5.21. It is max(full, swa, mamba), and with mamba at 0.62-0.70 it would read high. Only `full_token_usage` is the KV pool.
- **A null `cached_tokens` means 0** under `--enable-cache-report` (SGLang omits the details when the count is 0). Any analysis that treats null as "missing" understates the real cache regression.
- **The KV pool on Kaggle was 866,744 tokens, not the assumed W1 881,336.** The GPT-reported per-stream window 72,228 = 866,744 / 12, versus the 73,444 planned. The planning number differs from the runtime number.
- **An earlier Python call does not mean an earlier useful hypothesis** (GPT's main lesson). The first call came after 46 tokens, yet 7/12 b0 wrote no transform. First-tool latency is a misleading metric; use first-transform tokens and candidate yield.
- **Successful branches are slow to write transforms** (calls 15 and 18). Any call-count nudge would hit the winners too.
- **b1 grids view + a 12,288 first cap guarantees a cut first request** (first call at 12.4-18k). The cut was structural, not behavioural.
- **Uncached prefill (669,980) exceeded generated tokens (572,035)** in G4B. Prefill is a first-order cost, and the 15 cold requests in 9 branches are not explained by trims (only 2 branches trimmed). This part did not know a later finding (in E14 context): Duck's normalization deleted blank lines from the reasoning, so the previous reply was re-prefilled. That is a plausible explanation for exactly these "cold later requests without a trim". It was never tested here.
- **The phase matters for throughput comparisons:** the 12-running lines were all in the admission/prefill transient (42-497 tok/s alternating). Comparing 11 vs 12 running across different phases gave a false "12 is slow" conclusion, later withdrawn.
- **Contradictions and reversals in this part:**
  - the assistant said the bug was not the main cause of 2/19; GPT 1 ranked it P0 and GPT 2 ranked it #2 behind candidate production;
  - the assistant first called the mamba evidence "stronger than GPT says", then weakened it;
  - the fixed probe design was abandoned for the user's simpler rule.
- **The G4B report's `configuration` block shows `SAMPLING temperature 0.6`, `MAX_MODEL_LEN 106496`, `PROMPT_TOKEN_LIMIT 93952`, `THINK_BUDGET 60000`, `ANSWER_MAX_TOKENS 8192` (L1689-1697).** These are legacy DIRECT-pipeline fields. The Duck agent ran at T 1.0 (`E12_DUCK_TEMPERATURE`), with the rev 6 caps and window. A reader of the report could take the wrong settings as the active ones. It belongs to the same class as the documented header contradiction ("128 Ki context … T 0.7" donor defaults).
- **Gate 4b's 2/19 is a screening on 12 selected tasks** (EVAL6 + 6 informative EVAL30 tasks), not an LB projection. Small-sample accuracy is noise: identical code gave 2, 2, 3, 1 of 22.
- **The FP8 KV scaling-factor 1.0 warning** is in every run since rev 4 (8 lines). It is a baseline, not a regression.
- The GPT audits' own test runs were environment-limited (no `tokenizers`; a 30 s execution limit). Their "104/104" is a recorded result, not an independent re-run.
- **Times:** the chat timestamps are UTC; the user is at UTC+6.

---

## 7. Relevance to the current E14-A notebook (concrete checks and fixes)

1. **Admission gate in the 25-min Duck run (and any concurrency controller): KV-only, exact metric.**
   - Confirm that the rev 6g solver used in E14-A has **no residual queue/retraction step-down** (the rev 6d `probe_loop` at `arc2_duck.py:2395-2427` and `_e12_capacity_probe`, `build_e12_duck.py:802-816`). It must use only G4-1: 12 workers fixed, hold new admissions while `sglang:full_token_usage` ≥ 0.95, a 5 s poll, a bounded retry, then fail-open with a counter.
   - The pick must be the exact name. **Drop the `pick("token_usage")` fallback**: it is max(full, mamba) and mamba runs at 0.62-0.70.
   - Evidence: G4B 12 → 6 at KV 0.13-0.31. 5 real logs: the rule holds only in E12 EVAL30.
   - Guard: replay the recorded G4B / E11-05 / E11-06 / E12 EVAL30 metric series through the controller offline. Expect no hold, except during the EVAL30 KV ≥ 0.95 periods.
2. **Baseline windows at 12 running and the `--min-free-slots-delay 1` arm: verify real running == 12 from the decode log, and exclude the admission/prefill transient.**
   - In G4B, all 29 lines at exactly 12 running were in the first 13 min, swinging 42-497 tok/s.
   - The steady state without the fix is 11 running + 1 queued (E11-05 2,615/2,816 lines; G4B 101 of 188 queue-1 lines at running 11).
   - So the arm's success metric should be: the share of decode lines at `#running-req: 12` with `#queue-req: 0`, measured in a steady-state window at matched context. Not tok/s from the first minutes.
   - Reference numbers to compare against:
     - 11 running: 370-406 tok/s (median 370.9 at 20-40k context);
     - 6 running: 321.4 tok/s;
     - G4B aggregate: 277 tok/s, accept_len 4.2.
3. **Cache and token-level replay accounting: treat a missing `prompt_tokens_details.cached_tokens` as 0, but only if `--enable-cache-report` is on.**
   - Source: SGLang 0.5.21 `usage_processor.py:13-15`.
   - The adapter stores `cached = det.get("cached_tokens") if isinstance(det, dict) else None` (`arc2_duck.py:1200-1203`). Make sure E14 reports count null as 0 and assert the flag is in the launch command. Otherwise the cache % is overstated.
   - Baselines on the same aggregate denominator: rev 5 84.3 % (xhigh) / 88.7 % (low); G4B 75.0 %. 0-hit requests: G4B 11 %, rev 5 about 5 %.
4. **E14 "Duck history keeps raw reasoning" fix: validate it against exactly the G4B symptom.**
   - G4B had 15 later requests below 5 % cached in 9 branches, with trims in only 2.
   - Add a check that request n+1's token prefix equals request n's prompt + reply, except after a trim (the R6-D2 invariant). Log per request: cached_tokens, prompt_tokens, trimmed flag and mamba usage. This is G4-8's per-request instrumentation, which separates "the prefix changed" from "the cache state was lost".
5. **Int8 mamba-checkpoint arm and mamba cache size: record the #36935 ratio.**
   - The G4B config: kv_pool / chunked_prefill = 866,744 / 4,096 = 211.6, against `--max-mamba-cache-size 56`, with `extra_buffer_lazy` and `lpm`.
   - Any arm that changes the mamba checkpoint memory or `--max-mamba-cache-size` should log mamba usage (G4B max 0.70) and prefix-hit rate, so the effect can be attributed.
   - G4-9 (GPT and the assistant): do not change the mamba strategy, disable the radix cache or patch kernels without the replay measurement.
6. **Read the KV pool size from the server log at runtime; never hard-code 881,336.** The G4B run's real pool was 866,744 (window 72,228 per stream at 12), so any per-stream window or KV math in E14 should use the server's reported `max_total_num_tokens` (or equivalent).
7. **Greedy parity (A/A, concurrency 12, target-only): apply the PARITYR6 lessons.**
   - (a) **Every probe and the relaunch must respect one absolute deadline.** PARITYR6 had `urlopen(timeout=900)` × 24 probes = up to 6 h, and a `max(60, …)` floor on the relaunch wait (C6-1).
   - (b) Compare the tool **name and arguments** (C6-3).
   - (c) Use probes long enough to reach a tool call (12,288 thinking-on, plus one tool-result continuation probe). 2,048-token probes rarely reach a call (first reply p50 33.6k).
   - (d) Run the A/A same-config control first; judge at batch size 1 or on task-level outcomes. Under batching, spec vs plain shared only 152-166 of 512 tokens, and an unchanged repeat scored 0.1875 vs a 0.50 threshold. Token equality at concurrency 12 is not expected even A/A.
   - (e) Target-only is a separate launch (kill, then relaunch without the 4 `--speculative-*` flags), never alongside production. A same-server control 2/2 does **not** establish target-only equivalence (GPT 2).
8. **Deadline safety in the 25-min Duck run and the measurement phases.**
   - Confirm the per-request clamp. The T8 root cause was a timeout computed per turn and reused per request, so the overrun can reach about the analyzer timeout (1,800 s). In a 25-min window that would blow the 200-min profile budget.
   - Also confirm verification is clamped to deadline − margin, and there is one absolute join cutoff (deadline + 60 s, not per-thread +120 s).
   - G4B proof that these worked: 91 clamps, 0 stragglers, finished 9 s early.
9. **Launch watchdog and cold start.**
   - G4B SGLang was ready only after **1,197 s** (rev 5 about 700 s), with solve start 1,463 s. This matches the 10-21+ GCSFUSE slowdown, against a 30-min launch timeout. That supports the E14 `/dev/shm` ranged copy and the stall watchdog.
   - Record "ready_sec" and the copy time in the phase record, so E14 can show the gain against 1,197 s.
10. **Isolation preflight: three-state** (rev 6d C6-2). 12 workers only on PASS (own == n and no foreign marker); FAIL or INCONCLUSIVE → 6 plus a warning. Make sure the E14 latch/isolation test does not fail open on exceptions (the old `None` → kept 12).
11. **Report metrics (E14 Duck run report):**
    - Dedupe accepted events into unique programs and distinct predictions: check + submit of the same program = 2 events, 1 program (G4-7 / M1-A).
    - Report the observed first-check median, the no-check count and the censored proxy under separate names (M1-B).
    - Use aggregate cache denominators (M1-C).
    - **The G4B report's `configuration` block shows the legacy `SAMPLING temperature 0.6 / MAX_MODEL_LEN 106496 / THINK_BUDGET 60000`**, which are not the active Duck settings (T 1.0, rev 6 caps, the KV window). E14-A's report should record the actual Duck sampling and caps (or label the legacy fields), so nobody reads 0.6 as the run's temperature.
12. **Print guard in the rev 6g solver: check that G4-3 is present.** The indexed-row regex must cover `N [..]`, `N: [..]` and `row N: [..]`; tuples are still open. Without it, 173 rows leaked in G4B. Also check the regex stays linear-time: the rev 6 regex hang froze for over 11 min on long nested-list lines; the fix uses disjoint separators and skips rows over 600 characters.
13. **The b1 first-request cap** (if b1 runs in the 25-min Duck run): use 24,576, not 12,288 (G4-2). All 4 G4B b1 first requests were cut before a call (first call at 12.4-18k), wasting 21.6 % of the tokens.
14. **The FP8 KV "scaling factor 1.0" warning** (8 lines per run since rev 4) is a known baseline. The E14 log checks should not flag it as a new failure. They should flag a change in its count.
15. **Local harness and scratch:** C: was at 0 GB, and the user env vars now point TEMP/TMP/HF_HOME/PIP/UV to `D:\DevCache`. E14 CPU tests and venvs (`_scratch/e13_venv`) should stay on D:. The `tokenizers` package is required for the B3 estimate tests (4 rev 5 tests fail without it).
