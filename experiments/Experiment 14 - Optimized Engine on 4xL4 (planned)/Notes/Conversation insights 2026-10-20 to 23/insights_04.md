# Insights from part_04 (2026-10-21 10:00 to 11:26)

## Scope and structure of this part

- 3,272 lines, all read. Most of the part is two IDE selections of the running doc `experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Review/E12 next fix list (running, 2026-10-20).md`:
  - lines 2-1225: doc lines 1-1220 (selection sent with the 10:00 message);
  - lines 1639-2931: doc lines 1-1290 (selection sent with the 10:36 message). It repeats the first selection and adds the two new sections the assistant wrote at 10:32-10:34: "GPT final pre-GPU audit of rev 6c" and "GPT audits 2 and 3 of rev 6c".
- The live conversation itself is short: 5 user turns (10:00, 10:03, 10:06, 10:32 GPT paste, 10:36) and the assistant's work up to 11:26 (rev 6d patch set 1-5, built and CPU-tested, not pushed).
- The doc content is history from earlier parts (rev 5 / rev 6 / 6b / 6c). It is summarised below only where it carries numbers or rules that still matter. Items taken from the doc are marked "(doc)".

---

## 1. User instructions, rules, preferences and decisions

| Date/time | Verbatim (Bengali) | English meaning / effect |
|---|---|---|
| 2026-10-21 10:00 | "ARITYR6 f3876b40 DFlash ছাড়া parity: 4টা task, কোনো solve নয়। … এটার জন্য কি ক্ষতি হচ্ছে?" | "PARITYR6 … parity without DFlash: 4 tasks, no solving … Does this cause any harm?" A question, not a decision. Assistant: no score harm; it only costs GPU quota (≈45-60 min); it is optional. |
| 2026-10-21 10:03 | "মানে মোট কয়টা নোটবুক রান দিতে বলছো? দুইটা নাকি?" | "So how many notebooks are you asking me to run in total? Two?" Assistant: two are needed (G4B, then EVAL12R6); PARITY is optional. |
| 2026-10-21 10:06 | "জিপিটিকে অডিট করাই,লিঙ্ক দিও" | "I'll have GPT audit it, give me the link." Decision: an external GPT audit **before** any push. The assistant gave the zip path and a ready prompt. |
| 2026-10-21 10:32 | (pasted GPT report "E12 Rev6c — Final Pre-GPU Audit", Bengali + English) | The user's way of working: paste GPT audits; the assistant validates every claim against the code. |
| 2026-10-21 10:36 | "OK ১ থেকে ৫ করে bolio," | "OK, do 1 to 5 and tell me." **Approval of patches 1-5 only**: (1) isolation in three states, (2) one shared join cutoff, (3) stream step-down on retraction/KV pressure, (4) two smaller facts levels + over-target flag, (5) p90 fix, `stage1_left`/`tasks_exposed`, header text. **Not approved:** item 6 (the PARITY fixes) and item 7 (a full zip with tests). So PARITYR6 stays NO-GO. |

Standing rules visible in this part (from the doc, all still in force):
- "Rule (user, 2026-10-20): Read every GPT report in full and validate every claim. Add the validated items here. Change no code until the user says so." The assistant followed it: it validated the GPT audit (10:32-10:34) and changed no code until the user said "OK ১ থেকে ৫" (10:36).
- (doc) User correction 2026-10-21: "Those were different experiments. That something did not work there does not mean it will not work here." Past negatives are context for safeguards, never verdicts.
- (doc) Rev 6c approvals: "Gate 4b: permission granted. Take 12 streams. Permission for the rest too."
- (doc) The user approved evidence-based output cuts (2026-10-21) for the 49,152 cap. The 12,288 request cap needs a new decision per run: "12,288 = approved exception to the 60k floor, user 2026-10-19" applied to rev 1 / EVAL30 only.
- (doc) Coverage rule: "Duck runs on every task (full coverage)", which supersedes the "Duck as a Poetiq complement" idea (F7).
- (doc) Bundle changes ("one bundled run, evidence-based changes only"). The GPT audits keep asking for isolated steps; the doc records this as the user's decision.
- No push without a go: every assistant summary ends "আপনি go দিলে push করব" ("I'll push when you give the go"). At 11:26: "কিছু push বা submit করিনি" ("I pushed or submitted nothing").

---

## 2. Measured facts and results

### Runs and scores (doc, earlier GPU runs)
- **EVAL6R5 xhigh** (sha `f4fed1ac…`): 4/6 tasks, 5/8 inputs.
- **EVAL12R5_LOW** (sha `26fff41e…`): 7/12 tasks, 10/19 inputs. Submitted as **ref 57456562 (pending)**.
- EVAL30R5 (`415a68f2…`): not run.
- Earlier rev 4: EVAL6 actual 3/6 tasks, 4/8 inputs; the real-sandbox F1+F5 replay gives 5/6, 7/8. EVAL30: 4 tasks, 7/48 (branch 1 never ran).
- Noise: identical code gave "2, 2, 3, 1 of 22", so EVAL6/EVAL12 accuracy is a sanity check only.
- LB/eval ratios: 1.06 / 0.62 / 0.44 (49k / 86k / Poetiq).
- Selective branch-1 replay: xhigh 5/8 at 498,253 tokens vs 846,080 (−41.1 %); low 9/19 at 1,094,372 vs 10/19 at 1,743,889 (−37.2 %). The lost output is 135a2760.
- Late corrections: 135a2760_b1 71,306 → 97,972; 898e7135_b0 56,868 → 73,523; 898e7135_b1 77,859 → 79,662.
- Rev 5 per request: median completion 1,115 (xhigh) / 1,090 (low); median prompt 51.9k / 41.6k; cached 85.4 % / 91.0 %; uncached ≈888k / 1.577M.
- First check at a median of 55.1k (xhigh) / 53.7k (low). The rev 5 xhigh first reply has a p50 of 33.6k tokens before any tool call.

### Throughput, KV and engine numbers (doc + live)
- Rev 5 low: 1.744M generated tokens in 6,570 s at 6 streams ≈ **265 tok/s**. Full-coverage tokens per task: 11.3 h × 3,600 × 255 / 240 ≈ 43k.
- W1 KV pool: **881,336 tokens**.
  - 12 streams at fraction 1.0 → 73,444 per stream, 48,356 of history at the 24,576 cap (60.6k at 12,288).
  - Fraction 0.9 → 66,100 / 41,012.
  - Fraction 0.95 → 69,772 / 44,684.
  - Worst case 12 × 73,444 ≈ 881,328, i.e. no spare KV.
- D66: 12 slots at context 155,648, KV peak 90 %, 0 preemptions.
- Decode at 40-60k context is ≈25 % slower: 323 vs 429 tok/s. DFlash acceptance drops 4.43 → 3.77 as context grows.
- Cold prefill 1,400-1,650 tok/s (E11), i.e. ≈30-35 s per 50k resume.
- Engine config: `SGL_MAX_RUNNING` 12, server max-running 13 (the 11-of-12 workaround), CUDA-graph batch list 1-8, 10, 12 (9 and 11 pad up), mamba cache 56 slots at 4 per request (12 × 4 = 48 ≤ 56), fp16 SSM.
- SGLang 0.5.21 exports `sglang:num_retracted_requests_total` (`observability/metrics_collector.py:474-476`; the scheduler sets it at `managers/scheduler.py:4226`). Found by the assistant's grep at 10:32.
- E12 metrics: `_E11_SNAP_KEYS = ("num_running_reqs", "num_queue_reqs", "token_usage", "gen_throughput", "spec_accept", "cache_hit_rate", …` (cut at 500 characters). `get_metrics` maps `sglang:full_token_usage` to the `kv_cache_usage` name, and `token_usage = max(full, mamba)`.

### T8 deadline overrun (doc, EVAL30)
- The overrun was **679 s, not 1,412 s**: `deadline_in_sec` 13,367 is relative to the solve start; `started_sec` 733; absolute deadline ≈14,100 s; `finished_sec` 14,779.
- Log evidence:
  - ≈16 requests timed out together at 14,113 s.
  - A request with read timeout 419.6 s timed out at 14,436 s, 336 s past the deadline. Another with 469.7 s timed out at 14,492 s.
  - At 14,247 s: `num_running_reqs` 11, `num_queue_reqs` 8, `gen_throughput` 0.0.

### Timings and quota (live)
- Planned runs (10:04):
  - G4B ≈50 min (80 min budget);
  - EVAL12R6 ≈3 h (185 min budget);
  - PARITY ≈45-60 min (75 min budget; 2 launches of ≈10-13 min each);
  - total ≈4-4.5 h for two notebooks, ≈5-5.5 h for three.
- Quota: "সাপ্তাহিক 60 h-এর মধ্যে আগে ~31 h খরচ হয়েছিল" ("about 31 h of the weekly 60 h was already used"). The doc says 30.79/60 and 31.29/60.
- Gate 4b window = 12 × (42,720 − solve start) / 240:
  - hidden solve end 42,720 = 43,020 − 120 − 180;
  - at an 800 s start: ≈2,096 s ≈ 34.9 min (≈175 s per task);
  - harness: 177.9 s per task at a 14 s start.
- PARITY worst case: 24 probes × 900 s = **6 h** of HTTP waiting.
- Join worst case: 12 × 120 s = **24 min** past the deadline. The margin after the solve end is `E12_STOP_BEFORE_HARD_SEC` = 180 s.

### Rev 6d build (live, 10:37-11:26)
- Facts block sizes (Qwen tokenizer, 120 eval tasks):
  - full block: p50 1,237 / p90 1,752 / p99 2,861 / max 2,892;
  - rev 6c bounded: p50 736 / p90 837 / max 1,396, 18/120 over 800;
  - **rev 6d bounded: p50 699 / p90 779 / max 798, 0 over 800**;
  - levels used [(0,3),(1,55),(2,33),(3,11),(4,16),(5,2)];
  - lists dropped in 117/120 tasks;
  - raw grid text p50 4,881 / p90 8,950 / p99 15,828 / max 16,191 tokens; the bounded block is smaller on 120/120;
  - identical under PYTHONHASHSEED 0/1/12345 (hash `2c60dfdbf224d2d7`).
- Tests:
  - smoke_rev6 **104/104 three times in a row** (after intermediate runs at 100-103 PASS);
  - smoke_rev5 56/56; smoke_adapter 41/41;
  - harness EVAL12R6: r6 45/45, rerun 22/22, launchfail 6/6;
  - G4B 46/46; PARITY 19/19;
  - replays rev4 7/8, xhigh 5/8, low 10/19 ("kept_r6<kept_r5: []", "diffs vs 6b: []").
- New shas:
  - EVAL12R6 **`617c75de`**, EVAL12R6G4B **`1bdadd82`**, PARITYR6 **`af5011d9`**;
  - adapter `arc2_duck.py` **`7c60454c`**, embedded == source in all 3 (23 files).
  - Intermediate builds in the same session, superseded: `18963d44`, `7589b4eb`, `af8000da`.
  - Earlier rev 6c shas: `7d69e502` / `f70c3228` / `f3876b40`.
- Machine timing noise: the same scheduler sim took 3-18 s. bench_old 10.72 s and 6.0 s; new code 5.36 s and 14.48 s. CPU load was ≈11-15 %, with VS Code and Chrome as the top processes.
- The harness coverage line (EVAL12R6 r6): "tasks exposed 12 / 12 | stage-1 starts left 0 | workers 12 -> 12 | capacity events [] | stragglers 0". A second scenario printed "tasks exposed None / 12 | stage-1 starts left None | workers None -> None | capac…" (cut at 500 characters; which scenario it was is not visible).

---

## 3. Failures, mistakes and root causes

### Found by the GPT audit (10:32) and confirmed by the assistant (10:34)
1. **C6-1, PARITY deadline bypass (BLOCKER, PARITY only):** `build_e12_duck.py:852` `urlopen(req, timeout=900)` has no deadline; 24 probes → up to 6 h. The relaunch wait at `:874`, `wait_ready(min(SGL_RELAUNCH_TIMEOUT_SEC, max(60, HARD_STOP - time.perf_counter() - 600)))`, is bounded, but `max(60, …)` can still wait past the hard stop. **Not fixed** (the user approved 1-5 only).
2. **C6-2, isolation preflight fails open (HIGH):**
   - `:940` sets `isolation_ok = not leaks`, so 0/12 own markers or empty replies still PASS;
   - `:942`: an exception sets `None`;
   - `:972` keeps 12 workers unless the result `is False`.
   - Root cause: the test only looked for foreign markers, not for each reply's own marker. **Fixed in 6d.**
3. **C6-3:** PARITY compares only the tool arguments (`:855`), not the tool name. Not fixed.
4. **C6-4, the final join has no shared bound (HIGH for 12 h):** `arc2_duck.py:2421` gives each thread `t.join(timeout=max(0, deadline − now) + 120)`. **Fixed in 6d** (one cutoff, deadline + 60).
5. **C6-5:** the facts "800-token bound" was soft: 18/120 over, max 1,396; the last level was returned even when over. Fixed with levels 4 and 5 and a `facts_over_target` flag.
6. **A3-F3:** the p90 index `d2[int(0.9 × (n − 1))]` is not the nearest rank: [100, 200, 300, 400, 800] gives 400 instead of 800. Fixed as `ceil(0.9n) − 1`.
7. **Doc contradiction:** the common notebook header (`build_e12_duck.py:1093-1098` / `:1097`) still said "128 Ki context … T 0.7" (donor defaults) next to the rev 6 settings. Fixed.
8. **A2-3 / A3:** the GPT test bundle was not self-contained (`smoke_adapter.py` and the fixtures were missing; Windows paths). Confirmed by the assistant ("smoke_adapter.py in zip: False | fixtures: False"). Not fixed (item 7 not approved).

### The assistant's own errors in this part
- **Wrong claim at 10:34 (item 4 in the table):** "SGLang 0.5.21 retraction counter … কিন্তু আমাদের metrics filter সেটা বাদ দেয়। এটা আমি নিজে যাচাই করে পেয়েছি" ("but our metrics filter drops it; I verified this myself"). Corrected at 11:26: "সেটা ছিল vLLM সংস্করণের filter দেখে। SGLang নোটবুকে counter আগে থেকেই পৌঁছায়, শুধু probe সেটা ব্যবহার করত না" ("that came from looking at the vLLM version's filter; in the SGLang notebook the counter already arrives, only the probe did not use it"). Root cause: it grepped `_E12V_ALIAS` (the vLLM aliases) instead of the SGLang `_E11_SNAP_KEYS` path.
- **Overclaims, corrected at 10:34:**
  - "G4B দিয়ে LB/eval অনুপাত predict করা যাবে: ভুল" ("G4B can predict the LB/eval ratio: wrong"). The 12 tasks were picked from earlier results, and the queue is 12, not 240. It is a selected screening test, not an LB forecast.
  - "Conditional branch 1" is really branch 1 **prioritised** by evidence class, never skipped.
- **Patch mechanics failed twice:**
  - 10:38: the python string-replace patch failed with "AssertionError … if level == 4" (escape problem). It switched to the Edit tool.
  - 11:00: "smoke ফাইলটা CRLF বলে patch মেলেনি" ("the patch did not match because the smoke file is CRLF"). Edit tool again.
  - The Edit result warned "the file had been modified on disk since you last read it".
- **Replay output files not written (11:2x):** "replay_rev6d_rev4.jsonl: No such file or directory" ×3 and "[: : integer expression expected". The summary hit lines did print (7/8, 5/8, 10/19), but the per-run jsonl files were apparently not saved. Not mentioned in the 11:26 report.
- **A real defect found while fixing C6-4 (new, 11:26):** "publish করা record-এর ভেতরের list চলমান session-এর সাথে ভাগ করা ছিল। ফলে আটকে থাকা worker CELL 7-এর লেখার সময়ও সেগুলো বদলাতে পারত" ("the lists inside a published record were shared with the running session, so a stuck worker could still change them while CELL 7 wrote"). Fix: records are deep-copied when published, and nothing is published after close.
- **Timing-flaky tests:**
  - S1 ORDER2 tolerance (`< 20`, got 20);
  - S3 rounds;
  - the 6c p90 cutoff (`p90_skipped` 0, with durations [0.5 … 1.8] or [0.1 … 3.5]);
  - "S1: no stage-2 work before the last task's first exposure starts" (246712.478 vs 246712.553).
  - They were made "deterministic or tolerant". The p90 cutoff logic was also moved into a function, `stage2_cutoff_hit(left, state["dur2"], min_admit_sec)`. Risk: loosened tolerances could hide a real ordering regression. The S1 failure (a stage-2 start 0.075 s before the last stage-1 start) is a real ordering race under load, not only noise.

### Earlier errors recorded in the doc (context)
- 1,412 s overrun: the assistant subtracted the relative 13,367 from the absolute 14,779; the correct figure is 679 s.
- EVAL30 request timeouts: the assistant wrote "4"; the true number is 40.
- "Qwen temperature 0.6": wrong; the official thinking setting is 1.0.
- "The newest block always stays": wrong for tool-less replies.
- "No failed branch got one train pair exact": wrong.
- Rev 6 build defects:
  - regex catastrophic backtracking (`_GRID_ROW_RE` overlapping separators; the smoke hung for more than 11 min);
  - wrong opener after a rejected submit;
  - the caps metric counted trims, not requests;
  - CRLF conversion of `arc2_duck.py`.
- Rev 6b defects:
  - two workers drove the same agent (24 concurrent entries in the negative control);
  - fair-share starvation (one task took 792k).

---

## 4. Decisions made, with reasons; options rejected

| When | Decision | Reason |
|---|---|---|
| 10:01 | PARITYR6 is optional and kept out of production notebooks | GPT: "never alongside production". It only runs in the PARITYR6 profile in commit mode; the harness shows parity results empty in r6. It is never a submission notebook. |
| 10:04 | Run order: **G4B first (≈50 min), then EVAL12R6** | The short run checks the 12-stream engine, scheduler and KV before the 3 h run. |
| 10:06 | GPT audit before the push (the user's choice) | — |
| 10:34 | KV fraction **1.0 for G4B**. The hidden default (1.0 vs audit 2's 0.95) is decided from G4B's real running count, KV peak and retractions. | 0.9 leaves 41.0k of history, below the median prompts of 41.6-51.9k; D66 evidence. |
| 10:34 | Patches 1-5 recommended before G4B; PARITY fixes only if PARITY is run | Small changes, each with its own test. |
| 10:36 | User: do 1-5 | — |
| 10:37-11:26 | Step-down triggers: queue > 0 for 3 probes, **or** the retraction counter rising in 2 consecutive probes, **or** KV usage ≥ 98 % for 3 probes → 12 → 10 → 8 → 6, with the reason logged | GPT Q1: the queue alone misses eviction/preemption. |
| 11:26 | Facts are never truncated mid-line; emergency levels 4 (shapes) and 5 (counts) with an inspection notice; every TEST is listed | Do not lose train/test facts (GPT: no forced truncation). |
| doc | The stage-2 p90 cutoff is built but **off by default** | Requests and verifies are already deadline-clamped; a 24,576-token chunk at 25-30 tok/s per stream takes 14-16 min, so a cutoff would idle workers. |
| doc | No universal skip of branch 1 | LOW: a wrong first train-exact candidate was later rescued by branch 1 (135a2760). |

Rejected or not taken:
- PARITY C6-1 / C6-3 fixes and the long probes: not taken.
- A self-contained zip: not done.
- An `assert` on the tokenizer fallback: rejected earlier; record only, because an assert would turn a working fallback into a dead solver.
- (doc, "not in rev 6"): rolling summaries; stripping reasoning; forced or required `tool_choice` (Exp 4 HTTP 500 + 59 empty submits; SGLang #43514); REFRAIN or text early exits; penalties under DFlash (inactive in 0.5.21, #28180); images; the CPU rule library; NVARC as the attempt_2 filler.

---

## 5. Open items

1. **Push EVAL12R6G4B `1bdadd82`**: awaiting the user's go (11:26: "আপনি go দিলে … push করব"). EVAL12R6 `617c75de` comes after the G4B results.
2. **PARITYR6 `af5011d9` is still NO-GO**: no deadline clamp per probe or on the relaunch (C6-1), no tool-name comparison (C6-3), no 12,288-token long probes, no multi-turn tool-result continuation probe (A3-Q2). The 11:26 table says: "PARITY-র নিজের fix নেওয়া হয়নি, তাই এখনো চালানো যাবে না" ("PARITY's own fixes were not taken, so it still cannot be run").
3. **Self-contained test bundle** for GPT (`smoke_adapter.py`, fixtures, relative paths): not done.
4. **Hidden-run KV fraction** (1.0 vs 0.95): to be decided from G4B telemetry.
5. **Unexplained:** `gen_throughput` 0.0 with 11 running and 8 queued at 14,247 s (EVAL30). Thought to be KV thrash; not needed for the fix and never proven.
6. **SGLang issues** #38009 (DFlash vs target-only divergence in thinking), #36548 (concurrent state contamination), #37475 (`target_layer_ids` fallback), #31954, #39087 and z-lab/dflash #166: "not verified by me (C)".
7. **Rev 6c/6d is not qualified for the 240-task hidden submission.** Full coverage can be claimed only if `stage1_left == 0`. The first exposure (28k) is soft.
8. **Marginal value of the prioritised branch 1** (`branch1_yield_by_stage2_class`): measured only on GPU.
9. **GPT audits 2 and 3 of rev 6c:** the assistant says at 10:34 "তিনটা GPT অডিটই পুরো পড়েছি" ("I read all three GPT audits in full") and adds a doc section "GPT audits 2 and 3 of rev 6c (same day, user-pasted)". This part shows only one pasted audit (id 1891, 10:32). Audits 2 and 3 must come from outside this extract, or were not captured; their text cannot be checked here.
10. The jsonl replay outputs that failed to write (see §3): not re-run.
11. The harness "None" coverage line in one scenario: not explained (cut at 500 characters).
12. Still pending from the doc: gate 4b interpretation wording in reports; real running max and acceptance verified on GPU; mamba slots and CUDA-graph list unchanged ("permission used, but the numbers show no need").

---

## 6. Insights a fresh reader might miss

- **The vLLM vs SGLang metric-filter mix-up.** The notebook carries two metric paths: `_E12V_ALIAS` for vLLM names and `_E11_SNAP_KEYS` for SGLang. Grepping the wrong one gave a false "counter filtered out" claim. Any metric check must read the path the engine in use actually takes.
- **"Isolation PASS" meant only "no foreign marker".** Empty replies, 0/12 own markers or an exception (`None`) all let 12 workers run. Any concurrency or parity preflight must count positive evidence (own marker present), not just the absence of negative evidence, and must treat exceptions as INCONCLUSIVE → fail safe.
- **Per-turn timeout reuse (T8).** Duck computes the request timeout once per `analyze()` turn and reuses it for every request in that turn. A request started late in a long turn can overrun by up to one analyzer timeout (1,800 s in rev 5). The fix re-clamps per request to the absolute deadline. This is the root cause of the EVAL30 679 s overrun.
- **Shared mutable records.** The published snapshots shared list objects with live sessions. The final writer could serialise data a stuck thread was still changing. Deep copy at publish time.
- **G4B is not an LB predictor.** It gives the same average time share per task, but with a 12-task queue stage 2 starts almost at once; with 240 tasks it waits for 228 first exposures. The 12 tasks are also hand-picked (EVAL12R5 = EVAL6 + 6 informative EVAL30 tasks).
- **"Conditional" branch 1 is prioritised, never skipped.** A task with one verified candidate still gets branch 1 (by design: about a third of accepts are test-wrong).
- **Facts bound numbers changed between messages:**
  - rev 6b claimed "p50 736 / p90 837 / max 1,396";
  - the 10:37 measurement printed "bounded (shipped) p50 695 / p90 779 / max 798" (already with the new level 4);
  - after the edits it is p50 699 / p90 779 / max 798.
  - The doc's earlier "bounded to 3,600 characters" (rev 6) was superseded by token bounding.
- **The CUDA graph and max-running mismatch is easy to miss.** The server runs max-running 13 (the 11-of-12 workaround), but the graph list stops at 12, so a 13-request batch would run outside the captured graphs. The list 1-8, 10, 12 also pads 9 → 10 and 11 → 12.
- **Machine noise produced test failures that looked like logic failures** (3-18 s for the same sim). Some tolerances were loosened. One S1 failure (a stage-2 start 0.075 s before the last stage-1 start) is a real race window under load. The fix made the test deterministic rather than proving the race cannot happen on GPU.
- **Patch tooling:** python `str.replace` patches fail silently-ish (AssertionError) on CRLF files and escape differences. The workflow is: Edit tool, then check LF/CRLF (`file … | grep -c CRLF` → 0), pyflakes, rebuild, sha check, then embedded == source.
- **Approval was partial.** The user said "১ থেকে ৫" (1 to 5) only, so the assistant correctly did not touch PARITY or the zip. A later reader might assume all 7-9 items were done.
- **Reversals:**
  - the p90 last-admission cutoff: proposed (T8), rejected (6b), built but off (6c), index fixed (6d);
  - KV fraction: 0.9 (rev 6), then 1.0 (rev 6c), then an open question for the hidden run.

---

## 7. Relevance to the current E14-A notebook (concrete checks and fixes)

1. **Isolation / greedy-parity preflight must be three-state and fail safe.**
   - Evidence: C6-2 (`isolation_ok = not leaks`, exception → `None`, and `is not False` kept 12 workers).
   - For E14-A's greedy parity (A/A, concurrency 12, target-only) and any marker test: PASS only if own == n and there are no foreign markers. Empty replies, missing markers or an HTTP exception are INCONCLUSIVE and must never be reported as parity or isolation OK.
   - Check that the rev 6g preflight code E14-A embeds has the 6d version (`isolation_ok is True` gates 12 workers).
2. **Every parity probe and the target-only relaunch need an absolute deadline.**
   - Evidence: C6-1, 24 × 900 s = 6 h. The relaunch wait uses `max(60, HARD_STOP − now − 600)`, which can exceed the hard stop.
   - E14-A has a 200-min profile budget and many phases (NCCL bench, baselines, profiler, latch, replay, parity, arms, Duck run). Each probe's `timeout = min(cap, phase_deadline − now − margin)`; no probe starts with less than about 60 s left; the relaunch readiness wait is clamped without a `max(60, …)` floor. Partial results are written on exit.
3. **The parity method from earlier evidence.**
   - Run the same-config control (A/A) first.
   - Judge on task-level outputs (tool call parsed, same tool **name** + arguments, finish reason, error status) and batch-size-1 prefixes, not token equality under batching. Spec vs plain shared only 152-166 of 512 tokens under batching (E11-03); an unchanged repeat scored 0.1875 vs a 0.50 threshold.
   - 2,048-token probes rarely reach a tool call (rev 5 first reply p50 33.6k), so include long thinking-on probes (≥ 12,288) and one multi-turn tool-result continuation probe.
   - The target-only relaunch removes all four `--speculative-*` flags, and is never alongside the production run.
4. **Thread joins and the Duck run's deadline.**
   - Evidence: C6-4, per-thread +120 s → 24 min worst case; T8, the per-turn timeout reuse overran by 679 s.
   - For the 25-minute Duck run: confirm rev 6g has (a) the per-request timeout re-clamp to the absolute deadline, (b) the verify budget = min(2 × python timeout, deadline − now − margin), (c) one `join_until = deadline + 60`, and (d) deep-copied record snapshots and no publish after close.
   - Budget the phase end with the 180 s `E12_STOP_BEFORE_HARD_SEC` margin plus the 240 s admit gate (`E12_R6_MIN_ADMIT_SEC`): no new branch admissions in the last 4 min of the 25-min window.
5. **The capacity step-down must use SGLang's real counters.**
   - Read `sglang:num_retracted_requests_total` and `sglang:full_token_usage` from the SGLang path (`_E11_SNAP_KEYS` / `e11_prom`), not the vLLM alias table.
   - In the "baseline windows at 12 running" and the `--min-free-slots-delay 1` arm, assert real running == 12 from the decode log and record the retraction delta per window. Otherwise an 11-of-12 or retraction-thrash window gets mis-scored. This is the user's standing "verify real concurrency" rule and the 11-of-12 bug.
6. **CUDA graphs 1..13 arm vs the current list.** Today the server allows 13 running (the 11-of-12 workaround) but captures graphs only for 1-8, 10, 12, so batches of 9 and 11 pad up and 13 is uncaptured. The E14-A arm should log which batch sizes actually ran with graphs. Its baseline must use the same max-running (13) so the comparison is fair. Also check graph capture memory against the KV pool, because the pool sets the Duck window (next item).
7. **Derive the Duck window from the KV pool read back after launch, never a hard-coded 881,336.**
   - The int8 mamba checkpoint, DFlash width 6/4 and CUDA-graph arms change memory, so the KV pool changes.
   - Per-stream window = fraction × KV / streams (1.0 × 881,336 / 12 = 73,444); history = window − request cap − 512.
   - Also check mamba slots ≥ 4 × streams (48 for 12; currently 56) under the int8 mamba arm.
8. **Measure throughput at realistic context.** Decode is ≈25 % slower at 40-60k context (323 vs 429 tok/s), and DFlash acceptance falls 4.43 → 3.77. Duck prompts have a median of 41.6-51.9k. Baseline windows, the token-level replay and the DFlash width arms should run at Duck-like context lengths, or report tok/s by context bucket. Short-prompt numbers will overstate the gains.
9. **Prefix-cache invariant check on GPU** (ties to E14's "keep the raw reasoning" fix). The rev 6b fake-server invariant "request n+1 = request n + reply" gave 84 ok / 0 breaks, with 12 expected changes after cut replies. Uncached prefill already exceeded generated tokens (0.97M vs 0.85M xhigh). In E14-A, log the cache hit rate per request and the count of prefix breaks during the 25-min Duck run, to prove the blank-line normalisation re-prefill is gone. Any abort must wait for the server stream to end (0 % vs 52-99.7 % cache).
10. **Regex and parsing safety.** The rev 6 `_GRID_ROW_RE` backtracking hung for more than 11 min. Any new regex in E14-A (the watchdog log-growth parser, the decode-log parser `_E11_KV_RE = r"([A-Za-z#][A-Za-z#_ ()/\-]*?):\s*(-?\d+(?:\.\d+)?)"`, profiler output parsing) should get a linear-time test on long lines (≥ 10k characters), or a length guard (rev 6 skips lines longer than 600 characters).
11. **p90 / percentile formula.** If E14-A reports p90 of phase or request durations, use nearest rank `ceil(0.9n) − 1`, not `int(0.9 × (n − 1))` (A3-F3: [100, 200, 300, 400, 800] gave 400).
12. **Build hygiene before any push:**
    - Python string-replace patches failed on CRLF and escapes. Check LF after edits.
    - Check embedded package == source byte-for-byte (23 files) after every build.
    - Record the new sha. Never overwrite a Kaggle-run notebook (the `_REBUILD` redirect).
    - Make the test bundle self-contained if it goes to GPT (missing `smoke_adapter.py` and fixtures was an audit finding).
    - Save the replay outputs. The `replay_rev6d_*.jsonl` write failed silently in this part.
13. **Pre-run capacity statement.** Quota was ≈31 of 60 h at that time (stale now). The E14-A written estimate must list each phase's worst case (each server relaunch ≈10-13 min; a parity relaunch is a second full model load) against the 200-min budget.
