# Insights from part_02 (2026-10-21 05:44 to 08:10)

Source: `convo/part_02.txt`, 4,410 lines, read in full.

## How this part is laid out

- Lines 2-883, 1079-1998 and 2321-3310 are three IDE selections of the same document: `E12 next fix list (running, 2026-10-20).md` (lines 1-878, 1-916 and 1-986).
  - I checked by diff that they are the same text. Selection 2 adds the section "GPT review of the consolidated rev 6 plan, 1 of 2".
  - Selection 3 adds the corrected T8 row and the section "plan review 2 of 2 + T8 root cause".
  - Every new line was read.
- Two GPT plan reviews were pasted by the user: review 1 at 05:44 and review 2 at 05:50, both in Bengali.
- At 08:00 there is a context-compaction summary (lines 3818-4116).
- The part ends mid-investigation at 08:10.

---

## 1. User instructions, rules, preferences, decisions

| Date/time | Verbatim (Bengali) | English meaning / note |
|---|---|---|
| Rule (2026-10-20), quoted in the fix list header | (doc text) "Read every GPT report in full and validate every claim. Add the validated items here. Change no code until the user says so." | A standing rule: **no code changes until the user says so**. It was lifted at 06:58 (see below). |
| 2026-10-21 05:44 | "এটা একটা জিপিটি বলেছে,আর একটা পরে দিচ্ছি, ভ্যালিডেট korte পারো যা new idea" | "One GPT said this, I'll send another later; you can validate whatever is a new idea." Validate only the **new** ideas in GPT plan review 1. |
| 05:50 | (pasted GPT plan review 2, no extra text) | Implied: validate it the same way. |
| 05:55 | "Your response above was cut off mid-stream. Resume directly from where it stops — no apology, no recap." | A resume instruction after a cut. At 05:58 the assistant reply was "API Error: Can't reach the API server … (ENOTFOUND)". |
| 06:36 | "আপডেট দিও" | "Give me an update." The user waited 38 min after the API error. |
| 06:52 | "আমাদের ফিক্সগুলো কোন সমস্যার কোনটা সমাধান করবে বলিও" | "Tell me which of our fixes solves which problem." The answer was a problem → fix map at 06:53. |
| 06:58 | "ওকে আমি গ্রহণ করলাম।এবার নোটবুক আপডেট করো" | "OK, I accept. Now update the notebook." This starts the rev 6 build. **The assistant read it as approval of the whole recommended configuration** (see §6). |
| Earlier, quoted in the 08:00 summary | "আগে আলাদা এক্সপেরিমেন্ট ছিলো।সেখানে কাজ করেনি bole ekhane kaj korbe na emon na..." | "Those were separate experiments; that it didn't work there doesn't mean it won't work here." Standing rule: **past negatives from other setups are context, not verdicts**. It was saved to memory as past-negatives-not-verdicts.md. |
| Earlier, quoted in the summary | "আমাদের সমস্ত কাজ , পিডিএফ সব কিছু পড়াও,প্রয়োজনে আরো ৫ টা এজেন্ট নিও" | "Have all our work, the PDFs, everything read; take 5 more agents if needed." |
| Earlier, quoted in the summary | "এটার আগে তুমি জিপিটি গুলোর পরামর্শ আর তোমার এজেন্ট গুলোর পরামর্শ আবার দেখে আসো,হয়তো গুরুত্বপূর্ণ কিছু বাদ পড়তে পারে।আর ফিক্স লিস্ট আপডেট করো" | "Go back over the GPTs' advice and your agents' advice; something important may have been missed. Update the fix list." This produced the "Rev 6 gap check" (T1-T8). |
| Earlier, quoted in the summary | "প্রম্পট দিও … একদম শুরু থেকেই থিংকিং এর সকল ফল মডেল নিজের মধ্যে ক্যালকুলেশন না করে কোড চালাবে Sandbox এর মত কিছুতে।আর যত উপায়তে করা যায়…" | Code-first intent: from the start, the model should run code in a sandbox rather than compute in its head, "in as many ways as possible". |
| Standing constraints listed in the summary (08:00) | "অনুমতি ছাড়া কমিট করবি না" | "Don't commit without permission." The other constraints listed: never push/run/submit without an explicit go; never read or echo credentials; validate first; bundle evidence-based changes; make an audit zip before a push; scratch on D:; chat in Bengali; the 60k floor; a pre-run capacity check; read everything fully; trust user observations and never argue; **never propose running Duck on fewer than all tasks**; past negatives are not verdicts; never rebuild Kaggle-run notebooks. |
| Fix list F7 (superseded 2026-10-21) | (doc) | **The user's rule: Duck runs on every task (full coverage).** "Poetiq complement" is a separate, later question. |
| Fix list B5 | (doc) | "The user approved evidence-based cuts (2026-10-21)": the output cap 60,000 → 49,152. |
| Fix list "Low-effort variant" | (doc) "the same notebook, change only the xhigh agent" | The user chose low for EVAL12R5_LOW. |
| Fix list R6-Q1 approval history | (doc) "12,288 = approved exception to the 60k floor, user 2026-10-19" | That approval covered only the rev 1 / EVAL30 port, so rev 6 needed a new decision. The decision was then taken implicitly at 06:58. |

**GPT reviewer positions (user-supplied, not user rules, but they shaped the plan):**
- Review 1 (05:44): "GO for Rev6 development … design approval, executable notebook approval নয়" (design approval, not notebook approval).
- Review 2 (05:50): "GO মানে এখন REV6 implement করার অনুমোদন; GPU run-এর জন্য GO নয়।" ("GO means approval to implement REV6, not a GO for a GPU run.") GPT will audit the source diff, safety regressions, the scheduler sim and the source before a GPU GO/NO-GO.

---

## 2. Measured facts and results

### Runs and scores (from the fix-list header)

- **EVAL6R5 xhigh** (sha f4fed1ac…): 4/6 tasks, 5/8 inputs.
- **EVAL12R5_LOW** (sha 26fff41e…): 7/12 tasks, 10/19 inputs. Submitted as ref **57456562**, LB pending in this part.
- EVAL30R5 (415a68f2…): not run.
- Earlier rev 4 data:
  - EVAL6 (3/6, 4/8 recorded).
  - EVAL30 (4 tasks, 7/48).
  - The real-sandbox replay with F1+F5 gives EVAL6 5/6, 7/8.
- Identical code gave 2, 2, 3, 1 of 22 on EVAL runs, so EVAL6/12 accuracy is a sanity check only.
- Quota: 30.79 of 60 h used (check-only on hosen42), 31.29 / 60 while EVAL6R5 ran.
- Oracle = final: xhigh 5/8, low 10/19, so selection was not the lever.

### Token and behaviour metrics (rev 5 GPU runs)

- First `check()` median: **55.1k (xhigh)**, **53.7k (low)**.
- Never checked: 1/12 (xhigh), 3/24 (low).
- Pre-check tokens generated after the first "use code" phrase: 273k of 742k (37 %) xhigh; 279k of 1,394k (20 %) low.
- Self-detected misreads: 12/12 xhigh branches, 23/24 low (35/36 total).
- Rev 5 xhigh first reply: p50 33.6k tokens before any tool call.
- Requests: xhigh 127 requests / 117 python calls; low 449 / 437.
  - xhigh: the 28 requests ≥ 10k tokens carry 80 % of tokens.
  - low: 45 of 449 carry 58 %.
- Median completion per request: 1,115 (xhigh) / 1,090 (low).
- Median prompt: 51.9k / 41.6k.
- **Cached fraction 85.4 % / 91.0 %**; uncached about 888k / 1.577M.
- Uncached prefill already exceeds generated tokens: 0.97M vs 0.85M (xhigh).
- Late corrections:
  - 135a2760_b1: 71,306 → 97,972.
  - 898e7135_b0: 56,868 → 73,523.
  - 898e7135_b1: 77,859 → 79,662.
- Selective branch-1 replay:
  - xhigh 5/8 at 498,253 vs 846,080 tokens (−41.1 %).
  - low 9/19 at 1,094,372 vs 10/19 at 1,743,889 (−37.2 %); 135a2760 is lost.
- Low effort: 18.7 requests/branch vs 10.6 (xhigh); 72.7k vs 70.5k tokens/branch.
- First correct candidate positions:
  - xhigh: 32, 35, 54, 58, 62k.
  - low: 14-98k.
- Rev 4 EVAL6 output caps:
  - Tool-call replies: p50 2.3k, p90 31.6k, p95 35.6k, max 58.2k.
  - Replies > 40,960: 3; replies > 49,152: 1.
  - No solved-branch reply exceeded 44,779.
- Rev 4 drops: 11 of 84 later requests (426,781 tokens) not carried. 5 of them were avoidable system-only resets (about 279k tokens) caused by the chars/token estimate (1.40-1.72 on grid prompts). The real reset sizes were 65,203-69,461 tokens, but Duck estimated them at 77,269-111,375.
- Repeat (loop) share: EVAL6 29.4 % of characters, EVAL30 5.2 %. Duck's degeneracy threshold is 0.01 zlib, while our blocks score 0.037-0.19, so 0 detections.
- **EVAL30 request timeouts: 40** in 21 of 31 branches (28 full 900 s + 12 deadline-bounded; about 31,000 stream-seconds). The assistant had earlier written "4".
- Verification cost: 0.75-2.71 s each, ≤ 3 per branch (Windows, 16 verifications).
- Raw grid text per task: median 3,729.5 characters (range 1,914-8,036). Only 10 of 87 grids are ≤ 64 cells; 0 of 12 tasks meet report C's ≤ 8×8 / ≤ 256-cell rule.

### Throughput and engine facts

- Effective throughput today: **244-265 tok/s**. 11.3 h × 3,600 × 255 / 240 ≈ **43k tokens/task** under full coverage. At 350 tok/s that becomes 59.3k, at 416 it becomes 70.5k.
- Decode at 40-60k context is about **25 % slower (323 vs 429 tok/s)**.
- DFlash acceptance falls from **4.43 to 3.77** as the context grows.
- EVAL30 re-prefilled 8.4M tokens.
- 6 → 12 streams: +17 % in one measurement, +30-60 % in the short sweep.
- Cold prefill about **1,400-1,650 tok/s** (E11), i.e. about 30-35 s per 50k resume.
- **KV pool W1 = 881k (881,336) tokens.** 0.9 × 881k / 10 streams = 79.3k per stream. The rev 6 Duck window = min(131,072, 0.9 × kv / streams) = **79,320** at 10 streams.
- Context arithmetic: 131,072 − 49,152 − 512 = 81,408 input budget; with a 12,288 cap it is 118,272.
- Server context 155,648 vs Duck window 131,072, about 24k of margin.
- EVAL30 decode log, line 13947, at 232 min: branches done 7/60, `num_running_reqs 10`, `num_queue_reqs 1`, **`gen_throughput 380.2`**.
- Prefix-cache invariant evidence:
  - Stable order raised the cache from 11-18 % to 87 % (E11-06).
  - An abort without waiting for the stream to end gave 0 %, vs 52-99.7 % with waiting.
  - A tool-free request shares 15 % of the prefix vs 99 %.
  - A template conditional dropped it from 75 to 45 %.
  - `hash()` order changed an LB score (hosen42).
- Spec vs plain decoding shared only 152-166 of 512 tokens under batching (E11-03). An unchanged repeat scored 0.1875 against a 0.50 parity threshold (Deep Audit).

### T8 deadline-overrun facts (EVAL30, from logs and code, 06:36)

- `deadline_in_sec` = 13,367 is relative to the solve start; `started_sec` = 733. So the absolute solve deadline was about **14,100 s** (= HARD_STOP − 180). `finished_sec` = 14,779.
- Overrun = **679 s** (379 s past the 4 h TOTAL_BUDGET of 14,400).
- At 14,113 s about 16 requests timed out together; they were clamped correctly.
- At **14,436 s** a request timed out with "read timeout=419.6", so it started at about 14,017 s and ended 336 s past the deadline. Likewise 469.7 s at 14,492.
- At **14,247 s**: `num_running_reqs 11`, `num_queue_reqs 8`, **`gen_throughput 0.0`**.
- The report was written at 14,792.
- EVAL30 config:
  - THINK_BUDGET 60000, VERIFY_TIMEOUT_SEC 20.0, TOTAL_BUDGET_SEC 14400, MIN_THINK_BUDGET 4096.
  - hard_margin 120, verify_margin 60, drain_margin 480, admit_margin 120.
  - **SGL_LAUNCH_TIMEOUT_SEC 1800** (this result line is cut after that; the rest is not visible).

### Rev 6 build and test results (07:04-08:10)

- `arc2_duck.py` is now about 2,150 lines (backup `.pre_rev6_backup`, original sha 172371f1…); pyflakes clean.
- `smoke_rev6.py`: **66/66** after fixes. The first run had a FAIL: "V4: one long nested-list line is cut". It was fixed by changing the threshold from 0.35 to 0.25 × len, plus two test fixes.
- smoke_rev5 and the flag-off smoke_adapter: ALL OK.
- `replay_rev6`:
  - xhigh 5/8 (matches rev 5); kept_r6 ≥ kept_r5; 4c3d4a41_b1 auto-check added `transform_alt` (still wrong).
  - low: 10 of 12 tasks done at the last view; cbebaa4b and f931b4a8 pending.
  - A sample line: 135a2760 b0 had `grid_rows_suppressed` 92.
- Notebook builds:
  - EVAL12R6 first built as **e7b5617b**. After the final adapter/cap edits it was rebuilt as **a11e6e4c** (08:04-08:07).
  - Run-notebook shas checked as unchanged: 26fff41e LOW (after restore), 216c59ed S12, b65fb9dc EVAL30, 415a68f2 EVAL30R5, 53a66856 EVAL6, f4fed1ac EVAL6R5, cce9cb8f EVAL6R5_LOW, 0fea4d0c SMOKE1 (per the summary).
- Harness on a11e6e4c:
  - **r6 37/37 ok in 58 s; launchfail 6/6 in 19 s.**
  - rerun first showed a Traceback / `IndexError: list index out of range` (h9_EVAL12R6_rerun.txt L82-86; the cause is in a cut result). After a harness edit for the rerun scenario (edit text cut), **rerun 21/21**.
- The rerun metrics:
  - branches 10, tasks_exposed 5, tasks_meaningful 5;
  - first_tool_tokens_median 400, first_check median 600;
  - **timeouts_clamped 71**;
  - **resumes {"continue": 6, "first": 15}**; caps {"12288": 10, "24576": 70};
  - verify_sec_p50 3.34.
- Harness config seen in the rerun output:
  - E12_STREAMS 10, E12_LIVE_BRANCHES 10, E12_DUCK_MAX_OUTPUT 12288;
  - **E12_DUCK_REQUEST_TIMEOUT 1800**; E12_MAX_GEN_PER_BRANCH 100000; E12_STOP_BEFORE_HARD_SEC 180;
  - **"give_up_after_sec": 4500** (75 min);
  - E12_MAMBA_TRACK_64 false.
- The harness only runs in `_scratch/e13_venv` (Py 3.12.10, transformers 4.57.6, tokenizers 0.22.2). System Pythons 3.14 and 3.13 lack transformers.
- On Windows: "E12 WARNING: sandbox memory limit NOT verified … No module named 'resource'". The tokenizer loaded: probe 2,400 chars = 441 tokens.

---

## 3. Failures, mistakes and their root causes

1. **"1,412 s overrun" was wrong (the assistant's error, found 06:36).** The assistant subtracted the relative `deadline_in_sec` (13,367) from the absolute `finished_sec` (14,779). The true overrun is 679 s.
2. **T8 root cause (the real bug).**
   - `run_branch` computes `request_timeout_seconds = min(analyzer_timeout, remaining)` once per turn (arc2_duck.py:835-838).
   - Duck reuses that value as the HTTP read timeout for every request in the turn (tool_agent.py:6763-6764 → 5275).
   - `should_stop()` blocks new requests after the deadline, but a request started just before it carries the turn-start timeout.
   - Worst case in rev 5 is one analyzer timeout = **1,800 s ≈ 30 min** past the deadline.
   - Fix (R6-S6-a): clamp per request to deadline − now (min 0.5 s). `_verify_submission` gets budget = min(2 × python timeout, deadline − now − margin) and is skipped when under 1 s.
3. **`_verify_submission` not bounded by the global deadline.** `budget_end = now + 2 × python_timeout` (60 s) (arc2_duck.py:761). In `solve_all`, `for f in futures: f.result()` inside the `ThreadPoolExecutor` context waits for every worker (:1020-1032).
4. **T5 error (the assistant's, corrected by GPT review 1).** The claim "a duplicate output signature costs no verify slot" is impossible: a signature needs execution first. Review 2 added a trust boundary: an output signature from the untrusted sandbox must never skip the host verify (the serializer-spoof class). The order is source + dependency key → cached *trusted* verdict → clean host verify → output-signature dedupe only after acceptance.
5. **KV drain contradiction (the assistant's, corrected by review 2).** "Drain near 70-80k" conflicts with 79.3k per stream minus a 24.6k reserve = about 54.7k. Corrected to about 55k at a 24,576 cap and about 67k at 12,288 (Franzen drains at 58k).
6. **Explicit `check()` disabled by cap exhaustion (partly true).** Explicit check harvest shares the cap of 24 (`_arc2_harvest_checks`, arc2_duck.py:724-729); `submit()` is uncapped. Fix: separate auto/explicit caps, and a neutral note instead of "submit your best function", which ends the branch.
7. **Accidental overwrite of a Kaggle-run notebook (07:04-07:58).**
   - The builder rebuilt `Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL12R5_LOW.ipynb`, and its sha changed 26fff41e → bc1e1cd9.
   - It was restored from `_scratch/e12_rev5_results_zip/notebooks/`.
   - A builder rename guard was added: EVAL12R5_LOW / EVAL6R5 → `*_REBUILD.ipynb`.
   - A memory file `never-rebuild-run-notebooks.md` was written.
   - There was also a botched `sha256sum` over an unquoted path with spaces.
8. **Build tooling errors.**
   - A Bash heredoc with nested quotes failed: "unexpected EOF".
   - A docstring inside the builder's triple-quoted cell string caused a SyntaxError at build_e12_duck.py:781 (fixed by using comments).
   - The pyflakes unused `status` variable was removed.
   - A `sleep` chain was blocked by the tool.
   - The first `smoke_rev6` run hit the 600 s timeout with no output (buffering) and was rerun unbuffered.
9. **The harness could not run under the default python.** No `transformers`; it needs `_scratch/e13_venv`.
10. **The rerun-scenario IndexError** (h9 L82-86). The fix is in a cut result; its content is not visible.
11. **Earlier assistant errors recorded in the fix list.**
    - "No failed branch got even one train pair exact": wrong.
    - "The newest block always stays": wrong for tool-less replies.
    - "Temperature 0.6 is Qwen-like": wrong; the official thinking setting is 1.0.
    - "EVAL30 request timeouts: 4": wrong; there were 40.
    - "Must finish a full solve before a tool call": wrong; it is only a tendency (report C).
    - Labelling error: 62593bfd_b0 xhigh did check (2/2 at 58k); the never-solved branch was 135a2760_b1.
12. **The 05:58 API error (ENOTFOUND)** lost the response. The user had to ask for an update 38 min later.
13. **Engine stall left unexplained.** `gen_throughput 0.0` with 11 running and 8 queued at 14,247 s (EVAL30). It is "likely KV thrash (EVAL30 had all 60 branches admitted and re-prefilled 8.4M)" but not proven. Marked "not needed for the fix".

---

## 4. Decisions made, their reasons, rejected options

**Accepted at 06:58 (the assistant's reading of "ওকে আমি গ্রহণ করলাম"):**
- R6-Q1 request caps: 12,288 → 24,576 after the first error-free Python call → 32,768 after two consecutive no-call length cuts. Both GPT plan reviews preferred this.
- R6-Q2: a 25-30k soft first exposure (built as `first_exposure_tokens=28000`), resumed on the same agent object (`BranchSession`).
- Effort xhigh.
- Branch 1 uses the grids view (`B1_VIEW grids`, the representation hedge, exactly one deterministic alternative).
- Facts block 600-800 tokens first (built as `FACTS_MAX_CHARS 3600`, with an explicit "lists dropped" notice).

**Build values:**
- LIVE/workers 10; `MIN_WORKERS` 6; `MIN_ADMIT_SEC` 240.
- Check/auto-check caps 24 each; per-call auto-check 3.
- `DUCK_DRAIN_TOKENS` 16384; `SHOW_MAX_CELLS` 150; `KV_FRACTION` 0.9.
- Window = min(131072, 0.9 × kv / streams).
- Workers drop to 6 if isolation_ok is False.
- Capacity probe: every 60 s, 3 consecutive queued readings → allowed −2, floor 6.
- Join timeout = deadline + 120.

**Gate classes adopted (review 1 P8):**
- Mandatory blockers: helper retention + clean verify; auto-check dedupe / cap exhaustion / late correct candidate; facts exactness, no truth leak, multiple test inputs; rev 5 security; dynamic reserve/context consistency; 240-task admission, pause/resume, deadline-safe submission.
- Performance targets (not blockers): first python ≤ 12k, first check ≤ 25k, 10-stream gain (fallback allowed).

**Other design decisions:**
- **Auto-check scope narrowed** (review 1 P3, review 2 Q-3): names `transform*|solve*|solution*|predict*|*_alt*|*_v<n>`, new or changed, single-grid argument. It never auto-submits or auto-stops; `_verify_submission` stays the authority.
- **Trim at block level** (P5). A tool call and its result are kept or dropped together, never message-level deletion.
- **10 streams = target, not a guarantee** (P6, review 2 Q-1).
  - Admit by projected KV: Σ(resident + reserved generation) + new prompt + new reserve ≤ 0.9 × usable KV.
  - Fallback 10 → 8 → 6 is a controlled capacity step, not a failure.
  - **Host state kept ≠ GPU KV resident**, so a resume pays a re-prefill.
- **Facts in the system prompt** (T2), because a trim pops the opener.
- **Trusted helpers as retained sources** (T1), injected into the verify and test-prediction calls.
- **ARC-2 grid print guard** (T4): collapse ≥ 6 grid-like rows; `show()` limited to 150 cells.
- **Duck nudge "You have not acted yet. Investigate first." replaced** (T6).
- **Stage-2 classes** (T7): progress-no-candidate, nothing, one verified, ≥ 2 verified.

**Rejected / not in rev 6, with reasons:**
- Auto-stop at the first train-exact candidate: the late corrections in 135a2760 / 898e7135 argue against it.
- A 30k hard cap on all branches: xhigh would fall from 5/8 to 1/8.
- Forced or required `tool_choice`: HTTP 500 + 59 empty submits in Exp 4; SGLang #43514 (it poisons DFlash batches).
- Rolling summaries or stripping reasoning from history: Franzen reports a large regression.
- REFRAIN / text early exits: not validated on Qwen3.8 / ARC.
- Presence/repetition penalties under DFlash2: inactive in 0.5.21 (#28180). The DFlash path never calls `cumulate_penalty_output_tokens`; it would need a 2-line patch.
- Several generated views for branch 1: they would destabilize the prefix.
- A fixed 1,500-token facts block.
- An assert on the tokenizer fallback (record instead: an assert would kill the solver).
- `ARC2_CONTEXT_DRAIN_FLOOR` trims every turn: rejected in the rev 4 audit (they kill the prefix cache).
- A full judge, LoRA/RL, a large DSL, vision, a streaming rewrite, untested kernel patches, and an upstream kernel cherry-pick before reproducing the failure.
- NVARC as the attempt_2 filler: about 9.5 h per 240 tasks.
- **JIT cache dataset (−4 to −5 min): "outside the agent bundle"** at this time.
- Franzen patch #4 + `--watchdog-timeout 1800`: only if the prefix hit rate at 10 streams falls below rev 5.

---

## 5. Open items

1. **The "first" opener was sent 15 times across 10 branches** in the rerun harness (resumes {"continue": 6, "first": 15}). The assistant's last line (08:10): "কেন, সেটা কোডে দেখছি" ("checking in the code why"). **Unresolved at the end of the part.**
2. replay_rev6_low: cbebaa4b and f931b4a8 not finished (target 10/19).
3. Pending tasks still to do:
   - a "rev 6 implementation" section in the fix list;
   - the rev 6 audit zip + README for GPT (source diff, regressions, scheduler sim, GPU readiness);
   - re-running smoke_rev6 / smoke_rev5 / smoke_adapter after the final edits;
   - a Bengali report to the user.
4. Unexplained **`gen_throughput 0.0` with 11 running / 8 queued** (EVAL30, 14,247 s). Only hypothesized as KV thrash.
5. SGLang issues **not verified**:
   - #38009 (DFlash2 vs target-only diverge in thinking mode);
   - #36548 (concurrent state contamination);
   - #37475 (draft `target_layer_ids` fallback);
   - #31954 (LPM starves cold requests);
   - #39087 (quantized drafter acceptance collapse; ours is BF16);
   - z-lab/dflash #166 (multi-turn prefix-cache regression).
6. The R6-D1 option of a ~150k Duck window: "the rev 4 window change was reverted, so record why first". The reason is not recorded.
7. Gate 4b (a hidden-equivalent compute share, about 43k tokens/task) is proposed; it needs approval if it goes below 60k.
8. Still pending from before:
   - LB of 57456562;
   - E13 awaiting audit;
   - the S12 notebook (216c59ed) untested;
   - the 2-exact + 1-timeout rejection test (pending since rev 2/3);
   - `verify_sec` / verify-timeout metrics (pending since rev 2/3, now built in rev 6).
9. Whether the user really approved the R6-Q1 caps and the R6-Q2 pause (see §6.1).
10. The sandbox memory limit cannot be verified on Windows (no `resource` module). It must be verified in a Kaggle run.
11. The EVAL30 config line is cut after `SGL_LAUNCH_TIMEOUT_SEC = 1800 …[+841]`. The rest of the timing fields is not visible.

---

## 6. Insights a fresh reader might miss

1. **The 60k-floor approval was inferred, not stated.** The user said only "ওকে আমি গ্রহণ করলাম।এবার নোটবুক আপডেট করো" in reply to the problem → fix map. The assistant wrote "ধরে নিচ্ছি আপনি পুরো সুপারিশ করা configuration অনুমোদন করেছেন" ("I assume you approved the whole recommended configuration"). The summary records it as "the user approved the rev6 caps/first-exposure config by accepting". The 12,288 per-request cap and the 28k pause are justified as "not a branch ceiling", but the approval chain is thin.
2. **"Branch 1 = grids view" was a choice the assistant made.** GPT said rev 5 grids *or* CSV.
3. **The T8 fix generalizes:** any timeout computed once at the start of a multi-request unit (turn, phase) can overrun the deadline by a whole timeout. The rev 5 analyzer timeout of 1,800 s means up to 30 min.
4. **The 1,412 → 679 s correction came from mixing a relative and an absolute clock.** `deadline_in_sec` in the run report is relative to the solve start, while `finished_sec` / `started_sec` are relative to the notebook start. Any new report fields need one clock basis.
5. **KV per stream vs Duck window.** At 10 streams the window is 79,320, so with a 24,576 reserve history drains at about 55k. At 12 streams it would be 66.1k per stream and about 41.5k of history. More streams means less retained history per branch.
6. **Host state ≠ KV resident.** A paused branch's resume pays about 30-35 s of cold prefill per 50k. This works against high pause counts.
7. **Uncached prefill already exceeds generated tokens** (0.97M vs 0.85M xhigh). Prefill efficiency (prefix-cache invariants) is as large a lever as decode tok/s.
8. **Parity tests under batching are noisy by nature.** Spec vs plain shared only 152-166 of 512 tokens, and an unchanged repeat scored 0.1875 against a 0.50 threshold. Calibrate with an A/A control and judge on task-level outputs.
9. **Decode tok/s depends on context length** (323 vs 429 tok/s at 40-60k), and DFlash acceptance drops 4.43 → 3.77. Short-context benchmarks overstate production throughput.
10. **The scheduler trap S1:** `n_live` and `qpos` are both tied to `live_branches` (arc2_duck.py:976/984). With it unset, all 480 branches go live: the EVAL30 KV-thrash class.
11. **Duck's board guard `_board_like` never matches ARC-2 rows** (it needs ≥ 24 alnum characters with no spaces). Without R6-V4, hand-reading returns through tool output.
12. **The penalties-under-DFlash finding is static-code based** (0.5.21). Any arm relying on presence/repetition penalties is a no-op.
13. **The JIT cache, out of scope here ("outside the agent bundle"), became an E14 item later.** The −4 to −5 min estimate comes from this part.
14. **The rerun harness showed `timeouts_clamped` 71** over 80 requests in a short window. The per-request clamp fires on almost every request near the end, which is the expected behaviour.
15. **Low effort makes more requests** (18.7 vs 10.6 per branch) at about the same tokens, which matches the HF card's warning.

---

## 7. Relevance to the E14-A notebook (concrete checks / fixes)

| # | What to check or fix in E14-A | Why (evidence in this part) |
|---|---|---|
| 1 | **Every phase that issues multi-request work (25-min Duck run, token-level replay, greedy parity, latch test, baselines) must clamp each HTTP read timeout to the phase's absolute deadline per request.** A timeout computed once per turn or phase is not enough. Confirm that rev 6g's `_chat_completion` clamp (deadline − now, min 0.5) is active in E14-A's Duck run and that `_arc2_deadline` is set from the 25-min window, not the 200-min profile. | T8: a once-per-turn timeout ran requests 336 s past the deadline (read timeout 419.6 s at 14,436 s). With `E12_DUCK_REQUEST_TIMEOUT 1800`, one late request can add up to 30 min, which would blow the E14A 200-min budget. |
| 2 | **Use one clock basis for all phase-record / heartbeat timestamps** (seconds since kernel start), and label any relative field. | The 1,412 vs 679 s error came from mixing the relative `deadline_in_sec` with the absolute `finished_sec`. |
| 3 | **Runtime stall detector (not only at launch):** during measurement phases, flag `num_running_reqs > 0 && gen_throughput == 0` for N consecutive metric reads, and record it in the phase record. | EVAL30 at 14,247 s: 11 running, 8 queued, gen_throughput 0.0, unexplained. E14's watchdog covers launch only. |
| 4 | **Two launch timers must not conflict:** make sure the old `SGL_LAUNCH_TIMEOUT_SEC = 1800` (E11 engine cell) is raised or disabled where the new stall watchdog (75-min give-up) governs. The give-up should be stall-based, not a 30-min wall timer. | The EVAL30 config shows SGL_LAUNCH_TIMEOUT_SEC 1800; the harness config shows `give_up_after_sec 4500`. Memory: a GCSFUSE-slowed launch took 1,197 s, and a timeout = 0 LB. |
| 5 | **Guard `--watchdog-timeout 1800`** (SGLang flag in the E11 source): check it does not kill the server during long profiler, NCCL or latch phases where no request progresses. | The E11 flags include `"--watchdog-timeout", "1800"`. |
| 6 | **Greedy parity: run an A/A control first and set the threshold from it.** Judge on task-level outputs (tool call parsed, same check verdict) and batch-size-1 prefixes, not raw token equality at concurrency 12. **Target-only must be a separate bounded launch, never next to production.** | R6-X1: spec vs plain shared only 152-166 / 512 tokens under batching; an unchanged repeat scored 0.1875 vs a 0.50 threshold. |
| 7 | **Baseline windows "at 12 running" should be measured at a realistic context (40-60k resident), or report tok/s by context bucket.** Also record DFlash acceptance per window. | Decode 323 vs 429 tok/s at 40-60k; acceptance 4.43 → 3.77 over context. A short-context baseline overstates every arm equally and hides context-dependent arm effects. |
| 8 | **Assert real running == level** from the decode log or `/metrics` in every window (12 for baselines). For the `--min-free-slots-delay 1` arm, verify that 12-of-12 actually run. | The 11-of-12 bug (N4); "real running count from the decode log" (R6-S5); memory "verify real concurrency". |
| 9 | **The CUDA-graph batch list must cover every running size used, up to max-running + 1 (workaround)**: 1..13 for 12 streams. Read back `--cuda-graph-max-bs-decode` / the bs list from the server log. | R6-S5: "CUDA-graph batch list covers 1..10 including 9 and 10"; the E11 flags set `--cuda-graph-max-bs-decode = SGL_MAX_RUNNING`. |
| 10 | **Mamba/SSM state: keep fp16 SSM as the default. Treat the int8 mamba checkpoint arm as a measured arm with a parity check, never silently adopted.** Read back mamba slots ≥ the donor ratio (60 per 10 streams, so about 72 for 12). | R6-S5 "mamba slots ≥ donor ratio (60 per 10 streams), keep fp16 SSM"; memory fp32 → fp16 SSM (not bf16). |
| 11 | **Duck window derived from KV: compute `min(131072, 0.9 × kv_tokens / streams)` from the readback, and handle `kv_tokens = None`** (the parse fails) with a safe fallback plus a logged warning. At 12 streams this gives about 66k, i.e. about 41.5k history at a 24,576 cap. Decide whether the 25-min Duck run uses 10 or 12 streams and print the window. | Rev 6 build: window = min(131072, 0.9 × kv / streams) → `E12_DUCK_CONTEXT_WINDOW`; KV arithmetic Q-1 (55k / 67k drain). |
| 12 | **Capacity probe in the 25-min Duck run:** the rev 6 probe (60 s, 3 consecutive queued → −2, floor 6) can throttle the run and spoil the arm's tok/s measurement. Log every probe decision with the metric values. Memory says rev 6d probed 12 → 6 on a normal queue=1. | `solve_all_r6(... capacity_probe ..., min_workers=6)`. |
| 13 | **Prefix-cache invariant check for the raw-reasoning history fix:** log `prompt_tokens` and `cached_tokens` per request and assert that request n+1's prefix equals request n's prompt + reply (except after a trim). Abort paths must wait for the server stream to end. Never send a tool-free request. | R6-D2: abort without waiting gives 0 % vs 52-99.7 %; no tools gives 15 % vs 99 %; a template conditional gives 75 → 45 %. E14 removes Duck's blank-line normalization for the same reason. |
| 14 | **Count resume types in the Duck-run report; `resumes["first"]` should equal the number of branch starts.** | Rerun harness: "first" 15 in 10 branches, investigation open at the end of this part. An extra "first" opener after a pause/resume would re-prefill or change the prefix. |
| 15 | **No penalties and no named/required `tool_choice` in any E14 arm.** | #28180 (penalties inactive under DFlash2 in 0.5.21, static read); #43514 (required tool_choice poisons DFlash batches); Exp 4 HTTP 500. |
| 16 | **Isolation smoke (marker leak, #36548) at the concurrency actually used (12)** and a static check of the drafter `target_layer_ids` (#37475) from `SGL_DRAFT_PATH/config.json`. Rev 6 had `_e12_r6_preflight(n)`; make sure E14-A keeps it or does the equivalent. | R6-X1 / N6. |
| 17 | **Builder safety:** the E14-A builder must never overwrite a Kaggle-run notebook (rename guard → `_REBUILD`) and must print the shas of all run notebooks after every build. Quote paths with spaces in sha checks. | 26fff41e was overwritten (→ bc1e1cd9) in this part and restored; a `sha256sum` over unquoted paths failed. |
| 18 | **No docstrings inside generated cell strings** (use comments), and run a syntax check of every generated cell before the build is accepted. | SyntaxError at build_e12_duck.py:781 from a docstring inside the cell string. |
| 19 | **The CPU harness needs `_scratch/e13_venv`** (transformers 4.57.6 / tokenizers 0.22.2). Note that the sandbox memory limit is unverifiable on Windows, so E14-A should print the sandbox rlimit readback on Kaggle. | Harness failures at 08:02-08:04; "sandbox memory limit NOT verified … No module named 'resource'". |
| 20 | **Throughput bookkeeping for the decision:** report uncached prefill tokens next to generated tokens for each arm, because prefill exceeds generation (0.97M vs 0.85M). Use effective end-to-end tok/s (baseline 244-265) as the primary arm metric, not raw decode tok/s. | Rev 5 audit numbers; full-coverage arithmetic at 43k tokens/task. |
| 21 | **Phase budget arithmetic in the written estimate:** include a margin for one clamped request per phase, plus cold prefill of about 30-35 s per 50k context when a phase restarts with long histories. | Cold prefill 1,400-1,650 tok/s; T8. |

No E14-A notebook file was inspected (CPU-only reading task). The items above are checks implied by this part's evidence.
