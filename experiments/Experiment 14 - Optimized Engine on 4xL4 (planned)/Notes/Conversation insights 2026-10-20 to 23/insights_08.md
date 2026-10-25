# Insights from part_08 (2026-10-21 17:41 to 2026-10-22 11:34)

Source: `part_08.txt`, 5,216 lines, read in full in consecutive chunks.

**Scope.** This part covers four things:
- the E12 rev 6e self-audit and its GPT zip;
- three GPT audits of rev 6e (H6-1..H6-19);
- the rev 6f build;
- the start of a new self-audit, which the user asked for at 11:33.

**Structure.** The part opens with a very long IDE selection: the whole fix list `E12 next fix list (running, 2026-10-20).md`, as it stood at rev 6e. Many of the measured facts below come from that document. They are dated by the day they were recorded.

**Ordering oddity.** After the user message at 17:41 and the assistant message at 17:42 (Oct 21), the file shows assistant blocks stamped **13:38**. Then comes a context-compaction summary stamped "USER 17:44". The summary says that the messages after it are "kept verbatim" from before compaction. So some blocks are not in chronological order. The 13:38 blocks are the earlier G4B bundle work.

**Cut lines.** Many `[result]` lines are cut at 500 characters. Where the cut part matters, this is noted inline as **[cut]**.

---

## 1. User instructions, rules, preferences and decisions

| When | Verbatim (Bengali) | English meaning / note |
|---|---|---|
| 2026-10-20 (quoted in the fix list) | Rule: "Read every GPT report in full and validate every claim. Add the validated items here. Change no code until the user says so." | A standing rule for the fix list: validate, record, and change no code without a go. |
| 2026-10-21 (fix list §4) | "Those were different experiments. That something did not work there does not mean it will not work here." | Past negatives are context, not verdicts. They are used only to choose safeguards and metrics, "never as a reason not to try". |
| 2026-10-21 (fix list) | User approved evidence-based cuts (B5, 60,000 → 49,152). | An exception to the 60k floor rule, based on evidence. |
| 2026-10-19 (fix list approval history) | "12,288 = approved exception to the 60k floor, user 2026-10-19" | This approval covered rev 1 / EVAL30 only. Rev 6 needed a new decision, and got one: rev 6c. |
| 2026-10-21 (fix list, low variant) | "the same notebook, change only the xhigh agent"; the user chose low. | Only one variable changes: effort. |
| 2026-10-21 (Rev 6c) | "Gate 4b: permission granted. Take 12 streams. Permission for the rest too" | Approved: gate 4b (which cuts thinking below 60k as a diagnostic), 12 streams, and the remaining items. |
| 2026-10-21 (Rev 6d) | "OK, do 1 to 5" | Build only items 1–5 of the consolidated audit patch set. The PARITY fixes (6) and the self-contained bundle (7) were not taken. |
| 2026-10-21 (GPU log) | "run EVAL12R6G4B on the Foysal account" | GPU go for G4B rev 6d on foysalemonshanto. |
| 2026-10-21 (summary quotes) | "রান শেষ আউটপুট বিশ্লেষণ করো" | "The run is finished; analyze the output." |
| 2026-10-21 (summary quotes) | "এত এত রিভিউ এর পরও এত বড় বাগ আসলো কিভাবে?" | Complaint: "How did such a big bug get in after so many reviews?" |
| 2026-10-21 (summary quotes) | "এই probe stream 12 → 10 → 8 → 6 এটা এত জটিল করার কোনো মানেই ছিলো না।শুধু ৯৫% এর বেশি ক্যাশ থাকলে নতুন কোনো টাস্ক ডুকবে না,এটা সিম্পল একটা শর্ত দিলেই হতো।" | Correction and design decision: "There was no point making the 12→10→8→6 probe so complex. One simple condition would do: if the cache (KV) is above 95 %, no new task enters." This became G4-1. |
| 2026-10-21 (summary quotes) | "আমাদের কোড আর আউটপুট সহ রিডমি দিও ,আমি জিপিটি দিয়ে ভ্যালিডেট করাই … প্রতিটি ফিক্স এর কোনটা কাজ করেছে,কোনটা করেনি,আর কেন করে নি,কি করলে কাজ করবে … জিপ করো।প্রয়োজনীয় সব কিছু দিও জিপে" | Build a zip with a README, code and outputs for GPT to validate. GPT should say per fix what worked, what did not, why, and what would make it work. Include everything needed. |
| 2026-10-21 | "D drive use koro" | Use the D: drive. This is a standing rule; C: had 0 GB free at 13:38. |
| 2026-10-21 | "রিভিউ gulo abar dekhe nio....I think you missed something" | Correction: "Look at the reviews again; I think you missed something." The assistant had indeed missed the fix-by-fix verdicts and G4-14..G4-18. |
| 2026-10-21 | "কোন পরিবর্তন আমাদের কি ফিক্স করবে?" | "Which change will fix what for us?" |
| 2026-10-21 | "সবগুলোই nio...update" | Take all of G4-1..G4-18 and update (rev 6e). |
| 2026-10-21 17:41 | "তুমিই একবার ফিক্স লিস্টের সাথে তুলনা করে অডিট করিও,আর আমাদের আজকের পুরো কথোপকথন পড়িও,প্রয়োজনে আবার আপডেট করিও,শেষ জিপ দিও জিপিটির জন্য" | "Audit it yourself once against the fix list. Read today's whole conversation. Update again if needed. Finally, give the zip for GPT." |
| 2026-10-21 (summary, standing) | "অনুমতি ছাড়া কমিট করবি না" | "Don't commit without permission." This is a standing rule. |
| 2026-10-21 (summary, standing list) | — | Other standing rules: never push, run or submit on Kaggle without an explicit go; never read or echo tokens or credentials; validate every claim and read everything fully; scratch on D:\DevCache\Temp; chat in Bengali; 60k thinking floor (approved exceptions: rev 6 caps, gate 4b); a pre-run capacity check before any GPU run; never rebuild notebooks that ran on Kaggle; past negatives are not verdicts; trust user observations; bundle changes into one release. |
| 2026-10-22 04:42 | "ki holo?" | "What happened?" The session had stalled overnight and a tool call was interrupted. |
| 2026-10-22 07:17 / 09:06 / 09:08 | (pasted) three GPT audits of rev 6e | Audit 1 was pasted twice. Audit 3 was cut at Q6 ("উপরের finding ছাড়াও দুটি repo…") and the rest was never received. |
| 2026-10-22 09:22 | "জিপিটি গুলোর ফিক্স আরো একবার ডিপলি দেখে নাও,কিছু বাদ দিলে ফিক্স লিস্টে যুক্ত করে দিও।আর সত্যতা মাস্ট যাচাই করিও" | "Look at the GPT fixes deeply once more. Add anything missed to the fix list. You MUST verify truthfulness." Verification by reproduction is required, not code reading alone. |
| 2026-10-22 10:22 | "Ok\nupdate start" | The assistant took this as accepting all its recommendations: H6-1 at 24k through the tool result; H6-12 idle override kept and counted; H6-17 one retry on INCONCLUSIVE; must + should + telemetry in one release. **The user only said "Ok update start".** The mapping to the specific options was the assistant's assumption, stated at 10:23. |
| 2026-10-22 10:47–10:48 | "==", then "[Request interrupted by user]", then "কতদূর কাজ শেষ হলো?" | "How far has the work got?" The user wants progress updates. |
| 2026-10-22 11:33 | "তুমি নিজে থেকে একবার অডিট কর,সব ফিক্স কাজ করছে কি না,সব ফিক্স করা হয়েছে কি না,আগে করা কোনো ভুল আবার ফিরে এসেছে কি না।যত ধরনের টেস্ট করার করো" | "Audit it yourself: whether every fix works, whether every fix was made, whether any earlier mistake came back. Do every kind of test possible." The part ends while this audit is in progress. |

---

## 2. Measured facts and results

### Runs and scores (from the fix list and the chat)

**Rev 4 and rev 5 runs**
- **EVAL6 rev 4 xhigh:** 4/8 inputs, 1,217,904 tokens. 7 `length` replies at 60,000. EVAL30 had 40 request timeouts in 21 of 31 branches; the assistant had first written "4".
- **EVAL6R5 xhigh** (sha `f4fed1ac…`): 4/6 tasks, 5/8 inputs, 846,080 tokens (70.5k per branch).
- **EVAL12R5_LOW** (sha `26fff41e…`): 7/12 tasks, 10/19 inputs, 1,743,889 tokens (72.7k per branch). It ran 6,570 s at 6 streams, about **265 tok/s**. It was submitted as **ref 57456562 (LB pending)**.
- **Selection in the rev 5 runs** was lossless: oracle = final, 5/8 and 10/19.
- **Selective branch-1 replay:**
  - xhigh: 5/8 at 498,253 tokens vs 846,080 (−41.1 %).
  - low: 9/19 at 1,094,372 vs 1,743,889 (−37.2 %).
  - The task lost was 135a2760.
- **Rev 5 cache, aggregate:** xhigh 84.26 % (6,165,603 prompt / 5,195,178 cached); low 88.70 % (17,983,349 / 15,951,513). The request-level figures 85.4 / 91.0 had been used by mistake.

**G4B rev 6d run** (sha `1bdadd82…`, kernel `foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b` v1, 2026-10-21)
- GPU quota used before the run: 21.88 of 30 h.
- COMPLETE in **3,528 s**.
- Score: **2/19 inputs, task-norm 16.67 %** (bf45cf4b, b5ca7ac4). 12/12 tasks exposed, **47.7k tokens per task**, 572,035 tokens, 180 requests, **75.0 % cached** (2,678,857 / 2,008,877), 20/180 zero-cache requests.
- Load time: **1,197 s** (slow cold start).
- Report values: `started_sec` 1463, `deadline_in_sec` 2063, `finished_sec` 3517, `sec_per_task` 171.9, **`rev6_window` 72,228**.
- End-to-end throughput: **572,035 / 2,063 ≈ 277.3 tok/s** (GPT audit 2, confirmed).

**G4B bug and side measurements**
- The capacity probe stepped **12 → 6 on queue = 1 at KV 13–31 %**. At 12 in flight the steady state is 11 running + 1 queued.
- Throughput loss in the same 20–40k context bin: 370.9 → 321.4 tok/s = **13.3 %**. The assistant had first written 15–20 %.
- **Only 29 decode lines** show exactly 12 running, all between 11:59 and 12:12. They alternate between 42–250 tok/s and 382–497 tok/s, which fits prefill interfering with decode. The claim that 12 running is slower was withdrawn.
- First check: observed median 29,384.5 (6/16 branches checked); b0 median 26,228. The reported 33,690.5 was a censored proxy.
- Every b1 first request was cut at 12,288 before any call; the first call came at 12.4–18.0k. The 32,768 rescue cap was never used.
- b1 used 21.6 % of tokens (123,560) and produced 0 verified candidates.
- Fair share: every computed share (7,544 → 3,758) was below the 24,576 minimum chunk.
- Overshoot past the soft pause: p50 5.1k, max 15.8k.
- Deadline: 91 clamps, 0 stragglers, finished 9 s early.
- Facts block: 510–798 tokens. Mamba usage max 0.70–0.71 in all 7 logs.
- Print guard: 432 rows hidden; leaks were first reported as 173 and later corrected to 646 rows in 8 branches.

### Serving and metric facts

**SGLang 0.5.21 metrics**
- `sglang:token_usage` = max(full, swa, mamba). `sglang:full_token_usage` is the full-attention KV pool (`metrics_collector.py:88-92`).
- `num_retracted_requests_total` exists (`:474-476`).
- `prompt_tokens_details` is omitted when cached = 0 (`usage_processor.py:13-15`, with `--enable-cache-report`). So **null means 0 cached**.

**Decode-log line format** (an example of the format):
`[2026-10-21 11:58:22 TP0] Decode batch, #running-req: 11, #full token: 17257, full token usage: 0.02, mamba num: 35, mamba usage: 0.62, accept len: 4.61, accept rate: 0.52, cuda graph: True, gen throughput (token/s): 21.31, #queue-req: 1`

**KV and mamba**
- W1 KV assumed in the builder and harness: **881,336**. The issue #36935 ratio used **866,744 / 4,096 = 211.6** against a mamba cache of 56. The real G4B per-stream window was 72,228, i.e. about 866.7k / 12.
- **#36935** (open, seen on 0.5.17): branch reuse collapses when kv_pool / chunked_prefill > max_mamba_cache_size. Later downgraded, because mamba usage never reached full. The cause is still open.
- **#22935** (open): the mamba radix split gives a 0-hit.
- The engine needs 4 mamba slots per request: 12 × 4 = 48 ≤ 56.
- `SGL_MAX_RUNNING` 12, server max-running **13** (the "11-of-12 workaround").
- CUDA-graph list 1–8, 10, 12; batches of 9 and 11 pad up.
- FP8 KV warning "no scaling factors, defaulting to 1.0": 8 warnings per log, in every run, benign.

**Context and per-stream budgets**
- Duck window 131,072; server context 155,648.
- Per-stream window at KV fraction 1.0: 73,444 (12 streams). History budget: 48.4k at the 24,576 cap, 60.6k at 12,288.
- At KV fraction 0.9: 66.1k window, 41.0k history.
- At KV fraction 0.95: 69,772 window, 44,684 input.

**Penalties and parity**
- Presence, frequency and repetition penalties are **inactive with DFlash2 in 0.5.21**. The DFlash path never calls `cumulate_penalty_output_tokens` (= #28180).
- E11-03: spec vs plain shared only 152–166 of 512 tokens under batching. An unchanged repeat scored 0.1875 against a 0.50 threshold. So token equality under batching is not a usable parity criterion.

**Throughput and context**
- Decode at 40–60k context is about 25 % slower (323 vs 429 tok/s).
- DFlash acceptance falls from 4.43 to 3.77 over context.
- EVAL30 re-prefilled 8.4M tokens.
- Going from 6 to 12 streams gave +17 % in one measurement (+30–60 % in a short sweep).
- Cold prefill is about 1,400–1,650 tok/s, i.e. about 30–35 s per 50k resume.

**KV in historical logs**
- `kv_queue_scan`:
  - rev 4 EVAL6: max 6 running, KV p50/p90/max 0.24/0.39/0.46, mamba max 0.36;
  - rev 5 low: 0.28/0.35/0.47;
  - rev 5 xhigh: 0.24/0.36/0.43.
- `kv_gate_replay` over 7 logs: a hold only in **E12 EVAL30 rev 1** (219 samples, **18.2 min**, KV up to 100 %). No hold in G4B (max about 44 %), rev 4, rev 5 ×2, or E11-05/06 (65–73 %).

**Deadline (T8)**
- The EVAL30 overrun was **679 s**, not 1,412 s.
- Requests carry a turn-level timeout: one ran 336 s past the deadline (read timeout 419.6 s, started at about 14,017 s).
- At 14,247 s the log shows `num_running_reqs` 11, queue 8, `gen_throughput` 0.0. This is still unexplained.
- Maximum possible overrun = one analyzer timeout (1,800 s).

**Gate 4b and other settings**
- Gate-4b window = n × (42,720 − solve start) / 240. The hidden solve end is 43,020 − 120 − 180 = 42,720.
- `E12_STOP_BEFORE_HARD_SEC` 180; drain margin 480.
- Harness config (h15_r6) includes `SGL_LAUNCH_TIMEOUT_SEC` 1800, `SGL_RELAUNCH_TIMEOUT_SEC` 1200, `SGL_STREAM_IDLE_SEC` 1800, `SGL_MIN_RUNNING` 8, `SGL_MIN_KV` 600000, `SGL_KV_PER_REQUEST` 70000, `SGL_DRAIN_MARGIN_SEC` 480, `SGL_KV_GATE_MIN_ACTIVE` 10, **`SGL_KV_ADMIT_MAX` 0.85**, `give_up_after_sec` 4500. **[cut]**: the E12 line after it is cut.

**Tokenizer measurements**
- A 30×30 `show()` = 2,033 Qwen tokens (tool cap 3,072), about 2.26 tokens per cell.
- Facts block (rev 6d+, 120 eval tasks): p50 699 / p90 779 / max 798. Before the levels were added: p50 736 / p90 837 / max 1,396, with 18 tasks over 800.
- Indexed b1 view on 120 eval tasks: p50 5,782, p90 10,125, max 11,719. 0 tasks over 12,000; 5 tasks with a hidden grid.

### Rev 6e self-audit (2026-10-22, about 04:42–05:35)

**Guard leaks**
- Over all 149 transcripts: the rev 6e regex lets 1,564 grid rows reach the model (21 branches); the widened regex lets 438 through (9 branches).
- G4B: rev 6d regex 646 rows (8 branches); rev 6e regex 480 (5 branches); widened regex **1**.
- The bundled 64 transcripts: 865 → 210.

**240-task simulation** (`sim240.py`, scale 400, virtual window 41,257 s)
- At 371 tok/s with chunk 12,288: 240/240 exposed, 0 stage-1 starts left, 0 stragglers, about 15.0–15.3M tokens. Per task p10/p50/p90 49.6k / 62.8–63.2k / 72.1–72.5k; 85–90 tasks under 60k. Branch 1 started 176–181 times. Fair share p50 about 4.0k, below the minimum chunk in 200 of 200 cases. Simulated solves 30–35.
- At chunk 24,576: branch 1 started 103 times; 273–276 stage-2 runs; 36–37 simulated solves.
- At **277 tok/s**: about 47.6k per task; **169 tasks under 60k**; b0 resumed in stage 2 for 151/240 (vs 232/240 at 371), so 89 b0 never resume.
- **[cut]**: the full sim output lines are partly cut.

**Notebooks after the self-audit:** EVAL12R6 `c036215e…`, G4B_REBUILD `94a54754…`, PARITYR6 `bc9e2bc2…`. The adapter sha was `02ac03f2eeaa4b09…` (GPT audit 3).

**Tests:** smoke rev6 / rev5 / adapter 143 / 56 / 41. Harness g4b / r6 / rerun / launchfail / parity 53 / 51 / 22 / 6 / 19.

**Zip:** `E12_REV6E_IMPLEMENTATION_AUDIT_FOR_GPT_2026-10-22.zip`, 15.9 MB, 384 files.

### GPT audits of rev 6e (07:17) and the reproductions (09:23–09:27)

**Verdicts**
- Audit 1: Conditional NO-GO for G4B and EVAL12R6; GO for PARITY (diagnostic only).
- Audit 2: GO for G4B, GO for EVAL12R6 after G4B, NO-GO for PARITY; hidden run not qualified.
- Audit 3: cut off.

**Reproductions**
- **H6-1 (G4B transcripts):** with the nudge carried by an opener, **0 of 12** b0 had a new turn ≥ 27k inside stage 1. At 24k, an opener would still reach only 5/12.
- **H6-2:** reproduced in the real agent. `transform(a, b)` and `solve_row(r, i)` set `first_transform_tokens`.
- **H6-4:** a late write after scheduler return grew the file **399 → 544 bytes** (tokens 1,000 → 2,000). GPT's own reproduction: 395 → 543 bytes.
- **H6-7:** real Qwen tokenizer **21,963** (10 train + 3 test, 20×20) and 23,879 (12 + 1), against a 12,000 bound. GPT's fallback-estimator numbers (12,073 / 13,136) match exactly. The case occurs 0 times in the 120 eval tasks.
- **H6-6:** loss up to **2.7 %** (240 / 9,000 s).

### Rev 6f build (10:23–11:28)

**H6-1 effect on the G4B transcripts** (nudge in the tool result at 24k): 7 of 12 b0 would get it inside stage 1 (4 of them 3–10k tokens before the pause, 3 exactly at the pause). 3 had already written a transform. **2 never get it**: they were in one long tool-less think after 24k.

**Tests**
- smoke rev6 **162** / rev5 56 / adapter 41.
- Harness g4b **57/57**, r6 **55/55**, rerun 23, launchfail 6, parity 19.
- `window_check` 7/7. Regex ALL OK. KV replay 7/7. Facts size unchanged.
- Replays unchanged: rev 4 7/8, xhigh 5/8, low 10/19.
- Without `tokenizers`: smoke_rev6 exits 2 with "INCOMPLETE … 1 skipped"; smoke_rev5 exits 2 with 52 PASS + 4 SKIP.

**`verify_h6.py`:** H6-2, H6-4 and H6-7 reproduce on rev 6e and do **not** reproduce on rev 6f. After the fix, H6-7 gives 11,766 tokens with 11 grids hidden.

**Notebooks:** EVAL12R6 `344a516c…`, G4B_REBUILD `b98b020f…`, PARITYR6 `530bbc93…`. All 9 Kaggle-run notebooks are unchanged.

**Zip:** `E12_REV6F_IMPLEMENTATION_AUDIT_FOR_GPT_2026-10-22.zip`, 16.0 MB, 403 files. It includes `tests/requirements.txt` (tokenizers==0.23.2, transformers==5.20.2).

**Start of the self-audit at 11:34:** 126 IDs in the fix list; **44 never named in a test**: A3-F2 C1 C10 C2 C3 C4 C5 C6-1 C6-3 C6-6 C7 C8 C9 G4-13 G4-9 H6-12 H6-15 H6-17 N1 N2 N3 N4 N5 N6 N7 R5-6 R6-A R6-B R6-C R6-D R6-E R6-G R6-H R6-I R6-J R6-K3 R6-P1 R6-P3 R6-S4 SA-1 SA-2 T3 T4 T5.

---

## 3. Failures, mistakes and root causes

1. **Rev 6d capacity probe bug.** The assistant assumed a healthy server has queue = 0. Real steady state at 12 in flight is queue 1 (11 running + 1 queued, because max-running is 13). The probe dropped 12 → 6 at KV 13–31 % and cost about 13 % throughput. The fake test server always reported queue 0, so the tests passed. Fix: a KV-only gate (G4-1), and the fake server now reports queue 1.
   - The user's complaint: "How did such a big bug get in after so many reviews?" The memory lesson "replay controllers on REAL recorded metrics" comes from this.
2. **Guard-leak count understated.** The assistant reported 173 rows in 3 branches; the truth was 646 rows in 8 branches. Root cause: joined-digit rows such as `0 0000000000` and `print(r, ''.join(...))` were not caught (SA-2).
3. **SA-1.** The gate-4b window started before the cache diagnostic, which can take up to 240 of 2,063 s (about 12 %). The hidden run never runs the diagnostic, so the time was biased.
4. **H6-3.** Even after SA-1, the cache diagnostic (12 concurrent long churn prompts) ran **before** solving. It changes the radix, KV and mamba state that the hidden run never sees. Fix in rev 6f: run the diagnostic after solving.
5. **H6-4.** `publish(s.write())`: `write()` hits disk before `publish()` checks `closed`. A straggler could therefore rewrite `branch_result.json` after the scheduler returned. The existing C6-4 test checked only in-memory records.
6. **H6-5.** `smoke_rev5.py` and `smoke_adapter.py` printed "SOME FAILED" but exited 0. Earlier reports judged the "ALL OK" line, so no wrong result had been reported.
7. **H6-2.** The first-transform detection lacked the one-grid signature check that auto-check uses (`_r6_single_grid_arg`), so the nudge was suppressed by unusable functions.
8. **H6-7.** In `arc2_indexed_grids` the loop `if reasoning_tokens(text) <= limit or cells <= 400: break` exits early when all grids are small, so the 12k bound is not enforced.
9. **H6-8: a false reproduction (the assistant's error).** The assistant first marked it "Reproduced (1 of 6 at 4 workers)". The method was wrong: it used request-log order, but the first request is logged a few ms after the pop. Measured at the real pop moment (BranchSession creation under the scheduler lock), there were 0 events on both rev 6e and rev 6f. The fix was kept as a defensive change. Corrected at 11:16–11:28.
10. **H6-1.** The nudge was carried only by a new turn's opener. In G4B no b0 started a new turn after 27k within stage 1, so the nudge never arrived during stage 1.
11. **Earlier errors listed in the fix list:**
    - the EVAL30 overrun was 679 s, not 1,412 s (relative vs absolute time subtracted);
    - EVAL30 timeouts were 40, not "4";
    - "the newest block always stays" was wrong;
    - "temperature 0.6 Qwen-like" was wrong (official thinking setting: 1.0);
    - "no failed branch got a train pair exact" was wrong;
    - the 12-running slowdown claim was withdrawn;
    - README cache figures used request-level instead of aggregate values;
    - G4B was overclaimed as an "LB/eval predictor" (corrected to "a screening test");
    - "conditional branch 1" was really prioritised;
    - the Q1 audit note blamed the wrong snapshot filter (it was the vLLM one).
12. **Rev 6 / 6b / 6c defects (all fixed):**
    - a catastrophic regex backtracking hang: `_GRID_ROW_RE` had overlapping separators and froze for more than 11 minutes;
    - the wrong opener after a rejected submit;
    - the caps metric counted trims instead of requests;
    - two workers could drive the same agent (a data race: 24 concurrent entries in the negative control);
    - fair-share starvation: one task took 792k;
    - the isolation preflight failed open (`None` counted as pass);
    - the scheduler join had no shared bound (12 × 120 s = 24 min worst case);
    - `BranchSession.write` read `arc2_requests` directly;
    - a race on the first KV sample;
    - CRLF introduced by the assistant's edits.
13. **Environment failures.**
    - C: was full (0 GB free), which caused `cat: write error`.
    - Printing Bengali hit cp1252 `UnicodeEncodeError`; the fix is `PYTHONIOENCODING=utf-8`.
    - `transformers` was missing after the C: cleanup; 5.20.2 was reinstalled into Python 3.14 user-site.
    - Heredoc quoting failed ("unexpected EOF").
    - `build_e12_duck.py` had been saved as CRLF by an editor (1,353 CRLF), which broke a patch assert.
    - The overnight session ended mid-tool-call, with an unknown outcome. The assistant re-checked: the patch had not applied (`grep -c "G4-3+"` = 0).
    - Parallel replays hit a Windows temp-dir lock.
14. **Harness check bug in rev 6f.** The checks read the report's branch-row subset instead of the real `branch_result.json`, giving g4b 55/57 and r6 53/55. Fixed to read the files, then 57/57 and 55/55.
15. **H6-1 smoke test failed first.** "FAIL H6-1: a tool call at 36,864 (the request that crosses the 28k pause) carries the nudge into the history." A debug run showed `wants True`. The assistant then **edited the test setup**, not visibly the code, and got 162 ALL OK. **[cut]**: the replacement text of the test is cut, so it cannot be confirmed whether the expectation was kept as strong as before. Worth a check.
16. **GPT environments lacked `tokenizers` and `transformers`.** None of the three GPTs could reproduce the 143/143. Root cause: smoke_rev6 imports `tokenizers` at the top (H6-19). Fixed with SKIP and exit 2, plus `requirements.txt`.

---

## 4. Decisions made, and options rejected

**Taken**

| Decision | Reason / evidence |
|---|---|
| **G4-1: KV-only admission gate.** 12 workers fixed; no new task or branch while `sglang:full_token_usage` ≥ 0.95; poll every 5 s; fail-open if the metric is missing, stale (> 30 s) or invalid; one synchronous probe before the first admission | The user's rule. The replay of 7 logs holds only in the overloaded EVAL30 rev 1. Step-down on queue, retractions or KV 0.98 was removed. |
| **H6-12: keep the idle override** (admit one when running == 0 even at KV ≥ 95 %) and count `kv_idle_overrides` on the coverage line | It avoids a deadlock and almost never happens in real data. GPT: a documented exception to the literal rule. Assistant's recommendation; accepted via "Ok update start". |
| **H6-1: deliver the nudge in the next tool result** (`harness_note`) at **24,000** tokens instead of an opener at 27k | 0/12 delivered in stage 1 with the opener; 7/12 with the tool result. Cost: one solved G4B branch (b5ca7ac4_b0) would get the nudge about 2k tokens before it wrote its transform. |
| H6-17: isolation INCONCLUSIVE → one retry; FAIL → 6 workers | Avoids a transient drop from 12 to 6. |
| H6-3: cache diagnostic after solving; its time comes from the hard-stop margin; the solve deadline is never extended | The benchmark confound. |
| H6-6: in all commit profiles the solve window starts after preflight and diagnostic, bounded by HARD_STOP | Up to 2.7 % lost. |
| H6-5 / H6-19: nonzero exit on FAIL; SKIP + "INCOMPLETE" + exit 2 when a dependency is missing; `requirements.txt` | Test reliability. |
| G4-17: minimum stage-2 chunk 12,288; kept after both audits recommended it | The fair-share formula is inert, so the minimum chunk is the real allocation control. 24,576 looks better only under the sim's assumed solve model, which is not evidence. |
| KV fraction 1.0 (not 0.9 or 0.95) for 12 streams | 0.9 gives 41k of history, below the rev 5 median prompts (41.6–51.9k). D66 ran 12 slots with a 90 % KV peak and 0 preemptions. |
| p90 last-admission cutoff built but **off by default** | Every request and verify is already clamped to the deadline. A cutoff would idle 12 workers for about 15 min. |
| No engine change for mamba 56 / the CUDA-graph list (rev 6c) | 48 ≤ 56 slots; 9 and 11 pad up. |
| G4-9: no radix-cache, mamba-strategy or kernel change until G4-8 / G4-15 show the cause | Speculative. |
| No new allocation scheduler now (all three audits) | Decide after the G4B evidence. |
| H6-13 (arc_overview dedupe): no change | 0 occurrences in 120 tasks. |
| H6-15 (narrow labelled rows like `r2: 388883`): optional | A blanket 4–7 digit filter could hide legitimate counts. |
| PARITYR6 stays deferred and NO-GO | C6-1 (a 900 s urlopen with no deadline; 24 probes = up to 6 h), C6-3 (tool name not compared), Q2 (2,048-token probes rarely reach a tool call) are not fixed. |

**Rejected**

The fix-list "not in rev 6" lists:
- rolling summaries, stripping reasoning, `ARC3_PRUNE_*`;
- forced or required `tool_choice` (Exp 4 HTTP 500; #43514);
- REFRAIN and other early exits;
- penalties under DFlash;
- NVARC as the attempt_2 filler;
- images;
- the CPU rule library;
- a universal auto-stop at the first train-exact candidate (late corrections: 135a2760_b1 71,306 → 97,972; 898e7135_b0 56,868 → 73,523);
- a 30k hard cap;
- merging branch contexts;
- a judge ensemble;
- auto-check of every one-argument helper;
- an output signature from the untrusted sandbox skipping the host verify;
- fuzzy or AST source similarity for "same transform" (log source hashes instead).

---

## 5. Open items at the end of the part

1. **The self-audit requested at 11:33 is unfinished.** The part ends right after counting 44 fix-list IDs never named in any test. No result yet.
2. **No GPU run of rev 6f G4B (`b98b020f…`).** The assistant asked (11:28) whether to run it or send it to GPT first, and on which account. Not answered in this part. Per memory, later work moved on to rev 6g.
3. **Audit 3's tail was never received.** It is cut at Q6: "উপরের finding ছাড়াও দুটি repo…" ("apart from the findings above, two repo…"). Its last items are unknown.
4. **Request #63 "হামিদ ভাইয়ের আইডিতে কমিট" ("commit on Hamid bhai's ID") was never done.** The assistant flagged it at 05:35.
5. **Submission 57456562 (EVAL12R5_LOW): LB pending.**
6. **H6-8 has no real trigger.** It is kept as a defensive fix; GPT was asked for a concrete trigger.
7. **2 of 12 b0 branches still never get the nudge.** They stay in one long tool-less think after 24k.
8. **The cache regression is unexplained:** 75 % vs 84–89 % cached, with zero-hits doubled (11 % vs 5 %). Candidate causes: the mamba radix issue (#22935 / #36935), prefix change, or re-prefill after a pause. The G4-8 / G4-15 telemetry exists only with message-level prefix plus a payload hash (H6-10); token-level proof is not available.
9. **`gen_throughput` 0.0 with 11 running (EVAL30 at 14,247 s) is unexplained.** Likely KV thrash.
10. **Throughput risk.** The sim's 371 tok/s is optimistic. At 277 tok/s, 169 of 240 tasks get under 60k tokens and 89 b0 never resume. Full coverage within the 60k floor is not proven.
11. **Unverified SGLang issues:** #38009 (DFlash2 vs target-only divergence in thinking), #36548 (concurrent state contamination), #37475 (drafter `target_layer_ids` fallback), #31954 (LPM starves cold requests), #39087 (quantized drafter acceptance collapse), #43514 (required tool_choice). Target-only parity was never run.
12. **Hidden 240-task submission is not qualified.** All audits say so.
13. The isolation INCONCLUSIVE → 6 policy and the PARITY defects are unchanged unless the user asks.
14. EVAL30R6 has a builder variant but was never built.
15. The H6-1 smoke test edit (its content is cut) should be re-checked; see §3.15.

---

## 6. Insights a fresh reader might miss

- **Queue = 1 is normal at 12 in flight**, because the server runs max-running 13 as the "11-of-12" workaround. Any controller or test that treats queue > 0 as overload is wrong. The fake server was changed to report queue 1 always.
- **Null `prompt_tokens_details` means 0 cached**, not missing data. GPT audit 2 wrongly said 0-hits were "not confirmed".
- **Message-level prefix equality does not prove token-level prefix equality** (GPT, all audits). Template rendering, tool schema or flags can differ. This links directly to the later E14 finding: Duck's blank-line normalization changed the reply text, so the previous reply was re-prefilled.
- **The per-stream KV used in the harness (881,336) differs from the real G4B pool** (`rev6_window` 72,228 ≈ 866.7k / 12; the ratio calculation used 866,744). Read the real pool at runtime; do not hard-code it.
- **The fair-share logic is inert at both 12 and 240 tasks.** The minimum chunk decides everything; the scheduler is effectively "minimum-chunk round robin".
- **"Reproduced" can be a measurement artefact.** H6-8 was reproduced and then un-reproduced once measured at the right moment. Pair every reproduction with a negative control on the fixed code (done via `verify_h6.py` on rev 6e vs rev 6f).
- **Recorded PASS counts were not independently reproducible** by any of the three GPTs (missing dependencies). Hence `requirements.txt` and the explicit SKIP / INCOMPLETE exit code.
- **Turn-level vs request-level timeouts.** Duck reuses one timeout for every request in a turn, so a late request can run up to 1,800 s past the deadline. Every HTTP call needs a per-request clamp to an absolute deadline. This includes diagnostics: PARITY used `urlopen(timeout=900)` with no deadline.
- **Disk writes and in-memory freezes are different guarantees** (H6-4).
- **Contradiction between audits.** Audit 1 said PARITY GO (diagnostic) and G4B / EVAL12 conditional NO-GO. Audit 2 said G4B GO and PARITY NO-GO. The assistant built the union of the fixes.
- **Approval drift.** "Ok update start" was turned into four specific choices by the assistant. The user did not state them, but they were announced at 10:23.
- **"A1 can print a full grid on demand" contradicts every donor.** With grids hidden, hand-reading returns through tool output. The guard and `show()` paging were the answer. The leak rate was much higher than first reported.
- **Prefill interference.** Decode lines alternate between slow and fast right after admission prefill. A throughput window that includes those lines underestimates steady decode; one that excludes them overestimates end-to-end rate.
- **Load time 1,197 s** in G4B against a 1,800 s launch timeout (GCSFUSE slowdown period). This is the motivation for E14's `/dev/shm` copy and stall watchdog. The harness `give_up_after_sec` of 4500 s (75 min) matches E14's 75-min give-up.
- **Two KV thresholds exist side by side:** the E11 engine cell's `SGL_KV_ADMIT_MAX` 0.85 (with `SGL_KV_GATE_MIN_ACTIVE` 10) and the E12 rev 6e Duck gate 0.95. Which one governs Duck admission in the notebook was not discussed in this part.

---

## 7. Relevance to the current E14-A notebook (4×L4, W1 engine, Duck rev 6g, measurement phases + 25-min Duck run, 200-min budget)

1. **Clamp every probe and bench request to one absolute deadline.** This covers the latch test, token-level replay, greedy parity, baseline windows and the profiler.
   - Fix: timeout = min(given, deadline − now). Do not start a new probe with less than about 60 s left. Write partial results on exit.
   - Evidence: C6-1 (PARITY `urlopen(timeout=900)` × 24 probes = up to 6 h, which would eat the 200-min budget); T8 (a turn-level timeout let a request run 336 s past the deadline; max = one 1,800 s analyzer timeout). E14-A has many sequential phases, so one hung request can starve every later phase.
2. **Start the 25-min Duck run's window after the preceding phases end, capped by HARD_STOP.**
   - Fix: solve_start = now after the last measurement phase; _end = min(HARD_STOP − stop margin, solve_start + 25 min). Record both.
   - Evidence: H6-6 / SA-1 (the window was set before the diagnostic and silently shrank).
3. **Control the server state before the Duck run.** Arms #44401, min-free-slots, CUDA graphs, int8 mamba, DFlash width and NCCL all relaunch or reconfigure the server. Replay, parity and churn-like phases fill the radix and mamba cache.
   - Check: the Duck run on the adopted arm either starts on a fresh launch, or logs its initial gauges (`full_token_usage`, `mamba_usage`, cached fraction of the first requests).
   - Do **not** assume a cache-reset endpoint (for example `/flush_cache`) without first verifying it exists and works in 0.5.21. GPT: "Reset API যাচাই ছাড়া কোনো endpoint ধরে নেওয়া যাবে না" ("do not assume any endpoint without verifying the reset API").
   - Evidence: H6-3.
4. **Verify the real running count in every "12 running" baseline window from the decode log**, not from workers in flight.
   - Fix: parse `#running-req`, `#queue-req`, `full token usage`, `mamba usage`, `accept len`, `gen throughput` (format in §2). Assert that the window really sat at 12.
   - Flag decode lines right after a prefill batch separately (the `_r6e_concurrency` "after-prefill split"). Compare arms only within the same context bin.
   - Evidence: G4B had only 29 lines at exactly 12 running, all in the early phase. Values alternated 42–250 vs 382–497 tok/s. The 20 % vs 13.3 % error came from mixing context bins. The 11-of-12 problem appears as `#running-req: 11, #queue-req: 1`.
5. **The `--min-free-slots-delay 1` arm and the max-running 13 workaround interact.**
   - Check: with the 13-slot workaround and 12 workers, does the arm reach 12 running with queue 0? Record whether max-running is 12 or 13 in each arm.
   - The CUDA-graph 1..13 arm covers batch 13, which the old list (1–8, 10, 12) did not capture: 13 ran without a graph, and 9 and 11 padded up.
   - Evidence: the "server 13 (the 11-of-12 workaround)" fact, and the graph list.
6. **Use exact metric names.** Use `sglang:full_token_usage` for KV and `sglang:mamba_usage` for mamba, max over TP ranks. Never `sglang:token_usage` (= max of full / swa / mamba).
   - This applies to any watchdog or heartbeat field and any admission gate.
   - Evidence: G4-1 / G4B audit 1 #3; `metrics_collector.py:88-92`.
7. **Cache accounting rules.** Treat a missing `prompt_tokens_details` as 0 cached (`--enable-cache-report`). Report the aggregate Σcached / Σprompt, not the mean of per-request ratios.
   - Evidence: 84.3 / 88.7 % aggregate vs 85.4 / 91.0 % request-level; null = 0 per `usage_processor.py:13-15`.
8. **Measure the Duck raw-reasoning-history fix at token level, not message level.**
   - Fix: per request, log the payload SHA-256, the character or rendered-prompt common prefix vs the previous request, tools/template-change flags, cached tokens and idle seconds (H6-10 / G4-15).
   - Then confirm the blank-line fix raises the cached fraction above the G4B baseline (75.0 %, 20/180 zero-hit) toward the rev 5 level (84–89 %).
   - Evidence: all three GPT audits (Q4).
9. **Freeze the phase record, heartbeat and branch files at the final cutoff.**
   - Fix: a lock-protected persistence-closed flag checked inside `write()`; the final report built from the last frozen snapshot; one absolute join cutoff (deadline + about 60 s), not a per-thread `+120 s`.
   - If E14-A reuses rev 6g `BranchSession`, check that the H6-4 fix is present. Add a stuck-worker test: file hash, valid JSON and callback count unchanged 5 s after return.
   - Evidence: H6-4 (399 → 544 bytes after return); C6-4 (12 × 120 s = 24 min).
10. **Keep 12 workers in the 25-min Duck run.**
    - Check: the rev 6g isolation preflight drops to 6 workers on FAIL, or on INCONCLUSIVE after one retry (H6-17). Report `isolation_state`, `rev6_workers` and `kv_holds` / `kv_idle_overrides` together, so a 6-worker run is never mistaken for a 12-stream result.
    - Evidence: GPT audit 3; C6-2.
11. **Do not judge greedy parity on token equality at concurrency 12.**
    - Run the A/A same-config control first. Judge on tool-call presence, tool **name** and arguments (C6-3), the same `check` verdict, and batch-size-1 prefixes.
    - Include at least one thinking-on probe at the real first-request cap (12,288), and one tool-result continuation probe (Q2 / A3-Q2). 2,048-token probes rarely reach a tool call (rev 5 first-reply p50 33.6k).
    - Run target-only on its own bounded launch, never alongside production.
    - Evidence: E11-03 (152–166 of 512 shared tokens under batching; an unchanged repeat scored 0.1875 vs a 0.50 threshold).
12. **For the int8 mamba checkpoint arm and any mamba or chunked-prefill change:**
    - Log the #36935 ratio, kv_pool_tokens / chunked_prefill_size: 866,744 / 4,096 = 211.6 vs max mamba cache 56.
    - Log the zero-hit rate.
    - Check that the slots still satisfy 4 per request × 12 (or 13) ≤ the mamba cache (48 or 52 ≤ 56).
    - Mamba usage reached up to 0.70–0.71 in earlier logs.
    - Evidence: G4B audits (#22935, #36935); rev 6c engine notes.
13. **Read the KV pool size from the server at runtime.** Per-stream budgets (window = KV fraction × pool / 12) must come from the server, not from the 881,336 constant.
    - Evidence: the G4B real `rev6_window` 72,228 (≈ 866.7k pool) vs the 73,444 assumed.
14. **Clear the 60k-floor caps with the user if the Duck run is used for a coverage or accuracy comparison.** The rev 6g caps (12,288 first / 24,576 progress / 32,768 rescue; b1 first 24,576; 28k soft first exposure; nudge at 24k through the tool result) are approved exceptions. The 25-min Duck run is a short screening run on these settings; label it so, and never present it as an LB projection (the G4B wording correction).
15. **Watchdog false positives.** The launch-stall watchdog must not treat the 8 benign FP8 "no scaling factors, defaulting to 1.0" warnings, or the benign `wrapt` sitecustomize line, as errors.
    - The cold load was 1,197 s in G4B, under the 1,800 s launch timeout. That sits well inside a 75-min stall-based give-up, but a fixed 30-min launch timeout would be at risk.
16. **Do not add a penalty arm or forced `tool_choice` to E14-A.** Presence, frequency and repetition penalties do nothing under DFlash2 in 0.5.21 (#28180; the DFlash path never accumulates penalty tokens). Required `tool_choice` poisons DFlash batches (#43514) and caused HTTP 500s in Exp 4.
17. **Builder and test hygiene for the E14-A build.**
    - Builder literals need `\\n`.
    - Keep files LF and check before patch asserts (the editor CRLF trap).
    - Run CPU tests with `PYTHONIOENCODING=utf-8` and TEMP on D:.
    - Tests must exit nonzero on FAIL and exit 2 "INCOMPLETE" when `tokenizers` / `transformers` (pins 0.23.2 / 5.20.2) are missing.
    - Never rebuild a Kaggle-run notebook (use `_REBUILD`); check shas after every build.
    - Evidence: H6-5, H6-19; the CRLF assert failure at 10:2x; the 9 protected notebooks.
18. **Check for a gate collision.** The engine cell has `SGL_KV_ADMIT_MAX` 0.85 / `SGL_KV_GATE_MIN_ACTIVE` 10, while the Duck scheduler has `E12_R6_KV_ADMIT` 0.95. Confirm which gate applies to the 25-min Duck run, so that measured holds or admissions are attributed to the right one. Not resolved in this part.
