# Insights from part_01 (2026-10-20 23:35 to 2026-10-21 05:33)

Source: `convo/part_01.txt`, 5,811 lines, read in full in consecutive chunks. Times are the block headers in the file (UTC as written). "Summary" means the context-compaction summary posted at 2026-10-21 00:57. That summary restates user messages from before this part, which carry no exact time.

Truncated tool results that matter (their cut content is not guessed here):
- L94: submission list, "[+977]" cut. Only the top 3 rows are visible.
- L97: AskUserQuestion submission options, "[+650]" cut. The user rejected it anyway.
- L206: budget_sim table, "[+1428]" cut. The full table appears later in the 00:43 reply.
- L274 and L758: pre-check behaviour tables, "[+4353]" and "[+3569]" cut. Their aggregates appear in the 01:00 reply.
- L2822: dfranzen `_yield_resume_prompt_mode` docstring, "[+3454]" cut. It bears on the reversed "Stop deliberating… NOW" claim (section 6).
- L4786: the user's paste of GPT report B was cut by the platform ("[Message truncated - exceeded 50,000 character limit]") in the middle of §9, the serving-risks part, at "...এটি আমাদের producti". The assistant asked for the rest (04:30). Report C plus "B's tail" were then handled at 04:36.

---

## 1. User instructions, rules, preferences and decisions

| When | Verbatim (Bengali/mixed) | English meaning / effect |
|---|---|---|
| 10-20 23:41 | "zip kore de,,,,GPT k dekhabo" | Make a zip; I'll show it to GPT. This produced `E12_REV5_RESULTS_FOR_GPT_2026-10-21.zip` (6.1 MB, 240 files). |
| 10-20 23:50 | "আজকের submission baki" | Today's submission is still pending (not yet done). |
| 10-20 23:53 | (rejected the AskUserQuestion about which notebook to submit) | The user did not want a multiple-choice prompt. |
| 10-20 23:54 | "আরে ১২ টাই তো চালানো jay...update kor" | "Hey, the 12 one can be run… update it." The assistant **misread** this as "12 streams" and built S12 (see section 3). |
| 10-20 23:57 | (rejected the S12 diff/harness tool call), then "foysal Id te submit kor" | "Submit it on the Foysal ID." This was the explicit go to submit EVAL12R5_LOW v1. |
| 10-20 23:59 | "12 -১৬ ঘন্টা লাগে এখন.৩৩ ঘন্টা লাগে না।" | Correction: turnaround is 12–16 h now, not 33 h. Memory was updated. |
| 10-21 00:40 | "একই Qwen3.8-27B + Duck রেখে, reasoning ও tool-use optimize করে প্রতি task-এর token consumption কমানো, 240-task coverage অর্জন করা এবং accuracy বাড়ানো।" + Kris Dreemurr quote "xhigh-এ প্রতি task-এ গড়ে 70k token… প্রতি attempt-এ ~35k… CV-তে ~45%" + "attempt-এ ~35k আমরা এখন ইউজ করলে… বিশ্লেষণ করো যে কি হত" | Fixed goal: same model and Duck; cut tokens per task, reach 240-task coverage, raise accuracy. Analyse what ~35k per attempt would do to our notebook. |
| 10-21 00:46 | "ওকে" | Approves the next step: analyse what happens in the ~55k tokens before the first `check()`. |
| 10-21 02:57 | "প্রম্পট দিও,আর কিছু প্রতিযোগী কভারেজ দিয়েছে,সেটাও লিখে দিবে।… প্রয়োজনে একদম শুরু থেকেই থিংকিং এর সকল ফল মডেল নিজের মধ্যে ক্যালকুলেশন না করে কোড চালাবে Sandbox এর মত কিছুতে।আর যত উপায়তে করা যায়।আমরা আগেও কিছু এক্সপেরিমেন্টে মেবি অপ্টিমাইজ করেছি সেগুলো আবার দেখে আসো,এর পর রিডমি দিও।আর নোটবুক জিপিটির কাছে আগেই আছে" | Write a GPT prompt and list competitor coverage. From the very start the model should compute everything in code in a sandbox, never in its head, "by every possible means". Re-read our past optimisation experiments, then give a README. GPT already has the notebook. |
| 10-21 03:02 | "আমাদের সমস্ত কাজ , পিডিএফ সব কিছু পড়াও,প্রয়োজনে আরো ৫ টা এজেন্ট নিও" | Have all our work read, PDFs included. Take 5 more agents if needed. 8, then 14, reader agents were launched. |
| 10-21 03:28 | (pasted GPT audit 1) "New ফিক্স লিস্ট এগুলো লিখ" | Write these into the fix list as new items (N1–N7). |
| 10-21 03:48 | "জিপিটি কে কি কি দিবো,লিস্ট দে" → "সরি লিংক দে,পাচ্ছি না" | List what to give GPT, then "sorry, give the links, I can't find them". |
| 10-21 04:38 | "এটার আগে তুমি জিপিটি গুলোর পরামর্শ আর তোমার এজেন্ট গুলোর পরামর্শ আবার দেখে আসো,হয়তো গুরুত্বপূর্ণ কিছু বাদ পড়তে পারে।আর ফিক্স লিস্ট আপডেট করো" | Before building, re-check every GPT and agent recommendation because something important may be missing, and update the fix list. This led to the 3 gap-check agents and the T1–T8 traps. |
| 10-21 05:30 | "আগে আলাদা এক্সপেরিমেন্ট ছিলো।সেখানে কাজ করেনি bole ekhane kaj korbe na emon na..." | Standing rule: a past experiment was different, so "it didn't work there" does not mean "it won't work here". Saved as memory `past-negatives-not-verdicts.md`. Past negatives are used only for safeguards and metrics. |

Standing rules restated in the 00:57 summary (from earlier in the session; no exact time):
- "আগেই পরিবর্তন করিও না, আরো একটা দিচ্ছি, আগে দেখে নাও": don't change anything yet; read the next item first.
- "আরো একবার সমস্ত তিন জিপিটির টেক্স পরে আসো… কিছুই বাদ দিও না… মিনিমাল পরিবর্তন": re-read all three GPT texts, leave nothing out, minimal change.
- "তুমি সব না দেখে শর্ট করে দ্যাখো সবসময়, এটা কখনোই করবে না, মেমরিতে সেভ করে রাখ": you always shorten without reading everything; never do that; save it in memory (now `read-everything-fully.md`).
- "আমরা একটা রানে যত সমস্যা পাবো, সবগুলো ফিক্স করে, আবার রান দিবো… এখন যে পরিবর্তন আমরা ইভিডেন্স পাবো সেটা করবো": fix every problem found in a run, then run again; make only evidence-backed changes (shorter output too, if evidenced).
- "(a) temperature 0.7 থেকে 1.0 … এটা আমি নিলাম": the user took temperature 0.7 → 1.0.
- "একই নোটবুক, শুধু ~Qwen xhigh agent পরিবর্তন করবো": same notebook, change only the agent effort. The user chose **low**.
- Abusive message (summary item 19): "…এই মডেল ও এই ডাক দিয়েই ৮০+ এভালুশনে আর ৭০+ এলবি পাওয়ার প্রমাণ আছে। তারা পুরো কভারেজ দিয়েছে। আমরাও পারবো ইংশা আল্লাহ। আবার ডিস্কাশন পড়ে যায়": there is proof that this model plus Duck reach 80+ eval and 70+ LB with full coverage; read the discussions again.
- "…তুই যেখানে দেখতে পারিস না, সেখানে তর্কে যাস কেন, বিশ্বাস করে নে। মেমরিতে সেভ করে রাখ… ওরা সবাই Qwen3.8 + Optimized Duck Use করছে": don't argue where you can't see; believe it; they all use Qwen3.8 + optimized Duck. Saved as `trust-user-observations.md`.
- Summary "Standing constraints": no push/run/submit without an explicit go; "অনুমতি ছাড়া কমিট করবি না" (don't commit without permission); never read or echo token contents; validate first and bundle changes; make the audit zip before pushing; scratch on D:; Bengali chat; 60k thinking floor unless evidence-based and approved; pre-run capacity check; read everything fully; trust user observations; **never propose running Duck on fewer than all tasks**.

## 2. Measured facts and results

**Rev 5 GPU runs (reported 10-20 23:35, re-verified later):**
- **rev 4 EVAL6 (before):** 4/8 inputs, 3/6 tasks, 50.0 %, 1.22 M tokens, 101k per branch.
- **EVAL6R5 xhigh (hosen42, sha f4fed1ac):** 5/8 inputs, 4/6 tasks, 66.7 %, 846,080 tokens (−30 %), 70.5k per branch, 127 requests, 58 min solve, 244 tok/s end-to-end, cache 84.3 %.
  - attempt_2 gave **3 of the 5 hits** (b5ca7ac4, 62593bfd). In rev 4, attempt_2 was used 0 times in 56.
  - Loops 0–3 %, down from 29 %. Token-estimate error +0.4 % median. 0 timeouts, 0 cut tool calls.
- **EVAL12R5_LOW (Foysal, sha 26fff41e):** 10/19 inputs, 7/12 tasks, 58.3 %, 1,743,889 tokens, 72.7k per branch, 449 requests, 110 min, 265 tok/s, cache 88.7 %.
  - It solved 898e7135 for the first time with Duck.
  - On the 6 shared tasks: xhigh 4/6, low 3/6, union 5/6. Only 4c3d4a41 stayed unsolved.
- **4c3d4a41:** every branch in both runs accepted a train-exact, test-wrong program. Rev 4 had solved it (2/2).
- **Coverage:** ~9–10 min per task vs ~2.8 needed for 240 tasks, i.e. **3.4× short**.
  - Context per request p50 46–58k, p90 ~75k.
  - 10 streams fit p90: 750k of 867k KV. 12 fit the median (697k) but not p90 (890k).
  - 6→12 running measured 368 → 431 tok/s.
- **Branch lengths:** failed branches run to ~105–115k; successful ones end at 14k–98k.

**Submissions:**
- Visible at 23:51 (L91–93):
  - 10-19 23:32 "Q27 XHIGH E10 S1 55K Poetiq" COMPLETE **31.81** (foysal). The assistant corrected the memory record for S1-55K.
  - 10-18 E11-06 P1 60K **29.17** (esfersami50).
  - 10-17 S1 49K **32.08**.
- **10-20 23:58:34:** submitted `foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r5-low` v1, **ref 57456562**, PENDING.
  - Limits before submitting: numToday 0, numTotal 94, numAllowedNow 1, pending 0.
  - The hidden rerun runs 240 tasks, low effort, 6 branches, 49k output, T 1.0.
  - Still pending at 05:33.
- **LB (10-21 ~02:59):** we are **87th with 34.58**.
  - Top: Tufa Labs 88.06 (56 entries), Yi-Chia Chen 84.72, rabbithole 80.83, nvbanana 79.17, Majkel1337 75.56, ALbert RUehlman 72.64, ｶﾎｳｼｭｳｾｲ 72.50, _hans 71.81, Domdolus Tolus 71.53…
  - LB monitor JSON: the second value is minutes_taken; values above 750 are invalid.

**35k-per-attempt simulation (00:43, `budget_sim.py`):**

| Cap per branch | xhigh tasks | xhigh tok/task | xhigh 240-task need | low tasks | low tok/task | low 240-task need |
|---|---|---|---|---|---|---|
| 15k | 0/6 | 71k | 19 h | 1/12 | 50k | 12.5 h |
| 35k | 1/6 | 91k | 25 h | 2/12 | 83k | 21 h |
| 50k | 2/6 | 106k | 29 h | 4/12 | 99k | 25 h |
| 70k | 4/6 | 125k | 34 h | 5/12 | 124k | 31 h |
| none | 4/6 | 141k | 38.5 h | 7/12 | 145k | 36.5 h |

Time available: 11.3 h.
- **Soft caps overshoot:** a request that crosses the cap still runs to its end (often 30–49k at xhigh).
- **xhigh first reply:** p25 29,363, **p50 33,584**, p75 47,875, max 49,152 tokens.
- **low first reply:** p50 199 tokens, but a median of 8.5 requests come before the first check.
- **Tokens until the first check():** xhigh p50 55,070. First exact candidate at p50 58,066. Correct candidates came at 32,469 / 35,059 / 53,941 / 58,066 / 62,115.

**Pre-check analysis (00:48–01:00):**
- 308 tool calls before the first check: 127 inspect/print, **110 own-tests of a rule** (invisible to harvest), 44 helper functions, 27 test inspections.
- First check, median: xhigh **55.1k**, low **53.7k**.
- Reasoning after the model already said "code it": xhigh **37 %**, low **20 %**.
- Hand-count misreads noticed by the model: 12/12 xhigh, 23/24 low (regex-based).
- 2 branches in each run hit 49,152 before checking.
- **Low reader report (00:54):**
  - 4c3d4a41_b0: R4 used 49,152 tokens with no tool call. In R6 the prompt shrank from **66k to 19.9k** (history trimmed), and the next reply accepted a hard-coded overfit transform.
  - 62593bfd_b0: R1 used 49,152 tokens with 0 tool calls while saying "write code" six times.
  - 65b59efc: the rule was stated at ~27k but checked at 53.6k (~26k delay), plus a misread "plus" pattern.
  - cbebaa4b: ~24k spent on own-compare code.
  - Fast branches: bf45cf4b 14k, 247ef758 24k, b5ca7ac4 20k.
- **Correction (01:00):** xhigh 62593bfd_b0 actually checked at 58k and was accepted 2/2. The branch that never checked is 135a2760_b1.

**Past experiments (reader reports, 03:05–03:17):**
- **Exp 2 R6:** 2/22, 92,679 tokens per task (28 h for 240), 46/47 turns force-closed, 220.7 tok/s, MTP acceptance 0.678.
- **Exp 4 run 4 (v1.8.3):** agent 0/22.
  - 664k tokens, 27.7k per attempt, 304 turns, 59 empty-code submits, 5 vLLM HTTP 500.
  - **17/24 xhigh first turns hit the 10,240 cap with no tool call**, even with IN_PYTHON_RULE on. Median first python call at 10,498 tokens.
  - After a cut plus a terse hint, the next turn called a tool 17/17 (24/24 counting medium). The assistant verified this in attempts.jsonl.
  - Prefill 2.78 M vs 0.66 M generated.
- **Exp 5:** 1/22. The thinking budget was not enforced: 10/12 requests ran to the 57,344 cap (MTP bypasses the logits processor).
- **Exp 1:**
  - v1.3.5: 1/22, 12/12 at the 49k cap, 247.8 tok/s.
  - v1.4: LB 18.75. 4 natural closes gave 2 exact + 2 one-cell-wrong; 18 forced closes gave 0.
  - v1.4.1: 285.3 tok/s, prefix hit 0.786.
  - v1.4.4: LB 26.94, 274.6 tok/s, KV 78.7 %, setup 789 s.
- **Exp 6:** identical code gave 2/2/3/1 hits, so EVAL12 cannot resolve ±1–2 hits.
- **Exp 7:** ordering-only LB 27.92.
- **Exp 8:** a prompt_logprobs request OOMed all 4 GPUs (790 MiB all-gather). Natural closes hit 60 %, forced 3.8 %.
- **Exp 9:** 86k gave LB 22.92; 306.6 tok/s; KV 91.4 %.
- **Exp 10:**
  - ORDER2 EVAL120: 58/172, 36.94 %, 333 tok/s.
  - Poetiq R5: 48.61 % eval but LB 21.39, 281 tok/s, KV 99.5 % with 9 preemptions. 22.7 % of thinking is digit/coordinate tokens; 58 of 104 cap-hit traces never wrote `def transform`.
  - S1-60K: LB 34.58.
  - W3H low effort: 9.88 %, with 78 % of requests emitting EOS mid-thinking.
- **E11-03 (SGLang):**
  - The real running cap comes from mamba slots: A 22, B 10, C (DFlash2, fp32 SSM) **6**, D (fp16 SSM) 13, while `/get_server_info` reported 32.
  - D throughput at 1/4/8/12 running: 107.5 / 278 / 356 / **404** tok/s.
  - Accept length: MTP ~3.1; DFlash2 (block 8) 4.5–4.95.
  - **Custom allreduce is disabled on 4 PCIe GPUs → TP4 uses NCCL.**
  - Cold prefill ~1,400 tok/s; forced-close continuations ≥0.979 cached.
- **E11-04 (W1):**
  - Flags: `extra_buffer_lazy`, mamba cache 56, fp16 SSM, `--disable-prefill-cuda-graph`, mem 0.83, max running 12, DFLASH 8 with the **unquant drafter (mandatory, else acceptance ~1.0)**.
  - Result: **12 really running, KV 881,336**, 2.22 GB free. fp32 SSM gave only 599k.
  - PROD at 12 in flight: **416.5 tok/s** (<10k context 458, 10–20k 413, 20–40k 388).
- **E11-05:**
  - 328.9 tok/s overall; decode at 40–60k context **323**.
  - **The server ran 11 of 12:** 2,358 decode lines show 11 running + 1 queued, only 114 show 12. The 12th waited up to 24 min, costing ~8 %. Root cause unverified. Candidate fixes: `--max-mamba-cache-size 64`, or server 13 / client 12 (already used by E12).
  - Aborted-request drains reused the cache (52–99.7 %) only when the client waited for each stream to end.
  - `--stream-interval` must be 1 with DFLASH.
- **E11-06:** 7/22 vs 5/22 (not significant), 89 % of iterations hit 60k, 379.9 tok/s, 87.2 % of prompt tokens cached.
- **Archive (vLLM era):**
  - CUDA graphs took 48 → 129 tok/s at 8 sequences.
  - Custom allreduce at TP4 had no effect (0.9995×).
  - Prefix caching at 64 sequences killed the engine; TP2 gave 17k KV and failed.
  - PIV EVAL120: LB 1.67, 65.3 tok/s, 29 % of the time with ≤2 requests active, 49/120 zero-tool tasks.
  - V3.1-S12: 0/12 zero-tool, first tool after a median of 664 tokens, solves unchanged at 1/12.
  - EVO12 (3k cap): 0/12, 29.3k tokens per task.
- **E12 runs (03:17 table):**
  - SMOKE1: load 677 s, 204 tok/s at 2 running, accept 5.2.
  - vLLM SMOKE1 run 1: 0 requests (served-name `qwen` alias, 404s).
  - EVAL30: 4/30 tasks, 7/48 inputs. 2.36 M generated, prompt 23.78 M (9.60 M uncached). Branch 1 never started in 29/30 tasks; 40 timeouts; **finished 1,412 s past the deadline**; decode 431 at 12 running; overall 187 tok/s.
  - EVAL6 rev 4: 325 tok/s.
- **ARC-3:**
  - P1_SPARSE: 1,477 generated tokens per call, 1,340 calls, ~241 tok/s, prefix caching off (27.6 M prompt vs 1.98 M generated).
  - dfranzen v3: 46.49 mean, 17.5 M tokens / 100 trajectories in 8h11m.
  - dfranzen serving: mem 0.96, page-size 64, max-running 10, CUDA graph batch sizes {1,2,4,7,8,9,10}, mamba cache 60, lpm, fp8 KV, NEXTN 3 + FR-Spec, `--weight-loader-prefetch-checkpoints`, `--weight-loader-drop-cache-after-load`. SGLang needs ~10.5 min to load.
- **NVARC:**
  - The same version scored 27.36–32.64 over 10 submissions (mean 29.36; noise ±2.5).
  - EVAL120: NVARC 27.92 %, Q27 union 56.81 %, NVARC + Q27 union 61.81 %. NVARC takes ~9.5 h for 240 tasks.
- **Public notebooks:**
  - russcore LB 10.14: 36,257 tokens and 2,279 s per sample, 119.5 tok/s at ~7.5 running, 20.2 h projected for 240.
  - delongyang: TRITON_ATTN + env pin; `LD_PRELOAD` of the real libcuda (driver 580, not the 570 compat copy).
- **Token-share check (03:31):**
  - xhigh: requests ≥10k tokens are 28/127 but carry 80 % of tokens.
  - low: requests ≥10k are 45/449 and carry 58 %.
- **Selective-branch replay (03:34):**
  - xhigh: 5/8 at 498,253 tokens (−41.1 %), branch 1 needed only for 898e7135.
  - low: 9/19 at 1,094,372 (−37.2 %). 135a2760 is lost (branch 0 was train-exact but test-wrong).
  - Late-correct cases: 135a2760_b1 71,306 → 97,972; 898e7135_b0 56,868 → 73,523; 898e7135_b1 77,859 → 79,662.
  - Oracle coverage equals the final score (5/8, 10/19), so selection loses nothing.
  - Median completion per request: xhigh 1,115, low 1,090. Median prompt: 51.9k / 41.6k. Weighted cached fraction 85.4 % / 91.0 %. Uncached tokens 888,297 / 1,576,650.
- **Grid text:** median 3,730 characters per task. Only 10 of 87 grids are ≤64 cells, and 0 of 12 tasks qualify for report C's "≤8×8 and ≤256 cells" text exception.

## 3. Failures, mistakes and root causes

- **10-20 23:55: builder quoting bug, a repeat.** `' + str(LIVE) + '` was written instead of `''' + str(LIVE) + '''` in the `'''` cell string. The same bug happened earlier for `_E12_REV5` and `EFFORT`. Fixed.
- **10-20 23:55: a Kaggle-run notebook was overwritten.** Rebuilding LOW replaced the run file (new sha ca377682). It was restored from the results zip to sha **26fff41e**. This is the origin of the "never rebuild run notebooks" rule.
- **10-20 23:54–23:58: user intent misread.** "১২ টাই তো চালানো jay" was taken as 12 streams. The S12 notebook (sha 216c59ed) was built, untested and not pushed. The assistant admitted it at 23:58.
- **10-20 23:58: stale platform fact.** The assistant said the queue "may take ~33 h"; the user corrected it to 12–16 h.
- **Earlier wrong claims, from the summary:**
  - "the newest block always stays";
  - "no failed branch got a train pair exact";
  - "4 timeouts" (actually 40);
  - "temperature 0.6 is Qwen-like" (the official value is 1.0);
  - the S1-55K record (it was submitted and scored 31.81).
- **User anger, from the summary:**
  - The assistant proposed not running Duck on all tasks and argued about the LB.
  - The W3H low-effort evidence was omitted when recommending low.
  - Skimming.
- **01:00: wrong branch in the assistant's list.** 62593bfd_b0 had been called never-checked, but it was accepted at 58k. The cause: a failed check does not enter the `checked` list.
- **03:09: reader agent a986b1e54aaef8bcb handed back unfinished.** Its sub-readers' results were lost when it ended, and the work was re-split across 4 new agents. Lesson: a reader that spawns its own sub-agents can lose their output.
- **03:22: wrong claim corrected in code.** "'Stop deliberating… NOW' is sent in dfranzen's default mode" was false. The assistant had repeated it to the user at 03:11 and in the table at L1509. The code reading that corrected it (L2822) is cut in the extract.
- **03:22: the 6.4 % metric was narrow.** It counted only raw digit-row lines. The real hand-reading share is 22.7 % (digit tokens) or 45–50 % (paragraph level). So the 10-16 decision to downgrade the fact sheet does not apply to Duck.
- **04:36: PIV myth corrected.** "Qwen must finish a full solve before a tool call" is not a template rule, only a tendency. The assistant had repeated the PIV claim.
- **Bash heredoc failures:** L3410 "unexpected EOF while looking for matching `''", redone with Write. L929: a json.load cp1252 decode error, redone with utf-8.
- **Root causes of the token waste (01:00):**
  - hand counting and misreads;
  - "I'll write code" followed by more thinking;
  - a known rule checked late;
  - own `p['output']` compares;
  - small-error debug loops (898e7135_b0: 6 of 15 calls failed on forbidden `sys`/`traceback` imports and IndexError);
  - context loss when 66k prompt + 16k reply + a 49k reserve exceeds 131k, so the trim drops history and leads to an overfit.
- **Source-level weaknesses confirmed in code (03:34 / 04:30 / 04:47–04:52):**
  - The branch stop condition sees only `arc2_accepted`, not `arc2_checked`.
  - The resume text "Nothing has been submitted yet" is false once a candidate is harvested.
  - `run_branch()` builds a fresh agent each time, so a pause/resume would lose state.
  - **Scheduler trap:** with `live_branches=None`, all 480 branches start at once (`arc2_duck.py:976`), and the queue position is tied to the same switch (:984).
  - **T1:** helpers in the prelude are excluded from retention (`PRELUDE_NAMES`, :671–674), and the host verify (`_grid_call_code`, :216; :768–774) runs only retained sources. A `transform` that calls a helper therefore gets a NameError and is rejected (the 136b0064_b1 `patmap` reject).
  - **T2:** the facts block must go in the system prompt, because the first user message gets trimmed.
  - **T3:** auto-check keyed on the name `transform` misses `transform2` / `transform_alt`.
  - **T4:** Duck's board guard misses space-separated rows.
  - **T5:** the dedupe key hashes all retained sources, so any helper edit re-verifies the same transform and burns the cap of 24. At the cap, the note says "submit your best function", which ends the branch.
  - **T6:** Duck's nudge prefix still says "You have not acted yet. Investigate first." (`tool_agent.py:6983`).
  - **T7:** no stage-2 rank for one-candidate tasks, though about a third of accepted programs are test-wrong.
  - **T8:** EVAL30 ran 1,412 s past its deadline, cause unknown.
  - **4c3d4a41 overfit:** the model wrote at L1706 "+10 is specific to the training layout" and still accepted it. In the xhigh run, b1 wrote `transform_alt` but never called `check(transform_alt)`.

## 4. Decisions and reasons; options rejected

- **10-20 23:58: submit EVAL12R5_LOW v1 (Foysal, ref 57456562),** on the user's explicit go. This is Duck's first real LB, to learn hidden-set speed and reach.
- **00:43: a lower cap alone is rejected.** A 35k cap collapses accuracy (xhigh 4/6 → 1/6) and still needs 21–25 h. The lever is to make the agent write and check code earlier. To truly bound tokens, a hard per-request `max_tokens` = remaining budget would be needed (soft caps overshoot).
- **01:00: Rev 6 candidates R6-1..R6-6.**
  - R6-1..4 are prompt/helper-only and need no approval: facts block, check in the same reply, no own compares plus `check({'A':fA,'B':fB})`, line numbers in check errors plus an allowed-imports list.
  - R6-5 (stop the stream on "I'll write code" after ~3k, or cap the first request at 12k) and R6-6 (change the 49k reserve once context passes ~80k) **need approval** under the 60k floor.
- **03:14: no named/forced `tool_choice`.** Exp 4 got empty code and vLLM HTTP 500 from it, and SGLang #43514 (claimed) reports a DFlash poisoned batch. Use auto tool choice and auto-check.
- **03:31 (GPT audit 1 → N1–N7):**
  - N1 progress-aware budget (P0); cutting an "unproductive" branch's budget needs approval.
  - N2 coverage-first admission (P0); N3 state-aware nudges; N4 stage up to 10 streams with KV and isolation tests; N5 generalization feedback; N6 DFlash2 correctness preflight; N7 logging.
  - Selection rewrite rejected: oracle = final, so a better ranker cannot help.
- **03:34 (GPT audit 2 → C1–C10):**
  - C1 first-exposure-first, branch 1 conditional; C2 expose harvested progress; C3 global budget at request boundaries; C4 keep Duck state across phases; C5 short consistent prompt; C6 grid helpers; C7 10 streams with cache-aware admission; C8 overfit telemetry; C9 accounting.
  - C10 (judge ensemble, runtime rewrite) is "not now".
  - GPT's avoid list: no 30k universal cap, no auto-stop at the first train-exact, no merged branch contexts, no evaluation truth in routing, no wholesale ARC-3 priority formula.
- **04:30–04:36 (three code-first GPT reports A/B/C):** the agreed bundle:
  - grids out of the prompt with exact arrays in Python;
  - a facts block using "modal colour" (not "background") and both 4- and 8-connectivity;
  - helpers;
  - state-aware resume with a fixed precedence (cut → trimmed → verified → partial → no-tool);
  - auto-check that is never an auto-submit;
  - a short prompt without staged instructions;
  - **context reserve = the actual `max_tokens`;**
  - branch 0 for all tasks first;
  - 10 streams after qualification;
  - one generalization reminder;
  - a serving preflight;
  - all rev 5 safety fixes kept.
- **04:36: the assistant's own correction.** Duck's reply reserve follows `max_output` (`tool_agent.py:3498`), so a uniform cap stays in sync automatically, but a per-request varying cap (12k → 24k) needs both values changed together.
- **04:52:**
  - All items renamed R6-xx, because the IDs B1/B2/C1/F1 collided.
  - ORDER2 for first exposure; progress measured against the identity program.
  - "Facts, not advice"; nudges only after a cut or yield.
  - EVAL6/12 used as a sanity check only.
  - New **gate 4b**: eval at the hidden-equivalent budget.
  - Regression cases for multi-test tasks and previously successful trajectories.
  - A "not in rev 6" list: CPU rule library, images, judge, text-trigger cancel, effort ladder, context compaction, NVARC filler, PIV no-tool commit, and so on.
- **05:33 (after the user's 05:30 rule), recommendation:**
  - R6-Q1: 12,288 until the first tool call, then 24,576, with the branch total ≥60k and thinking kept in context.
  - R6-Q2: pause after 25–40k is kept as a testable option, enabled if the pilot's first-check median is ≤25k.
  - R6-E: xhigh by default.
  - Branch 1 (rescue) shows rev-5-style text grids, to hedge A1 and to learn which view solves which task.
  - Waiting for the user's go.
- **Rejected or deferred:**
  - presence penalty (inactive under DFlash in 0.5.21; needs a 2-line SGLang patch);
  - text-heuristic early stop (AUC 0.51–0.64);
  - medium effort;
  - the server thinking budget;
  - `lpm` used blindly (#31954, cold-request starvation, claimed);
  - copying dfranzen's bf16 SSM (keep fp16);
  - the 64-cell or "≤8×8" text exceptions (almost no eval task qualifies).

## 5. Open items at the end of this part

1. **Submission 57456562 (E12 rev 5 LOW) LB** is pending (expected 12–16 h after 23:58 UTC 10-20).
2. **The GPT report on `E12_REV5_RESULTS_FOR_GPT_2026-10-21.zip`** was received as audits 1 and 2. **The code-first GPT report** came as A, B and C. B's §9 tail was truncated by the 50k paste limit; the assistant marked "B-tail addendum" at 04:36, but the transcript does not show the missing §9 text was ever re-pasted.
3. **Three user decisions are still unanswered at 05:33:** R6-Q1 (cap ladder), R6-Q2 (pause after 25–40k), R6-E/branch-1 text view, plus the facts-block size (600–800 vs 1,200–1,500 tokens). The assistant offered to measure it with the Qwen tokenizer on 120 eval tasks.
4. **SGLang issues** #38009, #36548, #37475, #39087, #43514, #31954 and DFlash #166, plus HF discussion #173 and the Franzen write-up numbers: all **unverified**, kept as GPT claims.
5. **T8:** cause of the EVAL30 1,412 s deadline overrun is unknown. "Must be found before a hidden run."
6. **The `crop()` argument order differs between drafts:** `(r0,r1,c0,c1)` vs `(r0,c0,r1,c1)`. Not chosen.
7. **Not yet chosen:** a 10-stream KV cap of ≤~79k per stream (10 × 118k input budget ≈ 1.18 M > 881k KV); the R6-6 fallback window of ~150k if B1 is declined.
8. **Pending since rev 2/3:** record `verify_sec` / timeouts; the "2 exact + 1 timeout must be rejected" test.
9. **The S12 notebook (216c59ed):** built, untested, not pushed. Do nothing without direction.
10. **E13 Flash-Next SMOKE1:** awaiting audit and go.
11. **N6/G5/G8 parity design:** the parity gate fails by design without a control. Needs a same-config A/A first, batch-size-1 prefixes and a separate target-only launch.
12. **Unexplained:** the 11-of-12 admission in E11-05 (root cause not verified in this part).
13. **The ARC-3/public-notebook "Concrete techniques" section** of the failed reader was re-done by 4 agents. The deferred A4 CPU rule library has numbers (30 ms per task, 5.3–5.7 % on real pairs).
14. **Stale doc text fixed at 04:52:** the header status, the LOW sha cce9cb8f→26fff41e, F7, F10, F2(c).

## 6. Insights a fresh reader might miss

- **Low effort is not cheaper.** Per branch: low 72.7k vs xhigh 70.5k tokens, with 449 vs 127 requests, and more uncached prefill (1.58 M vs 0.89 M). The official Qwen card is said (by GPT) to warn of this for multi-turn agents.
- **The tool is used, just unproductively.** 92–97 % of requests contain a tool call. The waste is the long pre-call thinking (28/127 requests carry 80 % of xhigh tokens) plus own-compare calls that harvest cannot see.
- **The prompt can cause the delay.** Rev 5's "if rules disagree on a test input, compare their test outputs" line invites hand simulation before any check.
- **Selection is solved, generation is not:** oracle = final in both rev 5 runs.
- **Rev 5 fixed some tasks and broke 4c3d4a41** (rev 4 had solved it). Bundled changes hide which one caused it, so keep old successful trajectories as a regression corpus.
- **Approval scope:** "approved exception 12,288 (user 2026-10-19)" in the settings doc applied only to rev 1 / EVAL30. Approval does not transfer.
- **Reversed claims:**
  - "Stop deliberating… NOW" is sent by default (said 03:11, reversed 03:22).
  - "Qwen must solve before a tool call" (corrected 04:36).
  - "6.4 % hand reading, so downgrade the fact sheet" (overturned 03:13/03:22).
  - 62593bfd_b0 never checked (corrected 01:00).
  - Past negatives presented as "against" verdicts (corrected by the user at 05:30).
- **Two definitions of "first check" coexist,** 55.1k and 56.6k. Pick one for the gate.
- **dfranzen's strongest "think less" prompt lines are OFF** in his winning submission (PREFER_TOOL_CALLS, BOARD_DUMP_GUARD). What is on and structural: the 12k max_tokens, the 2,048-token yield, preserve_thinking / reasoning_content history, the drain hysteresis at 58k for the prefix cache, the warm-prefix priority gate, retained functions, tool output capped at 3,072 tokens with middle truncation.
- **Prefix-cache killers (dfranzen, measured in his comments):**
  - a request without tool definitions cuts shared prefix from 99 % to 15 %;
  - toggling `enable_thinking` changes the "Reasoning effort is set to…" sentence at the top of the system message (75 % → 45 % reuse);
  - llama.cpp drops `reasoning` (69 vs 3,670 prompt tokens);
  - front evictions invalidate everything, so evict rarely and in big blocks;
  - Python `hash()` ordering changed the LB between reruns in a hosen42 notebook.
- **Engine-level traps:**
  - `/get_server_info` reported 32 while the real running cap was 6 (E11-03).
  - The 11-of-12 pattern (E11-05) is a real ~8 % loss.
  - The thinking budget is silently ignored under MTP (vLLM).
  - The `unquant` drafter flag is mandatory (otherwise acceptance ~1.0).
  - `--stream-interval 1` is required with DFLASH.
  - T 1.0 lowers acceptance (DFlash 4.60 → 4.34; MTP 3.0 → 2.7 tokens per step).
  - Acceptance falls with context (4.43 → 3.77 at 40–60k) and decode slows ~25 % (429 → 323).
- **The parity test is noise-dominated.** Spec vs plain greedy outputs share only ~152–166 of 512 tokens because of batch nondeterminism. An unchanged-config repeat scored 0.1875 against a ≥0.50 threshold.
- **Coverage arithmetic tension:** 240 tasks × 60k = 14.4 M tokens, 15.1 h at 265 tok/s, ~10 h at 400. Two branches × 60k = 28.8 M, 18.6 h even at 430. Full coverage needs natural early completion plus selective branch 1 plus more throughput.
- **The KV fit at 10 vs 12 streams** depends on per-stream resident context. p90 75k × 12 = 900k > 867k, so 12 streams are borderline.
- **The user's view is fixed** (trust-user-observations): top teams use Qwen3.8 + optimized Duck with full coverage. Never argue it; never propose running Duck on fewer than all tasks.

## 7. Relevance to the current E14-A notebook (concrete)

1. **Check running == 12 in the decode log, not in server info (baseline windows, min-free-slots arm).**
   - Evidence: E11-03 `/get_server_info` said 32 while the real cap was 6. E11-05 showed 11 running + 1 queued in 2,358 decode lines, with only 114 at 12, and a 12th request waiting up to 24 min (~8 %).
   - Fix/guard: a baseline window counts only when `#running-req == 12` for the whole window (log the share of windows that qualify). The `--min-free-slots-delay 1` arm must be judged on that same running-count metric.
   - If the arm still shows 11/12, part 1's other unverified candidates are worth a quick arm: `--max-mamba-cache-size 64`, and server 13 / client 12.
   - Keep the mamba-slot math visible: extra_buffer_lazy uses 4 slots per request, so cache 56 allows at most 14 requests.
2. **Compare arms at matched context and temperature.**
   - Evidence: W1 throughput depends on context (458 / 413 / 388 / 323 tok/s for <10k / 10–20k / 20–40k / 40–60k). DFlash acceptance drops 4.43 → 3.77 at 40–60k, and T 1.0 lowers acceptance vs T 0.6.
   - Guard: every baseline/arm window should log mean context length, accept length and temperature. DFlash width 6/4, CUDA graphs 1..13, the int8 mamba checkpoints and the NCCL env arm must use the same prompt set (token-level replay) at T 1.0, top_p 0.95, top_k 20, or the comparison is confounded.
3. **Calibrate the greedy-parity threshold on the A/A result; add a marker-leak check.**
   - Evidence: speculative vs plain greedy share only ~152–166/512 tokens because of batch nondeterminism. A prior unchanged-config repeat scored 0.1875 vs a ≥0.50 threshold, so a fixed threshold fails a healthy engine.
   - Guard: E14's A/A at concurrency 12 must set the pass bar (target-vs-DFlash agreement within the A/A spread), never a fixed constant. Report batch-size-1 prefix agreement separately.
   - Add a cheap **distinct-marker cross-request leak check** at concurrency 12 (SGLang #36548 class; the E11-04 version passed).
   - Static-check that the drafter config has explicit `target_layer_ids` (#37475, claimed).
   - Keep the `unquant` drafter flag in every DFlash arm, including width 6/4. Without it, acceptance falls to ~1.0, which would look like a "width" effect.
4. **Show that the raw-reasoning history fix improves cache hits in the 25-minute Duck run.**
   - Evidence: rev 5 uncached prompt tokens exceeded generated tokens: 888k vs 846k (xhigh), 1.58 M vs 1.74 M (low); weighted cached fraction 85.4 % / 91.0 %. dfranzen's measured prefix killers: tools omitted (99 → 15 %), template conditional sentence (75 → 45 %).
   - Guard: log per-request `cached_tokens / prompt_tokens` and the total uncached tokens in the Duck phase, and compare with rev 5 (85.4 % weighted).
   - Assert that every request of a branch sends the same `tools` list and the same `reasoning_effort` / chat-template kwargs. Any change in the system-message header breaks the prefix from token 0.
5. **Add a hard deadline for the Duck phase and the 200-minute profile.**
   - Evidence: EVAL30 finished **1,412 s past its deadline**, cause unknown (T8/G20). Duck requests are non-streaming with read timeout = min(analyzer_timeout, left); a 49k reply at ~35–61 tok/s per stream takes ~800–1,400 s.
   - Guard: the 25-minute Duck phase needs a last-admission cutoff (stop launching new branches / requests well before the phase end) and a phase-end abort that waits for aborted streams to drain. Per E11-05, waiting for an abort to finish keeps the cache: 52–99.7 % vs 0 %.
   - Record any overrun in the phase record / heartbeat. The measurement phases then must not run past the 200-minute budget.
6. **Check KV capacity before the 25-minute Duck run at 12 streams.**
   - Evidence: rev 5 context p90 ~75k, and 12 × 75k = 900k > 866,744 / 881,336 KV. With the Duck window 131,072 and a 49,152 reserve, resident context can reach ~130k per stream. E12 EVAL30 KV thrash re-prefilled 8.4 M tokens; branch 1 never started in 29/30 tasks.
   - Guard: in the pre-run capacity estimate, either run the Duck phase at 10 streams, or log retractions/preemptions and the real running count. Do not read a thrash-limited Duck tok/s as an engine regression.
7. **Make the JIT cache robust to libcuda linking.**
   - Evidence: a past fork run died because FlashInfer could not link `-lcuda`. delongyang's notebook `LD_PRELOAD`s the real libcuda (driver 580, not the 570 compat copy).
   - Guard: the JIT cache restore and any fresh JIT compile must resolve the real driver libcuda. Fail-open means "compile normally", never "launch without kernels". Log which libcuda was loaded.
8. **Cover the full server batch range in the CUDA-graph arm.**
   - Evidence: E12 W1 uses server max-running 13 / client 12. dfranzen lists graph batch sizes up to and including his max-running (…, 9, 10).
   - Guard: the 1..13 arm must include 13 (and 12) so that no decode step at the server maximum falls back to eager.
9. **Keep the SSM dtype at fp16 in the int8-mamba-checkpoints arm.**
   - Evidence: fp32 SSM gave KV 599k vs 881k for fp16; memory says fp16, not bf16; the gap-check said not to copy dfranzen's bf16.
   - Guard: the int8 arm changes only the checkpoint format. Assert the SSM state dtype is still fp16, and re-check the real running count and a correctness sample (the parity phase), since slot or state changes caused the original 6/13 caps.
10. **Set expectations for the NCCL bench and NCCL env arm.**
    - Evidence: custom allreduce is disabled on 4 PCIe L4s, so TP4 runs over NCCL; custom allreduce at TP4 had no effect (0.9995×) in the vLLM era.
    - Note: the NCCL bench/env arm is the only all-reduce lever. A large gain from the env arm should be cross-checked against the replay at matched context (item 2).
11. **Keep `--stream-interval 1` if the replay or latch test streams.**
    - Evidence: larger stream intervals can hold back text with DFLASH (E11-05).
    - Guard: if token-level replay or the latch test measure timing by streaming, keep `--stream-interval 1`, or timing and latch readings are distorted.
12. **Verify startup flags exist in 0.5.21 before using them.**
    - Evidence: dfranzen used `--weight-loader-prefetch-checkpoints` and `--weight-loader-drop-cache-after-load` on the Pennyroyal fork. SGLang loads in ~10.5 min; SMOKE1 load 677 s; GCSFUSE-era ready 1,197 s.
    - Note: these are options next to E14's /dev/shm copy, not requirements. Verify that the 0.5.21 parser accepts them before adding them (GPT also warned that main-branch features may be missing in 0.5.21).
13. **Use one definition of "first check" in the Duck run.**
    - Evidence: two definitions coexist (55.1k vs 56.6k). The rev 5 resume text "Nothing has been submitted yet" is false after a harvest, and Duck's nudge "You have not acted yet. Investigate first." contradicts code-first.
    - Guard: if E14's 25-minute Duck run reports first-check tokens, cached fraction and candidates, use one definition (cumulative completion tokens at the first `check()` call), and confirm that the rev 6g solver no longer sends the two misleading texts.
    - The full thinking traces must be saved (standing rule).
14. **Use comparable throughput references.**
    - E11-04 PROD 416.5 tok/s at 12 (short context); E12 decode 431 at 12 running; E11-05 323 at 40–60k; rev 5 end-to-end 244 (xhigh) / 265 (low).
    - The E14 baseline at 12 running should be compared with the context-matched number, not with 416.5.
