# Insights from part_00 (2026-10-20 02:21 to 2026-10-20 23:26 UTC-ish session timestamps)

Source: `part_00.txt`, 7,078 lines, read in full (lines 1-7078). The part ends in the middle of the rev 5 results analysis: the per-branch tables at lines 7064, 7070 and 7074 are cut (`…[+4320]`, `…[+865]`, `…[+68]`), and no rev 5 result message to the user appears in this part. Where a cut `[result]` matters, this file says so and does not guess.

Timeline in brief: GitHub push, then E13 Flash-Next simple SMOKE1 built (not run), then back to Qwen3.8-27B Duck: rev 4 simulator, EVAL6 built, 3 GPT audits, fixed, final audits, EVAL6 run (3/6), deep weakness analysis, 3 GPT weakness reviews, a fix list, rev 5 built (B1-B9), 3 GPT rev 5 audits, R5-1..R5-5 fixes, EVAL6R5 (xhigh, hosen42) and EVAL12R5_LOW (low, Foysal) launched, discussion and leaderboard research, and both runs completed. The analysis was cut off.

---

## 1. User instructions, rules, preferences and decisions

| When | Verbatim (Bengali) | English meaning / effect |
|---|---|---|
| 10-20 02:21 | "গিটহাবে পুশ করো যা আছে" | Push everything we have to GitHub. Result: private repo Foysal348/ARC-AGI-2, commit 1fd6d8e. |
| 10-20 02:57 | "E13 সরল পথে, মানে E12 SMOKE1 notebook-এ শুধু engine বদলে, নতুন করে বানাব কি? ওকে" | Approves rebuilding E13 on the simple path: the E12 SMOKE1 notebook with only the engine changed. |
| 10-20 03:39 | "একসাথে কতগুলো টাস্ক চলে এই নোটবুকে?" | How many tasks run at once? (A concurrency question.) |
| 10-20 04:41 | "ভ্যালিডেট করাই জিপিটি কে দিয়ে,জিপ করো" | I will have GPT validate it; make a zip. This is a standing pattern: an audit zip comes before any run. |
| 10-20 11:13 | "Qwen 3.8 27Bএর কাজেব্যাক করতে চাই,আমাদের এই টার কাজ যেখানে থামিয়েছিলাম।সেখান থেকেই শুরু করো" | Go back to Qwen3.8-27B work and resume exactly where we stopped. |
| 10-20 11:45 | "আমার সুপারিশ (ক)। কম খরচে বোঝা যাবে…" | Chooses option (a): a cheap EVAL6 accuracy test. |
| 10-20 12:01 | "জিপ করে দিও,জিপিটি কে দেখাই" | Make a zip; I'll show it to GPT. |
| 10-20 13:21 | "আরো একবার সমস্ত তিন জিপিটির টেক্স পরে আসো,তোমার কথা ভ্যালিডেট করো।কিছুই বাদ দিও না।আর আমাদের মিনিমাল পরিবর্তন করার কথা মাথায় রাখিও" | Re-read all three GPT texts and validate your own claims. Leave nothing out. Keep changes minimal. |
| 10-20 13:44 | "দুইটা প্রতিযোগিতা একদম আলাদা।…এমন কঠোর হয়ো না যে আমাদের এক্সপেরিমেন্ট ভ্যালুহীন হয়,আবার এমনও নয় যে তুমি অতিরিক্ত পরিবর্তন করে বারোটা বাজিয়ে দাও।যা যা পরিবর্তন করতে হবে এবার করিও" | ARC-AGI-2 is not ARC-AGI-3, so make the changes this competition needs. Don't be so strict that the experiment becomes worthless, and don't over-change and break it. Now make the needed changes. (A standing rule.) |
| 10-20 13:58 | "নোটবুক টা দিও,লাস্ট অডিট করে নিই,কি নিয়েছো আর কি নাও নি এমন রিডমি ও দিও" | Give me the notebook for a last audit, plus a README of what you took and what you didn't. |
| 10-20 14:17 | "হামিদ ভাইয়ের একাউন্টে পাঠাও" | Send it to Hamid's account (explicit go for the EVAL6 run). |
| 10-20 15:43 | "আপডেট" | Status update request. |
| 10-20 16:22 | "আমাদের দূর্বলতা কি / আমাদের এজেন্টের দূর্বলতা বের করো ডিপলি" | Find our agent's weaknesses, deeply. |
| 10-20 17:24 | "তিনটা জিপিটির পরামর্শ দিবো,সব দেখবে আর ভ্যালিডেট করে পরের ফিক্স লিস্টে রাখবে।তুমি সব না দেখে শর্ট করে দ্যাখো সবসময়,এটা কখনোই করবে না,মেমরিতে সেভ করে রাখ" | I'll give three GPT reports. Read all of them, validate, and put them in the next fix list. "You always shorten instead of reading everything; NEVER do this. Save it to memory." This is a standing rule, saved as `read-everything-fully.md`. |
| 10-20 17:24 | "আগেই কিছু পরিবর্তনের দরকার নেই।ফিক্স লিস্টে যোগ করতে থাকো" | No changes yet; keep adding to the fix list. |
| 10-20 17:50 | "আমরা একটা রানে যত সমস্যা পাবো,সবগুলো ফিক্স করে,আবার রান দিবো,ছোট পরিবর্তন করে করে রান দেওয়ার টাইম নেই + জিপিইউ নেই এত বেশিও।তোমার এনালাইসিস ভ্যালিডেট করো ডিপলি,আর কিছু পাদ দিছো কিনা আরো একবার চেক করো" | Fix every problem found in one run, then run again. There's no time or GPU for small one-change runs. Validate deeply and check again for omissions. This is the bundle rule, and it overrides the GPTs' advice to isolate changes. |
| 10-20 17:50 | (on F3's 24-40k output being under the 60k rule) "এখন যে পরিবর্তন আমরা ইভিডেন্স পাবো সেটা করবো,এটাও প্রয়োজনে।" | Make changes that have evidence, including this one if needed. Conditional permission to go below 60k output per request when evidence supports it. Used for B5, 49,152. |
| 10-20 18:09 | "(a) temperature 0.7 থেকে 1.0 … এটা আমি নিলাম। প্ল্যান আপডেট করো।আমি জিপিটিকে দেখাবো।আর সবগুলো সমস্যা সমাধান করে আপডেটেড নোটবুক দিও" | Takes temperature 1.0. Update the plan (for GPT), fix all the problems, and deliver the updated notebook. |
| 10-20 19:22 | "ওকে নিয়ে,লাস্ট আবার validate kore dio" | OK, take R5-1..R5-5, then validate once more at the end. |
| 10-20 19:45 | "কয়টা এটাপ্ট কাজে লাগে? আর সাবমিশন তৈরি হয় কি ? সব ঠিক থাকলে জমা দেওয়া যাবে কি না?" | How many attempts are used? Is a submission produced? Can it be submitted if all is fine? |
| 10-20 19:49 | "চালাও,কমিট করো।" | Run it; commit. Explicit go for the EVAL6R5 commit run on hosen42. |
| 10-20 19:52 | "এজেন্টের তো আরো কম সময় লাগার কথা,বেশি কেন লাগছে? আপডেট করার পর কি কম লাগবে?" | The agent should take less time; why more? Will the update reduce it? |
| 10-20 19:56 | "একই নোটবুক,শুধু ~Qwen xhigh agent পরিবর্তন করবো" | Same notebook, change only the xhigh effort. Via AskUserQuestion the user chose "low (Recommended)". |
| 10-20 20:11 | "…প্রম্পট ইঞ্জিন্যারিং করে আমরা আরো কম টোকেনে কাজ করতে পারতাম কি না? … আমাদের টোকেন ইউজ কমাতে হবে,আর কভারেজ বাড়াতে হবে" | Could prompt engineering cut tokens? Review past work. We must reduce token use and increase coverage. |
| 10-20 20:17 | "বোকাচোদার বাচ্চা।এই মডেল ও এই ডাক দিয়েই ৮০+ এভালুশনে আর ৭০+ এলবি পাওয়ার প্রমাণ আছে।তারা পুরো কভারেজ দিয়েছে।আমরাও পারবো ইংশা আল্লাহ।আবার ডিস্কাশন পড়ে যায়" | (Insult.) There is proof of 80+ eval and 70+ LB with this model and this Duck, at full coverage. We can do it too. Go read the discussions again. This rejects the assistant's "don't run Duck on all tasks / Duck as complement" proposal. |
| 10-20 20:35 | "…এর লো নোটবুকে সবচেয়ে বেশি তথ্য বহুল আরো ৬ টা টাস্ক যুক্ত করে ফয়সাল আইডি তে কমিট করো" | Add the 6 most informative tasks to the LOW notebook and commit it on the Foysal account. This was an explicit go, given even though the assistant had advised against LOW at 20:12. |
| 10-20 20:52 | "বোকাচোদার বাচ্চা,আমি ওদের এলবি ও দেখেছি।তুই যেখানে দেখতে পারিস না,সেখানে তর্কে যাস কেন,বিশ্বাস করে নে।মেমরিতে সেভ করে রাখ।" + "ওরা সবাই Qwen3.8 + Optimized Duck Use করছে।" | (Insult.) I've seen their LB. Where you can't see, why argue? Believe it and save it to memory. They all use Qwen3.8 + an optimized Duck. This is a standing rule: trust the user's observations (`trust-user-observations.md`). |
| 10-20 20:56 | "আমরা যে vLLM to SGLang এ শিফট হলাম,এতে আমরা কি পেলাম,সেটা কি এই ডাকে এপ্লাই হয়েছে?" | What did the vLLM to SGLang move gain, and is it applied in this Duck? |

Standing constraints restated in the compaction summary (10-20 12:59): "অনুমতি ছাড়া কমিট করবি না" (no commit without permission); "আগে ভ্যালিডেট করবে,অনুমতি ছাড়া কোডে হাত দিও না" (validate first; don't touch code without permission); "অডিট লাগলে জিপ করিও আগে পুশ করার" (if an audit is needed, zip before pushing). Other constraints from the same summary:
- never read or echo token contents (the Hamid token is read in-process from `C:/Users/hp/hamid vai/kaggle token.txt`);
- use D: for scratch;
- chat in Bengali;
- 60k thinking floor;
- bundle changes;
- pre-run capacity check;
- minimal change, without drifting from the purpose.

Earlier complaints quoted in the summary:
- "(গ) দুটোই চলবে, এভাবে বার বার রান দেওয়া এক্সপেন্সিভ।আমাদের ৮ ঘন্টা নষ্ট হলো।তুমি সমস্যা যুক্ত কোড কেন দাও?" (8 GPU-h wasted on EVAL30; "why do you give buggy code?")
- "রিভিউ পড়তে পড়তে তুই আমাদের উদ্দেশ্য থেকে বাইরে চলে যাচ্ছিস" (you drift from the purpose while reading reviews).

---

## 2. Measured facts and results

**GitHub (10-20 02:50-02:53)**
- 5,113 files staged, commit `1fd6d8e`, private repo https://github.com/Foysal348/ARC-AGI-2.
- `_scratch` (6 GB, 63,074 files) was excluded, along with all .whl/.safetensors/.bin/.pt files. No file is over 30 MB, and no tokens were found.
- The E13 scripts and harness live in `_scratch`, so they are NOT in the repo.

**E13 Flash-Next simple SMOKE1 (10-20 03:13)**
- Notebook `Q27-XHIGH-E13_DUCK_FLASHNEXT_SMOKE1.ipynb`, sha `9a8d17ca…`. Cells [1,3,4,5,7,8] of 9 are unchanged from the E12 SMOKE1.
- Flash-Next tokenizer sha `06b95093` = RedHat INT4 = Franzen TOKENIZER_SHA. EOS [248046, 248044] is the same as the 27B. The FP8 tokenizer sha differs (`0997f410`).
- Changes from Franzen: NEXTN off, chunk 2048, mem 0.93, `--max-running-requests 4`, context 139,264, SMOKE1 time 50 to 90 min (Flash-Next load ~25 min).
- Harness: smoke 24/24, garbage 7/7, launchfail 7/7.
- Intel config: `quantization_config` at top level, auto-round.
- Audit zip `E13_SIMPLE_SMOKE1_AUDIT_FOR_GPT_2026-10-20.zip` (27 files, 0.53 MB). Not run on GPU.

**Quota**
| When | Account | Used |
|---|---|---|
| 03:13 | hosen42 | 20 to 27 h (another run happened in between); "33 h left" |
| 11:56 / 13:50 | hosen42 | 27.0 / 27.13 of 60 |
| 18:4x | hosen42 | 30.76 of 60 |
| 19:30 | hosen42 | 30.79 of 60 |
| 20:05 | hosen42 | 31.3 of 60 |
| 20:43 | Foysal | 17.77 of 30 |

L4×4 quota burns about 2.0× wall time; 125 min of wall time is about 4.2 h of quota.

**EVAL30 log and simulator (10-20 11:13-11:16)**
- 2,944 prefill lines; 11,232,209 new tokens; 15,348,388 cached (prefill lines).
- Median input throughput for chunks ≥2000 is **1,459 tok/s** (p25/p75 1,342/1,551). Log window 14,054 s; ~13,372 prefill-seconds.
- Simulator: 30 branches = 4 correct, 3 wrong, 23 unsolved. Generated tokens at acceptance: correct [9,482, 35,749, 38,777, 67,477]; wrong [20,680, 57,100, 70,448].
- Measured: 9.59 M prefill + 2.35 M generated = 12,018 s model time vs 12,634 s real. With no thrash: 0.39 M prefill, 5,706 s (2.1× faster).
- **29/30 first requests hit the 12,288 cap.** Unsolved branches were ~31 % at-cap vs ~15 % for solved ones.
- p90 context fits ~9 concurrent, p50 ~18, but EVAL30 had 30 active.
- Summary facts:
  - decode 431 tok/s at 12 running, 368 at 6, ~360 at 5;
  - KV 866,744 tokens;
  - prompts p50 47.9k / p90 97.7k (n=454);
  - 144/423 later requests lost their prefix;
  - branch 1 never started in 29/30 tasks;
  - Duck 4/30 tasks vs Poetiq R5 ~12 (16/48 inputs).
- EVAL30 request timeouts (corrected at 18:05): **40**, not 4. They hit 21 of 31 branches; 28 were full 900 s and 12 were deadline-bounded, 31,247 s waited in total.

**Duck code facts (verified 12:54-13:24)**
- Non-streaming (`openai_compat.py:52`, `stream=False`). Read timeout = min(analyzer_timeout 900, remaining).
- A ReadTimeout is not retried (`tool_agent.py:628`); Duck yields (`:7106`) and the in-flight reply is lost.
- Context budget = window − max_output − 512. At 131,072 / 60,000 that is 70,560; drain = min(58 Ki, budget/2) = 35,280. At 49,152 output the budget is 81,408.
- The per-branch cap is soft: it is checked after each request, so a branch can reach ~160k.
- The yield at 2,048 tokens is checked only between requests (`tool_agent.py:6718-6731`).
- `ARC3_HTTP_RETRY_INITIAL_SECONDS` 900 is a cold-start connect grace, not a read timeout.
- Summaries are off (`ARC3_SUMMARY_INTERVAL_TOKENS` unset).
- Degeneracy detector: zlib ratio threshold ~0.01. Real long thinking ran at 0.037-0.19, so it never fired.

**EVAL6 rev 4 run (hosen42, 10-20 14:18-16:09 UTC, sha `53a66856…`)**
- First push: ConnectionResetError, no kernel created. The second push succeeded. COMPLETE after ~110 min.
- run_id 20261020T145151Z; started_sec 811; deadline_in_sec 5400; branches_done 12; accepted 4; finished_sec 4555.
- **Inputs 4/8, task-norm 50 %, fully solved 3/6** (135a2760, 136b0064, 4c3d4a41). b5ca7ac4 was train 3/3 but test wrong.
- Solve took 62 min of the 90 min window. 0 request timeouts, 10 context trims, prefix-cache hit 78.8 %, ~10 GPU-min per task.
- Branch 0: 0/6 accepted; branch 1: 4/6 (identical prompts; the assistant called it possibly chance).
- 1,217,904 generated tokens over 96 requests.
- Completions by finish reason:

  | finish | n | p50 | p75 | p90 | p95 | max |
  |---|---|---|---|---|---|---|
  | tool_calls | 87 | 2,273 | 12,888 | 31,645 | 35,608 | 58,224 |
  | length | 7 | 60,000 | | | | |
  | stop | 2 | 13,109 | | | | |

- The longest reply in a successful branch was 44,779, which is the basis for B5 = 49,152.
- Prompt p50 38,745 / p90 63,699 / max 69,650. Context p50 48,515 / p90 69,440 / max 128,399. Uncached prefill 0.78 M vs generated 1.22 M.
- Output download: connection errors at first, then per-file retries with pagination worked (55 files).

**Weakness analysis (10-20 16:38-18:05)**
- Previous reply NOT carried into the next prompt: EVAL6 11/84 later requests (426,781 of 1,217,904 tokens = 35 %); EVAL30 12/423 (98,222 of 2,357,584 = 4 %).
- Example: 62593bfd_b0 produced a 60,000-token reply, and the next prompt was 9,395 (system only).
- Tokenizer proof (18:05): 5 cases (~279k tokens) were avoidable (real 65-69k fits, Duck estimated 77-111k); 6 were real overflows with the 60k reserve (with 48k output, 4 would fit).
- Repetition: EVAL6 29.4 % of thinking chars vs EVAL30 5.2 %. Loops start in the first third of a reply, and then the rest is ~100 % repeated (e.g. 136b0064_b0 reply 0: [41,100,99,99,100,98] % by sixth).
- No `check()` in 22/24 EVAL30 failed branches and 5/8 EVAL6 failed branches.
- 4/11 accepted programs were test-wrong. attempt_2 was the test-input copy in 56/56 slots.
- Real-sandbox replay with harvest + two attempts: EVAL6 → **5/6 tasks, 7/8 inputs**; EVAL30 unchanged (4 tasks, 7/48).
- Host verification takes **0.75-2.71 s** each, at most 3 per branch.
- Same-name redefinition after a check (N1): 0 of 405 tool calls.
- Length-cut tool calls in EVAL30: 5, all incomplete (3 half-written, 2 syntax errors). EVAL6: 0.
- Error types in EVAL6 tool results: NameError 4, ValueError 4, ImportError 1, IndexError 1.
- The `for i,(a,b) in enumerate(train)` dict-unpacking bug hit 3 of 4 tasks (EVAL30 reader).
- **SGLang 0.5.21: penalty token accumulation (`cumulate_penalty_output_tokens`) runs only in the EAGLE decode path; DFlash does not accumulate, so presence/frequency/repetition penalties have no effect under DFlash2.** This matches issue #28180. A 2-line patch would be needed.
- EVAL6 SGLang server log: `ModuleNotFoundError: No module named 'wrapt'` ×7 and "Error in sitecustomize; set PYTHONVERBOSE for traceback" ×7; `tl.make_block_ptr is deprecated` ×4; one `/abort_request` POST.
- Qwen3.8-27B HF card, thinking mode: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0. Instruct mode: 0.7 / 0.80 / 20 / presence 1.5. Duck ran at 0.7; the old DIRECT pipeline ran at 0.6.

**Rev 5 build (10-20 18:47-19:39)**
- EVAL6R5 `e1887ac1…`, then final **`f4fed1ac…`**. EVAL30R5 `70aff019…`, then final **`415a68f2…`**. LOW: EVAL6R5_LOW `cce9cb8f…`, EVAL12R5_LOW **`26fff41e…`**.
- Unit checks 53/53 rev 5 (56 with LOW), 41/41 flag-off. Harness: r5 32/32 (33/33 LOW), rerun 21/21, launchfail 6/6, old EVAL6 25/25, SMOKE1 22/22.
- Tokenizer check in the EVAL6 Kaggle run: ok (same vocab/merges/probe/EOS, 290 probe tokens), model path `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1`.
- Resume rewrite after a trim adds +216 chars (~50-110 tokens).

**vLLM vs SGLang (E11 numbers quoted at 20:57)**
| Measure | vLLM FP8 + MTP-3 | SGLang INT4 + DFlash2 |
|---|---|---|
| Solver-average speed | 255-281 tok/s | 329-380 tok/s |
| Decode, <10k context, 11 running | — | 429 |
| Decode, 40-60k context | — | 323 |
| DFlash2 accepted tokens per step | — | 4.5-5.5 |
| KV | — | 881k |

- The real-run gain was +17-29 % (1.4-1.6× was hoped for).
- E11-05: engine ready in 532 s / 522 s; KV 881,336, mamba cache 56, verify graph up to 12, 2.22 GB free.
- LB: E11-06 SGLang 29.17 vs vLLM S1-60K 34.58.
- Duck uses 6 streams (368 tok/s); the EVAL6 average was 325 tok/s (1.22 M / 3,744 s). 12 streams need ≤ ~72k context per branch (867k / 12).

**Leaderboard monitor / discussions (20:20-20:28, 20:52)**

| Team | Best LB | Runtime at best | Usual runtime |
|---|---|---|---|
| Tufa Labs | 88.06 | — | median 726 min |
| Yi-Chia Chen | 84.72 | — | median 683 min |
| rabbithole | 80.83 | — | median 729 min |
| nvbanana | 79.17 | 634 min | 565 min |
| Sakayori Lab | 68.06 | 710 min | 672 min |
| mikelou1 | 66.39 | 708 min | 649 min |

- Franzen (the Duck author): ARC-AGI-2 LB 57.64. NguyenThanhNhan 52.08 (pure reasoning ~23). Hiểu Vy 80 % on 120 eval, LB 44.
- Kris Dreemurr: xhigh median 70k tokens per task pass@2 (~35k/slot), ~45 % (52/120) CV.
- mikelou1: 79 % ± 5 % CV, 87 % under 4×L4-speed time limits. TheDivine Aurora: 300-500 tps gave ~19-20 %.
- Full coverage needs ~55-72k tokens per task at our speed; rev 4 Duck used ~200k (~3×).
- kagglesdk `get_topic` / `list_comments` work; the competition `list_topics` returns 403.

**W3H low-effort evidence (quoted at 20:12)**
- EVAL120 low effort: 17/172; all 15 solved tasks were already xhigh solves.
- 78 % of requests stopped mid-thinking. Requests were 3.4× faster, but train-pass was 13 % vs 46 %.
- Other cited evidence: S49 49k LB 32.08 vs 60k 34.58; ARC Prize medium 13.2 % < low 22.8 %; early-stop AUC 0.63 (saves 3-4 %, loses 18 % of train-passes).

**Rev 5 runs (completed 10-20 ~23:26)**
- **EVAL6R5** (hosen42, xhigh, sha f4fed1ac): started 19:50 UTC; started_sec 702; finished_sec 4166; branches_done 12; **accepted 9**; **inputs 5/8 (pass1 2, pass2 3), fully solved 4/6, task-norm 66.67**; 127 requests.
  - Prompt p50 51,917 / p90 71,504 / max 78,158; context p50 58,121 / p90 74,194 / max 95,121.
  - **Uncached prefill 0.97 M vs generated 0.85 M.**
- **EVAL12R5_LOW** (Foysal, low): COMPLETE after 163 min; 103 files.
  - 449 requests; prompt p50 41,629 / p90 72,552 / max 81,536; context p50 45,708 / p90 75,190 / max 114,787.
  - Uncached prefill "2.03…": the line is cut, so the value and the generated count are unknown.
- Per-task hits (line 7070, partly cut):
  - 136b0064 hit in all three runs;
  - 898e7135 hit only in R5 low;
  - 135a2760 hit in all three;
  - b5ca7ac4 hit 2 in R5 xhigh, none in R5 low;
  - the rest is cut.
- The per-branch table (line 7064) is cut, so the token, trim, timeout and loop values per branch are not in this part.

---

## 3. Failures, mistakes and root causes

| When | What | Root cause / fix |
|---|---|---|
| 10-20 11:56 (admitted) | **The builder overwrote the Kaggle-proven E12 SMOKE1 and EVAL30 notebooks** | Rebuild used the same names. Restored from the audit-zip copies (git show gave different bytes because of autocrlf), with shas 0fea4d0c / b65fb9dc confirmed. Guard added: non-EVAL6 builds are written as `_REV4`, later `_REBUILD`. |
| 10-20 11:48 | `os` not imported in `arc2_duck.py` | Would have been a NameError on Kaggle; added. |
| 10-20 03:0x | Null byte in a generated harness; missing transformers/PIL in e13_venv | Harness generation by script edits. |
| 10-20 11:5x | CRLF made a harness anchor non-unique | The anchor was made specific. |
| 10-20 12:52 (GPT audit 1, NO-GO) | 900 s non-stream timeout vs a 60k reply (6×60,000/368 ≈ 978 s) | Duck default; fixed to 1,800 s, EVAL6-only env `E12_DUCK_REQUEST_TIMEOUT`. |
| 10-20 12:54 → 13:01 | The assistant raised the Duck window to 154,624 with 5 live, then reverted it | No audit asked for it; it gave no gain with 60k replies, cost a live branch, and left ~1k margin. |
| 10-20 13:24 | Self-corrections | The "only ~1k margin → server rejects" fear was overstated (Duck re-trims and resends, `tool_agent.py:6802-6823`). "Watchdog also 1800" was unverified and dropped as a reason. |
| 10-20 14:02 | Stale notebook header text ("window 154,624, 5 live") | Code was right but the text was stale; rebuilt to sha 53a66856. |
| 10-20 14:0x | `rm -rf` blocked by the safety check | Rebuilt the bundle in a new folder. |
| 10-20 16:38 (admitted) | The audit doc said "the newest block is always kept" | Wrong for tool-less replies: Duck drops the opener and then everything up to the next user message, so the reply goes too. |
| 10-20 17:32 (GPT 1 caught it) | README W4: "no failed branch ever got a train pair exact" | Wrong; 62593bfd_b1 2/2, b5ca7ac4_b0 3/3, EVAL30 135a2760_b0 1/2. The assistant's own replay showed this. |
| 10-20 18:05 | EVAL30 timeout count reported as 4 | It was 40. |
| 10-20 17:47 | Claimed "Qwen recommends 0.6" | Wrong; the HF card says 1.0 for thinking. |
| 10-20 18:5x | Fake-server code broken by a newline inside a string literal (r5 scenario) | Fixed. |
| 10-20 18:5x | Harness found that Duck's own "cut off" message never reaches the model | Duck yields after each request (>2,048 tokens). Fixed by putting the cut note in the resume message. |
| 10-20 19:45 | **Textual contradiction: "LB ~7-11 … much better than our best Poetiq (34.58)"** | 7-11 is far worse. A wording error in the assistant's reply (line 6129). |
| 10-20 20:12 (admitted) | Recommended `low` effort without citing our own W3H evidence (17/172, 78 % mid-thinking stops) | Incomplete evidence review. |
| 10-20 20:17 (user angry) | Proposed "don't run Duck on all tasks / Duck as Poetiq complement / coverage beats depth" | Violated the "never call it impossible, focus on HOW" rule. |
| 10-20 20:52 (user angry) | Argued about other teams' methods/LB that it cannot see | Violated trust in user observations; saved as a memory rule. |
| 10-20 20:2x | Created a stray empty file `C:/Users/hp/.claude/memory_placeholder` | A heredoc memory append timed out; the file was removed. |
| 10-20 18:0x | Bash heredoc EOF error while editing the fix list | Switched to the Edit tool. |
| 10-20 13:4x | `kaggle` module not found in the push script's python | Ran with user site-packages. |
| Earlier (summary) | E13 probe issues | CRLF changed the notebook sha (the builder now writes bytes). The FA2 wheel ships flash_attn/cute files, so FA4 is reinstalled after it. A static assert would have blocked the P1 fallback. The graph-log regex didn't match. A late OOM went undetected. The pinned-memory test itself risked OOM. |
| Earlier (summary) | 8 GPU-h wasted on E12 EVAL30 | KV thrash (30 live contexts), branch 1 starvation. This produced the pre-run capacity-check rule. |
| Analysis caveat (GPT 1/2) | `replay_all.py` used host `exec` | Not faithful to the sandbox. Re-done in the real sandbox at 18:05 (same 5/6). |
| Statistic caveats (GPT 2) | "median 100 % tokens before first check" is censored (most branches never checked); "29 %" is characters, not tokens | Accepted. |

---

## 4. Decisions and reasons; options rejected

- **E13 simple path** (02:57): reuse the E12 SMOKE1, change only the engine cell. Built; not run; awaiting GPT audit.
- **Option (a) EVAL6** (11:45) over (b) a full rev 4 EVAL30 (~8 h quota) or (c) returning to Poetiq: the cheapest test of whether Duck-27B accuracy can rise.
- **Rev 4 EVAL6 bundle**: output 12,288 → 60,000; ≤6 live branches (KV-fit); both branches of a task start together (task-major); soft 100k per-branch cap.
- **Timeout 1,800 s** (not the GPTs' 1,500): margin for slower long-context decode; bounded by min(timeout, remaining).
- **Soft cap kept; GPT-B's hard clamp (second request 40k) rejected**: it re-truncates reasoning, which is the thing the test measures.
- **GPT-B's drain of 11,168 rejected**: a small drain trims almost every turn and invalidates the prefix cache; re-prefill was EVAL30's main loss (9.6 M).
- **Solve window 60 → 90 min, total 100 → 125 min**: 12 × ~120k = 1.44 M tokens = ~65 min decode + ~14 min prefill; at 60 min a 0-1 result would have been ambiguous.
- **Serializer spoof (json.dumps)**:
  - at the rev 4 final, NOT fixed. Reasons: the score uses ground truth; no gain over a lookup table; 0 uses in the transcripts; the proposed guard is incomplete (json.encoder); a fix would change the sha and restart the audit cycle;
  - at rev 5, FIXED as R5-1. GPT 2 reproduced it via `CHECKED:` and submit. The guard keeps correct functions that use json.dumps, and harvesting raised the stakes.
- **Harvest via `check()`-exact (GPT 1/3), not passive retained-`transform*` (GPT 2)**: passive harvesting found nothing new and lost 2 solutions (functions with other names).
- **No auto-stop on the first exact check**: in EVAL30 247ef758_b0 the first exact candidate was test-wrong and a later one was correct.
- **Temperature 0.7 → 1.0 (B9a) taken; presence penalty (B9b) rejected**: it needs an SGLang DFlash patch, and HF warns about language mixing.
- **B5 output 49,152** (below the 60k floor, under the user's evidence permission): the longest productive reply was 44,779; 40k would cut one successful reply; none of the 7 replies cut at 60k had a tool call.
- **Rev 5 bundle B1-B9**: B1 harvest + host re-verify; B2 two distinct attempts, latest per branch first; B3 real-tokenizer reasoning count; B4 truthful resume; B5 49,152; B6 cut tool calls not executed; B7/B8 prompt edits (code-first, lookup tables OK, train usage example); B9 temperature 1.0.
- **R5-1..R5-5 taken**: serializer guard; markup-recovery bypass closed for length-cut replies; resume rewrite on the `_force_reduce_messages` path; tokenizer status reported (not asserted, so the solver isn't killed); per-request estimate vs server usage recorded.
- **R5-6 rejected** (distinct function names per version): 0 occurrences in 405 calls.
- **Tokenizer: report, not assert** (GPT 2 wanted an assert): the run still works with the fallback.
- **Bundle rule kept over the GPTs' isolate-changes advice** (user, 17:50).
- **LOW variant**: built at the user's request (the AskUserQuestion answer was low). At 20:12 the assistant then recommended NOT running it (W3H evidence). The user ran EVAL12R5_LOW anyway (12 tasks: the EVAL6 six + 247ef758, bf45cf4b, f931b4a8, 65b59efc, 9385bd28, cbebaa4b).
- **Rejected as a direction (user)**: "run Duck only on Poetiq-weak tasks / coverage via Poetiq + Duck". Memory now says never propose "don't run Duck on all tasks".

---

## 5. Open items

1. **The rev 5 results were never reported to the user in this part.** The EVAL6R5 headline (5/8 inputs, 4/6 tasks) and the EVAL12R5_LOW numbers were computed, but the per-branch tables are cut, the analysis doc was written (`E12 rev 5 runs analysis (2026-10-21).md`), and no user message follows.
2. **E13 Flash-Next SMOKE1**: the GPT audit results on `E13_SIMPLE_SMOKE1_AUDIT_FOR_GPT_2026-10-20.zip` are still pending; no go yet.
3. **S1-55K resubmission**: the commit was cancelled (CANCEL_ACKNOWLEDGED); no decision.
4. **E13 scripts and harness are only in `_scratch`** (not in the GitHub repo). Moving them into the experiment folder was offered (02:53) and never done.
5. **EVAL30R5** (sha 415a68f2) was built and checked but not run. GPT advice: decide after EVAL6R5.
6. **Coverage (F7)**: target ~35k tokens per attempt / 55-72k per task, and more tok/s. 12 streams need ≤72k context per branch. No plan was decided.
7. **SGLang abort on client disconnect / timeout**: "was not verifiable locally" (12:54, 13:24). Still unknown whether the server keeps generating after a Duck read timeout.
8. **DFlash penalty patch** (2 lines) is not applied; penalties are a silent no-op.
9. **B3 is not exact chat-template accounting** (GPT 3). R5-5 records estimate vs server `usage.prompt_tokens`; nobody had checked that in the run outputs by the end of this part.
10. **B4 edge**: resume text added after the trim budget (+216 chars). A near-budget synthetic test was suggested (GPT 2) and not done.
11. **N1 same-name redefinition loss** is accepted as a risk; R1, the verification cost cap of 24, is to be monitored via `checks_verified` and `checks_accepted`.
12. **Paper numbers** (144/167 vs 128, Table 4) are unverified; the PDFs were not available to the assistant.
13. **Ambiguity "কমিট করো"**: the assistant read it as a Kaggle commit run and offered a separate git commit if that was meant. No answer in this part.
14. **Branch 0 vs branch 1 asymmetry** (0/6 vs 4/6 accepted in rev 4) was not explained.
15. **The "~2 h wait before the last request" in EVAL30 89565ca0** is unexplained by the log (subagent C).
16. **Rerun harness flake**: 19/21 once (a branch could not start within the fake 25 s deadline), then 21/21 three times. Labelled a timing flake, not root-caused.

---

## 6. Insights a fresh reader might miss

- **Rev 5 EVAL6R5 shows uncached prefill (0.97 M) larger than generated tokens (0.85 M)**, whereas rev 4 EVAL6 was 0.78 M vs 1.22 M. Keeping more reasoning (B3) made contexts bigger and prefill heavier. This is consistent with a later finding (in the E14 context) that Duck's history normalization broke prefix matching, so the previous reply was re-prefilled. The 78.8 % rev 4 cache hit is not a ceiling.
- **The "previous reply dropped" problem had two separate causes.** About half were estimator errors: grid-calibrated 1.40-1.72 chars/token made 126k chars of reasoning look like ~90k tokens. The other half were real overflows with the 60k reserve.
- **ARC-3 game words in transcripts were a logging artifact**: Duck logs the prompt before the adapter rewrites it. The model received ARC-2 text.
- **Penalties don't work at all under DFlash in 0.5.21**: not just the #28180 symptom; the accumulation is absent in the DFlash path.
- **No successful branch ever had a 60k reply.** Successes had a first reply of 21-45k, then many short tool requests; failures had repeated 20-60k replies with few tools.
- **Duck's yield check is between requests**, so its own "you were cut off" note never reached the model. Rev 5 moved it into the resume.
- **Early stop ≠ fewer tokens per solve.** Past evidence (low effort, 49k, 32k, early stop) consistently lost solves. Yet the user (backed by leaderboard data) insists full coverage at ~35k per attempt is achievable. That tension was unresolved at the end of the part.
- **Branch exposure matters for interpreting results**: in rev 4 EVAL6 every branch ran (8 stopped at the soft cap, none at the deadline), so 3/6 was a clean read.
- **The serializer-spoof finding had already been reproduced on 10-19 and left open**; it re-emerged and was fixed only when harvesting (`CHECKED:`) widened the path.
- **The user overrode the assistant's "don't run LOW" advice** within 25 minutes and ran a 12-task LOW on Foysal. R5 low then hit 898e7135 (not solved by xhigh) and missed b5ca7ac4 (solved by xhigh R5). Low is not simply worse per task.
- **Top teams use nearly the full 12 h** (565-729 min). Their high scores are not short runs, so the time budget is fully used.
- **Contradiction**: the 19:45 reply says LB ~7-11 is "much better" than 34.58. It is a typo-level error, but a fresh reader could take the wrong conclusion from it.
- **The 60k floor was explicitly relaxed for Duck output per request (49,152) with evidence** on 10-20 17:50. Memory still lists a 60k floor for THINK_BUDGET, admission and rescues. These are two different knobs, and any further cut needs approval.

---

## 7. Relevance to the current E14-A notebook (concrete checks / guards / defaults)

1. **sitecustomize failure in the SGLang processes (evidence: EVAL6 server log, 10-20 18:0x: "ModuleNotFoundError: No module named 'wrapt'" ×7 + "Error in sitecustomize" ×7).**
   - Risk: if E14 applies any patch through a sitecustomize or .pth hook (the #44401 backport, the `--min-free-slots-delay` / DFlash min-free-slots change, int8 mamba, NCCL env), a sitecustomize that errors fails silently in some or all TP worker processes.
   - Guard:
     - make each patch print a unique marker from every TP rank, and have the phase record assert that the expected markers appear N=4 times in the server log;
     - either ship `wrapt` in the wheelhouse or remove whatever sitecustomize imports it;
     - treat "Error in sitecustomize" as a phase failure (or at least a red flag in the report).
2. **Penalties are a no-op under DFlash in SGLang 0.5.21** (`cumulate_penalty_output_tokens` is EAGLE-only; verified in source 10-20 18:05).
   - No E14 arm, greedy parity or Duck run should set presence/frequency/repetition penalty and expect an effect.
   - If any request payload carries a penalty, the target-only vs DFlash parity can diverge for reasons unrelated to the arm under test. Assert penalty=default (presence 0, repetition 1.0) in the parity and replay requests.
3. **Tokenizer path after the /dev/shm copy.**
   - Rev 5+ B3 reads `E12_TOKENIZER_JSON = SGL_MODEL_PATH/"tokenizer.json"` and silently falls back to 1.9 chars/token if it is missing (GPT 2 audit, 19:16).
   - The E11 tokenizer check, on failure, switches `ENGINE_BACKEND = "vllm"` ("E11 tokenizer check FAILED -> S49 vLLM FP8 engine", line 212). The harness config shows `E11_ALLOW_VLLM_FALLBACK: true`.
   - With E14's ranged parallel copy to /dev/shm (fail-open), a partial copy or a path redirection could:
     - (a) make the Duck estimator fall back, which brings back the 60k-reply drops;
     - (b) flip the whole notebook to the vLLM engine, so every SGLang measurement phase would be invalid.
   - Guard:
     - include tokenizer.json / tokenizer_config / chat_template / generation_config in the copy verification;
     - after the copy decision, assert `tokenizer_status()=="loaded"` and record its path;
     - for E14 (an engine measurement notebook) disable or loudly fail the vLLM fallback instead of switching silently;
     - log which path (shm vs /kaggle/input) the engine, tokenizer and Duck each used.
4. **Duck request timeout.**
   - Duck is non-streaming; a read timeout is not retried and the in-flight reply is lost (`tool_agent.py:628/7106`). The default is 900 s, and only `E12_DUCK_REQUEST_TIMEOUT=1800` overrides it.
   - EVAL30 had 40 timeouts (31,247 s lost).
   - For the 25-minute Duck run, verify the rev 6g profile sets 1,800 (it is bounded by min(timeout, remaining) anyway). Also log `request_timeout (yielding` counts in the report.
5. **Orphan generation after a client timeout or abort is unverified** ("SGLang abort on client disconnect … not verifiable locally", 12:54/13:24).
   - This matters for E14's "11 of 12 running" diagnosis: if a client-side timeout leaves the server generating, `num_running` and KV are consumed by dead requests.
   - Check: in the latch test or after the Duck window, compare the client's in-flight count with the server's `num_running_reqs`, and record whether `/abort_request` reduces it.
6. **KV budget at 12 running with Duck-size contexts.**
   - Measured context p90 was 74,194 (EVAL6R5) and 75,190 (EVAL12R5_LOW), max 95k-115k.
   - 12 × 74k ≈ 890k > 866,744 measured KV (881,336 in E11-04/05). The rule is ≤ ~72k per branch for 12 streams.
   - The "baseline windows at 12 running" must state the context length used. If the Duck run is set to 12 streams on the adopted arm, expect eviction or retraction unless contexts stay under ~72k.
   - Report `#retracted`, `full token usage` and prefix-cache hit.
7. **Throughput reference numbers for E14 baselines (to judge any arm).**
   - decode 431 tok/s at 12 running, 368 at 6, ~360 at 5;
   - 429 tok/s with <10k context and 11 running, vs 323 at 40-60k context;
   - prefill median 1,459 tok/s for chunks ≥2,000 (p25/p75 1,342/1,551);
   - DFlash2 accept 4.5-5.5 tokens per step;
   - E11-05 engine ready 522-532 s; E12 solver start at 702-811 s.
   - Decode speed depends strongly on context length, so E14 windows and replays should bin by context (as the E11 analysis did), or the arms are not comparable.
8. **CUDA graphs 1..13 arm.** E11-05 shows "verify graph up to 12" while the server allows 13 running (`SGL_SERVER_MAX_RUNNING 13`, client 12). The 13th slot runs without a graph, which supports testing graphs up to 13. Record the capture list in the log so it isn't assumed.
9. **Prefix-cache and raw-reasoning fix: measure it.**
   - The rev 5 run shows uncached prefill (0.97 M) > generated (0.85 M), against 0.78 M vs 1.22 M in rev 4.
   - The E14 "Duck history keeps raw reasoning" fix should be validated in the 25-min Duck run by per-request `cached_tokens / prompt_tokens`. Expected: the uncached part ≈ the new turn only, and no re-prefill of the previous reply.
   - Also confirm the reasoning goes back under the key the SGLang Qwen template reads (adapter `ARC3_REASONING_HISTORY_KEY=reasoning_content`; the E12-V vLLM build sets `reasoning`). A wrong key silently drops reasoning (Duck's note: 69 vs 3,670 prompt tokens).
10. **Never rebuild a notebook that ran** (evidence: 10-20 11:56 overwrite of the SMOKE1/EVAL30 Kaggle-run notebooks).
    - The E14-A builder must write a new file name.
    - After every build, print and compare the shas of all Kaggle-run notebooks (as done here: 0fea4d0c, b65fb9dc, 53a66856, f4fed1ac, 26fff41e).
    - Write bytes, not text: CRLF changed the sha before.
11. **Push and download robustness.**
    - Both pushes had a ConnectionReset on the first try; in one case no kernel was created. Before re-pushing, confirm with `kernels.get` / status that no kernel exists (to avoid a double run on a 1-pending-per-team budget).
    - Output download needs paginated per-file retries (55 and 103 files listed; 0 failed only after retries).
12. **Quota estimate (written, before the GPU go).** L4×4 burns ~2.0× wall time (125 min wall = ~4.2 h quota). The E14A profile of 200 min means ~6.7 h of quota. The last known balances were hosen42 ~35/60 after EVAL6R5 and Foysal ~24/30 after EVAL12R5_LOW (estimated: 17.77 + ~6.2). State this in the run request.
13. **Sampling and the greedy-parity design.**
    - Duck rev 5+ runs at temperature 1.0 (B9a), with top_p 0.95 and top_k 20 per the HF card.
    - Greedy parity (A/A, target-only) must override to temperature 0 explicitly. Check that the Duck env, `E12_DUCK_TEMPERATURE`, does not leak into the parity/replay requests, and vice versa: the 25-min Duck run must use the rev 6g sampling, not the parity settings.
14. **Truncated tool calls and markup recovery (B6 + R5-2)** must be present in the rev 6g solver used by E14. Under length cuts at 49,152, Duck's raw `<tool_call>` markup recovery executed cut code in a reproduction (19:19). Confirm the rev 6g adapter still forces malformed handling on `finish_reason=length`.
15. **Tool-result trust boundary (R5-1)**: the serializer guard (`_grid_call_code` restores `json.dumps`) must be in the embedded adapter. If E14 re-embeds the adapter, hash-match it against the rev 6g source, as the audits did for 23/23 files.
16. **Stale config fields in the report.** The E12 run report still carries P1 fields (`MAX_MODEL_LEN 106496`, `PROMPT_TOKEN_LIMIT 93952`, `THINK_BUDGET 60000`, sampling temperature 0.6) that Duck doesn't use. E14's phase record should label which config block each phase actually used, so later readers don't misread an engine arm.
