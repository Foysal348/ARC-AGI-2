# Insights from part_10.txt (2026-10-22 20:08 to 2026-10-23 05:32)

Source: `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/convo/part_10.txt`, all 5,777 lines read in order.
Timestamps are the block headers as written. The headers run from "2026-10-22 20:08" to "2026-10-23 05:32", but the note files are dated 2026-10-23, so the headers are probably UTC, about 6 h behind the user's local time.

What happens in this part:
- 20:08–20:11: two more GPT notes are saved (Notes/03 and 04).
- 20:14: the user orders a deep re-read plus 5 more agents.
- 20:20–20:57: ten research agents report.
- 21:12: final report "৪xL4 ইঞ্জিন বটলনেক যাচাই.md" (476 lines).
- 02:23: user "ওকে", and the P0 CPU tasks start: offline token-diff and the E14-0 load-path module.
- 02:36: context compaction (the summary is inside this part).
- 03:18: two more GPT reviews (Notes/08, 09).
- 03:38–03:58: load path rev 2 and the re-prefill root cause.
- 04:25: Franzen question.
- 04:40–05:17: Duck raw-reasoning switch and fake-server prefix probe.
- 05:27–05:32: full E14-A plan (Notes/13).

---

## 1. User instructions, rules, preferences and decisions

| When | Verbatim (Bengali) | Meaning / effect |
|---|---|---|
| 10-22 20:08 / 20:11 | (pastes GPT notes 2 and 3, no text of his own) | Implicit rule already in memory: save every GPT paste verbatim in E14 Notes. The assistant saved them as Notes/03 and Notes/04. |
| 10-22 20:14 | "আবার তিনটা জিপিটি আর তোমার আগের ৫ টা এজেন্টের গবেষণা দেখে আসো,আর প্রয়োজনে আরো ৫ টা এজেন্ট চালিয়ে দিও।তুমি অনেক কিছুই মিস করো,স্কিপ করো,আবার দেখে আসো ডিপলি।পরে আরো ৫ টা এজেন্ট চালাও" | "Re-read the three GPT notes and your earlier 5 agents' research, run 5 more agents if needed. **You miss and skip a lot**; look again deeply. Then run 5 more agents." A complaint and a standing expectation: read everything deeply, never skip. |
| quoted in the 02:36 summary (said earlier) | "তুই রিপোর্টে অনেক কিছু স্কিপ করিস সর্বদা,এটা আর কখনোই করবি না।পূর্ণ report always আবার শুরু কর" | "You always skip a lot in reports; **never do that again. Full report always.** Start again." Standing rule (memory `read-everything-fully.md`). |
| summary (earlier) | "আর হা তুমি সব পড়ো না,এটা মেমরিতে রাখিও,তোমাকে সবই পড়তে হবে,পুর্ণ রিপোর্ট ই দিতে হবে।" | "You don't read everything; keep this in memory: you must read everything and give full reports." Standing rule. |
| summary (earlier) | "ডিপলি রিসার্স করো।আর একটা এক্সপেরিমেন্ট ফোল্ডার খুলো,আমরা Optimized Experiment এর দিকে যাবো। আর তোমার এজেন্ট গুলোর এনালাইসিস আর এইটা,আরো কিছু দিচ্ছি সব নোট করে রাখিও,ভুলে যেও না।" | Open the Experiment 14 folder; note every agent analysis and GPT paste; "don't forget". |
| summary (earlier) | "এক্সপেরিমেন্ট লিস্ট ও সাজাও প্রায়োরিটি অনুযায়ী।" | Keep the experiment list sorted by priority. |
| summary (earlier, with insults; the insults are left out here) | "তুই বললি ৬০ মিনিটের মত লাগবে তাই রান দিলাম।এরপর আমি ভাবছিলাম ,৩০ মিনিট থাকলে ১২ ঘন্টার করে ২ টা রান দিবো … এত বড় ভুল কিভাবে করলি? কোড আবার দেখ ,কই ভুল করছিস?" and "এই দিকে আর মাত্র ১১ দিন আছে।… আমায় হতাশ করে দিলি।" | The user ran G4B because of the "~60 min" estimate; it burned the esfersami50 quota, so his plan for two 12 h runs failed. "Only 11 days left." This is where the rule comes from: a written worst-case wall time, quota cost and stop rule before every GPU run. |
| summary (standing) | "অনুমতি ছাড়া কমিট করবি না" | Never git commit without permission. |
| 10-23 02:23 | "ওকে" | Approved the two CPU-only P0 tasks: the E14-0 load path (load cell only) and the offline token-diff. |
| 10-23 03:18 | "তোমার আইডিয়া বদলানো লাগলে বদলিও,শুধু জিপিটির পরামর্শ দেখে নোট আপডেট করো।হতে পারে নতুন কোনো তথ্য দিবে।" | "Change your idea if needed; just update the notes after reading GPT's advice; it may give new information." Scope at this point: **notes only, no code** (the assistant respected this at 03:28). |
| 10-23 03:18 (GPT reviews pasted by the user; not his own words, but he forwarded them as input) | "Decision: GO for E14-0 and E14-P0 CPU-only work. NO-GO for GPU execution until separate approval." | The CPU work was approved with mandatory constraints (section 4). |
| 10-23 03:37 | "আলাদা আলাদা মানুষ।একটা টিম।আর শুরু করো" | foysal, hosen42 and esfersami50 are **three different people on one team** (answers GPT's multi-account rule warning). "Start" = do the Notes/10 code fixes. Saved to memory `kaggle-arc2-platform-limits.md`. |
| 10-23 04:25 | "https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution?scriptVersionId=354090904 34.30 V1 এই নোটবুকের এই problem e Score kom hocche naki? মানে এই সমস্যায় কি এই প্রতিযোগিতায়ও স্কোর আটকে রাখতে পারে?" | "Is this notebook (34.30, V1) losing score to this same problem? Can this problem hold our score down in this competition too?" |
| 10-23 04:40 | "Duck-এর এই সংশোধন করব কি? / "আগের message বদলানো" সমস্যাটা fake server দিয়ে CPU-তে খুঁজব কি? / শুরু করো" | Approved both: (1) the Duck raw-reasoning fix (behind a switch) and (2) the CPU fake-server hunt for the "earlier message changed" misses. |
| 10-23 05:14 | "আবার শুরু করো" | "Start again" (after an API ENOTFOUND error at 05:08). |
| 10-23 05:27 | "E14-A এটার পুরো প্ল্যান দিও।আর পুরো পাইপ লাইন পরিবর্তন  করা লাগলেও করিও,বেস ভাঙবে এই সেই বলে আমার এক্সপেরিমেন্ট মাটি করিও না আবার।বেস সংরক্ষিত আছেই সমস্যা নেই।" | "Give the full E14-A plan. **Change the whole pipeline if needed; don't ruin my experiment again with 'it will break the base' excuses. The base is preserved, no problem.**" New standing rule, saved as memory `no-base-protection-excuses.md`. Effect: the proven fixes are default ON in E14. |

Standing constraints restated in the 02:36 summary (keep these):
- no push, run or submit without an explicit go;
- no commit without permission;
- never echo tokens;
- read everything and give full reports;
- scratch work on the D: drive;
- 60k thinking floor;
- pre-run capacity check;
- never rebuild a Kaggle-run notebook;
- rev 6g is the last version, so Duck code changes need approval;
- a written wall time (worst case, slow storage), quota at 2× and a stop point before any GPU run;
- chat in Bengali.

---

## 2. Measured facts and results

**Step time and acceptance (from logs; local validation agent, 20:41):**
- Step at 12 running:
  - E11-06 median 124.6 ms (p10–p90 116.2–132.5);
  - E11-05 128.4 ms;
  - E12 EVAL30 137.9 ms only with a gen > 100 tok/s filter (139.4 ms with all records, 138.4 ms with gen > 50).
- Below 10k context at 12 running, every run shows about 115 ms: E11-03 D 115.5 (n=67), E11-06 115.3, E12 115.4. It rises to about 133 ms at 50–60k.
- "Step time" = logging-window wall time / 40 forwards, prefill included (`metrics_reporter.py` L898-937, L501).
- Accept len means:
  - E11-06 at 8 running 4.4628 (n=408), at 12 4.3708 (n=278); E11-05 at 12 3.9130 (n=114).
  - Per-run means 3.87 / 4.58 / 5.10 (E11-05 / E11-06 / E12 EVAL30 rev1).
  - Total context at 12 running: 449k (E11-05) and 357k (E11-06).
- Accept len includes the bonus token; accept rate = (L−1)/7. It fits within 0.0057 across 5,017 records (example: accept len 4.07 with rate 0.44).
- DFlash agent: median accept 4.55 (E11-05, 11 running), 4.62 (E11-03 c12). The final-report agent noted the decode-line median is only 3.8.

**Memory per GPU, TP4 W1:**
- Weights: target 4.62 GB, drafter 1.49 GB.
- Mamba: conv 0.04, ssm 1.00, intermediate 1.97 GB (1.83 in E11-05), conv window 0.03.
- KV: target K/V 3.31+3.31 GB for 866,744 tokens; drafter K/V 1.03+1.03 GB.
- Free memory: 3.82 GB after the pools → 2.61 after the target graphs (0.68) → 2.26 after the draft graphs (0.35) → final 1.88 GB.
- The SGLang KV formula predicts 886,801 tokens vs 881,336 measured (W1).

**Startup timings:**

| Run | to "fired up" | weight load | CUDA graphs |
|---|---|---|---|
| E11-05 | 8:00 | 13 s | 383 s |
| E11-06 | 7:52 | 13 s | 382 s |
| G4B rev 6d | 19:09 | 670 s | 394 s |

- About 362 s of each capture is the first bs=12 bucket. The earlier figure for target-verify graph capture was 372.5 s.
- Relaunch in the same session: 11.9–13.7 s (E11-04).
- 10-21 load: 669.96 s total; the 18.6 GB shard took 657 s (~28 MB/s). On 10-19/10-20 the load took 12.8–58.7 s.
- G4B install 208.2 s.
- G4B profile_row: time_budget_sec 4800, give_up_after_sec 3780.

**NCCL:**
- The only measurement ever taken: 64 MiB, fp32, eager torch.distributed, 10 iterations.
  - E11-03: 20.89 / 20.89 / 20.85 / 20.86 ms. E11-04: 20.76–20.83 ms.
  - That is 3.21 GB/s algbw and 4.82 GB/s busbw.
- No smaller sizes were ever measured.
- "CustomAllReduceV2 is disabled…" and "CustomAllreduce is disabled…" appear on all 4 ranks in E11-05, E11-06, E12 EVAL30 and G4B. The rule is `world_size > 2 and not full_nvlink`.
- Collectives per step, counted from the source:
  - target verify: 129 all-reduces + 1 logits all-gather (11.9 MB per rank, 47.7 MB total);
  - draft: 11 all-reduces + 2 small all-gathers;
  - accept: 1–2 broadcasts;
  - in all 140 all-reduces of 983,040 B; the naive bandwidth model gives 45 ms.
- Locally measured BF16 GEMM: 75.9 / 78.8 TFLOPS.

**Agent effective throughput (agent report 20:39):**

| Run | prefill share | uncached tokens / generated token | effective tok/s | decode-only tok/s |
|---|---|---|---|---|
| EVAL30 | 55.1 % | 4.07 | 167.8 (187 over 12.6 ks) | ~376 |
| EVAL6 | 13.3 % | 0.64 | 325 | ~377 |
| EVAL6R5 | 18.6 % | 1.15 | 244 | ~302 |
| EVAL12R5_LOW | 20.4 % | 1.17 | 265 | ~335 |
| G4B rev 6d | 19.8 % | 1.17 | 279 | ~353 |

- Prefill runs at about 1,475–1,578 tok/s and blocks decode. One uncached prompt token costs about 0.27 of a generated token (Franzen's RTX PRO 6000: about 0.08).
- EVAL30:
  - below 12 running 67.8 % of the time;
  - "11 running + 1 queued" 50 % of the time (G4B 25 %), costing about 9 %;
  - KV p50/p90/max 0.77/0.95/1.0.
- G4B:
  - below 11 running 47 % of the time; exactly 6 running 27.7 % of the time (probe bug);
  - 14 zero hits + 15 partial hits = 317k tokens (47 % of uncached).
- Mamba usage at 12 running: 36 of 56 slots.
- Earlier figure (20:18): EVAL30 engine 412–431 tok/s vs effective ~187. 144 of 423 follow-up requests lost their prefix; 8.39 M tokens were re-prefilled.

**G4B cache accounting from server usage (03:52), 150 requests with a predecessor:**

| Class | requests | re-prefilled tokens |
|---|---|---|
| reply_lost_at_prefill_end | 68 (45 %) | 217,541 |
| reply_reused | 49 (33 %) | 3,446 |
| deeper_miss | 18 (12 %) | 66,771 |
| partial_reply | 14 (9 %) | 2,442 |
| history_shrank | 1 | 23,099 |

- Total re-prefilled 313,299 of 2,392,575 prompt tokens. This is "79 %" of uncached prompt tokens. (The rest of that result line is cut at "…[+254]"; the uncached total is not visible.)
- Reasoning chars median: reused 116 (41/49 ≤ 300) vs lost 1,712 (12/68 ≤ 300); deeper_miss 2,046.
- The reuse split by reply length: median 98 tokens for reused, 830 for lost.

**Token-diff and tokenizer:**
- First token-diff (02:32): 170 turns. After the script fix, A/B/C were 170/170.
  - This was later shown to be a **false pass** (section 6).
  - The first raw run gave C 0 / D_reason_ws_lost 170 / markup_equal 116 / D_param_ws_lost 54.
- Non-canonical tokenization check (03:26): `'\n\n'` canon [271] vs alt [198,198]; `'\n\n\n'` [1358] vs [198,198,198]; `' grid'` [5646] vs [220,70,81,72,67]. All give the same text, and re-encode(alt) == canon.

**Fake-server prefix probe (05:17):** real Duck, 2 tasks × 2 branches × 8 requests.
- Switch off: 24/24 follow-ups broke at the reply's first blank line (char 57–58).
- Switch on: 24/24 full match.
- "Earlier message changed": 0 in both arms.

**Tests:**
- Load path rev 1: 34 tests, 5/5 runs pass.
- Rev 2: 56 tests, 6/6 runs pass.
- Raw-reasoning switch: 15/15 pass.
- E12 smoke on the old (sha 7644aa06) and new (sha 758bddb5) arc2_duck gives identical lines:
  - smoke_adapter 41/0;
  - smoke_rev5 56/0;
  - smoke_rev6 66 PASS / 1 FAIL ("V4 show() over the limit refused"), plus a `solve_all_r6() got an unexpected keyword argument 'min_workers'` TypeError. Both already happen on the old code.
- Previous G4B fake harness: 46/46 ok in 144 s, cached 78.2 %.
- Module size: 893 lines (the assistant expected ~640; it checked for duplicates and found 0). Test file 874 lines. Diff file 1,255 lines.

**Quota and accounts:**
- esfersami50 30.21/30 h, exhausted. The quota went from 26.87 to 30.21 h for the failed G4B run, i.e. about 100 min of container time at 2×; the run itself lasted ~114 min and ended in ERROR.
- Remaining: hosen42 8.63 h, foysal 4.71 h.
- 4×L4 bills at 2× (participants, the staff fix around 10-05, and our own 2.0× measurement). An older memory line says one run used 1.05 quota-h per wall-hour, so verify each time.
- Queue time and data attached before the container starts are not billed.

**Hardware and environment:**
- Kaggle storage since about 10-21: GCSFUSE plus slower NFS. amn measured 0.95 → 0.026 GiB/s; single-thread 10.5 MB/s vs 16 threads 107 MB/s.
- Host RAM 202.6 GB; models 23.3 GB.
- Host CPU (assumed g2-standard-48): Xeon Platinum 8273CL, 48 vCPU / 24 cores.

**Research numbers (agents):**
- 2×TP2 KV: about 170k tokens per replica with DFlash2 at mem 0.83 (224k at 0.88; MTP 235k; no-spec 500k) vs 881k for TP4. Estimated throughput 0.7–1.0× of TP4.
- #36548: 111/304 wrong with DFlash2 vs 0/304 without speculation. Width 4, fp32 SSM, no decode graph or the #36014 beta fix each give 37/304; a dense BF16 lm_head gives 1/304.
- Our 4×L4 at c12: MTP 0.91×, DFlash2 W8 1.06× (fp32 SSM) and 1.39× (fp16 SSM). Target-only 289 tok/s; DFlash2 W8 fp16 SSM 404 tok/s, ~137 ms per step.
- Earlier greedy evidence: batch vs solo 0/4 identical; DFlash2 vs target-only 1/4; solo vs 16-batch common prefix 15.5 tokens.
- Token-efficient tunes: realistic −10 to −25 % thinking at ~60k. Licences are not OSI-approved, so they are likely not prize-eligible.
- S49: 49k gave LB 32.08 vs 60k LB 34.58.
- Franzen: prefill 10,553 tok/s, decode 837; 93.43 % cached; 15.25 % of time in prefill. His write-up says 27.89 LB; the user saw 34.30 on V1.

---

## 3. Failures, mistakes and root causes

1. **The G4B "55–62 min" runtime estimate (assistant's error, from the summary).**
   - The assistant gave 55–62 min, then contradicted it with "plus load". The run went about 114 min and ended in ERROR, which exhausted esfersami50 (30.21/30).
   - The user was furious; it cost him the planned 12 h runs.
   - Root cause per the startup agent (20:50): **our code never ran**. The log is `[]`, with no `nbformat MissingIDFieldWarning` (which normally appears 6–11 s after start) and no fallback submission.json. Likeliest cause: a dataset/model attach or mount stall after the container started.
   - The assistant first blamed the pinned old image (20:18 "own error #6"). That was then demoted to a weak candidate: staff say the image doesn't affect timers, and the same image started fine on 10-19/20/21.
   - The final ranking stays open: mount stall > pinned image > quota.
2. **The assistant's six errors (20:18):**
   - PCIe "64 GB/s" is bidirectional; one direction is ~32 GB/s.
   - "4.3 of 8 accepted, 46 % wasted" was wrong: accept len includes the bonus token, so ~47 % of the 7 drafts are accepted.
   - "Only 10–16 TFLOPS used" is not proven.
   - The 40–50 ms all-reduce figure extrapolated a large-message bandwidth to small messages.
   - The final report carried a stale PARITY notebook sha; the real one is `82b257a5`, never run.
   - It had omitted the old-image hypothesis.
3. **Earlier optimism about 2×TP2** ("the biggest lever") was proven wrong by the memory math (20:30).
4. **Token-diff false pass (02:32 → 03:58).** It reported 170/170, "template not the bug". GPT (03:18) warned that re-encoding text can't prove token identity. The real cause:
   - Duck's `_normalize_message_content` (`tool_agent.py` L3414-3418) deletes **every blank line** from the reasoning before it goes into history.
   - The transcript stores that same normalized text, so the diff compared normalized text with normalized text.
   - Evidence: there is no blank line in any of the 180 `[THINKING]` blocks, even though each branch holds 42–146k chars of reasoning.
   - It also took a script bug to get there: the first run gave C=0 and D_reason 170 because of how whitespace was cut from transcript blocks, and the assistant "fixed" that by rebuilding the reply canonically, which hid the real effect.
5. **The token-diff scope was narrower than claimed:** malformed tool calls, final or plain replies, and post-trim turns were excluded (03:28).
6. **Real bug in copy code rev 1:** `fo.write()` short writes were not handled, so a file could keep a hole in the middle that the first/last 1 MiB check would miss (03:28). Fixed in rev 2 with a write loop.
7. **Load-path rev-1 bugs fixed during the build (03:02):**
   - a hung FUSE read blocked the whole timeout (ThreadPoolExecutor join), fixed with daemon threads;
   - a size-only check passed on broken copies, fixed with a first/last 1 MiB comparison, later per-range hashes;
   - /dev/shm is RAM, so MemAvailable is now checked (later also the cgroup limit);
   - two clocks (monotonic vs the notebook's perf_counter) gave wrong deadlines;
   - **the biggest:** SGLang reads weights via mmap, which `rchar` doesn't count, and the single 18.6 GB file keeps the log silent. A slow but working ~37 min GCSFUSE load would have been killed at 15 min. Page faults now count as progress;
   - the draft model loads silently after "Load weight end"; also covered by page faults.
8. **Tooling problems:**
   - Git Bash heredoc/backslash escaping broke Python edits several times (an f-string, a write_text, the fixture `stat` line).
   - `sed` "unknown option to s".
   - `python -I` lost the user site-packages (kagglesdk).
   - pytest picked up a stray `D:\conftest.py` from ARC-AGI-3; fixed with the root dir and importlib import mode.
   - Two flaky timing tests had margins that were too tight.
   - A safety classifier stopped a response at 02:24 while inspecting a transcript dump.
   - The built-in safety check denied an `rm` with variable paths (05:00); rewritten without deletion.
   - API ENOTFOUND at 05:08.
9. **GPT claims found FALSE** by local validation:
   - Note 03: "115.5 ms context-normalized regression". It is the raw median below 10k.
   - Note 04: "width 4 can give more than 4 tokens per step". The maximum equals the width.
   - The local-validation agent's own "48 GDN layers BF16" (from the config ignore list) was also false. The final report resolved it: only the parent `linear_attn`, `norm`, `in_proj_a` and `in_proj_b` are ignored, so `in_proj_qkvz`/`out_proj` are INT4. BF16 would be ~6.5 GB/GPU, not 4.62.
10. **Per-step agent was wrong** that `--enable-linear-replayssm-spec` seems allowed for DFLASH. `kv_cache_configurator.py` L1125-1136 raises for non-KDA models (verified by the assistant).
11. **Agent coverage gap:** the profiling agent did not read all ~825 `_scratch/webresearch` files; it only grepped them (20:39).

---

## 4. Decisions and reasons; options rejected

- **2×TP2: nearly rejected (P5).**
  - About 170k KV per replica vs 881k for TP4; 12 streams of 55–76k don't fit with DFlash2.
  - It would worsen Duck's prefix thrash.
  - The router wheel is not in the wheelhouse and retries 5× by default.
  - PP and DP-attention are blocked for DFlash2.
- **Width 8→4 is not a free flag.**
  - The drafter is bidirectional within a block (`is_causal: false`), so width 4 is not a truncation of width 8.
  - Test 8/6/4 as arms with acceptance and correctness gates. Leave out width 3 (~2.5 tokens per step) and width 2 (crashes at graph capture).
- **NCCL tuning:**
  - The first round has only three arms: baseline, P2P (`NCCL_P2P_LEVEL`), `NCCL_SHM_USE_CUDA_MEMCPY=1` (GPT note 09 and the assistant's 03:28 table).
  - Don't touch `NCCL_PROTO` or `NCCL_ALGO=Tree`; never `NCCL_SHM_DISABLE=1` / `NCCL_P2P_DISABLE=1`.
- **PCIe-IPC all-reduce** (`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE`): deferred to P3. It needs FlashInfer ≥ 0.7, which means the 0.5.22 upgrade and a wheelhouse rebuild (~220 wheels / 5 GB), plus P2P/IPC on Kaggle (unknown).
- **#44401 backport:** chosen over upgrading. About 20 lines in `scheduler.py`, apply with #41872, fail clearly on a source mismatch. 0.5.22 doesn't contain it; only nightly ≥ dev20261016 does.
- **ReplaySSM, FR-Spec, HiCache, token-efficient tunes:**
  - ReplaySSM: blocked in code.
  - FR-Spec: `--speculative-token-map` is silently ignored by DFLASH.
  - HiCache: KV max 0.44 after rev 6, and open hybrid bugs #39830/#45400.
  - Tunes: licences, ARC accuracy unknown. Leaderboard-only, optional.
- **Load path (E14-0):**
  - range-parallel copy to /dev/shm → /tmp → in-place warm;
  - **bounded worker ramp 4→8→16→24**, stop when the gain is under 15 % or on error. GPT said "16–32 workers default করবেন না" (don't default to 16–32); rev 1 had used 24;
  - `.partial` directory plus atomic rename;
  - per-64 MiB-range hash, re-read check, full sha256;
  - safetensors check: no gap or overlap, each tensor size = dtype × shape;
  - cgroup and MemAvailable memory check;
  - phase record `E14_STATE` (PREPARE→COPY→VALIDATE→LAUNCH→READY);
  - `E14_LOADPATH=0` off switch;
  - JIT manifest with 13 mandatory fields;
  - fail-open to the original mount.
  - The assistant disagreed with GPT on one point: GPT warned against using the last JSON entry's end offset, but our code uses max(end). It accepted the stronger gap/overlap check anyway.
- **JIT cache:**
  - save without `nv/` and `cuda/` (#42917 stale driver cache gave PPL 9.98 vs 6.05);
  - restore only on an exact version/fingerprint match plus a known-answer greedy check;
  - the first run only saves; restore needs the user to turn that output into a dataset.
- **Baseline ordering:** unchanged W1 first, then +#44401, then +graph bs 1..13. This came from both GPT notes and was adopted (03:28).
- **Correctness gate:**
  - McNemar p ≥ 0.01 is rejected as an equivalence proof.
  - Use parity + A/A + paired CI on separate tasks + a predefined δ.
  - Admitted limit: 20–30 tasks give a CI of about ±15–20 points.
- **Re-prefill fix:**
  - `E12_RAW_REASONING_HISTORY=1` in arc2_duck (sha 7644aa06 → 758bddb5), default off for rev 6g.
  - It replaces only when the reply is the same, skips on markup or mismatch, and is consumed once.
  - The decision to make it **default ON in E14** came at 05:32, after the user's 05:27 "don't hide behind base" rule.
- **Deeper misses (18, 66,771 tokens):** attributed to server Mamba checkpoint eviction (E14-M), not to Duck. The fake server showed 0 earlier-message rewrites, and 10 of the misses landed on an older request's checkpoint.
- **E14-A plan (Notes/13, 05:32):**
  - A new `build_e14.py`, never touching E12 files.
  - Nine decisions D1–D9.
  - Two runs. A1: ≈61 min, ≈2 h quota, HARD_STOP 95 min (≤3.2 h). A2: ≈80–93 min, ≈2.7–3.1 h quota, HARD_STOP 110 min.
  - Phase timeout at worst-case +50 %.
  - If not ready at 45 min, stop with partial results (~1.5 h quota).
  - Two failed launches drop an arm; a parity break makes an arm REJECTED.
  - A1 on hosen42.
  - Calendar: submission 1 on 10-26, final day 11-02.
  - The user was asked to choose: two runs, or fold A2 into the E14-C commit.

---

## 5. Open items

- **Unanswered at the end of the part (05:32):**
  - Q1: split into A1 → A2 → E14-C, or fold A2 into E14-C?
  - Q2: start the build?
- **Promised, not done:**
  - `build_e14.py`;
  - `e14_measure.py` (workload, /metrics running check, log parse, profiler summary, replay, latch test, parity, report);
  - the NCCL bench script;
  - the #44401/#41872 backport with source-hash checks;
  - a full notebook run on the fake server;
  - the written per-run account / quota / wall / HARD_STOP / sha sheet.
- **Needs a GPU to settle:**
  - the kernel split of the 115–140 ms step (estimated 87–131, centre 108). The profiling agent left 60–80 ms unexplained, while the per-step agent explained nearly all of it; never reconciled with a trace;
  - the NCCL transport (SHM vs P2P), P2P/IPC availability on Kaggle, and the cost at 983,040 B;
  - acceptance at widths 6/4;
  - whether #44401 is the cause of 11+1 queued (GPT: "not yet proven by local trace");
  - token-level confirmation of the re-prefill fix (text match ≠ token-id match);
  - the 12 short-reasoning replies that still lost reuse (non-canonical sampled tokens or checkpoint eviction?);
  - the real `/dev/shm`, disk, nproc and cgroup limit on 4×L4;
  - whether 24+ threads beat 16 on GCSFUSE;
  - whether the JIT cache really saves ~360 s;
  - whether `flashinfer-jit-cache` is in the wheelhouse.
- **Unverified locally:**
  - real Linux /proc behaviour (only a fake /proc; no WSL or Docker);
  - whether the model thinks differently when it reads its own reasoning with paragraph breaks restored (no evidence either way).
- **G4B ERROR root cause** is still unconfirmed; no platform job events. Optional: a Kaggle forum post asking for a quota refund (user decides).
- smoke_rev6 is stale: 1 FAIL plus the `min_workers` TypeError on old code too. The test file dates from rev 6c and was never updated.
- DFlash2 correctness bugs #36548, #38009, #43514, #37475 and #39087 are all still open upstream.
- The final-report agent found that "note 02" mentioned in note 00 doesn't exist in the folder.

---

## 6. Insights a fresh reader might miss

1. **Big reversal on re-prefill:**
   - 02:32 said "template is not the cause, 170/170 match".
   - 03:58/05:17 said the cause is **Duck deleting blank lines from reasoning**: 24/24 breaks at the first blank line vs 24/24 full reuse with the switch.
   - The first test was blind because the transcript stores the already-normalized text.
2. **The same bug is in Franzen's notebook** (`tool_agent.py` L3414-3418 identical, from the Tufa harness). His write-up mentions the symptom ("re-tokenization…"), but he worked around it with a server checkpoint patch. On his GPU it costs only a few percent (prefill 10,553 tok/s); on 4×L4 it costs 3–4× more.
   - It slows us down; it does not cause wrong answers.
   - It does not affect our best LB of 34.58 (S1-60K, which is not Duck).
3. **The step time includes prefill** (window wall / 40 forwards). A raw "step ms" from the logs is not a pure decode step.
4. **accept len counts the bonus token**, so width W gives at most W tokens per step. Any "4.3/8" waste arithmetic is off.
5. **`speculative_draft_model_quantization: None` in server_args is correct.** It is how SGLang stores `unquant`. Don't "fix" it.
6. **Draft acceptance varies by run** (3.87–5.10), so a fixed 4.3 is a poor basis for width decisions.
7. **The PCIe-IPC "one in-tree option" is dead code on our wheelhouse** (FlashInfer 0.6.18 lacks the kernel). The NCCL agent had suggested it before the TP2 and engine-version agents found this.
8. **Conflicting claims about ReplaySSM:** the per-step agent says it is possible, the source says it raises. Trust the source.
9. **The GDN precision conflict was resolved as INT4.** Only `in_proj_a/b`, norm, lm_head, MTP and vision stay BF16. The lm_head being BF16 means the strongest #36548 mitigation is already in place, but 1/304 errors remained even there.
10. **11-of-12 cause:** this part attributes it to #44401 (held-row double counting during chunked prefill). The E14 context given for this task says the later finding was the DFlash min-free-slots delayer (`--min-free-slots-delay 1`). So the #44401 attribution here was later refined or reversed, and GPT already warned (03:18) that it was unproven.
11. **The E14-A shape changed later:** this part plans two runs (A1 HARD_STOP 95 min, A2 110 min) and a "45 min not ready → stop" rule, while the current notebook has one 200-minute profile with a 75-minute give-up. The E14-A plan here was not final.
12. **The 64 MiB NCCL number was fp32 in eager mode.** It is not representative of bf16 983 KB collectives inside CUDA graphs.
13. **CUDA graph capture took 382–394 s, and ~362 s of that is the first bs=12 bucket.** It is silent for about 6 min, so a short "no log line" timeout would kill a healthy server.
14. **A loader silence trap:** SGLang uses mmap, so `rchar` does not grow; with one 18.6 GB file the log is silent too. Only page faults show progress.
15. **Instrumentation can't diagnose a run that never starts.** If Kaggle never reaches the notebook (10-22), first-cell logging gives nothing.
16. **The user explicitly reversed the assistant's base-protection caution** (05:27). From then on, proven fixes are default-on in E14, and only engine changes are A/B arms.
17. **The multi-account question is settled:** three different people, one team (03:37).
18. **The SGLang Qwen3 thinking-budget logit processor hard-codes old token ids (151667/151668 vs 248068/248069)** and silently does nothing (#25536). `bench_serving` "random" downloads ShareGPT, so use `random-ids` or recorded ARC prompts offline.
19. **Top-team rumours:** the profiling agent guesses that Darren's "library code" fix is #44401-like and that mikelou1's "tiny kernel issue" is a silent correctness fault such as #42917. These are inferences only.

---

## 7. Relevance to the current E14-A notebook (concrete checks)

| # | What to check or fix in E14-A | Why / evidence (time) |
|---|---|---|
| 1 | Set **`SGLANG_JIT_CACHE_DIR` explicitly** and include it in the cache tar. | It defaults to `~/.cache/sglang/jit`, outside `SGLANG_CACHE_DIR`; our env never set it (startup agent, 20:50). |
| 2 | The cache tar must exclude **both `nv/` and `cuda/`**. | `e11_install` sets `CUDA_CACHE_PATH=/tmp/e11-cache/cuda` separately from `SGLANG_CACHE_DIR=/tmp/e11-cache/sglang`. #42917 (stale driver cache, PPL 9.98 vs 6.05). Rev 1 already excluded "nv/ and cuda/" (03:02); confirm the E14-A tar code still does. |
| 3 | JIT restore only with a full fingerprint (13-field manifest: package versions, CUDA, driver, sm, per-wheel sha256) plus a known-answer greedy token/logprob check; on a mismatch, wipe and go cold. | GPT note 08/09 (03:18) and the 03:28 plan. |
| 4 | Watchdog progress signals must include **page faults** (minflt/majflt of the server process tree), not only log size and `rchar`. | mmap reads don't count in rchar; one 18.6 GB file keeps the log silent; the draft loads silently after "Load weight end" (03:02). Without this, a slow but healthy ~37 min GCSFUSE load is killed at 15 min. |
| 5 | The compile-phase stall threshold must be **> 8 min** (12 min chosen). | The first capture bucket (bs=12) is silent for ~6 min; ~362 s of the 382–394 s capture (20:41, 20:50). |
| 6 | All deadlines on **one clock** (`time.perf_counter`, the same as `SGL_GIVE_UP_AT`). | Rev 1 mixed monotonic and perf_counter, giving wrong deadlines (03:02). |
| 7 | The copy worker pool must use **daemon threads that are never joined** after a stall. | A hung FUSE read blocked `ThreadPoolExecutor` join forever (03:02). |
| 8 | Copy safety: write loop for short writes; `.partial` directory + atomic rename; per-range hash plus re-read; safetensors gap/overlap and dtype×shape check; no launch from a partial copy; fail open to `/kaggle/input`; `E14_LOADPATH=0` off switch. | Rev 1 short-write bug (03:28); GPT gap 2/3 (03:18). |
| 9 | Worker count **ramps 4→8→16→24** (stop if the gain is under 15 % or on error). Don't hard-code 16–32. | GPT "16–32 workers default করবেন না" (03:18); rev 2 (03:58). |
| 10 | Memory guard before /dev/shm: use min(MemAvailable, **cgroup limit**) and leave host headroom for serving. | /dev/shm is RAM; 23.3 GB of 202.6 GB; GPT gap on cgroup exhaustion (03:18). |
| 11 | **First executable line:** print + flush, then log `/proc/mounts`, `df`, `free`, nproc and the GCSFUSE detection **before touching `/kaggle/input`**. Also add a phase record and heartbeat. | The 10-22 failure had an empty log and no diagnostics (20:50). This cannot help if Kaggle never reaches the notebook. |
| 12 | Order the arms as **unchanged W1 baseline first**, then +#44401, then +graph 1..13. The patch must **check source hashes and fail clearly** (never monkey-patch silently). | GPT gap 4 (03:18); plan table (03:28). Note: later work moved the 11/12 cause to the min-free-slots delayer. Keep #44401 and `--min-free-slots-delay 1` as separately attributed arms, with the latch test checking real 12 running, no negative free-slot count, and held-row counting. |
| 13 | CUDA graph arm: `--cuda-graph-bs-decode 1..13`. Check capture success, OOM headroom (only 1.88 GB free per GPU after the graphs at baseline) and the added capture time. | Captured [1..8,10,12] vs max running 13; batch 9/11 run padded and 13 runs without a graph (20:39). |
| 14 | Any 12-running window must be **verified from /metrics**; otherwise mark it invalid. | Probe bug: 27.7 % of G4B at 6 running; 11+1 queued (20:39); memory `verify-real-concurrency`. |
| 15 | Do **not** add `--enable-mixed-chunk`. | DFlash token-0 batch poisoning #43514 (20:22, 20:39). |
| 16 | Do **not** add an `--enable-linear-replayssm-spec` arm. | It raises for DFLASH on non-KDA models (`kv_cache_configurator.py` L1125-1136, verified 21:12). |
| 17 | Do **not** count on `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` with the 0.5.21 wheelhouse. | FlashInfer 0.6.18 lacks the kernel (20:29, 20:45). If set, it is dead code, so the arm result would be meaningless. |
| 18 | NCCL arm: start with only baseline / `NCCL_P2P_LEVEL=SYS` (or PHB) / `NCCL_SHM_USE_CUDA_MEMCPY=1`. Log `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,ENV,TUNING` and grep "via SHM" / "P2P". Check correctness first (P2P in a VM can hang silently). Never `NCCL_SHM_DISABLE=1`. | NCCL agent (20:20); GPT note 09. |
| 19 | NCCL bench at the **real sizes**: bf16 983,040 B × 140 inside a CUDA graph, plus an 11.9 MB-per-rank all-gather; also a 64 MiB point to reproduce 20.9 ms. Record `nvidia-smi topo -m` / `-p2p` and the `can_device_access_peer` matrix. | The old measurement was 64 MiB fp32 eager only (20:41). |
| 20 | Profiler: `/start_profile` with `num_steps`; `SGLANG_TORCH_PROFILER_DIR`; `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS=12`; **one profiler window per server process**; publish TPS only from non-profiled windows; `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` and `SGLANG_RECORD_STEP_TIME=1` are cheap counters. `--enable-layerwise-nvtx-marker` needs graphs off; `--enable-profile-cuda-graph` covers capture only. | Profiling agent (20:39); GPT notes 03/04. |
| 21 | Step-time metrics: the log proxy includes prefill. Compute the decode step from the device timer or decode-only windows, and bin by context (≤10k ≈115 ms, 50–80k 133–140). | 20:41, 21:12. |
| 22 | Width arms: 6 and 4 only (2 crashes; 3 is not useful). The maximum accept len equals the width. Health fails: accept rate 0.00 in any window; mean accept below 3.5 (W8) / 3.2 (W6) / 2.5 (W4); long single-token repeats. | DFlash agent (20:22); local validation (20:41). |
| 23 | Greedy parity design: A/A target-only, concurrency 12 vs serial, target-only reference; exact parity plus first-divergence index plus a logit-margin check (top-1 vs top-2 margin > 1 nat counts as a bug). **No McNemar-as-proof**; use a predefined δ and a CI. | GPT note 09 gap 5; DFlash agent preflight gate. Batch non-determinism already seen (0/4 identical). |
| 24 | Token-level replay: enable output ids with the server flag `--return-output-ids` (or per request `return_output_ids_in_sglext` / header `x-sglext-return-output-ids: 1`; `--return-input-ids` for prompts) in the **E14-A replay client**. Duck does not store raw responses. Name offline results "reconstructed_divergence". | 03:24–03:28; tokenizer shows `"\n\n"` = [271] or [198,198]. |
| 25 | The Duck run must have `E12_RAW_REASONING_HISTORY=1` **on** (arc2_duck sha **758bddb5**, not 7644aa06), and branch_result must carry its counters (kept/same/mismatch/markup). Check that the E14 bundle ships the new arc2_duck. | Fake-server probe 24/24 (05:17); the 05:32 plan says default on in E14. |
| 26 | Keep `--speculative-draft-model-quantization unquant`. **Don't assert server_args == "unquant"**, because it is logged as `None`. | Resolved at 20:42. |
| 27 | Reconcile the stop rules: this part plans "not ready at 45 min → stop" and HARD_STOP 95/110, while the current notebook has a 75-min give-up and a 200-min profile. Pick one and write it in the pre-run sheet (account, current quota, normal and worst wall time, 2× quota, HARD_STOP, sha). | 05:32 plan vs the current E14-A context; the user's anger over the 55–62 min estimate. |
| 28 | int8 Mamba checkpoint arm: the flags exist (`enable_int8_mamba_checkpoint`, `int8_mamba_ckpt_size`, `mamba_track_interval` default 256). It is incompatible with HiCache (irrelevant here). Measure deeper misses and resume parity. | 05:28 source check; E14-M. |
| 29 | If any TP2 test is ever added: pass `--enable-p2p-check` or `--disable-custom-all-reduce` (SGLang skips the P2P test by default). | TP2 agent (20:29). |
| 30 | Offline workloads: use recorded ARC/Duck prompts (random-token prompts give unrealistic DFlash acceptance); `bench_serving --dataset random` needs ShareGPT online, so use `random-ids`. | 20:39. |

No direct code defect in the current E14-A notebook can be confirmed from this part alone (the notebook wasn't built yet). Items 1, 2, 4, 5, 12, 17, 25 and 27 are the ones most likely to be wrong or missing and should be checked first.
