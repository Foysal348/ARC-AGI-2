# Insights from part_09 (2026-10-22 11:42 to 2026-10-22 20:08)

Source: `part_09.txt`, all 5,778 lines read in order. Times are as the transcript gives them. Bengali quotes are verbatim; the English meaning follows in brackets.

**What this part covers:**
- the end of the rev 6f self-audit (mutation tests, zip for GPT);
- three GPT audits of rev 6f, which became the H7 items;
- the rev 6g build ("last version") and 5 verification agents;
- the LB check of rev 5 LOW, then the G4B rev 6g commit run on esfersami50, which ended in **ERROR with zero output after ~114 min and used up the account's quota**;
- the user's anger about the time estimate;
- Kaggle discussions on the GCSFUSE storage slowdown;
- a 5-agent deep research on the fastest serving setup;
- the step-time bottleneck analysis, a GPT bottleneck audit, and the **creation of Experiment 14 (Optimized Engine on 4xL4)**.

**Cut `[result]` lines that matter, flagged where they come up:**
- the profile/hard-stop code (line 4226, `[+4280]`);
- the launch-retry code `e11_next` (line 3242, `[+3416]`);
- the harness re-run outputs (lines 1530 and 1580);
- the CELL 6 try/except positions (line 2177).

I do not guess what was cut.

---

## 1. User instructions, rules, preferences and decisions

| Date/time | Verbatim (Bengali) | English meaning / type |
|---|---|---|
| 10-22 13:42 | "ফিক্স লিস্ট সাজাও শুধু সত্যতা যাচাই করে,আরো দিচ্ছি," | "Organize the fix list only after verifying each claim is true; more coming." Rule: validate GPT claims before listing them. |
| 10-22 13:51 | "দেখে নাও" | "Take a look." This was after an API error interrupted the work. |
| 10-22 14:12 | "আবার জিপিটির টেক্স পড়ো,আর কিছু বাদ গেছে কি না দেখে নাও,এরপর আবার ভ্যালিডেট করে ফিক্স লিস্ট সাজাও" | "Read the GPT texts again, check whether anything was left out, then validate again and organize the fix list." Re-read rule. It found 2 missed items (H7-12, H7-13) and 2 half-written ones. |
| 10-22 14:25 | "এগুলো কি বাগ ছিলো?" | "Were these bugs?" |
| 10-22 14:25 | "লাস্ট আপডেট করে দিও," | "Make the last update." This triggered the rev 6g build (all H7 P1+P2 and the cheap P3 items; the offline tokenizers wheel left out). |
| 10-22 14:38 | "দুই তিন দিন ধরে এভাবে লুপে পড়লে আমরা হেরে যাবো,এটাই লাস্ট ভার্সন মনে রাখ" | "If we stay in this loop for two or three days we lose; remember this is the last version." **Standing rule:** rev 6g is the last version before GPU. Saved to memory as `stop-audit-loops.md`. The assistant's interpretation (14:38): code is touched only for a confirmed bug that changes the score (what the model sees or what is submitted); metric, naming and hardening findings go to the post-run list; no new GPT audit zip unless the user asks. |
| 10-22 15:37 | "আবার আগের মত বাগ এর জন্য পুরো রান নষ্ট হবে না তো?" | "Won't the whole run be wasted by a bug again, like before?" Worry about wasted runs. |
| 10-22 15:39 | "এটা কি সাবমিশন তৈরি করে না?" | "Doesn't this create a submission?" |
| 10-22 15:42 | "তুমি জিপ করে দিও,আমি জিপিটি দিয়ে আলাদা আলাদা কনফিগ বা স্টাইলে ভিন্ন ভিন্ন রান দিতে বলবো,যাতে বাগ ধরা পরা।এটা তুমিও করতে পারো।তুমিই করো" then "৫ টা এজেন্ট নিও" | "Zip it; I'll ask GPT to do runs with different configs or styles so bugs get caught. You can do this too. You do it." Then "take 5 agents". |
| 10-22 15:43 | (rejected the first two agent launches, "Bug-hunt 1: config sweep" and "Bug-hunt 2: fault injection") then "যা যা নতুন ফিক্স করা হয়েছে সেগুলো আদৌও কাজ করছে কি না,সেটা জানার জন্য" | "To find out whether the new fixes actually work at all." Correction: the purpose is verifying the fixes, not a generic bug hunt. |
| 10-22 16:27 (repeated 16:52) | "আমরা গত পড়শু একটা সাবমিট দিছিলাম,তুমি এলবি দেখে আর নোটবুক দেখে এটা যাচাই করো,আমাদের এজেন্ট আসলে কেমন? ঐটার কভারেজ তুমি জানো আগেই।" | "We submitted the day before yesterday; check the LB and the notebook — how good is our agent really? You already know its coverage." |
| 10-22 16:56 | "এই এত সময় কেন নিচ্ছে?" | "Why is this taking so long?" Complaint about slowness. |
| 10-22 17:09 | "রানটাইম লাস্ট এজেন্টের কত?" / "লাস্ট নোটবুকের যেটা বানালাম" / "G4B রান" | "What's the runtime of the last agent?" / "the last notebook we built" / "G4B run". |
| 10-22 17:12 | "সামি ভাইয়ের আর কত জিপিইউ আছে?" | "How much GPU does Sami bhai have left?" (the esfersami50 account) |
| 10-22 17:14 | "G4B commit run  করো সামি ভাইয়ের একাউন্টে" | "Do the G4B commit run on Sami bhai's account." **Explicit go** for the GPU run, given after a check-only. |
| 10-22 17:17 | "এটার কভারেজ কত ১২ ঘন্টায়?" | "What's its coverage in 12 hours?" |
| 10-22 17:20 | "আমাদের এই রানে টোকেন বাজেট কত? অন্য গুলোর সাথে তুলনা করো।" | "What's the token budget in this run? Compare it with the others." |
| 10-22 17:23 | "এই যে এই রানে কমানে,এটা কি প্রমাণ নির্ভর?" | "This reduction in this run — is it evidence-based?" It challenged the "2.5–3× reduction" sentence. |
| 10-22 17:27 | "চলো ডিস্কাশন গুলো আবার পড়ি এই একই মডেল নিয়ে।দেখে আসো,বাদ দিও না" | "Let's read the discussions again about this same model; go look, don't skip anything." |
| 10-22 18:07 | "27B FP8 + EAGLE এটা আবার কি মডেল?" | "27B FP8 + EAGLE — what model is that?" |
| 10-22 18:11 | "শত শত রিসোর্স দেখে রিসার্স করো,আর বলিও এখানকার কোন সেট আপ এই প্রতিযোগিতার জ ন্য সবচেয়ে ভালো ,দ্রুত?" (with an image) | "Research across hundreds of resources and tell me which setup is best and fastest for this competition." |
| 10-22 18:21 | "কত সময় লাগবে? আর ক্যাগলের রানের আপডেট দে" | "How long will it take? And give me an update on the Kaggle run." |
| 10-22 18:24 | "তুই না বললি হিসাবে রান লাগার কথা ৫৫–৬২ মিনিট? এবার বেশি কেন লাগতেছে ,জানুয়ারের বাচ্চা,GPU quata ar koto ache?" | "Didn't you say the run should take 55–62 min? Why is it taking longer now, [insult], how much GPU quota is left?" **Complaint** about the wrong estimate. |
| 10-22 18:28 / 18:35 | "এই ৯ টা এজেন্ট কি করতেছে?" / "কত সময় নিবে এরা?" | "What are these 9 agents doing?" / "How long will they take?" |
| 10-22 18:40 | "তুই রিপোর্টে অনেক কিছু স্কিপ করিস সর্বদা,এটা আর কখনোই করবি না।পূর্ণ report always" | "You always skip a lot in reports; never do it again. Full report always." **Standing rule: always a full report, never a digest.** Saved to memory `read-everything-fully.md`. |
| 10-22 18:46 | "আবার শুরু কর" | "Start again." This came after "Request timed out". |
| 10-22 18:56 | "ক্যাগলে রান শেষ হয়েছে কি> আর জিপিইউ কত আছে?" | "Has the Kaggle run finished? And how much GPU is left?" |
| 10-22 18:59 | "…তুই বললি ৬০ মিনিটের মত লাগবে তাই রান দিলাম।এরপর আমি ভাবছিলাম ,৩০ মিনিট থাকলে ১২ ঘন্টার করে ২ টা রান দিবো,… এত বড় ভুল কিভাবে করলি? কোড আবার দেখ ,কই ভুল করছিস?" (insults omitted) | "You said it would take about 60 min, so I ran it. Then I planned, if 30 min [of quota] remained, to start two 12-hour runs… how did you make such a big mistake? Look at the code again, where did you go wrong?" **Major complaint.** The user's plan relied on the estimate: leftover quota on esfer was to start two 12 h runs, using the observation that a run started with quota left finishes. |
| 10-22 19:04 | "এই দিকে আর মাত্র ১১ দিন আছে।তুই এত বড় ভুল কিভাবে করলি? আমায় হতাশ করে দিলি।" | "Only 11 days left. How could you make such a big mistake? You disappointed me." **Deadline:** 11 days left as of 10-22. |
| 10-22 19:38 | "আমাদের ইঞ্জিনের কোন সমস্যা এর গতি ৬০০–৮০০ এ যেতে দিচ্ছে না? সেটা কি বের করা যাবে?" | "What problem in our engine keeps it from 600–800 tok/s? Can it be found?" This is the start of the engine-speed track (E14). |
| 10-22 19:48 | "দুটো TP2 একটা TP4-এর ২.৪× এটা আমাদের ক্ষেত্রেও সত্য হলে আমরা কি কি পাবো?" | "Two TP2 = 2.4× one TP4 — if true for us too, what would we get?" |
| 10-22 20:02 | (pasted GPT "GPU Bottleneck Research & Validation") + "ডিপলি রিসার্স করো।আর একটা এক্সপেরিমেন্ট ফোল্ডার খুলো,আমরা Optimized Experiment এর দিকে যাবো। আর তোমার এজেন্ট গুলোর এনালাইসিস আর এইটা,আরো কিছু দিচ্ছি সব নোট করে রাখিও,ভুলে যেও না।আর হা তুমি সব পড়ো না,এটা মেমরিতে রাখিও,তোমাকে সবই পড়তে হবে,পুর্ণ রিপোর্ট ই দিতে হবে।" | "Research deeply. Open an experiment folder; we are moving to an Optimized Experiment. Note down your agents' analyses and this, and more I'll give — don't forget. And yes, you don't read everything — keep in memory: you must read everything and always give the full report." **Decisions and rules:** create Experiment 14; save every analysis and GPT paste in its Notes; read everything; full reports. The rule is saved in memory (`experiment-14-optimized-engine.md`; `read-everything-fully.md` reinforced). |

**Standing constraints restated in the compaction summary at 17:35, kept verbatim there:**
- never push, run or submit without an explicit go;
- "অনুমতি ছাড়া কমিট করবি না" ("don't commit without permission");
- never read or echo tokens;
- validate every claim and read fully;
- use the D: drive for scratch;
- 60k thinking floor (the rev 6 caps and gate 4b are approved exceptions);
- pre-run capacity check;
- never rebuild Kaggle-run notebooks;
- past negatives are not verdicts; trust user observations;
- rev 6g is the last version;
- chat in Bengali.

**Commitment the assistant made after the complaint (19:04), which binds future runs.** Before every run it will state in writing:
1. how many minutes the run may take if storage is slow, including the worst case;
2. how much quota each account will lose and how much remains after;
3. at what time to stop it if it runs longer than estimated.

It also said nothing runs until the user agrees.

---

## 2. Measured facts and results

### Rev 6f self-audit (11:42–13:06)
- **Mutation tests:** 18/18 caught (15 adapter, 3 notebook). The builder mutation run printed `clean [r6] 57/57`, `H6-3 55/57`, `H6-6 56/57`, `H6-17 52/57`.
- **Fix IDs:** 126 IDs mapped to tests. Real gaps: R6-P3, H6-12, H6-17.
- **Final rev 6f sha:** EVAL12R6 `ac7a46a8…`, EVAL12R6G4B_REBUILD `913aef97…`, PARITYR6 `7d35d72b…`.
- **Tests:**
  - smoke 163/56/41;
  - harness g4b 58, r6 57, rerun 23, launchfail 6, parity 19;
  - verify_h6: 6/6 claims "NOT REPRODUCED".
- **Zip:** `E12_REV6F_IMPLEMENTATION_AUDIT_FOR_GPT_2026-10-22.zip`, 16.0 MB, 409 files.
- **Without tokenizers:** smoke rev6 gives "PASS 162 SKIP 1 … exit 2". One timing test failed at 3.22 s against a 3.0 s limit; the limit was raised to 4 s.
- **240-task sim:** at 371 and 277 tok/s, 240/240 tasks exposed, no KV hold. Replay corpus 7/8, 5/8, 10/19, unchanged.

### GPT rev 6f audits (13:42–14:09)
**Audit 1:**
- H6-4 race reproduced: "Callback completion order: [2000, 1000] … Notebook's external latest record: 1000 ← WRONG".
- Historical gold: Rev 5 xhigh 5/8, Rev 5 low 10/19, Rev 6d G4B 2/19.

**Audit 2:**
- G4B window 42,720 − 1,463 → 2,062.85 s, 171.9 s per task.
- Nudge opportunity 7/12; reasoning still possible before the pause 4/12.
- Sim at 277 tok/s: 240/240 initialized, ~47.6k tokens/task. At 371 tok/s: ~63k.
- F6-4: PARITY still has `urlopen(req, timeout=900)` × 24 probes, and the tool-call identity check compares arguments only.

**Audit 3:**
- Nudge emitted on the pause request itself for 3/12: 4c3d4a41_b0 at 33,107 tokens, 9385bd28_b0 at 39,353, cbebaa4b_b0 at 38,432.
- KV replay: 6 of 7 logs never trigger the 0.95 gate. E12 EVAL30 rev 1 hit 100 % KV, with an 18.2 min hold in replay.

**Assistant's reproductions:**
- `verify_h7`: the ordering race is REPRODUCED, but the content was identical (1000 = 1000).
- `verify_h7_freeze`: disk 1 vs `_final` 2, and "60 freezes under a live mutating agent" → 60/60 mismatched.
- F6-2 reproduced: "agent_created true, analyze_calls 0, generated_tokens 0".

### Rev 6g build (14:26–15:20)
- **sha:** EVAL12R6 `74a220de876132e1…`, **EVAL12R6G4B_REBUILD `5d2bc39f44971029…`**, PARITYR6 `82b257a565c182c5…`. The embedded adapter equals the source. The 9 Kaggle-run notebooks are untouched.
- **Tests:**
  - smoke rev6 171 (after a first FAIL on H6-11, and a harness Traceback at `_pg.get("gpu_hours")` because `m` was used before it was defined; both fixed);
  - smoke rev5 56, adapter 41;
  - harness g4b 60/60, r6 59/59, rerun 23, launchfail 6, parity 19.
- **Without tokenizers:** smoke, verify_h6 and facts_size all print INCOMPLETE and exit 2.
- **Facts:** "n=120 p50=699 p90=779 p99=798 max=798 mean=672".
- **sim240 at 277 tok/s:** 240/240, KV max 0.40.
- The mutation suite was **not** re-run for rev 6g, because two anchors moved with H7-11.
- **Adapter sha256** `7644aa06…fdf1`, per verification agent 1.

### Verification agents (15:43–16:45)
**Agent 4 (metrics):**
- 278/280 comparisons match over 8 harness runs.
- F1 split of `new_for_task` (b0/b1): reported 9/3 vs 4/8 by verification time (G4B same-output); R6 same-output reported 10/2 vs 2/10.
- F2 `gpu_hours` inflation: R6 +15 % (0.2078 vs 0.1811), +13 %, +11 %; G4B +1–1.5 %. The diagnostic alone can add up to 240 s.
- Nudge run: 12 via tool, 12 seen, p50 4,200 tokens.

**Agent 1 (scheduler):**
- H6-4 works (6 configs).
- H7-11 is partial: "mismatches: 11 of 48" when forced, and 2 more unforced under load (`w=12 n=60`).
- H7-1 works: 1,350 publishes, contiguous seq. But `publish_seq` is in 0 of 6 `branch_result.json` files.
- 0 pop-time deferrals in 227 randomized runs.
- KV gate: holds at 0.95 and not at 0.9499; fail-open on {}, None, NaN, 1.7, −0.1, 'x', inf.
- An I/O error in `write()` kills the worker thread; the final `freeze()` raises out of `solve_all_r6`, and `SCHED_LAST` is not set.

**Agent 2 (tool path):**
- R3: a multi-tool-call reply rolls back in 5 of 1,829 replies (0.27 %).
- R4: the print guard hides 4-number tuple lines; 13 % of the collapsed parts in recorded transcripts are not grids.

**Agents 3 (notebook cells) and 5 (regressions + tools):** died with "API Error: Can't reach the API server (ENOTFOUND)" at 16:59.

### Leaderboard and agent quality (16:28–17:03)
- **Submissions read from Kaggle:**
  - `57456562 | 2026-10-20 23:58:34 | COMPLETE | public 17.50 | E12 Duck rev5 LOW`;
  - `57412468 | 2026-10-19 | public 31.81 | E10 S1 55K Poetiq`;
  - `57372299 | 2026-10-18 | public 29.17 | E11-06 P1 60K`.
- **Rev 5 LOW config (notebook `26fff41e`):** effort low, 6 live, 2 branches per task, 49,152 output per request, 100k soft cap per branch, ORDER2.
- **Per task:** xhigh 141k tokens / 9.7 min; low 145k / 9.1 min. Throughput 244 (xhigh) and 265 (low) vs 325 in rev 4.
- **Derived:** ~74 of 240 tasks reached (31 %) → ~21 solved of ~37 reached ≈ **56 %**, matching eval 7/12 = 58 %.
- **budget_sim (rev 5 recorded runs):**
  - low: full 7/12, 50k → 4/12, 35k → 2/12;
  - xhigh: full 4/6, 50k → 2/6, 35k → 1/6.
- **First check() timing:** rev 5 at ~55–57k tokens; rev 6d at ~29k. The solved branches wrote their transform at 18k and 26k; 7 of 12 branch 0s never wrote a transform.

### Rev 6g budget and coverage (as stated)
- **Caps:** branch 0 first request 12,288; 24,576 after the first successful Python call; 32,768 after two cuts in a row. Branch 1 first request 24,576.
- **Stage 1:** soft pause at 28,000 (28–44k actual, ~33k average). Stage 2 chunks ≥ 12,288. Branch soft cap 100,000. Context per stream 72,228 (KV pool ÷ 12).
- **Hidden 12 h sim (window 41,257 s, 12 workers):**

  | | 371 tok/s | 277 tok/s |
  |---|---|---|
  | Tasks reaching stage 1 | 240/240 | 240/240 |
  | Stage 1 finished at | 48 % of window (~5.5 h) | 64 % (~7.3 h) |
  | Tokens per task | ~63k | ~47.6k |
  | Branch 0 resumed in stage 2 | 232 | 133–151 |
  | Branch 1 started | ~177 | ~89 |
  | KV hold | 0 | 0 |

  Coverage needs about 240 × 33k ≈ 7.9M tokens, i.e. **≥ ~192 tok/s average**.
- **Comparison:** rev 4 EVAL6 ~203k per task; Kris Dreemurr ~70k per task (35k per attempt), CV ~45 %.

### G4B rev 6g capacity check and the run (15:20–19:13)
- **Pre-run estimate (15:20 and 17:12):** "প্রায় ৬০ মিনিট" ("about 60 min").
  - Load ~12–20 min (previous G4B 1,197 s), preflight < 1 min, solve ~2,063 s ≈ 34 min, diagnostic ≤ 4 min, report 1–2 min → total 55–62 min.
  - Profile hard limit 80 min (hard stop 2 min before). Rev 6d G4B took 3,528 s.
  - KV: 12 streams × 72k = the full pool; earlier max 0.44.
- **Check-only (17:12):**
  - "authenticated kernels owner(s): ['esfersami50']";
  - models OK: `foysalemonshanto/qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1`, `qwen3-8-27b-dflash2/PyTorch/bf16/1`, `qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`;
  - datasets OK: `sglang-0-5-21-cu13-wheelhouse` 20 files, `arc3-vllm-h100-wheelhouse-v3` 20 files;
  - "gpu quota used h 26.87 of 30.0";
  - sha OK.
- **Push (17:14–17:17):** the first attempt hit a connection reset. The status check returned "Permission 'kernels.get' was denied" (nothing had been saved). The second attempt worked: "saved + commit run queued: esfersami50/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b version 1". Kernel listed at 2026-10-22 17:16:19 UTC.
- **Status timeline:**
  - RUNNING at 17:39, 18:22, 18:25 and 18:31–19:09;
  - three empty API answers at 18:38–18:44;
  - **ERROR at 19:12** (~114 min after the push).
- **Quota on esfersami50 during the run:**
  - 29.18/30 at 18:25 ("left 0.82 h"), i.e. 2.31 h used in ~69 min;
  - **30.21/30 at 18:56 ("left -0.21 h")**, i.e. ~3.34 h used in ~100 min.
  - That is **~2× quota per wall hour on 4×L4**, confirmed later by the official rule quoted by GPT: "L4x4 notebooks … দ্বিগুণ quota consume করে" (consume double quota).
- **Output fetch (19:13):** "status: KernelWorkerStatus.ERROR | failure: | files listed: 0 | log chars 2". The CLI-downloaded log file was 2 bytes (`[]`). **Zero output files, empty log, empty failure message.**
- **Other quota (~19:00):**
  - foysalemonshanto 25.29/30 → **4.71 h left**;
  - hosen42 36.37/45 → **8.63 h left**;
  - esfersami50 exhausted.

### Kaggle discussions (17:27–17:46; 35 threads + LB history)
**Storage (751344):**
- `/kaggle/input` partly moved NFS → GCSFUSE from ~10-21. NFS 0.55 → 0.026 GiB/s.
- Reads: 1 thread ~10 MB/s, 16 threads ~107 MB/s. GCSFUSE 1.8–2.8 h per shard, and reads do not fill the page cache (Jeroen).
- Teams got 0.00 because the server did not start in time; others lost 5–10 points. CPMP had a 12 h Kaggle Error.

**Our own launch times:**
- 507–562 s on 10-19/20; **1,197 s on 10-21** (G4B rev 6d). The weight load alone was 670 s, per the research agent.
- Startup from server_args to ready, per GPT: E11-05 7 m 39 s, E11-06 7 m 30 s, rev 6d G4B 19 m 09 s.
- Builder code line 352: `VLLM_LAUNCH_TIMEOUT_SEC = 30 * 60`. Line 505: `deadline = time.perf_counter() + min(VLLM_LAUNCH_TIMEOUT_SEC, max(60, ADMIT_DEADLINE - time.perf_counter()))`.
- The assistant's reading: a timeout without error lines → no retry → LB ≈ 0. The `e11_next` relaunch logic output was cut (`[+3416]`).
- The prefetch is a single `cat`. 23.3 GB at 10 MB/s takes ~37 min, more than the 30-min timeout.

**Others' scores and methods:**
- Official Qwen3.8-27B xhigh 42.36 % at $0.447/task (≈ 85–90k tokens per attempt, per the assistant).
- mikelou1 36 → 66.39 → 69.72 (10-21), CV 79 ± 5 %.
- NguyenThanhNhan ~23 → 52.08 → 65.97 (reads ~180 GB; model undisclosed).
- Anil 45 (NVARC + Qwen ensemble); Kris CV 45 % / LB 34.31; TheDivine ~19–20 % at 300–500 tok/s.
- Hiểu Vy 26.81 (27B) / 44.03; Adarsh 15 → 21.53 (13 on rerun); Franzen 57.64.
- Rerun variance ±2–4.

**LB on 10-22:**
- Tufa Labs 88.06, Yi-Chia Chen 84.72, rabbithole 80.83, nvbanana 79.72, Majkel1337 75.56, Dawid Kopiczko 75.14 (from 70.83).
- Teams 3,096 → 3,166.

**"27B FP8 + EAGLE" explained:**
- NguyenThanhNhan's 10-05 setup: Qwen3.8-27B FP8 + its own MTP head via `--speculative-algorithm EAGLE` (steps 3, topk 1, draft 4).
- Ours: S1-60K is FP8 + MTP-3 on vLLM; E12 is INT4 + a separate DFlash2 drafter (3.85 GB).

### Deep research, 5 agents (18:19–19:30; report `reports/Qwen 27B দ্রুততম সার্ভিং সেটআপ.md`, 520 lines, 169 KB)

**Our measurements (by agent):**

| Setup | Throughput |
|---|---|
| W1: SGLang 0.5.21 TP4 + RedHat INT4 + fp8 KV + DFlash2 (block 8, unquant) + fp16 SSM + extra_buffer_lazy + mamba cache 56 + mem 0.83 + no prefill graph | E11-04 PROD 416.5 tok/s (verified at line 4944: "W1_lazy_fp16 … **416.5**"). Decode at 12 running: 500 under 10k context, 460 at 10–20k, 409 at 20–40k, 389 at 40–60k. Whole runs: E11-05 328.9, E11-06 379.9. Per stream 31–35 tok/s. |
| vLLM FP8 + MTP-3 at 16 running | 352 decode / 333.3 whole run. By running count: 44 at 1, 163 at 5, 268 at 10, 293 at 12, 352 at 16. Per stream ~22. |
| INT4, no speculation | 289 at 12, 347 at 16, 440 at 22. |

- **Acceptance:**
  - MTP-3 on vLLM: 3.0 (T 0.6), 2.67 (T 1.0), 3.45 (low effort).
  - DFlash2: 4.3–5.2. Poetiq production 3.8 by server-log median vs 4.55 per request. Falls 4.43 → 3.77 as context grows.
  - SGLang NEXTN on INT4: ~3.1.
- **Memory and admission:**
  - W1 KV 881,336 tokens (866,744 at server max 13); 1.84–2.2 GB free per GPU.
  - 11-of-12 admission bug (~−8 %); workaround server 13 / client 12.
- **Startup:**
  - SGLang cold start 520–680 s, of which 372–385 s is the verify CUDA-graph JIT.
  - vLLM init 291–295 s; prefetch cut its weight load from 64 s to ~5 s.
- **Duck agent:** KV thrash 8.39M re-prefill tokens; 187 tok/s effective vs ~420 decode.
- **Accuracy:** E11-05 11/48 vs R5 16/48. Train-pass but test-wrong: 7 of 15 vs 3 of 13. LB 29.17 vs 34.58, confounded with the P1 prompt change.

**External evidence (agents' claims, partly unverified):**
- **Correctness bugs:**
  - SGLang #36548: DFlash2 state corruption under concurrency on Qwen3.8-27B, ~90 % → ~50 %;
  - #38009: DFlash2 greedy output diverges with thinking on;
  - #43514: DFlash acceptance falls to 0 with grammar + mixed-chunk;
  - #39087: a quantized drafter gives ~0 % acceptance;
  - #42917: a stale `SGLANG_CACHE_DIR` kernel cache gave PPL 9.98 vs 6.05;
  - vLLM #55533: MTP on sm89 INT4 runs only ~3 sequences (649 vs 211).
- **Engines:** H100 SGLang 1,725 vs vLLM 1,610 (needed `--max-mamba-cache-size 256`).
- **Parallelism and all-reduce:**
  - 3090 2×TP2 = 2.40× TP4 at saturation; biMEMO TP2 518.6 vs TP4 469.4 at 8 requests;
  - forced custom all-reduce at TP4: +14.8 %, but greedy diverged on 4 of 5 prompts.
- **Hybrid-model state and kernels:**
  - mamba state slots per request: 1 / 3 / 4 / 5, plus 4 for MTP or 8 for DFlash2; the resulting cap is silent (`/get_server_info` shows 32);
  - SSM state 153.9 MB per slot in fp32, 78.4 MB in bf16; Nemotron: a straight FP16 cast raised verbosity by up to 40 %;
  - FP8 KV runs with scale 1.0. RedHat ships calibrated scales, but SGLang warned it was using 1.0.
- **Quantization:**
  - roofline INT4/FP8 1.38× at 8 streams, 1.28× at 16, 1.19× at 32;
  - RedHat INT4 AIME25 95.42 → 94.17, GPQA 89.23 → 87.88;
  - block-FP8 on L4 uses "Using default W8A8 Block FP8 kernel config … device_name=NVIDIA_L4" (untuned).
- **Storage load times:**

  | Weights | 16 threads | 1 thread |
  |---|---|---|
  | 27B INT4 + DFlash2 (23.3 GB) | ~3.6 min | ~37 min |
  | FP8 (30.9 GB) | 4.8 min | 49 min |
  | Flash-Next | 19–29 min | 3–5 h |

  Host RAM 202.6 GB.
- **Coverage arithmetic [E]:** one 60k attempt for each of 240 tasks in 12 h needs ~333 tok/s; two attempts need ~667.
- **Ranking in the report:**
  1. W1;
  1a. W1 + PR #44401, about +8 %;
  2. SGLang INT4 + NEXTN MTP 3/1/4 (297 at 8 running, 373.7 at 10), or no spec;
  3. vLLM FP8 + MTP-3;
  4. 2×TP2 INT4 (+20–100 % [E]).

  Ruled out: BF16, NVFP4/MXFP4, INT8, TensorRT-LLM, llama.cpp, EAGLE-3, n-gram, Flash-Next on L4.

### Step-time analysis from logs (19:40; Claude; derived, not profiled)
- **Step time** = (running × accept) ÷ gen tok/s:

  | Mode | 1 request | 12 requests | Marginal per request |
  |---|---|---|---|
  | No spec | 21.9 ms | 37.9 ms (50 ms at 22) | ~1.3 ms |
  | MTP | 40.3 ms | 79.9 ms (at 10) | ~4.4 ms |
  | DFlash2 | 39.9 ms | 115.5 ms | **~6.8 ms** |

- **Cost per verified token:** ~0.85–1.3 ms, which looks like an effective ~10–16 TFLOPS per GPU vs 121 peak.
- **Context cost:** 0 → 50k adds ~18 ms per step, consistent with KV reads (~16 ms).
- **All-reduce:** 64 MB took 20.9 ms → ~3 GB/s. Assumed ~128 all-reduces per step → ~0.42 ms per verified token in communication, about 40–50 ms of a 115 ms DFlash2 step [hypothesis].
- **DFlash2 width:** 8 tokens verified, ~4.3 accepted → ~46 % "wasted". 8 → 4 claimed to give ~550–650 tok/s [estimate].
- **2×TP2 (19:49):**
  - realistic 1.3–1.6× (biMEMO-derived 1.38×);
  - TP2 INT4 weights ~9.3 GB per GPU; DFlash2 KV ~0.35–0.4M per replica, i.e. ~5–6 long requests;
  - throughput → budget: 355 tok/s → ~14.4M tokens (~60k per task); 460 → ~78k; 570 → ~96k; 850 → ~144k.

### GPT bottleneck audit (20:02)
- **Logs:** E11-05 server log 3,470 lines, E11-06 1,876 lines. server_args: TP=4, DFLASH, width 8, 1 step, BF16 DFlash2 drafter, FP8 E4M3 KV.
- **Acceptance (log means):** E11-06 4.46 at 8 running (408 records), 4.37 at 12 running (278); E11-05 3.91 at 12 running (114).
- **Log line:** "CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs" (all 4 TP workers).
- **Drafter checkpoint config:** `"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "target_layer_ids": [5, 19, 33, 47, 61]`. SGLang `dflash_worker_v2.py` warns and calls `set_block_size()` when the width differs.
- **Median step at 12 running:** E11-06 124.6 ms (central 80 % 116–133); E11-05 128.4 ms.
- **Throughput model:** T8 ≈ 12 × 4.3 / 0.1155 ≈ 447 tok/s. With a constant α: α ≈ 0.811 → L4 ≈ 3.00, L6 ≈ 3.78 → a 4-width at 70 ms gives ~515 tok/s, not 550–650.
- **Weights per GPU at TP4:** INT4 4.62 GB + DFlash2 1.49 GB = 6.11 GB → TP2 ~12.22 GB.
- **Phase caps:**
  - topology + NCCL 5–8 min;
  - TP4 DFlash8 startup + profiler 15–25 min;
  - each width restart +10–20 min;
  - 2×TP2 needs a separate budget.

---

## 3. Failures, mistakes and root causes

1. **Wrong wall-time estimate and missing quota warning (assistant's error; user's major complaint 18:24, 18:59, 19:04).**
   - At 15:20 and 17:12 the assistant said "প্রায় ৬০ মিনিট" ("about 60 min"; 55–62 min including load). At 18:22 it said "তার ওপর লোডের সময়" ("plus load time"), contradicting itself. At 18:25 it admitted "আমার আগের দুটো কথা পরস্পরবিরোধী ছিল" ("my two earlier statements contradicted each other").
   - At ~17:46 it learned about the GCSFUSE slowdown, about 30 min after the push. It warned only about the load risk, **not that the run could exhaust Sami's quota** (19:01: "আমি শুধু লোড-ঝুঁকির কথা বলেছি, quota-র কথা বলিনি" — "I only mentioned the load risk, not the quota").
   - **Damage:** ~3.3 h of esfersami50 quota lost (26.87 → 30.21+). The user's plan of two 12 h runs from that account became impossible.
   - **Root causes:**
     - the estimate was based only on the previous run, ignoring the storage trend;
     - 2× quota per wall hour on 4×L4 was not stated up front;
     - Kaggle session time (quota) runs beyond the notebook's own clock (78-min hard stop from the first cell); time before the first cell and the output save are outside the notebook's control.
2. **G4B rev 6g run ended in ERROR with zero output (19:12).**
   - Empty log, 0 files, empty failure message, ~114 min.
   - The assistant's inference: CELL 0 prints at once and writes a fallback `submission.json` within minutes, and neither appeared → "খুব সম্ভবত আমাদের কোড কখনো চলতেই শুরু করেনি" ("very likely our code never started running"), i.e. stuck at the container or input-mount stage.
   - This matches others' "Kaggle Error" reports on the same day. **Not proven** (no log). At 19:01 the assistant had read every step of the notebook and found every step bounded by the 78-min hard stop or its own timeout.
3. **Old fixed 30-min first-launch timeout + single-thread `cat` prefetch** (found in this part, 17:46).
   - A server that does not start within 30 min and leaves no error lines in the log gets no retry → only the fallback is submitted → LB ≈ 0.
   - 10-21's load had already used 20 of the 30 minutes.
4. **Rev 6g metric defects, own errors found by agent 4 and agent 1:**
   - **F1:** `new_for_task` is credited in completion order, against the assistant's own H7-3 spec ("first discovery by host-verification time"). The assistant: "আমার নিজের H7-3 spec-এ verification-এর সময় ধরে দেওয়ার কথা ছিল, অথচ কোডে রানের শেষের ক্রম ব্যবহার করেছি" ("my own H7-3 spec said verification time, yet the code uses run-completion order").
   - **F2:** `gpu_hours` includes the preflight and the post-solve diagnostic (+11–15 %).
   - **H7-11 incomplete:** `run()` mutates `self.record` outside `_io_lock` (`:2461`, `:2502-2503`, `:2888-2889`), while `freeze()` reads it twice (`:2538` dumps, `:2549` deepcopy).
   - `publish_seq` is not written to disk.
   - **Older defect:** an I/O error in `write()` at `:2891` kills the worker thread, and the final `freeze()` loop raises out of `solve_all_r6`.
5. **Earlier self-audit misses.**
   - H7-6: `verify_h6.py` and `facts_size.py` crash without `tokenizers` ("আমার self-audit এটা মিস করেছিল" — "my self-audit missed this").
   - Two weak tests found by mutation: R6-P3 used a duplicate candidate; the H6-6 fake preflight was instant.
   - The assistant attributed most H7 defects to its own rev 6f code: H6-11 → H7-2, H6-14 → H7-3, H6-18 → H7-10, freeze → H7-11, the test script → H7-6. H7-1 and H7-7 were older.
6. **H6-8 / H7-9:** the H6-8 check "never runs". The queue order means branch 1 is always popped before its branch 0's next run. Agent 1 confirmed 0 deferrals in 227 runs. The defensive code is dead in production.
7. **Agent claim corrected by the assistant:** agent 1 said there is "no try around `E12A.solve_all_r6`". The assistant checked: CELL 6 has `try/except BaseException`, and the final submission write still runs. (The positions result was cut: `try: [1782, 1937, 3728] … except BaseException: [1907, 25605]`.)
8. **Research report used stale PARITY sha** (f3876b40 / 85da1be3 / bc9e2bc2). The current PARITYR6 is `82b257a5`, never run.
9. **Tooling and environment failures:**
   - API ENOTFOUND at 13:49, 16:56 and 16:59, which killed agents 3 and 5;
   - "Request timed out" at 18:44;
   - Kaggle discussion forum listing returns 403 (site-wide search works);
   - `kaggle` CLI not on PATH;
   - `ModuleNotFoundError: kagglesdk` from one cwd, which worked from `D:/DevCache/Temp`;
   - cp1252 UnicodeEncodeError without `PYTHONIOENCODING=utf-8`;
   - CRLF in the adapter broke a patch assert (CRLF 2892 → normalized to 0);
   - push connection reset on the first attempt.
10. **Misleading "2.5–3× reduction" sentence (17:21, corrected 17:24).** "এটা efficiency বৃদ্ধি নয়, বাজেট কাটা" ("this is not an efficiency gain, it is a budget cut"). The evidence at low budget was negative (budget_sim; rev 6d 2/12 at 47.7k).

---

## 4. Decisions and reasons; options rejected

**H7 fix list (14:17):**
- P1: H7-1 `publish_seq`, H7-11 atomic freeze, H7-10 nudge seen.
- P2: H7-2, H7-3, H7-7, H7-6.
- P3: H7-5, H7-12, H7-13, H7-9, H7-8, H7-4.
- Not changed: nudge wording, nudge once, H6-8, KV idle override, H6-13, H6-15, PARITY, scheduler design.

**Rev 6g = the last version (14:38):** only a confirmed score-affecting bug reopens the code. So F1, F2, R1–R4, H7-11-partial and the I/O-error path are **not fixed**; metrics are to be recomputed after the run with `verify6g/4_metrics/recompute.py`.

**Failed agents not relaunched (17:03):** "আর পরীক্ষা নয়, ব্যর্থ দুটো agent আবার চালাব না" ("no more tests; I won't rerun the two failed agents"). This followed the last-version rule.

**Commit run before submit (15:40):** proposed so a bug does not waste a 12–16 h submission slot. The user chose a commit run on Sami's account.

**PARITY:** audit 2 said NO-GO (old deadline and tool-identity bugs); audits 1 and 3 said GO (diagnostic only). It stays deferred "by your decision" (C6-1/C6-3).

**First-launch timeout 30 → 75 min (17:46):** recommended. "প্রথম launch-এর timeout ৩০ মিনিট থেকে বাড়িয়ে নোটবুকের আগে থেকে থাকা ৭৫ মিনিটের give-up সীমা পর্যন্ত" ("raise the first-launch timeout from 30 min up to the notebook's existing 75-min give-up bound"). Parallel copy to local disk was judged riskier (more code), "G4B-এর লোড-টাইম দেখে ঠিক করা ভালো" ("better decided after seeing the G4B load time"). **G4B produced no load time.** The research report then ranked "parallel copy to local disk or RAM + 75-min timeout" as step 0.

**Experiment 14 created (20:02–20:08):** `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/`, holding Notebooks, Review, Results and Notes. Its draft priorities:

| Priority | Item |
|---|---|
| P0 | Weight-load path (parallel copy to local disk or RAM; timeout 30 → 75 min) |
| P1 | Stage A: topology, small-message NCCL, TP4 baseline, profiler |
| P1 | Correctness gate (spec vs no-spec, serial and concurrent) |
| P2 | DFlash2 width 6 / 4 |
| P2 | 2×TP2, only if Stage A shows communication above ~30 % of the step |
| P3 | Confirm the winner on real ARC work |
| P3 | Other levers (Marlin, GDN, MTP vs DFlash2, #44401) |

A second 5-agent research was launched: validate GPT's claims locally (count all-reduces per step from the source; what `set_block_size()` does); NCCL P2P on Kaggle L4; DFlash2 width and correctness; 2×TP2 feasibility; profiling plan and engine levers.

**2×TP2:** the assistant advised measuring before building a two-server change: "মাপে ১.৩×-এর বেশি লাভ এলে তবেই দুই-server কোডের পরিবর্তনে যাওয়া যুক্তিসঙ্গত" ("only if the measured gain exceeds 1.3× is the two-server code change justified"). It also noted this would break the last-version rule.

**GPT's recommendation (20:02):**
- if quota is short, run only Stage A first;
- a correctness gate before any throughput arm ("কোনো throughput arm correctness gate fail করলে সেটিকে production candidate হিসেবে গ্রহণ করবেন না" — "if a throughput arm fails the correctness gate, do not accept it as a production candidate");
- width order 8 baseline → 6 → 4, with 3 optional;
- "550–650 tok/s: optimistic hypothesis, expected result নয়" ("not an expected result").

**Rejected:**
- the bug-hunt agents (the user rejected the tool use);
- an offline `tokenizers` wheel in the zip (left to the user, not built);
- new GPT audit zips after rev 6g;
- cancelling G4B mid-run (the assistant recommended waiting because "cancel করলে Kaggle কোনো আউটপুট save করে না" — "if you cancel, Kaggle saves no output").

---

## 5. Open items

1. **Is rev 6g good on GPU?** Unknown. G4B gave no data. The decision whether Duck rev 6g is "submission-worthy" is pending.
2. **First-launch timeout change (17:46, "বলবেন কি, timeout বাড়ানোর পরিবর্তনটা করব কি না?" — "will you tell me whether to make the timeout change?"):** no explicit user answer in this part. The parallel-copy plan offered at 19:30 ("আপনি বললে… পরিকল্পনা লিখে দেখাব" — "if you say so… I'll write up the plan") is also unanswered. Both were later folded into E14 P0.
3. **Asking Kaggle for quota or a rerun:** a post in discussion 751344 was offered; the user's decision is pending.
4. **Verification agents 3 and 5 never finished:**
   - notebook-cell checks: window shift, isolation retry, callback guard, coverage line, GPU-hour metric, cache-miss classes;
   - regression replay across rev 6d–6g; exit codes without `tokenizers`/`transformers`.
5. **Mutation suite not re-run on rev 6g.**
6. **Deferred rev 6g defects (post-run list):**
   - F1, F2;
   - H7-11 partial; `publish_seq` not on disk;
   - an I/O error kills the worker and the final loop raises;
   - R1 (`_arc2_autocheck` resets `_arc2_progress_names`);
   - R2 (nudge "seen" recorded before the request succeeds);
   - R3 (multi-tool rollback, 0.27 %);
   - R4 (print guard on tuple lines, 13 %).
7. **Hamid-account EVAL12R6 commit (#63 from 10-21):** never done.
8. **Second Experiment 14 research (5 agents, launched 20:05):** results not in this part. A full report with a "true / false / partial" verdict per GPT claim was promised.
9. **Questions left unanswered:**
   - #36548 in our runs: "#36548 আমাদের রানে ঘটেছে, এমন প্রমাণ নেই। আবার ঘটেনি, তারও প্রমাণ নেই।" ("no evidence it happened in our runs, and no evidence it did not");
   - whether RedHat's calibrated KV scales load;
   - whether the RedHat INT4 checkpoint contains the MTP head (check the safetensors index);
   - whether any grammar-constrained requests are sent (#43514);
   - the local scratch-disk size on the 4×L4 image;
   - whether P2P exists between the L4s;
   - the real per-step all-reduce count.
10. **Engine and pairing:**
    - the 11-of-12 root cause: the report says "source-verified chunked-prefill double count is better supported than the earlier mamba-gate guess";
    - PR #44401 not in 0.5.21, untested.
11. **Quota and time:** the weekly quota reset time is unknown. Quota left: foysal 4.71 h, hosen42 8.63 h. **11 days left** as of 10-22.
12. **Claimed but not opened:** the assistant said at 18:35 that #36548 was only the report's claim ("আমি নিজে লিংক খুলে যাচাই করিনি" — "I have not opened the link myself") and promised verification "রিপোর্ট লেখার সময়" ("while writing the report"). The final report summarizes it with tags; the transcript does not show the assistant opening the GitHub issue itself.

---

## 6. Insights a fresh reader might miss

- **Quota burns at 2× wall time on 4×L4.** Measured: ~69 min → 2.31 h, ~100 min → 3.34 h. GPT quoted the official rule. Any "X min run" costs ~2X quota hours. GPT corrected the assistant's "30–40 min ≈ 1–1.5 h quota" as valid only for a short run, not a 5-arm sweep.
- **The notebook clock does not bound the Kaggle session.** The G4B profile's 80-min budget / 78-min hard stop counts from the first cell. The session ran ~114 min and ended in ERROR with nothing saved. **Cancel also saves nothing.** Neither an in-notebook watchdog nor a user cancel recovers data if the container never reaches our code.
- **The G4B "error" is not evidence about rev 6g.** It is most likely infrastructure (no log lines at all). Rev 6g remains untested on GPU. Do not read it as a negative.
- **Contradictory GPT audits:**
  - H7-1 `publish_seq`: P0 / "এটি fix না করে পরবর্তী GPU run দেওয়া উচিত নয়" ("no GPU run before this is fixed") in audit 1, but "optional hardening" in audit 2;
  - PARITY: NO-GO in audit 2, GO in audits 1 and 3.

  The assistant reproduced the race and showed the content was identical in production.
- **"Budget cut ≠ efficiency"** (17:24). Rev 6g's 48–64k per task is a deliberate cut. Evidence at that budget is negative (budget_sim 4/12 at 50k; rev 6d G4B 2/12 at 47.7k). The bet is that rev 6e–6g bring the first transform to 15–25k tokens.
- **The agent is good where it reaches, but slow:** rev 5 LOW solves ~56 % of reached tasks at ~145k tokens per task, yet reaches only 31 %. This supports the memory note "coverage binds".
- **MTP vs DFlash2:** outside literature favors MTP at high concurrency and long thinking. Our L4 data favors DFlash2 (+39 % over no-spec at 12; acceptance 4.43 → 3.77, no collapse). No same-engine comparison at 12 running exists.
- **The SGLang lead is +17–30 % on whole runs, not 1.4–1.6×** (the short sweeps overstated it).
- **The 8 → 4 DFlash2 width idea is weaker than first claimed.** The checkpoint was trained at block 8. SGLang overrides with `set_block_size()` plus a warning, which is not a clean truncation. The constant-α model gives ~515 tok/s at 4-width, not 550–650. #36548 persisted even with a width change, per GPT.
- **Strongest new hard evidence for a communication bottleneck:** our own log line "CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs". Forcing custom all-reduce gave +14.8 % but changed greedy outputs on 4/5 prompts, so it is unusable without a parity gate.
- **The step-time "10–16 TFLOPS" and "40–50 ms all-reduce" figures are derived, not measured.** GPT warned that 64 MB bandwidth cannot be scaled down to 10 KB–1 MB message latency, and that the 64-layer × 2 all-reduce count is an assumption for this hybrid model.
- **NguyenThanhNhan's 65.97 run reads ~180 GB** → likely not the 27B (inference only). His known 27B FP8 + EAGLE setup is essentially our S1-60K engine.
- **Every reported 0.00 involved large reads** (180 GB, 110 GiB, 100.5 GiB). Our 23.3 GB is small. The risk was our fixed 30-min timeout plus single-thread prefetch, not the size. Page-cache warming does not work under GCSFUSE (Jeroen), so only a real copy (to RAM or local disk) helps.
- **Accuracy evidence against W1 is confounded:** LB 29.17 vs 34.58 (prompt change P1), and the E11-05 overfit signal (7/15 vs 3/13; the 49k cap).
- **11-of-12 running root cause, three versions:**
  - the research report: chunked-prefill double count (source-verified);
  - earlier: a mamba-gate guess;
  - the current E14 context (from later parts): the DFlash min-free-slots delayer.

  These differ; whichever is true should be checked against the real running count.
- **The H6-8 defensive deferral is unreachable** in production (0 in 227 runs), so it is harmless dead code.
- **The user cares about reliable estimates and full reports more than about the agents' speed.** Two standing rules came out of this part: full report always (18:40), and written time/quota/stop estimates (the assistant's 19:04 commitment).

---

## 7. Relevance to the current E14-A notebook (concrete fixes, guards, defaults and checks)

**A. Quota and wall-time estimate must use 2× and include time outside the notebook clock.**
- *Evidence:* esfersami50 used 2.31 h of quota in ~69 min and 3.34 h in ~100 min. Official rule (GPT): "L4x4 … দ্বিগুণ quota" ("double quota"). G4B ran ~114 min against a 78-min in-notebook hard stop.
- *What to do:*
  - the E14-A 200-min profile ≈ **6.7 quota hours** at minimum, plus container start and output save;
  - quota left: foysal **4.71 h**, hosen42 **8.63 h**, esfer **≤0**. Only hosen42 can absorb E14-A without going negative (runs started with quota still finish, per memory). E14-A on hosen42 would leave ~2 h, which affects the user's plan for 12 h runs;
  - the written estimate must give worst-case minutes (slow storage), quota per account before and after, and a **stop time**, as promised at 19:04.

**B. Self-terminate early and persist every phase result immediately; never rely on cancel.**
- *Evidence:* "cancel করলে Kaggle কোনো আউটপুট save করে না" ("if you cancel, Kaggle saves no output"); G4B ERROR → "files listed: 0", log 2 bytes.
- *What to do:*
  - the E14-A phase record and heartbeat should be written to `/kaggle/working` at first-cell time, and after every phase (atomic JSON);
  - the 200-min hard stop must be enforced inside the notebook with margin for output save, so the notebook exits cleanly;
  - define the external cancel threshold (budget + container overhead) in the written plan;
  - note that if the kernel never reaches our code (the suspected G4B case), no notebook guard helps. Only external status polling and a user stop rule do.

**C. The ranged parallel copy to `/dev/shm` addresses the confirmed failure mode. Add capacity and memory guards.**
- *Evidence:*
  - single-thread ~10.5 MB/s vs 16 threads ~107 MB/s (10-22 measurements); 23.3 GB is ~37 min single-threaded vs ~3.6 min with 16 threads;
  - GCSFUSE reads do not fill the page cache, so warming with `cat` is useless;
  - host RAM 202.6 GB, but "the local scratch-disk size on the 4×L4 image is unknown".
- *Guards:*
  - check `/dev/shm` free space ≥ 23.3 GB plus margin before copying; fail open to the `/kaggle/input` path;
  - tmpfs pages count against the container memory cgroup, so read `memory.max` / the cgroup limit and keep headroom for the SGLang host processes and the Duck sandbox workers. This is my reasoning, not from the transcript;
  - log copy MB/s per range so the next estimate uses measured bandwidth.

**D. Launch watchdog: make sure no old 30-min bound survives.**
- *Evidence:* builder line 352 `VLLM_LAUNCH_TIMEOUT_SEC = 30 * 60` and line 505 `deadline = … min(VLLM_LAUNCH_TIMEOUT_SEC, max(60, ADMIT_DEADLINE - now))`. With no error lines there is no retry → LB ≈ 0.
- *What to do:*
  - E14-A's stall watchdog with the 75-min give-up should replace both bounds;
  - check that an `ADMIT_DEADLINE`-style clamp does not shorten it silently, and that the relaunch logic (`e11_next`, whose code was cut in the transcript at `[+3416]`) does not treat "no error lines" as fatal;
  - record `engine_startup_sec` split into weight load, JIT/CUDA graph and ready. History: 507–562 s, 1,197 s (670 s weight load), E11 ~7.5 min, rev 6d G4B 19 m 09 s.

**E. JIT cache tar: key it strictly and re-verify after restore.**
- *Evidence:* the verify CUDA-graph JIT is 372–385 s of a 520–680 s cold start, so the cache is a real saving. But SGLang **#42917**: a stale kernel cache under `SGLANG_CACHE_DIR` gave PPL 9.98 vs 6.05 (silent corruption).
- *What to do:*
  - key the cache by SGLang/torch/CUDA version, GPU arch and the flags that change kernels: DFlash width, cuda-graph batch list (1..13), mamba dtype or checkpoint mode, NCCL env;
  - do not reuse one cache across arms blindly;
  - run the greedy parity check after a restored-cache launch.

**F. DFlash width 6/4 arms need extra checks.**
- *Evidence:* the checkpoint has `block_size 8`, `target_layer_ids [5,19,33,47,61]`. SGLang 0.5.21 warns and calls `set_block_size()`. #37475: missing `target_layer_ids` picks the wrong tap layers. Model: α ≈ 0.811 → expected L6 ≈ 3.78, L4 ≈ 3.00. A 4-width at 70 ms gives ~515 tok/s.
- *Guards:*
  - capture the width-override warning and the tap layers from the log;
  - log acceptance per arm and compare with 3.78 / 3.00;
  - require the parity gate before adoption;
  - judge an arm by effective output tok/s at real 12 running, not by step time.

**G. Greedy parity design.**
- *Evidence:*
  - GPT: "20–50টি fixed prompt নিয়ে greedy mode-এ target-only বনাম DFlash2 token IDs … Serial এবং concurrent দুটো mode রাখুন" ("take 20–50 fixed prompts and compare target-only vs DFlash2 token IDs in greedy mode … keep both serial and concurrent modes");
  - our MTP vs DFlash2 greedy common prefix is 152 vs 155 of 512 tokens, so drift alone cannot detect #36548/#38009;
  - forced custom all-reduce changed greedy on 4/5 prompts;
  - FP16 SSM casts can raise verbosity by up to 40 % (Nemotron).
- *What to do:*
  - E14-A's A/A + concurrency-12 + target-only design is right; add a **serial (concurrency 1)** leg if it is missing;
  - apply the same gate to the NCCL-env arm and the int8 mamba-checkpoint arm;
  - also compare output length per prompt (verbosity), not only token match.

**H. Assert real concurrency in every baseline window and arm.**
- *Evidence:* mamba slots silently cap running requests; `/get_server_info` still reports 32; the 11-of-12 admission bug (~−8 %, server 13 / client 12 workaround).
- *What to do:*
  - read `#running-req` from the TP0 "Decode batch" lines and fail or flag a window that is not at 12;
  - record the effective `max_running_requests` and mamba usage from the startup log;
  - for the `--min-free-slots-delay 1` and #44401 arms, report the running distribution (the 11-of-12 root cause is disputed: chunked-prefill double count vs min-free-slots delayer).

**I. NCCL bench specifics.**
- Run `nvidia-smi topo -m` and `nvidia-smi topo -p2p p`.
- Sweep all-reduce from 8 KB to 1 MB (latency) and re-measure 64 MB, for the TP4 group and for candidate TP2 pairs, with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET`.
- Offline only (torch.distributed if `all_reduce_perf` is absent).
- Record the "CustomAllReduceV2 is disabled …" line.
- Do not choose TP2 pairs blindly as 0–1 / 2–3.

**J. Profiler.**
- Trace CUDA-graph replay kernels (not Python outer timing).
- Classify into collectives / quantized GEMM / GDN + attention / speculative draft-accept-commit.
- Do not sum the four GPUs' kernel times for the critical path.
- Flush the trace and a JSON summary before the server process stops.
- Keep the window short (trace size goes to `/kaggle/working`).

**K. Phase time budget must include restarts.**
- *Evidence:* each SGLang restart costs ~7.5 min with a warm environment and up to ~19 min with slow storage. GPT caps: TP4 startup + profiler 15–25 min; +10–20 min per width restart.
- *What to do:* E14-A has many restarting arms (width 6, width 4, CUDA graphs, int8 mamba, NCCL env, #44401, min-free-slots). Add a per-arm deadline and skip-if-behind (fail-closed per arm), ordered by value:
  1. NCCL + baseline + profiler + parity first (GPT's "minimum viable experiment = Stage A");
  2. then the arms.

**L. Scorecard fields (GPT list).**
`engine_startup_sec, total_wall_sec, total_output_tokens, end_to_end_output_tok_per_sec, steady_state_output_tok_per_sec, median_decode_step_ms, accepted_tokens_per_verification, collective_gpu_critical_path_ms, peak_gpu_memory_gb, kv_and_mamba_usage, greedy_token_match_rate, request_failures`.

**M. Acceptance by context window.**
- *Evidence:* the long-output acceptance-collapse literature (3.7 → 1.5); our 4.43 → 3.77 with context.
- *What to do:* log accept length per 10k-token context bucket in the baselines and in the 25-min Duck run.

**N. KV scale and grammar checks (cheap log checks).**
- Record whether SGLang uses KV scale 1.0 or the RedHat calibrated scales.
- Confirm no grammar/json-schema-constrained requests are sent and mixed-chunk is off (#43514 drives DFlash acceptance to 0).

**O. The 25-minute Duck run uses rev 6g, with its known defects.**
- *Evidence:* F1 (completion-order credit), F2 (`gpu_hours` includes the preflight and diagnostic, +11–15 %), H7-11 file/frozen mismatch, `publish_seq` not on disk.
- *What to do:*
  - compute any per-GPU-hour or yield numbers over the solve window only, or run `verify6g/4_metrics/recompute.py` on the outputs;
  - an I/O error in `BranchSession.write()` kills a worker and the final freeze raises (CELL 6 try/except keeps the submission write). So **do not place Duck output directories on `/dev/shm`**, and check free disk on `/kaggle/working` before the run;
  - per-stream rate is ~31–35 tok/s, so a 60k request is ~29 min, longer than the 25-min window. Rely on the T8 per-request deadline clamp; the base request timeout is 1,800 s (`E12_DUCK_REQUEST_TIMEOUT`);
  - first caps are 12,288 tokens (~6 min).

**P. Account choice and timing.**
- The weekly reset is unknown, and Kaggle infrastructure was unstable on 10-22 (several "Kaggle Error" reports, an empty-log ERROR for us).
- The written plan should name the account, expected quota after the run, and what happens to the user's planned 12 h submissions (foysal 4.71 h, hosen42 8.63 h, 11 days left).
- Push only after the user's explicit go. The esfer G4B go was explicit: "G4B commit run করো সামি ভাইয়ের একাউন্টে" ("do the G4B commit run on Sami bhai's account").
