# 16 — 3-day conversation insights and E14-A notebook fixes (Claude, 2026-10-23)

**User request (2026-10-23 06:30 UTC):** "এর মধ্যে আমাদের গত ৩ দিনের সমস্ত কথোপথন পড়ে আসো,কোনো ইনসাইট পাও কি না? নোটবুকে কোনো ফিক্স বসাতে হবে কি না? আর গুরুত্বপূর্ণ সবকিছুই নোট করবে যাতে পরে দেখতে পারো।"
("Read all our conversations of the last 3 days. Any insights? Any fix needed in the notebook? Note everything important so you can look at it later.")

**How it was done:**
- The session transcript from 2026-10-20 to 2026-10-23 was split into 12 parts (`part_00` … `part_11`). The parts hold all user and assistant text; tool results are cut at 500 characters.
- 12 reader agents each read one part in full and wrote `insights_NN.md`. Each file has the same 7 sections:
  1. rules and decisions;
  2. measured facts;
  3. failures;
  4. decisions;
  5. open items;
  6. hidden insights;
  7. relevance to E14-A.
- I read all 12 insight files in full (449 KB).
- They are kept permanently next to this note in `Notes/Conversation insights 2026-10-20 to 23/insights_00.md … insights_11.md`. Those files hold every number and quote; this note is the consolidated index plus the notebook decisions.
- Every notebook claim below was checked against our code and against the real SGLang 0.5.21 source: `_scratch/session_1c6db7aa/sgl_src`, unpacked from the aarch64 0.5.21 wheel.

| Part | Time span (UTC) | Main content |
|---|---|---|
| 00 | 10-20 02:21–23:26 | GitHub push; E13 SMOKE1; rev 4 EVAL6 3/6; weakness analysis; rev 5 (B1–B9, R5-1..5); EVAL6R5 + EVAL12R5_LOW runs |
| 01 | 10-20 23:35–10-21 05:33 | rev 5 results; LOW submitted (57456562); 35k-per-attempt sim; pre-check analysis; code-first GPT reports; R6 plan; "past negatives ≠ verdicts" |
| 02 | 10-21 05:44–08:10 | GPT plan reviews; T8 deadline root cause; rev 6 build; run-notebook overwrite + restore |
| 03 | 10-21 08:26–09:42 | rev 6 → 6b → 6c; regex hang; gate 4b approved; 12 streams |
| 04 | 10-21 10:00–11:26 | GPT pre-GPU audit; rev 6d (items 1–5 only) |
| 05 | 10-21 11:28–12:37 | time estimate; G4B rev 6d pushed on foysal |
| 06 | 10-21 12:57–15:33 | G4B 2/19; capacity-probe bug 12→6; KV-only gate (user rule); GPT G4B audits |
| 07 | 10-21 15:58–16:10 | "you missed something"; "take all G4 items" |
| 08 | 10-21 17:41–10-22 11:34 | rev 6e audit; H6 items; rev 6f |
| 09 | 10-22 11:42–20:08 | rev 6g (last version); G4B rev 6g ERROR on esfersami50 (114 min, 0 output, quota exhausted); user anger over the estimate; GCSFUSE; serving research; E14 opened |
| 10 | 10-22 20:08–10-23 05:32 | engine bottleneck research; P0 load path; re-prefill root cause; raw-reasoning switch; E14-A plan |
| 11 | 10-23 05:50–06:30 | "one notebook"; E14 build; delayer found (11-of-12) |

---

## 1. Standing rules and decisions from the user (all still in force)

| Date (UTC) | Rule / decision (verbatim where short) | Effect |
|---|---|---|
| 10-20 17:24, 10-22 18:40, 10-22 20:02, 10-23 (three times) | "তুমি সব না দেখে শর্ট করে দ্যাখো সবসময়, এটা কখনোই করবে না" / "পূর্ণ report always" / "পুরো রিপোর্ট দিবি,তুই বাদ দিস,এটা তোর স্বভাব" | Read everything fully. **Every report lists every item.** Memory `read-everything-fully.md`, updated again today. |
| 10-20 17:50 | Fix every problem found in a run, then run again; no one-change runs | Bundle rule (memory `bundle-changes-big-jumps.md`). |
| 10-20 17:50 | Changes below 60k are allowed when there is evidence | Used for B5 (49,152 output cap). Every other cut needs approval (memory `no-token-cuts-below-base.md`). |
| 10-20 20:17, 20:52 | "this model + Duck has 80+ eval / 70+ LB with full coverage"; "trust what I saw, don't argue" | Never propose running Duck on fewer than all tasks; trust user observations. |
| 10-21 05:30 | "আগে আলাদা এক্সপেরিমেন্ট ছিলো… সেখানে কাজ করেনি bole ekhane kaj korbe na emon na" | Past negatives are context, not verdicts. |
| 10-21 09:26 | "অনুমতি দিলাম" (gate 4b); "১২ stream নাও"; "বাকি টার ও অনুমতি দিলাম" | 12 streams; gate-4b time share; rev 6 caps 12,288/24,576/32,768 + 28k first exposure approved. |
| 10-21 13:10 | "শুধু ৯৫% এর বেশি ক্যাশ থাকলে নতুন কোনো টাস্ক ডুকবে না" | KV-only admission gate (G4-1); keep code simple. |
| 10-22 14:38 | "এটাই লাস্ট ভার্সন মনে রাখ" | rev 6g is the last Duck version; no audit loops (memory `stop-audit-loops.md`). |
| 10-22 18:59–19:04 | Anger over the "~60 min" estimate (the run took ~114 min and burned esfersami50's quota); "only 11 days left" | **Before every GPU run, write down:** worst-case wall time (slow storage), quota per account at 2×, quota left after, stop time, and notebook sha. |
| 10-23 03:37 | "আলাদা আলাদা মানুষ।একটা টিম।" | foysal, hosen42 and esfersami50 are three people on one team. |
| 10-23 05:27 | "পুরো পাইপ লাইন পরিবর্তন করা লাগলেও করিও,বেস ভাঙবে এই সেই বলে আমার এক্সপেরিমেন্ট মাটি করিও না" | Proven fixes are on by default in E14; the base is preserved, so there are no base-protection excuses (memory `no-base-protection-excuses.md`). |
| 10-23 05:50 | "একটাই নোটবুক বানাও।আর সফল হলে… আর একটা ফিক্স করে ১২ ঘন্টার ২ থেকে ৩ টা রান" | One E14-A notebook. If it succeeds: one more fix, then 2–3 full 12 h runs. This **supersedes Notes/13's two-run plan (A1 95 min / A2 110 min)**. |
| always | No push / run / submit without an explicit go; no git commit without permission; never read or echo tokens; scratch on D:; chat in Bengali; never rebuild a Kaggle-run notebook; pre-run capacity check; save full thinking traces | — |

**Approval-drift lesson (parts 02, 08):** twice, a short "OK" from the user was expanded by the assistant into a set of specific options ("ওকে আমি গ্রহণ করলাম" → the whole config; "Ok update start" → four specific choices). When an approval is short, name the exact options it covers in the reply.

---

## 2. Measured facts that E14 decisions rest on

**Throughput (decode-log based):**
- W1 at 12 running:
  - <10k context ≈500 tok/s;
  - 10–20k 460;
  - 20–40k 409;
  - 40–60k 389 (E11-06).
- At 11 running, 20–40k: 370.9 tok/s (G4B). At 6 running: 321.4 tok/s.
- E11-04 PROD 416.5 tok/s.
- Whole runs: E11-05 328.9, E11-06 379.9.
- Step time at 12 running: ≈115 ms at ≤10k context, 133–140 ms at 50–80k. The log "step" includes prefill; windows that contain a prefill line have a median of 323 ms.
- Acceptance:
  - accept len includes the bonus token (accept len = 1 + 7 × rate, max deviation 0.0057 over 5,017 records);
  - per-run means 3.87 / 4.58 / 5.10;
  - falls 4.43 → 3.77 with context.

**Agent-effective throughput:**

| Run | Effective tok/s | Prefill share of the window |
|---|---|---|
| EVAL30 | 167.8 | 55.1 % |
| EVAL6 | 325 | — |
| EVAL6R5 | 244 | — |
| EVAL12R5_LOW | 265 | — |
| G4B rev 6d | 279 | 19.8 % |

- Prefill runs at ≈1,475–1,578 tok/s and blocks decode. One uncached prompt token costs ≈0.27 of a generated token.

**Prefix cache:**
- Aggregate cached fraction (Σcached / Σprompt):
  - rev 5: xhigh 84.3 %, low 88.7 %;
  - G4B: 75.0 %.
- The 85.4 / 91.0 % figures are request-level means (wrong denominator).
- A null `cached_tokens` means 0 under `--enable-cache-report` (usage_processor.py:13-15).
- The reply is lost at the end of the previous prefill in 72–91 % of requests. Root cause: Duck's `_normalize_message_content` deletes blank lines from the reasoning. Fixed by `E12_RAW_REASONING_HISTORY=1` (fake-server proof: 24/24 break → 24/24 full prefix).
- 18 deeper misses (66.8k tokens) remain. They point to Mamba checkpoint eviction.

**KV / mamba:**
- KV pool 881,336 (W1, E11-04) vs 866,744 measured at server max 13. The window is read from the server at runtime.
- Mamba 56 slots, 4 per request (13 × 4 = 52 ≤ 56).
- Mamba usage max 0.70–0.71 in every log.

**11-of-12 running:**
- E11-05 has 2,358 decode lines at 11 running + 1 queued against only 114 at 12. This costs ≈8–9 % (EVAL30 422 vs 462; G4B 397 vs 436).
- **Cause (source-verified today, not a guess):** the DFlash min-free-slots delayer.
  - `managers/min_free_slots_delayer.py`: `resolve_min_free_slots` → unset + DFlash + mrr ≥ 8 ⇒ `min(4, max(2, (mrr+5)//6))` = 3 at mrr 13. An explicit value wins, and **"threshold > 1 else None" ⇒ `--min-free-slots-delay 1` disables it**. The flag is defined in `arg_groups/fields/schedule.py:267` ("1 disables").
  - The delay is skipped while a chunked prefill is in progress (`scheduler.py:3852-3862`).
- Earlier attributions (the #44401 double count, a mamba gate) are superseded.

**Startup / platform:**
- `/kaggle/input` is partly on GCSFUSE since 10-21: 1 thread ≈10.5 MB/s, 16 threads ≈107 MB/s.
- Weight load: 12.8–58.7 s on 10-19/20, but 669 s on 10-21 (ready in 1,197 s).
- The first bs=12 verify-graph capture is ≈362 s of silence. A warm relaunch in the same session takes ≈12–14 s.
- 4×L4 bills quota at 2× wall time (esfersami50: 2.31 h in ≈69 min, 3.34 h in ≈100 min).
- Cancelling a run saves no output.

**Quota (last readings, 2026-10-22 ≈19:00 UTC; must be re-checked):**
- hosen42: 8.63 h left;
- foysal: 4.71 h;
- esfersami50: exhausted (30.21/30).
- Deadline 2026-11-02. About 1 submission per 2 days; 12–16 h turnaround.

**Scores (context):**
- LB: S1-60K 34.58 (best); S1-55K 31.81; E11-06 29.17; EVAL12R5_LOW (Duck) 17.50 (ref 57456562).
- Duck rev 5 solves ≈56 % of the tasks it reaches but reaches only ≈31 % of 240.
- G4B rev 6d: 2/19 at ≈47.7k tokens/task.

---

## 3. Failures and their lessons (classes that repeat)

1. **Fake servers test my assumptions, not reality.** The same lesson appeared five times:
   - capacity probe 12→6 (the fake had queue = 0);
   - blank-line normalization (the transcript stores normalized text, so the CPU diff compared normalized with normalized);
   - patch API key `"patches"` vs `"results"`;
   - E11-03's running cap;
   - **today: `/start_profile` returns plain text, not JSON** (http_server.py:1164-1172).

   **Rule:** every SGLang endpoint and field the notebook uses is checked against the 0.5.21 source. Today's checks are in section 4.
2. **Timeouts computed once per turn or phase overrun the deadline by a whole timeout** (T8: 679 s; PARITY: up to 6 h). Every HTTP call is clamped to the absolute deadline.
3. **Clock mixing:** relative vs absolute (the 1,412 s → 679 s error). One clock, `time.perf_counter`, everywhere.
4. **Statistic traps:**
   - request-level vs aggregate cache;
   - censored medians (first check);
   - mixing context bins (the 13.3 % vs "15–20 %" error);
   - admission-transient windows ("12 running is slower", withdrawn);
   - 29-line samples.
5. **Fail-open preflights** (isolation `not leaks` / `None`). Use three states; only PASS keeps the aggressive setting.
6. **Builder hygiene:**
   - docstrings inside `"""` cell strings (twice);
   - CRLF edits;
   - an overwritten Kaggle-run notebook (twice; `_REBUILD` guard);
   - quoting bugs;
   - **today: a literal CR character inside the new compile/CR-guard line broke the build.** Harness run 2 therefore ran on the previous build.
7. **Estimates:** the "~60 min" G4B estimate ignored the storage trend and 2× quota, and contradicted itself. The written worst-case sheet is now mandatory.
8. **Missing what the user actually asked** ("you missed something", 10-21 15:58; the gap check). Answer the question asked, item by item.

---

## 4. Notebook check list (every E14-relevant item from parts 00–11)

### 4a. Fixes applied (this session)

| # | Fix | Where | Evidence that made it necessary |
|---|---|---|---|
| 1 | Every HTTP request is capped by the phase deadline | `e14_measure.post_json` | T8, PARITY C6-1 (parts 02, 04, 06) |
| 2 | Parity requests still alive at the deadline are aborted | `parity_pass` | C6-1 |
| 3 | `wait_level` plateau: a window measures at the level the server really runs (11 under the delayer), recorded as `measured_level` / `real_running_reached` | `steady_window` | The delayer holds W1 at 11 of 12; without this every W1 window would be invalid and up to ~50 min lost |
| 4 | Concurrency-12 A/A (`aa_c12`) is the noise floor for the contamination decision; large-margin flips (> 1 nat) decide | CELL 5b parity | 152–166 of 512 tokens shared under batching even A/A (parts 01–08) |
| 5 | Every generated cell is compiled and checked for CR before the .ipynb is written | builder | Docstring and quoting traps (parts 02, 03) |
| 6 | Profiler request with `with_stack: false`, `record_shapes: false` | `profile_steps` | Defaults are true in 0.5.21 (tokenizer_control_mixin.py:403-410) |
| 7 | NCCL arm reads the bench's `adopt_candidate` (≥ 20 % faster per call at 983,040 B) | CELL 6b | Bench API |
| 8 | **nodelay arm** `--min-free-slots-delay 1`, kept only if the latch test + speed rule pass | CELL 5b | Delayer (source above) |
| 9 | **NEW today: `post_json` accepts plain-text replies** (`{"text": …}`) | `e14_measure.post_json` | `/start_profile` returns "Start profiling.\n". **Without this the GPU profiler phase fails with JSONDecodeError.** Caught by harness run 2 (A1-2 FAIL) |
| 10 | **NEW today: per-phase server-log signs.** Counts of "runs out of memory", "Not enough space", "Retract", "out of memory", "Traceback", "Exception" in the log written during each phase | `e14_phase` | Open checks for the nodelay arm (over-admission, mamba space); EVAL30 thrash |
| 11 | **NEW today: `/flush_cache` before the Duck run**, so it starts from an empty radix cache like the hidden run. The result is recorded | CELL 5b end | Probe prompts (including the Duck system prompt + tools in A1-3) would add cache hits. The endpoint is verified in 0.5.21 (http_server.py:987) |
| 12 | **NEW today: graphs13 is adopted only if the server really captured 12 and 13** (`verify_graph_bs`, when parsed); recorded as `captured_bs` / `covers_12_13` | CELL 5b | E11-06 resolved list `[1..8, 10, 12]` with mrr 13 (parts 04–07, 11) |
| 13 | **NEW today: builder literal-CR bug fixed** (`assert "\r"` was written with a real CR) | builder line 688 | The build was failing; harness run 2 had used the previous build |
| 14 | **NEW today: watchdog stats report peaks** (`rchar_mb`, `faults` = max seen). One unreadable `/proc/<pid>` sample, e.g. a worker exiting, no longer shrinks the reported total. The stall decision is unchanged | `e14_loadpath.StallMonitor.wait` | Found while chasing a flaky test (last sample read 22 faults) |
| 15 | Fake server gained `/flush_cache` (plain text, refused while busy, like 0.5.21). The harness has 2 new checks: the radix flush before the Duck run; graphs13 read the captured list and every phase recorded `server_log_signs` | `src/harness` | Makes the harness exercise today's guards |
| 16 | Test robustness, no module behaviour change. (a) Atomic fake `/proc` writes (`_put`; `write_text` truncation raced the monitor; on a stuck replace it falls back to a plain write instead of killing the writer thread). (b) Monitor tests are event-based: ready = writer finished (≥ 3 s) > stall limit 2 s, so they still fail if the counter is not counted as progress; long-load test ready = 0.3 s after the writer. (c) The slow-copy test has a 5 s stall limit. (d) The parallel-readers test compares `copy_sec`, not total `sec` (which includes verify). (e) The ramp-saturation test uses 40 MiB measurement windows (12 MiB ≈ 0.24 s against the 0.2 s poll let jitter fake a > 15 % gain). Root cause of the flakes: file writes on this Windows host take tens of ms (probably the AV scanner), so fixed-sleep pacing drifted | `src/test_e14_loadpath.py` | Failing tests changed from run to run (122/125 twice) |

### 4b. Checked against code / source, no change needed

| Item (raised in part) | Finding |
|---|---|
| Token-id field for the E14-P replay (11) | `return_token_ids` exists (protocol.py:893). The response carries choice `prompt_token_ids` / `response_token_ids` (protocol.py:1228-1229; serving_chat.py:2383-2418); non-streaming only, and our requests do not stream. `replay_conversation` reads exactly these. |
| Null cached = 0, aggregate cache % (06, 07, 08) | `int(det.get("cached_tokens") or 0)` with `det = {}` when null; `agent_replay` reports Σprompt and Σuncached. |
| KV pool read at runtime (06, 08) | `e14_launch` sets `ENGINE_INFO["kv_tokens"]` from each relaunch's caps, so the Duck window follows the last arm. |
| JIT cache (10, 11) | `SGLANG_JIT_CACHE_DIR` is set explicitly; the tar never ships `sglang/nv` or `cuda` (`_NEVER_SHIP`); restore needs an exact fingerprint + known-answer check. |
| Two KV gates (08) | E11's `SGL_KV_ADMIT_MAX` 0.85 feeds only the DIRECT solver (`E10_KV_ADMIT_MAX`). Duck uses `E12_R6_KV_ADMIT` 0.95 on the exact `sglang:full_token_usage` (build_e12_duck.py:851-862), with no `token_usage` fallback. |
| Duck window start (08) | `_end = min(_end, now + E12_SOLVE_WINDOW_SEC)` at solve time (build_e12_duck.py:658-659), i.e. after all A1 phases. |
| vLLM fallback wasting quota (00) | The Duck notebook has no vLLM engine path. A tokenizer-check failure sets `ENGINE_BACKEND="vllm"`, so the SGLang engine is skipped and the solver fails closed (CELL 0 fallback submission). No 31 GB FP8 load is possible. |
| Stale `.pyc` after the backport (11) | `e14_sglang_patches` deletes `__pycache__/<stem>.*.pyc`. |
| Latch decision under the real delayer (11) | With 11 decodes at mrr 13, arrivals wait (allocatable 2 < 3), so W1/#44401 give share_at_target 0 and stuck_one_below 0 (stuck at 11, not 12). Nodelay reaches 13. The `>=` / `<=` rule therefore keeps a no-change #44401 (it is not rejected for "no gain") and lets nodelay pass. |
| Relaunch wait floor `max(60, …)` (04) | Launches are skipped with < 15 min left, so the 60 s floor cannot pass the hard stop. |
| Stall watchdog (09, 10, 11) | Progress = log growth + rchar + 4096 × page faults (mmap load); compile stall 720 s > the 362 s silent capture. |
| `ARC2_RESUME_TRIMMED` wording (07) | It replaces only the R5 "nothing submitted" resume, so it is correct. |
| rev 6g safety in the embedded adapter (02–09) | `arc2_duck.py` sha 758bddb5 contains `_GRID_ROW_MAX` (regex guard), `join_until` (one join cutoff), `isolation_state` (3-state), `kv_admit_limit` (KV-only gate), `harness_note` (H6-1 nudge), `deepcopy` (published records), and per-request `deadline - time` clamps. |
| Sampling (00, 09) | Parity / replay set T 0 / T 1.0 explicitly in their own payloads; there are no penalties anywhere (penalties are a no-op under DFlash in 0.5.21, #28180). No `tool_choice` required/named, no `--enable-mixed-chunk`, no ReplaySSM, no PCIe-IPC flag. |
| sitecustomize `wrapt` error lines (00) | The backport edits source files (no sitecustomize/.pth hook), so that error cannot disable it. `check()` runs after `apply()`. |

### 4c. Only the GPU run can answer (the run records them)

1. Real running == 12/13 with nodelay; any over-admission or mamba-space errors (now counted per phase).
2. Whether the #44401/#41872 hunks match the x86_64 wheel. The reference is aarch64; on a mismatch the patch is refused and the run continues unpatched.
3. Re-prefill fix at token level on the real server: A1-3 replay `first_divergence`, the Duck run's cached fraction vs G4B 75.0 %.
4. The step split (profiler) and the NCCL transport (bench + `NCCL_DEBUG`).
5. Acceptance at widths 6 and 4.
6. Mamba eviction with and without int8 checkpoints (agent replay classes).
7. `/dev/shm` size, cgroup limit, copy MB/s, ready seconds vs 1,197 s.
8. Whether `/flush_cache` succeeds with the hybrid radix cache (recorded either way).

### 4d. Raised but deliberately not built (reason)

| Item | Reason |
|---|---|
| Long (12,288) thinking-on parity probes and tool-name comparison (PARITY C6-3, Q2) | E14 parity compares exact greedy token ids + top-2 margins on `/generate`, which is stricter than tool name + arguments. Long probes would cost minutes per prompt inside a 200-min budget. |
| A context-matched replay for every arm | Each arm has the same two bands (8k short, ~40k long) with `ctx_per_req_p50` in the record. A per-arm 60k band is not affordable in one notebook. |
| Legacy DIRECT config fields in the E12 report (`SAMPLING temperature 0.6`, `MAX_MODEL_LEN 106496`, `THINK_BUDGET 60000`) | Cosmetic; rev 6g is frozen. The Duck run's real settings are in the `e12`/`rev6` config block. **Read the report with this in mind.** |
| rev 6g metric defects F1 (credit order), F2 (`gpu_hours` includes preflight/diagnostic), H7-11 partial, `publish_seq` not on disk | Measurement-only; the rev 6g freeze applies. Recompute from raw files if needed. |
| `--max-mamba-cache-size 64`, mrr 14 | Only if nodelay does not reach 12; this needs a memory check first. |

### 4e. Plan reconciliation

- Notes/13 planned two runs: A1 ≈61 min, HARD_STOP 95, plus "not ready at 45 min → stop"; and A2 ≈80–93 min, HARD_STOP 110.
- The user's 05:50 decision ("একটাই নোটবুক") replaced that plan with **one E14A profile: `time_budget_sec` 200 min, phases skipped (never squeezed) when they don't fit before the hard stop, first launch up to the 75-min give-up (stall rule).**
- The pre-run sheet must state these numbers, not the Notes/13 ones.
- Quota: 200 min × 2 = **≈6.7 h** plus container start/save. Only hosen42 (8.63 h at the last reading) can hold it; that would leave ≈1.9 h.
- **Risk to state to the user:** if Kaggle never starts our code (the 10-22 G4B case), quota burns with no output; nothing inside the notebook can prevent that. The external stop rule is a status check at fixed times.

---

## 5. Open items carried forward

1. Pre-run sheet for E14-A (account, fresh quota check, normal/worst wall time, 2× quota, stop time, sha). **No push without the user's go.**
2. After E14-A: the user's "one more fix", then 2–3 twelve-hour runs. Which fix is not specified; decide from E14-A results.
3. LB of other Duck variants: rev 6g has never been measured on GPU (the G4B rev 6g ERROR was infrastructure, empty log).
4. Unverified upstream DFlash issues, kept as risks: #36548, #38009, #43514, #37475, #39087. E14's parity + target-only arms are the local check.
5. `smoke_rev6` is stale (1 FAIL + a `min_workers` TypeError on old code too). Not a notebook issue.
6. Weekly quota reset times are unknown.
7. The EVAL30 `gen_throughput 0.0` with 11 running + 8 queued is still unexplained (likely KV thrash). The E14 per-phase log signs + the Duck heartbeat make a repeat visible.

---

## 6. Test record for this round

- Harness run 2 (previous build, scale 0.1, `FAKE_SPEED` 300): **23/24 ok**, 507 s. The FAIL was A1-2 profiler: `start_profile: JSONDecodeError`. This is the real-API bug fixed as 4a #9.
- Harness run 2 decisions on the fake server:
  - dflash_contamination ok;
  - p44401 not adopted;
  - nodelay not adopted;
  - graphs13 adopted;
  - int8_mamba not adopted;
  - dflash_vs_target ok;
  - width6 and width4 not adopted (0.781×, 0.562× — fake speeds).

  **The fake server does not model the delayer**, so its latch numbers mean nothing for nodelay. Only the plumbing is tested.
- **Harness run 3** (build `4c3540df`, with 4a #9–#13): **24/24 ok**, 882 s.
  - The profiler phase passed; the plain-text reply was parsed.
  - graphs13 `captured_bs` = [1..13], `covers_12_13` true.
  - `/flush_cache` gave 404 on the fake server. The fake did not have the endpoint, so it was added (4a #15).
- **Harness run 4** (same notebook `4c3540df`, fake `/flush_cache` + 2 new checks): **26/26 ok**, 609 s.
- **Final build `a9dad9fcbe0003867d34f205732a23106dbfed30380535ef4531ba78806de494`** (713 kB; loadpath `48908f52`, measure `cf7c21de`, patches `98da9521`, nccl bench `e820a514`, trace `8d89ca10`, arc2_duck `758bddb5`). **Harness run 5: 26/26 ok**, 443 s.
- **Unit tests:**
  - First full run while the harness ran: 122/125. The 3 FAILs were the monitor and slow-copy timing tests (root causes in 4a #16).
  - After the test fixes: the 4 affected tests passed 10/10 runs in a row.
  - Full suite (loadpath 56 + patches 36 + nccl/trace 33) **125/125 three times in a row** while harness run 5 ran in parallel.
  - A later fresh run gave 122/125 again, with different tests: long-load, rchar, parallel readers. One more run failed the ramp-saturation test. After 4a #16 (b)–(e): ramp tests 12/12; **full suite 125/125 five times in a row** (logs in the GPT zip).
  - The tests are not embedded in the notebook, so the notebook sha stays `a9dad9fc`.
- `analysis/test_raw_reasoning.py`: 15/15 PASS (e13_venv; e14_venv lacks `requests`).
- E12 notebooks are unchanged after every build: `26fff41e 216c59ed 74a220de 1bdadd82 5d2bc39f b65fb9dc 415a68f2 53a66856 f4fed1ac cce9cb8f 8c82ff39 82b257a5 0fea4d0c 298852aa 0b152164 e85b69c2`.
