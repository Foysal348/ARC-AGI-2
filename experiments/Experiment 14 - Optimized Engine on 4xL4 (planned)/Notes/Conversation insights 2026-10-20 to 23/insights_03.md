# Insights from part_03 (2026-10-21, 08:26 to 09:42; E12 Duck rev 6 -> 6b -> 6c build)

Source: `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/convo/part_03.txt`. I read all 3,451 lines in order.

**Coverage.** About 1,750 of the lines are two IDE selections the user attached, both of the whole "E12 next fix list (running, 2026-10-20).md":
- lines 367-1356: the rev 6 state;
- lines 1997-3148: the rev 6b state, which repeats the first one and adds "Rev 6 implementation" and "Rev 6b".

The facts in those selections are taken as written in the doc. Each is dated by its doc section (2026-10-21) or by the message that carried it.

**Time zone.** The `### USER/ASSISTANT` timestamps appear to be UTC. PowerShell `Get-Date` printed "Wednesday, October 21, 2026 2:23:13 PM" right after the 08:26-block tool calls, and file mtimes read 13:59 and 14:12. So local time is UTC+6.

**Cut results that matter (I did not guess their content):**
- (a) L274/L3215: the builder comment on `--stream-interval 1` is cut. It reads: "rule is len(output_ids) % N == 1 (output_streamer.py:427-436); DFLASH accepts several tokens per step, so a …". The reason the interval must be 1 under DFlash is in the lost part.
- (b) L273-281: a Python 3.14 `pathlib.read_text` traceback while editing the fix list. The exception type and message are cut.
- (c) L1866: the end of the harness "E12 rev 6 metrics" line, which includes `verify_skipped_dead…`.
- (d) L3353: the harness engine-info JSON.
- (e) L3360: the "E12 parity" summary line. It starts `{"probes": 8, "c…` and the rest is cut.

---

## 1. User instructions, rules, preferences and decisions

| When | Verbatim (Bengali / as written) | English meaning / effect |
|---|---|---|
| 2026-10-21 08:41 | "সবগুলো ফিক্স নিয়েছো কি না দেখে আসো,মিলিয়ে নিও।না নিলে আবার আপডেট করবে।এরপর ভ্যালিডেট করবে।পরে জিপিটির জন্য জিপ করবে।" | "Go check whether you took every fix and cross-match them. If not, update again. Then validate. Then zip it for GPT." The order is fixed: audit against the problem→fix map, then update, then validate, then a GPT zip. This produced rev 6b. |
| 08:41 (pasted) | The user's own Bengali "Rev 6-এর প্রতিটা fix কোন মাপা সমস্যার সমাধান করবে" map (sections ক-ঝ). Its closing line: "চাইলে এই তালিকাটা fix list-এ 'Problem → fix map' নামে যোগ করে দিতে পারি।" | "Which measured problem each rev 6 fix solves". This became the checklist that every rev 6b row was verified against. It includes "মোট চিন্তা ৬০k বা তার বেশিই থাকবে। আপনার অনুমোদন লাগবে" ("total thinking stays ≥60k; your approval is needed" for R6-Q1 caps). |
| 09:26 | Quoting the assistant's line "Gate 4b (hidden-এর সমান বাজেটে, প্রতি task-এ প্রায় 43k token দিয়ে eval): এতে চিন্তা 60k-এর নিচে নামে, তাই আপনার অনুমতি ছাড়া করিনি।" and then: **"অনুমতি দিলাম।"** | "Permission granted" for gate 4b: an eval at the hidden-equivalent compute share, about 43k tokens per task, which is below the 60k floor. The assistant wrote it to memory: "gate-4b time-share profile approved 2026-10-21". |
| 09:26 | **"১২  stream- নাও"** | "Take 12 streams." This overrides the rev 6 design of 10 qualified streams (R6-S5), which both GPT plan reviews had recommended. |
| 09:26 | **"বাকি টার ও অনুমতি দিলাম"** | "I give permission for the rest too." It covers the other items the assistant had held back: the p90 last-admission cutoff, target-only DFlash parity, and engine-setting changes (mamba slots, CUDA-graph list). |

Standing rules recorded in the fix list, quoted in the user selections:
- **User, 2026-10-20:** "Read every GPT report in full and validate every claim. Add the validated items here. Change no code until the user says so."
- **User, 2026-10-21 (correction):** "Those were different experiments. That something did not work there does not mean it will not work here." Past negatives (V3.1/EVO12, PIV forced close, DIRECT pause/resume) are "context notes, not verdicts". They are used only to choose safeguards.
- **User, 2026-10-21:** "Duck runs on every task (full coverage)". F7, "Duck as complement to Poetiq", is superseded by this.
- **User, 2026-10-21:** "The user approved evidence-based cuts". This made B5, the output cap 60,000 → 49,152, possible.
- **User, 2026-10-21:** "take them, validate again". This applied R5-1..R5-5.
- **User, 2026-10-21:** "the same notebook, change only the xhigh agent"; the user chose low. The one that ran is EVAL12R5_LOW 26fff41e, not EVAL6R5_LOW cce9cb8f.
- **User, 2026-10-21:** "re-read every GPT and agent recommendation, something important may be missing". This produced the "Rev 6 gap check".
- **User, 2026-10-21:** "check every fix was taken, update, validate, then zip". This is the Rev 6b heading.
- **Approval history:** "12,288 = approved exception to the 60k floor, user 2026-10-19" applied only to rev 1 / EVAL30. Rev 6 therefore needed a new decision, which came at 09:26 ("বাকি টার ও অনুমতি দিলাম").
- **Assistant practice that holds through this part:** no GPU run, no push, no submit without the user's go. Every report ends with a variant of "GPU রান … শুরু হবে আপনার go পেলে" ("GPU runs will start when you give the go").

## 2. Measured facts and results

**Engine and serving (W1, from the notebook cells, 09:27-09:32):**
- `W1 = {"strategy": "extra_buffer_lazy", "mamba_cache": 56, "no_prefill_graph": True, "mem": "0.83"}`.
- Flags: `--tp-size 4 --kv-cache-dtype fp8_e4m3 --context-length SGL_CONTEXT_LEN --max-running-requests SGL_SERVER_MAX_RUNNING`, `--speculative-algorithm DFLASH --speculative-draft-model-quantization unquant --speculative-num-draft-tokens 8`, and the E12 additions `--incremental-streaming-output --stream-interval 1`.
- KV pool: 881,336 tokens. Harness: "window 79,320 = 0.9 × 881,336 / 10". The builder falls back to `ENGINE_INFO.get("kv_tokens") or 866744`.
- Server context 155,648; Duck window 131,072. That leaves about 24k of margin.
- CUDA-graph batch list: 1-8, 10, 12. Batches of 9 and 11 pad up.
- Real running count comes from `e11_decode_summary()`, which parses `#running-req` and builds a `running_hist`.

**Throughput and timing:**
- Rev 5 low: 1.744M generated tokens in 6,570 s at 6 streams, about 265 tok/s. The GPT audit range is 244-265 tok/s.
- Effective budget under full coverage: 11.3 h × 3,600 × 255 / 240 ≈ 43k tokens per task. Reaching 55-72k per task needs 330-430 tok/s.
- Decode at 40-60k context is about 25 % slower: 323 vs 429 tok/s. DFlash acceptance drops 4.43 → 3.77 as context grows.
- 6 → 12 streams gave +17 % in one measurement and +30-60 % in a short sweep.
- Cold prefill is about 1,400-1,650 tok/s (E11), so a 50k resume costs about 30-35 s.
- Rev 5 per-request medians:

| Measure | xhigh | low |
|---|---|---|
| Median completion | 1,115 | 1,090 |
| Median prompt | 51.9k | 41.6k |
| Cached | 85.4 % | 91.0 % |
| Uncached | ~888k | ~1.577M |

- Uncached prefill already exceeds generated tokens (xhigh 0.97M vs 0.85M).
- E11-06 prefix-cache evidence:
  - stable order raised the cache from 11-18 % to 87 %;
  - an abort that does not wait for the stream to end gives 0 % (52-99.7 % when it waits);
  - a request without tools shares only 15 % of the prefix (99 % with tools);
  - a template conditional dropped the cache from 75 % to 45 %;
  - `hash()` ordering once changed an LB (hosen42).

**Rev 5 run reports (L227-238):**
- EVAL12R5_low: 24 branches, 1,743,889 generated, max branch 114,538, 6 streams; `started_sec` 799, `deadline_in_sec` 9000, 17 of 24 accepted.
- EVAL6R5 xhigh: 12 branches, 846,080 generated, max 108,897.
- EVAL6 rev 4 xhigh: 12 branches, 1,217,904 generated, max 159,464.
- Model path: `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1`.
- Scores:
  - EVAL6R5 xhigh: 4/6 tasks, 5/8 inputs;
  - EVAL12R5_LOW: 7/12 tasks, 10/19 inputs, submitted as **ref 57456562 (pending)**;
  - EVAL30R5: not run.
- Identical code gave 2, 2, 3 and 1 of 22 on repeats, so EVAL6/12 accuracy is a sanity check only.
- Selective branch-1 replay:

| Run | Score | Tokens with selective branch 1 | Tokens actually used | Saving |
|---|---|---|---|---|
| xhigh | 5/8 (unchanged) | 498,253 | 846,080 | −41.1 % |
| low | 9/19 instead of 10/19 (lost 135a2760) | 1,094,372 | 1,743,889 | −37.2 % |

- First check: median 55.1k (xhigh) / 53.7k (low).
- Hand-count misreads appear in 35 of 36 branches.
- Late corrections:
  - 135a2760_b1: 71,306 → 97,972;
  - 898e7135_b0: 56,868 → 73,523;
  - 898e7135_b1: 77,859 → 79,662.

**EVAL30 deadline overrun (T8, corrected):**
- The overrun was **679 s past the solve deadline, not 1,412 s**, and 379 s past the 14,400 s total budget.
- At 14,436 s a request timed out with "read timeout = 419.6 s", so it started before the deadline but ran 336 s past it. Another did the same at 14,492 s.
- At 14,247 s the server showed `num_running_reqs` 11, `num_queue_reqs` 8 and `gen_throughput` 0.0. This was never explained (likely KV thrash: EVAL30 re-prefilled 8.4M tokens).
- EVAL30 had **40** request timeouts in 21 of 31 branches. The earlier figure "4" was wrong.

**SGLang 0.5.21 penalties.** A static read shows presence, frequency and repetition penalties are effectively inactive under DFlash2: `dflash_info_v2.prepare_for_decode` never calls `cumulate_penalty_output_tokens` (= SGLang #28180).

**Facts-block size (Qwen tokenizer, 120 eval tasks):**

| Version | p50 | p90 | p99 | max | > 800 tokens |
|---|---|---|---|---|---|
| Full | 1,237 | 1,752 | 2,861 | 2,892 | — |
| Bounded, first | 1,237 | 1,752 | 2,084 | 2,395 | 117 |
| Bounded, after level edit 1 | 738 | 890 | 1,451 | 1,480 | 29 |
| Shipped (4 levels) | 736 | 837 | 1,362 | 1,396 | 18 (0 > 1,500) |
| Raw grid text (rev 5 prompt) | 4,881 | 8,950 | 15,828 | 16,191 | — |

- The shipped block is identical under PYTHONHASHSEED 0, 1 and 12345 (hash `dece680c454a12b3`).

**Test results by build:**

| Build | Notebook sha | smoke_rev6 | smoke_rev5 | flag-off | Harness | Replay |
|---|---|---|---|---|---|---|
| Rev 6 (08:38) | EVAL12R6 `cef11ad3` (first `9d2ee2fb` before the LF fix) | 68/68 | 56/56 | 41/41 | r6 37/37, rerun 22/22, launchfail 6/6 | xhigh 5/8, low 10/19 (same as rev 5) |
| Rev 6b (09:20) | `fe51d1d4` | 91/91 | 56/56 | 41/41 | r6 41/41, rerun 22/22, launchfail 6/6 | rev 4 EVAL6 7/8 (4c3d4a41 still 2/2), rev 5 5/8 and 10/19 |
| Rev 6c (09:42) | EVAL12R6 `7d69e502`, EVAL12R6G4B `f70c3228`, PARITYR6 `f3876b40` | 93/93 | 56/56 | 41/41 | r6 43/43, rerun 22/22, launchfail 6/6; G4B 44/44; PARITY 19/19 | — |

- Rev 6b fake-server prefix check: 84 ok, 0 breaks, 12 expected changes after a cut reply.
- Fair share in the simulator: 30/30 tasks get stage-2 work (24k-72k each) vs 3-4 tasks without it, where one task took 792k.
- PARITY harness: the A/A control matched 8/8, and a deliberate fake difference was caught (6/8).
- Gate 4b harness: `{"hidden_solve_end_sec": 42720.0, "sec_per_task": 177.9, "tasks": 12, "window_sec": 2135}`.
- 12-stream window: 73,444 tokens per stream (= KV / 12, fraction 1.0). History budget is 48.4k at a 24,576 cap and 60.6k at 12,288.

**Quota and other:**
- hosen42 check-only: quota used 30.79 of 60 h (31.29 while EVAL6R5 ran).
- Harness Python is `_scratch/e13_venv` (transformers 4.57.6) because the C: user-site transformers was deleted. tokenizers is 0.23.2.
- Verification cost in the real sandbox: 0.75-2.71 s each, at most 3 per branch.

## 3. Failures, mistakes and root causes

1. **Regex hang (critical; 08:26).**
   - `_GRID_ROW_RE` had overlapping separators (`\s*,?\s+\d+|\s*,\s*\d+`), so a long `print(grid)` line backtracked exponentially.
   - smoke_rev6 hung for more than 11 minutes (the 600 s tool timeout moved it to the background). Found with faulthandler. On Kaggle it would have frozen an agent thread.
   - Fix: disjoint separators `(?:\s*,\s*|\s+)\d+`, and rows longer than 600 characters skip the regex. The new pattern takes under 6 ms on a 10k-character line. A linear-time test was added.
2. **Wrong opener after a rejected submit.** A rejected submit counts as an executed step, so the next turn got the first-turn "Solve the task…" text. Fix: a new state `rejected`, "Your submission `t1` was not accepted: …".
3. **The caps metric counted trims, not requests.** It moved to `_chat_completion`, one count per request.
4. **Line endings.** The assistant's Python edits turned `arc2_duck.py` into CRLF (130,287 → 128,125 bytes after restoring LF). The notebook sha changed from 9d2ee2fb to cef11ad3.
5. **SyntaxError in a smoke test.** A heredoc-inserted string with a literal newline broke smoke_rev6.py line 125.
6. **Rev 6b defect (assistant's own): two workers drove the same Duck agent.** A paused branch was re-queued while still running, which is a data race. The negative control gives 24 violations without the fix and 0 with it.
7. **Rev 6b defect (assistant's own): fair-share starvation.** Paused items kept their ORDER2 rank, so the first tasks won every round. Fixed with round robin. Before the fix, S3 failed with (3, 4) tasks receiving stage-2 work.
8. **Prefix check failure in the r6 harness.** It reported 40/41 with 12 "breaks". The cause was a cut reply where only the broken call is dropped, which is expected. The fake-server check was refined.
9. **Same trap twice in the builder (09:32).** A docstring inside a builder `"""` string gave "SyntaxError: leading zeros in decimal integer literals" in all three variants ("আগের সেই একই ফাঁদ", "the same old trap"). It was changed to a comment.
10. **Bash heredoc quoting failures** ("unexpected EOF"). The fix was to write patch scripts to files.
11. **Simulator debug crash.** `dbg_fair.py` failed with ModuleNotFoundError 'inference' because `_import_duck()` was not called first.
12. **Rev 6 notebook header carried the rev 5 description.** Fixed in 6c.
13. **Earlier numeric errors (in the doc):**
    - T8 overrun "1,412 s": the relative `deadline_in_sec` 13,367 was subtracted from the absolute `finished_sec` 14,779. The real figure is 679 s.
    - "4 request timeouts": the real figure is 40.
    - "temperature 0.6 Qwen-like": wrong; the official thinking setting is 1.0.
    - "No failed branch got even one train pair exact": wrong.
    - "The newest block always stays": wrong for tool-less replies.
    - "62593bfd_b0 never checked": a labelling error; that branch did check, and the one that never solved is 135a2760_b1.
14. **T8 mechanism.**
    - `run_branch` computes `request_timeout_seconds = min(analyzer_timeout, remaining)` once per turn (`arc2_duck.py:835-838`), and Duck reuses it for every request in that turn.
    - Worst case is one analyzer timeout (1,800 s) past the deadline.
    - Fix R6-S6-a: clamp each request (and `_verify_submission`) to `deadline − now`.
15. **The rev 6 build initially missed map rows** (found at 09:20):
    - S3 fair share;
    - D3 trim order;
    - the T7 class-2 order;
    - the M1 overfit signs and metrics;
    - the Q-5 facts size (1,237 vs a 600-800 target);
    - `object_diff`;
    - the X1 parity control;
    - the P4 deadline tests;
    - labelling of infrastructure errors.

## 4. Decisions and reasons; options rejected

**Facts block.** It is token-bounded in 4 levels with an explicit omission note (p50 736). The C option of 1,200-1,500 tokens was not taken, because both plan reviews said 600-800 first.

**R6-D3 trim order.** The oldest complete reasoning-only block is dropped first; blocks are removed whole, so call/result pairs stay together.

**R6-S3 fair share.** A stage-2 run pauses softly after max(24,576, share) tokens and is re-queued round robin. A pause is not a cut: the branch can still reach the 100k cap.

**R6-S5 streams.** The design said 10; the user overrode it with 12 at 09:26.

**KV share for 12 streams.**
- The assistant chose **fraction 1.0 (73,444 per stream)** instead of 0.9.
- Reason: 0.9 would leave only about 41k of history, which is below rev 5's median prompts (41.6-51.9k), so trims would be frequent.
- Supporting evidence: in D66 at 12 slots, KV peaked at 90 % with no preemption.
- If the server queue backs up, the probe still steps down 12 → 10 → 8 → 6.
- This is the first question in the GPT zip README ("12 stream-এ পুরো KV ভাগ (1.0) গ্রহণযোগ্য কি না", "is a full (1.0) KV share acceptable at 12 streams?").

**Gate 4b** is a **time share, not a token cap**: window = 12 × (42,720 s − measured start) / 240, about 175 s per task, about 35 min in total.

**p90 last-admission cutoff.** Built and tested but **off by default** (`E12_R6_P90_CUTOFF=1` turns it on). Reasons: every request is already clamped to the deadline, and the cutoff would leave 12 workers idle for about the last 15 minutes. The 240 s admit gate stays.

**Target-only parity** is a separate notebook (PARITYR6), never run alongside production. It runs DFlash twice (control), then relaunches without the `--speculative-*` flags. It uses 8 probes, thinking on and off, T 0, one request at a time.

**Engine settings were left unchanged even though the user permitted changes:**
- Mamba: the assistant states each request needs 4 slots, so 12 streams need 48 of the 56 available.
- CUDA graphs: batches of 9 and 11 pad up, which is SGLang's normal behaviour.
- The real running count will be checked in the GPU report.

**Tokenizer fallback.** It is recorded and printed, not asserted. An assert would turn a working fallback into a dead solver.

**Earlier rejections, listed in the doc so they are not re-proposed:**
- forced or required `tool_choice` (HTTP 500 plus 59 empty submits; #43514);
- REFRAIN and other text early exits;
- hopeless-thinking stops;
- rolling summaries and reasoning stripping;
- penalties under DFlash;
- NVARC as the attempt_2 filler (+6.1 eval, but about 9.5 h per 240 tasks);
- images;
- the CPU rule library;
- LOOPSTOP (deferred: 5/5 loops caught with 0 false positives, but loops are now rare);
- a 30k hard cap on all branches (xhigh coverage would fall from 5/8 to 1/8);
- auto-stop at the first train-exact candidate (late corrections show why).

**Representation hedge.** Branch 1 sees the rev 5 grids view, exactly one deterministic alternative.

## 5. Open items

**Waiting on the user or GPT:**
- The GPU runs EVAL12R6 (7d69e502), G4B (f70c3228) and PARITYR6 (f3876b40) await the user's go. Pre-run estimates:
  - EVAL12R6: worst case about 2.4M tokens in a 9,000 s window;
  - G4B: about 35 min solve plus about 13 min load, 80 min budget;
  - PARITY: two server loads plus 24 short requests, 75 min budget.
- The GPT audit of the rev 6c zip (36 files, 4 questions) is pending, including whether KV fraction 1.0 is acceptable at 12 streams.

**Not built or not measured:**
- EVAL30R6: the builder variant exists but was not built.
- 10- and 12-stream speed on this stack was never measured. At 265 tok/s, EVAL12R6's worst case is "ঠিক সীমায়" (exactly at the limit).
- The claim "4 mamba slots per request" is not verified. The gap check said "mamba slots ≥ the donor ratio (60 per 10 streams)"; at 12 streams that ratio means 72 versus the 56 available.

**Unverified upstream issues:**
- SGLang #38009 (DFlash2 vs target-only divergence in thinking mode);
- #36548 (concurrent state contamination);
- #37475 (draft `target_layer_ids` fallback);
- #31954 (LPM starves cold requests);
- #39087 (quantized-drafter acceptance collapse);
- #43514 (required tool_choice poisons DFlash batches);
- z-lab/dflash #166.

**Unexplained:**
- `gen_throughput` 0.0 with 11 running and 8 queued at EVAL30 14,247 s.

**Validation still due on GPU:**
- the B3 estimate against the server's `prompt_tokens`;
- the risks listed in the doc: A1 may remove needed visual information, short turns may fragment reasoning, auto-check overhead, more streams may be slower, coverage vs the 60k floor, and train-exact ≠ test-correct.

**Pending decisions not built:**
- `ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS`;
- Franzen patch #4 plus `--watchdog-timeout 1800`, only if the prefix-hit rate at 10 streams falls below rev 5;
- a JIT cache dataset (−4 to −5 min), outside the agent bundle.

**Other:**
- LB ref 57456562 (EVAL12R5_LOW) is still pending.

## 6. Insights a fresh reader might miss

**Reversals and corrections:**
- The 10-stream design (with its 0.9 KV rule, "about 79k per stream") was reversed by the user at 09:26 to 12 streams. The assistant then moved to **KV fraction 1.0**, so the gap check's own rule (Σ ≤ 0.9 × KV) is no longer met.
- GPT plan review 2 (Q-1) had already corrected "drain near 70-80k" to "about 55k at a 24,576 cap". At 12 streams it is 48.4k.
- The gap check (R6-S3/T8) required a last-admission cutoff, yet the built version keeps it off by default. The assistant's reasoning is that per-request clamps make it unnecessary.

**Measurement traps:**
- Token-level parity under batching is not expected even A/A. E11-03 found spec vs plain shared only 152-166 of 512 tokens under batching, and in the Deep Audit an unchanged repeat scored 0.1875 against a 0.50 threshold. Parity must be judged on task-level outputs and batch-size-1 prefixes, with a same-config control first.
- The fake-server prefix-invariant test passed (84 ok, 0 breaks). The real defect E14 fixes (Duck's normalization deleting blank lines, so the previous reply was re-prefilled) was nevertheless not caught. My inference is that the fake check compares what Duck sends with what Duck sent, so it is blind to a normalization applied before both. Only the real server's cached-token counts can confirm the prefix.
- The capacity probe ("queue → workers −2, floor 6") is built into rev 6. Per memory (outside this part), rev 6d later found it dropping 12 → 6 on a normal queue of 1. In this part it is presented as a safety feature.

**Test-harness artifacts:**
- The harness rerun needed `E12_R6_MIN_ADMIT_SEC=5`, because the fake window is 25 s and the 240 s admit gate "correctly admitted nothing". Short windows interact with the admit gate.

**Accounting rules:**
- A static output cap keeps Duck's reply reserve in step (`_reply_reserve_tokens = _max_output_tokens`). A dynamic per-request cap breaks that sync unless both values are updated together (R6-D1).
- Soft caps do not bound tokens. A request-boundary budget B can reach B plus one request cap; rev 5 overshot to 159k at a 100k cap.

**Sampling and history:**
- Penalties are inactive under DFlash2 in 0.5.21, so any presence-penalty arm is a no-op unless SGLang is patched.
- Official Qwen3.8 thinking settings: T 1.0, top_p 0.95, top_k 20.
- The doc lists "removing reasoning from history" (Franzen), "stripping reasoning (madarshbb visible_reply)" and "rolling summaries / compaction" as explicitly not done. This is consistent with E14 keeping the raw reasoning.

**Deadline accounting:**
- Request deadline arithmetic must use one clock. The T8 error came from mixing a relative `deadline_in_sec` with an absolute `finished_sec`.

## 7. Relevance to the current E14-A notebook (concrete)

1. **Batches of 13 have no CUDA graph today.**
   - Today's list is "CUDA-graph batch list 1-8, 10, 12".
   - The gap check required "max-running = 10 (+1 workaround), CUDA-graph batch list covers 1..10 including 9 and 10". At 12 streams that means max-running 13.
   - Today a batch of 13 has no graph, and batches of 9 and 11 pad up.
   - **Check in E14-A:** the CUDA-graph 1..13 arm, and every arm that keeps `--max-running-requests 13`, must read back the server's captured `cuda_graph_bs` and assert that 12 and 13 are present. Otherwise baseline numbers at 12-13 running may mix graph and eager steps.
   - If `--min-free-slots-delay 1` removes the 11-of-12 cause, record whether the "+1 workaround" is still applied, so arms are compared at the same max-running.
2. **Mamba slot headroom is an unverified claim.**
   - The engine has 56 mamba slots (W1). The assistant's "4 per request → 48 at 12" is not verified, and the donor ratio (6 per stream) would need 72.
   - **Guard:** in the baseline window at 12 running and in the int8 mamba-checkpoint arm, log the server's mamba/SSM slot usage, or the line that limits admission. A mamba-slot limit is another possible cause of "11 of 12 running" besides the min-free-slots delayer, so the arm attribution should exclude it.
3. **KV share at 12 streams.**
   - Rev 6c sets the per-stream window to the full KV / 12 = 73,444 (fraction 1.0, no headroom; "খারাপতম ক্ষেত্রে 12 × 73.4k = পুরো KV pool", "worst case 12 × 73.4k = the whole KV pool").
   - **Check:** the 25-minute Duck run and the 12-running baseline should report KV usage peak and preemption/retraction counts. A run with retractions is not a clean "12 running" measurement.
   - Put the window value actually used (73,444 or another) in the phase record.
4. **Context confound in tok/s.**
   - Decode is about 25 % slower at 40-60k context (323 vs 429 tok/s), and DFlash acceptance drops 4.43 → 3.77 with context.
   - **Fix/guard:** every baseline window and every arm (DFlash width 6/4, NCCL env, graphs, mamba int8) must record the mean/median resident context per running request and the acceptance length alongside tok/s. Arms must be compared at matched context, or by token-level replay at fixed prompts. Otherwise an arm measured later in a growing conversation looks slower.
5. **Greedy parity A/A at concurrency 12.**
   - Evidence in this part: batching alone breaks token equality (152-166 of 512 shared; an A/A repeat scored 0.1875).
   - **Fix:** E14's greedy-parity phase must treat A/A at concurrency 12 as the noise floor and compare the target-only and arm results against it, not against 100 % equality. Also report task-level outputs: tool call parsed, same check verdict, and the batch-size-1 prefix length.
   - The PARITYR6 design is a usable template: DFlash twice, then a relaunch without `--speculative-*`, 8 probes with thinking on and off, T 0, one at a time.
6. **Deadline clamp for the 25-minute Duck run.**
   - T8 root cause: the per-turn timeout is reused for later requests in the same turn, so the worst-case overrun is the analyzer timeout (1,800 s), longer than the whole 25-minute run.
   - **Check:** the rev 6g solver must carry R6-S6-a (per-request `min(given, deadline − now)` and the verify clamp).
   - The E14 phase that ends the Duck run must not rely on `ThreadPoolExecutor` exit (`for f in futures: f.result()` waits for every worker). Assert in the phase record that the Duck phase ended within its deadline plus margin before the next phase or the report starts.
7. **Admit gate vs short window.**
   - The 240 s admit gate admitted nothing in a 25 s fake window. G4B and E14 use short windows (about 35 min or 25 min).
   - **Check:** verify that the 240 s admission gate and the first exposure of 28,000 tokens leave time for real admissions in a 25-minute run. Log admitted / exposed / meaningful tasks.
   - The p90 cutoff should stay off (default) unless deliberately tested.
8. **The capacity probe would contaminate 12-running measurements.**
   - The probe ("server queue → workers −2, floor 6") can lower workers on its own.
   - **Guard:** during the baseline windows, latch test and arm phases, either disable the probe or assert that worker count == 12 and real `#running-req` == 12 for the whole window. Discard or flag windows where it stepped down. This applies the "verify real concurrency" rule; the decode-log `running_hist` parser (`e11_decode_summary`) exists.
9. **Prefix-cache correctness after the raw-reasoning fix.**
   - Known cache killers: an abort without waiting for stream end (0 %), a request without tools (15 %), a template conditional (75 → 45 %), and `hash()` ordering.
   - **Check:** in the Duck run, record per-request cached_tokens / prompt_tokens and assert the expected high reuse (rev 5: 85-91 %) after the raw-reasoning change. This is the real proof that the blank-line normalization is fixed. The fake-server prefix test in this part could not detect that defect.
   - Make sure no phase aborts streams without draining them before the next phase. That matters for the latch test and the stall watchdog's kill path.
10. **Line endings and builder strings.**
    - CRLF crept into `arc2_duck.py` from Python edits (the sha changed). Twice, triple-quoted docstrings inside the builder's `"""` cell strings broke every variant.
    - **Guard in the E14 build:** check LF line endings of the embedded package and that the embedded `arc2_duck.py` equals its source sha. Compile every generated code cell (`compile(src, cell, "exec")`) before writing the .ipynb. Check the shas of every Kaggle-run notebook after the build (never-rebuild rule).
11. **Regex hang.** Confirm that the rev 6g solver embedded in E14-A uses the fixed `_GRID_ROW_RE = r"^\s*[\[\(]*\s*\d+(?:(?:\s*,\s*|\s+)\d+){3,}\s*[\]\)]*\s*,?\s*$|^\s*\d{8,}\s*$"` with `_GRID_ROW_MAX = 600`. A stale copy would freeze agent threads during the 25-minute run, which the stall watchdog (launch-only) would not catch.
12. **Tokenizer path with the /dev/shm copy.**
    - Rev 5/6 set `E12_TOKENIZER_JSON = SGL_MODEL_PATH / "tokenizer.json"` and only record a fallback (1.9 chars/token) when it fails to load.
    - **Check:** after the ranged copy to /dev/shm and its fail-open, set the tokenizer path from the path the server actually launched with. Confirm `tokenizer.json` is part of the copy and its verification. Print the `rev5_tokenizer` status (loaded / path / probe tokens) in the phase record. A silent fallback would skew the B3 history estimate and the trims.
13. **Runtime zero-throughput stall.**
    - EVAL30 showed `gen_throughput` 0.0 with 11 running and 8 queued, and it was never explained.
    - The E14 stall watchdog covers launch only. **Suggested addition:** make the heartbeat flag windows with running > 0 and generation throughput 0 for more than N seconds. E12's `_e12_heartbeat(every=300)` already snapshots `num_running_reqs` and `num_queue_reqs`. It need not act on them, but it should record them so a runtime stall is visible in the phase record.
14. **Stream interval.** Keep `--incremental-streaming-output --stream-interval 1` in every arm unless a decision is made deliberately. The builder comment links that choice to DFlash multi-token steps (`len(output_ids) % N == 1`, `output_streamer.py:427-436`), but its rationale is cut in this extract. Re-read the full comment in `build_e12_duck.py` before changing it.
15. **No penalty arms.** Do not add presence or repetition penalty arms: they are inactive under DFlash2 in 0.5.21 (#28180, verified statically). Keep T 1.0 / top_p 0.95 / top_k 20 as the production sampling. Greedy (T 0) is only for parity diagnostics.
16. **Pre-run estimate template.** Every GPU-run report in this part ends with a token / time / budget estimate (for example G4B: about 35 min solve, about 13 min load, 80 min budget). E14-A's 200-minute profile needs the same per-phase wall-time and quota estimate before the go. Hosen42 had 30.79-31.29 of 60 h used on 2026-10-21; that is stale by now and must be re-checked.
