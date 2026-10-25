# Insights from part_05 (2026-10-21 11:28 to 12:37)

## What this part contains (read in full, lines 1-2535)

- **Lines 1-1332:** an IDE selection the user attached at **2026-10-21 11:28**. It holds lines 1-1327 of `experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Review/E12 next fix list (running, 2026-10-20).md`, from F1 up to the end of "Rev 6d".
- **Lines 1334-1462:** the only live dialogue in this part (11:28-11:33). The user asks for timings, then for the G4B run on the Foysal account. The assistant runs a check-only, queues a commit run and writes a memory note.
- **Lines 1464-2535:** a second IDE selection of the same document (lines 1-1334), attached at **12:37**.
  - It is the same text again. The part file ends inside it, at "GPT audit bundle for rev 6: see the zip next to this file."
  - **So the user's actual 12:37 message is not in this part.** Neither are the 7 extra document lines (1328-1334, probably the "GPU run log" entry added at 11:33).
- **Cut `[result]` lines that matter:**
  - The push-script edit (line 1416-1419, `[+507]`): the exact change to `VARIANT` (adding `--eval12r6g4b`) cannot be seen.
  - The kernels_status script (line 1437-1442, `[+1132]`).
  - The memory-append command (line 1445, `[+81]`). Its result line shows the full text that was written.
  - The 11:28 push-script docstring (`[+4895]`).
  - I do not guess any of their content.

---

## 1. User instructions, rules, preferences and decisions

### Live messages in this part

| Time | Verbatim (Bengali) | English meaning | Note |
|---|---|---|---|
| 2026-10-21 11:28 | "হামিদ ভাইয়ের আইডিতে একটা কমিট করো।" | "Make a commit on brother Hamid's ID." | "Commit" means a **Kaggle commit run**, not a git commit. The user interrupted right after ("[Request interrupted by user]"). |
| 2026-10-21 11:28 | "এর আগে বলিও কোনটা কত সময় লাগবে?" | "Before that, tell me how long each one will take." | The user wants a time estimate per notebook before any run. This matches the standing pre-run estimate rule. |
| 2026-10-21 11:30 | "EVAL12R6G4B এটা ফয়সাল একাউন্টে চালাও" | "Run EVAL12R6G4B on the Foysal account." | This is the explicit go for **one** notebook, G4B only, on account `foysalemonshanto`. It is **not** a go for EVAL12R6 or PARITYR6. It also replaces "Hamid's ID": the assistant had asked whether that meant `hosen42`. |

### Rules and decisions quoted inside the attached fix-list document

The document records them in English, not in the user's original Bengali.

- **Rule (user, 2026-10-20):**
  - "Read every GPT report in full and validate every claim."
  - "Add the validated items here."
  - "Change no code until the user says so."
- **Bundling:** "bundle changes, big jumps" is the user's rule. All three GPT reviewers asked for isolated steps, so the doc says "the order of testing is the **user's decision**".
- **2026-10-21, deep re-validation:** "user: one bundled run, evidence-based changes only".
- **Rev 5 scope:** "user: B1-B8 + B9(a) temperature 1.0".
  - The user approved evidence-based output cuts on 2026-10-21; B5 cut the output cap 60,000 → 49,152.
- **R5 fixes:** "take them, validate again" applied R5-1..R5-5. R5-6 was not taken (no evidence).
- **Low effort:** "the same notebook, change only the xhigh agent"; the user chose low.
  - The run and the submission used `EVAL12R5_LOW` (sha `26fff41e`), not `EVAL6R5_LOW` (`cce9cb8f`).
- **Correction (user, 2026-10-21):** "Those were different experiments. That something did not work there does not mean it will not work here."
  - So past negatives are context only, used to choose safeguards and metrics, "never as a reason not to try".
- **Gap check:** "re-read every GPT and agent recommendation, something important may be missing".
- **F7 superseded (2026-10-21):** "the user's rule is that Duck runs on every task (full coverage)".
- **Rev 6:** "after the user accepted the plan".
- **Rev 6b:** "check every fix was taken, update, validate, then zip".
- **Rev 6c approvals:** "Gate 4b: permission granted. Take 12 streams. Permission for the rest too".
  - This is the approved exception to the 60k floor, for the G4B time-share profile only.
- **Rev 6d:** "OK, do 1 to 5".
  - Not taken: item 6 (the PARITY fixes) and item 7 (a self-contained test bundle). PARITYR6 stays NO-GO.
- **12,288 history:** the settings doc records "12,288 = approved exception to the 60k floor, user 2026-10-19" for the rev 1 / EVAL30 milestone port only. "Rev 6 needs a new decision." It was later approved inside the rev 6 plan.
- **Standing rules the doc applies:**
  - no truth in production paths ("Offline ground truth is never a runtime input"; "Routing is never tuned on known truth");
  - never rebuild a notebook that ran on Kaggle (the builder writes `_REBUILD`; shas are checked after every build);
  - `--sha` is mandatory at push.

---

## 2. Measured facts and results (exact numbers)

### Live session

- **11:29, time estimate per notebook.** Basis: rev 5 runs took ~13 min from start to solve start (733-799 s).

| Notebook (sha) | Start-up | Solve | Report | Total | Budget |
|---|---|---|---|---|---|
| EVAL12R6G4B `1bdadd82` | ~13 min | ~35 min (12 × ~175 s) | ~1-2 min | **≈ 50 min** | 80 min |
| EVAL12R6 `617c75de` | ~13 min | 150-min window (rev 5 low at 6 streams took ~110 min) | ~2 min | **≈ 2-2.75 h** | 3 h 5 min |
| PARITYR6 `af5011d9` | two loads of ~10-13 min each | 24 short probes | ~1 min | ≈ 45-60 min | 75 min |

  - PARITYR6 is still **NO-GO** (deadline fix not taken), so it will not be pushed.
  - Running G4B and EVAL12R6 back to back ≈ 3-3.5 h of GPU quota, excluding queue wait.
- **11:30, check-only on foysalemonshanto:**
  - authenticated kernels owner `['foysalemonshanto']`;
  - models OK: `qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1`, `qwen3-8-27b-dflash2/PyTorch/bf16/1`, `qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`;
  - datasets OK: `sglang-0-5-21-cu13-wheelhouse` (20 files) and `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` (20 files);
  - **"gpu quota used h 21.88 of 30.0"**;
  - "notebook sha256 1bdadd8266e758ee OK".
- **11:30-11:33, push:**
  - command: `push_e12_duck.py --eval12r6g4b --commit --sha 1bdadd8266e758ee48c0947d3197d0aa804f5d751d28277a79e947e7e73980a5`;
  - output: "saved + commit run queued: foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b version 1 url https://www.kaggle.com/code/foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b error";
  - status: `KernelWorkerStatus.RUNNING`.
- **11:33, assistant's report:**
  - quota left ~8.1 h of 30; this run needs ~0.85 h;
  - planned checks after the run: real stream count, KV peak, retractions or step-downs, tasks exposed, `stage1_left`, tokens to first check, candidates, eval score.
- **Memory line written:** "2026-10-21: E12 rev 6d EVAL12R6G4B (sha 1bdadd82, gate 4b = hidden time share, 12 workers) commit-run on foysalemonshanto (kernel q27-xhigh-e12-duck-repl-sglang-eval12r6g4b v1); EVAL12R6 (617c75de) follows after the G4B review; PARITYR6 NO-GO."
- **Push script facts** (`_scratch/session_1c6db7aa/push_e12/push_e12_duck.py`, 85 lines):
  - "Account = the one in the credentials used (default ~/.kaggle -> foysalemonshanto)";
  - "L4 x4, internet OFF, pinned image (= S49 / E11-04)";
  - backup `push_e12_duck.py.pre_r6` made before the edit.

### From the fix-list document (results recorded before this part)

**Run results**

| Run | Result |
|---|---|
| EVAL6R5 xhigh `f4fed1ac` | 4/6 tasks, 5/8 inputs |
| EVAL12R5_LOW `26fff41e` | 7/12 tasks, 10/19 inputs; submitted as ref 57456562 (pending) |
| EVAL30R5 `415a68f2` | not run |

**Replays**

- Simulated F1+F5: EVAL6 5/6 tasks, 7/8 inputs (actual 3/6, 4/8); EVAL30 4 tasks, 7/48.
- Rev 6 / 6b / 6d replays: rev 4 EVAL6 **7/8**; rev 5 xhigh **5/8**; rev 5 low **10/19**; kept_r6 ≥ kept_r5 on every branch (9385bd28_b0 +1).
- Oracle = final: xhigh 5/8, low 10/19, so selection is not the lever.
- Selective branch 1:
  - xhigh 5/8 at 498,253 tokens vs 846,080 (−41.1 %);
  - low 9/19 at 1,094,372 vs 10/19 at 1,743,889 (−37.2 %); 135a2760 is lost.

**Throughput and capacity**

- Rev 5 low: 1.744M tokens in 6,570 s at 6 streams ≈ **265 tok/s**. Effective range 244-265 tok/s.
- Per-task budget for 240 tasks in 11.3 h: 255 tok/s → 43.2k; 350 → 59.3k; 416 → 70.5k.
- Decode at 40-60k context is about 25 % slower (**323 vs 429 tok/s**). DFlash acceptance falls **4.43 → 3.77** over context.
- 6 → 12 streams: +17 % in one measurement (+30-60 % from the short sweep).
- Cold prefill about **1,400-1,650 tok/s** (E11), i.e. 30-35 s per 50k resume.

**Prefix cache and requests**

- Cached fraction **85.4 % (xhigh) / 91.0 % (low)**; uncached about 888k / 1.577M; uncached prefill > generated (0.97M vs 0.85M, xhigh).
- Median completion 1,115 / 1,090; median prompt 51.9k / 41.6k.
- Cache history:
  - stable order raised the cache from 11-18 % to 87 % (E11-06);
  - an abort without waiting for the stream gives 0 % vs 52-99.7 % with waiting;
  - requests without tools: 15 % vs 99 % shared prefix;
  - a template conditional: 75 → 45 %;
  - `hash()` order changed an LB (hosen42).
- Parity under batching: spec vs plain shared only **152-166 of 512 tokens** (E11-03). An unchanged repeat scored **0.1875 vs a 0.50 threshold** (Deep Audit).

**Engine settings (W1 / rev 6c)**

- KV pool 881,336. Per-stream window 73,444 at 12 streams with fraction 1.0. 0.9 gives 66.1k; 0.95 gives 69,772 (44,684 input at the 24,576 cap).
- `SGL_MAX_RUNNING` 12, **server 13 (the 11-of-12 workaround)**.
- CUDA graph batch list **1-8, 10, 12**: 9 and 11 pad up.
- mamba cache 56 slots, 4 per request, so 12 streams need 48.
- fp16 SSM.

**Context and caps**

- Input budget 131,072 − 49,152 − 512 = 81,408; with a 12,288 cap it becomes 118,272.
- Duck window 131,072 vs server context 155,648.
- Context loss example (low 4c3d4a41_b0): prompt 66,189 + reply 16,365 + 49,152 reserve > 131,072, so the trim dropped the prompt to 19,856, and the next request accepted an overfit transform.

**EVAL30 deadline overrun (T8)**

- Solve ended **679 s** past its deadline (not 1,412 s); finished 14,779 vs a deadline of ~14,100. That is 379 s past the 4 h total.
- Requests with "read timeout = 419.6 s" and 469.7 s after the deadline.
- At 14,247 s: `num_running_reqs` 11, `num_queue_reqs` 8, **`gen_throughput` 0.0**.
- 40 request timeouts in 21 of 31 branches (28 full 900 s + 12 deadline-bounded, ~31,000 stream-seconds); 8.4M tokens re-prefilled.

**Pre-check behaviour (rev 5)**

- First check at a median of 55.1k (xhigh) / 53.7k (low).
- 273k of 742k (37 %) / 279k of 1,394k (20 %) of pre-check tokens came after the first stated code intent.
- Self-detected misreads in 12/12 / 23/24 branches.

**Facts block size**

- Before bounding: p50 1,237 / p90 1,752 tokens.
- Rev 6b: p50 736 / p90 837 / max 1,396.
- Rev 6d: 0 over 800 (max 798, p50 699, p90 779); 16 tasks at level 4, 2 at level 5.

**Tests**

- smoke rev6 104/104 (3 consecutive runs); rev5 56/56; adapter 41/41.
- Harness: EVAL12R6 r6 45/45, rerun 22/22, launchfail 6/6; G4B 46/46; PARITYR6 19/19.

**G4B formula**

- Solve window = n_tasks × (42,720 − measured solve start) / 240, where 42,720 = 43,020 − 120 − 180.
- At an ~800 s start this is ~175 s per task, ~35 min for 12 tasks. At 265 tok/s that is about 46k tokens per task.

**Other numbers**

- Verification cost 0.75-2.71 s each, ≤ 3 per branch.
- Quota on hosen42 was 30.79-31.29 of **60 h**; foysalemonshanto has **30 h**.
- Identical code on EVAL22 gave 2, 2, 3, 1 of 22, so small evals are noisy.

---

## 3. Failures, mistakes and root causes

**Live session**

- **QUICK_SAVE said, commit run done.** At 11:29 the assistant wrote "দুটোই QUICK_SAVE হিসেবে push হবে, তাই নিজে থেকে চলবে না" ("both will be pushed as QUICK_SAVE, so they will not run by themselves").
  - At 11:30 it pushed with `--commit`, i.e. an automatic commit run.
  - The user had asked for a "কমিট" (commit), so the action matched the request but contradicted the assistant's own statement.
  - Nobody pointed this out in this part.
- **The push output ends in the word "error"** even though the run queued and showed RUNNING. It looks like an empty error field printed after the URL. Nobody investigated. The cut-off part of the line does not let me confirm this.

**Errors recorded in the fix-list document**

Most are the assistant's own, caught by GPT reviews or by itself:

- "No failed branch got even one train pair exact": wrong (62593bfd_b1 2/2, b5ca7ac4_b0 3/3, 135a2760_b0 1/2).
- "The newest block always stays": wrong for tool-less replies.
- "Temperature 0.6 is Qwen-like": wrong; the official thinking setting is 1.0.
- EVAL30 request timeouts "4": actually **40**.
- **T8 "1,412 s overrun":** wrong; it was 679 s. The assistant had subtracted a relative 13,367 from an absolute 14,779.
  - **Root cause of the overrun:** `request_timeout_seconds = min(analyzer_timeout, remaining)` is computed once per turn (`arc2_duck.py:835-838`) and reused for every request in that turn.
  - With the 1,800 s rev 5 timeout, a run could overrun by up to ~30 min.
- Q1 audit note "metrics_snapshot drops num_retracted_requests_total": wrong. That was the vLLM-variant filter; the SGLang `_E11_SNAP_KEYS` already keep `retract` and `token_usage`. The real gap was that the probe did not use them.
- "G4B predicts the LB/eval ratio": too strong. It is a selected 12-task screening at the hidden time share, "never an LB projection".
- "Conditional branch 1" is really "prioritised by evidence class".
- A labelling error: 62593bfd_b0 *did* check (2/2 at 58k); the never-solved branch is 135a2760_b1.
- The old PIV wording "the model must finish a full solve before a tool call" is wrong.
- **Rev 6 build defects (found and fixed):**
  - **Regex hang:** overlapping separators in `_GRID_ROW_RE` backtracked exponentially; the smoke test hung > 11 min.
  - **Wrong opener after a rejected submit.**
  - **Caps metric counted trims, not requests.**
  - **CRLF:** the assistant's edits turned `arc2_duck.py` into CRLF.
- **Rev 6b defects:**
  - **Two workers driving the same agent:** a data race; the negative control showed 24 concurrent entries.
  - **Fair-share starvation:** one task took 792k.
- **Rev 6d defect:** published records shared nested lists with the live session (fixed with a deep copy).
- **Rev 6c audit findings:**
  - **PARITY deadline bypass:** `urlopen(req, timeout=900)` with no deadline; 24 probes could take up to 6 h.
  - **Isolation preflight fails open:** 0/12 own markers counts as a PASS; an exception (`None`) keeps 12 workers.
  - **Unbounded join:** each join adds 120 s, i.e. 24 min worst case.
  - **p90 off by one:** `[100,200,300,400,800]` → 400 instead of 800.

**Historical root causes in the document**

- 5 avoidable history resets: a grid-calibrated chars/token ratio (1.40) overestimated a 60k reply (estimate 77-111k vs real 65-69k).
- Truncated tool calls were executed: SGLang's parser returns closed JSON.
- The S1 scheduler trap: `n_live = len(jobs)` puts all 480 branches live.
- Penalties are inactive with DFlash2 in 0.5.21 (no `cumulate_penalty_output_tokens` call).

---

## 4. Decisions made and reasons; options rejected

**Made in this part**

- **Run order:** G4B first (≈ 50 min). EVAL12R6 only after the G4B review shows 12 streams are stable, with no retractions and no KV pressure.
- **PARITYR6 is not pushed:** NO-GO, because its deadline fix was not taken.
- **Account:** G4B runs on `foysalemonshanto` (the user's choice), not `hosen42`.
  - Reason given: the user said so. The assistant had asked because earlier check-only runs were on hosen42.
- **Credentials:** the assistant stated it would not read the credential file contents.

**Recorded in the document, with reasons**

- **KV fraction 1.0, not 0.9, for 12 streams.** With 0.9 the history is 41.0k at the 24,576 cap, below the rev 5 median prompts of 41.6-51.9k. D66 ran 12 slots at 155,648 context with a 90 % KV peak and 0 preemptions.
  - The hidden-run default (1.0 vs 0.95) is left to G4B's measured running count, KV peak, retractions and queue.
- **Engine unchanged:** mamba 56 (48 needed) and the CUDA graph list 1-8, 10, 12 ("9 and 11 pad up … avoids capture memory").
- **p90 last-admission cutoff: built but OFF by default.** Every request and verify is clamped to the deadline. A 24,576-token chunk takes ~14-16 min at 25-30 tok/s per stream, so the cutoff would idle up to 12 workers.
- **Tokenizer:** record its state rather than assert (an assert would turn a fallback into a dead solver).
- **Rejected options:**
  - `ARC3_CONTEXT_DRAIN_FLOOR`: trims every turn kill the prefix cache.
  - Passive harvest by name pattern: it missed 2 of 12 and added 0.
  - A 30k hard cap: xhigh coverage would fall from 5/8 to 1/8.
  - Universal auto-stop at the first train-exact candidate: late corrections (135a2760_b1 71,306 → 97,972; 898e7135_b0 56,868 → 73,523; b1 77,859 → 79,662).
  - Forced or required `tool_choice`: Exp 4 HTTP 500 and 59 empty submits; SGLang #43514.
  - Rolling summaries or stripping reasoning: Franzen's regression.
  - REFRAIN early exit, NVARC filler, images, the CPU rule library.
  - Penalties under DFlash without a patch.
  - Target-only parity alongside production ("never alongside production").
  - "Print a full grid on demand": changed to "a crop or a single small grid".
- **Not taken in rev 6d:** the PARITY fixes (C6-1, C6-3, Q2 long probes) and the self-contained test bundle.

---

## 5. Open items at the end of this part

1. **G4B run (`1bdadd82`)** is RUNNING on foysalemonshanto. Promised post-run checks:
   - real stream count, KV peak, retraction or step-down events;
   - tasks exposed, `stage1_left`;
   - tokens to first check, candidates, eval score.
   - No result is in this part.
2. **EVAL12R6 (`617c75de`)** waits for the G4B review and the user's go.
3. **PARITYR6 stays NO-GO.** C6-1 (deadline clamp on probes and relaunch), C6-3 (tool name + arguments), Q2 (2 long probes at 12,288) and A3-Q2 (a tool-result continuation probe) are not built.
4. **The self-contained test bundle is not built:** `smoke_adapter.py`, fixtures and relative paths are missing from the zip.
5. **EVAL30R6:** builder variant present, not built.
6. **EVAL12R5_LOW submission ref 57456562:** pending.
7. **Hidden-run KV fraction** (1.0 vs 0.95): decided from G4B data.
8. **Unexplained:** `gen_throughput` 0.0 with 11 running and 8 queued at 14,247 s in EVAL30 (probably KV thrash).
9. **SGLang issues not verified (marked "C"):** #38009, #36548, #37475, #31954, #39087, #43514, z-lab/dflash #166. Also the 0.5.21 LPM behaviour (E3).
10. **Account wording:** "হামিদ ভাইয়ের আইডি" (Hamid's ID) was never mapped to an account. The user just named Foysal for G4B. Which account EVAL12R6 should use is not stated.
11. **Quota:** foysalemonshanto had ~8.1 h left at 11:30 before G4B (~0.85 h). EVAL12R6 (~2-2.75 h) would leave ~5 h. Neither the quota reset date nor later usage is known here.
12. **The user's 12:37 message:** only the re-attached document is in this part. The question itself is in the next part.
13. **Pending design decisions** (some resolved later by approvals in the document):
    - the representation hedge (one alternative view for branch 1; built as "b1 grids view");
    - Franzen patch #4 + `--watchdog-timeout 1800` "only if prefix-hit at 10 streams falls below rev 5";
    - `ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS`;
    - the JIT cache dataset (−4 to −5 min), "outside the agent bundle".

---

## 6. Insights a fresh reader might miss

- **"কমিট" means a Kaggle commit run, not a git commit.** The assistant handled it that way. Its own "QUICK_SAVE" claim one minute earlier was silently reversed.
- **Memory is stale on shas.** It lists rev 6c shas (EVAL12R6 `7d69e502`, G4B `f70c3228`, PARITY `f3876b40`), but rev 6d superseded them (`617c75de`, `1bdadd82`, `af5011d9`). The run used `1bdadd82`.
- **Two Kaggle accounts with different quotas:** hosen42 has 60 h (30.79-31.29 used); foysalemonshanto has 30 h (21.88 used at 11:30). Quota planning must name the account.
- **The doc header says "Status 2026-10-21 (late night)"**, yet the user attaches it at 11:28 on 2026-10-21. The conversation clock and the document's local-time wording may differ; do not order events by that header alone.
- **The push succeeded although its output line ends in "error".** Status was RUNNING. Do not read the trailing "error" as a failure.
- **The rev 6b fake-server prefix test passed ("84 ok, 0 breaks").** Yet E14 later found that Duck's normalization deleted blank lines, so the previous reply was re-prefilled.
  - Inference (not stated in this part): the fake server's replies or comparison did not exercise blank lines in reasoning. The CPU invariant test was not enough proof of prefix reuse on the real server.
- **Gate 4b is a time share, not a token cap.** It applies only in commit mode; a competition rerun uses the normal rev 6 settings (dual mode). Its 12 tasks were selected from earlier solver results, so "never an LB projection".
- **The "11-of-12 workaround" is a +1 on server max-running (13).** Any fix to the DFlash min-free-slots delayer interacts with it: with the fix, 13 could really run.
- **The T8 overrun mechanism is general.** Any loop that computes a timeout once and reuses it for later HTTP calls (benchmarks, parity probes, replays) can overrun the deadline by up to one timeout. The PARITY routine had exactly that bug (900 s per probe, no deadline).
- **The isolation preflight was fail-open** (`isolation_ok = not leaks`; `None` treated as pass). This is a pattern to avoid in every new preflight: three states (PASS / FAIL / INCONCLUSIVE), and only PASS keeps the aggressive setting.
- **Under batching, token equality is not a parity measure** (152-166 of 512 shared; unchanged repeat 0.1875). A/A controls and batch-size-1 prefixes are required. The doc's guidance is to judge on task-level outputs.
- **Long-context decode is ~25 % slower** (323 vs 429 tok/s) and acceptance falls (4.43 → 3.77). Throughput windows from short-context requests overstate the hidden-run speed.
- **Uncached prefill already exceeds generated tokens** (0.97M vs 0.85M, xhigh). Prefix reuse is a first-order speed lever, not a detail.
- **The 13-minute start-up (733-799 s)** was the rev 5 baseline. G4B's window formula depends on the measured solve start, so a slower start (for example, GCSFUSE) shrinks the G4B solve window automatically.

---

## 7. Relevance to the current E14-A notebook (concrete)

1. **The CUDA graph list must cover the real max running count.**
   - Evidence: server max-running is 13 ("server 13 (the 11-of-12 workaround)") while the graph list is 1-8, 10, 12. 9 and 11 pad up, but **13 has no graph above it**.
   - If the `--min-free-slots-delay 1` arm removes the 11-of-12 effect, up to 13 requests can run and fall outside the captured graphs.
   - Fix or check:
     - either capture 1..13, as E14's "CUDA graphs 1..13" arm proposes, in **every** arm that keeps max-running 13;
     - or set max-running 12 once the delayer fix is adopted.
   - Record the real running max from the decode log in each arm. mamba is fine: 4 × 13 = 52 ≤ 56.
2. **Every measurement phase needs an absolute deadline clamp per HTTP request.**
   - Applies to: NCCL bench, baseline windows, latch test, token replay, greedy parity, the 25-min Duck run, and target-only relaunch waits.
   - Evidence: T8 (a per-turn timeout reused → 679 s overrun; up to ~30 min with the 1,800 s timeout) and C6-1 (`urlopen(timeout=900)` × 24 probes ≈ 6 h; relaunch wait `max(60, …)`).
   - Guard: `timeout = min(given, phase_deadline − now)`, with no new probe when less than 60 s is left; write partial results.
   - Use one shared join cutoff, `deadline + 60` (C6-4), so that the 200-min profile cannot be exceeded.
3. **Greedy parity (A/A, concurrency 12, target-only) needs the calibration rules.**
   - Run the A/A same-config control first and judge against it. Token equality under batching is expected to diverge (152-166 of 512 shared; an unchanged repeat scored 0.1875).
   - Judge on tool-call presence, the tool **name + arguments** (C6-3), finish reason and batch-size-1 prefixes.
   - Include at least one long thinking-on probe at the real first-request cap of 12,288. Probes of 2,048 tokens rarely reach a tool call; the rev 5 first reply was p50 33.6k.
   - Include one tool-result continuation probe (A3-Q2).
   - Target-only parity must be a separate, bounded relaunch, never alongside production. None of these PARITY fixes were built in rev 6d.
4. **Any marker/isolation preflight at 12 running must use the three-state rule.** PASS only when own == n and there is no foreign marker; FAIL or INCONCLUSIVE gives the fallback. Evidence: the C6-2 fail-open bug. Check that the rev 6g solver keeps `isolation_verdict` / `workers_for_isolation`.
5. **Baseline windows at 12 running should record context length.**
   - Evidence: 323 vs 429 tok/s at 40-60k vs short context; acceptance 4.43 → 3.77.
   - Arms compared at different context mixes (CUDA graphs, int8 mamba, DFlash width 6/4, NCCL env, the #44401 backport) are confounded unless each window logs the mean resident context and the acceptance length.
   - Also log `num_running_reqs`, `num_queue_reqs`, `gen_throughput`, `num_retracted_requests_total` and `token_usage`. Use SGLang's `_E11_SNAP_KEYS`, not the vLLM filter.
   - EVAL30 had `gen_throughput` 0.0 with 11 running and 8 queued, the KV-thrash signature the heartbeat should flag.
6. **The prefix-reuse check for the raw-reasoning history fix must run on the real server.**
   - The rev 6b CPU invariant ("84 ok, 0 breaks") missed the blank-line normalization bug.
   - In the 25-min Duck run, record per request `cached_tokens / prompt_tokens`, and check that request n+1's prompt starts with request n's prompt + reply byte for byte (blank lines included).
   - Baseline to beat: rev 5 cached 85.4 % / 91.0 %.
   - Also keep the R6-D2 rules:
     - same tools list and template kwargs on every request;
     - no tool-free request;
     - aborts wait for the stream to end (0 % vs 52-99.7 %).
7. **Deadline and records:**
   - Keep the rev 6 per-request clamp (`min(given, deadline − now)`) and the verify clamp (`min(2 × python_timeout, deadline − now − margin)`).
   - Keep deep-copied published records (the rev 6d fix).
   - These must be present in the rev 6g solver that E14 embeds. Verify by grepping the embedded adapter.
8. **Regex-hang guard:** confirm the embedded rev 6g `_GRID_ROW_RE` uses disjoint separators `(?:\s*,\s*|\s+)\d+` and skips rows longer than 600 characters. The old pattern froze an agent thread for > 11 min.
9. **Do not add a penalty arm.** Presence, frequency and repetition penalties are inactive under DFlash2 in 0.5.21 (static read; #28180). Never use a named or required `tool_choice` (#43514). Keep temperature 1.0 and fp16 SSM.
10. **Quota and estimate before the run:**
    - foysalemonshanto had 30 h with 21.88 used at 11:30 on 10-21, before G4B (~0.85 h) and possibly EVAL12R6 (~2-2.75 h).
    - E14-A's 200-min budget = 3.33 h. The written estimate must name the account and its current quota (hosen42 has 60 h).
11. **Timing baseline for the new /dev/shm copy and launch watchdog:** rev 5 start-to-solve was ~13 min (733-799 s). The JIT cache was estimated at −4 to −5 min. Report E14's phase record against these numbers, and note that G4B-style windows derive from the measured start.
12. **Push mode:** state QUICK_SAVE vs commit explicitly and do what the user asked. In this part the assistant announced QUICK_SAVE and then ran `--commit`. The push script's success line ends with "error"; check `kernels_status` rather than the text. The `--sha` assertion is mandatory.
