# Auditor 4 — rev 3f / 3g changes (2026-10-25)

## Scope

This audit covers the changes made since rev 3 and how they interact:

1. GPT's blocker fix in `solve_all_r6`.
2. The per-attempt budgets (`b_attempt_budget`, `max_generated_tokens_b1`, the report's `_bud`).
3. The two notebook variants from one builder: 45K30K and 40K35K.

It also answers the capacity, KV and time questions with numbers from the real EVAL120 G4B runs (Exp 14, 2026-10-24).

The audit was read-only. No project file was changed. Scratch files are under `D:\ARC AGI 2\_scratch\e15b\agent4\`.

### What was read and compared

- **Package diff, rev 3 → now.** `arc2_duck.py` differs in 65 diff lines. `tool_agent.py` and `python_tool_sandbox.py` are byte-identical after CRLF normalization.
- **Package sha.** `arc2_duck.py` sha256 = `bc26fa16…`. The payload `q27x_e15b_duck_b20804213563.bin` (sha `b2080421…`) contains exactly this file, and the unchanged vendor files.
- **Builder diff, rev 3 → now.** Note: the builder was edited again at 05:37, during this audit. `PUSHED` now also lists `560ed8b8…` (40K35K SUBMIT, foysal) and `5d8069b9…` (45K30K SUBMIT, esfersami50). So both SUBMIT variants are marked as already pushed or run on Kaggle.
- **Notebooks.** All four were dumped cell by cell and diffed against each other and against 2X35K:

  | Notebook | sha256 (prefix) | Size |
  |---|---|---|
  | SUBMIT 45K30K | `5d8069b9…` | 260,917 B |
  | EVAL120G4B 45K30K | `40b36fb7…` | 260,897 B |
  | SUBMIT 40K35K | `560ed8b8…` | 260,917 B |
  | EVAL120G4B 40K35K | `36c5cf15…` | 260,897 B |

- **SGLang 0.5.21 source, local copy.** `schedule_policy.py` (admission estimate) and `schedule_batch.py` (decode-memory check and retraction).

### Tests run (CPU only)

- **`test_duck_b.py`: ALL PASS** (87 checks, 0 skips) on the local Python 3.14.6. This includes section 15 (the GPT blocker) and section 16 (rev 3f budgets).
- **`verify_items.py` on all four notebooks: "ALL CODE ITEMS PRESENT"**, and size < 400 kB for each.
- **Scratch analyses** (scripts in `_scratch\e15b\agent4\`):
  - `ctx_overhead.py`: context growth.
  - `rates.py`: decode rates.
  - `budget_need.py`: tokens per attempt.
  - `kv_sim.py`: KV peak Monte Carlo.
  - `commit_sim.py`: commit-run timeline Monte Carlo.

---

## Findings, ranked by impact

### F1 — MEDIUM (capacity, not a code bug) — CONFIRMED by arithmetic on real data: the hidden rerun cannot give attempt 2 to all 240 tasks at the measured throughput

**Location.** The design itself:

- builder line 31: `ATTEMPT_TOKENS, ATTEMPT_TOKENS_B1`;
- notebook "CELL 1", E15B block, line ~358 of the cell;
- `solve_all_r6` stage order: attempt 1 on every task first (D-A), `arc2_duck.py:3345`.

**Scenario.** About 77 % of attempts run to their full budget, so the run needs about 69k tokens per task (table below). At the 12-stream aggregate measured in the real runs, attempt 1 covers all 240 tasks, but only about 2/3 of tasks get an attempt 2. Those runs:

- EVAL120G4B XHIGH noE16: 338 tok/s;
- XHIGH E16: 340 tok/s;
- LOW: 315 tok/s.

**Evidence.**

Tokens per attempt are taken from the 240 real branch-0 records of the two XHIGH EVAL120G4B runs. An attempt ends at its first accepted **submit** if that comes within the budget, otherwise at the budget. Only an accepted submit ends an attempt (`done_now` checks `arc2_accepted`); an accepted check does not.

| Budget | Mean tokens / attempt | Accepted within budget |
|---|---|---|
| 30k | 28.3-28.4k | 16-17 / 120 |
| 35k | 32.6-32.7k | 19-20 / 120 |
| 40k | 36.7-36.8k | 23-25 / 120 |
| 45k | 40.6-40.7k | 28 / 120 |

The hidden solve window is about 41,990 s: `hidden_solve_end_sec` 42,720 minus a solve start of about 730 s (READY was 12 min in these runs).

| Aggregate tok/s | Tokens available | 45K30K: all attempt 1 / attempt-2 coverage | 40K35K | 2X35K |
|---|---|---|---|---|
| 300 | 12.6 M | 9.74 M (9.0 h) / ~100 of 240 (42 %) | ~116 (48 %) | ~146 (61 %) |
| 340 (12 streams, measured) | 14.3 M | 9.74 M (8.0 h) / ~160 (67 %) | ~167 (70 %) | ~197 (82 %) |
| 380 | 16.0 M | 9.74 M (7.1 h) / ~219 (91 %) | ~219 (91 %) | all 240 |

- Attempt-1 coverage is safe in every case.
- The attempt-2 shortfall falls mostly on tasks whose attempt 1 already has a verified candidate: the stage-2 rank is `(1, rounds, class, …)`, and class 2/3 (verified) sorts after class 0/1. That is the least costly place to lose attempts.
- EVAL120G4B gets the same compute per task (175 s/task), so its `b1_started` count measures the same coverage directly.

**What the run report will show.** In the report's `rev6.scheduler`:

- `sessions` will have fewer than 240 `_b1` entries;
- `heap_left` > 0;
- `b_summary.attempt2_started` < 240;
- `eval_b_attempts.b1_started` < 120 in the EVAL120 variant.

The run itself finishes on time, because every request is clamped to the deadline.

**Minimal fix.** No code fix is needed. Read the real 16-stream aggregate from the 2X35K GPU run now going (`b_summary.server_gen_tokens` / solve seconds, and `gen_tok_s_p50_by_running`) before choosing between the variants. At 16 streams, attempt-2 coverage scales roughly linearly with that number (table above).

### F2 — MEDIUM-LOW — PLAUSIBLE: the wave-1 KV peak at 45k approaches the pool, and the KV admission gate cannot prevent it

**Location.**

- `solve_all_r6` starts 16 attempt-1 sessions at t = 0 (`arc2_duck.py:3210-3211`, then the workers);
- `kv_blocked()` (`arc2_duck.py:3289-3296`) only holds new admissions;
- notebook "CELL 1" line ~230: `SGL_KV_PER_REQUEST = 50_000`.

**Scenario.** All 16 attempt-1 sessions start together and decode at similar rates on one batched server. The attempts that are not accepted (about 77 %, more on the hard commit tasks) reach 40-45k generated tokens at about the same time. Their contexts are then 45k plus prompt and tool output each.

SGLang admits every new turn: the admission estimate clips `max_new_tokens` to `CLIP_MAX_NEW_TOKENS` = 4,096 (`schedule_policy.py:79`, and lines 1095 and 1501). So pressure shows up only as **decode retraction**: `check_decode_mem` → `retract_decode`, logged as "KV cache pool is full. Retract requests."

The scheduler's 0.95 gate cannot help, because the growth comes from running requests. Holding new admissions does not shrink them.

**Evidence: context growth.** From real untrimmed requests:

- Overhead (context minus generated) at ≥ 30k generated: p50 8.0-10.9k, p90 12.8-16.4k.
- **Context when generated crosses 45k:** p50 51.7-55.0k, p90 56.7-58.9k, max 60.2-63.6k.

So the builder comment "attempt 1 ends ≤ ~60k at 45k" is about the p90, not the maximum.

**Evidence: Monte Carlo.** The peak of the sum of resident contexts over 16 streams, as a fraction of a 852k pool. Per-attempt wall rates, overheads and accept points are sampled from the real data, and slots are refilled from the queue. This is an upper bound: an attempt's whole context is counted even while it runs a tool, although it is then evictable cache.

| Rate spread (CV) | Accepts | 45K30K p50 / p90 / max | 40K35K p50 / p90 / max | 2X35K p50 / p90 / max |
|---|---|---|---|---|
| 0.16 (real cross-run spread) | real | 0.76 / 0.82 / 0.96 | 0.70 / 0.75 / 0.83 | 0.64 / 0.69 / 0.76 |
| 0.16 | none (worst, hard tasks) | 0.83 / 0.87 / 0.96 | 0.75 / 0.79 / 0.85 | 0.68 / 0.72 / 0.76 |
| 0.08 | none | 0.90 / 0.93 / 1.00 | 0.82 / 0.85 / 0.90 | 0.73 / 0.77 / 0.81 |
| 0.04 (near lockstep) | none | **0.94 / 0.97 / 1.02** (P ≥ 0.95 = 0.37) | 0.85 / 0.88 / 0.93 | 0.77 / 0.80 / 0.83 |

- The peak falls about 25-35 minutes into solving, at 22-26 tok/s per stream.
- Streams on one batched server are probably closer to lockstep than the cross-run spread (CV 0.16) suggests.
- So a peak of about 0.9-0.97 of the pool in the commit run (16 hard tasks first) is plausible. Brief retraction at the end of wave 1 is possible but not likely.
- In the hidden run, the peak is lower and happens only once: after wave 1, attempts are at random phases (steady state about 0.55-0.6).

**Consequence if it happens.**

- Some requests are retracted and recomputed, which costs minutes, not correctness.
- A retracted request's client-side timer keeps running (see F7).
- The KV gate will show `kv_holds` > 0 only if new attempts are due exactly then.

**What the run report will show.**

- `rev6.scheduler.gauges` column `kv` (30-s samples): the maximum near t ≈ 1,500-2,100 s of the solve.
- `kv_holds` and `kv_hold_sec`.
- `b_summary.server_log`: `retract_lines`, `preempt_lines`, `oom_lines`.
- `server_queue_gt0_share` (retracted requests go back to the queue).
- `gen_tok_s_p50_by_running`.

**Minimal fix.** None is required before the run. The decision rule already set for R2 applies: read `retract_lines` and the KV maximum. But note that `E12_R6_KV_ADMIT = 0.85` does **not** address wave 1. If retraction is seen, the levers that work are:

- 15 streams;
- staggering the first 16 starts by a few minutes;
- 40K35K (p90 ≤ 0.88 even near lockstep).

### F3 — LOW-MEDIUM — PLAUSIBLE: the 40K35K commit window (85 min) is relatively tighter than the 45K30K one (92 min); both depend on the unmeasured 16-stream throughput

**Location.**

- builder line 35: `COMMIT_WINDOW_MIN, COMMIT_BUDGET_MIN = {"45K30K": (92, 115), "40K35K": (85, 110)}[BTAG]`;
- notebook "CELL 1", profile `E15B`, lines ~97-106, and the comment "40k + 35k: worst ~ 74 min".

**Scenario.** Both variants spend 75k tokens on the critical path: a late task's attempt 1 plus its attempt 2. Wave 1 is 4 minutes shorter in 40K35K. But in 45K30K, the late attempt 1 gets more of its tokens after the 12 wave-2 attempt-2 runs end, when only 4-8 streams run and each is faster. The two variants therefore finish within about 2 minutes of each other, while 40K35K has a 7-minute shorter window.

Deterministic check at 22 tok/s with 16 running, and about 38 tok/s with 4-8 running:

- 45K30K: the late attempt 2 ends at about 76.6 min.
- 40K35K: it ends at about 74.4 min.
- Margins: 15.4 vs 10.6 min.

The builder's "worst ~83 vs ~74" overstates the gap.

**Evidence.** Monte Carlo, 400 runs per row:

- 20 tasks, 16 slots, the D-A rule, attempt 2 queued when its task's attempt 1 ends;
- per-stream rate = min(38, aggregate / running), with the real per-attempt spread;
- the time when the last attempt ends, against the window.

| Aggregate | 45K30K (window 92) none / real accepts | 40K35K (window 85) none / real | 2X35K (window 78) none / real |
|---|---|---|---|
| 300 tok/s | p50 93 / 87 min; P(> window) 0.60 / 0.20 | p50 92 / 86; P 0.97 / 0.60 | p50 85 / 81; P 0.98 / 0.79 |
| 350 tok/s | p50 82 / 76; P 0.05 / 0.03 | p50 81 / 76; P **0.22** / 0.08 | p50 76 / 71; P 0.28 / 0.12 |
| 400 tok/s | p50 74 / 68; P 0.00 / 0.00 | p50 72 / 68; P 0.03 / 0.01 | p50 68 / 64; P 0.06 / 0.02 |

**Time budget.** The hard-stop clamp is not the limit when READY plus preflight is ≤ 18 min (45K30K: 115 − 2 − 3 = 110 min) or ≤ 20 min (40K35K: 105 min). If READY is slow (the 10-21 GCSFUSE case, about 20 min), 45K30K is clamped by about 2 min.

**Consequence.** This affects only the commit run's measurement: the last attempt 2 of a late task gets cut at the deadline. It does not affect the hidden rerun.

**Minimal fix.** Give 40K35K the same `(92, 115)`. Its hard-stop end (115 − 5 = 110 min) still allows 18 min of setup. Because `560ed8b8` is already in `PUSHED`, a rebuild writes `_REBUILD.ipynb`; or accept the risk for this commit. For both variants, read the 16-stream aggregate from the 2X35K run first. At 300 tok/s neither window holds.

### F4 — LOW — CONFIRMED: the blocker fix's failure record never reaches the notebook (harmless for the submission; reduced visibility; the test checks a different path)

**Location.**

- `arc2_duck.py:3368-3370`: the failure record is appended to the scheduler-local `records`;
- `arc2_duck.py:3474`: `return records`;
- notebook "CELL 6" line ~603: `E12A.solve_all_r6(...)`, return value discarded;
- "CELL 6" line ~59: `_e12_branch_done_r6`, the only feed of `E12_RECORDS`.

**Scenario.** A session setup fails. The record is never published, so `on_branch_done` is not called and `E12_RECORDS` has no row for that (task, branch).

- **The submission is still correct:** the identity fallback, or the other attempt's rows.
- The report's `branches` table, `b_summary` and `eval_b_attempts` do not show the failure.
- It is visible only as `rev6.scheduler.events[kind == "setup_failed"]` and `rev6.scheduler.setup_failed`.
- `b_summary` prints `worker_errors` but not `setup_failed`.
- Test section 15 builds the submission from the **returned** records, so it does not exercise the notebook's callback path. The result is the same here, because the record has no candidates.

**Evidence.** Reading the code. The CELL 6 call has no assignment; `grep setup_failed` over the notebook finds nothing.

**Minimal fix (optional).** Add `"setup_failed": E12_RUN["rev6_scheduler"].get("setup_failed")` to `b_summary`.

Publishing the record through `publish()` would also work, since `threading.Condition()` uses an RLock, so it is re-entrant. But the notebook callback would then run its 0.3-0.5 s submission write while holding the scheduler lock, so the counter is the better minimal change.

### F5 — LOW — CONFIRMED: the fixed 24k provisional nudge no longer matches the per-attempt budgets

**Location.**

- `arc2_duck.py:1420`: `_arc2_nudge_tokens = E12_R6_NUDGE_TOKENS` (24,000), the same for both attempts;
- `arc2_duck.py:1653-1655` and `2124-2131`: delivery;
- `arc2_duck.py:1635`: commit zone = `left <= reserve + 1024`.

**Scenario.**

- **Attempt 2 at 30k.** Its commit zone starts at 24,880 generated tokens, only 880 tokens after the 24k nudge threshold. An attempt with no transform at 24k gets the nudge and, almost immediately, the commit text; the commit has precedence in the opener (lines 1608-1619). The nudge stage is effectively gone for attempt 2.
- **Attempt 1 at 45k.** The nudge comes at 53 % of the budget, against 69 % in 2X35K and the H6-1 rationale (just before a 28k pause).
- **40K35K.** The fractions are 60 % and 69 %.

This changes behaviour, but it is not a crash. Whether it helps or hurts is unmeasured.

**Evidence.** Reading the code; the arithmetic above.

**Minimal fix (optional).** Either leave it and note it, or scale the threshold per attempt, for example `min(24000, int(0.69 * self._arc2_b_budget))` in B mode. That keeps 2X35K's 69 % and gives attempt 2 a nudge at about 20.7k.

### F6 — LOW — CONFIRMED (inert now): the fair-share pause cap still reads the single global budget

**Location.** `arc2_duck.py:3403`: `cap = max_generated_tokens or 0`.

**Scenario.** If `E12_R6_FAIR_SHARE` is ever turned on, attempt 2's pause point is compared with attempt 1's budget (45k), not its own (30k). The session cap (`done_now`) still stops it at 30k, so the effect is only a wrong "no pause" decision.

Both variants set `E12_R6_FAIR_SHARE: 0` ("CELL 1", line ~328), so `share_tokens()` returns None and this line has no effect.

**Minimal fix.** `cap = s.max_generated_tokens or 0`.

### F7 — LOW — PLAUSIBLE: less headroom for a single long request against the 3,000-s request timeout

**Location.**

- `arc2_duck.py:1740`: request cap = `left - reserve`, so attempt 1's first request may ask for 40,904 tokens;
- notebook `E12_DUCK_REQUEST_TIMEOUT` = 3,000 s.

**Scenario.** With no per-request caps, a reply can run to the cap in one request. In the E14S runs, 114-121 requests per run reached ≥ 20k and 84-98 ended at `length` under the old 24,576 cap.

The break-even rate is 40,904 / 3,000 = 13.6 tok/s (it was 10.3 tok/s at 35k). Real requests with ≥ 8k tokens at ≥ 11 running had p1 = 20.3-21.3 tok/s. Scaled to 16 streams (× 22/27), that is about 16.5 tok/s, or about 2,480 s for a full 40.9k reply. That still fits, but the margin falls from about 1.6× to about 1.2×.

Retraction (F2) or a relaunch prefill storm lengthens the wall time of exactly these long requests. Per auditor 3 L26, a read timeout loses the in-flight reply.

**Evidence.** `rates.py` / the inline analysis of 1,569 + 1,590 real requests. No real request was below the scaled 13.6 tok/s.

**Minimal fix.** None before the run. If the GPU report shows timeouts (`failures` per attempt, or `timeouts_clamped`), raise `E12_DUCK_REQUEST_TIMEOUT` to about 3,600 s.

### F8 — LOW — CONFIRMED: process — the rev 3f/3g package and notebooks were not run on the Kaggle CPU kit (Py3.12)

**Location.** `Experiment Results\Kaggle Runs\E15B CPU kit run1 (foysal, 2026-10-25, kit 78073ae3)\cpu_harness_report.json`.

**Scenario.** The CPU kit ran on `nb_sha` submit `0f0d6f67…` and eval120 `30653201…`, which are the rev 3 2X35K notebooks, with 77 unit checks. The new package `bc26fa16` and the four notebooks were tested only locally on Python 3.14.6 (87 checks pass). The memory rule "Kaggle CPU validation before GPU" was written after a Py3.14-only miss.

The new code uses no version-specific API (`int | None` was already in use), so the risk is low. Meanwhile, the builder already lists both SUBMIT shas as pushed.

**Minimal fix.** Run the CPU kit on `5d8069b9` / `40b36fb7` / `560ed8b8` / `36c5cf15` with payload `b2080421`, or record the accepted risk.

Also make sure the payload dataset version attached on **each** account (foysal and esfersami50) contains `q27x_e15b_duck_b20804213563.bin`. A missing file fails loudly in `q27x_payload`: CELL 1 raises `FileNotFoundError`.

### F9 — LOW — CONFIRMED: stale labels and documents

- **`verify_items.py:141` and `:144`.** The item labels hard-code "92 min for 45k + 30k" and "attempt 1 = 45k, attempt 2 = 30k". They print the same text for the 40K35K notebooks, although the values checked are right for each variant (line 17 picks A1, A2, CW, CB from the name). Fix: format the labels from `A1`, `A2`, `CW`.
- **Notes/09** says the rev 3f build is described in a "Notes/08 addendum / Notes/03". Neither file has a rev 3f/3g section, and the README does not mention 45k/30k or 40k/35k.
- **The builder / notebook comment "40k + 35k: worst ~ 74 min".** By the comment's own method it is about 76; by the simulation, the two variants are within about 2 min of each other (F3).
- **The `SGL_KV_PER_REQUEST` comment "attempt 1 ends ≤ ~60k".** Real data: p90 57-59k, maximum 61-64k (F2).

Fix: update the texts. There is no code effect.

---

## Answers to the questions

### Blocker fix under the scheduler lock

- **Correct?** Yes.
  - `sess()` runs under `cond`, before `running += 1` / `active.add()` (`arc2_duck.py:3361-3382`).
  - On an exception, nothing has been taken: no slot, no `active` entry. The job goes into `failed`, `setup_failed` is counted, an event is recorded, a record is appended, `notify_all()` is called, and the loop `continue`s inside the lock.
  - `threading.Condition()` is an RLock.
- **Can it deadlock stage 2 (`stage1_left`)?** No.
  - `stage1_left` is decremented exactly once per popped stage-1 item, in both the failure and the success path (lines 3372-3373 and 3379-3380).
  - Every other path that drops a popped item is stage-2 only (`prio[0] == 1`): the lazy re-rank, the stage-2 wait, the b1 deferral and the P90 skip.
  - So `stage1_left` reaches 0 once every attempt-1 item has been popped, and the waiting stage-2 items are released. The last failure also calls `notify_all()`.
  - Test section 15 confirms this: `stage1_left == 0`, `heap_left == 0`, the run ends at once, 36 sessions done, no worker error, no straggler.
- **Can attempt 2 be double-queued?** No.
  - Every push of `(tid, 1, "start1")` is guarded by `not in sessions`, `not in queued` and `not in failed`: the failure path (lines 3374-3376) and `stage2_items` (line 3229).
  - A task whose attempt 1 failed has no branch-0 session, so `stage2_items(tid)` runs only after its attempt 2 has run. By then `(tid, 1)` is in `sessions`, and the `resume0` part finds no paused session.
- **Does the failure record reach the final submission? Can it overwrite or hide a real record?**
  - In the notebook, no: see F4. In the returned records, yes, once.
  - It cannot hide a real record. The failed (task, branch) never got a session, so nothing else is ever published for that branch. `publish()` replaces rows only for the same branch, so the failure row survives the other attempt's publishes.
- **`rank()` / `stage2_class()` with no session for the task.**
  - `stage2_class([])` = 1 (no signatures, no progress).
  - The flag is 1, and rounds = 0, since `s_` is None.
  - `g412_defer_b1(…, None)` = False at pop.
  - So the queued attempt 2 is ranked like a task with no evidence. No exception.
- **Practical note.** `BranchSession.__init__` only assigns attributes (`arc2_duck.py:2840-2855`). It can fail only on a KeyError or MemoryError, so the fixed path is nearly unreachable.
  - The realistic setup failures are `mkdir`, `write_runtime_state` and agent construction (prompt, facts, indexed view). They occur in `run()` → `_ensure()` (`arc2_duck.py:2857-2870`), inside the worker's own try/except (lines 3395-3409).
  - That path was already safe in rev 3: the status becomes done, the record is published with no candidates, the slot is released at lines 3420-3427, and `stage2_items` queues attempt 2.

### Each place that uses an attempt's budget

| Use | Code | Right budget per attempt? |
|---|---|---|
| Agent budget | `arc2_duck.py:1410` `_arc2_b_budget = b_attempt_budget(branch)` | yes (45k / 30k; 40k / 35k) |
| Request cap and context budget | `_arc2_apply_cap` 1735-1743, via `_arc2_b_left()` 1630-1631 | yes, per agent |
| Commit zone | `_arc2_b_in_commit` 1633-1635; commit text 1637-1641 | yes, per agent |
| System prompt text | 1392-1393 → `arc2_system_prompt_r6(…, budget=b_attempt_budget(branch))` → `b_system_prompt(text, budget)` 1148-1155 | yes (test: "the system prompt states each attempt's own budget") |
| First opener text | 1601-1602 `R6_TEXT_B["first_budget"]` with `self._arc2_b_budget` | yes |
| Session cap (`done_now`) | `sess()` 3162-3163: `max_generated_tokens_b1` for b ≥ 1; `done_now` 2883-2887 uses `self.max_generated_tokens` | yes; the notebook passes `E15B["E12_B_ATTEMPT_TOKENS_B1"]` ("CELL 6" line ~614) |
| Harvest / partial test | 2949-2969 | uses no budget; runs when `finished or done_now()`, which is per-session |
| Per-attempt record | `_agent_state` 3011 `b_budget = a._arc2_b_budget` | yes |
| Report `at_budget`, `at_budget_with_progress` | "CELL 6" line ~620 `_bud(r)`: branch 1 → B1, else attempt 1 | yes |
| Eval A2 block | "CELL 7" line ~310 `_B_TOK = E15B["E12_B_ATTEMPT_TOKENS"]`, used only on branch 0 (`r0`) | yes (attempt 1 = branch 0) |
| Fair-share pause cap | 3403 `cap = max_generated_tokens` | **global**; inert (F6) |
| Nudge threshold | 1420 (24k fixed) | not budget-relative (F5) |

- **Agreement.** The agent and the session read the same numbers: the E15B block writes `os.environ` before any agent is built.
- **Assertions.** The notebook asserts `E15B["E12_B_ATTEMPT_TOKENS"] == E12_MAX_GEN_PER_BRANCH` and `0 < B1 <= A1`.
- **No other single-global budget consumer.** A grep for `E12_MAX_GEN_PER_BRANCH`, `ATTEMPT_TOKENS`, `35000` and `max_gen` over all cells and the package found only the uses in the table, the CONFIG echo, and the 35000 defaults of `_env_int`, which are overridden.

### Capacity and time at 45k for attempt 1, 16 streams, KV pool ~852-866k

- **Concurrency.** `kv // 50,000` = 17, so the server runs 16 streams. This is unchanged.
- **Wave-1 KV peak.** p50 about 0.76-0.85 and p90 about 0.82-0.91 of the pool. Near lockstep with no early accepts, it reaches 0.94-1.02. Details in F2.
- **Real retraction risk.** It exists only at the end of wave 1 of the commit run (hard tasks). In the hidden run it is lower and happens once.
- **The KV gate does not protect against it.**
- **SGLang behaviour.** SGLang admits regardless of `max_tokens`, because its estimate is clipped to 4,096.
- **Report fields to read.** `gauges` kv maximum, `retract_lines` / `preempt_lines`, `server_queue_gt0_share`, `kv_holds`.

### Commit windows (92 / 85 min) and the 12 h hidden run

- **Commit.** At an aggregate of ≥ 350 tok/s, both windows very probably let all 40 attempts finish:
  - 45K30K: P(last attempt late) 0.03-0.05;
  - 40K35K: 0.08-0.22, relatively tighter (F3).
  - At 300 tok/s, neither holds. The 16-stream aggregate is not yet measured.
- **Hidden.** It always ends on time, because requests are deadline-clamped (`min_admit` 300 s, stop 180 s before the hard stop, join grace 90 s). This is unchanged from rev 3.
  - Attempt 1 on all 240 tasks fits: about 7.1-9.0 h of solving.
  - Attempt 2 gets about 42-91 % of tasks depending on throughput, about 67 % at the measured 340 tok/s (F1).

### Consistency of the two variants

- **SUBMIT 45K30K vs SUBMIT 40K35K.** They differ only in:
  - the markdown title and changes line;
  - the commit window / budget (92/115 vs 85/110);
  - the `E12_MAX_GEN` default (45000 vs 40000);
  - the E15B block comment and its `E12_B_ATTEMPT_TOKENS` fallback;
  - `E12_B_ATTEMPT_TOKENS_B1` (30000 vs 35000).
- **SUBMIT vs EVAL120G4B of the same variant.** They differ only in the commit markdown line and the default profile.
- **Against 2X35K.** The payload name/sha, the module sha pins, the E15B assertion, `max_generated_tokens_b1` and `_bud` are the intended differences.
- **Markdown and profile numbers match the builder tables.**
- **The only stale text** is in `verify_items` and the documents (F9).

---

## Checked, no issue

- **Package files.** `tool_agent.py` and `python_tool_sandbox.py` are unchanged since rev 3. The payload `b2080421` contains `arc2_duck.py` = `bc26fa16` (the working copy) and the unchanged vendor files. The notebook's `_E12_MANIFEST` matches: `verify_items` reports no manifest mismatch.
- **`b_attempt_budget`.** With `E12_B_ATTEMPT_TOKENS_B1` unset it falls back to attempt 1's budget (tested), so 2X35K behaviour is preserved for any caller without the new variable.
- **`b_system_prompt(text, budget=None)`** keeps the old behaviour for callers without a budget.
- **The prompt edits** still match exactly once; the budget is formatted with thousands separators.
- **Prefix cache.** Attempts 1 and 2 now have system prompts that differ in the budget digits, so attempt 2 cannot reuse attempt 1's cached system-prompt suffix. The cost is negligible: attempt 2 starts hours later, when the cache has long been evicted, and its prompt is 2.5-14k tokens.
- **Context window.** For a 45k attempt, prompt + `max_tokens` ≈ overhead + budget − 4,096 ≤ ~67k. That is far below SGLang `--context-length` 155,648 and Duck's fixed 100k window. The context budget (window − cap − margin) stays above the attempt's history, so nothing is trimmed.
- **SGLang admission.** In 0.5.21 the estimate is `extend_input_len + min(max_new_tokens, 4096)`, with the running offset also clipped to 4,096 × `new_token_ratio`. The large first-request `max_tokens` (40,904) does not block or slow admission. The `add_req_state` / `req_states` path with the full `max_new_tokens` is only for `ignore_eos`.
- **KV probe.** It reads `sglang:full_token_usage`, the maximum over TP ranks, not `token_usage`, so evictable radix cache is not counted as used.
- **Harvest and partial test** after the budget: they are triggered by `done_now()`, which uses the session's own cap.
- **Report.**
  - `at_budget` / `at_budget_with_progress` use `_bud(r)`.
  - The eval A2 block's `_B_TOK` applies only to branch 0, which is attempt 1.
  - `b_summary.attempt2_started` counts branch-1 records.
- **Commit time budget.** Hard stop at budget − 2 min; the solve end at most 180 s before it. A 92-min window needs READY + preflight ≤ 18 min (45K30K); 85 min needs ≤ 20 min (40K35K). READY was 12-13 min.
- **EVAL120G4B variants.** Profile unchanged (6.5 h; window = 120 × hidden solve seconds / 240, about 5.8 h).
- **Builder protections.**
  - A notebook whose sha is in `PUSHED` is not overwritten; the builder writes `_REBUILD.ipynb` instead.
  - The 2X35K notebooks cannot be regenerated by accident: `--budgets=35000,35000` gives BTAG `35K35K`, which is not in the window table, so the build stops with a KeyError.
  - `assert 0 < B1 <= A1`.
  - The size limit of 400,000 B holds (about 261 kB).
- **Tests.** `test_duck_b.py` ALL PASS (87); `verify_items` ALL CODE ITEMS PRESENT for all four notebooks.
- **Payload loader.** A missing payload fails loudly with the dataset slug; it does not fail silently.
