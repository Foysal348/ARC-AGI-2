# Experiment 28: Flash-Next EVAL120 6 h, W4 vs klee100 (living plan)

## Goal
Two 6-hour EVAL120 runs, one per Flash-Next family, each with every tested speed lever. They run at the same time, on two accounts.
- The **winner** goes into the DFlash2 experiment (Exp 27) and decides the final submission path.
- The deadline is 2026-11-02 23:59.

**User decisions (2026-11-01):**
- Run both, together. W4 is not dropped.
- Include C12 and I8 as well (user decision, even though they were measured only at 8k / 40k).
- Raise thinking; the amount is pending (see section 4).
- DFlash2 work (Exp 27) continues in parallel, but these two notebooks come first.

## 1. The two notebooks

| | NB-W4 (from NB6Q) | NB-K100 (from K100Q) |
|---|---|---|
| Checkpoint | Merlin W4A16 (p1-p5) + FP8 PLE overlay | klee100 3bpw-MTP (W3 g128 / W2 g64 routed, humming) |
| KV | FP8 E4M3, static scales | FP8 E4M3, static scales (W4 sidecar reused: EXPERIMENTAL_REUSED_CALIBRATION) |
| Speculation | MTP K3 probabilistic | MTP K3 probabilistic |
| Q-defer (nb4_ppack rev 2) | ON (NB4S: +21.1 % 8k, +19.8 % 40k, oracle EXACT) | ON (K100Q: ENGAGED, exec share 0.9976) |
| C12 (12 streams) | ON (NB4T: 387.9 vs 302.9 at 8k@8 = +28 %, 374.6 vs 287.4 at 40k = +30 %; KV 0.99 at 40k; long bands unmeasured) | n/a (already 21 workers) |
| I8 (`DRAFT_HEAD=int8`) | ON (NB4T: +3.4 % 8k noisy, +11.4 % 40k; needs pin 3.29 GB) | ON (untested on klee100) |
| Pool / workers today | 214-215 blocks, 9 workers (pin 3.55 GB) | 479 blocks, F 6, 21 workers (pin 7.93 GB) |
| Other proven pieces | #60490, `MERLIN_SHARE_DRAFT_WORKSPACE`, prefix caching + align, `--language-model-only` | the same |
| Eval + private format | E18 R2 CELL 0 verbatim, E15B writer, rerun = every test task (NB6 copy) | the same |
| Budget | 6 h EVAL120 commit profile (new) | 6 h EVAL120 commit profile (new) |

- **Open:** C12 + I8 together on W4 have never been measured. I8 lowers the pin, so the pool drops below 214. At 100k+ contexts the admission gate will hold back most of the 12 streams; it waits, it does not preempt.

## 2. Measured base data (2026-11-01)

**Speed:**
- W4 Duck 225 tok/s (NB6, no Q-defer, 9 workers).
- klee100 + Q-defer, real Duck (K100Q, 30 tasks): window average **385 tok/s**, all-busy median 414; 0 preemptions; prefix hit 85.4 %.

**Quality, task level:**

| Run | Tasks | a1 right | a2 right | Both | a1 only | a2 only | Neither | Solved (pass@2) |
|---|---|---|---|---|---|---|---|---|
| klee100 K100Q, all 30 | 30 | 22 | 20 | 19 | 3 | 1 | 7 | 23 |
| klee100, NB5's 15 | 15 | 12 | 10 | 10 | 2 | 0 | 3 | 12 |
| W4 FP8 (NB5), 15 | 15 | 13 | 12 | 12 | 1 | 0 | 2 | 13 |
| W4 BF16 (NB5), 15 | 15 | 13 | 11 | 11 | 2 | 0 | 2 | 13 |

**Query level:**
- klee100: 31 / 38 (pass1 30, pass2 1).
- W4 FP8: 18 / 20 (pass1 17, pass2 1).
- W4 BF16: 18 / 20 (pass1 18, pass2 0).

**Attempt level, train vs test:**
- klee100, 60 attempts: 54 train-pass accepted, of which 41 right and **13 wrong**. 2 had a train pass but were cut before submitting; 4 never passed train.
- W4 FP8, 30 attempts: 26 train-pass accepted, of which 25 right and **1 wrong**. 4 never passed train: 3 cut by the 4,050 s window and 1 by the budget.
- On NB5's 15 tasks, klee100 had 8 train-pass-but-wrong submissions against W4 FP8's 1.

**Caps:**
- The 104k attempt budget / 131k window cut 3 of 90 attempts: klee100 16b78196 a2 (context exhausted at 95,825), klee100 20270e3b a1 (103,427), W4 2b83f449 a1 (103,761).
- The time window cut 7.

**Token use:** klee100 used ×1.44 W4's tokens on the same 15 tasks.

**Reading:** almost every solve comes from attempt 1, and attempt 2 rescued 1 task in 30. So coverage of attempt 1 on every task matters most.

## 3. Scheduling policy (user rule, 2026-11-01): replaces the plain b0 / b1 order
**Phase A: attempt 1 for every task.** Attempt-1 thinking **T1 = 120,000** (user decision). **No ORDER2** (user 2026-11-01: "order2 আর লাগবে না"), so tasks run in plain order.

**Phase B: re-run the attempt-1 tasks whose train is not correct.** Two categories, from the attempt-1 record:
- **B1, self-submitted but train wrong:** the model ended on its own (final answer / commit) with the best train match < 100 %. Re-run with the **same thinking T1**.
- **B2, cut by the cap:** budget end, context exhausted or a forced commit at the budget. Re-run with **200K thinking**. This needs a server max-model-len of at least about 204,800 (Flash-Next supports 262,144), so the server launches with max-model-len 262,144 for the whole run.
- Tasks cut by the time window are not B1 / B2. They are unfinished and are resumed or re-queued as in Phase A.

**Phase C: attempt 2.** Priority order:
1. tasks whose train is **still** not correct after Phase B, with thinking T1;
2. then the tasks that self-submitted with train correct, in **ascending order of attempt-1 thinking used** (≈30k first, then ≈40k, and so on).

**Answer slots:**
- **User decision:** the best train candidate goes to `attempt_1`. A Phase-B re-run with a full train match replaces attempt 1's answer, and attempt 1's old answer becomes the `attempt_2` fallback. If neither matches fully, the higher train match goes to `attempt_1`.
- `attempt_2` = the Phase-C answer, falling back to the next-best distinct candidate.

**Data behind it (K100Q + NB5, 90 attempts):**
- Every accepted attempt had a 100 % train match.
- No attempt was "self-submitted with train wrong", so B1 is expected to be rare.
- B2 cases: klee100 16b78196 a2 (95,825, context), klee100 20270e3b a1 (103,427), W4 2b83f449 a1 (103,761).
- Train-pass-but-test-wrong: klee100 13, W4 FP8 1. Phase C's ascending order covers these.

## 3b. ORDER2
Not used (user 2026-11-01). The A/B/C policy above replaces it.

## 4. Thinking (DECIDED 2026-11-01)
- **Attempt 1, the B1 re-run and the Phase-C still-train-fail attempts:** T1 = 120,000 per attempt (reserve 4,096, so the longest single request is 115,904). This is within the code assert (≤ 120,000) and the 120k rule.
- **B2 re-run (cap-cut):** 200,000. This needs the server at max-model-len 262,144 (Flash-Next's native maximum) and the Duck window raised for B2 attempts only. The assert `max(NB5_ATTEMPT) ≤ 120000` gets a B2 exception, by user decision.
- **Capacity check:** a 200k request needs about 49 + F blocks.
  - klee100 pool 479: a few B2 attempts fit alongside the other streams.
  - W4 pool 214, smaller with I8: each B2 attempt takes about 25 % or more of the pool.
  - The projected-peak admission gate handles it (it waits, it never preempts).
- **Untested:** 200k-context correctness on Flash-Next (RC2 validated 120k). Record it per B2 attempt.

## 5. Steps
1. ~~Thinking decision~~ DONE: T1 120k, B2 200k, no ORDER2, best-train → attempt_1.
2. Build NB-W4 and NB-K100 (parallel agents, split by file ownership). Each has the 6 h EVAL120 profile, the levers in section 1, the ORDER2 rule, the full saved responses and the dual-mode format.
3. CPU kits on Kaggle (units + real + e2e commit / rerun), then audit zips.
4. GPU: two 6 h commits on two accounts. **Needs the user's go.**
5. Analysis: score, tok/s, coverage, attempt table, train vs test. The winner goes to Exp 27 (DFlash2).
   - **Mandatory checks (GPT audit 3, 2026-11-01):**
     - `tasks_with_request_sent`;
     - phase-A coverage (tasks whose attempt 1 started / finished);
     - B2 routing (every B2 has a cap verdict or explicit budget evidence, 200k plan, window 261,632);
     - preemptions (run level and per attempt);
     - the final `submission.json` (240 ids, schema, identity fallback);
     - `actual_window_sec` (compare the two runs on the real exposure, not the planned 21,600 s).
   - **Hidden-rerun note (audits 1 and 3):** RERUN_PROD in both notebooks uses the E28 levers (A/B/C scheduler, I8, T1 120k / B2 200k, max-model-len 262,144). It is not byte-identical to the proven K100Q / NB6 rerun. CPU e2e-rerun kits pass, but GPU evidence comes only from the 6-h commit. Submit a notebook only with the configuration its commit measured (user decision pending).

## Change log
- **2026-11-01: user decision, klee100 workers 24** (measured 460 → 562 tok/s at 20k, 16 → 24 streams; the C12 analog), after a full lever review of Exp 25. The only other measured-but-absent lever, S (`CUDA_DEVICE_MAX_CONNECTIONS=32`, +11.6 % alone), was not added: it overlaps with Q-defer and S+QD was never measured.
- **2026-11-01: user decision: T1 104,000, B2 120,000, server max-model-len back to 131,072** (no 262k). Data: of the 90 attempts, 0 of the 5 above 95k tokens were right, and a 200k B2 (about 2 h 20 min) would mostly not finish after phase A. Request timeouts T1 7,200 s, B2 9,000 s.
- **2026-11-01: user rule, no idle worker.** The scheduler's "hold a later phase while an earlier one waits for KV" is removed. Waiting B2 items are checked first at every admission, and a free worker backfills the next item that fits. B2 wait and idle-worker seconds are recorded.
- **2026-11-01:** user rules: the A/B/C scheduler, T1 120k, B2 200k, no ORDER2, best-train → attempt_1.
- **2026-11-01:** folder created. Base data from K100Q (Exp 26) and NB5 QUALITY (Exp 25).
