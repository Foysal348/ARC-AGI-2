# E28 K100 EVAL120 6 h: audit request (GO / NO-GO for one 6-h GPU commit on 4x L4)

You are auditing ONE Kaggle notebook before a GPU run: `Q27-XHIGH-E28_K100_EVAL120_6H.ipynb` (sha256 in
`e28k100_build_facts.json`). Read every file in this zip. Answer with **GO** or **NO-GO**, then a numbered list of
**blockers** only (a blocker = something that makes the run crash, produce no / wrong answers, lose the submission, run
past the time limit, or silently not do what the user decided). For each blocker give the file, the line, the evidence and
the exact fix. Put non-blocking remarks in a separate short list. Do not propose new features or new experiments.

## What the notebook is
- **Base:** the K100Q notebook (sha 4506a71f) that ran VALID on GPU today (klee100 3bpw-MTP Flash-Next on Merlin vLLM 0.32,
  TP2 x PP2, FP8 KV + static QSA scales, MTP K3, Q-defer, pin 7,933,526,016 B -> pool 479, F 6, 21 workers; 23 / 30 tasks,
  385 tok/s, 0 preemptions). `build_e28_k100.py` rebuilds the K100Q cells byte for byte (asserted), then applies the marked
  "E28 K100" edits (list in the builder docstring). `e28k100_review_diff_vs_k100q.diff` = every changed line.
- **User decisions to verify (README.md):**
  1. Commit profile `E28EVAL120`: all 120 eval tasks, a fixed 6-h solve window, HARD_STOP at 6 h 50 min, the deadman at
     HARD_STOP + 5 min. The hidden rerun stays as proven (every test task, identity fallback, 12 h - 3 min, E18 ladder).
  2. I8 = `DRAFT_HEAD=int8` (measured on W4 only). A boot that fails with it must retry at the SAME pin with the BF16 head
     (`I8_UNSUPPORTED_ON_K100`); the pin 7.93 GB and 21 workers stay.
  3. Thinking (user decision, round 2): T1 = 104,000 per attempt (= K100Q, reserve 4,096) for attempt 1, the B1 re-runs
     and attempt 2; B2 re-runs 120,000; the server stays at K100Q's `--max-model-len 131072` (B2's Duck window = the
     normal 130,560); request timeouts T1 7,200 s, B2 9,000 s (`E28_B2_REQUEST_TIMEOUT=9000`).
  4. The A / B / C scheduler (`e28_sched_patch.py`, owned by another agent; API in `E28_SCHED_API.md`), no ORDER2.
  5. Full saved responses; dual mode (E18 R2 CELL 0 verbatim, the E15B writer); notebook < 400,000 B.
- **Decisions taken in this build (check them):** admission = "slots" with 21 workers, as the proven K100Q run (the engine
  schedules KV; a preemption is a recompute, counted and flagged per attempt in the report); a T1 attempt's
  projected peak = min(131,072, 8,192 + budget + 8,192) (the scheduler's N1 / N2 / N4, verbatim); e21_cell7 packed as a
  sha-pinned lzma + base85 literal (size); scheduler failure = NO-GO in the commit, the original scheduler in the rerun.

## Questions
1. Does every user decision above hold in the notebook text (`e28k100_code_export.py` = the readable cells)?
2. (Fixed after audit round 1: a real boot with the INT8 head must read back `I8_ENGAGED`, else
   `I8_READBACK_UNVERIFIED` stops that server and the same pin is retried with the BF16 head.) Can the I8 drop rule loop,
   skip the BF16 retry, or lower the pin before the BF16 head was tried? Does the e18 ladder cap
   (3 tries in the rerun) still leave a BF16 try at the first rung?
3. In "slots" mode, are the scheduler's N1 (C-9 peak; B2 needs 216,384) / N2 (C-7 above-pool check, 1 slot) / N4 (a
   non-blocking slot) edits inert or correct? Can a slot leak (a key taken by the N4 fit check but never released)?
4. Is anything still sized for the dropped 262k plan (B2 120,000 must fit 131,072 with an 8k prompt: 8,192 + 120,000 +
   8,192 = 136,384 > 131,072, so a long B2 ends CONTEXT_EXHAUSTED near 129k: check this is handled as a normal end)?
5. The time plan: can the solve window, the report and the bundle finish before HARD_STOP (410 min) and the deadman (415 min)
   with one or two failed boot tries?
6. The rerun path: does anything new (scheduler, I8, report) make the hidden rerun fail closed or lose the submission?
7. The CPU kits (`q27-e28-k100-kit-*.ipynb`): do their checks prove the above, and what would they miss?

## Known open risks (already reported to the user)
- I8 on klee100 is untested on GPU (static check + the CPU quantizer probe only); its extra PP1 memory at the 7.93 GB pin.
- B2 at 120,000 inside 131,072: a B2 attempt with a large prompt / tool history reaches the context end before its budget.
- "slots" lets the engine schedule KV: with 120k attempts and 200k B2 re-runs, KV can fill and vLLM preempts (recompute);
  K100Q at 104k had 0 preemptions (KV max 99.4 %).
- The scheduler module was still being finished when this zip was made (the no-idle backfill rule); this build embeds the
  version on disk at build time (sha in `e28k100_build_facts.json`, "sched").
