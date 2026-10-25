# Experiment 15 — Duck with 2 × 35k independent attempts (planned, opened 2026-10-24)

**Design (user decision, 2026-10-24, final):** "B: 2 × 35k independent attempts per task. The first attempt runs on every task first, then the second."
- This is an explicit approval for this design below the 60k-per-request floor.
- Attempt 2 is an independent Duck attempt (a fresh sample), not a resume and not the old indexed-view branch 1.

**Why:**
- The three EVAL120 G4B runs (2026-10-24; Exp 14 `Notes/EVAL120 analysis 2026-10-24/` A, B, C and GPT Notes/44-46) showed:
  - branch 1 gave 0 correct programs from 22-26 % of the tokens;
  - the 28k pause plus resumes caused 58-66 % of all prefill;
  - the 24,576 request cap cut 84-98 requests with no tool call;
  - two independent xhigh runs are complementary: 61 queries together vs 46 and 41 alone.
- The base was E14S (LB 22.78; EVAL120 XH 25.83 %, XH+E16 30.00 %).

**Out of scope here:**
- E15 engine experiment: off (user, 2026-10-24).
- Engine changes: the engine stays as E14S (SGLang 0.5.21, RedHat INT4, DFlash2 W6), **except** the user's request of 2026-10-24: the delayer is disabled and real 14-16 streams are used (Notes/02 items 36-44).
- The solver-design arms from GPT Notes/44-46: listed as later items in Notes/01.

**Rules for this experiment:**
- No Kaggle push, run or submit without the user's go.
- Kaggle CPU kit on the exact sha before any GPU.
- A written capacity check (tokens, time, quota, stop point) before any GPU run.
- The S1-60K Poetiq notebook (LB 34.58) stays the fallback submission.

## Notes

| File | What |
|---|---|
| `Notes/01 Claude - prioritized fix list for config B (2026-10-24).md` | Rev 1 of the fix list (restored on 2026-10-24 after an accidental delete); superseded by Notes/02 |
| `Notes/02 Claude - fix list rev 2, with real 14-16 streams (2026-10-24).md` | **Current fix list:** rev 1 in full plus the delayer fix and 14-16 streams (items 36-44), the smoke step, D9, the capacity check at 16 streams |
| `Notes/03 Claude - E15B build record and item status (2026-10-24).md` | **Build record (rev 2):** every Notes/02 item built and checked one by one (`src/verify_items.py`); unit tests 64/64; CPU harness commit 23/23, rerun 25/25, eval120 27/27; known risks. Audit pack: `Experiment Review/E15B_PRE_GPU_AUDIT_FOR_GPT_2026-10-24_rev2.zip` |
| `Notes/04 AI fix proposal 1 of 2 (pasted 2026-10-25).md` | AI fix proposal 1 for items 1-58, verbatim |
| `Notes/05 Claude - proposal 1 vs the E15B build (2026-10-25).md` | Item-by-item comparison with the build; 4 report-only changes worth adopting (A1-A4); decision D-A |
| `Notes/06 AI fix proposal 2 of 2 (pasted 2026-10-25).md` | AI fix proposal 2, verbatim |
| `Notes/07 Claude - final adopt list from proposals 1 and 2 (2026-10-25).md` | **Final adopt list A1-A7** (report metrics, min_admit 300, rule-library hardening after measuring 0/120 on eval, watchdog fault-injection test); decision D-A |
| `Notes/08 Claude - rev 3 decisions and the three agent audits (2026-10-25).md` | **Rev 3:** user decisions (D-A start barrier, 20-task commit with both attempts), the triage of all three agent audits (taken / rejected with reasons) |
| `Notes/agent audits/` | Auditor 1 (Duck package), Auditor 2 (notebook wiring and engine), Auditor 3 (regression ledger of 135 past issues) |
| `Notes/09 GPT - final pre-GPU audit of rev 3 and the blocker fix (2026-10-25).md` | GPT's final audit (verbatim): the scheduler setup-failure BLOCKER and its fix + test |
| `Notes/10 Claude - 2X35K GPU run findings and rev 3h (2026-10-25).md` | the 2X35K GPU run findings (stopped at 41 min), the rev 3f/3g runs (40K35K submitted, 45K30K on Sami), rev 3h fixes and shas |
| `Notes/11 Claude - 45K30K and 40K35K GPU commit results (2026-10-25).md` | 45K30K rev 3g commit 19/32, task-norm 65 %; 40K35K commit cancelled (submission pending); findings F1-F9; hidden projection 8.8 h; next = R3H 45K30K |
| `Notes/12 Claude - scheduling rules validated on our multi-attempt runs (2026-10-25).md` | ORDER2 / class order validated; stage 3 value; skip replay; time-sliced replay |
| `Notes/13 Claude - rev 3i R + stage 3, EVAL120 6-h notebook (2026-10-25).md` | rev 3i build (R + stage 3), EVAL120 6-h notebook 2c8eb52e, tests |
| `Notes/14 Claude - rev 3j two order notebooks, EVAL120 6 h (2026-10-25).md` | **Current:** R3J1 (a3 before accepted a2, 74a0eae5) and R3J2 (accepted a2 before a3, 5c44f752), EVAL120 6 h, tests |
