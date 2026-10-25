# 37 — 4-day review: what was missed, consolidated from agents A-D (Claude, 2026-10-24)

User request (2026-10-24): "আমাদের ৪ দিনের conversation পড়, আরো কিছু বাদ গেলো কি না… notebook ও পড়বে, আর সব resources" ("read our 4 days of conversation and check what else was missed… read the notebooks too, and all resources"), using 4 agents.

Full reports are in `_scratch/review_4day/`:
- `A_conversation.md`: 103 user instructions, 28 promises;
- `B_notes.md`, plus an appendix of ~3,800 rows;
- `C_notebooks.md`;
- `D_resources.md`.

## Corrections to earlier claims
- **"Long band ≈ 8.7k context" (Notes/30, 33, 34) is wrong.** The R2 long prompts had a median of 40,877 tokens. `ctx_per_req_p50` counts KV pool occupancy, and the 36k shared prefix is counted only once in it. Width 6 / width 4 never ran the long band, so width 6 at long context is still open (agent B).
- **nodelay crash:** it passed the parity gate. Only the server-log signs caught it (B).
- **"vLLM fallback" (Notes/31) does not exist.** `E11_ALLOW_VLLM_FALLBACK` has no code behind it. If SGLang fails, the submission is the identity file, LB ≈ 0 (C).
- **Duck admission uses KV 0.95.** The 0.85 gate is only on the Poetiq path (B).
- **E16 notebooks were not run through the Kaggle CPU kit.** They were checked with the local harness only (D).

## The 11-of-12 loss (user, 2026-10-24: "4 days of optimizing gave 5 %, and 11 streams cost 9 %")
- The submitted E14S runs 11 of 12 under churn: 103 of 108 decode lines show 11 running.
- R2 found the cause: DFlash's min-free-slots delayer.
- The only fix tested (nodelay) crashed at 13 running. Two safe fixes were never tried, and adding them was my job.
- They are now arms in E15:
  - `w6_mrr14`: delayer kept, mrr 14, graphs 1..14;
  - `w6_nodelay_mrr12`: delayer off, mrr 12.
  - Both get a churn latch test.
- **KV is mostly empty** (user: "KV 0.18-0.41 used"). New E15 arms run DFlash W6 at 14 / 16 / 20 real streams:
  - mrr 17 / 20 / 24 (delayer threshold min(4, max(2, (mrr+5)//6)));
  - mamba cache 4 × mrr;
  - graphs 1..mrr;
  - latch test and a long probe on each.

## Ranked open items

| # | Item | Expected impact | Effort | Source |
|---|---|---|---|---|
| 1 | **Engine correctness at length and concurrency.** The c12 × 1,024 reasoning-on probe: reference vs target-only, and at 14/16/20 streams. SGLang vs vLLM with the S1 prompt (29.17 vs 34.58 has never been explained) | −5 to +5 LB, decides the engine | probe added to E15; A/B under 1 day | D B1-B2, B #7, GPT 32/33 |
| 2 | **More real streams** (12 → 14-20) and the 11-of-12 fix | +8-25 % throughput (to measure) | E15 arms added; the Duck needs a `--streams N` build option | user, C M2, B #3 |
| 3 | **Hybrid:** Duck stage 1 + S1 iteration on unverified tasks; Duck branch 1 off | +2 to +6 LB | 1-2 days | Notes/36, D A1, B #1 |
| 4 | **MTP engine (E15)** | +31-43 % KV, 0 to +4 LB | run E15 | Notes/35 |
| 5 | **Engine watchdog / relaunch mid-run (H1)**, plus a launch fallback with the W1 flags (H2) | removes the "dead server = rest of 12 h lost" and LB ≈ 0 paths | under 1 day | C H1-H2 |
| 6 | **Earlier first check()** (12 of 17 never checked). E16 is under test (EVAL120 pair) | +? | after EVAL120 | A, B #2 |
| 7 | **CPU rule library** in the identity slots | +0.5-1.5 LB, no downside | under 1 day | D A7 |
| 8 | Chunked prefill 2048; Mamba checkpoint protection | +0-15 % / up to +14 % | arm / 1-2 days | B #4-5 |
| 9 | **Startup:** JIT tar fix + restore + known-answer check; wheelhouse as a Kaggle Model; the 12 h margin from container start (~300 s) | ~+0.3, lowers 0.00 risk | under 1 day | A, B #12, D C1 |
| 10 | M1: isolation INCONCLUSIVE locks 6 workers (needs user approval); M4: self-tests use the first hidden task; M5: scheduler worker loop partly outside try/except | risk | under 1 day | C |
| 11 | NVARC union | +3 to +8 | over 1 day | D F1 |
| 12 | 2×TP2 boot test; vocab-parallel logits; FP8 KV scales; SGLang 0.5.22; Flash-Next Duck | various | over 1 day | A, B, D |

## Rules and process
- **Token floor:** under the hidden time share E14S gives about 49k tokens per task. The 240-task simulation puts 89-110 tasks under 60k. The Duck exception to the 60k floor assumed total thinking would stay far above 60k, so this needs the user's view (D M1).
- **Cancel API:** untested. The user asked for it on 10-23 (A #49).
- **Deadline 2026-10-26:** every team member must have accepted the rules (user action).
- **Payload datasets:** keep `q27x_e12_duck_6f48b81af59a.bin` and `q27x_e14_modules_78029e146cd3.bin` in every account's payload until the pending hidden run is scored (C M6).
- **Kaggle CPU kit:** run it on the exact sha that will be pushed (C L6).
