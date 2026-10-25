# ARC-AGI-2 discussions + leaderboard (read 2026-10-21): what others reach with Qwen3.8, and what it implies for us

**How this was read:**
- The kagglesdk `DiscussionApiClient.get_topic` / `list_comments` work with our token. `list_topics` on the competition forum returns 403, but a site-wide `search_query` finds the threads.
- The full threads are saved next to this file (`topic_*.md`).
- Leaderboard history (no method info): `D:/ARC AGI 2/_scratch/lbmon/arc2_history.json`, from `tonghuikang--arc3-leaderboard-monitor-get-history.modal.run?comp=arc2`.

## Scores with Qwen3.8 (or a likely Qwen3.8) on the public LB (120 tasks; the rerun covers 240)

| Team / person | LB | Evidence of the method |
|---|---|---|
| **Daniel Franzen** (author of the Duck fork we vendor) | 29.86 (Apr) → 46.67 (10-14) → **57.64** (10-19) | No public ARC-AGI-2 notebook. His ARC-AGI-3 Milestone 2 = Duck + Qwen3.8-27B, then Qwen3.8-Flash-Next (`dfranzen/arc-agi-3-milestone-2-solution`). The ARC-AGI-2 method is not published |
| **mikelou1** | 28.61 → 36.67 (10-11) → 42.50 → 45.14 → **66.39** (10-17) | topic 748082: "Qwen scored ~25 %… 36 %… a tiny **kernel issue** made my program score only 40 % (instead of the 60 % projected by CV)… UPDATE: It scored 66 % (~60 % on CV)". "Not a NVARC ensemble." Closed source. The model (27B or FN) was asked about but not answered |
| **NguyenThanhNhan** | 32.78 → 41.25 (10-17) → **52.08** (10-18) | topic 745787: 27B FP8 + EAGLE, xhigh, pure reasoning with a forced commit at 49k gave ~23. "Pure reasoning … quite token-inefficient … you could solve many more tasks within 12 hours by implementing a **simple harness**." His SGLang flags are in the thread |
| **Hiểu Vy** | 34.86 → 44.03 | topic 748082: "26.81 with Qwen 3.8 27B and 8.61 with Qwen Flash. On the 120 validation tasks, Qwen 3.8 27B achieved **80 %**". Best = an NVARC finetune. Whether the 80 % was within 12 h was asked, not answered |
| artificialagencylab.com | 30.97 → **65.56** in 10 submissions (Oct) | method unknown |
| mtg | 40.56 → **54.03** in 5 submissions | method unknown |
| Adarsh (`madarshbb`) | 15 → 21.53 (re-run 13) | public Qwen3.8 REPL agent notebook |
| Top 10 | 70.8-88.1 (Tufa Labs 88.06, Yi-Chia Chen 84.72, rabbithole 80.83 …) | methods unknown |
| **Us (Hack2Publish)** | 34.58 (S1-60K Poetiq) | — |

## Other signals

- **amn (748082):** "FN with an acceleration (e.g. quant?) might reach a better score … by letting it work as an agent."
- **TheDivine Aurora (746149):** the October jump = "an inference setup with maximum throughput to already extract the existing model's ability".
- **Anil Thomas (745787):** xhigh median about 64k per generation; 240 puzzles (about 330 queries) is the hard part.
- **The second number in the monitor history is not the run time.** It reaches 2,000-3,000 for 12 h runs, so it is probably minutes from submission to score, including the queue. Do not use it for coverage.

## What it means for us (arithmetic)

- **Token budget for full coverage:**
  - Solve time ≈ 11.3 h at our measured ≈ 330-430 tok/s gives about 13-17 M generated tokens.
  - That is about **55-72k tokens per task** (both attempts) for 240 tasks.
- **Rev 4 EVAL6:** about 200k per task (2 branches x about 100k, the soft cap), i.e. **≈ 3x over budget.** Concurrency adds little on this engine: 6 → 12 running gives 368 → 431 tok/s.
- **Gap to close:** tokens per task (≈ 3x) and/or throughput.
- **Candidate levers (to be measured, not decided):**
  1. The rev 5 waste fixes (restarts, dead 60k replies, early `check`), measured by the running EVAL6R5.
  2. Flash-Next in the Duck harness (E13, built). It is the path of Franzen's ARC-AGI-3 work and of amn's comment; 6B active parameters means potentially several x the decode speed if it runs correctly on L4.
  3. A per-task token budget sized for 240 tasks (one branch first, the second branch only for unsolved tasks) and the ORDER2 order.
