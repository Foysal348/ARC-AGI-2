# Experiment plan — 2026-10-04 → 2026-11-02

Quota on 2026-10-04 09:49 BST: esfersami50 27.5 quota-h, foysalemonshanto 23.5 quota-h (both reset 2026-10-10 06:00 BST).
4×L4 costs 2 quota-h per wall-hour. One leaderboard submission per day. Each experiment has its pass/fail rule fixed
before it runs; a step is repeated at most twice before moving on.

| # | when | experiment | quota-h | gate |
|---|---|---|---|---|
| 1 | Oct 4 | **EVO24**: evolution over verified programs, same Qwen3.8-27B-FP8, 24 frozen tasks | ~3 | STRONG GO (≥5 solved, ≥3 NVARC misses) → 2 and 4; PROMISING → one tuned rerun; WEAK (≤2) → 1b |
| 1a | Oct 4, no GPU | stage ThinkingCap-Qwen3.8-27B-FP8 (31 GB) as a Kaggle model | 0 | — |
| 1b | only if WEAK | failure taxonomy of every EVO candidate (wrong rule / right rule wrong code / fixes one pair breaks another / train-fit wrong test / no code); redesign prompts using Nemotron-SFT-ARC-AGI successful traces | 0 | rerun EVO24 |
| 2 | Oct 5 | ThinkingCap smoke, 2–4 tasks: loads, kernel path on L4 (block-FP8 vs Marlin fallback, open vLLM issue for SM89), tok/s vs stock, sane output; also MTP speculative decoding throughput | ~1–1.5 | not slower than stock → 3 |
| 3 | Oct 5–6 | ThinkingCap EVO24 A/B: identical tasks, seeds, budget | ~3 | more candidates per wall-hour and ≥ same solves → keep ThinkingCap |
| 4 | Oct 6–9 | NVARC instrumentation: per-task confidence (candidate scores, augmentation agreement), measured runtime, used to route tasks to the agent | ~4–6 | runtime and confidence quality known → 5 |
| 5 | Oct 10–13 | best agent on all 120 eval tasks; union with the frozen NVARC map = projected combined score | ~12–16 | projected > 32.64 by a clear margin → 6 |
| 6 | Oct 13–15 | combined 12 h submission notebook: NVARC → triage → agent on uncertain tasks → pass@2 arbitration; first real submission | ~12/run | iterate on LB |
| 7 | Oct 16 → Nov 2 | tune agent time share, candidates, pass@2 selection; daily submissions. Nemotron SFT/LoRA only if the agent branch is stuck | — | — |

No submission before step 4: the current PIV would score 1.67 again.
