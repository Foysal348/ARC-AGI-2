# GPU EVAL12: Q27-AGENT v1.9.1 (foysalemonshanto/q27-agent v3, 2026-10-12)

**Run:** 4× L4, internet off, mode `EVALUATION (12 tasks)`.
- Status COMPLETE: 13:42 → 14:38 local, about 56 min.
- Breakdown: install 171 s, server ready in 510 s, synthesis 2,538 s.

## Scores (official = correct test outputs / 22)

| | Hits | Official |
|---|---|---|
| Agent (synthesis) | 1/22 | 4.55 % |
| Recorded DIRECT (run 7) | 3/22 | 13.64 % |
| Hybrid (DIRECT first, agent second) | 3/22 | 13.64 % (no gain) |

**Statuses:** 1 `verified_usable`, 9 `train_mismatch`, 2 `no_program`.

**The single verified program** was for 136b0064: train 3/3 and test correct. DIRECT had also solved this task, so the agent added nothing new.

## Root cause: `thinking_token_budget` was NOT enforced

**Per-task numbers** (from `attempts.jsonl`, thinking counted as completion − tokens(content) via `/tokenize`):

| Task(s) | Thinking | Result |
|---|---|---|
| 10 of 12 | **57,344** (= `max_tokens`) | `finish=length`, no content at all: the server never closed thinking at 49,152 |
| 4c3d4a41 | 56,238 | closed late, the answer was cut off |
| 136b0064 | 49,982 | closed naturally (just past 49,152), `finish=stop`, program verified |

**What happened as a result:**
- The 10 cut-off tasks went to the no-thinking commit request, which wrote a program from the thinking tail.
- 8 of those programs failed training (`train_mismatch`); for 2 tasks no program came back at all.

**Why the budget was not applied**
- The request was accepted, and `budget_contract_smoke` passed. But the smoke test only checked the **request shape** (`max_tokens` 64); it never checked that thinking actually closes at the budget.
- **Likely cause:** with MTP speculative decoding (×3), vLLM 0.19 does not apply the thinking-budget logits processor. Speculative decoding skips non-argmax-invariant logits processors. This is a hypothesis, not yet verified.

**Contrast with LB 26.94:** v1.4.4 never relied on `thinking_token_budget`. It closes thinking itself:
1. Phase 1 runs with `max_tokens` = 49,152.
2. If no `</think>` appeared, it appends Qwen's stop sentence + `</think>`.
3. Phase 2 generates the answer with 8,192 tokens.

## Infrastructure (healthy)

- **KV:** 846 blocks × 1,600 = **1,353,600 tokens**.
  - Peak usage 68 % (p95 65 %) with 12 sequences running at about 60k tokens each. There is room for more.
  - 0 preemptions.
- **Queue:** up to 12 running and up to 5 waiting.
- **MTP:** acceptance 0.634.
- **Throughput:** 712,767 generated tokens in about 2,400 s, roughly 297 tok/s overall and about 25 tok/s per stream.

## Fix for v1.9.2 (proposed)

1. **Close thinking the same way as v1.4.4**, using raw `/v1/completions` (the prefix cache reuses the long prefix):
   - phase 1 with `max_tokens` = 49,152;
   - if `</think>` is missing, append the stop sentence + `</think>` and run phase 2 with 8,192 tokens.
2. **A real enforcement smoke test:** run one short request with a forced close and check that the content arrives and the program parses. Reject the server rung if it fails.
