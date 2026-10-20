# E8 R5 quick check: blockers and serious bugs only (2026-10-13)

Notebook: `Q27-XHIGH-v1_4_4_E8_PER_UNIT_R5_FINAL_CPU.ipynb` (sha `16c6c688…`). CPU only; nothing ran on a GPU.

## Verified on CPU

| Check | Result |
|---|---|
| GPT's tests (`test_r5_cpu`, `test_r5_faults`, `test_r5_adversarial`) | all PASS |
| Harness (real Qwen tokenizer, fake vLLM), mixed / drain / rerun | 3/3 PASS, scheduler error none |
| Hidden rerun submission compared with ORDERING_ONLY | attempt_1 differs **0**; attempt_2 differs 2 (one-novel direct path) |
| Primary deadlines | unchanged (no 30-minute reserve) |
| Strict scorer token alignment (would disable scoring globally on a failure), real tokenizer | **0 failures** in 3,448 views (172 evaluation + 259 test units × 8) |

**Note:** the 2+ novel scoring path was not reached in the harness, because the fake model's thinking produced no candidates. It is covered only by GPT's own fake-engine tests.

## No CPU blocker found

## Serious risks (GPU)

1. **A scorer OOM now hits the primary.** Scoring is interleaved, so an OOM in a `prompt_logprobs` prefill (the full-vocabulary logits are not reserved at startup) would kill the shared engine while up to 15 primary units are running. The scheduler handler keeps the finished units, but the in-flight and unadmitted units are lost. In R2 the primary was already finished when scoring began.
   - **This is the main item for the GPU smoke test.**
   - **Cheap optional guard:** skip any score request above a fixed prompt length (for example 1,500 tokens) until the GPU measurement exists.
2. **Primary-first admission starves the scorer.** `e8_try_admit` needs `active < CONCURRENCY`, and the primary queue refills every free slot first. So 2+ novel ranking runs only after the primary queue is empty, or after the admission deadline.
   - In the PROXY48 commit run (deadline-constrained) that may leave only the last ~15 minutes, so few or no rankings may happen, and the memory question may not even be exercised.
   - This is safe, but check `e8_primary_waiting_steps` / `score_requests_started` in the run.
3. **The first scorer error disables scoring for the rest of the run** (fail-closed). This is safe, but a single bad request ends all ranking.

The one-novel direct path needs no GPU scoring and is not affected by items 2 and 3.
