# delongyang / "arc-agi-2 inference qwen3.8-27b-fp8 within 4 hours": deep analysis (Claude, 2026-10-17)

**Source:**
- Kaggle `delongyang/arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours`, pulled 2026-10-17. Public, 18 votes, last run 2026-10-06.
- Local copy: `D:\ARC AGI 2\_scratch\lb_monitor\pub\`.
- The notebook has no saved outputs, and no LB score is visible.

## What it does

| Part | Setting |
|---|---|
| Engine | vLLM OpenAI server, TP4, official FP8 checkpoint (`mikedan7/qwen3-8-27b-fp8-official`), `max_model_len` 131,072, `max-num-seqs` 16, `gpu-memory-utilization` 0.96, fp8 KV, **TRITON_ATTN** (also forced for the MTP drafter via `VLLM_FORCE_ATTN_BACKEND`), **MTP 6 tokens**, reasoning parser qwen3 |
| Reasoning | **`reasoning_effort: "medium"`** (server-default chat-template kwarg), `enable_thinking` true; **no thinking cap** (`max_tokens` 120,000) |
| Sampling | **temperature 0.0**, one request per task, `MAX_RETRIES` 0 |
| Method | **Direct / transductive.** The task is rendered as compact digit rows. A Chinese instruction asks for a JSON with `attempt_1`/`attempt_2` for 1-output tasks, or `attempt_1..n` for n outputs. A long tail asks for a `|` separator every 4 cells on wide rows, against column-count errors |
| Order | Shortest prompt first; a rolling pool of 16 |
| Deadline | 690 min from notebook start; queued tasks are abandoned after it |
| Fallback (unsolved tasks) | Rule-based, from the task's own train pairs: all train outputs identical → that grid; some train output equals its input → the test input; otherwise the most common train output. The second slot gets a different candidate |
| Robustness | Pre-filled submission; incremental save; JSON / literal_eval extraction; grid normalisation; thinking log |
| Data file in the committed version | `arc-agi_test_challenges.json` (placeholder = 240 *training* tasks in commit mode) with training solutions |

## Verified problems and caveats

1. **The submission format looks wrong for Kaggle.**
   - It writes `{task: {"attempt_1": g, "attempt_2": g, ...}}`, a dict per task.
   - The official format, per `sample_submission.json`, is `{task: [{"attempt_1": g, "attempt_2": g}, ...]}`, a list with one dict per test input.
   - Its own scorer encodes a different rule ("attempt_i matches output i" for multi-output tasks).
   - Unless Kaggle tolerates this, the notebook would error or score near 0 on the LB. *Verified from files (format); LB effect is a hypothesis.*
2. **"Within 4 hours" was measured on the placeholder test file,** which is 240 easy training tasks, not hidden-like tasks. The hidden set is about as hard as the evaluation set (our finding), so the hidden runtime would be longer.
3. **"FlashInfer FP8 KV crashes on L4" is not our experience.** Our R5/D66/L66 ran 12 h with FLASHINFER + fp8 KV + MTP-3 on vLLM 0.19 (with our libcuda shim). The Triton switch is not needed for us.

## Ideas worth taking (ranked by value for us)

1. **`reasoning_effort` "medium" instead of "xhigh": the biggest open lever.**
   - The model thinks less by design, so tasks finish faster.
   - S1-60K just showed that coverage dominates the LB (34.58 vs 21.39).
   - It is not a token cap, but it lowers thinking length, so it needs the user's decision under the 49k rule.
   - Its accuracy on eval is unknown. Measure it as one change on the S1 base (LB or EVAL120).
2. **A rule-based fallback for unreached tasks** (constant output / identity / most common train output) instead of plain test-input copies.
   - Free.
   - It helps the about 30 % of hidden tasks we never reach, and gives attempt_2 a different candidate.
3. **MTP draft length:** 6 at T 0 (high acceptance). At our T 1.0, acceptance is about 0.55 per position, so more draft tokens likely waste compute. Test MTP-4 only together with a temperature change.
4. **`gpu-memory-utilization` 0.96** (ours 0.90): more KV, so more slots or fewer preemptions. Low risk, but it must fit the CUDA-graph memory.
5. **Shortest-first ordering:** like our ORDER2 (already in S1).
6. **The `|`-every-4-cells output instruction for wide grids:** relevant to DIRECT answers, not to Poetiq programs.

## Not useful for us

- **Temperature 0 with one attempt:** gives no diversity. D66/L66 show independent attempts are complementary.
- **The custom attempt format:** see problem 1.
- **Chinese prompt:** the claim that it matters is unverified.
