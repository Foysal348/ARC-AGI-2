# Q27-AGENT v1.9.1 — GPU EVAL12 bug review (2026-10-12)

**Request to the reviewer: a DEEP, unrestricted audit.**
- Audit everything in this zip: the notebook, the source, the GPU outputs, the logs and the run-4 evidence. Find every bug, design flaw, wrong assumption or risk you can.
- The diagnosis, hypotheses and fix plan below are **Claude's suggestions only**. They are not limits on the audit. Challenge them, replace them, or go beyond them wherever the evidence points.
- Context:
  - The user plans to add puzzle ordering and a dynamic KV controller **after** these bugs are fixed.
  - Comments on them are welcome, but the priority is making the Long-Think → Program → Verify pipeline actually work on GPU.
- Standing constraints from the user:
  - never use less than 49,152 continuous thinking tokens per request without the user's explicit approval;
  - xhigh effort;
  - the LB-26.94 DIRECT method (v1.4.4) is the base.

## 0. What ran

| Item | Value |
|---|---|
| Notebook | `Q27_AGENT_v1_9_1_SYNTH_EVAL12.ipynb` (sha256 cd5c69da…), the version GPT approved as GO |
| Kernel | `foysalemonshanto/q27-agent` v3, 4× L4, internet off, pinned docker image |
| Mode | `EVALUATION (12 tasks)`, 22 test outputs |
| Wall time | 13:42 → 14:38 (about 56 min): install 171 s, server 510 s, synthesis 2,538 s |
| Server | rung 0 = vLLM 0.19.0 OpenAI server, TP=4, `--reasoning-config {"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}`, fp8 KV, `--language-model-only`, MTP ×3, prefix cache (mamba "align"), `max_model_len` 81,920, `max_num_seqs` 16, `max_num_batched_tokens` 4,096, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, `--enable-auto-tool-choice` |
| Request | `/v1/chat/completions`, no tools; `thinking_token_budget` 49,152, `max_tokens` 57,344, xhigh, temperature 0.6 / top_p 0.95 / top_k 20 |

## 1. Result

| | Hits / 22 | Official |
|---|---|---|
| Agent (synthesis) | 1 | 4.55 % |
| Recorded DIRECT (v1.4.4 run 7) | 3 | 13.64 % |
| Hybrid (DIRECT attempt_1 + agent) | 3 | 13.64 % |

**Statuses:**
- 1 `verified_usable`: 136b0064 (train 3/3, test correct, already solved by DIRECT);
- 9 `train_mismatch`;
- 2 `no_program`.

## 2. Confirmed bug B1: the thinking budget is NOT enforced by the server

**Evidence A: v1.9.1, this run** (`attempts.jsonl`, thinking = completion − tokens(content), counted exactly via `/tokenize`)

| Task | Thinking tokens | Finish | Program source |
|---|---|---|---|
| 10 tasks | **57,344** (= `max_tokens`), content empty | `length` | no-thinking commit request (or none) |
| 4c3d4a41 | 56,238 | `length` (answer cut off) | commit |
| 136b0064 | 49,982 | `stop` | primary (verified, correct) |

- With a working budget, thinking would be forced closed at about 49,152, and the remaining 8,192 tokens would hold the answer.
- Instead, 10 of 12 requests spent all 57,344 tokens thinking and returned **no content**.
- 136b0064 closed at 49,982, past the budget. It looks like a natural close, not a forced one.

**Evidence B: Experiment 4, run 4 (v1.8.3, same server rung with MTP ×3), found today in its `attempts.jsonl`**
- Per-turn budget 6,144, per-turn `max_tokens` 10,240.
- **22 xhigh turns ended at exactly 10,240 tokens, `finish=length`, content empty** (reasoning text 21–28k characters).
- So the budget was **never** enforced on this stack, in run 4 or now.
- This also explains part of run 4's failure (many empty turns or code).

**Why our gates missed it**
- `budget_contract_smoke` sent `thinking_token_budget` 49,152 with `max_tokens` 64 and only checked that the request was **accepted**. Nothing measured whether thinking was actually cut at the budget.
- The fake CPU server always returns a short reasoning plus a code block, so the DRYRUN cannot see this.
- GPT's GO audit did not flag it either.

### Hypotheses for the root cause (not verified; we have no vLLM source here)

- **H1: speculative decoding (MTP ×3) bypasses the thinking-budget logits processor.**
  - vLLM V1 runs some logits processors only on the non-speculative path; processors that are not argmax-invariant can be skipped with spec decode.
  - Both runs used MTP ×3.
- **H2: the processor never "enters" reasoning.**
  - The Qwen3.8 chat template puts `<think>\n` **in the prompt** (generation prompt). The generated tokens therefore never contain `reasoning_start_str`.
  - If the processor only starts counting after it sees `<think>` in the *output*, it never counts.
- **H3: the keithtyser serving overlay / vLLM 0.19.0 build** accepts the field but has no working budget processor for this model, or the `reasoning_config` keys differ from what the processor reads.
- **H4: `</think>` tokenisation.**
  - If `reasoning_end_str` maps to several tokens, the forced sequence could be malformed.
  - v1.4.4 asserts that `</think>` is a single token for this tokenizer, so H4 is unlikely.

**A probe that separates H1–H3 (about 12 min of GPU, optional):** run `thinking_token_budget`=256, `max_tokens`=1,024 under each of:
- (a) the MTP ×3 rung;
- (b) the same rung without `--speculative-config`;
- (c) (b) with a raw `/v1/completions` prompt that does not end with `<think>`.

A rung passes if `</think>` appears at about 256 tokens and the content is non-empty.

## 3. Consequences

1. **v1.9.1's agent result is not a fair test of Long-Think → Program.** In 10 of 12 tasks the model never got to write its answer; the program came from a no-thinking "commit" request fed only the last 7,000 characters of the thinking.
2. **The v1.8.3 / v1.9.x hidden-run DIRECT path (server `rescue`) uses the same `thinking_token_budget` mechanism**, so it would suffer the same bug.
   - **Never submit v1.8.3 / v1.9.x.**
   - The LB-26.94 notebook (v1.4.4) is unaffected: it uses the offline engine and closes thinking **itself** (below).

## 4. How v1.4.4 (LB 26.94) closes thinking — proven on the LB

The engine is an offline `vllm.LLM` with token-id prompts.
- **Phase 1:** `max_tokens` = min(49,152, room); stop when `</think>` (a single token id) is generated.
- **If `</think>` is missing after phase 1:** append `STOP_THINKING_TEXT` = `"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"`.
- **Phase 2:** `max_tokens` 8,192 (the answer); prefix caching avoids re-prefilling the about-50k prefix.
- An EOS before `</think>` must not precede the stop text (v1.1 F8). The answer is the text after the last `</think>`.

## 5. Claude's suggested fix: v1.9.2 (a suggestion; propose better if you see it)

**F1. Close thinking ourselves, as v1.4.4 does, on the existing OpenAI server:**
1. Build the prompt with the chat template (`enable_thinking=True`, `reasoning_effort=xhigh`) as raw text, then call **`/v1/completions`**.
   - Phase 1: `max_tokens` 49,152, `stop` = [`</think>`], `include_stop_str_in_output` = true, `skip_special_tokens` = false.
2. If phase 1 ended **without** `</think>`, call `/v1/completions` again with prompt = phase-1 prompt + phase-1 text + `STOP_THINKING_TEXT`, `max_tokens` 8,192.
   - The prefix cache should hit the long prefix. We log cached tokens to verify this.
3. If phase 1 ended **with** `</think>`, phase 2 continues from prompt + text with `max_tokens` 8,192.
   - This is one request with `stop` only on `</think>`, then continuation.
4. Remove `thinking_token_budget` from every request, so the result no longer depends on server enforcement. The budget rule (≥ 49,152 continuous thinking) is then enforced by our code, and `check_config` checks it.
5. Keep the no-thinking commit as a last resort only, and feed it the **full** answer + the end of the thinking.

**F2. A real enforcement smoke test on GPU before the run:**
- One short request: phase 1 with `max_tokens` = 128, then a forced close and phase 2 with `max_tokens` 256.
- Assert that the content is non-empty and that the `</think>` / stop text is present.
- Plus a `/tokenize` round trip. If it fails: NO-GO, write the fallback submission and stop. Do not spend 50 min.

**F3. An early-abort gate:**
- If the first 4 finished tasks all show `think_hit_budget` with empty content, or produce no program, stop the eval.

**F4. Tests:**
- The fake server learns `/v1/completions` with `stop`, plus a "never closes thinking" mode, so the CPU tests reproduce B1 and prove that F1 handles it.
- A regression test fails if any request contains `thinking_token_budget` while also relying on it.

**Unchanged:**
- 49,152 + 8,192 tokens;
- xhigh;
- temperature 0.6 / top_p 0.95 / top_k 20;
- no tools;
- `repair_turns` 0;
- recorded-DIRECT merge;
- the EVAL12 set. The next eval will use 20 tasks, but only after these fixes.

## 6. Other observations / suspicions (lower priority)

- **S1: KV accounting.**
  - The startup log line "GPU KV cache size: 337,600 tokens" disagrees with the `/metrics` figure of `num_gpu_blocks` 846 × `block_size` 1,600 = **1,353,600 tokens**.
  - Peak KV usage was **68 %** with 12 sequences at about 60k tokens each (about 0.72M tokens), so the 1.35M figure is consistent. The 337,600 line is probably per KV group or per GPU.
  - 0 preemptions.
- **S2: `waiting_peak` 5 while only 12 were running and `max_num_seqs` was 16.** Possible causes:
  - chunked prefill (4,096 batched tokens) queuing the commit requests;
  - the mamba "align" cache limiting admission.
  - It did not matter here, but it matters for the planned KV controller.
- **S3: the commit request saw only the last 7,000 characters of 116k–182k characters of thinking.** Most of the reasoning was lost; the programs then failed training (IndexError, UnboundLocalError, or wrong cells).
- **S4: 136b0064 closed at 49,982, past the 49,152 budget.** This is consistent with no enforcement: a natural close that happened to come late.
- **S5:** MTP acceptance was 0.634 (run 4: 0.736). Overall about 297 tok/s; about 25 tok/s per stream; request wall times 2,060–2,410 s.
- **S6: `budget_contract_smoke` is still named a "contract" pass.** In v1.9.2 it should be replaced by F2, a behavioural test, not a request-shape test.

## 7. Files in this zip

| Path | What |
|---|---|
| `README_v1_9_1_GPU_BUG_REVIEW.md` | this file |
| `notebook/Q27_AGENT_v1_9_1_SYNTH_EVAL12.ipynb` | the notebook that ran |
| `source/` | builder, modules (`synth.py`, `submit_main_v19.py`, `run_eval_v19.py`, `check_config.py`, `rescue.py`, `submit_main.py`, `fake_vllm.py`, …), `tests/test_v19.py` |
| `gpu_run_v1_9_1/` | `eval_report.json`, `attempts.jsonl` (with the programs and feedback), `progress.json`, `server_metrics.json`, `agent_server_report.json`, `vllm-server.log`, `q27-agent.log` |
| `evidence_run4_v1_8_3/` | `run4_turns.json` (every agent turn: tokens, finish, effort, reasoning and content length) and run 4's `vllm-server.log` |
| `reference_v1_4_4/Q27-XHIGH-v1_4_4.ipynb` | the LB-26.94 notebook: the forced-close method in §4 |
| `analysis/GPU EVAL12 v1.9.1 analysis (2026-10-12).md` | Claude's first analysis |
