# E12-V: the two GPT audits, validated (2026-10-19)

**Method.**
- **No project code was changed.**
- **Sources:**
  - vLLM v0.19.0 from https://raw.githubusercontent.com/vllm-project/vllm/v0.19.0/…
  - our real Kaggle vLLM logs;
  - the GitHub API for every cited issue.
- **Tests:** `_scratch/session_1c6db7aa/e12_validate/validate_e12v_findings.py`.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| B1 (both) | The CELL 5 smoke requires `prompt_tokens_details`, but vLLM 0.19 sets it only when `num_cached_tokens` is truthy, so the first (cold) request is rejected and the engine is never used | **VALID: real blocker** | `vllm/entrypoints/openai/chat_completion/serving.py:1601-1603` (non-stream; streaming 1214-1216). The predicate is `build_e12_duck.py:396`, and the test reply gives `ok=False`. Issue https://github.com/vllm-project/vllm/issues/44377 ("--enable-prompt-tokens-details omits zero cached tokens", closed after 0.19). The fake server always sent details, so the harness missed it. |
| GPT 2 #3 | If `/v1/models` fails, CELL 6 falls back to the INT4 SGLang path as the model id; vLLM serves `qwen` | **VALID (conditional)** | `build_e12_duck.py:514`. It happens only when `/v1/models` fails after the server has passed it. |
| GPT 2 #4 | The cache report counts unknown (`None`) as 0 → a fabricated 0 % | **VALID (reporting)** | `build_e12_duck.py:553/556`. Related: https://github.com/vllm-project/vllm/issues/44961 (details null in V1, closed; "incomplete fix"). The run's prefix-cache rate stays observable through the vLLM log / `/metrics`, which `e11_decode_summary` and `metrics_snapshot` already read. |
| B2 (GPT 1) | A tool-call XML inside `<think>` is lost by vLLM 0.19's parser | **VALID upstream (https://github.com/vllm-project/vllm/issues/39056, open); handled by Duck** | `tool_agent.py:6842-6851` recovers it from `reasoning`. Test `think_markup`: the markup was only in `reasoning` (duplicated), with no `tool_calls`. Result: accepted; the code ran **once**; the markup was stripped from the replayed history. |
| B3/B4 (GPT 1) | KV 337,600 tokens means ~28k per request for 12 streams; the fake's "7.66×" is inconsistent | **NOT VALID** | Real vLLM logs on this model: **337,600 tokens and 12.63× at 81,920** (Q27-AGENT run 4), **10.32× at 106,496** (E10 R5). That is ~1.03-1.10 M effective. vLLM's token line is pool/4 for the hybrid model, and the concurrency line is the real capacity. The fake's 7.66× = 1.035 M / 135,168 follows that formula. The real value at 135,168 is to be read from the GPU log. |
| B5 / GPT 2 #5 | MTP + prefix caching + tool calls can corrupt tool-call emission on repeated identical prompts | **Upstream report exists (https://github.com/vllm-project/vllm/issues/50188, open, a later version); not checkable on CPU** | S1-60K ran MTP-3 + prefix caching + FP8 on this model in vLLM 0.19 (LB 34.58), but without tool calls. Duck prompts are not identical repeats (each turn extends the prefix). Measure in the GPU smoke: the malformed / recovered tool-call counts (Duck logs both), and correctness on cache-hot turns. |
| GPT 2 | Hybrid MTP: the first repeat misses the prefix cache | upstream (https://github.com/vllm-project/vllm/issues/53504, closed) | performance only; measure the cache rate in the smoke |
| GPT 2 | vLLM 0.19.0 page-size startup regression for Qwen3.5 + MTP | upstream (https://github.com/vllm-project/vllm/issues/38979, closed) | Our vLLM 0.19.0 server with MTP-3 + prefix caching + FP8 on this model started fine (Q27-AGENT run 4). Watch the GPU startup. |
| PR | https://github.com/vllm-project/vllm/pull/48375 (MambaManager drop_eagle_block) | open, after 0.19 | not applicable without a patch; no action |

**Proposed fixes (need the user's go):**
1. **Smoke:** pass on content + integer usage counts; record whether `prompt_tokens_details` is present.
2. **Model id:** use `qwen` as the fallback on the vLLM path.
3. **Cache report:** keep unknown ≠ 0, with coverage counts.
4. **Harness:**
   - the fake's first reply omits details (B1 regression test);
   - a think-markup scenario;
   - the synthetic capacity values labelled as simulated.
