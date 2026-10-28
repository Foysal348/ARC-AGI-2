# Experiment 18: S1-60K on vLLM 0.32.0, 20 streams

**User (2026-10-27):** "S1-60K notebook, only the engine becomes vLLM 0.32 and 20 streams; Poetiq, prompt and 60k exactly as they are."

**Why:**
- S1-60K has scored LB 34.58 and 31.94 on vLLM 0.19 + FP8 + MTP 3 at 16 streams.
- Every SGLang / INT4 submission since then scored lower (17.50-29.17).
- Coverage of the 240 hidden tasks is the binding constraint. S1 decoded about 256 tok/s over a whole run.
- The Experiment 14 VMS matrix measured vLLM 0.32.0 with the same FP8 + MTP 3 recipe, at 8k / 40k / 90k context:

| Streams | 8k | 40k | 90k | Greedy gate |
|---|---|---|---|---|
| 16 | 434 tok/s | 399 tok/s | 281 tok/s | PASS |
| 20 | 452 tok/s | 423 tok/s | 281 tok/s | PASS |

- **Caveat:** those are steady decode windows. S1's 256 tok/s is a whole-run average with long contexts. The gain on a real 12-hour run is not proven; the GPU smoke measures it.

## The notebook

`Experiment Notebooks/Q27-XHIGH-E18_S1_60K_VLLM032_S20.ipynb`, built by `Build/build_e18.py` from S1-60K (Experiment 10, sha `ed530071`).

**Unchanged cells** (byte-identical; the builder asserts it):
- CELL 0;
- CELL 2 (install + tokenizer; it still installs vLLM 0.19 for the fallback, and the tokenizer stack is transformers 4.57.6);
- the task order;
- the Poetiq core;
- the solver;
- the report.

**CELL 1 changes** (three):
- `MAX_NUM_SEQS` 16 → 20;
- the commit profile `E18SMOKE`: 20 eval tasks, 45 min (admission until 35 min, drain at 36, hard stop at 42). It is the default for a commit; the hidden rerun stays `RERUN_ALL` (12 h);
- the vLLM 0.32 wheelhouse path.

**CELL 5 (new): `Build/e18_engine_cell.py`.**
- vLLM 0.32.0 runs as an OpenAI server from a venv. The wheelhouse is `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse`, installed with the VMS installer.
- A shim gives the solver the same names it uses in S1: `engine.add_request / step / abort_request`, `llm.get_metrics()`, `SamplingParams` and `RequestOutputKind`.
  - Each request is a streaming `/v1/completions` with the prompt token ids and `return_token_ids`.
  - An abort closes the stream (vLLM aborts on disconnect) and drops its undelivered deltas, as an in-process abort does.
- Flags = S1's `ENGINE_CONFIG`: FP8 model, TP4, fp8 KV, MTP 3, prefix caching, FlashInfer if `-lcuda` links (S1's own probe), 4,096 batched tokens, memory 0.90, custom all-reduce off, seed, max model length 106,496. Plus 20 sequences.
- The generation config stays `auto`, as `LLM()` uses it. Temperature, top_p and top_k are sent on every request.
- Ladder:
  1. S1 flags + `--language-model-only`;
  2. the exact flag set of the Kaggle-run VMS arm `fp8_mtp3_s20`;
  3. the same with Triton attention;
  4. S1's own in-process vLLM 0.19 engine cell, verbatim.

**CELL 8b (new):** writes `q27x/e18_engine_stats.json` (shim stats, tok/s, KV gate, end metrics) and stops the server.

## Checks

**Local harness** `src/harness/harness_e18_nb.py`. It runs the real cells against `fake_vllm_e18_server.py`; the fake scripts Poetiq replies per task, using the S1 harness behaviours. All six scenarios pass (2026-10-27):

| Scenario | Checks passed |
|---|---|
| mixed | 21 |
| kvgate | 16 |
| hardstop | 18 |
| rerun | 16 |
| lmofail | 16 |
| allfail | 7 |

What the checks cover:
- the E18SMOKE profile and its deadlines;
- the flags, against S1's config;
- the KV parse (concurrency = min(20, KV // 43,000));
- sampling (temperature 1.0, top_p 0.95, top_k 20, a per-request seed);
- phase 2 = the phase-1 prompt + its tokens;
- the phase-1 caps;
- the shim stats;
- F3, F1 and A2;
- forced then natural;
- iterations;
- the KV gate;
- drain aborts (server-side disconnects);
- the rerun submission;
- every ladder rung, including the failing fallback.

**Kaggle CPU kit:** `Build/build_cpu_kit_e18.py`, dataset `foysalemonshanto/q27-e18-cpu-kit`, kernel `foysalemonshanto/q27-e18-cpu-harness`.

## Pre-run capacity check (rev 2, corrected after audit B H1 and the GPU smoke)

**What the KV log line means.** vLLM 0.32's "GPU KV cache size" line is max-concurrency × max_model_len, not the raw pool.
- With 1,600-token blocks, a full-length request (106,496 tokens) needs 67 attention blocks + 12 GDN-state blocks = 79.
- The 0.32 pool is therefore about 9.45 × 79 ≈ 747 blocks without `--language-model-only` (VMS). With it, the smoke logged 1,110,415 tokens ≈ 10.4 × 79 ≈ 823 blocks.
- S1 on vLLM 0.19 had about 844 blocks.
- The "23 slots" in rev 1 was wrong. Concurrency is still 20, because min(20, KV // 43,000) = 20 in either unit.

**Smoke data.** 20 streams at 7k → 30k context used 39 % → 62 % of KV. Preemptions: 0.

**12-hour run.**
- Each unit holds on average prompt (~7k) + half the thinking (~30k) ≈ 37k tokens. By extrapolation from the smoke trend, that is about 75-85 % KV.
- When many units are long at the same time, KV goes above 0.95. The S1 gate then holds admission, and vLLM may preempt (recompute).
- Measured counts arrive in the CELL 8b print (rev 2).

**Time.** Startup is +3-4 min vs S1 (venv install 175 s; ready 550 s).
