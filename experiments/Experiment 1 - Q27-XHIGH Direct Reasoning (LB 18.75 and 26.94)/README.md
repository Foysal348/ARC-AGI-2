# Experiment 1 — Q27-XHIGH Direct Reasoning (LB 18.75 → 26.94)

**Status:** baseline in use. **Best: Q27-XHIGH v1.4.4 = public LB 26.94** (submission 57032990, scored 2026-10-12).

## Idea

Kaggle discussion 745787 (2026-10-05) showed that Qwen3.8 needs `reasoning_effort` **xhigh** with very long generations.
- Its author forced the commit after about 49k thinking tokens.
- So: one prompt per test input, full xhigh thinking up to 49,152 tokens, then Qwen's stop sentence + `</think>`, then a grid-only answer (≤ 8,192 tokens).

## Method (v1.4 / v1.4.4)

**Prompt and parsing**
- russcore-style free-form prompt (space-separated grids, "give ONLY the final output grid … in a ```grid block").
- Robust grid parser.

**Serving**
- vLLM 0.19 **offline engine**, TP=4, fp8 KV, `max_model_len` 81,920, `max_num_batched_tokens` 4,096 (fixes an MTP-drafter OOM).
- MTP speculative decoding ×3 (acceptance 0.661), libcuda shim for the FlashInfer JIT.

**Sampling:** temperature 0.6 / top_p 0.95 / top_k 20.

**Scheduler:** continuous admission with 16 concurrent queries.
- v1.4.4 adds **prefix caching** (mamba `align` mode) and **admission until 11:34**, drain at 11:36, hard stop 11:44, end 11:50.

**Attempts:** attempt_1 = the model's grid; attempt_2 = the test input (or its mirror).

## Versions and runs

| Version | Change | EVAL12 | LB |
|---|---|---|---|
| v1.3.4 | libcuda shim | run 2: OOM in MTP drafter | — |
| v1.3.5 | batched tokens 4,096 + CUDA-graph memory estimate | **1/22**; 12/12 hit the 49k cap | — |
| **v1.4** | russcore prompt, per query, forced commit at 49k | **2/22**; 4 natural closes → 2 hits | **18.75** |
| v1.4.1–v1.4.3 | prefix caching; 11 h 50 m budget | — | — |
| **v1.4.4** | NVARC-style admission until 11:34 (+ drain) | **3/22** (run 7; 274.6 tok/s; 0 preemptions) | **26.94** |

The +8.19 from v1.4 to v1.4.4 came with the same model, prompt and budget. It came only from **more queries finished in 12 h**: prefix caching avoids re-prefilling about 50k tokens after each forced close, and admission now runs until the end.

## Key lessons

- **Thinking needed:** in EVAL12 the correct answers needed 26–37k continuous thinking. Forced closes at 49k were correct 1 time in 37.
- **Score is coverage:** for DIRECT, score is about the number of queries finished in 12 h.
- **EVAL12 understates LB:** EVAL12 is harder than the hidden set, and LB has been about 2× the EVAL12 percentage.

## Folders

**Experiment Notebooks/**
- `Q27-XHIGH-v1 … v1_4_4.ipynb`.
- `Build/Q27X_build/`: builder `make_q27x.py`, plus R6 builder and harnesses. `history/` builders still point at the pre-2026-10-12 paths.

**Experiment Review/**
- GPT review bundles (v1.2 … v1.4 run 4).
- `R5 safe-fix review/`.

**Experiment Results/**
- `Kaggle Runs/Q27X_run1 … run7_v144` (records, answers, run reports), the downloaded executed notebooks, `fork_q27_check`.

## Next ideas (Track A)

Raise concurrency above 16. KV is 1.35M on the offline engine, and peak usage was 59.5% at 16, so more queries could finish in 12 h. This needs a throughput measurement first.
