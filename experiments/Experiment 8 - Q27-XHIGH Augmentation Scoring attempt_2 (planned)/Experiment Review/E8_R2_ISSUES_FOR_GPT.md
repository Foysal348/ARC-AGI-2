# E8 R2 (`Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R2.ipynb`, sha `d131c5e9…`): problems found in Claude's audit

Everything was checked on CPU on a laptop; **nothing ran on a GPU**. Each item says whether it was **measured**, **read from the source**, or **estimated**.

## What is correct (verified)

- **Primary cells unchanged:** cells 0, 1, 3–7 are identical to E7 ORDERING_ONLY (byte comparison).
- **Your own tests:** `test_e8_r2.py` all pass, after changing only the `/mnt/data` paths.
- **Harness:** the real Qwen tokenizer and chat template with a fake vLLM engine, scenarios mixed / drain / rerun, all PASS.
  - The checkpoint is written, attempt_1 is untouched, and the submission is valid.
  - The hidden `submission.json` is identical to ORDERING_ONLY's (E8 replaced nothing).
- **Views and alignment:** the 8 views are built with the real tokenizer, and answer-token alignment is OK (`boundary_equal` = True for every view on 5 PROXY48 units).
- **MTP:** vLLM 0.19 accepts `prompt_logprobs` with MTP (`SamplingParams._validate_spec_decode` rejects only `min_p` / `logit_bias`).

## 1. Blocker: the candidate pool cannot add a new correct answer (measured)

Your `e8_build_candidates`, cross-checked with an independent numpy implementation:

| Set | Tasks where a D4 transform fits every train pair | Tasks where a cell-wise colour map fits every train pair | Test inputs with a train-consistent candidate |
|---|---|---|---|
| ARC-AGI-2 evaluation (120 tasks / 172 test inputs) | **0** | **0** | **0 / 172** |
| ARC-AGI-2 training (1000 / 1076) | 7 | 5 | 11 / 1076 (all correct, but trivial; xhigh attempt_1 would solve these anyway) |

Other evidence:

| Measure | Result |
|---|---|
| identity / mirror as the correct answer, whole evaluation set | **0 / 172** |
| identity / mirror as the correct answer, our 4 EVAL12 GPU runs (88 units) | **0 / 88** attempt_2-only hits |
| wrong attempt_1 where a D8 transform of it is correct (73 units) | **0 / 73** |
| wrong attempt_1 where a recolouring of it is correct (73 units) | **0 / 73** |

**Consequence:** the only "novel" candidate in practice is the **mirror** of the test input.
- With a real attempt_1, the mirror passes the novelty check in **247/259** units of the test file and **169/172** of the evaluation file.
- So Stage 2 mostly ranks identity against mirror. Both are worth about 0.
- **Expected gain ≈ 0.**

The only source with a real chance is missing from the pool: **the grids that appear in Q27's own thinking and answer text**. The full thinking is in memory as `st["p1_ids"]` when a unit finishes. How often the correct answer appears there is unmeasured, because no previous run saved the full thinking.

## 2. Blocker: GPU OOM risk from `prompt_logprobs` (read from the vLLM 0.19.0 source; the size is estimated)

- **Full logits on every GPU:** `LogitsProcessor._gather_logits` all-gathers them because `Platform.use_all_gather()` returns True and `platforms/cuda.py` does not override it.
- **Computed for every prompt token:** `gpu_model_runner._get_prompt_logprobs_dict` computes `compute_logits` plus `Sampler.compute_logprobs` = `log_softmax(dtype=float32)` for **every scheduled prompt token** of a logprob request (up to `MAX_NUM_BATCHED_TOKENS` = 4,096 per step).
- **Not reserved at startup:** `profile_run` → `_dummy_sampler_run` computes logits for only one row per request (≤ 16 rows). The prompt-logprob logits are therefore **not** in the startup memory profile.
- **Size:** the vocabulary is 248,077, so the cost is about 1.5 MB per scored prompt token (bf16 logits + fp32 log_softmax). With `GPU_MEMORY_UTILIZATION = 0.90` on an L4 (about 2 GB free), the limit is roughly **1,400–2,000 prompt tokens per step**.
- **Measured request sizes** (test file, real tokenizer): median **1,300**, max **7,464** tokens per request; batch size 2.
- **Hang risk:** if a TP worker OOMs or hangs inside `engine.step()`, the Python deadline check cannot fire. In a hidden rerun that could run into Kaggle's time limit. This is plausible, not verified.
- **Possible mitigation:** the engine argument `long_prefill_token_threshold` (present in the 0.19 scheduler) caps one request's prompt tokens per step, for example 256–512 with batch size 1. It needs a GPU smoke test.

## 3. Major: scoring volume against the 28-minute window (measured tokens, estimated speed)

- **Volume (measured):** test file, 259 units → 247 scored, 5,944 requests, **10.0M prompt tokens**. Logprob requests do not read the prefix cache.
- **Required speed:** 10.0M ÷ 1,680 s = **5,952 prompt tok/s, sustained for 28 minutes.**
- **Observed (vLLM 10-second log averages, 5 GPU runs):**
  - the highest interval in any run was **5,662** tok/s;
  - intervals above 3,000 tok/s averaged about 4,700–5,240, and occurred only 1–20 times per run;
  - the run start, with 16 prompts of a few thousand tokens arriving, managed **about 900–1,800** tok/s.
- **E8's workload** (many small requests, batch 2, steps not full, extra full-vocabulary work) is closer to the run start than to the peaks.
- **Estimate:** only about 25–90 % of units would be scored before the deadline. This is safe, because the deadline guard works, but the scoring is incomplete.

## 4. Trade-off (not a bug): Stage 1 deadlines move 30 minutes earlier

- **Hidden:** last admission 11:34 → **11:04**, drain 11:36 → 11:06, primary hard stop 11:44 → 11:14.
- **PROXY48:** last admission 140 → **110 min** (−21 %).
- This is necessary in R2's post-primary design (an idle engine; `assert 0 < E8_FINALIZE_SEC < E8_RESERVED_SEC`).
- It conflicts with the approved "Stage 1 Frozen" box and with your Invariant 5 ("we will not reserve time for the scorer by reducing primary coverage").
- It also confounds a PROXY48 E8-vs-E7 comparison.
- **It pays only if item 1 is fixed.**
- **Alternative:** score each unit as soon as it finishes, inside the primary loop. That needs a small routing change, because outputs are dispatched by `request_id.rsplit("|", 1)` and an `e8_score_*` ID would break it. With that change, no reserve is needed.

## 5. Medium: scoring-prompt format (risk)

- `QwenFormatter` is NVARC's fine-tuning format: digits without separators, no instructions, raw `<|im_start|>` turns, no think block.
- NVARC's 4B was trained on it; Q27 was not. Whether Q27's NLL on this format separates correct from wrong grids is unknown.
- **Smoke test:** the NLL of the correct grid compared with slightly corrupted grids, on a few solved evaluation units.

## 6. Minor

- `test_e8_r2.py` hard-codes `/mnt/data` paths and the donor file name, so it does not run from the zip.

## Requested changes

1. **Candidate pool:** add every valid grid parsed from the full thinking and answer text (excluding attempt_1). Use identity and mirror only as the fallback; do not score them. Keep the train-consistent D4 and colour candidates if you like; they cost nothing.
2. **Memory:** set `long_prefill_token_threshold` (or an equivalent cap) with batch size 1, and verify it in a short GPU smoke test (memory, alignment with MTP, NLL sanity) before any full run.
3. **Stage 1 timing:** Claude recommends **per-unit scoring inside the primary loop** (score each unit as soon as it finishes, with the small `request_id` routing change), so Stage 1 deadlines stay exactly as in ORDERING_ONLY with no 30-minute reserve. If you keep the reserve instead, justify it explicitly against item 1.
   - **Per-unit scoring needs no reserve.** Scoring requests go into the same engine while other units are still thinking, so Stage 1 deadlines (hidden: admit 11:34, drain 11:36, hard stop 11:44) stay unchanged.
   - **Three costs:**
     - the scoring prefill shares the GPU with thinking, so the cost is spread out rather than removed, and primary decoding slows down by the scoring volume;
     - units that finish just before the hard stop may not get scored; they keep the fallback attempt_2;
     - the OOM risk in item 2 still applies, because the memory headroom is the same.
   - The primary loop's output routing, drain and hard-stop logic must ignore or route `e8_score_*` request IDs. Count scoring requests against `max_num_seqs` so primary units are not starved.
4. **Score only when there is a choice.** Let "novel" mean a valid candidate other than attempt_1, identity and mirror:
   - **0 novel candidates:** keep the fallback attempt_2; no scoring.
   - **Exactly 1 novel candidate:** use it as attempt_2 directly; no scoring (the fallback is worth about 0; see item 1).
   - **2 or more novel candidates:** run the 8-view scoring to rank them, and attempt_2 = the best one.

   This cuts the GPU cost to the units that actually need a ranking.
5. **Optional, no behaviour change:** save the full thinking text, so the oracle coverage of thinking grids can be measured in the next ORDERING_ONLY run.
