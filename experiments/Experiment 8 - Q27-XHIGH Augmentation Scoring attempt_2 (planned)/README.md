# Experiment 8: augmentation scoring for attempt_2 (NVARC `calc_scores` idea). PLAN

**Status (2026-10-13): CLOSED, NO-GO.** See **`E8_FINAL_REPORT.md`** for the results, the audit validation, the lessons and the next work. The R5 GPU run OOMed in the scorer, and thinking-mined candidates contain no correct answers (0/31).

**Earlier status:** GPT implementation **R2** received and audited (CPU only). It has **3 blockers** (Stage 1 not frozen, a candidate pool with zero expected gain, a `prompt_logprobs` OOM risk); see `Experiment Review/E8_R2_AUDIT (2026-10-13).md`. Nothing has been pushed or run.

## Folder layout

| Path | What |
|---|---|
| `Experiment Notebooks/BASE_Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb` | the frozen base (sha `8fd956ce…`) |
| `Experiment Notebooks/Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R2.ipynb` | GPT R2 (sha `d131c5e9…`), under review, not approved |
| `Experiment Results/GPT Outputs/` | the original R2 zip, plus its extracted contents in `R2/` |
| `Experiment Review/E8_R2_AUDIT (2026-10-13).md` | Claude's audit of R2 |
| `Experiment Review/R2 audit validation/` | harness logs and the tokenizer check script |

## User's goal (fixed spec: an audit must not weaken it)

- Adopt **only** NVARC's augmentation scoring (`calc_scores`).
  - Each candidate grid is shown to the model under 8 views: the 8 rotations and flips, each with a random colour permutation.
  - The model's log-likelihood of the candidate is summed over the 8 views.
  - The best-scored candidate becomes **attempt_2**.
- NVARC's dual selection (two rankers, idea #3) comes **later**, in a separate step. Never do both together.
- **Base:** `Experiment 7/Experiment Notebooks/Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb` (v1.4.4 LB 26.94 + Codex ordering).
- **Unchanged:**
  - the 49,152 + 8,192 xhigh thinking;
  - prompt, sampling and seeds;
  - attempt_1 (the model's grid);
  - deadlines;
  - concurrency 16;
  - dual mode;
  - the submission is always written.

## Design (minimal)

1. **Candidates for attempt_2**, for each test input:
   - the grids that appear in the model's thinking and answer text, excluding attempt_1;
   - the test input;
   - its mirror.

   The last two are what v1.4.4 already uses today.
2. **Scoring requests:**
   - **Prompt:** the same russcore prompt, rebuilt on the transformed puzzle (train pairs and test input, with the same rotation/flip and colour permutation). Chat template with thinking disabled. The assistant turn is the transformed candidate in a fenced ` ```grid ` block.
   - **Request:** `SamplingParams(max_tokens=1, prompt_logprobs=0)`.
   - **Score:** the sum of the log-probabilities of the answer tokens only.
   - **Total:** the sum over the 8 views. Higher is better.
3. **When:** as soon as a unit finishes, its scoring requests go into the same vLLM engine. They are prefill-only and need no generation. When all of them return, attempt_2 = the best candidate ≠ attempt_1, and the submission is rewritten.
4. **Fallback:** any error, a deadline, or no candidate except the input and mirror means attempt_2 stays exactly as in v1.4.4.
5. **Records:**
   - the candidate list with source and score;
   - the chosen attempt_2;
   - in eval, whether attempt_2 hits.

## Facts checked (2026-10-13)

- **vLLM 0.19.0 source (GitHub tag):**
  - `SamplingParams._validate_spec_decode` rejects only `min_p` and `logit_bias` when MTP is on. `prompt_logprobs` is allowed.
  - `prompt_logprobs` sets `skip_reading_prefix_cache`, so a request with prompt logprobs does not read the prefix cache. Others are unaffected.
  - Chunked prefill with prompt logprobs is supported (`in_progress_prompt_logprobs`).
- **Memory risk (must be smoke-tested on GPU):** `_get_prompt_logprobs_dict` computes full-vocabulary logits plus log-softmax for every prompt token in the chunk (up to 4,096 tokens per step).
  - Estimate: 4,096 × about 150k vocabulary × fp32 ≈ 2.5 GB.
  - This is not part of vLLM's startup memory profile, so an OOM is possible next to 16 long requests on an L4.
  - Possible mitigation: short scoring prompts, or scoring after phase 1 ends. To be decided after a GPU smoke test.
- **Candidate supply: the main open question.** The saved answer files keep only the **last 1,500 thinking tokens**. Across the 4 EVAL12 runs that have answer files (88 units):

  | Measure | Result |
  |---|---|
  | units with at least one extra grid in the saved thinking tail | 19 (47 extra grids in total) |
  | attempt_1 hits | 8 |
  | units where an extra grid was the solution while attempt_1 was wrong | **0** |

  - The full 49k-token thinking is still in memory during the run, so it may hold more grids. This is unmeasured.
  - EVAL12 is harder than the hidden set.
  - Scoring can only help if some candidate is correct, so the first measurement is the **full-thinking candidate yield**.

## Proposed steps (each needs the user's go)

| Step | What | GPU |
|---|---|---|
| 0 | Measure: save the full thinking text, then parse every grid in it (CPU-side code, no behaviour change). Count how often the correct answer appears as a candidate other than attempt_1 | the next eval run (or reuse a run) |
| 1 | GPU smoke: a few scoring requests next to 16 long requests. Check memory, the speed of prompt logprobs with MTP, and that the scores are sane (the correct grid should score better than a corrupted one) | short |
| 2 | Full attempt_2 scoring in the ordering-only base, PROXY48 commit run, compared with the same base without it | about 2.7 h |
