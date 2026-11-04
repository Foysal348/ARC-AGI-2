# Deep review request: Q27-XHIGH-v1 — Qwen3.8-27B-FP8 with full xhigh thinking on Kaggle 4×L4

You are an independent, skeptical reviewer. This notebook is about to run on paid Kaggle GPUs (4× NVIDIA L4 24 GB,
offline, 12-hour limit) for the ARC Prize 2026 ARC-AGI-2 competition. Find **real bugs, errors and design problems**.
For every claim:
- verify it against the source files in this zip, citing file + line;
- mark it **VERIFIED** (seen in the source) or **INFERRED** (reasoning only);
- do not give generic advice.

## Files in this zip
| path | what |
|---|---|
| `PROVENANCE.md` | **where every part comes from** (our notebooks with cell/line, discussion, russcore, vLLM source lines, new code) — check each REUSED/VERIFIED claim |
| `notebooks/Q27-XHIGH-v1.ipynb` | **the notebook under review** (9 cells) |
| `build/make_q27x.py` | builder that generates the notebook (same code, easier to read) |
| `build/harness_q27x.py`, `build/harness_output.txt` | local test harness: real cells + real Qwen tokenizer/template + real competition JSON + a FAKE vLLM step engine; 4 scenarios pass |
| `build/PROMPT_FOR_REVIEWER.md` | short design summary |
| `sources/kaggle_discussion_745787.md` | the Kaggle discussion that motivated the design |
| `sources/vllm_0_19_0/…` | the vLLM 0.19.0 Python source actually installed on Kaggle (speculative config, Qwen3.5 MTP, LLMEngine add_request/step/abort, LLM, SamplingParams, arg_utils, FP8 quantisation, outputs) |
| `sources/model_qwen3-8-27b-fp8-repacked-v1/` | config.json, safetensors index (incl. `mtp.safetensors` tensors), chat_template.jinja, tokenizer/generation config of the exact model |
| `sources/our_notebooks/PIV-Q27-V3_1-EVAL120-ALLFIX.ipynb` | our earlier Kaggle-proven REPL notebook; the install block, engine settings, chat rendering and stop text were taken from it |
| `sources/our_notebooks/PIV-Q27-SUB-v1_3-HIDDEN-TEST-SUBMISSION.ipynb` | our submission notebook that scored **LB 1.67** (thinking capped at 3–12k tokens) |
| `sources/our_notebooks/*.md` | our measured results (EVAL120 baseline 1.67 %, S12 subset run) |
| `sources/public_notebooks/arcagi2-qwen38-freeform-direct.ipynb` | russcore's public Qwen3.8 notebook from the discussion (vLLM 0.19, same model, xhigh, 57k tokens, concurrency 24) |

## Background
- **Why our old score was low:** our PIV REPL notebooks capped thinking at 3–12k tokens and scored LB 1.67.
- **Discussion 745787:**
  - Qwen3.8 needs xhigh effort; the median generation is ~64k tokens.
  - NguyenThanhNhan (49th) ran 27B FP8 + MTP at 65k tokens, one pass, forcing a commit after ~49k thinking tokens.
  - Anil Thomas (8th) estimates ~30 LB at ~350 tok/s aggregate.
  - Teams above 45 use a harness.

## Design of the notebook
- **Cell 0:** stdlib only; writes a fallback `submission.json` (test input as both attempts) first.
- **Cell 1:** profiles.
  - `EVAL12` (commit / interactive run): 12 public evaluation tasks, scored; it also measures throughput.
  - `RERUN_ALL` (`KAGGLE_IS_COMPETITION_RERUN`): every hidden test task.
- **Cell 2:** offline install of vllm 0.19.0 from the wheelhouse, with version parity (copied from V3.1); tokenizer;
  `</think>` id; stop-text ids.
- **Cell 3:** one prompt per task (grids as digit rows), rendered with
  `apply_chat_template(enable_thinking=True, reasoning_effort="xhigh")`.
- **Cell 4:**
  - Answer parsing: last ```` ```python ```` block containing `def solution`; JSON `{"test_outputs": [...]}`.
  - `solution()` runs in a fresh process on all train inputs (exact match), then on the test inputs.
  - Attempts: verified code > direct grid > unverified code > test input.
  - Self-tests.
- **Cell 5:** engine.
  - Same as the proven V3.1: TP4, `language_model_only`, CUDA graphs on, custom all-reduce off, prefix cache off.
  - Plus `max_model_len=65536`, `max_num_seqs=16`, `speculative_config={"method":"mtp","num_speculative_tokens":2}`.
  - If the engine refuses MTP, it is retried without it.
- **Cell 6:** continuous scheduler on `llm.llm_engine.add_request / step / abort_request` with DELTA token ids. A
  finished task frees its slot at once; there are no waves.
  - Phase 1: `max_tokens = 45,056`.
  - `</think>` present and EOS → done.
  - `</think>` present but cut by length → continue the answer (+8,192 tokens).
  - No `</think>` → Qwen's documented stop sentence + `</think>` appended token-exactly, then the answer (8,192).
  - Deadlines: admission stop; drain (in-flight thinking force-closed, answer 4,096); hard stop (abort everything).
  - `submission.json` is rewritten after every task.
- **Cell 7:** report (tok/s, modes, 240-task time projection) + bundle.

## Checklist (answer every item)
1. **vLLM 0.19 API.** Verify each call against the source:
   - the `LLM(**ENGINE_CONFIG, speculative_config=...)` kwargs;
   - `add_request(id, {"prompt_token_ids": ...}, params)`;
   - `step()` with DELTA outputs: are `outputs[0].token_ids` deltas, is `finish_reason` set on the last one, are
     request ids external?;
   - `abort_request` semantics;
   - `SamplingParams(detokenize=False, output_kind=DELTA, seed, top_k, …)`;
   - any interaction with speculative decoding.
2. **MTP on this checkpoint.** Does `model_type qwen3_5` become a `Qwen3_5MTP` draft? Do the weight names and the FP8
   ignore list match? Is `num_speculative_tokens=2` with one MTP layer OK or harmful? Is it compatible with
   `language_model_only`, hybrid GDN/mamba layers and CUDA graphs?
3. **Chat template.** Does xhigh render correctly? Is the forced-close continuation token-exact and in the format the
   template / model expect? Is the answer-continuation path correct?
4. **Scheduler correctness.** Check:
   - phase transitions and double finishing;
   - the drain path when thinking already closed;
   - hard stop;
   - busy loops or early exits;
   - `max_model_len` overflow and `PROMPT_TOKEN_LIMIT`;
   - token accounting.
5. **Verification and attempts.** Check:
   - parsing robustness (several blocks, malformed JSON);
   - sandbox safety, timeouts and the RLIMIT_AS;
   - grid validation;
   - per-query ARC scoring.
6. **Submission safety.** Is every task and every query covered? Is the write atomic? What happens if cell 5 or 6
   crashes? Dual mode.
7. **Feasibility on 4×L4.**
   - KV cache for 16 concurrent ~65k sequences (compute from config.json: full-attention layers, KV heads, head_dim,
     TP4) plus mamba state.
   - Does `gpu_memory_utilization=0.90` with `max_model_len 65536` fit?
   - Expected time for the 12-task commit run and the 240-task rerun, vs the discussion's numbers.
8. **Versus the discussion.** What does the notebook ignore or contradict? For example: total budget ~64k vs ours
   45k + 8k, forced-commit point 49k, fp8 KV, concurrency, number of passes, harness vs pure reasoning.
9. **Versus our proven notebooks.** Any regression against V3.1 / SUB v1.3: paths, wheelhouse, pinned versions, env
   vars, model path?
10. **Provenance.** Using `PROVENANCE.md`: check that each REUSED part really matches its source, and that each VERIFIED claim holds. Two open questions are flagged there: temperature 0.6 vs the model's `generation_config.json` 1.0, and BF16 vs fp8 KV. Give your recommendation on both.
11. **Versus russcore's public notebook.** Engine/sampling choices worth copying or avoiding.

## Output format
- **Verdict:** GO / GO-with-fixes / NO-GO, separately for (a) the 12-task commit run and (b) a hidden-test submission.
- **Findings table, ranked P0 / P1 / P2:**
  - P0 = will crash or waste the GPU run;
  - P1 = wrong results or large inefficiency;
  - P2 = minor.

  Each row: file + cell + line; quoted evidence; VERIFIED or INFERRED; the exact fix (code if possible).
- **Checked and correct:** a list with evidence.
- **Top 3 changes** that would most improve the expected leaderboard score, with reasons.
