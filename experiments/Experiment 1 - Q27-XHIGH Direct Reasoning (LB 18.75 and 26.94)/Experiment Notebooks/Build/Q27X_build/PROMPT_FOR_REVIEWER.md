# Review request: Q27-XHIGH-v1 (Qwen3.8-27B-FP8, full xhigh thinking, 4×L4, dual mode)

Report real defects only. Cite the cell and line for each, and say for every claim whether you verified it against
code or inferred it.

## Why this exists
- Our REPL notebooks (the PIV series) capped thinking at 3–12k tokens and scored LB 1.67.
- Kaggle discussion 745787:
  - Qwen3.8 needs `xhigh` effort, with a median of ~64k tokens per task.
  - NguyenThanhNhan (49th) submitted Qwen3.8-27B FP8 + MTP at 65k tokens, one pass, forcing a commit after ~49k
    thinking tokens.
  - Anil Thomas (8th) estimates ~30 on the LB at ~350 tok/s aggregate on 4×L4.

## Design (notebook `Q27-XHIGH-v1.ipynb`)
- **Cell 0, stdlib only:** writes a fallback `submission.json` (test input as both attempts) first.
  - Commit / interactive run = `EVAL12`: 12 public evaluation tasks, scored.
  - `KAGGLE_IS_COMPETITION_RERUN` = every hidden test task.
- **Cell 2:** the PIV/V3.1 offline install block (vllm 0.19.0, transformers 4.57.6, parity check). Proven on Kaggle.
- **Cell 3:** one prompt per task; grids as digit rows; `apply_chat_template(enable_thinking=True,
  reasoning_effort="xhigh")`.
  - The answer must contain a `solution(grid)` code block and a JSON block `{"test_outputs": [...]}`.
- **Cell 4:**
  - Parsing.
  - Fresh-process execution of `solution()`: all train inputs must match exactly, then the test inputs run.
  - Attempt ranking: train-verified program output → direct grid → unverified program output → test input.
  - Self-tests.
- **Cell 5:** engine.
  - Same as the PIV C1 run: TP4, language_model_only, CUDA graphs on, custom all-reduce off, prefix cache off.
  - Plus `max_model_len` 65536, `max_num_seqs` 16, and `speculative_config={"method":"mtp","num_speculative_tokens":2}`.
  - If the engine refuses MTP, it is retried without it.
- **Cell 6:** continuous scheduler on vLLM's step API (`llm.llm_engine.add_request/step/abort_request`, DELTA token
  ids). A finished task frees its slot at once; there are no waves.
  - Phase 1: `max_tokens = THINK_BUDGET` (45,056).
  - Thinking closed and EOS → done.
  - Thinking closed but cut by length → continue (`prompt + generated`, up to 8,192 more tokens).
  - Never closed → Qwen's documented stop sentence + `</think>` appended token-exactly, then answer (8,192).
  - Deadlines: no new task after the admission deadline; at the drain deadline in-flight thinking is force-closed
    (answer 4,096); at the hard stop everything is aborted.
  - `submission.json` is rewritten after every task.
- **Cell 7:** report, including tok/s, modes and a 240-task time projection, plus the bundle.

## Verified against vLLM 0.19.0 source (wheel from the wheelhouse we use)
- `Qwen3_5MTP` exists. For `model_type == "qwen3_5"`, `SpeculativeConfig` builds the MTP draft with
  `n_predict = mtp_num_hidden_layers` (=1 in our model) — `config/speculative.py` 323-332.
- Our checkpoint has `mtp.safetensors`: FP8 block-quantised weights with `weight_scale_inv`, and `mtp.fc` plus the
  norms in the FP8 ignore list. The draft prefix is `mtp`, so the ignore entries match.
- `LLMEngine.add_request(request_id, {"prompt_token_ids": ...}, params)`, `step()` and `abort_request(ids)`. Outputs
  carry the external request id (`v1/engine/llm_engine.py`, `output_processor.py`). `LLM.generate` itself is this loop
  (`entrypoints/llm.py` 1989).
- `SamplingParams(detokenize=False, output_kind=RequestOutputKind.DELTA)` exists.

## Local harness (`harness_q27x.py`, output in `harness_output.txt`)
Real notebook cells, real Qwen tokenizer and template, real competition JSON, and a fake vLLM step engine. All four
scenarios pass:
- **mixed:** natural close / forced close / cut-and-continue / wrong code with a right grid / crashing code / no answer.
  Every task gets the expected attempt source and score. The forced phase-2 prompt ends with the stop text.
- **drain:** in-flight tasks are force-closed at the drain deadline and aborted at the hard stop; every task is recorded
  and submitted.
- **rerun:** 240 hidden-test tasks; the submission covers every task and query.
- **MTP refused:** the engine is re-created without speculative decoding.

## Known risks (cannot be tested locally)
1. Real speed on 4×L4 with MTP and 16 concurrent 64k sequences (the commit run measures it and projects 240 tasks).
2. Whether the KV cache keeps 16 long sequences without preemption. The KV capacity cannot be read from the front-end
   process, so concurrency is a fixed 16.
3. If MTP fails at engine init, freeing GPU memory before the retry may not work.
4. Prompt quality: a single prompt, no REPL tool turns during thinking. Qwen3.8 only calls tools after `</think>`.

## Questions
1. Any defect in the scheduler's phase logic or deadline handling?
2. Any vLLM 0.19 API misuse?
3. GO / NO-GO for the commit run (EVAL12, ~1–1.5 h on 4×L4)?
