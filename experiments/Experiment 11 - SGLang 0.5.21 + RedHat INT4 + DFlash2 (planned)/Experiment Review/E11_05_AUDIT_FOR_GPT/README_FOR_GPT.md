# E11-05 audit request: S49 Poetiq solver ported from vLLM FP8 to SGLang 0.5.21 + INT4 + DFlash2

You are auditing a Kaggle notebook before it uses a 12-hour GPU run and a scarce leaderboard submission (about 1 every 2 days).
Please be adversarial. Find real bugs, not style issues. For every claim, say whether you verified it against the
SGLang 0.5.21 source (excerpts are in `sglang_src_excerpts/`; the full source is public, tag v0.5.21) or whether it is a
hypothesis. Do many searches (SGLang GitHub issues/PRs about streaming, abort, mamba/hybrid radix cache, DFLASH,
`--incremental-streaming-output`, `--stream-interval`, `/abort_request`) and a deep analysis.

## 1. Context

- **Competition:** ARC Prize 2026 (ARC-AGI-2) on Kaggle. Offline, 4x NVIDIA L4 (24 GB each), 12 h total, about 240 hidden tasks.
- **Model:** Qwen3.8-27B, a hybrid model (Gated DeltaNet linear attention + full attention). It reasons at xhigh effort.
- **Method (S49):** a Poetiq-style program-synthesis loop.
  - The model thinks for up to 49,152 tokens, then writes a Python `transform(grid)`.
  - The program is verified in a sandbox on the train pairs, and the train feedback is fed into the next iteration (up to 10).
  - Expert voting picks attempt_1 and attempt_2.
- **Scores:** S1-60K (same method, 60k thinking) scored LB 34.58 on vLLM 0.19 FP8 + MTP-3. S49 (49k thinking + an ordering tie-break) is pending.
- **Binding constraint:** coverage. A full-eval run solves about 48 % when it has time for all tasks; on the hidden set we run out of time.
- **Engine measurement (E11-04, 2026-10-18; `e11_04_reference/`):** SGLang 0.5.21 + RedHat INT4 W4A16 + the BF16 DFlash2 drafter (W1 flags) gave:
  - KV 881,336 tokens;
  - 12 requests really running;
  - 416 tok/s aggregate at ≤ 32k context, against about 255–300 tok/s for vLLM FP8 + MTP-3.
  - Speed above 40k context was **not** measured.

## 2. What E11-05 changes (everything else is byte-identical to S49)

| Cell | Change |
|---|---|
| CELL 1 config | E11 block: input paths with /kaggle/input discovery, W1 constants, give-up deadline (75 min) |
| CELL 2 install | Tokenizer stack only (vLLM later, only for the fallback). Prefetches INT4 + DFlash2 weights. **Tokenizer check:** the INT4 checkpoint must have the same vocab, probe token ids and EOS as the S49 FP8 tokenizer, otherwise vLLM is used |
| CELL 5 (new) | SGLang venv install (= E11-03/04), W1 launch with diagnosed relaunches (OOM → mem 0.78; lazy rejected → extra_buffer + mamba 70; prefill-graph error → keep the graph), caps parsed from the server log, smoke test (stream + abort), and a **shim** |
| CELL 5b | S49's original engine cell, indented under `if not ENGINE_READY` (the builder asserts the AST is identical) = vLLM FP8 fallback |
| report | Decode rate by context from the server log, shim counters (incl. prefix-cache `cached_tokens`), server stopped at the end |

The scheduler cell (Poetiq loop, ORDER2 queue, drain at hard-stop − 8 min, hard stop, KV gate, voting, submission) is **not touched**.
See `cell_diff_S49_vs_E11_05.diff`.

### The shim (CELL 5): the vLLM API the scheduler uses, mapped onto SGLang HTTP

- **`engine.add_request(rid, {'prompt_token_ids': ids}, params)`**
  - Starts a thread that POSTs `/generate` with `{"rid", "input_ids", "sampling_params": {max_new_tokens, temperature, top_p, top_k, skip_special_tokens: false, no_stop_trim: true}, "stream": true}`.
  - It parses the SSE `data:` lines into token deltas (`output_ids`; the server runs with `--incremental-streaming-output`, so deltas are disjoint).
  - `meta_info.finish_reason.type` (`stop` / `length` / `abort`) becomes the vLLM `finish_reason`. A transport error gives `error`.
- **`engine.step()`**
  - Raises if the server process died.
  - Otherwise it waits up to 50 ms for queued deltas and returns them merged per request, in arrival order.
- **`engine.abort_request(ids)`** POSTs `/abort_request {"rid": id}` for each id.
- **`llm.get_metrics()`**
  - Parses Prometheus `/metrics`. The scheduler's KV gate looks for names containing `kv_cache_usage`, so `sglang:full_token_usage` is exposed under that name.
  - The gate is effectively inactive anyway: CONCURRENCY = 12 = `E10_KV_GATE_MIN_ACTIVE`.
- **Seeds are not sent.** SGLang uses `sampling_seed` only with `--enable-deterministic-inference` (`sampling_batch_info.py:143-157`), so sampling is random. S49 used per-request seeds in vLLM.

### Scheduler behaviour that depends on the engine

- **Phase 1 (thinking):** max_tokens = 49,152. It ends naturally (`</think>` + code + EOS), or with `length`.
- **Forced close:** after `length`, phase 2 continues with a new request, `prompt + phase-1 tokens + "...</think>\n\n"`, with max 8,192 tokens.
- **Drain (hard stop − 8 min):**
  - It aborts every running phase-1 request and immediately starts phase 2 from the tokens received so far.
  - Late deltas of the aborted request are ignored, because the request is already in phase 2.
- **Hard stop:** aborts everything and uses the partial tokens.

## 3. Local harness (CPU, `harness/`)

- **What is real:** cells 0–4 (incl. the tokenizer check with the real S49 FP8 and RedHat INT4 tokenizer files), the SGLang cell (launch, caps, smoke, shim), the fallback cell, the scheduler and the report.
- **What is fake:** the server (`fake_sgl_stream_server.py`: real log format, HTTP/1.1 chunked SSE like uvicorn, disjoint deltas, abort, /metrics, decode lines), the sandbox verifier, and a `vllm` package for the fallback.
- **Results:** see `harness/logs/`. All 10 scenarios pass:

| Scenario | What it proves |
|---|---|
| mixed | ORDER2 admission order. Forced close: the phase-2 prompt is exactly the phase-1 prompt + every streamed delta + stop text. Cap 49,152. Poetiq sampling sent. Finish reasons mapped. Prefix-cache stats recorded. Full thinking saved. S1 fixes F1/F3/A2 hold |
| kvgate | The gate reads SGLang usage through the shim and holds at the floor |
| hardstop / slowverify | Aborts at the hard stop; verifications harvested; queued ones cancelled |
| drain | 2 slow phase-1 requests are aborted on the server. Phase 2 = prompt + received deltas (one had 32 tokens in flight, which were dropped as in vLLM) + stop text. No phase-1 admission after the drain |
| rerun | Hidden mode writes the submission |
| lazyfail | Lazy rejected → relaunch with extra_buffer/70 → 12 effective |
| launchfail | Permanent launch error → S49 vLLM path (fake) runs the whole solver |
| abortbroken | Server ignores abort → smoke fails → fallback |
| serverdeath | Server crash mid-run → solver stops with a clear error; submission and report still written |

## 4. What we want you to check (please answer each)

1. **Streaming protocol:**
   - With `--incremental-streaming-output`, are `output_ids` in every SSE chunk disjoint and complete (no token lost or duplicated), including with DFLASH speculative decoding (several tokens per step), `--stream-interval 8` (`output_streamer.py:427-436`: `len % 8 == 1`), and the final chunk?
   - Is `finish_reason` always present in the last chunk? Is its `type` always one of stop / length / abort?
2. **Abort:**
   - Does `/abort_request {"rid": X}` of a running *streaming* request free its KV + mamba slots immediately?
   - Does the stream then always end with a `finish_reason.type == "abort"` chunk?
   - What happens to an abort for a rid that is still queued or still prefilling (12 prompts of ~8–60k tokens at the drain)?
3. **Drain cost:** E11-04 saw 0 cached tokens for continuations of aborted requests (a 30k aborted prefix + 1,024 answer tokens took 24.5 s alone; finished prompts were 96.5 % cached, 10x faster; see `e11_04_reference/results.json` W1 `contcache`). Phase 2 after the drain therefore re-prefills prompt + up to 49k tokens for up to 12 requests, about 0.6M tokens, within 8 minutes.
   - Is that feasible on 4×L4 INT4 (TP4, chunked prefill 4096)?
   - Is there a better way (for example, not aborting, but sending phase 2 to the same rid)?
4. **Prefix cache with `extra_buffer_lazy`:** is a naturally finished request's prompt + output inserted into the radix cache (including mamba state at the right position), so that the forced-close phase 2 and Poetiq iteration 2 get cache hits?
5. **Long context speed:**
   - How will DFLASH (8 draft tokens) + INT4 decode behave at 40–60k context per request with 12 running?
   - Any SGLang flag that helps (e.g. `--speculative-num-draft-tokens`, attention backend for sm89, `--enable-mixed-chunk`)? Only suggest flags that exist in 0.5.21.
6. **Shim races:**
   - urllib streaming threads (one per request, ≤ 13 alive) and the queue merge in `step()`.
   - The rid lifecycle: phase-1 rid `tid~e0~iK|1`, then `|2`. rids are never reused while alive.
   - Is anything unsafe if SGLang emits a chunk after the finish chunk?
7. **Failure paths:**
   - The give-up deadline (75 min) and the fallback ordering (kill the server group, wait for GPU memory < 1 GB, pip install vLLM, run the S49 engine).
   - A server crash mid-run currently ends the solver (the same as a vLLM engine crash). Should we restart the server instead, and how risky is that?
8. **Tokenizer:**
   - The RedHat INT4 `tokenizer.json` has a slightly different pre-tokenizer regex (no `\p{M}`) than the S49 FP8 tokenizer; vocab and merges are equal.
   - We tokenize in-kernel with the S49 tokenizer and send `input_ids`, so the server tokenizer only detokenizes text we do not use. Any risk?
9. **Anything else** that could make this run worse than S49 on vLLM: correctness, speed or the clock.

## 5. Rules for your suggestions

- No thinking budget below 49,152 tokens.
- The full thinking of every request must be saved. It is, in `q27x/poetiq_thinking/`.
- Prefer the newest stack.
- Do not suggest moving the solver back to vLLM. The question is how to make the SGLang port correct and fast.
- Bundle changes: list every change you would make before the submission, most important first.

## 6. Files

- `notebook/Q27-XHIGH-v1_4_4_E11_05_S49_SGLANG_INT4_DFLASH2.ipynb`: the notebook to audit.
- `notebook/S49_source_Q27-XHIGH-v1_4_4_E10_S1_49K_POETIQ_FIX_ORDER2.ipynb`: the base.
- `build/build_e11_05_s49_sglang.py`: the builder (all patches, plus the AST check of the fallback).
- `cell_diff_S49_vs_E11_05.diff`
- `harness/`: harness, fake server, fixture and logs.
- `e11_04_reference/`: the E11-04 analysis, results.json and the W1 server log (real log format, flags, caps).
- `sglang_src_excerpts/`: the source lines the shim relies on.
