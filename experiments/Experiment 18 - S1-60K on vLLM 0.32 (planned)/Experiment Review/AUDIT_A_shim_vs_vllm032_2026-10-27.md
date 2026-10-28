# Audit A: E18 engine cell (vLLM 0.32.0 server + shim) vs the vLLM v0.32.0 source (2026-10-27)

Scope: `Build/e18_engine_cell.py`, embedded as cell 6 of `Q27-XHIGH-E18_S1_60K_VLLM032_S20.ipynb` (sha 07af34e3). Cell 6
was checked against the source file: identical except the inserted `S1_ENGINE_SRC = ...` line. The solver (cell 7) is
byte-identical to S1-60K cell 7. The in-process reference is cell 6 of `Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb`.
The audit was read-only. No project file was changed.

## Evidence used

- vLLM v0.32.0 from GitHub (`raw.githubusercontent.com/vllm-project/vllm/v0.32.0/...`). The local copies in
  `_scratch/vllm_whl/src032/` of `entrypoints/openai/api_server.py`, `completion/serving.py` and `completion/protocol.py`
  were diffed against GitHub v0.32.0: **identical**. Line numbers below refer to v0.32.0.
- Dependency pins from `_scratch/vllm_whl/vllm_0_32_0_requirements.lock`: uvicorn 0.54.0, starlette 1.10.0,
  httptools 0.9.0, transformers 5.19.0, prometheus-client 0.26.0. Sources were fetched from GitHub at those tags.
- The real Kaggle vLLM 0.32.0 server log of the VMS arm `fp8_mtp3_s20` (2026-10-26):
  `_scratch/ems_out/vms/q27x/vms/vllm_server_fp8_mtp3_s20.log`. This is the same model, wheelhouse, TP4 and 20 seqs.
- The model's `generation_config.json` (local copy in Experiment 1 `Q27X_build/sources/model_qwen3-8-27b-fp8-repacked-v1/`):
  `{"eos_token_id": [248046, 248044], "temperature": 1.0, "top_k": 20, "top_p": 0.95, "do_sample": true, ...}`.
  It has no penalty, min_p or max_new_tokens fields.

## Summary

| # | Severity | Finding |
|---|---|---|
| 1 | MEDIUM | A 1,800 s socket idle timeout on each stream has no in-process equivalent. A request that sits preempted or waiting for more than 30 min is ended as `"error"`, and the solver then forces its answer from partial thinking. |
| 2 | MEDIUM | KV capacity at 20 streams (~1.0M tokens) is below 20 × the typical late-stream length. Preemptions are likely, and that is what triggers finding 1. The concurrency formula still gives 20, but the logged number means something different now. |
| 3 | LOW | Race in abort: if `abort_request` runs before `urlopen` returns, the socket is not shut down. The request keeps running until the first chunk arrives. Normal or errored streams are also never closed explicitly. |
| 4 | LOW / UNVERIFIED | `--language-model-only` together with MTP + prefix caching has never run on 0.32 GPU. The source shows no incompatibility, and the ladder covers a failure at the cost of one extra load. |
| 5 | INFO | Every flag in both modes exists in 0.32.0 and means what the comments say. |
| 6 | INFO | `/v1/completions` request fields and the streaming format match the shim's parser (token ids, finish_reason, EOS, first chunk, usage chunk, error chunk, `[DONE]`). |
| 7 | INFO | Sampling matches the in-process `add_request` path for every field, including the ones the shim does not send. Per-request seed is honoured with MTP. |
| 8 | INFO | Abort on client disconnect is verified end to end (uvicorn 0.54 → starlette 1.10 → AsyncLLM.generate → engine-core abort). It is immediate, even for requests that are waiting. |
| 9 | INFO | Both log regexes match 0.32.0's real output. `vllm:kv_cache_usage_perc` is a 0..1 gauge and the only metric whose name contains `kv_cache_usage`. |
| 10 | INFO | The smoke request (temperature 0, top_k 1, top_p 1.0, max_tokens 16, seed) is valid. |
| 11 | INFO | Other checks: admission control is off by default, the first chunk can be large, the uvicorn access log keeps growing, the `wrapt` sitecustomize noise, model loading without `--trust-remote-code`. |

There are no BLOCKER or HIGH findings.

---

## 1. MEDIUM: stream idle timeout (1,800 s) ends long-waiting requests as "error"

**Evidence.**
- `e18_engine_cell.py:19` sets `E18_STREAM_IDLE_SEC = 1800.0`. Line 229 calls `urlopen(req, timeout=E18_STREAM_IDLE_SEC)`, and
  this timeout applies to every blocking `recv` of the SSE stream.
- A running request sends a chunk on every engine step. A request that is **preempted** (back in the waiting queue) or
  **waiting** sends no bytes at all. 0.32's SSE keep-alive is off by default: `sse_keep_alive_interval: int = 0`
  (`vllm/entrypoints/launchers/cli_args.py:156`), and `with_sse_keep_alive` returns the generator unchanged when the
  interval is <= 0 (`vllm/entrypoints/serve/utils/sse_keep_alive.py:21-33`).
- Lines 259-264 turn the resulting `socket.timeout` into `reason = "error"`. In phase 1 the solver (cell 7, lines
  309-314) then sees `reason != 'length'` with no `</think>`, calls `e10_start_phase2`, and force-closes from partial
  thinking. In-process vLLM 0.19 would simply have waited.
- Waiting more than 30 min is realistic at 20 streams. Each stream decodes at about 423/20 ≈ 21 tok/s (VMS s20, 40k
  band), so a 60k-token thinking takes about 48 min. A preempted request can only resume after enough KV blocks free up
  (see finding 2).

**Fix (either one; both is better).**
- (a) Set `E18_STREAM_IDLE_SEC = None` (no socket timeout). A dead server is already detected by `step()` through
  `self.server.alive()` (lines 282-283), and a dead socket raises immediately.
- (b) Add `--sse-keep-alive-interval 60` to `e18_flags` (both modes) and keep a finite timeout such as 600 s. The comment
  line `": keep-alive"` does not start with `data:`, so the existing filter at line 234 skips it. The flag exists and is
  validated as non-negative (`cli_args.py:156`, `:529-531`).

## 2. MEDIUM: KV capacity vs 20 streams; the logged number changed meaning

**Evidence.**
- 0.32.0 logs `"%s KV cache size: %s tokens, Maximum concurrency for %s tokens per request: %.2fx"`, where
  `num_tokens = int(max_concurrency * max_model_len)` (`vllm/v1/core/kv_cache_utils.py:2300-2333`). This is the
  **group-aware effective capacity**.
- S1 on 0.19 used `num_gpu_blocks * block_size`, the whole pool: 781 × 1,600 = 1,249,600 (cell 1 comment; S1 cell 6
  line 195). The two numbers mean different things.
- Real s20 log, line 250: `GPU KV cache size: 1,006,517 tokens, Maximum concurrency for 106,496 tokens per request: 9.45x`.
  So `CONCURRENCY = max(4, min(20, 1,006,517 // 43,000 = 23)) = 20`. The formula still gives the intended 20, so the
  change in meaning does no harm today.
- Capacity check: 20 streams × (prompt about 5-20k + up to 60,000 thinking + up to 8,192 answer) reaches 1.3-1.7M tokens
  when most streams are late at the same time. Most xhigh requests run to the cap (project memory: about 60k out tokens
  per request). That is well over 1.0M. The S1 KV gate (`E10_KV_ADMIT_MAX 0.95`, `MIN_ACTIVE 12`) only stops new
  admissions; it does not stop running streams from growing. Expect recompute preemptions late in long streams. VMS s20
  showed 0 preemptions, but only on short benchmark generations (KV max 0.41 / 0.48), so this is **UNVERIFIED for the
  ARC load**.

**Fix.** Apply finding 1 so that preemptions cost time and nothing else. Record `vllm:num_preemptions_total` in the E18
smoke report (it is already captured by the `preempt` filter in `metrics_snapshot`). Do not change the 60k budget or the
20 streams; that decision belongs to the user.

## 3. LOW: abort before `urlopen` returns, and responses never closed explicitly

**Evidence.**
- `abort_request` (lines 306-323) sets `st["halt"]` and then calls `st["resp"].fp.raw._sock.shutdown(...)`. If
  `st["resp"]` is still `None` (the stream thread is inside `urlopen`, lines 227-229), both calls fail silently. The
  thread then enters `for raw in st["resp"]` and blocks until the first data line arrives (after prefill), and only then
  breaks at line 231.
- On normal finish (break at line 257-258), on an error, and on halt, `_stream` never calls `resp.close()`. The socket
  closes only when `st` is garbage-collected after the thread exits.
- Impact today is small. The window is milliseconds, because the server sends response headers as soon as the handler
  returns the `StreamingResponse`. The solver aborts only requests admitted at least 1-2 min earlier (drain, hard stop),
  or requests that hit a unit error.

**Fix.** Right after line 229, add `if st["halt"]: st["resp"].close(); return`-style handling (still record stats). Wrap the
body of `_stream` in `try/finally: resp.close()` so every exit closes the socket deterministically.

## 4. LOW / UNVERIFIED: `--language-model-only` with MTP, prefix caching and the FP8 checkpoint

**Evidence.**
- The flag exists: `--language-model-only` (`vllm/engine/arg_utils.py:1459-1461`) → `MultiModalConfig.language_model_only`
  ("disables all multimodal inputs by setting all modality limits to 0", `vllm/config/multimodal.py:304-306`;
  `get_limit_per_prompt` returns 0, `:699-700`).
- `Qwen3_5ForConditionalGeneration` builds the vision tower inside `_mark_tower_model(vllm_config, {"image","video"})`
  (`models/qwen3_5.py:513`). With all limits at 0, that tower becomes `StageMissingLayer`
  (`models/interfaces.py:339-379`), and its weights are skipped when loading (`models/utils.py:382`). So the FP8
  checkpoint's `model.visual.*` tensors (listed in `modules_to_not_convert`) are not loaded.
- The only explicit incompatibility is `cudagraph_mm_encoder=True` (`config/vllm.py:1456-1468`). Its default is `False`
  (`config/compilation.py:528`).
- The MTP draft `ModelConfig` does not inherit `language_model_only` (`config/speculative.py:1289-1312`). The draft
  (`Qwen3_5MTP`) has no vision tower, so this has no effect.
- Prefix caching on this hybrid model switches the mamba cache to 'align'. The real s20 log, line 19, shows `Mamba cache
  mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled`, and that run
  worked with MTP 3. That run did not use `--language-model-only`.
- `--language-model-only` itself has not run on 0.32 GPU. (It ran on 0.19 in S1: `language_model_only: True`.)

**Fix.** None required: the ladder falls back to the VMS-exact flags. Plan for one extra load plus GPU release
(roughly 6-8 min, VMS ready time + `_wait_gpu_release`) if attempt 1 fails. The E18 commit run settles it. To confirm the
mode in the log, look for `All limits of multimodal modalities supported by the model are set to 0, running in
text-only mode.` (`config/model.py:950-953`), and check that no `vit attention` line appears.

## 5. INFO: every flag in both modes exists in 0.32.0 and does what the comment says

All references are to `vllm/engine/arg_utils.py` unless noted. The real s20 log, line 9 ("non-default args"), shows the
VMS set accepted as-is.

| Flag | Where in 0.32.0 | Notes |
|---|---|---|
| `--tensor-parallel-size 4` | 1177 | |
| `--kv-cache-dtype fp8_e4m3` | 1365; `CacheDType` literal includes `fp8_e4m3` (`config/cache.py:39-44`) | log line 12: "Using fp8_e4m3 data type to store kv cache"; log 117: scale 1.0 (no checkpoint scales). S1 had the same. |
| `--max-model-len 106496` | 966 | log: "Using max model len 106496" |
| `--max-num-seqs 20` | 1717 | |
| `--max-num-batched-tokens 4096` | 1703 | warning only (`config/vllm.py:2276`, log line 21); proven in VMS |
| `--gpu-memory-utilization 0.9` | 1357 | 0.32 subtracts CUDA-graph memory (log 245: effective 0.8613); capacity above already includes this |
| `--disable-custom-all-reduce` | 1336 | log 49: PYNCCL only |
| `--seed 20261005` | 953 | engine-level seed; per-request seeds are sent too |
| `--host` / `--port` / `--served-model-name q27` | 1006 (served name) | `model: "q27"` passes `_check_model` |
| `--attention-backend FLASHINFER` / `TRITON_ATTN` | 1086; parsed with `AttentionBackendEnum[value.upper()]` (`config/attention.py:183-193`); both enum members exist (`v1/attention/backends/registry.py:48`, `:71`) | must not be combined with `attention_config.backend` (2687-2691); the cell does not set that |
| `--distributed-executor-backend mp` (s1) | 1135; `Literal["ray","mp","uni","external_launcher"]` (`config/parallel.py:37`) | same as the default for single-node TP |
| `--dtype auto` (s1) | 952; `ModelDType` includes `auto` (`config/model.py:99`) | |
| `--load-format safetensors` (s1) | 1057; registered loader (`model_loader/__init__.py:46,63`) | |
| `--language-model-only` (s1) | 1459-1461 | see finding 4 |
| `--trust-remote-code` (vms) | 950 | harmless: the checkpoint has no custom code, and `qwen3_5` is native in transformers 5.19.0. Leaving it out in s1 is also fine. |
| `--enable-prefix-caching` / `--no-...` | 1370-1375 (BooleanOptionalAction, default None) | 'align' mamba mode, see finding 4 |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | 1826-1828 (`json.loads`), `qwen3_5_mtp` in the MTP list (`config/speculative.py:52`) | log 16-17, 27: MTP accepted, `num_spec_tokens=3` |

## 6. INFO: the /v1/completions request and the streaming format match the parser

**Request fields** (`vllm/entrypoints/openai/completion/protocol.py`):
- `prompt: list[Annotated[int, Field(ge=0)]] | ...` (52-58): a flat list of ints is accepted. The renderer passes token
  ids through without re-tokenizing. `add_special_tokens` defaults to True (117-123), but the tokenizer has
  `add_bos_token: false` and `bos_token: null`, so nothing is added.
- `seed` (66), `top_k` (77), `skip_special_tokens` (86), `return_token_ids` (171-180), `stream` (68), `max_tokens` (63),
  `temperature` (71) and `top_p` (72) are all declared fields.
- `max_tokens` is clamped by `get_max_tokens` = min(max_model_len − prompt_len, request.max_tokens, override, platform)
  (`serve/utils/api_utils.py:169-205`). The override is `None` for `generation_config="auto"` without
  `override_generation_config` (`completion/serving.py:82-86`), and the platform maximum is `sys.maxsize`
  (`platforms/interface.py:224-225`). The solver's caps (prompt + cap + 8,192 + 256 ≤ 106,496 in phase 1;
  `MAX_MODEL_LEN − len(pref) − 8` in phase 2) are never clamped or rejected.

**Streaming** (`completion/serving.py:279-503`):
- Token ids: each chunk carries `choices[0].token_ids = as_list(output.token_ids)`, the delta of that step (417-421;
  documented at protocol.py:171-180, "token_ids contains the delta tokens for each chunk"). Serialization uses
  `exclude_unset=True` (442), but `token_ids` and `finish_reason` are always passed to the constructor, so they are
  present (`token_ids` as a list; `finish_reason` as null or a string).
- The first chunk also carries `prompt_token_ids` (369-371, 416). The shim ignores that key, which is correct. Empty
  chunks during prefill are suppressed (373-379).
- Final chunk: `finish_reason` is set from `CompletionOutput.finish_reason = str(finish_reason)` ("stop" / "length" /
  "abort" / "error") (`v1/engine/output_processor.py:470-474`). A final chunk with empty `token_ids` is still forwarded
  by the shim (line 255).
- EOS is included in `token_ids`. `check_stop` finishes the request on the last sampled token without removing it
  (`v1/core/sched/utils.py:98-110`). The output processor forwards `new_token_ids` unchanged (722, 789). Tokens are
  trimmed only past a matched stop **string** (`_trim_surplus_tokens`, 768-773, 832-845), and none is used. This matches
  the in-process path, which runs the same OutputProcessor, so S1's EOS stripping in `e10_start_phase2` still applies.
- Usage chunk: sent only if `stream_options.include_usage` or `--enable-force-include-usage` (458-492;
  `should_include_usage` at api_utils.py:289-301). The shim sends neither, so there is no usage chunk. One would have
  `choices: []`, which the parser ignores anyway.
- Errors: `finish_reason == "error"` raises `GenerationError` (`generate/base/serving.py:200-207`), which is streamed as
  `{"error": {...}}` (183-217; `ErrorResponse{error: ErrorInfo}` in `serve/engine/protocol.py:79-88`). The shim checks
  `"error" in d` (line 240). `data: [DONE]` always comes last (503). An HTTP 400 before streaming makes `urlopen` raise
  `HTTPError`, which becomes `"error"` (the same handling as a timeout).
- Empirical: the VMS harness (Exp 14 `build_vms.py:216-293`) used the same body (token-id prompt, `return_token_ids`,
  stream) and the same parser on Kaggle with 4k / 40k / 90k prompts. Every arm reported 0 errors. The first chunk with
  a 90k-token `prompt_token_ids` list parsed fine; `HTTPResponse` readline has no body line limit.

## 7. INFO: sampling is equivalent to the in-process path

**Evidence.**
- Server defaults: `default_sampling_params = model_config.get_diff_sampling_param()` (`completion/serving.py:80`) reads
  only `repetition_penalty, presence_penalty, frequency_penalty, temperature, top_k, top_p, min_p, max_new_tokens` from
  generation_config (`config/model.py:1847-1902`). Real log line 302:
  `{'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}`.
- Precedence is request → server default → OpenAI default (`protocol.py:302-311`, `:351-359`). The shim always sends
  temperature (`POETIQ_SOLVER_TEMPERATURE` = 1.0), top_p 0.95 and top_k 20, so those are applied exactly. The fields the
  shim does not send resolve to repetition_penalty 1.0, presence 0.0, frequency 0.0 and min_p 0.0. Those are exactly
  the `SamplingParams()` defaults used in-process. Also unchanged: `min_tokens` 0 (protocol 85), `ignore_eos` False (84),
  `include_stop_str_in_output` False (83), `stop` [] (67), `n` 1 (64). `stop_token_ids` is [] plus any server-default
  stop ids; `get_diff_sampling_param` returns none (363-371).
- EOS / stop ids come from generation_config through `update_from_generation_config` in the InputProcessor, on **both**
  paths (`v1/engine/input_processor.py:64`, `:489-490`; `sampling_params.py:772-800`). Primary EOS 248046 plus stop id
  248044 are the same on both.
- `detokenize=False` cannot be sent over the OpenAI API, so the server also detokenizes text. That costs API-server CPU
  only; token ids are unaffected. `skip_special_tokens` likewise affects text only.
- Seed with MTP: a seeded request carries `SamplingType.RANDOM_SEED` (`sampling_params.py:1190-1196`). The sampler leaves
  FlashInfer and uses the native path with per-request generators whenever generators exist
  (`v1/sample/ops/topk_topp_sampler.py:283-297`). The rejection sampler draws its uniforms from the per-request
  generators (`v1/sample/rejection_sampler.py:465`, `:625-671`). The model-runner-V2 spec-decode sampler takes
  `sampling_states.seeds` (`v1/worker/gpu/spec_decode/rejection_sampler.py:278`). So the seed is honoured. Outputs are
  not bit-identical to 0.19 (different kernels and batch composition), but they were not deterministic across batches
  before either.
- `min_p` and `logit_bias` would be rejected under speculative decoding (`sampling_params.py:1180-1192`); neither is sent.

## 8. INFO: abort on client disconnect works, is immediate, and frees the KV

**Evidence chain** (dependency versions from the lock file; that the Kaggle wheelhouse holds exactly these versions is
**UNVERIFIED**, though the VMS run behaved consistently with them):
1. uvicorn 0.54.0 (httptools is installed, so `http=auto` picks it) advertises `"spec_version": "2.3"`
   (`uvicorn/protocols/http/httptools_impl.py:228`). On FIN/RST, `connection_lost` marks the cycle disconnected (116-126),
   and `receive()` then returns `{"type": "http.disconnect"}` (577-578).
2. starlette 1.10.0 `StreamingResponse.__call__`: for spec < 2.4 it runs `listen_for_disconnect(receive)` alongside the
   stream and cancels the task group on disconnect (`starlette/responses.py:262-285`). So a disconnect is noticed even
   while the request produces no tokens (waiting or preempted).
3. The cancellation reaches `AsyncLLM.generate`, which catches `(asyncio.CancelledError, GeneratorExit)` and calls
   `await self.abort(q.request_id, internal=True)` (`v1/engine/async_llm.py:779-781`). That aborts in the OutputProcessor
   and in EngineCore (`abort_requests_async`, 911-919), and the scheduler frees the request's blocks.
   `merge_async_iterators` (`utils/async_utils.py:100-133`) and `_keep_alive_stream` (`sse_keep_alive.py:57-65`) both
   `aclose()` upstream generators in `finally`, so cleanup does not depend on GC. Before the response starts,
   `with_cancellation` (`api_utils.py:51-93`) covers the handler.
4. The client side, `shutdown(SHUT_RDWR)` on the socket, sends a FIN at once, so the abort lands within one event-loop
   iteration. Exception: the race in finding 3.
- An explicit alternative exists if wanted: route `POST /inference/v1/abort_requests` (real log route list). It needs a
  known request id (the `request_id` body field, protocol.py:146-153). It is not needed.

## 9. INFO: log and metric parsing

- `r"GPU KV cache size: ([\d,]+) tokens"` matches `kv_cache_utils.py:2327-2333` (`f"{num_tokens:,}"`, prefix `GPU` from
  `_kv_cache_device_name`, 2309-2311; it would read `GPU + host` only with hisparse, which is unused). Real log line 250
  gives 1,006,517. The line is logged by EngineCore, whose stdout goes to the same log file.
- `r"version (\d+\.\d+\.\d+)"`: the log logo is `"... version ${w}%s${r}"`, with colours stripped when the formatter is
  not `color` (`api_utils.py:325-347`). Colour is off because stdout is a file (`logger.py:63-72`, isatty False). Real
  log line 5 reads `version 0.32.0` and is the first match in the file.
- `vllm:kv_cache_usage_perc` is a gauge set to `scheduler_stats.kv_cache_usage` in 0..1 (`v1/metrics/loggers.py:619-626`,
  `:1165`). It is the only 0.32 metric whose name contains `kv_cache_usage`, so the solver's gate
  (`'kv_cache_usage' in name`, max over series) reads it correctly. There is one series with TP=4 / DP=1.
- The Prometheus regex handles `name{labels} value`. `_bucket` and `_created` lines are dropped. Counters appear as
  `..._total` (e.g. `vllm:num_preemptions_total`), and the `preempt` / `generation_tokens` / `prompt_tokens` /
  `prefix_cache` / `spec_decode` substring filters still match. The snapshot rows have a different shape from S1's
  in-process objects (`_sum` / `_count` as separate names, no `values` list). This affects reporting only.

## 10. INFO: the smoke request is valid

`temperature 0.0` is sent explicitly and kept (`is not None`, protocol.py:353-356). Greedy then forces top_p 1.0,
top_k 0 and min_p 0 (`sampling_params.py:612-616`). top_k 1 passes validation (677-685). `n=1` passes
`_verify_greedy_sampling` (766-770). A seed with greedy is allowed. `max_tokens 16` ≥ 1 (688-690). The expected finish is
`"length"`, which the smoke accepts.

## 11. INFO: other checks

- Admission control: `max_num_queued_reqs` / `max_num_queued_tokens` default to `None`
  (`engine/arg_utils.py:606-607`), so `check_admission` never returns 503 (`v1/engine/async_llm.py:316-374`).
- `/health` answers only after the engine is up: in the real log, routes come up after "GPU KV cache size" and
  CUDA-graph capture. So `wait_ready` + smoke is a valid readiness test. Real s20 readiness was about 5.5 min
  (13:08:55 → 13:14:27).
- uvicorn logs one access line per `/metrics` GET. The KV gate polls every 5 s while 12 or more requests are active,
  which is roughly 8.6k lines in 12 h; harmless. `--disable-uvicorn-access-log` exists (`cli_args.py:309`) if wanted.
- `Error in sitecustomize ... No module named 'wrapt'` appears in the real VMS log (lines 1-2). It is harmless noise
  from the image's sitecustomize in the venv.
- `kill()` sends SIGTERM to the process group (`start_new_session=True`). In the real log, the 0.32 shutdown force-kills
  EngineCore and the 4 workers within the same second. `_wait_gpu_release` then guards the next attempt.
- The env passed to the server inherits `LIBRARY_PATH` (the libcuda shim; `ensure_libcuda_linkable` runs before
  `e18_install` copies `os.environ`), `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (already the 0.32 default, log 245),
  and `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- Worst-case startup if attempts 1 and 2 both fail late: 30 + 20 + 20 min of wait limits plus the 0.19 fallback load.
  That only matters if the server is broken, and the 12 h budget still holds.

## Fixes, in order of value

1. `E18_STREAM_IDLE_SEC = None`, or add `--sse-keep-alive-interval 60` with a finite timeout (finding 1).
2. Close the response in `_stream` on every exit, and handle `halt` right after `urlopen` returns (finding 3).
3. Report `vllm:num_preemptions_total` and `vllm:num_requests_waiting` from the smoke run, to measure finding 2 on real
   ARC streams.
