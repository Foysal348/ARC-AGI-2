# Why does SGLang 0.5.21 run only 11 of 12 requests? Root cause and fix

Please do hundreds of searches (SGLang GitHub issues and PRs, especially around v0.5.x: ReqToTokenPool / req slots, `get_num_allocatable_reqs`, mamba / hybrid radix cache, `extra_buffer_lazy`, overlap scheduler, DFLASH / speculative decoding, abort of streaming requests, `max_running_requests`), read the source excerpts here, and do a deep analysis.

For every claim, say whether it is **verified in the source**, **verified in the logs**, or a **hypothesis**. Give the root cause and a fix that we can verify in the next GPU run.

## Setup

- **Engine:** SGLang 0.5.21 (the tag). Qwen3.8-27B (hybrid: Gated DeltaNet + full attention), RedHat INT4 W4A16, BF16 DFlash2 drafter, 4x L4, TP4.
- **Launch flags:** see `e11_05_eval30_run/sglang_server_try0.log`, first lines.
  - `--max-running-requests 12`, `--cuda-graph-max-bs-decode 12`;
  - `--speculative-algorithm DFLASH --speculative-num-draft-tokens 8`;
  - `--mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size 56`;
  - `--mamba-ssm-dtype float16 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.83 --disable-prefill-cuda-graph`;
  - `--chunked-prefill-size 4096 --context-length 155648`;
  - `--incremental-streaming-output --stream-interval 1`.
- **Server state at start:** KV 881,336 tokens. Verify CUDA graphs bs 1–8, 10, 12. No "capped by the mamba state cache" warning (56 / 4 = 14 ≥ 12).
- **Client:** our scheduler always keeps exactly 12 requests in flight.
  - One streaming `/generate` per request (`input_ids`, `stream: true`, `max_new_tokens` 49,152 for phase 1).
  - When a phase-1 request ends with `length`, a phase-2 request with the same prefix + a stop text follows at once (~98 % prefix cached).
  - At the end, a drain aborts running requests via `/abort_request {"rid"}`.
- **Before production:** a smoke test sends a short greedy request (finishes), then a streaming request that is aborted after its first chunk.

## The problem (verified in the logs)

`derived_state_timeline.txt` shows (#running-req, #queue-req) from the TP0 `Decode batch` lines over 3.6 h:
- **11 running + 1 queued** in 2,358 lines; 12 running in only 114.
- Long stretches of 11+1 (up to 35 min), short bursts of 12+0 (20–60 s).

At 11 running the server is far from any memory limit (example line in the timeline):
- `mamba num` 33–35 of 56 (usage ~0.60);
- `full token usage` ~0.49.

**At the very start,** our 12 first requests are sent together after the smoke test. The server admits 11. The 12th (`247ef758~e0~i0|1`) waits ~1,430 s, until the first request finishes (07:03 → 07:25).

`sglang_requests.jsonl` also gives the implied queue waits of phase-1 requests (in the timeline file). They range from minutes up to 24 min.

**Contrast, E11-04** (`e11_04_reference/server_W1_lazy_fp16_try0.log`, same flags, non-streaming `/generate`):
- 12 requests ran together for the full 10-min PROD window (share 1.0).
- There, no request finished or was aborted inside the window, and no streaming abort happened before it.

## Our reading of the source (please verify or refute)

- `Scheduler.get_num_allocatable_reqs` (scheduler.py:3784-3802) limits new admissions by `req_to_token_pool.available_size()`.
- The pool is created with `size=max_num_reqs` = 12 (kv_cache_configurator.py:1066/1111/1138).
- **Hypothesis A:** a request row stays allocated after a request has finished or been aborted. Candidates:
  - the aborted smoke request;
  - overlap scheduling;
  - lazy mamba insert;
  - the spec-decode intermediate state slot (`mamba_spec_state_size=max_num_reqs`);
  - the streaming output path.
  
  That would leave only 11 free rows for new requests.
- **Hypothesis B:** some other PrefillAdder gate (`schedule_policy.py`: `budget_state`, `rem_mamba_slots`, `rem_input_tokens` / chunked prefill, `prefill_budget.check_prefill`) refuses the 12th.
- **Ruled out by the logs:** KV-token exhaustion and mamba-slot exhaustion (see the numbers above).

## What we need

1. **The root cause,** with source lines and any matching upstream issue or PR (with links).
2. **A fix that works in 0.5.21 without patching SGLang,** if one exists. For example:
   - `--max-running-requests 13` while the client keeps 12 (is that safe with `cuda-graph-max-bs-decode 12` and verify graphs up to 12?);
   - an env flag such as `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK`;
   - a different smoke/abort sequence;
   - something else.
   
   Also give its risks: memory, the decode batch exceeding the captured graph sizes, retractions.
3. **A cheap check for the next run** that proves 12 really run: which log lines or metrics to watch.
4. Whether this also explains the per-request rate dispersion. Median 28.3 tok/s per phase-1 request; some at 15.5 tok/s.

## Files

- `e11_05_eval30_run/sglang_server_try0.log`: the full server log, 3,425 lines.
- `e11_05_eval30_run/sglang_requests.jsonl`: one line per request (finish, tokens, sec, prompt / cached tokens).
- `e11_05_eval30_run/kernel_log.json`: the notebook output.
- `e11_05_eval30_run/poetiq_run_report.json`, `poetiq_iterations.jsonl`, `sglang_decode_summary.json`.
- `derived_state_timeline.txt`: (running, queue) runs, all prefill lines, implied waits.
- `e11_04_reference/`: the E11-04 W1 server log (12 really running) and our analyses.
- `notebook/`: the E11-05 EVAL30 notebook (engine cell = launch, shim, smoke).
- `sglang_src_excerpts/`: the scheduler, schedule_policy (PrefillAdder), prefill_budget, kv_cache_configurator (pool sizes, mamba ratio), memory_pool (ReqToTokenPool, HybridReqToTokenPool), batch_result_processor (release), common.py (`release_kv_cache`) and scheduler `abort_request`.
