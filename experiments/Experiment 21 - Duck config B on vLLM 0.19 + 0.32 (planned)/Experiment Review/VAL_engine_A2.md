# E21 engine cell: independent validation A2 (second reviewer)

**Date:** 2026-10-28. Read-only review: no project file was edited and nothing on Kaggle was touched.

**Scope:**
- the engine cell (notebook cell 7 = builder `engine_cell()`);
- the launch commands (`src/e21_flags.json`);
- RedHat INT4 compressed-tensors W4A16 + MTP on vLLM 0.19;
- `--mamba-ssm-cache-dtype float16` on vLLM 0.19;
- prefix caching + MTP (mamba `align`);
- the server lifecycle: start, ready, smoke, kill, GPU release, fallback, relaunch.

**Inputs read:**
- `README.md`, `build_e21.py` and `e21_engine_new.py`. The current file is 16,267 bytes (21:20). It already contains the `e11_kill_server(wait)` fix.
- Both built notebooks, `ad5b2d1f` (NB1) and `b968e322` (NB2):
  - their cell 7 is byte-identical to `engine_cell()` output;
  - the NB1 and NB2 engine cells are identical;
  - their cell 1 differs only in the variant block.
- Donor cells `E12V_SMOKE1_c06.py`, `E18_R2_c06.py`, `VMS_c06.py` and `VMS_c07.py`.
- vLLM v0.19.0 sources (raw GitHub). Files:
  - `qwen3_5.py`, `qwen3_5_mtp.py`, `qwen3_vl.py`;
  - `compressed_tensors.py` and `compressed_tensors/utils.py`;
  - `linear.py`, `model_loader/utils.py`, `base_loader.py`, `spec_decode/eagle.py`;
  - `models/config.py`, `layers/mamba/mamba_utils.py`, `gdn_linear_attn.py`;
  - `kv_cache_interface.py`, `kv_cache_utils.py`, `interfaces.py`, `openai/cli_args.py`;
  - plus the local `_scratch/vllm_src/v0.19.0` files.
- compressed-tensors 0.14.0.1: `quant_args.py`, `quant_scheme.py`, `quant_config.py`. This is the version vLLM 0.19 pins (`requirements/common.txt`).
- vLLM v0.32.0: `launchers/cli_args.py`, the tool and reasoning parser registries, and the tree listing.
- Real Kaggle logs:
  - E12-V SMOKE1 run 2 (vLLM 0.19 FP8 Duck server);
  - VMS rev 2g `_scratch/ems_out/vms/q27x/vms/*.log` (vLLM 0.32: INT4, FP8, FP8 + SSM fp16).
- INT4 `config.json` and `tokenizer_config.json` (the E13 fixture copies) and the real FP8 `model.safetensors.index.json`.

**Summary:**
- No BLOCKER found.
- By source reading (plus a matching simulation), the INT4 + MTP path loads on vLLM 0.19.
- **2 HIGH:**
  - the smoke accepts any non-empty output;
  - NB1's FP8 fallback is unstaged and cold.
- **6 MEDIUM, 8 LOW.**

---

## Findings

### H1 (HIGH): the launch smoke accepts any non-empty text, so a broken engine passes as "ready"

**Location:** `e21_engine_new.py` `e21_launch()`, line 131 (cell 7, the same code):

```python
"ok": bool(gen) and isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0
and isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] > 0}
```

**Evidence:**
- The smoke asks "what is 2 + 2?" but never checks the answer.
- E12-V run 2's real answer was `"2 + 2 equals 4."` (smoke log).
- E13 on 0.5.21 loaded and then produced `"!!!!"` garbage (E13 memory). That output passes this check.
- NB1 half A = RedHat INT4 on vLLM 0.19 is explicitly untested on GPU (README Finding 5). The hidden rerun runs half A only.

**Failure scenario:**
- INT4 on 0.19 loads with a wrong dequant path (for example a GDN `in_proj_qkvz` packing or a Marlin issue), or a bad KV-scale combination.
- The model answers with non-empty garbage, and the smoke passes.
- `ENGINE_INFO["used"] = "primary"` is recorded.
- The whole half (on the rerun: the whole LB submission) runs on a broken engine. The NB1 fallback (FP8, proven) is never tried.
- The report shows the primary config as working.

**Proposed fix:**
- Make the smoke content-checked: `ok = ... and "4" in gen` (the reply also must not be one repeated character). Raise otherwise, so the fallback runs.
- Optionally add a second deterministic prompt, for example "Reply with the digits 1 2 3 4 5 6 7 8".
- Optionally record the server's first "Mean acceptance length" line after the smoke. MTP acceptance near 1.0 means the target and the BF16 draft disagree, which is a garbage signal.

### H2 (HIGH, conditional): NB1's FP8 fallback reads 30 GB cold from `/kaggle/input`, so the safety net for the untested INT4-on-0.19 path can time out

**Locations:**
- cell 1, E21 block: `E21_STAGE_PATHS = [SGL_MODEL_PATH]` (NB1 stages only INT4);
- cell 2 (`E15B` verbatim): `if os.environ.get("Q27X_SKIP_INSTALL") != "1" and E14_STAGING is None:` (the page-cache prefetch is skipped whenever staging runs);
- `e21_model_path()`: `E21_STAGED.get(weights) or (SGL_MODEL_PATH if weights == "int4" else MODEL_PATH)`;
- `e21_launch()`: `wait_ready(max(60.0, min(E21_LAUNCH_TIMEOUT_SEC, left)))` with `E21_LAUNCH_TIMEOUT_SEC = 30 * 60`.

**Evidence:**
- FP8 = 18 shards, about 30 GB (16 layer shards + `outside` + `mtp`; real index).
- Memory: since 10-21, `/kaggle/input` GCSFUSE is slow; the SGLang INT4 (about 20 GB) needed 1,197 s to become ready.
- In NB1 nothing ever reads the FP8 files before the fallback launch.

**Failure scenario:**
1. NB1 half A primary (INT4 / 0.19) fails. This is the case the fallback exists for.
2. The FP8 fallback loads 30 GB at GCSFUSE speed: about 1.5 × 1,197 s for the read alone, plus init (291 s on E12-V).
3. That is around 30+ min, so `wait_ready` returns "timeout", the server is killed, and the result is `fatal = "no vLLM config became ready"`.
4. On the hidden rerun (half A only), the submission is CELL 0's fallback, so the LB score is about 0.

Side effect when staging is disabled (`E14L.loadpath_enabled()` false): cell 2's prefetch reads `(MODEL_PATH,)` = FP8 for every vLLM run. NB1 would then pre-warm the fallback but not the INT4 primary.

**Proposed fix:**
- NB1 stages both: `E21_STAGE_PATHS = [SGL_MODEL_PATH, MODEL_PATH]`. The /dev/shm headroom was 185 GB in every earlier run; INT4 + FP8 is about 50 GB.
- Or, at least, start a background `cat` of `MODEL_PATH/*.safetensors` when the primary launch starts.
- Give a fallback launch from an unstaged path its own longer timeout, for example 45 min, bounded by the half's end.

### M1 (MEDIUM): the pre-run capacity check ignores the per-request Mamba blocks; it reports "fits" where the block math says it does not, and it never lowers the stream count

**Location:** `e21_capacity()`:
- `eff = int(tokens * x)`;
- `streams = max(8, min(int(seqs), eff // SGL_KV_PER_REQUEST))`;
- `fits_p50_peak = eff >= seqs * 57000`.

**Evidence (vLLM 0.19 source):**
- `kv_cache_interface.py` `MambaSpec.max_memory_usage_bytes`, align branch: `return self.page_size_bytes * (2 + self.num_speculative_blocks)` = 5 blocks per GDN group with MTP-3.
- There are 3 GDN groups ("Add 3 padding layers": 48 GDN layers + 17 attention layers including MTP, group size 17).
- So a request needs 15 Mamba blocks regardless of its length.
- `kv_cache_utils.get_max_concurrency_for_kv_cache_config` = `num_blocks / blocks_per_request` at full max-model-len.
- E12-V real log: `GPU KV cache size: 337,600 tokens` (= num_blocks // 4 × 1600, so about 846 blocks) and `Maximum concurrency for 135,168 tokens per request: 8.46x`. Then 846 / 8.46 = 100 = 85 attention blocks + 15. Confirmed.
- On vLLM 0.32 it is 12 GDN-state blocks per request (E18 README "Pre-run capacity check").

So the effective-token figure x × max_len prices the 15 (or 12) blocks once per 135k tokens. A 57k request pays them in full.

Pools come from the real logs. Needs are at the R3K2 per-attempt p50 peak, 57k: 36 blocks of 1,600 tokens or 69 blocks of 832, plus the Mamba blocks.

| Half / config | Pool (blocks) | Needed at p50 peak | Block math | `e21_capacity` reports |
|---|---|---|---|---|
| NB1 A primary (0.19 INT4, 20) | ~1,060 (estimated: 846 + INT4's +2.8 GiB from VMS) | 20 × 51 = 1,020 | 96 % | eff ~1.43 M ≥ 1.14 M, fits (25 % margin) |
| NB1 A fallback = NB2 A fallback (0.19 FP8, 16) | 846 | 16 × 51 = 816 | 96 % | 1.14 M ≥ 0.91 M, fits |
| NB1 B primary (0.32 INT4, 20, no LMO) | 12.06 × 79 = 953 | 20 × 48 = 960 | **101 %, does not fit** | 1.33 M ≥ 1.14 M, **fits** |
| NB1 B fallback = NB2 B fallback (0.32 FP8, 16) | 9.45 × 79 = 747 | 16 × 48 = 768 | **103 %, does not fit** | 1.04 M ≥ 0.91 M, **fits** |
| NB2 B primary (0.32 FP8, SSM fp16, 832-token blocks) | 10.55 × 140 = 1,477 | 16 × 81 = 1,296 | 88 % | fits |
| NB2 A primary (0.19 FP8, SSM fp16, 832-token blocks) | ~1,690 (estimated) | 16 × 84 = 1,344 | ~80 % | fits |

**Failure scenario:**
- The halves run at or above the 0.95 KV admission gate most of the time. The real concurrency is below the configured 20 / 16, or vLLM preempts and recomputes.
- The report header still says "fits" and streams 20 / 16, so readers attribute the throughput difference to the engine version.
- This is not silent: `server_running_mean` will show it. But the pre-run check does not do its job: it was the reason for the "No GPU run without a capacity check" rule.

**Proposed fix:**
- Compute on blocks:
  - parse `Setting attention block size to (\d+) tokens` → B;
  - m = 15 on 0.19, 12 on 0.32 (or 3 × (2 + num_spec) for 0.19);
  - pool = round(x × (ceil(L / B) + m));
  - need(c) = s × (ceil(c / B) + m);
  - streams = min(seqs, floor(pool × 0.95 / (ceil(50,000 / B) + m))).
- Print both the p50-peak and the mean-context utilisation.
- Correct the README numbers. "FP8 effective KV is about 1.03-1.11 M" is the concurrency figure, not the per-request capacity.

### M2 (MEDIUM): half B (vLLM 0.32) is launched without `--language-model-only`, so it loads the vision tower and gets about 10 % less KV than it could; this confounds the 0.19-vs-0.32 comparison

**Location:**
- `e21_flags_v032()` (VMS arm: `--trust-remote-code`, no `--language-model-only`);
- all four `*:B_*` commands in `e21_flags.json` (`lmo False`).

**Evidence:**
- The VMS INT4 log on 0.32 shows the vision tower loaded:
  - "Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum feature size";
  - "Using backend AttentionBackendEnum.FLASH_ATTN for vit attention";
  - "Warmed position embedding and vision rotary kernels".
- E18 README: the 0.32 pool is 747 blocks without `--language-model-only` and about 823 with it. E18 R2's smoke ran 0.32 with `--language-model-only` (ladder rung 1), so the flag is proven on 0.32.
- FP8 model loading: 7.52 GiB on 0.32 (vision included) vs 7.15 GiB on 0.19 (language-model-only).

**Failure scenario:**
- Half B has less KV than half A for reasons that are not the engine version: vision tower, encoder-cache profiling, trust-remote-code.
- With M1, half B's fallback and NB1's half B primary are the configs that do not fit at the p50 peak.
- The "0.19 vs 0.32" result then partly measures this flag difference.

**Proposed fix:** add `--language-model-only` to `e21_flags_v032()`. `--trust-remote-code` can stay (it is harmless with local weights) or be dropped for parity with half A.

### M3 (MEDIUM): preemptions are reported from log lines, which vLLM does not write, so the report shows about 0 preemptions whatever happens; the counter is lost because CELL 7 kills the server first

**Locations:**
- cell 6 `_e15b_server_log_summary` (`"preempt_lines": _bad("preempt")`);
- `e11_decode_summary` (`"preemption_lines"`);
- CELL 9 `"preempt_lines"`;
- CELL 7 ends with `e11_kill_server(wait=False)` before `e21_end_half()` runs.

**Evidence:**
- Neither 0.19's nor 0.32's `core_scheduler.py` logs a preemption at INFO or WARNING (searched the local copies for `logger` + `preempt`; no hit).
- The real counter is the Prometheus metric `vllm:num_preemptions_total`. VMS reads it on 0.32; `metrics_snapshot()` also matches it but only prints selected keys.
- By the time `e21_end_half()` runs, the server is dead and `/metrics` is gone.

**Failure scenario:** under the M1 KV pressure, vLLM preempts and recomputes. The E21 report still prints `preempt lines 0`, so the half looks clean.

**Proposed fix:**
- In the redefined `e11_kill_server()` (already E21's own code), snapshot `e21_prom()` before killing. Store these counters under `E21_STATE["halves"][E21_HALF]["final_metrics"]`:
  - `vllm:num_preemptions_total`;
  - `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total`;
  - `vllm:prompt_tokens_cached_total`;
  - `vllm:generation_tokens_total`;
  - the spec-decode counters.
- Report them in CELL 9.

### M4 (MEDIUM): half A's solve window is cut by its own setup time, so the halves are not equal

**Location:** `e21_begin_half()`: `E21_HALF_END_PC = _stop if (...) else _stop - E21_HALF_SOLVE_SEC - E21_B_SETUP_RESERVE_SEC`, combined with cell 6 `_end = min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC)` and `_end = min(_end, now + E12_SOLVE_WINDOW_SEC)`.

**Evidence (E21X2EVAL120 profile):**
- HARD_STOP = T0 + 12 h − 3 min − 2 min = T0 + 11:55.
- `_stop` = T0 + 11:52.
- Half A must end by T0 + 11:52 − 5:30 − 0:25 = **T0 + 5:57**.
- So half A gets the full 5.5 h only if its solve starts by T0 + 27 min. Before that come:
  - the vLLM 0.19 install;
  - the Duck package and the sandbox test;
  - the staging join, which waits up to T0 + 55 min (`min(SGL_GIVE_UP_AT - 20 min, T0 + 55 min)`);
  - the launch (E12-V: about 9-11 min to ready);
  - the isolation preflight.
- Half B always gets min(5.5 h, rest), which is normally the full 5.5 h.

**Failure scenario:**
- Half A solves for about 4.9-5.4 h, half B for 5.5 h.
- The headline A-vs-B score (and the "attempt-1-only" score) compares unequal compute without saying so.
- `deadline_in_sec` is in the report, but nothing flags the mismatch.

**Proposed fix:** either
- (a) set half B's window to half A's realised solve seconds (`E21_STATE["halves"]["A"]["e12_run"]["deadline_in_sec"]`), or
- (b) lower `solve_window_sec` (for example to 5.25 h) and assert at half A's solve start that `E21_HALF_END_PC − now ≥ window`, printing a WARNING otherwise.

Also report `window_sec_effective` per half in CELL 9.

### M5 (MEDIUM): the CPU kit's vLLM 0.32 flag check imports a module that does not exist in v0.32.0, so it cannot validate the half-B commands

**Location:** `src/e21_flagcheck.py`: `from vllm.entrypoints.openai.cli_args import make_arg_parser`.

**Evidence:**
- v0.32.0 has no `vllm/entrypoints/openai/cli_args.py`: raw GitHub returns 404, and the tree listing shows the `vllm/entrypoints/openai/` top level holds only `__init__.py`, `api_server.py`, `run_batch.py` and subpackages.
- `make_arg_parser` lives in `vllm/entrypoints/launchers/cli_args.py`. `vllm/entrypoints/cli/serve.py:14` imports it from there.
- `vllm/engine/arg_utils.py` and `vllm/utils/argparse_utils.py` exist in 0.32.

**Failure scenario:**
- The 0.32 run of `e21_flagcheck.py` raises `ModuleNotFoundError` before printing `FLAGCHECK`.
- The kit's `summary` is None, so `all_ok` is False and the kit says NOT OK. Either another kit round is spent, or someone waives the 0.32 check and the half-B commands stay unparsed by the real parser.

**Proposed fix:**

```python
try: from vllm.entrypoints.openai.cli_args import make_arg_parser
except ImportError: from vllm.entrypoints.launchers.cli_args import make_arg_parser
```

Checked by source: `default_chat_template_kwargs`, `enable_auto_tool_choice`, `tool_call_parser` and `enable_prompt_tokens_details` exist in 0.32 `launchers/cli_args.py` (lines 104 / 116 / 122 / 148). `qwen3_coder` and `qwen3` are registered in the 0.32 tool and reasoning parser registries (`tool_parsers/__init__.py:193`, `reasoning/__init__.py:155`).

### M6 (MEDIUM): the INT4 tokenizer / processor files have never been loaded by transformers 4.57.6, which is what the vLLM 0.19 server in the kernel uses, and the kit does not test it

**Locations:**
- cell 2 pins `transformers 4.57.6` / `tokenizers 0.22.2`; the 0.19 server runs with the kernel's Python;
- `build_cpu_kit_e21.py` §3 compares only `tokenizers.Tokenizer.from_file(...)`;
- the earlier SGLang INT4 runs used their own venv, and E11's check used raw `tokenizers`.

**Evidence:** the INT4 `tokenizer_config.json` is in transformers-5 format (written by 5.16.1). Compared with the FP8 one, it:
- adds `backend: "tokenizers"`, `is_local`, `local_files_only`, `model_specific_special_tokens` and `processor_class: "Qwen3VLProcessor"`;
- has no `added_tokens_decoder`, no `additional_special_tokens` and no `chat_template`.

vLLM 0.19 builds its tokenizer (and possibly the HF processor of the multimodal architecture) through `AutoTokenizer` / `AutoProcessor` of the installed transformers 4.57.6. Probably fine, but never executed.

**Failure scenario:**
- A transformers-4 vs transformers-5 config incompatibility makes the INT4 server exit at startup, which falls into H2's slow fallback.
- Or, subtler: different special-token sets or a different chat template change the prompts of half A vs half B (half B uses transformers 5.16.1).

**Proposed fix:** in the CPU kit (vLLM 0.19 + transformers 4.57.6 installed), all on CPU:
- `AutoTokenizer.from_pretrained(INT4)`;
- `apply_chat_template` equality vs FP8 on a Duck-style multi-turn conversation with `reasoning`, `tool_calls` and `preserve_thinking=True`;
- `all_special_ids` equality;
- `AutoProcessor.from_pretrained(INT4)` (or vLLM 0.19's `cached_get_processor`);
- `CompressedTensorsConfig.from_config(cfg["quantization_config"])` + `apply_vllm_mapper(Qwen3VLForConditionalGeneration.hf_to_vllm_mapper)` + `should_ignore_layer(...)` for `mtp.fc`, `mtp.layers.0.self_attn.qkv_proj`, `language_model.model.layers.0.linear_attn.in_proj_qkvz` and `...in_proj_ba`. This is the real 0.19 classes version of the simulation in the "Verified" section.

### L1 (LOW): a watchdog relaunch in the last 20-31 min of a half can never become ready

**Location:**
- `e21_launch()` line 114: `left = E21_HALF_END_PC - time.perf_counter() - 30 * 60`, then `wait_ready(max(60.0, min(E21_LAUNCH_TIMEOUT_SEC, left)))`;
- cell 6 watchdog: `_WD_MIN_LEFT = 20 min`.

**Failure scenario:**
- The server dies with 20-31 min of solving left. The watchdog relaunches, but the wait is 60 s while vLLM needs 5-10 min, so it fails with "timeout".
- `srv.kill()` + `_wait_gpu_release()` (up to 180 s) follow, and the B3 retry repeats the same.
- The last 20-30 min are lost, although a 10-min relaunch would have given 10-20 min of solving.

**Proposed fix:** for relaunches, use `min(E21_LAUNCH_TIMEOUT_SEC, max(60, solve_end − now − 5 min))`. Or raise `_WD_MIN_LEFT` above 31 min, so the two rules agree.

### L2 (LOW): if the vLLM 0.32 install fails, the fallback config runs the same install again

**Location:** `e21_launch()` line 103: `if E21_STATE.get("v032_install") is None: E21_STATE["v032_install"] = e18_install()`. A raised exception leaves it None. `e18_install` has `timeout=2400`.

**Failure scenario:** a broken or slow wheelhouse install (up to 40 min, timeout) is retried for the fallback, which uses the same venv. That is up to 80 min of half B lost, ending with no engine.

**Proposed fix:**
- On failure, store the exception (`E21_STATE["v032_install"] = {"error": ...}`) and raise immediately for every later 0.32 config.
- Optionally make half B's fallback the 0.19 engine, clearly labelled.

### L3 (LOW): the GPU-release result is not acted on before the fallback launch or before half B

**Location:**
- `e21_launch()` except-branch: `info["gpu_release"] = _wait_gpu_release()`;
- `e21_end_half()`: `st["gpu_release"] = _wait_gpu_release()`;
- `e21_relaunch()`: `_wait_gpu_release()`.

None of them checks `released`. E18 broke the ladder on `not released`.

**Failure scenario:**
- Zombie workers keep memory, and the next launch fails at vLLM's free-memory check, or gets a smaller KV pool.
- The fallback fails for the wrong reason: the log shows only "not ready (exited rc=1)".

**Proposed fix:**
- If `released` is False: run `_e18_kill_gpu_pids()` once more, wait again, and record `gpu_not_released` in `ENGINE_INFO`.
- In `e21_end_half()`, store the flag so CELL 9 shows it.

### L4 (LOW): the recorded flag set can differ from the flags actually launched; some labels are hardcoded

**Locations:**
- `E21_FLAGS = {...(_cfg, E21_PORTS[_H], "FLASHINFER")}` (line 270), stored as `CONFIG["e21"]["flags"]` and in the kit's JSON;
- `e21_flags_v032()` hardcodes `{"method": "mtp", "num_speculative_tokens": 3}`, while `e21_flags_v019()` uses `SPECULATIVE_CONFIG` (env `Q27X_SPEC_TOKENS`);
- `ENGINE_INFO["backend"] = f"vllm-...-mtp3-s{seqs}"`.

**Failure scenario:**
- If the libcuda probe fails, both halves run `TRITON_ATTN`, but `CONFIG["e21"]["flags"]` says FLASHINFER. The per-attempt `info["flags"]` / `attention_backend` are correct.
- With an env override of the spec tokens, half A and half B silently differ, and the label still says mtp3.

**Proposed fix:**
- Build `E21_FLAGS` with the actually chosen backend.
- Use `SPECULATIVE_CONFIG` in both builders.
- Derive the label from the flags.

### L5 (LOW): a server-crash relaunch always reuses the crashed config

**Location:** `e21_relaunch()`: `H, cfg = E21_HALF, ENGINE_INFO.get("cfg")`.

**Failure scenario:**
- A config-specific runtime fault (for example an INT4 Marlin / GDN fault at long context on 0.19, or an SSM-fp16 numeric fault) crashes the primary mid-run.
- Both watchdog relaunches restart the same config and crash again. The proven fallback is never used for the rest of the half.

**Proposed fix:** make the second relaunch use the half's fallback cfg, and record it in `relaunches`.

### L6 (LOW): a two-half commit without a solve window starves half B

**Location:**
- cell 1, E21 block: `E21_HALF_SOLVE_SEC = E12_SOLVE_WINDOW_SEC or 0.0`;
- the assert only runs `if E12_SOLVE_WINDOW_SEC and not RERUN`.

**Failure scenario:**
- `Q27X_PROFILE` is set to another eval profile without `solve_window_sec`.
- Half A's end becomes `_stop − 25 min`, and half B gets 25 min minus its setup (about 0).

**Proposed fix:** `assert RERUN or len(E21_HALVES) == 1 or E12_SOLVE_WINDOW_SEC`.

### L7 (LOW): the two vLLM versions share some JIT caches in one container

**Evidence:**
- Half B's env sets `VLLM_CACHE_ROOT=/tmp/vllm-cache` (separate from 0.19's default).
- FlashInfer and Triton caches stay at their home-directory defaults for both versions.
- E21 is the first notebook that runs a 0.19 server and then a 0.32 server in one session. VMS and E18 ran 0.32 only.
- Both caches are keyed by version / hash, so the risk is small.

**Proposed fix:** give half B its own `FLASHINFER_WORKSPACE_BASE` and `TRITON_CACHE_DIR` under `/tmp/vllm-cache`.

### L8 (LOW, documentation and confounds)

1. **README Finding 5** says the INT4 checkpoint "ignores every `linear_attn` (GDN)" and "the GDN layers are BF16". The ignore list names only the module `...linear_attn`, its `norm`, `in_proj_a` and `in_proj_b`.
   - Under vLLM's exact or regex matching, `in_proj_qkv` / `in_proj_z` (fused `in_proj_qkvz`) and `out_proj` of every GDN layer are **INT4**. Only `in_proj_ba` is BF16.
   - Consistent with the checkpoint (VMS 0.32 ran it with MTP acceptance 2.88), but the README is wrong.
2. **The INT4 checkpoint has a `kv_cache_scheme` (static per-tensor fp8).** On INT4, vLLM loads the calibrated `k_scale` / `v_scale` (0.32 log: "Checkpoint does not provide a q scaling factor. Setting it to k_scale"). FP8 runs with scale 1.0 ("Using KV cache scaling factor 1.0").
   - So NB1 primary vs fallback differ in KV quantisation too, not only in weights and stream count.
3. **NB2's SSM fp16 halves the Mamba page.** The attention block size becomes 832 tokens instead of 1,600 (0.32 log: "Setting attention block size to 832 tokens").
   - Align-mode prefix-cache granularity doubles, so NB2 primary vs fallback (and vs SGLang) differ in cache-hit behaviour as well as SSM precision.
   - README Finding 7's "1,600-token blocks" holds only for the fallback.
4. **Half B's CELL 7 rewrites `SUBMISSION_PATH`** (outside the engine cell, noted for completeness). If half B's engine fails, the commit's final `submission.json` is half B's empty run, not half A's. The per-half scores in `e21_report.json` are unaffected, and so is the rerun (half A only).

---

## Verified correct

### Launch commands (`e21_flags.json`, 8 commands, checked by script)

- No duplicated flag in any command.
- `--enable-prefix-caching` is in all 8.
- MTP-3 is in all 8.
- `--language-model-only` is in all four 0.19 commands and none of the 0.32 ones (see M2).
- `--mamba-ssm-cache-dtype float16` appears only in `nb2:A_primary` / `nb2:B_primary`.
- `--max-num-seqs` is 20 for NB1 primaries and 16 elsewhere.
- Model paths: INT4 for NB1 primaries, FP8 elsewhere.
- Ports: 1234 for half A, 8011 for half B, distinct.
- Served names `Qwen/Qwen3.8-27B-INT4` / `-FP8` are not Duck preset aliases.
- The smoke uses `E21_SERVED[cfg["weights"]]`, which equals `--served-model-name`.
- The 0.19 command = the E12-V run 2 command (which worked on GPU) except:
  - the `--tensor-parallel-size` position;
  - `--max-num-seqs` per half;
  - the INT4 path;
  - the SSM flag.
  The E12-V log's non-default args match.
- The 0.32 command = the VMS `vms_flags()` arm + Duck's serving flags + the 135,168 context.

### INT4 compressed-tensors W4A16 + MTP on vLLM 0.19 (source reading, no GPU)

- **Config parsing.** compressed-tensors 0.14.0.1 `QuantizationArgs` (`extra="forbid"`) has every field the 0.18.1 checkpoint writes: `actorder: "static"` (`ActivationOrdering.STATIC`), `scale_dtype`, `zp_dtype`, `observer`, `observer_kwargs` and `block_structure`. `QuantizationScheme` also has `extra="forbid"`, and the fields match. vLLM's `from_config` skips `transform_config: {}` (falsy).
- **Target mapping.** `Qwen3_5ForConditionalGeneration` is not `SupportsQuant`, so `configure_quant_config()` applies `Qwen3VL.hf_to_vllm_mapper`. `CompressedTensorsConfig.apply_vllm_mapper()` maps layer paths (`model.language_model.` → `language_model.model.`) and **preserves `re:` entries** (`compressed_tensors.py:136-140`).
- **MTP prefix and quant config.**
  - The draft model is built with the target's `vllm_config.quant_config` (`eagle.py` `_create_draft_vllm_config()` returns `self.vllm_config`; `get_model(..., model_config=draft_model_config)`).
  - The prefix is `""` (`base_loader.load_model(prefix="")` → `initialize_model`).
  - So `Qwen3_5MTP` builds `Qwen3_5MultiTokenPredictor(prefix=maybe_prefix("", "mtp"))`, giving `mtp.fc` and `mtp.layers.0.{self_attn.qkv_proj, o_proj, mlp.gate_up_proj, down_proj}`.
  - `configure_quant_config` sets the MTP class's own `packed_modules_mapping`.
  - `re:^mtp.*` matches all of them, so they are unquantized. That agrees with the dense BF16 MTP weights (VMS static check on the real checkpoint: `mtp_keys 15, packed 0, not_ignored []`).
- **Matching simulation** (vLLM 0.19's `should_ignore_layer` reimplemented on the real INT4 `config.json`; script at `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\sim\sim.py`):
  - `in_proj_ba`: ignored in all 48 GDN layers, both shards consistent;
  - `in_proj_qkvz`, `out_proj`, `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`: quantized, consistent, so no "different quantization schemes" `ValueError`;
  - all MTP linears: ignored.
- **Weight loading.**
  - `Qwen3_5Model.load_weights` maps `in_proj_qkv` → `in_proj_qkvz` with shard `(0, 1, 2)` and `in_proj_z` → shard 3.
  - `MergedColumnParallelLinear.weight_loader_v2` handles a tuple shard for `PackedvLLMParameter` / group scales (`_load_fused_module_from_checkpoint`, `packed_dim` 1 ≠ `output_dim` 0, so no packing adjustment). `weight_shape` goes through the `BasevLLMParameter` branch.
  - INT4 `weight_scale` names pass through `maybe_remap_kv_scale_name` unchanged.
  - `self_attn.k_scale` → `self_attn.attn.k_scale` (kv_cache_scheme present, so `CompressedTensorsKVCacheMethod` creates the parameters).
  - `lm_head` is in the ignore list (dense), and `ParallelLMHead` is not `LinearBase` anyway.
  - The MTP loader remaps `mtp.` → `model.` and takes `embed_tokens` / `lm_head` from the target names.
- **Kernel shapes.** TP4 / group-128 divisibility holds for the row-parallel inputs: GDN `out_proj` 1,536, `down_proj` 4,352, `o_proj` 1,536 per rank; all multiples of 128.
- **The same checkpoint ran on vLLM 0.32 (VMS, Kaggle 2026-10-26)** with Marlin, MTP acceptance 2.88 (FP8: 2.97) and 518 tok/s at 20 streams.
- **Residual risk:** runtime only (Marlin on 0.19 sm89, the 0.19 GDN kernels with an INT4 `in_proj_qkvz`), covered by H1's content-checked smoke plus the FP8 fallback (after H2).

### `--mamba-ssm-cache-dtype float16` on vLLM 0.19

- `CacheConfig.MambaDType = Literal["auto", "float32", "float16"]` (`config/cache.py:24`).
- `Qwen3_5ForConditionalGenerationConfig.verify_and_update_config`: `auto` → the HF `mamba_ssm_dtype` (float32 for both checkpoints). An explicit `float16` is kept, with the warning "Qwen3.5 model specifies mamba_ssm_dtype='float32' ... Using the user-specified value" (`models/config.py:663-687`).
- `GatedDeltaNetAttention.get_state_dtype()` and `Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config()` both pass `mamba_ssm_cache_dtype`. `_mamba_state_dtype` maps "float16" to `torch.float16` (conv state stays model dtype).
- The GDN forward stores the final state with `.to(ssm_state.dtype)`; the Triton recurrent kernels take the state pointer dtype. The MTP draft layer is full attention (no Mamba state).
- Align-mode constraints: block (832) ≤ `max_num_batched_tokens` (4096); multiple of 8.
- Proven on 0.32 (VMS `fp8_mtp3_ssm_fp16` PASS; KV 1.12 M vs 1.02 M) and on SGLang (production fp16). Not yet run on 0.19: plausible by source, and the fallback covers it.

### Prefix caching + MTP (mamba `align`)

- All 8 commands set `--enable-prefix-caching` and MTP-3. No command disables prefix caching (E19: MTP + prefix off is the failing combination).
- No env var disables prefix caching.
- vLLM 0.19 auto-selects `align` for Qwen3.5 when prefix caching is on (`models/config.py:437-451`; E12-V log: "Mamba cache mode is set to 'align' ... when prefix caching is enabled").
- vLLM 0.32 does the same (VMS log). `Qwen3_5MTP` raises only for `all` mode, which is not used.

### Server lifecycle

- **Start.** `E18Server` (verbatim E18 R2) starts each server in its own session. Logs go per launch to `OUT/vllm_server_half<H>_<tag>.log` (OUT = the half's folder; CELL 9's glob `half*/vllm_server_*.log` matches).
- **Environment.** Half A gets the E12-V env and the libcuda shim (`LIBRARY_PATH` set before any launch). Half B gets the E18 installer env: no `PYTHONPATH`, `CUDA_HOME`, `HF_HUB_OFFLINE`, `VLLM_CACHE_ROOT`.
- **Ready.** Ready means `/health` answers AND that server's own `GPU KV cache size` line is in its log. Both 0.19 and 0.32 print it; 0.19 prints "Maximum concurrency" on its own line and 0.32 on the same line, and the regexes match both.
- **Silence watchdog.** It treats 600 s of log silence as a stall. Startup logs "No available shared memory broadcast block found in 60 seconds" every minute during compile / JIT, so a slow compile is not killed. Per-shard tqdm progress covers a slow GCSFUSE read (about 100 s per shard).
- **Smoke.** It matches E12-V's. A missing `prompt_tokens_details` on a cold request is tolerated (correct for 0.19).
- **Kill.** SIGTERM to the group, then SIGKILL to the group always, then `nvidia-smi` compute-PID kill (never this kernel), then wait for < 1 GB on every GPU.
- **The `e11_kill_server(wait=...)` bug is fixed.** It was a `TypeError` against `E18Server.kill(fast=...)` from CELL 7's final call. E21's own `e11_kill_server` (`e21_engine_new.py:3-7`) comes after the donor's in the cell, so it is the one in force.
- **Double kill.** Killing an already-killed server (CELL 7 kill, then `e21_end_half`) is harmless: errors are swallowed and `log_f.close()` is idempotent.
- **Fallback.** The primary is tried, then the fallback (spec override), and a third launch is not started with < 30 min left. Each attempt is recorded with its flags, backend, capacity, smoke and log tail. `ENGINE_INFO["used"]` / `backend` name the config that actually became ready, and CELL 9 prints it (not silent).
- **Relaunch.** The watchdog relaunch uses the ready cfg on the same port and URL. `SERVER_DOWN` makes the branches wait. The token counter is re-based on relaunch.
- **Ports.** Halves use different ports, so a lingering half-A listener cannot capture half B's traffic.
- **History key.** `ARC3_REASONING_HISTORY_KEY=reasoning` is set in CELL 3 and again at each half's begin. Nothing re-applies `duck_env()` later (checked `arc2_duck.py`). Duck's response parser accepts `reasoning` and `reasoning_content` (`openai_compat.py:151-154`, `tool_agent.py:3440-3442`).
- **Stats line and metrics.** The stats line format is identical in 0.19 and 0.32 ("Engine 000: Avg prompt throughput ... Prefix cache hit rate"). The Prometheus names CELL 6 uses (`vllm:kv_cache_usage_perc`, `num_requests_running` / `waiting`, `generation_tokens_total`) are the ones VMS read on 0.32.
- **Duck client.** Duck's base URL and model id come from `SGL_URL` and `/v1/models` at solve time (cell 6 `solve_all_r6(base_url=SGL_URL + "/v1", model_id=...)`), not from CELL 3's port-30000 `duck_env`.
- **Harness.** Sets `Q27X_GPU_WAIT_MAX=1`, so the unguarded `_wait_gpu_release()` calls do not stall the CPU harness.
