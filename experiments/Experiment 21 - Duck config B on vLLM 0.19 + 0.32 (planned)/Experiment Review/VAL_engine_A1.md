# VAL engine A1: E21 engine cell (cell 7) validation (2026-10-28)

## Scope and method

- **Scope:** the engine cell (cell 7) of both built notebooks. Cell 7 is byte-identical in NB1 and NB2. Its tail is
  identical to `Experiment Notebooks/Build/e21_engine_new.py` (checked with `diff --strip-trailing-cr`).
- **Line numbers:** "c7:N" is a line of cell 7 of `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb`. "c8:N" is a line of
  cell 8, which holds CELL 6 and CELL 7 wrapped as `e21_cell6()` and `e21_cell7()`.
- **Read-only:** no project file was edited and nothing touched Kaggle.
- **vLLM sources:** fetched from raw.githubusercontent.com at tags v0.19.0 and v0.32.0. They are kept in the scratchpad
  under `vsrc/`.
- **Real logs used as evidence:**
  - E12-V SMOKE1 run 2 on vLLM 0.19:
    `Experiment 12 .../Experiment Results/Kaggle Runs/E12-V vLLM SMOKE1 run2 (esfersami50, 2026-10-19)/q27x/vllm_openai_server.log`.
  - VMS on vLLM 0.32: `_scratch/ems_out/vms/q27x/vms/vllm_server_*.log` (2026-10-26).
  - The E18 smoke on vLLM 0.32: `Experiment 18 .../Experiment Results/E18_SMOKE_GPU_v1_2026-10-27.md`.
  - The E12-V audit note: `Experiment 12 .../Experiment Review/E12-V - GPT audits validated (2026-10-19).md`, line 17.

## Summary

There are no BLOCKERs. Every flag exists in its version and has a valid value. The failure paths kill the server they
own. The serious problems are capacity and comparability:

| # | Severity | Finding |
|---|---|---|
| H1 | HIGH | `e21_capacity` ignores the fixed 15 Mamba blocks per request. It overstates the KV for N streams by 16-58 % and reports "fits" for configurations that are at 95-104 % of the pool at the R3K2 p50 peak. |
| H2 | HIGH | The vLLM 0.32 flags drop `--language-model-only`. Half A has it, and E18 ran it on 0.32. Without it half B gets about 9 % fewer KV blocks, which puts NB1 half B (INT4, 20 streams) and every 0.32 FP8 fallback over the pool at the p50 peak. |
| H3 | HIGH | `src/e21_flagcheck.py` cannot import on vLLM 0.32 (`vllm.entrypoints.openai.cli_args` does not exist in v0.32.0). The CPU kit's 0.32 flag check will always report NOT OK. |
| M1 | MEDIUM | Duck's serving flags (tool parser, reasoning parser, chat-template defaults) never ran on 0.32. 0.32's unified Qwen3 parser ends reasoning at the first `<tool_call>`, which 0.19 does not. The smoke does not exercise either. |
| M2 | MEDIUM | The smoke only checks that the content is non-empty. It cannot catch garbage output on the two untested paths: INT4 on 0.19 and fp16 SSM on 0.19. |
| M3 | MEDIUM | The GPU-release result is recorded but never acted on. The fallback, the relaunch and half B launch on GPUs that may still be occupied. The E18 donor stops in that case. |
| M4 | MEDIUM | A failed `e18_install` is not cached. The 0.32 fallback runs the installer again: up to 2 × 40 min in the worst case. |
| M5 | MEDIUM | In `e21_end_half` the server kill is not in a `finally`. An exception in the records copy leaves half A's server holding the GPUs for half B. |
| M6 | MEDIUM | The watchdog relaunch's ready timeout drops to 60 s in the last 40 min of a half. A relaunch there cannot succeed but still kills every GPU process. |
| M7 | MEDIUM | Half A's end is fixed. Half A gets less than 5.5 h whenever its setup takes more than 27 min (the staging join can take up to T0+55 min), while half B gets the full 5.5 h. |
| L1 | LOW | `e11_kill_server(wait=...)` calls `E18Server.kill(wait=...)`, but that method only takes `fast`. The resulting TypeError is swallowed, so the kill at the end of CELL 7 silently does nothing. |
| L2 | LOW | `_E12V_STAT_RE` misses every stats line that has "Preemptions: N" (both versions) or "Deferred: N reqs" (0.32). The busiest intervals drop out of the throughput, running and KV summaries. |
| L3 | LOW | `kv_tokens_logged` means different things in the two halves: pool/4 on 0.19, max-concurrency × max-model-len on 0.32. |
| L4 | LOW | The harness fake server models neither the fixed Mamba blocks nor the "Preemptions" field, so it cannot catch H1 or L2. |
| L5 | LOW | A half-A watchdog thread that is in the middle of a relaunch when half A ends early can kill half B's GPU processes and set `E11_SERVER = None`. The window is narrow. |
| L6 | LOW | `srv.start()` is outside the `try`. Only the log file handle leaks. |
| L7 | LOW | Only the primary weights are staged. NB1's FP8 fallback loads from `/kaggle/input`, the GCSFUSE slow path. |

---

## 1. Flags (`e21_flags_v019`, `e21_flags_v032`, `e21_duck_serving_flags`; `src/e21_flags.json`)

### H2 (HIGH): the vLLM 0.32 commands drop `--language-model-only`

**Location:** c7:306-319 (`e21_flags_v032`). Compare c7:300, where the 0.19 commands have the flag.

**Evidence:**
- `src/e21_flags.json`: every `nb1:B_*` and `nb2:B_*` command lacks `--language-model-only`, and every `A_*` command has it.
- vLLM 0.32 has the flag: `vllm/engine/arg_utils.py:1460` registers `"--language-model-only"`.
- E18 R2 ran it on 0.32 on GPU. Rung `s1` in `E18_R2_c06.py:104-120` (`f += [... "--language-model-only"]`) was the rung
  that became ready (E18_SMOKE_GPU_v1: "Rung used **s1**: S1's flags + `--language-model-only`").
- The KV effect on 0.32, from real logs (FP8, 20 seqs, 106,496 context):
  - **with** `--language-model-only`: 1,110,415 tokens (E18 smoke);
  - **without** it: 1,006,517 tokens (`vllm_server_fp8_mtp3_s20.log:250`).
  - That is 855 vs 775 blocks: +10.3 % KV.
- VMS never used the flag. The INT4 rows (`vllm_server_int4_mtp3_s20.log:9`, non-default args) have no
  `language_model_only`.

**Failure scenario:**
- Half B (0.32) loads the vision tower and reserves the multimodal encoder budget. Half A (0.19) does not.
- So the two halves are not the same engine configuration, and the per-half A/B mixes "vLLM version" with
  "vision tower loaded".
- With H1's real per-stream need, at R3K2's p50 attempt peak (57k):
  - NB1 half B (INT4, 20 streams) needs 1,020 blocks of about 989: 103 %;
  - the 0.32 FP8 fallback (16 streams) needs 816 of 785: 104 %;
  - 16 × the mean 50k already needs 96 % (see the table in section 3).
- The result is vLLM preemptions in half B. With Mamba `align` mode, a preempted request recomputes most of its
  generated part (E18 smoke finding: align caches Mamba state only at block boundaries).

**Fix:** append `"--language-model-only"` to `e21_flags_v032`, as the E18 R2 `s1` rung did on the GPU.
- The INT4 checkpoint's quantization config already ignores the vision tower (README Finding 5), so dropping the tower
  only removes BF16 weights.
- INT4 together with `--language-model-only` has not run on 0.32. If that combination is a concern, keep the current
  flag set as an extra fallback rung, not as the primary.

### H3 (HIGH, the CPU kit that validates these flags): `e21_flagcheck.py` cannot import on vLLM 0.32

**Location:** `src/e21_flagcheck.py` lines 20-24:
```python
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except Exception:
    from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402
```

**Evidence:**
- The v0.32.0 tree (`_scratch/vllm_src/v0.32.0/full/tree.json`) has no `vllm/entrypoints/openai/cli_args.py`.
- `make_arg_parser` now lives in `vllm/entrypoints/launchers/cli_args.py`:
  - it is defined at line 419;
  - `vllm/entrypoints/cli/serve.py:14-18` imports it from there;
  - fetching `.../v0.32.0/vllm/entrypoints/openai/cli_args.py` returns 404.
- `vllm/entrypoints/openai/api_server.py` in 0.32 is a 59-line deprecation shim that does not re-export
  `make_arg_parser`.

**Failure scenario:**
- On 0.32 the script dies with ModuleNotFoundError before it prints `FLAGCHECK`.
- `REPORT["flagcheck"]["v032"]["summary"]` is None, so `all_ok` is False (cpu_kernel_flags notebook,
  `all(... for x in ("v019", "v032"))`).
- The kit reports NOT OK whatever the flags are, which costs one kit round. This matches the memory rule "batch fixes,
  test locally first".

**Fix:**
```python
try:
    from vllm.entrypoints.launchers.cli_args import make_arg_parser   # >= 0.32
except ImportError:
    from vllm.entrypoints.openai.cli_args import make_arg_parser      # 0.19
```
- Both versions' `make_arg_parser` add the positional `model_tag` (0.19 `openai/cli_args.py:321-326`, 0.32
  `launchers/cli_args.py:427-432`). So the 0.32 commands can be parsed as given, with the path positional. That matches
  `vllm serve` better than the conversion to `--model`.
- Optionally also call `validate_parsed_serve_args(ns)` (0.32 `launchers/cli_args.py:503`). It checks
  auto-tool-choice ⇒ tool-call parser and the chat template.

### M1 (MEDIUM): Duck's serving flags never ran on 0.32, and 0.32's Qwen3 parser behaves differently

**Location:** c7:279-285 (`e21_duck_serving_flags`), used by `e21_flags_v032` at c7:316. The smoke is at c7:393-405.

**Evidence:**
- VMS and E18 used only streaming `/v1/completions` with prompt token ids (`VMS_c06.py:143,164`;
  `E18_R2_c06.py:6,237,298`). No 0.32 run ever used `/v1/chat/completions`, `--tool-call-parser`,
  `--reasoning-parser` or `--default-chat-template-kwargs`.
- 0.32 maps `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` onto one unified parser engine:
  - `vllm/reasoning/__init__.py:155-157` → `qwen3_engine_reasoning_parser` → `Qwen3ParserReasoningAdapter`;
  - `vllm/tool_parsers/__init__.py:193-195` → `qwen3_engine_tool_parser`;
  - both come from `vllm/parser/qwen3.py`.
- Its grammar ends reasoning at a tool call. `vllm/parser/qwen3.py`:
  `(ParserState.REASONING, "TOOL_START"): Transition(ParserState.TOOL_PREAMBLE, (EventType.REASONING_END, EventType.TOOL_CALL_START))`.
  It also silently drops a later `</think>` in CONTENT.
- 0.19's `Qwen3ReasoningParser` splits only on `</think>` (`qwen3_reasoning_parser.py:81-90`:
  `reasoning, _, content = model_output.partition(self.end_token)`). The tool parser then sees only the content.
- So a literal `<tool_call>` written inside the thinking:
  - stays thinking on 0.19 (and on SGLang R3K2);
  - becomes an executed tool call on 0.32, and the rest of the thinking becomes content.
- Things checked and found fine:
  - 0.32 honors `enable_thinking: False` (`Qwen3Parser.extract_reasoning` returns `None, model_output`).
  - It merges the server's default chat-template kwargs (`chat_completion/serving.py:192-201`, `with_defaults`).
  - It accepts `reasoning_content` on input by renaming it (`chat_completion/protocol.py:562-578`).
  - It does not add a grammar for `tool_choice="auto"` with non-strict tools (`structural_tag_registry.py:186-191`).
    Duck sends `"auto"` (`tool_agent.py:2510-2521`) and no `strict` field.
- The engine smoke (c7:393-395) sends `enable_thinking: False` with no tools. It exercises neither reasoning separation
  nor tool parsing.

**Failure scenario:**
- Half B's Duck sees a different reasoning/content/tool split from half A for any reply that mentions `<tool_call>`
  inside its thinking.
- The re-rendered history then differs from the generated tokens, which also costs prefix-cache hits.
- The engine-level smoke passes either way, and the difference shows up only as a hard-to-explain half-B score delta.

**Fix:**
- Add a second smoke request to `e21_launch` (both versions): thinking on, one trivial tool, a prompt that requires
  one call. Assert:
  - `message.reasoning` is non-empty;
  - `tool_calls[0].function.name` is the tool;
  - `content` contains no `<tool_call>` text.
- Record in the half report how many replies had a tool call but no `</think>` before it. That makes the 0.32 parser
  difference measurable.

### Flags verified as correct

- **Each flag exists in its version, with a valid value.** Every flag and value in `src/e21_flags.json`:

  | Flag | vLLM 0.19 | vLLM 0.32 |
  |---|---|---|
  | `--mamba-ssm-cache-dtype float16` | `arg_utils.py:1019`; `config/cache.py:24` `MambaDType = Literal["auto", "float32", "float16"]` | `arg_utils.py:1396`; `config/cache.py:62` (adds "bfloat16") |
  | `--language-model-only` (0.19 only) | `arg_utils.py:1072` | — |
  | `--attention-backend FLASHINFER` / `TRITON_ATTN` | `v1/attention/backends/registry.py:48,62` | `registry.py:48,71` |
  | `--enable-prompt-tokens-details` | `openai/cli_args.py:135` | `launchers/cli_args.py:148` |
  | `--default-chat-template-kwargs` (JSON) | `json.loads` type at `openai/cli_args.py:170-171` | `launchers/cli_args.py:237-238` |
  | `--generation-config vllm` | `config/model.py:267,1408` | `config/model.py:354` |
  | `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` | `openai/cli_args.py:108,114` | `tool_parsers/__init__.py:193` |

  - `--generation-config vllm` means "no generation_config.json". Duck sends temperature, top_p and top_k itself
    (`openai_compat.py:54-60,92-93`), as in E12-V.
  - With `--enable-auto-tool-choice`, the required pairing with a tool-call parser holds (0.32 `cli_args.py:523`).
  - The rest ran on GPU in both versions, with exactly these values: `--tensor-parallel-size 4`,
    `--kv-cache-dtype fp8_e4m3`, `--speculative-config {"method": "mtp", "num_speculative_tokens": 3}`,
    `--max-num-batched-tokens 4096`, `--gpu-memory-utilization 0.9`, `--disable-custom-all-reduce`, `--seed`,
    `--enable-prefix-caching`. Evidence: E12-V log line 7 (non-default args); VMS logs line 9.

- **SSM float16 is honored on both versions.**
  - The checkpoint has `mamba_ssm_dtype: float32`.
  - 0.19 `model_executor/models/config.py:663-689` and 0.32 `:853-877` both log "Using the user-specified value".
  - The real 0.32 log `vllm_server_fp8_mtp3_ssm_fp16.log:19` shows exactly that warning, and the attention block drops
    to 832 (line 141).
- **No duplicates and no conflicts.** No flag is duplicated in any of the 8 commands; the counts are 21-23 distinct
  flags each.
- **The 0.19 commands match E12-V.** The 0.19 set equals the E12-V command that ran (`E12V_SMOKE1_c06.py:207-219`; log
  line 7) as a set. The only differences:
  - `--max-num-seqs` is 20 for NB1;
  - the model path is INT4 for NB1;
  - NB2 adds `--mamba-ssm-cache-dtype float16`.
  - The order of the flags does not matter.
  - The cell 1 constants are equal: KV_CACHE_DTYPE, SPECULATIVE_CONFIG, MAX_NUM_BATCHED_TOKENS 4096, 0.90 and SEED
    20261005 are the same in the E12-V and E15B CELL 1 (E12V_SMOKE1_c02 lines 37-70, E15B_R3K2_c02 lines 131-164).
- **The 0.32 commands match the VMS arm**, plus Duck's serving flags, with 135,168 instead of 106,496. `vms_flags` is
  at `VMS_c06.py:55-69`; the arms `fp8_mtp3_ssm_fp16`, `int4_mtp3_s20` and `ref_fp8_mtp3_s16` are at
  `VMS_c07.py:360,390,392`. The extra `--trust-remote-code` is harmless.
- **135,168 is valid on 0.32** (native context 262k), and `VLLM_ALLOW_LONG_MAX_MODEL_LEN` is set anyway.
- **Served names and ports.** The served names are not Duck aliases, and Duck has no logic keyed on the model name
  (checked in `duck_9e4b`). The ports differ between halves (1234 / 8011), so a half-A straggler cannot reach half B's
  server.

---

## 2. Launch, ready, smoke and failure paths

### M3 (MEDIUM): the GPU-release result is recorded, never acted on

**Location:**
- c7:408-411: `e21_launch`'s except path records `info["gpu_release"] = _wait_gpu_release()`.
- c7:478-481: `e21_begin_half` goes on to the fallback in any case.
- c7:423-424: `e21_relaunch` ignores the result.
- c7:511-512: `e21_end_half` records it, then half B launches anyway.

**Evidence:** the E18 R2 donor stops (`E18_R2_c06.py:503-508`):
```python
if not _att["gpu_release"]["released"]:
    ENGINE_INFO["fatal"] = "gpu_memory_not_released_after_failed_engine"
    break
```

**Failure scenario:**
- A primary dies with its workers stuck in CUDA or NCCL. SIGKILL does not free them at once, and
  `_wait_gpu_release` returns `released: False` after 180 s.
- The fallback, or half B, then starts. vLLM's startup memory check fails, or the run gets a smaller KV pool.
- That is a second wasted launch, up to 30 min of `wait_ready`, with the root cause hidden behind the second error.

**Fix:**
- In `e21_begin_half`: if `_info.get("gpu_release", {}).get("released") is False`, retry `_e18_kill_gpu_pids()` and
  `_wait_gpu_release()` once. If the GPUs are still not free, set `ENGINE_INFO["fatal"]` and break.
- Do the same in `e21_end_half` before half B: record it, and let `e21_begin_half(B)` refuse to launch while
  `nvidia-smi` shows ≥ 1 GiB used.

### M4 (MEDIUM): a failed vLLM 0.32 install is retried for the fallback

**Location:** c7:371-373:
```python
if E21_STATE.get("v032_install") is None:
    E21_STATE["v032_install"] = e18_install()
```

**Evidence:**
- `e18_install` raises on a missing wheelhouse, a non-zero return code, or `subprocess.TimeoutExpired` after
  `timeout=2400` (c7:155-161). It also raises KeyError if `VENV_PYTHON` or `CUDA_HOME` is missing from stdout
  (c7:162-164).
- None of these sets `E21_STATE["v032_install"]`, so half B's fallback config calls the installer again.

**Failure scenario:** the installer hangs, for example on the GCSFUSE slowdown. Half B then spends 40 min, then another
40 min, before giving up. That is 80 min of half B's window. Half B's end is the notebook stop, so the time is simply
lost.

**Fix:** cache the failure, and skip every 0.32 config once the install has failed:
```python
try:
    E21_STATE["v032_install"] = e18_install()
except Exception as e:
    E21_STATE["v032_install"] = {"error": repr(e)}
    raise
```
- Also consider a shorter install timeout. The real install took 175 s (E18 smoke).

### M5 (MEDIUM): `e21_end_half` kills the server only after code that can raise

**Location:** c7:491-514. The kill is at c7:505-512; before it, c7:496-498:
```python
if st.get("cell6_ran"):
    with _E12_LOCK:
        st["records"] = {t: list(v) for t, v in _recs.items() if v}
```

**Evidence:**
- `_E12_LOCK` is first created by CELL 6 at c8:29.
- Half A's CELL 6 can fail before that line, for example in the ORDER2 sort at c8:25, after the runner has already set
  `cell6_ran = True` (cell 9, `_e21_cell6_of`).
- In that case `with _E12_LOCK` raises NameError.
- The runner catches it as an "end" error and goes on to half B. The kill, `_e18_kill_gpu_pids()` and
  `_wait_gpu_release()` never ran.

**Failure scenario:** half A's server keeps about 20 GB on every GPU. Half B's 0.32 server then fails its memory check
or gets a tiny pool, so half B is lost.

**Fix:** wrap the record and summary part in `try: ... finally:` around the kill + `_e18_kill_gpu_pids()` +
`_wait_gpu_release()` block. Also use `globals().get("_E12_LOCK") or threading.Lock()`.

### M6 (MEDIUM): a late watchdog relaunch gets a 60-s ready window

**Location:**
- c7:383-385:
  ```python
  left = E21_HALF_END_PC - time.perf_counter() - 30 * 60
  ok, why = srv.wait_ready(max(60.0, min(E21_LAUNCH_TIMEOUT_SEC, left)))
  ```
- The relaunch at c7:415-431 uses the same code.
- The watchdog relaunches whenever ≥ `_WD_MIN_LEFT` = 20 min are left (c8:486, c8:523).

**Evidence:** real ready times are about 7 min on 0.19 (E12-V log: launch 07:25:47 → "Starting vLLM server" 07:32:38)
and 550 s on 0.32 (E18 smoke).

**Failure scenario:**
- The server dies 20-40 min before the half's end.
- The watchdog kills every GPU process and waits for release. The new server then gets 60 s, times out, and is killed.
- The retry (`B3`) does the same.
- The last 20-40 min of the half run without a server, so the branches burn their failure budget once `SERVER_DOWN`
  clears.

**Fix:**
- For relaunches, use `timeout = min(E21_LAUNCH_TIMEOUT_SEC, max(0, _e15b_solve_end - now - 5 * 60))`.
- Skip the relaunch (and keep the dead-server state) when that is below about 12 min.
- Or make the watchdog's `_WD_MIN_LEFT` at least the measured ready time + 10 min.

### M7 (MEDIUM): half A's window can be shorter than 5.5 h; half B's is not

**Location:** c7:448-450:
```python
_stop = HARD_STOP - E12_STOP_BEFORE_HARD_SEC
E21_HALF_END_PC = _stop if (H != "A" or len(E21_HALVES) == 1) else _stop - E21_HALF_SOLVE_SEC - E21_B_SETUP_RESERVE_SEC
```
The cap is applied in CELL 6 at c8:419-421 and c8:465.

**Evidence:**
- HARD_STOP = T0 + (12 h − 3 min) − 2 min = T0 + 11 h 55 min. `_stop` = T0 + 11 h 52 min.
- So half A ends at T0 + 5 h 57 min, and half A gets the full 5.5 h only if its solve starts by T0 + 27 min.
- Before half A can start:
  - the staging copy deadline is T0 + 45 min (cell 1b);
  - the join waits up to T0 + 55 min (c7:522);
  - then CELL 2's vllm install, the launch (7-10 min, more under GCSFUSE), the INT4-on-0.19 primary, and possibly the
    fallback.
- Half B has no such cap. Its end is `_stop`, and its setup is end-half (≤ 3.5 min) + 0.32 install (175 s) + launch
  (550 s), which fits the 25-min reserve.

**Failure scenario:** when IO is slow or NB1's INT4-on-0.19 primary fails, half A solves for 4.5-5.3 h and half B for
5.5 h. The half-vs-half comparison, which is the experiment's purpose, is then biased toward half B. The report shows
the times, but nothing equalizes them.

**Fix:** make half B's window equal to half A's realized solve window. Store A's
`E12_RUN["finished_sec"] - E12_RUN["window_shift"]["window_start_sec"]` and set half B's `E12_SOLVE_WINDOW_SEC` to that
value. Or, at minimum, print a warning in the E21 report when the two windows differ by more than 5 min.

### M2 (MEDIUM): the smoke cannot see wrong output

**Location:** c7:393-405. `"ok": bool(gen) and prompt_tokens > 0 and completion_tokens > 0`.

**Evidence:**
- NB1's primary on 0.19 (INT4 W4A16 + MTP) and NB2's primary on 0.19 (fp16 SSM) have never run on 0.19 (README
  Findings 3 and 5). The VMS correctness gate covered fp16 SSM only on 0.32.
- E13 SMOKE1's failure mode was a server that came up and produced "!!!!" (memory file `e13-flashnext-smoke1-oom`).
  That output passes `bool(gen)`.

**Failure scenario:** a numerically broken primary passes the smoke. The fallback is then never tried, and the whole
half solves with garbage, or with a silent long-context degradation.

**Fix:**
- Require `"4" in gen` (or "four").
- Add one deterministic thinking-off request with a known short answer, for example "Reply with the digits 1 2 3 4 5 6 7
  8", the S1 engine smoke (`E18_R2_c06.py` S1_ENGINE_SRC `_engine_smoke`).
- After the first real requests, check that the "Mean acceptance length" line is above about 1.5. An INT4 MTP-head
  mismatch shows up there first.
- A long-context check (one ~60k greedy request against a reference) would be the real gate for fp16 SSM on 0.19. That
  takes GPU time, so the user decides.

### L1 (LOW): `e11_kill_server(wait=...)` does not fit `E18Server.kill(fast=...)`

**Location:**
- c7:145-147 is E12-V verbatim: `E11_SERVER.kill(wait=wait)`.
- c7:213 is E18 verbatim: `def kill(self, fast=False)`.
- The callers are c8:1082, the end of CELL 7 (`try: e11_kill_server(wait=False) except Exception: pass`), and c8:302
  in `_e12_parity`, which is dead code.

**Failure scenario:**
- The call raises TypeError, which is swallowed, so the kill at the end of CELL 7 never happens.
- `e21_end_half` kills the server later, so nothing leaks.
- But any future caller that relies on `e11_kill_server` would leave the server running.

**Fix:** `def e11_kill_server(wait=True): E11_SERVER.kill(fast=not wait)`. The function is outside the asserted donor
list, or it can be wrapped.

### L5 (LOW): a watchdog relaunch can overlap the next half

**Location:** c7:415-431 and c7:406-411. The except path runs `_e18_kill_gpu_pids()` and sets `E11_SERVER = None` as a
module global.

**Failure scenario:**
1. Half A's server dies just as half A's solve returns early (all tasks done).
2. The watchdog thread is inside `e21_launch` when CELL 6 sets `_E12_STOP`.
3. Then `e21_end_half(A)` kills the relaunching server.
4. The watchdog's `wait_ready` returns "exited", and its except path runs `_e18_kill_gpu_pids()`.
5. If half B's server has started by then (after the 175-s install), its workers are killed, and
   `E11_SERVER = None` overwrites half B's server handle.

**Fix:** pass a per-half generation id into `e21_launch`. In the except path, skip `_e18_kill_gpu_pids()` and do not
clear `E11_SERVER` when `E21_HALF` (or the generation id) has changed. Optionally, `e21_end_half` can join the
watchdog thread (name `e15b_watchdog`) with a timeout before it kills anything.

### L6 (LOW): `srv.start()` outside the `try`

**Location:** c7:380.

If `Popen` raises (for example a missing `/tmp/vllm-venv/bin/vllm`), the log file handle opened in `start()` leaks and
the exception carries no `info`. `e21_begin_half` handles a missing `e.args[1]`, so this is only untidy.

**Fix:** move `srv.start()` inside the `try`.

### Launch path verified as correct

- **`E18Server.wait_ready` semantics hold on both versions** (c7:192-211). Ready means `/health` answers AND "GPU KV
  cache size" is in this server's log.
  - 0.19 prints the line from EngineCore (`kv_cache_utils.py:1319`, `logger.info_once("GPU KV cache size: %s tokens")`;
    E12-V log line 221).
  - 0.32 prints `"GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: X.XXx"` on one line
    (`kv_cache_utils.py:2325-2333`; VMS logs, e.g. `vllm_server_int4_mtp3_s20.log:217`).
  - A stall is 600 s of log silence. E18's longest startup gap was 60 s, and loading progress bars keep the log
    growing.
  - The function also exits on process death, and on the timeout `max(60, min(30 min, left))`.
- **The URLs are set before they are used.** `E18_URL` is set in `e21_launch` (c7:360) and `SGL_URL` in
  `e21_begin_half` (c7:451), both before any request.
- **Every failure inside the `try` releases the server.** Not ready, `/v1/models` failing, the capacity parse, the
  smoke request raising, and the smoke not ok all go to c7:406-411. That path keeps the last 3,000 log characters, runs
  `kill()` (SIGTERM to the group, a 30-s grace, then SIGKILL to the group always: c7:213-238), sets
  `E11_SERVER = None`, runs `_e18_kill_gpu_pids()` and `_wait_gpu_release()`, and raises
  `RuntimeError(msg, info)` with the info dict.
- **Failures before the `try` leave no server behind**: the version assert, `e18_install` and `Popen` (see L6).
  `e21_begin_half` records them with `_info = {}` (c7:477).
- **A BaseException is still cleaned up.** A KeyboardInterrupt or Kaggle cancel inside `e21_launch` bypasses
  `except Exception`. The runner (cell 9) catches BaseException and still runs the "end" step, which kills the server.
- **`e21_begin_half` resets every per-half global** (c7:440-488):
  - OUT → `half<H>`; E21_HALF; E21_HALF_END_PC; SGL_URL;
  - E15B_SOLVE_END_PC = inf, so half A's value cannot block half B's submission writes;
  - Duck's SERVER_DOWN / WAITS / LAST / SCHED_LAST;
  - `ARC3_REASONING_HISTORY_KEY`;
  - ENGINE_READY, CONCURRENCY, engine, llm;
  - a fresh ENGINE_INFO.
- **The config ladder works.** It tries primary then fallback (`e21_cfgs` merges the fallback over the primary,
  c7:434-437) and stops trying when less than 30 min would be left.
- **CONCURRENCY comes from `e21_capacity`**, and then `E12_STREAMS` and `E12_VLLM_CONCURRENCY` are set (c7:474,
  483-485). CELL 6 uses `min(CONCURRENCY, E12_STREAMS or CONCURRENCY)` (c8:417), and nothing in CELL 6 reassigns
  `E12_STREAMS`. So NB1 really runs 20 client streams.
- **The smoke is the E12-V smoke**, the same request and the same tolerance of a missing `prompt_tokens_details`. The
  served name in the request equals `--served-model-name`.
- **The relaunch mechanics are sound:**
  - `e21_relaunch` relaunches the same half with the config that became ready;
  - it kills the old server and every GPU process, waits for release, and gives each launch its own log name
    (`relaunch<n>`);
  - `E11_SERVER` and `E18_URL` are updated inside `e21_launch`;
  - Duck branches wait on `SERVER_DOWN` meanwhile (c8:526-543).
- **`e21_end_half` keeps everything the report needs** (c7:491-514): records, e12_run, engine info and decode summary.
  It then kills the server, kills GPU processes and waits for release. See M3 and M5 for the gaps.
- **`e18_install`'s env matches E18** (c7:151-168): PYTHONPATH and CUDA_PATH are dropped; CUDA_HOME is the venv's; PATH
  is prefixed; HF offline, `VLLM_CACHE_ROOT=/tmp/vllm-cache` and `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` are set.
  - `LIBRARY_PATH` with the libcuda shim is inherited from `os.environ`, because `ensure_libcuda_linkable()` runs at
    c7:534 before any launch. This is the E18 order (`E18_R2_c06.py:459`).
  - Adding `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` is a no-op on 0.32 (`envs.py:328`, default True).
  - The missing `PYTHONUNBUFFERED`, `USE_TF` and `TRANSFORMERS_NO_*` are harmless: vLLM's logging flushes each record,
    and E18/VMS ran without them.
- **The 0.19 env equals E12-V's** (c7:366-368 vs `E12V_SMOKE1_c06.py:202-204`). The launch module and `cwd=/tmp` are
  the same.
- **The staging join is sound** (c7:517-532). `E14L.Staging` is a daemon thread whose `.result` is (path map, report),
  or None while it is still running. The map's keys are `str(SGL_MODEL_PATH)` / `str(MODEL_PATH)`, as passed by
  CELL 1b. Only staged paths that differ from the source go into `E21_STAGED`, and `e21_model_path` falls back to the
  original path. Fail-open, as E15B.
- **Metric names are the same on both versions**: `vllm:num_requests_running`, `vllm:num_requests_waiting`,
  `vllm:kv_cache_usage_perc` and `vllm:generation_tokens(_total)` (0.19 `v1/metrics/loggers.py:437,447,483,636`; 0.32
  `:528,538,620,799`).

### L7 (LOW): only the primary weights are staged

**Location:** CELL 1 `E21_STAGE_PATHS` (one path); c7:527-529.

NB1's FP8 fallback (both halves) loads from `/kaggle/input`. Under the 10-21+ GCSFUSE slowdown that is the slow path,
and the launch can approach the 30-min `wait_ready` cap. This is acceptable as a fallback; it is noted so that a slow
fallback is not misread as an engine problem.

---

## 3. `e21_capacity` (c7:338-353)

### H1 (HIGH): the effective-KV formula ignores the fixed Mamba blocks per request

**Code:**
```python
eff = int(out["max_concurrency_logged"]["tokens"] * out["max_concurrency_logged"]["x"]) if mc else None
out["needed_p50_peak"] = seqs * 57000
out["streams"] = int(seqs) if eff is None else max(8, min(int(seqs), eff // SGL_KV_PER_REQUEST))
out["fits_p50_peak"] = None if eff is None else eff >= out["needed_p50_peak"]
```

**What vLLM computes:**
- **vLLM 0.19, `kv_cache_utils.py:802-819`.** Max concurrency = `num_blocks / cdiv(sum over groups of
  max_memory_usage_bytes, page)`. This is the number of requests at **max_model_len**.
- **vLLM 0.32, `kv_cache_utils.py:1195-1225`.** The same, as `num_blocks / Σ cdiv(max_memory_usage_bytes, page)`. Its
  "GPU KV cache size" is `int(max_concurrency × max_model_len)` (`:2295-2333`).
- **The Mamba groups' per-request usage is fixed and does not depend on context length:**
  - 0.19 `kv_cache_interface.py:293-300`, align mode: `page_size_bytes * (2 + num_speculative_blocks)`;
  - 0.32 `kv_cache_interface.py:1076-1082`: `2 + num_speculative_blocks + num_prefill_checkpoint_blocks`.
  - With MTP-3 that is 5 blocks per Mamba group. With 3 Mamba groups of 17 layers each ("Add 3 padding layers", E12-V
    log line 195), that is **15 blocks per request**.
  - In the default FP8 config one block is 1,600 tokens (E12-V log line 17), so the fixed part is about 24,000
    token-equivalents per request.

**Checked against every real log.** Real logs give 846 blocks for 0.19 FP8, consistent across max_model_len:

| max_model_len | Max concurrency | Blocks per max-length request | Pool |
|---|---|---|---|
| 81,920 | 12.63× | cdiv(81,920/1,600) + 15 = 67 | 846 |
| 106,496 | 10.32× | 82 | 846 |
| 135,168 | 8.46× (E12-V run 2, line 222) | 100 | 846 |

- The 81,920 and 106,496 values are from the E12-V audit note, line 17.
- The pool/4 line agrees: 846 // 4 × 1,600 = 337,600, as logged (line 221).
- The 0.32 VMS logs fit the same model with `num_prefill_checkpoint_blocks` = 0:
  - 10.19 GiB → 785 blocks → 785 / 82 = 9.57×, as logged;
  - INT4 12.84 GiB → 989 → 12.06×, as logged.

So mc × max_model_len is not a capacity. It is 1.035 M at 81,920 and 1.144 M at 135,168 for the **same** 846-block
pool. The real need of a stream at context L is `cdiv(L, block) + 15` blocks, so N streams need
`N × (cdiv(L, block) + 15)` blocks out of `num_blocks`.

**Real numbers at the 135,168 context.** Blocks needed as a share of the pool, at R3K2's per-attempt mean (50k), p50
peak (57k) and p90 peak (65k). Rows marked "real" come from Kaggle logs; "est." rows are scaled from them.

| Configuration (E21 use) | Pool (block size) | Code's `eff` | Code: fits p50, headroom | Real use at mean / p50 / p90 | Real max streams at 50k / 57k / 65k |
|---|---|---|---|---|---|
| 0.19 FP8, 16 streams (NB1-A fallback, NB2-A fallback), real | 846 (1,600) | 1,143,521 | yes, +25 % | 89 / **96** / 106 % | 18 / 16 / 15 |
| 0.19 FP8 + SSM16, 16 (NB2-A primary), est. | ~1,628 (832) | ~1,236,000 | yes, +36 % | 75 / 83 / 92 % | 21 / 19 / 17 |
| 0.19 INT4, 20 (NB1-A primary), est. | ~1,063 (1,600) | ~1,437,000 | yes, +26 % | 88 / **96** / 105 % | 22 / 20 / 18 |
| 0.32 INT4 without LMO, 20 (NB1-B primary, as built), real | 989 (1,600) | 1,336,811 | yes, +17 % | 95 / **103** / 113 % | 21 / 19 / 17 |
| 0.32 FP8 without LMO, 16 (NB1-B and NB2-B fallback, as built), real | 785 (1,600) | 1,061,068 | yes, +16 % | **96** / **104** / 114 % | 16 / 15 / 14 |
| 0.32 FP8 + SSM16 without LMO, 16 (NB2-B primary), real | 1,509 (832) | 1,145,890 | yes, +26 % | 81 / 89 / 100 % | 19 / 17 / 16 |
| 0.32 INT4 + LMO, 20 (H2 fix), est. | ~1,069 (1,600) | ~1,445,000 | yes, +27 % | 88 / 95 / 105 % | 22 / 20 / 19 |
| 0.32 FP8 + LMO, 16 (H2 fix), real | 855 (1,600) | 1,155,686 | yes, +27 % | 88 / 95 / 105 % | 18 / 16 / 15 |

- 0.32 pools come from the 106,496 logs. The pool depends on memory, not on max_model_len.
- The 0.19 INT4 and SSM16 values are estimates: the 0.19 FP8 KV memory plus the 0.32 weight saving (7.52 − 4.69 GiB),
  or the 0.32 blocks-per-GiB ratio.
- Attention tokens per stream at 16 streams, after the fixed Mamba blocks: about 60.6k (0.19 FP8) and about 54.5k (0.32
  FP8 as built). For comparison, the Experiment 14 results (line 26) give SGLang INT4 about 1.14 M pure attention
  tokens, which is about 71k per stream at 16. Config B was tuned on that.

**Failure scenario:**
- The printed capacity check (a user rule: "pre-run capacity check") reports 16-27 % headroom where the real headroom
  at R3K2's p50 peak is −4 % to +5 %.
- NB1 half B as built (INT4, 20 streams) and both 0.32 FP8 fallbacks need more blocks than the pool at the p50 peak.
  vLLM then preempts (recompute), and with Mamba align mode the recompute re-prefills the generated part.
- The stream decision itself happens to equal `seqs` in every row. But that is because `eff // 50,000` is inflated, not
  because the configuration fits. With a peak-based rule, the 0.32 FP8 fallback would get 15 streams and the 0.32 INT4
  primary 19.
- The `max(8, ...)` floor would also keep 8 streams on any pool too small for them.

**Fix:**
1. Parse the block size. Both versions log "Setting attention block size to N tokens":
   - 0.19 `config.py:281`, E12-V log line 17;
   - 0.32 `interface.py:1096`, VMS logs.
2. Rebuild the pool and the per-stream need:
   ```python
   bs = int(re.findall(r"Setting attention block size to (\d+) tokens", log)[-1])
   fixed = 3 * (2 + SPECULATIVE_CONFIG["num_speculative_tokens"])          # Mamba groups x (2 + spec) in align mode
   per_max = -(-E21_VLLM_MAX_MODEL_LEN // bs) + fixed
   num_blocks = round(mc_x * per_max)
   need = lambda L, n: n * (-(-L // bs) + fixed)
   out.update(block_size=bs, num_blocks=num_blocks,
              util_mean=need(50000, seqs) / num_blocks, util_p50_peak=need(57000, seqs) / num_blocks,
              util_p90_peak=need(65000, seqs) / num_blocks)
   out["streams"] = max(1, min(int(seqs), num_blocks // (-(-57000 // bs) + fixed)))
   ```
   - Optionally derive `fixed` from the logs instead: the 0.19 group count is `num_blocks / (kv_tokens_logged / bs)`.
3. Print `util_*` in the capacity line.
4. Decide with the user whether streams follow the p50 peak (57k) or the mean (50k). The present code sizes on the
   mean. Hold to the 60k thinking-floor rule: this changes streams, not budgets.

---

## 4. Logs and summaries in the engine cell

### L2 (LOW): the stats-line regex drops the busiest intervals

**Location:** c7:117-118, `_E12V_STAT_RE`. It requires
`"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"` as one
contiguous string. It is used by `e11_decode_summary` (c7:122-141) and `_e15b_server_log_summary` (c8:124-152).

**Evidence:**
- vLLM 0.19 `v1/metrics/loggers.py:232-239` inserts `"Preemptions: %d"` between Waiting and the KV usage whenever a
  preemption happened in that 10-s window.
- 0.32 `loggers.py:299-316` inserts `"Deferred: %d reqs"`, `"KV fetch: ..."` and `"Preemptions: %d"`. It also omits
  "Prefix cache hit rate" until the first prefix query (`:321-323`).
- The real 0.32 format is otherwise the same (`vllm_server_fp8_mtp3_ssm_fp16.log:745`).

**Failure scenario:** exactly the intervals with preemptions vanish from `running_hist`, `gen_tok_s_*` and
`kv_usage_pct_max`. `preempt_lines` still counts them, but the KV-max and throughput figures look healthier than they
were. This matters together with H1/H2.

**Fix:** parse each field with its own regex on lines that contain "Avg generation throughput"; make
Preemptions / Deferred optional groups.

### L3 (LOW): `kv_tokens_logged` is not comparable across halves

**Location:** c7:343-346 and the CELL 6 log summary `startup.kv_tokens` (c8:141-147).

On 0.19 the "GPU KV cache size" value is `num_blocks // groups × block` (`kv_cache_utils.py:1302-1306`), for example
337,600. On 0.32 it is `int(mc × max_model_len)` (`:2295-2299`), for example about 1.06 M. The E21 report shows both
under one name.

**Fix:** store `block_size`, `num_blocks` and `effective` (see the H1 fix), and label the raw number with its version.

### L4 (LOW): the harness cannot see H1 or L2

**Location:** `src/harness/fake_vllm_duck_server.py:165-166`
(`Maximum concurrency ... {KV_TOKENS / MAXLEN:.2f}x`, no fixed Mamba part) and `:342-344` (never prints
"Preemptions:").

**Fix:**
- Model the pool as `num_blocks` with `cdiv(ctx, bs) + 15` blocks per running request.
- Emit "Preemptions: N" (and admit or preempt accordingly) when the pool is exceeded.
- Print the 0.32 one-line capacity format when the server runs as 0.32.

### Verified as correct

- The capacity regexes match both real formats: the two 0.19 lines and the combined 0.32 line.
- The stats regex matches the normal 0.19 and 0.32 stats lines.
- `e21_prom` (c7:322-335) is the `_E12V_PROM_RE` parser with no filter. The names CELL 6 reads exist in both versions.

---

## 5. vLLM 0.32 install path (`e18_install`) and the 0.32 environment

All verified, except M4 (the failure is not cached) and H2 (the flag set).

- **The installer and the launch are E18 R2 verbatim.** The wheelhouse's `install_wheelhouse.py` builds
  `/tmp/vllm-venv` with the kernel's Python. The launch is `/tmp/vllm-venv/bin/vllm serve <model>`. E18 timings on
  GPU: install 175 s, ready 550 s.
- **Path and dataset.** `E18_VLLM_WHEELHOUSE` is
  `/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` (CELL 1). The dataset must be attached to both
  GPU kernels. No kernel-metadata exists yet for the two notebooks: check this at push time.
- **When the install runs.** It runs once per notebook, at half B's first launch, after half A's server has been killed
  and the GPUs released. So it never competes with half A for GPU. It does compete for `/tmp` disk with a staged copy
  that landed in `/tmp` because `/dev/shm` was too small; this is only a theoretical concern.
- **Env inheritance.** The 0.32 env inherits the kernel env at half B time. That includes `LIBRARY_PATH` with the shim
  (`ensure_libcuda_linkable` at c7:534), `MAX_JOBS=4`, and the Duck/ARC3 variables, which are harmless.
  - PYTHONPATH and CUDA_PATH are removed, so the kernel's vLLM 0.19 site-packages cannot leak into the venv.
  - The sitecustomize "No module named 'wrapt'" message seen in the VMS 0.32 logs is harmless (the servers ran).
- **The relaunch reuses the cached install** (`E21_STATE["v032_install"]`).

---

## 6. Observations outside the engine cell (only where they touch the engine)

- **The commit's final `submission.json` is half B's.** Both halves write `WORKING/submission.json` through
  `e12_write_submission`. If half B gets no engine, its CELL 6 still writes from its fresh, empty `E12_RECORDS`
  (c8:672). This is harmless for the LB, because the rerun runs half A only. But a commit's `submission.json` does not
  represent half A.
- **Context window.** `E12_DUCK_CONTEXT_WINDOW` is `min(131072, E12_B_WINDOW)` (c8:443-444) and max-model-len is
  135,168. vLLM rejects a request when prompt + max_tokens > max-model-len. Config B's contexts (max 79k) stay far
  below that, as in E12-V.
- **Prefix caching on vLLM** (E18 finding, README Finding 7). Mamba `align` mode caches only at block boundaries
  (1,600 or 832 tokens). With MTP the decode positions rarely land there, so each Duck turn re-prefills the previous
  turn's generated thinking. This is a property of the engine in both halves, not a code defect. It is a likely
  throughput difference from SGLang R3K2 (93 % cache hits) and should be read off the per-half prefix-hit numbers.

## Files

- Built notebook cells (extracted): `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\cells\NB1_c07.py` (engine), `...\NB1_c08.py` (CELL 6/7 functions).
- vLLM sources used: `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\vsrc\` (v019/, v032/, and the parser, logger and protocol files).
- Real 0.32 logs: `D:\ARC AGI 2\_scratch\ems_out\vms\q27x\vms\vllm_server_*.log`.
