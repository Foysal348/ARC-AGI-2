# E12-V: Duck REPL agent on our vLLM engine, audit request (2026-10-19)

**For:** the GPT reviewers who audited E12 rev 1–3.

**Ask.** Look only for **blockers or bugs** in the new vLLM engine path that would make a GPU smoke, EVAL30 or a hidden rerun fail, or that would silently break Duck's behaviour (tool calls, preserved thinking, prefix caching). Do not redesign. The user's spec is fixed: the same Duck notebook, and only the engine changes, to our best vLLM engine (S1-60K, LB 34.58). Please cite file and line for every finding, and give **full URLs** for any outside source.

## What is new

E12 rev 3 (SGLang) is already audited and running on GPU. E12-V changes only this:

| Cell | Change |
|---|---|
| CELL 1 | Default `E11_ENGINE=vllm`, so CELL 2 (verbatim) installs `vllm==0.19.0` exactly as S1-60K did. |
| CELL 3 | + `ARC3_REASONING_HISTORY_KEY=reasoning`. |
| CELL 5 | A new vLLM OpenAI-server cell. It is Duck's own Kaggle launcher with the S1-60K engine settings added. |
| CELL 0/2/4/6/7 | Byte-identical to the SGLang notebook. The SGLang notebooks did not change (SMOKE1 sha 0fea4d0c). |

Source of the new cell: `Experiment Notebooks/Build/build_e12_duck.py`, the `if VLLM:` block. Generated notebooks: `Experiment Notebooks/Q27-XHIGH-E12_DUCK_REPL_VLLM_{SMOKE1,EVAL30}.ipynb`.

## The engine command (CELL 5)

**Duck's launcher, verbatim** (`reference/duck_kaggle_vllm_launcher_kaggle.py`, `start_vllm_server`):
`--model --served-model-name --host --port --tensor-parallel-size --enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm --enable-prefix-caching --default-chat-template-kwargs {"preserve_thinking": true} --reasoning-parser qwen3 --max-model-len`.
keithtyser's Duck launch on our FP8 repack uses the same command plus `--kv-cache-dtype fp8`.

**S1-60K engine settings added:**
- `--tensor-parallel-size 4 --kv-cache-dtype fp8_e4m3`;
- `--speculative-config {"method":"mtp","num_speculative_tokens":3}`;
- `--attention-backend FLASHINFER` (TRITON_ATTN if the libcuda probe fails);
- `--max-num-batched-tokens 4096 --max-num-seqs 16 --gpu-memory-utilization 0.9`;
- `--language-model-only --disable-custom-all-reduce --seed`;
- env `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`;
- S49's `ensure_libcuda_linkable`, copied verbatim (dedented) from the E11 fallback cell.

**Added for Duck:**
- `--max-model-len 135168` (Duck's 128 Ki window; S1-60K used 106,496);
- `--enable-prompt-tokens-details` (`cached_tokens` in usage);
- 12 active streams (= the SGLang run).

**No launch ladder:** if the server is not ready (30 min, bounded by the admit deadline) or the smoke fails, the CELL 0 fallback submission stays.

## Facts checked in the vLLM v0.19.0 source

The files are in `reference/vllm_v0.19.0/`, downloaded from https://raw.githubusercontent.com/vllm-project/vllm/v0.19.0/…

1. **`reasoning` key.** Prior assistant reasoning is read only from the message key `reasoning`: `vllm/entrypoints/chat_utils.py` lines 1512 and 1536-1541, which copy it to `reasoning_content` for the template. Duck's milestone value `reasoning_content` (an SGLang setting) would therefore be dropped silently. E12-V uses `reasoning`, which is Duck's own default (`tool_agent.py` `_reasoning_history_keys`).
2. **Truncated thinking.** With `enable_thinking` true and no `</think>` (a request cut at 12,288 tokens), the whole output is reasoning: `vllm/reasoning/qwen3_reasoning_parser.py` `extract_reasoning`.
3. **The Q27-AGENT HTTP 500.** That error was `assert content is not None` at `vllm/entrypoints/openai/engine/serving.py:899`, and it is reached only for a **named** `tool_choice`. Duck sends `"auto"` (`tool_agent.py` `_request_tool_choice`). On the auto path an empty content becomes `""` (lines 930-938).
4. **Flags exist.** Every flag above is defined in `vllm/engine/arg_utils.py` (for example `--attention-backend` at line 806, `--language-model-only` at line 1072) or in `vllm/entrypoints/openai/cli_args.py` (`enable_prompt_tokens_details`, line 135).

## CPU evidence (Windows)

`harness/fake_vllm_duck_server.py` refuses to start unless every Duck and S1-60K flag above has the exact value. It also records which history key carried the prior reasoning.

| Command | Result |
|---|---|
| `python harness/harness_e12_nb.py "Experiment Notebooks/Q27-XHIGH-E12_DUCK_REPL_VLLM_SMOKE1.ipynb" smoke` | **21/21** |
| same, `rerun` | **20/20** |
| same, `launchfail` | **6/6** |
| SGLang smoke, after the harness changes | 18/18 |

The vLLM smoke checks:
- the reasoning goes back under `reasoning` only;
- KV 337,600 and 7.66× are parsed from the log;
- the engine stats (prefix-cache hit rate) are summarised;
- the model id `qwen` is read from `/v1/models`.

The logs are in `harness/logs/nblog_vllm_*.txt`.

## Questions

1. Is there any vLLM 0.19 behaviour on the Duck request path that would still break the following?
   - tool calls (`qwen3_coder` with `tool_choice: auto`, `reasoning` + `tool_calls` in one reply);
   - preserved thinking (the template's `reasoning_content` after `chat_utils` copies it);
   - `cached_tokens` (`--enable-prompt-tokens-details`).
2. Is anything wrong with combining **MTP-3 + prefix caching + FP8 KV + FlashInfer** on the hybrid Qwen3.8 (GDN + attention) in **server mode** at `--max-model-len 135168`? S1-60K ran this combination in offline `LLM()` mode at 106,496.
3. CELL 6/7 use names first defined by the SGLang cell: `SGL_URL`, `ENGINE_READY`, `CONCURRENCY`, `metrics_snapshot`, `E11_SERVER.log_path`, `e11_kill_server`, `e11_decode_summary`, `engine`. Does the vLLM cell define every one of them on both the success path and the failure path?
4. Any path where the hidden rerun ends without a valid `submission.json` that is specific to the vLLM cell?
