# Q27-XHIGH v1.3 checked line by line against Kaggle discussion 745787

- Source of the discussion text: `sources/kaggle_discussion_745787.md`.
- Notebook values were read from the built notebook (`Q27-XHIGH-v1_3.ipynb` (values unchanged from v1.2.1; v1.3 changed deadlines, engine checks, metrics, concurrency sizing 55k), cell 2 = configuration). They were
  confirmed by `build/check_defaults.py`, which runs that cell with the production defaults.
- vLLM facts are from the vLLM 0.19.0 wheel source.

## Matches
| discussion | who | notebook v1.2.1 | evidence |
|---|---|---|---|
| "you need xhigh for good results"; "if you turn off thinking, the results are abysmal" | Anil Thomas (8th) | `REASONING_EFFORT = "xhigh"`, thinking on | cell 2 line 19; template lines 47–52 |
| "27b fp8 + mtp" | NguyenThanhNhan (49th) | same model (Qwen3.8-27B-FP8) + MTP `{"method":"mtp","num_speculative_tokens":3}` | cell 2 line 39; vLLM `speculative.py` 323–332 |
| "at 65k tokens" | NguyenThanhNhan | THINK 49,152 + ANSWER 16,384 = **65,536** | cell 2 lines 20–21 (asserted in `check_defaults.py`) |
| "force the model to stop reasoning and commit with a solution grid after 49k tokens" | NguyenThanhNhan | forced close at 49,152 thinking tokens with Qwen's stop sentence + `</think>` (token-exact), then the answer (code + grid) | cell 2 lines 20, 23; cell 7 `start_phase2` |
| "not all 240 tasks would generate 65k tokens" | NguyenThanhNhan | a task that finishes early frees its slot at once (continuous scheduler, no waves) | cell 7 |
| "I could only do 1 pass" | NguyenThanhNhan | one generation per task, covering all of its test inputs | cell 4 / cell 7 |
| "simple harness" better than pure reasoning | NguyenThanhNhan | light harness: `solution()` checked on the train pairs in a fresh process, verified output first, the model's grid second | cell 5 |
| `--kv-cache-dtype fp8_e4m3` | NguyenThanhNhan | `KV_CACHE_DTYPE = "fp8_e4m3"` | cell 2 line 40; vLLM `config/cache.py` 14–19 |
| `--attention-backend flashinfer` | NguyenThanhNhan | not set by hand: with fp8 KV on sm89 vLLM picks FlashInfer, because FlashAttention-2 has no fp8 KV | vLLM `flash_attn.py` 165–170, `flashinfer.py` 320 |
| `--linear-attn-backend triton` | NguyenThanhNhan | vLLM runs Qwen3.5 GDN with the FLA triton kernels (FlashInfer GDN is SM90+ only) | vLLM `gdn_linear_attn.py` 23–30, 74–82 |
| `--mamba-ssm-dtype float32` | NguyenThanhNhan | float32: vLLM's "auto" takes the checkpoint's `text_config.mamba_ssm_dtype = "float32"` | vLLM `models/config.py` 666–676; model `config.json` |
| EAGLE 3 steps, topk 1, 4 draft tokens | NguyenThanhNhan | MTP 3 speculative tokens (3 drafted + 1 verified per step) | cell 2 line 39 |
| "turn_max_tokens 40960 is too low; set it to 64k and increase the total context length" | Anil Thomas | 65,536 tokens per task; context 81,920 | cell 2 lines 20–24 |
| "set it [MTP] up incorrectly and didn't get any gain" | Russell Kirk | vLLM stats logging on, so the MTP acceptance rate is visible in the commit run | cell 6 `disable_log_stats: False`; vLLM `llm.py` 272–273 |
| "you'd need ~360 tok/s … also fit the KV cache" | Darren | fp8 KV halves KV memory; concurrency = min(16, KV tokens // 40k); the commit run measures tok/s and projects 240 tasks | cell 6; cell 8 |

## Differences (and why)
| discussion | notebook | why / risk |
|---|---|---|
| `--chunked-prefill-size 2048`, `--max-prefill-tokens 4096` (SGLang) | `max_num_batched_tokens = 32768` (vLLM per-step budget, from our Kaggle-proven V3.1) | Different engine. Smaller chunks give smoother decode, but total prefill work is the same. Kept the proven value; the commit run can show whether decode stalls matter |
| `--mamba-radix-cache-strategy extra_buffer`, `--mamba-full-memory-ratio 0.5` (SGLang) | prefix caching off (vLLM) | SGLang-only prefix-cache / memory settings. With one generation per task there is little prefix to share; MTP also rejects vLLM's mamba prefix mode "all" (agent audit) |
| median ~64k at xhigh (Anil); ~85–90k per pass from the benchmark cost (Darren) | forced close at 49k, 65k total | We follow the only submitted 4×L4 recipe (NguyenThanhNhan). Tasks that need > 49k thinking are force-closed, as in his runs |
| 330 queries (Anil's estimate) | 240 tasks / 259 queries in the hidden-test file; one generation per task | Multi-input tasks are answered in one generation, so there are fewer generations than queries |
| "~350 tok/s plausible" (Anil) | not yet measured on our side | The 12-task commit run reports aggregate tok/s, the MTP acceptance and a 240-task end-to-end projection |
