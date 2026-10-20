# PROVENANCE — where every part of Q27-XHIGH (v1.2) comes from

The v1.2 changes and their reasons are in `REVIEW_RESPONSE_v1_2.md`.

Legend:
- **REUSED** = copied (or nearly copied) from our Kaggle-proven notebook.
- **ADOPTED** = a decision taken from the Kaggle discussion or russcore's public notebook.
- **VERIFIED** = checked against vLLM 0.19.0 / model source.
- **NEW** = written for this notebook; tested only in the local harness, never on a GPU.

Paths are inside this zip.

## Our notebooks
- `sources/our_notebooks/PIV-Q27-V3_1-EVAL120-ALLFIX.ipynb` ("V3.1"). It ran on Kaggle 4×L4 several times; the S12 run
  showed every mechanism working.
- `sources/our_notebooks/PIV-Q27-SUB-v1_3-HIDDEN-TEST-SUBMISSION.ipynb` ("SUB v1.3"): the hidden-test submission that
  scored LB 1.67.

Cell numbers below are 0-based notebook cell indices; line numbers are within that cell.

| Q27-XHIGH part (cell) | origin | exact source | status |
|---|---|---|---|
| Model path `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` (cell 2 / config) | V3.1 | cell 1 line 398 | REUSED |
| Wheelhouse `/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` | V3.1 | cell 1 line 399 | REUSED |
| Competition path | V3.1 | cell 1 line 400 | REUSED (+ fallback to `/kaggle/input/arc-prize-2026-arc-agi-2`) |
| Env vars (`CUDA_DEVICE_ORDER`, `TOKENIZERS_PARALLELISM`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, HF offline) | V3.1 | cell 1 lines 446–451 | REUSED (CUDA_VISIBLE_DEVICES not forced) |
| `model_type == "qwen3_5"` + FP8 quant check | V3.1 | cell 1 lines 475–478 | REUSED |
| Offline install: `EXPECTED_RUNTIME_PACKAGES` (vllm 0.19.0, transformers 4.57.6, tokenizers 0.22.2, hf-hub 0.36.2), `offline_pip_install`, parity check | V3.1 | cell 1 lines 495–526 | REUSED (the "already imported" guard now runs inside the install branch) |
| Engine settings: TP 4, `distributed_executor_backend="mp"`, `language_model_only=True`, `max_num_batched_tokens=32768`, `gpu_memory_utilization=0.90`, `enforce_eager=False`, `disable_custom_all_reduce=True`, `enable_prefix_caching=False`, `load_format="safetensors"` | V3.1 | cell 7 lines 49–65; cell 1 lines 430–436 | REUSED |
| `max_model_len` 29,696 → **81,920** (v1.2); `max_num_seqs` 16; concurrency = min(16, KV tokens // 40k) | ours, for the ~61k budget; KV read from the front end (agent F3) | — | NEW value |
| Chat rendering `apply_chat_template(enable_thinking=True, reasoning_effort=...)` | V3.1 `render_agent_chat` | cell 4 line 147 | REUSED, but **no tools** advertised and effort = `xhigh` |
| `xhigh` is a valid effort in this model's template (the default, with an xhigh instruction) | model template | `sources/model_.../chat_template.jinja` lines 47–52 | VERIFIED |
| Forced-close text `"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"` | V3.1 (Qwen's documented two-stage thinking budget) | cell 1 lines 221–222 | REUSED |
| `</think>` id via `tokenizer.encode("</think>")` | V3.1 | cell 8 line 279 | REUSED (+ single-token assert) |
| Fresh-process verification of `solution()` on train, then test | V3.1 `fresh_verify_solution` | cell 6 line 866 | RE-IMPLEMENTED (smaller, same semantics: fresh process, exact train equality, then test execution) |
| Fallback `submission.json` written first, stdlib only, atomic `os.replace` | SUB v1.3 bootstrap | cell 1 lines 9–56 | REUSED idea (identity grids as both attempts) |
| Dual mode (commit = evaluation, `KAGGLE_IS_COMPETITION_RERUN` = hidden test) | our rule / EVO24 | — | REUSED pattern |

## Kaggle discussion 745787 (`sources/kaggle_discussion_745787.md`)
| decision | from | status |
|---|---|---|
| `reasoning_effort="xhigh"` (no low/medium, no 3k cap) | Anil Thomas (8th): "you need xhigh for good results"; "if you turn off thinking, the results are abysmal" | ADOPTED |
| Thinking budget **49,152** + answer **12,288** (≈61k per task); a long prompt gets a smaller thinking cap | NguyenThanhNhan (49th): forced commit after 49k, total 65k; Anil Thomas: median ~64k | ADOPTED (v1.2) |
| Forced commit when thinking runs too long | NguyenThanhNhan's "custom intervention logic" | ADOPTED (our token-exact stop text) |
| MTP speculative decoding | NguyenThanhNhan: "27b fp8 + mtp" (they used SGLang EAGLE, 3 steps); the GPT review cites vLLM's Qwen3.8 recipe with 3 tokens (not in our sources) | ADOPTED (vLLM `method:"mtp"`, **3** tokens in v1.2) |
| One pass per task | NguyenThanhNhan: "I could only do 1 pass" | ADOPTED |
| Program + verifier instead of pure reasoning | NguyenThanhNhan: "pure reasoning … token-inefficient … implement a simple harness" | ADOPTED in a light form (code + direct grid in one answer) |
| ~350 tok/s aggregate target | Anil Thomas estimate | target only; the commit run measures it |
| fp8 KV cache (`--kv-cache-dtype fp8_e4m3`) | NguyenThanhNhan's SGLang config | **ADOPTED in v1.2** (`kv_cache_dtype="fp8"`). On sm89 vLLM then uses FlashInfer (FA2 has no fp8 KV): `flash_attn.py` 165–170, `flashinfer.py` 320. Falls back to BF16 if the engine refuses (reported) |

## russcore public notebook (`sources/public_notebooks/arcagi2-qwen38-freeform-direct.ipynb`)
| decision | russcore value | ours | status |
|---|---|---|---|
| temperature / top_p / top_k | 0.6 / 0.95 / 20 | **1.0** / 0.95 / 20 | v1.2 uses the checkpoint's `generation_config.json` (temperature 1.0), not russcore's 0.6 |
| context length | 65,536 | 81,920 | ours larger (fp8 KV) |
| concurrency | 24 | 16 | ours lower (KV safety) |
| max_tokens | 57,344 direct / 40,960 per turn | 49,152 + 12,288 | similar range |
| samples per query | 2 | 1 | ours = 1 pass |

## vLLM 0.19.0 source (`sources/vllm_0_19_0/`)
| fact used | file : lines | status |
|---|---|---|
| `model_type qwen3_5` → MTP draft `Qwen3_5MTP`, `n_predict = mtp_num_hidden_layers` (=1 in our config) | `config/speculative.py` 323–332 | VERIFIED |
| `method` "mtp"; >1 speculative token on one MTP layer gives a warning about acceptance | `config/speculative.py` 367–371, 505–515 | VERIFIED |
| Draft prefix "mtp", so `mtp.fc` / norms match the FP8 ignore list in `config.json` | `models/qwen3_5_mtp.py` (`maybe_prefix(prefix,"mtp")`), `quant_utils.py` `is_layer_skipped` | VERIFIED |
| `mtp.*` weights present with `weight_scale_inv` | `sources/model_.../model.safetensors.index.json` | VERIFIED |
| `LLM.llm_engine.add_request(request_id, prompt, params)` / `step()` / `abort_request(ids)` | `v1/engine/llm_engine.py` 210–300 | VERIFIED |
| Outputs carry the external request id; the internal id gets a random suffix | `v1/engine/input_processor.py` 176–193, `output_processor.py` 345–364 | VERIFIED |
| `LLM.generate` itself is the add/step loop | `entrypoints/llm.py` 1989–2030 | VERIFIED |
| `SamplingParams(detokenize=False, output_kind=RequestOutputKind.DELTA)` fields | `sampling_params.py` 207–252 | VERIFIED |
| `speculative_config`, `language_model_only`, `max_num_seqs` are engine args | `engine/arg_utils.py` 474, 488, 547 | VERIFIED |
| KV block count: set in the engine core (`v1/engine/core.py` 270) **and copied to the front end at start-up** (`v1/engine/utils.py` 1121–1123) | full wheel | VERIFIED (agent F3); my v1.0 note was wrong |
| `LLM()` sets `disable_log_stats=True` unless it is passed | `entrypoints/llm.py` 272–273 | VERIFIED → v1.2 passes `False` |
| fp8 KV dtype strings `fp8` / `fp8_e4m3` are valid | `config/cache.py` 14–19 | VERIFIED |

## Our model (`sources/model_qwen3-8-27b-fp8-repacked-v1/config.json`)
- 64 layers: 48 linear-attention (GDN) + 16 full-attention.
- 4 KV heads, head_dim 256; TP4 gives 1 KV head per GPU.
- MTP 1 layer; FP8 block 128×128.
- Used for the KV estimate: ~16 KB per token per GPU in BF16 → ~650k tokens per GPU at 0.90. **INFERRED**; real
  value unknown until the run.

## NEW code (written for this notebook, tested only in the local harness)
- **Cell 3:** prompt text (grids as digit rows, the two-part answer contract).
- **Cell 4:** answer parsing (last python block with `def solution`; JSON `test_outputs`); attempt ranking; per-query
  scoring.
- **Cell 6:** the whole continuous scheduler: phases, continue vs forced close, admission / drain / hard stop,
  per-task `submission.json` rewrite.
- **Cell 7:** report, 240-task projection, bundle.
- **Harness:** `build/harness_q27x.py` (fake vLLM step engine). Results are in `build/harness_output.txt`: scenarios
  mixed / drain / rerun / MTP-refused all pass.
