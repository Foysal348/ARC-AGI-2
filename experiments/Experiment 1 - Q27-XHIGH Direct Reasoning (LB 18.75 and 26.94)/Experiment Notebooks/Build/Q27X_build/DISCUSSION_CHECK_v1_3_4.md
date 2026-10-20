# Q27-XHIGH v1.3.4 checked against Kaggle discussion 745787 (values read from the built notebook, sha 713b84a1…)

## Same as the discussion
| discussion (who) | v1.3.4 | where |
|---|---|---|
| xhigh effort (Anil Thomas) | `REASONING_EFFORT = "xhigh"` | cell 1 L19 |
| Qwen3.8-27B FP8 + MTP (NguyenThanhNhan) | same model; MTP 3 speculative tokens (vLLM's closest counterpart to his EAGLE 3 steps, not an exact equivalent) | cell 1 L39 |
| 65k tokens, forced commit after 49k (NguyenThanhNhan) | 49,152 thinking + 16,384 answer = 65,536 | cell 1 L20–21 |
| one pass (NguyenThanhNhan) | one generation per task | cells 6–7 |
| "simple harness" (NguyenThanhNhan) | `solution()` checked on the train pairs, the model's grid as second attempt | cell 5 |
| `--kv-cache-dtype fp8_e4m3` | `KV_CACHE_DTYPE = "fp8_e4m3"` | cell 1 L40 |
| `--attention-backend flashinfer` | **first engine attempt: FlashInfer, explicit** (`attention_config={"backend": "FLASHINFER"}`), after the libcuda link probe passes. v1.2–v1.3.3 relied on vLLM choosing it implicitly | cell 6 ladder |
| `--linear-attn-backend triton` | vLLM's GDN uses FLA triton kernels | vLLM `gdn_linear_attn.py` 23–30 |
| `--mamba-ssm-dtype float32` | float32 from the checkpoint's `mamba_ssm_dtype` (vLLM "auto") | vLLM `models/config.py` 666–676 |
| "turn_max_tokens 40960 too low; 64k + larger context" (Anil Thomas) | 65,536 per task; context 81,920 | cell 1 |
| MTP misconfigured = no gain (Russell Kirk) | vLLM stats on; MTP drafted / accepted / acceptance rate saved in the report | cells 6–8 |

## Differences
| item | discussion | v1.3.4 | why |
|---|---|---|---|
| chunked prefill / max prefill | 2048 / 4096 (SGLang) | `max_num_batched_tokens 32768` (vLLM, our V3.1-proven value) | different engine; same total work |
| mamba radix cache / full-memory ratio | SGLang-only settings | prefix caching off | no vLLM equivalent needed for one pass |
| concurrency | not stated (Darren asked; no answer) | min(16, KV tokens // 30k); run 1 measured 312,000 fp8 KV tokens → 10 slots | our own tuning, not from the discussion; preemptions are reported |
| fallbacks | none described | fp8 + Triton, then bf16 + FlashAttention-2, then bf16 without MTP | used **only if** the discussion configuration fails to start or decode; every attempt is in the report |
| libcuda shim + link probe | not mentioned | before vLLM starts | Kaggle-specific fix for run 1's `ld: cannot find -lcuda`; it is what lets the discussion's FlashInfer configuration run |
