# E15 MTP preflight (local, read-only research), 2026-10-24

Scope: SGLang 0.5.21 + RedHatAI/Qwen3.8-27B-INT4 (compressed-tensors W4A16, Marlin) + built-in MTP (EAGLE 3/1/4) instead of DFlash2, 4x L4 TP4.
Source: `_scratch/session_1c6db7aa/sgl_src/src/sglang` (paths below are relative to it, `srt/...`).
Tags: **[SRC]** read in source, **[LOG]** seen in a real Kaggle server log, **[INF]** inferred, not run.

Memory model script: `mem_model.py` (same folder). It reproduces the measured W8 KV (867k vs 866,744 logged) and W6 KV (916k vs 917,816).

## 0. Most important finding: MTP already ran on our exact checkpoint and stack

E11-03 rev3b config B (2026-10-17, Kaggle 4xL4, SGLang 0.5.21, the same RedHat INT4 Kaggle model):
`experiments/Experiment 11 - .../Experiment Results/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/server_B_int4_mtp_try0.log`

- Flags (builder `Experiment 11/.../Build/build_e11_03_gpu_validation.py:572-575`): `--speculative-algorithm NEXTN --speculative-draft-model-path <INT4 path> --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. No mem fraction, no mamba flags, so SGLang auto-chose: mem 0.6187, extra_buffer (not lazy), SSM float32, flashinfer attention, triton linear attention (log line 8).
- Head loaded [LOG]: line 92-99 `Load weight end. elapsed=4.62 s, type=Qwen3_5ForCausalLMMTP, quant=compressed-tensors, avail mem=15.63 GB, mem usage=1.45 GB` (all 4 TP ranks). 0 lines `skip loading` / `not found in params_dict`.
- Acceptance [LOG]: 875 decode lines, accept len mean 3.05 (min 2.08, median 3.05, max 3.85 of 4), accept rate mean 0.68. `/metrics`: `sglang:spec_accept_length 2.8`, `spec_accept_rate 0.6`, `spec_num_steps 3.0`, `spec_num_draft_tokens 4.0`.
- Server gen throughput medians [LOG]: 1 running 81 tok/s, 4: 217, 8: 341, 10: 374 (accept len ~3.07). DFlash2 W8 fp16 (config D, same run): 1: 108, 8: 483, 12: 477, 13: 489 (accept len ~4.65).
- **Why it capped at 10** [LOG, line 104]: `max_running_requests is capped to 10 by the mamba state cache (max_mamba_cache_size=50, 5 state slots per request)`. Cause: auto mem fraction 0.62 + fp32 SSM (36 MiB/slot/GPU) + extra_buffer with overlap = 3 + 2 = 5 slots per request (`kv_cache_configurator.py:2232-2261`, constants `:179-184`) + K auto-solved from the joint budget (`:2491-2509`). Nothing MTP-specific. Mamba line 109: ssm 1.79 GB, intermediate ssm 1.55 GB (= 11 x 4 x 36 MiB), KV 474,203 tokens.
- So the MTP-vs-DFlash comparison was never fair: MTP fp32 SSM and capped at 10, DFlash D fp16 SSM and 13.

## 1. How 0.5.21 loads the Qwen3.5/3.8 MTP head [SRC]

1. Arch rewrite: when the draft ModelConfig has `is_draft_model=True` and arch is `Qwen3_5ForConditionalGeneration` (ours), it becomes `Qwen3_5ForCausalLMMTP` and `num_nextn_predict_layers = 1` is forced on both hf_config and text_config (`configs/model_config.py:980-1003`). The checkpoint key `mtp_num_hidden_layers` is never read (grep: no hits in `srt`).
2. Model: `models/qwen3_5_mtp.py:94-157`: `fc` = nn.Linear(2*5120 -> 5120) (replicated, not TP), `pre_fc_norm_embedding`, `pre_fc_norm_hidden` (GemmaRMSNorm), one `Qwen3_5ForCausalLM` layer with `num_hidden_layers=1`, `full_attention_interval=1` (= one full-attention layer, no GDN, so no draft mamba state), prefix `mtp`, own `embed_tokens` + `lm_head` (tie_word_embeddings False).
3. Weight names (`qwen3_5_mtp.py:341-503`): keeps only names containing `mtp`; `mtp.` -> `model.`, `model.fc` -> `fc`, `model.pre_fc*` -> `pre_fc*`, drops `.self_attn`; stacks q/k/v -> qkv_proj and gate/up -> gate_up_proj; also loads `model.language_model.embed_tokens.weight` into its own embed. Unmatched names only warn (`:499-501` "Parameter ... not found in params_dict, skip loading") - silent-failure mode, grep it.
4. Files: the loader takes the checkpoint dir; RedHat ships `model.safetensors` (18.6 GB), `model_mtp.safetensors` (849,400,392 B) and `model.safetensors.index.json` (`Experiment 11/.../Build/hf_pins.json:31-45`). E11-03 B proves the index/glob path reaches the mtp tensors (accept 3.05; a random head gives ~1). The draft load iterates all target weights and filters (elapsed 4.62 s from page cache) [LOG/INF: with the GCSFUSE slowdown this second pass may be slower if the page cache was evicted].
5. Embed/head sharing: after load, EAGLE v2 calls `set_embed_and_head(target embed, target head)` (`speculative/eagle_worker_v2.py:349-367`; `qwen3_5_mtp.py:171-181` deletes its own copies). The KV pool is sized before this, so the 1.45 GB draft footprint (MTP layer ~0.29 GB + embed 0.64 + lm_head 0.64 GB/GPU) is charged to the static budget, then about 1.2 GB comes back as idle headroom [LOG: B pool end 8.39 GB -> 9.05 GB before the first graph capture (+0.66), whereas DFlash W6 goes 3.74 -> 3.19 (-0.55) at the same stage; INF: the ~1.2 GB delta = the freed copies].
6. Quantization of the draft: `--speculative-draft-model-quantization unquant` -> None (`arg_groups/serving_hook.py:647-652`) -> the draft ModelConfig re-detects `compressed-tensors` from config.json (`configs/model_config.py:1915-1916`). So `unquant` is a no-op here. `_mtp_quant_config` returns the target's quant config for compressed-tensors (`qwen3_5_mtp.py:50-91`, no special case), and RedHat's ignore list has `re:^mtp.*` (plus `lm_head` and every `linear_attn` parent/norm/in_proj_a/in_proj_b) so every mtp Linear stays BF16 (`UnquantizedLinearMethod`). [LOG: `type=Qwen3_5ForCausalLMMTP, quant=compressed-tensors`; config.json harness copy `_scratch/session_1c6db7aa/e11_03_harness/run_main/input/.../config.json` = HF config sha c16a4740, 20,952 B.]

### Does EAGLE/NEXTN with NO draft path use the target's MTP weights?
- NEXTN is an alias of EAGLE (`arg_groups/speculative_hook.py:108-115`).
- Auto-default of the draft path to `--model-path` exists only for an arch list (`speculative_hook.py:1007-1035`). **Qwen3_5ForConditionalGeneration is not in it.** [SRC]
- The draft worker would still load from the target dir: `tp_worker.py:455-471` passes `speculative_draft_model_path` (None) and `ModelConfig.from_server_args` uses `model_path or cfg.model_path` (`configs/model_config.py:796`). [SRC; never run by us]
- **But a no-path launch sizes the KV pool differently** [SRC]: the target's aux config takes `eagle_draft_num_layers` from the draft ModelConfig only when a draft path is set (`model_executor/model_runner_components/spec_aux_hidden_state.py:88-105`); without it, it reads the target config, whose `num_nextn_predict_layers` is None (`model_config.py:1414-1416`; our config only has `mtp_num_hidden_layers`). Then the target KV cell size ignores the draft layer (`model_executor/pool_configurator.py:218-229, 643-650`) and the draft KV (512 B/token/GPU, ~0.57 GiB at 1.2M tokens) is allocated on top of the budget [INF: less headroom or OOM].
- **Recommendation: pass `--speculative-draft-model-path <INT4 path>` explicitly**, as E11-03 B did (verified working). The cookbook's no-path form is not verified on this arch.

### Blocking assertions? (all checked)
| Combination | Verdict | Evidence |
|---|---|---|
| compressed-tensors target | allowed, ran | `qwen3_5_mtp.py:50-91` no block; [LOG] B |
| TP 4 | allowed, ran | [LOG] B, no TP check for EAGLE in `speculative_hook.py:961-1170` |
| GDN hybrid + extra_buffer | allowed, ran | [LOG] B (`mamba_radix_cache_strategy: extra_buffer`) |
| extra_buffer_lazy | allowed [SRC], not run with MTP | `arg_groups/mamba_hook.py:113-121` ("eagle/ngram/dspark/dflash all verify through prepare_mamba_track_for_verify"); lazy needs overlap on (`kv_cache_configurator.py:2249-2251`); EAGLE keeps overlap on [LOG] |
| mamba_track_interval | must be >= draft tokens (4): 256 or 64 OK | `mamba_hook.py:122-125` |
| `--mamba-ssm-dtype float16` | no check on sm89 [SRC], not run with MTP | only SM100 flashinfer checks (`arg_groups/attention_hook.py:262-290`) |
| CUDA graphs | allowed, ran | [LOG] B: target verify (4 tok/req), draft decode (1), draft extend (4), same bs list |
| page_size 64 | allowed with topk 1 | `speculative_hook.py:1156-1170` only rejects topk > 1 |
| num_draft_tokens != steps+1 | auto-corrected to steps+1 | `speculative_hook.py:1143-1154` |
| `--speculative-adaptive` | avoid | known crash with MTP on hybrid GDN (#23330, Speed review A2 line 369) |
| max_running_requests unset | reset to 48 | `speculative_hook.py:973-981` (we pass 13) |

### ReplaySSM (`--enable-linear-replayssm-spec`) with EAGLE in 0.5.21 [SRC]
- Field: `arg_groups/fields/exec_.py:458-470` (default False; ring length `--linear-replayssm-cache-len` default 16).
- Requirements (`arg_groups/attention_hook.py:361-435`): topk in {None, 1} (ours 1); linear-attn decode backend triton or flashinfer (ours triton [LOG]); `SGLANG_RAGGED_VERIFY_MODE=static` (default, `environ.py:1383`); not a PD prefill server; not together with `--enable-linear-replayssm` (the decode variant, which also requires no_buffer - that restriction is **not** on the spec variant).
- SSM dtype: unset -> forced float32; float16 -> warning "validate long-sequence accuracy" (`attention_hook.py:417-434`).
- The DFLASH/DSPARK block (`kv_cache_configurator.py:1124-1135`: "requires a KDA model") does **not** apply to EAGLE. So **ReplaySSM is allowed with MTP** on our GDN model.
- Effect: no intermediate SSM buffer (`mem_cache/memory_pool.py:758-788`), configurator stops reserving it (`kv_cache_configurator.py:2429-2456, 2468-2482`), adds a GDN ring per active request: (max_running + 1) slots (`kv_cache_configurator.py:1140`, `memory_pool.py:655-667`), ~6.0 MiB per slot per GPU at L=16 (d, k, g + bf16 low parts, `memory_pool.py:664-755`). GDN uses the non-fold path (`replayssm_spec_fold` is KDA-only, `memory_pool.py:569-571`; commit `speculative/spec_utils.py:985-1035`).
- Risk: never run by us; "compact replay not bit-exact" (#42209, Speed review A2 line 478). Needs the correctness gate.

## 2. Does our RedHat INT4 checkpoint contain the MTP head? Yes
- `model_mtp.safetensors` 849,400,392 B, sha256 1d8268aa... (`hf_pins.json:41-45`), revision 91bd022d.
- config.json: `text_config.mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`, `mamba_ssm_dtype: float32`; quant ignore includes `re:^mtp.*` and `lm_head`; `kv_cache_scheme` static fp8 tensor.
- E10 analysis (`Experiment 10/.../Analysis/W4A16 checkpoints - MTP and vLLM 0.19 compatibility (2026-10-15).md:14`): BF16 MTP, 15 keys, ignored via `re:^mtp.*`.
- Loaded and accepted on GPU: section 0.

## 3. Memory per GPU (TP4), mem-fraction 0.83, `extra_buffer_lazy` (4 slots/request)

Formulas [SRC]: persistent `(K+1) x (ssm + conv)`; intermediate SSM `(min(mrr, K//4) + 1) x D x ssm` (`kv_cache_configurator.py:2462-2482`, allocation `memory_pool.py:777-788`), D = `speculative_num_draft_tokens` (MTP 4, DFlash width). ssm/slot = 48 GDN layers x 12 heads x 128 x 128 x bytes = 18.0 MiB fp16 / 36.0 MiB fp32; conv/slot 0.70 MiB. KV/token/GPU: target 8,192 B (16 attn layers x 1 kv head x 256 x fp8 x K+V); DFlash2 draft +2,560 B (5 layers) [LOG: 1.09 GiB K for 917,816 tokens]; MTP draft +512 B (1 layer) [LOG: 0.11 GiB K for 474,203]. Pool budget (KV + mamba) = 11.72 GiB measured at mem 0.83 (W8 8.68 + 3.04; W6 9.18 + 2.54), +0.04 for MTP (1.45 vs 1.49 GB weights). The "D" is per running request, not per persistent slot. Mamba slot count per request does not change with MTP (4 with lazy).

| running (mrr, K) | arm | mamba total GiB | intermediate | KV tokens | per stream |
|---|---|---|---|---|---|
| 12 (13, 56) | DFlash W8 (now) | 3.04 | 1.97 | 867k (logged 866,744) | 72k |
| 12 (13, 56) | DFlash W6 (E14S) | 2.54 | 1.48 | 916k (logged 917,816) | 76k |
| 12 (13, 56) | **MTP 3/1/4 fp16** | 2.04 | 0.98 | **~1,198k** | 100k |
| 12 (13, 56) | MTP + ReplaySSM fp16 | 1.14 | 0 (+0.08 ring) | ~1,310k | 109k |
| 12 (13, 56) | MTP + ReplaySSM fp32 | 2.14 | 0 (+0.08 ring) | ~1,186k | 99k |
| 12 (13, 56) | MTP 2/1/3 fp16 | 1.80 | 0.74 | ~1,229k | 102k |
| 14 (15, 60) | DFlash W8 | 3.40 | 2.25 | 831k | 59k |
| 14 (15, 60) | DFlash W6 | 2.83 | 1.69 | 888k | 63k |
| 14 (15, 60) | MTP fp16 | 2.26 | 1.12 | ~1,172k | 84k |
| 14 (15, 60) | MTP + ReplaySSM fp16 | 1.23 | 0 | ~1,299k | 93k |
| 14 (15, 60) | MTP + ReplaySSM fp32 | 2.30 | 0 | ~1,167k | 83k |
| 16 (17, 68) | DFlash W8 | 3.83 | 2.53 | 788k | 49k |
| 16 (17, 68) | DFlash W6 | 3.19 | 1.90 | 852k | 53k |
| 16 (17, 68) | MTP fp16 | 2.55 | 1.27 | ~1,136k | 71k |
| 16 (17, 68) | MTP + ReplaySSM fp16 | 1.39 | 0 | ~1,279k | 80k |
| 16 (17, 68) | MTP + ReplaySSM fp32 | 2.60 | 0 | ~1,130k | 71k |

[INF] MTP rows are model predictions from the measured DFlash pools; the DFlash rows reproduce the logs. Extra after-startup headroom with MTP: ~+1.2 GB (freed embed/head, section 1.5), so post-capture free should be ~3 GB instead of 1.6-1.8 GB at 0.83. Each +0.01 mem fraction = ~0.225 GiB = ~28k MTP tokens; mem 0.86 would add ~83k tokens and still leave ~2.3 GB [INF].
MTP + ReplaySSM fp32 costs about the same memory as plain MTP fp16 and returns to fp32 state precision.

## 4. Recommended MTP arm launch (E15-M1), built from `e11_flags` (+ E14S arm)

Remove from the DFlash command: `--speculative-algorithm DFLASH`, `--speculative-draft-model-path <DFLASH>`, `--speculative-draft-model-quantization unquant` (no-op for MTP; may stay), `--speculative-num-draft-tokens 8` and the E14_SUBMIT_ARM override `--speculative-num-draft-tokens 6` (else it is auto-corrected to 4 with a warning). The DFlash2 model input is no longer needed (3.85 GB less to read).

```
python -m sglang.launch_server
  --model-path <SGL_MODEL_PATH>  --tp-size 4  --kv-cache-dtype fp8_e4m3
  --context-length 155648  --max-running-requests 13  --chunked-prefill-size 4096
  --mamba-ssm-dtype float16  --enable-metrics  --disable-flashinfer-autotune
  --random-seed <SEED>  --host 127.0.0.1  --port <PORT>  --log-level info
  --watchdog-timeout 1800  --decode-log-interval 40
  --speculative-algorithm EAGLE
  --speculative-draft-model-path <SGL_MODEL_PATH>        # same dir as --model-path; see section 1
  --speculative-num-steps 3  --speculative-eagle-topk 1  --speculative-num-draft-tokens 4
  --disable-prefill-cuda-graph
  --mamba-radix-cache-strategy extra_buffer_lazy  --max-mamba-cache-size 56
  --mem-fraction-static 0.83
  --incremental-streaming-output  --stream-interval 1
  --reasoning-parser qwen3  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs '{"preserve_thinking": true}'
  --schedule-policy lpm  --enable-cache-report
  --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13
  [--mamba-track-interval 64 --page-size 64   only if E12_MAMBA_TRACK_64]
```
Measured arms on top (one launch each):
- M1-mem: `--mem-fraction-static 0.86` (uses the freed embed/head memory; gate SGL_MIN_FREE_GB 1.0 must still pass).
- M1b: `--speculative-num-steps 2 --speculative-num-draft-tokens 3`.
- M2: `--enable-linear-replayssm-spec` with `--mamba-ssm-dtype float32` (the source's default for it) and, separately, float16 (warning path). Same 56 cache.
- M3: 14 / 16 real: `--max-running-requests 15 / 17`, `--max-mamba-cache-size 60 / 68`, `--cuda-graph-bs-decode 1..15 / 1..17`, client 14 / 16.
- Do not use `--speculative-adaptive` (#23330). `--speculative-token-map` (FR-Spec, Franzen) is an option for later: it cuts the 3 full-vocab lm_head passes per step.

Notebook items: `e11_slots` (lazy 4) and `e11_caps` still parse MTP logs (the first TP0 KV line is the target, `Capture target verify CUDA graph begin` exists for EAGLE [LOG B line 150]). `e11_next` "lazy" fallback goes to extra_buffer + K 70 (5 slots) - fine.

## 5. Log lines to grep (verification)

1. `server_args=` contains `'speculative_algorithm': 'EAGLE'`, `'speculative_num_steps': 3`, `'speculative_eagle_topk': 1`, `'speculative_num_draft_tokens': 4`, `'speculative_draft_model_path': '<INT4 dir>'`, `'mamba_radix_cache_strategy': 'extra_buffer_lazy'`, `'mamba_ssm_dtype': 'float16'`, `'enable_linear_replayssm_spec': False|True`.
2. Head loaded, 4 lines (one per TP): `Load weight end. ... type=Qwen3_5ForCausalLMMTP, quant=compressed-tensors, ... mem usage=1.45 GB`.
3. Must be ABSENT: `not found in params_dict, skip loading`; `max_running_requests is capped to` (or capped >= 13).
4. `Mamba Cache is allocated. max_mamba_cache_size: 56, conv_state size: 0.04GB, ssm_state size: 1.00GB intermediate_ssm_state_cache size: 0.98GB intermediate_conv_window_cache size: 0.02GB` (predicted for M1). ReplaySSM: `intermediate_ssm_state_cache size: 0.00GB` (`memory_pool.py:872-886`), ssm 2.00GB if fp32. With fp16 + ReplaySSM also the warning `--enable-linear-replayssm-spec with --mamba-ssm-dtype=float16`.
5. Two `KV Cache is allocated` lines per TP with the same `#tokens` (~1.2M predicted): target K ~4.6 GB, draft K ~0.29 GB (1 layer).
6. `Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=4, bs=[1, ..., 13]`, `Capture draft decode CUDA graph begin. backend=full, num_tokens_per_req=1`, `Capture draft extend CUDA graph begin. backend=full, num_tokens_per_req=4`.
7. `max_total_num_tokens=..., max_running_requests=13, ... available_gpu_mem=X GB` (expect ~3 GB at 0.83).
8. Acceptance: `Decode batch, #running-req: N, ... mamba num: M, ... accept len: X, accept rate: Y, cuda graph: True, gen throughput (token/s): Z` (`managers/scheduler_components/metrics_reporter.py:961`). Expect accept len ~3.0-3.1 (B: mean 3.05, rate 0.68). Red flag: accept len near 1.0-1.3 = head not loaded.
9. `/metrics`: `sglang:spec_accept_length`, `sglang:spec_accept_rate`, `sglang:spec_num_steps 3.0`, `sglang:spec_num_draft_tokens 4.0`, `sglang:spec_verify_calls_total`.
10. Real concurrency: decode lines with `#running-req: 12` (and 13 with the +1 slot) and `mamba num` <= 56.
