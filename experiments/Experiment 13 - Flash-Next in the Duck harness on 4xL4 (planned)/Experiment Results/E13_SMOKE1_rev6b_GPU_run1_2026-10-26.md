# E13 SMOKE1 rev 6b, GPU run 1 (2026-10-26): weight-load OOM on SGLang 0.5.21

- Kernel `foysalemonshanto/q27-xhigh-e13-duck-flashnext-smoke1` v1, notebook sha 74411bc0, 4x L4. Outputs are in `SMOKE1_rev6b_GPU_run1_2026-10-26/`; the top-k `.pt` fixtures were dropped.
- Status COMPLETE, after about 15 min. The engine never became ready. E12 / Duck did not run. `submission.json` is the CELL 0 identity fallback (240 tasks).

## What ran

| Stage | Result |
|---|---|
| Install (SGLang 0.5.21 venv) | OK, 133.7 s |
| FA2 2.8.3.post1 from the foysal dataset | OK, resolves to `flash_attn.flash_attn_interface` |
| Tokenizer check, sandbox self-test | OK |
| fast_topk test | t1 `fast_topk` 2 / 98 failures (both `uniform`); t2 flashinfer ragged 0 / 98; graph replays 4 / 4 for both |
| Host-RAM guard | PLE 102.4 GB; MemAvailable 197.8 GB; cgroup free 155.1 GB; need 122.4 GB; OK |
| Weight load | 38 / 38 shards in 9:00 (`--weight-loader-prefetch-num-threads 1`, no CELL 2 `cat`) |
| Post-load processing | **CUDA OOM** on TP0 in `gptq_marlin_moe_repack` (`gptq_kernels.py:71`, from `gptq_moe.py:329`): tried 200 MiB with 154.56 MiB free; 20.86 GiB allocated by PyTorch plus 652 MiB reserved but unallocated; process total 21.81 GiB of 22.03 GiB. Then the server exited with rc -9. |
| Speed stage, post-speed smoke, Duck | not reached |

The `Error in sitecustomize: No module named 'wrapt'` lines come from the image. E3 printed them too, so they are harmless.

## Comparison with the only successful Flash-Next launch on 4x L4 (E3 FN-PROBE v4.2.1, 2026-10-10)

| | E3 v4.2.1 (ready) | E13 rev 6b (OOM) |
|---|---|---|
| SGLang | 0.5.19+gd00d88efc8d6 (Franzen's / Penny build) | 0.5.21 (+ Franzen patch #3 + FA2) |
| Model | the same 15 Intel non-PLE shards + 10 FP8 PLE overlay shards (124.8 GB) | Franzen's Intel package: the same non-PLE shards + 22 BF16 PLE shards (38 files) |
| Free before load | 21.62 GB | 21.62 GB |
| After load + repack | **19.06 GB used, 2.55 GB free** | **>= 20.86 GiB allocated before the repack finished**, OOM |
| TP / EP / quant / MoE runner / PLE offload | TP4 / EP4 / auto-round / auto / on | the same |

The repack path is the same in both versions, so it does not explain the extra memory:
- `gptq_marlin_moe_repack`, `postprocess_weights`, `_modules_with_quant_method` and `device_loading_context` / `stage_module_for_post_load` match between 0.5.19 and 0.5.21.
- The only `gptq_moe.py` change is the half vs `params_dtype` scale dtype, which patch #3 already restores.
- The PLE offload still builds the table on meta and allocates it on the host.

So 0.5.21 holds about 1.8 GiB more weights or buffers per rank at load time than 0.5.19 did with the same non-PLE weights. The source is **not identified (UNVERIFIED)**. The two remaining differences are the BF16 vs FP8 PLE package and the qwen4_exp model code changes between the versions. `--language-only` is not an option, because in 0.5.21 it requires an EPD encoder URL.

## Options for the next launch (not yet chosen)

1. **Diagnose on 0.5.21.** A short load-only commit (about 15-20 min) with a sitecustomize hook. It logs `torch.cuda.memory_allocated` after `load_weights_only` and around every `process_weights_after_loading`, plus the largest GPU tensors by module. It names the extra ~1.8 GiB before any fix.
2. **Proven stack.** Launch on 0.5.19+gd00d88efc8d6, the E3 / Franzen build that loaded at 19.06 GB on 4x L4. Caveat: the E3 outputs were garbage (VALIDATION #18; the cause is not the Marlin scale bug), and the 2.55 GB left for KV, graphs and activations is tight.
3. **FP8 PLE package on 0.5.21.** Use the overlay `foysalemonshanto/qwen38-flash-next-fp8ple-overlay`, which E3 loaded, to test whether the BF16 PLE package is the cause.
