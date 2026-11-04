# E13 NANPROBE: audit brief (2026-10-30)

## Goal
Qwen3.8-Flash-Next with Franzen's Duck harness is what the leading ARC-AGI-2 teams use (Kaggle topic 750546). On our 4x L4
(SGLang 0.5.21, TP4 + EP4) the E13 LOADPROBE run on 2026-10-27 found two blockers:
1. **Load OOM** in the post-load Marlin MoE repack, on 1 rank. The old buffers are freed late (a reference cycle that only
   the cyclic GC frees). Allocated memory grew by 0.30 GiB per MoE layer up to 20.86 GiB.
2. **Garbage output** once loaded. Greedy "1, 2, ... 8," continues as `!` x 32, which is typically NaN or Inf logits.
   - E3 had the same symptom with bf16 KV, Triton attention and no CUDA graphs.
   - So the cause is not the KV dtype, not FA2 and not graphs.
3. **Small KV after load:** only 94,528 fp8 tokens with mem 0.93, 4 running and mamba cache 20 slots.

This notebook measures blockers 2 and 3 and applies the candidate fix for blocker 1, in one GPU session of about 50 min
(budget ceiling 75).

## What is new (vs the Kaggle-run LOADPROBE notebook 107aa182)
- `e13_scripts/mp_nanprobe.py` is appended to the same sitecustomize as `mp_memtrace.py`. It never edits a SGLang file and is inert without `E13_MEMTRACE_DIR`.
  - **gc fix:** before and after each module's `process_weights_after_loading`, if allocated memory grew more than 256 MiB since the last check, it runs `gc.collect()` and logs the freed bytes.
  - **NaN locator** (`E13_NANPROBE=1` only):
    - After `postprocess_weights`, a forward hook goes on every submodule, plus a pre/post hook on the top model.
    - For the first 120 model forwards, each rank logs the first modules in execution order whose floating output is non-finite or has |x| > 1e4. It also logs every decoder-layer output absmax.
    - The hooks are removed afterwards.
- **Arm 1 `nan_eager`:** BF16 PLE (Franzen's Intel package, the rev 6b flags), `--disable-cuda-graph`, NaN hooks; then the smoke (count + 2+2).
- **Arm 2 `kv_lean`:** the same model, graphs on, no hooks, `--max-mamba-cache-size 8` (4 running x 2; it was 20) and `--mem-fraction-static 0.95` (it was 0.93); KV tokens and the smoke.
  - FP8 PLE is not an arm. LOADPROBE showed that it changes host memory only: the GPU tensors are byte-identical.
- **vLLM 0.32 registry check:**
  - It installs our wheelhouse venv, the same as the E23 install (`foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` is added to the metadata).
  - It reports whether `Qwen4ExpForConditionalGeneration` is in `ModelRegistry.get_supported_archs()`, plus all supported Qwen archs.
  - It is an import check only: no server and no GPU work.
- **Summary:** `e13_loadprobe.json` per arm has KV tokens, caps, the smoke, and per rank `memtrace` with `gc` and `nan` (the first bad forward and its first bad modules, and the first-forward layer absmax).

## Validation
- **Local hook test:** a fake loader and a 4-layer model with NaN injected in `layers.2`. The hook names `layers.2` as the first bad module, records the absmax of `layers.0` and `layers.1`, and removes the hooks after 2 forwards.
- **E13 notebook harness, new `nanprobe` scenario:** the real notebook cells against the fake SGLang server. **11/11 OK** (`results/harness_nanprobe.txt`):
  - the arms are in order;
  - the flags are checked per arm;
  - both arms reach ready, the smoke passes and the KV is parsed;
  - the gc and NaN summaries are correct;
  - the vLLM check is recorded;
  - the hook is written.
- **Unchanged:** the LOADPROBE notebook that ran on Kaggle (107aa182).

## What to audit
1. Can anything in the hook break or hang a real SGLang 0.5.21 TP4 + EP4 launch? Consider:
   - forward hooks on every module with eager decode;
   - `.item()` syncs for 120 forwards;
   - the warm-up forwards that count towards the 120;
   - the `gc.collect()` timing.
2. Will the NaN locator really name the broken component? Consider:
   - Is the top-level `model` that `postprocess_weights` receives the one whose `forward` runs?
   - Are some modules called functionally (custom ops, TileLang kernels) and so invisible to hooks?
3. Is `kv_lean` a valid KV lever? Consider:
   - Are mamba slots = running x 2 enough for the extra_buffer strategy?
   - Is mem 0.95 safe on L4?
4. The vLLM check: is `get_supported_archs()` the right API in 0.32, and is the install (2-3 min, CPU) placed well?
5. Anything that wastes the GPU session.

## Files
- `notebooks/q27-xhigh-e13-flashnext-nanprobe.ipynb` (sha c48adfa5) + `kernel-metadata.json` (foysalemonshanto, not pushed).
- `notebooks/build/build_e13_duck.py` (`--nanprobe`), `e13_scripts/mp_memtrace.py`, `e13_scripts/mp_nanprobe.py`.
- `results/`:
  - the harness output;
  - the local hook test;
  - the LOADPROBE 2026-10-27 results;
  - topic 750546 (JSON).
