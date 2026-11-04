# E13 NANPROBE rev 2: GPU run 1 (2026-10-30)

**Run**

| | |
|---|---|
| Kernel | `hosen42/q27-xhigh-e13-flashnext-nanprobe` v1 |
| Notebook | sha 4c11a70b (commit 7d59eed) |
| Machine | 4xL4 |
| Status | COMPLETE |
| Wall time | about 46 min |

Output: `_scratch/e13_nanprobe_v1/` (logs, `q27x/e13_loadprobe.json`, per-rank traces in `q27x/e13_memtrace/`).

## Q1: where does `!!!!` come from? (arm `nan_eager`)

**Setup:**
- SGLang 0.5.21, Intel W4A16 AutoRound, TP4 + EP4.
- `--disable-cuda-graph`, mem 0.93, mamba 20.
- Hooks were active for 120 forwards. 1,649 modules were hooked, with 0 hook errors.

**The answer is the same on every rank and every forward (prefill and decode):**
- The NaN first appears in **`model.layers.2.mlp.experts`**, a `FusedMoE` with the `GPTQMarlinMoEMethod`.
  - `inputs_bad = false`: the inputs were finite.
  - The previous clean module was `model.layers.2.mlp.topk`.
  - So the NaN is **made inside the routed-expert MoE of layer 2**. It is not propagated from earlier code.
- `model.layers.0` and `model.layers.1` give finite outputs: prefill [-1.24, 1.50], decode [-0.33, 0.32]. Layers 2-47 are all non-finite. The logits are all NaN, so argmax = token 0 = `!`.
- There was no collapse (no constant outputs) and no value with \|x\| > 1e4.
- The QSA indexer ints look plausible: min -1, max = seq_len - 1. -1 entries are padding (prefill 122 x 2051 → 242,719 entries of -1).
- The count re-run after the hook window is still `!!!!`, so this is not a timing effect.

**What is special about layer 2:**
- The Intel checkpoint quantizes the routed experts the same way in every layer: int4, group 128, sym, `auto_round:auto_gptq`.
- In `extra_config`, the only per-layer difference between layers 1 and 2 is `layers.1.ple.key_proj` / `value_proj` (bf16).
- `config.json` has `ple_layer_ids = [2]`. This is 1-based: vLLM uses `layer_idx + 1 in ple_layer_ids`. So the PLE block sits in layer index 1, and **layer 2's MoE is the first MoE after the PLE layer**.
- The PLE table is offloaded to pinned host memory (`ple_offload_backend=pinned`, 23.84 GiB per rank).

**Candidate causes, in order (UNVERIFIED):**
1. The expert weights or scales of layer 2 are corrupt on the device, for example a buffer reused while the PLE path or the delayed-free repack still writes into it.
2. A PLE side-stream / host-lookup buffer race that clobbers the Marlin workspace of the next MoE.
3. An EP4 expert-map issue that only layer 2 exercises.

**Next check:** scan the weights of the layer-2 experts (`w13/w2` scales, aminmax) at hook install and again after forward 0, compared with layers 0, 1 and 3.

**Other facts:**
- SGLang logs `mamba_ssm_dtype=bfloat16`, but the config says `float32`. This is not the NaN cause, since the MoE inputs are finite, but it should be fixed.
- First eager prefill with hooks: 428 s (includes the TileLang JIT). Later forwards: about 22 s each with hooks.

## Q2: KV capacity (arm `kv_lean`, mem 0.95, mamba 20, CUDA graphs on)

| | mem 0.93 (LOADPROBE) | mem 0.95 |
|---|---|---|
| KV tokens (fp8) | 94,528 | **161,984** (+71 %) |
| KV GB per rank | 0.54 | 0.92 |
| Free after graphs | 1.05 GB | 0.62 GB |

- The capacity gate is **OK**: running is not capped (mamba 20 slots = 4 running).
- Long test: 4 x 20,000-token prompts, 37.6 s, no OOM, server alive. Max running seen was 2 and max full-token usage 0.37.
- The output is still `!!!!`; this arm uses the same weights path.

## Q3: vLLM 0.32 registry (after the arms)

- vLLM 0.32.0 installed in 175 s.
- It registers `Qwen4ExpForConditionalGeneration`, `Qwen4ExpForCausalLM` and `Qwen4ExpMTP`.
- The implementation is `vllm.models.qwen4_exp`. The `impl_error` comes only from my module-path guess; it is not a support issue.
- Source reading (see memory `flashnext-vllm032-support`):
  - INC W4A16 MoE → Marlin;
  - QSA in Triton + FA2;
  - PLE in pinned host memory via `--engram-config`;
  - `qwen4_exp_mtp` / `dflash`.
- The vLLM MoE / PLE / QSA path is a separate implementation, so it does not share the SGLang code that makes the NaN.
