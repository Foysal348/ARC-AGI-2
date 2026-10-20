# Prompt for the reviewer — paste this as your first message, with the zip attached

You are reviewing **FN-SMOKE-FLASHNEXT-4XL4**, a small Kaggle notebook that has **not run yet**. Its only job is to find
out whether the model **Qwen3.8-Flash-Next (Intel W4A16 AutoRound)** can be **loaded and served** on the Kaggle
**ARC Prize 2026 – ARC-AGI-2** hardware (**4× NVIDIA L4**, 24 GB each, Ada / sm89, offline, no internet). GPU quota is
scarce (4×L4 burns 2 quota-hours per wall-hour), so tell us **before** the run whether it can work and what would make it
fail or waste time. You may use web search (GitHub, Hugging Face, SGLang docs) to check facts. Separate **confirmed
defects** (cell/line + concrete failure) from risks and suggestions.

## Where the model and engine come from

* dfranzen's public ARC-AGI-3 Milestone-2 notebook (`reference/franzen_launcher_cell12.py` is its server launcher cell,
  copied verbatim). It runs on **one RTX PRO 6000** (Blackwell, sm120, 96 GB), Python 3.12 wheels, **CUDA 13** toolkit
  installed from the wheelhouse, engine **Pennyroyal v2.5.3** = an SGLang fork tuned for sm120
  (github.com/jpezzulli/sglang-rtxpro6000, commit d00d88e; patches from mratsim / gabrielolympie "sm120").
* Inputs we attach (public Kaggle sources): dataset `dfranzen/pennyroyal-v253` (wheelhouse incl. `sglang-*.whl`, `uv`,
  `flashinfer_jit_cache 0.6.17+cu130`, `cuda_toolkit 13.0.3`), model
  `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1` (181 GB on Kaggle). We do **not** attach
  the MTP drafter (speculative decoding is off in this probe).

## Model facts (from `reference/config.json`)

`model_type: qwen4_exp`, 48 layers (36 linear-attention / GDN + 12 full-attention, `full_attention_interval 4`),
hidden 2560, 24 heads / 2 KV heads (head_dim 256), **512 experts, 10 active, moe_intermediate 640**, shared expert 640,
vocab 248,320. Quantization: AutoRound INT4, **group_size 128**, sym, `auto_round:auto_gptq` packing; embeddings,
hyper-connections, some projections kept 16-bit. **PLE** (per-layer n-gram embedding) ≈ **102 GB BF16**, which
dfranzen offloads to host RAM with `--ple-offload-embedding`.

## What the notebook does (`notebook/FN-SMOKE-FLASHNEXT-4XL4.ipynb`, built by `build/make_fn_smoke.py`)

`LOAD_ONLY = True`, hard stop 45 min. Every stage writes `fn_smoke_report.json`; the first failing stage stops the run;
the report and a bundle are always written. A valid fallback `submission.json` is written first (in the competition
rerun the notebook only writes that fallback — it is a probe, not a solver).

0. environment — `nvidia-smi` driver + max CUDA version, RAM, disk. **Stops** if the driver's max CUDA < 13 or host RAM
   < 140 GB (PLE offload).
1. inputs present; model config / quant summary.
2. offline install exactly like dfranzen: `uv venv --python <system 3.12>`, `uv pip install -r requirements.lock`, then
   the sglang wheel `--no-deps`.
3. CUDA 13 toolkit from the venv (`nvidia/cu13`), `libcuda.so` symlink, `TORCH_CUDA_ARCH_LIST=8.9` (dfranzen: 12.0),
   SM120-specific env flags set to 0; probe `torch.cuda` + a matmul inside the venv.
4. launch `sglang serve` with `--quantization auto-round --kv-cache-dtype fp8_e4m3 --ple-offload-embedding
   --context-length 40960 --max-running-requests 8 --disable-cuda-graph` and, in order until one is healthy
   (18 min each):
   a. `--tensor-parallel-size 4 --ep-size 4 --enable-dp-attention --dp-size 4` + flashinfer linear-attention backends
   b. same with triton linear-attention backends
   c. `--tensor-parallel-size 4 --ep-size 4` + triton
   d. `--tensor-parallel-size 4` + triton
   Reason for EP: with TP=4 each expert's 640 columns become 160, not a multiple of the INT4 group size 128.
5. one 64-token chat completion (non-thinking) to prove the loaded model generates.

Local dry run on a PC without GPU (`dryrun/fn_smoke_report.json`): stage 0 fails as expected, report + bundle +
fallback submission written. The ARC-test path (used only when `LOAD_ONLY=False`) was unit-tested with mocked server
responses.

## What I want from you

1. **Can this work at all on 4×L4?** Driver/CUDA 13 on Kaggle L4 machines; Pennyroyal/SGLang kernels and the
   `qwen4_exp` model code on **sm89** (anything hard-wired to sm90/sm100/sm120: flashinfer GDN / mamba kernels, MoE
   runner, AutoRound/Marlin INT4 MoE, fp8 KV on Ada); memory per GPU (INT4 experts ≈ 60 GB total + replicated parts +
   KV + mamba state) and host RAM for the 102 GB PLE.
2. **Flags**: do `--ep-size`, `--enable-dp-attention`, `--dp-size`, `--mamba-backend triton`,
   `--linear-attn-decode-backend triton`, `--linear-attn-prefill-backend triton`, `--ple-offload-embedding`,
   `--disable-cuda-graph`, `--default-chat-template-kwargs` exist in this fork, and is the attempt order sensible? Would
   TP=4 alone really fail on the 640/128 group issue, or does the fork handle it?
3. **Defects** in the notebook code (install, env, symlinks, launch loop, health check, kill/cleanup, report, bundle,
   45-min cap, stage-0 thresholds).
4. **Waste**: any way it runs long without learning anything? Anything to cut or add to get a clear yes/no faster?
5. Verdict: GO / NO-GO for this probe, and if NO-GO, the minimal changes.

## Map of the zip

```
PROMPT_FOR_REVIEWER.md
notebook/FN-SMOKE-FLASHNEXT-4XL4.ipynb      the notebook under review (sha256 in SHA256SUMS.txt)
build/make_fn_smoke.py                       builds the notebook (all cell code is in this file)
reference/franzen_launcher_cell12.py         dfranzen's original launcher cell (RTX PRO 6000), verbatim
reference/config.json                        the model's config.json (from the Kaggle model)
reference/ple_reshard_manifest.json, quantization_config.json
dryrun/fn_smoke_report.json, dryrun_console.txt
PUSH_PLAN.md                                 account, kernel, inputs, image, expected time/quota
SHA256SUMS.txt
```
