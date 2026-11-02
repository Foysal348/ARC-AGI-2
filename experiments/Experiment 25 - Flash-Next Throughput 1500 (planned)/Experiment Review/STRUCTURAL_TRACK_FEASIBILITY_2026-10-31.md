# Structural track feasibility: Flash-Next on 4×L4 (CPU-only research, 2026-10-31)

**Scope.** The structural branch of PLAN_1500 v3.6 (§1.1: a steady 1,700 needs a fixed floor of ~6-12 ms, a ≥ 3.5, more
resident streams, fewer weight bytes and/or a single-batch layout; §3 C6, C7 and the alternative branches). No GPU, no
Kaggle API, no pushes, no commits.

**Inputs read in full:** `PLAN_1500.md` v3.6, `05_GPT_REPORT_2_*` + verification, `06_GPT_REPORT_3_*` + verification and roofline.

**Also checked:**
- the Merlin v3.0.0 source tree (`D:/ARC AGI 2/_scratch/e23/p2/merlin-src/`);
- the Merlin lockfile (`_scratch/e23/p2/lockcheck.json`);
- the humming wheel shipped in the Merlin wheelhouse (`_scratch/e23/p2/depwheels/humming_kernels-0.1.17-cp310-abi3-manylinux_2_28_x86_64.whl`, unpacked as text only);
- the RC2 result (`Experiment 13/.../E13_RC2_GPU_v1_2026-10-31.md`);
- the Hugging Face model cards, GitHub pages and issues listed in §9 (WebFetch on 2026-10-31).

**Evidence labels:**

| Label | Meaning |
|---|---|
| MEASURED | A hardware run with numbers, or a source line I read myself |
| CLAIMED | An author's statement or a model-card number I did not reproduce |
| UNVERIFIED | A plausible mechanism, not shown on our stack |
| (est.) | My arithmetic. The model and its constants are in §3.1; the script is `scratchpad/residency.py`, see §10 |

---

## 0. Verdicts in one place

1. **Merlin already contains a sub-4-bit MoE path.** Marlin is 4/8-bit only, but Merlin's vLLM 0.32 tree routes 2-, 3-, 5-, 6- and 7-bit WNA16 experts to the **humming** JIT kernel library. This holds for both AutoRound/INC (`auto_round:auto_gptq` / awq) and compressed-tensors `pack_quantized` checkpoints. Humming supports sm_75+ and has an explicit `Sm89Heuristics` table. Its wheel, plus the nvcc/nvrtc/cccl 13.0 wheels it JIT-compiles with, are already in Merlin's lockfile. (MEASURED in source and lockfile; end-to-end on L4 UNVERIFIED.)
   - **So a 3-bit or 2-bit Flash-Next can run inside the current Merlin TP2×PP2+EP stack. No engine switch is needed.** This is the main finding.
2. **Two all-512-expert, vLLM-loadable, sub-4-bit Flash-Next checkpoints exist:**
   - `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`: 2/3-bit experts at ~2.998 bpw, 4-bit MTP experts, `auto_round:auto_gptq` packing.
   - `HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang`: 2-bit g128 sym experts, `auto_round:auto_gptq` packing, **no MTP**.

   Neither has an accuracy comparison against the base on reasoning benchmarks. klee100 reports KLD 0.1208. (CLAIMED)
3. **The best published accuracy per bit is in non-vLLM formats:**
   - GSQ-RCO GGUF IQ3_XXS at 3.0 bpw: GPQA-D 91.41 vs 91.92, AIME25 100 vs 100;
   - EXL3 self-calibrated 2.68-3.18 bpw: GPQA-D 91.4 vs 91.7.

   Merlin cannot load either (it has no GGUF quantization module and no EXL3 path). (CLAIMED numbers; MEASURED absence in source)
4. **Residency math (est.)** in today's TP2×PP2 layout, with the C5 pin headroom included:

   | Expert format | Blocks (today 149) | Resident at 68k | Resident at 90k |
   |---|---|---|---|
   | INT4 (today, with C5) | 220 | 9 | 7 |
   | 3-bit g128 | ≈ 452 | 19 | 16 |
   | 2-bit g128 | ≈ 680 | 29 | 24 |

5. **Single-batch layouts do not help while KV binds:**
   - **TP4/EP4** duplicates the 2 QSA KV heads, and QSA rejects context parallelism (`qsa.py:571-577`). At equal bits it holds about half the streams of TP2×PP2, which cancels its merged expert read. The MoE-plus-floor ceiling is the same within ±3 %.
   - **2 × TP2 replicas** do not fit any vLLM-loadable checkpoint: about 1.7 bpw experts are needed just to leave KV room.
6. **Deeper drafters are blocked in Merlin by source:**
   - the spec-method allowlist is `{mtp, ngram, ngram_gpu}` (`model_executor/models/config.py:1049-1058`);
   - `Qwen4ExpForCausalLM` lacks the EAGLE3/DFlash aux-hidden-state interface.

   The only Flash-Next drafter with numbers, incoai DFlash2, is block 4: at most 4 tokens per step, the same ceiling as MTP K3. Its published acceptance is 2.83-3.26 vs MTP 2.50-3.08 at xhigh with a 4k cap (CLAIMED). That is about +10 %, not the a ≥ 3.5 the model asks for.
7. **The roofline still says "all levers together".** With 2-bit experts at the resident C ≈ 29: 857 tok/s steady at today's 24.2 ms floor, 1,165 at a 12 ms floor, 1,415 at 6 ms (250 GB/s, a = 2.72, est.). **Low-bit experts are the single largest structural lever (about 3× the steady ceiling via capacity), but 1,700 also needs the floor cut.**

---

## 1. Lower-bit Flash-Next checkpoints (question 1)

Today's reference:
- **Merlin INT4 g128 experts** (`halt95/Qwen3.8-Flash-Next-W4A16-Merlin`): 115.3 GiB on disk including the FP8 PLE (MEASURED, CPU kit v1).
- **GPU weights** 17.33 GiB (PP0) / 16.87 GiB (PP1) per card (MEASURED, RC2 `Model loading took …`).
- **One INT4 g128 expert** = 3 × 2560 × 640 × 4.125 bit = 2.534 MB (est., matches PLAN §1).
- **All 24,576 routed experts** ≈ 62.3 GB = 58.0 GiB (est.; Minachist's card independently states "about 58.0 GiB" for INT4 g128 experts, CLAIMED).

| # | Checkpoint (URL) | Format / bits (experts) | All 512 experts? | Size | Published accuracy vs base | Engine it needs | Label |
|---|---|---|---|---|---|---|---|
| 1 | klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP — https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP | AutoRound 0.15.0, `quant_method: auto-round`, `packing_format: auto_round:auto_gptq`, default `bits 2, group_size 64, sym`, `extra_config` per module; card: "2/3-bit routed experts (~2.998 effective bpw), 8-bit main projections, BF16 sensitive paths, and 4-bit MTP experts" | yes (no pruning stated; 512-expert layout in config) | loaded "approximately 47.3 GiB VRAM" (single GPU, incl. MTP); 13 shards; PLE "95.37 GiB BF16" on SSD | "KLD vs original weights: 0.1208"; no task scores | vLLM commit `a5a30471ff2b…` + `ple_ssd_io.so` patch; tested on "a 64 GiB SM80 GPU"; 1 req 111 tok/s, 16 req 667 tok/s aggregate (MTP = 1, ≤ 32k); "an arithmetic consistency issue remains" | CLAIMED (card, config.json) |
| 2 | HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang — https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang | AutoRound 0.14.2, `auto_round:auto_gptq`; "512 experts per layer are 2-bit (group size 128, symmetric)"; QSA q/k/v/o, GDN in/out, shared expert, lm_head INT8 g128; routers, HC, embeddings BF16 | yes ("None are pruned") | 38,755,352,600 B LM tensors (36.09 GiB); ~89.99 GB with the 51.2 GB FP8 PLE; **no MTP**, no vision | none vs base ("not a comparison with the FP8 base model"); self-referential NLL, needles 5/5 at 41k/248k | patched SGLang `73a255206f` only ("does not load in … vLLM … or unpatched SGLang"); sm_120 RTX PRO 4000; 1 stream 50.8-57 tok/s | CLAIMED |
| 3 | turboderp/Qwen3.8-Flash-Next-exl3 — https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3 | EXL3 trellis; plain 2.05/3.05/4.05/5.05/6.05 bpw; self-calibrated (SC) 1.66/2.18/2.68/3.18/3.68/4.18 bpw | not stated (1.66: "mostly K=1.5 expert tensors") | `2.05bpw_h4_ng4`: 62,815,131,785 B total = 36.41 GB model shards + 26.24 GB `ngram_embedding.safetensors` (MEASURED via HF API tree listing); 1.66 ≈ 50.2 GB (GPT report 3, UNVERIFIED) | GPQA-D: SC 1.66 86.4 %, 2.18 89.4 %, 2.68 91.4 %, 3.18 91.4 %, 3.68 91.9 % vs original 91.7 %; ERQA 62.7-68.5 vs 68.8 | ExLlamaV3 (+ TabbyAPI); vllm-exl3 plugin ran 0.62× at c1 (adrienbrault) | CLAIMED |
| 4 | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF — https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF | GGUF: Q2_0 2.40, IQ2_XS 2.50, IQ3_XXS 3.00, IQ3_S 3.50 bpw (transformer weights); PLE IQ4_NL 4.5 bpw | yes (the separate Coder repo is the pruned one) | weights 37.6 / 39.2 / 47.0 / 54.8 GB + PLE 28.8 GB = 66.4 / 68.0 / 75.8 / 83.6 GB | BF16 / Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S: AIME25 100 / 96.67 / 96.67 / 100 / 100; GPQA-D 91.92 / 89.39 / 87.37 / 91.41 / 92.93; LCB v6 87.43 / 81.14 / 83.43 / 86.29 / 86.86; task avg 93.12 / 89.07 / 89.16 / 92.57 / 93.26 | llama.cpp / Ollama / LM Studio ("unmodified"), NInfer; MTP not mentioned | CLAIMED |
| 5 | ggml-org/Qwen3.8-Flash-Next-GGUF — https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF | Q2_K, IQ4_NL, Q8_0 | yes | 83.1 / 102 / 163 GB; sidecars: MTP Q4_0 2.2 GB, Q8_0 4.14 GB, BF16 7.77 GB; DFlash2 Q4_0 351 MB, Q8_0 652 MB, BF16 1.22 GB | none ("TODO add info") | llama.cpp (`llama serve`) | CLAIMED |
| 6 | trymirai/Qwen3.8-Flash-Next-S-small-experimental — https://huggingface.co/trymirai/Qwen3.8-Flash-Next-S-small-experimental | QTIP-style trellis "Mirai S" (`qwen_next_s`), routed experts avg 1.85 bpw (gate_up 1.73, down 2.03); attention/GDN/shared/LM head 5 bit | implied (per-expert rates), not explicit | trellis 30.7 GB + 4.2 GB other (incl. 2.7 GB FP8 MTP) + n-gram 51.2 GB FP8 / 26.2 GB int4; **GPU weights 36.5 GiB** | ROMA KL 0.0742 (FP8 table) / 0.0692 (int4), top-1 agreement 91 %; hard set 51/55 vs FP8 model 52/55 | vLLM 0.31.0 plugin; **single GPU only, no TP**; tested only on B200; backbone decoded to int8 after load | CLAIMED |
| 7 | Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound — https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound | compressed-tensors; experts **INT4 g128** (58.0 GiB), attention/GDN/shared INT6 g64, HC INT8; 4.178 body bpw | yes | experts same bytes as Merlin; PLE 95.37 GiB BF16 | GPQA-D 90.4 vs 91.7; LCB 92.4 vs 91.9; IFBench 81.0 vs 81.3; KLD mean 0.0147 (`main`) / 0.0117 (`new-main`) | vLLM 0.29.1rc1 + patches; 3×3090 | CLAIMED. **No byte gain** (INT4 experts): not a structural candidate |
| 8 | Intel/Qwen3.8-Flash-Next-W4A16-AutoRound; cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4 | INT4 | yes | INT4 class (exact bytes not fetched) | Intel: correct on stock vLLM 0.32 L4 short tests [A3]; AWQ: none fetched | stock vLLM / SGLang | **No byte gain**: not structural |
| 9 | nvidia / RadixArk Qwen3.8-Flash-Next-NVFP4 | NVFP4 (4.5 bpw incl. scales) | yes | 133 GB (GPT report 3) | GPQA-D 91.5 vs FP8 92.0 (GPT report 3) | Blackwell FP4; on sm_89 only Marlin-FP4 emulation | CLAIMED. **More bytes than INT4 g128**: not structural |
| 10 | cnygaard GLQ 3 bpw (`Qwen3.8-Flash-Next-GLQ-trellis-3inst-3bpw-ple4`) — https://github.com/cnygaard/glq | GLQ trellis 3 bpw | — | "That checkpoint has since been withdrawn" | none for Flash-Next | GLQ vLLM plugin (vLLM 0.30.0), single card; **L4 measured 17.211 tok/s B = 1 with an 8 GiB expert arena** (experts offloaded) | CLAIMED. Withdrawn: not usable |
| 11 | ISTA-DASLab …GSQ-RCO-Coder-GGUF; sh0wie REAP-288-MLX; neopolita Niwaki | expert-pruned (256 / 288 experts) or MLX | **no** (pruned) | 58.4 GB; ~68 GB; ~29.5 GB | coding only; "long multi-step regression" | llama.cpp / MLX | CLAIMED. Pruned, against the "keep 512 experts" preference; MLX is not CUDA |

**Two candidates are also vLLM-format-loadable sub-4-bit with all 512 experts: rows 1 and 2.** Everything at ≤ 3.5 bpw with
published reasoning-benchmark parity (rows 3, 4) is in a format Merlin does not read (§5).

---

## 2. Engine capability per candidate (question 2)

| Engine (candidate rows) | CUDA path on sm_89 | Multi-GPU | MTP / drafter | ≥ 120k multi-stream serving | Prefix caching (multi-turn tools) | OpenAI-compatible server | Offline install on the Kaggle image (Py 3.12, glibc 2.35) |
|---|---|---|---|---|---|---|---|
| **Merlin v3.0.0 + humming** (rows 1, 2, any self-made CT/INC 2-3-bit) | humming: "SM75+", `Sm89Heuristics` (`tune/__init__.py` heuristics_map 89; `tune/sm8x.py:112`), default MMA = `mma.sync` on sm_89 (`config/mma.py:97-104`), NVRTC JIT `--gpu-architecture=sm_89` (`jit/compiler.py:186-188`). MEASURED in source; decode speed on L4 UNVERIFIED | TP2×PP2+EP as today. Humming MoE supports `expert_map` / EP (`fused_humming_moe.py:674-760`), gemm type `grouped_contiguous` under EP (`:108-123`), `_supports_parallel_config` → True (`:417-418`). MEASURED source | native MTP (K3); klee100 ships a 4-bit MTP; Haberstroh has none | the same engine that ran RC2 116k@4, 90k@3 | same Merlin prefix/ReplaySSM path (independent of the expert kernel) | yes (vLLM serve) | humming 0.1.17 + `nvidia-cuda-nvcc 13.0.88`, `nvidia-cuda-nvrtc 13.0.88`, `nvidia-cuda-cccl 13.0.85` are in Merlin's lockfile (MEASURED `_scratch/e23/p2/lockcheck.json`); wheel tag manylinux_2_28 (MEASURED). JIT cubins cache under `HUMMING_CACHE_DIR` (`utils/jit.py:118-119`), so they can ship in the B1 cache Model. Import on the RC2 venv UNVERIFIED |
| **klee100's own vLLM fork** (row 1) | tested on SM80 only (CLAIMED) | single GPU (CLAIMED) | MTP = 1 measured | 262k advertised, timing to 32k only | stock vLLM prefix | yes | arbitrary vLLM commit + custom .so: not worth a second fork; use Merlin instead |
| **Patched SGLang** (row 2) | sm_120-tuned Triton 2-bit configs (CLAIMED); sm_89 UNVERIFIED | single GPU | none | 1 stream to 258k | SGLang radix | yes | another SGLang fork; we already carry an unexplained SGLang LB gap (PLAN §0) |
| **ExLlamaV3 + TabbyAPI** (row 3) | release 1.5.2 "Experimental Turing support", 1.6.1 "Turing (sm_75) optimizations" ⇒ Ampere/Ada covered (CLAIMED by implication; sm_89 never named) | "TP support for Qwen3.8-Flash-Next" (1.5.1) and EP (CLAIMED); adrienbrault ran a 2-GPU layer split `[30, 30]` (CLAIMED) | MTP with "dynamic draft sizing with TP model" (1.5.3); served static table depth 3-4 at ≤ 4 jobs, 2 at 5-8 (CLAIMED) | 8 slots on an 819,200-token pool (8-bit KV), 2×5090 (CLAIMED); multi-stream decode measured with 1,024-token generations only; long context 1-stream only (252.8 t/s at 199,822) | page-pool sharing exists; prefix numbers not published (UNVERIFIED) | TabbyAPI (MEASURED by adrienbrault) | wheels `exllamav3-1.6.5+cu128.torch2.10.0-cp312-…-linux_x86_64.whl` (CLAIMED, releases page); a separate torch 2.10 cu128 venv; glibc of the .so UNVERIFIED (plain `linux_x86_64` tag, not manylinux) |
| **llama.cpp (+ lukolszewski multigpu fork)** (rows 4, 5) | CUDA builds cover Ada (fork: "cuda-12.9 covers V100 through RTX 5090"; "Only sm_86 runs every day") CLAIMED | layer split (sequential stages), row split; no EP | MTP/DFlash2 sidecars exist (ggml-org); fork author: MTP "didn't help, slowed me down in most cases"; n-gram only ≤ 2 active decodes | 5 slots × 262k on 6×3090: **per session 27.3 tok/s ⇒ ~136 aggregate** (CLAIMED) | per-slot prompt cache; KV is per slot (n_ctx/np), not a shared paged pool | `llama-server` (yes) | prebuilt Linux CUDA binaries (Ubuntu builds; glibc fine, CLAIMED) |
| **NInfer-all** (row 4 Q2_0) | builds noted `sm_86` / `120a` (CLAIMED); sm_89 not named | 2 GPUs: one PP stage each, or `--tensor-parallel` | `--spec mtp --draft-tokens N` (1-15) | ≤ 8 concurrent requests; 1-stream numbers only (2×4090 Q2_0 92.6-100.7 tok/s) | not described | `ninfer-serve` (OpenAI compatibility not stated) | source build / Docker only |
| **Mirai plugin** (row 6) | not stated; B200 only | **none (single GPU)**; 36.5 GiB weights > 22.5 GiB L4 | MTP-3 | — | stock vLLM 0.31 | yes | needs a TP2 port first |
| **GLQ plugin** (row 10) | L4 measured (17.2 tok/s, offloaded) | single card | — | — | — | yes | checkpoint withdrawn |

---

## 3. Weight residency and freed KV per card (question 3)

### 3.1 Method (all est. unless marked)

- **Expert bytes:** expert = 4,915,200 params × b_eff / 8, with b_eff = bits + 16/group (BF16/FP16 group scales, symmetric).
  - INT4 g128 = 4.125 → 2.534 MB.
  - INT3 g128 = 3.125 → 1.920 MB.
  - klee100 ≈ 2.998 → 1.842 MB.
  - INT2 g64 = 2.25 → 1.382 MB.
  - INT2 g128 = 2.125 → 1.306 MB.
  - EXL3 2.05 / 1.66 for comparison.
- **Experts per card:**
  - TP2×PP2+EP: PP0 25 layers × 256 = 6,400; PP1 23 × 256 + 256 MTP = 6,144.
  - TP4/EP4: 48 × 128 + 128 = 6,272.
  - 2 × TP2 replicas: 48 × 256 + 256 = 12,544.
- **Non-expert weights per card** from RC2 (MEASURED 17.33 / 16.87 GiB minus INT4 experts): PP0 2.22 GiB, PP1 2.37 GiB.
  - Held constant across checkpoints. This is optimistic for klee100 (8-bit main projections + BF16 sensitive paths) and close for Haberstroh (INT8 dense).
  - TP4: 3.16 GiB (est.: embed + lm_head + draft head /4, replicated HC ~27 MB × 48, sharded layer weights /4).
  - DP2: 4.90 GiB (est.).
- **KV headroom**, calibrated on RC2 (MEASURED initial free 21.67 GiB, pin 2.31 GiB, min free 1,607 MiB PP0 / 1,633 MiB PP1):
  - KV_max = 21.67 − W − overhead − 0.5 GiB margin;
  - overhead 0.46 GiB (PP0) / 0.857 GiB (PP1 and other layouts);
  - today's PP0 KV_max = 3.38 GiB = 2.31 + 1.57 − 0.5. This reproduces the ~215-220-block C5 estimate.
- **Blocks:** 2,479,838,700 B / 150 = 15.77 MiB per 4096-token block per card (PLAN v3.6). Each request needs ceil(L/4096) + F blocks, F = 6.
  - **TP4:** the token part costs 2× per card, because each rank holds one of the 2 KV heads for all 12 QSA layers instead of 6. Replication is source-confirmed at `nvidia/qsa.py:594-600`, with no context parallelism at `:571-577`. The GDN state part costs about 1× (36 layers / 4 ≈ 25 layers / 2 × 36/48).
  - **2 × TP2 replicas:** each request sits on 2 cards at 2× per-card cost, across 2 replicas. Total capacity equals the PP2 formula at the same free bytes per card: no duplication, but the weights are doubled.
- **Roofline:**
  - **Approach:** PLAN §1.1 step model with the MoE term scaled by bits; D(T) = 512(1 − (1 − 10/512)^T), a = 2.72.
  - **PP2:** tick = 24.18 ms + 24 × D(2C)/2 × E/BW; steady = C·a/(2·tick).
  - **TP4:** step = floor + 48 × D(4C)/4 × E/BW.
  - **DP2:** step = floor + 48 × D(2C)/2 × E/BW.
  - **Uncertainty:** the 24.18 ms floor is FITTED on RC2 at C 2-8. Applying it to TP4/DP2 is an assumption, and extrapolation to C 16-48 is UNVERIFIED (PLAN §1.1).

### 3.2 Residency table (est.)

| Expert format | Layout | Weights / card GiB | KV max / card GiB | Blocks (15.77 MiB units) | Δ vs today 149 | Resident C at 68k | at 90k |
|---|---|---|---|---|---|---|---|
| Merlin INT4 g128 (today, MEASURED 149) | TP2×PP2+EP | 17.33 / 16.87 | 2.31 (pinned) | 149 | 0 | 6 | 5 |
| Merlin INT4 + C5 pin headroom | TP2×PP2+EP | 17.33 / 16.87 | 3.38 | 220 | +71 | 9 | 7 |
| Merlin INT4 | TP4/EP4 | 17.96 | 2.35 | 153 (token part 2×) | +4 | 3 | 3 |
| Merlin INT4 | 2×TP2 replicas | 34.51 | −14.2 | infeasible | — | 0 | 0 |
| INT3 g128 | TP2×PP2+EP | 13.67 / 13.35 | 6.96 | 452 | +303 | 19 | 16 |
| INT3 g128 | TP4/EP4 | 14.38 | 5.94 | 386 | +237 | 9 | 7 |
| INT3 g128 | 2×TP2 replicas | 27.33 | −7.0 | infeasible | — | 0 | 0 |
| klee100 2/3-bit (~2.998) | TP2×PP2+EP | 13.20 / 12.91 | 7.41 | 481 | +332 | 20 | 17 |
| klee100 2/3-bit | TP4/EP4 | 13.92 | 6.39 | 415 | +266 | 10 | 8 |
| INT2 g64 | TP2×PP2+EP | 10.46 / 10.28 | 10.03 | 652 | +503 | 28 | 23 |
| INT2 g64 | TP4/EP4 | 11.23 | 9.08 | 590 | +441 | 14 | 11 |
| INT2 g128 (Haberstroh) | TP2×PP2+EP | 10.01 / 9.84 | 10.47 | 680 | +531 | 29 | 24 |
| INT2 g128 | TP4/EP4 | 10.79 | 9.53 | 619 | +470 | 15 | 12 |
| INT2 g128 | 2×TP2 replicas | 20.15 | 0.16 | ~10 | — | 0 | 0 |
| EXL3 2.05 (other engine, for scale) | 2×TP2 replicas | 19.61 | 0.70 | ~45 | — | 0 | 0 |
| EXL3 1.66 (other engine) | 2×TP2 replicas | 16.82 | 3.50 | 227 | +78 | 8 | 8 |
| EXL3 1.66 | TP2×PP2-equivalent | 8.30 / 8.20 | 12.11 | 786 | +637 | 34 | 28 |

**Notes on the table:**
- The "Δ vs today" for low-bit rows already includes the C5 headroom (+71). The low-bit gain alone is about +232 (3-bit) and about +460 (2-bit g128).
- **Required blocks for the PLAN §1.1 STRUCTURAL_TARGET rows** at 68k / 90k: C 16: 368 / 448; C 24: 552 / 672; C 32: 736 / 896.
  - 3-bit reaches C 16 at 68k (452 ≥ 368).
  - 2-bit reaches about C 24-29 at 68k.
  - Nothing vLLM-loadable reaches C 32 at 90k.
- **The F = 6 fixed blocks become a larger share:** at C 29, 174 of 680 blocks (26 %). C1 (shrinking F) compounds with low bits.
- **Engine limits at these concurrencies:**
  - Merlin's "seqs 8, the only concurrency the engine is gated at" (PLAN §6 Q18) must be lifted;
  - captures must cover (C/2)(1+K) per micro-batch: C 29 → 58 → capture 64.

### 3.3 Throughput ceiling at the resident C (68k), a = 2.72 (est.)

| Expert format | Layout | C | MoE-only, 250 GB/s | + 24.18 ms floor, 250 GB/s | + floor, 173 GB/s (RC2-implied) |
|---|---|---|---|---|---|
| INT4 + C5 | PP2 | 9 | 658 | 286 | 240 |
| INT4 | TP4 | 3 | 622 | 219 | 189 |
| INT3 g128 | PP2 | 19 | 1,038 | 527 | 430 |
| INT3 g128 | TP4 | 9 | 1,020 | 508 | 416 |
| klee100 | PP2 | 20 | 1,101 | 556 | 454 |
| INT2 g64 | PP2 | 28 | 1,676 | 812 | 668 |
| INT2 g128 | PP2 | 29 | 1,804 | 857 | 707 |
| INT2 g128 | TP4 | 15 | 1,833 | 879 | 724 |
| EXL3 1.66 | DP2 | 8 | 1,604 | 576 | 497 |

**Floor sensitivity (PP2, 250 GB/s):**

| Expert format | C | MoE ms/tick | 24.18 ms floor | 12 ms floor | 6 ms floor |
|---|---|---|---|---|---|
| INT4 + C5 | 9 | 18.6 | 286 | 400 | 497 |
| INT3 | 19 | 24.9 | 527 | 701 | 837 |
| INT2 g64 | 28 | 22.7 | 812 | 1,097 | 1,326 |
| INT2 g128 | 29 | 21.9 | 857 | 1,165 | 1,415 |

**MoE bytes per generated token** (whole system; GB/token):

| Expert format | PP2 C 16 | PP2 C 32 | TP4 single batch C 16 | TP4 single batch C 32 |
|---|---|---|---|---|
| INT4 | 1.340 | 1.026 | 1.026 | 0.658 |
| INT3 | 1.015 | 0.777 | 0.777 | 0.499 |
| klee100 | 0.974 | 0.746 | 0.746 | 0.478 |
| INT2 g64 | 0.731 | 0.560 | 0.560 | 0.359 |
| INT2 g128 | 0.690 | 0.529 | 0.529 | 0.339 |

**Reading:**
- **At equal bits,** a TP4 single batch reads 23 % (C 16) to 36 % (C 32) fewer MoE bytes per token than PP2. TP4 loses about half the streams, though, so at the resident C the two layouts tie.
- **Lower bits cut bytes per token by** −24 % (3-bit), −27 % (klee100), −45 % (INT2 g64) and −48 % (INT2 g128). They also roughly double to triple C, which lowers bytes per token further through D(T) saturation: INT4 at C 9 → INT2 at C 29 is about 1.34 → 0.55 GB/token.

---

## 4. Deeper drafters for Flash-Next (question 4)

| Drafter (URL) | What it is | Published acceptance / speed | Long reasoning? | Engine | Label |
|---|---|---|---|---|---|
| incoai/Qwen3.8-Flash-Next-DFlash2 — https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2 | 0.6B BF16 block-diffusion drafter, **block size 4 = 3 draft tokens per verify, "limited by the current QSA implementation"**; target Qwen3.8-Flash-Next-FP8 | c1, T 1.0, top-p 0.95, xhigh, max 4,096 new tokens incl. reasoning, 128 samples. Acceptance MTP vs DFlash2: GSM8K 3.08 / 3.26; MATH-500 2.93 / 3.17; HumanEval 2.52 / 2.94; MBPP 2.65 / 3.06; MT-Bench 2.50 / 2.83. GB300 TP1 c8: e.g. MATH 1,248.4 (MTP) vs 1,473.1 (DFlash2); c32 MATH 2,566.6 vs 3,173.3 | **no long-context results**; capped at 4k tokens | patched SGLang `3c560089…` + `qwen38-dflash2-capture.patch`, TP1 only | CLAIMED |
| PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash — https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash | DeepSpec DFlash, 5 layers, hidden 2560, 498M params BF16, taps [3, 15, 23, 35, 43]; block 5 headline (trained 7) | greedy, non-thinking, 256 output tokens, 8k context, eager, c1: block 5 52.322 vs MTP k4 50.371 tok/s (+3.87 %); **served accepted length 2.883 vs native MTP k4 3.367**; chat −5.90 % | "no long-context measurement", "no thinking-mode or sampling measurements" | vLLM `e962733e…` + 3 patches + 6-file overlay; "SGLang cannot serve it"; 2× DGX Spark | CLAIMED |
| ggml-org DFlash2 / MTP sidecars — https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF | DFlash2 Q4_0 351 MB / Q8_0 652 MB / BF16 1.22 GB (BF16 size ≈ a 0.6B model, consistent with incoai's) | none | none | llama.cpp (usage not documented) | CLAIMED |
| EAGLE / EAGLE3 heads for Flash-Next | — | **none found** (web search 2026-10-31) | — | — | NOT FOUND |
| Native MTP deeper than K3 | the MTP layer applied recursively | pulse K6 3.75-3.89 tokens/step, noisy [G1v]; Merlin CMP K4 −24 % at 4 streams [A3]; incoai MTP numbers above are K3 | — | Merlin | CLAIMED / prior |

**vLLM issue #56088** (https://github.com/vllm-project/vllm/issues/56088, open since 2026-09-09; no PRs linked in its Development section). Five blockers (CLAIMED by the issue author, one boot each):
1. The V1 runner never prepares PLE inputs ("PLE inputs were not prepared").
2. The V2 runner subscripts a None MTP hidden-state getter.
3. The DFlash speculator rejects DeepSpec's `sample_from_anchor=True`.
4. The spec-method allowlist omits `dflash` / `dspark`.
5. Adaptive verification is incompatible with the GDN backends: no device-side query-length trimming, and it forces full CUDA graphs.

The issue also states the target model needs "contracted 2560-width auxiliary hidden states", a `query_zero_predicts_next` path and an allowlist entry in `llm_base_proposer.py`. The overlay lives in github.com/PixelML/deepspec-qwen38-flash-next. The last comment references roozich/Qwen3.8-Flash-Spark commit "unblock dflash/DFlash2 drafting on Qwen4Exp (patch 23 …)" (Oct 27).

**Merlin source state (MEASURED):**
- **Blocker 4 is present:** `vllm/model_executor/models/config.py:1047-1058` raises `NotImplementedError("Qwen4Exp speculative decoding supports only its native MTP checkpoint and linear n-gram proposers")` for any method outside `{"mtp", "ngram", "ngram_gpu"}`.
- **The DFlash and DFlash2 draft classes exist,** but only for Qwen3 targets (`model_executor/models/registry.py:643-645`: `DFlashDraftModel → qwen3_dflash`, `DFlash2DraftModel → qwen3_dflash2`). The DFlash slot rule is in `config/speculative.py:1960-1975` (DFlash = K extra slots).
- **No aux-hidden-state interface:** `Qwen4ExpForCausalLM` (`models/qwen4_exp/nvidia/model.py:866-876`) implements HasInnerState, SupportsHCSP, SupportsLoRA, SupportsMRoPE, SupportsPP, IsHybrid and SupportsReplaySSM, but **not SupportsEagle3**. `v1/worker/gpu_model_runner.py:5268-5273` raises "Model does not support EAGLE3 interface but aux_hidden_state_outputs was requested".

**What Merlin would need for DFlash2** (est. effort: high, 1-2 weeks of patching plus GPU validation):
- allowlist entry;
- SupportsEagle3-style aux taps on the 2560-width hidden stream (the HC 4-stream contraction);
- a DFlash2 draft model class bound to Qwen4Exp embed/lm_head;
- a non-causal draft attention backend that works on sm_89 (the incoai card uses `trtllm_mha`);
- RecoverSSM/GDN rollback for parallel drafts (the K-slot rule);
- PP2 placement of the drafter on the last rank;
- graph captures for the new query shapes.

**Ceiling:** block 4 still caps a ≤ 4. The published xhigh-4k gain over MTP is +0.18…+0.42 accepted tokens (about +6-17 %), unmeasured at 60-120k. **A ≥ 3.5 needs a block ≥ 6-8 drafter that QSA does not support yet** (per the incoai card).

---

## 5. Can Merlin run non-INT4 expert formats? (question 5)

**Marlin MoE is 4/8-bit only (MEASURED):**
- `fused_moe/experts/marlin_moe.py:337` and `:519`: `assert num_bits in [4, 8]`.
- Supported weight keys at `:623-642`: kInt4*, kInt8Static, FP8, MXFP4, MXFP8, NVFP4; no int2/int3.
- `fused_moe/config.py:669-674`: the GPTQ-Marlin quant config raises `Unsupported weight_bits` for anything other than 4 or 8.
- `quantization/inc/schemes/inc_wna16_scheme.py:247`: `use_marlin = layer_config.bits in (4, 8) …`.
- The Triton WNA16 experts are int4/int8 only (`fused_moe/experts/triton_moe.py:639-647`).

**The sub-4-bit path is humming, not Marlin (MEASURED source):**

1. **INC / AutoRound** (`quant_method: auto-round`):
   - `quantization/inc/inc.py:37` `SUPPORTED_BITS = {2, 3, 4, 5, 6, 7, 8}`.
   - `inc_wna16_scheme.py:23-28`: "On CUDA the Marlin/GPTQ/AWQ kernels only cover 4/8-bit. The remaining widths (2/3/5/6/7) … are dispatched to the humming kernel", `CUDA_HUMMING_SUPPORTED_BITS = {2, 3, 5, 6, 7}`.
   - MoE: `:169-174` → `_build_humming_moe_method`. Linear: `:141-148` → `_build_humming_linear_method`.
   - Only gptq- or awq-packed int checkpoints (`:276-297`).
   - So klee100 (`auto_round:auto_gptq`, 2/3-bit experts + 4-bit MTP) and Haberstroh (`auto_round:auto_gptq`, 2-bit) resolve to humming for the 2/3-bit MoE and to Marlin for their 4/8-bit layers inside one model ("so a single model can mix 4/8-bit (Marlin) and 2/3/5/6/7-bit (humming) layers").
2. **compressed-tensors** (`pack_quantized`):
   - `schemes/compressed_tensors_wNa16.py:36-54`: types for 2-8 bits (`uint2b2`, `uint3b4`, …).
   - `compressed_tensors_moe/compressed_tensors_moe.py:~92`: MoE accepts `WNA16_SUPPORTED_BITS`.
   - `compressed_tensors_moe_wna16.py:65-110` builds a non-int4/int8 weight key and calls the oracle.
   - `fused_moe/oracle/int_wna16.py:126-143`: the priority list is RDNA3, FLASHINFER_TRTLLM, MARLIN, BATCHED_MARLIN, TRITON, HUMMING, EMULATION. Humming moves ahead of Marlin only on cc 9.0 (`quantization/utils/humming/priority.py:12`). Marlin and Triton reject the 2/3-bit key, so the oracle falls to HUMMING.
   - `fused_humming_moe.py:100-105` accepts any unsigned-int weight key with 2 ≤ bits ≤ 8; `:400-406` requires `has_humming()` and cc ≥ 7.5.
   - So a self-made 3-bit compressed-tensors checkpoint in Merlin's own format (the current Merlin checkpoint is compressed-tensors W4A16 g128, per `rc2_final_bundle/docs/MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:113, 244`) would also route to humming.
3. **Packaging (MEASURED):**
   - `requirements/cuda.txt:30-31` `humming-kernels[cu13]==0.1.17` (`setup.py:1507-1508` swaps in cu12);
   - Merlin's lockfile pins humming 0.1.17 plus nvcc/nvrtc 13.0.88 and cccl 13.0.85;
   - the wheel is in `_scratch/e23/p2/depwheels/`.
4. **What the humming wheel says about itself** (MEASURED from the wheel, CLAIMED performance):
   - "JIT-compiled GPU GEMM kernels for quantized inference";
   - BF16 activations on "SM80+" with "Symmetric INT1-8" weights;
   - FP8 e4m3 activations on "SM89+" with "Symmetric INT1-5" weights;
   - "Supports both Dense GEMM and MoE GEMM".
   - The sm_89 heuristic class exists (`tune/sm8x.py:112`, `Sm89Heuristics(Sm86Heuristics)`, max smem 99 KiB). On sm_89 the default MMA is `mma.sync` (`config/mma.py:97-104`). Compilation is NVRTC with `--gpu-architecture=sm_89` (`jit/compiler.py:186-188`), with a cubin cache under `HUMMING_CACHE_DIR`.
   - vLLM patches humming's DeviceInfo so it traces under fullgraph compile (`vllm/utils/humming.py:107-157`), which suggests the CUDA-graph integration is intended.
5. **What is NOT in Merlin (MEASURED):**
   - no GGUF quantization module (`model_executor/layers/quantization/` has no `gguf.py`);
   - no EXL3/QTIP trellis path;
   - so rows 3, 4, 5, 6 and 10 need another engine or a conversion.

**Remaining unknowns (UNVERIFIED, each a CPU-kit or 1-GPU check):**
- (a) humming's decode-shape speed on L4 at M = 16-64 tokens per micro-batch with 256 local experts and top-10. Nobody has published sm_89 MoE numbers. Marlin's own L4 number is unknown too (PLAN D5).
- (b) Every FusedMoE layer needs one bit-width across its 512 experts, because the quant method is per module. klee100's "2/3-bit routed experts" must be per layer or per projection, not per expert. Check its `extra_config`.
- (c) Merlin-specific load hooks: FP8 PLE with a global scale (`ple_format_guard.py`), INT8 GDN, MTP draft quant config (`mtp.py:167-200`), the HC mixer layout. Third-party checkpoints carry a BF16 PLE (klee100: 95.37 GiB). That does not fit the cgroup budget next to the page cache (RC2: the 47.7 GiB FP8 pin already sat at the limit). Merlin's FP8 PLE table plus `text_config.ple_embedding_dtype = float8_e4m3fn` would have to be grafted in. The table is model-independent (EXL3 card: "`ngram_embedding.safetensors` is swappable between quants"), but the graft is UNVERIFIED.
- (d) Graph capture of humming launches inside Merlin's FULL_AND_PIECEWISE graphs.
- (e) JIT compile time at boot.

**Verdict:** a 3-bit or 2-bit expert path inside Merlin is realistic (dispatch, packaging and the sm_89 heuristic exist). It needs a checkpoint, a CPU resolution check and one L4 kernel benchmark. It does not need another engine.

---

## 6. Ranked shortlist for the next CPU kit (question 6)

### S1 (rank 1): Low-bit experts inside Merlin through humming, layout unchanged (TP2×PP2+EP, MTP K3)

**Arms:** 3-bit (klee100 as the off-the-shelf checkpoint) first; 2-bit g128 (Haberstroh format) as the second arm.

| Item | 3-bit (klee100 ~2.998 / INT3 g128) | 2-bit g128 |
|---|---|---|
| Expert bytes per token at equal C | −24 to −27 % | −48 % |
| Bytes per token at resident C (68k) | 1.34 → ~0.83 GB/token | 1.34 → ~0.55 |
| KV blocks | 149 → ~452-481 (incl. C5); resident 6 → 19-20 at 68k, 5 → 16-17 at 90k | 149 → ~680; resident 29 / 24 |
| Steady ceiling at the 24 ms floor (est.) | 527-556 | 857 |

**Accuracy risk:**
- **3-bit, medium.** Good 3-bit quantizers reach parity: GSQ IQ3_XXS 3.0 bpw GPQA-D 91.41 vs 91.92, AIME25 100 vs 100; EXL3 SC 3.18 91.4 vs 91.7. klee100's own KLD of 0.1208 is high, though: Minachist INT4 has a mean of 0.0147, and the trace differs. Our 27B INT4 lost about 5 task-norm points in Duck.
- **2-bit, high.** GSQ Q2_0 task average 89.07 vs 93.12; EXL3 2.18 GPQA 89.4.
- Both need the paired ARC run (PLAN §0) before adoption.

**Effort (est.):**
- CPU kit 0.5-1 day;
- 1-L4 kernel benchmark ~30 GPU-min;
- checkpoint graft (Merlin's FP8 PLE + MTP) 1-2 days;
- Kaggle Model upload of ~50 GB;
- one boot plus the paired ARC check (≈ 3 GPU-h, PLAN v3.3 design).

**First cheap test (CPU kit, no GPU):**
1. In the RC2 Merlin venv on the Kaggle CPU image: `import humming`, run `humming.utils.nvrtc.may_build_nvrtc_compile_binary()`, and NVRTC-compile one MoE kernel for `sm_89` with a fixed config. This checks that the JIT toolchain and libnvrtc load offline.
2. Download only `config.json` and `model.safetensors.index.json` of klee100 and Haberstroh. Build Merlin's Qwen4Exp on the meta device with each quant config and record, per module, the resolved method: humming MoE vs Marlin vs unquantized. Assert one bit-width per FusedMoE layer, the MTP draft quant method, and the PLE method / `check_ple_table_format` outcome with Merlin's FP8 table grafted in.
3. Compute the exact per-card bytes from the index (replaces this report's est.).

**First cheap GPU test:**
- Extend `Experiment 25/.../Build/nb3_moe_bench.py` with humming W3 g128 / W2 g128 / W4 arms against Marlin W4 on one L4:
  - local E = 256, top-10, hidden 2560, intermediate 640;
  - M = 8, 16, 32, 48, 64 tokens;
  - real router ids from NB1 when available, uniform routing otherwise;
  - bitwise / greedy output check vs a dequantized BF16 reference.
- **Gate:** humming W3 time ≤ 0.80× Marlin W4, and W2 ≤ 0.60×, at M 32-64.
- The capacity gain is valuable even at kernel parity: C 9 → 19 at INT4 speed is about +80 % steady in the model. So a kernel loss up to ~1.3× would still be net positive. That threshold is the go/no-go for a full boot.

### S2 (rank 2): An accuracy-controlled, Merlin-format 3/4-bit (per-layer mixed) compressed-tensors checkpoint built by us

**Why:** it is S1 with better accuracy control. klee100's runtime notes "an arithmetic consistency issue remains", its KLD is 0.12, and it carries a BF16 PLE and 8-bit dense layers that Merlin does not expect.

**Design:**
- Keep Merlin's exact non-expert tensors: INT8 GDN, FP8 PLE, MTP, HC.
- Requantize only the routed experts from the BF16/FP8 base with GPTQ/AutoRound to 3-bit g64/g128, written as compressed-tensors `pack_quantized`. This routes to humming (§5.2).
- Keep sensitive layers at INT4: per-layer choice, since bits must be uniform within a FusedMoE layer. Choose them by per-layer KLD on ARC-like traces.

| Item | Value |
|---|---|
| Bytes per token (experts) | −20 to −24 % (mixed 3.25-3.125 bpw) |
| KV | +200-300 blocks (est.); resident about 16-19 at 68k |
| Accuracy risk | lowest of the sub-4-bit options (we choose the layers); still needs the paired ARC run |
| Effort | high: GPU hours to calibrate 24,576 experts (AutoRound/GPTQ on 4×L4 or rented GPUs, offline), a converter, a Kaggle Model upload of ~45 GB |

**First cheap test (CPU):**
- RTN 3-bit g64 of two layers' experts from the FP8 base on CPU;
- measure weight reconstruction error and the KLD of one layer's MoE output on recorded hidden states (NB1 dumps), vs Merlin INT4.
- This tells whether plain RTN is enough or GPTQ/AutoRound is required, before any GPU time.

### S3 (rank 3): DFlash2 drafter port into Merlin (a: 2.72 → ~2.9-3.1 if the xhigh-4k gains hold at 60-120k)

| Item | Value |
|---|---|
| Bytes per token | −6 to −13 % (more accepted tokens per verify read); the drafter adds ~1.2 GB BF16 weights (+ draft KV) on the last PP rank |
| KV capacity | −3 to −5 % (est.; drafter weights and its own KV). Can be offset with an int8 drafter (ggml-org ships Q8_0 / Q4_0 variants) |
| Accuracy risk | low in principle (lossless verification, CLAIMED); medium in practice (GDN rollback with parallel drafts; #63593-class state bugs) |
| Effort | high (§4 list: allowlist, aux taps, draft class, non-causal sm_89 attention, RecoverSSM rollback, PP placement, captures). It cannot exceed 4 tokens per step (block 4, a QSA limit), so it **does not** deliver the a ≥ 3.5 row |

**First cheap test (CPU):**
- Load `incoai/Qwen3.8-Flash-Next-DFlash2` `config.json` + safetensors index;
- map its expected target taps and embed/lm_head binding onto Merlin's Qwen4Exp module names;
- list every code site the patch must touch (the source map above), with a line count.

**First GPU test:** offline acceptance on our recorded RC2 xhigh traces (teacher-forced, MTP vs DFlash2 per position). The PixelML card warns that offline E[L] did not predict served acceptance, so treat it as an upper bound.

### Considered and not shortlisted

| Option | Why not |
|---|---|
| **TP4/EP4 single batch** | Equal ceiling at the resident C (§3.3); −50 % streams; TP4 all-reduces over SHM at world 4 |
| **2 × TP2 replicas** | Needs ≤ ~1.7 bpw experts just to fit, and gives about 8 streams at 68k even then |
| **ExLlamaV3 + EXL3 as an engine switch** | Best accuracy per bit (SC 2.68 bpw ≈ base GPQA). But it is a new engine with no Merlin features (RecoverSSM, retention, cost-aware admission). Measured multi-stream only with 1k-token generations on sm_120; vllm-exl3 0.62×. Keep as the fallback if S1's humming kernel loses badly on L4: then EXL3's trellis kernel is the other sm_89-capable low-bit path. First cheap test then: install the 1.6.5 cp312 cu128 wheel on the Kaggle CPU image, `readelf` the .so glibc versions, import |
| **llama.cpp / NInfer / GGUF** | Aggregate multi-session throughput far below target (~136 tok/s for 5 sessions at 250k on 6×3090); no shared paged KV; MTP reported negative |
| **Mirai S-small** | Single GPU, 36.5 GiB > 22.5 GiB, B200 only |
| **GLQ** | Withdrawn |
| **NVFP4 / AWQ / Intel / Minachist INT4** | No byte gain |
| **Pruned 256/288-expert models** | Against the keep-512 preference; "long multi-step regression" reported |

---

## 7. Findings list (complete)

1. MEASURED (source): Merlin's Marlin MoE asserts `num_bits in [4, 8]` (`marlin_moe.py:337, 519`). Its supported keys exclude int2/int3 (`:623-642`). The Triton WNA16 experts are int4/int8 only (`triton_moe.py:639-647`).
2. MEASURED (source): Merlin routes 2/3/5/6/7-bit INC (AutoRound gptq/awq-packed) MoE and linear layers to humming on CUDA (`inc_wna16_scheme.py:23-28, 141-148, 169-174, 276-297`; `inc.py:37`).
3. MEASURED (source): compressed-tensors `pack_quantized` 2-8-bit MoE is accepted (`compressed_tensors_wNa16.py:36-54`). The WNA16 oracle falls through Marlin/Triton to HUMMING for 2/3-bit keys (`oracle/int_wna16.py:126-143, 210-226, 256-363`; `fused_humming_moe.py:100-105, 400-406`).
4. MEASURED (lockfile / wheel): humming-kernels 0.1.17 plus nvcc/nvrtc 13.0.88 and cccl 13.0.85 are in Merlin's lockfile. The wheel is manylinux_2_28 and has an `Sm89Heuristics` table, `mma.sync` on sm_89, and NVRTC JIT with a cacheable cubin directory (`HUMMING_CACHE_DIR`).
5. CLAIMED (humming README): BF16-activation INT1-8 weights on SM80+, FP8-activation INT1-5 on SM89+, dense and MoE GEMM.
6. UNVERIFIED: humming MoE decode speed on L4; graph capture inside Merlin; JIT boot time; whether `import humming` works in the RC2 venv.
7. CLAIMED: klee100 3bpw-MTP is a vLLM-format (`auto-round`, `auto_round:auto_gptq`) all-512-expert checkpoint: 2/3-bit experts (~2.998 bpw), 4-bit MTP experts, 47.3 GiB GPU weights, KLD 0.1208, 16-request 667 tok/s on SM80, "arithmetic consistency issue remains".
8. CLAIMED: Haberstroh int2-mixed is 2-bit g128 sym, all 512 experts, 36.09 GiB LM tensors, `auto_round:auto_gptq`, no MTP, patched SGLang only, no accuracy vs base.
9. CLAIMED: EXL3 SC 2.68/3.18 bpw GPQA-D 91.4 vs 91.7, 2.18 → 89.4, 1.66 → 86.4. MEASURED (HF API): the 2.05 bpw branch is 62.82 GB = 36.41 GB model + 26.24 GB n-gram table.
10. CLAIMED: GSQ-RCO IQ3_XXS 3.0 bpw ≈ base (AIME25 100, GPQA-D 91.41, LCB 86.29 vs 100 / 91.92 / 87.43). Q2_0 2.40 bpw task average 89.07 vs 93.12. GGUF only.
11. CLAIMED: ggml-org GGUF Q2_K 83.1 GB, with MTP (Q4_0 2.2 GB) and DFlash2 (BF16 1.22 GB) sidecars; no numbers.
12. CLAIMED: Mirai S-small is 1.85 bpw experts, 36.5 GiB GPU weights, single GPU, B200 only, ROMA KL 0.069-0.074.
13. CLAIMED: Minachist INT4/INT6 keeps INT4 g128 experts (58.0 GiB). No byte gain; GPQA-D 90.4 vs 91.7.
14. CLAIMED: GLQ measured Flash-Next on an L4 only with experts offloaded (17.2 tok/s B = 1). The checkpoint is withdrawn.
15. MEASURED (source): Merlin has no GGUF and no EXL3 quantization path.
16. MEASURED (source): QSA rejects context parallelism (`nvidia/qsa.py:571-577`) and replicates KV heads when TP > 2 (`:594-600`). So TP4 doubles per-card per-token KV, consistent with the 3090 −57 % pool [A3].
17. (est.) In TP2×PP2, 3-bit experts free ~3.6 GiB per card and 2-bit ~7.1 GiB. Blocks 149 → ~452 / ~680 (incl. C5 +71); resident at 68k 6 → 19 / 29; at 90k 5 → 16 / 24.
18. (est.) TP4/EP4 at equal bits holds about half the streams of TP2×PP2. Its merged-batch expert read (−23 % at C 16, −36 % at C 32 MoE bytes per token) only cancels that. Ceilings tie within ±3 %.
19. (est.) 2 × TP2 replicas need about ≤ 1.7 bpw experts to leave any KV. Not feasible with a vLLM-loadable checkpoint; EXL3 1.66 gives about 8 streams at 68k.
20. (est.) Steady ceilings at the resident C (68k), a = 2.72, 250 GB/s:

    | Expert format | 24.2 ms floor | 12 ms floor | 6 ms floor |
    |---|---|---|---|
    | INT4 + C5 | 286 | 400 | 497 |
    | INT3 | 527 | 701 | 837 |
    | INT2 g128 | 857 | 1,165 | 1,415 |

    1,700 steady needs low bits and a floor ≤ ~6 ms, plus a higher a, together.
21. MEASURED (source): Merlin allows only `mtp`, `ngram` and `ngram_gpu` for Qwen4Exp (`models/config.py:1047-1058`). `Qwen4ExpForCausalLM` lacks SupportsEagle3 (`nvidia/model.py:866-876`; `gpu_model_runner.py:5268-5273`). The DFlash/DFlash2 classes exist only for Qwen3 targets (`registry.py:643-645`).
22. CLAIMED: incoai DFlash2 (0.6B, block 4 = 3 drafts, a QSA limit) acceptance 2.83-3.26 vs MTP 2.50-3.08 at xhigh, 4k cap, c1, GB300, patched SGLang TP1. No long-context data.
23. CLAIMED: PixelML DeepSpec DFlash served acceptance 2.883 vs native MTP k4 3.367; +3.87 % c1 tok/s, non-thinking 8k on GB10, vLLM with 3 patches + an overlay.
24. CLAIMED: vLLM #56088 lists five blockers. It is open, and no fix PRs are linked.
25. NOT FOUND: any EAGLE/EAGLE3 head for Flash-Next.
26. CLAIMED: ExLlamaV3 1.6.5 ships cp312 cu128 torch 2.10 wheels and claims TP for Flash-Next (1.5.1) and Turing support (implying sm_89). The 2×5090 2.68 bpw run reached 795 tok/s at 8 streams on 1k-token generations; vllm-exl3 ran at 0.62×.
27. CLAIMED: the llama.cpp multigpu fork reaches 5 × 27.3 tok/s at 250k on 6×3090, and MTP "slowed me down". Not a throughput path.
28. Ranked shortlist: S1 Merlin + humming low-bit experts (klee100 3-bit, then a 2-bit arm); S2 our own Merlin-format per-layer 3/4-bit compressed-tensors checkpoint; S3 DFlash2 port into Merlin. Each has its first CPU test and gate in §6.

---

## 8. What this report did not verify
- Any GPU number for humming, EXL3, llama.cpp or NInfer on an L4.
- The full klee100 `extra_config`: WebFetch saw the first 100k of 4.86M characters, so the per-layer 2- vs 3-bit split and the per-FusedMoE-layer uniformity are open.
- The klee100 on-disk size. Only "47.3 GiB VRAM" and 13 shards are stated.
- The cyankiwi AWQ and Intel AutoRound byte sizes (not fetched; INT4 class, irrelevant to the byte lever).
- Whether Merlin's MTP experts are INT4 in the Merlin checkpoint. The per-card split assumes yes for PP1.
- The 24.18 ms floor and the 0.46/0.86 GiB overhead terms outside RC2's C 2-8 range and layouts.

## 9. Sources (fetched or read 2026-10-31)

**Hugging Face model cards:**
- https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP (+ `/raw/main/config.json`)
- https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang (+ config.json)
- https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3; https://huggingface.co/api/models/turboderp/Qwen3.8-Flash-Next-exl3/tree/2.05bpw_h4_ng4
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF
- https://huggingface.co/trymirai/Qwen3.8-Flash-Next-S-small-experimental
- https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound
- https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2
- https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash

**GitHub:**
- https://github.com/cnygaard/glq
- https://github.com/turboderp-org/exllamav3; https://github.com/turboderp-org/exllamav3/releases
- https://github.com/adrienbrault/qwen3.8-flash-next-2x-rtx5090
- https://github.com/ggml-org/llama.cpp/discussions/30071
- https://github.com/iamwavecut/ninfer-all/blob/master/docs/qwen3-8-flash-next.md
- https://github.com/vllm-project/vllm/issues/56088

**Local:**
- `D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/…` (paths and lines inline above);
- `D:/ARC AGI 2/_scratch/e23/p2/lockcheck.json`;
- `D:/ARC AGI 2/_scratch/e23/p2/depwheels/humming_kernels-0.1.17-cp310-abi3-manylinux_2_28_x86_64.whl`;
- `D:/ARC AGI 2/_scratch/rc2_final_bundle/docs/MERLIN_FN_L4_SYNTHESIS_2026-10-31.md`;
- `D:/ARC AGI 2/experiments/Experiment 13 - …/Experiment Results/E13_RC2_GPU_v1_2026-10-31.md`.

## 10. Reproducing the arithmetic
The residency and roofline tables come from a 120-line script with the constants in §3.1. Its path:
`D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/residency.py`.
It is scratch and not committed; copy it into `Experiment Notebooks/Build/` if the CPU kit should reuse it.
