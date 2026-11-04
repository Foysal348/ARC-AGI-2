# Research task: 1500+ tok/s for Qwen3.8-Flash-Next on Kaggle 4×L4 — resources + plan

You are a research engineer. Search GitHub, Hugging Face, Kaggle (code, discussions, models, datasets), arXiv, vendor docs and the open web (blogs, Reddit r/LocalLLaMA, X/Twitter threads, Discord logs if public) and produce (A) a verified resource list and (B) a staged, concrete plan to reach **≥ 1500 tok/s aggregate decode throughput** for **Qwen3.8-Flash-Next** on **Kaggle 4× NVIDIA L4**, under the constraints below. Cite a URL for every claim. Mark anything you could not verify as **UNVERIFIED**. Do not answer "impossible" or "needs fine-tuning"; find the HOW. Prefer primary sources (code, commit, PR, model card, benchmark log) over summaries.

## The goal and why
- ARC-AGI-2 Kaggle competition (arc-prize-2026-arc-agi-2): 240 hidden tasks, one 12 h notebook run on 4×L4, no internet during the run.
- Our score is limited by coverage: at our current speed most tasks cannot get the ≥ 60k thinking tokens they need. 1500 tok/s aggregate = about 62M tokens in 11.5 h = about 259k tokens per task (two full 120k attempts per task).
- Public reports (Kaggle discussion https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546): "many of the leading teams are using Qwen3.8-Flash-Next along with the Duck harness" (started from Daniel Franzen's ARC-AGI-3 milestone-2 solution https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution); one team reports 980 tok/s peak / 710 average over 6 h on 4×L4 ("the secret sauce is how to have high concurrency when one problem reasons for more than 100k tokens"), model unstated.

## Hardware and platform (fixed)
- 4× L4: sm_89 (Ada), 23,034 MiB each (CUDA sees 22.03 GiB), ~300 GB/s HBM-equivalent bandwidth (GDDR6), 72 W power cap, PCIe, **no NVLink, no usable P2P** (NCCL P2P disabled; P2P was 5.5× slower than NCCL SHM in our tests).
- Host: ~173 GiB cgroup RAM, /dev/shm 14 GB, 4-8 CPU cores, glibc 2.35, pinned Docker image with Python 3.12. Internet OFF at run time: every wheel, kernel and checkpoint must be attached as a Kaggle Model/Dataset (a checkpoint loads about 15× faster from a Kaggle Model than from a notebook output).
- Notebook commit size limit ≈ 400 kB; payloads go in Models/Datasets.

## The model and what we already know (do not re-research these, extend them)
- Qwen3.8-Flash-Next (`qwen4_exp`): 125B-A5B MoE, 512 experts top-10, hybrid GDN (linear attention) + QSA (sparse attention), PLE (per-layer embedding table, ~51 GB FP8, kept in pinned host RAM), MTP head, 2 KV heads, vocab 248,320.
- Checkpoint in use: `halt95/Qwen3.8-Flash-Next-W4A16-Merlin` rev cec53e9f (INT4 experts, INT8 GDN, FP8 PLE), 115.3 GiB total. Also available: `Intel` W4A16 AutoRound (dfranzen mirror on Kaggle), albucino FP8-PLE overlay, an albucino DFlash drafter (4.2 GB), Qwen official FP8, an NVIDIA NVFP4 variant.
- Engine in use: **Merlin v3.0.0** (https://github.com/halt95/merlin-engine), a vLLM 0.32.0 fork; profile `fn-quad-24g-sm89`: TP2×PP2 + expert parallel, partition 25/23, FP8 E4M3 QSA KV with a scales sidecar, MTP K=3 always on, prefix caching, `--mamba-cache-mode align`, `--block-size 4096`, `--use-replayssm`, `--prefix-match-unit 64`, chunk 1024, capture sizes 1..32, max_num_seqs 8. Status "untested" on any Ada card.
- Knobs we found in Merlin v3.0.0 source: `KV_BYTES` (per-card KV pin; Merlin says adjust in 0.25 GiB steps), `CHUNK`, `BLOCK_SIZE`, `DRAFT_HEAD=bf16|int8|int4` (Merlin CHANGELOG: int8 = 6.5-7.2 % lower ms/token at 32K/131K/258K with the same MTP acceptance), `P2P`, `WARM`, `SCALES`, `AUTO_TOPO`. vLLM 0.32 `--language-model-only` skips loading the Qwen4Exp vision tower.
- Our L4 KV numbers (Merlin's block rule, blocks per request = ceil(L/4096)+7): FP8 KV pin 2,479,838,700 B = 150 blocks ≈ 558k tokens ≈ 5 streams at 70k; BF16 ≈ 292k ≈ 3 streams.
- Measured elsewhere (Merlin bench card, 4×RTX 3090): Flash-Next single stream 168-184 tok/s; 638 tok/s at 8 agents (16K context, thinking off); six 126,976-token sessions 574.5 tok/s. Our L4/3090 decode ratio estimate is 0.43-0.56 → projection on 4×L4: ~72-100 single stream, ~200-320 at 5-6 long streams (UNVERIFIED projections; our first real measurement is running now).
- Engines that failed for us: SGLang 0.5.21 with EP4 GPTQ-Marlin MoE gives wrong expert outputs (`!!!!`, NaN from layer 2) for this model on 4×L4. Stock vLLM 0.32.0 + Intel W4A16 (TP4 + EP, BF16 KV) gives correct output on 4×L4 but only a 30,208-token KV pool at TP4.
- Hard constraints from our side: reasoning effort xhigh; thinking budget never below 60k, target up to 120k (max-model-len 131072); MTP must keep prefix caching ON; any lossy change (KV dtype, SSM dtype, quantization) needs a correctness check against BF16 on long reasoning, not just short smoke tests.

## What to find (search all of these)
1. **Anyone running Flash-Next (or any qwen4_exp / Qwen3.8-Next-class hybrid MoE) on L4, 4090, 3090, A10, T4 or other consumer/sm_8x multi-GPU without NVLink**: engine, version, flags, layout (TP/PP/EP/DP), KV dtype, concurrency, measured tok/s at long context, and the exact repo/notebook/model card. Include Kaggle notebooks for ARC-AGI-2 and ARC-AGI-3 that use Flash-Next (Franzen's solutions and forks), and their engine settings.
2. **Engines and forks**: Merlin (all branches, issues, PRs, bench cards, changelogs after v3.0.0, any L4/Ada profile results), vLLM upstream qwen4_exp PRs and open issues (sm_89 paths, QSA Triton tuning tables, cooperative/persistent top-k, FlashInfer FP8 on sm_89, Marlin MoE for EP, MTP for qwen4_exp, DFlash drafter support), SGLang qwen4_exp status (is the EP4 GPTQ-Marlin MoE bug known/fixed?), llama.cpp / ik_llama / ExLlamaV3 / TensorRT-LLM support for this architecture, KTransformers or other CPU-GPU hybrid MoE engines that could use the 173 GiB host RAM (experts or PLE on host).
3. **Checkpoints on Hugging Face** for Flash-Next: every quantization (W4A16, W4A8, FP8, NVFP4, MXFP4, GGUF, EXL3, AWQ, GPTQ), pruned or reduced-expert variants (e.g. REAP/expert pruning), smaller-PLE variants, distilled "Swift"-style variants that think with fewer tokens (Merlin mentions Swift1.5: 31-58 % fewer thinking tokens), and draft models (MTP heads, DFlash/EAGLE drafters). For each: size, accuracy evidence, which engine loads it, whether it fits 4×22 GiB with room for KV.
4. **Throughput levers specific to this hardware**, with evidence and expected gain:
   - decode-time levers: MTP depth vs concurrency, DFlash/EAGLE drafters, draft head precision, CUDA graph capture under speculative decoding (are capture sizes counted in tokens = seqs×(1+k)?), chunked prefill size, scheduling policy;
   - memory/capacity levers: KV pin, KV dtype on QSA, block size and Mamba/GDN state size (SSM state dtype, align vs other cache modes), text-only mode, PLE placement (host vs GPU) and its CPU cost, expert-parallel vs tensor-parallel memory;
   - layout levers on PCIe without P2P: TP2×PP2 vs TP4 vs DP (2 replicas of TP2? does it fit?), all-reduce backends for PCIe (NCCL tuning, custom all-reduce off/on, FlashInfer PCIe IPC), expert placement;
   - kernel levers on sm_89: Marlin MoE tuning, Triton QSA tuning tables for sm_89, FP8 GEMM paths, attention backend choice.
5. **Workload levers** for a 12 h ARC run (these move the run average, not the peak): keeping concurrency high in the tail, prefix reuse across multi-turn agent calls (Duck harness), second pass only for unsolved tasks, ordering, early stopping.

## Output format
**A. Resource list** — a table: name | URL | type (repo/PR/issue/model/notebook/discussion/paper/post) | what it gives us | hardware it was measured on | evidence quality (measured / claimed / UNVERIFIED) | date.

**B. Lever ranking** — a table: lever | expected gain on 4×L4 (with the source and how you scaled it) | exact setting for vLLM 0.32 / Merlin v3.0.0 (flag, env, file) | correctness risk | prerequisite | cost to test (boots, minutes).

**C. Staged plan to ≥ 1500 tok/s** — 3-5 stages, each: what to change, what to measure (aggregate tok/s at 8k/40k/90k/116k contexts, real running streams, preemptions, KV usage, MTP acceptance, correctness vs BF16), pass/fail gate, and the expected cumulative tok/s. State the arithmetic (tokens per step × steps per second) that makes 1500 reachable, and which combination gets there.

**D. Alternatives if Merlin TP2×PP2 tops out** — other engines, layouts, checkpoints or drafters, with evidence.

**E. Risks and unknowns** — what could make a lever not work on sm_89 / no-P2P / glibc 2.35 / offline Kaggle, and how to detect it quickly.

Be complete: list everything relevant you find, even if minor, and say what you searched where results were empty.
