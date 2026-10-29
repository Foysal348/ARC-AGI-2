# Sources: Experiment 24 (SGLang speed track)

This file lists every primary URL cited in `../Qwen3_8_27B_SGLang_0_5_21_Research_BN_v5_FINAL.pdf` (v5.0, 2026-10-29, 48 pages), grouped by topic, plus the local evidence the plan relies on. The PDF section in brackets is where the source is used.

## SGLang source, releases and docs
- https://github.com/sgl-project/sglang/tree/v0.5.21: the pinned engine source [J S01].
- https://github.com/sgl-project/sglang/releases and https://github.com/sgl-project/sglang/releases/tag/v0.5.22: these hold the backport candidates #41736 / #41737 / #43029 [W4, F S-P3-04].
- https://docs.sglang.ai/ [J S22].
- https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx: the Qwen3.8-27B cookbook (prefill chunk 2048 guidance, Mamba slots) [W3, W5, AA4].
- https://raw.githubusercontent.com/sgl-project/sglang/main/test/registered/radix_cache/test_mamba_cache_kl.py: the Mamba-cache KL regression test [J S17].

## Correctness: DFlash, speculative state and Mamba cache (P0)
| Source | What it says |
|---|---|
| https://github.com/sgl-project/sglang/issues/36548 | Concurrent DFlash gives wrong answers on Qwen3.8-27B: serial 0/100, concurrent DFlash8 111/304, target-only 0/304. A BF16 lm_head export mitigated it to 1/304. Closed as stale, not fixed. [C, O3, AA5] |
| https://github.com/sgl-project/sglang/issues/35150 | Forced-reject target state drift. [O3, AA5] |
| https://github.com/sgl-project/sglang/issues/43514 | DFlash acceptance collapses into "!" output under grammar plus multi-chunk prefill. [S-P0-03, AA5] |
| https://github.com/sgl-project/sglang/issues/38009 | DFlash2 greedy output is unstable across restarts in thinking mode. [S-P0-03, AA5] |
| https://github.com/sgl-project/sglang/issues/42917 | W4A16 PPL gap (9.98 vs 6.05). The author's follow-up found a stale `~/.cache/sglang/nv` CUDA driver compute cache; a fresh cache gave PPL 6.28. [N1, N2, AA5] |
| https://github.com/sgl-project/sglang/pull/43567 | Stale lazy Mamba checkpoint is invalidated after a failed optional allocation. [S-P0-02, O2, AA5] |
| https://github.com/sgl-project/sglang/pull/43559 | Fixes the "Can not alloc mamba cache" assert: unlock the prefix and retry. [S-P0-02, O2] |
| https://github.com/sgl-project/sglang/pull/43029 | The speculative committed length can differ from the accepted length, giving the wrong Mamba checkpoint boundary. [S-P0-03, O1, W4] |
| https://github.com/sgl-project/sglang/pull/41736 | DFlash grouped-conv taps can contaminate across requests. [W4, AA5] |
| https://github.com/sgl-project/sglang/pull/41737 | DFlash2 all-NaN score row leads to an out-of-range greedy index. [W4, AA5] |
| https://github.com/sgl-project/sglang/pull/45151 | MTP cache-slot duplication / starvation (MTP branch only). [S-P2-04] |

## Capacity, Mamba slots and prefix reuse (P1)
| Source | What it says |
|---|---|
| https://github.com/sgl-project/sglang/issues/36889 | A hidden Mamba-slot admission cap: max-running 8 gave 2 actual. The fix measured +75 % in that setup. [W3, Y5, AA4] |
| https://github.com/sgl-project/sglang/issues/40865 | 93 % of requests retreat to an older checkpoint; about 30 % of prefill compute is wasted. Retention policy is recommended over replay. [Y2, AA6] |
| https://github.com/sgl-project/sglang/issues/22935 | A radix split creates `mamba_value=None`, which gives 0 cache hits. [W7, AA6] |
| https://github.com/sgl-project/sglang/issues/22326 | Related radix / Mamba issue. [W7] |
| https://github.com/sgl-project/sglang/issues/37834 | ReplaySSM forces no_buffer. Prefix hit fell 71.91 → 29.23 % and output 134.5 → 93.4 tok/s. [S-P3-01, R3, Y9] |
| https://github.com/sgl-project/sglang/issues/28511 | ReplaySSM design and performance. [S-P3-01] |
| https://github.com/sgl-project/sglang/pull/28695 | ReplaySSM implementation. [S-P3-01] |
| https://github.com/vllm-project/vllm/issues/54919 | SM89 long-prefill / decode interference reproduction. This is vLLM, used as the contrast recipe. [S-P2-01, W5, Y7] |

## Communication: NCCL and host-staged all-reduce
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html: NCCL_MIN_CTAS / MAX_CTAS / ALGO / PROTO / NTHREADS [D, Q1, W6].
- https://github.com/NVIDIA/nccl/blob/master/docs/userguide/source/troubleshooting/performance_and_tuning.rst [J S16].
- https://github.com/NVIDIA/nccl-tests: message-size replay [W6, Y8].
- https://github.com/sgl-project/sglang/pull/39605: the no-P2P host-staged collective (TP2, open), which is the base for the 4-rank port [S-P3-03, AA3].
- https://github.com/syv-ai/HyperQwen/issues/254 [AA3].

## Parallelism, alternative checkpoints and speculation
- https://github.com/sgl-project/sglang/issues/43885: SGLang pipeline-parallel support [S-P3-02, S3].
- https://github.com/halt95/merlin-engine and https://github.com/halt95/qwen38-27b-3090s: Merlin TP1x4 873 tok/s on CMP 170HX, plus kernel ideas (in_proj_ba split-K, FlashInfer plan cache, no-sync MTP, RecoverSSM) [Q2, S2].
- https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin: a 16 GiB checkpoint that fits on one 24 GB GPU [S2].
- https://github.com/eagle-hq/EAGLE-4 and https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4: on B300 at c1, EAGLE4 863.8 vs DFlash2 672.6 vs MTP 534.5 tok/s [S1].

## Kernels on SM89
- https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4: single-L4 Qwen3.8-27B with DFlash2 on llama.cpp. Low-M dispatch fix `GGML_MMVQ_MAX=2` gave +16.2 % [Y4, AA7].

## Agent harness (Duck)
- https://github.com/da-fr/arc-agi-3-solution and https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md: Franzen's Duck uses retained prefill checkpoints, LRU refresh and larger trim blocks [Y3, AA6].

## Kaggle (deployment context only)
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2 and https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- https://www.kaggle.com/competitions/arc-prize-2025
- https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2
- https://www.kaggle.com/competitions/gemma-4-developer-agent
- https://www.kaggle.com/competitions/konwinski-prize
- https://www.kaggle.com/code/abdullahmeda/load-72b-awq-model-using-vllm-on-l4-x4

## Local evidence (in this folder)
| File | Content |
|---|---|
| `../../Base Engine/results/ems_report.md` | EMS SAFE engine matrix (notebook `8f01847c`, hosen42, 2026-10-26): every SGLang tok/s number in the README. |
| `../../Base Engine/server_logs/*.log` | The server logs of the EMS arms; the exact launch argv is read from these. |
| `measurements/E14A_nccl_bench.json` | Raw NCCL on Kaggle 4x L4. Graph-replay all-reduce of 983,040 B: SHM 343.88 µs, forced P2P 1,880.52 µs. NCCL 2.29.7, torch 2.13+cu130. |
| `research/final_report_bn.md` | Claude E22 research report, including the E14A 116 ms step breakdown and the 26 ms + 0.94 ms × rows fit. |
| `research/agent_reports/*.md` | 5 research notes with URLs: community evidence, engines / Flash-Next / Duck, KV capacity and harness, PCIe parallelism and graphs, spec decode and kernels. |
| `research/GPT_prompt_L4x4_speed.md` | The prompt used for the GPT research rounds. |

The earlier Bengali research reports are already in the repo under `reports/` and `research_notes/` (root): "Qwen 27B দ্রুততম সার্ভিং সেটআপ", "৪xL4 ইঞ্জিন বটলনেক যাচাই" and "৪xL4 এ ৯০০ টোকেন গতির উপায়".
