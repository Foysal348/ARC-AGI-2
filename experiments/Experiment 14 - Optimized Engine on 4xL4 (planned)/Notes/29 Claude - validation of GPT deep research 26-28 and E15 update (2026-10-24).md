# 29 — Validation of GPT deep research 26-28; E15 update (Claude, 2026-10-24)

The three GPT reports are kept verbatim in Notes/26, 27 and 28. **Below, every claim is checked against our own measurements.** These are author-reported numbers from the web; I did not reproduce them.

## 1. Speed evidence and what it means for us

| Source (from 26-28) | Their number | Matches / conflicts with ours | Use |
|---|---|---|---|
| Alexander Ollman, 2×3090, two replicas, patched vLLM, ~1k prompts | C8: **MTP k4 465** vs DFlash2 k7 411; C16: **MTP 610 vs DFlash2 402 (+52 %)** | Our E11-03 had MTP capped at 10 running (297 vs DFlash 356 at 8): **never a fair high-C test** | Strongest evidence for an MTP arm at 12-16 |
| moki1, 1×3090 | C8: MTP 380 vs DFlash2 374 | Tie | — |
| ashalliants NInfer, 1×3090 | MTP3 523 at C8; **DFlash2 C8 did not fit** (406 at C4) | DFlash memory cost (our 1.49 GB/GPU weights + 8 verify states) | Memory argument for MTP |
| Mr_Moonsilver, 4×3090 vLLM AWQ | MTP3 **slower than no-spec at C12 (−10 %) and C16-32 (−17 %)** | Our DFlash2 is +39 % over no-spec at 12 (E11-03) | The MTP win is **not guaranteed**: it must be measured |
| Inco AI, H200 C8 | DFlash2 1,291 vs MTP 934 | H200 only | Not transferable |
| tonyd2wild, 4×3090 TP4 DFlash2 n7 | **Prose C8: 145 tok/s**; code 230-243 single | Our workload is mostly reasoning prose plus code | DFlash may be weak exactly on our long-prose thinking |
| **Merlin (halt95), 4×3090 TP4, vLLM 0.30 + 5 patches** | **KV pool 1.95 M tokens; C8 702, C32 1,121 tok/s** | Our KV is 867k on the same 24 GB cards → we spend memory elsewhere (DFlash weights, DFlash verify states, BF16 lm_head / embedding, mem 0.83) | Memory ideas: **INT4 lm_head + MTP head**, **embedding table in host RAM**, calibrated FP8 KV scales. The author says it was not measured on non-P2P or sm_89 hosts. Not a drop-in for SGLang |
| SGLang cookbook (Qwen3.8-27B) | Mamba slots per request: extra_buffer_lazy 4 **plus D = 8 speculative intermediate states for DFlash2**; BF16 state 78.4 MB/slot | Explains part of the memory gap. MTP with 3-4 drafts has a smaller D; width 4 also halves D | Memory gain for **both** the MTP and width-4 arms |
| Our NCCL: "140 × 345 µs = 48.3 ms, not 56" (report 26) | — | **Resolved:** 56.3 = 48.3 all-reduce + 8.0 all-gather (Notes/22) | No change |
| 2×4090D vLLM: single 128 / TP2 163 / **PP2 120** at C4 | — | PP2 slowest | Supports leaving PP out |
| SGLang roadmap #43885 | PP + DFlash unsupported; PP + EAGLE/MTP partly | — | PP is not for now |
| HyperQwen 4×5060Ti PHB | P2P +2-16 % | Ours: P2P 5.5× slower | P2P stays rejected (different box) |
| SGLang **0.5.22 released 2026-10-19** | Hybrid-cache / spec fixes; FlashInfer ≥ 0.7 | — | Separate branch only |

## 2. Correctness evidence (a reason to test MTP)

- **#36548 (Gabriel Bryk's reproduction):** DFlash2 at concurrency 8 gave wrong outputs in **111/304** cases of an affected subset.
  - Target-only: 0/304. Serial: 0/100.
  - Width 4: 37/304. FP32 SSM: 37/304. CUDA graphs off: 37/304.
  - Dense BF16 lm_head: 1/304, on an NVFP4 target.
  - It is not reproduced on our stack. Our A1-4 gate (12 prompts × 256 tokens, c12) showed 0 large-margin flips. **That is too small a probe to exclude it.**
- **#38009:** greedy divergence with thinking on, DFlash2 TP4 on 3090s.
- **Effect on E15:** MTP is the lower-risk speculation path. Every E15 arm keeps the gate. E15 also adds **one longer concurrent probe** (c12 × 1,024 tokens, reasoning on) against target-only.

## 3. Platform / schedule facts

- **Kaggle entry and team-merger deadline: 2026-10-26.** Final submission: 2026-11-02 23:59 UTC. Check that every team member has accepted the rules.
- The 23 Oct leaderboard snapshot matches the user's list. Some later scores differ (ALbert 72.64 → 75.56, Domdolus 72.92 → 73.47) because of time, not an error.
- **No top team's ARC-AGI-2 method is public.** Tufa will publish after the deadline. mikelou1's statements are claims.
- Official Qwen3.8-27B ARC-AGI-2: xhigh 42.4, low 22.8, medium 13.2, none 1.5 (matches memory).

## 4. E15 changes (supersedes the arm list in Notes/25 §3 where they differ)

| Arm | Change |
|---|---|
| E15-M1 | **MTP exactly as the SGLang cookbook gives it:** `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, no draft model. Record the resolved mamba slots and KV pool |
| E15-M1b | MTP with steps 2 / draft 3, a cheaper verify (Mr_Moonsilver's regression warning) |
| E15-M3 | MTP at 12, 14, 16 really running, with mamba cache sized from the cookbook formula (D for MTP) |
| E15-W | **Not repeated:** width 4/6 come from R2 |
| E15-T1 | 2×TP2 + MTP stays, **after** M1. Stop if M1 loses to DFlash2 at 12 |
| E15-N | NCCL bench: baseline, `NCCL_P2P_DISABLE=1`, `NCCL_PROTO=Simple / LL / LL128`, `NCCL_ALGO=Ring / Tree` at 983 KB inside a graph |
| E15-C | **A chunked-prefill 2048 arm** on the best engine: decode stalls during prefill (the cookbook advice) |
| Gate | Plus a c12 × 1,024-token reasoning-on probe against target-only (the #36548 class) |

**Deferred (memory ideas from Merlin, an SGLang port is needed):** INT4 lm_head, embedding in host RAM, calibrated FP8 KV scales.
