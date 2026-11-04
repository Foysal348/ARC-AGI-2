# Experiment priority list toward 1500 tok/s, NB-K100 (v5, 2026-11-01): GPT audit #4 "Architecture-Level Research" integrated

**v4 stays valid** (`EXPERIMENT_PRIORITY_LIST_K100_v4_2026-11-01.md`). v5 adds audit #4, the Merlin source checks for every item it names, and research agent B's DFlash2 findings.

## A. Correction to audit #4's throughput math (important)
- **Audit #4** uses T ≈ B_active × A / t_step. With 24 streams and 50 ms that needs A = 3.125.
- **Under our PP2 with two micro-batches, every request advances once per two stage ticks.** The correct form (research agent B, checked against data) is **T = N × AL / (2 × tick)**.
  - K3ONLY v2 check: 24 × ~3.0 / (2 × 0.0687) = 524, measured 562 ✓.
  - Real Duck: 24 × 2.8 / (2 × 0.0572 projected) ≈ 590 at full occupancy vs 385-414 measured. The gap is occupancy, idle time and prefill.
- **For 1500 at 24 streams:** AL / tick ≥ 1500 × 2 / 24 = 125 per second. At tick 57 ms that is AL 7.1, above MTP K3's ceiling of 4. So the path to 1500 runs through more resident streams and a shorter tick, together with a higher AL.
- **The levers that move it:**
  - more resident streams N (capacity);
  - a shorter tick (stage-1 compute, MoE kernel);
  - higher AL (drafter);
  - occupancy, i.e. real active streams close to N (Duck idle, replay).
  - Example: N 40, tick 45 ms, AL 3.4 → 40 × 3.4 / 0.090 = 1511.
- **So capacity (N), not only AL, is the binding lever.** This raises P0-3 / P0-4 (KV capacity) and the PLE / weight-memory items that free GPU memory for KV.

## B. Audit #4 items vs Merlin ad42e37 (verified today)
| Item | Audit #4 claim | Merlin | Consequence |
|---|---|---|---|
| **DSpark method** (shadowpa0327/Qwen3.8-Flash-Next-DSpark: 5 layers, 664.6M, block 7, confidence head; coding-agent fine-tune 2.69 → 3.70 accepted / verify, below the MTP 7-step reference 4.24; step 1800 not served by the author) | new candidate | **Method present**: `method="dspark"` (`config/speculative.py:71`, `_validate_qwen3_omni_dspark`), `v1/worker/gpu/spec_decode/dspark/`, models `qwen3_dspark.py` (Qwen3DSparkModel / Qwen3OmniDSparkModel), gemma4, kimi_k3, deepseek_v4 | A Flash-Next (qwen4_exp) target needs the same aux-hidden-state taps + PP2 relay as our DFlash2 seam; our seam is DFlash-specific. **Port = extend `dflash_seam.py` to dspark.** Its own data say AL 3.70 < MTP-7 4.24, so lower priority than DFlash2 |
| **Adaptive verification (DScale idea; vLLM #47808 / #58065 / #57632 / #52816)** | Project 1, first choice | **Present but DSpark only**: `enable_adaptive_verification` ("Currently only supported for method='dspark'"), `v1/worker/gpu/spec_decode/adaptive_verification.py` (variable-length verify, its own CUDA-graph route) | Not for DFlash2 / MTP by flag. Two routes: (a) DSpark + adaptive verification, both by config once the seam supports dspark; (b) port adaptive verification to dflash2, which is larger |
| **Dynamic speculative decoding** | listed in the vLLM docs | **Present, method-generic**: `num_speculative_tokens_per_batch_size: [(lo, hi, K), …]` (`config/speculative.py:493`, scheduler `:348`). Full CUDA graphs only with the V2 runner (`config/vllm.py:1068-1085`), which we use. Disabled only with DP | **NEW cheap config lever (P1-9):** a K schedule per batch size, e.g. K4 at ≤ 8 running, K3 at 9-24. It helps the tails of the 6-h run (A/B/C re-runs and the end of the run at low concurrency). **MTP K4 has never been measured** (no K4 result in the Exp 25 results; an E23 prep command exists) → measure K4 at 1 / 4 / 8 streams first. MTP K2 at 8 streams was −7 %, so a lower K is not the direction |
| **Suffix decoding** | in the docs | Present (`suffix_decoding_*`), needs `arctic-inference` | P2: an ARC-trace / tool-output repetition test is cheap on CPU. It would combine with MTP only through the hybrid path, and it is untested |
| **DFlare / DScale / DSpark papers** | research | DFlare needs a new trained checkpoint | Out of scope before 11-02 |
| **perkel/Qwen3.8-Flash-Next-MC** (2-bit experts + 4-bit trunk + quantized DFlash2 drafter; RTX 5090, 8 conversations: DFlash2 993 vs MTP 779 tok/s, **×1.27**) | alternative engine | external format | **The most useful item for NBDF-K100: the first published sign that DFlash2 keeps a ×1.27 advantage over MTP on a 2-bit-expert target.** Research agent B found no published 2-3-bit data. This lifts the central DFlash2 estimate. The engine itself is single-device and not a 4×L4 path |
| **primitive-ai PLE quant** (FP8 49 GB, INT4 32 GB, NVFP4-style 28.8 GB; mmap overlay) | indirect | we already run FP8 PLE pinned on the host | An INT4 PLE halves host pinned memory and PLE PCIe traffic. Kaggle host RAM limits pinning, and the warm speed may be lower. **P2**: host-RAM / boot benefit, not decode |
| **CPU/GPU overlap (MRV2 async)** | Project 2 | Already on: V2 runner + async scheduling | The open part is Duck-side: separate a ready-for-GPU queue from waiting-for-tool, and prepare the next request during decode → **= v4 P1-7**, now with audit #4's 3 instruments: GPU idle intervals, ready-queue depth, scheduler CPU overhead |
| **PP stage feedback overlap** | Project 2 | Q-defer already did this for the sampled-token receive | Only the residual bubbles; the NB-S1 profiler will show them |
| **ARC-specialized drafter** (speculators framework; train on non-eval ARC traces) | Project 3 | — | Not feasible before 11-02 (audit #4 agrees). Long-term: our saved full thinking traces of every request are exactly this training data (keep training tasks separate from eval tasks) |
| **4 × TP1 replicas / compact engine** (antirez ds4, gguf, yalis offload_prefetch) | Project 4 | — | One L4 cannot hold the model plus KV (audit #4 agrees). Expert offload makes the tick PCIe-bound; this matches our measured negatives (live KV offload, TP / DP). **Feasibility-only paper study, no GPU** |
| **ExpertFlow / CRISP** | research | — | ExpertFlow targets offloaded MoE (not ours). CRISP is lossy (−3.6 % quality) → NO for exact-match ARC |

## C. Research agent B findings that change the list (report saved verbatim in Exp 27 Agent Reports)
- **DFlash2 on klee100, central estimate at 24 streams 8k:** D3s ×1.15 / D4s ×1.19 vs the BF16 head; **×1.09 / ×1.13 vs I8**. Pessimistic ×0.91-1.01. Real Duck projection 400-430 tok/s.
- **Default K = 3 (D3s).** K4 only if it beats D3s at 8k@24 by more than A/A.
- **Structural risks are shared with W4** (S1-S9). **S7** (prefix-cache warm turns with unwritten draft KV, 85 % hit in Duck) and **S8** (SW-1024 lowers the target engine block to 1024 too; kit R9 did not model it) are the big ones.
- **klee100 is not run-to-run deterministic** (repeat agreement 0.856-0.949). The EXACT greedy oracle would false-fail the D arms, so Q's oracle runs twice and the D arms are judged inside Q's A/A envelope.
- **Not adopted:**
  - B's "no-hope cut" (AL3 < 2.40) and the "D4s only if c ≥ 0.80" gate are measurement-label gates (the NBDF failure rule), so they are recorded only.
  - Zero-accept < 2 % stays a real failure, but it stops only its own arm.

## D. Ranking update (v5; unchanged rows as in v4)
| # | Experiment | Change |
|---|---|---|
| P0-1 | NBDF-K100 DFlash2 K3 first, then K4 | perkel ×1.27 on 2-bit raises confidence; B's A/A-envelope oracle; S7 / S8 checks |
| P0-3 / P0-4 | KV capacity (ledger, F, admission) + indexer fp8 | **raised in importance by the math in A: N is the binding lever** |
| **P1-9 (new)** | Dynamic K schedule (`num_speculative_tokens_per_batch_size`), after measuring MTP K4 at 1 / 4 / 8 streams | config only; for low-concurrency phases |
| **P1-10 (new)** | DSpark + adaptive verification: extend the seam to `dspark`, then `enable_adaptive_verification` | AL 3.70 < MTP-7 4.24 on the author's data → after DFlash2 |
| P1-7 | Duck async / ready queue + audit #4's 3 instruments | merged with audit #4 Project 2 |
| P2 | INT4 PLE (host RAM), suffix decoding test, ARC drafter training (after the competition), compact-engine paper study | new P2 rows |
| NO | CRISP (lossy), TP1 replicas with expert offload | added to the do-not list |
