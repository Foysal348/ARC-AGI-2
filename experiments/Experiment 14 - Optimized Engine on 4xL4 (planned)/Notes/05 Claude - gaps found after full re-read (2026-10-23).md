# What was missed or skipped: full re-read of the 5 research notes and GPT notes 01/03/04 (Claude, 2026-10-23)

The user asked for this re-read: "আবার দেখে আসো ডিপলি… তুমি অনেক কিছুই মিস করো" ("look again, deeply… you miss a lot").

**What I read completely:**
- `research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/`: our_measurements (615 lines), speculative_decoding (278), engines_and_config (405), weight_quantization (493) and community_evidence (596).
- GPT notes 01, 03 and 04.

## A. My own errors (now corrected)

1. **"PCIe Gen4 x16 64 GB/s"** is the bidirectional aggregate. One direction is ~31.5–32 GB/s (GPT 03/04).
2. **"4.3 of 8 accepted, 46 % wasted."**
   - SGLang's "accept len" includes the bonus token. At width 8 there are 7 draft positions, so 4.3 means 3.3 correct drafts and a draft-accept rate of ~47 %.
   - Not all rejected work is waste either: a wider verify saves future sequential passes (GPT 04).
   - The log's "accept rate" field confirms it: accept len 4.07 with rate 0.44 gives (4.07−1)/7 = 0.44.
3. **"Only 10–16 TFLOPS used, so compute is not the bottleneck"** is not established. The marginal step cost also includes communication, launches, dequant, GDN and spec overhead (GPT 01/03/04).
4. **"128 all-reduces × 64 MB bandwidth = 40–50 ms"** applies large-message bandwidth to small messages. Small collectives are latency-bound, so this is a hypothesis, not a measurement.
5. **PARITY notebook shas:** the final report quoted stale ones (f3876b40 / 85da1be3 / bc9e2bc2). The rev 6g PARITYR6 is `82b257a5…`, and it never ran.
6. **The cm391 comment in topic 751344** ("old environment image … would take ages for anything to work … you should just update") was dismissed as "not our issue", because we pin an old Python 3.12 image on purpose.
   - The rev 6g G4B commit on esfersami50 ran ~114 min and ended in ERROR with an empty log and 0 output files. Our first cell prints at once, so an empty log means the notebook code probably never started.
   - A slow start or pull of the pinned old image is now a candidate cause. It is not verified.

## B. Things that were in the notes but not carried into the decisions

1. **Effective rate ≠ decode rate is the biggest gap.**
   - E12 Duck EVAL30: decode ~412–431 tok/s, effective only ~187 tok/s over the solve window. 144 of 423 non-first requests lost their prefix; 8.39 M re-prefill tokens.
   - G4B rev 6d: effective 277 tok/s.
   - So agent-side cache and admission loss (about 35–55 %) is bigger than any engine lever we discussed. GPT 04 says the same: "throughput ≠ score".
   - ARC-AGI-3 winners' lessons all point here: rellik13 went 14.5 → 22.5 from KV and history alone. The levers are KV-resident admission, trim hysteresis, Mamba checkpoint retention, `--schedule-policy lpm`, `--page-size 64` and `--mamba-track-interval 64`.
2. **SGLang HiCache** (hierarchical KV cache to host RAM) was used in the club-3090 #1288 4×3090 run (927 tok/s at c16). Kaggle 4×L4 has 202.6 GB of host RAM, and evicted Duck prefixes could live there instead of being re-prefilled. Never considered.
3. **DSpark drafter:** the RedHat INT4 card recommends DSpark with 7 draft tokens "for optimal performance", and it needs block+1 state slots. Never tested; vendor accept length is 3.62 vs DFlash2 4.80.
4. **`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`:** lazy goes from 4 to 3 slots per request. Untested.
5. **ReplaySSM** (D=0 verify slots, fp32 state) works with MTP/EAGLE but not with DFlash2 (KDA-only). It is a lever for an MTP arm.
6. **FP8 KV scales:** the RedHat checkpoint ships 32 calibrated scales, but both engines log "defaulting to 1.0". Whether loading them helps accuracy is unverified.
7. **Swift-Qwen3.8-27b / Swift-1.5 / ThinkingCap** claim 37–58 % fewer thinking tokens. That would be equivalent to a 1.6–2.4× coverage gain if ARC accuracy holds.
   - Never researched: ARC accuracy, licence (Swift has a >US$1M revenue clause; ThinkingCap is gated, PolyForm) vs the prize's open-source rule, and DFlash2 acceptance on a fine-tuned target.
   - The 60k thinking floor (user rule) is a budget rule. A model that needs fewer tokens is a different lever, but the user decides.
8. **Per-step costs no one quantified:**
   - **lm_head:** vocab 248,320 × hidden 5,120. Verify logits for 96 positions per step means 96 × 248k logits, plus the drafter's lm_head.
   - **Sampling:** top-k 20 / top-p 0.95 over 248k per position.
   - **FR-Spec:** Franzen used a 64k hot-token map for the drafter's vocab.
9. **Newer engine versions:**
   - SGLang after 0.5.21: the #44401 11-of-12 admission fix; DFlash fixes for #36548, #38009 and #43514?
   - vLLM 0.31+: DFlash2 merged? async scheduling; all-reduce options.
   - No one checked changelogs or offline wheel feasibility for our image. The user prefers the newest stack ([[prefer-latest-stack]]).
10. **Load path facts still missing:** local disk size and `/dev/shm` cap on the 4×L4 image; whether SGLang 0.5.21 has `--weight-loader-prefetch-checkpoints`; JIT/CUDA-graph cache shipping (would save ~6 min, but the #42917 stale-cache risk applies); GCSFUSE detection; and the pinned-image startup above.

## C. New research launched (5 agents, 2026-10-23), saved in the same notes folder

1. Agent-side effective throughput: prefix retention, HiCache, KV-resident admission, re-prefill, lpm / page-size / track-interval.
2. Token-efficient model variants and reasoning efficiency: Swift, ThinkingCap, effort levels, accuracy, licence, drafter compatibility.
3. Newer SGLang / vLLM versions: changelogs for our exact issues, and offline wheel feasibility.
4. Startup and environment: load path, local disk / shm, JIT-cache shipping, the pinned-image start latency, the ERROR-with-empty-log cause, quota accounting.
5. Per-step overheads beyond GEMM/NCCL: lm_head and logits at a 248k vocab, sampling cost, FR-Spec, drafter compute, host CPU on g2-standard-48.
