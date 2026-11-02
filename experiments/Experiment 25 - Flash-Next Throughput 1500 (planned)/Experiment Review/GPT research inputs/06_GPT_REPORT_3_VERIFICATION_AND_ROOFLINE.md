# GPT research report 3: verification + the lead's roofline analysis (2026-10-31)

## 1. Claim check

| Claim | Verdict |
|---|---|
| Capacity table with F = 7 and "Merlin's 248-block profile" | SUPERSEDED: F = 6 (RC2 analysis); the 4.1e9 / 924,993 profile is the RTX 4090 estimate (see 05 verification); L4 headroom gives ~215-220 blocks at +1.0-1.1 GiB pin |
| T = N × A / t "full iteration" (Route A 8 × 4 / 21.3 ms; Route B 16 × 5 / 53.3 ms) | Formula has no PP2 factor. Under PP2 t = cycle = 2 ticks; RC2 tick fit 24.2 + 0.1758·D(4r) ms |
| Capture grid: 8 requests × 4 = 32 fits | Under PP2 a micro-batch holds C/2 requests, so 32 captured tokens cover up to 16 streams at K3; 24 streams need 48, 32 streams need 64 |
| PR #60532 FULL verify (P1) | Not applicable to Flash-Next (QSA/GDN already FULL in RC2); see 05 |
| PR #60234 missing → HIGH | Low risk (RC2 PP2 + PLE + MTP ran); see 05 |
| Forced LL128 corruption warning | Agrees with our NB3 decision (dropped) |
| INT4 draft head hurts acceptance (Saren-Arterius) | Consistent with NB2's choice of int8 (not int4); NOTED |
| Issue #62383 MTP + graph illegal memory access > 262K prompt | Out of our range (max-model-len 131,072); NOTED |
| llama.cpp-multigpu no-P2P: 5 × 250K sessions 27.3 tok/s each (≈ 136 aggregate) | External, UNVERIFIED; even as stated it is a long-context memory path, not a throughput path |
| NInfer Q2 2-GPU, Mirai S-small (512 experts, 1.85-bit avg, 36.5 GiB GPU weights incl. MTP, vLLM 0.31 plugin, B200 single-GPU), GSQ-RCO Q2 37.6 GB + PLE 28.8 GB, EXL3 1.66 bpw ~50 GB | External, UNVERIFIED on our stack; relevant to the low-bit branch (C6) — see the roofline below for why it matters |
| No public 4×L4 Flash-Next ≥ 1500 evidence | Matches all our mining |

## 2. Calibration: RC2 is at Merlin's own bandwidth-normalised level

Merlin's best long-context reference: six 127k sessions 574.5 tok/s on 4×RTX 3090 with P2P (936 GB/s per card).
L4 = 300 GB/s per card (ratio 3.12) → 574.5 / 3.12 ≈ 184 tok/s. RC2 measured 160 (116k@4) and 190 (40k@5) without P2P.
So RC2 is not mis-tuned relative to the engine's own reference: the remaining gap to 1500 must come from structural
changes (more tokens per weight byte, fewer weight bytes, a cheaper fixed floor), not from one missing flag.

## 3. MoE weight-bandwidth roofline (est., MoE term only, no fixed floor)

Assumptions: D(T) = 512(1 − (1 − 10/512)^T) distinct experts per layer, 2.53 MB per INT4 g128 expert, 48 layers,
K3 with a = 3.05 accepted tokens per request per cycle (RC2), 4 cards. Under PP2 each cycle has two micro-batches of
C/2 requests (T = 2C tokens each), and **each micro-batch re-reads its stage's experts** (two reads per cycle).
A single-batch layout (one batch of C requests, T = 4C) reads them once. Bits scale = bytes relative to INT4.

Ceiling tok/s at 250 GB/s per card (realistic; RC2's implied rate is 173 GB/s, L4 peak 300):

| Streams C | PP2 INT4 (today) | single-batch INT4 | PP2 3-bit | PP2 2-bit |
|---|---|---|---|---|
| 8 | 725 | 838 | 967 | 1,450 |
| 16 | 838 | 1,095 | 1,118 | 1,677 |
| 24 | 962 | 1,386 | 1,282 | 1,924 |
| 32 | 1,095 | 1,706 | 1,459 | 2,189 |
| 48 | 1,386 | 2,409 | 1,848 | 2,772 |

At 173 GB/s (RC2's implied MoE rate) every number is × 0.69; at 300 GB/s × 1.2.

Measured fit (RC2, tick = 24.21 + 0.1758·D(4r) ms, r = C/2, cycle = 2 ticks): 251 tok/s at C 8, 368 at 16, 462 at 24,
550 at 32. The fixed floor (~24 ms per tick) is about half of today's 48.6 ms tick at C 8.

Required tick for 1500 at a = 3.05: 16.3 ms at C 16, 24.4 ms at C 24, 32.5 ms at C 32 (today's fit: 66.3 / 79.3 / 88.7).

## 4. What 1500 needs (from the roofline; all est.)

1. **Both terms must shrink:** the ~24 ms fixed floor (dense GEMMs, BF16 vocab/draft heads, all-reduces, launch, PP
   hand-off, eager RecoverSSM commit) and the MoE term (173 → ≥ 250 GB/s kernel efficiency; NB3 D5 measures it).
2. **More tokens per expert read:** with INT4 under PP2 the MoE ceiling alone reaches 1500 only near C ≈ 48 at 250 GB/s.
   Levers that raise tokens per read: more resident streams (KV pin headroom, F = 6, live-block admission C2),
   higher acceptance per verify token (K tuning, a better drafter), and a layout without the PP2 double read.
3. **Fewer bytes per token** (C6 low-bit experts) moves the whole table: 3-bit ≈ × 1.33, 2-bit ≈ × 2 — accuracy-gated.
4. **The PP2 double read is a new structural item:** two micro-batches per cycle each touch D(2C) experts, vs D(4C) for
   one batch. At C 32 that is 1,095 vs 1,706 (MoE-only, 250 GB/s). Any layout that keeps PP2's KV capacity but merges the
   micro-batches' expert reads (or a TP/EP layout whose KV cost is solved) is worth a design study. Measure first: NB1's
   profiler window must report MoE time per micro-batch and distinct experts per micro-batch on real ARC routing.

## 5. Actions taken
- NB1/NB2: capture sizes must cover the per-micro-batch token count (C/2)(1+K) when seqs > 8 (24 streams → 48, 32 → 64).
- NB1 profiler: MoE ms and distinct experts per micro-batch per stage, real routing.
- PLAN §1 gets this roofline (v3.4).
