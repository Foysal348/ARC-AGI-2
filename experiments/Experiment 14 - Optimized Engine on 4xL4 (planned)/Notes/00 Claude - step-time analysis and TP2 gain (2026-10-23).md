# Step-time analysis from our server logs, and what a faster engine would buy (Claude, 2026-10-23)

This note records two answers I gave the user in chat on 2026-10-23. Every number below is my own calculation from our logs or an estimate, as marked. GPT's audit of this note is in note 01; my validation of that audit is in note 02.

## 1. Where does decode time go? (derived from logs, no GPU used)

**Formula:** step time = running × accept len / gen tok/s. The inputs are the SGLang TP0 "Decode batch" lines from E11-03 (sweep) and E11-06 (production), about 4,000 lines in total.

| Setup | 1 running | 12 running | Marginal cost per extra request |
|---|---|---|---|
| INT4, no speculation (1 token verified per request) | 21.9 ms | 37.9 ms (22 running: 50.0 ms) | about 1.3 ms |
| INT4 + MTP (about 4 tokens verified per request) | 40.3 ms | 79.9 ms (10 running) | about 4.4 ms |
| INT4 + DFlash2 (8 tokens verified per request) | 39.9 ms | 115.5 ms (E11-03 D); 115–133 ms by context (E11-06) | about 6.8 ms |

**What the table shows:**
- **Marginal cost.** Each extra token verified per step adds about 0.85–1.3 ms. My claim of "only 10–16 TFLOPS achieved" assumes all of that is GEMM; GPT correctly says this is not established (note 01, §4.1).
- **Context cost is bandwidth-shaped.** At 12 running, going from 0 to 50k per request adds about 18 ms per step. The KV-read arithmetic (12 × 50k × 8 KB ≈ 4.8 GB ÷ 300 GB/s) gives about 16 ms, so this part is not a bug.

## 2. Two suspects

**Suspect 1: TP4 all-reduce without P2P.**
- Our measured 64 MB NCCL all-reduce took 20.9 ms, about 3.1 GB/s algorithm bandwidth (E11-03 report).
- Under a naive bandwidth model (an estimate), 128 all-reduces per step × 10 KB per token works out to about 0.42 ms per verified token. That is about half the marginal cost.
- At 12 × 8 = 96 tokens per step it means about 40–50 ms of a 115 ms step.
- GPT notes that small messages are latency-bound, so this extrapolation is unverified (note 01, §4.3).

**Suspect 2: DFlash2 verifies 8 tokens and accepts about 4.3.**
- An HF discussion reports that positions 4–7 are rarely accepted.
- Under my model (an estimate), going from 8 to 4 draft tokens would cut the step to about 70 ms and might give 550–650 tok/s.
- GPT's geometric model gives L4 ≈ 3.0 and about 515 tok/s instead (note 01, §5).
- GPT also says the checkpoint is trained at block 8, and SGLang calls set_block_size() when the configured width differs.

## 3. If 2×TP2 were faster, what would we gain?

- The 2.4× figure comes from a 3090 run at 64 concurrent with short answers (TP4 307 vs 2×TP2 737). It does not transfer to our 12-stream, 60k-context regime.
- **Closer analog (biMEMO, 3090, Qwen3.8-27B INT4):**
  - c8: one TP2 518.6 vs one TP4 469.4 tok/s.
  - c4: one TP2 323, so two replicas give about 646 vs TP4 at c8 469, i.e. **about 1.38×**.
  - My estimate for us: 1.3–1.6×.
- **Budget arithmetic:** 11.3 h of solving, 240 tasks, W1 whole-run about 355 tok/s today.

  | Speed | Total tokens | Per task | Meaning |
  |---|---|---|---|
  | 1× | ~14.4 M | ~60k | one 60k attempt per task, just barely |
  | 1.3× | ~18.8 M | ~78k | one full 60k + part of a second attempt |
  | 1.6× | ~23 M | ~96k | 60k + ~36k second attempt |
  | 2.4× | ~34.7 M | ~144k | two full 60k attempts |

- **The 2.4× case vs rev 5:** rev 5 LOW used about 145k per task and solved about 56 % of the tasks it reached, but it reached only 31 % of them.
  - At 2.4×, all 240 tasks would get that budget, which suggests an LB near 50 if the quality held.
  - Caveat: ORDER2 picks easy tasks first, so the 56 % may be inflated.

**Costs of 2×TP2:**
- Two servers on separate ports; the Duck branch-to-server routing and a per-replica KV gate have to be built.
- Memory is tight: INT4 TP2 is about 9.3 GB of weights per GPU, and with DFlash2 a replica has about 0.35–0.4M KV, so about 5–6 long requests.
- Two weight reads under the GCSFUSE slowdown, unless the weights are copied locally first.
- Never measured on L4.
- It breaks the rev 6g "last version" rule, so it is the user's decision.

**Recommendation in chat:** one measurement run (topology, NCCL, profiler, draft width 8 vs 4, TP4 vs 2×TP2) before any code change. Build the 2-server version only if the gain is above 1.3×.
