# Our pipeline vs writeup 755321 (notnitsuj, LB 75.69): where we went wrong (2026-11-03)

Our numbers come from the E28 K100 6-h EVAL120 report (`e28_k100_report.json`) unless noted otherwise.

## Side by side

| Item | Theirs | Ours |
|---|---|---|
| Throughput on Kaggle | 1,087-1,120 tok/s | 419.5 (E28 MTP, real run); DFlash2 bench 763 at 20k@24 |
| Running sequences | 48 (2 DP engines × 24) | 24 (one TP2×PP2 engine) |
| Per-stream speed | ≈ 23 tok/s | ≈ 17.9 tok/s |
| KV capacity | 974k tokens per engine (FP8 KV, packed state, one BF16 GDN checkpoint: +49 % seqs) | pool 479-521 blocks for 24 seqs; 438 preemptions in E28 |
| Prompt work per generated token | 6.1 → 0.2 (server-side sessions) | 11.2 prompt tokens, of which 2.26 fresh prefill. At 0.0745 generated tokens per fresh prefill token (E25 Duck timeline report) that is ≈ 1.5 M tokens ≈ 17 % of generation lost |
| Collectives | GPU-driven host int8 inside CUDA graphs: +8.6 % | E23 host all-reduce 0.93× (a naive version); dropped |
| Draft | MTP K3, fine-tuned for accepted length (+6.1 %), int8 32k-vocab head (+19 %) | MTP K3 + I8 head (E28) → DFlash2 D3s (AL 2.8-3.2); 32k draft vocab prepared, never GPU-run |
| Model | pruned 320/512 (format-aware), GPTQ 4/3-bit, RL on scales (+7.0 pass@2) | klee100 3bpw, all experts, no RL |
| Harness | REPL, `ak` helper lib, 11 invariants, cap 98k / 24 calls, stop at first verified | Duck REPL, train-exact check only, cap 104k (B2 120k) |
| Sessions per task | ~3 + voting, fewest-sessions-first, warm start from the best unverified program | 1 (A), +1 if not pass (B), attempt 2 = 1 more Duck (E28 C) |
| Precision of a "pass" | verified 75 % right (0.73 precision on the shipped build) | accepted 100 → exact 64 = 64 % (E28 A; 104 train-full) |
| Answers | votes; tie-break by symmetry kept + no test-only code; attempt_2 = most common other grid | best-train; attempt_2 = second attempt |
| Direct answering | measured 45.9 vs REPL 70.9 at equal tokens: rejected | E29 / E33 add Direct (Flash-Next Direct 11/46 in the ledger) |
| 27B | REPL 33.75 on 40 tasks; decode 725 → 129 as contexts grow: rejected on Oct 27 | our main track until 2026-10-31 (27B best LB 34.58; our best LB overall 54+, user 2026-11-03) |
| Evaluation | equal-token comparisons, 120 × 2 copies, replica noise ≈ 2.8, same-sign halves rule | single runs, varying token totals; LB noise sd ≈ 1.4-1.5 |

## Where we went wrong (by size of the lost points, in our judgement)
1. **Throughput is accuracy, and we lose it in four places.**
   - **Concurrency:** 24 vs 48 sequences, and throughput ∝ running^0.95. The KV and state memory (FP8 KV, one GDN checkpoint, packed state) is what sets this.
   - **Re-prefill:** each Duck turn resends the conversation, about 17 % of generation lost. They keep sessions on the server.
   - **Collectives:** our host collectives were judged on a naive implementation (0.93×). Theirs are GPU-driven inside the graph: +8.6 %.
   - **Draft fine-tune:** never done.
2. **Breadth and voting instead of chains.**
   - They run about 3 independent sessions per task and vote over verified programs.
   - We spend the second-to-fifth tries in a strict chain with different methods (Duck re-run, Poetiq, Direct). Each chain item is long and late. The E33 sim shows attempt 2 starved at our speed.
3. **The precision filter.** Our "train pass" is right 64 % of the time. Theirs is right 75 %, from 11 invariants plus votes plus a symmetry and test-only-code tie-break. Verified-but-wrong is their largest loss bucket (38 %), and it is ours too (36 accepted-not-exact in E28 A).
4. **Direct and long caps.**
   - They measured that tokens spent in direct mode buy far less than REPL tokens on Flash-Next.
   - They capped sessions at 98k, and cap raises gave only +0.9 on the shipped build.
   - Our B2 at 120k and E (Direct 104k) are expensive per point. This is evidence for planning, not a verdict on E33.
5. **The 27B detour.** They dropped 27B on Oct 27 from one measurement. We optimized it until Oct 31.
6. **The model was never repaired for the agent loop.** Compression damage shows up only in REPL mode (format loss), and they fixed it with format-aware pruning and RL on the scales. We never measured klee100 against FP8 inside Duck at equal tokens.
7. **Evaluation discipline.** They compare at equal tokens; one "+14.5" was only +4.8. Several of our decisions were made from single runs with different token totals.

## What we already did right (same as theirs)
- The Duck-style REPL agent on Flash-Next. Their prompt and feedback format also came from the Duck harness.
- n-gram/PLE embedding in pinned host memory.
- Flag-engaged read-backs (our "verify real concurrency" rule).
- Temperature 1.0 for Flash-Next.
- A deadline drain.
- KV-aware admission.

## Our 54+ submission (user 2026-11-03)
- **The notebook:** `foysalemonshanto/q27-xhigh-e25-nb6-submit`, scriptVersionId 364349249.
- **Model:** Flash-Next W4A16 on Merlin (5 parts + the FP8-PLE overlay).
- **Engine:** FP8 KV, MTP K3, **9 workers**, no Q-defer. The GPU KV cache is 738k tokens.
- **Solver:** the E15B Duck B attempt policy.
- **Speed:** W4 Duck measured ≈ **225 tok/s** in NB6 (E28 README).
- **Against theirs:** 54+ at ≈ 225 tok/s and 9 streams, vs 75.69 at 1,087-1,120 tok/s and 48 streams.
  - That is about 4.9× the tokens per hour and 5.3× the concurrency for +21 LB points.
  - On their own curve, tokens pay with diminishing returns (pass@2 80.6 → 89.9 % from 80k to 320k tokens per input). So most of the gap is speed and concurrency, not the harness.
- **Our next speed points:**
  - klee100 + Q-defer at 24 workers: 419.5 tok/s in the real E28 run (EVAL120 6 h: 65.0 % task-norm).
  - DFlash2 D3s: 763 at 20k@24 (bench).
