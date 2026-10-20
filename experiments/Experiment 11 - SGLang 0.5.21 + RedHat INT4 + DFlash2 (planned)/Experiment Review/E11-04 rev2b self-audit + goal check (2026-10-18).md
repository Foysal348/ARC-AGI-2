# E11-04 rev 2b: Claude self-audit + goal check (2026-10-18)

## A. Self-audit of the notebook (verified against the SGLang 0.5.21 source in `_scratch/session_1c6db7aa/sgl_src`)

| # | Check | Result |
|---|---|---|
| 1 | `extra_buffer_lazy` with DFLASH | **OK.** `mamba_hook.py:114-120`: "eagle/ngram/dspark/dflash all verify through prepare_mamba_track_for_verify (lazy plan wired)". Overlap is on with DFLASH (E11-03 D server_info: `disable_overlap_schedule False`, logged 5 slots = overlap). `mamba_track_interval` 256 ≥ 8 draft tokens. |
| 2 | `--disable-prefill-cuda-graph` with DFLASH | **OK.** It is a generic switch to eager prefill (`cuda_graph_hook.py:67-68`); no speculative constraint found. |
| 3 | `#full token` as the context proof | **OK, conservative.** `num_used = max_total − (available + evictable)` (`pool_stats_observer.py:221-223`). Cached, evictable prefixes are excluded; a shared prefix is counted once. |
| 4 | Single point of failure: all arms share the lazy + no-prefill-graph flags; rev 2 retried only on OOM | **FIXED (rev 2b).** One diagnosed fallback per cause: OOM → mem 0.78; lazy rejected → extra_buffer + mamba cache 70 (14 requests); prefill-graph complaint → keep the graph. Causes are read from error lines only (normal startup lines also say "Mamba Cache"). Max 3 launches. New harness scenario `lazyfail`. |
| 5 | Long-prefix answer test could FAIL falsely: PROD's early long outputs may be evicted from the cache by later PROD requests | **FIXED.** It prefers responses that arrived after the window (most recent, still cached), then the longest. |
| 6 | W2's 30 min PROD may not reach ≥ 40k mean context for long enough (cycle 1 needs about 20–25 min) | **FIXED.** W2 window 35 min. |
| 7 | Marker companions had 512 tokens to write their codename | **FIXED.** Companions get 1,024 (greedy prompts stay at 512). |
| 8 | Slots per request were taken from the requested flags, not the launched ones | **FIXED.** `slots_per_request(cur_flags)`, and `launch_flags` is recorded. |
| 9 | PROD cap 60k while production is now 49,152 (S49) | **Kept on purpose.** It is harsher (more KV). Results are conservative for S49. |
| 10 | Throughput baseline (vLLM about 255 tok/s long context) is from Poetiq runs, not this workload | **Known limit.** A paired vLLM run on the same workload (E11-05) remains optional. |

**Harness (rev 2b):** main, shortctx, oomfail, capfail, contam, lazyfail. Results are in the chat log of 2026-10-18. Notebook sha after rev 2b: see the builder output.

## B. Are we drifting from the goal? (honest)

**Goal (user):** LB 60–80+ with Qwen3.8-27B on 4×L4 / 12 h. Top teams are 80–88; artificialagencylab went 30.97 → 63.06 in one day. **Ours: 34.58** (S1-60K), with S49 pending.

**The direction of E11 is right:**
- Coverage is the proven binding constraint: 77.8k → 60k gave 21.39 → 34.58.
- R5 at full eval coverage is 48.6 %, so full coverage of hidden should give about 45+.
- E11 (about 1.3×) plus S49's 49k cap is about what full coverage needs.

**But the ceiling of this line is about 45–50, not 60–80** (eval data, full coverage):

| Union | eval task-norm |
|---|---|
| Qwen Poetiq R5 | 48.6 % |
| Qwen any of our runs | 56.8 % |
| NVARC dual ∪ R5 | 56.0 % |
| **NVARC ∪ DIRECT ∪ Poetiq** | **61.8 %** |

- Reaching 60 needs about 3 different solvers on every task: about 3× today's compute, plus selection without the answers.
- Sequential NVARC + Qwen on the same 12 h simulates to only +1–2 (`lever_analysis/hybrid_sim.py`).

**Where time went (2026-10-15 → 18):**
- E11 engine validation: 4 notebooks, 4 audits.
- D66/L66: 2 × 12 h.
- Effort probes: M3H dropped, W3H run (low effort, negative).
- Lever analyses: early-stop, ordering, pause-resume (mostly negative).
- **Two LB submissions (34.58, S49)**, about 15 days before the Nov 2 deadline.

**Verdict:**
- The engine track is not drift, but it is slow and capped at about 45–50.
- The method gap (per-task accuracy 48 % → 80 %) has had **no active track for days**. That is the drift.

**Recommendations:**
1. Run E11-04 now; no more audit rounds. If it passes, port S49 to SGLang immediately as the next big submission (full coverage, about +10).
2. Open a method track in parallel, and test the cheap ones on eval first:
   - **Swift-Qwen3.8-27B** (a shorter-thinking finetune, −58 % thinking reported). Same family and size, so a drop-in that could double coverage if ARC accuracy holds.
   - A **real second attempt** with agreement selection once speed allows.
   - **NVARC + Qwen union** when compute allows (61.8 % eval ceiling with DIRECT + Poetiq).
3. Every submission bundles all evidence-backed changes (user rule 2026-10-18).
