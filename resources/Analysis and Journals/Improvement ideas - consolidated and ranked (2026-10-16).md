# Improvement ideas: consolidated and ranked (2026-10-16)

**Sources:**
- 4 research agents covering every past transcript (13 slices), the reference notebooks, our notebook code and the thinking traces;
- 2 GPT audits, checked against our data;
- the cross-run analysis in `Qwen3.8-27B LB and run deep analysis (2026-10-16).md`.

Agent outputs are in `scratchpad/agent_ideas_{transcripts,refs,code,traces}/`.

**Premise (user):** hidden is about as hard as eval, so finishing about 240 tasks in 12 h is the problem.
- Public LB = 120 tasks.
- On an eval-like set, 240 tasks means about 344 test inputs (1.43 per task).
- The placeholder file (259 inputs, training-like) is not the hidden set.

## A. Evidence that changed priorities today

| Finding | Number | Effect on the plan |
|---|---|---|
| **NVARC and Qwen solve different tasks** (team's NVARC notebook, LB 27.4–32.6) | EVAL120: NVARC 27.92 %. Unions: NVARC ∪ Qwen 49k direct = 43.89 % (13 units in common); NVARC ∪ R5 = 55.97 %; all three = 61.81 %. NVARC takes 4.6 h wall for 120 tasks and needs 17 GB per L4, so it cannot share GPUs | Biggest union lever, but **the user rejected NVARC as the path on 09-18**. The user decides |
| **One direct request per multi-test task** | Duplicated thinking is 2.97–3.23 M tokens = **28–31 % of ORDER2**. Poetiq (one request per task) on the same 49 tasks got 38 vs 26 inputs with fewer tokens | User-approved earlier ("একটা ভুল মানে তো সব ভুল", one wrong means all wrong anyway); never built. **High** |
| R5 attempt_2 | Duplicate in **134 / 172** units | Fixes plus a real second method |
| R5 per-query loss (one test input crashes, so the whole task is lost) | 5 units, including 2 train-verified tasks | Fix (F3) |
| Direct grid in R5's free attempt_2 slots | 48.61 → **53.06 %** (+9 units) | Poetiq → direct routing |
| Poetiq iteration-1 order on the hidden-size clock | Simulation: alphabetical 26.7 % → ORDER2 31.2 % (**+4.5**) | 3-line change |
| R5 prefix-cache recompute stalls | 28 stalls, about 38 min lost; vLLM hit rate 0.3–8 % vs about 80 % for direct | KV admission gate |
| Setup time | About 12 min, of which about 6 min is cacheable compile / JIT | Ship the caches as a dataset |
| `MAX_NUM_SEQS=16` | A setting, not a KV limit: KV median 0.6 at 49k, 0.69 at 86k, 0.77 Poetiq | 18–20 sequences with a KV gate; +4–10 % throughput |
| MTP acceptance | Direct T 0.6: 3.01 tokens/step; Poetiq T 1.0: 2.67 (349 vs 298 tok/s) | Temperature A/B; MTP k=4 A/B |
| One exception ends the whole scheduler loop | Code audit; R5 lost 5 units to one input | Isolate exceptions per output (robustness) |
| Early exit on train pass (code during thinking) | Saves only **0.33 M (2.9 %)**. Final programs appear at a median 91 % of the thinking; upper bound 12 % | **Downgraded** |
| Digit hand-reading | Digit-heavy lines are only 6.4 % of R5 thinking | Fact sheet loses its token-saving rationale. **Downgraded** |
| Code written by 30k | With code: 81 % close naturally, 77 % solved. Without: 29 % / 30 %. Past 49k: 50.7 vs 7.7 hits per M tokens | Protect code-writing units from preemption (allowed). Capping weak ones at 49k = rejected type |
| "Stop when a grid repeats" / convergence stop | 0 of 22 stops correct; 0 of 28 traces converge | **Avoid** |

## B. Ranked plan (respecting the rules)

Rules: ≥ 49,152 thinking; one change at a time; simple code; user intent is fixed; latest stack.

**Tier 1: biggest expected LB gain**
1. **E11 throughput:** SGLang 0.5.21 + INT4 + DFlash2, more than 16 sequences. The target is ≥ 2–2.5× for full Poetiq coverage. GPU test tomorrow, with the E11-03 additions in Tier 3.
2. **One direct request per multi-test task:** saves 28–31 % of direct tokens. GPU A/B needed for quality.
3. **Poetiq iteration-1 in ORDER2 order:** +4.5 in the hidden-clock simulation. CPU-only change.
4. **Poetiq → direct routing for tasks that are not train-verified:**
   - offline 53.1–55.1 % vs 48.6 %;
   - needs Tier-1 speed;
   - direct per task (item 2) makes it cheaper.
5. **R5 correctness fixes:**
   - per-query outputs (F3);
   - distinct attempt_2;
   - last valid `transform` block (F1);
   - harvest verifications at the hard stop (F2);
   - isolate exceptions per output (C1).

**Tier 2: speed and robustness, no quality risk**

6. Compile / JIT cache dataset and weight prefetch: −4 to −5 min per run.
7. KV admission gate (≤ 0.85) and protection of the prefix cache: stops R5's 38-minute stalls.
8. 18–20 sequences behind the KV gate (direct). The user's own cap-24 KV controller is never-run and fits here.
9. Deadline-aware admission and a clock measured from container start.
10. Engine watchdog plus requeue (E8 engine death).
11. E7 `MAX_MODEL_LEN` guard, so long prompts never get < 49,152 thinking.

**Tier 3: A/B tests, one variable each**

12. Poetiq temperature 1.0 → 0.6: about +13–17 % speed; quality unknown.
13. MTP k = 4 at T 0.6.
14. Grid-format A/B: CSV or explicit (R×C).
15. Forced-close salvage: program from the cached prefix, checked on train. Pilot on 20–30 tasks.
16. Verified checkpoint at 49k: force a program, test it on train, resume thinking on fail. Nets about 0.35 M.
17. CPU checks and leave-one-out ranking on train-pass programs: 22 of 71 train-verified programs were wrong on test.
18. Shape guard with an answer-only rewrite: 29 of 114 misses had the wrong shape.
19. Fact sheet (downgraded).

**E11-03 additions before tomorrow** (GPT plus the code audit):
- pick the winner from the long 16×24k stage, not the 4k sweep; keep A eligible;
- a stage beyond 49k;
- longer greedy parity checks;
- a mixed-batch contamination check at 1 / 8 / 16 / 24 sequences;
- record second-request `cached_tokens`;
- `--mamba-radix-cache-strategy extra_buffer`, guarded by `--help`.

**Avoid:**
- direct cap at 49k for all units;
- early stop on a repeated grid or on convergence;
- suppressing digit reading;
- unrestricted extra experts or iterations at today's speed.

**User-rejected (do not re-propose as-is):**
- COVER-FIRST;
- a 49k cap on Poetiq;
- older stacks;
- NVARC as the path (09-18). A union is a separate question.
