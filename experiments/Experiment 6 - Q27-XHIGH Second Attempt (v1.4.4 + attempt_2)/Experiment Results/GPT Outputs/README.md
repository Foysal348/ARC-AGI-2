# Reviews of the dual-answer v2 GPU result (2026-10-12)

The user received two independent AI reviews of `Q27_DUAL_v2_GPU_EVAL12_RESULT_FOR_GPT_2026-10-12.zip`. The full texts were pasted into the chat on 2026-10-12.

## Common verdicts

- **Implementation:** dual v2 integrity PASS.
- **Second attempt:** the same-reasoning second attempt is **NO-GO** for the hidden run.
- **3/22 → 1/22:** run-to-run variance, not a regression; the cause is unproven.
- **Independent full reasoning:** NOT established as a gain over the best single run (run 7 ∪ dual v2 = 3 = run 7 alone). Deferred to a separate experiment.
- **Experiment 7:**
  - base = dual v2 with `Q27X_SECOND_BRANCH=0` set explicitly;
  - exact Codex/NVARC ordering;
  - engine cap 20 with 16 active as a probe, then 18 active; 20 is conditional; **no direct 24**;
  - the KV controller is conditional; the completed-only p90 and online least-squares design is rejected as an initial deployment controller.
- **Add:**
  - read-only answer diagnostics (ragged widths, > 30 rows or columns, repetitive rows, length exhaustion), with no silent repair;
  - stage-specific prefix-cache and KV metrics;
  - an explicit `oversized_prompt` status instead of the latent `phase1_cap` budget reduction.
- **Measurement:** PROXY-48 (deadline-constrained) for the ordering and concurrency comparisons. EVAL12 cannot show an ordering gain, because all 22 inputs finish.

## Claude's checks of their factual claims (all confirmed)

| Claim | Result |
|---|---|
| 20a9e565#0 second answer is repetitive | 2,048 rows, 2,047 of width 2 ("4 4 / 6 6 …") |
| Total prefix tokens in the second stage | 1,110,400 |
| Prefix-cache hit rate during the second stage | about 80 % at the start, falling to about 7–9 %, so prefixes were mostly recomputed |
| `vllm_metrics.json` covers only the primary stage | 31 snapshots, the last at 60.1 min, then "final" |
| Codex `get_puzzle_priority` | identical to the 32.22 reference and to `puzzle_order.py` |
