# Final audit (Claude): `Q27-XHIGH-v1_4_4_DUAL_ANSWER_EVAL_v2.ipynb` (sha256 `1f8d7ea4…`), 2026-10-12

## Verdict: GO for a GPU EVAL12 run. Hidden submission stays OFF (default) until GPU evidence exists.

## Base freeze (checked against `Q27-XHIGH-v1_4_4.ipynb`, sha256 `95995980…`, the LB-26.94 kernel)

- **Byte-identical cells:** 0 (markdown), 1, 3, 4, 5, 6.
- **Cell 7:** the original scheduler is byte-identical; the second-answer stage is **appended** after it (+14,785 characters).
- **Cell 2:** additive `SECOND_BRANCH_*` settings plus the policy.
- **Cell 8:** additive report fields plus offline policy counterfactuals.
- All outputs are cleared.

## What v2 changes over v1

- **Selection policy (`protect_supported`, the default).** The original identity or mirror fallback is kept only when that transformation reproduces **every** training pair; otherwise a novel valid grid replaces attempt_2. The policy never reads solutions.
  - `replace_any` and `keep_fallback` are available as EVAL ablations.
  - The report computes all three policies counterfactually.
- **Novelty.** A grid identical to the old attempt_2 is not counted as a replacement.
- **Records.** `records.jsonl` stays the primary event log. `records_final.json`, `second_branch_candidates.json` (every raw valid alternative, selected or not) and `eval_generated_dual_submission.json` are checkpointed after every candidate.
- **Error handling.** Per-candidate errors are isolated (`candidate_errors`).

## Tests (run by Claude)

| Test | Result |
|---|---|
| GPT legacy suite (8 categories) | PASS |
| GPT regression suite (23 assertions: F1–F4, freeze, error isolation, counterfactuals) | PASS |
| pyflakes over all code cells joined | no real errors (only unused imports, the same as the base, plus one unused `global`) |
| Experiment 1 harness `mixed` (real Qwen tokenizer, fake vLLM step engine) | PASS; primary 16/22 = base; `primary_unchanged_all` true; nothing lost |
| harness `drain` | PASS; second stage skipped (`already_past_second_answer_deadline`); primary unchanged |
| harness `rerun` (259 hidden units) | PASS; branch disabled |
| **Hidden-mode equivalence:** base v1.4.4 vs v2 on the same harness `rerun` | **`submission.json` identical (240 tasks); 259 records identical** (ignoring wall time) |
| harness with phase-3 support (second answer = the correct grid) | 22/22 generated; 6 novel and selected; 6 marginal hits; 0 lost. Counterfactuals: keep_fallback 16, replace_any 22, protect_supported 22. The final assert fails by design, because `eval_query_hits` is now the dual score. |

## Notes (non-blocking)

1. `protect_supported` only differs from `replace_any` when the input or mirror reproduces every training pair, which is rare. In practice it behaves like `replace_any`, with a sensible safety exception.
2. Natural-close units show `origin_mode` = `None` (v1.4.4 leaves `mode` empty for natural closes). This is cosmetic.
3. Expected real benefit is small: the second answer comes from the **same** reasoning. GPU EVAL12 will measure distinct grids, marginal hits and GPU time.
4. Push to a **new kernel slug** (for example `foysalemonshanto/q27-xhigh-dual`) with the base's inputs: model `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`, dataset `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`, NvidiaL4, pinned docker image `37c64f7d…`, internet off. **Never overwrite `q27-xhigh`.**
5. **Estimated GPU time:** setup about 13 min + primary about 64 min (run 7) + second stage about 5–20 min (unmeasured), about 1.5 h. The EVAL12 hard stop is at 155 min.
