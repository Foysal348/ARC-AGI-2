# E8 R2 — Source comparison and independent re-audit

## Updated vs frozen E7 Ordering Only

- **Identical source:** notebook cells 0, 1, 3, 4, 5, 6, and 7 (intro plus all E7 primary executable cells other than configuration).
- **Modified Cell 2 (configuration only):** primary deadline moves earlier to make an explicit 1,800-second E8 reserve. The 120-second finalisation margin leaves a 1,680-second post-primary scoring window by default. Primary model, prompt, ordering, sampling, 49,152 thinking, 8,192 answer, seeds, engine recipe, and primary scheduler logic are unchanged.
- **Original E7 report Cell 8 is replaced by Cell 11:** Stage 2 is executed before final validation, final scoring, reporting and ZIP. The EVAL diagnostic score is recalculated from the final selected pair; the baseline score is retained separately.
- **Added Cell 9:** Codex vendor loader. **Added Cell 10:** candidate builder, 8-view augmentation, teacher-forced Q27 scoring, selection and checkpoint/transaction logic.
- **Original base notebook is not modified**; the new notebook is a separate artifact.

## Updated vs Codex donor

- Donor `ArcDataset` class, `augment`, `mod`, `mod_single`, `shuffle_ex`, `permute_mod`, `permute_rnd_all_` have **identical AST** to the donor `arc_loader.py` code extracted from notebook Cell 3.
- Only loader changes: remove unused `AutoTokenizer` import and type annotation from `QwenFormatter.__init__`.
- Donor's Torch `calc_scores` is **not copied directly**: model logits/forward implementation is adapted to Q27's existing vLLM engine using `SamplingParams(prompt_logprobs=1)` and chosen-token NLL over precisely tracked answer token IDs.
- Donor per-puzzle LoRA/DFS candidate generation and Probmul/Kgmon ranking **are not part of this experiment**. E8 uses train-consistent deterministic candidates and independent normalized-NLL / raw-NLL ranking.

## F1–F7 fixes and validations

| ID | Fix | Verified CPU test |
|---|---|---|
| F1 | Proper `len(challenges[tid]['test'])` and strict task schema | 1,2,3-query valid/invalid submission |
| F2 | Retain all train-supported candidates before optional legacy fallback; no truncation of supported hypotheses | synthetic 90-degree rotation |
| F3 | Reserve E8 window by moving E7 primary-only deadlines earlier; Stage 2 uses `E8_STAGE_DEADLINE` | 3-hour and hidden 12-hour budget monotonicity |
| F4 | EVAL generated predictions and exact post-E8 score, baseline/net gain fields | controlled wrong/correct grid transition and integrated run |
| F5 | Explicit no-extra-special-token encoding, suffix/answer index parity, token-boundary diagnostics | chosen-token logprob and fake-engine batch |
| F6 | Cleanup includes failed request ID even after partial registration, abort failures escalate; checkpoint rollback | injected add/abort/submission-write failures |
| F7 | E8 diagnostic report write isolated from final submission | injected report I/O failure |
| Extra | Expected primary deadline hard-stop no longer disables secondary scoring on completed units | cutoff-with-active-aborts fake engine |

## Critical remaining evidence limitations

- The notebook has **not** been run on Kaggle 4×L4; Q27/vLLM 0.19.0/MTP3 `prompt_logprobs` compatibility, throughput and real GPU KV pressure remain unverified.
- The **1,800-second time allocation is an explicit experiment setting**, not a measured optimum, and changes primary coverage/deadlines in this E8 copy. This must be reported in comparisons with E7.
- Ranking may lose correct baseline fallback hits; the selection is experimentally active as requested. A positive net pass@2 benefit is **not** assumed.
- A huge number of augmented candidate views may exceed the allocated stage window; partial scoring does not invalidate the baseline checkpoint.
