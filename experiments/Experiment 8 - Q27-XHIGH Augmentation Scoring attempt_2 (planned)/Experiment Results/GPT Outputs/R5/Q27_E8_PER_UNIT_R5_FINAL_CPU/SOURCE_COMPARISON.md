# R5 source audit and donor parity

## E7 baseline -> R5
- E7 cells 0–6 (**initial markdown and cells 0–5**) source identical.
- E7 configuration, all admission/drain/hard-stop timestamps, primary model/sampling/49,152 +8,192 token allowances, ordering, seeding, MTP/FP8 settings **unchanged**.
- E7 scheduler cell 6 is intentionally patched: after `finish` it checkpoints full thinking, queues candidate scoring; output dispatch differentiates `e8|` score requests from `uid|phase` primary requests; both consume global concurrency. The scheduler is therefore **not byte-identical** and primary GPU throughput can still change.
- E7 final report is adapted to compute baseline E7 vs E8 final pass@2; mixed throughput projection is flagged invalid.

## Codex donor -> R5
- Exact AST match for donor `ArcDataset.augment`, `mod`, `mod_single`, `shuffle_ex`, `forward_mod`, `invert_mod`, `permute_mod`, `permute_rnd_all_`.
- `QwenFormatter` only type annotation/unused `AutoTokenizer` import adaptation.
- **Not copied:** donor 4B per-task LoRA, Torch `past_key_values` recursive `turbo_dfs`, donor beam/probmul metadata; those are not directly compatible with Q27's vLLM engine.
- Donor 8 views retained; Q27 scoring uses `prompt_logprobs` on the same vLLM engine, with one scoring request in flight.

## R4 -> R5
- Text-offset chronology across fenced, JSON and row-run candidates; conservative context filtering of scratch/example/incorrect grids.
- Original fallback is protected when identity/mirror operation is supported by every training pair; unknown hidden correctness is still unproven.
- Primary unit admission is processed before scorer admission at each scheduling step (deadlines unchanged; shared GPU interference remains).
- Score-token decode + joint boundary checks fail closed on unverifiable formats; this can reduce scoring coverage on real tokenizer.
- Pending cancellation counts unique active/current/queued units once.
- Auxiliary E8 report is written after EVAL final metrics; embedded E8 report refreshed.
- 0/1/2+ routing and donor eight-view architecture retained; no new model or speculative DFS added.

## Not validated
- Real 4xL4 vLLM0.19 / FP8-KV / MTP3 prompt-logprob API memory and throughput.
- Whether mined grids include any **additional correct** answer on real Q27 runs (full reasoning wasn't saved in prior artifacts).
- Primary pass@1/coverage under mixed GPU contention and pass@2 gain; require paired GPU tests.
- Normal Kaggle completion was not run; only CPU and fake-engine tests.
