# E8 R5 GPU run (PROXY48 commit run): outputs for a deep analysis (2026-10-13)

**Request:** please analyse this run deeply. Verify every number below against the files; do not trust this README.

## Run identity

| | |
|---|---|
| Kaggle | `foysalemonshanto/q27-xhigh-e8`, version 1, scriptVersionId **357913772**, Save & Run All (commit run) |
| Notebook | `notebook/Q27-XHIGH-v1_4_4_E8_PER_UNIT_R5_FINAL_CPU.ipynb`, sha256 `16c6c688…` (GPT R5, pushed unchanged) |
| Base | `context/BASE_Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb`: v1.4.4 (LB 26.94) + exact Codex ordering |
| Profile | PROXY48: 48 evaluation tasks / 56 test inputs, 3 h budget; admit 140 min, drain 147 min, hard stop 155 min (notebook clock) |
| Engine | Qwen3.8-27B-FP8, vLLM 0.19.0, TP4 on 4×L4, MTP×3, fp8 KV, prefix caching, max_model_len 81,920, max_num_seqs 16, max_num_batched_tokens 4,096, gpu_memory_utilization 0.90 |
| Budget | xhigh, 49,152 thinking + 8,192 answer, host-forced close |

## What happened (Claude's first reading; please verify)

1. **The primary stage ran normally until about 124 min:**
   - 16 active units, about 353 generated tok/s, 0 preemptions, KV about 67 %, MTP acceptance 0.663;
   - all 56 units were admitted, and the queue was empty at about 108 min of scheduler time.
2. **The scorer started for the first time at about 124 min (09:43 UTC),** right after unit `7b80bb43#0` finished and freed a slot.
   - Primary-first admission kept it waiting until then: `e8_primary_waiting_steps` = 32,726.
3. **The first scorer request (1,665 prompt tokens, `prompt_logprobs=1`) caused `torch.OutOfMemoryError` on all 4 GPUs.**
   - The error is in `LogitsProcessor._gather_logits` → `tensor_model_parallel_all_gather`: "Tried to allocate 790.00 MiB … 417 MiB free".
   - 790 MiB ≈ 1,665 tokens × 248,077 vocab × 2 bytes, i.e. the full-vocabulary bf16 logits all-gathered for every prompt position.
   - **EngineCore died** (`EngineDeadError`), and the scheduler error handler kept the finished units.
4. **Result:**
   - 41 of 56 units finished; **15 in-flight units were lost**, about 31 minutes before the drain;
   - eval score **10 / 56 (17.86 %)**, and the E8 final is the same as the baseline: **net gain 0**;
   - modes: 15 natural, 26 forced_close.
5. **E8 stage counters** (`q27x/e8_per_unit_report.json`):
   - 19 units with a novel candidate; 9 with exactly one, which was applied directly (`attempt2_replaced` 9);
   - 0 units ranked; 10 incomplete (queued for 8-view scoring when the engine died);
   - `oracle_new_hits` 0: no mined candidate was correct where the baseline was wrong;
   - 52 context-accepted and 385 context-rejected grids.
6. **The submission is valid** (eval mode writes the identity test submission). `run_report.json` and the bundle were written.

## Questions for the analysis

1. Confirm or refute the OOM root cause and the memory arithmetic. What, if anything, makes `prompt_logprobs` safe on this engine (per-request prefill cap, a separate engine, a different scoring API, CPU-side scoring), and at what cost?
2. **Candidate mining value:** with 0 oracle-new hits on 41 units (and 0/88 in our earlier EVAL12 tail analysis), is thinking-derived candidate mining worth continuing? Use `e8_full_thinking/` (the full thinking text of every finished unit), `answers/` and `records.jsonl`.
3. **The 9 direct one-novel replacements:** were any of them plausible, or harmful (none can hurt pass@2 here, because the baseline attempt_2 never hits)?
4. **Primary-stage quality versus earlier runs:**
   - natural 15 / forced 26; hits by tier and category (EDA tags are in `context/ledger_all120.csv`);
   - compare with ARC Prize's official Qwen3.8-27B per-task results, also in that CSV (`arcprize_*` columns).
5. **Ordering:** did Codex easy-first help or hurt here, given that the NVARC fine score does not predict Qwen solves (AUC 0.435 on 120 tasks)?
6. What should the next experiment be, under these constraints:
   - at most about 1 Kaggle submission per 2 days;
   - the user's rules: no thinking budget below 49,152 + 8,192, keep v1.4.4 / E7 ordering as the base, and keep changes simple.

## Files

| Path | What |
|---|---|
| `notebook/` | the exact notebook that ran |
| `outputs/q27-xhigh-e8.log` | the raw Kaggle log (JSON lines) |
| `outputs/q27-xhigh-e8_stdout.txt` | the same log flattened to text |
| `outputs/q27x/records.jsonl` | the 41 finished units (mode, think tokens, score, tier, admit/finish minutes) |
| `outputs/q27x/run_report.json` | run report (scheduler, engine, metrics, E8 summary) |
| `outputs/q27x/e8_per_unit_report.json` | E8 rows per unit (candidate sources, choice, oracle flags, errors) |
| `outputs/q27x/e8_primary_baseline.jsonl` | the baseline attempts checkpoint |
| `outputs/q27x/e8_full_thinking/` | the full thinking text per finished unit |
| `outputs/q27x/answers/` | answer text plus the 1,500-token thinking tail per unit |
| `outputs/q27x/vllm_metrics.json`, `outputs/q27x/vllm_frontend.log` | vLLM metrics snapshots and the engine log |
| `outputs/q27x/e8_eval_final_generated_submission.json`, `outputs/submission.json` | the eval pairs used for scoring, and the (identity) test submission |
| `context/` | the base notebook, Claude's pre-run R5 check (it predicted this OOM risk), the per-puzzle capability ledger (EDA categories + ARC Prize per-task results) |
