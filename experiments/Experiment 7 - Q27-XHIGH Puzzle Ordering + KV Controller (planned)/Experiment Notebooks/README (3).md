# Q27-XHIGH v1.4.4 — Experiment 7: Exact Codex Puzzle Ordering + KV Admission

**Generated:** 2026-10-12  
**Notebook:** `Q27_XHIGH_v1_4_4_E7_CODEX_ORDERING_KV.ipynb`  
**Base:** Original Q27-XHIGH v1.4.4, frozen source SHA-256 `959959806170960c665249362fe74bfb5678708b0183976ea58e05535d5fd56e`  
**Donor:** Codex notebook `starter.py` `get_puzzle_priority()`, exact body SHA-256 `3f6536e80d445389da50ed61b3c4b80edc05ff25989aa1ecf7fdef0626196230`.

## Important

This notebook **has not been GPU-tested or submitted**. Only static/CPU/fake-engine checks are completed. Do not deploy an adaptive-capacity hidden submission until the engine20/active16 probe and fixed18/adaptive18 comparisons pass. Original v1.4.4 and Dual Answer v2 are unchanged.

The notebook contains the original v1.4.4 primary inference architecture; it **does not** include Dual Answer v2's extra answer branch. It does not change the original prompt, sampler, tokenizer, parser, MTP3, FP8 cache configuration, or 49,152 / 8,192 generation allowances. It does change request-admission failure handling transactionally, and by default enforces the full thinking allowance for long prompts.

## Quick configuration (edit top of CELL 0, after `import os` and before CELL 1)

Use these **one at a time** for a controlled comparison:

### E7A: Exact Codex ordering, original concurrency 16

```python
os.environ['Q27X_E7_ORDERING'] = 'codex'
os.environ['Q27X_E7_ADMISSION'] = 'baseline'
os.environ['Q27X_MAX_NUM_SEQS'] = '16'
```

### E7B: Engine20 / active16 (engine overhead probe)

```python
os.environ['Q27X_E7_ORDERING'] = 'original'
os.environ['Q27X_E7_ADMISSION'] = 'baseline'
os.environ['Q27X_MAX_NUM_SEQS'] = '20'
```

### Fixed18 comparator (requires approved GPU safety probe)

```python
os.environ['Q27X_E7_ORDERING'] = 'original'
os.environ['Q27X_E7_ADMISSION'] = 'fixed'
os.environ['Q27X_E7_EXTRA_CAP'] = '18'
os.environ['Q27X_MAX_NUM_SEQS'] = '20'
```

### E7KV: Adaptive up to 18 (experimental; GPU approval required)

```python
os.environ['Q27X_E7_ORDERING'] = 'original'  # 'codex' only in the combined comparison
os.environ['Q27X_E7_ADMISSION'] = 'adaptive'
os.environ['Q27X_E7_EXTRA_CAP'] = '18'
os.environ['Q27X_MAX_NUM_SEQS'] = '20'
```

The KV controller uses real `llm.get_metrics()` telemetry (`kv_cache_usage_perc`, running, waiting, preemptions). The initial soft threshold 0.62, projection limit 0.78, hard threshold 0.90 and 15-second interval are **unvalidated pilot parameters**, not guaranteed memory bounds. It can be entirely disabled by setting `Q27X_E7_ADMISSION=baseline`.

The two feature families are independent; apply `Q27X_E7_ORDERING=codex` to the best safely validated admission configuration only after separate A/B comparisons.

### Budget invariant

`Q27X_E7_STRICT_FULL_BUDGET=1` **default**: a prompt exceeding the configured context for the full 49,152-token thinking + 8,192-token answer cap is explicitly skipped with a valid fallback rather than silently shortening thinking. `Q27X_E7_STRICT_FULL_BUDGET=0` restores the historical v1.4.4 reduced-budget behavior for a carefully documented parity test **only**. Existing deadline-driven forced close remains unchanged and can still interrupt a running request early.

## Output artifacts in Kaggle

- `/kaggle/working/submission.json` — original required submission; always written by CELL 0 fallback and updated incrementally.
- `/kaggle/working/q27x/experiment7_ordering.json` — ordering metadata, donor hash, tier counts and actual task order.
- `/kaggle/working/q27x/experiment7_kv_report.json` — admission mode, measured metrics, counters, blocking/admission events.
- `/kaggle/working/q27x/vllm_metrics.json` — frozen base periodic engine metrics.
- `/kaggle/working/q27x/run_report.json` — frozen base results with `scheduler.experiment7` and `scheduler.kv_controller_counters`.

An adaptive run can silently fall back to baseline if the requested engine recipe or pool is unavailable. Such a run is **not valid evidence for adaptive admission benefit**. Inspect `scheduler.experiment7.effective_admission` in the report.

## Tests

```bash
python tests/test_exp7.py
```

Reported **10/10 CPU checks passing**, covering:

- Valid notebook and parsed code cells.
- Frozen base hash/source invariants.
- Exact Codex source parity; JSON stable tie ordering and EVAL filter.
- Real stored vLLM 0.19 metric sample and malformed/duplicate rejection.
- Controller baseline/fixed/adaptive logic, cooldown, stale/phase safety, capacity limits.
- Transactional Phase2 failure/rollback and EOS trim.
- Oversized context budget preflight.
- Deterministic fake-engine end-to-end run and injected request-add failure.
- Change allowlist: only Cells 1, 3, 5, 6 are changed.

**CPU-only tests cannot verify** vLLM startup at 20, CUDA graphs, GPU memory behaviour, prefix cache physical usage, no preemptions, model accuracy, or leaderboard score. They are independent GPU GO/NO-GO gates.

To regenerate the notebook byte-for-byte from bundled frozen sources:

```bash
python build_exp7.py
python tests/test_exp7.py
```

## Files

- `source/`: exact frozen base and full Codex reference notebook.
- `donor_get_puzzle_priority.py`: reference function copied verbatim from Codex `starter.py`.
- `kv_controller_source.py`: the exact controller source embedded in CELL 6.
- `build_exp7.py`: deterministic notebook builder.
- `SOURCE_DIFF.md`: complete code cell unified differences.
- `BASE_MANIFEST.json`: source and donor fingerprints.
- `CPU_VALIDATION_REPORT.json`: run verdict.
- `tests/dual_v2_metrics_fixture.json`: telemetry from supplied GPU EVAL12 result for parser regression only.

## Pending GPU validation (no runs performed)

1. Baseline engine16/active16.
2. Engine20/active16 with preferred FP8 KV+MTP3 recipe (setup overhead + KV pool).
3. Fixed18 with same workload: peak KV, preemptions, total completed, correctness, wall time.
4. Adaptive18: compare against fixed18, not just fixed16.
5. Conditional Codex+best-KV combined notebook. Hold 20+ active / 24 engine without new evidence.

**All GPU runs and submissions require the user's approval.**
