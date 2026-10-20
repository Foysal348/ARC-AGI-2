# Q27 E8 R5 — Final CPU-hardened Per-unit Experiment

**Active notebook:** `Q27-XHIGH-v1_4_4_E8_PER_UNIT_R5_FINAL_CPU.ipynb`

## Start here
This is a CPU-hardened, **GPU-untested** experiment notebook, not a certified Kaggle submission. Run a short real-engine 4×L4 smoke test first. Original E7, donor, and R3 reference are included inside `sources/` and are **not overwritten**.

## Approved architecture — unchanged
E7 original primary inference -> per-unit checkpoint and full thinking capture -> contextual candidate mining -> 0 novel: fallback / 1 novel: direct candidate unless existing fallback is train-supported / 2+ novel: Codex 8-view augmentation + Q27 teacher-forced NLL -> protected attempt_1/attempt_2 -> validated atomic submission and baseline-vs-E8 reports.

No new model, per-puzzle LoRA, donor Torch DFS or direct-grid multisampling. No 30-minute reservation. E7 admission/drain/hard-stop absolute offsets, original model, prompt, ordering, MTP3, FP8-KV, 49,152 thinking and 8,192 answer allowances are unchanged. E7 **scheduler logic is deliberately modified for E8 hooks and routing**; GPU contention may still reduce primary throughput.

## R4 -> R5 changes (CPU-verified)
1. **R4-F1 Context:** Output-hypothesis cues required. Scratch, demonstration, training, intermediate, rejected and unlabelled grids are not promoted. Remaining errors are still possible because this is a text heuristic and not a correctness proof. Explicitly train-supported identity/mirror fallback is also protected.
2. **R4-F2 Ordering:** Store original character offset and sort newest-first across fence/row/JSON extractors. Candidate deduplication preserves first valid occurrence.
3. **R4-F3 Admission:** Primary queue gets free global concurrency slots before any scorer request, with scorer progress when capacity exists. `e8_primary_waiting_steps` records contention opportunities.
4. **R4-F4 GPU memory:** NOT fixed or claimed safe on CPU. One score request in flight, per-request token-length telemetry and full explicit GPU smoke-test gate. **No unmeasured `long_prefill_token_threshold` or original engine setting altered.** The vLLM full-vocab `prompt_logprobs` temporary-buffer OOM risk remains.
5. **R4-F5 Token alignment:** Record scorer token sequence and refuse score when joint/separate tokenizations or round-trip decode cannot be verified. This intentionally sacrifices some scorer coverage rather than silently assigning unreliable NLL. The forced-token donor alignment itself still needs real tokenizer/API verification.
6. **R4-F6 Cancellation:** Unique pending unit set across active/current/queued; idempotent abort and accounting, tested for 1 active + 2 queued and repeated finalization.
7. **R4-F7 Reports:** E8 EVAL hit metrics computed before auxiliary report; main `run_report.json` embedded E8 summary refreshed after final scoring.

## Source/diff
- `SOURCE_COMPARISON.md` / `SOURCE_AUDIT.json`: SHA and donor AST audit.
- `E7_SCHEDULER_TO_R5.patch`, `R4_TO_R5_HELPERS.patch`, `R4_TO_R5_REPORT.patch`: exact deltas.
- `build_r5_final.py` reconstructs the updated notebook using `sources/Q27-XHIGH-v1_4_4_E8_PER_UNIT_R4.ipynb`, with all inputs **inside the ZIP**. The updated .ipynb itself is standalone for Kaggle.

## Reproducible CPU tests
Run from this folder:

```bash
python test_r5_cpu.py
python test_r5_faults.py
python test_r5_adversarial.py
python source_audit_r5.py
```

`CPU_VALIDATION_OUTPUT.txt` stores the output. All targeted tests passed here. The tests use a fake tokenizer and fake engine for completion/inference routing. No actual 4×L4 model load, real prompt-logprob scoring, Kaggle leaderboard run or candidate-oracle improvement was demonstrated.

## GPU smoke-test gates (before any full run)
See `GPU_SMOKE_CHECKLIST.md`. Particularly memory peak/VRAM free, alignment failure rate, true scoring throughput, scored-vs-skipped view counts, primary completion delta, oracle gained/lost hits. Stop if `prompt_logprobs` OOMs, corrupt/missing chosen-token likelihoods, repeated scorer request failures, or primary coverage regresses significantly.
