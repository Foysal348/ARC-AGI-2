# S1-60K rerun-path audit (2026-10-27)

Notebook: `experiments/Experiment 10 - .../Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb`
(S1-60K, LB 34.58 / 31.94). Read-only audit. Cell numbers below are notebook cell indices 0-8 (cell 1 = "CELL 0", cell 2 = "CELL 1", ...); line numbers are within the cell source.

Evidence runs used (same scheduler code family, GPU, Kaggle 4xL4, vLLM 0.19):
- R5 EVAL120 (`GPU EVAL120 Poetiq R5 (hosen42 ...)`): 120 tasks, 77.8k, 172 requests, 281 tok/s, no drain units.
- D66 (`GPU HARD66 D66 77.8k (foysal ...)`): **same S1 scheduler code as S1-60K** (cell 7 diff = logging only), **same RERUN_ALL margins** (drain 480 s, admit 120 s, hard 120 s, budget 43,020 s). It is the only GPU run that exercised the RERUN_ALL drain with a full slot set.
- L66 (`GPU HARD66 L66 122.9k`): same margins, 12 slots.

There is no S1-60K run on the RERUN_ALL clock outside the two hidden LB runs (no S1-60K EVAL120 in Results).

## Bottom line

No hidden-only bug of the "tiny kernel issue" kind was found: the RERUN branch loads only `arc-agi_test_challenges.json`, uses no solutions, writes a valid full-schema submission at every commit, and the two scored LB runs prove the sandbox, engine and writer all work in the rerun. The LB is fully explained by (a) coverage: 240 hidden tasks vs about 205-225 iteration-1 requests in 12 h, no iteration 2 at all, and (b) a **drain window that is too short**: D66, with identical code and margins, lost 14 of 15 drained units (no code at the hard stop). Estimate: iteration-1 rate ~41 % (R5: 58.3 task-norm hits minus 9 from iteration 2, at 77.8k) x hidden coverage ~85-90 % = 33-36 %, which matches LB 34.58 / 31.94 (ESTIMATE).

## Findings

### F1 — HIGH — The 8-minute drain cannot finish; drained units are lost at the hard stop (GPU evidence on the identical code path)
- Code: cell 2 lines 18-21 (`drain_margin_sec 8*60`, `admit_margin_sec 2*60`, `hard_margin_sec 2*60`), lines 75-80 (DRAIN = HARD_STOP - 480 s = T0+42,420 s = 11:47; ADMIT = 11:45; HARD_STOP = 11:55). Cell 7 lines 270-276: at the drain every phase-1 request is aborted and re-added as a new phase-2 request with prompt = prompt + all thinking + STOP text (`e10_start_phase2`, cell 7 lines 98-110). That new request must re-prefill the whole prefix.
- Evidence (D66 `poetiq_vllm_metrics.json`, `poetiq_iterations.jsonl`): drain at 42,420 s with 15 active. From 42,370 s to 42,853 s, `prompt_tokens_by_source local_compute` rose 7,765,485 -> 8,374,754 (+609k, still rising at the hard stop, ~1.7k tok/s), `local_cache_hit` only +157k, and `generation_tokens` only +19.6k in 8 min (normal rate ~255 tok/s = ~120k). KV usage fell 0.88 -> 0.25 after the aborts (cache not reused). Result: 15 units drained, **14 ended `finish=hard_stop`, `has_python=False`** (p1 7k-64.5k tokens, p2 737-1,530 tokens each: explanation started, code block never closed); 1 (smallest, p1 10,970) finished. L66 (12 slots) got all 12 drained units through, so the loss depends on slots x prefix length.
- Hidden S1-60K: up to 16 units active at 11:47, prefix ~13k prompt + 0-60k thinking (avg ~30k) -> ~700k tokens to re-prefill at ~1.7k tok/s = ~7 min of the 8-min window, then 1.5-3k answer tokens at ~15 tok/s per sequence. Most drained units will produce no code (same as D66).
- Units admitted between ~10:50 and 11:45 are exactly these units; they are tail tasks of ORDER2 (bins 0/1).
- Expected LB impact (ESTIMATE): ~14-16 hidden tasks per run end as identity instead of a forced-close answer; at ~10-20 % solve rate for partial-thinking forced closes -> +0.5 to +1.5 LB points if fixed.
- Fix (any one; first two are config-level):
  1. Drain earlier: `drain_margin_sec` ~25 min (sum of active prefix lengths / 1.5k tok/s + 3k answer tokens / 15 tok/s, x2 safety). Or compute it dynamically each loop: drain when `now >= HARD_STOP - (sum(len(prompt)+len(p1)) / 1500 + 240) * 1.5`.
  2. Staggered drain: force-close the longest-thinking units first, 3-4 at a time, so their prefill does not starve every other decode.
  3. Best: avoid the abort/re-prefill entirely with an in-engine forced close (a vLLM v1 logits processor that emits STOP_IDS when the per-request budget or a global drain flag is hit). This also removes the in-run re-prefill cost of F2b. vLLM 0.19 API details UNVERIFIED.

### F2 — HIGH (structural, not a code bug) — Coverage: 240 hidden tasks do not fit; ~20-35 tasks get identity and no task gets iteration 2
- Code: cell 4 lines 12-15 (all test tasks), cell 7 lines 26-38 (ORDER2: `[>=3 train] + [1 test]`, ties by sorted id; one expert; failed units go to the back), cell 7 line 277 (admit until 11:45).
- Evidence: R5 EVAL120 at 77.8k: 172 requests, 11.65M generated tokens, 281 tok/s, every one of 120 tasks got iteration 1, 52 got iteration 2; iteration 2 added ~9 task-norm hits (16 tasks train-passed only at iteration 2) = 7.5 points of eval score. Hidden = 240 tasks in the same 12 h. At 60k cap (mean request ~55k generated tokens vs ~67.7k in R5) -> ~205-225 iteration-1 requests (ESTIMATE) -> ~15-35 tasks never admitted (identity, attempt_1 = input) plus ~16 drained (F1). Never-admitted tasks are ORDER2 bin 0 (<3 train and >1 test; 15/120 eval tasks, R5 task-norm 0.33) then the tail of bin 1 (0.42).
- Expected LB impact (ESTIMATE): never-admitted tasks ~2-3 LB points; missing iteration 2 another ~3-7 points vs the eval number. This is the main eval-vs-LB gap, not a hidden-only bug.
- Fix: throughput levers (F2b, F1.3), higher effective concurrency, and starting units so that the last ones still complete; no thinking cut (60k floor rule).

### F2b — MEDIUM — Every in-run forced close re-prefills its thinking (engine stall)
- Code: cell 7 lines 309-314 (phase 1 ends at `length` -> `e10_start_phase2` new request with prompt+p1).
- Evidence: R5 104/172 requests were forced closes; `prompt_tokens` 10.04M = 1.37M fresh prompts + 8.67M phase-2 prefixes; local compute 4.18M, cache hits 5.85M -> ~2.8M re-prefill tokens computed, at ~1.7k tok/s with decode almost stalled (D66 tail) = ~27 min of the run (~4 %). D66: 8.37M computed vs 2.81M cached.
- Expected LB impact (ESTIMATE): ~8-10 more hidden tasks admitted -> ~+1 LB point.
- Fix: in-engine forced close (F1.3); same patch.

### F3 — LOW (catastrophic tail, no evidence it happened) — Exception isolation gaps can stop the whole solver
- Isolated (C1): only the per-output body, cell 7 lines 292-328.
- Not isolated: `e10_admit` (cell 7 line 281: `apply_chat_template`, `tokenizer.encode`, `engine.add_request`), `e10_poll_verifications` -> `e10_commit_iteration` -> `e10_update_task_submission` -> `e10_write_submission` (line 263; includes the jsonl append, cell 7 line 177), the drain and hard-stop blocks (lines 264-276), `engine.step()` (line 291). Any raise there exits `e10_run_solver`; cell 7 lines 357-363 abort all active requests; the `finally` writes what exists; every remaining task stays identity for the rest of the 12 h. No engine restart.
- Evidence it has not happened: `unit_errors` = [] and `error` = None in R5, D66, L66; two LB runs scored >30.
- Expected LB impact: 0 in normal runs; up to the full score if it fires early.
- Fix: wrap the admission body per key (on error: mark the unit finished, log, continue), wrap the per-future commit in `e10_poll_verifications`, wrap the drain/hard-stop per-request calls; on `engine.step()` failure, try one engine re-create if time allows.

### F4 — LOW — An exception in the `finally` loop would fail the notebook (Kaggle submission error, no score)
- Code: cell 7 lines 364-366 `for _tid in TASK_IDS: e10_update_task_submission(_tid)` is not wrapped. If the same deterministic error that stopped the solver (F3) comes from ranking/writing, it is raised again here, cell 7 raises, Run All stops and Kaggle marks the rerun as failed even though `submission.json` exists from the last good write. Cell 8 is wrapped (lines 2, 40-41); cells 1-6 raise only on environment faults that would also break the commit.
- Fix: `try/except` per task inside the finally, plus a final `try: e10_write_submission() except: pass`.

### F5 — LOW — Verifications running at the hard stop are discarded
- Code: cell 7 lines 339-346. After the hard stop the harvest waits at most `POETIQ_SANDBOX_TIMEOUT*2` = 10 s; `pool.shutdown(wait=False, cancel_futures=True)` cancels queued ones; leaving the `with` block then waits for the <=4 running ones (worst case (n_train + n_test) x (5 s + 2 s kill), ~70 s) but their results are never committed (no poll after shutdown).
- Time check: hard stop 11:55:00 + 10 s + <=70 s + finally (~240 small writes, seconds) + cell 8 -> done by ~11:56:30 worst case vs budget end 11:57 and the 12:00 limit. No timeout risk.
- Expected LB impact (ESTIMATE): ~0-0.2 points (only units that complete code in the last seconds; with F1 most have none).
- Fix: after the with-block, call `e10_poll_verifications()` once more (completed futures are still readable), and harvest until `T0+TOTAL_BUDGET_SEC-60` instead of 10 s.

### F6 — LOW — KV admission gate never blocks; fail-open path itself is harmless
- Code: cell 7 lines 237-254; cell 2 lines 118-119.
- The metric exists in vLLM 0.19 (`vllm:kv_cache_usage_perc`, Gauge; seen in R5 metrics), so the fail-open (exception -> 0.0) branch is not normally taken; if it is, behaviour equals R5 (no gate).
- But the gate is structurally ineffective: it is checked only right after a unit frees its blocks, when usage is low. D66: `kv_gate_blocks` = 0 with 66 preemptions; L66: 0 blocks. R5 (no gate): 9 preemptions. Each vLLM v1 preemption recomputes the sequence.
- Expected LB impact (ESTIMATE): small at 60k (shorter sequences than D66); <0.5 point.
- Fix: gate on projected KV (sum over active of prompt + cap) instead of current usage, or cap slots at 12-14 for the last hours.

### F7 — INFO — attempt_2 is effectively identity on the hidden run
- Code: cell 5 lines 246-265 (A2 rule) is correct: never repeats attempt_1; falls back to the test input, then its mirror.
- On the hidden run almost every task has exactly one program (one expert, no iteration 2, F2), so `grids` has one element and attempt_2 = test input (or mirror). attempt_2 contributes ~0 on LB, whereas in eval R5 iteration-2 programs fed it.
- Lever, not bug: any cheap second candidate (e.g. second grid from the same program on augmented input, or the best non-identical program from the forced-close answer) would use the slot.

### F8 — INFO — Admission floor and context skip
- Cell 7 lines 74-88: the floor is hard-coded 49,152, not the 60,000 user floor; prompts > 37.8k tokens are admitted with < 60k thinking. A first-iteration prompt > 48,896 tokens is `insufficient_context` forever -> identity. Max R5 eval prompt = 23,687 tokens, `skipped_context` = 0 in R5/D66/L66, so practically inert; hidden max prompt UNVERIFIED.
- Fix (policy): use `THINK_BUDGET` (60,000) as the floor, and for an oversized first prompt drop shuffled examples rather than skip.

### F9 — INFO (verified correct) — Task loading, order, ids, multi-test, schema, writes
- RERUN detection: cell 1 line 17 (`KAGGLE_IS_COMPETITION_RERUN` in {1,true,yes}); PROFILE = RERUN_ALL (cell 2 line 2). The LB scores prove it was set.
- Data: cell 4 lines 12-15: `CHALLENGES = dict(test_challenges)`, `TASK_IDS = sorted(...)`, `SOLUTIONS = None`. No evaluation file or solution is read on the rerun path; `poetiq_verify_code` (cell 5 lines 172-214) never consults test truth; cell 8 scoring is skipped when `SOLUTIONS is None`.
- ORDER2 (cell 7 lines 26-30) uses only train/test counts of the hidden tasks; seeds use the sorted index (deterministic; max seed 20,261,005 + 239,000 + ... < 2^31).
- Request id regex (cell 7 lines 295, 323) is greedy on the task id; 8-hex ids are safe.
- Multiple / different-size test inputs: each test input is run separately (cell 5 lines 196-200); F3 keeps a train-passing program even if one test input fails (cell 7 lines 167-172); attempts are chosen per test input from valid grids (cell 5 lines 249-254).
- Schema: `{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}` with one entry per test input; grids pass `poetiq_normalize_grid` (int 0-9, 1-30 x 1-30, rectangular). `e10_write_submission` (cell 7 lines 226-232) starts from the identity submission of every hidden task and replaces a task only if the row count matches; written atomically (`write_json_atomic`, tmp + `os.replace`, cell 1 lines 33-37) after every commit and again in the `finally` and cell 8 line 32. Unadmitted tasks get attempt_1 = input, attempt_2 = mirror (cell 8 lines 3-5 / finally loop).
- No path found where a committed answer is computed and not written, or where a later write replaces a good grid with a placeholder: E10_FINAL is recomputed from all candidates each time (best by score, `>=` tie -> latest).

### F10 — INFO — Clock and environment
- `NOTEBOOK_T0` = perf_counter at cell 0 minus (now - ipykernel process create_time), capped at 600 s (cell 1 lines 7-15). Deadlines: ADMIT 11:45, DRAIN 11:47, HARD_STOP 11:55, VERIFY 11:56, budget end 11:57, Kaggle 12:00. Whether Kaggle counts the 12 h from an earlier point (container start) is UNVERIFIED; two scored LB runs show no timeout.
- Engine start: solver started at 863 s (R5) / 907 s (D66). The /kaggle/input GCSFUSE slowdown since 2026-10-21 may lengthen model load for a new submission (UNVERIFIED for this vLLM path); each extra 10 min ~ 3-4 fewer tasks admitted (ESTIMATE).
- No time-zone use (perf_counter for all deadlines, UTC only in RUN_ID). Paths fixed to /kaggle/working and /tmp in both modes. Disk: R5 wrote 34 MB thinking + 1.8 MB programs for 172 requests; hidden ~250 requests -> ~50 MB. Child RLIMIT_AS 4 GB per sandbox, 4 workers. Sandbox requires root + libseccomp; the self-test gates the engine (cell 6 lines 143-144), and the LB scores prove it passes in the rerun.

## Fix priority (bundle)
1. F1 drain: earlier / staggered drain (config + ~15 lines), or in-engine forced close (also fixes F2b).
2. F3 + F4 exception wrapping (no behaviour change in normal runs).
3. F5 post-shutdown harvest (2 lines).
4. F6 projected-KV gate; F8 floor = 60,000.
5. F2 / F7 are throughput and attempt_2 levers, not bugs.
