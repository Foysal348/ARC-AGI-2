# Experiment 7 — Q27-XHIGH: Puzzle Ordering + KV-Cache Admission Controller (plan, for GPT validation)

**Status (2026-10-13), BASE:** `Experiment Notebooks/Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb` (sha256 `8fd956ce…`, builder `Build/E7B_build/build_order_only.py`) is the **new base for all further work** (user decision).
- **Contents:** v1.4.4 (LB 26.94) + exact Codex/NVARC ordering (commit and hidden) + PROXY48 commit profile + per-unit tier/rank/admit/finish in the records. No KV controller; cap 16.
- **CPU harness:** mixed, drain and rerun all PASS; units are admitted in Codex order; the hidden `submission.json` equals v1.4.4's.
- **Not pushed or run.**
- The KV-controller notebooks (COMBINED v4/v5, GPT FIXED24) are shelved.

**Earlier status (2026-10-12):** implementation **v3** built (CPU-validated), awaiting the final GPT re-audit.

- **Choice of implementation:** GPT delivered two (A, B). A has a backend-name bug that keeps adaptive off on a real GPU, so **B was chosen**.
- **v2:** Claude's builder (`Experiment Notebooks/Build/E7B_build/build_e7b.py`).
- **v3:** fixes all findings of the two final audits:
  - shadow decisions are counted once per telemetry poll;
  - `start_phase2` is transactional;
  - the final submission check is attempt-wise and runs before the report/ZIP;
  - matched PROXY48 controls A0/A1;
  - hidden-rerun guard;
  - `not_admitted` is set correctly on a scheduler error;
  - a portable test suite with fixtures.
- **Notebooks:** A0_PROXY48_BASE16, A1_PROXY48_ORDER16, B1_PROBE, PILOT_ADAPT18, CONTROLLER_v3.
- **Tests:**
  - `tests/test_e7_v3.py` 10/10, including an integrated adaptive 16 → 17 → 18 run;
  - GPT B's suite 5 × PASS;
  - harness 15/15;
  - the ZIP submission matches the disk file;
  - hidden rerun == v1.4.4.
- **Superseded:** `…E7_CONTROLLER_v2.ipynb` and the v2 zip are kept for history only.
- **Audit zip:** `Experiment Review/Q27_E7_v3_FOR_FINAL_GPT_AUDIT_2026-10-12.zip`.
- The plan below is the original proposal; the amendments are in the v3 README (cap 20 / active 16 first, then 18, no direct 24).

**Request to the reviewer:** validate this plan deeply and without limits. Everything below is a proposal; challenge any part of it.

**Hard rules from the user:**
- never less than 49,152 continuous xhigh thinking + 8,192 answer per test input;
- v1.4.4 (LB 26.94) is the base: its prompt, sampling, MTP ×3, host-forced close and parser stay unchanged;
- one experiment folder per experiment; nothing is pushed, run or submitted without approval.

**Not projecting hidden coverage from evaluation puzzles.** The private hidden puzzles are much easier than the evaluation tasks: thinking closes earlier and live sequences are shorter. That is exactly why dynamic KV management can help on the hidden set.

## 1. Goal

Finish **more test inputs within the 12 h hidden run**, and make the inputs that do not finish **the hardest ones**. Two changes, implemented together:

| Part | What | Why |
|---|---|---|
| **A. Puzzle ordering** | queue the tasks easy-first with NVARC's `get_puzzle_priority` (copied verbatim) | with time pressure, the inputs left unadmitted are the extreme tasks, not an alphabetical tail |
| **B. KV admission controller** | admit new test inputs based on measured and predicted KV usage, instead of a fixed 16 | easy (short) inputs leave KV idle at a fixed 16; long ones risk preemption above 16 |

## 2. Evidence from our own runs (v1.4.4 base, offline vLLM 0.19.0, 4× L4)

| Item | Value | Source |
|---|---|---|
| KV pool | 846 blocks × 1,600 = **1,353,600 tokens** | run 7 `run_report.engine.kv_tokens` |
| Concurrency today | `min(MAX_NUM_SEQS=16, kv_tokens // 30000)` = 16, fixed for the whole run | v1.4.4 cell 5 |
| KV usage with 16 running (EVAL12) | 0.30 at 2 min, rising about +0.017 per min to a **peak of 0.72–0.79** at 30–34 min; preemptions 0 | runs 4, 6, 7 `vllm_metrics.json` (every 2 min) |
| Rough decomposition | the about 0.27 offset at start ≈ prompts (about 5k × 16) + per-sequence state (mamba/GDN state blocks, about 1.3 % per sequence); then about **0.74 %** per 10k generated tokens | fit to the run 7 timeline (to be refined) |
| Live tokens per EVAL12 query | p50 54.9k, p90 56.8k, max 57.5k; 19/22 force-closed at 49,175 | run 7 report |
| Throughput | 249–285 tok/s with 16 running; MTP acceptance 0.66 | runs 4, 6, 7 |
| Tail inefficiency (EVAL12) | after about 36 min the active count drops 16 → 13 → 8 → 6 → 1 while nothing is queued | run 7 timeline |
| Hidden set | easier, so shorter lives and more natural closes; the actual numbers cannot be observed (no hidden logs) | user correction |

**Implication:** with 16 long sequences the peak is about 79 %, so **20 × 57k would reach about 98 %**. More than 16 is only safe if the controller knows how long the live sequences are and are likely to become. On the easier hidden set, many sequences end early, and KV sits idle with a fixed 16.

## 3. Part A — puzzle ordering (exact NVARC semantics)

**Source:** `resources/Puzzle Ordering (NVARC 32.22 ref)/puzzle_order.py`.
- `get_puzzle_priority` is byte-identical to both reference `starter.py` files (verified by `test_puzzle_order.py`).
- The key is `(tier, fine)`, ascending, so the easiest come first:
  - tier 3 = dynamic and max ratio > 30;
  - tier 2 = dynamic and ratio > 5, or average input > 400 cells;
  - tier 1 otherwise.
  - `fine` = size / ratio / colours / dynamic score.
- Python's sort is stable, so ties keep the **JSON file order**.

**Integration into v1.4.4 (cell 3 + the cell 6 queue only):**
1. `ORDERED_TASKS = order_task_ids(<challenges dict in json.load order>)`.
   - In the hidden rerun this is `test_challenges` as loaded.
   - In EVAL it is the selected subset, taken in file order, so it mirrors the reference, which sorts the file dict and then filters.
2. `ORDERED_UNITS` = for each task in that order, `task#0, task#1, …` (q ascending).
3. **Seeds unchanged:** `UNIT_INDEX` stays built from the original alphabetical `UNITS`, so every unit keeps the same `SEED + 10*UNIT_INDEX + k` as v1.4.4. Only the queue order changes:
   - `queue = [u for u in ORDERED_UNITS if u in PROMPT_IDS]`.
4. Logged: the tier counts, the first and last 5 tasks, and each unit's admit rank.

**CPU tests:**
- the queue is a permutation of `UNITS` (no duplicates or missing units);
- tie order equals file order;
- seeds are identical to v1.4.4 for every unit;
- the sequence of prompts is identical, only the order differs;
- the harness `rerun` gives the same `submission.json` as v1.4.4 when time is unlimited.

## 4. Part B — KV admission controller

### 4.1 Engine setting

- Raise the engine cap `MAX_NUM_SEQS` from 16 to **S_MAX** (proposal: 24; GPT, please judge). This is fixed at engine start in vLLM.
- Effects to verify on GPU:
  - per-sequence mamba state may reserve memory, so log the KV pool at 24 against 16 (1,353,600);
  - CUDA graph capture sizes;
  - MTP drafter memory.
- **Fallback:** if the engine fails at S_MAX, retry with 16 using the existing ladder logic, and the controller degrades to v1.4.4's fixed 16.

### 4.2 What the controller knows at every scheduler loop (no extra GPU work)

- **Per active unit, exact from our own state:** `live_i = len(prompt_ids) + len(p1_ids) + len(p2_prefix) + len(p2_ids)`, plus its phase and the remaining budget (`phase1_cap − len(p1_ids)` in phase 1; `answer_cap − len(p2_ids)` in phase 2).
- **From vLLM** (`llm.get_metrics()`, as `metrics_snapshot()` already does): `kv_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`, `num_preemptions_total`. Polled every T_POLL ≈ 10 s (cheap; to be measured).

### 4.3 Usage model (calibrated online)

Predicted usage = `a · n_active + b · Σ live_i`:
- `b` starts at 1 / KV_TOKENS;
- `a` starts at the per-sequence overhead estimated from run 7 (≈ 0.013);
- both are refined by least squares on (measured usage, n_active, Σ live) samples, with clamped bounds.
- **Validation target:** a prediction error under about 0.03 on the run 4, 6 and 7 timelines (offline CPU replay from `vllm_metrics.json` plus `records.jsonl`).

### 4.4 Admission rule (all must hold to admit the next unit in ordered-queue order)

1. `n_active < S_MAX`, before `ADMIT_DEADLINE`, and the queue is not empty.
2. **Projected peak** ≤ `U_TARGET` (proposal 0.90). The projected peak = predicted usage if every active unit and the candidate grow to their expected final length:
   - **expected final** = `live_i + E[remaining_i]`;
   - **warm-up** (fewer than W = 8 finished units): E[remaining] = the full remaining budget (worst case, so never more than about 16 long sequences);
   - **after warm-up:** E[remaining] = the empirical p90 of final live length among finished units, minus `live_i`, floored at 0. It uses only finished units' own lengths and is never read from solutions;
   - the candidate's expected final = `prompt_len + p90(final generated length)`.
3. No preemption increase during the last `T_COOL` (proposal 120 s), and measured usage < `U_HARD` (proposal 0.93).
4. **On a preemption increase:** freeze admission for `T_COOL`, lower `U_TARGET` by 0.03 (floor 0.80), and log it. vLLM's own preemption (recompute) handles the overflow; the controller **never aborts or shortens** a running request.

### 4.5 Invariants (asserted in code)

- Every request still gets `phase1_cap = 49,152` (prompts ≤ about 9k) and answer 8,192. **The controller only decides *when* to admit, never *how much* a unit may think.**
- Drain (11:36), hard stop (11:44) and the end (11:50) are unchanged. Admission still ends at 11:34.
- `SamplingParams` keeps only `max_tokens`, `seed`, `temperature`, `top_p`, `top_k`, `detokenize`, `skip_special_tokens`, `output_kind`: nothing that MTP silently ignores (vLLM #39573).
- If metrics are unavailable or malformed, fall back to the fixed v1.4.4 rule (`min(16, kv_tokens // 30000)`).

### 4.6 Logging (every 60 s and in the report)

- n_active, engine running and waiting counts, measured usage, predicted usage, projected peak;
- `U_TARGET`, the admissions and blocks with their reason (cap / KV / cooldown / deadline), preemptions;
- tok/s per window, and the calibrated a and b.
- Per unit: admit time, admit rank, tier, KV at admit, final live length, mode.

## 5. Interaction with Experiment 6 (dual answer)

- Experiment 6's GPU EVAL12 is running now (`foysalemonshanto/q27-xhigh-dual`).
- Experiment 7 is built on **v1.4.4**, with the dual-answer stage **off**. Merging comes later, once both results exist.

## 6. Validation plan

**CPU (before any GPU run):**
1. Part A unit tests (§3).
2. **Controller unit tests with a simulated engine:**
   - the simulator models usage = a · n + b · Σ live, plus preemption above 1.0;
   - scripted length mixes: all-long (EVAL12-like), all-short (hidden-like), mixed;
   - **expectations:** all-long admits no more than about 16 and has no preemption; all-short goes up to S_MAX; mixed adapts.
3. **Offline replay:** the run 7 timeline is predicted within the target error.
4. **Experiment 1 harness** (real tokenizer, fake vLLM step engine) with the fake engine extended to report usage and preemptions, scenarios `mixed`, `drain`, `rerun`:
   - no scheduler change other than order and admission;
   - with the controller forced to fixed 16 and ordering off, `submission.json` must equal v1.4.4's exactly (regression proof).
5. pyflakes over all joined cells, plus the existing cell 4 self-tests.

**GPU (one run per step, each needs approval):**
- **G1. Engine probe within an EVAL run:** start at S_MAX = 24; log the KV pool and the startup time. Abort criteria: the KV pool more than about 10 % smaller than at 16, or engine failure. In that case use S_MAX = 20 or 16.
- **G2. EVAL on a set with more units than S_MAX** so the controller is exercised:
  - proposal: the **EDA PROXY-48** set (48 tasks, 56 outputs), the user's planned LB-mirror proxy;
  - controller on, ordering on; EVAL time budget about 3–3.5 h (to be set);
  - **report:** time to finish all 56, overall tok/s, active-count timeline, peak usage, preemptions, model error, accuracy (primary hits, sanity only).
- **G3. Baseline comparison** on the same set with `Q27X_KV_CONTROLLER=0` and fixed 16 (or reuse when equivalent):
  - **success** = materially less wall time (or more units finished by a fixed deadline) with preemptions ≈ 0 and no accuracy drop beyond noise.
- **Hidden submission:** only after G2 and G3 pass, and only with the user's explicit go.

## 7. Questions for the reviewer

1. Is S_MAX = 24 sensible on 4× L4 with MTP ×3 and mamba "align" prefix caching? Or should we probe 20 first?
2. Is the linear usage model (`a · n + b · Σ live`) adequate for the hybrid attention/GDN cache with prefix caching, or is there a better signal (for example block counts)?
3. Is U_TARGET 0.90 / U_HARD 0.93 the right margin, given MTP lookahead and chunked prefill (4,096 batched tokens)?
4. Using the p90 of finished units' final length after warm-up: is there a risk on the hidden set (ordering puts easy units first, so early lengths underestimate later, harder ones)? Proposal: compute the expectation **per tier** (tier 1 / 2 / 3), falling back to worst case for tiers with fewer than W finished units.
5. Is PROXY-48 the right GPU evaluation set for this (its distribution comes from the placeholder visible test file)? Or EVAL20-B plus EVAL12?
6. Anything else: risks, simpler designs, or a reason to split A and B.

## 8. Files

| Path | What |
|---|---|
| this `README.md` | the plan |
| `Experiment Notebooks/Build/` | (later) the builder derived from `Experiment 1/.../Build/Q27X_build/make_q27x.py`, plus tests |
| `Experiment Review/` | (later) GPT review zips |
| `Experiment Results/` | (later) GPT outputs and Kaggle runs |

**References:**
- `resources/Puzzle Ordering (NVARC 32.22 ref)/`;
- `resources/Eval Subset (EDA mirror)/`;
- `Experiment 1/.../v1.4.4 vs all experiments - remaining issues (2026-10-12).md` (P2, P4);
- run 4, 6 and 7 outputs in `Experiment 1/Experiment Results/Kaggle Runs/`.
