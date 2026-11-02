## Report: GENTAIL_CKPT, the Option B prototype patch (2026-11-01)

The Option B patch is built as a seam patch with CPU tests: **31 / 31 new tests pass, and the earlier 11 / 11 still pass**. It has not touched the real vLLM classes yet: no local venv has vLLM's dependencies, so the logic was tested against fakes.

All coordinator inputs are folded in: the GPT audit, fix list G-1 to G-5, and GPT research section 5. No GPU, Kaggle or git; Merlin source is untouched (the pristine tree still verifies `not_applied`, `git status` clean). The one experiment file I changed is the design doc: a v2 section was appended as asked (§7).

I could not save this report as a file: the tool blocked report-file writes from a subagent. So `REPORT_gentail_optionB_2026-11-01.md` is not in `Experiment 30/src/gentail/`. Please save this message there (and in Agent Reports). The other 8 files are copied.

### 1. Files and shas

Work folder: `D:/ARC AGI 2/_scratch/gentail_ckpt/`. Copy: `D:/ARC AGI 2/experiments/Experiment 30 - Capacity port + generated-tail checkpoint (planned)/src/gentail/` (8 files, shas identical; the 31 + 11 tests also pass when run from there).

| File | sha256 | What it is |
|---|---|---|
| `gentail_seam.py` | efc5896afd71d17e76837e6aaf4aac33f1caa977d719f9e8c0b95e8e833e0ce8 | the seam: embeds the new module; `EDITS` (exact-once anchors), PRE / POST / NEW_SHA, `apply_gentail_patch` / `verify_gentail_patch` / `revert_gentail_patch` (fail-safe restore), `tree_state`, `server_env`, `parse_log`, `readback` |
| `gentail_ckpt_module.py` | db5f3beef28bcab27055f52893576aa61dff394b610e6cbda51925099dc1e597 | source of the new Merlin file `vllm/v1/core/gentail_ckpt.py` (= NEW_SHA) |
| `gentail_seam_template.py` | 83857c0a92dcb55b5eb4f77a2ff356e8d1b8f80ca4fda5e6a61c2cc6934e500a | template the builder fills in |
| `build_gentail_seam.py` | 75a67cff7973b1bedf8b5e64114012d869a698f6b96f0b2bea3968b1b8f43951 | builder: computes the shas from the pristine Merlin files |
| `test_gentail_seam.py` | 859234b5dde5387fcd72a4663d8ab5c1535ca8363b04e04aef48ff88dd7a814b | 31 CPU tests |
| `test_gentail_logic.py` | 0405778ec2b701c8ce2565ff8e35d7705fead32be4d430e7dd18f0666b400a8e | earlier 11 tests; now fail (no longer return) when the CSV is missing (G-4) |
| `gentail_logic.py` | 5c3646a32f758992e16f984dbb2410b5d500cb99dd9ab1e1362b2cdd62d0c690 | earlier prototype, unchanged |
| `GPU_TEST_SPEC_gentail.md` | 6c4df7ca23ca70015cf76e83b2c15965ef1f91ad4daa930950652b66ff07ed52 | GPU test spec |

`tree_copy/` (a 51 MB copy of Merlin `vllm/`, currently in the patched state) stays in `_scratch` only. The tests take `GENTAIL_PRISTINE`, `GENTAIL_TREE_COPY` and `GENTAIL_DUCK_TIMELINE` env overrides for the Kaggle kit.

Merlin files (ad42e37, tree 07c16a79):

| File | pre sha256 | post sha256 |
|---|---|---|
| `vllm/v1/core/sched/scheduler.py` | ea56781115c350cf0c053912bc855f86d6bb47a19cd914e1d2e045b99b2a4a7f | 2b670430460b9a4518009b2998d6802fbf156524a2f7f9ccf98bdf9208b18d1e |
| `vllm/v1/core/kv_cache_manager.py` | 9f7ebfd75099a9ea784c54be97fbc87dfb87d2355b88d6a7caed4c9f8ad1dc93 | 45a8d99c36d7decab341aa4a61ab112c27526ca0c5488c4ef8b52c7a46f0614f |
| `vllm/v1/core/gentail_ckpt.py` (new) | absent | db5f3beef28bcab27055f52893576aa61dff394b610e6cbda51925099dc1e597 |

These files don't overlap with E25 NB4 (`pp_utils.py`, `model_runner.py`, `gpu_worker.py`) or any E27 `dflash_seam` file; the seam refuses to load if they ever do.

### 2. The patch: 8 anchored insertions and 1 new file

**`scheduler.py`:**
- class attribute `_gentail = None`;
- `self._gentail = maybe_install_gentail(self)` right after `configure_recoverssm_pp_checkpoints(...)`. The per-step guard is kept and never flipped (tested);
- `on_schedule(request, num_scheduled_token, self.sched_step_seq)` before `request.num_computed_tokens += ...`;
- `on_step_output(request, num_tokens_scheduled, len(generated_token_ids), self.processed_step_seq, output_is_stale)` after the rejection adjustment;
- `drain(self.processed_step_seq)` right after `_drain_deferred_frees()`.

**`kv_cache_manager.py`:**
- class attribute `_gentail = None`;
- `on_free(request)` before the table is popped, and `after_free()` after the grace pin, in both `free()` and `pop_blocks_for_free()`.

**Off by default.** Every hook is `if self._gentail is not None:`. With `GENTAIL_CKPT` unset or 0, the install returns None before reading anything. The tests check every inserted line for this guard.

**What the new module does:**
1. **Record.**
   - `commit_plan_boundary` is an exact port of the align branch of `prepare_commit_plan_kernel` (`ops/recoverssm.py:64-115`).
   - It is fed with the window's committed count at schedule time (only when nothing else is in flight for that request), the query length and the sampled count.
   - Its result is cross-checked against the scheduler's rejection-adjusted count; on a mismatch nothing is recorded and `plan_mismatch` goes up.
   - When the plan stores the boundary state at b, the block of column b/4096 − 1 in every Mamba group gets one extra ref (`touch`). Retirement or free then drops only the table ref.
   - A newer crossing replaces the record; the old hold is released behind the fence.
   - The prompt-tail CoW hold is released at the same time, so memory stays neutral (`GENTAIL_CKPT_KEEP_PROMPT_HOLD=1` keeps it).
2. **Finish.** The PendingTail is queued, never published here, and only if all of these hold:
   - status is `FINISHED_STOPPED` (PR #50551);
   - the request generation is unchanged;
   - no stop truncation (strict by default: discard when committed ≥ num_tokens);
   - producer rule: T + pp_size ≤ L + 1;
   - b > the previous prompt;
   - b ≤ floor(valid/64)·64 − 64, with valid = min(committed, num_tokens) − reprefillable.

   The attention and MTP-draft groups get their key at A = floor(valid/64)·64. G-1 fix: when A falls on a block end there is no partial key; the full block A/4096 − 1 must be hashed at A. The token hash at b is recorded.
3. **Drain.** Publish only if all of these hold:
   - `processed_step_seq ≥ T + pp_size`;
   - the generation is unchanged;
   - b ≤ the limit;
   - the prefix hash at b still equals the recorded one.

   Then every Mamba block is keyed at b and handed to the turn's grace pin (else to LRU), and the pin budget is enforced. Every other path releases the hold behind the fence.
4. **Grace pins (G-3).** `MAX_GRACE_PINS` = `max_num_seqs` (21), set on the instance. A pin budget caps all grace-pinned blocks at 25 % of the pool (`GENTAIL_CKPT_PIN_BUDGET_FRAC`); oldest go first, the newest always stays.
5. **Counters (G-5):** recorded, superseded, queued, deferred, published (+_pinned / _lru / _tokens / _store), skipped + skip_<reason>, released, adopted (+_tokens), evicted, stop_truncated, final_window, pin_budget_released, index_reuse, plan_mismatch, seq_mismatch, no_block, sched_overlap. They go out as `E30_GENTAIL` stats lines (with the grace-pin load). An optional per-publish line carries the block ids for gate 3.
6. **Install checks.** The patch stays inactive (logs why, changes nothing) without prefix caching, RecoverSSM, Mamba groups, align mode, or spec decode, or with speculative Mamba blocks. It raises a RuntimeError if `defer_block_free` is off, as you asked.
7. **Future hook:** `tail_store.offer(pend) -> bool`, an interface point for an INT8 cached-checkpoint store in the style of SGLang #28185. Not implemented.
8. **Negative control:** `GENTAIL_CKPT_NOFENCE=1` lifts the fence, the producer rule, the limit, and the length-capped and truncation rules. For gate 4 only.

**Env** (all unset = original behaviour): `GENTAIL_CKPT=1`, `_STRICT_TRUNC=0`, `_KEEP_PROMPT_HOLD=1`, `_PIN_BUDGET_FRAC`, `_LOG_PUBLISH=1`, `_STATS_EVERY`, `_NOFENCE=1`.

### 3. Test results (Python 3.14 + torch 2.14 CPU, about 45 s)

**A. Patch mechanics, on `tree_copy` (refreshed from pristine each time):**
- apply → verify → revert gives back byte-identical files and removes the new file; re-applying reports `already_applied`;
- a foreign file is refused and nothing is written; a mixed state is detected;
- a forced `py_compile` failure restores the originals;
- every anchor occurs exactly once in pristine;
- every inserted line is guarded, and default off is proven;
- the hooks sit in the right order, and the per-step guard is untouched.

**B. Logic.** The shipped `gentail_ckpt.py` from the patched tree runs against fakes of BlockPool, the managers, KVCacheManager and Request, driven by a two-rank PP model:
- PP1 commits in the step;
- PP0 commits at call T + pp from a ring of depth pp, and skips if the slot was freed first;
- the batch queue is 2 deep.

Every Mamba key is checked at the moment it is written: both ranks must hold exactly that state, and the block must not have been reused.

Covered:
- commit-plan port; position limit with reprefillable 0 and 2;
- G-1 at A = 4032 / 4096 / 4160 / 8192;
- busy two-microbatch (asserts a second microbatch was in flight at finish);
- idle with a crossing in the final window (skipped);
- idle with an earlier crossing (published with no deferral at pp 2; deferred then published at pp 24);
- **negative control:** NOFENCE exposes PP0's missing commit in at least 5 of 20 seeds; the same seeds with the fence are clean;
- abort at a boundary: no publish, no leak;
- preemption;
- index reuse before the fence and while pending;
- strict vs relaxed truncation;
- the PR #50551 status rule;
- MTP all-accepted and all-rejected;
- superseded hold, and a retired block that stays held;
- grace-pin hand-off, adoption, the LRU path, the evicted counter, the pin budget;
- install gates;
- a 6-seed × 8-turn random soak with no hazard and no leaked ref;
- publish log and readback.

**C. Data replays.** Both fail if their CSV is missing, and a test proves it.
- **K100Q, 677 turns, under the patch's own rules:** saves **1,688,832 of 2,430,527 tokens (69.5 %)** across 297 turns. 9 turns lose a gain of at most 127 tokens to the b > prompt rule; 3 follow a length-capped finish and are discarded.
- **Grace-pin pressure (G-3, from `gaps.csv`):** a pin holds all hashed blocks of the turn.

  | Pin interval | Peak pins | Peak pinned tokens | Share of the 1,652,197-token pool |
  |---|---|---|---|
  | finish → next request | 3 | 176,608 | 10.7 % |
  | plus a 2 s admission wait | 5 | 209,760 | 12.7 % |

  Raising the cap from 16 to 21 never binds in this data.

### 4. Where I departed from the inputs, and why

1. **The PP model found a bug in the design's producer rule.** "T < L" is only right for pp_size 2. The general rule is **T + pp_size ≤ L + 1**; for pp 2 the two are identical.
2. **The block hold is an extra ref taken when the crossing is recorded**, not an intercept inside `_retire_state_blocks`. The effect is the same (tested: a retired pending block stays out of the free queue; replacing it releases the old ref), and `single_type_kv_cache_manager.py` stays unpatched.
3. **The producer step comes from the exact kernel port, fed with the worker's inputs, plus a consistency check.** Real commit-plan metadata would need a `model_runner.py` change, and NB4 owns that file.
4. **The request generation is a scheduler-side mirror.** The real slot generation lives on the worker. The producer rule plus the fence keep a skipped PP0 commit from ever reaching a published state; gate 4 measures this.
5. **Stop truncation is strict by default, as the audit asked.** `GENTAIL_CKPT_STRICT_TRUNC=0` relaxes it. The relaxed rule is also exact, because b is at least 16 windows before the end, and `stop_truncated` measures what strict costs.

### 5. Open risks

1. **Not yet run on the real vLLM classes.** The local venvs lack cbor2 and cachetools. The next check is a Kaggle CPU kit run on the real KVCacheManager, coordinator and BlockPool built from the K100Q kv_cache_config (spec section 0).
2. **Commit-plan input assumption.** It assumes the worker's count equals the scheduler's committed count at schedule time; the GPU run must show `plan_mismatch` = 0.
3. **Stop strings.** If Duck ends turns with a stop string, the frontend aborts the engine request, and an aborted turn never publishes. On the GPU, `skip_abort` vs `published` shows which case we are in.
4. **Strict truncation has an unknown cost.**
5. **Sibling branches.** Releasing the prompt-tail hold may cost sibling-branch hits under pool pressure. The A/B switch is `GENTAIL_CKPT_KEEP_PROMPT_HOLD=1`.
6. **Reader-side hit is untested on real code.** The hit through a published block is not tested against the real coordinator (gates 2 and 3 cover it).
7. **The fence cannot show its value in a normal pp 2 flow.** The position limit already keeps every final-window state out, which is why the negative control has to lift every rule.
8. **Gate 3 tolerance** depends on a measured chunking-noise baseline.

### 6. GPU test (`GPU_TEST_SPEC_gentail.md`)

- **Section 0, CPU kit first:** run both test files, a real-class import and install check, then a revert.
- **Two in-process boots:** A with `GENTAIL_CKPT=1` plus the publish log; B with NOFENCE, used only for the negative control.
- **Workload:** one recorded 10-turn K100Q chain, cold vs warm.
- **Fail-fast gates:**
  1. **Token-ID identity, measured:** any published hit must have m_k ≥ b_k.
  2. **Position:** `cached_tokens` equals the prediction; the publish and adopted counters agree.
  3. **State:** max-abs difference per layer per rank against a cold reference whose checkpoint sits at b, within a measured chunking baseline; plus next-turn greedy and logprob parity.
  4. **Fence:** busy, idle, and abort with no leak; the negative control should show a difference on PP0 only. If PP0 is clean, the result is reported as inconclusive.
- **Budget:** about 30-40 minutes. It needs your go.

### 7. Design doc

`Experiment 25/Experiment Review/DESIGN_generated_tail_checkpoint_2026-11-01.md` has a new section 9 (v2) appended; sections 0-8 are unchanged:
- **9.1:** the v1 → v2 change table (G-1 to G-5, PR #50551, the per-step guard, the producer-rule correction, the negative control, the store hook);
- **9.2:** the 69.5 % gain under the patch's rules;
- **9.3:** reference designs: #53803, #50551, #55697, SGLang `mamba_checkpoint_pool.py` / #28185, Marconi;
- **9.4:** open risks.