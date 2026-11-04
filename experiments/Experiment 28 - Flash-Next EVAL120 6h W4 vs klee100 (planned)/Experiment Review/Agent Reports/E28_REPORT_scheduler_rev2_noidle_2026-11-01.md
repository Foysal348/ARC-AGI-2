# E28 scheduler agent: rev 2 (no idle worker) report, verbatim

## E28 SCHED rev 2: the hold is gone; no worker idles while something fits. 31 tests, all pass on Python 3.12 (5 full runs).

**The two builders must re-embed the module and apply the new N4 edit.** `build_e28_k100.py` L128 asserts the edit labels are exactly `["N1","N2","N3"]`, so it will fail until it is updated: `NOTEBOOK_EDITS` now has 4 items.

### Final shas (all files LF, no CR), in `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build/`

| What | sha256 |
|---|---|
| `e28_sched_patch.py` (68,463 B, 1,215 lines) | `47ab9874a9f3a83f2c9a0fbf9b674bd260e4d3b34f3076b0b9601b5bdb5382ef` |
| `arc2_duck.py` pre | `b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128` |
| `arc2_duck.py` post | `da2c8682096e858ca7f29df84c7f0717da3e823b0e74dfe7f4e831b001aa02a0` |
| patch diff | `b8d17e9305b01f3190f765df4a052b6e076b6f0104b299205b228a2446598f0f` |
| applier | `b871f0e409c5f44ad22a44cb61051138efc220657010b3a85ba4b53a72340e88` (unchanged) |
| `e28_sched_tests.py` (1,040 lines, 31 tests) | `21a0ea768145b635f2cc83da5d5e6c918b0a69d4f3bc720343cdabe07f46fd9b` |
| `E28_SCHED_API.md` | `421ab22c81df753774d563bfcf721a63ed368f4e42d566884de5ad2f24974605` |

- The API file's status line now has the new shas (the old `1ff9943a` is gone).
- The patch has the same 8 hunks as before (original lines @@ -1444, -1456, -1464, -2769, -2829, -2839, -3247, -3659), now +740 / −3. Only the E28 block appended at the end of the file changed (+719 lines).

### What changed in the scheduler
1. **Hold removed.** A free worker never waits for an earlier-phase attempt.
2. **Priority with backfill.** At every admission check (a worker frees, KV frees) the queued items are tried in this order:
   - B2 items that already waited (did not fit before) first;
   - then A (task order), then B (task order), then C1 (task order), then C2 (tokens ascending).
   - The first item that fits starts at once. A head that does not fit stays at the head and is re-checked at every check, while a lower item that fits starts instead.
   - A worker waits only when nothing fits at all. With nothing running, the head is admitted anyway, so there is no deadlock.
3. **Fit check.** It comes from a new hook, `E12A.E28_TRY_RESERVE(key, plan)`, installed by the new edit **N4** (CELL 3, after `E12A.BranchSession.run = _nb5_session_run`).
   - The hook takes or refuses a reservation in `NB5_RES` immediately and never waits. The attempt's own `res.acquire` in `_nb5_session_run` then finds its key already held.
   - The block count is 1 in slots mode, `start` in live mode, N2's blocks in projected mode. Grant rules are the same as `Reservation.acquire`.
   - `E12A.E28_RELEASE` is called after every run (idempotent), so no reservation leaks.
   - Without the hook, everything counts as fitting and the order is plain A → B → C.
4. **Slots mode (the notebooks' mode).** Each attempt needs 1 slot and capacity = workers, so every free worker immediately takes the next item in priority order, with no fit failures and no backfill.
   - I also found and fixed an idle source here: the KV gate (reservation fraction, limit 1.0, 5-s probe) could hold a just-freed worker on a stale "100 % full" sample for up to 5 s.
   - A gate sample taken before the latest attempt end is now ignored (counted in `kv_superseded`).
5. **Unchanged:** the KV gate (never preempts), deadline clamp, `min_admit_sec`, phase budgets and windows, the answer slots, and env-off identity.

### New records
- **Per B / C attempt row:** `queued_sec`, `first_tried_sec`, `wait_sec` (from runnable to start), `fit_fails`, `backfill_starts_while_waiting`.
- **Run level, in `SCHED_LAST["e28"]`:**
  - `idle_worker_seconds` = time waiting while something was queued but nothing fit, plus time held by the KV gate;
  - `idle_sec_by_reason` = {`nothing_fits`, `kv_gate`, `no_runnable`};
  - `backfill_starts`, `backfill_log`, `forced_admits`, `admission_checks`, `fit_checks`, `fit_fails`, `fit_errors`, `fit_hook`.

### Tests
The test that failed earlier, `test_backfill_b2_waits_c_starts`, now passes in every run. It had two test-side faults, not scheduler faults:
- its script gave the C attempt the wrong answer value;
- its timing was racy (the attempt-1 ends are now staggered).

The 31 tests include these new ones:
- **`test_slots_mode_no_idle_b2_at_its_turn`** (real `Reservation` in slots mode, 6 workers, 40 tasks, 5-s probe, limit 1.0):
  - `idle_worker_seconds` < 0.05 and `nothing_fits` = 0;
  - 0 fit failures, 0 backfills, 0 forced admits, no leaked slot;
  - the B2s start right after the last attempt 1, in task order, before any C;
  - every C starts with empty A and B queues.
- **`test_backfill_b2_waits_c_starts`** (projected, capacity 90: two T1 attempts fit, a B2 fits only alone):
  - while the B2 waits, free workers start C attempts (backfill is logged);
  - the B2 is checked first at every check and is granted exactly once, at the first check where it fits;
  - its `wait_sec`, `fit_fails` and backfill count are recorded; no leak; the answer slots are right.
- **`test_no_kv_constraint_no_idle_and_abc_order`**: idle exactly 0, the A → B → C order holds.
- **`test_reservation_sim_no_deadlock_no_leak`**: 60 tasks, 12 workers, projected mode; every B2 starts, nothing leaks, the grant counts add up.
- **`test_n4_try_reserve`**: grant / refuse without waiting, the attempt's own acquire finds its key held, idle grant, live hold, the over-capacity case, slots mode needing 1 slot, and no reservation at all.

The existing single-worker exact-order test (A, then B, then C1, then C2 ascending, when KV is not the constraint) and the 120 tasks × 21 workers simulation still pass. Each notebook export takes all 4 edits exactly once and still compiles.

### Open points (unchanged from rev 1)
- `MAX_LEN` and `E21_VLLM_MAX_MODEL_LEN` must be 262144 in the notebooks.
- The T1 request timeout of 3,600 s is a risk.
- Do not set a fixed `LOCAL_ANALYZER_SEED`.
- CELL 7's metrics treat b2 as attempt 3+.
- 200k-context correctness on Flash-Next is untested.
- In projected mode a B2 can wait as long as backfilled items keep the pool full; this is by the user's choice. In slots mode it does not apply.
- Still to do: the Kaggle CPU kit run.
