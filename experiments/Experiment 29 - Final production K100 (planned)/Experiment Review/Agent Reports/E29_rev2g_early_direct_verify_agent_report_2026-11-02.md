**GO.** I found no correctness blocker in rev 2g. One HIGH finding is a design trade-off for the user to decide, not a code bug.

**Code reviewed.** I rebuilt the patched arc2_duck.py the same way `regen_sched` does: Duck payload 9e4b8928 → `apply_e29_sched` → `make_dsched`. The generated module sha is b88efa86… and equals Build/e29d_sched_patch.py byte for byte. The post sha is 08eeb370…, the pre sha d8603109…. The rebuilt file is `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\verify_early\arc2_duck_direct_post.py`; line numbers below marked "post" refer to it. Against the backup, the only changes are K6/K7/K8, `E29D_EARLY_FROM`, `_e29d_early_on`, two stats counters, comments and SCHED_PIN.

## Findings, ranked

### BLOCKER
None.

### HIGH-1: putting early DIRECT ahead of B (K7) pushes the B2 re-runs past the window end
- **Where:** K7 is e29_direct_build.py:125-136 (post 4014-4023).
- **Mechanism:** A always goes first, so every train-fail task's follow-ups wait until the A queue drains. In E28 that was at 13,956 s (233 min). After that, 16 DIRECT items and 16 B2 items compete for 24 slots.
- **E28 data:** the 8 B2 re-runs that reached TRAIN_PASS finished very close to the end. For example, 16b78196's B2 was accepted at 20,876 s of 21,600, and that was the task's only hit.
- **Replay** (`sim_des.py`): the real E28 A-tail end times, 24 slots, E28 B2 durations.
  - DIRECT at 18 tok/s per stream (the E28 per-stream rate at about 24 running): rev 2g loses 5 of the 8 TRAIN_PASS B2 re-runs (16b78196, 89565ca0, 8b7bacbf, 981571dc, b9e38dc0). Two of those were exact solves: 16b78196 and 981571dc.
  - At 30 tok/s: 4 lost, including 16b78196.
  - What DIRECT gains: all 16 items, thinking time to the drain 27-96 min, median 93.
- **Crude 12-h / 240-task extrapolation** (`sim_12h.py`): 12 of 16 TRAIN_PASS B2s lost at 18 tok/s, 4 of 16 at 30 tok/s. The 12-h run has the same pile-up after the A queue drains at about 465 min.
- **Alternative (user's call): drop K7 only, keep K8.** Early items have c_key (1, 0, idx), so they already sort ahead of every C2 item. They would run after B but before C2.
  - 6 h: 0 B2 lost; DIRECT gets 12 of 16 items at 18 tok/s and 16 of 16 at 30 tok/s, median 64 / 53 min, minimum about 3 min.
  - 12 h: 0 lost; 23 of 32 items (median 55 min) or 32 of 32 (median 41 min).
- **Baseline for comparison:** under rev 2f in this same E28 timeline, DIRECT would have run on 0 tasks. The C1 items started at 346-352 min, after the admission close at 333 min.
- **Caveat:** these are replays of the E28 timeline with fixed durations, not GPU measurements.
- A reference point: in e29_direct.py's REF_TABLE (Direct27B column) the 27B direct solver scored on a47bf94d (1.0) and b6f77b65 (0.5). Rev 2g gives them 88 / 73 min of thinking; the alternative order gives them about 20-26 min.

### MEDIUM-1: DIRECT also runs on tasks whose B later passes, and those tasks lose their Duck C2
- **Where:** K8, post 4221-4238.
- **Scenario X4:** B reaches TRAIN_PASS on an early task. Result: no C2 is queued, and DIRECT has already run. attempt_1 = the B grid, attempt_2 = the DIRECT grid.
- **Same scenario on pre-2g:** t002 gets a Duck C2 and DIRECT is not called.
- **E28 numbers:** 8 of the 16 B re-runs passed train. 5 of those 8 were accepted but wrong, so a DIRECT attempt_2 still has value there. Only 1 of the 8 (981571dc) had its C2 started in E28, and that one was redundant.
- **Gap:** nothing converts or skips an early item if B passes before the item is taken. That ordering is rare: it needs a failed fit check or a waiting B2. This partly conflicts with "never spend time on tasks Duck already passed".

### MEDIUM-2: DIRECT can be admitted with very little thinking time left (pre-existing since rev 2d)
- **Where:** CAN_START is `now + 60 s + 26 min < deadline` (build_e29_final.py:796), and K8 relies on it at the A end.
- **Effect:** a DIRECT item can start with only about 1 minute before the T-26 drain. Tasks whose A ends between T-60 and T-27 min get under 33 minutes.
  - In the E28 replay: e12f9a14 gets 38 min and f560132c gets 27 min. At 18 tok/s that is about 29-41k thinking tokens, below the 60k floor.
- **Cost:** the forced-close grid then replaces the displaced A / r5 attempt_2.

### MEDIUM-3: the GPU smoke cannot exercise the early path
- **Why:** the E29SMOKE window is 40 min with drain_min 26. CAN_START is True only in the first ~13 min (about 8 min in the v1 smoke's 2,089-s window). Train-fail A attempts run much longer than that.
- **Expected smoke result:** early_queued = 0.
- **The SUBMIT folder holds the smoke bytes:** that notebook's commit run is the smoke too. The hidden rerun uses RERUN_PROD (`PROFILE = "RERUN_PROD" if RERUN`).
- **Coverage:** rev 2g is validated only by the CPU tests.

### LOW-1: an early item that is late when taken runs Duck C before its own B
- **Where:** post 4340-4350.
- **Confirmed by scenario X1:** one C item, one B, one release, no second C — but the Duck C starts before the B.
- **When it happens:** only if the A drain or a fit wait pushes the pop past T-27 min. Neither the E28 replay nor the 12-h extrapolation produces it.
- **Harm:** small, since both attempts fall in the last 27 min.

### LOW-2: when DIRECT errors or returns nothing, there is no Duck C fallback (same as rev 2f)
- **What attempt_2 becomes:** the E29 rule over A + B (+ supply), i.e. the displaced answer, else r5, else the test input. Scenarios X2 / X3 verify this.
- **E28 value of Duck C on C1 tasks:** 3 started at 346-352 min (20a9e565, 21897d95, 2d0172a1), 3.4k-8.1k tokens each, all INCOMPLETE_WINDOW, 0 hits. The other 5 queued C1 items never started. So nothing measurable is lost.

### LOW-3: the negative control is weak, and some coverage was missing
- **Pre-2g results:** against f2c60690, 5 tests fail:
  - one real behavioural FAIL: test_early_before_b_one_worker sees c_from B1 instead of A_DIRECT_EARLY;
  - four are only a KeyError on the missing 'early_queued' counter (three TestEarlyDirect tests plus TestC1Scope.test_c1_default_direct_on_train_fail_only).
- **Not discriminating:** test_early_attempts passes on pre-2g.
- **Gaps:** these cases had no test, so I wrote them as X1-X7:
  - an item late at pop;
  - DIRECT returning None / raising on an early item;
  - B passing on an early task;
  - the window ending while an item is queued;
  - supply with a cap of 5;
  - 4 workers with B running while its DIRECT runs.

### LOW-4: the code comment misstates the E28 data
- The comment says "C1 items started at 308-352 min" (e29_direct_build.py:254, and in the post file).
- In the report, the c_priority-1 items started at 346.1-352.1 min (20,768-21,129 s). The 308-min item (981571dc, 18,502 s) was a C2.

### LOW-5: stale build records
- SCHED_DONE.json still holds the rev 2d sha 4cc807f2….
- The SCHED_PIN assert message (build_e29_final.py:1256) still says "rev 2d build". Record only.

### LOW-6: two TestC1Scope tests now compare C2 records order-free
- They drop the ROW_ORDER keys. This is justified because the queue order really changes; attempt content and answers are still compared.

## Check results

**1. Per-task correctness of the early path** — verified by code reading plus tests and scenarios X1-X7:
- Exactly one branch-1 item per task. The B end of an early task queues no second C item (post 4235).
- `E28_RELEASE((task, 1))` happens exactly once, in `_e29d_item` (post 4252).
- `started["C"]` is incremented at K3 (post 4344) and `done["C"]` in `_e29d_item` (post 4282); they stay consistent.
- B is queued once and runs once. Its record feeds attempt_1 through `select_attempts_e29d` → e28/e29 on the non-DIRECT records.
- attempt_2 = the DIRECT grid when valid and different from attempt_1.
- A-pass (C2) tasks are never early; `_e29d_can_start` returns c2_duck for them.
- DIRECT (branch 1) and B (branch 2) running together on one task share nothing:
  - sessions, E28_PLANS, meta and active are all keyed (task, branch);
  - `publish` replaces records per branch, and the notebook callback is keyed by branch and publish_seq;
  - reservation keys (t, 1) and (t, 2) are separate in `_e28_try_reserve`;
  - DIRECT writes under out_dir/direct/reports/<tid>;
  - LIVE output is print-only and does not depend on order.

**2. Edge cases:**
- A ends in TIME: `e28_b_phase` returns None, so no early item.
- B1 and B2 classes both get an early item (all 16 E28 tasks were B2).
- Window ending while an item is queued (X5): no crash, the item stays in left C, no record.
- Supply: an early item in qc counts as "queues not empty", the same as any C item. sup["c"] and sup["n"] are counted at take time (post 4335). Branches stay distinct (X6).
- `force` admission puts the head item through as before.

**3. Scheduling effect:** early items never starve A (A goes first, and a failed fit on the same 104k budget skips them). They are behind waiting B2 items. They do delay non-waiting B items; see HIGH-1 for the numbers.

**4. Off / scope C / no hook:** `_e29d_early_on` is False, K7's `ce` is False for every item, and K8's else branch is unchanged. That makes the order and records identical to before. These tests confirm it: TestSwitchOff.test_off_identity (strict), test_switch_off_no_early, test_c1_switch_off_parity, test_scope_c_no_early and test_no_early_when_not_admissible. The hooks are never called when OFF.

**5. Consistency — all OK:**
- SCHED_PIN = b88efa86 = the generated module = the Build file = the payload file.
- Notebook E29D_FILES (direct 8b42802c, nbx 531d18ec, sched b88efa86) = the payload folder files = SHA256SUMS. The SHA256SUMS keys equal the file set.
- Production notebook 9a097847… is 396,918 B; the smoke notebook 19928629… is 396,916 B. The only source difference is the E29_PROFILE line.
- The SUBMIT push folder's ipynb is byte-identical to the smoke notebook.
- The CPU "direct" kit carries the current e29d_direct_tests.py (a68c404a, includes TestEarlyDirect). Its sched tests and patch match Build.

## Test commands and results (PYTHONIOENCODING=utf-8, run from Build)
- `python -B e29d_direct_tests.py` → 34 run, 0 failures, 0 errors, 0 skipped.
- `python -B e29_rev2d_adversarial_tests.py` → 59 of 59 PASS.
- `python -B e29_sched_tests.py` → 29 run, 0 failures.
- **Negative control** (scratch copy `negctl\`, pre-2g e29d_sched_patch.py f2c60690): 34 run, 1 failure, 4 errors → fails as required (see LOW-3 for how).
- **Extra scenarios** (`extra_adv.py`, X1-X7, run on a scratch copy of current Build): 44 of 44 PASS.
- **Contrast** (`x4_neg.py` on pre-2g): t002 gets a Duck C2 and DIRECT is called only for t004.

## What I could not check
- Real GPU behaviour, and the DFlash per-stream tok/s at 24 streams. The replays assume 18 or 30 tok/s and the E28 durations.
- DIRECT's actual hit rate on train-fail tasks: no E31 results folder exists.
- Whether the KV reservation sized from budget 104k covers DIRECT's real need of 104k think + 8,192 answer + prompt (pre-existing, not examined).
- Whether the Kaggle CPU kits have run (no Kaggle API).

Everything I wrote is in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\verify_early\`: recon.py, sim_e28.py, sim_alt.py, sim_des.py, sim_12h.py, extra_adv.py, x4_neg.py, cons.py, and the copies cur\ and negctl\. No project files were edited.