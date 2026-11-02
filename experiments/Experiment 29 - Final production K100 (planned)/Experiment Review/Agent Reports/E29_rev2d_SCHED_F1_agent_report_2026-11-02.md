## F1 / F1+ is fixed: a late phase-C item now runs the original Duck C session instead of a DIRECT no-op

I regenerated the patch, and **20/20 tests pass**, run the way the kit runs them. Nothing was pushed or uploaded, and I edited none of E29-B's files.

### What changed in the patch (`e29d_sched_patch.py`)
I rebuilt it from `e29_direct_build.py` against the pinned E29 post `arc2_duck.py` (d8603109). It still applies forward and reverses exactly with the E28 applier. `verify_direct` reports `DIRECT_PATCHED`; that is this module's name for the fully patched state.
- **K1:** a new module hook `E29D_CAN_START = None`, next to `E29D_RUN`. Its comment: `(deadline) -> bool; DIRECT only when True, else the original Duck C`.
- **K3 (the phase-C dispatch):**
  - The DIRECT branch is taken only if `p["phase"] == "C" and _e29d_on()` and `_e29d_can_start(deadline)` returns True.
  - `_e29d_can_start` returns True only when the hook is callable and returns truthy for the scheduler's monotonic deadline.
  - Otherwise it returns False with a reason: no hook or not callable → `no_hook`; False → `late`; an exception → `hook_error <Exc>`. It never raises.
  - On False, the same item falls through to the original `sess(tid, p, qpos)` Duck C path, which is unchanged.
- **Record-only counters:** I added these so the smoke can tell the two routes apart.
  - In `e28["direct"]`: `duck_c` (a count), `duck_c_why` (`{why: n}`) and `duck_c_tasks` (`{task: {why, sec}}`).
  - The `direct` dict is now created by one helper, `_e29d_stats`, used in both K3 and K5. It has the same 5 keys plus these 3.
  - This goes slightly beyond "keep everything else identical"; with the switch OFF it never runs, and the OFF parity tests still pass.
- Everything else in the patch is identical: K2, K4, the rest of K5, K6, K7, the answer rule, `cell3`, and apply / verify / revert.

### Changes to `e29_direct_build.py`
- The new edits above, plus updated docstrings.
- A standalone regenerate command: `python e29_direct_build.py --regen-sched` (add `--dry` to only print the shas). It does exactly what `build_e29_final.build_direct()` does for this file: same inputs, same forward / backward and module checks.
- I checked it against the old version: the old builder through this path reproduces the old file cfb5834c byte for byte.
- `make_direct` is untouched, so `e29_direct.py` is still 8b42802c.

### Shas
| Item | New | Old (rev 2c) |
|---|---|---|
| `e29d_sched_patch.py` | `4cc807f24523b6dfd9b7bbf7b6ab2b78f2a225ebdb8401b414dace4a07db7d92` (18,394 B) | cfb5834c… |
| pre (unchanged) | `d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2` | same |
| post | `937e6cac97f87d6d8991728951d1ce5690fef1fef6d66a1afec4fc82c68901e9` | ef3895a3… |
| patch | `a34a5d6d9e6e5cf0438425c79b783ebd7eb4f24e62a3965af3b196f6e8791660` | 035088d9… |
| `e29_direct_build.py` | `5433f414c51d40d67bf039055c24dcf705d9d5e2a4e5808acd088edb96ba9ef2` | f80cb5c4… |
| `e29d_direct_tests.py` | `cdc46ed492b79bdedc52f9aac945fb220157a5abb524ecf1d2630e41a8c6ee40` | 00628af6… |

All three files are LF (0 CR).

### Tests (`e29d_direct_tests.py`: 12 → 20)
**Existing tests:** `TestCOrder` and `TestE2E` now set `E29D_CAN_START` as the notebook will; `TestE2E` uses the real-clock formula. `test_off_identity` now also sets both hooks with the env unset and asserts neither hook is called.

**New class `TestF1LateC` (8 tests):**
- **Formula and boundary:** the notebook formula (drain 26, margin 60 s) on a fake clock: 31 and 27.5 min → DIRECT; 20, 1, 0 and −3 min → Duck. The exact boundary (1,620 s, strict `<`) → Duck; one second more → DIRECT. Also covered: hook None, a non-callable hook, a raising hook, and the real-clock lambda.
- **31 min left:** all C items go to DIRECT. The hook is asked once per item, always with the scheduler's deadline, and attempt_2 is the DIRECT grid.
- **20 min and 1 min left:** all C items run Duck C. Records, answers and `SCHED_LAST["e28"]` (minus the `direct` counters) equal today's E29 post run.
- **Hook None or raising:** Duck C, with parity to today and the reason recorded.
- **Switch OFF with both hooks set:** the hooks are never called, there is no `direct` key, and records, answers and the e28 record equal today's.
- **Mixed (1 worker):** the first 3 C items in C order go to DIRECT and the last 3 run Duck C. attempt_1 never changes.
- **Mixed (4 workers):** the same invariants hold with C items running at the same time.
- **Real solver, late window (drain 90 s > the 60 s window):** the rev 2c route is reproduced (6 DIRECT calls, 0 records, 6 no_record, 0 Duck C). The fixed route runs Duck C everywhere with today's answers. An early window still gives 6 DIRECT records.

**Checked on every path:** `E28_RELEASE((task, 1))` is called exactly once per task, `done.C` and `started.C` are each 6 for 6 items, every task has exactly one branch-1 record and one C row, and no item starts twice.

**How I ran them (the kit way):** the files in a `build/` layout with `e28dir/e28_sched_patch.py`, env `E29_TEST_E28_DIR`, `E28_TEST_PAYLOAD` (Duck payload 9e4b8928) and `E28_TEST_TMP`, without `-I`.
- Py 3.14.6: 20/20, four separate runs.
- Py 3.12.10: 20/20.
- Negative control: the new tests against the old rev 2c patch give 6 failures and 1 error, so they do catch the bug.

The `!!!!!!!!!! E29 DIRECT diff not applied` line printed during `test_cell3_step` is expected: that test feeds in a bad file on purpose.

### Handoff for E29-B
Written to `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Build\SCHED_DONE.json` (sha ae4c6cf8…), with `"tests": "20/20 PASS"` and `"hook_name": "E29D_CAN_START"`. What E29-B needs to do:
1. In `e29_direct_setup`, set `E12A.E29D_CAN_START = lambda deadline: time.monotonic() + 60.0 + float(E29_DIRECT2['drain_min']) * 60.0 < deadline` where `E29D_RUN` is set, and set it to None wherever `E29D_RUN` is cleared (OFF or failure). If the notebook never sets the hook, every C item runs Duck C and DIRECT never runs; that is safe, but it means DIRECT is silently off.
2. Rebuild. `build_direct()` will produce the same bytes (4cc807f2) from the updated `e29_direct_build.py`. Then update the payload dataset file, `SHA256SUMS` and the pins in the facts files and the code export, which still reference cfb5834c / ef3895a3.
3. If `e29_kit_lib_tests.py` (around line 554, the fake E12 class) checks the hook state, add `E29D_CAN_START = None` there.
4. The direct kit's `direct_tests_min` will become 20 automatically, because it counts the test functions in the file.

All my scratch copies, including the rev 2c backups, are in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\sched\`.