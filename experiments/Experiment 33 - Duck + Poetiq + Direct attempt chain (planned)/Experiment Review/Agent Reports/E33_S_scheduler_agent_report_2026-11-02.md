## E33 attempt-chain scheduler diff: done

The diff is built and tested. **E33 suite: 38/38 pass, three runs in a row.** **Negative control** (the suite run against the unpatched E29 post): 23/38 fail, which is every behavioural test. **The E29 suites are unchanged:** e29_sched_tests 29/29, e29d_direct_tests 34/34, e29_rev2d_adversarial 59/59. **The E29 tree is untouched:** all 142 files are byte-identical before and after (sha256 lists compared), and its __pycache__ was not touched. There were no Kaggle calls and no commits.

### 1. Files
All three are new, LF only (0 CR bytes), written with write_bytes. Folder: `D:\ARC AGI 2\experiments\Experiment 33 - Duck + Poetiq + Direct attempt chain (planned)\Build\`

| File | Lines | sha256 |
|---|---|---|
| `e33_sched_build.py` (generator; `python e33_sched_build.py [--dry]`) | 956 | cc79756d75873347c8a832ae218a6facd119f45eee938f32192b61edaa8c88cf |
| `e33_sched_patch.py` (generated applier) | 872 | 1b4e84956952b70e49d6b79a080ab62de587848ee086bf039b26aeb5f7a61a05 |
| `e33_sched_tests.py` | 1054 | ca5b198918e0018e7a950fcfd61857611375382b153919739aeedd4f08a82011 |

A re-run of the generator reproduces the module sha exactly (this is also tested).

### 2. Shas
- **pre** (the E29 post arc2_duck.py): d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2.
  - Rebuilt exactly as `regen_sched()` does: payload 9e4b8928 → `e29_sched_patch.apply_e29_sched`, with the E28 and E29 Build dirs on the path.
- **post** (the E33 post): 4243797a3aefb7a503c4a638f8fc033e3d593b1d1b6abdf65a42fb50b4720307
- **patch**: f9d3137b8b4710aae3e7990df5a6fa28828b5f0284ebe99485194f19432c1400
- **applier** (E28): b871f0e409c5f44ad22a44cb61051138efc220657010b3a85ba4b53a72340e88
- **Diff size:** 15 hunks, 4 lines removed, 573 added. Every hunk starts after line 3685 (the E28 block start), so only the E28/E29 block changes.
- **How it applies:** strict n=3 unified diff, checked forward and backward with the E28 applier and through the module in text mode.
- **It is an alternative to the E29 DIRECT diff, not an add-on:** it applies to the same pre file, and `apply_e33` refuses the DIRECT post (tested).

### 3. Edits K1-K15 (each matches exactly once)
Every inserted line does nothing while `e33 is None` / `qe` is empty / no record carries an E33 tag.

- **K1 state:** `qe = []` (the chain queue) and `e33 = _e33_state(...) if _e33_on() else None`.
- **K2 candidates:** when E33 is on, `candidates()` returns `_e33_candidates(qe, meta, e33)` (the priority, section 5).
- **K3 take:** `where == "E33"` → `e33_take` (branch, queue position, busy mark).
- **K4 route:** after the supply `_e29_note_end`, E33 items go to `e33_route` (the next stage, or the chain ends).
- **K5 has:** `has = bool(qa or qb or qc or qe)`, so supply runs only when the chain queue is also empty.
- **K6 admit:**
  - `qlen` gets `len(qe)` appended, only when E33 is on.
  - A Poetiq/Direct item calls `_e33_can_start`. If it may start, it is counted like a Duck start (running, active, first/last start). If not, `e33_skip` runs: recorded, reservation released, chain ends, and the loop continues.
- **K7 run:** `s is None` → `_e33_item(...)` runs the hook in the worker slot.
- **K8 closures,** inserted before `def worker`:
  - `push_e33`: plan per stage. Duck stages get the E28 plan of A/B1/B2; Poetiq/Direct get the reservation budget.
  - `e33_seed`: the A:1 items in task order replace `qa`.
  - `e33_take`, `e33_route`.
  - `e33_skip(release=True|False)`.
  - `_e33_item`: hook call, release, the solver's own not-run → skip, publish, route.
- **K9 row tags:** Duck rows/records carry `e33_attempt / e33_stage / e33_name / e33_kind / e33_group / e33_from / e33_q1_left`.
- **K10 select:** `select_attempts_e28` sends records with E33 tags to `select_attempts_e33` (before the E29 supply rule).
- **K11 seed:** `e33_seed()` runs before the first admission.
- **K12 gauge:** the "queued" gauge adds `len(qe)`.
- **K13 slot src:** answer_slots maps E33 labels to branches (`E33_SRC`).
- **K14 heap_left:** adds `len(qe)`.
- **K15 report:** `SCHED_LAST["e28"]["e33"] = _e33_report(...)`.
- **The tail** (the E33 block at the end of the file) holds:
  - the constants, `_e33_on`, `e33_config`, `_e33_state`, `_e33_candidates`, `_e33_next`, `_e33_can_start`;
  - the train-pass, score and best-train helpers, `_e33_tag`, `_e33_cand`, `_e33_chain`;
  - `select_attempts_e33`, `_e33_report`.

### 4. Branch-id map
| Attempt | A | B (B1 or B2) | C (Poetiq it1) | D (Poetiq it2) | E (Direct) |
|---|---|---|---|---|---|
| 1 | 0 (= E28 A, facts view) | 2 (= E28 B) | 113 | 114 | 115 |
| 2 | 1 (= E28's attempt-2 branch) | 122 | 123 | 124 | 125 |

- **Supply under E33** starts at **200, 201, ...** per task (`sup["next_b"]` is seeded at 200). E29 supply only counts upward, so it can never reach an E33 id.
- **Reservation key** = (task, branch).
- **Phase labels** (rows, records, LIVE): `A:1, B1:1, B2:1, C-POETIQ1:1, D-POETIQ2:1, E-DIRECT:1`, and the same with `:2`.
- **Duck queue positions:** A:1 = idx, B:1 = 100000+idx, attempt-2 Duck = 200000+n, supply 300000+ (unchanged).
- **Views:** branch 0 uses facts; every other branch uses b1_view, as in E28.

### 5. Priority implementation
- `_e33_candidates` sort key: (attempt, 0 if a B2 that already waited else 1, stage rank A<B<C<D<E, group key, queue seq).
  - "Waited" = the E28 `meta[(task, branch)]["waiting"]` flag.
  - Attempt-1 group = (0, 0, task index).
  - Attempt-2 group = `e28_c_key(A:1 was TRAIN_PASS, A:1 tokens, idx)`: (1, 0, idx) for not passed, (2, tokens, idx) for passed. It is fixed per task for all attempt-2 stages.
- Fit, backfill and force-when-idle are E28's `pick()`, unchanged. Attempt-2 items only start when a slot is free that no attempt-1 item can use.
- Each row records `e33_q1_left`: the attempt-1 items still queued when it was taken. Tests show it is 0 for every attempt-2 start when no KV hook is set.
- Strictly sequential: a stage is queued only in `route` after the previous one ended. The `busy` set filters as a second guard.
- Attempt 2 is queued at the A:1 end, except on TIME. A TIME end queues nothing.
- Under E33, the E29 supply cap no longer counts a C still to come (`sup["c"]` gets the task at every E33 take), because E33 never queues a phase C.

### 6. Hook contracts (module globals, read at call time)
- **`E33_RUN_POETIQ(tid, task, plan, deadline, base_url, model_id, out_dir, iteration, prior_record)`**
  - iteration 1 = stage C, prior None.
  - iteration 2 = stage D, prior = a deep copy of the same (task, attempt)'s C record as the hook returned it.
- **`E33_RUN_DIRECT(tid, task, plan, deadline, base_url, model_id, out_dir)`**
- **`E33_CAN_START(stage, deadline) -> bool`**
  - `stage` is the bare letter "C" / "D" / "E". `e33_lib.stage_solver` maps "C-POETIQ1" to None, which would mean always admit.
  - A `(go, why, need)` tuple is read as its first item.
  - It is asked once per Poetiq/Direct item at admission, never for Duck stages.
- **Plans passed to the hooks** carry `attempt` (1 or 2), `branch`, `e33_attempt`, `e33_stage`, `budget`, `phase` and the group. This follows agent P's contract that attempt 2 must pass `attempt=2`.
- **out_dir:** `out_dir/poetiq/a{attempt}_it{iteration}` and `out_dir/direct/a{attempt}`.
- **Skip cases.** Each one ends that (task, attempt) chain, is recorded in the row and in the per-stage `skipped` counts, and is never a Duck fallback or a crash:
  - no run hook → `no_hook`;
  - no `E33_CAN_START` → `no_can_start_hook`;
  - `E33_CAN_START` returns False → `late`;
  - `E33_CAN_START` raises → `hook_error <type>`;
  - the run hook raises, or returns None / False / a non-dict → row `NO_RECORD`, nothing published;
  - per agent P: a record with status other than "done" (`not_admitted` / `not_run`), or `poetiq.admitted is False` → a skip after start, not published, reason = the status, `skipped_after_start` counted.
- **Published records** keep the solver's shape (`submissions[{test_outputs, accepted, train_exact}]`, `poetiq` / `e31` blocks) plus `branch`, `task_id`, `e33` = row and `e28` = row; each submission gets `branch`.
- **Train pass:**
  - Duck = kind TRAIN_PASS;
  - Poetiq = `rec["poetiq"]["train_pass"]` or `rec["accepted"]`;
  - Direct = never.
- **Switches** (env; exactly "1" = on, anything else = off; the stage is then left out of the chain, e.g. Poetiq off gives A → B → E):
  - `E33_SCHED` (with `E28_SCHED=1`), `E33_POETIQ`, `E33_DIRECT`, `E33_ATTEMPT2`;
  - reservation budgets `E33_POETIQ_BUDGET` (default 68,192) and `E33_DIRECT_BUDGET` (default 112,192).
- **Module API:**
  - `apply_e33` / `verify_e33` / `revert_e33` / `tree_state` ("E29_POST" | "E33_PATCHED" | "OTHER");
  - `server_env(chain)` builds the env from the `e33_lib.check_chain` dict;
  - `cell3(e12_dir, E28_SCHED_REC, rec, chain)` never raises: it clears every E33 env name first, sets env only after apply + verify, and reverts on failure;
  - `ENV_NAMES`, `BRANCH_MAP`, `SUPPLY_BRANCH_BASE`.
  - `build_e33.py` already calls `cell3(..., E33_CHAIN)`.

### 7. Answer rule (`select_attempts_e33`; per test K, valid 0-9 grids only)
- **Records are grouped** by their E33 tag into attempt 1 / attempt 2 per stage (latest publish_seq wins). Supply records (phase S*) join the attempt-1 pool after D, in branch order.
- **Candidates per record:**
  - Duck = `_e28_cand`.
  - Poetiq = accepted submissions first, then the first valid grid. It is full if that submission is accepted, or if the record passed and it is the first submission. A None or invalid test grid falls through, which covers `test_usable` False. Match = train_exact fraction, else `poetiq.train_score` (else `score`).
  - Direct = the first valid grid, no match.
- **`_e33_chain` for one attempt** = the E28/E29 best-train loop over A, B, C, D (+ supply): full beats non-full, otherwise a strictly higher fraction wins, ties keep the earlier. This is step 1 (pass) or step 2 (best). Then the rest by (full, fraction, earlier), then the Direct grid (step 3).
- **attempt_1** = the attempt-1 chain's head, else 4 `select_attempts_r5` over all Duck records (the E28 fallbacks), else 5 the test input.
- **attempt_2:**
  - the attempt-2 chain's head when it differs from attempt_1 (step `a2:<n>`);
  - else the next distinct attempt-1 candidate (`a1-next`);
  - else the next distinct fallback, else the input, else "same".
- **Slot rows** per query: source label, `a1_step` / `a2_step`, train, and both chain orders. The report's answer_slots map the labels back to branches.

### 8. Test results (`e33_sched_tests.py`, 38 tests)
Uses the E29 harness ST (real arc2_duck, real BranchSession, scripted fake Duck, Env/RunLog/ResKit, `_solve`, `_norm_*`) with stub hooks.

| Class | Tests | What they cover |
|---|---|---|
| TestPatch | 7 | shas and states; strict diff; dir and file modes; refusals (ORIGINAL / E28 / DIRECT post / CRLF / one-character change); hunks and LF; the generator reproduces the file; `server_env`; `cell3` |
| TestChain | 7 | full 20-item order on 1 worker; stop at the first pass for A/B/C/D, both attempts; B1 vs B2 (budget 120000, window 130560, timeout scaled) vs straight-to-C vs TIME (no attempt 2); attempt-2 groups/order [t001, t003, t002, t004, t000] and group keys; switch items; pure candidate order including waiting-B2 and busy; next-stage table |
| TestConcurrency | 3 | 4 workers × 16 tasks; 4 workers with the real NB5 Reservation (73 blocks; no leak; grants + forced = starts + skips); A:2 runs during B1:1 of the same task. Invariants checked: one start / done / release per (task, branch), no overlapping stages per (task, attempt), attempt-2 `q1_left` = 0 |
| TestHooks | 5 | no Poetiq hook; no Direct hook / no CAN_START; CAN_START False / raising / tuple; run hook raises or returns None / False / non-dict; solver not_admitted / not_run / admitted False at C and D. Each ends the chain, publishes nothing, releases once, runs only Duck branches {0, 2, 1, 122} |
| TestWindow | 2 | window closes mid-chain (D left 2, `skipped_for_time` 2, no start after the window); late admission skips D, chain ends |
| TestAnswerRule | 4 | attempt_1 cases 1-5; attempt_2 cases (attempt-2 Duck / Poetiq / Direct, a1-next, fallback, input, same); supply and multi-test (invalid grid at test K); untagged records keep the E29 rule |
| TestMultiTestE2E | 1 | two-test end-to-end run |
| TestSupply | 1 | 1 and 2 workers: supply only when every queue including qe is empty (`queues_at_start [0,0,0,0]`), branches 200-203, cap 14 counts E33 items, a Poetiq pass stops supply |
| TestOffParity | 6 | against the E29 post with hooks and sub-switches set and E33_SCHED unset / "true" / "0", supply on and off: 1 worker strict; 4 workers; KV reservation; no admission window; E28 off = the original rev 6; real K100/W4 record selection. Hooks are never called and there is no `e33` key |
| TestAgentAndLive | 2 | real Arc2ToolAgent on branches 122 (B2:2) and 1; E29 LiveLog prints the E33 labels |

- **Negative control** (`E33_TEST_NEGCTL=1`): 13 failures + 10 errors = 23. The 15 tests that pass are TestPatch 7 + TestOffParity 6 + the untagged-rule test + the agent test; none of them depends on E33 code in the module under test.
- **E29 suites** (run from the E29 folder): 29/29, 34/34, 59/59 pass.

### 9. Contract facts from agent P (e33_poetiq.py 397acb4c), as built
- It always returns a record. Status "not_admitted" / "not_run", or `poetiq.admitted` False, is treated as a skip and the chain ends.
- `plan["attempt"]` and `plan["branch"]` are in every plan.
- prior = the same (task, attempt)'s iteration-1 record.
- Train pass = accepted = `poetiq.train_pass`.
- `poetiq.train_score` is the match.
- None test grids are skipped per test.
- The notebook builds `E33_CAN_START` from `can_admit`.

### 10. Risks and what I could not do
- **Direct attempt 2 may repeat attempt 1.** In `e33_nbx_src.py`, `e33_run_direct` seeds by `task_index` only, not `plan["attempt"]`. E′ would then run with E's seeds and could give the same output. The fix belongs on the notebook side; it is not mine.
- **Step 2 is literal E28, so Direct rarely gets attempt_1.** A Duck or Poetiq candidate with 0 exact train pairs still beats the Direct grid, because Direct only reaches attempt_1 when no other candidate exists.
- **Attempt-2's lower candidates are skipped.** When attempt-2's head equals attempt_1, the rule goes to attempt 1's next candidate, as the spec says. Attempt-2's other candidates are only reachable through the r5 fallbacks, which are Duck-only.
- **Supply records join attempt 1.** I put them in the attempt-1 pool after D; the spec says nothing about supply answers.
- **Forced Poetiq/Direct items run without a reservation.** With the real Reservation the idle grant means this does not happen in practice.
- **A Poetiq/Direct hook still running at the join** leaves no row. The solvers' own hard stop should prevent this.
- **The Duck run tag `stage` is 3 for every E33 label.** It is record-only.
- **Not done here:** no Kaggle CPU kit or GPU run (outside my scope). The other agents' files need the new patch sha 1b4e8495 if they pinned an earlier draft.
- **Not my file:** `Build/__pycache__/e33_sched_build.cpython-314.pyc` (09:40) was written by another agent's process importing my builder.
