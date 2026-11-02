## E31 Flash-Next DIRECT 30: build report (nothing pushed, no datasets uploaded)

The notebook and its three CPU kits are built. All design changes and every item of the consolidated audit fix list are applied. Local results:
- unit tests: **80/80 PASS** (Python 3.14)
- units-kit layout simulation: **75/75 PASS**, verdict OK
- fake-server smoke with the solver: passed, including a server kill and relaunch

The e2e-commit and real kits have not run on Kaggle yet.

### 1. Files (all in `D:\ARC AGI 2\experiments\Experiment 31 - Flash-Next Direct measurement (planned)\`)

| File | sha256 | Notes |
|---|---|---|
| `Experiment Notebooks\Q27-XHIGH-E31_FN_DIRECT30.ipynb` | `1939954665c723dcf9321ac383cd1f9fdf15c00b483782936223baf91661e55b` | **394,400 B** (5.6 kB under 400 kB), 10 cells (markdown + 9 code) |
| `Build\build_e31_direct.py` | `ab2a08a3…` | builder |
| `Build\e31_direct.py` | `8041a2bb7f7d32b5462642c12264b3a5c9f50e4606d379308aefbc61ca70e508` | the direct solver |
| `Build\e31_code_export.py` | `7f561d41dac99a31a3ede11c7c31bf6b882b0604e7acf8e7a2ef2a4bd527a8d5` | readable cells |
| `Build\e31_unit_tests.py` | `644adc3e…` | |
| `Build\e31_build_facts.json` | `89878a51…` | |
| `Build\build_e31_cpu_kit.py` | `cce4fcc4…` | |
| `Build\e31_kit_facts.json` | `e5bbe5a0…` | |
| `Experiment Review\Audit Zips\E31_FN_DIRECT30_audit_FLAT_2026-11-02.zip` | — | 398,924 B; all of the above, the fix list, both spec reports, the three local test logs |

**GPU push folder (written, not pushed):** `D:\ARC AGI 2\_scratch\push_e25\q27-xhigh-e31-fn-direct30__esfersami50\`
- id `esfersami50/q27-xhigh-e31-fn-direct30`, NvidiaL4, private, no internet
- same datasets, models and docker image as E28 K100

**Kit folders:** all on esfersami50, CPU, the pinned image.

| Kit | Folder (`_scratch\push_e25\`) | sha256 | Size |
|---|---|---|---|
| units | `q27-e31-fn-direct30-kit-units__esfersami50` | `4280ee779b91f005…` | 208,816 B |
| e2e-commit | `q27-e31-fn-direct30-kit-e2e-commit__esfersami50` | `8909654cf6049256…` | 212,177 B |
| real | `q27-e31-fn-direct30-kit-real__esfersami50` | `7bb87414048eb455…` | 281,844 B |

### 2. The E31_CFG line (first code line of CELL 1)
```
E31_CFG = {'think': 104000, 'answer': 8192, 'window_min': 170, 'workers': 24, 'temperature': 0.6, 'drain_min': 23}
```
- It is a plain literal and is not inside any packed or sha-locked text, so the user can change it without a rebuild.
- `drain_min` was added because audit E31-3 asked for it.
- `E31_CFG_JSON` can override it only when `E11_HARNESS=1` (the kits use this).

### 3. Notebook design (derived from the E28 K100 notebook d3d132da, read and sha-pinned)
**Unchanged byte for byte** (checked against the E28 cell shas): CELL 0 (E18 R2), CELL 3, the E10 task cell, CELL 5 (Merlin klee100 boot and all gates), CELL 6/7, CELL 8, CELL 9.

**CELL 1 edits:**
- **D1:** the E31_CFG line; `E31_BUDGET_MIN = 15 + window_min + 20` = 205.
  - The deadman is written as two lines: the literal `K100_DEADMAN_SEC = 12300` (the kits' regex reads it), then `= int(E31_BUDGET_MIN*60)`, so it follows any change to E31_CFG.
  - `NB5_DEFAULT_PROFILE` and `K100_PROFILES` are `E31DIRECT30` / `RERUN_PROD`.
- **D2:** `E31_GROUPS` / `E31_TASKS` (30 tasks).
  - `E31_LEVERS` = `E28_LEVERS` without `e28_sched`, so Duck's arc2_duck.py stays the original payload file.
  - The `E31DIRECT30` row: source eval, `task_list`, phase `Q_E31_FN_DIRECT30`, fixed window = `window_min`, used as a cap.
  - `RERUN_PROD` = E28's row with `E31_LEVERS`, so the hidden rerun sends every test task through the same solver with the E15B writer.
- **D3:** one sha-pinned lzma+b85 literal holding `e28k100_lib.py` (unchanged, ff923ef2) and `e31_direct.py`. The E28 scheduler module is dropped.
- **D6 (size):** comment-only lines were removed from two Duck-configuration regions. Every code line is unchanged (asserted).

**New CELL 7b:**
- Rebinds `E12A.solve_all_r6` to the direct solver (same signature, return value and record contract).
- Rebinds `E12A.build_submission` to a DIRECT-only adapter.
- Asserts `SGL_URL` equals the booted server's URL (`E11_SERVER.url`, port `NB5_PORT`), and that `base_url == SGL_URL + "/v1"`.
- Workers = min(e21_cell6 workers, `E31_CFG['workers']`).

**Builder self-checks:**
- It reproduces the E28 notebook bytes from E28's own cells.
- The E28 readable cells equal the E28 export.
- The embedded per-task reference table equals the Exp 28 results file.
- Window math: a boot at 15 or 14.65 min keeps the full 170-min window.
- The drain reserve check: 3 tests × 8,192 tokens ÷ 18 tok/s = 22.75 min ≤ 23.

### 4. The direct solver (`e31_direct.py`)
**One unit per task** (30 requests, not 46).
- Single-test prompt: ORDER2's `build_user_message`, verbatim (unit test compares against an embedded E10 fixture and the E10 notebook text).
- Multi-test prompt: `FINAL INPUT k:` blocks, and the closing asks for labeled ```` ```grid k ```` blocks ending "Output nothing after the last block." Every other sentence is ORDER2's.

**Request flow:**
- Render with the checkpoint's own template (xhigh, enable_thinking True) via `NB5_LOCAL_TOK` (pinned hashes verified), then token ids.
- Requests go to streaming `/v1/completions`. This is E18's token-id client only: no E18 relaunch, and the URL is this server's.
- Every sampling field is sent: T 0.6, top_p 0.95, top_k 20, min_p 0, presence/frequency 0, repetition 1.0, seed, n 1, ignore_eos False, stream, return_token_ids, skip_special_tokens False.

**Budgets and seeds:**
- Phase-1 cap = min(think, 131072 − prompt − answer − 256).
- The real 30 prompts are 1,804–10,301 tokens, so every task keeps the full 104,000 cap and none is below the 60k floor. The below-floor case is recorded, never gated.
- Phase 2 = prompt ids + phase-1 ids (trailing EOS stripped) + the 25 STOP_IDS (forced close), or "continue" when `</think>` closed but the answer was cut. Cap = min(answer × N tests, room). Ids only, never re-tokenized.
- ORDER2 order; seeds = SEED + 10 × sorted index + phase.

**Window phases inside the solve end T** (audit E31-3):
- No admission after T − 12 min.
- At T − `drain_min`: phase 1 is aborted (streamed ids kept) and a forced-close phase 2 starts.
- Every request is aborted at T − 1 min and the partial answer is parsed.
- The solver **returns as soon as every task is done**. e21_cell6 and nb5_phase do not sleep to the window end (I checked the code), so the notebook goes straight to CELL 7, CELL 9 and the server kill.

**Request errors** (audit E31-6): an error is never counted as a finish.
- The unit waits while `E12A.SERVER_DOWN` is set or `/health` fails, then continues: a phase-1 error goes to forced close, a phase-2 error resends phase 2.
- At most 2 retries; after that the task ends with end_reason `error`.
- The E15B watchdog / `e21_relaunch` stay the only owner of the server lifecycle.
- This bounded retry is something I added to meet E31-6. It is not in ORDER2.

**Parser** (audit E31-1):
- Single test: ORDER2's `parse_grid`, unchanged.
- Multi test: if any `grid K` label is present, mapping is by label only (the last valid occurrence wins). Positional mapping only when there are no labels and exactly N valid blocks.
- A test with no grid gets ORDER2's fallback (test input / mirror) for that test only.
- Each record carries `query_index`, `mapping_source` and `fallback_reason`; the query→grid mapping is asserted before scoring.

**Adapter** (audit E31-2): `build_submission_direct` uses only the direct record's ORDER2 `make_attempts` per test. A task with no record gets the identity. No CPU rule or Duck candidate ever enters.

**Records:**
- One completed branch-0 record per task through `on_branch_done` (E28 contract: `submissions[{test_outputs}]`, `t0`/`t1` epoch, `publish_seq`).
- `E12A.SCHED_LAST` is filled: gauges, workers, the `e31` section.
- `q27x/e31_direct_records.jsonl`: one row per test input.
- `q27x/e31_direct_report.json` contains:
  - per group (main / partial / control / all / multi_test / single_test): query_hits, fully_solved_tasks and task_norm, kept separate;
  - coverage at 60k / 78k / 104k;
  - modes, end reasons and mapping sources;
  - speed: solve wall seconds, aggregate decode tok/s, per-request tok/s, TTFT, server counter deltas (generated, prompt, prefix hits, fresh prefill, preemptions), sampler tok/s;
  - the comparison against K100 / W4 / Direct27B / Duck, and the E28 reference (419.5 tok/s, 20.5 M fresh prefill).
- Full thinking of every request is saved to `phase_*/duck/<task>_b0/transcript.txt` + `token_ids.json` and is in the Duck bundle. I never opened any trace.

### 5. Timeline estimate (30 requests, 24 streams)
- **Boot:** about 15 min (E28 real: window start at +14.65 min).
- **Solve:** wave 1 is 24 tasks; a 104k trace at ~18–20 tok/s per stream takes about 87–96 min. The other 6 start as slots free up and run faster (~25–30 tok/s). Expected solve wall about 95–125 min. Worst case (all 30 at the cap): the drain at T−23 min (147 min) forced-closes the last ones and the solve ends by 169 min.
- **After the solve:** reports and bundle about 3 min.
- **Total:** typically about 2 h 00 – 2 h 25. Worst case about 3 h 08. Hard cap 205 min (HARD_STOP 200, solve-end cap 197); deadman at 205 min.

### 6. Tests
**Local unit tests: 80/80.** They cover:
- provenance and that unchanged cells match E28;
- the E31_CFG line and the harness override;
- budget, profile and window math;
- prompt bytes equal ORDER2 on all 15 single-test tasks, exact multi-test bytes, and b6f77b65 (3 tests);
- the real klee100 tokenizer render equals the spec text; STOP_IDS 25 with 248069 at index 23; EOS [248046, 248044];
- phase-1 caps;
- ORDER2 parser self-tests plus 18 multi-test parser regression cases;
- make_attempts;
- the adapter with 1/2/3 tests, partial, no grid, grid equal to input;
- the notebook's own CELL 7 scorer and the E28 `score_from_generated` agree;
- "DIRECT wrong + CPU rule right → scored wrong", contrasted with the real payload `select_attempts_r5`;
- CELL 7b's rebind, its refusal of port 8011, and the E15B writer's rerun overlay;
- the solver against an in-process token-id fake server:
  - early return;
  - forced-close prefix exact;
  - continue and EOS strip;
  - label mapping;
  - an error that recovers;
  - a drain of 24 streams;
  - the hard abort;
  - the admission stop leaving not-admitted tasks at the identity;
  - an unrecovered error ending as `error`;
  - SERVER_DOWN holding admissions;
- the request fields against the Merlin source.

**Units-kit layout simulation:** the kit's own blob was unpacked, Duck was taken from the payload zip, the pinned tokenizer was used, with the kit's exact env. Result **75/75**; the only skips are the two allowed ones (E10 notebook, Merlin source).

**Fake-server smoke** (the e2e kit's own fake, 30 tasks, 24 workers):
- **Normal run:** 30 finished in 5 s (23 natural, 7 forced close), all 7 forced-close prefixes token-exact, 24 parallel streams on the fake.
- **Kill + one relaunch:** 30 finished, 9 request errors, 9 retries, 0 error finishes, returned early, prefix-exact checks recorded across the kill.

**Kits** (built only, not run):
- **units:** the E31 unit tests, plus the tf-ref, payload, ds-pin and Duck pre-sha checks; the klee100 tokenizer is taken from the mounted Model.
- **e2e-commit:** the E28 harness with my E31 checks.
  - Harness cfg: think 6000, answer 1024, window 50, drain 3; budget 85 min.
  - The fake is extended for direct streaming: never-closing thinking, partial and labeled grids, a kill at the 33rd direct stream.
  - Checks:
    - one E15B relaunch;
    - CELL 7, the E28 report VALID, the E31 report and records;
    - the DIRECT-only generated attempts;
    - sampling fields, port, 24 streams;
    - the deadman, CELL 0's fallback, E18 R2 CELL 0 / writer equality;
    - the klee100 merge and boot gates.
- **real:** the E28 real kit with the scheduler check changed to "Duck original, no e28_sched" in both the commit and rerun namespaces. E31 additions:
  - CELL 7b run on the real merged klee100 tokenizer after CELL 8's own `NB5_LOCAL_TOK` statement;
  - the request body through the installed Merlin `CompletionRequest.to_sampling_params`, with the server's override defaults;
  - `CKPT/generation_config.json` EOS check.

### 7. Verified vs assumed, open risks
**Verified locally:**
- The Merlin 3.0.0 / vLLM 0.32 source at `_scratch\e23\p2\merlin-src` has every request field, streams `token_ids` when `return_token_ids` is set, and runs no reasoning parser on completions.
- The tokenizer fixture sha equals the pinned klee100 TOK_SHA.
- The klee100 `generation_config.json` (from the upload-fetch copy) has EOS [248046, 248044].

**Assumed:**
- That `merlin_offline_tar.bin` is this same Merlin source. The real kit checks the installed tree.
- Per-stream speeds and the tail effect. 24 long contexts exceed the 1.65 M-token KV pool, so preemptions are expected and accepted, as in E28.

**Open risks:**
- No relaunch happens if a crash leaves under ~35 min before the solve end; pending tasks then end at the hard stop.
- No e2e-rerun kit was built. The rerun overlay is only unit-tested.
- Duck-oriented labels in nb5/E28 tables are record-only: outcomes may read FAILED_OR_INTERRUPTED / BUDGET_END, the tracer is empty, and running_check may flag.
- The error-retry rule (bounded at 2) is my own addition for audit E31-6.
- E2E-commit and real kits have not run on Kaggle.

**Hygiene:**
- No files outside the E31 folder and my kit folders were changed by me. The `k100q_ref27b.json` / `nbs1_*` timestamps in E25 Build come from another builder's run at 00:08.
- The `__pycache__` my smoke created in E31 Build was removed.
- All files are LF.
