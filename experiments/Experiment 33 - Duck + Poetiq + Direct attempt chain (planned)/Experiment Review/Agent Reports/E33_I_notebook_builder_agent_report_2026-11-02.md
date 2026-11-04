## E33 notebook builder: final report

The E33 notebook is built, integrated with the final agent S and agent P files, and every test suite passes. The E29 production files are byte-identical. One blocker-class risk stays open: the 40-minute E29SMOKE window can never start a Poetiq or Direct stage, so a smoke on the current profile would not test the new paths (risk 1).

### 1. Files and shas (all LF, 0 `\r`)

**Built notebooks** (folder `D:\ARC AGI 2\experiments\Experiment 33 - Duck + Poetiq + Direct attempt chain (planned)\Experiment Notebooks\`)

| File | sha256 | Bytes |
|---|---|---|
| `Q27-XHIGH-E33_CHAIN.ipynb` | 0a0172d67ad84ff4ae66760f4b64c985c4d85bebb288edfe415bc0df9c451f3c | 398,100 (< 400,000) |
| `Q27-XHIGH-E33_CHAIN_SMOKE.ipynb` | 9a7b7e65ac05d671ea0124c2891e1ee8c2abbf60dd88cb1afd10da31b58ca3c4 | one line differs from the production notebook |

**My files in `...\Experiment 33 ...\Build\`**

| File | sha256 | What it is |
|---|---|---|
| `build_e33.py` | 46444ebd6401f5e6dec69f69ef6d9cb367c4d5bb88a739ed997d2b9af23201c6 | the thin wrapper |
| `e33_lib.py` | 7bd6cb9f5f3abaabdca98d3f6c812921ba3140d34a110b6d155fef81b202cd4e | payload module: the switch checker and the Direct admission rule |
| `e33_nbx_src.py` | 45018ea7ca6db153d6218e3d7b24c6c72c04ee8125811b08c1857131d94d43bf | source of the notebook-side functions |
| `e33_nbx.py` | 8c3c15089f32cabd8aff68702bca2c48e976fb3f8d9a79560d3eae8319573bb8 | generated payload module |
| `e33_nbx_tests.py` | 0339f119708eccf6b183137a5668a753557536db033cb6a8b5955ca8c323c058 | quick local tests |
| `e33_chain_code_export.py` | af33b9b1751f129000b893665910afce4dddaec6864f5474a33edcee473a4fff | the readable code export |
| `e33_chain_build_facts.json` | abf8d7bcb52737f3e67eaaf734f46ba530df5a19b7b37455f4ba94216ebee6e4 | build facts |

`e33_nbx.py` is built from three parts:
- E29's `e29_live.py` LiveLog class, verbatim;
- E29's `e29d_tok`, verbatim (cut from `E29_DIRECT_FNS` and asserted present in E29's `e29_nbx.py`);
- `e33_nbx_src.py`.

**Push folders** (`D:\ARC AGI 2\_scratch\push_e25\`, owner foysalemonshanto, nothing pushed)
- `q27-xhigh-e33-chain__foysalemonshanto` holds the 0a0172d6 notebook.
- `q27-xhigh-e33-chain-smoke__foysalemonshanto` holds the 9a7b7e65 notebook.
- `q27-xhigh-e33-chain-submit__foysalemonshanto` holds the smoke bytes (9a7b7e65).
- **Kernel metadata:** each folder has a `kernel-metadata.json`.
  - Datasets: `q27-e25-nb0-tf-ref`, `q27-e15b-payload`, `q27-e29-final-payload`, plus the new `q27-e33-chain-payload`.
  - Models: E29's four models, none removed: klee100 3bpw-mtp, fp8ple-overlay, e23-merlin-env, dflash2v2 bf16-embed.
  - Docker image, NvidiaL4, GPU on, internet off: same as E29.

**Payload dataset** `D:\ARC AGI 2\_scratch\push_e25\ds_q27-e33-chain-payload\` (also has `SHA256SUMS` sha c108a153… and `dataset-metadata.json`, id `foysalemonshanto/q27-e33-chain-payload`, private, CC0)

| File | sha256 | Bytes | Source |
|---|---|---|---|
| `e29_direct.py` | 8b42802c… | 62,177 | E29's own bytes |
| `e33_lib.py` | 7bd6cb9f… | 7,924 | mine |
| `e33_nbx.py` | 8c3c1508… | 35,362 | mine (generated) |
| `e33_poetiq.py` | 397acb4c… | 87,749 | agent P final |
| `e33_sched_patch.py` | 1b4e8495… | 51,565 | agent S final |

The `E33D_FILES` pins in CELL 1 equal these shas; the build asserts this and reads each file back from disk.

### 2. Wrapper edits (`build_e33.py`)

**How it builds**
- It imports `build_e29_final` as a module and never calls its `build()`, `write_*` or `write_dataset`.
- `F.build_direct()` runs with `F.HERE` pointed at the scratch folder `D:\ARC AGI 2\_scratch\e33_nb\_e29_build_direct_redirect\`. Each of its three outputs is asserted equal to E29's file on disk.
- It then rebuilds E29_DFLASH_D3S in memory and asserts the bytes match 9a097847, and checks the two E29 notebooks on disk.
- The marked E33 edits are swapped in for E29's DIRECT texts and restored afterwards.

**The once-only edits**

| Edit | Where | Change |
|---|---|---|
| W1 | CELL 1, top | the `E29_DIRECT2` line becomes the `E33_CHAIN` literal; `E29_QSA`, the `E29_PROFILE` line and `E29_SMOKE` stay unchanged |
| W2 | CELL 1 | E29's `E29D_FILES` + `direct2_cfg` block becomes `E33D_FILES`; `e33_lib.py` loaded from the payload; `E33_CHAIN = E33L.check_chain(E33_CHAIN, E33_CHAIN_JSON only under E11_HARNESS)`; a lib miss turns the chain off |
| W3 | CELL 3 | E29's DIRECT diff step becomes `E33DS.cell3(E12_DIR, E28_SCHED_REC, E33_SCHED_REC, E33_CHAIN)` from the payload; `CONFIG["e33"]` recorded |
| W4 | engine | `e29_extras()` runs `e33_nbx.py` (name + sha), then `e33_setup()` and `e33_live_setup()`; a miss pops the chain env, clears the hooks and records `extras_error`; its last line is E29's `BLOCK_END`, so E29's engine pack still works |
| W5 | CELL 9 | the E33 chain report block goes before the report write: `E28_OUT["e33"]` and a `[E33 CHAIN REPORT]` print |
| W6 | markdown | the E33 header |

**Checks in the build**
- No `E29_DIRECT2`, `E29D_FILES` or `e29_direct_setup` text is left in any cell.
- CELL 0 and cells 4, 6 and 7 are byte-identical to E29's.

### 3. Cell map (bytes per cell)

| Cell | Type | Bytes | Content |
|---|---|---|---|
| 0 | markdown | 1,010 | the E33 header |
| 1 | code | 2,237 | CELL 0 = E18 R2, verbatim |
| 2 | code | 105,149 | CELL 1, with W1 + W2 |
| 3 | code | 30,176 | CELL 3, with W3 |
| 4 | code | 965 | profiles, as E29 |
| 5 | code | 180,384 | engine; the E29 block is packed and includes W4 |
| 6 | code | 29,384 | as E29 |
| 7 | code | 5,702 | runner, as E29 |
| 8 | code | 17,863 | report, with W5 |

### 4. Switch lines

**CELL 1 literal:**

`E33_CHAIN = {'on': True, 'poetiq': True, 'direct': True, 'attempt2': True, 'poetiq_think': 60000, 'poetiq_answer': 8192, 'direct_think': 104000, 'direct_answer': 8192, 'temperature': 1.0, 'drain_min': 26}`

**What `check_chain` does with it (never raises):**
- `on` not a bool, or not a dict: the whole chain is off.
- A bad Poetiq or Direct value turns only that solver off:
  - not a bool;
  - think < 60,000 (the floor) or not a positive int;
  - answer not a positive int;
  - think + answer ≥ 131,072.
- A bad `temperature` or `drain_min` turns off both solvers, because both use them.
- A bad `attempt2` turns only attempt 2 off.
- A missing key turns its item off; unknown keys are listed under `ignored`.
- The kit override `E33_CHAIN_JSON` (only with `E11_HARNESS=1`) is exempt from the floor, as in E29.

**Smoke one-line diff:**
- production: `E29_PROFILE = "E28EVAL120"`
- smoke: `E29_PROFILE = "E29SMOKE"`

The rerun profile is RERUN_PROD, unchanged from E29.

### 5. How the hooks are wired (`e33_setup`)

**Setup**
- Chain off, or S's cell3 not ok: every hook is `None`, every S env name is removed, and the state is recorded as `OFF_SWITCH` or `OFF_SETUP`. The run is then the E29 post scheduler.
- Otherwise each solver loads from the payload:
  - `e29_direct.py` sets `E12A.E33_RUN_DIRECT = e33_run_direct`;
  - `e33_poetiq.py` first runs `sandbox_ready()` (E18's rule), then sets `E12A.E33_RUN_POETIQ = e33_run_poetiq`.
- If a solver misses (load error or a failed sandbox), its hook becomes `None` and its env `E33_POETIQ` / `E33_DIRECT` is set to `"0"` before the solve. S then drops that stage from the chain: Poetiq off gives A → B → E.
- `E12A.E33_CAN_START = e33_can_start`, and a rate thread starts that reads `e21_prom` every 30 s.
- `E29_RT["direct"] = E29_RT["e33"] = E33_RT`, so E29's `effective()` and report show the chain state.

**`e33_run_poetiq`** (stages C and D)
- It runs the `e29_direct_run` server-url assert, then calls `poetiq_run`:
  - `iteration=` 1 for C, 2 for D; `prior=` the prior record;
  - `attempt=plan["attempt"]`; cfg `branch=plan["branch"]`;
  - cfg: think 60,000, answer 8,192, temperature 1.0, `tok_s_measured`;
  - `tok=e29d_tok()`; `task_index=` the sorted full set; `server_down=E12A.SERVER_DOWN`; `harness=`, `log=`.
- Poetiq keeps its own drain (9 min). No `E10_POETIQ_*` variable is set.

**`e33_run_direct`** (stage E)
- It calls `e29_direct.solve_all_direct` on the one task: workers 1, think 104,000, answer 8,192, temperature 1.0, `drain_min` 26, `solutions=None`, `server_down=E12A.SERVER_DOWN`, the url check.
- **Attempt-2 seeds:** the seed index is `i + (attempt - 1) × N`, where i is the task's index in the sorted full set and N is the number of tasks. Attempt 1 gives index i, exactly as E29. With `e29_direct.task_seed = SEED + 10 × index + phase`, attempt 2 uses seeds no attempt-1 task uses. The index is recorded in `E33_RT["runs"]` as `seed_index`.

**`e33_can_start(stage, deadline)`** gets the bare letters C, D and E.
- C and D use `e33_poetiq.can_admit(deadline, pcfg)`: the 60k floor, Poetiq's 9-min drain, the T − 12 min admission stop, and the lowest of 18 tok/s and the measured rates.
- E uses `e33_lib.can_start`: `now + 60 s + 60000 / rate ≤ T − 26 min` and `now < T − 12 min`, with rate = the lower of 18 tok/s and the measured server tok/s ÷ (running + 1).
- Duck stages return True. Every decision is counted per stage. It never raises.

**LIVE**
- E29's live wrapper is copied, using an `E33Live` class with labels `a<attempt>:<stage>`, for example `a1:A`, `a1:B1`, `a1:C-POETIQ1`, `a2:E-DIRECT`. The labels come from S's `e33_*` row tags.
- Each finished stage prints one line: the train result (Duck TRAIN PASS, or the Poetiq score; Direct has none), exact vs gold only in eval (solutions are `None` in the rerun), and totals per label plus task-norm.
- The `[LIVE start] algo` line shows the chain state, each item's state, the hooks, the budgets, the floor and the scheduler status.
- The 15-min line adds the CAN_START counts and the rate.

**Report and traces**
- `E28_OUT["e33"]` holds:
  - the switch and the effective state;
  - S's statistics from `SCHED_LAST["e28"]["e33"]`, verbatim;
  - per label: finished, train pass, exact tasks, tokens and worker-seconds;
  - the stage reached per task and attempt, with a histogram;
  - skipped-for-time counts;
  - the hook runs, the tokenizer facts and the traces.
- Traces: every file under the run output in `poetiq*` / `direct*` directories is counted and zipped into `WORKING/Q27X_E33_TRACES.zip`. The solvers write `transcript.txt` and `token_ids.json`; the E15B Duck bundle also picks up those transcripts.

### 6. Contract mismatches found (all adapted on my side; no agent file edited)

1. **S plan keys.** The plan carries `attempt` and `branch` (also `e33_attempt`); I read both.
2. **S record tags.** They sit in `rec["e33"]` / `rec["e28"]` as `e33_attempt` / `e33_stage` / `e33_name`; the LIVE stage resolver reads those.
3. **S hook `None`.** S treats a `None` hook as "skip and end the chain". So a solver miss also sets its env to `"0"` before the solve, which makes S skip that stage and keep going (A → B → E) instead of ending the chain.
4. **S `cell3` result.** It returns `env`, not an `env_names` field; my notebook code pops the keys of `rec["env"]` and the module's `ENV_NAMES`.
5. **P's drain.** P's `drain_min` default is 9.0, but the switch's `drain_min` 26 is Direct's (E31). I do not pass `drain_min` to Poetiq, so each solver keeps its own drain. Please confirm this reading.
6. **P's admission.** P's admission uses the lowest rate (never above 18 tok/s), so my Direct rate rule also uses the lower of 18 and the measured rate, to be consistent.
7. **P and S changed during my build.** The early S test run failed (2 failures, 18 errors) on an earlier `e33_sched_patch.py` that has since been replaced. The final pins are S 1b4e8495 and P 397acb4c.

### 7. Test results (all run with `python -B`)

| Suite | Result |
|---|---|
| `e33_nbx_tests.py` (mine) | 18/18 OK — switch checker, admission maths, a late time gives False for C, D and E (the C and D checks use P's real `can_admit`), hook kwargs, E′ seed offset, setup on / off / solver miss / sandbox fail, LIVE lines in eval and rerun (no gold), report |
| `e33_kit_switch_tests.py` (agent K) on the built notebook | 16/16 OK |
| `e33_sched_tests.py` (S) | 38/38 OK |
| `e33_poetiq_tests.py` (P) | 32/32 OK |
| E29 `e29d_direct_tests.py` | 34/34 OK |
| E29 `e29_rev2d_adversarial_tests.py` | 59/59 PASS |
| E29 `e29_sched_tests.py` | 29/29 OK |
| Scratch CELL 3 integration with S's builder | ALL OK — `cell3` env for on / Poetiq invalid / off, apply then revert back to the E29 post bytes, the E28 path refused |

Build checks that passed:
- pyflakes finds no undefined names over all notebook code cells plus `e33_nbx.py` as one module;
- every cell compiles;
- the notebook is under 400,000 bytes;
- the smoke differs by exactly one line;
- the payload shas equal the `E33D_FILES` pins;
- the SUBMIT copy equals the smoke bytes.

### 8. E29 byte-identity proof

- **In memory:** the build regenerates E29_DFLASH_D3S and asserts sha 9a0978474fb2686209ac922c212b2c457cf7d63d7321ba4bbce3118f03bc0885.
- **On disk, notebooks:**
  - `Q27-XHIGH-E29_DFLASH_D3S.ipynb` is 9a097847…;
  - `Q27-XHIGH-E29_DFLASH_D3S_SMOKE.ipynb` is 19928629ac35276fa4652757bfbb4a0eddc66265e88c017a3b3cac5e43701b29.
- **In-build tree check:** before == after over 179 files: the E29 experiment folder (`__pycache__` excluded), `ds_q27-e29-final-payload`, `q27-xhigh-e29-*` and `q27-e29-*`.
- **Separate shell snapshot:** taken before any work and again after all builds and tests, it is identical for every E29 file. The only diff lines are two new files in other experiments' folders that happen to match the `*e29*` pattern: `ds_q27-e32-levers-payload/kit_…e29…ipynb` and my own `ds_q27-e33-chain-payload/e29_direct.py`.
- **Side write:** importing the E28/E26 builders rewrites E25's `k100q_ref27b.json` with the same bytes; E28 asserts that.

### 9. Risks

1. **The smoke cannot run C, D or E.** At 18 tok/s the 60k floor needs 56.6 min:
   - Poetiq can start only with at least 65.6 min left;
   - Direct only with at least 82.6 min left;
   - the E29SMOKE window is 40 min.

   On this profile the smoke will skip every Poetiq and Direct stage; it would only test the boot, Duck A/B, the diff, the hook setup and the "late" skip path. Exercising the chain needs a longer smoke window, a user decision since the spec fixes E29SMOKE. Changing `E29_SMOKE` window_min / budget_min is one CELL 1 literal.
2. **Production timing.** The last Poetiq start is at about T − 66 min and the last Direct start at about T − 83 min. Late-chain tasks will be skipped for time; the report records it.
3. **Size headroom** is about 1.9 kB under 400,000.
4. **Rerun sandbox.** Poetiq needs the root + libseccomp sandbox. If `sandbox_ready()` fails in the rerun, Poetiq is turned off (recorded) and the chain runs A → B → E.
5. **The drain reading** (mismatch 5) needs confirmation.

### 10. Left for agent K and the GPU smoke

**Agent K (kit)**
- The kit expects names that the build now produces: `Q27-XHIGH-E33_CHAIN.ipynb`, `e33_chain_build_facts.json`, `ds_q27-e33-chain-payload`.
- The real kit must run the notebook's own:
  - CELL 1 E33 block (`check_chain` from the payload);
  - CELL 3 `cell3`;
  - `e29_extras` → `e33_setup` / `e33_live_setup` with E12A hooks;
  - `e33_can_start` with the letters;
  - the report and traces zip;
  - the writer.
- Under `E11_HARNESS`, short budgets come through `E33_CHAIN_JSON`.
- Re-check the pins if P or S regenerate (rebuild with `python -B build_e33.py`; it fails on missing files; `--draft` writes only to scratch).

**The GPU smoke** (needs the user's go, and the window decision from risk 1)
- boot and ladder, as E29;
- `E33 chain sched ok`;
- `[LIVE start]` showing chain ON with both items ON and the sandbox ok;
- at least one C, D and E stage started on the GPU;
- the Poetiq client finishing on Merlin;
- `[E33 CHAIN REPORT]`;
- `Q27X_E33_TRACES.zip` present;
- the submission check ok.
