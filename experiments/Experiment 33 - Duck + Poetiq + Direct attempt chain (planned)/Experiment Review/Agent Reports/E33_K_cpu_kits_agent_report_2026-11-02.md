## E33 CPU validation kits: built, not pushed

I built four kits against the final E33 notebook (`Q27-XHIGH-E33_CHAIN.ipynb`, sha 0a0172d6…, 398,100 B). Nothing was pushed, and there were no Kaggle API calls and no git commits. The local layout simulation and all the quick local units pass (details below); nothing has run on Kaggle yet.

There are four kits instead of three. The units kit plus E29's notebook suites would go over 400 kB, so the E29 notebook suites are in their own kit, **units-nb**.

The builder is `D:\ARC AGI 2\experiments\Experiment 33 - Duck + Poetiq + Direct attempt chain (planned)\Build\build_e33_cpu_kit.py` (sha 61eb4547…).
- Run: `python build_e33_cpu_kit.py [--sim] [--local-units] [--only units,units-nb,real,chain]`.
- Its record is `Build\e33_kit_facts.json` (d0cb3f17…).
- It loads E29's kit builder unchanged (exec of its source, one edit: its facts cache goes under E33 Build). A self-check rebuilds E29's real-dflash-d3s kit from it byte for byte (356f2e26… = the ALL_OK kit), so the wrapper is faithful.

### Kit folders, shas, metadata
All four are in `D:\ARC AGI 2\_scratch\push_e25\`, owner foysalemonshanto.

| Kit | Folder | Notebook sha | Size | kernel-metadata sha |
|---|---|---|---|---|
| units | `q27-e33-kit-units__foysalemonshanto` | 64a70ee92665a195ec889ecc42ce23de1b2e5ea2eebb618bd7a263eaba4dfd16 | 388,544 B | 0abb99e9… |
| units-nb | `q27-e33-kit-units-nb__foysalemonshanto` | 07f6b4d1451760d84b8ae3426c3bba07b745474ce64533af89814935beb47be0 | 381,645 B | 39a13c46… |
| real | `q27-e33-kit-real__foysalemonshanto` | 18aa7880ecd96aa2ccf4224dd9cb7f075ca7c897138b9fa9144f4dbd4f805718 | 391,529 B | 5c66cea7… |
| chain | `q27-e33-kit-chain__foysalemonshanto` | 8c3a7dc8b3001715d383f2a5248f9030f61a7b0b0a23217ad880d1b6d1d31893 | 80,646 B | 3393233f… |

What all four kernel-metadata files share, copied from the notebook's own:
- the image is gcr.io/kaggle-private-byod/python@sha256:37c64f7d… (Py3.12);
- datasets: q27-e25-nb0-tf-ref, q27-e15b-payload, q27-e29-final-payload, q27-e33-chain-payload;
- competition: arc-prize-2026-arc-agi-2;
- GPU off, internet off, private, machine_shape None.

Only **real** has models. It has the notebook's 4 models: klee100-3bpw-mtp, fp8ple-overlay, e23-merlin-env, dflash2v2 bf16-embed.

### Inputs per kit
**Every kit:**
- the Duck payload q27x_e15b_duck_9e4b8928.bin, from q27-e15b-payload.

**units:**
- Staged by sha from q27-e33-chain-payload:
  - e33_sched_patch 1b4e8495;
  - e33_poetiq 397acb4c;
  - e29_direct 8b42802c;
  - e33_lib 7bd6cb9f;
  - e33_nbx 8c3c1508.
- e29d_sched_patch.py, from q27-e29-final-payload.
- The competition's eval challenges.
- Bundled in the kit: the test files, the E18 R2 notebook (116d2d41), the K100/W4 branch records, nb5_lib.py, and a copy of the notebook's CELL 0 + CELL 1.

**units-nb:**
- tf-ref: the payload and the pinned files.
- From q27-e29-final-payload: k100q_fixture.json, k100q.ipynb, the seam and the NB-S1 log fixture.
- e33_nbx.py, from the E33 dataset.

**real:**
- Everything E29's real kit uses: the 4 models, tf-ref, the E29 dataset.
- Plus the E33 dataset.

**chain:**
- The E33 dataset modules.

### What each kit checks
**units** (every suite must have 0 skips):
- `e33_sched_tests.py` (S): 38 tests on the real payload.
- The same suite as a negative control (`E33_TEST_NEGCTL=1`). It must fail with at least 5 failures (S measured 23).
- `e33_poetiq_tests.py` (P): 32 tests with the REAL E18 sandbox (root + libseccomp). They run through my runner `e33_kit_run_poetiq_tests.py`, which only points P's builder at the kit's copies of the E18 notebook and e29_direct.py. Without that, the E18-parity and rebuild tests would skip on Kaggle.
- `e33_nbx_tests.py` (I): 18 tests.
- My `e33_kit_switch_tests.py`: 16 tests. It covers:
  - the CELL 1 literal and the notebook's own `check_chain` statement;
  - a bad value turns only that item off;
  - the harness override works only with E11_HARNESS;
  - the 60k-floor admission and items_state.
- E29 suites that must still pass: `e29_sched_tests.py` (29) and `e29d_direct_tests.py` (34).
- Python is 3.12.

**units-nb:**
- E29's retargeted E28 unit tests on the E33 notebook: 44/44.
- `e29_kit_lib_tests.py`, retargeted to read the E33 nbx. It gives 36/44, with the 8 failures exactly on the expected list: R11, R12, R13, R14, R16, R17, R27 (the E29 DIRECT/LIVE parts that E33 replaces by design, W1–W4), plus R28 (see the findings). The verdict fails on any unexpected failure.

**real:** E29's full real kit on the E33 notebook, plus an E33 section and a step-coverage table.
- CELL 1:
  - the switch is all on, with budgets 60000/8192/104000/8192 and temperature 1.0, in both the commit and rerun namespaces;
  - the pins equal the dataset files.
- CELL 3, the E33 diff:
  - applied and verified, sha_after = 4243797a, with the env, in commit and rerun;
  - reverted to the E29 post (d8603109), then E28 revert and re-apply, then E33 re-applied;
  - the patched module is loaded.
- `e29_extras` runs `e33_setup` and `e33_live_setup`:
  - the state is ON and the three hooks are on E12A;
  - the solvers are loaded by sha.
- The notebook's own hooks on a fake token-id server with the REAL klee100 tokenizer (CELL 8's `NB5_LOCAL_TOK.update` and `e29d_tok`, cut verbatim):
  - Poetiq C (wrong program) → D (feedback prompt, pass), checked with the real sandbox;
  - phase-1 max_tokens is 60000 for Poetiq and 104000 for Direct, temperature 1.0;
  - Direct E returns its grid;
  - `e33_can_start` is True far from the end and False near it;
  - the traces are saved;
  - `e33_report`, the report text and the traces zip are produced.
- Poetiq `sandbox_ready`, `poetiq_sandbox_selftest` and a timeout probe run as euid 0 on the Kaggle CPU kernel.
- The writer is the notebook's own `e12_write_submission` + `_e12_write_json_atomic`, cut verbatim from e21_cell6:
  - commit mode writes the identity file;
  - rerun mode writes E33 rows on real test tasks: 1-pass/a2:1-pass, 2-best/a1-next, and a 2-test Direct grid where grid K goes to test K;
  - the schema is strict and every other task stays identity.

**chain** (`e33_kit_chain_e2e.py`, 9 tests):
- **Run setup:**
  - 6 tasks through A → B1/B2 → C → D → E and attempt 2;
  - the real payload arc2_duck with the E29 and E33 diffs;
  - the notebook's own e33_nbx hooks with the real e33_poetiq, the real sandbox and the real e29_direct;
  - a fake token server.
- **Start order:** the exact order on 1 worker (attempt 1 before attempt 2; groups and task order).
- **Counts and answers:**
  - per-stage done / train-pass counts from `SCHED_LAST e28.e33`;
  - the answer slots with source labels and steps for all 6 tasks, also through `build_submission`.
- **Poetiq:** iteration 2 carries iteration 1's program and feedback.
- **Invariants:**
  - one row and one start per branch, and the branch map;
  - one LIVE line per finished stage;
  - with 4 workers: the same answers and no overlap per (task, attempt).
- **Late window:** the real 60k floor makes every C stage late → skipped, the chain ends, nothing is published.

### Local results
**Layout simulation** (`D:\ARC AGI 2\_scratch\kits_e33_local\`, a /kaggle/input-shaped tree): all four kits print `@@KIT <name> LAYOUT ALL OK@@`.
- units and chain: their own code finds and stages every input by sha.
- units-nb and real: their HEAD plus a probe.
- In real, the notebook's own `q27x_payload` finds every E33D_FILES pin.
- The Models were not simulated (tens of GB).

**Quick units:**
- switch 16/16 and nbx 18/18;
- E29 unit tests 44/44 and lib tests 36/44 (only the expected 8 fail);
- chain e2e 9/9 (with a local in-process sandbox stand-in, which the kit never uses);
- the real kit's writer block and hooks block, run against the real modules: PASS (`e33_kit_writer_quick.py`, `e33_kit_hooks_quick.py`, local-only).

**Static checks:**
- pyflakes finds no undefined names in any kit code.
- Build-time A7: all 10 changed hunks of E33 vs E29_DFLASH_D3S carry an E33 marker.

### What I adapted to the other agents' files (none edited)
- I's facts are not in the E28/E29 facts format. The builder writes an E28-format facts file (E29 DFLASH facts + the E33 notebook, export, push, smoke records, `direct=None`) to `Build\_kit_local_e33\e33_chain_kit_facts.json`.
- The scheduler report lives in `SCHED_LAST e28.attempts` (no "rows" key).
- LIVE labels are `a<attempt>:<stage name>`.
- Poetiq has its own 9-min drain. e33_lib has a 12-min admission stop.
- Direct attempt-2 seeds are offset. The chain e2e identifies requests by a model-id tag, so it works with I's final seed fix.
- `build_submission` follows the E33 rule only while E28_SCHED / E33_SCHED are set (as in the notebook run). The tests call it inside that env.

### Findings and risks
1. **The LIVE 15-min line no longer has the ledger** (R28, record only). E29's line printed `" | ledger " + e29_ledger(...)`; `e33_live_setup`'s does not. This is for agent I.
2. **e29_direct.py is loaded from q27-e29-final-payload, not the E33 dataset.** Both have the same sha and `q27x_payload` takes the first match in sorted order. It is harmless; both kit checks accept it.
3. **Margins under the 400 kB checks I applied:** units 388.5 kB, real 391.5 kB, units-nb 381.6 kB.
4. **Never run on Kaggle:**
   - the E33 section of the real kit (the real tokenizer / sandbox / root paths);
   - P's tests with the real sandbox;
   - the negative control.
   On the real kit:
   - the 60000 / 104000 max_tokens checks assume a short prompt for the synthetic task;
   - the expected-fail list is fixed by ID.
5. **The smoke profile cannot exercise C/D/E** (I's report, risk 1).
6. **If P, S or I rebuild,** re-run the builder: it re-reads the pins from the notebook and stages the dataset bytes.

Kit-only files I own, all in `Build\`, all LF:
- `e33_kit_chain_e2e.py` b2d44eba;
- `e33_kit_switch_tests.py` 52897b1c;
- `e33_kit_run_poetiq_tests.py` 19d1e539;
- `e33_kit_writer_quick.py` 686438dd;
- `e33_kit_hooks_quick.py` d940e78c.

Local work dirs: `Build\_kit_local_e33\` and `Build\_kit_local_e29chain\`.
