# E28 W4 notebook agent: final report, verbatim

## E28 W4 notebook: built, local tests pass, nothing pushed

The notebook builds cleanly with the final scheduler module (`e28_sched_patch.py` sha `47ab9874…`; arc2_duck.py post sha `da2c8682…`). The local units pass: 75 / 75, 1 POSIX-only skip. Four CPU kit folders are written and **not run yet**. The audit zip is written. No git, no Kaggle push, no GPU.

## Notebook
- Path: `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Q27-XHIGH-E28_W4_EVAL120_6H.ipynb`
- sha256 `6e13d0d70d7ab7cc6b16456004996567b55857bf38ee7ed1c4f150b250a5b0b3`, **390,262 B** (under 400,000), 9 cells, LF, no CR.
- It is NB6Q (`499a4319`, rebuilt and asserted byte for byte) plus the E28 edits. The review diff vs NB6Q is +276 / −12 lines.

## Files (all LF; Build = `…/Experiment Notebooks/Build/`)

| File | sha256 | Bytes |
|---|---|---|
| Build/build_e28_w4.py | 16abf1b00ea38068ac6fbfd96477ee1ed9d009c0a70706f15696ef19894f14f5 | 44,239 |
| Build/build_e28_w4_cpu_kit.py | 491a24d3c5b6cc09d48ca17c7483071729f63a1180a00c01122e70fbe2452746 | 22,828 |
| Build/e28w4_kit_inject.py (template for the kit builder) | bb618fae2a9042d17a86b151cb8275746f666ba13e4e8401b31ae3aebeceec8e | 12,474 |
| Build/e28w4_lib.py | 5dcbd0cf4349202ada69b89d16f58a263e025a17e6f293f614b6171ea4490c61 | 15,660 |
| Build/e28w4_unpack.py (review view) | 0b6c6703cbd1e9049acddff6c9e415d4e8b94b9a9a00c6609a5851cc3a89b393 | 3,549 |
| Build/e28w4_unit_tests.py | ef3f0ea290ff96aefe2b7bae438a31400c6dcd38c5521d6a1d3dd9d58a2ef6c5 | 45,656 |
| Build/e28w4_build_facts.json | c72ac93cbd384cbb9278593de41403d01fcb3921d053484a4d5eb89352962b7d | 3,310 |
| Build/e28w4_review_diff_vs_nb6q.diff | aca3c2b88b8ab5d97aa736cb4a3c582ec845ae9d96dc775d952c574ac1dc5e15 | 50,236 |
| Build/e28w4_code_export.py | 5b9ad5c94df5510d4652078e294e1b538a90a5dd744ee6cbdcf5b3954075c705 | 362,241 |
| Build/e28w4_nb4t_log_fixture.json (real NB4T GPU log lines) | 36db3b196b4db8978d9bb6a488b2e66908e04a2f832f6fb345a8298a7d240358 | 11,450 |

- **GPU push folder:** `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-w4-eval120-6h__foysalemonshanto/`. It holds the notebook (same sha) and kernel-metadata.json (`e3d0e671…`), with NB6Q's sources, image and L4, private.
- **Embedded modules (W12):** e28w4_lib.py and e28_sched_patch.py `47ab9874`.
- **K100Q deadman code** sha `ec72face…`, cut by ast from `build_e26_k100_quality.py`.

## Changes (line numbers are in e28w4_code_export.py)
- **W1, L51:** the default commit profile is E28EVAL120. The hidden rerun stays RERUN_PROD.
- **W2, L147-:** E28_LEVERS = NB6_LEVERS plus:
  - T1 120,000 for b0 and b1;
  - `seqs_fixed` 12 and `workers_fixed` 12;
  - **admission "slots"** (NB6Q's, as you specified);
  - `E12_DUCK_REQUEST_TIMEOUT` 7200;
  - `pin_start_gib` 0.75 and `DRAFT_HEAD=int8` in the build env;
  - `--language-model-only`, server max-model-len 262144, B2 200,000, no_order2, e28_sched.
- **W2 profiles:**
  - E28EVAL120: eval source, `n_tasks` 120, budget 412 min, hard margin 2 min, so HARD_STOP is at 410 min (6 h 50 min). It uses a fixed 6-h window (K100Q's mechanism: shortened, never skipped).
  - RERUN_PROD: 12 h − 3 min, test source, the same levers.
- **W5, L200-:** the deadman is K100Q's code with names changed. It fires at HARD_STOP + 300 s (415 min), commit only.
- **W3, L306 / L318 / L335:**
  - the launch args gain `--language-model-only`;
  - the E28 constants, plus a B2-only assert (8192 + 200000 + 8192 ≤ 262144);
  - a C12 captures-cover assert (CAPS16 max 64 ≥ 48);
  - `CONFIG["e28"]`.
- **W4, L440:** `E21_VLLM_MAX_MODEL_LEN` = 262144.
- **W12, L490:** the packed modules, at the end of CELL 1.
- **SCHED, L560-:** in CELL 3, after the manifest assert and before `import arc2_duck`:
  - it runs `apply_e28_sched` + verify, then sets the module's `server_env()` in the notebook process only;
  - in the harness, T1 and B2 come from the notebook's numbers;
  - on any failure it re-extracts the original arc2_duck.py (sha re-checked), logs loudly, leaves the env unset and falls back to NB6Q's scheduler.
- **Scheduler edits N1-N4:** all four are applied by label. N1 is at L721, N2 at L897, N4 at L932-; N3 is the ORDER2 line.
  - **W19 (L724, L898, L944):** B2 is capped at the server's 262,144. A, B1 and C keep `MAX_LEN` = 131,072, so B2 reaches 216,384.
  - On your point 1, I kept `MAX_LEN` at 131072 deliberately, so that normal attempts keep NB6Q's guard. The server flag, the E21 clamp and the B2 caps are 262144.
- **W6, L1395-1447:** `--max-model-len` comes from `E28_SERVER_MAX_LEN` (real and fake). `DRAFT_HEAD` goes into `build()` only, as NB4T did, and is recorded in the plan.
- **W18, L1453:** a size-only packing of NB6Q's engine-donor text (verbatim, sha-checked).
- **W7, L1665:** workers = 12 in both admission modes. The C-8 NO_GO is kept.
- **W10, L2359 / L2384:** `e28_readback` after the warm-up and at the end: I8 label, seqs 12, 262144, text-only, pool / F / workers. Record only.
- **W8, L2403:** the ladder starts at 0.75 GiB (3,285,145,068 → 3,016,709,612 → 2,479,838,700). If #60490 fails, it starts at the RC2 pin.
- **W9, L2420:** `--max-num-seqs 12` on every rung.
- **W11, L2520:** no ORDER2, also when the scheduler is not loaded.
- **W17, L3243:** a size-only packing of e21_cell7 (verbatim).
- **W16:** nb5_lib and the lab are re-encoded as lzma + b85 (decoded text identical).
- **W13, L3292:** the fixed window plus `NB5_REPORT["e28_sched"]`.
- **W14, L3434:**
  - `e28_summary` and `e28_sched_report` read `SCHED_LAST["e28"]` (phase counts, rows, B2 rows, slot sources, idle / backfill) and write `e28_sched_report.json`;
  - **preemptions** are recorded at run level (sampler + server counter) and per attempt, using `preempt_events` / `flag_preemptions` copied verbatim from e28k100_lib.
- **W15:** the markdown header.
- **Request timeouts:**
  - Duck's per-request timeout is `E12_DUCK_REQUEST_TIMEOUT` (CELL 1, E15B #47; it was 3600). It is now 7200 for A, B1 and C.
  - The scheduler takes its base from the same env, so B2 = 7200 × 200k / 120k = 12,000 s.
  - Per-request deadline clamps are unchanged.
- **Full saved responses:** NB5's tracer already writes `phase_EVAL/duck/<task>_b<branch>/nb5_requests.jsonl` with the full reasoning, content and tool calls of every request. Duck's transcript.txt is also kept. The e2e checks both.

## Tests (local, Python 3.14)
- `e28w4_unit_tests.py`: **75 / 75 PASS**, 1 skip (the killpg deadman path is POSIX-only; the Linux kit runs it). It covers:
  - the review diff equals the shipped diff, and every hunk is marked E28;
  - CELL 0 + 1 really executed for commit and rerun;
  - a real deadman fire;
  - I8 parsing on the real NB4T logs;
  - the W7 / W8 / W9 / N1 / N2 / N4 / W19 statements executed on stubs;
  - the CELL 3 scheduler block on the real Duck payload (applied / not requested / foreign file / not embedded / round trip / import);
  - the push folder;
  - the preemption flags.
- NB6Q's own `nb6q_unit_tests.py` S2 + S4 on this notebook: **31 / 31 PASS**, 1 POSIX skip.

## CPU kit folders (not pushed)
They are in `D:/ARC AGI 2/_scratch/push_e25/`, private, CPU only, no internet, the pinned image. `build_e28_w4_cpu_kit.py` wraps the proven NB6Q kit builder.

| Kit | Size | sha256 | What it runs |
|---|---|---|---|
| `q27-e28-w4-kit-units__foysalemonshanto` | 349,008 B | 4d7c0f3d… | e28w4 units T1-T7 on the mounted payload |
| `q27-e28-w4-kit-e2e-rerun__foysalemonshanto` | 311,213 B | dd416a0e… | rerun + rerun-retry |
| `q27-e28-w4-kit-e2e-commit__foysalemonshanto` | 311,204 B | f271ab5c… | E28EVAL120, 120 tasks, fake deadline 2,700 s |
| `q27-e28-w4-kit-real__foysalemonshanto` | 393,248 B | 9257281f… | real Merlin + flags |

- **e2e (both modes)** adds checks for:
  - the scheduler being applied and used (phase A for every task, B2 re-runs with the B2 budget and the 261,632 window, phase C);
  - no ORDER2;
  - launch argv / env: 262144, seqs 12, text-only, DRAFT_HEAD;
  - the I8 ACTIVE read-back via the kit fake's lines;
  - slots admission with 12 workers;
  - full responses;
  - the scheduler report and preemptions;
  - timeouts 7200 / 10800 in the harness;
  - the deadman.
- **rerun-retry:** the fake OOM is above RC2 + 0.6 GiB, so 0.75 fails and 0.5 boots.
- **real:**
  - plans from the 0.75 pin with seqs 12;
  - argv / argparse: 262144, language_model_only, spec `draft_lm_head_quant` int8, build env recorded;
  - `SpeculativeConfig` accepts int8 (CPU first, then E27's fake-L4 platform);
  - the checkpoint's max_position_embeddings ≥ 262144;
  - scheduler applied in both namespaces, then verify → revert → re-apply on the real tree;
  - the Q-defer checks and NB6Q's S2 / S4.

## Audit zip
- `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Review/Audit Zips/E28_W4_audit_FLAT_2026-11-01.zip`, 30 files, sha256 `70a79c81…`.
- It includes `00_GPT_PROMPT.md` (GO / NO-GO plus blockers, with 7 explicit questions), the notebook, the builders, the libs, the tests and their logs, the diff, the export, the scheduler module, API and report, the README, the NB4T results, and the 4 kit notebooks.

## Open risks
1. **C12 + I8 together were never measured.** I8 alone needed the lower pin. This build also uses NB6Q's CAPS16 captures (up to 64), while NB4T ran with Merlin's default (up to 32), and `--max-model-len` is 262144. That means more graph memory, so a boot OOM is possible. The ladder 0.75 → 0.5 → 0 GiB (≤ 3 tries, 30 / 15 / 15 min) eats into the 6-h window; a late boot shortens the window, it never skips it.
2. **Slots + 12 workers on a pool of about 197 blocks.** The 12 resident attempts exhaust the pool at a mean context of about 60k, so vLLM will preempt (recompute) at 100k+ and with 200k B2s. Each B2 is about 58 blocks. This is now recorded, not prevented.
3. **200k context on Flash-Next is untested.** RC2 validated 120k. B2 correctness is recorded per attempt only.
4. **`--language-model-only` was not in NB6Q.** I added it as part of NB4T's measured flags; this is a change from NB6Q.
5. **Long requests and the timeout.** A 115,904-token request at about 25 tok/s takes about 77 min, close to the 7,200-s timeout.
6. **NB5 regression dropped from the units kit.** The NB5 QUALITY unit run that NB6 / NB6Q's units kit carried is gone, because together with this notebook it exceeds 400 kB. NB5Q itself is unchanged.
7. **Relaunch pin (unchanged NB6 behaviour).** The watchdog relaunch uses the RC2 pin, not the boot's pin.
8. **Scheduler failure means fallback, not stop.** The run continues on NB6Q's scheduler with two attempts per task and plain order, so the commit would not measure A/B/C.
9. **The CPU kits have not run on Kaggle yet.** They need to run before any GPU go.

