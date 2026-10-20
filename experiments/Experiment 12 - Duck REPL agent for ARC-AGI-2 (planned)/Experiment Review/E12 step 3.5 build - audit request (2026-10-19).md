# E12 step 3.5: GPU smoke build, audit request (2026-10-19)

**For:** a GPT reviewer who already knows the E11 notebooks.

**Ask:** look only for **blockers or bugs** that would make the GPU smoke, the EVAL30 run or a hidden rerun fail or score wrongly. Please do not propose redesigns. The plan is fixed by the user:
- Duck harness verbatim plus a thin ARC-2 adapter;
- Duck milestone-2 settings;
- 2 branches per task;
- the E11-06 W1 engine.

Cite the file and line for every finding.

## What is in the zip

| Path | What |
|---|---|
| `PLAN.md` | the plan. §2a Rev 1 = approved fixes; §2b/§2c = step 2/3 status |
| `Duck milestone-2 exact settings (from dfranzen notebook).md` | the settings copied from the milestone notebook |
| `duck_arc2/vendor/` | Duck `inference/` package, 20 files byte-identical to da-fr/arc-agi-3-solution @ 10882e33 (Apache-2.0, NOTICE) |
| `duck_arc2/arc2_duck.py` | **the only new runtime code** (~600 lines): `Arc2ToolAgent`, prelude, host re-verification, driver, selection, submission |
| `Experiment Notebooks/Build/build_e12_duck.py` | builder; source = E11-06 P1-60K EVAL14 notebook (sha256 f0575ff3…, already audited) |
| `Experiment Notebooks/Q27-XHIGH-E12_DUCK_REPL_SGLANG_{SMOKE1,EVAL30}.ipynb` | built notebooks (identical except the default profile) |
| `harness/` | CPU tests: `smoke_adapter.py` (26 checks), `harness_e12.py` + `fake_openai.py` (10 scenarios, 113 checks), `harness_e12_nb.py` + `fake_sgl_duck_server.py` (whole notebook: smoke 15/15, rerun 13/13, launchfail 4/4), logs |
| `push_e12_duck.py` | push script (not run) |

## Notebook cells

1. **CELL 0, 1, 2, the task load and CELL 5 (engine)** are the E11-06 cells. The changes:
   - CELL 1: +SMOKE1 and EVAL30 profiles, +E12 block.
   - CELL 5: + Duck serving flags only: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder --default-chat-template-kwargs {"preserve_thinking": true} --schedule-policy lpm --enable-cache-report`.
2. **CELL 3:** the embedded package (base64 zip, sha256 manifest checked) and a sandbox self-test. The engine cell's existing gate uses it.
3. **CELL 6:** Duck solver.
   - ORDER2 + cells tie-break.
   - All branches are admitted; Duck's priority gate keeps `CONCURRENCY` (12) conversations active.
   - Deadline = hard stop − 60 s (SMOKE1: 15 min window).
   - In a rerun, submission.json is rewritten after every finished branch.
4. **CELL 7:** report, score (per query + task-normalised), bundle with every transcript (full thinking).

## Verified facts (not open)

- **Chat template:** the INT4 checkpoint's template = HF pin c0c686f9 = the FP8 template. `reasoning_effort` defaults to xhigh; `preserve_thinking` true keeps all past reasoning.
- **Requests:** Duck sends non-stream `/v1/chat/completions` with `chat_template_kwargs` {enable_thinking, preserve_thinking}, `max_tokens` 12,288, T 0.7 / 0.95 / 20.
- **Windows-only shim:** the sandbox bootstrap (65 KB) is passed as `python -c`. Windows caps a command line at 32,767 characters, so the local harness writes it to a file. Linux allows 128 KiB per argument, so Kaggle needs no shim.

## Open items (the user decides; flagged, not changed)

1. **Sandbox memory.**
   - Duck's sandbox sets RLIMIT_CPU, FSIZE and NOFILE only. There is no RLIMIT_AS, no setuid and no seccomp (E10's Poetiq runner had RLIMIT_AS 4 GB + setuid + seccomp).
   - Up to 12 model programs run at once in the kernel's container. A runaway allocation could invoke the OOM killer on the SGLang server.
   - Proposed adapter-only fix: add a `preexec_fn` that sets RLIMIT_AS 4 GB, by wrapping `Popen` in the sandbox module's namespace (vendored file unchanged).
2. **KV pool vs Duck's context (EVAL30 / rerun, not the smoke).**
   - Duck grows a conversation to 128 Ki and trims it to ~58 Ki, so the average is ~93k.
   - 12 × ~93k ≈ 1.1M tokens against an 881k KV pool. SGLang would then run fewer than 12 at once or retract requests.
   - The smoke measures the KV per conversation and `cached_tokens`; the decision follows.
3. **Coverage.**
   - A Duck branch releases its slot only when accepted or at its first context trim.
   - Estimate: ~20M tokens for 240 first branches against ~14M per 12 h.
   - Duck-native lever: `E12_MAX_GEN` (= `max_generated_tokens_per_game`), default off as in the milestone.

## Questions

1. Is there any path where `submission.json` ends up invalid or missing in a hidden rerun? Consider especially: a crash inside `solve_all`, a branch exception, the deadline, and the server dying mid-run.
2. Can a submitted program pass host verification without being exact on every train pair, or can the host accept an invalid test grid? See `_verify_submission` and `valid_grid`.
3. Can the gate deadlock or leave branches waiting forever? See `solve_all`'s initial slot hold and release, `run_branch`'s acquire/release, and Duck's handover on trim.
4. Is any Duck env setting read before CELL 3 sets it, or set too late (Duck reads most settings at import)?
5. Does anything in CELL 6/7 depend on a variable that the E11 engine cell defines only on the success path?
