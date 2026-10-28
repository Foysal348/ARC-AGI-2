# E21 audit request: Duck config B on vLLM 0.19 + vLLM 0.32 (two halves per notebook), before any GPU run

## What this is

Kaggle ARC Prize 2026 (ARC-AGI-2), Qwen3.8-27B on 4× L4 (24 GB each, PCIe, no NVLink, offline). Time limit 12 h. The hidden test has 240 tasks.

Our Duck REPL agent "config B" has only ever run on SGLang 0.5.21 + RedHat INT4 + DFlash2. On 120 evaluation tasks in 6 h it scored task-norm 33.47 % (R3K1) and 39.03 % (R3K2); its best LB is 22.92.

The Poetiq method scored 4-7 LB points higher on vLLM (S1-60K: 34.58) than on SGLang (29.17 / 26.94). Per-request quality on eval looked equal.

**E21 runs the same Duck config B on vLLM** in two notebooks:

| Notebook | Weights / streams | Half A (~0-6 h) | Half B (~6-12 h) | Fallback |
|---|---|---|---|---|
| NB1 | RedHat INT4 W4A16 + MTP-3, 20 streams | vLLM 0.19.0 (in the kernel) | vLLM 0.32.0 (/tmp/vllm-venv) | FP8 MTP-3, 16 streams |
| NB2 | FP8 + MTP-3 + `--mamba-ssm-cache-dtype float16`, 16 streams | vLLM 0.19.0 | vLLM 0.32.0 | the same without the SSM flag |

- **Commit run:** both halves solve the same 120 evaluation tasks with fresh Duck state and a 5.5-h window each.
- **Hidden rerun:** half A only, on 240 tasks for 12 h.
- **Report:** per half, the full score and the attempt-1-only score.

## Files in this zip

- `README.md`: the donor map (where every cell / function comes from) and 7 findings made while building.
- `notebooks/`: the two built notebooks (NB1 sha `ad5b2d1f`, NB2 `b968e322`). These are the files to audit.
- `build/`:
  - `build_e21.py` builds the notebooks from the donors and asserts every anchor;
  - `e21_engine_new.py` is the new engine code;
  - `build_cpu_kit_e21.py` is the Kaggle CPU kit.
- `src/`:
  - `e21_flagcheck.py` parses every launch command with the real vLLM parser;
  - `e21_flags.json` holds the 8 commands, with Kaggle paths;
  - `harness/` holds the fake vLLM server + the notebook harness + its README.
- `harness_runs/`: the local CPU harness logs, 12 runs, all PASS. Modes: commit, rerun, fallbackA, halfAfail, death, reasoningkey; both notebooks.
- `donors/`:
  - the donor notebooks: R3K2 (Duck config B on SGLang, the base), E12-V SMOKE1 (the vLLM 0.19 Duck server that worked on GPU, 1 task), E18 R2 (vLLM 0.32 server cell), VMS (vLLM 0.32 engine-matrix flags);
  - `donor_cells/`: their cells as .py files;
  - `duck_9e4b/`: the Duck package source the notebooks load.
- `evidence/`:
  - the vLLM 0.19 / 0.32 `chat_utils.py` (reasoning key);
  - the R3K2 run report (the runtime config and timings of the SGLang run);
  - the E12-V run 2 server log and request logs (Duck turns on vLLM 0.19);
  - the E18 vLLM 0.32 smoke report.

## Already done

- Local CPU harness: 12/12 PASS (see `harness_runs/`).
- A Kaggle CPU kernel is parsing the 8 launch commands with the real vLLM 0.19 / 0.32 parsers and comparing the real FP8 / INT4 model files. Result pending.
- 10 independent Claude agents are reviewing 5 scopes, 2 each: engine, per-half state, clock/rerun, Duck↔vLLM API, fidelity to R3K2. Results pending.

## Please audit: find what would break or silently change results on the GPU

1. **Engine cell (cell 7).**
   - Every flag of both vLLM versions (`src/e21_flags.json`): valid? any conflict? any difference from the commands that ran (E12-V for 0.19, VMS / E18 for 0.32)?
   - INT4 compressed-tensors W4A16 + MTP on vLLM 0.19: the checkpoint ignores `re:^mtp.*` and every `linear_attn`. Will 0.19 load it, and will MTP work?
   - `--mamba-ssm-cache-dtype float16` on 0.19 with MTP and prefix caching (mamba 'align').
   - The lifecycle: ready, smoke, kill, GPU release between halves, fallback, watchdog relaunch.
   - The capacity formula (`e21_capacity`).
2. **Running Duck twice in one kernel.** The run and report cells are wrapped as functions, with every module-level name declared global. Is that equivalent to module-level execution? What state leaks from half A into half B (Duck module globals, threads, env, files)?
3. **Clock.**
   - Commit: do both halves get an equal, fair window? Can anything pass 12 h?
   - Rerun: half A only; solve end; is submission.json complete and valid in every failure path?
4. **Duck ↔ vLLM contract.**
   - The reasoning history key: vLLM reads only `reasoning`; Duck's DUCK_ENV used `reasoning_content`; E21 sets `reasoning`.
   - The tool-call parser output, `finish_reason` length with a cut tool call, `usage` / `cached_tokens`, error wording for a too-long prompt.
   - Metrics names and units for the KV admission gate (`vllm:kv_cache_usage_perc`, 0..1?).
   - The stats-line regex for both versions.
   - Prefix-cache behaviour on Duck turns (E12-V run 2: turn 2 hit 4,800 of 19,215).
5. **Fidelity.** Is everything about Duck config B identical to R3K2 except the intended engine / halves changes? The only stream difference: NB1 uses 20 by design.
6. **Anything else** that makes the half-A vs half-B (and E21 vs R3K2 SGLang) comparison unfair or invalid.

**Please report:** each finding with severity (BLOCKER / HIGH / MEDIUM / LOW), file + cell + line, the evidence, a concrete failure scenario, and a fix. Also list what you checked and found correct.
