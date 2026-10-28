# VAL fidelity E2: does E21 = R3K2 Duck config B with only the engine changed? (independent second reviewer, 2026-10-28)

This was a read-only review. No project file was edited and nothing was pushed. All scratch work is in `D:/ARC AGI 2/_scratch/e21_val_E2/`.

## What I reviewed

| Item | Version |
|---|---|
| `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb` | sha256 `ad5b2d1fd00d`, 215,239 B, 21:21:02 |
| `Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16.ipynb` | sha256 `b968e3227e7e`, 215,205 B, 21:21:03 |
| `build_e21.py` | sha256 `f8f025a3a4bb` (18:11) |
| `e21_engine_new.py` | sha256 `3f3ca38991df` (21:20:58; this version adds the `e11_kill_server` override) |
| CPU kit `q27x_e21_cpu_kit_83b652715c68.bin` | contains exactly these two notebook shas |
| Donors | E15B R3K2 `c058d58be3cc…`, E12-V `e85b69c2c5f0…`, E18 R2 `116d2d4101a5…` |
| Runtime truth | R3K2 `q27x/e12_run_report.json`: the `configuration` block plus `e12_run` / `engine` |

**Version note.** The notebooks were rebuilt while I was reviewing them. My first extraction was the 18:17 build (`85677597…` / `a1a9e3c7…`). Every result below was then re-run on the 21:21 build. The only difference between the two builds is the `e11_kill_server` override in CELL 5.

---

## Findings

### HIGH-1: Duck's priority gate is process-wide and frozen by the first half

**Location**
- `vendor/inference/agent/tool_agent.py` 2143-2154, `_priority_gate()`: the gate is built once with the `ARC3_MAX_ACTIVE_STREAMS` value at first use, and never rebuilt.
- `tool_agent.py` 2013-2024, `configure_clock`: the earliest start and the earliest deadline win.
- `arc2_duck.py` 2992-2995 (`BranchSession`): calls `gate.configure_clock(...)` and then `gate.acquire(...)` for every session.
- `e21_engine_new.py` `e21_begin_half`: it resets `SERVER_DOWN`, `SERVER_WAITS`, `SERVER_LAST` and `SCHED_LAST`, but **not** `TA._PRIORITY_GATE` or `TA._GATE_STATS`.

**Evidence**
- CELL 6 sets `os.environ["ARC3_MAX_ACTIVE_STREAMS"] = E12_RUN["streams"]` on each half.
- Half A's first session creates `_PriorityGate(slots_A)`.
- Half B gets the same object. Its slot count is A's, and its clock is A's start and A's deadline, because `min()` keeps the earlier values.
- `DUCK_ENV` has `ARC3_PRIORITY_TAIL_FADE=1`. In half B, `remaining = clock_deadline_A − now` is ≤ 0 for the whole half, so every priority computed in B is in the "tail fully faded" mode, with elapsed time counted from A's start.
- I searched every built cell: none resets the gate.

**Consequence**
- **NB1** (by design the scenario the fallback exists for, since INT4 is untested on 0.19): suppose half A falls back to FP8 with 16 streams, or the capacity check gives A fewer than 20 streams. Then half B (INT4 on 0.32, 20 workers) can have only 16 sessions active. The other 4 workers block in `gate.acquire`. Meanwhile `e12_run.streams` / `workers` in B's report still say 20, so the 0.32 arm is silently measured at A's concurrency.
- Whenever the gate does block, half B's admission order uses half A's stale clock.
- **NB2** is unaffected as long as both halves end up at 16 streams.
- The hidden rerun is unaffected, because it runs one half.
- R3K2 never hit this, because it had one solve per process.

**Fix**
- In `e21_begin_half`, rebuild the gate:
  ```python
  with _E12_TA._PRIORITY_GATE_LOCK:
      _E12_TA._PRIORITY_GATE = None
  ```
  Optionally also zero `_E12_TA._GATE_STATS`.
- Record the gate's slot count in each half's report.
- Add a harness mode with A = 16 / B = 20 that asserts B really runs 20.

---

### MEDIUM-1: the half-B (vLLM 0.32) command has no `--language-model-only`; half A has it

**Location:** `e21_engine_new.py` `e21_flags_v032`, compared with `e21_flags_v019`. Confirmed in `src/e21_flags.json`: `nb1:B_*` and `nb2:B_*` have no `--language-model-only`, while every `A_*` entry has it.

**Evidence**
- E18 smoke v1 (`Experiment 18 … /E18_SMOKE_GPU_v1_2026-10-27.md`) ran vLLM 0.32 with S1's flags plus `--language-model-only`. It logged **1,110,415** KV tokens. VMS without the flag logged **1,006,517** (−9.4 %).
- So the flag works on 0.32 and frees memory. It is left out only because the VMS arm did not use it.

**Consequence**
- Half B has about 10 % less KV than half A at the same stream count.
- NB2 needs 16 × p90 65k ≈ 1.04 M. That is at or above B's likely pool (about 1.0 M), but below A's (about 1.1 M).
- So B alone gets more KV-gate holds, prefix evictions and preemption risk. The A-vs-B result then mixes the vLLM-version effect with a flag effect.
- This is not an "engine version" difference. It is a flag-set asymmetry between the halves.

**Fix:** add `"--language-model-only"` to `e21_flags_v032`, then re-run the CPU kit's 0.32 argparse check. Otherwise, state in the README that B deliberately differs.

---

### MEDIUM-2: NB2's Duck token counter silently depends on the INT4 model being attached

**Location:** E15B CELL 1, kept verbatim: `os.environ["E12_TOKENIZER_JSON"] = str(SGL_MODEL_PATH / "tokenizer.json")`. Read by `arc2_duck.py` 136-176 (`reasoning_tokens`, `tokenizer_status`).

**Evidence**
- NB2 serves FP8 and never loads INT4.
- If NB2's kernel metadata does not attach `qwen3-8-27b-int4-redhat`, two things happen:
  - `_e11_discover` scans all of `/kaggle/input` (rglob for `config.json`), which is slow on GCSFUSE;
  - the path does not exist, so `reasoning_tokens` falls back to "a conservative chars/token fallback".
- R3K2 ran with `rev5_tokenizer.loaded = true` (441 tokens on the probe).

**Consequence:** rev 5 B3 reasoning-token accounting is what Duck uses for history and budget decisions. Without the INT4 file it changes vs R3K2 without any error. The only trace is `CONFIG["e12"]["rev5_tokenizer"]["loaded"] = false`.

**Fix:** either
- for NB2, point `E12_TOKENIZER_JSON` at `MODEL_PATH / "tokenizer.json"` (identical vocab and merges per README Finding 4); or
- make the INT4 model a required attachment of NB2 and fail closed in commit mode when `rev5_tokenizer.loaded` is false.

---

### LOW-1: CONFIG is snapshotted before the E21 block, so the recorded "configuration" is partly false

**Location:** E15B CELL 1 builds `CONFIG["e11"]`, `CONFIG["e12"]` and `CONFIG["e12"]["rev6"] = dict(E12_R6)`, and prints them. The appended E21 block then changes runtime values without updating those copies.

**Evidence:** I executed CELL 0 + CELL 1 in a sandbox. Every per-half `e12_run_report.json` "configuration" and the CELL 1 print would show:
- `e11.ENGINE_BACKEND = "sglang"`; the runtime is `"vllm"`.
- `e12.rev6.E12_R6_CACHE_DIAG = 1`; the runtime is 0 (env `0`, `E12_R6` dict 0).
- `e12.E12_STREAMS = 16` and `MAX_NUM_SEQS = 16` in NB1; NB1 runs 20 (via `e21_begin_half` and the half spec).
- `E14_RESULTS.build.variant = "E15B"` (CELL 1b).

**Consequence:** only the report is wrong; behaviour is unaffected. `CONFIG["e21"]` and `e12_run.streams` hold the truth. But anyone diffing configuration blocks, as this review does, gets wrong values.

**Fix:** after the E21 block, update the copies:
```python
CONFIG["e11"]["ENGINE_BACKEND"] = ENGINE_BACKEND
CONFIG["e12"]["rev6"]["E12_R6_CACHE_DIAG"] = 0
```
Also record the per-half streams in `CONFIG["e21"]`.

---

### LOW-2: `wrap()`'s multi-line-string guard misses f-strings on Python ≥ 3.12

**Location:** `build_e21.py` 118-128.

**Evidence**
- On Python 3.12+, f-strings tokenize as `FSTRING_START` / `FSTRING_MIDDLE` / `FSTRING_END`, not `STRING`.
- Test on Python 3.14.6: `s = f"""a\n{x}\nb"""` passed the guard, and after wrapping the value silently became `'a\n    1\n    b'`.
- Today's cells 6 and 7 contain 0 multi-line `STRING` tokens and 0 multi-line f-string parts, so the current notebooks are **not** affected.

**Fix:** also reject `FSTRING_START` / `FSTRING_MIDDLE` tokens that contain `"\n"`, or more simply any token with `tok.start[0] != tok.end[0]` other than NEWLINE / NL / COMMENT.

---

### LOW-3: the PUSHED guard is inert and fragile

**Location:** `build_e21.py` 27 and 541-546.

**Evidence**
- `PUSHED` is empty. That is correct for now, since nothing has been pushed.
- The guard compares the **full** 64-hex sha (`sha(out.read_bytes()) in PUSHED`). The project's convention is 8-hex prefixes (`c058d58b`, …). Adding a prefix would silently never match.
- The 400 kB size assert runs after `out.write_bytes(data)`, so an oversize build has already overwritten the previous file.
- An existing `_REBUILD.ipynb` is overwritten without a check.

**Fix:**
- match prefixes: `any(sha(...).startswith(p) for p in PUSHED)`;
- assert the size before writing;
- apply the same guard to the `_REBUILD` path;
- fill `PUSHED` as part of the push step.

---

### LOW-4: the "no SGLang call left" check covers CELL 6 only, with 5 tokens; SGLang-specific remnants stay in live code

**Location:** `build_e21.py` 386-392. Built cell 6 (`e21_cell6`) and cell 7 (`e21_cell7`).

**Evidence:** remnants left outside `_e12_no_spec` / `_e12_parity`:
- `_e12_r6_preflight` reads `SGL_DRAFT_PATH / "config.json"`. It records a DFlash "drafter_target_layer_config" that does not exist on vLLM, or an error string if DFlash is not attached.
- The `/v1/models` failure fallback is `E12_RUN["model_id"] = str(SGL_MODEL_PATH)`. That would be the wrong model id for NB2. The launch smoke already called `/v1/models`, so this is unlikely.
- CELL 7 `_r6e_concurrency` parses SGLang `TP0]` / `Decode batch` lines and returns `{}` on vLLM.
- The summary is written as `sglang_decode_summary.json`.
- Dead code: `E11Server` is referenced only in `_e12_parity`. A free-name scan of the whole notebook finds no other unbound name.

**Consequence:** reporting noise only.

**Fix:** optional. Extend the check to cell 7 and to `SGL_DRAFT_PATH` / `SGL_MODEL_PATH`, or label these fields as not applicable on vLLM.

---

### LOW-5: CELL 2 with `ENGINE_BACKEND="vllm"` is not isolated, and its prefetch branch picks the FP8 weights

**Location:** CELL 2 (E15B CELL 2, verbatim), lines 19-56.

**Evidence**
- A missing `arc3-vllm-h100-wheelhouse-v3`, a pip error or a parity mismatch raises, and "Run All" stops. Half B is then lost too, although B (the `/tmp/vllm-venv` 0.32) does not need 0.19 in the kernel. In RERUN the result is the identity submission.
- The same holds in S1. R3K2's SGLang path had a try/fallback around its tokenizer install.
- With `E14_LOADPATH=0` (staging off), the `cat` prefetch reads `MODEL_PATH` (FP8) even for NB1, which loads INT4. The default has staging on.

**Install facts (E12-V run 2, the same code):**
- the 0.19 install took about 85 s;
- `numpy 2.0.2` was already satisfied and torch was not reinstalled, so the Duck sandbox stack is unchanged;
- `runtime parity PASS: vllm 0.19.0, transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.36.2`. These are the same tokenizer packages R3K2 installed.

**Fix:** optional. Catch an install failure, mark half A unavailable, and still run half B in commit mode.

---

### LOW-6: NB1's fallback weights (FP8) are not staged

**Location:** CELL 1 E21 block, `E21_STAGE_PATHS = [SGL_MODEL_PATH]` for NB1.

**Evidence**
- A fallback launch in either half reads about 28 GiB from `/kaggle/input` (NFS mount per R3K2 `env.paths`).
- The single-stream read probe gave 34 MB/s. That throughput is only for a cold, single-stream read: vLLM's 4 ranks read in parallel and page caching helps, and S1 always loaded FP8 from `/kaggle/input` and was ready at 668-733 s.
- The fallback runs only after the INT4 launch failed, which can take up to 600 s of log silence plus a kill.

**Consequence**
- Half A has 1,620 s of slack: A must start solving by T0 + 1,620 s, otherwise `E21_HALF_END_PC` (T0 + 21,420 s) truncates its 19,800-s window.
- A slow INT4 failure followed by an FP8 load from NFS can use up that slack.

**Fix:** for NB1, stage both INT4 and FP8. 49 GB fits: `/dev/shm` had 91.8 GB free and the headroom check needs 49 + 24 GiB of the 185 GB available. Otherwise accept the risk and note it.

---

### LOW-7: half B's 25-min setup reserve is adequate but not generous

**Location:** CELL 1 `E21_B_SETUP_RESERVE_SEC = 1500`; `e21_begin_half` sets A's end to `HARD_STOP − 180 − 19,800 − 1,500` = T0 + 21,420 s.

**Evidence:** the reserve has to cover:
- A's tail: join grace 90 s, the summary, CELL 7, then the kill (up to 30 + 30 s) and `_wait_gpu_release` (up to 180 s);
- `e18_install` (175 s in E18);
- the 0.32 launch (550 s in E18) and the smoke;
- B's preflight.

That comes to about 18-20 min, leaving roughly 5-7 min of margin. A B fallback launch, or a slower JIT, clamps B's window at `HARD_STOP − 180`. A always gets its full window, so the bias is against B. The report records each half's `deadline_in_sec`, so the bias is measurable.

**Fix:** optional. Run `e18_install()` in a background thread during half A (CPU only, `/tmp`), which removes about 3 min from B's critical path.

---

### LOW-8: documentation drift and minor items

- **README donor table:** it says `e11_kill_server` is taken verbatim from E12-V. Since 21:20 the engine cell overrides it (`e21_engine_new.py`: `E11_SERVER.kill(fast=not wait)`). The override is correct, because the E12-V version called `E18Server.kill(wait=...)`, which raised `TypeError`, and CELL 7 swallowed it silently. The README line should say "overridden".
- **Stream-count override:** `e21_begin_half` sets `E12_STREAMS = CONCURRENCY`, so an `E12_STREAMS` env override (honoured by R3K2's CELL 6) is ignored. This matters only for manual overrides.
- **Window override:** `E21_HALF_SOLVE_SEC` (env) changes the runtime window. `CONFIG["e12"]["E12_SOLVE_WINDOW_SEC"]` still shows the profile value.
- **Profile without a window:** if `Q27X_PROFILE` is set to a profile without `solve_window_sec`, a two-half commit gives half A everything except 25 min and half B almost nothing. The "two windows must fit" assert is skipped in that case.
- **Tokenizer check:** `CONFIG["e11"]["tokenizer_check"]` becomes `{"ok": None, "skipped": True}`; R3K2 had `ok: true`. On vLLM the server tokenizes and templates with its own model folder, as SGLang did in R3K2, so the gate is not needed. The README says the CPU kit re-checks the real files.

---

## (1) Runtime-config parity: commit-mode CONFIG vs R3K2's real `configuration`

**Method**
- I executed the built CELL 0 + CELL 1 of NB1 and NB2 with stub paths (`runcfg.py`, `cmpcfg.py`, `envdiff.py`), in commit mode and in RERUN mode.
- I compared, key by key:
  - against R3K2's `configuration` (145 flattened keys);
  - the environment against R3K2's own CELL 0 + 1 executed the same way.
- Windows paths (`\kaggle\...`) are a sandbox artifact, not a difference.

**Result: the environment after CELL 1 differs from R3K2 in exactly one variable**, `E12_R6_CACHE_DIAG` (1 → 0). The following are byte-identical to R3K2:
- every Duck, rev 5, rev 6 and E15B knob;
- sampling, budgets, `E12_DUCK_*`, `E12_B_*`, `E12_R6_*` (except `CACHE_DIAG`), `E12_TOKENIZER_JSON`, `E12_ARC2_REV5/REV6`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`.

**Key-by-key differences**

| Key | R3K2 (runtime truth) | E21 | Class |
|---|---|---|---|
| `PROFILE` | E15BEVAL120R3I | E21X2EVAL120 (RERUN: RERUN_ALL) | intended (halves) |
| `TOTAL_BUDGET_SEC`, `profile_row.time_budget_sec` | 21,600 | 43,020 | intended (12-h two-half commit) |
| `profile_row.hidden_equiv` / `solve_window_sec` | true / – | – / 19,800 | intended. **Effective window:** R3K2 real `e12_run.deadline_in_sec` = **20,467 s** (g4b window 20,945 clamped by its 6-h hard stop; start 833 s). E21 = **19,800 s per half** (−667 s, −3.3 %). So the halves are comparable with each other, not exactly with R3K2's 33.47 / 39.03. |
| `e12.E12_SOLVE_WINDOW_SEC` | null | 19,800 (RERUN: null) | intended |
| `e12.rev6.E12_R6_CACHE_DIAG` | 1 (and in R3K2 it was skipped: `left_sec −31`) | recorded 1, **runtime 0** | intended; recorded value false (LOW-1) |
| `e11.ENGINE_BACKEND` | sglang | **recorded "sglang"**, runtime vllm | intended engine change; record false (LOW-1) |
| `e11.give_up_after_sec` | 4,500 | 4,500 (RERUN 14,400, as R3K2's RERUN would be) | same |
| `e11.tokenizer_check` | ok: true | `{"ok": None, "skipped": True}` (CELL 2) | engine consequence (LOW-8) |
| `e12.sandbox_selftest`, `e12.rev5_tokenizer`, `e14.*` | filled at runtime | filled later by CELL 1b / 3 (absent after CELL 1 only) | same code; NB2 `rev5_tokenizer.loaded` depends on INT4 being attached (MEDIUM-2) |
| `e14.submit_arm` | DFlash W6 text | "E21: vLLM halves, see CONFIG['e21']" | label |
| `e21.*` (new) | – | variant, halves, specs, 5.5-h window, 1,500-s reserve, max_model_len 135,168, ports 1234 / 8011, served names, stage path, 0.32 wheelhouse, `reasoning_history_key` | intended |
| `e12.E12_STREAMS`, `MAX_NUM_SEQS` | 16 | recorded 16; NB1 runs 20, NB2 16 | intended (NB1 variant); record false for NB1 (LOW-1) |

**Runtime differences that are not in CONFIG** (from the CELL 3 / 5 / 6 code)

| Item | R3K2 | E21 | Class |
|---|---|---|---|
| Engine | SGLang 0.5.21, INT4, DFlash2 W6, `--mamba-ssm-dtype float16`, context 155,648, lpm, 17 server slots | vLLM 0.19 (A) / 0.32 (B), MTP-3, fp8 KV, prefix caching (mamba "align"), 135,168, max-num-seqs = spec | intended engine. NB1 uses vLLM's default SSM dtype; NB2 uses float16 = R3K2. |
| `ARC3_REASONING_HISTORY_KEY` | reasoning_content | reasoning | intended engine adaptation (vLLM reads only `reasoning`) |
| `ARC3_MAX_ACTIVE_STREAMS` / workers | 16 / 16 | NB1 20 (capacity permitting), NB2 16 | intended; **but the gate is frozen across halves (HIGH-1)** |
| `model_id` | staged path | served name (`Qwen/Qwen3.8-27B-INT4` / `-FP8`; never "qwen", which Duck's preset match avoids) | engine |
| Watchdog relaunch | SGLang: 12-stream flags on OOM or a second relaunch | same-config relaunch (`e21_relaunch`) | engine |
| Solve end | `HARD_STOP − 180`, g4b clamp | `min(HARD_STOP − 180, E21_HALF_END_PC)`, the 19,800-s window, the same preflight shift | intended halves |
| Half B flags | – | no `--language-model-only` | **unintended asymmetry (MEDIUM-1)** |

**Context fits (arithmetic):**
- R3K2 per-attempt peak context: p50 57k, p90 65k, max 79k.
- The request cap is at most 50,000 − 4,096 = 45,904.
- So prompt + max_tokens ≲ 125k, below vLLM's 135,168 limit. vLLM rejects a request whose prompt + max_tokens exceeds that limit, so this margin matters.
- The Duck window stays at the fixed 100,000 (`E12_B_WINDOW`), as in R3K2.

---

## (2) Builder correctness

**Donor shas.** Full sha256 of each donor notebook starts with the asserted prefix:

| Donor | sha256 | Cells |
|---|---|---|
| E15B | `c058d58be3cc7fa3…` | 13 |
| E12-V | `e85b69c2c5f00939…` | 9 |
| E18 | `116d2d4101a53eb4…` | 10 |

The 8-hex prefix is about 32 bits, enough for this purpose. `_scratch/donor_map/E15B_R3K2_c*.py` equals the notebook cells exactly (0 mismatches).

**Cell identity**
- CELL 0 is identical in E15B[1], E12V[1] and E18[1]. The task-order cell is identical in E15B[6], E12V[5] and E18[4]. I checked both independently.
- Built CELL 0 (c01), CELL 2 (c04) and the order cell (c06) are byte-equal to the E15B donor cells.

**`rep()` anchors**
- Every call asserts a count of 1, and a successful build proves each count.
- I checked each one semantically against the donor text:
  - cell1: the profile default and the `E14SEVAL120G4B` row;
  - cell1b: the read probe, the staging condition, the staging call, `submit_arm`;
  - cell3: the `duck_env` line;
  - cell6: `_end = HARD_STOP − …\n` (the cache-diagnostic line `_left = HARD_STOP − …` does not match) and the `_e12_shifted_end` line;
  - cell7: the bundle name and the cache note.
- All hit the intended line.
- The watchdog splice asserts that the end marker appears exactly once and that `e11_flags` and `E11Server(` are inside the region. The start marker is unique in the donor.
- An unwrapped diff of built CELL 6 / 7 against the donor shows only the changes the README lists:
  - the three probes;
  - the watchdog body;
  - the two `_end` caps;
  - the bundle name and the cache note.

**`segment()`**
- I checked all 17 names (8 from E12-V, 4 from E18, 5 from E15B CELL 6). Each has exactly **one** definition anywhere in its donor cell (an `ast.walk` search, including nested ones). None is decorated, which matters because `get_source_segment` would drop decorators. Each extracted segment occurs once in its cell. So "first top-level match" picks the effective definition.
- A free-name scan of all built code cells, using `symtable` against every module-level binding and `global` declaration, finds only `E11Server`, used only in dead `_e12_parity`.

**"No SGLang call left" check (`cell6()`):** correct for what it checks (0 hits after removing `_e12_no_spec` / `_e12_parity`). Its scope is narrow (LOW-4).

**`wrap()` / `module_bindings()`**
- `symtable` confirms every top-level binding of cells 6 and 7 is declared global.
- The only bindings outside the global list are comprehension variables (`i k r t v x` / `f s u x`). These are correctly local in both module and function form, because PEP 709 keeps comprehension isolation.
- Nested threads (`_e15b_watchdog`, the heartbeat) resolve the global-declared names as module globals.
- Per-half re-creation of `_E12_STOP`, `E12_RECORDS` and `E12_RUN` is safe: half A's threads exit on the old event, which is set at the end of A's CELL 6.
- The guard gap is LOW-2.

**Determinism**
- I ran a copy of `build_e21.py`, with only `EXP` patched so it writes to scratch, twice.
- Both runs gave `ad5b2d1f` / `b968e322`, **byte-identical to the notebooks on disk**.
- An early mismatch I saw came from my own PowerShell re-encoding of the builder copy, not from the builder.

**PUSHED guard:** see LOW-3.

---

## (3) NB1 vs NB2

- Cells 1 and 3-10 are byte-identical between the two notebooks.
- CELL 1 differs only inside the E21 block:
  - `E21_VARIANT`;
  - `E21_HALF_SPECS`. NB1: INT4, 20 streams, ssm None, fallback FP8 / 16. NB2: FP8, 16 streams, ssm float16, fallback ssm None, in both halves.
  - `E21_STAGE_PATHS`: `[SGL_MODEL_PATH]` vs `[MODEL_PATH]`.
- The markdown cell differs only in its title and description line.
- No unintended difference. The derived launch commands per variant, half and fallback match `src/e21_flags.json`.

---

## (4) CELL 1b (staging) and CELL 2 (vLLM 0.19 install)

### CELL 1b vs R3K2

Changes:
- staging now also runs for `vllm`;
- it stages `E21_STAGE_PATHS`: NB1 INT4 only (19.5 GB), NB2 FP8 only (about 30 GB; vLLM logs 7.04 GiB per TP rank). R3K2 staged INT4 + DFlash (23.3 GB);
- the read probe runs on the variant's weights;
- the `submit_arm` label changes.

Unchanged from R3K2:
- `E12_RAW_REASONING_HISTORY=1`;
- the heartbeat;
- the E14 modules (payload sha verified, as in R3K2);
- the staging deadline: `max(T0+10 min, min(T0+45 min, SGL_GIVE_UP_AT − 20 min))` = T0 + 45 min in commit and in RERUN;
- the join timeout in CELL 5: up to T0 + 55 min.

### Destination for NB2's FP8 copy

`choose_dest` uses R3K2's measured machine:
- `/dev/shm` 91.8 GB free;
- MemAvailable 194.6 GB, cgroup limit 185.8 GB;
- headroom 185.2 GB.

`/dev/shm` passes both checks:
- free space ≥ 30 × 1.02 + 2 GiB;
- headroom ≥ 30 GB + 24 GiB.

So `/dev/shm` is chosen. The tmpfs pages are charged to the cgroup, and vLLM's safetensors mmap uses those same pages, so nothing is counted twice. That leaves about 150 GB for:
- the TP workers;
- the 16-20 sandbox children (each under RLIMIT_AS 4 GiB);
- the 0.32 venv, which lives on `/tmp` disk.

At about 680 MB/s (R3K2's staging rate) the copy takes about 45 s, plus a full sha256 verification. It is well inside the 45-min deadline. The copy stays in `/dev/shm`, so half B reloads from it. `/tmp` (1.1 TB free) remains the fallback. No problem found.

The staged→original mapping in CELL 5 matches: `stage_models` keys are `str(Path(m))`, and E21 looks them up with `str(SGL_MODEL_PATH)` / `str(MODEL_PATH)`. `_files()` copies every file, including tokenizer and template files.

### CELL 2 (verbatim) with `ENGINE_BACKEND="vllm"`

Behaviour:
- prefetch is skipped while staging runs;
- `pip install vllm==0.19.0` from the S1 wheelhouse;
- the parity check is now enforced for vllm (R3K2 skipped it);
- the kernel tokenizer is still FP8 (as in R3K2);
- the INT4↔FP8 tokenizer check is skipped.

E12-V run 2 measured the same code: install about 85 s, numpy and torch untouched, parity PASS. Isolation and the prefetch branch are covered in LOW-5.

### New attachments E21 needs

- the vLLM 0.19 wheelhouse;
- the vLLM 0.32 wheelhouse (fixed path, no discovery);
- FP8 (both notebooks);
- INT4 (NB1, and NB2 for MEDIUM-2);
- the payload dataset.

DFlash and the SGLang wheelhouse are no longer needed. If they are left out, `_e11_discover` rglob-scans `/kaggle/input` for them, which is slow on GCSFUSE.

---

## Verified (no finding)

- **Donor provenance:** donor shas, CELL 0 and order-cell identity, donor-map files equal to the cells.
- **Builder:**
  - each `rep()` anchor is correct and unique;
  - each `segment()` target is unique and undecorated;
  - no unbound free names except in dead code;
  - the rebuild is deterministic and identical to disk;
  - the size is 210 kB, under 400 kB.
- **CPU kit:** `83b65271` contains exactly the reviewed notebooks.
- **Environment:** after CELL 1, the environment equals R3K2 except `E12_R6_CACHE_DIAG`. All Duck, E15B and rev 6 knobs are unchanged.
- **CELL 6 / 7 changes** are exactly the README list. The scheduler call (`solve_all_r6` with every config-B argument) is unchanged.
- **Per half:**
  - `OUT`, `E12_OUT`, server logs and reports go to `q27x/half<H>/`;
  - `SERVER_DOWN`, `SERVER_WAITS`, `SERVER_LAST`, `SCHED_LAST` and `E15B_SOLVE_END_PC` are reset;
  - Duck's other module state carries over harmlessly. `_RULE_CACHE` is a pure function of the task. Retained functions are per session.
  - The exception is the priority gate (HIGH-1).
- **Requests:**
  - Duck passes `base_url` explicitly, so CELL 3's port-30000 env value is unused;
  - no `priority` field is sent (vLLM's FCFS scheduler would reject one);
  - the served names avoid Duck's "qwen" preset;
  - `ARC3_REASONING_HISTORY_KEY` is set before `_import_duck()` and again per half.
- **Timing:**
  - "two windows must fit" holds (2 × 19,800 + 1,500 = 41,100 < 42,900);
  - half A's real slack is 1,620 s of setup;
  - E12-V and E18 data put the solve start at about 11-15 min, so it fits;
  - the hidden rerun = half A only, ending at `HARD_STOP − 180`, as R3K2's rerun would.
- **Duck-history fix:** the `e11_kill_server` override is correct; it fixes the swallowed `TypeError` in CELL 7.
