# VAL fidelity E1: E21 vs R3K2 cell-by-cell diff audit (2026-10-28)

Read-only validation. No project file was edited and Kaggle was not touched. The only file written is this report.

## 0. What was audited

| Item | sha256 (first 8) | Source |
|---|---|---|
| R3K2 donor notebook `Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K2_DUCK_SGLANG.ipynb` | c058d58b | re-extracted from the notebook; `_scratch/donor_map/E15B_R3K2_c00..c12.py` are byte-equal to it (CRLF ignored), all 13 cells |
| E12-V donor `Q27-XHIGH-E12_DUCK_REPL_VLLM_SMOKE1.ipynb` | e85b69c2 | re-extracted; equal to `donor_map/E12V_SMOKE1_c*.py` |
| E18 R2 donor `Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb` | 116d2d41 | re-extracted |
| E21 NB1 `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb` | **ad5b2d1f** (215,239 B, 21:21) | audited version |
| E21 NB2 `Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16.ipynb` | **b968e322** (215,205 B, 21:21) | audited version |
| Duck payload `q27x_e15b_duck_9e4b89285136.bin` | 9e4b8928 (full sha 9e4b89285136616d…ab3ee566) | Exp 15 Build/payload and `_scratch/e15b/hamid_payload` |

**The notebooks were rebuilt during this audit.** The first extraction got NB1 85677597 / NB2 a1a9e3c7 (18:17). At 21:21 they became ad5b2d1f / b968e322. I re-extracted and re-diffed. The only change between the two builds is in CELL 5: the new `e11_kill_server(wait)` override, which calls `E18Server.kill(fast=not wait)`. Every finding below refers to ad5b2d1f / b968e322.

I also rebuilt both notebooks in scratch from the current `build_e21.py` + `e21_engine_new.py`, with the donor paths unchanged and the output directory redirected. The result was byte-identical: ad5b2d1f / b968e322. So the notebooks on disk are exactly what the builder produces.

Cell map (R3K2 index → E21 index):

| R3K2 cell | E21 cell | Relation |
|---|---|---|
| c00 markdown | c00 markdown | new text (variant line) |
| c01 CELL 0 | c01 | **byte-identical** |
| c02 CELL 1 | c02 | 1 line changed, 5 lines inserted (profile), 36 lines appended (E21 block) |
| c03 CELL 1b | c03 | 4 lines changed |
| c04 CELL 2 | c04 | **byte-identical** |
| c05 CELL 3 | c05 | 5 lines inserted (reasoning key) |
| c06 task order | c06 | **byte-identical** (also equal to E12-V c05, E18 c04, S1-60K c04, VMS c05) |
| c07 CELL 5 SGLang engine | (dropped) | replaced by the E21 engine library (c07) |
| c08 CELL 5b E14-A1 | (dropped) | in R3K2 it ran only `if E14_DO_MEASURE and ENGINE_READY` (False for R3I), so it was a no-op |
| c09 CELL 6 | c08 `e21_cell6()` | wrapped; see §2 |
| c10 CELL 7 | c08 `e21_cell7()` | wrapped; 2 lines changed |
| c11 CELL 6b E14-A2 | (dropped) | no-op in R3K2 (`if E14_DO_MEASURE and E14_CTX is not None`) |
| c12 CELL 7b E14 report / JIT / bundle | (dropped) | in R3K2 (no measure) it did `E14_STATE.end("done")`, `e14_report.md` and `Q27X_E14A_BUNDLE.zip`; CELL 9 partly replaces it |
| — | c09 CELL 8 runner | new |
| — | c10 CELL 9 two-half report | new |

NB1 vs NB2: all code cells are byte-identical except CELL 1. In CELL 1 the only differences are `E21_VARIANT`, `E21_HALF_SPECS` and `E21_STAGE_PATHS` (`[SGL_MODEL_PATH]` vs `[MODEL_PATH]`). The markdown cell differs in the title and the weights line. The diff is in Appendix B.

## 1. Findings

Severity scale: BLOCKER = the run is invalid or crashes. HIGH = a silent change in a Duck config-B condition on a plausible path. MEDIUM = a conditional change, or a semantic difference from R3K2 that should be fixed or documented. LOW = reporting, documentation or failure-path only.

**No BLOCKER found.**

### HIGH-1: Duck's process-wide priority gate and its clock carry over from half A into half B ("fresh Duck state" is not true)

- **Location:** `vendor/inference/agent/tool_agent.py` (Duck package 9e4b, unchanged from R3K2):
  - `_PRIORITY_GATE` singleton (L2139-2153): built once, on first use, with `slots = ARC3_MAX_ACTIVE_STREAMS` read at that moment;
  - `_PriorityGate.configure_clock` (L2018-2024): `_clock_start = min(old, new)` and `_clock_deadline = min(old, new)`.

  E21 `e21_begin_half` (CELL 5) resets `E12A.SERVER_DOWN`, `SERVER_WAITS`, `SERVER_LAST` and `SCHED_LAST`, but not `TA._PRIORITY_GATE` or `TA._GATE_STATS`.
- **Evidence:**
  - `arc2_duck.BranchSession.run` (L2991-2994) calls `gate.configure_clock(...)` and `gate.acquire(...)` for every session, and releases in `finally` (L3043-3044).
  - Duck's analyze loop calls `gate.handover(...)` every turn (tool_agent L6403-6419).
  - The gate is created in half A's first solve with half A's `E12_RUN["streams"]`. CELL 6 sets `ARC3_MAX_ACTIVE_STREAMS` per half, but `_priority_gate()` returns the existing object and never re-reads the slot count.
- **Consequence:**
  1. **Slot count.** Half B runs with half A's slot count. If half B's streams or workers are larger than half A's, the extra workers silently block in `gate.acquire`. Real concurrency is then capped at half A's value, while the report still shows half B's `streams` and `workers`. Plausible paths:
     - NB1: half A on vLLM 0.19 + INT4 is untested (README Finding 5). If it falls back to FP8 with 16 seqs, half B (0.32 + INT4, 20 seqs) runs at 16 real streams. That breaks NB1's main question, the 20-stream INT4 measurement on 0.32.
     - Either notebook: `e21_capacity` derives streams from each half's own KV (`min(seqs, eff // 50,000)`). A lower value in half A caps half B.
  2. **Clock.** Half B keeps half A's deadline as `_clock_deadline`. With `ARC3_PRIORITY_TAIL_FADE=1` / `FRACTION=0.2` (DUCK_ENV), `remaining = 0` for all of half B, so the tail-fade phase is permanently on. This changes the priorities only when there is blocking, which point 1 makes possible.

  R3K2 never had this state: one solve per process, slots = workers = 16, `blocked = 0`.
- **Fix:** in `e21_begin_half`, after the server is up and before CELL 6:

  ```python
  TA = E12A._import_duck()[0]
  with TA._PRIORITY_GATE_LOCK:
      TA._PRIORITY_GATE = None
  TA._GATE_STATS.update(enqueued=0, blocked=0, blocked_ms=0, handover_ms=0)
  ```

  Stragglers from half A hold their own reference to the old gate object, so releasing it is harmless. Add a harness mode with half A at 16 streams and half B at 20, and assert that half B's `_GATE_STATS["blocked"] == 0` and that its server running mean ≈ 20. Record `_GATE_STATS` per half in the E21 report.

### MEDIUM-1: half A's solve window is shorter than 5.5 h whenever half A's setup takes more than ~27 min (the halves are asymmetric)

- **Location:** `e21_begin_half`: `E21_HALF_END_PC = _stop - E21_HALF_SOLVE_SEC - E21_B_SETUP_RESERVE_SEC` for half A. CELL 6: `_end = min(HARD_STOP-180, E21_HALF_END_PC)`, then `min(now + 5.5 h)`. The shift is bounded by the same minimum.
- **Evidence:**
  - TOTAL_BUDGET = 43,020 s, HARD_STOP = T0 + 42,900, `_stop` = T0 + 42,720.
  - Half A's end = T0 + 42,720 − 19,800 − 1,500 = **T0 + 21,420 s**.
  - So half A gets the full 19,800 s only if its window starts at or before T0 + 1,620 s (27 min).
  - Half B starts after half A's end, report, kill, the 0.32 install and the launch, and normally gets the full 5.5 h.
  - R3K2 started solving at 830 s (its report: `started_sec 830`, window 20,467 s). E12-V run 2 started at 649 s.
  - The risky paths:
    - NB1's primary INT4 on 0.19 fails and the FP8 fallback launches from `/kaggle/input`. FP8 is not staged (`E21_STAGE_PATHS` = INT4 only). The GCSFUSE slowdown since 10-21 was ~20 min, and the launch timeout is 30 min.
    - A staging join (up to T0 + 55 min) delays the launch.

    In both cases half A loses 20-60 min against half B.
- **Consequence:** the README states "a solve window of 5.5 h from each half's own solve start". The 0.19 vs 0.32 comparison is then biased against half A: fewer attempt-2 starts, and fewer attempt-1 completions if attempt 1 does not finish. R3K2 started only 78 of 120 attempt 2s in 5.69 h. The bias is visible afterwards in `deadline_in_sec` per half, but it is not controlled.
- **Fix:** give both halves the same window. For example, at half A's solve start, set the per-half window to `min(5.5 h, (HARD_STOP - 180 - reserve - solve_start_A) / 2)` and use it for both halves. Alternatively, cap half B at half A's actual `deadline_in_sec`. Also stage NB1's fallback weights, or accept and document the asymmetry.

### MEDIUM-2: NB2 still depends on the INT4 model being attached (Duck's B3 reasoning tokenizer), and E21 has no input list for the GPU notebooks

- **Location:** CELL 1 (unchanged from R3K2): `os.environ["E12_TOKENIZER_JSON"] = str(SGL_MODEL_PATH / "tokenizer.json")`. `SGL_MODEL_PATH` is the INT4 folder found under `/kaggle/input`. `arc2_duck.reasoning_tokens` (L136-167) loads it, and otherwise uses a chars/token fallback ratio. `tokenizer_status()` is only printed (CELL 3), never enforced.
- **Evidence:**
  - NB2 loads only FP8 weights (`E21_STAGE_PATHS = [MODEL_PATH]`).
  - The INT4 folder is needed in NB2 only for this tokenizer, which is not documented.
  - There is no kernel-metadata or input list for NB1/NB2 in Experiment 21. Only the CPU kit's metadata lists models.
  - R3K2 had the INT4 tokenizer loaded (report: `rev5_tokenizer.loaded = true`).
- **Consequence:** if NB2 is pushed without the INT4 model, Duck's rev-5 B3 history-size estimates silently use the fallback ratio. Trim and budget decisions then differ from R3K2: a config-B behaviour change with no error.
- **Fix:**
  - List the required inputs per notebook in the README: competition data, payload dataset `esfersami50/q27-e15b-payload`, FP8 model, INT4 model (NB1 weights and NB2 tokenizer), vLLM 0.19 wheelhouse, vLLM 0.32 wheelhouse.
  - Fail closed (sandbox self-test `ok=False`) if `E12A.tokenizer_status()["loaded"]` is False.
  - Optionally use `MODEL_PATH/tokenizer.json` when INT4 is absent. README Finding 4 says the vocab and encodings are equal; re-check that in the kit.

### MEDIUM-3: the stream-count capacity check is fail-open, where R3K2's was fail-closed

- **Location:** `e21_capacity` (CELL 5):

  ```python
  streams = seqs if eff is None else max(8, min(seqs, eff // 50000))
  ```

  `fits_p50_peak` is only recorded. `e21_begin_half` accepts any config whose `/health`, KV line and smoke pass.
- **Evidence:** R3K2's `e11_capacity_gate` (c07 L284-298) was fail-closed:
  - every limit had to be logged;
  - `conc = min(16, mamba//slots, graph bs, kv//50,000)`;
  - it refused below `SGL_MIN_RUNNING = 8` and below `SGL_MIN_KV = 600,000`, then tried a 12-stream relaunch, then the vLLM fallback.

  In E21:
  - a missing "Maximum concurrency" line (for example a vLLM 0.32 log-format change) gives the full `seqs` (20 for NB1);
  - an effective KV below 400k still gives 8 streams;
  - no outcome leads to the fallback config.
- **Consequence:** Duck can run at more streams than the KV supports, which means preemptions and recomputation. The scheduler's KV admission gate (0.95) protects only new admissions. This is an engine-move consequence that changes the stream count Duck runs at, against R3K2's semantics.
- **Fix:** if `eff is None`, fail that config (fallback), or estimate it from `kv_tokens_logged`. If `eff // 50k < 8`, fail the config instead of forcing 8. Consider failing over when `fits_p50_peak` is False for NB1's 20-stream primary.

### LOW-1: CONFIG records values that differ from what runs (report only)

- `CONFIG["e12"]["rev6"]` is copied before the E21 block, so it records `E12_R6_CACHE_DIAG: 1`. The run uses 0.
- `CONFIG["e12"]["E12_STREAMS"] = 16` is recorded although `e21_begin_half` overwrites `E12_STREAMS` with the capacity value (20 for NB1).
- `CONFIG["e11"]["ENGINE_BACKEND"] = "sglang"`, and the SGLang `give_up_after_sec` is printed, although `ENGINE_BACKEND = "vllm"`.
- Fix: update these three entries in the E21 block, and write `E12_STREAMS` into `CONFIG["e12"]` in `e21_begin_half`.

### LOW-2: Duck's environment base URL points at a dead port

- **Location and evidence:** CELL 3 sets `LOCAL_ANALYZER_BASE_URL` / `OPENAI_BASE_URL` = `http://127.0.0.1:30000/v1` (inherited from R3K2, where it was the live SGLang port). E21 serves on 1234 / 8011.
- **Why it is harmless:** `solve_all_r6(base_url=SGL_URL + "/v1")` reaches `BranchSession` and then `agent_cls(base_url=...)`. In `ToolAgent.__init__` (L3483-3492) an explicit base_url overrides the environment value.
- **Fix (hygiene):** in `e21_begin_half`, set both variables to `SGL_URL + "/v1"`.

### LOW-3: watchdog failure path is weaker than R3K2's

- **Location:** CELL 6 `_e15b_watchdog` → `e21_relaunch()` → `e21_launch` with `wait_ready(max(60, min(1800, E21_HALF_END_PC - now - 1800)))`.
- **Evidence:**
  - R3K2: `min(SGL_RELAUNCH_TIMEOUT_SEC = 1200, max(60, left - 600))`, plus a degraded 12-stream relaunch after an OOM or RuntimeError, or on the second relaunch (B3).
  - E21 relaunches with the same config. With 20-35 min left in a half, the ready timeout is 60 s, so the relaunch cannot succeed (vLLM takes ≥ 5 min).
- **Consequence:** this matters only after a server crash. It is not a config-B change on the normal path.
- **Fix:**
  - Use `max(60, min(1800, _left - 600))`.
  - On an OOM or a second relaunch, consider the half's fallback config.
  - Document that the 12-stream degraded relaunch was removed (it is not in the README).

### LOW-4: half-A straggler callbacks are not bound to their half

- **Location:** CELL 6 `_e12_branch_done_r6` writes into the global `E12_RECORDS` / `E12_RUN`, which CELL 6 rebinds in half B.
- **Evidence:** `join_grace_sec = 90`. Stragglers end quickly after the deadline, because requests are clamped to the remaining time and the server is killed in `e21_end_half`. Half B's CELL 6 starts only after the 0.32 install and launch, at least 10 min later. So cross-half contamination is practically impossible. A straggler that finishes between half A's CELL 7 snapshot and `e21_end_half` changes only the end-half snapshot, which the CELL 9 attempt-1 score uses, versus the CELL 7 full score.
- **Fix (optional):** capture the half's dicts in the callback (a closure over the local `E12_RECORDS`), or drop callbacks whose `rec` arrives while `E21_HALF` differs.

### LOW-5: legacy profiles and dead code still reachable

- **Profiles:** PROFILE_TABLE keeps every R3K2 profile. If `Q27X_PROFILE` selects a profile without `solve_window_sec` (for example `E15BEVAL120R3I`, 6 h, `hidden_equiv`), then `E21_HALF_SOLVE_SEC = 0` and half A takes all but 25 min. `PARITYR6` would call `_e12_parity`, which references the undefined `E11Server` and `e11_flags` (the builder deliberately leaves them only in dead code).
- **Dead code:** `_e12_cache_diag` is now never called (CACHE_DIAG 0). `_e12_no_spec` and `_e12_branch_done` are unused. The README's "dead code kept" names only `_e12_parity`.
- **Fix:** assert `PROFILE in ("E21X2EVAL120", "RERUN_ALL")` (or require `solve_window_sec` when not RERUN) in the E21 block, and complete the dead-code list.

### LOW-6: vLLM-only report fields are empty or renamed (report only)

- CELL 7 `_r6e_concurrency()` parses SGLang "Decode batch" lines. On vLLM it returns `{}`.
- `_r6_metrics_more.accept_len_median` reads `dec["accept_len_median"]`, but the E12-V `e11_decode_summary` names it `spec_mean_acceptance_median`, so the value is None.
- CELL 7 bundles `sglang_requests.jsonl`, which is never produced on vLLM (skipped as a missing file).
- The CELL 6 server-log summary drops R3K2's startup split (A3), by design.

None of these affects the solve. Optionally map these fields to vLLM equivalents.

### LOW-7: README donor-map inaccuracies and omissions

1. **CELL 0 row.** It says CELL 0 is "identical in E15B R3K2, E12-V, E18, S1, VMS (asserted)". `build_e21.py` asserts only E15B == E12-V == E18 (L44-45). I verified that S1-60K and VMS are also identical, but that is not asserted. The same applies to the order cell.
2. **CELL 5 row.** It lists the E12-V kill as verbatim. The current build also adds a new `e11_kill_server` (E18Server signature, `fast=not wait`) that overrides the verbatim E12-V one. This is not in the donor map. It fixes a real TypeError: before the 21:21 build, CELL 7's final `e11_kill_server(wait=False)` raised and was swallowed.
3. **CELL 6 row.** It omits:
   - the removed R3K2 watchdog degraded 12-stream relaunch (`_tail`, OOM markers, `e11_flags(streams12)`, `SGL_MAX_RUNNING/SGL_SERVER_MAX_RUNNING = 12/13`);
   - the removed explicit `_srv.kill()` and `_e11_gpu_free()`, now inside `e21_relaunch`: `E11_SERVER.kill()`, `_e18_kill_gpu_pids()`, `_wait_gpu_release()`;
   - the removed SGLang startup split in the log summary;
   - two comment-only edits.
4. **CELL 1 row.** It does not list `E21_B_SETUP_RESERVE_SEC`, `E21_LAUNCH_TIMEOUT_SEC`, `E21_VLLM_MAX_MODEL_LEN`, `E18_SILENCE_SEC`, `E21_FAKE` / `E18_FAKE`, the "two windows must fit" assert, or `CONFIG["e21"]`. It also does not say that the new profile drops R3K2's `hidden_equiv` window (R3I = 120 × hidden per-task share, about 5.8 h, clamped to 5.69 h). The "Why" section mentions ~5.7 h.
5. **"Dropped from E15B".** It omits that CELL 7b's `E14_STATE.end("done")`, `e14/e14_report.md` and `Q27X_E14A_BUNDLE.zip` are gone. CELL 9's bundle carries `e14/*` and `e14_state.json`, without a terminal state.
6. **Notebook shas.** The README / notes should record the current shas ad5b2d1f / b968e322.

### LOW-8: engine-side differences in the serving flags (engine move; listed for completeness, not Duck config-B changes)

1. **`--generation-config vllm`** (Duck's own launch flag, E12-V verbatim). The model's `generation_config.json` holds only temperature 1.0, top_p 0.95 and top_k 20. Duck sends all three explicitly (DUCK_ENV top_p 0.95 / top_k 20, `LOCAL_ANALYZER_TEMPERATURE` 1.0, `build_chat_payload` provider vllm), so sampling equals R3K2. Its eos list `[248046, 248044]` is probably not added as extra stop ids in "vllm" mode (the tokenizer's eos is used). This is the same as E12-V and S1, and the effect is negligible (`<|endoftext|>` is rarely emitted in chat).
2. **Flag differences between the halves:**
   - 0.19: `--language-model-only`, no `--trust-remote-code`;
   - 0.32: `--trust-remote-code`, no `--language-model-only` (the VMS arm), so 0.32 may load the vision tower and have less KV.
3. **SGLang-only flags with no vLLM equivalent:** `--schedule-policy lpm`, `--max-running-requests` = client + 1, DFlash2.
4. **Context headroom.** vLLM rejects prompt + max_tokens > 135,168 instead of clamping. Duck's window is 100,000, R3K2's per-attempt peak was max 79k, and request caps are ≤ remaining budget − 4,096, so there is ≥ 35k headroom.
5. **KV gauge semantics.** `vllm:kv_cache_usage_perc` is the hybrid pool (attention + mamba blocks). SGLang's `full_token_usage` was attention only. The 0.95 admission gate therefore binds at a different point.

## 2. Every changed line (summary; full unified diffs in Appendix A)

### CELL 1 (R3K2 c02 → E21 c02)

| # | Line(s) | Change | Class |
|---|---|---|---|
| 1 | L2 | default profile `E15BEVAL120R3I` → `E21X2EVAL120` | two halves (12-h commit; R3I = 6 h, hidden_equiv window) |
| 2 | +L116-120 | new profile row `E21X2EVAL120` (eval, n_tasks 120, 12 h − 3 min, the same margins as RERUN_ALL/EVAL120, `solve_window_sec` 5.5 h) | two halves; window 5.5 h vs R3K2's 5.69 h is a documented time difference (MEDIUM-1 for the A/B asymmetry) |
| 3 | +L394-429 | E21 block: `E21_VARIANT`, `E21_HALF_SPECS`, `E21_HALVES`, `ENGINE_BACKEND="vllm"`, `E21_STAGE_PATHS`, `E12_SOLVE_WINDOW_SEC` re-assigned (env `E21_HALF_SOLVE_SEC`), `E21_HALF_SOLVE_SEC`, `E21_B_SETUP_RESERVE_SEC` (25 min), `E21_LAUNCH_TIMEOUT_SEC`, `E21_VLLM_MAX_MODEL_LEN` 135,168, `E21_PORTS` 1234/8011, `E21_SERVED`, `E18_VLLM_WHEELHOUSE`, `E18_SILENCE_SEC`, `E21_FAKE`/`E18_FAKE`, `E12_R6["E12_R6_CACHE_DIAG"]=0` + env, the window assert, `CONFIG["e21"]`, print | engine move (backend, ports, served names, context, wheelhouse) / two halves (specs, halves, window, reserve) / cache diag off (allowed by the brief: it ran after solving in R3K2, H6-3, so the solve is unaffected) |

Nothing above L394 changes except rows 1-2. The E15B dict, E12_R6 dict, E12 settings, sampling, engine constants, SGL_* constants and CONFIG lines are all byte-identical text.

### CELL 1b (R3K2 c03 → E21 c03)

| # | Line | Change | Class |
|---|---|---|---|
| 1 | L68 | `read_probe(SGL_MODEL_PATH) if sglang else None` → `read_probe(E21_STAGE_PATHS[0])` | engine move (report only) |
| 2 | L73 | staging condition: `sglang` → `in ("sglang", "vllm")` | engine move |
| 3 | L75 | `Staging([SGL_MODEL_PATH, SGL_DRAFT_PATH], ...)` → `Staging(E21_STAGE_PATHS, ...)` | engine move (NB1 INT4, NB2 FP8; no drafter) |
| 4 | L129 | `submit_arm` label → `"E21: vLLM halves, see CONFIG['e21']"` | report only |

`E12_RAW_REASONING_HISTORY = "1"` (L80) is unchanged.

### CELL 3 (R3K2 c05 → E21 c05)

| # | Line | Change | Class |
|---|---|---|---|
| 1 | +L49-53 | 4 comment lines + `os.environ["ARC3_REASONING_HISTORY_KEY"] = "reasoning"`, after `duck_env(...)` and before `_import_duck()` | engine move. It keeps config B's intent (prior thinking in the prompt) on a server that reads only `reasoning`. Duck reads the key at call time (`tool_agent._reasoning_history_keys`, L1262-1278). The arc2_duck B3 counter, the E14-P raw-reasoning swap (L1742) and the reasoning harvest (L2335) all use the same key function, so they stay consistent. E12-V CELL 3 (e85b69c2, L99-101) set the same value. I verified the E12-V run-2 prompt growth against the run's branch_result: 6,882 → 19,215 after 12,288 generated; the second prompt hit 4,800 cached. The vLLM source line numbers (chat_utils.py:1512 in 0.19, renderers/chat_utils.py:2083 in 0.32) could **not** be checked locally: no vLLM source is on disk, and `_scratch/vllm_whl` holds no readable wheel. |

The Duck package name and sha, the manifest (23 entries), the self-test, the sandbox memory limit and the log filter are unchanged.

`e21_begin_half` sets the key again for each half (redundant, harmless).

### CELL 6 (R3K2 c09 → E21 `e21_cell6()`)

1. **Mechanical wrap.**
   - Every non-blank line is indented 4 spaces.
   - One `global` statement lists all 53 module-level bindings.
   - The builder asserts that the cell has no multi-line string.
   - `symtable` on the built cell: the only function-locals left are comprehension iteration variables (`i, t, k, v, r, x`), which were isolated at module level as well (PEP 709 inlining, Py 3.12).
   - Every module-level name, including `E15B_SOLVE_END_PC`, `_e15b_solve_end`, `_end`, `_workers`, `_E12_STOP`, `E12_RECORDS` and `E12_RUN`, resolves as a global, as in R3K2.
   - **Same behaviour.**
2. `_e15b_gen_tokens_total`: SGLang counter `sglang:generation_tokens_total` (max over tp_rank, summed) → `vllm:generation_tokens_total` (summed over rows) via `e21_prom()`. Engine move, report only (#32).
3. `_e15b_server_log_summary`: SGLang "Decode batch" parsing and the startup split → vLLM 10-s stats lines (`_E12V_STAT_RE`), "Mean acceptance length", "GPU KV cache size". New fields: `gen_tok_s_mean`, `prefix_cache_hit_pct_last`, `kv_usage_pct_max`. The `_bad` filter `retraction_policy` → `non-default args`. Engine move, report only.
4. `_e12_capacity_probe`:
   - `sglang:full_token_usage` / `mamba_usage` / `num_running_reqs` / `num_queue_reqs` → `vllm:kv_cache_usage_perc` / `num_requests_running` / `num_requests_waiting`;
   - `mamba_usage = None`.

   Engine move. `kv_usage` feeds the scheduler's 0.95 admission gate; it is a 0-1 fraction on vLLM, so `_finite01` accepts it. `mamba_usage` is only logged (`SERVER_LAST`, gauges).
5. L413: `_end = HARD_STOP - 180` → `min(HARD_STOP - 180, E21_HALF_END_PC)`. Two halves.
6. L459: the shift bound `HARD_STOP - 180` → `min(HARD_STOP - 180, E21_HALF_END_PC)`. Two halves.
7. Watchdog:
   - `global E11_SERVER, SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING` removed (comment added);
   - the B2 comment reworded (code identical);
   - the relaunch body replaced: R3K2's `_tail`, `_srv.kill()`, `_e11_gpu_free()`, the B3 12-stream degraded flags, `E11Server(...)`, `wait_ready(min(1200, max(60, left - 600)))` and the `E11_SERVER` swap → `old_logs.append` first, the token accounting kept, then `_ok, _why = e21_relaunch()`.

   Engine move (LOW-3 for the timeout and the lost degraded relaunch).

**Unchanged in CELL 6 (verified line by line):**
- `_E12_INDEX`, `_e12_order2_score`, `_e12_total_cells`, `E12_ORDER` (ORDER2 + cells tie-break + index);
- `E12_RECORDS` / `E12_RUN` init, `e12_write_submission`, `_e12_branch_done_r6` (publish_seq guard, `E15B_SOLVE_END_PC` B1);
- `_e12_g4b_window`, `_e12_shifted_end`, `_e15b_all_server_logs`, `_e12_cache_diag` (now not called);
- `_e12_no_spec`, `_e12_parity` (dead);
- `_e12_r6_preflight` (isolation smoke with `streams` concurrent markers, retry on INCONCLUSIVE, `workers_for_isolation`);
- the heartbeat;
- `streams = min(CONCURRENCY, E12_STREAMS or CONCURRENCY)` and `ARC3_MAX_ACTIVE_STREAMS`;
- the window and hidden_equiv logic;
- `rev6_window = min(131072, E12_B_WINDOW)` and `E12_DUCK_CONTEXT_WINDOW`;
- the watchdog poll / fail / min-left limits;
- **the whole `solve_all_r6(...)` call**;
- the `b_summary` block, the CACHE_DIAG guard, the error path and the final submission write.

### CELL 7 (R3K2 c10 → E21 `e21_cell7()`)

1. The mechanical wrap: 56 globals. Same check: only comprehension locals `s, f, x, u`.
2. L227: the `cache_note` text ("vLLM 0.19 / 0.32, as SGLang 0.5.21"). Report only.
3. L388: the bundle name `Q27X_E12_DUCK_BUNDLE.zip` → `Q27X_E21_{VARIANT}_HALF{H}_DUCK_BUNDLE.zip`. Two halves.

Everything else is unchanged: the scoring (`eval_*`, task_norm), `eval_b_attempts`, the stage-3 accounting, per-GPU-hour, `e12_write_submission()` and the final `e11_kill_server(wait=False)`. That call now resolves to the E21 override `kill(fast=True)`.

### CELL 5 engine library (new; donor pieces verified verbatim)

These segments are byte-equal to their donors, checked by AST source-segment comparison against the re-extracted notebooks:
- E12-V CELL 5: `ensure_libcuda_linkable`, `_e12v_request_json`, `_E12V_PROM_RE`, `_E12V_ALIAS`, `metrics_snapshot`, `_E12V_STAT_RE`, `e11_decode_summary`, `e11_kill_server` (then overridden);
- E18 R2 CELL 5: `e18_install`, `E18Server`, `_e18_kill_gpu_pids`, `_wait_gpu_release`. `_wait_gpu_release` is also in S1-60K c06.

The vLLM 0.19 flags equal the E12-V command (same flag set and values), except: the model path (INT4 for NB1), `--max-num-seqs` per half instead of `MAX_NUM_SEQS` 16, and `--mamba-ssm-cache-dtype` for NB2.

The vLLM 0.32 flags equal VMS `vms_flags` (TP4, fp8_e4m3, seqs, 4096, 0.90, custom-AR off, seed, backend, trust-remote-code, prefix caching, MTP-3), plus Duck's serving flags, with `--max-model-len 135168` instead of 106,496 and the served name and port changed.

### CELL 8 / CELL 9 (new)

The runner and the two-half report. The runner calls `begin → e21_cell6 → e21_cell7 → end` per half, each step in its own try. A failed begin sets `ENGINE_READY = False`, and CELL 6 then records "not run". In RERUN there is only half A, with `E21_HALF_END_PC = HARD_STOP − 180`, the same end as R3K2's RERUN.

## 3. Verified identical at run time (Duck config B)

| Setting | R3K2 (c058d58b, its run report) | E21 NB1 / NB2 | Status |
|---|---|---|---|
| `E12_B_MODE` | 1 | 1 | identical (same text, env written in the same loop) |
| `E12_B_ATTEMPT_TOKENS` | 50,000 | 50,000 | identical |
| `E12_B_ATTEMPT_TOKENS_B1` | 45,000 | 45,000 | identical |
| `E12_B_RESERVE` | 4,096 | 4,096 | identical |
| `E12_B_WINDOW` → `rev6_window` / `E12_DUCK_CONTEXT_WINDOW` | 100,000 | 100,000 | identical |
| `E12_B_SKIP_B1_IF_ACCEPTED` | 0 | 0 | identical |
| `E12_B_EXTRA_ATTEMPTS` | 1 | 1 | identical |
| `E12_B_MAX_ATTEMPTS` | 8 | 8 | identical |
| `E12_B_WEAK_COMMIT` | 1 | 1 | identical |
| `E12_B_B1_ACC_MODE` | 2 → "before3" | 2 → "before3" | identical |
| asserts (attempt budget = session cap; 0 < b1 ≤ b0) | present | present | identical |
| E12_R6 (all 26 keys): CAP_FIRST 12288, CAP_PROGRESS 24576, CAP_RESCUE 32768, FIRST_EXPOSURE 0, B1_VIEW facts, MIN_ADMIT_SEC 300, CHECK_VERIFY_CAP 24, AUTOCHECK_CAP 24, AUTOCHECK_PER_CALL 3, DUCK_DRAIN_TOKENS 16384, FACTS_MAX_CHARS 3600, SHOW_MAX_CELLS 900, KV_FRACTION 1.0, P90_CUTOFF 0, FACTS_MAX_TOKENS 800, FAIR_SHARE 0, KV_ADMIT 0.95, KV_POLL 5.0, CAP_FIRST_B1 24576, SHOW_CALL_CELLS 1000, NUDGE_TOKENS 24000, B1_GRID_MAX_TOKENS 12000, MIN_CHUNK 12288 | as listed | as listed | identical |
| `E12_R6_CACHE_DIAG` | 1 (ran after solving) | 0 | intended change; no effect on solving (H6-3) |
| `E12_BRANCHES` | 2 | 2 | identical |
| `E12_STREAMS` / `ARC3_MAX_ACTIVE_STREAMS` / workers | 16 / 16 / 16 (server running mean 15.62) | NB1 `min(20, effKV//50k)`, at least 8; NB2 `min(16, effKV//50k)`, at least 8 | by design for NB1 (20) and NB2 (16); see MEDIUM-3 for the fail-open path and HIGH-1 for half B's gate |
| `E12_MAX_GEN_PER_BRANCH` | 50,000 | 50,000 | identical |
| `E12_DUCK_MAX_OUTPUT` → `LOCAL_ANALYZER_MAX_OUTPUT` | 12,288 | 12,288 | identical |
| `E12_DUCK_TEMPERATURE` → `LOCAL_ANALYZER_TEMPERATURE` | 1.0 | 1.0 | identical |
| top_p / top_k (DUCK_ENV, sent in every payload) | 0.95 / 20 | 0.95 / 20 | identical (generation-config "vllm" does not override explicit values) |
| `E12_DUCK_REQUEST_TIMEOUT` | 3,600 | 3,600 | identical |
| effort (`E12_REASONING_EFFORT`) | "" (template default xhigh); no `ARC3_REASONING_EFFORT_LADDER` | same | identical. The chat template: NB1 serves the INT4 folder's, as R3K2 did; NB2 serves the FP8 folder's, which README Finding 4 says is equal (sha c3cf9e34); not re-verified here |
| `preserve_thinking` | server default kwargs + Duck `ARC3_PRESERVE_THINKING` 1 | same | identical |
| `E12_ARC2_REV5` / `E12_ARC2_REV6` | 1 / 1 | 1 / 1 | identical |
| `E12_RAW_REASONING_HISTORY` | 1 | 1 | identical |
| `ARC3_REASONING_HISTORY_KEY` | `reasoning_content` (DUCK_ENV) | `reasoning` | intended engine-move change (see §2 CELL 3) |
| `ARC3_YIELD_AFTER_BATCH`, `ARC2_SANDBOX_SYS_SHIM` (via `E12_B_MODE=1` in `duck_env`) | 1 / 1 | 1 / 1 | identical |
| the rest of DUCK_ENV (provider vllm, tool steps 0, tool timeout 30, yield 2048, context 128 Ki → overridden by 100k, drain 16,384, history turns 150/30/coalesce, tool output 3072, middle truncation, memory off, persistent functions, priority settings, HTTP retry 900) | — | — | identical (the same package and the same `duck_env` call) |
| `E12_TOKENIZER_JSON` | INT4 `tokenizer.json` (loaded = true) | same path | identical for NB1; NB2 needs the INT4 model attached (MEDIUM-2) |
| `E12_REQUEST_LOGS` | False | False | identical |
| `E12_SANDBOX_MEM_MB` / memory-limit self-test / fail-closed on posix | 4096 / yes / yes | same | identical |
| sandbox self-test task `E15B_SELFTEST_TASK` | fixed identity task | same | identical |
| Duck package | `q27x_e15b_duck_9e4b89285136.bin`, sha 9e4b8928…6 | same name + sha | identical; I verified the payload sha on two local copies |
| package manifest (23 files incl. `arc2_duck.py` b9f67010…, `tool_agent.py` f7454201…, `python_tool_sandbox.py` d29d807e…) | — | same dict | identical; the extracted files (`_scratch/donor_map/duck_9e4b`) hash equal to every manifest entry |
| E14 modules payload `q27x_e14_modules_55c1eb7911c6.bin` + the 5 module shas | — | same | identical |
| `solve_all_r6` arguments | tasks in E12_ORDER, order, out_dir, deadline, base_url, model_id, branches 2, max_generated_tokens 50,000, save_request_logs False, on_branch_done r6, workers, first_exposure None, b1_view facts, min_admit 300, capacity_probe, kv_admit 0.95, kv_poll 5.0, min_chunk 12,288, fair_share False, stage2_p90_cutoff False, join_grace 90, max_b1 45,000, skip_b1 False, extra True, max_attempts 8, b1 mode before3, weak_commit True | the same text | identical (the deadline is bounded by the half's end; the capacity_probe reads vLLM gauges) |
| ORDER2 task order | `(-[≥3 train] + [1 test], total cells, index)` over `TASK_IDS = sorted(sample(seed 20261005, 120))` | same cells | identical (the order cell is byte-identical; CELL 6's sort is unchanged; the profile has n_tasks 120 and the same seed) |
| isolation preflight → workers | PASS → 16 | the same code | identical logic |
| deadline guard `E12_STOP_BEFORE_HARD_SEC` | 180 | 180 | identical |
| CELL 0 (clock anchor, fallback submission, dual mode) | — | byte-identical | identical |
| CELL 2 install | tokenizer-only (vLLM not installed) | full `vllm==0.19.0` into the kernel | engine move. The E12-V run-2 install log shows numpy (2.0.2 in the image), scipy and torch are not reinstalled, so the sandbox's numeric stack is the same. numba 0.61.2, llvmlite, protobuf 6.33.6 and setuptools are added or changed (negligible for model code) |
| commit submission (identity) / rerun submission (from records) | — | the same code | identical |
| prompts, views, nudges, harvest, auto-check (all in `arc2_duck.py`) | — | the same file | identical |

## 4. README donor map check

The README map is correct in every listed item:
- CELL 0, CELL 2 and the order cell are verbatim;
- CELL 1b and CELL 3 changes are exact;
- CELL 6 and CELL 7 changes are listed as changed; the scheduler call is unchanged;
- the engine donors and their shas are right;
- Finding 2 (Duck package 9e4b, CELL 3 line 9) is confirmed;
- Finding 6 (R3K2 server running mean 15.62) is confirmed;
- Finding 7 (E12-V turn 2: 4,800 / 19,215 cached) is confirmed.

It is incomplete in the points listed under LOW-7. It also has no list of required Kaggle inputs (MEDIUM-2), and no statement that half A's window can be shorter than half B's (MEDIUM-1).

## 5. Limits of this audit

- The vLLM 0.19 / 0.32 source line references in README Finding 1 were not checked (no vLLM source on disk). The behavioural evidence (E12-V run-2 prompt growth) was checked.
- The equality of the INT4 and FP8 chat templates and tokenizers (README Finding 4) was not re-checked. The Kaggle CPU kit is meant to check it on the real files.
- The CPU harness and the CPU kit were not run or reviewed (out of scope).
- Engine flags were compared against their donors, not parsed by a real vLLM argument parser.

## Appendix A: full unified diffs, R3K2 (c058d58b) → E21 NB1 (ad5b2d1f)

CELL 6 and CELL 7 are shown after removing the wrapper: the `def e21_cellN():` line, the doc-comment line and the `global` line are removed, and 4 spaces are de-indented. Unchanged cells (CELL 0, CELL 2, the order cell) are omitted because they are byte-identical.

```diff
--- R3K2/CELL1 (c058d58b cell)
+++ E21-NB1/CELL1 (ad5b2d1f)
@@ -1,5 +1,5 @@
 # ==================================================================== CELL 1 — configuration
-PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", "E15BEVAL120R3I").strip().upper()
+PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", "E21X2EVAL120").strip().upper()
 PROFILE_TABLE = {
     # commit run: 12 public evaluation tasks, scored; also measures throughput for the 12-hour projection
     "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},
@@ -113,6 +113,11 @@
     "E15BEVAL120R3I": {"source": "eval", "n_tasks": 120, "time_budget_sec": 6.0 * 3600, "admit_frac": 0.90,
                        "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
                        "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "hidden_equiv": True},
+    # E21 (user 2026-10-28): two halves (vLLM 0.19, then 0.32) on all 120 evaluation tasks in one 12-h commit; each half
+    # gets a fixed solve window from its own start (E21_HALF_SOLVE_SEC); the hidden rerun = RERUN_ALL with half A only
+    "E21X2EVAL120": {"source": "eval", "n_tasks": 120, "time_budget_sec": 12.0 * 3600 - 3 * 60, "admit_frac": 0.90,
+                     "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
+                     "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "solve_window_sec": 5.5 * 3600},
     "E14SEVAL120G4B": {"source": "eval", "n_tasks": 120, "time_budget_sec": 6.5 * 3600, "admit_frac": 0.90,
                        "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
                        "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "hidden_equiv": True},
@@ -386,3 +391,39 @@
     CONFIG["e12"]["rev6"] = dict(E12_R6)
 CONFIG["e12"]["duck_commit"] = "10882e332d883af3e61616df8c8a17d03a328a75"
 print("E12:", json.dumps(CONFIG["e12"]))
+
+
+# ==================================================================== E21 (Experiment 21, user 2026-10-28)
+# Duck config B (= R3K2, everything above unchanged) on vLLM OpenAI servers. Half A = vLLM 0.19.0 in this kernel (the
+# S1-60K engine, launched as the E12-V Duck server); half B = vLLM 0.32.0 in /tmp/vllm-venv (the VMS installer and arm).
+# Both halves solve the same 120 evaluation tasks with fresh Duck state; the hidden rerun runs half A only.
+E21_VARIANT = "NB1"
+E21_HALF_SPECS = {"NB1": {"A": {"vllm": "0.19.0", "weights": "int4", "seqs": 20, "ssm": None,
+              "fallback": {"weights": "fp8", "seqs": 16, "ssm": None}},
+        "B": {"vllm": "0.32.0", "weights": "int4", "seqs": 20, "ssm": None,
+              "fallback": {"weights": "fp8", "seqs": 16, "ssm": None}}}}
+E21_HALVES = ["A"] if RERUN else ["A", "B"]
+ENGINE_BACKEND = "vllm"                       # CELL 1b stages the variant's weights; CELL 2 installs vllm==0.19.0 here
+E21_STAGE_PATHS = [SGL_MODEL_PATH]        # the weights both halves load (NB1 INT4, NB2 FP8)
+E12_SOLVE_WINDOW_SEC = (float(os.environ.get("E21_HALF_SOLVE_SEC", P["solve_window_sec"]))
+                        if P.get("solve_window_sec") else None)          # per half, from its own solve start
+E21_HALF_SOLVE_SEC = E12_SOLVE_WINDOW_SEC or 0.0
+E21_B_SETUP_RESERVE_SEC = float(os.environ.get("E21_B_SETUP_RESERVE_SEC", 25 * 60))   # switch + 0.32 install + launch + report
+E21_LAUNCH_TIMEOUT_SEC = 30 * 60
+E21_VLLM_MAX_MODEL_LEN = 135168               # E12-V: Duck's 128 Ki context window + margin
+E21_PORTS = {"A": 1234, "B": 8011}
+E21_SERVED = {"fp8": "Qwen/Qwen3.8-27B-FP8", "int4": "Qwen/Qwen3.8-27B-INT4"}   # never "qwen" (Duck alias, E12-V run 1)
+E18_VLLM_WHEELHOUSE = Path(os.environ.get("E18_VLLM_WHEELHOUSE", "/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"))
+E18_SILENCE_SEC = float(os.environ.get("E18_SILENCE_SEC", 600))
+E21_FAKE = os.environ.get("E21_FAKE_SERVER") if os.environ.get("E11_HARNESS") == "1" else None   # CPU harness only
+E18_FAKE = E21_FAKE
+E12_R6["E12_R6_CACHE_DIAG"] = 0               # the SGLang cache diagnostic (after solving) is not run on vLLM
+os.environ["E12_R6_CACHE_DIAG"] = "0"
+if E12_SOLVE_WINDOW_SEC and not RERUN:
+    assert 2 * E12_SOLVE_WINDOW_SEC + E21_B_SETUP_RESERVE_SEC < HARD_STOP - NOTEBOOK_T0, "two windows must fit"
+CONFIG["e21"] = {"variant": E21_VARIANT, "halves": E21_HALVES, "specs": E21_HALF_SPECS[E21_VARIANT],
+                 "half_solve_sec": E12_SOLVE_WINDOW_SEC, "b_setup_reserve_sec": E21_B_SETUP_RESERVE_SEC,
+                 "max_model_len": E21_VLLM_MAX_MODEL_LEN, "served": E21_SERVED, "ports": E21_PORTS,
+                 "stage": [str(p) for p in E21_STAGE_PATHS], "vllm032_wheelhouse": str(E18_VLLM_WHEELHOUSE),
+                 "reasoning_history_key": "reasoning"}
+print("E21:", json.dumps(CONFIG["e21"], default=str), flush=True)
--- R3K2/CELL1b (c058d58b cell)
+++ E21-NB1/CELL1b (ad5b2d1f)
@@ -65,14 +65,14 @@
 E14_RESULTS["env"] = E14L.env_report([str(SGL_MODEL_PATH), str(SGL_DRAFT_PATH), str(SGL_WHEELHOUSE), str(MODEL_PATH),
                                       "/kaggle/input"])
 try:
-    E14_RESULTS["read_probe"] = E14L.read_probe(SGL_MODEL_PATH) if ENGINE_BACKEND == "sglang" else None
+    E14_RESULTS["read_probe"] = E14L.read_probe(E21_STAGE_PATHS[0])   # E21: the weights this variant loads
 except Exception as _e:
     E14_RESULTS["read_probe"] = {"error": repr(_e)[:200]}
 print("E14 env:", json.dumps(E14_RESULTS["env"])[:1500], "| probe:", json.dumps(E14_RESULTS["read_probe"]), flush=True)
 E14_STAGING = None
-if ENGINE_BACKEND == "sglang" and E14L.loadpath_enabled():
+if ENGINE_BACKEND in ("sglang", "vllm") and E14L.loadpath_enabled():       # E21: vLLM halves too
     _e14_dest = tuple(x for x in os.environ.get("E14_DEST", "/dev/shm,/tmp").split(",") if x)
-    E14_STAGING = E14L.Staging([SGL_MODEL_PATH, SGL_DRAFT_PATH], workers=24, stall_sec=600.0,
+    E14_STAGING = E14L.Staging(E21_STAGE_PATHS, workers=24, stall_sec=600.0,
                                deadline=max(NOTEBOOK_T0 + 10 * 60, min(NOTEBOOK_T0 + 45 * 60, SGL_GIVE_UP_AT - 20 * 60)), log=print, state=E14_STATE,
                                dest_candidates=_e14_dest)   # E15B #27
     E14_STAGING.start()
@@ -126,6 +126,6 @@
 E14_TS = float(os.environ.get("E14_TIME_SCALE", "1"))    # the CPU harness shortens every window and phase cap; 1 on Kaggle
 _ts = lambda s: max(1.5, float(s) * E14_TS)
 CONFIG["e14"] = {"measure": E14_DO_MEASURE, "staging": E14_STAGING is not None, "raw_reasoning": True,
-                 "submit_arm": None if E14_DO_MEASURE else "DFlash width 6, 16 streams: mrr 17, graphs 1..17, mamba 68, min-free-slots-delay 1 (Exp 15 Notes/02)",
+                 "submit_arm": "E21: vLLM halves, see CONFIG['e21']",
                  "modules": _E14_MOD_SHA, "arc2_duck_sha": "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128"}
 e14_save()
--- R3K2/CELL3 (c058d58b cell)
+++ E21-NB1/CELL3 (ad5b2d1f)
@@ -46,6 +46,11 @@
 # Duck reads its settings at import: set them first. The base URL is the engine cell's (same port); the stream count is
 # set again from the engine's concurrency before the gate is first used (read at call time).
 os.environ.update(E12A.duck_env(f"http://127.0.0.1:{int(os.environ.get('E11_PORT', 30000))}/v1", "pending", SGL_MAX_RUNNING))
+# E21: vLLM 0.19 / 0.32 read only message["reasoning"] of an assistant history message (vllm/entrypoints/chat_utils.py
+# 1512 in v0.19.0; vllm/renderers/chat_utils.py 2083 in v0.32.0); "reasoning_content" (DUCK_ENV, read by SGLang) would
+# be dropped from the prompt. E12-V run 2 sent "reasoning" and its prompts grew by the generated tokens (6,882 + 12,288
+# -> 19,215). Duck reads the key at call time (tool_agent._reasoning_history_keys).
+os.environ["ARC3_REASONING_HISTORY_KEY"] = "reasoning"
 _E12_TA, _E12_DP, _e12_run_sandboxed, _ = E12A._import_duck()
 E12_SANDBOX_LIMIT = E12A.install_sandbox_memory_limit(E12_SANDBOX_MEM_MB)   # rev 2: every sandbox child, before its code
 
--- R3K2/CELL6 (c058d58b cell)
+++ E21-NB1/CELL6 (ad5b2d1f)
@@ -92,16 +92,11 @@
 
 
 def _e15b_gen_tokens_total():
-    # E15B #32: the server's own generated-token counter, to count tokens the report misses. B6 (auditor 2 F6): max over
-    # the TP ranks of each label set, summed over the other label rows (e.g. is_streaming)
+    # E15B #32 on vLLM (E21): the server's own generated-token counter (vllm:generation_tokens_total, summed over rows)
     try:
-        import re as _re
-        _g = {}
-        for r in e11_prom():
-            if getattr(r, "name", "") == "sglang:generation_tokens_total" and getattr(r, "value", None) is not None:
-                _k = _re.sub(r'tp_rank="[^"]*",?', "", str(getattr(r, "labels", "") or ""))
-                _g[_k] = max(_g.get(_k, 0.0), float(r.value))
-        return sum(_g.values()) if _g else None
+        _v = [float(r.value) for r in e21_prom() if getattr(r, "name", "") == "vllm:generation_tokens_total"
+              and getattr(r, "value", None) is not None]
+        return sum(_v) if _v else None
     except Exception:
         return None
 
@@ -122,82 +117,52 @@
 
 
 def _e15b_server_log_summary(text):
-    # E15B #42 / #53: the engine block from the server log: running-level histogram of the decode lines, queue > 0
-    # share, gen throughput, retract / preempt lines (server_args excluded), OOM / Traceback
+    # E15B #42 / #53 on vLLM (E21): vLLM's 10-s engine stats lines (E12-V _E12V_STAT_RE) instead of SGLang decode lines:
+    # running-level histogram (lines with work), queue > 0 share, generation throughput, prefix-cache hit rate, KV usage,
+    # MTP acceptance, preemption / OOM / Traceback lines, and the startup KV line
     import re as _re
-    _hist, _q, _tp, _n, _hq, _acc, _tpl = {}, 0, [], 0, {}, [], {}
-    for _l in text.splitlines():
-        if "Decode batch" not in _l:
-            continue
-        _m = _re.search(r"#running-req: (\d+)", _l)
-        _mq = _re.search(r"#queue-req: (\d+)", _l)
-        _mt = _re.search(r"gen throughput \(token/s\): ([\d.]+)", _l)
-        if _m:
-            _n += 1
-            _hist[int(_m.group(1))] = _hist.get(int(_m.group(1)), 0) + 1
-            if _mq and int(_mq.group(1)) > 0:     # A1 (Notes/07): the running level while requests wait in the server
-                _q += 1
-                _hq[int(_m.group(1))] = _hq.get(int(_m.group(1)), 0) + 1
-            if _mt:
-                _tp.append(float(_mt.group(1)))
-                _tpl.setdefault(int(_m.group(1)), []).append(float(_mt.group(1)))     # B6: tok/s per running level
-            _ma = _re.search(r"accept len: ([\d.]+)", _l)
-            if _ma:
-                _acc.append(float(_ma.group(1)))
+    rows = [tuple(float(x) for x in m.groups()) for m in _E12V_STAT_RE.finditer(text)]
+    work = [r for r in rows if r[2] > 0 or r[3] > 0]
+    _hist, _hq, _tpl = {}, {}, {}
+    for r in work:
+        k = int(r[2])
+        _hist[k] = _hist.get(k, 0) + 1
+        if r[3] > 0:
+            _hq[k] = _hq.get(k, 0) + 1
+        _tpl.setdefault(k, []).append(r[1])
+    _tp = sorted(r[1] for r in work)
+    _acc = sorted(float(a) for a in _re.findall(r"Mean acceptance length: ([\d.]+)", text))
+    _kv = _re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
     _bad = lambda k: sum(1 for _l in text.splitlines() if k in _l.lower() and "server_args" not in _l.lower()
-                         and "retraction_policy" not in _l)
-    _tp.sort()
-    # A3 (Notes/07): the startup split from TP0's lines (weight loads, each CUDA graph capture, KV pool, log start -> ready)
-    import datetime as _dt
-    _st = {"load_sec": {}, "graph_capture": {}, "kv_tokens": None, "ready_span_sec": None}
-
-    def _ts(_l):
-        _m = _re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", _l)
-        return _dt.datetime.strptime(_m.group(1), "%Y-%m-%d %H:%M:%S") if _m else None
-    _t_first = _t_ready = None
-    for _l in text.splitlines():
-        _t = _ts(_l)
-        if _t is not None and _t_first is None:
-            _t_first = _t
-        if "The server is fired up and ready to roll" in _l and _t is not None and _t_ready is None:
-            _t_ready = _t
-        if "TP0]" not in _l:
-            continue
-        _m = _re.search(r"Load weight end\. elapsed=([\d.]+) s, type=(\w+)", _l)
-        if _m:
-            _st["load_sec"][_m.group(2)] = float(_m.group(1))
-        _m = _re.search(r"Capture (.+?) CUDA graph end\. elapsed=([\d.]+) s, mem usage=([\d.]+) GB, avail mem=([\d.]+) GB", _l)
-        if _m:
-            _st["graph_capture"][_m.group(1)] = {"sec": float(_m.group(2)), "mem_gb": float(_m.group(3)),
-                                                 "avail_gb_after": float(_m.group(4))}
-        _m = _re.search(r"KV Cache is allocated\. dtype: \S+, #tokens: (\d+)", _l)
-        if _m and _st["kv_tokens"] is None:
-            _st["kv_tokens"] = int(_m.group(1))
-    if _t_first is not None and _t_ready is not None:
-        _st["ready_span_sec"] = (_t_ready - _t_first).total_seconds()
-    _st["graph_capture_sec_total"] = round(sum(v["sec"] for v in _st["graph_capture"].values()), 2)
+                         and "non-default args" not in _l.lower())
+    _n = len(work)
     return {"decode_lines": _n, "running_hist": dict(sorted(_hist.items())),
-            "running_hist_queue_gt0": dict(sorted(_hq.items())), "startup": _st,
-            "accept_len_p50": sorted(_acc)[len(_acc) // 2] if _acc else None,
+            "running_hist_queue_gt0": dict(sorted(_hq.items())),
+            "startup": {"kv_tokens": int(_kv[0].replace(",", "")) if _kv else None},
+            "accept_len_p50": _acc[len(_acc) // 2] if _acc else None,
             "gen_tok_s_p50_by_running": {k: sorted(v)[len(v) // 2] for k, v in sorted(_tpl.items())},
             "running_mean_lines": round(sum(k * v for k, v in _hist.items()) / _n, 2) if _n else None,
-            "queue_gt0_line_share": round(_q / _n, 3) if _n else None,
+            "queue_gt0_line_share": round(sum(_hq.values()) / _n, 3) if _n else None,
             "gen_tok_s_p50": _tp[len(_tp) // 2] if _tp else None,
+            "gen_tok_s_mean": round(sum(_tp) / len(_tp), 1) if _tp else None,
+            "prefix_cache_hit_pct_last": rows[-1][5] if rows else None,
+            "kv_usage_pct_max": max((r[4] for r in rows), default=None),
             "retract_lines": _bad("retract"), "preempt_lines": _bad("preempt"), "oom_lines": _bad("out of memory"),
             "traceback_lines": sum(1 for _l in text.splitlines() if "Traceback" in _l)}
 
 
-def _e12_capacity_probe():   # rev 6e G4-1: exact names; the max over TP ranks; never sglang:token_usage (max of pools)
+def _e12_capacity_probe():   # E21: vLLM 0.19 / 0.32 gauges (the max over label rows); vLLM has no separate Mamba gauge
     try:
-        rows = [r for r in e11_prom() if getattr(r, "value", None) is not None]
+        rows = [r for r in e21_prom() if getattr(r, "value", None) is not None]
     except Exception:
         rows = []
     pick = lambda name: [float(r.value) for r in rows if r.name == name]
     out = {}
-    for key, name in (("kv_usage", "sglang:full_token_usage"), ("mamba_usage", "sglang:mamba_usage"),
-                      ("running", "sglang:num_running_reqs"), ("queue", "sglang:num_queue_reqs")):
+    for key, name in (("kv_usage", "vllm:kv_cache_usage_perc"), ("running", "vllm:num_requests_running"),
+                      ("queue", "vllm:num_requests_waiting")):
         v = pick(name)
         out[key] = max(v) if v else None
+    out["mamba_usage"] = None
     return out
 
 
@@ -445,7 +410,7 @@
             print("E12: /v1/models failed, using the model path:", repr(_e)[:200], flush=True)
         E12_RUN["streams"] = min(int(CONCURRENCY), E12_STREAMS or int(CONCURRENCY))
         os.environ["ARC3_MAX_ACTIVE_STREAMS"] = str(E12_RUN["streams"])
-        _end = HARD_STOP - E12_STOP_BEFORE_HARD_SEC
+        _end = min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC)     # E21: this half's latest end
         if E12_SOLVE_WINDOW_SEC:
             _end = min(_end, time.perf_counter() + E12_SOLVE_WINDOW_SEC)
         if P.get("hidden_equiv"):   # gate 4b (user 2026-10-21): this run gets the hidden run's compute per task
@@ -491,7 +456,7 @@
             # preflight, bounded by the hard stop; the per-task compute of gate 4b stays the hidden run's. The hidden
             # rerun keeps its absolute end (no window). H6-3: nothing else runs before solving
             _shift = (time.perf_counter() - NOTEBOOK_T0) - E12_RUN["started_sec"]     # started_sec: set when _end was fixed
-            _end = _e12_shifted_end(_end, _shift, HARD_STOP - E12_STOP_BEFORE_HARD_SEC)
+            _end = _e12_shifted_end(_end, _shift, min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC))
             E12_RUN["window_shift"] = {"shift_sec": round(_shift, 1),
                                        "window_start_sec": round(time.perf_counter() - NOTEBOOK_T0)}
             if E12_RUN.get("g4b"):
@@ -516,7 +481,9 @@
         E12_RUN["watchdog"]["limits"] = {"poll_sec": _WD_POLL, "fail_sec": _WD_FAIL, "min_left_sec": _WD_MIN_LEFT}
 
         def _e15b_watchdog():
-            global E11_SERVER, SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING
+            # E21: the E15B watchdog on vLLM: a dead server (or /health failing for _WD_FAIL s while its log is silent)
+            # is relaunched once with the same half's ready config (e21_relaunch, CELL 5); while it is down the
+            # branches wait (arc2_duck.SERVER_DOWN) instead of burning their consecutive-failure budget
             _bad, _sz_last = None, {}
             _wd = E12_RUN["watchdog"]
             while not _E12_STOP.wait(_WD_POLL):
@@ -526,8 +493,8 @@
                     _tk = _e15b_gen_tokens_total()      # B6: the last counter value of THIS server (it resets on relaunch)
                     if _tk is not None:
                         _wd["tok_last"] = _tk
-                    try:     # B2 (auditor 2 F2 / F8): a growing server log = alive (a prefill storm logs batches, decode
-                        _sz = os.path.getsize(_srv.log_path)     # logs every 40 steps); /health only when it is silent
+                    try:     # B2: a growing server log = alive; /health only when it is silent
+                        _sz = os.path.getsize(_srv.log_path)
                     except Exception:
                         _sz = None
                     if _sz is not None and _sz != _sz_last.get(id(_srv)):
@@ -556,37 +523,14 @@
                 print("E15B watchdog: server down -> relaunch", json.dumps(_rec), flush=True)
                 try:
                     try:
-                        _tail = _srv.log_text()[-20000:].lower()
-                    except Exception:
-                        _tail = ""
-                    try:
-                        _srv.kill()
+                        _wd["old_logs"].append(str(_srv.log_path))
                     except Exception:
                         pass
-                    _wd["old_logs"].append(str(_srv.log_path))
                     if _wd.get("tok_last") is not None and _wd.get("tok_base") is not None:   # B6
                         _wd["tok_acc"] += max(0.0, _wd["tok_last"] - _wd["tok_base"])
                     _wd["tok_base"], _wd["tok_last"] = 0.0, None
-                    _e11_gpu_free()
-                    # B3 (auditor 2 F3, auditor 3 R1): a 16-stream-specific crash (OOM / RuntimeError) or a second relaunch
-                    # -> the 12-stream flags (the clients then queue up to 4 requests at the server)
-                    _cfg0 = dict(E14_VENV.get("cfg") or {})
-                    _cmd = list(_srv.cmd)
-                    if (any(k in _tail for k in E11_OOM_MARKERS) or "runtimeerror" in _tail or _wd["relaunches"]) \
-                            and _cfg0 and not _cfg0.get("streams12") and not _cfg0.get("w1") and E14_VENV.get("launch"):
-                        _cfg1 = dict(_cfg0, streams12=True, mamba_cache=56)
-                        SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING = 12, 13
-                        _cmd = list(E14_VENV["launch"]) + e11_flags(_cfg1)
-                        E14_VENV["cfg"] = _cfg1
-                        _rec["streams12"] = True
-                    _new = E11Server(_cmd, _srv.env, f"relaunch{len(E12_RUN['watchdog']['relaunches'])}")
-                    _new.start()
-                    _ok, _why = _new.wait_ready(min(SGL_RELAUNCH_TIMEOUT_SEC, max(60.0, _left - _WD_MIN_LEFT / 2)))
+                    _ok, _why = e21_relaunch()
                     _rec.update(ok=_ok, why=_why)
-                    if _ok:
-                        E11_SERVER = _new
-                    else:
-                        _new.kill()
                 except Exception as _e:
                     _rec.update(ok=False, why=repr(_e)[:300])
                 finally:
--- R3K2/CELL7 (c058d58b cell)
+++ E21-NB1/CELL7 (ad5b2d1f)
@@ -224,7 +224,7 @@
                                                             or q.get("template_same") is False),
             "cached_not_reported_requests": sum(1 for q in reqs if q.get("cached_reported") is False),
             "cache_null_detail_requests": sum(1 for u in us if u.get("cached") is None),
-            "cache_note": "aggregate over all requests; a missing prompt_tokens_details means 0 cached (SGLang 0.5.21)",
+            "cache_note": "aggregate over all requests; a missing prompt_tokens_details means 0 cached (vLLM 0.19 / 0.32, as SGLang 0.5.21)",
             "concurrency": _r6e_concurrency()}
 
     def _zero_classes(zero):   # H7-13
@@ -385,7 +385,7 @@
           "| task-norm", report.get("eval_task_norm_pct"), "| accepted branches", E12_RUN["branches_accepted"],
           "| cached", report["usage_all"].get("cached_pct"), "%", flush=True)
     e12_write_submission()
-    _pack = WORKING / "Q27X_E12_DUCK_BUNDLE.zip"
+    _pack = WORKING / f"Q27X_E21_{E21_VARIANT}_HALF{E21_HALF}_DUCK_BUNDLE.zip"
     with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
         for fp in (SUBMISSION_PATH, OUT / "e12_run_report.json", OUT / "e12_generated_attempts.json",
                    OUT / "sglang_decode_summary.json", OUT / "sglang_requests.jsonl"):
```

## Appendix B: NB1 (ad5b2d1f) vs NB2 (b968e322)

Code cells c01 and c03-c10 are byte-identical. CELL 1:

```diff
--- NB1/CELL1
+++ NB2/CELL1
@@ -397,14 +397,14 @@
 # Duck config B (= R3K2, everything above unchanged) on vLLM OpenAI servers. Half A = vLLM 0.19.0 in this kernel (the
 # S1-60K engine, launched as the E12-V Duck server); half B = vLLM 0.32.0 in /tmp/vllm-venv (the VMS installer and arm).
 # Both halves solve the same 120 evaluation tasks with fresh Duck state; the hidden rerun runs half A only.
-E21_VARIANT = "NB1"
-E21_HALF_SPECS = {"NB1": {"A": {"vllm": "0.19.0", "weights": "int4", "seqs": 20, "ssm": None,
-              "fallback": {"weights": "fp8", "seqs": 16, "ssm": None}},
-        "B": {"vllm": "0.32.0", "weights": "int4", "seqs": 20, "ssm": None,
-              "fallback": {"weights": "fp8", "seqs": 16, "ssm": None}}}}
+E21_VARIANT = "NB2"
+E21_HALF_SPECS = {"NB2": {"A": {"vllm": "0.19.0", "weights": "fp8", "seqs": 16, "ssm": "float16",
+              "fallback": {"ssm": None}},
+        "B": {"vllm": "0.32.0", "weights": "fp8", "seqs": 16, "ssm": "float16",
+              "fallback": {"ssm": None}}}}
 E21_HALVES = ["A"] if RERUN else ["A", "B"]
 ENGINE_BACKEND = "vllm"                       # CELL 1b stages the variant's weights; CELL 2 installs vllm==0.19.0 here
-E21_STAGE_PATHS = [SGL_MODEL_PATH]        # the weights both halves load (NB1 INT4, NB2 FP8)
+E21_STAGE_PATHS = [MODEL_PATH]        # the weights both halves load (NB1 INT4, NB2 FP8)
 E12_SOLVE_WINDOW_SEC = (float(os.environ.get("E21_HALF_SOLVE_SEC", P["solve_window_sec"]))
                         if P.get("solve_window_sec") else None)          # per half, from its own solve start
 E21_HALF_SOLVE_SEC = E12_SOLVE_WINDOW_SEC or 0.0
```

Markdown cell: the title says NB1 / NB2, and the weights line is "RedHat INT4 (W4A16) + MTP-3, 20 streams (fallback: FP8 MTP-3, 16 streams)" vs "FP8 + MTP-3 + `--mamba-ssm-cache-dtype float16`, 16 streams (fallback: SSM dtype default = S1's engine)".
