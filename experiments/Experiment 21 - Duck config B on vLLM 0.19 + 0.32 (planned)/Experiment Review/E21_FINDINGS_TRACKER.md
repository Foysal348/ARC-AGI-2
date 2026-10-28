# E21 findings tracker: every finding of the 10 validation reports, with its status (2026-10-28)

**Notebooks after the fixes:** rev 2b, NB1 `84876a8b`, NB2 `9b6e7359`.

**Status values:**
- **FIXED** — in code;
- **MEASURED** — not changeable, reported per half;
- **NOTE** — documented; no code change needed;
- **OPEN-USER** — needs the user's decision;
- **DEFERRED** — consciously not done, with the reason.

The reports are `VAL_engine_A1/A2`, `VAL_state_B1/B2`, `VAL_timing_C1/C2`, `VAL_api_D1/D2` and `VAL_fidelity_E1/E2`.

**User correction (2026-10-28):** the first rev-2 pass silently skipped several findings. Rev 2b is the pass over every heading of every report. Rows marked "2b" were missing from rev 2.

## HIGH

| Finding | Reports | Status | Where |
|---|---|---|---|
| Duck's process-wide priority gate (slots + clock) carries over into half B | B1 H1, B2 HIGH-1, C2 HIGH-1, E1 HIGH-1, E2 HIGH-1 | FIXED | `e21_begin_half`: `_PRIORITY_GATE = None`, `_GATE_STATS` reset; gate stats per half in the report |
| vLLM returns finish "tool_calls" for a call cut at max_tokens; Duck's cut handling keys on "length" | D1 F1 (source-verified 0.19 L1550-1560, 0.32 L1099-1116) | FIXED | CELL 3 patch `_e21_chat_completion`; `length_restored` per half |
| No real check of the Duck tool/reasoning contract on 0.32 | D1 F2, A1 M1 | FIXED | `e21_contract_probe`: 2 turns, tools + thinking, history prompt growth |
| Smoke accepts any non-empty text ("!!!!") | A2 H1, A1 M2, C2 M4 | FIXED | "4" required, non-degenerate |
| Capacity ignores the 15 fixed Mamba blocks per request | A1 H1, A2 M1, D2 F3, E1 M3 (fail-open) | FIXED | `e21_capacity`: block-based, checked on the E12-V log (846 blocks); fail-closed; refuses < 8 streams |
| 0.32 without `--language-model-only` | A1 H2, A2 M2, E2 M1 | FIXED | `e21_flags_v032` |
| flagcheck cannot import the 0.32 parser | A1 H3, A2 M5, D1 F7 | FIXED | import fallback. Also (CPU kit run 1) the parser itself needs a device → stub platform |
| NB1 FP8 fallback not staged (cold GCSFUSE) | A2 H2, A1 L7, C1 M4, E2 LOW-6 | FIXED | NB1 `E21_STAGE_PATHS = [INT4, FP8]` |
| vLLM align mode never caches a previous reply (+~11 M prefill per half) | D2 F1, D1 F3 | MEASURED | `recomputed_prompt_tokens`, `prompt_tokens`, whole-run prefix hits. Levers → OPEN-USER below |

## MEDIUM

| Finding | Reports | Status | Where / reason |
|---|---|---|---|
| Unequal windows (A shorter when its setup > 27 min) | C1 M2, C2 M1, B1 M2, B2 MEDIUM-1, E1 MEDIUM-1, A1 M7, A2 M4 | FIXED | B gets A's achieved window; `windows_equal` + warning |
| **2b** B cut by its own setup (the reverse case) | C1 M3, C2 3.2, E2 LOW-7 | FIXED | W 5.5 → 5.45 h; reserve 25 → 30 min; 0.32 venv built in the background during half A; 0.32 ready timeout 20 min; preflight request timeout 180 → 60 s |
| Teardown after the last half near the 12-h limit | C1 M1, C2 M2, B1 L5, B2 LOW-5 | FIXED | last half: kill fast, no GPU wait |
| Relaunch timeout not the donor's (60 s late in a half) | B1 M1, A1 M6, A2 L1, E1 LOW-3, C1 L1, D2 F14 note | FIXED | `e21_relaunch`: min(30 min, end − 5 min); none under 12 min |
| GPU release never acted on | A1 M3, A2 L3, B1 L4, B2 LOW-6, C2 LOW-8 | FIXED | `e21_gpus_free` before every launch |
| 0.32 install failure not cached (2 × 40 min) | A1 M4, A2 L2, B1 L7, C1 M3 | FIXED | cached error; background install |
| `e21_end_half` kill not in `finally` | A1 M5, B1 L2 | FIXED | — |
| Stats regex drops Preemptions / Deferred lines | D2 F2, A1 L2 | FIXED | rev 2 regex + `e21_stat_rows` (both consumers) |
| Watchdog "log grows = alive" defeated by our /metrics polls | D2 F4 | FIXED | `--disable-uvicorn-access-log` (exists in 0.19 and 0.32) |
| **2b** Isolation preflight never exercises the prefix-cache path | D2 F5 | FIXED | `e21_cache_probe`: a 9k shared prefix, repeat must hit ≥ 1 block, own marker, equal at T 0 |
| Preemptions reported as ~0 (no per-event log; the counter is lost at kill) | A2 M3, D2 F6 | FIXED | counters snapshot before every kill (`e21_counters`) |
| **2b** NB2 Duck tokenizer depends on the INT4 file | E1 MEDIUM-2, E2 MEDIUM-2 | FIXED | falls back to the FP8 `tokenizer.json` (identical encode) if INT4 is missing; inputs list in the README |
| **2b** INT4 tokenizer_config (transformers-5 format) never loaded by transformers 4.57.6 | A2 M6 | FIXED (kit) | `src/e21_tokcheck.py` in the kit, with the kernel's transformers (the 0.19 server) and the 0.32 venv: both folders load, render a Duck-like conversation to identical ids, `AutoProcessor` recorded |
| NB1: if A fell back, the A/B is confounded | C2 MEDIUM-3 | OPEN-USER (built as option ka) | `E21_B_FOLLOWS_A = True` |
| Half B flags differ from A (LMO) | E2 M1 | FIXED | — |

## LOW

| Finding | Reports | Status | Where / reason |
|---|---|---|---|
| `e11_kill_server(wait=)` vs `kill(fast=)` | A1 L1, B2 LOW-4, C1 L4, C2 L1, D2 F11 | FIXED | override (+ counter snapshot) |
| `/v1/models` fallback model id is a path (404) | B1 L6, B2 LOW-3, D2 F13, E2 LOW-4 | FIXED | served name |
| `E12_RECORDS` / `E12_RUN` / `ENGINE_INFO` not reset first | B1 L3, B2 LOW-2, C2 L5 | FIXED | reset at the top of `e21_begin_half` |
| Half-A watchdog thread may act in half B | B1 L1, B2 LOW-1, A1 L5, D2 F14 | FIXED | join in `e21_end_half`; failure path never kills the GPUs of another half; `E11_SERVER` cleared only if it is this server |
| `srv.start()` outside `try` | A1 L6 | FIXED | — |
| `wrap()` blind spots (f-strings, match, del, exec, locals, docstring) | B1 L9, B2 LOW-7, E2 LOW-2 | FIXED | fail-closed token / AST / symtable checks |
| PUSHED guard (full sha vs 8-hex; size after write) | E2 LOW-3 | FIXED | — |
| CONFIG records stale values | E1 LOW-1, E2 LOW-1, B2 LOW-8, C1 L6 | FIXED | refreshed after the E21 block |
| Window 0 / non-window profile starves B | C1 L3, A2 L6, E1 LOW-5, C2 L6, E2 LOW-8 | FIXED | asserts (profile, window > 0) |
| **2b** Second relaunch reuses the crashed config | A2 L5, E1 LOW-3 | FIXED | the second relaunch takes the half's other config |
| **2b** 0.19 and 0.32 share JIT caches | A2 L7 | FIXED | 0.32 gets its own `TRITON_CACHE_DIR` / `FLASHINFER_WORKSPACE_BASE` (0.19 keeps the proven defaults) |
| **2b** Duck env base URL points at port 30000 | E1 LOW-2 | FIXED | set per half (hygiene; agents get the URL explicitly) |
| **2b** `accept_len_median` key mismatch (CELL 7 reads it) | E1 LOW-6 | FIXED | alias key in the decode summary |
| **2b** 0.32 parser ends reasoning at a `<tool_call>` inside thinking | D1 F4, A1 M1 | MEASURED | `E21_CALL_WITH_CONTENT` (tool call + visible text) in the report |
| **2b** MTP acceptance as an INT4 MTP-head health signal | A1 M2 | MEASURED | warning when the median is < 1.5 |
| **2b** attempt-1-only score needs stage 1 done in both halves | C2 LOW-4 | MEASURED | `stage1` (attempt-1 records, missing, cut, all-done hour) + strict pass1 per half |
| Recorded flags always FLASHINFER; labels hardcoded | A2 L4 | FIXED | both backends recorded; backend label includes the real backend and spec count |
| kv_tokens means different things per version | A1 L3, D2 F8 | FIXED | `kv_tokens_logged` vs `effective_kv_tokens` / `num_blocks` / `block_size` |
| Prefix hit is a rolling rate | D2 F7 | FIXED | labelled rolling; whole-run counter hit % added |
| Harness cannot see H1 / L2 / cut calls / gate | A1 L4, C2 L9, B2 (gate test) | IN PROGRESS | harness agent: real block math, Preemptions lines, "tool_calls" cut, contract + cache probe, `gateAB` mode |
| Commit's final `submission.json` is half B's (identity) | A1 6, A2 L8 | NOTE | commit only; the rerun runs A only; per-half scores are in the reports |
| `_r6e_concurrency` / preflight drafter config / heartbeat keep filter are SGLang-shaped | D2 F9, F10, F12, E2 LOW-4, B1 L8 | DEFERRED | report-only fields: empty or not applicable on vLLM; the README says so. Changing donor code for cosmetics adds risk |
| KeyboardInterrupt swallowed by the runner | B1 L11 | DEFERRED | interactive only; commit and rerun are unaffected |
| Straggler callbacks not bound to their half | E1 LOW-4 | DEFERRED | timing makes it practically impossible (≥ 10 min between halves); Duck's publish drops records after close |
| CELL 2 install failure stops Run All (half B lost too) | E2 LOW-5 | DEFERRED | same as R3K2 / S1; the vLLM 0.19 install from the wheelhouse is proven (S1, E12-V, E15B) |
| A totally failed half A leaves ~5 h unused | C1 L5 | NOTE | fixed-window design, for comparability |
| Qualnames of wrapped functions | B1 L10 | NOTE | nothing pickles them |
| prompt + max_tokens > max-model-len | D1 F5 | NOTE | unreachable: Duck window 100k − cap |
| `prompt_tokens_details` presence differs | D1 F6 | NOTE | handled by arc2 |
| kwargs "replace" comment wrong | D1 F8 | NOTE | harmless |
| No lpm scheduling on vLLM | D1 F9, E1 LOW-8 | NOTE | engine difference |
| vLLM KV gauge = the whole hybrid pool incl. Mamba blocks; SGLang's was attention only | E1 LOW-8.5 | NOTE | the 0.95 admission gate therefore holds earlier than on SGLang; it is the true pool |
| INT4 brings calibrated fp8 KV scales (FP8 uses 1.0) | A2 L8 | NOTE | NB1 primary vs fallback also differ in KV quantisation |
| SSM fp16 halves the cache block (1,600 → 832) | A2 L8, D2 F1 | NOTE | NB2 primary vs fallback differ in prefix-cache granularity (seen in the 0.32 VMS log) |
| README Finding 5 wrong: GDN `in_proj_qkv` / `in_proj_z` / `out_proj` are INT4 (only `in_proj_a/b` + norm BF16) | A2 L8 | FIXED | README |
| README donor map inaccuracies and omissions (CELL 0 assert covers 3 donors, override of `e11_kill_server`, CELL 6 omissions, dropped list, dead-code list, shas, input list) | E1 LOW-7, E2 LOW-8 | FIXED | README rev 2b |

## OPEN-USER (decisions)

1. **NB1 half A fallback.** Should half B follow it? Built as yes (`E21_B_FOLLOWS_A`).
2. **Run both notebooks, or NB2 first** (quota ~24 GPU-h each)?
3. **vLLM prefix-cache levers** (D2 F1), not verified:
   - vLLM 0.32 `disable_eagle_block_drop` (experimental) would save the MTP block drop, about 3.4 M recompute per half;
   - SSM fp16 already halves the block in NB2.
   - Adding the 0.32 flag is an extra change to half B, so it needs a decision.
4. **A long-context greedy gate for SSM fp16 on vLLM 0.19** (A1 M2) costs GPU time; not built.

## GPT audits 1-3 (2026-10-28, on the rev 1 shas ad5b2d1f / b968e322) → rev 2c (NB1 `8bd3b2bf`, NB2 `6ed0cfb1`)

| GPT finding | Already fixed before? | rev 2c status |
|---|---|---|
| CRITICAL / BLOCKER: a half-A late or in-flight callback writes into half B's records and submission (all 3 audits, reproduced; verified on the source: `_e12_branch_done_r6` used the globals that half B rebinds) | no (B1 / B2 called the thread part LOW) | FIXED: the callback is bound to its half via default args (`_recs` / `_run` / `_lock` / `_half`); it never writes a submission for another half; counted in `late_cross_half_callbacks` |
| BLOCKER 1 (audit 1): a busy watchdog / heartbeat re-reads the re-created `_E12_STOP` and keeps running in half B | partly (rev 2b joined the watchdog) | FIXED: both threads are bound to their half's event, run state and half id; both are named and joined in `e21_end_half` |
| HIGH: KV capacity fail-open (missing line → full seqs; `max(8, …)` floor) | yes, rev 2 (block-based, fail-closed, refuse < 8) | FIXED (rev 2) |
| HIGH: `--language-model-only` missing on 0.32 | yes, rev 2 | FIXED, + the effective check `lmo_in_log` |
| HIGH: GPU release failure → half B still launches | yes, rev 2 (`e21_gpus_free` before every launch, fatal if still held) | FIXED (rev 2) |
| HIGH / CRITICAL-2: the NB2 rerun ran 9,187 s and still passed | — | ROOT CAUSE: a host suspend. `e14_heartbeat.log` (60 s) has no line from 18:31:20 to 21:04:09, while all Duck requests ended by 18:31:41; 2 requests were clamped to the deadline (read timeout 2,077 s). FIXED as evidence: CELL 6 records `solve_overrun_sec` / `planned_solve_end_sec` / `stragglers` / `deadline_violation`; the harness enforces the deadline contract and reports HOST_SUSPENDED on a heartbeat gap |
| HIGH-4: Duck's KV gate admits when /metrics is missing or stale | no | FIXED in our probe (Duck package unchanged): a failed /metrics read reports `kv_usage` 1.0, so admissions wait while work runs; the idle override prevents a deadlock; `probe_failures` counted |
| Handoff gate "stragglers must be 0, else block half B" | — | NOT ADOPTED as a hard block. With per-half binding a straggler cannot touch half B, and it talks to half A's (killed) port. Blocking half B would lose the 0.32 arm for a harmless thread. Recorded per half as `stragglers` |
| Tool + thinking smoke with Duck's tools | yes, rev 2 contract probe (+ rev 2b cache probe) | FIXED |
| CONFOUNDED_FALLBACK status / comparison validity | partly | FIXED: `config_status` + `comparison_valid` (equal windows within 1 %, the same config in both halves, no deadline violation) |

## Second deep pass over GPT audits 1-3 + the 10 agent reports (user 2026-10-28) → rev 2d

Notebooks: NB1 `dddb5d0c`, NB2 `3ce943a2`, NB1_SMOKE `f063fea7`, NB2_SMOKE `18285c9c`.

| Item still open after rev 2c | Source | rev 2d |
|---|---|---|
| The contract probe used a stub tool; GPT asked for Duck's actual tools | GPT 3 | FIXED: `E21_PROBE_TOOLS` = Duck's `_tools()` schema with `E12A.ARC2_TOOL_DESCRIPTION` |
| A /metrics outage blocked admissions forever, so Duck would degrade to one stream (via the idle override) | GPT 1 HIGH-4 ("bounded error / controlled stop") | FIXED: block for ≤ 120 s, then Duck's own fail-open (counted `probe_fail_open`) |
| Handoff: half A's workers are not waited for | GPT 2 CRITICAL-1 ("explicit cancellation / join") | FIXED: a bounded wait (≤ 120 s, after half A's server is gone, so they fail fast) for `solve_all_r6` threads; `handoff_scheduler_threads_left` recorded. Not a hard block (see 2c) |
| Regression test for a failed GPU release | GPT 1 HIGH-5 | FIXED: `E21_FAKE_GPU_HELD` hook for the harness |
| Hybrid MTP + prefix-cache silent corruption (#53912): the cache probe compared only 24 tokens | GPT 2, 3 | FIXED: 96 greedy tokens, cold vs cached must be equal. The real check is the GPU smoke |
| INT4 compressed-tensors + BF16 MTP loader risk (#58807) | GPT 2 | NOTE: our checkpoint excludes MTP (`re:^mtp.*`); the NB1 fallback exists; the GPU smoke loads it for real |
| Capacity sizing: GPT 2 proposed p90 65k per stream; we size at the R3K2 p50 peak 57k with block math | GPT 2 HIGH-3, A1 H1 step 4 | DECISION (ours): 57k. At 65k, 0.19 FP8 fits 15, not 16 (846 // (41 + 15)), and NB1 INT4 fits 18, not 20, which changes R3K2's 16. The 0.95 KV gate + vLLM preemption cover the p90 tail; `util_p90_peak_65k` is recorded per half. Can be switched to 65k on request |
| GPU smoke plan | user | the SMOKE notebooks: the same code, profile `E21SMOKE` (12 eval tasks, 20 min per half, 2 h), run on the user's account (foysal) after the CPU kit |

## rev 2e: Kaggle CPU kit (rev 2d) failures, all report-only

| Item | Source | rev 2e |
|---|---|---|
| `prefix_cache_hit_pct_last` read, `..._last_rolling` written: always None | kit, VAL2_cells R2, VAL2_config_smoke F8, VAL2_engine F13 | FIXED |
| `config_status` compared tags, not configs: under option ka half B "primary" == half A "fallback" was CONFOUNDED | kit, VAL2_cells R1, VAL2_engine F2 (report part) | FIXED: compares the cfg dicts (without `vllm`) |
| A mid-half config switch not detected | kit | FIXED: `SWITCHED_MIDHALF` |
| `comparison_valid` ignored unequal streams | kit, VAL2_engine F9 | FIXED: equal streams required |

## rev 2f: the VAL2 second pass (cells, config + smoke), only run-breaking or result-falsifying items

Notebooks: NB1 `c6dd915c`, NB2 `59cb8cfd`, NB1_SMOKE `eb6b1df9`, NB2_SMOKE `ab11df38`.

| Item | Source | rev 2f |
|---|---|---|
| The watchdog "binding" was a module-level line: global after `wrap()`, so half B rebinds it | VAL2_cells F1 (reproduced) | FIXED: default arguments; half re-checked right before a relaunch |
| `attempt1_cut_by_deadline` compared with the last end (always ≥ 1) | VAL2_cells F2 | FIXED: `solve_end_epoch` |
| `probe_fail_since` carries over to half B | VAL2_cells F3 (2), VAL2_engine F8 | FIXED: reset per half. F3 (1)/(3)/(4) LOGGED (report/telemetry only) |
| Smoke budget 2 h: half A's fallback has too little time | VAL2_config_smoke F2, VAL2_engine F12 | FIXED: 2.5 h |
| Cache probe skipped when turn 1 has no tool call | VAL2_config_smoke F3, VAL2_engine F3 | FIXED |
| Probes are hard gates in the hidden rerun (a false failure on both configs = no engine = identity submission) | VAL2_config_smoke F5, VAL2_engine F4 | FIXED: `e21_soft`: history + cache checks are warnings in RERUN; HTTP error, markup in content, degenerate text and broken arguments stay fatal |
| Cache-probe prefix was one repeated digit | VAL2_config_smoke F6, VAL2_engine F5 | FIXED: one RNG, 6,000 digits |
| foysal payload may lack the R3K2 Duck package | VAL2_config_smoke F1 | FIXED: new dataset version with `q27x_e15b_duck_9e4b89285136.bin` (verified present) |

## rev 2g: VAL2_engine (2026-10-29)

Notebooks: NB1 `8a957b7c`, NB2 `14e034c8`, NB1_SMOKE `a0b10da7`, NB2_SMOKE `daec777f`.

| Item | Source | rev 2g |
|---|---|---|
| The second relaunch switches NB1 INT4 → FP8: the served name changes, while Duck's `model_id` keeps the INT4 name → 404 on every request | VAL2_engine F1 (HIGH, rare path) | FIXED: the switch only goes to a config with the same weights (NB1 relaunches the same config; NB2 may still switch SSM16 → FP8 16, same served name, same 16 streams) |
| The smoke rejects a correct short answer ("4", "4.") | VAL2_engine F10 | FIXED: only text of ≥ 8 characters with ≤ 2 distinct characters is degenerate |
| Under option ka half B retries the identical config (≤ 20 min) | VAL2_engine F2 (time part) | LOGGED: only after a launch failure |
| The relaunch ignores the GPU-release result | VAL2_engine F6 | LOGGED: a launch on held GPUs fails and is recorded |
| A half-A relaunch thread may still launch during half B (the watchdog join is 30 s) | VAL2_engine F7 | LOGGED: narrow window (solve end within ~4 min of a relaunch decision taken with ≥ 20 min left); rev 2f re-checks the half before the relaunch starts |
| ENGINE_INFO / counters keep the first server's values after a relaunch | VAL2_engine F11 | LOGGED (report only) |
| VAL2_cells F4-F7, VAL2_config_smoke F4 / F7 / F9-F11 | VAL2 | LOGGED (report, hardening, documentation) |

## VAL2_dryrun (adversarial fake server, rev 2d, 2026-10-29): no BLOCKER

| Item | rev 2g status |
|---|---|
| F1 HIGH: a hung engine (/health 200, no tokens, silent log) is never detected | LOGGED: the same watchdog as the proven R3K2 / S1 donors; never observed on vLLM 0.19 (S1, E18, E19 runs). The GPU smoke and the per-half `server_gen_tokens` show it after the fact. Candidate for after the run: a no-progress rule (`generation_tokens_total` flat ≥ 10 min while requests run) |
| F2 HIGH (rev 2d): a config-switching relaunch: served-name 404 + stale records | the 404 part FIXED in rev 2g (= VAL2_engine F1); the stale `used` / `backend` / `capacity` fields LOGGED (report only; `SWITCHED_MIDHALF` invalidates the comparison) |
| F3: `comparison_valid` ignored unequal streams | FIXED in rev 2e; the "s20" label for a 19-stream half LOGGED |
| F4: no-call reply skipped the cache probe | FIXED in rev 2f; the history check on that path LOGGED |
| F5: a dropped history `reasoning` passes when the reasoning is short | LOGGED: Duck's reasoning key was verified on the vLLM source of both versions and by the E12-V real run; a deterministic check is a candidate for after the run |
| F6: `attempt1_cut_by_deadline` | FIXED in rev 2f |
| F7: num_blocks ±1 from the 2-decimal factor | LOGGED (no stream count changes at today's margins) |
| F8: `prefix_cache_hit_pct_last` None | FIXED in rev 2e |
| F9: non-string tool arguments | LOGGED (vLLM always sends a string) |
| F10: a skipped relaunch counts for the switch rule | LOGGED (no effect) |
| F11: a real prefix-cache miss refuses both configs in commit / smoke | LOGGED: by design in commit / smoke (it shows in the smoke); soft in the hidden rerun (rev 2f) |

## rev 2h: GPT audits of the v3 zip (2026-10-29, two audits)

Notebooks: NB1 `4cade9fa`, NB2 `d84efecc`, NB1_SMOKE `0fb2d386`, NB2_SMOKE `7cd3ab70`.

| GPT finding | Verified | rev 2h |
|---|---|---|
| Audit 1 HIGH-1: a hung engine (/health 200, silent log, no tokens) is never relaunched (= VAL2_dryrun F1) | yes (dry-run stall run) | FIXED: `_e21_progress()` reads running + waiting and prompt + generated token counters from /metrics. Requests on the server with no progress for `E15B_WD_STALL_SEC` (600 s) = dead, which takes the existing relaunch path. Recorded as `stalls` / `no_progress_sec_max` per half. Tool execution never counts (Duck's request has finished then). Tested: `_scratch/val2_dryrun` stall run with a 40-s limit: detected at 41 s, 6 requests running, twice; relaunch correctly skipped (window too short); `comparison_valid` False |
| Audit 1 HIGH-2: `comparison_valid` ignores solver / step errors and incomplete scoring | yes | FIXED: per half `solver_error`, `scoring_complete` (`eval_query_total` == all test queries of `TASK_IDS`), `watchdog_stalls`; `comparison_valid` requires no errors, no solver error, complete scoring and no stall in both halves |
| Audit 1 HIGH-3: the late callback calls the global `e12_write_submission` (half B's writer) | checked: NOT A DEFECT | in commit `e12_write_submission` writes the identity submission only (records are read only `if RERUN`), and the hidden rerun has half A only; nothing of half A can reach a submission. Not changed |
| Audit 2 HIGH: a half-A relaunch in flight when half A ends can kill / launch during half B (= VAL2_engine F7) | yes (code-level) | FIXED: `e21_end_half` marks the half as ending first; `e21_relaunch` re-checks it right before launching; a watchdog still alive after the 30-s join gets its in-flight server and GPU processes killed (half B has not begun), then a second join (≤ 180 s); still alive -> `abort_next_half` (fail closed, GPT's guard): half B is not started and gets a `handoff` error. Unit test `_scratch/e21/test_rev2h_race.py`: 5/5 (no launch while ending; in-flight relaunch cancelled + joined + half B allowed; stuck watchdog -> abort; the last half never aborts; the runner skips half B) |
