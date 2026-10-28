# VAL2 dry run: the E21 notebooks driven by execution against an adversarial fake vLLM server (rev 2d, 2026-10-29)

## Scope and method

**What was run:** the notebooks' own cells, in order, in one Python 3.12.10 process (`_scratch/e14_venv312`).
- NB1 `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb`, sha256 `dddb5d0c…`.
- NB2 `Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16.ipynb`, sha256 `3ce943a2…`.
- Both were copied to `_scratch/val2_dryrun/` at 23:33. The only edit, made in memory by the driver, is `E21_PORTS` (asserted string replace).

**Note: the notebooks were rebuilt while I ran.** At 00:07 the four notebooks were rebuilt from a working builder (rev 2e-2g). Every finding below is against rev 2d. Where the working builder source (`build_e21.py` 23:57, `e21_engine_new.py` 00:07) already changes the code path, the finding says so. I checked that by reading only; no rebuilt notebook was executed.

**Own tools** (none of the project's harness / fake was run). All are in `D:/ARC AGI 2/_scratch/val2_dryrun/`:
- `val2_fake.py`: my fake vLLM OpenAI server. It accepts both argv forms (0.19 `--model <p>`, 0.32 `<p> ...`).
  - **Startup lines in the real layouts.**
    - 0.19, two lines: `GPU KV cache size: N tokens` and `Maximum concurrency for M tokens per request: X.XXx`, with N = num_blocks // 4 × B.
    - 0.32, one line: `GPU KV cache size: N tokens, Maximum concurrency for ...: X.XXx`, with N = X × M, and the four `(Worker_TPk)` block-size lines.
    - The 0.32 stats line includes `Deferred:` / `KV fetch:` / `Preemptions:` exactly in the order of 0.32 `vllm/v1/metrics/loggers.py:283-322`.
  - **Knobs:**
    - `V2_CONTRACT`: notool / nested / dictargs / dropreason;
    - `V2_DETAILS`: absent / null / zero / nullval;
    - `V2_RKEY`: reasoning / reasoning_content / both;
    - `V2_STALL_AFTER`: /health stays 200, every chat request hangs, the log goes silent;
    - `V2_METRICS=rich`: label rows, `_created`, deprecated duplicates, by-source rows, histograms;
    - `V2_STATS=preempt`;
    - `V2_DIE_AFTER` / `V2_DIE_TIMES`;
    - `V2_CODE`;
    - `V2_CUT`: a cut call, finish "tool_calls" with completion == max_tokens;
    - `V2_BLOCKS`.
- `drive.py`: the driver. Its setup is modelled on the harness (Windows shims, fixtures `_scratch/g4b_bundle_verify/tests/fixtures`, the E15 payload in the dataset shape, the INT4 dir in the Kaggle path shape). It sets `E11_HARNESS=1` and `E21_FAKE_SERVER=val2_fake.py`, plus the profile EVAL6 (6 tasks), `E21_HALF_SOLVE_SEC` (90 s unless stated) and `Q27X_TIME_BUDGET=10800`.
- `val2_hooks.py`: per-scenario hooks.
  - **units:** cells 0-6, then direct calls of `e21_capacity`, `e21_stat_rows` and `e21_begin_half` / `e21_end_half` against `val2_fake.py` variants.
  - **The other scenarios:** a snapshot after cell 9, and a thread sampler.

**Ports:** 18234-19711, distinct per run. 1234 and 8011 were never bound or connected to. At the end `netstat` shows no listener left in 18xxx/19xxx.

**Runs** (all `cell_errors: []`, i.e. no uncaught exception in any cell). Logs are in `_scratch/val2_dryrun/*.log`, artefacts in `_scratch/val2_dryrun/runs/<run>/` (`val2_result.json`, `q27x/`, `fake_state/`).

| run | command (in `_scratch/val2_dryrun`, `PY=_scratch/e14_venv312/Scripts/python.exe`) |
|---|---|
| units NB1 | `$PY drive.py nb1 units 6` → `runs/nb1_units_235338` |
| rich NB1 | `V2_PORT_A=18634 V2_PORT_B=18711 $PY drive.py nb1 rich` → `runs/nb1_rich_000251` (`V2_METRICS=rich V2_STATS=preempt V2_DETAILS=0.19:null,0.32:zero V2_RKEY=0.19:both,0.32:reasoning V2_CONTRACT=nested V2_CUT=1`) |
| switch NB1 | `V2_PORT_A=18834 V2_PORT_B=18911 V2_WIN=1500 $PY drive.py nb1 switch` → `runs/nb1_switch_000251` (`V2_DIE_AFTER=0.19:6 V2_DIE_TIMES=2`, watchdog poll 2 s / fail 20 s / min-left 30 s) |
| stall NB1 | `V2_PORT_A=19034 V2_PORT_B=19111 V2_WIN=400 NBX_E15B_WD_POLL_SEC=2 NBX_E15B_WD_FAIL_SEC=20 NBX_E15B_WD_MIN_LEFT_SEC=30 $PY drive.py nb1 stall` → `runs/nb1_stall_000521` (`V2_STALL_AFTER=0.19:4`) |
| loop NB1 | `V2_PORT_A=19234 V2_PORT_B=19311 $PY drive.py nb1 loop` → `runs/nb1_loop_000519` (`V2_CODE="0.19:while True: pass" V2_TOOLSTEPS=1`) |
| rcontent NB2 | `V2_PORT_A=19434 V2_PORT_B=19511 $PY drive.py nb2 rcontent` → `runs/nb2_rcontent_000521` (`V2_RKEY=reasoning_content V2_DETAILS=nullval`) |
| streams NB1 | `V2_PORT_A=19634 V2_PORT_B=19711 $PY drive.py nb1 streams` → `runs/nb1_streams_001502` (`V2_BLOCKS=0.32:1000`) |

Two early runs (`nb1_rich_235955`, `nb1_switch_235956`) were invalid setups of mine and are not used as evidence:
- the 0.19 details were "absent" even with cache hits, which vLLM never does;
- the driver had cleared `V2_WIN`.

They are kept only for completeness. The first one does show F11 (a cache-probe miss kills both configs).

## Summary

| # | Severity | Where (rev 2d) | Finding | Working builder (read only) |
|---|---|---|---|---|
| F1 | **HIGH** | CELL 6 `_e15b_watchdog` (built from `build_e21.py:352-380`) | A hung engine (/health 200, no tokens, silent log) is never detected. The rest of the half is lost and the report says `comparison_valid: true`. | open (no progress check) |
| F2 | **HIGH** (rev 2d) | CELL 5 `e21_relaunch` (`e21_engine_new.py:373-405`) + CELL 9 | A config-switching relaunch leaves every engine record but `cfg` stale. The report reads PRIMARY_BOTH / int4-s20 / 20 streams / `comparison_valid: true` while the server ran FP8-s16. The served name changes, too (404 on real vLLM, = VAL2_engine F1). | switch limited to the same weights; status SWITCHED_MIDHALF. The stale `used` / `backend` / `capacity` / `kv_tokens` / `CONCURRENCY` are still not updated |
| F3 | MEDIUM (rev 2d) | CELL 9 `comparison_valid` | It ignores unequal stream counts. 20 vs 19 streams gave `comparison_valid: true`. The backend label says "s20" for a 19-stream half. The real logs show 0-1 stream margins. | streams equality added (rev 2e); the label is still the cfg's seqs |
| F4 | MEDIUM | CELL 5 `e21_contract_probe` (`e21_engine_new.py:218`) | "Thinks long, no tool call" → `ok: true` with neither the cache probe nor the history check (= VAL2_engine #3, VAL2_config_smoke F3) | the cache probe now runs; the history check is still skipped |
| F5 | MEDIUM | CELL 5 `e21_contract_probe` (`e21_engine_new.py:231`) | A server that drops the history `reasoning` passes the probe whenever the reasoning is short (< ~128 tokens). The probe itself asks for "very short" thinking. | open |
| F6 | MEDIUM (rev 2d) | CELL 9 `_e21_stage1` | `attempt1_cut_by_deadline` is measured against the last attempt-1 end. It read 6/6 "cut" in a half that ended 79 s early (= VAL2_cells F2). | fixed (`solve_end_epoch`) |
| F7 | LOW | CELL 5 `e21_capacity` (`e21_engine_new.py:168`) | num_blocks is rebuilt from the 2-decimal X, and was +1 over the truth on NB2's 0.32 line (1,629 vs 1,628) | open |
| F8 | LOW (rev 2d) | CELL 9 | `prefix_cache_hit_pct_last` is always None (key `..._rolling`). Also visible in the harness / kit FAIL lines. | fixed |
| F9 | LOW / NOTE | CELL 5 `e21_contract_probe` | Non-string tool-call `arguments` raise a TypeError, which refuses BOTH configs (the half has no engine) | open (vLLM always sends a string) |
| F10 | NOTE | CELL 5 `e21_relaunch` | A `{"skipped": ...}` entry counts as "already crashed once" for the config switch. Harmless today (time only shrinks). | open |
| F11 | NOTE (design) | CELL 5 `e21_cache_probe` | A real prefix-cache miss (`cached_tokens` < 1,024 on the repeat) refuses both configs, and the half has no engine (= VAL2_engine #4 / config_smoke F5 for the rerun) | rerun: soft; commit: still fatal |

No BLOCKER was found in the paths I could drive.

## Findings

### F1 — HIGH: a hung engine is never detected; the half is silently lost and the comparison is reported valid

**Scenario.** The engine core hangs: for example an NCCL / CUDA deadlock, the "engine stall that is not an execute_model RPC timeout" of VAL_api_D2 F4.
- In vLLM V1, /health only fails once the engine-core process is dead (errored), so it keeps answering 200.
- The 10-s stats logger switches to `logger.debug` when throughput is 0 (`loggers.py:276-280`, "idle"), so the log goes silent.
- With `--disable-uvicorn-access-log` (rev 2 fix for D2 F4) nothing else writes to it.

**The watchdog sees this as "alive."**
- A growing log means alive.
- When the log is silent it probes /health, gets 200, and resets `_bad`.
- It never compares progress, although it reads `vllm:generation_tokens_total` into `_wd["tok_last"]` every poll.

**Evidence** (stall run, NB1; the fake stalls every chat request after 4 Duck tool requests, /health stays 200, no stats line):

```
$ V2_PORT_A=19034 V2_PORT_B=19111 V2_WIN=400 NBX_E15B_WD_POLL_SEC=2 NBX_E15B_WD_FAIL_SEC=20 NBX_E15B_WD_MIN_LEFT_SEC=30 $PY drive.py nb1 stall
E12 solving: 6 tasks x 2 branches | streams 20 | model Qwen/Qwen3.8-27B-INT4 | stops in 400 s
E12 [5 min] branches done 0/12, accepted 0 | {"num_running_reqs": 0.0, ...}
E12 rev 6 coverage: ... response 4 / 6 | ... stragglers 0
E21 half A ended at 425 s | ...
(no "E15B watchdog" line at all in the whole log)
```

From `runs/nb1_stall_000521/val2_result.json`:

```
watchdog {"relaunches": [], "health_fail_sec_max": 0, "tok_last": 1516.0, "health_probes": 194, "tok_base": 392.0,
          "limits": {"poll_sec": 2.0, "fail_sec": 20.0, "min_left_sec": 30.0}}
half A: deadline_in_sec 400, solve 19 -> 419 s, attempts_started 6 (attempt 2: 0), stage1.attempt1_records 6, relaunches null, errors null
fake (port 19034): tool_requests 10, stalled 6, stats_lines 1, /health answered
config_status PRIMARY_BOTH  comparison_valid True  windows_sec [400, 400]
report keys mentioning stall/health/watchdog: []
```

- For about 390 s no token was generated: `tok_last` was frozen at 1,516 while 6 requests were in flight.
- The watchdog probed /health 194 times, all OK, and never acted.
- Half A finished with 4 of 6 attempt-1 replies and no attempt 2, yet `comparison_valid` is True.
- On Kaggle this means the remainder of a 5.45-h half is lost (Duck's requests are deadline-clamped), and nothing in the report says so.
- The round-1 report VAL_api_D2 F4 proposed exactly the progress check. The tracker marks D2 F4 FIXED by the access-log flag. That flag does not cover this case.

**Fix.**
1. In `_e15b_watchdog`, treat "requests running (or Duck waiting) and `vllm:generation_tokens_total` unchanged for ≥ 10 min" as dead. Then relaunch (`e21_relaunch`) and record `stall_detected_sec`. Prefill of a 100k prompt at 4,096 batched tokens is well under 10 min.
2. Per half, record `no_progress_sec_max`. Make `comparison_valid` False when a half lost more than ~2 % of its window to no-progress.
3. For a quick regression test, the fake knob `V2_STALL_AFTER` reproduces it.

### F2 — HIGH (rev 2d): a config-switching relaunch leaves the engine record, Duck's streams and the served name stale

**Scenario.** The half-A server dies twice. The second relaunch switches NB1 from INT4-s20 to FP8-s16 (rev 2b "second relaunch uses the half's other config"). `e21_relaunch` only sets `ENGINE_INFO["cfg"]` and `relaunch_cfg_switched`. `used`, `backend`, `capacity`, `kv_tokens`, `concurrency` / `CONCURRENCY`, Duck's gate slots / workers and `E12_RUN["model_id"]` all stay at the first config.

**Evidence:**

```
$ V2_PORT_A=18834 V2_PORT_B=18911 V2_WIN=1500 $PY drive.py nb1 switch
E15B watchdog: server down -> relaunch {"t": 28, "dead": true, "left_sec": 1492}
E21 half A relaunch0: ready in 3s | capacity {... "num_blocks": 1069 ... "streams": 20}
E15B watchdog: server down -> relaunch {"t": 43, "dead": true, "left_sec": 1477}
E21 half A launch relaunch1: vllm 0.19.0 ... --model ...m1-fixtures-tok_fp8-... --served-model-name Qwen/Qwen3.8-27B-FP8 ... --max-num-seqs 16
E21 half A relaunch1: ready in 2s | capacity {"num_blocks": 846, ... "streams": 16}
```

From `runs/nb1_switch_000251/val2_result.json`:

```
ENGINE A {"used": "primary", "backend": "vllm-0.19.0-int4-mtp3-s20-FLASHINFER",
          "cfg": {"weights": "fp8", "seqs": 16}, "kv_tokens": 1710400, "concurrency": 20,
          "relaunch_cfg_switched": {"weights": "fp8", "seqs": 16}}
e21_report half A: engine_used vllm-0.19.0-int4-mtp3-s20, config_used primary, streams 20, workers 20,
                   capacity.num_blocks 1069 / streams 20 (the INT4 server's)
A gate {"slots_env": "20", "free_now": 20}; A solve 20 -> 65 s, i.e. ~50 % of the solve ran on the FP8 server
launches: INT4 ready, died, INT4 ready, died, FP8(seqs 16, served Qwen/Qwen3.8-27B-FP8) ready, 0.32 INT4 ready
Duck model_id A: Qwen/Qwen3.8-27B-INT4
config_status PRIMARY_BOTH   comparison_valid True
B spec: INT4 s20, used primary, spec_note None   (E21_B_FOLLOWS_A not applied: A's "used" is still "primary")
```

**Consequences on Kaggle:**
- **(a) The model name.** Duck keeps sending `model: Qwen/Qwen3.8-27B-INT4` to a server that serves `Qwen/Qwen3.8-27B-FP8`. vLLM answers 404 to every request, so the rest of the half is lost. My fake does not validate the model name, so this run looked healthy. That is a gap in my fake. VAL2_engine F1 found it from the source; my launch records reproduce the mismatch.
- **(b) The streams.** 20 Duck workers / gate slots run against a `--max-num-seqs 16` server.
- **(c) The report.** Every report field of the half describes the wrong config, and `comparison_valid` is True.
- **(d) Half B.** B ran INT4 while A ended on FP8.

**Fix.**
- The working builder restricts the switch to the same weights (rev 2g) and reports SWITCHED_MIDHALF. That removes (a) and invalidates the comparison.
- Still missing: on any successful relaunch, write `ENGINE_INFO.update(used="relaunch:"+tag, backend=..., capacity=info["capacity"], kv_tokens=...)`. That matters for NB2, whose SSM16 → no-SSM switch changes B from 832 to 1,600.
- If `info["capacity"]["streams"] < E12_RUN["streams"]`, record `streams_after_relaunch`. The Duck gate is a per-half singleton, so at least record the oversubscription.

### F3 — MEDIUM (rev 2d): `comparison_valid` ignores unequal stream counts; the real logs show 0-1 stream margins

**Evidence (streams run).** The fake's 0.32 server logs 1,000 blocks. That is close to the real VMS 0.32 INT4-S20 log, which gives 989.

```
$ V2_PORT_A=19634 V2_PORT_B=19711 $PY drive.py nb1 streams
A vllm-0.19.0-int4-mtp3-s20-FLASHINFER streams 20 workers 20 cap 1069 20 True window 90
B vllm-0.32.0-int4-mtp3-s20-FLASHINFER streams 19 workers 19 cap 1000 19 False window 90
config_status PRIMARY_BOTH comparison_valid True windows_equal True
```

The backend label of half B says "s20" (it is built from `cfg["seqs"]`) although 19 streams ran.

**Real-log capacities** (`e21_capacity` on the original files, units run):

| log | X at M | num_blocks | streams (seqs) | fits_p50 |
|---|---|---|---|---|
| E12-V run 2, 0.19 FP8 (two-line format) | 8.46 at 135,168 | 846 | 16 (16) | True, with 30 blocks of headroom (16 × 51 = 816) |
| VMS 0.32 INT4 S20 (one-line) | 12.06 at 106,496 | 989 | **19** (20) | False |
| VMS 0.32 INT4 S16 | 12.18 | 999 | 16 (16) | True |
| VMS 0.32 FP8 ref S16 | 9.57 | 785 | **15** (16) | False |
| VMS 0.32 FP8 SSM fp16 (B 832) | 10.55 | 1,509 | 16 (16) | True |
| VMS 0.32 tp2x2 (two KV lines) | 1.38 | 113 | refused (< 8) | — |

- The VMS logs ran without `--language-model-only`, which gives about +10 % KV (E18: 1,110,415 vs 1,006,517). So the E21 halves should land at 20 / 16, but only by 1-2 streams of margin.
- A single-stream difference between versions is therefore realistic.

**Fix.**
- rev 2e adds streams equality to `comparison_valid`. Good. But with these margins a harmless 19 vs 20 will often void the A/B.
- Consider instead pinning both halves to one stream count: run half B at min(B's fit, A's streams), and record `streams_capped_to_A`. Or report `streams_equal` separately from `comparison_valid`.
- Build the backend label from `CONCURRENCY`, not from `cfg["seqs"]`.

### F4 — MEDIUM: "thinks long, makes no tool call" passes the launch gate with neither the cache probe nor the history check

**Evidence** (units, `V2_CONTRACT=notool`: 5,039 chars of reasoning, finish "length" at 3,072, no call):

```
E21 half A primary: ready in 2s | ... | contract {"ok": true, "finish": "length", "reasoning_chars": 5039, "content": "",
  "tool_calls": 0, "prompt": 394, "completion": 3072, "turn2": "inconclusive: no tool call in turn 1"}
[val2 units] contract_notool: {"used": "primary", "fatal": null, ...}     (no "cache" key: e21_cache_probe never ran)
```

In rev 2d, `if not calls: out.update(ok=True, ...); return out` returns before `out["cache"] = e21_cache_probe(served)`. The only pre-run check of #53912 (MTP + prefix-cache corruption) is then skipped, and so is the reasoning-history check. The same finding is VAL2_engine #3 and VAL2_config_smoke F3.

**Working builder:** the cache probe now runs on the no-call path. The history check is still skipped.

**Fix:** on a no-call reply, retry turn 1 once (for example max_tokens 6,144, "Think in at most three sentences"). Then run the history check of F5's fix, which needs no tool call at all.

### F5 — MEDIUM: the contract probe cannot detect a dropped history `reasoning` when the reasoning is short

**Scenario:** the server ignores the assistant `reasoning` key when it renders the history. That is exactly README Finding 1, the bug class the probe exists for: a template or parser change in 0.32, or a key mismatch.
- With reasoning kept: shortfall = (prompt1 + completion1) − prompt2 ≈ −(tool turn + wrappers), about −15 on real vLLM.
- With reasoning dropped: shortfall ≈ reasoning tokens − 15.
- The rule `short > max(128, len(reas) // 6)` therefore only fires when the reasoning is longer than about 143 tokens. The probe prompt asks: "Keep your thinking very short."

**Evidence** (units, `V2_CONTRACT=dropreason`: the fake renders the history without `reasoning`, 105 chars of reasoning):

```
[val2 units] contract_dropreason: {"used": "primary", "fatal": null, ... "contract": {"ok": true, "finish": "tool_calls",
  "reasoning_chars": 105, "tool_calls": 1, "prompt": 394, "completion": 92, "turn2_prompt": 490, "turn2_shortfall": -4, "cache": [...]}}
compare normal: "completion": 176, "turn2_prompt": 623, "turn2_shortfall": -53
```

The dropped reasoning still passed, and Duck would then run 5.45 h with its thinking silently removed from every history. It is caught only if Qwen3.8 happens to think more than ~140 tokens at reasoning_effort "low".

**Fix (deterministic, no generation):** send the turn-2 history twice with `max_tokens: 1`, once with `reasoning` and once without. Require `prompt_tokens(with) − prompt_tokens(without) ≥ 0.8 × tokens(reasoning)`, counting the tokens with `E12_TOKENIZER_JSON`, and the reasoning ≥ 50 tokens (pad the probe's reasoning by asking for "a four-step plan" if needed). This works with or without a tool call (F4) and has no length threshold.

### F6 — MEDIUM (rev 2d): `stage1.attempt1_cut_by_deadline` counts attempts that ended near the last attempt, not near the deadline

**Evidence:**
- Rich run, both halves: solve 60 → 71 s against `planned_solve_end_sec` 149.8, i.e. finished 79 s early.
  ```
  stage1 {"attempt1_records": 6, "tasks": 6, "attempt1_missing": 0, "attempt1_cut_by_deadline": 6, "attempt1_all_done_h": 0.0}
  ```
- Stall run: `attempt1_cut_by_deadline: 4`.

`dl = max(ends)`, so `dl - e < 90` always counts at least the last attempt. `run` is passed in but unused. In a real half where stage 1 ends 3 h early, it still reports cuts. That misleads the "attempt-1-only score is comparable only when stage 1 finished" rule. Same as VAL2_cells F2.

**Fix:** done in the working builder (`E12_RUN["solve_end_epoch"]`, `dl - x < 90` against it). Also count `None` as "no deadline" (the rerun).

### F7 — LOW: num_blocks is rebuilt from the 2-decimal concurrency factor (±1 block)

`nb = round(X × (cdiv(M, B) + F))`. X is printed with 2 decimals, so the error is up to ±0.005 × (cdiv + F) blocks: ±0.5 at B 1,600 (divisor 100) and ±0.9 at B 832 (divisor 178).

**Evidence:** NB2 rcontent run, half B:

```
"max_concurrency_logged": {"tokens": 135168, "x": 9.15}, "block_size": 832, "num_blocks": 1629
```

The fake's true count is 1,628.

This is harmless at today's margins (1,629 // 84 = 1,628 // 84 = 19). At an exact boundary it can grant one stream more than fits.

**Fix:**
- On 0.19, intersect with the KV line: N = nb // 4 × B gives nb ∈ [4N/B, 4N/B + 3].
- On 0.32, use `floor(X_hi × divisor)` with X_hi = X − 0.005 as a conservative bound, or `floor((X − 0.005) × divisor)`.

### F8 — LOW (rev 2d): `prefix_cache_hit_pct_last` is always None in `e21_report.json`

**Evidence (rich run):** both halves `prefix_cache_hit_pct_last null`. The server-log summary only has `prefix_cache_hit_pct_last_rolling` (e.g. 62.99 % whole run in counters).

The existing harness already flags it: `_scratch/e21_kit_h3/*.log`, `[e21 FAIL] half A: e21_report prefix_cache_hit_pct_last is filled`. Fixed in the working builder (`build_e21.py:545`).

### F9 — LOW / NOTE: non-string tool-call `arguments` refuse both configs

**Evidence** (units, `V2_CONTRACT=dictargs`):

```
E21 half A primary FAILED: 'TypeError: the JSON object must be str, bytes or bytearray, not dict'
E21 half A fallback FAILED: 'TypeError: the JSON object must be str, bytes or bytearray, not dict'
[val2 units] contract_dictargs: {"used": null, "fatal": "no vLLM config became ready", ...}
```

vLLM always serializes `arguments` as a JSON string (OpenAI schema), so this is not expected on 0.19 / 0.32.

**Fix (hardening):** `a = c["function"]["arguments"]; a = json.loads(a) if isinstance(a, str) else a`.

### F10 — NOTE: a "skipped" relaunch counts as a crash for the switch rule

`len(ENGINE_INFO.get("relaunches") or []) >= 1` also counts `{"skipped": "only N s for a relaunch"}` entries. In the invalid early switch run (90-s window), both deaths produced skipped entries:

```
E15B watchdog: {"t": 39, ..., "ok": false, "why": "too little time for a relaunch"}
```

Once a relaunch has been skipped, every later one is skipped too (the time left only shrinks), so this has no effect today. Count only entries with `ready` / `error`.

### F11 — NOTE (design): a prefix-cache miss in the cache probe refuses both configs

**Evidence** (units `details_absent`, and the invalid early rich run). When the repeat request reports no `cached_tokens`:

```
E21 half A primary FAILED: 'RuntimeError: cache probe failed: [{"prompt": 18060, "cached_repeat": 0, "own_marker": true, "equal": true}, ...]'
E21 half A fallback FAILED: (same)
-> "fatal": "no vLLM config became ready"; half A: no engine; config_status NOT_RUN
```

On real vLLM this happens only on a real cache miss: 0.19 sets details only when cached > 0; 0.32 `_make_prompt_tokens_details` gives `{"cached_tokens": 0, ...}`. A broken prefix cache makes Duck slower, not wrong. The `equal` (cold == cached) part is the correctness check.

**Suggestion:** fail closed only on `equal` / `own_marker`. Record `cached_repeat < 1024` as `cache_probe_miss` and run the half anyway, with `comparison_valid` False when either half had a miss. (VAL2_engine #4 and VAL2_config_smoke F5 raise the same for the rerun. The working builder makes it soft only in RERUN.)

## Requested scenarios that produced no bug (what was exercised, with evidence)

1. **Capacity lines in the real formats.**
   - 0.19 two-line format: E12-V log → B 1,600, X 8.46, 846 blocks, 16 streams. That equals the tracker's hand check (846 // 4 × 1,600 = 337,600).
   - 0.32 one-line format and Worker block-size lines: all VMS logs parse (table in F3). With multiple KV lines (tp2x2) the last one is taken and < 8 streams is refused.
   - The fake's own 0.32 launches report `kv_tokens_logged` 1,444,945 and 1,069 blocks → 20 streams, `lmo_in_log` True.
   - **F = 15 holds on 0.32 too.** The VMS MTP-2 vs MTP-3 FP8 logs give 9.92 × (67 + 12) = 783.7 vs 9.57 × (67 + 15) = 785 blocks (0.2 % apart). Without the fixed Mamba blocks they would be 665 vs 641 (3.7 % apart).
2. **Stats lines.**
   - `e21_stat_rows` parses 46 / 46 real 0.19 lines and 58 / 58 real 0.32 lines.
   - Synthetic lines all parse with the right fields: 0.32 `Deferred: 2 reqs, KV fetch: 0 waiting to start, 1 in progress, 0 completed waiting, Preemptions: 3`, `Deferred` alone, 0.19 `Preemptions: 2`, no prefix field (→ None), a `Corrupted` tail, `Engine 001`.
   - **Rich run:** `preempt_lines` 6 / 7.
3. **/metrics with label rows, `_created`, deprecated duplicates, by-source rows and histograms.**
   - `e21_counters` sums `vllm:request_success_total` over its 4 `finished_reason` rows.
   - It ignores `vllm:num_preemptions_created`, `vllm:gpu_prefix_cache_queries_total`, `vllm:prompt_tokens_by_source_total{source=...}` and `_bucket` / `_count` / `_sum`.
   - The per-half counters equal the fake's own:

     | half | preemptions | prefix queries | prefix hits |
     |---|---|---|---|
     | A | 24 | 221,004 | 139,200 |
     | B | 30 | 221,004 | 139,200 |

   - `_e12_capacity_probe` and `_e15b_gen_tokens_total` worked against the same output: `probe_failures` None, `server_gen_tokens` filled.
4. **`prompt_tokens_details` absent / null / `{"cached_tokens": 0}` / `{"cached_tokens": null, "created_cache_tokens": null, ...}`.** The smoke, contract and cache probe, Duck's usage, CELL 7 and CELL 9 never crash; `cached_pct_requests` is 59-89 %. Only a genuine miss on the cache-probe repeat fails (F11).
5. **0.32 `reasoning` vs `reasoning_content`.**
   - Responses carrying only `reasoning_content` (NB2 rcontent run, both halves) or both keys (rich, 0.19) are accepted by the probe and by Duck.
   - Duck's history always carries `reasoning`: fake counters `hist_reasoning` 2,197 / `reasoning_content_only` 0 per half on NB2, and 19 / 0 per half on NB1 rich.
6. **Tool-call `arguments` with nested escapes:** `{"code": "s = \"6 * 7 = \\\"{}\\\"\\n\".format(6 * 7)\nprint(s.strip(), '\\t', {\"k\": [1, 2]})"}` passes the contract checks (units `contract_nested_escapes`, rich run).
7. **Cut tool calls** (finish "tool_calls", completion == max_tokens): `length_restored` 18 = the fake's 18 per half, and `cut_code_saved` 18 (rich run).
8. **A relaunch that keeps the config** (switch run, relaunch0): ready, Duck continued, the tokens were accumulated across servers (`counters` from the last server, `tok_acc`).
9. **The handoff with half-A Duck threads.** I tried to get a half-A worker alive at half B's launch through server behaviour:
   - the stall run: every request hung until the deadline;
   - the loop run: `while True: pass` as the tool code.

   Neither produced one.
   - Requests and sandbox runs are deadline-clamped: loop run `solve_overrun_sec` 0.9, stragglers 0.
   - `handoff_scheduler_threads_left` 0, `handoff_wait_sec` 0.0.
   - The thread sampler showed 21 `solve_all_r6.<locals>.*` threads during half A, 0 after A's end, and 21 again only once B was solving.
   - The callback binding itself was not re-tested here (the harness's late-callback check covers it). The VAL2_cells F1 claim about the watchdog binding was not in my reach: in my scenarios A's watchdog never outlived A.
10. **NB2's SSM fp16 path on both versions** (B 832): capacity 1,629 blocks (F7) → 16 streams, `lmo_in_log` True, Duck ran in both halves, no cell error.
11. **The last half** kills fast and does not wait for the GPU release (`GPU release null` for B in every run). The non-last half waits (`released: false` is the fake's no-nvidia-smi answer, accepted under `E21_FAKE`).

## Limits of this dry run

- My fake does not check the request's `model` against `--served-model-name`. That is why F2(a) shows up only in the launch records, not as 404s. Add this check to any future fake.
- The Duck program of my fake is minimal: two python calls, then a plain answer. Scores are 0. The runs test plumbing, not solving.
- Stalls *inside* the launch probes (smoke 300 s, contract 2 × 600 s, cache probe 4 × 300 s; up to 2,400 s per config) were not executed. VAL2_config_smoke F7 covers their timing.
- Windows host: killpg is shimmed to `os.kill(pid, 9)`, which matches Linux killpg for my single-process fake.
