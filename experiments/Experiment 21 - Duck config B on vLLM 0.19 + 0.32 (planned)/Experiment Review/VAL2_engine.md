# VAL2 engine: second-round bug hunt of the E21 engine cell, rev 2d (2026-10-29)

## Scope and method

- **What was checked:** the engine cell as built (notebook cell 7). Its new part is `Build/e21_engine_new.py`: cell line = source line + 276.
- **Identity checks:**
  - The cell 7 tail (579 lines) is line-for-line identical to `e21_engine_new.py`.
  - Cell 7 is identical in NB1 `dddb5d0c`, NB2 `3ce943a2`, NB1_SMOKE `f063fea7` and NB2_SMOKE `18285c9c`. All four shas were re-hashed.
- **Background read:** README, `E21_FINDINGS_TRACKER.md`, `VAL_engine_A1.md` (capacity section) and the headings of A2 / D1 / D2. No item the tracker marks FIXED is re-reported unless the fix itself is wrong (findings 1 and 2).
- **Evidence used:**
  - **Real vLLM logs:**
    - 0.19: E12-V SMOKE1 run 2, `vllm_openai_server.log`;
    - 0.32: all 14 VMS logs, plus the E18 S1 log.
  - **vLLM source:**
    - v0.19.0 / v0.32.0 fetched from raw.githubusercontent.com: `v1/metrics/loggers.py`, `v1/core/kv_cache_utils.py`, `v1/kv_cache_interface.py`, `model_executor/models/config.py`, `qwen3_5.py`, `mamba_utils.py`, `reasoning/qwen3_reasoning_parser.py`, `entrypoints/chat_utils.py`, `mamba/abstract.py`, `config/cache.py`;
    - earlier copies in the session scratchpad `vsrc/` (0.32 `parser/qwen3.py`, `serve/engine/serving.py`).
  - **Duck package `9e4b89285136`:** `arc2_duck.py` sha b9f67010 and `vendor/inference/agent/tool_agent.py`.
  - **Harness outputs:** `_scratch/e21/harness_runs/` (latest NB1 / NB2 runs, 23:17-23:43). These runs ran before the 23:47 rebuild. The CELL 9 logic they ran is the same as the current build.
- **Isolated experiments:** in `D:/ARC AGI 2/_scratch/val2_engine/exp/`, with functions extracted from the built cell. No server was started, and ports 1234 / 8011 were never touched:
  - `names.py`: definitions before use;
  - `statre.py`: stats regex vs every real log;
  - `cap.py`: `e21_capacity` on every real log;
  - `gaps.py`: startup log silence;
  - `follow.py`: B-follows-A labels and the relaunch config switch.

## Summary

| # | Severity | Where | Finding |
|---|---|---|---|
| 1 | HIGH (rare path) | `e21_relaunch` (c7:634-663) | The second relaunch switches NB1 from INT4 to FP8. The new server serves `Qwen/Qwen3.8-27B-FP8`, but Duck keeps sending `Qwen/Qwen3.8-27B-INT4`, so every request gets a 404 for the rest of the half. The report still says PRIMARY_BOTH and comparison_valid True. |
| 2 | MEDIUM | `e21_begin_half` (c7:700-705) + CELL 9 `config_status` | E21_B_FOLLOWS_A runs A's fallback config in half B under the tag "primary". CELL 9 therefore reports CONFOUNDED_FALLBACK and comparison_valid False in exactly the case option ka exists for. The "fallback" attempt is an identical retry. |
| 3 | MEDIUM | `e21_contract_probe` (c7:494-496) | When turn 1 makes no tool call, the probe returns ok=True before `e21_cache_probe` runs, so the prefix-cache gate is silently skipped. |
| 4 | MEDIUM (design) | launch gate in the hidden rerun | Smoke, contract and cache probe are greedy and the same for both configs, with no fail-open. In RERUN_ALL (half A only), one deterministic false failure on both configs leaves no engine, so the LB score is about 0. |
| 5 | LOW | `e21_cache_probe` (c7:517) | `_rnd.Random(7).randrange(10)` builds a new generator per digit, so the "random" 9,000-digit prefix is "5 5 5 …". Every cache block has the same content, which weakens the corruption check. |
| 6 | LOW | `e21_relaunch` (c7:654-656) | Kills and waits for the GPUs but never checks the release (`e21_gpus_free` is not used), so it launches onto held GPUs. |
| 7 | LOW | `e21_relaunch` / `e21_launch` (thread) | After the GPU wait, nothing re-checks that the half is still current. A relaunch that outlives `e21_end_half(A)` can rebind `E18_URL` / `E11_SERVER` during half B. The window is narrow. |
| 8 | LOW | per-half state | `E21_STATE["probe_fail_since"]` (set by the CELL 6 probe) is not reset in `e21_begin_half`. Half B's first /metrics failure can fail open at once. |
| 9 | LOW | capacity → comparison | Streams per half come from each server's own pool. NB1 half A (0.19 INT4, estimated) can get 19 while B gets 20. `comparison_valid` ignores streams. |
| 10 | LOW | launch smoke (c7:611) | `len(set(gen.replace(" ", ""))) > 2` rejects a correct "4" or "4.". The same greedy answer would fail both configs. |
| 11 | LOW | `e21_relaunch` success path, `e21_snapshot_counters`, `e11_decode_summary` | After a relaunch, ENGINE_INFO keeps the old config's backend / capacity / streams. `final_counters` and the decode summary cover only the last server. |
| 12 | LOW | SMOKE profile, launch timeout | The half-A ready timeout is `END − now − 30 min`, with END = T0 + 65 min. A launch that starts after about T0 + 25 min gets less than about 10 min, and a late primary failure gets no fallback. |
| 13 | LOW (outside the engine cell) | CELL 9 c10:69 | It reads `prefix_cache_hit_pct_last`, but `_e15b_server_log_summary` writes `prefix_cache_hit_pct_last_rolling`, so the report field is always None. |

**Streams per config:** every variant config fits at least 16 streams; the floor of 8 never binds. The table is in "Capacity: what each config gets".

---

## Findings

### 1. HIGH (rare path): the second relaunch's config switch makes every NB1 Duck request a 404

**Location:** `e21_relaunch`, c7:639-642 (source 363-366), and c7:657-659.

```python
if len(ENGINE_INFO.get("relaunches") or []) >= 1:          # the same config already crashed once: try the other one
    _alt = [c for _t, c in e21_cfgs(ENGINE_INFO.get("spec") or E21_HALF_SPECS[E21_VARIANT][H]) if c != cfg]
    if _alt:
        cfg = _alt[0]
...
srv, info = e21_launch(H, cfg, f"relaunch{...}", timeout=t)
ENGINE_INFO["cfg"] = cfg
```

**Evidence:**

1. **The served name follows the weights.** `e21_duck_serving_flags(E21_SERVED[cfg["weights"]], port)` gives `"Qwen/Qwen3.8-27B-INT4"` for int4 and `"Qwen/Qwen3.8-27B-FP8"` for fp8.
2. **Duck's model id is fixed once per half.** CELL 6 (c8:432) sets `E12_RUN["model_id"]` from `/v1/models` of the first server, then passes `model_id=E12_RUN["model_id"]` into `solve_all_r6` (c8:578).
3. **vLLM rejects any other name.** 0.32 `entrypoints/serve/engine/serving.py:58-82`, `_check_model`, returns `"The model ... does not exist."` with `NotFoundError` / HTTP 404. vLLM 0.19 has the same check.
4. **Reproduced on the harness run `run_NB1_death_232546`:**
   - half A has `relaunch1` with `cfg {"weights": "fp8", "seqs": 16}` (`relaunch_cfg_switched` in `halfA/e12_run_report.json`);
   - `"model_id": "Qwen/Qwen3.8-27B-INT4"` is unchanged;
   - the fake server ignores the model field (fake_vllm_duck_server.py:579), so the harness cannot see the 404.
   - That run's report says `config_status` PRIMARY_BOTH, `comparison_valid` True, and `engine_used` vllm-0.19.0-int4-mtp3-s20 for half A.
   - The harness itself expects CONFOUNDED_FALLBACK in this case (harness_e21_nb.py:569-571), so this check fails on the current code.
5. `exp/follow.py` prints `NB1-B 2nd relaunch switches int4 -> fp8 | Duck model id stays Qwen/Qwen3.8-27B-INT4 | new server serves Qwen/Qwen3.8-27B-FP8`.

**Failure scenario:**
1. In NB1 (either half), the INT4 server dies twice with at least 20 min left.
2. The second relaunch brings up FP8. Its own smoke, contract and cache probes pass, because they use the right name.
3. `SERVER_DOWN` clears, and every Duck request then gets a 404.
4. Each branch burns its 10 consecutive failures with 10-s backoff (arc2_duck.py:2933-3027), so every remaining attempt of the half fails within about 2 min each. The rest of the half is lost.
5. The report shows a valid, primary-config half.

**Related problems on the same path:**
- **Known-bad config.** If half A had already fallen back (INT4 failed at launch), the second relaunch switches back to INT4, the config known not to start. That wastes up to 30 min, and the watchdog then stops (2 relaunches).
- **Stale records.** The switched server's capacity, backend label and streams are never written to ENGINE_INFO / `CONCURRENCY`. Duck keeps 20 workers against a 16-seq server.

**Fix (pick one):**
- **(a) One name for both weights.** Serve both names on every server: `--served-model-name Qwen/Qwen3.8-27B-INT4 Qwen/Qwen3.8-27B-FP8`. vLLM accepts several names; the first is reported. Or use one notebook-level name for both weights.
- **(b) Switch within the same weights only.** Restrict `_alt` to configs with the same `weights`. For NB1 that means: retry the same config. Never switch to a config whose `ENGINE_INFO["attempts"]` shows a launch failure.
- **Also:** on a switch, update `ENGINE_INFO["used"]` (for example "relaunch_fallback"), `backend` and `capacity`, so CELL 9 reports CONFOUNDED.

### 2. MEDIUM: with E21_B_FOLLOWS_A, half B is labelled "primary", so the report calls the comparison invalid

**Location:** `e21_begin_half`, c7:700-705 (source 424-429), and CELL 9 c10:94-103.

```python
if H != "A" and E21_B_FOLLOWS_A and _a.get("used") == "fallback":
    spec = dict(spec, **spec["fallback"])        # the "fallback" key stays in spec
...
for tag, cfg in e21_cfgs(spec):                   # -> [("primary", A's fallback cfg), ("fallback", the same cfg)]
    ...
    ENGINE_INFO.update(cfg=cfg, used=tag, ...)    # used = "primary"
```

CELL 9:

```python
E21_REPORT["config_status"] = (... "PRIMARY_BOTH" if all(v[0] == "primary" ...) else
                               "FALLBACK_BOTH" if all(v[0] == "fallback" ...) else "CONFOUNDED_FALLBACK")
```

**Evidence:**
- `exp/follow.py`: B follows A gives `[('primary', {...fp8, 16...}), ('fallback', {...the same...})]`, and the status with A = "fallback", B = "primary" is `CONFOUNDED_FALLBACK`.
- **Harness run `run_NB2_fallbackA_231944`:** `config_status` CONFOUNDED_FALLBACK, `comparison_valid` False.
  - half A has `config_used` fallback;
  - half B has `config_used` primary with `spec_note` "half A ran its fallback config; half B uses the same config";
  - both engines are `vllm-*-fp8-mtp3-s16`.
  - The harness expects FALLBACK_BOTH here (harness_e21_nb.py:569), so this check fails.
- `_cfgs` (c10:96) is computed but never used. The comparison was meant to look at the configs.

**Failure scenario:**
1. NB1 half A's INT4 primary fails on vLLM 0.19, the case this option was built for.
2. Half B runs the same FP8 config, as the user decided.
3. The report flags the A/B as confounded and invalid, so the one valid comparison is discarded or misread.
4. Also: B's "fallback" attempt relaunches the identical config. A deterministic failure costs B a second ready timeout (≤ 20 min) and shortens B's window.

**Fix:**
- When following A, launch only `[("fallback", cfg)]`. Or keep one identical retry, but tag both "fallback".
- In CELL 9, decide `config_status` from the cfg dicts: equal apart from the `vllm` key gives PRIMARY_BOTH or FALLBACK_BOTH by A's tag; different gives CONFOUNDED.

### 3. MEDIUM: the cache probe is skipped whenever turn 1 makes no tool call

**Location:** `e21_contract_probe`, c7:494-496 (source 218-220) vs c7:507.

```python
if not calls:
    out.update(ok=True, turn2="inconclusive: no tool call in turn 1")
    return out                       # e21_cache_probe below never runs
...
out["cache"] = e21_cache_probe(served)
```

**Failure scenario:**
- The model answers without a tool call, or its low-effort thinking runs past 3,072 tokens.
- The launch passes with no prefix-cache check at all. That check is the align-mode / MTP prefix-corruption gate (tracker: D2 F5; GPT 2/3 #53912).
- The server is accepted, and `info["contract"]` shows ok=True with no `cache` key. Nothing marks the cache gate as not run.

**Fix:** call `e21_cache_probe` before the `if not calls` early return, or in both branches. Record `cache_probe: "not run"` only if it raised.

### 4. MEDIUM (design risk): in the hidden rerun, a probe false-failure on both configs is fatal for the LB

**Location:** the `e21_launch` gate (c7:600-616) with `e21_begin_half`'s loop (c7:736-755). `RERUN_ALL` runs half A only.

**Evidence:**
- The smoke, the contract probe and the cache probe use fixed prompts at temperature 0. The primary and the fallback of a half differ only in weights (NB1) or SSM dtype (NB2).
- A failure caused by the probe's own logic repeats on both configs. Examples: finding 10 ("4."), a quirk of the repeated-digit prompt from finding 5, or a template / parser edge.
- The half then has no engine (`fatal = "no vLLM config became ready"`), and the rerun writes CELL 0's fallback submission.
- S1 / R3K2 had no such gate.
- No real-GPU evidence shows any probe failing today. The real smoke answer recorded in earlier runs, `"2 + 2 equals 4."`, passes. This is a risk on the LB path, not an observed bug.

**Fix:**
- In RERUN only: if every config fails, and the failures are probe failures only (server ready, capacity OK, smoke answer non-degenerate), relaunch the primary and run it.
- Record `gate_failed_open` in ENGINE_INFO.
- The commit keeps the strict gate, which is the place to find such failures.

### 5. LOW: the cache probe's "random" prefix is one repeated digit

**Location:** c7:517 (source 241).

```python
rows = " ".join(str(_rnd.Random(7).randrange(10)) for _ in range(9000))
```

**Evidence:** a new `Random(7)` is built for every element, so every digit is the same. Running it gives `1 distinct digit: '5 5 5 5 …'`.

**Failure scenario:**
- Every 1,600- or 832-token block of the shared prefix has the same token content. Only positions (RoPE) and the GDN state history differ.
- A wrong-block mapping or a stale prefix-cache state is therefore much less likely to change the 96-token reply than with a random prefix.
- A degenerate 18k-token run of "5" is also an odd prompt for the greedy marker echo (see finding 4).

**Fix:** `_r = _rnd.Random(7); rows = " ".join(str(_r.randrange(10)) for _ in range(9000))`.

### 6. LOW: the watchdog relaunch never checks GPU release

**Location:** `e21_relaunch`, c7:648-656.

```python
e21_snapshot_counters()
try:
    if E11_SERVER is not None:
        E11_SERVER.kill()
except Exception:
    pass
_e18_kill_gpu_pids()
_wait_gpu_release()               # result ignored
```

**Failure scenario:**
- The crashed server's workers keep GPU memory past the 180-s wait.
- The relaunch starts anyway and fails with OOM. That burns one of the watchdog's two relaunches and up to its ready timeout.
- The tracker row "GPU release never acted on → `e21_gpus_free` before every launch" does not cover this launch.

**Fix:** use `if not e21_gpus_free("before relaunch"): record and return False, "GPU memory still held"`.

### 7. LOW: a relaunch in flight can outlive its half and rebind half B's globals

**Location:** `e21_relaunch` (c7:634-663) → `e21_launch` (c7:550-628), run on the half's watchdog thread.

**Evidence:**
- The watchdog checks `E21_HALF != _wd_half` only at the top of its loop (CELL 6, c8:517).
- `e21_relaunch` then spends up to 30 + 30 + 180 s killing and waiting before `e21_launch`.
- `e21_launch` assigns `global E18_URL` (the port used by `E18Server.wait_ready`) and `E11_SERVER = srv` without re-checking the half.
- `e21_end_half` joins the watchdog for only 30 s (c7:785-788).

**Failure scenario:**
1. Half A's solver returns while the watchdog is inside the relaunch's kill / GPU wait. This needs the solve to end within about 4 min of a relaunch decision taken with at least 20 min left, which is very unlikely in EVAL120.
2. `e21_end_half(A)` → `e21_begin_half(B)` starts B's launch.
3. The half-A thread then calls `e21_launch("A", …)`. It sets `E18_URL` to port 1234, so B's `wait_ready` probes the wrong server. It sets `E11_SERVER` to the A server.
4. Two servers now load on the same GPUs.

**Fix:**
- In `e21_relaunch`, after `_wait_gpu_release()`: `if E21_HALF != H or _stop_event_set: return False, "half ended"`.
- In `e21_launch`, pass the URL to `wait_ready` instead of using the global.

### 8. LOW: the probe-failure clock carries over from half A into half B

**Location:**
- CELL 6 probe (c8:177-183): `_t0 = E21_STATE.setdefault("probe_fail_since", time.monotonic())`, popped only on a successful read;
- `e21_begin_half` (c7:686-764) never clears it.

**Failure scenario:**
1. Half A's last probe reads fail, for example after CELL 7 kills the server while a probe loop is still alive.
2. `probe_fail_since` stays set.
3. Half B's first failed /metrics read before any success is already "more than 120 s" old. Duck's gate fails open at once instead of holding admissions.
4. `probe_failures` / `probe_fail_open` are also cumulative over both halves.

**Fix:** in `e21_begin_half`: `for k in ("probe_fail_since", "probe_failures", "probe_fail_open"): E21_STATE.pop(k, None)`. Store the counts per half in ENGINE_INFO at `e21_end_half`.

### 9. LOW: the two halves can run different stream counts; `comparison_valid` ignores it

**Location:**
- `e21_capacity` (c7:452): `streams = min(seqs, nb // (cdiv(57k, B) + 15))`;
- CELL 9 `comparison_valid` (c10:101-103).

**Evidence:**
- NB1 half A (0.19 INT4, 20 seqs) has no real log. The estimate is about 1,050 blocks (see the capacity table), which gives 1,050 // 51 = 20 with only about 30 blocks of margin. A slightly smaller pool gives 19.
- NB1 half B (0.32 INT4 + LMO) is estimated at about 1,069 blocks, which gives 20.
- `comparison_valid` checks windows, config tag and deadline, not `streams`.

**Fix:** add `streams` equality to `comparison_valid`, or a `streams_equal` field. Alternatively, cap both halves at the smaller of the two (B's begin knows A's streams).

### 10. LOW: the smoke rejects a correct terse answer

**Location:** c7:611 (source 335).

```python
"ok": ("4" in gen or "four" in gen.lower()) and len(set(gen.replace(" ", ""))) > 2 ...
```

**Evidence and scenario:**
- `"4"` gives 1 distinct character and `"4."` gives 2, so both fail. The real answer seen in past runs, `"2 + 2 equals 4."`, passes.
- If a config answers tersely, greedy decoding gives the same answer on the fallback, so the half has no engine (finding 4).

**Fix:** keep the degenerate check only for long text: `len(gen) >= 8 and len(set(...)) <= 2` is degenerate. Otherwise accept a "4" / "four" answer.

### 11. LOW: after a relaunch, the record keeps the old server's data

**Locations and evidence:**
- **ENGINE_INFO on a successful relaunch (c7:657-660):** only `cfg` is updated. `backend`, `capacity`, `kv_tokens`, `used` and `CONCURRENCY` stay those of the first server.
- **`e21_snapshot_counters` (c7:348-352):** it overwrites `final_counters` with the last live server's counters. vLLM counters restart with each server, so after a relaunch the half's preemptions, prefix hits and tokens cover only the last server.
- **`e11_decode_summary` (c7:300-324):** it reads only `E11_SERVER.log_text()`, the last server's log. CELL 6's own log summary does include the old logs (`_e15b_all_server_logs`).

**Fix:**
- Keep `final_counters` as a list per server, appended at every snapshot, and sum them in the report.
- Pass the old logs (watchdog `old_logs`) into `e11_decode_summary`.
- Update the ENGINE_INFO fields on a relaunch (see finding 1).

### 12. LOW: in the SMOKE notebooks, half A's ready timeout is tight

**Location:** `e21_launch` (c7:586-588) and `e21_begin_half` (c7:708, c7:754-755).

**Timeline (E21SMOKE):**
- `HARD_STOP = T0 + 118 min`, so `_stop = T0 + 115 min`.
- Half A END = 115 − 20 (window) − 30 (reserve) = **T0 + 65 min**.
- The ready timeout is `max(60, min(30 min, END − now − 30 min))`:
  - a launch at T0 + 25 min gets 10 min;
  - a launch at T0 + 30 min gets 5 min.
- After a primary failure later than END − 30 min (T0 + 35), `break` skips the fallback.

**Failure scenario:**
- A slow CELL 2 install or slow staging pushes half A's launch past about T0 + 25 min. E15B staging of 23 GB took about 93-105 s, so this is normally fine, but GCSFUSE slowdowns have been seen.
- The 0.19 launch (about 9 min ready in E12-V) then times out, and the smoke never exercises half A.

**Fix:** in the smoke profile, use `END − now − max(5 min, window)` instead of the fixed 30 min, or raise `time_budget_sec` to 2.5 h.

### 13. LOW (outside the engine cell, found on the way): CELL 9 reads a key the log summary does not write

**Location:** CELL 9 c10:69, `"prefix_cache_hit_pct_last": _sl.get("prefix_cache_hit_pct_last")`.

**Evidence:** `Build/e21_logsum.py` returns `"prefix_cache_hit_pct_last_rolling"`, so the report column and the printed table ("prefix hit … %") are always None.

**Fix:** read `prefix_cache_hit_pct_last_rolling`, and print the whole-run counter value `counters.prefix_hit_pct_whole_run` next to it.

---

## Capacity: what each config gets (`e21_capacity`)

**Method:**
- Pool = `num_blocks`.
- Per stream at the 57k sizing = `cdiv(57,000, B) + 15`. That is 51 blocks for B = 1,600 and 84 blocks for B = 832.
- Streams = min(seqs, pool // per-stream); the config is refused below 8.
- The pool depends on memory and on `max_num_seqs`, not on `max_model_len`, so the 106,496-context VMS pools carry over to E21's 135,168.

**Real measurements behind the table:**
- **0.19 FP8:** E12-V log, 846 blocks (FP8, 16 seqs, LMO, FLASHINFER, 0.90).
- **0.32 without LMO:** 785 (FP8, 16 seqs), 775 (FP8, 20), 989 (INT4, 20), 999 (INT4, 16), 1,508 (FP8 + SSM16, 16; B = 832).
- **0.32 with LMO:** S1, 855 (FP8, 20).
- **LMO gain:** +80 blocks at B = 1,600.

| Config (who uses it) | Pool (blocks) | Source | Streams | Use at 57k / 65k |
|---|---|---|---|---|
| 0.19 FP8 MTP-3, 16 seqs (NB1-A fallback, NB2-A fallback) | 846 | **real** | **16** (16 × 51 = 816) | 96.5 % / 106 % |
| 0.19 INT4, 20 seqs (NB1-A primary) | ~1,050 (846 + INT4 saving 214 − ~10 for 20 seqs) | estimate | **20** (19 if pool < 1,020) | ~97 % / ~107 % |
| 0.19 FP8 + SSM16, 16 seqs (NB2-A primary) | ~1,625 (846 × 1,508 / 785) | estimate | **16** (fits 19) | ~83 % / ~92 % |
| 0.32 FP8 + LMO, 16 seqs (NB1-B fallback, NB2-B fallback; half B when it follows A) | ~865 (785 + 80) | estimate (S1 at 20 seqs: 855 → 16, real) | **16** | ~94 % / ~104 % |
| 0.32 INT4 + LMO, 20 seqs (NB1-B primary) | ~1,069 (989 + 80) | estimate | **20** (19 without the LMO gain: 989 // 51) | ~95 % / ~105 % |
| 0.32 FP8 + SSM16 + LMO, 16 seqs (NB2-B primary) | ~1,668 (1,508 + 2 × 80) | estimate (1,508 without LMO → 17, real) | **16** | ~81 % / ~89 % |

**Conclusions:**
- Every config gets 16 to 20 streams, so the floor of 8 never binds.
- The configs that run at about 95 % or more of the pool at the 57k p50 peak are: both FP8 16-stream configs, and both INT4 20-stream configs. At p90 (65k) they exceed the pool, so preemptions are possible in the tail. Duck's 0.95 KV gate covers this, as the tracker's 57k decision states.

**Checks of the block math itself (`exp/cap.py`):**
- The math reproduces integer pools on every align-mode MTP-3 0.32 log: 785.0, 792.0, 775.0, 764.0, 989.0, 999.0, 1,508.0 and 855.0. The value is computed exactly from the logged KV tokens / max_model_len.
- It reproduces 846 on 0.19: 8.46 × 100. The A1 points at 81,920 / 106,496 also give 846.
- On the logged two-decimal X, the error is at most ±0.89 block (B = 832, 178 blocks per max-length request).
- `F = 3 × (2 + 3) = 15` matches the 0.32 source:
  - `MambaSpec.max_memory_usage_bytes`, align mode, is `2 + num_speculative_blocks + num_prefill_checkpoint_blocks`;
  - `num_prefill_checkpoint_blocks` defaults to 0 (kv_cache_interface.py:1042, 1079);
  - `mamba/abstract.py` sets `num_speculative_blocks = num_speculative_tokens` (no ReplaySSM);
  - "Add 3 padding layers" = 17 attention layers (16 + MTP) and 48 GDN layers in 3 groups of 17.
- The TP 2×2 VMS logs (2 and 6 streams) are correctly refused (< 8). They are not E21 configs.

---

## Verified as correct

**Definitions and globals:**
- A `symtable` scan of every function and module statement in cell 7 against cells 0-6 (`exp/names.py`) found every name defined before use. The only ones not bound earlier are `E21_HALF`, `E21_HALF_END_PC` and `E21_HALF_T0`. `e21_begin_half` assigns them (global) before any reader runs: `e21_launch`, `e21_relaunch`, CELL 6.
- **Defined earlier:**
  - CELL 0: `threading`;
  - CELL 1: `CONFIG["e21"]`, `SGL_GIVE_UP_AT`, `SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": 3}`, `KV_CACHE_DTYPE`;
  - CELL 1b: `E14_*`;
  - CELL 2: `pkg_version`;
  - CELL 3: `E12A`, `_E12_TA`, `E10_SANDBOX_SELFTEST`.
- **Duck attributes exist:**
  - `E12A.ARC2_TOOL_DESCRIPTION` (arc2_duck.py:458);
  - `E12A.SERVER_DOWN` / `SERVER_WAITS` / `SERVER_LAST` / `SCHED_LAST` (3656-3661);
  - `_E12_TA._PRIORITY_GATE` / `_PRIORITY_GATE_LOCK` (tool_agent.py:2139-2140) and `_GATE_STATS` (1919).
- `E21_STATE` is created at module level before any function that uses it is called.
- **Gate reset works:** the gate is rebuilt with the env read at creation (`_priority_gate()`, tool_agent.py:2143-2154), and CELL 6 sets `ARC3_MAX_ACTIVE_STREAMS` before the preflight.

**Stats regex and summaries:**
- The rev-2 `_E12V_STAT_RE` matches every stats line of all 16 real logs: E12-V 46/46, S1 164/164, every VMS log n/n.
- It also matches synthetic lines with `Deferred`, `KV fetch`, `Preemptions`, no prefix field, and a trailing `Corrupted`.
- **Line formats confirmed in the source:**
  - 0.19 `loggers.py`: Preemptions only when > 0;
  - 0.32: Deferred, KV fetch, Preemptions, `%s KV cache usage` with `kv_cache_device = "GPU"` on CUDA (loggers.py:117), and the prefix field omitted while empty.
- `e21_stat_rows` and both consumers (`e11_decode_summary`, `_e15b_server_log_summary`) use the new group layout. No other user of the old layout is left.

**Prometheus counters:** `vllm:num_preemptions`, `prefix_cache_queries` / `hits`, `generation_tokens`, `prompt_tokens` and `request_success` exist in both versions (counters → `_total`). `e21_prom` parses them, and `e21_counters` sums the label rows.

**Capacity regexes:** they match the 0.19 two-line form and the 0.32 one-line form. "Setting attention block size" is always logged for this model, because the default `block_size` is below the attention block size (0.19 `models/config.py:276-285`; 0.32 `interface.py:1096`, seen in every VMS log). A missing line correctly fails closed.

**Contract probe against the real parsers:**
- **History rendering:**
  - 0.19 reads the history key `message["reasoning"]` (chat_utils.py:1512);
  - assistant tool-call `arguments` are converted from a JSON string to a dict for the template (1573-1578).
- **Truncated thinking:**
  - 0.19 `Qwen3ReasoningParser`: thinking enabled with no `</think>` puts everything in reasoning, with content None. So truncated thinking is inconclusive, not a markup failure.
  - 0.32 `parser/qwen3.py`: a duplicate `</think>` in CONTENT state is absorbed (lines 134-139), and `<tool_call>` inside reasoning ends the reasoning. So `</think>` cannot leak into content, and the D1 F4 case does not trigger a false markup failure.
- **Tool schema:** `E21_PROBE_TOOLS` equals Duck's `ToolAgent._tools()` schema (tool_agent.py:5184-5205) with the ARC2 description (arc2_duck.py:1572).
- **Shortfall threshold:** `max(128, len(reas) // 6)` against a normal growth of about −15 tokens (tool message + `<|im_end|>`) leaves no false failure from content dropped before a call.

**Smoke and the language-model-only check:**
- The past real smoke answer `"2 + 2 equals 4."` passes.
- The marker `'language_model_only': True` appears in real 0.19 (E12-V line 7) and 0.32 (S1) logs.

**Version check:** `pkg_version("vllm") == "0.19.0"` cannot fire spuriously. CELL 2 already raises unless the installed version is exactly "0.19.0".

**Readiness:**
- `wait_ready` needs /health plus the KV line. The KV line is logged long before /health answers on both versions.
- The longest startup log silence in the real logs is 46-65 s, far below the 600-s stall rule.

**GPU release check:** the threshold of 1,024 MiB is above Kaggle L4 idle (recorded `last_used_mb [300, 300, 300, 300]`), so `e21_gpus_free` passes on a fresh box. That matters for the first launch of the hidden rerun.

**vLLM 0.32 install thread:**
- A failure is cached as `{"error"}`, and `e21_launch` raises it without reinstalling.
- The join (≤ 2,400 s) happens before the timeout is computed.
- A second, concurrent install happens only if the thread runs more than 40 min past half B's start.
- The 0.32 env captured during half A differs from half B's only in Duck URL variables, which are irrelevant to the server.

**`e21_end_half`:**
- The kill is always in `finally`.
- The last half kills fast and skips the GPU wait. CELL 7's `e11_kill_server(wait=False)` → `kill(fast=True)` snapshots the counters first.
- The thread names `e15b_watchdog` / `e12_heartbeat` match CELL 6.
- **Scheduler-thread count:** `e21_scheduler_threads` works on Python 3.12, where `Thread._target` stays set while the thread runs. Duck's workers and probe are `solve_all_r6.<locals>.worker` / `probe_loop` (arc2_duck.py, single `solve_all_r6` at 3220; threads at offsets 364-366).

**Equal windows:**
- CELL 6's `deadline_in_sec` is the relative window measured after the preflight shift (c8:489), so `_wa < E21_HALF_SOLVE_SEC` compares like with like.
- B's window is restored by the same shift.
- In the hidden rerun there is no window: RERUN_ALL has no `solve_window_sec`, so `E21_HALF_END_PC = _stop`.

**Flags:**
- `E21_FLAGS` records all 8 commands per variant: 2 halves × primary / fallback × both backends, with the staged paths.
- The 0.19 flag set matches the E12-V non-default-args line, plus `--max-num-seqs` per half, `--disable-uvicorn-access-log` and the optional SSM flag.
- `--mamba-ssm-cache-dtype` reaches GDN in 0.19 (`qwen3_5.py:708-716` → `MambaStateDtypeCalculator.gated_delta_net_state_dtype` with `mamba_ssm_cache_dtype`).

**Staging join:** it is the E15B donor logic (join until min(SGL_GIVE_UP_AT − 20 min, T0 + 55 min); result `(paths, report)`). It fails open to the original paths.

## Files

- Report: `D:/ARC AGI 2/experiments/Experiment 21 - Duck config B on vLLM 0.19 + 0.32 (planned)/Experiment Review/VAL2_engine.md`
- Extracted cells (all four notebooks): `D:/ARC AGI 2/_scratch/val2_engine/cells/`
- Experiments: `D:/ARC AGI 2/_scratch/val2_engine/exp/names.py`, `statre.py`, `cap.py`, `gaps.py`, `follow.py`
- vLLM sources fetched for this pass: `D:/ARC AGI 2/_scratch/val2_engine/vsrc/`
- Harness evidence:
  - `D:/ARC AGI 2/_scratch/e21/harness_runs/run_NB1_death_232546/q27x/` (finding 1);
  - `D:/ARC AGI 2/_scratch/e21/harness_runs/run_NB2_fallbackA_231944/q27x/e21_report.json` (finding 2).
