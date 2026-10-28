# Audit B: E18 notebook behaviour vs S1-60K (2026-10-27)

**Scope.** A read-only audit before the GPU run. No project file was changed.

**What was audited:**
- `Experiment Notebooks/Q27-XHIGH-E18_S1_60K_VLLM032_S20.ipynb` (sha `07af34e3`), built by `Build/build_e18.py`, with the engine cell `Build/e18_engine_cell.py`;
- the base: S1-60K, `Experiment 10 .../Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb` (sha `ed530071`, LB 34.58);
- the push metadata `_scratch/push_e18/.../kernel-metadata.json`;
- the harness `src/harness/harness_e18_nb.py` with `fake_vllm_e18_server.py`;
- the Kaggle CPU kit output `_scratch/e18_cpu_kit_v1`;
- the evidence: the Experiment 14 VMS run (`_scratch/ems_out/vms`), the vLLM 0.32.0 source excerpts (`_scratch/vllm_whl/src032`) and the Experiment 10 GPU runs (R5, D66, W3H).

**Cell numbering.** "Cell i" is the index in the `.ipynb`. The 0.32 engine cell is index 6 and the solver is index 7. The harness numbers code cells only, so its indices are one lower.

## Summary

- **BLOCKER:** none. No path found that writes an invalid `submission.json`.
- **HIGH:** 1 (H1). At 20 streams the vLLM 0.32 KV pool is oversubscribed; the README's capacity check is wrong.
- **MEDIUM:** 3 (M1 to M3). Server death mid-run, process-group kill, worst-case startup.
- **LOW:** 9 (L1 to L9).
- **INFO:** the verified equivalences, and what the harness does not cover.

---

## 1. Cell-by-cell diff against S1-60K

| Index | Cell | E18 vs S1-60K |
|---|---|---|
| 0 | markdown | changed (E18 note prepended, S1 note kept) |
| 1 | CELL 0 | **byte-identical** (`abe694be`) |
| 2 | CELL 1 | changed (see below) |
| 3 | CELL 2 install + tokenizer | **byte-identical** (`269a0d95`) |
| 4 | task order | **byte-identical** (`35a38d67`) |
| 5 | Poetiq core | **byte-identical** (`21d72950`) |
| 6 | CELL 5 engine | new; equals `e18_engine_cell.py` plus the inserted `S1_ENGINE_SRC` line |
| 7 | solver | **byte-identical** (`59f619b6`) |
| 8 | report | **byte-identical** (`ef1e6261`) |
| 9 | CELL 8b | new |

The hashes are the first 8 hex digits of the sha256 of each cell's source.

**Also checked:**
- `S1_ENGINE_SRC`, after `json.loads`, equals S1's cell 6 exactly.
- The engine cell minus the inserted lines equals `e18_engine_cell.py`.
- The notebook metadata equals S1's.
- The S1 base equals the version pulled from Kaggle (`_scratch/s1_55k/pull/q27-xhigh-e10-s1-60k-poetiq.ipynb`): all 9 cell sources are identical.

**CELL 1 diff (`diff s1 e18`).**
- The default commit profile goes from `S1COMMIT4` to `E18SMOKE`, and a new `E18SMOKE` row is added (lines 2 and 9-13).
- `MAX_NUM_SEQS` default goes from 16 to 20 (line 59).
- New `E18_VLLM_WHEELHOUSE` path (lines 36-37).
- **A 4th edit not in the README's "three changes":** `CONFIG["e18"] = {...}` (lines 99-100). It is informational only: it is printed and stored in the report. No behaviour change → INFO.

**Other CELL 1 checks:**
- The `E18SMOKE` deadlines are admit 35, drain 36, hard stop 42 and verify 44 min. The assert on line 88 holds.
- All 20 `E18SMOKE` task ids are in the evaluation set (20 tasks, 22 test inputs).
- Every other value (`THINK_BUDGET` 60,000, answer 8,192, `MAX_MODEL_LEN` 106,496, sampling, KV gate 0.95 / 12, `RERUN_ALL` margins) is unchanged.

---

## 2. Findings

### H1 — HIGH: at 20 streams the vLLM 0.32 KV pool is oversubscribed; the README capacity check is wrong

**What the number means in vLLM 0.32.**
- The "GPU KV cache size" line is not the raw pool. `get_kv_cache_capacity` returns `max_concurrency × max_model_len` (`src032/vllm_v1_core_kv_cache_utils.py` 2300-2307).
- `max_concurrency = num_blocks / Σgroups blocks-per-max-len-request` (1195-1227).
- The VMS `fp8_mtp3_s20` log reads: `GPU KV cache size: 1,006,517 tokens, Maximum concurrency for 106,496 tokens per request: 9.45x` (`vllm_server_fp8_mtp3_s20.log` line 250, block size 1600, line 135).

**Pool in blocks.**
- A max-length request needs 67 attention blocks plus 12 GDN-state blocks (3 groups × (1 + 3 MTP); S1 CELL 1 comment lines 70-72). That is 79 blocks.
- So the pool is about 9.45 × 79 ≈ **747 blocks**.
- Cross-check: in the VMS 8k band, 20 requests used 38.4-41 % of the pool (VMS log, `Running: 20 reqs ... 38.4%`). That is about 14-15 blocks per request: 3-4 attention blocks plus about 11-12 GDN blocks. This matches.

**S1 on vLLM 0.19.**
- The log line is `GPU KV cache size: 337,600 tokens` = pool // 4 groups (W3H and R5 logs).
- So the pool is 1,350,400 = **844 blocks**.

**Per stream.**
- S1: 844 / 16 = **52.8 blocks**.
- E18 (VMS flag set): 747 / 20 = **37.4 blocks** (−29 %).

**Demand at 60k.**
- R5 iteration-1 prompts: mean 6,610 tokens, max 17,617 (`poetiq_iterations.jsonl`).
- Life average ≈ 6.6k + 30k = 36.6k tokens → 23 + 12 = **35 blocks**.
- Peak ≈ 6.6k + 60k + up to 8k → **56-59 blocks**.

**Pool use.**
- E18: 20 × 35 = 700 blocks ≈ **94 % of the pool on average**.
- S1: 16 × 35 = 560 = 66 %.
- S1 was already near full at its peaks: R5 (77.8k, 16 streams) reached max KV 0.995 with 9 preemptions; D66 reached 0.999 with 66.

**Consequence.**
- E18 runs in the KV-gate regime for most of the 12 h. The gate (0.95 usage, ≥ 12 active; solver lines 249-254) holds admission, so the real concurrency is below 20.
- vLLM preempts the newest running request when a growing one cannot get blocks. The preempted request is recomputed (scheduler 797-802).
- The VMS 20-stream gain was +4 % / +6 % / 0 % at 8k / 40k / 90k. The 40k band shared a 36k system prefix, so its KV peaked at only 0.48.
- So the measured upside is at most about 6 %, and the downside (gate throttling plus recompute) is unmeasured.
- This is the main way E18 can score below S1 even when everything works.

**What the README gets wrong.**
- "1,006,517 // 43,000 = 23 ≥ 20" divides a number that already includes the GDN state at maximum length. At 43k live per request the pool holds 747 / (27 + 12) ≈ **19** requests, not 23.
- The smoke estimate "20 × (10k + 29k) = 0.78M below 1.0M, no preemption" leaves out the 12 GDN blocks per request (19,200 token-equivalents each, 384k for 20). Corrected: 20 × (25 + 12) = 740 blocks ≈ 0.99 at the drain. With R5's 6.6k mean prompt: 20 × (21 + 12) = 660 ≈ 0.88.
- `CONCURRENCY` itself is still 20 under either unit (S1 units: 1,195,200 // 43,000 = 27). So the code behaves as intended; only the reasoning is wrong.

**The smoke cannot show the 12-hour regime:** its contexts reach only about 33k by the drain.

**UNVERIFIED:** rung 1's pool with `--language-model-only`. VMS never ran that flag. It may recover part of the 844 − 747 gap (the vision encoder is not loaded).

**Fix (no change to S1 semantics needed):**
1. Correct the README capacity section (block arithmetic above).
2. In CELL 8b, also print `vllm:num_preemptions_total`, `vllm:num_requests_waiting`, the KV usage and the rung-1 KV line. Today `metrics_end` is written but excluded from the printed line.
3. After the smoke, compute blocks per request from `poetiq_vllm_metrics.json` (KV usage vs active, every 120 s), and decide 20 vs 16 before the rerun on that measurement.
4. If the user wants a measured arm, existing env knobs do it without code changes: `Q27X_MAX_NUM_SEQS=16` (S1's value) or a lower `E10_KV_ADMIT_MAX`. The user decides; 20 is the user's spec.

### M1 — MEDIUM: server or engine death mid-run is terminal (no restart); an alive API server with a dead engine burns iterations first

**What happens on a crash (solver cell 7):**
- `E18Engine.step()` raises when `proc.poll()` is set (engine cell lines 281-283).
- The solver's `except BaseException` (lines 357-363) aborts, and `finally` rewrites `submission.json` from identity plus `E10_FINAL` (lines 364-374).
- **The submission is valid, with the results so far, and the rest of the 12 h is lost.**
- S1 behaves the same on an in-process `EngineDeadError`. But E18 adds new failure surfaces: a separate API process, HTTP, and a new engine version that has never run 12 h continuously (VMS launches were short windows).

**The gap before the process exits:**
- Until the API server process exits, streams end with an error event (`serving.py` 497-503 yields an error then `[DONE]`), or with connection resets.
- The shim turns each into `finish_reason "error"` (lines 262-264). The solver treats that as a finished phase 1 → forced-close phase 2 → error → `e10_commit_iteration` with no result → re-queue → admit → error.
- This loop can consume many iterations in seconds. It does not lower the submission (failed iterations add no candidate), but the run log looks "complete".
- **UNVERIFIED:** whether 0.32's API server exits on engine death. The launcher source is not in `src032`.

**Fix (the solver stays byte-identical):**
- Restart the server once inside the shim: `step()` sees the dead server → relaunch with the same command → `wait_ready` (15 min) → emit an `"error"` finish for every open stream.
- `add_request` blocks while restarting.
- Also count `"error"` finishes per minute. If more than 10 arrive in 60 s with the server alive, raise in `step()` (fail fast instead of a burn).

### M2 — MEDIUM: `E18Server.kill()` can skip SIGKILL for the process group

**The code** (engine cell lines 150-169):
- SIGTERM is sent to the group, then the cell waits only for the main API process (`proc.wait(30)`).
- If the main process exits within 30 s, the loop breaks and **SIGKILL is never sent**.
- The engine core and the TP workers are in the same process group. If they are stuck in a CUDA or NCCL call (the likely state after a `wait_ready` timeout or hang), they ignore SIGTERM and keep their GPU memory.

**The chain:**
- Once the API server is gone they are no longer children of the kernel.
- `_wait_gpu_release` (180 s) then fails → `ENGINE_INFO["fatal"] = gpu_memory_not_released_after_failed_engine` (lines 426-428).
- No further rung and no S1 fallback (line 473 requires `fatal is None`) → **identity submission in a rerun**.
- S1's own ladder avoided this with psutil kills of every descendant; here the workers are grandchildren of an exited process.
- **Probability:** low. vLLM's shutdown normally terminates and then kills its children. **Severity:** high.

**Fix:**
- After the SIGTERM wait, always `os.killpg(pgid, SIGKILL)` (ignore `ProcessLookupError`).
- Then kill any PID still listed by `nvidia-smi --query-compute-apps=pid --format=csv,noheader` before `_wait_gpu_release`.

### M3 — MEDIUM: worst-case startup in hang scenarios is about 1-2 h (S1: about 11 min)

**Normal startup:**
- S1-family engine ready at 668-733 s (W3H, R5 logs).
- E18 estimate: CELL 2 about 90 s + install 210 s (VMS) + cold server 537 s (VMS first launch) + smoke ≈ 14-15 min. That is +3-4 min ≈ 0.5 % of 12 h, fine.

**When a rung hangs instead of exiting**, each rung can take:
- `wait_ready` 30 / 20 / 20 min (line 407);
- plus the smoke up to 600 s (line 354);
- plus kill up to 90 s;
- plus the GPU wait up to 180 s.

**Totals:**
- Rungs 1 and 2 time out and rung 3 works: about 70 min lost.
- All three fail by hang, then the S1 fallback: about 130 min (18 % of the run).
- Failures that exit fast (flag rejected, OOM at load) cost only 1-6 min each.
- In `E18SMOKE` a late rung-1 failure leaves no admission time. That is harmless for a commit, but read the smoke's report with this in mind.

**Fix:**
- Fail a rung when its log has not grown for 8 min (stall detector inside `wait_ready`).
- Rungs 2 and 3: 15 min (the weights are in the page cache by then).
- Smoke cap: 120 s (16 tokens).

### L1 — LOW: the 1800 s stream timeout ends a waiting request as an "error" iteration, and the stream is not closed explicitly

**The behaviour:**
- `urlopen(timeout=E18_STREAM_IDLE_SEC)` applies to every recv (lines 19, 229).
- A request that waits more than 30 min in the server (preempted under the H1 pressure, then not rescheduled) hits `socket.timeout` → `"error"`.
- The solver then starts a forced-close phase 2 from the tokens received so far, possibly none. That is a low-quality iteration. S1 has no such timeout.

**The leak:**
- On that exception the shim does not close `st["resp"]`. The server keeps decoding the orphan request (up to 60k tokens, one slot plus its KV) until CPython's GC closes the socket.
- Refcounting makes that prompt, but nothing guarantees it.

**Probability:** low. In vLLM a preempted request goes back to the front of the waiting queue, and requests finish about every 2.5 min.

**Fix:**
- Close `resp` in a `finally` in `_stream`.
- Raise the idle timeout to ≥ 3 h. Server death is already detected by `proc.poll()` in `step()`, and a hung engine would hang S1 too.

### L2 — LOW: an abort before `st["resp"]` is set only takes effect at the first chunk

**The behaviour:**
- If `abort_request` runs while `urlopen` is still waiting for the response headers, `st["resp"]` is None and no socket is shut (lines 312-323).
- The thread stops at the first line it reads (`if st["halt"]: break`). The server keeps the request until then (one prefill).
- Rare: Starlette sends the headers immediately.

**Fix:** after `urlopen` returns, check `st["halt"]` and close at once.

### L3 — LOW: the outer `except` path launches the fallback without a GPU release wait

**The behaviour:**
- Lines 431-436 kill `E18_SERVER` but do not call `_wait_gpu_release` before the S1 fallback at line 473.
- S1's first `LLM()` attempt could hit leftover memory. It would fail and then wait, costing one S1 rung.

**Fix:** call `_wait_gpu_release()` in that branch, and set `fatal` if the memory is not released.

### L4 — LOW: CELL 8b's server stop can take up to 90 s at the end of the rerun

**The behaviour:**
- SIGTERM wait 30 s plus SIGKILL wait 60 s (lines 153-165), after the hard stop at budget − 2 min (budget = 12 h − 3 min).
- Normally vLLM exits in about 5-10 s.

**Fix:** in RERUN, send SIGKILL to the group with no graceful wait, or skip the stop (container teardown kills it). The submission is already final.

### L5 — LOW: the fallback is vLLM 0.19 at 20 streams, not S1 verbatim

**The behaviour:**
- `S1_ENGINE_SRC` reads `MAX_NUM_SEQS` (20) through `ENGINE_CONFIG`.
- With the 0.19 pool (1.35M // 43,000 = 31) `CONCURRENCY` becomes 20. S1 never ran 0.19 at 20 seqs.

**Fix:** if the fallback is meant to be "the proven S1", set `MAX_NUM_SEQS = 16` before the `exec`. Otherwise keep it, and document it.

### L6 — LOW: rung 1 (`--language-model-only`, `--load-format safetensors`, no `--trust-remote-code`) never ran on a GPU under 0.32

**What is known:**
- The flag exists (`src032/vllm__engine__arg_utils.py` 633, 1460).
- VMS's greedy gate covered only the VMS flag set (rung 2).
- The smoke checks load and a 16-token decode only. A failure costs one rung (M3).

**Fix:** none required. Compare rung-1 MTP acceptance and KV in the smoke's metrics against VMS.

### L7 — LOW: the E18SMOKE drain window and its expectations

**The drain window:**
- Drain 36 → hard stop 42 min = 6 min. At 20 streams about 21-22 tok/s per stream (VMS 423-453 tok/s aggregate).
- A full 8,192-token answer needs about 6.5 min, so long answers are cut at the hard stop. Typical code answers (1-3k tokens) fit.

**KV:** by the drain the smoke may approach the KV gate (H1); the README says no preemption is expected.

**Late engine:** if the engine is ready late (GCSFUSE slowdown: SGLang took 1,197 s on 2026-10-21; or rung 1 fails), only a few minutes of thinking remain.

**Fix:** none in code. Read the smoke as a functional and throughput check, not an accuracy estimate.

### L8 — LOW: no foreign-listener guard on port 8011

**The behaviour:**
- VMS rev 2g-b failed a launch whose own log lacked the KV line ("another process on the port").
- E18 accepts `/health` 200 and then parses the KV line. If the line is missing, `kv_tokens` is None and `CONCURRENCY` becomes 20 without a check.
- The smoke would still catch a wrong server in most cases.

**Fix:** require the KV line in this server's log before accepting the rung, as VMS does.

### L9 — LOW: `/metrics` is a synchronous HTTP call in the solver's main loop (10 s timeout)

**The behaviour:**
- The KV gate calls it at most once per 5 s with ≥ 12 active (solver lines 237-247), and `metrics_snapshot` calls it every 120 s.
- A slow API event loop stalls admission and `step()` for up to 10 s. No tokens are lost (they queue in the shim).
- On a timeout the gate reads 0.0 and stays **open**, as in S1.

**Fix:** a 2 s timeout for the gate read.

---

## 3. Solver (cell 7) on the shim: verified equivalences (INFO)

**`add_request`.**
- The shim posts `/v1/completions` with the kernel's token ids. The prompt ids are byte-identical to S1's: the same transformers 4.57.6 tokenizer and chat template run in the kernel, and the server never re-tokenizes.
- Fields sent: `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stream`, `return_token_ids`, `skip_special_tokens`.

**Sampling parity** (vLLM 0.32 source plus the model's `generation_config.json`):
- The OpenAI server fills unset fields from the generation config (`completion/protocol.py` 303-311 and 346-359; `config/model.py` 1864-1887).
- The model config has only `temperature`, `top_k` and `top_p` (plus eos and pad). All three are sent explicitly.
- `repetition_penalty`, `presence_penalty`, `frequency_penalty` and `min_p` keep the neutral defaults, as with S1's explicit `SamplingParams`.
- There is no `max_new_tokens`, so no server cap on `max_tokens`.
- EOS / stop ids come from the generation config in both engines.
- Per-request seeds work with spec decoding in 0.32 (`spec_decode/speculator.py` seeds tensors).

**Streaming token ids.**
- `token_ids` is set on every chunk when `return_token_ids` is set (`serving.py` 403-421). The final chunk carries both the last delta ids and `finish_reason`, so finish and tokens are never misordered.
- The first chunk also carries `prompt_token_ids` (a line of up to about 600-700 kB for a phase-2 prefix). urllib's `readline` has no line limit; VMS ran this on Kaggle with 90k-id prompts.
- Empty prefill chunks are skipped (`serving.py` 373-378).

**`step()`.**
- It blocks at most 50 ms and returns the deltas merged per request. Tokens after a finish are dropped. In-process `step()` also returns nothing while the engine is busy.
- The solver loop re-checks the hard stop, the drain and admission each pass, so behaviour is the same.
- The vLLM 0.19 in-process engine core also runs in its own process, so GPU progress does not depend on how often the solver calls `step()` in either engine.

**Abort and deltas after an abort.**
- `abort_request` adds the id to `aborted`, shuts the socket (so a reader blocked in `recv` returns; needed because `BufferedReader.close` takes the buffer lock), then closes it.
- `step()` drops every queued item of an aborted id. The solver also ignores outputs whose phase or iteration does not match (lines 301-302).
- The server aborts on disconnect: `AsyncLLM.generate` catches `CancelledError`/`GeneratorExit` and calls `abort` (`async_llm.py` 776-783).
- On Kaggle, the VMS s20 log shows running requests falling from 20 to 1 when the client closed the streams.

**Drain** (abort phase 1, then phase 2 at once). Phase 2's prefix is the prompt plus the phase-1 tokens the solver had consumed (plus `STOP_IDS` when not closed).
- Tokens that reached the shim queue after the last `step()` are dropped. That is at most about 1 s of decode (about 20 tokens). In-process vLLM drops its undelivered outputs on abort too.
- The new prefix is self-consistent, so nothing is corrupted.
- If the abort has not reached the server yet, the server briefly holds up to 40 requests; the phase-2 requests wait in its queue for milliseconds.
- Prefix-cache hits on the freed phase-1 blocks behave as in 0.19 (mamba `align` mode in both: VMS log line 19).

**Hard stop.** Every active stream is aborted, then `e10_after_decode('hard_stop')` runs, as in S1.

**Unit-error path.** An exception in the per-output processing aborts that id through the shim and marks the unit `unit_error`, as in S1.

**Other per-request errors** become `finish_reason "error"`: HTTP 4xx/5xx raise `HTTPError`, and an in-stream error event or a cut stream is caught. S1 would have raised instead. The phase caps keep `prompt + max_tokens ≤ 106,496` (`phase1_cap` and `MAX_MODEL_LEN − len − 8`), so no 400 is expected. 0.32 admission control is off by default (`max_num_queued_reqs/tokens = None`, `async_llm.py` 316-373).

**KV gate.**
- `vllm:kv_cache_usage_perc` is a 0-1 gauge set on every scheduler iteration (`metrics/loggers.py` 620, 1165), the same name and units as 0.19's `llm.get_metrics()`.
- Admission lag is the same in both: a request takes KV only at the next schedule, and the solver caches the value for 5 s.
- The shim parses Prometheus text, skipping `_bucket` and `_created`. Preemptions appear as `vllm:num_preemptions_total` (the filter "preempt" matches).

**`metrics_snapshot`.** The same filter as S1. The "type" field is `prometheus` instead of the class name (reporting only).

**Report cell.** Byte-identical. It reads `ENGINE_INFO`, `E10_SCHED` and `CONFIG`; all are JSON-serialisable in the E18 path (the install `env` is excluded).

**Threads and memory.**
- Threads: up to 20 stream threads, 4 verification workers and the sandbox subprocesses.
- Memory: bodies up to about 700 kB per stream, about 14 MB in total.
- The shim queue is drained every step.
- The sandbox child runs as uid 65534 with `socket`/`connect` blocked by seccomp, so model code cannot reach port 8011.

---

## 4. Fallback path (engine cell lines 473-485)

**It runs when:**
- every 0.32 rung failed, or the install raised;
- `fatal` is None (see M2 for when it is not);
- the sandbox self-test passed;
- `E18_NO_FALLBACK` is not set.

**What it rebinds.**
- `exec(S1_ENGINE_SRC, globals())` rebinds every name the solver, the report and CELL 8b need (checked from the AST of the embedded source): `SamplingParams` and `RequestOutputKind` (from vllm 0.19), `llm`, `engine`, `ENGINE_READY`, `CONCURRENCY`, `ENGINE_INFO` (a new dict), `make_params`, `metrics_snapshot`, `CUDA_LINK` and `CONFIG["cuda_link"]`.
- The solver resolves `SamplingParams` and `RequestOutputKind` at call time (`e10_make_params`), so it uses vLLM's classes.
- **CELL 2 still installs `vllm==0.19.0`** (byte-identical; the parity check raises if the install fails, exactly as in S1). So the in-process import works.

**GPU and state:**
- The kernel itself never touched CUDA on the 0.32 path (the shim imports no torch), so the in-process engine starts clean.
- After every failed rung, `kill()` and `_wait_gpu_release()` run (except M2 and L3).
- `E18_SERVER` stays None, so CELL 8b skips the stop.
- `getattr(engine, "stats", None)` returns None for vLLM's `LLMEngine`, and `metrics_snapshot` is S1's. CELL 8b works.
- If S1's ladder also fails, it sets `ENGINE_READY = False` without raising, and the identity submission stays (harness `allfail`).
- After the exec, the code labels `ENGINE_INFO["engine"]` "vllm-0.19.0-in-process (E18 fallback)" even when the fallback itself failed. That is cosmetic.

**Caveats:**
- The fallback is 0.19 at 20 streams (L5).
- A successful fallback was never exercised (vLLM 0.19 cannot run in the CPU harness) → UNVERIFIED. But it is S1's own code, which ran on Kaggle.

---

## 5. Deadlines and time

**E18SMOKE** (`TOTAL_BUDGET` 2,700 s from the kernel start):
- Admit until 35 min, drain at 36, hard stop at 42, verify at 44.
- Engine ready at about 14-15 min nominal. That leaves about 20 min of thinking: about 26-27k tokens per unit at about 22 tok/s.
- Most units are force-closed at the drain. Answers have 6 min (L7).
- After the hard stop: the harvest takes at most 10 s, then the report and CELL 8b (up to 90 s). The run ends by about 44 min.
- Kaggle's own limit is 12 h, so nothing is cut. The 45 min is the notebook's own budget.
- In commit mode `submission.json` stays the identity of the test file (as in S1); the score is in `poetiq_run_report.json`.

**RERUN_ALL.**
- The margins are unchanged from S1: hard stop at 12 h − 5 min, drain 8 min earlier, admit 2 min before that.
- The extra startup is about 3-4 min nominal (M3 for the hang case).
- End of run: the solver's `finally` writes the submission at the hard stop. CELL 8b adds up to 90 s (L4).

---

## 6. Failure modes the harness does not cover

**What the harness covers:** 6 scenarios. They passed on the Kaggle CPU kit: Py 3.12.13, notebook sha `07af34e3`, `cpu_harness_report.json` all_ok.

**What it does not cover:**

| Not covered | What happens in the notebook | Finding |
|---|---|---|
| Server crash mid-run | Solver error → valid submission with the results so far; no restart | M1 |
| API alive, engine dead | Error burn of iterations, then as above | M1 |
| HTTP 4xx/5xx or in-stream error event | Unit ends as `"error"` → forced-close phase 2 | §3 |
| A stream that never sends a chunk (preempted or waiting > 30 min) | `socket.timeout` → `"error"` iteration; orphan request until GC | L1 |
| Smoke failure after `/health` 200 | Next rung (code path read, not run) | — |
| GPU memory not released | `fatal` → no fallback → identity | M2 |
| A successful 0.19 fallback | S1 code; the rebinding was verified statically | §4 |
| More than 8 concurrent streams; 20-stream saturation | The harness has 8 tasks; `CONCURRENCY` is never reached | H1 |
| Real KV dynamics, preemptions | The fake reports 0.30 / 0.97 constants and 0 preemptions | H1 |
| Real chunk format (prompt ids in the first chunk, MTP-size deltas) | Verified against the 0.32 source and the VMS run instead | §3 |
| Real timing (deadlines are compressed) | §5 | M3, L7 |

**Disk and `/dev/shm`.**
- The run adds `/tmp/vllm-venv` (the 0.32 stack with torch 2.13, cu13 libs and FlashInfer) on top of CELL 2's system install of vLLM 0.19.
- VMS ran only the tokenizer pins in the system, not vLLM 0.19 (VMS builder line 1107). So the combined footprint has never run in one kernel. The venv is isolated (`venv --without-pip`, `--no-deps`, no system site-packages), so there is no package conflict.
- Disk free space and `/dev/shm` size were not measured → UNVERIFIED. Both installs worked separately on the same image. TP4 uses only small shm message queues.
- Server log growth is small: VMS s20 was 114 kB with request logging off by default.

**Kernel metadata.**
- The same docker digest as S1 (`37c64f7d...`), `NvidiaL4`, internet off.
- The competition source `arc-prize-2026-arc-agi-2` and the model `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1` match `MODEL_PATH`.
- Datasets: `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` (`WHEELHOUSE`) and `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` (`E18_VLLM_WHEELHOUSE`, the same mount path VMS used successfully).
- Every path the notebook reads is attached.
- The notebook is 146 kB, under the 400 kB commit limit.

**Rerun detection.** `KAGGLE_IS_COMPETITION_RERUN` (CELL 0, unchanged) → `RERUN_ALL` (CELL 1 line 2, unchanged).

---

## 7. Other INFO

- **Engine flags.**
  - Rung 1 = S1's `ENGINE_CONFIG` mapped to `vllm serve` flags (TP4, fp8 KV, `max_model_len` 106,496, 4,096 batched tokens, 0.90, custom all-reduce off, seed, MTP 3, prefix caching, FlashInfer only if the `-lcuda` probe passes) with 20 seqs.
  - Rung 2 = the exact VMS `fp8_mtp3_s20` set (log line 9).
  - The env matches VMS: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` and `VLLM_WORKER_MULTIPROC_METHOD=spawn`, both set in CELL 1 and inherited; `PYTHONPATH` and `CUDA_PATH` are stripped.
- **Engine numerics.** In 0.32 the SM 8.9 FP8 kernel is `MarlinFP8ScaledMM`, and spec decode with FlashInfer drops to PIECEWISE CUDA graphs (VMS log lines 115 and 225). VMS's greedy gate vs the oracle was PASS (BOUNDARY: near-tie divergences only). Samples differ from 0.19 anyway; that is inherent and not a defect.
- **No invalid-submission path found.**
  - CELL 0 writes the identity first.
  - Every engine failure keeps the identity.
  - Solver errors keep partial results.
  - CELL 8b and the report are fully wrapped in try/except.
  - Only a CELL 2 parity failure stops Run All early, exactly as in S1, and the identity file is valid then too.

---

## 8. Fixes, by priority

1. **H1.**
   - Correct the README KV arithmetic (747 blocks; 12 GDN blocks per request; 20 × 35 ≈ 94 % of the pool vs S1's 66 %).
   - Print preemptions, waiting requests and KV in CELL 8b.
   - Decide 20 vs 16 from the smoke's measured blocks per request before the rerun (env `Q27X_MAX_NUM_SEQS`; the user decides).
2. **M2.** Always SIGKILL the process group after the SIGTERM grace, and kill the remaining GPU PIDs from `nvidia-smi` before `_wait_gpu_release`.
3. **M1.** One in-shim server restart, plus a fail-fast on an error burst (the solver stays byte-identical).
4. **M3.** A log-stall detector; rungs 2-3 at 15 min; smoke cap 120 s.
5. **L1-L4, L8, L9.**
   - Close `resp` in `finally`.
   - Idle timeout ≥ 3 h.
   - Halt check after `urlopen`.
   - GPU wait in the outer except.
   - Immediate kill in RERUN.
   - KV-line guard.
   - 2 s gate timeout.
6. **L5.** Decide whether the fallback runs at 16 (S1 verbatim) or at 20.
