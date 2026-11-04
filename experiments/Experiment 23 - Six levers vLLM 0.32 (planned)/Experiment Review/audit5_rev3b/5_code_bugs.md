# E23 rev 3b: plain-bug audit (agent 5)

Scope: `src/e23_lab.py`, `Build/build_e23_notebook.py` (CELL1..CELL6, the extracted `nb3b_*.py`), `Build/build_cpu_kit_e23.py`, the
Exp 14 `fake_vllm_server.py`. Evidence: `nbtest/run{A,B,F,B3}.log` and `out*/e23_results.json`, `kitout_v1/`.
I ran local checks only in `_scratch/e23/audit5_rev3b/work/`:
- `t_degen*.py`: `degenerate()` on 384 real Qwen3.8 outputs (E19 GPU run1 rows);
- `t_lab.py`: `compare()` on an error row, and the barrier path against the fake server;
- `t_merlin.py`: Merlin `launch.build()` with the notebook's own arguments;
- the PR #60532 dry-run against stock v0.32.0 `flashinfer.py`;
- `trace_summary()` on a real torch trace.

Line numbers: `L` = `src/e23_lab.py`. `B` = `build_e23_notebook.py`.

---

## BLOCKER

### B1. `degenerate()` LOW_VARIETY flags healthy ARC reasoning as garbage, so nearly every GPU window becomes FAIL_GARBAGE_UNDER_LOAD
- **Where:**
  - L:430-432 (`words = re.findall(r"\w+", t)` ... `< 0.08 * len(words)`);
  - used at L:663-667 (steady), L:455-457 (compare), B:467 (oracle `garbage`) and B:760 (PP4 gate).
- **Evidence:** I ran `degenerate()` on the 384 real Qwen3.8-27B outputs in
  `Experiment 19 .../GPU_run1_2026-10-27/*/q27x/e19/e19_rows.jsonl`. These are healthy 1024-token ARC thinking outputs from vLLM 0.32.
  - Tail 4000 chars (what steady checks): **120/384 = 31 % flagged LOW_VARIETY**.
  - First 250 chars (about the 64-token SMOKE oracle row): **96/384 = 25 %**. First 450 chars (about 128 tokens): 91/384.
  - Cause: grid dumps such as `Row 0: 3 3 3 3 3 ...` and `5 5 5 5 ...` are many `\w+` "words" with about 10 distinct values.
  - In the fake, the text is a hex hash: one long `\w` run, never more than 40 words. So the check never runs in any fake run or kit.
- **Failure:**
  - (a) `steady()`: a window of 16-24 streams is STEADY only when no stream is flagged. P(at least one flagged) = 1 - 0.69^16 = 99.7 %.
    So virtually every STEADY or CAPACITY_LIMITED window becomes FAIL_GARBAGE_UNDER_LOAD.
    - The numbers are kept, but the statuses are wrong.
    - `steady_multi` never gives AGGREGATE_STEADY.
    - `_tp1_level()` falls back to its defaults (2 / 1), so the replica steps run at the wrong level.
    - CELL6 shows "0/N" steady windows everywhere.
  - (b) PP4: `any(o["oracle"]["garbage"])` gives a false NO_GO_GARBAGE_OR_ERROR (about 1 - 0.75^4 = 68 %), and the PP4 windows are never measured.
  - (c) `compare()`: when the oracle and the candidate diverge (NVFP4 KV, MTP near-ties), the candidate's text alone can be flagged.
    The row then becomes FAIL_HIGH_MARGIN. In `s5(0)` that gives NO_GO_CORRUPT, which skips the NVFP4 census and `nvfp4_mtp3`.
- **Caveat:** the E19 prompts had spaced grids; our prompts use unspaced rows. The model often rewrites grids with spaces when it reasons.
  Even at a much lower rate, 16-24 streams per window make a false FAIL near-certain.
- **Fix (tested on the same 384 rows: 0 false positives at the tail and at 250 chars; `"the cat " * 300` is still flagged):**
  ```python
  words = re.findall(r"[A-Za-z]{3,}", t)          # grid digits are not words
  if len(words) > 60 and len(set(words)) < 0.08 * len(words):
      return "LOW_VARIETY"
  ```
  Also give the fake server real-looking text, for example grid rows plus English, so the kit exercises this branch.

---

## HIGH

### H1. `teacher_forced()` (prompt_logprobs at 40k / 90k) can OOM-kill the stock server, and it runs before the windows it shares a server with
- **Where:** L:484-504; called at B:639 (`s6(0)`, before all 9 fp8_mtp0 windows) and B:723 (`s5(0)`, before the NVFP4 90k / 120k census).
- **Why:**
  - In stock **vLLM v0.32.0**, `gpu_model_runner._get_prompt_logprobs_dict` has no budget. I checked it in `_scratch/e23/p4/vllm`, tag v0.32.0.
  - It calls `compute_logits(hidden_states[offset:offset+num_logits])` on the whole scheduled chunk (up to `--max-num-batched-tokens 4096`).
    Then `compute_logprobs` (fp32 log_softmax) and `gather_logprobs` (a rank compare over [num_logits x vocab]).
  - At 4096 x 150-250k vocab that is about 1.2-2 GB of bf16 logits, 2.5-4 GB of fp32 and about 0.6-1 GB of bool, all transient.
    vLLM's profile run does not see any of it, because its dummy sampler covers max_num_seqs rows only. It sits on top of `--gpu-memory-utilization 0.90`.
  - Merlin's own fork added `vllm/merlin/prompt_logprobs.py` with a "per-step GPU tensor budget" of 384 MiB and 128-row scoring chunks. That is a strong sign the stock path OOMs on these cards.
  - The fake server always answers with SSE, so `json.loads` fails and the function returns `{"error":...}`. It has never run against a real server.
- **Failure:** CUDA OOM in EngineCore, then the server dies.
  - In `s6(0)` every 40k / 90k / 120k window of the MTP-0 anchor is lost.
  - In `s5(0)` the whole NVFP4 residency census is lost.
  - Each later level then spends 60 s in `wait_idle` on a dead port (about 9 min per step), and the step reports SERVER_DIED.
- **Fix:**
  - Move `tforced(...)` to the very end of both bodies, after `levels(...)`.
  - Guard it: `if ss[0].alive(): o["teacher_forced"] = tforced(...)`, then record `alive_after_tf = ss[0].alive()`.
  - Optional: run the TF on its own short-lived server with `--max-num-batched-tokens 1024` (logits ≤ 1 GB fp32).

### H2. `compare()` turns a candidate request error into FAIL_HIGH_MARGIN
- **Where:** L:444-457. An errored row gets `st = "INCONCLUSIVE"`. Then `degenerate("")` returns "EMPTY", and the oracle is clean, so `st = "FAIL_HIGH_MARGIN"`.
- **Reproduced** in `work/t_lab.py`: a `GREEDY_TIMEOUT` row with empty text versus a clean oracle gives `FAIL_HIGH_MARGIN`.
- **Failure:** any request error on the candidate is reported as a high-margin divergence. Examples:
  - an HTTP 400 for an unsupported parameter (for example `logprobs` on the EAGLE tree-verify fork);
  - the 40k LONG_P row hitting its 120 s timeout late in the budget;
  - a server that dies mid-oracle.
- **Consequences:**
  - `s5(0)` sets NO_GO_CORRUPT, which skips the NVFP4 windows and `nvfp4_mtp3`.
  - EAGLE, PP4 and G1-G3 read as "corrupt", when the truth is "request failed". The docstring says missing data means INCONCLUSIVE.
- **Fix:** run the garbage override only on clean rows:
  ```python
  if g and not b.get("error") and b.get("tokens") and degenerate(a.get("text")) is None:
      st = "FAIL_HIGH_MARGIN"
  ```

---

## MEDIUM

### M1. Nothing a step measures is persisted until the step returns, so a hard kill loses the whole step
- **Where:**
  - B:484-495: `levels()` calls `save()`, but its `out` is local.
  - B:421-450: `with_servers()` builds `out` locally.
  - B:407-408: `rec.update(out)` runs only after `fn()` returns.
- **Failure:**
  - A Kaggle session kill (12 h overrun, host OOM with 4 Merlin replicas plus page cache, kernel death) during a 60-100 min step loses every window and server fact of that step.
  - The `save()` calls inside `levels()` write nothing new.
  - The B:440 comment "an exception in the body keeps everything measured so far" is not true: when `body()` raises, its local `o` (probes, oracle, earlier `steady` dicts) is dropped. Only BODY_ERROR is recorded.
- **Fix:**
  - Give `step()` a live record and write into it as you go.
  - For example, set a module global `CUR = rec` in `step()`.
  - `with_servers` then writes `CUR["servers"] = out["servers"]` right after ready.
  - `levels()` does `CUR.setdefault("steady", {})[key] = res` before each `save()`.
  - Bodies can take a dict `o` created by `with_servers` and fill it in place, so an exception keeps what was already measured.

### M2. Replica barrier is not aborted on `INCONCLUSIVE_STREAMS_ENDED_IN_ADMISSION`
- **Where:** L:556-558: the return sits inside `try` and has no `barrier.abort()`. Every other early return aborts.
- **Reproduced** (`work/t_lab.py`, fake servers, `admit=60`):
  - replica B ended in admission at 10 s;
  - replica A then waited on the barrier and returned `INCONCLUSIVE_REPLICA_BARRIER` after **78 s**.
- **Failure:** with the real `admit = max(420, 120 + level*plen/700)` (420-2100 s), the healthy replicas idle on GPU for 7-35 minutes while their streams decode.
  This happens whenever one Merlin TP1 replica dies or rejects its requests.
- **Fix:** `if barrier: barrier.abort()` before that `return r`.

### M3. A P4 bench timeout can leave GPU 3 held and abort the whole FULL lab; `join_p4()` ignores the release gate
- **Where:** B:531-544 (`subprocess.run(..., timeout=3600, start_new_session=True)`) and B:558 (`L.wait_gpu_release(180, gpus="3")`, result discarded).
- **Mechanism:**
  - `bench_fp8_gemm.py` runs each shape in a child via its own `subprocess.run(timeout=900)`.
  - 5 shapes x 900 s = 4500 s, which is more than the outer 3600 s (SMOKE: 2 x 900 = 1800 s against 1500 s).
  - On `TimeoutExpired`, `subprocess.run` kills only the bench parent. The running child shares that process group and survives on GPU 3.
  - Then `c_merlin_tp1_x4` replica 3 and `d_merlin_tp4` launch on a held GPU 3 and fail with LAUNCH_FAILED.
  - `with_servers`' gate on `"0,1,2,3"` then sees more than 1024 MiB used on GPU 3, returns FAIL_GPU_RELEASE and sets ABORT. Every remaining step is skipped.
- **Fix:**
  ```python
  p = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, env=..., start_new_session=True)
  try:
      out_txt = p.communicate(timeout=T)[0]
  except subprocess.TimeoutExpired:
      os.killpg(p.pid, signal.SIGKILL)
      out_txt = p.communicate()[0]
      rc = -9
  ```
  In `join_p4()`: `if L.wait_gpu_release(...) is False: res["gpu3_released"] = False`, and skip or mark the steps that use GPU 3, rather than running into the abort.

---

## LOW

- **L1.** Hard-stop status mapping and fallback launch after the stop.
  - **Where:** B:414-415, B:574-576, B:619-620, B:487-488.
  - The watchdog kill is recorded as SERVER_DIED (body) or LAUNCH_FAILED (wait_ready). Only DONE becomes PARTIAL_HARD_STOP, so the last step reports an engine death that the watchdog caused.
  - LAUNCH_FAILED also triggers the TP1 / TP4 fallback. That launches a new server after the watchdog has finished its single kill pass; it is bounded by `timeout = 60` and killed by `with_servers`.
  - `levels()` labels ABORT skips as SKIPPED_BUDGET.
  - **Fix:**
    - in `step()`: `if ABORT["why"] and not rec["status"].startswith("SKIPPED"): rec["status_raw"] = rec["status"]; rec["status"] = "PARTIAL_HARD_STOP"`;
    - gate both fallbacks on `not ABORT["why"]`;
    - use `"SKIPPED_ABORT" if ABORT["why"] else "SKIPPED_BUDGET"`.
- **L2.** A dead server keeps getting levels.
  - **Where:** B:486-494.
  - Each level does `wait_idle(port, 60)` and returns INCONCLUSIVE_NOT_IDLE_AT_START: about 1 min per level, 9 min for `s6(0)` / `s6(3)` in FULL.
  - **Fix:** `levels(port, ...)` takes the Server, or checks `L.port_listening(port)`, and stops when the server is dead.
- **L3.** `prom()` is not wrapped in `try` in the steady window or in the admission summary.
  - **Where:** L:583, L:601.
  - One transient `/metrics` timeout (15 s) under load raises, and `levels()` records the whole level as ERROR.
  - **Fix:** `try/except: continue` in the window loop (count misses), and wrap the admission `prefill_tok_s` read.
- **L4.** `save()` races the prefetch thread.
  - **Where:** B:169-174, B:64-68.
  - `R["setup"]["prefetch_<tag>"] = ...` runs without `_SAVE_LOCK`. If it lands while the main thread's `json.dumps(R)` is iterating `R["setup"]`, the main thread gets "dictionary changed size during iteration".
  - `step()`'s `save()` sits outside its `try`, so this would kill CELL5. The window is tiny (two inserts per run).
  - **Fix:** compute `res = L.prefetch(...)`, then `with _SAVE_LOCK: R["setup"][...] = res`, then call `save()`.
- **L5.** The revert check is tautological.
  - **Where:** B:665-667. `_patch(reverse=True)` compares the restored file with the backup it was just copied from, so it is always True.
  - **Fix:** compare it with `pre["orig_sha"]`, the sha recorded before the patch.
- **L6.** A failed P4 gate is reported as BENCH_ERROR.
  - **Where:** B:544. `s4_thread` maps rc 1 to BENCH_ERROR, but `bench_fp8_gemm.py` returns 1 when a gate fails (`all_ok` False). That case should be CHECK_FAILED, as in `s2`.
- **L7.** The log handle leaks when Popen fails.
  - **Where:** L:73-76. `Server.start()` opens `_fh` before `Popen`. If Popen raises (bad argv, ENOENT), `kill()` returns early because `proc is None`, and the handle is never closed.
- **L8.** The kit verdict ignores window statuses.
  - **Where:** `build_cpu_kit_e23.py` L:155-158.
  - Kit v1 had SMOKE_B `120k@12 = INCONCLUSIVE_STREAMS_ENDED_IN_WINDOW` and still counted as a pass on that axis.
  - **Fix:** require every fake window to be STEADY or CAPACITY_LIMITED.
- **L9.** Graph-verdict fallback.
  - **Where:** L:769-774. With no per-forward ranges, any eager attention kernel gives PIECEWISE. The MTP drafter runs outside the target graph, so a FULL target can read as PIECEWISE.
  - The per-forward path is correct: in the 0.32 source, `"gpu_model_runner: forward"` and `"gpu_model_runner: draft"` are separate scopes.
  - **Fix:** add a `basis` note, or restrict the fallback to kernels inside `forward` ranges when they exist.
- **L10.** CELL6 cosmetics: `G3_patch_revert` records have no `status` and print as "None" rows in the step table.
- **L11 (input / config, not code).**
  - Kit v1 had `e23-merlin-env/Other/default/1` attached, and `merlin_offline_tar.bin` was not found (MERLIN_PAY = None), so every Merlin step was ERROR.
  - The kit now uses `Other/v1/1`. No GPU-notebook `kernel-metadata.json` exists yet; it must attach `.../e23-merlin-env/Other/v1/1` too.
- **L12 (measurement note).** Memory note E19 records that MTP with prefix caching OFF flips tokens.
  - `s7_eagle` runs spec decode with `prefix=False`.
  - An EAGLE `vs_fp8_kvauto_mtp0` FAIL may therefore be that known engine issue, not EAGLE itself.
- **L13 (time).** FULL step estimates add up to 680 of the 690 min.
  - The Merlin install and the cold checkpoint loads happen inside the steps but are not part of the estimates.
  - Any overrun silently drops the tail first (`int4_mtp3_s20_control`, then both PP4 arms) as SKIPPED_BUDGET. This is by design, but there is no slack.

---

## Paths the fake runs never exercised (mentally executed with real responses)

| Path | Result |
|---|---|
| `merlin_install` → `import launch` → `merlin_server` | I ran Merlin's `launch.build()` (payload source) with the notebook's arguments. `argv()` = `[venv/bin/vllm, "serve", ckpt, *args, *extra]`, with `"serve"` at index 1, so the `help_check` prefix is right. Profile flags come first and our extras last, so `--served-model-name q27`, `--max-model-len 131072`, `--port`, `--max-num-seqs` and `--compilation-config` win (argparse last-wins). `distributed_ports()` returns a tuple, so the `zip` branch works: TP1 ports 8100-8103 map to MASTER 29600-29603 and VLLM_PORT 40000-40300. The env has `CUDA_VISIBLE_DEVICES = GPUS`. The install script prints MERLIN_VENV / TREE / SHIM, which match the `MERLIN_[A-Z]+` regex. The payload tar has 231 regular members and 0 symlinks, so `extractall(filter="data")` is safe. Our capture JSON copies Merlin's B27_INDUCTOR exactly. OK. |
| `_patch` (G3) | The extracted flashinfer hunk dry-runs cleanly against stock v0.32.0 `flashinfer.py` (13/13 hunks, offsets only). OK. L5 applies to the revert check. |
| `profile()` / `profiled_load()` | In the 0.32 source, `/start_profile` takes `StartProfileRequest{delay_iterations, max_iterations}` (extra="ignore"), so the bounded mode works. `ProfilerConfig` has `torch_profiler_dir`, `torch_profiler_with_stack` and `ignore_frontend`. A `/stop_profile` after the automatic stop is ignored without error. OK. |
| `trace_summary()` | Parsed a real torch trace (E14 TP-0: 31.8k kernels, 40 graph launches) in 3 s with no crash. Verdict PIECEWISE, basis cupti_correlation. The vLLM 0.32 file name `dp0_pp0_tp0_dcp0_ep0_rank0.<pid>.<ts>.pt.trace.json.gz` matches the `rank0\.` regex. OK (see L9). |
| `teacher_forced()` | **H1.** The response parsing is right: `prompt_logprobs[pos]` is keyed by `str(token_id)`, and rank 1 is top-1. |
| `chat_probes()` on a real server | `message.content` / `message.reasoning` are read correctly. The fake has no chat route (404), so probes are always False in fakes. As a result the **PP4 steady path never ran** in any fake run (the NO_GO gate fired first). It is the same `levels()` code. |
| TP1 / TP4 `LAUNCH_FAILED` fallbacks, `s7_eagle`, PP4 windows, P4 / P5 on GPU, real `wait_gpu_release` | Reviewed by reading. The issues are M3 and L1. The `nvidia-smi` CSV parsing (`isdigit`, uuid→index map) handles empty and `[N/A]` output. |

## Verified correct
- **Closures and threads:** `s1_rep(n)`, `s6` / `s3` / `s5` factories, the A/A loop arguments and the `steady_multi` threads (`args=(i, j)`) bind their values. No late-binding bug.
- **`Stream`:**
  - the `_halt` versus `err` semantics;
  - token-id validation;
  - `[DONE]` plus `finish_reason` required;
  - `finally` closes the response;
  - `stop()` before `_resp` exists is harmless in practice (vLLM sends headers before it schedules the request).
- **`steady()`:**
  - the status ladder;
  - every division is guarded (`dt`, `rm`, `drf`, `dtk`, `steps`, `kvs`, `gmem`);
  - the `errs` snapshot is taken before `stop()`;
  - `finally` mutates the returned dict.
- **`with_servers`:** the `finally` block (kill, then GPU-release gate on the servers' own GPUs, then ABORT) changes the dict it returns.
- **Process lifecycle:**
  - `kill()` sends SIGTERM, then always SIGKILL to the whole group;
  - `port_listening` treats TIME_WAIT as free;
  - the watchdog's `list(LIVE)` is atomic under the GIL.
- **Prompts:** the MAX_NEW headroom and the 116k band. At real decode rates the earliest 120k stream does not reach 14,560 new tokens before the window ends. The fake ends early only because it runs 300 tok/s per stream.
- **Serialization:** results are always JSON-serializable (`default=str`, NaN dropped in `prom()`). The notebooks are 103 kB, under the 400 kB commit limit.
- **Builder:**
  - the three variants differ only in the two baked lines (diffed);
  - all cells compile;
  - the kit's `bake()` assertion holds;
  - the kit v1 FULL `tmp_FULL` crash and the missing-tmp-dir race are fixed in the current kit builder.
