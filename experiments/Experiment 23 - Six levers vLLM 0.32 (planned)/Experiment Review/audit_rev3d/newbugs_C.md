# E23 rev 3c -> rev 3d: new-bug audit (reviewer C)

Scope: the new and changed code in `build_e23_notebook.py` (CELL1-CELL6), `src/e23_lab.py` (the lab diff is against git HEAD,
because `_scratch/e23/e23_lab.pre3d.py` is byte-identical to the current lab: it was copied after the lab edits),
`src/harness/gate_tests.py`, `fake_vllm_server.py` and `build_cpu_kit_e23.py`. I changed no project files.

## What I ran (all under `_scratch/e23/audit_rev3d/work_C/`)
- **Builds.** I built the three notebooks with a copy of the builder (`Build/build_copy.py`; only HERE and E23 were patched). All code cells compile. The notebooks are 149 kB each, under the 400 kB limit.
- **pyflakes.** I ran pyflakes on the joined FULL code and on the lab. It reports 0 findings, so there are no undefined names: `threading`, `hashlib`, `XID`, `NVSMI_OK`, `EST_SCALE`, `COMPILE_ERR`, `TF` and `ROWS` are all defined before use. Every `L.<attr>` used by the notebook exists in the lab.
- **gate_tests.py.** I ran it against the freshly extracted FULL code: `all_ok True`, 0 failed, 0 missing functions. The ast extraction works inside the `if not RERUN:` blocks.
- **CPU kit builder.** A copy (`Build/kit_copy.py`) builds p1-p5, and the p1 cell compiles.
- **Fake SMOKE_B.** I ran it on shifted ports (8160/8260/8360): rc 0, 0 tracebacks, 4.7 min. CELL6 renders the new `correctness` and `graph proof` columns and the notes.
- **Fake FULL.** I ran it on ports 8180/8280/8380 with budget 6000 s, a 10 s window and EST_SCALE 0.15. The result is at the end of this file.
- **Input copy.** I needed a copy of the fake input with a real kit manifest (see F9).
- **Fake-server limits.** The fake server has no `/v1/chat/completions`, no `prompt_logprobs` and no profiler endpoints. In the fake runs, probes are always False, every TF step is INCONCLUSIVE_TF and every graph proof is INCONCLUSIVE_GRAPH. The happy paths of `nvfp4_gate`, `graph_proof` and PP4 are only covered by gate_tests.

## Checked and found correct (no finding)
- **Dict keys match the producers:**
  - `graph_downgrade_lines` matches `Server.facts()`.
  - `summary.all_checks_pass` matches `bench_host_ar.summarize()`.
  - `shapes[*].weights_source.kind` and `all_ok` match `bench_fp8_gemm.orchestrate()`; `--weights real` is a valid choice.
  - The Merlin manifest is `files: {name: [size, sha]}`, which matches both CELL1 and `_merlin_sha`.
  - The kit manifest has the 4 names in `_need`.
  - The lab constants PROMPT/GEN/DRF/ACC are used inside the lab only.
- **PP4 with `DO_A` false.** `step()` and the `else` branch both use `setdefault`, so `R["sections"]["1_tp1x4_pp4"]` is created. `_tp1_level` indexes the section directly, but only s1_rep calls it, and s1_rep only runs after `a_merlin_tp1`.
- **Merlin sha thread.** `MERLIN_SHA["th"]` is set only for a thread that is started. The FAKE path returns None. The join timeout `max(60, min(1800, left()-600))` is sane.
- **G0 PROVEN_PIECEWISE can come true.** In vLLM 0.32, `gpu_model_runner: forward` wraps only `_model_forward`, and the drafter runs in a separate `gpu_model_runner: draft` scope (checked in `_scratch/vllm_src/v0.32.0/gpu_model_runner.py:4201/4364`). A PIECEWISE forward therefore gives per-forward graph launches > 1. `spec=True` is satisfied by MTP3 drafts. The fake run shows `window_deltas.drafts > 0` and `prompt_tokens 0`.
- **Status precedence.** `gated` sets a status only on None/DONE. After the body, `with_servers` raises the status to SERVER_DIED, and in `finally` to FAIL_GPU_RELEASE. `step()` maps DONE/SERVER_DIED/LAUNCH_FAILED/BODY_ERROR to PARTIAL_HARD_STOP only under ABORT. This is consistent.
- **CELL6 does not crash on the new fields.** I checked this in the fake SMOKE_B run.
- **Earlier fixes still in place:** port connect-check, live records, `levels` budget guard, the A/A loop structure and the patch revert.

---

## Findings

### F1 HIGH: the Xid taint is session-global and is triggered by the arms that are expected to fault. It cancels PP4-compiled and the whole FULL A/A fill.
**Where:** build_e23_notebook.py:610-618 (`taint_check`), :1108 (`pp4`), :1254 (`_done`), :485 (`xid` set only after the body).

**Scenario 1 (FULL).**
1. In FULL the order is `... nvfp4_mtp3 -> pp4() -> A/A fill`.
2. The arm `f_pp4_int4_compiled` exists to test #63505, a known illegal-memory-access / gibberish failure on the hybrid GDN model.
3. If it logs `illegal memory access`, `R["setup"]["tainted_xid"]` is set.
4. From then on, `_done()` is False for every main configuration, so **every** A/A repeat is skipped. That is all the spare time of the 690-minute run, and the one place rev 3d still gets repeat evidence.

**Scenario 2 (SMOKE_B and FULL).** The same happens in SMOKE_B and FULL after the experimental G3 (#60532 patched FlashInfer) arm crashes after ready. PP4-compiled is then skipped and labelled `SKIPPED_EAGER_NOT_PROVEN`, even when eager is DONE (mislabel).

I reproduced this: `taint_check({"servers": {"int4_pp4_compiled": {"xid": True}}})` sets `tainted_xid` globally.

**The rule is also inconsistent:**
- (a) A crash during startup (LAUNCH_FAILED returns before `xid` is computed) never taints. The same crash after ready does.
- (b) Merlin steps (`merlin_run`, s1_tp1/rep/tp4), P4 and P5 are never taint-checked.
- (c) The docstring says tainted steps "never count as correctness evidence". Yet `nvfp4_gate`, `oracle()` reference storage and `s1_pp4` all accept evidence recorded after a taint.

**Fix:**
- Compute the taint per arm, not per session.
  - Record `xid` in the LAUNCH_FAILED branch as well (move the `xid=` assignment into the `if not ok:` block too).
  - After any Xid hit, run a cheap health probe on the affected GPUs (`nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total,pstate`, plus a 1 s torch matmul per GPU in a subprocess). Set `R["setup"]["tainted_xid"]` only when the probe fails.
- Exempt the expected-fault arms (`f_pp4_int4_compiled`, `G3_*`, `eagle4_*`) from session tainting; record `fault_observed` on the arm itself.
- Use a dedicated status for a taint skip: in `pp4()`, use `SKIPPED_TAINTED_SESSION` when `rec["status"] == "DONE"`.
- If a global taint is kept, apply it consistently: in `nvfp4_gate` (`if R["setup"].get("tainted_xid"): return False, {..., "why": "TAINTED_SESSION"}`) and in the reference store of `oracle()`.

### F2 HIGH: `gated()` ignores garbage under load, so an arm whose decode output turns to garbage under load is DONE + CLEAN
**Where:** build_e23_notebook.py:586-607 (`oracle_gate` / `gated`).

**Scenario.**
1. `oracle_gate` looks only at `o["oracle"]` and the `extra` comparisons.
2. A steady window that ends `FAIL_GARBAGE_UNDER_LOAD` does not change `status` or `correctness`. That window is the lab's check on all streams under load, and exactly the #63505 "gibberish under PP" symptom.
3. Failed chat probes (`probes.ok False`) do not change them either.

I reproduced this: `gated({"oracle": clean, "probes": {"ok": False}, "steady": {"40k@16": {"status": "FAIL_GARBAGE_UNDER_LOAD"}}}, need=("fp8_mtp0",))` gives `status None -> DONE, correctness CLEAN`.

**Consequences:**
- `pp4()` starts the compiled arm after an eager arm that produced garbage under load.
- `_done()` lets the A/A reps of a garbage-producing mtp3, nvfp4_mtp3 or Merlin-x4 configuration run.
- The CELL6 report shows CLEAN for them.

`s1_pp4` sets `correctness CLEAN` before its steady windows and returns without a status. So the same gap applies even though it does not use `gated`.

**Fix:** in `oracle_gate`, after the oracle checks, scan the load windows.
```python
wins = list((o.get("steady") or {}).values())
for k in ("40k", "90k"):
    wins += [w for w in ((o.get(k) or {}).get("replicas") or []) if w]
if any((w or {}).get("status") == "FAIL_GARBAGE_UNDER_LOAD" for w in wins):
    return "NO_GO_GARBAGE_UNDER_LOAD"
if (o.get("probes") or {}).get("ok") is False:
    return "NO_GO_SHORT_PROBE"
```
- In `s1_pp4`, call `gated(o)` (or `oracle_gate`) after the steady block instead of the early `o["correctness"] = "CLEAN"`.
- Add both cases to gate_tests.

### F3 HIGH (policy and code): the NVFP4 tolerance from an exact A/A collapses to fixed floors that 4-bit KV will very likely miss. The rank-1 clamp also ignores A/A noise.
**Where:** build_e23_notebook.py:1000-1005 (`nvfp4_gate`).

**What happens with an exact A/A.** The FP8-vs-FP8 A/A uses a fresh launch, a single request, the same chunking and teacher forcing, so it is expected to be bit-exact (mean 0, max 0, rank-1 1.0). The limits then become fixed:
- rank-1 agreement >= 0.99, that is at most 2 of 256 positions may differ;
- mean |dlogprob| <= 0.05;
- max |dlogprob| <= 1.0 nat over 256 positions;
- and this must hold at both 40k and 90k context.

I probed `nvfp4_gate` with an exact A/A:

| NVFP4 (rank-1, mean, max) | passes? |
|---|---|
| 0.99 / 0.03 / 0.8 | yes |
| 0.98 / 0.03 / 0.8 | no |
| 0.99 / 0.06 / 0.8 | no |
| 0.99 / 0.03 / 1.2 | no |
| 0.97 / 0.08 / 2.5 | no |

**Why this blocks the census.** The teacher-forced positions are ARC grid tokens: many are near-ties and many have low log-probabilities. A 4-bit E2M1 KV cache at 40k-90k context will plausibly exceed max 1.0 or 2/256 rank flips. That blocks `nvfp4_mtp3` (85 min, the #5 residency census + MTP3 and its A/A reps) as SKIPPED_CORRECTNESS_GATE. The gate turns a measurement in the user's plan into a precondition.

**Code bugs in the same lines:**
- **The rank-1 limit is not derived from A/A.** It is `clamp(AA - 0.01, 0.98, 0.99)`. With a noisy A/A of 0.95, NVFP4 must reach **0.98**, better than FP8 against itself. This contradicts the docstring ("A/A-derived"; ">= 0.99 and >= A/A - 0.01").
- **A falsy A/A value is treated as perfect.** `a.get("rank1_agreement") or 1.0` turns an A/A of 0.0 into 1.0. I probed it: an A/A rank-1 of 0.0 still gives the limit 0.99.

**Fix:**
- Separate the correctness verdict from the measurement. Block `nvfp4_mtp3` only on hard failures: garbage, request error, FAIL_HIGH_MARGIN, or TF incomplete. Run it otherwise, and record `correctness: "TF_OUTSIDE_TOLERANCE"` on it, with A/A `_done` requiring CLEAN as now.
- If a screen is wanted, use `rank1_min = min(0.99, aa_r1 - 0.01)` without the 0.98 floor, a p99 |dlogprob| instead of the max, and floors sized for 4-bit KV (for example mean <= 0.15, p99 <= 2.0). Pass the floors as constants that the user approves.
- Replace `(x or 1.0)` with an explicit `None` check.
- Add an exact-A/A plus realistic-NVFP4 case to gate_tests.

### F4 MEDIUM: the FULL plan is now over budget by its own estimates (702 vs 690 min), so PP4-compiled is skipped and the A/A fill is zero
**Where:** build_e23_notebook.py:1213-1249.

**The arithmetic.** Rev 3d adds `int4_mtp0_target_only_ref` (12 min), `tf_fp8kv_aa_control` (15 min) and makes PP4 eager + compiled always (25 + 25 min). The FULL estimates now sum to:
- 30 + 55 + 25 + 25 + 70 = 205
- + 45 + 12 + 40 + 30 + 70 = 402
- + 30 + 30 + 30 = 492
- + 15 + 15 + 30 + 15 = 567
- + 85 = 652
- + 25 + 25 = **702 min**

Rev 3c was about 650 min, with 25 min of PP4 in the normal path.

**Scenario at the estimates.** Eager starts at 652 min with 38 min left (needs 30), so it runs. Compiled then has 13 min left and needs 30 (25 + 5): `SKIPPED_BUDGET`. The A/A loop then gets nothing, while the user's rule is that spare time goes to A/A repeats.

**Fix:** budget the new steps explicitly. Either trim levels in steps that rev 3d did not touch (for example mtp1 to 40k [8,16,20], or G1/G2 40k [16] only, about -15 min), or cut the PP4 estimates to the staged-early-exit reality. Add an assert at build time that the FULL estimate sum is <= 690 - reserve.

### F5 MEDIUM: `merlin_sha_gate` gates TP1 only. Replicas and TP4 still run on a checkpoint whose sha failed or never finished. The gate also puts prefetch + sha on the TP1 critical path.
**Where:** build_e23_notebook.py:751-754 (s1_tp1), :773-817 (s1_rep and s1_tp4 have no gate), :1179-1185, :1188-1201.

**Scenario A (gate bypass).**
1. `_merlin_sha` reports `bad=["model.safetensors"]`, or the join times out with `ok None`.
2. `a_merlin_tp1` becomes FAIL_CHECKPOINT_SHA.
3. `b_merlin_tp1_x2`, `c_merlin_tp1_x4` and `d_merlin_tp4` (FULL, 120 min) and `c_merlin_tp1_x4` (SMOKE_A) still launch Merlin on the same bytes. They then burn GPU time producing INCONCLUSIVE_NO_ORACLE rows.

**Scenario B (time).**
1. `model.safetensors` is one 16.8 GB file (manifest total 17.05 GB). `prefetch()` gives one file per thread (`big[i::threads]`), so the warm-up is a single-stream read.
2. `_merlin_sha` then hashes it again in one thread. On a CPU without SHA-NI that is about 0.4-0.5 GB/s, so roughly 35-45 s from the page cache. If the 17 GB was evicted by the parallel FP8 prefetch (~30 GB) and P4's weight reads, it is a second cold read.
3. TP1 now waits for prefetch + sha. Rev 3c launched TP1 at once and overlapped its read with the prefetch.
4. At a normal mount speed (200-400 MB/s) the added delay is about 1-3 min. Under the 10-21 GCSFUSE slowdown (the run that took 1,197 s to load) the prefetch alone takes 12-20 min.
5. The join cap is 1800 s. Past it, TP1 is lost entirely with `FAIL_CHECKPOINT_SHA done False`, although the checkpoint is probably fine. The thread then keeps reading in the background.
6. SMOKE_A has little slack: 12 + 40 + 40 + 20 = 112 of 120 min. Any gate wait pushes `c_merlin_tp1_x4`, the 4-replica headline, to SKIPPED_BUDGET.

**Fix:**
- Call `merlin_sha_gate()` at the top of `s1_rep(...).f` and `s1_tp4` as well: one gate, cached result.
- Split the prefetch of a single huge file across threads with ranges (`os.pread` at offsets), or start the sha in the same pass as the prefetch (hash while reading, one read instead of two).
- On a join timeout, return `INCONCLUSIVE_SHA_PENDING` and let TP1 run with `identity_unverified: True` instead of discarding it. Re-check `MERLIN_SHA["ok"]` before storing `ROWS["merlin"]`.
- Add the expected wait to the TP1 estimate: `est += sha_est_min`.

### F6 MEDIUM: the P4 bench still starts after the new P5 release gate has aborted the lab
**Where:** build_e23_notebook.py:1213-1217.

**Scenario.**
1. `s2()` now sets `ABORT["why"]` when GPU release is False or unverified (rev 3d #7).
2. The next lines still run `start_prefetch()` (17 + 30 GB of reads) and `P4_FUT = submit(s4_thread)` with no ABORT check.
3. So P4 starts on GPU 3, which P5 may still hold. `a_merlin_tp1` is SKIPPED_ABORT, and `join_p4()` then blocks up to 60 min (`min(3600, left()-600)`) on a lab that has already aborted.

This path did not exist in 3c, where `s2` never set ABORT.

**Fix:** change :1216 to `if not ABORT["why"] and left() > ...:` and guard `start_prefetch()` with `if not ABORT["why"]`.

### F7 MEDIUM: the EAGLE try loop stops at the first non-LAUNCH_FAILED cutedsl result, and its runtime is unbounded by the step estimate
**Where:** build_e23_notebook.py:1137-1154.

**Scenario 1 (early stop).**
1. CuTe-DSL kernels may JIT on first use. A missing sm_89 kernel then appears as BODY_ERROR, SERVER_DIED, NO_GO_ORACLE_REQUEST_ERROR or garbage, not as LAUNCH_FAILED.
2. The loop returns that record (`if o.get("status") != "LAUNCH_FAILED": return o`).
3. The default linear backend, the arm most likely to work on L4, is never tried.

**Scenario 2 (time).**
1. Each try gets `with_servers(..., 2400, ...)`, and the cutedsl JIT can hang until startup timeout.
2. Three tries can take up to about 3 x 40 min, against a step estimate of 26 min.
3. In SMOKE_B, PP4 eager and compiled come after EAGLE and get SKIPPED_BUDGET (only about 11 min of slack remain; see F10).

**Scenario 3 (label).** If ABORT fires after the first try, the result says `SKIPPED_UNSUPPORTED_SM89` (because `attempts` is non-empty) instead of SKIPPED_ABORT.

**Fix:**
- Continue to the next try on `status in ("LAUNCH_FAILED", "SERVER_DIED", "BODY_ERROR")` or when `correctness` starts with `NO_GO`, but only for the cutedsl try.
- Bound the loop with an overall deadline: `t_end = time.time() + 26*60`, and per-try timeout `min(2400, t_end - time.time())`.
- Return `"SKIPPED_ABORT" if ABORT["why"] else ...`.

### F8 LOW-MEDIUM: the kernel-kit hash hard stop blocks SMOKE_B, which uses no kit file
**Where:** build_e23_notebook.py:126-131.

**Scenario.** SMOKE_B (stock vLLM + EAGLE + PP4; `DO_A` False) never runs P4 or P5. A re-uploaded kit dataset with a changed `.so` or bench makes it raise `KERNEL_KIT_HASH_FAILED` in CELL1 and lose the whole smoke.

**Fix:** hard-stop only when `MODE == "FULL" or PART in ("A", "ALL")`. Otherwise record the error in `R["errors"]`.

### F9 LOW (harness regression): the local fake harness can no longer pass CELL1
**Where:** build_e23_notebook.py:126-131, together with `_scratch/e23/nbtest/input/.../q27_e23_kernels_manifest.json`, which is `{}`.

**Scenario.** `run3c.sh` / the nbtest pattern now always dies in CELL1 with `KERNEL_KIT_HASH_FAILED` (all 4 names missing). I had to regenerate the manifest in my input copy (`work_C/input`). The Kaggle CPU kit uses the real mounts, so it is not affected.

**Fix:** regenerate the nbtest manifest from the stub files, or skip the hard stop when `FAKE` and the manifest is empty (record it instead).

### F10 LOW: the SMOKE_B slack shrank from 24 to 11 min
**Where:** CELL1 budget plus :1229-1249.

**The arithmetic.** SMOKE_B estimates are now 12 + 12 + 20 + 20 + 11 + 11 + 25 + 11 + 11 + 26 + 15 + 15 = 189 min, against a budget of 200. Rev 3c was 151 against 175.

**Scenario.** Any overrun before EAGLE (the INT4 reference is a cold INT4 load plus compile, with the INT4 prefetch now also running in SMOKE) pushes PP4-compiled to SKIPPED_BUDGET.

**Fix:** budget the new INT4 reference and A/A steps (budget 215), or trim the SMOKE levels.

### F11 LOW: `s4_thread` reports a crashed or timed-out shape as "synthetic weights"
**Where:** build_e23_notebook.py:673-677.

**Scenario.** A shape child that times out or crashes gets `{"errors": [...], "gate_pass": False}` with no `weights_source` (bench_fp8_gemm.py:571-576). `kind != "real"` is then True, so the step says `CHECK_FAILED / SYNTHETIC_WEIGHTS_NOT_ALLOWED`. A timeout or crash is misdiagnosed as a weights-policy violation. This is the normal outcome on the CPU kit, and it is also what happens on GPU if a real-weight load is slow.

**Fix:**
- `synth = [n for n, v in shapes.items() if ((v or {}).get("weights_source") or {}).get("kind") == "synthetic"]`
- `missing = [n for n, v in shapes.items() if not (v or {}).get("weights_source")]` -> `BENCH_ERROR, why="SHAPE_NO_RESULT"`.

### F12 LOW: a thread race can turn a successful Merlin sha into FAIL_CHECKPOINT_SHA
**Where:** build_e23_notebook.py:1174-1176 together with :1195-1199.

**Scenario.**
1. `_merlin_sha` sets `ok=True`, then calls `save()`.
2. `save()` uses `json.dumps(R, indent=1)`, which is the pure-Python encoder because indent is set. It iterates `dict.items()` live.
3. If another thread inserts a key while it iterates, for example the FP8 prefetch thread doing `R["setup"]["prefetch_fp8"] = ...`, `save()` raises `RuntimeError: dictionary changed size during iteration`.
4. `_after`'s `except` then overwrites the result with `ok=False, bad=["RuntimeError ..."]`.

The window is small (one dumps call, one insert). The same race can also crash the main thread's `save()` outside `step()`'s `try`.

**Fix:**
- Wrap `save()` inside `_merlin_sha` in `try/except` (the verdict must not depend on it).
- In `save()`, serialise under the lock with a retry: `for _ in range(3): try: s = json.dumps(R, ...); break; except RuntimeError: time.sleep(0.05)`.
- Make the background writers assign under `_SAVE_LOCK`.

### F13 LOW: "clean references only" is applied to `ROWS[key]` but not to the other references
**Where:** build_e23_notebook.py:739 (`ROWS["merlin_90k"] = g90` is stored unconditionally), :527 (`TF[key] = rows` is stored unconditionally), :976.

**Scenario.**
- An errored 90k TP1 row is stored. TP4 then compares against it and gets INCONCLUSIVE_ORACLE instead of INCONCLUSIVE_NO_ORACLE.
- `tf_fp8kv_reference` is DONE when `n == 256`, even if every position is `None`: a `prompt_logprobs` key mismatch yields n=256 with all positions None. The failure then shows up later, as INCONCLUSIVE_TF on the A/A, rather than on the reference step itself.

**Fix:**
- Store `merlin_90k` only when its rows are clean.
- In key mode, require `all(p is not None and p[0] is not None for p in row["positions"])`.

### F14 LOW: the graph-mode downgrade regex may miss the vLLM 0.32 wording
**Where:** e23_lab.py:146.

**What I could and could not check.** The regex matches the 0.19-era messages (`; setting cudagraph_mode=...`). In 0.32 the resolution moved into `CompilationConfig.resolve_cudagraph_mode_and_sizes`, which is not in the local source snapshot, so I could not verify its wording. 0.32's `vllm.py:1883/1893` uses `Overriding cudagraph_mode to PIECEWISE/FULL_DECODE_ONLY.`

**Impact:** limited. A real downgrade still shows up in the trace as MISMATCH_EXPECTED, but the explicit DOWNGRADED status would never fire.

**Fix:** extend the pattern with `|Overriding cudagraph_mode to|cudagraph_mode.*(downgrad|fall(ing)? back)`.

---

## Fake FULL run (ports 8180/8280/8380, budget 6000 s, window 10 s, EST_SCALE 0.15)
**Outcome:** 0 tracebacks up to the A/A fill. I stopped the run myself at 34.5 min, during `merlin_tp1_x4_rep1`, and killed the notebook and all 4 fake servers. No listener is left on 8160-8183, 8260-8283 or 8360-8383.

**Order and statuses:**
1. P5: BENCH_ERROR (no CUDA).
2. Merlin a/b/c/d: DONE/CLEAN, even with `probes.ok False`. This supports F2.
3. mtp0, int4_mtp0_ref, int4_mtp3, mtp1, G0, G1, G2, G3: DONE. Every G graph_proof is INCONCLUSIVE_GRAPH, because the fake server writes no trace.
4. tf_ref, tf_aa, tf_nvfp4: INCONCLUSIVE_TF (the fake server answers SSE, never prompt_logprobs).
5. nvfp4_mtp3: SKIPPED_CORRECTNESS_GATE (TEACHER_FORCED_INCOMPLETE).
6. PP4 eager: NO_GO_SHORT_PROBE; compiled: SKIPPED_EAGER_NOT_PROVEN.
7. A/A: `mtp3_control_rep1` DONE in 270 s, then `merlin_tp1_x4_rep1` started. The A/A loop works.

**F11 reproduced:** the P4 record is `CHECK_FAILED / SYNTHETIC_WEIGHTS_NOT_ALLOWED` with `weights_kind` None for all 5 shapes. The real cause in the tail is `RuntimeError: CUDA is not available`.

**CELL6:** I re-executed CELL6 on the live FULL `e23_results.json`. It renders the status/correctness/graph-proof table, the window table and the rev 3d notes without error.
