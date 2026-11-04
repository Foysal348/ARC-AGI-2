# E23 rev 3d: new-bug audit D (independent, real-GPU walk) — 2026-10-30

Scope: the changes from rev 3c to rev 3d. The baseline is `build_e23_notebook.pre3d.py`, which is byte-identical to git HEAD. For the lab, the baseline is git HEAD: `e23_lab.pre3d.py` is identical to the CURRENT lab, so that copy was taken after the edit. Line numbers are rev 3d: B = `Experiment Notebooks/Build/build_e23_notebook.py`, L = `src/e23_lab.py`.

Mechanical checks:
- Every code cell of SMOKE_A, SMOKE_B and FULL compiles (6 cells each).
- `gate_tests.py` against freshly extracted FULL code and the current lab gives `all_ok True`, 0 failed, 0 missing functions. This was run on local Py 3.14; the kit runs it on 3.12.

## Findings

### 1. HIGH: a Xid / illegal-access taint blocks the whole FULL A/A fill, and the expected crash arms trigger it
- **Where:** B:423 (`XID` regex), B:610-618 (`taint_check`), B:1108 (pp4), B:1251-1254 (`_done`).
- **Scenario:** in FULL, G1/G2 (TRITON FULL graphs + MTP3 on the hybrid GDN model), G3 (patched FlashInfer) and compiled PP4 (#63505: illegal-memory faults) are exactly the arms most likely to die with `CUDA error: an illegal memory access was encountered`.
  - A body-time crash is SERVER_DIED, and `with_servers` then records `xid=True`.
  - `taint_check` sets the global `tainted_xid`.
  - `_done()` then returns False for every config, so `8_aa_repeats` runs nothing.
  - FULL ends with hours of the 690-min budget unused and no noise control for the main comparisons. A G-arm crash also skips compiled PP4.
- **Why it is wrong:** "illegal memory access" is a per-process CUDA context error. It is not a device Xid that persists, and a container cannot see dmesg Xids anyway. Fix list #5c said "later steps are still run but flagged", not "skipped".
- **Fix (pick one):**
  - (a) Let `_done()` ignore the taint and only tag A/A records `tainted_session`.
  - (b) After a taint, run a short health check: relaunch `fp8_mtp0`, run its oracle, and compare with `ROWS["fp8_mtp0"]`. Clear the taint when the result is EXACT/NEAR_TIE.
  - (c) At minimum, run `f_pp4_int4_compiled` after the A/A fill.
  - Also restrict `XID` to `\bXid\b` plus "unspecified launch failure". Keep "illegal memory access" as a per-arm `SERVER_DIED_CUDA_FAULT`.

### 2. HIGH (design risk): the NVFP4 teacher-forced screen is uncalibrated, so NVFP4 + MTP3 will most likely never run
- **Where:** B:997-1009 (`nvfp4_gate`), B:1239-1245, L:507-523.
- **What the A/A gives:** the TF reference and the A/A are two launches with identical settings, one request each, the same 512-token chunks and NCCL all-reduce. On a deterministic engine both runs give identical numbers: 0.0 deltas and rank-1 agreement 1.0. The limits then collapse to fixed floors:
  - rank-1 ≥ 0.99, which allows at most 2 flips of 256 (253/256 = 0.988 fails);
  - mean |dlogprob| ≤ 0.05;
  - max |dlogprob| ≤ 1.0.

  So the A/A calibrates nothing; it only proves determinism.
- **Expected drift (reasoned estimate, not measured):**
  - Error sources: NVFP4 KV (E2M1 values, a block scale per 16 elements) has about 8-10 % relative RMS error per element; FP8 E4M3 has about 2-3 %. Qwen3.8 has 16 full-attention layers, and the context is 40k / 90k.
  - Resulting noise: plausible final-logit noise of σ ≈ 0.1-0.3 nat.
  - Mean: mean |Δ| over the last 256 prompt positions (mostly ARC grid digits) ≈ 0.03-0.1. That is borderline against 0.05.
  - Max: max |Δ| over 256 positions ≈ 3σ to 4σ ≈ 0.5-1.2, plus heavy tails at low-probability tokens. That is borderline-to-failing against 1.0.
  - Rank-1: digit continuations have many near-ties (margin < 0.2 nat at perhaps 5-10 % of positions), which suggests 2-5 % rank-1 flips, so agreement ≈ 0.95-0.98 < 0.99. Likely FAIL.
  - Our own production FP8-KV config would probably fail the same screen if compared against BF16 KV.
- **Consequence:**
  - `nvfp4_mtp3` (85 min) is SKIPPED_CORRECTNESS_GATE.
  - The NVFP4 A/A reps are skipped too.
  - The section-5 residency and streams census is never measured.
  - The SMOKEs never run `nvfp4_gate`, so FULL is the first time this policy meets real numbers.
- **Metric note:** L:521 counts agreement on "is the actual token rank 1", not argmax-token agreement. A position where both sides have a different rank-1 token but the actual token at rank 2 counts as agreement.
- **Fix:**
  - Add a known-acceptable lossy calibration arm: a TF launch with `kv="auto"` (BF16 KV), using the same short-launch settings, compared with `fp8kv_tf`. Then define the limits relative to the FP8-vs-BF16 drift, e.g.:
    - NVFP4-vs-FP8 mean ≤ 3 × the FP8-vs-BF16 mean;
    - p99 instead of max;
    - rank-1 counted only at positions where the reference margin ≥ `FAIL_MARGIN` (0.5), mirroring the greedy oracle;
    - the greedy oracle clean.
  - Alternatively, run the census labelled `MEASURED_OUTSIDE_SCREEN` instead of skipping it.
  - Fix the docstring at B:987: the code floors rank-1 at 0.98 (B:1003), not 0.99.

### 3. MEDIUM: the Merlin full-sha gate covers TP1 only; TP4 and the replicas run on an unverified (or known-bad) checkpoint
- **Where:** B:751-754 (only `s1_tp1` calls `merlin_sha_gate`); B:773-792 (`s1_rep`) and B:795-817 (`s1_tp4`) do not.
- **Scenario:**
  - SMOKE_A: a sha mismatch or join timeout gives `a_merlin_tp1_gpu0 = FAIL_CHECKPOINT_SHA`. `d_merlin_tp4` and `c_merlin_tp1_x4` still launch Merlin on the same files and report throughput.
  - FULL: the same happens with b, c, d and the A/A reps.
  - On a timeout, the sha thread keeps reading while TP4 loads, so the two contend for I/O.
- **Fix:** call `g = merlin_sha_gate(); if g: return g` at the top of `s1_rep.f` and `s1_tp4` (or inside `merlin_run`). Make the gate return `INCONCLUSIVE_SHA_TIMEOUT` when `MERLIN_SHA["ok"] is None` (B:1183-1184 currently labels a timeout FAIL_CHECKPOINT_SHA with `done: False`).

### 4. MEDIUM: the sha gate serialises prefetch + sha + Merlin install
- **Where:** B:752 (the gate runs before `merlin_run` → `merlin_server` → `merlin_install`), B:1188-1201.
- **Scenario (real numbers):**
  - The checkpoint is 17.05 GB, and `model.safetensors` alone is 16.8 GB. `L.prefetch` gives each file one reader thread, so the 16.8 GB file is read by a single stream from GCSFUSE. At 50-200 MB/s that takes 85-340 s, and longer during the 10-21 slowdown.
  - Then `_merlin_sha` hashes it single-threaded. With no SHA-NI on the L4 hosts' Xeons, about 400 MB/s gives roughly 45 s from page cache. In FULL, the concurrent 28-30 GB FP8 prefetch can evict it, and the hash then rereads from GCSFUSE.
  - Only then does `merlin_install` start: tar extract plus uv offline install, several minutes of CPU and /tmp work.
  - In rev 3c, install and server load overlapped with the prefetch. Now the delay is about 3-10 min in SMOKE_A (120-min budget), and more in FULL.
- **Fix:** in `s1_tp1`, call `merlin_install()` before `merlin_sha_gate()`; the install does not read the checkpoint. Optionally, hash inside the prefetch reader (one pass), as fix list #14 suggested.

### 5. MEDIUM: the FULL estimates now exceed the 690-min budget, and PP4 is skipped by the budget guard
- **Where:** B:1212-1249, B:50.
- **Arithmetic:** sum of FULL `est_min` = 702 min:
  - A block: 30+55+25+25+70 = 205;
  - B block: 45+12+40+30+70+30+30+30+15+15+30+15+85+25+25 = 497.
  - Rev 3c was 675; rev 3d adds `int4_mtp0` (12) and the TF A/A (15).
- **Effect:** with about 5 min of install, `e_pp4_int4_eager` starts at about 662 min with 28 min left. It needs est 25 + 5 = 30, so it gets SKIPPED_BUDGET, and compiled PP4 goes with it. This holds unless `nvfp4_mtp3` is gate-skipped (likely, see #2) or steps beat their estimates.
- **Fix:** either move PP4 before `nvfp4_mtp3`, or trim the FULL estimates or levels so that the sum plus install ≤ 680.
- **SMOKE_B (LOW):** the sum is 189 + install against 200. The realistic SMOKE G0 is about 28 min against est 20: the 40k@16 admission of 640k tokens at about 700 tok/s is about 15 min. So compiled PP4 is the likely casualty there too.

### 6. MEDIUM: the G0 control cascade is stricter than specified, and the G0 settle is newly fatal
- **Where:** B:949-952, L:840 and L:869-871.
- **Problem:** fix list #6d asked for "G0 reads FULL → INCONCLUSIVE_GRAPH". The code turns every G0 status other than PROVEN_PIECEWISE into INCONCLUSIVE_GRAPH for G1, G2 and G3. That includes G0 NOT_SETTLED, a missing trace, SKIPPED_BUDGET and ERROR. The arm's own PROVEN_FULL survives only under `graph_proof.own`.
- **Settle scenario (FULL `s6(3)`):**
  - The 90k/120k levels run before the profile and can evict the 40k prefixes.
  - `profiled_load` at 40k@16 must then re-prefill up to 640k tokens. The settle limit is `120 + 16*40000/700` = 1034 s, against about 914 s of prefill at exactly 700 tok/s.
  - A prefill 12 % slower than 700 tok/s now returns INCONCLUSIVE_PREFILL_NOT_SETTLED, which voids every G verdict. In rev 3c the same case was only a note.
  - Mitigation: the 120k/90k prompts of the same variant share the 40k prefix, so hybrid prefix-cache hits are likely but not guaranteed.
- **Fix:**
  - Profile on the 8k band (`PROMPTS["8k"]`, 16 streams). The batch shape is the same (16 × 4 verify tokens), so it proves the same graph mode, and it settles in a few minutes.
  - Or use `settle = 120 + 1.5 * level * plen / prefill_tok_s`.
  - Label an uncontrolled run as `UNCONTROLLED_<own status>` and keep INCONCLUSIVE_GRAPH for G0 = FULL / MISMATCH only.

### 7. MEDIUM: EAGLE tries — a CuTe-DSL runtime crash gets no fallback, and the third try is unbudgeted
- **Where:** B:1137-1154.
- **Fallback gap:** the next variant is tried only on LAUNCH_FAILED. Suppose `--linear-backend flashinfer_cutedsl` passes argparse and startup (the sm_89 kernel is missing or JIT-compiled lazily) and then dies on the first prefill or decode in `chat_probes`. That is SERVER_DIED, so the loop returns it, and the default-linear-backend EAGLE, the variant that would work on sm_89, never runs.
- **Mislabel risk:** if the fork silently falls back on sm_89, the result is labelled `flashinfer_cutedsl` without proof.
- **Unchecked flag:** `help_check` (B:352) records missing flags but nothing acts on them.
- **Budget:** each try may take up to `min(2400, left-120)`. Three tries against an est of 26 min can eat the rest of SMOKE_B, and PP4 then gets SKIPPED_BUDGET.
- **Fix:**
  - Fall through on `status in ("LAUNCH_FAILED", "SERVER_DIED")` or a NO_GO_ORACLE* for the cutedsl variant.
  - Skip the cutedsl launch when `HELP[name]["missing"]` contains `--linear-backend`.
  - Record the backend actually selected from the log.
  - Give the cutedsl try a 900 s startup cap and re-check `left()` before each try.
  - Label "all failed" by cause, not always SKIPPED_UNSUPPORTED_SM89.

### 8. LOW: the downgrade check can mask a real FULL verify result
- **Where:** B:870-872, L:146.
- **Detail:** the regex matches vLLM's "...is not supported with spec-decode ...; setting cudagraph_mode=X", and through re.I also "Setting cudagraph_mode=NONE". A FULL_AND_PIECEWISE → FULL_DECODE_ONLY downgrade still gives FULL verify graphs, yet it is reported DOWNGRADED before the verdict is looked at.
- **Fix:** parse X. Report DOWNGRADED only when X ∈ {NONE, PIECEWISE}; otherwise use the verdict, with the line attached.

### 9. LOW: a `_merlin_sha` save() race can flip a good sha to FAIL
- **Where:** B:1174-1176, B:1196-1199.
- **Detail:** `MERLIN_SHA.update(ok=True)` happens before `save()`. If `save()` raises, for example "dictionary changed size during iteration" because the FP8/INT4 prefetch thread writes `R["setup"]` outside the lock while the pure-Python indent encoder iterates, then `_after`'s except overwrites ok with False.
- **Fix:** wrap the save in `_merlin_sha` in its own try/except.

### 10. LOW: a P4 crash or timeout is labelled SYNTHETIC_WEIGHTS_NOT_ALLOWED
- **Where:** B:673-677.
- **Detail:** a child that wrote no JSON, or timed out, has no `weights_source`, so `kind != "real"` and the run reads as "synthetic". The real reason is hidden.
- **Fix:** test `kind == "synthetic"`. Treat a missing kind through the rc / `all_ok` path (CHECK_FAILED / BENCH_ERROR with `child_tail`).
- Verified: the FP8 dir has `model.safetensors.index.json` (CPU-kit identity), so `--weights real` resolves.

### 11. LOW: the kit-hash hard stop also fires in SMOKE_B, which never uses P4/P5
- **Where:** B:126-131.
- **Fix:** apply `_need` only when `MODE == "FULL" or PART in ("A", "ALL")`. Record it, without raising, for SMOKE_B.

### 12. LOW: Merlin steps are never taint-checked, and an unclean 90k reference is stored
- **Where:**
  - B:751-817: no `taint_check` around `merlin_run`. A Merlin CUDA fault does not taint the later stock steps, which is inconsistent with #1's policy, whichever way that is fixed.
  - B:739: `ROWS["merlin_90k"]` is stored even when errored, unlike `oracle()`'s new clean rule (fail-closed downstream, so only a consistency issue).
- **Fix:** wrap the Merlin results in `taint_check`, and store `merlin_90k` only when the row is clean.

### 13. LOW: `gate_tests.py` tests fewer behaviours than it claims
- **Behaviour-tested:** `oracle_gate`, `gated`, `compare_tf`, `nvfp4_gate`, `graph_proof`, `trace_summary`, `s2`, `s4_thread`, `_merlin_sha`.
- **Static string checks only:**
  - PP4 order and reference;
  - the int4 ref order;
  - the preflight / kit-hash `raise` text (not the conditions);
  - the EAGLE order.
- **Not tested at all (fix list #11 asked for these):**
  - "short-probe failure → zero 2k/40k requests", and the 2k → 8k staging;
  - "Xid line in a fake log → TAINTED_XID" (`with_servers`/`taint_check`);
  - "3-GPU mock → no P5";
  - "one changed .so byte → STOP" (the condition);
  - the `s5_tf` status rule;
  - the `s5(0)` early NO_GO;
  - `profiled_load`'s new statuses (NO_SPEC / PREFILL_IN_PROFILE);
  - the s3 G0 cascade;
  - `merlin_sha_gate` timeout and thread exception;
  - the EAGLE SERVER_DIED path (#7).
- **Fix:** add stubbed behaviour tests for these, particularly #1, #3, #6 and #7 above.

## Verified as fine
- **Merlin prefetch thread throws:**
  - `L.prefetch` swallows per-file errors.
  - If `prefetch_bg.go` raises, the thread just ends; `_after` still joins and hashes.
  - `_merlin_sha` exceptions (missing file, bad manifest) are caught and give ok=False.
  - The manifest is `[size, sha]` for all 15 files, so no file is skipped silently.
  - The join timeout `max(60, min(1800, left-600))` gives 1800 s in SMOKE_A, which is sane.
- **TF on real vLLM:**
  - 90k at `--max-num-batched-tokens 512` is about 176 chunks. Each chunk's logits (512 × ~248k fp32) are about 0.5 GB, which fits at mem 0.80.
  - Expected time is about 3-6 min per 90k request, well under the 1800 s timeout. SMOKE runs only 40k.
  - V1 skips the prefix cache for prompt-logprob requests, so the numbers are not contaminated.
  - The actual prompt token is always in the prompt-logprob dict, keyed `str(id)`. `pl` has length `len(ids)`, so all 256 positions are valid.
  - INCONCLUSIVE_TF therefore comes only from a real error or timeout, which correctly blocks NVFP4 + MTP3.
- **`compare_tf` completeness:** 1/256 gives INCONCLUSIVE_INCOMPLETE_TF, and a length mismatch is caught.
- **`profiled_load` with spec=True:**
  - `vllm:spec_decode_num_drafts_total` exists on 0.32 (seen in E18 R2 `e18_engine_stats.json`).
  - The m0 → m1 window spans the 6 s profile, stop_profile, the 20 s sleep and the trace parse, with 16 MTP3 streams decoding (max_new 90,560 at 40k, so none end), so the drafts delta is > 0.
  - In V1, prompt_tokens counts only at first prefill completion, so it stays flat.
  - No off-by-one: `ok >= 3` is tested inside the loop, and `settled` is computed after it.
  - The early return runs the `finally` (stream stop + `wait_idle`).
  - A missing `status` key means OK, and `graph_proof` treats it so.
- **G0 PIECEWISE verdict:** PIECEWISE with GDN + attention as splitting ops gives about 65 graph launches per target forward (median > 1), so PIECEWISE is the likely and correct verdict. The drafter runs outside the "gpu_model_runner: forward" scope. The verdict does depend on that scope name existing in 0.32: if it is absent, the verdict is UNPROVEN, which cascades through #6. The trace is kept in `OUT/traces` (< 400 MB) for offline re-analysis.
- **`Server.facts()`:** it does produce `graph_downgrade_lines`, from the spec-decode / "setting cudagraph_mode=" lines. `graph_proof` reads them from `o["servers"]`.
- **`oracle()`:** `errors` / `garbage` are None for clean rows, so `any()` is correct. An errored row shows both an error and EMPTY garbage, and the error is checked first.
  - Reference rows are stored only when clean.
  - MTP3 vs MTP0 greedy (same weights, prefix ON) differ only numerically. The flips are NEAR_TIE: the 0.5-nat threshold is far above the ~0.01-0.05 nat noise. E19 found S1 MTP vs MTP-off PASS with prefix ON; its high-margin flips were MTP + prefix OFF, which is not used in any rev 3d stock arm except the labelled EAGLE noprefix fallback.
- **PP4 staged:**
  - Indices [0], [2], [3] map to 2k v0, 8k v2 and 40k v3 in both `ORACLE_P + LONG_P` and `ROWS["int4_mtp0"]`.
  - The same `NTOK` and logprobs=2 are used.
  - `L.compare` on 1-row lists is valid.
  - A short-probe failure returns before any greedy.
  - Eager comes first; compiled runs only on eager DONE.
- **CELL1:**
  - The manifest keys are exactly `p4_sm89_fp8.so`, `p5_host_ar.so`, `bench_fp8_gemm.py` and `bench_host_ar.py`, so there is no name mismatch.
  - `gpu_mem()` parses `index, used, total`; `int()` tolerates the spaces. A "[N/A]" row raises, gives None, and then a hard stop, which is correct.
  - On 4×L4, 4 rows are returned.
- **`s2`:** `bench_host_ar.py` writes `summary.all_checks_pass`. Release False/None gives FAIL_GPU_RELEASE + ABORT.
- **`s4_thread`:** `bench_fp8_gemm.py` writes `all_ok` and per-shape `weights_source.kind`.
- **`nvfp4_gate` plumbing:** the status names match the step names, and the A/A has `key=None` (no TF overwrite).
- **CELL6:** every new field is read with `.get` and a default. Records without a status (patch_revert, p4_bench_gpu3) and a profile without a trace render as "-"/None. Nothing can raise.
- **Gate tests:** extracting the FULL functions by ast and exec-ing them works with the 4-space RERUN indent. The kit adds `gates.all_ok` to the p1 verdict.
