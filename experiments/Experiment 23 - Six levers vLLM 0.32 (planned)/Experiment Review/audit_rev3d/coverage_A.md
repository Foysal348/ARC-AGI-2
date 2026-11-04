# E23 rev 3d: coverage review A (were all 20 fixes from the four GPT audits implemented, and implemented correctly?)

Reviewer: independent, read-only. Date: 2026-10-30.
Spec: `Experiment Review/REV3C_GPT4_PRIORITY_FIX_LIST_2026-10-30.md` (20 items + 2 rejected), plus two later decisions: #13 keeps eagle.5, #20 licence accepted.

## 0. Baseline and method

- **Builder:** `_scratch/e23/build_e23_notebook.pre3d.py` is byte-identical to git HEAD (rev 3c), so it is a valid baseline. Diff to rev 3d: +295 / -69 lines. `patch_rev3d.py` + `cell5_rev3d.py` are the patch.
- **Lab:** `_scratch/e23/e23_lab.pre3d.py` is NOT the rev 3c lab. It is byte-identical to the current rev 3d `src/e23_lab.py`, because it was copied after the edit (mtime 08:26 vs 08:25). The real rev 3c lab is git HEAD (`git show HEAD:.../src/e23_lab.py`). The lab diff (+37 / -15) was taken against HEAD. Anyone diffing against the "pre3d" copy sees zero lab changes.
- **Generated notebooks:** I regenerated the cells in memory from the current builder. `Q27-XHIGH-E23_SIX_LEVERS_{SMOKE_A,SMOKE_B,FULL}.ipynb` match it cell for cell (6/6 code cells each). The `gpu_kernels/*` copies are byte-identical. Size is 152.7 kB, below the 400 kB commit limit.
- **CPU kit notebooks** (`cpu_kernel_e23_p1..p5`, 08:28) embed the rev 3d FULL code and `gate_tests.py`.
- **Gate tests:** a local run exists (`_scratch/e23/nbtest/gates_local.txt`, 08:28): 40 checks, `all_ok: true`. **No Kaggle CPU kit run of rev 3d exists.** The newest kit outputs (`kitout3_p*`, 05:34) are rev 3c.
- Line numbers: B = current `build_e23_notebook.py`, L = current `src/e23_lab.py`, G = `src/harness/gate_tests.py`, K = `build_cpu_kit_e23.py`.

## 1. Verdict table

| # | Item | Verdict | Gate test |
|---|---|---|---|
| 1a | compare_tf completeness | IMPLEMENTED | behavioural |
| 1b | s5(0) stops on any unclean oracle | IMPLEMENTED | indirect (oracle_gate only) |
| 1c | s5(3) / A/A only after a complete, clean gate | IMPLEMENTED (control flow verified) | nvfp4_gate behavioural; the s5(3) "no launch" path is not tested |
| 2 | FP8-vs-FP8 A/A tolerance | PARTIAL (code is looser than its own docstring and the 0.99 screen; floors dominate) | behavioural, no boundary case |
| 3 | Shared correctness gate + `_done()` | IMPLEMENTED for the listed arms; `_done()` is untested; probes are not gated | gated tested, `_done` not tested |
| 4 | INT4 target-only oracle | IMPLEMENTED (SMOKE_B order verified) | string checks only |
| 5a | PP4 staged fail-fast | IMPLEMENTED | string check only |
| 5b | PP4 eager first | IMPLEMENTED | string check only |
| 5c | Xid taint | PARTIAL (not on Merlin runs, not on LAUNCH_FAILED, not used by nvfp4_gate) | NOT TESTED |
| 6a | No annotation, no FULL | IMPLEMENTED | behavioural |
| 6b | Settle required | IMPLEMENTED | only as a pass-through in graph_proof |
| 6c | Spec decode active | IMPLEMENTED (drafts only, not accepted) | NOT TESTED |
| 6d | G0 negative control | IMPLEMENTED (stricter than asked) | the s3 propagation is not tested |
| 6e | Count verify forwards only; downgrade = INCONCLUSIVE_GRAPH | PARTIAL | partial |
| 7 | P5 release + result gate | PARTIAL (P4 still launches after FAIL_GPU_RELEASE) | behavioural |
| 8 | GPU preflight hard stop | IMPLEMENTED | string check only |
| 9 | Kit hash hard stop | IMPLEMENTED | string check only |
| 10 | P4 real weights only | IMPLEMENTED | behavioural |
| 11 | Fresh consolidated CPU verdict | PARTIAL (gates wired into p1; kit not re-run on Kaggle; BAD list not extended) | — |
| 12 | EAGLE cutedsl + fallback + label | PARTIAL (the SKIPPED_UNSUPPORTED_SM89 label is wrong; no fallback on a post-launch crash) | string check only |
| 13 | EAGLE pin eagle.5 (decision) | PARTIAL (the choice is not recorded in DESIGN, notebook or results) | — |
| 14 | Merlin full sha | PARTIAL (only TP1 is gated; the replicas, TP4 and their A/A reps can run after a sha FAIL) | _merlin_sha behavioural; the gate is not tested |
| 15 | Budget | PARTIAL (FULL estimates add to 702 min > 690; SMOKE_B compiled PP4 is skipped by its own estimates) | — |
| 16 | Merlin patch A/B | PARTIAL (label only; the switch check is not done or recorded) | — |
| 17 | P5 label | IMPLEMENTED | — |
| 18 | Throughput comparability note | IMPLEMENTED | — |
| 19 | Run order | IMPLEMENTED | — |
| 20 | EAGLE-4 licence (decision) | DONE by user decision (not recorded in the repo) | — |

## 2. Item by item

### #1 NVFP4 admission gate

**(a) IMPLEMENTED.** L:507-523 (`compare_tf`):
- returns `INCONCLUSIVE_INCOMPLETE_TF` when positions are empty or of different length (L:514-515);
- returns `INCONCLUSIVE_INCOMPLETE_TF` when any pair has a None position or None logprob, with valid/expected counts (L:516-518);
- MEASURED only when all positions are valid.

rev 3c (HEAD L:507-517) gave MEASURED on 1 of 256 valid.

The "256 per band" count is not asserted explicitly. `teacher_forced` always produces 256 slots for these long prompts (L:497), and any missing slot is None, so the effect is equivalent.

Test: G:57-64. 1/256 gives INCONCLUSIVE_INCOMPLETE_TF; full gives MEASURED; the 24.95-nat fixture stays MEASURED (a measurement, not a pass). Behavioural.

**(b) IMPLEMENTED.** B:1027-1031. `s5(0)` calls `oracle_gate(o, need=("fp8_mtp0",))` and returns early with that status on any of these:
- a request error → NO_GO_ORACLE_REQUEST_ERROR;
- garbage → NO_GO_ORACLE_GARBAGE;
- FAIL_HIGH_MARGIN → NO_GO_ORACLE;
- a missing reference → INCONCLUSIVE_NO_ORACLE;
- any other non-EXACT/NEAR_TIE status → INCONCLUSIVE_ORACLE.

It does no census levels after a failure. Missing and inconclusive cases are named INCONCLUSIVE_*, not NO_GO_ORACLE_*. That is acceptable, because #1c requires `target_only == DONE` and these cases block MTP3 either way.

A timed-out candidate row carries `error="GREEDY_TIMEOUT"` (L:414), which the gate catches. Request errors are tested through `oracle_gate` (G:47). The `s5(0)` path itself is not exercised.

**(c) IMPLEMENTED.** `nvfp4_gate()` (B:983-1012) is checked at the very top of `s5()` when `spec` is set (B:1017-1020). It returns `SKIPPED_CORRECTNESS_GATE` before `stock_server`/`with_servers` is called, so **no MTP3 process is launched** when the gate fails (control flow traced). The A/A reps `s5(3, tag=...)` (B:1259) go through the same `s5`, so the gate is re-evaluated for them, and `_done("5_nvfp4_kv/nvfp4_mtp3")` is also required.

The gate requires:
- target_only DONE and its vs_fp8_mtp0 in CLEAN (B:993);
- tf_reference, tf_aa_control and tf_nvfp4 DONE (B:995);
- every NVFP4 band inside the limits (B:998-1009);
- at least one band (B:1010).

On "both bands complete": `s5_tf(ref=...)` gives DONE only if every band in `rows` is MEASURED (B:973-975). With `SM_TF` False (FULL), the bands are 40k and 90k. The reference step `tf_fp8kv_reference` (key only) checks only `n` (B:976). `teacher_forced` returns n=256 even when every position is None (L:504), so that DONE does not prove completeness by itself. Completeness is enforced transitively, because both the A/A comparison and the NVFP4 comparison run `compare_tf` against that reference. This is OK, but the reference status alone is weak.

Gaps:
- **The tainted session is not checked.** `nvfp4_gate` does not look at `R["setup"]["tainted_xid"]`, so it accepts evidence that was produced after a Xid.
- **The gate result is not recorded on a pass.** `ok, why = nvfp4_gate()` keeps `why` only when it fails (B:1018-1020). On a pass, the band limits and values that justified launching MTP3 are not written anywhere.
- **No gate verdict in SMOKE.** SMOKE has no MTP3 step, so nvfp4_gate is never evaluated, although the CELL6 NOTES (B:1311) refer to it.

Tests: G:68-85. Clean allowed; INCONCLUSIVE_TF, missing oracle, FAIL_HIGH_MARGIN, target not DONE, no A/A and the 24.95-nat fixture are each blocked. Behavioural on `nvfp4_gate`. **`s5(3)`'s "no launch" is not tested** (it follows from the code, but no test drives it).

### #2 Teacher-forced limits from an FP8-vs-FP8 control — PARTIAL

The A/A launch is implemented: B:1240-1241 adds `tf_fp8kv_aa_control = s5_tf("fp8_e4m3", ref="fp8kv_tf", name="tf_fp8_e4m3_aa")`. That is a fresh launch with the same flags (B:969), in SMOKE_B and in FULL, before NVFP4 target-only. MEASURED stays a measurement (L:507-510 docstring, NOTES B:1311).

**Defect: the code does not match its own docstring or GPT-4's 0.99 screen.** The docstring (B:986-987) says "rank-1 agreement >= 0.99 and >= A/A - 0.01". The code (B:1000-1003) is:

```python
lim = {"rank1_min": min(0.99, (a.get("rank1_agreement") or 1.0) - 0.01) if a.get("rank1_agreement") else 0.99, ...}
lim["rank1_min"] = max(lim["rank1_min"], 0.98)
```

That is `min`, not `max`, so the effective threshold is clamped to the range [0.98, 0.99]:
- A/A r1 = 1.0 gives 0.99;
- A/A r1 = 0.995 gives 0.985;
- A/A r1 ≤ 0.99 gives 0.98.

NVFP4 can therefore pass with rank-1 agreement as low as 0.98, which contradicts both the docstring and the "≥ 0.99" screen.

**The tolerance is mostly fixed floors, not derived from the A/A run.** With a deterministic FP8 A/A (0 difference expected for same flags and same prompt), the limits become fixed floors: mean ≤ 0.05 nat and max ≤ 1.0 nat. The list asked for limits "derived from" the A/A run. The floors decide in practice. That is a policy choice, but it should be stated in the report.

**Missing A/A band.** If an A/A band is missing, `a = {}` and the defaults (0.99 / 0.05 / 1.0) are used silently, with no "no A/A band" failure. This is guarded only indirectly, because the A/A status must be DONE.

The test (G:84-85) covers the gross 24.95-nat case only. There is no boundary test (for example NVFP4 r1 = 0.985 with A/A 1.0 must block; with A/A 0.995 the code allows it, against the docstring).

### #3 A shared correctness gate for every arm — IMPLEMENTED for the listed arms; gaps below

The helpers are `oracle_gate` / `gated` (B:583-607). `gated` sets `correctness` and replaces a None/DONE status with the NO_GO/INCONCLUSIVE status. It never overwrites SERVER_DIED and similar statuses, because `with_servers` sets those after `body`.

| Arm | Gate | Lines |
|---|---|---|
| mtp0 (reference) | errors/garbage only | B:850 (`need=()`) |
| K1 | need fp8_mtp0 | B:850 |
| K3/G0 + A/A reps | need fp8_mtp0 | B:850 |
| G1/G2/G3 | need fp8_mtp0 (vs fp8_g0 is recorded, not gated) | B:946 |
| NVFP4 target-only / MTP3 | need fp8_mtp0 | B:1028, B:1037 |
| INT4 ref | errors/garbage | B:1048 |
| INT4 MTP3 control | need int4_mtp0 | B:1063 |
| EAGLE | need fp8_kvauto_mtp0 | B:1134 |
| kvauto oracle | errors/garbage | B:1118 |
| PP4 | own staged gate | B:1090-1099 |
| Merlin TP1 | 90k garbage/errors as an extra | B:746-748 |
| Merlin replicas | replica_greedy vs TP1 | B:790 |
| Merlin TP4 | need merlin + 90k vs TP1 | B:808 |

`oracle()` now never stores an errored, empty or garbage reference, so consumers get INCONCLUSIVE_NO_ORACLE (B:510-514).

`_done()` (B:1251-1254) requires status DONE, `correctness == "CLEAN"` and no taint. NO_GO and INCONCLUSIVE arms therefore have status ≠ DONE and are excluded, and LAUNCH_FAILED records have no `correctness` key and are excluded. Verified.

Gaps:
- **`_done()` is not tested.** G does not extract it (WANT, G:18). The only test is `gated_nested_fail_not_done` (G:53-54). It checks the output of `gated` and does not cover the claim that "`_done()` does not pick it for A/A".
- **Chat probes are not part of the gate.** An arm with `probes.ok == False` (wrong arithmetic, which is the E3/E13 symptom) can still be DONE/CLEAN, except PP4. The audits' "oracle-failure propagation" theme implies this should count as at least INCONCLUSIVE.
- **The Merlin TP1 90k reference is stored unconditionally.** `ROWS["merlin_90k"] = g90` (B:739) runs even when the row errored or is garbage. This is inconsistent with the new rule at B:510-514. TP4's comparison would then be INCONCLUSIVE/FAIL instead of INCONCLUSIVE_NO_ORACLE. That is still not DONE, so the risk is low.

### #4 INT4 target-only oracle — IMPLEMENTED

- `s4_int4_ref` (B:1044-1048) launches `int4_mtp0_ref` with spec 0, seqs 8, TP4 and kv fp8_e4m3 (the same KV dtype as the MTP3 control and PP4). It runs the oracle only (2k/2k/8k/40k greedy, B:509) and keeps it as key `int4_mtp0`, gated.
- **SMOKE_B order (verified, B:1226-1249):** mtp0 → **int4_mtp0_target_only_ref** (B:1230, outside `if not SM`) → mtp3/G0 → G3 → TF ref → TF A/A → NVFP4 target-only → TF NVFP4 → [kvauto + EAGLE] → **pp4()** (B:1249). int4_mtp0 is therefore created before PP4. The INT4 prefetch now also runs in SMOKE (B:1227, `if INT4:`).
- The MTP3 control compares with `int4_mtp0` (B:1059, B:1063). PP4 compares every stage with `ROWS["int4_mtp0"]` (B:1089-1092). PP4 returns `SKIPPED_NO_ORACLE` when the reference is missing (B:1074-1075). FP8 is not used as a substitute.
- A reference timeout produces an errored row. The row is not stored (B:510-514), so the MTP3 control gets INCONCLUSIVE_NO_ORACLE and PP4 gets SKIPPED_NO_ORACLE. Neither is DONE.

Tests: G:171-173 are string checks only. They prove the text exists and its order vs the FULL-only control. They **do not prove the step is outside `if not SM`** (indentation is not checked), and they do not test the behaviour "a high-margin change gives NO_GO" or "reference timeout gives INCONCLUSIVE" for this arm. Those are covered only generically by `oracle_gate`.

### #5 PP4

**(a) Staged fail-fast — IMPLEMENTED.** B:1083-1101:
1. probes fail → `NO_GO_SHORT_PROBE`, no greedy;
2. then 2k (idx 0), 8k (idx 2), 40k (idx 3), each a separate `L.greedy` call compared with the matching `int4_mtp0` rows; error, empty, garbage or a non-CLEAN status → `NO_GO_<stage>` and return;
3. throughput (`levels`) only after all three stages pass.

A short-probe failure therefore sends zero greedy requests, and a 2k failure sends zero 8k/40k requests (traced).

Only one of the two 2k oracle prompts is used. That is fine, but stage "2k" covers 1 row.

**(b) Eager first — IMPLEMENTED.** `pp4()` (B:1106-1112) runs eager first. Compiled runs only if eager status is DONE and the session is not tainted; otherwise it records `SKIPPED_EAGER_NOT_PROVEN`. Eager DONE implies its stages passed, because the status is set only on failure.

**(c) Tainted session — PARTIAL.**
- XID regex (B:423); `xid` is recorded per server after the body (B:483-485); `taint_check` (B:610-618) sets `R["setup"]["tainted_xid"]` and flags the step; `_done` and compiled PP4 respect it; the CELL6 note prints it.

Gaps:
- **No xid on LAUNCH_FAILED.** `with_servers` returns early on LAUNCH_FAILED (B:471-474) before the xid scan. A server that hits an illegal memory access during start-up or CUDA-graph capture is never flagged. This is exactly the #63505 compiled-PP pattern. Compiled PP4 is the last B step, and the FULL A/A repeats run after it.
- **Merlin runs are not taint-checked.** `merlin_run` calls `with_servers` directly (B:711, B:726) without `taint_check`. A Xid in Merlin TP1, the replicas or TP4 (the whole A block, which runs first) is recorded as `servers[*].xid=True` but does not taint the session.
- **"Nothing after it is used as correctness evidence" holds only for `_done` and compiled PP4.** `nvfp4_gate`, the INT4 control vs `int4_mtp0`, and the per-arm `correctness` column still say CLEAN in a tainted session; only `tainted_session` is added.
- **Not tested.** No gate test covers XID or `taint_check` (the list asked for "an Xid line in a fake log means TAINTED_XID").
- **PP4 stage tests are string checks only** (G:170-171). No behavioural test covers "a probe fail sends zero 2k/40k requests" or "a 2k fail sends zero 8k/40k".

### #6 FULL-graph proof

**(a) IMPLEMENTED.** L:785-795. Without per-forward annotations the verdict is `UNPROVEN`, with the CUPTI result kept only as a "hint". rev 3c (HEAD L:769-774) could return FULL. Tested: G:98-101 gives UNPROVEN; G:102-106 shows the CPU annotation gives FULL. `graph_proof` maps UNPROVEN/UNKNOWN/None to INCONCLUSIVE_GRAPH (B:868-869).

**(b) IMPLEMENTED.** L:868-871. When the run is not settled, the status is `INCONCLUSIVE_PREFILL_NOT_SETTLED` and the function returns before `profile()`. The trace is not taken. A prompt-counter delta during the window gives `INCONCLUSIVE_PREFILL_IN_PROFILE`, and missing metrics give `INCONCLUSIVE_METRICS_UNAVAILABLE` (L:872-889). `graph_proof` passes these statuses through (B:864-865). Tested only as that pass-through (G:89). `profiled_load` itself is not tested.

Note: the "window" deltas span `profile()`, which is secs + 20 s sleep + trace parse (L:820-830). That is conservative for the prompt check.

**(c) IMPLEMENTED (drafts only).** `spec=True` is passed for G0 (B:846) and G1-G3 (B:943); G1-G3 use `stock_flags`' default spec=3. A zero drafts delta gives `INCONCLUSIVE_NO_SPEC_DECODE` (L:888-889). The accepted delta is recorded but not required to be non-zero (the list said "draft/accept"). Not tested.

**(d) IMPLEMENTED, stricter than asked.** G0's `graph_proof(o, expect="PIECEWISE")` (B:852-853). In `s3`, any G0 status other than `PROVEN_PIECEWISE` (FULL, unsettled, downgraded, missing) makes G1-G3 `INCONCLUSIVE_GRAPH` and keeps the arm's own result under `own` (B:949-952). In both SMOKE_B and FULL, G0 runs before the G steps.

Risk: G0 is launched with `cudagraph_mode=PIECEWISE`. If vLLM 0.32 logs any line matching `setting cudagraph_mode\s*=` for it (L:146), G0 becomes DOWNGRADED and every G verdict becomes INCONCLUSIVE_GRAPH, even though no real downgrade happened.

Test: `gp_g0_mismatch` (G:94) checks only G0's own MISMATCH_EXPECTED. **The s3 propagation ("G0 reads FULL gives INCONCLUSIVE_GRAPH for G1-G3") is not tested.**

**(e) PARTIAL.**
- No per-forward filter to verify forwards (query tokens > 1 per request) exists. `trace_summary` still takes the median over every `gpu_model_runner … forward` annotation (L:761-768). The proxy is "decode-only window + drafts > 0". With MTP every target decode step is a verify step, so the proxy is close, but forwards are not classified. Drafter forwards, if they carry the same annotation name, would be mixed in.
- A downgrade line gives status `DOWNGRADED` (B:870-872), not `INCONCLUSIVE_GRAPH` as specified. It is still never PROVEN, so the effect is the same. The test (G:91-93) codifies `DOWNGRADED`.
- A missing trace gives INCONCLUSIVE_GRAPH (B:866-867), as specified.

### #7 P5 release and result gate — PARTIAL

`s2()` (B:629-652):
- release is checked on "0,1,2,3";
- None with nvidia-smi working at start becomes False;
- False → `ABORT` + `FAIL_GPU_RELEASE`;
- DONE only if rc == 0 and `summary.all_checks_pass is True`, which is the real key in `bench_host_ar.py` (summarize, :272-275; the list said "all_ok", but P5 has no top-level all_ok);
- otherwise CHECK_FAILED or BENCH_ERROR.

`step()` skips every Merlin step with SKIPPED_ABORT. Tested behaviourally (G:131-140): False or None gives FAIL_GPU_RELEASE + ABORT; clean gives DONE; all_checks_pass False gives CHECK_FAILED.

Gaps:
- **The P4 bench still starts on GPU 3 after FAIL_GPU_RELEASE.** B:1215-1217 (`P4_FUT = ...submit(s4_thread)`) has no `ABORT["why"]` check. Merlin is protected, but a GPU process is still launched on GPUs that P5 was holding.
- **An exception skips the release gate.** Any exception inside `s2()` before the release line (for example Popen failure, or `json.loads` on a bad file; the bench writes atomically, so this is unlikely) becomes ERROR through `step()`, with no release check and no ABORT. Merlin would then start.
- The `or bool(FAKE)` in B:651 makes P5 DONE in fake mode regardless of checks. That is harness-only, but it means the kit never exercises CHECK_FAILED through the notebook.

### #8 GPU preflight hard stop — IMPLEMENTED

B:134-139. Fewer or more than 4 GPUs, or nvidia-smi failing (`gpu_mem()` returns [] or None, L:181-187), leads to `save()` then `raise RuntimeError("GPU_PREFLIGHT_FAILED…")` in CELL 1. That is before CELL 2 (install) and CELL 5 (P5/Merlin), so no GPU process starts. FAKE is exempt.

Test: G:174 is a string check only. The list asked for "a mock with 3 GPUs or nvidia-smi failure means no P5/Merlin process"; that is not exercised.

### #9 Kit hash hard stop — IMPLEMENTED

B:126-131. All four of `p4_sm89_fp8.so`, `p5_host_ar.so`, `bench_fp8_gemm.py` and `bench_host_ar.py` must be `is True` in `kits_sha_ok`; a file missing from the manifest also fails. Otherwise `save()` + `raise KERNEL_KIT_HASH_FAILED`, in CELL 1 before any GPU process.

Test: G:175 is a string check only (no "one changed byte" test).

### #10 P4 real weights only — IMPLEMENTED

B:660 passes `--weights real` (the bench supports it: `bench_fp8_gemm.py:600`, and it raises when no index exists, :226-227). B:673-677: any shape whose `weights_source.kind != "real"`, including a missing kind after a child crash, gives `CHECK_FAILED / SYNTHETIC_WEIGHTS_NOT_ALLOWED`. DONE needs rc 0 + `all_ok`; the bench sets `all_ok` False when `shapes` is empty (:582). Tested behaviourally (G:141-148).

Minor: a child that crashes before loading weights (for example the sm_89 check) is also labelled SYNTHETIC_WEIGHTS_NOT_ALLOWED, which is a misleading reason.

### #11 Fresh consolidated CPU kit verdict — PARTIAL

Done:
- K:25/40 embed `gate_tests.py`;
- K:110-115 run it in p1 against the FULL notebook code and the real lab, and parse `GATES {json}`;
- K:258 adds `gates.all_ok is True` to the p1 verdict next to `unit_ok`;
- the verdict is `REP["all_ok"]`, not the exit code;
- the local run passes (40/40).

Missing or partial:
- **The kit has not been run on Kaggle for rev 3d** (the list: "Rerun the full kit (or all splits) on the fixed code"). There is no rev 3d kit output; only the local gate run exists. There is no consolidated p1-p5 aggregation. Each part has its own `all_ok`, and nothing combines the five reports.
- **The kit's `BAD` step list (K:249) is unchanged.** It is ERROR, BODY_ERROR, SERVER_DIED, FAIL_GPU_RELEASE, LAUNCH_FAILED and PATCH_FAILED. The new statuses (NO_GO_ORACLE*, INCONCLUSIVE_NO_ORACLE, SKIPPED_NO_ORACLE, FAIL_CHECKPOINT_SHA, SKIPPED_CORRECTNESS_GATE, KERNEL/GPU preflight raises) are not checked against expectations in the fake end-to-end runs. A gate that wrongly blocks everything (for example int4_mtp0 never stored on the fake server, so PP4 is always SKIPPED_NO_ORACLE) would pass the kit.
- **The gate tests run only in p1** and do not cover:
  - `_done`;
  - `taint_check` / XID;
  - the `s5(3)` no-launch path;
  - PP4 staging and the compiled-PP4 skip;
  - s3 G0 propagation;
  - `profiled_load` (unsettled, prefill-in-window, no-spec);
  - `oracle()`'s clean-store rule;
  - `merlin_sha_gate` and its use by the replicas and TP4;
  - the EAGLE fallback and its labels;
  - `s4_int4_ref`;
  - preflight and kit hash (string only).

### #12 EAGLE linear backend — PARTIAL

B:1135-1154. `tries` = cutedsl (`--linear-backend flashinfer_cutedsl`, prefix on), then default (prefix on), then default with prefix off (labelled confounded). Each result is labelled with `variant` / `linear_backend` / `prefix_caching`, and earlier attempts are kept.

Defects:
- **The label is wrong when everything fails.** If all three launches fail, the overall status is `SKIPPED_UNSUPPORTED_SM89` (B:1154). That hides a genuine launch failure of the default backend as a "skip". The list said SKIPPED_UNSUPPORTED_SM89 "for that variant only". When cutedsl fails and default succeeds, the cutedsl attempt is stored as plain `LAUNCH_FAILED` in `earlier_attempts`, not as SKIPPED_UNSUPPORTED_SM89. The label is therefore applied in exactly the wrong case.
- **No fallback after a post-launch crash.** Fallback happens only on `LAUNCH_FAILED`. If CuTe-DSL launches but fails at the first request because no sm_89 kernel exists (SERVER_DIED, BODY_ERROR, or NO_GO from garbage), the default backend is never tried.
- **The budget is unchanged.** Up to 3 launches × `with_servers(…, 2400)` (40 min start-up timeout each) against a 26-min estimate and the `(11+26+10)*60` guard (B:1246).

Test: G:177 is a string-order check only.

### #13 EAGLE pin (decision: keep eagle.5) — PARTIAL

No code change was needed. The list said "keep eagle.5 for the feasibility smoke **and record the choice**". "eagle.5" appears only in the fix list and in `prep/P6_eagle4/build_p6_kernels.py`. DESIGN.md, the notebook MD and the results do not record it, and `eagle_install()` (B:294-320) does not record the installed vLLM/EAGLE version in `R["setup"]`.

### #14 Merlin checkpoint full sha — PARTIAL

`_merlin_sha` (B:1160-1176) computes the full sha256 of every manifest file (except .gitattributes) against the P1 manifest. The real manifest has `[size, sha]` for all 15 files (`build_p1_kernel.py:26-42`, :418). It runs in a thread after the Merlin prefetch (B:1188-1201) and writes `inputs.merlin_sha_ok`. `merlin_sha_gate` (B:1179-1185) joins it (up to 30 min) and returns `FAIL_CHECKPOINT_SHA` unless ok is True. Tested: G:150-167 (good passes, one changed byte fails).

Gaps:
- **Only `s1_tp1` calls the gate** (B:752-754). `s1_rep` (B:773-792), `s1_tp4` (B:795-817) and the A/A `merlin_tp1_x4_rep*` do not. After TP1 returns FAIL_CHECKPOINT_SHA, or if the sha thread had not finished, `b_merlin_tp1_x2`, `c_merlin_tp1_x4` and `d_merlin_tp4` still launch Merlin on an unverified checkpoint. They can also be DONE/CLEAN and feed A/A.
- `merlin_sha_gate` itself is not tested.
- `inputs.merlin_files_ok` (size-only) stays, which is fine because the name is not "identity". `R["identity"]["merlin"]` is still the config/tokenizer/index hash (L:208-217). The report should say that identity = `merlin_sha_ok`.
- The join wait (up to 1,800 s) is not in the TP1 estimate.

### #15 Budget — PARTIAL

B:50: SMOKE B goes from 175 to 200 min (the list said about 195), ALL from 290 to 315, FULL stays 690. New step estimates: int4 ref 12, TF A/A 11/15, compiled PP4 15/25.

**Own-estimate sums (step `est_min`):**

| Plan | rev 3c | rev 3d | Budget |
|---|---|---|---|
| FULL | ≈ 650 min (compiled PP4 only; eager only on failure) | **702 min** (A block 205 + B block 497) | 690 |
| SMOKE_B | — | **189 min** (122 before EAGLE, EAGLE 37, PP4 eager 15 + compiled 15) | 200 |

FULL is over budget by its own estimates before any install time or the 5-min per-step guard. The A/A margin is gone, and the PP4 tail is at risk of SKIPPED_BUDGET. The list said "FULL estimates change to match, while FULL stays at 690". Nothing was trimmed.

SMOKE_B time is plus the stock install, tokenisation and EAGLE install inside the step. The EAGLE guard reserves only 10 min for PP4, while PP4 eager needs 20 (15 + 5-min guard). By the estimates, compiled PP4 is SKIPPED_BUDGET in SMOKE_B, and eager is borderline.

Not budgeted: EAGLE's 3 attempts (#12) and the Merlin sha join (#14).

Stale comments: the run-order comment (B:1207-1210) still lists the rev 3c order (no int4_mtp0, no TF A/A, "PP4" without eager-first). The module docstring (B:8) says "PP4 last with an --enforce-eager arm".

### #16 Merlin low-M patches — PARTIAL

The label is present in the INT4 control record note (B:1060) and in the CELL6 NOTES (B:1314). The list's action "check whether Merlin has switches to turn the patches off; if it does, add an A/B arm in FULL" is not done or recorded anywhere (no search result, no note saying "no switch found").

### #17 P5 interpretation — IMPLEMENTED

CELL6 NOTES (B:1312-1313): 983,040 B, 4-rank graph microbenchmark vs NCCL 343.88 µs; "not an engine gain until the collective runs inside vLLM".

### #18 Throughput comparisons — IMPLEMENTED

NOTES (B:1315-1316): the PDF baselines are compared only with the same workload, running_mean and accepted-output accounting. The CELL6 window table already has running/accept-length columns (B:1285).

### #19 Run order — IMPLEMENTED

P5 stays first (B:1213). int4_mtp0 moved to the second B step (B:1230).

### #20 EAGLE-4 licence — DONE (user decision)

Nothing in the code. The acceptance is not recorded in DESIGN.md or in the notebook; adding one line is optional.

## 3. What the audit themes imply but neither the list nor the code covers

1. **The Xid taint does not cover the most likely Xid paths:** LAUNCH_FAILED start-up crashes (graph capture, compiled PP) and the whole Merlin block (B:471-474, B:711/726). The taint also does not reach `nvfp4_gate` or the per-arm `correctness` column.
2. **The P4 bench launches after a P5 FAIL_GPU_RELEASE** (B:1215-1217, no ABORT check). An exception in `s2()` bypasses the release gate entirely.
3. **The Merlin sha gate protects TP1 only;** the replicas, TP4 and their A/A reps can run on a checkpoint that failed or was not verified.
4. **The NVFP4 tolerance** is looser in code than in its docstring and GPT-4's 0.99 (min vs max). It is not recorded on a pass and not evaluated in SMOKE.
5. **Chat-probe failures do not affect DONE/CLEAN** for any arm except PP4.
6. **The EAGLE status semantics are inverted** (SKIPPED_UNSUPPORTED_SM89 = everything failed), and there is no fallback when cutedsl launches but dies on first use.
7. **Graph proof has no per-forward verify filter.** A downgrade line is DOWNGRADED, not INCONCLUSIVE_GRAPH. A benign "setting cudagraph_mode=" line on G0 would void every G verdict.
8. **The budget arithmetic was not redone** (FULL 702 > 690; compiled PP4 is effectively unschedulable in SMOKE_B).
9. **CPU kit:** no Kaggle run of rev 3d. The BAD list does not include the new statuses. Behavioural gate tests are missing for `_done`, taint, the s5(3) no-launch path, PP4 staging, G0 propagation, profiled_load, oracle storage, the Merlin sha gate, EAGLE, and preflight/kit-hash (string only).
10. **Records:** the eagle.5 choice, the licence acceptance, the installed EAGLE version, and the Merlin patch-switch check are not recorded.
11. **Baseline hygiene:** `_scratch/e23/e23_lab.pre3d.py` is actually rev 3d. Use git HEAD for the rev 3c lab.
12. **Minor:** `ROWS["merlin_90k"]` is stored without the clean check (B:739). The `tf_fp8kv_reference` DONE checks only `n`, which is 256 even when every position is None. The P4 SYNTHETIC label also covers early child crashes.

The two rejected claims (690 min; .so files not in the zip) are handled as decided. #9 makes the dataset sha check a hard stop. No manifest-with-hashes file was added for "the next zip" yet; that was noted in the list as a next-zip item.
