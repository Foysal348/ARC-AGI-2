# E23 rev 3b audit 5 / part 3: plan fidelity and time budget

Scope: `build_e23_notebook.py` (rev 3b, CELL1-CELL6), `src/e23_lab.py`, `DESIGN.md`, local fake runs `runA.log`, `runB.log`,
`runB3.log`, `runF.log`, and the matching `out*/e23_results.json`. READ-ONLY audit. Builder line numbers are file lines of
`build_e23_notebook.py`, and `lab:` means `src/e23_lab.py`.

Fake-run caveat: `runA`, `runB` and `runF` were made with the rev 3 code (`nb_*.py`, 120k band = 120,000). The only rev 3b
run is `runB3` (`nb3b_SMOKE_B.py`), and every stock step in it failed with `PORT_BUSY 8200`. `runF` held port 8200 at the
same moment (local parallel runs share one host). So no fake run has completed on rev 3b, and the A/A loop and the
PP4-after-A/A path have never run, even in a fake run: `runF` used `E23_BUDGET_SEC` = 90 min and was still in mtp1.

---------------------------------------------------------------------------------------------------------------------

## (a) The plan mapped to code

| Plan step | Code | Mode | Verdict |
|---|---|---|---|
| 1. Merlin W4A16 on one L4 + memory fit | `s1_tp1` L572-581 (GPU 0, `KV_TP1`, seqs 4, capture `CAP_TP1`); facts + `gpu_mem_after_ready` + `host_rss` in `with_servers` L431-440; peak VRAM per window in `steady` (lab L635) | A, FULL | OK |
| 1. 8k / 40k / 90k correctness | `oracle(port, key="merlin")` L565, which uses `ORACLE_P + LONG_P` = 2k, 2k, 8k, 40k (L376-377) | A, FULL | **90k MISSING** (F5). The TP1 rows are only a reference: TP4 is compared with TP1 (L613). No independent reference exists |
| 1. 2 replicas | `s1_rep(2)` L807 | FULL only | OK in FULL, **never smoked** (F4) |
| 1. 4 replicas | `s1_rep(4)` L808 | FULL only | as above |
| 1. Matched throughput vs TP4 | `s1_tp4` L610-623 (40k 8/16/24, 90k 8/12/16, 120k 8/12) | A, FULL | partly matched (F13) |
| 1. Separate TP1xPP4 branch | `s1_pp4(True)` + `s1_pp4(False)` L834-835 (stock + RedHat INT4, spec 0, seqs 16). Stop on garbage L760-762 | B, FULL (last) | **extra arm** (eager + compiled, F7). **Skipped by budget** in FULL (F1) and SMOKE_B (F3). No oracle in SMOKE (F8) |
| 1. Decisive number: streams per replica at 90k | TP1 90k levels `[1, 2]` L567. The replicas reuse the best level found in step a (L593) | A, FULL | capped at 2 although seqs = 4 (F13) |
| 2. Oracle -> 4-rank reduction -> graph capture -> INT8 arm | `s2` L516-528 runs `bench_host_ar.py` steps a,b,c,d (default `--steps a,b,c,d`, prep bench L1133), `--quick` in SMOKE. First, all four GPUs free (L802) | A, FULL | OK. vLLM integration is correctly left for later |
| 3. G0 FlashInfer PIECEWISE | `s6(3, profile=True)` L814, with PIECEWISE pinned at L630 | B, FULL | OK |
| 3. G1 TRITON FULL_AND_PIECEWISE | `s3("G1", ...)` L816 | FULL only | OK; engine risk not smoked (F19) |
| 3. G2 TRITON FULL_DECODE_ONLY | `s3("G2", ...)` L817 | FULL only | as above |
| 3. G3 FlashInfer + #60532 | `s3("G3", ..., patch=True)` L818, dry-run / apply / sha-verified revert L657-679, ABORT on a failed revert L707 | B, FULL | OK |
| 3. Trace proof | `L.profiled_load` in every arm (L645, L698) -> `trace_summary` graph verdict (lab L730-783) | B, FULL | OK, in the plan's order G0 -> G1 -> G2 -> G3 |
| 4. #61224 CUTLASS vs MarlinFP8, same weights, actual shapes, numeric match first | `s4_thread` L531-544 on GPU 3, in parallel with TP1 on GPU 0. Joined before any GPU-3 step (L805). All 5 TP4 shapes, M default 1..128 in FULL; 2 shapes and M 1/16/80 in SMOKE | A, FULL | OK |
| 4. Merlin low-M patches, INT4 separately | Merlin TP4 (d) vs `s4_int4` L737-746 (stock + RedHat INT4) | FULL only | confounded and partly matched (F9). `s4_int4` is a late step and gets skipped (F1) |
| 5. FP8 control | mtp0 (target only) and G0 (MTP3) L811, L814 | B, FULL | OK |
| 5. NVFP4 target-only (correctness) | `s5(0)` L819: greedy vs fp8_mtp0, teacher-forced at 40k / 90k, NO_GO gate L724-726 | B, FULL | OK, but it also runs the full census sweep, which the plan does not ask for (F6) |
| 5. MTP3 -> 90k census -> more streams -> matched throughput | `s5(3)` L821 (90k 12/16/20/24, 120k 12/16/20, 40k 16/20), gated on the target-only result (L715-716) | FULL only | OK; not smoked (F19) |
| 6. MTP OFF / K1 / K3 at 8/12/16/20 | `s6(0)` L811, `s6(1)` L813, `s6(3)` L814, all with 40k `[8,12,16,20]` (L640) | FULL (SMOKE: 40k@16, no K1) | OK |
| 6. EAGLE-4 later, separate | `s7_*` L770-790 only `if SM or E23_EAGLE_FULL` L823 (an env var a Kaggle commit cannot set) | B only | OK, matches "EAGLE only as SMOKE feasibility" |
| SMOKE before FULL, 120/150/690, 120k band | L45-52, L372 (116k start), L375 (`MAX_NEW` 14,560 at 120k) | all | OK |
| A/A repeats fill FULL | L826-832 | FULL | does not fill the time; starves PP4 (F1, F2) |

Steps the plan does not ask for: the PP4 eager arm (F7); the NVFP4 target-only census sweep with 9 levels (F6); the 90k
levels in G1/G2/G3 (useful, but the main time sink that pushes PP4 out, F1). The TP1/TP4 memory fallbacks, the
kv-auto oracle (needed for EAGLE) and `s4_int4` (the plan's "INT4 separately") are justified.

Order: #1 a -> b -> c -> d follows the plan's sequence. #3 runs G0 -> G3 in order. #5 follows control -> target-only ->
MTP3. PP4 runs last on purpose (#63505 abort risk). None of the steps is out of order against the plan. The problems are
budget and reserve, below.

---------------------------------------------------------------------------------------------------------------------

## (c) Time model

These inputs are measured or taken from the repo:

- Stock TP4 start 430-545 s (E21 vLLM 0.32 INT4 log: 487 s). I use 8.5 min, plus about 1 min for kill (SIGTERM 40 s,
  then SIGKILL) and the GPU-release gate.
- Wheelhouse install: 124-182 s measured.
- Merlin: 4.3 GB extract plus the offline install, about 6 min (4-10). The first TP1 start, including FlashInfer JIT,
  compile and capture, about 12 min (8-20). Later TP1 starts about 8-9 min: there is a per-port `TRITON_CACHE_DIR`, so
  replicas 1-3 recompile Triton. Merlin TP4 start about 10 min.
- EAGLE: 4.25 GB extract plus install, about 6 min; start about 10 min.
- TP4 prefill about 2,000 tok/s at 8-40k and 1,600 at 90-116k. Sources: the E21 0.32 log shows 2.4-2.7k; E14 and E17
  measured 1.45-1.7k on 0.19. Merlin TP1 prefill about 600 tok/s (one L4).
- KV pool: FP8 about 1.05M tokens (E18/E21 0.32 logs 1.0-1.1M), INT4 1.45M (E21 log), NVFP4 about 1.8x FP8 = 1.9M.
  A KV-bound level only prefills what fits.
- Cost of one level = prefill of the NEW streams (prefix caching is on, so streams 0..n_prev-1 hit the cache) + WIN +
  about 25 s. The 25 s covers the idle check at entry, 3 settle samples, stop/join, and idle-after. The fake FULL run
  shows about 8 s of admission overhead plus the stop/join on top of the 120 s windows, which agrees.
- WIN = 30 s in SMOKE and 120 s in FULL. Without cache hits, the long-band levels take 1.3-2x longer.

### SMOKE_A (budget 120 min)

| Step | Code est | Realistic | Basis |
|---|---|---|---|
| setup (cell 1-3, stock install) | - | 4 | install 2.5-3 min |
| p5_bench (quick) | 12 | 3 | microbench; 50 iters x 3 reps |
| a_merlin_tp1_gpu0 | 45 | 39 (fallback launch: about 70) | install 6 + start 12 + oracle 2 + 40k@2,4 + 90k@1,2 + 120k@1 = 17 min of levels |
| (P4 on GPU 3 in parallel) | - | 3-8, hidden behind a | FP8 shards read cold (SMOKE_A does not prefetch FP8) |
| d_merlin_tp4 | 40 | 38 | start 10 + 40k@8,16 + 90k@8 (720k tokens) + 120k@8 (928k tokens) = 26 min |
| **total** | 97 | **about 84 (range 80-100)** | fits with about 36 min spare |

With the TP1 memory fallback (LAUNCH_FAILED), step a takes about 70 min. d then starts with about 43 min left and needs
45 (`est 40 + 5`), so it is SKIPPED_BUDGET: TP4 Merlin is not smoked in that case.

### SMOKE_B (budget 150 min)

| Step | Code est | Realistic | Basis |
|---|---|---|---|
| setup + first `--help=all` | - | 5 | |
| mtp0_target_only | 20 | 19.3 | start 8.5 + oracle 1 + TF 40k/90k 2.5 + 40k@16 (640k tokens) 6.3 + 1 |
| mtp3_eq_G0_piecewise | 25 | 19.3 | same, plus a profile on cached prompts, about 2.5 |
| G3_flashinfer_pr60532 | 20 | 19.5 | |
| nvfp4_target_only | **25** | **40.6** | 90k@12 = 1.08M tokens (12.2 min) + 120k@12 = 1.39M tokens (15.4 min) + start + TF |
| fp8_kvauto_target_only_oracle | 15 | 10.5 | a full server start for 4 greedy rows |
| eagle4_fp8_tp4 | 30 | 25.5 | install 6 + help 1 + start 10 + 8k@8,16 + 40k@8 |
| e_pp4_int4_eager | 20 | 14 | INT4 not prefetched in SMOKE (+2) |
| f_pp4_int4_compiled | 20 | 14.5 | |
| **total** | 175 | **about 169** | **over 150** |

Gate simulation (a step needs `left >= est + 5` min). The steps up to G3 end at 63 min. nvfp4 starts with 87 min left
and ends at 104. kvauto starts with 46 left and ends at 114. EAGLE starts with 35.8 left against 35 needed, so it runs
on the edge and ends at 140. **PP4 eager and PP4 compiled are both SKIPPED_BUDGET.** About 2 min either way decides
whether EAGLE or the PP4 arms are lost. In both outcomes, one of the two risky pieces SMOKE_B exists for does not run.

### FULL (budget 690 min)

| Step | Code est | Realistic | Basis |
|---|---|---|---|
| setup | - | 5 | |
| p5_bench | 30 | 6 | |
| a_merlin_tp1_gpu0 | 75 | 55 | 40k@2,3,4 = 11.7 min, 90k@1,2 = 9.8, 120k@1,2 = 11.3 |
| b_merlin_tp1_x2 | 30 | 24 | start 8, 40k lv4 = 6.9, 90k lv2 = 7.4 |
| c_merlin_tp1_x4 | 35 | 25 | |
| d_merlin_tp4 | **60** | **69** | 40k x3 = 15.3, 90k x3 = 22.3, 120k x2 = 19.3 |
| mtp0_target_only | **50** | **65** | 40k x4 = 16.3, 90k x3 = 19.2 (KV-bound at 12 and 20), 120k x2 = 16.2, TF 2.5 |
| mtp1 | 35 | 27 | 40k x4 |
| mtp3_eq_G0_piecewise | **60** | **69** | as mtp0, plus a profile of 6.8 (40k evicted by 120k) |
| G1 / G2 / G3 | **35 / 35 / 40** | **41 each** | 40k@16,20 = 11.5, 90k@12 = 12.7, profile 6.8 |
| nvfp4_target_only | **55** | **83** | 90k x4 = 29.1, 120k x3 = 27.3, 40k x2 = 13.2 |
| nvfp4_mtp3 | **45** | **80** | same levels |
| int4_mtp3_s20_control | 35 | 38 | 40k x3 = 13.9, 90k@12 = 13.7 |
| A/A iteration (mtp3 rep + merlin x4 + nvfp4 rep) | **50 + 35 + 45 = 130** | **62 + 25 + 80 = 167** | the mtp3 rep runs all 9 levels (L641-643 apply to tagged depth 3 too) |
| e / f PP4 | 30 / 30 | 21 / 21 | start 8 + oracle 1.5 + 40k@8,16 |
| **main sequence (P5 ... INT4)** | **620** | **about 668** | |

Gate simulation, central. INT4 ends at about 668 min with 22 min left. The A/A guard needs 120, so the loop breaks.
**PP4 e needs 35 and is SKIPPED_BUDGET; f is SKIPPED_BUDGET.** The run ends at about 669 min, so about 20 min of the
session is unused and **zero A/A repeats** run.

- Pessimistic (prefill 1,500 / 1,250 tok/s, few cache hits): the main sequence is over 690. nvfp4_mtp3 is truncated
  level by level, and INT4 and PP4 are skipped.
- Optimistic (prefill 2,500 / 2,000 tok/s, starts 7.2 min): the main sequence is about 590 min. 100 min are left, which
  is still below the 120 guard, so there are no A/A repeats. PP4 runs (about 42 min) and the run ends at about 632 min,
  leaving **about 58 min unused**.
- By the code's own estimates: the main sequence is 625 min, the A/A guard breaks, and PP4 e and f run. The run ends at
  about 667 min. So even by its own numbers, FULL never reaches an A/A repeat unless sections fail fast.

The hard stop itself behaves. The watchdog (lab L239-254) kills every live server at 690 min. After that:
- `step()` returns SKIPPED_ABORT (L396).
- `levels()` skips (L487).
- `with_servers` clamps the ready timeout to `left()` with a 60 s floor (L425).
- Streams die on connection reset, so the admission loop returns at lab L556.
- `wait_idle` takes 90 s or less, and the GPU-release gate 180 s or less.

The worst-case tail is about 6 min, or about 13 min in the replica-barrier case (F15). So the notebook ends by about
696-703 min, which leaves about 17-24 min of the 720 for Kaggle's output save. The output is small: the profiler
directories are deleted (L646, L699).

---------------------------------------------------------------------------------------------------------------------

## Findings

### F1 HIGH: FULL does not fit the six sections plus PP4, and the A/A filler never runs
- Where: builder L801-835 (order and estimates), L826-829 (guard).
- Scenario: in the central case, PP4 e and f (plan #1's separate branch) are SKIPPED_BUDGET, there are zero A/A
  repeats, and about 20 min are left idle. Pessimistic: INT4 control and part of NVFP4+MTP3 are lost as well.
  Optimistic: PP4 runs, but about 58 min are left idle. So in no case are all six sections done AND the session filled
  with A/A.
- The underestimates: d (60 vs 69), mtp0 (50 vs 65), mtp3 (60 vs 69), G1-G3 (35-40 vs 41), nvfp4_target_only (55 vs 83),
  nvfp4_mtp3 (45 vs 80).
- Fix, applied together:
  1. Trim the levels the plan does not need. These cut about 110 min:
     - G1/G2/G3: drop `90k@[12]` (L697-698). The plan's #3 measure is the trace verdict plus 40k throughput.
     - `s5(0)`: correctness only (oracle + TF + one `90k@[12]` census). Change L727-731 to
       `o["steady"] = levels(port, "90k", [12]) if spec == 0 else ...` and keep the full sweep for `spec == 3` (see F6).
     - `s6(0)`: change L642-643 so that `depth == 0` gets `90k [12, 20]` only, with no 120k. The 120k band stays on G0,
       on NVFP4+MTP3 and on Merlin.
  2. New order: P5 -> Merlin a, b, c, d -> mtp0 -> **int4 control** (moved up; it also gives PP4 its oracle, F8) ->
     mtp1 -> mtp3/G0 -> G1 -> G2 -> G3 -> nvfp4 target-only -> nvfp4 mtp3 -> **PP4 (compiled, then eager only on
     failure, F7)** -> A/A. If PP4 aborts, only the filler is lost.
  3. Put realistic estimates into `step()`. Use 70 for d, 45 for mtp0 after the trim, 70 for mtp3, 30 for each
     G-arm after the trim, 30 for nvfp4_t after the trim, 85 for nvfp4_mtp3, 40 for int4, 25 for PP4, and 60 / 30 / 85
     for the A/A reps. Then the gate means something.
  4. Replace the A/A loop with a greedy fill (code in F2).
- Result (central): the main sequence plus PP4 is about 578 min. A/A then fills about 90 min (one mtp3 rep and one
  merlin x4 rep) and the run ends at about 665-680 min. A short filler (a 40k-only G0 rep of about 27 min) uses the
  last gap.

### F2 HIGH: the 70-min PP4 reserve is not enforced
- Where: builder L826-832, L487 (levels guard), L400 (step guard).
- Scenario:
  - The reserve is only checked before an A/A iteration (`left() < (70+50)*60`). One iteration is 130 min by the
    code's own estimates and about 167 min in reality.
  - Once an A/A step starts, its `levels()` keep going until `left() < WIN + 300`, which is 7 min.
  - Example: the sections fail fast (say Merlin install fails, so the Merlin block takes about 10 min). A/A starts with
    about 170 min left. The mtp3 rep (62 min) runs, then the nvfp4 rep starts with about 105 min left and runs its
    levels down to about 7 min. PP4 needs 35 and is skipped.
  - Even with the code's estimates, starting at 120 min left: mtp3 rep -> 70 left, merlin rep -> 35 left, nvfp4 rep
    skipped, PP4 e runs with 35 left, f skipped.
- Fix: with PP4 moved ahead of A/A (F1), A/A needs no reserve. Replace L826-832 with:
  ```python
  if not SM:
      AA = lambda i: [(f"mtp3_control_rep{i}", 60, s6(3, tag=f"fp8_mtp3_rep{i}")),
                      (f"merlin_tp1_x4_rep{i}", 30, s1_rep(4)),
                      (f"nvfp4_mtp3_rep{i}", 85, s5(3, tag=f"nvfp4_mtp3_rep{i}"))]
      for i in range(1, 7):
          todo = [(n, e, f) for n, e, f in AA(i) if left() >= e * 60 + 300]
          if not todo or ABORT["why"]:
              break
          for n, e, f in todo:
              step("8_aa_repeats", n, e, f)
  ```
  If PP4 must stay last, then make `step()` and `levels()` use `left() - RESERVE` for every non-PP4 step, with
  `RESERVE = 50*60 if (not SM and DO_B) else 0`.

### F3 HIGH: SMOKE_B needs about 169 min against 150; EAGLE or both PP4 arms are lost
- Where: builder L819 (nvfp4 SMOKE est 25, real about 41), L727-728 (SMOKE levels `90k@12` + `120k@12` =
  2.47M prefill tokens), L811 (mtp0 SMOKE level), L824 (kvauto runs even when EAGLE cannot fit), L834-835.
- Scenario: the central gate simulation above. EAGLE runs on the edge (35.8 min left vs 35 needed) and both PP4 arms
  are SKIPPED_BUDGET. If the steps take about 2 min longer, EAGLE is skipped instead and the 10.5-min kvauto oracle
  was wasted.
- Fix, keeping the user's 150:
  1. SMOKE `s5(0)`: levels `120k@[8]` only. That is 928k tokens and keeps a stock 120k window; 90k correctness is
     covered by the TF 90k row. Set the estimate to 25.
  2. SMOKE `s6(0)`: no steady level (`if not SM:` around L640). The mtp3/G0 step smokes the steady path.
  3. PP4: compiled first, eager only on failure (F7).
  4. Gate kvauto + EAGLE together: `if left() >= (15 + 30 + 10) * 60:` before L824.
- Result: about 131 min, with 19 min spare.

### F4 HIGH: multi-replica Merlin is never smoked
- Where: builder L806-808 (`if not SM`).
- Scenario: per-replica ports and `MASTER_PORT`/`VLLM_PORT` (L249-258), per-port Triton caches, 4 concurrent Merlin
  processes in host RAM, and the `steady_multi` barrier (lab L671-697) all run for the first time in FULL. If they
  fail there, b, c and every A/A merlin rep fail, and plan #1's decisive number is lost after a 12-h commit.
  SMOKE_A has about 36 min spare.
- Fix: in SMOKE run `step("1_tp1x4_pp4", "c_merlin_tp1_x4", 20, s1_rep(4))`. `s1_rep` already runs 40k only in SM, and
  the step costs about 16 min. Place it after `d_merlin_tp4` in SMOKE so that a slow JIT or a fallback skips the
  replica smoke rather than TP4. SMOKE_A becomes about 100 min.

### F5 HIGH: plan #1 "8k/40k/90k correctness": the 90k row is missing, and Merlin has no independent reference
- Where: builder L565, L376-377 (oracle = 2k, 2k, 8k, 40k), L613.
- Scenario: Merlin W4A16 correctness at 90k is only checked by the degenerate-text check under load. TP4 is compared
  with TP1, which runs the same engine and the same quantized weights, so a quantization or engine error that both
  share passes.
- Fix:
  1. Add `MERLIN_P = ORACLE_P + LONG_P + [PROMPTS["90k"][4]]`, use it for both the TP1 key and the TP4 compare
     (`oracle()` gets a `prompts=` parameter), and add `o["teacher_forced"] = tforced(port, key="merlin")` to
     `_tp1_body`. Cost: about 4 min on TP1 and about 1.5 min on TP4.
  2. In CELL6, add `R["merlin_vs_fp8_tf"] = {b: L.compare_tf(TF["fp8_mtp0"][b], TF["merlin"][b]) for b in TF_P}`
     when both exist, labelled "cross-weight drift, not a same-weight oracle". That gives a 40k/90k accuracy number
     against FP8.

### F6 MEDIUM: NVFP4 target-only runs the whole census sweep (extra), the largest single time sink
- Where: builder L727-731 for `spec == 0`.
- Scenario: the plan's order is target-only (correctness gate) -> MTP3 -> 90k census -> more streams -> matched
  throughput. The code runs 9 levels on target-only (about 55 min of levels), and the same 9 levels again on MTP3.
- Fix: target-only = probes + oracle + TF + `90k@[12]`. The census, more streams and matched levels stay on
  `s5(3)` only. This matches the plan and saves about 57 min in FULL.

### F7 MEDIUM: PP4 has two arms (eager + compiled), one more than the plan asks for
- Where: builder L834-835, L750-766.
- Scenario: the plan asks for one TP1xPP4 branch. The code always runs two launches (about 21 min each in FULL).
- Fix: run `s1_pp4(False)` first. Run `s1_pp4(True)` only when the compiled status is LAUNCH_FAILED,
  SERVER_DIED, NO_GO_GARBAGE_OR_ERROR or ERROR, in the same pattern as the TP1 memory fallback. Record it as
  `fallback_eager`.

### F8 MEDIUM: PP4 correctness has no reference in SMOKE, and in FULL whenever INT4 is skipped
- Where: builder L758 (`refs=("int4_mtp3",)`), L822 (`s4_int4` is FULL-only and the last main step).
- Scenario:
  - `runB` shows `int4_mtp3=INCONCLUSIVE_NO_ORACLE`, so SMOKE never compares PP4 with anything.
  - In FULL central, INT4 runs just before PP4, and under the pessimistic case it is skipped. Then PP4 has probes and
    a garbage check only.
- Fix:
  1. Move `s4_int4` to directly after mtp0 in FULL (F1).
  2. In SMOKE, the PP4 arm saves `key="int4_pp4_first"`, and an eager fallback compares against it.
  3. Report "garbage/probe only" explicitly when no INT4 reference exists.

### F9 MEDIUM: plan #4 "Merlin low-M patches, INT4 separately" A/B is confounded and only partly matched
- Where: builder L609-623 vs L737-746.
- Scenario: Merlin TP4 (Merlin checkpoint, Merlin profile, seqs 24, capture to 120) is compared with stock (RedHat
  checkpoint, seqs 20). The levels match only at 40k@8, 40k@16 and 90k@12. A tok/s difference mixes checkpoint, engine
  patches and config.
- Fix:
  1. Give `s4_int4` the Merlin TP4 levels (`40k [8,16,24]`, `90k [8,12,16]`, seqs 24).
  2. Label the row in CELL6 "engine + checkpoint A/B".
  3. If stock vLLM 0.32 can load the Merlin checkpoint (compressed-tensors), serve that checkpoint in `s4_int4`
     instead, so that only the patches differ.

### F10 MEDIUM: the per-level guard ignores prefill, so the last level can be killed mid-admission
- Where: builder L487 (`left() < WIN + 300`).
- Scenario: a 90k@24 NVFP4 level needs about 24 min of prefill plus the window, but starts when only 8 min are left.
  The watchdog kills it at 690 min and nothing is measured. In SMOKE the same happens to the last 90k/120k level.
- Fix: `need = WIN + 90 + min(3600, n * BANDS[band] / 1500)` and skip when `left() < need + 120`. Record
  `SKIPPED_BUDGET` with `need_min`.

### F11 MEDIUM: the TP1 admission cap assumes 700 tok/s prefill
- Where: lab L528 (`prefill_tok_s=700`), builder L566-568, L600-604.
- Scenario: in replicas b and c, 90k@2 per fresh replica is 180k tokens against a 420 s cap. It settles only if
  one-L4 prefill is at least about 440 tok/s; below that, the decisive "streams per replica at 90k" window is
  INCONCLUSIVE_NO_SETTLE.
- Fix: pass `prefill_tok_s=300` in the TP1 calls. `levels(..., prefill_tok_s=300)` and `L.steady_multi(...,
  prefill_tok_s=300)` already forward `**kw`. This only raises the cap and costs nothing when prefill is fast.

### F12 MEDIUM: A/A repeats relaunch configs that already failed in the main pass
- Where: builder L830-832; the `s5` gate L715 checks target-only only.
- Scenario: if nvfp4_mtp3 LAUNCH_FAILED, or c_merlin_tp1_x4 or G0 failed, each A/A iteration spends about 10 min on
  the same failing launch.
- Fix: in the A/A list, keep a rep only if the main step's status is DONE. For example,
  `R["sections"]["5_nvfp4_kv"].get("nvfp4_mtp3", {}).get("status") == "DONE"`, and the same for
  `1_tp1x4_pp4/c_merlin_tp1_x4` and `6_mtp_depth/mtp3_eq_G0_piecewise`.

### F13 MEDIUM: the decisive and matched levels are capped by fixed lists
- Where: builder L567 (TP1 90k `[1, 2]` while `seqs = 4`), L614-615 (TP4 Merlin levels fixed).
- Scenario:
  - If one L4 holds 3 streams at 90k, the decisive number reads 2. At FP8 KV that means 8.05 GB / 32 KB = 245k
    tokens, so 2 is likely but not certain.
  - TP4 has no level equal to 4 x the replica level when `lv40 = 3` (12) or `lv90 = 1` (4).
- Fix:
  1. TP1 90k `[1, 2, 3]` in FULL. A KV-bound 3 costs about 3 min and returns CAPACITY_LIMITED.
  2. In `s1_tp4`, use `sorted({8, 16, 24, 4 * _tp1_level("40k", 2)})` and `sorted({8, 12, 16, 4 * _tp1_level("90k", 1)})`.

### F14 LOW: rev 3b's 116k band is unvalidated in any fake run
- Where: `runB3.log` (PORT_BUSY 8200 because `runF` ran on the same host).
- Scenario: `runA`, `runB` and `runF` are rev 3, where 120k windows ended in the window (`runF`: 120k@1/2, 120k@8/12
  INCONCLUSIVE_STREAMS_ENDED_IN_WINDOW). The 116k fix and every rev 3b path are unproven locally.
- Fix: rerun fake SMOKE_A and SMOKE_B serially with `nb3b_*`. Run one fake FULL with `E23_BUDGET_SEC` of about
  300 min so that the A/A loop and the PP4 ordering execute at least once.

### F15 LOW: one dead replica stalls the other replicas at the barrier
- Where: lab L556-558.
- Scenario: the `INCONCLUSIVE_STREAMS_ENDED_IN_ADMISSION` return does not call `barrier.abort()`. The surviving
  replicas wait up to `admit` (420-451 s) and then report INCONCLUSIVE_REPLICA_BARRIER. At HARD_STOP this adds about
  7.5 min to the shutdown, which still fits inside 720.
- Fix: add `if barrier: barrier.abort()` before that `return r`.

### F16 LOW: step skips are not logged to the console
- Where: builder L396-403.
- Scenario: `runF` shows `d_merlin_tp4` SKIPPED_BUDGET only in the JSON. The console goes straight from `c` to
  `mtp0`, so a reader of the Kaggle log cannot see the skip.
- Fix: `log("---", sec, name, rec["status"], rec.get("left_min"), rec.get("est_min"))` in both skip branches.

### F17 LOW: plan items not smoked (engine risk only; the code path is shared and fails closed)
- G1/G2 (TRITON_ATTN FULL graphs + MTP on the hybrid), NVFP4 + MTP3, and K1. A failure costs at most one launch each
  in FULL, and F12 stops the A/A waste. No change is needed if F1 to F4 are applied. Optionally, SMOKE_B could run
  `s5(3)` with `90k@[12]` if time remains after F3.

### F18 LOW: FULL prefetches INT4 (17 GB) at t = 0
- Where: builder L797.
- Scenario: INT4 is first used at about 9 h, or about 4 h after the reorder. The read competes on GCSFUSE with the
  Merlin extract and the stock install at the start, and the pages may be evicted by the time INT4 is used.
- Fix: start `prefetch_bg([INT4], "int4")` right before the step that needs INT4.

### F19 LOW: DESIGN.md times are stale
- DESIGN.md L7-16 says "about 5.5 h" and gives 110/15/60/20/45/75 min per section. Those figures come from before
  max-len 131072, the 120k band, 120 s windows and the A/A loop. Update the table with the realistic column above.

---------------------------------------------------------------------------------------------------------------------

## What is correct

- Every one of the user's six sequences has code, in the plan's own order: #1 a -> b -> c -> d, #3 G0 -> G1 -> G2 -> G3,
  #5 control -> target-only -> MTP3. The exceptions are the F5 90k row and the F6/F7 extras.
- P5 runs first with all four GPUs free and covers oracle, reduction, graph capture and the separate INT8 arm. vLLM
  integration is left for later, as asked.
- P4 benchmarks the same weights and scales on the five real TP4 shapes, numeric match first. It runs on GPU 3 in
  parallel with TP1 on GPU 0, and the join is bounded (L553). The GPU-release gate is per GPU (L455, L558).
- #60532 is applied with a dry-run, its revert is checked by sha, and a failed revert aborts the lab.
- EAGLE-4 is SMOKE-only (feasibility), as decided. The FULL switch is an env var a commit cannot set.
- The modes are baked per notebook, with budgets 120 / 150 / 690. `MAX_LEN` is 131072. The 120k band starts at 116k,
  leaving 14,560 new tokens; at real decode rates a stream decodes no more than about 10k tokens by the end of the
  window, so streams do not end inside the window.
- The hard stop works. The watchdog kills the servers. `step` and `levels` stop on ABORT. `with_servers` clamps its
  ready timeout. The admission loop exits on dead streams. The cleanup tail is 6 min or less (about 13 in the F15
  case), so FULL ends by about 700 min and leaves 17-24 min of the 12 h for output save. The outputs are small.
- `step()` never marks PASS by default. Every skip is recorded with `left_min` and `est_min` (seen in `runF`).
- SMOKE_A (about 84 min central) fits 120 with room for the F4 replica smoke. The SMOKE estimates for mtp0, mtp3, G3,
  EAGLE and PP4 are realistic or slightly generous. Only nvfp4 is low.
