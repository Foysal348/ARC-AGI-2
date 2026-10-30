# E24 build and local validation (2026-10-30)

**Status:** built and validated locally; not run on Kaggle. No Kaggle CPU or GPU run has happened, and every Kaggle run needs the user's go.

**Two validation rounds:**
1. **First round.**
   - unit tests;
   - end-to-end runs of all three notebooks against fake SGLang 0.5.21 / vLLM 0.32 servers, with injected faults;
   - two independent read-only reviews (decision logic, real platform).
2. **Second round** (asked for as "more validation").
   - an independent re-review of the first round's fixes;
   - an independent conformance audit of the code against DESIGN.md (about 250 checkable items);
   - a Monte Carlo study of the verdict rules under the null;
   - property-based fuzz tests;
   - a mutation check of the decision tests;
   - four new end-to-end fault scenarios.

Every finding was checked against the code and the source evidence, then fixed or listed in section 6.

## 1. Artifacts

| File | sha256[:8] | Size |
|---|---|---|
| `Experiment Notebooks/Q27-XHIGH-E24_SGLANG_SPEED_SMOKE.ipynb` | 49e977eb | 383 kB |
| `Experiment Notebooks/Q27-XHIGH-E24_SGLANG_SPEED_FULL_A.ipynb` | 34ce7335 | 383 kB |
| `Experiment Notebooks/Q27-XHIGH-E24_SGLANG_SPEED_FULL_B.ipynb` | 031cda2a | 383 kB |
| `Experiment Notebooks/Build/build_e24_notebook.py` | ce14cbe1 | |
| `Experiment Notebooks/Build/cpu_kernel_e24/kit_data/q27x_e24_kit_4d2a1f7db812.bin` | 4d2a1f7d | 460,838 B |
| `src/e24_lab.py` | 52d0d742 | |
| `src/e24_hook.py` | 24b4b478 | |
| `src/e24_sglang_patches.py` | 96a97251 | |
| `src/e24_unit_41736_41737.py` | 86cb257c | |

**Notebook size.** The notebooks carry the cells without comments or blank lines. The builder keeps the commented source, and it asserts that every cell's Python AST is unchanged by the stripping, so the notebook runs exactly the reviewed code. That leaves about 17 kB under the builder's 400,000-byte assert; Kaggle commits above about 407-444 kB never start.

**CPU kit.** The kit notebook pins the `.bin` by sha256. The `.bin` must be uploaded as a dataset version before the kernel is pushed. Rebuild the kit after every notebook rebuild.

## 2. Validation results

| Check | Result |
|---|---|
| `src/test_e24_decisions.py`: the notebooks' verdict logic (cells read with ast, real lab), including 6 seeded property-based fuzz tests | 40 / 40 OK |
| Mutation check: each of 27 review fixes reverted one at a time | 27 / 27 caught |
| `src/test_e24_lab.py` | 67 / 67 OK |
| `src/test_e24_hook.py` (real 0.5.21 source) | 25 / 25 OK |
| `src/test_e24_patches.py` (real 0.5.21 source) | 28 / 28 OK |
| `src/harness/test_fakes.py` | 49 / 49 OK |
| `src/harness/mock_inputs.py --selftest` | 26 / 26 |
| Backport semantic checks, pristine vs patched 0.5.21 | 28 / 28 |

**FAKE end to end.** The runner is `src/harness/run_local_e24.py` at time scale 0.02, with its own ports and a per-arm fault wrapper.
- **run4** used the round-2 notebooks: 3 variants and 14 fault scenarios, 148 / 150 checks. Both failures came from the launch_streak scenario itself, which still used fast exits after the guard was changed to count only startup timeouts. The scenario now uses startup hangs, and the guard tells configs apart by flags plus the arm's declared env.
- **run5** used the final notebooks, which differ from run4's only in that one guard line: 52 / 52 checks.

| Run | Checks | What it showed |
|---|---|---|
| Build | 7 / 7 | every notebook < 400 kB; identical code except the baked MODE / PART; the code compiles; `plan_argvs_e24.json` |
| SMOKE (run4, run5) | 30 / 30 | Nine arms GO. The pre-flight now runs after M1 and right before M3. The section-6 arms read INCONCLUSIVE, as they must in FAKE: no GPU unit tests, no real allocator. |
| FULL_A | 30 / 30 | Every section-1 arm GO. NO_HIGH_MARGIN_LINK with no arm verdicts or notes, so there are no false findings on clean data. Pass 2 is exercised on the r2 boots. The cold / warm controls are OK. There is no empty section-3 table. |
| FULL_B | 27 / 27 | **Section 2:** all GO, and no `best` (FAKE boots differ by 6 %). **Order:** the A/A rounds now run before `s4_lowcap` and `s5_mixed`. **Section 4:** the low-cap census reports 300,000 tokens, KV-bound. **Section 5:** the chunk arms read STOP_STARVATION (expected, DESIGN §4.5). **Report:** no section-1 block. |
| crash_start (run4, run5) | 3 / 3 | One arm fails fast; the rest run, and the launch guard does not trip. |
| hard_stop (run4, run5) | 5 / 5 | The watchdog fires, later arms are SKIPPED_ABORT, and no arm reads ERROR_*. |
| diverge | 6 / 6 | c ↔ d6: 20 STATE_FLIPs, Fisher p ≈ 6e-14, DECISIVE_PAIR_FOUND. `s1_d6` reads NO_GO_STATE_FLIP. |
| leak_cache | 4 / 4 | `s1_d4` is INVALID_CACHE_ISOLATION; the other arms stay ISOLATED. |
| mamba_assert | 4 / 4 | The capacity arm reads NO_GO_ENGINE_ERROR; the challenger without control rows reads INCONCLUSIVE_PARITY. |
| flag_unsupported | 3 / 3 | NO_GO_FLAG_UNSUPPORTED; never launched. |
| logprob_oom | 4 / 4 | The armed control proves the fault; the chunk env keeps every scorer alive. |
| long_prefill_census | 5 / 5 | No early plateau during a 90k prefill. |
| tokenizer_mismatch | 3 / 3 | Aborts before any launch. |
| startup_gate (run4, run5) | 3 / 3 | NO_GO_STARTUP_600; measurements kept. |
| alloc_warning (new) | 4 / 4 | The allocator's recovered-OOM warning after every flush, inside the census, never fails the census, the window or the arm. |
| launch_streak (new, run5) | 4 / 4 | Startup timeouts of three distinct configs stop SGLang; later SGLang arms are SKIPPED_DEPENDENCY; the vLLM arms are GO. |
| restore_corruption (new) | 5 / 5 | Restore corruption is caught by `s1_d6`'s own pass 1 ↔ pass 2 (NO_GO_STATE_FLIP). The restore-state-matched c16 A/A stays clean, and no other arm is affected. |
| grid_text (new) | 3 / 3 | Digit-grid output (which the old rule called LOW_VARIETY) keeps the window STEADY and the arm GO. |

No fake process was left behind.

**CPU kit dry run** (on the Mac, with mock mounts, the bootstrap and the sha-pinned `.bin`; final build). These checks pass through the kit's own path:
- SMOKE: every runner check, plus input discovery, real-tokenizer id equality and checkpoint identity;
- the crash_start (3 / 3) and alloc_warning (4 / 4) fault modes;
- all five unit-test files on the Linux code path;
- the `SGLANG_*` environ check.

Only the four checks that need the Kaggle venvs cannot run on a Mac, as in WP5's dry run: the NGRAM C++ build and the real SGLang / vLLM parser flag checks. They run on Kaggle.

**What FAKE mode cannot prove.** The fakes reproduce 0.5.21's log formats, metrics, endpoints, radix / Mamba-checkpoint behaviour, NCCL kernel naming and fault modes. They do not reproduce:
- real numerics or real timing;
- NCCL;
- CUDA graphs;
- the hook on real CUDA processes;
- real logprob memory pressure;
- the Kaggle installs. The CPU kit covers those.

The fakes also carry fixed per-step process overheads that do not shrink with the 0.02 time scale. That is why some FAKE windows and loads read INCONCLUSIVE (section 7).

## 3. Statistical validation of the verdict rules (null Monte Carlo)

The study asks how often a rule names a finding when there is none. Scripts: `stats/null_rates.py` and `stats/fisher_rule.py` in the session scratchpad.

**Section 1, the decisive pair.**
- **Setup.** 25-row pairs. A/A per arm from c16 (16 rows), plus boots (25 rows) for c and d6. Under the null, pairs and A/A share the same per-row FLIP rate p.
- **Result.** The EMS rule alone ("count > A/A + 1") names a false decisive link in a sizeable share of null chains. Adding a one-sided Fisher exact test of the pair against the pooled pure A/A, at α 0.05 per link, cuts that to about 5 % while keeping most of the power.

| FLIP rate p | False decisive (EMS) | False decisive (EMS + Fisher) | Power at +30 % FLIP rows (EMS / EMS + Fisher) |
|---|---|---|---|
| 1 % | 6.0 % | 0.4 % | 99 % / 98 % |
| 3 % | 15.8 % | 2.9 % | 97 % / 96 % |
| 5 % | 18.3 % | 4.9 % | 96 % / 94 % |
| 10 % | 22.0 % | 6.8 % | 92 % / 88 % |

Section 1 now reports DECISIVE_PAIR_FOUND only for a significant link. A link that passes the EMS rule alone reads DECISIVE_PAIR_SUGGESTIVE. FAIL_HIGH_MARGIN is still computed as DESIGN defines it, and STATE_FLIP and garbage links need no test. The NO_GO drift and restore verdicts, which are FLIP counts, also need p < 0.05; otherwise they are recorded as notes.

**Section 2, the best challenger.**
- **Setup.** Four challengers with no real difference.
- **Result.** With the selection-time spread from one A/A boot (2 points), "best" is wrong 13.5 % of the time. Each reproduced A/A-round repeat cuts this: 1.5 % with one repeat, 0.13 % with two, 0.01 % with three. A confirmed `best` now needs at least one reproduced repeat.

## 4. Independent reviews: findings and fixes

### 4.1 First round
**Decision logic, 12 findings, all fixed.**
1. Section 2 `best` ignored every gate.
2. Section 2 never used its confirmation data.
3. Section 5 said NOT_WORSE from partial loads.
4. Section 1 turned restore and warm-cache errors into A/A allowance.
5. STATE_FLIP NO-GO used the wrong allowance and never reached the arm status.
6. Unevaluated chain links counted as clean.
7. A T_pair above 4 hid STATE_FLIPs.
8. Section 6 dropped or under-tested patched arms.
9. Patched NaN / sentinel arms FAILed on one noise FLIP.
10. Hard-stop NO-GOs stood.
11. Section 3 confirmation arms were always INCONCLUSIVE.
12. Lower-severity items.

**Real platform, 6 of 7 fixed.**
- **#1 (high).** The PyTorch allocator's recovered warning, `CUDACachingAllocator … memory allocation failed with OOM`, was read as FAIL_OOM (EMS BASE S20 log, line 1062). It would have failed every section-4 census.
- **#2.** The NCCL_PROTO env proof always failed. Since NCCL 2.19 kernel names carry the specialised variant: the real E14A trace shows `ncclDevKernel_AllGather_RING_LL` for an 11.9 MB all-gather.
- **#3.** Digit grids read as garbage under load.
- **#4.** Nothing stopped repeated failed launches.
- **#6.** Raw rows lived only in /tmp.
- **#7.** A save race.
- **#5 (not fixed).** SMOKE probes of the FULL-only paths: see section 8.
- **Also found from a FAKE run.** The steady-window admission budget left the 120k band about 40 s to settle at the real 116k prefill rate.

### 4.2 Second round: re-review of the fixes (8 confirmed or plausible, all fixed)

| # | Finding | Fix |
|---|---|---|
| 1 | Section 2 `best` stayed CONFIRMED when the winner's own A/A-round repeat crashed or failed a gate, and also with no evaluable repeat. | A repeat that ran and failed a gate is NOT_REPRODUCED_GATE. `best` needs at least one REPRODUCED repeat; with none it reads UNCONFIRMED_NO_REPEAT. |
| 2 | `s6_safety` ignored flush validity: a false FAIL after a refused flush, and a masked leak (false CLEAN) when the baseline flush was refused. | The baseline flush counts. A leak needs a valid flush. A reading above the baseline marks the baseline invalid. A leak FAILs only when the safety status is OK; a stale restore FAILs regardless. |
| 3 | `s6_combined` missed a STATE_FLIP, and read FAIL on a dead server. | A STATE_FLIP beyond its allowance gives FAIL. A dead server or missing rows give INCONCLUSIVE before the tool-probe check. |
| 4 | A clean A/A with one timed-out row was dropped, so a cleaner A/A gave a weaker verdict. | Any judged rows count. |
| 5 | The launch guard tripped when three arms shared one config (s6_ffu / s6_ffp / s6_np). | It now counts only startup timeouts of three distinct argvs. |
| 6 | Section 5 never checked the control's own gates. | Challenger verdicts need a CONTROL (or starving-control) verdict. |
| 7 | Some consequences of a watchdog kill still stood after a hard stop. | NO_GO_TOOL_PROBE is treated as a dead-server gate. Error signs are counted only before our SIGTERM. |
| 8 | The report deleted the per-arm raw snapshots even when the tar had skipped files. | It keeps them in that case. |
| 9 (low) | The boot A/A used c16a rows without checking the flush. The FULL_B report printed a section-1 block, and FULL_A an empty section-3 table. | Both fixed. |

### 4.3 Second round: DESIGN conformance audit (about 250 items, 231 conforming)
**The six material deviations, all fixed:**
1. A pre-flight failure dropped M1, self, pass 2 and M2, and later arms ignored it. Now the pre-flight runs right before M3 (DESIGN §3.1), and a failure skips only M3 / M4, on that arm and on every later one.
2. Equal N was not enforced on the streams actually running. Gain claims now need STEADY windows; long bands may be CAPACITY_LIMITED, since running fewer streams is the collapse being checked.
3. The optional `s5_mixed` (and `s4_lowcap`) ran before the A/A rounds and ate their time. They now run after, and a time-short A/A round no longer `break`s out of the plan.
4. DESIGN §4.6 "identical counters" was not enforced. A positive section-6 verdict now needs the verdict counters to agree on all four ranks; otherwise it reads INCONCLUSIVE_COUNTERS_DESYNC.
5. The isolation proof missed new `/tmp/torchinductor_*` directories. The default directories are now globbed again after the arm.
6. The section-6 estimates were 41 min above DESIGN §5. They are aligned.

**Minor items fixed:**
- pass 2 added on the r2 boots (DESIGN M1);
- the `s4_lowcap` census-cap check;
- M4 `s1_c` vs `s1_c32` reported as the fp16-accumulation effect, not as a verdict (DESIGN §4.1);
- the KV-pool report column.

Minor items left as documented limits are in section 7.

### 4.4 Second round: what the new FAKE scenarios found
- **alloc_warning, launch_streak, grid_text** confirmed three fixes end to end.
- **restore_corruption** found a real design point. A stale restored state was injected only into requests that resumed from a cached prefix. The ARC prompts share a 64-token header, so "cold" rows after the first are partial restores, and which rows hit depends on admission order.
  - **What it means for the comparisons.** The production d6 path really does diverge from target-only, so the c ↔ d6 finding is correct. But c16a and c16b differed in which rows hit, which added restore differences to the A/A allowance.
  - **Fix.** The A/A now compares only rows whose cached-token count is equal in both runs. Cross-arm pairs keep the production path.
  - **Coverage.** The scenario's own pass 1 ↔ pass 2 verdict caught the stale restore (NO_GO_STATE_FLIP), and no other arm was affected.

## 5. Fuzz and mutation testing
- **Fuzz.** Six seeded property-based tests run the verdict functions of sections 1, 2, 3, 5 and 6 and the hard-stop relabel on random, partial and broken records. The generators start from valid records and corrupt 0-2 fields, so every outcome is exercised; in 300 section-2 draws, for example, there were 160 candidates and 27 confirmed bests.
  - Properties checked include:
    - a `best` is eligible, better, holds its long bands and has a reproduced repeat;
    - NOT_WORSE means every load is usable on both sides with a valid control;
    - `best_two` arms are GO and faster than the control;
    - the section-1 verdicts are idempotent;
    - a hard stop never leaves ERROR_ or dead-server statuses, while a NO-GO measured before the stop stands.
  - The fuzz coverage analysis also found that a slow launch turned the section-5 verdict into STOP; it now only marks adoption as blocked.
- **Mutation.** 27 mutants each revert one fix in a copy of the builder. Every mutant makes its named test fail. Two first-round tests had become vacuous behind the new repeat rule; they were sharpened.

## 6. Departures from DESIGN rev 2

**WP4:**
- the lab is embedded as gzip+b64;
- section-5 windows are preset at 1,100 tok/s + 30 s;
- FLIP self teacher forcing on vLLM arms runs only in SMOKE;
- c16 uses prompt variants 0-3 and 9-20;
- the cross-weights link is STATE_FLIPs minus the A/A's, with no +1;
- M3 also runs on d6fr, d6_r2 and c_r2;
- T_pair is the max of the per-trace q99.9;
- the A/A round repeats the top challenger;
- an unevaluable W5 / W7 rule skips both arms.

**Integration and reviews:**
- **Flag check.** `resolve_once` runs only in SMOKE and is advisory.
- **Status rules.** An engine death after a known error sign reads NO_GO_ENGINE_ERROR; an arm stopped by the hard-stop watchdog reads INCONCLUSIVE_HARD_STOP.
- **A/A (§3.2).** Pure repeats only (c16 after flushes, r2 boots), compared only on rows with an equal restore state. The allowance is scaled to each pair's judged rows: ceil(rate × rows).
- **Significance (§3.2 / §4.1).** A one-sided Fisher test at α 0.05 qualifies the decisive pair (FOUND vs SUGGESTIVE) and gates the FLIP-count NO-GO verdicts.
- **NCCL (§4.3).** NCCL_PROTO is proven by its log line.
- **Section 2.** The threshold is the A/A spread, gain claims need STEADY windows, and a confirmed `best` needs a reproduced repeat.
- **Section 4.** A shared-load prefix-recompute gate.
- **Section 5.** A slow launch blocks adoption but not the measured verdict.
- **Section 6.** `s6_ffpd` runs the safety cases (estimate 9 min).
- **Steady windows.** The admission budget uses 1,000 tok/s.
- **Notebook packaging.** The cells carry no comments (AST-verified).

## 7. Open risks and documented limits
1. **First Kaggle contact.**
   - The CPU kit has never run on Kaggle; the NGRAM C++ build (g++ 11.4) and the CPU parser checks are unproven there.
   - Several real-engine paths first run inside FULL: eager DFlash with NaN injection, NGRAM, fp32 SSM, `--disable-radix-cache`, PDI and mixed chunk, W5 / W7, the 116k and shared census, `s4_mem086`, section-5 D0 / D1 / D3, and 120k-context teacher forcing.
2. **Statistical limits.**
   - Moderate effects stay hard to see with 25-row pairs: power is only 20-70 % at +15 % FLIP rows.
   - The section-2 `best` depends on the A/A rounds getting time. Without one, there is no confirmed best.
3. **Header restores.** The production SGLang path restores the shared 64-token header, so cross-arm pairs include short restores by design. Deep restores are tested by pass 1 ↔ pass 2.
4. **PDI arms.** They can read INCONCLUSIVE_INJECT_NO_FIRST_TOKEN if their slower prefill misses the preset D3 window (about 306 s against 357 s at PDI 4).
5. **Documented limits, not implemented.**
   - The XL-collapse check uses tok/s only; running p50 is reported.
   - The profile has no separate drafter category, because drafter GEMMs are not separable by kernel name.
   - M4 entropy needs top-k logprobs; NLL is reported instead.
   - SGLang 0.5.21 ignores `sampling_seed` without deterministic inference, so M4 and the M2 T=1 batch are not seed-reproducible on SGLang. Each arm scores its own traces, so the verdicts do not depend on it.
   - The vLLM arms have no tool-call probe (their argv has no parser).
   - Shard sha256 is recorded, not gated.
6. **Timing.** FULL_A ran 10-15 min over its estimate in FAKE (WP4). The hard stops protect the 12 h session.
7. **LB-gap premise.** E15P's 26.94 is still unverified (submission 57644341). If it holds, the gap is 3.2-5.5 points, not 6-7 (DESIGN §5.1).

## 8. Next steps (each needs the user's go)
1. **CPU kit.** Upload `cpu_kernel_e24/kit_data` as a dataset version, then push the kit kernel (CPU, no GPU quota).
2. **SMOKE.** 4× L4, 210 min hard stop, about 6 quota-h. Its machinery summary decides whether FULL can start.
3. **FULL_A and FULL_B.** About 14 and 21 quota-h. They need two accounts within one week, and the inputs shared with both.
4. **Optional, before FULL** (first round §4.1, real platform #5): add launch-plus-one-request probes of the FULL-only argvs to SMOKE, on warm cache copies. Cost: about 20-50 min of SMOKE.
5. **After FULL_A.** Build the paired ARC sample session (DESIGN §5.1).
