# Audit request: E25 NB4 TICK, the stage-0 stall arms (2026-11-01)

All files in this zip are flat (no folders). Read every file in full before you answer. `00_ZIP_MANIFEST_SHA256.txt` lists each file with its source path and sha256.

The facts in this prompt describe the build at the time of writing. If a file in the zip says something different, the file is authoritative. Report the difference as a finding.

## What you are auditing
- **The notebook:** `Q27-XHIGH-E25_NB4_TICK.ipynb`. One GPU commit run on Kaggle 4×L4, hard stop 150 min. Its sha256 and size are in `nb4_build_facts.json` and the zip manifest.
- **Code map:**
  - builder: `build_e25_nb4.py`. It replays the NB0 / NB1 R2 builders, checks their cells byte for byte against the Kaggle-run notebooks, then adds NB4's cells 7 and 8.
  - pure helpers: `nb4_lib.py` (arms, read-back, labels, winner rule, planner, core timer plugin);
  - every notebook cell as one file: `nb4_code_export.py`. Audit this file, because it is what runs;
  - build facts: `nb4_build_facts.json` (pins, hooks, arms at build time).
- **Hook modules:**
  - `nb4_timeline.py`, the in-graph stall classifier, embedded in the notebook (sha-checked). Tests: `nb4_timeline_tests.py`.
  - `nb4_ppack.py`, the Q-pack / Q-defer patch of Merlin's V2 model runner (`pp_utils.py` + `model_runner.py`), embedded. Tests: `nb4_ppack_tests.py`.
  - `nb4_p5_engine.py`, the P5 host-staged TP all-reduce patch of `cuda_communicator.py`, plus its backend, gate and worker-extension sources. Tests: `nb4_p5_tests.py`.
- **Payload dataset:** `foysalemonshanto/e25-nb4-payload` holds `nb4_p5_engine.py` and `p5_src/` (`build_host_ar.py`, `host_ar.cu`, `host_ar_binding.cpp`, `host_ar_check.h`, `host_ar_layout.h`). In the zip they are flat; `payload_SHA256SUMS.txt` keeps their dataset paths. The notebook pins them (`NB4_PIN.hooks`, `P5_SRC_SHA`).
- **Unit tests:** `nb4_unit_tests.py` has 39 tests, including a CELL 8 flow test with a stub boot runner.
- **Spec and research:**
  - `NB4_TICK_SPEC_2026-11-01.md` is the user's request. The lead's later addenda changed the arm list, as described below.
  - `NB4_RESEARCH_graphs_2026-11-01.md` covers the source of the FULL graph, commit graph, PP broadcast and P5 wiring.
  - `NB4_RESEARCH_folders_2026-11-01.md` and `NB4_RESEARCH_conversations_2026-11-01.md` are sweeps of our past runs and decisions.
- **Results context:** `E25_GPU_NB1_R2_2026-10-31.md` (the stall analysis), `E25_GPU_NB5_QUALITY_2026-11-01.md` (FP8 KV accuracy) and `PLAN_1500.md` (the target and the standing rules).
- **CPU kits:**
  - builder: `build_e25_nb4_cpu_kit.py`;
  - kit notebooks: `q27-e25-nb4-kit-real.ipynb` and `q27-e25-nb4-kit-units.ipynb`, with their `*_kernel-metadata.json` files;
  - the NB4 kernel metadata: `q27-xhigh-e25-nb4-tick_kernel-metadata.json`;
  - kit reports `nb4_kit_real_report.json` / `nb4_kit_units_report.json`, if present. A report is in the zip only when the notebook sha it tested equals this notebook. If a report is missing, the kit has not run on Kaggle yet: judge the kit code instead.

## Context
- **Hardware and competition:** Kaggle ARC-AGI-2 on 4×L4 (24 GB each, sm_89, PCIe, no P2P).
- **Model:** Qwen3.8-Flash-Next W4 (Merlin checkpoint in 5 parts + FP8 PLE overlay).
- **Engine:** Merlin v3.0.0, a vLLM 0.32 fork, at TP2×PP2+EP with MTP K3.
- **Target:** a whole-run average of 1500 tok/s (`PLAN_1500.md`). Speed numbers count only as engine end-to-end measurements. Microbenchmarks never count (PLAN §0).

### The stall (offline, three independent readings of the NB1 R2 B1 traces)
- The stage-0 FULL graph stops at node #173 for about 21.7 ms per tick.
- It resumes a few µs after the PP sampled-token broadcast of the **other** micro-batch ends.
- The next graph has no data dependency on that receive, so this is a false dependency (hardware queue or stream ordering), not a data wait.
- It occurs in 39 of 39 steady ticks.
- Decode already runs FULL graphs; the stall is inside the graph, not host or scheduler time.
- See `E25_GPU_NB1_R2_2026-10-31.md` (re-analysis section), the `nb4_ppack.py` docstring, and the research notes.
  - Some files quote "kernel 172 of 1,036" and "36 of 36 ticks" from an earlier pass.
  - Treat these as the same stall. Report any real inconsistency.

### NB1 R2 B3 = the base configuration, measured
- 8k@8: 254.6 tok/s at tick 48.3 ms. 40k@8: 249.5 tok/s at tick 49.9 ms.
- Pool: 214 blocks (pin 3,553,580,524 B).
- The recorded windows are in `nb4_build_facts.json` (`nb1r2_b3_windows`).

### NB5 QUALITY (FP8 vs BF16 KV on 15 ARC tasks × 2 attempts)
- FP8 solved 13 / 15, the same as BF16 (13 / 15).
- FP8 gives 1.9× the pool and 9 workers vs 5.
- This is why the base is FP8 and seqs is 9.

### Base (every arm = base + exactly one change)
- W4 Merlin, FP8 E4M3 QSA KV with static scales, pin 3,553,580,524 B (214 blocks).
- MTP K3, `--language-model-only`, #60490 applied.
- seqs 9, CAPS16 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64].
- NB1's read-only worker extension plus NB4 methods, dev mode.
- `--cudagraph-metrics`, `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`, and the idle torch profiler config.
- The flag set is identical on every arm, T0b included.

### Arms, in run order (as built; verify in `nb4_lib.make_arms` / `next_arm` and CELL 7 / 8)
| order | arm | the one change | proof that it engaged |
|---|---|---|---|
| 1 | T0 | Host timeline: profiled windows 8k@8 and 8k@9 on all 4 workers, vLLM's iteration log, the engine-core step timer plugin. Booted with `VLLM_RECOVERSSM_COMMIT_GRAPH=graph` and flipped to eager before any measurement, so T0's windows are the base. **C** is a same-boot runtime A/B (eager vs graph commits) inside T0. | 4 rank traces analysed by `nb4_timeline.analyze`; C: commit replays > 0, greedy graph vs eager EXACT |
| 2 | S | `CUDA_DEVICE_MAX_CONNECTIONS=32` | the env in every worker; in-graph stall > 1 ms in ≤ 5 % of steady graphs |
| 3 | S2 | Q-pack: one packed PP broadcast instead of three (`NB4_PPACK=1`; patch applied before the boot, reverted after) | 4 `NB4_PPQ` init lines + packed send / recv counters; the stall gone |
| 4 | S3 | Q-defer: stage 0 posts the parked receive behind the NEXT forward launch (`NB4_PPDEFER=1`; same patch file) | init lines + defer counters; the stall gone |
| 5 | G | capture list [1..9, 12, 16, 20, 24, 28, 32, 36], so every n = 1..9 is FULL at seqs 9; diagnostic; decided on the level-9 windows | 36-token steps FULL > 0, 0 eager decode rows in 8k@9 / 40k@9 |
| 6 | P | P5 world-2 host-staged all-reduce in the TP all-reduce, ≤ 184,320 B, NCCL above. Diagnostic. The `.so` is rebuilt in the Merlin venv, then a 4-process gate runs with the GPUs free, then a strict patch; it is reverted after. | gate PASS, 4 inits, P5 calls > 0 |
| 7 | T0b | the base again, 2 replicates of every window (A/A) | — |
| opt | K4, INT8 | MTP K4 with its capture list; `DRAFT_HEAD=int8` | only if the budget allows |

- **Per arm:**
  - read-back, fail closed (`CONFIG_MISMATCH` / `NOT_ENGAGED` stop the body);
  - probes and sampled output;
  - greedy oracle and TF 2k / 8k / 40k vs the base;
  - windows 8k@8 / 40k@8 / 8k@9 / 40k@9 (90 s each), with the `--cudagraph-metrics` rows and the micro-batch split;
  - a short profiled window at the compare band.
- **Winner rule:** correctness PASS; every decisive window gains more than max(2 %, the measured A/A spread) on equal splits; no window SLOWER; the mechanism is proven.
- **Labels:** `nb4_lib.final_label`.

### Budget (as built)
- **Hard stop:** 150 min (`NB4_BUDGET_SEC`).
- **Per-arm estimates `EST_MIN`:** T0 38, S / S2 / S3 24, G 25, P 30, T0b 28, K4 25, INT8 24 min.
- **Body caps `CAP_MIN`:** T0 30, T0b 25, the rest 19 min. The watchdog kills at cap + 5 min.
- **The mandatory arms alone need 193 min of estimates, before A0** (install, #60490, sha, core timer). So G and P are likely `SKIPPED_BUDGET`.
- **T0b is protected:** before any other arm X, if X + T0b do not both fit, T0b runs now.

### What the CPU kits check (verify that they really do; the kit code is in the zip)
**Units kit:** the 4 test suites on Py3.12 in the pinned image:
- nb4_unit_tests 39;
- nb4_timeline_tests 20 (the 5 real-trace tests skip);
- nb4_p5_tests 19, on the real Merlin `cuda_communicator.py`;
- nb4_ppack_tests 182 checks, on the real Merlin `pp_utils.py` / `model_runner.py`.

**Real kit:** on the real mounts:
- the payload shas vs the notebook pins;
- the notebook loaded (RC2_UNIT: no main, no boot);
- the Merlin install and #60490;
- the core timer install;
- for every arm in run order, the way `nb4_arm_step` does it without the boot:
  - the Q-pack / Q-defer prepare on the **installed** tree (apply, verify, revert; file shas before == after);
  - P5: source check, a recorded-only CPU compile probe of the `.so`, the gate `SKIPPED_GPU_ONLY`, apply / verify / revert;
  - `launch.build()` + `host_phase()` + `--help` cli flags;
  - the notebook's own `nb4_facts()` on a server stub;
- each plan's exact server env → a Merlin-venv child process → the extension's `nb4_env()`;
- vLLM's argparse field checks;
- each plan vs T0b over the full argv and env (exactly the planned change);
- the never-do list on every final argv + env;
- the tree's final state and the checkpoint sha256.

### Known finding from the kit build (verify it, and look for siblings)
- `nb4_facts` (CELL 7) builds its patch-state read-back as `for pk, arm_k in (("p5", "P"), ("ppack", "S2"))`. For **S3**, the Q-pack file is applied by `ppack_prepare("Q-defer")`, but the check expects `ppack patch not applied`.
- So `config_ok` would be False and S3 would stop as `CONFIG_MISMATCH` after its boot. That is wasted GPU.
- The real kit's check `notebook_readback_patch_states_ok` is built to catch this.

## Answer format
**Line 1: `GO` or `NO-GO` for one GPU commit run, in one sentence.**

A **blocker** is only one of these:
- wrong raw data, meaning a measurement that is recorded but wrong, mislabeled or mis-paired;
- a mid-run crash that loses data;
- wasted GPU time, for example a boot that cannot succeed or cannot prove its change, or a budget that starves an arm that matters.

Wording, labels and style are not blockers. List all blockers right after line 1. Give each one with file, function or line, the concrete failure scenario and the minimal fix.

### Part A: the arms' correctness and mechanism proofs
For each item, give a verdict (correct / wrong / partial), the evidence (file + function or line) and, if not correct, the failure scenario and the minimal fix.

1. **S, the env reach:**
   - Does `CUDA_DEVICE_MAX_CONNECTIONS=32` reach every CUDA process before its context is created: the 4 workers, the EngineCore and the PLE offload process?
   - Is the read-back real: `nb4_env` over `/collective_rpc` plus `/proc/<pid>/environ` of the session?
   - Can Merlin's `build()` (`FN_UNSET`, the `MERLIN_*` strip) or the NB1 env hook drop or override it?
   - Is "stall > 1 ms in ≤ 5 % of steady graphs" a sound mechanism test for this lever?
2. **S2 Q-pack and S3 Q-defer patches:**
   - Are the patches right against Merlin ad42e37: strict pre / post sha, marker, py_compile, revert to the exact pre-sha?
   - Is the wire format identical on sender and receiver?
   - Is the int64 → int32 count cast exact?
   - Is the ordering correct: the slot's ready time unchanged for Q-pack; Q-defer posting after the next forward launch, with the T − 2 wait kept?
   - Are they inert when the env is unset?
   - Graph capture: does a flush ever run during capture?
   - Are the read-back (`readback_label`: 4 inits, flags, counters) and the S3 patch-state read-back above right?
   - Does the revert always run, including after a failed boot, before the next arm?
3. **G, the capture list and the level-9 windows:**
   - Does [1..9, 12, 16, 20, 24, 28, 32, 36] give FULL for every real decode n at seqs 9, K3? Check micro-batch token counts (C/2)·(1+K), the draft model's graph mode and the uniform-batch rules.
   - Do 8k@9 / 40k@9 really reach running == 9 with both micro-batches equal, so that a FULL 36-token step exists?
   - Is `mechanism("capture_coverage")` reading the `--cudagraph-metrics` table correctly (unpadded = n × (1 + K))?
4. **C, the runtime A/B inside T0:**
   - Is the flip to eager (`set_recoverssm_commit_mode`) applied on every rank before any T0 measurement, and verified?
   - Do the graph windows really replay commit graphs: are the captured n's covered by n = 4 per micro-batch at 8@8?
   - Is the greedy graph-vs-eager row a valid bit-identity test?
   - Is the commit relaunch rule in `retry_decision` (a capture failure → T0 without the commit env, C dropped) safe?
5. **P5 integration and gate:**
   - Is the `cuda_communicator.py` patch point right: the call path, the size gate ≤ 184,320 B, identical routing on both TP ranks, never on ep / pp groups?
   - Is it graph-capture safe?
   - Fail closed: `NB4_P5_INIT_FAILED` → `P5_NOT_INTEGRATED`, never silent NCCL.
   - Does the 4-process gate prove concurrency on both TP pairs?
   - Are the counters (`p5_executed`) real device truth?
   - Is the build in the Merlin venv right (GCC 13 first, sm_89)?
   - Does the revert leave the tree exact for T0b?
6. **T0 timeline:**
   - Is the profiler config identical on every arm, and idle outside the profiled window?
   - Is the core-timer plugin dormant everywhere but T0's EngineCore?
   - Are `nb4_timeline.analyze`'s categories (compute / NCCL peer wait / transfer / host launch / scheduler-IPC) computed correctly from 4 rank traces?
7. **Exactly one change:**
   - Do `validate_arms`, `readback_checks`, `diff_vs_ref` / `one_change_ok` and the CPU kit's full-env diff together guarantee that each arm differs from the base by its one change only?
   - Is anything (for example a port-derived env, the profiler dir or `NB4_CORE_TIMER`) wrongly ignored or wrongly flagged?

### Part B: budget and labels
1. **The 150-min stop and the order T0 → S → S2 → S3 → G → P → T0b:**
   - With realistic A0 (Merlin install ~3 min, sha ~3 min, cold first load) and ready times (NB1 R2: 297-642 s), which arms run?
   - Is T0b's protection correct?
   - Can an arm start that cannot finish its decisive windows, wasting a boot?
   - G and P are likely skipped for budget. Should the order change (for example S3 before S2, or G before S2) for the most information per GPU-minute?
2. **The A/A:**
   - T0 (profiled, booted with the commit env, then flipped) vs T0b (2 replicates). Is that a fair noise floor for S / S2 / S3?
   - Does T0's profiler or core timer change its own windows?
3. **Labels:**
   - Can any path give `WIN_SPEED` with an incomplete, mis-paired or split-mismatched reading?
   - Is every failing gate named?
   - Are `SKIPPED_BUDGET` and `NOT_INTEGRATED` never reported as measurements?
4. **Data loss:** is every request and stream saved in full, and is `nb4_verdict.json` written on every exit path, including a mid-run crash and the hard stop?

### Part C: short advice (at most 10 lines)
1. If S, S2 or S3 removes the stall, what is the next GPU step toward the 1500 tok/s whole-run average? Consider: stage 1 then binds at ~34.6 ms ideal tick, with seqs > 9, K4, and the C / G results.
2. If none removes it, what is the next cheapest test of the false dependency? For example: stream priorities, a separate broadcast communicator, or moving the receive to a host thread.
3. Anything in `PLAN_1500.md` that NB4's result would change.

Keep Parts A, B and C separate.
