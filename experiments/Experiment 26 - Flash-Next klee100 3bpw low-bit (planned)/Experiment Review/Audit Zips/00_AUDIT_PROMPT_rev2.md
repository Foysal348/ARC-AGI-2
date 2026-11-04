# Fix verification + audit request, rev 2: E25 klee100 GPU load test notebook (2026-11-01)

All files in this zip are flat (no folders). Read every file in full before answering. `00_ZIP_MANIFEST_SHA256.txt` lists each file with its source path and sha256.

Facts stated in this prompt describe the build at the time of writing. If a file in the zip says something different, the file is authoritative. Report the difference as a finding.

## What changed since rev 1
- **Rev 1:** notebook sha 78b94c49, 376 kB. Three GPT audits all returned NO-GO as-is and GO after fixes. Two independent internal reviewers then added 30 more items.
- **The 43 verified items** are in `KLEE100_LOAD_AUDIT_FIXES_2026-10-31.md`:
  - K-1 .. K-13 (GPT audits);
  - R1-1 .. R1-18 (reviewer 1: GPU crash / OOM / budget);
  - R2-1 .. R2-12 (reviewer 2: data / labels).
- **New in rev 2: the K3 boot.** The builder tags this item R3-1. K3 is an FP8-first boot added on the user's decision. It is not in the fix list, so it is audited as new code in Part B.
- **The notebook is rebuilt.** It is `Q27-XHIGH-E25_KLEE100_LOAD.ipynb`. Its sha256 and size are in `00_ZIP_MANIFEST_SHA256.txt` and `k100load_build_facts.json`.
- **The E23 lab is no longer embedded (R1-18).** It is now a sha-checked file of the reference dataset, `k100load_lab.py`, which keeps the notebook under the ~400 kB Kaggle commit limit.
- **Code map:**
  - builder: `build_e25_klee100_load.py`;
  - pure helpers: `k100load_lib.py`;
  - tests: `k100load_unit_tests.py` (quick units) and `k100load_tests_realdata.py` (real-data paths, R2-6), plus any other `k100load_*test*.py` in the zip;
  - all notebook cells as one file: `k100load_code_export.py`;
  - CPU kit builder: `build_e25_klee100_load_cpu_kit.py`;
  - kit notebooks: `q27-e25-k100load-kit-real.ipynb` and `q27-e25-k100load-kit-units.ipynb`;
  - kit reports: `k100load_kit_*_report.json`, when present. Each report names the notebook sha it tested. Check that it equals the notebook in this zip.

**Name clash, please keep these apart:**
- "K3" is the **boot** name, in K3 → K0 → K → K2.
- "MTP K3" means 3 speculative tokens.
- "K7 + K3" is the **streaming sha256 gate**, named after the klee100 CPU-kit check list.

## Context
- **Competition and hardware:** Kaggle ARC-AGI-2 on 4×L4 (24 GB each, sm_89, PCIe, no P2P).
- **Model:** Qwen3.8-Flash-Next: 125B parameters, 6B active, 48 layers (36 GDN + 12 QSA), 512 experts top-10, hyper-connections, and a PLE table.
- **Engine:** Merlin v3.0.0, a vLLM 0.32 fork, running TP2×PP2+EP with MTP K3.
- **Target:** a whole-run average of 1500 tok/s (`PLAN_1500.md`). Capacity, meaning resident streams, is the binding term.
- **Current base (RC2):** the W4 checkpoint with FP8 QSA KV. Its pool is 513,940 tokens (149 usable 4096-blocks, F ≈ 6 fixed blocks per request). It reaches 259 tok/s at 8 streams.
- **klee100 checkpoint** (`klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`, rev 600715b3):
  - routed experts W3 g128 on layers 0-11 / 36-47 and W2 g64 on layers 12-35, through humming;
  - MTP experts W4 g128;
  - main projections W8 GPTQ.
- **Loading needs a config-only "fixkeys" overlay** plus Merlin's FP8 PLE shards grafted into the index. See `k100kit_core.py`, `FIXKEYS_DIFF.json` and `E25_klee100_cpu_kit_v1_2026-10-31.md`. The klee100 CPU kit v2 result is ALL OK.
- **Kaggle Model:** `foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/PyTorch/autoround-3bpw-mtp-nople/1`, 33 files, 50.68 GB.
- **Capacity estimate:** 463 blocks is the FP8-equivalent figure. With BF16 KV it is about 244 blocks (K-11). The measured worker blocks are authoritative.

### NB1 R2 (W4, measured 2026-10-31; full doc `E25_GPU_NB1_R2_2026-10-31.md`)
- **B3 = B1 + pin +1.0 GiB (3,553,580,524 B):**
  - pool 738,142 tokens / 214 blocks, F 6.3, clean;
  - free MiB per card after ready: nvidia-smi 1629 / 1629 / 939 / 925, worker 639 / 639 / 205 / 191;
  - GPU peak 22,377 of 23,034 MiB;
  - 90k@5 all resident, 0 preemptions;
  - 40k@8 at 249.5 tok/s;
  - step time unchanged versus B1, so the pin changes capacity only.
- **B1 profiler:** stage 0 is the binding stage, with a 50.6 ms critical path and 27.2 ms GPU-busy. NCCL broadcast wait is 63.5 % of kernel time, so the tick is bound by sync and host, not MoE bandwidth.
- **B2 (SSM bf16 + seqs 16 + captures to 64 + commit graph): NO_GO_ORACLE.**
  - SSM bf16 gives zero capacity: its state is padded to the same 2 MiB page.
  - The RecoverSSM commit graph captured only n=16.
- **Prefix reuse on Merlin:** 93-98.5 %. An aborted request keeps its cache.
- **Decisions:** adopt pin +1.0 GiB; keep text-only (`--language-model-only`) and #60490; park SSM bf16.

### NB5 QUALITY (W4, ARC accuracy pairing, 2026-10-31 / 11-01)
- **FP8 phase (done):**
  - 13 / 15 tasks, 18 / 20 test units, task-norm 86.67 %;
  - 9 streams, pool 214 blocks, 0 preemptions;
  - gen p50 224.9 tok/s at 9 running.
- **BF16 phase:** pool 112 blocks, 5 streams. It is still running; there is no result yet.
- **Why K3 now goes first:** the FP8 phase's result (13 / 15) is why the user put the FP8 boot first on klee100. The BF16 comparison is not finished yet.

### K3, the new FP8-first boot (as built; verify in the code)
**Order and budget:**
- Order: A0 → **K3** → K0 → K → K2.
- Hard budget 150 min (`K100_BUDGET_SEC`).
- Per-boot estimates `EST_MIN` = K3 45 / K0 27 / K 33 / K2 43 min.
- Body caps `CAP_MIN` = K3 26 / K0 12 / K 24 / K2 34 min. The watchdog kills at cap + 5 min.

**Config:** klee100, FP8 E4M3 QSA KV with static K/V scales, MTP K3, seqs 16, captures to 64.

**Scales sidecar** (`qsa_kv_scales_262k.json`, 12 QSA layers, sha 5cbe6ae8…):
- It is the W4 Merlin p1 checkpoint's sidecar.
- It is taken from the installed Merlin tree first, else from the reference dataset, then sha-checked.
- If it is missing, K3 is `SKIPPED_NO_SCALES` and the BF16 boots still run.
- It was calibrated on W4, not on klee100.

**Pin ladder** (`k3_pin_steps`), with no probe boot before it:
- Per card: headroom = B3 worker free + (W4 measured load 16.89 GiB − klee100 CPU-kit Gate D estimate for that stage, PP0 11.973 / PP1 12.026 GiB) − 1 GiB safety − 102.4 MiB graphs − 768 MiB workspace.
- `est_extra` = the minimum over cards = 3,277.3 MiB, on binding card 3.
- Steps = [B3 + 0.8 × est, B3 + 0.5 × est, B3, Merlin FP8 base] = 6,301,941,760 / 5,270,142,976 / 3,553,580,524 / 2,479,838,700 B.
- The next rung runs only after a launch OOM, never at or above a pin that already OOMed.
- An extra rung needs `left() ≥ (45 + 27 + 33) min`, so the BF16 boots are never shortened by a K3 retry.

**Body:**
1. Read-back, method map and scales application.
2. Probes (engine faults only), sampled output, oracle against RC2's FP8 rows.
3. TF 2k / 8k / 40k against NB0 A (W4 BF16), fresh + repeat. K3 against K (klee100 BF16, same ids) is computed after K ran.
4. Warm-up.
5. 8k@8, then 40k@N and the 64k admission, only when `capacity_ok`.
6. Arithmetic probe, recorded and not gated.
7. Every stream traced.

**Feedback into K:** K3's free-memory reading also feeds K's ladder (`ladder_min` of the K0 and K3 readings, per frac the lower pin; K3 with MTP reserve 0).

**Label:** a K3 sub-label (`K3_FP8_LOAD_PASS + <reading>`, `K3_SKIPPED_*`, `K3_FP8_GARBAGE_UNDER_LOAD`, `K3_FP8_<status>`) is put at the head of the final label: `<K3 part> | <BF16 part>`. It never changes the K / K2 verdict.

### Other new parts to audit
- **Method map (K-7):** `NB1WorkerExt.k100_method_map` is read-only. It is called by rpc on every rank. `method_map_check` checks 48 routed-expert layers (humming method, bits / group against the checkpoint) plus the MTP experts (W4 g128, AutoGPTQ MoE). It is a gate on K / K2 / K3 and recorded on K0.
- **Trace saving (K-8, R2-10):** `k100_trace_install` wraps `L.Stream.__init__` on the klee100 side, leaving LAB_SRC unchanged. After each window, outside the timed part, it writes every stream's full text, token ids and finish to `k100/traces/<boot>_<window>.jsonl`. This covers the admission streams too. Decode token ids are logged on success and on error.
- **Labels:** see `tf_label` / `tf_reading` / `final_label` / `final_label_bf16` / `k3_label` / `k2_tag` in `k100load_lib.py`.
  - `GPU_LOAD_PASS` requires all four TF bands to be measured; otherwise the label is `GPU_LOAD_INCOMPLETE_TF`.
  - R2-3 adds `_K_<status>_AT_PIN<gib>`.
  - R2-12 adds `_K2_GARBAGE_UNDER_LOAD`.
  - R2-1 adds `+ LONG_<verdict>`, which is never a gate.
  - Accuracy is always NOT_TESTED.

## Answer format
**Line 1: `GO` or `NO-GO` for one GPU commit run, in one sentence.**

A **blocker** is only one of these:
- wrong raw data, meaning a measurement that is recorded but wrong, mislabeled or mis-paired;
- a mid-run crash that loses data;
- wasted GPU time, for example a boot that cannot succeed, or a budget that starves a later boot that matters.

Wording, labels and style are not blockers. List all blockers right after line 1.

### Part A: fix verification, item by item
Give one table row per item, for all 43: K-1 .. K-13, R1-1 .. R1-18, R2-1 .. R2-12. Do not group items or skip any.

| item | fixed: yes / no / partial | evidence (file + function or line; for partial / no: the concrete failure scenario and the minimal fix) |
|---|---|---|

Notes for Part A:
- **Overlaps in the fix list:** R2's H2, M1 and M3 equal R1-1/R1-10, R1-5 and R1-3. R2-11 overrides the part of K-4 that made the RC2 oracle identity a hard gate. R1-5 and K-12 together mean the arithmetic probe is recorded and never gated. Judge each item against the final intended behaviour.
- **A test is not evidence of the fix by itself.** Check that the code path the notebook actually runs, in `k100load_code_export.py`, has the fix. A fix only in `k100load_lib.py` that the notebook does not call is a `no`.
- **R1-18:** check that the notebook loads `k100load_lab.py` from the dataset with a sha check, and fails closed. Check that the dataset's `SHA256SUMS.json` lists it, and the scales file, with matching shas.

### Part B: audit of the new parts (blockers first)
For each finding, give: file, function or line, a concrete failure scenario, severity (BLOCKER / HIGH / MED / LOW), and a minimal fix.

1. **K3 pin ladder:**
   - Is the estimate sound? It starts from B3, a W4 boot with MTP on at seqs 8 / captures to 32, and moves to klee100 at seqs 16 / captures to 64.
   - Is the Gate D weight estimate (CPU, "calibrated on RC2") safe to subtract from a measured W4 load?
   - Are the humming workspace, W8 GPTQ buffers, #60490 headroom and the first long prefill (40k TF, 64k admission) covered?
   - Can the top rung OOM in the **body**, not at launch, and so waste a boot without stepping down?
   - Does a K3 OOM or device fault leave the GPU clean for K0?
2. **FP8 scales sidecar:**
   - Is it applied to all 12 QSA layers on every rank / stage?
   - Is the MTP layer's default scale correct?
   - Is the "applied" evidence real, meaning read back from the engine and not only the file?
   - What happens if the Merlin tree's copy differs from the dataset's?
   - Is a W4-calibrated scale on klee100 a correctness risk that the K3 TF bands would actually detect?
3. **K3 TF pairing:**
   - K3 (FP8 KV, MTP on, klee100) is compared against NB0 A (BF16 KV, W4) and against K (BF16, klee100). The yardstick is NB0's FP8-vs-BF16 A/B.
   - Are the ids, positions, top-k and valid masks the same?
   - Is any threshold misapplied?
   - Does MTP on versus off change prompt logits in the TF path?
4. **Budget order K3 → K0 → K → K2 inside 150 min:**
   - The estimates add up to 45 + 27 + 33 + 43 = 148 min, before A0. On NB1 R2, A0 took: Merlin install 170 s, sha 191 s (W4 files), and RC2 model load 327-433 s per worker. Here the sha covers 101.8 GB of klee100 + PLE files, and K3 is a cold first load.
   - What gets cut, and in which order?
   - Can K2, the MTP speed boot at BF16, still run in a realistic case?
   - Is any reserve double-counted, or missing (`k2_reserve_sec`, `k2_plan`, K3 `can_try`)?
5. **Method map:**
   - Can the rpc fail or hang on a rank and stop a clean boot?
   - Are layer indices under PP2 mapped to global layer ids, not stage-local ones?
   - Can the MTP layer (index 48 after the remap) be confused with a routed layer?
6. **Trace saving:**
   - Does the `Stream.__init__` wrap change timing inside a timed window?
   - Can it lose data on a crash, or fill the disk (full thinking up to 120k tokens per stream)?
7. **Labels:**
   - Can any path give `GPU_LOAD_PASS` or `K3_FP8_LOAD_PASS` with an incomplete or mis-paired reading?
   - Can a K3 failure change the BF16 label?
   - Is every failing gate named?
8. **Kit coverage:** do the CPU kits exercise the K3 path, the scales sidecar, the dataset lab load and the CELL 8 flow, so that notebook-level crashes are caught before the GPU run?

### Part C: short advice (at most 10 lines)
1. If K3 loads clean with a CLOSE / DIVERGENT reading, what is the next GPU experiment with the best expected gain per GPU-hour toward the 1500 tok/s whole-run average? Candidates:
   - more resident streams on the measured pool;
   - seqs 16-32;
   - K3 vs K4;
   - humming tile tuning;
   - the stage-0 sync gap.
2. What is the minimum ARC accuracy check before klee100 FP8 becomes a production candidate? The NB5 QUALITY pairing is 15 tasks × 2.
3. Anything in `PLAN_1500.md` that is wrong for the klee100 path.

Keep Parts A, B and C separate.
