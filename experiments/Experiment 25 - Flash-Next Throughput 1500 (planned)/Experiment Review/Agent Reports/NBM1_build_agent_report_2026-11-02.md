## E25 NB-M1: build report (closed build done; nothing pushed to Kaggle, no datasets uploaded)

The notebook, its 3 copies, the units kit and the real kit are all built. The units kit passes in a local simulation; the real kit has not run (you run the kits). The closed build has `gpu_ready` false, so the gated rebuild is still to do.

**Copy run times are over your target for B and C.** Estimated: A 163 min, B 175 min, C 175 min. Your request was about 150-170. To fit, the hard stop is 180 min (deadman 185) for all three copies. The builder asserts each copy runs ≤ 175 min and fits all its arms inside the 180 with the 5-min start reserve.

### 1. Files
All builder files are LF. No file I did not create was changed. Every Exp 27 file, the NB-S1/NB-S2 files, nb4t_lib, nb4_ppack, nb4_timeline and k100q_ref27b.json are hash-checked before and after each build: all unchanged.

Build folder `D:\ARC AGI 2\experiments\Experiment 25 - Flash-Next Throughput 1500 (planned)\Experiment Notebooks\Build\` (sha prefixes):

| File | sha | Size (bytes) |
|---|---|---|
| `nbm1_lib.py` | e64482c4 | 37,146 |
| `build_e25_nbm1.py` | 0f8356e7 | 31,255 |
| `nbm1_unit_tests.py` | ebe63cd5 | 42,461 |
| `build_e25_nbm1_units_kit.py` | 1fc370ee | 6,962 |
| `build_e25_nbm1_cpu_kit.py` | 79fa7b21 | 33,319 |
| `nbm1_cell7_body.py` (outputs) | 8287839d | 132,380 |
| `nbm1_code_export.py` | 8afe08cd | 375,481 |
| `nbm1_build_facts.json` | 8935a7e9 | 43,918 |

Notebooks in `...\Experiment Notebooks\`:

| Notebook | Role | sha | Size (bytes) |
|---|---|---|---|
| `Q27-XHIGH-E25_NBM1.ipynb` | every arm on; the kit notebook | 6505026007ed23cf6c4c69e905638bc0fd2ac99556d271eb7156fdd76504b286 | 394,333 |
| `Q27-XHIGH-E25_NBM1_A.ipynb` | copy A | 0f3ba0480b51a6d1 | 394,342 |
| `Q27-XHIGH-E25_NBM1_B.ipynb` | copy B | e3d424283e77c9f3 | 394,341 |
| `Q27-XHIGH-E25_NBM1_C.ipynb` | copy C | 5215ddff211271de | 394,340 |

The copies differ only in the `NBM1_ON` line; the builder asserts their code equals the export with that one line replaced.

Push folders (in `D:/ARC AGI 2/_scratch/push_e25/`):
- `q27-xhigh-e25-nbm1-a__foysalemonshanto`
- `q27-xhigh-e25-nbm1-b__esfersami50`
- `q27-xhigh-e25-nbm1-c__foysalemonshanto`

Their metadata uses NB-S1's sources: the tf-ref dataset, the 4 K100 models and the competition. The vocab dataset is listed only on copy A. The builder deletes the all-on base push folder, so only the three copies remain as pushes.

### 2. Design

**How it is built:** NB-S1's pattern. `build_e27_nbdf.py` runs with `NBDF_VARIANT=k100`, and the K100 module is re-pointed in memory. Its `edit_cells`, `edit_body`, `edit_meta` and gate are wrapped: the K100 edits run first, then the NB-M1 edits, each as an exact-once replacement. `nbm1_lib` is embedded and its sha is checked at run time.

**Base Q:** NBDF-K100's Q arm verbatim:
- MTP K3 probabilistic, I8 int8 head on (`q_i8` True is asserted), Q-defer, seqs 24.
- Pin 7,933,526,016 and its ladder [7,933,526,016, 7,665,090,560].
- A/A, ladder, oracle plus oracle A/A, long oracle, prefix test.
- Windows 8k@16 ×2, 20k@24, 40k@16, 90k@8, 116k@6; profiles 8k@16 and 116k; TF.

Only the name (`nbm1_q`) and the time budget fields change. The 4 NB-M1 env keys are added to Q's expected env as "absent".

**Arms (each = Q + one change; `check_arms` enforces it):**
- **V65 / V32:** `VLLM_DRAFT_VOCAB_FILE` plus `VLLM_DRAFT_VOCAB_SHA256`. Merlin reads `_SHA256`, not `_FILE_SHA256` (draft_head.py:33). Merlin's `build()` pops both variables (launch.py:424-425), but NB1_ENV is applied after `build()`, so they reach the server.
- **I4:** `DRAFT_HEAD=int4` goes into Merlin's `build()`, which writes `draft_lm_head_quant` int4.
- **IDX:** `--attention-config {"indexer_kv_dtype":"fp8"}`.
- **S28 / S32:** `--max-num-seqs 28/32` plus captures from Merlin's `capture_sizes_for(28 or 32, 3)` (top 112/128).
- **LP:** the three long-prefill args.
- **G1 / G2:** the GDN env switches.
- **K4:** trailing spec with K 4 plus Q's head, captures 5…120.
- **DYN:** K4 plus [[1,8,4],[9,24,3]], captures b×5 for b ≤ 8 and b×4 for b 9-24.
- **PROF:** two 30-s torch.profiler windows (8k@16 and 20k@24) inside Q's own boot, after Q's TF, with their own trace folders. It has its own record (DONE / NOT_INTEGRATED / SKIPPED_SWITCH_OFF). Q's base items are untouched.

**Draft head of the levers:** Q's (int8 while Q_I8 is on, otherwise BF16). I4 is always int4. The K4/DYN trailing spec follows the head.

**Lever items:**
- Each lever runs facts, a smoke without the ladder (any FAIL_* label stops the arm, hard labels only), oracle against Q, and warmup.
- The ladder, prefill probe, long oracle, prefix test and profiles run on Q only.
- TF runs only on IDX and LP (the two levers that change target numerics).
- Window sets: A levers get 8k@16, 20k@24, 40k@16, 90k@8, 116k@6. IDX, LP, G1, G2, K4 and DYN get 8k@16, 90k@8, 116k@6. S28/S32 get 20k@28 or 20k@32, 90k@8, 116k@6. 90k and 116k are always included.
- A KV-bound lever window steps down one level on any band; it is never skipped.

**Real-failure read-back per lever (NOT_INTEGRATED for that arm only):**
- the env keys in all 4 workers, and absent on every other arm;
- the "REDUCED DRAFT VOCAB ACTIVE" lines on both PP1 ranks with the right N and file;
- the INT8 head built on the reduced shard (N/2 rows);
- INT4 head lines for I4;
- the indexer `compressed_key_cache` dtype is fp8 (the GPU default is bf16, checked in the K3 v2 record);
- the logged seqs, batched tokens / long-prefill threshold / adaptive, spec K and dynamic schedule;
- the DYN fallback warnings.

**Other edits:**
- **OOM fix:** NB-S1's GPU-v1 line, `oom = bool(o["oom_in_log"]) and not o["xid_in_log"]`, so an OOM at any stage steps the Q ladder.
- **Seam tolerance:** `tree_check` accepts the seam as ORIGINAL or FULLY_PATCHED for MTP arms. The NBDF-K100 seam sequence is kept as it was.
- **Main loop:** the D loop is replaced by the NB-M1 loop: CELL 1 switch, then the kit gate plus the vocab table, then budget, then `pseq.before_arm`. A failed arm never skips the next one; the K100 deferred iso rerun is kept.
- **Records:** finalize writes `nbm1_records`, with per-band gains vs Q.

**Switches:** a plain `NBM1_ON` line at the top of CELL 1; a missing key means that arm is off.
- Base: every arm on.
- A = Q, V65, V32, I4
- B = Q, IDX, S28, S32, LP
- C = Q, G1, G2, K4, DYN, PROF

### 3. Time estimates (minutes)

| | Setup | Q | Levers | Total |
|---|---|---|---|---|
| A | 6 | 61 | V65 32, V32 32, I4 32 | 163 |
| B | 6 | 61 | IDX 27, S28 27, S32 27, LP 27 | 175 |
| C | 6 | 69 (incl. PROF) | G1 24, G2 24, K4 26, DYN 26 | 175 |

The model comes from the NBDF-K100 GPU run of 2026-11-01:
- setup 5.6;
- Q ready 543 s;
- Q body hit its 41-min cap before TF;
- window spacing 110 / 215 / 183 / 275 / 272 s;
- Q post 6.4.

It also uses NB-S1 GPU v1: first boot ready 732 s, warm boots 255-285 s. A boot with a new compile key is assumed to take 7 min to ready. Q's cap is raised to 48, or 55 with PROF, so TF fits.

### 4. Vocab tables
They are in `foysalemonshanto/q27-e29-final-payload`, mounted at `/kaggle/input/datasets/foysalemonshanto/q27-e29-final-payload/`:
- n65536: sha a915fa6d…
- n32768: sha a164d1a7…

The dataset exists on Kaggle. The E29 MTP-V64 real kit mounted it and matched the 64k file's sha. The 32k file is assumed present (it is in the same local upload folder); the NB-M1 real kit checks both. No upload should be needed. If a table is missing, only V65/V32 are recorded NOT_INTEGRATED.

**Vocab and the int8 head (from the Merlin source; not tested on GPU):** the vocab works both with and without the int8 head. Merlin's `draft_lm_head_quant` wraps the drafter's own reduced `ParallelLMHead` (speculator.py:547-603). The shard sizes are Marlin-aligned (32768 or 16384 rows, % 64 = 0; hidden 2560 % 128 = 0), and the reduced head is never shared with the target. So V65/V32 = Q's int8 + vocab, which is a valid "Q + one change". If Q drops I8, they follow it to BF16. The read-back fails the arm if the INT8 head is built on the full 124,160-row shard.

### 5. Why the E29_MTP_V64 kit failed
Both failing checks have one cause, and it is in the kit's drive, not in the vocab. Every E29 try ran at Merlin's FP8 pin 2,479,838,700 (gib 0) with seqs 6 and the I8 head off (the E29 setup log shows `i8_on` false) instead of the E28 top rung: 7,933,526,016 / seqs 24 / int8.
- **`e29_nb5_boot_every_try_planned_d_ladder_drops_mtp_fallback`** failed with "last try not the plain E28 top rung" (wrong `kv_bytes`). The drop chain itself worked: idx dropped, then vocab dropped, then the plain try finished.
- **`e29_vllm_argparse_every_built_e29_argv`** failed only on "max_num_seqs 6". Parsing was fine, with no unknown flags, and indexer fp8 was parsed. The vocab probe passed: n 65536, sorted, wrong sha refused.

For a vocab arm this matters because vocab + int8 at the top pin and seqs 24 was never built or parsed. NB-M1 avoids the cause:
- there is no nb5_boot ladder;
- every arm boots through NBDF `arm_boot` at the fixed pin with seqs 24 and Q's int8;
- the real kit builds, launch-checks and argparses V65/V32 at exactly that pin, seqs and head, and asserts each lever's head in the build env.

### 6. Tests
- **Unit tests:** `nbm1_unit_tests.py` (Py3.14) 142/142 PASS. They cover:
  - the arm plan and refusals;
  - `launch_check` per arm, with the int8 and BF16 heads;
  - every read-back, positive and negative;
  - the switches and the 3 copies;
  - the time plan;
  - `nbdf_main` with fakes: each copy, Q off, the closed gate, a missing vocab, a failed arm;
  - `arm_boot` with fakes: an OOM after ready steps to 7,665,090,560 and port+1, an Xid does not retry, the ladder end, and the heads in the build env;
  - `run_window` step-down;
  - the verdict;
  - the switch line read from the code export.
- **Units kit:** simulated in an empty folder on Py3.12: 143/143, `@@KIT units ALL OK@@`.
- **Gated path, tested with a synthetic report in `--test-out`:**
  - all keys True: `gpu_ready` true, 395,638 B, `nbdf_gate_verify` ok;
  - one NB-M1 key False: the gate stays closed.
- **Build:** deterministic (two builds gave the same sha). Notebook is 394.3 kB closed, about 395.6 kB gated.
- **Real kit:** static check finds no undefined names; built at 382,745 B.

### 7. Kit folders (both foysalemonshanto, in `D:/ARC AGI 2/_scratch/push_e25/`)
- **`q27-e25-nbm1-kit-units__foysalemonshanto`** (sha 4563f01e…, 227,567 B): the CELL 7 stub is omitted for size, and the cells are sha-checked against the facts.
- **`q27-e25-nbm1-kit-real__foysalemonshanto`** (sha d6d9bde2…): uses the tf-ref dataset, the e29 payload and the 4 models. It runs:
  - the notebook's own pre-launch sequence for all 12 arms;
  - the `--help` gate;
  - vLLM argparse of every full argv;
  - KV probes: K4, IDX (> Q's 480), and S28/S32 max concurrency;
  - Merlin `load_draft_vocab_ids` on both tables;
  - the klee100 `lm_head` check (BF16 [248320, 2560], untied);
  - Q-defer pack views up to 32 sequences.

  It adds six gate keys, `nbm1_*_ok`. One kit-side substitution: the kit calls `pre_gate` with the gated build's `{"passed": True}`.

**After the real kit passes:**
1. `python build_e25_nbm1.py --kv-gate-report <nbdf_kit_real_report.json>`
2. `nbdf_gate_verify.py nbm1_build_facts.json nbm1_cell7_body.py ..\Q27-XHIGH-E25_NBM1.ipynb`

The kits are bound to closed sha 65050260; any builder change means rebuilding the kits.

### 8. Open risks
- **Timing:** B and C estimates sit at the 175 cap. If boots are slower, the budget guard records the last arm (LP or DYN) as SKIPPED_BUDGET.
- **OOM rule (yours, as in NB-S1):** a late Q OOM restarts the whole Q one rung lower (about 60 min) if 66 min or more are left. That would starve that copy's levers.
- **Never run on GPU:** IDX fp8 on sm_89; MTP K4 and DYN on klee100; Q-defer at seqs 28/32.
- **Accounts:** that esfersami50 can see foysal's private Models and the tf-ref dataset is assumed, as NB-S2 did.
- **Size margin:** about 4.4 kB under the 400 kB rule.
- **Thinner lever evidence:** levers have a single 8k window (Q's A/A spread is the noise threshold) and B/C levers have no 40k. In nbdf_lib's own verdict the speed label may read INCONCLUSIVE; labels are record-only, and the raw numbers are in `nbm1_records`.

### 9. Verified vs assumed
- **Verified:**
  - from Merlin source: the env names, the build pop, vocab + int8, the I4 path, dynamic K on MTP;
  - from the GPU records: the bf16 indexer default, the timing data, the older-lib 116k LOW_VARIETY flag that is now record-only;
  - from the E29 report: the 64k table on Kaggle;
  - locally: the builds, tests, units kit and synthetic gate.
- **Assumed (real kit or GPU will tell):**
  - the 32k table is on Kaggle;
  - the klee100 `lm_head` is BF16 and untied;
  - the `--help` gate accepts the LP and IDX flags;
  - every GPU behaviour and time estimate above.
