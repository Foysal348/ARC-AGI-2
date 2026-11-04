## E32 lever-bundle notebook: built and tested locally; nothing pushed, no GPU run

The new notebook is the GPU-proven E29_DFLASH_D3S engine plus all 7 speed levers at once, in one 6-h EVAL120 run, and it is still dual mode. It is built and tested locally. Nothing was pushed or committed, no Kaggle API was called, and no git commands were run. All 7 levers, alone and combined, have never run on a GPU, so the smoke is their first real exposure.

**Before any GPU commit or kit:** upload the new payload folder `D:\ARC AGI 2\_scratch\push_e25\ds_q27-e32-levers-payload\` as the dataset `foysalemonshanto/q27-e32-levers-payload`. If it is missing, every lever turns off and the run is plain E29_DFLASH_D3S (recorded as `OFF_SETUP`).

### Design
- **Wrapper, not a copy.** `Build\build_e32_levers.py` imports the E29 builder but never runs its build, dataset or direct-file writers.
  - It first rebuilds the E29_DFLASH_D3S notebook in memory and asserts it equals the E29 notebook on disk (`9a097847…`), byte for byte.
  - The scheduler patch, nbx and DIRECT files are read from the E29 build outputs. Each is checked against E29's SCHED_PIN and its copy in the E29 payload folder.
  - So a later E29 change (the pending early-DIRECT order and 60k floor, which I did not implement) needs only a rebuild here. If an anchor no longer matches, the build stops.
- **Ten marked once-only edits (X1–X10).**
  - X1: the markdown header.
  - X2: the switch lines at the top of CELL 1.
  - X3: the smoke budget, 70 → 100 min.
  - X4: the E32 block, placed inside E29's packed engine block; `e29_env` gains G1/G2; `e29_extra` gains the LP/SPC/IDX argv before the DFlash tail; hooks in `e29_drop('df')` and `e29_mtp_boot`.
  - X5: `nb5_capacity` counts the launched 28 seqs as the top rung.
  - X6: in `nb5_boot_try`, the bundle read-back after the DFlash read-back, and the speed pass after the F2 probe.
  - X7: in `nb5_boot`, the E32 seqs, the try-name tag, the drop rule (after E29's QSA rule, before its OOM rule), and the top-rung 24-seqs override no longer applies to D tries.
  - X8: in `e21_relaunch`, an in-run death drops one lever first.
  - X9: `e32_setup()` after `e29_setup()`.
  - X10: `E28_OUT["e32"]` in the report.
- **Lever code lives in the new payload, not the notebook.**
  - `e32_lib.py` = a verbatim, sha-pinned cut of NB-D1's `nbd1_lib.py` (`92fd3e8b`) plus `e32_core.py`.
  - `e32_nbx.py` = the notebook glue, executed into the notebook namespace.
  - Every hook is `e32_call(name, default, ...)`: if the payload is missing, E32 is not ON, or anything raises, the default (= E29 behaviour) applies.
- **Shared E29 modules untouched.** `e29_lib`, `e29d_sched_patch`, `e29_nbx` and `e29_direct` are not edited. Only `E29L.effective` is wrapped at runtime, to add the E32 lever states and fingerprint to the LIVE lines; the setup failure path restores it.
- **Scope.** Levers act on DFlash tries only. Every MTP try is exactly the E29 config.

### Default switch values (plain literal lines, editable without a rebuild)
```
E32_SEQS = 28        # 24 = off | 28 | 32   (S28/S32 = one switch)
E32_SPEC = 'DYN'     # 'K3' = off | 'D4s' | 'DYN'   (D4s/DYN = one switch)
E32_LP = True
E32_SPC = True
E32_IDX = True
E32_G1 = True
E32_G2 = True
E32_SPEED = {'on': True, 'windows': [['8k', 16], ['20k', 24], ['90k', 8], ['116k', 6]], 'cap_min': 15}
```
- A bad or missing value turns that lever off, recorded as `invalid->off`; it never stops the run.
- With every lever off the engine is exactly E29_DFLASH_D3S (tested). Q256 is not in the bundle (it is a control).

### Each lever: apply / read-back / fallback
Read-backs are record-only. The one exception is the DYN fallback lines, which count as a real failure.

| Lever | Apply (D tries only) | Read-back | Fallback |
|---|---|---|---|
| SEQS 28 | `--max-num-seqs 28`, captures b×(1+K) for b 1..28; workers = 28 | logged `max_num_seqs` (C-11 already gates it) | back to 24: memory order 2nd, a CONFIG_MISMATCH on seqs, or the C-8 capacity gate |
| SPEC DYN | trailing spec = E29 dflash spec + K4 + `num_speculative_tokens_per_batch_size` [[1,8,4],[9,seqs,3]]; captures = NB-D1 `dyn_captures` | logged K and schedule; the DYN fallback lines → status `E32_DYN_FALLBACK` (real failure) | → K3 (the E29 D3s spec) |
| LP | `--max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive` | vLLM non-default args line | memory order 1st |
| SPC | `--enable-mamba-shared-prefix-checkpoint` | non-default args flag | general order 2nd |
| IDX | `--attention-config {"indexer_kv_dtype":"fp8"}` | worker kv_cache_specs dtype (nb1_cache_info RPC), else the logged config | general order 3rd |
| G1 | `VLLM_GDN_DECODE_FASTPATH=1` | /proc environ of the 4 workers | general order 4th |
| G2 | `VLLM_GDN_SHARED_ALIGN_INDEX=1` | /proc environ of the 4 workers | general order 5th |

- At 24 seqs the DYN schedule is NB-D1's kit-validated one exactly; at 28/32 the dense table is the same (asserted).
- LP's argv sits before the DFlash tail; Merlin's own 1024 comes earlier in the argv, and the last value wins.

### Drop order on a real failure of a D try
One lever per failed try, and the same D rung boots again:
1. SPEC is DYN and a DYN fallback line is in the log → SPEC.
2. A lever whose own flag text is in an error line (FAIL_CLI_FLAGS, argparse "unrecognized arguments") → that lever.
3. CONFIG_MISMATCH on the seqs check → SEQS; on the spec check → SPEC. NO_GO_INSUFFICIENT_KV → SEQS.
4. Memory (an OOM line that is not a device fault, or the F2 probe's LOW_FREE): **LP → SEQS → SPEC → SPC**. If none of these is on, E29's rule takes the next D rung and the other levers stay on.
5. Anything else, including an XID device fault: **SPEC → SPC → IDX → G1 → G2 → LP → SEQS**.

- With no E32 lever left, E29's rules run unchanged (D ladder, then the MTP K3 top rung with I8). When DFlash is dropped, the levers still on are recorded as dropped with it.
- In-run: a death drops one lever by the same rule and the booted config is rebuilt (tested equal to a fresh boot). A memory death with no memory lever left goes to E29's in-run plan. The path to MTP strips the E32 argv, the G1/G2 env and the 28 seqs.
- The F2 130k probe and the D ladder stay in force with every lever on.

### Measurement outputs
- **Speed pass** (record-only, commit only): 8k@16, 20k@24, 90k@8, 116k@6, using E29's `L.steady` 90-s windows. It runs once, after the F2 probe, on the first D try that passed every gate.
  - Prefill rate = the F2 probe's (about 6,100 tok/s). The 15-min cap never skips the 90k or 116k bands. Expected 12–14 min.
  - Prints `[LIVE speed …]` lines.
- **Run numbers:** E29's run summary (whole-run tok/s, per-15-min tok/s, AL, preemptions) plus `[E32 RUN SUMMARY]` and `q27x/e32_report.json`.
- **Engine fingerprint** = argv + the engine env keys + KV pin + Merlin facts (tree, launch.py sha, commit ad42e37f) + each lever's on/off/dropped state, hashed with sha256.
  - An `E32 FINGERPRINT` line prints for each booted D try.
  - The short sha and lever states appear in the `[LIVE start]` and every `[LIVE 15-min]` line.
  - The full fingerprint per try goes in the report.
- All E29 LIVE lines are unchanged.

### Expected KV and capacity at 7,665,090,560 B with the defaults
- **Pool:** 521 blocks, F = 12 (E29 smoke).
  - K4/DYN: same pool (NB-D1 kit: 540 = 540).
  - IDX: 0 extra blocks on the DFlash layout. The 480 → 508 gain you quoted applies to the MTP layout only.
  - SEQS, LP, SPC: no pool change. LP lowers free memory after the probe (base 903 MiB minimum against the 512 MiB gate).
- **Workers 28.** The r2 rule says 18 at 64k, which is recorded only, as for E29's 24.
- **Resident streams:** 8k 37 · 20k 30 · 40k 23 · 64k 18 · 90k 15 · ~112k 13 · 116k 12 · 131k 11. So 28 streams are resident only while contexts are short; past that the engine preempts, which is counted.
- **The speed-pass bands fit:** 240 / 408 / 280 / 246 blocks against 521.

### Size and timing
- Notebook 398,296 B; margin to the 400,000 B limit is 1,704 B.
- Boot ≈ 38 min with the speed pass. The 6-h window stays full if the boot ends by 46 min, i.e. at most one failed lever try.
- The smoke copy differs by one line (`E29_PROFILE = "E29SMOKE"`): budget 100 min, 40-min window, the 12 E29 smoke tasks. That covers boot + the 130k probe + the speed pass + one lever retry.

### Risks, highest first
1. **Nothing in the bundle has booted on a GPU**, and DYN/D4s/LP/SPC/IDX at 28 seqs together were never validated (NB-D1's real kit checked each flag alone at 24). Each failed try costs about 6.5 min; worst case is 7 tries plus D rungs. A rejected flag is cheap (caught before launch).
2. **SPC can cause silent accuracy loss:** prefix-checkpoint plus hybrid-state corruption is a known risk class, and only the eval score would show it.
3. **IDX fp8 on sm_89 can cause silent accuracy loss:** its numerics have never run on GPU (the FP8 QSA KV path failed in NB0), and it adds no blocks on DFlash, so its only possible gain is speed.
4. **28 workers against 11–18 resident streams** at long contexts means more preemption and recompute; this is recorded, not gated.
5. **LP may push the F2 probe below 512 MiB.** The expected outcome is that LP is dropped, costing one try.
6. **DYN may fall back to PIECEWISE graphs.** SPEC is then dropped, costing one try.
7. **The speed pass adds ~13 min to the boot.** Two or more failed tries shorten the 6-h window by a few minutes.
8. **Size margin is 1.7 kB.** An E29 rebuild that grows more than that would need more packing.
9. **Naming by error line** could drop the wrong lever if an unrelated error mentions its flag; recorded.
10. **An in-run crash unrelated to a lever** still costs a lever.
11. **The smoke runs only 12 tasks**, so it cannot fill 28 Duck streams; the speed pass covers 24 at 20k.

### Test results
- **Local unit tests** (`python -B e32_tests.py`, Py 3.14): **22/22 PASS**.
  - All 288 combinations parse and give the expected argv/env/tail through the notebook's own `e29_extra`/`e29_env`.
  - Bad values turn the lever off without raising.
  - Levers off (and payload missing) give exactly the E29 `nb5_boot` tries over 8 scripted ladders.
  - Injected failures follow the drop order above: OOM at boot, LOW_FREE, rejected flags, the DYN regex, zero-accept, XID, CONFIG_MISMATCH, the capacity gate.
  - Also covered: in-run rebuild and the MTP path, the speed pass, the read-back, the fingerprint.
  - CELL 0 / CELL 3 / cell 4 / CELL 6/7 (scheduler, writer, clock) are byte-identical to E29. The smoke is one line apart. Push folders, payload and E29 integrity checked.
- **Kit test runner:** **18/18 PASS** on Py 3.12 and 3.14. The kit-cell dry run on Py 3.12 printed `@@KIT e32-units ALL OK@@`.
- **E29 suites, run from the E29 Build folder:** `e29d_direct_tests` 34/34 OK, `e29_rev2d_adversarial_tests` 59/59 PASS, `e29_sched_tests` 29/29 OK. All 142 E29 experiment files have the same shas before and after.
- **Bug found and fixed:** the seqs hooks had a wrong signature, so `e32_call` swallowed the error and silently fell back to 24 seqs. The ladder tests now assert that no E32 hook error was swallowed.

### E29 production files: confirmed byte-identical
- `Q27-XHIGH-E29_DFLASH_D3S.ipynb` = `9a0978474fb2686209ac922c212b2c457cf7d63d7321ba4bbce3118f03bc0885`
- `Q27-XHIGH-E29_DFLASH_D3S_SMOKE.ipynb` = `19928629ac35276fa4652757bfbb4a0eddc66265e88c017a3b3cac5e43701b29`
- The prod and submit push folders and `ds_q27-e29-final-payload` are unchanged file by file; the builder asserts this before and after every build.

### What I could not do
- No GPU run, push or upload (your rule).
- **No real CPU kit.** That kit would install Merlin and run the help gate and vLLM argparse on the combined argv through the notebook's own `merlin_fn_server`. Building it means retargeting the E26 → E28 → E29 kit chain, whose checks hard-code 24 seqs, the K3 tail and the E29 captures. Only the units kit was built.
- The pending E29 scheduler decisions are not implemented (as instructed).
- `e32_tests.py` cannot run on local Py 3.12, because the E25 builder chain it imports fails its NB5 reproduction there. That is in the existing E25 builder, not E32 code; the kit tests do run on 3.12.

### Files and sha256
Experiment folder `D:\ARC AGI 2\experiments\Experiment 32 - E29 hybrid + 1500 lever bundle (planned)\`:

| File | sha256 |
|---|---|
| `README.md` | 5f39bc64… |
| `Build\build_e32_levers.py` | ecb40e82… |
| `Build\e32_core.py` | 619cd031… |
| `Build\e32_lib.py` | f0ea5856f4e11627e4343f266d840e305a0c82e3a433f38f832b27a220c23920 |
| `Build\e32_nbx.py` | 218c7f8f58e866e0144d894e81b6c91e1ef3b8dc121a3a1e30684ed9fa5d4ed2 |
| `Build\e32_checks.py` | 943f503d… |
| `Build\e32_kit_tests.py` | 44686abc… |
| `Build\e32_tests.py` | ce0a6071… |
| `Build\build_e32_cpu_kit.py` | f3837fe7… |
| `Build\e32_levers_code_export.py` (readable code) | 110606ec… |
| `Build\e29_vs_e32_export.diff` | e43a9d1b… |
| `Build\e32_build_facts.json` | 1b5fdf33… |
| `Build\e32_kit_facts.json` | 5d9389cd… |
| `Experiment Notebooks\Q27-XHIGH-E32_LEVERS.ipynb` | 41f32d5f145aa7b8fa7c5811c8a51e497ff7016d8c8c22582ab9eb95eccd02e1 (398,296 B) |
| `Experiment Notebooks\Q27-XHIGH-E32_LEVERS_SMOKE.ipynb` | 4ca333979c56511087eb71860cf760bdede8c68c07f16df80321b9df4d5a3fda (398,294 B) |

Push folders and dataset under `D:\ARC AGI 2\_scratch\push_e25\`:
- **`q27-xhigh-e32-levers__foysalemonshanto`:** notebook = the production sha; metadata b0cad008….
- **`-smoke`:** notebook = the smoke sha; metadata dd53c2f6….
- **`-submit`:** notebook = the smoke bytes; metadata 5b95edfe….
- **All three push folders:** owner foysalemonshanto, private, NvidiaL4, E29's docker image.
  - Models: E29's 4, unchanged.
  - Datasets: E29's 3 in the same order, plus `foysalemonshanto/q27-e32-levers-payload`.
- **`ds_q27-e32-levers-payload`:**
  - `e32_lib.py` and `e32_nbx.py` (shas above).
  - Two kit fixtures: `kit_q27-xhigh-e32-levers.ipynb` (41f32d5f…) and `kit_q27-xhigh-e29-dflash-d3s.ipynb` (9a097847…).
  - `SHA256SUMS` d43c450b…; `dataset-metadata.json` e33db104….
- **`q27-e32-kit-units__foysalemonshanto`** (CPU, built, not pushed): kit notebook 7afa2695e1345fc7258e0051ead7fc1de39b3b7273606913c38a635abb6e5459 (21,115 B); metadata b9f85d9d…. Its datasets are the E32 payload and `q27-e25-nb0-tf-ref`.

The README covers the design, the switch lines, the drop order and every lever's read-back.
