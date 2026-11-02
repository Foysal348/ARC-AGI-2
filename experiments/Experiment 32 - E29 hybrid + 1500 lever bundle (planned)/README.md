# Experiment 32: E29 hybrid + the 1500 lever bundle (planned)

## Goal
One notebook that runs every 1500+ speed lever from the research **together**, in one 6-h EVAL120 run, on the GPU-proven
E29_DFLASH_D3S engine. It is dual mode like every notebook of ours: the commit = EVAL120 scored, the competition rerun = every
hidden task (CELL 0 = E18 R2, RERUN_PROD, the E15B writer, the clock: all E29's, byte for byte).

- **Base:** `Experiment 29/Experiment Notebooks/Q27-XHIGH-E29_DFLASH_D3S.ipynb` (sha `9a097847…`, rev 2g): klee100 3bpw on Merlin
  (vLLM-0.32 fork) TP2 x PP2, DFlash2 D3s (K3, SW-1024, BF16 drafter KV, FLASH_ATTN, seam 341d859b + GRD), EMBED_HOST from the first
  D try, QSA 128 / 64, D ladder from 7,665,090,560 B, 24 seqs, the A / B / C scheduler + supply + DIRECT (C1), T1 104k / B2 120k /
  max-model-len 131,072, MTP K3 = the fallback.
- **Builder:** `Build/build_e32_levers.py`, a thin wrapper over `Experiment 29/Build/build_e29_final.py` (imported; its build(),
  write_dataset() and build_direct() never run; nothing in Experiment 29, its push folders or its payload folder is written).
  1. It first assembles the E29 builder's E29_DFLASH_D3S cells exactly as E29's build() does and asserts the result equals the E29
     production notebook **on disk**. The DIRECT / scheduler / nbx files come from the E29 build outputs (`Build/e29_direct.py`,
     `e29d_sched_patch.py` = the E29 builder's SCHED_PIN, `e29_nbx.py`; each = its copy in the E29 payload folder).
     **A later E29 change (e.g. the pending early-DIRECT / 60k admission decisions) = one rebuild here** (the once-only anchors
     must still match; a miss stops the build).
  2. Then the marked once-only E32 edits X1-X10 (below), the E29 packing (e21_cell7, the E29 engine block + the E21 donor block,
     e21_cell6), the size check (< 400,000 B), the SMOKE copy (one line), the push folders, the new payload folder, the facts.
  3. Before and after the build it checks the E29 production files byte for byte (the two notebooks at their pinned shas, the prod
     and submit push folders, the E29 payload folder file by file).

## The switch lines (top of CELL 1; edit freely, no rebuild)
```
E32_SEQS = 28      # 24 = off (= E29) | 28 | 32              (the S28 / S32 pair = one switch)
E32_SPEC = 'DYN'   # 'K3' = off (= E29 D3s) | 'D4s' | 'DYN'   (the D4s / DYN pair = one switch)
E32_LP = True
E32_SPC = True
E32_IDX = True
E32_G1 = True
E32_G2 = True
E32_SPEED = {'on': True, 'windows': [['8k', 16], ['20k', 24], ['90k', 8], ['116k', 6]], 'cap_min': 15}
```
- A bad / missing value = that lever OFF (recorded as `invalid->off`), never a stop. Every lever OFF = the E29_DFLASH_D3S engine
  exactly (tested: the same nb5_boot tries, names, pins, seqs, argv, env, drops, I8).
- The levers act on the **DFlash tries only**. Every MTP try (E29's fallback) is the E29 config exactly.
- Q256 (QSA 256 / 256) is a control, not a lever: it is not in the bundle.
- The speed pass is record only, commit only (the hidden rerun skips it). A bad 'on' = off; a window list without 90k + 116k =
  the default set (user rule: 90k and ~116k always run; the cap never skips them).

## Levers: apply / read-back / fallback
All argv / env / read-back pieces are NB-D1's own (`E25 Build/nbd1_lib.py`, cut verbatim by name, sha-pinned `92fd3e8b`).

| lever (default) | apply (D tries) | read-back (record; booted try) | real failure -> fallback |
|---|---|---|---|
| SEQS 28 | `--max-num-seqs 28` (nb5_boot); captures every b x (1+K) for b 1..28 (nbdf capture_sizes = NB-D1 dflash_captures); nb5_capacity counts the launched 28 as the top rung (workers = 28, as E28's 24) | logged max_num_seqs (C-11 already gates seqs vs argv) | memory order #2, CONFIG_MISMATCH on seqs, the capacity gate C-8 -> 24 (the E29 captures) |
| SPEC DYN | trailing `--speculative-config` = the E29 dflash spec + `num_speculative_tokens` 4 + `num_speculative_tokens_per_batch_size` [[1,8,4],[9,seqs,3]] (= NB-D1's [[1,8,4],[9,24,3]] at 24; the same dense table at 28 / 32, asserted); `--compilation-config` = NB-D1 dyn_captures(seqs) (b x 5 for b <= 8, b x 4 above); D4s = K4 + dflash_captures(seqs, 4) | logged num_speculative_tokens + schedule; **the DYN fallback lines ('Overriding cudagraph_mode' / 'Disabling num_speculative_tokens_per_batch_size') = a real failure** (status E32_DYN_FALLBACK) | general order #1; the DYN lines; zero-accept; CONFIG_MISMATCH on spec -> K3 (the E29 D3s spec) |
| LP on | `--max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive` (before the DFlash tail; Merlin's own 1024 earlier in the argv, last wins) | the vLLM non-default args line (2048 / 1024 / True) | memory order #1 (OOM, the F2 probe LOW_FREE), a named error line -> off |
| SPC on | `--enable-mamba-shared-prefix-checkpoint` | non-default args line flag True | general order #2, memory order #4, named -> off |
| IDX on | `--attention-config {"indexer_kv_dtype":"fp8"}` (E29L.IDX_ARGS = NB-D1's) | the workers' QSA indexer kv_cache_specs dtype (nb1_cache_info RPC), else the logged attention config | general order #3, named -> off |
| G1 on | `VLLM_GDN_DECODE_FASTPATH=1` (server env, replayed by relaunches) | /proc environ of the 4 workers | general order #4, named -> off |
| G2 on | `VLLM_GDN_SHARED_ALIGN_INDEX=1` | /proc environ of the 4 workers | general order #5, named -> off |

Every read-back is record only (`applied` True / False / None per lever in `trec["e32_readback"]` and the report), except the DYN
fallback lines (NB-S1's real-failure rule).

## The drop rule (real failures only; `e32_lib.pick`)
A pre-phase failure of a D try with an E32 lever on (launch / not ready / OOM / probes / C-11 / the DFlash read-back / the warm-up
or zero-accept canary / the F2 probe / death before the phases / a rejected flag) drops **ONE** lever and boots **the same D rung**
again. The order:
1. SPEC = DYN and a DYN fallback line in the log -> SPEC.
2. A lever whose own flag / env text is in an error line (Error / Traceback / error: / FAIL_ / unrecognized arguments) -> it
   (e.g. `FAIL_CLI_FLAGS: missing flags: --enable-mamba-shared-prefix-checkpoint` -> SPC; the help gate needs no boot).
3. E29_DFLASH_CONFIG_MISMATCH: a failing seqs check -> SEQS, a spec check -> SPEC; NO_GO_INSUFFICIENT_KV -> SEQS.
4. Memory (an OOM line that is not a device fault, or E29_DFLASH_LOW_FREE): **LP -> SEQS -> SPEC -> SPC**. No memory lever left ->
   E29's rule (the next D rung, the remaining levers kept).
5. Anything else (incl. a device fault): **SPEC -> SPC -> IDX -> G1 -> G2 -> LP -> SEQS** (newest / riskiest code path first).

E29's QSA-mismatch rule runs first (same rung, Merlin 256 / 256). With no E32 lever left, E29's rules run unchanged (D ladder,
then the MTP K3 top rung with I8, the E28 ladder). When DFlash is dropped every E32 lever still on is recorded `dropped`
(WITH_DFLASH). **In-run:** a death of a DFlash server with an E32 lever on drops one lever by the same rule (the booted config is
rebuilt: argv, tail, seqs, env = what nb5_boot builds fresh, tested) and the watchdog relaunches the same rung; a memory death with
no memory lever left = E29's in-run plan; the in-run zero-accept / no-rung path to MTP strips every E32 argv / env and the 28 seqs
(24 = the E28 top rung). Every drop is in `E32_RT["drops"]`, `E29_RT["dropped"]` (`E32:<lever>`) and a `!!!!!!!!!! E32 lever
dropped` line. Labels never gate.

The F2 first-prefill probe (130k) and the D ladder stay in force with every lever on (the probe runs on every D try; LOW_FREE is a
memory failure).

## Measurement outputs (record only)
- **Speed pass:** on the first D try that passed every gate incl. the F2 probe, before the phases, once per run, commit only:
  L.steady windows (WIN = 90 s after the prefill settles) at 8k@16, 20k@24, 90k@8, 116k@6, prefill rate = the F2 probe's measured
  rate; one `[LIVE speed <min>] <band>@<n> tok/s | AL | running | preempt | status` line per window. Expected ~12-14 min (window
  spacing from NB-S1: 2.1 / 3.7 / 4.6 / 4.6 min).
- **Run:** E29's `[E29 RUN SUMMARY]` (whole-run tok/s, per-15-min tok/s, AL, preemptions, prefix hit) + `[E32 RUN SUMMARY]` +
  `q27x/e32_report.json` + `E28_OUT["e32"]` in `q27x/e28_k100_report.json`.
- **Engine fingerprint:** argv + the engine env keys (VLLM_ / DFLASH_ / NCCL_ / NB4_ / DRAFT_ / MERLIN_ / KV_BYTES / P2P / SCALES /
  CUDA_*) + the KV pin + Merlin (tree, launch.py sha256, commit pin ad42e37f) + every lever's state, sha256 over the canonical JSON.
  One `E32 FINGERPRINT <12 hex>` line per booted D try; the short sha + the lever states in the `[LIVE start]` line and every
  `[LIVE 15-min]` line (via E29L.effective, wrapped at runtime); the full fingerprint per try in the report.
- **[LIVE]:** every E29 LIVE line is unchanged.

## Expected KV and capacity at 7,665,090,560 B (default levers)
- **Pool:** 521 blocks (E29 GPU smoke, D3s, F = 12 from the max-concurrency line). K4 / DYN: the same pool (NB-D1 real kit: 540 =
  540 at 7.93 GB for K3 vs K4). IDX: **0 extra blocks on the DFlash D-SW layout** (NB-D1 kit R9 540 = 540; the 480 -> 508 gain is
  the MTP Q layout only). SEQS: no pool change (max concurrency is seqs-independent, kit mc_by_seqs 12.25). LP / SPC: no pool
  change (the pin is fixed); LP lowers the post-probe free memory (base 903 MiB min at 7.665 GB; the gate is 512 MiB).
- **Workers 28** (launch 28 = the top-rung rule). The E28 r2 rule `floor(pool / (ceil(65536/4096) + F))` = 18 at 64k
  (RULE_BELOW_LAUNCH, recorded, as for E29's 24).
- **Resident streams** (ceil(L / 4096) + 12 blocks each): 8k 37, 20k 30, 40k 23, 64k 18, 90k 15, ~112k (T1 end) 13, 116k 12,
  131k 11. So 28 workers are resident only while contexts are short; long phases run on the engine's own scheduling (slots
  admission, preemption = recompute, counted), as E29 does with 24.
- **Speed pass bands fit:** 8k@16 240 blocks, 20k@24 408, 90k@8 280, 116k@6 246 (<= 521).

## Size and time
- Notebook 398,296 B (margin 1,704 B; E29 396,918 B). The E32 engine code lives in the payload (e32_lib + e32_nbx); the notebook
  keeps only the hooks, the E32 block inside E29's packed engine block, and the switch lines.
- Boot: setup ~4.6 min + ready ~18 min + canary / probe ~2 min + speed pass ~13 min ≈ 38 min; the fixed 6-h window is kept while the
  boot ends by 46 min (one failed E32 try ~6.5 min fits; two shorten the window by a few minutes).
- **SMOKE copy:** = the production bytes with ONE line `E29_PROFILE = "E29SMOKE"`; the E29SMOKE row here has budget **100 min**
  (E29: 70; HARD_STOP 95, deadman 100), the same 12 tasks and the 40-min window: boot + the first 130k prefill (F2) + the speed pass
  (90k / 116k long prefills with every lever) + one lever retry still keep the full window.

## Files
| file | what |
|---|---|
| `Build/build_e32_levers.py` | the wrapper builder (X1-X10) |
| `Build/e32_core.py` | the E32 pure logic (switch parse, argv / env, DFlash plan, drop rule, read-back, fingerprint, speed cfg) |
| `Build/e32_lib.py` | generated = the NB-D1 cut + e32_core (the payload file the notebook imports) |
| `Build/e32_nbx.py` | the notebook glue `_e32x_*` (exec'd into the notebook namespace by e32_setup) |
| `Build/e32_checks.py`, `Build/e32_kit_tests.py`, `Build/e32_tests.py` | the checks (shared), the kit runner, the local tests |
| `Build/build_e32_cpu_kit.py` | the CPU units kit builder |
| `Build/e32_levers_code_export.py` | the readable code cells |
| `Build/e29_vs_e32_export.diff` | the unified diff E29_DFLASH_D3S -> E32 (readable cells) |
| `Build/e32_build_facts.json`, `Build/e32_kit_facts.json` | build / kit facts |
| `Experiment Notebooks/Q27-XHIGH-E32_LEVERS.ipynb`, `..._SMOKE.ipynb` | the notebooks |

**Push folders** (`D:/ARC AGI 2/_scratch/push_e25/`, owner foysalemonshanto, private, GPU NvidiaL4, the E29 docker image):
`q27-xhigh-e32-levers`, `q27-xhigh-e32-levers-smoke`, `q27-xhigh-e32-levers-submit` (= the smoke bytes, the E15B SUBMIT
pattern). Models and datasets = E29's, unchanged and in order, + `foysalemonshanto/q27-e32-levers-payload`.

**Payload dataset to upload (NEW, before any GPU commit or kit):** `D:/ARC AGI 2/_scratch/push_e25/ds_q27-e32-levers-payload/` ->
`foysalemonshanto/q27-e32-levers-payload`: `e32_lib.py`, `e32_nbx.py` (pinned in the notebook's E32_FILES), the two kit fixtures
`kit_q27-xhigh-e32-levers.ipynb` (= this build) and `kit_q27-xhigh-e29-dflash-d3s.ipynb` (= E29 production; kit only), SHA256SUMS.
A miss = every lever off = E29_DFLASH_D3S (recorded `OFF_SETUP`). The E29 payload dataset is still required (seam, DIRECT, nbx).

**Kit (built, not pushed):** `D:/ARC AGI 2/_scratch/push_e25/q27-e32-kit-units__foysalemonshanto/` (CPU, the pinned Py3.12 image):
e32_checks on the mounted files (found by name + sha256), nb5_lib from the tf-ref dataset's k100q_payload_emb.json, the E28 / E29
modules from the notebook's own CELL 1 literal; prints `@@KIT e32-units ALL OK@@`. Datasets: q27-e32-levers-payload + q27-e25-nb0-tf-ref.

## Open / not done
- No GPU run. Every lever is unproven on GPU in this engine; the combination (and DYN / D4s / SPC / LP / IDX at 28 seqs) has never
  booted. The smoke is the first exposure.
- No real CPU kit (Merlin install + the help gate + vLLM argparse on the combined E32 argv through the notebook's own
  merlin_fn_server): it would need the E26 -> E28 -> E29 real-kit chain retargeted (its checks hard-code seqs 24, the K3 tail, the
  E29 captures). NB-D1's real kit validated each flag alone at 24 seqs (ALL_OK). A rejected flag on GPU costs one cheap try
  (FAIL_CLI_FLAGS before the launch) or one boot.
- The pending E29 scheduler decisions (early DIRECT vs B order, a 60k DIRECT admission floor) are not implemented here (by design:
  rebuild after E29 changes).

## Change log
- 2026-11-02: folder, builder, e32_lib / e32_nbx, notebooks + smoke + submit, payload folder, units kit; local tests 22/22 (Py 3.14),
  kit tests 18/18 (Py 3.12 + 3.14), E29 suites unchanged and passing.
