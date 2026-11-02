## NB-S1 (DFlash2 only): notebook and units are done; the real kit is not built, so the notebook cannot open yet

NB-S1 follows your last two messages: DFlash2 only, B = NBDF-K100 D3s, arms G1 / G2 / EH / F8 / DYN, plus the profile on B. There is no MTP base, no REF arm, and no V32 / V64 / V64I8 / I4 arm. Those are listed as N/A_DFLASH (REF as removed) in the notebook flags and records.

The build is closed (`gpu_ready: false`), which is correct. **The NB-S1 real CPU kit has not been built.** Until a real-kit report exists for this notebook and it is rebuilt with `--kv-gate-report`, B is recorded SKIPPED_KV_KIT_GATE and nothing launches (the same rule as NBDF). So the step-coverage table for the real kit is not done this round.

Nothing was pushed and git was not run. No NBDF / K100 file was changed: the builder hashes 13 frozen files before and after the build and asserts they are identical. The files are `nbdf_lib.py`, `build_e27_nbdf.py`, `build_e27_nbdf_k100.py`, the two kit builders, `dflash_seam.py`, both bodies, the K100 export, both facts files and both NBDF notebooks.

### Outputs

| Item | sha256 | Bytes |
|---|---|---|
| Notebook `Experiment 25/Experiment Notebooks/Q27-XHIGH-E25_NBS1.ipynb` | **6b3be0b193164b94cdd0f8c2545d7f79c8ec04e23f7c5d12a2657f0ecfb67a89** | 381,669 (< 400 kB) |
| Units kit `D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbs1-kit-units__hosen42/` | a879c9f5919e7c2ccb6e572bb9adcca5a88ad9c87742e07353782352d0e02563 | 379,858 |
| Audit zip `Experiment 25/Experiment Review/Audit Zips/E25_NBS1_audit_FLAT_2026-11-01.zip` (flat; includes the read-only NBDF dependencies for context) | 9e6555c130ec5079dd815c289a4796665893e2a94859b193f80aca2f20fc37f4 | 635,723 |

- **GPU push folder:** `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nbs1__hosen42/`. Its metadata id is `hosen42/q27-xhigh-e25-nbs1`, private, L4, the pinned image.
- **Mounts it expects:**
  - the klee100 3bpw-MTP model;
  - the FP8 PLE overlay;
  - the Merlin env;
  - the DFlash2 drafter `foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1`;
  - the dataset `foysalemonshanto/q27-e25-nb0-tf-ref`.
- **No new dataset to upload.** With DFlash2 only, the vocab tables are not used.
- **Account access:** every source is owned by foysalemonshanto, and the drafter is private. The hosen42 account must be able to see all of them before the GPU push.
- **CELL 0:** this is E18 R2 verbatim. The NBDF builder asserts the cell and its sha.

### Files (in `Experiment 25/Experiment Notebooks/Build/`)

| File | sha256 (prefix) | Role |
|---|---|---|
| `build_e25_nbs1.py` | 50c1f7eb | Builder: see "How it is built" |
| `nbs1_lib.py` | 1a9c1eae | Arm plan, `install()`, `check_arms()`, `base_gate()`, read-backs, profile plan, records. Stdlib only, embedded in the notebook, sha-checked at run time |
| `nbs1_unit_tests.py` | 34e4f1f8 | 56 tests |
| `nbs1_cell7_body.py` | b08ca1f2 | Readable body (generated) |
| `nbs1_code_export.py` | 2d99b85e | Export (generated) |
| `nbs1_build_facts.json` | — | Build facts |
| `build_e25_nbs1_units_kit.py` | 110128f5 | Units kit builder |

### How it is built
`build_e25_nbs1.py` runs the NBDF builder exactly the way NBDF-K100 is built (`NBDF_VARIANT=k100`). It points the K100 module's output names, slug and title at NB-S1 in memory only, then wraps `edit_body` / `edit_meta` / `k100_gate_extra`: the K100 edits run first, then the NB-S1 edits.

- **K100_CFG, in memory:** `q_i8 = False` (S4: no draft head on DFlash2), and `wkeys` uses `profile_20k@24`.
- **Arm plan:** after `check_arm_plan`, `S1.s1_arms(N, ARMS["D3s"], ARMS["Q"])`, then `S1.install` and `S1.check_arms`. Any violation raises, so nothing boots.
- **Why B uses the key "Q":** B runs under nbdf_lib's base key "Q", named `nbs1_b`. Every A/A, the TF envelope, k100_rel and the verdict are built on that key, so B's A/A and TF A/A are the reference for every arm.
- **arm_body:**
  - B, being DFlash2, now reads its DFlash read-back (`is_q and not is_d`).
  - New: each arm's own change is read back before the boot gates. A real failure marks that arm NOT_INTEGRATED and stops only that arm.
  - The GDN keys are added to the worker env RPC.
  - The profile is 20k@24 on B only, with a 30 s load. The torch trace stays bounded at 40 iterations, as in NBDF.
- **nbdf_main:** B now gets the D-arm prerequisites before it launches (drafter ok, kit gate passed, seam applied). Every other line is NBDF-K100's: the seam apply / verify / revert, the D loop where each arm runs its own checks, the deferred reruns, and `finalize`, which now also writes `nbs1_records`.
- **Gate:** the K100 gate (it includes D3s per-K success), plus three NB-S1 real-kit keys (`nbs1_arm_plan_ok`, `nbs1_launch_path_all_arms_ok`, `nbs1_help_flag_gate_real_ok`). A future NB-S1 real kit must emit these; the gate can never be opened by hand.

### Arm table (each arm = B + one change; run in this order; no cross-arm skip)

| Arm | Change vs B | Launch | Real-failure read-back (that arm only) |
|---|---|---|---|
| **B** (key Q, `nbs1_b`) | NBDF-K100 D3s exactly: seam, drafter mount, `DFLASH_DRAFT_SW_BLOCK=1024`, BF16 drafter KV, FLASH_ATTN, no DRAFT_HEAD, seqs 24, pin 7,933,526,016, D ladder with EMBED_HOST first; A/A and TF A/A; profile 20k@24 for 30 s | D3s argv | no GDN switch in any worker; plus NBDF's D gates |
| **G1** | `VLLM_GDN_DECODE_FASTPATH=1` | B argv | the switch is "1" on all 4 workers |
| **G2** | `VLLM_GDN_SHARED_ALIGN_INDEX=1` | B argv | the switch is "1" on all 4 workers |
| **EH** | `DFLASH_DRAFT_EMBED_HOST=1` at the same pin | B argv | `embed_host_as_planned` is True; pool blocks and residents are recorded |
| **F8** | drafter KV fp8_e4m3 on TRITON_ATTN + `DFLASH_DRAFT_SW_BLOCK=2048` | spec changed | `draft_backend` (TRITON_ATTN on both PP1 ranks) and `draft_sw_block_applied` are True |
| **DYN** | `num_speculative_tokens` 4 + `num_speculative_tokens_per_batch_size` [[1,8,4],[9,24,3]]; captures b×5 for b ≤ 8 and b×4 for b 9-24 | spec + compilation changed | no "Overriding cudagraph_mode" line (FULL graphs kept) and no "Disabling num_speculative_tokens_per_batch_size" line |

- **Measurements on every arm:** NBDF-K100's item list unchanged. That means smoke + ladder, the greedy oracle vs B's A/A (k100_rel), the long oracle, the prefix test, windows 8k@16 ×2, 20k@24, 40k@16, 90k@8 and 116k@6, TF at 2k / 8k / 40k vs B's TF A/A with TF_REQUIRED, real-failure gates only, and the deferred NBDF reruns (Q-defer-off isolation, nometa, tap).
- **Records per arm:** AL, per-position acceptance, tok/s / tick / preemptions per window, pool blocks, residents, pin and tries, drafter released MiB, the S1 read-back, TF and oracle status.
- **Profile on B:** the stage tick split comes from NBDF's existing trace analysis. **Draft ms and verify ms per token are not parsed separately yet;** only what the existing analysis gives is recorded.
- **Time budget:** this is from nbdf_lib's per-arm estimates. B takes about 50 min (Q's estimate, because of the A/A items) and each lever arm about 40 min. That is roughly 250 min plus 6 boots and the install. Each arm's own `arm_fits` check uses the session time left; there is no 150-min cap in this DFlash2 build.

### Merlin source checks (ad42e37)
- **F8 can be configured, so it stays.**
  - `v1/attention/backends/triton_attn.py:302-345,381`: TRITON_ATTN accepts fp8 / fp8_e4m3 KV, block % 16, sliding window and non-causal attention, on any compute capability (sm_89 included). FA2 on sm_89 has no FP8 KV path.
  - The seam's `DFLASH_DRAFT_SW_BLOCK` accepts any divisor of 4096, so 2048 works.
  - **Risk:** a Triton FP8 sliding-window decode at scale 1.0 (the #54690 class). F8's own smoke / oracle / TF gates would catch it, for that arm only.
- **DYN:**
  - `config/speculative.py:493` and `v1/spec_decode/dynamic/utils.py` define the field: inclusive ranges, clamped to K = 4.
  - `config/vllm.py:1068-1085` forces PIECEWISE graphs unless the V2 model runner is used. NBDF runs on the V2 GPU runner, so FULL graphs should be kept; the read-back makes a fallback a real failure.
  - `v1/worker/gpu/cudagraph_utils.py:283-300` captures every tier.
  - **Not confirmed in source:** that the DFlash2 speculator itself handles a per-batch K. The scheduler picks K per batch, so DYN is the most likely arm to fail at its smoke.

### Tests

| Run | Result |
|---|---|
| `nbs1_unit_tests.py` (local, on the real nbdf_lib / nb4t_lib / nb4_ppack and the built body) | **56/56 PASS** |
| NBDF `nbdf_k100_unit_tests.py` (unchanged files) | 63/63 |
| NBDF `nbdf_unit_tests.py` (unchanged files) | 281/281 |
| NB-S1 units kit (local run) | the tests pass 56/56; the kit line says NOT OK only because it requires Python 3.12 and the local Python is 3.14 |

What `nbs1_unit_tests.py` covers:
- The arm plan, and `check_arms` refusing a second change.
- `nbdf_lib.launch_check` passing all six planned launches. It refuses a GDN or EMBED_HOST variable on an arm that does not plan it, and refuses DRAFT_HEAD (S4).
- The DYN captures.
- `install()`.
- Every real-failure read-back, and `base_gate`.
- `nbdf_main` on the built body:
  - B then G1, G2, EH, F8, DYN launch in order.
  - When B fails, every arm still launches.
  - A failed arm never skips the next one. NBDF's deferred Q-defer-off isolation rerun (G1_ISO) runs at the end.
  - No drafter, a closed gate, or a failed seam apply means nothing launches, with the right status for B.
- `nbdf_lib.verdict` over the NB-S1 arms.

One thing the tests caught: `launch_check` refuses the SW-block env on B until `install()` has run. The notebook calls `install()` before any launch, so the launch path is fine.

### Still to do before a GPU commit
1. **NB-S1 real kit**, mirroring the notebook's launch path. The only way that keeps the NBDF files read-only is to derive it from the text of `build_e27_nbdf_cpu_kit.py`, in memory, the same way this builder derives the notebook. That means:
   - point it at NB-S1's notebook, facts and export, with slug `q27-e25-nbs1-kit-*`;
   - keep the K100 substitutions (the sidecar step, `VLLM_TARGET_DEVICE=cpu` for `--help`, the `PSEQ` / `SSEQ` calls, launch_check at the pin);
   - replace the K100 checks that name Q / D3s / D4s (captures, spec method, R9 runs) with NB-S1 ones: B = K3_swblock1024, DYN = K4_swblock1024, and new KV probes for F8 (fp8 drafter KV, SW 2048) and EH (host embedding);
   - emit the three `nbs1_*` keys and the step-coverage table.
2. Push the units and real kits to Kaggle, then do the gated rebuild: `python build_e25_nbs1.py --kv-gate-report <NB-S1 real report>`.
3. Give hosen42 access to the foysalemonshanto models and dataset.
4. Optionally, parse draft ms and verify ms per token from B's profile trace.

**Separate observation:** the K100 notebook on disk is now **c5c43770** with `gpu_ready` True. Its facts and push folder were rewritten at 19:10, which looks like the gated rebuild after its kit; the NB-S1 build did not touch it (the frozen-file check covers it). NB-S1 was built from the unchanged K100 builder code, so this rebuild does not affect NB-S1.
