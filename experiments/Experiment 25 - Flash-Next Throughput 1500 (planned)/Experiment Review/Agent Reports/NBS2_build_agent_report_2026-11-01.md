## NB-S2 capacity: notebook, kits, push folders and audit zip are built (nothing pushed)

NB-S2 is built on the DFlash2 base as you specified: B = NBDF-K100's D3s, plus IDX, S28, LP, S32 and SPC, with DYN dropped. The units pass and the units kit prints `@@KIT units ALL OK@@` when run locally. The real kit compiles but has not run anywhere; it needs the Merlin venv, so its first run is on Kaggle.

**One rule slip:** I ran `git status` once (read-only, output discarded) while checking the notebook size. Nothing was changed.

### Before you push
1. **Create the new dataset `foysalemonshanto/q27-e25-nbs2-payload`** from `D:/ARC AGI 2/_scratch/push_e25/ds_e25_nbs2_payload/`. It holds the CELL 7 body (3f124167…) and `nbs2_lib.py` (f9f23786…), and the stub pins both shas. Make it visible to esfersami50, the same way tf-ref is.
   - Moving these out keeps the notebook at 343,930 B. Embedded, it was 398,312 B, too close to the 400 kB limit.
   - Without the dataset, the real kit still runs the rest on an embedded copy, but its `nbs2_payload_dataset_mounted` check fails. So the kit can only print ALL OK after the dataset exists.
2. **The sami push needs the drafter Model `foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1` visible to esfersami50.** The NBDF kit builder notes it as foysal-private. If sami cannot see it, every arm is NOT_INTEGRATED, so use the foysal push folder in that case.
3. **Budget:** the default hard stop is 175 min (150 GPU-min of arms + setup / boot), with the deadman at 180 min. The planned walk runs only **B, IDX, S28**; LP, S32 and SPC end SKIPPED_BUDGET. `python build_e25_nbs2.py --budget-min 300` is a one-flag rebuild that runs all 6 (215 min adds LP, 255 adds S32, 300 adds SPC).

### Files (all LF)
Folder: `D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/`

| File | sha256 |
|---|---|
| `Q27-XHIGH-E25_NBS2_CAPACITY.ipynb` (343,930 B, 11 cells) | **a0faf93c6ac156ee03a3bf07f6a9e1fa319091b3fa265f13ee6968fefda18d8f** |
| `Build/build_e25_nbs2.py` | d9d533b3… |
| `Build/nbs2_body_add.py` | 6cc98b09… |
| `Build/nbs2_lib.py` | f9f23786… |
| `Build/nbs2_unit_tests.py` | 3759aa37… |
| `Build/build_e25_nbs2_cpu_kit.py` | baa86207… |
| `Build/nbs2_cell7_body.py` | 3f124167… |
| `Build/nbs2_code_export.py` | cccadcdb… |
| `Build/nbs2_build_facts.json` | 9a36ecd3… |

**How it is built:** the NB-S2 builder runs the NBDF builder in its own process twice, writing only to `_scratch/nbs2_build_ref`. It asserts the closed result is 456c47e6 and the gated result is **c5c43770 byte for byte**. So B's boot path is exactly the proven NBDF-K100 one:
- merge, K6, sha gate;
- k100_setup with the scales sidecar;
- the `--help` flag gate and the launcher;
- Q-defer, plus the seam apply / verify / revert;
- the drafter gate;
- the D pin ladder (EMBED_HOST first);
- seqs 24, pin 7,933,526,016;
- the embedded NBDF-K100 KV gate (passed).

Other points:
- No NBDF file was touched; the builder checks the Exp 27 file shas and reports 0 changed.
- CELL 0 is E18 R2 verbatim (asserted).
- Every body change against NBDF-K100 is in `nbs2_body_vs_nbdf_k100.diff`, inside the zip.
- NB-S2 adds no closed gate of its own.

### Tests
| Run | Result |
|---|---|
| `nbs2_unit_tests.py`, local | 129/129 PASS |
| Units kit, local run | `@@KIT units ALL OK@@` (the notebook-file checks skip there by design) |
| Real kit | built, compiles, not run |

### Kits (foysal) and GPU push folders
Kits are in `D:/ARC AGI 2/_scratch/push_e25/`:

| Kit | Bytes | sha256 |
|---|---|---|
| `q27-e25-nbs2-kit-units__foysalemonshanto` | 142,362 | 04d12128bec15ab63b848d79e1a738ede7839421b7959d8a8d4932d51a730a9b |
| `q27-e25-nbs2-kit-real__foysalemonshanto` | 377,461 | 6b863853bb5dbc55e418d755354db25788dfa2cdbeebdb6900016b13b499461a |

The real kit attaches the same 4 Models plus tf-ref and the payload dataset.

GPU push folders, both private with the same notebook (a0faf93c), the same 4 Models and the 2 datasets:
- `q27-xhigh-e25-nbs2-capacity__esfersami50`
- `q27-xhigh-e25-nbs2-capacity__foysalemonshanto`

**What the real kit runs:** the NBDF-K100 real kit's own template, read-only, re-pointed at the NB-S2 export. It runs the notebook's own functions in `nbs2_main`'s order:
- merge, `merlin_install`, `apply_60490`, `k100_setup`, `sha_gate`;
- `PSEQ.before_arm(B)`, then `SSEQ.apply`;
- per arm, `SSEQ.check` and `PSEQ.before_arm`;
- `arm_boot`'s first try;
- `merlin_fn_server`: K6, Merlin build, host_phase, `launch_check`, and the `--help` gate with VLLM_TARGET_DEVICE=cpu.

It then checks:
- the spawned-child env, the argv fields, R10 spec preflight and R12 drafter load;
- R9 KV configs, including a new B_idxfp8 pool run and max concurrency at 21 / 24 / 28 / 32 sequences;
- every arm's final argv through vLLM's own serve parser and EngineArgs, confirming each lever differs from B by exactly its field set;
- AttentionConfig resolving fp8, and the LP / SPC config fields and the SPC activation condition;
- captures per arm and the Q-defer pack views up to 32 requests;
- then the revert.

The step-coverage table (notebook step → kit equivalent) is `REP["checks"]["NBS2_STEPS"]` and is also in the kit builder.

### Audit zip
`Experiment 25 .../Experiment Review/Audit Zips/E25_NBS2_CAPACITY_audit_FLAT_2026-11-01.zip`, sha **1a2c3910243638afa323146d068313b24df914da96502a42ee98c959b548301e**. It is flat, 33 files, with `MANIFEST_SHA256.json`. It holds all NB-S2 files, both kits, the body diff, the NBDF references and the relevant Merlin source files. `GPT_AUDIT_PROMPT_NBS2_CAPACITY.md` states the design, the gate rule, and that the only embedded gate is NBDF-K100's passed KV gate (any other closed gate is a defect).

### Arms
All arms: one change vs B, seqs 24 unless changed, pin 7,933,526,016, DFlash2 K3 with SW-1024, Q-defer, no I8.

| Arm | Change | Windows | Primary verdict | Est. min |
|---|---|---|---|---|
| B | NBDF-K100 D3s; A/A = oracle twice + ladder + TF fresh/repeat + 8k@16 twice; prefix test, sibling and stall probes as controls | 8k@16 ×2, 20k@24, 40k@16, 90k@8, 116k@6 | reference | 50 |
| IDX | `--attention-config {"indexer_kv_dtype":"fp8"}`; real failure if the indexer K cache is not fp8 (IDX_NOT_ENGAGED) | standard | any band PASS, none REGRESSION | 38 |
| S28 | `--max-num-seqs 28`, captures 4..112 | 8k@28 ×2, 20k@28, 40k@28 (all capped by residency, using the arm's seqs), 90k@8, 116k@6 | 20k@28 vs B's 20k@24, no long-band regression | 40 |
| LP | `--max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive` | standard + the stall probe | injection p95 stall ≤ 0.9 × B's, no band regression | 41 |
| S32 | as S28 at 32, captures 4..128 | as S28 at 32 | as S28 | 40 |
| SPC | `--enable-mamba-shared-prefix-checkpoint` | standard + prefix test + sibling probe | any band PASS (activation label recorded) | 42 |
| K4, PP4 | — | — | N/A_DFLASH | — |
| DYN | — | — | MOVED_TO_NB_S1 | — |

- **Recorded on every arm:** KV ledger per window (usage high-water, preemptions, recompute tokens, waiting by reason, a block estimate), K_eff per window, pool and residents.
- **Correctness for every lever:** TF at 2k / 8k / 40k against B's A/A, with TF_REQUIRED (no WIN without TF PASS); short oracle, ladder and long oracle judged relative to B's A/A.
- **Gates follow your rule:** a failure stops only its own arm, and B not usable is recorded while the levers still run. A unit test checks this on the notebook's real `nbs2_main` with fakes: B failing, a lever failing and a tree-check failure on one arm all leave the other arms running. Labels, memory, prefix, ledger and probes are record-only.

### Merlin-source findings that change an arm
1. **IDX flag spelling.** `--attention-config.indexer_kv_dtype fp8` would be refused by the notebook's `--help` flag gate, which checks literal flag strings. The arm uses the equivalent JSON form `--attention-config {"indexer_kv_dtype":"fp8"}`. The flash-next argv passes no other `--attention-config`, and the indexer accepts only bf16 / fp8 (`indexer_qsa.py:154-166`).
2. **DYN, for NB-S1.** The DFlash speculator ignores the per-step K (`dflash/speculator.py:365-470` takes `num_speculative_tokens` but never uses it). Every step drafts K_max = 4 in its FULL graph; the runner then cuts the drafts to K (`model_runner.py:2797-2803`). So on DFlash2, a D4 / D3 schedule only shrinks the target verify width; the draft pass stays at K4.
3. **SPC can activate on DFlash2.** `use_eagle` includes dflash (`speculative.py:2011`), and the coordinator flags every KV group when none is marked (`kv_cache_coordinator.py:129-135`). Partial hits work with block 1024 and prefix-match-unit 64, and there are no re-prefillable tokens. The sibling probe records activation (6,900-token prefix; a hit above 6,144 = SPC_JUNCTION_HIT).
4. **LP baseline.** Merlin's flash-next profile has CHUNK / LPT knobs, but they are unset, so B runs chunk 1024 with no long-prefill threshold (`launch.py:329, 463-464`). LP passes the flags as trailing argv; the last occurrence wins.
5. **KV ledger.** Merlin exposes no running-required or reclaimable block metric, so reclaimable is recorded as NOT_READABLE and running-required is a labelled estimate. Recompute tokens come from `prompt_tokens_by_source{source="computed"}` inside the steady window.
6. **PP4 confirmed N/A.** The seam's aux relay assumes the PP2 split (taps [6, 16 | 26, 36, 46]).
