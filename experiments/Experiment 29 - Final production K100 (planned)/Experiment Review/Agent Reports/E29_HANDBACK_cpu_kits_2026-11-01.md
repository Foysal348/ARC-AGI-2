# Subagent a334ea2179e33951c hand-back (verbatim)

## E29 FINAL CPU kits: report

I built 6 CPU kits; nothing was pushed, and no git, no Kaggle API calls and no local kit or notebook runs. Local results: `e29_e28k100_unit_tests.py` 44/44 PASS and `e29_kit_lib_tests.py` 14/14 PASS for all three variants. The real kits and the sched tests have not run anywhere yet. I found two notebook bugs; `build_e29_final.py` is unchanged.

Full report (English, LF): `D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Experiment Review/Agent Reports/E29_REPORT_cpu_kits_2026-11-01.md`

### Files
- Kit builder: `D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build/build_e29_cpu_kit.py`, sha256 `d220ec6fa0a0cb8290b924120b919e1dd0234f2a6b4c8083175194367bbf3869`.
- New unit file: `Build/e29_kit_lib_tests.py`, sha256 `af24b203d7bf169a876018c6f5f17c064e6ae2688d3274af0ec50ed0dbcdc203`.
- Build record: `Build/e29_kit_facts.json`.
- Local run dirs and LF facts copies: `Build/_kit_local/`.

All files are LF with 0 CR. The builder wraps the E28 kit chain (E28 kit builder, which wraps the E26 K100Q kit builder) by exec, with exact-once E29 edits:
- the facts path points to the variant facts;
- slug / owner are `q27-e29-kit-*` / foysalemonshanto;
- the readable cells unpack both packed literals (E29 e21_cell6 and E28 e21_cell7);
- the modules check covers all 5 modules.

### Kit folders
All under `D:/ARC AGI 2/_scratch/push_e25/`. All are private, no GPU, no internet, on the E28 pinned image.

| folder | bytes | sha256 |
|---|---|---|
| `q27-e29-kit-units-default__foysalemonshanto` | 380,697 | 4ed93aca… |
| `q27-e29-kit-units-dflash-d3s__foysalemonshanto` | 381,044 | e530666f… |
| `q27-e29-kit-units-mtp-v64__foysalemonshanto` | 380,595 | 7e984559… |
| `q27-e29-kit-sched__foysalemonshanto` | 211,764 | 75d0f46b… |
| `q27-e29-kit-real-dflash-d3s__foysalemonshanto` | 339,703 | 603e1f40… |
| `q27-e29-kit-real-mtp-v64__foysalemonshanto` | 339,178 | 96696f44… |

- Every kit mounts the notebook's own datasets and models.
- The b/c units kits and both real kits also mount `foysalemonshanto/q27-e29-final-payload`, so that dataset must be uploaded first.
- The real-dflash kit also mounts the drafter Model `qwen38-flash-next-dflash2v2/transformers/bf16-embed/1`.

### What each kit checks

**units (×3)**
- E28's `e28k100_unit_tests.py`, retargeted to the E29 notebook:
  - A4: all 5 modules match their files, and e21_cell6 matches its sha.
  - A5: every hunk vs K100Q is marked E28 or E29.
  - New A6: every hunk vs the E28 K100 notebook (sha checked) is marked E29.
  - C3 / C4 / G2 check their E29 anchor text.
  - G3 / G4 run nb5_boot with the E29 defs and no lever on, which must equal E28.
  - A2 (CELL 0 == E18 R2) and the rest are unchanged.
- `e29_kit_lib_tests.py`:
  - check_cfg on all variants;
  - df_tries == NBDF-K100 d_retry;
  - canary, lever_to_drop, strip_df_argv, next_try_index, df_config_check, the e29_nbdf spec / captures 4..96;
  - the notebook's own nb5_boot with df+idx, idx+vocab and default on scripted tries (5 D tries, then MTP K3 fallback with I8 restored, then the drop order, in commit and rerun).
- E28's tf-ref, payload and Duck checks.

**sched**
- `e29_sched_tests.py -v` with `sys.executable` (never `-I`), with `E29_TEST_E28_DIR` pointing at a bundled `e28_sched_patch.py`.
- Inputs: the real Duck payload from the mounted q27-e15b-payload (sha 9e4b8928), the bundled real K100Q / W4 branch records, the competition eval challenges, `nb5_lib.py`.
- Gate: all pass, no skips.
- The scheduler module is asserted identical across all three variants, so one sched kit covers a, b and c.

**real (b, c)**
- E28's real kit, unchanged in what it covers: cells 0-4 in the commit and rerun namespaces on the real mounts, install, #60490, sha gate, Q-defer, k100_setup / scales, I8 static + INT8 quantizer, every E28 plan + argparse, the MM probe, CELL 0 vs E18 R2, submissions.
- The scheduler check is retargeted to E29 S-1: E29 post sha, `E29_SUPPLY=1`, revert and re-apply through E29S.
- Then the E29 section:
  1. `e29_setup()` on the real mounts.
     - (b): drafter_facts on the drafter Model; seam sha 7356d473 loaded from the new dataset; SeamSeq FULLY_PATCHED with ppack still applied; spec, captures, mask row, SW 1024, I8 suspended.
     - (c): the 64k vocab file plus its sha.
  2. The notebook's own nb5_boot with a scripted nb5_boot_try. Each try calls the real `merlin_fn_server` (K6, build, host_phase, HELP + `KL.cli_flags_check`). The scripted OOM / DONE outcomes walk every D try, every lever drop and the MTP fallback. Per-try argv and env checks; the launch argv is recorded.
  3. HELP-gate verdict for every try.
  4. vLLM argparse with `VLLM_TARGET_DEVICE=cpu` on every E29 argv: indexer_kv_dtype fp8, dflash K3 spec + captures, mtp otherwise.
  5. (b): NBDF's SpeculativeConfig preflight. (c): Merlin's `load_draft_vocab_ids(path, 248320, tp_size=2)` with the sha env, and a wrong sha must be refused.
  6. (b): seam revert to ORIGINAL, ppack kept.
- The report carries the step-coverage table and gates on it.

**e2e:** not built. The E28 e2e driver hard-codes the E28 scheduler sha and report, so retargeting it means rewriting its sched checks. Units and real came first.

### Step-coverage table (notebook order -> kit verdict keys; PASS / FAIL is computed on Kaggle)
1. CELL 0 + fallback submission -> cell0_same_test_file…, submission_commit/rerun_strict…, profile_commit / profile_rerun (+ units A2)
2. CELL 1 modules literal + E29_CFG -> cells_before_6_7_ran…, e29_modules_loaded_cfg_eq_facts_no_sched_error (+ A4, L1)
3. CELL 1 tf-ref pins + payload -> tf_ref_*, payload_loaded_from_dataset_files, payload_sha_ok_under_input
4. CELL 3 Duck + sandbox -> duck_cell_loaded, sandbox_selftest…
5. CELL 3 scheduler E29 S-1 apply / verify / revert / re-apply -> e28_sched_applied_commit_rerun_revert_reapply (retargeted) + units F1/F2 + sched kit
6. nb5_rc2_inputs -> inputs_found
7. prompts / merge / overlay / K6 -> klee100_merge…, prompt_hashes…
8. merlin_install -> merlin_install_ok
9. #60490 -> patch_60490_applied_verified
10. sha_gate -> k7_k3_digests_all_ok
11. k100_setup (worker ext, Q-defer, scales) -> k100_setup…, worker_ext…, scales_sidecar…, qdefer_* (7 keys)
12. I8 static + INT8 quantizer -> e28_i8_static…, e28_i8_merlin_int8_quantizer…
13. e29_setup -> e29_setup_levers_on…; (b) drafter / seam / SeamSeq / ppack / spec / captures / I8-suspend keys; (c) e29_vocab_64k…
14. E28 ladder plans -> rungs, plans, host_phase, help, argv_env, argparse, no_server_started, int8 first rung
15. nb5_boot with E29 levers -> e29_nb5_boot_every_try…, e29_k6_host_phase_ok… (+ units B1-B4)
16. HELP gate -> e29_help_gate_cli_flags_ok_every_try, e29_indexer_flag_reached_a_built_argv
17. argparse on the E29 argvs -> e29_vllm_argparse_every_built_e29_argv
18. (b) spec preflight / (c) vocab loader
19. MM probe -> method_map_*
20. (b) seam revert -> e29_seam_revert_exact_merlin_files_ppack_kept

### Local build results (python 3.14, `build_e29_cpu_kit.py --local-units`)
- All 6 kits built; each is under 400 kB, compiles, and has 0 CR.
- `e29_e28k100_unit_tests.py`: 44/44 PASS for each of the 3 variants. E4 also ran locally; on the kit it skips by design, so the kit minimum is 43.
- `e29_kit_lib_tests.py`: 14/14 PASS for each of the 3 variants.
- The real kits' E29 section has only been compiled, never executed, so the first Kaggle run may surface a kit-side error.
- `e29_sched_tests.py` was not run locally; it runs on the sched kit.

### Notebook bugs (build_e29_final.py, not edited)
1. **The facts JSON is written as CRLF.**
   - Where: `build_e29_final.py:644` (`vf.write_text`, per-variant facts) and `:661` (`facts_f.write_text`, e29_build_facts.json); `:601` (dataset-metadata.json) too.
   - Effect: the files carry 357-383 CR bytes each, and the E26 / E28 kit chain's `lf_text` refuses them.
   - Fix: `write_bytes((json.dumps(...) + "\n").encode("utf-8"))` at all three places.
   - Workaround in the kit builder: LF copies under `Build/_kit_local/facts/`, with the JSON asserted equal.
2. **Likely: the indexer lever is silently dropped on every try.**
   - The flag is `e29_lib.py:29` `IDX_ARGS = ["--attention-config.indexer_kv_dtype", "fp8"]`.
   - The notebook's K-9 gate in `merlin_fn_server` runs `L.help_check("merlin", argv[:serve+1], env, argv)`, which requires every `--` token to be a substring of `serve --help=all`; `KL.cli_flags_check` then raises `FAIL_CLI_FLAGS`.
   - vLLM's help lists only `--attention-config` (arg_utils.py:1856); dotted keys appear only in the generic JSON tip.
   - So the first try with idx on gets `E29_LAUNCH_PREP_FAILED`, `lever_to_drop` names "indexer_kv_dtype", and idx is dropped in both DFLASH_D3S and MTP_V64. Nothing crashes, but indexer fp8 never runs.
   - The real kits will confirm it on Kaggle (`e29_help_gate_cli_flags_ok_every_try` / `e29_indexer_flag_reached_a_built_argv` NOT OK).
   - Fix (either):
     - (a) an E29 edit on `HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)` that passes `[a.split(".", 1)[0] if a.startswith("--") else a for a in argv]`;
     - (b) `IDX_ARGS = ["--attention-config", '{"indexer_kv_dtype": "fp8"}']`. Merlin's launch.py emits no `--attention-config`, so nothing would be overwritten.
3. **Minor, marker convention only, not functional.** The CELL 1 `E28_SRC` modules-literal line changes without an E29 marker, and the E5 e18-cap line is marked only by `e29_ntries`. A6 tolerates both: the literal's shas are checked by A4, and e29_* names count as markers.

### Push order
1. Upload `ds_q27-e29-final-payload` as `foysalemonshanto/q27-e29-final-payload`.
2. Push the 6 kits on foysalemonshanto. It has 5 kernel slots, so the sixth waits for a slot.
3. GPU commit only on the user's go.
