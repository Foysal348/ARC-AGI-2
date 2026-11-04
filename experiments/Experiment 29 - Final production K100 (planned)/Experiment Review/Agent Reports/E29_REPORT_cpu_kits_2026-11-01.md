# E29 FINAL CPU kits: build report (2026-11-01)

Nothing was pushed. I made no git or Kaggle API calls. No whole kit or notebook ran locally. `build_e29_final.py` is unchanged.

## Files
- Kit builder: `D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build/build_e29_cpu_kit.py`, sha256 `d220ec6fa0a0cb8290b924120b919e1dd0234f2a6b4c8083175194367bbf3869` (LF, 0 CR).
- New unit file: `Build/e29_kit_lib_tests.py`, sha256 `af24b203d7bf169a876018c6f5f17c064e6ae2688d3274af0ec50ed0dbcdc203`.
- Build record: `Build/e29_kit_facts.json` (builder sha, E28 kit sha, notebook shas, every kit folder / sha / bytes, local results).
- `Build/_kit_local/`:
  - `facts/` holds LF copies of the variant facts (see bug 1).
  - `<variant>/` holds the local unit-run directories.
- Builder chain: it execs `build_e28_k100_cpu_kit.py` (sha `E28_KIT_SHA` in the record), which execs `build_e26_k100q_cpu_kit.py`. Each variant gets exact-once E29 edits:
  - the facts path points to the variant's facts (the LF copy);
  - SLUG / OWNER are set to `q27-e29-kit-*` / foysalemonshanto;
  - `E28_READABLE` unpacks both packed literals (E29 `e21_cell6` and E28 `e21_cell7`);
  - the modules check covers all 5 packed modules (`e29_nbdf.py` is taken from the notebook literal, sha = facts).

## Kit folders (all under `D:/ARC AGI 2/_scratch/push_e25/`)
All kits are private, with no GPU and no internet, on the docker image `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…` (same as E28). Each prints `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: …@@`.

| folder | bytes | sha256 | datasets / models |
|---|---|---|---|
| `q27-e29-kit-units-default__foysalemonshanto` | 380,697 | 4ed93aca81b0… | tf-ref, e15b-payload / none |
| `q27-e29-kit-units-dflash-d3s__foysalemonshanto` | 381,044 | e530666f5abe… | + q27-e29-final-payload / none |
| `q27-e29-kit-units-mtp-v64__foysalemonshanto` | 380,595 | 7e984559abff… | + q27-e29-final-payload / none |
| `q27-e29-kit-sched__foysalemonshanto` | 211,764 | 75d0f46b87e4… | tf-ref, e15b-payload / none |
| `q27-e29-kit-real-dflash-d3s__foysalemonshanto` | 339,703 | 603e1f400f2e… | tf-ref, e15b-payload, q27-e29-final-payload / klee100, fp8ple, merlin-env, **dflash2v2 bf16-embed** |
| `q27-e29-kit-real-mtp-v64__foysalemonshanto` | 339,178 | 96696f449325… | tf-ref, e15b-payload, q27-e29-final-payload / klee100, fp8ple, merlin-env |

The dataset and model lists are the notebook's own `kernel-metadata` lists. All kits must see the new dataset `foysalemonshanto/q27-e29-final-payload`, so it has to be uploaded first.

## What each kit checks

### units-{default, dflash-d3s, mtp-v64}
This is E28's units body, with these changes:

1. **`e29_e28k100_unit_tests.py`**: E28's `e28k100_unit_tests.py` retargeted to the E29 notebook with exact-once edits:
   - `cell6_text` / `cell7_text` pick the packed literal by its `compile()` label;
   - READ unpacks both literals;
   - A4: the 5 modules match the Build files and the facts, and `e21_cell6` matches the facts sha;
   - A5: every hunk vs K100Q is marked E28 or E29;
   - **new A6**: every hunk vs the E28 K100 notebook (sha checked against `facts.e29.e28_notebook_sha256`) is marked E29. The only exception is the E28_SRC modules-literal line, whose shas A4 already checks;
   - C3 / C4 / G2 check their E29 anchor text;
   - G3 / G4 run the notebook's nb5_boot together with its e29_env / e29_extra / e29_ntries / e29_drop defs, with no E29 lever on (it must equal E28);
   - unchanged checks include A2 (CELL 0 == E18 R2 cell 1, verbatim), B1-B7, C1-C11, D1-D5, E1-E9 and F1-F2 (the scheduler on the payload's real arc2_duck.py).
2. **`e29_kit_lib_tests.py`** (14 checks):
   - L1: the notebook's E29_CFG literal matches the facts and `check_cfg`, and the packed e29_lib matches the file.
   - L2: check_cfg passes all 4 builder variants and raises on 6 bad configs.
   - L3: df_tries matches the NBDF-K100 ladder (q_pin + d_retry, cut from `nbdf_k100_cell7_body.py`, EMBED_HOST on the retry rungs).
   - L4: lever_to_drop order, named-in-log first.
   - L5: df_spec_argv / strip_df_argv round trip.
   - L6: canary on synthetic counters.
   - L7: inrun_zero, hourly_al, next_try_index, is_oom.
   - L8: df_config_check.
   - L9: the notebook's e29_nbdf spec / captures, 4..96 = NBDF q_captures max.
   - B1-B4: the notebook's own nb5_boot with df+idx, idx+vocab, or nothing on, using scripted tries:
     - B1: 5 D tries at the NBDF pins, 24 seqs, the dflash tail last, the seam env, no DRAFT_HEAD; then df dropped and the MTP top rung with I8 restored + idx; then idx dropped and plain E28. Run for commit and rerun (E18 3-try cap).
     - B2: a NOT_ENGAGED failure goes straight to MTP; a FAIL_CLI_FLAGS naming the indexer drops idx at the same D pin.
     - B3: vocab env on the top rung, then the drop order idx, vocab.
     - B4: DEFAULT is identical to E28.
   - B5: the E29 edit anchors.
3. E28's checks of the tf-ref version, the payload and the Duck payload (pre sha).

### sched
- Runs `e29_sched_tests.py -v` with `sys.executable` (never `-I`).
- `E29_TEST_E28_DIR` points to a bundled e28dir holding `e28_sched_patch.py`.
- Inputs: the Duck payload is `find_payload()` on the mounted q27-e15b-payload (sha 9e4b8928 checked). The REAL K100Q (60) and W4 branch_result.json records are bundled. Eval challenges come from the competition mount, `nb5_lib.py` is bundled in exports/, and `E28_TEST_TMP=K`.
- Gates: all tests pass with no skips, and the test count is at least the number of `def test_` in the file.
- The module is asserted byte-identical across the three variants' facts, so a single sched kit covers b and c.

### real-{dflash-d3s, mtp-v64}
This is E28's real kit on the E29 notebook. The notebook's cells 0-4 run in the commit and rerun namespaces on the real mounts, followed by everything E28 / K100Q check:
- payload loading from the tf-ref files; inputs / merge / overlay / K6 / prompts;
- merlin_install, #60490, sha gate;
- Q-defer (`nb4_ppack_tests` + the unit test's real-tree section, apply / revert / injected-failure restore / re-apply, the rerun-namespace already_applied);
- k100_setup (worker ext, scales sidecar tree / dataset / model dir);
- I8 static + Merlin's INT8 quantizer on real lm_head rows;
- every E28 boot plan through merlin_fn_server + vLLM argparse;
- the MM probe; CELL 0 vs E18 R2; strict-schema submissions.

The scheduler check is retargeted: E29 S-1 applied (path E29, sha_after = E29 post, `E29_SUPPLY=1`), then revert to the exact manifest bytes, then re-apply through E29S.

**E29 section, in order:**
1. **Setup:**
   - The modules are loaded and the E29_CFG literal matches the facts.
   - `e29_setup()` runs on the real mounts, with levers on as the cfg says and nothing dropped.
   - **(b) DFlash:**
     - drafter_facts on the real drafter Model: pins ok, embed in checkpoint, bf16-embed dir;
     - dflash_seam.py loaded by q27x_payload from the new dataset (sha 7356d473, module file sha);
     - SeamSeq applied and tree_state FULLY_PATCHED, with nb4_ppack still applied;
     - spec = dflash K3 / drafter dir / auto / FLASH_ATTN; captures 4..96; mask-row sha; SW block 1024; I8 suspended N/A_DFLASH.
   - **(c) vocab:** the 64k file was found in the new dataset with the right sha.
2. **The notebook's own nb5_boot**, with a scripted `nb5_boot_try`:
   - Each try calls the real `merlin_fn_server` (K6, launch.build(), host_phase(), HELP + `KL.cli_flags_check`). The kit asserts at build time that it mirrors nb5_boot_try's E29 except-branch verbatim.
   - Scripted outcome: OOM while an E29 lever is on, DONE without one. This walks every D try, every lever drop and the MTP fallback try, using NB5_BOOT_EXTRA / NB5_BOOT_LEVERS exactly as nb5_boot builds them (e29_env / e29_extra).
   - Per-try checks:
     - D pins and order; the D tail last; seqs 24;
     - seam env in NB5_BOOT_LEVERS and on the server env; no DRAFT_HEAD on D tries;
     - the dflash spec / captures in the argv;
     - the MTP fallback = top pin, I8 int8, no DFLASH env, mtp spec;
     - vocab env (c); idx argv present exactly when idx is on;
     - no server started; the last try is plain E28 top rung.
3. **The HELP gate:** every try must have `cli_flags.ok == True` with no launch-prep raise, and the indexer flag must reach a built argv.
4. **vLLM's own argparse** (Merlin venv, `VLLM_TARGET_DEVICE=cpu`) on every built E29 argv: parse ok, no unknown flags, `from_cli_args` ok, max_num_seqs 24, kv bytes, max_model_len 131072, `attention_config.indexer_kv_dtype == fp8` when idx is on, dflash K3 spec + captures on D tries, mtp K3 otherwise.
5. **(b)** NBDF's SpeculativeConfig preflight (SPECPROBE + FakeL4 PLATFORM, cut verbatim from `build_e27_nbdf_cpu_kit.py`) on the notebook's dflash spec.
6. **(c)** Merlin's `load_draft_vocab_ids(path, 248320, tp_size=2)` with `VLLM_DRAFT_VOCAB_SHA256`: n = 65536, sorted and unique, a wrong sha is refused, and the env names are `VLLM_DRAFT_VOCAB_FILE` / `_SHA256`.
7. **(b)** `SSEQ.finish()`: seam reverted to ORIGINAL with ppack still applied.
8. Recorded: the launch argv of every try.
9. A **step-coverage table** (`REP["step_coverage"]`, printed in notebook order). It is also a gate: every step must be covered.

### e2e kits
I did not build the e2e kits. The E28 e2e driver hard-codes the E28 scheduler post sha and the E28 report checks, so retargeting it to S-1 / E29 means rewriting its sched checks. The units and real kits came first.

## Step-coverage table (notebook order -> kit check -> result on Kaggle)
| # | notebook pre-launch step | kit check(s) (real kit verdict keys) |
|---|---|---|
| 1 | CELL 0 mode + fallback submission (E18 R2), commit + rerun | cell0_same_test_file_and_fallback_as_e18_r2_real_mount, submission_commit/rerun_strict_full_schema_identity, profile_commit, profile_rerun_RERUN_PROD_source_test_all_tasks (+ units A2) |
| 2 | CELL 1 modules literal (5 modules) + E29_CFG + profiles | cells_before_6_7_ran_commit_and_rerun, e29_modules_loaded_cfg_eq_facts_no_sched_error (+ units A4, L1, B1-B6) |
| 3 | CELL 1 tf-ref pins + K100Q payload blobs | tf_ref_k100q_version_mounted, tf_ref_every_staged_file_sha_ok, payload_loaded_from_dataset_files, payload_sha_ok_under_input |
| 4 | CELL 3 Duck payload + manifest + sandbox | duck_cell_loaded, sandbox_selftest_ok_incl_memory_limit |
| 5 | CELL 3 scheduler E29 S-1 apply / verify (-> E28 on failure), env; revert / re-apply | e28_sched_applied_commit_rerun_revert_reapply (retargeted to E29); units F1/F2 + sched kit |
| 6 | nb5_rc2_inputs | inputs_found |
| 7 | nb5_rc2_prompts: merge, overlay, K6, prompt hashes | klee100_merge_overlay_eq_v2_model_dir_k6, prompt_hashes_ok_klee100_tokenizer |
| 8 | merlin_install | merlin_install_ok |
| 9 | nb5_apply_60490 | patch_60490_applied_verified |
| 10 | sha_gate | k7_k3_digests_all_ok |
| 11 | k100_setup (worker ext, Q-defer, scales) | k100_setup_ran_on_real_tree, worker_ext_…, scales_sidecar_…, qdefer_* (7 keys) |
| 12 | e28_i8_static + INT8 quantizer | e28_i8_static_supported_real_klee100_installed_tree, e28_i8_merlin_int8_quantizer_on_real_lm_head_rows |
| 13 | e29_setup (b: drafter / seam / SeamSeq / ppack / spec / I8 suspend; c: vocab) | e29_setup_levers_on_as_cfg_nothing_dropped + (b) e29_drafter_facts…, e29_seam_from_new_dataset…, e29_seamseq_applied…, e29_ppack_still_applied…, e29_df_spec…, e29_df_captures…, e29_i8_suspended_on_dflash; (c) e29_vocab_64k_file_found… |
| 14 | nb5_boot E28 ladder plans -> merlin_fn_server + argparse | commit_rungs_equal_facts, rerun_rungs_e18_first_3, plans_every_rung…, host_phase_ok_all_plans, help_cpu_every_flag_present_all_plans, argv_env_…, vllm_argparse_fields_all_plans, no_server_started, e28_first_rung…int8_head |
| 15 | nb5_boot with E29 levers (D ladder / drops / MTP fallback) | e29_nb5_boot_every_try_planned_d_ladder_drops_mtp_fallback, e29_k6_host_phase_ok_every_built_try (+ units B1-B4) |
| 16 | HELP gate per E29 try (FAIL_CLI_FLAGS) | e29_help_gate_cli_flags_ok_every_try, e29_indexer_flag_reached_a_built_argv |
| 17 | vLLM argparse on every E29 argv (argv recorded) | e29_vllm_argparse_every_built_e29_argv |
| 18 | (b) dflash spec preflight / (c) vocab loader | e29_dflash_spec_preflight_speculativeconfig_fields_literals / e29_merlin_load_draft_vocab_ids_64k_tp2_sha_env |
| 19 | worker-side MM method-map probe | method_map_probe_real_merlin_objects_pass, method_map_src_anchors_ok |
| 20 | (b) seam revert (kit cleanup = the notebook's revert path) | e29_seam_revert_exact_merlin_files_ppack_kept |

The result column is computed on Kaggle (PASS / FAIL: <keys> / NOT_COVERED).

## Local build results (python 3.14)
- `python build_e29_cpu_kit.py --local-units` built all 6 kits. Each is under 400 kB, compiles, and has 0 CR.
- `e29_e28k100_unit_tests.py` passed **44/44** for each of default, dflash-d3s and mtp-v64. Locally E4 ran too, because the local klee config and Merlin source are present; on the kit E4 skips by design (min = 43).
- `e29_kit_lib_tests.py` passed **14/14** for each of the three variants.
- Not run locally:
  - `e29_sched_tests.py`: its runtime is unknown, and it runs on the sched kit.
  - The real kits: user rule. Their E29 section has only been compiled, never executed, so a first-run kit-side error on Kaggle is possible.

## Notebook bugs found (build_e29_final.py not edited)
1. **The facts JSON is written as CRLF.**
   - Where: `build_e29_final.py:644` (`vf.write_text(...)`, per-variant facts) and `:661` (`facts_f.write_text(...)`, e29_build_facts.json); also `:601` (dataset-metadata.json).
   - Effect: on this Windows box the files come out CRLF (357-383 CR bytes each). The E26 / E28 kit chain reads facts with `lf_text`, which asserts no CR, so the chain cannot read the facts as written.
   - Fix: `vf.write_bytes((json.dumps(f28, indent=1, default=str) + "\n").encode("utf-8"))`, and the same for the other two.
   - Workaround in the kit builder: LF copies in `Build/_kit_local/facts/`, with the JSON asserted equal.
2. **Likely: the indexer lever never runs (the HELP gate drops it).**
   - The flag: `e29_lib.IDX_ARGS = ["--attention-config.indexer_kv_dtype", "fp8"]` (`e29_lib.py:29`).
   - The gate: `merlin_fn_server` runs `L.help_check("merlin", argv[:serve+1], env, argv)` (e23_lab.help_check: every `--` token of the argv must be a substring of `serve --help=all`), then `KL.cli_flags_check` → `RuntimeError("FAIL_CLI_FLAGS")` (E26 K100 wrapper, export ~l.1907-1910).
   - Why it fails: vLLM's help lists `--attention-config` (arg_utils.py:1856). Dotted sub-keys only appear in the generic JSON tip (`--json-arg.key1`), so `--attention-config.indexer_kv_dtype` is not in the help text.
   - Effect: every try with idx on gets `E29_LAUNCH_PREP_FAILED`. `lever_to_drop` names "indexer_kv_dtype" and drops idx on the first try in both DFLASH_D3S and MTP_V64. Nothing breaks, but the indexer-fp8 lever is silently dead.
   - Confirmation: the real kit checks this on Kaggle (`e29_help_gate_cli_flags_ok_every_try` and `e29_indexer_flag_reached_a_built_argv` will be NOT OK if it is real).
   - Fix (either):
     - (a) an E29 edit in `build_e29_final.py` engine() on `HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)` that passes `[a.split(".", 1)[0] if a.startswith("--") else a for a in argv]`, so the stem `--attention-config` is checked; vLLM's argparse already validates the key;
     - (b) `IDX_ARGS = ["--attention-config", '{"indexer_kv_dtype": "fp8"}']`. Merlin's launch.py emits no `--attention-config`, so nothing would be overwritten. The kit's argparse check covers either form.
3. **Minor, marker convention only (not functional).**
   - What: the CELL 1 `E28_SRC = json.loads(...)` modules-literal line changes with no E29 marker, and the E5 "e18 cap" line is marked only by `e29_ntries`.
   - Handling: A6 allows the literal line (A4 checks its shas) and counts e29_* names as markers.

## Push order (for the lead / user)
1. Upload the dataset `ds_q27-e29-final-payload` as `foysalemonshanto/q27-e29-final-payload`.
2. Push the 6 CPU kits on foysalemonshanto (5 kernel slots per account, so the sixth waits for one slot or goes to a second batch).
3. GPU commit only on the user's go.
