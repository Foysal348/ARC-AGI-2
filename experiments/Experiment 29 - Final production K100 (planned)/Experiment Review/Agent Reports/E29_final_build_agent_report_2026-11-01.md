## E29 FINAL production: the switchable build is done
Three notebooks are built, the CPU kits are built, and the audit zip is written. Nothing was pushed, and no git or Kaggle API calls were made. The Kaggle CPU kits have not run yet. The GPU commit needs the user's go.

### Folder: `D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/`
- `README.md`: the flag table, the shas, the fallbacks and what is still open.
- `Build/build_e29_final.py`: the builder.
  - It imports the E28 builder, so the K100Q 4506a71f and E28 base asserts still run.
  - It applies exact-once `# E29` edits.
  - `FINAL_CFG` holds every flag, and the default (all OFF) = E28 K100.
  - Run it with `python` (3.14). The E25 builder chain does not reproduce its NB5 sha under 3.12.
- `Build/e29_lib.py`: the flag helpers (stdlib only).
- `Build/e29_sched_patch.py` (scheduler S-1) and `Build/e29_sched_tests.py`.
- `Build/build_e29_cpu_kit.py`: the kits.
- `Build/<variant>_build_facts.json` and `Build/<variant>_code_export.py` (the readable code), plus `Build/e29_build_facts.json`.
- Agent reports, verbatim, in `Experiment Review/Agent Reports/`.
- Audit zip: `Experiment Review/Audit Zips/E29_FINAL_audit_FLAT_2026-11-01.zip`. It is flat, 52 files, sha `eff110ad…`.

### Notebooks (all under 400 kB; CELL 0 = E18 R2 verbatim is asserted; the E15B writer is unchanged)
| notebook | flags | sha256 | GPU push folder (foysalemonshanto, private) |
|---|---|---|---|
| (a) `Q27-XHIGH-E29_DEFAULT.ipynb` | attempts_supply | `5c74abf869bee204de3da97f26cfc0d9abe3fd9d5eb7cef69ae285925b871af9` (397,092 B) | `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e29-default__foysalemonshanto/` |
| (b) `Q27-XHIGH-E29_DFLASH_D3S.ipynb` | + dflash_d3s + indexer_fp8 | `c775f653e2353c8d1957fb2f101995388491b79619546a23466cba05f1d70210` (397,156 B) | `.../q27-xhigh-e29-dflash-d3s__foysalemonshanto/` |
| (c) `Q27-XHIGH-E29_MTP_V64.ipynb` | + draft_vocab 64k + indexer_fp8 | `13419f696db4054a508551f97ba7711a050a01df595b45e416704c12d1292a27` (397,143 B) | `.../q27-xhigh-e29-mtp-v64__foysalemonshanto/` |

- I also built a check notebook, `ALL_OFF` (sha `d7c39120…`), and it got its own push folder too. That folder is not for a GPU commit.
- **Inputs:** all three use the same Models and datasets as E28 K100. (b) adds the drafter Model `foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1`. (b) and (c) add the new dataset below.
- **Size:** E28's e21_cell7 lzma packing is kept, and e21_cell6 is packed the same way.

### Dataset to upload (needed for (b) and (c), and for their kits)
- Folder: `D:/ARC AGI 2/_scratch/push_e25/ds_q27-e29-final-payload/`.
- Slug the notebook expects: **`foysalemonshanto/q27-e29-final-payload`**.
- Contents:
  - `dflash_seam.py`, sha 7356d473 (= NBDF-K100's seam);
  - `draft_vocab_k100_arc_n32768.pt`, sha a164d1a7;
  - `draft_vocab_k100_arc_n65536.pt`, sha a915fa6d;
  - `dataset-metadata.json`.
- The notebook finds each file by name and sha under `/kaggle/input`.

### How each flag is built, and its fallback (real failures only)
- **attempts_supply (S-1):**
  - It is a second strict diff on top of the E28 patch to `arc2_duck.py` (E29 post sha `d8603109…`), gated by `E29_SUPPLY=1`.
  - When the A / B / C queues are empty and time remains, free workers start extra attempts: S1 = tasks with no train pass (T1 budget); S2 = tasks whose attempts were all cap-cut (B2 120k, window 130,560).
  - Order: fewest attempts first, at most 16 attempts per task. A, B and C items always come first. The deadline guard (`min_admit_sec`) is unchanged.
  - Answers: best train goes to attempt_1. With no supply records, the slots are identical to E28's.
  - Fallback: if the E29 apply or verify fails, the file is reverted to the original and the E28 scheduler is applied (E28's own commit / rerun rule).
- **drafter dflash:** the NBDF-K100 path.
  - It reuses `drafter_resolve` / `_sha_file` / `drafter_facts` and the nbdf_lib pieces (pins, `drafter_check`, `dflash_spec`, `capture_sizes`, `compilation_json`, `SeamSeq`), cut verbatim with ast.
  - Boot settings: `DFLASH_DRAFT_SW_BLOCK=1024`, mask row sha, BF16 drafter KV "auto" with FLASH_ATTN, captures 4..96, DRAFT_HEAD never set (I8 is suspended).
  - D pin ladder = NBDF d_retry: first the top rung, then EMBED_HOST at the same pin, then 7.665 / 7.397 / 6.860 GB with EMBED_HOST. Only an OOM moves down one rung.
  - Gates on every D try:
    - C-11 without the MTP checks, plus the `DFlash2DraftModel` line and the method check;
    - `dflash_seam.readback_label` must be ENGAGED (draft_load_check per worker, aux taps on 4 ranks, the mask row);
    - the zero-accept canary over the warm-up (≥ 200 draft tokens and < 2 % accepted).
  - Fallback: any D failure goes to the proven MTP K3 at the top rung, with I8 restored and the seam reverted.
  - In-run canary: the sampler holds the spec counters. On a hit, NB5_ACTIVE_BOOT is switched to MTP and the server is stopped, so the E15B watchdog relaunches it with MTP.
  - Hourly AL is recorded only.
- **indexer_fp8:** `--attention-config {"indexer_kv_dtype":"fp8"}`, the JSON form.
  - The kit agent found that the dotted `--attention-config.indexer_kv_dtype fp8` would fail the notebook's K-9 help gate on every try, because the dotted key is not listed in `--help`. So I switched to the JSON form.
  - Fallback: the same pin without it.
- **draft_vocab / gdn_fastpath / gdn_align:** server env.
  - Vocab = `VLLM_DRAFT_VOCAB_FILE` + `_SHA256`, MTP only. GDN = `VLLM_GDN_DECODE_FASTPATH=1` and `VLLM_GDN_SHARED_ALIGN_INDEX=1`.
  - Fallback: drop the lever, same pin.
- **workers:** 24 (E28's K100_CAPS). For 28 / 32, Merlin's `capture_sizes_for(w, 3)` is baked in. D captures = multiples of (1 + K) up to w × (1 + K).
- **General drop rule:** any pre-phase failure with an E29 lever on drops one lever. The lever the log names goes first, otherwise DFlash, indexer, vocab, GDN. This runs before E28's I8 rule.
  - A launch-prep raise (for example FAIL_CLI_FLAGS) with a lever on counts as a failed try, not a crash.
  - E18's 3-try cap counts only tries without an E29 lever.
  - With no lever left, the E28 ladder runs unchanged.
- **Timing / deadline:** the E28 logic is untouched (the 6-h commit window and the 12 h − 3 min RERUN_PROD).

### Tests
- `e29_sched_tests.py`: 29/29 on Py 3.12 and 3.14.
  - Covers: E29_SUPPLY unset == E28 post behaviour; S1 / S2 order; A / B / C priority; per-task cap; deadline guard; 0 idle seconds while supply exists; answer cases; real Arc2ToolAgent on branches 3-5.
- Kits, local unit files for each of the 3 variants:
  - retargeted E28 unit tests 44/44 (including A6: every hunk vs the E28 notebook is marked E29; nb5_boot with no lever on == E28);
  - `e29_kit_lib_tests.py` 14/14 (lib, plus the notebook's own nb5_boot on scripted tries: 5 D tries, then the MTP fallback with I8 restored, then the drop order, in commit and rerun).

### CPU kits (not pushed; `D:/ARC AGI 2/_scratch/push_e25/`, foysalemonshanto)
- **Units:** `q27-e29-kit-units-default__…` (bedc0c86), `q27-e29-kit-units-dflash-d3s__…` (65ab6cff), `q27-e29-kit-units-mtp-v64__…` (6fb0d90e).
- **Sched:** `q27-e29-kit-sched__…` (dcb283d1). It runs `e29_sched_tests` on the real mounted Duck payload and covers all three variants (the module is identical in each).
- **Real:** `q27-e29-kit-real-dflash-d3s__…` (b94ace14) and `q27-e29-kit-real-mtp-v64__…` (ee68a5aa).
  - Each runs E28's real kit on the E29 notebook.
  - Then the E29 section:
    - `e29_setup()` on the real mounts: for (b), drafter_facts on the Model, the seam from the dataset, SeamSeq FULLY_PATCHED with ppack still applied, then revert; for (c), the vocab file and its sha;
    - the notebook's own nb5_boot with a scripted try that calls the real `merlin_fn_server` (K6, build, host_phase, HELP + `cli_flags_check`);
    - vLLM argparse with `VLLM_TARGET_DEVICE=cpu` on every E29 argv;
    - for (b), the SpeculativeConfig preflight; for (c), Merlin's `load_draft_vocab_ids` (a wrong sha must be refused).
  - The step-coverage table (20 notebook steps → kit keys) is in the kit report.
- **Push order:** upload the dataset first, then the 6 kits. Foysal has 5 CPU slots, so the sixth waits or goes on another account that can see the inputs.

### Open / not done
- The real kits have never executed (only compiled and built), so the first Kaggle run may show a kit-side error.
- The E28 e2e kits (fake server, commit / rerun) are not retargeted to E29.
- DFlash2 has never had a full Duck run, and the indexer fp8 sm_89 compile is unmeasured on GPU. Both fall back automatically.
- NBDF's "no-hope" low-AL cut is not used (real-failure-only rule).
- The kit agent noted two minor marker-convention items (the CELL 1 modules-literal line and the e18-cap line carry no explicit `E29` comment). They are not functional, and the unit A6 accepts them.
- Its CRLF facts finding is fixed: all JSON is now written LF.