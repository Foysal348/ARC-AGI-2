# Experiment 29: final production K100 (switchable build)

## Goal
One builder for the final submission notebook. Tonight's GPU results set only the flags.
- **Base:** E28 K100 EVAL120 (klee100 3bpw-MTP on Merlin TP2 x PP2, MTP K3 + Q-defer + I8, 24 workers, A / B / C scheduler, T1 104k / B2 120k / max-model-len 131,072, CELL 0 = E18 R2 verbatim, the E15B writer).
- **Builder:** `Build/build_e29_final.py`. It imports the E28 builder (`Experiment 28/.../Build/build_e28_k100.py`) and applies marked `# E29` edits.

## Flags (`FINAL_CFG`; all OFF = E28 K100)
| flag | values | what it does | fallback on a real failure |
|---|---|---|---|
| `drafter` | `mtp` / `dflash_d3s` / `dflash_d4s` | NBDF-K100 DFlash2 boot path: drafter Model `foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1` (pinned shas), `dflash_seam.py` 341d859b (rev 2; was 7356d473) via SeamSeq, seam env (DFLASH_SEAM=1, DFLASH_MASK_ROW_SHA, DFLASH_DRAFT_SW_BLOCK=1024), BF16 drafter KV, dflash spec JSON, captures (1 + K) x 1..workers, I8 off, D pin ladder (EMBED_HOST, then 0.25-GiB rungs) | boot failure / C-11 / seam read-back not ENGAGED / zero-accept canary -> MTP K3 at the top rung (I8 restored, seam reverted); in-run canary -> MTP relaunch through the watchdog |
| `indexer_fp8` | bool | `--attention-config {"indexer_kv_dtype":"fp8"}` (the JSON form of `--attention-config.indexer_kv_dtype fp8`: the K-9 help gate only knows listed flags) | same try without it |
| `embed_host` | bool (dflash only) | `DFLASH_DRAFT_EMBED_HOST=1` from the first D try | (part of the D ladder) |
| `draft_vocab` | None / `32k` / `64k` (MTP only) | `VLLM_DRAFT_VOCAB_FILE` + `VLLM_DRAFT_VOCAB_SHA256` | same try without it |
| `gdn_fastpath` / `gdn_align` | bool each | `VLLM_GDN_DECODE_FASTPATH=1` / `VLLM_GDN_SHARED_ALIGN_INDEX=1` | same try without it |
| `workers` | 24 / 28 / 32 | top-rung workers = `--max-num-seqs`; 28 / 32 use Merlin `capture_sizes_for(w, 3)` | (E28 ladder) |
| `attempts_supply` | bool | scheduler S-1 (`Build/e29_sched_patch.py`): with A / B / C queues empty, idle workers start extra attempts (S1: no train pass, T1; S2: all cap-cut, B2 120k); best train -> attempt_1 | E29 apply / verify fails -> revert -> E28 scheduler |

**Drop rule:** a pre-phase failure with an E29 lever on drops one lever, then retries. The lever named in the log goes first; otherwise the order is DFlash, indexer, vocab, GDN. With no E29 lever left, the E28 ladder (including its I8 rule) runs unchanged.

## Notebooks (2026-11-01)
| notebook | flags | sha256 | bytes |
|---|---|---|---|
| (a) `Q27-XHIGH-E29_DEFAULT.ipynb` | attempts_supply | `5c74abf869bee204de3da97f26cfc0d9abe3fd9d5eb7cef69ae285925b871af9` | 397,092 |
| (b) `Q27-XHIGH-E29_DFLASH_D3S.ipynb` (rev 2) | + dflash_d3s + embed_host + DIRECT switch (indexer_fp8 removed) | `d3aab2e74f63172f6af79183762363bd78c5279654293b889bb805a99999ec34` | 396,385 |
| (c) `Q27-XHIGH-E29_MTP_V64.ipynb` | + draft_vocab 64k + indexer_fp8 | `13419f696db4054a508551f97ba7711a050a01df595b45e416704c12d1292a27` | 397,143 |
| check build `Q27-XHIGH-ALL_OFF.ipynb` | none | `d7c391200bd9ab344b1fd2b4db196d13b3c79a70b2f1033372dfc7f8457f8381` | 397,060 |

- **GPU push folders** (foysalemonshanto, private): `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e29-<default|dflash-d3s|mtp-v64>__foysalemonshanto/`.
- **Inputs:** the same Models / datasets as E28 K100. (b) adds the drafter Model; (b) and (c) add the dataset below.
- **Payload dataset to upload:** `D:/ARC AGI 2/_scratch/push_e25/ds_q27-e29-final-payload/` as `foysalemonshanto/q27-e29-final-payload`. It holds `dflash_seam.py`, `draft_vocab_k100_arc_n32768.pt` and `draft_vocab_k100_arc_n65536.pt`. The notebook finds each file by name + sha under `/kaggle/input`.
- **Size:** E28's e21_cell7 packing is kept, and e21_cell6 is packed the same way, so every notebook stays under 400 kB.

## Kits
- **Builder:** `Build/build_e29_cpu_kit.py`, a wrapper over the E28 kit chain.
- **Kits:** units x 3, sched, real-dflash-d3s and real-mtp-v64, under `D:/ARC AGI 2/_scratch/push_e25/q27-e29-kit-*__foysalemonshanto/`.
- **Local results:** retargeted E28 units 44/44, e29 lib / nb5_boot units 14/14 for each variant, and e29_sched_tests 29/29 on Py 3.12 and 3.14.
- **Not run yet:** the real kits; they run on Kaggle.
- **e2e kits:** not built.

## Open / not done
- The real kits have not run.
- The E28 e2e kits have not been retargeted, so there is no fake-server e2e of the E29 drop rule. The scripted nb5_boot units cover it.
- The DFlash and indexer GPU behaviour is unmeasured. (b) is DFlash's first full Duck run.
- NBDF's "no-hope" low-AL cut is not used, by the real-failure-only rule. Only the zero-accept canary gates; hourly AL is recorded.

## Rev 2 (2026-11-02): E29_DFLASH_D3S only (DFlash2 that works on GPU + DIRECT as the 2nd branch)
DEFAULT `5c74abf8`, MTP_V64 `13419f69` and ALL_OFF `d7c39120` are rebuilt and asserted byte-identical (`AUDITED` in the builder; nothing is written for them). They keep `e29_lib` 1.0 (`Build/e29_lib_v1_0.py`, `d9fba907`). The E27 `nbdf_lib.py` moved on (`cd2c7281` -> `b0147751`), but the pieces E29 cuts are byte-identical (asserted), so `e29_nbdf.py` stays `68e975c6`.

| item | value |
|---|---|
| notebook `Q27-XHIGH-E29_DFLASH_D3S.ipynb` (rev 2e) | `d3aab2e74f63172f6af79183762363bd78c5279654293b889bb805a99999ec34`, 396,385 B (rev 2 `814fc0f0`, 2b `beebb252`, 2c `ed2e23a7`, 2d `80d4aeff` superseded) |
| GPU smoke `Q27-XHIGH-E29_DFLASH_D3S_SMOKE.ipynb` | `4b6bfd631716f1a360eabe61a19782a4928b5a6e40ec7117511a0ec9436b06f1`, 396,383 B = the production bytes with ONE line: `E29_PROFILE = "E29SMOKE"`; push folder `q27-xhigh-e29-dflash-d3s-smoke__foysalemonshanto` (rev 1 `c775f653` superseded; it is kept in the 2026-11-01 audit zip) |
| flags | attempts_supply + dflash_d3s + **embed_host** (indexer_fp8 **removed**) |
| seam | `dflash_seam.py` `341d859b` = E27's current file, with the GRD edit of Merlin `vllm/v1/e2_guards.py` (`query_cudagraph_manager`, post `d50d754e`). It is pinned by its own sha; the old pin `7356d473` is recorded. |
| `e29_lib` 1.1 | `99f97226` (DFlash notebook only; 2c: + mtp_top / QSA; 2d: + probe / seam / effective / submission_check helpers) |
| size | the whole E29 engine block is ONE packed lzma literal in this notebook (the `e21_cell6` / `e21_cell7` technique); the readable text is in `Build/e29_dflash_d3s_code_export.py` |

What the NB-S1 GPU runs proved, and what E29 now does:
- **B1 (capture death):** with the old seam every D arm died at capture. Now `e29_df_setup` checks the GRD file state after SeamSeq apply (raises `SEAM_GRD_NOT_APPLIED`). The boot read-back requires GRD applied + SeamSeq `FULLY_PATCHED`; otherwise the label is `NOT_INTEGRATED_GRD`, a real failure that sends the run to MTP K3. The e2-guard `query_cudagraph_manager` log line is recorded only.
- **Memory:** without EMBED_HOST the first request OOMs on PP1. EMBED_HOST is now on from the FIRST D try. The D ladder is 7,933,526,016 -> 7,665,090,560 -> 7,396,655,104 -> 6,859,784,192 B, with EMBED_HOST on every rung.
- **OOM ladder:** `df_oom_step` = an OOM line and not XID and not a DFlash gate. It is checked BEFORE any lever naming, at any stage of a D try: launch, first request / probes, the new warm-up gate (`E29_DFLASH_WARMUP_FAILED`: dead / request errors / an empty stream), and the new pre-phase death gate (`E29_DFLASH_DIED_BEFORE_PHASES`). In-run, `e29_relaunch_plan` in `e21_relaunch` moves to the next D rung; with no rung left it uses MTP K3. XID = no D-ladder step: DFlash is dropped and MTP K3 runs.
- **Lever naming fix (1.1):** a lever is named only by an error line that contains its own flag text. In 1.0, "indexer" appears in every Merlin log, so any failure named idx. MTP_V64 still has 1.0 (open item).

### DFlash gates (real failures only)
| gate | acts on | decision |
|---|---|---|
| C-11 (`df_config_check`) | KV dtype / seqs / pin + DFlash arch line + method dflash K3 | real (config mismatch); passes on the real NB-S1 log |
| seam read-back (`readback_label` + GRD + SeamSeq) | taps, embed / mask row, EMBED_HOST, load check, ingraph meta, GRD file | real |
| warm-up gate (new) | server dead, request errors, a 0-token stream | real; slow = not gated |
| zero-accept canary (warm-up) | >= 200 draft tokens and < 2 % accepted (nbdf structural zero-accept rule) | real; a low AL is not gated |
| in-run canary -> MTP relaunch | the same zero-accept rule on the sampler rows | real |
| hourly AL | AL < 0.7 x first hour | record only (read only in the report) |
| in-run OOM (new) | OOM line, not XID, after the phases started | next D rung / MTP K3 |

### indexer_fp8 (removed from this variant, user decision 2026-11-02)
On the DFlash D-SW layout the kit measured 0 extra KV blocks, and the setting has never run on sm_89 GPU. The rule is one unproven change per run. It may come back after the NB-M1L IDX arm. The lever code stays for the other variants. MTP_V64 still carries it (with the 1.0 naming caveat above).

### DIRECT = the second branch (attempt_2), user fixed spec 2026-11-02
- **Switch:** the first line of CELL 1, a plain literal: `E29_DIRECT2 = {'on': True, 'think': 104000, 'answer': 8192, 'temperature': 1.0, 'drain_min': 26}`. `'on': False` means the DIRECT diff is never applied, so the run is E29_DFLASH_D3S without DIRECT (tested: same records / slots / report). A bad value turns DIRECT off and never stops the run.
- **Scheduler:** `e29d_sched_patch.py` (`cfb5834c`, payload dataset) applies a strict diff on arc2_duck.py, E29 post `d8603109` -> `ef3895a3`. Each phase-C item runs ONE DIRECT request in its worker slot (hook `E12A.E29D_RUN`; CELL 3 calls `e29d_sched_patch.cell3`), in C's priority order. A, B, supply and the deadline guard are unchanged. The record is branch 1 (`submissions[{test_outputs}]`, accepted False, train_exact []). Report: `SCHED_LAST["e28"]["direct"]` + the e29 key.
- **Solver:** `e29_direct.py` (`8b42802c`, payload dataset) = E31 `e31_direct.py` `9967a6e0` verbatim, plus N1 (a phase-2 resend keeps the streamed phase-2 ids: prompt = prefix + p2_ids, cap = remaining) and `task_index` (ORDER2 seeds over the full task set). Full thinking is saved under `duck/direct/<task>_b0/`.
- **Answers:** attempt_1 never changes. attempt_2 = the DIRECT grid for that test (grid K -> test K) when it is valid and differs from attempt_1; otherwise today's rule. With no DIRECT result, today's rule applies.

### Payload dataset (must be uploaded as a NEW VERSION before any GPU commit / kit)
`D:/ARC AGI 2/_scratch/push_e25/ds_q27-e29-final-payload/` -> `foysalemonshanto/q27-e29-final-payload`: `dflash_seam.py` 341d859b, `e29_direct.py` 8b42802c, `e29d_sched_patch.py` 4cc807f2 (rev 2d, the SCHED agent's F1), `e29_nbx.py` 1388aa90 (2c: the DIRECT + LIVE notebook functions; a miss = DIRECT and LIVE off), `e29_nbs1_log_fixture.json` 46ef4839 + `k100q_fixture.json` f59cee51 + `k100q.ipynb` 4506a71f (kit tests only), the two vocab tables (unchanged), `SHA256SUMS`. If the old version stays mounted:
- the DFlash setup fails and the run uses MTP K3;
- DIRECT turns off;
- MTP_V64 is not affected.

### Kits (rev 2, built, not pushed)
| kit | folder | sha256 | bytes |
|---|---|---|---|
| units-dflash-d3s | `q27-e29-kit-units-dflash-d3s__foysalemonshanto` | `d429b84c2df0702b…` | 377,275 |
| real-dflash-d3s | `q27-e29-kit-real-dflash-d3s__foysalemonshanto` | `ad71f8af8e885101…` | 363,433 |
| direct (new) | `q27-e29-kit-direct__foysalemonshanto` | `f51640fb6b131ff9…` | 84,724 |

Local results: the retargeted E28 units 44/44, e29 kit-lib 44/44 (30 R checks incl. rev 2d R20-R28, rev 2e R29-R30), and e29d_direct_tests 20/20 (SCHED's F1 tests). Each also ran in an empty folder from the kit's own blob, on Py 3.12 and on 3.14 (units with `python -I`; direct as the kit runs it).

## Change log
- 2026-11-01: folder, builder, the S-1 scheduler, kits. The indexer flag was switched to the JSON form after the kit agent found that the help gate would drop it.
- 2026-11-02 rev 2e (the GPU smoke): CELL 1 line 3 `E29_PROFILE = "E28EVAL120"` (production default unchanged) + the `E29SMOKE` row (= the production row except: budget 70 min, the 12 eval tasks plain sorted, the fixed 40-min window; HARD_STOP 65, deadman 70; a boot ending by 21 min keeps the full window; build-time + runtime assert). Smoke tasks: 1818057f 2ba387bc 45a5af55 bf45cf4b (K100 solved, short A) | 446ef5d2 (2 tests) b6f77b65 (3 tests) 7b5033c1 800d221b (K100 failed) | 31f7f899 d59b0160 409aa875 a47bf94d (mixed). The report prints `[SMOKE SUMMARY]` (`[E29 RUN SUMMARY]` in production) + JSON (`e29_lib.run_summary`, record only). The payload dataset is unchanged vs rev 2d.
- 2026-11-02 rev 2d (the rev 2c audits, FIX_LIST_E29_REV2C_AUDITS): F1 hook `E12A.E29D_CAN_START` (late C -> Duck C; SCHED's patch 4cc807f2); F2 forced 130k first-prefill probe + worker free (RPC) on every DFlash try before the phases (OOM -> D-ladder step, < 512 MiB -> one step); F3 the 4-rank QSA read-back required before the first prefill (mismatch / missing rank -> the same rung with Merlin 256/256 + EMBED_HOST); F4 the in-run MTP relaunch runs C-11 + method map + I8 read-back (INT8 not engaged -> BF16 relaunch); F10 in-run device fault first -> MTP (GPU_DEVICE_FAULT recorded); F11 pre-apply seam snapshot, verify + re-restore after finish, unverified -> q_seam_inert on the MTP boot (never a stop of the run); F5 / F14 effective_algorithm (+ experiment_validity DIRECT_OFF in EVAL) at the start, in the report and LIVE; F6 / F13 host cgroup + KV ledger (ready, after the probe, LIVE 15-min); F12 submission_check after every writer callback and at the end; F9 the verbatim E21 donor functions packed as a 2nd literal (393,048 B). Units kit: the K100Q notebook staged from the payload dataset.
- 2026-11-02 rev 2c (user-approved fixes): (1) after DFlash is dropped (ladder end, XID, a gate, GRD, in-run) the NEXT try is the E28 production top rung (MTP K3, 7,933,526,016 B, 24 seqs, INT8 head restored; `e29_lib.mtp_top`), then E28's own ladder; the in-run switch to MTP uses the same rung + I8 (`e29_mtp_boot`). The rev 2b real kit's pin0 / 6 seqs / I8-off fallback came from the kit not mirroring CELL 8 (NB5_PIN #60490 + I8 want unset before e29_setup: E28's fail-safe start = the RC2 pin); the kit now mirrors it. E29_MTP_V64 has no D tries, so the notebook bug does not apply there, but its (not rebuilt) real kit has the same kit-mirroring gap. (2) CELL 1 line 2 `E29_QSA = {'max_logits_mb': 128, 'granule_mb': 64}` -> `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` / `VLLM_E2_QSA_LOGITS_GRANULE_MB` on the DFlash tries only (after Merlin's build(), via NB5_BOOT_LEVERS). The MTP fallback keeps Merlin's GPU-proven 256 / 256. Record-only read-back of the 4 workers' spawn env (/proc environ) + the logged granule; backing bound <= 128 MiB by Merlin's formula. (3) size: the DIRECT + LIVE notebook functions moved to the payload module `e29_nbx.py` (executed into the namespace by `e29_extras()`).
- 2026-11-02 rev 2b (user): DIRECT temperature 1.0 (switch default; top_p 0.95 / top_k 20 / min_p 0 sent explicitly, as before). LIVE progress lines, print only: one `[LIVE <min> min]` line per finished attempt (task, phase A / B1 / B2 / C-DIRECT / supply, train k/n, exact vs gold only when SOLUTIONS are loaded, per-phase totals, queries correct + task-norm from the CURRENT build_submission selection; DIRECT: mode, tokens, grids, attempt_2 hits) and a `[LIVE 15-min]` line (server tok/s, running streams). `e29_lib.LiveLog` + `e29_live_setup` wrap `E12A.solve_all_r6`'s `on_branch_done`: the original callback runs first, every LIVE exception is caught. CELL 3's DIRECT step moved into `e29d_sched_patch.cell3` (size). The units kit takes the K100Q fixture from the payload dataset (kit size).
- 2026-11-02 rev 2: E29_DFLASH_D3S only: seam 341d859b + GRD read-back, EMBED_HOST first, OOM-any-stage ladder + in-run plan, 1.1 lever naming, indexer fp8 removed (user), DIRECT as the 2nd branch (switch, E31 solver + N1, the C-phase diff), the engine block packed; kits rebuilt.
