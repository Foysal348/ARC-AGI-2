## E29_DFLASH_D3S rev 2: DFlash2 fixed for GPU, DIRECT added as the second branch (nothing pushed or uploaded)

Only the DFlash notebook changed. I rebuilt E29_DEFAULT, E29_MTP_V64 and ALL_OFF and they are byte-identical to the audited builds. The builder asserts this before it writes anything and also checks the on-disk notebook and push copies:
- DEFAULT `5c74abf869bee204de3da97f26cfc0d9abe3fd9d5eb7cef69ae285925b871af9`
- MTP_V64 `13419f696db4054a508551f97ba7711a050a01df595b45e416704c12d1292a27`
- ALL_OFF `d7c391200bd9ab344b1fd2b4db196d13b3c79a70b2f1033372dfc7f8457f8381`

Two things had to be frozen to keep them identical:
- **e29_lib:** those notebooks keep the audited 1.0 (`Build/e29_lib_v1_0.py`, d9fba907). Only the DFlash notebook gets 1.1.
- **e29_nbdf.py:** Exp 27's `nbdf_lib.py` changed since the audited build (cd2c7281 → b0147751). The pieces E29 cuts from it are byte-identical (asserted), so `e29_nbdf.py` stays 68e975c6. The drafter gate functions are asserted at d60a80f3.

**Tests:** all local tests pass: retargeted E28 units 44/44, kit-lib 28/28, DIRECT tests 11/11. Each also passes when run from the built kit's own files in an empty folder, on Py 3.12 and 3.14.

**Biggest open point:** the payload dataset must be uploaded as a new version before any GPU commit or kit. With the old version, DFlash falls back to MTP K3 and DIRECT turns off; the run does not stop.

### New notebook and push folder
- `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Experiment Notebooks\Q27-XHIGH-E29_DFLASH_D3S.ipynb`
  - sha `814fc0f000cffa660e610d5edc8761afdd8bd2b95d36ace31131ddaf311f2cc6`, 397,808 B (2.2 kB under the 400 kB limit)
  - the same file is in `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e29-dflash-d3s__foysalemonshanto/`
  - rev 1 `c775f653` is superseded; a copy is in the 2026-11-01 audit zip
- Flags: attempts_supply, dflash_d3s, embed_host. indexer_fp8 is removed (your decision).
- To fit the size limit, the whole E29 engine block is now one sha-pinned lzma literal (the same technique as e21_cell6 / e21_cell7). The readable text is in `Build\e29_dflash_d3s_code_export.py`.

### 1. Seam 341d859b
- The seam is pinned by its own sha and asserted equal to Exp 27's current `dflash_seam.py`. It no longer depends on the NBDF-K100 facts file. The old pin 7356d473 is recorded.
- The builder also asserts that the GRD edit is part of the seam's core files (`vllm/v1/e2_guards.py`, post sha d50d754e, `query_cudagraph_manager`).
- **After the seam is applied:** if the GRD file is not applied, setup raises `SEAM_GRD_NOT_APPLIED`. That is a real failure: DFlash is dropped and the run uses MTP K3.
- **Boot read-back:** passes only with the GRD file applied and SeamSeq fully patched; otherwise the label is `NOT_INTEGRATED_GRD` and the run goes to MTP. The e2-guard log line is recorded, not gated.

### 2. EMBED_HOST from the first D try
- The `embed_host` lever existed but was OFF for this variant in rev 1. It is now ON.
- The D ladder is 7,933,526,016 → 7,665,090,560 → 7,396,655,104 → 6,859,784,192 bytes, with EMBED_HOST on every rung.
- A test with the real seam's `server_env` confirms the first try's env has `DFLASH_DRAFT_EMBED_HOST=1`. The real kit also checks it on the built server env.

### 3. OOM at any stage steps the D ladder
- **The retry condition:** E29's equivalent was in the nb5_boot drop rule. The new `df_oom_step` = an OOM line, no XID / device fault, and not a DFlash gate status. It is checked before any lever naming.
- **Bug found:** in 1.0, the word "indexer" is in every Merlin log (`qsa_indexer.py` INFO lines, the config dump, and the real OOM traceback). So with idx on, any failure dropped idx first. In 1.1 a lever counts as "named" only in an error line that contains its own flag text.
- **New stages covered:** the first request and probes (as before), a new warm-up gate (`E29_DFLASH_WARMUP_FAILED`), and a new death-before-phases gate (`E29_DFLASH_DIED_BEFORE_PHASES`). Before this, a server that died in the warm-up would still have started the phases.
- **In-run:** a DFlash server that dies of an OOM is relaunched one rung lower; with no rung left it relaunches as MTP K3 at the top rung.
- **XID:** no ladder step. DFlash is dropped at once and MTP K3 runs.
- **Ladder end:** the existing drop rule falls back to MTP K3 at the top rung with I8 restored.
- These are tested on the real NB-S1 OOM log.

### 4. Gate audit
| Gate | Acts on | Decision |
|---|---|---|
| C-11 config check | KV dtype, seqs, pin, DFlash arch line, method dflash K3 | Real failure; kept. It passes on the real NB-S1 boot log and argv; the MTP check fails on that log, as it should. |
| Seam read-back | taps, embed / mask row, EMBED_HOST, load check, in-graph meta, plus the new GRD check | Real; kept. |
| Warm-up gate (new) | server dead, request errors, a stream with 0 tokens | Real. Slow is never gated. |
| Zero-accept canary (warm-up) | at least 200 draft tokens and under 2 % accepted (the nbdf zero-accept rule) | Kept as a real failure. A low AL is not gated (tested: AL 1.3 at 10 % accepted passes). |
| In-run canary → MTP relaunch | the same zero-accept rule | Kept. |
| Hourly AL | AL below 0.7 × the first hour | Record only. A test asserts only the report reads it and that no label gate exists in the boot or run cells. |
| Lever naming | — | Was a substring heuristic; now error lines only (1.1). |
| Existing E26 / E28 gates on D tries | method map, short correctness probes | Real. The NB-S1 DFlash boot passed the method map. |

### 5. indexer_fp8
- Removed from D3S per your decision. Reasons: the kit found 0 extra KV blocks on the DFlash D-SW layout, it has never run on sm_89 GPU, and only one unproven change per run. It may come back after NB-M1L's IDX arm.
- This is recorded in the README and the facts. The lever code is unchanged for the other variants.
- **MTP_V64 still has indexer fp8 and the 1.0 naming bug.** Any failure there drops idx first, costing one extra boot try. It never kills the run, but it could not be fixed without changing MTP_V64's bytes.

### DIRECT as the second branch
- **Switch:** the first line of CELL 1 is a plain literal, outside any packed text:
  `E29_DIRECT2 = {'on': True, 'think': 104000, 'answer': 8192, 'temperature': 0.6, 'drain_min': 26}`
  - With `'on': False` the DIRECT patch is never applied, so the run is E29_DFLASH_D3S without DIRECT (tested equal).
  - A bad value turns DIRECT off and never stops the run.
- **Scheduler:** a new patch (`e29d_sched_patch.py`, in the dataset) changes arc2_duck.py from the E29 post file (d8603109) to ef3895a3, applied in CELL 3.
  - Each phase-C item runs one DIRECT request in its worker slot, in C's existing order: train-fail tasks first, then accepted tasks by ascending attempt-1 tokens.
  - A, B, supply and the deadline guard are unchanged.
  - The record is branch 1 (submissions[{test_outputs}], accepted False, train_exact []), with the E31 fields per task in the report.
- **Solver:** `e29_direct.py` = E31's `e31_direct.py` (9967a6e0) verbatim plus two marked edits:
  - audit item N1: a phase-2 resend keeps the already streamed ids (prompt = prefix + p2 ids, cap = remaining);
  - `task_index`: keeps ORDER2's seeds over the full task set, because the scheduler calls the solver one task at a time.
  - It runs on the same server URL, never relaunches the server itself, and saves full thinking under `duck/direct/<task>_b0/`.
- **Answer rule:** attempt_1 never changes. attempt_2 = DIRECT's grid for that test (grid K → test K) when it is valid and differs from attempt_1; otherwise today's E28 attempt_2. A task with no DIRECT result keeps today's rule.
- **Rerun:** uses the same switch and branches; commit profile settings are unchanged.

### Files changed or added (Build folder unless noted)
| File | sha | Notes |
|---|---|---|
| `build_e29_final.py` | 788fb068 | |
| `e29_lib.py` (1.1) | 6e5f9679 | |
| `e29_lib_v1_0.py` | d9fba907 | new, frozen copy |
| `e29_direct_build.py` | 4dee54a6 | new |
| `e29_direct.py` | 8b42802c | generated |
| `e29d_sched_patch.py` | 4cb5c4a1 | generated |
| `e29_nbs1_log_fixture.json` | 46ef4839 | server-log lines only; no traces opened |
| `e29_kit_lib_tests.py` | 7f255e6c | |
| `e29d_direct_tests.py` | b70f9efb | new |
| `build_e29_cpu_kit.py` | e971c87e | |
| `README.md` | — | rev 2 section added |
| `e29_dflash_d3s_code_export.py` | 82ee711c | |
| `e29_dflash_d3s_build_facts.json` | 8180f392 | |
| `e29_build_facts.json`, `e29_kit_facts.json` | — | updated |

Audit zip: `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Experiment Review\Audit Zips\E29_DFLASH_D3S_REV2_DIRECT_audit_FLAT_2026-11-02.zip`. All files are LF; no Exp 25 / 27 / 28 / E31 files were edited.

### Payload dataset to upload (new version)
Folder `D:/ARC AGI 2/_scratch/push_e25/ds_q27-e29-final-payload/` → `foysalemonshanto/q27-e29-final-payload`:

| File | sha |
|---|---|
| `dflash_seam.py` | 341d859b |
| `e29_direct.py` | 8b42802c |
| `e29d_sched_patch.py` | 4cb5c4a1 |
| `e29_nbs1_log_fixture.json` | 46ef4839 (kits only) |
| `draft_vocab_k100_arc_n32768.pt` | a164d1a7 (unchanged) |
| `draft_vocab_k100_arc_n65536.pt` | a915fa6d (unchanged) |
| `SHA256SUMS`, `dataset-metadata.json` | — |

MTP_V64 is not affected by the new version.

### Tests and kits
- New kit-lib checks run on the real NB-S1 logs, covering:
  - the seam sha;
  - the GRD read-back;
  - C-11 on the real log;
  - EMBED_HOST on the first try;
  - OOM after ready stepping all four rungs;
  - XID not retried;
  - ladder end → MTP;
  - the 1.1 lever naming;
  - every gate real-failure-only;
  - the in-run relaunch plan;
  - the switch line and its validation;
  - DIRECT off when the switch is off or the patch is missing;
  - DIRECT files by sha;
  - CELL 3 ordering.
- DIRECT tests (real arc2_duck, fake Duck agents, real solver against a fake streaming token-id server) cover:
  - the patch;
  - switch off equal to the old path;
  - DIRECT admitted in C order with no Duck attempt 2;
  - the attempt_2 rule for 1, 2 and 3 tests;
  - attempt_1 unchanged;
  - phase-1 resend;
  - phase-2 resend keeping the ids;
  - drain inside the solve end and no admission after the drain point;
  - an end-to-end run with Duck and DIRECT together.
- One local quirk: run with `python -I`, the DIRECT tests cannot import Duck's `requests` from user site-packages, so I ran that layout without `-I`, the way the kit runs it (same as the sched kit). The units kit was run with `-I`.
- Kits rebuilt, not pushed:

| Kit | Folder | sha | Size |
|---|---|---|---|
| units-dflash-d3s | `q27-e29-kit-units-dflash-d3s__foysalemonshanto` | c4883eb9 | 398,949 B |
| real-dflash-d3s | `q27-e29-kit-real-dflash-d3s__foysalemonshanto` | edd147ae | 352,585 B |
| direct (new) | `q27-e29-kit-direct__foysalemonshanto` | 9b663d85 | 80,616 B |

  The real kit adds GRD, EMBED_HOST and DIRECT checks (patch on the real Duck payload, hook installed, files from the dataset). The default and MTP kits were not rebuilt.

### Timeline: 240 hidden tasks in 12 h
Measured base, E28 K100 6 h on 120 tasks (MTP K3, 24 workers): 419.5 tok/s, 9.06 M tokens in total.
- **A (attempt 1):** mean 54.3 k tokens; all 120 started by 65 % of the window.
- **B2 (re-runs):** 16 tasks, 1.55 M tokens.
- **C (Duck attempt 2):** 53 started between 4.2 and 5.9 h, mean 17 k tokens, 17 still running at the window end.

12 h for 240 tasks is the same time per task, so I expect the same shape:
- **Window:** boot about 15 min (NB-S1 cold DFlash ready in 714 s; each extra rung about 4–5 min warm), leaving a solve window of about 11.4 h.
- **Capacity at the E28 speed:** about 17.2 M tokens.
  - A: about 13.0 M tokens (76 %); all A started by about 7.4 h.
  - B2: about 32 tasks, 3.1 M tokens (18 %), started about 7.4–9.8 h.
  - DIRECT: about 1.1 M tokens left (about 6 %). About 100 DIRECT requests (C1 train-fail tasks first) start from about 8 h. At ~18 tok/s per stream, a full 104 k think takes about 96 min, so only requests that start before about 9.4 h can finish one. The rest are force-closed at T−26 min, and none start after that.
- **If DFlash's steady-band speed carries over to Duck:** NB-S1 measured 763 tok/s at 20k × 24 streams vs E28's 506 tok/s production median. This is not measured in a Duck run. Capacity would be about 22–26 M tokens, giving DIRECT about 6–10 M tokens, enough for roughly 60–100 full-length requests.

### Open risks
- The dataset must be uploaded as a new version first (see above).
- DFlash has never done a full Duck run. NB-S1 proved boot and steady windows only (EMBED_HOST, 0 preemptions at 16 / 24 / 8 / 6 streams).
- DIRECT has never run on GPU, either in E31 or inside this scheduler.
- Size margins are thin: the units kit is about 1 kB under 400 kB and the notebook 2.2 kB.
- Each OOM rung costs one boot. After an in-run rung step, admission still uses the old pool size, so preemptions are possible (speed only).
- DIRECT is also active in the commit run, because the same switch applies there.
- DIRECT runs are not fed into supply's bookkeeping, so supply eligibility is as before.
