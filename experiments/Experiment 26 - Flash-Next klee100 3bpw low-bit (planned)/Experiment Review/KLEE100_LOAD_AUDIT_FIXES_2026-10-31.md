# klee100 LOAD notebook (sha 78b94c49) — verified fix list from 3 GPT audits (2026-10-31)

All three audits are NO-GO as-is and GO after the fixes. Every claim below was checked in the source (`Experiment 25 …/Experiment Notebooks/Build/`: `k100load_code_export.py`, `k100load_lib.py`, `build_e25_klee100_load.py`).

Verdict column:
- **CONFIRMED** = reproduced in code.
- **PARTIAL** = the mechanism exists but does not gate.

| # | Sev | Source | Finding (verified) | Fix |
|---|---|---|---|---|
| K-1 | HIGH | A1 H1 | **CONFIRMED.** `tf_suite()` / `sc_suite()` check only the global `left()`, not the per-boot `boot_left()` / `due()`. A 90k TF request can start after the boot body cap; the watchdog then kills at cap + 300 s. | Gate every TF / SC request on `due(EST[band] + 300)` → `SKIPPED_BOOT_BUDGET`. Keep `long_prefill()` as is. |
| K-2 | HIGH | A1 H2, A3 H2 | **CONFIRMED.** `k2_pin_from_k()` returns `SAME_PIN_AS_K` when K's free memory is missing. Partial cards or NaN pass too. | Missing, partial (cards ≠ {0,1,2,3}) or non-finite → K2 uses the base BF16 pin `KV_DEFAULT`, `rule FALLBACK_BASE_PIN_UNVERIFIED_HEADROOM`, `headroom_verified False` (diagnostic; recorded in the verdict). Valid four-card data → the existing logic. |
| K-3 | HIGH | A1 H3 | **CONFIRMED.** `k100_capacity()` computes `n40` / `n64` / `resident_bound` for any P, even when `pool_check` is MISMATCH / GAUGE_ONLY / UNKNOWN. | `capacity_ok` only when `pool_check == "MATCH"` and F is measured. Otherwise `n40 = n64 = resident_bound = None` and `why KV_POOL_<chk>` / `F_NOT_MEASURED`. K2 runs 40k@N and the 64k admission only when `capacity_ok is True`; 8k@8 still runs as a diagnostic. |
| K-4 | HIGH | A1 H4, A2 F1, A3 H1 | **CONFIRMED.** `ref_load()` / `prompt_identity()` results are not in GATES; K0 → K → K2 boot even when the NB0 reference is missing or corrupt. | Before install / K0, add three gates: `GATES["REFERENCE_DATASET"]` (`ref.ok is True`); `GATES["PROMPT_IDENTITY_READY"]` (every TF and SC band `used` ∈ {rendered, nb0_ids}, all bands present); `GATES["RC2_ORACLE_IDENTITY"]` (all `rc2_oracle_prompts_match` true). The later GATES assignments must not overwrite them. |
| K-5 | HIGH | A1 H5, A2 F1, A3 MED | **CONFIRMED.** `tf_label()` accepts `PARTIAL` and returns CLOSE. `tf_reading()` returns CLOSE from a subset of bands (only an `incomplete` key). `final_label()` returns `GPU_LOAD_PASS + TF_NOT_MEASURED`. | `tf_label`: status ≠ MEASURED → `INCONCLUSIVE_<status>`; `n` ≠ `expected` (when known) → `INCONCLUSIVE_INCOMPLETE_POSITIONS`. `tf_reading`: any of 2k / 8k / 40k / 90k not measured → reading `TF_INCOMPLETE`, plus `worst_measured` and `missing`; a 90k dropped on purpose under budget → `TF_PARTIAL_BUDGET`. `final_label`: `GPU_LOAD_PASS` only with a complete TF reading; else `GPU_LOAD_INCOMPLETE_TF` (MTP-off-only variant likewise). Accuracy stays NOT_TESTED. |
| K-6 | HIGH | A2 F2 | **CONFIRMED.** K2 has a single calculated pin plus the boot's one 0.25 GiB retry; there is no K2 ladder. | K2 pins = dedupe [pin2, max(KV_DEFAULT, pin2 − 512 MiB), KV_DEFAULT]. Go to the next pin only on `LAUNCH_FAILED` + `oom_at_launch`. An Xid / illegal-memory-access / device fault stops at once. Each try checks the budget (else `SKIPPED_BUDGET`). Record launch OOM vs body OOM separately. |
| K-7 | MED | A1 | **CONFIRMED.** The `k100_facts` quantization check is `quantization not in (None, "None")`. Humming / method coverage per layer is not required. | GPU method map via the worker extension (rpc on every rank): per routed-expert layer the quant method class, bits and group size, plus the MTP experts. Gate `GPU_METHOD_MAP_PASS` = 48/48 equal to the checkpoint map (0-11 / 36-47 W3 g128, 12-35 W2 g64) + MTP W4, on K and K2. A mismatch means the boot is not clean. Do not ban Unquantized globally. |
| K-8 | MED | A1 | **CONFIRMED.** The embedded lab `steady()` keeps only a 200-char `sample_tail`; per-stream text and tokens are not saved. This violates the standing rule to save full traces. | After each window (outside the timed part) write every stream's full text, token ids and finish to `k100/traces/<boot>_<window>.jsonl`. The same applies to the admission streams. |
| K-9 | MED | A2 F3 | **PARTIAL.** `help_check` runs on GPU per boot (`HELP[name]`) but is only recorded. | The first boot's plan: if `--help` ran and any required flag is missing → do not launch (`FAIL_CLI_FLAGS`). If help itself errors, record it and continue (the NB1 R2 boots with the same argv family worked). |
| K-10 | MED | A1 §3 | **PARTIAL.** Only K's 90k TF has a budget drop. K's 120k long row can starve K2. | Reserve K2's minimum (24 min + boot) before K's optional items. K's 120k row and 90k TF run only if `left()` minus that reserve covers them; else `SKIPPED_K2_RESERVE`. |
| K-11 | DOC | A1 §2 | **CONFIRMED (arithmetic).** 463 blocks is the FP8-equivalent estimate. K / K2 run BF16 KV: RC2 BF16 is 2,464,415,744 B / 79 blocks = 29.75 MiB per block vs FP8 15.77 → the same pin ≈ 244 BF16 usable blocks. | The `k100_capacity` note and report list both (FP8-equivalent and BF16 estimate); the measured worker blocks are authoritative. PLAN updated. |
| K-12 | LOW | A3 §3 | Advice: an arithmetic probe for the author's "arithmetic consistency issue". | 8 fixed exact-answer arithmetic prompts (multi-digit add / sub / mul, carries), greedy, short, on K and K2; record exact-match, not gated. |
| K-13 | LOW | A1, A3 | **CONFIRMED.** 3 unit tests use hard-coded `D:/ARC AGI 2/...` fixture paths (37/40 for the auditors). | Resolve the fixtures relative to the repo or skip with a reason when they are absent. |

**Not adopted:**
- **A2 F4 (K0 peak as the upper bound):** the ladder already uses K0's measured free memory per card, minus safety and the MTP reserve. With K-6, a K2 OOM steps down.
- **The Haberstroh-first fallback:** all three audits agree it is not first. No change.

## Round 2: independent reviewer 1 (GPU crash / OOM / budget lens), verified 2026-11-01
These were verified in code (the rebuilt export, 3,495 lines). F3 was verified against NB1 R2 data. All 18 were sent to the fix agent for the same revision.

| # | Sev | Finding (verified) | Fix |
|---|---|---|---|
| R1-1 | MED | K2 windows (`levels()` / `window()`) check only the global `left()`, not the boot budget | `due()` / `boot_left()` per level |
| R1-2 | MED | K2 boot reserve `est2 − 8` is smaller than the measured MTP ready times (423-777 s) plus teardown; EST_MIN 17 / 30 / 40 is too low | cap = min(cap, `left()` − 480); K2 cap from K's measured ready; EST_MIN ≈ 27 / 33 / 43 |
| R1-3 | MED | Ladder uses nvidia-smi free, which reads 734-990 MiB above the worker's free (NB1 R2: 1629 / 939 vs 639 / 191) | Per card, min(nvidia-smi, worker) |
| R1-4 | MED | Inner 0.25 GiB retry inside each ladder rung (up to 6 launches); a rung can sit above a pin that already OOMed; no K2 reserve; capacity uses the planned pin | One OOM policy; skip rungs ≥ the lowest OOM pin; reserve K2; pass the live pin |
| R1-5 | MED | 3 trivial arithmetic probes are a hard gate: one miss ends the run (contradicts K-12) | Gate on engine faults only; record arithmetic correctness |
| R1-6 | LOW-MED | Engine leftovers are cleared only after the 300 s GPU-release wait | `clear_engine_procs()` first |
| R1-7 | LOW-MED | sha wait up to 40 min; install timeout 90 min | Budget-aware limits |
| R1-8 | LOW | OOM detection reads only the last 4,000 chars of the log | Search the whole launch log |
| R1-9 | LOW | Retries reuse the HTTP / torch / ZMQ ports | Fresh port per try |
| R1-10 | LOW | Any OOM death in the body is labelled TF_OOM | Record the running item; `BODY_OOM_<item>` |
| R1-11 | LOW | CELL 8 has no try/finally | Always write the verdict |
| R1-12 | LOW | A boot record is saved only at `step()` return | Attach it to the live record |
| R1-13 | LOW | The sha thread mutates R without `_SAVE_LOCK` | Lock |
| R1-14 | LOW | 2 s nvidia-smi sampler inside the timed windows | 5 s |
| R1-15 | LOW | DROP / WD skip paths are dead code | Use `due()` only |
| R1-16 | LOW | `admission_test` budget leaves out ~150 s | +150 s |
| R1-17 | LOW | klee100 boots lack `--language-model-only` (~440 MiB on PP0) | Add it (NB1 forward test EXACT) |
| R1-18 | PROCESS | Notebook at 395.9 kB vs the 400 kB limit | Move blobs into the sha-checked dataset if needed |

Checked by the reviewer with no finding:
- the RERUN path;
- model dir / env (no W4 paths, no SCALES with BF16);
- read-back regexes;
- release after a partial failure;
- daemon threads;
- trace sizes;
- K0 block use.

## Round 3: independent reviewer 2 (data / label lens), verified 2026-11-01
Its H2, M1 and M3 are the same as R1-1/R1-10, R1-5 and R1-3. New verified items, sent to the fix agent:

| # | Sev | Finding (verified) | Fix |
|---|---|---|---|
| R2-1 | HIGH | K's T 0.6 120k long-row FAIL verdict sets K's status to NO_GO, so K2 never runs (export ~3168) | Verdict goes in quality fields only; K2 depends on transport cleanliness |
| R2-2 | MED | Watchdog kill is reported as TF_SERVER_DIED; admission gate is smaller than its own wait limit | BOOT_CAP_KILL label; admission gate = limit + 60 s |
| R2-3 | MED | After K TF_OOM a clean K2 is dropped from the label (FAIL_BOOT_K_TF_OOM); with_servers overwrites specific statuses | `GPU_LOAD_PASS_K_<status>_AT_PIN…` + K2; keep both statuses |
| R2-4 | MED | K vs A mixes the weights change with the pin change; K0 (A's config) runs only 90k TF | K0 TF 2k/8k/40k as well; report weights-only, pin-only and combined per band |
| R2-5 | MED | Retries truncate the server log and overwrite plans / help / load monitor | Try suffix per launch name |
| R2-6 | MED | Unit tests do not exercise the real data paths (A/A and A vs C compare byte-identical files) | Mock vLLM-JSON TF test, real `ref_load` + AB reproduction, CELL 8 flow test |
| R2-7 | LOW | Gauge pool can match a false smaller divisor | Worker P first; ≥ 4 distinct values |
| R2-8 | LOW | Capacity unit bases are inconsistent | Report total- and usable-block bases separately |
| R2-9 | LOW | RC2_WINDOWS references partly come from NB1 R1 | Add the NB1 R2 reference set |
| R2-10 | LOW | Decode tokens lost on error; probes truncated / not traced | Save everything |
| R2-11 | LOW | RC2 oracle identity as a hard gate blocks the run for a diagnostic | Record only (overrides part of K-4) |
| R2-12 | LOW | K2 garbage-under-load gives a generic MTP_OFF_ONLY label | Explicit label |

Checked and correct per the reviewer:
- position alignment;
- tokenizer (all requests send ids; template sha equal);
- top-k;
- yardsticks;
- the K2-vs-C pairing;
- sampling;
- acceptance / tok/s accounting;
- the oracle is not gated;
- no cross-boot overwrite of TF files.

## Round 4: GPT audit of rev 2 (sha 51b56e59), 2026-11-01
GPT re-checked the 43 earlier items: most are fixed, with partial gaps in K-5, K-7, R1-2 and R1-4 on the new K3 path. Its 4 new items were all verified in code:

| # | Sev | Finding (verified) | Fix |
|---|---|---|---|
| V-1 | HIGH | `method_map_check`: rank errors and missing ranks are not in `why`, so 2 of 4 ranks still pass | 4 unique ranks, no RPC errors, per-rank PP-local layers, MTP only on PP1, layer 48 never counted as routed |
| V-2 | HIGH | `k3_label`: LOAD_PASS for any TF reading | LOAD_PASS only with a complete K3 TF; else `K3_FP8_INCOMPLETE_TF` |
| V-3 | HIGH | EST 45 + 27 + 33 + 43 = 148 of 150 min; K3 `can_try` lacks the K2 reserve; a body OOM does not step down | K3 short core (~30 min); optional items gated on K0 + K + K2 reserves; first rung conservative; one controlled body-OOM retry; schedule-replay test |
| V-4 | HIGH | `scales_facts` checks only aggregated logs | Per-rank scale coverage on all 4 ranks with per-stage subsets (+ RPC readback if cheap) |

### Round 4, GPT audits 2 and 3 of rev 2: items added (verified in code)
| # | Sev | Finding | Fix |
|---|---|---|---|
| V-5 | MED | `k100_scales` takes the first EXISTING candidate, not the first SHA-verified one | Pick the first sha-verified candidate |
| V-6 | MED | K3 vs K labelled "FP8 KV effect alone", but the MTP / seqs / captures differ | Rename it; add the clean K3 vs K2 pair (K2 also runs TF 2k / 8k); report the MTP on/off parity |
| V-7 | LOW | Scales reused from W4 | Record EXPERIMENTAL_REUSED_CALIBRATION |
| V-8 | LOW | Traces are written only at the window end | Write per stream as each finishes; disk check |
| V-9 | DESIGN | 150-min four-boot run is risky | Profiles ALL / K3_ONLY / BF16_ONLY; K3_ONLY and BF16_ONLY can run as 2 parallel commits |
| V-10 | HIGH | `k2_plan` `fits=False` ignored by the caller | Skip with 0 launches |
| V-11 | HIGH | After a K3 device fault, K0 starts without a health check | GPU health gate (CUDA probe + nvidia-smi), else ABORT_DEVICE_FAULT |
| V-12 | MED | Only a text label | Machine-readable predicates per boot |
| V-13 | LOW | Windows default test path; label precedence after a watchdog kill | Fix |
