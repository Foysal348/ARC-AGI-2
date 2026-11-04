# Fix list: NB-S1, NB-S2, GEN-TAIL (audits #1 and #2 of 3; 2026-11-01)

**Rules:**
- Collect every item. Dispatch ONCE, when the user says go, with one round per agent and one kit run.
- Every fix is readback / verdict logic or a design correction. None may stop or skip an arm (real-failure-only gates); labels are recorded only.

| # | Where | Item | Audit | Fix | Tests |
|---|---|---|---|---|---|
| **S1-1** (HIGH) | `nbs1_lib.py:185-216` (DYN readback); `nbdf_lib.py:860-868` checks only max K=4 | DYN passes with no evidence that the per-batch K schedule was applied: it only scans for the 2 fallback strings | #2 | Read back the effective `num_speculative_tokens_per_batch_size` == [[1,8,4],[9,24,3]], plus runtime tier counts (K4 batches, K3 batches) via new instrumentation, e.g. per-step spec K from the SpecDecoding / scheduler stats. Schedule absent → `DYN_SCHEDULE_NOT_APPLIED` (real failure, DYN only). A tier not exercised → INCONCLUSIVE (recorded, no WIN). Both tiers seen → DYN_ENGAGED | static K4 → FAIL; config ok but one tier → INCONCLUSIVE; both tiers → PASS |
| S1-2 | NB-S1 build | Closed KV gate | #1 | No code change. Real kit v2 is running → `build_e25_nbs1.py --kv-gate-report` → `nbdf_gate_verify` (embedded == facts, notebook sha) | — |
| **S2-1** (HIGH) | `nbs2_lib.idx_readback` 299-315; caller body 1630-1644 | IDX is "engaged" even when expected ranks are missing (1-rank-only → engaged) | #1, #2 | Expected indexer ranks come from the live layer ownership (installed Qwen4Exp partition), never hardcoded. Any applicable rank missing → `INCONCLUSIVE_IDX_READBACK` (no WIN). bf16 on an applicable rank → `IDX_NOT_ENGAGED` (real failure, IDX only). All fp8 → IDX_ENGAGED | one rank only → INCONCLUSIVE; one bf16 → FAIL; all fp8 → PASS |
| **S2-2** (HIGH) | `nbs2_lib.py:435-455`; caller body 1579-1628 | LP p95 is a 20 ms poll update gap, not a per-token latency (blind to burst size) | #1 | Rename to `p95_observed_update_gap_ms`; use SSE token-event timestamps if available. The LP verdict also needs decode tok/s during the injection, prefill TTFT and aggregate output tokens on a fixed concurrent decode load; the poll gap alone never decides WIN | same interval with different bursts; same throughput with different intervals; an injected prefill slowdown |
| **S2-3** (HIGH) | `nbs2_lib.py:590-600` best-arm | The ranking mixes objectives (max band gain across arms) | #1 | Per-objective winners: `best_capacity_arm` (S28 / S32, 20k@seqs vs B), `best_decode_arm` (IDX / SPC, standard bands), `best_stall_arm` (LP). No global best from max band gain | IDX best at 8k, S28 best at 20k, LP best at stall → 3 separate winners |
| **G-1** (design bug) | `DESIGN_generated_tail_checkpoint.md:130-133` | Attention partial key at A = floor(valid/64)·64 with `block_idx = A // 4096` points at the NEXT block when A % 4096 == 0. Merlin's `_cache_partial_tail_block` skips the partial key at a full boundary (`single_type_kv_cache_manager.py:911-912`) | #2 | If A % block_size == 0: no partial key; verify the preceding full block is hashed. Else: idx = A // block_size, publish the partial | A at 4032 / 4096 / 4160 / 8192 |
| G-2 | design + prototype | The PP fence alone is not enough; bind the publish to the slot generation | #1, #2 | `PendingTail` (request_id, req_idx_gen, position, block_id, producer_step from the commit-plan metadata, valid_tokens). Publish only if all required PP ranks committed, the generation is unchanged and pos ≤ the safe limit; else discard. Invalidate on abort / stop truncation / index reuse | idle, busy 2-microbatch, abort at boundary, index reuse, fence-off negative control |
| G-3 | design | MAX_GRACE_PINS 16 → 21 may pin all hashed blocks of a request; the pool peaked at 99.4 % | #2 | Measure the pinned blocks per request (CPU sim + GPU telemetry); cap the pinned bytes; never assume zero extra pages | pool-pressure fixture |
| G-4 | prototype tests | The 677-turn replay tests returned early (CSV missing in the zip) | #1 | Point at the real CSV under `Experiment 25/Experiment Results/prep_0gpu/duck_timeline/`; fail (do not skip) if it is absent; ship the CSV in the review zip | — |
| X-1 | process | A matched MTP K3 control for production promotion | #1 | No code change: NBDF-K100 (running) has Q = E28 MTP vs D3s / D4s in the same windows. E29's drafter choice comes from it | — |

## Validation against the source (coordinator, 2026-11-01 ~14:15 UTC)
| # | Verified | Evidence |
|---|---|---|
| S1-1 | TRUE | `nbs1_lib.readback` 185-216 checks only `DYN_FAIL_RX` strings. `nbdf_lib.py:860-868` checks spec_method / spec_k and never `num_speculative_tokens_per_batch_size`. **Refinement:** the schedule can be read from the same parsed engine config (`f["speculative_config"]`). Tier evidence can come from the existing SpecDecoding counters per window: drafted tokens / draft steps = effective verify K. Windows at ≤ 8 running (ladder 2 / 8) should show K 4; at 16-24, K 3. No new engine instrumentation is needed |
| S2-1 | TRUE | `idx_readback` 299-315: `engaged = len(fp8) == len(seen)`, never compared with the expected ranks. Caller body 1637-1639 |
| S2-2 | TRUE | `stall_stats` 435-455 polls the token-count increase gap; `p95_ratio` feeds `primary_verdict` (509) for LP |
| S2-3 | TRUE | 590-600: `elig` sorted by max PASS band gain across all levers → a single `best_arm` |
| S2-4 | TRUE | real-kit report: `argparse.B.error = LocalEntryNotFoundError`; the only failing check |
| S2-6 | TRUE | 456-469: `SPC_JUNCTION_HIT if hc >= base + 128` with base = (p // block)·block → 6272 counts as a hit for p 6900 / block 1024 |
| G-1 | TRUE | design line 130: "register `cache_partial_block` at A = floor(valid/64)·64 on block A // 4096". Merlin skips the partial at a full boundary (`single_type_kv_cache_manager.py:911-912`) |
| G-4 | TRUE | `test_gentail_logic.py:54-57` and `225-228` `return` when the CSV is missing (`os.path.exists(DATA)` at 38) |

## Items the first list missed (from the audits)
| # | Item | Audit |
|---|---|---|
| G-2+ | Add the invariant `state.token_hash == request.prefix_hash_at(state.position)` to the publish condition (with the generation, the producer step from the commit-plan metadata, and all ranks committed) | #3 |
| G-5 | Telemetry counters: checkpoints published / deferred / skipped / adopted / evicted (per run, for the GPU test) | #3 |
| S2-1+ | No WIN for IDX unless engagement is verified on every expected rank. `lever_verdict` today does not consult `idx_readback` | #1, #3 |
| (covered) | NB-S1 "rebuilt arms must match the kit-verified config": `nbdf_gate_verify` checks embedded == facts plus the notebook sha. S28 / S32 graph-workspace pressure: free MiB per rank after each window is already recorded (NBDF read-backs). LP cold-prefill injection (#60476): already in the LP probe | #1, #3 |

## Audit #3 additions
| # | Where | Item | Fix | Tests |
|---|---|---|---|---|
| **S2-6** (record-only) | `nbs2_lib.py:458-469` (sibling label); caller body 1560-1575 | `SPC_JUNCTION_HIT` false positive: with p_len 6900 and block 1024, base is 6144, so a hit of 6272 is already called a junction hit | Label `SPC_JUNCTION_CANDIDATE` when hc ≥ (p // 64)·64 − eagle margin, using the expected junction from Merlin's Mamba / EAGLE rules. `SPC_JUNCTION_HIT_VERIFIED` only with an actual checkpoint read-back. Compare B vs SPC with identical sibling prompts after a cache reset | P 6900, block 1024, hit 6272 → never verified |

Audit #3 also confirms:
- NB-S1 is GO after the gated rebuild (done: 585cf18c).
- NB-S2's B / IDX / S28 GPU diagnostic is GO as built, because NB-S2 embeds NBDF-K100's passed gate.
- Generated-tail Option B is GO for prototype work only.
- No other confirmed bug.

## Kit results (wave 2, 14:00 UTC)
- **NB-S1 real kit:** ALL OK (notebook 0c432f93). Gated rebuild done: **585cf18c**, gpu_ready, `nbdf_gate_verify` ok. This build does NOT have S1-1 (DYN readback). Its G1 / G2 / EH / F8 arms are GO per audit #2; its DYN result counts only as a record.
- **NB-S2 units:** ALL OK.
- **NB-S2 real:** NOT OK, `nbs2_argparse_one_change_per_arm` only. Kit-only: building EngineArgs in the kit tried an HF hub lookup offline (`LocalEntryNotFoundError`). Every other check passed, including `nbs2_payload_dataset_mounted`.

| # | Where | Item | Fix |
|---|---|---|---|
| **S2-4** | NB-S2 real kit, argparse probe | `LocalEntryNotFoundError` (HF lookup offline) when the kit builds EngineArgs | Pass the local model / tokenizer paths and `HF_HUB_OFFLINE=1` / `local_files_only` (as the notebook launch does), or compare the argv field sets without constructing a model config |
| **S2-5** (finding) | kit R9 | **IDX on the DFlash2 base gives 0 extra blocks:** `IDX_B_idxfp8` = 540 = B. The D-SW pool is bound by another page, so the indexer fp8 page saving only shows on the MTP layout (480 → 508) | Keep IDX in NB-S2 for its bandwidth / speed effect only. E29's `indexer_fp8` capacity gain applies to the MTP drafter only |

**Already sent before this list:** S2-1/2/3 went to the NB-S2 agent, which is now told to STOP and hold; G-2 and G-4 went to the gentail agent.
**Pending:** audit #3.
**Dispatch plan:** NB-S1 agent gets S1-1; NB-S2 agent gets S2-1..3; the gentail agent gets G-1..4. One message each, then one kit round for NB-S1 and NB-S2.

## DECISION 2026-11-01 ~14:20 UTC (user)
Non-blocking bugs are ignored to save time. S1-1, S2-1..S2-6 are DEFERRED: all are verdict / label / record-only / kit-only items. NB-S1 (gated 585cf18c) and NB-S2 (a0faf93c) run as built. Their verdict lines are not trusted; the raw numbers (tok/s per window, AL, pool, residents, IDX dtype per rank, SSE stall data) are read by hand. Gentail G-1..G-5 continue (research patch, not on the critical path).
