# NB-S2 "capacity" rebuild design (E25), 2026-11-01

Status: rebuild in progress. The notebook in this zip is the OLD build (a0faf93c). It has not run on GPU.

## What NB-S2 measures
NB-S2 tests capacity levers on the DFlash2 base B = NBDF-K100's D3s:
- base B: DFlash2 K3, dflash_seam, drafter SW block 1024, BF16 drafter KV on FLASH_ATTN, no draft head, Q-defer on;
- run setup: 24 seqs, pin 7,933,526,016, klee100 3bpw-MTP on Merlin v3.0.0, 4xL4 TP2xPP2.

Every arm makes exactly ONE change against B. Budget priority:

| Arm | Change | Question |
|---|---|---|
| B | NBDF-K100 D3s verbatim, plus the A/A pair | base |
| IDX | `--attention-config {"indexer_kv_dtype":"fp8"}` | Does indexer KV fp8 cut bandwidth and add speed? The kit found 0 extra blocks on the D-SW pool, so this arm is about speed only. |
| S28 | `--max-num-seqs 28`, captures 4..112 | Does a 28-stream residency raise aggregate tok/s? |
| LP | `--max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive` | How long do running decodes stall when a long prefill is injected (#60476)? |
| S32 | as S28, at 32 seqs (captures 4..128) | 32-stream residency |
| SPC | `--enable-mamba-shared-prefix-checkpoint` | Can sibling requests with the same prefix reuse the Mamba state instead of re-prefilling? |

Dropped arms:
- K4: MTP-only.
- PP4: the seam's aux relay works only with PP2.
- DYN: moved to NB-S1.

## Why it must be rebuilt
The old build reproduces the OLD NBDF-K100 (closed 456c47e6 / gated c5c43770). That version lacks two fixes found on the 2026-11-01 NBDF-K100 GPU run:
- **B1 (GRD):** Merlin `vllm/v1/e2_guards.py` does not know the DFlash speculator's `query_cudagraph_manager`, so every DFlash arm died at capture (LAUNCH_FAILED). The new seam is 341d859b.
- **B2 (hard-only gates):** a LOW_VARIETY heuristic took ARC digit grids for garbage and stopped Q. Now only EMPTY / REPEATED_CHAR / BANG_GARBAGE, request errors, NaN, zero accept or a launch failure stop an arm. Heuristics are recorded only (nbdf_lib b0147751).

Every NB-S2 arm is a DFlash arm, so the old build would lose the whole run.

## Rebuild plan
1. Retarget the reproduced base to the current NBDF-K100:
   - closed build e2198285;
   - real kit ALL OK: `_scratch/kits_wave3/q27-e27-nbdf-k100-kit-real/nbdf_kit_real_report.json`;
   - gated sha = the NBDF builder's `--test-out` output with that report (asserted byte-identical, as before).
2. Re-anchor every NB-S2 exact-once text edit onto the new body and cells, keeping each edit's intent. Audit everything NB-S2 adds: nothing may stop or skip an arm except a real failure of that arm, the time budget or its switch. The verdict labels S2-1..S2-6 stay record-only, as the user decided; raw numbers are read by hand.
3. Per-arm switches, a plain line at the top of CELL 1 (outside the packed body and the payload dataset):
   `NBS2_ON = {'B': True, 'IDX': True, 'S28': True, 'LP': True, 'S32': True, 'SPC': True}`
   - An arm switched off is not booted and is recorded as SKIPPED_SWITCH_OFF.
   - With B off, the levers still run.
   - This is the same pattern as NB-S1 (which runs on GPU now, c110103d), so changing the line needs no rebuild and no new kit.
4. Unit tests for the new base plus switch tests. The units kit layout is simulated locally first, because the kit does not ship the .ipynb.
5. Kits:
   - **units.**
   - **real:** it must call the notebook's own pre-launch sequence verbatim. It needs the kit-only fix S2-4: the argparse probe hit an offline HF lookup, `LocalEntryNotFoundError`. Fix: local paths plus HF_HUB_OFFLINE, or compare argv field sets.
6. The payload dataset `foysalemonshanto/q27-e25-nbs2-payload` (the CELL 7 body and nbs2_lib, sha-pinned) needs a new version before the GPU commit.
7. Budget 175 min (deadman 180). Notebook < 400 kB.

## Questions for the reviewer (blockers / severe only)
1. Can any arm be skipped or stopped by anything except a real failure, the time budget or its switch?
2. S28 / S32 at pin 7,933,526,016: the D-SW pool has 540 blocks, F_adm 12 / F_real 9 at 131k. Is a 28 or 32 residency possible only at short bands? Are the long-band windows correctly stepped down rather than skipped?
3. SPC with DFlash: Merlin needs an eagle group (dflash = use_eagle) and partial hash hits. Can it engage at all on this layout? What read-back proves it?
4. LP: is the decode-stall measurement a valid signal? (S2-2: the poll gap is not a per-token latency; the verdict is record-only.)
5. Anything that would make DFlash2 fail at boot, capture or first decode on GPU that the CPU kit cannot see.
