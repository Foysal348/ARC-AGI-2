## Capacity port: klee100 + DFlash2 on Merlin (4×L4, TP2×PP2). Report text; the harness blocked writing REPORT.md

Everything is CPU-only: no GPU, no Kaggle, no git, and no edits to the Merlin source or any experiment file. All patches went onto a copy, `D:/ARC AGI 2/_scratch/capacity_port/merlin_copy/`.
- **Outputs:** in `D:/ARC AGI 2/_scratch/capacity_port/`, copied to `D:/ARC AGI 2/experiments/Experiment 30 - Capacity port + generated-tail checkpoint (planned)/src/capacity/`.
- **Line references:** the pristine source **M** = `D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/`.

### Files
| File | What it is |
|---|---|
| `capacity_patch.py` | P1 / P2 / P4 seam patch: apply / verify / revert, with pre and post shas |
| `sw_bt_mask_patch.py` | Port of #60767 (stale sliding-window block-table entries) |
| `ledger_hook_patch.py` + `capacity_ledger.py` | The per-group capacity ledger and its scheduler hook |
| `measure_capacity.py` | The measurement (run with `venv/Scripts/python measure_capacity.py <tree>`) |
| `summarize.py` | Turns the results into tables |
| `capacity_results.json`, `capacity_table.md`, `residents_real_holding.md` | Results and tables |
| `pressure_tight_*.json`, `ledger_D4s_baseline.json` | Pressure runs and the ledger record |
| `stubs/` | uvloop and xgrammar stubs: Windows has no wheels for them, and the planner never uses either |

### How it was measured
**Which functions ran:**
- Merlin's real `get_kv_cache_configs`, `generate_scheduler_kv_cache_config`, `get_max_concurrency_for_kv_cache_config` and the real `KVCacheManager` / `MambaManager` / `SlidingWindowManager`.
- They ran in a venv on Python 3.14 + torch-cpu, with the kit's FakeL4 platform. The compiled-op modules are empty stubs.

**Inputs:**
- Exactly the R9 input: the klee100 K3_ONLY v2 GPU worker record (`k100_results.json`), the K-scaling of the state rows, the real layer names from the klee100 config, and the 5 drafter SWA layers on PP1.
- The config is the kit's `real_vc`: PP2×TP2, V2 runner, async, 1024 batched tokens, align Mamba, fp8 KV, block 4096, pin 7,933,526,016.

**Calibration:**
- Q reproduces the GPU exactly: 480 blocks; strides 14,680,064 / 16,515,072; max concurrency 12.61; F 6.
- D-SW1024 reproduces R9: 540 blocks; PP1 stride 14,155,776; drafter page 10,485,760.

**Real-holding simulation:** the real `KVCacheManager` admits fresh requests, prefills them in 1024-token chunks, then decodes 200 steps of (1+K) tokens with every step settled. It then counts the blocks each group actually holds.

### Main finding
- **F = 12 for D-SW is a startup and admission bound, not what a request holds.**
- On Merlin's real manager a decoding request holds **9** fixed blocks on D (6 Mamba + 3 SW) and **6** on Q. The live ledger confirms it: Mamba holds 1 running + 1 retained checkpoint per group, and SW holds 3 per request.
- With real holding, **D-SW baseline (539 usable) already matches or beats Q (479)**: residents at 64k are 21 vs 21, at 116k 14 vs 13. Research A's "D holds fewer long residents" came from the F 12 formula.
- Any harness rule of the form `pool / (ceil(L/4096) + F)` should use F_real = 9 for D, and block size 4096.

### Q1. Where F comes from
**Mamba, 2 per group × 3 groups = 6: real.**
- `MambaSpec.max_memory_usage_bytes` reserves 2 + spec blocks + checkpoint blocks in align mode (M/v1/kv_cache_interface.py:1076-1083). On this config `num_speculative_blocks` = 0 under ReplaySSM/RecoverSSM (M/model_executor/layers/mamba/abstract.py:77-82).
- The real holding is 1 running state block, plus the newest hashed state kept as a prefix checkpoint (`_RETAIN_CHECKPOINT = True`, M/v1/core/single_type_kv_cache_manager.py:51-53; `_retire_state_blocks` :1673-1702; `_hold_checkpoint` :1704-1725).
- The previous block is retired on the processed-token basis (`remove_skipped_blocks` :1947-1978).
- A transient 3rd block per group appears at each 4096-token crossing: `physical_block_cap` :2105-2110 and `allocate_new_blocks` :2129-2255.
- Measured 2 per group on the real manager.

**Sliding window, 6 at admission vs 3 real.**
- The cap is `cdiv(sliding_window - 1 + extra_retained_tokens + max_in_flight_tokens, block) + 1` (M/v1/kv_cache_interface.py:860-886).
- `max_in_flight_tokens` = (PP+1) × 1024 = 3072 under V2 + async (M/config/vllm.py:607-628).
- So the cap is cdiv(2047 + 0 + 3072, 1024) + 1 = 6.
- It is applied only through `apply_admission_cap`:
  - the full-sequence admission gate (`allocate_slots(full_sequence_must_fit=scheduler_reserve_full_isl)`: M/v1/core/kv_cache_manager.py:545-567, M/v1/core/sched/scheduler.py:1431, default True at M/config/scheduler.py:190);
  - the in-flight prefill reservation (scheduler.py:3370-3405);
  - the coordinator sum (M/v1/core/kv_cache_coordinator.py:203-275);
  - the single-type cap (M/v1/core/single_type_kv_cache_manager.py:222-235).
- Real holding during settled decode is 3 blocks (measured). It peaks at 6 only while a long prefill has 3 chunks in flight; freeing happens on the processed basis (kv_cache_manager.py:590-597).

**What is pure scheduling-time bound:**
- About 3 of the 6 SW blocks: the 3072 in-flight term, plus the +1 slack.
- The F 12 used by boot max-concurrency (`get_max_concurrency_for_kv_cache_config`, kv_cache_utils.py:1196-1239).
- Neither reserves memory for running decodes.

**Minimal safe reduction:** none is needed in the engine; the fix is to stop sizing with F 12. A real cut means fewer SW blocks: an FP8 drafter KV with SW block 2048 gives admission cap 4 and real holding 2 (F_real 8). The other real cut is P4 (retention off).

### Q2. Mamba padding
**Source:** `_align_hybrid_block_size` pads every Mamba page to the attention page, 4096 × 512 B = 2,097,152 (M/platforms/interface.py:1106-1126; `spec.page_size_padded` set at abstract.py:64-83). The real GDN state is 1,732,992 B at K3 (1,767,904 at K4).
- That is 17.4 % of the allocated page wasted, or the page 21 % above the payload.
- The PLE short-conv page is only 245,760 B but is also padded to 2 MiB, then folded into a Mamba group (kv_cache_utils.py:2639-2722).

**Why PP0 binds D:** the packed planner (kv_cache_utils.py:1894-2028; stride = widest group, :1532-1563) puts 7 padded GDN layers into one PP0 group: 14,680,064 B, against the 14,155,776 B attention group. So D's PP0 stride is set by Mamba padding. `get_kv_cache_configs` then takes the minimum block count across ranks (:2966-2982).

**Fix P1 (= the #60967 idea):** with `MERLIN_MAMBA_PAD_TO_STATE=1`, pad only to the largest Mamba state page. All Mamba layouts keep one common page, so bucketing and folding still work.
- Mamba views are carved by byte offset within the block (abstract.py:28-41; worker/utils.py:599-661), so no kernel stride assumption changes.
- Measured: **D 540 → 560 blocks (+20, +3.7 %)**; both ranks' strides become 14,155,776. **Q stays at 480**, because PP1's 7-layer attention group with the MTP layer (16,515,072) binds.

**Mamba group count (P2, `MERLIN_MAMBA_GROUPS=n`) is a measured negative:**
- 2 groups: Q 457 blocks, F 4, +1 resident at 64k only; D 448 blocks, worse at every length.
- 4 groups: D 560 blocks but F 11 real, worse.
- Keep 3 groups.

### Q3. Draft / target KV sharing; #58207, #58380, #60967, #58345
**#58380 (BLHNC packing): already present.** Merlin resolves BLNHC and builds packed block-outermost groups (`_get_packed_kv_cache_groups`). D-SW has 6 groups: 3 Mamba, the QSA ring (1024 slots per block), attention, and the drafter SW group.

**#58207 (group-size selector): does not apply.** The packed path returns before the uniform-page planner (kv_cache_utils.py:2239-2245). Forcing a layer-outermost layout (LBHNC) gives 76-78 groups and 3,783 × 2 MiB blocks, F_adm 87 / 116, max concurrency 6.8. By real simulated holding, residents are about 15 / 11 / 9 at 40k / 64k / 90k against 21+ at 64k packed. That is strongly worse.

**#58345 trap (packed SW blocks silently shrinking): absent.**
- The drafter group reads back block 1024 on every rank.
- The block-outermost path never splits kernel blocks (worker/utils.py:697-707).

**Bucket sharing:**
- The drafter group has 0 layers on PP0, but block ids are global, so each real SW block wastes a full PP0 block (3 per request ≈ 42 MB). On PP1 each wastes 3.7 MB (10.49 of 14.16 MB used).
- Sharing pages with a target group is not possible: there are different block counts per group, Mamba blocks are shared prefix checkpoints, and the target attention group has no spare room on PP1.
- A split drafter pool (#59117) still needs about 755 MB on PP1 for 24 streams. With equal per-rank pins that leaves about 506 blocks, which is net-zero.
- The drafter's SW block size and dtype are the lever that works:
  - SW 2048 FP8 keeps the same page size and gives F_adm 10 / F_real 8.
  - SW 2048 BF16 widens the block to 20.97 MB and gives 378 blocks: bad.
  - SW 512 gives F 17: bad.

### Q4. Measured numbers
Pin 7,933,526,016. Residents are min(seqs, (usable − ring) // (ceil(L/4096) + F_real)), using F_real from the real-manager simulation. Each cell is 40k / 64k / 90k / 116k.

| Arm | Pool (usable) | F_adm / F_real | seqs 24 | seqs 28 | seqs 32 |
|---|---|---|---|---|---|
| Q baseline | 480 (479) | 6 / 6 | 24/21/17/13 | 28/21/17/13 | 29/21/17/13 |
| Q + P1 | 480 | 6 / 6 | same as Q | same | same |
| Q + idxfp8 (known lever) | 508 | 6 / 6 | 24/23/18/14 | 28/23/18/14 | 31/23/18/14 |
| D3s / D4s baseline SW1024 BF16 | 540 (539) | 12 / 9 | 24/21/17/14 | 28/21/17/14 | 28/21/17/14 |
| D + P1 | 560 (559) | 12 / 9 | 24/22/18/14 | 28/22/18/14 | 29/22/18/14 |
| D SW2048 FP8 | 540 | 10 / 8 | 24/22/17/14 | 28/22/17/14 | 29/22/17/14 |
| **D SW2048 FP8 + P1** | **560** | **10 / 8** | **24/23/18/15** | **28/23/18/15** | **31/23/18/15** |
| D SW1024 FP8 | 540 | 12 / 9 | same as baseline | same | same |
| D SW2048 BF16 | 378 | 10 / 8 | 20/15/12/10 | 20/15/12/10 | 20/15/12/10 |
| D + P1 + idxfp8 | 593 | 12 / 9 | 24/23/19/15 | 28/23/19/15 | 31/23/19/15 |
| D + P1 + SW2048 FP8 + idxfp8 | 593 | 10 / 8 | 24/24/19/15 | 28/24/19/15 | 32/24/19/15 |
| P4 retention off (Q / D, formula) | — | Q 3, D 7 | Q 24/24/19/14; D 24/23/18/14 | | |

`capacity_table.md` also has residents on the F_admission and F_steady formulas for every arm.

### Q5. Risks and GPU gates
**P1 (layout only):**
- Risk: misaligned state pointers. Offsets are multiples of 1,732,992 (128 B aligned), and every view is a byte offset.
- Gates:
  - the boot log line "CAPACITY-PORT P1 ... 1732992" (K3) or 1767904 (K4);
  - num_gpu_blocks 560 on all 4 ranks;
  - FULL graph capture;
  - teacher-forced D-vs-Q logprobs at 2k / 8k / 40k / 90k / 116k inside Q's A/A envelope;
  - prefix-hit cold vs warm equivalence;
  - no NaN in GDN states.

**SW2048 FP8:** needs TRITON_ATTN for the drafter (#54690 risk). Gates: acceptance not below the BF16 arm's, and the same oracle.

**P4:**
- Risk: a running request's prompt checkpoint becomes evictable. That means re-prefilling whole Duck prompts, which can cost more than the +3 residents it frees.
- Run it only as an A/B arm, with prefix-hit and recompute counters.

**Admission deadlock:** none of these patches change the admission formula.

**Page reuse before PP work completes:** freeing stays on the processed-token basis; nothing touched.

### GPT research items
1. **#57050: present.** Merlin has the `physical_block_cap` clamp (single_type_kv_cache_manager.py:2105-2110). Upstream's extra spec-block term on a checkpoint step is a no-op here, because `num_speculative_blocks` = 0. No port.
2. **F ledger:** answered in Q1 above. The ledger record (`ledger_D4s_baseline.json`) shows per-group live, retained, reclaimable and max-admission blocks: Mamba 1 live + 1 retained per request per group; SW 3 live, 6 admission, plus 8 hashed SW blocks reclaimable.
3. **SW matrix (D3):**

   | Arm | Pool | F_adm / F_real |
   |---|---|---|
   | 1024 BF16 | 540 | 12 / 9 |
   | 1024 FP8 | 540 | 12 / 9 (no gain) |
   | 2048 BF16 | 378 | 10 / 8 (bad) |
   | 2048 FP8 | 540 | 10 / 8 |
   | 2048 FP8 + P1 | 560 | 10 / 8 |

4. **Planner comparisons, each run separately:** #58207 = n/a and worse if forced; #58380 = present; #60967 = P1 (+20 D, 0 Q).
5. **#62752 port** (real manager, 24 DFlash streams at 26k-66k, all drafts accepted, 6 tool-return bursts of 8k tokens):
   - D baseline at 540: 2 preemptions, 141k recompute tokens.
   - SW2048 FP8 + P1 at 560: 0 preemptions, minimum free 23 blocks.
   - My simple headroom policy changed nothing: the preemptions come from decode growth, not from the bursts.
6. **#60767: absent in Merlin.** `gather_block_tables` and its kernel copy rows verbatim (M/v1/worker/gpu/block_table.py:149-179, :243-312). Rows are append-only (:114-133).
   - The port is `sw_bt_mask_patch.py`, gated by `MERLIN_SW_BT_MASK=1`. It adds a new kernel; the original kernel is untouched, and with the env unset the gather is identical.
   - Patched sites: model_runner.py:824-835 (set the reach per group) and :1847-1849 (pass `num_computed_tokens`). The DFlash speculator reuses the same gathered tables.
   - CPU test on the real SlidingWindowManager, 624 steps: **PASS**. Up to 12 stale ids per row; all of them masked; 0 in-window ids masked.
   - The Triton kernel itself has not run, because Triton is not on this machine. Run it with `TRITON_INTERPRET=1` in a Kaggle CPU kit, then GPU gates: NaN/Inf checks and acceptance at 1 / 8 / 16 / 24 streams.
7. **Ledger:** `capacity_ledger.py` provides `static_ledger` (per rank: blocks, stride, groups, page / unpadded page / max-admission per group) and `live_ledger` (live / retained / reclaimable per group, free blocks, running, preemptions, recompute tokens).
   - `ledger_hook_patch.py` logs a line `CAPACITY-LEDGER {json}` every N steps when `MERLIN_CAP_LEDGER_EVERY=N` is set (scheduler.py:705-706).
   - Notebooks read it back with `parse_ledger_lines`.
8. **INT8 checkpoint store (design only):**
   - Retired and retained hashed Mamba states would go through an index on the `MambaManager` side, hash → store slot.
   - The worker would get a separate INT8 tensor per GDN layer, outside the KV pool: SSM [24,128,128] int8 plus a per-(head,row) scale, about 0.47 MB vs 1.73 MB.
   - Quantize and dequantize copies would ride the existing `kv_cache_block_copies` path (M/v1/core/kv_cache_manager.py:1321-1327 → M/v1/core/sched/output.py:305 → M/v1/worker/gpu/model_runner.py:1511-1515).
   - The cache-hit lookup would consult the store; the active state stays FP32. It must obey `publish_decode_checkpoints = False` under RecoverSSM + PP (scheduler.py:102).

### Ranking (blocks gained × safety)
1. **SW mask (#60767):** correctness, not capacity. It is cheap and protects acceptance at 24+ streams. Run it first.
2. **P1 Mamba pad-to-state:** +20 D blocks, about +1 resident at 64k / 90k. Layout only, medium-high safety.
3. **SW2048 FP8 drafter KV:** F_real 9 → 8, plus 1-2 fewer admission blocks. 0 preemptions in the tight pressure test when combined with P1. Medium safety (TRITON FP8 path).
4. **idxfp8 (known lever):** +28 Q / +33 D blocks. Quality risk, tracked separately.
5. **P4 retention off:** about +3 residents at 64k, at a prefix-hit cost. A/B arm only.
6. **Do not use:** P2 group changes, SW512, SW2048 BF16, or the layer-outer planner.

### Caveats
- The P1 patch site itself (`interface.py`) is not exercised on CPU. The measurement sets the same padded page directly. The kit or boot log must confirm 560 blocks.
- The Triton kernel is unrun.
- The simulation settles every step, so in-flight PP transients (+1 SW block, +1 Mamba block per group at crossings) are not in F_real.
