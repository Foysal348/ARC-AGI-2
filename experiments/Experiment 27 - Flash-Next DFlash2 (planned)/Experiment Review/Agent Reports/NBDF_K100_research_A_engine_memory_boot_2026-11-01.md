# Can DFlash2 run on the klee100 3bpw target? Feasibility report

**Verdict: GO to build NBDF-K100, NO-GO for GPU until three fixes land and the K100 CPU kit passes.** Nothing in Merlin's source blocks DFlash2 on klee100: the seam, the PP2 relay, the embedding, the KV layout and the drafter loader behave exactly as they do on W4. What can kill or waste the 150-minute run is all on the harness side:
- **Blocker 1:** klee100 does not give the same output twice for the same prompt. NBDF's token-ID gates would stop the D arms for that alone.
- **Blocker 2:** the W4 constants (8 streams, W4 capture sizes, W4 ladder) put klee100 at the wrong operating point.
- **Blocker 3:** the 1-GiB out-of-memory step on the pin ladder leaves D with less KV than Q.

Read-only throughout: no file edits, no Kaggle, no git. An NBDF-K100 build is already in progress in `Experiment 27 .../Build/` (`build_e27_nbdf_k100.py`, `nbdf_k100_cell7_body.py`, `nbdf_k100_unit_tests.py`). Its `K100_CFG` is marked PROVISIONAL until this research arrives (`build_e27_nbdf_k100.py:22-24, 44-52`). Every fix below targets those files.

Path shorthand: Merlin = `D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/`; Build = `D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build/`; E25B = `D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build/`.

The K100Q server log is `D:/ARC AGI 2/_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/engine/logs/server_merlin_fp8_mtp_pin508.log` ("K100Q log" below).

---

## Blockers, with fixes

### B1. klee100 is not deterministic, so NBDF's token-ID gates would kill healthy D arms
**Evidence of the noise:**
- The K3_ONLY v2 teacher-forced repeat on klee100 used identical prompts with no cache hit (`cached_tokens: 0`), yet still differed:
  - 2k: mean |dlogprob| 0.659, p99 4.59, max 5.66, rank-1 agreement 0.856;
  - 8k: 0.098 / rank-1 0.949;
  - 40k: 0.283 / rank-1 0.914.
  - Source: `_scratch/e25_gpu_out_k3_v2/.../k100/k100_results.json`, `/sections/K3/k100_K3_fp8_mtp3_pin739_r0/tf/{2k,8k,40k}/aa`.
- W4's A/A on NB0 is exactly 0.0 (same file, `/analysis/tf_k_vs_A/*/noise_floor/nb0_A_aa_*`).
- Humming takes a lock buffer for its reductions (`Merlin/utils/humming.py:100-109`, used at `fused_humming_moe.py:141`). That is a likely, unproven, source of order-dependent results.
- Even W4's Q ladder was already NEAR_TIE at 4 and 7 concurrent requests (`_scratch/e27_gpu_out/.../nbdf_results.json`, Q `smoke.ladder_brief`).

**What the gates do with that noise:**
- The margin threshold is FAIL_MARGIN 0.5 nats (`E25B/nb4_lib.py:59`). klee100's A/A p99 of 0.68-4.6 nats will cross it.
- A crossing becomes FAIL_HIGH_MARGIN:
  - the ladder turns it into CONCURRENCY_CONTAMINATION (`nbdf_lib.py:1466-1483`) → FAIL_SMOKE (`nbdf_lib.py:1345-1347`) → the D arm stops as NO_GO_SMOKE (`nbdf_k100_cell7_body.py:835-837`);
  - the greedy oracle gives NO_GO_QUALITY, so the windows are skipped (`nbdf_k100_cell7_body.py:862-866`);
  - the long oracle gives CORRECTNESS_FAIL_LONG (`nbdf_lib.py:1649-1651, 1871-1873`);
  - the verdict side returns early with CONCURRENCY_CONTAMINATION or CORRECTNESS_FAIL (`nbdf_lib.py:1855-1870`).

**Fix (gates on real failures only, measured against klee100's own noise floor):**
- Q runs its greedy oracle twice, with a cache reset in between, plus its own ladder. That gives `q_aa`.
- D's token-ID failures count only when Q's A/A is clean. Hard stops stay as they are: garbage, request errors, short output, zero acceptance.
- Add a correctness gate that still works on a noisy target: teacher-forced prompt logprobs at 2k / 8k / 40k, D vs Q, compared with Q's A/A envelope. DFlash does not change the target's forward pass apart from the taps.

```python
# nbdf_lib (K100 variant)
def k100_rel(d_status, q_aa_status):
    if q_aa_status in CLEAN:            # deterministic base: the W4 rule is unchanged
        return d_status
    return "NONDET_BASE_RECORDED" if d_status in ("FAIL_HIGH_MARGIN", "CONCURRENCY_CONTAMINATION") else d_status
# body :835  -> if is_d and sg.startswith("FAIL") and not (sg.endswith("CONTAMINATION") and Q_AA_NONDET): stop
# body :862  -> o["oracle_vs"] = k100_rel(status, Q["oracle_aa"]["status"])
# new hard gate: tf_d_vs_q mean|dlp| <= q_aa_mean + 0.05 and rank1 >= q_aa_rank1 - 0.03 per band
```

### B2. The W4 constants put klee100 at the wrong operating point
**What the build inherits:**
- `SEQS = 8`, `Q_CAPTURES = [1..32]` and `CONC_LADDER = (2, 4, 7)` (`nbdf_lib.py:58, 65, 114`); no K100 edit overrides them.
- `capture_sizes(SEQS, k)` therefore stops at 40 (D4) and 32 (D3) (`nbdf_lib.py:165-168`).

**What klee100 actually runs:**
- K100Q ran `max_num_seqs` 21 with captures [1,2,3,4,6,8,12,16,24,32,40,48,64,80,84] (K100Q log line 8).
- The user set E28 klee100 to 24 workers (E28 `README.md:114`).
- The baselines are 460 tok/s at 20k@16 and 562 at 20k@24 (`E26_K3ONLY_GPU_v2_2026-11-01.md` lines 33-34). Measuring at 8 streams answers a different question.

**Capture coverage:** K100Q's list, rounded up to multiples of 5 for K4, keeps graphs only up to 80 tokens (the rounded 85 is dropped because it exceeds the largest capture). That covers at most 16 requests.
- Steps with 17-21 requests would run the drafter eagerly, since DFlash has no PIECEWISE fallback (`Merlin/v1/worker/gpu/spec_decode/dflash/speculator.py:139-143`).
- For K3, the same list covers 21 requests.

**Fix:**
```python
K100_CFG["seqs"] = 24
N.SEQS = 24; N.CONC_LADDER = (2, 8, 23)
```
- **Q:** Merlin's registry captures for 24 sequences (the K100Q list, plus 96).
- **D4s:** `capture_sizes(24, 4)` = 5..120 (24 sizes).
- **D3s:** `capture_sizes(24, 3)` = 4..96.
- **Trap:** `level_for_band(..., seqs=SEQS)` binds `SEQS` when the function is defined (`nbdf_lib.py:558`), so pass `seqs` explicitly. Also check `expect_cfg.max_num_seqs` (`nbdf_lib.py:460, 789`).

### B3. The out-of-memory step-down wastes the run
- On a D launch OOM the provisional build steps 7,933,526,016 → 6,859,784,192 (`build_e27_nbdf_k100.py:47, 218-221, 243-247`).
- That 1-GiB step leaves D with 467 blocks (= floor(6,859,784,192 / 14,680,064)), below Q's 479, and costs a failed boot of about 8 min.
- **Fix:** try `DFLASH_DRAFT_EMBED_HOST=1` at the same pin first. It frees 606 MiB, and the seam supports it (UVA, guards tested on CPU only). Then step down by 256 MiB:

```python
"pin_ladder": [7933526016, 7665090560, 7396655104, 6859784192],   # D pools 540 / 522 / 503 / 467
"d_retry_env": [{"DFLASH_DRAFT_EMBED_HOST": "1"}],                  # first retry, same pin
```

---

## Answers to the seven checks

### 1. Boot gates
- `method_map_check` already accepts having no MTP rows when `need_mtp` is False: `mtp_ok = not mtp_bad and not mtp_pp0 and (bool(mtp) or not need_mtp)` (`E25B/k100load_lib.py:619`).
- The DFlash2 drafter is dense, so `_k100_scan` gives no rows for it. It keeps only modules whose `quant_method` is a FusedMoE method (`build_e25_klee100_load.py:671-690`), and the drafter's linears are `UnquantizedLinearMethod`.
- The minimal correct handling is already in the build: `need_mtp = arm.get("method") != "dflash"` (`build_e27_nbdf_k100.py:203`). The arms do carry `"method"` (`nbdf_lib.py:457`).
- If a D arm accidentally built MTP, the spec read-back would catch it (`nbdf_lib.py:851-854`). So the method-map gate is **not a blocker**.
- **Static scales:** `scales_check` requires the 12 target QSA layers per stage. The MTP layer's DEFAULT 1.0 line (K100Q log 454-455) is only recorded when present (`k100load_lib.py:688-693`). On D that line is simply absent: PASS.
- The `k100_kv_scales` RPC will list the drafter's `self_attn.attn` modules at 1.0 (`build_e25_klee100_load.py:757-780`). That is recorded only. The drafter KV is BF16 and needs no scale.

### 2. PP1 memory on klee100 (per card, against 719 MiB free as seen by the worker)
**Adds:**

| Item | MiB |
|---|---|
| Drafter weights | 745 |
| Drafter embedding (kept on PP1, own copy) | 606 |
| fp32 `draft_logits` = max_num_seqs × K × 248,320 × 4 (`speculator.py:248-256`, `dflash2/speculator.py:200-204`) | 21 seqs: K4 79.6 / K3 59.7; 24 seqs: K4 91.0 / K3 68.2 |
| Speculator and aux buffers (context hidden [1024 × 12800] bf16 = 25; reserve slots 2 × [1024 × 2560] = 10) | about 35-60 |
| Extra graph memory vs K100Q's 0.63 GiB (K100Q log 478-480) | about +120-320 |

- The drafter's own lm_head (606 MiB) is freed by the B5 release before the KV is allocated, so it costs no steady memory.
- Graph-memory evidence: W4 PP1 used 0.43 GiB with captures to 32; K100Q used 0.63 GiB with captures to 84.

**Frees:**

| Item | MiB |
|---|---|
| MTP routed experts (W4 g128, 256 local × 3 × 2560 × 640 × 0.5 B + scales / zeros / g_idx) | about 626 |
| MTP's own GPU embedding (`mtp.py:244-249`) | 606 |
| MTP attention / fc / shared expert / HC | about 30-50 |
| MTP bf16 `draft_logits` | 30 |
| `_mtp_hidden_buffer` (`nvidia/model.py:656-668`) | 20 |

**Net:**
- Static: +69 to +89 MiB.
- Total: about +250 to +500 MiB.
- Predicted worker-free on PP1 for D4s at 21 sequences: **about 220-465 MiB** (nvidia-smi about 690-935). D3s is about 20 MiB better; 24 sequences costs about 15-90 MiB more.
- W4's Q ran 116k@5 with 0 preemptions at worker-free [663, 687, 177, 175] (`e27_gpu_out/.../nbdf_results.json`).

**So pin 7.93 GB should hold; no rung is needed.** The 512 MiB gate will likely read False, which is record-only (`nbdf_lib.py:1977-1978`; flag mode at `nbdf_k100_cell7_body.py:746`). With `DFLASH_DRAFT_EMBED_HOST=1` the prediction is about 830-1,070 MiB.

### 3. KV packing (measured layouts transfer exactly)
**klee100 and W4 have identical block layouts:**
- block bytes [14680064, 16515072] and groups (4096, 4096, 4096, 8, 4096) (K100Q log 360, 362 vs `e27_gpu_out/.../server_nbdf_q.log` 240-242);
- the W4 rank-2 layout is listed in `nbdf_results.json` Q `kv_layout`.

**Predicted pools at pin 7.93 GB:**

| Arm | What binds | Blocks | F | Max conc. 131k | 64k residents | 90k / 116k |
|---|---|---|---|---|---|---|
| Q | PP1 page 16,515,072 | 480 (479 usable) | 6 | 12.61 | 21 | 17 / 13.7 |
| D4s / D3s (drafter SW 1024) | PP0 page 14,680,064 (PP1 drops to 14,155,776 once the MTP layer is gone; drafter page 10,485,760; kit v5 R9 K4_swblock1024) | 540 (about 539 usable, +12.5 %) | about 12 | about 12.2 | 18-20 | 15.8 / 13.1 |
| D4 (block 4096) | drafter page 41,943,040 | 189 | 9 | about 4.6 | — | — |

- The scheduler unifies the pool to the minimum across workers (`kv_cache_utils.py:2314-2316`).
- The D F of about 12 is 6 Mamba + 6 sliding-window blocks: `cdiv(2047 + 3×1024, 1024) + 1` (`kv_cache_interface.py:879-886`), with in-flight tokens = (PP + 1) × 1024 under V2 + async (`config/vllm.py:607-628`). The kit's 7-block drafter cost is the same mechanism.
- **The catch:** despite +60 blocks, D-SW holds fewer long requests than Q: 64k residents 18-20 vs 21, and 116k about 13 vs 13.7.
- **Possible later lever:** FP8 drafter KV at SW block 2048 has a page of 10.5 MB that fits, and F about 10. It needs TRITON_ATTN (#54690 risk).

**The engine-wide drop to block 1024 is not a problem for klee100's Mamba or QSA groups:**
- `core.py:434-448` lowers `cache_config.block_size` only in the engine-core process.
- Workers copy only the layout, hash block and alignment (`gpu_worker.py:1127-1141`).
- The scheduler block is the LCM = 4096 (`kv_cache_utils.py:849-855`).
- The Mamba geometry uses the MambaSpec block (`scheduler.py:446-455`; `mamba_hybrid.py:159-165`).
- The CircularBuffer ring is excluded (`core.py:435-443`), and the QSA ring reads its spec block.
- The hash unit is `prefix_match_unit` 64 (argv), which divides 1024 (`kv_cache_utils.py:879-887`).
- `compressor_utils.py:141` runs in the worker, where the block is still 4096.
- The drafter spec is not flagged non-causal, so prefix caching and chunked prefill stay on (`core.py:368-382`; only `prefill_prefix_lm_attention.py:85` sets that flag).
- **One residual risk:** the frontend learns block_size 1024 (`core.py:1890` → `core_client.py:860`), so `cache_config_info` will report 1024. NBDF's MIXED_KV_BLOCK_SIZES path handles this (`nbdf_lib.py:563-574`; `engine_block` at `:644-650`). Any K100Q-style worker rule `floor(pool / (ceil(65536 / B) + F))` must use B = 4096.

### 4. Humming MoE with DFlash verify
**Verify tokens per micro-batch:**

| Streams | Balanced, K3 / K4 | Worst parity (W4 saw 7 / 1), K3 / K4 |
|---|---|---|
| 8 | 16 / 20 | 28 / 35 |
| 16 | 32 / 40 | 60 / 75 |
| 21 | 44 / 55 | 80 / 100 |
| 24 | 48 / 60 | 92 / 115 |

The W4 parity is from `timeline_q.md:33`.

**Humming, measured only at T8-64** (`E25_GPU_wave1_NB3_NB2SMOKE_2026-10-31.md:15-25`):
- ratios vs Marlin W4 at T8/16/32/64: W3 0.84 / 0.98 / 0.84 / 0.85, W2 0.60 / 0.70 / 0.60 / 0.61;
- T16 is the worst point. Note that Q at 8 streams sits right on T16.
- Above T64, nothing is measured on humming; Marlin goes from 1,838 µs (T64) to 2,411 µs (T128) per layer.

**No hard limit:** Merlin has no per-token-count limit in this path. The gemm type is fixed (`fused_humming_moe.py:109-123`) and the heuristics are chosen once at init (`:157-179`). Prefill already pushes 1,024-token chunks through humming, and K100Q captured FULL graphs to 84 tokens. Speed at 100-120 tokens is the open question.

**Captures:** use every multiple of 1+K up to SEQS × (1+K), as in B2.

### 5. The seam on klee100
- **Unchanged:**
  - the HC weights are 16-bit in klee100 (`extra_config` `.*mlp_hyper_connection.*` bits 16, `fixkeys_overlay/config.json`);
  - the HC params are bf16 (`nvidia/model.py:634-646`) and `combine` is pure;
  - the taps are [T, 2560] bf16, the same as W4. The quantized parts sit inside the blocks (experts humming W3 / W2; attention / GDN projections W8 Marlin).
- **PP2 relay:** unchanged. The partition [25, 23] is verified on klee100 (`k100_quality_report.json` `boot.k100_gates.partition`).
- **Loader:**
  - the target drops `mtp.*` weights (`nvidia/model.py:1139, 1358`);
  - DFlash does not inherit the target's quantization; only mtp / dspark do (`config/speculative.py:1190-1200`). So the `UnquantizedLinearMethod` check passes;
  - the shared lm_head is the BF16 target head (not in `block_name_to_quantize`).
- **Embedding:** byte-identical across W4 and klee100, with the same mask-row sha (`README.md:41`; `nbdf_lib.py:77-85`).
- **Acceptance risk:** 3 of the 5 taps (layers 15, 25, 35) sit in the W2 g64 band, layers 12-35. MTP reads the final state instead.

### 6. Q-defer, the worker extension, the scales sidecar and the overlay
- **Q-defer:** the same nb4_ppack shas on the same ad42e37 tree (`k100_quality_report.json` `k100_setup.qdefer_tree.shas`). The pack follows the spec K (K100Q log lines 54-58: `"k": 3`), and unit test M2 covers [n, 4]. No klee100-specific issue.
- **Worker extension:** the merge asserts the NB1 parts are identical (`build_e27_nbdf_k100.py:99-108`).
- **`MERLIN_SHARE_DRAFT_WORKSPACE`:** a no-op with DFlash (`merlin/draft_workspace.py:71-95`); it already did nothing on K100Q (K100Q log 364-365).
- **Overlay and scales sidecar:** these touch target files only. The drafter is a separate mount.

### 7. Other things that could stop or waste the run
- **E28's I8 lever does not apply to D.** E28 klee100 runs I8 (MTP int8 draft head, untested on klee100) at 24 workers (E28 `README.md:23, 114`). DFlash shares the BF16 target head (`mtp.py:497` `supports_draft_lm_head_quant` is MTP-only). D must beat E28's Q, not only K100Q's.
- **Long-window budget.** klee100 long windows at levels 13-17 need 1.2-1.6M prefill tokens. At the W4-measured 3.9-4.1k tok/s that is 5-6.5 min each, against `admit_limit_sec` 420 (`nbdf_results.json` steady windows).
- **Tokenizer.** klee100's `tokenizer.json` sha (06b95093) differs from the pinned one (0997f410), although the `tokenizer_config.json` files are identical. Assert in the kit that id 248077 is unused.

---

## Risk list
1. **DFlash2 acceptance on klee100 is unknown.** Rank-1 agreement vs BF16 is 0.51-0.65, and 3 of the 5 taps come from W2 layers. The MTP AL was 2.8 on Duck and 3.21 on synthetic windows.
2. **Capacity:** D-SW's F of about 12 vs 6 holds fewer long requests (64k: 18-20 vs 21). This matters for the E28 port at 24 workers.
3. **Memory:** 220-465 MiB predicted on PP1. That is below the 512 MiB gate (recorded only) but above the W4 floor proven at 175 MiB.
4. **Humming speed above T64 is unmeasured,** and skewed micro-batches can reach 100-115 tokens.
5. **`cache_config_info` reports block_size 1024 on SW arms,** which breaks any capacity math that uses it.
6. **The prefix cold/warm AL gate** (warm < 0.8 × cold) is noisy on a non-deterministic base. At worst it costs a SHORT_ONLY_WIN, not a stop.
7. **The tokenizer sha differs;** a quick check closes it.
8. **The 150-min budget at klee100's long levels is tight.**
9. **The W4 scales sidecar is reused on klee100 (EXPERIMENTAL).** This is the same for Q and D, so it adds no new risk.
10. **E28 is already on GPU.** Run NBDF-K100 on another account (limit: 2 GPU commits per account).

## Recommendations
- **Pins:**
  - Q and the D arms all start at 7,933,526,016, so the A/B compares equal KV bytes and D gets about 539 blocks.
  - On a D OOM: first `DFLASH_DRAFT_EMBED_HOST=1` at the same pin, then 7,665,090,560 (522 blocks), then 7,396,655,104 (503).
  - Never take the −1 GiB step first.
- **Arms:**
  - **Q:** the K100Q config (MTP K3, Q-defer, `MERLIN_SHARE_DRAFT_WORKSPACE=1`), seqs 24.
  - **D4s:** K4, SW 1024, captures 5..120.
  - **D3s:** K3, SW 1024, captures 4..96.
  - No D4 (189 blocks), no D7 (60-115 verify tokens at 16-24 streams), no D4b / D4t on klee100.
- **Windows:**
  - 8k@8 × 2 (comparable with W4), 20k@16 and 20k@24 (the K3_ONLY baselines), 40k@16;
  - 90k and 116k at one paired level for both arms, e.g. 12 and 10, which fits the 420 s admit limit;
  - one D window per long band.
- **Gates:**
  - **Hard:** GPU_METHOD_MAP_PASS (MTP rows required on Q only); STATIC_SCALES_APPLIED; `draft_load_check` × 2; embedding own + mask-row sha; lm_head shared with released bytes > 0; "Using FLASH_ATTN" on PP1; drafter FULL_DECODE_ONLY graph count equal to the capture count; garbage; zero acceptance; request errors; the teacher-forced D-vs-Q check inside Q's A/A envelope (B1).
  - **Recorded only:** memory, `KV_GROUP_EXPLOSION`, prefix hits, `PP0_GRAPH_DEGRADED`, and token-ID gates whenever Q's own A/A is not clean.

## What the K100 CPU kit must measure first
1. **R9 on the real klee100 config** at 7,933,526,016: Q must calibrate to 480 blocks (GPU shows 479 usable, page 16,515,072). D-SW K3 and K4 should give 540, with a 10,485,760 B drafter page and a 14,155,776 B PP1 page.
2. **F for D-SW** from `SlidingWindowSpec.max_admission_blocks_per_request`, using the real VllmConfig `max_in_flight_tokens` (PP2 + V2 async); expected 12. Also the max concurrency at 131k with 21 and 24 sequences.
3. **The core.py block drop, simulated:** `cache_config.block_size` 1024, scheduler block 4096, hash 64. Then show that NBDF's `level_for_band` returns non-None levels at block 4096 (MIXED path).
4. **The real freed and added bytes** from the safetensors headers on the mounted klee100 Model: the `mtp.*` tensor bytes per PP1 card (experts split over EP2, attention over TP2) against the drafter (0.727 GiB) and its embedding.
5. **R12 with the klee100 target ModelConfig** (quantization inc): the draft quantization is None, the drafter linears are Unquantized, and `draft_load_check` gives 67 = 67.
6. **`_k100_scan` on the meta-built DFlash2 drafter** returns no rows, and `method_map_check(need_mtp=False)` gives PASS. Run this through the real worker-extension code, not only the unit fixtures.
7. **Seam apply / verify / revert** on the klee100 install tree, together with nb4_ppack and #60490 (three-state check).
8. **klee100 tokenizer:** the highest added-token id is ≤ 248076, the vocab is 248,320, and the mask-row sha from klee100's `model-00002` equals 8606…1a19.
9. **Capture coverage** (`v2_decode_graphs` covers every batch size) and argv read-backs for seqs 24 (D4s 5..120, D3s 4..96, Q registry list), plus Q-defer pack views at [n ≤ 24, 3 or 4].
10. **The B1 relative-gate logic and the B3 rung walk** as unit fixtures. One fixture must use a non-deterministic Q A/A, which must not stop a D arm.
