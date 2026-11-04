# E34 build report: SSM bf16 + block 2048 capacity (2026-11-03)

Status: built, **closed** (`gpu_ready` false: every arm SKIPPED_KV_KIT_GATE until the gated rebuild with the E34 real-kit report).
Nothing was pushed to Kaggle, no dataset was uploaded, no GPU ran. Every file I wrote is under
`D:\ARC AGI 2\experiments\Experiment 34 - SSM bf16 + block 2048 capacity (planned)\`, all LF (0 CR, checked with Python).
The builders assert byte equality before and after of: every file in the Exp 25 Build folder + the NB-D1 notebooks, every Exp 27 Build
file + notebook, every file under Exp 29 (148 files), the E31 trace files read, and the capacity port's measure_capacity.py.
`git status` shows nothing changed outside the E34 folder.

## 0. Findings that change the plan (read first)

1. **Block 2048 is accepted, but only with the drafter SW block lowered to 512.** At block 2048 with the production
   `DFLASH_DRAFT_SW_BLOCK=1024`, the drafter's 5-layer SW page (10,485,760 B) becomes the largest page on PP1 and sets the pool stride:
   pool 756 blocks of 2048 (mc@131k 9.93, worse than B's 12.25). With SW 512 the page is 5,242,880 B, the stride is the Mamba page
   (7,340,032 B on PP0), pool **1,080 blocks of 2048** (= B's 540 x 4096 tokens), mc@131k **13.32**. So S16B2 = SSM bf16 + `--block-size 2048`
   + `DFLASH_DRAFT_SW_BLOCK=512`. (Merlin's real planner on CPU, below.)
2. **S16 (bf16 at block 4096) gives zero capacity**: the Mamba page is padded to the 2 MiB attention page per layer either way
   (pool 540, F 12, mc 12.25 = B; padding goes 21.01 % -> 121.6 %). S16 is the pure accuracy / speed control, as planned.
3. **fp32 at block 2048 is impossible**: the platform raises the block back to 4096 (fp32 GDN state 1,732,992 B > the 1,048,576 B attention
   page at 2048). So `Setting attention block size to 4096` on S16B2 = the bf16 flag not engaged (a real failure in the read-back).
4. **F does not halve.** F6 estimated 6 x 2,048 = 12,288 token-equivalents. Merlin's planner: F_adm 12 -> **17** blocks, because the SW
   drafter group at block 512 needs 11 blocks instead of 6 (the 2,047-token window + in-flight tokens). Token-equivalent: 12 x 4,096 =
   49,152 (B; the GPU mc line gives F 13 = 53,248) -> 17 x 2,048 = **34,816** (-29 %; GPU-log F at mc 13.32 = 18 = 36,864).
5. **Residents (static admission, Merlin planner, pool - 1, ring 1 block):**

   | context | B (540 x 4096, F 12) | S16B2 (1080 x 2048, F 17) | gain |
   |---|---|---|---|
   | 40k | 24 (seqs cap; pool allows 24) | 29 at seqs >= 29 | +21 % only with W |
   | 64k | 19 | 22 | +16 % |
   | 90k | 15 | 17 | +13 % |
   | 116k | 13 | 14 | +8 % |

   With the body's growth-aware level rule (nbdf_lib.level_for_band, 3,000 tok/s admission, 90-s window) the window levels are
   B 20 / 13 / 11 vs S16B2(W) 22 / 14 / 12 at 40k / 90k / 116k.
6. **The never-do list forbids this lever today.** `nb4t_lib.NEVER_DO_ARG_VALUES = {"--mamba-ssm-cache-dtype": ("bfloat16",)}`
   ("C3 SSM bf16: parked (0 capacity + B2 FAIL)"). Source: E25 NB1 R2 (2026-10-31, W4 Flash-Next): a combined B2 boot (SSM bf16 + a
   shared workspace + the commit graph) gave an oracle FAIL_HIGH_MARGIN (row 1 diverged at token 31, margin 0.75; 3 rows NEAR_TIE) and
   zero capacity at block 4096. That run never isolated the cause. nb4t_lib is NOT edited. E34 lifts exactly that one entry, only on
   S16 / S16B2 / S16B2W (`E34.launch_check` / `check_arm_plan` drop exactly the string `argv --mamba-ssm-cache-dtype bfloat16` when the
   arm's plan carries the flag); every other never-do item still refuses (unit-tested). This is your request; please confirm the lift.
7. **TF noise on this model is high.** The klee100 LOAD K3 run (pin 7,933,526,016) measured TF A/A fresh-vs-repeat on the same boot:
   2k mean|dlp| 0.66 / rank-1 0.856, 8k 0.098 / 0.949, 40k 0.283 / 0.914. B's own A/A in E34 is the envelope for S16*'s numbers.
8. **"KL" = NBDF's TF metric, not a full-vocabulary KL.** NBDF's TF (`L.teacher_forced`, prompt_logprobs=1) gives the forced token's
   logprob, its rank and the rank-1 token per position (last 256 positions): reported as mean / p99 / max |delta logprob| and rank-1
   agreement. A true KL needs full distributions = new code; I kept NBDF's code as instructed.

## 1. Files (sha256, bytes)

Build `...\Experiment 34 ...\Build\`:

| File | sha256 | Bytes |
|---|---|---|
| build_e34.py (builder) | 94fdd45a2e8d31a9fe5aff6fa25f8cff3a9da0d13cf04e479072cc582c891068 | 37,843 |
| e34_lib.py (arm plan, checks, read-back, capacity, W, TF set, time, LIVE, gate) | 65703276bf9bd93a11d4a088687efb98f62310ca80e8150968acca5df98d7d79 | 43,192 |
| e34_body_add.py (the inserted body blocks) | 8f5d84a1ded74daa0ef4cacdb68e27eeef55ac946d4bac62d32a94a7dd496928 | 8,403 |
| e34_unit_tests.py | ed866c92a70727941ca02b8f4954e7b2caa3ed5dbae0ad8f19189d41e9fb1ac3 | 51,502 |
| build_e34_units_kit.py | 08a930c312a5b0d8d82a51a8f7ab269de76314ce9afa5dda3f0c6a290aacf9de | 7,800 |
| build_e34_cpu_kit.py (real kit) | 0d2e257639c256ad9d745439d74a9e6bfe294c70e7bbc8ce3fc8f486a3a48c46 | 30,565 |
| e34_capacity_probe.py (Merlin planner on CPU) | a6be881a869e3880dfc78a2be8afe81c456c2851dabef1d6cb31cd20fa27291d | 6,939 |
| make_e34_fixture.py | 582b233d6e0496f349cdcba8c780456e2cfc738ced160a018a025437c16b561f | 1,968 |
| e34_log_fixture.json (NB-S1 GPU worker record + boot log lines of B's boot path; no generated text) | c4befc199b99b1888f6eae1db60189e4d0168d3fa0a5cd94a24de6b9fa332cd6 | 26,203 |
| e34_cell7_body.py (output) | 65cc785e188481579a0aeac0a09f243ce17d605fb3f72d5129db5e8250b08b9d | 139,261 |
| e34_code_export.py (output, FULL's cells) | eeab94bbcb0b1e02b48a071c177a06bb5b85ea791f293c11137ef20c0f26a396 | 328,814 |
| e34_build_facts.json (output) | 3554a19e4d3742e18f720413e9a66dd8d7dedeae58382180052e98e8e294c63c | 48,824 |

`Experiment Results\e34_capacity_probe.json` (CPU planner output): 3b82ebc7be2cc96b9ad4b8d90660f22145669e903a1c920b0c25c7820477432a, 73,909 B.

Notebooks `...\Experiment Notebooks\` (11 cells; the two differ only in the CELL 1 copy block, asserted):

| Notebook | Role | sha256 | Bytes |
|---|---|---|---|
| Q27-XHIGH-E34_FULL.ipynb | FULL copy = the kit notebook = the closed build | be5fcd583bd6842fd09a5788eb39fdd831462f023b6e13eea2788006ae9ddcaf | 348,088 |
| Q27-XHIGH-E34_SMOKE.ipynb | SMOKE copy | c0ddc0b31dcf2a79ef1c7a4f79ff466cef857258a0c7102b48bbeeacd9ee305c | 348,090 |

Push folders (not pushed) under `Build\push\`:
- `q27-xhigh-e34-full__foysalemonshanto` and `q27-xhigh-e34-smoke__foysalemonshanto` (GPU, NvidiaL4, pinned image, private).
- `q27-e34-kit-units__foysalemonshanto`: ipynb 6c8a38f85effa010ce37d9b29630261bfde9a1f8bb1c15ca5d7a6a0de4f7b15f, 381,729 B, no sources.
- `q27-e34-kit-real__foysalemonshanto`: ipynb 100f3d2402815eddc1650163aae6091f9e24ad3958a72d9e27f79a88374cdd7d, **394,708 B** (5.3 kB under 400 kB;
  its E34 items are lzma-packed inside the kit to fit). Sources: tf-ref dataset, q27-e34-tf-long dataset, the 4 Models.
- `ds_e34_payload` (dataset foysalemonshanto/q27-e34-payload): e34_cell7_body.py, nbd1_lib.py (af0f795f, = NB-D1's), e34_lib.py,
  SHA256SUMS.json, dataset-metadata.json.
- `ds_e34_tf_long` (dataset foysalemonshanto/q27-e34-tf-long): e34_tf_long.json bd7c1db3abebb5a88633a5f72a543aceeb5d1402957ee0c58199bc49e09ecb20,
  1,035,260 B (the 3 recorded traces), SHA256SUMS.json, dataset-metadata.json.

Audit zip: `Experiment Review\Audit Zips\E34_build_audit_2026-11-03.zip` (33 files: the Build sources and outputs, both notebooks, both
kits, the payload folder, the CPU planner results, the F6 note and this report; the 1 MB TF-trace file is not in it).

Notebook sources: datasets `foysalemonshanto/q27-e25-nb0-tf-ref`, `foysalemonshanto/q27-e34-payload` (new), `foysalemonshanto/q27-e34-tf-long` (new);
Models `qwen38-flash-next-klee100-3bpw-mtp/PyTorch/autoround-3bpw-mtp-nople/1`, `qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1`,
`e23-merlin-env/Other/v1/1`, `qwen38-flash-next-dflash2v2/transformers/bf16-embed/1` (all foysalemonshanto; owner of both copies: foysalemonshanto).

## 2. How it is built

NB-D1's pattern, one level up:
1. NB-D1's builder (Exp 25 `build_e25_nbd1.py`) runs in-process with `--test-out Build/_ref_nbd1` (deleted after the build). Its closed
   notebook must be byte-identical to NB-D1's **abbf21df** (asserted); inside it, NBDF-K100's e2198285 / c20eba7f, seam 341d859b,
   nbdf_lib b0147751 are asserted by NB-D1's builder. So B's boot path IS NB-D1's B (E29_DFLASH).
2. CELL 0 = E18 R2 verbatim (asserted). A competition rerun writes CELL 0's identity submission (E18 R2 format).
3. CELL 1 gets a copy block at the top: `E34_ON = {...}` + `E34_BUDGET_MIN = N`; BUDGET / deadman come from it; out dir `/kaggle/working/e34`,
   results `e34_results.json`, verdict `nbd1_verdict.json` + `e34_verdict.json`.
4. The CELL 7 body = NB-D1's body + exact-once marked edits (below). Body, nbd1_lib and e34_lib come from the payload dataset through the
   sha-pinned loader `_e34_ds` (aliased `_nbd1_ds`, env fallback `E34_PAYLOAD_DIR`); the recorded traces from their own sha-pinned dataset.
5. Deterministic: a `--test-out` rebuild gives the same be5fcd58 / c0ddc0b3.

Body edits (each exact-once, in `build_e34.py`):
- E34 setup after NB-D1's setup: `ARMS = E34.arm_plan(...)` on NB-D1's B, `E34.check_arm_plan` (raises: no boot), `NBD1_SWITCH = E34.switch_map(E34_ON)`.
- Distinct prompts: 20k / 40k to 40 variants, 90k / 116k to 24 (the W windows must not repeat a prompt).
- The TF set: `TF_P = E34.tf_set(...)` = RC2's 8k / 40k TF prompts (variant 5) + the three traces; SMOKE: the 113k trace only.
- `_nbdf_launch_check` calls `E34.launch_check` (NBDF's check minus exactly the lifted entry + E34's own SSM / block / seqs / SW checks).
- `e34_capacity_rec` right after `_k100_residents`: the boot capacity + read-back record, printed as `[E34 READBACK]`, `[E34 BOOTLOG]` and a
  `[LIVE] <arm> boot` line.
- `d1_readback_failure`: + `E34.readback(...)`; a mismatch = `E34 read-back: ...` = NOT_INTEGRATED for that arm only.
- `_live`: + `e34_item_live` (a `[LIVE]` line after smoke / oracle / oracle A/A / long oracle / prefix test / TF ref / TF A/A / TF vs B).
- `d1_live`: `E34.live_line` (NB-D1's line + the window's prefix hit rate; a W window is compared with B's W window of the band).
- An empty window set = no window and no warmup (the smoke plan); before, `[] or K100_CFG["windows"]` would have run the K100 set.
- TF: the smoke plan stores its own reference + A/A (`is_q or arm.get("e34_smoke")`).
- `nbd1_main`: before S16B2W, `e34_w_replan` (W from S16B2's measured boot, re-checked plan; a failed check keeps the planned arm, recorded).
- `nbd1_finalize`: `v["e34"] = E34.records(...)` (+ W + TF set), written to `e34_verdict.json`; CELL 8 prints `E34 RECORDS`.
- NBDF's oracle / long oracle / prefix test / TF code is NB-D1's own (its `D1.item_on` switches, now on); no new copies of that code.

## 3. Arms and plans

Every arm: DFlash2 K3, SW BF16 drafter KV, FLASH_ATTN drafter, seam 341d859b + GRD, Q-defer, EMBED_HOST from the first try, QSA 128 / 64,
pin 7,933,526,016, captures every multiple of 4 up to 4 x seqs, `--language-model-only`. Merlin passes `--block-size 4096` and
`--prefix-match-unit 64` itself (launch.py:334 / :474-475); the notebook's extra args come last, so the last `--block-size` wins.

| Arm | Change from B (argv appended / env) | seqs | Windows | Items |
|---|---|---|---|---|
| B | none (fp32 SSM, block 4096, SW 1024) | 24 | 8k@16 #1, 8k@16 #2, 20k@24, 40k@16, 90k@8, 116k@6 + BMAX 40k / 90k / 116k at max residency (requested 29, capped at run time: predicted 20 / 13 / 11) | prefill probe, oracle + its A/A, long oracle (stores B's rows), prefix test, TF ref + A/A |
| S16 | `--mamba-ssm-cache-dtype bfloat16` | 24 | 8k@16, 20k@24, 40k@16, 90k@8, 116k@6 | oracle vs B, long oracle vs B, prefix test, TF vs B |
| S16B2 | `--mamba-ssm-cache-dtype bfloat16 --block-size 2048` + `DFLASH_DRAFT_SW_BLOCK=512` | 24 | same as S16 | same as S16 |
| S16B2W | S16B2 + the trailing compilation pair + `--max-num-seqs W` | W (planned 29) | S16's set + 20k / 40k / 90k / 116k at W (capped at run time: predicted 29 / 22 / 14 / 12) | same as S16 |

- **W rule:** W = (S16B2's num_gpu_blocks - 2) // (ceil(40,000 / attention block) + F), F from S16B2's max-concurrency line
  (nbdf_lib.f_blocks formula), clamped 24 .. 40. CPU planner: (1080 - 2) // (20 + 17) = 29; with the GPU log's 2-decimal mc 13.32 F becomes
  18 and W 28. If S16B2 is unreadable or did not run, the planned W (29) is kept (recorded). Every W 24 .. 40 is a checked plan (unit + kit).
- **Real failures (that arm only):** crash; OOM (the ladder); a flag rejected at launch (E34.launch_check); the read-back of the arm's
  planned SSM dtype (worker `cache_config.mamba_ssm_cache_dtype` + the GDN MambaSpec state dtype), attention block (worker
  `cache_config.block_size`, the FullAttentionSpec block, any `Setting attention block size` line) or drafter SW block (the SW spec block,
  the seam's draft_sw_block events) with another value; NB-D1's existing hard gates (smoke FAIL_* = garbage / zero accept / errors, NaN
  through the hard garbage checks, spec mismatch, Q-defer, draft load check). Unreadable read-backs, KL / oracle / speed labels: recorded only.
- **Read-backs logged explicitly per arm:** `[E34 READBACK] <arm> {cache_config.mamba_ssm_cache_dtype per rank, cache_config.block_size per
  rank, drafter SW block per rank, GDN state dtypes, GDN unpadded / padded page, log: kv cache group sizes, padding %, setting-attention-
  block, GPU KV cache size, max concurrency, the Qwen3.5 dtype-override warning, the non-default dtype / block}`, every matching boot log
  line as `[E34 BOOTLOG]`, then `[LIVE m] <arm> boot | pool P x block | F f (tok-eq) | mc@131k | KV tokens | pad % | groups | ssm | block | sw | readback ok`.
- **Records per arm (`e34_verdict.json`):** capacity (pool, attention block, mc, F, F token-equivalent, KV tokens, stride per stage, unused pin
  per stage, static residents at 40k / 64k / 90k / 116k, the CPU planner's prediction), the read-back, W, oracle / oracle A/A / long oracle
  status vs B, prefix gate, TF numbers and gate, smoke gate, every window's tok/s, AL, running max, level, preemptions and prefix hit rate.
  NB-D1's KV ledgers, K_eff and band tables vs B are kept. Every response is saved (gzip jsonl per arm, `responses/`), as in NB-D1.
- **Pin / kv_cache_memory_bytes per arm:** unchanged (7,933,526,016, the same D ladder 7.93 -> 7.67 -> 7.40 -> 6.86 G, OOM at any stage, no
  same-pin EH rung). Re-derived: pool = floor(pin / largest group page of the rank); B 7,933,526,016 / 14,680,064 = 540.4 -> 540, S16B2 /
  7,340,032 = 1,080.9 -> 1,080: both waste about 6.3 MB, so no pin change gains a block, and equal KV bytes keep the comparison fair. The
  non-KV side: SSM bf16 and block 2048 change no non-KV allocation; S16B2W captures 25 .. 29 more graphs (risk below).
- **TF traces (E31 direct, Flash-Next recorded thinking, prompt + p1):** tf69k = 5545f144 (2,734 + 66,737 = 69,471), tf94k = 409aa875
  (3,772 + 90,223 = 93,995), tf113k = 0934a4d8 (9,386 + 104,000 = 113,386).
- **Prefix:** NBDF's prefix test (2 prompts x 3 turns, warm vs cold after a cache reset) at the arm's own engine attention block (4096 on
  B / S16, 2048 on S16B2*), + every window's prefix hit rate in its `[LIVE]` line.

## 4. Time per copy (minutes; estimate / body cap)

Model: NB-S1 full GPU data (first ready 714 s, warm 252-282 s, windows 8k@16 2.1 .. 116k@6 4.55), NBDF-K100's Q GPU run (oracle 0.28 + A/A
0.30, long oracle 0.52, prefix 1.94, warmup 0.76); TF (never GPU-run on this path) 2.7 min a pass of 324k prompt tokens; unmeasured windows
= admission at 3,000 tok/s + 90 s + 40 s (+1 min reset on long bands); recompiles 7 / 7.5 / 8.5.

| Copy | Arms (est / cap) | Planned runtime | Full-cap end | Hard stop / deadman |
|---|---|---|---|---|
| FULL | setup 6, B 77 / 65, S16 40 / 33, S16B2 40 / 33, S16B2W 75 / 67 | **238 (about 4.0 h)** | 239.3 | 245 / 250 |
| SMOKE | setup 6, S16B2 smoke 20 / 9 (first-boot ready 12, facts 1.5, smoke 1.3, long oracle 90k + 116k 0.7, TF 113k ref + A/A 2.0, end 1.2) | **26** | 27.8 | 35 / 40 |

FULL is under 11 h, so it is one copy (no A / B split). The SMOKE copy: boot of S16B2 + the first requests (probes, sampled check, smoke)
+ the long prefills (90k and 116k greedy + the 113k TF prompt, twice) + every read-back. I added the 113k TF to the smoke because TF
(prompt logprobs) has never run on this pin with the drafter on PP1 (see risks): the smoke shows it before the 4-h run.

## 5. Merlin source verification (pristine Merlin ad42e37 at `_scratch/e23/p2/merlin-src`; the capacity-port copy has the same lines except interface.py / kv_cache_utils.py env-gated patches)

SSM bf16 path:
- `vllm/engine/arg_utils.py:1396` `--mamba-ssm-cache-dtype`; `vllm/config/cache.py:63` MambaDType includes "bfloat16", `:186` default "auto".
- `vllm/model_executor/models/config.py:908-934` (Qwen3_5 hook; Qwen4ExpForConditionalGenerationConfig `:966-973` inherits it): "auto" ->
  the HF `mamba_ssm_dtype` (float32 on this model); an explicit value wins with the warning at `:928` ("... Using the user-specified value.").
- `vllm/models/qwen4_exp/nvidia/model.py:1016-1027` GDN state dtypes = (conv, ssm) from cache_config + an fp32 replay record under RecoverSSM;
  `vllm/model_executor/layers/mamba/mamba_utils.py:99-112` the SSM dtype = `cache_config.mamba_ssm_cache_dtype`.
- `vllm/config/vllm.py:3417-3423` RecoverSSM accepts bf16 / fp32 SSM caches (its only refusal: stochastic rounding); `:3477-3491` fp32 is
  required only for NemotronH Mamba-2.
- `vllm/model_executor/layers/mamba/gdn/recoverssm_gdn.py:254` only the replay record must be fp32 (no checkpoint dtype check); `:440` / `:445`
  the state store casts to the cache dtype; `:495` every layer the same state dtype.
- GDN state per layer (from the GPU record): conv bf16 [6, 5120] 61,440 B + SSM [24, 128, 128] (fp32 1,572,864 / bf16 786,432 B) + replay fp32
  [24, 4, 257] 98,688 B = 1,732,992 B fp32 / 946,560 B bf16.

Block 2048:
- `vllm/platforms/interface.py:940-1122` `_align_hybrid_block_size`: kernel alignment = max(min backend block, block_size) (`:1077-1086`),
  attention block = alignment x cdiv(Mamba page, alignment x 512 B) (`:1087-1093`), raised + "Setting attention block size" when smaller
  (`:1095-1101`), align mode `mamba_block_size = block_size` (`:1103-1104`), the Mamba page padded to block x 512 B (`:1107-1122`, "Padding
  mamba page size"). bf16 946,560 <= 1,048,576 at 2048 (padding 10.78 %); fp32 -> raised to 4096.
- FA2 paged-KV rule on sm_89: `vllm/v1/attention/backends/flash_attn.py:354-365` -> `MultipleOf(16)` (FA4 / sm90 sizes not taken on sm_89);
  the target's QSA backend `vllm/models/qwen4_exp/nvidia/qsa.py:377-379` `MultipleOf(16)`. 2048 and 512 are both multiples.
- QSA ring group: `vllm/models/qwen4_exp/common/qsa_cache.py:954-970` capacity 4 x ceil((4 + K) / 4) = 8 must divide the block: 8 | 2048.
- Indexer: `qsa_cache.py:976-986` the compressed-key cache = an MLAAttentionSpec at the cache block with tokens_per_state 4 (512 rows a 2048
  block); `vllm/models/qwen4_exp/nvidia/indexer_qsa.py:187-200` page_cap = blocks x block / 4 = 32,768 rows at 4096 and at 2048 (unchanged).
- prefix_match_unit 64: `vllm/v1/core/kv_cache_utils.py:805-905` scheduler block = LCM of the group blocks (`:852`, 2048), every
  prefix-cacheable group block divisible by 64 (`:884`; 2048, 512), the 4-token compression alignment (`:904`; 64 % 4 = 0).
- Align mode: `vllm/v1/kv_cache_interface.py:1077-1083` 2 pages a request (running + CoW checkpoint) with num_speculative_blocks 0 under
  RecoverSSM; the checkpoint is every 2048 tokens at block 2048.
- Engine block: `vllm/v1/engine/core.py:431-450` = min over prefix-cacheable groups = the SW drafter block (1024 on B, 512 on S16B2);
  `vllm/config/vllm.py:3289-3318` validate_block_size (align: chunked MM input only).
- Pool: `vllm/v1/core/kv_cache_utils.py:1251-1262` (+ `:1304` unify) one pool block = the largest group page of the rank.
- DFlash2: Exp 27 `dflash_seam.py` ATT_PATCH (`:1116-1180`) `DFLASH_DRAFT_SW_BLOCK` must divide the attention block (512 | 2048; 1024 | 2048
  too, but its page sets the stride, finding 1); the seam's draft_sw_block event logs block_size / draft_block (read back).
- Merlin launcher: `tools/merlin/launch.py:334` BLOCK_SIZE default 4096 -> `:474` `--block-size`; `:475` `--prefix-match-unit 64`; extra
  args last (`:110-111`).

Merlin's real planner on CPU (`Build/e34_capacity_probe.py` = the capacity port's measure_capacity.py, sha 27d8b1c9, with exact-once edits
for block / SSM dtype / SW block; the klee100 K3 GPU worker record; output `Experiment Results/e34_capacity_probe.json`):

| Config | pool | stride PP0 / PP1 (B) | F adm / steady | mc@131k |
|---|---|---|---|---|
| B: fp32, 4096, SW 1024 | 540 | 14,680,064 / 14,155,776 | 12 / 10 | 12.25 |
| S16: bf16, 4096, SW 1024 | 540 | same | 12 / 10 | 12.25 |
| bf16, 2048, SW 1024 | 756 | 7,340,032 / 10,485,760 | 12 / 10 | 9.93 |
| S16B2: bf16, 2048, SW 512 | 1,080 | 7,340,032 / 7,077,888 | 17 / 12 | 13.32 |
| fp32, 2048, SW 512 | BLOCK_RAISED | - | - | - |

B's row reproduces the NB-S1 GPU boot exactly (pool 540, mc 12.25).

## 6. Local tests

- `e34_unit_tests.py`: **180/180 PASS** on Py3.14.6 and on Py3.12 (local). Covers: the plan for FULL / SMOKE / default (B = NB-D1's B
  exactly; each lever = B + exactly its change; 17 mutations refused; every W 24 .. 40 valid); the never-do lift (exact string, SSM arms
  only, other never-do items still refused); E34.launch_check on Merlin's argv + the extra (passes all arms; 6 negatives caught); W from the
  boot capacity (29 / 28 / unreadable / clamps); the read-back on the NB-S1 GPU worker record (B passes; S16 on an fp32 boot fails; S16 /
  S16B2 expected records pass; raised block, a `Setting attention block size` line, SW 1024, seam events disagreeing, fp32 all fail; unreadable
  recorded); the capacity record (B on the GPU record: pool 540 x 4096, mc 12.25, F 13 = 53,248 tok-eq, groups, padding 21.01 %) + the boot
  line; the log parsers (non-default dtype / block, the override warning); the CPU planner file = E34.CPU_PLANNER; TF set; time / budgets;
  [LIVE] lines; the kit gate; body markers + compile + KV gate vs facts; arm_body with fakes per copy (windows exactly the plan, oracle x2 on
  B / x1 levers / 0 smoke, long oracle, prefix, TF x2 on B and smoke / x1 levers, capacity record once, no warmup on smoke, read-back failure ->
  NOT_INTEGRATED before the smoke, smoke FAIL -> NO_GO_SMOKE); the body's `d1_readback_failure` on GPU-derived records; arm_boot with fakes
  (OOM at any stage -> 7,665,090,560 with SW 512 / argv kept; 4-rung ladder; Xid no retry; gating); nbd1_main with fakes (FULL order, the W
  re-plan from S16B2's capacity incl. a changed W, the CPU-planner fallback, SMOKE runs only S16B2, the closed gate boots nothing, a failed B
  never skips the levers); run_window (the [LIVE] line with prefix hit and B's W window); `_live` prints one line per test item; facts /
  export / copy block / CELL 0 / the SMOKE copy differs only in the copy block.
- Negative control: the same tests against NB-D1's body / facts / export give 120/151 (they tell the builds apart).
- Units-kit layout simulation (empty folder, `python -I`, Py3.12): **178/178 PASS, `@@KIT units ALL OK@@`** (the 2 notebook checks are the
  allowed `SKIP notebook`).
- Gated path in `--test-out`: a synthetic all-True report on the closed FULL build -> `gpu_ready` true; one key False -> closed (checked on the a6229315 build; the COMP edit changes no gate logic).
- Static check: no undefined names in the notebook cells; the body and the real kit compile; the kit's packed E34 items compile.
- Not run locally (by the rules): the real kit (needs the Merlin venv on Kaggle), any end-to-end notebook run.

## 7. Kits (built, not pushed)

- **Units kit** `q27-e34-kit-units__foysalemonshanto` (381,729 B): e34_unit_tests.py on the built body / export / facts with the real
  nbdf_lib / nb4t_lib / nb4_ppack / nbd1_lib, e34_lib, the NB-S1 fixture and the CPU planner file. Report `e34_kit_units_report.json`.
- **Real kit** `q27-e34-kit-real__foysalemonshanto` (394,708 B): NB-D1's real kit (itself NB-S2's = NBDF-K100's real kit template) retargeted
  to the E34 FULL export: the notebook's own pre-launch sequence for all 4 arms (merge -> merlin_install -> 60490 -> k100_setup -> sha gate ->
  PSEQ(B) -> SSEQ.apply -> per arm SSEQ.check + PSEQ.before_arm + arm_boot first try + merlin_fn_server with K6, Merlin build, host phase,
  the notebook's E34.launch_check and the `--help` flag gate), the spawned-child env. E34 items: vLLM's own parser + EngineArgs on each final
  argv (B + exactly mamba_ssm_cache_dtype / block_size / seqs / captures), the installed Merlin's source facts of the bf16 / 2048 path and
  MambaStateDtypeCalculator's GDN dtypes, Merlin's planner on the GPU record (B 540, S16 540, S16B2 1080, SW 1024 756, fp32 x 2048 raised,
  engine block 1024 / 512, scheduler 2048, hash 64, F 17-18), the never-do lift, the launch env SW block, the plan (FULL / SMOKE / every W),
  the TF dataset. Gate keys: `e34_arm_plan_ok, e34_launch_path_all_arms_ok, e34_help_flag_gate_real_ok, e34_argparse_change_per_arm,
  e34_kv_probes_ok, e34_launch_env_ok, e34_captures_argv_packviews_ok, e34_switch_default_ok, e34_neverdo_lift_exact, e34_merlin_source_ok,
  e34_tf_long_ok`. Report `e34_kit_real_report.json` -> `python -B build_e34.py --kv-gate-report <it>`.
- Both kits are bound to the closed FULL sha be5fcd58; any builder change means rebuilding both.

## 8. Steps to GPU (each needs your go where marked)

1. Upload `Build/push/ds_e34_tf_long` as **foysalemonshanto/q27-e34-tf-long** (static, independent of the gate; too big to embed in a kit).
2. Push the units kit and the real kit (CPU, foysalemonshanto).
3. `python -B build_e34.py --kv-gate-report <...>/e34_kit_real_report.json`; check facts `gpu_ready` true. This rewrites the notebooks, the body
   and `Build/push/ds_e34_payload`.
4. Upload `Build/push/ds_e34_payload` as **foysalemonshanto/q27-e34-payload**.
5. GPU commit of the SMOKE copy (needs go), then the FULL copy (needs go).

## 9. Open risks

- **The lever is on the never-do list** (finding 6). Lifted for this notebook's 3 SSM arms only; the earlier B2 oracle FAIL is unisolated
  context, not a verdict. Needs your confirmation.
- **SW 512 + engine block 512 never ran on GPU.** The seam's docs warn that readers of a lowered `cache_config.block_size` are untested below
  the GPU-proven 1024; Merlin seeds the Mamba state column from mamba_block_size (fine). The SMOKE copy covers boot + long prefill on exactly
  this layout. The CPU planner also shows the SW group needs 11 blocks at 512 (6 at 1024), which is why F does not halve.
- **TF (prompt logprobs) on this pin with the drafter on PP1 never ran.** NB-S1's B had 599 MiB free on the PP1 cards after ready; prompt
  logprobs build [1024 x 248,320] fp32 logits per chunk on the last stage. The klee100 MTP boot at the same pin ran TF at 40k without OOM,
  but D3s's PP1 is fuller. An OOM in TF counts as an OOM at any stage (your rule) and re-boots the arm one rung lower, which can cost a whole
  arm (B: 77 min) and leaves B and the levers on different pins. The SMOKE copy runs the 113k TF twice first to show this.
- **S16B2W's extra graphs** (captures to 116 instead of 96) add graph-pool memory on the tight PP1 cards; the ladder handles an OOM (7.67 G
  -> about 1,046 blocks), recorded.
- **W windows on B (BMAX) and S16B2W** are capped by the boot's growth-aware resident bound, so the "W" levels are what each pool holds for a
  90-s window after a long admission (predicted B 20 / 13 / 11, S16B2W 29 / 22 / 14 / 12 at 20k / 40k / 90k / 116k). BMAX is my addition
  (switch key, default on, about +26 min) so that S16B2W's max-resident tok/s is compared with B's max-resident tok/s, not with B's 16 / 8 / 6
  windows. Set `BMAX: False` to drop it.
- **TF noise:** the klee100 TF A/A is not deterministic (finding 7); with 256 positions per prompt, small bf16 effects may sit inside B's
  A/A envelope. The long traces are the most sensitive test of SSM drift (69k-113k recurrent steps).
- **The oracle 40k row (PROMPTS["40k"][3]) is also a 40k@16 window prompt** (inherited from NBDF-K100): one window stream gets a prefix hit
  on every arm alike; recorded, not changed.
- **Timing:** one model; TF, the W / BMAX windows and the recompile times for a block / dtype change are estimates, not GPU data. FULL has
  7 min between the planned runtime and the hard stop (245); a slower arm makes the next one SKIPPED_BUDGET (no cross-arm skip otherwise).
- **Real kit size:** 394,708 B (5.3 kB margin). Any growth of the body or the export needs the margin checked again.
- **The W read is 2-decimal:** the log's mc 13.32 gives F 18 / W 28 instead of the planner's 17 / 29; both are valid plans.
- **Kit prerequisite:** the TF dataset must be uploaded before the real kit, else `e34_tf_long_ok` is False and the gate stays closed.

## 10. Addendum (2026-11-03, rev 2): the competition mount layout

- **Symptom:** the real kit failed 3 times at R1 with `FileNotFoundError: /kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json`.
  That kernel had the legacy mount `/kaggle/input/arc-prize-2026-arc-agi-2` and no `competitions/` dir.
- **Cause:** CELL 1's `COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"` (export line 161, inherited from NB-D1 / NBDF / E29). CELL 0
  (E18 R2, export lines 18-22) already falls back to `/kaggle/input/arc-prize-2026-arc-agi-2`, and it stays verbatim.
- **Fix:** one exact-once CELL 1 edit in `build_e34.py` (`c1 comp layouts`). COMP is now the first existing of `INP / "competitions" / ...`
  and `INP / "arc-prize-2026-arc-agi-2"`; when neither exists, the old path is the default.
- **Other hard-coded copies:** none. The body, the units kit and the real kit (HEAD + REAL + the packed items) have no `competitions/`
  path; the kits read COMP through the export they run. COMP's only use is `ARC = COMP / "arc-agi_evaluation_challenges.json"`
  (export line 226).
- **Unit test:** the COMP statement from the export runs against temp dirs for both layouts and for neither (the default). The test also
  checks that the old bare statement is gone.
- **Results:** 180/180 PASS on Py3.14 and Py3.12; units-kit simulation 178/178 `@@KIT units ALL OK@@`.
- **New shas:**
  - FULL be5fcd583bd6842fd09a5788eb39fdd831462f023b6e13eea2788006ae9ddcaf (348,088 B);
  - SMOKE c0ddc0b31dcf2a79ef1c7a4f79ff466cef857258a0c7102b48bbeeacd9ee305c (348,090 B);
  - units kit 6c8a38f85effa010ce37d9b29630261bfde9a1f8bb1c15ca5d7a6a0de4f7b15f (381,729 B);
  - real kit 100f3d2402815eddc1650163aae6091f9e24ad3958a72d9e27f79a88374cdd7d (394,708 B);
  - export eeab94bb.
- **Unchanged:** the body e34_cell7_body.py 65cc785e (it is not in CELL 1), so the payload dataset folder is unchanged. Both kits are
  rebound to be5fcd58. The push folders are rebuilt; nothing was pushed.
- **The same hard-coded path elsewhere (not edited).** Each line below uses `ARC = COMP / "arc-agi_evaluation_challenges.json"`, so a
  legacy-mount kernel fails the same way:
  - E29 `Build/e29_dflash_d3s_code_export.py:1097` (used at :2512); `Build/e29_default_code_export.py:1072`; `Build/e29_mtp_v64_code_export.py:1072`;
    `Build/all_off_code_export.py:1072`.
  - E32 `Build/e32_levers_code_export.py:1105` (used at :2577).
  - E33 `Build/e33_chain_code_export.py:1106` (used at :2527); also the draft copy `Build/_kit_local_e33/draft/export.py:1104`.
  - Each of these also has CELL 0's line 18 `ARC_ROOT` default, which already has the legacy fallback at lines 19-22.
