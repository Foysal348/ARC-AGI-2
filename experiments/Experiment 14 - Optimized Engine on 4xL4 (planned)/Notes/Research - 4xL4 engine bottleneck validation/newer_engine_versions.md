# Newer engine versions (SGLang after 0.5.21; vLLM 0.30 / 0.31 / 0.32 / nightly) for Qwen3.8-27B on Kaggle 4x L4: what they fix, and whether we can install them offline

Current as of 2026-10-23.

**How the status labels were checked.**
- Each PR or issue state was read from the GitHub API on 2026-10-23.
- "In vX" means the merge commit is an ancestor of that release tag. This was checked with the GitHub compare API (`compare/<tag>...<merge_sha>` returned `behind`).
- "Main-only" means the commit is not in any release tag (compare returned `diverged`).
- Labels used below:
  - **MERGED-in-vX**: merged and shipped in release vX.
  - **MERGED-main-only**: merged, but in no release yet.
  - **OPEN**: an open PR or issue.
  - **CLOSED-unmerged**: closed without merging.
  - **CLAIMED**: a statement by an issue author that nobody verified.

**Release timeline**
- **SGLang:**
  - v0.5.21: 2026-10-02.
  - v0.5.22: 2026-10-19 (GitHub), 2026-10-16 (PyPI upload).
  - No v0.5.23 release yet. Nightly `0.5.23.devYYYYMMDD` wheels exist up to 2026-10-22.
- **vLLM:**
  - v0.30.0: 2026-09-22.
  - v0.31.0: 2026-10-05.
  - **v0.32.0: 2026-10-18.** This version is newer than the ones named in the brief.
  - Nightly is `0.32.1rc1.dev328+g2de0531d8` (2026-10-22).
- Sources: [SGLang releases](https://github.com/sgl-project/sglang/releases), [SGLang PyPI history](https://pypi.org/project/sglang/#history), [vLLM releases](https://github.com/vllm-project/vllm/releases), [vLLM nightly index](https://wheels.vllm.ai/nightly/vllm/), [SGLang nightly cu130 index](https://github.com/sgl-project/whl/blob/main/cu130/sglang/index.html).

---

## Q1. SGLang releases after v0.5.21: which entries touch our issues?

### Takeaway
SGLang has one release after 0.5.21: v0.5.22 (2026-10-19, 950 PRs).

What v0.5.22 changes for us:
- **Not in v0.5.22:** the real fix for the 11-of-12 admission bug, PR #44401. It merged on 2026-10-15, after the release branch was cut, so it exists only in nightly ≥ 2026-10-16. v0.5.22 does carry the related empty-batch latch fix, #41872.
- **Still open:** every Qwen3.8-27B DFlash2 correctness bug we track (#36548, #38009, #43514/#43518, #37475, #39087).
- **Still open:** the issues on dynamic verification width (#36127) and on the mamba-cache concurrency cap (#36889, #38846). ReplaySSM for DFlash on GDN is not available.
- **The biggest concrete gain is indirect.** SGLang 0.5.21 already contains an opt-in FlashInfer PCIe-IPC all-reduce for NVLink-less boxes (`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE`). It needs FlashInfer ≥ 0.7.0, and 0.5.21 pins 0.6.18, so the backend is dead code in our current install. v0.5.22 pins FlashInfer 0.7.1, which makes it usable.

### Cited Findings

**Release v0.5.22 headline and fixes**
- v0.5.22 contains "950 PRs from 245 contributors" and now requires Python ≥ 3.11 — [v0.5.22 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).
- The "important fix" it highlights: "Hybrid linear models could resume from wrong mamba states when speculative decoding and structured output were used together" ([#43029](https://github.com/sgl-project/sglang/pull/43029), "Use committed length in spec mamba track boundary check").
  - MERGED-in-v0.5.22, on 2026-10-10 — [release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22); [PR #43029](https://github.com/sgl-project/sglang/pull/43029).
- None of the release-note headline numbers are for Qwen3.8-27B or Ada. They are for:
  - DeepSeek-V4.1 on H200;
  - Qwen3.8-Flash-Next decode +12% on B200 (#42175: 242 → 271 tok/s on 4x B200 at concurrency 1);
  - GLM-5.3-Flash DFlash +10% on B300;
  - the InstantTensor loader, "10x faster";
  - [v0.5.22 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).

**1. The 11-of-12 admission bug (chunked-prefill row double-count, `batch_is_full` latch)**
- **PR [#44401](https://github.com/sgl-project/sglang/pull/44401)**, "Count a request's held req_to_token row once at admission". MERGED-main-only (2026-10-15T05:29Z). It is not in v0.5.22.
  - Compare of the merge commit `f9bdb74a` against v0.5.22 returns `diverged`. "#44401" does not appear anywhere in the v0.5.22 notes.
  - Its description: a continuing chunked-prefill request "is in `can_run_list`, and its row is already excluded from `req_to_token_pool.available_size()`, so the gate … counted it twice and declared the batch full with a row still free. Under chunked prefill this capped the running batch at `max_running_requests - 1`, and when the final chunk's pass set `batch_is_full`, nothing cleared it until a request finished, so the waiting queue starved meanwhile."
  - It fixes [#38118](https://github.com/sgl-project/sglang/issues/38118) and [#35537](https://github.com/sgl-project/sglang/issues/35537) (both closed) and supersedes [#30295](https://github.com/sgl-project/sglang/pull/30295) (CLOSED-unmerged).
  - The diff is small: `managers/scheduler.py` +14/−6, the dLLM mixin +4/−1, and a 136-line CPU unit test — [PR #44401 files](https://github.com/sgl-project/sglang/pull/44401/files).
- **The nightly carries #44401.**
  - Nightly `0.5.23.dev20261016+gf9df33494a` and later contain it (compare: `ahead`). `0.5.22.dev20261015+ga12814c158` does not (`behind`).
  - Sources: [nightly index](https://github.com/sgl-project/whl/blob/main/cu130/sglang/index.html); [compare API](https://github.com/sgl-project/sglang/compare/f9bdb74a5ae9e0ae6cf37237d54af0086c2d55c9...b453996b7d).
- **PR [#41872](https://github.com/sgl-project/sglang/pull/41872)**, "stop latching batch_is_full when nothing can run". MERGED-in-v0.5.22 (2026-10-12).
  - Scope: it fixes only the empty-batch case. The NO_TOKEN path set `batch_is_full = True` on an empty running batch, and "every later `_get_new_batch_prefill_raw` call bailed". Its own text says "Behavior only changes in the empty + nothing-admitted case".
  - So it does not fix our non-empty 11-of-12 case. #44401 is the fix for that — [PR #41872](https://github.com/sgl-project/sglang/pull/41872).
- **Related scheduler fixes in v0.5.22 (all MERGED-in-v0.5.22)** — [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22):
  - [#43298](https://github.com/sgl-project/sglang/pull/43298) "Requeue preempted requests when admission admits nothing";
  - [#43290](https://github.com/sgl-project/sglang/pull/43290) "Hold back decode rows on a mixed step when they do not fit the KV pool";
  - [#43564](https://github.com/sgl-project/sglang/pull/43564) "Reserve remaining tokens for ignore_eos requests during prefill admission".
- **Local 0.5.21 source check (our `sgl_src`):**
  - `get_num_allocatable_reqs()` and the admission gate in `_get_new_batch_prefill_raw` have exactly the pre-patch shape of #44401: `available = self.req_to_token_pool.available_size()` and `if len(adder.can_run_list) >= self.get_num_allocatable_reqs(...)`.
  - `Req.kv.holds_kv` already exists as a property (`schedule_batch.py:968-970`).
  - The #41872 read-site clear ("An empty batch … would stay shut") is absent.
  - Sources: local `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/srt/managers/scheduler.py:3784-3800, 3962-3980`; [PR #44401 diff](https://github.com/sgl-project/sglang/pull/44401/files).

**2. DFlash / DFlash2 correctness on Qwen3.8-27B: all core bugs still open**
- **[#36548](https://github.com/sgl-project/sglang/issues/36548)**, "DFlash2 state corruption with concurrent requests" — OPEN.
  - On GB10 a commenter measured, at greedy temp 0:
    - DFlash8 serial: 0/100 wrong;
    - DFlash8 at concurrency 8: 7/100 wrong (111/304, 36.5%, on affected prompts);
    - no speculation at concurrency 8: 0/304 wrong.
  - Ablations (fp32 SSM, `--disable-decode-cuda-graph`, depth 4) each cut errors to 37/304 but did not remove them.
  - The commenter localizes it to "the selected-step SSM transition output" (DECODE vs TARGET_VERIFY kernel difference, amplified by batch composition). A dense BF16 lm_head cut errors to 1/304, "not a fix".
  - CLAIMED, by one commenter on a different quantization (NVFP4).
- **[#38009](https://github.com/sgl-project/sglang/issues/38009)**, "[DFlash2] Greedy output diverges from target-only Qwen3.8-27B when thinking is enabled" — OPEN.
  - A follow-up shows DFlash2 thinking output is deterministic within one process but differs across server restarts. Accept length swings 2.5–3.5 vs 4.4–6.2 per boot. The commenter recommends A/A runs per profile — [#38009](https://github.com/sgl-project/sglang/issues/38009).
- **[#43514](https://github.com/sgl-project/sglang/issues/43514)**, "DFlash + Qwen3.8-27B: accept rate drops to 0 during serving" — OPEN.
  - Trigger: grammar-constrained requests (tool_choice=required) plus multi-chunk admissions. "Both ingredients are required." v0.5.19 and v0.5.20 are affected.
  - The candidate fix **[#43518](https://github.com/sgl-project/sglang/pull/43518)** ("Clear stale grammar masks before DFlash and DSpark verification") is OPEN. As of 2026-10-13 it was waiting for a maintainer to add CI labels — [#43518](https://github.com/sgl-project/sglang/pull/43518).
- **[#37475](https://github.com/sgl-project/sglang/issues/37475)**, "DFlash target_layer_ids fallback guesses wrong tap layers; count collision fails silently" — OPEN, no comments, no linked fix — [#37475](https://github.com/sgl-project/sglang/issues/37475).
- **[#28180](https://github.com/sgl-project/sglang/issues/28180)**, DFlash infinite loop with repetition_penalty — CLOSED on 2026-08-17. This predates 0.5.21, so 0.5.21 already has it.
  - Follow-ups to apply penalties properly under DFLASH are still OPEN: [#41738](https://github.com/sgl-project/sglang/pull/41738) "Feed committed tokens to the penalizers" and [#43207](https://github.com/sgl-project/sglang/pull/43207) "Apply sampling penalties under DFLASH and DSPARK" — [#28180 timeline](https://github.com/sgl-project/sglang/issues/28180).
- **[#39087](https://github.com/sgl-project/sglang/issues/39087)**, a quantized DFlash2 draft silently gives about 0% acceptance — still OPEN.
  - Keep `--speculative-draft-model-quantization unquant` in any version — [#39087](https://github.com/sgl-project/sglang/issues/39087).
- **DFlash robustness fixes MERGED-in-v0.5.22** — [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22):
  - [#41736](https://github.com/sgl-project/sglang/pull/41736) "Keep grouped-conv taps inside each request's block so NaN/Inf cannot leak across requests" (merged 2026-10-01). This is a cross-request leak fix, but only for NaN/Inf.
  - [#41737](https://github.com/sgl-project/sglang/pull/41737) "Keep the greedy selector walk in range for all-NaN score rows".
  - [#42709](https://github.com/sgl-project/sglang/pull/42709) "Fix wrong tokens from compact-verify CUDA graphs when a token tier is captured more than once".
  - [#42002](https://github.com/sgl-project/sglang/pull/42002) "Fix Qwen3.5 text model EAGLE3/DFLASH aux-layer capture".
  - None of these is linked to #36548 or #38009.
- **Mamba-slot-related correctness, still open:**
  - [#36885](https://github.com/sgl-project/sglang/pull/36885), the KDA sentinel remap fix, was MERGED on 2026-08-28. It applies to KDA models, not GDN.
  - [#35150](https://github.com/sgl-project/sglang/issues/35150), "Qwen3.8 DSpark forced-reject is not lossless: accumulated GDN state drift", is OPEN.
  - [#36014](https://github.com/sgl-project/sglang/pull/36014), "Align GDN target-verify beta semantics with packed decode", is OPEN.
  - Sources: [#36885](https://github.com/sgl-project/sglang/pull/36885), [#35150](https://github.com/sgl-project/sglang/issues/35150), [#36014](https://github.com/sgl-project/sglang/pull/36014).
- **Quality bug [#42917](https://github.com/sgl-project/sglang/issues/42917)** (INT4 W4A16 PPL 9.98 vs 6.05) was resolved as environmental. The reporter found a stale CUDA *driver* compute cache in `$SGLANG_CACHE_DIR/nv`. With only that directory emptied, PPL was 6.2784, matching vLLM; the reporter offered to close the issue.
  - The issue is still formally open. The lesson for any version: use a fresh `SGLANG_CACHE_DIR` / `CUDA_CACHE_PATH` — [#42917](https://github.com/sgl-project/sglang/issues/42917).

**3. Dynamic verification width (#36127) and new verification modes**
- **[#36127](https://github.com/sgl-project/sglang/issues/36127)**, "[Feature] Dynamic verification for DFlash2" — OPEN, no comments — [#36127](https://github.com/sgl-project/sglang/issues/36127).
- **New in v0.5.22: opt-in block verification.**
  - [#42297](https://github.com/sgl-project/sglang/pull/42297) adds `--speculative-use-block-verification` (arXiv 2403.10444, Algorithm 2) for EAGLE/NEXTN.
  - [#43012](https://github.com/sgl-project/sglang/pull/43012) extends it to DFLASH/DSPARK, but only on the sampled-draft path.
  - The #43012 text says greedy DFlash drafts "propose argmax tokens", so block verification applies to the candidate-selector or LiLiCorr sampled drafts.
  - It is a different acceptance rule, not a dynamic width. Both PRs are MERGED-in-v0.5.22 — [#43012](https://github.com/sgl-project/sglang/pull/43012), [#42297](https://github.com/sgl-project/sglang/pull/42297).
- **Open bug:** with block verification under PP, [#45162](https://github.com/sgl-project/sglang/issues/45162) reports "PP + EAGLE/MTP: --speculative-use-block-verification … crash on first sampled" (OPEN, 2026-10-19). We use no PP — [#45162](https://github.com/sgl-project/sglang/issues/45162).

**4. Mamba / hybrid radix cache, extra_buffer_lazy, mamba cache sizing**
- **[#36889](https://github.com/sgl-project/sglang/issues/36889)** (DFLASH mamba cache caps concurrency at 5 state slots per request) — OPEN.
- **[#38846](https://github.com/sgl-project/sglang/issues/38846)** (expose effective `max_running_requests`) — OPEN.
  - A commenter verified on 0.5.20 that the effective value is already published as `internal_states[0].effective_max_running_requests_per_dp` in `/get_server_info`.
  - The cap log line is now `logger.warning`.
  - Measured ratio: 5 slots/request for `extra_buffer` with overlap; 3 for `no_buffer` without overlap; 1 for `--disable-radix-cache` — [#38846 comments](https://github.com/sgl-project/sglang/issues/38846).
- **Source check:** v0.5.22 still has `MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3`, plus extra-buffer tracking slots from `get_mamba_tracking_slots(extra_buffer, overlap, lazy)`. So the per-request slot cost is unchanged in principle.
  - Sources: [kv_cache_configurator.py@v0.5.22 L222, L2743-2768](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/mem_cache/kv_cache_configurator.py); local 0.5.21 `kv_cache_configurator.py:179-184`.
- **Open fixes in this area:**
  - [#44372](https://github.com/sgl-project/sglang/pull/44372) "Fit automatic hybrid speculative state caches to available memory": with spec decoding and no `--max-running-requests`, the default becomes 48 and can consume the KV budget.
  - [#43484](https://github.com/sgl-project/sglang/pull/43484) "Mamba slot admission for the classic hybrid pool": `AssertionError: Can not alloc mamba cache` with two concurrent long requests on a 27B GDN model at `--max-mamba-cache-size 10`.
  - Sources: [#44372](https://github.com/sgl-project/sglang/pull/44372), [#43484](https://github.com/sgl-project/sglang/pull/43484).
- **Hybrid-cache changes MERGED-in-v0.5.22:**
  - [#36266](https://github.com/sgl-project/sglang/pull/36266) "Warm cache COW kernel before serving (first-wave cold excess 924 to 28 ms)";
  - [#42354](https://github.com/sgl-project/sglang/pull/42354) "Run mamba models on `UnifiedRadixCache` when the radix cache is disabled"; `ChunkCache` is removed ([#43201](https://github.com/sgl-project/sglang/pull/43201));
  - [#43567](https://github.com/sgl-project/sglang/pull/43567) "Invalidate lazy Mamba checkpoints before in-place tracking";
  - [#42467](https://github.com/sgl-project/sglang/pull/42467) "Keep prefix_indices current for chunked requests kept out of the tree";
  - [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).
- **After 0.5.22 (MERGED-main-only):**
  - [#45435](https://github.com/sgl-project/sglang/pull/45435) "Keep prefill rows' checkpoint tracking through mixed chunked prefill" (2026-10-21);
  - [#45472](https://github.com/sgl-project/sglang/pull/45472) "Plan hybrid Mamba extend batches without host-device syncs" (2026-10-21);
  - [#45435](https://github.com/sgl-project/sglang/pull/45435), [#45472](https://github.com/sgl-project/sglang/pull/45472).
- **ReplaySSM with DFlash:** unavailable for GDN models in every version.
  - In 0.5.21, v0.5.22 and nightly `b453996b7d`, the configurator raises "`--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA`".
  - The PRs that would extend it to GDN are OPEN: [#36683](https://github.com/sgl-project/sglang/pull/36683) "Support ReplaySSM spec-verify for DFlash and DSPARK on hybrid GDN models" and [#42209](https://github.com/sgl-project/sglang/pull/42209) "Exact snapshot-free GDN verify for DFLASH and NEXTN".
  - #42209 notes that the existing compact replay is not bit-exact: "79 of 80 requests diverged from stock in tokens".
  - Sources: [kv_cache_configurator.py@v0.5.22 L1555](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/mem_cache/kv_cache_configurator.py); local 0.5.21 `kv_cache_configurator.py:1134`; [#36683](https://github.com/sgl-project/sglang/pull/36683); [#42209](https://github.com/sgl-project/sglang/pull/42209).
- **Draft-KV pool sizing:** [#42651](https://github.com/sgl-project/sglang/pull/42651) "Back the DFlash-family draft KV pool to the post-capture token count" is MERGED-in-v0.5.22.
  - The bug: under `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`, the DFlash draft pool "was allocated eagerly at that upper bound … so the target pool ended up smaller than the GPU allowed".
  - The env var already exists in 0.5.21 (default False; local `environ.py:590`), so this fix only matters if that opt-in is used — [#42651](https://github.com/sgl-project/sglang/pull/42651).

**5. Spec V2 overlap on FlashInfer/Triton**
- I found no v0.5.22 entry that changes Spec V2 overlap for FlashInfer or Triton attention on Qwen3.5/3.8.
- Merged related items:
  - [#34076](https://github.com/sgl-project/sglang/pull/34076) "overlap scheduler: record_stream mix_running_indices for the forward-stream relay gather";
  - [#41154](https://github.com/sgl-project/sglang/pull/41154) "enable Qwen3.5 EAGLE3 capture and streaming overlap";
  - [#43037](https://github.com/sgl-project/sglang/pull/43037) "Allow Qwen3.5 backends to opt into packed target verification" (all MERGED-in-v0.5.22);
  - the remaining Spec V2 PRs are pipeline-parallel signature fixes;
  - [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22); [PR search](https://github.com/sgl-project/sglang/pulls?q=spec+v2).

**6. sm89 / Ada kernels (Marlin, Triton GDN, FlashInfer + fp8 KV)**
- I found no v0.5.22 entry that targets SM89 for dense W4A16 Marlin, Triton GDN or FlashInfer attention — [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).
- **New fused kernel in v0.5.22:** [#40785](https://github.com/sgl-project/sglang/pull/40785) "Fuse FP8 KV-cache quantization into the prefix-valid commit kernel". It is on the DFLASH path.
  - The OPEN PR [#45488](https://github.com/sgl-project/sglang/pull/45488) explains that this kernel compiles `fp8e4nv` in Triton, "which only works on SM89+ (Ada / Hopper)", and crashes SM86 at startup. L4 is SM89, so it should take the fused path.
  - This is inference; nobody has measured the kernel on L4 — [#45488](https://github.com/sgl-project/sglang/pull/45488).
- **GDN-related merged items:**
  - [#43570](https://github.com/sgl-project/sglang/pull/43570) "Compact GDN causal-conv grids for skewed prefill batches";
  - [#43227](https://github.com/sgl-project/sglang/pull/43227) "Avoid gate copies in gated RMSNorm";
  - [#43332](https://github.com/sgl-project/sglang/pull/43332) "Fix CuTe DSL decode to read the [N, HV, V, K] state layout";
  - [#41486](https://github.com/sgl-project/sglang/pull/41486) "Tune SM90 GDN recurrent verify launch", which is SM90 only;
  - [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).
- **New SM86 Marlin bug reports after v0.5.22** (all OPEN, 2026-10-18; SM86, not SM89):
  - [#45000](https://github.com/sgl-project/sglang/issues/45000) FP8 Marlin with the overlap scheduler;
  - [#44999](https://github.com/sgl-project/sglang/issues/44999) Marlin MoE AWQ;
  - [#44998](https://github.com/sgl-project/sglang/issues/44998) NVFP4 Marlin fallback on SM80-SM90.

**7. Custom all-reduce / new all-reduce backends on PCIe**
- **The stock custom all-reduce gate is unchanged.**
  - v0.5.22 `custom_all_reduce_utils.py:492` still has `if world_size > 2 and not full_nvlink:`, so TP4 on PCIe still disables the standard custom all-reduce.
  - Sources: [custom_all_reduce_utils.py@v0.5.22](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py); local 0.5.21, same line.
- **A new backend already exists in 0.5.21:** [#34528](https://github.com/sgl-project/sglang/commit/c7be3e935b) "[SM120] Add optional FlashInfer PCIe-IPC all-reduce for switch-free hosts", committed 2026-09-29. `pcie_ipc_ar.py` has the same size (17,248 bytes) at v0.5.21 and v0.5.22.
  - Enablement: opt-in via `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` (EnvBool, default False). There is also `SGLANG_PCIE_IPC_MAX_NUMEL`.
  - Limits: world sizes (2, 4, 8) only; the `tp` group only; **bf16 only**; disabled under deterministic inference.
  - Workspace: sized for decode, as `cuda_graph max_bs × speculative_num_draft_tokens × hidden`, so prefill reductions stay on NCCL.
  - On import failure it logs "this FlashInfer build has no pcie_ipc_comm; falling back to NCCL all-reduce."
  - Sources: [pcie_ipc_ar.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py); local `environ.py:1354-1355`, `parallel_state.py:489-510, 973-978`.
- **Numbers in the module docstring** (8x SM120, 2 NUMA nodes, hidden 6144 bf16), FlashInfer vs NCCL — [pcie_ipc_ar.py docstring](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py):
  - 12 KiB: 5.3 µs vs 205.1 µs (38.8x);
  - 96 KiB: 15.3 µs vs 584.9 µs;
  - 384 KiB: 34.6 vs 1169.3 µs;
  - at TP8 with 8k context, a decode-sized workspace cut TPOT 21.14 → 13.62 ms and raised output from 35.02 to 48.05 tok/s, with TTFT level.
- **The FlashInfer kernel is only in 0.7.0 and later.**
  - The kernel is FlashInfer PR [#4393](https://github.com/flashinfer-ai/flashinfer/pull/4393) ("custom all reduce kernel", merged 2026-08-20).
  - `flashinfer/comm/pcie_ipc_ar.py` is **absent at tags v0.6.18 and v0.6.18.post1**, and present at v0.7.0 (44,053 B) and v0.7.1 (49,719 B).
  - So with our 0.5.21 wheelhouse (flashinfer 0.6.18) the SGLang backend can never activate. With v0.5.22 (flashinfer 0.7.1) it can.
  - Sources: [FlashInfer #4393](https://github.com/flashinfer-ai/flashinfer/pull/4393); GitHub contents API on the tags.
- **The kernel has no architecture restriction.** `gen_pcie_ipc_comm_module()` says: "No architecture restriction: the kernels use only plain PTX loads/stores and CUDA IPC … orthogonal to the SM version." An optional "memory-operation protocol requires SM120" — [flashinfer/jit/comm.py@v0.7.1](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/flashinfer/jit/comm.py); [csrc/pcie_ipc_all_reduce.cu@v0.7.1](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/csrc/pcie_ipc_all_reduce.cu).
- **Open SGLang PCIe item:** [#42863](https://github.com/sgl-project/sglang/pull/42863) "[SM120] all-gather: Speed up TP4 logits and DSpark vocab all-gathers on PCIe-only hosts" — OPEN.

**8. HiCache for hybrid models**
- **MERGED-in-v0.5.22** — [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22):
  - [#36345](https://github.com/sgl-project/sglang/pull/36345) "Support Mamba in buffer-only host memory mode";
  - [#40223](https://github.com/sgl-project/sglang/pull/40223) "Preserve all MTP KV and recurrent state";
  - [#43596](https://github.com/sgl-project/sglang/pull/43596) "Build Mamba host pools from device buffer infos";
  - [#40915](https://github.com/sgl-project/sglang/pull/40915) "Declare hybrid Mamba indexer host pool".
- **New OPEN bug:** [#45400](https://github.com/sgl-project/sglang/issues/45400) "HiCache on hybrid GDN model: host-tier hits drop to zero once evicted working set exceeds ~3 long prefixes" (2026-10-20).

**9. Weight-loader prefetch and startup / CUDA-graph capture**
- [#40453](https://github.com/sgl-project/sglang/pull/40453) "Add InstantTensor loader support". MERGED-in-v0.5.22. "DeepSeek-R1 on 8x H200 loads in 15.3 s instead of 160 s from 50 GB/s storage". New optional extra `sglang[instanttensor]` (instanttensor ≥ 0.1.9) — [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).
- Startup-related, all MERGED-in-v0.5.22 — [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22):
  - [#43172](https://github.com/sgl-project/sglang/pull/43172) "Clamp default decode CUDA graph max_bs to max_running_requests" (fewer graphs to capture);
  - [#43177](https://github.com/sgl-project/sglang/pull/43177) "Overlap scheduler startup with the data parallel controller";
  - [#43461](https://github.com/sgl-project/sglang/pull/43461) "Allow mixed chunk prefill on the Full prefill CUDA graph backend".
- **Breaking:** the `tc_piecewise` prefill CUDA-graph backend is removed; `breakable` stays the CUDA default — [#42287](https://github.com/sgl-project/sglang/pull/42287), via the [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).

### Inferences
- **v0.5.22 alone does not fix our 11-of-12 cap.** Only nightly ≥ 2026-10-16, or a backport, does.
  - The #44401 diff applies to the 0.5.21 code shape almost verbatim: the same `get_num_allocatable_reqs` signature and the same gate lines, and `req.kv.holds_kv` already exists.
  - A local monkey-patch of 0.5.21 is a small, low-risk alternative to an engine upgrade. To fully match upstream it should also include #41872's empty-batch read-site clear.
  - I did not execute this backport.
- **No newer SGLang version removes the DFlash2-on-Qwen3.8-27B correctness risks** (#36548 concurrency corruption, #38009 thinking divergence). Any upgrade still needs the greedy, concurrency and A/A checks in our plan. The NaN-leak fix #41736 is the only merged cross-request-isolation fix, and it is narrower than #36548.
- **The PCIe-IPC all-reduce is the clearest concrete upgrade justification for TP4 on 4x L4.**
  - We measured TP4 NCCL at about 3 GB/s algorithm bandwidth, and the per-layer decode all-reduces are small (12–400 KiB), which is the regime where the docstring reports 34–39x over NCCL on SM120.
  - Whether L4/PCIe on Kaggle gets a similar gain is unmeasured.
  - It requires working CUDA IPC peer access between the L4s, which is unknown for Kaggle (see `nccl_p2p_l4.md`).
- Block verification (v0.5.22) is not the dynamic width requested in #36127, and per #43012 it does not apply to greedy DFlash drafts. It is a low-priority experiment for our temp-1.0 sampling.

### Gaps
- No SGLang release after v0.5.22 exists yet, so I cannot list 0.5.23 changelog items. Nightly commits after 2026-10-19 were sampled only through targeted PR searches.
- I could not verify whether Kaggle's 4x L4 supports CUDA IPC / P2P, which the PCIe-IPC kernel needs. No source covers GCP or Kaggle L4 P2P.
- I found no source that measured the SGLang PCIe-IPC backend on Ada or at TP4. All published numbers are for 8x SM120.
- I did not diff the full 0.5.21 vs 0.5.22 GDN/Marlin kernel sources for silent SM89 performance changes.

---

## Q2. vLLM 0.30 / 0.31 / 0.32 / nightly for Qwen3.5/3.8 hybrids

### Takeaway
- **DFlash2 has been in vLLM since 0.29** (#52816).
- **v0.30** adds the opt-in FlashInfer PCIe-IPC all-reduce (#53576). It is unusable in 0.30 because that version pins FlashInfer 0.6.18.post1.
- **v0.31** adds async scheduling for DFlash (#58065). It moves to FlashInfer 0.7.0.post1, which carries the PCIe kernel.
- **v0.32** adds the DFlash2 prefix-hit mamba-state-index fix (#55601) and the hybrid prefix-cache boundary fixes.
- **Still open:**
  - the MTP + prefix-cache corruption on hybrids (#53912);
  - spec-decode prefix-hit loss (#54360);
  - the MTP concurrency window (#55533, really a mamba-block budget);
  - the FP8 draft-KV crash on SM89 (#54690).
- The thinking-budget + spec-decode fix in Model Runner V2 (#58231) is in nightly only.
- The "block-FP8 Triton tuning" PR #59232 is for **L20**, not L4, and is open.

### Cited Findings

**Release scope**
- **v0.30.0** (762 commits) — [v0.30.0 notes](https://github.com/vllm-project/vllm/releases/tag/v0.30.0):
  - "an opt-in FlashInfer PCIe IPC all-reduce for NVLink-less boxes (#53576)";
  - Model Runner V2 "adaptive verification for every draft-model speculator through an online acceptance estimator (#52228)";
  - "gc frozen during graph capture, cutting capture from 12s to 2s and engine init from 28.9s to 8.2s on H200 (#54646)";
  - "Qwen3.5/Qwen3-Next Triton kernels (#54797) and GDN gated RMSNorm (#54251) warmed at startup";
  - "Mamba state cached at the EAGLE resume position (#53945) and preserved for padded prompt tails (#55178)";
  - "FlashInfer ReplaySSM backend for Mamba2 (#52506)";
  - the `all` Mamba cache mode deprecated.
- **v0.31.0** (717 commits) — [v0.31.0 notes](https://github.com/vllm-project/vllm/releases/tag/v0.31.0):
  - "async scheduling for DFlash (#58065) with the context K/V precompute captured in the draft CUDA graph (#57632)";
  - `--max-num-active-seqs` (#56758);
  - "the waiting queue was reworked so requests already holding KV blocks are scheduled first (#58947)";
  - "a throughput cliff when `max_num_seqs` is not a multiple of 8 (#57355)";
  - "prompt-tail hits with MTP restored (#58368)";
  - "fused DFlash2 grouped convolution (#55960)";
  - "CuteDSL BF16 GDN prefill diverging from FLA on Qwen3.5 (#53864)" fixed;
  - FlashInfer 0.7.0.post1;
  - `--enable-mamba-fine-grained-prefix-cache` renamed to `--enable-mamba-shared-prefix-checkpoint`.
- **v0.32.0** (814 commits, 2026-10-18) — [v0.32.0 notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0):
  - "NVFP4 KV cache now runs on SM8x and SM12x (#46963)";
  - "in-graph metadata for DFlash/DSpark drafts (#60692)";
  - "correctness fixes for ngram on hybrid recurrent models (#56531), Mamba prefill checkpoints with MTP and prefix caching (#59759, #60797)";
  - "prefix-cache bounds (#55601, #60533)";
  - "custom all-reduce under `VLLM_BATCH_INVARIANT=1` (#58623)";
  - "Qwen3-VL/Qwen3.5/Qwen3.8 M-RoPE cache 4x smaller";
  - Python 3.10 dropped;
  - `mamba_cache_mode="all"` removed;
  - `max_tokens` is "an upper bound instead of a reservation" (#57035).

**Specific items**
- **DFlash2 PR [#52816](https://github.com/vllm-project/vllm/pull/52816)** "DFlash2: local convolution + candidate selector" — MERGED-in-v0.29.0 (2026-08-21).
- **[#55601](https://github.com/vllm-project/vllm/pull/55601)** "Seed hybrid mamba state index with mamba_block_size on prefix-cache hits" — MERGED-in-v0.32.0 (2026-10-07; not in 0.31.0).
  - HyperQwen describes the bug it fixes (vLLM #58894): "DFlash2 acceptance drops to 0% for good after the first prefix-cache hit (Orin), or an illegal memory access on a cache hit (RTX 6000 Ada)."
  - The fix divides by `cache_config.mamba_block_size` instead of `cache_config.block_size` (+5/−1).
  - Issue [#58894](https://github.com/vllm-project/vllm/issues/58894) itself is still OPEN.
  - Sources: [HyperQwen #344](https://github.com/syv-ai/HyperQwen/issues/344), [#58894](https://github.com/vllm-project/vllm/issues/58894).
- **MTP + prefix caching corruption:**
  - [#43559](https://github.com/vllm-project/vllm/issues/43559) was CLOSED on 2026-08-06 (after [#51113](https://github.com/vllm-project/vllm/pull/51113) merged).
  - The follow-up [#53912](https://github.com/vllm-project/vllm/issues/53912) "prefix caching + MTP still corrupts output on hybrid Mamba/GDN models in v0.28.0 (#43559 closed but unfixed)" is OPEN, with 26 comments.
  - Candidate fixes still OPEN:
    - [#48375](https://github.com/vllm-project/vllm/pull/48375) "Honor drop_eagle_block in MambaManager";
    - [#57128](https://github.com/vllm-project/vllm/pull/57128);
    - [#43650](https://github.com/vllm-project/vllm/pull/43650);
    - [#53919](https://github.com/vllm-project/vllm/pull/53919) "Await the accepted-token copy before moving batch rows".
  - [#50729](https://github.com/vllm-project/vllm/pull/50729) "Fix overlapping state copy race" is MERGED (2026-08-17).
  - The latest report (2026-10-14): 0.28.0 + #48375 alone still garbled; it was serving 0.28.0 + #48375 + #50729 under observation.
  - Several comments in that thread are self-declared AI-agent analyses with retractions, so treat the mechanisms as CLAIMED — [#53912](https://github.com/vllm-project/vllm/issues/53912).
- **[#54360](https://github.com/vllm-project/vllm/issues/54360)** "Speculative decoding (mtp and dflash) silently disables prefix-cache hits for hybrid GDN models" — OPEN.
  - On the v0.31.0 wheel with DFlash, sibling prompts alternate between a hit (3.2 s TTFT) and a full re-prefill (about 33 s).
  - On ROCm v0.31.0 with MTP at 64 sessions, the hit rate is 42% vs 87.5% without spec.
  - A 2026-10-17 comparison of candidate fixes covers v0.31.0 and a 2026-10-15 nightly. `disable_eagle_block_drop` restores the hits (0/10 lengths losing a block).
  - Source: [#54360](https://github.com/vllm-project/vllm/issues/54360).
- **[#55533](https://github.com/vllm-project/vllm/issues/55533)**, the MTP scheduler cap on a 27B hybrid (reported on RTX 4090 D SM89 with GPTQ INT4) — OPEN.
  - Symptom: at bs 8 only 3 sequences are scheduled per step; MTP is "2-3x slower than no-spec".
  - Independent reproduction on RTX 6000 Ada with `RedHatAI/Qwen3.8-27B-INT4`: the window is `floor((num_gpu_blocks − 1) / (1 + G·(1+k)))` with G = 3 mamba groups. MTP reserves `1+k` mamba blocks per group per request. The MTP head (0.88 GiB) also shrinks the pool.
  - The repro only collapses when the pool is small, i.e. it is a capacity effect — [#55533](https://github.com/vllm-project/vllm/issues/55533).
- **[#54690](https://github.com/vllm-project/vllm/issues/54690)**, speculative `kv_cache_dtype=fp8` crashes hybrid GDN startup — OPEN.
  - Reported on 4x RTX 4090 (SM89, PCIe, TP4) with a DFlash2 drafter: FlashInfer "Unrecognized dtype: auto". With the target set to bf16 explicitly, there is a native crash on SM89.
  - A commenter measured DFlash at about 185k context degrading single-stream decode from about 70 to about 16 tok/s, because the drafter re-scans the full draft KV.
  - Source: [#54690](https://github.com/vllm-project/vllm/issues/54690).
- **[#59232](https://github.com/vllm-project/vllm/pull/59232)**, "Tune small-M block-FP8 Triton configs for NVIDIA **L20**" — OPEN.
  - Claimed effect, measured on a local `Qwen3.8-27B-FP8`: TP1 decode TPOT −0.82% to −1.15%, output TPS +0.73% to +1.15%.
  - It does not cover L4.
  - Related SM89 items, both OPEN: [#61224](https://github.com/vllm-project/vllm/pull/61224) "CUTLASS blockwise FP8 GEMM for SM89 (Ada)" and [#60171](https://github.com/vllm-project/vllm/pull/60171).
  - Source: [#59232](https://github.com/vllm-project/vllm/pull/59232).
- **Async scheduling for hybrids:**
  - [#58065](https://github.com/vllm-project/vllm/pull/58065) "Enable async scheduling for DFlash" — MERGED-in-v0.31.0.
  - Caveat from HyperQwen ([#121](https://github.com/syv-ai/HyperQwen/issues/121), vLLM 0.28): on RTX 4090 SM89, "concurrent MTP3 + FP8 Triton returns empty/garbage output"; `--no-async-scheduling` avoided it in 9/9 runs, at a decode-speed cost.
- **All-reduce on PCIe:**
  - [#53576](https://github.com/vllm-project/vllm/pull/53576) "Add opt-in FlashInfer PCIe IPC all-reduce backend" — MERGED-in-v0.30.0. Opt-in via `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`; it "self-disables when the installed FlashInfer lacks the required API"; tested E2E on 4x RTX 6000 Pro (SM120) TP4.
  - The follow-up import fix [#54869](https://github.com/vllm-project/vllm/pull/54869) is MERGED-in-v0.30.0.
  - vLLM 0.30.0 pins `flashinfer-python==0.6.18.post1` (no kernel). 0.31.0 and 0.32.0 pin `0.7.0.post1`, which has it.
  - Sources: [#53576](https://github.com/vllm-project/vllm/pull/53576); [PyPI vllm 0.30.0](https://pypi.org/project/vllm/0.30.0/), [0.31.0](https://pypi.org/project/vllm/0.31.0/), [0.32.0](https://pypi.org/project/vllm/0.32.0/).
- **`thinking_token_budget` with MTP:**
  - [#39573](https://github.com/vllm-project/vllm/issues/39573) was CLOSED on 2026-08-29. Spec-decode budget support originally came from [#34668](https://github.com/vllm-project/vllm/pull/34668), merged 2026-04-29.
  - A commenter then measured that with Model Runner V2 (default since 0.29) + MTP, the budget is still not enforced ("V2 + MTP gives 2000 reasoning tokens and empty content at budgets 64 and 200, while V1 + MTP gives 63 and 199").
  - The follow-up issue [#54906](https://github.com/vllm-project/vllm/issues/54906) was closed by [#58231](https://github.com/vllm-project/vllm/pull/58231) "Fix split top-p masking every token for huge logits (thinking_token_budget + spec decode emits token 0 …)". That PR merged on 2026-10-19 and is MERGED-main-only (compare with v0.32.0: `diverged`).
  - Sources: [#39573](https://github.com/vllm-project/vllm/issues/39573), [#54906](https://github.com/vllm-project/vllm/issues/54906), [#58231](https://github.com/vllm-project/vllm/pull/58231).
- **Other hybrid fixes and their releases:**
  - [#60533](https://github.com/vllm-project/vllm/pull/60533) "Fix inconsistent hybrid Mamba prefix cache boundaries" — MERGED-in-v0.32.0;
  - [#59759](https://github.com/vllm-project/vllm/pull/59759) "Retire stale Mamba spec blocks in prefill checkpoint steps" — MERGED-in-v0.32.0;
  - [#60797](https://github.com/vllm-project/vllm/pull/60797) "Store conv_kernel - 1 inputs in Mamba2 prefill checkpoints" — MERGED-main-only;
  - [#58368](https://github.com/vllm-project/vllm/pull/58368) "Restore prompt-tail prefix-cache hits with MTP" — MERGED-in-v0.31.0;
  - [#52928](https://github.com/vllm-project/vllm/pull/52928) "FlashInfer ReplaySSM support for MTP" — MERGED-in-v0.32.0. This is for Mamba2.
- **v0.31 regression report:** a 2x RTX 3090 PP=2 setup with fp8 KV and prefix cache crash-loops on 0.31.0 (Xid 31 illegal memory access, 75 restarts in about 10.5 h). 0.30 "ran 7 days with zero Xids". The user rolled back to 0.30 — [HyperQwen #328](https://github.com/syv-ai/HyperQwen/issues/328).

### Inferences
- **For DFlash2 on vLLM, 0.32 is the first release with the prefix-hit mamba-state fix (#55601).** Earlier 0.30/0.31 risk acceptance collapse or an illegal memory access on prefix hits. HyperQwen reproduced that IMA on Ada. Our agent harness depends on prefix hits for continuation.
- **vLLM 0.31+ can use the FlashInfer PCIe-IPC all-reduce at TP4, the same kernel as SGLang 0.5.22.** Both engines therefore offer the same PCIe all-reduce lever, and it should be compared under the same P2P prerequisite.
- **#55533 is a block-budget effect, not a hard scheduler bug.** With TP4 and a large KV pool on 4x L4 it may not bind at 12 streams. Read `num_gpu_blocks` and the real running count before attributing slowness to MTP.
- **FP8 draft KV (#54690) is still blocked on SM89 in vLLM.** Plan for a bf16 draft KV.
- **We do not depend on #58231 if we cap thinking ourselves.** Our harness enforces budgets client-side or by max_tokens, so the nightly-only thinking-budget fix matters only if we switch to `thinking_token_budget` with MTP/DFlash on the V2 runner.

### Gaps
- I found no source that ran vLLM 0.32 with DFlash2 on SM89 / L4, nor one that confirms #55601 cures #58894 (the issue is still open).
- I did not verify whether vLLM's PCIe-IPC backend handles TP4 on SM89. The PR was tested only on SM120.
- I did not read the full vLLM nightly commit log after 2026-10-18.

---

## Q3. Offline install on our Kaggle image (Python 3.12, glibc 2.35, CUDA 13.0 driver)

### Takeaway
Both upgrades are feasible offline with our existing E11-02 wheelhouse pipeline. Each is a full rebuild, about 5 GB and roughly one Kaggle CPU run plus the publish step; it is not an incremental change.
- **SGLang 0.5.22** moves torch 2.13.0 → 2.14.1. torch 2.14.1 still uses the CUDA 13.0.3 toolkit, so the driver and the nvcc 13.0.88 pin are unchanged.
  - FlashInfer moves 0.6.18 → 0.7.1, plus jit-cache `0.7.1+cu130`. The '+' trap still applies.
  - It still pins the sdist-only `cuda-tile==1.6.0rc5`.
  - Its cp312 manylinux_2_34 wheel installs on glibc 2.35.
- **SGLang nightly** (needed for #44401) drops the cuda-tile pin and moves to FlashInfer 0.7.2 and sglang-kernel 0.4.10.
  - Its wheel is tagged **manylinux_2_39**, which pip rejects on glibc 2.35, even though its native `.so` files need only GLIBC_2.34. A wheel retag would be required.
- **vLLM 0.32** keeps torch 2.13.0 (the same as our 0.5.21 wheelhouse) with FlashInfer 0.7.0.post1. It ships as a cp38-abi3 manylinux_2_28 wheel. Its install shape is the simplest of the three.

### Cited Findings
- **SGLang 0.5.22 PyPI metadata** — [PyPI sglang 0.5.22](https://pypi.org/project/sglang/0.5.22/); [v0.5.22 Dependencies section](https://github.com/sgl-project/sglang/releases/tag/v0.5.22):
  - `requires_python >=3.11`;
  - wheels for cp311/cp312/cp313, `manylinux_2_34_x86_64`, about 34 MB;
  - pins: `torch==2.14.1`, `torchaudio==2.11.0`, `flashinfer_python[cu13]==0.7.1`, `sglang-kernel==0.4.9`, `cuda-tile==1.6.0rc5`, `transformers==5.19.0`, `nvidia-cutlass-dsl[cu13]==4.8.0`, `quack-kernels==0.6.5`, `tilelang==0.1.12`, `sgl-deep-gemm==0.2.2`, `sgl-deep-ep==0.1.2.post1`, `apache-tvm-ffi==0.1.11`, `xgrammar==0.2.7`, `torchcodec==0.17.0`, `cuda-python>=13.0`.
- **torch 2.14.1** — [PyPI torch 2.14.1](https://pypi.org/project/torch/2.14.1/):
  - requires `cuda-toolkit[...]==13.0.3`, the same CUDA 13.0 toolkit set as torch 2.13;
  - plus `nvidia-cudnn-cu13==9.24.0.43`, `nvidia-nccl-cu13==2.30.7` and `triton~=3.8.0`;
  - a cp312 manylinux_2_28 x86_64 wheel exists on PyPI.
- **cuda-tile:**
  - `cuda-tile 1.6.0rc5` is sdist-only (`cuda_tile-1.6.0rc5.tar.gz`), so the trap persists in 0.5.22;
  - binary wheels exist for `1.6.0` and `1.7.0` (2026-10-19, cp310 manylinux2014);
  - flashinfer 0.7.x needs only `cuda-tile>=1.4.0`;
  - sources: [PyPI cuda-tile history](https://pypi.org/project/cuda-tile/#history); [PyPI flashinfer-python 0.7.1](https://pypi.org/project/flashinfer-python/0.7.1/).
- **FlashInfer packages:**
  - `flashinfer_python 0.7.1` is a `py3-none-any` wheel;
  - the cu130 jit-cache wheels `flashinfer_jit_cache-0.7.1+cu130` and `0.7.2+cu130` (cp39-abi3 manylinux_2_28) are on the FlashInfer index;
  - `flashinfer-cubin` has no 0.7.1 release on PyPI (404), the same as with 0.6.18;
  - sources: [FlashInfer cu130 jit-cache index](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/); [PyPI flashinfer-cubin](https://pypi.org/project/flashinfer-cubin/).
- **sglang-kernel:**
  - `0.4.9` and `0.4.10` on PyPI are `cp310-abi3-manylinux2014_x86_64` (371–373 MB) and require `torch==2.14.1`;
  - the cu130 index also carries `sglang_kernel-0.4.9+cu130` (with a '+');
  - sources: [PyPI sglang-kernel 0.4.9](https://pypi.org/project/sglang-kernel/0.4.9/); [sgl-project/whl cu130 sglang-kernel](https://github.com/sgl-project/whl/tree/main/cu130/sglang-kernel).
- **SGLang nightly:**
  - wheels are named like `sglang-0.5.23.dev20261022+gb453996b7d-cp312-cp312-manylinux_2_39_x86_64.whl`, with a '+' in the name and the **manylinux_2_39** tag;
  - they are hosted as GitHub release assets `nightly-YYYY-MM-DD-<sha>`;
  - the main pyproject at `b453996b7d` pins `torch==2.14.1`, `flashinfer_python[cu13]==0.7.2`, `sglang-kernel==0.4.10`, `transformers==5.19.0`, and has **no cuda-tile pin**;
  - sources: [nightly index](https://github.com/sgl-project/whl/blob/main/cu130/sglang/index.html); [nightly release asset](https://github.com/sgl-project/whl/releases/tag/nightly-2026-10-22-b453996b7d); [pyproject.toml@b453996b7d](https://github.com/sgl-project/sglang/blob/b453996b7d/python/pyproject.toml).
- **My glibc scan** (a string scan of `GLIBC_2.x` symbol versions):
  - the native extensions are the same five in both the 0.5.22 PyPI wheel and the 2026-10-22 nightly: `mem_cache`, `_bench`, `_grpc`, `_multimodal`, `_server` (`.cpython-312-x86_64-linux-gnu.so`, Rust);
  - in both wheels, the highest symbol version referenced is **GLIBC_2.34**;
  - this is a heuristic scan, not `readelf` or `auditwheel`;
  - source: my inspection of the [0.5.22 wheel](https://pypi.org/project/sglang/0.5.22/#files) and the [nightly wheel](https://github.com/sgl-project/whl/releases/tag/nightly-2026-10-22-b453996b7d).
- **SGLang nightly Docker tags** exist (`dev-*`, `nightly-cu134`). An image cannot be used on Kaggle, which runs its own pinned image — [Docker Hub lmsysorg/sglang tags](https://hub.docker.com/r/lmsysorg/sglang/tags).
- **vLLM wheels** — [PyPI vllm 0.30.0](https://pypi.org/project/vllm/0.30.0/), [0.31.0](https://pypi.org/project/vllm/0.31.0/), [0.32.0](https://pypi.org/project/vllm/0.32.0/):
  - every release is `cp38-abi3-manylinux_2_28_x86_64`, 315–327 MB;
  - all three pin `torch==2.13.0`, `torchaudio==2.11.0` and `torchvision==0.28.0`;
  - FlashInfer: 0.30 pins `flashinfer-python==0.6.18.post1`; 0.31 and 0.32 pin `0.7.0.post1`;
  - cutlass-dsl: 4.7.1 in 0.30 and 0.31, 4.8.0 in 0.32;
  - transformers: 0.32 needs `>=5.16.1,<5.20.0` and Python `>=3.11`;
  - `instanttensor` is a hard dependency (≥ 0.1.9; ≥ 0.2.2 in 0.32).
- **vLLM 0.32 notes:**
  - "`flashinfer-jit-cache` excluded from wheel `install_requires` (#60052)";
  - a new `vllm download-kernels` installs FlashInfer's precompiled kernels (#58765);
  - for offline Kaggle we must bundle the jit-cache ourselves;
  - source: [v0.32.0 notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0).
- **vLLM nightly** wheels are named `vllm-0.32.1rc1.dev328+g2de0531d8-cp38-abi3-manylinux_2_28_x86_64.whl` (with a '+') — [vLLM nightly index](https://wheels.vllm.ai/nightly/vllm/).
- **The nvcc mismatch trap is still upstream.**
  - The OPEN SGLang PR [#43591](https://github.com/sgl-project/sglang/pull/43591) "Pin pip CUDA JIT toolchain to the CUDA minor required by torch" says `humming-kernels[cu13]` can pull an unconstrained `nvidia-cuda-nvcc` 13.4.92 while torch pins runtime 13.0. That mismatch causes a CCCL error in the JIT.
  - Its proposed pin is `cuda-toolkit[nvcc,nvvm,cccl]==13.0.3` (nvcc 13.0.88), which matches our E11-02 rev 2 pin.
- **Project traps already found in E11-02** (internal memory note `C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/sglang-int4-dflash2-plan.md`; not a web source):
  - the cuda-tile sdist override;
  - jit-cache not cubin;
  - Kaggle drops '+' from dataset file names, fixed by `install_wheelhouse.py`, which restores names from SHA256SUMS;
  - a venv install, not `--target`;
  - `CUDA_HOME=<site>/nvidia/cu13` with `lib/libcudart.so` and `lib64` symlinks;
  - a fresh `SGLANG_CACHE_DIR`.

### Inferences
- **Effort, SGLang 0.5.22.** Rerun the E11-02 builder with new pins:
  - torch 2.14.1 set;
  - flashinfer 0.7.1 + jit-cache 0.7.1+cu130;
  - sglang-kernel 0.4.9;
  - transformers 5.19.0;
  - cutlass-dsl 4.8.0;
  - keep the cuda-tile override (1.6.0 or 1.7.0, `--no-deps`);
  - keep the nvcc 13.0.88 pins and the '+' restore installer.
  - Expect about the same about 220 wheels / about 5 GB, because the torch + CUDA wheels dominate and all change. This is one CPU build run plus one publish run with Secrets, as before.
  - The CUDA driver risk is unchanged, because the toolkit stays 13.0.
- **Effort, SGLang nightly.** The same as 0.5.22, plus two steps:
  - Retag or rename the sglang wheel from `manylinux_2_39` to `manylinux_2_34` (or install it with `--no-deps` from a renamed file). The symbol scan suggests the binaries would load on glibc 2.35, but this is unverified on Kaggle.
  - Bundle flashinfer 0.7.2 + jit-cache 0.7.2+cu130 and sglang-kernel 0.4.10.
  - The upside: #44401 plus the post-release hybrid fixes, with no cuda-tile sdist problem.
  - The alternative with lower build risk: stay on 0.5.22 wheels and monkey-patch #44401 + #41872 in-process.
- **Effort, vLLM 0.32.** The torch 2.13.0 / CUDA 13.0 base equals our current SGLang 0.5.21 wheelhouse, so many CUDA wheels can be reused. The new pieces are the vLLM wheel (327 MB), flashinfer 0.7.0.post1 + jit-cache, cutlass-dsl 4.8.0, instanttensor 0.2.2 and transformers 5.16–5.19.
- **Both SGLang 0.5.22 and vLLM 0.31+ make the PCIe-IPC all-reduce reachable.** The wheelhouse must include the FlashInfer 0.7.x jit-cache. Otherwise the `pcie_ipc_comm` module JIT-compiles on the GPU at first use, which needs our wheel nvcc setup to work.

### Gaps
- Not tested: whether pip on Kaggle accepts a retagged SGLang nightly wheel, and whether its Rust `.so` files load on glibc 2.35.
- I did not enumerate the full 0.5.22 transitive lock, so the exact wheel count and size of a new wheelhouse is an estimate.
- I did not check whether `torchaudio==2.11.0`, as pinned by sglang 0.5.22, imports cleanly against torch 2.14.1. It is not needed for text-only serving.

---

## Q4. Benchmarks: reported deltas from SGLang 0.5.21 to newer versions, and between vLLM 0.30 / 0.31 / 0.32, for Qwen3.5/3.8 hybrids

### Takeaway
I found no published like-for-like benchmark of SGLang 0.5.21 vs 0.5.22, or of vLLM 0.30 vs 0.31 vs 0.32, for Qwen3.5/3.8 hybrids on any GPU.
- What exists is per-PR kernel or model numbers on Blackwell/Hopper, one L20 FP8 tuning PR (about 1% TPOT), and community stability reports.
- One community report is negative: vLLM 0.31 crash-looped on 2x 3090 PP=2 while 0.30 was stable.

### Cited Findings
- **Per-PR numbers in the SGLang v0.5.22 notes, none for Qwen3.8-27B on Ada** — [v0.5.22 notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22):
  - Qwen3.8-Flash-Next BF16 decode on 4x B200: 242 → 271 tok/s at concurrency 1 ([#42175](https://github.com/sgl-project/sglang/pull/42175));
  - GLM-5.3-Flash DFlash on 8x B300: 1091.9 → 1199.7 tok/s ([#42489](https://github.com/sgl-project/sglang/pull/42489));
  - QSA decode step 28.75 → 26.54 ms at bs64 ([#43463](https://github.com/sgl-project/sglang/pull/43463));
  - mamba COW warm-up: first-wave cold excess 924 → 28 ms ([#36266](https://github.com/sgl-project/sglang/pull/36266)).
- **PCIe-IPC all-reduce** (8x SM120, from the SGLang module docstring): TPOT 21.14 → 13.62 ms and output tok/s 35.02 → 48.05 at TP8 with 8k context, vs NCCL — [pcie_ipc_ar.py@v0.5.21](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py).
- **NCCL P2P on consumer PCIe at TP4** (4x RTX 5060 Ti, vLLM 0.30.0, DFlash2, `--disable-custom-all-reduce`):
  - NCCL's default leaves P2P off (uses SHM) above `PXB`;
  - `NCCL_P2P_LEVEL=PHB` gave "decode … 2 to 6% faster at C1 and 10 to 16% faster at C4", with no change in prefill;
  - source: [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254).
- **vLLM L20 block-FP8 tuning** (Qwen3.8-27B-FP8, TP1): TPOT −0.82% to −1.15%. CLAIMED; the PR is open — [vLLM #59232](https://github.com/vllm-project/vllm/pull/59232).
- **vLLM 0.30 → 0.31 stability:** a 0.31.0 crash-loop with 75 restarts on 2x 3090 PP=2 + fp8 KV + prefix cache; "0.30 ran 7 days with zero Xids"; the user rolled back — [HyperQwen #328](https://github.com/syv-ai/HyperQwen/issues/328).
- **vLLM 0.31 hybrid prefix hits with DFlash** (DGX Spark, Qwen3.8-27B-NVFP4): sibling prompts alternate between a 3.2 s and a 33 s TTFT, because the hit is lost every other request — [vLLM #54360](https://github.com/vllm-project/vllm/issues/54360).
- **The SGLang Qwen3.8-27B cookbook measures on v0.5.19, SM12x only:**
  - DFlash2 on RTX 5090 bf16: "4.92 ms median TPOT at an accept length of 4.29";
  - `extra_buffer_lazy` lowers per-request state slots "from 5 slots to 4";
  - no Ada, PCIe or INT4 data;
  - source: [LMSYS cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B).
- **Third-party DFlash2 write-ups** (MindStudio, LinkedIn) give conflicting speedups and no engine versions, so they are not usable for version deltas — [MindStudio DFlash2](https://www.mindstudio.ai/blog/qwen3-8-27b-dflash2-speculative-decoding); [z-lab model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2).

### Inferences
- **No upgrade can be justified on published throughput for our setup.** The justification has to be the specific fixes:
  - #44401: 12/12 running instead of 11/12, about +9% slots at our cap;
  - the PCIe-IPC all-reduce: unmeasured on L4;
  - for vLLM, #55601.
- **Any upgrade should be validated by our own paired measurement.** That means the same notebook, 12 real running, tok/s by context band, and greedy plus A/A correctness, run before production.

### Gaps
- I found no benchmark of any engine version on L4 / sm89 at TP4 for Qwen3.8-27B. I did not search Chinese-language forums or Discord/Slack logs.
