# vLLM hybrid GDN + MTP + prefix-cache fixes: presence in v0.19.0 (S1-60K) and v0.32.0 (E18)

Date: 2026-10-27. Read-only research.
Sources:
- GitHub REST API via `gh`: PR metadata, the compare API for tag containment, PR diffs, and issue comments.
- Raw source files at tags v0.19.0, v0.28.0, v0.29.0 and v0.32.0, saved under `D:\ARC AGI 2\_scratch\vllm_src\<tag>\`.
- All 26 comments of #53912 are saved in `D:\ARC AGI 2\_scratch\vllm_src\issue53912_comments.txt`; the #43559 comments are in `issue43559_comments.txt`.

Tag containment method: `gh api repos/vllm-project/vllm/compare/<tag>...<merge_sha>`. A status of `behind` or `identical` means the tag contains the commit. `diverged` means it does not.
Note: **tag v0.28.1 does not exist.** The stable releases are ... v0.27.1, v0.28.0, v0.29.0, v0.30.0, v0.31.0, v0.32.0 (2026-10-18). Only the pre-release tag `v0.28.1rc0` exists.

## 1. PR status table

| PR | Title | State | Merged (UTC) | First stable tag containing it | Evidence |
|---|---|---|---|---|---|
| #50729 | [Bugfix][Mamba] Fix overlapping state copy race | merged | 2026-08-17 06:29 (a02cfccb) | **v0.29.0** (also pre-release v0.28.1rc0) | compare: v0.28.0 diverged (ahead 2), v0.28.1rc0 behind, v0.29.0 behind, v0.32.0 behind. The v0.29.0 release notes list "Mamba state copy race (#50729)". |
| #43650 | [Bugfix][Core] MTP + enable prefix caching + mamba accuracy fix | **open, not merged** (mergeable_state blocked; last update 2026-09-29) | - | none | API `merged=false` |
| #48375 | [BugFix] Honor drop_eagle_block in MambaManager | **open, not merged** (dirty / conflicts; last update 2026-09-08) | - | none | API `merged=false` |
| #51113 | [Bugfix] Keep mamba align prefill chunks block-aligned past last_cache_position | merged | 2026-08-06 17:21 (c56f169d) | **v0.28.0** | compare: v0.27.0 and v0.27.1 diverged, v0.28.0 behind, v0.32.0 behind. #43559 was closed at 2026-08-06 17:21:01, one second after this merge. |
| #53919 | [Bugfix] Await the accepted-token copy before moving batch rows | **open, not merged** (blocked; label `mrv1-only`; last update 2026-08-28) | - | none | API `merged=false` |
| #57128 | [Bugfix] MambaManager.find_longest_cache_hit ignores drop_eagle_block, silent corruption with prefix caching + MTP/EAGLE | **open, not merged** (labels `needs-rebase`, dirty; last update 2026-10-27) | - | none | API `merged=false` |

Only 2 of the 6 PRs are merged: #51113 and #50729. The 4 cache-read / async fixes are not in any release.

## 2. Presence in v0.19.0 and v0.32.0, with code

### 2a. #51113 (align chunk-end alignment; prefix-cache poisoning under concurrent prefills)
- **v0.19.0: ABSENT.** In `vllm/v1/core/sched/scheduler.py` `_mamba_block_aligned_split`, alignment is enforced only below `last_cache_position`. Past it, any chunk end is accepted:
  ```python
  if self.use_eagle:
      last_cache_position = max(last_cache_position - block_size, 0)
  num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
  if num_computed_tokens_after_sched < last_cache_position:
      # align to block_size
      num_new_tokens = num_new_tokens // block_size * block_size
  elif (num_computed_tokens < last_cache_position < num_computed_tokens_after_sched):
      # force to cache the last chunk
      num_new_tokens = last_cache_position - num_computed_tokens
  else:
      # prefill the last few tokens
      pass
  ```
- **v0.32.0: PRESENT.** The gate is now the prefill end (the exact #51113 hunk):
  ```python
  prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
  ...
  # Invariant: slot p holds the state after exactly (p + 1) * block_size
  # tokens. State is written at chunk ends, so chunk ends must be block
  # aligned. Exempt: the prompt's last chunk, ...
  if end < prefill_end:
  ```
  The mid-block re-align stop also matches the PR: `next_block_boundary if start % block_size != 0 else 0`.

### 2b. #50729 (overlapping same-block conv-state copy race)
- **v0.19.0: ABSENT.**
  - `vllm/v1/worker/mamba_utils.py` `batch_memcpy_kernel` loads and stores 1024-byte tiles with no overlap ordering:
    ```python
    data = tl.load(curr_src_ptr, mask=mask)
    tl.store(curr_dst_ptr, data, mask=mask)
    ```
  - The same-block shifted copy is reachable: `collect_mamba_copy_meta` returns early only when `src_block_idx == dest_block_idx and accept_token_bias == 0`, so a same-block copy with bias > 0 goes through `batch_memcpy`.
  - (v0.19.0 has no fused `_memcpy_u64_tiled` kernel. The generic kernel is the only path.)
- **v0.32.0: PRESENT.** Same file:
  ```python
  is_left_overlap = dst_ptr < src_ptr and dst_ptr + size > src_ptr
  ...
  data = tl.load(curr_src_ptr, mask=mask)
  if is_left_overlap:
      # Preserve each lane's source before a lower-address lane stores
      # over it. The condition is uniform within the program.
      tl.debug_barrier()
  tl.store(curr_dst_ptr, data, mask=mask)
  ```
  The fused SD-conv path also has the ordered copy: "Distinct blocks and exact self-copies cannot have a destructive overlap ... Copy tokens from low to high."

### 2c. #43650 / #48375 / #57128 (MambaManager drops the trailing block on a cache hit)
- **v0.19.0: ABSENT.**
  - `MambaManager.find_longest_cache_hit(..., use_eagle: bool, ...)` never reads `use_eagle`. Its body is:
    ```python
    for i in range(max_num_blocks - 1, -1, -1):
        if cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids):
            ...
            computed.extend([block_pool.null_block] * i)
            computed.append(cached)
            break  # we just need the last match - early stopping
    return computed_blocks
    ```
    `FullAttentionManager` does read it: `if use_eagle and computed_blocks[0]: ... computed.pop()`.
  - Mitigating detail (v0.19.0 `kv_cache_coordinator.py`, hybrid loop): full attention runs first and pops its last block. The Mamba finder is then called with `max_length=curr_hit_length`, the length after that pop. So Mamba's search ceiling is already one block lower. The Mamba hit can equal that reduced length, but it cannot exceed it.
- **v0.32.0: ABSENT (the unmerged PRs are not applied).**
  - `MambaManager.find_longest_cache_hit(..., drop_eagle_block: bool, ...)` still never reads `drop_eagle_block`, in either branch:
    - The fine-grained branch: `for fine_idx in range(max_num_partial_units - 1, -1, -1): ... hit_length = num_tokens; break`.
    - The coarse branch: `for i in range(max_num_blocks - 1, -1, -1): ... hit_length = (i + 1) * block_size; break`.
  - The coordinator (`kv_cache_coordinator.py`) states the design explicitly:
    ```python
    # Eagle matches one extra drop unit ... and then drops it, landing back at the candidate length.
    # No margin for mamba: its finder never drops (draft models have no mamba
    # layers), so the hit would grow past the candidate.
    if drop_eagle_block and not isinstance(spec, MambaSpec):
    ```
  - v0.32.0 replaced the read-side drop with a write-side shift. The coordinator sets `manager.drop_eagle_checkpoint_block = True` for every MambaManager when `use_eagle`. `kv_cache_interface.get_mamba_prefill_checkpoint_position` then places the prefill checkpoint one block lower:
    ```python
    proof_limit = num_tokens if drop_eagle_block else num_tokens - 1
    checkpoint_position = proof_limit // hash_block_size * hash_block_size
    if drop_eagle_block:
        checkpoint_position -= hash_block_size
    ```
    `speculative_config.disable_eagle_block_drop` defaults to `False`, so this shift is on for our MTP config.
  - Whether this write-side shift fully replaces the read-side drop is disputed in #53912:
    - Reports that it does not: naeemakhtar110 (0.28.0, coarse path; the #48375 unit test fails on plain 0.28.0), ming616 (0.30.0 code), "v0.31.0 still accepts drop_eagle_block ... and never uses it".
    - cch-zuzuche withdrew their own fine-grained drop hunk because it broke 9 upstream prefix-cache tests: "The mamba search bound is already one drop unit below the attention hit".
  - **UNVERIFIED** whether v0.32.0 is actually exposed through this path.

### 2d. #53919 (async race: accepted-token counts permuted twice)
- **v0.32.0, model runner V1: ABSENT.**
  - `gpu_model_runner.py` `_update_states` has no `num_accepted_tokens_event.synchronize()` before `add_request` / `condense`. The only `synchronize()` is in `_prepare_inputs`, after the row moves, and it is followed by the gather the PR removes:
    ```python
    needs_cpu_accepted_counts = self.num_accepted_tokens_event is not None and not (
        self.use_async_scheduling and self.cache_config.mamba_cache_mode != "align"
    )
    if needs_cpu_accepted_counts:
        self.num_accepted_tokens_event.synchronize()
        if self.use_async_scheduling and prev_req_id_to_index:
            prev_idx = self.prev_positions.np[:num_reqs]
            ...
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[np.where(new_mask, 0, prev_idx)])
    ```
  - **BUT model runner V2 is the default in v0.32.0 for our config, and V2 does not use this code.**
    - `vllm/config/vllm.py` `use_v2_model_runner`: if `VLLM_USE_V2_MODEL_RUNNER` is unset (envs default `None`), Triton is present, and `_get_v2_model_runner_unsupported_features()` is empty, it returns `True`.
    - Our config hits none of the unsupported entries. The spec-decode entries are suffix / medusa / mlp_speculator / custom_class, parallel drafting, and heterogeneous vocab; the others are PP with external_launcher and ubatching.
    - V2 (`vllm/v1/worker/gpu/model_states/mamba_hybrid.py`) keeps the counts GPU-resident, indexed by request slot: `self.num_accepted_tokens_gpu = torch.ones(self.max_num_reqs, ...)`. They are updated by `_scatter_num_accepted_kernel(idx_mapping, num_sampled, ...)`, and the align pre/post copies run as fused GPU kernels. There is no pinned CPU buffer that is permuted by row moves.
    - The PR's own label is `mrv1-only`.
  - => On v0.32.0 + MRv2, the #53919 mechanism does not exist (code reading).
  - **UNVERIFIED from our logs** that E18 actually runs MRv2: check the E18 server log for the V2 runner (or the "Model Runner V2 does not yet support ...; using the V1 model runner" warning).
- **v0.19.0: ABSENT, and the bug path is live, probably in a worse (deterministic) form. Code-reading inference, UNVERIFIED by experiment.**
  - v0.19.0 uses MRv1 only (`VLLM_USE_V2_MODEL_RUNNER` default `False`).
  - In align mode, `_update_states_after_model_execute` (called from `sample_tokens` of step N) writes the counts **synchronously** in step N's row order:
    ```python
    if self.cache_config.mamba_cache_mode == "align":
        for i, num_tokens in enumerate(self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()):
            self.input_batch.num_accepted_tokens_cpu[i] = num_tokens
    ```
    (The event is recorded only in the non-align `else:` branch.)
  - Step N+1 `_update_states` then calls `condense()` and `_may_reorder_batch()`. `gpu_input_batch.py` `swap_states` and `condense` move `num_accepted_tokens_cpu` (lines 631-633 and 756), so the values are now in the current order.
  - `_prepare_inputs` then applies the async gather unconditionally:
    ```python
    if self.num_accepted_tokens_event is not None:
        self.num_accepted_tokens_event.synchronize()
        # Async mode: condense() reordered indices, use prev_positions mapping
        if self.use_async_scheduling and prev_req_id_to_index:
            prev_idx = self.prev_positions.np[:num_reqs]
            ...
            self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[np.where(new_mask, 0, prev_idx)]
    ```
  - That is the "copy landed before the row moves -> second permute" case that #53919 describes. The PR notes: "the wait alone is worse (7.6% vs 4.3% baseline), because it guarantees the early landing that triggers the double permute".
  - In v0.19.0 the early landing is guaranteed by the synchronous `.cpu()`. Any step where a row moves (a request finishes and leaves a hole, or a decode-first reorder swap) can give a request another row's count. That count is used as `accept_token_bias` in `preprocess_mamba` and as the GDN spec-decode state selector.
  - `--no-async-scheduling` removes the gather branch entirely.

### 2e. Summary matrix

| Fix | v0.19.0 | v0.32.0 |
|---|---|---|
| #51113 chunk-end alignment | absent | **present** |
| #50729 overlap copy race | absent | **present** |
| #53919 accepted-count race | absent; path live (async on, MRv1, sync align write => double permute on row moves; inference) | absent in MRv1; **mechanism not present in MRv2 (default)** |
| #43650/#48375/#57128 Mamba trailing-block drop | absent (Mamba ceiling = full attention's post-drop length) | absent (replaced by the write-side checkpoint shift; adequacy disputed, UNVERIFIED) |

## 3. Issue #53912 status and latest reports
- **OPEN** (created 2026-08-26, 26 comments, last 2026-10-14). Title: "prefix caching + MTP still corrupts output on hybrid Mamba/GDN models in v0.28.0 (#43559 closed but unfixed)".
- OP (0.28.0, Qwen3.5-arch 27B FP8, MTP k=2, TP1, async on by default): 4/495 malformed (`!!!!` or empty). On 0.26.0: 25/2,636.
- 2026-08-26 johny-jose: "16/288 with async scheduling, 0/288 without, 0/288 with prefix caching but no MTP" (the #53919 repro on Qwen3.8-27B-FP8, MTP=3). In the PR, 0.26.0 at concurrency 100: 186/5400 baseline, 0/5400 patched, 0/1800 with `--no-async-scheduling`.
- 2026-08-28 ionut-anghelina (OP): "I managed to check fix #50729 and it solves the issue."
- 2026-08-31 karls0r (0.27.1, Qwen3.8-27B INT8, 2x3090 TP2, fp8_e4m3 KV, MTP k=2, async default on): sustained corrupted window. Disabling prefix caching with MTP kept on: 0/30 corrupted.
- 2026-09-02 / 09-15 rdlh: 0.27.1 gives 12/5000 (async on). Nightly `0.28.1rc1.dev284` gives **0/5000** and 0/4800 with the same flags. "So this looks fixed on current main / nightly (post-0.28.0)."
- 2026-09-24 Suppressor72 (0.28.1rc1.dev642, DFlash, with the `disable_eagle_block_drop=true` opt-out): rejected-draft contamination is reproduced only with the opt-out. Removing the opt-out restored the correct boundary.
- 2026-09-28 SergiioB (XPU fork 0.26.1.dev, W4A16, MTP4): 5/6 corrupted, **including with `--no-async-scheduling`** and on zero-hit cold requests. Disabling prefix caching: 6/6 clean.
- 2026-09-29 to 10-04 cch-zuzuche (0.29.0, DFlash, not MTP): `!` walls on long reused prefixes. Their proposed fine-grained drop hunk was withdrawn as wrong.
- 2026-10-11 SebastianGode: Flash-Next MTP acceptance 0 % at a 97.6 % prefix hit rate. Version not stated (UNVERIFIED).
- **Latest, 2026-10-13 / 10-14, naeemakhtar110** (0.28.0, GB10, Qwen3.6-35B-A3B-NVFP4, MTP 3, `--async-scheduling`, align, coarse path):
  - 1 garbled run of ~40.
  - With #48375 applied: "#48375 alone was not enough here. A garbled reply came back".
  - Next step: "We are now serving 0.28.0 with both backported [#48375 + #50729] and MTP still on". No follow-up yet.
- **No comment in #53912 reports a test on v0.30, v0.31 or v0.32 releases.**
  - v0.29.0 reports: cch-zuzuche (DFlash, not MTP) and the #57128 author (v0.29.0, "Before this fix ... corruption risk"; the PR text does not include a measured corruption count).
  - ming616 cites v0.30.0 code only; their field run was a fork based on main 2026-09-10.
  - Whether it reproduces on v0.32.0 with MTP: **UNVERIFIED (no report found)**.
- #43559: closed 2026-08-06 by #51113.
  - Comment 2026-07-25 (cduk): could not reproduce on 0.19.1rc1.dev168 with Qwen3.5-9B. Whether MTP was on in that run is not stated, so this is weak evidence.
  - amittell summary: "0.19 clean (@cduk), 0.21-0.25.1 deterministic positional corruption, 0.26 positional corruption fixed but engine-fatal CUDA crash plus temp=0 divergence".
  - dmih: "on 0.19 MTP for Qwens is also should not be used due to parser bugs" (a tool/reasoning parser issue, not the cache).

## 4. Async scheduling default
- **v0.19.0: ON for our setup.**
  - In `vllm/config/vllm.py` (lines ~755-792), `async_scheduling is None` is resolved to `True` unless the spec method is not in `EagleModelTypes`/`NgramGPUTypes`, `disable_padded_drafter_batch` is set, or the executor lacks support.
  - `EagleModelTypes = Literal["eagle", "eagle3", "extract_hidden_states", MTPModelTypes]`, and `"mtp"` is in `MTPModelTypes`.
  - `MultiprocExecutor.supports_async_scheduling()` returns `True` (TP4 uses mp).
  - Our own 0.19.0 log confirms it: E10 ORDER2 `q27-xhigh-e10-order2.log` line 561, `[vllm.py:790] Asynchronous scheduling is enabled.`, together with `speculative_config=SpeculativeConfig(method='mtp'` and `enable_prefix_caching=True`.
  - The S1-60K log itself was not checked. Same engine family; UNVERIFIED for S1 specifically.
- **v0.32.0: ON for our setup.**
  - Same `None -> True` resolution (`vllm/config/vllm.py` ~1585-1644).
  - The extra disable branches do not apply to us: pooling, ROCm DeepEP DBO, and PP>1 without the V2 runner.
  - `EagleModelTypes` still includes `MTPModelTypes`.
  - `max_concurrent_batches` = `pp_size + 1` = 2 with the V2 runner.

## 5. Bottom line

**v0.19.0 (S1-60K) is the more exposed engine.** It lacks all four mechanisms' fixes:
- #51113: chunk-end poisoning under concurrent chunked prefills. We run ~20 concurrent seqs with a 4,096-token batch budget, so prefills share budget.
- #50729: overlap copy race.
- #53919: accepted-count race. Async is on and the MRv1 align write is synchronous, so by code reading a double permute happens on every row move.
- The Mamba trailing-block drop.

**v0.32.0 (E18) contains #51113 and #50729.** With the V2 runner (the default) the #53919 mechanism is structurally absent. The only remaining known suspect is the Mamba read-side drop (#57128 family). v0.32.0 addresses it by design with the write-side checkpoint shift, but no #53912 report tests v0.32 (UNVERIFIED).

Safest flag sets:

| Engine | Flag set | Removes | Cost |
|---|---|---|---|
| 0.19.0 | `--no-enable-prefix-caching` (alone suffices) | Sets `mamba_cache_mode='none'` (`model_executor/models/config.py`: "Mamba cache mode is set to 'none' when prefix caching is disabled"). That disables align split (#51113), align pre/post copies (#50729 path), the align sync count write that feeds the double permute (#53919 form), and every Mamba cache hit (#57128 family). | Loses prefix reuse. Our E18 0.32 smoke measured only 7.7 % prefix-cache hits (49,600 / 643,503), and phase 2 already re-prefills under align+MTP, so the expected loss is small. 0.19 hit rate UNVERIFIED. |
| 0.19.0 | `--no-async-scheduling` (keep prefix caching) | The #53919 gather branch only (it is gated on `self.use_async_scheduling`). Does NOT fix #51113, #50729 or the Mamba drop. | #53919 measured 1.2-4.4 % throughput loss at the knee on an H200. On 4xL4 TP4 the host share may differ (UNVERIFIED). |
| 0.19.0 | both flags | as row 1 | as rows 1 + 2; async-off is redundant for correctness once prefix caching is off, unless a non-align count race exists. In v0.19.0 the non-align path also reads CPU counts with the gather; v0.32 later excluded it "for this reason". So for 0.19 **both flags is the safest set**. |
| 0.32.0 | default flags + confirm MRv2 in the log | #53919 not applicable on MRv2; #51113 and #50729 present | none |
| 0.32.0 | `--no-enable-prefix-caching` | the remaining Mamba read-side suspect | same as the 0.19 row 1 (about 7.7 % of prompt tokens recomputed in the smoke) |
| 0.32.0 | `--no-async-scheduling` | nothing additional on MRv2 (needed only if the log shows the V1 runner) | 1-4 % throughput |
| either | MTP off | the whole speculation-gated class | ~40-60 % decode throughput (#43559/#53912 reports); not recommended |

The flags are mapped to the mechanisms by code reading. That none of these mechanisms actually fired in S1-60K's run is UNVERIFIED.
A direct test: replay the S1 workload on 0.19.0 with seed and temperature fixed, in arms {default, `--no-async-scheduling`, `--no-enable-prefix-caching`}. Count empty, `!!!!` and punctuation-run outputs and MTP-acceptance collapses (acceptance 0 % windows in the SpecDecoding metrics).
