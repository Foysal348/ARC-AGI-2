# 14 Claude agent: #44401 and #41872 backport for SGLang 0.5.21 (2026-10-23)

Built on CPU only. Nothing was pushed, nothing ran on a GPU, Kaggle was not touched.

Files:

- `src/e14_sglang_patches.py`: the patcher. Stdlib only. It is meant to be embedded as text in the notebook and run against `/tmp/sglang-venv/lib/python3.12/site-packages/sglang`.
- `src/test_e14_sglang_patches.py`: 36 pytest tests (CPU only).
- This note.

Inputs that were read in full:

- Upstream diffs, downloaded with curl from `patch-diff.githubusercontent.com/raw/sgl-project/sglang/pull/{44401,41872}.diff`:
  - #44401 is 9,634 bytes: 3 files, 5 hunks plus 1 new test file.
  - #41872 is 4,209 bytes: 1 file, 2 hunks plus 1 new test file.
- Local 0.5.21 source at `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/`. `_version.py` says `__version__ = version = '0.5.21'`. The source comes from the sglang-0.5.21 wheel and has LF line endings with no CR bytes. File hashes:
  - `srt/managers/scheduler.py`: sha256 `b61df5b7…06c4`
  - `srt/dllm/mixin/scheduler.py`: sha256 `29628477…5817`
- Code paths I read to confirm that 0.5.21 has the semantics the PR assumes. All line numbers are 0.5.21:
  - `scheduler.py`:
    - L3531-3535 `stash_chunked_request`
    - L3641-3782 `get_next_batch_to_run`
    - L3784-3802 `get_num_allocatable_reqs`
    - L3831-4109 `_get_new_batch_prefill_raw`, the whole admission path
    - L1183-1200, the min-free-slots delayer and the pp budget
  - `schedule_batch.py` L927-994: `ReqKvInfo`, with `holds_kv` at L968-970 defined as `req_pool_idx is not None`.
  - `schedule_policy.py` L1114-1182: `PrefillAdder.add_chunked_req`, which appends the continuing chunk to `can_run_list` at L1166.
  - `mem_cache/memory_pool.py`:
    - L314-363: `ReqToTokenPool.available_size`, `alloc`, `alloc_rows` and `free`.
    - L1434-1489: `HybridReqToTokenPool.alloc`, the Mamba/GDN pool our model uses.
  - `mem_cache/allocation.py` L289-329: `alloc_req_slots`, which fails loudly.
  - `mem_cache/chunk_cache.py` L82-87 and `mem_cache/unified_radix_cache.py` L1161-1230: `cache_unfinished_req`.
  - `mem_cache/common.py` L172-176: `maybe_cache_unfinished_req`.
  - `dllm/mixin/scheduler.py` L360-400: `process_dllm_incoming_reqs`.
  - `min_free_slots_delayer.py`, the whole file.
  - `arg_groups/fields/schedule.py` L264-279: `--min-free-slots-delay`.

---

## 1. Most important finding: the backport probably does NOT remove our "11 running + 1 queued"

I read the whole admission path. A different 0.5.21 gate, the one that only turns on with DFlash, fits our symptom exactly. #44401 does not touch that gate.

`scheduler.py` L1183-1193:
```python
        # DFlash auto-enables the legacy formula; other workloads opt in via
        # --min-free-slots-delay. Built independently of the prefill delayer.
        self.min_free_slots_delayer: Optional[MinFreeSlotsDelayer] = None
        min_free_slots = resolve_min_free_slots(
            get_schedule().min_free_slots_delay,
            self.max_running_requests,
            is_dflash_family=self.spec_algorithm.is_dflash_family(),
        )
```
`min_free_slots_delayer.py` L17-22 and L37-38:
```python
    if user_value is not None:
        threshold = min(user_value, max_running_requests)
        return threshold if threshold > 1 else None
    if is_dflash_family and max_running_requests >= 8:
        return min(4, max(2, (max_running_requests + 5) // 6))
    return None
...
    def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
        return running_bs > 0 and num_allocatable_reqs < self._min_free_slots
```
`scheduler.py` L3851-3863 runs this check before every fresh admission. It is skipped only while a chunked request is in flight.

What the formula gives on our server (DFLASH, 12 client streams; "rows" means req_to_token rows):

| `--max-running-requests` | threshold | 11 running + 1 waiting: allocatable = min(mrr-11, free rows) | delayed? |
|---|---|---|---|
| 12 | min(4, max(2, 17//6=2)) = **2** | 1 | **yes**: the 12th request waits until a 2nd slot frees |
| 13 | min(4, max(2, 18//6=3)) = **3** | 2 | **yes** |
| 14 | 3 | 3 | no |
| 15 | min(4, max(2, 20//6=3)) = 3 | 4 | no |

So with mrr 12 or 13 and 12 streams, a request that finishes leaves 11 running and the next request queued. The queued request is admitted only after a second request finishes and running drops to 10. Then both are admitted together. That is the "11 running + 1 queued for long stretches" pattern. It is also the 2,358 vs 114 decode-line split noted for E11-05 S49 (mrr 12).

Where #44401 can bind on our setup:

- #44401 only matters on a pass where a continuing chunk is already in `can_run_list`.
- At mrr 13 the double count blocks a waiting request only if 11 running + 1 chunk + 1 waiting are all in the system at once. That is 13 requests, which 12 client streams cannot produce, except for a moment while a finished request's row is still held.
- At mrr 12 it can bind: 10 running + 1 chunk + 1 waiting, 1 free row; the gate in 0.5.21 says full.

The test `test_dflash_min_free_slots_delayer_is_not_changed_by_the_backport` runs the real delayer statements, cut from both the unpatched and the patched scheduler, together with the real `resolve_min_free_slots`. It confirms the table: mrr 12/11 running and 13/11 are delayed; 13/10, 14/11 and 15/11 are not. The result is the same with or without the patch.

Levers that come out of this. I did not apply either one. Each changes the server flags and needs your go:

- **`--min-free-slots-delay 1`** turns the delayer off. L17-19: an explicit value is capped by mrr, and a threshold ≤ 1 returns None. The cost: each DFlash admission is a separate draft prefill. The delayer exists to batch those admissions.
- **`--max-running-requests 14` (or 15)** with the client still at 12 keeps the delayer, but 11 running then leaves 3 allocatable, so the 12th request is admitted. The cost: more req rows and Mamba speculative state. `kv_cache_configurator.py` L582-598 sizes `mamba_spec_state_size` from mrr. The KV/Mamba budget must be checked before a run.

The backport is still correct and worth carrying. It removes a real double count, it is upstream behaviour from v0.5.22 onward, and it matters if we ever run mrr = streams. But the GPU run must not expect it alone to bring running to 12.

---

## 2. What each upstream PR changes

### PR #41872: empty-batch `batch_is_full` latch fix (in v0.5.22)

Upstream diff, `python/sglang/srt/managers/scheduler.py`, hunk `@@ -3942,6 +3942,10 @@ def _get_new_batch_prefill_raw(`:
```diff
         if self.enable_priority_preemption or self.is_hybrid_swa:
             # Reset batch_is_full to try preemption with a prefill adder.
             running_batch.batch_is_full = False
+        if running_batch.is_empty():
+            # An empty batch never reaches update_running_batch, the only
+            # other place that clears the flag, so admission would stay shut.
+            running_batch.batch_is_full = False
```
Hunk `@@ -4127,17 +4131,10 @@`:
```diff
                 if res == AddReqResult.NO_TOKEN:
-                    if (
-                        self.enable_hierarchical_cache
-                        or self.enable_lmcache
-                        or self.enable_unified_cache_external_linker
-                    ):
-                        # Set batch_is_full after making sure there are requests that can be served
-                        running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
-                            not running_batch.is_empty()
-                        )
-                    else:
-                        running_batch.batch_is_full = True
+                    # Set batch_is_full after making sure there are requests that can be served
+                    running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
+                        not running_batch.is_empty()
+                    )
```
Bug in 0.5.21:

- `_get_new_batch_prefill_raw` returns early at L3846-3849 while `running_batch.batch_is_full` is set and no chunk is in flight.
- The flag is set on an empty batch by the slot check at L3870-3876 or by NO_TOKEN at L4029-4040.
- Nothing ever shrinks an empty batch, so the paths that clear the flag never run:
  - filter in `get_next_batch_to_run` L3700-3703
  - prefill-only L3718-3721
  - `update_running_batch` L4197 / L4269
- So admission stays shut forever.
- The upstream test file `test_scheduler_batch_is_full_latch.py` shows this: `ScheduleBatch(reqs=[], batch_is_full=True)` must still reach `policy.calc_priority`.

### PR #44401: count a request's held req_to_token row once at admission (merged 2026-10-15)

Upstream diff, `managers/scheduler.py`, hunk `@@ -3928,9 +3928,12 @@ def get_num_allocatable_reqs(`:
```diff
         running_batch: Optional[ScheduleBatch] = None,
+        num_held_rows: int = 0,
     ) -> int:
         pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
-        available = self.req_to_token_pool.available_size()
+        # Requests the caller counts that already hold a row (a continuing
+        # chunk, a session borrow) need no free one; give their rows back.
+        available = self.req_to_token_pool.available_size() + num_held_rows
```
Hunk `@@ -4114,16 +4117,21 @@ def _get_new_batch_prefill_raw(`:
```diff
+            num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)
             if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                 running_bs,
                 candidate_beam_width,
                 running_batch=running_batch,
+                num_held_rows=num_held_rows,
             ):
...
-                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
+                if (
+                    len(adder.can_run_list) - num_held_rows
+                    >= self.req_to_token_pool.available_size()
+                ):
```
Hunk `@@ -4009,8 +4012,8 @@`, comment only:
```diff
-        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
-        # as the space for the chunked requests has just been released.
+        # In the non-PP case, a chunked request keeps its req_to_token row across chunks,
+        # so continuing it needs no free row.
```
Hunk `@@ -3985,8 +3988,8 @@`, comment only. It rewords the comment that #41872 added:
```diff
-            # An empty batch never reaches update_running_batch, the only
-            # other place that clears the flag, so admission would stay shut.
+            # An empty batch has no request left to leave, and only a leaving
+            # request clears the flag elsewhere, so admission would stay shut.
```
`python/sglang/srt/dllm/mixin/scheduler.py`, hunk `@@ -376,7 +376,10 @@ def process_dllm_incoming_reqs(`:
```diff
-            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
+            num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)
+            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
+                running_bs, num_held_rows=num_held_rows
+            ):
```
There is also a new test file, `test/registered/unit/managers/test_scheduler_held_row_admission.py`: 4 rows, 2 running, a final chunk that holds a row, 1 free row. It expects `can_run_list == [final_chunk, waiting]` and `batch_is_full` False in NULL and PREFILL modes.

### 0.5.21 already has everything #44401 relies on

- **The continuing chunk keeps its row between chunks.**
  - `stash_chunked_request` (L3531-3535) calls `cache_unfinished_req(req, chunked=True)`.
  - `ChunkCache.cache_unfinished_req` (chunk_cache.py L82-87) and `UnifiedRadixCache.cache_unfinished_req` (L1161-) read `req.kv.req_pool_idx` and do not free it.
  - The only free is `ReqToTokenPool.free` (memory_pool.py L360-363), reached through `release_kv_cache` (common.py L292-335).
- **The continuing chunk is in `adder.can_run_list` before the gate runs.** `add_chunked_req` L1166 does `self.can_run_list.append(req)`, called at scheduler.py L3926-3934, before the waiting-queue loop at L3954.
- **Allocation reuses held rows.** `ReqToTokenPool.alloc` (L317-334) does:
  ```python
  reusing = [i for i, r in enumerate(reqs) if r.kv.holds_kv]
  ...
  select_index = self.alloc_rows(len(reqs) - len(reusing))
  ```
  So a held row is never taken from `free_slots` again. `HybridReqToTokenPool.alloc` (L1434-) calls `super().alloc` and likewise reuses `holds_mamba` slots.
- **`holds_kv` exists**: `ReqKvInfo.holds_kv` (schedule_batch.py L968-970).

So the 0.5.21 gate compares `len(can_run_list)`, which includes the chunk, with `available_size()`, which excludes the chunk's row because the row is already taken. The chunk is counted twice. That is exactly the double count #44401 removes. No 0.5.22+ behaviour is needed.

The comment at L3866-3867 in 0.5.21 ("the space for the chunked requests has just been released") is wrong for 0.5.21 itself, so the comment hunk applies as is.

Other callers of `get_num_allocatable_reqs` in 0.5.21, found by grep:

- scheduler.py L3858 (min-free-slots delayer) and L3871 (slot check). Both are reached only when `self.chunked_req is None`, are unchanged upstream, and keep `num_held_rows=0`.
- dllm mixin L243 is unchanged upstream.
- `beam_search/coordinator.py:335` only mentions the gate in a message.

The PP mixin does not call it.

---

## 3. Hunks as ported to 0.5.21

All six hunks were checked against the local 0.5.21 source:

- each `old` occurs exactly once and each `new` zero times;
- no two hunks overlap;
- the two patches are independent, and either order gives identical bytes.

Each hunk carries `old_sha256`; a corrupted embedding is refused.

Patch `41872`:

| # | file | 0.5.21 lines | change | old_sha256 |
|---|---|---|---|---|
| 0 | srt/managers/scheduler.py | L3842-3848 | insert `if running_batch.is_empty(): running_batch.batch_is_full = False` (upstream wording) after the preemption reset, before the early return | `7bc08f1e…5b1e` |
| 1 | srt/managers/scheduler.py | L4028-4040 | NO_TOKEN: drop the hicache/lmcache/linker condition; always `batch_is_full = len(adder.can_run_list) > 0 or (not running_batch.is_empty())` | `17f586b7…343b` |

Patch `44401`:

| # | file | 0.5.21 lines | change | old_sha256 |
|---|---|---|---|---|
| 0 | srt/managers/scheduler.py | L3788-3791 | `get_num_allocatable_reqs(..., num_held_rows: int = 0)`; `available = available_size() + num_held_rows` | `1c2a815a…a228` |
| 1 | srt/managers/scheduler.py | L3865-3867 | comment only: "a chunked request keeps its req_to_token row across chunks" | `a8cf5910…6835` |
| 2 | srt/managers/scheduler.py | L3967-3981 | gate: `num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)`, passed to the call; PREFILL check uses `len(can_run_list) - num_held_rows` | `bda3d801…64c5` |
| 3 | srt/dllm/mixin/scheduler.py | L377-380 | dLLM gate: same count and pass-through (we do not use dLLM; ported because it matches exactly and keeps parity with upstream) | `c3099b40…aeb` |

The resulting diff against the pristine 0.5.21 file, both patches applied, generated with difflib from the real output:
```diff
--- a/srt/dllm/mixin/scheduler.py
+++ b/srt/dllm/mixin/scheduler.py
@@ -377,5 +377,8 @@
             # Check if batch is full
             running_bs = len(running_batch.reqs)
-            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
+            num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)
+            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
+                running_bs, num_held_rows=num_held_rows
+            ):
                 running_batch.batch_is_full = True
 
--- a/srt/managers/scheduler.py
+++ b/srt/managers/scheduler.py
@@ -3787,7 +3787,10 @@
         beam_width: Optional[int] = None,
         running_batch: Optional[ScheduleBatch] = None,
+        num_held_rows: int = 0,
     ) -> int:
         pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
-        available = self.req_to_token_pool.available_size()
+        # Requests the caller counts that already hold a row (a continuing
+        # chunk, a session borrow) need no free one; give their rows back.
+        available = self.req_to_token_pool.available_size() + num_held_rows
 
         active_batch = running_batch or self.running_batch
@@ -3843,4 +3846,8 @@
             # Reset batch_is_full to try preemption with a prefill adder.
             running_batch.batch_is_full = False
+        if running_batch.is_empty():
+            # An empty batch never reaches update_running_batch, the only
+            # other place that clears the flag, so admission would stay shut.
+            running_batch.batch_is_full = False
 
         if (
@@ -3864,6 +3871,6 @@
 
         # Ignore the check if self.chunked_req is not None.
-        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
-        # as the space for the chunked requests has just been released.
+        # In the non-PP case, a chunked request keeps its req_to_token row across chunks,
+        # so continuing it needs no free row.
         # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
         # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
@@ -3969,8 +3976,10 @@
                 req.beam_group.beam_width if req.beam_group is not None else None
             )
+            num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)
             if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                 running_bs,
                 candidate_beam_width,
                 running_batch=running_batch,
+                num_held_rows=num_held_rows,
             ):
                 running_batch.batch_is_full = True
@@ -3978,5 +3987,8 @@
                 # In prefill mode, prealloc queue and transfer queue can also take memory,
                 # so we need to check if the available size for the actual available size.
-                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
+                if (
+                    len(adder.can_run_list) - num_held_rows
+                    >= self.req_to_token_pool.available_size()
+                ):
                     running_batch.batch_is_full = True
 
@@ -4028,15 +4040,8 @@
             if res != AddReqResult.CONTINUE:
                 if res == AddReqResult.NO_TOKEN:
-                    if (
-                        self.enable_hierarchical_cache
-                        or self.enable_lmcache
-                        or self.enable_unified_cache_external_linker
-                    ):
-                        # Set batch_is_full after making sure there are requests that can be served
-                        running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
-                            not running_batch.is_empty()
-                        )
-                    else:
-                        running_batch.batch_is_full = True
+                    # Set batch_is_full after making sure there are requests that can be served
+                    running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
+                        not running_batch.is_empty()
+                    )
                 # revert matched mamba idx to avoid memory leak, if req is not added.
                 # Only free if the slot was freshly allocated in this batch (not
```
sha256 of the files after both patches; the patcher reports `matches_both_patched_0_5_21`:

- `scheduler.py`: `4ae05197445f5be254027a4d2bd2b14f98b9140c8ca3732bae2c199315547fc8`
- `dllm/mixin/scheduler.py`: `d86972c5ef3256779047fe946e0c327b3ccde3389ef6c528ca814e3ce675d636`

## 4. What was left out, and why

- **#44401's rewording of the comment that #41872 adds** (`@@ -3985,8 +3988,8 @@`). The upstream #44401 diff was made on a base that already contained #41872. That hunk's `old` text exists only after #41872 has been applied. Keeping it in `44401` would make the patch refuse whenever `41872` is not applied, which would break the independent, all-or-nothing design. It is a comment-only change with no behaviour, so it is left out. The `41872` hunk inserts the upstream #41872 wording verbatim.
- **Both upstream test files** (`test_scheduler_held_row_admission.py`, `test_scheduler_batch_is_full_latch.py`). The task says the patch does not need them. They also import `sglang.test.*`, `maybe_stub_sgl_kernel` and the full `Scheduler`, which need torch. Our CPU tests cover the same scenarios differently (section 6).
- **Nothing else.** Every behaviour hunk of both PRs is ported. The dLLM hunk is included although we do not use dLLM.

## 5. Risks

1. **The installed wheel must match byte for byte in the hunk regions.** The reference source came from an aarch64 cp310 0.5.21 wheel; Kaggle uses the x86_64 cp312 install in `/tmp/sglang-venv`. The pure-Python files should be the same, but I could not check that on CPU. Guard: `check()` reports `matches_unpatched_0_5_21`, and `apply()` refuses ("refused: source mismatch") with no edit if any hunk does not match exactly once. The server then runs unpatched and the run must log this.
2. **The backport may not change our running count** (section 1). The DFlash min-free-slots delayer is the more likely cause of 11-of-12 at mrr 12 and 13.
3. **More admissions next to a continuing chunk.** Patched, a waiting request can enter the same prefill batch as the final or middle chunk when exactly one row is free. Row allocation is safe: `alloc` reuses the chunk's row, so 1 free row is enough. But the new request also needs a Mamba slot (`HybridReqToTokenPool.alloc` L1456-1459 asserts `mid is not None`) and KV tokens. Those budgets are owned by `PrefillAdder` (Mamba gap reserve, `alloc_group_begin`) and are not changed. `alloc_req_slots` (allocation.py L321-327) raises `RuntimeError("alloc_req_slots runs out of memory…")` if the gate over-admits. That cannot happen by construction, since available + held ≥ len(can_run_list). The GPU log is the proof.
4. **The #41872 NO_TOKEN change.** Without hicache/lmcache/linker (our case), NO_TOKEN used to latch `batch_is_full = True`. Now it latches only if `can_run_list` or the running batch is non-empty. The only changed case is empty batch plus nothing admitted. There the flag is now False and the next pass retries admission; that is the PR's intent. Before, the server could stay shut forever. The cost is at most a retry per scheduler pass while idle.
5. **The PREFILL-disaggregation hunk and the dLLM hunk** are not exercised by our configuration (NULL mode, autoregressive model). They are byte-faithful to upstream.
6. **Patch order and process lifetime.** The scheduler subprocess imports `scheduler.py` at server start, so `apply()` must run before the server is launched. The patcher removes `__pycache__/scheduler.*.pyc` next to every edited file, so no stale bytecode is used. Python would also recompile by mtime.
7. **Backups.** `<file>.e14bak` is written once, before the first edit, and always holds the pristine bytes, even when the two patches are applied in separate calls. `revert()` restores every `.e14bak` under the package, checks the bytes and deletes the backup.

## 6. What the CPU tests verify, and what they cannot

Verified on CPU, 36 tests:

- **Hunk literals:**
  - each `old_sha256` matches;
  - `old` and `new` are not substrings of each other;
  - each `old` occurs once in the 0.5.21 text and each `new` zero times;
  - no overlaps across both patches;
  - the reference files are the unpatched 0.5.21 files (sha and `_version.py`).
- **Apply, check and revert:**
  - **Apply both:** both patches apply; both files compile; `check` then says applied; the sha256 equals the expected patched value; `.e14bak` equals the original; the stale `scheduler.*.pyc` is removed and another stem's pyc is kept.
  - **Order:** 41872→44401 gives the same bytes as 44401→41872.
  - **Second apply:** returns "already applied" for both, with byte-identical files, unchanged mtimes and unchanged backups.
  - **Revert:**
    - every original sha256 is restored, `.e14bak` is removed, and `check` says not_applied again;
    - apply→revert→apply works;
    - a revert with nothing to do is a no-op.
  - **Separate calls:** the backup is still the pristine original after two separate applies.
- **One changed character:** for each of the 6 hunks, one changed letter in its `old` text (the file still compiles) makes that patch "refused: source mismatch". None of its hunks are applied, no backup is made for files only it touches, and the other patch still applies.
- **Other refusals:**
  - a refused patch on its own writes nothing;
  - a half-applied state ("partial") is refused;
  - a duplicated `old` text (2 occurrences) is refused;
  - a missing file is refused;
  - a `new` text that does not compile gives "refused: compile failed", nothing written;
  - a file broken outside the hunks gives "refused: compile failed";
  - a failure after writing (injected) gives "refused: write failed (rolled back)" with bytes restored;
  - an unknown name is refused.
- **Output:** reports pass `json.dumps`, and the CLI `check` works.
- **Semantics.** These run the real patched source statements, cut out of the patched `scheduler.py` with `ast`, not a re-implementation:
  - **`get_num_allocatable_reqs`** (whole function): 13 rows, 11 running, 1 free.
    - Unpatched gives 1. Patched with the default `num_held_rows=0` also gives 1, so the default is unchanged.
    - `num_held_rows=1` gives 2. It is still capped by pp_budget.
    - The unpatched function rejects the keyword.
  - **The admission-gate statements of the waiting-queue loop**, from `running_bs = …` to the PREFILL check, in NULL and PREFILL modes. Setup: 11 running, a continuing chunk that holds a row, 1 free row, 1 waiting.
    - **Unpatched: full**, so the waiting request is blocked. That is the bug.
    - **Patched: not full**, so it is admitted.
    - With [chunk, w1] already admitted and still 1 free row, **patched: full**. There is no over-admission.
    - Six cases where nothing in `can_run_list` holds a row give identical results unpatched and patched.
  - **Entry of `_get_new_batch_prefill_raw`** (#41872): an empty batch latched full with one request waiting.
    - Unpatched returns `(None, rb)`, stays shut and leaves the flag True.
    - Patched reaches admission and clears the flag.
    - A non-empty latched batch still returns early.
    - An empty waiting queue still returns early.
  - **The NO_TOKEN branch** (#41872):
    - Unpatched latches True in all cases.
    - Patched: empty running batch + empty `can_run_list` gives False; any running request or admitted request gives True.
  - **The min-free-slots delayer** (section 1 table): unchanged by the backport.

Cannot be verified on CPU:

- Whether the installed Kaggle file matches. `check()` on the GPU box answers that.
- The real `Scheduler` loop, `PrefillAdder` budgets, Mamba slot accounting, overlap scheduling, DFlash draft prefill, and the whole `_get_new_batch_prefill_raw` run end to end. Those need torch, the model runner and a GPU. The upstream unit tests instantiate `Scheduler` with `maybe_stub_sgl_kernel()`, which needs an importable sglang and torch; our CPU venv has neither.
- Throughput and running-count effects.

## 7. Test results

Command, run from `src/`, with a new `--basetemp` under the scratchpad each time:
```
"D:/ARC AGI 2/_scratch/e14_venv/Scripts/python.exe" -m pytest -q -p no:cacheprovider --noconftest --rootdir=. --confcutdir=. --import-mode=importlib --basetemp=<new dir> test_e14_sglang_patches.py
```
Python 3.14.6, pytest 9.1.1.

- **Final suite: 36 passed, 0 failed, on 6 consecutive runs** (runs 7-12; 25.8-46.5 s each).
- History:
  - Run 1: 29 passed, 1 failed, 30 tests in total. The failure was in the test, not the patcher. My "change one character" picked a letter inside the keyword `or`, which made the whole file invalid Python. The patcher then correctly refused that patch and also the other one, with "refused: compile failed". I changed the test to alter a letter inside a long identifier and added an explicit test for a file broken outside the hunks.
  - Runs 2-6: 31 passed ×5.
  - I then added the patched-sha assertions and the 5-case delayer test.
- The reference tree was unchanged after all runs (sha256 re-checked, no `.e14bak` left there). I removed the two pyc files my runs wrote into `src/__pycache__`. That folder also holds pyc files of other E14 modules, which I left alone.

## 8. Notebook usage (when you give the go)

```python
import json, e14_sglang_patches as SP            # or exec() the embedded text
PKG = "/tmp/sglang-venv/lib/python3.12/site-packages/sglang"
print(json.dumps(SP.check(PKG)))                 # before: both "not_applied", matches_unpatched_0_5_21 true
rep = SP.apply(PKG)                              # BEFORE launching the server
print(json.dumps(rep)); assert rep["ok"], rep    # or fail open: log and continue unpatched
print(json.dumps(SP.check(PKG)))                 # after: both "applied", matches_both_patched_0_5_21 true
# SP.revert(PKG) restores the original bytes
```
Inside the venv, after apply, a cheap import-level proof that the server will load the patched code:
`/tmp/sglang-venv/bin/python -c "import inspect, sglang.srt.managers.scheduler as s; print('num_held_rows' in inspect.signature(s.Scheduler.get_num_allocatable_reqs).parameters)"`

## 9. What the GPU run must check

1. **Patch state.** `check` before and after, printed in the log:
   - before: both `not_applied` and `matches_unpatched_0_5_21`;
   - after: both `applied` and `matches_both_patched_0_5_21`;
   - plus the import-level proof above.
   If it says refused or mismatch, the run is unpatched and must be labelled so.
2. **Running count during chunked prefill.** Among the prefill log lines of passes that carry a continuing chunk, count those where a waiting request is admitted together with the chunk (`#new-seq` ≥ 2 while one request is mid-chunk). Count decode lines with `#running-req` = 12 vs 11. Read the server's own caps from the log (`max_running_requests`, req pool size), as in the "verify real concurrency" rule.
3. **Expect the delayer.** If mrr stays 13 with DFlash, expect 11-of-12 to persist (section 1). To test that, an A/B with `--min-free-slots-delay 1` or mrr 14 is needed, which changes flags and needs your approval. Log the resolved threshold. 0.5.21 has no log line for it, so compute it with `resolve_min_free_slots(None, mrr, True)` from the logged args.
4. **No new errors in the scheduler log.** None of these may appear:
   - `alloc_req_slots runs out of memory`
   - `Not enough space for mamba cache`
   - `a reused row must carry allocated KV`
   - a `RuntimeError` or traceback from `scheduler.py`
   - invariant-checker messages
   - a hang where the queue is non-empty, running is 0 and no prefill happens; that would be the latch #41872 fixes, or a new one.
5. **No negative free-slot accounting.** `ReqToTokenPool.available_size()` is `len(free_slots)` and cannot go negative. An over-admission would surface as the `alloc_req_slots` RuntimeError in item 4 instead. Also check that the token and Mamba usage numbers in the log never exceed capacity.
6. **Output sanity.** The 8-puzzle sanity set still gives 8/8, and there is no rise in error or abort counts versus the last unpatched run.
