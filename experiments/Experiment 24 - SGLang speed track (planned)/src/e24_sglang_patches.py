"""E24 guarded, revertible text patches for an INSTALLED SGLang 0.5.21 package (rev 2, 2026-10-30).

Base: E14's patcher (`Base Engine/src_e14_modules/e14_sglang_patches.py`): file sha256 guards, every hunk's `old` text
pinned by its own sha256, all-or-nothing per patch, `.e14bak` backups of the pristine bytes, atomic writes, a compile
check before and after the write, stale `__pycache__/*.pyc` dropped. E24 adds the five section-6 backports (DESIGN
rev 2 §3.4, `understand/sglang_backports.md` §3; the hunks come from `understand/patches/e24_backport_hunks_0521.py`,
uniqueness/overlap/byte-checked against the 0.5.21 wheel sha256 ac300998…; provenance diffs in `patches/pr*_0521.diff`):
- "41872"  E14: an empty running batch never keeps admission closed (scheduler.py).
- "44401"  E14: a request's held req_to_token row is counted once at admission (scheduler.py, dllm/mixin/scheduler.py).
- "43567"  PR #43567 (v0.5.22): invalidate the lazy keep slot's checkpoint length when the optional pending slot cannot
           be allocated (schedule_batch.py). ONE definition: E14's hunks (the generated E24 entry anchors on more
           context but gives byte-identical files).
- "43559"  PR #43559 (no release yet), PORTED: upstream keeps the tree lock in `req.lock` (TreeLock), which 0.5.21 does
           not have; the port uses req.last_node / req.lock_receipt / req.swa_prefix_lock_released. The upstream text
           applies cleanly to 0.5.21 but raises AttributeError exactly where the assert fired: NEVER use it on 0.5.21.
           Requires 43567 (upstream: releasing the lock alone exposes the stale in-place donation).
- "43029"  PR #43029 (v0.5.22): spec boundary check uses the grammar-committed length (batch_result_processor.py).
- "41736"  PR #41736 (v0.5.22): DFlash grouped-conv taps stay inside each request's block (models/dflash.py).
- "41737"  PR #41737 (v0.5.22): DFlash2 greedy selector walk stays in range on all-NaN rows (kernels/.../dflash.py).
The five E24 files are disjoint from each other and from 44401/41872: any order gives the same bytes.

API (stdlib only; never imports sglang; every function returns a plain JSON-serializable dict and never raises on a
source problem):
- NAMES, DEFAULT_NAMES, E24_NAMES, REQUIRES, PATCHES, REFERENCE_FILE_SHA256, REFERENCE_PATCHED_SHA256
- check(site, names=None)        names=None -> all 7. {"version", "names": {n: applied|not_applied|conflict|unknown},
                                 "files": {rel: sha256|None}, "file_info", "patches", "files_verified", "ok"}
- apply(site, names, strict=True) dependency order, refuses 43559 unless 43567 is applied or requested; strict: the
                                 touched files must verify before and the package check must be ok after; all-or-nothing
                                 across the call (a failure restores every file of the call); returns a post check
- revert(site, names=None)       inverse hunks (dependents first; refuses 43567 while 43559 stays applied); removes a
                                 backup once its file equals it again; verifies the pristine 0.5.21 sha
- restore_backups(site)          E14's revert: restore every .e14bak byte-identical (recovery only)
- state_ok(chk, expect_applied)  exactly the expected set applied, the other names not applied, files verified
- site_dir(vpy)                  the package dir of `sglang` in that interpreter via importlib find_spec (no import)
CLI: python e24_sglang_patches.py {check|apply|revert} (--site DIR | --vpy PY | DIR) [--names a,b | name ...]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

BACKUP_SUFFIX = ".e14bak"          # shared with E14 so either patcher can see the other's backups
TMP_SUFFIX = ".e14tmp"
DEFAULT_NAMES = ("44401", "41872")  # E14's defaults (independent hunks)
E24_NAMES = ("43567", "43559", "43029", "41736", "41737")
NAMES = DEFAULT_NAMES + E24_NAMES   # every name check() covers by default (DESIGN rev 2 §3.4)
REQUIRES = {"43559": ("43567",)}
REFERENCE_VERSION = "0.5.21"

_SCHED = "srt/managers/scheduler.py"
_DLLM = "srt/dllm/mixin/scheduler.py"
_SBATCH = "srt/managers/schedule_batch.py"
_MAMBA = "srt/mem_cache/unified_cache/components/mamba.py"
_BRP = "srt/managers/scheduler_components/batch_result_processor.py"
_DFLASH_MODEL = "srt/models/dflash.py"
_DFLASH_KERNEL = "kernels/ops/speculative/dflash.py"

# sha256 of the whole, unpatched 0.5.21 files (sglang-0.5.21-cp312 manylinux wheel = the Kaggle wheelhouse file).
REFERENCE_FILE_SHA256 = {
    _SCHED: "b61df5b750adc748a99ceb381b8494c199fb3f704d793da5a0217bd6bbb606c4",
    _DLLM: "29628477323d55eb15c6bada659588ae690080418dd70ea74b260ef00e9c5817",
    _SBATCH: "b2f78c781de88c36ce27cc5d44b343420338b4da299e885c575665d6fab843ca",
    _MAMBA: "c1086e3ab077a2f75966f8455b77346ead9510497e9012094805f6dd5f29d9cc",
    _BRP: "2f5baf667e6879e96b0fd4435750778d3989d03674e01c2963a08bed32091736",
    _DFLASH_MODEL: "10cba6aeac6986537114a7cc26df296d308e45c849c18f690d451bf63a9c4871",
    _DFLASH_KERNEL: "21f11619531f493bbe9f71d465912ca707f2d722991832e446ff65aecb40528c",
}

# sha256 with EVERY name touching the file applied (scheduler.py: 44401+41872 in either order).
REFERENCE_PATCHED_SHA256 = {
    _SCHED: "4ae05197445f5be254027a4d2bd2b14f98b9140c8ca3732bae2c199315547fc8",
    _DLLM: "d86972c5ef3256779047fe946e0c327b3ccde3389ef6c528ca814e3ce675d636",
    _SBATCH: "5a98dd8df27274ac484a9890e99d7f863736c62a25d1772e517588350677eda1",
    _MAMBA: "e91cf0352b22d6919b03e9d16afa046e8ba3e62cb8c92c45b4c59ba57b4dafc7",
    _BRP: "87eb43b43eb783e5d6d166967c9114a88d55d6c51633cd3db2f35f49f7ca1410",
    _DFLASH_MODEL: "d201e015156a03f5966e1c23871ac3fd578f09c54ccec67b21bce08faf5f2200",
    _DFLASH_KERNEL: "fd1cb8f84a83a1b17ce57607371e5a3e89f82a1b6d109160566da260a0e3cb4a",
}

# ------------------------------------------------------------------------------------------------- PR #43567 (E14)
# 0.5.21 srt/managers/schedule_batch.py L3489-3493 and L3524-3527 (the exact upstream diff, whitespace-identical)
_43567_PREALLOC_OLD = '            if new_slot is not None:\n                pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])\n                req.kv.mamba_next_track_idx = other_idx\n\n    def mamba_lazy_spec_prepare(self, mamba_track_interval: int, max_draft_tokens: int):\n'
_43567_PREALLOC_NEW = "            if new_slot is not None:\n                pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])\n                req.kv.mamba_next_track_idx = other_idx\n            else:\n                # The in-flight write invalidates the keep slot's old checkpoint.\n                req.kv.mamba_last_track_seqlen = None\n\n    def mamba_lazy_spec_prepare(self, mamba_track_interval: int, max_draft_tokens: int):\n"
_43567_SPEC_OLD = '                if new_slot is not None:\n                    pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])\n                    has_pending = True\n            # On failure the verify scatters in place into the keep slot.\n'
_43567_SPEC_NEW = '                if new_slot is not None:\n                    pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])\n                    has_pending = True\n                else:\n                    req.kv.mamba_last_track_seqlen = None\n            # On failure the verify scatters in place into the keep slot.\n'

# ------------------------------------------------------------------------------------------------- PR #41872 (E14)
# 0.5.21 srt/managers/scheduler.py L3842-3848 (start of _get_new_batch_prefill_raw).
_41872_EMPTY_OLD = """        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            running_batch.batch_is_full = False

        if (
            running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
"""
_41872_EMPTY_NEW = """        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            running_batch.batch_is_full = False
        if running_batch.is_empty():
            # An empty batch never reaches update_running_batch, the only
            # other place that clears the flag, so admission would stay shut.
            running_batch.batch_is_full = False

        if (
            running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
"""

# 0.5.21 srt/managers/scheduler.py L4028-4040 (NO_TOKEN branch of the waiting-queue loop).
_41872_NOTOKEN_OLD = """            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if (
                        self.enable_hierarchical_cache
                        or self.enable_lmcache
                        or self.enable_unified_cache_external_linker
                    ):
                        # Set batch_is_full after making sure there are requests that can be served
                        running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
                            not running_batch.is_empty()
                        )
                    else:
                        running_batch.batch_is_full = True
"""
_41872_NOTOKEN_NEW = """            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    # Set batch_is_full after making sure there are requests that can be served
                    running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
                        not running_batch.is_empty()
                    )
"""

# ------------------------------------------------------------------------------------------------- PR #44401 (E14)
# 0.5.21 srt/managers/scheduler.py L3788-3791 (get_num_allocatable_reqs signature and `available`).
_44401_ALLOC_OLD = """        running_batch: Optional[ScheduleBatch] = None,
    ) -> int:
        pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
        available = self.req_to_token_pool.available_size()
"""
_44401_ALLOC_NEW = """        running_batch: Optional[ScheduleBatch] = None,
        num_held_rows: int = 0,
    ) -> int:
        pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
        # Requests the caller counts that already hold a row (a continuing
        # chunk, a session borrow) need no free one; give their rows back.
        available = self.req_to_token_pool.available_size() + num_held_rows
"""

# 0.5.21 srt/managers/scheduler.py L3865-3867 (comment only; the old one was wrong for 0.5.21 too).
_44401_COMMENT_OLD = """        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
"""
_44401_COMMENT_NEW = """        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, a chunked request keeps its req_to_token row across chunks,
        # so continuing it needs no free row.
"""

# 0.5.21 srt/managers/scheduler.py L3967-3981 (admission gate in the waiting-queue loop).
_44401_GATE_OLD = """            running_bs = len(running_batch.reqs)
            candidate_beam_width = (
                req.beam_group.beam_width if req.beam_group is not None else None
            )
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                running_bs,
                candidate_beam_width,
                running_batch=running_batch,
            ):
                running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    running_batch.batch_is_full = True
"""
_44401_GATE_NEW = """            running_bs = len(running_batch.reqs)
            candidate_beam_width = (
                req.beam_group.beam_width if req.beam_group is not None else None
            )
            num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                running_bs,
                candidate_beam_width,
                running_batch=running_batch,
                num_held_rows=num_held_rows,
            ):
                running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if (
                    len(adder.can_run_list) - num_held_rows
                    >= self.req_to_token_pool.available_size()
                ):
                    running_batch.batch_is_full = True
"""

# 0.5.21 srt/dllm/mixin/scheduler.py L377-380 (process_dllm_incoming_reqs gate; unused by our AR model).
_44401_DLLM_OLD = """            # Check if batch is full
            running_bs = len(running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                running_batch.batch_is_full = True
"""
_44401_DLLM_NEW = """            # Check if batch is full
            running_bs = len(running_batch.reqs)
            num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                running_bs, num_held_rows=num_held_rows
            ):
                running_batch.batch_is_full = True
"""

# ------------------------------------------------------------------------------------------------- PR #43559 (PORTED)
# 0.5.21 srt/mem_cache/unified_cache/components/mamba.py: L1-6 (import), L500-510 (_alloc_mamba_slot gets `req` and
# releases this request's consumed Mamba prefix lock before the eviction retry, lazy strategy only), L583-589 and
# L596-609 (the three prepare_for_caching_req call sites pass `req`). Only the lock block differs from upstream.
_43559_IMPORT_OLD = 'from __future__ import annotations\n\nfrom collections import defaultdict\nfrom typing import TYPE_CHECKING, Callable, Optional, Sequence\n\nimport torch\n'
_43559_IMPORT_NEW = 'from __future__ import annotations\n\nfrom collections import defaultdict\nfrom dataclasses import replace\nfrom typing import TYPE_CHECKING, Callable, Optional, Sequence\n\nimport torch\n'
_43559_ALLOC_OLD = '        if cd.lock_ref == 0:\n            self.tree_core._update_evictable_leaf_sets(node)\n\n    def _alloc_mamba_slot(self) -> torch.Tensor:\n        """Allocate one mamba pool slot, evicting if necessary."""\n        slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)\n        if slot is None:\n            self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))\n            slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)\n            assert slot is not None, "Can not alloc mamba cache"\n'
_43559_ALLOC_NEW = '        if cd.lock_ref == 0:\n            self.tree_core._update_evictable_leaf_sets(node)\n\n    def _alloc_mamba_slot(self, req: Optional[Req] = None) -> torch.Tensor:\n        """Allocate one mamba pool slot, evicting if necessary."""\n        slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)\n        if slot is None:\n            if req is not None and get_exec().mamba.enable_mamba_extra_buffer_lazy:\n                ct = self.component_type\n                if (\n                    req.last_node is not None\n                    and not req.swa_prefix_lock_released\n                    and ct not in req.lock_receipt.skipped_lock_components\n                ):\n                    # Prefill has consumed the COW source; retain only its KV locks.\n                    self.cache.dec_lock_ref(\n                        req.last_node,\n                        replace(\n                            req.lock_receipt,\n                            skipped_lock_components=tuple(\n                                c for c in self.cache.tree_components if c != ct\n                            ),\n                        ),\n                    )\n                    req.lock_receipt.skipped_lock_components += (ct,)\n            self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))\n            slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)\n            assert slot is not None, "Can not alloc mamba cache"\n'
_43559_INT8_CALL_OLD = '            # Donate the mamba index to the radix cache instead of copying.\n            if self.int8_ckpt_pool is not None:\n                if self.cache.enable_mamba_extra_buffer:\n                    new_slot = self._alloc_mamba_slot()\n                    src_active = (\n                        self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(\n                            req, new_slot\n'
_43559_INT8_CALL_NEW = '            # Donate the mamba index to the radix cache instead of copying.\n            if self.int8_ckpt_pool is not None:\n                if self.cache.enable_mamba_extra_buffer:\n                    new_slot = self._alloc_mamba_slot(req)\n                    src_active = (\n                        self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(\n                            req, new_slot\n'
_43559_CALLS_OLD = '                        req.kv.mamba_pool_idx.view(-1)\n                    )\n            elif self.cache.enable_mamba_extra_buffer:\n                new_slot = self._alloc_mamba_slot()\n                mamba_value_donated = (\n                    self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(\n                        req, new_slot\n                    )\n                )\n            else:\n                mamba_value_donated = self._alloc_mamba_slot()\n                # mamba_pool is a pure PHYSICAL store; translate both slot ids\n                # virtual->physical (identity for the non-unified memory pool) first.\n                translate = self.cache.req_to_token_pool.translate_mamba_indices\n'
_43559_CALLS_NEW = '                        req.kv.mamba_pool_idx.view(-1)\n                    )\n            elif self.cache.enable_mamba_extra_buffer:\n                new_slot = self._alloc_mamba_slot(req)\n                mamba_value_donated = (\n                    self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(\n                        req, new_slot\n                    )\n                )\n            else:\n                mamba_value_donated = self._alloc_mamba_slot(req)\n                # mamba_pool is a pure PHYSICAL store; translate both slot ids\n                # virtual->physical (identity for the non-unified memory pool) first.\n                translate = self.cache.req_to_token_pool.translate_mamba_indices\n'

# ------------------------------------------------------------------------------------------------- PR #43029
# 0.5.21 srt/managers/scheduler_components/batch_result_processor.py L1522-1528 (_mamba_check_track_boundary, spec).
_43029_BOUNDARY_OLD = '                return True, committed_len\n        elif result.num_correct_drafts_per_req_cpu is not None:\n            cur = req.seqlen - 1\n            prev = cur - result.num_correct_drafts_per_req_cpu[i] - 1\n            if cur // interval != prev // interval:\n                return True, cur // interval * interval\n\n'
_43029_BOUNDARY_NEW = "                return True, committed_len\n        elif result.num_correct_drafts_per_req_cpu is not None:\n            cur = req.seqlen - 1\n            num_non_draft = result.num_non_draft_tokens_per_req\n            retained = result.grammar_retained_tokens\n            if retained is not None and retained[i] is not None:\n                # NOTE(kpham-sgl): grammar-truncated run, minus non-draft\n                # tokens ahead of the tracked verify window (UNO's root).\n                num_committed = max(len(retained[i]) - (num_non_draft - 1), 0)\n            else:\n                num_committed = result.num_correct_drafts_per_req_cpu[i] + 1\n            prev = cur - num_committed\n            if cur // interval != prev // interval:\n                return True, cur // interval * interval\n\n"

# ------------------------------------------------------------------------------------------------- PR #41736
# 0.5.21 srt/models/dflash.py L473-479 (_grouped_conv: where-select instead of a 0/1 multiply, NaN*0 = NaN).
_41736_TAP_OLD = '        position = position % block_size\n    for tap in range(1, taps):\n        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))\n        out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)\n    return out.flatten(-2)\n\n\n'
_41736_TAP_NEW = '        position = position % block_size\n    for tap in range(1, taps):\n        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))\n        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0)\n        out = out + coefficients[:, tap] * shifted\n    return out.flatten(-2)\n\n\n'

# ------------------------------------------------------------------------------------------------- PR #41737
# 0.5.21 kernels/ops/speculative/dflash.py L271-278 (_selector_walk_kernel greedy branch: torch.argmax's NaN rule).
_41737_GREEDY_OLD = '            tl.float32\n        )\n        if greedy:\n            best = tl.max(scores, axis=0)\n            index = tl.min(tl.where(scores == best, offsets, top_k), axis=0)\n            probabilities = tl.where(offsets == index, 1.0, 0.0)\n        else:\n            scaled = scores / temperature\n'
_41737_GREEDY_NEW = '            tl.float32\n        )\n        if greedy:\n            # Same pick as torch.argmax: the first NaN if any, else the first max.\n            is_nan = scores != scores\n            best = tl.max(tl.where(is_nan, -float("inf"), scores), axis=0)\n            has_nan = tl.max(is_nan.to(tl.int32), axis=0) > 0\n            hit = tl.where(has_nan, is_nan, scores == best)\n            index = tl.min(tl.where(hit, offsets, top_k), axis=0)\n            probabilities = tl.where(offsets == index, 1.0, 0.0)\n        else:\n            scaled = scores / temperature\n'


def _h(file, old, new, old_sha256):
    return {"file": file, "old": old, "new": new, "old_sha256": old_sha256}


PATCHES = {
    "41872": [
        _h(_SCHED, _41872_EMPTY_OLD, _41872_EMPTY_NEW, "7bc08f1e0e763b0c6b738686783aa70cbb700aa79f5612a04b16eca22d6f5b1e"),
        _h(_SCHED, _41872_NOTOKEN_OLD, _41872_NOTOKEN_NEW, "17f586b7c5a6baef1ebbf8b662c2f2beea082b5f513dd03ae0e0d99b511a343b"),
    ],
    "43567": [
        _h(_SBATCH, _43567_PREALLOC_OLD, _43567_PREALLOC_NEW, "bce78b2ab8f978033d1a8f1cfb73bac8cec8c1c775a46900c01916d006cb3b97"),
        _h(_SBATCH, _43567_SPEC_OLD, _43567_SPEC_NEW, "26e249bfb527571f9e4c6933e8ffd5041d3d1aa79215d8c1b1d3b615e5ef6a16"),
    ],
    "44401": [
        _h(_SCHED, _44401_ALLOC_OLD, _44401_ALLOC_NEW, "1c2a815a61a49f3a239698680c49f081fbfecfe0270f756647f579122d52a228"),
        _h(_SCHED, _44401_COMMENT_OLD, _44401_COMMENT_NEW, "a8cf5910f52f02404c677e5eca089c74a4d4de7b4c6a18964ca84dc39c826835"),
        _h(_SCHED, _44401_GATE_OLD, _44401_GATE_NEW, "bda3d8015142de899644bad5b41710d0e8a40420763b9b88ac1913aeafaf64c5"),
        _h(_DLLM, _44401_DLLM_OLD, _44401_DLLM_NEW, "c3099b40a270bcf0601ddc6c08d167875fe83da267ea960bd241e7315aacbaeb"),
    ],
    "43559": [
        _h(_MAMBA, _43559_IMPORT_OLD, _43559_IMPORT_NEW, "cad4419511705dfa85db0e4db1295ec2d412b8a34c6010da560989b8c33f1aad"),
        _h(_MAMBA, _43559_ALLOC_OLD, _43559_ALLOC_NEW, "c6000d0546735a7ace768a17ef4b35583834a9c7ee9ad9ca1c0262d4e4820301"),
        _h(_MAMBA, _43559_INT8_CALL_OLD, _43559_INT8_CALL_NEW, "9374f01b761ffa55a425b122a58888f66149495df8e45a729f0b1154aba81868"),
        _h(_MAMBA, _43559_CALLS_OLD, _43559_CALLS_NEW, "f9bd1ec50820cabdf049d5474ca2ceb05a7063d835fdd1e5028a2721b233532a"),
    ],
    "43029": [
        _h(_BRP, _43029_BOUNDARY_OLD, _43029_BOUNDARY_NEW, "9d91d5a845d2652daebdf2e37a2f73afa93b301b921a2da83c7424b61b2a6a4a"),
    ],
    "41736": [
        _h(_DFLASH_MODEL, _41736_TAP_OLD, _41736_TAP_NEW, "f844ffb658d208c0ecd42088d5dbd0d7d60e7d0bc15cae4ffd2f0eb3b7f057ef"),
    ],
    "41737": [
        _h(_DFLASH_KERNEL, _41737_GREEDY_OLD, _41737_GREEDY_NEW, "6416450f702bf9e324b47acaa431d692206b39edfa2bca1b07c8b9fc53a4cf07"),
    ],
}

_FILES_OF = {n: sorted({h["file"] for h in hs}) for n, hs in PATCHES.items()}
_NAMES_OF_FILE = {}
for _n in NAMES:
    for _f in _FILES_OF[_n]:
        _NAMES_OF_FILE.setdefault(_f, []).append(_n)


# ------------------------------------------------------------------------------------------------- helpers
def _sha(b):
    if isinstance(b, str):
        b = b.encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _path(site, rel):
    return os.path.join(str(site), *rel.split("/"))


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _write_atomic(path, data):
    tmp = path + TMP_SUFFIX
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    except OSError:
        pass
    os.replace(tmp, path)


def _write_atomic_new(path, data):
    """Write `data` to a path that must not exist yet (the backup)."""
    tmp = path + TMP_SUFFIX
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.remove(tmp)
        raise OSError("backup appeared concurrently: %s" % path)
    os.replace(tmp, path)
    if _read(path) != data:
        raise OSError("backup verify failed: %s" % path)


def _compile_check(text, path):
    """None if `text` compiles as a module, else the error string."""
    try:
        compile(text, path, "exec", dont_inherit=True)
        return None
    except Exception as e:  # SyntaxError, ValueError (null bytes), ...
        return "%s: %s" % (type(e).__name__, e)


def _drop_pyc(path):
    """Delete __pycache__/<stem>.*.pyc next to `path` so no stale bytecode is used."""
    removed = []
    d, base = os.path.split(path)
    stem = os.path.splitext(base)[0]
    cache = os.path.join(d, "__pycache__")
    try:
        names = os.listdir(cache)
    except OSError:
        return removed
    for n in names:
        if n.startswith(stem + ".") and n.endswith(".pyc"):
            try:
                os.remove(os.path.join(cache, n))
                removed.append(n)
            except OSError:
                pass
    return removed


def _version(site):
    try:
        txt = _read(_path(site, "_version.py")).decode("utf-8", "replace")
        m = re.search(r"__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)['\"]", txt) or re.search(
            r"__version__\s*=\s*['\"]([^'\"]+)['\"]", txt)
        return m.group(1) if m else None
    except OSError:
        return None


def _resolve(names):
    """None -> every known name; 'a,b' or ['a', 'b'] -> list (order kept, duplicates dropped)."""
    if names is None:
        return list(NAMES)
    if isinstance(names, str):
        names = [x for x in re.split(r"[,\s]+", names) if x]
    out = []
    for n in names:
        n = str(n).strip()
        if n and n not in out:
            out.append(n)
    return out


def _dep_order(names):
    """Dependencies first (43567 before 43559), otherwise NAMES order, unknown names last."""
    known = [n for n in NAMES if n in names]
    out = []

    def visit(n):
        if n in out:
            return
        for d in REQUIRES.get(n, ()):
            if d in known:
                visit(d)
        out.append(n)

    for n in known:
        visit(n)
    return out + [n for n in names if n not in NAMES]


def _hunk_state(text, hunk):
    """State of one hunk against `text` (None = file missing)."""
    info = {"file": hunk["file"], "old_sha_ok": _sha(hunk["old"]) == hunk["old_sha256"]}
    if not info["old_sha_ok"]:
        info.update(status="mismatch", reason="hunk text corrupted (old_sha256 differs)")
        return info
    if text is None:
        info.update(status="mismatch", reason="file missing")
        return info
    n_old, n_new = text.count(hunk["old"]), text.count(hunk["new"])
    info.update(old_count=n_old, new_count=n_new)
    if n_old == 1 and n_new == 0:
        info["status"] = "applicable"
    elif n_old == 0 and n_new == 1:
        info["status"] = "applied"
    else:
        info["status"] = "mismatch"
        info["reason"] = "old occurs %d times, new occurs %d times (need exactly one of them once)" % (n_old, n_new)
    return info


def _load_texts(site, files):
    """rel -> (raw bytes, decoded text) or None if unreadable."""
    out = {}
    for rel in files:
        try:
            raw = _read(_path(site, rel))
            out[rel] = (raw, raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError):
            out[rel] = None
    return out


def _patch_state(name, texts):
    """E14 states: not_applied | applied | mismatch | partial (+ per-hunk detail)."""
    states = []
    for i, h in enumerate(PATCHES[name]):
        t = texts.get(h["file"])
        s = _hunk_state(t[1] if t else None, h)
        s["index"] = i
        states.append(s)
    kinds = {s["status"] for s in states}
    if kinds == {"applicable"}:
        state = "not_applied"
    elif kinds == {"applied"}:
        state = "applied"
    elif "mismatch" in kinds:
        state = "mismatch"
    else:
        state = "partial"
    return state, states


def _public_state(state):
    return state if state in ("applied", "not_applied") else "conflict"


def _transform(text, hunks, forward):
    """Apply (forward) or undo every hunk of one patch in memory; None + reason if a hunk is not unique."""
    seq = hunks if forward else list(reversed(hunks))
    for h in seq:
        src, dst = (h["old"], h["new"]) if forward else (h["new"], h["old"])
        if text.count(src) != 1 or text.count(dst) != 0:
            return None, "hunk %d not unique (%d / %d)" % (hunks.index(h), text.count(src), text.count(dst))
        text = text.replace(src, dst, 1)
    return text, None


def _file_verify(site, rel, raw, states):
    """Is `raw` exactly (pristine 0.5.21 + the hunks of the names currently applied to this file)?"""
    applied = [n for n in _NAMES_OF_FILE.get(rel, []) if states.get(n) == "applied"]
    other = [n for n in _NAMES_OF_FILE.get(rel, []) if states.get(n) not in ("applied", "not_applied")]
    ref = REFERENCE_FILE_SHA256.get(rel)
    info = {"applied": applied}
    if raw is None:
        info.update(verified=False, reason="file missing")
        return info
    if other:
        info.update(verified=False, reason="patch state conflict: %s" % ",".join(other))
        return info
    if not applied:
        info.update(verified=_sha(raw) == ref, reason=None if _sha(raw) == ref else "not the pristine 0.5.21 file")
        return info
    # pristine bytes: the backup if it is the 0.5.21 file, else nothing to rebuild from
    base = None
    bak = _path(site, rel) + BACKUP_SUFFIX
    try:
        b = _read(bak)
        if _sha(b) == ref:
            base = b.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        pass
    if base is None:
        if set(applied) == set(_NAMES_OF_FILE.get(rel, [])) and _sha(raw) == REFERENCE_PATCHED_SHA256.get(rel):
            info.update(verified=True, reason="all-patched reference sha")
        else:
            info.update(verified=False, reason="no pristine backup to rebuild the expected bytes from")
        return info
    text = base
    for n in applied:
        hs = [h for h in PATCHES[n] if h["file"] == rel]
        text, err = _transform(text, hs, True)
        if text is None:
            info.update(verified=False, reason="rebuild failed: %s %s" % (n, err))
            return info
    ok = text.encode("utf-8") == raw
    info.update(verified=ok, reason=None if ok else "differs from pristine + applied hunks")
    return info


# ------------------------------------------------------------------------------------------------- API
def check(site, names=None):
    """Per-name state for every known name (or `names`), file sha256s, and whether every file is exactly the pristine
    0.5.21 file plus the hunks of its applied names. No writes."""
    names = _resolve(names)
    known = [n for n in names if n in PATCHES]
    rep = {"site": str(site), "version": _version(site), "names": {}, "files": {}, "file_info": {}, "patches": {},
           "files_verified": True, "ok": True}
    files = sorted({f for n in known for f in _FILES_OF[n]})
    # file verification needs the state of every name touching a file, checked or not
    all_files_names = sorted({n for f in files for n in _NAMES_OF_FILE.get(f, [])} | set(known))
    texts = _load_texts(site, sorted({f for n in all_files_names for f in _FILES_OF[n]}))
    states = {}
    for n in all_files_names:
        st, hs = _patch_state(n, texts)
        states[n] = st
        if n in known:
            rep["patches"][n] = {"state": st, "hunks": hs}
            rep["names"][n] = _public_state(st)
    for n in names:
        if n not in PATCHES:
            rep["names"][n] = "unknown"
            rep["patches"][n] = {"state": "unknown patch", "hunks": []}
    for rel in files:
        t = texts.get(rel)
        raw = t[0] if t else None
        sha = _sha(raw) if raw is not None else None
        rep["files"][rel] = sha
        info = {"exists": raw is not None, "sha256": sha, "pristine_0_5_21": sha == REFERENCE_FILE_SHA256.get(rel),
                "all_patched_0_5_21": sha == REFERENCE_PATCHED_SHA256.get(rel),
                "backup_exists": os.path.exists(_path(site, rel) + BACKUP_SUFFIX)}
        info.update(_file_verify(site, rel, raw, states))
        rep["file_info"][rel] = info
        if not info["verified"]:
            rep["files_verified"] = False
    for d, deps in REQUIRES.items():
        if states.get(d) == "applied" and any(states.get(x) != "applied" for x in deps):
            rep.setdefault("errors", []).append("%s applied without %s" % (d, ",".join(deps)))
    rep["ok"] = (all(v in ("applied", "not_applied") for v in rep["names"].values()) and rep["files_verified"]
                 and not rep.get("errors"))
    return rep


def state_ok(chk, expect_applied=(), strict=True):
    """True iff `chk` (from check(site) over all 7 names) shows exactly `expect_applied` applied, every other known
    name not applied, the REQUIRES pairs satisfied and (strict) every file verified. Fail-closed on anything else."""
    try:
        exp = set(_resolve(list(expect_applied)))
        if not exp <= set(NAMES):
            return False
        if any(d in exp and not set(deps) <= exp for d, deps in REQUIRES.items()):
            return False
        names = chk.get("names") or {}
        for n in NAMES:
            if names.get(n) != ("applied" if n in exp else "not_applied"):
                return False
        if strict and not chk.get("files_verified"):
            return False
        return bool(chk.get("ok"))
    except Exception:
        return False


def apply(site, names, strict=True):
    """Apply `names` in dependency order. Refuses 43559 unless 43567 is applied or requested. strict (default): every
    touched file must verify (pristine 0.5.21 + known hunks) before, and the whole package check must be ok after.
    All-or-nothing across the call: if any name fails (or the strict post-check), every file written by this call gets
    its pre-call bytes back and a backup created by this call is removed. Never raises for source problems."""
    names = _resolve(names)
    rep = {"site": str(site), "version": _version(site), "requested": names, "order": [], "results": {}, "ok": True,
           "strict": bool(strict)}
    unknown = [n for n in names if n not in PATCHES]
    if unknown:
        rep.update(ok=False, error="refused: unknown patch name(s) %s" % ",".join(unknown))
        rep["check"] = check(site)
        return rep
    before = check(site)
    for n in names:
        for d in REQUIRES.get(n, ()):
            if d not in names and before["names"].get(d) != "applied":
                rep.update(ok=False, error="refused: %s requires %s (apply them together; never %s alone)" % (n, d, n))
                rep["check"] = before
                return rep
    order = _dep_order(names)
    rep["order"] = order
    files = sorted({f for n in order for f in _FILES_OF[n]})
    if strict:
        bad = [rel for rel in files if not before["file_info"].get(rel, {}).get("verified")]
        if bad:
            rep.update(ok=False, error="refused: not the pristine 0.5.21 file plus known hunks: %s (nothing written)"
                       % ", ".join(bad))
            rep["check"] = before
            return rep
    pre = {}
    for rel in files:
        try:
            pre[rel] = _read(_path(site, rel))
        except OSError:
            pre[rel] = None
    created_baks = []
    for n in order:
        res = _apply_one(site, n)
        rep["results"][n] = res
        created_baks += [rel for rel, f in res.get("files", {}).items() if f.get("backup_created")]
        if res["status"] not in ("applied", "already applied"):
            rep["ok"] = False
            rep["error"] = "%s: %s" % (n, res["status"])
            break
    if rep["ok"] and strict:
        post = check(site)
        if not post["ok"]:
            rep.update(ok=False, error="post-apply check failed: %s" % (post.get("errors") or "files not verified"))
    if not rep["ok"]:
        rolled, errs = [], []
        for rel in files:
            p = _path(site, rel)
            try:
                if pre[rel] is not None and _read(p) != pre[rel]:
                    _write_atomic(p, pre[rel])
                    rolled.append(rel)
                _drop_pyc(p)
                if rel in created_baks and pre[rel] is not None and _read(p + BACKUP_SUFFIX) == pre[rel]:
                    os.remove(p + BACKUP_SUFFIX)
            except OSError as e:
                errs.append("%s: %s" % (rel, e))
        rep["rolled_back"] = rolled
        if errs:
            rep["rollback_errors"] = errs
    rep["check"] = check(site)
    return rep


def _apply_one(site, name):
    hunks = PATCHES[name]
    files = _FILES_OF[name]
    texts = _load_texts(site, files)
    state, states = _patch_state(name, texts)
    res = {"status": None, "state_before": state, "hunks": states, "files": {}}
    if state == "applied":
        res["status"] = "already applied"
        return res
    if state != "not_applied":
        res["status"] = "refused: source mismatch"
        res["reason"] = "%s: every hunk must match exactly once (no partial edit was made)" % state
        return res

    # Build every edited text in memory first.
    new_texts = {}
    for rel in files:
        new_texts[rel], err = _transform(texts[rel][1], [h for h in hunks if h["file"] == rel], True)
        if err:
            res["status"] = "refused: source mismatch"
            res["reason"] = "%s: %s" % (rel, err)
            return res
    for rel in files:
        err = _compile_check(new_texts[rel], _path(site, rel))
        if err:
            res["status"] = "refused: compile failed"
            res["reason"] = "%s: %s (nothing written)" % (rel, err)
            return res

    # Backups of the pristine original (kept if one already exists from an earlier patch).
    try:
        for rel in files:
            bak = _path(site, rel) + BACKUP_SUFFIX
            if not os.path.exists(bak):
                _write_atomic_new(bak, texts[rel][0])
                res["files"].setdefault(rel, {})["backup_created"] = True
    except OSError as e:
        res["status"] = "refused: backup failed"
        res["reason"] = str(e)
        return res

    # Write, re-read, verify; on any failure restore every file of this patch to its pre-patch bytes.
    written = []
    try:
        for rel in files:
            p = _path(site, rel)
            data = new_texts[rel].encode("utf-8")
            _write_atomic(p, data)
            written.append(rel)
            back = _read(p)
            if back != data:
                raise RuntimeError("%s: re-read bytes differ from written bytes" % rel)
            err = _compile_check(back.decode("utf-8"), p)
            if err:
                raise RuntimeError("%s: compile after write failed: %s" % (rel, err))
    except Exception as e:
        rolled = []
        for rel in written:
            try:
                _write_atomic(_path(site, rel), texts[rel][0])
                rolled.append(rel)
            except OSError:
                pass
        for rel in files:
            _drop_pyc(_path(site, rel))
        res["status"] = "refused: write failed (rolled back)"
        res["reason"] = "%s: %s" % (type(e).__name__, e)
        res["rolled_back"] = rolled
        return res

    for rel in files:
        p = _path(site, rel)
        f = res["files"].setdefault(rel, {})
        f["sha256_before"] = _sha(texts[rel][0])
        f["sha256_after"] = _sha(_read(p))
        f["pyc_removed"] = _drop_pyc(p)
    res["status"] = "applied"
    return res


def _revert_one(site, name):
    hunks = PATCHES[name]
    files = _FILES_OF[name]
    texts = _load_texts(site, files)
    state, states = _patch_state(name, texts)
    res = {"status": None, "state_before": state, "hunks": states, "files": {}}
    if state == "not_applied":
        res["status"] = "not applied"
        return res
    if state != "applied":
        res["status"] = "refused: source mismatch"
        res["reason"] = "%s: every hunk must be applied exactly once (no partial edit was made)" % state
        return res
    new_texts = {}
    for rel in files:
        new_texts[rel], err = _transform(texts[rel][1], [h for h in hunks if h["file"] == rel], False)
        if err:
            res["status"] = "refused: source mismatch"
            res["reason"] = "%s: %s" % (rel, err)
            return res
        err = _compile_check(new_texts[rel], _path(site, rel))
        if err:
            res["status"] = "refused: compile failed"
            res["reason"] = "%s: %s (nothing written)" % (rel, err)
            return res
    written = []
    try:
        for rel in files:
            p = _path(site, rel)
            data = new_texts[rel].encode("utf-8")
            _write_atomic(p, data)
            written.append(rel)
            if _read(p) != data:
                raise RuntimeError("%s: re-read bytes differ from written bytes" % rel)
    except Exception as e:
        for rel in written:
            try:
                _write_atomic(_path(site, rel), texts[rel][0])
            except OSError:
                pass
        for rel in files:
            _drop_pyc(_path(site, rel))
        res["status"] = "refused: write failed (rolled back)"
        res["reason"] = "%s: %s" % (type(e).__name__, e)
        return res
    for rel in files:
        p = _path(site, rel)
        res["files"][rel] = {"sha256_before": _sha(texts[rel][0]), "sha256_after": _sha(_read(p)),
                             "pyc_removed": _drop_pyc(p)}
    res["status"] = "reverted"
    return res


def revert(site, names=None):
    """Undo `names` (None = all 7) with the inverse hunks, dependents first. Refuses to revert 43567 while 43559 stays
    applied. A file whose names are all reverted must equal its .e14bak backup (then removed) and the pristine 0.5.21
    sha256; anything else is an error (nothing is clobbered)."""
    names = _resolve(names)
    rep = {"site": str(site), "version": _version(site), "requested": names, "order": [], "results": {}, "files": {},
           "errors": [], "ok": True}
    unknown = [n for n in names if n not in PATCHES]
    if unknown:
        rep["errors"].append("unknown patch name(s) %s" % ",".join(unknown))
        rep["ok"] = False
        rep["check"] = check(site)
        return rep
    before = check(site)
    for d, deps in REQUIRES.items():
        for x in deps:
            if x in names and d not in names and before["names"].get(d) == "applied":
                rep["errors"].append("refused: reverting %s would leave %s applied alone; revert both" % (x, d))
    if rep["errors"]:
        rep["ok"] = False
        rep["check"] = before
        return rep
    order = list(reversed(_dep_order(names)))
    rep["order"] = order
    for n in order:
        res = _revert_one(site, n)
        rep["results"][n] = res
        if res["status"] not in ("reverted", "not applied"):
            rep["errors"].append("%s: %s" % (n, res["status"]))
    after = check(site)
    states = {n: after["names"].get(n) for n in NAMES}
    for rel in sorted({f for n in names for f in _FILES_OF[n]}):
        if any(states.get(n) != "not_applied" for n in _NAMES_OF_FILE.get(rel, [])):
            continue                         # another name still patches this file: keep its backup
        p = _path(site, rel)
        f = {"backup_removed": False}
        try:
            raw = _read(p)
            f["sha256"] = _sha(raw)
            f["pristine_0_5_21"] = f["sha256"] == REFERENCE_FILE_SHA256.get(rel)
            bak = p + BACKUP_SUFFIX
            if os.path.exists(bak):
                if _read(bak) == raw:
                    os.remove(bak)
                    f["backup_removed"] = True
                else:
                    rep["errors"].append("%s: differs from its pristine backup after revert (left untouched)" % rel)
            if not f["pristine_0_5_21"] and (after.get("version") == REFERENCE_VERSION):
                rep["errors"].append("%s: sha256 is not the pristine 0.5.21 file after revert" % rel)
        except OSError as e:
            rep["errors"].append("%s: %s" % (rel, e))
        rep["files"][rel] = f
    rep["check"] = check(site)
    rep["ok"] = not rep["errors"] and rep["check"]["ok"]
    return rep


def restore_backups(site):
    """E14's revert (recovery only): restore every <file>.e14bak under `site` byte-identical, remove the backup and
    the stale pyc."""
    rep = {"site": str(site), "restored": [], "errors": [], "ok": True}
    baks = []
    for root, _dirs, fnames in os.walk(str(site)):
        for fn in fnames:
            if fn.endswith(BACKUP_SUFFIX):
                baks.append(os.path.join(root, fn))
    for bak in sorted(baks):
        target = bak[: -len(BACKUP_SUFFIX)]
        rel = os.path.relpath(target, str(site)).replace(os.sep, "/")
        try:
            data = _read(bak)
            _write_atomic(target, data)
            if _read(target) != data:
                raise OSError("restored bytes differ from backup")
            os.remove(bak)
            rep["restored"].append({"file": rel, "sha256": _sha(data), "pyc_removed": _drop_pyc(target)})
        except OSError as e:
            rep["errors"].append({"file": rel, "error": str(e)})
            rep["ok"] = False
    rep["check"] = check(site)
    return rep


_FIND_SPEC = ("import importlib.util, json, os\n"
              "s = importlib.util.find_spec('sglang')\n"
              "locs = list(s.submodule_search_locations or []) if s else []\n"
              "print(json.dumps({'origin': s.origin if s else None, 'locations': locs}))\n")


def site_dir(vpy, timeout=120):
    """The `sglang` package directory of interpreter `vpy` (importlib find_spec in `vpy -I`: the package itself is
    never imported). Raises RuntimeError when it cannot be found."""
    try:
        cp = subprocess.run([str(vpy), "-I", "-c", _FIND_SPEC], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError("site_dir: cannot run %s: %s" % (vpy, e))
    line = (cp.stdout or "").strip().splitlines()[-1:] or [""]
    try:
        d = json.loads(line[0])
    except ValueError:
        raise RuntimeError("site_dir: rc=%s stdout=%r stderr=%r" % (cp.returncode, cp.stdout[-300:], cp.stderr[-300:]))
    locs = d.get("locations") or []
    if not locs and d.get("origin"):
        locs = [os.path.dirname(d["origin"])]
    if not locs or not os.path.isdir(locs[0]):
        raise RuntimeError("site_dir: sglang not found by %s (%r)" % (vpy, d))
    return os.path.realpath(locs[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description="E24 guarded SGLang 0.5.21 patches")
    ap.add_argument("cmd", choices=("check", "apply", "revert", "restore_backups"))
    ap.add_argument("rest", nargs="*", help="E14 form: <pkg_dir> [name ...]")
    ap.add_argument("--site", help="sglang package dir (…/site-packages/sglang)")
    ap.add_argument("--vpy", help="resolve --site from this interpreter (find_spec)")
    ap.add_argument("--names", help="comma list (default: all 7 for check/revert)")
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    rest = list(a.rest)
    site = a.site or (site_dir(a.vpy) if a.vpy else (rest.pop(0) if rest else None))
    if not site:
        ap.error("--site, --vpy or a positional <pkg_dir> is required")
    names = _resolve(a.names) if a.names else (rest or None)
    if a.cmd == "check":
        rep = check(site, names)
    elif a.cmd == "apply":
        if not names:
            ap.error("apply needs --names")
        rep = apply(site, names)
    elif a.cmd == "revert":
        rep = revert(site, names)
    else:
        rep = restore_backups(site)
    print(json.dumps(rep, indent=1))
    return 0 if rep.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
