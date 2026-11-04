"""SW block-table mask seam (port of vLLM #60767, fix for #60560): env-gated, MERLIN_SW_BT_MASK unset = the original
gather kernel runs with the original arguments (byte-identical forward inputs).

Problem (#60560): SlidingWindowManager.remove_skipped_blocks frees blocks below the window in the scheduler, but the V2
worker's persistent block table is append-only (block_table.py:114-133 append_block_ids), so the freed ids stay in the
row. After reuse by another request those pages can hold NaN / foreign data; a kernel that loads a masked tile of an
out-of-window block multiplies NaN by 0 -> NaN. Upstream: DFlash2 acceptance 40 % -> 27-29 % at 32 sessions.
Merlin ad42e37: ABSENT. v1/worker/gpu/block_table.py:149-179 gather_block_tables + :243-312 _gather_block_tables_kernel
copy rows verbatim (no window nulling); no sliding_window_reach / num_retained_tokens anywhere in the tree.

Port (forward-pass copy only; the persistent rows are never changed, same as upstream):
  block_table.py   new method set_sliding_window_reach(reach) + optional gather_block_tables(num_computed_tokens=)
                   + a NEW kernel _gather_block_tables_sw_kernel (the original kernel is untouched): for group g with
                   reach r_g = sliding_window - 1 + extra_retained_tokens (> 0 only for SlidingWindowSpec groups), row
                   entries [0, max(first_pos - r_g, 0) // block_size_g) are written as 0 (the null block),
                   first_pos = num_computed_tokens[req] (the first scheduled position of the step).
  model_runner.py  after BlockTables(...) (:824-835): set the reach when MERLIN_SW_BT_MASK=1 (zeros under DCP/PCP,
                   cp_size > 1, as upstream); prepare_attn (:1847-1849) passes req_states.num_computed_tokens.gpu.
The DFlash speculator (spec_decode/dflash/speculator.py:405-425) reads the same gathered input tables for its draft
group, so it is covered by the one gather in prepare_attn.

CLI:  python sw_bt_mask_patch.py apply|verify|revert <dir that contains the vllm package>
"""
import hashlib, sys, py_compile
from pathlib import Path

MARK = "CAPACITY-PORT SWMASK"

BT_OLD_SIG = '''    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs
        else:
            assert out_ptrs is not None
            assert len(out) == self.num_kv_cache_groups
        num_reqs = idx_mapping.shape[0]
'''
BT_NEW_SIG = '''    def set_sliding_window_reach(self, reach: list[int]) -> None:  # CAPACITY-PORT SWMASK
        # CAPACITY-PORT SWMASK: per group sliding_window - 1 + extra_retained_tokens
        # (0 = not a sliding-window group); vLLM #60767.
        assert len(reach) == self.num_kv_cache_groups  # CAPACITY-PORT SWMASK
        if self.cp_size > 1:  # CAPACITY-PORT SWMASK
            reach = [0] * self.num_kv_cache_groups  # CAPACITY-PORT SWMASK
        self._cp_sw_reach_any = any(r > 0 for r in reach)  # CAPACITY-PORT SWMASK
        self._cp_sw_reach = torch.tensor(  # CAPACITY-PORT SWMASK
            reach, dtype=torch.int32, device=self.device  # CAPACITY-PORT SWMASK
        )  # CAPACITY-PORT SWMASK

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,  # CAPACITY-PORT SWMASK
    ) -> tuple[torch.Tensor, ...]:
        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs
        else:
            assert out_ptrs is not None
            assert len(out) == self.num_kv_cache_groups
        num_reqs = idx_mapping.shape[0]
        if (  # CAPACITY-PORT SWMASK
            num_computed_tokens is not None  # CAPACITY-PORT SWMASK
            and getattr(self, "_cp_sw_reach_any", False)  # CAPACITY-PORT SWMASK
        ):  # CAPACITY-PORT SWMASK
            _gather_block_tables_sw_kernel[  # CAPACITY-PORT SWMASK
                (self.num_kv_cache_groups, num_reqs_padded)  # CAPACITY-PORT SWMASK
            ](  # CAPACITY-PORT SWMASK
                idx_mapping,  # CAPACITY-PORT SWMASK
                self.block_table_ptrs,  # CAPACITY-PORT SWMASK
                out_ptrs,  # CAPACITY-PORT SWMASK
                self.block_table_strides,  # CAPACITY-PORT SWMASK
                self.num_blocks.gpu,  # CAPACITY-PORT SWMASK
                self.num_blocks.gpu.stride(0),  # CAPACITY-PORT SWMASK
                num_reqs,  # CAPACITY-PORT SWMASK
                num_computed_tokens,  # CAPACITY-PORT SWMASK
                self._cp_sw_reach,  # CAPACITY-PORT SWMASK
                self.block_sizes_tensor,  # CAPACITY-PORT SWMASK
                BLOCK_SIZE=1024,  # type: ignore  # CAPACITY-PORT SWMASK
            )  # CAPACITY-PORT SWMASK
            return tuple(bt[:num_reqs_padded] for bt in out)  # CAPACITY-PORT SWMASK
'''

BT_OLD_TAIL = '''@triton.jit
def _compute_slot_mappings_kernel('''
BT_NEW_TAIL = '''@triton.jit(do_not_specialize=["num_reqs"])  # CAPACITY-PORT SWMASK
def _gather_block_tables_sw_kernel(  # CAPACITY-PORT SWMASK
    batch_idx_to_req_idx,  # CAPACITY-PORT SWMASK
    src_block_table_ptrs,  # CAPACITY-PORT SWMASK
    dst_block_table_ptrs,  # CAPACITY-PORT SWMASK
    block_table_strides,  # CAPACITY-PORT SWMASK
    num_blocks_ptr,  # CAPACITY-PORT SWMASK
    num_blocks_stride,  # CAPACITY-PORT SWMASK
    num_reqs,  # CAPACITY-PORT SWMASK
    num_computed_tokens_ptr,  # [max_num_reqs]  # CAPACITY-PORT SWMASK
    reach_ptr,  # [num_kv_cache_groups]  # CAPACITY-PORT SWMASK
    block_sizes_ptr,  # [num_kv_cache_groups]  # CAPACITY-PORT SWMASK
    BLOCK_SIZE: tl.constexpr,  # CAPACITY-PORT SWMASK
):  # CAPACITY-PORT SWMASK
    # CAPACITY-PORT SWMASK: the original gather (row copy, zeroed padded rows,
    # clamped row count) plus the vLLM #60767 window mask: entries wholly below
    # first_pos - reach go out as the null block 0.
    group_id = tl.program_id(0)  # CAPACITY-PORT SWMASK
    batch_idx = tl.program_id(1)  # CAPACITY-PORT SWMASK
    stride = tl.load(block_table_strides + group_id)  # CAPACITY-PORT SWMASK
    max_num_blocks = stride  # CAPACITY-PORT SWMASK
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)  # CAPACITY-PORT SWMASK
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride  # CAPACITY-PORT SWMASK
    req_idx = tl.load(  # CAPACITY-PORT SWMASK
        batch_idx_to_req_idx + batch_idx, mask=batch_idx < num_reqs, other=-1  # CAPACITY-PORT SWMASK
    )  # CAPACITY-PORT SWMASK
    if (batch_idx >= num_reqs) or (req_idx < 0) or (req_idx >= num_blocks_stride):  # CAPACITY-PORT SWMASK
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):  # CAPACITY-PORT SWMASK
            offset = i + tl.arange(0, BLOCK_SIZE)  # CAPACITY-PORT SWMASK
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)  # CAPACITY-PORT SWMASK
        return  # CAPACITY-PORT SWMASK
    num_blocks = tl.load(num_blocks_ptr + group_id * num_blocks_stride + req_idx)  # CAPACITY-PORT SWMASK
    num_blocks = tl.minimum(tl.maximum(num_blocks, 0), max_num_blocks).to(tl.int32)  # CAPACITY-PORT SWMASK
    reach = tl.load(reach_ptr + group_id)  # CAPACITY-PORT SWMASK
    bs = tl.load(block_sizes_ptr + group_id)  # CAPACITY-PORT SWMASK
    first_pos = tl.load(num_computed_tokens_ptr + req_idx)  # CAPACITY-PORT SWMASK
    num_null = tl.maximum(first_pos - reach, 0) // bs  # CAPACITY-PORT SWMASK
    num_null = tl.where(reach > 0, num_null, 0)  # CAPACITY-PORT SWMASK
    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)  # CAPACITY-PORT SWMASK
    src_row_ptr = src_block_table_ptr + req_idx * stride  # CAPACITY-PORT SWMASK
    for i in tl.range(0, num_blocks, BLOCK_SIZE):  # CAPACITY-PORT SWMASK
        offset = i + tl.arange(0, BLOCK_SIZE)  # CAPACITY-PORT SWMASK
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)  # CAPACITY-PORT SWMASK
        block_ids = tl.where(offset < num_null, 0, block_ids)  # CAPACITY-PORT SWMASK
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)  # CAPACITY-PORT SWMASK


def sw_mask_reference(rows, num_blocks, first_pos, reach, block_size):  # CAPACITY-PORT SWMASK
    """CPU reference of the mask (unit tests): a list of row lists in, the gathered rows out."""  # CAPACITY-PORT SWMASK
    out = []  # CAPACITY-PORT SWMASK
    for row, n, p in zip(rows, num_blocks, first_pos):  # CAPACITY-PORT SWMASK
        k = max(p - reach, 0) // block_size if reach > 0 else 0  # CAPACITY-PORT SWMASK
        out.append([0 if j < k else row[j] for j in range(n)])  # CAPACITY-PORT SWMASK
    return out  # CAPACITY-PORT SWMASK


@triton.jit
def _compute_slot_mappings_kernel('''

MR_OLD_INIT = '''            cp_interleave=self.cp_interleave,
        )
        self.model_state.set_block_tables(self.block_tables)
'''
MR_NEW_INIT = '''            cp_interleave=self.cp_interleave,
        )
        if __import__("os").environ.get("MERLIN_SW_BT_MASK") == "1":  # CAPACITY-PORT SWMASK
            from vllm.v1.kv_cache_interface import (  # CAPACITY-PORT SWMASK
                SlidingWindowSpec as _CpSW,  # CAPACITY-PORT SWMASK
                UniformTypeKVCacheSpecs as _CpU,  # CAPACITY-PORT SWMASK
            )  # CAPACITY-PORT SWMASK

            _cp_reach = []  # CAPACITY-PORT SWMASK
            for _cp_g in self.kv_cache_config.kv_cache_groups:  # CAPACITY-PORT SWMASK
                _cp_s = _cp_g.kv_cache_spec  # CAPACITY-PORT SWMASK
                if isinstance(_cp_s, _CpU) and _cp_s.kv_cache_specs:  # CAPACITY-PORT SWMASK
                    _cp_s = next(iter(_cp_s.kv_cache_specs.values()))  # CAPACITY-PORT SWMASK
                _cp_reach.append(  # CAPACITY-PORT SWMASK
                    _cp_s.sliding_window - 1 + _cp_s.extra_retained_tokens  # CAPACITY-PORT SWMASK
                    if isinstance(_cp_s, _CpSW)  # CAPACITY-PORT SWMASK
                    else 0  # CAPACITY-PORT SWMASK
                )  # CAPACITY-PORT SWMASK
            self.block_tables.set_sliding_window_reach(_cp_reach)  # CAPACITY-PORT SWMASK
            logger.info("CAPACITY-PORT SWMASK: sliding-window reach per group %s", _cp_reach)  # CAPACITY-PORT SWMASK
        self.model_state.set_block_tables(self.block_tables)
'''
MR_OLD_CALL = '''        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping, num_reqs_padded=input_batch.num_reqs_after_padding
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].'''
MR_NEW_CALL = '''        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
            num_computed_tokens=self.req_states.num_computed_tokens.gpu,  # CAPACITY-PORT SWMASK
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].'''

FILES = {
    "vllm/v1/worker/gpu/block_table.py": {
        "pre": "ee601f2914991ddfe0b4d1fe745fe51a94a8d10f0165e92e51e225993cdc1411",
        "post": "14905839f0616e83c20bbf52317ef0e843620485490212ed1ad64d89743603c6",
        "edits": [(BT_OLD_SIG, BT_NEW_SIG), (BT_OLD_TAIL, BT_NEW_TAIL)]},
    "vllm/v1/worker/gpu/model_runner.py": {
        "pre": "05e889afba5dd60b6995b78aaa1f69051770ab02b638680097eccbf4e592d64c",
        "post": "81f18a5f04b80e08a48bb703b2d0a5403a40cdb5e7c9ffe5694acaa7ebdf33e7",
        "edits": [(MR_OLD_INIT, MR_NEW_INIT), (MR_OLD_CALL, MR_NEW_CALL)]},
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def state(tree):
    out = {}
    for rel, f in FILES.items():
        p = Path(tree) / rel
        b = p.read_bytes()
        h = sha(b)
        out[rel] = {"sha": h, "state": "PRISTINE" if h == f["pre"] else "PATCHED" if h == f["post"] else
                    ("FOREIGN_MARKED" if MARK.encode() in b else "FOREIGN")}
    return out


def apply(tree):
    res = {}
    for rel, f in FILES.items():
        p = Path(tree) / rel
        t = p.read_text(encoding="utf-8")
        if MARK in t:
            res[rel] = "already patched"
            continue
        for old, new in f["edits"]:
            n = t.count(old)
            if n != 1:
                raise SystemExit(f"{rel}: anchor matched {n} times, refusing")
            t = t.replace(old, new)
        p.write_bytes(t.encode("utf-8"))
        py_compile.compile(str(p), doraise=True)
        for pyc in p.parent.glob(f"__pycache__/{p.stem}.*.pyc"):
            pyc.unlink()
        res[rel] = sha(p.read_bytes())
    return res


def revert(tree):
    res = {}
    for rel, f in FILES.items():
        p = Path(tree) / rel
        t = p.read_text(encoding="utf-8")
        if MARK not in t:
            res[rel] = "not patched"
            continue
        for old, new in f["edits"]:
            if t.count(new) != 1:
                raise SystemExit(f"{rel}: patched block not intact, refusing")
            t = t.replace(new, old)
        assert MARK not in t
        p.write_bytes(t.encode("utf-8"))
        for pyc in p.parent.glob(f"__pycache__/{p.stem}.*.pyc"):
            pyc.unlink()
        res[rel] = sha(p.read_bytes())
    return res


if __name__ == "__main__":
    cmd, tree = sys.argv[1], sys.argv[2]
    if cmd == "apply":
        print(apply(tree))
    elif cmd == "revert":
        print(revert(tree))
    print(state(tree))
