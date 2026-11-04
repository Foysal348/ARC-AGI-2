"""Decode context parallelism for SGLang 0.5.20's Qwen sparse attention (SGLANG_FLASH_NEXT_QSA_DCP=1).

patch_sglang (flash_next_serving.py) installs this file as sglang/srt/layers/attention/qsa/flash_next_dcp.py. Only
the target's QSA backend uses it, and only when the launcher serves with --decode-context-parallel-size > 1: the
NEXTN draft never joins the DCP group, so its pool keeps spanning the scheduler's logical slots.

Stock QSA has no DCP path. Under DCP the scheduler hands out logical slots over dcp_size times each rank's KV rows,
and QSA wrote and read every token at its logical slot: DCP split nothing, and the first slot past the physical
rows failed store_kvcache (cache kernel version 6; phi1 repro, codex_lite_kaggle_20261028_v28c16/findings.md).
Here SGLang's owner rule holds: logical slot s lives on DCP rank s % dcp_size, at row s // dcp_size.

- KV writes: owned tokens go to their row, the others to row 0, the padding row the allocator never hands out.
- Sparse attention (decode, target verify, chunked prefill): every rank all-gathers its DCP group's query heads,
  attends to the selected tokens it owns, and the partial outputs are merged through their natural-log LSE, as
  SGLang's cp_lse_ag_out_rs_mha does for its Triton backend, but with all-gathers only: that helper all-reduces the
  outputs, and the DCP group's custom all-reduce fails CUDA-graph capture on Kaggle's L4s, which have no usable peer
  path (cache kernel version 7: custom_all_reduce.cuh:614, invalid argument). The kernel is QSA's chunk-prefill
  Triton kernel with an LSE output, a per-row key count and an online softmax that tolerates an all-masked first
  block, so it needs no FlashAttention (FA4 does not run on L4) and is the same code on L4 and H200. Decode and
  verify hand it absolute pool rows rather than packed offsets, so its token offsets are int64.
- A request's first prefill chunk attends only to its own K/V, which every rank holds, so it needs no merge.
- Indexer state stays replicated: its compressed-key cache spans the logical slots (qsa_index_copies times the
  rank's KV rows), and the KV pool's per-token cost counts those copies.
"""
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ENV = "SGLANG_FLASH_NEXT_QSA_DCP"


def _dcp_size():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().attn_dcp_size


def enabled():
    return os.environ.get(ENV) == "1" and _dcp_size() > 1


def qsa_index_copies(is_draft_worker):
    """Logical slots per KV row of a pool, which its replicated indexer cache must cover."""
    return _dcp_size() if enabled() and not is_draft_worker else 1


def create_qsa_dcp(backend, runner):
    """The target backend's DCP helper, or None (the draft, a CPU runner, or DCP off)."""
    if runner is None or getattr(runner, "is_draft_worker", False) or not enabled():
        return None
    return QSADecodeContextParallel(backend)


@triton.jit
def _owned_rows(
    req_to_token,
    req_indices,
    positions,
    lengths,
    rows,
    counts,
    topk,
    req_stride,
    position_stride,
    row_stride,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per query row: the KV rows this rank owns among the selected positions, packed at the front, -1 after."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_range = cols < topk
    length = tl.load(lengths + row)
    req = tl.load(req_indices + row).to(tl.int64)
    position = tl.load(positions + row * position_stride + cols, mask=in_range, other=-1)
    valid = in_range & (position >= 0) & (position < length)
    slot = tl.load(req_to_token + req * req_stride + tl.where(valid, position, 0), mask=valid, other=0)
    owned = valid & (slot % DCP_SIZE == DCP_RANK)
    owned_count = owned.to(tl.int32)
    count = tl.sum(owned_count, 0)
    tl.store(rows + row * row_stride + tl.cumsum(owned_count, 0) - 1, (slot // DCP_SIZE).to(tl.int32), mask=owned)
    tl.store(rows + row * row_stride + cols, tl.full([BLOCK], -1, tl.int32), mask=in_range & (cols >= count))
    tl.store(counts + row, count)


@triton.jit
def _sparse_gqa_lse(
    q,
    k,
    v,
    out,
    lse,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    counts,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sl_m: tl.constexpr,
    sl_h: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
):
    """qsa/sparse_attn.py's _sparse_gqa_chunk_prefill, writing fp32 outputs and their natural-log LSE."""
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    if HAS_COUNTS:
        row_topk = tl.minimum(row_topk, tl.load(counts + query).to(tl.int64))
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    heads = (offs_h < GROUP_SIZE)[:, None]
    q_values = tl.load(
        q + query * sq_m + (group * GROUP_SIZE + offs_h[:, None]) * sq_h + offs_d[None, :] * sq_d,
        mask=heads,
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        # int64: paged() passes absolute pool rows, and past row 2**31 / sk_n (8,388,608 on an H200 rank's 256-element
        # rows) int32 offsets wrapped into other buffers: the held-server corruption of l4_hclocal_20261029.
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1).to(tl.int64)
        valid = token >= 0
        keys = tl.load(k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d, mask=valid[None, :], other=0.0)
        values = tl.load(v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d, mask=valid[:, None], other=0.0)
        # FP8 pool rows are cast to Q's dtype, as in the stock kernel (QSA writes the pool without scales).
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        # A block with no owned token before any owned one would otherwise give exp2(-inf - -inf) = NaN.
        shift = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.math.exp2(max_value - shift)
        probabilities = tl.math.exp2(scores - shift[:, None])
        accumulator = tl.dot(probabilities.to(values.dtype), values, accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    has_mass = normalizer > 0
    safe = tl.where(has_mass, normalizer, 1.0)
    tl.store(
        out + query * so_m + (group * GROUP_SIZE + offs_h[:, None]) * so_h + offs_d[None, :] * so_d,
        accumulator / safe[:, None],
        mask=heads,
    )
    # Scores are in base 2 (q carries log2(e)): ln(sum e^s) = (max + log2(normalizer)) * ln 2.
    tl.store(
        lse + query * sl_m + (group * GROUP_SIZE + offs_h) * sl_h,
        tl.where(has_mass, (max_value + tl.log2(safe)) * 0.6931471805599453, -float("inf")),
        mask=offs_h < GROUP_SIZE,
    )


def sparse_gqa_lse(q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q, counts=None):
    """Sparse GQA over ``indices`` (rows of ``k``/``v`` from ``cu_k``), returning fp32 outputs and natural-log LSE."""
    from sglang.srt.layers.attention.qsa.sparse_attn import _get_best_config

    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty((total_q, num_q_heads, v.shape[-1]), dtype=torch.float32, device=q.device)
    lse = torch.empty((total_q, num_q_heads), dtype=torch.float32, device=q.device)
    _sparse_gqa_lse[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        lse,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        kv_lens if counts is None else counts,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=max(16, triton.next_power_of_2(group_size)),
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        HAS_COUNTS=counts is not None,
        num_warps=warps,
        num_stages=stages,
    )
    return out, lse


class QSADecodeContextParallel:
    """The owner rule for one target QSA backend: write rows, owned reads and the cross-rank LSE merge."""

    def __init__(self, backend, group=None, size=None, rank=None):
        if group is None:
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            group, size, rank = parallel.dcp_group, parallel.attn_dcp_size, parallel.attn_dcp_rank
        if (group.world_size, group.rank_in_group) != (size, rank):
            raise RuntimeError(f"DCP group is rank {group.rank_in_group} of {group.world_size}, not {rank} of {size}")
        self.backend, self.group, self.size, self.rank = backend, group, size, rank

    def write_loc(self, loc):
        """Each token's row on this rank, or row 0 (padding) for the tokens another rank owns."""
        return torch.div(loc, self.size, rounding_mode="floor") * (torch.remainder(loc, self.size) == self.rank)

    def _merge(self, out, lse):
        """This rank's heads of the LSE-weighted sum of every rank's partial outputs ([T, H_all, D], [T, H_all])."""
        lses = self.group.all_gather(lse.contiguous(), dim=0).view(self.size, *lse.shape)
        scale = torch.exp(lse - torch.logsumexp(lses, dim=0)).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        out = out.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0) * scale.unsqueeze(-1)
        heads = out.shape[1] // self.size
        own = slice(self.rank * heads, (self.rank + 1) * heads)
        if self.size == 2:
            # Each rank sends its partner only the partner's heads.
            partner = 1 - self.rank
            sent = out[:, partner * heads:(partner + 1) * heads].contiguous()
            return out[:, own] + self.group.all_gather(sent, dim=0).view(2, *sent.shape)[partner]
        return self.group.all_gather(out.contiguous(), dim=0).view(self.size, *out.shape).sum(0)[:, own]

    def _attend(self, q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q, counts=None):
        q_all = self.group.all_gather(q.contiguous(), dim=1).contiguous()
        out, lse = sparse_gqa_lse(q_all, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q, counts)
        return self._merge(out, lse).to(q.dtype).reshape(q.shape[0], -1)

    def paged(self, q, layer, forward_batch, metadata, topk_indices):
        """Decode and target verify: one query per row, reading this rank's pool rows directly."""
        pool = self.backend.token_to_kv_pool
        req_to_token = self.backend.req_to_token_pool.req_to_token
        req_indices = (
            metadata.row_req_pool_indices
            if metadata.row_req_pool_indices is not None
            else forward_batch.req_pool_indices
        )
        num_rows, topk = topk_indices.shape
        rows = torch.empty_like(topk_indices)
        counts = torch.empty(num_rows, dtype=torch.int32, device=q.device)
        _owned_rows[(num_rows,)](
            req_to_token,
            req_indices,
            topk_indices,
            metadata.sequence_lengths,
            rows,
            counts,
            topk,
            req_to_token.stride(0),
            topk_indices.stride(0),
            rows.stride(0),
            DCP_SIZE=self.size,
            DCP_RANK=self.rank,
            BLOCK=triton.next_power_of_2(topk),
            num_warps=8,
        )
        return self._attend(
            q,
            pool.get_key_buffer(layer.layer_id),
            pool.get_value_buffer(layer.layer_id),
            rows,
            torch.arange(num_rows + 1, dtype=torch.int32, device=q.device),
            torch.zeros(num_rows + 1, dtype=torch.int32, device=q.device),
            metadata.sequence_lengths,
            layer.scaling,
            1,
            counts,
        )

    def chunk_prefill(self, q, layer, forward_batch, topk_indices, sequence_lens, cu_seqlens_q):
        """A chunk with a cached prefix: pack each request's context in logical order, owned rows only."""
        pool = self.backend.token_to_kv_pool
        req_to_token = self.backend.req_to_token_pool.req_to_token
        requests = forward_batch.req_pool_indices.tolist()
        slots = torch.cat([req_to_token[requests[i], :length] for i, length in enumerate(sequence_lens)]).long()
        owned = torch.remainder(slots, self.size) == self.rank
        rows = torch.div(slots, self.size, rounding_mode="floor") * owned
        lengths = torch.tensor(sequence_lens, dtype=torch.int32, device=q.device)
        cu_k = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
        extend_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
        request_of_row = torch.repeat_interleave(torch.arange(len(sequence_lens), device=q.device), extend_lens)
        positions = topk_indices.long()
        start = cu_k[:-1].long()[request_of_row].view(-1, *([1] * (positions.ndim - 1)))
        packed = (start + positions.clamp(min=0)).clamp(max=slots.numel() - 1)
        indices = torch.where((positions >= 0) & owned[packed], positions, -1).to(torch.int32).contiguous()
        return self._attend(
            q,
            pool.get_key_buffer(layer.layer_id).index_select(0, rows),
            pool.get_value_buffer(layer.layer_id).index_select(0, rows),
            indices,
            cu_seqlens_q,
            cu_k,
            lengths,
            layer.scaling,
            int(extend_lens.max()),
        )
