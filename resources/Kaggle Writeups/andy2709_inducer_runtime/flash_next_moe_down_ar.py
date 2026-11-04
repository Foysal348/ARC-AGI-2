"""Fused MoE down projection + host-memory all-reduce for Flash-Next's EP4 target at decode (2026-10-30).

For one decode MoE layer, `flash_next_moe_down_ar.cuh` replaces the routed experts' second Marlin GEMM,
moe_sum_reduce, fused_gate_sigmoid_mul_add and the TP all-reduce with one launch. Its CTAs compute whole output tiles
in column-chunk order; its last CTAs, in groups that take chunks round-robin, reduce each finished chunk and push it
through pinned host memory while the GEMM works on later ones. See the .cuh header for the arithmetic contract.

Serving (SGLANG_FLASH_NEXT_MOE_DOWN_AR=1 with the host all-reduce, flash_next_serving.py): parallel_state builds one
MoeDownAllReduce beside the TP group's host all-reduce; qwen2_moe.py's MoE block computes the shared expert and its
gate value and runs the routed experts inside `down_context` (in decode-graph capture as stock does, the shared expert
on the current stream beside the routed experts on the alt stream); fused_marlin_moe.py hands its down projection to
`try_fused_down`, which takes it only when the call matches the kernel (else the stock path finishes the layer) and
first waits for the shared output when it was computed on another stream.
`check_flash_next_moe_down_ar.py` and `bench_flash_next_moe_down_ar.py` drive the kernel against the stock kernels.

The communicator follows flash_next_host_ar.py: rank 0 creates a /dev/shm file, every rank maps and registers it with
the driver, and setup failures are agreed so all ranks fall back together.
"""
import contextlib
import ctypes
import functools
import logging
import mmap
import os
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

SOURCE = Path(__file__).with_name("flash_next_moe_down_ar.cuh")
WORLD = 4
PAGE = 4096
MAX_CHUNKS = 20  # n / (128 * chunk_tiles) at most: 2560 / 128
COMM_SLOTS = 32  # kCommSlots in the .cuh
TRACE_COMM = 2 + MAX_CHUNKS  # kTraceComm
TRACE_LENGTH = TRACE_COMM + MAX_CHUNKS * COMM_SLOTS * 8  # kTraceLength
PUBLISH = {"fence": 0, "release": 1, "ll": 2}  # Tail::publish
TIMEOUT_CYCLES = 1 << 40  # about nine minutes at L4 clocks, then the kernel traps


@functools.cache
def kernel_module():
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    args = make_cpp_args(torch.bfloat16)
    return load_jit(
        "flash_next_moe_down_ar",
        *args,
        cuda_files=[str(SOURCE)],
        cuda_wrappers=[("moe_down_ar", f"moe_down_ar<{args}>")],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
    )


@triton.jit
def _shared_gate_kernel(hidden_states_ptr, gate_weight_ptr, gate_ptr, hidden_dim: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    # sglang's _fused_gate_sigmoid_mul_add_kernel up to gate_val, stored instead of applied.
    pid = tl.program_id(axis=0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(hidden_states_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(gate_ptr + pid, tl.sigmoid(tl.sum(h * w, axis=0)))


def shared_gate(hidden_states, gate_weight):
    """sigmoid(hidden . gate_weight) per token in fp32, with the stock kernel's block and warp configuration."""
    num_tokens, hidden_dim = hidden_states.shape
    gate = torch.empty(num_tokens, dtype=torch.float32, device=hidden_states.device)
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 32), 4)
    if num_tokens >= 1024:
        num_warps = min(num_warps, 8)
    _shared_gate_kernel[(num_tokens,)](hidden_states, gate_weight, gate, hidden_dim=hidden_dim,
                                       BLOCK_SIZE=triton.next_power_of_2(hidden_dim), num_warps=num_warps)
    return gate


def _agree(ok, group):
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


class MoeDownAllReduce:
    """One four-rank group's buffer and state for `moe_down_ar` calls (one stream per rank, like the host AR)."""

    def __init__(self, group, device, rank, driver, size_n=2560, max_m=64):
        self.group, self.device, self.rank, self.driver = group, device, rank, driver
        self.size_n, self.max_m = size_n, max_m
        flags = -(-2 * WORLD * MAX_CHUNKS * COMM_SLOTS * 4 // PAGE) * PAGE
        # Chunk c of a half: 16 scatter + 4 result quarters of m * chunk_cols / 32 int4, so 10 * m * n bytes in all;
        # twice that for LL lines (4 data bytes in every 8).
        self.half = -(-20 * max_m * size_n // PAGE) * PAGE
        size = flags + 2 * self.half
        path = [f"/dev/shm/flash-next-moe-down-ar-{os.getpid()}-{uuid.uuid4().hex}" if rank == 0 else None]
        error = None
        if rank == 0:
            try:
                descriptor = os.open(path[0], os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                os.ftruncate(descriptor, size)
                os.close(descriptor)
            except OSError as exc:
                error = exc
        dist.broadcast_object_list(path, src=dist.get_global_rank(group, 0), group=group)
        if not _agree(error is None, group):
            raise RuntimeError(f"moe_down_ar shared buffer creation failed: {error}")
        try:
            descriptor = os.open(path[0], os.O_RDWR)
            try:
                self.buffer = mmap.mmap(descriptor, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(descriptor)
            address = driver.register(ctypes.addressof(ctypes.c_char.from_buffer(self.buffer)), size)
        except Exception as exc:  # noqa: BLE001 - every rank must reach the agreement below.
            error = exc
        mapped = _agree(error is None, group)
        if rank == 0:
            os.unlink(path[0])
        if not mapped:
            raise RuntimeError(f"moe_down_ar shared buffer registration failed: {error}")
        self.flags, self.data = address, address + flags
        self.state = torch.zeros(2, dtype=torch.int32, device=device)
        self.done = torch.zeros(MAX_CHUNKS, dtype=torch.int32, device=device)
        self.no_trace = torch.empty(0, dtype=torch.int64, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def __call__(self, act, c, w2, w2_scales, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
                 shared, gate, top_k, *, fused=True, chunk_tiles=1, comm_ctas=40, comm_groups=20, scale=1.0, out=None,
                 trace=None, publish="release"):
        """Down projection into `c` (zeroed rows of non-local slots stay zero); fused: also the all-reduced output.

        trace: an int64 tensor of TRACE_LENGTH for globaltimer stamps (see the .cuh), reset with reset_trace first.
        publish: "fence" (every thread fences at system scope, then a flag), "release" (one st.release.sys per CTA
        after bar.sync) or "ll" (no flags or fences: LL lines carry the epoch in every 8 bytes).
        """
        m = shared.shape[0]
        assert m <= self.max_m and shared.shape[1] == self.size_n == c.shape[1]
        if out is None:
            out = torch.empty_like(shared)
        kernel_module().moe_down_ar(
            act, c, w2, w2_scales, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, shared, gate,
            out, self.done, self.state, self.no_trace if trace is None else trace, self.data, self.flags, self.half,
            TIMEOUT_CYCLES, self.rank, top_k, m, self.size_n, act.shape[1], chunk_tiles, comm_ctas, comm_groups,
            PUBLISH[publish], fused, scale)
        return out


def reset_trace(trace):
    """Zero a trace buffer, with its first-start slot at the largest unsigned value (the kernel takes atomicMin)."""
    trace.zero_()
    trace[:1].fill_(-1)  # a device fill, so it can be captured in a CUDA graph
    return trace


def read_trace(trace, chunks, comm_ctas, comm_groups):
    """Microseconds from the first CTA's start: GEMM end, each chunk's last tile, and each comm CTA's eight events."""
    values = trace.cpu().tolist()
    start = values[0] % (1 << 64)
    us = lambda v: round(((v % (1 << 64)) - start) / 1000, 2) if v else None
    slots = comm_ctas // comm_groups
    comm = {}
    for c in range(chunks):
        for slot in range(slots):
            base = TRACE_COMM + (c * COMM_SLOTS + slot) * 8
            comm[f"{c}.{slot}"] = [us(v) for v in values[base:base + 8]]
    return {"gemm_end": us(values[1]), "chunk_done": [us(v) for v in values[2:2 + chunks]], "comm": comm}


def create_moe_down_all_reduce(group, device, rank, host_ar):
    """The fused communicator for a four-rank TP group that already has a host all-reduce, or None on every rank.

    The kernel builds here (the JIT cache keeps it), so a rank that cannot compile it fails setup with the others.
    """
    error = None
    try:
        kernel_module()
    except Exception as exc:  # noqa: BLE001 - every rank must reach the agreement.
        error = exc
    if not _agree(error is None, group):
        logger.warning("flash_next_moe_down_ar: kernel build failed on some rank (%s); stock MoE tail", error)
        return None
    try:
        communicator = MoeDownAllReduce(group, device, rank, host_ar.driver,
                                        max_m=host_ar_max_rows(host_ar))
    except RuntimeError as exc:
        logger.warning("flash_next_moe_down_ar: %s; stock MoE tail", exc)
        return None
    if rank == 0:
        logger.info("flash_next_moe_down_ar: enabled, m <= %d, %d-byte halves", communicator.max_m, communicator.half)
    return communicator


def host_ar_max_rows(host_ar, hidden=2560):
    """Rows of a [m, hidden] bf16 tensor the host all-reduce takes (its two-shot limit)."""
    from sglang.srt.distributed.device_communicators.flash_next_host_ar import MAX_BYTES

    return min(64, MAX_BYTES // (hidden * 2))


class _Context:
    def __init__(self, comm, shared, gate, ready):
        self.comm, self.shared, self.gate, self.ready, self.out = comm, shared, gate, ready, None


_context = None
_logged = False


@contextlib.contextmanager
def down_context(comm, shared, gate, ready=None):
    """Offer the next eligible fused_marlin_moe call of this layer to the fused kernel (out is set if it took it).

    `ready` is a CUDA event recorded after `shared` and `gate` when they come from another stream than the experts.
    """
    global _context
    previous, _context = _context, _Context(comm, shared, gate, ready)
    try:
        yield _context
    finally:
        _context = previous


def try_fused_down(act, c, w2, w2_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, *,
                   top_k, block_size_m, expert_map, num_bits, w2_zeros, w2_bias, g_idx2, w2_global_scale,
                   routed_scaling_factor, activation, is_gated, clamp_limit, gemm1_alpha, is_k_full):
    """fused_marlin_moe's down projection through the fused kernel, or None to run the stock one.

    Only inside down_context, for the call the kernel was built for: EP experts (expert map), uint4b8 weights with bf16
    group-128 scales, 8-row MoE blocks, silu-gated activation, no bias, zeros or act order, hidden width the communicator
    was sized for. `c` is fused_marlin_moe's zeroed down-projection buffer [m * top_k, hidden].
    """
    context = _context
    if context is None or context.out is not None:
        return None
    comm, shared = context.comm, context.shared
    m, hidden = shared.shape
    eligible = (
        expert_map is not None and num_bits == 4 and w2_zeros is None and w2_bias is None
        and w2_global_scale is None and (g_idx2 is None or g_idx2.numel() == 0) and is_k_full
        and block_size_m == 8 and activation == "silu" and is_gated and clamp_limit is None and gemm1_alpha is None
        and act.dtype == torch.bfloat16 and w2_scale.dtype == torch.bfloat16 and w2_scale.dim() == 3
        and w2_scale.shape[1] * 128 == act.shape[1] and hidden == comm.size_n and c.shape == (m * top_k, hidden)
        and act.shape[0] == m * top_k and m <= comm.max_m and top_k <= 16 and act.is_contiguous()
    )
    if not eligible:
        return None
    global _logged
    if not _logged and comm.rank == 0:
        logger.info("flash_next_moe_down_ar: first fused MoE layer (m=%d, %d local experts)", m, w2.shape[0])
        _logged = True
    if context.ready is not None:
        torch.cuda.current_stream().wait_event(context.ready)  # the comm CTAs read shared and gate
    context.out = comm(act, c, w2, w2_scale, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
                       shared, context.gate, top_k,
                       scale=1.0 if routed_scaling_factor is None else float(routed_scaling_factor))
    return context.out
