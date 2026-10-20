# sglang 0.5.21 (tag v0.5.21) mem_cache/memory_pool.py lines 274-400
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    enable_mamba_extra_buffer_lazy: bool = False
    # Extra pre-allocation headroom (reserved for in-transfer decode requests).
    # 0 for a plain pool; the decode-flavored pool (DecodeReqToTokenPool) sets a
    # positive value. Declared here so callers can read it without getattr.
    pre_alloc_size: int = 0
    # Class default: some decode pools borrow another __init__ (see
    # DecodeReqToTokenPool) but inherit alloc_rows.
    _on_alloc_rows: Optional[Callable[[List[int]], None]] = None

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0: cuda-graph padded batches default
        # req_pool_indices to 0, so dummy reads/writes land here harmlessly.
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)
        self._aux_cache: Any = None

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.kv.holds_kv]
        assert all(reqs[i].kv.kv_allocated_len > 0 for i in reusing), (
            "a reused row must carry allocated KV"
        )

        select_index = self.alloc_rows(len(reqs) - len(reusing))
        if select_index is None:
            return None
        offset = 0
        for r in reqs:
            if not r.kv.holds_kv:
                r.kv.req_pool_idx = select_index[offset]
                offset += 1
        return [r.kv.req_pool_idx for r in reqs]

    def alloc_rows(self, need_size: int) -> Optional[List[int]]:
        """Take need_size rows and bump their generation, with no Req bound to
        them. alloc() layers Req binding on top; beam member rows have no Req."""
        if need_size > len(self.free_slots):
            return None
        if need_size == 0:
            # Handled separately: free_slots[-0:] is the entire list, not [].
            return []
        # Pop from the tail: O(need_size), unlike a prefix pop which is
        # O(len(free_slots)).
        select_index = self.free_slots[-need_size:]
        del self.free_slots[-need_size:]
        self.req_generation[select_index] += 1
        if self._on_alloc_rows is not None:
            self._on_alloc_rows(select_index)
        return select_index

    def free_rows(self, indices: List[int]) -> None:
        # Per-row, so beam member rows release their aux entries too: they reach
        # alloc_aux_to_lengths via the decode batch's req_pool_indices_cpu.
        if self._aux_cache is not None:
            for index in indices:
                self._aux_cache.free(index)
        self.free_slots.extend(indices)

    def free(self, req: Req):
        assert req.kv.holds_kv, "request must have req_pool_idx"
        self.free_rows([req.kv.req_pool_idx])
        req.kv.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation.zero_()
        if self._aux_cache is not None:
            self._aux_cache.clear()

    def attach_aux_cache(self, aux_cache: Any) -> None:
        assert self._aux_cache is None
        self._aux_cache = aux_cache

    def register_on_alloc_rows(self, hook: Callable[[List[int]], None]) -> None:
        assert self._on_alloc_rows is None
        self._on_alloc_rows = hook

    def reset_aux_cache_allocator(self) -> None:
        if self._aux_cache is not None:
            self._aux_cache.reset_allocator()

    def schedulable_token_capacity(self, physical_capacity: int) -> int:
        if self._aux_cache is None:
            return physical_capacity
        return self._aux_cache.dense_capacity

    def alloc_aux_to_lengths(
        self,
        *,
        req_pool_indices_cpu: torch.Tensor,
        target_seq_lens_cpu: torch.Tensor,
    ) -> None:
        if self._aux_cache is not None:
            self._aux_cache.alloc_to_lengths(
                req_pool_indices_cpu=req_pool_indices_cpu,
                target_seq_lens_cpu=target_seq_lens_cpu,
            )


