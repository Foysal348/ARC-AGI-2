# sglang 0.5.21 (tag v0.5.21) mem_cache/memory_pool.py lines 1224-1520
class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    mamba_pool_cls = MambaPool

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        enable_mamba_extra_buffer_lazy: bool = False,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        short_conv_layer_ids: Optional[List[int]] = None,
        short_conv_state_shape: Optional[Tuple[int, int]] = None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = enable_mamba_extra_buffer_lazy
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.ple_window_cache = None
        self._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            enable_linear_replayssm=enable_linear_replayssm,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            mamba_envelope_layout=mamba_envelope_layout,
            enable_linear_replayssm_spec=enable_linear_replayssm_spec,
            short_conv_layer_ids=short_conv_layer_ids,
            short_conv_state_shape=short_conv_state_shape,
            ngram_context_len=ngram_context_len,
            ngram_eos_token_id=ngram_eos_token_id,
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        short_conv_layer_ids: Optional[List[int]] = None,
        short_conv_state_shape: Optional[Tuple[int, int]] = None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
    ):
        self.mamba_pool = self.mamba_pool_cls(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            enable_linear_replayssm=enable_linear_replayssm,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            envelope_layout=mamba_envelope_layout,
            enable_linear_replayssm_spec=enable_linear_replayssm_spec,
        )
        self.mamba_allocator = MambaSlotAllocator(
            size=mamba_size,
            device=device,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}

        # Qwen4-Exp PLE side states; built disabled rather than None without a config,
        # so every hybrid model has both attributes.
        from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool

        self.short_conv_pool = ShortConvPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            state_shape=short_conv_state_shape,
            layer_ids=short_conv_layer_ids or [],
            dtype=cache_params.dtype.conv,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.ngram_pool = NGramPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            context_len=ngram_context_len,
            eos_token_id=ngram_eos_token_id,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        # Disabled pools stay off the sibling list so the host-offload payload
        # keeps its legacy shape for every non-PLE hybrid model.
        if self.short_conv_pool.enabled:
            self.mamba_pool.register_slot_state(self.short_conv_pool)
        if self.ngram_pool.enabled:
            self.mamba_pool.register_slot_state(self.ngram_pool)

        # Optional int8 checkpoint pool: the radix caches states here (int8) instead
        # of holding them in the active bf16 pool -> ~2x cached-prefix capacity at
        # fixed memory. Strategy-agnostic (no_buffer / extra_buffer / spec).
        from sglang.srt.mem_cache.mamba_checkpoint_pool import (
            maybe_init_int8_mamba_checkpoint_pool,
        )

        self.mamba_ckpt_pool = maybe_init_int8_mamba_checkpoint_pool(
            mamba_size=mamba_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
        )
        if self.mamba_ckpt_pool is not None and (
            self.short_conv_pool.enabled or self.ngram_pool.enabled
        ):
            # The int8 checkpoint pool frees the bf16 slot after donating its state,
            # taking the bf16-slot-indexed PLE side states with it.
            raise ValueError(
                "--enable-int8-mamba-checkpoint is incompatible with Qwen4-Exp "
                "PLE side states"
            )

        self.device = device
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def clone_with_new_mamba(
        self,
        *,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        device: str,
        enable_mamba_extra_buffer: bool,
        draft_model_idx: int,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
    ) -> HybridReqToTokenPool:
        """Shallow copy that shares the req_to_token mapping but owns a fresh mamba
        pool keyed on a single draft layer. Used by multi-layer EAGLE draft workers:
        each draft head shares the target's request-to-token mapping but needs its
        own sconv/mamba cache at layer_id=draft_model_idx.
        """
        clone = copy.copy(self)
        clone._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=[draft_model_idx],
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
        )
        clone.req_index_to_mamba_index_mapping = self.req_index_to_mamba_index_mapping
        if enable_mamba_extra_buffer:
            clone.req_index_to_mamba_ping_pong_track_buffer_mapping = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping
            )
        return clone

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        fresh_req_rows = [req.kv.req_pool_idx is None for req in reqs]
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        spec_write_pos = getattr(self.mamba_pool, "replayssm_spec_write_pos", None)
        if spec_write_pos is not None:
            fresh_indices = [
                idx for idx, fresh in zip(select_index, fresh_req_rows) if fresh
            ]
            if fresh_indices:
                spec_write_pos[fresh_indices] = 0
                self.mamba_pool.replayssm_cache_base[fresh_indices] = 0
                self.mamba_pool.replayssm_is_flush[fresh_indices] = 0

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            if req.kv.holds_mamba:  # for radix cache / continuing chunked
                pass
            else:
                mid = self.mamba_allocator.alloc(1)
                assert mid is not None, (
                    f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_allocator.available_size()=}, {len(reqs)=}"
                )
                req.kv.mamba_pool_idx = mid[0]
                req.kv.mamba_needs_clear = True
                # GDN ReplaySSM: a freshly (re)assigned slot starts an empty
                # ring. write_pos=0 means "ring empty", so the decode kernel
                # ignores ring contents and reads only the checkpoint state
                # (the post-prefill state that prefill wrote into this slot).
                if self.mamba_pool.replayssm_write_pos is not None:
                    self.mamba_pool.replayssm_write_pos[req.kv.mamba_pool_idx] = 0
            mamba_indices.append(req.kv.mamba_pool_idx)
            if self.enable_mamba_extra_buffer:
                if req.kv.mamba_ping_pong_track_buffer is None:
                    self._alloc_ping_pong_buffer(req)
                mamba_ping_pong_track_buffers.append(
                    req.kv.mamba_ping_pong_track_buffer
                )
        assert len(select_index) == len(mamba_indices), (
            "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        )
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(mamba_ping_pong_track_buffers), (
                "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
            )
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    @property
    def mamba_v2p_table(self) -> Optional[torch.Tensor]:
        """The mamba virtual->physical slot table, or None when the ids this
        pool hands out are already physical."""
        return None

    @property
    def mamba_translate_is_fusable(self) -> bool:
        """Whether `fused_replay_state_indices` can reproduce this pool's
        `translate_mamba_indices` in its own launch.

        The kernel expresses exactly two shapes: the identity, and one gather
        through `mamba_v2p_table`. A subclass that replaces the translate with
        anything else is excluded here rather than silently mis-served.
        """
        if self.mamba_v2p_table is not None:
            return True
        return (
            type(self).translate_mamba_indices
            is HybridReqToTokenPool.translate_mamba_indices
        )

    def translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        """Virtual->physical mamba-slot translate. Identity for a static pool
        (slots are physical); UnifiedHybridReqToTokenPool overrides it for the
        unified memory pool, where mamba slot ids are virtual. Callers translate
        before calling the pool's physical-id state ops (copy_from / clear_slots
