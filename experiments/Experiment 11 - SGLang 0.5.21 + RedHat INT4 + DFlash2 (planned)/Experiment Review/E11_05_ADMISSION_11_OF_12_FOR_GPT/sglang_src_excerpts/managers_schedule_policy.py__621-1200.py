# sglang 0.5.21 (tag v0.5.21) managers/schedule_policy.py lines 621-1200
class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        num_mixed_decode_tokens: int = 0,
        priority_scheduling_preemption_threshold: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: Optional[int] = None,
        prefill_max_requests: Optional[int] = None,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
        dllm_config: Optional[DllmConfig] = None,
        waiting_queue_len: int = 0,
        prefill_tile_block_m: int = 64,
    ):
        self.page_size = page_size
        self.prefill_tile_block_m = prefill_tile_block_m
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        self.chunked_req_limit: Optional[int] = None
        self.dllm_config = dllm_config
        self.exact_chunk_fill = _use_exact_chunk_fill() and dllm_config is None

        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= num_mixed_decode_tokens
        self.memory_budget = token_to_kv_pool_allocator.create_prefill_budget(
            tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

        self.req_states = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        self.reprocessed_log_hit_tokens = 0
        self.log_device_hit_tokens = 0
        self.log_host_hit_tokens = 0
        self.log_storage_hit_tokens = 0
        self.log_input_tokens = 0
        self.log_replay_tokens = 0
        self.reprocessed_log_input_tokens = 0

        if running_batch is not None:
            # Estimate the offset in the remaining token space
            self.memory_budget.total_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )

        # DeepSeek V4 HiSparse wraps an SWATokenToKVPoolAllocator internally and
        # exposes the full SWA allocator interface.
        self.is_hybrid_swa = isinstance(
            self.token_to_kv_pool_allocator,
            (SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),
        )
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()
        # A new state slot eats shared-gap bytes that `rem_total_tokens` counts
        # as free, so reserve per slot or admission over-commits. Gate on the
        # ALLOCATOR, not `is_hybrid_ssm_cache`: that is False for `ChunkCache`,
        # which would skip the reservation on the chunk-cache path.
        self._mamba_slot_cost = 0
        if isinstance(
            self.token_to_kv_pool_allocator,
            (
                UnifiedMambaTokenToKVPoolAllocator,
                UnifiedMambaSWATokenToKVPoolAllocator,
            ),
        ):
            self._mamba_slot_cost = (
                self.token_to_kv_pool_allocator.mamba_slot_full_token_cost()
            )

        # `mamba_gap_reserve` is charged to `rem_total_tokens`, which INCLUDES
        # `full_evictable_size()` — but `alloc_req_slots` can only recover
        # MAMBA-recoverable bytes for a mamba slot (shared gap + peer holes +
        # mamba-evictable radix), NOT full-evictable. Gate new mamba slots on
        # that mamba-recoverable budget separately or an over-admit hits the
        # fail-loud `RuntimeError`. `None` outside the unified Mamba pool.
        self.rem_mamba_slots = None
        if self._mamba_slot_cost:
            self.rem_mamba_slots = self.token_to_kv_pool_allocator.mamba_allocator.schedulable_available_size()
            if self.is_hybrid_ssm_cache:
                self.rem_mamba_slots += self.tree_cache.mamba_evictable_size()

        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        self.max_running_requests = max_running_requests
        # Align sharded chunk boundaries to physical pages; 0 disables alignment.
        kv_shard_size = page_interleave_shard_size(self.token_to_kv_pool_allocator)
        self.kv_shard_granule = (
            self.token_to_kv_pool_allocator.page_size if kv_shard_size > 1 else 0
        )
        self.kv_shard_size = kv_shard_size
        # Each admission must fit the batch's prefix and chunk scratch regions.
        self.kv_shard_scratch_spec = (
            self.token_to_kv_pool_allocator.shard_spec if kv_shard_size > 1 else None
        )
        self.kv_shard_block_bound_pages = 0
        self.kv_shard_chunk_pages = 0
        # Reserve one extra page per shard class for allocation rounding.
        self.per_req_token_overhead = (
            kv_shard_size * self.kv_shard_granule
            if self.kv_shard_granule
            else self.page_size
        )
        self.prefill_max_requests = prefill_max_requests
        self.prefill_delayer_single_pass = prefill_delayer_single_pass
        self.max_prefill_bs = max_prefill_bs
        # Snapshot of scheduler waiting_queue length at the start of this
        # prefill pass. Used by PrefillDelayer's queue-based trigger.
        self.waiting_queue_len = waiting_queue_len

    def _admitted_extend_lens(self) -> List[int]:
        return [int(getattr(req, "extend_input_len", 0)) for req in self.can_run_list]

    def _tile_admission_metric_key(self) -> str:
        return f"{PREFILL_TILE_BUDGET_MODE}_q_tiles_per_head"

    def _candidate_tile_metrics(self, candidate_extend_len: int) -> Dict[str, object]:
        return estimate_prefill_extend_tile_metrics(
            [*self._admitted_extend_lens(), int(candidate_extend_len)],
            block_m=self.prefill_tile_block_m,
        )

    def _check_prefill_tile_budget(
        self, candidate_extend_len: int
    ) -> Optional[AddReqResult]:
        # AMD-only: leave non-AMD scheduler admission unchanged even if the env
        # budget is set.
        if not _IS_HIP or PREFILL_TILE_BUDGET <= 0:
            return None

        if not self.can_run_list:
            return None

        metrics = self._candidate_tile_metrics(candidate_extend_len)
        candidate_metric = int(metrics.get(self._tile_admission_metric_key()) or 0)
        if candidate_metric <= PREFILL_TILE_BUDGET:
            return None

        return AddReqResult.OTHER

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests

        self.rem_dllm_tokens = (
            self.rem_input_tokens
            if dllm_config.requires_separate_context_encoding
            else max_running_reqs * self.dllm_block_size
        )

    def _get_running_request_total_token_offset(self, req: Req) -> int:
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio
        )

    @property
    def rem_total_tokens(self):
        return self.memory_budget.remaining_total

    @property
    def cur_rem_tokens(self):
        return self.memory_budget.remaining_current

    def _swa_new_tokens(self, req: Req) -> int:
        """Tokens a request may still decode, for SWA headroom sizing. Mirrors
        the remaining-then-clip order of add_one_req's max_new: clip-then-subtract
        would zero out a request that has already generated >= CLIP tokens but
        still has a long decode ahead, under-reserving its window."""
        return min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )

    def _check_prefill_budget(
        self,
        req: Req,
        *,
        extend_input_len: int,
        total_tokens: int,
        swa_host_hit_length: int,
    ) -> tuple[bool, Optional[int]]:
        return self.memory_budget.check_prefill(
            extend_input_len=extend_input_len,
            total_tokens=total_tokens,
            max_new_tokens=self._swa_new_tokens(req),
            input_tokens=len(req.full_untruncated_fill_ids),
            swa_host_hit_length=swa_host_hit_length,
            chunk_limit=self.rem_chunk_tokens,
        )

    def _mamba_gap_budget_for_req(self, req: Req) -> int:
        """Shared-gap reservation (full-token-equivalents) for a request's new
        mamba state. Charged only on the SHARED Mamba pool (`_mamba_slot_cost > 0`)
        and only when the req has no state yet (`mamba_pool_idx is None`, mirroring
        `HybridReqToTokenPool.alloc`); 0 keeps baseline / SWA / non-Mamba unchanged.

        Conservative by design (`_mamba_slot_cost` rounds UP). Does NOT reserve
        radix COW headroom or locked-but-evictable bytes — that residual is
        backstopped by the fail-loud RuntimeError in `alloc_req_slots`. FIXME: if
        over-admission crashes under pressure, make this more conservative (e.g.
        multiply by `MAMBA_STATE_PER_REQ_PREFIX_CACHE`)."""
        if self._mamba_slot_cost and not req.kv.holds_mamba:
            return self._mamba_slot_cost
        return 0

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        no_token = not self.memory_budget.has_capacity()
        # Gate new mamba slots separately: rem_total_tokens' full_evictable can't
        # cover a mamba slot, which needs mamba-recoverable bytes (see __init__).
        if not no_token and self.rem_mamba_slots is not None:
            no_token = self.rem_mamba_slots <= 0
        if no_token:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER
        else:
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self,
        prefix_len: int,
        extend_input_len: int,
        max_new_tokens: int,
        retracted_stain: bool,
        mamba_gap_reserve: int = 0,
        is_chunked_continuation: bool = False,
        compute_charge: Optional[int] = None,
    ):
        """Charge one admitted request against the prefill budgets.

        `compute_charge` is what the compute budgets (`rem_chunk_tokens`,
        `rem_input_tokens`, `rem_dllm_tokens`) are billed, counted in
        forward-pass tokens; the KV budgets are always billed page-ceiled
        tokens. It defaults to the ceiled count, so only the exact-chunk-fill
        path parts from upstream behaviour. Both compute budgets have to take
        it: they are usually configured to the same value, so leaving either one
        ceiled makes it hit zero first and stop admission with the rounding
        slack unspent.
        """
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        raw_extend_input_len = extend_input_len
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        if compute_charge is None:
            compute_charge = extend_input_len

        self.memory_budget.reserve(
            extend_input_len,
            max_new_tokens,
            # reserve() already charges one page; add the remaining shard pages.
            extra_tokens=(
                mamba_gap_reserve + self.per_req_token_overhead - self.page_size
            ),
            chunk_limit=self.rem_chunk_tokens,
            is_chunked_continuation=is_chunked_continuation,
        )
        # The new mamba slot also consumes one mamba-recoverable slot (gated
        # separately so full_evictable can't cover it — see __init__).
        if mamba_gap_reserve and self.rem_mamba_slots is not None:
            self.rem_mamba_slots -= 1
        self.rem_input_tokens -= compute_charge

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= compute_charge
        elif self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= compute_charge

        # reprocessed_log_* is a subset of log_*; metrics_reporter subtracts it
        # when computing the first-attempt prefix cache hit rate.
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += raw_extend_input_len
        if retracted_stain:
            self.reprocessed_log_hit_tokens += prefix_len
            self.reprocessed_log_input_tokens += raw_extend_input_len

    def _account_prefill_cache_admission(self, req: Req, prefix_len: int) -> None:
        if get_exec().features.enable_encoder_swa_bounded_replay and (
            req.kv.req_pool_idx is None or req.is_retracted
        ):
            replay_tokens = min(prefix_len, 128)
            self.log_replay_tokens += replay_tokens
            self.rem_input_tokens -= replay_tokens
            if self.rem_chunk_tokens is not None:
                self.rem_chunk_tokens -= replay_tokens
        if req.retracted_stain:
            # Retraction attribution is intentionally omitted for now; discard
            # its lifecycle state so a later abort cannot report it as a drop.
            self.tree_cache.discard_storage_prefetch_accounting(
                req.cache_request_handle
            )
            return

        if prefix_len > 0:
            device_hit, host_hit, storage_hit = split_cached_prefix_by_tier(
                prefix_len=prefix_len,
                host_hit_len=req.materialized_host_hit_len(),
                storage_hit_len=req.storage_hit_length,
                storage_hit_start=req.storage_hit_start,
                host_hit_is_storage=req.host_hit_is_storage,
            )
            self.log_device_hit_tokens += device_hit
            self.log_host_hit_tokens += host_hit
            self.log_storage_hit_tokens += storage_hit

        fulfilled_storage_hit = req.fulfilled_storage_hit_len(prefix_len)
        reason = None
        if fulfilled_storage_hit < req.storage_hit_length:
            reason = (
                "device_capacity"
                if req.needs_host_load_back()
                and req.host_loaded_length < req.host_hit_length
                else "cache_admission_shortfall"
            )
        self.tree_cache.finish_storage_prefetch_admission(
            req.cache_request_handle,
            fulfilled_tokens=fulfilled_storage_hit,
            reason=reason,
        )

    def _get_dllm_remain_tokens(self, req: Optional[Req] = None) -> int:
        _rem_tokens = min(self.rem_dllm_tokens, int(self.rem_total_tokens))
        if not (
            self.dllm_config.requires_separate_context_encoding
            and req is not None
            and req.is_dllm_prefill()
        ):
            _rem_tokens = min(_rem_tokens, self.dllm_block_size)
        if _rem_tokens <= 0:
            _rem_tokens = self.rem_dllm_tokens

        if self.dllm_config.requires_separate_context_encoding:
            _rem_tokens = min(_rem_tokens, int(self.cur_rem_tokens) - self.page_size)
            if self.is_hybrid_swa:
                _rem_tokens = min(
                    _rem_tokens, int(self.memory_budget.remaining_swa) - self.page_size
                )
        return _rem_tokens

    def _get_dllm_extend_len(self, req: Req, prefix_len: int) -> int:
        if self.dllm_config.requires_separate_context_encoding:
            remaining = (
                req.dllm_block_offset - prefix_len
                if req.is_dllm_prefill()
                else len(req.full_untruncated_fill_ids) - prefix_len
            )
            limit = self.rem_dllm_tokens
            if not req.is_dllm_prefill():
                limit = min(limit, self.dllm_block_size)
            trunc_len = min(limit, remaining)
        else:
            trunc_len = (
                min(self.rem_dllm_tokens, self.dllm_block_size)
                // self.page_size
                * self.page_size
            )

        return trunc_len

    def _add_dllm_req(self, req: Req, prefix_len: int):
        trunc_len = self._get_dllm_extend_len(req, prefix_len)
        req.set_extend_range(prefix_len, prefix_len + trunc_len)

        self.can_run_list.append(req)

        self._update_prefill_budget(
            prefix_len,
            trunc_len,
            0,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
        )
        self._account_prefill_cache_admission(req, prefix_len)

    def _req_inc_lock_ref(self, req: Req):
        # Persist the release receipt.
        req.lock_receipt = self.tree_cache.inc_lock_ref(req.last_node).to_dec_params()

    def _kv_shard_reserve_scratch(self, prefix_len: int, extend_len: int) -> bool:
        """Reserve scratch or return False to defer; no-op when sharding is off.

        Prefixes use N * sum_i ceil(prefix_pages_i / N) pages; extends use
        sum_i ceil(extend_i / ps). A request must fit an empty batch or raise.
        """
        if self.kv_shard_scratch_spec is None:
            return True
        ps = self.kv_shard_granule
        shard_size = self.kv_shard_size
        block = self.kv_shard_block_bound_pages + -(-(prefix_len // ps) // shard_size)
        chunk = self.kv_shard_chunk_pages + -(-extend_len // ps)
        if (
            shard_size * block * ps > self.kv_shard_scratch_spec.max_prefix_tokens
            or chunk * ps > self.kv_shard_scratch_spec.chunk_tokens
        ):
            if self.kv_shard_block_bound_pages == 0 and self.kv_shard_chunk_pages == 0:
                # Deferring a request that cannot fit an empty batch would livelock.
                raise RuntimeError(
                    "request cannot fit the sharded assembly scratch even in "
                    f"an empty batch (prefix_len={prefix_len}, extend_len="
                    f"{extend_len}, max_prefix_tokens="
                    f"{self.kv_shard_scratch_spec.max_prefix_tokens}, "
                    f"chunk_tokens={self.kv_shard_scratch_spec.chunk_tokens}); "
                    "KV sharding requires chunked prefill sized within the "
                    "assembly scratch"
                )
            return False
        self.kv_shard_block_bound_pages = block
        self.kv_shard_chunk_pages = chunk
        return True

    def add_dllm_staging_req(self, req: Req):
        assert self.dllm_config is not None
        _rem_tokens = self._get_dllm_remain_tokens(req)

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        if (
            self.dllm_config.requires_separate_context_encoding
            and req.is_dllm_prefill()
        ):
            cand_extend_input_len = min(
                cand_extend_input_len,
                req.dllm_block_offset - len(req.prefix_indices),
            )
        if (
            req.dllm_incomplete_ids
            or (
                self.dllm_config.requires_separate_context_encoding
                and not req.is_dllm_prefill()
            )
        ) and cand_extend_input_len > _rem_tokens:
            return AddReqResult.NO_TOKEN
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            max_new_tokens,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
        )

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens(req) <= 0
            else AddReqResult.CONTINUE
        )

    def add_chunked_req(self, req: Req):
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens(req)
        else:
            _rem_tokens = self.memory_budget.available_chunk_tokens(
                self.rem_chunk_tokens
            )
            if _rem_tokens is None:
                return req
            if self.kv_shard_granule:
                # Align the absolute boundary to keep pages within one scratch region.
                # If no budget remains, use a full chunk to make progress.
                prefix_len = len(req.prefix_indices)
                floored = (
                    prefix_len + _rem_tokens
                ) // self.kv_shard_granule * self.kv_shard_granule - prefix_len
                _rem_tokens = floored if floored > 0 else self.rem_chunk_tokens

        # A mid-chunk rank prefills this pass regardless of the delayer
        # verdict, so report prefillable=True and ignore the result.
        if self.prefill_delayer_single_pass is not None:
            self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )

        if self.chunked_req_limit is not None:
            assert self.chunked_req_limit > 0
            _rem_tokens = min(_rem_tokens, self.chunked_req_limit)

        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        _rem_tokens = self.memory_budget.fit_chunk(
            extend_input_len=cand_extend_input_len,
            max_new_tokens=self._swa_new_tokens(req),
            chunk_limit=_rem_tokens,
        )
        if _rem_tokens is None:
            return req
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        # The continuing chunk must fit. Keep reservation outside assert for -O.
        reserved = self._kv_shard_reserve_scratch(
            prefix_len=len(req.prefix_indices), extend_len=new_len
        )
        if not reserved:
            raise RuntimeError("chunked request exceeds the sharded assembly scratch")
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            is_chunked_continuation=True,
            compute_charge=req.extend_range.length if self.exact_chunk_fill else None,
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        dec_lock_params = None
        try:
            result = self.tree_cache.inc_lock_ref(last_node)
            if self.tree_cache.is_tree_cache():
                # Replay the acquire's receipt (SWA boundary uuid, mamba flag)
                # so release takes back exactly what this temporary lock took.
                dec_lock_params = result.to_dec_params()
            yield None
        finally:
            if dec_lock_params is not None:
                self.tree_cache.dec_lock_ref(last_node, dec_lock_params)
            else:
                self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req):
