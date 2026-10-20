# sglang 0.5.21 (tag v0.5.21) mem_cache/kv_cache_configurator.py lines 2232-2345
    def _calculate_mamba_ratio(self) -> int:
        if get_memory().disable_radix_cache:
            return 1

        skip_decode_lock = envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.get()
        base = MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO - (
            MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP if skip_decode_lock else 0
        )

        additional_ratio = 0
        if get_exec().mamba.enable_mamba_extra_buffer:
            # ping-pong buffer size is 2 when overlap schedule is on, 1 otherwise.
            # Lazy mode saves 1 slot (2 → 1) for overlap; non-overlap already uses 1.
            if not get_schedule().disable_overlap_schedule:
                if get_exec().mamba.enable_mamba_extra_buffer_lazy:
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY
                else:
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                assert not get_exec().mamba.enable_mamba_extra_buffer_lazy, (
                    "Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)"
                )
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP
        elif skip_decode_lock:
            # no_buffer under skip: add the base drop back so effective stays 3,
            # the prefill->decode peak needs ~3 slots/req and this leaf-only mode
            # has no ping-pong to absorb it.
            additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_BUFFER

        return base + additional_ratio

    def _apply_token_constraints(self, token_capacity: int) -> int:
        """Apply external constraints to token capacity: user cap, PP sync.

        Page alignment is handled by the configurator, not here.
        If constraints change the value, the configurator re-runs and re-aligns.
        """
        user_limit = get_schedule().max_total_tokens

        # Apply user-specified upper bound
        if user_limit is not None:
            if user_limit > token_capacity:
                logging.warning(
                    f"max_total_tokens={user_limit} is larger than the profiled value "
                    f"{token_capacity}. Use the profiled value instead."
                )
            token_capacity = min(token_capacity, user_limit)

        # Sync across PP ranks (each may have different layer counts)
        if get_parallel().pp_size > 1:
            tensor = torch.tensor(token_capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_parallel().world_group.cpu_group,
            )
            token_capacity = tensor.item()

        return token_capacity

    def resolve_max_num_reqs(self, token_capacity: int) -> int:
        """Compute max concurrent requests (per dp worker) from the finalized
        token capacity."""
        # Estimate pool size (used as upper bound when user specifies max_running_requests)
        estimated = int(token_capacity / self.model_config.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)

        max_num_reqs = get_schedule().max_running_requests
        if max_num_reqs is not None:
            requested_per_worker = max_num_reqs // self.attn_dp_size
            max_num_reqs = min(requested_per_worker, token_capacity // 2)
        else:
            requested_per_worker = None
            max_num_reqs = min(estimated, token_capacity // 2)

        capped_by_mamba = False
        if self.mambaish_config is not None:
            ratio = self._calculate_mamba_ratio()
            mamba_cap = get_schedule().max_mamba_cache_size // ratio
            if mamba_cap < max_num_reqs:
                capped_by_mamba = True
                logger.warning(
                    "max_running_requests is capped to %d by the mamba state "
                    "cache (max_mamba_cache_size=%d, %d state slots per "
                    "request). To raise it: increase --mamba-full-memory-ratio "
                    "or --max-mamba-cache-size, or halve the state size with "
                    "--mamba-ssm-dtype bfloat16.",
                    mamba_cap,
                    get_schedule().max_mamba_cache_size,
                    ratio,
                )
            max_num_reqs = min(max_num_reqs, mamba_cap)

            if max_num_reqs <= 0:
                raise RuntimeError(
                    f"Hybrid (mamba/linear-attention) state cache is too small to serve "
                    f"any requests. max_mamba_cache_size={get_schedule().max_mamba_cache_size}, "
                    f"mamba_ratio={ratio}, resulting max_num_reqs={max_num_reqs}. "
                    f"Try: (1) reduce --max-running-requests, "
                    f"(2) increase --mem-fraction-static, or "
                    f"(3) use GPUs with more memory."
                )
        if (
            requested_per_worker is not None
            and max_num_reqs < requested_per_worker
            and not capped_by_mamba
        ):
            logger.warning(
                "max_running_requests was reduced from the requested %d to %d "
                "(per dp worker) due to the available KV cache capacity.",
                requested_per_worker,
                max_num_reqs,
            )
        return max_num_reqs
