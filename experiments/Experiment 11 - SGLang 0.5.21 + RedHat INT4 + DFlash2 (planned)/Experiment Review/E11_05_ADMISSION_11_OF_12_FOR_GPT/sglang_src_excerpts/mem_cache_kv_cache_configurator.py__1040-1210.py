# sglang 0.5.21 (tag v0.5.21) mem_cache/kv_cache_configurator.py lines 1040-1210
        ]
        return {
            "short_conv_layer_ids": short_conv_layer_ids,
            "short_conv_state_shape": (
                self.mambaish_config.short_conv_state_shape
                if short_conv_layer_ids
                else None
            ),
            "ngram_context_len": (
                self.mambaish_config.ngram_context_len if short_conv_layer_ids else 0
            ),
            "ngram_eos_token_id": int(self.mambaish_config.eos_token_id),
        }

    def _build_hybrid_mamba_decode_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
        pre_alloc_size: int,
    ) -> ReqToTokenPool:
        from sglang.srt.disaggregation.decode import (
            HybridMambaDecodeReqToTokenPool,
        )

        req_to_token_pool = HybridMambaDecodeReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            cache_params=self.mambaish_config.mamba2_cache_params,
            mamba_layer_ids=self._get_mamba_layer_ids_for_req_pool(),
            speculative_num_draft_tokens=max_speculative_num_draft_tokens(),
            speculative_eagle_topk=get_spec().speculative_eagle_topk,
            enable_mamba_extra_buffer=get_exec().mamba.enable_mamba_extra_buffer,
            pre_alloc_size=pre_alloc_size,
            enable_overlap_schedule=not get_schedule().disable_overlap_schedule,
            mamba_size=get_schedule().max_mamba_cache_size,
            start_layer=self.layer_info.start_layer,
            **self._get_ple_req_pool_kwargs(),
            linear_replayssm_cache_len=get_exec().mamba.linear_replayssm_cache_len,
            mamba_envelope_layout=get_memory().enable_page_major_kv_layout,
            # ReplaySSM spec-verify is for linear-attn models (GDN fold or KDA
            # fold); activate the pool machinery only for those, so any other
            # mamba-ish model (Mamba2/Nemotron, lightning, ...) run with the
            # flag set stays byte-identical to flag-off.
            enable_linear_replayssm_spec=(
                get_exec().mamba.enable_linear_replayssm_spec
                and (
                    self.hybrid_gdn_config is not None
                    or self.hybrid_kda_config is not None
                )
            ),
        )
        return req_to_token_pool

    def _build_decode_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
        pre_alloc_size: int,
    ) -> ReqToTokenPool:
        if _is_npu and is_deepseek_v4(self.model_config.hf_config):
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                DSV4NPUDecodeReqToTokenPool as DecodeReqToTokenPool,
            )
        else:
            from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

        req_to_token_pool = DecodeReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )
        return req_to_token_pool

    def _build_hybrid_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
    ) -> ReqToTokenPool:
        # DSPARK/DFLASH commit routes through the backend fold (KDA-only); a
        # non-KDA model there would scatter a None intermediate_ssm and crash.
        _algo = (get_spec().speculative_algorithm or "").upper()
        if (
            get_exec().mamba.enable_linear_replayssm_spec
            and _algo in ("DSPARK", "DFLASH")
            and self.hybrid_kda_config is None
        ):
            raise ValueError(
                "--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA "
                "model; got a non-KDA model."
            )
        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=get_schedule().max_mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
            cache_params=self.mambaish_config.mamba2_cache_params,
            mamba_layer_ids=self._get_mamba_layer_ids_for_req_pool(),
            enable_mamba_extra_buffer=get_exec().mamba.enable_mamba_extra_buffer,
            enable_mamba_extra_buffer_lazy=get_exec().mamba.enable_mamba_extra_buffer_lazy,
            **self._get_ple_req_pool_kwargs(),
            # A PD prefill server never runs TARGET_VERIFY, so skip the
            # verify-only per-draft-token state snapshots (see the draft-head
            # case above: None => the pool skips SpeculativeState).
            speculative_num_draft_tokens=(
                None
                if get_disagg().disaggregation_mode == "prefill" and not _is_npu
                else max_speculative_num_draft_tokens()
            ),
            speculative_eagle_topk=get_spec().speculative_eagle_topk,
            enable_overlap_schedule=not get_schedule().disable_overlap_schedule,
            start_layer=self.layer_info.start_layer,
            enable_linear_replayssm=get_exec().mamba.enable_linear_replayssm,
            linear_replayssm_cache_len=get_exec().mamba.linear_replayssm_cache_len,
            mamba_envelope_layout=get_memory().enable_page_major_kv_layout,
            # ReplaySSM spec-verify is for linear-attn models (GDN fold or KDA
            # fold); activate the pool machinery only for those, so any other
            # mamba-ish model (Mamba2/Nemotron, lightning, ...) run with the
            # flag set stays byte-identical to flag-off.
            enable_linear_replayssm_spec=(
                get_exec().mamba.enable_linear_replayssm_spec
                and (
                    self.hybrid_gdn_config is not None
                    or self.hybrid_kda_config is not None
                )
            ),
        )
        return req_to_token_pool

    def _build_default_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
    ) -> ReqToTokenPool:
        # DSV4 on NPU needs an extended ReqToTokenPool holding per-req
        # swa/c4/c128/c{4,128}_state tables; others stay on the stock one.
        req_to_token_pool_cls = ReqToTokenPool
        if _is_npu and is_deepseek_v4(self.model_config.hf_config):
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                DSV4NPUReqToTokenPool,
            )

            req_to_token_pool_cls = DSV4NPUReqToTokenPool

        req_to_token_pool = req_to_token_pool_cls(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=get_exec().features.enable_memory_saver,
        )
        return req_to_token_pool

    def _build_token_to_kv_pool(
        self,
        *,
        sizes: _PoolSizes,
        is_dsa_model: bool,
        is_dsv4_model: bool,
        req_to_token_pool: ReqToTokenPool,
    ) -> KVCache:
        # Page-granularity envelope layout for the MHA-shaped (full / SWA) pools,
        # selected by swapping in the PageMajorMHATokenToKVPool subclass. The
