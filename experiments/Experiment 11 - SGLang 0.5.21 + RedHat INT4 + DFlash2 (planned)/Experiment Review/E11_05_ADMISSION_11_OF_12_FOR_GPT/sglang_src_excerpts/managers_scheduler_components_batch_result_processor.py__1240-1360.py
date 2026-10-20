# sglang 0.5.21 (tag v0.5.21) managers/scheduler_components/batch_result_processor.py lines 1240-1360
            )
        assert status == SamplingMaskStatus.INVALID
        return FINISH_ABORT(
            "The sampling backend selected a token outside its captured sampling "
            "support.",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "InternalServerError",
        )

    def _handle_sampling_mask_abort(self, req: Req) -> None:
        """Release a request whose sampled token must not be committed."""
        if req.multimodal_inputs is not None and req.session is None:
            req.multimodal_inputs.release_features()
        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.finalize_release_on_finish(req)
        else:
            if get_memory().enable_hisparse:
                self.hisparse_coordinator.request_finished(req)
            prepare_release = getattr(
                self.model_worker, "prepare_for_kv_cache_release", None
            )
            if callable(prepare_release):
                prepare_release(req)
            release_kv_cache(req, self.tree_cache, is_insert=False)
        req.time_stats.set_completion_time()

    def _handle_finish_state_updated_req(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
        logits_output: LogitsProcessorOutput,
    ):
        lazy = get_exec().mamba.enable_mamba_extra_buffer_lazy
        known_mamba_boundary = None
        completed_mamba_boundary = None
        lookahead = 0
        if batch.mamba_track_mask_cpu is not None:
            completed_mamba_boundary = bool(batch.mamba_track_mask_cpu[i])
            lookahead = req.decode_batch_idx - batch.mamba_decode_batch_idx_cpu[i]
            assert lookahead in (0, 1), (
                f"mamba result lookahead={lookahead} for req {req.rid}; "
                "overlap advanced more than one decode batch"
            )
            if lookahead == 0:
                known_mamba_boundary = bool(batch.mamba_track_mask_cpu[i])
            else:
                known_mamba_boundary = bool(batch.mamba_track_mask_next_cpu[i])

            if completed_mamba_boundary and not lazy:
                req.kv.mamba_last_track_idx = batch.mamba_track_buffer_indices[i]
                req.kv.mamba_last_track_seqlen = req.kv.kv_committed_len - lookahead
            elif (
                req.finished()
                and lazy
                and lookahead == 1
                and known_mamba_boundary
                and req.kv.mamba_next_track_idx == req.kv.mamba_last_track_idx
            ):
                req.mamba_lazy_is_insert = False

        # Called here (after update_finish_state) so req.finished() is valid
        # for mamba_lazy_post_decode_at_boundary inside.
        should_update = completed_mamba_boundary if lazy else known_mamba_boundary
        if should_update is None or should_update:
            self._mamba_prefix_cache_update(
                req,
                batch,
                result,
                i,
                known_boundary=not lazy and known_mamba_boundary is True,
            )

        if (
            get_disagg().disaggregation_decode_enable_offload_kvcache
            and not req.finished()
        ):
            self.decode_offload_manager.offload_kv_cache(req)

        if req.finished():
            # isinstance narrowing: create_worker may also return plain
            # TpModelWorker-based drafts, which carry no spec-worker hooks.
            if isinstance(self.draft_worker, BaseSpecWorker):
                self.draft_worker.note_request_finished(
                    rid=req.rid,
                    natural_stop=isinstance(req.finished_reason, FINISH_MATCHED_TOKEN),
                )

            # delete feature to save memory
            if req.multimodal_inputs is not None and req.session is None:
                req.multimodal_inputs.release_features()
            self._maybe_collect_routed_experts(req)
            self._maybe_collect_indexer_topk(req)

            if get_disagg().disaggregation_decode_enable_offload_kvcache:
                # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                if not self.decode_offload_manager.offload_kv_cache(req):
                    self.decode_offload_manager.finalize_release_on_finish(req)
            else:
                if get_memory().enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                prepare_release = getattr(
                    self.model_worker, "prepare_for_kv_cache_release", None
                )
                if callable(prepare_release):
                    prepare_release(req)
                is_insert = (
                    req.mamba_lazy_is_insert
                    if get_exec().mamba.enable_mamba_extra_buffer_lazy
                    else True
                )
                release_kv_cache(req, self.tree_cache, is_insert=is_insert)

            req.time_stats.set_completion_time()

        self._maybe_collect_customized_info(i, req, logits_output)

    def _maybe_update_reasoning_tokens(
        self,
        req: Req,
