# sglang 0.5.21 (tag v0.5.21) managers/scheduler.py lines 3784-3990
    def get_num_allocatable_reqs(
        self,
        running_bs: int,
        beam_width: Optional[int] = None,
        running_batch: Optional[ScheduleBatch] = None,
    ) -> int:
        pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
        available = self.req_to_token_pool.available_size()

        active_batch = running_batch or self.running_batch
        available = max(
            available - self.beam_coordinator.pending_member_rows(active_batch), 0
        )
        res = min(pp_budget, available)
        if beam_width is not None:
            # A beam candidate owns beam_width rows once decoding.
            res = min(res, available // beam_width)

        return res

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get max usage across all pools for prefill delay decision
            max_pool_usage = (
                self.pool_stats_observer.get_pool_stats().get_max_pool_usage()
            )
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )

        ret, running_batch = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            running_batch=running_batch,
        )

        if self.prefill_delayer:
            observed_prefill_bs = prefill_delayer_single_pass.finalize(
                actual_prefill_bs=ret.batch_size() if ret is not None else 0
            )
            if observed_prefill_bs > 0:
                self.max_prefill_bs = self.prefill_bs_tracker.observe_attempt(
                    observed_prefill_bs
                )

        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def _get_new_batch_prefill_raw(
        self,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor],
        running_batch: ScheduleBatch,
    ) -> Tuple[Optional[ScheduleBatch], ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            running_batch.batch_is_full = False

        if (
            running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None, running_batch

        running_bs = len(running_batch.reqs)
        # Skipped during a chunked prefill: that pass must proceed regardless.
        if (
            self.min_free_slots_delayer is not None
            and self.chunked_req is None
            and self.min_free_slots_delayer.should_delay(
                running_bs=running_bs,
                num_allocatable_reqs=self.get_num_allocatable_reqs(
                    running_bs, running_batch=running_batch
                ),
            )
        ):
            return None, running_batch

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs, running_batch=running_batch) <= 0
            and self.chunked_req is None
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return None, running_batch

        # Get priority queue
        self.policy.calc_priority(
            self.waiting_queue,
            running_batch,
            processed_tokens=self.processed_tokens_counter,
        )

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None, running_batch

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.dynamic_chunk_sizer is not None:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.dynamic_chunk_sizer.predict(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        # Get BLOCK_M from the backend for tile-budget admission logic
        attn_backend = self.tp_worker.model_runner.attn_backend
        if hasattr(attn_backend, "extend_attention_block_m"):
            prefill_tile_block_m = attn_backend.extend_attention_block_m
        else:
            prefill_tile_block_m = 64  # Fallback for non-Triton backends

        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=int(self.max_prefill_bs),
            max_running_requests=self.max_running_requests,
            prefill_max_requests=get_schedule().prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
            waiting_queue_len=len(self.waiting_queue),
            prefill_tile_block_m=prefill_tile_block_m,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            adder.chunked_req_limit = self.policy.shortest_prefill_chunk_limit(
                self.chunked_req,
                self.waiting_queue,
                adder.rem_chunk_tokens or 0,
                self.page_size,
            )
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {
                req.lora_id for req in running_batch.reqs if not req.finished()
            }
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            running_loras.update(req.lora_id for req in adder.can_run_list)

            if self.lora_drainer:
                self.lora_drainer.update_draining_state(
                    self.waiting_queue,
                    running_batch.reqs,
                )

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.waiting_queue))
        buffer_pipeline = self.tree_cache.buffer_pipeline
        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and not self.can_schedule_lora_req(req, running_loras):
                continue

            # A forward batch runs one pooling mode, so setwise readout requests
            # (token_indices_to_pool) cannot share a batch with last-token ones.
            # Admit only the first request's mode; the other stays queued.
            if adder.can_run_list and (
                (req.token_indices_to_pool is not None)
                != (adder.can_run_list[0].token_indices_to_pool is not None)
            ):
                continue

            running_bs = len(running_batch.reqs)
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

            if running_batch.batch_is_full:
                if not self.enable_priority_preemption or not adder.preempt_to_schedule(
                    req
                ):
                    break

            if self.enable_hicache_storage or self.enable_lmcache:
                prefetch_done = self.tree_cache.check_prefetch_progress(
