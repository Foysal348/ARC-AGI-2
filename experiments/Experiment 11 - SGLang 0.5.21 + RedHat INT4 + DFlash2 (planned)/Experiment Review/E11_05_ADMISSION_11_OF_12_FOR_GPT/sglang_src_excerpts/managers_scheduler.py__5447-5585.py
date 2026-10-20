# sglang 0.5.21 (tag v0.5.21) managers/scheduler.py lines 5447-5585
    def abort_request(self, recv_req: AbortReq):
        if (chunked_req := self.chunked_req) is not None:
            if recv_req.abort_all or chunked_req.rid.startswith(recv_req.rid):
                self._pending_chunked_abort_req = chunked_req

        # todo hisparse, release resources for abort requests in hisparse coordinator
        # Abort requests still waiting for encoder embeddings (EPD language-only)
        if self.mm_receiver is not None:
            self.mm_receiver.abort_waiting_requests(recv_req)

        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            self._release_aborted_request(req)
            self.beam_coordinator.retire_group(req)
            # Without the initiator's reason the tokenizer falls back to a
            # generic abort message.
            self.ipc_channels.send_to_tokenizer.send_output(
                _make_abort_req(req, finished_reason=recv_req.finished_reason), req
            )
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if get_disagg().disaggregation_decode_host_receive_threshold > 0:
                    discard_kv_cache_backup(req, self.tree_cache, "host_pool")
                release_kv_cache(req, self.tree_cache)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.release_aborted_prefill_waiting_req(req)

            # For mamba radix cache
            if req.kv.holds_mamba and self.disaggregation_mode not in (
                DisaggregationMode.PREFILL,
                DisaggregationMode.DECODE,
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        if self.dllm_config is not None:
            for req in self.dllm_manager.pop_aborted_reqs(
                recv_req.abort_all, recv_req.rid
            ):
                self._release_aborted_request(req)
                self.ipc_channels.send_to_tokenizer.send_output(
                    _make_abort_req(req), req
                )
                if req.kv.holds_kv or req.kv.holds_mamba:
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                logger.debug(f"Abort dLLM queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    self._release_aborted_request(req)

                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()
                    if get_parallel().pp_size > 1:
                        prepare_abort(req, "Aborted by AbortReq.")

            # Abort in-flight requests
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()
                    if get_parallel().pp_size > 1:
                        prepare_abort(decode_req.req, "Aborted by AbortReq.")

            # Abort requests waiting for kvcache to release tree cache
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    if decode_req.host_staged:
                        # Keep the host destination alive until prefill stops writing.
                        prepare_abort(decode_req.req, "Aborted by AbortReq.")
                        continue
                    # The receiver arms drain-ack accounting before sending the
                    # ABORT (see CommonKVReceiver._send_abort_notification), so
                    # an ack racing back is never dropped.
                    decode_req.kv_receiver.abort()

            # Abort requests whose KV is already backed up for retraction.
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        discard_kv_cache_backup(
                            decode_req,
                            self.tree_cache,
                            get_disagg().disaggregation_decode_retraction_backup,
                        )
                        self.ipc_channels.send_to_tokenizer.send_output(
                            _make_abort_req(decode_req), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        for req in self.collect_inflight_reqs():
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                if recv_req.abort_message:
                    # Timeout aborts carry an SLA message + 503 for the client.
                    req.to_finish = FINISH_ABORT(
                        recv_req.abort_message, HTTPStatus.SERVICE_UNAVAILABLE
                    )
                else:
                    req.to_finish = FINISH_ABORT()

