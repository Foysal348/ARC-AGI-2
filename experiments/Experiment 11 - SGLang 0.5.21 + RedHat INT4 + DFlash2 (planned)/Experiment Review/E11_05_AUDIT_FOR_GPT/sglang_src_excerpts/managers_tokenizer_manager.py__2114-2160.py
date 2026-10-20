# sglang 0.5.21 managers/tokenizer_manager.py lines 2114-2160
    def abort_request(self, rid: str = "", abort_all: bool = False):
        # Empty rid would startswith-match every request on the scheduler.
        if not abort_all and not rid:
            logger.warning("Ignore abort_request with empty rid and abort_all=False")
            return
        state = None if abort_all else self.rid_to_state.get(rid)
        if not abort_all:
            if state is not None:
                if state.abort_sent:
                    return
                state.abort_sent = True
            elif get_serving().tokenizer_worker_num == 1:
                return
        req = AbortReq(rid=rid, abort_all=abort_all)
        try:
            self._dispatch_to_scheduler(req)
        except BaseException:
            if state is not None:
                state.abort_sent = False
            raise
        if self.enable_metrics:
            # TODO: also use custom_labels from the request
            self.metrics_collector.observe_one_aborted_request(
                self.metrics_collector.labels
            )

    async def pause_generation(self, obj: PauseGenerationReqInput):
        async with self.is_pause_cond:
            self.is_pause = True
            if obj.mode != "abort":
                await self._async_dispatch_to_scheduler(obj)
            else:
                # we are using the model_update_lock to check if there is still on-going requests.
                while True:
                    # TODO: maybe make it async instead of fire-and-forget
                    self.abort_request(abort_all=True)
                    is_locked = await self.model_update_lock.is_locked()
                    if not is_locked:
                        break
                    await asyncio.sleep(1.0)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        async with self.is_pause_cond:
            self.is_pause = False
            await self._async_dispatch_to_scheduler(obj)
            self.is_pause_cond.notify_all()

