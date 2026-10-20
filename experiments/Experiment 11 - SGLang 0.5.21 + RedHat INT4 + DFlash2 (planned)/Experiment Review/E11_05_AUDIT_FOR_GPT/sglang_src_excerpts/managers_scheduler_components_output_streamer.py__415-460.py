# sglang 0.5.21 managers/scheduler_components/output_streamer.py lines 415-460
        return req.is_beam_leader and req.finished()

    def accept(self, *, req: Req) -> None:
        if req.beam_group is not None and not self._beam_admits(req=req):
            return
        if req.finished():
            assert not req.finished_output
            req.finished_output = True
            if req.finished_len is None:
                req.finished_len = len(req.output_ids)
            should_output = True
        else:
            if req.stream:
                stream_interval = (
                    req.sampling_params.stream_interval or self.default_stream_interval
                )

                # origin stream_interval logic
                should_output = (
                    len(req.output_ids) % stream_interval == 1
                    if stream_interval > 1
                    else len(req.output_ids) % stream_interval == 0
                )

                if should_output:
                    # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                    should_output &= not req.check_match_stop_str_prefix()
            else:
                should_output = (
                    len(req.output_ids) % self.default_force_stream_interval == 0
                )

        if not should_output:
            return

        send_token_offset = req.send_token_offset
        send_output_token_logprobs_offset = req.send_output_token_logprobs_offset
        self.rids.append(req.rid)
        self.output_reqs.append(req)
        self.finished_reasons.append(
            req.finished_reason.to_json() if req.finished_reason else None
        )
        # Exclude the tokens after stop condition
        output_ids_ = req.output_ids_through_stop
        self.output_ids.append(output_ids_[send_token_offset:])
        req.send_token_offset = len(output_ids_)
