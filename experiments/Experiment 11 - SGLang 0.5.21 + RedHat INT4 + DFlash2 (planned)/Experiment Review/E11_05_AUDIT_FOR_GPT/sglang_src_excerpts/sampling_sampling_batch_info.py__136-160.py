# sglang 0.5.21 sampling/sampling_batch_info.py lines 136-160
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs],
            dtype=torch.float,
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        sampling_seed = (
            torch.tensor(
                [
                    (
                        r.sampling_params.sampling_seed
                        if r.sampling_params.sampling_seed is not None
                        else 42
                    )
                    for r in reqs
                ],
                dtype=torch.int64,
                pin_memory=_pin,
            ).to(device, non_blocking=True)
            if enable_deterministic
            else None
        )

        logit_bias = None
