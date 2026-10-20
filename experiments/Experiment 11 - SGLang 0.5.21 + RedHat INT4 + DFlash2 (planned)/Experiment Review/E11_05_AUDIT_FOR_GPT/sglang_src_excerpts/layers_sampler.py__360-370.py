# sglang 0.5.21 layers/sampler.py lines 360-370
                    selected_weight=selected_weight,
                    batch_rows=capture_rows,
                )
        else:
            backend = get_exec().kernel.sampling_backend
            if backend in ("flashinfer", "intel_xpu"):
                assert sampling_info.sampling_seed is None, (
                    f"Sampling seed is not supported for {backend} backend"
                )
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
