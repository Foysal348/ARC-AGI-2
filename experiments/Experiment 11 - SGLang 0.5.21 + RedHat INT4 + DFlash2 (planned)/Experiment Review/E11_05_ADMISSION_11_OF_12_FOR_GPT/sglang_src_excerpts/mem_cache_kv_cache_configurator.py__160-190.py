# sglang 0.5.21 (tag v0.5.21) mem_cache/kv_cache_configurator.py lines 160-190
    if not is_multimodal:
        return 0.0
    reserved_mb = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
    if mm_feature_transport in ("cuda_ipc", "cuda_vmm"):
        reserved_mb += envs.SGLANG_MM_FEATURE_CACHE_MB.get()
    if reserved_mb > 0:
        logger.info(
            "Reserving %.2f GB of the KV budget for post-sizing multimodal "
            "allocations (feature-transport pools + embedding cache).",
            reserved_mb / 1024,
        )
    return reserved_mb / 1024


# base ratio of mamba pool size to max_running_requests. Under
# SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK the decode-time skip frees one resident slot
# per running request, so the base drops by 1 (overlap 5->4, lazy 4->3). no_buffer
# stays at effective 3 either way: its binding limit is the prefill->decode peak,
# which the decode-time drop does not shrink.
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_BUFFER = 1


def _pp_local_per_request_bytes(
    total_bytes: int,
    layer_ids: list[int],
    start_layer: int,
