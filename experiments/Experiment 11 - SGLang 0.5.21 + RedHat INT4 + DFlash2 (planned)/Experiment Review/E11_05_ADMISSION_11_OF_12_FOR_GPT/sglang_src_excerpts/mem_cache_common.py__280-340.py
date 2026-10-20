# sglang 0.5.21 (tag v0.5.21) mem_cache/common.py lines 280-340
        req.kv.retraction_backup = None
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.kv.retraction_backup is None:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    unified_cache.discard_kv_cache_backup(req.kv.retraction_backup)
    req.kv.retraction_backup = None


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    """Give the request's kv row back; with ``is_insert`` the tree first keeps
    what it can key."""
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    # A mamba-capable cache may alloc mamba state before alloc KV cache
    if not req.kv.holds_kv:
        assert tree_cache.supports_mamba(), (
            "Only a mamba-capable tree cache allows freeing before alloc"
        )
        # TODO (csy, hanming): clean up this early allocation logic
        if req.kv.holds_mamba:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1)
            )
            req.kv.mamba_pool_idx = None
        return
    if tree_cache.claim_kv_row(req):
        # A streaming session detached the kv record to keep the row.
        assert not req.kv.holds_kv
        return

    owned_kv_len = req.owned_kv_len()
    is_insert = is_insert and not req.skip_radix_cache_insert
    if is_insert:
        tree_cache.insert_req(req, up_to=owned_kv_len)
    # The protected prefix is not this req's to free.
    tree_cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, owned_kv_len)])
    tree_cache.unpin(req)
    _release_overallocated_kv_indices(
        req, owned_kv_len, req.kv.kv_allocated_len, tree_cache
    )
    tree_cache.on_release(req, inserted=is_insert)

    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert req.kv.holds_mamba, (
            "mamba state is freed while the tree cache does not manage mamba states"
        )
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv.mark_kv_released()


def _release_overallocated_kv_indices(
    req: Req, start_p: int, end_p: int, tree_cache: BasePrefixCache
