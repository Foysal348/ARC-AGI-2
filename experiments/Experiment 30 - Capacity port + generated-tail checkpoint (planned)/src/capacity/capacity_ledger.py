"""Per-rank / per-group KV capacity ledger (GPT capacity report section 7 fields) for Merlin (vLLM 0.32 fork).

Two entry points:
  static_ledger(vllm_config, kv_cache_configs)  -> list[dict], one per worker rank: num_blocks, uniform_page_bytes (the
      pool stride get_kv_cache_config_from_groups divides by), num_kv_groups, group_specs (type, block_size, page_bytes,
      unpadded_page_bytes, layers, max_admission_blocks). Works on CPU (measure_capacity.py) and on GPU from the worker
      record (any place that has the KVCacheConfig list, e.g. the kit's R9 or a worker-extension RPC).
  live_ledger(scheduler_or_kv_cache_manager)     -> dict: per group live_blocks (non-null table entries of running
      requests), retained_checkpoint_blocks (align-Mamba prefix holds), reclaimable_blocks (free but still hashed =
      evictable prefix cache, attributed to the group of its hash), plus free_blocks, running_requests, preemptions
      (sum of request.num_preemptions over running + waiting) and recompute_prefill_tokens (tokens of preempted
      requests that will be re-prefilled). Call it inside the engine-core process.

GPU use: ledger_hook_patch.py (separate seam, default off) adds one env-gated call at the top of Scheduler.schedule:
  MERLIN_CAP_LEDGER_EVERY=<N>  -> every N scheduler steps one log line "CAPACITY-LEDGER {json}" in the server log
which the notebooks parse with parse_ledger_lines(log_text). Unset = no call, nothing logged.
"""
import json
import math


def _inner_specs(spec):
    inner = getattr(spec, "kv_cache_specs", None)
    return list(inner.values()) if isinstance(inner, dict) else [spec]


def _admission_blocks(spec, vllm_config):
    try:
        from vllm.v1.kv_cache_interface import SlidingWindowSpec, MambaSpec
    except Exception:  # pragma: no cover
        return None
    first = _inner_specs(spec)[0]
    if isinstance(first, SlidingWindowSpec):
        return first.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len)
    if isinstance(first, MambaSpec):
        return math.ceil(first.max_memory_usage_bytes(vllm_config) / first.page_size_bytes)
    return None  # context-proportional (full attention / MLA) or slot-packed ring


def static_ledger(vllm_config, kv_cache_configs, ranks=None):
    from vllm.v1.core import kv_cache_utils as KU
    out = []
    for i, c in enumerate(kv_cache_configs):
        groups = []
        for g in c.kv_cache_groups:
            sp = g.kv_cache_spec
            inner = [s for n, s in getattr(sp, "kv_cache_specs", {}).items() if n in g.layer_names] or _inner_specs(sp)
            unpadded = sum(getattr(s, "real_page_size_bytes", None) or getattr(s, "unpadded_page_size_bytes", None)
                           or s.page_size_bytes for s in inner)
            groups.append({"type": sorted({type(s).__name__ for s in inner}), "block_size": sp.block_size,
                           "layers": len(g.layer_names), "page_bytes": sum(s.page_size_bytes for s in inner),
                           "unpadded_page_bytes": unpadded, "slots_per_block": getattr(g, "slots_per_block", 1),
                           "max_admission_blocks": _admission_blocks(sp, vllm_config)})
        out.append({"rank": ranks[i] if ranks else i, "num_blocks": c.num_blocks,
                    "uniform_page_bytes": KU._pool_bytes_per_block(vllm_config, c.kv_cache_groups),
                    "num_kv_groups": len(c.kv_cache_groups), "group_specs": groups})
    return out


def live_ledger(obj):
    """obj = a Scheduler (has .kv_cache_manager, .running, .waiting) or a KVCacheManager."""
    from vllm.v1.core.kv_cache_utils import get_group_id
    sched = obj if hasattr(obj, "kv_cache_manager") else None
    m = sched.kv_cache_manager if sched is not None else obj
    pool = m.block_pool
    mgrs = m.coordinator.single_type_managers
    reclaim = [0] * len(mgrs)
    for b in pool.free_block_queue.get_all_free_blocks():
        h = b.block_hash
        if h is not None and not b.is_null:
            try:
                reclaim[get_group_id(h)] += 1
            except Exception:
                pass
    groups = []
    for gi, mg in enumerate(mgrs):
        live = sum(sum(1 for x in bl if not x.is_null) for bl in mg.req_to_blocks.values())
        retained = len(getattr(mg, "_retained_checkpoints", {}) or {})
        groups.append({"group": gi, "manager": type(mg).__name__, "block_size": mg.block_size,
                       "live_blocks": live, "retained_checkpoint_blocks": retained, "reclaimable_blocks": reclaim[gi],
                       "max_admission_blocks": getattr(mg, "max_admission_blocks_per_request", None)})
    rec = {"num_blocks": pool.num_gpu_blocks, "free_blocks": pool.get_num_free_blocks(), "groups": groups}
    if sched is not None:
        running = list(sched.running)
        waiting = list(getattr(sched, "waiting", []) or [])
        rec["running_requests"] = len(running)
        rec["waiting_requests"] = len(waiting)
        rec["preemptions"] = sum(getattr(r, "num_preemptions", 0) for r in running + waiting)
        rec["recompute_prefill_tokens"] = sum(r.num_tokens for r in waiting if getattr(r, "num_preemptions", 0) > 0)
        rec["step"] = getattr(sched, "current_step", None)
    return rec


def ledger_tick(scheduler):
    """Called by the ledger hook seam at the top of Scheduler.schedule (env-gated there)."""
    import os
    every = int(os.environ.get("MERLIN_CAP_LEDGER_EVERY", "0") or 0)
    if every <= 0 or (getattr(scheduler, "current_step", 0) % every):
        return
    try:
        from vllm.logger import init_logger
        init_logger("vllm.capacity_ledger").info("CAPACITY-LEDGER %s", json.dumps(live_ledger(scheduler)))
    except Exception as e:  # never break scheduling
        try:
            from vllm.logger import init_logger
            init_logger("vllm.capacity_ledger").warning("CAPACITY-LEDGER error %s", e)
        except Exception:
            pass


def parse_ledger_lines(log_text):
    """Notebook side: every CAPACITY-LEDGER json record from a server log."""
    out = []
    for line in log_text.splitlines():
        i = line.find("CAPACITY-LEDGER {")
        if i >= 0:
            try:
                out.append(json.loads(line[i + len("CAPACITY-LEDGER "):]))
            except ValueError:
                pass
    return out
