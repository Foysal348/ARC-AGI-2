"""Capacity measurement on CPU: Merlin's REAL get_kv_cache_configs on the klee100 K3_ONLY v2 GPU record (the same input the
E27 K100 kit's R9 uses), baseline vs each capacity patch.

Usage:  venv/Scripts/python measure_capacity.py <merlin_tree_parent> [out.json]
  <merlin_tree_parent> = the directory that CONTAINS the `vllm` package (a COPY with capacity_patch.py applied; env-gated
  patches are OFF unless the arm sets their env var, so the patched tree also reproduces the baseline).
Each arm runs in its own subprocess (env vars are read at call time, but a fresh process keeps arms independent).

Spec construction = the kit's KVPROBE (build_e27_nbdf_cpu_kit.py:484-569): MambaSpec / CircularBufferSpec / MLA / Full specs
from the GPU worker record, the K-dependent state rows scaled to K, MTP layer 48 dropped on D arms, the 5 DFlash2 SWA layers
added on PP1 (4 KV heads x 128 per TP rank, sliding_window 2048). Config = the kit's real_vc (PP2 x TP2, V2 runner, async,
max_num_batched_tokens 1024, align Mamba, prefix caching, fp8 KV, block 4096). Layout resolved with the GPU-recorded
supported layouts [BLNHC, BLHNC] (R9 rec["supported_layouts"]).
"""
import json, os, subprocess, sys, math, copy, dataclasses, re
from pathlib import Path

REC = Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out_k3_v2/q27-xhigh-e25-klee100-load-k3only/k100/k100_results.json")
CKPT_CFG = Path(r"D:/ARC AGI 2/_scratch/klee100_upload_fetch_2026-10-31/config.json")
PIN = 7933526016
LENS = (40000, 64000, 90000, 116000)
SEQS = (24, 28, 32)

GDN_UNPADDED = None  # filled from the record (unpadded_page_bytes)


def child(arm_json, tree):
    sys.path.insert(0, str(Path(__file__).with_name("stubs")))   # uvloop has no Windows wheel: a tiny stub
    sys.path.insert(0, tree)
    arm = json.loads(arm_json)
    for k, v in (arm.get("env") or {}).items():
        os.environ[k] = str(v)
    os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)   # set after VllmConfig() (its V2 validation needs Triton)
    import logging
    import torch
    # the kit's FakeL4 platform (build_e27_nbdf_cpu_kit.py:268-303): sm_89 "NVIDIA L4" before any vllm config import
    import types as _types
    for _m in ("vllm._C_stable_libtorch", "vllm._qutlass_C", "vllm._C"):   # compiled ops: never used by the planner
        sys.modules.setdefault(_m, _types.ModuleType(_m))
    import vllm.platforms as VP
    from vllm.platforms.interface import DeviceCapability
    from vllm.platforms import cuda as VPC

    class FakeL4(VPC.NonNvmlCudaPlatform):
        @classmethod
        def get_device_capability(cls, device_id=0):
            return DeviceCapability(8, 9)

        @classmethod
        def has_device_capability(cls, capability, device_id=0):
            want = tuple(capability) if isinstance(capability, tuple) else (capability // 10, capability % 10)
            return (8, 9) >= want

        @classmethod
        def get_device_name(cls, device_id=0):
            return "NVIDIA L4"

        @classmethod
        def get_device_total_memory(cls, device_id=0):
            return 23034 * 2 ** 20

        @classmethod
        def is_fully_connected(cls, physical_device_ids):
            return False
    VP._current_platform = FakeL4()
    from types import SimpleNamespace as NS
    from vllm.v1 import kv_cache_interface as KI
    from vllm.v1.core import kv_cache_utils as KU
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends import utils as AU
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as MBE

    msgs = []

    class _Cap(logging.Handler):
        def emit(self, r):
            try:
                msgs.append(r.getMessage())
            except Exception:
                pass
    logging.getLogger("vllm").addHandler(_Cap())

    rec = json.loads(REC.read_text(encoding="utf-8"))["sections"]["K3"]["k100_K3_fp8_mtp3_pin739_r0"]
    ranks = rec["workers"]["ranks"]
    LT = json.loads(CKPT_CFG.read_text(encoding="utf-8"))
    LT = (LT.get("text_config") or LT).get("layer_types")
    tdt = lambda v: getattr(torch, str(v).replace("torch.", "")) if v else None
    K = int(arm.get("k", 3))

    def k_shapes(s, k):
        s = copy.deepcopy(s)
        sh = [list(x) for x in s.get("state_shapes") or []]
        ex = str(s.get("example") or "")
        if s["type"] == "MambaSpec" and sh:
            sh[0][0] += k - 3
            if "linear_attn" in ex and len(sh) >= 3:
                sh[2][1] += k - 3
        s["state_shapes"] = sh
        return s

    def state_bytes(s):
        return sum(math.prod(sh) * tdt(dt).itemsize for sh, dt in zip(s["state_shapes"], s["state_dtypes"]))

    # the padded Mamba page the platform would set (interface.py:1040-1126): baseline = the attention page (block 4096 x
    # 512 B fp8 = 2 MiB); MERLIN_MAMBA_PAD_TO_STATE=1 = the largest Mamba state page at this K (patch P1)
    mamba_pages = [state_bytes(k_shapes(s, K)) for w in ranks for s in w["kv_cache_specs"] if s["type"] == "MambaSpec"]
    max_mamba = max(mamba_pages)
    if os.environ.get("MERLIN_MAMBA_PAD_TO_STATE") == "1":
        pad = max_mamba
    else:
        pad = 4096 * 512
    def mk(s, name=""):
        cls = getattr(KI, s["type"])
        f = {x.name for x in dataclasses.fields(cls)}
        kw = {}
        if "block_size" in f: kw["block_size"] = s["block_size"]
        if "num_kv_heads" in f: kw["num_kv_heads"] = s.get("num_kv_heads")
        if "head_size" in f: kw["head_size"] = s.get("head_size")
        if "dtype" in f and s.get("dtype"): kw["dtype"] = tdt(s["dtype"])
        if "shapes" in f: kw["shapes"] = tuple(tuple(x) for x in s.get("state_shapes") or ())
        if "dtypes" in f: kw["dtypes"] = tuple(tdt(x) for x in s.get("state_dtypes") or ())
        if "page_size_padded" in f:
            kw["page_size_padded"] = pad if s["type"] == "MambaSpec" else s.get("page_size_padded")
        if "mamba_type" in f and s.get("mamba_type"): kw["mamba_type"] = MBE[str(s["mamba_type"]).split(".")[-1]]
        if "mamba_cache_mode" in f and s["type"] == "MambaSpec": kw["mamba_cache_mode"] = "align"
        if "tp_replicated" in f and ".ple" in name: kw["tp_replicated"] = True
        if "sliding_window" in f and s.get("sliding_window"): kw["sliding_window"] = s["sliding_window"]
        sp = cls(**{k: v for k, v in kw.items() if v is not None})
        want = s.get("page_size_bytes")
        if "tokens_per_state" in f and isinstance(want, int) and sp.page_size_bytes != want and sp.page_size_bytes % want == 0:
            sp = dataclasses.replace(sp, tokens_per_state=sp.page_size_bytes // want)
        return sp

    def layer_names(stage, s, n):
        lo, hi = (0, 25) if stage == 0 else (25, 48)
        ex = str(s.get("example") or "layer")
        suf = ex.split(".", 4)[-1]
        if ".ple" in ex:
            idx = [1]
        elif "linear_attn" in ex:
            idx = [i for i in range(lo, hi) if LT[i] == "linear_attention"]
        else:
            idx = [i for i in range(lo, hi) if LT[i] != "linear_attention"]   # klee100: "qwen_sparse_attention" (kit sub :831)
        names = [f"language_model.model.layers.{i}.{suf}" for i in idx]
        if stage == 1 and len(names) + 1 == n and "linear_attn" not in ex:
            names.append(f"mtp.layers.48.{suf}")
        assert len(names) == n, (ex, len(names), n)
        return names

    idx_fp8 = bool(arm.get("idx_fp8"))
    drafter = arm.get("drafter")
    ws = []
    for w in ranks:
        d = {}
        stage = 0 if w["rank"] < 2 else 1
        for s in w["kv_cache_specs"]:
            s = k_shapes(s, K)
            if idx_fp8 and s["type"] == "MLAAttentionSpec" and "compressed_key_cache" in str(s.get("example")):
                s = dict(s, dtype="torch.float8_e4m3fn", page_size_bytes=s["page_size_bytes"] // 2)
            for nm in layer_names(stage, s, int(s.get("layers") or 1)):
                if drafter and nm.startswith("mtp."):
                    continue
                d[nm] = mk(s, nm)
        if drafter and w["rank"] >= 2:
            for i in range(5):
                d[f"model.layers.{48 + i}.self_attn.attn"] = mk(drafter)
        ws.append(d)

    def real_vc(seqs):
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        vc = VllmConfig()
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"   # served config: launch.py sets it (max_concurrent_batches = PP + 1)
        vc.model_config = NS(max_model_len=131072, original_max_model_len=131072, hf_config=NS(model_type="qwen4_exp"),
                             use_mla=False, is_hybrid=True, get_num_layers=lambda *a: 48)
        cc, pc, sc = vc.cache_config, vc.parallel_config, vc.scheduler_config
        cc.block_size, cc.enable_prefix_caching, cc.cache_dtype = 4096, True, "fp8_e4m3"
        cc.prefix_match_unit = 64
        cc.mamba_cache_mode = "align"
        pc.pipeline_parallel_size, pc.tensor_parallel_size = 2, 2
        sc.async_scheduling = True
        sc.max_num_batched_tokens = 1024
        sc.max_num_seqs = int(seqs)
        return vc

    vc = real_vc(24)
    if arm.get("layout"):
        vc.cache_config.kv_cache_layout = arm["layout"]   # counterfactual planner path (resolution honors a preset)
    specs_all = [s for w in ws for s in w.values()]
    lay = AU.resolve_kv_cache_layout(vc, [["BLNHC", "BLHNC"]] * len(ws), specs_all)
    cfgs = KU.get_kv_cache_configs(vc, ws, [PIN] * len(ws))
    if arm.get("layout"):
        def real_vc(seqs, _f=real_vc):
            v = _f(seqs)
            v.cache_config.kv_cache_layout = arm["layout"]
            return v
    assert vc.use_v2_model_runner and vc.max_concurrent_batches == 3, (vc.use_v2_model_runner, vc.max_concurrent_batches)
    out = {"arm": arm["name"], "layout": lay.name, "num_blocks": [c.num_blocks for c in cfgs],
           "max_mamba_state_page": max_mamba, "mamba_page_used": pad,
           "max_in_flight_tokens": vc.max_in_flight_tokens}
    out["stride_per_rank"] = [KU._pool_bytes_per_block(vc, c.kv_cache_groups) for c in cfgs]
    out["groups_rank"] = {}
    for r in (0, 2):
        out["groups_rank"][r] = [[type(g.kv_cache_spec).__name__ if not hasattr(g.kv_cache_spec, "kv_cache_specs") else
                                  sorted({type(x).__name__ for x in g.kv_cache_spec.kv_cache_specs.values()})[0],
                                  len(g.layer_names), g.kv_cache_spec.block_size, g.kv_cache_spec.page_size_bytes,
                                  getattr(g, "slots_per_block", 1)] for g in cfgs[r].kv_cache_groups]
    sched = KU.generate_scheduler_kv_cache_config(cfgs)
    parts, F, shared_slots = [], 0, []
    sw_block, sw_window = None, None
    mamba_groups = 0
    for i, g in enumerate(sched.kv_cache_groups):
        sp = g.kv_cache_spec
        slots = getattr(g, "slots_per_block", 1)
        row = {"g": i, "spec": type(sp).__name__, "block": sp.block_size, "layers": len(g.layer_names)}
        if slots > 1:
            row["slots_per_block"] = slots
            shared_slots.append(slots)
        else:
            req = math.ceil(sp.max_memory_usage_bytes(vc) / sp.page_size_bytes)
            row["required_at_131072"] = req
            if isinstance(sp, (KI.FullAttentionSpec, KI.MLAAttentionSpec)) and not isinstance(sp, KI.SlidingWindowSpec):
                row["kind"] = "context"
            else:
                F += req
                row["kind"] = "fixed"
                if isinstance(sp, KI.SlidingWindowSpec):
                    sw_block, sw_window = sp.block_size, sp.sliding_window
                    row["sw_admission_blocks"] = sp.max_admission_blocks_per_request(vc.max_in_flight_tokens, 131072)
                if isinstance(sp, KI.MambaSpec):
                    mamba_groups += 1
        parts.append(row)
    out["sched_groups"] = parts
    out["kv_groups"] = len(sched.kv_cache_groups)
    out["F_admission"] = F
    # steady decode real-held fixed blocks: align Mamba = running state + the retained prompt checkpoint per group
    # (single_type_kv_cache_manager.py:1673-1702, 1947-1978); SW = cdiv(window - 1 + in-flight decode tokens, block) + 1
    # with in-flight decode tokens = max_concurrent_batches x (1 + K) (remove_skipped_blocks on the processed basis)
    sw_steady = 0
    if sw_block:
        sw_steady = math.ceil((sw_window - 1 + vc.max_concurrent_batches * (1 + K)) / sw_block) + 1
    out["F_steady"] = 2 * mamba_groups + sw_steady
    out["F_steady_P4_no_retain"] = 1 * mamba_groups + sw_steady
    out["mamba_groups"] = mamba_groups
    out["sw_steady_blocks"] = sw_steady
    mcs = {}
    for s in SEQS:
        mcs[s] = KU.get_max_concurrency_for_kv_cache_config(real_vc(s), sched)
    out["max_concurrency_131072"] = mcs
    usable = min(out["num_blocks"]) - 1
    res = {}
    for basis in ("F_admission", "F_steady", "F_steady_P4_no_retain"):
        f = out[basis]
        res[basis] = {}
        for s in SEQS:
            shared = sum(math.ceil(s / x) for x in shared_slots)
            res[basis][s] = {L: min(s, (usable - shared) // (math.ceil(L / 4096) + f)) for L in LENS}
    out["usable_blocks"] = usable
    out["residents"] = res
    try:   # GPT report section 7 ledger fields (capacity_ledger.py)
        sys.path.insert(0, str(Path(__file__).parent))
        import capacity_ledger as CL
        out["ledger_static"] = CL.static_ledger(vc, cfgs, ranks=[w["rank"] for w in ranks])
    except BaseException as e:
        out["ledger_static"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    out["notes"] = [m for m in msgs if "Side-cache" in m or "Padding" in m or "MERLIN" in m][:6]
    if arm.get("drafter") and arm.get("name", "").startswith("D4s baseline"):
        try:
            out["swmask_test"] = swmask_test(KU, sched, vc, K, tree)
        except BaseException as e:
            import traceback
            out["swmask_test"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1800:]}
    if arm.get("pressure"):
        try:
            out["pressure"] = {pol: pressure(KU, sched, vc, K, pol, arm["pressure"]) for pol in ("fcfs", "headroom")}
        except BaseException as e:
            import traceback
            out["pressure"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1800:]}
    try:
        out["sim"] = simulate(KU, sched, vc, K, arm)
    except BaseException as e:
        import traceback
        out["sim"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1800:]}
    print("@JSON " + json.dumps(out, default=str))


def pressure(KU, sched, vc, K, policy, cfg):
    """#62752-style CPU reproducer on Merlin's REAL KVCacheManager (port of the issue's fixture to Merlin APIs):
    S DFlash streams (prompt lengths staggered), every step each running stream decodes (1 + K) slots with all drafts
    accepted (fastest boundary crossings: 4096-token attention block + align-Mamba state block per group), and at
    `burst_step` B streams receive a tool return of `burst_tokens` tokens that is prefilled in 1024-token chunks
    (one chunk per stream per step, chunk budget 1024 per step like --max-num-batched-tokens 1024).
    policy fcfs: decodes first, then burst chunks; an allocation failure preempts the youngest running stream
    (vLLM's recompute preemption: free, count its tokens as recompute, re-admit when the full sequence fits).
    policy headroom: same, but a burst chunk runs only if, after it, the free pool still covers the next-step
    boundary demand of every decoding stream (streams whose next step crosses a 4096 boundary need
    1 attention + 1 Mamba block per Mamba group + 1 SW block)."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    sbs, hbs = KU.resolve_kv_cache_block_sizes(sched, vc)
    hfn = get_hash_fn_by_name("sha256_cbor")
    KU.init_none_hash(hfn)
    hasher = KU.get_request_block_hasher(hbs, hfn)
    m = KVCacheManager(kv_cache_config=sched, max_model_len=131072, scheduler_block_size=sbs, hash_block_size=hbs,
                       max_in_flight_tokens=vc.max_in_flight_tokens, enable_caching=True)
    nm = sum(1 for g in sched.kv_cache_groups if type(g.kv_cache_spec).__name__ == "MambaSpec"
             or "Mamba" in str(type(getattr(g.kv_cache_spec, "first_spec", None))))
    S, steps = cfg["streams"], cfg["steps"]
    lens = [cfg["base_len"] + i * cfg["len_step"] for i in range(S)]
    reqs, prefill_left, waiting = {}, {}, []
    st = {"preemptions": 0, "recompute_tokens": 0, "admit_fail": 0, "min_free": 10 ** 9, "burst_done_step": None,
          "stall_steps": 0, "decode_tokens": 0}

    def admit(rid, toks):
        r = Request(request_id=rid, prompt_token_ids=toks, sampling_params=SamplingParams(max_tokens=10 ** 6),
                    pooling_params=None, block_hasher=hasher)
        done, first = 0, True
        while done < len(toks):
            c = min(1024, len(toks) - done)
            if m.allocate_slots(r, c, full_sequence_must_fit=first) is None:
                m.free(r)
                return None
            first = False
            done += c
            r.num_computed_tokens = done
        return r

    for i, L in enumerate(lens):
        toks = [5 + (i * 104729 + j * 7) % 240000 for j in range(L)]
        r = admit(f"s{i}", toks)
        if r is None:
            st["admit_fail"] += 1
            waiting.append((f"s{i}", toks))
        else:
            reqs[r.request_id] = r
    order = list(reqs)
    for step in range(steps):
        if step == cfg["burst_step"]:
            for rid in order[: cfg["bursts"]]:
                if rid in reqs:
                    reqs[rid].append_output_token_ids([11] * cfg["burst_tokens"])
                    prefill_left[rid] = cfg["burst_tokens"]
        # decode every running stream that is not prefilling a burst
        for rid in list(order):
            r = reqs.get(rid)
            if r is None or prefill_left.get(rid):
                continue
            r.append_output_token_ids([7] * (1 + K))
            while m.allocate_slots(r, 1 + K, num_lookahead_tokens=K) is None:
                victim = order[-1]
                vr = reqs.pop(victim)
                order.remove(victim)
                st["preemptions"] += 1
                st["recompute_tokens"] += vr.num_tokens
                m.free(vr)
                prefill_left.pop(victim, None)
                waiting.append((victim, list(vr.all_token_ids)))
                if victim == rid:
                    break
            if rid in reqs:
                r.num_computed_tokens = r.num_tokens - 0
                st["decode_tokens"] += 1 + K
        # burst prefill chunks
        budget = 1024
        for rid in list(order):
            left = prefill_left.get(rid)
            if not left or rid not in reqs or budget <= 0:
                continue
            r = reqs[rid]
            c = min(1024, left, budget)
            if policy == "headroom":
                need = 0
                for o in order:
                    ro = reqs.get(o)
                    if ro is None or prefill_left.get(o):
                        continue
                    if (ro.num_tokens % 4096) + 1 + K > 4096:
                        need += 1 + nm + 1
                if m.block_pool.get_num_free_blocks() - (c // 4096 + 2) < need:
                    st["stall_steps"] += 1
                    continue
            if m.allocate_slots(r, c) is None:
                st["stall_steps"] += 1
                continue
            r.num_computed_tokens += c
            prefill_left[rid] = left - c
            budget -= c
            if prefill_left[rid] == 0:
                prefill_left.pop(rid)
        if cfg["burst_step"] < step and not prefill_left and st["burst_done_step"] is None:
            st["burst_done_step"] = step
        # re-admit waiting (recompute) when the full sequence fits
        if waiting and step % 10 == 0:
            rid, toks = waiting[0]
            r = admit(rid, toks)
            if r is not None:
                waiting.pop(0)
                reqs[rid] = r
                order.append(rid)
        st["min_free"] = min(st["min_free"], m.block_pool.get_num_free_blocks())
    st.update(running_end=len(reqs), waiting_end=len(waiting), pool=m.block_pool.num_gpu_blocks, mamba_groups=nm,
              cfg=cfg)
    return st


def swmask_test(KU, sched, vc, K, tree):
    """CPU test of the #60767 port on Merlin's REAL scheduler-side managers: one long DFlash stream (prefill 12k in
    1024 chunks, decode 3000 tokens in (1 + K) steps). The worker's persistent table is emulated exactly as V2 does it
    (append-only: every allocate_slots result is appended, block_table.py:114-133). After every step:
      stale  = row positions whose scheduler entry is the null block (freed) but the worker row still holds an id;
      masked = the patch's sw_mask_reference(row, first_pos = num_computed_tokens, reach = window - 1 + extra);
    PASS = every stale id is masked to 0, and no in-window live id is masked. Then a second stream is allocated so the
    freed ids are REUSED, and we count how many of stream 1's stale row ids now belong to stream 2 (the #60560
    hazard: foreign / NaN pages behind a stale id)."""
    import importlib.util
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1 import kv_cache_interface as KI
    src = (Path(tree) / "vllm/v1/worker/gpu/block_table.py").read_text(encoding="utf-8")
    i0 = src.index("def sw_mask_reference(")
    i1 = src.index("@triton.jit", i0)
    ns = {}
    exec(src[i0:i1].replace("  # CAPACITY-PORT SWMASK", ""), ns)
    ref = ns["sw_mask_reference"]
    sbs, hbs = KU.resolve_kv_cache_block_sizes(sched, vc)
    hfn = get_hash_fn_by_name("sha256_cbor")
    KU.init_none_hash(hfn)
    hasher = KU.get_request_block_hasher(hbs, hfn)
    m = KVCacheManager(kv_cache_config=sched, max_model_len=131072, scheduler_block_size=sbs, hash_block_size=hbs,
                       max_in_flight_tokens=vc.max_in_flight_tokens, enable_caching=True)
    gi = next(i for i, g in enumerate(sched.kv_cache_groups) if isinstance(g.kv_cache_spec, KI.SlidingWindowSpec))
    sp = sched.kv_cache_groups[gi].kv_cache_spec
    reach = sp.sliding_window - 1 + sp.extra_retained_tokens
    mgr = m.coordinator.single_type_managers[gi]
    res = {"group": gi, "block": sp.block_size, "reach": reach, "steps": 0, "max_stale": 0, "unmasked_stale": 0,
           "masked_live_in_window": 0}

    def run(rid, P, D, seed):
        r = Request(request_id=rid, prompt_token_ids=[seed + (j * 13) % 200000 for j in range(P)],
                    sampling_params=SamplingParams(max_tokens=10 ** 6), pooling_params=None, block_hasher=hasher)
        row = []
        done, first = 0, True

        def step(n, look=0):
            b = m.allocate_slots(r, n, num_lookahead_tokens=look, full_sequence_must_fit=False)
            assert b is not None
            row.extend(b.get_block_ids()[gi])
        while done < P:
            c = min(1024, P - done)
            first_pos = done
            step(c)
            check(r, row, first_pos)
            done += c
            r.num_computed_tokens = done
        for _ in range(D // (1 + K)):
            r.append_output_token_ids([9] * (1 + K))
            first_pos = r.num_computed_tokens
            step(1 + K, K)
            check(r, row, first_pos)
            r.num_computed_tokens = r.num_tokens
        return r, row

    def check(r, row, first_pos):
        sched_row = mgr.req_to_blocks[r.request_id]
        n = len(row)
        masked = ref([row], [n], [first_pos], reach, sp.block_size)[0]
        stale = [j for j in range(min(n, len(sched_row))) if sched_row[j].is_null and row[j] != 0]
        res["steps"] += 1
        res["max_stale"] = max(res["max_stale"], len(stale))
        res["unmasked_stale"] += sum(1 for j in stale if masked[j] != 0)
        lo = max(first_pos - reach, 0)
        res["masked_live_in_window"] += sum(1 for j in range(n) if masked[j] == 0 and row[j] != 0
                                            and (j + 1) * sp.block_size > lo)
    r1, row1 = run("a", 12000, 3000, 101)
    stale_ids = {row1[j] for j, b in enumerate(mgr.req_to_blocks["a"]) if b.is_null and row1[j] != 0}
    r2, row2 = run("b", 12000, 0, 70001)
    res["stale_ids_stream1"] = len(stale_ids)
    res["stale_ids_now_owned_by_stream2"] = len(stale_ids & set(row2))
    res["pass"] = res["unmasked_stale"] == 0 and res["masked_live_in_window"] == 0 and res["max_stale"] > 0
    return res


def simulate(KU, sched, vc, K, arm):
    """Merlin's REAL KVCacheManager on the scheduler KV config: admit N fresh requests (unique tokens, no prefix hits) with
    a P-token prompt each, prefill them in 1024-token chunks through allocate_slots(full_sequence_must_fit=True on the
    first chunk, like the scheduler's admission), then decode D steps of (1 + K) slots each (1 verified + K lookahead),
    every step settled (num_in_flight_tokens 0). Records the real blocks held per group per request (non-null table
    entries + retained Mamba checkpoints) and the pool free count: the measured F."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    sbs, hbs = KU.resolve_kv_cache_block_sizes(sched, vc)
    hfn = get_hash_fn_by_name("sha256_cbor")
    KU.init_none_hash(hfn)
    hasher = KU.get_request_block_hasher(hbs, hfn)
    res = {"scheduler_block": sbs, "hash_block": hbs, "runs": []}
    for P, N, D in ((40000, 4, 200), (64000, 4, 200), (90000, 2, 200)):
        m = KVCacheManager(kv_cache_config=sched, max_model_len=131072, scheduler_block_size=sbs, hash_block_size=hbs,
                           max_in_flight_tokens=vc.max_in_flight_tokens, enable_caching=True)
        pool_free0 = m.block_pool.get_num_free_blocks()
        reqs = []
        base = 1000
        for n in range(N):
            toks = [base + (n * 7919 + i * 31) % 200000 for i in range(P)]
            r = Request(request_id=f"r{n}", prompt_token_ids=toks, sampling_params=SamplingParams(max_tokens=10000),
                        pooling_params=None, block_hasher=hasher)
            done = 0
            first = True
            while done < P:
                c = min(1024, P - done)
                b = m.allocate_slots(r, c, full_sequence_must_fit=first)
                assert b is not None, ("prefill alloc failed", n, done)
                first = False
                done += c
                r.num_computed_tokens = done
            reqs.append(r)
        peak_free_after_prefill = m.block_pool.get_num_free_blocks()
        for step in range(D):
            for r in reqs:
                r.append_output_token_ids(7)
                b = m.allocate_slots(r, 1, num_lookahead_tokens=K)
                assert b is not None, ("decode alloc failed", step)
                r.num_computed_tokens = r.num_tokens
        held = {}
        for gi, mgr in enumerate(m.coordinator.single_type_managers):
            per = []
            for r in reqs:
                bl = mgr.req_to_blocks.get(r.request_id, [])
                nn = sum(1 for x in bl if not x.is_null)
                ret = getattr(mgr, "_retained_checkpoints", {}).get(r.request_id)
                per.append(nn + (1 if ret is not None else 0))
            held[f"g{gi}:{type(mgr).__name__}"] = per
        ctx = -(-(P + D) // 4096)
        fixed = [sum(v[i] for k, v in held.items() if "FullAttention" not in k and "MLA" not in k
                     and "CircularBuffer" not in k) for i in range(N)]
        used = pool_free0 - m.block_pool.get_num_free_blocks()
        try:
            import capacity_ledger as CL
            live = CL.live_ledger(m)
        except BaseException as e:
            live = {"error": f"{type(e).__name__}: {e}"[:300]}
        res["runs"].append({"P": P, "N": N, "decode_steps": D, "held_per_group": held, "ledger_live": live, "context_blocks": ctx,
                            "fixed_per_request": fixed, "pool_used": used, "pool_used_per_request": used / N,
                            "free_after_prefill_used": pool_free0 - peak_free_after_prefill})
    return res


PRESS = {"streams": 24, "steps": 1200, "base_len": 24000, "len_step": 1700, "burst_step": 300, "bursts": 4, "burst_tokens": 6000}
ARMS = [
    {"name": "Q (MTP K3) baseline", "k": 3},
    {"name": "Q + P1 mamba pad->state", "k": 3, "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"}},
    {"name": "Q + P1 + P2 mamba 2 groups", "k": 3, "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1", "MERLIN_MAMBA_GROUPS": "2"}},
    {"name": "Q + idxfp8 (known lever, record)", "k": 3, "idx_fp8": True},
    {"name": "Q + idxfp8 + P1", "k": 3, "idx_fp8": True, "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"}},
    {"name": "D4s baseline (SW1024 bf16)", "k": 4, "drafter": "bf16_1024", "pressure": PRESS},
    {"name": "D3s baseline (SW1024 bf16)", "k": 3, "drafter": "bf16_1024"},
    {"name": "D4s + P1 mamba pad->state", "k": 4, "drafter": "bf16_1024", "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"},
     "pressure": PRESS},
    {"name": "D4s + P1 + P2 mamba 2 groups", "k": 4, "drafter": "bf16_1024",
     "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1", "MERLIN_MAMBA_GROUPS": "2"}},
    {"name": "D4s + P1 + P2 mamba 4 groups", "k": 4, "drafter": "bf16_1024",
     "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1", "MERLIN_MAMBA_GROUPS": "4"}},
    {"name": "D4s P3 SW2048 fp8 (TRITON)", "k": 4, "drafter": "fp8_2048"},
    {"name": "D4s P3 SW2048 fp8 + P1", "k": 4, "drafter": "fp8_2048", "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"},
     "pressure": PRESS},
    {"name": "D3 SW1024 fp8", "k": 3, "drafter": "fp8_1024"},
    {"name": "D3 SW2048 bf16", "k": 3, "drafter": "bf16_2048"},
    {"name": "D3 SW2048 fp8", "k": 3, "drafter": "fp8_2048"},
    {"name": "D3 SW2048 fp8 + P1", "k": 3, "drafter": "fp8_2048", "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"}},
    {"name": "D3 SW1024 bf16 + P1", "k": 3, "drafter": "bf16_1024", "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"}},
    {"name": "Q layer-outer LBHNC (uniform-page planner, #58207 path)", "k": 3, "layout": "LBHNC"},
    {"name": "D3s layer-outer LBHNC (uniform-page planner, #58207 path)", "k": 3, "drafter": "bf16_1024", "layout": "LBHNC"},
    {"name": "D4s SW2048 bf16 (widens, record)", "k": 4, "drafter": "bf16_2048"},
    {"name": "D4s SW512 bf16 + P1", "k": 4, "drafter": "bf16_512", "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"}},
    {"name": "D4s + P1 + idxfp8", "k": 4, "drafter": "bf16_1024", "idx_fp8": True, "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"}},
    {"name": "D4s + P1 + P3 fp8 2048 + idxfp8", "k": 4, "drafter": "fp8_2048", "idx_fp8": True,
     "env": {"MERLIN_MAMBA_PAD_TO_STATE": "1"}},
]
DRAFTERS = {
    "bf16_1024": {"type": "SlidingWindowSpec", "block_size": 1024, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.bfloat16", "sliding_window": 2048},
    "bf16_2048": {"type": "SlidingWindowSpec", "block_size": 2048, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.bfloat16", "sliding_window": 2048},
    "bf16_512": {"type": "SlidingWindowSpec", "block_size": 512, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.bfloat16", "sliding_window": 2048},
    "fp8_1024": {"type": "SlidingWindowSpec", "block_size": 1024, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.uint8", "sliding_window": 2048},
    "fp8_2048": {"type": "SlidingWindowSpec", "block_size": 2048, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.uint8", "sliding_window": 2048},
}


def main():
    tree = sys.argv[1]
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("capacity_results.json")
    results = []
    for arm in ARMS:
        a = dict(arm)
        if a.get("drafter"):
            a["drafter"] = DRAFTERS[a["drafter"]]
        env = {k: v for k, v in os.environ.items() if not k.startswith("MERLIN_MAMBA_")}
        p = subprocess.run([sys.executable, __file__, "--child", json.dumps(a), tree], capture_output=True, text=True,
                           env=env, timeout=900)
        line = next((l for l in p.stdout.splitlines() if l.startswith("@JSON ")), None)
        r = json.loads(line[6:]) if line else {"arm": arm["name"], "error": (p.stderr or p.stdout)[-2500:]}
        r["drafter_key"] = arm.get("drafter")
        results.append(r)
        print(arm["name"], "->", r.get("num_blocks"), "F", r.get("F_admission"), r.get("F_steady"), r.get("error", "")[:300] if "error" in r else "", flush=True)
    outp.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("wrote", outp)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
