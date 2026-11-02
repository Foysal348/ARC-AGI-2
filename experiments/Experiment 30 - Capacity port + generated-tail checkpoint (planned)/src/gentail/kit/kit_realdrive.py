# E30 gentail kit, real-class drive (runs in the Merlin venv, after the PLATFORM preamble the kit prepends).
# argv: mode (on|off), fixture json (K100 K3 GPU worker kv specs), pin bytes, klee100 ckpt dir or "None".
# 1 imports the real patched modules; 2 builds the real KV cache config (the NBDF K100 kit's R9 recipe:
# get_kv_cache_configs on the GPU-recorded per-worker specs) + generate_scheduler_kv_cache_config; 3 a real
# KVCacheManager (coordinator + BlockPool + managers) configured as the Scheduler does; 4 a synthetic 3-turn chain of
# real Request objects (no forward), the scheduler hook calls in the patched scheduler's order, the real (patched)
# KVCacheManager.free; 5 per turn the real get_computed_blocks hit vs the prediction (gate 2 on CPU).
import json, os, sys, random, traceback
from types import SimpleNamespace as NS

out = {"platform": PLAT, "mode": sys.argv[1]}
MODE = sys.argv[1]
fx = json.loads(open(sys.argv[2]).read())
PIN = int(sys.argv[3])
ckpt = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "None" else None
for k in [x for x in os.environ if x.startswith("GENTAIL_")]:
    os.environ.pop(k, None)
if MODE == "on":
    os.environ["GENTAIL_CKPT"] = "1"
    os.environ["GENTAIL_CKPT_STATS_EVERY"] = "1"
BS, UNIT, K_SPEC = 4096, 64, 3
try:
    import dataclasses, re, copy
    import torch
    from vllm.v1 import kv_cache_interface as KI
    from vllm.v1.core import kv_cache_utils as KU
    from vllm.config import VllmConfig
    import vllm.v1.core.sched.scheduler as SCH
    import vllm.v1.core.kv_cache_manager as KVM
    import vllm.v1.core.kv_cache_coordinator as KVC
    import vllm.v1.core.block_pool as BP
    import vllm.v1.core.single_type_kv_cache_manager as STM
    import vllm.v1.core.gentail_ckpt as GT
    from vllm.v1.request import Request, RequestStatus
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    out["imports"] = {m.__name__: m.__file__ for m in (SCH, KVM, KVC, BP, STM, GT)}
    out["hooks_present"] = {"Scheduler._gentail": hasattr(SCH.Scheduler, "_gentail"),
                            "KVCacheManager._gentail": hasattr(KVM.KVCacheManager, "_gentail"),
                            "maybe_install_gentail": hasattr(GT, "maybe_install_gentail")}

    # ---- 2. the real KV cache config: the NBDF K100 kit's R9 recipe (KVPROBE mk / k_shapes / layer_names / worker_specs)
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as MBE
    tdt = lambda v: getattr(torch, str(v).replace("torch.", "")) if v else None

    def mk(s, name=""):
        sp = _mk(s, name)
        f = {x.name for x in dataclasses.fields(type(sp))}
        want = s.get("page_size_bytes")
        have = getattr(sp, "page_size_bytes", None)
        if "tokens_per_state" in f and isinstance(want, int) and isinstance(have, int) and have != want and want and have % want == 0:
            sp = dataclasses.replace(sp, tokens_per_state=have // want)
        return sp

    def _mk(s, name=""):
        cls = getattr(KI, s["type"])
        f = {x.name for x in dataclasses.fields(cls)}
        kw = {}
        if "block_size" in f: kw["block_size"] = s["block_size"]
        if "num_kv_heads" in f: kw["num_kv_heads"] = s.get("num_kv_heads")
        if "head_size" in f: kw["head_size"] = s.get("head_size")
        if "dtype" in f and s.get("dtype"): kw["dtype"] = tdt(s["dtype"])
        if "shapes" in f: kw["shapes"] = tuple(tuple(x) for x in s.get("state_shapes") or ())
        if "dtypes" in f: kw["dtypes"] = tuple(tdt(x) for x in s.get("state_dtypes") or ())
        if "page_size_padded" in f: kw["page_size_padded"] = s.get("page_size_padded")
        if "mamba_type" in f and s.get("mamba_type"): kw["mamba_type"] = MBE[str(s["mamba_type"]).split(".")[-1]]
        if "mamba_cache_mode" in f and s["type"] == "MambaSpec": kw["mamba_cache_mode"] = "align"
        if "tp_replicated" in f and ".ple" in name: kw["tp_replicated"] = True
        if "sliding_window" in f and s.get("sliding_window"): kw["sliding_window"] = s["sliding_window"]
        return cls(**{k: v for k, v in kw.items() if v is not None})

    LT = None
    try:
        _c = json.load(open(f"{ckpt}/config.json"))
        LT = (_c.get("text_config") or _c).get("layer_types")
    except BaseException:
        LT = None
    out["layer_types_from_ckpt"] = bool(LT)

    def layer_names(stage, s, n):
        lo, hi = (0, 25) if stage == 0 else (25, 48)
        ex = str(s.get("example") or "layer")
        suf = ex.split(".", 4)[-1] if ex.startswith("language_model.model.layers.") else None
        if LT and suf is not None:
            if ".ple" in ex:
                idx = [1]
            elif "linear_attn" in ex:
                idx = [i for i in range(lo, hi) if LT[i] == "linear_attention"]
            else:
                idx = [i for i in range(lo, hi) if LT[i] == "full_attention"]
            names = [f"language_model.model.layers.{i}.{suf}" for i in idx]
            if stage == 1 and len(names) + 1 == n and "linear_attn" not in ex:
                names.append(f"mtp.layers.48.{suf}")
            if len(names) == n:
                return names
        return [re.sub(r"layers\.\d+", f"layers.{1000 + 100 * stage + i}", ex) for i in range(n)]

    ws = []
    for w in fx["q_workers"]["ranks"]:
        d = {}
        stage = 0 if w["rank"] < 2 else 1
        for s in w["kv_cache_specs"]:
            for nm in layer_names(stage, s, int(s.get("layers") or 1)):
                d[nm] = mk(s, nm)
        ws.append(d)
    MC = {}
    try:
        from vllm.config import ModelConfig
        MC["mc"] = ModelConfig(model=ckpt, tokenizer=ckpt, dtype="bfloat16", max_model_len=131072) if ckpt else None
    except BaseException as e:
        MC["error"] = f"{type(e).__name__}: {e}"[:300]
    out["model_config"] = {k: v for k, v in MC.items() if k != "mc"}
    vc = VllmConfig()
    if MC.get("mc") is not None:
        vc.model_config = MC["mc"]
    else:
        vc.model_config = NS(max_model_len=131072, original_max_model_len=131072, hf_config=NS(), use_mla=False, is_hybrid=True,
                             get_num_layers=lambda *a: 48)
    vc.cache_config.block_size = BS
    vc.cache_config.enable_prefix_caching = True
    vc.cache_config.cache_dtype = "fp8_e4m3"
    # the engine core's own layout resolution, exactly as the NBDF K100 real kit R9 (v1/engine/core.py:388-392)
    import vllm.v1.attention.backends.utils as AU
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
    from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
    _sup = [l.name for l in AU.get_supported_kv_cache_layouts([Qwen4ExpQSAFlashAttentionBackend, QSAStateBackend])]
    out["layout"] = AU.resolve_kv_cache_layout(vc, [_sup] * len(ws), [s for w in ws for s in w.values()]).name
    cfgs = KU.get_kv_cache_configs(vc, ws, [PIN] * len(ws))
    sched_cfg = KU.generate_scheduler_kv_cache_config(cfgs)
    out["kv_config"] = {"num_blocks": [c.num_blocks for c in cfgs], "scheduler_num_blocks": sched_cfg.num_blocks,
                        "groups": [{"spec": type(g.kv_cache_spec).__name__, "block_size": getattr(g.kv_cache_spec, "block_size", None),
                                    "layers": len(g.layer_names)} for g in sched_cfg.kv_cache_groups],
                        "retention_interval": getattr(sched_cfg, "prefix_cache_retention_interval", "n/a")}

    # ---- 3. the real KVCacheManager, configured as Scheduler.__init__ does (scheduler.py:366-394)
    kvm = KVM.KVCacheManager(kv_cache_config=sched_cfg, max_model_len=131072, scheduler_block_size=BS, hash_block_size=UNIT,
                             enable_caching=True, use_eagle=True, num_prefill_lookahead=K_SPEC)
    n_cfg = SCH.configure_recoverssm_pp_checkpoints(kvm.coordinator.single_type_managers)
    fake_sched = NS(kv_cache_manager=kvm, cache_config=NS(use_recoverssm=True), parallel_config=NS(pipeline_parallel_size=2),
                    scheduler_config=NS(max_num_seqs=21), num_spec_tokens=K_SPEC, defer_block_free=True)
    ctl = GT.maybe_install_gentail(fake_sched)
    out["managers"] = [(type(m).__name__, m.kv_cache_group_id, m.block_size) for m in kvm.coordinator.single_type_managers]
    out["coordinator"] = {"cls": type(kvm.coordinator).__name__, "reprefillable": kvm.coordinator.num_reprefillable_tokens,
                          "partial_hash_hits": getattr(kvm.coordinator, "enable_partial_hash_hits", None),
                          "recoverssm_pp_guard_groups": n_cfg,
                          "publish_decode_checkpoints": [getattr(m, "publish_decode_checkpoints", None)
                                                         for m in kvm.coordinator.single_type_managers if isinstance(m, STM.MambaManager)]}
    out["ctl"] = None if ctl is None else {"mamba": [m.kv_cache_group_id for m in ctl.mamba], "attn": [m.kv_cache_group_id for m in ctl.attn],
                                           "repref": ctl.repref, "bs": ctl.bs, "unit": ctl.unit, "max_grace_pins": kvm.MAX_GRACE_PINS}
    assert (ctl is not None) == (MODE == "on") and (kvm._gentail is ctl)
    PUB = []
    if ctl is not None:
        _orig_pub = ctl._publish

        def _pub(pend):
            n0 = ctl.stats["published"]
            _orig_pub(pend)
            if ctl.stats["published"] > n0:
                PUB.append({"rid": pend.request_id, "position": pend.position, "block_ids": list(pend.block_ids)})
        ctl._publish = _pub

    # ---- 4. the chain: real Requests, the patched scheduler's hook order, the real patched free
    hfn = get_hash_fn_by_name("sha256")
    KU.init_none_hash(hfn)
    hasher = KU.get_request_block_hasher(UNIT, hfn)
    rng = random.Random(7)
    SEQ = {"s": 0}
    free0 = kvm.block_pool.get_num_free_blocks()

    def step_schedule(req, qlen):
        kvm.new_step_starts()
        SEQ["s"] += 1
        s = SEQ["s"]
        if ctl is not None:                        # update_after_schedule: hook, then the counts advance
            ctl.on_schedule(req, qlen, s)
        req.num_computed_tokens += qlen
        req.num_in_flight_tokens += qlen
        req.last_sched_seq = s
        return s

    def step_output(req, s, qlen, n_sampled, decode):
        if ctl is not None:                        # update_from_output: processed_step_seq += 1; drain
            ctl.drain(s)
        req.num_in_flight_tokens -= qlen
        if decode:
            req.num_computed_tokens -= qlen - n_sampled
        if ctl is not None:
            ctl.on_step_output(req, qlen, n_sampled, s, False)
        req.append_output_token_ids([rng.randrange(1000, 200000) for _ in range(n_sampled)])

    def turn(rid, prompt, gen):
        req = Request(rid, list(prompt), SamplingParams(max_tokens=gen + 16, temperature=0.0), None, block_hasher=hasher)
        blocks, nhit, *_ = kvm.get_computed_blocks(req)
        hit_mamba = {}
        for m in kvm.coordinator.single_type_managers:
            if isinstance(m, STM.MambaManager):
                g = blocks.blocks[m.kv_cache_group_id]
                last = next((b for b in reversed(g) if not b.is_null), None)
                hit_mamba[m.kv_cache_group_id] = None if last is None else last.block_id
        req.status = RequestStatus.RUNNING
        qlen = len(prompt) - nhit
        new = kvm.allocate_slots(req, qlen, num_new_computed_tokens=nhit, new_computed_blocks=blocks)
        assert new is not None, f"{rid}: prefill allocation refused"
        req.num_computed_tokens = nhit             # Scheduler.schedule: the hit, before update_after_schedule
        s = step_schedule(req, qlen)
        step_output(req, s, qlen, 1, False)
        while req.num_output_tokens < gen:
            room = gen - req.num_output_tokens
            n_sampled = min(1 + rng.randint(0, K_SPEC), room)
            new = kvm.allocate_slots(req, 1 + K_SPEC)
            assert new is not None, f"{rid}: decode allocation refused at {req.num_computed_tokens}"
            s = step_schedule(req, 1 + K_SPEC)
            step_output(req, s, 1 + K_SPEC, n_sampled, True)
        req.status = RequestStatus.FINISHED_STOPPED
        kvm.free(req)                              # Scheduler._free_request_blocks: last_sched_seq <= processed -> free()
        return req, nhit, hit_mamba

    turns = []
    prompt = [rng.randrange(1000, 200000) for _ in range(5000)]
    prev = None
    for i, gen in enumerate((7000, 6000, 300)):
        if prev is not None:
            prompt = list(prev.all_token_ids) + [rng.randrange(1000, 200000) for _ in range(300)]
        req, nhit, hm = turn(f"t{i}", prompt, gen)
        p_prev = prev.num_prompt_tokens if prev is not None else None
        committed_prev = (prev.num_computed_tokens - prev.num_in_flight_tokens) if prev is not None else None
        pred = None
        if prev is not None:
            valid = min(committed_prev, prev.num_tokens) - kvm.coordinator.num_reprefillable_tokens
            limit = valid // UNIT * UNIT - UNIT
            pred_b = min(limit, (len(prompt) - 1) // UNIT * UNIT) // BS * BS
            pred = pred_b if pred_b > p_prev else None
        turns.append({"rid": req.request_id, "prompt": len(prompt), "gen": gen, "hit": nhit, "pred_tail": pred,
                      "prev_prompt": p_prev, "hit_mamba_block_ids": hm})
        prev = req
    out["turns"] = turns
    out["published"] = PUB
    out["stats"] = dict(ctl.stats) if ctl is not None else None
    # release the grace pins: every block back
    pins = list(kvm._grace_pins)
    for rid in pins:
        kvm.release_grace_pin(rid)
    out["free_blocks"] = {"start": free0, "end": kvm.block_pool.get_num_free_blocks(), "grace_pins_released": len(pins),
                          "queue": len(ctl._queue) if ctl else 0, "pending": len(ctl._pending) if ctl else 0}
    # ---- 5. checks
    chk = {}
    chk["imports_ok"] = all(out["hooks_present"].values())
    chk["guard_kept"] = n_cfg > 0 and all(v is False for v in out["coordinator"]["publish_decode_checkpoints"])
    chk["no_leak"] = out["free_blocks"]["end"] == out["free_blocks"]["start"] and out["free_blocks"]["queue"] == 0
    if MODE == "on":
        pos = {p["position"] for p in PUB}
        chk["published_two"] = [p["position"] for p in PUB] == [8192, 16384]
        chk["tail_hit_turn1"] = turns[1]["hit"] == turns[1]["pred_tail"] == 8192
        chk["tail_hit_turn2"] = turns[2]["hit"] == turns[2]["pred_tail"] == 16384
        chk["hit_uses_published_mamba_blocks"] = all(
            set(t["hit_mamba_block_ids"].values()) <= set(PUB[i]["block_ids"]) and None not in t["hit_mamba_block_ids"].values()
            for i, t in enumerate(turns[1:]))
        chk["adopted_two"] = (ctl.stats.get("adopted") == 2 and ctl.stats.get("adopted_tokens") == 8192 + 16384)
        chk["no_mismatch"] = not ctl.stats.get("plan_mismatch") and not ctl.stats.get("seq_mismatch") and not ctl.stats.get("no_block")
    else:
        chk["off_no_controller"] = ctl is None and kvm._gentail is None
        chk["off_hit_below_tail"] = all(t["hit"] < 8192 * (i) or i == 0 for i, t in enumerate(turns))
    out["checks"] = chk
    out["ok"] = all(chk.values())
except BaseException as e:
    out.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
print("@JSON " + json.dumps(out, default=str))
