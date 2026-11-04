"""E25 NB-S1 library (coordinator / user 2026-11-01: DFlash2 ONLY). Levers on top of B = NBDF-K100's D3s (klee100 3bpw-MTP
target, Merlin v3.0.0 ad42e37, 4 x L4, seqs 24, pin 7,933,526,016): the NBDF seam apply / verify / revert, the DFlash2 drafter
mount, DFLASH_DRAFT_SW_BLOCK=1024, BF16 drafter KV on FLASH_ATTN, no draft head (S4), the D pin ladder (EMBED_HOST first).
Pure logic, stdlib only; nbdf_lib (N) and nb4t_lib (T) are read through the module objects handed in, never edited.

B runs under nbdf_lib's base key "Q" (the key every NBDF comparison, A/A, TF envelope and k100_rel is built on), named
nbs1_b. Arms (each = B + ONE change, run in this order; no cross-arm skip):
  G1   VLLM_GDN_DECODE_FASTPATH=1
  G2   VLLM_GDN_SHARED_ALIGN_INDEX=1
  EH   DFLASH_DRAFT_EMBED_HOST=1 at the same pin (the drafter embedding on the host: ~606 MiB more KV per PP1 rank)
  F8   FP8 drafter KV (fp8_e4m3) on TRITON_ATTN + DFLASH_DRAFT_SW_BLOCK=2048 (F8_SOURCE_CHECK)
  DYN  dynamic K: num_speculative_tokens 4 + num_speculative_tokens_per_batch_size [[1, 8, 4], [9, 24, 3]] (DYN_SOURCE_CHECK)
Real failures (that arm only): the GDN switch absent from a worker / present on another arm; EH without the host embedding;
F8 without TRITON_ATTN or the 2048 drafter block on both PP1 ranks; DYN with cudagraph_mode overridden to PIECEWISE or
the dynamic schedule disabled."""
import copy
import json

LIB_VERSION = "nbs1-lib 2026-11-01 r3 (DFlash2 only)"
ARM_KEYS = ("Q", "G1", "G2", "EH", "F8", "DYN")      # "Q" = B (nbs1_b)
GDN_ENV = {"G1": "VLLM_GDN_DECODE_FASTPATH", "G2": "VLLM_GDN_SHARED_ALIGN_INDEX"}
ENV_EMBED_HOST, ENV_SW_BLOCK = "DFLASH_DRAFT_EMBED_HOST", "DFLASH_DRAFT_SW_BLOCK"
S1_ENV_KEYS = (GDN_ENV["G1"], GDN_ENV["G2"])
N_WORKERS, PP1_TP = 4, 2
DYN_SCHEDULE = [[1, 8, 4], [9, 24, 3]]
DYN_K_MAX = 4
F8_SPEC = {"kv_cache_dtype": "fp8_e4m3", "attention_backend": "TRITON_ATTN"}
F8_SW_BLOCK = 2048
PORTS = {"Q": 8211, "G1": 8221, "G2": 8231, "EH": 8241, "F8": 8251, "DYN": 8261}
TITLES = {"Q": "B = NBDF-K100 D3s (DFlash2 K3, seam, SW-1024, BF16 drafter KV, FLASH_ATTN, no draft head), A/A",
          "G1": "B + VLLM_GDN_DECODE_FASTPATH=1", "G2": "B + VLLM_GDN_SHARED_ALIGN_INDEX=1",
          "EH": "B + DFLASH_DRAFT_EMBED_HOST=1 (same pin: the freed drafter embedding goes to KV)",
          "F8": "B + FP8 drafter KV on TRITON_ATTN + drafter SW block 2048",
          "DYN": "B + dynamic K (K4 at <= 8 running, K3 at 9-24)"}
NOT_BUILT = {"V32": "N/A_DFLASH", "V64": "N/A_DFLASH", "V64I8": "N/A_DFLASH", "I4": "N/A_DFLASH", "REF": "REMOVED (user: DFlash2 only)"}

F8_SOURCE_CHECK = (
    "Merlin ad42e37 v1/attention/backends/triton_attn.py:302-345,381: TRITON_ATTN lists fp8 / fp8_e4m3 KV, block % 16, "
    "sliding window, non-causal (the DFlash2 drafter's block attention) and every compute capability (sm_89 included); "
    "FLASH_ATTN (FA2 on sm_89) has no FP8 KV path there (its fp8 handling is the sm90 FA4 path). The drafter backend / KV "
    "dtype are the dflash spec's attention_backend / kv_cache_dtype (NBDF's D4b used the same pair). dflash_seam's "
    "Attention.get_kv_cache_spec patch takes DFLASH_DRAFT_SW_BLOCK=<n> for any positive divisor of the 4096 block: 2048 "
    "is one. -> configurable; F8 stays in the plan. Risk (recorded, #54690 class): Triton FP8 sliding-window decode at "
    "fp8 scale 1.0 (no drafter scales) -> caught by the smoke / oracle / TF gates of this arm only.")
DYN_SOURCE_CHECK = (
    "Merlin ad42e37 config/speculative.py:493 num_speculative_tokens_per_batch_size = [(start, end, K)] inclusive ranges, "
    "clamped to num_speculative_tokens (v1/spec_decode/dynamic/utils.py); the scheduler (v1/core/sched/scheduler.py:348) "
    "picks K per batch size. config/vllm.py:1068-1085 overrides FULL cudagraphs to PIECEWISE unless the V2 model runner "
    "is used (NBDF runs the V2 GPU runner: v1/worker/gpu); v1/worker/gpu/cudagraph_utils.py:283-300 then captures the "
    "decode graphs of every tier. -> num_speculative_tokens 4 + [[1, 8, 4], [9, 24, 3]], captures = every b x 5 for b <= 8 "
    "and every b x 4 for b 9-24. Per-batch K reaches the drafter: v1/worker/gpu/model_runner.py:2786-2803 passes "
    "num_speculative_tokens=<this batch's K> to speculator.propose() and sets draft_tokens[:, K:] = -1 when K < the max; "
    "DFlashSpeculator.propose (spec_decode/dflash/speculator.py:343-369, the DFlash2 base) accepts the argument and drafts "
    "its full fixed block (num_query_per_req = 1 + num_speculative_steps, :62), so the drafter cost stays K4 at every batch "
    "size and only the target's verify length shrinks at 9-24 running (not a 'cannot': DYN stays, the read-back makes a "
    "fallback a real failure). Real-failure read-back: an 'Overriding cudagraph_mode' or 'Disabling "
    "num_speculative_tokens_per_batch_size' log line.")
DYN_FAIL_RX = ("Overriding cudagraph_mode", "Disabling num_speculative_tokens_per_batch_size")


def dyn_captures():
    """b x (1 + 4) for b = 1..8 and b x (1 + 3) for b = 9..24 (every decode micro-batch of both tiers in an exact graph)."""
    return sorted({b * (DYN_K_MAX + 1) for b in range(1, 9)} | {b * 4 for b in range(9, 25)})


def _set_arg(extra, flag, value):
    """The LAST occurrence of `flag` in argv gets `value` (the effective value: vLLM's last-wins parsing)."""
    out = list(extra)
    idx = [i for i, x in enumerate(out) if x == flag]
    if not idx or idx[-1] + 1 >= len(out):
        raise ValueError(f"{flag} not in the base argv")
    out[idx[-1] + 1] = value
    return out


def s1_arms(N, ad, aq):
    """ad = NBDF-K100's D3s arm, aq = its Q arm (B takes Q's full-minimum / cap: the A/A items). Returns {key: arm}."""
    def expect_of(a):
        e = {w: a["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED}
        for k in S1_ENV_KEYS:
            e[k] = a["env"].get(k)
        return e

    b = copy.deepcopy(ad)
    b.update(key="Q", name="nbs1_b", port=PORTS["Q"], title=TITLES["Q"], long_aa=True, lever=False, parent=None,
             est_min=int(aq["est_min"]), cap_min=int(aq["cap_min"]), s1_change=None, needs_seam=True, optional=False,
             short_only=False, iso=False)
    b["expect"] = expect_of(b)
    b["expect_cfg"]["draft_head"] = None
    arms = {"Q": b}
    for k in ARM_KEYS[1:]:
        a = copy.deepcopy(b)
        env_add, spec_add = {}, {}
        if k in GDN_ENV:
            env_add[GDN_ENV[k]] = "1"
        elif k == "EH":
            env_add[ENV_EMBED_HOST] = "1"
        elif k == "F8":
            env_add[ENV_SW_BLOCK] = str(F8_SW_BLOCK)
            spec_add = dict(F8_SPEC)
        elif k == "DYN":
            spec_add = {"num_speculative_tokens": DYN_K_MAX, "num_speculative_tokens_per_batch_size": DYN_SCHEDULE}
        a["env"].update(env_add)
        if spec_add:
            a["spec"] = dict(a["spec"], **spec_add)
            a["extra"] = _set_arg(a["extra"], "--speculative-config", N.spec_json(a["spec"]))
            for f in ("kv_cache_dtype", "attention_backend", "num_speculative_tokens"):
                if f in spec_add:
                    a["expect_cfg"][f] = spec_add[f]
        if k == "DYN":
            a["k"] = DYN_K_MAX
            a["captures"] = dyn_captures()
            a["extra"] = _set_arg(a["extra"], "--compilation-config", N.compilation_json(a["captures"]))
            a["expect_cfg"]["captures"] = a["captures"]
            a["coverage"] = N.coverage(a["captures"], N.SEQS, DYN_K_MAX)
            ok_r, cap_r = N.qsa_ring(DYN_K_MAX)
            a["ring_needed"], a["ring_capacity_expected"] = not ok_r, cap_r
        a["expect"] = expect_of(a)
        a.update(key=k, name=f"nbs1_{k.lower()}", port=PORTS[k], title=TITLES[k], long_aa=False, lever=True,
                 est_min=int(ad["est_min"]), cap_min=int(ad["cap_min"]), s1_change={"env": env_add, "spec": spec_add})
        arms[k] = a
    return arms


def install(N, arms):
    """Rebind nbdf_lib's arm-set constants to the NB-S1 arms (what restrict_arms does for K100). Every arm is a DFlash2
    SW-block arm (B included: its key is Q). Idempotent. Returns the new ORDER."""
    N.ORDER, N.D_ARMS = tuple(arms), tuple(k for k in arms if k != "Q")
    N.OPTIONAL = N.SHORT_ARMS = N.CONV_TORCH_ARMS = N.SHORT_EXIT_ARMS = N.ADMISSION_PROBE_ARMS = ()
    N.SW_BLOCK_ARMS = tuple(arms)
    N.KV_RUN_OF = {k: ("K4_swblock1024" if k == "DYN" else "K3_fp8_sw2048" if k == "F8" else "K3_swblock1024") for k in arms}
    N.K_OF.update({k: a["k"] for k, a in arms.items()})
    return N.ORDER


def check_arms(N, T, arms):
    """Raises unless: B first, the listed order, unique names / ports, every arm DFlash2 with no draft head (S4), no
    never-do item, and every arm = B's env + exactly its planned env change, B's argv except the planned spec /
    compilation values."""
    if tuple(arms) != ARM_KEYS:
        raise ValueError(f"NB-S1 arms {tuple(arms)} != {ARM_KEYS}")
    b = arms["Q"]
    if len({a["name"] for a in arms.values()}) != len(arms) or len({a["port"] for a in arms.values()}) != len(arms):
        raise ValueError("NB-S1: duplicate arm name / port")
    for k, a in arms.items():
        if a.get("method") != "dflash" or a["spec"].get("method") != "dflash" or "draft_lm_head_quant" in a["spec"] or a["env"].get("DRAFT_HEAD"):
            raise ValueError(f"NB-S1 {k}: not a DFlash2 arm without a draft head (S4)")
        bad = T.never_do_violations(a["env"], a["extra"])
        if bad:
            raise ValueError(f"NB-S1 {k}: never-do {bad}")
        if k == "Q":
            continue
        diff = {e: a["env"].get(e) for e in set(a["env"]) | set(b["env"]) if a["env"].get(e) != b["env"].get(e)}
        if diff != a["s1_change"]["env"]:
            raise ValueError(f"NB-S1 {k}: env diff {diff} != the planned change {a['s1_change']['env']}")
        want_extra = list(b["extra"])
        if a["s1_change"]["spec"]:
            want_extra = _set_arg(want_extra, "--speculative-config", N.spec_json(dict(b["spec"], **a["s1_change"]["spec"])))
        if k == "DYN":
            want_extra = _set_arg(want_extra, "--compilation-config", N.compilation_json(dyn_captures()))
        if a["extra"] != want_extra:
            raise ValueError(f"NB-S1 {k}: argv differs from B beyond the planned change")
    return True


def base_gate(tree_ok, drafter, kv_gate, seam_ok, seam_why):
    """B is a DFlash2 arm: the per-D-arm prerequisites of nbdf_main, applied to B. None = launch; else (status, why)."""
    if not tree_ok:
        return None                                     # nbdf_main already recorded Q NOT_INTEGRATED
    if not (drafter or {}).get("ok"):
        return "NOT_INTEGRATED", f"{(drafter or {}).get('status') or 'DRAFTER'}: {(drafter or {}).get('why')}"
    if not (kv_gate or {}).get("passed"):
        return "SKIPPED_KV_KIT_GATE", "the NB-S1 real CPU kit gate did not pass: " + str((kv_gate or {}).get("why"))
    if not seam_ok:
        return "NOT_INTEGRATED", str(seam_why)
    return None


# ------------------------------------------------------------------ read-backs (real failures)
def env_values(env_rb, key):
    """[value per worker] of `key` from T.parse_env_readback's rows (None = absent)."""
    return [((r or {}).get("values") or {}).get(key) for r in ((env_rb or {}).get("workers") or [])]


def readback(arm, log_text, env_rb, drb):
    """The arm's own change, read back from its server log, its workers' environment and nbdf_lib.dflash_readback (drb).
    {"ok", "real_failure", "why", ...}. B: no GDN switch in any worker."""
    k, why = arm["key"], []
    chk = (drb or {}).get("checks") or {}
    rec = {"arm": k}
    for g, key in GDN_ENV.items():
        vals = env_values(env_rb, key)
        rec[key] = vals
        if k == g:
            if len(vals) != N_WORKERS or any(v != "1" for v in vals):
                why.append(f"{key} in the workers {vals}, expected '1' on {N_WORKERS}")
        elif any(v is not None for v in vals):
            why.append(f"{key} set in a worker of {k}: {vals}")
    if k == "EH":
        rec["embed_host_as_planned"] = chk.get("embed_host_as_planned")
        if chk.get("embed_host_as_planned") is not True:
            why.append(f"DFLASH_DRAFT_EMBED_HOST=1 not applied on the drafter ranks: {chk.get('embed_host_as_planned')}")
    if k == "F8":
        rec.update(draft_backend=chk.get("draft_backend"), draft_sw_block_applied=chk.get("draft_sw_block_applied"))
        if chk.get("draft_backend") is not True:
            why.append(f"the drafter's backend is not TRITON_ATTN on both PP1 ranks: {chk.get('draft_backend')}")
        if chk.get("draft_sw_block_applied") is not True:
            why.append(f"DFLASH_DRAFT_SW_BLOCK={F8_SW_BLOCK} not applied: {chk.get('draft_sw_block_applied')}")
    hits = [l.strip()[:200] for l in (log_text or "").splitlines() if any(x in l for x in DYN_FAIL_RX)]
    rec["dyn_fallback_lines"] = hits[:4]
    if hits:
        why.append(f"dynamic speculative decoding fell back (cudagraph FULL lost / schedule disabled): {hits[:2]}")
    rec["why"] = "; ".join(why) or None
    rec["real_failure"] = bool(why)
    rec["ok"] = not why
    return rec


def profile_plan(arm, cfg):
    """[(band, level)]: the profiled window on B only (cfg profile = [band, level])."""
    return [tuple(cfg["profile"])] if arm["key"] == "Q" else []


# ------------------------------------------------------------------ records
def arm_records(rec, key):
    """The per-arm read-back summary: AL, per-position acceptance, tick, tok/s per window, preemptions, pool blocks,
    residents, the drafter KV / embed facts."""
    rec = rec or {}
    st = rec.get("steady") or {}
    acc = rec.get("oracle_acceptance") or {}
    drb = rec.get("dflash_readback") or {}
    return {"arm": "B" if key == "Q" else key, "status": rec.get("status"), "al": acc.get("al"), "per_pos": acc.get("per_pos"),
            "windows": {k: {f: (w or {}).get(f) for f in ("status", "tok_s", "tick_ms", "preemptions", "running_eq_level", "level")}
                        for k, w in st.items() if isinstance(w, dict)},
            "pool_blocks": (rec.get("workers") or {}).get("num_gpu_blocks"), "residents": rec.get("k100_residents"),
            "pin": rec.get("pin"), "tries": rec.get("tries"),
            "released_mib": (drb.get("rec") or {}).get("released_mib"),
            "s1_readback": {k: (rec.get("s1_readback") or {}).get(k) for k in ("ok", "why")},
            "tf_gate": (rec.get("tf_gate") or {}).get("status"), "oracle_vs": rec.get("oracle_vs")}


def summary_json(records):
    return json.dumps(records, default=str)
