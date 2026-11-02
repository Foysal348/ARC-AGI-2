"""E25 NB-S2: the code blocks the NB-S2 builder (build_e25_nbs2.py) inserts into NBDF-K100's CELL 7 body (text constants only;
nothing here runs at import). Each block is placed by an exact-once anchor in build_e25_nbs2.py."""

# ---- after NBDF-K100's checked arm plan (N.check_arm_plan(ARMS) on Q / D3s / D4s): B = its D3s, the levers on B
S2_SETUP = r'''
# ==================================================================== E25 NB-S2: B = NBDF-K100's D3s (the plan checked above), capacity levers on B
NBDF_ARMS = ARMS                            # NBDF-K100's own plan (Q, D3s, D4s), checked by nbdf_lib.check_arm_plan
NBS2_LIB_SHA = "__S2LIBSHA__"
NBS2_LIB_SRC = _nbs2_ds("nbs2_lib.py", NBS2_LIB_SHA)   # the payload dataset (sha-pinned)
if hashlib.sha256(NBS2_LIB_SRC.encode("utf-8")).hexdigest() != NBS2_LIB_SHA:
    raise RuntimeError("EMBED_SHA_MISMATCH: nbs2_lib")
(TMP / "nbs2_lib.py").write_text(NBS2_LIB_SRC, encoding="utf-8")
S2 = importlib.import_module("nbs2_lib")
S2_CFG = json.loads(r"""__S2CFG__""")
N.restrict_arms(("Q", "D3s"))                # N.SW_BLOCK_ARMS = (D3s,): every NB-S2 arm has parent D3s (launch_check, KV summary)
ARMS = S2.arm_plan(NBDF_ARMS["D3s"], seam_env_k4=SEAM_ENVS.get("D4s"), pin=int(K100_CFG["q_pin"]), seqs=int(K100_CFG["seqs"]))
ARMS = {k: ARMS[k] for k in S2_CFG["arms"]}
S2.check_arm_plan(ARMS, NBDF_ARMS["D3s"])    # B == D3s; each lever = B + exactly its change (raises: no boot)
NBS2 = R.setdefault("nbs2", {})
NBS2.update(lib=S2.LIB_VERSION, lib_sha256=NBS2_LIB_SHA, cfg=S2_CFG, dropped=S2.DROPPED,
            question="which capacity lever (indexer KV fp8, 28 / 32 sequences, long-prefill threshold, "
                     "shared-prefix checkpoint) beats the DFlash2 base B (NBDF-K100 D3s) at 8k-116k with TF inside B's A/A?",
            seam_env_d3s_vs_d4s_equal=SEAM_ENVS.get("D3s") == SEAM_ENVS.get("D4s"),
            arms={k: {x: a.get(x) for x in ("name", "port", "change", "spec", "captures", "streams", "windows_fixed", "est_min",
                                             "cap_min", "title", "s2", "env", "extra")} for k, a in ARMS.items()})
_nbs2_on = globals().get("NBS2_ON")          # CELL 1 arm switches (E25 NB-S2): absent -> every arm on, a missing key -> that arm off
NBS2_SWITCH = S2.switch_map(_nbs2_on, tuple(ARMS))
NBS2["switch"] = S2.switch_record(_nbs2_on, tuple(ARMS))
print("E25 NB-S2 arm switches:", NBS2_SWITCH, flush=True)
'''

# ---- after the K100 prompt set: 20k prompts for 32 sequences (variants 0-23 unchanged: B's 20k@24 = NBDF-K100's)
S2_PROMPTS = r'''
if len(PROMPTS["20k"]) < 32:                # E25 NB-S2: S32 at 20k needs 32 distinct prompts (variants 24-31 added)
    PROMPTS["20k"] = list(PROMPTS["20k"]) + [prompt(BANDS["20k"], v) for v in range(len(PROMPTS["20k"]), 32)]
'''

# ---- the probes, the read-backs, the main sequence and the verdict (before _gpu_util)
S2_MAIN = r'''
# ==================================================================== E25 NB-S2: ledger, probes, read-backs, main, verdict
class S2Ledger:
    """The KV ledger of one window: /metrics every 2 s in a daemon thread (perf_counter time, the clock of steady()'s t0 /
    t1): KV usage, running / waiting, waiting by reason, preemptions, prompt tokens by source (recompute)."""

    def __init__(self, port, every=2.0):
        self.port, self.every, self.samples, self.stop_ev, self.th = port, every, [], threading.Event(), None

    def _one(self):
        try:
            txt = L.U.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=15).read().decode("utf-8", "replace")
            self.samples.append((time.perf_counter(), S2.parse_metrics(txt)))
        except Exception:
            pass

    def _loop(self):
        while not self.stop_ev.wait(self.every):
            self._one()

    def __enter__(self):
        if not FAKE:
            self._one()
            self.th = threading.Thread(target=self._loop, daemon=True, name=f"nbs2_ledger_{self.port}")
            self.th.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()
        if self.th is not None:
            self.th.join(10)
        if not FAKE:
            self._one()
        return False


def s2_window_ledger(o, r, band, n):
    """The window's KV ledger with THIS boot's pool and F (records only)."""
    smp = r.pop("_s2_samples", None) or []
    res = o.get("k100_residents") or {}
    f_ = ((res.get("40k") or {}).get("F"))
    blk = ((res.get("40k") or {}).get("block")) or N.BLOCK_TOKENS
    return {"window": S2.ledger_summary(smp, r.get("t0"), r.get("t1"), res.get("pool"), n, BANDS.get(band), f_, int(blk),
                                        r.get("gen_tokens") or 0),
            "with_admission": S2.ledger_summary(smp, None, None, res.get("pool"), n, BANDS.get(band), f_, int(blk), r.get("gen_tokens") or 0)}


def sibling_probe(port, o):
    """SPC activation read-back (recorded): after a cache reset, three siblings A / B / C = P + a different suffix (P =
    S2_CFG sibling prefix_tokens, not a block multiple); C's prefix-hit tokens show whether the shared-prefix junction was
    checkpointed (S2.sibling_label)."""
    c = S2_CFG["sibling"]
    block = int(o.get("engine_block") or N.BLOCK_TOKENS)
    p0 = prompt(int(c["prefix_tokens"]), 80)
    out = {"prefix_tokens": len(p0), "block": block, "reset": reset_prefix(port), "hits": {}, "rows": {}}
    for nm, v in (("A", 81), ("B", 82), ("C", 83)):
        ids = list(p0) + list(prompt(int(c["suffix_tokens"]), v))
        c0 = _prefix_counters(port)
        r = _one(port, ids, int(c["gen"]))
        d = _prefix_delta(c0, _prefix_counters(port))
        out["hits"][nm] = d.get("hits")
        out["rows"][nm] = {"prompt": len(ids), "queries": d.get("queries"), "tokens": len(r.get("tokens") or []), "error": r.get("error")}
    out["label"] = S2.sibling_label(out, block)
    return out


def stall_probe(port, arm):
    """The decode stall during prefills (LP's metric; B is the control): `decoders` 2k streams decode, a baseline of
    `baseline_sec`, then `inject` cold 40k prompts arrive; every decoding stream's token count is polled every poll_ms; the
    gaps between token increases before vs during the injected prefills (S2.stall_stats: p50 / p95 / max)."""
    c = S2_CFG["stall"]
    inj_ids = [prompt(int(c["inject_tokens"]), 90 + j) for j in range(int(c["inject"]))]
    dec = [L.Stream(port, PROMPTS["2k"][i % len(PROMPTS["2k"])], 4096, temperature=TEMP, seed=1300 + i, ignore_eos=True)
           for i in range(int(c["decoders"]))]
    inj = []
    series = {i: [] for i in range(len(dec))}
    dt = float(c["poll_ms"]) / 1000.0

    def poll(until, stop_when=None):
        while time.perf_counter() < until:
            now = time.perf_counter()
            for i, g in enumerate(dec):
                series[i].append((now, len(g.tokens)))
            if stop_when and stop_when():
                return
            time.sleep(dt)
    try:
        for g in dec:
            g.start()
        t_ = time.perf_counter()
        while time.perf_counter() - t_ < 180 and sum(g.t_first is not None for g in dec) < len(dec):
            time.sleep(0.2)
        decoding = sum(g.t_first is not None for g in dec)
        poll(time.perf_counter() + float(c["baseline_sec"]))
        t_inj = time.perf_counter()
        g0 = sum(len(g.tokens) for g in dec)
        inj = [L.Stream(port, ids, 8, temperature=0.0, ignore_eos=True) for ids in inj_ids]
        for g in inj:
            g.start()
        poll(t_inj + 300.0, lambda: all(g.t_first is not None or not g.is_alive() for g in inj))
        t_done = time.perf_counter()
        g1 = sum(len(g.tokens) for g in dec)
        poll(time.perf_counter() + 2.0)
    finally:
        for g in dec + inj:
            g.stop()
        for g in dec + inj:
            g.join(10)
        L.wait_idle(port, 60)
    return {"decoders": len(dec), "decoding_at_start": decoding, "inject": len(inj_ids), "inject_tokens": int(c["inject_tokens"]),
            "inject_ttft_s": [round(g.t_first - g.t_start, 2) if g.t_first and g.t_start else None for g in inj],
            "inject_errors": sorted({g.err for g in inj if g.err})[:3], "decode_errors": sorted({g.err for g in dec if g.err})[:3],
            "injection_sec": round(t_done - t_inj, 2), "decode_tokens_during_injection": g1 - g0,
            "decode_tok_s_during_injection": round((g1 - g0) / max(1e-3, t_done - t_inj), 1),
            "stats": S2.stall_stats(series, t_inj, t_done), "poll_ms": c["poll_ms"]}


def s2_readback_failure(o, arm, t):
    """NB-S2 real failures of THIS arm (a silent path fallback = the arm would measure the wrong thing): IDX's indexer K
    cache not fp8 on a rank; DYN's graphs not FULL (the dynamic-SD PIECEWISE override). None = no failure (a missing
    read-back is recorded only)."""
    if FAKE:
        return None
    if arm["key"] == "IDX":
        o["idx_readback"] = S2.idx_readback(o.get("workers"))
        if o["idx_readback"].get("engaged") is False:
            return "IDX_NOT_ENGAGED: " + str(o["idx_readback"].get("why"))[:300]
    if arm["key"] == "DYN":
        o["dyn_graphs"] = S2.dyn_graph_readback(t, arm.get("captures"))
        if o["dyn_graphs"].get("status") == "FAIL":
            return "DYN_GRAPH_FALLBACK: " + str(o["dyn_graphs"].get("why"))[:300]
    return None


def s2_records(o, arm, t):
    """Record-only facts of one arm: K_eff per window, the dynamic-K tiers (DYN), the KV ledgers, the pool / residents,
    the long-prefill / shared-prefix flags in the logged config."""
    st_ = o.get("steady") or {}
    rec = {"k_eff": {k: S2.k_eff(w) for k, w in st_.items() if isinstance(w, dict) and not k.startswith("profile")},
           "kv_ledger": {k: w.get("kv_ledger") for k, w in st_.items() if isinstance(w, dict) and w.get("kv_ledger")},
           "residents": o.get("k100_residents"), "pool": (o.get("k100_residents") or {}).get("pool"),
           "engine_block": o.get("engine_block"),
           "log_flags": {f: (f in t) for f in ("long_prefill_token_threshold", "enable_mamba_shared_prefix_checkpoint",
                                               "indexer_kv_dtype", "num_speculative_tokens_per_batch_size")}}
    if arm["key"] == "DYN":
        rec["dyn_tiers"] = S2.dyn_tiers(o, seqs=int(arm["streams"]))
        rec["dyn_graphs_end"] = S2.dyn_graph_readback(t, arm.get("captures"))
    if arm["key"] == "IDX":
        rec["idx_readback_end"] = S2.idx_readback(o.get("workers_end") or o.get("workers"))
    return rec


def nbs2_finalize():
    try:
        recs = {k: _arm_rec(k) for k in ARMS}
        v = S2.verdict({k: r for k, r in recs.items() if r is not None}, ARMS)
    except Exception as e:
        v = {"verdict": "ERROR_IN_VERDICT", "why": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
    v["arms_status"] = {k: (_arm_rec(k) or {}).get("status") for k in ARMS}
    v["patch"], v["seam"], v["drafter_ok"] = NBDF.get("patch"), NBDF.get("seam_seq"), (NBDF.get("drafter") or {}).get("ok")
    v["kv_kit_gate"], v["responses_saved"] = KV_KIT_GATE.get("passed"), RESP["counts"]
    v["minutes"] = round((time.time() - T_START) / 60, 1)
    NBDF["verdict"] = NBS2["verdict"] = v
    try:
        (OUT / "nbs2_verdict.json").write_text(json.dumps(v, indent=1, default=str))
    except Exception:
        pass
    save()
    log("NB-S2 VERDICT", v.get("verdict"), "|", str(v.get("why"))[:600])
    return v


def nbs2_main(runner=None, pseq=None, sseq=None):
    """NBDF-K100's launch path, B first: merge -> install -> #60490 + extension -> k100_setup (merge + K6, the QSA scales)
    -> sha gate -> nb4_ppack (PSEQ.before_arm(B)) -> dflash_seam (SSEQ.apply) -> every arm in priority order with its own
    drafter / KV-gate / seam / budget / tree checks (B > IDX > S28 > LP > S32 > SPC). A failure stops only that arm;
    B not usable is recorded and the levers run. An arm switched off in CELL 1 (NBS2_ON) is recorded SKIPPED_SWITCH_OFF
    before any other check and never booted (B off: the levers still run). The seam is reverted first, then nb4_ppack; the verdict in the finally."""
    runner = runner or run_arm
    pseq = pseq or PSEQ
    sseq = sseq or SSEQ
    try:
        for k in ARMS:                           # E25 NB-S2: CELL 1 arm switch off -> not booted (recorded; the other arms unaffected)
            if not NBS2_SWITCH.get(k, True):
                skip_arm(k, "switched off in CELL 1 (NBS2_ON)", "SKIPPED_SWITCH_OFF")
        RUN_KEYS = [k for k in ARMS if NBS2_SWITCH.get(k, True)]
        NBS2["run_keys"] = list(RUN_KEYS)
        if not RUN_KEYS:
            NBDF["stop"] = "ALL_ARMS_SWITCHED_OFF: no setup, no boot"
            return
        m = R["sections"]["A0"]["merge"]
        if not m.get("ok"):
            NBDF["stop"] = "MERGE_FAILED: no boot started"
            for k in RUN_KEYS:
                skip_arm(k, NBDF["stop"])
            return
        step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        pt = step("A0", "patch_60490", 3, apply_60490)
        if pt.get("status") != "DONE" or not pt.get("ok"):
            for k in RUN_KEYS:
                skip_arm(k, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
            return
        ks = step("A0", "k100_setup", 3, k100_setup)
        if ks.get("status") != "DONE" or ks.get("ok") is not True:
            for k in RUN_KEYS:
                skip_arm(k, f"K100 setup: {ks.get('why') or ks.get('error') or ks.get('status')}", "NOT_INTEGRATED", not_integrated=True)
            return
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            for k in RUN_KEYS:
                skip_arm(k, "sha gate: " + why, "SKIPPED_SHA_GATE")
            return
        ok_p, why_p = pseq.before_arm(ARMS["B"])
        NBDF["patch"] = _patch_rec(pseq)
        NBS2["ppack_first_apply"] = {"ok": ok_p, "why": why_p}
        if SEAM_API is None and sseq is SSEQ:
            seam_ok, seam_why = False, "NOT_INTEGRATED: " + str(SEAM_LOAD.get("error"))
        else:
            seam_ok, seam_why = sseq.apply()
        ring = N.ring_status(sseq) if seam_ok else (False, False, "no seam")
        NBDF["seam_seq"], NBDF["seam_ok"], NBDF["seam_why"] = _seam_rec(sseq), seam_ok, seam_why
        NBDF["ring"] = {"available": ring[0], "verified": ring[1], "why": ring[2]}
        try:
            NBDF["ppack_after_seam"] = P.verify_ppack_patch(_tree_py()[0])["state"] if not FAKE else "fake"
        except Exception as e:
            NBDF["ppack_after_seam"] = repr(e)[:200]
        save()
        log("NB-S2 seam:", seam_ok, seam_why, "| drafter:", DRAFTER.get("status"), DRAFTER.get("why"), "| walk:", NBDF["budget"].get("planned_walk"))
        for k, a in ARMS.items():
            if k not in RUN_KEYS:                # E25 NB-S2: switched off (recorded above, before any other check)
                continue
            if not DRAFTER.get("ok"):
                skip_arm(k, f"{DRAFTER.get('status') or 'DRAFTER'}: {DRAFTER.get('why')}", "NOT_INTEGRATED", not_integrated=True)
                continue
            if not KV_KIT_GATE.get("passed"):
                skip_arm(k, "NBDF-K100's real-kit get_kv_cache_configs gate did not pass: " + str(KV_KIT_GATE.get("why")), "SKIPPED_KV_KIT_GATE")
                continue
            if not seam_ok:
                skip_arm(k, seam_why, "NOT_INTEGRATED", not_integrated=True)
                continue
            if a.get("ring_needed") and not ring[1]:
                skip_arm(k, f"K {a['k']} needs the QSA ring widening: {ring[2]}", "SKIPPED_NO_RING")
                continue
            if not N.arm_fits(left(), a):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(priority " + " > ".join(RUN_KEYS) + ")", "SKIPPED_BUDGET")
                continue
            ok_s, why_s = sseq.check(k)
            ok_p, why_p = pseq.before_arm(a)
            NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
            if not ok_s or not ok_p:
                skip_arm(k, why_s if not ok_s else why_p, "TREE_INVALID_MIXED" if "INVALID" in (why_s + why_p) else "NOT_INTEGRATED",
                         not_integrated=True)
                NBS2.setdefault("tree_notes", []).append(f"{k}: " + (why_s if not ok_s else why_p))
                continue
            r = step(k, a["name"], a["est_min"], lambda k=k: runner(k))
            if k == "B":
                bq = N.base_usable(r)
                NBS2["base"] = {"usable": bq[0], "label": bq[1], "why": bq[2], "role": "recorded only: the levers run anyway"}
                if not bq[0]:
                    log("NB-S2: B not a usable base (recorded, the levers run):", bq[1], bq[2])
            for kind, hit in (("nometa", N.needs_nometa(r)), ("iso", N.needs_isolation(r)),
                              ("tap", (r.get("smoke_pos0") or {}).get("label") in ("STRUCTURAL", "TAP_SUSPECT"))):
                if hit:                          # recorded: NB-S2 runs no reruns (the budget goes to the levers)
                    NBS2.setdefault("rerun_flags", []).append([kind, k])
            save()
    finally:
        try:
            sseq.finish()
        except Exception as e:
            sseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        try:
            pseq.finish()
        except Exception as e:
            pseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
        nbs2_finalize()


'''
