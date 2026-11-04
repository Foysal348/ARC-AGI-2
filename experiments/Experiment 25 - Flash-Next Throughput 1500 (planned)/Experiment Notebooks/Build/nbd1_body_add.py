"""E25 NB-D1: the code blocks the NB-D1 builder (build_e25_nbd1.py) inserts into NBDF-K100's CELL 7 body (text constants only;
nothing here runs at import). Each block is placed by an exact-once anchor in build_e25_nbd1.py. Derived from NB-S2's
nbs2_body_add.py (copied, renamed; NB-S2's files are not edited)."""

# ---- after NBDF-K100's checked arm plan (N.check_arm_plan(ARMS) on Q / D3s / D4s): B = the E29_DFLASH config, the levers on B
D1_SETUP = r'''
# ==================================================================== E25 NB-D1: B = E29_DFLASH production (NBDF-K100 D3s + EMBED_HOST + QSA 128/64), levers on B
NBDF_ARMS = ARMS                            # NBDF-K100's own plan (Q, D3s, D4s), checked by nbdf_lib.check_arm_plan
NBD1_LIB_SHA = "__D1LIBSHA__"
NBD1_LIB_SRC = _nbd1_ds("nbd1_lib.py", NBD1_LIB_SHA)   # the payload dataset (sha-pinned)
if hashlib.sha256(NBD1_LIB_SRC.encode("utf-8")).hexdigest() != NBD1_LIB_SHA:
    raise RuntimeError("EMBED_SHA_MISMATCH: nbd1_lib")
(TMP / "nbd1_lib.py").write_text(NBD1_LIB_SRC, encoding="utf-8")
D1 = importlib.import_module("nbd1_lib")
D1_CFG = json.loads(r"""__D1CFG__""")
_nbd1_on = globals().get("NBD1_ON")          # CELL 1 arm switches (E25 NB-D1): absent -> every key on, a missing key -> off
NBD1_SWITCH = D1.switch_map(_nbd1_on)
N.restrict_arms(("Q", "D3s"))                # N.SW_BLOCK_ARMS = (D3s,): every NB-D1 arm has parent D3s (launch_check, KV summary)
ARMS = D1.arm_plan(N, NBDF_ARMS["D3s"], seam_env_k4=SEAM_ENVS.get("D4s"), switch=NBD1_SWITCH, pin=int(K100_CFG["q_pin"]),
                   seqs=int(K100_CFG["seqs"]))
D1.check_arm_plan(N, T, ARMS, NBDF_ARMS["D3s"])   # B = D3s + EMBED_HOST + QSA 128/64; each lever = B + exactly its change (raises: no boot)
NBD1 = R.setdefault("nbd1", {})
NBD1.update(lib=D1.LIB_VERSION, lib_sha256=NBD1_LIB_SHA, cfg=D1_CFG, switch=D1.switch_record(_nbd1_on),
            walk=D1.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60), need_min=D1.copy_need_min(ARMS, NBD1_SWITCH),
            question="which 1500+ speed lever (QSA workspace, K4, dynamic K, GDN switches, indexer fp8, long prefill, seqs 28 / 32, "
                     "shared-prefix checkpoint) beats the DFlash2 production base B at 8k-116k (raw numbers, labels record-only)?",
            seam_env_d3s_vs_d4s_equal=SEAM_ENVS.get("D3s") == SEAM_ENVS.get("D4s"), dyn_source_check=D1.DYN_SOURCE_CHECK, b_ref=D1.B_REF,
            arms={k: {x: a.get(x) for x in ("name", "port", "d1_change", "spec", "captures", "streams", "windows_fixed", "prof_windows",
                                             "est_min", "cap_min", "time_model", "title", "d1_items", "env", "extra")} for k, a in ARMS.items()})
print("E25 NB-D1 arm switches:", NBD1_SWITCH, "| walk:", json.dumps(NBD1["walk"])[:700], flush=True)
'''

# ---- after the K100 prompt set: 20k prompts for 32 sequences (variants 0-23 unchanged: B's 20k@24 = NBDF-K100's)
D1_PROMPTS = r'''
if len(PROMPTS["20k"]) < 32:                # E25 NB-D1: S32 at 20k needs 32 distinct prompts (variants 24-31 added)
    PROMPTS["20k"] = list(PROMPTS["20k"]) + [prompt(BANDS["20k"], v) for v in range(len(PROMPTS["20k"]), 32)]
'''

# ---- the ledger, the SPC probe, the read-backs, the main sequence and the records (before _gpu_util)
D1_MAIN = r'''
# ==================================================================== E25 NB-D1: ledger, SPC probe, read-backs, main, records
class D1Ledger:
    """The KV ledger of one window: /metrics every 2 s in a daemon thread (perf_counter time, the clock of steady()'s t0 /
    t1): KV usage, running / waiting, waiting by reason, preemptions, prompt tokens by source (records only)."""

    def __init__(self, port, every=2.0):
        self.port, self.every, self.samples, self.stop_ev, self.th = port, every, [], threading.Event(), None

    def _one(self):
        try:
            txt = L.U.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=15).read().decode("utf-8", "replace")
            self.samples.append((time.perf_counter(), D1.parse_metrics(txt)))
        except Exception:
            pass

    def _loop(self):
        while not self.stop_ev.wait(self.every):
            self._one()

    def __enter__(self):
        if not FAKE:
            self._one()
            self.th = threading.Thread(target=self._loop, daemon=True, name=f"nbd1_ledger_{self.port}")
            self.th.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()
        if self.th is not None:
            self.th.join(10)
        if not FAKE:
            self._one()
        return False


def d1_window_ledger(o, r, band, n):
    """The window's KV ledger with THIS boot's pool and F (records only)."""
    smp = r.pop("_d1_samples", None) or []
    res = o.get("k100_residents") or {}
    f_ = ((res.get("40k") or {}).get("F"))
    blk = ((res.get("40k") or {}).get("block")) or N.BLOCK_TOKENS
    return {"window": D1.ledger_summary(smp, r.get("t0"), r.get("t1"), res.get("pool"), n, BANDS.get(band), f_, int(blk),
                                        r.get("gen_tokens") or 0),
            "with_admission": D1.ledger_summary(smp, None, None, res.get("pool"), n, BANDS.get(band), f_, int(blk), r.get("gen_tokens") or 0)}


def d1_live(o, arm, key, r):
    """The [LIVE] line after every window (print only; never raises)."""
    try:
        b_rec = None if arm["key"] == D1.BASE_KEY else _arm_rec(D1.BASE_KEY)
        print(D1.live_line((time.time() - T_START) / 60.0, arm["key"], key, r, b_rec), flush=True)
    except Exception as e:
        print(f"[LIVE] {arm.get('key')} {key}: line error {type(e).__name__}: {e}"[:200], flush=True)


def sibling_probe(port, o):
    """SPC activation read-back (recorded): after a cache reset, three siblings A / B / C = P + a different suffix (P =
    D1_CFG sibling prefix_tokens, not a block multiple); C's prefix-hit tokens show whether the shared-prefix junction was
    checkpointed (D1.sibling_label)."""
    c = D1_CFG["sibling"]
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
    out["label"] = D1.sibling_label(out, block)
    return out


def d1_readback_failure(o, arm, t):
    """NB-D1 real failures of THIS arm (its own one change read back with another value, or another lever's change present:
    the arm would measure the wrong thing): Q256's QSA env / the GDN switch in the 4 workers, the indexer cache dtype, the
    logged seqs / batched tokens / long-prefill / shared-prefix flags, the logged K / dynamic schedule, the DYN fallback lines
    (NB-S1). B's base settings (QSA 128 / 64, EMBED_HOST) read back differently are recorded only. None = no failure."""
    o["d1_readback"] = D1.readback(arm, t, o.get("env_readback_raw"), o.get("config_readback"), o.get("workers"), o.get("dflash_readback"))
    if o["d1_readback"].get("base_mismatch"):   # B's base settings read back differently: recorded, the arm measures on
        o.setdefault("recorded_gates", []).append("NB-D1 base settings (recorded, arm continues): " + "; ".join(o["d1_readback"]["base_mismatch"])[:400])
    if FAKE:
        return None
    if o["d1_readback"].get("real_failure"):
        return "NB-D1 read-back: " + str(o["d1_readback"].get("why"))[:400]
    return None


def _k_eff(w):
    a, r = (w or {}).get("accept_per_draft"), (w or {}).get("accept_rate")
    if isinstance(a, (int, float)) and isinstance(r, (int, float)) and r > 0:
        return round(a / r, 3)
    return None


def d1_records(o, arm, t):
    """Record-only facts of one arm: K_eff per window, the KV ledgers, the pool / residents, the QSA read-back over the whole
    log (Merlin's logged granule at the first prefill), the logged flags, the indexer dtype at the end."""
    st_ = o.get("steady") or {}
    rec = {"k_eff": {k: _k_eff(w) for k, w in st_.items() if isinstance(w, dict) and not k.startswith("profile")},
           "kv_ledger": {k: w.get("kv_ledger") for k, w in st_.items() if isinstance(w, dict) and w.get("kv_ledger")},
           "residents": o.get("k100_residents"), "pool": (o.get("k100_residents") or {}).get("pool"),
           "engine_block": o.get("engine_block"), "qsa_end": D1.qsa_readback(arm, o.get("env_readback_raw"), t),
           "idx_end": D1.idx_readback(o.get("workers_end") or o.get("workers")),
           "log_flags": {f: (f in t) for f in ("long_prefill_token_threshold", "enable_mamba_shared_prefix_checkpoint",
                                               "indexer_kv_dtype", "num_speculative_tokens_per_batch_size")},
           "dyn_fallback_lines": [l.strip()[:200] for l in t.splitlines() if any(x in l for x in D1.DYN_FAIL_RX)][:4]}
    return rec


def nbd1_finalize():
    try:
        recs = {k: _arm_rec(k) for k in ARMS}
        v = D1.verdict(T, {k: r for k, r in recs.items() if r is not None}, ARMS, NBD1_SWITCH, N.LONG_THR_FLOOR)
    except Exception as e:
        v = {"verdict": "ERROR_IN_RECORDS", "why": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
    v["arms_status"] = {k: (_arm_rec(k) or {}).get("status") for k in ARMS}
    v["patch"], v["seam"], v["drafter_ok"] = NBDF.get("patch"), NBDF.get("seam_seq"), (NBDF.get("drafter") or {}).get("ok")
    v["kv_kit_gate"], v["responses_saved"] = KV_KIT_GATE.get("passed"), RESP["counts"]
    v["minutes"] = round((time.time() - T_START) / 60, 1)
    R["sections"].setdefault("PROF", {})["nbd1_prof"] = v.get("prof") or D1.prof_record(_arm_rec(D1.BASE_KEY), NBD1_SWITCH)
    NBDF["verdict"] = NBD1["verdict"] = v
    try:
        (OUT / "nbd1_verdict.json").write_text(json.dumps(v, indent=1, default=str))
    except Exception:
        pass
    save()
    log("NB-D1 RECORDS", v.get("verdict"), "|", str(v.get("why"))[:600])
    return v


def nbd1_main(runner=None, pseq=None, sseq=None):
    """NBDF-K100's launch path, B first: merge -> install -> #60490 + extension -> k100_setup (merge + K6, the QSA scales)
    -> sha gate -> nb4_ppack (PSEQ.before_arm(B)) -> dflash_seam (SSEQ.apply) -> every switched-on arm in order with its own
    drafter / KV-gate / seam / budget / tree checks. A failure stops only that arm (no cross-arm skip); B not usable is
    recorded and the levers run. An arm switched off in CELL 1 (NBD1_ON) is recorded SKIPPED_SWITCH_OFF before any other
    check and never booted. The seam is reverted first, then nb4_ppack; the records in the finally."""
    runner = runner or run_arm
    pseq = pseq or PSEQ
    sseq = sseq or SSEQ
    try:
        for k in ARMS:                           # E25 NB-D1: CELL 1 arm switch off -> not booted (recorded; the other arms unaffected)
            if not NBD1_SWITCH.get(k, True):
                skip_arm(k, "switched off in CELL 1 (NBD1_ON)", "SKIPPED_SWITCH_OFF")
        RUN_KEYS = [k for k in ARMS if NBD1_SWITCH.get(k, True)]
        NBD1["run_keys"] = list(RUN_KEYS)
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
        ok_p, why_p = pseq.before_arm(ARMS[D1.BASE_KEY])
        NBDF["patch"] = _patch_rec(pseq)
        NBD1["ppack_first_apply"] = {"ok": ok_p, "why": why_p}
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
        log("NB-D1 seam:", seam_ok, seam_why, "| drafter:", DRAFTER.get("status"), DRAFTER.get("why"), "| walk:", NBD1.get("walk"))
        for k, a in ARMS.items():
            if k not in RUN_KEYS:                # E25 NB-D1: switched off (recorded above, before any other check)
                continue
            if not DRAFTER.get("ok"):
                skip_arm(k, f"{DRAFTER.get('status') or 'DRAFTER'}: {DRAFTER.get('why')}", "NOT_INTEGRATED", not_integrated=True)
                continue
            if not KV_KIT_GATE.get("passed"):
                skip_arm(k, "the NB-D1 real-kit gate did not pass: " + str(KV_KIT_GATE.get("why")), "SKIPPED_KV_KIT_GATE")
                continue
            if not seam_ok:
                skip_arm(k, seam_why, "NOT_INTEGRATED", not_integrated=True)
                continue
            if a.get("ring_needed") and not ring[1]:
                skip_arm(k, f"K {a['k']} needs the QSA ring widening: {ring[2]}", "SKIPPED_NO_RING")
                continue
            if not N.arm_fits(left(), a):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(order " + " > ".join(RUN_KEYS) + ")", "SKIPPED_BUDGET")
                continue
            ok_s, why_s = sseq.check(k)
            ok_p, why_p = pseq.before_arm(a)
            NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
            if not ok_s or not ok_p:
                skip_arm(k, why_s if not ok_s else why_p, "TREE_INVALID_MIXED" if "INVALID" in (why_s + why_p) else "NOT_INTEGRATED",
                         not_integrated=True)
                NBD1.setdefault("tree_notes", []).append(f"{k}: " + (why_s if not ok_s else why_p))
                continue
            r = step(k, a["name"], a["est_min"], lambda k=k: runner(k))
            if k == D1.BASE_KEY:
                bq = N.base_usable(r)
                NBD1["base"] = {"usable": bq[0], "label": bq[1], "why": bq[2], "role": "recorded only: the levers run anyway"}
                R["sections"].setdefault("PROF", {})["nbd1_prof"] = D1.prof_record(r, NBD1_SWITCH)
                if not bq[0]:
                    log("NB-D1: B not a usable base (recorded, the levers run):", bq[1], bq[2])
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
        nbd1_finalize()


'''
