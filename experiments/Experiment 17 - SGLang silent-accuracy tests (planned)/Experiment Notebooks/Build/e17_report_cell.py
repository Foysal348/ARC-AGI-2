# ==================================================================== CELL E17-R — report, bundle
try:
    E14_STATE.enter("REPORT")
    _E = globals().get("E17") or {}
    _p0 = _E.get("phase0") or {}
    L = ["# E17: SGLang silent-accuracy tests (#43567 chain, DFlash parity)", "",
         f"profile {PROFILE}, rerun {RERUN}, engine {ENGINE_INFO.get('backend')}, plan {_E.get('plan')}",
         f"E17 cell error: {(_E.get('error') or 'none')[-800:]}", "",
         "## Phase 0", "",
         f"- schedule_batch.py sha256 {_p0.get('schedule_batch_sha256')}; #43567 at start {_p0.get('patch_43567_state')}; "
         f"after apply {_E.get('patch_check_after_apply')}; at the end {_E.get('patch_state_end')}",
         f"- DFlash taps {_p0.get('dflash_target_layer_ids')} (num_target_layers {_p0.get('dflash_num_target_layers')}, "
         f"block {_p0.get('dflash_block_size')}): ok {_p0.get('dflash_taps_ok')}",
         f"- vocab config {_p0.get('vocab_size_config')}, used {_E.get('vocab_used')}; GDN backend line {_p0.get('gdn_backend_line')}, "
         f"KDA line {_p0.get('kda_backend_line')}",
         f"- checkpoint KV-scale keys {_p0.get('checkpoint_kv_scale_n')} {_p0.get('checkpoint_kv_scale_keys')}; scale log lines "
         f"{_p0.get('kv_scale_log_lines')}", ""]
    _ks = None
    for _a in (_E.get("arms") or {}).values():
        _ks = _ks or (((_a or {}).get("trace") or {}).get("kv_scales"))
    L.append(f"- runtime KV scales (TP0, first {len((_ks or {}).get('rows') or [])} of {(_ks or {}).get('n')} layers): "
             f"{json.dumps(((_ks or {}).get('rows') or [])[:6], default=str)[:900]}")
    L += ["", "## T1 arms (full-vocabulary KL, reference = the same prompt after flush + warm prefixes)", "",
          "| arm | launch | slots / interval | verdict | prompt | ref cached | seed cached | reuse cached | A/A KL max | tol | reuse KL max | "
          "top-1 changed | alloc attempts / failed | stale candidate / confirmed | spec fallback / crossed | safety leaks / stale |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _n in _E.get("order") or []:
        _a = (_E.get("arms") or {}).get(_n) or {}
        if "t1" not in _a:
            continue
        _k = ((_a.get("trace") or {}).get("counters") or {})
        _s = _a.get("safety") or {}
        for _t in _a["t1"]:
            L.append(f"| {_n} | {_a.get('launch')} | {_a.get('slots')} / {_a.get('interval')} | {_a.get('verdict')} | {_t['prompt']} | "
                     f"{_t['ref_cached']} | {_t['seed_cached']} | {_t['reuse_cached']} | {_t['aa_kl_max']:.2e} | {_t['tol']:.1e} | "
                     f"{_t['reuse_kl_max']:.3e} | {sum(1 for x in _t['reuse'] if x['top1_changed'])}/{len(_t['reuse'])} | "
                     f"{_k.get('optional_alloc_attempts', 0)} / {_k.get('optional_alloc_failed', 0)} | "
                     f"{_k.get('stale_donation_candidate', 0)} / {_k.get('stale_donation_confirmed', 0)} | "
                     f"{_k.get('spec_inplace_fallback', 0)} / {_k.get('spec_crossed', 0)} | "
                     f"{_s.get('leaks', '')} / {_s.get('stale', '')} |")
    for _n in _E.get("order") or []:
        _a = (_E.get("arms") or {}).get(_n) or {}
        if "t1" not in _a and _n.startswith("L") and _n[1:2].isdigit() and ("verdict" in _a or "launch_ok" in _a or "error" in _a):
            if _n.startswith(("L1_", "L2_", "L3_", "L4_", "L5_", "L7_", "L8_", "L9_")):
                L.append(f"| {_n} | — | {_a.get('slots')} | {_a.get('verdict') or ('LAUNCH_FAILED: ' + str(_a.get('why')) if _a.get('launch_ok') is False else _a.get('error', '')[:200])} | | | | | | | | | | | | |")
    L += ["", f"**Decision A:** {json.dumps(_E.get('decision_A'))}", ""]
    _l6 = (_E.get("arms") or {}).get("L6_class2_mtp")
    L.append(f"**L6 control (no_buffer + MTP, #43387 path):** {json.dumps(_l6, default=str)[:1200] if _l6 else _E.get('skipped', {}).get('L6_class2_mtp', 'not run')}")
    L += ["", "## T2 natural pressure (production flags, 16 streams)", ""]
    for _n in ("L11_t2_patched", "L10_t2_unpatched"):
        _a = (_E.get("arms") or {}).get(_n)
        if _a:
            _k = ((_a.get("trace") or {}).get("counters") or {})
            L.append(f"- {_n}: verdict {_a.get('verdict')}; stats {_a.get('stats')}; failures per 1,000 boundaries "
                     f"{_a.get('fail_per_1000_boundaries')}; stale per 1,000 prefix hits {_a.get('stale_per_1000_prefix_hits')}; "
                     f"counters {_k}; alloc {(_a.get('trace') or {}).get('alloc')}")
            for _pp in _a.get("probes") or []:
                L.append(f"  - base {_pp['base']} ({_pp['len']} tok): warm cached {_pp['warm_cached']}, cold {_pp['cold_cached']}, "
                         f"KL cold||warm {_pp['kl_cold_warm']['kl']:.3e} (top-1 {_pp['kl_cold_warm']['top1_changed']}), cold A/A {_pp['kl_cold_cold']:.3e}")
    if "T2" in (_E.get("skipped") or {}):
        L.append(f"- T2 skipped: {_E['skipped']['T2']}")
    L += ["", "## T3b DFlash vs target-only (radix off, greedy)", ""]
    for _n in ("L12_t3b_target", "L13_t3b_dflash"):
        _a = (_E.get("arms") or {}).get(_n) or {}
        _cmp = {k: {kk: v.get(kk) for kk in ("compared", "expected", "identical", "divergence_p50", "large_margin_flips", "missing_margins")}
                for k, v in _a.items() if isinstance(v, dict) and "large_margin_flips" in v}
        L.append(f"- {_n}: verdict {_a.get('verdict')}; compares {json.dumps(_cmp)}; rows {json.dumps(_a.get('rows'))}; "
                 f"accept {_a.get('accept_lines')}; error {(_a.get('error') or '')[:300]}")
    L += ["", "## T3a fp16 vs fp32 SSM (long prefixes, target-only, radix off)", "",
          f"- fp16 A/A: {json.dumps(((_E.get('arms') or {}).get('L12_t3b_target') or {}).get('t3a_fp16'), default=str)}",
          f"- fp32 vs fp16: {json.dumps(((_E.get('arms') or {}).get('L15_t3a_fp32') or {}).get('rows'), default=str)[:1500]}",
          "", f"skipped: {json.dumps(_E.get('skipped'))}", "",
          "## Launches", "", "| launch | ok | sec | hook self-test | caps |", "|---|---|---|---|---|"]
    for _n, _r in (_E.get("launches") or {}).items():
        L.append(f"| {_n} | {_r.get('ok')} {_r.get('skipped') or ''} | {_r.get('sec')} | {(_r.get('hook_selftest') or {}).get('ok') if _r.get('hook') else 'off'} | "
                 f"{json.dumps(_r.get('caps'))[:200]} |")
    L += ["", "Verdict ceiling of this session: ENGINE_CORRECTNESS_VALIDATED. No LB uplift is claimed."]
    (E14_OUT / "e17_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    E14_STATE.end("done")
    e14_save()
    _pack = WORKING / "Q27X_E17_BUNDLE.zip"
    with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(E14_OUT.rglob("*")):
            if fp.is_file() and fp.stat().st_size < 300e6:
                z.write(fp, arcname="e14/" + fp.relative_to(E14_OUT).as_posix())
        for fp in sorted(OUT.glob("sglang_server_*.log")):
            z.write(fp, arcname="logs/" + fp.name)
        for fp in (WORKING / "e14_state.json", WORKING / "e14_heartbeat.log"):
            if fp.exists():
                z.write(fp, arcname=fp.name)
    print("E17 bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
except BaseException:
    print("E17 report error:", traceback.format_exc()[-2000:], flush=True)
try:
    e11_kill_server(wait=False)
except Exception:
    pass
