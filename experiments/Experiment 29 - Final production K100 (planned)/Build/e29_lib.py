"""E29 FINAL (2026-11-01): pure helpers for the switchable production notebook (build_e29_final.py). Stdlib only.

The flags (FINAL_CFG in the builder; every one OFF = E28 K100 exactly):
  drafter         "mtp" (E28) | "dflash_d3s" | "dflash_d4s"  (NBDF-K100 DFlash2 boot path; fallback = MTP K3 at the same pin)
  indexer_fp8     --attention-config.indexer_kv_dtype fp8      (fallback: the same try without it)
  embed_host      DFLASH_DRAFT_EMBED_HOST=1 from the first DFlash try (dflash only)
  draft_vocab     None | "32k" | "64k": VLLM_DRAFT_VOCAB_FILE + VLLM_DRAFT_VOCAB_SHA256 (MTP only; fallback: without)
  gdn_fastpath    VLLM_GDN_DECODE_FASTPATH=1                   (fallback: without)
  gdn_align       VLLM_GDN_SHARED_ALIGN_INDEX=1                (fallback: without)
  workers         24 (E28) | 28 | 32: workers = --max-num-seqs on the top rung; captures scale with it
  attempts_supply the E29 scheduler S-1 (e29_sched_patch.py); fallback: the E28 scheduler

Real-failure gates only: a pre-phase boot failure (launch / OOM / not ready / probes / C-11 / the DFlash read-back /
the zero-accept canary) drops ONE E29 lever (lever_to_drop) and retries the same pin; with no E29 lever left the E28
ladder runs unchanged. Labels (AL per hour, prefix hit, engine block) are recorded only.

1.1 (2026-11-02, the DFlash notebook only; DEFAULT / MTP_V64 keep 1.0 = Build/e29_lib_v1_0.py byte for byte), from the E25
NB-S1 GPU runs 2026-11-01/02: the seam pin = E27's dflash_seam.py 341d859b (adds the GRD edit of Merlin vllm/v1/e2_guards.py,
query_cudagraph_manager; with 7356d473 every DFlash arm died at capture); an OOM at ANY stage of a D try (launch, first
request / probes, warm-up, before the phases, in-run) that is not a device fault steps the D ladder (df_oom_step,
df_relaunch_plan); a lever is "named" only by an error line with its own flag text (the substring "indexer" is in every
Merlin log: qsa_indexer.py INFO lines, the config dump); the warm-up gate (df_warmup_failure: dead server / request errors /
an empty stream = real failures).
"""
import json
import math
import re

LIB_VERSION = "e29_lib 1.1 (2026-11-02)"
DRAFTERS = ("mtp", "dflash_d3s", "dflash_d4s")
K_OF = {"dflash_d3s": 3, "dflash_d4s": 4}
LEVER_ORDER = ("df", "idx", "vocab", "gdn_fastpath", "gdn_align")      # the drop order on a pre-phase failure
GDN_ENV = {"gdn_fastpath": "VLLM_GDN_DECODE_FASTPATH", "gdn_align": "VLLM_GDN_SHARED_ALIGN_INDEX"}
VOCAB_FILES = {"32k": ("draft_vocab_k100_arc_n32768.pt", "a164d1a71d27dab600c014673c34accfdeb8695141a63be8aa4bfb4026695248"),
               "64k": ("draft_vocab_k100_arc_n65536.pt", "a915fa6d5ad711817976de5c988a3950f43e7b2fbc058b04325e75c8501cf022")}
SEAM_FILE = ("dflash_seam.py", "341d859be5b2cc619247ff7b645304b9473d09f07995358ead2ee02be116c60b")   # 1.1: E27 seam + GRD edit
SEAM_PREV_SHA = "7356d4738962bdbab4091264861c4bfbbdcd7098d6c75b9c48db69eeb73e6937"   # 1.0's pin (no GRD edit: died at capture)
GRD_REL = "vllm/v1/e2_guards.py"                                   # the seam's GRD edit (B1 fix): file + post-patch sha
GRD_POST = "d50d754e8479cf77b577f283ef6456a8a8db550b179ca9aa6a4c26d3126bde11"
GRD_LOG = "draft.query_cudagraph_manager=FULL_DECODE_ONLY"         # Merlin's e2-guard runtime line (NB-S1: on every D boot)
XID = re.compile(r"NVRM: Xid|\bXid \d+|CUDA error: an illegal memory access|illegal memory access was encountered|"
                 r"unspecified launch failure|CUDA error: misaligned address")   # = the notebook's XID (device faults)
DF_GATES = ("E29_DFLASH_NOT_ENGAGED", "E29_DFLASH_ZERO_ACCEPT", "E29_DFLASH_CONFIG_MISMATCH")   # DFlash gates: never a rung step
IDX_ARGS = ["--attention-config", '{"indexer_kv_dtype":"fp8"}']   # = --attention-config.indexer_kv_dtype fp8 (vLLM's JSON form;
# the K-9 help gate needs a flag that `serve --help=all` lists: the dotted key is not listed; Merlin emits no --attention-config)
D_RETRY = [[7933526016, True], [7665090560, True], [7396655104, True], [6859784192, True]]   # NBDF-K100 K100_CFG d_retry
D_PIN0 = 7933526016                                                # the K100Q / E28 top K3 rung (the MTP fallback top pin)
D_START_PIN = 7665090560   # rev 2f (GPU smoke 2026-11-02): with EMBED_HOST + QSA 128/64 the D3s boot at D_PIN0 always failed the
                           # F2 probe (PP1 465 MiB < 512 -> one step, ~6.5 min lost); 7,665,090,560 passed (903 MiB): the first D try
FP8_PIN = 2479838700                                               # Merlin's FP8 pin (nb5_lib.pin_tries base)
CANARY_MIN_DRAFTS, CANARY_MIN_ACCEPT = 200, 0.02                   # nbdf_lib SMOKE_MIN_DRAFTS / SMOKE_MIN_ACCEPT
AL_DROP = 0.7                                                      # hourly AL < 0.7 x the first hour -> flagged (record)
SD_DRAFTS, SD_DRAFT_TOK, SD_ACC = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
                                   "vllm:spec_decode_num_accepted_tokens_total")
DF_ARCH = "DFlash2DraftModel"


def default_cfg():
    return {"drafter": "mtp", "indexer_fp8": False, "embed_host": False, "draft_vocab": None, "gdn_fastpath": False,
            "gdn_align": False, "workers": 24, "attempts_supply": False}


def check_cfg(cfg):
    """Raises on an unknown / inconsistent flag value (build time and notebook start)."""
    c = dict(default_cfg(), **(cfg or {}))
    assert set(c) == set(default_cfg()), sorted(set(c) ^ set(default_cfg()))
    assert c["drafter"] in DRAFTERS, c["drafter"]
    assert c["draft_vocab"] in (None, "32k", "64k"), c["draft_vocab"]
    assert int(c["workers"]) in (24, 28, 32), c["workers"]
    assert not (c["embed_host"] and c["drafter"] == "mtp"), "embed_host is a DFlash flag"
    assert not (c["draft_vocab"] and c["drafter"] != "mtp"), "draft_vocab is an MTP flag"
    for k in ("indexer_fp8", "embed_host", "gdn_fastpath", "gdn_align", "attempts_supply"):
        assert isinstance(c[k], bool), (k, c[k])
    return c


def is_default(cfg):
    return check_cfg(cfg) == default_cfg()


def gib_of(kv_bytes):
    """nb5_lib.pin_tries' GiB step over Merlin's FP8 pin (the boot-try name uses it)."""
    return (int(kv_bytes) - FP8_PIN) / 2 ** 30


def df_tries(embed_host_first):
    """The NBDF-K100 D pin ladder: the top K3 rung (EMBED_HOST only when the flag asks), then EMBED_HOST at the same pin,
    then 0.25-GiB rungs with EMBED_HOST (an OOM moves one step; any other failure ends DFlash -> MTP at the top rung).
    rev 2f: with EMBED_HOST first, the rungs above D_START_PIN are skipped (the GPU smoke: they always step on the F2 probe)."""
    out = _df_tries_all(embed_host_first)
    return [t for t in out if t["kv_bytes"] <= D_START_PIN] if embed_host_first else out


def _df_tries_all(embed_host_first):
    out = [{"gib": gib_of(D_PIN0), "kv_bytes": D_PIN0, "df": {"embed_host": bool(embed_host_first), "rung": 0}}]
    for j, (kvb, host) in enumerate(D_RETRY):
        if embed_host_first and j == 0:
            continue                                   # the first try already had EMBED_HOST at this pin
        out.append({"gib": gib_of(kvb), "kv_bytes": int(kvb), "df": {"embed_host": bool(host), "rung": j + 1}})
    return out


def next_try_index(tries, i, oom, df_dropped):
    """After a failed DFlash try i: an OOM -> the next D try (if any); else (or none left) -> the first non-D try."""
    if oom and not df_dropped and i + 1 < len(tries) and tries[i + 1].get("df"):
        return i + 1
    return next((j for j, t in enumerate(tries) if not t.get("df")), len(tries))


def lever_to_drop(on, log_text="", status=None):
    """One E29 lever to drop after a pre-phase failure: a lever the log names first (indexer / draft vocab / GDN env),
    then the DFlash change, then the fixed order. None = no E29 lever is on (E28 rules)."""
    on = [k for k in LEVER_ORDER if k in set(on or ())]
    if not on:
        return None
    # 1.1: only error lines, only the lever's own flag text (1.0 matched "indexer" anywhere: every Merlin log has it)
    t = "\n".join(ln for ln in (log_text or "")[-400000:].splitlines() if re.search(r"Error|Traceback|error:|FAIL_", ln)).lower()
    named = {"idx": ("indexer_kv_dtype", "attention_config", "attention-config"), "vocab": ("draft_vocab",),
             "gdn_fastpath": ("gdn_decode_fastpath",), "gdn_align": ("gdn_shared_align",)}
    for k in on:
        if k != "df" and any(w in t for w in named.get(k, ())):
            return k
    return on[0]


def is_oom(log_text):
    t = (log_text or "")[-400000:]
    return bool(re.search(r"CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError|No available memory for the cache blocks", t))


def is_device_fault(log_text):
    return bool(XID.search((log_text or "")[-400000:]))


def df_oom_step(status, log_text):
    """A failed D try -> the next D rung: an OOM line at any stage (launch, first request / probes, warm-up), not a device
    fault (XID: no retry), not a DFlash gate (read-back / C-11 / zero accept end DFlash at once). = E25 NB-S1's fix
    `oom = oom_in_log and not xid_in_log` (its v1 retried only on LAUNCH_FAILED)."""
    return is_oom(log_text) and not is_device_fault(log_text) and str(status) not in DF_GATES


def df_warmup_failure(warmup, alive):
    """The warm-up of a D try (the first W-stream requests after ready) failed for real: the server died, a request
    errored, or a stream returned no token. None = passed (token counts / speed are never gated)."""
    if not alive:
        return "SERVER_DIED"
    rows = (warmup or {}).get("rows") or []
    errs = [str(e)[:160] for r in rows for e in (r.get("errors") or [])]
    if errs:
        return "REQUEST_ERRORS: " + json.dumps(errs[:3])
    if any(int(n or 0) == 0 for r in rows for n in (r.get("tokens") or [])):
        return "EMPTY_STREAM"
    return None


def df_relaunch_plan(cur, log_text, tries):
    """The in-run death of a DFlash server (E15B watchdog relaunch). rev 2d (F10): a device fault FIRST -> "mtp" (XID alone or
    with an OOM: DFlash dropped, the E28 top rung); a plain OOM -> the D try after the booted one (`cur` = its df plan; tries =
    df_tries), none left -> "mtp"; any other death -> None (E15B: the same config again)."""
    if is_device_fault(log_text):
        return "mtp"
    if not is_oom(log_text):
        return None
    rungs = [t for t in tries if t.get("df")]
    i = next((j for j, t in enumerate(rungs) if cur and t["df"].get("rung") == cur.get("rung")), None)
    return rungs[i + 1] if i is not None and i + 1 < len(rungs) else "mtp"


def grd_check(verify_rec):
    """The seam's GRD edit on the installed tree (dflash_seam.verify_dflash_patch record): e2_guards.py applied (post sha +
    marker). False = a real failure (without it every DFlash boot dies at capture)."""
    f = ((verify_rec or {}).get("files") or {}).get(GRD_REL) or {}
    return f.get("state") == "applied" and f.get("sha") == GRD_POST


def df_spec_argv(spec_json_text, comp_json_text):
    """The trailing DFlash argv (launch.py:110-111: extra args come last, the last occurrence wins)."""
    return ["--speculative-config", spec_json_text, "--compilation-config", comp_json_text]


def strip_df_argv(extra, df_tail):
    """The MTP argv of a booted DFlash config (the in-run fallback): the trailing DFlash args removed (exact match)."""
    extra, n = list(extra or []), len(df_tail or [])
    for i in range(len(extra) - n, -1, -1):
        if n and extra[i:i + n] == list(df_tail):
            return extra[:i] + extra[i + n:]
    return extra


def df_config_check(cr, log_text, k):
    """C-11 for a DFlash try: nb5_lib.config_readback's non-speculative checks (KV dtype, max_num_seqs, pin bytes) +
    the DFlash facts: 'Resolved architecture: DFlash2DraftModel' in the log (NBDF read-back) and, when the log's
    speculative config was parsed, method dflash with K tokens (a wrong method = a real failure)."""
    cr = dict(cr or {})
    chk = {a: b for a, b in (cr.get("checks") or {}).items() if a not in ("speculative_config", "engine_spec", "argv_speculative_json")}
    sc = ((cr.get("log") or {}).get("speculative_config")) if isinstance(cr.get("log"), dict) else None
    chk["dflash_arch_line"] = DF_ARCH in (log_text or "")
    if isinstance(sc, dict) and sc.get("method") is not None:
        chk["dflash_spec_method"] = sc.get("method") == "dflash" and int(sc.get("num_speculative_tokens") or 0) == int(k)
    es = (cr.get("log") or {}).get("engine_spec") if isinstance(cr.get("log"), dict) else None
    if isinstance(es, dict) and es.get("method") is not None:
        chk["dflash_engine_spec"] = es.get("method") == "dflash" and int(es.get("num_spec_tokens") or 0) == int(k)
    ok = all(v is True for v in chk.values())
    return dict(cr, ok=ok, verdict="PASS" if ok else "INCONCLUSIVE_CONFIG_MISMATCH", checks=chk, e29_dflash=True,
                mtp_checks_dropped=[a for a in ("speculative_config", "engine_spec") if a in (cr.get("checks") or {})])


def spec_counters(rows):
    """{drafts, draft_tokens, accepted} from parsed prom rows (objects with .name / .value), summed over labels."""
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0}
    for r in rows or ():
        n = getattr(r, "name", None)
        if n == SD_DRAFTS:
            out["drafts"] += float(r.value)
        elif n == SD_DRAFT_TOK:
            out["draft_tokens"] += float(r.value)
        elif n == SD_ACC:
            out["accepted"] += float(r.value)
    return out


def canary(c0, c1, k):
    """The zero-accept canary (nbdf_lib smoke rule): >= 200 draft tokens and accepted / drafted < 2 % -> hit (a structural
    port fault: exactly 0 % upstream #61072). AL = 1 + accepted / drafts."""
    c0, c1 = c0 or {}, c1 or {}
    dt = float(c1.get("draft_tokens", 0)) - float(c0.get("draft_tokens", 0))
    da = float(c1.get("accepted", 0)) - float(c0.get("accepted", 0))
    dd = float(c1.get("drafts", 0)) - float(c0.get("drafts", 0))
    rate = da / dt if dt > 0 else None
    out = {"draft_tokens": dt, "accepted": da, "drafts": dd, "accept_rate": None if rate is None else round(rate, 4),
           "al": round(1 + da / dd, 3) if dd > 0 else None, "k": k, "enough": dt >= CANARY_MIN_DRAFTS}
    out["hit"] = bool(out["enough"] and rate is not None and rate < CANARY_MIN_ACCEPT)
    out["label"] = "DFLASH_ZERO_ACCEPT" if out["hit"] else ("OK" if out["enough"] else "TOO_FEW_DRAFTS")
    return out


def hourly_al(samples, t0=None):
    """Sampler rows {t, sd_drafts, sd_draft_tok, sd_acc} -> AL / accept rate per hour of the phase; flagged when an hour's
    AL < 0.7 x the first hour's (record only: checklist item 8)."""
    rows = [s for s in samples or [] if isinstance(s, dict) and s.get("sd_drafts") is not None and s.get("t") is not None]
    if len(rows) < 2:
        return {"hours": [], "flag": None, "rows": len(rows)}
    t0 = float(t0 if t0 is not None else rows[0]["t"])
    out, prev = [], rows[0]
    by = {}
    for s in rows:
        by.setdefault(int((float(s["t"]) - t0) // 3600), []).append(s)
    for h in sorted(by):
        a, b = prev, by[h][-1]
        dd, dt, da = (float(b["sd_drafts"]) - float(a["sd_drafts"]), float(b["sd_draft_tok"]) - float(a["sd_draft_tok"]),
                      float(b["sd_acc"]) - float(a["sd_acc"]))
        out.append({"hour": h, "drafts": dd, "draft_tokens": dt, "accepted": da, "al": round(1 + da / dd, 3) if dd > 0 else None,
                    "accept_rate": round(da / dt, 4) if dt > 0 else None})
        prev = b
    first = next((x["al"] for x in out if x["al"]), None)
    low = [x["hour"] for x in out if first and x["al"] and x["al"] < AL_DROP * first]
    return {"hours": out, "first_al": first, "flag": "AL_DROP_BELOW_0.7_FIRST_HOUR" if low else None, "low_hours": low}


def inrun_zero(samples, k):
    """The in-run canary on the 15-s sampler rows: from the first row with counters, >= 200 draft tokens and < 2 %
    accepted -> hit."""
    rows = [s for s in samples or [] if isinstance(s, dict) and s.get("sd_draft_tok") is not None]
    if len(rows) < 2:
        return {"hit": False, "label": "TOO_FEW_ROWS"}
    c0 = {"drafts": rows[0]["sd_drafts"], "draft_tokens": rows[0]["sd_draft_tok"], "accepted": rows[0]["sd_acc"]}
    c1 = {"drafts": rows[-1]["sd_drafts"], "draft_tokens": rows[-1]["sd_draft_tok"], "accepted": rows[-1]["sd_acc"]}
    return canary(c0, c1, k)


def mtp_captures_for(workers, k100_caps, registry_caps):
    """Workers 24 = E28's K100_CAPS (exactly); 28 / 32 = Merlin registry capture_sizes_for(workers, 3) (precomputed at build)."""
    return list(k100_caps) if int(workers) == 24 else list(registry_caps[str(int(workers))])


def mtp_top(tries, j, top_gib, top_kv=D_PIN0):
    """rev 2c: after DFlash is dropped the NEXT try is the E28 production top rung (MTP K3, the top pin, top_rung_seqs 24, I8
    restored by e29_drop); E28's own ladder (from its own start) follows it. Inserts that rung at j unless tries[j] is it."""
    t = tries[j] if j < len(tries) else None
    if not (t and not t.get("df") and int(t.get("kv_bytes") or 0) == int(top_kv) and abs(float(t["gib"]) - float(top_gib)) < 1e-9):
        tries.insert(j, {"gib": float(top_gib), "kv_bytes": int(top_kv), "e29_mtp_top": True})
    return j


PROBE_TOKENS = 130000          # rev 2d F2: the forced first prefill (max_model_len 131,072 - room for 8 tokens + margin)
PROBE_FREE_MIB = 512           # rev 2d F2: worker-side free floor after it (NB-S1 gate 512; min seen 577 at 256/256 + EMBED_HOST)


def worker_free_mib(rpc_res):
    """nb1_cache_info results (one per rank) -> [free MiB per rank] (torch.cuda.mem_get_info inside the worker)."""
    rs = (rpc_res or {}).get("results") or []
    return [int(r["mem"]["free_mib"]) for r in rs if isinstance(r, dict) and isinstance(r.get("mem"), dict) and r["mem"].get("free_mib") is not None]


def seam_dirty(tree, pristine):
    """rev 2d F11: the seam-touched files whose sha256 != the pre-apply snapshot (a missing file counts)."""
    import hashlib
    from pathlib import Path
    bad = []
    for rel, h in sorted((pristine or {}).items()):
        try:
            if hashlib.sha256((Path(tree) / rel).read_bytes()).hexdigest() != h:
                bad.append(rel)
        except Exception:
            bad.append(rel)
    return bad


def seam_inert(log_text):
    """rev 2d F11 = nbdf_lib.q_seam_inert: an MTP boot on a tree whose seam revert is unverified must show no DFlash2 model
    and no seam events (DFLASH_SEAM unset = the original code paths)."""
    t = log_text or ""
    archs = re.findall(r"Resolved architecture: (\w+)", t)
    ev = re.findall(r"E27_DFLASH \{", t)
    ok = DF_ARCH not in archs and not ev
    return {"ok": ok, "archs": archs[:6], "events": len(ev), "label": "SEAM_REVERT_UNVERIFIED_INERT" if ok else "SEAM_NOT_INERT"}


def effective(rt, rerun=False):
    """rev 2d F5 / F14: what this run really runs (printed at the start, in the report and the LIVE 15-min line)."""
    rt = rt or {}
    d = rt.get("direct") or {}
    df = "df" in (rt.get("on") or []) or bool(rt.get("df_live"))
    out = {"drafter": "DFlash2 D3S" if df else "MTP K3", "direct": d.get("state") or "OFF (not set up)",
           "direct_why": d.get("error") or rt.get("extras_error"), "qsa": rt.get("qsa_state") or ("REQUESTED (no DFlash boot yet)" if df else "MERLIN_256_256"),
           "embed_host": bool((rt.get("cfg") or {}).get("embed_host")) and df, "seam": (rt.get("seam_restore") or {}).get("label")}
    out["experiment_validity"] = "OK" if (rerun or out["direct"] == "ON") else "DIRECT_OFF"
    return out


def submission_check(path, tasks):
    """rev 2d F12 (record only): submission.json has every task id, one row per test input, attempt_1 / attempt_2 valid 0-9
    grids (the CELL 0 fallback rows count: they are the test inputs)."""
    from pathlib import Path
    out = {"file": str(path), "exists": False}
    try:
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        out["exists"] = True
        g = lambda x: (isinstance(x, list) and 1 <= len(x) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in x)
                       and len({len(r) for r in x}) == 1 and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in x for v in r))
        miss = [k for k in tasks if k not in s]
        bad_n = [k for k in tasks if k in s and len(s[k] or []) != len(tasks[k].get("test") or [])]
        bad_g = [k for k in tasks if k in s and not all(isinstance(r, dict) and g(r.get("attempt_1")) and g(r.get("attempt_2")) for r in s[k] or [])]
        out.update(tasks=len(tasks), rows=len(s), missing=miss[:20], n_missing=len(miss), bad_rows=bad_n[:20], bad_grids=bad_g[:20],
                   ok=not miss and not bad_n and not bad_g)
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}"[:200])
    return out


def run_summary(rt, e28_out, sched, samples, boots, profile=None):
    """rev 2e (record only): the at-a-glance checks of a run / the GPU smoke ([SMOKE SUMMARY]): DFlash engaged (seam + GRD),
    EMBED_HOST, QSA, the F2 probe + per-worker min free, ladder steps / fallbacks, tok/s per 15 min + the window, AL,
    preemptions, prefix hit, A / B / C / DIRECT counts, Duck-C late routes, attempt_2 from DIRECT, submission check, eval
    score, host cgroup peak."""
    rt, e28o = rt or {}, e28_out or {}
    e28 = (sched or {}).get("e28") or {}
    d = e28.get("direct") or {}
    tries = [t for b in boots or [] if isinstance(b, dict) for t in (b.get("tries") or []) if isinstance(t, dict)]
    booted = next((t for t in reversed(tries) if t.get("phases_started")), None) or {}
    rb = booted.get("e29_df_readback") or {}
    probes = [{"try": t.get("name"), "status": t.get("status"), "min_free_mib": min((t.get("e29_probe") or {}).get("workers_free_mib") or [0]) or None,
               "workers_free_mib": (t.get("e29_probe") or {}).get("workers_free_mib"), "sec": (t.get("e29_probe") or {}).get("sec")}
              for t in tries if t.get("e29_probe")]
    rows = [s for s in samples or [] if isinstance(s, dict) and s.get("gen") is not None and s.get("t") is not None]
    per15, al = [], None
    if len(rows) >= 2:
        t0, prev = float(rows[0]["t"]), rows[0]
        by = {}
        for s in rows:
            by.setdefault(int((float(s["t"]) - t0) // 900), []).append(s)
        for k in sorted(by):
            b = by[k][-1]
            dt = float(b["t"]) - float(prev["t"])
            per15.append({"min": 15 * k, "tok_s": round((float(b["gen"]) - float(prev["gen"])) / dt, 1) if dt > 0 else None})
            prev = b
        sd = [s for s in rows if s.get("sd_drafts") is not None]
        if len(sd) >= 2 and float(sd[-1]["sd_drafts"]) > float(sd[0]["sd_drafts"]):
            al = round(1 + (float(sd[-1]["sd_acc"]) - float(sd[0]["sd_acc"])) / (float(sd[-1]["sd_drafts"]) - float(sd[0]["sd_drafts"])), 3)
    w = (rows[-1]["t"] - rows[0]["t"]) if len(rows) >= 2 else 0
    dtasks = ((rt.get("direct") or {}).get("tasks") or {})
    slots = e28.get("slots") or {}
    eng, sc = e28o.get("engine") or {}, ((e28o.get("score") or {}).get("generated") or {})
    return {"profile": profile, "effective": effective(rt, profile == "RERUN_PROD"),
            "dflash_engaged": bool(booted.get("e29_df")) and rb.get("label") == "ENGAGED" and rb.get("grd") is True,
            "booted_try": booted.get("name"), "readback": rb.get("label"), "grd": rb.get("grd"),
            "embed_host": (booted.get("server_env_levers") or {}).get("DFLASH_DRAFT_EMBED_HOST") == "1", "qsa": rt.get("qsa_state"),
            "probe": probes, "tries": [[t.get("name"), t.get("status")] for t in tries], "dropped": rt.get("dropped"),
            "relaunches": rt.get("relaunch_plans"), "gpu_device_fault": rt.get("gpu_device_fault"), "seam": (rt.get("seam_restore") or {}).get("label"),
            "tok_s_per_15min": per15, "tok_s_window": round((rows[-1]["gen"] - rows[0]["gen"]) / w, 1) if w > 0 else eng.get("solver_tok_s"),
            "solver_tok_s": eng.get("solver_tok_s"), "al": al, "preemptions": eng.get("preemptions"), "prefix_hit_pct": eng.get("prefix_hit_pct"),
            "started": e28.get("started"), "done": e28.get("done"),
            "direct": {k: d.get(k) for k in ("started", "records", "no_record", "errors", "duck_c", "duck_c_why")},
            "direct_modes": {m: sum(1 for x in dtasks.values() if x.get("mode") == m) for m in {x.get("mode") for x in dtasks.values()}},
            "attempt_2_from_direct": sum(1 for v in slots.values() if isinstance(v, list) for x in v if isinstance(x, dict) and x.get("attempt_2") == "D"),
            "submission_ok": (rt.get("submission_check") or {}).get("ok"),
            "eval": {k: sc.get(k) for k in ("eval_query_hits", "eval_query_total", "eval_task_norm_pct", "tasks_solved", "tasks_scored")},
            "host_peak_mib": rt.get("host_peak_mib"), "host_max_mib": rt.get("host_max_mib")}


def summary_text(s):
    """The [SMOKE SUMMARY] block lines (from run_summary)."""
    s = s or {}
    pr = s.get("probe") or []
    return "\n".join([
        f"  algorithm: {json.dumps(s.get('effective'))}",
        f"  DFlash engaged {s.get('dflash_engaged')} (try {s.get('booted_try')}, read-back {s.get('readback')}, GRD {s.get('grd')}) | EMBED_HOST "
        f"{s.get('embed_host')} | QSA {s.get('qsa')} | seam {s.get('seam')}",
        f"  F2 probe: " + "; ".join(f"{p['try']} {p['status']} min free {p['min_free_mib']} MiB {p['workers_free_mib']}" for p in pr),
        f"  tries: {s.get('tries')} | dropped {s.get('dropped')} | relaunches {s.get('relaunches')} | GPU device fault {s.get('gpu_device_fault')}",
        f"  tok/s window {s.get('tok_s_window')} (solver {s.get('solver_tok_s')}) | per 15 min {[x['tok_s'] for x in s.get('tok_s_per_15min') or []]} | "
        f"AL {s.get('al')} | preemptions {s.get('preemptions')} | prefix hit {s.get('prefix_hit_pct')} %",
        f"  started {s.get('started')} | done {s.get('done')}",
        f"  DIRECT {s.get('direct')} | modes {s.get('direct_modes')} | attempt_2 from DIRECT {s.get('attempt_2_from_direct')}",
        f"  submission ok {s.get('submission_ok')} | eval {s.get('eval')} | host cgroup peak {s.get('host_peak_mib')} / {s.get('host_max_mib')} MiB"])


QSA_KEYS = ("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "VLLM_E2_QSA_LOGITS_GRANULE_MB")   # Merlin launch.py:419-420 sets both 256


def qsa_cfg(cfg):
    """rev 2c: the CELL 1 switch E29_QSA {'max_logits_mb', 'granule_mb'} (the QSA indexer prefill-logits workspace, DFlash
    tries only). A bad value = Merlin's 256 / 256 (recorded 'invalid'), never a stop."""
    try:
        c = dict(cfg or {})
        assert set(c) == {"max_logits_mb", "granule_mb"}, sorted(c)
        assert isinstance(c["max_logits_mb"], int) and not isinstance(c["max_logits_mb"], bool) and c["max_logits_mb"] > 0
        assert isinstance(c["granule_mb"], int) and not isinstance(c["granule_mb"], bool) and c["granule_mb"] >= 0
        return c
    except Exception as e:
        return {"invalid": f"{type(e).__name__}: {e}"[:200], "given": cfg}


def qsa_env(cfg):
    c = cfg or {}
    return {} if "invalid" in c or not c else {QSA_KEYS[0]: str(int(c["max_logits_mb"])), QSA_KEYS[1]: str(int(c["granule_mb"]))}


def proc_env_readback(root_pid, keys, proc="/proc"):
    """The spawn env of every vLLM worker under root_pid (comm 'VLLM::Worker*'), from /proc/<pid>/environ: what the worker
    processes really see after Merlin's build(). [{pid, comm, env}] (read-only; [] when /proc is not readable)."""
    import os
    try:
        kids = {}
        for d in os.listdir(proc):
            if d.isdigit():
                try:
                    with open(f"{proc}/{d}/stat", encoding="utf-8", errors="replace") as f:
                        st = f.read()
                    kids.setdefault(int(st[st.rindex(")") + 2:].split()[1]), []).append(int(d))
                except Exception:
                    pass
        todo, desc = [int(root_pid)], []
        while todo:
            p = todo.pop()
            for c in kids.get(p, []):
                desc.append(c)
                todo.append(c)
        out = []
        for p in sorted(desc):
            try:
                with open(f"{proc}/{p}/comm", encoding="utf-8", errors="replace") as f:
                    comm = f.read().strip()
                if not comm.startswith("VLLM::Worker"):
                    continue
                with open(f"{proc}/{p}/environ", "rb") as f:
                    env = dict(x.split(b"=", 1) for x in f.read().split(b"\0") if b"=" in x)
                out.append({"pid": p, "comm": comm, "env": {k: env.get(k.encode(), b"").decode() or None for k in keys}})
            except Exception:
                pass
        return out
    except Exception:
        return []


def qsa_readback(workers, want, log_text="", n_workers=4):
    """Record only (rev 2c): every worker's QSA env == the switch's values (Merlin default 256 when the switch is off), the
    granule Merlin logged at the first prefill, and the backing bound: _prefill_backing_numel rounds the chunk need up to the
    granule INSIDE the max_logits cap (rows_per_chunk = cap // (width x 4)), so a backing <= max_logits_mb (one row wider than
    the cap = the only exception, qsa_indexer.py)."""
    w = {k: str(v) for k, v in (want or {}).items()}
    got = [x.get("env") or {} for x in workers or []]
    gl = sorted({int(m) for m in re.findall(r"QSA prefill logits backing: granule (\d+) MiB", log_text or "")})
    ok = len(got) == n_workers and all(all(g.get(k) == v for k, v in w.items()) for g in got)
    return {"workers": len(got), "env_ok": ok, "env": got[:n_workers], "want": w, "granule_logged_mib": gl,
            "backing_bound_mib": int(w.get(QSA_KEYS[0]) or 256),
            "label": ("QSA_VERIFIED" if ok else "UNVERIFIED_QSA" if len(got) < n_workers else "QSA_ENV_MISMATCH")}   # rev 2d F3 / F14


DIRECT2_KEYS = ("on", "think", "answer", "temperature", "drain_min", "phase")   # the CELL 1 switch E29_DIRECT2 (user 2026-11-02)
DIRECT2_PHASES = ("C1", "C")   # rev 2f: "C1" = DIRECT only on train-fail C items (default), "C" = every C item (rev 2e)
THINK_FLOOR = 60000                                                    # user rule: thinking >= 60k (approval needed below)


def direct2_cfg(cfg, harness_json=None):
    """The DIRECT switch (a plain literal at the top of CELL 1) -> the checked dict. A bad value never stops the run: it turns
    DIRECT off (recorded as 'invalid'). harness_json (kits only: E11_HARNESS=1 + E29_DIRECT2_JSON) overrides keys; the
    60k thinking floor applies to the switch itself, not to a kit override."""
    try:
        c = dict(cfg or {})
        if harness_json:
            c.update(json.loads(harness_json))
        c.setdefault("phase", "C1")   # rev 2f: absent = the C1 default
        assert set(c) == set(DIRECT2_KEYS), sorted(set(c) ^ set(DIRECT2_KEYS))
        assert isinstance(c["on"], bool), "on must be True / False"
        for k in ("think", "answer", "drain_min"):
            assert isinstance(c[k], int) and not isinstance(c[k], bool) and c[k] > 0, (k, c[k])
        assert isinstance(c["temperature"], (int, float)) and 0 <= float(c["temperature"]) <= 2, c["temperature"]
        assert c["phase"] in DIRECT2_PHASES, ("phase must be one of", DIRECT2_PHASES, c["phase"])
        if not harness_json:
            assert c["think"] >= THINK_FLOOR, ("thinking floor 60,000 (user rule)", c["think"])
        c["harness_override"] = bool(harness_json)
        return c
    except Exception as e:
        return {"on": False, "invalid": f"{type(e).__name__}: {e}"[:300], "given": cfg}


def summary(rt):
    """The E29 record for the report (every flag, what stayed on, every drop with its reason; 1.1: + the DIRECT branch and
    the in-run relaunch plans)."""
    rt = rt or {}
    return {k: rt.get(k) for k in ("cfg", "on", "dropped", "df", "vocab", "idx", "gdn", "supply", "tries", "canary", "inrun",
                                   "hourly_al", "seam", "drafter", "errors", "relaunch_plans", "direct", "qsa_state", "qsa_off",
                                   "seam_restore", "gpu_device_fault", "mtp_relaunch_gates", "extras_error", "effective_algorithm",
                                   "submission_check")}
