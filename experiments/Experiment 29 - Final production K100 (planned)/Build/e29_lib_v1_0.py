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
"""
import json
import math
import re

LIB_VERSION = "e29_lib 1.0 (2026-11-01)"
DRAFTERS = ("mtp", "dflash_d3s", "dflash_d4s")
K_OF = {"dflash_d3s": 3, "dflash_d4s": 4}
LEVER_ORDER = ("df", "idx", "vocab", "gdn_fastpath", "gdn_align")      # the drop order on a pre-phase failure
GDN_ENV = {"gdn_fastpath": "VLLM_GDN_DECODE_FASTPATH", "gdn_align": "VLLM_GDN_SHARED_ALIGN_INDEX"}
VOCAB_FILES = {"32k": ("draft_vocab_k100_arc_n32768.pt", "a164d1a71d27dab600c014673c34accfdeb8695141a63be8aa4bfb4026695248"),
               "64k": ("draft_vocab_k100_arc_n65536.pt", "a915fa6d5ad711817976de5c988a3950f43e7b2fbc058b04325e75c8501cf022")}
SEAM_FILE = ("dflash_seam.py", "7356d4738962bdbab4091264861c4bfbbdcd7098d6c75b9c48db69eeb73e6937")   # = NBDF-K100's seam
IDX_ARGS = ["--attention-config", '{"indexer_kv_dtype":"fp8"}']   # = --attention-config.indexer_kv_dtype fp8 (vLLM's JSON form;
# the K-9 help gate needs a flag that `serve --help=all` lists: the dotted key is not listed; Merlin emits no --attention-config)
D_RETRY = [[7933526016, True], [7665090560, True], [7396655104, True], [6859784192, True]]   # NBDF-K100 K100_CFG d_retry
D_PIN0 = 7933526016                                                # every D try starts at the K100Q / E28 top K3 rung
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
    then 0.25-GiB rungs with EMBED_HOST (an OOM moves one step; any other failure ends DFlash -> MTP at the top rung)."""
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
    t = (log_text or "")[-400000:].lower()
    named = {"idx": ("indexer_kv_dtype", "indexer"), "vocab": ("draft_vocab", "draft vocab"),
             "gdn_fastpath": ("gdn_decode_fastpath",), "gdn_align": ("gdn_shared_align",)}
    for k in on:
        if k != "df" and any(w in t for w in named.get(k, ())):
            return k
    return on[0]


def is_oom(log_text):
    t = (log_text or "")[-400000:]
    return bool(re.search(r"CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError|No available memory for the cache blocks", t))


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


def summary(rt):
    """The E29 record for the report (every flag, what stayed on, every drop with its reason)."""
    rt = rt or {}
    return {k: rt.get(k) for k in ("cfg", "on", "dropped", "df", "vocab", "idx", "gdn", "supply", "tries", "canary", "inrun",
                                   "hourly_al", "seam", "drafter", "errors")}
