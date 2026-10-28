# ==================================================================== CELL E19-J - verdict rules (pure functions, no GPU, no globals)
# E19: does vLLM 0.19's MTP + prefix caching (and fp8 KV) silently corrupt Qwen3.8-27B greedy outputs on the S1 stack?
# This cell holds ONLY pure functions over saved rows, so the same code judges the rows inside the notebook and offline
# (src/e19_compare.py imports this file). Rows are dicts saved one per line in q27x/e19/e19_rows.jsonl.
# Rules:
#   - rows are compared by integer token id; the first divergence is a LARGE-MARGIN FLIP when a known top-1 minus top-2
#     logprob margin at that position (either run) is > 1.0 nat (the Experiment 14 VMS / EMS gate); a divergence with no
#     margin > 1 and a missing margin is missing evidence (never a pass); low-margin divergences are tolerated (greedy +
#     MTP is not bit-exact across batch compositions);
#   - a corrupt row = empty output, degenerate text (32+ identical characters, < 8 distinct characters over 200, < 10 %
#     letters / digits; ARC grids excluded), a run of 64+ identical token ids, 8+ CJK characters, or a greedy token that
#     is not the top-1 of its own logprobs;
#   - the main discriminator is WITHIN the arm: serial cold vs cold repeat (the arm's own noise), serial cold vs warm
#     prefix hit, serial vs churn-concurrent (batch reordering), natural vs continuation over cached generated tokens,
#     plus MTP acceptance per /metrics window (collapse to ~0);
#   - the churn check is calibrated by the noise of arms A (MTP off, prefix off) and B (MTP on, prefix off); the token
#     comparison against arm A is informational only;
#   - an arm with planned rows not run because of the time budget is NOT_TESTED_BUDGET, never PASS.
import json as _json
import math as _math
import re as _re

E19_FLIP_NAT = 1.0
E19_MARGIN_COVERAGE = 0.95        # rev 2 (audit B4): share of a row's tokens that must carry a finite top-1 - top-2 margin
E19_SAME_BUDGET = ("aa_serial", "cache_serial")   # + every cont_<stage>: same max_tokens on both sides (rev 2, audit B2)
E19_BUDGET_STATUSES = ("not_tested_budget", "missing")
E19_STATUS_ORDER = ["FAIL", "LAUNCH_FAILED", "INCOMPLETE", "NOT_TESTED_BUDGET", "INCONCLUSIVE", "PASS"]
_CJK_RANGES = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF))
_CJK = _re.compile("[" + "".join(chr(a) + "-" + chr(b) for a, b in _CJK_RANGES) + "]")


def e19_grid_like(s):
    """VMS rev 2g: an ARC grid by structure (digits / brackets / commas / whitespace, every row <= 30 cells)."""
    s = (s or "").strip()
    return bool(_re.fullmatch(r"[0-9,\[\]\s]+", s)) and len(_re.findall(r"\d", s)) >= 2 \
        and all(len(_re.findall(r"\d", r_)) <= 30 for r_ in _re.split(r"\]|\n", s))


def e19_text_bad(x):
    """VMS rev 2e-c: empty, or (not grid-like) a 32-character run of one character, < 8 distinct characters over 200,
    or < 10 % letters / digits."""
    s = (x or "").strip()
    if not s:
        return True
    if e19_grid_like(s):
        return False
    if _re.search(r"(.)\1{31,}", s):
        return True
    return len(s) >= 200 and (len(set(s)) < 8 or sum(ch.isalnum() for ch in s) / len(s) < 0.10)


def e19_token_run(ids, n=64):
    run, prev = 0, None
    for t in ids or []:
        run = run + 1 if t == prev else 1
        prev = t
        if run >= n:
            return True
    return False


def e19_row_flags(r):
    """Corruption flags of one completed row (text = the decoded output, saved by the client)."""
    toks, text = list(r.get("tokens") or []), r.get("text") or ""
    f = {"empty": not toks or not text.strip(),
         "degenerate": bool(toks) and e19_text_bad(text),
         "token_run": e19_token_run(toks),
         "cjk": len(_CJK.findall(text)) >= 8,
         "argmax_mismatch": len(r.get("top1_mismatch") or [])}
    f["corrupt"] = bool(f["empty"] or f["degenerate"] or f["token_run"] or f["cjk"] or f["argmax_mismatch"])
    return f


def e19_margin_coverage(r):
    """rev 2 (audit B4): share of the row's tokens with a finite top-1 - top-2 logprob margin."""
    toks, mg = list(r.get("tokens") or []), list(r.get("margins") or [])
    if not toks or len(mg) != len(toks):
        return 0.0
    return sum(1 for x in mg if isinstance(x, (int, float)) and not isinstance(x, bool) and _math.isfinite(x)) / len(toks)


def e19_key(r):
    return (r.get("stage"), r.get("pass"), r.get("idx"))


def e19_ok(r):
    return r is not None and r.get("status") == "ok"


def e19_index(rows, arm=None):
    """Last row per key (a row is written once; the index is defensive against duplicates)."""
    out = {}
    for r in rows:
        if arm is None or r.get("arm") == arm:
            out[e19_key(r)] = r
    return out


def e19_first_div(ta, ma, tb, mb):
    """Compare two token lists (+ margins aligned to them). Returns one comparison record."""
    ta, tb = list(ta or []), list(tb or [])
    n = min(len(ta), len(tb))
    k = next((j for j in range(n) if ta[j] != tb[j]), None)
    if k is None:
        # identical over the common length; a length difference alone (one stopped earlier) is not a token flip
        return {"identical": len(ta) == len(tb), "first_divergence": None if len(ta) == len(tb) else n,
                "length_only": len(ta) != len(tb), "large_margin_flip": False, "missing_margin": False,
                "margin_a": None, "margin_b": None, "compared_len": n}
    ma_k = (ma or [None] * (k + 1))[k] if k < len(ma or []) else None
    mb_k = (mb or [None] * (k + 1))[k] if k < len(mb or []) else None
    known = [m for m in (ma_k, mb_k) if isinstance(m, (int, float))]
    big = any(m > E19_FLIP_NAT for m in known)
    return {"identical": False, "first_divergence": k, "length_only": False, "large_margin_flip": big,
            "missing_margin": (not big) and len(known) < 2, "margin_a": ma_k, "margin_b": mb_k, "compared_len": n}


def e19_pair_view(r, part):
    """Token / margin view of a row for a comparison. part 'all' = the whole output; 'tail' = the natural row after its
    first g_pre tokens (what the continuation request must reproduce)."""
    toks, mg = list(r.get("tokens") or []), list(r.get("margins") or [])
    if part == "tail":
        g = int(r.get("g_pre") or 0)
        return toks[g:], mg[g:]
    return toks, mg


def e19_compare(pairs):
    """pairs: list of (row_a, part_a, row_b, part_b). Missing / failed rows are counted as not compared."""
    per, not_compared, nc_budget = [], 0, 0
    for ra, pa, rb, pb in pairs:
        if not (e19_ok(ra) and e19_ok(rb)):
            not_compared += 1
            if any(r_ is None or r_.get("status") in E19_BUDGET_STATUSES for r_ in (ra, rb)):
                nc_budget += 1
            continue
        ta, ma = e19_pair_view(ra, pa)
        tb, mb = e19_pair_view(rb, pb)
        if not ta or not tb:
            not_compared += 1
            continue
        c = e19_first_div(ta, ma, tb, mb)
        c["idx"] = ra.get("idx")
        per.append(c)
    div = [c for c in per if not c["identical"] and not c["length_only"]]
    fd = [c["first_divergence"] for c in div]
    return {"planned": len(pairs), "compared": len(per), "not_compared": not_compared,
            "not_compared_budget": nc_budget, "not_compared_other": not_compared - nc_budget,
            "identical": sum(1 for c in per if c["identical"]), "length_only": sum(1 for c in per if c["length_only"]),
            "differ": len(div), "large_margin_flips": sum(1 for c in div if c["large_margin_flip"]),
            "missing_margins": sum(1 for c in div if c["missing_margin"]),
            "first_divergence": fd, "flip_margins": [[c["margin_a"], c["margin_b"]] for c in div],
            "per": per}


def e19_churn_ref(ix, pc, r):
    """The serial reference of a churn row: the same prompt alone in the same arm, from the cache when the churn row was
    served from the cache (prefix-ON arm), else cold."""
    if pc and (r.get("cached_tokens") or 0) > 0:
        return ix.get(("serial", "warm", r.get("idx")))
    return ix.get(("serial", "cold", r.get("idx")))


def e19_arm_pairs(ix, meta):
    """The within-arm comparisons (the main discriminator).
    aa_serial    - serial cold vs serial cold repeat (prefix-ON: cache reset before each; prefix-OFF: the warm pass is a
                   second cold run, nothing is cached): the arm's own numeric noise;
    cache_serial - serial cold vs serial warm (prefix-cache hit), prefix-ON arms only;
    churn        - every churn row (12-16 concurrent requests, staggered lengths and arrivals: the batch reorders
                   continuously) vs the same prompt run alone (serial) in the same arm, over their common length;
    cont_<stage> - the natural continuation's tail vs the continuation request over the server's own cached tokens."""
    pc = bool(meta.get("prefix_caching"))
    ser = list(meta.get("serial_idx") or [])
    g = lambda st, ps, i: ix.get((st, ps, i))
    out = {"aa_serial": [(g("serial", "cold", i), "all", g("serial", "cold2" if pc else "warm", i), "all") for i in ser]}
    if pc:
        out["cache_serial"] = [(g("serial", "cold", i), "all", g("serial", "warm", i), "all") for i in ser]
    out["churn"] = []
    for idx, rep, _m in meta.get("churn_jobs") or []:
        r = g("churn", f"j{rep}", idx)
        out["churn"].append((e19_churn_ref(ix, pc, r) if r else None, "all", r, "all"))
    for st in meta.get("long_stages") or []:
        out["cont_" + st["name"]] = [(g(st["name"], "nat", i), "tail", g(st["name"], "cont", i), "all") for i in st["idx"]]
    return out


def e19_planned_keys(meta):
    pc = bool(meta.get("prefix_caching"))
    keys = []
    for i in meta.get("serial_idx") or []:
        keys += [("serial", "cold", i), ("serial", "warm", i)] + ([("serial", "cold2", i)] if pc else [])
    keys += [("churn", f"j{rep}", idx) for idx, rep, _m in meta.get("churn_jobs") or []]
    for st in meta.get("long_stages") or []:
        for i in st["idx"]:
            keys += [(st["name"], "nat", i), (st["name"], "cont", i)]
    return keys


def e19_worst(*statuses):
    s = [x for x in statuses if x]
    return min(s, key=E19_STATUS_ORDER.index) if s else "PASS"


def e19_accept(meta):
    """MTP acceptance per /metrics window (accepted / draft tokens). Collapse = a window with >= 300 draft tokens and an
    acceptance below 0.10 (Qwen3.8 MTP 3 normally accepts about half of the draft tokens)."""
    w = [x for x in (meta.get("accept_windows") or []) if (x.get("draft_tokens") or 0) >= 300]
    rates = [round((x.get("accepted") or 0) / x["draft_tokens"], 3) for x in w]
    return {"windows": len(w), "min": min(rates) if rates else None,
            "median": sorted(rates)[len(rates) // 2] if rates else None, "collapsed": sum(1 for r_ in rates if r_ < 0.10)}


def e19_judge_arm(rows, meta):
    """In-arm verdict of one arm from its rows and meta. The churn comparison is NOT judged here (e19_verdicts() judges
    it against the noise of the calibration arms A / B, or strictly against the arm's own serial A/A)."""
    if meta.get("launch_failed"):
        return {"arm": meta.get("arm"), "verdict": "LAUNCH_FAILED", "reasons": [meta.get("launch_failed")], "comparisons": {},
                "corrupt_rows": [], "aa_lmf": None, "churn_lmf": None, "churn_compared": 0}
    ix = e19_index(rows, meta.get("arm"))
    planned = e19_planned_keys(meta)
    pc = bool(meta.get("prefix_caching"))
    st_count = {}
    for k in planned:
        s_ = (ix.get(k) or {}).get("status", "missing")
        st_count[s_] = st_count.get(s_, 0) + 1
    reasons = []
    flags = {k: e19_row_flags(ix[k]) for k in planned if e19_ok(ix.get(k))}
    corrupt_rows = [list(k) for k, f in flags.items() if f["corrupt"]]
    partner = {}
    for k in planned:
        st, ps, i = k
        if ps in ("warm", "cold2"):
            partner[k] = (st, "cold", i)
        elif ps == "cont":
            partner[k] = (st, "nat", i)
        elif st == "churn" and e19_ok(ix.get(k)):
            ref = e19_churn_ref(ix, pc, ix[k])
            partner[k] = e19_key(ref) if ref else None
    hit_corrupt = [list(k) for k, p in partner.items()
                   if p and k in flags and p in flags and flags[k]["corrupt"] and not flags[p]["corrupt"]]
    cached = lambda k: (ix.get(k) or {}).get("cached_tokens")
    warm_keys = [k for k in planned if k[1] in ("warm", "cont") and e19_ok(ix.get(k))]
    cold_keys = [k for k in planned if k[0] == "serial" and k[1] in ("cold", "cold2") and e19_ok(ix.get(k))]
    churn_keys = [k for k in planned if k[0] == "churn" and e19_ok(ix.get(k))]
    cache_ev = {"warm_rows": len(warm_keys), "warm_hits": sum(1 for k in warm_keys if (cached(k) or 0) > 0),
                "cold_rows": len(cold_keys), "cold_hits": sum(1 for k in cold_keys if (cached(k) or 0) > 0),
                "churn_rows": len(churn_keys), "churn_hits": sum(1 for k in churn_keys if (cached(k) or 0) > 0),
                "cont_generated_in_hit": [((cached(k) or 0) - int((ix.get(k) or {}).get("pf_len") or 0))
                                          for k in planned if k[1] == "cont" and e19_ok(ix.get(k))],
                "usage_seen": sum(1 for k in planned if (ix.get(k) or {}).get("usage_seen"))}
    comps = {name: e19_compare(pairs) for name, pairs in e19_arm_pairs(ix, meta).items()}
    aa_lmf = comps["aa_serial"]["large_margin_flips"]
    acc = e19_accept(meta)
    status = "PASS"
    if hit_corrupt:
        status = "FAIL"
        reasons.append(f"corrupt rows (cache hit / churn / continuation) whose clean partner is not corrupt: {hit_corrupt[:8]}")
    for nm in comps:
        if nm in ("aa_serial", "churn"):
            continue
        if comps[nm]["large_margin_flips"] > aa_lmf:
            status = "FAIL"
            reasons.append(f"{nm}: {comps[nm]['large_margin_flips']} large-margin flips > the arm's serial A/A {aa_lmf}")
    if meta.get("spec") and acc["collapsed"]:
        status = "FAIL"
        reasons.append(f"MTP acceptance collapsed in {acc['collapsed']} of {acc['windows']} /metrics windows (min {acc['min']})")
    # rev 2 (audit B1): corrupt outputs without a clean partner are never a clean PASS
    if status != "FAIL" and corrupt_rows:
        status = e19_worst(status, "INCONCLUSIVE")
        reasons.append(f"{len(corrupt_rows)} anomalous outputs with no clean partner to establish relative corruption: "
                       f"{corrupt_rows[:8]}")
    if status != "FAIL":
        # rev 2 (audit B2): an unexplained length mismatch between same-budget rows (one stopped early) is not a PASS
        for nm, c in comps.items():
            if (nm in E19_SAME_BUDGET or nm.startswith("cont_")) and c.get("length_only", 0) > 0:
                status = e19_worst(status, "INCONCLUSIVE")
                reasons.append(f"{nm}: {c['length_only']} unexplained output-length mismatches (same max_tokens)")
        # rev 2 (audit B3): rows with an unvalidated status (e.g. na_short) and pairs not compared for a non-budget reason
        unexpected = {k_: v for k_, v in st_count.items() if k_ not in ("ok", "error", "server_dead") + E19_BUDGET_STATUSES}
        if unexpected:
            status = e19_worst(status, "INCOMPLETE")
            reasons.append(f"unvalidated row statuses: {unexpected}")
        nco = {nm: c["not_compared_other"] for nm, c in comps.items() if c.get("not_compared_other")}
        if nco:
            status = e19_worst(status, "INCOMPLETE")
            reasons.append(f"pairs not compared (non-budget): {nco}")
        # rev 2 (audit B4): every completed row needs logprob-margin evidence
        thin = [list(k) for k in planned if e19_ok(ix.get(k)) and e19_margin_coverage(ix[k]) < E19_MARGIN_COVERAGE]
        if thin:
            status = e19_worst(status, "INCOMPLETE")
            reasons.append(f"insufficient logprob-margin evidence (< {E19_MARGIN_COVERAGE:.0%} of tokens) in {len(thin)} rows")
        # rev 2 (audit 8): the cold controls need a cache reset that answered 200 and usage records that report cache use
        if pc:
            bad_reset = [list(k) for k in planned if k[0] == "serial" and k[1] in ("cold", "cold2") and e19_ok(ix.get(k))
                         and (ix[k].get("reset") not in (200, None) or ix[k].get("reset") is None)]
            if bad_reset:
                status = e19_worst(status, "INCONCLUSIVE")
                reasons.append(f"cache reset not confirmed (HTTP 200) before {len(bad_reset)} cold rows")
        if st_count.get("ok") and not cache_ev["usage_seen"]:
            status = e19_worst(status, "INCONCLUSIVE")
            reasons.append("no row carried a usage record (cached tokens unknown): cache evidence missing")
    if status != "FAIL":
        if meta.get("server_died"):
            status = e19_worst(status, "INCOMPLETE")
            reasons.append("server died during the arm: " + str(meta.get("server_died"))[:300])
        n_err = sum(v for k_, v in st_count.items() if k_ in ("error", "server_dead"))
        if n_err:
            status = e19_worst(status, "INCOMPLETE")
            reasons.append(f"{n_err} planned rows errored / lost to the server")
        if any(c["missing_margins"] for c in comps.values()):
            status = e19_worst(status, "INCOMPLETE")
            reasons.append("divergences without a usable logprob margin (missing evidence)")
        n_bud = st_count.get("not_tested_budget", 0) + st_count.get("missing", 0)
        if n_bud:
            status = e19_worst(status, "NOT_TESTED_BUDGET")
            reasons.append(f"{n_bud} of {len(planned)} planned rows not run (time budget)")
        if meta.get("spec") and not acc["windows"] and st_count.get("ok"):
            status = e19_worst(status, "INCONCLUSIVE")
            reasons.append("MTP arm without a single /metrics acceptance window (spec-decode counters missing)")
        if pc and cache_ev["warm_rows"] and cache_ev["warm_hits"] == 0:
            status = e19_worst(status, "INCONCLUSIVE")
            reasons.append("prefix caching ON but no warm / continuation row reported cached tokens: the cache was not exercised")
        if pc and cache_ev["cold_hits"]:
            status = e19_worst(status, "INCONCLUSIVE")
            reasons.append(f"{cache_ev['cold_hits']} serial cold rows were served from the cache (reset not effective)")
        if not pc and cache_ev["warm_hits"] + cache_ev["cold_hits"] + cache_ev["churn_hits"]:
            status = e19_worst(status, "INCONCLUSIVE")
            reasons.append("prefix caching OFF but rows reported cached tokens")
    gen_reuse = any(x > 0 for x in cache_ev["cont_generated_in_hit"])
    cache_ev["trailing_mamba_cache_tested"] = gen_reuse      # rev 2 (audit 7): NOT_EXERCISED when False
    if pc and not gen_reuse:
        reasons.append("trailing Mamba block over generated tokens NOT_EXERCISED (no continuation hit reached into the "
                       "server's generated tokens): no safety claim for that path")
    return {"arm": meta.get("arm"), "verdict": status, "reasons": reasons, "planned_rows": len(planned),
            "row_status": st_count, "corrupt_rows": corrupt_rows, "cache_hit_corrupt": hit_corrupt, "cache": cache_ev,
            "accept": acc, "aa_lmf": aa_lmf, "churn_lmf": comps["churn"]["large_margin_flips"],
            "churn_compared": comps["churn"]["compared"],
            "comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "per"} for k, v in comps.items()}}


def e19_vs_ref(rows_x, meta_x, rows_ref, meta_ref):
    """SECONDARY, informational: arm X vs arm A token by token on rows with identical prompt ids. MTP on vs off can
    legitimately differ at low-margin points; this never decides a verdict."""
    ix, ir = e19_index(rows_x, meta_x.get("arm")), e19_index(rows_ref, meta_ref.get("arm"))
    groups, mismatch = {}, 0
    for k, r in ix.items():
        rr = ir.get(k)
        if rr is None or not (e19_ok(r) and e19_ok(rr)):
            continue
        if r.get("prompt_sha") != rr.get("prompt_sha"):
            mismatch += 1
            continue
        groups.setdefault(k[0] + "/" + ("churn" if k[0] == "churn" else k[1]), []).append((rr, "all", r, "all"))
    comps = {g: e19_compare(p) for g, p in sorted(groups.items())}
    return {"prompt_mismatch_rows": mismatch,
            "comparisons": {k: {kk: vv for kk, vv in v.items() if kk not in ("per", "flip_margins")} for k, v in comps.items()}}


E19_CALIBRATORS = ("A", "B")      # A = MTP off + prefix off (pure batch noise); B = MTP on + prefix off (MTP batch noise)


def e19_verdicts(rows, metas):
    """Every arm: the in-arm verdict, then the churn and corrupt-row checks calibrated by the noise of arms A / B.
    Calibration: A qualifies when its in-arm verdict is not FAIL / LAUNCH_FAILED, it compared churn rows and its churn
    flips <= its own serial A/A flips; B qualifies likewise with its churn flips <= A's (or, without A, <= its own A/A).
    An arm's churn ceiling = the max churn large-margin flips over the OTHER qualified calibrators (corrupt rows
    likewise); with none, the arm is judged strictly against its own serial A/A ('uncalibrated').
    Final = the worse of the in-arm and the churn verdicts. The token comparison against A is informational."""
    J = {n: e19_judge_arm(rows, m) for n, m in metas.items()}
    good = lambda n: n in J and J[n]["verdict"] == "PASS" and (J[n].get("churn_compared") or 0) > 0   # rev 2
    qual = {}
    if good("A") and J["A"]["churn_lmf"] <= J["A"]["aa_lmf"]:
        qual["A"] = J["A"]
    if good("B") and J["B"]["churn_lmf"] <= (J["A"]["churn_lmf"] if "A" in qual else J["B"]["aa_lmf"]):
        qual["B"] = J["B"]
    out = {}
    for n, j in J.items():
        ch = {"calibrators": sorted(q for q in qual if q != n), "reasons": []}
        if j["verdict"] == "LAUNCH_FAILED" or not j.get("churn_compared"):
            ch["verdict"] = None          # nothing compared: the in-arm status (launch / budget / incomplete) speaks
        else:
            others = [q for q in qual if q != n]
            cf = max((qual[q]["churn_lmf"] for q in others), default=None)
            cc = max((len(qual[q]["corrupt_rows"]) for q in others), default=None)
            lim = cf if cf is not None else j["aa_lmf"]
            ch.update(mode="calibrated" if cf is not None else "uncalibrated", ceiling_flips=lim, ceiling_corrupt=cc,
                      verdict="PASS")
            if j["churn_lmf"] > lim:
                ch["verdict"] = "FAIL"
                ch["reasons"].append(f"churn: {j['churn_lmf']} large-margin flips vs serial > {ch['mode']} ceiling {lim}")
            if cc is not None and len(j["corrupt_rows"]) > cc:
                ch["verdict"] = "FAIL"
                ch["reasons"].append(f"{len(j['corrupt_rows'])} corrupt rows > the calibrators' {cc}")
        vs = (e19_vs_ref(rows, metas[n], rows, metas["A"]) if "A" in metas and n != "A"
              and J["A"]["verdict"] != "LAUNCH_FAILED" else None)
        out[n] = {"final": e19_worst(j["verdict"], ch["verdict"]), "in_arm": j, "churn": ch, "vs_A_info": vs}
    return out


def e19_fmt(c, k):
    return f"{c[k]['identical']}/{c[k]['compared']} ({c[k]['large_margin_flips']})" if k in c else "-"


def e19_report_lines(V, arms_cfg, state_arms, title):
    L = [title, "",
         "| arm | what | final | in-arm | churn (mode) | rows ok / planned | serial A/A ident/cmp (flips) | "
         "cache serial | churn vs serial | cont8k | cont30k | corrupt rows | MTP accept min / median | cache warm hits | "
         "launch s | KV tokens | block |", "|" + "---|" * 17]
    for name, v in V.items():
        j, ch, cfg = v["in_arm"], v["churn"], arms_cfg.get(name, {})
        c, a = j.get("comparisons") or {}, state_arms.get(name, {})
        acc, cache = j.get("accept") or {}, j.get("cache") or {}
        L.append(f"| {name} | {cfg.get('what', '')} | **{v['final']}** | {j['verdict']} | {ch.get('verdict') or '-'} "
                 f"({ch.get('mode', '-')}) | {(j.get('row_status') or {}).get('ok', 0)}/{j.get('planned_rows', '-')} | "
                 f"{e19_fmt(c, 'aa_serial')} | {e19_fmt(c, 'cache_serial')} | {e19_fmt(c, 'churn')} | "
                 f"{e19_fmt(c, 'cont_long8k')} | {e19_fmt(c, 'cont_long30k')} | {len(j.get('corrupt_rows') or [])} | "
                 f"{acc.get('min', '-')} / {acc.get('median', '-')} | {cache.get('warm_hits', '-')}/{cache.get('warm_rows', '-')} | "
                 f"{(a.get('launch') or {}).get('sec', '-')} | {(a.get('log') or {}).get('kv_tokens', '-')} | "
                 f"{(a.get('log') or {}).get('block_size', '-')} |")
    L += ["", "## Reasons", ""]
    for name, v in V.items():
        rs = (v["in_arm"].get("reasons") or []) + (v["churn"].get("reasons") or [])
        L.append(f"- **{name}** ({v['final']}): " + ("; ".join(rs) or "none"))
    L += ["", "## Secondary: token comparison against arm A (informational; MTP on vs off may differ at low margins)", ""]
    for name, v in V.items():
        if v.get("vs_A_info"):
            L.append(f"- {name}: " + "; ".join(f"{g} {e19_fmt(v['vs_A_info']['comparisons'], g)}"
                                              for g in sorted(v["vs_A_info"]["comparisons"])))
    L += ["", "Cells: identical / compared (large-margin flips). A large-margin flip = the first token divergence where a "
          "known top-1 minus top-2 logprob margin is > 1 nat. In-arm FAIL: corrupt rows with clean partners, cache-hit or "
          "continuation flips above the arm's serial A/A, MTP acceptance collapse. Churn FAIL: flips vs serial above the "
          "calibrators' (A / B) churn flips, or above the arm's own serial A/A when no calibrator ran ('uncalibrated'). "
          "Rows not run in the budget make an arm NOT_TESTED_BUDGET, never PASS."]
    return L


def e19_load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(_json.loads(line))
                except ValueError:
                    pass          # a torn last line after a crash
    return rows
