"""Analysis 2: re-prefill breakdown per Duck turn for K100Q (numeric fields only; run extract.py first).

For every later turn k (k >= 1) of an attempt, with P = prompt tokens, C = cached (prefix-hit) tokens, G = completion tokens:
  uncached U_k = P_k - C_k
  (c) old-prompt tail not hit  = P_{k-1} - C_k            (prefix unit 64 alignment; no eviction seen if 64..127)
  (a) turn k-1 generated tokens = G_{k-1}                  (the reasoning + tool call re-rendered inside prompt k)
  (b) new tool result / user content + template wrappers = P_k - P_{k-1} - G_{k-1}
  identity: U_k = (c) + (a) + (b)   (exact, by construction; valid because C_k <= P_{k-1} for every turn, checked)
First turns (k = 0) are a separate class (d): the fresh task prompt (no cross-attempt sharing: per-attempt markers).
Usage: python -I reprefill.py <data_dir> <out_dir> [analysis1_summary.json]
"""
import csv, json, math, statistics as st, sys, collections
from pathlib import Path

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
A1 = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
W0, W1 = 1793505683.4254444, 1793512283.8254445
T = W1 - W0
SERVER = {"prompt_tokens": 24149339, "prefix_hit_tokens": 20806080, "fresh_prefill_tokens": 3343259,
          "generated_tokens": 2543542, "requests_success": 762}
PROBE_PREFILL_TOK_S = 4504.6       # boot probe: 40,000-token prompt, TTFT 8.9 s, idle engine (nb5_report boots[0].probes)
PROM_PREFILL_MEAN_SEC, PROM_N = 1.68, 762


def f(x):
    if x in ("", "None", None):
        return None
    try:
        return float(x)
    except ValueError:
        return x


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "sum": round(sum(xs)), "mean": round(st.mean(xs), 1), "p10": round(pct(xs, 10), 1),
            "p50": round(pct(xs, 50), 1), "p90": round(pct(xs, 90), 1), "p99": round(pct(xs, 99), 1),
            "min": round(min(xs), 1), "max": round(max(xs), 1)} if xs else {"n": 0}


R = [{k: f(v) for k, v in r.items()} for r in csv.DictReader(open(DATA / "requests.csv", encoding="utf-8"))]
by = collections.defaultdict(list)
cut = []
for r in R:
    if r["t1"] is None:
        continue                      # 16b78196 b1 i20: CONTEXT_EXHAUSTED, no request sent
    if r["error"]:
        cut.append(r)                 # in flight at the window end: no usage returned
        continue
    by[(r["task"], int(r["branch"]))].append(r)

turns, firsts = [], []
for key, rs in by.items():
    rs.sort(key=lambda r: r["i"])
    for j, r in enumerate(rs):
        P, C, G = int(r["prompt"]), int(r["cached"]), int(r["completion"])
        if j == 0:
            firsts.append({"task": key[0], "branch": key[1], "i": 0, "P": P, "C": C, "U": P - C, "G": G})
            continue
        p = rs[j - 1]
        Pp, Gp = int(p["prompt"]), int(p["completion"])
        c_ = Pp - C
        a_ = Gp
        b_ = P - Pp - Gp
        turns.append({"task": key[0], "branch": key[1], "i": int(r["i"]), "P": P, "C": C, "U": P - C, "G": G,
                      "Pp": Pp, "Gp": Gp, "a_gen_prev": a_, "b_new_content": b_, "c_old_tail": c_,
                      "C_le_Pp": C <= Pp, "c_in_64_127": 64 <= c_ <= 127, "C_mod64": C % 64,
                      "a_share": round(a_ / (P - C), 4) if P > C else None, "kind": r["kind"], "prev_finish": p["finish"],
                      "prefix_ok": r["m_prefix_ok"], "trimmed": r["m_trimmed"], "t_rel": round(r["t0"] - W0, 1)})

# checks
checks = {"later_turns": len(turns), "first_turns": len(firsts),
          "C_le_prev_prompt_all": all(t["C_le_Pp"] for t in turns),
          "C_gt_prev_prompt": sum(1 for t in turns if not t["C_le_Pp"]),
          "old_tail_in_64_127": sum(1 for t in turns if t["c_in_64_127"]),
          "old_tail_gt_127_eviction_like": sum(1 for t in turns if t["c_old_tail"] > 127),
          "b_negative": sum(1 for t in turns if t["b_new_content"] < 0),
          "b_min": min(t["b_new_content"] for t in turns),
          "cached_multiple_of_64": sum(1 for t in turns + firsts if t["C"] % 64 == 0),
          "first_turn_cached_gt0": sum(1 for t in firsts if t["C"] > 0),
          "prefix_ok_all": all(t["prefix_ok"] in (True, "True") for t in turns),
          "trimmed_any": any(t["trimmed"] in (True, "True") for t in turns),
          "cut_requests_at_window_end": len(cut)}

tot = {"a_gen_prev": sum(t["a_gen_prev"] for t in turns), "b_new_content": sum(t["b_new_content"] for t in turns),
       "c_old_tail": sum(t["c_old_tail"] for t in turns), "d_first_turn": sum(t["U"] for t in firsts)}
U_all = sum(tot.values())
client = {"requests": len(turns) + len(firsts), "prompt": sum(t["P"] for t in turns + firsts),
          "cached": sum(t["C"] for t in turns + firsts), "uncached": U_all,
          "completion": sum(t["G"] for t in turns + firsts)}
client["hit_pct"] = round(100 * client["cached"] / client["prompt"], 2)

# the 4 requests cut at the window end: estimate their (fresh) prefill from their previous turn
cut_est = []
med_b = pct([t["b_new_content"] for t in turns], 50)
for r in cut:
    rs = by[(r["task"], int(r["branch"]))]
    prev = max(rs, key=lambda x: x["i"])
    Pp, Gp = int(prev["prompt"]), int(prev["completion"])
    cut_est.append({"task": r["task"], "branch": int(r["branch"]), "prev_prompt": Pp, "prev_gen": Gp,
                    "est_prompt": Pp + Gp + med_b, "est_fresh": round(95 + Gp + med_b)})
cut_fresh = sum(c["est_fresh"] for c in cut_est)
cut_prompt = sum(c["est_prompt"] for c in cut_est)

shares_all = {k: round(v / U_all, 4) for k, v in tot.items()}
later_U = sum(t["U"] for t in turns)
shares_later = {k: round(tot[k] / later_U, 4) for k in ("a_gen_prev", "b_new_content", "c_old_tail")}
shares_server_fresh = {k: round(v / SERVER["fresh_prefill_tokens"], 4) for k, v in tot.items()}
other_server = SERVER["fresh_prefill_tokens"] - U_all
shares_server_fresh["not_in_duck_records"] = round(other_server / SERVER["fresh_prefill_tokens"], 4)

per_turn = {"U": dist([t["U"] for t in turns]), "a_gen_prev": dist([t["a_gen_prev"] for t in turns]),
            "b_new_content": dist([t["b_new_content"] for t in turns]), "c_old_tail": dist([t["c_old_tail"] for t in turns]),
            "a_share_of_U": dist([t["a_share"] for t in turns if t["a_share"] is not None]),
            "first_turn_U": dist([t["U"] for t in firsts]), "prompt_later": dist([t["P"] for t in turns])}
# by turn index bucket
buck = collections.defaultdict(list)
for t in turns:
    buck["1" if t["i"] == 1 else "2-4" if t["i"] <= 4 else "5-9" if t["i"] <= 9 else "10-19" if t["i"] <= 19 else "20+"].append(t)
by_turn = {k: {"n": len(v), "U_sum": sum(x["U"] for x in v), "a_sum": sum(x["a_gen_prev"] for x in v),
               "b_sum": sum(x["b_new_content"] for x in v), "c_sum": sum(x["c_old_tail"] for x in v),
               "a_share": round(sum(x["a_gen_prev"] for x in v) / max(1, sum(x["U"] for x in v)), 4)}
           for k, v in sorted(buck.items(), key=lambda kv: ["1", "2-4", "5-9", "10-19", "20+"].index(kv[0]))}
# by size of the previous generation
gb = collections.defaultdict(list)
for t in turns:
    g = t["a_gen_prev"]
    gb["<1k" if g < 1000 else "1k-4k" if g < 4000 else "4k-16k" if g < 16000 else ">=16k"].append(t)
by_gen = {k: {"n": len(v), "a_sum": sum(x["a_gen_prev"] for x in v), "U_sum": sum(x["U"] for x in v),
              "a_share": round(sum(x["a_gen_prev"] for x in v) / max(1, sum(x["U"] for x in v)), 4)}
          for k, v in sorted(gb.items(), key=lambda kv: ["<1k", "1k-4k", "4k-16k", ">=16k"].index(kv[0]))}

# savings of a perfect generated-tail checkpoint: the hit of turn k would end at align(P_{k-1} + G_{k-1}) instead of
# align(P_{k-1}); saved ~= G_{k-1} (the 64..127 alignment tail moves to the new end). Bounds per turn: G - 63 .. G + 63
save = tot["a_gen_prev"]
save_lo = sum(max(0, t["a_gen_prev"] - 63) for t in turns)
save_hi = sum(t["a_gen_prev"] + 63 for t in turns)
# the 4 cut requests would also save their previous generation (they did run their prefill on the server)
save_cut = sum(c["prev_gen"] for c in cut_est)
slope = None
if A1:
    slope = -A1["decomposition"]["prefill_regression_at21"]["slope_gen_per_prompt_tps"]
    slope_w = -A1["decomposition"]["prefill_regression_within_n_slope"]
sv = {}
for name, S in (("records", save), ("records_plus_cut_requests", save + save_cut), ("low", save_lo), ("high", save_hi)):
    e = {"prefill_tokens_saved": S, "share_of_server_fresh": round(S / SERVER["fresh_prefill_tokens"], 4),
         "share_of_server_prompt": round(S / SERVER["prompt_tokens"], 4),
         "engine_sec_at_probe_rate": round(S / PROBE_PREFILL_TOK_S, 1),
         "engine_pct_of_window_at_probe_rate": round(100 * S / PROBE_PREFILL_TOK_S / T, 2)}
    if slope:
        e["gen_tokens_recovered_reg21"] = round(slope * S)
        e["gen_pct_reg21"] = round(100 * slope * S / SERVER["generated_tokens"], 2)
        e["tok_s_gain_reg21"] = round(slope * S / T, 1)
        e["gen_tokens_recovered_within_n"] = round(slope_w * S)
        e["gen_pct_within_n"] = round(100 * slope_w * S / SERVER["generated_tokens"], 2)
        e["equiv_prefill_tok_s_reg21"] = round((A1["decomposition"]["prefill_regression_at21"]["intercept_pure_decode_tok_s"]) / slope)
    # Prometheus per-request prefill phase time, scaled by token share (overlaps decode; not exclusive engine time)
    e["prom_prefill_phase_sec_share"] = round(PROM_PREFILL_MEAN_SEC * PROM_N * S / SERVER["fresh_prefill_tokens"], 1)
    sv[name] = e

res = {"checks": checks, "totals_tokens": tot, "uncached_total_records": U_all, "client_totals": client,
       "server_totals": SERVER, "server_fresh_minus_records": other_server,
       "cut_requests_estimate": {"rows": cut_est, "est_fresh_sum": cut_fresh, "est_prompt_sum": cut_prompt},
       "shares_of_records_uncached": shares_all, "shares_of_later_turn_uncached": shares_later,
       "shares_of_server_fresh": shares_server_fresh, "per_turn": per_turn, "by_turn_index": by_turn,
       "by_prev_generation_size": by_gen, "perfect_generated_tail_checkpoint": sv,
       "per_turn_prefill_time_at_probe_sec": {"a_gen_prev": dist([t["a_gen_prev"] / PROBE_PREFILL_TOK_S for t in turns])},
       "notes": {"a_is_completion_count": "turn k-1 completion tokens as returned by usage; the template re-render of the "
                 "same reasoning + tool call can differ by a few tokens (wrappers fall into b)",
                 "first_turn": "every first request had cached = 0 (per-attempt isolation marker in the opener)"}}
json.dump(res, open(OUT / "analysis2_summary.json", "w"), indent=1)
with open(OUT / "reprefill_per_turn.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(turns[0].keys())); wr.writeheader(); wr.writerows(turns)
with open(OUT / "first_turns.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(firsts[0].keys())); wr.writeheader(); wr.writerows(firsts)
print(json.dumps(res, indent=1))
