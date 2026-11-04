# E25 NB-M1L overlay (2026-11-02, user decision: the LIGHT build). exec'd into nbm1_lib's namespace right after nbm1_lib
# (notebook: embedded + sha-checked; builder / tests: the same). Only these changes:
#  1. Q (copies B / C): facts, the smoke / first request (no ladder), the prefill probe (the windows' admission rate),
#     warmup, its windows 8k@16 x2 / 20k@24 / 40k@16 / 90k@8 / 116k@6. NO ladder, oracle, oracle A/A, long oracle, prefix
#     test, TF, profiles. PROF (copy C) keeps its two 30-s windows inside Q's boot.
#  2. Levers: NO TF, NO oracle vs Q; facts, the smoke hard gate, the read-back of the one change, warmup, the windows.
#  3. Real-failure rules, the OOM ladder, NBM1_ON, the hard-only gates: unchanged; a label that needed a removed item is
#     INCONCLUSIVE / NOT_IN_PLAN (record only).
#  4. Time: body caps sized on the body's own due() rule (win_need); copies B (hosen42) / C (foysalemonshanto).
assert not globals().get("LIGHT"), "nbm1l overlay applied twice"
LIGHT = True
LIB_VERSION = LIB_VERSION + " + nbm1l-overlay 2026-11-02 r1 (LIGHT)"
COPIES = {"b": dict(COPIES["b"]), "c": dict(COPIES["c"])}
OWNERS = {"b": "hosen42", "c": "foysalemonshanto"}
TITLES = dict(TITLES, Q="Q = NBDF-K100 Q verbatim (MTP K3, I8, Q-defer, seqs 24, pin 7,933,526,016), LIGHT: facts + smoke + windows")
Q_ITEMS = {"ladder": False, "prefill_probe": True, "oracle": False, "oracle_aa": False, "long_oracle": False, "prefix": False,
           "profiles": False, "tf": False}
LEVER_ITEMS = dict(LEVER_ITEMS_OFF, oracle=False, oracle_aa=False, tf=False)
PLAN = {k: (v[0], v[1], False) for k, v in PLAN.items()}      # no TF on any lever (IDX / LP included)
BAND_TOK = {"8k": 8192, "20k": 20000, "40k": 40000, "90k": 90000, "116k": 116000}
WIN_SEC, ADMIT_TOK_S, LONG_RESET_SEC = 90.0, 3000.0, 60.0    # body: WIN, K100 prefill_admit_floor_tok_s, the long reset's +60 s
COPY_RUNTIME_MAX = 150      # build-time assertion: each light copy's planned runtime <= this
BUDGET_MIN = 155            # the light hard stop: max over B / C of (runtime, every arm's full cap) + 5 min, rounded up to 5
DEADMAN_MIN = BUDGET_MIN + 5


def _c(x):
    return int(math.ceil(round(x, 6)))


def win_need_sec(band, n, prof=False):
    """= the body's win_need() at the admission rate min(prefill, 3000 tok/s): a gated item starts only with this left."""
    if prof:
        return 360 + 10 * max(0, n - 8) + n * BAND_TOK[band] / ADMIT_TOK_S
    return WIN_SEC + 120 + n * BAND_TOK[band] / ADMIT_TOK_S


def body_plan(items):
    """[(name, min, need_sec | None)] in body order -> (body_min, cap_min, rows): the cap holds every gated item's due() need
    from its planned start, + 1 min."""
    t, need, rows = 0.0, 0.0, []
    for nm, m, nd in items:
        if nd is not None:
            need = max(need, t + nd / 60.0)
        rows.append([nm, m, None if nd is None else round(nd / 60.0, 2), round(t, 2)])
        t += m
    return round(t, 2), _c(max(t, need) + 1.0), rows


def body_items(windows, q, prof):
    it = [("facts", ITEM_MIN["facts"], None), ("smoke", ITEM_MIN["smoke"], 240)]
    if q:
        it.append(("prefill_probe", ITEM_MIN["prefill_probe"], 300))
    it.append(("warmup", ITEM_MIN["warmup"], 150))
    for b, n, tag in windows:
        nd = win_need_sec(b, int(n)) + (LONG_RESET_SEC if b in ("90k", "116k") else 0.0)
        it.append((f"{b}@{int(n)}{tag}", WINDOW_MIN[f"{b}@{int(n)}"], nd))
    if prof:
        for b, n in PROF_WINDOWS:
            it.append((f"prof_{b}@{n}", ITEM_MIN["prof"] / len(PROF_WINDOWS), win_need_sec(b, n, prof=True)))
    it.append(("end", ITEM_MIN["end"], None))
    return it


def q_time(prof_on):
    body, cap, rows = body_plan(body_items(Q_WINDOWS, True, prof_on))
    post = POST_MIN["q"] if prof_on else POST_MIN["lever"]
    return _c(READY_MIN["first"] + body + post), cap, {"ready": READY_MIN["first"], "body": body, "post": post, "items": rows, "light": True}


def lever_time(key):
    rc, wins, _ = PLAN[key]
    body, cap, rows = body_plan(body_items(wins, False, False))
    return (_c(READY_MIN[rc] + body + POST_MIN["lever"]), cap,
            {"ready": READY_MIN[rc], "ready_class": rc, "body": body, "post": POST_MIN["lever"], "items": rows, "light": True})


_m1_arms_full = m1_arms


def m1_arms(N, aq, switch=None, vocab=None):
    """nbm1_lib.m1_arms (Q + exactly one change per lever, unchanged) + the light item plans (Q_ITEMS / LEVER_ITEMS)."""
    arms = _m1_arms_full(N, aq, switch, vocab)
    arms["Q"].update(m1_items=dict(Q_ITEMS), m1_no_ladder=True)
    for k in LEVERS:
        arms[k]["m1_items"] = dict(LEVER_ITEMS)
    return arms


def copy_need_min(arms, switch):
    """The latest planned end of a copy with every arm's FULL body cap (start + ready + cap + post): the hard stop holds it
    + 5 min, so no switched-on arm's cap is clipped by the budget."""
    sw = switch_map(switch)
    t, end = SETUP_MIN, SETUP_MIN
    for k, a in arms.items():
        if not sw.get(k, True):
            continue
        est, cap, tb = q_time(sw.get("PROF") and sw.get("Q")) if k == "Q" else (a["est_min"], a["cap_min"], a["time_model"])
        end = max(end, t + tb["ready"] + cap + tb["post"])
        t += est
    return round(end, 1)
