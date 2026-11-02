"""E33 chain lib (Experiment 33, user 2026-11-02; DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md). Stdlib only.

The notebook loads this file from the E33 payload dataset by name + sha256 (E33D_FILES["lib"], CELL 1) and uses it for:
  - check_chain(E33_CHAIN): the CELL 1 switch -> the checked dict. A bad value never stops the run: a bad item turns only
    that item off (recorded in "invalid"); a bad 'on' / a non-dict turns the whole chain off. The 60,000 thinking floor
    (user rule) applies to the switch; a kit override (E11_HARNESS=1 + E33_CHAIN_JSON) is exempt, as E29's direct2_cfg.
  - can_start(): the E33_CAN_START rule for stage E (Direct; design section 3, admission): it starts only when its thinking
    can reach the 60,000 floor before its drain point (deadline - drain_min) and before the admission stop (deadline - 12 min,
    E29 / E31). The Poetiq stages C / D use e33_poetiq.can_admit (the same rule with Poetiq's own drain). Duck stages: not gated.
  - per_stream_rate(): the per-request decode speed used by can_start = the LOWER of the E28 K100 measured 18 tok/s per
    stream and the measured server rate / (running + 1) (= e33_poetiq.admission_rate's "lowest wins" rule).
"""
import json

LIB_VERSION = "e33_lib 1.0 (2026-11-02)"
THINK_FLOOR = 60000                 # user rule: thinking >= 60,000 (approval needed below)
PER_STREAM_TOK_S = 18.0             # E28 K100 measured: 419.5 tok/s at 23.4 running (= e29_direct.PER_STREAM_TOK_S)
START_MARGIN_SEC = 60.0             # = E29's E29D_CAN_START margin (now + 60 s + ...)
ADMIT_STOP_BEFORE_END_SEC = 12 * 60  # = e29_direct.ADMIT_STOP_BEFORE_END_SEC (no new Direct task from T - 12 min)
MAX_MODEL_LEN = 131072              # the E28 K100 / E29 server --max-model-len (unchanged)
CHAIN_KEYS = ("on", "poetiq", "direct", "attempt2", "poetiq_think", "poetiq_answer", "direct_think", "direct_answer",
              "temperature", "drain_min")
DEFAULT_CHAIN = {"on": True, "poetiq": True, "direct": True, "attempt2": True, "poetiq_think": 60000, "poetiq_answer": 8192,
                 "direct_think": 104000, "direct_answer": 8192, "temperature": 1.0, "drain_min": 26}
SOLVERS = ("poetiq", "direct")
# stage name -> the solver whose switch / budgets / floor apply (C = Poetiq iteration 1, D = iteration 2, E = Direct);
# A / B (Duck) are not gated by can_start
STAGE_SOLVER = {"C": "poetiq", "D": "poetiq", "E": "direct", "POETIQ": "poetiq", "DIRECT": "direct",
                "P1": "poetiq", "P2": "poetiq"}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def check_chain(cfg, harness_json=None):
    """-> {"on", "poetiq", "direct", "attempt2", the budgets, "temperature", "drain_min", "invalid": {item: why},
    "ignored": [unknown keys], "given": cfg, "harness_override": bool, "lib": LIB_VERSION}. Never raises."""
    out = {"on": False, "poetiq": False, "direct": False, "attempt2": False, "invalid": {}, "ignored": [], "given": cfg,
           "harness_override": bool(harness_json), "lib": LIB_VERSION, "think_floor": THINK_FLOOR}
    try:
        c = dict(cfg)
        if harness_json:
            c.update(json.loads(harness_json))
    except Exception as e:
        out["invalid"]["chain"] = f"{type(e).__name__}: {e}"[:300]
        return out
    out["ignored"] = sorted(str(k) for k in c if k not in CHAIN_KEYS)
    if not isinstance(c.get("on"), bool):
        out["invalid"]["chain"] = f"'on' must be True / False (got {c.get('on')!r})"
        return out
    floor = 1 if harness_json else THINK_FLOOR
    t, dm = c.get("temperature"), c.get("drain_min")
    t_ok = isinstance(t, (int, float)) and not isinstance(t, bool) and 0.0 <= float(t) <= 2.0
    dm_ok = _is_int(dm) and dm > 0
    out["temperature"], out["drain_min"] = (float(t) if t_ok else None), (dm if dm_ok else None)
    for s in SOLVERS:
        want, think, answer = c.get(s), c.get(s + "_think"), c.get(s + "_answer")
        why = []
        if not isinstance(want, bool):
            why.append(f"'{s}' must be True / False (got {want!r})")
        if not (_is_int(think) and think > 0):
            why.append(f"{s}_think must be a positive int (got {think!r})")
        elif think < floor:
            why.append(f"{s}_think {think} < the thinking floor {floor} (user rule)")
        if not (_is_int(answer) and answer > 0):
            why.append(f"{s}_answer must be a positive int (got {answer!r})")
        if not why and think + answer >= MAX_MODEL_LEN:
            why.append(f"{s}_think + {s}_answer {think + answer} >= max-model-len {MAX_MODEL_LEN}")
        if not t_ok:
            why.append(f"temperature must be a number in [0, 2] (got {t!r})")
        if not dm_ok:
            why.append(f"drain_min must be a positive int (got {dm!r})")
        ok = not why
        if why and want is not False:
            out["invalid"][s] = "; ".join(why)[:400]
        out[s] = bool(ok and want)
        out[s + "_think"], out[s + "_answer"] = (think if ok else None), (answer if ok else None)
    a2 = c.get("attempt2")
    if isinstance(a2, bool):
        out["attempt2"] = a2
    else:
        out["invalid"]["attempt2"] = f"'attempt2' must be True / False (got {a2!r})"
    out["on"] = c["on"]
    if not out["on"]:
        out.update(poetiq=False, direct=False, attempt2=False)
    return out


def items_state(cfg):
    """{item: "ON" | "OFF" | "OFF (invalid: ...)"} of the chain switch (the [LIVE start] line / the report)."""
    cfg = cfg or {}
    inv = cfg.get("invalid") or {}
    st = {}
    for k in ("on",) + SOLVERS + ("attempt2",):
        name = "chain" if k == "on" else k
        why = inv.get(name)
        st[name] = "ON" if cfg.get(k) else ("OFF (invalid: " + str(why)[:120] + ")" if why else "OFF")
    if not cfg.get("on"):
        st = {k: (v if k == "chain" else "OFF (chain off)") for k, v in st.items()}
    return st


def stage_solver(stage):
    """"C" / "D" -> "poetiq", "E" -> "direct", A / B (Duck) -> None (the attempt-2 prime and case are ignored)."""
    return STAGE_SOLVER.get(str(stage or "").strip().upper().rstrip("'"))


def per_stream_rate(tok_s=None, running=None, fallback=PER_STREAM_TOK_S):
    """The decode speed one NEW request can expect: the lower of the E28 K100 measured 18 tok/s per stream and the measured
    server generation rate shared by the running requests plus the new one; no measurement (or an idle server) -> 18."""
    try:
        if tok_s is None or running is None or float(tok_s) <= 0 or int(running) < 1:
            return float(fallback)
        return min(float(fallback), float(tok_s) / (int(running) + 1))
    except Exception:
        return float(fallback)


def floor_tokens(cfg, solver):
    """The thinking tokens a stage must be able to reach: the 60,000 floor; a kit override (harness) uses min(floor, think)."""
    think = (cfg or {}).get(solver + "_think") or THINK_FLOOR
    return min(THINK_FLOOR, int(think)) if (cfg or {}).get("harness_override") else THINK_FLOOR


def can_start(now, deadline, cfg, stage, rate=PER_STREAM_TOK_S):
    """-> (go, why, need_sec). Duck stages (no solver) -> (True, "duck", 0). A Poetiq / Direct stage: its switch must be
    on, now must be before the admission stop (deadline - 12 min) and now + 60 s + floor / rate must not pass the drain point
    (deadline - drain_min * 60)."""
    solver = stage_solver(stage)
    if solver is None:
        return True, "duck", 0.0
    cfg = cfg or {}
    if not (cfg.get("on") and cfg.get(solver)):
        return False, "off", None
    try:
        r = max(1e-3, float(rate))
        need = START_MARGIN_SEC + floor_tokens(cfg, solver) / r
        drain_at = float(deadline) - float(cfg["drain_min"]) * 60.0
        ok = float(now) < float(deadline) - ADMIT_STOP_BEFORE_END_SEC and float(now) + need <= drain_at
        return ok, ("ok" if ok else "late"), round(need, 1)
    except Exception as e:
        return False, f"error {type(e).__name__}"[:60], None
