"""E10 ORDER2 + LOOPSTOP (Claude, 2026-10-15, user request). Deterministic.

Source: Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb (sha b62ae567...; LB-candidate DIRECT 86k, ORDER2 order; EVAL120 run by
hosen42 = 58/172). ONE change: a thinking trace that has degenerated into an exact short repetition is force-closed early
(same path as a trace that reaches the cap: stop text + </think> + answer phase), freeing the slot.
  * active only after 49,152 thinking tokens (the user's floor) and only while </think> has not been written
  * loop = over the last 1,024 thinking tokens, a period p <= 128 tokens with >= 97 % of tokens equal to the token p back,
    and the repeating unit contains a letter (digit-only repetition = grid rows, never flagged); confirmed on 2 checks
    256 tokens apart
  * offline (session scratchpad order2/loop_detector_offline.py): 5/5 real loops found in 333 saved thinking tails
    (4 ORDER2 + 1 dual v2), 0 false positives, 0 flags in 41 full E8 traces (1.86 M tokens) checked from token 1,024
Everything else unchanged (77,824 + 8,192, ORDER2, EVAL120 commit default, 16 sequences, hidden rerun).
usage: python build_e10_order2_loopstop.py
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb"
SRC_SHA = "b62ae567aa4d8af26c652dcfcfc16fdf8d1d4b8e1a32ba04c6e3892b71d0b6b2"
OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_ORDER2_LOOPSTOP.ipynb"


def once(text, old, new, where):
    if text.count(old) != 1:
        sys.exit(f"replacement not unique ({text.count(old)}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "ORDER2 source changed"
nb = json.loads(raw)
c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
put = lambda i, t: c[i].__setitem__("source", t.splitlines(keepends=True))

# ---- cell 1 (configuration)
t = src(2)
t = once(t, 'CONFIG = {k: globals()[k] for k in ["PROFILE"',
         '# LOOPSTOP (2026-10-15): a thinking trace that has degenerated into an exact short repetition ("row? row? row? ..."; 4 of\n'
         '# the 74 forced closes in the ORDER2 EVAL120 run) is force-closed early, exactly like a trace at the cap, and the answer\n'
         '# phase follows. Never before 49,152 thinking tokens (the user\'s floor). Offline: 5/5 real loops found in 333 saved thinking\n'
         '# tails, 0 false positives, 0 flags in 41 full traces; digit-only repetition (grid rows) is never flagged.\n'
         'LOOP_STOP = os.environ.get("Q27X_LOOP_STOP", "1") == "1"\n'
         'LOOP_MIN_THINK = 49152\n'
         'LOOP_WINDOW, LOOP_MAX_PERIOD, LOOP_MATCH, LOOP_EVERY, LOOP_CONFIRM = 1024, 128, 0.97, 256, 2\n'
         'CONFIG = {k: globals()[k] for k in ["PROFILE"', "config block")
t = once(t, '"TOTAL_BUDGET_SEC", "MIN_THINK_BUDGET"]}',
         '"TOTAL_BUDGET_SEC", "MIN_THINK_BUDGET", "LOOP_STOP", "LOOP_MIN_THINK", "LOOP_WINDOW", "LOOP_MAX_PERIOD",\n'
         '          "LOOP_MATCH", "LOOP_EVERY", "LOOP_CONFIRM"]}', "config keys")
put(2, t)

# ---- cell 6 (scheduler)
t = src(7)
t = once(t, '"aborted_at_hard_stop": 0, "not_admitted": 0, "max_active": 0, "steps": 0}',
         '"aborted_at_hard_stop": 0, "not_admitted": 0, "max_active": 0, "steps": 0,\n         "loop_stopped": 0}', "SCHED")
t = once(t, 'def _req(uid, phase):\n    return f"{uid}|{phase}"\n',
         'def _req(uid, phase):\n    return f"{uid}|{phase}"\n\n'
         '_LOOP_ALPHA = re.compile(r"[A-Za-z]")\n\n'
         'def loop_period(ids):\n'
         '    """LOOPSTOP: smallest period p <= LOOP_MAX_PERIOD with >= LOOP_MATCH of the last LOOP_WINDOW tokens equal to the token\n'
         '    p back, whose repeating unit contains a letter (digit-only repetition = grid rows, legal thinking); 0 = no loop"""\n'
         '    a = ids[-LOOP_WINDOW:]\n'
         '    if len(a) < LOOP_WINDOW:\n'
         '        return 0\n'
         '    for p in range(1, LOOP_MAX_PERIOD + 1):\n'
         '        allow = int((1.0 - LOOP_MATCH) * (LOOP_WINDOW - p)); bad = 0\n'
         '        for i in range(p, LOOP_WINDOW):\n'
         '            if a[i] != a[i - p]:\n'
         '                bad += 1\n'
         '                if bad > allow:\n'
         '                    break\n'
         '        else:\n'
         '            if _LOOP_ALPHA.search(tokenizer.decode(a[-p:], skip_special_tokens=False)):\n'
         '                return p\n'
         '    return 0\n\n'
         'def loop_check(st):\n'
         '    """LOOPSTOP: True when a phase-1 trace past LOOP_MIN_THINK thinking tokens (</think> not yet written) has been in an\n'
         '    exact loop on LOOP_CONFIRM consecutive checks LOOP_EVERY tokens apart"""\n'
         '    n = len(st["p1_ids"])\n'
         '    if n < LOOP_MIN_THINK or n - st.get("loop_last", 0) < LOOP_EVERY:\n'
         '        return False\n'
         '    st["loop_last"] = n\n'
         '    if THINK_CLOSE_ID in st["p1_ids"]:\n'
         '        return False\n'
         '    st["loop_streak"] = st.get("loop_streak", 0) + 1 if loop_period(st["p1_ids"]) else 0\n'
         '    return st["loop_streak"] >= LOOP_CONFIRM\n', "loop functions")
t = once(t, '                else:\n                    st["p2_finish"] = fr; finish(st, st["mode"])\n',
         '                else:\n                    st["p2_finish"] = fr; finish(st, st["mode"])\n'
         '            elif phase == 1 and LOOP_STOP and loop_check(st):          # LOOPSTOP: force-close a degenerate loop\n'
         '                engine.abort_request([_req(uid, 1)])\n'
         '                SCHED["loop_stopped"] += 1; st["p1_finish"] = "loop"; st["loop_stop_at"] = len(st["p1_ids"])\n'
         '                print(f"  LOOPSTOP {uid}: exact repetition at {len(st[\'p1_ids\'])} thinking tokens -> forced answer", flush=True)\n'
         '                if not start_phase2(st, ANSWER_MAX_TOKENS):\n'
         '                    finish(st, "no_room_for_answer")\n', "loop hook")
t = once(t, '"order_score": ORDER_SCORE[UNIT_TASK[uid]], "admit_rank": ORDER_ADMIT_RANK[uid]}',
         '"order_score": ORDER_SCORE[UNIT_TASK[uid]], "admit_rank": ORDER_ADMIT_RANK[uid],\n'
         '           "loop_stop_at": st.get("loop_stop_at")}', "record field")
put(7, t)

c[0]["source"] = (src(0) + "\n\n**LOOPSTOP (2026-10-15):** a thinking trace that has degenerated into an exact short repetition "
                  "(e.g. \"row? row? row? …\") is force-closed early - the same path as a trace at the cap (stop text + `</think>` + "
                  "answer) - and its slot is freed. Only after 49,152 thinking tokens and only while `</think>` is not written; "
                  "digit-only repetition (grid rows) is never flagged. Records carry `loop_stop_at`, the report `scheduler.loop_stopped`. "
                  "`Q27X_LOOP_STOP=0` turns it off. Everything else is ORDER2 unchanged.\n").splitlines(keepends=True)
for cell in c:
    if cell["cell_type"] == "code":
        cell["outputs"] = []; cell["execution_count"] = None
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
OUT.write_bytes(data)
print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())
