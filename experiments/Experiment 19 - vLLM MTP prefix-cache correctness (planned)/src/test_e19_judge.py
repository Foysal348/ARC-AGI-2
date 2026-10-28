"""rev 2 negative regression tests of src/e19_judge.py (the GPT strict pre-GPU audits of 2026-10-27).
Each case must NOT come out PASS where the audit showed a false PASS; the clean control must stay PASS.
usage: python test_e19_judge.py"""
import copy
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("j", Path(__file__).with_name("e19_judge.py"))
J = importlib.util.module_from_spec(spec); spec.loader.exec_module(J)

TEXT = "The rule maps each colored object to its mirror; the output grid keeps the frame and fills the interior."


def row(stage, ps, idx, n=300, toks=None, text=TEXT, status="ok", cached=0, reset=200, margins="full", **kw):
    toks = list(range(1000, 1000 + n)) if toks is None else toks
    mg = [3.0] * len(toks) if margins == "full" else ([None] * len(toks) if margins == "none" else
                                                      [3.0 if i % 2 else None for i in range(len(toks))])
    r = {"arm": "D", "stage": stage, "pass": ps, "idx": idx, "status": status, "tokens": toks, "margins": mg,
         "top1_mismatch": [], "text": text, "cached_tokens": cached, "usage_seen": True, "reset": reset}
    r.update(kw)
    return r


META = {"arm": "D", "prefix_caching": True, "spec": 3, "serial_idx": [0], "churn_jobs": [],
        "long_stages": [{"name": "long30k", "idx": [0]}], "accept_windows": [{"draft_tokens": 600, "accepted": 300}]}


def clean_rows():
    return [row("serial", "cold", 0), row("serial", "warm", 0, cached=3200), row("serial", "cold2", 0),
            row("long30k", "nat", 0, n=1024, g_pre=512),
            row("long30k", "cont", 0, n=512, toks=list(range(1512, 2024)), cached=33600, pf_len=32000)]


def verdict(rows, meta=META):
    return J.e19_judge_arm(rows, meta)["verdict"]


fails = []


def check(name, got, want_not_pass=True, want=None):
    ok = (got == want) if want else (got != "PASS" if want_not_pass else got == "PASS")
    print(("PASS " if ok else "FAIL ") + f"{name}: verdict {got}")
    if not ok:
        fails.append(name)


check("control: clean rows", verdict(clean_rows()), want="PASS")
# N1: all of cold / warm / cold2 corrupted the same way ('!' x 300)
r = clean_rows()
for k in range(3):
    r[k]["text"] = "!" * 300; r[k]["tokens"] = [0] * 300; r[k]["margins"] = [3.0] * 300
check("N1 all serial rows corrupted (no clean partner)", verdict(r))
# corrupt warm only, clean cold: must FAIL
r = clean_rows(); r[1]["text"] = "!" * 300
check("warm corrupted, cold clean -> FAIL", verdict(r), want="FAIL")
# N2: cold 300, warm 200 (same budget, warm stopped early)
r = clean_rows(); r[1]["tokens"] = r[1]["tokens"][:200]; r[1]["margins"] = r[1]["margins"][:200]
check("N2 warm stopped early (length-only)", verdict(r))
# N3: natural stopped short; continuation na_short
r = clean_rows(); r[3]["tokens"] = r[3]["tokens"][:50]; r[3]["margins"] = r[3]["margins"][:50]
r[4] = row("long30k", "cont", 0, n=0, toks=[], status="na_short", margins="none")
check("N3 continuation na_short", verdict(r))
# N4: identical token ids, no margins anywhere
r = clean_rows()
for x in r:
    x["margins"] = [None] * len(x["tokens"])
check("N4 margins missing everywhere", verdict(r))
check("N4b half the margins missing", verdict([dict(x, margins=[3.0 if i % 2 else None for i in range(len(x["tokens"]))])
                                               for x in clean_rows()]))
# reset not confirmed before a cold row
r = clean_rows(); r[0]["reset"] = "URLError('refused')"
check("cache reset failed before cold", verdict(r))
# no usage record at all
r = clean_rows()
for x in r:
    x["usage_seen"] = False; x["cached_tokens"] = None
check("no usage records", verdict(r))
# trailing-block coverage flag
j = J.e19_judge_arm(clean_rows(), META)
print(("PASS " if j["cache"]["trailing_mamba_cache_tested"] else "FAIL ") + "trailing block exercised flag (cont hit 33,600 > pf 32,000)")
r = clean_rows(); r[4]["cached_tokens"] = 32000
j = J.e19_judge_arm(r, META)
ok = (not j["cache"]["trailing_mamba_cache_tested"]) and any("NOT_EXERCISED" in s for s in j["reasons"])
print(("PASS " if ok else "FAIL ") + "trailing block NOT_EXERCISED reported when the hit stops at the prompt")
if not ok:
    fails.append("not_exercised")

# HIGH (calibrator): an INCOMPLETE calibrator must not raise the churn ceiling
def churn_meta(arm, pc, spec_):
    return {"arm": arm, "prefix_caching": pc, "spec": spec_, "serial_idx": [0], "churn_jobs": [[0, 0, 1024]],
            "long_stages": [], "accept_windows": [{"draft_tokens": 600, "accepted": 300}] if spec_ else []}


def churn_rows(arm, flip, pc, incomplete=False):
    base = list(range(1000, 1300))
    ser = [dict(row("serial", "cold", 0), arm=arm), dict(row("serial", "warm" if True else "", 0, cached=3200 if pc else 0), arm=arm)]
    if pc:
        ser.append(dict(row("serial", "cold2", 0), arm=arm))
    ch = base[:]
    if flip:
        ch[100] = 7
    cr = dict(row("churn", "j0", 0, toks=ch, cached=0), arm=arm)
    out = ser + [cr]
    if incomplete:
        out[0]["margins"] = [None] * 300          # makes the calibrator INCOMPLETE (thin margins)
    return out


metas = {"A": churn_meta("A", False, 0), "D": churn_meta("D", True, 3)}
rows = churn_rows("A", True, False, incomplete=True) + churn_rows("D", True, True)
V = J.e19_verdicts(rows, metas)
print("   A in-arm:", V["A"]["in_arm"]["verdict"], "| D churn:", V["D"]["churn"])
check("incomplete calibrator A does not license D's churn flip", V["D"]["final"])
print("\n" + ("ALL REGRESSION TESTS PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
