"""Audit check after a gated rebuild (W4 or K100): the build facts and the CELL 7 body's embedded KV_KIT_GATE agree, the
gate passed, gpu_ready, every arm of the variant has an explicit per-K success (nbdf_lib per_k booleans), the .ipynb's own
bytes hash to the facts' sha256. K100 (facts["variant"] == "k100"): per_k_success must be a dict and D3s and D4s must each
have success, kv_configs_ok and k100_r9_run_ok all True (build_e27_nbdf_k100.k100_gate_extra's fields); a missing entry fails.
Fails closed on every missing field. Never edits anything.
usage: python nbdf_gate_verify.py <build facts json> <cell 7 body py> <notebook .ipynb> [D3s D4s ...]"""
import hashlib
import json
import re
import sys

K100_ARMS = ("D3s", "D4s")
K100_FIELDS = ("success", "kv_configs_ok", "k100_r9_run_ok")


def verify(facts, body, nb_bytes, arms):
    g = facts.get("kv_kit_gate") if isinstance(facts.get("kv_kit_gate"), dict) else {}
    m = re.search(r'KV_KIT_GATE = json.loads\(r"""(.*?)"""\)', body, re.S)
    try:
        emb = json.loads(m.group(1)) if m else None
    except ValueError:
        emb = None
    nb_sha = hashlib.sha256(nb_bytes).hexdigest() if nb_bytes is not None else None
    k100 = facts.get("variant") == "k100"
    res = {"variant": facts.get("variant"), "notebook_sha256": facts.get("sha256"), "notebook_file_sha256": nb_sha,
           "notebook_sha_equal": nb_sha is not None and nb_sha == facts.get("sha256"),
           "closed_build_sha256": facts.get("closed_build_sha256"), "passed": g.get("passed"), "gpu_ready": facts.get("gpu_ready"),
           "embedded_equals_facts": emb is not None and emb == g,
           "per_k": {a: (g.get("per_k") or {}).get(a) for a in arms}}
    why = []
    if g.get("passed") is not True:
        why.append("kv_kit_gate.passed is not True")
    if facts.get("gpu_ready") is not True:
        why.append("gpu_ready is not True")
    if not res["embedded_equals_facts"]:
        why.append("the CELL 7 KV_KIT_GATE differs from the facts (or is missing)")
    if not res["notebook_sha_equal"]:
        why.append(f"notebook bytes sha {nb_sha} != facts sha256 {facts.get('sha256')}")
    if not arms or not all(v is True for v in res["per_k"].values()):
        why.append(f"per_k not True for every arm: {res['per_k']}")
    if k100:
        pks = g.get("per_k_success")
        res["per_k_success"] = pks
        if not isinstance(pks, dict):
            why.append("K100: per_k_success missing or not a dict")
        else:
            for a in K100_ARMS:
                e = pks.get(a)
                if not isinstance(e, dict):
                    why.append(f"K100: per_k_success[{a}] missing")
                    continue
                bad = [f for f in K100_FIELDS if e.get(f) is not True]
                if bad:
                    why.append(f"K100: per_k_success[{a}] not True: {bad}")
    res["why"], res["ok"] = why, not why
    return res


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    facts_ = json.load(open(sys.argv[1], encoding="utf-8"))
    body_ = open(sys.argv[2], encoding="utf-8").read()
    try:
        nb_ = open(sys.argv[3], "rb").read()
    except OSError:
        nb_ = None
    r = verify(facts_, body_, nb_, sys.argv[4:] or list(K100_ARMS))
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["ok"] else 1)
