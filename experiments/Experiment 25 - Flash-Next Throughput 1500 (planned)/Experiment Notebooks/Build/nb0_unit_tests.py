"""E25 NB0 fault-injection unit tests (pattern: Experiment 13 rc2_unit_tests.py). Runs the exported notebook code with
NB0_UNIT=1 (RC2 definitions + A0 merge + prompts + NB0 library, no boots) against the RC2 local fake input tree, then
tests every new NB0 function, with real data where it exists (RC2's Kaggle server logs, the real Merlin launch.build,
the real Merlin ad42e37 QSA sources).

Usage: python nb0_unit_tests.py <fake input dir> <work dir>
"""
import hashlib
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INP, WORK = Path(sys.argv[1]), Path(sys.argv[2])
MERLIN_SRC = Path(os.environ.get("NB0_TEST_MERLIN_SRC") or r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")       # Merlin ad42e37 (kit: the needed files)
RC2_LOGS = Path(os.environ.get("NB0_TEST_RC2_LOGS") or r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2/logs")   # RC2 GPU v1 server logs
shutil.rmtree(WORK, ignore_errors=True)
for d in ("out", "tmp", "shm", "proc", "cgroup"):
    (WORK / d).mkdir(parents=True)
os.environ.update(NB0_UNIT="1", E23_FAKE_SERVER=str(HERE / "nb0_fake_server.py"), E23_OUT=str(WORK / "out"),
                  E23_TMP=str(WORK / "tmp"), E23_INPUT=str(INP), RC2_SUBMISSION=str(WORK / "out/submission.json"),
                  RC2_SHM=str(WORK / "shm"), NB0_BUDGET_SEC="3600", NB0_PROC=str(WORK / "proc"), NB0_CGROUP=str(WORK / "cgroup"))
os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
t0 = time.time()
ns = runpy.run_path(str(HERE / "nb0_code_export.py"), run_name="nb0")["analyze"].__globals__     # live module globals
print(f"export loaded in {time.time() - t0:.0f} s")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


L, np_ = ns["L"], ns["np"]
ns["FAKE"] = None                     # gates and host checks as on the real GPU box

# ---------------------------------------------------------------- 1. gates (external audits 2 / 3) incl. the negative tests
sg, tg = ns["screening_gate"], ns["triage_gate"]
aa = {"status": "MEASURED", "mean_abs_dlogprob": 0.0, "rank1_agreement": 1.0}
rc2 = {"status": "MEASURED", "mean_abs_dlogprob": 1.8764, "rank1_agreement": 0.3672, "p99_abs_dlogprob": 5.464}
clean = {"status": "MEASURED", "mean_abs_dlogprob": 0.0, "rank1_agreement": 1.0}
nan = {"status": "MEASURED", "mean_abs_dlogprob": float("nan"), "rank1_agreement": 0.99}
check("screening: missing TF -> INCONCLUSIVE_FP8", sg(aa, None) == "INCONCLUSIVE_FP8")
check("screening: TF incomplete -> INCONCLUSIVE_FP8", sg(aa, {"status": "INCONCLUSIVE_INCOMPLETE_TF"}) == "INCONCLUSIVE_FP8")
check("screening: reference A/A missing -> INCONCLUSIVE_REFERENCE", sg(None, clean) == "INCONCLUSIVE_REFERENCE")
check("screening: NaN mean -> INCONCLUSIVE_NONFINITE", sg(aa, nan) == "INCONCLUSIVE_NONFINITE")
check("screening: RC2 0.3672 / 1.8764 -> HOLD", sg(aa, rc2) == "HOLD_FP8_NUMERICAL_DIVERGENCE")
check("screening: rank-1 0.89 alone -> HOLD", sg(aa, {**clean, "rank1_agreement": 0.89}) == "HOLD_FP8_NUMERICAL_DIVERGENCE")
check("screening: mean 0.16 alone -> HOLD", sg(aa, {**clean, "mean_abs_dlogprob": 0.16}) == "HOLD_FP8_NUMERICAL_DIVERGENCE")
check("screening: clean match -> SCREENING_PASS", sg(aa, clean) == "SCREENING_PASS")
check("triage: missing TF -> INCONCLUSIVE_TF", tg(None) == "INCONCLUSIVE_TF")
check("triage: NaN -> INCONCLUSIVE_NONFINITE", tg(nan) == "INCONCLUSIVE_NONFINITE")
check("triage: RC2 0.3672 / 1.8764 -> NO_GO", tg(rc2) == "NO_GO_FP8_TF_DIVERGENCE")
check("triage: rank-1 0.49 -> NO_GO", tg({**clean, "rank1_agreement": 0.49}) == "NO_GO_FP8_TF_DIVERGENCE")
check("triage: mean 1.01 -> NO_GO", tg({**clean, "mean_abs_dlogprob": 1.01}) == "NO_GO_FP8_TF_DIVERGENCE")
check("triage: clean match -> PENDING_LONG_ARC_ACCURACY (never GO)", tg(clean) == "PENDING_LONG_ARC_ACCURACY")
check("triage: short oracle failed -> NO_GO_SHORT_ORACLE", tg(clean, short_ok=False) == "NO_GO_SHORT_ORACLE")
check("triage: short oracle unknown -> INCONCLUSIVE", tg(clean, short_ok=None) == "INCONCLUSIVE_SHORT_ORACLE")
w = ns["worst"]
check("worst: HOLD dominates INCONCLUSIVE and PASS", w(["SCREENING_PASS", "INCONCLUSIVE_FP8", "HOLD_FP8_NUMERICAL_DIVERGENCE"]) == "HOLD_FP8_NUMERICAL_DIVERGENCE")
check("worst: a missing band keeps PENDING from passing", w(["PENDING_LONG_ARC_ACCURACY", "INCONCLUSIVE_TF"]) == "INCONCLUSIVE_TF")
check("worst: no data -> INCONCLUSIVE_NO_DATA", w([]) == "INCONCLUSIVE_NO_DATA")
gates_all = {sg(aa, x) for x in (None, nan, rc2, clean)} | {tg(x) for x in (None, nan, rc2, clean)}
check("no gate output is ever a GO", not any("GO" == g or g.startswith("GO") for g in gates_all), sorted(gates_all))

# ---------------------------------------------------------------- 2. prompt ids = RC2's (25 distinct prompts RC2 sent on Kaggle)
pc = ns["prompt_check"]()
check("prompt_check: every RC2 Kaggle prompt reproduced locally", pc["ok"] and pc["reproduced"] == pc["rc2_recorded"] == 25,
      json.dumps({k: pc[k] for k in ("reproduced", "rc2_recorded", "missing")}))
check("prompt_check: NB0 40k v5 is RC2's TF_P", pc["rc2_tf_prompt_is_40k_v5"])
print("TF prompts:", json.dumps(pc["tf_prompts"]))
print("SC prompts:", json.dumps(pc["sc_prompts"]))

# ---------------------------------------------------------------- 3. argv edits on the REAL Merlin launch.build (ad42e37)
sys.path.insert(0, str(MERLIN_SRC / "tools" / "merlin"))
import launch as ML  # noqa: E402

RC2_REF, BOOTS, nb0_argv, flags = ns["RC2_REF"], ns["BOOTS"], ns["nb0_argv"], ns["flags"]
plans = {}
for kv, port, kvb in (("fp8", 8001, 2479838700), ("bf16", 8000, 2464415744)):
    base = {"GPUS": "0,1,2,3", "PORT": str(port), "KV_BYTES": str(kvb), "P2P": "off", "VENV": "/tmp/merlin_off/venv",
            "TREE": str(MERLIN_SRC), "CHAT_TEMPLATE": "", "WARM": "0", "CACHE_ROOT": "/tmp/merlin-cache", "PATH": "/usr/bin",
            "CUDA_HOME": "/tmp/cu13"}
    if kv == "fp8":
        base["SCALES"] = "/tmp/fn_merlin/qsa_kv_scales_262k.json"
    extra = ["--served-model-name", "q27", "--max-model-len", "131072", "--port", str(port), "--default-chat-template-kwargs",
             json.dumps({"enable_thinking": True, "reasoning_effort": "xhigh"})]
    plan = ML.build("flash-next", "fn-quad-24g-sm89", "/tmp/fn_merlin", base, extra=extra, kv=kv, checkpoint="flash-next", status="untested")
    plans[kv] = plan.argv()
    check(f"launch.build({kv}) argv tail == RC2's recorded Kaggle tail", plans[kv][-60:] == RC2_REF[kv]["argv_tail"],
          "" if plans[kv][-60:] == RC2_REF[kv]["argv_tail"] else str([(a, b) for a, b in zip(plans[kv][-60:], RC2_REF[kv]["argv_tail"]) if a != b][:4]))
table = {}
for key, b in BOOTS.items():
    a = nb0_argv(plans[b["kv"]], b)
    f = flags(a)
    env = dict(RC2_REF[b["kv"]]["env"])
    if b["byte"]:
        env.update(NB0_QSA_FORCE_BYTE_DECODER="1", VLLM_CACHE_ROOT="/tmp/merlin-cache-bytedec")
    diff = ns["config_diff"](a, env, RC2_REF[b["kv"]]["kv_bytes"], "fp8")
    table[key] = diff
    check(f"boot {key}: prefix caching on", "--enable-prefix-caching" in a)
    check(f"boot {key}: MTP {'on' if b['mtp'] else 'off'}", ("--speculative-config" in a) == b["mtp"])
    check(f"boot {key}: max-num-seqs {b['want_seqs']}", f["--max-num-seqs"] == str(b["want_seqs"]), f["--max-num-seqs"])
print("CONFIG DIFF vs RC2 FP8 (flags [RC2 FP8, boot], env [RC2 FP8, boot], kv pin):")
for key in ns["ORDER"]:
    print(f"  {key} ({BOOTS[key]['title']}): {json.dumps(table[key])}")
check("D = RC2 FP8: no flag / env / pin diff", table["D"] == {"flags": {}, "env": {}})
check("B = D minus --speculative-config only", list(table["B"]["flags"]) == ["--speculative-config"] and table["B"]["env"] == {})
check("E = D plus the byte-decoder env + its own compile cache only", table["E"]["flags"] == {} and
      sorted(table["E"]["env"]) == ["NB0_QSA_FORCE_BYTE_DECODER", "VLLM_CACHE_ROOT"])
check("A vs B: KV dtype only (its flag + Merlin's per-dtype KV pin + the FP8 scales env)", sorted(set(table["A"]["flags"]) ^ set(table["B"]["flags"])) == ["--kv-cache-dtype", "--kv-cache-memory-bytes"]
      and table["A"]["flags"]["--kv-cache-dtype"] == ["fp8_e4m3", "bfloat16"] and "--max-num-seqs" not in table["A"]["flags"]
      and "--compilation-config" not in table["A"]["flags"], json.dumps(table["A"]["flags"]))
check("C = RC2 BF16 (diff vs RC2 BF16 empty)", ns["config_diff"](nb0_argv(plans["bf16"], BOOTS["C"]), RC2_REF["bf16"]["env"],
                                                                    2464415744, "bf16") == {"flags": {}, "env": {}})
try:
    nb0_argv(plans["fp8"] + ["--speculative-config", "{}"], BOOTS["B"])
    check("two --speculative-config -> ValueError", False)
except ValueError:
    check("two --speculative-config -> ValueError", True)
try:
    nb0_argv([x for x in plans["fp8"] if x != "--enable-prefix-caching"], BOOTS["D"])
    check("prefix caching missing -> ValueError", False)
except ValueError:
    check("prefix caching missing -> ValueError", True)
# vLLM itself: --speculative-config absent -> create_speculative_config returns None -> num_speculative_tokens 0, and
# Qwen4Exp + --use-replayssm then takes the stock GDN path with a warning (config/vllm.py:3397-3405, read in source)
vsrc = (MERLIN_SRC / "vllm/config/vllm.py").read_text(encoding="utf-8")
check("Merlin source: Qwen4Exp MTP-off warning text is what boot_facts looks for",
      "needs speculative decoding; num_speculative_tokens=0, so the \"\n                \"stock GDN path is used" in vsrc
      or "num_speculative_tokens=0, so the stock GDN path is used" in vsrc.replace('"\n                "', ""))

# ---------------------------------------------------------------- 4. boot E patch: audit, apply, semantics, restore, refusals
QP = ns["QSA_PATCH"]
ops_src = (MERLIN_SRC / QP["ops_rel"]).read_text(encoding="utf-8")
qsa_src = (MERLIN_SRC / QP["qsa_rel"]).read_text(encoding="utf-8")


class Cap:
    def __init__(self, v): self.v = v
    def to_int(self): return self.v
    def as_version_str(self): return f"{self.v // 10}.{self.v % 10}"


def fn_src(txt, name, deco=""):
    i = txt.index(deco + f"def {name}(") if deco else txt.index(f"def {name}(")
    j = txt.index("\n\n\n", i)
    return txt[i:j]


def native_fn(patched, cap):
    txt = ops_src.replace(QP["ops_old"], QP["ops_new"]) if patched else ops_src
    g = {"lru_cache": __import__("functools").lru_cache, "current_platform": type("P", (), {"get_device_capability": staticmethod(lambda: Cap(cap))})}
    exec(fn_src(txt, "qsa_e4m3_uses_native_cast", "@lru_cache(maxsize=1)\n"), g)
    return g["qsa_e4m3_uses_native_cast"]


def validated(patched, cap, native):
    txt = qsa_src.replace(QP["qsa_old"], QP["qsa_new"]) if patched else qsa_src
    plat = type("P", (), {"is_cuda": staticmethod(lambda: True), "get_device_capability": staticmethod(lambda: Cap(cap)),
                          "get_device_name": staticmethod(lambda: "L4"), "fp8_dtype": staticmethod(lambda: "float8_e4m3fn")})
    # KIT-4: Py3.12 evaluates the `-> torch.dtype | None` annotation at def time (Py3.14 defers it): a torch stub
    # whose `dtype` is a class (so `torch.dtype | None` is a valid union) instead of None
    g = {"current_platform": plat, "qsa_e4m3_uses_native_cast": lambda: native, "torch": type("TorchStub", (), {"dtype": type("dtype", (), {})}),
         "_is_qsa_fp8_cache_dtype": lambda d: d in ("fp8", "fp8_e4m3")}
    exec(fn_src(txt, "_validated_qsa_fp8_dtype"), g)
    try:
        return g["_validated_qsa_fp8_dtype"]("fp8_e4m3")
    except ValueError as e:
        return "RAISE"


os.environ.pop("NB0_QSA_FORCE_BYTE_DECODER", None)
check("unpatched native_cast on sm_89 -> True", native_fn(False, 89)() is True)
check("patched native_cast, env unset, sm_89 -> True (boots D/C/B/A unchanged)", native_fn(True, 89)() is True)
check("patched validation, env unset, native True -> admits", validated(True, 89, True) == "float8_e4m3fn")
check("UNPATCHED validation with native False on sm_89 -> RAISES (why qsa.py:231 needs the patch)", validated(False, 89, False) == "RAISE")
os.environ["NB0_QSA_FORCE_BYTE_DECODER"] = "1"
check("patched native_cast, env set, sm_89 -> False (byte decoder)", native_fn(True, 89)() is False)
check("patched validation, env set, native False, sm_89 -> admits", validated(True, 89, False) == "float8_e4m3fn")
check("patched validation, env set, sm_80 -> still refused", validated(True, 80, False) == "RAISE")
os.environ.pop("NB0_QSA_FORCE_BYTE_DECODER", None)
check("patched validation, env unset, native False, sm_89 -> refused (gate only under the env)", validated(True, 89, False) == "RAISE")


def make_tree(root, mutate=None, list_qsa=False):
    shutil.rmtree(root, ignore_errors=True)
    for rel, txt in ((QP["ops_rel"], ops_src), (QP["qsa_rel"], qsa_src)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((mutate(rel, txt) if mutate else txt).encode("utf-8"))
    (root / ".build-products.sha256").write_text(("0" * 64 + "  " + QP["qsa_rel"] + "\n") if list_qsa else ("0" * 64 + "  vllm/_C.abi3.so\n"))
    return root


class SP:                                       # subprocess shim: `<venv>/bin/python` -> this interpreter
    PIPE = subprocess.PIPE

    @staticmethod
    def run(cmd, **kw):
        if str(cmd[0]).endswith("/bin/python"):
            cmd = [sys.executable] + list(cmd[1:])
        return subprocess.run(cmd, **kw)


real_sp, real_dev = ns["subprocess"], ns["byte_decoder_device_check"]
ns["subprocess"] = SP
tree = make_tree(WORK / "tree")
ns["MERLIN_ENG"] = {"kv": {"MERLIN_TREE": str(tree), "MERLIN_VENV": "/x/venv"}, "cuda_home": ""}
ns["byte_decoder_device_check"] = lambda txt: {"ok": True, "unit": True, "has_decoder": "_decode_e4m3_to_bf16" in txt}
pre = ns["byte_decoder_prepare"]()
check("byte_decoder_prepare on the audited source -> ok, both files patched", pre.get("ok") is True and len(pre.get("patched", [])) == 2, pre.get("why"))
check("audit: installed sha == audited sha", all(v["sha_matches_audited"] for v in pre["audit"].values()))
pt = (tree / QP["ops_rel"]).read_text(encoding="utf-8"), (tree / QP["qsa_rel"]).read_text(encoding="utf-8")
check("patched files = source + exactly one marker each", pt[0] == ops_src.replace(QP["ops_old"], QP["ops_new"]) and
      pt[1] == qsa_src.replace(QP["qsa_old"], QP["qsa_new"]) and all(x.count(QP["marker"]) == 1 for x in pt))
rs = ns["byte_decoder_restore"]()
check("restore -> original bytes (sha == audited)", rs.get("ok") and hashlib.sha256((tree / QP["ops_rel"]).read_bytes()).hexdigest() == QP["ops_sha"]
      and hashlib.sha256((tree / QP["qsa_rel"]).read_bytes()).hexdigest() == QP["qsa_sha"])
make_tree(tree, mutate=lambda rel, t: t.replace("        if qsa_e4m3_uses_native_cast():\n            warm_kv_dtype = 3\n", "        warm_kv_dtype = 3\n"))
pre = ns["byte_decoder_prepare"]()
check("a changed call site in the installed tree -> skip E (unsafe)", pre.get("ok") is False and "differ" in (pre.get("why") or ""), pre.get("why"))
check("... and nothing was patched", QP["marker"] not in (tree / QP["ops_rel"]).read_text(encoding="utf-8"))
make_tree(tree, list_qsa=True)
pre = ns["byte_decoder_prepare"]()
check("patched file listed as a verified build product -> skip E", pre.get("ok") is False and "build products" in (pre.get("why") or ""), pre.get("why"))
make_tree(tree)
ns["byte_decoder_device_check"] = lambda txt: {"ok": False, "n_bad": 3, "bad_codepoints": [1, 2, 3]}
pre = ns["byte_decoder_prepare"]()
check("on-device decoder check fails -> skip E, tree untouched", pre.get("ok") is False and QP["marker"] not in (tree / QP["qsa_rel"]).read_text(encoding="utf-8"), pre.get("why"))
m = __import__("re").search(r"@triton\.jit\ndef _decode_e4m3_to_bf16\(b\):\n(?:(?:    [^\n]*)?\n)+", ops_src)
check("device check extracts the installed decoder body (ends before the next kernel)", m and "return decoded.to(tl.bfloat16)" in m.group(0)
      and "_qsa_sparse_paged_gqa_splitk_kernel" not in m.group(0))
dt = ns["decoder_script"](m.group(0))
compile(dt, "decoder_test", "exec")
check("device-check script compiles", True)
class SPBoom:                                  # py_compile cannot even start: the tree must come back unpatched
    PIPE = subprocess.PIPE

    @staticmethod
    def run(cmd, **kw):
        raise FileNotFoundError(cmd[0])


make_tree(tree)
ns["byte_decoder_device_check"] = lambda txt: {"ok": True}
ns["subprocess"] = SPBoom
pre = ns["byte_decoder_prepare"]()
check("py_compile cannot run -> skip E and the tree is restored (no patched file left)", pre.get("ok") is False
      and (pre.get("restore") or {}).get("ok") is True and QP["marker"] not in (tree / QP["ops_rel"]).read_text(encoding="utf-8")
      and hashlib.sha256((tree / QP["ops_rel"]).read_bytes()).hexdigest() == QP["ops_sha"], pre.get("why"))

# ---------------------------------------------------------------- 4b. A-1 / A-2 / A-3: exact sha, verified restore, fatal restore failure
BYTE, ABORT, R_ = ns["BYTE"], ns["ABORT"], ns["R"]
OPS_P, QSA_P = tree / QP["ops_rel"], tree / QP["qsa_rel"]
bak_of = lambda p: p.with_name(p.name + ".nb0_orig")
sha_of = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tree_clean = lambda: (sha_of(OPS_P) == QP["ops_sha"] and sha_of(QSA_P) == QP["qsa_sha"] and not bak_of(OPS_P).exists()
                      and not bak_of(QSA_P).exists())
ns["subprocess"] = SP
ns["byte_decoder_device_check"] = lambda txt: {"ok": True}
for kk in ("ops", "qsa"):                     # one added comment line: anchors and name counts still match, the sha does not
    BYTE.clear()
    make_tree(tree, mutate=lambda rel, t, kk=kk: t + "# an unreviewed comment\n" if rel == QP[kk + "_rel"] else t)
    before = {p: p.read_bytes() for p in (OPS_P, QSA_P)}
    pre = ns["byte_decoder_prepare"]()
    a = (pre.get("audit") or {}).get(kk) or {}
    check(f"A-1: {kk} sha differs with every anchor matching -> UNREVIEWED_SOURCE_SHA, nothing patched",
          pre.get("ok") is False and pre.get("status") == "UNREVIEWED_SOURCE_SHA" and a.get("anchors_failed") == 0
          and a.get("sha_matches_audited") is False and set(pre["audit"]) == {"ops", "qsa"}
          and all(p.read_bytes() == before[p] for p in before) and not bak_of(OPS_P).exists() and not bak_of(QSA_P).exists()
          and not BYTE.get("applied"), pre.get("why"))
ABORT["why"] = None
o = ns["run_boot"]("E")
check("A-1: run_boot(E) on an unreviewed sha -> status UNREVIEWED_SOURCE_SHA with the audit, no ABORT (B / A still run)",
      o.get("status") == "UNREVIEWED_SOURCE_SHA" and "audit" in (o.get("byte_decoder") or {}) and ABORT["why"] is None, o.get("status"))

BYTE.clear()
make_tree(tree)
pre = ns["byte_decoder_prepare"]()
bak_of(OPS_P).unlink()                         # the ops backup is lost while boot E runs
rs1 = ns["byte_decoder_restore"]()
check("A-2: missing backup -> restore FAIL, that file stays pending, the other one restored + verified",
      pre.get("ok") is True and rs1.get("ok") is False and rs1.get("pending") == [str(OPS_P)] and BYTE.get("applied") == [str(OPS_P)]
      and rs1["files"][str(QSA_P)].get("restored") is True and sha_of(QSA_P) == QP["qsa_sha"] and QP["marker"] in OPS_P.read_text(encoding="utf-8"),
      json.dumps(rs1)[:300])
rs2 = ns["byte_decoder_restore"]()
check("A-2: a second restore call still FAILS (the pending list was kept)", rs2.get("ok") is False and rs2.get("pending") == [str(OPS_P)]
      and BYTE.get("applied") == [str(OPS_P)], json.dumps(rs2)[:300])
pre2 = ns["byte_decoder_prepare"]()
check("A-2: prepare refuses a new patch while one is still pending", pre2.get("ok") is False and "pending" in (pre2.get("why") or ""), pre2.get("why"))
bak_of(OPS_P).write_bytes(ops_src.encode("utf-8"))
rs3 = ns["byte_decoder_restore"]()
check("A-2: the backup is back -> restore ok, verified against the audited sha, nothing pending", rs3.get("ok") is True
      and rs3.get("pending") == [] and not BYTE.get("applied") and tree_clean(), json.dumps(rs3)[:300])
BYTE.clear()
make_tree(tree)
ns["byte_decoder_prepare"]()
bak_of(QSA_P).write_bytes(b"# not the original\n")      # a corrupted backup must not be written back as "restored"
rs4 = ns["byte_decoder_restore"]()
check("A-2: corrupted backup (sha != audited) -> FAIL, file stays pending and patched", rs4.get("ok") is False
      and rs4.get("pending") == [str(QSA_P)] and QP["marker"] in QSA_P.read_text(encoding="utf-8"), json.dumps(rs4)[:300])

# A-3: restore failure after boot E -> ABORT (QSA_PATCH_RESTORE_FAILED), later Merlin boots blocked
BYTE.clear()
make_tree(tree)
ABORT["why"] = None
real_mb, real_body = ns["merlin_boot"], ns["nb0_body"]
seen = []


def mb_lose_backup(*a, **k):
    seen.append(a[0])
    bak_of(OPS_P).unlink()
    return {"status": "DONE"}


ns["merlin_boot"], ns["nb0_body"] = mb_lose_backup, (lambda key: None)
oE = ns["step"]("NB0", BOOTS["E"]["step"], 1, lambda: ns["run_boot"]("E"))
check("A-3: restore FAIL after boot E -> ABORT QSA_PATCH_RESTORE_FAILED, boot E recorded PARTIAL_HARD_STOP",
      (ABORT["why"] or "").startswith("QSA_PATCH_RESTORE_FAILED") and oE.get("status") == "PARTIAL_HARD_STOP"
      and ((oE.get("byte_decoder") or {}).get("restore") or {}).get("ok") is False and len(seen) == 1, f"{ABORT['why']} | {oE.get('status')}")
seen.clear()
ns["merlin_boot"] = lambda *a, **k: (seen.append(a[0]), {"status": "DONE"})[1]
oB = ns["step"]("NB0", BOOTS["B"]["step"], 1, lambda: ns["run_boot"]("B"))
check("A-3: ... the next Merlin boot (B) is SKIPPED_ABORT and never launched", oB.get("status") == "SKIPPED_ABORT" and not seen, oB.get("status"))
bak_of(OPS_P).write_bytes(ops_src.encode("utf-8"))
ns["byte_decoder_restore"]()

# A-3 atomic: a write that fails half-way through the 2nd file -> both files restored from their verified backups
RealPath = ns["Path"]


class BoomPath(type(Path())):
    mode = "write"                             # "write": the patched qsa.py write fails half-way; "all": every write to it fails

    def write_bytes(self, data):
        if self.name == "qsa.py" and self.parent.name != "ops" and (BoomPath.mode == "all" or QP["marker"].encode() in data):
            super().write_bytes(data[:100])
            raise OSError("injected: disk full")
        return super().write_bytes(data)


BYTE.clear()
make_tree(tree)
ABORT["why"] = None
ns["Path"] = BoomPath
pre = ns["byte_decoder_prepare"]()
ns["Path"] = RealPath
check("A-3: write exception mid-patch (2nd file, half-written) -> skip E, every touched file restored + verified, nothing pending",
      pre.get("ok") is False and "patch step failed" in (pre.get("why") or "") and (pre.get("restore") or {}).get("ok") is True
      and tree_clean() and not BYTE.get("applied") and len((pre.get("restore") or {}).get("files") or {}) == 2, pre.get("why"))
BYTE.clear()
make_tree(tree)
ABORT["why"] = None
ns["Path"], BoomPath.mode = BoomPath, "all"
seen.clear()
oE = ns["run_boot"]("E")
ns["Path"], BoomPath.mode = RealPath, "write"
check("A-3: patch fails AND its restore fails -> boot E skipped, ABORT QSA_PATCH_RESTORE_FAILED, no server launched",
      oE.get("status") == "SKIPPED_BYTE_DECODER_UNSAFE" and (ABORT["why"] or "").startswith("QSA_PATCH_RESTORE_FAILED") and not seen
      and BYTE.get("applied") == [str(QSA_P)], f"{oE.get('status')} | {ABORT['why']}")
ns["merlin_boot"], ns["nb0_body"] = real_mb, real_body
ABORT["why"] = None
BYTE.clear()
for st_ in (BOOTS["E"]["step"], BOOTS["B"]["step"]):
    (R_["sections"].get("NB0") or {}).pop(st_, None)
make_tree(tree)
ns["subprocess"], ns["byte_decoder_device_check"] = real_sp, real_dev
ns["FAKE"] = "x"
check("FAKE: byte_decoder_prepare does not touch any tree", ns["byte_decoder_prepare"]().get("fake") is True)
ns["FAKE"] = None

# ---------------------------------------------------------------- 5. /proc session scan + reap (synthetic /proc; the kill hook never signals real pids)
PROC = WORK / "proc"


def proc(pid, comm, state, session, majflt=0, rss=1000, rb=0):
    d = PROC / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    f = [state, "1", str(session), str(session), "0", "-1", "4194304", "100", "0", str(majflt), "0", "50", "20", "0", "0",
         "20", "0", "8", "0", "123", "999", str(rss)] + ["0"] * 30
    (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(f) + "\n")
    (d / "io").write_text(f"rchar: {rb * 2}\nwchar: 0\nsyscr: 5\nsyscw: 0\nread_bytes: {rb}\nwrite_bytes: 0\n")
    (d / "cmdline").write_bytes(f"/venv/bin/python -m {comm}".encode() + b"\0")


proc(4444, "VLLM::Worker_PP0", "S", 4000, majflt=10, rb=5 << 30)
proc(4445, "python3 (with space)", "R", 4000, majflt=20)
proc(4446, "zombie", "Z", 4000)
proc(5000, "other", "S", 5000)
ns["SIDS"].clear()
ns["SIDS"].add(4000)
sp = ns["session_procs"]({4000})
check("session scan: live members of the server session only (zombie, other session excluded)", sorted(p["pid"] for p in sp) == [4444, 4445])
check("stat parse: comm with spaces / parens, majflt, read_bytes", any(p["comm"] == "python3 (with space)" and p["majflt"] == 20 for p in sp)
      and any(p.get("read_bytes") == 5 << 30 for p in sp))
killed = []
ns["_KILL"] = lambda pid, sig: (killed.append((pid, sig)), shutil.rmtree(PROC / str(pid)))
r = ns["reap_leftovers"](wait=2)
check("reap: SIGKILL to the leftovers, then ok", r["ok"] and sorted(killed) == [(4444, 9), (4445, 9)], json.dumps(r)[:200])
proc(4447, "stuck", "D", 4000)
ns["_KILL"] = lambda pid, sig: None
r = ns["reap_leftovers"](wait=1)
check("reap: a process that survives SIGKILL -> ok False (fail closed)", r["ok"] is False and r["still"] == [[4447, "stuck"]])
shutil.rmtree(PROC / "4447")

# ---------------------------------------------------------------- 6. cgroup snapshot + host gate (synthetic cgroup v2)
CGD = WORK / "cgroup"
MiB = 1 << 20


def cgroup(current_mib, max_mib, inactive_mib, anon_mib=1000):
    (CGD / "memory.current").write_text(str(current_mib * MiB))
    (CGD / "memory.max").write_text(str(max_mib * MiB))
    (CGD / "memory.stat").write_text(f"anon {anon_mib * MiB}\nfile {(current_mib - anon_mib) * MiB}\nactive_file {(current_mib - anon_mib - inactive_mib) * MiB}\n"
                                     f"inactive_file {inactive_mib * MiB}\nshmem 0\npgmajfault 77\n")
    (CGD / "memory.events").write_text("low 0\nhigh 0\nmax 12\noom 0\noom_kill 0\n")
    (CGD / "memory.pressure").write_text("some avg10=1.50 avg60=0.10 avg300=0.00 total=1\nfull avg10=0.75 avg60=0.0 avg300=0.0 total=1\n")


cgroup(109735, 177152, 3196)                         # RC2's FP8 boot gate: 108 GB of ACTIVE checkpoint cache
s = ns["cg_snapshot"]()
check("cg_snapshot: Merlin's rule max - (current - inactive_file) = RC2's 70,613 MiB", s["avail_mib"] == 177152 - (109735 - 3196), s["avail_mib"])
check("cg_snapshot: stat / events / pressure recorded", s["stat_mib"]["inactive_file"] == 3196 and s["stat_count"]["pgmajfault"] == 77
      and "oom_kill" in s["memory.events"] and "full avg10" in s["memory.pressure"])
t = time.time()
hp = ns["host_prep"]()
check("host gate: RC2's FP8-boot state (68.96 GiB < 73 GiB) -> ok False", hp["ok"] is False and hp["host_gate"]["avail_mib"] < 74752
      and "host memory" in hp["why"], f"{hp['host_gate']} {time.time() - t:.0f}s")
check("host gate: a second drop + pause was tried before failing", "fadvise_2" in hp)
cgroup(5000, 177152, 2000)
hp = ns["host_prep"]()
check("host gate: after a page-cache drop (174 GiB avail) -> ok True", hp["ok"] is True and hp["host_gate"]["source"] == "cgroup", hp["host_gate"])
cgroup(5000, 177152, 2000)
(CGD / "memory.max").write_text("max")
hp = ns["host_prep"]()
check("host gate: no cgroup limit on a real box -> MemAvailable or fail closed", hp["host_gate"]["source"] == "MemAvailable"
      and (hp["ok"] is False or hp["host_gate"]["avail_mib"] is not None))
ns["FAKE"] = "x"
cgroup(109735, 177152, 3196)
hp = ns["host_prep"]()
check("FAKE harness: a failing host gate is recorded (fake_not_applied), never blocks the fake server", hp["ok"] is True
      and hp["host_gate"].get("fake_not_applied") is True and hp["host_gate"]["avail_mib"] == 70613)
ns["FAKE"] = None
proc(4448, "leftover", "S", 4000)
cgroup(5000, 177152, 2000)
ns["_KILL"] = lambda pid, sig: None
ns["reap_leftovers"].__defaults__ = (1,)
hp = ns["host_prep"]()
check("host prep: a leftover server process -> ok False, psm cleanup skipped", hp["ok"] is False and "skipped" in hp["shm"])
shutil.rmtree(PROC / "4448")
(WORK / "shm" / "psm_dead1").write_text("x")
(WORK / "shm" / "torch_keep").write_text("x")
r = ns["unlink_own_psm"]()
check("psm cleanup: only psm_* that appeared after NB0 started", r["removed"] == ["psm_dead1"] and (WORK / "shm" / "torch_keep").exists())
bg = ns["boot_gate"]()
check("boot_gate = host prep + RC2's GPU gate; no nvidia-smi here -> not ok (callers launch only on ok True)", "host_prep" in bg and bg["ok"] is not True, str(bg["ok"]))

# ---------------------------------------------------------------- 7. raw teacher-forced arrays, the RC2-comparable metric, full arrays
rng = np.random.default_rng(0)


def fake_pl(ids, k=5, noise=0.0, nan_at=None, drop=None, amb=None, shift=0, key_style="token_id"):
    pl = [None]
    for p in range(1, len(ids)):
        if drop is not None and p == drop:
            pl.append(None)
            continue
        base = (ids[p - 1 + shift] if 1 <= p - 1 + shift < len(ids) else 0) * 7 % 1000 + 10
        cands = [base + j for j in range(k)]
        lps = [-0.1 - 0.5 * j for j in range(k)]
        key = (lambda t: f"token_id:{t}") if key_style == "token_id" else str
        d = {key(c): {"logprob": lps[j] + noise * ((p * 13 + j) % 7) / 7, "rank": j + 1} for j, c in enumerate(cands)}
        tok = ids[p]
        lp = -0.1 if tok == cands[0] else -6.0 - (tok % 5) * 0.1
        if nan_at is not None and p == nan_at:
            lp = float("nan")
        if tok in cands[1:]:                          # the actual token is a listed candidate: its own rank (as vLLM)
            lp = lps[cands.index(tok)]
        d[key(tok)] = {"logprob": lp + noise * (p % 5) / 5, "rank": cands.index(tok) + 1 if tok in cands else 30}
        if amb is not None and p == amb:
            d[key(cands[1])]["rank"] = 1
        pl.append(d)
    return pl


ids = [int(x) for x in rng.integers(10, 2000, 3000)]
pa = ns["pl_arrays"]
A0 = pa(fake_pl(ids), ids, 5)
check("pl_arrays: every position valid, top-k filled, ids kept", A0["valid"][1:].all() and not A0["valid"][0] and (A0["topk_ids"][1:, 4] >= 0).all()
      and list(A0["ids"]) == ids)
A1 = pa(fake_pl(ids, key_style="plain"), ids, 5)
check("pl_arrays: '123' and 'token_id:123' keys give the same arrays", np.array_equal(A0["lp"][1:], A1["lp"][1:]) and np.array_equal(A0["top1"], A1["top1"]))
Aamb = pa(fake_pl(ids, amb=2000), ids, 5)
check("pl_arrays: an ambiguous rank 1 -> top1 -1 (E23's rule)", Aamb["top1"][2000] == -1 and Aamb["top1_ambiguous"] == 1)


def run(arr, meta_sha="x"):
    return {"meta": {"prompt_sha256": meta_sha}, "arr": arr}


c = ns["cmp_last256"](run(A0), run(A1))
check("cmp_last256 (RC2's compare_tf): identical -> MEASURED 0 / 1.0", c["status"] == "MEASURED" and c["mean_abs_dlogprob"] == 0 and c["rank1_agreement"] == 1.0, c)
An = pa(fake_pl(ids, nan_at=2990), ids, 5)
check("cmp_last256: a NaN in the window -> INCONCLUSIVE_NONFINITE (RC2's guard)", ns["cmp_last256"](run(A0), run(An))["status"] == "INCONCLUSIVE_NONFINITE")
Ad = pa(fake_pl(ids, drop=2995), ids, 5)
check("cmp_last256: a missing position -> INCONCLUSIVE_INCOMPLETE_TF", ns["cmp_last256"](run(A0), run(Ad))["status"] == "INCONCLUSIVE_INCOMPLETE_TF")
check("cmp_last256: different prompt -> INCONCLUSIVE_PROMPT_MISMATCH", ns["cmp_last256"](run(A0, "a"), run(A1, "b"))["status"] == "INCONCLUSIVE_PROMPT_MISMATCH")
check("cmp_last256: missing run -> INCONCLUSIVE_MISSING_TF", ns["cmp_last256"](None, run(A1))["status"] == "INCONCLUSIVE_MISSING_TF")
f = ns["cmp_full"](A0, An)
check("cmp_full: a NaN anywhere -> INCONCLUSIVE_NONFINITE", f["status"] == "INCONCLUSIVE_NONFINITE")
Ag = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in A0.items()}
Ag["lp"][2048:] -= 0.6                                # divergence only past 2048
f = ns["cmp_full"](A0, Ag)
check("cmp_full: context bins show where it starts; first window over 0.15 near 2048", f["status"] == "MEASURED" and f["bins_4096"][0][2] > 0
      and 1792 <= f["first_window_over_screen_mean"] <= 2048, json.dumps({k: f[k] for k in ("bins_4096", "first_window_over_screen_mean")}))
Ac = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in A0.items()}
pos = np.arange(len(ids))
Ac["lp"][(pos % 1024) < 4] -= 1.0                     # divergence only at the 1024-token chunk starts
f = ns["cmp_full"](A0, Ac)
check("chunk check: divergence at the 1024 chunk edges -> localised", f["chunk"]["1024"]["localised"] is True, json.dumps(f["chunk"]))
check("chunk check: uniform divergence -> not localised", not any(v["localised"] for v in ns["cmp_full"](A0, Ag)["chunk"].values() if v["mid_mean"] > 0.3))
As = pa(fake_pl(ids, shift=1), ids, 5)
sh = ns["shift_check"](A0, As)
check("shift check: candidate scored one position late -> alignment artefact", sh["alignment_artefact"] and sh["best_shift"] in (-1, 1), json.dumps(sh)[:300])
sh0 = ns["shift_check"](A0, A1)
check("shift check: identical rows -> best shift 0, no artefact", sh0["best_shift"] == 0 and not sh0["alignment_artefact"])

# ---------------------------------------------------------------- 8. decode: self-consistency, cold vs cached, cross-boot parity
P, N = 2000, 64
gen = [int(x) for x in rng.integers(10, 2000, N)]
full = ids[:P] + gen
T = pa(fake_pl(full), full, 5)


def dec_from(T, P, N, dl=0.0, toks=None, top2=-3.0, err=None):
    toks = toks or [int(x) for x in T["ids"][P:P + N]]
    return {"meta": {"error": err}, "tok": toks, "lp": T["lp"][P:P + N].astype(np.float64) + dl,
            "top1": T["top1"][P:P + N].astype(np.int64), "top1_lp": T["top1_lp"][P:P + N].astype(np.float64),
            "top2_lp": np.full(N, top2)}


D0 = dec_from(T, P, N)
s0 = ns["sc_compare"](D0, {"arr": T}, P)
check("sc_compare: decode == teacher-forced -> MEASURED 0, CLEAN", s0["status"] == "MEASURED" and s0["mean_abs_dlogprob"] == 0 and s0["screen"] == "CLEAN", s0)
s1 = ns["sc_compare"](dec_from(T, P, N, dl=0.5), {"arr": T}, P)
check("sc_compare: decode 0.5 nats off -> DIVERGENT (prompt-logprob path artefact signal)", s1["screen"] == "DIVERGENT")
check("sc_compare: tokens not in the TF prompt -> PROMPT_MISMATCH", ns["sc_compare"](dec_from(T, P, N, toks=[1] * N), {"arr": T}, P)["status"] == "INCONCLUSIVE_PROMPT_MISMATCH")
check("sc_compare: decode error -> INCONCLUSIVE", ns["sc_compare"](dec_from(T, P, N, err="HTTP 500"), {"arr": T}, P)["status"] == "INCONCLUSIVE_MISSING")
try:                                            # A-4: 0 / 1 / short TF / 256 tokens
    r = ns["sc_compare"](dec_from(T, P, 0), {"arr": T}, P)
    check("A-4: empty decode -> INCONCLUSIVE_EMPTY_DECODE (no IndexError)", r["status"] == "INCONCLUSIVE_EMPTY_DECODE", r)
except Exception as e:
    check("A-4: empty decode -> INCONCLUSIVE_EMPTY_DECODE (no IndexError)", False, repr(e))
r = ns["sc_compare"](dec_from(T, P, 1), {"arr": T}, P)
check("A-4: 1-token decode -> MEASURED n 1, CLEAN", r["status"] == "MEASURED" and r["n"] == 1 and r["screen"] == "CLEAN", r)
Tsh = pa(fake_pl(full[:P + N // 2]), full[:P + N // 2], 5)
r = ns["sc_compare"](D0, {"arr": Tsh}, P)
check("A-4: TF shorter than prompt + decode -> INCONCLUSIVE_TF_TOO_SHORT", r["status"] == "INCONCLUSIVE_TF_TOO_SHORT"
      and r["need_len"] == P + N and r["tf_len"] == P + N // 2, r)
check("A-4: decode ending exactly at the TF end -> MEASURED (not too short)", s0["status"] == "MEASURED" and len(T["ids"]) == P + N)
gen256 = [int(x) for x in rng.integers(10, 2000, 256)]
T256 = pa(fake_pl(ids[:P] + gen256), ids[:P] + gen256, 5)
r = ns["sc_compare"](dec_from(T256, P, 256), {"arr": T256}, P)
check("A-4: full 256-token decode -> MEASURED n 256, CLEAN", r["status"] == "MEASURED" and r["n"] == 256 and r["screen"] == "CLEAN", r)
dc = ns["decode_compare"]
check("decode_compare: identical -> EXACT", dc(D0, dec_from(T, P, N))["status"] == "EXACT")
D1 = dec_from(T, P, N)
D1["tok"] = D1["tok"][:10] + [9] + D1["tok"][11:]
D0hi = dec_from(T, P, N, top2=-5.0)
r = dc(D0hi, D1)
check("decode_compare: diff at 10 with reference margin >= 0.5 -> FAIL_HIGH_MARGIN", r["status"] == "FAIL_HIGH_MARGIN" and r["first_diff"] == 10, r)
D0lo = dec_from(T, P, N)
D0lo["top2_lp"] = D0lo["top1_lp"] - 0.1
r = dc(D0lo, D1)
check("decode_compare: diff at a 0.1 margin -> NEAR_TIE", r["status"] == "NEAR_TIE" and abs(r["ref_margin_at_diff"] - 0.1) < 1e-6, r)

# ---------------------------------------------------------------- 9. decision table + verdict on synthetic analyses
def an_with(screens, aa_div=None, sc_div=None, parity=None):
    pairs = {pk: {"what": pk, "bands": {b: {"screen": s} for b, s in bands.items()}, "parity": (parity or {}).get(pk, {})}
             for pk, bands in screens.items()}
    pb = {}
    for k in "ABCDE":
        pb[k] = {"aa": {b: (rc2 if aa_div == (k, b) else clean) for b in ("2k", "8k", "40k", "90k")},
                 "sc": {"40k": {"decode_vs_tf": {"screen": "DIVERGENT" if sc_div == k else "CLEAN"}, "cold_vs_cached": {"status": "EXACT"}}},
                 "prefix_consistency": {"consistent": True}, "scales": None}
    return {"pairs": pairs, "per_boot": pb}


ALL = ("2k", "8k", "40k", "90k")
Cl = {b: "CLEAN" for b in ALL}
names = lambda an: {x["finding"] for x in ns["decide"](an)}
fs = names(an_with({"AB": Cl, "CD": {b: "DIVERGENT" for b in ALL}, "CE": Cl, "AC": Cl, "BD": Cl, "DE": {b: "DIVERGENT" for b in ALL}}))
check("decide: C-D divergent from 2k, A-B clean, C-E clean -> KV_FROM_2K + MTP_ONLY + NATIVE_SM89 defect",
      {"KV_DIVERGENCE_FROM_2K", "MTP_ONLY", "NATIVE_SM89_E4M3_PATH_DEFECT", "READ_PATH_CHANGES_LOGITS"} <= fs, sorted(fs))
fs = names(an_with({"AB": {"2k": "CLEAN", "8k": "CLEAN", "40k": "DIVERGENT", "90k": "DIVERGENT"}, "CD": Cl, "CE": Cl}))
check("decide: divergent only at 40k / 90k -> KV_DIVERGENCE_LONG_ONLY", "KV_DIVERGENCE_LONG_ONLY" in fs and "MTP_ONLY" not in fs, sorted(fs))
fs = names(an_with({"AB": Cl, "CD": {b: "DIVERGENT" for b in ALL}, "CE": {b: "DIVERGENT" for b in ALL}}))
check("decide: C-D and C-E both divergent -> not the native cast", "FP8_DIVERGENCE_NOT_THE_NATIVE_CAST" in fs and "NATIVE_SM89_E4M3_PATH_DEFECT" not in fs)
fs = names(an_with({"AB": Cl, "CD": Cl, "CE": Cl}, aa_div=("D", "40k")))
check("decide: only the repeat differs -> REPEAT_DIVERGENCE (cache / state)", "REPEAT_DIVERGENCE" in fs)
fs = names(an_with({"AB": Cl, "CD": Cl, "CE": Cl}, sc_div="D"))
check("decide: decode vs prompt logprobs disagree on D -> path artefact finding", "DECODE_VS_PROMPT_LOGPROB_MISMATCH" in fs)
fs = names(an_with({"AB": Cl, "CD": Cl, "CE": Cl, "BD": {"2k": "DIVERGENT"}}, parity={"BD": {"40k": {"status": "FAIL_HIGH_MARGIN"}}}))
check("decide: MTP changes FP8 target logits + high-margin greedy split -> both findings",
      {"MTP_CHANGES_TARGET_LOGITS_FP8", "MTP_GREEDY_DIVERGENCE_HIGH_MARGIN"} <= fs)
fs = names(an_with({"AB": Cl, "CD": Cl, "CE": Cl}))
check("decide: all clean -> KV_CLEAN on every KV pair, no cause finding", fs == {"KV_CLEAN", "RC2_ANCHOR"}, sorted(fs))
fs = names(an_with({"AB": {b: "INCONCLUSIVE" for b in ALL}, "CD": {b: "INCONCLUSIVE" for b in ALL}, "CE": {b: "INCONCLUSIVE" for b in ALL}}))
check("decide: nothing measured -> no KV verdict at all (no false CLEAN)", "KV_CLEAN" not in fs, sorted(fs))
an = {"pairs": {"CD": {"bands": {"2k": {"screening_gate": "SCREENING_PASS", "triage_gate": "PENDING_LONG_ARC_ACCURACY"},
                                 "8k": {"screening_gate": "SCREENING_PASS", "triage_gate": "PENDING_LONG_ARC_ACCURACY"},
                                 "40k": {"screening_gate": "HOLD_FP8_NUMERICAL_DIVERGENCE", "triage_gate": "NO_GO_FP8_TF_DIVERGENCE"}}}}}
v = ns["fp8_verdict"](an)
check("verdict: one NO_GO band -> NO_GO for the production FP8 config", v["fp8_native_production"] == "NO_GO_FP8_TF_DIVERGENCE" and v["CD"]["screening"].startswith("HOLD"))
check("verdict: missing pair -> INCONCLUSIVE, not PASS", v["fp8_byte_decoder"].startswith("INCONCLUSIVE"))

# ---------------------------------------------------------------- 10. boot_facts on RC2's REAL Kaggle server logs
class S:
    def __init__(self, p): self.t = p.read_text(encoding="utf-8", errors="replace")
    def log(self): return self.t


fp8, bf16 = S(RC2_LOGS / "server_merlin_fp8.log"), S(RC2_LOGS / "server_merlin_bf16.log")
f = ns["boot_facts"](fp8, "D")
check("RC2 FP8 log as boot D: config_ok (kv fp8_e4m3, seqs 8, MTP K3 in args + engine, prefix caching)", f["config_ok"], json.dumps(f["checks"]))
sc = f["scales"]
check("RC2 FP8 log: static scales applied on both stages, all 12 file layers logged with the file's values",
      sc["ok"] and sc["matched"] == 12 and sc["applied_stages"] == ["0-25", "25-48"] and sc["applied_counts"] == ["6 of 12"], json.dumps({k: sc[k] for k in ("matched", "mismatched", "missing", "applied_stages")}))
check("RC2 FP8 log: the MTP layer's DEFAULT 1.0 line found", sc["mtp_default_scale"] == ["mtp.layers.48.self_attn.attn"] and sc["not_in_file"] == ["mtp.layers.48.self_attn.attn"])
check("RC2 FP8 log: RecoverSSM path lines + SpecDecoding metrics present", f["recoverssm_path_lines"] > 0 and f["spec_metric_lines"] > 0)
print("scales sidecar sha256", sc["sha256"], "| values", json.dumps(sc["values"])[:300], "...")
check("RC2 FP8 log as boot B (MTP off) -> CONFIG_MISMATCH on mtp", ns["boot_facts"](fp8, "B")["checks"]["mtp"] is False)
check("RC2 FP8 log as boot E -> CONFIG_MISMATCH (no byte-decoder marker)", ns["boot_facts"](fp8, "E")["checks"]["byte_decoder"] is False)
f = ns["boot_facts"](bf16, "C")
check("RC2 BF16 log as boot C: config_ok (bfloat16, seqs 3, MTP K3, no E4M3 lines)", f["config_ok"], json.dumps(f["checks"]))
f = ns["boot_facts"](bf16, "A")
check("RC2 BF16 log as boot A -> mismatch on mtp, seqs, captures", not f["config_ok"] and not f["checks"]["mtp"] and not f["checks"]["seqs"]
      and not f["checks"]["captures"])
check("RC2 BF16 log as boot D -> mismatch on kv", ns["boot_facts"](bf16, "D")["checks"]["kv_args"] is False)
sp_log = S.__new__(S)
sp_log.t = fp8.t.replace("/tmp/fn_merlin/qsa_kv_scales_262k.json", "D:/ARC AGI 2/x y/qsa_kv_scales_262k.json")
check("scales lines parse with a space in the sidecar path", ns["scales_facts"](sp_log.t)["applied_stages"] == ["0-25", "25-48"])
lm = ns["_roles"](fp8.t)
check("worker role map from the real log (pid 4447 = Worker_PP1_TP1_EP1, RC2's 873 s loader)", lm.get(4447) == "Worker_PP1_TP1_EP1")

# ---------------------------------------------------------------- summary
bad = [r for r in RESULTS if not r[1]]
print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS" + ("" if not bad else "; FAILED: " + "; ".join(r[0] for r in bad)))
sys.exit(1 if bad else 0)
