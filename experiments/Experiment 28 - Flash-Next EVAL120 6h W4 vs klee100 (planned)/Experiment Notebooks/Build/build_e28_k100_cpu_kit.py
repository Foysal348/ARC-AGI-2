"""E28 K100 EVAL120 6 h Kaggle CPU kits (lead 2026-11-01): independent CPU kernels on esfersami50 (the klee100 inputs are shared
with that account; K100Q's kits and GPU run were there), the notebook's pinned Py3.12 image, no GPU, no internet, private. Each
kernel is fail-soft, writes its JSON report and prints exactly one line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.
NOTHING is pushed here.

Built as a wrapper over build_e26_k100q_cpu_kit.py (the K100Q kits, ALL OK on Kaggle before the K100Q GPU run; imported by
exec of its source with exact-once retargeting edits, never written): its HEAD, the e2e driver (NB6's e2e on THIS notebook with
the fake server, the real Duck payload, the klee100 path on the real mounts) and the real kit (the real Merlin install, the
klee100 inputs, every boot plan's argv / env through Merlin's own build(), vLLM's argparse, the method-map probe on real Merlin
objects, the Q-defer apply / revert on the installed tree, CELL 0 vs E18 R2) run on the E28 notebook; only the marked E28 edits
below change them. Everything about the notebook under test comes from e28k100_build_facts.json and the notebook itself.

  units       e28k100_unit_tests.py on Py3.12 (all PASS: provenance, profiles, time plan, the edits, the N1 / N2 numbers, the
              e28k100_lib tests, the report on K100Q's REAL GPU output (a reduced fixture of it), the scheduler module on the
              payload's REAL arc2_duck.py extracted from the mounted q27-e15b-payload) + the tf-ref dataset version check + the
              K100Q payload / pinned-file checks (the same dataset version K100Q read; nothing new to push)
  e2e-rerun   modes rerun + rerun-retry: RERUN_PROD on a fake 6-task hidden set, the A / B / C scheduler applied, plain
              order, I8 on the first try; rerun-retry: FAKE_OOM between K3 rung 1 and rung 2 -> try 1 (rung 1, INT8 head) OOM ->
              I8_UNSUPPORTED_ON_K100 (OOM_WITH_INT8_HEAD) -> try 2 (rung 1, BF16 head) OOM -> try 3 (rung 2) boots (E18's <= 3)
  e2e-commit  E28EVAL120 (all 120 eval tasks, the 415-min profile) with a FAKE SHORT DEADLINE (Q27X_TIME_BUDGET = 3,000 s):
              the window shortened (planned 21,600 s), scored, CELL 0's fallback kept, the deadman armed and not fired, the E28
              report VALID with the scheduler's rows (phases A / B1 / B2 / C), the harness B2 budget 9,000 (E28H_B2_TOKENS)
  real        K100Q's real kit on this notebook + E28: every boot plan with K100Q's --max-model-len 131,072 (argv + vLLM argparse), the
              first rung of both profiles built WITH DRAFT_HEAD=int8 (Merlin's build() -> "draft_lm_head_quant":"int8" in the
              spec), the others without; the scheduler applied in the commit and the rerun namespace (FULLY_PATCHED at E12_DIR,
              the patched arc2_duck loaded, E28_SCHED=1), revert -> the exact manifest bytes, re-apply; e28_i8_static on the
              REAL merged klee100 dir + the INSTALLED Merlin tree (SUPPORTED_STATIC); the INT8 draft-head quantizer (Merlin's own
              draft_lm_head_quant._quantize_chunked / marlin_quantize, FakeL4 platform as the E25 / E27 kits) on 8,192 real rows
              of klee100's lm_head (shapes, dequantised error)
Kit lessons kept: dataset-metadata.json is never on the mount; no probe file named after a stdlib module; JSON compared by
parsing; vLLM imports on CPU through the FakeL4 platform.
Rules: no Kaggle API calls, no git, no local whole-notebook runs (user rule: long tests on Kaggle CPU kits only).
Run: python build_e28_k100_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e28-k100-kit-<name>__esfersami50/
"""
import ast
import base64
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
BASE_KIT = E25B / "build_e26_k100q_cpu_kit.py"
E28_FACTS = HERE / "e28k100_build_facts.json"
SLUG, OWNER = "q27-e28-k100-kit", "esfersami50"
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, what, count=1):
    assert text.count(old) == count, (what, text.count(old))
    return text.replace(old, new)


# ------------------------------------------------------------------ the K100Q kit builder, retargeted (exec of its source)
_src = BASE_KIT.read_text(encoding="utf-8")
assert "\r" not in _src
BASE_KIT_SHA = sha(_src)
_src = once(_src, 'FACTS_PATH = HERE / "e26_k100q_build_facts.json"\n', f'FACTS_PATH = Path(r"{E28_FACTS}")   # E28 K100\n', "facts path")
_src = once(_src, 'assert "\\n\\n".join(CODE) == lf_text(EXPORT_PATH), "code export != the notebook\'s code cells"\n',
            'assert "\\n\\n".join(E28_READABLE(CODE)) == lf_text(EXPORT_PATH), "code export != the notebook\'s readable code cells"   # E28\n',
            "export check")
_src = once(_src, 'COMMIT_PROFILE = re.search(r\'^NB5_DEFAULT_PROFILE = "(\\w+)"$\', CODE[1], re.M).group(1)\n',
            'COMMIT_PROFILE = re.search(r\'^NB5_DEFAULT_PROFILE = "(\\w+)"\', CODE[1], re.M).group(1)   # E28: a trailing comment\n',
            "profile regex")
_src = once(_src, '_dm = re.search(r"^K100_DEADMAN_SEC = (\\d+)$", CODE[1], re.M)\n',
            '_dm = re.search(r"^K100_DEADMAN_SEC = (\\d+)\\b", CODE[1], re.M)   # E28: a trailing comment\n', "deadman regex")
_src = once(_src, 'KQ_OK_STATUS = "VALID" if "K100Q.notebook_status(" in CODE[-1] else "BUILT"\n'
                  'assert f\'"{KQ_OK_STATUS}"\' in (CODE[-1] + lf_text(HERE / "k100q_report.py")), KQ_OK_STATUS\n',
            'KQ_OK_STATUS = "VALID"   # E28: the E28 report\'s success status (e28k100_lib.build_e28_report)\n', "kq status")
_src = once(_src, 'assert sha(REPORT_FILES["nb5_fp8_reference.json"]) == FACTS["embedded"]["nb5_fp8_reference.json"]\n', "",
            "nb5 reference (not embedded in E28)")
_src = once(_src, 'SLUG = "q27-e26-k100q-kit"\nOWNER = "foysalemonshanto"\n', f'SLUG = "{SLUG}"\nOWNER = "{OWNER}"   # E28\n', "slug")
_E28_READABLE_SRC = r'''
def E28_READABLE(code):
    """E28: the notebook's CELL 6/7 carries e21_cell7 as a sha-pinned packed literal; the readable cells put its text back."""
    c = code[5]
    i = c.index("# ---- E28 K100 (size: the 400-kB commit limit)")
    call = next(n for n in ast.walk(ast.parse(c)) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
    expr = ast.get_source_segment(c, call.args[0].args[0])
    body = eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<e28 packed e21_cell7>", "eval"))
    return code[:5] + [c[:i] + body] + code[6:]
'''
_src = once(_src, "# ---------------------------------------------------------------- the notebook under test (from the builder's outputs)\n",
            _E28_READABLE_SRC + "# ---------------------------------------------------------------- the notebook under test (from the builder's outputs)\n",
            "readable helper")
NS = {"__file__": str(BASE_KIT), "__name__": "e28kit_base"}
exec(compile(_src, str(BASE_KIT), "exec"), NS)

FACTS = NS["FACTS"]
NB_TXT, NB_SHA, NB_RAW = NS["NB_TXT"], NS["NB_SHA"], NS["NB_RAW"]
CODE = NS["CODE"]
assert NB_SHA == FACTS["notebook_sha256"] and FACTS["default_profile"] == NS["COMMIT_PROFILE"] == "E28EVAL120"
assert NS["PHASE"] == FACTS["phase"] == "Q_E28_K100_EVAL120" and NS["DEADMAN_SEC"] == FACTS["deadman_sec"] == 24900
assert NS["OWNER"] == OWNER and FACTS["kernel_metadata"]["id"].startswith(OWNER + "/")
READABLE = NS["E28_READABLE"](CODE)
MODS = FACTS["modules"]
assert {n: sha((HERE / n).read_bytes()) for n in MODS} == MODS, "the Build modules != the ones packed in the notebook (rebuild)"
SCHED_SHA = FACTS["sched"]["e28_sched_sha"]

# ------------------------------------------------------------------ HEAD (the K100Q kit's, renamed)
HEAD = NS["HEAD"]
HEAD = once(HEAD, "E26 K100Q CPU kit", "E28 K100 CPU kit", "titles", 2)
HEAD = once(HEAD, '"/tmp/k100qkit"', '"/tmp/e28k100kit"', "tmp dir")
HEAD = once(HEAD, 'f"k100q_kit_{KIT}_report.json"', 'f"e28k100_kit_{KIT}_report.json"', "report name")

# ------------------------------------------------------------------ units (NEW body: the E28 unit tests on Py3.12)
UNITS = r'''
U = {"dataset": dataset_version_check()}
REF = ref_mount()
PC = payload_check(REF) if REF is not None else {"ok": False, "error": "tf-ref dataset not mounted"}
PC.pop("_payload", None)
U["payload"] = PC
U["ds_pins"] = {n: {"pin": h, "ok": REF is not None and (REF / n).is_file() and sha_file(REF / n) == h} for n, h in KF["ds_pins"].items()}
REP["checks"]["units"] = U; save()
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    (B / n).parent.mkdir(parents=True, exist_ok=True)
    (B / n).write_text(v, encoding="utf-8", newline="\n")
# the payload's REAL arc2_duck.py (pre sha b9f67010), from the mounted q27-e15b-payload zip (the notebook's own source)
PAY = find_payload()
U["payload_duck"] = {"used": PAY.get("used")}
try:
    import zipfile as _zf
    with _zf.ZipFile(PAY["used"]) as z_:
        _duck = z_.read("arc2_duck.py")
    (B / "arc2_duck_original.py").write_bytes(_duck)
    U["payload_duck"].update(sha256=hashlib.sha256(_duck).hexdigest(), pre_ok=hashlib.sha256(_duck).hexdigest() == KF["sched_pre"])
except BaseException as e:
    U["payload_duck"].update(pre_ok=False, error=f"{type(e).__name__}: {e}"[:400])
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5", "NB6", "E12_", "E15B_", "Q27X_", "RC2_", "E23_", "FAKE_", "K100", "E28"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
T = {}
try:
    r = sh("e28k100_unit_tests", [sys.executable, str(B / "e28k100_unit_tests.py")],
           {"E28KT_NB": str(B / "e26.ipynb"), "E28KT_FACTS": str(B / "e28k100_build_facts.json"), "E28KT_DUCK": str(B / "arc2_duck_original.py"),
            "E28KT_K100Q_FIXTURE": str(B / "k100q_fixture.json"), "E28KT_E18": str(B / "e18r2_cells01.ipynb"),
            "E28KT_K100Q_NB": str(B / "k100q.ipynb"), "E28KT_NB5LIB": str(B / "nb5_lib.py"), "E28KT_KLEE_CFG": str(B / "no_klee_cfg_here"), "E28KT_MERLIN_SRC": str(B / "no_merlin_here"),
            "E28KT_K100Q_OUT": str(B / "no_k100q_out_here"), "E28KT_SOL": str(B / "no_sol_here"), "PYTHONIOENCODING": "utf-8"}, 1800, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    T.update(rc=r["rc"], sec=r["sec"], summary=m.group(0) if m else None, fails=[l[:600] for l in lines if l.startswith("FAIL ")][:60],
             skips=[l[:300] for l in lines if l.startswith("SKIP ")], tail=r["tail"][-1500:])
    T["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not T["fails"] and int(m.group(2)) >= KF["unit_checks_min"]
    # the ONE allowed skip here: E4 (the real klee100 config + Merlin source): it runs in the real kit on the merged dir / tree
    T["skips_other"] = [l for l in T["skips"] if not l.startswith("SKIP E4 ")]
except BaseException as e:
    T.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["unit_tests"] = T
REP["checks"]["units"] = U; save()
log("e28 unit tests", json.dumps({k: v for k, v in T.items() if k != "tb"})[:3000])
DSV = U["dataset"]
verdict({"tf_ref_k100q_version_mounted": DSV.get("new_version_mounted") is True,
         "tf_ref_every_staged_file_sha_ok": DSV.get("ok") is True,
         "payload_sha_entries_blobs_record_ok": PC.get("ok") is True,
         "dataset_pinned_files_sha_eq_notebook_pins": bool(U["ds_pins"]) and all(v["ok"] for v in U["ds_pins"].values()),
         "duck_payload_arc2_duck_pre_sha": U["payload_duck"].get("pre_ok") is True,
         "e28_unit_tests_all_pass": T.get("all_pass") is True, "e28_unit_tests_only_E4_skipped": T.get("skips_other") == []},
        recorded={"summary": T.get("summary"), "fails": T.get("fails"), "skips": T.get("skips"), "prereq": DSV.get("prereq"),
                  "on_mount_not_staged": DSV.get("on_mount_not_staged"), "dataset_dir": DSV.get("dir")})
'''

# ------------------------------------------------------------------ e2e driver (the K100Q kit's, + E28)
E2E_DRV = NS["E2E_DRV"]
E2E_DRV = once(E2E_DRV, 'if RR:\n    os.environ.update(KAGGLE_IS_COMPETITION_RERUN="1", Q27X_TIME_BUDGET=str(BUDGET))     # the fake deadline\n',
               'if RR:\n    os.environ.update(KAGGLE_IS_COMPETITION_RERUN="1", Q27X_TIME_BUDGET=str(BUDGET))     # the fake deadline\n'
               'else:      # E28: the commit too gets a FAKE SHORT deadline (the profile is 415 min); the harness B2 budget\n'
               '    os.environ["Q27X_TIME_BUDGET"] = str(BUDGET)\n'
               'os.environ["E28H_B2_TOKENS"] = "9000"   # E28 (harness only): the B2 re-run budget (T1 = NB5_ATTEMPT_B0 6,000)\n',
               "commit fake deadline")
_c = E2E_DRV.index('check("commit: the baked ')
_ce = E2E_DRV.index("\n", _c) + 1
E2E_DRV = E2E_DRV[:_c] + ('check(f"commit: the profile\'s baked budget = 415 min (24,900 s), the E28 kit\'s fake short deadline in force "\n'
                          '      f"(TOTAL_BUDGET_SEC == {BUDGET})", lambda: P["time_budget_sec"] == 24900 and G["TOTAL_BUDGET_SEC"] == BUDGET,\n'
                          '      G.get("TOTAL_BUDGET_SEC"))\n') + E2E_DRV[_ce:]
_c = E2E_DRV.index('    check("commit: the eval subset (')
_ce = E2E_DRV.index("    check(f\"commit: the profile's baked budget", _c)
E2E_DRV = E2E_DRV[:_c] + r'''    check("commit (E28): EVAL120 = every eval task (no task list), scored by exact match over the gold of all 120 (the gold sets the "
          "query count), >= 1 test query hit (the fake's lookup)",
          lambda: P["source"] == "eval" and "task_list" not in P and len(G["TASK_IDS"]) == 120 and sc["eval_query_total"] >= 120
          and sc["eval_query_hits"] >= 1, json.dumps({"score": sc, "tasks": len(G.get("TASK_IDS") or [])}, default=str)[:900])
''' + E2E_DRV[_ce:]
E2E_DRV = once(E2E_DRV, 'SOLVE_IDS = sorted(test) if RR else sorted(P.get("task_list") or [])\n',
               'SOLVE_IDS = sorted(test) if RR else list((G.get("E12_ORDER") or [])[:12])   # E28: the commit\'s first 12 (plain order) under the fake deadline\n',
               "solve ids")
# the K3 ladder retry, E28: try 1 (rung 1, INT8 head) OOM -> I8 dropped -> try 2 (rung 1, BF16) OOM -> try 3 (rung 2) boots
_r0 = E2E_DRV.index("if RETRY:          # E26: the K3 ladder (rung 1 OOM -> rung 2), E18 R2's engine ladder")
_r1 = E2E_DRV.index("else:\n    check(\"one boot (FP8, MTP on), the first K3 rung booted", _r0)
E2E_DRV = E2E_DRV[:_r0] + r'''if RETRY:          # E28: the K3 ladder with I8: rung 1 + INT8 head OOM -> the SAME rung, BF16 head (I8 dropped) OOM -> rung 2
    _t3 = b0.get("tries") or []
    check("boot retry (E18 ladder, <= 3 tries) with I8: try 1 = rung 1 (pin +__G1__ GiB) with DRAFT_HEAD=int8 failed with an OOM "
          "line -> I8_UNSUPPORTED_ON_K100 (OOM_WITH_INT8_HEAD), a DRAFT_HEAD lever drop; try 2 = the SAME rung with the BF16 head "
          "failed (OOM); try 3 = rung 2 (pin +__G2__ GiB) booted; the lever env kept; the later ready limit 15 min",
          lambda: len(boots) == 1 and len(_t3) == 3 and [round(t_["pin_gib"], 6) for t_ in _t3] == [round(__G1__, 6), round(__G1__, 6), round(__G2__, 6)]
          and [t_.get("e28_i8") for t_ in _t3] == [True, False, False] and _t3[0]["ladder"]["oom_lines"] and _t3[1]["ladder"]["failed"]
          and _t3[2].get("status") in (None, "DONE") and abs(b0["pin_used"]["gib"] - __G2__) < 1e-6
          and any(d_.get("keys") == ["DRAFT_HEAD"] and "I8_UNSUPPORTED_ON_K100: OOM_WITH_INT8_HEAD" == d_.get("why") for d_ in b0.get("levers_dropped") or [])
          and (G["E28_I8"].get("dropped") or {}).get("kind") == "OOM_WITH_INT8_HEAD" and G["E28_I8"]["on"] is False
          and all(t_["levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"} for t_ in _t3) and _t3[0]["name"].endswith("_i8")
          and b0["kv_dtype"] == "fp8" and b0["mtp"] is True and G.get("NB6_READY_SEC") == 15 * 60,
          _bsum + " | " + json.dumps({"dropped": b0.get("levers_dropped"), "i8": G.get("E28_I8")}, default=str)[:900])
''' + E2E_DRV[_r1:]
E2E_DRV = once(E2E_DRV, "      and len(b0[\"tries\"]) == 1 and G.get(\"NB6_READY_SEC\") == (30 * 60 if RR else G.get(\"NB5_READY_TIMEOUT\")), _bsum)\n",
               "      and len(b0[\"tries\"]) == 1 and G.get(\"NB6_READY_SEC\") == (30 * 60 if RR else G.get(\"NB5_READY_TIMEOUT\"))\n"
               "      and b0[\"tries\"][0].get(\"e28_i8\") is True and b0[\"tries\"][0][\"name\"].endswith(\"_i8\"), _bsum)   # E28: the INT8 head booted\n",
               "one boot i8")
E2E_DRV = once(E2E_DRV, "      and all(b0[\"server_env_levers\"][k_] == (G.get(\"NB5_BASE_ENV\") or {}).get(k_) for k_ in "
                        "set(b0[\"server_env_levers\"]) - set(_lv)),",
               "      and all(b0[\"server_env_levers\"][k_] == (G.get(\"NB5_BASE_ENV\") or {}).get(k_) for k_ in "
               "set(b0[\"server_env_levers\"]) - set(_lv) - {\"DRAFT_HEAD\"})\n"
               "      and b0[\"server_env_levers\"].get(\"DRAFT_HEAD\") == (None if RETRY else \"int8\"),   # E28: the INT8 head on the booted try (not after the drop)",
               "lever env i8")
E2E_DRV = once(E2E_DRV, 'KQP = W / "q27x" / "k100_quality_report.json"\nKQ = json.loads(KQP.read_text(encoding="utf-8")) if KQP.is_file() else {}\nif ',
               'KQP = W / "q27x" / "e28_k100_report.json"   # E28: the run report\nKQ = json.loads(KQP.read_text(encoding="utf-8")) if KQP.is_file() else {}\nif ',
               "report path 1")
_k0 = E2E_DRV.index('KQP = W / "q27x" / "k100_quality_report.json"\nKQ = json.loads(KQP.read_text(encoding="utf-8")) if KQP.is_file() else {}\nREC["k100_quality"]')
_k1 = E2E_DRV.index("if __DEADMAN__:") if "if __DEADMAN__:" in E2E_DRV else E2E_DRV.index("if True:          # the commit's absolute deadman")
E2E_DRV = E2E_DRV[:_k0] + r'''KQP = W / "q27x" / "e28_k100_report.json"   # E28
KQ = json.loads(KQP.read_text(encoding="utf-8")) if KQP.is_file() else {}
REC["e28_report"] = {k: KQ.get(k) for k in ("status", "error", "errors", "score", "attempt_summary", "b2")}
_SR = G.get("E28_SCHED_REC") or {}
check("E28: the A / B / C scheduler applied in CELL 3 (FULLY_PATCHED, the module's post sha), E28_SCHED=1 in the notebook env, the "
      "harness T1 = NB5_ATTEMPT_B0 (6,000) / B2 = 9,000, the patched arc2_duck loaded (_e28_on True)",
      lambda: _SR["applied"] is True and _SR["after"] == "FULLY_PATCHED" and _SR["sha_after"] == "__SCHED_POST__"
      and os.environ.get("E28_SCHED") == "1" and _SR["env"]["E28_T1"] == "6000" and _SR["env"]["E28_B2_TOKENS"] == "9000" and _SR["env"]["E28_B2_REQUEST_TIMEOUT"] == "9000"
      and G["E12A"]._e28_on() is True and hasattr(G["E12A"], "select_attempts_e28"), json.dumps(_SR, default=str)[:900])
_E28R = [r_.get("e28") or {} for v_ in (G.get("E12_RECORDS") or {}).values() for r_ in v_]
check("E28: every attempt record carries its E28 row (phase A / B1 / B2 / C; branch A=0, C=1, B=2); >= 1 phase-A attempt",
      lambda: _E28R and all(e_.get("phase") in ("A", "B1", "B2", "C") for e_ in _E28R)
      and all({"A": 0, "C": 1, "B1": 2, "B2": 2}[e_["phase"]] == int(r_.get("branch") or 0)
              for v_ in G["E12_RECORDS"].values() for r_ in v_ for e_ in [r_.get("e28") or {}])
      and sum(1 for e_ in _E28R if e_.get("phase") == "A") >= 1,
      json.dumps({"phases": sorted({str(e_.get("phase")) for e_ in _E28R}), "n": len(_E28R)}))
check("E28: no ORDER2: E12_ORDER == the plain task order (sorted TASK_IDS)",
      lambda: list(G["E12_ORDER"]) == list(G["TASK_IDS"]) == sorted(G["TASK_IDS"]), (G.get("E12_ORDER") or [])[:8])
_la8 = [L_ for L_ in launches if L_["outcome"] == "ready"]
check("E28: the server launch argv carries K100Q's --max-model-len 131072 (the last one wins; no 262k)",
      lambda: len(_la8) == 1 and _last(_la8[0]["argv"], "--max-model-len") == "131072", json.dumps(_la8, default=str)[:600])
check("E28: I8 record: wanted, the static check ran (fake: INCONCLUSIVE, no Merlin tree), the booted try's read-back = FAKE_SERVER",
      lambda: G["E28_I8"]["want"] is True and (G["E28_I8"]["static"] or {}).get("verdict") == "INCONCLUSIVE"
      and ((b0.get("tries") or [{}])[-1].get("i8_readback") or {}).get("label") == "FAKE_SERVER", json.dumps(G.get("E28_I8"), default=str)[:900])
if RR:
    check("rerun: the E28 report is SKIPPED_RERUN (written, no scoring); E28_SCHED applied in the rerun too",
          lambda: KQ["status"] == "SKIPPED_RERUN" and _SR["applied"] is True, REC["e28_report"])
else:
    _wp = main.get("window_plan") or {}
    check("commit: the fixed solve window planned = __WINDOW__ s (plan fixed), shortened to the fake deadline's room (unequal)",
          lambda: _wp["planned_sec"] == __WINDOW__ and _wp["plan"]["fixed"] is True and _wp["unequal"] is True, json.dumps(_wp, default=str))
    _pc = ((rep.get("quality_ab") or {}).get("per_config") or {}).get("__PHASE__") or {}
    check("commit: the per-task table covers every one of the 120 tasks (>= 2 attempts each: started or NOT_STARTED, + any B re-run), "
          "outcomes in the NB5 set",
          lambda: set(_pc["tasks"]) == set(G["TASK_IDS"]) and len(_pc["tasks"]) == 120 and all(len(v["attempts"]) >= 2 for v in _pc["tasks"].values())
          and all(a["outcome"] in ("ACCEPTED", "BUDGET_END", "INCOMPLETE_WINDOW", "FAILED_OR_INTERRUPTED", "NOT_STARTED")
                  for v in _pc["tasks"].values() for a in v["attempts"]), json.dumps(_pc.get("summary"), default=str)[:1200])
    check("commit: the E28 report: VALID, the score of the written submission == the run report's, the per-attempt rows (one per "
          "record), the scheduler section from SCHED_LAST['e28'] (started per phase, rows), the markdown table, no report errors",
          lambda: KQ["status"] == "VALID" and KQ["score"]["agree"] is True and len(KQ["attempts"]) == len(_E28R) > 0
          and KQ["scheduler"]["present"] is True and (KQ["scheduler"].get("started") or {}).get("A", 0) >= 1
          and len(KQ["sched_attempts"]) >= 1 and bool(KQ.get("table_md")) and not KQ.get("errors"),
          json.dumps({k: KQ.get(k) for k in ("status", "error", "errors", "score", "attempt_summary")}, default=str)[:1500])
''' + E2E_DRV[_k1:]
E2E_DRV = once(E2E_DRV, 'check("E26: the bundle zip carries k100_quality_report.json",\n      lambda: any(n_.endswith("k100_quality_report.json")',
               'check("E28: the bundle zip carries e28_k100_report.json",\n      lambda: any(n_.endswith("e28_k100_report.json")', "bundle")
E2E_DRV = E2E_DRV.replace("__SCHED_POST__", SCHED_SHA["arc2_duck.py"]["post"]).replace("__PHASE__", NS["PHASE"])
# kit v5 (user 2026-11-01: workers / --max-num-seqs 24 on the TOP rung only): the booted try's capacity tuple (pin, fake pool,
# F, launch seqs, workers) = 24 / 24 on the top rung (commit / rerun); the second rung (rerun-retry) keeps K100Q's 18 / 18
_FC = NS["FAKE_CAP"]
E2E_DRV = once(E2E_DRV, f"_cw = {_FC[1]!r} if RETRY else {_FC[0]!r}     # E26: pin, fake pool, F, launch seqs (K100 table), workers",
               f"_cw = {_FC[1]!r} if RETRY else {(_FC[0][0], _FC[0][1], _FC[0][2], 24, 24)!r}"
               "     # E28: pin, fake pool, F, launch seqs, workers (top rung 24 / 24; rung 2 = K100Q's 18 / 18)", "capacity tuple 24")
E2E_DRV = once(E2E_DRV, 'rule), launch max_num_seqs {_cw[3]} (seqs table), workers = min({_cw[1]} // (16 + {_cw[2]}), {_cw[3]}) = {_cw[4]}",',
               'rule), launch max_num_seqs {_cw[3]} (E28: 24 on the top rung, K100Q\'s table below), workers {_cw[4]} (E28: 24 on the top "\n'
               '      f"rung, the R2 min below)",',
               "capacity label 24")
for _k, _v in (("__G1__", repr(NS["GIBS"][0])), ("__G2__", repr(NS["GIBS"][1])), ("__WINDOW__", str(FACTS["window_sec"]))):
    E2E_DRV = E2E_DRV.replace(_k, _v)
assert "__" + "SCHED_POST__" not in E2E_DRV and "k100_quality_report" not in E2E_DRV, "E28 e2e placeholders / the old report left"
compile(E2E_DRV, "<e28_e2e>", "exec")

E2E = NS["E2E"]
E2E = once(E2E, '    _max = 3500 if mode != "commit" else __COMMIT_MAX__   # commit: the notebook\'s own budget + 30 min\n',
           '    _max = 3500 if mode != "commit" else 4200   # E28: the commit runs on a fake 3,000-s deadline (+ the report / bundle)\n',
           "commit max")
E2E = once(E2E, '"E26_E2E_BUDGET": "1800",', '"E26_E2E_BUDGET": "3000" if mode == "commit" else "1800",', "commit budget")

# ------------------------------------------------------------------ real (the K100Q kit's, + E28)
REAL = NS["REAL"]
REAL = once(REAL, '            g["NB5_BOOT_KV"], g["NB5_BOOT_LEVERS"] = kv, dict(lev0)\n',
            '            _w8 = "int8" if (i == 0 and P_.get("i8")) else None     # E28: the notebook boots rung 1 with the INT8 head first\n'
            '            g["NB5_BOOT_KV"], g["NB5_BOOT_LEVERS"] = kv, dict(lev0, **(g["E28_I8_ENV"] if _w8 else {}))\n', "plans i8 levers")
REAL = once(REAL, '                    chk.append(f"spec {spec}")\n',
            '                    chk.append(f"spec {spec}")\n'
            '                if mtp == "on" and isinstance(_spj, dict) and _spj.get("draft_lm_head_quant") != _w8:   # E28 I8 via Merlin\'s build()\n'
            '                    chk.append(f"draft_lm_head_quant {_spj.get(\'draft_lm_head_quant\')!r} vs {_w8!r}")\n', "plans i8 spec")
REAL = once(REAL, '                b.update(ok=True, host_phase=', '                b.update(i8=_w8, ok=True, host_phase=', "plans record i8")
_mmp = NS["MM_SEC"]
_plat = _mmp[_mmp.index('MM_PROBE = r"""') + len('MM_PROBE = r"""'):]
_plat = _plat[:_plat.index('    print("@JSON " + json.dumps(out, default=str)); raise SystemExit(0)\n') +
              len('    print("@JSON " + json.dumps(out, default=str)); raise SystemExit(0)\n')]
assert "class FakeL4(VPC.NonNvmlCudaPlatform):" in _plat and _plat.startswith("\nimport copy, json, os, re, sys, traceback, types\n")
I8_PROBE = _plat.replace("ext_src, cfg = open(sys.argv[1]).read(), json.load(open(sys.argv[2]))\n", "") + r'''
try:
    from safetensors import safe_open
    ck = sys.argv[1]
    idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))
    key = next(k for k in idx["weight_map"] if k.endswith("lm_head.weight") and "mtp" not in k)
    with safe_open(os.path.join(ck, idx["weight_map"][key]), "pt") as h:
        sl = h.get_slice(key)
        shape = list(sl.get_shape())
        w = sl[0:8192]
    import vllm.model_executor.layers.draft_lm_head_quant as D
    q, s = D._quantize_chunked(w, D._QUANT_TYPES["int8"])
    w_kn = w[:2048].to(torch.float32).t().contiguous()
    w_ref = D.marlin_quantize(w_kn, D._QUANT_TYPES["int8"], D._GROUP_SIZE)[0]
    rel = float((w_ref.float() - w_kn).abs().max() / w_kn.abs().max().clamp_min(1e-12))
    out["steps"]["i8"] = {"ok": True, "key": key, "shape": shape, "dtype": str(w.dtype), "rows": int(w.shape[0]),
                          "q_shape": list(q.shape), "s_shape": list(s.shape), "q_dtype": str(q.dtype), "rel_max_err_2048_rows": round(rel, 5),
                          "group": D._GROUP_SIZE, "shard_rows_tp2": (-(-shape[0] // 64) * 64) // 2}
except BaseException as e:
    out["steps"]["i8"] = {"ok": False, "error": err(e), "tb": traceback.format_exc()[-1500:]}
print("@JSON " + json.dumps(out, default=str))
'''
compile(I8_PROBE, "<e28 i8 probe>", "exec")
E28_REAL = r'''
# ---------------------------------------------------------------- E28: the scheduler on the real Duck payload, I8 on the real klee100 / Merlin
E28R = {}
try:
    S_ = G["E28S"]
    E12D = Path(os.environ.get("E12_DIR") or (K / "e12_duck"))
    E28R["sched_commit"] = {k: (G.get("E28_SCHED_REC") or {}).get(k) for k in ("applied", "after", "sha_after", "env", "error")}
    E28R["sched_rerun"] = {k: (GR.get("E28_SCHED_REC") or {}).get(k) for k in ("applied", "after", "sha_after", "env", "error")}
    E28R["module_sha"] = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in (G.get("E28_SRC") or {}).items()}
    E28R["state"] = S_.tree_state(E12D)
    E28R["patched_loaded"] = bool(getattr(G.get("E12A"), "_e28_solve_all", None)) and G["E12A"].__file__.startswith(str(E12D))
    rv = S_.revert_e28_sched(E12D)
    E28R["revert"] = {k: rv.get(k) for k in ("reverted", "after", "sha_after")}
    E28R["revert_eq_manifest"] = hashlib.sha256((E12D / "arc2_duck.py").read_bytes()).hexdigest() == G["_E12_MANIFEST"]["arc2_duck.py"] == KF["sched_pre"]
    ap = S_.apply_e28_sched(E12D)
    E28R["reapply"] = {k: ap.get(k) for k in ("applied", "after", "sha_after")}
    E28R["sched_ok"] = (E28R["sched_commit"]["applied"] is True and E28R["sched_rerun"]["applied"] is True
                        and E28R["sched_commit"]["sha_after"] == KF["sched_post"] and E28R["state"] == "FULLY_PATCHED"
                        and E28R["patched_loaded"] and E28R["revert"]["after"] == "ORIGINAL" and E28R["revert_eq_manifest"]
                        and E28R["reapply"]["after"] == "FULLY_PATCHED" and E28R["module_sha"] == KF["e28_modules"]
                        and E28R["sched_commit"]["env"]["E28_T1"] == "104000" and E28R["sched_commit"]["env"]["E28_B2_TOKENS"] == "120000"
                        and E28R["sched_commit"]["env"]["E28_B2_REQUEST_TIMEOUT"] == "9000")
except BaseException as e:
    E28R.update(sched_ok=False, sched_error=f"{type(e).__name__}: {e}"[:800], sched_tb=traceback.format_exc()[-1500:])
try:
    E28R["i8_static"] = G["e28_i8_static"]()
    E28R["i8_static_ok"] = E28R["i8_static"].get("verdict") == "SUPPORTED_STATIC" and len(E28R["i8_static"].get("files") or []) == 3
except BaseException as e:
    E28R.update(i8_static_ok=False, i8_static_error=f"{type(e).__name__}: {e}"[:600])
try:
    r = venv_py(I8_PROBE, [str(G["CKPT"])], name="e28_i8_probe", extra_env=dict(globals().get("mm_env") or {"CUDA_VISIBLE_DEVICES": ""}))
    js = r["json"] or {"error": r["tail"]}
    E28R["i8_probe"] = (js.get("steps") or {}) if isinstance(js, dict) else js
    _i8 = (E28R["i8_probe"] or {}).get("i8") or {}
    E28R["i8_probe_ok"] = (_i8.get("ok") is True and _i8.get("shape") == [248320, 2560] and _i8.get("dtype") == "torch.bfloat16"
                           and _i8.get("shard_rows_tp2") == 124160 and (_i8.get("rel_max_err_2048_rows") or 1) < 0.02)
except BaseException as e:
    E28R.update(i8_probe_ok=False, i8_probe_error=f"{type(e).__name__}: {e}"[:600])
_p1 = [b for pl in PLANS.values() for n, b in pl.items() if b.get("i8")]
E28R["plans_i8"] = {n: (b.get("argv_flags") or {}).get("--speculative-config") for pl in PLANS.values() for n, b in pl.items()}
E28V = {"e28_sched_applied_commit_rerun_revert_reapply": E28R.get("sched_ok") is True,
        "e28_i8_static_supported_real_klee100_installed_tree": E28R.get("i8_static_ok") is True,
        "e28_i8_merlin_int8_quantizer_on_real_lm_head_rows": E28R.get("i8_probe_ok") is True,
        "e28_first_rung_both_profiles_built_with_int8_head": len(_p1) == 2 and all(b.get("ok") and not b.get("argv_check") for b in _p1)}
REP["checks"]["e28"] = E28R; save()
log("E28 real", json.dumps({k: v for k, v in E28R.items() if not k.endswith("tb")}, default=str)[:4000])
'''
# kit v2 (real kit v1 on Kaggle: NameError I8_PROBE): the probe source is a builder-side string; inject it into the kit body
E28_REAL = ("\nI8_PROBE = json.loads(" + repr(json.dumps(I8_PROBE)) + ")   # E28 kit v2: the INT8 draft-head probe source\n"
            + E28_REAL)
assert json.loads(json.dumps(I8_PROBE)) == I8_PROBE and "__" + "SHAS__" not in I8_PROBE and "__BLOB__" not in I8_PROBE
_v = REAL.index("sc, sr = SUB.get(\"commit\") or {}, SUB.get(\"rerun\") or {}\nverdict({")
REAL = REAL[:_v] + E28_REAL + REAL[_v:]
# kit v5: the TOP rung of both profiles launches --max-num-seqs 24 (top_rung_seqs), lower rungs K100Q's table; captures = K100Q's,
# the PP2 micro-batch bound
REAL = once(REAL, '            seqs = (g["K100_SEQS_TABLE"][round(float(tr["gib"]), 2)] if g["NB5_SEQS_RULE"] == "k100"\n'
                  '                    else L5.seqs_for(kv, tr["gib"]) if g["NB5_SEQS_RULE"] == "table" else None)\n',
            '            seqs = (g["K100_SEQS_TABLE"][round(float(tr["gib"]), 2)] if g["NB5_SEQS_RULE"] == "k100"\n'
            '                    else L5.seqs_for(kv, tr["gib"]) if g["NB5_SEQS_RULE"] == "table" else None)\n'
            '            if P_.get("top_rung_seqs") and abs(float(tr["gib"]) - g["K100_PIN_STEPS_GIB"][0]) < 1e-9:   # E28: the top rung = 24\n'
            '                seqs = int(P_["top_rung_seqs"])\n', "plans seqs 24")
REAL = once(REAL, 'if caps != KF["captures"] or not L5.captures_cover(caps or [], seqs or 1): chk.append(f"captures {caps}")',
            'if caps != KF["captures"] or max(caps or [0]) < -(-(seqs or 1) * 4 // 2): chk.append(f"captures {caps}")   # E28: per PP2 micro-batch',
            "plans captures pp2")
REAL = once(REAL, "    **QDV,\n}, recorded={", "    **QDV, **E28V,\n}, recorded={\"e28\": {k: v for k, v in E28R.items() if not k.endswith(\"tb\")}, ", "verdict e28")
REAL = once(REAL, '    "tf_ref_new_version_mounted": DSV.get("new_version_mounted") is True,\n',
            '    "tf_ref_k100q_version_mounted": DSV.get("new_version_mounted") is True,\n', "real dataset label")

# ------------------------------------------------------------------ kit table
KF = dict(NS["KF"])
KF.update(sched_pre=SCHED_SHA["arc2_duck.py"]["pre"], sched_post=SCHED_SHA["arc2_duck.py"]["post"], e28_modules=MODS)
assert FACTS["workers"] == 24
KF["seqs"] = [FACTS["workers"]] + list(KF["seqs"][1:])   # kit v5: the top rung launches 24, lower rungs K100Q's table (18 / 14 / 9 / 6)
_UT = (HERE / "e28k100_unit_tests.py").read_text(encoding="utf-8")
KF["unit_checks_min"] = len(re.findall(r'^\s*check\("', _UT, re.M)) - 1      # every check (E4 skips in this kit)
NS["KF"] = KF
UT_NS = {"__name__": "e28kt_fixture_only"}
_fx_src = _UT[_UT.index("def k100q_fixture(out_dir, sol_path):"):_UT.index("\n\n_FX = ")]
exec(compile("import json\nfrom pathlib import Path\n" + _fx_src, "<e28 fixture>", "exec"), UT_NS)
FIXTURE = json.dumps(UT_NS["k100q_fixture"](Path(r"D:/ARC AGI 2/_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x"),
                                            Path(NS["EVAL_SOL"] if "EVAL_SOL" in NS else
                                                 r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/"
                                                 r"Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json")),
                     separators=(",", ":"))
K100Q_NB_TXT = Path(FACTS["base"]["k100q_notebook"]).read_bytes().decode("utf-8")
NB5LIB_TXT = (E25B / "nb5_lib.py").read_bytes().decode("utf-8")
assert sha(NB5LIB_TXT) == FACTS["embedded"]["nb5_lib.py (in the payload file)"], "Build nb5_lib.py != the payload's"
assert sha(K100Q_NB_TXT) == FACTS["base"]["k100q_sha256"]
UNIT_FILES = {"e26.ipynb": NB_TXT, "e28k100_build_facts.json": NS["FACTS_TXT"], "e28k100_unit_tests.py": _UT,
              "e28k100_lib.py": (HERE / "e28k100_lib.py").read_text(encoding="utf-8"),
              "e28_sched_patch.py": (HERE / "e28_sched_patch.py").read_text(encoding="utf-8"),
              "e18r2_cells01.ipynb": NS["E18_REDUCED"], "k100q.ipynb": K100Q_NB_TXT, "k100q_fixture.json": FIXTURE,
              "nb5_lib.py": NB5LIB_TXT}   # kit v3: G3 / G4 run the notebook's nb5_boot on the real nb5_lib (sha = the payload's)
FAKE_SRC = NS["K5"]["src"]("nb5_fake_server.py")   # kit v4: the proven fake as is (the server stays at 131,072)
E2E_FILES = {"e26.ipynb": NB_TXT, "e26_e2e.py": E2E_DRV, "nb5_fake_server.py": FAKE_SRC, **NS["E18_FILES"]}
REAL_FILES = dict(NS["REAL_FILES"])
assert REAL_FILES["e26.ipynb"] == NB_TXT
NS["HEAD"] = HEAD
NS["KITS"] = {
    "units": {"body": UNITS, "title": "Q27 E28 K100 Kit Units", "models": [], "files": UNIT_FILES},
    "e2e-rerun": {"body": E2E.replace("__MODE__", "rerun,rerun-retry"), "title": "Q27 E28 K100 Kit E2E Rerun", "models": NS["MODELS"], "files": E2E_FILES},
    "e2e-commit": {"body": E2E.replace("__MODE__", "commit"), "title": "Q27 E28 K100 Kit E2E Commit", "models": NS["MODELS"], "files": E2E_FILES},
    "real": {"body": REAL, "title": "Q27 E28 K100 Kit Real", "models": NS["MODELS"], "files": REAL_FILES},
}
KIT_FACTS = HERE / "e28k100_kit_facts.json"

if __name__ == "__main__":
    print("notebook under test", NS["NB_NAME"], "sha256", NB_SHA, len(NB_RAW), "bytes | profile", NS["COMMIT_PROFILE"], "| phase", NS["PHASE"],
          "| base kit", BASE_KIT.name, BASE_KIT_SHA[:12], flush=True)
    only = [a for a in sys.argv[1:] if a in NS["KITS"]]          # e.g. `python build_e28_k100_cpu_kit.py real` (kit v2)
    prev = json.loads(KIT_FACTS.read_text(encoding="utf-8")) if KIT_FACTS.is_file() else {}
    out = dict(prev.get("kits") or {}) if only and prev.get("notebook_sha256") == NB_SHA else {}
    for n in (only or NS["KITS"]):
        d, h, nbytes = NS["build"](n)
        out[n] = {"folder": str(d), "sha256": h, "bytes": nbytes, "kit_version": "v5"}   # v3: the notebook with the I8 gate; units carry nb5_lib.py; real = v2's probe fix
    KIT_FACTS.write_bytes((json.dumps({"notebook_sha256": NB_SHA, "base_kit": str(BASE_KIT), "base_kit_sha256": BASE_KIT_SHA,
                                       "kits": out, "kf": {k: KF[k] for k in ("sched_pre", "sched_post", "e28_modules", "unit_checks_min")}},
                                      indent=1) + "\n").encode("utf-8"))
