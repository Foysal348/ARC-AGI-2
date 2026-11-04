"""E25 NB6Q SUBMIT Kaggle CPU kits (lead 2026-11-01): the same four kernels as NB6 (units, e2e-rerun, e2e-commit, real) on
foysalemonshanto, pinned Py3.12 image, no GPU, no internet, private. A wrapper: build_e25_nb6_cpu_kit.py (NB6's kit builder,
never written) is read as text, the marked "NB6Q kit" edits below are applied, and it is executed with --nb = the NB6Q
notebook (+ its code export). Each kernel prints exactly one line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.

NB6Q kit edits (on top of every NB6 kit check, which stays as it is):
  K1  slugs / titles / report names / tmp dir -> nb6q; the push folder under test = q27-xhigh-e25-nb6q-submit (the NB6 one
      is shipped as a reference for the metadata comparison)
  K2  units: nb6q_unit_tests.py sections S1 (static diff vs NB6 87afcc84, packed-literal shas, placements) + S3
      (nb6_unit_tests.py on NB6Q = every check PASS but exactly the NB6-only push-id check; the NB6Q push folder) replaces the
      plain nb6_unit_tests run (0 skips required); nb5_unit_tests on NB5 QUALITY (gated) and on the notebook under test
      (recorded) stay as in NB6's kit
  K3  e2e: the kit's fake Merlin tree = the real Merlin ad42e37 pp_utils.py / model_runner.py / gpu_worker.py (NB6Q_FAKE_TREE,
      FAKE mode only); rerun + commit: the patch must be APPLIED (post shas), NB4_PPDEFER=1 must reach every fake server
      process, the read-back after the warm-up and at the end of the booted try must be recorded ENGAGED; rerun-retry: the
      tree is FOREIGN (pre-sha mismatch) -> the patch is refused, the files stay byte for byte, no launch carries NB4_PPDEFER,
      every read-back NOT_APPLIED, and the run still completes (every NB6 rerun-retry check). The kit-local copy of
      nb5_fake_server.py records NB4_PPDEFER / NB4_PPACK per launch and, with NB4_PPDEFER=1, prints the NB4_PPQ init / stats
      line shapes of a TP2 x PP2 boot (fake only, no engine claim).
  K4  e2e-commit: the stale NB6 kit expectation "40-min budget" is fixed: NB6COMMIT = 50 min (TOTAL_BUDGET_SEC 3000)
  K5  real: after NB6's install step the NB6Q state must be APPLIED on the REAL Merlin tree installed from the payload (pre
      shas = Merlin ad42e37, post shas verified, verify_ppack_patch 'applied'); the rerun namespace verifies the same tree
      (already applied -> APPLIED); every boot plan of both profiles carries NB4_PPDEFER=1 and no NB4_PPACK; then on the real
      tree: revert -> every pre sha, nb6q_unit_tests.py sections S2 (the notebook's NB6Q block: applied / failure / restore /
      read-back paths, incl. the POSIX venv-python path; 0 skips) + S4 (nb4_ppack_tests.py, reused) on THIS pristine real
      tree, re-apply -> every post sha
  K6  all kits: the file blob is base85 (NB6's kit: base64), so the units kit stays < 400 kB with the NB6Q notebook
Rules: no Kaggle API calls here (the lead pushes); long tests only on the Kaggle CPU kits.
Run: python build_e25_nb6q_cpu_kit.py
  -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb6q-kit-{units,e2e-rerun,e2e-commit,real}__foysalemonshanto/
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
NB6KIT = HERE / "build_e25_nb6_cpu_kit.py"
NBQ = HERE.parent / "Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb"
EXPQ = HERE / "nb6q_code_export.py"
NB6_NB = HERE.parent / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb"
MSRC = Path("D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
PP_FILES = {"vllm/v1/worker/gpu/pp_utils.py": ("6be931deaf8bee6dc9571c3c84d7e92c9f26c618c09dcb58a8b384bff487a662",
                                               "8b7fa4a6e4598ff4d4a56eb6e595efb1e494276b12b187b099973c262e90c69a"),
            "vllm/v1/worker/gpu/model_runner.py": ("05e889afba5dd60b6995b78aaa1f69051770ab02b638680097eccbf4e592d64c",
                                                   "2c25fda60dbd5f2506ff989c29ccfe8fc0ff082fbad457fc3ae9c89ed9a185d6"),
            "vllm/v1/worker/gpu_worker.py": ("a97d08ef042a53fcd139bae041fff407fce75796650d936c200c699f0d63a489",
                                             "5642b89d918b30e5b70b1067d53776e379613035b7b6e91e93cf0ee4279cfef5")}
QPRE = {r: v[0] for r, v in PP_FILES.items()}
QPOST = {r: v[1] for r, v in PP_FILES.items()}


def once(text, old, new, what, count=1):
    assert text.count(old) == count, (what, text.count(old))
    return text.replace(old, new)


import hashlib  # noqa: E402

for r, (pre, _) in PP_FILES.items():
    assert hashlib.sha256((MSRC / r).read_bytes()).hexdigest() == pre, ("Merlin source copy is not ad42e37", r)
MTREE = {"mtree/" + r: (MSRC / r).read_bytes().decode("utf-8") for r in PP_FILES}
assert not any("\r" in v for v in MTREE.values())

S = NB6KIT.read_text(encoding="utf-8").replace("\r\n", "\n")
# ---------------------------------------------------------------- K1 names
S = once(S, 'NB_PATH = _arg("--nb", HERE.parent / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb")',
         'NB_PATH = _arg("--nb", HERE.parent / "Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb")   # NB6Q kit K1', "K1 nb")
S = once(S, 'EXPORT_PATH = _arg("--export", HERE / "nb6_code_export.py")',
         'EXPORT_PATH = _arg("--export", HERE / "nb6q_code_export.py")   # NB6Q kit K1', "K1 export")
S = once(S, 'SLUG = "q27-e25-nb6-kit"', 'SLUG = "q27-e25-nb6q-kit"   # NB6Q kit K1', "K1 slug")
S = once(S, 'HEAD = once(HEAD, "E25 NB5 CPU kit", "E25 NB6 CPU kit", "titles", 2)',
         'HEAD = once(HEAD, "E25 NB5 CPU kit", "E25 NB6Q CPU kit", "titles", 2)   # NB6Q kit K1', "K1 head title")
S = once(S, """HEAD = once(HEAD, '"/tmp/nb5kit"', '"/tmp/nb6kit"', "tmp dir")""",
         """HEAD = once(HEAD, '"/tmp/nb5kit"', '"/tmp/nb6qkit"', "tmp dir")   # NB6Q kit K1""", "K1 tmp")
S = once(S, """HEAD = once(HEAD, 'f"nb5_kit_{KIT}_report.json"', 'f"nb6_kit_{KIT}_report.json"', "report name")""",
         """HEAD = once(HEAD, 'f"nb5_kit_{KIT}_report.json"', 'f"nb6q_kit_{KIT}_report.json"', "report name")   # NB6Q kit K1""",
         "K1 report")
S = once(S, """    blob = base64.b64encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")""",
         """    blob = base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")   # NB6Q kit K6
    assert '"' not in blob and chr(92) not in blob""", "K6 b85 encode")
S = once(S, '''HEAD += r\'\'\'
def grid_ok(g):''', '''HEAD = once(HEAD, 'base64.b64decode("__BLOB__")', 'base64.b85decode("__BLOB__")', "NB6Q kit K6: base85 blob")
HEAD += r\'\'\'
def grid_ok(g):''', "K6 b85 decode")
S = once(S, 'PUSH6 = ROOT / "q27-xhigh-e25-nb6-submit__foysalemonshanto"',
         'PUSH6 = ROOT / "q27-xhigh-e25-nb6q-submit__foysalemonshanto"   # NB6Q kit K1: the folder under test', "K1 push6")
for kit in ("Units", "E2E Rerun", "E2E Commit", "Real"):
    S = once(S, f'"title": "Q27 E25 NB6 Kit {kit}"', f'"title": "Q27 E25 NB6Q Kit {kit}"', "K1 title " + kit)

# ---------------------------------------------------------------- K2 units: nb6q_unit_tests.py
S = once(S, '''    if not n.startswith(("rc2/", "merlin/", "push")):''',
         '''    if not n.startswith(("rc2/", "merlin/", "push", "mtree/")):   # NB6Q kit K2''', "K2 write filter")
S = once(S, '''for d_ in ("push6", "push5"):''', '''for d_ in ("push6", "push5", "push6ref"):   # NB6Q kit K2: push6ref = the NB6 folder (reference)''',
         "K2 push dirs")
_OLD_N6 = '''N6 = {}
try:
    r = sh("nb6_units", [sys.executable, str(B / "nb6_unit_tests.py")],
           {"NB6T_NB": str(B / "nb6.ipynb"), "NB6T_NB5": str(B / "nb5q.ipynb"), "NB6T_PAYLOAD": str(PAY["used"]),
            "NB6T_PUSH": str(K / "push6"), "NB6T_PUSH5": str(K / "push5"),
            "PYTHONIOENCODING": "utf-8"}, 3600, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\\d+) / (\\d+) PASS", l) for l in lines if re.match(r"\\d+ / \\d+ PASS", l)), None)
    N6.update(rc=r["rc"], sec=r["sec"], fails=[l[:400] for l in lines if l.startswith("FAIL ")][:40],
              summary=m.group(0) if m else r["tail"][-800:], tail=r["tail"][-800:],
              all_pass=bool(m) and r["rc"] == 0 and m.group(1) == m.group(2) and "@@NB6_UNITS_OK@@" in r["_out"])'''
_NEW_N6 = '''N6 = {}
try:                                       # NB6Q kit K2: nb6q_unit_tests.py S1 + S3 (S3 runs nb6_unit_tests.py itself)
    r = sh("nb6q_units", [sys.executable, str(B / "nb6q_unit_tests.py")],
           {"NB6QT_SECTIONS": "S1,S3", "NB6QT_NB": str(B / "nb6.ipynb"), "NB6QT_NB6": str(B / "nb6ref.ipynb"),
            "NB6QT_PPACK": str(B / "nb4_ppack.py"), "NB6QT_LIB": str(B / "nb6q_lib.py"),
            "NB6QT_NB6_UNITS": str(B / "nb6_unit_tests.py"), "NB6QT_PUSH": str(K / "push6"), "NB6QT_PUSH6": str(K / "push6ref"),
            "NB6T_NB5": str(B / "nb5q.ipynb"), "NB6T_PAYLOAD": str(PAY["used"]), "NB6T_PUSH5": str(K / "push5"),
            "PYTHONIOENCODING": "utf-8"}, 3600, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\\d+) / (\\d+) PASS \\((\\d+) skipped\\)", l) for l in lines if re.match(r"\\d+ / \\d+ PASS", l)), None)
    N6.update(rc=r["rc"], sec=r["sec"], fails=[l[:400] for l in lines if l.startswith("FAIL ")][:40],
              skips=[l[:300] for l in lines if l.startswith("SKIP ")], summary=m.group(0) if m else r["tail"][-800:], tail=r["tail"][-800:],
              all_pass=bool(m) and r["rc"] == 0 and m.group(1) == m.group(2) and m.group(3) == "0" and "@@NB6Q_UNITS_OK@@" in r["_out"])'''
S = once(S, _OLD_N6, _NEW_N6, "K2 units run")
S = once(S, '''     "sha_nb5q_written": hashlib.sha256((B / "nb5q.ipynb").read_bytes()).hexdigest() == "__NB5QSHA__"}''',
         '''     "sha_nb5q_written": hashlib.sha256((B / "nb5q.ipynb").read_bytes()).hexdigest() == "__NB5QSHA__",
     "sha_nb6ref_written": hashlib.sha256((B / "nb6ref.ipynb").read_bytes()).hexdigest() == "87afcc848cd60eee290509b6a1171f04a754015130d69363d9046d265b2c1e2b"}   # NB6Q kit K2''',
         "K2 nb6ref sha")
S = once(S, '''"embedded_notebooks_sha_ok": U["sha_nb6_written"] and U["sha_nb5q_written"],''',
         '''"embedded_notebooks_sha_ok": U["sha_nb6_written"] and U["sha_nb5q_written"] and U["sha_nb6ref_written"],''', "K2 verdict sha")
S = once(S, '''         "nb6_units_all_pass": N6.get("all_pass") is True},''',
         '''         "nb6q_units_S1_S3_all_pass_0_skipped": N6.get("all_pass") is True},   # NB6Q kit K2''', "K2 verdict units")
S = once(S, '''log("nb6 units", json.dumps({k: v for k, v in N6.items() if k != "tb"})[:1500])''',
         '''log("nb6q units", json.dumps({k: v for k, v in N6.items() if k != "tb"})[:1500])''', "K2 log")

# ---------------------------------------------------------------- K3 / K4 e2e driver
S = once(S, '''  commit RERUN unset -> NB6COMMIT (baked default, its own 40-min budget): the eval subset scored, CELL 0's fallback kept''',
         '''  commit RERUN unset -> NB6COMMIT (baked default, its own 50-min budget): the eval subset scored, CELL 0's fallback kept
NB6Q kit K3: the fake Merlin tree (NB6Q_FAKE_TREE) = the real ad42e37 files; rerun-retry makes it FOREIGN (the failure path)''',
         "K4 e2e doc")
S = once(S, '''ROOT = Path(os.environ["NB6_E2E_ROOT"])\n''',
         '''ROOT = Path(os.environ["NB6_E2E_ROOT"])\nQSRC = Path(os.environ["NB6Q_TREE_SRC"])         # NB6Q kit K3: the real Merlin ad42e37 files\n''',
         "K3 tree src")
S = once(S, '''for k in [k for k in os.environ if k.startswith(("NB5", "E12_", "E15B_", "FAKE_", "Q27X_"))] + [''',
         '''for k in [k for k in os.environ if k.startswith(("NB5", "E12_", "E15B_", "FAKE_", "Q27X_", "NB6Q_", "NB4_"))] + [   # NB6Q kit K3''',
         "K3 env pop")
S = once(S, '''os.environ["NB6_HB_SEC"] = "5"                                                        # the heartbeat loop runs in the e2e\n''',
         '''os.environ["NB6_HB_SEC"] = "5"                                                        # the heartbeat loop runs in the e2e
import hashlib                                     # NB6Q kit K3: the fake Merlin tree (a fresh copy per mode)
QT = WORK / "merlin_tree"
shutil.copytree(QSRC, QT)
if RETRY:                                          # the failure path: a foreign tree (pre-sha mismatch)
    with open(QT / "vllm/v1/worker/gpu/pp_utils.py", "a", encoding="utf-8", newline="\\n") as fh_:
        fh_.write("# NB6Q kit: a foreign edit (the patch must be refused)\\n")
QTREE_BEFORE = {p.relative_to(QT).as_posix(): p.read_bytes() for p in QT.rglob("*.py")}
os.environ["NB6Q_FAKE_TREE"] = str(QT)
''', "K3 tree")
S = once(S, '''    check("commit: the baked 40-min budget (no override)", lambda: G["TOTAL_BUDGET_SEC"] == 2400, G.get("TOTAL_BUDGET_SEC"))''',
         '''    check("commit: the baked 50-min budget (NB6COMMIT, no override; NB6Q kit K4: the NB6 kit's 40-min expectation was stale)",
          lambda: G["TOTAL_BUDGET_SEC"] == 3000, G.get("TOTAL_BUDGET_SEC"))''', "K4 budget")
_E2E_Q = '''
# ---- NB6Q kit K3: the Q-defer patch on the kit's fake Merlin tree (the real Merlin ad42e37 source files)
QX = G.get("NB6Q") or {}
QR = rep.get("nb6q_ppdefer") or {}
QPOST, QPRE = __QPOST__, __QPRE__
qsha = lambda r: hashlib.sha256((QT / r).read_bytes()).hexdigest()
_qtries = [t_ for b in boots for t_ in b.get("tries") or []]
_qboot = [t_ for t_ in _qtries if (t_.get("nb6q_readback") or {}).get("after_warmup")]
REC["nb6q"] = {k: QX.get(k) for k in ("state", "applied", "error", "pre", "post", "launch_env", "readbacks", "revert")}
REC["nb6q"]["env_nb4"] = [L_.get("env_nb4") for L_ in launches]
if not RETRY:
    check("NB6Q: the Q-defer patch APPLIED right after merlin_install (before any boot) on the kit's Merlin tree (the real ad42e37 "
          "files): pre shas = ad42e37, post shas of pp_utils / model_runner / gpu_worker verified on disk, recorded in nb5_report.json",
          lambda: QX["state"] == "APPLIED" and QX["applied"] is True and QX["pre"] == QPRE and QX["post"] == QPOST
          and all(qsha(r) == QPOST[r] for r in QPOST) and QR.get("state") == "APPLIED" and QR.get("post") == QPOST, REC["nb6q"])
    check("NB6Q: NB4_PPDEFER=1 (and no NB4_PPACK) reached the (fake) server process of every launch",
          lambda: launches and all(L_.get("env_nb4") == {"NB4_PPDEFER": "1", "NB4_PPACK": None} for L_ in launches), REC["nb6q"]["env_nb4"])
    check("NB6Q: read-back recorded after the warm-up and at the end of the booted try, both ENGAGED (the fake's NB4_PPQ lines), "
          "also in the report; the run went on",
          lambda: len(_qboot) == 1 and _qboot[0]["nb6q_readback"]["after_warmup"]["label"] == "ENGAGED"
          and _qboot[0]["nb6q_readback"]["final"]["label"] == "ENGAGED"
          and [x["stage"] for x in QR["readbacks"]][-2:] == ["after_warmup", "final"], json.dumps(REC["nb6q"], default=str)[:1200])
else:
    check("NB6Q failure path: a FOREIGN Merlin tree (pre-sha mismatch) -> the patch refused (FAILED_*, applied False), every file "
          "byte for byte unchanged, recorded in nb5_report.json",
          lambda: QX["state"].startswith("FAILED") and QX["applied"] is False and "PRE_SHA_MISMATCH" in QX["error"]
          and {p.relative_to(QT).as_posix(): p.read_bytes() for p in QT.rglob("*.py")} == QTREE_BEFORE and QR.get("state") == QX["state"],
          json.dumps(REC["nb6q"], default=str)[:1200])
    check("NB6Q failure path: no launch carries NB4_PPDEFER (it is set only when the patch is verified applied); every read-back "
          "recorded NOT_APPLIED (each try: final; the booted try: after_warmup too)",
          lambda: launches and all((L_.get("env_nb4") or {}).get("NB4_PPDEFER") is None for L_ in launches)
          and QX["readbacks"] and all(x["label"] == "NOT_APPLIED" for x in QX["readbacks"]) and len(_qboot) == 1,
          json.dumps(REC["nb6q"], default=str)[:1200])
    check("NB6Q failure path: the run went on WITHOUT the patch (EVAL DONE, submission.json written; the NB6 checks above hold)",
          lambda: (ph.get("EVAL") or {}).get("status") == "DONE" and SUBP.is_file())
'''.replace("__QPOST__", json.dumps(QPOST)).replace("__QPRE__", json.dumps(QPRE))
S = once(S, '''bad = [c for c in checks if not c[1]]\nfor label, ok, info in checks:''',
         _E2E_Q + '''bad = [c for c in checks if not c[1]]\nfor label, ok, info in checks:''', "K3 e2e checks")
S = once(S, '''for n in ("nb6.ipynb", "nb6_e2e.py", "nb5_fake_server.py", "e18_cell0.py", "e18_writer.py"):
    (B / n).write_text(FILES[n], encoding="utf-8", newline="\\n")''',
         '''for n in ("nb6.ipynb", "nb6_e2e.py", "nb5_fake_server.py", "e18_cell0.py", "e18_writer.py"):
    (B / n).write_text(FILES[n], encoding="utf-8", newline="\\n")
MTQ = K / "mtree"                                   # NB6Q kit K3: the real Merlin ad42e37 source files (the fake tree)
for rel in [r_[6:] for r_ in FILES if r_.startswith("mtree/")]:
    (MTQ / rel).parent.mkdir(parents=True, exist_ok=True)
    (MTQ / rel).write_bytes(FILES["mtree/" + rel].encode("utf-8"))''', "K3 e2e write")
S = once(S, '''"NB6_E2E_ROOT": str(K / "e2e"), "NB6_E2E_BUDGET": "1800",''',
         '''"NB6_E2E_ROOT": str(K / "e2e"), "NB6Q_TREE_SRC": str(MTQ), "NB6_E2E_BUDGET": "1800",''', "K3 e2e env")

# ---------------------------------------------------------------- K5 real
S = once(S, '''step("patch_60490", lambda: G["nb5_apply_60490"]())\n''',
         '''step("patch_60490", lambda: G["nb5_apply_60490"]())
step("nb6q_patch", lambda: {"nb6q": {k: G["NB6Q"].get(k) for k in ("state", "applied", "pre", "post", "tree", "apply", "verify_state",
                                                                  "error", "env", "revert", "sec")}})   # NB6Q kit K5 (applied in install)
''', "K5 step")
S = once(S, '''                         env_levers={k: s.env.get(k) for k in levers}, env_vllm_target=s.env.get("VLLM_TARGET_DEVICE"), argv_check=chk)''',
         '''                         env_levers={k: s.env.get(k) for k in levers}, env_vllm_target=s.env.get("VLLM_TARGET_DEVICE"), argv_check=chk,
                         env_nb4={k: s.env.get(k) for k in ("NB4_PPDEFER", "NB4_PPACK")})   # NB6Q kit K5''', "K5 plan env")
_REAL_Q = '''
# ---------------------------------------------------------------- NB6Q kit K5: the Q-defer patch on the REAL installed Merlin tree
NQ = {}
try:
    PQ = G["nb6q_module"]()
    QTREE = Path(G["NB6Q"]["tree"])
    QPY = (G.get("MERLIN_ENG") or {})["kv"]["MERLIN_VENV"] + "/bin/python"
    NQ["tree"] = str(QTREE)
    NQ["tree_is_install"] = str(QTREE) == (G.get("MERLIN_ENG") or {})["kv"]["MERLIN_TREE"]
    NQ["commit"] = {k: G["NB6Q"].get(k) for k in ("state", "applied", "pre", "post", "verify_state", "error", "env")}
    NQ["rerun"] = {k: GR["NB6Q"].get(k) for k in ("state", "applied", "post", "error", "env")} | {"apply": (GR["NB6Q"].get("apply") or {}).get("applied")}
    NQ["verify_before_revert"] = PQ.verify_ppack_patch(QTREE)["state"]
    rv = PQ.revert_ppack_patch(QTREE, QPY)
    NQ["revert"] = {"ok": rv.get("ok"), "after": {r_: hashlib.sha256((QTREE / r_).read_bytes()).hexdigest() for r_ in PQ.PATCHES}}
    PPT = K / "ppt"                        # nb6q_unit_tests.py S2 (the notebook's NB6Q block) + S4 (nb4_ppack_tests.py) on THIS tree
    PPT.mkdir()
    for n_ in ("nb4_ppack.py", "nb4_ppack_tests.py", "nb6q_lib.py", "nb6q_unit_tests.py", "nb6ref.ipynb"):
        (PPT / n_).write_text(FILES[n_], encoding="utf-8", newline="\\n")
    (PPT / "nb6q.ipynb").write_text(FILES["nb6.ipynb"], encoding="utf-8", newline="\\n")
    r_ = sh("nb6q_units_real_tree", [sys.executable, str(PPT / "nb6q_unit_tests.py")],
            {"NB6QT_SECTIONS": "S2,S4", "NB6QT_NB": str(PPT / "nb6q.ipynb"), "NB6QT_NB6": str(PPT / "nb6ref.ipynb"),
             "NB6QT_MERLIN_SRC": str(QTREE), "NB6QT_PPACK": str(PPT / "nb4_ppack.py"), "NB6QT_LIB": str(PPT / "nb6q_lib.py"),
             "NB6QT_PPACK_TESTS": str(PPT / "nb4_ppack_tests.py"), "PYTHONIOENCODING": "utf-8"}, 2400, cwd=PPT)
    m_ = re.search(r"(\\d+) / (\\d+) PASS \\((\\d+) skipped\\)", r_["_out"])
    NQ["units_s2_s4"] = {"rc": r_["rc"], "sec": r_["sec"], "summary": m_.group(0) if m_ else r_["tail"][-800:],
                         "fails": [l_[:300] for l_ in r_["_out"].splitlines() if l_.startswith("FAIL ")][:20],
                         "all_pass": r_["rc"] == 0 and "@@NB6Q_UNITS_OK@@" in r_["_out"] and bool(m_) and m_.group(1) == m_.group(2)
                         and m_.group(3) == "0" and int(m_.group(2)) >= 25}
    a2 = PQ.apply_ppack_patch(QTREE, QPY)
    NQ["reapply"] = {"ok": a2.get("ok"), "applied": a2.get("applied"), "py_compile": a2.get("py_compile"),
                     "after": {r_: hashlib.sha256((QTREE / r_).read_bytes()).hexdigest() for r_ in PQ.PATCHES}}
except BaseException as e:
    NQ.update(error=f"{type(e).__name__}: {e}"[:1200], tb=traceback.format_exc()[-2000:])
REP["checks"]["nb6q"] = NQ; save()
log("nb6q real tree", json.dumps({k: v for k, v in NQ.items() if k != "tb"}, default=str)[:3000])
QPRE_X, QPOST_X = __QPRE__, __QPOST__
'''.replace("__QPOST__", json.dumps(QPOST)).replace("__QPRE__", json.dumps(QPRE))
S = once(S, '''NB5L = G.get("NB5L")             # (the NB5 kit's commit-turn section below uses G / NB5L)\n''',
         _REAL_Q + '''NB5L = G.get("NB5L")             # (the NB5 kit's commit-turn section below uses G / NB5L)\n''', "K5 real tree")
S = once(S, '''    "rerun_cells_read_test_file_only_at_COMPD_never_eval_solutions": bool(RERUN_OPENS["test_challenges"])''',
         '''    "nb6q_patch_applied_on_the_real_installed_tree_pre_post_shas": (NQ.get("commit") or {}).get("state") == "APPLIED"
        and NQ.get("tree_is_install") is True and NQ["commit"].get("pre") == QPRE_X and NQ["commit"].get("post") == QPOST_X
        and NQ.get("verify_before_revert") == "applied" and NQ["commit"].get("env") == {"NB4_PPDEFER": "1"},
    "nb6q_rerun_namespace_verified_the_same_tree": (NQ.get("rerun") or {}).get("state") == "APPLIED" and NQ["rerun"].get("post") == QPOST_X
        and NQ["rerun"].get("apply") == "already_applied",
    "nb6q_every_plan_NB4_PPDEFER_1_no_NB4_PPACK": bool(ALLP) and all(b.get("env_nb4") == {"NB4_PPDEFER": "1", "NB4_PPACK": None} for b in ALLP),
    "nb6q_real_tree_revert_pre_shas_then_reapply_post_shas": (NQ.get("revert") or {}).get("ok") is True and NQ["revert"]["after"] == QPRE_X
        and (NQ.get("reapply") or {}).get("ok") is True and NQ["reapply"]["after"] == QPOST_X,
    "nb6q_units_S2_S4_on_the_real_tree_all_pass_0_skipped": (NQ.get("units_s2_s4") or {}).get("all_pass") is True,
    "rerun_cells_read_test_file_only_at_COMPD_never_eval_solutions": bool(RERUN_OPENS["test_challenges"])''', "K5 verdict")

# ---------------------------------------------------------------- kit file tables
S = once(S, '''UNIT_FILES = {"nb6.ipynb": NB_TXT, "nb5q.ipynb": NB5Q_TXT, "nb6_unit_tests.py": K5["src"]("nb6_unit_tests.py"), **PUSH_FILES,''',
         '''UNIT_FILES = {"nb6.ipynb": NB_TXT, "nb5q.ipynb": NB5Q_TXT, "nb6_unit_tests.py": K5["src"]("nb6_unit_tests.py"), **PUSH_FILES,
              **NB6Q_UNIT_FILES,   # NB6Q kit K2''', "files units")
S = once(S, '''E2E_FILES = {"nb6.ipynb": NB_TXT, "parts.json": K5["PARTS"], "make_fake.py": K5["MAKE_FAKE"], "nb6_e2e.py": NB6_E2E,
             "nb5_fake_server.py": K5["src"]("nb5_fake_server.py"), **E18_FILES}''',
         '''E2E_FILES = {"nb6.ipynb": NB_TXT, "parts.json": K5["PARTS"], "make_fake.py": K5["MAKE_FAKE"], "nb6_e2e.py": NB6_E2E,
             "nb5_fake_server.py": NB6Q_FAKE_SERVER, **E18_FILES, **NB6Q_MTREE}   # NB6Q kit K3''', "files e2e")
S = once(S, '''REAL_FILES = {"nb6.ipynb": NB_TXT, "e18_cell0.py": E18_FILES["e18_cell0.py"],''',
         '''REAL_FILES = {"nb6.ipynb": NB_TXT, "e18_cell0.py": E18_FILES["e18_cell0.py"], **NB6Q_REAL_FILES,   # NB6Q kit K5
              ''', "files real")
S = once(S, '''if __name__ == "__main__":\n    print("notebook under test"''', '''if __name__ == "__nb6_kit_main__":\n    print("notebook under test"''',
         "no main")


def src(name):
    return (HERE / name).read_text(encoding="utf-8").replace("\r\n", "\n")


# the kit-local fake server (K3): records NB4_* per launch; with NB4_PPDEFER=1 prints the NB4_PPQ line shapes of a TP2 x PP2 boot
FAKE = src("nb5_fake_server.py")
FAKE = once(FAKE, '''"env_levers": {k: os.environ.get(k) for k in ("MERLIN_SHARE_DRAFT_WORKSPACE",)}, **kw}) + "\\n")''',
            '''"env_levers": {k: os.environ.get(k) for k in ("MERLIN_SHARE_DRAFT_WORKSPACE",)},
                                "env_nb4": {k: os.environ.get(k) for k in ("NB4_PPDEFER", "NB4_PPACK")}, **kw}) + "\\n")   # NB6Q kit K3''',
            "fake env_nb4")
FAKE = once(FAKE, '''print(f"fake merlin argv: {' '.join(argv)}", flush=True)\n''',
            '''print(f"fake merlin argv: {' '.join(argv)}", flush=True)
# NB6Q kit K3 (FAKE ONLY, no engine claim): with NB4_PPDEFER=1 a real Merlin TP2 x PP2 boot prints one 'NB4_PPQ init' line per
# worker (4: 2 last-PP-rank, 2 non-last) and 'stats' lines on the non-last ranks (nb4_ppack rev 2); the fake prints the same
# shapes (clean flush ratios) so the notebook's read-back path runs end to end (label ENGAGED)
if os.environ.get("NB4_PPDEFER") == "1":
    for _qi, _qlast in enumerate((True, True, False, False)):
        _qpid = os.getpid() * 10 + _qi
        print(f"(Worker_PP{0 if not _qlast else 1}_TP{_qi % 2} pid={_qpid}) NB4_PPQ " + json.dumps(
            {"event": "init", "pid": _qpid, "t": time.time(), "env_ppack": "0", "env_ppdefer": "1", "ppack": False,
             "ppdefer": not _qlast, "is_last_rank": _qlast, "k": 3, "ring_depth": 2, "device": f"cuda:{_qi}"}, sort_keys=True), flush=True)
    for _qi in (2, 3):
        _qpid = os.getpid() * 10 + _qi
        _qst = {"packed_stash": 0, "packed_send": 0, "packed_recv": 0, "defer_park": 1000, "defer_post": 1000, "flush_exec": 990,
                "flush_empty": 10, "flush_recv": 0, "flush_consume": 0, "flush_rpc": 0, "flush_shutdown": 0}
        for _qk in ("defer_park", "flush_exec"):
            print(f"(Worker_PP0_TP{_qi % 2} pid={_qpid}) NB4_PPQ " + json.dumps(
                {"event": "stats", "pid": _qpid, "t": time.time(), "key": _qk, "stats": _qst}, sort_keys=True), flush=True)
''', "fake ppq lines")
compile(FAKE, "nb5_fake_server.py (NB6Q kit)", "exec")
assert all(f"\nimport {m}\n" in FAKE for m in ("json", "os", "time")), "fake server imports"

NB6_RAW = NB6_NB.read_bytes()
assert hashlib.sha256(NB6_RAW).hexdigest() == "87afcc848cd60eee290509b6a1171f04a754015130d69363d9046d265b2c1e2b"
PUSH6REF = Path(r"D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb6-submit__foysalemonshanto/kernel-metadata.json").read_text(encoding="utf-8")
NS = {"__file__": str(NB6KIT), "__name__": "nb6q_kit",
      "NB6Q_UNIT_FILES": {"nb6ref.ipynb": NB6_RAW.decode("utf-8"), "push6ref/kernel-metadata.json": PUSH6REF,
                          **{n: src(n) for n in ("nb6q_unit_tests.py", "nb6q_lib.py", "nb4_ppack.py")}},
      "NB6Q_MTREE": MTREE, "NB6Q_FAKE_SERVER": FAKE,
      "NB6Q_REAL_FILES": {"nb6ref.ipynb": NB6_RAW.decode("utf-8"),
                          **{n: src(n) for n in ("nb4_ppack.py", "nb4_ppack_tests.py", "nb6q_lib.py", "nb6q_unit_tests.py")}}}
_argv = sys.argv
sys.argv = [str(NB6KIT), "--nb", str(NBQ), "--export", str(EXPQ)]
try:
    exec(compile(S, str(NB6KIT) + " (NB6Q kit edits)", "exec"), NS)
finally:
    sys.argv = _argv
assert NS["NB_PATH"] == NBQ.resolve() and NS["EXPORT_OK"] is True and NS["PUSH6_OK"] is True
compile(NS["NB6_E2E"], "<nb6q e2e>", "exec")

if __name__ == "__main__":
    print("notebook under test", NS["NB_PATH"].name, "sha256", NS["NB_SHA"], "| export", NS["EXPORT_PATH"].name, "equal", flush=True)
    for n in NS["KITS"]:
        NS["build"](n)
