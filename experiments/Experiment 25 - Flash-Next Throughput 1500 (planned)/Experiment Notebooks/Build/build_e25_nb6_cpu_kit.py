"""E25 NB6 SUBMIT Kaggle CPU kits (lead 2026-11-01, URGENT): four independent CPU kernels on foysalemonshanto, pinned Py3.12
image, no GPU, no internet, private. Generic over the notebook under test: --nb <notebook.ipynb> [--export <code export>]
(defaults: ../Q27-XHIGH-E25_NB6_SUBMIT.ipynb + nb6_code_export.py; the export must equal the notebook's code cells).
Pattern: build_e25_nb5_cpu_kit.py (NB5Q kits ALL OK on Kaggle; its HEAD / fake-tree maker / RC2 unit data / commit-turn
section are imported from it, not copied) + build_e25_klee100_load_cpu_kit.py v2 (VLLM_TARGET_DEVICE=cpu: real
`vllm serve --help` and vLLM's own argparse on a GPU-less box). Each kernel is fail-soft, writes its JSON report and prints
exactly one line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.

  units       nb5_unit_tests.py on the Kaggle-run NB5 QUALITY notebook (9c71ba89 = NB6's base, kit mode, gated: all PASS,
              11 build-time SKIP), nb5_unit_tests.py on the NB6 notebook (recorded only: NB5-profile checks are expected to
              differ), nb6_unit_tests.py on the NB6 notebook (gated: all PASS, @@NB6_UNITS_OK@@).
  e2e-rerun   THE production path: every code cell of the real NB6 notebook with KAGGLE_IS_COMPETITION_RERUN=1, E11_HARNESS=1,
              nb5_fake_server.py (OpenAI-compatible fake Merlin, real Flash-Next tokenizer), the real Duck payload, a fake
              tree from the real part-1 small files, and a FAKE HIDDEN TEST SET of 6 tasks (training tasks, not in the eval
              set; >= 1 with several test inputs) mounted as arc-agi_test_challenges.json: 4 the fake can solve, 1 the fake
              knows without a solution (its accepted program returns the input), 1 the fake does not know (no accepted
              answer -> identity fallback). Fake deadline Q27X_TIME_BUDGET = 1800 s. Asserts: PROFILE == RERUN_PROD (notebook
              + report), RERUN True, TASK_IDS == every test task, no eval data loaded; one EVAL phase DONE; the solver ran
              (traced requests, fake tool requests, every test task attempted); submission.json exists, is valid JSON,
              covers EVERY test task id with one {attempt_1, attempt_2} per test input, every grid a non-empty rectangular
              list of lists of ints 0-9; both unsolved tasks = the identity fallback; solved tasks only carry the solution or
              the input, >= 1 solved task pass@2 fully correct; the notebook finished before NOTEBOOK_T0 + TOTAL_BUDGET_SEC
              (= 1800) and submission.json was last written before the hard stop; capacity = the production pin ladder's
              first step (pool 214, F 5 fake lever, 9 workers), the lever env reached the fake server; bundle zip.
  e2e-commit  the same with RERUN unset: PROFILE == NB6COMMIT (baked default, 40-min budget unchanged), the eval subset (its
              task_list) scored by exact match with >= 1 hit, the CELL 0 fallback submission.json = the identity of the
              test file (full schema, ints 0-9), finished before its budget.
  real        the notebook's cells 0-4 (CELL 0, 1, 3 Duck, task order, 5 engine; never 6/7, 8, 9) on the REAL mounts (NB5
              QUALITY's kernel-metadata sources: the 7 RC2 Models + the payload dataset + the competition), twice: commit
              namespace (NB6COMMIT) and rerun namespace (KAGGLE_IS_COMPETITION_RERUN=1 -> RERUN_PROD; the state set by the
              commit namespace's inputs / prompts / install steps is shared, no second merge or sha pass). Payload sha +
              path, sandbox self-test, inputs, merge, prompt hashes, full sha256, the real Merlin offline install, #60490
              applied + verified (post sha 88fe0a8f), then for EVERY boot try of BOTH profiles' plans (each (MTP, KV) group
              of P['nb5_phases'] x the pin ladder, seqs table, lever env, CAPS16) launch.build() + host_phase() + `vllm serve
              --help=all` with VLLM_TARGET_DEVICE=cpu (every flag present) + argv checks (KV bytes, seqs, captures cover seqs
              x 4, lever env in the server env, speculative config, KV dtype = RC2's) + vLLM's own argparse in the Merlin venv
              (make_arg_parser + parse_known_args + AsyncEngineArgs.from_cli_args: no unknown flag, the fields), no server
              started, RERUN_PROD plans == NB6COMMIT plans; the commit-turn A/C test + budget guard on the real tokenizer
              (NB5 kit section, imported); CELL 0's fallback on the real test file in both namespaces (strict full schema).

Rules: no Kaggle API calls here (the lead pushes); no local e2e / whole-notebook runs (user rule: long tests on Kaggle only).
Run: python build_e25_nb6_cpu_kit.py [--nb PATH] [--export PATH]
  -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb6-kit-{units,e2e-rerun,e2e-commit,real}__foysalemonshanto/
"""
import base64
import hashlib
import json
import lzma
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _arg(flag, default):
    return Path(sys.argv[sys.argv.index(flag) + 1]).resolve() if flag in sys.argv else default


NB_PATH = _arg("--nb", HERE.parent / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb")
EXPORT_PATH = _arg("--export", HERE / "nb6_code_export.py")
NB_RAW = NB_PATH.read_bytes()
assert b"\r" not in NB_RAW, "notebook has CR bytes"
NB_TXT = NB_RAW.decode("utf-8")
NB_SHA = hashlib.sha256(NB_RAW).hexdigest()
_code = ["".join(c["source"]) for c in json.loads(NB_TXT)["cells"] if c["cell_type"] == "code"]
EXPORT_OK = None
if EXPORT_PATH.is_file():
    EXPORT_OK = "\n\n".join(_code) == EXPORT_PATH.read_bytes().decode("utf-8")
    assert EXPORT_OK, f"{EXPORT_PATH.name} != the code cells of {NB_PATH.name} (rebuild one of them)"
NB5Q_PATH = HERE.parent / "Q27-XHIGH-E25_NB5_QUALITY.ipynb"
NB5Q_SHA = "9c71ba89e4db75b5f9a63a4be652e860b629120685e8f534bed3bb8d06769049"
NB5Q_TXT = NB5Q_PATH.read_bytes().decode("utf-8")
assert hashlib.sha256(NB5Q_PATH.read_bytes()).hexdigest() == NB5Q_SHA, "NB5 QUALITY is not the Kaggle-run 9c71ba89"

K5 = runpy.run_path(str(HERE / "build_e25_nb5_cpu_kit.py"), run_name="nb5kit")      # module only (its build() is not called)
KLEE = (HERE / "build_e25_klee100_load_cpu_kit.py").read_text(encoding="utf-8")
# ---- the proven eval / private-run rules (lead 2026-11-01: user requirement): E18 R2 (LB 31.53), its CELL 0 (mode + path +
# identity fallback) and its submission writer e10_write_submission, cut by ast from the notebook (sha-pinned)
E18_NB = Path(r"D:/ARC AGI 2/experiments/Experiment 18 - S1-60K on vLLM 0.32 (planned)/Experiment Notebooks/"
              r"Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb")
E18_SHA = "116d2d4101a53eb4ff833e9c97e5a779f0f2bae5735129a4663ad1b42ad79a1d"
assert hashlib.sha256(E18_NB.read_bytes()).hexdigest() == E18_SHA, "E18 R2 notebook changed"
_e18 = ["".join(c["source"]) for c in json.loads(E18_NB.read_bytes().decode("utf-8"))["cells"] if c["cell_type"] == "code"]
E18_CELL0 = _e18[0]
assert E18_CELL0.startswith("# ==================================================================== CELL 0") and "identity_submission" in E18_CELL0
import ast  # noqa: E402
_w = [n for c in _e18 for n in ast.parse(c).body if isinstance(n, ast.FunctionDef) and n.name == "e10_write_submission"]
assert len(_w) == 1
E18_WRITER = ast.get_source_segment(next(c for c in _e18 if "def e10_write_submission" in c), _w[0])
assert "submission=identity_submission(test_challenges)" in E18_WRITER and "if RERUN:" in E18_WRITER
E18_FILES = {"e18_cell0.py": E18_CELL0.replace("\r\n", "\n"), "e18_writer.py": E18_WRITER.replace("\r\n", "\n")}
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
SLUG = "q27-e25-nb6-kit"
IMAGE = K5["IMAGE"]
PAYLOAD_DS = K5["PAYLOAD_DS"]
RC2_MODELS = K5["RC2_MODELS"]
QMETA = json.loads((ROOT / "q27-xhigh-e25-nb5-quality__foysalemonshanto" / "kernel-metadata.json").read_text(encoding="utf-8"))
assert QMETA["docker_image"] == IMAGE and QMETA["model_sources"] == RC2_MODELS and QMETA["dataset_sources"] == [PAYLOAD_DS]


def once(text, old, new, what, count=1):
    assert text.count(old) == count, (what, text.count(old))
    return text.replace(old, new)


def between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


# ---- HEAD: the NB5 kit's (find_payload, copy_tok, sh, verdict, payload names), retargeted to the NB6 notebook
HEAD = K5["HEAD"]
HEAD = once(HEAD, 'FILES["nb5.ipynb"]', 'FILES["nb6.ipynb"]', "embedded notebook key")
HEAD = once(HEAD, "E25 NB5 CPU kit", "E25 NB6 CPU kit", "titles", 2)
HEAD = once(HEAD, '"/tmp/nb5kit"', '"/tmp/nb6kit"', "tmp dir")
HEAD = once(HEAD, 'f"nb5_kit_{KIT}_report.json"', 'f"nb6_kit_{KIT}_report.json"', "report name")
HEAD += r'''
def grid_ok(g):
    return (isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) and len(r) > 0 and len(r) == len(g[0])
            and all(type(v) is int and 0 <= v <= 9 for v in r) for r in g))
def strict_schema(sub, test):
    """every test task id, one {attempt_1, attempt_2} per test input, every grid rectangular ints 0-9 (no extra ids)"""
    bad = []
    if not isinstance(sub, dict):
        return ["not a dict"]
    if set(sub) != set(test):
        bad.append(f"ids: missing {sorted(set(test) - set(sub))[:5]} extra {sorted(set(sub) - set(test))[:5]}")
    for t in test:
        rows = sub.get(t)
        if not isinstance(rows, list) or len(rows) != len(test[t]["test"]):
            bad.append(f"{t}: {type(rows).__name__} len {len(rows) if isinstance(rows, list) else None} != {len(test[t]['test'])}")
            continue
        for j, a in enumerate(rows):
            if not isinstance(a, dict) or set(a) != {"attempt_1", "attempt_2"} or not all(grid_ok(a[k]) for k in ("attempt_1", "attempt_2")):
                bad.append(f"{t}[{j}] entry")
    return bad
'''

# ---------------------------------------------------------------------------------------------------------- units
UNITS = r'''
PAY = find_payload(); REP["checks"]["payload"] = PAY; save()
log("payload", json.dumps(PAY)[:1200])
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    if not n.startswith(("rc2/", "merlin/", "push")):
        (B / n).write_text(v, encoding="utf-8", newline="\n")
for d_ in ("push6", "push5"):
    (K / d_).mkdir()
    (K / d_ / "kernel-metadata.json").write_text(FILES[d_ + "/kernel-metadata.json"], encoding="utf-8", newline="\n")
(K / "push6" / FILES["push6/code_file"]).write_bytes((B / "nb6.ipynb").read_bytes())    # = the NB6 build (sha checked below)
RC = K / "rc2"
for rel in [r_[4:] for r_ in FILES if r_.startswith("rc2/")]:
    (RC / rel).parent.mkdir(parents=True, exist_ok=True)
    (RC / rel).write_text(FILES["rc2/" + rel], encoding="utf-8", newline="\n")
MT = K / "merlin_tools"
for rel in [r_[7:] for r_ in FILES if r_.startswith("merlin/")]:
    (MT / rel).parent.mkdir(parents=True, exist_ok=True)
    (MT / rel).write_text(FILES["merlin/" + rel], encoding="utf-8", newline="\n")
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5", "NB6", "E12_", "E15B_", "Q27X_", "RC2_", "E23_", "FAKE_"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
U = {"sha_nb6_written": hashlib.sha256((B / "nb6.ipynb").read_bytes()).hexdigest() == "__NBSHA__",
     "sha_nb5q_written": hashlib.sha256((B / "nb5q.ipynb").read_bytes()).hexdigest() == "__NB5QSHA__"}
PAY_OK = PAY["used"] is not None
def run_nb5_units(label, nb):
    u = {}
    try:
        r = sh(label, [sys.executable, str(B / "nb5_unit_tests.py")],
               {"NB5T_KIT": "1", "NB5T_RC2": str(RC), "NB5T_TOKDIR": str(K / "fn_tok"), "NB5T_PAYLOAD": str(PAY["used"]),
                "NB5T_NB": str(nb), "NB5T_CT_RESULT": str(B / "ct_result.json"), "NB5T_MERLIN_TOOLS": str(MT),
                "PYTHONIOENCODING": "utf-8"}, 3600, cwd=B)
        lines = r["_out"].splitlines()
        m = next((re.match(r"(\d+)/(\d+) PASS(?: \((\d+) build-time)?", l) for l in lines if re.match(r"\d+/\d+ PASS", l)), None)
        u.update(rc=r["rc"], sec=r["sec"], fails=[l[:300] for l in lines if l.startswith("FAIL ")][:60],
                 summary=m.group(0) if m else r["tail"][-800:], passed=int(m.group(1)) if m else None,
                 total=int(m.group(2)) if m else None, skipped=int(m.group(3) or 0) if m else None, tail=r["tail"][-600:])
        u["all_pass"] = (bool(m) and r["rc"] == 0 and u["passed"] == u["total"] and u["total"] >= __UEXP_RUN__
                         and u["skipped"] == __UEXP_SKIP__ and not u["fails"])
    except BaseException as e:
        u.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
    return u
try:
    U["tok_shas"] = copy_tok(K / "fn_tok")
except BaseException as e:
    U["tok_error"] = f"{type(e).__name__}: {e}"[:600]
U["nb5_units_on_nb5q"] = run_nb5_units("nb5_units_nb5q", B / "nb5q.ipynb"); save()
log("nb5 units on NB5 QUALITY", json.dumps({k: v for k, v in U["nb5_units_on_nb5q"].items() if k != "tb"})[:1500])
U["nb5_units_on_nb6"] = run_nb5_units("nb5_units_nb6", B / "nb6.ipynb"); save()            # recorded only
log("nb5 units on NB6 (recorded)", json.dumps({k: v for k, v in U["nb5_units_on_nb6"].items() if k != "tb"})[:1500])
N6 = {}
try:
    r = sh("nb6_units", [sys.executable, str(B / "nb6_unit_tests.py")],
           {"NB6T_NB": str(B / "nb6.ipynb"), "NB6T_NB5": str(B / "nb5q.ipynb"), "NB6T_PAYLOAD": str(PAY["used"]),
            "NB6T_PUSH": str(K / "push6"), "NB6T_PUSH5": str(K / "push5"),
            "PYTHONIOENCODING": "utf-8"}, 3600, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+) / (\d+) PASS", l) for l in lines if re.match(r"\d+ / \d+ PASS", l)), None)
    N6.update(rc=r["rc"], sec=r["sec"], fails=[l[:400] for l in lines if l.startswith("FAIL ")][:40],
              summary=m.group(0) if m else r["tail"][-800:], tail=r["tail"][-800:],
              all_pass=bool(m) and r["rc"] == 0 and m.group(1) == m.group(2) and "@@NB6_UNITS_OK@@" in r["_out"])
except BaseException as e:
    N6.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["nb6_units"] = N6
REP["checks"]["units"] = U; save()
log("nb6 units", json.dumps({k: v for k, v in N6.items() if k != "tb"})[:1500])
verdict({"payload_sha_ok_under_input": PAY_OK, "embedded_notebooks_sha_ok": U["sha_nb6_written"] and U["sha_nb5q_written"],
         "nb5_units_all_pass_on_nb5_quality": U["nb5_units_on_nb5q"].get("all_pass") is True,
         "nb6_units_all_pass": N6.get("all_pass") is True},
        recorded={"nb5_units_nb5q": U["nb5_units_on_nb5q"].get("summary"), "nb5_units_nb6": U["nb5_units_on_nb6"].get("summary"),
                  "nb5_units_nb6_fails": U["nb5_units_on_nb6"].get("fails"), "nb6_units": N6.get("summary"),
                  "payload_used": PAY.get("used")})
'''

# ---------------------------------------------------------------------------------------------------------- e2e driver
NB6_E2E = r'''"""E25 NB6 e2e (Kaggle CPU kit only): every code cell of the real NB6 notebook, in order, in one namespace, against
nb5_fake_server.py (E11_HARNESS=1, no GPU), the real Duck payload and a fake 6-task hidden test set.
usage: python nb6_e2e.py <notebook.ipynb> rerun|commit
  rerun  KAGGLE_IS_COMPETITION_RERUN=1 -> RERUN_PROD on the fake hidden test file, fake deadline NB6_E2E_BUDGET (1800 s)
  rerun-retry  the same + FAKE_OOM_ABOVE_BYTES = RC2 pin + 0.8 GiB (the +1.0 GiB first engine attempt fails with an OOM
         line -> NB6 P3 = E18 R2's engine ladder: the next attempt, pin +0.75 GiB, 15-min ready limit, boots), NO evaluation
         file mounted (F1: the probe prompts come from the test file), budget NB6_E2E_BUDGET_RETRY (3000 s: every further
         ladder try needs >= 25 min left)
  commit RERUN unset -> NB6COMMIT (baked default, its own 40-min budget): the eval subset scored, CELL 0's fallback kept
Output: one line per check, then @@NB6_E2E_OK@@ (exit 0) or @@NB6_E2E_FAIL@@ (exit 1)."""
import faulthandler
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

NB = Path(sys.argv[1]).resolve()
MODE = sys.argv[2]
assert MODE in ("rerun", "rerun-retry", "commit"), MODE
RR, RETRY = MODE != "commit", MODE == "rerun-retry"
HERE = Path(__file__).resolve().parent
FAKE = HERE / "nb5_fake_server.py"
SRC_INPUT, DATA, PAYSRC = Path(os.environ["NB6_E2E_INPUT"]), Path(os.environ["NB6_E2E_DATA"]), Path(os.environ["NB6_E2E_PAYSRC"])
ROOT = Path(os.environ["NB6_E2E_ROOT"])
BUDGET = int(os.environ.get("NB6_E2E_BUDGET_RETRY", 3000)) if RETRY else int(os.environ.get("NB6_E2E_BUDGET", 1800))
sys.stdout.reconfigure(encoding="utf-8")
faulthandler.dump_traceback_later(float(os.environ.get("NB6_E2E_MAX_SEC", 3000)), exit=True)
WORK = ROOT / f"run_{MODE}_{time.strftime('%H%M%S')}"
WORK.mkdir(parents=True)
INP = WORK / "inputs"
shutil.copytree(SRC_INPUT, INP)
COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"
COMP.mkdir(parents=True, exist_ok=True)
ch = json.load(open(DATA / "arc-agi_evaluation_challenges.json"))
sol = json.load(open(DATA / "arc-agi_evaluation_solutions.json"))
for n_ in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
    shutil.copy(DATA / n_, COMP / n_)


# ---- the fake hidden test set: training tasks (never in the eval set), the smallest first, the fake's task signature unique
def _counts(g):
    d = {}
    for r in g:
        for v in r:
            d[v] = d.get(v, 0) + 1
    return dict(sorted(d.items()))


def sig_lines(t, k=2):         # = nb5_fake_server._r6_sig (k=2) / the same lines for every train pair (k=None)
    out_ = []
    for p in (t["train"][:k] if k else t["train"]):
        for tag, g in (("in ", p["input"]), ("out", p["output"])):
            out_.append(f"  {tag} {len(g)}x{len(g[0])} colors {_counts(g)}")
    return out_


SRC_KIND = "training"
if (DATA / "arc-agi_training_challenges.json").is_file() and (DATA / "arc-agi_training_solutions.json").is_file():
    tch, tsol = json.load(open(DATA / "arc-agi_training_challenges.json")), json.load(open(DATA / "arc-agi_training_solutions.json"))
    pool = {t: v for t, v in tch.items() if t not in ch and t in tsol}
else:                           # fallback (recorded): eval tasks outside the commit / smoke lists
    SRC_KIND, tch, tsol = "evaluation", ch, sol
    pool = {t: v for t, v in ch.items() if t not in {"1818057f", "2ba387bc", "45a5af55", "0934a4d8"}}
cells_ = lambda t: sum(len(p["input"]) * len(p["input"][0]) + len(p["output"]) * len(p["output"][0]) for p in t["train"]) + sum(
    len(q["input"]) * len(q["input"][0]) for q in t["test"])
cand = sorted(pool, key=lambda t: (cells_(pool[t]), t))


def unique(t, others):
    full = set(sig_lines(pool[t], None))
    return all(not set(sig_lines(o)) <= full for k, o in others.items() if k != t) and not any(
        set(sig_lines(pool[t])) <= set(sig_lines(o, None)) for k, o in others.items() if k != t)


multi, single = [], []
for t in cand:
    if len(multi) >= 3 and len(single) >= 3:
        break
    lst = multi if len(pool[t]["test"]) >= 2 else single
    if len(lst) >= (3 if lst is multi else 5) or len(pool[t]["train"]) < 2:
        continue
    if unique(t, {**ch, **{k: pool[k] for k in multi + single}}):
        lst.append(t)
N_MULTI_REAL = len(multi)
while len(multi) < 3 and len(single) > 3:      # fewer multi-test tasks than wanted: fill with single ones (recorded)
    multi.append(single.pop())
single = single[:3]
assert N_MULTI_REAL >= 1 and len(multi) == 3 and len(single) == 3, (multi, single)
SOLVED = multi[:2] + single[:2]
KNOWN_NOSOL = multi[2]          # the fake knows the task but has no test solution: its accepted program returns the input
UNKNOWN = single[2]             # the fake does not know the task: no accepted answer -> identity fallback
HIDDEN = sorted(SOLVED + [KNOWN_NOSOL, UNKNOWN])
test = {t: {"train": pool[t]["train"], "test": [{"input": q["input"]} for q in pool[t]["test"]]} for t in HIDDEN}
json.dump(test, open(COMP / "arc-agi_test_challenges.json", "w"))
FA = WORK / "fake_arc"         # what the fake server knows (its own CH / SOL files; the notebook never reads this dir)
FA.mkdir()
json.dump({**ch, **{t: pool[t] for t in SOLVED + [KNOWN_NOSOL]}}, open(FA / "arc-agi_evaluation_challenges.json", "w"))
json.dump({**sol, **{t: tsol[t] for t in SOLVED}}, open(FA / "arc-agi_evaluation_solutions.json", "w"))
SET = {"source": SRC_KIND, "solved": SOLVED, "known_no_solution": KNOWN_NOSOL, "unknown": UNKNOWN,
       "test_inputs": {t: len(test[t]["test"]) for t in HIDDEN}}
print("[nb6 e2e] fake hidden test set", json.dumps(SET), flush=True)
pd = INP / "datasets" / "foysalemonshanto" / "q27-e15b-payload"
pd.mkdir(parents=True, exist_ok=True)
shutil.copy(PAYSRC / "q27x_e15b_duck_9e4b89285136.bin", pd / "q27x_e15b_duck_9e4b89285136.bin")
for d in ("tmp", "shm", "working", "fake_state"):
    (WORK / d).mkdir()
with socket.socket() as s_:
    s_.bind(("127.0.0.1", 0))
    PORT = s_.getsockname()[1]
for k in [k for k in os.environ if k.startswith(("NB5", "E12_", "E15B_", "FAKE_", "Q27X_"))] + [
        "KAGGLE_IS_COMPETITION_RERUN", "E11_HARNESS", "LOCAL_ANALYZER_SEED"]:
    os.environ.pop(k, None)
os.environ.update(E11_HARNESS="1", E23_FAKE_SERVER=str(FAKE), E23_INPUT=str(INP), E23_TMP=str(WORK / "tmp"), E23_WIN_SEC="4",
                  E23_EST_SCALE="0.01", RC2_SHM=str(WORK / "shm"), Q27X_WORKING=str(WORK / "working"), Q27X_ARC_ROOT=str(COMP),
                  Q27X_PAYLOAD_DIR=str(INP), E12_DIR=str(WORK / "e12_duck"), NB5_ATTEMPT_B0="6000", NB5_ATTEMPT_B1="4000",
                  NB5_RESERVE="1000", E12_R6_MIN_ADMIT_SEC="5", E12_B_MAX_ATTEMPTS="3",
                  ARC3_HTTP_RETRY_BASE_SECONDS="0.05", ARC3_HTTP_RETRY_MAX_SECONDS="0.1", NB5_FAKE_ARC=str(FA),
                  FAKE_STATE_DIR=str(WORK / "fake_state"), FAKE_SPEED="3000", PYTHONIOENCODING="utf-8", NB5_SAMPLER_SEC="2",
                  E15B_WD_POLL_SEC="2", NB5_PORT=str(PORT), NB5_SMOKE_GATE_SEC="300", NB5_THINK_FLOOR="3000",
                  NB5_FAKE_HOST_FLOOR_MIB="1024")   # KIT-6 (FAKE only: the 30 GiB CPU box)
os.environ["NB6_HB_SEC"] = "5"                                                        # the heartbeat loop runs in the e2e
if RR:
    os.environ.update(KAGGLE_IS_COMPETITION_RERUN="1", Q27X_TIME_BUDGET=str(BUDGET))     # the fake deadline
if RETRY:
    os.environ["FAKE_OOM_ABOVE_BYTES"] = str(2479838700 + int(0.8 * 2 ** 30))         # RC2 FP8 pin + 0.8 GiB
    for n_ in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
        (COMP / n_).unlink(missing_ok=True)                                            # F1: no eval file mounted
NO_EVAL = not any((COMP / n_).exists() for n_ in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"))
nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
heads = [c.splitlines()[0][:90] for c in code]
print(f"[nb6 e2e] {NB.name} | mode {MODE} | {len(code)} code cells | work {WORK}", flush=True)
OPENS, _REC = [], [False]          # every ARC data file the notebook process opens while its cells run (audit hook)


def _audit(ev, args):
    if ev == "open" and _REC[0] and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p_ = os.fsdecode(args[0])
        if "arc-agi_" in p_:
            OPENS.append(p_)


sys.addaudithook(_audit)
G = {"__name__": "__main__"}
cell_err, cell_sec = None, []
t0 = time.perf_counter()
_REC[0] = True
for i, c in enumerate(code):
    t_ = time.perf_counter()
    try:
        exec(compile(c, f"<cell{i}>", "exec"), G)
    except BaseException:
        import traceback
        cell_err = (i, heads[i], traceback.format_exc()[-4000:])
        print(f"[nb6 e2e] UNCAUGHT in code cell {i}: {cell_err[2]}", flush=True)
        break
    finally:
        cell_sec.append(round(time.perf_counter() - t_, 1))
T_END = time.perf_counter()
_REC[0] = False
faulthandler.cancel_dump_traceback_later()
print(f"[nb6 e2e] cells done in {T_END - t0:.0f} s | per cell {cell_sec}", flush=True)

checks = []


def check(label, cond, info=""):
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, info = False, f"check raised {type(e).__name__}: {e} | {info}"
    checks.append((label, ok, "" if ok else str(info)[:1500]))


W = WORK / "working"
rep = json.load(open(W / "q27x" / "nb5_report.json", encoding="utf-8")) if (W / "q27x" / "nb5_report.json").exists() else {}
ph = rep.get("phases") or {}
launches = [json.loads(l) for l in open(WORK / "fake_state" / "launches.jsonl")] if (WORK / "fake_state" / "launches.jsonl").exists() else []
stats = [json.load(open(p)) for p in (WORK / "fake_state").glob("stats_*.json")]
SUBP = W / "submission.json"
sub_err, sub = None, None
try:
    sub = json.loads(SUBP.read_text(encoding="utf-8"))
except Exception as e:
    sub_err = f"{type(e).__name__}: {e}"
P = G.get("P") or {}
want_profile = "RERUN_PROD" if RR else "NB6COMMIT"
check("every code cell ran without an uncaught exception", cell_err is None, cell_err)
check("NB5 layout: 8 code cells (CELL 0, 1, 3, order, 5, 6/7, 8, 9)", len(code) == 8, heads)
check(f"PROFILE == {want_profile} in the notebook, CONFIG and the report; RERUN == {RR}",
      lambda: G["PROFILE"] == want_profile and G["CONFIG"]["PROFILE"] == want_profile and rep["profile"] == want_profile
      and G["RERUN"] is (RR) and rep["rerun"] is (RR), (G.get("PROFILE"), rep.get("profile"), G.get("RERUN")))
check("NB5 report has no runner error", lambda: rep and not rep.get("errors"), rep.get("errors"))
check("Duck payload taken from datasets/foysalemonshanto/q27-e15b-payload (path recorded in the report config)",
      lambda: Path(rep["config"]["payload_used"]["q27x_e15b_duck_9e4b89285136.bin"]["path"]) == pd / "q27x_e15b_duck_9e4b89285136.bin",
      (rep.get("config") or {}).get("payload_used"))
want_ph = [p[0] for p in P.get("nb5_phases") or []]
check("phases run = the profile's phases (['EVAL']), all DONE",
      lambda: want_ph == ["EVAL"] and list(ph) == want_ph and all(ph[n].get("status") == "DONE" for n in want_ph),
      {n: (p.get("status"), (p.get("error") or "")[-600:]) for n, p in ph.items()})
boots = rep.get("boots") or []
b0 = boots[-1] if boots else {}          # the boot that ran the phase
_bsum = json.dumps([{k: b.get(k) for k in ("tag", "status", "kv_dtype", "mtp", "nb6_error")} | {"tries": [(t_.get("pin_gib"),
                    t_.get("status"), bool((t_.get("ladder") or {}).get("oom_lines"))) for t_ in b.get("tries") or []]} for b in boots], default=str)[:1500]
if RETRY:
    check("boot retry (NB6 P3 = E18 R2 engine ladder): one boot; attempt 1 (pin +1.0 GiB, lever env) failed with an OOM line "
          "before any phase, attempt 2 (pin +0.75 GiB, the same lever env) booted (FP8, MTP on, probes ok); <= 3 attempts; "
          "the ready limit of the later attempt is 15 min (E18 R2: 30 / 15 / 15)",
          lambda: len(boots) == 1 and [t_["pin_gib"] for t_ in b0["tries"]] == [1.0, 0.75] and b0["tries"][0]["ladder"]["failed"]
          and b0["tries"][0]["ladder"]["oom_lines"] and b0["tries"][1].get("status") in (None, "DONE") and b0["pin_used"]["gib"] == 0.75
          and b0["tries"][1]["levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"} and b0["kv_dtype"] == "fp8" and b0["mtp"] is True
          and (b0.get("probes") or {}).get("fatal") is None and G.get("NB6_READY_SEC") == 15 * 60, _bsum + f" | NB6_READY_SEC {G.get('NB6_READY_SEC')}")
else:
    check("one boot (FP8, MTP on), first engine attempt booted (30-min ready limit), probes without a fatal",
          lambda: len(boots) == 1 and b0["kv_dtype"] == "fp8" and b0["mtp"] is True and (b0.get("probes") or {}).get("fatal") is None
          and (G.get("NB6_READY_SEC") in (None, 30 * 60) if "NB6_READY_SEC" in code[4] else True), _bsum)
cap = b0.get("capacity") or {}
_cw = (0.75, 197, 5, 8, 8) if RETRY else (1.0, 214, 5, 9, 9)     # pin, fake pool, F, launch seqs, workers
check(f"capacity of the booted try: pin +{_cw[0]} GiB, fake pool {_cw[1]} from cache_config_info, F_log {_cw[2]} used as F (measured "
      f"rule), launch max_num_seqs {_cw[3]} (seqs table), workers = min({_cw[1]} // (16 + {_cw[2]}), {_cw[3]}) = {_cw[4]}",
      lambda: b0["pin_used"]["gib"] == _cw[0] and cap["pool_blocks"] == _cw[1] and cap["pool_source"] == "cache_config_info"
      and abs(cap["f_log"] - _cw[2]) < 0.06 and cap["f_eff"] == _cw[2] and cap["caps"]["max_num_seqs"] == _cw[3] and cap["workers"] == _cw[4],
      json.dumps({"tries": [t_.get("pin_gib") for t_ in b0.get("tries") or []], "cap": cap}, default=str)[:900])
_lv = {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}
check("lever env MERLIN_SHARE_DRAFT_WORKSPACE=1 reached the (fake) server process that booted and the boot record",
      lambda: [L_["env_levers"] for L_ in launches if L_["outcome"] == "ready"] == [_lv] and b0["server_env_levers"] == _lv,
      json.dumps({"launches": [(L_.get("outcome"), L_.get("env_levers")) for L_ in launches], "b0": b0.get("server_env_levers")}))
check("no pure decode window (no_pure_decode in both NB6 profiles)", "pure_decode" not in b0)
main = ph.get("EVAL") or {}
tr = main.get("tracer") or {}
check("the solver ran: EVAL requests traced (tool-call rate + prefix hits reported), fake tool requests > 0, no non-leading system",
      lambda: tr["requests"] > 0 and tr["tool_call_rate"] is not None and tr["prefix_hit_pct"] is not None
      and tr["non_leading_system_requests"] == 0 and sum(s.get("tool_requests", 0) for s in stats) > 0,
      json.dumps({k: tr.get(k) for k in ("requests", "tool_call_rate", "prefix_hit_pct", "non_leading_system_requests")}))
check("commit turns sent as variant A (NB6 profiles), with tool calls",
      lambda: "C" not in (tr.get("commit_by_variant") or {}) and tr["commit_by_variant"]["A"]["with_tool_call"] > 0,
      json.dumps(tr.get("commit_by_variant")))
_txt = " ".join(p.read_text(encoding="utf-8", errors="replace") for p in (W / "q27x").rglob("*.json") if p.stat().st_size < 64 * 2 ** 20) \
    + " ".join(str(p) for p in (W / "q27x").rglob("*"))
SOLVE_IDS = sorted(test) if RR else sorted(P.get("task_list") or [])
check(f"every task of the solve set was attempted (its id in the run's records): {SOLVE_IDS}",
      lambda: SOLVE_IDS and all(t in _txt for t in SOLVE_IDS), [t for t in SOLVE_IDS if t not in _txt])
check("reservation: grants > 0, never over capacity, released at the end",
      lambda: main["reservation"]["grants"] > 0 and main["reservation"]["max_used"] <= main["reservation"]["capacity_blocks"]
      and main["reservation"]["used"] == 0, json.dumps(main.get("reservation"))[:600])
check("no BUDGET_LIMITED attempt (reachable floor)", lambda: main["budget_limited_attempts"] == 0, main.get("budget_limited_attempts"))
check("fake server: turn N+1 prompt extended turn N's prompt in every checked Duck turn",
      lambda: sum(s.get("prefix_chain_checked", 0) for s in stats) > 0
      and all(s.get("prefix_chain_ok") == s.get("prefix_chain_checked") for s in stats),
      [(s.get("prefix_chain_ok"), s.get("prefix_chain_checked")) for s in stats])
# ---- submission.json
def grid_ok(g):
    return (isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) and len(r) > 0 and len(r) == len(g[0])
            and all(type(v) is int and 0 <= v <= 9 for v in r) for r in g))
check("submission.json exists and is valid JSON", SUBP.is_file() and sub_err is None, sub_err)
check("submission.json covers EVERY test task id (no extra), one {attempt_1, attempt_2} per test input, every grid a non-empty "
      "rectangular list of lists of ints 0-9",
      lambda: isinstance(sub, dict) and set(sub) == set(test) and all(len(sub[t]) == len(test[t]["test"]) for t in test)
      and all(isinstance(a, dict) and set(a) == {"attempt_1", "attempt_2"} and grid_ok(a["attempt_1"]) and grid_ok(a["attempt_2"])
              for t in test for a in sub[t]), {t: (len((sub or {}).get(t) or []), len(test[t]["test"])) for t in test})
check(">= 1 multi-test-input task in the test set, and it got one entry per input", lambda: any(
    len(test[t]["test"]) >= 2 and len(sub[t]) == len(test[t]["test"]) for t in test), SET["test_inputs"])
check("no leftover temp files next to submission.json", lambda: not list(W.glob("submission.json*.tmp")), list(W.glob("*.tmp")))
ident = lambda t: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in test[t]["test"]]
REC = {}
if RR:
    tsol_ = {t: tsol[t] for t in SOLVED}
    full_ok = [t for t in SOLVED if all(sub[t][j]["attempt_1"] == tsol_[t][j] or sub[t][j]["attempt_2"] == tsol_[t][j]
                                        for j in range(len(test[t]["test"])))] if isinstance(sub, dict) else []
    REC.update(solved_pass2=full_ok, attempts={t: sub.get(t) == ident(t) for t in test} if isinstance(sub, dict) else None)
    check("unsolved tasks (the fake's unknown task, and the known task without a solution) = the identity fallback, every input",
          lambda: sub[UNKNOWN] == ident(UNKNOWN) and sub[KNOWN_NOSOL] == ident(KNOWN_NOSOL),
          {t: sub.get(t) for t in (UNKNOWN, KNOWN_NOSOL)} if isinstance(sub, dict) else sub_err)
    check("solved tasks carry only the solution or the input (no other grid) in both attempts",
          lambda: all(sub[t][j][k] in (tsol_[t][j], test[t]["test"][j]["input"]) for t in SOLVED for j in range(len(test[t]["test"]))
                      for k in ("attempt_1", "attempt_2")), "")
    check(">= 1 solved task pass@2 fully correct (model answers reach submission.json, not only the fallback)",
          lambda: len(full_ok) >= 1, REC)
    check("rerun: TASK_IDS == every test task; no eval challenges / solutions loaded (NB5_EVAL_CH empty, SOLUTIONS None)",
          lambda: sorted(G["TASK_IDS"]) == sorted(test) and G["NB5_EVAL_CH"] == {} and G["SOLUTIONS"] is None
          and G["P"]["source"] == "test", (len(G.get("TASK_IDS") or []), len(G.get("NB5_EVAL_CH") or {})))
    check(f"rerun: the fake deadline is in force (TOTAL_BUDGET_SEC == {BUDGET})", lambda: G["TOTAL_BUDGET_SEC"] == BUDGET,
          G.get("TOTAL_BUDGET_SEC"))
else:
    check("commit: submission.json = CELL 0's identity fallback of the test file (the eval subset never writes it)",
          lambda: sub == {t: ident(t) for t in test}, "")
    sc = main.get("score") or {}
    check("commit: the eval subset (NB6COMMIT task_list) scored by exact match, >= 1 test query hit (the fake's lookup)",
          lambda: P["source"] == "eval" and sorted(G["TASK_IDS"]) == sorted(P["task_list"]) and sc["eval_query_total"] >= len(P["task_list"])
          and sc["eval_query_hits"] >= 1 and set(main["task_solve"]) == set(P["task_list"]),
          json.dumps({"score": sc, "task_solve": main.get("task_solve"), "task_list": P.get("task_list")}, default=str)[:900])
    check("commit: the baked 40-min budget (no override)", lambda: G["TOTAL_BUDGET_SEC"] == 2400, G.get("TOTAL_BUDGET_SEC"))
_hard_wall = time.time() - (time.perf_counter() - G["HARD_STOP"]) if "HARD_STOP" in G else None
REC.update(finish_sec=round(T_END - G.get("NOTEBOOK_T0", t0)), budget_sec=G.get("TOTAL_BUDGET_SEC"),
           hard_stop_sec=round(G["HARD_STOP"] - G["NOTEBOOK_T0"]) if "HARD_STOP" in G else None,
           sub_written_before_hard_stop_by_sec=round(_hard_wall - SUBP.stat().st_mtime) if _hard_wall and SUBP.is_file() else None,
           e15b={k: (G.get("E15B") or {}).get(k) for k in ("E12_B_SKIP_B1_IF_ACCEPTED", "E12_B_B1_ACC_MODE", "E12_B_EXTRA_ATTEMPTS",
                                                           "E12_B_ATTEMPT_TOKENS", "E12_B_ATTEMPT_TOKENS_B1")},
           cell_sec=cell_sec, fake_set=SET)
check("the notebook finished before its deadline (NOTEBOOK_T0 + TOTAL_BUDGET_SEC) and submission.json was last written "
      "before the hard stop", lambda: T_END < G["NOTEBOOK_T0"] + G["TOTAL_BUDGET_SEC"] and SUBP.stat().st_mtime <= _hard_wall, REC)
check("bundle zip written", (W / "Q27X_NB5_BUNDLE.zip").exists())
HB = json.loads((W / "nb6_heartbeat.json").read_text(encoding="utf-8")) if (W / "nb6_heartbeat.json").is_file() else {}
REC["heartbeat"] = HB
_hbr = HB.get("boot_retry")
HAS_HB = "nb6_heartbeat" in "".join(code)
REC["heartbeat_in_build"] = HAS_HB
if HAS_HB:
  check("NB6 heartbeat file exists: last stage 'final', engine_ready True, the profile, every test input has a submission row, "
      "sandbox verdict PASS*, " + ("the boot retry recorded (first status, GPUs released, not skipped)" if RETRY else
      "no boot retry (the first try booted)") + (", >= 1 task non-identity" if RR else ""),
      lambda: HB["stage"] == "final" and HB["engine_ready"] is True and HB["profile"] == want_profile and HB["rerun"] is (RR)
      and HB["submission_rows"] == sum(len(v["test"]) for v in test.values()) and str((HB.get("sandbox") or {}).get("verdict")).startswith("PASS")
      and ((isinstance(_hbr, dict) and _hbr.get("first_status") and not _hbr.get("skipped")
            and (_hbr.get("release") or {}).get("released") is not False) if RETRY else _hbr is None)
      and (not RR or HB["submission_tasks_non_identity"] >= 1), HB)
if RR and "NB6 P6" in "".join(code):
    check("rerun: no sha gate (NB6 P6 = E18 R2 / E15B SUBMIT), recorded as such",
          lambda: str(((G.get("R") or {}).get("setup") or {}).get("sha_gate", {}).get("why", "")).startswith("rerun: no sha gate"),
          ((G.get("R") or {}).get("setup") or {}).get("sha_gate"))
if RETRY:
    _inp = (G.get("R") or {}).get("inputs") or {}
    check("F1: no evaluation file mounted -> the rerun built the probe prompts from the test file (R inputs arc = the test file, "
          "prompt render recorded with the hashes ok, no runner error)",
          lambda: NO_EVAL and Path(_inp["arc"]).resolve() == (COMP / "arc-agi_test_challenges.json").resolve()
          and rep["prompt_render"] and rep["prompt_render"].get("hashes_ok") is True and not rep.get("errors"),
          json.dumps({"no_eval": NO_EVAL, "arc": _inp.get("arc"), "prompt_render": rep.get("prompt_render")}, default=str)[:800])


# ---- E18 R2 (LB 31.53, the proven eval / private-run rules): its CELL 0 and its submission writer on the SAME fake test set
def schema_sig(s):
    """per task: per entry (keys, per key: container type, row type, value types) -- the shape of the format, not the values"""
    if not isinstance(s, dict):
        return type(s).__name__
    return {t: [(sorted(a), [(type(a[k]).__name__, sorted({type(r).__name__ for r in a[k]}) if isinstance(a[k], list) else None,
                              sorted({type(v).__name__ for r in a[k] if isinstance(r, list) for v in r}) if isinstance(a[k], list) else None)
                             for k in sorted(a)]) if isinstance(a, dict) else type(a).__name__ for a in rows]
            if isinstance(rows, list) else type(rows).__name__ for t, rows in s.items()}


def run_cell0(src, label):
    w = WORK / f"{label}_working"
    w.mkdir()
    old = os.environ.get("Q27X_WORKING")
    os.environ["Q27X_WORKING"] = str(w)
    g = {"__name__": "__main__"}
    n0 = len(OPENS)
    _REC[0] = True
    try:
        exec(compile(src, f"<{label}>", "exec"), g)
    finally:
        _REC[0] = False
        os.environ["Q27X_WORKING"] = old
    g["_opens"] = OPENS[n0:]
    g["_fallback"] = json.loads((w / "submission.json").read_text(encoding="utf-8"))
    return g


E18, e18_sub = {}, None
blk = lambda s: s[s.index("ARC_ROOT = "):s.index("SUBMISSION_PATH")]
try:
    E18_C0 = open(HERE / "e18_cell0.py", encoding="utf-8").read()
    GE = run_cell0(E18_C0, "e18_cell0")
    GN = run_cell0(code[0], "nb6_cell0")
    exec(compile(open(HERE / "e18_writer.py", encoding="utf-8").read(), "<e18 e10_write_submission>", "exec"), GE)
    GE["E10_FINAL"] = {t: rows for t, rows in (sub or {}).items()}       # NB6's final answers through E18's writer
    GE["e10_write_submission"]()
    e18_sub = json.loads(Path(GE["SUBMISSION_PATH"]).read_text(encoding="utf-8"))
    E18.update(arc_root={"e18": str(GE["ARC_ROOT"]), "nb6_cell0": str(GN["ARC_ROOT"]), "nb6_run": str(G.get("ARC_ROOT"))},
               rerun={"e18": GE["RERUN"], "nb6": GN["RERUN"]}, opens={"e18": GE["_opens"], "nb6_cell0": GN["_opens"]},
               path_block_equal=blk(E18_C0) == blk(code[0]), cell0_text_equal=E18_C0 == code[0])
except BaseException as e:
    import traceback
    E18.update(error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
REC["e18"] = {k: v for k, v in E18.items() if k != "tb"}
E18_TEST = (Path(E18.get("arc_root", {}).get("e18", "/none")) / "arc-agi_test_challenges.json").resolve()
check("E18 R2 CELL 0 vs NB6 CELL 0: the same ARC_ROOT / test-file resolution code, the same resolved test file, the same mode",
      lambda: E18["path_block_equal"] and len(set(E18["arc_root"].values())) == 1 and E18["rerun"]["e18"] is E18["rerun"]["nb6"] is (RR)
      and [Path(p).resolve() for p in E18["opens"]["e18"]] == [Path(p).resolve() for p in E18["opens"]["nb6_cell0"]] == [E18_TEST], E18)
check("E18 R2 CELL 0 and NB6 CELL 0 write the same fallback submission.json on the same test set (the identity, every input)",
      lambda: GE["_fallback"] == GN["_fallback"] == {t: ident(t) for t in test}, "")
check("NB6 final submission vs E18 R2's writer fed NB6's answers: the same task ids, entries per task, attempt keys and grid "
      "types (schema), and NB6 covers every test task",
      lambda: isinstance(e18_sub, dict) and set(sub) == set(e18_sub) == set(test)
      and all(len(sub[t]) == len(e18_sub[t]) == len(test[t]["test"]) for t in test) and schema_sig(sub) == schema_sig(e18_sub),
      {"nb6": schema_sig(sub) if sub is not None else sub_err, "e18": schema_sig(e18_sub)})
check("E18 R2's writer reproduces NB6's submission.json exactly (same overlay rule: identity, then rows of matching length "
      f"{'in the rerun' if RR else '-- commit: the identity only'})", lambda: e18_sub == sub, "")
TEST_READS = [p for p in OPENS if p.endswith("arc-agi_test_challenges.json")]
SOL_READS = [p for p in OPENS if "evaluation_solutions" in p]
REC["opens"] = {"test_challenges": sorted(set(TEST_READS)), "evaluation_solutions": sorted(set(SOL_READS)),
                "other": sorted({p for p in OPENS if p not in TEST_READS and p not in SOL_READS})[:20]}
check("NB6 run read arc-agi_test_challenges.json, only from E18 R2's resolved path",
      lambda: TEST_READS and all(Path(p).resolve() == E18_TEST for p in TEST_READS), REC["opens"])
if RR:
    check("rerun: NB6 never opened arc-agi_evaluation_solutions.json (audit hook over every cell)", not SOL_READS, REC["opens"])
    check("rerun: the final submission.json is NOT the bare identity (the fake solver produced answers)",
          lambda: REC["solved_pass2"] and sub != {t: ident(t) for t in test}, REC.get("solved_pass2"))
bad = [c for c in checks if not c[1]]
for label, ok, info in checks:
    print(f"[nb6 e2e {'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f" -> {info}"), flush=True)
print("NB6_E2E_RECORD " + json.dumps(REC, default=str), flush=True)
print(f"{len(checks) - len(bad)}/{len(checks)} PASS | work {WORK}", flush=True)
print("@@NB6_E2E_OK@@" if not bad else "@@NB6_E2E_FAIL@@", flush=True)
sys.exit(0 if not bad else 1)
'''

E2E = r'''
PAY = find_payload(); REP["checks"]["payload"] = PAY; save()
log("payload", json.dumps(PAY)[:1200])
B = K / "build"; B.mkdir()
for n in ("nb6.ipynb", "nb6_e2e.py", "nb5_fake_server.py", "e18_cell0.py", "e18_writer.py"):
    (B / n).write_text(FILES[n], encoding="utf-8", newline="\n")
(K / "parts.json").write_text(FILES["parts.json"]); (K / "make_fake.py").write_text(FILES["make_fake.py"])
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5", "NB6", "E12_", "E15B_", "Q27X_", "RC2_", "E23_", "FAKE_"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
U, E = {}, {}
U["comp_files"] = {p.name: [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()[:16]] for p in sorted(COMPD.glob("*.json"))}
U["comp_has_eval_solutions"] = (COMPD / "arc-agi_evaluation_solutions.json").is_file()
U["fake_tree"] = sh("fake_tree", [sys.executable, str(K / "make_fake.py"), INPUT_ROOT, str(K / "fin"), str(K / "parts.json"), str(COMPD)])
U["fake_tree"].pop("_out")
log("fake tree", json.dumps(U["fake_tree"])[:600])
EALL = {}
for mode in "__MODE__".split(","):          # one or more e2e modes, one after the other (each its own fake tree copy)
    try:
        r = sh("e2e_" + mode, [sys.executable, str(B / "nb6_e2e.py"), str(B / "nb6.ipynb"), mode],
               {"NB6_E2E_INPUT": str(K / "fin"), "NB6_E2E_DATA": str(COMPD), "NB6_E2E_PAYSRC": str(Path(PAY["used"]).parent),
                "NB6_E2E_ROOT": str(K / "e2e"), "NB6_E2E_BUDGET": "1800", "NB6_E2E_BUDGET_RETRY": "3000", "NB6_E2E_MAX_SEC": "3500",
                "PYTHONIOENCODING": "utf-8"}, 3700, cwd=B)
        lines = r["_out"].splitlines()
        E = {"rc": r["rc"], "sec": r["sec"], "ok": r["rc"] == 0 and "@@NB6_E2E_OK@@" in r["_out"],
             "summary": next((l for l in lines if re.match(r"\d+/\d+ PASS", l)), None),
             "fails": [l[:600] for l in lines if "e2e FAIL]" in l][:30],
             "passes": [l[:200] for l in lines if "e2e PASS]" in l],
             "record": next((json.loads(l[15:]) for l in lines if l.startswith("NB6_E2E_RECORD ")), None),
             "fake_set": next((l[31:] for l in lines if l.startswith("[nb6 e2e] fake hidden test set ")), None), "tail": r["tail"][-1500:]}
    except BaseException as e:
        E = {"ok": False, "fatal": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-2000:]}
    EALL[mode] = E
    log("e2e", mode, json.dumps({k: v for k, v in E.items() if k not in ("tail", "passes")}, default=str)[:3000])
    REP["checks"]["e2e"] = EALL; REP["checks"]["setup"] = U; save()
verdict({"payload_sha_ok_under_input": PAY["used"] is not None,
         "competition_eval_challenges_and_solutions": bool(U.get("comp_has_eval_solutions")),
         "fake_tree_from_real_part1": (U.get("fake_tree") or {}).get("rc") == 0,
         **{f"e2e_{m}": bool(E.get("ok")) for m, E in EALL.items()}},
        recorded={m: {"summary": E.get("summary"), "fails": E.get("fails"), "record": E.get("record")} for m, E in EALL.items()}
        | {"payload_used": PAY.get("used")})
'''

# ---------------------------------------------------------------------------------------------------------- real
K1_CT = between(K5["K1"], "# ---------------------------------------------------------------- commit-turn A / C",
                "# ---------------------------------------------------------------- dual mode")
ARGPARSE = between(KLEE, 'ARGPARSE = r"""', '\n"""\n') + '\n"""\n'
VENV_PY = between(KLEE, "def venv_py(", "\n\n\nIMPORT_QSA")
assert "make_arg_parser" in ARGPARSE and "AsyncEngineArgs.from_cli_args" in ARGPARSE and "subprocess.run([py" in VENV_PY

REAL = r'''
# ---------------------------------------------------------------- cells 0-4 of the NB6 notebook on the real mounts, two namespaces
PAY = find_payload(); REP["checks"]["payload"] = PAY; save()
log("payload", json.dumps(PAY)[:1200])
NB = json.loads(FILES["nb6.ipynb"])
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
HEADS = [c.splitlines()[0][:100] for c in CODE]
N_PRE = next(i for i, h in enumerate(HEADS) if "CELL 6/7" in h)       # run cells before CELL 6/7 only (5 in the NB5 layout)
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5_", "NB5H_", "NB6", "E12_", "E15B_", "Q27X_", "RC2_", "E23_"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN", "VLLM_TARGET_DEVICE"]:
    os.environ.pop(k_, None)
os.environ.update(E23_INPUT=INPUT_ROOT, E23_TMP=os.environ.get("KIT_E23_TMP", "/tmp"), Q27X_ARC_ROOT=str(COMPD),
                  RC2_CPU_KIT="1", RC2_SHM=str(K / "shm"), E12_DIR=str(K / "e12_duck"))
(K / "shm").mkdir()
if INPUT_ROOT != "/kaggle/input":
    os.environ["Q27X_PAYLOAD_DIR"] = INPUT_ROOT
def run_ns(label, rerun):
    work = K / f"working_{label}"; work.mkdir()
    os.environ["Q27X_WORKING"] = str(work)
    if rerun:
        os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    else:
        os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    g, cells = {"__name__": "__main__"}, {}
    try:
        for i in range(N_PRE):
            t = time.time()
            try:
                exec(compile(CODE[i], f"<nb6 {label} cell {i}>", "exec"), g)
                cells[i] = {"ok": True, "sec": round(time.time() - t, 1), "head": HEADS[i]}
            except BaseException as e:
                cells[i] = {"ok": False, "head": HEADS[i], "error": f"{type(e).__name__}: {e}"[:1500], "tb": traceback.format_exc()[-3000:]}
                if i == 2 and "E12A" not in g:
                    g["E12A"] = types.SimpleNamespace(ARC2_TOOL_DESCRIPTION="(kit stub: the Duck package did not load)")
                    cells[i]["stub"] = "E12A"
            log(label, "cell", i, json.dumps({k: v for k, v in cells[i].items() if k != "tb"})[:1200])
    finally:
        os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    g["_WORK"] = work
    return g, cells
ENV0 = dict(os.environ)
G, CELLS_C = run_ns("commit", False)
REP["checks"]["cells_commit"] = CELLS_C; save()
D3 = {"payload_used": G.get("Q27X_PAYLOAD_USED"), "sandbox_selftest": G.get("E10_SANDBOX_SELFTEST"),
      "manifest_files": len(G.get("_E12_MANIFEST") or {})}
REP["checks"]["duck"] = D3; save()
log("duck", json.dumps(D3, default=str)[:1500])
R1 = {}
def step(name, fn):
    t = time.time()
    try:
        v = fn()
        R1[name] = {"ok": True, "sec": round(time.time() - t, 1), **(v if isinstance(v, dict) else {"value": v})}
    except BaseException as e:
        R1[name] = {"ok": False, "sec": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"[:1500], "tb": traceback.format_exc()[-2500:]}
    log(name, json.dumps({k: v for k, v in R1[name].items() if k != "tb"}, default=str)[:1500]); REP["checks"]["R1"] = R1; save()
    return R1[name]
_ids0 = {k: id(v) for k, v in G.items()}
step("inputs", lambda: (G["nb5_rc2_inputs"](), {"inputs": G["R"]["inputs"]})[1])
step("prompts", lambda: (G["nb5_rc2_prompts"](), {"merge": G["R"]["sections"]["A0"]["merge"], "host": G["R"]["sections"]["A0"]["host"],
                                                   "prompt_render": {k: v for k, v in G["R"]["prompt_render"].items() if "text" not in k}})[1])
step("install", lambda: (lambda e: {"kv": dict(e["kv"]), "cuda_home": e["cuda_home"]})(G["merlin_install"]()))
step("patch_60490", lambda: G["nb5_apply_60490"]())
step("sha256", lambda: (lambda ok_why: {"sha_ok": ok_why[0], "why": ok_why[1], "detail": G["R"]["setup"].get("sha256")})(G["sha_gate"](3600)))
SHARED = sorted(k for k, v in G.items() if (k not in _ids0 or _ids0[k] != id(v)) and not k.startswith("__")) + ["R"]
OPENS, _REC = [], [False]          # ARC data files opened in this process while the rerun namespace runs (audit hook)
def _audit(ev, args):
    if ev == "open" and _REC[0] and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p_ = os.fsdecode(args[0])
        if "arc-agi_" in p_:
            OPENS.append(p_)
sys.addaudithook(_audit)
_REC[0] = True
try:
    GR, CELLS_R = run_ns("rerun", True)
finally:
    _REC[0] = False
RERUN_OPENS = {"test_challenges": sorted({p for p in OPENS if p.endswith("arc-agi_test_challenges.json")}),
               "evaluation_solutions": sorted({p for p in OPENS if "evaluation_solutions" in p}),
               "other": sorted({p for p in OPENS if not p.endswith("arc-agi_test_challenges.json") and "evaluation_solutions" not in p})}
REP["checks"]["rerun_opens"] = RERUN_OPENS
# E18 R2 CELL 0 (LB 31.53) vs NB6 CELL 0 on the REAL mount, Q27X_ARC_ROOT unset (the notebooks' own default + fallback path)
C0 = {}
C0_RUN = """import json, os, sys, hashlib
src = open(sys.argv[1], encoding="utf-8").read()
g = {"__name__": "__main__"}
exec(compile(src, sys.argv[1], "exec"), g)
t = g["ARC_ROOT"] / "arc-agi_test_challenges.json"
s = json.load(open(g["SUBMISSION_PATH"]))
print("C0_JSON " + json.dumps({"arc_root": str(g["ARC_ROOT"]), "rerun": g["RERUN"], "test_sha": hashlib.sha256(t.read_bytes()).hexdigest(),
      "n_tasks": len(g["test_challenges"]), "sub_sha": hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest(),
      "identity": s == {k: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in v["test"]] for k, v in g["test_challenges"].items()}}))
"""
(K / "c0_run.py").write_text(C0_RUN)
(K / "e18_cell0.py").write_text(FILES["e18_cell0.py"], encoding="utf-8"); (K / "nb6_cell0.py").write_text(CODE[0], encoding="utf-8")
for lab, f in (("e18", "e18_cell0.py"), ("nb6", "nb6_cell0.py")):
    for rr in ("1", ""):
        env = {k: v for k, v in ENV0.items() if k not in ("Q27X_ARC_ROOT", "KAGGLE_IS_COMPETITION_RERUN")}
        env.update(Q27X_WORKING=str(K / f"c0_{lab}_{rr or 'commit'}"), PYTHONIOENCODING="utf-8")
        if rr:
            env["KAGGLE_IS_COMPETITION_RERUN"] = rr
        try:
            r = subprocess.run([sys.executable, str(K / "c0_run.py"), str(K / f)], capture_output=True, text=True, timeout=600, env=env)
            C0[f"{lab}_{'rerun' if rr else 'commit'}"] = next((json.loads(l[8:]) for l in r.stdout.splitlines() if l.startswith("C0_JSON ")),
                                                              {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]})
        except BaseException as e:
            C0[f"{lab}_{'rerun' if rr else 'commit'}"] = {"error": f"{type(e).__name__}: {e}"[:400]}
REP["checks"]["cell0_vs_e18"] = C0; save()
log("cell 0 vs E18 R2", json.dumps(C0)[:1500])
REP["checks"]["cells_rerun"] = CELLS_R
for k in SHARED:                       # the rerun namespace reuses the commit namespace's inputs / merge / install state
    GR[k] = G[k]
REP["checks"]["shared_into_rerun"] = SHARED; save()
log("shared into the rerun namespace", SHARED)

# ---------------------------------------------------------------- every boot try of both profiles' plans (no server started)
eng = G.get("MERLIN_ENG") or {}
os.environ["VLLM_TARGET_DEVICE"] = "cpu"     # kit only: help_check inherits it via the server env (klee100 kit v2 ALL OK)
CPU_ENV = {"VLLM_TARGET_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": ""}
''' + VENV_PY + r'''
''' + ARGPARSE + r'''
def plans_for(g, label):
    out, P_, L5 = {}, g["P"], g["NB5L"]
    groups = []
    for sp in P_["nb5_phases"]:
        sp = list(sp) + [None] * (7 - len(sp))
        key = (sp[3], sp[5] or g["NB5_KV_DTYPE"])
        if not groups or groups[-1][0] != key:
            groups.append([key, []])
        groups[-1][1].append(sp[0])
    rc2_fp8_dtype = L5.argv_last(g["NB5_RC2_FP8_ARGV_TAIL"]).get("--kv-cache-dtype")
    for (mtp, kv), phs in groups:
        base_extra = list(g["NB5_EXTRA_ARGS"]) + ([] if mtp == "on" else list(g["NB5_MTP_OFF_ARGS"]))
        tries = L5.pin_tries(g["NB5_KV_BYTES"][kv], None) if P_.get("pin_ladder") else [{"gib": 0.0, "kv_bytes": g["NB5_KV_BYTES"][kv]}]
        lev0 = dict(g["NB5_LEVER_ENV"])
        if P_.get("e18_ladder"):             # NB6 P3: E18 R2's engine ladder = at most 3 attempts
            tries = tries[:3]
        if P_.get("boot_retry_once"):        # NB6 E8 / E9: ONE try at the top pin with the lever env, then ONE retry at the RC2 pin
            tries = [dict(tries[0], levers=lev0)] + [dict(t_, levers={}) for t_ in L5.pin_tries(g["NB5_KV_BYTES"][kv], 0.0)]
        else:
            tries = [dict(t_, levers=lev0) for t_ in tries]
        for i, tr in enumerate(tries):
            levers = tr["levers"]
            name = f"kit_{label}_{kv}_{'mtp' if mtp == 'on' else 'nomtp'}_pin{int(round(tr['gib'] * 100))}{'' if levers or not lev0 else '_nolever'}"
            g["NB5_BOOT_KV"], g["NB5_BOOT_LEVERS"] = kv, dict(levers)
            seqs = L5.seqs_for(kv, tr["gib"]) if g["NB5_SEQS_RULE"] == "table" else None
            g["NB5_BOOT_EXTRA"] = base_extra + (["--max-num-seqs", str(seqs)] if seqs else [])
            b = {"phases": phs, "kv": kv, "mtp": mtp, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"], "seqs": seqs, "levers": levers}
            try:
                s = g["merlin_fn_server"](name, kv, 8100 + 10 * i + (0 if label == "commit" else 5), tr["kv_bytes"])
                al = L5.argv_last(s.argv)
                caps = json.loads(al.get("--compilation-config") or "{}").get("cudagraph_capture_sizes")
                chk = []
                if str(al.get("--kv-cache-memory-bytes")) != str(tr["kv_bytes"]): chk.append(f"kv bytes {al.get('--kv-cache-memory-bytes')}")
                if seqs is not None and str(al.get("--max-num-seqs")) != str(seqs): chk.append(f"seqs {al.get('--max-num-seqs')}")
                if P_.get("caps16") and (caps != g["NB5_CAPS16"] or not L5.captures_cover(caps, seqs or 1)): chk.append(f"captures {caps}")
                if any(s.env.get(k) != v for k, v in levers.items()): chk.append(f"lever env {[s.env.get(k) for k in levers]}")
                if not levers and any(s.env.get(k) not in (None, "") for k in lev0): chk.append(f"retry carries the lever env {[s.env.get(k) for k in lev0]}")
                spec = al.get("--speculative-config")
                if (mtp == "on") != (spec not in (None, "None")): chk.append(f"spec {spec}")
                if kv == "fp8" and al.get("--kv-cache-dtype") != rc2_fp8_dtype: chk.append(f"kv dtype {al.get('--kv-cache-dtype')} vs RC2 {rc2_fp8_dtype}")
                if str(al.get("--max-model-len")) != "131072": chk.append(f"max-model-len {al.get('--max-model-len')}")
                b.update(ok=True, host_phase=g["HOST_PHASE"].get(name), help=g["HELP"].get(name), argv=s.argv, server_started=s.proc is not None,
                         argv_flags={f: al.get(f) for f in ("--kv-cache-dtype", "--kv-cache-memory-bytes", "--max-num-seqs",
                                                            "--speculative-config", "--max-model-len")}, captures=caps,
                         env_levers={k: s.env.get(k) for k in levers}, env_vllm_target=s.env.get("VLLM_TARGET_DEVICE"), argv_check=chk)
            except BaseException as e:
                b.update(ok=False, error=f"{type(e).__name__}: {e}"[:1500], host_phase=(g.get("HOST_PHASE") or {}).get(name),
                         help=(g.get("HELP") or {}).get(name), tb=traceback.format_exc()[-2500:])
            out[name] = b
            log("plan", name, json.dumps({k: v for k, v in b.items() if k not in ("argv", "tb")}, default=str)[:1500])
    g["NB5_BOOT_EXTRA"], g["NB5_BOOT_LEVERS"] = [], {}
    return out
PLANS = {}
for label, g in (("commit", G), ("rerun", GR)):
    try:
        PLANS[label] = plans_for(g, label)
    except BaseException as e:
        PLANS[label] = {"_fatal": {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500], "tb": traceback.format_exc()[-2500:]}}
    REP["checks"]["plans"] = PLANS; save()
try:
    R1["help_text_head"] = (G["L"]._HELP.get("merlin") or "")[:2500]
    R1["help_text_len"] = len(G["L"]._HELP.get("merlin") or "")
except Exception as e:
    R1["help_text_head"] = repr(e)
ALLP = [b for pl in PLANS.values() for b in pl.values()]
def _strip(argv):
    a = list(argv or [])
    for f in ("--port",):
        if f in a:
            i = a.index(f); a[i + 1] = "*"
    return a
PL_EQ = {"commit": [(b["kv"], b["mtp"], b["pin_gib"], _strip(b.get("argv"))) for b in PLANS.get("commit", {}).values()],
         "rerun": [(b["kv"], b["mtp"], b["pin_gib"], _strip(b.get("argv"))) for b in PLANS.get("rerun", {}).values()]}
R6 = {}
try:
    argvs = {f"{lab}:{n}": b["argv"] for lab, pl in PLANS.items() for n, b in pl.items() if b.get("argv")}
    (K / "argvs.json").write_text(json.dumps(argvs))
    r = venv_py(ARGPARSE, [str(K / "argvs.json")], name="argparse", extra_env=CPU_ENV)
    R6["parsed"] = r["json"] or {"error": r["tail"]}
    fields = {}
    for lab_, argv in argvs.items():
        lab, n = lab_.split(":", 1)
        b = PLANS[lab][n]
        x = (R6["parsed"] or {}).get(lab_) or {}
        bad = []
        if x.get("ok") is not True: bad.append("parse " + str(x.get("error"))[:160])
        if x.get("unknown"): bad.append(f"unknown flags {x.get('unknown')}")
        if b["seqs"] is not None and x.get("max_num_seqs") != b["seqs"]: bad.append(f"max_num_seqs {x.get('max_num_seqs')}")
        if x.get("kv_cache_memory_bytes") != b["kv_bytes"]: bad.append(f"kv {x.get('kv_cache_memory_bytes')}")
        if x.get("kv_cache_dtype") != (b.get("argv_flags") or {}).get("--kv-cache-dtype"): bad.append(f"kv_cache_dtype {x.get('kv_cache_dtype')}")
        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")
        if x.get("enable_prefix_caching") is not True: bad.append(f"prefix caching {x.get('enable_prefix_caching')}")
        if x.get("model") != str(G["CKPT"]): bad.append(f"model {x.get('model')}")
        if (x.get("tensor_parallel_size"), x.get("pipeline_parallel_size"), x.get("enable_expert_parallel")) != (2, 2, True): bad.append("layout")
        if b.get("captures") and (x.get("capture_sizes") or []) != b["captures"]: bad.append(f"captures {x.get('capture_sizes')}")
        spec = x.get("speculative_config")
        if b["mtp"] == "on":
            okm = (spec.get("method") == "mtp" and spec.get("num_speculative_tokens") == 3) if isinstance(spec, dict) else \
                  ("'num_speculative_tokens': 3" in str(spec) and "mtp" in str(spec))
            if not okm: bad.append(f"spec {str(spec)[:80]}")
        elif spec not in (None, "None"): bad.append(f"spec should be None: {str(spec)[:80]}")
        fields[lab_] = bad
    R6["field_check"] = fields
    R6["ok"] = bool(fields) and len(fields) == len(ALLP) and all(not v for v in fields.values())
except BaseException as e:
    R6.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
os.environ.pop("VLLM_TARGET_DEVICE", None)
log("argparse", json.dumps({k: (R6.get("parsed") or {}).get(k) for k in ("_env", "_platform", "import_error")}), json.dumps(R6.get("field_check"))[:2000])
REP["checks"]["argparse"] = R6; save()

NB5L = G.get("NB5L")             # (the NB5 kit's commit-turn section below uses G / NB5L)
''' + K1_CT + r'''
# ---------------------------------------------------------------- dual mode: CELL 0's fallback on the real test file, both namespaces
SUB = {}
try:
    test = json.load(open(COMPD / "arc-agi_test_challenges.json"))
    for lab, g in (("commit", G), ("rerun", GR)):
        sp = g["_WORK"] / "submission.json"
        s_ = json.loads(sp.read_text(encoding="utf-8"))
        bad = strict_schema(s_, test)
        SUB[lab] = {"tasks": len(s_), "test_tasks": len(test), "schema_bad": bad[:20], "strict_full_schema": not bad,
                    "identity": s_ == {t: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in v["test"]] for t, v in test.items()},
                    "profile": g.get("PROFILE"), "rerun": g.get("RERUN"), "source": (g.get("P") or {}).get("source"),
                    "phases": (g.get("P") or {}).get("nb5_phases"), "task_ids_eq_test": sorted(g.get("TASK_IDS") or []) == sorted(test),
                    "budget_sec": g.get("TOTAL_BUDGET_SEC")}
    SUB["comp_files"] = {p.name: [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()[:16]] for p in sorted(COMPD.glob("*.json"))}
except BaseException as e:
    SUB.update(error=f"{type(e).__name__}: {e}"[:1200], tb=traceback.format_exc()[-2000:])
REP["checks"]["submission"] = SUB; save()
log("submission", json.dumps(SUB, default=str)[:2000])

pu = (D3.get("payload_used") or {}).get(PAYLOAD_NAME) or {}
PC, PR = PLANS.get("commit") or {}, PLANS.get("rerun") or {}
sc, sr = SUB.get("commit") or {}, SUB.get("rerun") or {}
verdict({
    "payload_sha_ok_under_input": PAY["used"] is not None,
    "duck_cell_loaded": bool(CELLS_C.get(2, {}).get("ok")) and D3["manifest_files"] > 0,
    "notebook_used_that_payload_path": pu.get("path") is not None and Path(pu["path"]).resolve() == Path(PAY["used"] or "/none").resolve(),
    "sandbox_selftest_ok_incl_memory_limit": bool((D3.get("sandbox_selftest") or {}).get("ok")),
    "cells_before_6_7_ran_commit_and_rerun": N_PRE == 5 and all(CELLS_C.get(i, {}).get("ok") for i in range(N_PRE))
        and all(CELLS_R.get(i, {}).get("ok") for i in range(N_PRE)),
    "profile_commit_NB6COMMIT": G.get("PROFILE") == "NB6COMMIT" and G.get("RERUN") is False,
    "profile_rerun_RERUN_PROD_source_test_all_tasks": GR.get("PROFILE") == "RERUN_PROD" and GR.get("RERUN") is True
        and sr.get("source") == "test" and sr.get("task_ids_eq_test") is True,
    "inputs_found": bool(((R1.get("inputs") or {}).get("inputs") or {}).get("merlin_inputs_ok")),
    "merge_ok": bool(((G.get("R") or {}).get("sections", {}).get("A0", {}).get("merge") or {}).get("ok")),
    "prompt_hashes_ok": bool(((R1.get("prompts") or {}).get("prompt_render") or {}).get("hashes_ok")),
    "sha256_all_ok": bool((R1.get("sha256") or {}).get("sha_ok")),
    "merlin_install_ok": bool((R1.get("install") or {}).get("ok")),
    "patch_60490_applied_verified": bool((R1.get("patch_60490") or {}).get("ok")) and (R1.get("patch_60490") or {}).get("sha_after")
        == "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b",
    "plans_commit_every_pin_step": len(PC) >= 1 and all(b.get("ok") for b in PC.values()),
    "plans_rerun_every_pin_step": len(PR) >= 1 and all(b.get("ok") for b in PR.values()),
    "host_phase_ok_all_plans": bool(ALLP) and all(b.get("host_phase") == "ok" for b in ALLP),
    "help_cpu_every_flag_present_all_plans": bool(ALLP) and all((b.get("help") or {}).get("ran") and not (b.get("help") or {}).get("missing")
                                                                for b in ALLP),
    "argv_pin_seqs_captures_lever_spec_dtype_all_plans": bool(ALLP) and all(b.get("argv") and not b.get("argv_check") for b in ALLP),
    "rerun_plans_equal_commit_plans": bool(PL_EQ["commit"]) and PL_EQ["commit"] == PL_EQ["rerun"],
    "vllm_argparse_fields_all_plans": bool(R6.get("ok")),
    "no_server_started": bool(ALLP) and not any(b.get("server_started") for b in ALLP),
    "commit_turn_A_C_reproduced_real_ckpt": CTR.get("reproduced") is True and CTR.get("c_ids_equal") is True and CTR.get("c_common") == 12248
        and CTR.get("b_common") == 3,
    "budget_guard_real_renders": bool((CTR.get("guard") or {}).get("ok")),
    "submission_commit_strict_full_schema_identity": sc.get("strict_full_schema") is True and sc.get("identity") is True
        and sc.get("profile") == "NB6COMMIT",
    "submission_rerun_strict_full_schema_identity": sr.get("strict_full_schema") is True and sr.get("identity") is True
        and sr.get("profile") == "RERUN_PROD",
    "cell0_same_test_file_and_fallback_as_e18_r2_real_mount": all(C0.get(f"{a}_{m}", {}).get("test_sha") for a in ("e18", "nb6")
        for m in ("rerun", "commit")) and all(C0[f"e18_{m}"][k] == C0[f"nb6_{m}"][k] for m in ("rerun", "commit")
        for k in ("arc_root", "rerun", "test_sha", "n_tasks", "sub_sha", "identity")) and C0["nb6_rerun"]["rerun"] is True
        and C0["nb6_commit"]["rerun"] is False and C0["nb6_rerun"]["identity"] is True,
    "rerun_cells_read_test_file_only_at_COMPD_never_eval_solutions": bool(RERUN_OPENS["test_challenges"])
        and all(Path(p).resolve() == (COMPD / "arc-agi_test_challenges.json").resolve() for p in RERUN_OPENS["test_challenges"])
        and not RERUN_OPENS["evaluation_solutions"],
}, recorded={"payload_used": PAY.get("used"), "plans": {lab: {n: {k: b.get(k) for k in ("pin_gib", "kv_bytes", "seqs", "argv_flags", "captures",
             "env_levers", "host_phase", "argv_check", "error")} for n, b in pl.items()} for lab, pl in PLANS.items()},
             "help_text_len": R1.get("help_text_len"), "argparse_env": (R6.get("parsed") or {}).get("_env"),
             "argparse_platform": (R6.get("parsed") or {}).get("_platform"), "budget_sec": {"commit": sc.get("budget_sec"), "rerun": sr.get("budget_sec")}})
'''

# ---------------------------------------------------------------------------------------------------------- kit table
UF = K5["UNIT_FILES"]
# nb6_unit_tests section 7 reads the NB6 + NB5 QUALITY push folders' kernel-metadata.json (NB6T_PUSH / NB6T_PUSH5): shipped
PUSH6 = ROOT / "q27-xhigh-e25-nb6-submit__foysalemonshanto"
PUSH6_META = (PUSH6 / "kernel-metadata.json").read_text(encoding="utf-8")
PUSH6_OK = (PUSH6 / json.loads(PUSH6_META)["code_file"]).read_bytes() == NB_RAW
assert PUSH6_OK, f"the NB6 push folder's notebook != {NB_PATH.name} (the units kit would FAIL its metadata check)"
PUSH_FILES = {"push6/kernel-metadata.json": PUSH6_META, "push6/code_file": json.loads(PUSH6_META)["code_file"],
              "push5/kernel-metadata.json": (ROOT / "q27-xhigh-e25-nb5-quality__foysalemonshanto" / "kernel-metadata.json").read_text(encoding="utf-8")}
UNIT_FILES = {"nb6.ipynb": NB_TXT, "nb5q.ipynb": NB5Q_TXT, "nb6_unit_tests.py": K5["src"]("nb6_unit_tests.py"), **PUSH_FILES,
              **{k: v for k, v in UF.items() if k not in ("nb5.ipynb", "nb5_e2e.py", "nb5_fake_server.py", "make_fake.py", "parts.json")}}
# (the units kit carries only what nb5_unit_tests.py / nb6_unit_tests.py read: < 400 kB)
assert UF["nb5.ipynb"] == NB5Q_TXT, "the NB5 kit's embedded notebook is NB5 QUALITY 9c71ba89"
E2E_FILES = {"nb6.ipynb": NB_TXT, "parts.json": K5["PARTS"], "make_fake.py": K5["MAKE_FAKE"], "nb6_e2e.py": NB6_E2E,
             "nb5_fake_server.py": K5["src"]("nb5_fake_server.py"), **E18_FILES}
REAL_FILES = {"nb6.ipynb": NB_TXT, "e18_cell0.py": E18_FILES["e18_cell0.py"], "commit_turn_prefix_test.py": K5["src"]("commit_turn_prefix_test.py"), "ct_result.json": K5["CT_RESULT"]}
KITS = {
    "units": {"body": UNITS, "title": "Q27 E25 NB6 Kit Units", "models": [RC2_MODELS[0]], "files": UNIT_FILES},
    "e2e-rerun": {"body": E2E.replace("__MODE__", "rerun,rerun-retry"), "title": "Q27 E25 NB6 Kit E2E Rerun", "models": [RC2_MODELS[0]], "files": E2E_FILES},
    "e2e-commit": {"body": E2E.replace("__MODE__", "commit"), "title": "Q27 E25 NB6 Kit E2E Commit", "models": [RC2_MODELS[0]], "files": E2E_FILES},
    "real": {"body": REAL, "title": "Q27 E25 NB6 Kit Real", "models": QMETA["model_sources"], "files": REAL_FILES},
}
compile(NB6_E2E, "<nb6_e2e>", "exec")


def build(name):
    spec = KITS[name]
    files = spec["files"]
    assert "\r" not in "".join(v for k, v in files.items() if not k.startswith("rc2/")), name
    blob = base64.b64encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    shas = {n: hashlib.sha256(v.encode("utf-8")).hexdigest() for n, v in files.items()}
    code = HEAD + spec["body"]
    for k, v in (("__SHAS__", json.dumps(shas)), ("__NBSHA__", NB_SHA), ("__NB5QSHA__", NB5Q_SHA), ("__NAME__", name),
                 ("__UEXP_RUN__", str(K5["UNITS_EXPECT"]["run"])), ("__UEXP_SKIP__", str(K5["UNITS_EXPECT"]["skipped"])),
                 ("__BLOB__", blob)):
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"{SLUG}-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__foysalemonshanto"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"foysalemonshanto/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": list(QMETA["dataset_sources"]), "kernel_sources": [],
            "competition_sources": list(QMETA["competition_sources"]), "model_sources": spec["models"], "docker_image": IMAGE,
            "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out.name, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), flush=True)
    return out


if __name__ == "__main__":
    print("notebook under test", NB_PATH.name, "sha256", NB_SHA, "| export", EXPORT_PATH.name, "equal" if EXPORT_OK else EXPORT_OK, flush=True)
    for n in KITS:
        build(n)
