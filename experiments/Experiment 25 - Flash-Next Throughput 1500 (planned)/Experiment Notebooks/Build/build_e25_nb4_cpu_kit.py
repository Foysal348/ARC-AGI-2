"""E25 NB4 TICK: the two Kaggle CPU kits (pinned Py3.12 image, no GPU, no internet, private, account foysalemonshanto).
Pattern: build_e25_klee100_load_cpu_kit.py (ran ALL OK). Each kernel is fail-soft, writes its JSON report and prints exactly
one line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.

  real   q27-e25-nb4-kit-real (the 7 NB4 Models, the payload dataset foysalemonshanto/e25-nb4-payload, the competition):
         the notebook's own code (RC2_UNIT + RC2_CPU_KIT: everything but the boots) on the REAL mounts:
           PAY  every payload file on the mount vs the notebook pins (build facts payload_files = NB4_PIN.hooks + the P5
                module's P5_SRC_SHA), SHA256SUMS agrees, unpinned extras listed; after load the notebook resolves the P5
                module from the mount and its p5_src the way p5_prepare does
           R1   inputs, merge, prompt render hashes, embedded sha, hook modules (timeline + Q-pack embedded, P5 from the
                payload), arms valid at load (validate_arms + never_do), the oracle prompt identity (recorded)
           R3   the real Merlin offline install; R4 #60490 on the installed tree (+ an import probe, NB1 kit rule)
           CT   core_timer_install() (entry point resolves in the Merlin venv) + a recorded EngineCore wrap probe
           PLANS  in the notebook's arm order T0 -> S -> S2 -> S3 -> G -> P -> T0b (+ optional K4 / INT8), as nb4_arm_step
                does but without the boot: the patch arm's prepare (S2 / S3: the notebook's ppack_prepare() on the INSTALLED
                tree; P: the source check, a recorded CPU compile probe of the .so, the P5 gate = SKIPPED_GPU_ONLY,
                apply_p5_patch()), make_arms() + validate_arms(), nb4_plan(k, 0), merlin_fn_server() = launch.build() +
                host_phase() + `vllm serve --help=all` cli flags (VLLM_TARGET_DEVICE=cpu, kit only), the notebook's own
                nb4_facts() on a server stub (its patch-state read-back keys + one_change_vs_ref vs T0), then the revert
                (ppack_finish / p5_finish) with every file sha back to its pre-patch value
           ENV  each plan's EXACT server env -> a Merlin-venv child process -> the worker extension's nb4_env(): the
                arm's env reaches the worker process (S: CUDA_DEVICE_MAX_CONNECTIONS=32, S2 NB4_PPACK, S3 NB4_PPDEFER, P NB4_P5*)
           R6   vLLM's own argparse (make_arg_parser + parse_known_args + AsyncEngineArgs.from_cli_args) on every argv:
                no unknown flag, seqs 9, pin, FP8 KV, captures (CAPS16 / G / K4 lists exactly), MTP K, draft head,
                profiler, --cudagraph-metrics, iteration details on T0 only, layout, prefix caching, extension
           DIFF every plan vs T0b (the pure base) over the FULL argv and env: differs by exactly the planned change
                (literal expectations below, checked against nb4_lib at build time); no new duplicate flags
           NEVER nb4_lib.NEVER_DO on every final argv + env (not only the arm's change); never_do(arm) empty
           R7   host gates on the real cgroup (recorded), the installed tree back to the notebook's state, the
                checkpoint sha256 gate (last: it hashes in the background from load)
  units  q27-e25-nb4-kit-units (the payload dataset only): nb4_unit_tests.py (39), nb4_timeline_tests.py (the real-trace
         class skips: the B1 traces are not on Kaggle), nb4_ppack_tests.py (copies of the real Merlin ad42e37 pp_utils.py /
         model_runner.py, sha-checked) and nb4_p5_tests.py (nb4_p5_engine.py from the payload mount, sha-checked; the real
         cuda_communicator.py) on Py3.12, in the build-folder layout (the notebook next to Build/, the NB4 push
         kernel-metadata at the path the unit test names).
Rules: no Kaggle API calls here; the lead pushes. Run: python build_e25_nb4_cpu_kit.py [--selfcheck]
  -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb4-kit-{real,units}__foysalemonshanto/
  --selfcheck: the units kit code on this machine (KIT_* paths in the scratchpad, the local payload folder as the mount)
"""
import ast
import base64
import hashlib
import json
import lzma
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_NAME = "Q27-XHIGH-E25_NB4_TICK.ipynb"
NB_PATH = HERE.parent / NB_NAME
NB_RAW = NB_PATH.read_bytes()
NB_SHA = hashlib.sha256(NB_RAW).hexdigest()
assert b"\r" not in NB_RAW, "notebook has CR bytes"
EXPORT = (HERE / "nb4_code_export.py").read_text(encoding="utf-8")
assert "\r" not in EXPORT
_nb = json.loads(NB_RAW.decode("utf-8"))
assert "\n\n".join("".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code") == EXPORT, "export != notebook cells"
FACTS = json.loads((HERE / "nb4_build_facts.json").read_text(encoding="utf-8"))
assert FACTS["notebook_sha256"] == NB_SHA, "nb4_build_facts.json is not for this notebook"
PUSH = Path(r"D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb4-tick__foysalemonshanto")
assert hashlib.sha256((PUSH / "q27-xhigh-e25-nb4-tick.ipynb").read_bytes()).hexdigest() == NB_SHA, "push copy != notebook"
NB4_META_TXT = (PUSH / "kernel-metadata.json").read_text(encoding="utf-8")
NB4_META = json.loads(NB4_META_TXT)
PAYLOAD_DIR = Path(r"D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb4_payload")
PAYLOAD_SLUG = "foysalemonshanto/e25-nb4-payload"
MERLIN_SRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODELS = list(NB4_META["model_sources"])
DATASETS = [PAYLOAD_SLUG]
assert NB4_META["docker_image"] == IMAGE and NB4_META["dataset_sources"] == DATASETS and len(MODELS) == 7
TESTS = ["nb4_unit_tests.py", "nb4_timeline_tests.py", "nb4_p5_tests.py", "nb4_ppack_tests.py"]

# ---------------------------------------------------------------- build-time facts from the export (fail loudly on drift)
_TREE = ast.parse(EXPORT)


def export_assign(name):
    nodes = [n for n in ast.walk(_TREE) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert nodes, f"{name} not assigned in the export"
    return nodes[-1].value


def export_json(name):
    v = export_assign(name)
    assert isinstance(v, ast.Call) and getattr(v.func, "attr", None) == "loads", f"{name} is not json.loads(<literal>)"
    return json.loads(ast.literal_eval(v.args[0]))


NB4_PIN = export_json("NB4_PIN")
PAY_PINS = dict(FACTS["payload_files"])                      # published path in the dataset -> sha256
assert PAY_PINS["nb4_p5_engine.py"] == NB4_PIN["hooks"]["nb4_p5_engine.py"], "build facts vs the notebook's hook pin"
assert NB4_PIN["hook_source"]["nb4_p5_engine.py"] == "payload " + PAYLOAD_SLUG
assert NB4_PIN["hook_source"]["nb4_timeline.py"] == "embedded" and NB4_PIN["hook_source"]["nb4_ppack.py"] == "embedded"
sys.path.insert(0, str(HERE))
import nb4_lib as NLB          # noqa: E402  (build-time checks only)
import nb4_p5_engine as P5B    # noqa: E402
import nb4_ppack as PKB        # noqa: E402
assert hashlib.sha256((HERE / "nb4_p5_engine.py").read_bytes()).hexdigest() == PAY_PINS["nb4_p5_engine.py"]
for _n, _h in P5B.P5_SRC_SHA.items():
    assert PAY_PINS["p5_src/" + _n] == _h, f"P5 source pin {_n}"
assert set(PAY_PINS) == {"nb4_p5_engine.py"} | {"p5_src/" + n for n in P5B.P5_SRC_SHA}
for _n, _h in PAY_PINS.items():                              # the local dataset folder = what was uploaded
    assert hashlib.sha256((PAYLOAD_DIR / _n).read_bytes()).hexdigest() == _h, f"local payload {_n}"
_sums = dict(reversed(l.split("  ", 1)) for l in (PAYLOAD_DIR / "SHA256SUMS").read_text().splitlines() if l.strip())
assert _sums == PAY_PINS, "local SHA256SUMS != pins"
assert hashlib.sha256((HERE / "nb4_lib.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest() == NB4_PIN["lib_sha256"]
for _n in ("nb4_timeline.py", "nb4_ppack.py"):
    assert hashlib.sha256((HERE / _n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == NB4_PIN["hooks"][_n], _n

# the names / signatures the kit drives in the notebook's globals
G_NAMES = ["ARMS", "NL", "NB4", "NB4_PIN", "NB4_P5", "NB4_PK", "NB4_TL", "NB4_EMB_DIR", "NB4_TIMER", "NB4_PROF_DIR", "NB4_DISABLED",
           "LAUNCH", "BOOT", "RECS", "HELP", "HOST_PHASE", "NB1_KV", "NB1_ENV", "NB1_EXTRA", "NB4_PREBUILD", "EXT_FLAG", "DIAG_ENV",
           "P60490_PRE", "P60490_POST", "P60490_REL", "CKPT", "TMP", "MERLIN_ENG"]
for _n in G_NAMES:
    assert re.search(rf"^\s*{_n}\s*=|^\s*{_n}\s*,|,\s*{_n}\s*=|\b{_n}, |^\s*import \S+ as {_n}\b", EXPORT, re.M), f"export lacks {_n}"
SIGS = ["def nb4_plan(arm_key, try_i, rung=0, drop_commit=False):", "def merlin_fn_server(name, kv, port, kv_bytes):",
        'def ppack_prepare(variant="Q-pack"):', "def ppack_finish():", "def p5_finish():", "def nb4_facts(s, a, o):",
        "def patch_states():", "def core_timer_install():", "def _build_env(eng, cc=True):", "def _tree_py():",
        "def merlin_install():", "def apply_60490():", "def sha_gate(timeout):", "def nb4_main():", "def clear_engine_procs(",
        "def drop_page_cache(", "def host_ram_gate(", "def cgroup_snapshot(tag):", "def profiler_tools("]
for _s in SIGS:
    assert _s in EXPORT, f"signature changed: {_s}"
for _t in ('new = NL.make_arms(prep if k == "S2" else None, NB4_PIN.get("g_override"), prep if k == "P" else None,',
           'prep if k == "S3" else None, NB4_DISABLED)[k]', 'HELP[name] = L.help_check("merlin", s.argv[:s.argv.index("serve") + 1], s.env, s.argv)',
           'raise RuntimeError("FAIL_CLI_FLAGS: " + " ".join(h["missing"]))', 'if not os.environ.get("RC2_CPU_KIT"):',
           'for pk, arm_k in (("p5", "P"), ("ppack", "S2")):', 'rec["patch"] = NB4_P5.apply_p5_patch(tree, py)',
           'rec.update(ok=True, env=NB4_P5.server_env(so, so_sha, max_bytes=NL.P5_MAX_BYTES), init_rx=r\'NB4_P5 \\{.*"event": "init"\',',
           'src = Path(str(NB4.get("p5_module"))).parent / "p5_src"', '"--src", str(src)], capture_output=True, text=True, timeout=900, env=_build_env(eng, mode == "merlin_cc"))',
           'if NB4["core_timer"].get("ok") is not True:', 'ARMS["T0"]["core_timer"] = False'):
    assert _t in EXPORT, f"the kit's reproduction of the notebook is stale: {_t!r}"

# literal expectations of every arm's ONE change vs the pure base (T0b); checked here against nb4_lib.make_arms
CAPS16 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]
G_CAPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 36]
K4_CAPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 35, 40, 45]
B3_PIN = 3553580524
P5_KEYS = ["NB4_P5", "NB4_P5_SO", "NB4_P5_SO_SHA256", "NB4_P5_MAX_BYTES", "NB4_P5_TIMEOUT_MS", "NB4_P5_LOG_S", "NB4_P5_SELFTEST"]
EXPECT = {"T0": {"env": {"VLLM_RECOVERSSM_COMMIT_GRAPH": "graph"}, "timer": True, "flags": ["--enable-logging-iteration-details"],
                 "caps": CAPS16, "k": 3, "draft": None, "iter": True},
          "S": {"env": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}, "caps": CAPS16, "k": 3, "draft": None},
          "S2": {"env": {"NB4_PPACK": "1"}, "caps": CAPS16, "k": 3, "draft": None, "patch": "ppack", "variant": "Q-pack"},
          "S3": {"env": {"NB4_PPDEFER": "1"}, "caps": CAPS16, "k": 3, "draft": None, "patch": "ppack", "variant": "Q-defer"},
          "G": {"env": {}, "caps": G_CAPS, "k": 3, "draft": None},
          "P": {"env_keys": P5_KEYS, "env_fixed": {"NB4_P5": "1", "NB4_P5_MAX_BYTES": "184320"}, "caps": CAPS16, "k": 3, "draft": None,
                "patch": "p5"},
          "T0b": {"env": {}, "caps": CAPS16, "k": 3, "draft": None},
          "K4": {"env": {}, "caps": K4_CAPS, "k": 4, "draft": None, "optional": True},
          "INT8": {"env": {}, "caps": CAPS16, "k": 3, "draft": "int8", "prebuild": {"DRAFT_HEAD": "int8"}, "optional": True}}
MANDATORY = ["T0", "S", "S2", "S3", "G", "P", "T0b"]
assert list(NLB.ARM_ORDER) == MANDATORY + ["K4", "INT8"] and list(NLB.OPTIONAL) == ["K4", "INT8"]
assert NLB.CAPS16 == CAPS16 and NLB.G_CAPS == G_CAPS and NLB.K4_CAPS == K4_CAPS and NLB.B3_PIN == B3_PIN and NLB.SEQS == 9
_A = NLB.make_arms()
for _k, _e in EXPECT.items():
    _a = _A[_k]
    assert NLB.compile_dict(_a)["cudagraph_capture_sizes"] == _e["caps"], _k
    assert (_a.get("k") or 3) == _e["k"], _k
    assert dict(_a.get("prebuild_env") or {}) == dict(_e.get("prebuild") or {}), _k
    assert [str(x) for x in _a.get("extra") or []] == list(_e.get("flags") or []), _k
    if "env" in _e and not _e.get("patch"):
        assert dict(_a.get("env") or {}) == _e["env"], _k
    assert _a.get("patch") == _e.get("patch") and _a.get("variant") == _e.get("variant"), _k
    assert not NLB.never_do(_a), _k
assert bool(_A["T0"].get("core_timer")) and _A["T0"]["commit_ab"] is True
assert PKB.server_env("Q-pack") == EXPECT["S2"]["env"] and PKB.server_env("Q-defer") == EXPECT["S3"]["env"]
assert sorted(P5B.server_env("x", "0" * 64, max_bytes=NLB.P5_MAX_BYTES)) == sorted(P5_KEYS) and NLB.P5_MAX_BYTES == 184320
assert json.loads(NLB.spec_json(4)) == {"method": "mtp", "num_speculative_tokens": 4, "draft_sample_method": "probabilistic"}
NEVER_KEYS = [[kind, key] for kind, key, _ in NLB.NEVER_DO]
MERLIN_PRE = {PKB.PP_REL: PKB.PP_PRE, PKB.MR_REL: PKB.MR_PRE, P5B.CC_REL: P5B.CC_PRE}
MERLIN_FILES = {}
for _rel, _h in MERLIN_PRE.items():
    _b = (MERLIN_SRC / _rel).read_bytes()
    assert hashlib.sha256(_b).hexdigest() == _h and b"\r" not in _b, _rel
    MERLIN_FILES[_rel] = _b.decode("utf-8")
UT_EXPECT = {}                       # unittest suites: (tests, skips): skips = the real-data classes (absent on Kaggle and in
for _t, _skips in (("nb4_unit_tests.py", 3), ("nb4_timeline_tests.py", 5), ("nb4_p5_tests.py", 0)):   # the local mirror run)
    UT_EXPECT[_t] = [len(re.findall(r"^    def test_", (HERE / _t).read_text(encoding="utf-8"), re.M)), _skips]
assert UT_EXPECT["nb4_unit_tests.py"][0] == 39, UT_EXPECT


def blob(files):
    """{relative path: text} -> lzma + b85 (keeps each kit below the 400 kB commit limit)."""
    return base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


EXPORT_BLOB = blob({"nb4_code_export.py": EXPORT})
_units = {NB_NAME: NB_RAW.decode("utf-8")}
for _n in ("nb4_lib.py", "nb4_timeline.py", "nb4_ppack.py", "nb4_unit_tests.py", "nb4_timeline_tests.py", "nb4_ppack_tests.py",
           "nb4_p5_tests.py"):
    _units[_n] = (HERE / _n).read_bytes().decode("utf-8").replace("\r\n", "\n")
_units.update({"merlin-src/" + r: t for r, t in MERLIN_FILES.items()})
UNITS_BLOB = blob(_units)
UNITS_SHAS = {n: hashlib.sha256(t.encode("utf-8")).hexdigest() for n, t in _units.items()}

HEAD = r'''# E25 NB4 TICK CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, lzma
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")
K = Path(os.environ.get("KIT_TMP", "/tmp/nb4kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E25 NB4 TICK CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}}
PAY_PINS = json.loads(r"""__PAYPINS__""")
def save():
    (W / f"nb4_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def unpack(s):
    return json.loads(lzma.decompress(base64.b85decode(s)).decode("utf-8"))
def sh(name, args, env=None, timeout=3600, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})}, cwd=str(cwd or K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    return {"rc": rc, "sec": round(time.time() - t), "tail": out[-1500:], "_out": out}
def verdict(ok, recorded=None, extra_line=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "recorded": recorded or {}, "minutes": round((time.time() - T0) / 60, 1)}
    save()
    bad = [k for k, v in ok.items() if not v]
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:8000], flush=True)
    if extra_line:
        print(extra_line, flush=True)
    print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
def sha_file(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()
def payload_mount():
    return next((d for d in (Path(INPUT_ROOT) / "datasets/foysalemonshanto/e25-nb4-payload", Path(INPUT_ROOT) / "e25-nb4-payload")
                 if (d / "nb4_p5_engine.py").is_file()), None)
def payload_check():
    """Every pinned payload file on the mount vs the notebook pins; SHA256SUMS agrees; extras listed."""
    d, o = payload_mount(), {}
    o.update(dir=str(d), pins=len(PAY_PINS), files={})
    for n, want in sorted(PAY_PINS.items()):
        p = d / n if d is not None else None
        if p is None or not p.is_file():
            o["files"][n] = {"ok": False, "why": "missing"}
            continue
        h = sha_file(p)
        o["files"][n] = {"ok": h == want, "size": p.stat().st_size, "sha256": h[:16], **({} if h == want else {"want": want[:16]})}
    if d is not None:
        try:
            sums = dict(reversed(l.split("  ", 1)) for l in (d / "SHA256SUMS").read_text().splitlines() if l.strip())
        except Exception as e:
            sums, o["sha256sums_error"] = {}, repr(e)[:200]
        o["sha256sums_disagree"] = sorted(n for n in set(sums) | set(PAY_PINS) if sums.get(n) != PAY_PINS.get(n))
        o["on_mount_unpinned"] = sorted(str(p.relative_to(d)).replace(os.sep, "/") for p in d.rglob("*") if p.is_file()
                                        and str(p.relative_to(d)).replace(os.sep, "/") not in PAY_PINS
                                        and p.name not in ("SHA256SUMS", "dataset-metadata.json"))
    o["ok"] = d is not None and all(f["ok"] for f in o["files"].values()) and not o.get("sha256sums_disagree") and "sha256sums_error" not in o
    return o
'''

REAL = r'''
FILES = unpack("__EXPORTBLOB__")
EXPECT = json.loads(r"""__EXPECT__""")
MANDATORY = json.loads(r"""__MANDATORY__""")
(K / "nb4_code_export.py").write_text(FILES["nb4_code_export.py"], encoding="utf-8", newline="\n")
REP["export_sha256"] = hashlib.sha256(FILES["nb4_code_export.py"].encode("utf-8")).hexdigest()
PAY, R1, R3, R4, CT, PL, ENVR, R6, DF, NV, R7, FIN = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
G = None
SAFE_ENV = ("VLLM_", "NCCL_", "NB4_", "CUDA_", "TORCH", "PYTORCH_", "OMP_", "MERLIN_", "DRAFT_", "KV_", "P2P", "SCALES", "PYTHONPATH",
            "LD_PRELOAD", "PORT", "MASTER_PORT", "HF_", "TRANSFORMERS_", "TRITON_", "FLASHINFER_", "LIBRARY_PATH", "LD_LIBRARY_PATH",
            "CC", "CXX", "CUDAHOSTCXX", "GPUS", "VENV", "TREE", "CHAT_TEMPLATE", "WARM", "CACHE_ROOT", "COUNTERS")
def safe_env(env):
    """Recorded env: known engine keys with values; anything else (or a key that looks like a secret) by name only."""
    out = {}
    for k, v in sorted((env or {}).items()):
        secret = re.search(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|AUTH", k, re.I)
        out[k] = v if (k.startswith(SAFE_ENV) and not secret) else "<not recorded>"
    return out

# ---------------------------------------------------------------- PAY: the payload dataset vs the notebook pins
try:
    PAY.update(payload_check())
except Exception as e:
    PAY.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
log("PAY", json.dumps({k: v for k, v in PAY.items() if k not in ("files", "tb")})[:900], "| bad",
    [n for n, f in (PAY.get("files") or {}).items() if not f.get("ok")])
REP["checks"]["PAY"] = PAY; save()

# ---------------------------------------------------------------- R1: the notebook's CELLs 1-8 (RC2_UNIT: no main, no boots)
try:
    os.environ.update(RC2_UNIT="1", RC2_CPU_KIT="1", E23_OUT=str(K / "out"), E23_TMP=str(K / "tmp"), E23_INPUT=INPUT_ROOT,
                      RC2_SUBMISSION=str(K / "out/submission.json"), RC2_SHM=str(K / "shm"), NB4_BUDGET_SEC=str(6 * 3600))
    os.environ.pop("E23_FAKE_SERVER", None)
    os.environ.pop("NB4_DISABLE", None)
    for d in ("out", "tmp", "shm"):
        (K / d).mkdir(exist_ok=True)
    t = time.time()
    G = runpy.run_path(str(K / "nb4_code_export.py"), run_name="nb4kit")["nb4_main"].__globals__
    R, NB, NL = G["R"], G["NB4"], G["NL"]
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"],
              prompt_render={k: v for k, v in R["prompt_render"].items() if "text" not in k}, prompts=R.get("prompts"),
              embedded=NB.get("embedded_sha_ok"), modules={k: NB.get(k) for k in ("timeline_module", "p5_module", "ppack_module")},
              arm_problems=NB.get("arm_problems"), disabled=NB.get("disabled"),
              arms_never_do={k: NL.never_do(a) for k, a in G["ARMS"].items()},
              arms_at_load={k: {x: a.get(x) for x in ("integrated", "why_not", "pending_prepare", "patch", "variant")} for k, a in G["ARMS"].items()},
              oracle_prompt_identity=NB.get("oracle_prompt_identity"), host=R["sections"]["A0"]["host"])
    pm = Path(str(NB.get("p5_module")))
    emb = Path(str(G["NB4_EMB_DIR"]))
    R1["module_sources_ok"] = {
        "p5_from_payload_mount": G["NB4_P5"] is not None and PAY.get("dir") not in (None, "None") and pm.parent.resolve() == Path(PAY["dir"]).resolve(),
        "timeline_embedded": G["NB4_TL"] is not None and Path(str(NB.get("timeline_module"))).parent.resolve() == emb.resolve(),
        "ppack_embedded": G["NB4_PK"] is not None and Path(str(NB.get("ppack_module"))).parent.resolve() == emb.resolve()}
    mount = Path(PAY["dir"]) if PAY.get("dir") not in (None, "None") else None
    R1["notebook_pins_vs_mount"] = {
        "hook_pin": G["NB4_PIN"]["hooks"].get("nb4_p5_engine.py") == PAY_PINS.get("nb4_p5_engine.py"),
        "p5_src_sha_vs_mount": {n: (mount is not None and (mount / "p5_src" / n).is_file() and sha_file(mount / "p5_src" / n) == h)
                                for n, h in (G["NB4_P5"].P5_SRC_SHA.items() if G["NB4_P5"] is not None else {})}}
    src = emb / "p5_src"                       # p5_prepare's own source lookup (embedded with the module, else next to the payload copy)
    if not src.is_dir():
        src = pm.parent / "p5_src"
    R1["p5_prepare_src"] = str(src)
    R1["p5_prepare_src_bad"] = [n for n, h in G["NB4_P5"].P5_SRC_SHA.items() if not (src / n).is_file() or sha_file(src / n) != h] \
        if G["NB4_P5"] is not None else ["NB4_P5 is None"]
    log("R1 inputs", json.dumps(R["inputs"])[:500], "| modules", json.dumps(R1["modules"])[:600], json.dumps(R1["module_sources_ok"]))
    log("R1 merge", json.dumps(R1["merge"])[:400], "| problems", R1["arm_problems"], "| src bad", R1["p5_prepare_src_bad"])
    save()
except Exception as e:
    R1["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R1["tb"] = traceback.format_exc()[-2500:]
    log("R1 FATAL", R1["fatal"])
REP["checks"]["R1"] = R1; save()

# ---------------------------------------------------------------- R3: Merlin offline install
eng = None
try:
    t = time.time()
    eng = G["merlin_install"]()
    R3.update(ok=True, sec=round(time.time() - t), kv=dict(eng["kv"]), cuda_home=eng["cuda_home"])
except Exception as e:
    R3.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R3 install", json.dumps({k: v for k, v in R3.items() if k != "tb"})[:600])
REP["checks"]["R3"] = R3; save()
TREE = Path(eng["kv"]["MERLIN_TREE"]) if eng else None
VPY = eng["kv"]["MERLIN_VENV"] + "/bin/python" if eng else None
CPU_ENV = {"VLLM_TARGET_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": ""}   # kit only: vLLM's DeviceConfig on a GPU-less box


def venv_py(code, args=(), name="probe", timeout=900, extra_env=None, full_env=None):
    """A probe in the Merlin venv: with the tree on PYTHONPATH (what Merlin's build() exports), or with `full_env` = a plan's
    exact server env."""
    if full_env is not None:
        env = dict(full_env)
    else:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD")}
        env.update(PYTHONPATH=str(TREE), CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PATH=f"{eng['kv']['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}", **(extra_env or {}))
        if eng["kv"].get("MERLIN_SHIM"):
            env["LD_PRELOAD"] = eng["kv"]["MERLIN_SHIM"]
    (K / f"{name}.py").write_text(code)
    try:
        r = subprocess.run([VPY, str(K / f"{name}.py"), *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    js = next((json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("@JSON ")), None)
    return {"rc": rc, "json": js, "tail": out[-1200:]}


def import_probe(mods, name):
    code = ("import json, importlib\nout = {}\nfor m in " + json.dumps(mods) + ":\n    try:\n        x = importlib.import_module(m)\n"
            "        out[m] = {'ok': True, 'file': getattr(x, '__file__', None)}\n    except BaseException as e:\n"
            "        out[m] = {'ok': False, 'error': repr(e)[:300]}\nprint('@JSON ' + json.dumps(out))\n")
    r = venv_py(code, name=name, extra_env=CPU_ENV)
    return r["json"] or {m: {"ok": False, "error": "probe: " + r["tail"][-300:]} for m in mods}


def import_rule(before, after, mods):
    """NB1 kit rule: a module that imported before the patch must import after it (CPU-only import failures are recorded)."""
    return all((after.get(m) or {}).get("ok") is True or (before.get(m) or {}).get("ok") is not True for m in mods)


def tree_shas(rels):
    return {r: (sha_file(TREE / r) if (TREE / r).is_file() else None) for r in rels}


# ---------------------------------------------------------------- R4: #60490 on the installed tree (+ the read-only worker extension)
IMPORT_QSA = "import json\ntry:\n    import vllm.models.qwen4_exp.nvidia.qsa as m\n    print('@JSON ' + json.dumps({'ok': True, 'file': m.__file__, 'has_topk': hasattr(m, '_TOPK_WORKSPACE_BYTES')}))\nexcept BaseException as e:\n    print('@JSON ' + json.dumps({'ok': False, 'error': repr(e)[:400]}))\n"
try:
    R4["pre_sha"] = sha_file(TREE / G["P60490_REL"])
    rec = G["apply_60490"]()
    R4["apply"] = rec
    R["sections"]["A0"]["patch_60490"] = dict(rec, status="DONE")      # what CELL 8's step() records (nb4_facts reads it)
    R4["import_after"] = venv_py(IMPORT_QSA, name="r4_import_after", extra_env=CPU_ENV)
    ia = R4["import_after"]["json"] or {}
    R4["ok"] = (R4["pre_sha"] == G["P60490_PRE"] and rec.get("ok") is True and rec.get("applied") == "applied" and rec.get("marker") is True
                and rec.get("sha_after") == G["P60490_POST"] and (rec.get("py_compile") or {}).get("rc") == 0
                and (ia.get("ok") is not True or ia.get("has_topk") is True))
except Exception as e:
    R4.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R4 patch", json.dumps({k: (v if k != "apply" else {x: v.get(x) for x in ("applied", "sha_after", "marker", "py_compile", "vllm_origin")})
                            for k, v in R4.items() if k != "tb"}, default=str)[:1200])
REP["checks"]["R4"] = R4; save()

# ---------------------------------------------------------------- CT: the engine-core step timer (CELL 8 installs it before the arms)
TIMER_PROBE = r"""
import json, os, multiprocessing as mp
out = {}
try:
    mp.current_process().name = "EngineCore_kit"
    os.environ["NB4_CORE_TIMER"] = os.path.join(os.getcwd(), "kit_core_timer")
    import nb4_core_timer as T
    out["file"] = T.__file__
    T.register()
    import vllm.v1.engine.core as c
    ec = c.EngineCore
    out["wrapped"] = {n: hasattr(getattr(ec, n, None), "__wrapped__") for n in ("step", "step_with_batch_queue")}
    out["flag"] = getattr(ec, "_nb4_timer", False)
    out["state"] = {k: v for k, v in T._ST.items() if k != "errors"}
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"[:400]
print("@JSON " + json.dumps(out, default=str))
"""
try:
    CT["install"] = G["core_timer_install"]()
    G["NB4"]["core_timer"] = dict(CT["install"], status="DONE")
    if CT["install"].get("ok") is not True:                       # = nb4_main: T0 keeps vLLM's iteration log only
        G["ARMS"]["T0"]["core_timer"] = False
    CT["wrap_probe"] = venv_py(TIMER_PROBE, name="ct_wrap_probe", extra_env=CPU_ENV)["json"]
    CT["ok"] = CT["install"].get("ok") is True
except Exception as e:
    CT.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("CT core timer", json.dumps({k: v for k, v in CT.items() if k != "tb"}, default=str)[:900])
REP["checks"]["CT"] = CT; save()

# ---------------------------------------------------------------- PLANS: every arm as nb4_arm_step builds it (prepare -> plan -> revert)
class Stub:
    """A launched-but-not-started server for the notebook's nb4_facts(): the plan's argv / env / port, no process, no log."""
    def __init__(self, s):
        self.argv, self.env, self.port, self.proc = s.argv, s.env, s.port, None
    def log(self):
        return ""
    def alive(self):
        return False


plans, servers = {}, {}
os.environ["VLLM_TARGET_DEVICE"] = "cpu"     # kit only: the help check subprocess inherits it via Merlin's build() env
PP_RELS = sorted(G["NB4_PK"].PATCHES) if G and G.get("NB4_PK") is not None else []
CC_REL = G["NB4_P5"].CC_REL if G and G.get("NB4_P5") is not None else None
try:
    ARMS, NL = G["ARMS"], G["NL"]
    for k in list(NL.ARM_ORDER):
        a = ARMS[k]
        b = {"arm": k, "optional": k in NL.OPTIONAL}
        plans[k] = b
        prep, fin = None, None
        try:
            if a.get("patch") == "ppack":
                b["pre_shas"] = tree_shas(PP_RELS)
                b["import_before"] = import_probe(["vllm.v1.worker.gpu.pp_utils", "vllm.v1.worker.gpu.model_runner"], f"pp_import_before_{k}")
                prep = G["ppack_prepare"](a["variant"])              # the notebook's own prepare: apply_ppack_patch + server_env
                b["prepare"] = {x: prep.get(x) for x in ("ok", "why", "env", "variant", "init_rx")}
                b["prepare_patch"] = {x: (prep.get("patch") or {}).get(x) for x in ("applied", "after", "py_compile", "ok", "patch_sha")}
                b["state_after_prepare"] = G["NB4_PK"].verify_ppack_patch(TREE)["state"]
                b["patch_states_after_prepare"] = G["patch_states"]()
                b["import_after"] = import_probe(["vllm.v1.worker.gpu.pp_utils", "vllm.v1.worker.gpu.model_runner"], f"pp_import_after_{k}")
                b["import_rule_ok"] = import_rule(b["import_before"], b["import_after"], ["vllm.v1.worker.gpu.pp_utils", "vllm.v1.worker.gpu.model_runner"])
            elif a.get("patch") == "p5":
                P5 = G["NB4_P5"]
                b["pre_shas"] = tree_shas([CC_REL, P5.BACKEND_REL, P5.WORKER_EXT_REL])
                src = Path(R1["p5_prepare_src"])
                b["src_bad"] = [n for n, h in P5.P5_SRC_SHA.items() if not (src / n).is_file() or sha_file(src / n) != h]
                tries, so, so_sha = [], None, None
                tree_, py_, eng_ = G["_tree_py"]()
                outd = Path(G["TMP"]) / "nb4_p5"
                for mode in ("merlin_cc", "system_cc"):          # RECORDED ONLY: p5_prepare's build command (build_host_ar.py
                    t = time.time()                              # says no GPU is needed; the run's gate + init still need GPUs)
                    try:
                        cp = subprocess.run([py_, str(src / "build_host_ar.py"), "--out", str(outd), "--build-dir",
                                             str(Path(G["TMP"]) / f"nb4_p5_build_{mode}"), "--src", str(src)], capture_output=True, text=True,
                                            timeout=1500, env=G["_build_env"](eng_, mode == "merlin_cc"))
                        txt = cp.stdout + cp.stderr
                        (W / f"kit_p5_build_{mode}.log").write_text(txt)
                        bj = next((json.loads(l[len("BUILD_JSON "):]) for l in txt.splitlines() if l.startswith("BUILD_JSON ")), {})
                        tries.append({"mode": mode, "rc": cp.returncode, "ok": bj.get("ok"), "so": bj.get("so"), "so_sha256": bj.get("so_sha256"),
                                      "sec": round(time.time() - t), "error": (bj.get("error") or "")[-600:], "spill_lines": bj.get("spill_lines")})
                    except Exception as e:
                        tries.append({"mode": mode, "ok": False, "error": f"{type(e).__name__}: {e}"[:300], "sec": round(time.time() - t)})
                    if tries[-1].get("ok") and tries[-1].get("so"):
                        so = tries[-1]["so"]
                        so_sha = sha_file(so)
                        break
                b["so_cpu_compile_probe"] = {"recorded_only": True, "tries": tries, "so": so, "so_sha256": so_sha}
                b["p5_gate"] = "SKIPPED_GPU_ONLY"               # nb4_p5_gate.py: 4 processes on the 4 GPUs
                b["p5_engine_init"] = "SKIPPED_GPU_ONLY"        # NB4_P5 init inside CudaCommunicator.__init__ needs the GPUs
                b["so_status"] = "SKIPPED_GPU_ONLY" if not so else "SKIPPED_GPU_ONLY (the CPU compile probe built a .so: used for the plan env)"
                pay = P5.write_payload(K / "nb4_p5_pay")
                b["payload_write"] = pay
                b["gate_argv"] = P5.gate_argv(py_, pay["nb4_p5_gate.py"]["path"], K / "nb4_p5_pay", so or "<so>", so_sha or "0" * 64,
                                              K / "p5_gate.json", max_bytes=NL.P5_MAX_BYTES)
                b["import_before"] = import_probe(["vllm.distributed.device_communicators.cuda_communicator"], "p5_import_before")
                b["apply"] = P5.apply_p5_patch(tree_, py_)        # = p5_prepare's step 3 (strict, origin check)
                b["state_after_prepare"] = P5.verify_p5_patch(TREE)
                b["patch_states_after_prepare"] = G["patch_states"]()
                mods = ["vllm.distributed.device_communicators.cuda_communicator", "nb4_p5_backend", "nb4_p5_worker_ext"]
                b["import_after"] = import_probe(mods, "p5_import_after")
                b["import_rule_ok"] = import_rule(b["import_before"], b["import_after"], mods[:1]) and \
                    (b["import_after"].get("nb4_p5_backend") or {}).get("ok") is True     # the worker's nb4_p5_stats imports it
                so_used = so or str(outd / "p5_host_ar.so")
                so_sha_used = so_sha or P5.P5_SO_NB3_SHA
                b["so_for_plan"] = {"so": so_used, "sha256": so_sha_used, "source": "cpu_compile_probe" if so else "placeholder (NB3 D7 sha)"}
                prep = {"ok": True, "env": P5.server_env(so_used, so_sha_used, max_bytes=NL.P5_MAX_BYTES), "init_rx": r'NB4_P5 \{.*"event": "init"',
                        "counters_rpc": "nb4_p5_stats", "so": so_used, "so_sha256": so_sha_used}
            if prep is not None:                               # = nb4_arm_step: the arm from its prepare result
                new = NL.make_arms(prep if k == "S2" else None, G["NB4_PIN"].get("g_override"), prep if k == "P" else None,
                                   prep if k == "S3" else None, G["NB4_DISABLED"])[k]
                if prep.get("ok") is not True:
                    raise RuntimeError(f"{k} prepare not ok: {prep.get('why')}")
                ARMS[k] = new
                a = new
                b["arm_problems_after_prepare"] = NL.validate_arms(ARMS)
            b["never_do_arm"] = NL.never_do(a)
            b["arm"] = {x: a.get(x) for x in ("integrated", "why_not", "env", "extra", "compile", "spec", "prebuild_env", "patch", "variant", "core_timer")}
            name, port = G["nb4_plan"](k, 0)
            kvb = G["NB1_KV"][name]
            b.update(name=name, port=port, kv_bytes=kvb, launch=dict(G["LAUNCH"][name]))
            s = None
            try:
                s = G["merlin_fn_server"](name, "fp8", port, kvb)
                b.update(ok=True, server_started=s.proc is not None)
            except Exception as e:
                b.update(ok=False, error=f"{type(e).__name__}: {e}"[:800])
                if str(e).startswith("FAIL_CLI_FLAGS"):           # kit-only recording pass: the argv for R6 / DIFF despite K-9
                    real_help, orig = G["HELP"].get(name), G["L"].help_check
                    G["L"].help_check = lambda *x, **y: {"ran": None, "why": "kit recording pass"}
                    try:
                        s = G["merlin_fn_server"](name, "fp8", port, kvb)
                        b["argv_from"] = "recording_pass_after_FAIL_CLI_FLAGS"
                    except Exception as e2:
                        b["recording_pass_error"] = f"{type(e2).__name__}: {e2}"[:400]
                    finally:
                        G["L"].help_check = orig
                        G["HELP"][name] = real_help
            b.update(host_phase=G["HOST_PHASE"].get(name), help=G["HELP"].get(name))
            if s is not None:
                servers[k] = s
                b["argv"] = list(s.argv)
                b["env"] = safe_env(s.env)
                b["plan_record"] = {x: ((G["R"]["setup"].get("nb4_plans") or {}).get(name) or {}).get(x) for x in ("arm", "kv_bytes", "d8", "prebuild_env", "diff_vs_rc2_fp8")}
                G["BOOT"]["tag"] = name                          # the notebook's own read-back on a stub (no log: only the
                o = {}                                           # patch-state keys and one_change_vs_ref are meaningful)
                try:
                    G["nb4_facts"](Stub(s), a, o)
                    ck = (o.get("readback") or {}).get("checks") or {}
                    b["stub_facts"] = {"patch_keys": {x: v for x, v in ck.items() if " patch " in x}, "patch_60490": ck.get("patch_60490"),
                                       "one_change_vs_ref": ck.get("one_change_vs_ref"), "diff_vs_ref": o.get("diff_vs_ref"),
                                       "patch_states": o.get("patch_states"), "all_checks": ck}
                except Exception as e:
                    b["stub_facts"] = {"error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1200:]}
                if k == "T0":                                    # T0 = the run's reference arm (nb4_arm_step: the first clean base)
                    G["NB4"]["ref"] = "T0"
                    G["RECS"]["T0"] = {"launch_name": name, "kv_bytes": kvb}
        except Exception as e:
            b.update(ok=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
        finally:
            if a.get("patch") == "ppack" or (plans[k].get("pre_shas") and k in ("S2", "S3")):
                try:
                    b["finish"] = {x: v for x, v in G["ppack_finish"]().items() if x in ("ok", "reverted", "after", "py_compile", "error", "skipped")}
                except Exception as e:
                    b["finish"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
                b["post_shas"] = tree_shas(PP_RELS)
            elif a.get("patch") == "p5" or k == "P":
                try:
                    b["finish"] = G["p5_finish"]()
                except Exception as e:
                    b["finish"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
                P5 = G["NB4_P5"]
                b["post_shas"] = tree_shas([CC_REL, P5.BACKEND_REL, P5.WORKER_EXT_REL])
            if b.get("pre_shas") is not None:
                b["patch_states_after_finish"] = G["patch_states"]()
        log("PLAN", k, json.dumps({x: v for x, v in b.items() if x not in ("argv", "env", "help", "tb", "plan_record", "import_before", "import_after",
                                                                         "payload_write", "gate_argv", "stub_facts")}, default=str)[:1500])
        log("PLAN", k, "stub facts", json.dumps({x: (b.get("stub_facts") or {}).get(x) for x in ("patch_keys", "one_change_vs_ref", "patch_60490", "error")}, default=str)[:600])
        save()
except Exception as e:
    PL["fatal"] = f"{type(e).__name__}: {e}"[:800]; PL["tb"] = traceback.format_exc()[-1500:]
G["NB4"]["ref"] = None
G["RECS"].pop("T0", None)
try:
    _ht = G["L"]._HELP.get("merlin") or ""
    PL.update(help_text_head=_ht[:3000], help_text_len=len(_ht), help_real="--max-num-seqs" in _ht and "Failed to infer device" not in _ht)
except Exception as e:
    PL.update(help_text_head=repr(e)[:200], help_real=False)
PL["plans"] = plans


def ppack_ok(k):
    b = plans.get(k) or {}
    pr, fi = b.get("prepare") or {}, b.get("finish") or {}
    return (pr.get("ok") is True and pr.get("env") == EXPECT[k]["env"] and (b.get("prepare_patch") or {}).get("ok") is True
            and ((b.get("prepare_patch") or {}).get("py_compile") or {}).get("rc") == 0 and b.get("state_after_prepare") == "applied"
            and (b.get("patch_states_after_prepare") or {}).get("ppack") == "applied" and fi.get("ok") is True
            and (b.get("patch_states_after_finish") or {}).get("ppack") == "not_applied"
            and b.get("pre_shas") == b.get("post_shas") and all(b.get("pre_shas", {}).values()) and b.get("import_rule_ok") is True)


def p5_ok():
    b = plans.get("P") or {}
    ap, fi, st = b.get("apply") or {}, b.get("finish") or {}, b.get("state_after_prepare") or {}
    pre, post = b.get("pre_shas") or {}, b.get("post_shas") or {}
    P5 = G["NB4_P5"]
    return (ap.get("ok") is True and ap.get("sha_after") == P5.CC_POST and ap.get("marker") is True and (ap.get("py_compile") or {}).get("rc") == 0
            and st.get("state") == "applied" and (b.get("patch_states_after_prepare") or {}).get("p5") == "applied"
            and fi.get("ok") is True and fi.get("sha_after") == P5.CC_PRE and pre.get(CC_REL) == post.get(CC_REL) == P5.CC_PRE
            and post.get(P5.BACKEND_REL) is None and post.get(P5.WORKER_EXT_REL) is None
            and (b.get("patch_states_after_finish") or {}).get("p5") == "not_applied" and b.get("import_rule_ok") is True
            and not b.get("src_bad") and all(v.get("sha256") for v in (b.get("payload_write") or {}).values())
            and len(b.get("payload_write") or {}) == 3)


PL["ppack_S2_ok"] = ppack_ok("S2") if G else False
PL["ppack_S3_ok"] = ppack_ok("S3") if G else False
PL["p5_patch_ok"] = p5_ok() if G else False
for k in plans:
    b = plans[k]
    b["plan_ok"] = (b.get("ok") is True and b.get("host_phase") == "ok" and isinstance(b.get("help"), dict) and b["help"].get("ran") is True
                    and not b["help"].get("missing") and bool(b.get("argv")) and not b.get("never_do_arm") and not b.get("arm_problems_after_prepare")
                    and "fatal" not in b and (b.get("launch") or {}).get("drop_commit") is False)
REP["checks"]["PL"] = PL; save()

# ---------------------------------------------------------------- ENV: each plan's exact server env reaches a worker-side process
ENV_PROBE = r"""
import json, sys
out = {}
try:
    import nb1_worker_ext as X
    out["file"] = X.__file__
    from vllm.utils.import_utils import resolve_obj_by_qualname
    out["resolve"] = resolve_obj_by_qualname("nb1_worker_ext.NB1WorkerExt") is X.NB1WorkerExt
    out["methods"] = {m: callable(getattr(X.NB1WorkerExt, m, None)) for m in ("nb4_env", "nb4_p5_stats", "nb1_cache_info")}
    class D(X.NB1WorkerExt):
        pass
    out["nb4_env"] = D().nb4_env(sys.argv[1])
    try:
        from vllm.v1.worker.gpu_worker import Worker
        out["worker_methods"] = {m: hasattr(Worker, m) for m in ("set_recoverssm_commit_mode", "get_recoverssm_commit_stats")}
        out["conflicts"] = [a for a in ("nb4_env", "nb4_p5_stats") if hasattr(Worker, a)]
    except BaseException as e:
        out["worker_import_error"] = repr(e)[:300]
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"[:500]
print("@JSON " + json.dumps(out, default=str))
"""
ENV_KEYS = ["CUDA_DEVICE_MAX_CONNECTIONS", "VLLM_RECOVERSSM_COMMIT_GRAPH", "NB4_PPACK", "NB4_PPDEFER", "NB4_CORE_TIMER", "DRAFT_HEAD",
            "VLLM_SERVER_DEV_MODE", "VLLM_CUSTOM_SCOPES_FOR_PROFILING"] + [x for x in EXPECT["P"]["env_keys"]]
try:
    for k, s in servers.items():
        r = venv_py(ENV_PROBE, [",".join(ENV_KEYS)], name=f"env_probe_{k}", full_env=s.env)
        js = r["json"] or {"error": r["tail"][-600:]}
        got = ((js.get("nb4_env") or {}).get("env") or {})
        want = {x: None for x in ENV_KEYS}
        want.update(VLLM_SERVER_DEV_MODE="1", VLLM_CUSTOM_SCOPES_FOR_PROFILING="1")
        want.update((plans[k].get("arm") or {}).get("env") or {})
        if k == "T0" and (plans[k].get("arm") or {}).get("core_timer"):
            want["NB4_CORE_TIMER"] = str(G["NB4_TIMER"])
        want_opt = {x: v for x, v in (EXPECT[k].get("prebuild") or {}).items()}    # DRAFT_HEAD: kept or consumed by build()
        bad = {x: [want[x], got.get(x)] for x in ENV_KEYS if got.get(x) != want[x] and not (x in want_opt and got.get(x) in (None, want_opt[x]))}
        ENVR[k] = {"got": got, "bad": bad, "resolve": js.get("resolve"), "methods": js.get("methods"), "worker_methods": js.get("worker_methods"),
                   "conflicts": js.get("conflicts"), "error": js.get("error"), "worker_import_error": js.get("worker_import_error"),
                   "ok": not bad and js.get("resolve") is True and all((js.get("methods") or {"x": False}).values()) and not js.get("error")}
        log("ENV", k, json.dumps(ENVR[k], default=str)[:700]); save()
except Exception as e:
    ENVR["fatal"] = f"{type(e).__name__}: {e}"[:800]; ENVR["tb"] = traceback.format_exc()[-1500:]
REP["checks"]["ENV"] = ENVR; save()

# ---------------------------------------------------------------- R6: vLLM's own argparse on the exact argvs (field checks)
ARGPARSE = r"""
import json, os, sys
argvs = json.load(open(sys.argv[1]))
out = {"_env": {k: os.environ.get(k) for k in ("VLLM_TARGET_DEVICE", "CUDA_VISIBLE_DEVICES")}}
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.engine.arg_utils import AsyncEngineArgs
except BaseException as e:
    print("@JSON " + json.dumps({"import_error": repr(e)[:400]})); raise SystemExit(0)
try:
    from vllm.platforms import current_platform
    out["_platform"] = type(current_platform).__name__
except BaseException as e:
    out["_platform"] = "ERROR " + repr(e)[:300]
def g(o, k):
    v = getattr(o, k, None)
    return v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)
for label, argv in argvs.items():
    try:
        a = argv[argv.index("serve") + 1:]
        if a and not a[0].startswith("--"):
            a = ["--model", a[0]] + a[1:]
        p = make_arg_parser(FlexibleArgumentParser())
        ns, unknown = p.parse_known_args(a)
        r = {"unknown": list(unknown)[:40]}
        try:
            ea, r["source"] = AsyncEngineArgs.from_cli_args(ns), "engine_args"
        except BaseException as e:
            ea, r["source"], r["from_cli_args_error"] = ns, "namespace", f"{type(e).__name__}: {e}"[:300]
        cc = getattr(ea, "compilation_config", None)
        caps = cc.get("cudagraph_capture_sizes") if isinstance(cc, dict) else getattr(cc, "cudagraph_capture_sizes", None)
        mode = cc.get("cudagraph_mode") if isinstance(cc, dict) else getattr(cc, "cudagraph_mode", None)
        r.update({k: g(ea, k) for k in ("model", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes", "kv_cache_dtype",
                                         "worker_extension_cls", "max_model_len", "block_size", "mamba_cache_mode", "enable_prefix_caching",
                                         "tensor_parallel_size", "pipeline_parallel_size", "enable_expert_parallel", "language_model_only",
                                         "mamba_ssm_cache_dtype", "disable_custom_all_reduce", "async_scheduling")})
        r["capture_sizes"] = list(caps) if caps else None
        r["cudagraph_mode"] = str(mode)
        sc = getattr(ea, "speculative_config", None)
        r["speculative_config"] = sc if isinstance(sc, (dict, type(None))) else str(sc)[:300]
        pc = getattr(ea, "profiler_config", None)
        r["profiler"] = pc.get("profiler") if isinstance(pc, dict) else getattr(pc, "profiler", None)
        r["profiler_dir"] = pc.get("torch_profiler_dir") if isinstance(pc, dict) else getattr(pc, "torch_profiler_dir", None)
        r["profiler_record_shapes"] = pc.get("torch_profiler_record_shapes") if isinstance(pc, dict) else getattr(pc, "torch_profiler_record_shapes", None)
        r["ns_cudagraph_metrics"] = g(ns, "cudagraph_metrics")
        r["ns_iteration_details"] = g(ns, "enable_logging_iteration_details")
        r["default_chat_template_kwargs"] = str(getattr(ns, "default_chat_template_kwargs", None))[:120]
        r["ok"] = True
    except BaseException as e:
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
    out[label] = r
print("@JSON " + json.dumps(out, default=str))
"""
try:
    argvs = {k: v["argv"] for k, v in plans.items() if v.get("argv")}
    (K / "argvs.json").write_text(json.dumps(argvs))
    r = venv_py(ARGPARSE, [str(K / "argvs.json")], name="r6_argparse", extra_env=CPU_ENV)
    R6["parsed"] = r["json"] or {"error": r["tail"]}
    fields = {}
    for k in plans:
        e, x, bad = EXPECT[k], (R6["parsed"] or {}).get(k) or {}, []
        if x.get("ok") is not True: bad.append("parse " + str(x.get("error"))[:120])
        if x.get("unknown"): bad.append(f"unknown flags {x.get('unknown')}")
        if x.get("source") != "engine_args": bad.append(f"from_cli_args {x.get('from_cli_args_error')}")
        if x.get("max_num_seqs") != 9: bad.append(f"max_num_seqs {x.get('max_num_seqs')}")
        if x.get("capture_sizes") != e["caps"]: bad.append(f"captures {x.get('capture_sizes')}")
        if "FULL_AND_PIECEWISE" not in str(x.get("cudagraph_mode")): bad.append(f"cudagraph_mode {x.get('cudagraph_mode')}")
        if x.get("kv_cache_memory_bytes") != __B3PIN__: bad.append(f"kv {x.get('kv_cache_memory_bytes')}")
        if x.get("kv_cache_dtype") != "fp8_e4m3": bad.append(f"kv_cache_dtype {x.get('kv_cache_dtype')}")
        if x.get("language_model_only") is not True: bad.append(f"language_model_only {x.get('language_model_only')}")
        if x.get("worker_extension_cls") != "nb1_worker_ext.NB1WorkerExt": bad.append(f"worker_extension_cls {x.get('worker_extension_cls')}")
        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")
        if x.get("enable_prefix_caching") is not True: bad.append("prefix caching")
        if x.get("model") != str(G["CKPT"]): bad.append(f"model {x.get('model')}")
        if (x.get("tensor_parallel_size"), x.get("pipeline_parallel_size"), x.get("enable_expert_parallel")) != (2, 2, True): bad.append("layout")
        if x.get("profiler") != "torch" or x.get("profiler_dir") != str(G["NB4_PROF_DIR"]): bad.append(f"profiler {x.get('profiler')} {x.get('profiler_dir')}")
        if x.get("ns_cudagraph_metrics") is not True: bad.append(f"--cudagraph-metrics {x.get('ns_cudagraph_metrics')}")
        if bool(x.get("ns_iteration_details")) != bool(e.get("iter")): bad.append(f"iteration details {x.get('ns_iteration_details')}")
        spec = x.get("speculative_config")
        sd = spec if isinstance(spec, dict) else {}
        if not sd or sd.get("method") != "mtp" or sd.get("num_speculative_tokens") != e["k"]: bad.append(f"spec {str(spec)[:120]}")
        if sd.get("draft_lm_head_quant") != e.get("draft"): bad.append(f"draft head {sd.get('draft_lm_head_quant')}")
        fields[k] = bad
    R6["field_check"] = fields
    R6["ok"] = bool(fields) and all(not fields.get(k, ["missing"]) for k in MANDATORY)
    R6["optional_ok"] = {k: not fields.get(k, ["missing"]) for k in ("K4", "INT8")}
except Exception as e:
    R6.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
os.environ.pop("VLLM_TARGET_DEVICE", None)
log("R6 argparse", json.dumps({k: (R6.get("parsed") or {}).get(k) for k in ("_env", "_platform", "import_error")}), json.dumps(R6.get("field_check"))[:1500])
REP["checks"]["R6"] = R6; save()

# ---------------------------------------------------------------- DIFF: every plan vs T0b (the pure base): exactly the planned change
IGN_FLAGS, IGN_ENV = ("--port",), ("PORT", "MASTER_PORT", "VLLM_PORT")
def flagmap(argv):
    a, pairs = [str(x) for x in argv], []
    for i, x in enumerate(a):
        if x.startswith("--"):
            pairs.append((x, a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else True))
    names = [f for f, _ in pairs]
    return dict(pairs), sorted({f for f in names if names.count(f) > 1})
def jl(v):
    try:
        return json.loads(v)
    except Exception:
        return ("<not json>", v)
BASE_SPEC = {"method": "mtp", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic"}
try:
    base = servers.get("T0b")
    if base is None:
        raise RuntimeError("no T0b plan: no base to diff against")
    bf, bdup = flagmap(base.argv)
    DF["base_duplicates"] = bdup
    bchk = {"seqs 9": bf.get("--max-num-seqs") == "9", "pin": bf.get("--kv-cache-memory-bytes") == str(__B3PIN__),
            "kv fp8_e4m3": bf.get("--kv-cache-dtype") == "fp8_e4m3", "--language-model-only": bf.get("--language-model-only") is True,
            "--cudagraph-metrics": bf.get("--cudagraph-metrics") is True, "--enable-prefix-caching": bf.get("--enable-prefix-caching") is True,
            "extension": bf.get("--worker-extension-cls") == "nb1_worker_ext.NB1WorkerExt",
            "compile CAPS16": jl(bf.get("--compilation-config")) == {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": EXPECT["T0b"]["caps"]},
            "spec K3": jl(bf.get("--speculative-config")) == BASE_SPEC,
            "profiler": jl(bf.get("--profiler-config")) == json.loads(G["NL"].profiler_json(G["NB4_PROF_DIR"])),
            "no iteration details": "--enable-logging-iteration-details" not in bf,
            "env VLLM_SERVER_DEV_MODE=1": base.env.get("VLLM_SERVER_DEV_MODE") == "1",
            "env VLLM_CUSTOM_SCOPES_FOR_PROFILING=1": base.env.get("VLLM_CUSTOM_SCOPES_FOR_PROFILING") == "1",
            "env no commit graph": base.env.get("VLLM_RECOVERSSM_COMMIT_GRAPH") in (None, ""),
            "env no max connections": base.env.get("CUDA_DEVICE_MAX_CONNECTIONS") in (None, ""),
            "env no NB4 arm keys": not [x for x in base.env if x.startswith("NB4_")]}
    DF["base_checks"] = bchk
    DF["base_ok"] = all(bchk.values())
    DF["arms"] = {}
    for k, s in servers.items():
        if k == "T0b":
            continue
        e = EXPECT[k]
        cf, cdup = flagmap(s.argv)
        fd = {f: [bf.get(f), cf.get(f)] for f in sorted(set(bf) | set(cf)) if f not in IGN_FLAGS and bf.get(f) != cf.get(f)}
        ed = {x: [base.env.get(x), s.env.get(x)] for x in sorted(set(base.env) | set(s.env)) if x not in IGN_ENV and base.env.get(x) != s.env.get(x)}
        want_f = {}
        for f in e.get("flags") or []:
            want_f[f] = True
        if e["caps"] != EXPECT["T0b"]["caps"]:
            want_f["--compilation-config"] = {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": e["caps"]}
        if e["k"] != 3 or e.get("draft"):
            want_f["--speculative-config"] = dict(BASE_SPEC, num_speculative_tokens=e["k"], **({"draft_lm_head_quant": e["draft"]} if e.get("draft") else {}))
        if k == "P":
            want_e = dict((plans[k].get("arm") or {}).get("env") or {})
            fixed_ok = all(want_e.get(x) == v for x, v in e["env_fixed"].items()) and sorted(want_e) == sorted(e["env_keys"])
        else:
            want_e, fixed_ok = dict(e.get("env") or {}), True
        if k == "T0" and (plans[k].get("arm") or {}).get("core_timer"):
            want_e["NB4_CORE_TIMER"] = str(G["NB4_TIMER"])
        opt_e = dict(e.get("prebuild") or {})
        bad = []
        for f, (bv, cv) in fd.items():
            if f not in want_f:
                bad.append(f"unexpected flag {f}: {str(bv)[:80]} -> {str(cv)[:80]}")
            elif want_f[f] is True:
                if cv is not True: bad.append(f"flag {f} value {cv}")
            elif jl(cv) != want_f[f]:
                bad.append(f"flag {f} value {str(cv)[:160]}")
        bad += [f"missing flag change {f}" for f in want_f if f not in fd]
        for x, (bv, cv) in ed.items():
            if x in want_e:
                if cv != want_e[x]: bad.append(f"env {x} = {cv!r}, want {want_e[x]!r}")
            elif x in opt_e and cv == opt_e[x] and bv is None:
                pass
            else:
                show = x.startswith(SAFE_ENV) and not re.search(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|AUTH", x, re.I)
                bad.append(f"unexpected env {x}" + (f": {str(bv)[:80]!r} -> {str(cv)[:80]!r}" if show else ""))
        bad += [f"missing env change {x}" for x in want_e if x not in ed]
        if cdup != bdup: bad.append(f"duplicate flags differ: {cdup} vs base {bdup}")
        if not fixed_ok: bad.append(f"P env keys / fixed values {sorted(want_e)}")
        DF["arms"][k] = {"flag_diff": {f: [str(v)[:200] for v in vv] for f, vv in fd.items()},
                         "env_diff_keys": sorted(ed), "expected_flags": sorted(want_f), "expected_env": sorted(want_e), "bad": bad, "ok": not bad}
        log("DIFF", k, json.dumps(DF["arms"][k], default=str)[:900])
    DF["ok"] = DF["base_ok"] and all((DF["arms"].get(k) or {}).get("ok") is True for k in MANDATORY if k != "T0b")
    DF["optional_ok"] = {k: (DF["arms"].get(k) or {}).get("ok") is True for k in ("K4", "INT8")}
except Exception as e:
    DF.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
REP["checks"]["DIFF"] = DF; save()

# ---------------------------------------------------------------- NEVER: the measured never-do list on every FINAL argv + env
try:
    for k, s in servers.items():
        hits, a = [], [str(x) for x in s.argv]
        for kind, key, bad in G["NL"].NEVER_DO:
            if kind == "env" and key in s.env and (bad is None or bad(s.env[key])):
                hits.append(f"env {key}={s.env[key]}")
            if kind == "flag":
                for i, x in enumerate(a):
                    if x == key:
                        v = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else ""
                        if bad is None or bad(v):
                            hits.append(f"flag {key} {v}".strip())
        NV[k] = {"hits": hits, "arm_never_do": plans[k].get("never_do_arm"), "P2P": s.env.get("P2P"),
                 "NCCL": {x: v for x, v in s.env.items() if x.startswith("NCCL_")}}
    NV["ok"] = bool(servers) and all(not NV[k]["hits"] and not NV[k]["arm_never_do"] for k in servers if k in MANDATORY) \
        and all(k in servers for k in MANDATORY)
    NV["optional_ok"] = {k: k in servers and not NV[k]["hits"] and not NV[k]["arm_never_do"] for k in ("K4", "INT8")}
    NV["list"] = [[kind, key] for kind, key, _ in G["NL"].NEVER_DO]
except Exception as e:
    NV.update(ok=False, error=f"{type(e).__name__}: {e}"[:800])
log("NEVER", json.dumps({k: v for k, v in NV.items() if k != "list"}, default=str)[:1500])
REP["checks"]["NEVER"] = NV; save()

# ---------------------------------------------------------------- R7: host gates, the tree's final state, the checkpoint sha256 gate
try:
    R7["engine_procs"] = G["clear_engine_procs"]()
    R7["page_cache_before"] = G["cgroup_snapshot"]("kit_before")
    R7["drop"] = G["drop_page_cache"]()
    R7["host_ram_gate"] = G["host_ram_gate"]("kit")
    R7["profiler_tools"] = G["profiler_tools"]()
except Exception as e:
    R7["error"] = f"{type(e).__name__}: {e}"[:600]; R7["tb"] = traceback.format_exc()[-1200:]
log("R7 host", json.dumps({k: v for k, v in R7.items() if k in ("engine_procs", "drop", "error")}, default=str)[:900])
REP["checks"]["R7"] = R7; save()
try:
    FIN["patch_states"] = G["patch_states"]()
    FIN["qsa_sha"] = sha_file(TREE / G["P60490_REL"])
    FIN["pp_shas"] = tree_shas(PP_RELS)
    FIN["cc_sha"] = sha_file(TREE / CC_REL) if CC_REL else None
    FIN["ok"] = (FIN["patch_states"] == {"p5": "not_applied", "ppack": "not_applied"} and FIN["qsa_sha"] == G["P60490_POST"]
                 and FIN["cc_sha"] == G["NB4_P5"].CC_PRE and FIN["pp_shas"] == {G["NB4_PK"].PP_REL: G["NB4_PK"].PP_PRE, G["NB4_PK"].MR_REL: G["NB4_PK"].MR_PRE})
    t = time.time()
    ok, why = G["sha_gate"](3 * 3600)
    FIN["sha"] = {"ok": ok, "why": why, "wait_sec": round(time.time() - t), "detail": G["R"]["setup"].get("sha256")}
except Exception as e:
    FIN.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1200:])
log("FIN", json.dumps({k: v for k, v in FIN.items() if k != "tb"}, default=str)[:900])
REP["checks"]["FIN"] = FIN
mg, pr = R1.get("merge") or {}, R1.get("prompt_render") or {}
emb = R1.get("embedded") or {}
optional = {k: {"plan": (plans.get(k) or {}).get("plan_ok"), "argparse": (R6.get("optional_ok") or {}).get(k),
                "diff": (DF.get("optional_ok") or {}).get(k), "env": (ENVR.get(k) or {}).get("ok"), "never": (NV.get("optional_ok") or {}).get(k)}
            for k in ("K4", "INT8")}
verdict({"payload_files_all_sha_ok": PAY.get("ok") is True,
         "payload_pins_equal_notebook": (R1.get("notebook_pins_vs_mount") or {}).get("hook_pin") is True
                                        and bool((R1.get("notebook_pins_vs_mount") or {}).get("p5_src_sha_vs_mount"))
                                        and all((R1.get("notebook_pins_vs_mount") or {}).get("p5_src_sha_vs_mount").values()),
         "notebook_loaded": "fatal" not in R1 and G is not None,
         "embedded_sha_ok": bool(emb) and all(emb.values()),
         "hook_modules_from_their_sources": bool(R1.get("module_sources_ok")) and all(R1["module_sources_ok"].values()),
         "p5_sources_resolvable_like_p5_prepare": R1.get("p5_prepare_src_bad") == [],
         "arms_valid_at_load": R1.get("arm_problems") == [] and not any((R1.get("arms_never_do") or {"x": ["?"]}).values()),
         "inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")),
         "merge_ok": mg.get("ok") is True,
         "prompt_hashes_ok": pr.get("hashes_ok") is True and pr.get("source") == "merlin",
         "merlin_install_ok": R3.get("ok") is True,
         "patch_60490_applied_verified": R4.get("ok") is True,
         "core_timer_entry_point_ok": CT.get("ok") is True,
         "qpack_S2_apply_verify_revert_ok": PL.get("ppack_S2_ok") is True,
         "qdefer_S3_apply_verify_revert_ok": PL.get("ppack_S3_ok") is True,
         "p5_patch_apply_verify_revert_ok": PL.get("p5_patch_ok") is True,
         "plans_T0_S_S2_S3_G_P_T0b_built": all((plans.get(k) or {}).get("plan_ok") is True for k in MANDATORY) and "fatal" not in PL,
         "host_phase_all_ok": all((plans.get(k) or {}).get("host_phase") == "ok" for k in MANDATORY),
         "cli_flags_all_ok": all(isinstance((plans.get(k) or {}).get("help"), dict) and plans[k]["help"].get("ran") is True
                                 and not plans[k]["help"].get("missing") for k in MANDATORY),
         "help_real_on_cpu": PL.get("help_real") is True,
         "argparse_fields_ok": R6.get("ok") is True,
         "base_plan_T0b_ok": DF.get("base_ok") is True,
         "every_plan_differs_from_base_by_exactly_its_change": DF.get("ok") is True,
         "notebook_one_change_vs_ref_ok": all(((plans.get(k) or {}).get("stub_facts") or {}).get("one_change_vs_ref") is True
                                              for k in MANDATORY if k != "T0"),
         "notebook_readback_patch_states_ok": all(bool(((plans.get(k) or {}).get("stub_facts") or {}).get("patch_keys"))
                                                  and all(((plans.get(k) or {}).get("stub_facts") or {}).get("patch_keys").values())
                                                  for k in MANDATORY),
         "arm_env_reaches_worker_process": all((ENVR.get(k) or {}).get("ok") is True for k in MANDATORY),
         "commit_rpc_methods_exist": all(((ENVR.get("T0") or {}).get("worker_methods") or {"x": False}).values())
                                     and (ENVR.get("T0") or {}).get("conflicts") == [],
         "never_do_absent_every_plan": NV.get("ok") is True,
         "tree_back_to_notebook_state": FIN.get("ok") is True,
         "checkpoint_sha256_ok": (FIN.get("sha") or {}).get("ok") is True,
         "page_cache_drop_ran": (R7.get("drop") or {}).get("available") is True},
        recorded={"optional_K4_INT8": optional, "p5_so": (plans.get("P") or {}).get("so_status"),
                  "p5_so_cpu_compile_probe": (plans.get("P") or {}).get("so_cpu_compile_probe"),
                  "p5_gate": (plans.get("P") or {}).get("p5_gate"), "p5_engine_init": (plans.get("P") or {}).get("p5_engine_init"),
                  "p5_so_for_plan": (plans.get("P") or {}).get("so_for_plan"),
                  "import_probes": {k: {"before": (plans.get(k) or {}).get("import_before"), "after": (plans.get(k) or {}).get("import_after")}
                                    for k in ("S2", "S3", "P")},
                  "core_timer_wrap_probe": CT.get("wrap_probe"), "help_text_len": PL.get("help_text_len"),
                  "argparse_platform": (R6.get("parsed") or {}).get("_platform"),
                  "stub_patch_keys": {k: ((plans.get(k) or {}).get("stub_facts") or {}).get("patch_keys") for k in plans},
                  "plan_errors": {k: p.get("error") or p.get("fatal") for k, p in plans.items() if p.get("error") or p.get("fatal")},
                  "diff_bad": {k: v.get("bad") for k, v in (DF.get("arms") or {}).items() if v.get("bad")},
                  "argparse_bad": {k: v for k, v in (R6.get("field_check") or {}).items() if v},
                  "env_bad": {k: v.get("bad") for k, v in ENVR.items() if isinstance(v, dict) and v.get("bad")},
                  "oracle_prompt_identity": R1.get("oracle_prompt_identity"), "host_ram_gate": (R7.get("host_ram_gate") or {}).get("why"),
                  "payload_unpinned_on_mount": PAY.get("on_mount_unpinned"), "plan_order": list(plans)},
        extra_line="@@KIT real OPTIONAL " + json.dumps({k: all(v is True for v in o.values()) for k, o in optional.items()}) + "@@")
'''

UNITS = r'''
R8 = {}
UT_EXPECT = json.loads(r"""__UTEXPECT__""")
UNITS_SHAS = json.loads(r"""__UNITSSHAS__""")
MERLIN_PRE = json.loads(r"""__MERLINPRE__""")
NB4_META_TXT = json.loads(r"""__NB4META__""")["text"]
def ut_parse(out):
    ran = re.findall(r"^Ran (\d+) tests? in", out, re.M)
    fin = [l.strip() for l in out.splitlines() if re.match(r"^(OK|FAILED)( \(.*\))?\s*$", l)]
    skips = re.findall(r"\.\.\. skipped '(.*)'\s*$", out, re.M)
    fails = [l[:300] for l in out.splitlines() if re.match(r"^(FAIL|ERROR): ", l)]
    return {"ran": int(ran[-1]) if ran else None, "status": fin[-1] if fin else None, "skips": skips[:60], "fails": fails[:60]}
try:
    F = unpack("__UNITSBLOB__")
    B = K / "experiments" / "E25" / "Experiment Notebooks" / "Build"      # the build-folder depth the tests' parents[] expect
    B.mkdir(parents=True)
    for rel, txt in F.items():
        p = (B.parent / rel) if rel == "__NBNAME__" else (B / rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8", newline="\n")
    R8["files_sha_ok"] = {rel: hashlib.sha256(txt.encode("utf-8")).hexdigest() == UNITS_SHAS.get(rel) for rel, txt in F.items()}
    nbp = B.parent / "__NBNAME__"
    R8["notebook_sha_ok"] = sha_file(nbp) == "__NBSHA__"
    cells = ["".join(c["source"]) for c in json.loads(nbp.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    (B / "nb4_code_export.py").write_text("\n\n".join(cells), encoding="utf-8", newline="\n")   # the export = the notebook's code cells
    R8["export_sha256"] = sha_file(B / "nb4_code_export.py")
    R8["export_sha_ok"] = R8["export_sha256"] == "__EXPORTSHA__"
    R8["payload"] = payload_check()
    pay = Path(R8["payload"]["dir"]) if R8["payload"].get("dir") not in (None, "None") else None
    pm = pay / "nb4_p5_engine.py" if pay is not None else None
    R8["p5_module_from_mount"] = pm is not None and pm.is_file() and sha_file(pm) == PAY_PINS["nb4_p5_engine.py"]
    if R8["p5_module_from_mount"]:
        shutil.copyfile(pm, B / "nb4_p5_engine.py")
    R8["merlin_src_sha_ok"] = {rel: sha_file(B / "merlin-src" / rel) == h for rel, h in MERLIN_PRE.items()}
    km = Path("D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb4-tick__foysalemonshanto/kernel-metadata.json")
    if not km.is_absolute():                  # Linux: the unit test's Windows path is relative to its cwd (= Build/)
        (B / km).parent.mkdir(parents=True, exist_ok=True)
        (B / km).write_text(NB4_META_TXT, encoding="utf-8", newline="\n")
    R8["kernel_metadata_at"] = str(B / km) if not km.is_absolute() else f"{km} (absolute on this OS: not written)"
    v = sh("versions", [sys.executable, "-c", "import sys, json, numpy, torch; print(json.dumps({'python': sys.version, 'numpy': numpy.__version__, 'torch': torch.__version__}))"], cwd=B, timeout=600)
    R8["versions"] = v["tail"].strip().splitlines()[0] if v["rc"] == 0 else v["tail"]
    R8["suites"] = {}
    runs = {"nb4_unit_tests.py": ([sys.executable, str(B / "nb4_unit_tests.py"), "-v"], {}),
            "nb4_timeline_tests.py": ([sys.executable, str(B / "nb4_timeline_tests.py"), "-v"], {"NB4_B1_TRACES": str(K / "no_b1_traces_on_kaggle")}),
            "nb4_p5_tests.py": ([sys.executable, str(B / "nb4_p5_tests.py"), "-v"], {"NB4_MERLIN_SRC": str(B / "merlin-src")}),
            "nb4_ppack_tests.py": ([sys.executable, str(B / "nb4_ppack_tests.py"), "--merlin-src", str(B / "merlin-src")], {})}
    for t, (args, env) in runs.items():
        r = sh(t[:-3], args, env, 1800, cwd=B)
        if t == "nb4_ppack_tests.py":
            js = None
            m = re.search(r"^\{\s*\n\s*\"checks\".*?^\}", r["_out"], re.M | re.S)
            try:
                js = json.loads(m.group(0)) if m else None
            except Exception:
                js = None
            fails = [l[:400] for l in r["_out"].splitlines() if l.startswith("FAIL ")]
            u = {"rc": r["rc"], "sec": r["sec"], "checks": (js or {}).get("checks"), "pass": (js or {}).get("pass"), "fail": (js or {}).get("fail"),
                 "all_pass_line": "ALL PASS" in r["_out"], "fails": fails[:60]}
            u["all_pass"] = r["rc"] == 0 and bool(js) and js.get("fail") == 0 and js.get("pass") == js.get("checks") and (js.get("checks") or 0) >= 150 \
                and u["all_pass_line"] and not fails
        else:
            p = ut_parse(r["_out"])
            n_want, sk_want = UT_EXPECT[t]
            u = dict(p, rc=r["rc"], sec=r["sec"], want_tests=n_want, want_skips=sk_want)
            u["skips_allowed"] = all(re.search(r"absent|not found", s) for s in p["skips"])
            u["all_pass"] = (r["rc"] == 0 and p["ran"] == n_want and str(p["status"]).startswith("OK") and not p["fails"]
                             and len(p["skips"]) == sk_want and u["skips_allowed"])
        if not u["all_pass"]:
            u["tail"] = r["tail"]
        R8["suites"][t] = u
        log(t, json.dumps(u)[:2500]); save()
except Exception as e:
    R8["fatal"] = f"{type(e).__name__}: {e}"[:800]; R8["tb"] = traceback.format_exc()[-1500:]
REP["checks"]["R8"] = R8
S8 = R8.get("suites") or {}
verdict({"files_sha_ok": bool(R8.get("files_sha_ok")) and all(R8["files_sha_ok"].values()), "notebook_sha_ok": R8.get("notebook_sha_ok") is True,
         "export_equals_build": R8.get("export_sha_ok") is True, "payload_files_all_sha_ok": (R8.get("payload") or {}).get("ok") is True,
         "p5_module_from_payload_mount": R8.get("p5_module_from_mount") is True,
         "merlin_originals_sha_ok": bool(R8.get("merlin_src_sha_ok")) and all(R8["merlin_src_sha_ok"].values()),
         **{t[:-3] + "_all_pass": (S8.get(t) or {}).get("all_pass") is True for t in __TESTS__}},
        recorded={"versions": R8.get("versions"),
                  **{t: {k: (S8.get(t) or {}).get(k) for k in ("ran", "status", "skips", "checks", "pass", "fail", "sec")} for t in __TESTS__}})
'''

KITS = {"real": {"body": REAL, "title": "Q27 E25 NB4 Kit Real", "owner": "foysalemonshanto", "models": MODELS, "datasets": DATASETS},
        "units": {"body": UNITS, "title": "Q27 E25 NB4 Kit Units", "owner": "foysalemonshanto", "models": [], "datasets": DATASETS}}


def kit_code(name):
    code = HEAD + KITS[name]["body"]
    for k, v in (("__EXPORTBLOB__", EXPORT_BLOB), ("__UNITSBLOB__", UNITS_BLOB), ("__PAYPINS__", json.dumps(PAY_PINS, sort_keys=True)),
                 ("__EXPECT__", json.dumps(EXPECT)), ("__MANDATORY__", json.dumps(MANDATORY)), ("__B3PIN__", str(B3_PIN)),
                 ("__UTEXPECT__", json.dumps(UT_EXPECT)), ("__UNITSSHAS__", json.dumps(UNITS_SHAS)), ("__MERLINPRE__", json.dumps(MERLIN_PRE)),
                 ("__NB4META__", json.dumps({"text": NB4_META_TXT})),("__TESTS__", json.dumps(TESTS)), ("__NBNAME__", NB_NAME),
                 ("__EXPORTSHA__", hashlib.sha256(EXPORT.encode("utf-8")).hexdigest()), ("__NBSHA__", NB_SHA), ("__NAME__", name)):
        code = code.replace(k, v)
    left = sorted(set(re.findall(r"__[A-Z0-9]{4,}__", code)) - {"__file__", "__name__", "__init__", "__wrapped__", "__main__"})
    assert not left, f"{name}: placeholders left {left}"
    compile(code, f"<kit {name}>", "exec")
    return code


def build(name):
    spec = KITS[name]
    code = kit_code(name)
    slug = f"q27-e25-nb4-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__{spec['owner']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"{spec['owner']}/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": spec["datasets"], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| notebook", NB_SHA[:8], flush=True)
    return out


def selfcheck():
    """The units kit code on this machine: KIT_* paths in the scratchpad, the local payload folder as the dataset mount."""
    import shutil
    base = Path(os.environ.get("NB4_KIT_SELFCHECK_DIR", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/nb4kit_selfcheck"))
    shutil.rmtree(base, ignore_errors=True)
    inp = base / "input" / "datasets" / "foysalemonshanto" / "e25-nb4-payload"
    shutil.copytree(PAYLOAD_DIR, inp)
    (base / "code").mkdir(parents=True)
    (base / "code" / "units_kit.py").write_text(kit_code("units"), encoding="utf-8", newline="\n")
    env = dict(os.environ, KIT_INPUT=str(base / "input"), KIT_TMP=str(base / "tmp"), KIT_WORK=str(base / "work"))
    cp = subprocess.run([sys.executable, str(base / "code" / "units_kit.py")], capture_output=True, text=True, env=env, cwd=str(base / "code"), timeout=1800)
    print(cp.stdout[-4000:], cp.stderr[-2000:])
    return cp.returncode


if __name__ == "__main__":
    print("payload pins", len(PAY_PINS), "| tests", TESTS, "| unittest expectations", UT_EXPECT, flush=True)
    if "--selfcheck" in sys.argv:
        sys.exit(selfcheck())
    for n in KITS:
        build(n)
