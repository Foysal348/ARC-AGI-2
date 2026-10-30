"""E24 local end-to-end runner (WP5, rev 2, 2026-10-30; stdlib only). BUILD_SPEC §6 + the rev-2 WP5 overrides.

1. builds the three notebooks with the WP4 builder (in place, or in a mirror of E24 with --mirror-build);
2. writes a mock /kaggle/input (mock_inputs.py) unless --input points at a real one (the CPU kit passes /kaggle/input);
3. runs every variant's code cells, concatenated into one .py, in a FRESH python3.12 subprocess with E24_FAKE_SERVER=1,
   E24_FAKE_DIR, E24_INPUT, E24_OUT, E24_TMP (fresh, created first: the E23 kit v1 bug), HOME=<temp> (default-cache checks
   see a fake home), E24_TIME_SCALE (default 0.02), E24_BUDGET_SEC (scaled; plus slack for the fakes' unscaled process
   overheads), E24_FORCE_CONDITIONAL=1 (both branches of every conditional arm), E14_LOADPATH=0 and E24_FAKE_SGLANG_SITE (a
   temp copy of the SGLang 0.5.21 source, so the patches apply and revert for real);
4. validates e24_results.json, the report and the launch records (P-checks below), runs the negative scenarios (N-checks),
   writes a JSON + markdown summary. Exit code 0 only if every check passes.

E24_FAKE_DIR is a per-run wrapper dir: its fake_sglang_server.py / fake_vllm_server.py log every invocation (argv + the
SGLANG_/E24_/NCCL_/cache env it received, the arm id) to <run>/launches/, apply the scenario's per-arm env rules, then
os.execv() the real WP2 fake (same pid, same stdout/stderr, so multiprocessing children and the server log are unchanged).
That is how faults reach ONE arm without any per-arm knob in the notebook, and how the env contract (DESIGN §2 fixed env,
§3.3 per-arm cache roots, the TEST/SIMULATE/INJECT whitelist) is checked on what the servers really received.
Port isolation (default): the notebook's one `SGL_PORT, VLLM_PORT = 30000, 8200` line becomes two free ports per run, so a
run never talks to another run's server (2026-10-30: a parallel local run by another agent on 30000 hijacked a scenario).

usage: /opt/homebrew/bin/python3.12 run_local_e24.py [--variants SMOKE,FULL_A,FULL_B] [--negatives all|none|a,b]
         [--time-scale 0.02] [--work DIR] [--keep] [--mirror-build | --no-build | --nb-dir DIR] [--input DIR] [--no-site]
         [--timeout-min 90] [--budget-slack 4] [--allow-budget-skips] [--list]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
E24 = HERE.parents[1]
NBDIR = E24 / "Experiment Notebooks"
BUILDER = NBDIR / "Build" / "build_e24_notebook.py"
NB_FILE = "Q27-XHIGH-E24_SGLANG_SPEED_{}.ipynb"
VARIANTS = {"SMOKE": ("SMOKE", "ALL", 210), "FULL_A": ("FULL", "A", 480), "FULL_B": ("FULL", "B", 690)}   # DESIGN §1
PY = os.environ.get("E24_PY") or next((p for p in ("/opt/homebrew/bin/python3.12",) if Path(p).exists()), sys.executable)
NB_MAX_BYTES = 400_000            # DESIGN §7 "< 400 kB" (Kaggle: 407 kB ran, 444 kB never started); decimal = the strict reading
OUT_MAX_FILES = 300               # DESIGN §3.6
# the unpacked sglang-0.5.21 wheel (the WP3 tests use the same copy); E24_SGLANG_SRC overrides (the CPU kit: the installed one)
SGL_SRC_DEFAULT = ("/private/tmp/claude-501/-Users-md-hamidhosen-Documents-Kaggle-ARC-AGI-2/e21dd5ac-4964-44ae-8fe0-b7d1156db571"
                   "/scratchpad/e24/downloads/sglang_source_0521/src/sglang")
# DESIGN §4-§5 arm ids per variant that a positive FAKE run with E24_FORCE_CONDITIONAL=1 must execute (both branches of
# every conditional; "x__" = at least one arm with that prefix). "If time remains" arms (s1_c_r2, G-L, A/A rounds) are not
# required here; the notebook's own R["plan"] is checked separately (P6b).
DESIGN_ARMS = {
    "SMOKE": ["s1_a", "s1_b", "s1_d6", "s1_d6fr", "s1_d4", "s1_c", "s6_unit", "s6_ffu", "s6_ffp", "s2_w6s20", "s2_w4s20",
              "s3_cta4"],
    "FULL_A": ["s1_a", "s1_b", "s1_d6", "s1_d6fr", "s1_d4", "s1_c", "s1_c_warm", "s1_c_nocache", "s1_c32", "s1_d6_r2",
               "s6_unit", "s6_ffu", "s6_ffp", "s6_np", "s6_ffud", "s6_ffpd", "s6_gu", "s6_gp", "s6_gu_df", "s6_gp_df",
               "s6_sent_u", "s6_sent_p", "s6_nan_u", "s6_nan_p", "s6_combined"],
    "FULL_B": ["s2_w6s20", "s2_w6s20_aa", "s2_w4s20", "s2_w6s16", "s2_w4s16", "s2_w5s20", "s2_w7s20", "s2_best_long__",
               "s3_ctrl", "s3_cta2", "s3_cta4", "s3_cta8", "s3_ring", "s3_tree", "s3_simple", "s3_ll", "s3_nthreads512",
               "s3_conf__", "s3_ctrl_rerun", "s4_m84", "s4_m68", "s4_mem080", "s4_mem086", "s4_m88", "s4_m92",
               "s5_c2048", "s5_c4096", "s5_c8192", "s5_pdi1", "s5_pdi4"]}
# positive FAKE runs: NO_GO_* / INVALID_* only where the DESIGN expects a negative: s4_mem086 (an OOM is expected-negative,
# §4.4) and the chunk-size arms of section 5 (§4.5: in 0.5.21 prefill pre-empts decode, so a cold 90k injection starves the
# residents whatever the chunk size; that measured negative is what the section quantifies; the PDI arms may not starve)
_STARVE = re.compile(r"^NO_GO_STARVATION$")
EXPECTED_NEGATIVE = {"s4_mem086": re.compile(r"^NO_GO_(OOM|LAUNCH)"), "s4_lowcap": re.compile(r"^NO_GO_"),
                     "s5_c2048": _STARVE, "s5_c4096": _STARVE, "s5_c8192": _STARVE, "s5_mixed": _STARVE}
# section-1 pairs (DESIGN §4.1; short names = arm id minus "s1_"), present when both arms are in the variant
S1_PAIRS = [("a", "b"), ("b", "c32"), ("c32", "c"), ("c", "d6"), ("c", "d4"), ("c", "d6fr"), ("c", "c_warm"),
            ("c", "c_nocache"), ("b", "c")]
ARM_ID = re.compile(r"^(s[1-6]_[a-z0-9_]+|smoke_[a-z0-9_]+|aa_[a-z0-9_]+)$")
TERMINAL = re.compile(r"^(GO|PASS|FAIL(_[A-Z0-9_]+)?|NO_GO(_[A-Z0-9_]+)?|INVALID_[A-Z0-9_]+|INCONCLUSIVE(_[A-Z0-9_]+)?|"
                      r"SKIPPED_[A-Z0-9_]+|ERROR_[A-Za-z0-9_]+|MEASURED|NOT_MEASURED|NOT_APPLICABLE(_[A-Z0-9_]+)?)$")
DEFAULT_STATUSES = {"", "NOT_RUN", "DONE", "RUNNING", "PENDING", "STARTED", "OK"}
CACHE_STATUSES = {"ISOLATED", "INVALID_NOT_LOGGED", "INVALID_PATH_MISMATCH", "INVALID_DEFAULT_DIR_WRITTEN", "INVALID_EMPTY_CACHE"}
CMP_STATUS = re.compile(r"^(EXACT(_TO_\d+|_TO_N)?|NEAR_TIE|BOUNDARY|FAIL_HIGH_MARGIN|FAIL_STATE_FLIP|MEASURED|"
                        r"INCONCLUSIVE(_[A-Z0-9_]+)?|NOT_APPLICABLE(_[A-Z0-9_]+)?|NOT_EXERCISED|INVALID(_[A-Z0-9_]+)?|"
                        r"SKIPPED(_[A-Z0-9_]+)?)$")
CACHE_VARS = ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
              "FLASHINFER_WORKSPACE_BASE", "CUDA_CACHE_PATH", "TILELANG_CACHE_DIR", "TORCH_EXTENSIONS_DIR")   # DESIGN §3.3
GUARDED_ENV = ("SGLANG_TEST_", "SGLANG_SIMULATE_", "E24_INJECT_")                                       # DESIGN §2
GUARDED_OK = {"SGLANG_SIMULATE_": re.compile(r"^s1_d6fr"), "SGLANG_TEST_": re.compile(r"^s6_ff|^s6_np"),
              "E24_INJECT_": re.compile(r"^s6_(sent|nan)_")}
CLEAN_ENV_PREFIX = ("E24_", "FAKE_", "Q27X_", "KAGGLE_", "E14_", "E23_", "SGLANG_", "NCCL_")
# happy-path fake env (WP4/WP2 integration 2026-10-30): the vLLM fake models a 500 s cold start x the time scale; with the
# real cost of spawning its 5 processes that sat at or over the scaled 600 s gate, so 3 real seconds make the GO path run.
# The startup_gate scenario sets 1.3 x the scaled gate instead and expects NO_GO_STARTUP_600 with diagnostics kept.
FAKE_ENV = {"FAKE_VLLM_STARTUP_SEC": "3"}

WRAPPER = r'''# E24 local-runner wrapper ({engine}, scenario {scenario}): logs this invocation, applies the scenario's per-arm env
# rules, then execv()s the real fake: same pid, same stdout/stderr, so multiprocessing children and server logs are unchanged.
import json, os, re, sys, time
REAL, ENGINE, RULES, LOGDIR = {real!r}, {engine!r}, {rules!r}, {logdir!r}
ARM_ID = re.compile(r"^(s[1-6]_[a-z0-9_]+|smoke_[a-z0-9_]+|aa_[a-z0-9_]+)$")
argv = sys.argv[1:]
fc = bool(argv) and argv[0] == "--e24-flag-check"
checked = json.loads(argv[1]) if fc and len(argv) > 1 else list(argv)
env = os.environ


def arm_of():
    if env.get("E24_ARM"):
        return env["E24_ARM"], "E24_ARM"
    for key in ("E24_HOOK_DIR", "E24_RAW_DIR") + tuple(sorted(k for k in env if k.endswith(("_CACHE_DIR", "_CACHE_PATH",
                                                                                         "_CACHE_ROOT", "_BASE")))):
        for part in reversed((env.get(key) or "").split("/")):
            if ARM_ID.match(part):
                return part, key
    return "", None


def has(seq, pair):
    if pair[0] not in seq:
        return False
    if len(pair) == 1:
        return True
    i = seq.index(pair[0])
    return i + 1 < len(seq) and str(seq[i + 1]) == str(pair[1])


arm, src = arm_of()
applied = []
for n, r in enumerate(RULES):
    m, ok = r.get("match") or {{}}, True
    ok = ok and m.get("engine", ENGINE) == ENGINE
    ok = ok and (m.get("flag_check") is None or m["flag_check"] == fc)
    ok = ok and (not m.get("arm_re") or bool(re.search(m["arm_re"], arm)))
    ok = ok and all(has(checked, p) for p in m.get("argv_has", ())) and not any(has(checked, p) for p in m.get("argv_lacks", ()))
    ok = ok and all(k in env and (v is None or str(v) in env[k]) for k, v in (m.get("env_has") or {{}}).items())
    ok = ok and not any(k in env for k in m.get("env_lacks", ()))
    if ok:
        applied.append(n)
        env.update({{k: str(v) for k, v in (r.get("env") or {{}}).items()}})
        if fc and r.get("add_flag"):
            checked = checked + [r["add_flag"]]
            argv = ["--e24-flag-check", json.dumps(checked)] + argv[2:]
keep = {{k: v for k, v in env.items() if k.startswith(("SGLANG_", "E24_", "NCCL_", "FAKE_", "HF_", "TRANSFORMERS_", "VLLM_",
        "TRITON_", "TORCHINDUCTOR_", "TORCH_EXTENSIONS", "CUDA_CACHE", "FLASHINFER_", "TILELANG_", "PYTHONPATH", "HOME"))}}
rec = {{"t": time.time(), "pid": os.getpid(), "ppid": os.getppid(), "engine": ENGINE, "flag_check": fc, "arm": arm,
       "arm_src": src, "argv": checked, "env": keep, "rules": applied}}
try:
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "%.6f_%d_%s.json" % (time.time(), os.getpid(), "fc" if fc else "launch")), "w") as f:
        json.dump(rec, f)
except Exception:
    pass
os.execv(sys.executable, [sys.executable, REAL] + argv)
'''


# ------------------------------------------------------------------------------------------------ small helpers
def now():
    return time.strftime("%H:%M:%S")


def say(*a):
    print(f"[{now()}]", *a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nb_code(nb_path):
    """All code cells of a notebook, in order, joined as the E23 kit does."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    return "".join("".join(c["source"]) + "\n\n" for c in nb["cells"] if c["cell_type"] == "code")


BAKE_RE = {k: re.compile(r'(os\.environ\.get\(\s*"Q27X_E24_%s"\s*,\s*")([A-Z_]+)("\s*\))' % k) for k in ("MODE", "PART")}


def bake(code, variant):
    """The code of `variant` derived from any variant's code by replacing the two baked defaults (E23 kit `bake`)."""
    m, p, _ = VARIANTS[variant]
    for k, v in (("MODE", m), ("PART", p)):
        code, n = BAKE_RE[k].subn(lambda g: g.group(1) + v + g.group(3), code, count=1)
        if n != 1:
            raise ValueError(f"baked Q27X_E24_{k} default not found exactly once")
    return code


def walk(obj, path=""):
    """Every dict in a JSON tree, with its path."""
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def arms_of(R):
    """{arm id: record} over every section (arm ids are unique across sections)."""
    out = {}
    for sec, st in (R.get("sections") or {}).items():
        if isinstance(st, dict):
            for a, rec in st.items():
                if isinstance(rec, dict) and "status" in rec:
                    out[a] = dict(rec, _section=str(sec))
    return out


def order_arms(R):
    return [str(x).split("/")[-1] for x in (R.get("order") or [])]


def has_arm(A, a):
    return any(x.startswith(a) for x in A) if a.endswith("__") else a in A


def plan_entries(R):
    """R["plan"] = [[section, arm, est_min], ...] (WP4): (required ids, conditional ids). "a|b" = two alternatives,
    a trailing "?" = conditional; "s2_best_long" / "s3_conf__best*" become prefixes."""
    req, cond = [], []
    for e in R.get("plan") or []:
        a = str(e[1] if isinstance(e, (list, tuple)) and len(e) > 1 else e)
        c = a.endswith("?") or "|" in a
        for x in a.rstrip("?").split("|"):
            x = "s2_best_long__" if x == "s2_best_long" else "s3_conf__" if x.startswith("s3_conf__best") else x
            (cond if c else req).append(x)
    return req, cond


def short(a):
    return a[3:] if a.startswith("s1_") else a


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _proc_table():
    """[(pid, command line + environment)] of this user's processes: /proc on Linux, `ps -E` on macOS."""
    rows = []
    if Path("/proc/self/environ").exists():
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    blob = (p / "cmdline").read_bytes() + b"\0" + (p / "environ").read_bytes()
                    rows.append((int(p.name), blob.replace(b"\0", b" ").decode(errors="replace")))
                except Exception:
                    pass
        return rows
    try:
        txt = subprocess.run(["ps", "-axwwE", "-o", "pid=,command="], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return rows
    for line in txt.splitlines():
        pid, _, rest = line.strip().partition(" ")
        if pid.isdigit():
            rows.append((int(pid), rest))
    return rows


def kill_tree_matching(markers):
    """SIGKILL every process whose command line or environment contains one of `markers` (leftover fakes of one run: the
    wrapper execv()s the real fake, so the run dir survives only in the environment, E24_FAKE_DIR=<run>/fake_dir)."""
    killed = []
    for pid, blob in _proc_table():
        if pid not in (os.getpid(), os.getppid()) and any(m and m in blob for m in markers):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except Exception:
                pass
    return killed


# ------------------------------------------------------------------------------------------------ build
def build_notebooks(mode, work, nb_dir=None):
    """mode: "inplace" (the repo builder writes the repo notebooks), "mirror" (a copy of E24 under work/, so the repo is
    untouched), "none" (use the notebooks as they are, in nb_dir or the repo). Returns {ok, nbdir, notebooks, ...}."""
    res = {"mode": mode, "ok": False}
    nbdir, builder = Path(nb_dir).resolve() if nb_dir else NBDIR, BUILDER
    if mode == "mirror":
        root = Path(work).resolve() / "mirror" / E24.parent.name / E24.name
        for sub in ("src", "Base Engine/src_e14_modules", "Experiment Notebooks/Build"):
            if (E24 / sub).is_dir():
                shutil.copytree(E24 / sub, root / sub, ignore=shutil.ignore_patterns("__pycache__", "cpu_kernel_e24"),
                                dirs_exist_ok=True)
        for f in ("README.md", "DESIGN.md"):
            if (E24 / f).is_file():
                shutil.copy2(E24 / f, root / f)
        for f in E24.parent.glob("Experiment 21 - */src/e21_flagcheck.py"):     # the kit builder's vLLM parser check
            dst = root.parent / f.relative_to(E24.parent)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        nbdir, builder = root / "Experiment Notebooks", root / "Experiment Notebooks" / "Build" / BUILDER.name
    if mode in ("inplace", "mirror"):
        if not builder.is_file():
            res["error"] = f"builder missing: {builder}"
            return res
        t = time.time()
        r = subprocess.run([PY, str(builder)], cwd=str(builder.parent), capture_output=True, text=True, timeout=900)
        res.update(rc=r.returncode, sec=round(time.time() - t, 1), tail=(r.stdout + r.stderr)[-3000:])
        if r.returncode != 0:
            res["error"] = "builder failed"
            return res
    res["nbdir"] = str(nbdir)
    res["notebooks"] = {}
    for v in VARIANTS:
        p = nbdir / NB_FILE.format(v)
        if p.is_file():
            res["notebooks"][v] = {"path": str(p), "bytes": p.stat().st_size, "sha8": sha256(p)[:8]}
    res["plan_argvs"] = str(nbdir / "Build" / "plan_argvs_e24.json") if (nbdir / "Build" / "plan_argvs_e24.json").is_file() else None
    res["ok"] = len(res["notebooks"]) == len(VARIANTS)
    if not res["ok"]:
        res["error"] = f"notebooks missing: {sorted(set(VARIANTS) - set(res['notebooks']))}"
    return res


def build_checks(b):
    """B-checks on the build: three notebooks, each < 400 kB, identical code except the two baked lines, plan argvs."""
    out = [("B1 builder ran and wrote the three notebooks", b.get("ok"), b.get("error") or b.get("tail", "")[-800:])]
    if not b.get("ok"):
        return out, {}
    sizes = {v: n["bytes"] for v, n in b["notebooks"].items()}
    out.append(("B2 every notebook < 400 kB (400,000 B)", all(s < NB_MAX_BYTES for s in sizes.values()), sizes))
    codes = {v: nb_code(n["path"]) for v, n in b["notebooks"].items()}
    try:
        same = {v: bake(codes["SMOKE"], v) == codes[v] for v in codes}
        out.append(("B3 code cells identical across variants except the baked MODE/PART defaults", all(same.values()), same))
    except Exception as e:
        out.append(("B3 code cells identical across variants except the baked MODE/PART defaults", False, repr(e)))
    for v, c in codes.items():
        try:
            compile(c, f"<{v}>", "exec")
            ok, info = True, ""
        except SyntaxError as e:
            ok, info = False, repr(e)
        out.append((f"B4 {v} code compiles", ok, info))
    pa = b.get("plan_argvs")
    out.append(("B5 plan_argvs_e24.json written", bool(pa), pa))
    return out, codes


# ------------------------------------------------------------------------------------------------ one notebook run
def sglang_source():
    for c in (os.environ.get("E24_SGLANG_SRC"), SGL_SRC_DEFAULT):
        if c and (Path(c) / "srt" / "server_args.py").is_file():
            return Path(c)
    return None


def load_module(run, name):
    """A module exactly as the notebook wrote it into its TMP (WORK/mods/<name>.py)."""
    import importlib.util
    for f in sorted(Path(run["tmp"]).rglob(f"{name}.py")):
        spec = importlib.util.spec_from_file_location(f"{name}_{abs(hash(str(f)))}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


def write_wrappers(d, scenario, rules, fake_dir):
    """The per-run E24_FAKE_DIR (see the module docstring)."""
    d.mkdir(parents=True, exist_ok=True)
    for eng, f in (("sglang", "fake_sglang_server.py"), ("vllm", "fake_vllm_server.py")):
        (d / f).write_text(WRAPPER.format(engine=eng, scenario=scenario, real=str(Path(fake_dir) / f),
                                          rules=[r for r in rules if (r.get("match") or {}).get("engine", eng) == eng],
                                          logdir=str(d.parent / "launches")), encoding="utf-8")
    for p in Path(fake_dir).glob("*.py"):          # anything else the notebook might read from E24_FAKE_DIR
        if p.name not in ("fake_sglang_server.py", "fake_vllm_server.py") and not (d / p.name).exists():
            shutil.copy2(p, d / p.name)


PORT_LINE = re.compile(r"(?m)^(\s*)SGL_PORT, VLLM_PORT = (\d+), (\d+)\s*$")


def isolate_ports(code):
    """The notebook's fixed server ports (30000 / 8200) -> two free ports for this run, so a run never talks to another
    run's server (parallel runs, another agent's local run, a leftover). Only that one constant line changes; the argv,
    the BASE check (it uses SGL_PORT) and every measurement are otherwise the notebook's own. Returns (code, info)."""
    if len(PORT_LINE.findall(code)) != 1:
        return code, {"isolated": False, "why": "SGL_PORT, VLLM_PORT line not found exactly once: the notebook's own ports"}
    p1 = free_port()
    p2 = free_port()
    while p2 == p1:
        p2 = free_port()
    return PORT_LINE.sub(lambda g: f"{g.group(1)}SGL_PORT, VLLM_PORT = {p1}, {p2}", code, count=1), \
        {"isolated": True, "sglang": p1, "vllm": p2}


def run_notebook(name, code, work, input_root, variant, scale=0.02, rules=(), extra_env=None, site=True,
                 timeout_min=90, budget_slack=4.0, fake_dir=HERE, force_conditional=True, ports=True, unset=()):
    """Run one notebook (all code cells as one .py) in a fresh subprocess. Returns a dict with the run facts."""
    d = Path(work).resolve() / name
    if d.exists():
        shutil.rmtree(d)
    out, tmp, home = d / "out", d / "tmp", d / "home"
    for p in (out, tmp, home):
        p.mkdir(parents=True)
    code, port_info = isolate_ports(code) if ports else (code, {"isolated": False, "why": "disabled"})
    (d / "nb.py").write_text(code, encoding="utf-8")
    write_wrappers(d / "fake_dir", name, list(rules), fake_dir)
    m, p, bmin = VARIANTS[variant]
    env = {k: v for k, v in os.environ.items() if not k.startswith(CLEAN_ENV_PREFIX)}
    env.update(E24_FAKE_SERVER="1", E24_FAKE_DIR=str(d / "fake_dir"), E24_INPUT=str(input_root), E24_OUT=str(out),
               E24_TMP=str(tmp), HOME=str(home), E24_TIME_SCALE=str(scale), E14_LOADPATH="0", PYTHONUNBUFFERED="1",
               E24_BUDGET_SEC=str(int(bmin * 60 * scale * budget_slack + 600)), Q27X_E24_MODE=m, Q27X_E24_PART=p)
    if force_conditional:
        env["E24_FORCE_CONDITIONAL"] = "1"
    site_dir = None
    if site:
        src = sglang_source()
        if src is not None:
            site_dir = d / "site" / "sglang"
            shutil.copytree(src, site_dir, ignore=shutil.ignore_patterns("__pycache__"))
            env["E24_FAKE_SGLANG_SITE"] = str(site_dir)
    env.update(FAKE_ENV)
    env.update({k: str(v) for k, v in (extra_env or {}).items()})
    for k in unset or ():
        env.pop(k, None)
    t0 = time.time()
    say(f"run {name} ({variant}, scale {scale}, budget {env['E24_BUDGET_SEC']} s, site {'yes' if site_dir else 'no'}, "
        f"ports {port_info.get('sglang', 'notebook')}/{port_info.get('vllm', 'notebook')})")
    with open(d / "nb.log", "w", encoding="utf-8") as lf:
        pr = subprocess.Popen([PY, str(d / "nb.py")], cwd=str(d), stdout=lf, stderr=subprocess.STDOUT, env=env,
                              start_new_session=True)
        try:
            rc = pr.wait(timeout=timeout_min * 60)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out, rc = True, -9
            try:
                os.killpg(pr.pid, signal.SIGKILL)
            except Exception:
                pass
    killed = kill_tree_matching([f"E24_FAKE_DIR={d / 'fake_dir'}"])
    txt = (d / "nb.log").read_text(encoding="utf-8", errors="replace")
    rj = out / "e24_results.json"
    try:
        R = json.loads(rj.read_text(encoding="utf-8")) if rj.is_file() else {}
    except Exception as e:
        R = {"_parse_error": repr(e)}
    launches = []
    for f in sorted((d / "launches").glob("*.json")):
        try:
            launches.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    res = {"name": name, "variant": variant, "dir": str(d), "rc": rc, "timed_out": timed_out, "sec": round(time.time() - t0, 1),
           "tracebacks": txt.count("Traceback (most recent call last)"), "tail": txt[-3000:], "R": R, "launches": launches,
           "out": str(out), "tmp": str(tmp), "home": str(home), "site": str(site_dir) if site_dir else None,
           "killed_leftovers": killed, "ports": port_info,
           "env": {k: env[k] for k in env if k.startswith(("E24_", "Q27X_", "E14_", "FAKE_"))}}
    if site_dir is not None:          # the patch state the notebook left behind (its own patcher), then drop the ~100 MB copy
        try:
            P = load_module(res, "e24_sglang_patches")
            chk = P.check(str(site_dir)) if P else None
            res["site_end"] = {"names": chk.get("names"), "files_verified": chk.get("files_verified")} if chk else "no patcher"
        except Exception as e:
            res["site_end"] = repr(e)
        shutil.rmtree(d / "site", ignore_errors=True)
    (d / "run.json").write_text(json.dumps({k: res.get(k) for k in ("name", "variant", "rc", "timed_out", "sec", "killed_leftovers",
                                                                   "site", "site_end", "ports", "env")}, default=str), encoding="utf-8")
    say(f"  {name}: rc {rc}{' TIMEOUT' if timed_out else ''} in {res['sec']} s, {len(arms_of(R))} arms, "
        f"{len([l for l in launches if not l['flag_check']])} launches, tracebacks {res['tracebacks']}")
    return res


def load_run(d):
    """A finished run dir back as run_notebook's result (run.json + results + launches + log), to re-validate it."""
    d = Path(d)
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    rj = d / "out" / "e24_results.json"
    R = json.loads(rj.read_text(encoding="utf-8")) if rj.is_file() else {}
    txt = (d / "nb.log").read_text(encoding="utf-8", errors="replace") if (d / "nb.log").is_file() else ""
    launches = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((d / "launches").glob("*.json"))]
    return dict(meta, dir=str(d), R=R, launches=launches, tail=txt[-3000:], tracebacks=txt.count("Traceback (most recent call last)"),
                out=str(d / "out"), tmp=str(d / "tmp"), home=str(d / "home"))


# ------------------------------------------------------------------------------------------------ positive checks
def _argv_val(argv, flag):
    if flag in argv:
        i = argv.index(flag)
        vals = []
        for x in argv[i + 1:]:
            if str(x).startswith("--"):
                break
            vals.append(str(x))
        return vals
    return None


def flag_records(rec):
    return [(p, d) for p, d in walk(rec) if {"ran", "ok"} <= set(d) and ("method" in d or "unknown" in d)]


def cache_verdicts(rec):
    return [(p, d) for p, d in walk(rec) if d.get("status") in CACHE_STATUSES and "cache" in p.lower() + " ".join(d)]


def comparisons(R):
    """{(ref_short, cand_short): record} from R["comparisons"]["1"]["pairs"] (WP4: keys "s1_c~s1_d6"); a flat
    R["comparisons"]["1"] of pair records is accepted too (keys parsed with the known arm names)."""
    c1 = ((R.get("comparisons") or {}).get("1") or (R.get("comparisons") or {}).get(1) or {})
    if isinstance(c1, dict) and isinstance(c1.get("pairs"), dict):
        c1 = c1["pairs"]
    known = sorted({short(a) for a in arms_of(R)} | {x for p in S1_PAIRS for x in p}, key=len, reverse=True)
    out = {}
    for k, v in (c1.items() if isinstance(c1, dict) else []):
        if not isinstance(v, dict):
            continue
        ref, cand = v.get("ref"), v.get("cand")
        if isinstance(ref, str) and isinstance(cand, str):
            out[(short(ref), short(cand))] = v
            continue
        kk = re.sub(r"s1_", "", str(k))
        parts = [x for x in re.split(r"~|__|\|\||\||↔|<->|->|:|,|\s+vs\s+|\s+", kk) if x]
        if len(parts) == 2:
            out[(parts[0], parts[1])] = v
            continue
        for a in known:                                     # "c_d6"-style keys: split where both sides are known names
            if kk.startswith(a + "_") and kk[len(a) + 1:] in known:
                out[(a, kk[len(a) + 1:])] = v
                break
    return out


def env_checks(run):
    """E-checks on what the servers received (wrapper launch records)."""
    L = [x for x in run["launches"] if not x["flag_check"]]
    sg = [x for x in L if x["engine"] == "sglang"]
    out = [("E1 servers were launched and every launch has an arm id (E24_ARM / hook / cache path)",
            bool(L) and all(x["arm"] for x in L), sorted({x["argv"][0] if x["argv"] else "?" for x in L if not x["arm"]})[:5]
            or ("no launch" if not L else ""))]
    bad = [(x["arm"], k, x["env"].get(k)) for x in sg for k, v in (("SGLANG_LOGPROB_CHUNK_SIZE", "128"),
                                                                    ("SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK", "0")) if x["env"].get(k) != v]
    out.append(("E2 every SGLang launch: SGLANG_LOGPROB_CHUNK_SIZE=128 and SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0", not bad and bool(sg),
                bad[:8] or ("no SGLang launch" if not sg else "")))
    bad = [(x["arm"], k) for x in L for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") if x["env"].get(k) != "1"]
    out.append(("E3 every launch offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)", not bad, bad[:8]))
    bad = []
    for x in L:
        for k in x["env"]:
            for pre in GUARDED_ENV:
                if k.startswith(pre) and not GUARDED_OK[pre].search(x["arm"] or ""):
                    bad.append((x["arm"], k))
    out.append(("E4 SGLANG_TEST_* / SGLANG_SIMULATE_* / E24_INJECT_* only in the arms that set them", not bad, bad[:8]))
    bad = []
    for x in L:
        if not (x["arm"] or "").startswith("s1_"):
            continue
        need = CACHE_VARS if x["engine"] == "sglang" else ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT")
        miss = [k for k in need if not x["env"].get(k)]
        base = re.sub(r"_(warm|nocache)$", "", x["arm"])          # s1_c_warm / s1_c_nocache use s1_c's root by design
        wrong = [k for k in CACHE_VARS if x["env"].get(k) and f"/{base}/" not in x["env"][k] + "/"
                 and f"/{x['arm']}/" not in x["env"][k] + "/"]
        if miss or wrong:
            bad.append((x["arm"], "missing", miss, "outside the arm root", wrong))
    out.append(("E5 section-1 launches: every DESIGN §3.3 cache var set, under the arm's own root", not bad, bad[:6]))
    bad = []
    for x in sg:
        a = [str(t) for t in x["argv"]]
        mr, gb, mm = _argv_val(a, "--max-running-requests"), _argv_val(a, "--cuda-graph-bs-decode"), _argv_val(a, "--max-mamba-cache-size")
        if "--random-seed" not in a:
            bad.append((x["arm"], "no --random-seed"))
        if mr and gb and "--disable-decode-cuda-graph" not in a and max(int(g) for g in gb) < int(mr[0]):
            bad.append((x["arm"], f"graphs 1..{max(int(g) for g in gb)} < running {mr[0]}"))
        strat = _argv_val(a, "--mamba-radix-cache-strategy")
        if mr and mm and strat and strat[0] == "extra_buffer_lazy" and int(mm[0]) < 4 * int(mr[0]):
            bad.append((x["arm"], f"mamba {mm[0]} < 4 x running {mr[0]}"))
    out.append(("E6 SGLang argv invariants: --random-seed; decode graphs cover max-running; lazy Mamba slots >= 4/req",
                not bad, bad[:8]))
    return out


def positive_checks(run, allow_budget_skips=False, require_site=True):
    """P-checks (BUILD_SPEC §6 + rev-2 WP5): one positive FAKE run of one variant."""
    R, v = run["R"], run["variant"]
    m, p, _ = VARIANTS[v]
    A = arms_of(R)
    out = [("P1 notebook exit code 0, no timeout", run["rc"] == 0 and not run["timed_out"], f"rc {run['rc']} {run['tail'][-1200:]}"),
           ("P2 no traceback in the notebook output", run["tracebacks"] == 0, run["tail"][-1500:] if run["tracebacks"] else "")]
    need = ("mode", "part", "rev", "budget_min", "start_utc", "inputs", "setup", "sections", "comparisons", "errors", "order", "shas")
    out.append(("P3 e24_results.json has the BUILD_SPEC §5 keys", bool(R) and all(k in R for k in need),
                [k for k in need if k not in R] or R.get("_parse_error", "")))
    out.append(("P4 mode/part as baked for the variant", (R.get("mode"), R.get("part")) == (m, p), (R.get("mode"), R.get("part"))))
    stepped = list(dict.fromkeys(order_arms(R)))
    bad = {a: (A.get(a) or {}).get("status") for a in stepped if not TERMINAL.match(str((A.get(a) or {}).get("status")))
           or str((A.get(a) or {}).get("status")) in DEFAULT_STATUSES}
    out.append(("P5 every stepped arm has a terminal non-default status", bool(stepped) and not bad, bad or "no arms stepped"))
    miss = [a for a in DESIGN_ARMS[v] if not has_arm(A, a)]
    out.append(("P6 every DESIGN arm of the variant is in the results (both conditional branches forced)", not miss, miss))
    req, cond = plan_entries(R)
    forced = bool(R.get("force_conditional"))
    miss = [a for a in req + (cond if forced else []) if not has_arm(A, a)]
    out.append((f"P6b every entry of the notebook's own R['plan'] stepped ({'conditionals forced' if forced else 'conditionals optional'})",
                bool(req) and not miss, miss or ("no R['plan']" if not req else "")))
    errs = {a: r["status"] for a, r in A.items() if str(r.get("status", "")).startswith("ERROR")}
    out.append(("P7 no ERROR_* status and R['errors'] empty", not errs and not R.get("errors"),
                {"statuses": errs, "errors": [str(e.get("where", "")) + ": " + str(e.get("error", e))[:300] if isinstance(e, dict)
                                              else str(e)[:300] for e in (R.get("errors") or [])][:6]}))
    neg = {a: r["status"] for a, r in A.items() if str(r.get("status", "")).startswith(("NO_GO", "INVALID"))
           and not (a in EXPECTED_NEGATIVE and EXPECTED_NEGATIVE[a].match(str(r["status"])))}
    out.append(("P7b no NO_GO_* / INVALID_* in a positive FAKE run (gates hold on clean fakes)", not neg,
                {a: (s, (A[a].get("gates") or {})) for a, s in neg.items()}))
    def required(a):
        return any(a.startswith(d) if d.endswith("__") else a == d for d in DESIGN_ARMS[v])
    skipped = {a: r["status"] for a, r in A.items() if required(a) and str(r["status"]) == "SKIPPED_BUDGET"}
    out.append(("P8 no required arm SKIPPED_BUDGET (scaled budget fits the plan)", allow_budget_skips or not skipped, skipped))
    launched = sorted({x["arm"] for x in run["launches"] if not x["flag_check"] and x["arm"]})
    bad = {}
    for a in [x for x in launched if x.startswith("s1_")]:
        cv = cache_verdicts(A.get(a) or {})
        if not cv or any(d["status"] != "ISOLATED" for _, d in cv):
            bad[a] = [d["status"] for _, d in cv] or "no cache verdict"
    s1_planned = any(a.startswith("s1_") for a in DESIGN_ARMS[v])
    out.append(("P9 section-1 cache verdicts ISOLATED (DESIGN §3.3 proof)" + ("" if s1_planned else ": n/a, no section-1 arm"),
                (not s1_planned) or (bool([a for a in launched if a.startswith("s1_")]) and not bad),
                bad or ("no section-1 launch" if s1_planned and not any(a.startswith("s1_") for a in launched) else "")))
    bad = {}
    for a in launched:
        fr = flag_records(A.get(a) or {})
        if not fr or not all(d.get("ran") and d.get("ok") for _, d in fr):
            bad[a] = [(d.get("ran"), d.get("ok"), d.get("unknown"), d.get("ambiguous"), d.get("parse_error")) for _, d in fr] or "none"
    out.append(("P10 flag check ran and ok for every launched arm", bool(launched) and not bad, bad))
    rt = R.get("argv") or {}
    miss = [a for a in launched if a not in rt]
    out.append(("P11 runtime argv recorded in R['argv'] for every launched arm", bool(rt) and not miss, miss[:10]))
    pr = [(pth, d) for pth, d in walk(R) if "patch" in pth.lower() and "ok" in d and isinstance(d.get("ok"), bool)]
    if run.get("site"):
        out.append(("P12 patch state / apply / revert ok on the real 0.5.21 source copy", bool(pr) and all(d["ok"] for _, d in pr),
                    [pth for pth, d in pr if not d["ok"]][:8] or ("no patch record" if not pr else "")))
        pg = {a: (r.get("gates") or {}).get("patches") for a, r in A.items() if "patch_state" in r}
        applied = [a for a, r in A.items() if r.get("patches_expected") and (r.get("gates") or {}).get("patches") == "GO"]
        out.append(("P12b every SGLang arm reached exactly its expected patch set (gate GO), patched arms included",
                    bool(pg) and all(g == "GO" for g in pg.values()) and (bool(applied) or not any(
                        r.get("patches_expected") for r in A.values())), {"not_go": {a: g for a, g in pg.items() if g != "GO"},
                                                                         "patched_arms_go": applied}))
    else:
        out.append(("P12 patch apply/revert exercised (E24_FAKE_SGLANG_SITE)", not require_site, "no SGLang source copy"))
    C = comparisons(R)
    want = [(x, y) for x, y in S1_PAIRS if f"s1_{x}" in A and f"s1_{y}" in A]
    miss = [f"{x}<->{y}" for x, y in want if (x, y) not in C]
    badst = {f"{k[0]}<->{k[1]}": r.get("status") for k, r in C.items() if not CMP_STATUS.match(str(r.get("status")))}
    fails = {f"{k[0]}<->{k[1]}": r.get("status") for k, r in C.items() if str(r.get("status", "")).startswith("FAIL")}
    if want:
        out.append(("P13 section-1 comparisons present for every pair of the variant, statuses in the §3.2 vocabulary",
                    not miss and not badst, {"missing": miss, "bad_status": badst}))
        out.append(("P14 no FAIL comparison in a positive run (fakes agree up to NEAR_TIE)", not fails, fails))
        dec = [pth for pth, _ in walk(R.get("comparisons") or {}) if "decisive" in pth.lower()] + \
              [k for k in ((R.get("comparisons") or {}).get("1") or {}) if "decisive" in str(k).lower()]
        out.append(("P15 decisive-pair record present", bool(dec), ""))
    rep = Path(run["out"]) / "e24_report.md"
    txt = rep.read_text(encoding="utf-8", errors="replace") if rep.is_file() else ""
    out.append(("P16 e24_report.md written (with the NOT_MEASURED denominators)", len(txt) > 300 and "NOT_MEASURED" in txt,
                f"{len(txt)} chars"))
    n_out = sum(1 for _ in Path(run["out"]).rglob("*") if _.is_file())
    out.append(("P17 OUT holds < 300 files", n_out < OUT_MAX_FILES, n_out))
    out.append(("P18 raw outputs packed into OUT/e24_raw.tar.gz", (Path(run["out"]) / "e24_raw.tar.gz").is_file(), ""))
    shas = R.get("shas") or {}
    keys = ("lab", "patches", "hook", "unit", "loadpath", "trace_summary", "notebook_builder")
    out.append(("P19 R['shas'] records every embedded module", all(shas.get(k) for k in keys), [k for k in keys if not shas.get(k)]))
    W = windows(R)
    good = [p for p, s in W if s.split("(")[0] in GOOD_WINDOWS]
    badw = {p: s for p, s in W if s.startswith(("FAIL", "ERROR"))}
    out.append(("P20 window statuses: none FAIL_* / ERROR_*, at least one valid window (E23's kit ignored them)",
                not badw and (bool(good) or not W), {"bad": badw, "valid": len(good), "all": len(W),
                                                     "inconclusive": {p: s for p, s in W if s.startswith("INCONCLUSIVE")}}))
    hk = {a: (r.get("gates") or {}).get("hook") for a, r in A.items() if r.get("engine") == "sglang" and (r.get("gates") or {}).get("hook")}
    hfail = sorted({json.dumps(f, default=str)[:200] for r in A.values() for f in ((r.get("hook_selftest") or {}).get("failed") or [])})
    out.append(("P21 hook self-test GO on every SGLang arm (the fake's stub sglang classes carry every e24_hook target method)",
                bool(hk) and all(g == "GO" for g in hk.values()), {"not_go": {a: g for a, g in hk.items() if g != "GO"},
                                                                  "failures": hfail[:3]}))
    out.extend(env_checks(run))
    return out


GOOD_WINDOWS = {"STEADY", "CAPACITY_LIMITED", "AGGREGATE_STEADY", "MEASURED", "ADMITTED_ALL", "PLATEAU"}


def windows(R):
    """Every measurement window in the sections (E23 kit rule: a dict with a status and a tok_s)."""
    return [(p, str(d["status"])) for p, d in walk(R.get("sections") or {}) if "status" in d and "tok_s" in d]


# ------------------------------------------------------------------------------------------------ negative scenarios
def _status(R, arm):
    return str((arms_of(R).get(arm) or {}).get("status"))


def _continued(run, after_arm):
    """The notebook went on after `after_arm`: a later arm in R['order'] ran (not SKIPPED_*), rc 0, report written."""
    o = order_arms(run["R"])
    later = o[o.index(after_arm) + 1:] if after_arm in o else []
    ran = [a for a in later if not _status(run["R"], a).startswith("SKIPPED")]
    return run["rc"] == 0 and bool(ran) and (Path(run["out"]) / "e24_report.md").is_file(), {"later_ran": ran[:6], "rc": run["rc"]}


def _server_log(run, arm):
    logs = sorted(Path(run["out"], "logs").glob(f"server_{arm}*.log"))
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)


def find_rows(run, arm, tags=("m1", "c16a")):
    """The rows a section-1 pair compares (WP4: m1 + c16a, dumped as TMP/e24/<arm>/rows_<tag>.json; rows carry key,
    tokens, margins, top2). Returns (rows, [files]) or (None, [files tried])."""
    rows, files = [], []
    for t in tags:
        f = Path(run["tmp"]) / "e24" / arm / f"rows_{t}.json"
        files.append(str(f))
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None, files
        rows += d if isinstance(d, list) else []
    return (rows or None), files


def load_lab(run):
    return load_module(run, "e24_lab")


def check_diverge(run, ctx):
    R = run["R"]
    C = comparisons(R)
    cd6 = C.get(("c", "d6")) or {}
    out = [("N1a injected high-margin divergence on s1_d6 reached only that arm",
            {x["arm"] for x in run["launches"] if x["rules"] and not x["flag_check"]} == {"s1_d6"},
            sorted({x["arm"] for x in run["launches"] if x["rules"]}))]
    out.append(("N1b with A/A (the notebook's own report-cell verdict): c<->d6 is FAIL_HIGH_MARGIN",
                cd6.get("status") == "FAIL_HIGH_MARGIN", {k: cd6.get(k) for k in ("status", "aa_high", "hm", "counts", "t_pair")}))
    others = {f"{k[0]}<->{k[1]}": r.get("status") for k, r in C.items() if "d6" not in k and str(r.get("status", "")).startswith("FAIL")}
    out.append(("N1c no other pair FAILs (the divergence is attributed to the right pair)", not others, others))
    dec = (((R.get("comparisons") or {}).get("1") or {}).get("decisive") or {})
    out.append(("N1d decisive pair of the a -> b -> c32 -> c -> d6/d4 chain is s1_c~s1_d6", dec.get("pair") == "s1_c~s1_d6",
                {k: dec.get(k) for k in ("pair", "verdict", "upstream_not_normalisable")}))
    d6 = arms_of(R).get("s1_d6") or {}
    out.append(("N1f the STATE_FLIP verdict reaches s1_d6's arm status (decision review #5), the measured status kept",
                str(d6.get("status")).startswith("NO_GO_STATE_FLIP") and d6.get("status_measured") is not None,
                {k: d6.get(k) for k in ("status", "status_measured", "s1_verdicts")}))
    rc_, f_c = find_rows(run, "s1_c")
    rd_, f_d = find_rows(run, "s1_d6")
    lab = load_lab(run)
    if lab is None or rc_ is None or rd_ is None:
        out.append(("N1e without A/A: the lab's compare() on the same rows gives BOUNDARY", False,
                    {"lab": bool(lab), "rows_c": f_c, "rows_d6": f_d}))
    else:
        tp = cd6.get("t_pair") or 0.5
        try:
            st = lab.compare(rc_, rd_, kind="same_weights", t_pair=tp, aa_high=None).get("status")
            stA = lab.compare(rc_, rd_, kind="same_weights", t_pair=tp, aa_high=cd6.get("aa_high")).get("status")
        except Exception as e:
            st, stA = repr(e), None
        out.append(("N1e same rows, no A/A (aa_high None) -> BOUNDARY; with the notebook's A/A -> FAIL_HIGH_MARGIN",
                    st == "BOUNDARY" and stA == "FAIL_HIGH_MARGIN", {"aa_none": st, "aa_notebook": stA, "aa_high": cd6.get("aa_high"),
                                                                     "rows": len(rc_), "t_pair": tp}))
    return out


def check_leak(run, ctx):
    A = arms_of(run["R"])
    v4 = [d["status"] for _, d in cache_verdicts(A.get("s1_d4") or {})]
    out = [("N2a FAKE_SG_LEAK_DEFAULT_CACHE on s1_d4: cache verdict INVALID_DEFAULT_DIR_WRITTEN", "INVALID_DEFAULT_DIR_WRITTEN" in v4, v4),
           ("N2b s1_d4 arm status INVALID_CACHE_ISOLATION", _status(run["R"], "s1_d4").startswith("INVALID"), _status(run["R"], "s1_d4"))]
    others = {a: [d["status"] for _, d in cache_verdicts(A.get(a) or {})] for a in ("s1_d6", "s1_c") if a in A}
    out.append(("N2c the other section-1 SGLang arms stay ISOLATED", all(v and set(v) == {"ISOLATED"} for v in others.values()), others))
    c = comparisons(run["R"]).get(("c", "d4")) or {}
    unused = (not c) or str(c.get("status", "")).startswith(("INVALID", "INCONCLUSIVE", "SKIPPED")) or c.get("used") is False
    out.append(("N2d c<->d4 not used as evidence", unused, c.get("status")))
    return out


def check_mamba(run, ctx):
    hit = sorted({x["arm"] for x in run["launches"] if x["rules"] and not x["flag_check"]})
    arm = hit[0] if hit else None
    st = _status(run["R"], arm) if arm else "no target launched"
    out = [("N3a FAKE_SG_FAULT=mamba_assert reached exactly one capacity arm", len(hit) == 1, hit),
           ("N3b that arm is FAIL_* / NO_GO_* (not GO, not ERROR_*)", st.startswith(("FAIL", "NO_GO")), st)]
    if arm:
        out.append(("N3c its server log shows the assertion", "Can not alloc mamba cache" in _server_log(run, arm), ""))
        ok, info = _continued(run, arm)
        out.append(("N3d the notebook continued after the crash", ok, info))
    return out


def check_flag(run, ctx):
    st = _status(run["R"], "s1_d4")
    launched = [x for x in run["launches"] if x["arm"] == "s1_d4" and not x["flag_check"]]
    out = [("N4a an unsupported flag on s1_d4 gives NO_GO_FLAG_UNSUPPORTED", st == "NO_GO_FLAG_UNSUPPORTED", st),
           ("N4b s1_d4 was never launched", not launched, len(launched))]
    ok, info = _continued(run, "s1_d4")
    out.append(("N4c the notebook continued", ok, info))
    return out


def armed_logprob_control(work, fake_dir=HERE, input_root=None):
    """Proves the logprob_oom fault is armed: the fake WITHOUT the chunk env crashes on a > 256-row scoring request, and
    WITH SGLANG_LOGPROB_CHUNK_SIZE=128 it answers. Returns (ok, info)."""
    fake = Path(fake_dir) / "fake_sglang_server.py"
    if not fake.is_file():
        return False, "fake_sglang_server.py missing"
    info = {}
    model = str(Path(input_root or "/kaggle/input") / "models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1")
    for label, chunk in (("no_chunk_env", None), ("chunk_128", "128")):
        port = free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith(CLEAN_ENV_PREFIX)}
        env.update(FAKE_SG_FAULT="logprob_oom", FAKE_SG_STARTUP_SEC="0.3", E24_TIME_SCALE="0.02",
                   HOME=str(Path(work) / "home_armed"))
        if chunk:
            env["SGLANG_LOGPROB_CHUNK_SIZE"] = chunk
        logf = open(Path(work) / f"armed_{label}.log", "w")
        pr = subprocess.Popen([PY, str(fake), "--model-path", model, "--host", "127.0.0.1", "--port", str(port), "--tp-size", "4",
                               "--kv-cache-dtype", "fp8_e4m3", "--mem-fraction-static", "0.83", "--max-running-requests", "4",
                               "--max-mamba-cache-size", "16", "--mamba-ssm-dtype", "float16", "--enable-metrics",
                               "--random-seed", "1"], stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            t = time.time()
            ready = False
            while time.time() - t < 60 and pr.poll() is None:
                try:
                    ready = urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2).status == 200
                except Exception:
                    ready = False
                if ready:
                    break
                time.sleep(0.2)
            body = json.dumps({"input_ids": [11 + (i % 50) for i in range(300)], "return_logprob": True,
                               "logprob_start_len": 0, "top_logprobs_num": 2, "return_text_in_logprobs": False,
                               "sampling_params": {"max_new_tokens": 1, "temperature": 0}}).encode()
            try:
                r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/generate", body,
                                                                  {"Content-Type": "application/json"}), timeout=30)
                answered = r.status == 200
            except Exception:
                answered = False
            time.sleep(1.0)
            info[label] = {"ready": ready, "answered": answered, "alive_after": pr.poll() is None}
        finally:
            try:
                os.killpg(pr.pid, signal.SIGKILL)
            except Exception:
                pass
            logf.close()
    ok = info.get("no_chunk_env", {}).get("ready") and not info["no_chunk_env"]["alive_after"] and \
        info.get("chunk_128", {}).get("answered") and info["chunk_128"]["alive_after"]
    return bool(ok), info


def check_logprob(run, ctx):
    sg = [x for x in run["launches"] if x["engine"] == "sglang" and not x["flag_check"]]
    bad = [(x["arm"], x["env"].get("SGLANG_LOGPROB_CHUNK_SIZE")) for x in sg
           if not str(x["env"].get("SGLANG_LOGPROB_CHUNK_SIZE", "")).isdigit() or int(x["env"]["SGLANG_LOGPROB_CHUNK_SIZE"]) > 256]
    out = [("N5a armed control: the fake crashes without the chunk env and answers with 128", ctx.get("armed_ok"), ctx.get("armed")),
           ("N5b FAKE_SG_FAULT=logprob_oom on every SGLang launch, each with SGLANG_LOGPROB_CHUNK_SIZE <= 256", bool(sg) and not bad
            and all(x["rules"] for x in sg), bad[:6])]
    A = arms_of(run["R"])
    s1 = {a: r["status"] for a, r in A.items() if a.startswith("s1_") and a not in ("s1_a", "s1_b")}
    died = {a: s for a, s in s1.items() if s.startswith(("ERROR", "FAIL")) or "DIED" in s or "LAUNCH" in s}
    oom = [a for a in s1 if re.search(r"out of memory|OutOfMemoryError", _server_log(run, a), re.I)]
    out.append(("N5c the fault never fired: no section-1 SGLang arm died / FAILed, no OOM line", bool(s1) and not died and not oom,
                {"died": died, "oom_logs": oom}))
    pf = {a: (r.get("gates") or {}).get("preflight") for a, r in A.items() if (r.get("gates") or {}).get("preflight")}
    out.append(("N5d the >= 4,096-row scoring pre-flight ran and passed (gate GO) with the fault armed",
                bool(pf) and all(v == "GO" for v in pf.values()), pf))
    return out


def _census_records(R):
    """WP1 census() results (offered + plateau_hold_sec + running stats)."""
    return [(pth, d) for pth, d in walk(R.get("sections") or {}) if "offered" in d and "plateau_hold_sec" in d]


def check_census(run, ctx):
    """DESIGN §4.4 / review F1: during a long chunked prefill the gauges stay flat (republished unchanged), so a
    gauge-only plateau ends admission mid-prefill. Here every 90k prefill outlasts the plateau hold: the census must still
    wait for the prefill counter and the Prefill batch lines, then measure a flat window with no prefill in it."""
    recs = [(p, d) for p, d in _census_records(run["R"]) if d.get("band") and int(d["band"]) >= 80000]
    tps = float(ctx.get("prefill_tps") or 0)
    out = [("N6a a 90k census ran on the control under the slow-prefill rule", bool(recs), [p for p, _ in recs]),
           ("N6b the slow-prefill env reached the census arm only", sorted({x["arm"] for x in run["launches"]
                                                                          if x["rules"] and not x["flag_check"]}) == ["s2_w6s20"],
            sorted({x["arm"] for x in run["launches"] if x["rules"]}))]
    for pth, d in recs[:2]:
        st, rg, rl = str(d.get("status")), d.get("running_gauge") or {}, d.get("running_lines") or {}
        per = int(d["band"]) / tps if tps else None
        out.append((f"N6c {pth}: CAPACITY_LIMITED / ADMITTED_ALL (not INCONCLUSIVE_NO_PLATEAU, not FAIL)",
                    st in ("CAPACITY_LIMITED", "ADMITTED_ALL"), {"status": st, "admission": d.get("admission"),
                                                                "max_wait_sec": d.get("max_wait_sec")}))
        out.append((f"N6d {pth}: no Prefill batch line inside the window and running flat (p10 == p90): no early plateau",
                    d.get("prefill_lines_in_window") == 0 and rg.get("n") and rg.get("p10") == rg.get("p90")
                    and (not rl.get("n") or rl.get("p10") == rl.get("p90")),
                    {"prefill_lines_in_window": d.get("prefill_lines_in_window"), "running_gauge": rg, "running_lines": rl}))
        out.append((f"N6e {pth}: the test is not vacuous: one prefill ({per and round(per, 2)} s) outlasts the hold "
                    f"({d.get('plateau_hold_sec')} s) and admission took >= admitted x prefill",
                    bool(per) and per > float(d.get("plateau_hold_sec") or 1e9) and
                    float((d.get("admission") or {}).get("sec") or 0) >= 0.8 * per * max(1, int(d.get("admitted_client") or 1) - 1),
                    {"admitted_client": d.get("admitted_client"), "admission_sec": (d.get("admission") or {}).get("sec")}))
    return out


def check_crash(run, ctx):
    st = _status(run["R"], "s1_d4")
    hit = sorted({x["arm"] for x in run["launches"] if x["rules"] and not x["flag_check"]})
    out = [("N7a FAKE_SG_FAULT=crash_start reached s1_d4 only", hit == ["s1_d4"], hit),
           ("N7b the crashed launch is FAIL_* / NO_GO_* / INCONCLUSIVE_* (never GO, never ERROR_*)",
            st.startswith(("FAIL", "NO_GO", "INCONCLUSIVE")), st)]
    ok, info = _continued(run, "s1_d4")
    out.append(("N7c the notebook continued", ok, info))
    return out


def check_hard_stop(run, ctx):
    R = run["R"]
    blob = json.dumps(R, default=str)
    A = arms_of(R)
    o = order_arms(R)
    out = [("N8a the watchdog fired (HARD_STOP recorded) while an arm hung", "HARD_STOP" in blob, ""),
           ("N8b notebook exit 0 and the report written after the hard stop",
            run["rc"] == 0 and (Path(run["out"]) / "e24_report.md").is_file(), run["rc"]),
           ("N8c every arm terminal; arms after the stop are SKIPPED_*",
            all(TERMINAL.match(str((A.get(a) or {}).get("status"))) for a in o) and
            any(str((A.get(a) or {}).get("status", "")).startswith("SKIPPED") for a in o), {a: (A.get(a) or {}).get("status") for a in o}),
           ("N8d no live fake left behind (the watchdog killed the servers)", not run["killed_leftovers"], run["killed_leftovers"]),
           ("N8e no arm reads ERROR_* (engine deaths are NO_GO_*, the arm the watchdog stopped INCONCLUSIVE_HARD_STOP or NO_GO_*)",
            not any(str((A.get(a) or {}).get("status", "")).startswith("ERROR") for a in o),
            {a: (A.get(a) or {}).get("status") for a in o})]
    return out


def check_startup(run, ctx):
    """The vLLM fake made to start in 1.3 x the scaled 600 s: every launch over 600 s is NO_GO_STARTUP_600 (README gate on
    every launch, cold included), its measurements are kept and labelled diagnostic (DESIGN §3.6)."""
    A = arms_of(run["R"])
    v = {a: A.get(a) or {} for a in ("s1_a", "s1_b")}
    out = [("N10a vLLM start over the scaled 600 s: s1_a / s1_b NO_GO_STARTUP_600 (README 600 s gate on every launch)",
            all(r.get("status") == "NO_GO_STARTUP_600" and (r.get("gates") or {}).get("launch") == "NO_GO_STARTUP_600"
                for r in v.values()), {a: (r.get("status"), r.get("ready_sec")) for a, r in v.items()}),
           ("N10b labelled diagnostic, measurements kept (startup time does not change numerics)",
            all(r.get("diagnostic") and any(isinstance(x, dict) and not x.get("error") for x in (r.get("measured") or {}).values())
                for r in v.values()), {a: {"diagnostic": r.get("diagnostic"), "measured": sorted(r.get("measured") or {})}
                                       for a, r in v.items()})]
    ok, info = _continued(run, "s1_b")
    out.append(("N10c the notebook continued", ok, info))
    return out


def check_tokenizer(run, ctx):
    R = run["R"]
    launched = [x for x in run["launches"] if not x["flag_check"]]
    o = order_arms(R)
    return [("N9a an INT4 tokenizer with different ids gives NO_GO_TOKENIZER_MISMATCH",
             (R.get("gates") or {}).get("tokenizer") == "NO_GO_TOKENIZER_MISMATCH", (R.get("gates") or {}).get("tokenizer")),
            ("N9b fail closed: no server launched, every stepped arm SKIPPED_*", not launched and bool(o) and all(
                _status(R, a).startswith("SKIPPED") for a in o), {"launches": len(launched), "statuses": {a: _status(R, a) for a in o}}),
            ("N9c exit 0 and the report written", run["rc"] == 0 and (Path(run["out"]) / "e24_report.md").is_file(), run["rc"])]


def _hit(run):
    return sorted({x["arm"] for x in run["launches"] if x["rules"] and not x["flag_check"]})


def check_alloc_warning(run, ctx):
    """Review (real platform #1): the PyTorch allocator's recovered-OOM warning, logged after every flush on the census
    control (EMS BASE S20 log L1062), never fails the census, the windows or the arm."""
    R, A = run["R"], arms_of(run["R"])
    rec = A.get("s2_w6s20") or {}
    cen = ((rec.get("measured") or {}).get("census_90k")) or {}
    w8 = ((rec.get("windows") or {}).get("8k")) or {}
    seen = ((cen.get("errors") or {}).get("allocator_oom_warnings")) or 0
    return [("N11a FAKE_SG_FAULT=oom_log reached the census control only", _hit(run) == ["s2_w6s20"], _hit(run)),
            ("N11b the warning sits inside the census slice (armed): allocator_oom_warnings >= 1", seen >= 1,
             {"census_errors": {k: (cen.get("errors") or {}).get(k) for k in ("allocator_oom_warnings", "oom_lines")},
              "log_count": _server_log(run, "s2_w6s20").count("CUDACachingAllocator.cpp:3933]")}),
            ("N11c census CAPACITY_LIMITED / ADMITTED_ALL, never FAIL_OOM", cen.get("status") in ("CAPACITY_LIMITED", "ADMITTED_ALL"),
             cen.get("status")),
            ("N11d the 8k window STEADY and the arm GO (no NO_GO_OOM anywhere)",
             w8.get("status") == "STEADY" and rec.get("status") == "GO", {"window": w8.get("status"), "arm": rec.get("status")})]


def check_launch_streak(run, ctx):
    """Review (real platform #4) + review 2 #5: startup timeouts of three distinct SGLang configs in a row stop the engine;
    later SGLang arms are SKIPPED_DEPENDENCY instead of each waiting out a launch; vLLM arms and the report are unaffected."""
    R, A, o = run["R"], arms_of(run["R"]), order_arms(run["R"])
    dead = ["s1_d6", "s1_d6fr", "s1_d4"]
    later = [a for a in o[o.index("s1_d4") + 1:] if a in A] if "s1_d4" in o else []
    sgl_later = [a for a in later if not a.startswith(("s1_a", "s1_b"))]
    launched = sorted({x["arm"] for x in run["launches"] if not x["flag_check"] and x["arm"] in sgl_later})
    return [("N12a the three arms read NO_GO_LAUNCH", all(_status(R, a) == "NO_GO_LAUNCH" for a in dead), {a: _status(R, a) for a in dead}),
            ("N12b every later SGLang arm SKIPPED_DEPENDENCY 'launches stopped'",
             bool(sgl_later) and all(_status(R, a) == "SKIPPED_DEPENDENCY" and "launches stopped" in str((A[a] or {}).get("why"))
                                     for a in sgl_later), {a: (_status(R, a), str((A[a] or {}).get("why"))[:60]) for a in sgl_later}),
            ("N12c no SGLang server launched after the third failure", not launched, launched),
            ("N12d vLLM arms GO, exit 0, report written", _status(R, "s1_a") == "GO" and _status(R, "s1_b") == "GO" and run["rc"] == 0
             and (Path(run["out"]) / "e24_report.md").is_file(), {"s1_a": _status(R, "s1_a"), "s1_b": _status(R, "s1_b"), "rc": run["rc"]})]


def check_restore(run, ctx):
    """Review (decision logic #4) + the second round: a stale restored state (divergence only in requests resumed from a
    cached prefix) on s1_d6. The ARC prompts share a 64-token header, so "cold" rows are partial restores too: the
    production d6 path diverges from target-only (c <-> d6 is a true finding), its own pass 1 <-> pass 2 verdict catches
    the restore, the c16 A/A compares only rows with the same restore state (the admission-order difference stays out
    of the allowance), and no other arm is touched."""
    R = run["R"]
    c1 = (R.get("comparisons") or {}).get("1") or {}
    w = (c1.get("within") or {}).get("s1_d6") or {}
    pp, aa = w.get("pass1_pass2") or {}, w.get("c16_aa") or {}
    A = arms_of(R)
    d6 = A.get("s1_d6") or {}
    others = {a: (A.get(a) or {}).get("status") for a in ("s1_a", "s1_b", "s1_c", "s1_d4", "s1_d6fr")}
    fails = {k: v.get("status") for k, v in (c1.get("pairs") or {}).items() if "s1_d6" not in k.split("~")
             and str(v.get("status") or "").startswith("FAIL")}
    return [("N13a the restore divergence reached s1_d6 only", _hit(run) == ["s1_d6"], _hit(run)),
            ("N13b pass 1 <-> pass 2 exercised and shows STATE_FLIPs",
             w.get("pass2_restore") == "EXERCISED" and (pp.get("state_flips") or 0) > 0,
             {"restore": w.get("pass2_restore"), "status": pp.get("status"), "state_flips": pp.get("state_flips")}),
            ("N13c s1_d6 reads NO_GO_STATE_FLIP from its pass 1 <-> pass 2 verdict, status_measured kept",
             str(d6.get("status")).startswith("NO_GO_STATE_FLIP") and any("pass1_pass2" in v for v in d6.get("s1_verdicts") or [])
             and d6.get("status_measured") is not None, {k: d6.get(k) for k in ("status", "status_measured", "s1_verdicts")}),
            ("N13d the c16 A/A drops the rows whose restore state differed between the batches and stays clean",
             w.get("c16_aa_pure") is True and not (aa.get("high") or 0),          # mismatch rows depend on admission order
             {"pure": w.get("c16_aa_pure"), "mismatch_rows": w.get("c16_aa_rows_restore_mismatch"), "high": aa.get("high")}),
            ("N13e the corruption stays on s1_d6: every other section-1 arm GO, no pair without d6 FAILs",
             all(v == "GO" for v in others.values()) and not fails, {"others": others, "fails_without_d6": fails,
                                                                  "c~d6": ((c1.get("pairs") or {}).get("s1_c~s1_d6") or {}).get("status")})]


def check_grid_text(run, ctx):
    """Review (real platform #3): streams writing space-separated digit grids are not garbage under load; armed = the
    same texts fail the old variety rule (digits counted as words)."""
    R, A = run["R"], arms_of(run["R"])
    rec = A.get("s2_w6s20") or {}
    w8 = ((rec.get("windows") or {}).get("8k")) or {}
    f = Path(run["tmp"]) / "e24" / "s2_w6s20" / "parity_rows.json"
    try:
        texts = [r.get("text") or "" for r in json.loads(f.read_text(encoding="utf-8"))]
    except Exception:
        texts = []

    def old_low(t):
        words = re.findall(r"\w+", t)
        return len(words) > 40 and len(set(words)) < 0.08 * len(words)
    return [("N14a FAKE_SG_TEXT_STYLE=grids reached the W6 S20 control only", _hit(run) == ["s2_w6s20"], _hit(run)),
            ("N14b armed: its rows are digit grids that the old variety rule calls LOW_VARIETY",
             bool(texts) and all(old_low(t) for t in texts), {"rows": len(texts), "sample": (texts[0][:40] if texts else None)}),
            ("N14c the 8k window STEADY (never FAIL_GARBAGE_UNDER_LOAD) and the arm GO",
             w8.get("status") == "STEADY" and rec.get("status") == "GO", {"window": w8.get("status"), "arm": rec.get("status")})]


def scenarios(scale):
    """Negative scenarios (rev-2 WP5 list + the review's Linux fault modes). Rules are applied by the wrapper fake dir
    (module docstring); "env" goes to the notebook itself (e.g. a tight E24_BUDGET_SEC)."""
    ctrl = {"engine": "sglang", "flag_check": False, "arm_re": r"^s2_w6s20$"}       # the SMOKE W6 S20 control (90k census)
    return {
        "crash_start": {"variant": "SMOKE", "check": check_crash,
                        "rules": [{"match": {"engine": "sglang", "arm_re": r"^s1_d4$", "flag_check": False},
                                   "env": {"FAKE_SG_FAULT": "crash_start"}}]},
        "hard_stop": {"variant": "SMOKE", "check": check_hard_stop,
                      "env": {"E24_BUDGET_SEC": str(int(VARIANTS["SMOKE"][2] * 60 * scale))},
                      "rules": [{"match": {"engine": "sglang", "flag_check": False}, "env": {"FAKE_SG_FAULT": "hang_decode"}}]},
        "diverge": {"variant": "SMOKE", "check": check_diverge,
                    "rules": [{"match": {"engine": "sglang", "arm_re": r"^s1_d6$"}, "env": {"FAKE_SG_DIVERGE_AT": "24"}}]},
        "leak_cache": {"variant": "SMOKE", "check": check_leak,
                       "rules": [{"match": {"engine": "sglang", "arm_re": r"^s1_d4$"}, "env": {"FAKE_SG_LEAK_DEFAULT_CACHE": "1"}}]},
        "mamba_assert": {"variant": "SMOKE", "check": check_mamba,
                         "rules": [{"match": ctrl, "env": {"FAKE_SG_FAULT": "mamba_assert"}}]},
        "flag_unsupported": {"variant": "SMOKE", "check": check_flag,
                             "rules": [{"match": {"engine": "sglang", "arm_re": r"^s1_d4$", "flag_check": True},
                                        "add_flag": "--e24-not-an-option"}]},
        "logprob_oom": {"variant": "SMOKE", "check": check_logprob, "armed": True,
                        "rules": [{"match": {"engine": "sglang"}, "env": {"FAKE_SG_FAULT": "logprob_oom"}}]},
        # one 90k prefill = 1.3 x the scaled plateau hold (TS x 1.5 x 90000 / 1300 s), so the gauges stay flat for longer
        # than the hold while a prefill is still running; ~8 admitted stay inside max_wait (TS x (24 x 90000 / 1300 + 120))
        "long_prefill_census": {"variant": "SMOKE", "check": check_census, "ctx": {"prefill_tps": int(667 / scale)},
                                "rules": [{"match": ctrl, "env": {"FAKE_SG_PREFILL_TPS": str(int(667 / scale))}}]},
        "tokenizer_mismatch": {"variant": "SMOKE", "check": check_tokenizer, "rules": [], "mock": {"tokenizer_mismatch": True},
                               "local_only": True},
        # the vLLM fake's startup at 1.3 x the scaled 600 s gate (its default cold start is now ~500 s x scale, under the gate)
        "startup_gate": {"variant": "SMOKE", "check": check_startup, "rules": [],
                         "env": {"FAKE_VLLM_STARTUP_SEC": str(round(600 * scale * 1.3, 2))}},
        # second validation round (2026-10-30): end-to-end proofs of the review fixes
        "alloc_warning": {"variant": "SMOKE", "check": check_alloc_warning,
                          "rules": [{"match": ctrl, "env": {"FAKE_SG_FAULT": "oom_log"}}]},
        # startup HANGS (the guard counts timeouts of distinct configs, never fast exits: review 2 #5); crash_start covers
        # the fast-exit side (one arm fails, the rest run)
        "launch_streak": {"variant": "SMOKE", "check": check_launch_streak,
                          "rules": [{"match": {"engine": "sglang", "arm_re": r"^s1_(d6|d6fr|d4)$", "flag_check": False},
                                     "env": {"FAKE_SG_FAULT": "slow_start", "FAKE_SG_SLOW_START_SEC": "100000"}}]},
        "restore_corruption": {"variant": "SMOKE", "check": check_restore,
                               "rules": [{"match": {"engine": "sglang", "arm_re": r"^s1_d6$"}, "env": {"FAKE_SG_RESTORE_DIVERGE_AT": "24"}}]},
        "grid_text": {"variant": "SMOKE", "check": check_grid_text,
                      "rules": [{"match": ctrl, "env": {"FAKE_SG_TEXT_STYLE": "grids"}}]},
    }


# ------------------------------------------------------------------------------------------------ summary + main
def summarize(results, path_json, path_md):
    allc = [(r["name"], lbl, ok, info) for r in results for lbl, ok, info in r["checks"]]
    passed = sum(1 for _, _, ok, _ in allc if ok)
    S = {"all_ok": bool(allc) and passed == len(allc), "passed": passed, "checks": len(allc), "t": time.strftime("%Y-%m-%d %H:%M:%S"),
         "runs": [{k: r.get(k) for k in ("name", "kind", "variant", "rc", "sec", "dir", "statuses", "error")} |
                  {"checks": [{"label": l, "ok": bool(o), "info": i if not o else ""} for l, o, i in r["checks"]]} for r in results]}
    Path(path_json).write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    md = [f"# E24 local run: {'ALL OK' if S['all_ok'] else 'NOT OK'} ({passed}/{len(allc)} checks)", "", f"{S['t']}", ""]
    for r in S["runs"]:
        md += [f"## {r['name']} ({r.get('kind')}, {r.get('variant') or '-'}) rc {r.get('rc')} in {r.get('sec')} s", ""]
        if r.get("statuses"):
            md += ["Arm statuses: " + ", ".join(f"`{a}` {s}" for a, s in r["statuses"].items()), ""]
        md += ["| check | ok | info |", "|---|---|---|"]
        for c in r["checks"]:
            md.append(f"| {c['label']} | {'PASS' if c['ok'] else '**FAIL**'} | {json.dumps(c['info'], default=str)[:400] if c['info'] else ''} |")
        md.append("")
    Path(path_md).write_text("\n".join(md), encoding="utf-8")
    return S


def run_all(variants, negatives, scale=0.02, work=None, keep=False, build="inplace", input_root=None, site=True,
            timeout_min=90, budget_slack=4.0, allow_budget_skips=False, codes=None, summary_dir=None, fake_dir=HERE,
            nb_dir=None, ports=True):
    """The whole local (or kit) campaign. `codes` = {variant: code} skips the build (the CPU kit passes the notebook code)."""
    work = Path(work or tempfile.mkdtemp(prefix="e24_local_")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    say(f"work dir {work}")
    results = []
    if codes is None:
        b = build_notebooks(build, work, nb_dir=nb_dir)
        bc, codes = build_checks(b)
        results.append({"name": "build", "kind": "build", "rc": b.get("rc"), "sec": b.get("sec"), "checks": bc, "dir": b.get("nbdir")})
        if not codes:
            S = summarize(results, work / "e24_local_summary.json", work / "e24_local_summary.md")
            return S, work
        snap = Path(b["nbdir"]).parent / "src" / "harness"       # the fakes of the SAME E24 tree as the notebooks (a mirror
        if fake_dir == HERE and (snap / "fake_sglang_server.py").is_file():   # snapshot never mixes with live WP2 edits)
            fake_dir = snap
        say(f"fakes from {fake_dir}")
    if input_root is None:
        sys.path.insert(0, str(HERE))
        import mock_inputs
        input_root = Path(mock_inputs.build(work / "input")["root"])
    if not (Path(fake_dir) / "fake_sglang_server.py").is_file():
        results.append({"name": "fakes", "kind": "precondition", "checks": [("fake servers present (WP2)", False, str(fake_dir))]})
    for v in variants:
        r = run_notebook(f"pos_{v}", codes[v], work, input_root, v, scale=scale, site=site, timeout_min=timeout_min,
                         budget_slack=budget_slack, fake_dir=fake_dir, ports=ports)
        r.update(kind="positive", checks=positive_checks(r, allow_budget_skips=allow_budget_skips, require_site=site),
                 statuses={a: x["status"] for a, x in arms_of(r["R"]).items()})
        results.append(r)
    S0 = scenarios(scale)
    for n in negatives:
        sc = S0[n]
        ctx = dict(sc.get("ctx") or {})
        if sc.get("armed"):
            ctx["armed_ok"], ctx["armed"] = armed_logprob_control(work, fake_dir=fake_dir, input_root=input_root)
        inp = input_root
        if sc.get("mock"):                       # a scenario-specific mock tree (local only: the real mounts are fixed)
            sys.path.insert(0, str(HERE))
            import mock_inputs
            inp = Path(mock_inputs.build(Path(work) / f"input_{n}", **sc["mock"])["root"])
        r = run_notebook(f"neg_{n}", codes[sc["variant"]], work, inp, sc["variant"], scale=scale, rules=sc["rules"],
                         extra_env=sc.get("env"), site=site, timeout_min=timeout_min, budget_slack=budget_slack, fake_dir=fake_dir,
                         ports=ports, unset=sc.get("unset", ()))
        try:
            chk = sc["check"](r, ctx)
        except Exception as e:
            chk = [(f"{n}: scenario check raised", False, repr(e))]
        r.update(kind="negative", checks=[(f"[{n}] {l}", o, i) for l, o, i in chk],
                 statuses={a: x["status"] for a, x in arms_of(r["R"]).items()})
        results.append(r)
    sd = Path(summary_dir or work)
    sd.mkdir(parents=True, exist_ok=True)
    S = summarize(results, sd / "e24_local_summary.json", sd / "e24_local_summary.md")
    if not keep:
        for r in results:
            if r.get("kind") in ("positive", "negative") and S["all_ok"]:
                shutil.rmtree(r["dir"], ignore_errors=True)
    return S, work


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="SMOKE,FULL_A,FULL_B")
    ap.add_argument("--negatives", default="all", help="all | none | comma list of: " + ",".join(scenarios(0.02)))
    ap.add_argument("--time-scale", type=float, default=0.02)
    ap.add_argument("--work", default=None, help="work dir (default: a new temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep every run dir (they are kept anyway when a check fails)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mirror-build", action="store_true", help="build in a copy of E24 under the work dir (repo untouched)")
    g.add_argument("--no-build", action="store_true", help="use the notebooks already in the repo")
    g.add_argument("--nb-dir", default=None, help="use the notebooks in this dir (e.g. an earlier --mirror-build), no build")
    ap.add_argument("--input", default=None, help="an existing /kaggle/input tree (default: a fresh mock)")
    ap.add_argument("--no-site", action="store_true", help="no SGLang source copy: patch steps are SKIPPED_FAKE")
    ap.add_argument("--timeout-min", type=float, default=90)
    ap.add_argument("--budget-slack", type=float, default=4.0,
                    help="E24_BUDGET_SEC = minutes x 60 x scale x slack + 600: the fakes' unscaled process overheads and FULL-size "
                         "traces run ~3x their scaled est_min (FULL_A measured 2026-10-30); budget skips are tested by hard_stop")
    ap.add_argument("--allow-budget-skips", action="store_true")
    ap.add_argument("--no-port-isolation", action="store_true",
                    help="keep the notebook's fixed ports 30000 / 8200 (default: two free ports per run, see isolate_ports)")
    ap.add_argument("--summary-dir", default=None)
    ap.add_argument("--list", action="store_true", help="list the variants and negative scenarios, then exit")
    ap.add_argument("--revalidate", nargs="+", metavar="RUN_DIR",
                    help="re-run the checks on kept run dirs (pos_<VARIANT> or neg_<scenario>) without running anything")
    a = ap.parse_args(argv)
    if a.revalidate:
        results, S0 = [], scenarios(a.time_scale)
        for rd in a.revalidate:
            r = load_run(rd)
            n = Path(rd).name
            if n.startswith("neg_") and n[4:] in S0:
                sc = S0[n[4:]]
                ctx = dict(sc.get("ctx") or {})
                if sc.get("armed"):                      # the fault's armed control is cheap: run it again
                    ctx["armed_ok"], ctx["armed"] = armed_logprob_control(Path(rd).parent, input_root=(r.get("env") or {}).get(
                        "E24_INPUT"))
                chk = [(f"[{n[4:]}] {l}", o, i) for l, o, i in sc["check"](r, ctx)]
            else:
                chk = positive_checks(r, allow_budget_skips=a.allow_budget_skips, require_site=not a.no_site)
            r.update(kind="revalidate", checks=chk, statuses={x: y["status"] for x, y in arms_of(r["R"]).items()})
            results.append(r)
        sd = Path(a.summary_dir or Path(a.revalidate[0]).parent)
        sd.mkdir(parents=True, exist_ok=True)
        S = summarize(results, sd / "e24_revalidate_summary.json", sd / "e24_revalidate_summary.md")
        for r in S["runs"]:
            for c in r["checks"]:
                say(f"{r['name']:24} {'PASS' if c['ok'] else 'FAIL'} {c['label']}" + ("" if c["ok"] else
                                                                                       ": " + json.dumps(c["info"], default=str)[:500]))
        say(f"{'ALL OK' if S['all_ok'] else 'NOT OK'} {S['passed']}/{S['checks']} | {sd / 'e24_revalidate_summary.md'}")
        return 0 if S["all_ok"] else 1
    if a.list:
        print("variants:", ", ".join(VARIANTS))
        print("negatives:", ", ".join(scenarios(a.time_scale)))
        return 0
    variants = [v for v in a.variants.split(",") if v]
    negs = list(scenarios(a.time_scale)) if a.negatives == "all" else [] if a.negatives == "none" else a.negatives.split(",")
    bad = [v for v in variants if v not in VARIANTS] + [n for n in negs if n not in scenarios(a.time_scale)]
    if bad:
        ap.error(f"unknown: {bad}")
    S, work = run_all(variants, negs, scale=a.time_scale, work=a.work, keep=a.keep,
                      build="mirror" if a.mirror_build else "none" if (a.no_build or a.nb_dir) else "inplace", nb_dir=a.nb_dir,
                      input_root=Path(a.input) if a.input else None, site=not a.no_site, timeout_min=a.timeout_min,
                      budget_slack=a.budget_slack, allow_budget_skips=a.allow_budget_skips, summary_dir=a.summary_dir,
                      ports=not a.no_port_isolation)
    for r in S["runs"]:
        fails = [c for c in r["checks"] if not c["ok"]]
        say(f"{r['name']:28} {'OK' if not fails else 'FAIL'}  {len(r['checks']) - len(fails)}/{len(r['checks'])}")
        for c in fails:
            say(f"    FAIL {c['label']}: {json.dumps(c['info'], default=str)[:600]}")
    say(f"{'ALL OK' if S['all_ok'] else 'NOT OK'} {S['passed']}/{S['checks']} | summary {Path(a.summary_dir or work) / 'e24_local_summary.md'}")
    return 0 if S["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
