"""E23 CPU kit (rev 3h): the three E23 GPU notebooks to run next (TARGET_A, TARGET_B, FULL) run against a fake vLLM server on a Kaggle CPU
kernel, pinned image (Linux semantics: TIME_WAIT, killpg, ps --sid), with the REAL /kaggle/input mounts of every E23
input (mount discovery, manifest sizes, kit sha256, identity). Plus the lab unit tests and a competition-RERUN check
(fallback submission written, no exception, nothing else run). Kernel foysalemonshanto/q27-e23-cpu-kit (Build/cpu_kernel_e23/)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E23 = HERE.parents[1]
NB = {v: "".join("".join(c["source"]) + "\n\n" for c in json.load(open(HERE.parent / f"Q27-XHIGH-E23_SIX_LEVERS_{v}.ipynb",
                                                                       encoding="utf-8"))["cells"] if c["cell_type"] == "code")
      for v in ("TARGET_A", "TARGET_B", "FULL")}
BAKE = {"TARGET_A": ("TARGET", "A"), "TARGET_B": ("TARGET", "B"), "FULL": ("FULL", "ALL")}


def bake(code, v):
    m, p = BAKE[v]
    return code.replace('os.environ.get("Q27X_E23_MODE", "FULL")', f'os.environ.get("Q27X_E23_MODE", "{m}")', 1)                .replace('os.environ.get("Q27X_E23_PART", "ALL")', f'os.environ.get("Q27X_E23_PART", "{p}")', 1)


for v in NB:                     # the kit carries FULL only; the other two are FULL with the two baked lines changed
    assert bake(NB["FULL"], v) == NB[v], v
FAKE = (E23 / "src" / "harness" / "fake_vllm_server.py").read_text(encoding="utf-8")       # E23 copy: varied words
LAB = (E23 / "src" / "e23_lab.py").read_text(encoding="utf-8")
GATES = (E23 / "src" / "harness" / "gate_tests.py").read_text(encoding="utf-8")     # rev 3d: injected-fault gate tests
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

CODE = r'''# E23 CPU kit: the three GPU notebooks against a fake vLLM server, real /kaggle/input mounts, Linux semantics
import json, os, re, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path
T0 = time.time()
W, T = Path("/kaggle/working"), Path("/tmp/e23kit")
T.mkdir(parents=True, exist_ok=True)
NBF = ''' + repr(NB["FULL"]) + r'''
NB = {"FULL": NBF, "TARGET_A": NBF.replace('os.environ.get("Q27X_E23_MODE", "FULL")', 'os.environ.get("Q27X_E23_MODE", "TARGET")', 1)
      .replace('os.environ.get("Q27X_E23_PART", "ALL")', 'os.environ.get("Q27X_E23_PART", "A")', 1)}
NB["TARGET_B"] = NB["TARGET_A"].replace('os.environ.get("Q27X_E23_PART", "A")', 'os.environ.get("Q27X_E23_PART", "B")', 1)
(T / "fake_vllm_server.py").write_text(''' + repr(FAKE) + r''')
(T / "e23_lab.py").write_text(''' + repr(LAB) + r''')
(T / "gate_tests.py").write_text(''' + repr(GATES) + r''')
REP = {"checks": {}, "runs": {}, "part": "__KP__"}
KP = "__KP__"                      # p1 unit + TARGET_A | p2 TARGET_B + rerun | p3 FULL | p4 Merlin GCC 13 | p5 Merlin GCC 11
DO_UNIT = KP == "p1"
SMOKES = {"p1": ("TARGET_A",), "p2": ("TARGET_B",)}.get(KP, ())
DO_RERUN = KP == "p2"
DO_FULL = KP == "p3"
DO_MERLIN = KP in ("p4", "p5")
FI_TAGS = {"p4": ("gcc13",), "p5": ("gcc11",)}.get(KP, ())


def walk(d, path=""):
    """Every measurement window in a results tree (a dict with status and tok_s)."""
    if isinstance(d, dict):
        if "status" in d and "tok_s" in d:
            yield path, d["status"]
        for k, v in d.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from walk(v, f"{path}[{i}]")


def run(name, code_or_cmd, env, timeout, is_code=True):
    t = time.time()
    if is_code:
        (T / f"{name}.py").write_text(code_or_cmd)
        cmd = [sys.executable, str(T / f"{name}.py")]
    else:
        cmd = code_or_cmd
    with open(T / f"{name}.log", "w") as f:
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env={**os.environ, **env}, cwd=str(T))
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    txt = (T / f"{name}.log").read_text(errors="replace")
    REP["runs"][name] = {"rc": rc, "sec": round(time.time() - t, 1), "tracebacks": txt.count("Traceback"), "tail": txt[-1500:]}
    print(f"[{time.time() - T0:7.1f}s] {name}: rc {rc} tracebacks {txt.count('Traceback')}", flush=True)
    return rc, txt


# 1. lab unit tests on Linux (port reuse right after a kill: the TIME_WAIT case; compare(); steady; capacity; replicas)
UNIT = r"""
import sys, json, time, os
sys.path.insert(0, "/tmp/e23kit")
import e23_lab as L
FAKE = "/tmp/e23kit/fake_vllm_server.py"
env = dict(os.environ, FAKE_SPEED="300")
def srv(name, port, seqs=8):
    return L.Server(name, [sys.executable, FAKE, "fp8-model", "--port", str(port), "--max-num-seqs", str(seqs), "--tensor-parallel-size", "1",
                           "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3})], env, port, "/tmp/e23kit").start()
out = {}
a = srv("u1", 18121); out["ready1"] = a.wait_ready(60)[0]
P = [[100 + i] * 50 for i in range(8)]
g = L.greedy(18121, P[:2], n_tok=20); out["compare_same"] = L.compare(g, L.greedy(18121, P[:2], n_tok=20))["status"]
bad = [dict(r, tokens=r["tokens"][:5] + [999] + r["tokens"][6:]) for r in g]
out["compare_fail"] = L.compare(g, bad)["status"]
s = L.steady(18121, P, 4, secs=10, max_new=200000); out["steady"] = s["status"]
a.kill(); t = time.time()
b = srv("u2", 18121); out["port_reuse_after_kill_ready"] = b.wait_ready(60)[0]; out["port_reuse_sec"] = round(time.time() - t, 1)
c = srv("u3", 18122, seqs=2); c.wait_ready(60)
out["capacity"] = L.steady(18122, P, 4, secs=10, max_new=200000)["status"]
m = L.steady_multi([(18121, P, 3), (18122, P, 2)], secs=10, max_new=200000); out["multi"] = m["status"]
out["rss_mb"] = b.rss_mb()
b.kill(); c.kill()
out["killed_dead"] = (not b.alive()) and (not c.alive())
print("UNIT " + json.dumps(out))
"""
rc, txt = run("unit", UNIT, {}, 900) if DO_UNIT else (0, "")
if DO_UNIT:                       # rev 3d: the real gate functions of the FULL notebook against injected faults
    (T / "nb_gates_FULL.py").write_text(NB["FULL"])
    # rev 3h: the top-level kit file next to the manifest (the dataset also unpacks an older copy under p5_src/)
    _kb = next(Path("/kaggle/input/datasets/foysalemonshanto/q27-e23-sm89-kernels").rglob("q27_e23_kernels_manifest.json")).parent         / "bench_host_ar.py"
    _rc, _txt = run("gates", [sys.executable, str(T / "gate_tests.py"), str(T / "nb_gates_FULL.py"), str(T / "e23_lab.py"),
                              str(_kb)], {}, 600, is_code=False)
    _m = re.search(r"^GATES (.*)$", _txt, re.M)
    REP["checks"]["gates"] = json.loads(_m.group(1)) if _m else {"all_ok": False, "tail": _txt[-1500:]}
m = re.search(r"^UNIT (.*)$", txt, re.M)
REP["checks"]["unit"] = u = json.loads(m.group(1)) if m else None
REP["checks"]["unit_ok"] = bool(u) and u.get("compare_same") == "EXACT" and u.get("compare_fail") == "FAIL_HIGH_MARGIN" and \
    u.get("steady") == "STEADY" and u.get("port_reuse_after_kill_ready") and u.get("capacity") == "CAPACITY_LIMITED" and \
    u.get("multi") == "AGGREGATE_STEADY" and u.get("killed_dead") and isinstance(u.get("rss_mb"), float)

# 2. TARGET A or TARGET B (ports 8100-8103 vs 8200 / 8300), real mounts, fake server
FENV = {"E23_FAKE_SERVER": str(T / "fake_vllm_server.py"), "FAKE_SPEED": "300"}
procs = {}
for v in SMOKES:
    (T / f"nb_{v}.py").write_text(NB[v])
    os.makedirs(f"/tmp/e23kit/tmp_{v}", exist_ok=True)
    lf = open(T / f"nb_{v}.log", "w")
    procs[v] = (subprocess.Popen([sys.executable, str(T / f"nb_{v}.py")], stdout=lf, stderr=subprocess.STDOUT, cwd=str(T),
                                 env={**os.environ, **FENV, "E23_OUT": f"/tmp/e23kit/out_{v}", "E23_TMP": f"/tmp/e23kit/tmp_{v}"}), lf)
for v, (p, lf) in procs.items():
    try:
        rc = p.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        p.kill(); rc = -9
    lf.close()
    txt = (T / f"nb_{v}.log").read_text(errors="replace")
    REP["runs"][v] = {"rc": rc, "tracebacks": txt.count("Traceback"), "tail": txt[-2500:]}
    rj = Path(f"/tmp/e23kit/out_{v}/e23_results.json")
    R = json.loads(rj.read_text()) if rj.is_file() else {}
    REP["checks"][v] = {"inputs": R.get("inputs"), "identity": R.get("identity"), "prompts": R.get("prompts"),
                        "steps": {f"{s}/{n}": r.get("status") for s, st in (R.get("sections") or {}).items() for n, r in st.items()
                                  if isinstance(r, dict)},
                        "windows": dict(walk(R.get("sections") or {})), "errors": R.get("errors"),
                        "report_md": R.get("report_md")}
    print(v, "rc", rc, "tracebacks", txt.count("Traceback"), flush=True)

# 3. competition RERUN: fallback submission only, no exception, no server
rc, txt = run("rerun_FULL", NB["FULL"], {"KAGGLE_IS_COMPETITION_RERUN": "1", "E23_OUT": "/tmp/e23kit/out_rerun"}, 600) \
    if DO_RERUN else (0, "")
sub = Path("/kaggle/working/submission.json")
ok_sub = False
if sub.is_file():
    s = json.loads(sub.read_text())
    tasks = json.load(open(next(Path("/kaggle/input").rglob("arc-agi_test_challenges.json"))))
    ok_sub = set(s) == set(tasks) and all(len(s[k]) == len(tasks[k]["test"]) and all(set(a) == {"attempt_1", "attempt_2"} for a in s[k]) for k in s)
    sub.unlink()                  # a CPU kit leaves no submission behind
REP["checks"]["rerun"] = {"rc": rc, "tracebacks": txt.count("Traceback"), "submission_format_ok": ok_sub}

# 4. FULL with a 30-minute budget: order and budget guard on Linux (the 90-minute run is the local harness)
os.makedirs("/tmp/e23kit/tmp_FULL", exist_ok=True)
rc, txt = run("nb_FULL", NB["FULL"], {**FENV, "E23_OUT": "/tmp/e23kit/out_FULL", "E23_TMP": "/tmp/e23kit/tmp_FULL",
                                      "E23_BUDGET_SEC": "1800"}, 3000) if DO_FULL else (0, "")
rj = Path("/tmp/e23kit/out_FULL/e23_results.json")
R = json.loads(rj.read_text()) if rj.is_file() else {}
REP["checks"]["FULL"] = {"order": R.get("order"), "steps": {f"{s}/{n}": r.get("status") for s, st in (R.get("sections") or {}).items()
                                                             for n, r in st.items() if isinstance(r, dict)}, "minutes": R.get("minutes")}

# 5. Merlin, for real (rev 3c: fake mode skips it): the offline install on this image, the serve-time compiler probe
# (GCC 13 from the payload vs the system GCC 11: C++20 CUDA compile + dlopen, and a FlashInfer JIT build inside the
# Merlin venv, which is Merlin's first-start compile path), and launch.build() with the notebook's arguments
MP = {}
try:
    if not DO_MERLIN:
        raise RuntimeError("SKIP_PART")
    pay = next(Path("/kaggle/input/models/foysalemonshanto/e23-merlin-env").rglob("merlin_offline_tar.bin"))
    t = time.time()
    shutil.rmtree("/tmp/mpay", ignore_errors=True)
    os.makedirs("/tmp/mpay")
    with tarfile.open(pay) as tf_:
        tf_.extractall("/tmp/mpay", filter="data")
    MP["extract_sec"] = round(time.time() - t)
    rc, txt = run("merlin_install", ["bash", "/tmp/mpay/merlin_offline/install_merlin_offline.sh", "/tmp/mpay/merlin_offline",
                                     "/tmp/merlin_off"], {}, 5400, is_code=False)
    kv = dict(re.findall(r"^(MERLIN_[A-Z]+)=(.*)$", txt, re.M))
    MP.update(install_rc=rc, install_ok="INSTALL OK" in txt, kv=kv)
    shutil.rmtree("/tmp/mpay", ignore_errors=True)
    venv = kv["MERLIN_VENV"]
    py = venv + "/bin/python"
    site = subprocess.run([py, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], capture_output=True,
                          text=True).stdout.strip()
    ch = site + "/nvidia/cu13"
    nvcc = ch + "/bin/nvcc"
    lib13 = Path(kv["MERLIN_CC"]).parents[1] / "lib"
    MP["nvcc"] = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout[-160:]
    MP["lib13_libstdcxx"] = sorted(p.name for p in lib13.glob("libstdc++.so*"))
    (T / "p20.cu").write_text('#include <concepts>\ntemplate <std::integral I> __global__ void k(I* x) { x[0] += 1; }\n'
                              'template __global__ void k<int>(int*);\nextern "C" int probe() { return 20; }\n')
    for tag, cxx, ldp in (("gcc13", kv["MERLIN_CXX"], str(lib13)), ("gcc11", "/usr/bin/g++", "")):
        so = T / f"p20_{tag}.so"
        r = subprocess.run([nvcc, "-std=c++20", "-arch=sm_89", "-shared", "-Xcompiler", "-fPIC", "-ccbin", cxx, str(T / "p20.cu"),
                            "-o", str(so)], capture_output=True, text=True, timeout=600)
        ld = subprocess.run([sys.executable, "-c", f"import ctypes;print(ctypes.CDLL('{so}').probe())"], capture_output=True,
                            text=True, env={**os.environ, "LD_LIBRARY_PATH": ldp}) if r.returncode == 0 else None
        MP[f"cxx20_{tag}"] = {"rc": r.returncode, "err": (r.stderr or "")[-600:], "dlopen": ld.stdout.strip() if ld else None,
                              "dlopen_err": ld.stderr[-300:] if ld else None}
    (T / "fi_probe.py").write_text(
        "import inspect, time, torch\n"
        "from flashinfer.jit import gen_single_decode_module as g\n"
        "sig = inspect.signature(g)\n"
        "kw = dict(dtype_q=torch.float16, dtype_kv=torch.float16, dtype_o=torch.float16, head_dim_qk=256, head_dim_vo=256,\n"
        "          head_dim=256, pos_encoding_mode=0, use_sliding_window=False, use_logits_soft_cap=False)\n"
        "print('SIG', list(sig.parameters), flush=True)\n"
        "spec = g(**{k: v for k, v in kw.items() if k in sig.parameters})\n"
        "t = time.time(); spec.build(); print('FI_BUILD_OK', round(time.time() - t), flush=True)\n")
    for tag, cc, cxx, ldp in (("gcc13", kv["MERLIN_CC"], kv["MERLIN_CXX"], str(lib13)), ("gcc11", "/usr/bin/gcc", "/usr/bin/g++", "")):
        if tag not in FI_TAGS:
            continue
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(CC=cc, CXX=cxx, CUDAHOSTCXX=cxx, CUDA_HOME=ch, PATH=f"{venv}/bin:{ch}/bin:" + env.get("PATH", ""),
                   FLASHINFER_CUDA_ARCH_LIST="8.9", FLASHINFER_WORKSPACE_BASE=f"/tmp/fi_{tag}", LD_PRELOAD=kv.get("MERLIN_SHIM", ""),
                   LIBRARY_PATH="/usr/local/cuda-12.8/lib64/stubs", MAX_JOBS="4")
        if ldp:
            env["LD_LIBRARY_PATH"] = ldp
        try:
            r = subprocess.run([py, str(T / "fi_probe.py")], capture_output=True, text=True, timeout=2400, env=env)
            o_ = r.stdout + r.stderr
            MP[f"flashinfer_jit_{tag}"] = {"rc": r.returncode, "ok": "FI_BUILD_OK" in o_, "tail": o_[-1500:]}
        except subprocess.TimeoutExpired:
            MP[f"flashinfer_jit_{tag}"] = {"rc": -9, "ok": False, "tail": "timeout 2400 s"}
    sys.path.insert(0, kv["MERLIN_TREE"] + "/tools/merlin")
    import launch as ML
    ck = next(Path("/kaggle/input/models/foysalemonshanto/qwen3-8-27b-w4a16-merlin").rglob("q27_e23_p1_manifest.json")).parent
    plan = ML.build("27b", "27b-tp1-sm89-24g", str(ck), {**os.environ, "GPUS": "0", "PORT": "8100", "KV_BYTES": "8055029760",
                                                          "P2P": "off", "VENV": venv},
                    extra=["--served-model-name", "q27", "--max-model-len", "131072", "--port", "8100", "--max-num-seqs", "4"],
                    checkpoint="27b", status="untested")
    MP["launch_argv_tail"] = plan.argv()[-30:]
    dp = getattr(ML, "distributed_ports", None)
    MP["distributed_ports_8101"] = repr(dp(8101)) if dp else None
except Exception as e:
    MP["error"] = f"{type(e).__name__}: {e}"[:800]
    MP["tb"] = traceback.format_exc()[-1500:]
REP["checks"]["merlin_real"] = MP
print("MERLIN", json.dumps({k: v for k, v in MP.items() if k != "kv"}, default=str)[:4000], flush=True)

# verdict: no tracebacks in any notebook run, inputs found on the real mounts, unit tests, rerun format, no step that
# errored (CPU-expected: P4 / P5 benches without CUDA, PP4 NO_GO and EAGLE SKIPPED on the fake server), the real
# Merlin install, and a GCC 13 C++20 CUDA build that loads
inp = (REP["checks"].get("TARGET_A") or {}).get("inputs") or {}
BAD = ("ERROR", "BODY_ERROR", "SERVER_DIED", "FAIL_GPU_RELEASE", "LAUNCH_FAILED", "PATCH_FAILED", "NO_GO_GPU_FAULT", "FAIL_CHECKPOINT_SHA")
REP["checks"]["bad_steps"] = {v: {k: st for k, st in ((REP["checks"].get(v) or {}).get("steps") or {}).items() if st in BAD}
                              for v in SMOKES}
inp = inp or (REP["checks"].get("TARGET_B") or {}).get("inputs") or {}
ok = []                                                      # per-part verdict: only what this part ran
if DO_UNIT:
    ok.append(bool(REP["checks"]["unit_ok"]))
    ok.append((REP["checks"].get("gates") or {}).get("all_ok") is True)
for v in list(SMOKES) + (["rerun_FULL"] if DO_RERUN else []) + (["nb_FULL"] if DO_FULL else []):
    ok.append(REP["runs"].get(v, {}).get("rc") == 0 and REP["runs"].get(v, {}).get("tracebacks") == 0)
if SMOKES:
    ok += [bool(inp.get("merlin_files_ok")), all((inp.get("kits_sha_ok") or {}).values()), not any(REP["checks"]["bad_steps"].values()),
           all(inp.get(k) not in (None, "None") for k in ("FP8", "INT4", "MERLIN", "MERLIN_PAY", "EAGLE_PAY", "EAGLE_DRAFT", "KITS", "ARC"))]
if DO_RERUN:
    ok.append(ok_sub)
if DO_MERLIN:
    ok += [MP.get("install_ok") is True, (MP.get("cxx20_gcc13") or {}).get("dlopen") == "20"]
REP["all_ok"] = bool(ok) and all(ok)
REP["sec"] = round(time.time() - T0)
(W / "e23_cpu_kit_report.json").write_text(json.dumps(REP, indent=1, default=str))
with tarfile.open(W / "e23_cpu_kit_logs_tar.bin", "w") as tf:
    for p in sorted(T.rglob("*")):
        if p.is_file() and (p.suffix in (".log", ".json") and p.stat().st_size < 50e6):
            tf.add(p, str(p.relative_to(T)), recursive=False)
print(json.dumps({k: REP["checks"].get(k) for k in ("unit", "unit_ok", "rerun")}, default=str)[:3000])
print("inputs:", json.dumps(inp)[:1500])
print("ALL OK" if REP["all_ok"] else "NOT OK", "|", REP["sec"], "s", flush=True)
'''
for KPART in ("p1", "p2", "p3", "p4", "p5"):
  code = CODE.replace("__KP__", KPART)
  compile(code, "<kit>", "exec")
  nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
  KD = HERE / f"cpu_kernel_e23_{KPART}"
  KD.mkdir(exist_ok=True)
  slug = f"q27-e23-cpu-kit-{KPART}"
  (KD / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
  (KD / "kernel-metadata.json").write_text(json.dumps({
    "id": f"foysalemonshanto/{slug}", "title": f"Q27 E23 CPU Kit {KPART.upper()}", "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": ["foysalemonshanto/q27-e23-sm89-kernels", "foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"],
    "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
    "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1",
                      "foysalemonshanto/qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1",
                      "foysalemonshanto/qwen3-8-27b-w4a16-merlin/Transformers/default/1",
                      "foysalemonshanto/e23-merlin-env/Other/v1/1",
                      "foysalemonshanto/e23-eagle4-env/Other/default/1",
                      "foysalemonshanto/qwen3-8-27b-eagle4/Transformers/default/1"],
    "docker_image": IMAGE, "machine_shape": "None"}, indent=2), encoding="utf-8", newline="\n")
  print("kit kernel", KD.name, len(code), "chars", f"{len(json.dumps(nb)) / 1024:.0f} kB")
