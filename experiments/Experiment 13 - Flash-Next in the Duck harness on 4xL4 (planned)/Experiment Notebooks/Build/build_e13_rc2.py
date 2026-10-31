"""E13 RC2 (user 2026-10-31): Qwen3.8-Flash-Next on Kaggle 4xL4 with the Merlin engine (vLLM 0.32.0 fork), a correctness-
first diagnostic. Design: Experiment Review/MERLIN_FN_L4_SYNTHESIS_2026-10-31.md (4 investigations + 4 verifiers) and the
RC1 result (Experiment Results/E13_RC1_GPU_run1_2026-10-31.md: stock vLLM 0.32 + Intel W4A16 is correct on 4xL4; SGLang
EP4 GPTQ-Marlin MoE is wrong).

Reuse, verbatim from Experiment 23 (Kaggle-proven on 4xL4): the lab module src/e23_lab.py (Server, greedy, chat probes,
teacher-forced, steady windows), and from its notebook builder the run() helper, the -lcuda shim, merlin_install() (the
offline payload with the glibc 2.35 fix), the step / with_servers runner, oracle / tforced / levels.

Flash-Next specific (new here): checkpoint merge (5 kernel-output parts + the 10 FP8 PLE shards of
dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple, every size checked, every sha256 checked in the background before the
first launch), Merlin launch.build("flash-next", "fn-quad-24g-sm89", kv=bf16|fp8) with the L4 KV pin, the checkpoint's own
template at reasoning_effort xhigh, Merlin's host_phase checks (recorded), the arms below, a 60k-token generation check,
and a stock-vLLM fallback (Intel W4A16, TP4+EP, with the shim) only when Merlin cannot serve.

Run: python build_e13_rc2.py  ->  ../Q27-XHIGH-E13_FLASHNEXT_RC2.ipynb
"""
import hashlib
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
E13 = HERE.parents[1]
E23 = Path(r"D:/ARC AGI 2/experiments/Experiment 23 - Six levers vLLM 0.32 (planned)")
E23_BUILDER = (E23 / "Experiment Notebooks" / "Build" / "build_e23_notebook.py").read_text(encoding="utf-8").replace("\r\n", "\n")
LAB = (E23 / "src" / "e23_lab.py").read_text(encoding="utf-8").replace("\r\n", "\n")
PARTS = json.loads((E13 / "prep" / "merlin_fn_ckpt" / "kernels" / "parts.json").read_text(encoding="utf-8"))
OUT_NB = HERE.parent / "Q27-XHIGH-E13_FLASHNEXT_RC2.ipynb"
PUSHED = set()   # shas of RC2 versions that ran on Kaggle (never overwrite one; rebuilds go to *_REBUILD)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")


def cut(text, start, end, label):
    assert text.count(start) == 1, f"{label}: start x{text.count(start)}"
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j]


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: x{text.count(old)}"
    return text.replace(old, new)


E23_CELL2 = cut(E23_BUILDER, "CELL2 = r'''", "\n'''\n\nCELL3", "e23 cell2")
E23_CELL4 = cut(E23_BUILDER, "CELL4 = r'''", "\n'''\n\nCELL5", "e23 cell4")
E23_CELL5 = cut(E23_BUILDER, "CELL5 = r'''", "\n'''\n", "e23 cell5")
GATES = cut(E23_CELL5, 'CLEAN = ("EXACT", "NEAR_TIE")', "\n# rev 3f (GPT rev 3e audit)", "CLEAN + _windows + oracle_gate + gated")
RUN_AND_SHIM = cut(E23_CELL2, "def run(name, cmd", "if FAKE:\n    STOCK = ", "run + shim")
MERLIN_INSTALL = cut(E23_CELL2, "def prefetch_bg(paths, tag):", "def merlin_server(", "prefetch + merlin_install")
STEP_RUNNER = cut(E23_CELL4, "WIN = float(", "\nsave()", "step runner")
STEP_RUNNER = once(STEP_RUNNER, 'WIN = float(os.environ.get("E23_WIN_SEC") or {"SMOKE": 30.0, "TARGET": 60.0, "FULL": 120.0}[MODE])',
                   'WIN = float(os.environ.get("E23_WIN_SEC") or 90.0)', "window")

PLE = PARTS["ple_from_overlay"]          # name -> [size, sha256] (Merlin checkpoint.sha256 at merlin-v3.0.0)

MD0 = """# E13 RC2: Qwen3.8-Flash-Next on 4×L4 with Merlin (vLLM 0.32.0 fork): correctness first

RC1 (2026-10-31) showed stock vLLM 0.32 + Intel W4A16 gives correct Flash-Next output on 4×L4, while SGLang's EP4
GPTQ-Marlin MoE is wrong. RC2 serves the Merlin W4A16 checkpoint (INT4 experts, INT8 GDN, FP8 PLE in host RAM) with the
Merlin engine's 4×24 GB sm_89 profile (TP2×PP2+EP, MTP K3) and measures, in order:

- **A0** environment, checkpoint merge + sha256, Merlin offline install;
- **A1** BF16-KV boot: probes, greedy references, teacher-forced 40k reference, KV census;
- **A3** FP8-KV boot (the 2× KV pool): the same probes, FP8 vs BF16 parity gate, then a throughput ladder (1/2/4/5/8 streams,
  40k prompts) and two 60k-token generations (finite, no loops, MTP acceptance);
- **F** stock vLLM fallback (Intel W4A16, TP4+EP, -lcuda shim) only if Merlin cannot serve.

Commit run only; a competition rerun writes a fallback submission. Every measurement is a recorded counter; anything
missing is INCONCLUSIVE, never PASS.
"""

CELL1 = r'''# ==================================================================== CELL 1 — config, inputs, lab module
import os, sys, json, math, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
from pathlib import Path
T_START = time.time()
MODE = "FULL"
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # two cold Merlin boots + ladder + 90k/116k census + long rows
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/rc2"))
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = OUT / "traces"                                           # memory rule: the full text of every long generation
TRACES.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"what": "E13 RC2 Flash-Next on Merlin", "budget_min": BUDGET // 60, "max_len": MAX_LEN,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()


def save():
    with _SAVE_LOCK:
        tmp = OUT / "rc2_results.json.tmp"
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(OUT / "rc2_results.json")


def left():
    return HARD_STOP - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


def find(pattern, root=INP, must=True):
    hits = sorted(Path(root).rglob(pattern)) if Path(root).exists() else []
    if not hits and must:
        raise FileNotFoundError(f"{pattern} not under {root}: attach the input")
    return hits[0] if hits else None


def write_fallback_submission():
    tp = find("arc-agi_test_challenges.json", must=False)
    if tp:
        tasks = json.load(open(tp))
        sub = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]] for k, v in tasks.items()}
        Path(os.environ.get("RC2_SUBMISSION", "/kaggle/working/submission.json")).write_text(json.dumps(sub))


if RERUN:
    write_fallback_submission()
    print("E13 RC2 diagnostic: competition rerun -> fallback submission.json only")
else:
    LAB_SRC = __LAB__
    (TMP / "e23_lab.py").write_text(LAB_SRC)
    sys.path.insert(0, str(TMP))
    import e23_lab as L
    REQ_LOG = TRACES / "requests.jsonl"                           # memory rule: the full text of EVERY request
    _stream_run = L.Stream.run

    def _traced_run(self):
        try:
            _stream_run(self)
        finally:
            try:
                r = self.row()
                pr = self.body.get("prompt") or []
                r.update(port=self.port, sampling={k: v for k, v in self.body.items() if k != "prompt"}, prompt_len=len(pr),
                         prompt_sha=hashlib.sha256(json.dumps(pr).encode()).hexdigest()[:16],
                         ttft=round(self.t_first - self.t_start, 2) if self.t_first and self.t_start else None,
                         sec=round(self.t_end - self.t_start, 2) if self.t_end and self.t_start else None, t=time.time())
                with _SAVE_LOCK, open(REQ_LOG, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            except Exception:
                pass
    L.Stream.run = _traced_run
    PLE_EXPECTED = json.loads(r"""__PLE__""")                     # 10 FP8 PLE shards: [size, sha256]
    PARTS = {}
    for k in range(1, 6):                                           # the five checkpoint parts (kernel outputs)
        mf = find(f"q27_fn_merlin_part{k}_manifest.json", must=False)
        PARTS[k] = (mf.parent, json.loads(mf.read_text())) if mf else None
    _ple0 = find("model-plefp8-00000.safetensors", INP / "models" / "dfranzen", must=False)
    PLE_DIR = _ple0.parent if _ple0 else None
    MERLIN_PAY = find("merlin_offline_tar.bin", INP / "models" / "foysalemonshanto" / "e23-merlin-env", must=False)
    _intel = find("quantization_config.json", INP / "models" / "dfranzen" / "intel-qwen3.8-flash-next-w4a16-autoround", must=False)
    INTEL = _intel.parent if _intel else None
    WHEELHOUSE = find("install_wheelhouse.py", INP / "datasets" / "foysalemonshanto" / "vllm-0-32-0-cu13-wheelhouse", must=False)
    ARC = find("arc-agi_evaluation_challenges.json")
    MERLIN_OK = all(PARTS.values()) and PLE_DIR is not None and MERLIN_PAY is not None
    R["inputs"] = {"parts": {k: (str(v[0]) if v else None) for k, v in PARTS.items()}, "ple_dir": str(PLE_DIR),
                   "merlin_pay": str(MERLIN_PAY), "intel": str(INTEL), "wheelhouse": str(WHEELHOUSE), "arc": str(ARC),
                   "merlin_inputs_ok": MERLIN_OK}
    ABORT = {"why": None}
    NVSMI_OK = bool(L.gpu_mem()) and not FAKE
    R["setup"]["gpus_at_start"] = L.gpu_mem()
    if not FAKE and len(R["setup"]["gpus_at_start"] or []) != 4:
        R["errors"].append(f"expected 4 GPUs, nvidia-smi shows {R['setup']['gpus_at_start']}")
        save()
        raise RuntimeError("GPU_PREFLIGHT_FAILED: " + repr(R["setup"]["gpus_at_start"]))
    L.start_watchdog(HARD_STOP, lambda: ABORT.update(why="HARD_STOP reached: watchdog killed live servers"))
    log("RC2 | budget", BUDGET // 60, "min |", json.dumps(R["inputs"])[:900])
    save()
'''

CELL2 = r'''# ==================================================================== CELL 2 — host facts, checkpoint merge, engines
''' + RUN_AND_SHIM + r'''

def host_facts():
    """A0: what the Kaggle box gives us (glibc, driver, GPUs, PCIe, peer access, cgroup memory, /dev/shm, disk)."""
    f = {}
    for k, cmd in (("glibc", "ldd --version | head -1"), ("uname", "uname -a"),
                   ("nvidia_smi", "nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader"),
                   ("cuda_version", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"), ("shm", "df -h /dev/shm | tail -1"),
                   ("tmp_disk", f"df -h {TMP} | tail -1"), ("nproc", "nproc"),
                   ("cgroup_max", "cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes"),
                   ("cgroup_cur", "cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes"),
                   ("meminfo", "grep -E 'MemTotal|MemAvailable' /proc/meminfo"), ("ulimit_l", "ulimit -l"),
                   ("omp", "echo OMP_NUM_THREADS=$OMP_NUM_THREADS"), ("topo", "nvidia-smi topo -m; nvidia-smi topo -p2p r"),
                   ("pcie_now", "nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv,noheader")):
        try:
            f[k] = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout.strip()[:600]
        except Exception as e:
            f[k] = repr(e)[:200]
    try:
        import torch
        n = torch.cuda.device_count()
        f["peer_access"] = [[torch.cuda.can_device_access_peer(i, j) if i != j else None for j in range(n)] for i in range(n)]
    except Exception as e:
        f["peer_access"] = repr(e)[:200]
    return f


def host_mem():
    """MiB. avail_mib = max - (current - inactive_file), Merlin's own rule (RC1's guard counted page cache as used and
    skipped arm 3 with MemAvailable 55 GB); /dev/shm counts against the cgroup."""
    out = {}
    try:
        cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
        mx = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        st = dict(l.split() for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines() if len(l.split()) == 2)
        out.update(cgroup_cur_mib=cur >> 20, cgroup_max_mib=None if mx == "max" else int(mx) >> 20,
                   **{k + "_mib": int(st[k]) >> 20 for k in ("anon", "file", "shmem", "inactive_file") if k in st})
        if mx != "max":
            out["avail_mib"] = (int(mx) - (cur - int(st.get("inactive_file", 0)))) >> 20
    except Exception:
        pass
    try:
        for l in open("/proc/meminfo"):
            if l.startswith(("MemAvailable", "MemTotal")):
                out[l.split(":")[0] + "_mib"] = int(l.split()[1]) >> 10
    except Exception:
        pass
    try:
        out["dev_shm_used_mib"] = shutil.disk_usage("/dev/shm").used >> 20
    except Exception:
        pass
    return out


TELEMETRY = OUT / "telemetry.jsonl"


def _telemetry():
    """Host RAM + GPU memory every 10 s for the whole run (Merlin's floor ~69 GiB inside a ~173 GiB cgroup)."""
    while True:
        try:
            with open(TELEMETRY, "a") as fh:
                fh.write(json.dumps({"t": round(time.time() - T_START, 1), "host": host_mem(), "gpu": L.gpu_mem()}) + "\n")
        except Exception:
            pass
        time.sleep(10)


threading.Thread(target=_telemetry, daemon=True).start()


def shm_cleanup():
    """Between boots: what the dead server left in /dev/shm (PLE file-system-shared storages, torch / psm segments)."""
    out = {"host_before": host_mem(), "removed": [], "kept": []}
    try:
        for p in sorted(Path("/dev/shm").iterdir()):
            try:
                sz = p.stat().st_size >> 20
            except Exception:
                sz = None
            if re.match(r"(torch_|psm_|vllm|ple|nccl|sem\.mp-|cuda\.shm)", p.name, re.I):
                try:
                    p.unlink() if not p.is_dir() else shutil.rmtree(p)
                    out["removed"].append([p.name, sz])
                except Exception as e:
                    out["kept"].append([p.name, sz, repr(e)[:80]])
            else:
                out["kept"].append([p.name, sz])
    except Exception as e:
        out["error"] = repr(e)[:200]
    out["host_after"] = host_mem()
    out["removed"], out["kept"] = out["removed"][:60], out["kept"][:60]
    return out


class MServer(L.Server):
    """Merlin passes --shutdown-timeout 60 and the PLE transport unlinks its /dev/shm storages only on a graceful
    teardown: SIGTERM, wait 75 s, then E23's kill (SIGTERM + 40 s + SIGKILL of the whole session)."""
    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 15) if hasattr(os, "killpg") else self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=75)
            except Exception:
                pass
        super().kill()


def boot_gate():
    """Merlin's pin leaves ~0.9 GiB free per card: every boot needs near-empty GPUs (< 150 MiB, not E23's 1024)."""
    g = L.wait_gpu_release(300, fake=bool(FAKE), gpus="0,1,2,3", free_mib=150)
    return {"ok": g, "gpus": L.gpu_mem()}


CKPT = TMP / "fn_merlin"
SHA = {"thread": None, "ok": None, "bad": [], "sec": None}


def merge_ckpt():
    """Symlink the 5 parts + 10 PLE shards into one directory; names and sizes now, sha256 in a background thread."""
    shutil.rmtree(CKPT, ignore_errors=True)
    CKPT.mkdir(parents=True)
    expected = {}
    for k, (d, man) in PARTS.items():
        for n, (size, sha) in man["files"].items():
            expected[n] = (d / man.get("model_subdir", ".") / n, size, sha)
    for n, (size, sha) in PLE_EXPECTED.items():
        expected[n] = (PLE_DIR / n, size, sha)
    bad = []
    for n, (src, size, sha) in sorted(expected.items()):
        if not src.is_file():
            bad.append(f"missing {n}")
            continue
        if not FAKE and src.stat().st_size != size:
            bad.append(f"{n}: {src.stat().st_size} bytes, expected {size}")
        try:
            (CKPT / n).symlink_to(src)
        except OSError:
            if not FAKE:
                raise
            shutil.copyfile(src, CKPT / n)               # local Windows harness only (no symlink privilege)
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())
    need = set(idx["weight_map"].values())
    bad += [f"index needs {n}, not merged" for n in sorted(need - set(expected))]
    cfg = json.loads((CKPT / "config.json").read_text())
    if (cfg.get("text_config") or {}).get("ple_embedding_dtype") != "float8_e4m3fn":
        bad.append("config.json: text_config.ple_embedding_dtype is not float8_e4m3fn")
    rec = {"files": len(expected), "index_files": len(need), "bad": bad[:20], "ok": not bad}

    def hash_all():
        t = time.time()
        import concurrent.futures as CF

        def one(item):
            n, (src, size, sha) = item
            if FAKE:
                return n, True
            with open(src, "rb") as fh:
                return n, hashlib.file_digest(fh, "sha256").hexdigest() == sha
        with CF.ThreadPoolExecutor(6) as ex:
            res = list(ex.map(one, sorted(expected.items())))
        SHA.update(ok=all(ok for _, ok in res), bad=[n for n, ok in res if not ok], sec=round(time.time() - t, 1))
        R["setup"]["sha256"] = {k: SHA[k] for k in ("ok", "bad", "sec")}
        save()
    if rec["ok"]:
        SHA["thread"] = threading.Thread(target=hash_all, daemon=True)
        SHA["thread"].start()
    return rec


def sha_gate(timeout):
    """The first launch waits for the full sha256 check (Merlin itself never hashes shards)."""
    th = SHA["thread"]
    if th is not None:
        th.join(max(1, timeout))
    if th is None or th.is_alive():
        return False, "sha256 check not finished"
    return bool(SHA["ok"]), ("ok" if SHA["ok"] else f"sha256 mismatch: {SHA['bad'][:5]}")


''' + MERLIN_INSTALL + r'''
KV_FP8, KV_BF16 = 2479838700, 2464415744      # Merlin's own L4 pins (23,034 MiB cards; preflight.kv_bytes_for_vram, 150 / 79 blocks)
PROFILE = "fn-quad-24g-sm89"
XHIGH = json.dumps({"enable_thinking": True, "reasoning_effort": "xhigh"})
HOST_PHASE = {}


def merlin_fn_server(name, kv, port, kv_bytes):
    """Merlin's launch.build() for Flash-Next on the L4 profile: KV pin lowered for 22.49 GiB cards, the checkpoint's own
    chat template at xhigh (Merlin passes medium otherwise), Merlin's host_phase checks recorded, E23's env overrides."""
    eng = merlin_install()
    extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
             "--default-chat-template-kwargs", XHIGH]
    if FAKE:
        return MServer(name, [sys.executable, FAKE, str(CKPT) + "-w4a16", "--tensor-parallel-size", "2", "--max-num-seqs",
                              "8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN)], os.environ, port, LOGS)
    kvv = eng["kv"]
    gcc_bin = str(Path(kvv["MERLIN_CC"]).parent) if kvv.get("MERLIN_CC") else ""
    base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "CUDA_HOME", "LD_PRELOAD", "CC", "CXX")}
    base.update(GPUS="0,1,2,3", PORT=str(port), KV_BYTES=str(kv_bytes), P2P="off",
                VENV=kvv["MERLIN_VENV"], TREE=kvv["MERLIN_TREE"], CHAT_TEMPLATE="", WARM="0", CACHE_ROOT=f"{TMP}/merlin-cache",
                PATH=f"{kvv['MERLIN_VENV']}/bin:{gcc_bin}:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
                CUDA_HOME=eng["cuda_home"])
    if kv == "fp8":
        base["SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
    plan = eng["ML"].build("flash-next", PROFILE, str(CKPT), base, extra=extra, kv=kv, checkpoint="flash-next", status="untested")
    try:
        eng["ML"].host_phase(plan)
        HOST_PHASE[name] = "ok"
    except BaseException as e:                         # recorded, not fatal (E23 served without host_phase)
        HOST_PHASE[name] = f"{type(e).__name__}: {e}"[:600]
    env = dict(plan.env)
    env.update(LD_PRELOAD=kvv["MERLIN_SHIM"], CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", LIBRARY_PATH=f"{SHIM}:{os.environ.get('LIBRARY_PATH', '')}",
               PATH=base["PATH"], VLLM_CACHE_ROOT=f"{TMP}/merlin-cache", TRITON_CACHE_DIR=f"{TMP}/merlin-triton",
               FLASHINFER_WORKSPACE_BASE=f"{TMP}/merlin-fi", CUDA_DEVICE_ORDER="PCI_BUS_ID", VLLM_PLE_OFFLOAD_READY_TIMEOUT="3600")
    for k in ("CC", "CXX", "CUDAHOSTCXX"):
        env.pop(k, None)
    if MERLIN_CC["mode"] == "gcc13" and kvv.get("MERLIN_CC") and kvv.get("MERLIN_CXX"):
        env.update(CC=kvv["MERLIN_CC"], CXX=kvv["MERLIN_CXX"], CUDAHOSTCXX=kvv["MERLIN_CXX"])
        lib13 = Path(kvv["MERLIN_CC"]).parents[1] / "lib"
        if (lib13 / "libstdc++.so.6").exists():
            env["LD_LIBRARY_PATH"] = f"{lib13}:{env.get('LD_LIBRARY_PATH', '')}"
    try:                                               # Merlin's own scheme: PORT 8000..8355 -> 29500+d / 30000+100d
        mp, vp = eng["ML"].distributed_ports(str(port))
        env.update(MASTER_PORT=str(mp), VLLM_PORT=str(vp))
    except Exception:
        env.update(MASTER_PORT=str(29500 + port - 8000), VLLM_PORT=str(30000 + 100 * (port - 8000)))
    argv = plan.argv()
    HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)
    R["setup"].setdefault("merlin_plans", {})[name] = {"argv_tail": argv[-60:], "status": plan.status, "host_phase": HOST_PHASE[name],
                                                       "env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES"))}}
    save()
    return MServer(name, argv, env, port, LOGS)


STOCK = None


def stock_install():
    """Fallback only: stock vLLM 0.32 venv (E18 / E23 installer) with the -lcuda shim on LIBRARY_PATH (RC1's sampler crash)."""
    global STOCK
    if STOCK is not None:
        return STOCK
    if FAKE:
        STOCK = {"vllm": [sys.executable, FAKE], "env": dict(os.environ)}
        return STOCK
    rc, txt = run("install_stock", f"{sys.executable} {WHEELHOUSE} {TMP}/vllm-venv", timeout=2400)
    kv = dict(l.split("=", 1) for l in txt.splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
    if rc or "CUDA_HOME" not in kv:
        raise RuntimeError("stock vLLM 0.32 install failed (logs/install_stock.log)")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "LD_PRELOAD")}
    env.update(CUDA_HOME=kv["CUDA_HOME"], PATH=f"{TMP}/vllm-venv/bin:{kv['CUDA_HOME']}/bin:{env.get('PATH', '')}",
               LIBRARY_PATH=f"{SHIM}:{env.get('LIBRARY_PATH', '')}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=f"{TMP}/vllm-cache", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    STOCK = {"vllm": [f"{TMP}/vllm-venv/bin/vllm", "serve"], "env": env}
    return STOCK


def stock_fn_server(name, port):
    eng = stock_install()
    flags = [str(INTEL), "--served-model-name", "q27", "--host", "127.0.0.1", "--port", str(port),
             "--tensor-parallel-size", "4", "--enable-expert-parallel", "--max-model-len", "32768", "--max-num-seqs", "4",
             "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.88", "--disable-custom-all-reduce",
             "--language-model-only", "--engram-config", json.dumps({"cpu_offload": True}), "--load-format", "safetensors",
             "--dtype", "bfloat16", "--kv-cache-dtype", "fp8_e4m3", "--default-chat-template-kwargs", XHIGH]
    # diagnostic only: RC1 had a 30,208-token BF16 pool at TP4; no MTP here, so prefix caching (default ON) is unaffected
    if FAKE:
        flags = [str(INTEL) + "-w4a16", "--tensor-parallel-size", "4", "--max-num-seqs", "4", "--port", str(port), "--max-model-len", "32768"]
    return L.Server(name, list(eng["vllm"]) + flags, eng["env"], port, LOGS)
save()
'''

CELL3 = r'''# ==================================================================== CELL 3 — prompts (Flash-Next tokenizer + its own template at xhigh)
R["sections"]["A0"] = {}
_t = time.time()
R["sections"]["A0"]["host"] = host_facts()
R["sections"]["A0"]["merge"] = merge_ckpt() if MERLIN_OK else {"ok": False, "why": "Merlin inputs missing", "inputs": R["inputs"]}
R["sections"]["A0"]["host_mem"] = host_mem()
log("A0 host", json.dumps(R["sections"]["A0"]["host"])[:1200])
log("A0 merge", json.dumps(R["sections"]["A0"]["merge"])[:800])
save()
TOKDIR = TMP / "fn_tok"
shutil.rmtree(TOKDIR, ignore_errors=True)
TOKDIR.mkdir()
_src = CKPT if (CKPT / "tokenizer.json").exists() else None
for _n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if _src is not None:
        shutil.copyfile(_src / _n, TOKDIR / _n)
from tokenizers import Tokenizer
from transformers import AutoTokenizer
TOK = Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
_HF = AutoTokenizer.from_pretrained(str(TOKDIR))
_tpl = (TOKDIR / "chat_template.jinja").read_text()
_tasks = json.load(open(ARC))
_ids = sorted(_tasks)


def _grid(g):
    return "\n".join("".join(str(v) for v in row) for row in g)


def _task_text(k):
    t = _tasks[k]
    parts = [f"## Task {k}"]
    for i, ex in enumerate(t["train"]):
        parts.append(f"Example {i + 1} input:\n{_grid(ex['input'])}\nExample {i + 1} output:\n{_grid(ex['output'])}")
    for ex in t["test"]:
        parts.append(f"Test input:\n{_grid(ex['input'])}")
    return "\n\n".join(parts)


_INTRO = "Find the transformation rule of each ARC task below and solve the test inputs.\n\n"
_PH = "E13_BODY"


def _render(effort):
    return _HF.apply_chat_template([{"role": "user", "content": _INTRO + _PH}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True, reasoning_effort=effort, chat_template=_tpl)


_RX, _RM = _render("xhigh"), _render("medium")
assert _RX.count(_PH) == 1
_head_txt, _tail_txt = _RX.split(_PH)
_HEAD = TOK.encode(_head_txt, add_special_tokens=False).ids
_TAIL = TOK.encode(_tail_txt, add_special_tokens=False).ids
TPL_MSG = [{"role": "user", "content": _INTRO + _task_text(_ids[0])}]
TPL_IDS = TOK.encode(_HF.apply_chat_template(TPL_MSG, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                             reasoning_effort="xhigh", chat_template=_tpl), add_special_tokens=False).ids
R["prompt_render"] = {"head_text": _head_txt[-600:], "tail_text": _tail_txt, "head_tokens": len(_HEAD), "tail_tokens": len(_TAIL),
                      "xhigh_differs_from_medium": _RX != _RM, "xhigh_sentence": "xhigh" in _head_txt,
                      "template_sha256": hashlib.sha256(_tpl.encode()).hexdigest(),
                      "tokenizer_json_sha256": hashlib.sha256((TOKDIR / "tokenizer.json").read_bytes()).hexdigest()}
_TASK_IDS = {k: TOK.encode(_task_text(k) + "\n\n", add_special_tokens=False).ids for k in _ids}


def prompt(band, variant):
    body, i = [], (variant * 7) % len(_ids)
    need = band - len(_HEAD) - len(_TAIL)
    while len(body) < need:
        body += _TASK_IDS[_ids[i % len(_ids)]]
        i += 1
    return _HEAD + body[:need] + _TAIL


TOK_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"   # = the 27B Merlin tokenizer.json
TPL_SHA = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"   # = the 27B FP8 chat_template.jinja
R["prompt_render"]["hashes_ok"] = (R["prompt_render"]["tokenizer_json_sha256"] == TOK_SHA and
                                   R["prompt_render"]["template_sha256"] == TPL_SHA)
assert R["prompt_render"]["hashes_ok"], "Flash-Next tokenizer / template differ from the pinned Merlin checkpoint"
assert R["prompt_render"]["xhigh_sentence"] and R["prompt_render"]["xhigh_differs_from_medium"], "xhigh render missing"
BANDS = {"2k": 2048, "8k": 8192, "40k": 40000, "90k": 90000, "116k": 116000}
NSTREAM = 8
PROMPTS = {b: [prompt(n, v) for v in range(NSTREAM)] for b, n in BANDS.items()}
MAX_NEW = {b: MAX_LEN - n - 512 for b, n in BANDS.items()}
ORACLE_P = [PROMPTS["2k"][0], PROMPTS["2k"][1], PROMPTS["8k"][2]]
LONG_P = [PROMPTS["40k"][3]]
TF_P = {"40k": PROMPTS["40k"][5]}
R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}
log("prompts", json.dumps(R["prompts"]), "| render", json.dumps({k: v for k, v in R["prompt_render"].items() if "text" not in k}))
save()
'''

CELL4 = r'''# ==================================================================== CELL 4 — the step runner (E23, verbatim)
HELP = HELP if "HELP" in globals() else {}
''' + STEP_RUNNER + r'''
save()
'''

CELL5 = r'''# ==================================================================== CELL 5 — the arms
LADDER = [1, 2, 4, 5]                    # 40k: 8 streams would need 8 x (10 + 7) blocks of 4096 > 149 FP8 blocks (preempts)
TEMP = 0.6                               # production sampling (Merlin's --override-generation-config is T 1.0)
E23_TF_SCALE = {"tp4_vs_tp1_layout": 0.0772, "nvfp4_vs_bf16_kv": 0.3787}   # E23 SMOKE rev3g mean |d logprob|, for scale
PREFILL = {"tok_s": 700.0, "measured": False}
OOM_STEP = 268435456                     # Merlin's note: lower KV_BYTES in 0.25 GiB steps if the boot OOMs
# Pre-run capacity (Merlin's rule: blocks per request = ceil(L / 4096) + 7; FP8 149 usable blocks, BF16 ~78)
R["capacity_plan"] = {
    "fp8": {"pool_blocks": 149, "40k@5": 5 * (math.ceil(130560 / 4096) + 7), "8k@8": 8 * (math.ceil(32000 / 4096) + 7),
            "90k@4": 4 * (math.ceil(131072 / 4096) + 7), "116k@4": 4 * (math.ceil(131072 / 4096) + 7),
            "note": "blocks if every stream grows to max length; the ladder windows are 90 s, so real use is far lower"},
    "bf16": {"pool_blocks": 78, "40k@3": 3 * (math.ceil(130560 / 4096) + 7)},
    "long_rows": "1 natural xhigh ARC row (max 120,000 new) + 1 forced 60,000-token row, temperature 0.6",
}
''' + GATES + r'''


def kv_census(s):
    log_t = s.log() or ""
    lines = log_t.splitlines()
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log_t)
    conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", log_t)
    pick = lambda rx, n=24: [l[:240] for l in lines if re.search(rx, l, re.I)][:n]
    return {"kv_tokens": [int(x.replace(",", "")) for x in kv][:2], "max_concurrency": conc[:2],
            "scale_lines": pick(r"K/V scale|kv scale|scaling factor|DEFAULT 1\.0|static K/V"),
            "engine_lines": pick(r"WNA16 MoE backend|MarlinExperts|PLE offload|engram|cudagraph_mode|CUDAGraphMode|PIECEWISE"),
            "startup_lines": pick(r"Loading weights took|Model loading took|torch\.compile takes|torch\.compile took|"
                                  r"Graph capturing finished|init engine|PLE.*ready|Available KV cache memory"),
            "oom_lines": [l[:240] for l in lines if OOM_ERR.search(l)][:6], "host_mem": host_mem()}


def sampled_check(s):
    """RC1 died at its first sampled request (FlashInfer sampling JIT: ld cannot find -lcuda): one short sampled request
    right after ready, then the engine must still be alive."""
    g = L.Stream(s.port, PROMPTS["2k"][1], 64, temperature=TEMP, seed=5)
    g.start()
    g.join(900)
    if g.is_alive():
        g.stop()
        g.join(10)
    r = g.row()
    return {"tokens": len(r["tokens"]), "error": r["error"], "finish": r["finish_reason"], "alive_after": s.alive(),
            "garbage": L.degenerate(r["text"]), "ok": len(r["tokens"]) == 64 and not r["error"] and s.alive()}


def prefill_probe(port):
    """Time to first token of one cold 40k prompt (1 new token): the measured prefill rate the ladder admission uses."""
    ids = PROMPTS["40k"][7]
    g = L.Stream(port, ids, 1, temperature=0.0)
    g.start()
    g.join(1800)
    if g.is_alive():
        g.stop()
        g.join(10)
    sec = (g.t_first or g.t_end or time.perf_counter()) - (g.t_start or time.perf_counter())
    ok = not g.err and g.t_first is not None
    if ok:
        PREFILL.update(tok_s=round(len(ids) / max(1e-3, sec), 1), measured=True)
    return {"prompt": len(ids), "ttft_sec": round(sec, 1), "tok_s": round(len(ids) / max(1e-3, sec), 1), "error": g.err, "ok": ok}


def long_rows(port, secs_cap=2700):
    """One natural xhigh ARC request (its own stop, up to 120,000 new tokens) and one forced 60,000-token row
    (ignore_eos: 'synthetic' past its first EOS). Full text of both in traces/ and requests.jsonl."""
    cap = max(60, min(secs_cap, left() - 900))
    jobs = {"natural": L.Stream(port, TPL_IDS, min(120000, MAX_LEN - len(TPL_IDS) - 512), temperature=TEMP, seed=11,
                                ignore_eos=False),
            "forced60k": L.Stream(port, PROMPTS["8k"][0], 60000, temperature=TEMP, seed=12, ignore_eos=True)}
    t = time.time()
    for g in jobs.values():
        g.start()
    for g in jobs.values():
        g.join(max(1, cap - (time.time() - t)))
    for g in jobs.values():
        if g.is_alive():
            g.stop()
            g.join(10)
    out = {}
    for name, g in jobs.items():
        row = g.row()
        (TRACES / f"long_{port}_{name}.txt").write_text(row["text"])
        ids = row["tokens"]
        n_tok = len(ids)
        sec = (g.t_end or time.perf_counter()) - (g.t_start or time.perf_counter())
        sl = {}
        for k, end in (("to_8k", 8000), ("to_32k", 32000), ("to_60k", 60000), ("to_100k", 100000)):
            sl[k] = L.degenerate(TOK.decode(ids[end - 2000:end])) if n_tok >= end else "NOT_REACHED"
        out[name] = {"tokens": n_tok, "sec": round(sec, 1), "tok_s": round(n_tok / max(1e-6, sec), 1), "error": row["error"],
                     "finish": row["finish_reason"], "garbage": sl, "think_closed": "</think>" in row["text"],
                     "cut_by_cap": row["finish_reason"] is None, "tail": row["text"][-400:]}
    out["forced60k"]["reached_60k"] = out["forced60k"]["tokens"] >= 60000
    out["forced60k"]["label"] = "synthetic past the first EOS (ignore_eos)"
    try:
        m = {k: v for k, v in L.prom(port).items() if k.startswith(("vllm:spec_decode", "vllm:num_preemptions", "vllm:kv_cache_usage"))}
    except Exception as e:
        m = {"error": repr(e)[:200]}
    acc, drf = m.get(L.ACC), m.get(L.DRF)
    out["metrics_end"] = m
    out["mtp_accept_per_draft"] = round(acc / drf, 3) if acc and drf else None
    return out


def tf_last(s, key=None, ref=None, aa=False):
    """Teacher-forced 40k LAST in a boot (chunk x 248k-vocab prompt logits can OOM the last PP rank): free memory first,
    a server death during it is TF_OOM, not a model failure."""
    o = {"gpu_before": L.gpu_mem()}
    if key:
        o["ref"] = tforced(s.port, key=key)
        if aa and s.alive():
            o["aa"] = tforced(s.port, ref=key)              # A/A: the same boot, the same prompt (the noise floor)
            TF["aa"] = o["aa"]
    if ref:
        o["vs_" + ref] = tforced(s.port, ref=ref)
        o["aa_" + ref] = TF.get("aa")
        o["e23_scale"] = E23_TF_SCALE
    o["alive_after"] = s.alive()
    return o


def merlin_boot(name, kv, port, timeout, body, need=()):
    """Boot gate (< 150 MiB per GPU), one Merlin boot, /dev/shm cleanup; a boot OOM retries once 0.25 GiB lower."""
    kvb = KV_FP8 if kv == "fp8" else KV_BF16
    tries = []
    while True:
        gate = boot_gate()
        if gate["ok"] is False:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries}
        o = with_servers([merlin_fn_server(name, kv, port, kvb)], timeout, body)
        o.update(kv_bytes=kvb, boot_gate=gate, shm_cleanup=shm_cleanup())
        srv = (o.get("servers") or {}).get(name) or {}
        oom = o.get("status") == "LAUNCH_FAILED" and bool(OOM_ERR.search(srv.get("log_tail") or ""))
        tries.append({"kv_bytes": kvb, "status": o.get("status"), "oom": oom})
        if not oom or len(tries) > 1 or left() < timeout + 900:
            break
        kvb -= OOM_STEP
        log("boot OOM:", name, "retry with KV_BYTES", kvb)
    o["tries"] = tries
    if o.get("tf_died") and o.get("status") == "SERVER_DIED":
        o["status"] = "TF_OOM"
        o["status_note"] = "the server died in the teacher-forced call (run last); everything before it is valid"
    g = gated(o, need=need)
    if g.get("status") == "TF_OOM" and g.get("correctness") != "CLEAN":
        g["status"] = g["correctness"]
    return g


def arm_bf16():
    def body(ss):
        s = ss[0]
        p = s.port
        o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
        o["prefill"] = prefill_probe(p)
        o["oracle"] = oracle(p, key="bf16")
        o["tf"] = tf_last(s, key="bf16", aa=True)
        o["tf_died"] = not o["tf"]["alive_after"]
        o["census_end"] = kv_census(s)
        return o
    ok, why = sha_gate(min(1500, left() - 900))
    R["setup"]["sha_gate"] = {"ok": ok, "why": why}
    if not ok:
        return {"status": "SKIPPED_SHA_GATE", "why": why}
    return merlin_boot("merlin_bf16", "bf16", 8000, 45 * 60, body)


def arm_fp8():
    def body(ss):
        s = ss[0]
        p = s.port
        o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
        if not PREFILL["measured"]:
            o["prefill"] = prefill_probe(p)
        o["oracle"] = oracle(p, refs=("bf16",))
        vs = (o["oracle"].get("vs_bf16") or {}).get("status")
        o["fp8_gate"] = ("FAIL" if vs == "FAIL_HIGH_MARGIN" or any(o["oracle"].get("garbage") or []) else
                         "PASS" if vs in CLEAN else "INCONCLUSIVE")
        if o["fp8_gate"] != "FAIL":
            kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"]}
            o["steady"] = levels(p, "40k", LADDER, **kw)
            o["steady"].update(levels(p, "8k", [8], **kw))
            o["census_long"] = {**levels(p, "90k", [3, 4], **kw), **levels(p, "116k", [3, 4], **kw)}
            o["long"] = long_rows(p)
        o["tf"] = tf_last(s, ref="bf16")
        o["tf_died"] = not o["tf"]["alive_after"]
        o["census_end"] = kv_census(s)
        return o
    return merlin_boot("merlin_fp8", "fp8", 8001, 40 * 60, body, need=("bf16",))


def arm_bf16_load():
    """Only when FP8 fails the greedy comparison: the ladder, the census and the long rows on BF16 KV (seqs 3)."""
    def body(ss):
        s = ss[0]
        p = s.port
        kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"]}
        return {"census": kv_census(s), "probes": L.chat_probes(p), "steady": levels(p, "40k", [1, 2, 3], **kw),
                "census_long": levels(p, "116k", [1, 2], **kw), "long": long_rows(p), "census_end": kv_census(s)}
    return merlin_boot("merlin_bf16_load", "bf16", 8002, 30 * 60, body)


def arm_stock():
    """Fallback when Merlin cannot serve: stock vLLM 0.32 + Intel W4A16 (RC1's correct engine), -lcuda shim, FP8 KV."""
    def body(ss):
        s = ss[0]
        p = s.port
        rows = L.greedy(p, ORACLE_P, n_tok=NTOK)                    # 2k / 2k / 8k (32k max length: no 40k row)
        orc = {"garbage": [L.degenerate(r["text"]) for r in rows], "errors": [r["error"] for r in rows],
               "text_head": [r["text"][:120] for r in rows]}
        return {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s), "oracle": orc,
                "steady": levels(p, "8k", [1, 2, 3], server=s, temperature=TEMP)}
    gate = boot_gate()
    o = gated(with_servers([stock_fn_server("stock_fn", 8003)], 40 * 60, body))
    o["boot_gate"] = gate
    return o


if not RERUN:
    m = R["sections"]["A0"]["merge"]
    if m.get("ok"):
        rec = step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        a1 = step("A1", "merlin_bf16", 45, arm_bf16)
        if a1.get("status") in ("DONE", "TF_OOM") and a1.get("correctness") == "CLEAN":
            a3 = step("A3", "merlin_fp8", 110, arm_fp8)
            if a3.get("fp8_gate") == "FAIL":
                step("A3", "merlin_bf16_load", 60, arm_bf16_load)
        else:
            step("F", "stock_fn", 40, arm_stock)
    else:
        step("F", "stock_fn", 40, arm_stock)
    R["host_phase"] = HOST_PHASE
    R["prefill"] = PREFILL
    R["minutes"] = round((time.time() - T_START) / 60, 1)
    save()
    log("RC2 done", json.dumps({f"{s}/{n}": r.get("status") for s, st in R["sections"].items() if isinstance(st, dict)
                                for n, r in st.items() if isinstance(r, dict) and "status" in r}))
'''

CELL6 = r'''# ==================================================================== CELL 6 — report + fallback submission
if not RERUN:
    write_fallback_submission()
    def _brief(r):
        if not isinstance(r, dict):
            return r
        keep = ("status", "correctness", "fp8_gate", "probes", "sampled", "prefill", "census", "steady", "census_long", "long",
                "tf", "tries", "kv_bytes", "boot_gate", "why", "error", "status_note")
        return {k: r.get(k) for k in keep if k in r}
    R["summary"] = {f"{s}/{n}": _brief(r) for s, st in R["sections"].items() if isinstance(st, dict) for n, r in st.items()
                    if isinstance(r, dict) and "status" in r}
    save()
    print(json.dumps(R["summary"], default=str)[:6000])
'''

subs = {"__LAB__": json.dumps(LAB), "__PLE__": json.dumps(PLE)}
cells_src = []
for i, c in enumerate((CELL1, CELL2, CELL3, CELL4, CELL5, CELL6)):
    for k, v in subs.items():
        c = c.replace(k, v)
    if i:                                      # E23: a competition rerun skips cells 2-6 without an exception
        c = "if not RERUN:\n" + textwrap.indent(c, "    ")
    compile(c, "<rc2 cell>", "exec")
    cells_src.append(c)
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
               [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cells_src],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
data = json.dumps(nb, indent=1).encode("utf-8")
assert len(data) < 400_000, len(data)
OUT_NB.write_bytes(data)
(HERE / "rc2_code_export.py").write_text("\n\n".join(cells_src), encoding="utf-8")   # one file, for the local harness
print("wrote", OUT_NB.name, "sha256", hashlib.sha256(data).hexdigest(), "|", len(data), "bytes")
