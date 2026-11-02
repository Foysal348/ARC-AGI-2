# ==================================================================== CELL 1 — config, inputs, lab module (RC2 verbatim + E25 klee100 marked edits)
import os, sys, json, math, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
import struct   # E25 klee100
from pathlib import Path
T_START = time.time()
MODE = "FULL"
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
K100_PROFILE = (os.environ.get("K100_PROFILE") or "ALL").upper()   # V-9: ALL (K3 -> K0 -> K -> K2) / K3_ONLY / BF16_ONLY
K100_STOPS_MIN = {"ALL": 190, "K3_ONLY": 115, "BF16_ONLY": 110}                                          # V-3 / V-9: each profile's hard stop (schedule replay)
BUDGET = int(os.environ.get("K100_BUDGET_SEC") or K100_STOPS_MIN.get(K100_PROFILE, K100_STOPS_MIN["ALL"]) * 60)
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/k100"))   # E25 klee100
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = OUT / "traces"                                           # memory rule: the full text of every long generation
TRACES.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"what": "E25 klee100 GPU load test: klee100 3bpw-MTP on Merlin",   # E25 klee100
      "budget_min": BUDGET // 60, "max_len": MAX_LEN,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()
NB1_EXTRA, NB1_ENV, NB1_KV = {}, {}, {}   # E25 klee100 (NB1 hooks): per-boot trailing vllm flags, env and KV pin
NO_INNER_RETRY = set()                    # E25 klee100 R1-4: ladder boots (K / K2) get no inner 0.25 GiB OOM retry


def save():
    with _SAVE_LOCK:
        for _i in range(5):                                      # E25 klee100 R1-13: another thread may mutate R mid-dump
            try:
                _txt = json.dumps(R, indent=1, default=str)
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            return
        tmp = OUT / "k100_results.json.tmp"                      # E25 klee100
        tmp.write_text(_txt)
        tmp.replace(OUT / "k100_results.json")                   # E25 klee100


def left():
    return HARD_STOP - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"


def write_fallback_submission():
    tp = COMP / "arc-agi_test_challenges.json"
    if tp.is_file():
        tasks = json.load(open(tp))
        sub = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]] for k, v in tasks.items()}
        Path(os.environ.get("RC2_SUBMISSION", "/kaggle/working/submission.json")).write_text(json.dumps(sub))


if RERUN:
    write_fallback_submission()
    print("E25 klee100 load test: competition rerun -> fallback submission.json only")
else:
    K100_LAB_SHA = "ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e"     # E25 klee100 R1-18: the E23 lab = a sha-checked file of the NB0 reference dataset (< 400 kB)
    K100_LAB_FILE = next((d / "k100load_lab.py" for d in (INP / "datasets/foysalemonshanto/q27-e25-nb0-tf-ref", INP / "q27-e25-nb0-tf-ref",
                                                          Path(os.environ.get("K100_PAYLOAD_DIR") or "/nonexistent"))
                          if (d / "k100load_lab.py").is_file()), None)
    if K100_LAB_FILE is None or hashlib.sha256(K100_LAB_FILE.read_bytes()).hexdigest() != K100_LAB_SHA:
        raise RuntimeError(f"K100_PAYLOAD_LAB missing or sha mismatch: {K100_LAB_FILE} (dataset q27-e25-nb0-tf-ref, k100load_lab.py)")
    LAB_SRC = K100_LAB_FILE.read_bytes().decode("utf-8")
    (TMP / "e23_lab.py").write_text(LAB_SRC)
    sys.path.insert(0, str(TMP))
    import e23_lab as L
    REQ_LOG = TRACES / "requests.jsonl"                           # memory rule: the full text of EVERY request
    _stream_run = L.Stream.run

    def _traced_run(self):
        try:
            _stream_run(self)
        finally:
            if not self.err and any(m is not None and not (isinstance(m, (int, float)) and math.isfinite(m))
                                    for m in self.margins):
                self.err = "NONFINITE_LOGPROB"           # final audit H2: NaN / Inf logprobs are never data
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
    PLE_EXPECTED = json.loads(r"""{"model-plefp8-00000.safetensors": [5200027199, "dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd"], "model-plefp8-00001.safetensors": [5200027202, "899eaa0716e28594468a1389ee58cb065c23907f1270de3831f5ecb0a4f82d56"], "model-plefp8-00002.safetensors": [5200027198, "06fd8a11abf0419a669f89397b8d70dd6ff42d401e6b2a037c65e49704faaf71"], "model-plefp8-00003.safetensors": [5200027189, "c6aaa1fc08e84eced3c8151ac8679ed943888eba0fcef2556963693430f95bd9"], "model-plefp8-00004.safetensors": [5200027190, "d94e97c96d3ea09208614da016960f3f4b429f47a044c898a51b493f42f74ba2"], "model-plefp8-00005.safetensors": [5200027190, "03d5d4792e14a4ab55bae50bb459624b01786f818bbcfcaed1a5d0235af484c6"], "model-plefp8-00006.safetensors": [5200027190, "586cccbc12383021bc9bc02f206d9b19fbd1672373ed9a5d91e3b0ce34c2418f"], "model-plefp8-00007.safetensors": [5200027190, "c4a23bcc10f3cde6b633e82c282a9a518e3a64d433628283e1bc592c94cf3d6c"], "model-plefp8-00008.safetensors": [5200027190, "2dc8098c0d020bff277c9cf499a6b908e17836b70a5f949dfa24793371c9a87e"], "model-plefp8-00009.safetensors": [4400023163, "61de98b89bb79f386795787d7a76827a26f1e292c26edbb0a1b613da455f5a9c"]}""")                     # 10 FP8 PLE shards: [size, sha256]
    # exact Kaggle mount paths (foysalemonshanto Models, checked through the API 2026-10-31; no recursive search)
    MD = INP / "models" / "foysalemonshanto"
    PART_DIRS = {1: MD / "qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1",
                 2: MD / "qwen3-8-flash-next-w4a16-merlin-p2/transformers/default/1",
                 3: MD / "qwen3-8-flash-next-w4a16-merlin-p3/transformers/default/1",
                 4: MD / "qwen3-8-flash-next-w4a16-merlin-p4/transformers/default/1",
                 5: MD / "qwen3-8-flash-next-w4a16-merlin-p5/transformers/default/1"}
    PARTS = {}
    for k, d in PART_DIRS.items():                                  # the five checkpoint parts (Models from the kernel outputs)
        mf = d / f"q27_fn_merlin_part{k}_manifest.json"
        PARTS[k] = (d, json.loads(mf.read_text())) if mf.is_file() else None
    PLE_DIR = MD / "qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1"      # our FP8 PLE overlay = albucino's shards
    PLE_DIR = PLE_DIR if (PLE_DIR / "model-plefp8-00000.safetensors").is_file() else None
    MERLIN_PAY = MD / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin"
    MERLIN_PAY = MERLIN_PAY if MERLIN_PAY.is_file() else None
    INTEL = INP / "models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
    INTEL = INTEL if (INTEL / "config.json").is_file() else None
    WHEELHOUSE = INP / "datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse/install_wheelhouse.py"
    WHEELHOUSE = WHEELHOUSE if WHEELHOUSE.is_file() else None
    ARC = COMP / "arc-agi_evaluation_challenges.json"
    if not ARC.is_file():
        raise FileNotFoundError(f"{ARC}: attach the competition data")
    PARTS = {}                                     # E25 klee100: no W4 parts; klee100 + the FP8 PLE overlay + the Merlin env
    K100_DIR = MD / "qwen38-flash-next-klee100-3bpw-mtp/pytorch/autoround-3bpw-mtp-nople/1"            # E25 klee100
    if not (K100_DIR / "KAGGLE_MIRROR_MANIFEST.json").is_file():                                          # E25 klee100
        _g = sorted((MD / "qwen38-flash-next-klee100-3bpw-mtp").glob("*/*/*/KAGGLE_MIRROR_MANIFEST.json"))  # E25 klee100
        K100_DIR = _g[0].parent if len(_g) == 1 else None                                                 # E25 klee100
    REF_DIR = next((d for d in (INP / "datasets/foysalemonshanto/q27-e25-nb0-tf-ref", INP / "q27-e25-nb0-tf-ref")   # E25 klee100
                    if (d / "nb0_reference.json").is_file()), None)                                       # E25 klee100
    MERLIN_OK = K100_DIR is not None and PLE_DIR is not None and MERLIN_PAY is not None                   # E25 klee100
    R["inputs"] = {"klee100_dir": str(K100_DIR), "ref_dir": str(REF_DIR), "ple_dir": str(PLE_DIR),   # E25 klee100
                   "merlin_pay": str(MERLIN_PAY), "intel": str(INTEL), "wheelhouse": str(WHEELHOUSE), "arc": str(ARC),
                   "merlin_inputs_ok": MERLIN_OK}
    ABORT = {"why": None}
    NVSMI_OK = bool(L.gpu_mem()) and not FAKE
    R["setup"]["gpus_at_start"] = L.gpu_mem()
    if not FAKE and not os.environ.get("RC2_CPU_KIT") and len(R["setup"]["gpus_at_start"] or []) != 4:   # CPU kit: no GPUs
        R["errors"].append(f"expected 4 GPUs, nvidia-smi shows {R['setup']['gpus_at_start']}")
        save()
        raise RuntimeError("GPU_PREFLIGHT_FAILED: " + repr(R["setup"]["gpus_at_start"]))
    L.start_watchdog(HARD_STOP, lambda: ABORT.update(why="HARD_STOP reached: watchdog killed live servers"))
    log("K100 | budget", BUDGET // 60, "min |", json.dumps(R["inputs"])[:900])
    save()


if not RERUN:
    # ==================================================================== CELL 2 — host facts, checkpoint merge, engines
    def run(name, cmd, timeout=3600, env=None):
        t = time.time()
        try:
            with open(LOGS / f"{name}.log", "w") as f:
                r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                   env={**os.environ, **(env or {})})
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        txt = (LOGS / f"{name}.log").read_text(errors="replace")
        R["setup"][name] = {"rc": rc, "sec": round(time.time() - t, 1), "tail": txt[-1500:]}
        log(name, "rc", rc, f"{time.time() - t:.0f}s")
        save()
        return rc, txt


    SHIM = TMP / "e23_libcuda"          # FlashInfer JIT links -lcuda; Kaggle has only libcuda.so.1
    SHIM.mkdir(exist_ok=True)
    for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/compat", "/usr/lib64"):
        c = sorted(Path(d).glob("libcuda.so.1")) if Path(d).is_dir() else []
        if c and not (SHIM / "libcuda.so").exists():
            (SHIM / "libcuda.so").symlink_to(c[0])


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


    SHM = Path(os.environ.get("RC2_SHM", "/dev/shm"))


    def shm_names():
        try:
            return {p.name for p in SHM.iterdir()}
        except Exception:
            return set()


    def shm_cleanup(before):
        """After a boot's server is dead: only /dev/shm entries that appeared during THIS boot (absent from `before`) and
        match the engine's prefixes are unlinked; everything else is recorded, never touched."""
        out = {"host_before": host_mem(), "removed": [], "kept": []}
        try:
            for p in sorted(SHM.iterdir()):
                try:
                    sz = p.stat().st_size >> 20
                except Exception:
                    sz = None
                if p.name not in before and re.match(r"(torch_|psm_|vllm|ple|nccl|sem\.mp-|cuda\.shm)", p.name, re.I):
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
        return {"ok": g, "gpus": L.gpu_mem(), "host": host_mem()}       # callers launch only on ok is True


    CKPT = TMP / "fn_merlin"
    SHA = {"thread": None, "ok": None, "bad": [], "sec": None}


    def merge_ckpt():
        """Symlink the 5 parts + 10 PLE shards into one directory; names and sizes now, sha256 in a background thread."""
        SHA.update(thread=None, ok=None, bad=[], sec=None)        # a stale result never gates a new merge
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


    def prefetch_bg(paths, tag):
        """rev 3: warm the page cache of a cold 17-30 GB /kaggle/input checkpoint in the background (many readers)."""
        def go():
            R["setup"]["prefetch_" + tag] = L.prefetch([p for p in paths if p]) if not FAKE else {"fake": True}
            save()
        th = threading.Thread(target=go, daemon=True)
        th.start()
        return th


    MERLIN_ENG = None


    def merlin_install():
        """P2 payload (.bin = a tar Kaggle does not unpack) -> install_merlin_offline.sh (UV_OFFLINE); payload deleted after;
        a failure is cached so later Merlin steps fail fast (rev 3)."""
        global MERLIN_ENG
        if MERLIN_ENG is not None:
            if MERLIN_ENG.get("failed"):
                raise RuntimeError("Merlin install failed earlier: " + MERLIN_ENG["failed"])
            return MERLIN_ENG
        if not MERLIN_OK or MERLIN_PAY is None:
            MERLIN_ENG = {"failed": "Merlin checkpoint / env not attached or incomplete"}
            raise RuntimeError(MERLIN_ENG["failed"])
        if FAKE:
            MERLIN_ENG = {"fake": True}
            return MERLIN_ENG
        try:
            pay_root = TMP / "merlin_pay"
            shutil.rmtree(pay_root, ignore_errors=True)
            pay_root.mkdir()
            t = time.time()
            with tarfile.open(MERLIN_PAY) as tf:
                tf.extractall(pay_root, filter="data")
            R["setup"]["merlin_extract_sec"] = round(time.time() - t, 1)
            pay = pay_root / "merlin_offline"
            shutil.rmtree(TMP / "merlin_off", ignore_errors=True)
            rc, txt = run("install_merlin", f"bash {pay}/install_merlin_offline.sh {pay} {TMP}/merlin_off", timeout=int(max(600, left() - 90 * 60)))   # E25 klee100 R1-7
            kv = dict(re.findall(r"^(MERLIN_[A-Z]+)=(.*)$", txt, re.M))
            shutil.rmtree(pay_root, ignore_errors=True)
            if rc or "INSTALL OK" not in txt:
                raise RuntimeError("Merlin offline install failed (logs/install_merlin.log)")
            sys.path.insert(0, kv["MERLIN_TREE"] + "/tools/merlin")
            import launch as ML
            site = subprocess.run([kv["MERLIN_VENV"] + "/bin/python", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                                  capture_output=True, text=True).stdout.strip()
            MERLIN_ENG = {"kv": kv, "ML": ML, "cuda_home": site + "/nvidia/cu13"}
            return MERLIN_ENG
        except Exception as e:
            MERLIN_ENG = {"failed": f"{type(e).__name__}: {e}"[:300]}
            raise


    MERLIN_CC = {"mode": "gcc13"}       # rev 3c: Merlin README:229 / install-env.sh: FlashInfer JIT at first start needs GCC >= 13



    KV_FP8, KV_BF16 = 2479838700, 2464415744      # Merlin's own L4 pins (23,034 MiB cards; preflight.kv_bytes_for_vram, 150 / 79 blocks)
    PROFILE = "fn-quad-24g-sm89"
    XHIGH = json.dumps({"enable_thinking": True, "reasoning_effort": "xhigh"})
    HOST_PHASE = {}


    def merlin_fn_server(name, kv, port, kv_bytes):
        """Merlin's launch.build() for Flash-Next on the L4 profile: KV pin lowered for 22.49 GiB cards, the checkpoint's own
        chat template at xhigh (Merlin passes medium otherwise), Merlin's host_phase checks recorded, E23's env overrides."""
        eng = merlin_install()
        extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
                 "--default-chat-template-kwargs", XHIGH] + list(NB1_EXTRA.get(name) or [])   # E25 NB1: per-boot flags (last wins)
        if FAKE:
            return MServer(name, [sys.executable, FAKE, str(CKPT) + "-w4a16", "--tensor-parallel-size", "2", "--max-num-seqs",
                                  "8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN),
                                  "--kv-cache-memory-bytes", str(kv_bytes)] + list(NB1_EXTRA.get(name) or []),   # E25 NB1 (fake only)
                           {**os.environ, **(NB1_ENV.get(name) or {})}, port, LOGS)
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
        try:                                               # audits: tree products, venv, checkpoint marker, PLE dtype, toolchain,
            eng["ML"].host_phase(plan)                     # scales file: a failure stops this boot before any server starts
            HOST_PHASE[name] = "ok"
        except BaseException as e:
            HOST_PHASE[name] = f"{type(e).__name__}: {e}"[:600]
            R["setup"].setdefault("host_phase_error", {})[name] = HOST_PHASE[name]
            save()
            raise RuntimeError("MERLIN_HOST_PHASE_FAILED: " + HOST_PHASE[name])
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
        env.update(NB1_ENV.get(name) or {})              # E25 NB1: after build(), which strips every MERLIN_* variable
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


if not RERUN:
    # ==================================================================== CELL 2b — E25 klee100: fixkeys overlay, model dir, K6, digests (NEW)
    import base64 as _b64
    import lzma as _lzma
    K1 = R.setdefault("k100", {})
    K100_EMB = json.loads("{\"k100load_lib.py\": [\"0ba5f7c6328cdaa24276ac92eb22d4b2015b578675e9c36c711d1ad1a7f7f1b9\", \"{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;OJUPB3%FxXbna~Q3<Lo$NONZq)WnKr<cv&>X!>&Bo-1*GJ5f{&H?@_x7}G^%0MlKB@w5R+I+zdtaqMS)cJ1Dp~1L7Eza7UWuRfHa5QrM<EIu!_@gqY0@t}ds96uIkR~{@L?CLXfk2kr<a_+QBE-QvRAv4c3q(d#I645@1q5LOr!W@-EX%1y;4#3U1{{{~x+Z&+W0S#74?A-2$~3=PPzU$HB{(!hj4kE8K`Vl9O{JD)?cD4LPwljP*CoXWu;kNj>8%T{$nc$x32S~Kr!K?yL2MCli(AsIZYafEn%s5h&jL;$YFvedz>#}M?4K><hQy0`;M)w=)IvBl4rI+7Fzo~qP^kYAF3}gf{3emH$q>I!B}hCsTE<ym1W}bjbwUmXVB7alfuKMOIg=Rfq$wfF`)CV7<hSH(!gKg(bQ>;cJ5$M@FI0QXx5M{o%2(A?MEakL*%YDIH@<H`NIJy&s6*{nvJBb0og9=_;7&ml`0e!ycR7CZcI^R@{m~u`%B3Dk4k#lz<Mrd1<t5Y=qgRIMTT^MR`X1Evf7;6{<B(b}h5XpR{PNlRwW#b-NWLfNUgpC4>TX)5#BcT!1zp0v3o)!jfRVDXwQ3L+Yk!0#F+Wa7{HuCS(jzKb!!SFF*ZaLLc7m#b<IaCQ|4?>(?q4QzNaVWH?)K8(J~31luHgu>5=0O$SN#^TP`uaR(Zh4WG4_UL({s<TnH#4&r)g+HZal%I-Q1W(oo&@Eg#r-Mk|16j9Edi&XsA@{U@?ac!Z)7RL;t<=N*P8-2Uw$J=ZMUmGSMy5=Nv?E!wBiWXJ=nA41!!1qS~_&UF+3FTEV4N@W*q+a6{#*H&2ARnU)SrgxQ~8F7X2>ZMwH#^98OQZmu;04m8rQca&n+^*gM$3d;8nMzBE&(3#|AP*TIrleY(1mLL`r31zqNtMune<k0z2bWDKvQU*5{rsS%KIH{?3qyP<)fEHryalqcd4zYdH!8JmrfP$zGBsFsV(8tjUTH`j7A344OH&k3Iv1RG}wTmrvEJ!c?tq+K*`{a0-!E>v3uK?;io6`}sPAT|+=?kefPI#l&w^l}TYV(FIGrdSM80F`gbvet$1$F8G1AUoWL0~S<ddGR(zw&q-;@XEeQl)ZCKsue=vxJFWq0MBwS_n@Zu<PL4tUKgRbeivNtORzgYW%T1Js3m(E1I*ji_V_apRfS5JV`N8>bqqIp07?2F!02th~=Z0Ja2ULEvNv<Fi~y!loQF$O$KKSQh|k(R#}kqF}xmGP*h8rG<sbgBR+I@&Rf)@G?GdLbEvZ?moIm4=X|%Fe)bsOjq`sy+Bt~Ju$6>HlMB!5OYS%f5^e%X7dwZuN)ArI%>HSaC&B{bCx>Sz;nbs8k|DXn<Ea?cYeeWsTA#-z*;O)e^630p<Q<f1MiS=ioD71<yT@v%byAg5)`?z%G)7@+ej$v$j+$Swf~I)*k%GR4Dl&y;%22KhCCO>}vueU195{kMor0C`JB`P34@p!Jkq?7OgSThmsS020Yb_Pc;9}@|K)!sWjcJ#valGYKG|4jJ^iFPXA(ie1YGWN`I$o6^Zs2LgOcya9M3ew49})}w({H~`w0deSr-Z73ZLX8h?C^yL7<0V!hZ9OXRq`krWA!pqs|7o@<j85LRuHHTlptJMU{&g1Bh|&mSiT6)A7oilw+`ATL6NtIn0`#c1wDe;JAh~SW19iP4z(cWO%U<Jy;oG&b`Zl>9<6QZoR&^ENe{m2dI~K*{5xtZY;rlC11cOiDgt2&=}tP)73sJ02v7SebPJr1;8~JNYrfIbW%G3!&3KC6=YaRjHW0SFs-#3&mqlpRIB>s87GiiN18my!%{c20K$69Ai2P{n<#Q5OOsBc2SSNc|+t28f<eo7g{n96bddX5^GRI&F4kQ0CemPA}=`u9?^@(>N=t3r@+v*S++j{IZ>4NXAcHr```frxf9wqKGwYYtv*bE!|!?V<t1@fxVWJ-U;D+ME0HbU1@H3+bwUst?)8~MJ3jir{VzqF$Cv@|KvZ6|HI5tiRuV?T_l8Xf=h5(1E@sZxDJ-GX)Kj@8UEbe>yem7LTiClPyCWKFWcECP^<Dlt*Hu7P)htcJHaQExX~+E7(_mji70`X4k*70tuH1s{gknTpZ1qgJ-2C5g-*wT?~mV#;TY<}t}P1xIYGpf_LXT&RMUOM~VQfX1`5a`G)ux2qZ6H`s;=lw5|vITNeNZm6^QXtFe&#)>3U+(iwwN^r8FNY20&+SgfB`-_5fl%mg?C~3eb8(>jzBlQ2UBR(u4-h%?dkl7mfuL_ZcDzrlAJUszF%X=ma?UBrGITzdTbClAYtMp^c0GU$r5d*W1)#_YUo_%P^9n2zoQ{J-dJ8Ysst`F0xHgf<?K(X5W@N7W7*TA-3pB+}u+zB!uMM?T?eLtN2^e;TLF;M$<KiJ^G%7xerfNbl=^L<!xr2jc>hj^0^qrSBq^+Y7ZwGLKJ#nIqtkdu)<_g)~7l@8HJ&NXgB!;(zL&b6mtgAIhtPT>!T)kruThhz+o18d*WDSh5SlGqxBo^X*HYAU!w{;&`QKq55)TG0WAEpaFtEDjc%iW<orz)MGfoye{Wd5)5QlCPcvqgy;4`i-~ExS}H6zpAU4|J~R=+M7p^Y=;Vgx?Ei|##R&HJz3C5F-i}|ORwu}VaxhE_p}}yn!Q!u6zeJYSw{1F99`WAi`53J;|dNeJuAKyA{{|@M?`-c;W`wFXGI74wywn#)gNkp#e9OFz~@9WC$Zyv8{{w`<)YM2%ewb9li8X>mw<Lu6$~2J*;HUfv|3BDF}H<H)^#!+?+0ICOMK<rVFw+lN`FY_D9(Vrp0P|$wn_PsczVHQv(vtax;zQjxQ%P}Lhfg(jTq1DtDqBrj<1*(XwM~P;>;W{iwfoD)yNgobF``pt7xPm08guUrFICxE&mE<cNgOa=R0kuY&<=%9=}9lIveuN)4<S8Vz_bP#(AbeP)~+59p`A%=tSQNF$vZ7)t}1X<lW7V$yGbEE4b_Xdqz~o$nZHcjSgkm04yzg(^89}&V*#Dc^qt2s2(r!;*o`$iN#W2)OmmcWkACeHEy~RAP=0%)@T<+ZHD1z{6ZCjbI*0wkN8`#1svmYSSM05ua|9{L&W*kayOYBo6Sk?1gMFi5lu7?1^#O`C>=mo4;FUIyV$cbj3%%kot`|Pev?AN?i1U5R`YRkE}7N4=7B-S!>xD2M_nc6{tAw|3Xs)q9JDX=(Ge><LBjQk%zAOgWP9D-ZN9yf_kXNcixjm1*0f5>Fw*<AmxK*LTO6B}W7yr+nU%R19h-HdJkg)gY}(d7&y<FU`$pdd+L2C=s^}sS&vga#XJ6vwnzqRv(kWk)(h$vq7>n{Zc)BO-S-AF?J>IheCy-geo7iIqs)omd@+(ZCRjagx3S%kFH|^_-h)MgJLe^?#?57N22lJIdnW@Ib0B?(SR)q-nm3`_v2*OL>i%*43w*caAJ@j7MEvR+cMPb(6gz)u5ui!lyY3Ix!@Xh>YNzPv69P126c?#*Qi$@oDd|Gdwh9BtmKo6LpR1U7?#8pSQw_L|x(@wV)fIxUCwazMFwSL6`Z+pf1QlU$k?YA|**Bl@I(;Xxpg4a}IQ2Zhjb6?1Ro-yId5V8dXfKgQ#s2ss}j6Zc-LTLzK=_~#IJx>QX%wh)k1Uln3I$t^^5hX2}i#J{IJhVRt|Ca`H=uSQaw>$;}nYuP{!p$5Ko%@`@YX{z_2)CN-mDuy?C&2*LP|TNnqw`fZr})}W?x8U7tz*gB{4?3$^rZ=qz?{b%qDX4!N9qL4xmO!4Q7GLMpsmpU3lfw+r0dhvAhU|4!p_)MET*JZNv2R2YAV$maU|gxYjUymvR#kH=&F#?{}*RDdcVX{>3A-*tIa_B4x=NVE(em03a?q8BJ!bQy13Yi#5fH1O`0=zsfuvRCDmcUR&6wH|N0Wjpv!nBzkk_Ru^(&WOI;{I_yqN3iMPM?{ZDnHRY7uH(`FuXLZGU*!H3OooZ-o{M}SvmrTH*Ggw6=Hv@Ltizj9YDPD(|<xb%Yx*^A;m^aAn8HOP9Hx?ei+iVzVy2&5ZEejYQFgCp8H3-37gXop+}h7_BF_IB2X-CZ+39#ml(N;fc9=;w<DW5`?f0vmiMqJ#PF!x(3@*CJJSQDr2VA5EkBi_C@lBnX?O=zZ!o7(=byJ_W3oMS*83_o3J0E*J#Sx9@6}e;_aYH^Kf51$+H!8~K`1uakjPJMT;F1#;Y*!Eu84mw$y|q#n3mr56h}NK2{}tRTG^M{^TFFbH|@+`MMWI`~N+eo9AG;F-_8_0GEF#RMA&5m@l_^tQQmIn@|E8`suNdUK_!0zHGuOGK(1Ro9G`@pETLv+<hng(zc(Ch|qhB3Qze0e?}c;|gzYX&}>BZ30RqI*U(9m~o_6p08+=Q3Bmm6`N}z;4x_g(N{F5x=m9%#ipqo#t-bqfNFrj_5hGqS(`GtM4`S_j@IyMSgk9uZ~#2!tSzpNXP-1DO`%RxR&_${&)#4{yXLl+sCA=U*5J_uK>K89kuO}VxxSspGR4M-1PchNt+}6iUdtLwjl?3{1W4YZg<sh6&`*{HwOQ9xDLv_Eg|>WYUh5A~V9vtrE6w?2#WJ?y4FuaMI$$UGhbOWk)J*Ms&;RZrcsBO{4qgcb7@tk;;EfzdzQZGTP|Q2>1D4EptDLkYERF>chMd-ol7m0*<C4x(Pr&WTeeQ2+OMQu_Elob0Uy({lEZMJC-{>h#BNv;Lg`jtl+I<v8!2jiYcouFm8EfZ|h>sHKjcpi)LoW-8z|2Fvl4xANS2oDvRi-6InO8-_^@1@t;Dg+Mimb&XR24s>_8`5&Cj+N3d^cB17EcT{?_Z?y^D<!~JQEI@K36X}6tKU04eh@`Ia_j!^3CyDNtDt?M`uk$6|JTM3h7Cmlxnxrp4^A60okL(qgUmQ(@_b_#bdBxsA7m+y1pw=He0McBG!N)t!yy@gYP6zEwUck!5^qXk<*j-yMOX*&{ZU^b<8VvOsqw@N}kA;*X_Z7Z8p0tiWFeL+#OSH?AY@70mgo~@-3Gq!*qf6bsndiLpDQisH5VM!x;ih4(lea)Oq0GpC5<ogLY$eh(vnWSS6Awg#lg?Z*8Js>DVEbk%$xBRnj3L>shhm^}L&Y7ya^X=qjyR+(4zR5rK!$Y|OW`lCYh9ueJZ|y~Gxm1@nouQ|iRRbml?ju}%PZnAQTriij<<My(j{zI1-Zl)L@d4n~$GA>B28vpI<GCvDTp2`cKhE~;f9(u8gw9^rCwcP#&B`i#>{cb~44oKoT%VehUuEz*X`AF9Av4S2V}=EVsOx_emSR^rWEb3{aCF%WJ#*;2+)z%ZXbt^h_C$G3&tIL4MTnP5Fx>Kx>Iy{A4+5i$`5;j7`$dFM0Q$;l{59njkC4n1RwA_D`t!uJ82L|wOgf<Rgq+mQ%n;w-H@U$6=b>PB2KJs}-mX;WZegr;BhP(m=4IX3&LU%**Nq`z1aoRWvj@*L{3M2wP9<G%ve&%K8MqE;wPBIB<ivb06wE#fY@<#HF~k`drTVWaP+iM;#d9zMcAaemWP%Bz2lwxJIdMm(NJv#*0QfN+DGm|Q+a7&8rnt0Q8$R=Y>ICaYX0p9rY(km|!JXtkh@=Jr(Cg{{v=Gq0vYHIe5rx6I$rz9tfU?DIetqx4FYwxb%r{kwA3+25CK5Q4+53v&BY2@;eDvEg2RZzoRqoj#FV-(I_I*_k3GadUNQTD?JTg!$#43#dawVGUHfK#vY+Z_gO|G0BJ2$7$GP9T7xA)hu<R&RY}rVFsk5T{E_|G94_s<!S0^0j~ntk%uDx+aA{kf!b0XZ2=TFke|erEk<{riSw$tT}))%i}#$XfV>=#E}8cd|AFDv@1;2Kl`OF3gpKs9qBgcr<yO}DDIlpQj`D#HX^J@@IG*m!Fux|&HS@Sa{EYZVMIlBhG<({PEU(*FU|LUAB^>kFLvBL=+e_lAZOr<8<x{aya<IekM(C|8!_VHqDK;{STXtRf1zwIZ^fqnxcul?T^`y_;XE*{-^o|JqZtG@GxJU86U4+QW=GnC#pn%m+SSmbo?n^vYIw%HH^f>pb*<U!}u&wV3W&Z=^?wtviMq+B+rX|UHQ76$Q#!@R#G>MMgA*9!P-0T$mgk$%cLPRyislXR>j-NPRnD&5aRT6?lhosq@e>+Z8losi^*W-uH1>NHUy1k(T9IF9(c0GA^1RuBytkqj?KMM`SByfiT^(l<P`Pb$nD<4o(cf}JASb^M}iY*8HY-P2&+dDIEN(g!;=e{v)<Z3*xaDOX$`IP%qxYZ6rWJIi+)=<du<}b4v*+l$lxAlV(|4-C7+a2aI#6HDwYh8H}P++0Sn1&FNBIi-dw8~fNea_czD!dNwSf+_$aEaxcWfOdJKIKY3;OEG7I|TD~a|Cx<!R1C!EaA`50_l^xU~%YsH*uMkGHt<=wBm&7(nhcJrj$FI6k|@}eK~w%dDC4fJf~vcJEp=1(LGJN?8CG0AizZ~RQ{-4)Ko^^)v1PH$wQE2Y{6i6b_eBU7RAh|)hBW5-jh-shxpeC8_c3#?1EBhCww{JFaEYXNKVkSrA4W4|6xvsGc;%Bm1^AX!Sfz&!*`&(dTU&}ir;QOZy(|4{0y+KfA!G~aJ+58Qc7@m<uYkEL($EwKy7|&Lx0`_961a&K&2dbjWF}a{W(EkqZp_!B2_NaWGw>JZlf6<Kq`_GynCxA$v5EH#n@G_9I5NL?6bTrb0t9^qHT>t3xmEJvhn2O<_#>(*16z0!KBOc%^8RHXkL)-=VFVC==AiO7Jy<ZgL1psb7elxj49<&nqtE}l?T~MOT_*K?d{ofzMu*vK}8mnUj>TJU0<&(;MP>wl=S_lBnU5>*BYc_0k4(`mSh08-|kC-GV-s<koN_CH8_L+I}d8IpPRo<`*YE!;sGhbQl*e}<j}nU4U-~mMJ`>L<p3eFx%uhST&6gU_jQtmu*-gZuUs+;MqK~`MhC)|9aH{Ep6=GhmqOF^*)OG0x%~`iRJnzv2cJ(SZKYj_RU`xIXUf6l8Buz)S_Y(+ZGjGzauku*c{hYSA(<zn3V<W{EYT8*#T>`*I2-^qFCL7e=C;796)z4J<G)O&Ji0-gY`r~Za2;dlmLJ<k+auxeu%7hCyL@1I8LVeT0kW-{nA_PQndHb`YSD0^jXYY8&<!?7GG~A?SeKo`LB)XYxu%PJo=HEh96*w*bEm3XR65?6K|!Rn2)9~SmfpoNd1b-#g)Of0>+`37Wb%*tjDV~Dtboa)oN+B}dTal)dMT&@4p%7qvbn_MzF+?T#%J0?%Rwd_)!>O`I5Dquc&2J77<nRJ!rC6C7y;|ie;icj&C3AinLa2}RS?cSvxd=v{Uh0>0ip^D@E6oeP8vQtmR2*Rv=s7MVji+PkK%?Ne_~%j^K|nMlG~4jP?~%s-@PH!fM%t?9SYydi`-*h35v);XZ96MFEZ-DiR;L|8t)0Pbw~Y4DRFWqgKy_4R-F{m+@msN4H1mOv#P<h=YXN`cGuh%Shj@o2yKbi{;@w~HCr>T!o-|Z+IH=pjwJITMyZJKsJ}OMo9qAIjXm6`?U!~9c7@;QZEqJ{V-=aU7DH+~ZAmq}LTeCfn5iDTHB(RlehgUF8FYLJT#vL^-hzn4d&KX;3Be?UA0SJp&V0C1IK~rSq)V_d*C?k_K};k!9RZ#s27KQ;N{x#LIfeF0UA!#U6QIH+yDN`r7d@m~XWxL7@vPX+3DJDNPg1g?)0aXg1Ret$70&D$5J~2$yEB<Nlm!67({y-bsC@NKK4&v8j+^wtS`CxXL4nird1zo5qgwcr?II<)5k8<56bj_!3AhVpvC(j{v}k`riQ)L1Tm%c&mXF2{Ou16iX{$tKQGC>Dgo+h1QY_WZWOX0ONS%Y&!pv-|lnnJFhq0pYGS?3off-;EcZoKn7|1x$ivc#)?to>n>U^}J%2lFE9VJ`pnMpN2(oy<;MEPvB;x#I@=oR?oXu|;ewA{Z6oKNFs<#D+@XSqXLSh8TkZD`TL_3(llLhZj!ujdJz0Q#@rKK6_f*@+tf_1yq_^Bgbt$K^m$yJzo@Y`ib!K$2~9hbZ;vpjrm}kndM{rvere`G0@KQuOxHTwFPhfSXMk|0h`xBN#6-svA5H;|Mit_5+gi6avo$QcC*5xIZ@vaQy|G0)7bMkz#>}A1CFOR3^gaGp~tba*{EF@h+Syu8qzk@DtQMrLNpbPxT(0pnP&k-(g{U^zN7MPIOK;($wWE1X{X;!(>Y5C#*c+<*imzlx*xxDd)Cv=`}bjB8ZGl1H9FMG^`Kg^7IR(@wCEg-<+Dmnbl^AgzHMvQ4HIJ4U~;=^x6JH5WrS5?y|8*k4rOq$8$D7tuQH*Bqf1^)V7}F4<eLFlRihmpD?(F!JMw(qYejIu@PeeZ)>6*%cB{&P(vy%A~=RE0k5aV%A^tmxG+=U3ax6yUQ*OT|3vQd_p=;VnAOR~hue@KY<91-Nqd#8U$H2umCJ*-jLBF=*AD{TwIYTNT^lcrYBGn?B?_O2Qq>!Whh%WBBjhKO)Lk(Ub8PARR2r=wsQcCM&4|3~6kh=F`^nCMXSU%3ntK(4<C5z=q-qmq37lg_l{Okoy{S~hX1^-*q5tAhc5LXS3YO`Nr?-j}+E2{9iw=;wWd?jbZ$}?xdrcD}bL$Z{aT5*bmoJszv1d{WVq{RA+;UB8w<$S(Axl&KfBLz%U&{e0U{`E{;`f?Gl99=WM7bqzJL>m3O=_gS>I9q?&2SN^u*e7Oq2t|uf+h_>BZM654}vpD3rjbG*B;O+2}XX*kEZWq1T!!ltcy8!BbuJ^+tAn)ba|tI@=!6+)=q^v!*RfzE=n~ak;_!sPocI*KaNFP4aLcVii;Sj=fvqfdv4$x?W0wUaKFQX0|!d}&|cgs3OhYz7H;u_6X6*`vnMy(&+eefcJ6kG8R4NesX*;#{@*QVrS6P$2L%w7<otFRjopT<04w=@9S=}hP?fi0v<f`S{fan3pz_1~SJ#4g2p&a&e?i7e;QKUHDji`w0qk9b%xQoy2q#ke*6iuZ_#58(p4KA?LHq0BH|29xtKRr!E&o8zX>p<<D{UHAjbK^DhqO{+Lzm31N%3OU0LN#Sc@K9+^sK8z+KiyNlBiN}cU4koTr$FY=0n8?D%?B0cgM){;Yt_B+|$~#*1nk(Mv+GOqqN^Y2YL%+{1iMz-^Pi=hr(PP*5TPCo+aNkKj^-A*CZDUA-t)Hfl7O98gdV31c5>!v<DX_LZ)Vo|4%3jv+CddwLMR<=XrG`Y|e%fB1V6>p8L*)UW<5XI^_8+Vq`HsiT3F6v-r$l2jCiqxSb|ic)%*g(HEHL?vcyG71*7b)965P?$4xGxCAY6a)XN+R=sOodZrMc?4ZOo&6Vzkt?@;9x=2C<BFBt8=Y=XVXYrvRltiQhw&o(gOw#?JX|hOMucuhT`&$o+2h;&7KqiHzIfB1B-f>%Zfu-J`_c!%KhYWd`F+>C0Y2bq5u2sw5Xz-acqgNtWr_BLhMx$ktP?~D*nlE4M&&(mm02y8=)K})x3+9Xwe{T1niF>yv^=INvju_{AR%)X)0ueM*`G)0yt=*rtM81sOvg%4cR-NB3dk!sbu3t+umYOk&5e+)S2SSM%I{*g?dn`)^Q#;{2YXAL}xMZukY9^}qIIFaH;}@H>!<BHWg<dao3_p#%AcT}}2IGo?s}<aVTmh~+tOD}({TRbYCI>{x32a<kL!?YG7Cx<5=><eWo8vkVZfThWcqA~YU*sK}9Batov|nn@8i(=d>eZ)&a#xPQiq~e}$V;(PPJ0hXhN}e}iWp<78qKXrx3dv?JWb7N2Q*c6hfKti<UsY8m_&s%fDOx^%7ZqIT~&dt=CBGhnPzMADw?#z_RK8!{Zp)STW3XfI4C4@xzQu-d3`I;hTeC;g6vKx2A<a1AEWWzhC761I0({)FiEn$>sA4I;bO&9cj@2*itWl=@?>HVS~%yu(B7~!Rl@f93obGsqcl4-&IYj}#BUJJ*GJ8j#*4X8ys|IOSygA<SB2qMa<Ygys6jt%T|S`30~rS)_`ji@gPRMZD7{U=k!%1F<!Q0a;6oIr$J^mjp1Y|BM!6L51at`_qZm31KJt04_0f_B*Vm`G-a?H0eJpcsT&`n{0eJ|nUGe4v`K9DMwcc26eD&I+b&|Ol{|wrq03BGMBUsErDnj#D*%$EL+W8-_Di4-eqD*!g6@H*ykCrz7_bl5o0<BrU_Z2X|v5P})L@=acFSD=O|E@+K{Ez6%)7rV1AVE*_e-v1357AF_+1jL*GrB3rmuj0@=cu6T9p8Lw0JD$0?3iyjwpIsTU6Kt*B^LD5=!D0354}q&!NFPE{fFIj%E`qjY8y8MTz5==9RLTrqb$q&1-z0OkD!9l#281I^z_5?^`*0l3w==r924wg$#SVqI`(T+opH0u!*x5K+La+((Xe#U^})?W+cKnrq5Kl?9rve&Vumlk@H0>An!VJev|zjH)?D8)ljMT&8}4@*!1FBr(Cl)^=EGyyL&Bcw*v2p^W{t3QyOwPdkYwchWgc;FhgHz5tp8fQyl)C-e<5|><vyIg+xumlOk*;2;3hL878w7OSie{HnG)DiAmfO&YpDg-zGNtI(;=U|aS}zg3_u8=u{+Wf9oa}XM+dShN+@`vFRz(Uv8_>GwnX=n45sAAq}Y4r@rP`mtY3Qwu@zHO+v?>f_7_}>CXh;1Q-Z*Si2c(yYF6<KUwA|fc4^w5n@13f`X|%IdA`io!=q|7>46Cr`}<TYG;rl&U0Wbf){N?ebt|Y^XRQ*+KgC146riRFmwj#Zlr;xmaC-=Ee?}%OVXU^uje(HAxwf>KBS*=$HAYE<&hi6lPWl{5K~^wK<^t&gL0LE#bx=mv4*fQUC!jWI1Cu|j<zn*nTXndzYCvl`+VJO8oYzm#R|8Y&D>8!3AOR{jY#vBz%!5@PL38}Z9)Q|`-PNZP?Z~w5Z*kKX1?oW|@|USd`t}ji_>>4<6Xq(gz=U|jBMnY~gIqp)?<jt(kbF+%b8tWzD!BB&wI#v#xG)q<aCnl#B*;sxpdy<8E><_T6j2a$teD3!MpK!p=}e+|FfvpoX5YMZ7@K-DhFJii_a{jo`S>294cV1=S|B?G<c~Ku2>^ilER^&)&xEoZ>HB+V0}@>HJy3u0iV%kB<c4+0#s_17z~rRZ&%#j1K3p)%9j5hFHs|y(SRbb9I9Sj0@dhV=%_56`ar9|8aKgO}esuCH4k&focq!laCa;{lAkM7Y;=OuEYofnXpE0WkHaflhk&>VT!sGkM<*up?iGwGsD^H52t?lz1E#7Qo(TcXob`4n^oOGZJlvn;R$H5$VZc=|6?Y0z!*b6`eWA*ibF$w_h1Mn+0ut>E<xI2HojN3QoIV;20#R?5y!;P0Fm4uuGw+PiVhW0ICqzy_C(y-+g69aS8EE^@A-~uPxp(ilNrY4xbHMT$%6=vC3{%Tufw|KOmlAQ@<hL4`C#IL+_uk3rqZroRMkzKS^!3S7`E+>PhHVFIvJwL4#!owl<xM{VlW=&NqLUnzqgWj#DhD<M7&Au(_9d8W%_wyw?3;>aD&9_qUdV2c{OI_r0^|r1YcsOay?<HN~DywsqfZ(hkRx_VT>e$J62rw%K!Be#0*P}?8Iu7&P@mQBgyp`RHmY!n45_VY+VrtgvDD*GGorn}yZ4-Jz?4;@T1a+G1=X5`vwBl3SDQ70$xhdzB`|QCYa(U(30TW%G(iTV`0nx9o!ToN_lE9y!T+kjQ<-I)5yKXJ<H?pUY-+Y`Oy6YDD(OnEis7Umd_N%))IbC`>N|WRA7dH<Nx!GauSQTVup=3F2AEpizVm4st(-Cyl7l=GWyTy$AK0|%}xG|FvTi0_IQ5=I}qDp+<5}$NR^_hDd=kWV}k(HB)oB0lQ=_SV6&K%*leuS*;dm11*J&IOj43jG=)X`mNBxjHxK|2^A<7eSu+dVQPOr!%%-pnH%o(QofE!fe9Hsq*~l+{WmvlLdNF_Ld9Uv^}c&=$%tUlX(mkZQibz+BVMh0mzJ@d3%bz8yv=#nSa<{4H*gJLrneS)B`;K*|*m8e6mgHB^Q+j@h5Jt|^}hlwrjZsTiT?6gZC*2eizTMjlvpKqrNCK(f0MgVh40J5ZxIKBKMu&(#H62(%RFy}whqe>YV#4&|jKXf|NkgdZ;q-dy1CZ<FjRz;J(mL`7C%-w=F<CRY6@LenU<XBrHxvJaUk_-Z^TXSdSw?qHO>by9HqxThkhj19<MNpmz_ngY7w48m)D98Z6h<7hnXf@tLNGyL0~0$~1}B!vi)`$^ITNZGk7FVfPXNf_5PX|rlFM{}tBO)bZdWI~l=?au`2Vex*84kIhw&$8oKLsoW+p5SS6)d|)};{91U;ig5TFcQ4qIz+wxqSoakoH0d39hL7^$}q3>N<T;uW>3ACXmTcoPjw$n3hfw{Dof{&kbV+=as$VwZC3(jd)yCxk;Q!Imj<|CSdQ3jAKDWjU0PL<s2+a%0r2;3G^f_wuK+DZWy>J+DLgb_^m$C`QV5Wcl<5}PE(kZXB2Lx)OvXtX8X|^*h>JI4z#pElzvW_^r2bJbqEmbdk{<68V$Km1uD=EBb)qeAheH))9k$;ASh*-4IW?2Rmv#5wG$#qn$hRrX4P<49v?p6@f-Z+8jyB*e4pa!l)arq6%)i_?dd<t2%9mXo4jDKT2<7S0?yW#VX6758IyvX3i8lKza98V})%cObmEbQ9`@_HM%zNTZViB8QdTBw<Oagy~q8>gYn1;06ROJabkmqybjUtVdhKZrBB7DgXYC=>peaxZSO?$Sr`takf${Uuziaf#MIZR4gg4%{8X3f=q1W3sk(RY^!v%LU7O$U{+gnC0ke#afoo|0DKA7VfVc|?q8JmSi3b=`eV+mM&AN_etJbso^n?P%;j{n=aTi&yCa**h3aY!BeH$@X{I8S0lJnb2SX{w`q(Z{{A(#j@fMzCeDW8S+7O{(!BL82BWJ$h<OGuN(Q~DiR;6jWL0%Ty<XysvC7Tp3!o97c<|xgQMyh0-SP{+?)2Gvp0MYUUmWA#Y|g;4mA3mQ%GSkKEdfFXw^efYEkYY_io|eGmYPLW_R&k3NhUNqO_~R6d&|MJg8;$W!$bh_e-AKPYfW}FqGK|MoFu!eo_k74~3=aKKT<%HaCXJIY9fcr1NGhw_>c-i;(R>j3E+q_1|?O&@+?5&T^&kp5kBfiF<dr)xHvxKyT$WN3v1aPX5xdKNE6VJ=}(r5B9I#V$M=<ODzDT?#l$j899gwuR{$fxxQ2HqmYpVz0WXoiE<i#Vm|Emy9Ng~3uQcucVEVeTbqN16<G?VTaxc1_x?p_1okgO>D=%8{__c_{xAHnW~Y!xKJc%AQPJP4?Gdc<BmIx^2P!MG%}Wgz&t@CkZ$)Nl_D=-AkBSS<IEtYFerHQ?8s{ABnp(d1&%sKD_%^lL3FkO1OjywcMVk4D&Oyd%Xs&2Bmk~$~wnF`Y5^O;d_+&IZ5t6a_b9^sqTV<!*<x0ylf>G21+_MJ2&R#e^K{Mr@=0$4<L{ufehy%VUKMFo0%uSbwg*#+jWC46d3+Jx);i^v;+^LY=;c23f<dEbK_=M7b9VHt^fgkB9!s&J^%T~=B_GI~gAT7frx;uoIb3|MSH4$9_YR%9xDympkK7RHF8c8cEjDs$3L#}Mk=b=pqRJl&onaO;?;_N@c&=GF1FM}wMc4OQoG}~6Qr9{J@agSF^mxIMRLY4yMH}>X)^6>iM3-Q9~(DRFX&Im}e32|>LkNX7eR`z6$haS|w?M8aSdyV338U!NiFKiyh7Gg`m2mK0ZW}o0*!KINGhLoe%)}Q)G%Rk5b@fj48wO+#4BzU-aE+OKr9-{H5V6I<Bo%T|vF92)jMwGEk{bIsOj143Q3BQ@s1fcA5<=1JhND1G^oON13+)gLg+%Ss}+*b3NdZxksPTtOuuUDJzD<H6Owwy(1gpfhoD438#jT*R&zDNkmrCcbaIFtc0ZEA&8$9G0gCkNZ=r=~HD3~>-?ZwVW>*7;dIaL*SKMPe{Ax@ZsrQt40awFj_$hgukr3miaY5ryd;sm}N%L=kt{(h4fucbe!!(3Z(%T~B?dW?KutDSU_$$3eWBU@45$wCsOrGnkjK`ajnHCot>|xA{bY6^~1HbF6gvCDe5XxEgaU9p^N0tUxDV$}eA0ZvgNTou{L&rhR*TJJqS^R`PfloIR=pb{1%&G3m185#`Jc<1DaTh5DpsVASdUI2_ZzCpQ%QjBaat)hlCP@V@zPCG-Z7drsgsfvwTm75K$kFjcnHE>b{JuuI8r@#;7yLzvUgduU5x{7=J>`hAlF-nx3W(*!m<nu*tTS=8*Lktz9ONeaAlh5B3jH@q%N#&DwQ25skn$gd<V&FfG&YWYJZ$69_=m5Ap04G;0WG#}uH3HKV{gq?BthY(x(NiSU@4N+pO-lI`_BUg`#!)&iyldIsh;6Qo=DgbSh_JG4{5j0zg?7gVMy-eRP4rJDAPjran0qez1n3<rfia5;LXl5+F56(^#1ejRQw=v(iNZi|<J3^p(pvj9WKt|v^jgPKMN=n5YsGL5N2YWcE`fid4*%Z>gC8&;26Yf4=OksSOb%v)(Q?KUdozqjm9vZi(&x4zf)6{DAuFj)VyCx%$RutRhtZAdt8X+7nEhOC-95GPG7@v`1<CDTiB~iIE_ts2q;Hyo@==oEbo&u4=0z3x*m6DNzLiP(!tlXSvcsryt(z3S;tlSv%j+@!yr@sf0bDZPd5~Zk%1YH$m5LNH)I7iyn<Qty@o8%F47;#Fm_&@~ry~i!`a?Un?3FZs6fu|w~fbP5LGRDqMXUufnL4Fez6e!+q5d_!&7^8}Fg{+K$#f=*#MhNbE!N}+BEJFebH1;s>R>)fI!krcpcSRm@^^)Yyk4yp=rjba4UYb{l_wO^~m-Z23HsGMfkqM<AWol1fNmACXa%!gc<OlOAn`Jt#Xng;{psTf9x>Hqn9w+&Q`<z<U25?0_>1AF%+ax3OQBO>S%DH!2k}FHBqN;;V+9FhM0jk-TLiSG`FBbDif9~U`wC9;4Ajer4bPs{Kr-vwoTLsY7S4JW0=u3Giv(wvkQtaG~s9`=ReX!Sb2{vMJO|R;eYbYZSu2@3;t1zfa_m-DCvHP^Bnt|*;=32k!`Pl8G*?%|#1T^0E3wa#PxpZneJ2Uo%FbR|covr>s4QSDU#++Lon9FoOB3YyiH=>GHBI2H=^nyVLpH$Yj+ePAaq{|_{qyO|AqG#O2$t-m%y!{yOltA}fO2I^w-;x|!lx3HG*)TN+=L_noN_a6fnE=}+16LpXweDsChEcpN=H)``|I~*3VA8wGnxD;x3otu>3kyV<7}(@q_ia-`KWxYl#J@@g{?t~`-XrIL@nZCxMI=u58Peez5e|INVZ$dp9u1THMtAyghjz1gXaGWlh?soBA;D8Zi9KRVURj2LG?#X#=7OsTNuYE$Z*l?%sLmC;0<I&zmTdl!h!3$m;e$`9Bijqd8T1C_KIacVQ!YyhWyu*nq@_*uH*YWrLbZlvo3$p#dTJL%Uoo7w!edY3{9IVh6bv1m?|03#blj{7Wk_km0r)cUBP9_k`+x`!qF%6`Y0y9Py-{e?6y#;()f-PmFg0YAInDfgyJ%+8gpco>P+zA9q;&bO5awAIhQF2wBt_ND_aY8f*lhrrrks%f2nyD1B^m8f%|&@k2MPDs(oIG|quvsJB!EZ%a&Esit&oc}`#<=3Y`1TH3m!CjmlOX_>ldesZ+AxNU~{)dsgN$1d?KC8(irBf{X&dxxnhZp>=H4^z*{>FT^$KQOe`l98pv4<yh=RZOveVLmc>-Y^2yOO%avdJ_k>5LP!onl*$ZojORK^*8>Q%tss4TrOvaDNBo{l4-gDBm6rIh_Dj2T&x$&^LON~wr0nKQ}in&Pu+oI-D0yv@6_TerXQO17dG^-!@6iH5k*Qp&<!^c3Y8v=p`6n+S?CC7&!WDe3pUjb^OPl+*O>s7Hqcb%tV#IEW~DSZM{t1hl^2z&Q$J0c1mh6+4H@`ZM`FQsL4j~J06-0)iMCTYU-NI~pvhDiRzj#AjQtli3%lt>k#Y=^;)SOcM<PX0RYMKjh<>N%z@zG`8ZzJxD$GII~WWp4+vg5g}X!2@=LJYhuT#=ZS21@AxF#?39wuLbzwyP(a5d*AC*-%CHmbmCRA^aMSJ^i%jQW&d%G(>tr<Lpt$tgjyVqb*{`oKSu7rbXJop6T5-c6so|?wdBrBf=^kYfk-NIWLy|Y25m2|R&AoEA~gG-*ehe5HP*tOi>8*IAxjdl<31a#zF~Tn2`o;EN35tem?B+wC!`Mqw(LZv#T{QkhQ9E-eeWhV>8PD`%SG~D`g>;^28I?AATT5h{$l4U`&e?uNTfFJ)bOjqpHGZ%i6+TGsKJJT@;JsDuEI+)OT%wBDlB(;7H%WT<T@zh^H<||?>+;v_(hN|j~*knql>A3zC>RvB740{Pi;Fa!20M)U8D#vwFvghoUJ);n@FB+!?BIO&6*;77BJ?v5mm!|F}@MCwV`)Mqi=%Ff`pn$XemMbSn~9vkmiU9aqHt9z*5-uS5kZQ1kFS|N7v3pI_&vgfgk^fkP*X(<!nV<d+yq}#Ja+$w>bsNki7%$VX0%d<F@Mu4J3~I1^KHu(~!pcQefHWXT{0x^xj8TA=f<@=Pn9_83VdltOUvwVF6OSvL8e)?y`5nE_gc2O-n89bM30P_(`WAj{YFTySz!+5p&r&KF#G6;QD%m-Xs}4m#fBhftoR~rsebsW(u~7n9i5tc5GWo0r<3NsON?6@=3{0WA9O3O4K4Z=~}4I&JdY2mB48vI(%cs7j$eu@X*;=KJ4EKnkhhDGCD0NApF+8HcaC#OZ$c5HS`}qa0^Aq4co5by}=rpiHB4fb;|yo%t0vK_teq18Hp0tw8{9t>4`kJ$VVp;GRPq!eiUxlH&C=^LaJA6;PiO9Op2F#I?$DC*hO!%euO}eaAFk^ANaIl3#e}&<ek^hWD8RS5EScSoS!H1?y4rhzud>Ls)8utociAI78V#-y=Ere+Gozf;3GoQyHEkEf=W0R#)Ap%w~I`KmJlWjTJ#9f7@PwqU_<7DrEC!FCqyYLFgnK?^wLGL+NP|;LmvyT-kXSy@~>LrisL8G=})f{q0RW-Kc;=CLrrMSs>}(7bc;ZVdEz|TAr?iJ8b03QmFn0PfFsF&lZ8`l$aj#p_o};nwKv6bo)<pHivPqf=rP>=g{ad=n`P%|;-xU7%v!s%@EKO)<)BTmvg8>CGn~s3EKBUhTy(HxZ&EGIR@#D%L1!mu)mAd)=dQsUm~Zf@8B(A?Qp=Tbc4;OPS?Zb1QtiVL;;w|%(;qPg&c#f*h#U=pt-H&TMz3lJYZUF%aiziy2pGg#vi)hYg>#Hn6F6w;WDSzCwZiN5u~8|(tFXUg;qvv^Xz(s!G)#Hn-FmW?3sw`*06=clyi#5N*)nzz%SGb%l`snn^l*~zE$p}bp=bwnGOQah@+61z;C~D%YJC3@w}9&g2W>!u5Pa;)9PPdsMbY_gPZA|4iYH1ibdD;;UDcTo0V#EHbmUBZ7xj5ne789?sZ-*%@8VyoGjZ<q-3{cIl>s4dzuXlYzP(JPP3?uIS(2PDLp?rbvFWIzvpFVwX_J1NIH@A0;mTP7F;wtDRn1&nM5)T}4S*9OxQwcfFnl;QNNgvZzKiUVQ=r&@2p1sBCw8OjDr|9xlyr#T*aBU@Oa-woSPDcD@DXhL-)%`}RLDh$zy^6VuqP_FjK5qRgf^RwFnl2wxg>~=a5fb1XVWytNuhSlmUQ|Wi+D#QcvOHct4%A-$`ZI-jrJl{koF5~#rn7mk4~$bx3C;mG7TgK8TlHX)lY45C+P};5^Z}wa=Uflltd;n$P_Emm{DtRbmz!UzzPMEhY#g&CKPZJkLWwJ=d9E@)E6W*j+lg_vE1d&_In1eD|%L3Yi*i}p`F9?cu@Y%gC`|jssZ=$k%biWhwC#SC?y#X?8`+~tkHuDjv{wHyNAxdTxi+)!qRP2?EkO7_f*4hUwT60tHwC|_!f~>C<YQfv4bJCgL$hEKI$IQhA{Rhl}QMTxG>>|ex+Hcx^9Mn=bq0oVmiUKl4@C(H&bq5_p2U)Oj5JJx%S;7Dpj<vUb=38f!sYe76u5w)zRXEs5V6EmPQir`wy1!1sR1tgV5QDq|#tA{{nNk6>(ivZgZPO#v~<BGhR#)I01hl)032Jots{Fa~#el{F@+yV#(AdfcsS0geTTh4}*(D)`duL+=CX1dFzxtNVpFc-!BrM8d5(*H3^x8aaZEXHe3W;*Q10GB%y{%Cy7b?PP}IA(h<K}i+R5RK;&fr)#x>o_v>kBnFiUk4%?z6b!@X#qgBnHYFx()^Af!%>|&%)23z<-x4WG>wS-Yet5QRvPBC6k+H~kTINp<hv%Ea0a&SRk$V7eZ>%Q#P+&_twGuvs}OB=K3CHXW&2#T)PM`q>VK%$oDL%Ti=>#`oX7a|)6sg}I;ms~aWw?)KE0OZ|xYMoY2>Mo#SRcRwC0V~3th!%Q&o*mi(jVn}UzF;>@(kj8-Cg(07yxx~@qfkEZwQ-aCqoU6Mt5*5UZFG%if@ohP7eOF`s_twSz#8qVNb*j_PV-z8gMaiz<sUrNYds_uA<%6hG1uPtO=lB71y1cv9k>~cm;=6gya1{A9<YWKUy;()hDC;$(N}Je5mjc5NJ{ePiv)_v(#><G55+V?<l&j>_RFlh_~NK7da6YayhLbr&i-Es8t52xgg~}<XO<s6GGqaUYSf-H%_wEs;YLe7aF)r*6$n^65cE)R!UA9~fhp9U)PgEr-~qrDn+6W|?)GdGw%yCx_8*Y<S|0NJ^8DS72rSmWV!UE%z)~e8T)~_1N&IA@QZ13i(%yJ$-|T~YA~&LQ!?Vk!nuSjG1%&LWBAR(GIK05OpfcG*pXZG+QZ+Z%yLlub-EC9g^^@5Co*s~K^ljmQu0d|^37&}4=jm>bM<($Qk7x;GE5QpSk^X_fSlRVUobB1xMo~LJB?cog<}SQ0lqz|6Dxsiq9iAoY#KVafgBfrJxbgJjQRWF8*Ge_qbjwYl6TX7GAeH<SVlN+*nqtsee<yxql(v*U(NY9Vm1YU^^Ao5j^$A^uj1nih1(X6S;O|NZ7djwPj@^!&G-U-)z`t^jlv=Iw?QH~eAi=9LJ%1kUVg++SL5TW6O}zQkkd5&nSlw+S)}!@qDwm>Zo)GW|)(d+2-)?BNTb5H+CEI`%<u3MiJ<9PPgh|_wbhs=$r*}Q!O<YlG^e~8aLvonH&-#P8EUhhPs9oNl9(BDOqX~PaqGpLPWI49)NKzA>rKibem{6@Zko)k#SHkKi4S*GkOR7|i3oV+?JeMjYzk6MU6GR<(WQ?dcAbLw>Ue!K|=p1137nc*V##3GefG9%Q!(nI9T+n?n<9-8@GK65W?#_BQZT~z$0^f}RKg@)o_=Pi&>(&1JW=t)8=rD3ZkWIFysWEZ>EWQkjAb@Qt0B>^fiP!J|eG#x^w=Z-N<iw*}#O5UVF#(&cBbMtABN=W!AYm;C+w;DrA&Xke3>nZd*WyYq7e-Pe>hC{0>2WcK8iQmZmR#~ZIO&MAG%crpH#2Q_Oj{a9$`a<^TZU_x>XS+AzV;Z5`zP+7c9$3Ay>&70?QT^}S0mqYNDVH{?&n}SoJSnAXwd^4f)?<6_!O9?P!WRmQf$Mwfb#6Z7DqPm>=KZd;0XM*p!>L%>&|?g3wmJ|^1)3vfWDrUwnz^iRMr4Dt%~&c5OD~%)kd8Sl@*LVK;D-yl(Xd9_l+YVy>e+pjA)VQQ|PrjM;Py(b2sR1;=e}D&ha~M%q$7vdUeN-yO;$QTDk|$g;o*Jc33ifWTgyQvX&<*hC10(CKetWs3mB=z%fKYvURJqBMCZ(0$_SZ{Zdt1C)#m$c|Up>7+QpzdCfe_T|zB|J;esEj&2qUNgl4exTdjLA5M_tiw!}V<pJCq<V{D+5Sy+oI+P7f1YD5Zv%9x-e-4~M{DK4)*@t@RM>i!Caa5!GVKhB5JS8WFl0Wp~&7a_GVF=@lq1h;L63Cig8z_E9+UUlkBJtoTZEY?R%o9cLjeHvBHvZwR8E~szn=x?NH^*<MHRPbf#fH~2O(r>NLU&&~7O;^G4c8>#$m*uv7^2LR&G3*ivq2JIy?W8E^Y0QF66cHfPPa|{%ZY;gZ#odyJ={uN!o1}Z{|mp;*t)aRHi_SKbpcpwWGl})uOO$*K;vc_0wrp2U>LwR@G}c0U{eT#$Md3=<%GO9(grQEW_~QR3?rPb_RH&=Hzu<u(;nSI6K#`*5;X`+n^uy5CE0P9MWF!P#swfIv+!pnL`Ii!rG5;b{d`f>LRNkU#cB;y@MJYIJgOSrnG{_qzl$e{gN92j94nvN6dA`#Aq9|>w2n|TAJiKOh3CAwfzT@Os%*cLHMPwwh3+@zu$|IEK_>s~0{xsOX;;&X)etCB6QS9($=8Z<F4$qy)KMh{QdbHvb_0u)sD^u8x}Bh(V?rC5pFobx#JOy?cKx=J<T>sEgN4>gQ?#t-%dPb@A+Qb4e6^<>UVG*t79H6QUi}t){g(Db#8yP{GcowL_*wssEFL_BKgt)kOa*UXBYpF<cNv!KW|=$R=y;`OZg%o-O3-laU2)_0cyl?rft&1iekI7XpF+_jxRLi`;TLoYS3$blbXhvWKbdS}BMhMxFl?iVNDst~r85SR4+>Kg*>{)oG=jz{9dX3DRp$^K(!f?wx9Q|gNfg)O*WZr9HS}=GT-02dyo=X@QFSI&3mw*6ggr}%sJWpi*ZzUHp084M!H9K{LPs%|kGK8v4$TVArJ+EZcY(b}iTAeNh`6Bp(m?h@*O68Qvz)CLUaMSQts4K;)a`=Mi{ACHYBNfL^03d^Z%}QpcwFG?i(0Sy;kb<4uBz2e&p45uD&aIj5@-w9@9fI%f`fQW!oqk|IFo8#NvwNZ2fTx;ocWB$3zQZk;*`#+AluA@4Pf5N_jo2mIWrdP5AC!7Um4^$D9yURn|gl&D`(SoE-*HjO9v+Xpuq|D<FKO091`p`sg*V}X<KU`%@Sq{Z7?P$k9m^9TT&q{8*d&FcA&oo5-?<`710Zs*mv9?|K0l*;V9<oi91#ge~km3r|}w9pZEkg@nl9>02%Bx>x8Wi?mIrssVSYUBUVZRTKzeAJ$_u!`}v&Ngt``cf&?U`Vw~}*>`Ij0E|~NQ5mlL-q2mf0y|q@OpTJlnqq_*cr{;R86+N;W9;%$xsiCOBb$jtZAL96zaK%{&n$L>^{lbFqe^UwU5h=GOun+vFAGGok@D&?Rx=*@rQw^K`Jywo8dl3e(kNgoj%?-MOm8v0b)<7-)A9A_*GCI#1bdggkQbYdV;1RL_<7FcXd+k+|DA*te@;aiW=&7L&Z+MDr1$ob=TJpzEA*^rqJ(*C6X|-N{N`bC=M+!nQ8e>==EZ|k!&;P_j0hKG0k|)_pAOg>W;!=3qOKKwd$XsAd_lQ8cTOA%asNoI7FqyvEAx5yJ2T#p0klF?-&u8Pus>KcQBtZZPQ?+SWT9{ZJjshn#ckwvHODqV#Uh}HKac`Hw)C0{eI1e5W#|P2)95!~-l^BTO8*3>)JCgWtVIzl_s3rI$Nm7UOw2}ikg~Uel0Yc$H@m^YfeI!yf193P3K=(SGnPLP?woPd)hvm1|Q;--OupI~Z5uKwIsDmn#R+yq8Tn8MyUZRE0Yv#b1r>2%`KyS19w9y9tlprfz=k+(b>@1(p&r-Qzc53z9C#mz#`G;D<ICyDOEKaZreR?%gm`oDcv3oQM4glKIg$3XUuVx>uHmUusL3iIRTKxkCQW3yQgwT+3T;BA8-UHpUvYEn;>U+lGaNgFv)_8o)!lNzfk1&?PMq|Kdg3q*E*<P$H+FH9c&y@`w_dP6dvCpD;zC?_gTxvsGK8+*CSP+6{JsO!LabsigdnT%k0Ns?x)CW^#z?_cBCE0;HSwL7YseVZMMjf5&<Mp?t+YwyCOkQmx@g7|HC_oj&Ui2I6BvQP8Hz&l7y3x4CkYRWfm^I&7H-su!S(%TRj~$toKV(yr@X6o>B#4MKL<8nNoatm>zb2#I@c}A;<vzJ`y-ciKXPN`l(g*E)s;W=EqCl0&2EA_C-h0}sKw!#hg}_g$$v2gipNMu2nuD0Y%f~quXI0DfQ(s9KmX*aXF+M9=Cl0(2RLEpk?pt1*kVt@gwEPP!JDBL=Hr1p{V_6OBlo>gQ;ap!ub6F*n9gb)pVEpZ5^qZgQj@w$9J5n%1x{}fA-}X{CWY%Qb`YzT<)pg=IX!>JC@+5p}-r<yqywz}#<e*h-qP}izZI*E*3-i~Wjf7tbqx{@#*NvMwGs8RS)T(O&15X^~G5WHuHq%61Eyb{L0xm;`5gKnL=1sn7zb2#JL>fe2SR@=hsBhU#A*O^>wq$>JBBd+jtP@t2Ll-Y#6lnMnCpIYNGI-wYv92ksK_j!~nM(DAGxPFY^T}APSV>Ej;?7&U+=-H5L63A1TIvNX=kJDAm1b0IsBm>7<>98WQ7k6JOA7V8^Pm_(eQ*vEsahhXj5}tf`ZW!+)8_2HwORMV0Y|qW5ZJ=w%T{vZsPP~rDm^yRx}I;TN`Pc357{2TQ=?hK1k|nq@@|zC=Ny#YmAX2#GXF*Rg~DbK97^hg((hf%Z%;HxzLf4ls5t8@%)d3HLd>hllhZC*O8dh|u9fvDGL#q;A^*14Yke?!v5ESSX7Mkuivpb_sZfQz7L{omHU_5lTRzFSeyj&pdbkbWZzZ^O{wCR?K5%zMaqCk3_@O!Q$%U>D1D)gEy&CQjspGfWCSfV(nsd>e*>S$n6sHy1Zam>pWXk!`BKI_!07}5~X-%_U#@O*a6G%Cv{ih8xALG(n6v&WUDPa{7{xtMMR>lQ;LUVLVgC&aPH~Ts2op7_GQ1HisK8~-eWWabQ>@Yuo%N4+LJ?rBl%V~*n=vm6TeclvY2X*_RT%E2?JgqfLPe#nsLYPc=k6XifeaRWn5i-mYO#ysP`}rlXC_5OE2C{6jO<uP@w{)@ml0(LGSy5&0i6}*y%;m0z4t&jg71zTtdNQd4{e`i4#<)ow1UK@T?NjLu=WX2G|H)E&4|9v7Knvk1KQ{1E?>`SIZ@l}Y!xROlnKLIx(gc|0TN^n^rbnCvJM<!MOmQ_8esD4%bMAjZ9i6M4;meU$!kcm0JVsNt3^WVbYo1&y84%T;(=iOILr0T&w~WTcD{?|>qtA}TkWyEo)m^)kHmumX@HwPd5e$Z@{<i`pP1&RArxS{{y1<2%XvJ(P?G9e-oxNN%ntkAAeWtZ%kF2K-P?%R~N!aY@M?_#nwk!-ekV2#Em16kY__S4ImECq#Q0+uhEQBEMZH_k6l$+rV8xX@B7%unuHu@IOh-!%-2Fr)Jxx1_eU>xX{$w}gmLn>x%$vW1Ba=u;Nd%FrLeSkh?FK992Mc`WKsh#`RD^EH)*)MQye=WF<RDg=f8YQF018F$PwK8BOxUp4LY6E6(v4B!e#Oyt7040Xc`X%4UBQNj2VPk%kjgHfEoWHaAcAVU`;3v*xxrr*uY2wX1ygw&6X|OFk9(q#wwyO4bCRPH|j!^r>s+qlNse&Lq!1@Q%3RT~^hRZ``6J3@+uP>A-$1IYwXvBf&!I=lxPEGZM1(ki|zu)!48Oj(NKawh3ZTglYSs+29zU$;Tb!#ZocZW=B+H~|Vsvn5p)fo4mTQ9+_u~)b5m31_i{?uw0Y?8iUpLu%>Zq~mh>e~^!mbGNI3KGF=s0c5P3zY;$v33WCWeS*<jTv49o$B_7)~U=&i9%9*!hJgsTw|(UgXTc?*H5%_S1rUvLB_FzQ!9-*n48V0T_!l!f7Scws4wO8a9$4Djpyl<>$+!-NndW_1iZh23J73cgyaB@3~1dCeT5$%KFldurQ&=0T+ooe$e-|g3U>}aCgc*-7Apt~-g}J5ef-nFWj3u_E#H-=ZTvLbzyJ9U_i%HR|I0$86lwB>Go*w2JDPa<bww~H>V+!tpq`%FrRb#k-)zSBchY0)l026XC_|<R(KbrT<CK$(gs2tMNAwT%Pe81~BG5z<Su|G{fIcOvMv#9E$gQW@X2;mYkR#*4F)65gvh+7U&vRG98YCFGmDO8|S1=F(!StB5iC(t7KnH~VlcL#Awk`r%NN8Mr^Pt@}_uM4ag!gh=Ii<$0y(0~7kHdUO)X(t&7OWTYmaIGVKbG&LA-xDeNfe8mTmj1HfaK@q*8H23ho}YM9?j_abOHb_00000AnbBpFMihr00F*~0o%|684g0`vBYQl0ssI200dcD\"], \"k100kit_core.py\": [\"db19969dcd6be4da263271904f4bdb598cef237a307a9b31960ce2723bcb63ef\", \"{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;Rx+Wn_U1AXbna~Q3<Lo$NONZq)Sj`J;pN0V2$UzMgFW8HdaH=y2%hY|8d=QL0_Akt1}oU6ucrz>W1*tIAZMnMd$v}F~(xZFGOXhQ*2fnJkhKRuAVGVsQzy63qLCRhuh$xM&8&dh0ZZldaSU03GbRV1F=Q;PP)@Y+G+zI#cMdsUOjr^R7|xTO_SMPanSQoD2aC3$^3c|GIZ`T4U+GCqgpYfXdk{c>ByD@fwm({y9bK@zr63t=XXIeR|`@HTZbxn$u>j6x(srPI=hB`hievGL@|wW>=6~TU^>I4w5Wn98*O7LT8QVGh84cdh&ZoD{>`jAo6byv$I&vM=6Hs#2}LIgW>88LI7!??n6L?(j`gA!aqG8!v*%h4y4fVz>wKmmC#$N!ccgtEPq9?-ef8jwMBf@DSA(y0g%a>wJyiK5K|B4>y8Jdo78V_c<TV-*)?L@$-39IV<u2Uax4}>5l>Qr;GcE{6YU_s8r@^HSp1E(k?%~ao-7|BPLySkSrq>RgLD@>=VWzeSaDXfR?yZ^|{jV(YVWxyfr#2MTVMN%qEmM8$68B}{DmuW7HiqM~A*g1Re-OUqp|XuDu2-)2D8sj3Z=H-}7A5A5ymtdTqQrzZ0%*EX76RT*DxTzk1FF-QF$|j%StH|H7i;zLfE(Z*yA0!cvp*Z}mP>gOZ}fF^Z5R9~XG2BDj0XE2Q^@GgpWUS0Xr}>5qpo9C&1M#<;9T)@?mPC9S7+UyGN?*hK{G&vx_2fWQI~9aq6ctP>$=jkmGsP$1t9oiOW;4O!3gT~^<z|@QAo17c-!)0OwDpB;m=tr9<cmdx-@`9lJSH@Pz^I>cf)a`FJy^%E{x-A4`!Ae$)xeJ<N4J#2l-{Y8C>#4us){Z9rpozzG6KawN~KiHxKWk%1byDs|+hv&2j@Gi+6Ko!GoJh#Lek)z8qI5Vpk-cuGo*+ivI$w^u_NsgN=UzI?r{#`QAKP=Kba{%FB>ZKFutaZ#r{GX)*_Cj2wF<VVYTlvE!G1QF#Y#$_|6rH~&eAdX))Cc!_t&z#ttqiN@C(f~OL`>K2ZQt!0hopTGeVsKOh&!&43V8{~0mvZYOT;L;7LxY&Ywt&ab&S^eu63N84eS520SMncM{U2b@GZfGd*5uxT)O1>`N8!u<k_R_23rC7pxq+t!Z9BMazr+;=9Stb3HE<a*O7qR7m<s$})T$^1rWzC)~;}|PsS_u|N)aYU9@kln<!$`21=%bQ?=V!Sag#Og|FuVx-{ZXJ#sZ(V~CNck*0WS!2GW$lRsT6BEevkcaMBa#~YUB!}zj(!a^||Mya8yR=yS9qB!_cajLMtJ`<_#T?>1z6TK2s#G*YHJIfVP&7Ph*H*i*0k_S?aq6c#xlny)wI>7G*#_Vhr9RBtUUWlbS4!b8Wym_gY-K7mGe9=U~m$0VLXV16__ITKwF+$@2sVZgJ+3#hIs-_7FoE+Xc!n{7Lha3?qSeCJDR7%hARH=$2;zSbC8=GDOiv7h%+;sFnT=k2XdpEB`XT=Fb<7Fjt-8C-e9#6^;Pkof)S4{E$4>C~d#Rz?Po3{Dcn{4hSiGv4i2oA^+v)Md-oHXB4JL;pC{_a0V)At3yfLR|0Q10KGb_?`K~`<qfWcd7*LO7&<YoRj8QFJ4f~wWS?RewL4E{l|B(ILTDzw=MJ=aB8uQM`I#DhdM((~!kDncQj!N_h+Jszab;aGn5rzMn9nUY8Zu>0TI2PTCSsZuQu1e=x9T(;xt{84?fI-Y=TG!365O~37bd}&KH{BWj3d5VuQAKvzuYA5(^T+m8qXZ}1Z{hVPgON)#*qdaL~HSWP|X0`$v)H|u@C6&+ZV5bY<;G&S|E?3Vd<p;-^RV?ITA(>lK0OB-GKX<-%)Ib%w7~=FJ8&F-c6>EVW(D*xZ)9;-$Dgj>E{%;MA}_h!B&Y@ef$fP>IS;>2o01l$0`A?-X85C841f@2e;W}U3O`C=DNo>%xtEm2{ME_*s)UJ!n<mq^oy*80Z-rU<I`Gb1a$u5;r>H@|CZ~Z#n|xYOhT>33gVY8mvM>b`CzsoiNzj%tn&c~4eiqZT#9Iq+px<L!}5BX*sRzp=4U~Dg<F7;mak5?sBSXEs6Wjl#!UNAToPgZFF+lZ5iOaU$L*P9DV)fnT+vC2cXBiOrd>{vSvuz(k0wj`pGzJzJ!%)dZT{vN2aI-6>{}LlV~!$ijG}pFWt)$XvfMJ@S~mq;M1sW^W`q@;syjqJd5XGKeN?Xsn0*+BVv|XcbLgF(8Vn798^o)I2vcRpI-syR&*7Bis4j{;jW(c_To9{VUCOGCUx{OM@uw2~TsE?@5g8T=fq?|baTpc$sOHmlBRP~OlXI{DbV!a!vehxyhfYqR1Q|?sdeQp;Xo^*qd4OUNW3P%m$VUTJoZW45xyhu*#V54!;jRGG(SMi3Su6xM-iRk@w^%d^+37$Yh`Cz*4CKu}b|w77LgVD@+Ih7?9XSDJlk@%Hyf(jDD!@@WnDrq|dwF=O*J)?TTp#oK37!uu-@ZMA0yp|HuHsK~)5gi~44nn9=;c+hyFi3t$7n45`QV8it?{(>lhO*^FmK6Z*%25PhSkqSgO)?SYFq-~(}_zWVep|liWHUQO;iWWC1Kr?C$FDMy@=?m9e`M1!l8dq;FJP@>MAp6%{a->7oc_JpEMVp`;{6(&;o>68*5(6JL-=)4-$J?zRa!4ueA>Pt==jZR(nrkoy(&bjzfi4jp9)R@DRt_TgrdUA2FFjw*`15CF?igFU@@YSgQRxLRCuY&qzW^KB5oqr-+R~jwM_DziwR&4m=2qC_?KoFX@ypE{42fb(61MSau5KRK+l2zDa;ThcxXVn9_U184FNvq!YkIhMT5~0d*q`KE>Zk4{xI~TphW$awOTW1S5_Oc7*BotK6~sS#Aaanlh#TTJrWdubsDp(u>y1o6{(CX5mMYw(78jg;oyQo$LUI@nb3vG~;4PFpX(NrSReJ*DH$FLg9udSY-8=Wx@O+#Gpm_5DE`7W$ifX+Om~v2ev#s_;gQQP)yC+DWppOoyBqomV6mm3s=^YKCBhB{2&VEFLYjZ?ix58^U+4ewK6DD4R0(Ynt1e7jk^EpHl3MR<x!Vo?iwT4ZNi1PpVXP*1-0^<P9X`o3}+K|^m{GGhkEo3#Fm^R@%cl=bi0)5@k8Gh<s2;A&qZ~+yUvyHMB-waI=r^i8f{n0rjS9=QZZ<wWhzDZT0n<%n>*4c&gRYbCKY!593M-|m-SjXjw~uz7+`FhY7@Qb83S?fJKA#KY&`Qfu^|ZZy?0hNrA_zjv8W4gnN3LZso{~42h9m1_*ZYd$eF0KgTGzM$BTkcR1rdp$P0s5BU#X^-}eC**M1u91cQ!7rh~k$t6^3UnnER)93+{X{_18?bCb3jSxf#W_v0QY#w}Hv={@^0lko@jCf;t%NxZMr8!+hR`0XF7MlBC199JAj{T9!jdG|GKN6??Jv(>Fg0t85tOoeX_B&_9D$nU|-*;lBtE7HiFO-;*k*C`^8>z-Az00!Nn>P1Q{Xf#MJrfej%4`<!U-DG<<OXGid2Ub_hJeWOCSO>wME)7Nih9z`*5PJU;{kx<j`g-@wH<}=JR}m=Pqm+bji)=&?)N(k=lGW7#I4&_#VTrv<`$e`zw4wO_Y$fQx*&*cEnF{8df@oD4`ysU7aYJ4=7^8HFK*s`D>H~XClXeDxIHaVd89q-Sq@+Xyk#mgDp6_w4XoQT=NZmOVNK>|Hn)C`nIGkD4C$=mNmT@DJUxA2#+LA7*e!xUx7?6zJ8s+>2gLIcJ6iY+M36*w#uBl0-A&Ek10jUF{9CoL+iM@!Eyl8uS_zk7!w>v-5XB!Y(p~P;dSU0k?HwEvH4js2gIHt^1RJN_QfQ6Ny_ca<QZ_*asMY)fapW1|LvOQCa&zJu@%MU$?Y5UGHJG?r5yXt%;geC&&CyxNNTyt@(2||^(*DanYH^hAN_UFmEu!D@~E2W}Q!hYVTOILGc={`XufV%%?xYZslWaP+viP%WaqFwmR8~tSaW?h@2!?x>(j_e6B)k06!KhMCnN2ztzJ5n%PF(#(dvk~aQ>|SjszkhqW^-V-nd~@(jB0~TW9A-spROZrF%Fqpc7(`4a5Xw@6F@K<9P%`%S`!Ac)w8&Xa!cgMkDEQ{&S|F+*&EAY**3&ifiid@=UW^7vS!Jo%HUs@S<*?!GlPuqB3E-MtzW6=s%mUB@%5d&DIC&`~ZyT3*Z+r(HxCUCpbw`?o4B7)p!xu`aX2ePqP|&~dMb1kj)M72EQf{Bq^-(t~XhgNpkoozLn(I?MVKMB*C8iMy)A1yLoDAre9QGXBsuF;&eD>d5=}L848k(j)vBdLfN(<|~6|^4NnuG3`P~-C4yQu_!zD#<Yz<`5O`h^_S;}^3p0CwPchhA^=CYb!Y;-vx}jV~hoTa+Vy^8f#iPTBK9IR_9P4~OdQ_>D@O9sE@`Hq7^F;XG+PAq7io?k(s+IRHeeXpJ(!H}Crc;kYQbczQhK+t9~C1Znjqhyu{7#{eIY;mg*++9FSeZh@;>&}a20IyFJmUPu)w8M7>QFBN#8SQF9Ba9w9}co>OgT3S4(;F_>F_8#O$ybym)e{Na^jMg!5041>AM+Oo2kq1fma!l1*>Ki(w!6keJ<NnB77fPL{O-j10#sK>APiB096wr5s0I7uK>zEmNM{M+BCm10BNKzu6pr|c3EcnV7^Lwc+m6N5J!NFb(q5d08#8Gc;aCn@{<?YliQ8=aGvkA{5%6`TajE>Xq`qC}*1?*1HA8}6<&}Z_Or)G(i$S-W4lJynq+;_eyPinT9(f$spUrGdiQW;`glrqsa^JJ})X|6YhVCR5Mjhn*B8E-JLQ~_>H$z>oiSY&Qi;pcc=g<j8-3Ek1{Pokupyq!T;1hNgNd{igy-=?)~bsjhUT=m}e6T(Db`nAF6FYjz`k$4@2^M<`2uqoafR*34Nr&f}e<RC>MdDGH#%eryKP6c<X>-upP$__{8F@2Xgm-{WcTSO_b3#>+|SX~ju@SdPTHer<ttDL!R!stL|c+1vgISkBCiOj}PZY_)?6uRZ9-|iysZN*=y6F*NYI|K-~VQs_?1s2{^T2o2|vk90*#Fa>-EB_zxQVy6fI;0EJ%Bl^l6``yea#3DD%}zO!e(#vI{|OpECeLB66kXq&pp$Uw0Ir%fZr?f?Hi!LW0n;+fdc^P-vr_OTZn^-u3}$i!mo1UAKWDvUYz-k*hCZG0W_he2fu7|t@o++r@iL51CyPc2mSOK_-ondZw~9!AreFO{Y8V7a06{)@OO&ipeIqcULa=(J2RV#<W9|a~)C+F+uC3Vm4XX6I#z2XngieQ_V;%t-Oz3*UPk@0*@QSC}j+Hl%uEvOrg`SDN!h}d49ACpIR#_<+ntG3I-}wMn@ma@AgevLS_4kcf_j@jeh!$lWOss%g)<TWE@&TvZj3?|!EaY2uY3>5tKCg6*oE#f0vNz}h&z4uiAfEb)eb18*GZ9$Z0x;8hnTjdwHJNFjfM^p{G5`TKxj^ce9x4Oe7VU*m@Vz^^>E1~tR{FUqn6KZLQ;#;Vc)>~YFexb6tz>3GIjPs`>T2U@u%$1@3`gf852A?%&_xHt>$+@tu_jpNEg&-GlBTQ`;N!~0-1ZtvG6|B}fgS!)qdw!^`Y6p;_L23yYBrN@d74<JWG$LJ^vN2Sk1kg1WG?F(<?M*@tTTVR?I&75JxZT$>zyx#J|@UzlH?nrinwYCTH(k)@!*gE<d;?=l8V7*i`_1YM38Bqs_f*OIr=Dt4=h`(`jp@uLSSrf$H4CSfUCkOH`*11u`A!+Vx*D!HX?r%I?#Qnj-^wd@**2RCeA!!urCEOA(S`NQ7tO~b@^hcA=#+nC_(<!HE(z+PIbtGVS4ET3m&ak9?2gAN{5*CzZT47Y?fQWAp~wH1Q2ySo8Dw=4D3KneBg;CTKGlUu!dl`8*e<Wo0h3hhj7i})ye36D?yugeQ9Y7LJ1LiclM;)LzlSN$!rX@R>!w`!&NAq@DCd_nts;FA7m}ms8xOT$>z7*60jOdcaa&@#5KI_{r*{)!pxpU1%%9m@9S6+Q{Vdt+Nv%Tfuh{04m(^yIW~1(UZY}Fe`k25&dI`r)!}S25Hqi`M4zcx6!vEoIz;~e&o1v)P59V~7Qqe-M}~E=rrs%gmnqKGxj#)Bo9X$a8t2$B{QrFa?<j_@EPO3maME!qCcSk2F-+}=pZbqh>86E&<%Hnc`>Y>R{KJ8lidy&fYN+O<8Q|Z<u4l%Z01}FwxR)cWR@*xS+cZ0JKC!jCHq74+zA!-^t0amK8@wO}8I!USY@hH_R_Aea(tX{Anp`44N=gmQk5mmMBfOAnK+}n!7F+9MVkz~-u|8E*x(h`1SYA2d(i>W(xKY_)BUHVb%Ap>Jm3V~2<!|ila=9bncH1X?_H6W5&|o|c{N9)XqezwX%Ojtrf9^_YubZ=l$9yXy+cC%Y)Jf8%We<2@hDTzf)UonKZBCBo8Z;|38C{WRTV7OLg}0@DaTpKK2Cc7aMWnzaS}Ku=%*5ai=)saZ*)*cMk-m2@Uka2#;<3-Ru{Hbte1^m`7hJBYAme?mTh8QZ%Y-rw;eV!mcJ_Cu8W2n;fd_hde}gRHK?BLTBw8S7*IGcxYDOT=v<R|=$Kzsibi}pmjq*SG(b@NhvXlC{b0-0e;Lc+E+AdRseFxeooB}4F8iMxGDs5i=eHd(jxhIZjvwNZ_r5DRcmqk^a7y>FZ9mpRM{~T@CZ4s}`e8{I_FglT!q#4+qR83Xphlx=S&vPuQkq`iB6HZwoQ;Eps4{`0UF3Oss7$mS{c&Dvj+8IpQT9K;#MFno+76_BX(ktaCi}?|pa}Pb*w@ka{+Ev$rbLlt-QD;hushx00#ww{VzK`WWGyiCfd1{vmHWSxH?#8hTTBRuEW3LX4?PE(kdX{j=@<~Sf8V->5kiZ9U3GFW0N!nJ3lkEoWz4KZ7grZ*T7?7yhlr<>6rKDI03ITt*i@*0I)2v`qt@Ofk58imA+l`1o7}^DciJ-8+5&LBz%~4J33e)(m_RoM?5}ljMNvaB?<d1fQ)Nxne56?MlkC%UKxe<MJ?2q5w6XUH6u0x32u-Q#3V&3GNHy+8@AKrtSTZ)ot?CoI@BW`6fhzC|@ZFXq@=5lsq&CJtPMNwmj2;0P<WgFj8;9-%ihzZ}H7v`B|LtRKW*`|_1quY(o0|e*>K?>CxV}59YHFR&HduMGrLFZ>z$+nP}fdg<Rf;S{nY6R_6ry<GF0?a5Q@Y?+1nzZ0kDFhBz;9U=E_H#uTCLj&x!Y))*eN-vTQZ_;n%u)jMe1|!G7_E?2D?0<TIdXZ|&@}vyJbkEyMm<N@%z!RUu}~T=B4HwRAyh=Kd$qn%&tChL+!XGa9atYUuI{bsCBeb98#pugEQZ@04kH5fbYTwWOeCpXQOO(m;={g>-no-=Lzlp=3piUf4(O@%G$T{}p>TZlvOgz61z7_u^B>pp&>#p4vG|7&+&I@<aXuAKQ*-{^a5}~d0hHWDma@Z@q=1VEvf@WY<Q`HEIa{j4`6zBP(Vm)No0(}L^QFy+v|@d$AdGL<SCheiZ7ZZb-ip0)ludIT9n@AEdlU4E-l1CWF;8<Xhh_g-4`vTpU=9xoeb%)7*D~ZK7I+nWv(Ok$J(0WaKeX&XRGBx8wdpI+^<uNfrU%wE%)48bq{=k^tn{1XExGc%3N7qwE!a_gKMp0!cFgpYg?_*hWfNUL$rV6vV4hy46@xVdl@8%>O+Tp5R+S7`reP4gx?#`q??GQbb{d!?oe4?gpHaHZ0ZND{&K`E}lsV3BufekyKjG&65FvkBH+%#A{bBNzIFJoNfHq=$f9wmMr85n_^gsqc2-3}$(nkmXr<_;)sh-V^VAw8qdH&uEU(sVXh8$*zU=ij0stuyQ&q=$h--fqZ=|x2|<6*&|`1_bew4HfJDR0xLtGFepEQ|8J@`9EXC>;34_hOd?^J$B@m!#jeC{#87Y2KLYzus^?q@Ub-uVo!_HoD|zX<pUPo+ek#ydx3NdkNy+!TZOz`vi1^2oM_;VWatLNihHt-W8f7_%!mT%7MkB3p66<VR+>BQ$=Q;A*~0O<3(?IlakiM5!e1tx}v6*L`?8AsEl1apq^|yvuTJpr(R6ticpn7Df+>?2_5yv_pfDT?ViRvOk_E7w~2CQgg_r(Xfhd4icP4&_(To6uqWLO`F5*MNF?e&7OTAFgWEGIXb+{ZjGqrBwSHLP+3dbWd}ScWlZ_+n%BLQZ`kOdW_Jg{N)WEEaB9zj7)2z45rc(-sU)DliGhLbYW}+wTXX|IIh$bbQ(i-;{AyXY1GH|O5_?|gGA4vz;;h~?J_*PitxZVdZaWU7v<RQO`ud@KuJ6(NK?~V)Mt8X@V>39x%)`c{Rd9^*?w0f}hLCRha|E?dBnnTXS*lSMG<cs2@-=+RwW*=;ot6a0OUb0DUpkucQ4b~m?#}5_AThdNLv6tqChyjT9Jbdp2xt(pq3LSK;H>HD@d{lxyg(Nm>+&%&z1VaVG?D<LMvye|jaZZ_ib<llAEm>9&U6drxsp__iF-T3<s!^R_f{YVMDm9a`k)KU;Jy02i?a4d+`y8)sbTBVMVsWcfIZVJb_>qkS`{latlz~!1Kb9y#-FYjrBh01OmO@SurfPxtXYZ*G$NhjJ>z4m?`aLHN)IxjJco#FoP7=CZ65Cfu(QQR#UUp?g?~gElyF>Od&bXf)nC#FG9HcNF-6UqQkuxBStCIsZZE!GnYW|C3s%lFRrV-V{o7e0isvHqL(JH!*#K4h%Q4IUAo9#Zid;%e_8OFKtr1tOfkq~hgmv5-HZv{zDBe#u;B3!aoY%gliJiRH^z5;nBTp#hMV=5?&wo(B!W^qIh2!A}|frscs00bHm*AUHx%jzM;Y`hTmQg}5tf~MR3e@VYs6|6?@Skc5ZKUk^XQh|UmL$J+DmK*8TEH|o}Nx8Vp>U_%}B`2gyfnnL|a(yYT3nFYikLmKda9G&RwK~YijjvbubFI7&wAL#mlb+hNMd|I$Z7{n`P-G@~dA3tWtlpvhqFs!*SOY+x{%gdJZrvL{!EixaHF>bSiu%KZ>rU$(TXAw*79++7{5wL^+2N+iwew8dzM#1?mS;A4$^VA+qRb$naxYb3z_Btzj13FBWUpY*+{ErSgJ~df+<z{lVRHd2<8ud6luT{}i%zTH%~k2jZiGRkr#-8Q52+$?7Lu61MJLe-OZ)HX+H!eac3+M0Ohm;T>E!Kg5xUC+CqBXT`5WQ^M*b}6^LMqe>-gQe$LX*kyJs>?rn8nUgG~g0GJZZSP+R+iJ`FOOSLV#{WA_bQ$*mF=6F(9IS&QdWjbf2oT`+<SVHJQvPcmjtjHaku7Eb`Sm-8x><LLOy!R?LVBTI!Jv=022R8~8o9K~DI$Qj;0M;9c@Bnavp(dXL2S&H8^t)l`IS>oIpaz`OIsbm`*%+U7Jr2#c&@$a;<jx!ypE&-qrNq_`$1BiMl0E^FJPh6=giD*=rmwD?EN<Ol_2L!@pktnM#uhu-r|M+kg!3+@y@xH~E!F3p|qVfk2J=s1SxA_VZef3MX+xNAnKEZ$z_7XXxAMX3PPq`f8r}{r~Q^&j0k4~B{1-cs7=<}E|rO@-eOzC<8#rYqN7Vt5voj3Dut&P)JE#QM*b9J8Iu<LOAXGrdUGjT0FHQs#dEGOLJP>Nmm?1(f}=~GDUCFJxqnRI7dO+*g`O6sURj=v|W^4P3#cEyhkso$ikG~rs9OBi8%WQ7t`>P+xt>-TqcM=~JmR>86FG-3+#{y~_|7{fgh>(0W8jbfF(>uOJXeM>0T+;2uKkPNfbCDyYBj0`lbnQ%gX$GL}!%T)Rp_kXm|bqe7Ibas2QU`%h-ZYo<#raPd_XYns}LI&a6mS50#Z-BR>ra$qsQR10*|IE5G#`_N&*k8~3Q2M~6e}$uSecejv)Q-T9vO4yIDeU~)W~;kCYw9aPuXh81zDan*jpAT~DlnkrcU7f2VglQY(;YO=Cb6|A`hUPsK}a>>;*yC5Y@(uq8;f!V4ibBJgRLZ5Z{x1m`qNAd*(<xH!JR-aXg{DXj9DW4$Rx+|rbx(9Wou1nCrI%!jQTTP=zXuL&^2$Yn-|^8p0w!Y1a(vLtG^_|dpSVdTzSOT$Rna7TSGxs32z9LK_S6IH+(c?*mTFS<qO+eEwdu~KlyK}mgxXKk(4QkaQMwU9x+p#BS6FRt~pA$L|O-(7}#4@NU6Ms6@Fu~BhCZJ+IBt3NiiW#HE)$O-h4y6C$(T*Oe(6MBAIY?Jcg1q-^Klr#_|O=H)xkJR+6m}<@gUWtpKp4a5l-h`;hb&@!K3&5AfQs2dFo*O*oIX>`<J~=~$nfmtFBvuU1bWU^#r;qAK&3sV6IN&rpiB9NK)<7~Fz-wkUsBe6$rCcIzaU*f#s-89}6gfj<5;Ufu49lG;QF0Dg9>=*3=CqecBa{XzNlLCzji5>n%vo6d+N=t6~7k=bl=2B(xijc2&!H#g}zy}M3}KvB#4un7wRV~H4EH^Hq=j_+=Pbeo9=*^*>9e8}GSR){h*|Lr<$*_2P`>T`RApzuSUO2z^Ffle6Gz+iuSI9(U5S`Y|OkTT>CTvfz+A|D)IPm=vu+Q#A%)I=Ak@tgo~3nvjgR=H;>=olX5LY8Ht)U=^SG#xJv3n>nlfN6qtPVHUUToGRy(2C&w-+J$b*Q<kOPu1JCJ|#=Y8+Q-b;yAw8eX4#N{y?+Isn8+y4nry^4F#YEdttHp;BTK=WanBhJBtD!Ird{hoaw$wS>#~27};ZDwNrxU!IM+aR2#)!rlL57F!_1kl$Ye#)@9O4Wh|gwbZWkuG6@f-Iru6tMjnWiN8aVQI#bzoKn^E215&AArjV=0op5Hs`fd?AKw0suaB@AOu4^x-@rqdW)2Fx()h44og%?UAnF~#2Xz?I{UUN?-;-swPFKF}UkpkJP8c^&ZHhj;;J;`upF`mSQ4;2l>?0arS1G(y$jEjX;Cst`Hxgh+pkUUa~@qd>IW0;`Wn1m3Gu%OFnqA?vBr}np*Qcqxkyx8EdF<IIn>=&Z2B4CZsu!ixzrBZ@Q)ky#i;GZ0K_BYO)YL-XrlnBF&_pWsBhX?&9@)HQSR(CzpH_qncVD-<zJZlx8gwUGVi2{+My2B;B><8~)7P;sp-A<#lTC3BwqIhsHzGz8(6`Q2XbmUB2Cd}%OD1FxqXTvNY&vvyL2qDMwKYwqi!m*=NOqto-_p5sKu(kqzkaIwr{jN^$zDyjf&QK)P;D>@<5WdimaB!9gLjGZkIEg;rGS#~Se9Q1Zp8hxW8-l%Jk?3w77^wwhJDZgYvdyc6>{L>hE$R<dhzv(fcO{hu2}V<}d}_){qoV$fM(PUS1q_8EbAaFcF>b7mdE7`gL{-D+w#&;G2&mYw=|zrld#BJP=Wj0z2Rc>PlbkX^7ZcMU4}*lonrh?{u}bR|K^TPKgdtIeEZInT`ABiJz;KNdNl`H*ImB@CDETLT*nHRGRl8cYWV(CyMvd|#^KV1yUV}5TTA0<?|80^;Y>dbdnZA5>e}INK0@jh1HmO)&2CqXr@@6-61U=~h`)gVCkB6mx5KU6HvNGze-VU|h@4?MPswvejG+gCEUrhbdaq$d*<C!YzR1fwjXP;~dPOrJI@${)%zYbrj7)slIuphqN2oVWq)~sW@+o~RRT?>`=Y8^o&XV;FTGeggjA#TyY_?zX!Qm!yU)qbE9?A(CrO;t?GqW53uy1P+oy1)4@#}4;Piv=#djJy?*#@ru4Sm<H_zbVWV?!ao^cW{|CI`r;LxA!jtEq0K_86k+zMiPvX&E65zv@#`_gI7ERTC$B4!Q=BoRph!d+W)$SgsKn2-^_29(oOaCxi!lSB%0E!-hr0?6euZ5<w!uuiTA8)A0eQ5z|;I-bS35RzbkOz&xwZw0bMV(QqwBK`3Wu*0F0sA(@;qfMkMi(kWU?Q=qeP_zZUJlLCQol*@^c8CFvO?3w!=?!m><~you|30y7HKrn<6TdVOn>D0=>aY;ITq)$J#I2e-SS8giWsPpgc+vq>Ab8G-2^;jfNLn%NTJi^j@Bt?Ez_pNy!OpCkS^cFQATE<{d1^C`R`P&Q$z(xT4h#HWXM3O@ApZa>=~9fc!e9=nE^IuqqNM0W$%$O_=~teEX26qe39N9Xjxo_#o9x^`BxKoN{`bOrv-d>w1c(eMo04Z^3>UTk<k{uH7=DP|4542XfeA2}0H1BE794Yp$8hj=7|VQ|*s{20TUD<12QKcC#Z<oLr=o(OG}JW~n<1}>dT4eURyZ<J~W!}C^#d+X=<K{ld=g<*}V0Qg~7+)2z|cT%eep~`D~u@}n6IyH`;hZwW)UxRKJ;4EuIU?ahU_x()P6e_#t?o`s1($uj?k~EDeDv{J8{1B2^%f%PBWj@#MXX4(xn!)Hcb}FNoDal^2!^AbUFp@e%L^C@RxB?aKx*ke{MZQmMjmrR^yPly@K0d<Q7l-8FdRGEbAwH_~{_1){F^9%&J+BR*|F3@b_<S>Uru12{d>QP#jt!PDms0BN@D-c#<_S#@dTz+Zxdvea^4A=GLo$m%1APs}!=J7e%E*b1;`&P_Z5!LtnFCaid;Vsdn;01E9M#7xJo-H!+a?`EOeL{46>Ay4EkHDa$$g*5#0(R6!jcpwb2{mLKepUvG$ImDFHmFFmcnlg<PNzCs*2h(RTpP(LF0}n(Oc}7^W@U#mZ6~v;bT@I)97$4Ucj{yy^Y{vQIZ!et;-)RX0T2twJbtS3y0W(La<|;qUPcfx#ZL;{`_RuQZw`I#XVrmSAlZ1Q&E%&#F)M**ZjN>lGgx7etW;4J91VNRtBhqR$qGMRV+gwMG|;ddbmSggA>fcO-zIaEEv>{EOfNjJS?@GII$@UX%36|N$mS14El*g7zroGqH43UXhWAg&9)$F_()=CYDsP!;=lZbMs0IV-#d>x5XEI7Zb;Oj>96mz@WthzI^_gQgva=mt!5|T(axm`S(UJ{%GdJ|-Oi$ga*$muiB8OjeU<D9;MlN7lG{a_M7f@fd>v)fkg1tfHW2BZZ<I*@G-e{>{GR<MG}w&%n|2J<KiQoARVYKY-dltTZonxSY3#u1-uIqgNRP&bvUO565%U?WoT&t^+3x@`K&fe)6$BBuLU!y1j!KLm18eEWC?L>I4hR|*VeOh!*d16CNxJk#p&E?C3d3>)BRAX8Cpko{hCVbo-_n+1I#hYZca{5AwA^~zY>4U{@5B7gxeM<#G#(TDKy5$8){Tyk-ZprciEaPZo{JUmqHCJSuw6L<JFgfXn5n)MvS<?bd*edU&M;&@xcBhSx7P~EX{1<E$m<Ha3;P}W-kF7jYQnl>o2WfJLm8wKs`Z#Ca`$x7Nv0yOi6&8c#$$bcnNIv%7a@5UH^{L<uMQ;GBtTm-j8N{c_Rco}5t~5($sGqe;beOjqFt0AtlVX<akT=EV&4dQ>bNf+j%NyFC8>i~OGvgIDKH7-?x{k}`(c+{oGfV=dYXu#GiG*C3aMbQceq7S+$R#w=}{RP7gX#G*$dnPwS@Me)-&rIE(e+h7))Ks!ZCtReAHATysJss?@SB^D?vB&%E<~yvP|tg#?!ZmU6lngvvQC1s;1Axd6rYqbx*PfYQ9ymN^XWX$=r#|6dUb6YrUqG?v_su<n1s$+N!|l&;{JVh!t#Ps)mD%hHGq)6WM`N^?*N=J?Z+quE;=~2{`+z-d87S6=~2ZaSDn<MFXsc!ygBYFW+||l?z1Q28r+aaL1t6roT-Vn@j7nTa-n>u-UvIES`;<cPK{fm}p-Or?fUdS2`nCD>h&P=LaXD(R%{4i%Pnu4IWY5c9b5de+t$#wqjdPtdJcowe(IKz$y{e9VLIWY30vKdjs1%CZ=EnmdOkG6mpO70e}7g`{pf^v#y#RXDm?%t^Y>?oo|At@M%7`VnVwxl76a!zKGU<uB)_!+%HuIy9joKAhaZI{<9QG_^2VFj_37cf2h$D_NMOpDyjxV_i_pE$3_xv-}{1Q5YkNaUCb5b>#gK$2K~^1aI@?+@5wxLKTxtb`w77tVa`0th!p^L@Irwq65tt@guyG(Hl($<pG%r7<t&C_i>1##IWR(=h^lI8jmz+^@g+;JSPi6Er=^*z1d)E2M|E{IAeFQ%D>LX%!aL80nT(zwQsmo_w6IXvVODSI;?L;SmH{+g25`>79%gNl8jhm1nUs23gQKfrvmD+BjfRS}Ll`<B<9rwG2cpI=rsu!?6*1H@$M9YUYH}Zk^Y9z5jk6>)bS(np`>XV70>@2{{Z6OatGc-Km`m3TnQ^W-N?oI<C_A2YsV9YVyQ$d~t-+_PQoE44AU8zii$wBj<BCMwiC52X6D9OGm_H&jT6%C4^c@O1&q})WP<RqiPOwhxF`Re2uUdR%R|wsNKg_pnFATiX!*&f-=n5zy&B;XyBR026#kNp?ts|}P%&Xq|0EPbNXw@RegHxT9G&deaIyv^9UzDC;9Vh<T7yKzpyoE6c0|0s&DR(yslAr(Nq;SCr_p>5p1*vpqy<NvJqpA#i?Eu|}7L1zss@=_Qi-Fy69Dr<;-WxKQxVTY7!gm{CynxsA_c%;>8H;$zby{&T0^XWRFcaiWgqk{h^i!X~POz?9xBj=SyXcI;#mf<l|Cr35F_A!Qfd)LbG?}3mx9qZ;^!&z06=y@iU)4=&Z^x#feV29I+0m^^h+%lX$GPO6J`Rs2Od~bEo$$@vh2(Sa4%NTn-qH{fnXYB(^VnWQzrQ$xgg3xdSii5M7aMQ{HHa6<geJ@@02jwa?+NoQJ9zS%&~<0V1s*fZtt8#0_pTj07t(bUEhO-OCf)jLn^ggfD`Uc?QR~CRk;eszhOm@HX}AVv(;;lDjgCbK>4IUficzVZOU;VL`(B)AYhRa&1Yj%;2v1f{g$?f>2kar73R&Mcec6H762x6vo(9D-hT^i-<Z1KZ4~7BAT1TMb-KpG>s#Vl`N)IdPlg<^RG&+JLQuI{utK!)mMoHJ&^whkLOlah-KW;WwG;AcaMN!TqlWPvjbvlXXeXSzZJ}VEq6Md2AE9iHUR`sCGOoE{|`3h@8P~#BFxh}hN>-L_#(<;Kd75S_zzF+xgiTyGinmz(b{5PUt925BR<>;h2p4xl4<Nxre(_Og?rPLTtB=J00K$Z!BKly60q=FyR5tGok?lvWO$dEn-H~NR<wWDU)fTRp=W2ZY*(CeAmH$$dK^*9L;8HiqTxpFd>!+S;lg)nV@1Ktxgy+b4h25P`z@F%_tgc;uw4<7RJ06~s0)nNP%JE+Ne7p~O=91gxjB8<p;^=(-7XI&`$ivmAa^tCmvQB8>tvg0MR5_U?z4+vpje)EFk;H(1^0(zPYm?E!nIn?8<%TFz9B$-d9=ed(qz+I7rc&BzpF_|YAPbJ2>dLfT@d-0!&tPr;UW>>eJO~N<CHO2IU=N_aBkpn8C^6VnRTWOeSaNo2!MgxDKJ$Sk62!-VzX5UAAT8;Sv&Z5=bHIE*buObfGd(%}js{m&9j0?m-w5H$^(MqTUnLpt;g!5^;s$)Ru1#l<%K@lagRg<By_9UL{<ka4S4v(aW4=nICR#BwEHkW?%OA5oV1H9SP;?QnN`x9!xN}QjiE^D=@*PXxZex)L_Cr2fxjj93-FlTFg4>;K08DG+1L&gZ>qrkE*CC9jppa!T$PbnAH5Ps&Up8VdcJ&3RF$fd1QYUH2-aG8%U5Vs5R74TI(Eu0$C?$(pu{%KC~(hYQ-zf|V{KtSNV#Ieu%WP2s`y|R7Fw!)?hf$WZ+NFn8+A{j<;dz$+648xQMm?|B#MK}J08@M}53UU>63Fh)Aa-J;Jm;@Q03#cvShFCq8O5sd8<TXq+JMI1Ih{<=kAO@9+UI=U_XsZrfzYc{k$h)4BBukF!luca1E@eux?RU9$bPnjvT$<~ouG&?PNm}bs6ABOUFAaS2qLJA*z2IBWQl48_nBRb)9#j|>b(Jz{KXVbtiQ<2UxVTUqTOA;E8&8$a7-|%)E{sVsPmTpsNQt1o0ewfjJ16YmmUqZALo+T?llsN;;!(7XT39z)Li0pqxNM!)1Ws!8v>GYas>qRZ2{dhrZ@9?U&8W4=J2sw^0dKI8IPHQ9MUQBvmkQbxCk4Y&4~p|mhxj4NZ#l0uDiU<%#v}<_8y^sCD;1Xo?*HjS%&BM;0U2m>I)*HqyMmlhE^J($b6-2BXo`qN5>NxTQLTLDZ=mPQ&n2$q%NqON{CwE+la6fw;B~`FyJdFlFXJUb{s+g}fCegEdk?H#CM3IX!%jsAJR3FZSxls;qY1U4doAVJ`uEn>Wub9-G7;j)maPq}vWeLPlCx*zmzXpVwTvVAe~;d#3aT!>S3>$e0eq}taV9R>IJA;PU7qeCj(3Xx2F$^<8r5x4Boq~NShx+zosQjz{1UYx*DpjwFNSNRwhaDL`E>!}53$O(i$!E+a&UN+XwtGD{J~WuyAk>volbs2;>r4pb=wGqg1(*Tl+_2@`_eHzQ%#r})gC>CbR+EBa7eFgiiPf#*seB_wUW?%AF>4lZRWU2AATh#JR4qi@wX#aj-ODXqDZ0{xT)@pW|&!WTTD;ui_9n`(Y+V%)N>|dV~&N<P-{YAT)WEFCDB~SoTr4uOucrOIC6;sDc;&lm6yE?Xh;j-OB#nAZW5g?sSBZeT=yv~(WgqYHKYt*-UXZM^3rS$lXx~RkRe-?K0{2U+hU=Fvevg+T?;wrU<g%k^x(so`gFLdq(@!9mHY?x^_k<Yh3_hOdBL*<R0{c@Yny?AKyP`~3)%VG{^3DLv;)THSLW?mZZ@=Ib!W3i$ac&H#(XJ?TV+u-4={a>9t2uHTR`_+v>0D5gX`bM#jm9_H;}nFAad&@@{=waB!9wrBKGd$uy1ix6P>pNJ(u?V7o7gE7kAjr=#x<PvRZ7ADgxMsNO*quRMdUIqIaSEWf+e8LO|l`_-ebgn6G<mf*$8DJ2Ph^sCs1_c+YRQsi2qKe*gOP;`Qc3M=AKrS}t!;IL9=Of9q<MG&+4^S>_GwFIjrs2s&g|*A%L;2_hr|Us$a4TN0lPzga6QbxHzR*fEWKQio!UZvc^06?qX(F3C%qkshd<wfiYrTo=qqCx%75IcTLl5LBBuSuqW&=ki0jZ~W=@l=3ba7lJ7lw_0r|%1J*FG8%EhRnW}!&Z<-4B}xvseMO=miRqY#5a-O3>`*p1$q^x=5e5h40HUCjjLRw~Vj`Ew2y>gj4pYncw^1)TzLXNN-f6OlTVItfUJ|euW?H-tskQaoX!5mp)B78#%_hY0?2_2;1pYL8p-e-|LJ|pqbBeU!2QDHmT}XiX40w@_qhd(dA6O2^->@_M1N^swKK~O53RWt@r^TVy?OhGzeE=(_<h{Jnm{rfZB(=`XOte43DE>d1@{Ix}jdB4O@WQ(A`pJ<Bn5sKPS6n;VeT9?7*3>%w*`W(|`^lwt93T;STaPC*2|nd%?_LcHKd<X@{Y~IhvsxB7PZgT7f@yEIW!oxMdQO~|lQlmP+IoEBiY#OA`C>W<g_;p449{$&gKf$MwrkQbFE~9#Zy)f_gh^Ac%}VXfc$C*^>PKakDdeMJvi2=)ou=Q{ws!LWy*TkdPv9#mfBYK498m;Y{`<t+-*=j3a#Mc$bx1wR5_z6UBO{5rS-jD?!yZ_e;JLTpkbAO8#778Qmglf%#V<|x1j2|Zt)@e|r>JeaFTm@A%lSh3^M>}KgB9vD3sV(KULq~MHHv4T1mGNw1DB6|kxO1Eh&=WjeTv%QT8r(yj2buJ_y<wY-il-w1^@uPKcPcpg{J@Q(Y;I$T;S>w(UmRU89Ms1oKs&AL@|n6xc9OSbIQc+JClZpy2*s+rclk(2`NFOc}mG=)T+k(Ir$-ir|b;=C*;hlW3{o8c!CF+A{$MnCBh6Q04eUS9n77;XLNkC;m{fEWh*;5W*wIie?a5{#oC2b;+^k8{F(=ZSikX<S9pE9e&S*C`BLG2;M9Xa2PX~CK~3`}O&)UVA%pv1A>vBRiCUZ4^tcoDQ*@k);)Lv05(&8S(n8Y2k?J0QCHm_tz=}2`5f|8_Z;A^?tltNVRwJ5ZnoSCeHi3fZD4942sG9yxZ6x7o9F2qY;*9pIsxFr1rY_P?Y;H~m_FM=`Q?I=NSRssUHFw+}25uRJ3}+_)^G_U7D~<;->$A}btEUyLq>e84vm@Z>A&^MO^5pW0ao+6?g;}l34Sn=VZrC<1Q%he^He+`g6WW>~wS_F`KEA9Yi2abP^c7{*p#c@}FLG!cEv2%9AT@4~JDhSznPkqMx1pB<yLt{?KAq2LRF~ziICx+!DZ_iV4ME|NJB5G#^k;zyYtRMEX<_@4CvCb)Q%*63kt#DrvhAoL8YFTpFQXF}%7+^0R)7~aD0GFM+Vdkc_3uc4UNu!@Ii(_VMec&z1nv^$(L-|-ktyf^TlpxUYrxmaU~prk|5M#%)RaC8YO66BQWs?=*p?p%f<VrT;{8pJg=&y&phhIW{B~yHXk?ep<`t%^oB8f&n9MzIS2JUp=P>F@O&_WL`x%3{MF?he9PhwEn^tQ>1Kj(7H%QfdqrA^CR-Pa$Vwb7pG}xuMb9!s<`Mb@-pTbz*H-*j{bM-NUjHXt!)Eo6QWrl`@c)@b6p4YY^<7ts)Ga$1R{3@0RpjaUy059u+cx<y3U@|r&lX%mE8uVh+!GYR|-zIyBsy=9d5TxDUFCAH>k5p@Xj?a~H-LKf@2g40C06)@kEf|i%?^$G6<PK6CCMQUS-M6DuZ(u-Url4>6>lqHvYJud{#x1F7s*&ero?38i_A8|eHfpi)v=)02&PfL8)&jZetQLXU{0fYRkFTnhY9Ny04_HeEZlzF)c;BtJ#Fp%ip(=!kevryLph1-?E*)T`kfSg?tqJjouTwoQD+pOElV;@dp1KpZySi23cgF09UY%0Rg&6gJR@QkSLlE1K7eW^O4f)j};D`rrCo7Tuv3Sw>&vjI9CA>kY=9QiqAY`FUAk0;d?8h9kf4=`NlfausR@+#hT2-eItk6&opkF`VVGBL?xG(k0`bA4PqW?DmqB`;A02*beDraBkqU7ylEgeT~lNjxDvZ@R*e4$~#bn%B~$G{^v`rxMlx^)<1>)R&9X2+Gu3?)2JqZnMuR$!EC!lxq(BPl`<V$pSR!%-MqU)ir+WQEm&AsKljI6Lqwux2o&SQcfT7n`N_((p%UL_WLyNgpqA;SS|&sqSS?qP`q2C$@&V<PTg#Ij9*gIqR~|<FyAVv!Ja@X>#e*CEKK2rRmE{J+9Bj6c07=(STfOhq?dt#V=1B1ifU@zn&#004XXzie~rw{st@!-7bvq6=)w)Ow{c!-D^h3wfKz0QCOT3S9~AFY0<pVX=$Nr7U6$yKbk;+m6E2QFhZ9B6XK0MYkiU=-RA6{07!zl`EzmyjUaE)$6#juC}MGviO8BsBoL3J@mp;q0Mw#&&gFk#4`+?1d6io5gLC7kkSmEjZl7;?8r_Xb13Z#TaRDf}3&XVzOe$mQOcXLK)GkEuYbsP7#G>hP)#3R0AM?$H3#)!$*BeCC^a}y1T*S;=10%6uRl^DmXw>j-!PE9MkaIT}c6`fsNnlsdZ_zCW)i}@$ist)xcIkeL*pb1@)R*Y=DQ8%=M0Ot6Q;i?RrcIEeFtyZMHQ&L4iiS1ke8UWQL^65MV%uler8;)CvBNODSeHJ82qp|M?V?8X;dFkraxxk)S^OKOxZT1R1Yw{|2~_-*tOk9~6c=+8a|$>Q5K2>u8i*5ouSfzhs@yDN*s<=0lPl+Y8(a1sXjN7m8LvIhOx<Y;imJn9`Xp@H7g#iW%o5uEG#Dw)DX%J^krq5m5{hf8ZZ0R1?7e&x6gpn1wUYO;@k4WFgMqAPwFfYep;bKWR<KNqV4w*nPxR`-<+D@NM|3yKeA1H+TmSB2Y$2hN1M=8*G<o<eOB=Y4@kafBe19xyLC)#s<Bso2>{5R$c)wA%-5$FEVKf2LmtfPnu+W6=r&`i+_L&7LdY0fzjB-=Q@6S1Bg|||i>(WmOP?#6s+N@(&icqHHDS_*$>}>Cp|0fE4FqchiLSEhg8;#9?X&QD0S#4c~W?YX#@5=@MK&EJsog{e00Thei?lu+ZQCh`)*jI&7eP6K!#dHOC2wQ8K5@l%VQaajCZa}%(wRo!`tO3op0aJPzc|Bc#_x}$O1zqkwD#$}yd#kD9v`x$W0u^9Lx5%<8Z$8=&&Dwf~^-?YvwzATVOBv<{7ATXkEUvNJH^(qvRnM;vo&xTyj?azBHYlGmxkAlb{La<AEPn;f#5g4>=C6+2W}Q7$;LHgH06wSezj4MEN@cAQ@u<#;ASSA;1pb7AL2qKBx((b6G10F;XEr=F!%o@|s6TNNMI8EU2X^+QltcbA83e2v)obX!Vps2zEbrp>IH(wEDY8^4113}^$)>;zM~-W<Fo@Tj7IKO(+GR1Vq-VjD+%SBoQ^W%QYA5e+*-8y%iOQ>cWtBpvSpw9)K$1WWnkZlR*MrI49yWX#7Gl3PU{Q2{#T*2Q*y$}I0G5EgP;fy>OdKOf+4VrXL8ZYz;?)J`lJ*S`@h0d!MmS`a)0~2(9O1+<VnC^u=G37~4cyo{WWb4oWS|*2Aqze8yhK}c;4|U8a@5`Qho)pwwE??SKEgOVj}~tK<WI)MTxB@sl6JHWTkrhC>yzbHLL&KmcQfMXEEwS_q_LBP<L2RT=5J?}Wn)KK0!pSSb#I7#vF&?VO1WBdl3SdQUZHw+k&Nhya^@)>GZwp;BcIN7RkJN^C3xI0X5d)|vCE?o<QJ_V#+Q#q>o4^wvMuXL)57((CKP>hHY|WbX?P>?w6Y5^vcX-_3Yo(35|5{L?XPe(9bjk}9EGG#J;(W?K!}w(DXkB7?ZFa;Kj(U=115%ElQr%HfCqxc=9&<C86&T)41Wo9yD4yNM_9I(J-Y!hrQ9qk(#BXjH*JGc)y#FMGQ6Kystu@E+kn%RaDWv~a9r#35O7>7iUnjo@{XO-m-o-o5g3ALMHv>3>E<rXGKujbNofx9&#nKhPss;0i}?3fZ!r+?Le634#woNX5XcFfqo&>Ywg1(1P<r^C8)R&(WLlE}aq0eW@mhEv#~{UmA<g^PdEeXr8D|6agU>61--^yBm2GtK7lIrrkkd`jul@D;DvwsbMsYQ)LtaL_r1nMGG!#P8%ESo3bt30Z??ULlbS+<Pa|C94s>DQ;Y=bqNghA~Vq>s`EPhho5riGZW)Z?>T$sHxHvnQ7F&)f=k%#~z7KrfFM0xkH~l2FOiphzoh(2SiC&QEHLS{=s^-0kl-(<DV&VWGJvklK9jqS?;$`1aWx53p|xj>;P%o6*VjRj79a)i7DPi$42!1~Cv&zQ?oQY(4t@^nv@*+??Can0Ru)pJN$ll^0B7VD<DhTHvW;3X0iVsPS*przWX*2QTd`)Sp#AeBMLAW4-M253$Mb0}Xx9M<m>R8EI%0{7gs>3yu-K73EEUp!FSfSFFB5ndKuI+@NyuJa0>XSPD`g4g2s*I)n4olKag4B~+AUKRVdk;K3zBj>t9S04Rsg9JlZ*u~WG&i`H2{w%X7IaxFOhXZ45!O=mUU>hz&-t&7LcJ0!F3A@@9!%|cMJkS_IlHmHvzGRezro46YT@=jNmaESj*rvPJpOYeJ1wmM+Q|F@&+I1RX}?bQ2FWzYPGOZ9~zDbh9Zj4iVqOnO=-uY}WoGd!ukSJNR0cHu4~iJD}x6NR}jpEE`5-~`}dezHVm?|{+_Gb_i}Ooi%&W7$By9I=RaIgX@;4F8q_iRT)5XwN;A*_a(gW;jlI*7}V`{aWOX+(y93KgD%dWf-*mxIVU}oU^lidNmDj=ySb=|4=-NqWF`aV}Vd2iytU><iYg2u<12qcOIa5e%K_H%vTx?Xl1wIB()eADQul_2&sw7lDQ&-vwQpm>j_z8-sDh(`V8O7%ynP1XTormY}BVJJKXhXD_+3<AQkrB8df4KtJ#%Hx|Zqdu$uL!V1#PNR;?rh6)O5B$b19(q-xj8E)7$*)C;`!NMg@ehCX0p6}c5q<^QI-h;*iERN9$X@%c0~Cqhf4ve$n$m*Rrd6h(`Kw}NMGl_Gb1HdcuU7%Dv1Gl3G4Zm1B6h_{MNh;HaV%}52nVkLZ8$F*nBo*c|UxQhKV@ZF({6A<9`6SgquPyMP@;<iIY90Wt+rX)NrtMPj784<$Hr>4t^gw)eb1;2>@|K@?*sw3}^olsz?edfjfQ2fV*2Sq@}9u1u0AOr+#P+ac(Qd^@nIz_su-faeaivh_260hR|$G=^I3nt%F5u$0Ds^a$X-k!Xmu<dE=9s~<dF6zhT;a_U%nEX_n$oN(Xc~eNC&D~_?_)oHgcme(lz$23jCFMU|^h-T)2yr%{Onk~g;SvX1Q_ngpC`&h_I{QpEjx-I3aN+yWecQ;oX$gm&BSx3P?I1M82HSsr!zkCrQNDgph&e1&yG)aA(UTfeWGJ0o6Eqz)?rsq|{Z_AJLTN^(2Gq2mb|w<TcQxZUC)~+3u7QWY6&Q~uR=|VYu$YHjv;Zn2@G1K5Z!X<iP2a#OC=@P%sLytiNT#?Rr)^yOD6ltx+xC4(Dbi!7QLCR5M`)evYl_D;?Aa{Hw=F4sZl$jqC%zj<);>SSX7vRX0N}@)^JJyY!T<Nc6_Q3o+#$Y~lCPsV6^F=@%B-Yv5u#h}2GSqa2$OIUS1G;~j?0!QzaSESj6;i->V`&&kautGDN&}}6ZzdS{q!-R^_(*{FyC#b5mDZf0v<;gOBSf+ZVzRhHRQY^>j|Rr>r|6(dPwgIP|009|G0w<fW&PyCg}n6%jKT$Ry-0U^htE9q>2C%Jr+@r0BAST=S!lF)jhRqQ_G43m5p|Mu%q!XcHN}#7DkpJs=Ob_p}IE5M;Kcr<{4BZS%B~Rhg_sgZS3SYBhhbsVG;_ddzT(i&^Kh<kG_OTA@?LI5D(hgx^Q29fJwF$L;o#6HO!y_V>k3}DJP}0Scq4Z*T#MoNUcVE;bD8p$AyBNSUQd#u>%)a4UFf_Ya5a9*`I{vG~}g5g9HOm#Ohtjo?t<J!lubEVG1c|rOVv3j@SKPHYZG!p8kkUDZoC3k7CX>asKTHD$B-|!BgArd6*o4_mf)ljRyqo{}n%`-uIE`_S;z2_SuGODw-WZE<^G+ZI9G<Ckw&BsY2bCUmjfr`{Q7|x;J!~dvJK>DpN?|QGdbSd(Y=mYV{*ZfL`9)lA9y*<M4{z30#@k)^DCPYlL&m;BVZ$1p2n-_h5Q#e!i7F;q<|V1_|;xKrSMxwBBx`KG0O9g6ax+o*X8LV)zfv*+rkYy03OryXJa;zd713q#kTwczf4oXeF-t`*;3z25c4bxB5k8E*PF`?C--E3CxY0;W9Y11AsKFS$}~)>%XM^^;2cM0M;TaoyvjSENU3;|FCFBv^hhgQB`&TSR<X7&U`ZMx;}*Rq~IIys_~V3$BL-BbZ>`dpjD|6o69Sje+)I-<Iic9B=v;yer8>6>{tJtCOK4k@UTkxO^b)Nn)UdRjJ3&N8)|jpsWKC*mQqYF=L0>y9^W!?1Af{Xhs4(&A$h=fpdsnFRr>bwy6-4Wj*$5c`h`3IFjUdvt6WYJ%Pf{TA9A`B0rY6tM==*gY!P*~1kk#Vpd&VavU>_BF~iQOT#D!3seUb7ESkOSO%+TOzdmqz8uRP@f{Q+RefkX(DbDZ(f$Nc)T{m&)@0~<D9a3crrGZtqGYWQ{BRFi-UNn^Cqj<z2ai3o6RhU8Fb`Ytzc5rw3t7XNdJ<wiL*QsAVgtb_lC8;1DBme*aa7q+O^Pr^)00Fm=0q&6mJsb;MvBYQl0ssI200dcD\"], \"tables/klee100.json\": [\"aaa3e10450befb7231a41e878755968f41fe4b85064e674543a90d9a535866b4\", \"{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;XdaOBV7PJh>Av(LHPGsxMSJ<{m0{pSZ{joAJeMN!RqDBUUP%4yJJOi^>=-N!_W&#6j-lH!X(gB(py-AgCr7h^5j0*Zli2j)}FIXFU>o1#Ne<gS5Tbl#jq(0TLL(e$cIv}1u(+h9Eq7et6Tf0w@%Az5}g)f3sFsH4=6J-K*(b0vGc8Ey86?+@mHBRP%Pz?J!7AU?`A9?LxcfE)Q^m1)P3Z7L|m*L_%Wo_$$O%?a!Y(boJinM9;A)ISy5}fJ<RDj8$QiyRSd**>KA;~SdfD#IcPY-&)ZnBX48)3kI{fk2R$1`Hw<~cnXY8P>^M6Itog~nFKxSP@+;_EG3nmMUn#=9Dkjw)d>FtC;4qRa)r7q(O;tV-^GrolSy>#t-0L_+dRc_>T_{&73*#Gq4i@r16~PCsyt3oD^=@r0naNSxdAd2Au4z&ryBw=&2bsb9vY<pCKd4>>F`P@Vh%JZb*h5E!6V^?ak0&KCpKG2MyH>R0nUU`C<t7RT7l)zLHL9w^;Gd3sAs~b%a2I*g?k?}IxK6tRCp_~U3YGx+-4d+zFc^%@E^7abWXlwyeI2>p>cw-@CL3b7F5a_Aw=l+jNDthg+aU9%g6?W=%nVR`qKq*%#_~C>1K>Ol7H6Ftaj+?%bP*WaNgWZ~c49m$HCdL=7M<GWsk6<|3{SK!3(3A3NGl!E!}Ib1TC(_h3*7Qw1?r%#!I5`@Sgqkw@_+tXQe>(jFp!`$61gjpb%&d=ngEHjThwC|L_s&r3;JTHOiV)czWf=xUNXZBHot9la_>=O8|;1M!2cPhlbI(FnO_btEq#vPbISkI!51RTzgf7`GmfNHkWnd)uZ;E1Jm0{^sT>|%VM^_$6Jk%5<eHOM=oaX`5ueD0D=OzV$0~()wQgx_SEK4=<fJC%#|aU7c+YlS_{3F;M7yJ+1{$Gep>n;TnQeMY&9FT{Z3W+n_4$$J9NBz^*fl}>f}?x%vLPnKxplt8jjF6hR{m%L<J^k2`>YgzvXCvpJ2oZs?h#Vy<f7;Ukl*i|y9nu(d^!DVG_0UILM!hg`o;`-7fTnj1AWwyTiUeU2Mr?>Z1adMJyIqld(5B6<<UdQTMRjEIIE~2XY}an80bCKYSVDavB#5y@9<0}pW0rz92ti-T*;=!OJtD`t4{v_NSvNm0~_AvoV_b|${#xS%L~*bD8hNBPPb*Ac&?R9;Ze`wtmefqsL%myW1^3&Oy3)^GAa5i-^UXCnMy?L6Rv943Nr@)Ofu(eZAYSIJe{+J@IXZM!xC65z-ppzropm2OM_<q-EU3x`j&<0i=scP0OzRjv;R&+X}rw+oMUurjoGCV`u(`SK7;mvSZdIr87pzvRQV|)*R?{*BRDBA3B_0ZYV$Q1p&S!A6GPR4yQ)2ct_JZ=L4b7CbKw;Xle28kijQg@%D<L`jd0%lQQV|S@19`Dy<Cv#I}3q%y<}>!cT--x<KL%2h??J>L$6M%tg>SEwRZ9npw9~OlU7P1677<;oJQ65pmj{JxR7F_KBBu-Fji8KV}wZDSXXe0V>7p4TSp0BO%T~!cMH8=j=ax*s7Vzy$qp_mNT!`hnl-J<4O4VtV=TFh(x0@i#0AzSrnnR(ucgIZI48x_0_osd>dyz<n4W>Fob7q9i8Cyjt2nb+nuwd-P8`S!W_|Z@YdkL|J&DB#sB*5=7BU+B;?<^yy+tyBd<usz+y)&7Z!GHDg#iYSCLsxF=Z;Myh0U<rwMld*)fFjftOg)=6h%Evf@7C(rGrHc`coQFSbT=pu+Pa|c3c9>V?VGjW|k5zAFteD(8Rnrx^=>z$U9hEvyb0Jdf3!b!U+NbVKXP<{9Aan6B$XL)X*_r!6rfz&;Qd=#By%5O@<z6)b=&$Ma=>Y&;9)y_c22l(gwZqMfvSN>M$2yWc%Y8XEAf`_z47<?46KYD|2@2@=Kqa8ELj6@wg2C`<<P=m5I(58YO&SD6%K*+yUvFAsJ=@Yrz4f7R6*Sd939=zPv4ZN3&JIj&D0H_<d@4iG$qmgXfih%lIpeb_YhrnJvWpg1GEYj<5=exzSvp^EAj(sQBVPNy+xRu!T??$=z*GUJ`s;|IYcd<#_)4DgRzo8*m&VgU8^4^~r^(9{>p)6h^rRH=yJWJ_e+l<1hTifFZ2?LOdaM={?h5&}pv*ArsJYPE^<6dP|9M@{z=ZtLlZJ%-NWgeKT#p70AaR+N@m->FBM*RJ9shsvS)%#yH?RL&_315@ArU97?EE^gGWBhEKOy^UW7?a11C}Fq9Dt=u4bsQ~WcY(9{-vb%rNk=zx9c(*t?u>8~dL1o9k6B(X(9r>FXR#Bd7gGL6%;D@Q*(ke4{Eyw=$TMT;^(p>LY%rW0*ZXq@XU|4e94j$~V_a2zpU+3N)BQR<2|4A26?_h06)=PrhbZVJ_1P2bIb<8arx^+@ps#|B3E5iT)rcS=`VLl$p;>Tw%38F!tql<*~RnELNX#KX9SplGAI;TbQhBwc-Bs!^Al@N?N|U0y^abcxOy37dtnQEGNPuT7)e6Kux-PQ~KgO-`HXG1LY!0lKKPgKqHqb^H%F%AkV-a>3;I<4tNW%GXIyLy}aQeI>$&&+29|?1>2iEinsGC%9;o5>}a-bTr$3V!G<U2^LnoCi>i-bevM)H&KfrnWdVpN{({fEj_bb50i^NR*;R}tyVmOeDCXwj7ux9mB-rlv4Y(#^<da!CK52*%<S?$&9nnKhAaP7w8@~QdU~jyFQm5Q4nNsH4fnFpfX79D6hBP9@?*v41gPy`k)=F6F)9aR<67<ar6SZ*%OO&??|xEgq5+FP;|TI%YToUF2G_#^p?a@`8b6{#xH%@tzMC{<q7*{ylC$*wqdyD-TqSBzX7t}rIrYuY^z`Q1aiYtbnws=3t4+EsY8XE%lU0W9|N1oF5*yCz*QRwK*-H;Sp0}b)qE%3EB`@y$&RW0LirY`Fb-9TUw?6s|PtTp+_N@epk{OE&;$_#mZeUz3&(^b+U!Z9>*X?|rJC`G-p1P~*jomqMJNlkWuif?Rd2xG3hJQQ)nFZqNRO@1-(0wqpDUGVG=}uCm;0JNQ>@yiPWew<UtU~h^hi`TN-0Hj8lPPGacTNt=QK<B|qlrQ4gg8Pu&MaP(ZH5u=aHD4{a5{SPr^X&VAQ#6OTYvhrA%vdSDRb$pBmBm8$q|qLGk$bq2yXb2AqqQNW+MB|#QmEg<x7mB|Llm&L}v6PQ1o_hT@`>>Q?lE%0?o}wRW+ifdn<mEIhWmYa{XZ3yZ%q;#?0KUS_GmoCMM@$1d{7|tYW`1(|%nW6$9~iO4FUOGw;j#L?#+VQA$4#0T6Oj7|bS0v{v)n)^z|iQ`QRDc0nN2i;PkpGA_!xU(G{asVu57045rqluNTsC@2B#uGfwuu|w|(ey7@RbWGd=#3YgYGxIpib!s)9uW%R;6W=p;p#Lr+FkcyCcUGY|1A03In4F5LklS~g{gTc1i`m&th3GwKl!L52FnpSouG0SA4CVgGAC(cFsI##k`=-GcvNgzKX(qClBDbEep*Whr(0c_v|G_m(mXAa?a%1l-Hl{*1Yg4Jc-b|V&a5m`AOxGWh*=WvM9IbZ7_QRZJrq{(tdpwup#}g??EBJ5%H&)6-_H>_*WLGkkq?DW<KH6g&6@G|#WU4?*vj{c+VF#f|TL?4x-X?Cqg!O*+eJx0kgI_B+T-2xR{+q~4U+7hDT+}fFAWBc>(u;TqB_OV~7QY#bV6n>Lkc1YO4NpfJR!@~xFXGo6&WVOMg{^+|n(Pk*&z=ePfHn3>iS56t%LKHP5GgU^i9bX*2UFxfJ9Y@9W20g1noi@KuSyu{#d$OdhCLS^#otm3=!ftETfK6Ks=CToMo#rH`#&9I%^~efM`B?p^w~r8LDB2GMHcV(+&+uR+FIH|Jav(1%R}7F|3CKunX-AfOu<zDGKr7a8e#+1$3|N{G2)G?NxIiJRzu?hA*7f{s5X#oMLPIv@uEwDcQip;wZM2MgYeq=v1P12YXhW|dJ-UO?2(P`cgO5LO;M9hATvm`s7S6f+N%<lt0`jfovu&KqkeVBLf~icpn8`!+&T(^J7ir#n=Diu?AS$$HnMY&_ME^n@*?kFhZy656j9%4>6CruUrhx#63koRgmbnD0INq=(o)iNK$rf&IaD3z&0($pJ7%URM#Venkb+{Gr4{ue@e+LmJG9OAsaP8ZTF{Lo2#VoluTik%#BWmLJdy3lO^R4*ldjw%6Vl*J9|3!7L5Lb<t1A7LOP8xLYWBzta<b0!!Gu4$hF5&~czVF7;76v0@ESOCM+)WD>^_o6xIu6u2m%sgFR~I_2<xTY$@6_xy^+m~yV6Teu;~{c2UYY0o##nWc%wH3+NAg7IjPOGH5k*5C{#Ua8ej8m>vCmK^qm5h2K}=C(_p|Gd`7%8Q4YAVo54YiHDsLIFSH*F3fKoMt0T`vZk&nt%dP-#hvM)g`w6|58`f4VBfPirz6R;-W=iL4Lx2P6YC*RF(8Yzm&wWe1X#ysbNP=v$Qes8-__hKC=4=7LE|*oR6x@AX(GyqnlPFLE7Lsikr}!g<SC9*|k~;9JnTxGP!Ko|P*izt;ESVxX6dSk#XXPLGn#@Sh8JuHNxF5%;{76oYAWG=s_41pFQAl*ntk4hlmNo#ErfZd>I1Y4SPr2Bq8(x`y1Sd*qRBL6mTjnBMO$eMs&E>T3FtMgw^cII)u=~%k{+2MS6|<o8;sROt^7s0K>gr<(>&CO!!UQfq4D2e13VZ&|-X1pG)w;g_Qw=WPw$pFg&ufAI=Y%r%dEfT|f1`C&El3SDViiwiuY|E(1ylo=E(BVsE)uDNg*VVjq`fQ6dxVbtrNh@@A|ghE>A9InWg*1_G`juU(a#Pm-go-0+}8lFG|jM$I<~5ji9E*gy7)BYdI-Q(lShj*L~HFr=32hE^J8PEO3CDrW8jdmKlQ@obxTUk%vPlM%0t;({QXB@SK2i8zX#_}=L?vpW+ZZH8t(>3i8j?Vb>APX(?vH%5N05FKOl^Xb{=Yo*AUx%@uF{r{ye`j2I1CsqY0FjtJH-=jU>Xi0-|jhr$iUh<n<qC5?E>p%b$6GoT#BJ9%A|cr1`@bV(17b`qbonFbzwY6HahBfZv$DqF3;-{5gEzD<?_P-qiKZsq}D~m_fxP&zq`YrK%>G6s{ib%k7Om@rynHI@}OqIFXp_sTW)L!$)hNwiXkFxm9!&@v9F-7H=G95s6`h@QGD60`=i0Ao7eZ%CN%f7c49L&O_62fA)k(P#vMh+79px66Ao0E*^N17_=jVL=gc!n|uHOLUf;{%*H#h00F-s==}r$DHgLMvBYQl0ssI200dcD\"], \"tables/merlin.json\": [\"cbcc4bdb7670bebfb08abd9335ce25a24dd1938a2d1cd7cf7171677b42b16839\", \"{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;bU<U?p**qh>Av(LHPGsxMD4xF@wkU0^eyl@}#b6`FYwT5mm1da<I&#d`ljBUTr!wk4a(Jz3r=Jb}B9hKoFL5uF*&TY*(aEWsP_aKP+{E9}Nc6ztnRx@>@i#DAUoVbAaH|>nO_sCMX#7dFwVLo@N66ZH$(3UFJa`mjaUo9nEkjcjz0`d3o$#q37)=wZVxEfs#6l#&{ir9cJdt#kXyxUusx1=6-b8(zO$tcu(?wJDkj-l)iY6=n0h#@yUuEsZXA7HGS-7d!?J!&)&jWZQs0BRUVXv8+7t+7d`E{L(@n6AMtZv<?%E44)H*-A6|Qrfv@8G&vG|&o6JAlBmO%$Lux`DfvG&SJ{*}&!YFHU?wcP6)_#y@+WS9gU{)&NcVVQmkZX3Fh^+Q+N?*4`o+<%B{$T=DAW#1=fzU)fqh{zjjKCyPkCuZ3*nBQ*`Nqq04l4t<F<8PIu6ssyz9?ro?Z?u<#r+~@1D6Ijjxw7Ky|E2gWan<wB-1nKka=6@-)N(J{+Ccn0VAY-zbeO%SVPcnP^zk__dFr3q6@DS&7%2<A8VKuui`EWuvCjo#HMnnFUs&W>)h<dz2vo*ckBw#*h+<q;Vk8DVb>x>mB+6I|H|ZG{u)pt7h_<h@evP)a8^A-V$>%za%7(<Mn1bgK4l)BVG$UheA}TDr&^1ZKzh5bhsm@FpcZ(FfXjRGCqvR4`3Ny%QG$%myHXD-k~ekVt|-F9KCp|6^JCuy#9m$aMo$UBfd7k>$zd$+SbLg5`JAe1h`K4s@#35jt3=Y8Uo#w0(`)UM0Jotr^-llSMB^5`BtWtxrO;uW9DIcE!8PVw0DkO5QnW-v>%O<WCjox#LhDSyn!iWw7wUA(JP4X5T}4cK_sJzhLu1Fm@Y-y#5_aS<a>%zglaBOZ<@dqn`es6nLAo0=U+7e(&YYoMhDWn9T(?@?jO~Pn>K3vIlcGesujtnylMXmfoE5=}#}~LZc3`Z6tDj|TAoTa?<9Dm2ju>Cy!+?4gv%N=(4t`NZhH4r!L>$ld9;*y*K)Xx3mp}%0@kZ;x^#!v6aArD{TMmZ2Y+rU`GIi1l&F&Tazhyc0cEObW(D{WBf6lJqA0p?&8^T)6xw+C%#0%<(S(pmdGphl`&6OY%kCQHmQu`|2c%sgpAbd;3nNZJQ5a6KPLj!x@%;Z(t%#g~Ss5$rAc7aZE5Is+S4Xo(7#2Zx?=D1WWPoRf7)OI78s*|5gi;&OjT53B$QGR861U@3?1dRY5IdybjS9T!c_wP(OC5~gRXA{<wbMvzf)2n$<(rM5s!5s1+rYm<OELzO^3dFE0k2<1M;=8%0l|yzDDSd!K@<~=93zT+#QPBf)<>}lOp3A5jo~++QFU?--dw$HF=Nc+|E5l%QIFI#RE{_MP>iX6}21iR<!BSI<3|3x+!)Red<{M?hlYTFx5^HNgeq{oq@D+{>3n{|i!$&jAugM=!BeXG~M60jhX^L{`UH&JuQc_0Wh&i+^SDN3oe;$X$syfPMxCTILB^iR$7sW84a9N19jzHl0@rM48^||Sa-KCCl$#!j=bsFw$0Zn_(IT3hLsoVWYoTfF7si{8a{Z$)_Z{rkMZy^s)73myxe6I2MwH|`aQ;tBq$V0iFiZw*Bm6FvG?M$jO9frDn&}Elf6?a2xZBg{8oDA~6WxnHxn%Y;CGlz|U^-={7DWFOQqZD7eo8W_6cPAdHcWbM*cwCWU(W1NauSI~~R^KQN4i!|t#4FwwzTXb#%AChTmVL;WhrU~asN?XA>U5FAyl3iI&t|9I-sNigM%dcG6S&Ui2Ic??Qt&w}Is*}hu4iU~Ma#KUnJhrN;NwdP4exvV^Zh(#ZCx1tvt33#X(#y_E7O5kG1s45g?$uaiWVFB!D&3YyzIrmxuwl!9IzoPMht;m7z-#CUSk%kLQp!fY`)!kqOA~<#wQ3be3PG-szu$xU6}#c4Si*DP6UGuwF1bM-0tC7q&v#{wEvDO0<4#|iB*w%nv^{Hr&yyFyx4fGdM2yKVVqiwE5|9|mNmU7jG)q&VH1GaA0uopDT-H%+8AUSl-hBd0;gn;q#OuQGLY?JhOQVMj{ppPLK;g70}1W~RQ3e*{bD*n^Dnkgw724b;*%}>2>+uSy|sOcRa)fT{cR}!A&YRab%qG?;&?PDMHtQ`^)p4+zPuEB(U)GaY$um^`c<`uY&%%@Y}M#qvfsiOz!qhO{YP-a0K)BMRgGFtlUo+Qt8%~&jVxgN1mBg11k)B2SI=g@S|{0B1cI`u2B&WU_xlO$j9V6n{PTjAn0X3-e9nhIV5sHa80ItYcPS<br5J`CM-wA0t+WaXJ)uM@&lFueRVMMd4_GRvJnPD+vUzD%Dpn?eBnbfDC>`Z#H=yVZf8V%srwzD?SD6(zx@|F4eq3Oh8&<lomBaLn>Y^I<y4GgCzLgX1#;*5>iogkqu`RmX271py)NJJ5PjNYXcVuI!rpCC$Kr`)GO!J6X))AAGn9WT$!cz;b?0y9_e5UGEX+8W)KKlP6xLXToN%8@Xib0IB(q5^nD=xDlhwolnv`zot9lJ9{2?e=|o(LwiYYKaiYrLVx?>QXYdg+8y!Mw6f#K(oT9GV5p+TOkJA8L=MAoec4C(%d44RKQawq8eCPNJZt!5tq|YlAQMofbB+!C+QcOlW3I$e<o5e!^KpP!sBmyL%E17u&~};IjX+pUATtk}Cu!!s5Tgo!Vct6!!4%&7&2kqb1=JZ_JN)T23yTf)^@um9QNQ53Cs1CwbiGepYL;Hf6T*nZWa1tmkjUW^3|glx27`qZFhLQ@tTCw})li!pf@c4|M>8q$<VfK|CK=$-?5no)*8G!=J&=_=Fd?HsKNM{;ys?hEF8MKkuGEm36V{#gAY)4pfm|F}{(#^YKb3Fv81TyRsX@(GdlB;$B-`_c$WoxAhBo9+-exMZ;DM)&I%{x?tc`D!@qV$$GR=0|+$^LJG6n8t?=mBkB4A(rQsM-k^=>6Wv7W$766ItTAtxG-B=T&@5*JUN25kk7a@f)IRg&d>ejjm%o#|$lJn|<)sA@cpnZ|53m#V)5|BNL?Cxv3xh*vHj0c~ct`mMHWW&2$<SZY(-VL4IW_|NW<FGn_=RTyZ*X-T_{8qbMh?6~XbX*57>*xU!yGHM>x*LZO8@G85kd%jgOdD(`w8r}by9tPplmLL`w<~a8%Ph>5X>DPAG;+Tpgi~#F`RWg6s-YY`YCgh3_Rx24&aUH>*TzF{9fD95{&{<;_mm63W=6|RDM%g<b+MbQOG)XU1W)rw)ej|kcnw@FknFGkjRe9Xj|VF)|64>op{O40FaI>f(|#ZF}-R@Chc8=NE1!`rCdB<Zwe#C?$|Wo{ZNdr2-s?yqh+J5{Y~8{CA+SIUQ~NeMf0aS3rtKdfckL>vkMzye@d)drrq%PXb^UL6o;p5>^MVVSH~`q`6PH*^BzXyJ?2g#Ib_7FzW1CZG*C>degPe6xwum}Yx*7@43Wjqkl_CY7%!}fHGMTb5c&(X$XFeKxVx_PflF(+CAlkXi0%Wz*?TYY-uLZu@NplFws;!aQnjt<1k6{4#fOKxXe6>#LatGV@9w<xjXo)RtdqLCOukfwv(LtUw7EJt|9QuWKOM`%v#IFI&^yO9-cpLg|3shyQ-{c-r>@;HiEh~O;#N9s{0BOR4XI0Aty7`9*1G*^e7$D6a(cKHo~#dt!$qt?<++4(01p;+1w~>3l@9({B_42RzB^7s{J%NbSM}&mMrxnPAi@5A!$WV<PwOy^m<QzAAFpFd*JbIvcS+$B(4!U&pWj55epi2>KjDT41X@CQoWQo`8$qkap)+P=|JobkfSm7&Dx4opwB9uJ4eNk@wborXD8$s4n&yWhaV{#xLb%$7&vP;nyA{eL19CFxGSPq7-_zVz)vH2mN$YdL6Jra!!OCnK+x+9qiBw#h#PNVO6cW6%m;b8|g(UCwn`aMB-;u!1gwReQbn_OtGY#j;b(rQ#kV3t<3e3@nf!W0Zl``4^w#Z>OEy=b_M5s;+Awi0B1`e_iink)kf3=$MR;C4P-Ra%6wRZ{};Y9C6VC+i~{A&AiFJ^9v)Bn~XF-~E`D2s8rhC&VgG-pBDK0EM>-?QeqJ^Z_Rt1Vx|-5sI3u=FmkX>aSMLsO$AZgopBa9SDA+7G@&Ze!X6mE!l=kkDzygw5quACZeL+{UpMN#ZfsRCRJll$*69UUw0#m?vzXST9I>));-rgj|q*^BDfjZf<wFuRlK{M1_6gj%{68?Xaq^3oz{O;UP{QcdoTmqB;YK7@)chTD47=L;)M;hosFXd|ToJ%U*9={b?xKUE7?N<GH@f4EKF+gZZ)1kY3AsGpspN4!75p5b<bPA$Jcz0)nrJz4mxfck1Tv#?u)A%-jw7_y$J<MnO17d}1#eldZ^u=$nwbj1Uv*BxSqY-Mr93xNGQtotPYx+;__@(6?`6vFEMZ7CEV-I6}>ZEyh^dkcditD~AFac%JgN3p!Tmd-YRQyb-C=68$ceW`BXPNc^XJ++cKQZ_-O6kMQeJ<9&Ojxig$oEw-83UP`r-rLlXnQ)SWT=hp|wFOJ`OCHvB5hZYKAjxvP9jtd^i1<y%0jl!n?CJK!=TgZcTNDej%p<ZYFxys2Lm_#ME^GW|DljPibg+~b%hYvMs4?54r<TLq}G%kGXzh&UsHsfUnQz*b6n3{?biEz3E5y?PPE2i5~t=qdl93G=^+l+Sln9N1{`t=e;S$R0;=p)7Z1f6JZ94GT2W}nZ_`jzs}35@g*(;<l*eLNrd=kOaG(;7KJweHAkx3INS;&eDd&cmF`XdihHazMR#4loLRi)wZ-53ywbx_Ej&wO2Nh8ix<=o&{I^E*Cs+GKWEKJ?h^e!*K}WgbJY-wlU>`ICabty1>0uQe^e<AU{pDio|ENi)_=@S+0dA7_O$L0l{NdZ&xIAWQ|(N@&B8NNtok1k|U*`*9oQT0hi>R?nqc}%Sk7*i3p~5>iz~J3vyTnh<VbwwI?%OxTg61Z0I;~JY&jgE>ug}mgk6_Tq-~~SA7}h{<!ch_>aCQY8}<G&lu9`M}IamcL?ZUd=^mpn>ptvPZvxaI~XqTpSmy@s4!vw-C7SMuxDJ0L&@pR#E=1tC_&g>I2o3+v}l~9Rxjx1VKfADrWFwlR@{Bj^G2!5Rk(E692<lxr^Y3^YMfQ=bd~v@U1(J7ISaR=*Ho?D=P|!?L=^um5gWWd27(X=*^?c;Qlj56Ns7Bq{*XNJfY0c0y+}*#8~KRyI?%K1S^uD`L2(Nbu7P{ml1Q%fXroe0y-;c{Yiyde7;(2X^%ad0#Ah2luG#;M!qxR*w0-iTV^dnjn!X;DGj{o6gsx9%H!prrGgagun%)z*Lo9$a73EzEo2H2QOKy>FMoa#bJ0@Tww8bS_O;C%n?jWQ03ixgeI%6s)--m$eE?!HPRwA~Yr*c%@u0>T>czc}5u<h2NXm3TY51dG+n(V(|27_@OGo!|wUK@n3W5p&KNZUqIF`HRo5DY@FBRKq+pycY*^b+Ty1dK*Q9*tB<g_6d6c7TU6=S}=8Tk^Ihy7$$zyu`-Qnv-G+4ovNz`5I0z!KF$LIjI~b(g6|ktde-4P#j6-6AS|V05GUJ*f8&XZnkZF?L0Y!@&12M>+dIPBjqTv&UW`x0pZ`~Y)JE7qOAmDj5zoah~zN3XeZ^JyisWhWy+_OyMx=k;;|dP?v^90XkKPvbde!Ab?gEWgfUO=5rpe-)wMwx6oC9VA*uHySc+%YeXzs^v&LAh|MOt3itgU}RLM>m2!{#%qL`X5Vo|3c5X+up((3ff_S_#^YR?6()YAu>^KCdbBc-VwX|BVIAtkk8S|3k+M1+kjnzJ4_MfS3xmH@jmf5(Qhqt)(;DM?hMZSX@+$z*|>FgGDM6l5Uxr?I7h(lQ)W9P=bhfxoUG2`jL!=PLHGA$Z6FM-*d$a{|!DxFWseiWFdM)v$DjZpkupW2cq0N=CgT1n9Sik4vmWq&hH@w}B0Z|9|Gv+31~C8oDqLx)4G20V5j-JSwzdm)SJ#z<D%-5y`UMT$Kv!5@vO5+Y?D5S{?wc!UoU?Gd&7NCvsLyY0}d+sBE7tMm|9bemN%|JF#*mWQW8B&Ewc=UL$x&bm{CBaw(Rlnkb_+j>Q+tQM?MTW$+V00>+*ef#uLAH*L~CvcQ#WP+gyZ!b&BL)FBDjx>@u20e5kY?f?J)$ZkJA16;S*00D|7^2P-KT=~WJvBYQl0ssI200dcD\"], \"klee100_shard_digests.json\": [\"b50690a074a884298771cc408238dad9fd881bddbbac89372616114a65e1ae50\", \"{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;4*#)LtOwpf`9=dPR$DjSGJY0zHgir&3ec7CWe`}d6UI_R)sdVsxjMHBN)`lt3E!%OjX1T@SAXcg3-57126~QPc3;iZa?TFj+#Tl>4V`!-5!38cj~Q*qtbiOD&odseSNY6#9#5C`I|&xS}YAH9U)twLXuawF3mKD29nPUfJ%gbIWS1WQe7mP7gJc3$VzWDSyE{m1kJs&4YqxpRfaqpXe9KJpC$J~gfAlj<SNH$Fp=)#0bTL|#-;)7%Tt@W5Ay>}wPU)4yn5m?i+i;ZbZnzpp?+VjZ_^|1Ek!syz=A$YrXOi%&v!Em{bTUKc%QIL2=8yX8(FK<zj1U!$}Jf(H$(D|{71PEGF&5$7#GAyR=G-|<F50>$D+GviNjJuuwcY3pyDjZdUwg%oK~Kj73v&_&qJxnT0UpQOHGyW*@WErbbLD~o-w|Lf;Sq#_m!n8q@Y3do0v01%!#g;HMb!V!Yj3bplW~A>kul__>nN)tJZ8#n(ETiPT(9O%uwyGZObxU+z>kfTUrJfN-{ctuOPUrcvRj<9>a~F6q)LPw6>q_{ikcDB;Lam;LQu!qyi>sY|rinZT7fYJ6M;!bfou(Y=!CM>X8l(GWC<#g{Y`=N?~xJdX~(VmbZsa!X|v=LK7&V+~|3kxw2`cMjs5!!9u%m1&f-@Zfg=PfwoPX^B{e#S=!Z=r{1ydpu~glq!O6Qu>w~HF$x)cb|<aqc%YC1eqfA41fhXYOSMR8Yx8F=;KY`|9z}stqw>`-o-D8I?!|JZEC<BEIkG-fDs}+fgsWB?#NujQT}IWiND=N$SuFsn;wbPqZh#XOHdf)$b)9&nY~f899+1}C$Pv{q<iP4w9y*7Se8j_OVN+(3{u)w9)-;L@{naF^*0iYYwjYz=tf|j&Oy+iaYNcT(SHd<VVjPmQ6Sd2QgE-BtQ%@#n%MSrBB|F0Zx%ZFCg?L$bX3$ey9^eGYYOn5u0*F#f7uIn!LRQu=Jbfa%qOg=R8*hyOb>5bl?@Dud;%PUfaf93xupOV&got+Dj3=RSU>SFLx_<5v72e11ak)7HXylXmn;3M~bki={&<ohWl660{0ZonykE{)z$@p%034E(51^&L)v0MEzj~IroJr92S(D}SXmnu*%oXs+KKQHo#tS-XlLLE)(6L<}uVS_GB7nb?uE`YXoh;CKX+!BLev$5kk*HilRZJs2u<25O>Eh*ghS#M;L%d)dOYc;X8;N#@@Q&9u(3&6-=D6*oDDZmlM-r+E?J2722nOj&f)h!Itre{PKjGPst5)ASrDRLjD%n~_r-BMb>k(!t8)P!Qrz2`+zam<lYq$doE^eyl2#SeDx3`;wN-3Qja74)bh6WV0r9m=lzQu^+=t~^~9bZcN_>aApQRL}E+-ev=riFYof*p(v0B3X))V*3onD>IQPs$OUH3N~a0o<?C~yVc75JT|7UwcUn*)Nkq2S?r$%3<;mhP^Y5$(G;9<4mbM45G)(S$rU9%ukTC0G6Co;JEgH`<7T10xn{+rsS)0%`Vl?l-yO8y1U7_;D1t)SByhkB(-ktU^bw`(@UT}!{baXBO^AD%vr$v~=xP9J(6h1DGG4BRLuwf7tBi6E_%z<l@D{~^BK8%(%iJxtQgs&X1DYc<pvmdaDo7Q{7K`?Q*JND2O~$VB&ksUA*POlg14?|h9xQ~0s_ttkzn5<&@h`$+J;=*TN;Z^t>Rg5e=(w962MuP!m5{dM+=yFEAR^p#T|JAW+Jyxq4AHsrQXq*irvjmApk;_vQ=a4RYFlF-t+&v1`Cn74mlkiY*%}5T%`ai_58r8-*wc`L%s%x+yp}_7xbE2d!WuQbqSZ%yBBW=*uP->Wd9Uc%oYEhoT#_V*U%>4wg}Wy3Hw8(p8uBv8@8KCFf&79gN#&(!dJ)|jRm`&GE2?F>0-NrUJ<ufU$YrAujG9ypv!pzX7&5E$I<Ap}V<GoN)Fwt?a`^DD!L+W;E-a!knT$*Da9UNTs!k=FSkTZ?HkVlv-P*0fgox;v(ukz1Sf!l{emWYu#p$J#hwczcDf5nKPOXG==T79Waq~HYQsW61TpZ8aI4gQdr%O+?_0pi8CK|tFa8817xREYoAv{&~pay7uc<=HH->RmssKS|V)6e?E1LS*HJ5@}P&l{>z*?^f?ylXWX2yU51YZ_;{I}JQtmh-a$e(J=BVw2!q%7KdSd#7lIOQ=z7kY@HTlWRw1o2rfM25QX6sN^PebpN7W7yyiT30qGmW}93KUGYeV?qUQf|IQSP24O2}7xb=h5G2jI_|8ol_N7|#qHK^)MNdeva7g$_ApnsUN_dD?I4@s+67i4mV(v@_Ut>fNagvb?6?b|-ozKhICEJu}E$Jq0Ann`4PQRo;X^A23gTJML1noS~a4?=#2S62OqgL{V|4a>#Hf>1Bl7(6tMa56V5HnUQDx~t{g(Hq%)mYpSZ@}6(Q<T8$WHbM_gZb&fu3U9VSK8-f;*bz;aCV1RYB48AmIsc|b(5Uq$|Pk_gtMQr6cUA!|J=PS;cp?;wn8X@3YV=$3)oi4Lh4`lW@v{gNsVV`n|~=E76Q(~zuMfAJcol(z%g71uzs)~(U>ctV7`RpZYYHC=%IX9Zvd=BThW7Mg)8ZT4`&2o(pICA0T+V0_eL^^@lbYui#|8U(K?*${G`mePD&cdMZA{qmgctI{x5}Tn~sxi#nY2rujVxjJ-L4N3#$xf8vp<R`)RC4?2^%700G|+|6~9FsE7v9vBYQl0ssI200dcD\"]}")       # build-time files [sha256, lzma + b85]: k100load_lib.py, the v2 kit's core + tables, digests
    K100_PIN = json.loads("{\"klee100_config_sha256\": \"fb17316df83cc1cf6cb4f5336fefb33566102cfeda16d1e9fc89ae1a4c61b9ce\", \"klee100_index_sha256\": \"873191b509904c7007ee9a791ecc18a939afbf11797e1549c0e3279bf31cefd4\", \"klee100_tokenizer_sha256\": \"06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523\", \"overlay_sha256\": {\"config.json\": \"c78d48f36421925a0fdb1c75aafe667c8d4e554d4431faa8f87f8924f09b9e27\", \"quantization_config.json\": \"e55659bc3c7cf29c668fafa45804dec879eb24ff5ab1e25eaa587ca462f2ed35\", \"model.safetensors.index.json\": \"c21d8c98821febdbab08331a4e55cb34c8e56bf4b902f5f45f1a047536213872\", \"FIXKEYS_DIFF.json\": \"70d06119bbea99b1db466fc026b0f4fb584a50b196f05fe2bc9e48595d6d8ab1\"}, \"digests_sha256\": \"b50690a074a884298771cc408238dad9fd881bddbbac89372616114a65e1ae50\", \"ref_sha256\": {\"A_2k_fresh.npz\": \"ea5c2a5a68d50e2e4abe7bd42595599775b867e817bb37b045ee0e0d3a24fe55\", \"A_2k_repeat.npz\": \"ea5c2a5a68d50e2e4abe7bd42595599775b867e817bb37b045ee0e0d3a24fe55\", \"A_40k_fresh.npz\": \"0254ad67ffd27d7e5952c894e9ba6a4b098ae91785b550d008bddd31e9ada1b9\", \"A_40k_repeat.npz\": \"0254ad67ffd27d7e5952c894e9ba6a4b098ae91785b550d008bddd31e9ada1b9\", \"A_8k_fresh.npz\": \"e3332ec70a40ab4377c6815d384fcfbade455a451404d5293c4a5e9d6c6cfe21\", \"A_8k_repeat.npz\": \"e3332ec70a40ab4377c6815d384fcfbade455a451404d5293c4a5e9d6c6cfe21\", \"A_90k_fresh.npz\": \"30c5800f6fa6e1d8cd3848f21ed1ebfa3b24e2b0207ca1553e9dfbdd0e520b35\", \"A_90k_repeat.npz\": \"30c5800f6fa6e1d8cd3848f21ed1ebfa3b24e2b0207ca1553e9dfbdd0e520b35\", \"A_dec2k_cached.npz\": \"da9098138876abc1e321d51eeaff6ffae270fca0d0d79db9dff4fb96e9e81463\", \"A_dec2k_cold.npz\": \"da9098138876abc1e321d51eeaff6ffae270fca0d0d79db9dff4fb96e9e81463\", \"A_dec40k_cached.npz\": \"062fcaebb2e31f489f97e057639d62195414e7a64660f5243abb275eaff15e76\", \"A_dec40k_cold.npz\": \"062fcaebb2e31f489f97e057639d62195414e7a64660f5243abb275eaff15e76\", \"A_dec8k_cached.npz\": \"bf9fbfb9904fe44b8dda56b6fba3b0c5f2ebb8f0e1ca463b4012f5bfac91edff\", \"A_dec8k_cold.npz\": \"bf9fbfb9904fe44b8dda56b6fba3b0c5f2ebb8f0e1ca463b4012f5bfac91edff\", \"A_sc2k_tf.npz\": \"4c0885ce1deeb48da3178843d4a29bf44944be71e4c097246b4d6eaf5d666902\", \"A_sc40k_tf.npz\": \"1e6f07a49cee93c57470c177834633a614c7ccaa8e2a6a22d42459b1cfc45994\", \"A_sc8k_tf.npz\": \"446eec9e4554172bde0fd73f96812ad2e06b779b61adeaca6eb5fb79246319da\", \"C_2k_fresh.npz\": \"ea5c2a5a68d50e2e4abe7bd42595599775b867e817bb37b045ee0e0d3a24fe55\", \"C_2k_repeat.npz\": \"ea5c2a5a68d50e2e4abe7bd42595599775b867e817bb37b045ee0e0d3a24fe55\", \"C_40k_fresh.npz\": \"0254ad67ffd27d7e5952c894e9ba6a4b098ae91785b550d008bddd31e9ada1b9\", \"C_40k_repeat.npz\": \"0254ad67ffd27d7e5952c894e9ba6a4b098ae91785b550d008bddd31e9ada1b9\", \"C_8k_fresh.npz\": \"e3332ec70a40ab4377c6815d384fcfbade455a451404d5293c4a5e9d6c6cfe21\", \"C_8k_repeat.npz\": \"e3332ec70a40ab4377c6815d384fcfbade455a451404d5293c4a5e9d6c6cfe21\", \"k100load_lab.py\": \"ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e\", \"nb0_reference.json\": \"e1e7429678abd83315313bd2f095ce1da5e5aab333013f31a801c60e7ece51a7\", \"qsa_kv_scales_262k.json\": \"5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67\"}, \"kit_v2_notebook_sha256\": \"b39db932e2227ab697b7fded01354440d80bb907c52cc99ef5a84a98a6df407b\", \"kit_core_sha256\": \"db19969dcd6be4da263271904f4bdb598cef237a307a9b31960ce2723bcb63ef\", \"lib_sha256\": \"0ba5f7c6328cdaa24276ac92eb22d4b2015b578675e9c36c711d1ad1a7f7f1b9\", \"mtp_tensor_bytes\": 1425540096, \"lab_payload_sha256\": \"ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e\", \"scales_sha256\": \"5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67\", \"nb0_builder_sha256\": \"166b5afd7f6d2a052c9496885b358fbc86f397816ca829ad3081cf21f7de65de\", \"nb1_builder_sha256\": \"15a926ad5e23055602a625009dee6a07cce3426972b8e92017e9f0e7c7bc288d\"}")       # build-time facts: v2 overlay shas, klee100 input shas, the NB0 reference SHA256SUMS
    EMB = TMP / "k100_emb"
    _emb_ok = {}
    for _n, (_want, _blob) in K100_EMB.items():
        _raw = _lzma.decompress(_b64.b85decode(_blob))
        _emb_ok[_n] = hashlib.sha256(_raw).hexdigest() == _want
        (EMB / _n).parent.mkdir(parents=True, exist_ok=True)
        (EMB / _n).write_bytes(_raw)
    K1["embedded_sha_ok"] = _emb_ok
    if not all(_emb_ok.values()):
        raise RuntimeError("K100_EMBEDDED_SHA_MISMATCH: " + json.dumps(_emb_ok))
    sys.path.insert(0, str(EMB))
    import k100load_lib as KL                # E25 klee100 pure helpers (unit-tested: merge / K6 / digests / pin ladder / labels)
    import k100kit_core as KC                # the klee100 CPU kit v2's own core (fixkeys / graft / write_overlay)
    CKPT = TMP / "fn_klee100"                # E25 klee100: rebinds RC2's merged-checkpoint path (CELL 2)
    OVERLAY = TMP / "k100_fixkeys_overlay"
    DIGESTS = json.loads((EMB / "klee100_shard_digests.json").read_text())
    K100_SHA_THREADS = int(os.environ.get("K100_SHA_THREADS") or min(8, max(2, os.cpu_count() or 4)))


    def st_header(p):
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(n))


    def k100_overlay():
        """The fixkeys overlay regenerated by the v2 kit's own code (Kit.fixkeys / graft / write_overlay) from the MOUNTED
        klee100 config.json (sha pinned) and the kit's compact tensor tables; the three files must equal the v2 kit artifact
        byte for byte. Cross-checks: the mounted index (sha pinned) weight_map == the klee100 table; the mounted FP8 PLE
        shard headers == Merlin's PLE entries the graft writes into the index."""
        rec = {}
        raw = (K100_DIR / "config.json").read_bytes()
        rec["config_sha256"] = hashlib.sha256(raw).hexdigest()
        rec["config_sha_ok"] = rec["config_sha256"] == K100_PIN["klee100_config_sha256"]
        iraw = (K100_DIR / "model.safetensors.index.json").read_bytes()
        rec["index_sha256"] = hashlib.sha256(iraw).hexdigest()
        rec["index_sha_ok"] = rec["index_sha256"] == K100_PIN["klee100_index_sha256"]
        tabs, enc = {}, {}
        for who in ("klee100", "merlin"):
            enc[who] = json.loads((EMB / f"tables/{who}.json").read_text())
            tabs[who] = KC.expand_table(enc[who])
            rec[f"table_{who}_ok"] = KC.canon_sha(tabs[who]) == enc[who]["canon_sha256"] and len(tabs[who]) == enc[who]["n_tensors"]
        rec["mounted_index_equals_table"] = json.loads(iraw)["weight_map"] == {n: v[2] for n, v in tabs["klee100"].items()}
        ple = {n: v for n, v in tabs["merlin"].items() if n.startswith(KC.PLE_PFX + "ngram_embedding.")}
        bad = []
        for f in sorted({v[2] for v in ple.values()}):
            try:
                h = st_header(PLE_DIR / f)
                got = {n: (v["dtype"], tuple(v["shape"]), f) for n, v in h.items() if n != "__metadata__"}
                if got != {n: v for n, v in ple.items() if v[2] == f}:
                    bad.append(f)
            except Exception as e:
                bad.append(f"{f}: {e!r}"[:120])
        rec["ple_headers_equal_graft"], rec["ple_header_mismatch"] = not bad, bad
        kit = KC.Kit.__new__(KC.Kit)
        kit.R, kit.log, kit.tabs, kit.enc_klee100 = {"artifacts": {}}, log, tabs, enc["klee100"]
        kit.kcfg_raw, kit.kcfg = raw, json.loads(raw)
        cfg_fix, diff = kit.fixkeys()
        idx, _tab, chk = kit.graft()
        files = kit.write_overlay(OVERLAY, cfg_fix, diff, idx)
        rec.update(graft={k: v for k, v in chk.items() if k != "files"}, diff_problems=diff["problems"], files=files,
                   equal_v2_artifact={n: files[n]["sha256"] == K100_PIN["overlay_sha256"][n] for n in KL.OVERLAY_FILES})
        rec["ok"] = bool(rec["config_sha_ok"] and rec["index_sha_ok"] and rec["table_klee100_ok"] and rec["table_merlin_ok"]
                         and rec["mounted_index_equals_table"] and rec["ple_headers_equal_graft"] and chk["ok"] and not diff["problems"]
                         and all(rec["equal_v2_artifact"].values()))
        shutil.copyfile(OVERLAY / "FIXKEYS_DIFF.json", OUT / "FIXKEYS_DIFF.json")
        return rec


    def mirror_compare(plan):
        """KAGGLE_MIRROR_MANIFEST.json (the uploader's streaming hashes) vs the trusted digest manifest, for every planned file
        it lists (recorded; the streaming sha256 below is the gate)."""
        try:
            raw = (K100_DIR / "KAGGLE_MIRROR_MANIFEST.json").read_bytes()
            mm = json.loads(raw)
        except Exception as e:
            return {"error": repr(e)[:200]}
        want, found = {n: s for n, _, _, s, _ in plan}, {}

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in want and isinstance(v, dict) and isinstance(v.get("sha256"), str):
                        found[k] = v["sha256"]
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(mm)
        return {"sha256": hashlib.sha256(raw).hexdigest(), "listed": len(found), "agree": sum(found[k] == want[k] for k in found),
                "disagree": sorted(k for k in found if found[k] != want[k])}


    K100_SCALES_SHA = "5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67"      # R3-1: qsa_kv_scales_262k.json (12 QSA layers) = the W4 Merlin p1 checkpoint's sidecar
    SCALES_CALIBRATION = "EXPERIMENTAL_REUSED_CALIBRATION (the W4-calibrated QSA sidecar reused on klee100)"   # V-7


    def k100_scales(tree=None):
        """R3-1: the FP8 E4M3 QSA static K/V scales sidecar for K3: <MERLIN_TREE>/models/flash-next/scales/qsa_kv_scales_262k.json
        when the install has it, else qsa_kv_scales_262k.json of the reference dataset; sha-checked (K100_SCALES_SHA) and copied
        into the klee100 model dir, where Merlin's FP8 plan reads it (SCALES / VLLM_QSA_KV_SCALES). Missing / changed -> ok
        False: K3 is skipped (SKIPPED_NO_SCALES); the BF16 boots do not need it."""
        cands = ([Path(tree) / "models/flash-next/scales/qsa_kv_scales_262k.json"] if tree else []) + \
                ([REF_DIR / "qsa_kv_scales_262k.json"] if REF_DIR is not None else [])
        looked, src, raw = [], None, None
        for c in cands:                                         # V-5: the first SHA-VERIFIED candidate, every one recorded
            if not c.is_file():
                looked.append({"path": str(c), "exists": False})
                continue
            b_ = c.read_bytes()
            h_ = hashlib.sha256(b_).hexdigest()
            looked.append({"path": str(c), "exists": True, "sha256": h_, "ok": h_ == K100_SCALES_SHA})
            if h_ == K100_SCALES_SHA and src is None:
                src, raw = c, b_
        if src is None:
            return {"ok": False, "why": "no sha-verified qsa_kv_scales_262k.json (Merlin tree, then the reference dataset)", "looked": looked}
        (CKPT / "qsa_kv_scales_262k.json").write_bytes(raw)
        return {"ok": True, "sha256": K100_SCALES_SHA, "layers": len(json.loads(raw)), "src": str(src), "looked": looked,
                "file": str(CKPT / "qsa_kv_scales_262k.json"), "calibration": SCALES_CALIBRATION}


    def merge_ckpt():
        """E25 klee100 (rebinds RC2's merge; same contract): the fixkeys overlay (regenerated, sha-checked), one model dir of
        symlinks (every klee100 file + the 10 FP8 PLE shards) + the overlay files, K6 (every indexed shard resolves), then the
        streaming sha256 of every used file vs the trusted HF LFS digest manifest in a background thread (sha_gate waits for
        it before the first launch). Fail closed: any miss -> ok False, no boot."""
        SHA.update(thread=None, ok=None, bad=[], sec=None)
        rec = {"dir": str(CKPT)}
        try:
            rec["overlay"] = k100_overlay()
        except Exception as e:
            rec["overlay"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1500:]}
        if not rec["overlay"].get("ok"):
            return dict(rec, ok=False, bad=["FIXKEYS_OVERLAY"],
                        why="FIXKEYS_OVERLAY: " + json.dumps({k: v for k, v in rec["overlay"].items() if k not in ("files", "tb")}, default=str)[:600])
        md = KL.build_model_dir(CKPT, K100_DIR, PLE_DIR, OVERLAY, sorted(PLE_EXPECTED))
        rec["model_dir"] = {k: (len(v) if k in ("klee100_links", "ple_links") else v) for k, v in md.items()}
        rec["k6"] = k6 = KL.k6_check(CKPT)
        plan = KL.digest_plan(DIGESTS, K100_DIR, PLE_DIR)
        rec["digest_plan"] = {"files": len(plan), "gb": round(sum(x[2] for x in plan) / 1e9, 2), "threads": K100_SHA_THREADS,
                              "names": [x[0] for x in plan]}
        ple_d = {f: [v["size"], v["sha256"]] for f, v in DIGESTS["repos"]["merlin_fp8_ple"]["shards"].items()}
        rec["ple_expected_equals_digests"] = ple_d == {k: list(v) for k, v in PLE_EXPECTED.items()}
        rec["digest_manifest_sha_ok"] = hashlib.sha256((EMB / "klee100_shard_digests.json").read_bytes()).hexdigest() == K100_PIN["digests_sha256"]
        rec["mirror_manifest"] = mirror_compare(plan)
        bad = ([] if md["ok"] else ["MODEL_DIR: " + "; ".join(md["errors"][:3])]) + ([] if k6["ok"] else [k6["why"]])
        if not rec["ple_expected_equals_digests"]:
            bad.append("PLE_EXPECTED (RC2) != the digest manifest's FP8 PLE shards")
        if not rec["digest_manifest_sha_ok"]:
            bad.append("digest manifest sha")
        rec.update(ok=not bad, bad=bad, why="; ".join(bad) or None)

        def hash_all():
            res = KL.verify_digests(plan, threads=K100_SHA_THREADS, log=log)
            with _SAVE_LOCK:                   # R1-13: never mutate R while another thread's save() dumps it
                SHA.update(ok=res["ok"], bad=res["bad"], sec=res["sec"])
                K1["digests"] = res
                R["setup"]["sha256"] = {k: res[k] for k in ("ok", "bad", "n_planned", "n_checked", "gb", "sec", "gb_per_s", "threads")}
            save()
        if rec["ok"]:
            SHA["thread"] = threading.Thread(target=hash_all, daemon=True)
            SHA["thread"].start()
        return rec
    save()


if not RERUN:
    # ==================================================================== CELL 3 — prompts (Flash-Next tokenizer + its own template at xhigh)
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
    # the Merlin checkpoint's tokenizer when the merge is OK; otherwise the Intel (stock fallback) checkpoint's own
    MERLIN_PROMPTS = bool(R["sections"]["A0"]["merge"].get("ok")) and (CKPT / "tokenizer.json").exists()
    _src = CKPT if MERLIN_PROMPTS else (K100_DIR or INTEL)   # E25 klee100: the mount's own tokenizer when the merge failed
    if _src is None or not (_src / "tokenizer.json").exists():
        raise FileNotFoundError("no tokenizer: the Merlin merge failed and the Intel fallback checkpoint is not attached")
    for _n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (_src / _n).exists():
            shutil.copyfile(_src / _n, TOKDIR / _n)
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    TOK = Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
    _HF = AutoTokenizer.from_pretrained(str(TOKDIR))
    _tpl = ((TOKDIR / "chat_template.jinja").read_text() if (TOKDIR / "chat_template.jinja").exists()
            else json.loads((TOKDIR / "tokenizer_config.json").read_text())["chat_template"])
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


    TOK_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"   # E25 klee100: klee100@600715b3 tokenizer.json (HF LFS; ids checked vs NB0 in CELL 7)
    TPL_SHA = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"   # = the 27B FP8 chat_template.jinja
    R["prompt_render"]["source"] = "merlin" if MERLIN_PROMPTS else "intel_fallback"
    R["prompt_render"]["hashes_ok"] = (R["prompt_render"]["tokenizer_json_sha256"] == TOK_SHA and
                                       R["prompt_render"]["template_sha256"] == TPL_SHA)
    if MERLIN_PROMPTS:                    # the pinned hashes belong to the Merlin checkpoint only (Intel's are recorded)
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


if not RERUN:
    # ==================================================================== CELL 4 — the step runner (E23, verbatim)
    HELP = HELP if "HELP" in globals() else {}
    WIN = float(os.environ.get("E23_WIN_SEC") or 90.0)   # 120 s FULL windows
    EST_SCALE = float(os.environ.get("E23_EST_SCALE") or 1.0) if FAKE else 1.0   # local harness only: walk the whole order
    SM_TF = MODE == "SMOKE"                                            # SMOKE teacher-forced: the 40k row only
    ROWS = {}            # oracle greedy rows by key (same weights)
    TF = {}              # teacher-forced rows by key
    NTOK = 128 if MODE == "FULL" else 64
    LIVE = {"rec": None}                 # rev 3c: windows land in the step record before every save (a hard kill keeps them)
    COMPILE_ERR = re.compile(r"ninja: build stopped|Error building extension|error: command '.*(nvcc|g\+\+|gcc|c\+\+)' failed|"
                             r"GLIBCXX_[0-9.]+' not found|unsupported GNU version|fatal error: ")
    OOM_ERR = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory")
    XID = re.compile(r"NVRM: Xid|\bXid \d+|CUDA error: an illegal memory access|illegal memory access was encountered|"
                     r"unspecified launch failure|CUDA error: misaligned address")       # rev 3e: device faults, not any CUDA error


    def keep_after_error(rec, e):
        """E25 NB1 R2: an exception after the measurements (post-processing) keeps the whole record + postprocess_error /
        postprocess_tb. A clean boot (status None / NOT_RUN / DONE / TF_OOM, correctness CLEAN, quality PASS) keeps its status
        (None / NOT_RUN -> DONE), so its dependent boots still run; any other record -> POSTPROCESS_ERROR (status_before kept).
        R2 audit N1-3: postprocess_ok False; whether the dependents run is decided by core_measurements_complete (core_check)."""
        rec["postprocess_ok"] = False
        rec["postprocess_error"] = f"{type(e).__name__}: {e}"[:800]
        rec["postprocess_tb"] = traceback.format_exc()[-2500:]
        if (rec.get("status") in (None, "NOT_RUN", "DONE", "TF_OOM") and rec.get("correctness") == "CLEAN"
                and rec.get("quality_gate") == "PASS"):
            if rec.get("status") in (None, "NOT_RUN"):
                rec["status"] = "DONE"
        else:
            rec["status_before_postprocess"] = rec.get("status")
            rec["status"] = "POSTPROCESS_ERROR"
        return rec


    def step(sec, name, est_min, fn):
        """One measured step: skipped when the budget cannot hold it, recorded on any exception, never PASS by default."""
        rec = R["sections"].setdefault(sec, {}).setdefault(name, {"status": "NOT_RUN"})
        R["order"].append(f"{sec}/{name}")
        if ABORT["why"]:
            rec.update(status="SKIPPED_ABORT", why=ABORT["why"])
            log("---", sec, name, "SKIPPED_ABORT")
            save()
            return rec
        if left() < est_min * EST_SCALE * 60 + 300:
            rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), est_min=est_min)
            log("---", sec, name, "SKIPPED_BUDGET", f"(est {est_min} min, left {left() / 60:.0f} min)")
            save()
            return rec
        LIVE["rec"] = rec
        LIVE["out"] = None                       # E25 NB1 R2: the body's record once it returned (set by watched() / run_boot())
        t = time.time()
        log(">>>", sec, name, f"(est {est_min} min, left {left() / 60:.0f} min)")
        out = None                               # E25 NB1 R2
        try:
            out = fn() or {}
            rec.update(out)
            if rec.get("status") in (None, "NOT_RUN"):
                rec["status"] = "DONE"
        except Exception as e:
            part = out if isinstance(out, dict) and out else LIVE.get("out")   # E25 NB1 R2: never discard what the body produced
            if isinstance(part, dict) and part:                               # E25 NB1 R2
                rec.update(part)                                              # E25 NB1 R2
                keep_after_error(rec, e)                                      # E25 NB1 R2: clean -> DONE + postprocess_error
            else:                                                             # E25 NB1 R2
                rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
        rec["sec"] = round(time.time() - t, 1)
        LIVE["rec"] = None
        LIVE["out"] = None                       # E25 NB1 R2
        if ABORT["why"] and rec["status"] in ("DONE", "SERVER_DIED", "LAUNCH_FAILED", "BODY_ERROR"):
            rec["status"] = "PARTIAL_HARD_STOP"
        log("<<<", sec, name, rec.get("status"), f"{rec['sec']:.0f}s")
        save()
        return rec


    def with_servers(servers, timeout, body):
        """Start, wait ready, record facts + memory, run body(servers), always kill; then the GPU-release gate on these
        servers' GPUs only; abort the lab if they stay held."""
        out = {"servers": {}}
        timeout = max(60, min(timeout, left() - 120))
        try:
            for s in servers:
                s.start()
            for s in servers:
                ok, why = s.wait_ready(timeout)
                out["servers"][s.name] = {"ready": ok, "why": why, **s.facts(), "argv_tail": s.argv[-44:], "gpus": s.gpus,
                                          "help_check": HELP.get(s.name)}
                if not ok:
                    out["status"] = "LAUNCH_FAILED"
                    out["servers"][s.name]["log_tail"] = s.log()[-4000:]
                    _xl = [l for l in (s.log() or "").splitlines() if XID.search(l)][:8]   # rev 3e/3f: start-up / capture faults
                    out["servers"][s.name].update(xid=bool(_xl), xid_lines=_xl)
                    return out
            mem0 = L.gpu_mem()
            for s in servers:
                out["servers"][s.name].update(gpu_mem_after_ready=mem0, host_rss_mb=s.rss_mb())
            try:                             # rev 3: an exception in the body keeps everything measured so far
                out.update(body(servers) or {})
            except Exception as e:
                out.update(status="BODY_ERROR", body_error=f"{type(e).__name__}: {e}"[:800], body_tb=traceback.format_exc()[-2000:])
            for s in servers:
                out["servers"][s.name].update(s.facts(), gpu_mem_end=L.gpu_mem(), host_rss_mb_end=s.rss_mb(),
                                              alive_after_body=s.alive(), rc=None if s.alive() else s.proc.returncode,
                                              xid_lines=[l for l in (s.log() or "").splitlines() if XID.search(l)][:8])
                out["servers"][s.name]["xid"] = bool(out["servers"][s.name]["xid_lines"])
                if not s.alive():
                    out["server_status"] = "SERVER_DIED"         # E25 klee100 R2-3: a specific body status is kept
                    if out.get("status") in (None, "DONE", "NOT_RUN"):
                        out["status"] = "SERVER_DIED"
                    out["servers"][s.name]["log_tail"] = s.log()[-4000:]
            return out
        finally:
            for s in servers:
                s.kill()
            gpus = ",".join(sorted({g for s in servers for g in s.gpus.split(",")}))
            out["engine_procs_before_release"] = clear_engine_procs()      # E25 klee100 R1-6
            rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)
            if rel is False:                 # rev 3c: one more kill + 300 s before the lab aborts
                for s in servers:
                    s.kill()
                out["engine_procs_before_fail"] = clear_engine_procs()     # E25 klee100 R1-6: before FAIL_GPU_RELEASE
                rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus=gpus)
            if rel is None and NVSMI_OK:     # nvidia-smi worked at start: a failure now is not "released"
                rel = False
            out["gpu_released"] = rel
            if rel is False:
                ABORT["why"] = f"GPUs {gpus} still held after {[s.name for s in servers]}"
                out["status"] = "FAIL_GPU_RELEASE"


    def oracle(port, key=None, refs=()):
        """Greedy rows (2k, 2k, 8k) + one 40k row; saved under `key` (a target-only oracle) or compared with ROWS[ref]."""
        rows = L.greedy(port, ORACLE_P + LONG_P, n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
        clean = all(not r.get("error") and r.get("tokens") and not L.degenerate(r.get("text")) for r in rows)
        if key and clean:            # rev 3d: an errored / empty / garbage reference is never stored (consumers: NO_ORACLE)
            ROWS[key] = rows
        elif key:
            R["errors"].append(f"oracle {key}: reference not stored (error / empty / garbage row)")
        res = {"garbage": [L.degenerate(r["text"]) for r in rows], "errors": [r["error"] for r in rows],
               "text_head": [r["text"][:120] for r in rows]}
        if not FAKE:                 # rev 3h: chat-path render vs the production xhigh render (recorded per arm)
            res["template_probe"] = L.template_probe(port, TPL_MSG, TPL_IDS)
        for ref in refs:
            res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
        return res


    def tforced(port, key=None, ref=None):
        """Teacher-forced 40k / 90k prompt logprobs (last 256 positions); saved under `key` or compared with TF[ref].
        rev 3c: on stock vLLM only on its own short launch (prompt logprobs = chunk x 248k-vocab logits: OOM at 0.90)."""
        rows = {b: L.teacher_forced(port, ids) for b, ids in TF_P.items() if not (SM_TF and b != "40k")}
        if key and L.tf_complete(rows):                                 # rev 3e: complete references only
            TF[key] = rows
        elif key:
            R["errors"].append(f"teacher-forced {key}: reference not stored (incomplete band)")
        if ref:
            return {b: L.compare_tf((TF.get(ref) or {}).get(b), rows[b]) for b in rows}
        return {b: {"n": (rows[b] or {}).get("n"), "error": (rows[b] or {}).get("error")} for b in rows}


    def levels(port, band, lv, server=None, **kw):
        """rev 3c: the guard includes admission (prefill) time; a band stops once KV-bound (the fit is recorded); a dead
        server stops the band; every window is written into the live step record before save()."""
        out, fit = {}, None
        for n in lv:
            k = f"{band}@{n}"
            need = WIN + 90 + min(3600, n * BANDS[band] / 1500)
            if ABORT["why"] or left() < need + 120:
                out[k] = {"status": "SKIPPED_BUDGET", "need_sec": round(need)}
            elif not due(need + 120):                         # E25 klee100 R1-1: the boot body cap, not only the run budget
                out[k] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
            elif fit is not None:
                out[k] = {"status": "SKIPPED_KV_BOUND", "streams_fit": fit}
            elif server is not None and not server.alive():
                out[k] = {"status": "SKIPPED_SERVER_DEAD"}
            else:
                try:                         # rev 3: one failing level never loses the others
                    out[k] = L.steady(port, PROMPTS[band], n, secs=WIN, max_new=MAX_NEW[band], **kw)
                except Exception as e:
                    out[k] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]}
                rm = out[k].get("running_max")
                if out[k].get("status") == "CAPACITY_LIMITED" or (rm is not None and 0 < rm < n and
                                                                  str(out[k].get("status")).startswith("INCONCLUSIVE")):
                    fit = rm
            if LIVE["rec"] is not None:
                LIVE["rec"].setdefault("live_windows", {})[k] = out[k]
            save()
        return out
    save()


if not RERUN:
    # ==================================================================== CELL 5 — the arms
    LADDER = [1, 2, 4, 5]                    # 40k: 8 streams would need 8 x (10 + 7) blocks of 4096 > 149 FP8 blocks (preempts)
    TEMP = 0.6                               # production sampling (Merlin's --override-generation-config is T 1.0)
    E23_TF_SCALE = {"tp4_vs_tp1_layout": 0.0772, "nvfp4_vs_bf16_kv": 0.3787}   # E23 SMOKE rev3g mean |d logprob|, for scale
    PREFILL = {"tok_s": 700.0, "measured": False}
    OOM_STEP = 268435456                     # Merlin's note: lower KV_BYTES in 0.25 GiB steps if the boot OOMs
    # Pre-run capacity (Merlin's rule: blocks per request = ceil(L / 4096) + 7; FP8 149 usable blocks, BF16 ~78)
    R["capacity_plan"] = {
        "fp8": {"pool_blocks": 149, "40k@5": 5 * (math.ceil(130560 / 4096) + 7), "8k@8": 8 * (math.ceil(32000 / 4096) + 7),
                "90k@4": 4 * (math.ceil(131072 / 4096) + 7), "116k@3": 3 * (math.ceil(131072 / 4096) + 7),
                "116k@4": 4 * (math.ceil(131072 / 4096) + 7),
                "note": "blocks if every stream grew to max length; the ladder windows are 90 s, so real use is far lower; "
                        "90k@4 and 116k@4 are capacity-limit probes"},
        "bf16": {"pool_blocks": 78, "40k@3": 3 * (math.ceil(130560 / 4096) + 7)},
        "long_rows": "1 natural xhigh ARC row (max 120,000 new) + 1 forced 60,000-token row, temperature 0.6, run CONCURRENTLY "
                     "(concurrency 2): not a single-stream latency number",
    }
    R["correctness_meaning"] = ("Both Merlin boots run MTP K3: CLEAN = BF16_MTP3 vs FP8_MTP3 parity plus probes / sampled / greedy "
                                "evidence; it is NOT target-only (MTP off) correctness. A production GO needs an MTP-off oracle and "
                                "a forced-rejection test.")
    CLEAN = ("EXACT", "NEAR_TIE")


    def _windows(o):
        """Every steady window of a step record (levels dicts + steady_multi replicas)."""
        out = []
        for v in (o.get("steady") or {}).values():
            if isinstance(v, dict):
                out.append(v)
        for k in ("40k", "90k"):
            for r in ((o.get(k) or {}).get("replicas") or []) if isinstance(o.get(k), dict) else []:
                if isinstance(r, dict):
                    out.append(r)
        return out


    def oracle_gate(o, need=(), extra=()):
        """None when clean; otherwise the NO_GO / INCONCLUSIVE status. `need` = refs that must compare clean; `extra` =
        further comparison dicts (replica greedy, 90k greedy) that must compare clean too. rev 3e (audits B-O1, C-F2):
        garbage under load in any window and failed arithmetic probes (real servers; the fake has no chat) are NO_GO too."""
        orc = o.get("oracle") or {}
        if any(orc.get("errors") or []):
            return "NO_GO_ORACLE_REQUEST_ERROR"
        if any(orc.get("garbage") or []):
            return "NO_GO_ORACLE_GARBAGE"
        if any(w.get("status") == "FAIL_GARBAGE_UNDER_LOAD" for w in _windows(o)):
            return "NO_GO_GARBAGE_UNDER_LOAD"
        if not FAKE and isinstance(o.get("probes"), dict) and o["probes"].get("ok") is False:
            return "NO_GO_SHORT_PROBE"
        sts = [(orc.get("vs_" + k) or {}).get("status") for k in need] + [(x or {}).get("status") for x in extra]
        if any(s == "FAIL_HIGH_MARGIN" for s in sts):
            return "NO_GO_ORACLE"
        if any(s not in CLEAN for s in sts):
            return "INCONCLUSIVE_NO_ORACLE" if any(s in (None, "INCONCLUSIVE_NO_ORACLE") for s in sts) else "INCONCLUSIVE_ORACLE"
        return None


    def gated(o, need=(), extra=()):
        g = oracle_gate(o, need, extra)
        o["correctness"] = g or "CLEAN"
        if g and o.get("status") in (None, "DONE"):
            o["status"] = g
        return o




    def _finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


    _compare_e23, _compare_tf_e23 = L.compare, L.compare_tf


    def _compare_rc2(ref, cand):
        """Final audit H2: a non-finite margin on either side makes the comparison INCONCLUSIVE before any token logic
        (E23's compare read a NaN reference margin as NEAR_TIE)."""
        for rows in (ref or [], cand or []):
            for r in rows:
                if any(m is not None and not _finite(m) for m in (r.get("margins") or [])):
                    return {"status": "INCONCLUSIVE_NONFINITE"}
        return _compare_e23(ref, cand)


    def _compare_tf_rc2(ref, cand):
        """Final audit H2: a non-finite teacher-forced logprob is never MEASURED."""
        for side in (ref, cand):
            for p in ((side or {}).get("positions") or []):
                if p and p[0] is not None and not _finite(p[0]):
                    return {"status": "INCONCLUSIVE_NONFINITE"}
        return _compare_tf_e23(ref, cand)


    L.compare, L.compare_tf = _compare_rc2, _compare_tf_rc2


    def short_gate_failed(o):
        """Final audit H1: a failed probe (real servers; the fake has no chat) or a failed / garbage sampled request stops
        the boot's body at once; gated() then gives the NO_GO verdict."""
        p, s = o.get("probes") or {}, o.get("sampled") or {}
        return (p.get("ok") is not True and not FAKE) or s.get("ok") is not True or bool(s.get("garbage"))


    def oracle_ok(orc):
        return bool(orc) and orc.get("complete") is True and not any(orc.get("errors") or []) and not any(orc.get("garbage") or [])


    def _windows(o):
        """Every load window of a record: the ladder AND the long census (audit: census_long was not gated)."""
        out = []
        for key in ("steady", "census_long"):
            for v in (o.get(key) or {}).values():
                if isinstance(v, dict):
                    out.append(v)
        return out


    def evidence_gate(o, expected_rows, require_sampled=True):
        """Fail-closed evidence (three pre-GPU audits): probes, the sampled path and a COMPLETE oracle are mandatory;
        a missing piece is never CLEAN. expected_rows None = no oracle in this arm (diagnostic arms)."""
        pr = o.get("probes")
        if not isinstance(pr, dict):
            return "INCONCLUSIVE_PROBES_MISSING"
        if pr.get("ok") is not True and not FAKE:          # the fake server has no chat endpoint (E23 rule)
            return "NO_GO_SHORT_PROBE"
        if require_sampled:
            sm = o.get("sampled")
            if not isinstance(sm, dict):
                return "INCONCLUSIVE_NO_SAMPLED_TEST"
            if sm.get("ok") is not True or sm.get("garbage"):
                return "NO_GO_SAMPLED_PATH"
        if expected_rows is None:
            return None
        orc = o.get("oracle")
        if not isinstance(orc, dict):
            return "INCONCLUSIVE_NO_ORACLE"
        if any(orc.get("errors") or []):
            return "NO_GO_ORACLE_REQUEST_ERROR"
        if any(orc.get("garbage") or []):
            return "NO_GO_ORACLE_GARBAGE"
        if orc.get("row_count") != expected_rows or orc.get("complete") is not True:
            return "INCONCLUSIVE_ORACLE_INCOMPLETE"
        return None


    _gated_e23 = gated


    def gated(o, need=(), extra=(), expected_rows=4, require_sampled=True):
        g = evidence_gate(o, expected_rows, require_sampled)
        if g:
            o["correctness"] = g
            if o.get("status") in (None, "DONE"):
                o["status"] = g
            return o
        return _gated_e23(o, need, extra)


    def oracle(port, key=None, refs=(), prompts=None):
        """E23's oracle plus completeness: every expected row present, no error, exactly NTOK tokens (greedy rows run with
        ignore_eos, so a short row is a defect); a reference is stored only when complete and garbage-free."""
        prompts = list(prompts) if prompts is not None else ORACLE_P + LONG_P
        rows = L.greedy(port, prompts, n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
        garbage = [L.degenerate(r.get("text")) for r in rows]
        complete = len(rows) == len(prompts) and all(not r.get("error") and len(r.get("tokens") or []) == NTOK for r in rows)
        if key and complete and not any(garbage):
            ROWS[key] = rows
        elif key:
            R["errors"].append(f"oracle {key}: reference not stored (incomplete / error / garbage)")
        res = {"garbage": garbage, "errors": [r.get("error") for r in rows], "text_head": [(r.get("text") or "")[:120] for r in rows],
               "row_count": len(rows), "expected_rows": len(prompts), "tokens": [len(r.get("tokens") or []) for r in rows],
               "complete": complete}
        if not FAKE:
            res["template_probe"] = L.template_probe(port, TPL_MSG, TPL_IDS)
        for ref in refs:
            res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
        return res


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


    def window_garbage(ids, win=2000, min_len=200):
        """Final audit H3: the first garbage 2,000-token window of a row (None if every window is clean). A whole-text check
        would read 100k tokens of healthy reasoning as LOW_VARIETY (unique words < 8 %), so windows are checked one by one."""
        for i in range(0, len(ids), win):
            seg = ids[i:i + win]
            if len(seg) < min_len:
                break
            v = L.degenerate(TOK.decode(seg))
            if v:
                return {"start_token": i, "verdict": v}
        return None


    def long_quality(name, r):
        if not isinstance(r, dict):
            return "MISSING_LONG"
        if r.get("error"):
            return "FAIL_LONG_REQUEST"
        if r.get("window_garbage"):
            return "FAIL_LONG_GARBAGE"
        if name == "forced60k" and not r.get("reached_60k"):
            return "INCOMPLETE_LONG"
        if name == "natural" and r.get("cut_by_cap"):
            return "INCOMPLETE_LONG"
        return "PASS_LONG"


    def long_gate(o):
        """None when the record has no long rows or both rows PASS_LONG; otherwise the verdict for the top-level status.
        Priority: request error > natural garbage > forced garbage (synthetic past its first EOS) > incomplete."""
        lo = o.get("long")
        if not isinstance(lo, dict):
            return None
        q = {n: long_quality(n, lo.get(n)) for n in ("natural", "forced60k")}
        lo["long_quality"] = q
        if "MISSING_LONG" in q.values():
            return "INCONCLUSIVE_LONG_MISSING"
        if "FAIL_LONG_REQUEST" in q.values():
            return "NO_GO_LONG_REQUEST_ERROR"
        if q["natural"] == "FAIL_LONG_GARBAGE":
            return "NO_GO_LONG_GARBAGE"
        if q["forced60k"] == "FAIL_LONG_GARBAGE":
            return "NO_GO_FORCED60K_GARBAGE"
        if "INCOMPLETE_LONG" in q.values():
            return "INCONCLUSIVE_LONG_INCOMPLETE"
        return None


    def long_rows(port, secs_cap=3000):
        """One natural xhigh ARC request (its own stop, up to 120,000 new tokens) and one forced 60,000-token row
        (ignore_eos: 'synthetic' past its first EOS), CONCURRENTLY (concurrency 2). Full text in traces/ + requests.jsonl.
        A row cut by the time cap is incomplete, not a correctness failure."""
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
        out = {"concurrency": 2, "cap_sec": round(cap)}
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
                         "finish": row["finish_reason"], "garbage": sl, "window_garbage": window_garbage(ids),
                         "think_closed": "</think>" in row["text"], "cut_by_cap": row["finish_reason"] is None,
                         "tail": row["text"][-400:]}
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
        """Teacher-forced 40k LAST in a boot (chunk x 248k-vocab prompt logits can OOM the last PP rank); free memory first.
        How a server death here is classified is decided from its log in merlin_boot (TF_OOM vs TF_SERVER_DIED)."""
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


    def merlin_boot(name, kv, port, timeout, body, need=(), expected_rows=4, require_sampled=True):
        """Boot gate (ok must be True), one Merlin boot, cleanup of the /dev/shm entries this boot created; a boot OOM
        retries once 0.25 GiB lower. A server death during TF is TF_OOM only with an OOM line and no device fault."""
        kvb = NB1_KV.get(name) or (KV_FP8 if kv == "fp8" else KV_BF16)   # E25 NB1: the B3 pin step
        tries = []
        while True:
            gate = boot_gate()
            if gate["ok"] is not True:
                return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries}
            shm0 = shm_names()
            o = with_servers([merlin_fn_server(name, kv, port, kvb)], timeout, body)
            o.update(kv_bytes=kvb, boot_gate=gate, shm_cleanup=shm_cleanup(shm0))
            srv = (o.get("servers") or {}).get(name) or {}
            oom = o.get("status") == "LAUNCH_FAILED" and bool(OOM_ERR.search(srv.get("log_tail") or ""))
            tries.append({"kv_bytes": kvb, "status": o.get("status"), "oom": oom})
            if not oom or len(tries) > 1 or left() < timeout + 900:
                break
            kvb -= OOM_STEP
            log("boot OOM:", name, "retry with KV_BYTES", kvb)
        o["tries"] = tries
        if o.get("tf_died") and o.get("status") == "SERVER_DIED":
            srv = (o.get("servers") or {}).get(name) or {}
            if OOM_ERR.search(srv.get("log_tail") or "") and not srv.get("xid"):
                o["status"] = "TF_OOM"
                o["status_note"] = "the server died of OOM in the teacher-forced call (run last); everything before it is valid"
            else:
                o["status"] = "TF_SERVER_DIED"
                o["status_note"] = "the server died in the teacher-forced call without an OOM line (or with a device fault)"
        g = gated(o, need=need, expected_rows=expected_rows, require_sampled=require_sampled)
        if g.get("status") == "TF_OOM" and g.get("correctness") != "CLEAN":
            g["status"] = g["correctness"]
        lg = long_gate(g)                                  # final audit H3: the long rows have their own verdict
        g["long_verdict"] = lg or ("PASS_LONG" if isinstance(g.get("long"), dict) else None)
        if lg and g.get("status") in ("DONE", "TF_OOM"):
            g["status"] = lg
        return g


    def arm_bf16():
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s))
                return o
            o["prefill"] = prefill_probe(p)
            o["oracle"] = oracle(p, key="bf16")
            o["census_end"] = kv_census(s)
            if not oracle_ok(o["oracle"]):
                o["early_stop"] = "ORACLE_INCOMPLETE"          # no reference: the TF A/A would measure nothing usable
                return o
            o["tf"] = tf_last(s, key="bf16", aa=True)
            o["tf_died"] = not o["tf"]["alive_after"]
            return o
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            return {"status": "SKIPPED_SHA_GATE", "why": why}
        return merlin_boot("merlin_bf16", "bf16", 8000, 45 * 60, body)


    def fp8_gate(o):
        vs = (o["oracle"].get("vs_bf16") or {}).get("status")
        if (o.get("sampled") or {}).get("ok") is not True:
            return "FAIL_SAMPLED"
        if (o.get("probes") or {}).get("ok") is not True and not FAKE:
            return "FAIL_PROBES"
        if any(o["oracle"].get("errors") or []) or any(o["oracle"].get("garbage") or []) or vs == "FAIL_HIGH_MARGIN":
            return "FAIL"
        if o["oracle"].get("complete") is True and vs in CLEAN:
            return "PASS"
        return "INCONCLUSIVE"


    def arm_fp8():
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                         fp8_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
                return o
            if not PREFILL["measured"]:
                o["prefill"] = prefill_probe(p)
            o["oracle"] = oracle(p, refs=("bf16",))
            o["fp8_gate"] = fp8_gate(o)
            if o["fp8_gate"] == "PASS":                      # load tests only on a PASS (audits: INCONCLUSIVE never runs them)
                kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"]}
                o["steady"] = levels(p, "40k", LADDER, **kw)
                o["steady"].update(levels(p, "8k", [8], **kw))
                o["census_long"] = {**levels(p, "90k", [3, 4], **kw), **levels(p, "116k", [3, 4], **kw)}
                o["long"] = long_rows(p)
            else:
                o["load_tests_skipped"] = f"FP8 comparison gate: {o['fp8_gate']}"
                o["status"] = "NO_GO_FP8_KV" if o["fp8_gate"].startswith("FAIL") else "INCONCLUSIVE_FP8_KV"
            o["census_end"] = kv_census(s)
            if o["fp8_gate"] == "PASS" or os.environ.get("RC2_TF_ON_FP8_FAIL") == "1":   # default: fail fast
                o["tf"] = tf_last(s, ref="bf16")
                o["tf_died"] = not o["tf"]["alive_after"]
            return o
        return merlin_boot("merlin_fp8", "fp8", 8001, 40 * 60, body, need=("bf16",))


    def arm_bf16_load():
        """Only when FP8 fails: the ladder, the census and the long rows on BF16 KV (seqs 3). DIAGNOSTIC_ONLY: this arm
        has no oracle of its own (the A1 boot is the BF16 reference)."""
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
            else:
                kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"]}
                o["steady"] = levels(p, "40k", [1, 2, 3], **kw)
                o["census_long"] = levels(p, "116k", [1, 2], **kw)
                o["long"] = long_rows(p)
            o["census_end"] = kv_census(s)
            o["certification"] = "DIAGNOSTIC_ONLY"
            return o
        return merlin_boot("merlin_bf16_load", "bf16", 8002, 30 * 60, body, expected_rows=None)


    def arm_stock():
        """Fallback when Merlin cannot serve: stock vLLM 0.32 + Intel W4A16 (RC1's correct engine), -lcuda shim, FP8 KV."""
        if INTEL is None or WHEELHOUSE is None:
            return {"status": "FALLBACK_UNAVAILABLE", "why": f"intel={INTEL} wheelhouse={WHEELHOUSE}"}

        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
                return o
            o["oracle"] = oracle(p, key="stock", prompts=ORACLE_P)          # 2k / 2k / 8k (32k max length: no 40k row)
            if oracle_ok(o["oracle"]):
                o["steady"] = levels(p, "8k", [1, 2, 3], server=s, temperature=TEMP)
            return o
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate}
        shm0 = shm_names()
        o = gated(with_servers([stock_fn_server("stock_fn", 8003)], 40 * 60, body), expected_rows=len(ORACLE_P))
        o.update(boot_gate=gate, shm_cleanup=shm_cleanup(shm0))
        return o


    save()


if not RERUN:
    # ==================================================================== CELL 6 — reused tools, verbatim (NB0 CELL 6, NB1 CELL 6 / 7)
    import statistics
    import textwrap
    import numpy as np
    import urllib.request as UR
    import urllib.error as UE
    from collections import Counter
    R.pop("capacity_plan", None)          # RC2-only records written by RC2's CELL 5 definitions
    R.pop("correctness_meaning", None)
    RC2_REF = json.loads(__import__("lzma").decompress(__import__("base64").b85decode("{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;8%AOzg+-5h=|8RDjkYRmvS64H_b8*T>49vqXC^JC>U$8v=ZE<TD~6Ac3OlLpBdPUqPwVl?0s`NtkJZAa%1~=?7ZuHyB^PI9^xtc6Zj?092;0QN9ajL2%mGsdH+{i*#LXg82R2-;w=li@t|4IW7{jRF?NGq_6HkEHwO~bC=);F3P+mkuG;5dY*O_bV6x3Ydr0;%RdSZ_Ak_o63eyzcth@gP2X`ZF`2{4|x2@FBOVPB5{@s1q5Bw$4#1gzSkD>RAC1lvE2j?6*>r~b%ST<LdjD{$^l5O1HZlmZYEOZ1rx`JmYQXj@fo=bj5+kR&yu;BcTuXGXevRN#gp);Oths!0qB|ecuTB$+GGPbd!s7ibVUe9gRBWcgvGe`U5ii9=lOXhk-eQ&DEw)uPkTT)P%q{q%LH<M00l<z?4-KS{z;By^Mj^PY9Ui|UWKy_J2J6NMzW<`XgxyG_j^@8Em9y&BpEgBoQ%<$Kl2U2Z#_j=B~*Xq<hf7BgPz9u5-g4=#;=dU5eX;KpQg$3N>?GV`*St)a@rMwokQLUMI90J>4Z7IHHdG=(j3MwOj;oXgvGi_ADo0=Xzq$?FdyEObdf50{RK};mPN?hky*-&zA4l+h;U$$tZ)|TUpm<AMf?}uvv3G8&LDAib5?ldS2f@AvhTdq18_n(e~f|hR}+;mR((z&UFK{fS+2dP5tz`tL8_7WrDEn47v_IsKibcl4WERk}$?G!_rw?UtD(dL<>9kfbr%CU>BzTo_lt{C1|TISMlO2W$t58@^0)YQmc*|V`6g|b4O){@3G_uOexL@>tb8mCLo6`sh8aC;|&NboXUQ0#j;VsYsJD|!9kqL{hX^W|-^pa-Zle@Y$A_Z~N-!CT+_-F97DkR=?g_(`syuCAxU2)F3M(2He+?hry?i|XSp)zt}?wK);N)q2jpiC0nKogXyF(-ZVHSsO_9J@nAPgvhMd?K#^cyCk&TJr%(rP+q)^K)M+^0#>Ouir31G%YrG4=dDBi786U0=39o;a6P}8H&_QT-epvLT#q6{;S6h*<p3O;9&YSAudVq@YEN(Za<U4O-z&2NN@jV%J^%y?7)AgT)P5Y6t2jMLl52~8f6Zyn4o@~`bLTy+a0*UiSAiPZ9F<8!s#8y7e=JRbn}Q&zqFaO7&yEJ+?-p{7AS$e>5X~AVl7zmlKG8D<S?VxvzV~PS!^6J|A9rQOQWPU*y$64V0;EKX$@MV*>4k=hJ9k7wsC&7uzGHrcmsb$L+LkgOAkyVQ0<YERXcQm{`?(|7xVE5yoLB;VfR7E;sbgQsZ(l==#w=SA6p9^nY=aP=4Ycl%BPkZ|SL{NDmmqdF>=fpWdi#BYMlj-D*K=}E+?InHW9_=LPa87oXkgZ;V3k}W^%OCuPJB*}|4aCu%fwg^gLTBFRsX4mM5_(mBmkDg{RQM-Q4#aSh{AGmoalNGFZ1224H&Ks9jb$r<Dn%uT+yP)BjekwsnE1t8Vt~57&c-YPaP1WR~BKa4lRsv7fygKdV=i@fqK@nlu=?RNz=p;;=*mUOvRM3VoPNEPK*&J6anT?l3Ej>Ho~y3;#@D(>LKoCp*aTn`7Q87jW2sYlLyQ`viR#u<V*DUQlgd(-0ioX3KiMhp<5pXBlNS{?U-Xuv5Vusg3}tLi~52M(G8mhbqE{iJz<M4a4Sgl8+49djz6kfTP})Fyplh0_-AmR<|D;sVqCnoIZgL~!A4>*?x#Q5;AghsT#-=^b`@Uz8}-HMGXMYrPBb#Qy?2+=mNS~`k}8(*ej+||XgznxCE>~$X&x(G;*e?tika~#NZoD9#=q-LIR;)N;VSXx<?X|L1Mmf0XBOqO$Y#M8B5(gYg5sVz$%z0*>Gz+_9Dk?H%X?CyIa3b}UR(ImVKFAyZuy9}=a}-F1ouy-;m6(=HQ0Y}cf-8l4cXkg#8w7gIUw<Vr@!%zra;`y_swyr({cbs0lo%}t=J7P@rgQ)4>}Dz1odPC3+vkwBIDMZS!41&C5ad5q7`odyMAbv@@+(-UVdg5Q6ahhrHf>AE}6>8Gj1Js-2<wJcN!X?SZuHQQvGB?fu|p|0@l{9ea7tgoTo{D5d{#|^#AJsT4%X7^fOZzc>+xiuzjO4h5OlHm=Ad*^c<rn+sX{6yacO_l~T`PvN=VpXun@tr{v+I0dOhhmXslt^L6bLb=8N^TopqIQfJyV?_qHH8g{Ipms^HDTP?5yG3mCR>;@W7UwHn?-83DRvAdoe|8vVkIIxHk!P~*b6}uIiK;u;o(tG05=XZ3#ztCt?SbRqt50~Pi{R8{wO{d#`xV{Lx@Q#_q>}f!{UY<j9KnchvQ?q1L5L;P$Q<koit~O)>jvCbFPSt(J9JiGpp>`#a`_?2WLS;4SqmEv#WJnRb`$rjD;WWo^6W^mtp%amQ?q^|Nx>rhiUg!cm_6jp!JC&ZbPW=U%##1HYU?^3vFKa+0!SVhJ?K9jRsi26Op+}2q9ZRlmN1syqHvSh6{UhyMX^wh?QcsJqkPq&}T&1BTKdBy@c=V7Iyw$M3ds8H{P<yK}{fqTc_YN?}VR830RGb7v2pE%;yP*16b%CfBIa_P#5&G+OLnX;QHT<79(*=4NtPGZ!7*`HL_1v}40$`IqtY$01IYW^+TK^?-C{Gx-l#NiXuJ&1*_zcn{vvJ`EcBR0&=MgR!j9JK&FH^#SG`EkS;YJW7H(t#nQ=#AkM<A#o)u}DQiVIarkSCT*Hv!@PsC{52^!hBLNU;;1xXM0tLvgu8{7DHb7Ql+))5*0R_D8YU;rhcmke73TPxB3HHb}GMVeU(?La+dcvAlR1d@z{X8ePV{UU<oes6@7Thm?M#Xki+jfeZ~898ZwL;q9m0N=MWUNAND=dYX&!>^%wRxRT_8v1M0@?gd|!{JL&9EB+_YQLkGli{l#%`uVH@lvgy3eZQ6BZ*q!Q%OdO|C|*B2ZZ;M9E<-VJW1S@5&%J;T+`)U~=j+bJRy5A==B6Td9U6j0p=pgcMDDGLq0T;VswOR=alCGOrQ3(bC?TSP4vFcG9T~3a)srz@sd=aR7sbCIK%A8H0~Qxzrj^CmuA_p)?R~pFw~*;Mn?o-5U*IOv%LCV@sgu$1w3fzyV#riuLLaa<^V-XiW+vTW2jp&VqkEl*>_Dp~ft&ZTQd=ZuHrYTQ?4pJxC_`KCw$ZHa0bauXKUMr2!H<<eQ~Q$tGuc~o0*D+7(!}&W!QOPkXA7yz6e9B?FNz;~Z9wF;Y%3XT@YS^B6X`1EXkIfTm!yeO{LX7vHmjGS*T&<aQ#VAP^p7o*0R=S1I*<#7T%N;zLy5TH(sD>Y@7tB=7JcpKcJEc>RmC-rWI&Avejlo4&$CeEVBDqQ!i%-Dxi}?VM$3ROLobmFF&*Rnc<4s*A-VB~5!kh|e1TGQLOMe4R#wqEKqvqh&lf~}k~&RE;JEYKkA$)SXBmahKp3h=k8zCHiL&l-7}@ExvgK%7J|Zi?zZ*M#Kj=%8ge(HUzClX^in%x*`IhE&A!nMa)(5nSa@L=F0nD-?E}6WXU=pQ%lkEAsd8F?;VRDYCy<=8n(w(9NOaiBW>P~Ny82)m!TWF#yN_N_};<Fo+ikHeft$e=-yG*`sDOrw@x0KkP8cUDmqX(S|-U`Yv0Bpu*M}(fX?zgLRfmWZmo*iI;NC#4rc&*6pJHly<lVB*M`GNiEI*5BF5P{Cc_8~@3xe*uW^4SF^9(>F+L#=MlLVtr(!u5v1YHFelh2csyNcAWL-RT06N=B@>4z?aHRaT@IjR~bNMkZR&ASBlKyqun+6EWiH9jtIfwl8(7_wWnZ?y2&|((%+auFjM_r>*f;0*XBpmX+X#|2doC7}14}V@Q*Nl)Dd+ov!#Wh6};+`_`;+rJ1vDVKu!g^C6bjC3~ktdjmZ{(nteobISKhSKl*(ieD)X+2cb$&?*YLfwXPcG)#O}u<A3jK2&2QVQFAb(!JA*<bTV=Q%n}_dMoqjs>{(1qkcskUpOXZ$Sh=*L<L5pRm{jj5`de`Ss)4#$%h5r34aU*%_GS!+zLD06Fn4LLGzTz0Bz#03-Qjgk_dO4?9>VO!W09i+i##BBF)Rf+05ZiOu*mCcLV&8>+Nr6YO{&JCZcq%9fizNDeco=IjE%b0#i!Nk|;%S!1HMR`w-e5*F&on<DCTx0fwt?);=tq`4g190`Y=Z@}|7YUb}h$Oz%&trEzkyt0rXJxueBQ!w;T`1ak_<t9dt)vm!CRJI2383YpUGYbM{}7YyzIMIHk!dS6rbaRknpg^B*(c9W)x_Fd_5Pfg!SCj!uuihm;Uf=$AJxOP^i<@)n^cx8R60BzS(oDx~M2+*A?>NG$X(?VI9pzO`gmP2JD*X-zfX@JGt1daplYwbhe&ej`AvAD$$Itx0N<a_MmVSo*BO<W+jt=pG1XWB_%4)?KIsH@4qN65DtpGhp*IbW=+%x>Yq55W1nvFdGf(OzLhVzMJj-fDleI9_;GE+HL&Nu?3W%8esNO*)XS-GrDs^F30ok%qJT0PS<$JmSB%L5bbnh@;0SPJW-4-K!WHW@!7s(oXP>pfK@4$KyEZZ#<<7_s=*eLjZ<CeXphE8wp^T{kDc)Bh#kFMS0GcM-tP?q6CSB@I@SxA^sqLmEHA$uJp+|v^|TEUHDQ0&Bo+U;}!8$rx5pQi(^ysRrN>_T2~YYv_1C)hcD+E$nH1Zotp+<ayO|l#Ft=g;BSa>l)(`A0z_gN#Y^ZRcf}Fqy@R{G6XTB9EKITnjM%|fJ5=^j2!etyTem053rk=op95;WDxtS=mr_hD_dT51Te;R<3TLtt-Q(u?Jiq|w;u(1^TnZK<#=pN2&M<>aka@8viy_Zp)1ytBA?misFSVl&4oO5oMk@FO=!((auu?uKbQc9t+PyMTl>zc{sO}P$Si55rtwc?p$s(;rbQhK8rhY*AZVx_u33D*d)+NIxC3vpgG5D0yH2D}cw|1z{u-zNZJr&2j-rRM>IMS;Ws!UXLX->!#5wHYYeK=wM=m}9rgdFg9FZ`3N1_JF(K$901S-1vjqr&NzLK5?!gE)F0#5bOxU{ZPWQTcOKXGq#&sJ0Y|_!#6lVY4D>bKa*-z$wS!{@Kj=5FQJQ-VmsJHKv~@O%6rkTd~e+s|Pc#y&GEY?(>t;>zpoOYr`UY>P@+GivU;xif_yk6|(7TJAMA;wc`{qO@w+fdZs~@7p&Lf03hdQ_a2BtVJy3qxq8=i3ED`XYyDHB5DDYuf^VhpijLk_Z05N@)`*AuR_~^XVM-FlK69lJ?3_4gW_<WN4OE&>pH2v6x`#HZK_J-eRfYxbxQmn&Y__tvA42*W?;BEsus|{PCL6-wg&0i7jidv?I{oHQo7bn?p=8v6nvf%t<Zs4CAV87e8k$EvuRYg3sR>OMHNycv4rq>t<PeKB8W-J~7Kq2#dy8sge45j9*{T<m&DSdRE6G1Zl$(6Focz;KMUZ`HJ#Z+B-96`ziJZ_&o;0=4<V-QIJ}LYdV;W#vUNBt>9d$(wmOVNk_=YIkPWY?(&b#VUXGPfJ?^dmyyaw&gXKJ=<#NpSsG+*{jiXWXG{M32}vcD|YEFBn=PUhay`+WB9YHA*c3A;+7f=MNJIPjRDN~rO%^|hRCo}HCZ-{9Is@lQ-M33l8q*IP3{tZ>2u&*-v{$Q0D3UBgIdQ|@q*=%dw?CRMio>JK#BR=8J9rCDzJgE#{)i&<2WZT@xS7vqd_tnl|yRXJf6WepaZ;w$wJ5=Wp#=+%c4$VM~-J(hL{S3CGuOi)s#3rYDh`t)DI<=*!Q9SX9lhejBzBVu0DnO66_4_D9#j~1)35q-VY0Y>=6W*wbgf!@E@!uLWenk#1a<YlZxlYm-9jmj8_c50qAKOiaB(g`8hw`OOyb?9<nx>AlVL4GRJQOte>_hpDp6a96QJWW&p*X%<6h}d&A&o-6u)Mz1yB&xO)xBQvM;PjBy{0t%DeLa-wW85>{w|nshv2zd;mLJ?R2GyzEvll7#b@t`dmO)jF2p8b0*bD3AINJoOGZwa&3K7uhr-bH6y`v38)7}RJcq_qKHWDbwly$P{<If&AD>;v4sw6|Nw>BE1DGzZBouCQ{eJ?1g(Hh)g2CkF<K$n6^80}FuN;MT}6CUBFo3#{$4u&4ozQ(Zu-6FIz64l$7*n*rDy}@|Zv0H41Xt~a|!n7w;>?_Mq!VI{K-+Fj#CFVPW=gxksF#(n}a=g_#rBVt++;d!d-MKV(=L?3b+xPG1Uu)ttQK{2n4QX@RA8=)~k$5!p8c(!e!>Mxw<$6;h+U>#wcOXD*I$=M$YZc9Uoei+6IZeqP{A%58WaV+uyHOGeb(pNV0D0Mp+O$M;7Re=QE;e1S&TW`>Fk-2Ik*)iY<w09fKKS0t0Syu0+J~_;f;7;)06C~(Mta#)AHww|e!U4p7}H`)GfhA2`FTD0cIz^d7?J+<yvVpLZTpEd3gbI<jF{-SkF%Uf%NYB~^4I_X&7j0Qyx;7V00G-4_^tr}Mzmr<vBYQl0ssI200dcD")).decode("utf-8"))      # NB0's build-time RC2 facts: argv tails / env / pins / TF anchor / oracle rows / prompt shas
    # the globals the picked NB0 / NB1 functions read (NB0 CELL 6 values; NB1 CELL 6 values)
    NB0 = {"topk": int(os.environ.get("NB0_TF_TOPK") or 5), "sc_tok": int(os.environ.get("NB0_SC_TOKENS") or 256),
           "tf_variant": 5, "sc_variant": {"2k": 6, "8k": 7, "40k": 6}, "tf_bands": ["2k", "8k", "40k", "90k"], "sc_bands": ["2k", "8k", "40k"],
           "last": 256, "boot_watchdog_sec": 1800.0, "ready_sec": 1500.0, "host_floor_mib": (69 + 4) * 1024}
    SCREEN = {"rank1_min": 0.90, "mean_max": 0.15}       # NB0 (external audit 2): conservative screening, not a published limit
    TRIAGE = {"mean_max": 1.0, "rank1_min": 0.5}         # NB0 (external audit 3): severe-discrepancy triage
    RAW_DIR = OUT / "tf_raw"
    RAW_DIR.mkdir(exist_ok=True)
    NB0_REQ = TRACES / "k100_requests.jsonl"       # every non-streaming request (teacher-forced, decode) with its full text
    TFR, SCR, LOADMON, SIDS = {}, {}, {}, set()       # E25 klee100 R1-15: DROP / WD were never filled (removed)
    PROC = Path(os.environ.get("NB0_PROC", "/proc"))
    CG = Path(os.environ.get("NB0_CGROUP", "/sys/fs/cgroup"))
    HOSTMEM = OUT / "hostmem"
    HOSTMEM.mkdir(exist_ok=True)
    BLOCK, PP, GEN_RATE = 4096, 2, 45.0           # NB1: block size, pipeline stages (one engine step = one micro-batch), window growth
    RAM_NEED_MIB = (69 + 4) * 1024                 # Merlin's ~69 GiB resident floor + 4 GiB margin (NB1 host_ram_gate)
    EXT_FLAG = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"]
    DIAG_ENV = {"VLLM_SERVER_DEV_MODE": "1"}      # /collective_rpc (read-only extension) + /reset_prefix_cache; 127.0.0.1 only


    def flags(argv):
        """--flag -> value, the LAST occurrence wins (argparse); a flag followed by another flag is a switch (True)."""
        out = {}
        for i, a in enumerate(argv):
            if isinstance(a, str) and a.startswith("--"):
                nxt = argv[i + 1] if i + 1 < len(argv) else None
                out[a] = nxt if (nxt is not None and not str(nxt).startswith("--")) else True
        return out


    ENV_KEYS = ("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "NB0_")


    def _set_value(a, flag, value):
        idx = [i for i, x in enumerate(a) if x == flag]
        if len(idx) != 1:
            raise ValueError(f"{flag} occurs {len(idx)} times in the Merlin argv (expected 1)")
        a[idx[0] + 1] = value


    def nb0_argv(argv, boot):
        """Merlin's argv with this boot's edits. MTP off = the same command without --speculative-config (Merlin
        models/flash-next/CHANGELOG.md: "the no-MTP line is the same command without --speculative-config"; launch.py:441/452
        always emits it, R.MTP_K = 3 has no env knob). seqs / captures: the value of the existing flag (no new flags)."""
        a = [str(x) for x in argv]
        if not boot["mtp"]:
            idx = [i for i, x in enumerate(a) if x == "--speculative-config"]
            if len(idx) != 1:
                raise ValueError(f"--speculative-config occurs {len(idx)} times (expected 1)")
            del a[idx[0]:idx[0] + 2]
        if boot.get("seqs"):
            _set_value(a, "--max-num-seqs", str(boot["seqs"]))
        if boot.get("captures"):
            _set_value(a, "--compilation-config", boot["captures"])
        if "--enable-prefix-caching" not in a:
            raise ValueError("prefix caching must stay on in every NB0 boot")
        return a


    def config_diff(argv, env, kv_bytes, ref="fp8"):
        """Flag / env / KV-pin diff of a boot vs RC2's recorded plan (last occurrence wins on both sides)."""
        rf, cf = flags(RC2_REF[ref]["argv_tail"]), flags(argv)      # RC2's tail starts at --host's value (127.0.0.1)
        head = set(flags(argv[:-len(RC2_REF[ref]["argv_tail"]) + 2])) - set(rf)   # Merlin flags before RC2's recorded window
        fl = {k: [rf.get(k), cf.get(k)] for k in sorted(set(rf) | set(cf))
              if k not in ("--port", "--host") and k not in head and rf.get(k) != cf.get(k)}
        renv = RC2_REF[ref]["env"]
        cenv = {k: v for k, v in env.items() if k.startswith(ENV_KEYS)}
        skip = ("VLLM_PORT", "MASTER_PORT", "NCCL_VERSION")
        ev = {k: [renv.get(k), cenv.get(k)] for k in sorted(set(renv) | set(cenv)) if k not in skip and renv.get(k) != cenv.get(k)}
        out = {"flags": fl, "env": ev}
        if kv_bytes != RC2_REF[ref]["kv_bytes"]:
            out["kv_bytes"] = [RC2_REF[ref]["kv_bytes"], kv_bytes]
        return out


    def cg_snapshot(full=True):
        """cgroup v2 memory files (MiB for sizes). avail_mib = max - (current - inactive_file), Merlin's own rule."""
        o, kv = {}, {}
        for n in ("memory.current", "memory.max", "memory.peak"):
            try:
                o[n] = (CG / n).read_text().strip()
            except Exception:
                pass
        try:
            st = (CG / "memory.stat").read_text()
            kv = dict(l.split() for l in st.splitlines() if len(l.split()) == 2)
            o["stat_mib"] = {k: int(kv[k]) >> 20 for k in ("anon", "file", "active_file", "inactive_file", "shmem", "file_mapped",
                                                           "file_dirty", "file_writeback") if k in kv}
            o["stat_count"] = {k: int(kv[k]) for k in ("pgmajfault", "workingset_refault_file", "pgscan", "pgsteal") if k in kv}
            if full:
                o["memory.stat"] = st[:6000]
        except Exception:
            pass
        for n in ("memory.events", "memory.pressure"):
            try:
                o[n] = (CG / n).read_text().strip()[:800]
            except Exception:
                pass
        try:
            mx = o.get("memory.max")
            if mx and mx != "max":
                o["avail_mib"] = (int(mx) - (int(o["memory.current"]) - int(kv.get("inactive_file", 0)))) >> 20
        except Exception:
            pass
        return o


    def _pids():
        try:
            return [int(p.name) for p in PROC.iterdir() if p.name.isdigit()]
        except Exception:
            return None


    def _stat(pid):
        try:
            t = (PROC / str(pid) / "stat").read_text()
        except Exception:
            return None
        i = t.rfind(")")
        f = t[i + 2:].split()
        try:
            return {"pid": pid, "comm": t[t.find("(") + 1:i], "state": f[0], "ppid": int(f[1]), "session": int(f[3]),
                    "minflt": int(f[7]), "majflt": int(f[9]), "utime": int(f[11]), "stime": int(f[12]), "rss_pages": int(f[21])}
        except Exception:
            return None


    def _io(pid):
        try:
            return {k: int(v) for k, v in (l.split(": ") for l in (PROC / str(pid) / "io").read_text().splitlines())
                    if k in ("rchar", "read_bytes", "syscr")}
        except Exception:
            return {}


    def session_procs(sids):
        out = []
        for pid in _pids() or []:
            st = _stat(pid)
            if st and st["session"] in sids and st["state"] != "Z":
                st.update(_io(pid))
                out.append(st)
        return out


    def _roles(text):
        return {int(p): n for n, p in re.findall(r"\((\w+) pid=(\d+)\)", text or "")}


    def load_monitor(s, key, deadline):
        """Every 5 s from launch to ready: cgroup current / avail / stat / pressure and every process of the server session
        (majflt, minflt, rchar, read_bytes, rss, cpu) -> logs/load_monitor_<boot>.jsonl + a per-process summary."""
        path = LOGS / f"load_monitor_{key}.jsonl"
        while s.proc is None and time.time() < deadline:
            time.sleep(0.5)
        if s.proc is None:
            return
        sid = s.proc.pid
        SIDS.add(sid)
        tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        t0, first, last, n = time.time(), {}, {}, 0
        peak = {"cg_current_mib_max": None, "avail_mib_min": None, "pressure_full_avg10_max": None}
        while s.alive() and s.ready_sec is None and time.time() < deadline:
            cg = cg_snapshot(False)
            ps = session_procs({sid})
            try:
                with open(path, "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - t0, 1), "current": cg.get("memory.current"),
                                         "avail_mib": cg.get("avail_mib"), "stat_mib": cg.get("stat_mib"),
                                         "stat_count": cg.get("stat_count"), "pressure": cg.get("memory.pressure"),
                                         "procs": ps}) + "\n")
            except Exception:
                pass
            for p in ps:
                first.setdefault(p["pid"], p)
                last[p["pid"]] = p
            try:
                c = int(cg["memory.current"]) >> 20
                peak["cg_current_mib_max"] = max(peak["cg_current_mib_max"] or 0, c)
            except Exception:
                pass
            if cg.get("avail_mib") is not None:
                peak["avail_mib_min"] = min(x for x in (peak["avail_mib_min"], cg["avail_mib"]) if x is not None)
            m = re.search(r"full avg10=([\d.]+)", cg.get("memory.pressure") or "")
            if m:
                peak["pressure_full_avg10_max"] = max(peak["pressure_full_avg10_max"] or 0.0, float(m.group(1)))
            n += 1
            time.sleep(0.5 if FAKE else 5)
        roles = _roles(s.log())
        per = {}
        for pid, a in first.items():
            z = last[pid]
            per[str(pid)] = {"role": roles.get(pid, a["comm"]), "majflt": z["majflt"] - a["majflt"],
                             "minflt": z["minflt"] - a["minflt"],
                             "read_gib": round((z.get("read_bytes", 0) - a.get("read_bytes", 0)) / 2 ** 30, 2),
                             "rchar_gib": round((z.get("rchar", 0) - a.get("rchar", 0)) / 2 ** 30, 2),
                             "rss_gib_end": round(z["rss_pages"] * 4096 / 2 ** 30, 2),
                             "cpu_s": round((z["utime"] + z["stime"] - a["utime"] - a["stime"]) / tck, 1)}
        loads = re.findall(r"\((\w+) pid=(\d+)\)[^\n]*Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", s.log() or "")
        LOADMON[key] = {"samples": n, "ready_sec": s.ready_sec, "alive": s.alive(), "per_process": per, "peak": peak,
                        "model_loading": [[w, int(p), float(g), float(sec)] for w, p, g, sec in loads], "file": path.name}


    def _post(port, body, timeout=1800):
        t = time.perf_counter()
        try:
            r = json.loads(UR.urlopen(UR.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"}), timeout=timeout).read())
            return r, None, round(time.perf_counter() - t, 2)
        except UE.HTTPError as e:
            try:
                msg = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                msg = ""
            return None, f"HTTP {e.code}: {msg}", round(time.perf_counter() - t, 2)
        except Exception as e:
            return None, repr(e)[:300], round(time.perf_counter() - t, 2)


    def _req_log(rec):
        try:
            with _SAVE_LOCK, open(NB0_REQ, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass


    def ids_sha(ids):
        return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()


    def _cached(r):
        return (((r or {}).get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


    def pl_arrays(pl, ids, k):
        """Every prompt position p >= 1 of an OpenAI prompt_logprobs list: the actual token's logprob and rank, the unique
        rank-1 id (E23's rule: an ambiguous rank 1 is -1), the best logprob, the top-k ids / logprobs by rank. NaN is kept
        (a non-finite value is never data); a missing position is valid=False."""
        n = len(ids)
        a = {"ids": np.asarray(ids, np.int32), "lp": np.full(n, np.nan, np.float32), "rank": np.full(n, -1, np.int32),
             "top1": np.full(n, -1, np.int32), "top1_lp": np.full(n, np.nan, np.float32),
             "topk_ids": np.full((n, k), -1, np.int32), "topk_lp": np.full((n, k), np.nan, np.float32), "valid": np.zeros(n, bool)}
        amb = 0
        for pos in range(1, n):
            d = pl[pos] if pos < len(pl) else None
            if not isinstance(d, dict):
                continue
            e = d.get(str(ids[pos])) or d.get(f"token_id:{ids[pos]}")
            if not isinstance(e, dict):
                continue
            a["valid"][pos] = True
            v = e.get("logprob")
            a["lp"][pos] = float(v) if isinstance(v, (int, float)) else np.nan
            a["rank"][pos] = int(e.get("rank") or -1)
            ent = []
            for kk, vv in d.items():
                if isinstance(vv, dict) and isinstance(vv.get("logprob"), (int, float)):
                    t = L._tok_id(kk)
                    if isinstance(t, int):
                        ent.append((vv.get("rank") or 10 ** 9, -vv["logprob"], t, vv["logprob"]))
            ent.sort()
            r1 = [x for x in ent if x[0] == 1]
            if len(r1) == 1:
                a["top1"][pos] = r1[0][2]
            else:
                amb += 1
            if ent:
                a["top1_lp"][pos] = max(x[3] for x in ent)
            for j, x in enumerate([x for x in ent if x[0] <= k][:k]):
                a["topk_ids"][pos, j], a["topk_lp"][pos, j] = x[2], x[3]
        a["top1_ambiguous"] = amb
        return a


    def e23_rows(a, last=256):
        """The same per-position rows E23's teacher_forced() keeps ([logprob, rank, rank-1 id]) for the last `last` positions,
        so RC2's compare_tf (with its non-finite guard) computes the RC2-comparable metric."""
        n, out = len(a["ids"]), []
        for pos in range(max(1, n - last), n):
            if not a["valid"][pos]:
                out.append(None)
                continue
            out.append([float(a["lp"][pos]), int(a["rank"][pos]), int(a["top1"][pos]) if a["top1"][pos] >= 0 else None])
        return {"positions": out, "n": len(out)}


    def tf_request(key, port, ids, band, run, k=None):
        """One teacher-forced request (max_tokens 1, T 0, prompt_logprobs k, ids as ids). k > 1 refused by Merlin's prompt-
        scoring budget (prompt_logprobs.py validate) -> retried with k 1 = RC2's value, recorded. Raw arrays -> tf_raw/*.npz."""
        k = k or NB0["topk"]
        body = {"model": "q27", "prompt": [int(x) for x in ids], "max_tokens": 1, "temperature": 0, "prompt_logprobs": k,
                "return_tokens_as_token_ids": True}
        r, err, sec = _post(port, body)
        note = None
        if err and k > 1 and "prompt-scoring tensor bound" in err:
            note = f"prompt_logprobs {k} refused ({err[:160]}); retried with 1 (RC2's value)"
            k = body["prompt_logprobs"] = 1
            r, err, sec = _post(port, body)
        meta = {"boot": key, "band": band, "run": run, "k": k, "note": note, "sec": sec, "error": err, "prompt_len": len(ids),
                "prompt_sha256": ids_sha(ids), "cached_tokens": _cached(r), "t": round(time.time() - T_START, 1)}
        arr = None
        if r is not None:
            try:
                ch = r["choices"][0]
                pl = ch.get("prompt_logprobs") or []
                meta.update(prompt_logprobs_len=len(pl), gen_token_ids=ch.get("token_ids"), gen_text=ch.get("text"))
                arr = pl_arrays(pl, ids, k)
                vv = arr["valid"]
                meta.update(valid=int(vv.sum()), expected=len(ids) - 1, nonfinite=int((~np.isfinite(arr["lp"][vv])).sum()),
                            top1_ambiguous=arr["top1_ambiguous"])
                if meta["valid"] < meta["expected"] and not meta["error"]:   # E25 klee100: a 200 without usable prompt logprobs
                    meta["error"] = f"INCOMPLETE_TF valid {meta['valid']} of {meta['expected']}"
                f = RAW_DIR / f"{key}_{band}_{run}.npz"
                np.savez_compressed(f, **{x: arr[x] for x in ("ids", "lp", "rank", "top1", "top1_lp", "topk_ids", "topk_lp", "valid")})
                meta["raw"] = f.name
            except Exception as e:
                meta["error"] = meta["error"] or f"PARSE {type(e).__name__}: {e}"[:300]
                arr = None
        _req_log({**meta, "kind": "teacher_forced", "body": {kk: v for kk, v in body.items() if kk != "prompt"}})
        return {"meta": meta, "arr": arr}


    def decode(key, port, ids, n, band, run):
        """Greedy decode (T 0, ignore_eos, logprobs k): tokens, chosen-token logprobs, rank-1 id / logprob, rank-2 logprob."""
        body = {"model": "q27", "prompt": [int(x) for x in ids], "max_tokens": n, "temperature": 0, "logprobs": NB0["topk"],
                "return_tokens_as_token_ids": True, "return_token_ids": True, "ignore_eos": True}
        r, err, sec = _post(port, body, timeout=900)
        meta = {"boot": key, "band": band, "run": run, "sec": sec, "error": err, "prompt_len": len(ids), "prompt_sha256": ids_sha(ids),
                "cached_tokens": _cached(r), "t": round(time.time() - T_START, 1)}
        toks, lps, t1, t1lp, t2lp = [], [], [], [], []
        if r is not None:
            try:
                ch = r["choices"][0]
                lp = ch.get("logprobs") or {}
                toks = ch.get("token_ids") or [L._tok_id(t) for t in (lp.get("tokens") or [])]
                lps = [x if isinstance(x, (int, float)) else float("nan") for x in (lp.get("token_logprobs") or [])]
                for top in lp.get("top_logprobs") or []:
                    it = sorted(((v, L._tok_id(kk)) for kk, v in (top or {}).items() if isinstance(v, (int, float))), reverse=True)
                    t1.append(it[0][1] if it and isinstance(it[0][1], int) else -1)
                    t1lp.append(it[0][0] if it else float("nan"))
                    t2lp.append(it[1][0] if len(it) > 1 else float("nan"))
                meta.update(tokens=len(toks), finish=ch.get("finish_reason"), text=ch.get("text"))
                if any(not isinstance(x, int) or isinstance(x, bool) for x in toks):
                    meta["error"] = "INVALID_TOKEN_IDS"
                elif len(toks) != n or len(lps) != n or len(t1) != n:
                    meta["error"] = f"INCOMPLETE_DECODE tokens {len(toks)} logprobs {len(lps)} top {len(t1)} of {n}"
            except Exception as e:
                meta["error"] = meta["error"] or f"PARSE {type(e).__name__}: {e}"[:300]
        out = {"meta": meta, "tok": [int(x) for x in toks if isinstance(x, int)], "lp": np.asarray(lps, np.float64),
               "top1": np.asarray(t1, np.int64), "top1_lp": np.asarray(t1lp, np.float64), "top2_lp": np.asarray(t2lp, np.float64)}
        if not meta["error"]:
            f = RAW_DIR / f"{key}_dec{band}_{run}.npz"
            np.savez_compressed(f, ids=np.asarray(ids, np.int32), tok=np.asarray(out["tok"], np.int32), lp=out["lp"],
                                top1=out["top1"], top1_lp=out["top1_lp"], top2_lp=out["top2_lp"])
            meta["raw"] = f.name
        _req_log({**meta, "kind": "decode", "token_ids": out["tok"], "body": {kk: v for kk, v in body.items() if kk != "prompt"}})
        return out


    def cmp_last256(ra, rb):
        if not ra or not rb or ra.get("arr") is None or rb.get("arr") is None:
            return {"status": "INCONCLUSIVE_MISSING_TF"}
        if ra["meta"]["prompt_sha256"] != rb["meta"]["prompt_sha256"]:
            return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
        return L.compare_tf(e23_rows(ra["arr"], NB0["last"]), e23_rows(rb["arr"], NB0["last"]))


    def chunk_check(pos, d, edge=8):
        """|dlogprob| at the first / last `edge` positions of each 1024-token scheduler chunk (--max-num-batched-tokens) and
        each 256-row prompt-logprob chunk (VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE) vs mid-chunk."""
        out = {}
        for unit in (1024, 256):
            off = pos % unit
            e = (off < edge) | (off >= unit - edge)
            if e.any() and (~e).any():
                me, mm = float(d[e].mean()), float(d[~e].mean())
                out[str(unit)] = {"edge_mean": round(me, 4), "mid_mean": round(mm, 4), "edge_n": int(e.sum()),
                                  "localised": bool(me > 3 * mm + 0.05)}
        return out


    def cmp_full(a, b, bins=True):
        """Every prompt position: mean / p99 / max |dlogprob|, rank-1 agreement, 4096-token bins (does it grow with context?),
        the first 256-position window over the screening mean, chunk-edge localisation."""
        if a["ids"].shape != b["ids"].shape or not np.array_equal(a["ids"], b["ids"]):
            return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
        va, vb = a["valid"] & np.isfinite(a["lp"]), b["valid"] & np.isfinite(b["lp"])
        if (a["valid"] & ~np.isfinite(a["lp"])).any() or (b["valid"] & ~np.isfinite(b["lp"])).any():
            return {"status": "INCONCLUSIVE_NONFINITE"}
        pos = np.nonzero(va & vb)[0]
        n = len(a["ids"])
        if not len(pos):
            return {"status": "INCONCLUSIVE_NO_VALID"}
        d = np.abs(a["lp"][pos].astype(np.float64) - b["lp"][pos].astype(np.float64))
        t1 = (a["top1"][pos] == b["top1"][pos]) & (a["top1"][pos] >= 0)
        ds = np.sort(d)
        out = {"status": "MEASURED" if len(pos) == n - 1 else "PARTIAL", "n": int(len(pos)), "expected": n - 1,
               "mean_abs_dlogprob": round(float(d.mean()), 4), "p99_abs_dlogprob": round(float(ds[min(len(ds) - 1, int(round(0.99 * (len(ds) - 1))))]), 3),
               "max_abs_dlogprob": round(float(d.max()), 3), "rank1_agreement": round(float(t1.mean()), 4)}
        if bins:
            bl = []
            for lo in range(0, n, 4096):
                m = (pos >= lo) & (pos < lo + 4096)
                if m.any():
                    bl.append([lo, int(m.sum()), round(float(d[m].mean()), 4), round(float(t1[m].mean()), 4)])
            out["bins_4096"] = bl                        # [start, n, mean |dlogprob|, rank-1 agreement]
            if len(d) >= 256:
                cs = np.convolve(d, np.ones(256) / 256, "valid")
                idx = np.nonzero(cs > SCREEN["mean_max"])[0]
                out["first_window_over_screen_mean"] = int(pos[idx[0]]) if len(idx) else None
            out["chunk"] = chunk_check(pos, d)
        return out


    def shift_check(a, b, window=2048, shifts=(-2, -1, 0, 1, 2)):
        """Alignment test on the last `window` positions: reference rank-1 id at p vs candidate rank-1 id at p + s. A shifted
        agreement >= 0.9 that beats s = 0 by > 0.2 = a position-alignment artefact."""
        n = min(len(a["top1"]), len(b["top1"]))
        i0 = np.arange(max(1, n - window), n)
        res = {}
        for s in shifts:
            j = i0 + s
            m = (j >= 1) & (j < n)
            i, j = i0[m], j[m]
            v = a["valid"][i] & b["valid"][j] & (a["top1"][i] >= 0) & (b["top1"][j] >= 0)
            if not v.any():
                res[str(s)] = None
                continue
            dl = np.abs(a["top1_lp"][i][v].astype(np.float64) - b["top1_lp"][j][v].astype(np.float64))
            res[str(s)] = {"n": int(v.sum()), "top1_agree": round(float((a["top1"][i][v] == b["top1"][j][v]).mean()), 4),
                           "top1_lp_mean_abs_d": round(float(np.nanmean(dl)), 4) if np.isfinite(dl).any() else None}
        ok = {k: x for k, x in res.items() if x}
        best = max(ok, key=lambda k: (ok[k]["top1_agree"], k == "0")) if ok else None
        a0 = (ok.get("0") or {}).get("top1_agree")
        art = bool(best and best != "0" and ok[best]["top1_agree"] >= 0.9 and (a0 is None or ok[best]["top1_agree"] > a0 + 0.2))
        return {"shifts": res, "best_shift": int(best) if best is not None else None, "alignment_artefact": art}


    def screen(c):
        if not isinstance(c, dict) or c.get("status") != "MEASURED":
            return "INCONCLUSIVE"
        m, r1 = c.get("mean_abs_dlogprob"), c.get("rank1_agreement")
        if not (_finite(m) and _finite(r1)):
            return "INCONCLUSIVE_NONFINITE"
        return "DIVERGENT" if (r1 < SCREEN["rank1_min"] or m > SCREEN["mean_max"]) else "CLEAN"


    def decode_compare(ref, cand):
        """Greedy rows token for token: first divergence, the reference's top-1 - top-2 margin there (E23's 0.5 rule:
        FAIL_HIGH_MARGIN at or above, NEAR_TIE below), mean |dlogprob| over the common prefix."""
        if not ref or not cand or ref["meta"].get("error") or cand["meta"].get("error") or not ref["tok"] or not cand["tok"]:
            return {"status": "INCONCLUSIVE_MISSING"}
        x, y = ref["tok"], cand["tok"]
        d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        c = len(x) if d is None else d
        dl = np.abs(ref["lp"][:c] - cand["lp"][:c]) if c else np.zeros(0)
        if not np.isfinite(dl).all() or (d is not None and not math.isfinite(float(ref["top1_lp"][d] - ref["top2_lp"][d]))):
            return {"status": "INCONCLUSIVE_NONFINITE", "first_diff": d}
        out = {"n": len(x), "first_diff": d, "common_prefix_mean_abs_dlogprob": round(float(dl.mean()), 4) if c else None,
               "common_prefix_max_abs_dlogprob": round(float(dl.max()), 4) if c else None}
        if d is None:
            out["status"] = "EXACT" if len(x) == len(y) else "INCONCLUSIVE_LENGTH"
        else:
            m = float(ref["top1_lp"][d] - ref["top2_lp"][d])
            out.update(ref_margin_at_diff=round(m, 4), status="FAIL_HIGH_MARGIN" if m >= L.FAIL_MARGIN else "NEAR_TIE")
        return out


    def sc_compare(dec, tf, P):
        """Decode-vs-prompt-logprob self-consistency on ONE boot: the greedy decode's chosen-token logprob at step i vs the
        teacher-forced logprob of the same token at position P + i (prompt + those tokens), rank-1 ids, +-2 shift."""
        if not dec or dec["meta"].get("error") or not tf or tf.get("arr") is None:
            return {"status": "INCONCLUSIVE_MISSING"}
        a, n = tf["arr"], len(dec["tok"])
        if n == 0:                                             # A-4: nothing decoded, nothing to compare (no IndexError)
            return {"status": "INCONCLUSIVE_EMPTY_DECODE"}
        pos = np.arange(P, P + n)
        if pos[-1] >= len(a["ids"]):                           # the TF prompt does not cover every decoded position
            return {"status": "INCONCLUSIVE_TF_TOO_SHORT", "need_len": int(P + n), "tf_len": int(len(a["ids"]))}
        if not np.array_equal(a["ids"][pos], np.asarray(dec["tok"], np.int32)):
            return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
        dl, tl, v = dec["lp"], a["lp"][pos].astype(np.float64), a["valid"][pos]
        if not np.isfinite(dl).all() or not np.isfinite(tl[v]).all():
            return {"status": "INCONCLUSIVE_NONFINITE"}
        if not v.all():
            return {"status": "INCONCLUSIVE_INCOMPLETE_TF", "valid": int(v.sum()), "expected": n}
        d = np.sort(np.abs(dl - tl))
        r1 = float((dec["top1"] == a["top1"][pos]).mean())
        sh = {}
        for s in (-2, -1, 0, 1, 2):
            j = pos + s
            m = (j >= 1) & (j < len(a["ids"]))
            if m.any():
                sh[str(s)] = round(float((dec["top1"][m] == a["top1"][j[m]]).mean()), 4)
        out = {"status": "MEASURED", "n": n, "mean_abs_dlogprob": round(float(d.mean()), 4),
               "p99_abs_dlogprob": round(float(d[min(n - 1, int(round(0.99 * (n - 1))))]), 3), "max_abs_dlogprob": round(float(d[-1]), 3),
               "rank1_agreement": round(r1, 4), "greedy_self_top1": round(float((np.asarray(dec["tok"]) == dec["top1"]).mean()), 4),
               "shift_top1_agree": sh}
        out["screen"] = screen(out)
        return out


    TF_EST = {"2k": 15, "8k": 25, "40k": 60, "90k": 150}


    SC_EST = {"2k": 40, "8k": 60, "40k": 120}


    def tf_suite(key, s, bands, runs=("fresh", "repeat")):    # E25 klee100 R2-4
        out = {}
        for band in bands:
            ids = PROMPTS[band][NB0["tf_variant"]]
            rec = TFR.setdefault(key, {}).setdefault(band, {})
            o = {"prompt_len": len(ids), "prompt_sha256": ids_sha(ids), "variant": NB0["tf_variant"]}
            for run in runs:                            # E25 klee100 R2-4: K0 runs the fresh requests only
                if not s.alive():
                    o[run] = {"status": "SKIPPED_SERVER_DEAD"}
                elif ABORT["why"] or left() < TF_EST[band] + 240:        # E25 klee100 R1-15: WD (never set) removed
                    o[run] = {"status": "SKIPPED_BUDGET"}
                elif not due(TF_EST[band] + 300):          # E25 klee100 K-1: the boot body cap, not only the run budget
                    o[run] = {"status": "SKIPPED_BOOT_BUDGET"}
                else:
                    rec[run] = tf_request(key, s.port, ids, band, run)
                    o[run] = rec[run]["meta"]
            o["aa"] = cmp_last256(rec.get("fresh"), rec.get("repeat"))
            out[band] = o
            save()
        return out


    def sc_suite(key, s):
        out = {}
        for band in NB0["sc_bands"]:
            ids = PROMPTS[band][NB0["sc_variant"][band]]
            if not s.alive():
                out[band] = {"status": "SKIPPED_SERVER_DEAD"}
                continue
            if ABORT["why"] or left() < SC_EST[band] + 240:          # E25 klee100 R1-15: WD (never set) removed
                out[band] = {"status": "SKIPPED_BUDGET"}
                continue
            if not due(SC_EST[band] + 300):                  # E25 klee100 K-1: the boot body cap, not only the run budget
                out[band] = {"status": "SKIPPED_BOOT_BUDGET"}
                continue
            cold = decode(key, s.port, ids, NB0["sc_tok"], band, "cold")
            cached = decode(key, s.port, ids, NB0["sc_tok"], band, "cached")
            tf = tf_request(key, s.port, list(ids) + list(cold["tok"]), "sc" + band, "tf") if not cold["meta"].get("error") else None
            SCR.setdefault(key, {})[band] = {"cold": cold, "cached": cached, "tf": tf, "P": len(ids)}
            out[band] = {"cold": cold["meta"], "cached": cached["meta"], "tf": (tf or {}).get("meta"),
                         "decode_vs_tf": sc_compare(cold, tf, len(ids)), "cold_vs_cached": decode_compare(cold, cached),
                         "cold_was_cold": cold["meta"].get("cached_tokens") in (0, None)}
            save()
        return out


    def scales_facts(t):
        """The static scales sidecar's values, the per-stage 'applied ... to N of 12' lines, every 'E4M3 KV active on <layer>'
        line matched to the file (logged %.6g), and the MTP layer's DEFAULT 1.0 line."""
        path = CKPT / "qsa_kv_scales_262k.json"
        try:
            raw = path.read_bytes()
            file, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
        except Exception as e:
            file, sha = {}, f"unreadable: {e!r}"[:120]
        applied = re.findall(r"\((\w+) pid=(\d+)\)[^\n]*applied static K/V scales from (.+?) to (\d+) of (\d+) layer\(s\) "
                             r"\(pipeline layers (\d+)-(\d+)\)", t)
        act = {}
        for name, k, v, dflt in re.findall(r"QSA E4M3 KV active on (\S+): k_scale=([-\d.eE+]+) v_scale=([-\d.eE+]+)(\s+\(DEFAULT 1\.0)?", t):
            act.setdefault(name, set()).add((float(k), float(v), bool(dflt)))
        tol = lambda a, b: abs(a - b) <= 1e-5 * max(1e-12, abs(b))
        matched, mismatched = [], []
        for name, sc in sorted(file.items()):
            got = act.get(name)
            if got:
                (matched if all(tol(k, sc["k_scale"]) and tol(v, sc["v_scale"]) and not d for k, v, d in got) else mismatched).append(name)
        stages = sorted({f"{a}-{b}" for *_, a, b in applied})
        return {"file": str(path), "sha256": sha, "values": file, "applied_lines": len(applied), "applied_stages": stages,
                "applied_counts": sorted({f"{n} of {m}" for _, _, _, n, m, _, _ in applied}),
                "applied_paths": sorted({p for _, _, p, *_ in applied}), "active_layers": sorted(act), "matched": len(matched),
                "mismatched": mismatched, "missing": sorted(set(file) - set(act)), "not_in_file": sorted(set(act) - set(file)),
                "mtp_default_scale": sorted(n for n, g in act.items() if any(d for _, _, d in g)),
                "ok": bool(file) and not mismatched and not (set(file) - set(act)) and stages == ["0-25", "25-48"]}


    def psha(ids):
        return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


    P60490 = "# vLLM PR #60490 (issue #60489) rebased onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5,\n# tree 07c16a79d077f1e06c19d0eea84efd64f0b63540 = the tree the Kaggle payload installs, install_merlin.log).\n# Same code as upstream; only the import context differs (Merlin's import block). One marker comment added.\n# pre-patch sha256  a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653\n# post-patch sha256 88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b\n--- a/vllm/models/qwen4_exp/nvidia/qsa.py\n+++ b/vllm/models/qwen4_exp/nvidia/qsa.py\n@@ -18,8 +18,9 @@\n from torch import nn\n from transformers import Qwen4ExpTextConfig\n \n+import vllm.envs as envs\n from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n-from vllm.config import VllmConfig\n+from vllm.config import VllmConfig, get_current_vllm_config_or_none\n from vllm.config.cache import CacheConfig, CacheDType\n from vllm.distributed import get_tensor_model_parallel_world_size\n from vllm.forward_context import get_forward_context\n@@ -45,6 +46,7 @@\n )\n from vllm.platforms import current_platform\n from vllm.platforms.interface import DeviceCapability\n+from vllm.utils.math_utils import cdiv\n from vllm.utils.torch_utils import (\n     LayerNameType,\n     _encode_layer_name,\n@@ -75,6 +77,7 @@\n from . import model\n from .indexer_qsa import QSAIndexer\n from .ops.qsa import qsa_e4m3_uses_native_cast\n+from .ops.qsa_indexer import _TOPK_WORKSPACE_BYTES\n \n logger = init_logger(__name__)\n \n@@ -782,6 +785,31 @@\n         if isinstance(metadata, list):\n             metadata = metadata[0]\n         if not isinstance(metadata, dict):\n+            # E25 NB1: backport of vLLM PR #60490 (issue #60489).\n+            # Profiling/dummy run: the indexer allocates its selection scratch\n+            # on demand, so touch its peak here for the memory profiler (as\n+            # sparse_attn_indexer does): the prefill logits budget or the\n+            # worst-case decode logits [decode tokens, compressed columns],\n+            # plus the top-k workspace.\n+            logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024\n+            cfg = get_current_vllm_config_or_none()\n+            if cfg is not None:\n+                spec = cfg.speculative_config\n+                num_spec = spec.num_speculative_tokens if spec is not None else 0\n+                decode_tokens = min(\n+                    cfg.scheduler_config.max_num_seqs * (num_spec + 1),\n+                    cfg.scheduler_config.max_num_batched_tokens,\n+                )\n+                block_size = cfg.cache_config.block_size\n+                columns = cdiv(cfg.model_config.max_model_len, block_size) * cdiv(\n+                    block_size, self.indexer.compress_ratio\n+                )\n+                logits_bytes = max(logits_bytes, decode_tokens * columns * 4)\n+            _ = torch.empty(\n+                logits_bytes + _TOPK_WORKSPACE_BYTES,\n+                dtype=torch.uint8,\n+                device=output.device,\n+            )\n             output.zero_()\n             return\n         main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])\n"


    P60490_REL = "vllm/models/qwen4_exp/nvidia/qsa.py"


    P60490_PRE, P60490_POST = "a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653", "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b"


    P60490_MARKER = "# E25 NB1: backport of vLLM PR #60490 (issue #60489)."


    NB1_EXT_SRC = __import__("lzma").decompress(__import__("base64").b85decode("{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;7&pozg++kXbna~Q3)_)a?O~7&8s&XzCrDUt$=>+AW^&O`u)m#g}wasYu6yziojx^fdlzib|`tVSZ!7j=mk{#>a}I2B*tnxVF|^0b_hG(N<&EA(FGYC2|8MwvElLUfib7m-MSKl6EQ{lF}xbgJL7kvZ7ewB^?M*n1-%AU;SUbD47>A$+pJ0E%wrYu@Nmb)VnUw>Fe*^ieWq+;b3WNh*=sSXY3`^l4kKGp(Xu}+a%KP5EMFO3ng>O%pyW7uh+RsmX)w&4`P20>-~z_=$wD!B{8+%APE(zV*Zndo@9K~{JPIgBE6N+=XU*6Y8@thuiqKM`*OWtEmo!t7n7*hlJW_)L1?$&ML~|e^bBT<AwgX^^YEbB%-Lmpy*XoAWeEg`fnPgnsu5En`YdPX76~p4QyK8eLm6LP|zq5|Mb8CW?D)8bYr>J}UVSHU+B~|+RzT=;1lVLpWqwQ&lvxRWTtgDIdEhIIq@WWcNOlMA+LiQzPG)8B)&bvR&+}}3YJ1r&s;#If;Wl;dccZpjeSBPV2pdq1juSt2md{=xf7U;ZdKmO3LEq>$0DI}V1X{2GaloO<FyLz8!;DjuP@bN->rX7w(F2yRSQSE_Dr6}SUdp`ZHBZDiI*I2zHP<)#?zk*ZIY8tuK!2?d>Yev^xdCHR8H^R%IPU&Z?j`~=)T@z7-IAjjZQoc>1y{Fw}?M-S;K<Cp5I-ka*ZNy?+*qaRtyVdpeVs>lkZ*ZC~m#hX}%4=JT@d~IQOvh1*m(9Xp@~tM!Lk|0|KmD&<T}Egao4|9u)51UHhyw_tP^3U}MOzzD3@7|}?kPC`#HK$_`;Fkoq+g3<@#sYg=}Vd*(YBF#L|cKa^Vb@!nc-&*!5ELk<#Q{7G;CDI{<?gDZZtCNQc*1S$w`#=_$OuoH<MCi;>A-#m@2DG6A5QWCZNZhj&EOu%Bs6VMGcucouf%DSl$gdN&?!Ngl_Z$sj55^0s!)^ewT)_(UZbAIHn~0%=(y@jS913St?%oO?56SswWd+du2Knf?P2M<6ICSoa3|dXx}uP#S1p`6XfW=FXErd!;N@b$b*n$M9aMK-k3NS(i~00@WNPUb7$6kB&h6BCV*1-VZkV`Hfea=D8ZcAPE=F$p$ZyMcS8qWdXKj5HV>u5M`DuxY$DdzE#<wpPE>OdUabdnW;RE)-y|7@(<{z+VCf5VW=I5}sSv#SwpDr*4nm2St|1P(sY=ntwI}q{P4PWe1Ot`BhXH!`n|`8v((uRHrZUH1e~?9cwk~_A<nNrc{nBP{wL-ld0{5e3>036~7zo`9c_GhGQl!JOO@o!J6x7tyUYu7wIwGtvek7VlUvG5|%m`}`Sf3J2PdpTdU_riLA9t{&DnyD$U{SHt{z~XL8V!#%vRK~H7gI9Sy>yJK+i1YzFR>{(bF*>3K(_8uFMM1u8@LlZY!WOgNn26LO#zYWFm-;HM09*sw`I)bKM%4*4nEv~%$-#Z#8Os+Yf@1!EA`yE*370KLYVUvF(jTs6DhxQz37))ei*83pJ)qCqxh5m{*`;29PrCL2D!2AD=37YSZ?LG?{aJ|RC~Q=5u1@CkR%z4``;Uws{mNTR@|h2a&S7Ia(ZLADFQX3|BJX>KJWwxgc!UJ$({iYEj`k|Q;k%4t)JnZLGWe`>JVA9V|%=DJPn+5esiHn?&eXP4+XHUhSW(#1XE|PtJApty1(a<zTMsq=$Lwy-MiHJD!LJP6rW*K)eFEO{R?t4EKsL8^N1J;5mvutl;zkG<f6_kb!dH~`<y8N+=vq~CUv}!dD9t|4UV6udk8@M{MJ&=cb*gHP9FRBu?0D=)fJ%KLi*QT(rwW5_lGx=P1wvA#W7`bppY2;$AAc}Z2q>p+Ka!#2oTIWuAuZL!0Pbet#g-E2h_Q&V(MwJ*rQj12x1=Is87fWj-R95-kar}sCnSu$5ItEt%z#*efUvWfSC`d^C4)FVnwce^_l3$bmPDU&XH))3jt=#HD<wMD&C7~;TTq?H3co6{HbeTFKWCakJj=(Rl?`1a+H;aW)>2>6^x<voBy?iwK!QgSBgQv+J}{sK=o)crR>OtHQ>;xeU*~@&~%KkT8=?8j7We<bbKwe+1YDXQo1b)T&KZp+jBYBut~~9IY9Ba@tL(k8m@Y6QoJn{UAPi10knTDoCcr)qitTd+v8X!2J`737M=j76an_iVHD>qma{ED(XZ=@NFbUE{APV%zrLN}PDKo~l$l`n?;wiqx<Df5YqAcA6~Wm+QF347D7Ba$rg6>xZChUZ_N}ZKb<vJYX5T0G_`8i|)GzKj8CnEx`9Q#tMJ}06VkH~{DR^y!r|aVXWce(|mXKzPXzuVp!wb?dpRwBxQTmtZjvT)BX)?<yJP6kcWM^|T@%h=#9{5I;$z-iz+lB;80t`h>>HNC02MJbQP*q`xwW^mwjdS4%B&d?23IirKU1(3rL*$E6sYS8{)!<lSj+i9H!Z2X*26Ek-ySt=OY@`M)q+iinc?ZwkE9v34^J)*|7u-)HA4XRXQ*64trh1mAF>I2pEX0tYU1q2az-%ZAmuI(aJKp%p`uf$#mc=@ZV8<3fvF(W8!_zR3!?(P@#S$@g27#H*%lJTq%?RhE$!{gy$h(|@D>v#Kc;je)WB+E_ys2$}_bS`-2eI!xs(^C5;W4}@+VIY&npUMF1@XXqut-&pM<-)ar8F0z<)dR{?3dhv!vi(2c#FUrDBU{EJ=SH15YfOzas`AYRLNx&>Kv(z;@)-XliL4zpcP0gCm4ZG<8Yap>prPFVOK9cgT^ab8&}h+>7Kfo!eOuONo4#cVq=;Qg^kE_sD)Il>bWRz-)hT0H+htF6Wb|Ty(qg`u_IGb(TTK_u%MroIChOv?0?H3c_9IE_6c;6$y}O_(ueq{#GeJ1@Z`>@T*Y)m;X~I@)p|=71ItPXD#$)X<|qD)mhXK57LTvf5Gw1@eR4(fE_bqsFlWXx?VpCo{6LqW9E@$~Q?LxJ4>o001l+K-+Y&-o>m}Wax+CD;B={;*H3cm?UE5(2bwW_ApC;)Le-5e=pn-q!r}H;m*|8TR>v0R%?1Tm3AUL=%8umKK4g}~Dht4PZtqDY?3jS&gAm@%=S6B|>xODW+bdC@cXFw>`%I*S@3R<21UQa+MtE##EuvtuFfCf&eq!Ko&4RY*t6Eb@ljC5P#Frv!v+lCieDFMsQ;=cSYyNFDA4L&6;w4P~A7EoL~K7*^2C$4~$L0L7oC@Yl!d4ijxVPhWpVvrW4L>iRddyp9W&zZvRemSp#$gzENebd+zD2*=5_94Gk@Gr{BYq4<hLYL1NjqIbX_v@#`l0L&f=ShoESwJlw*eBeZ8r&lT8pEu(r~<xs#q^<nFehgpk9+AH<lLPaCt*r^g^?qULGf2y9T}oMQp2gcvS1Vt^{S-0@L0-5xqcYYNX$z*(C(qb#MVy(Y7-GW0O`?z+Eq2Y@eYIpi;<0M$$pMsp-tp))queC#fLa@6&2hGRFMEb+YD(LKZp11_gTMvxG0IBBk6oeP+|w**Gh=TjF-TFg#{TF*Xhxzle!~x4ij?l@xdtQwo^K9h=!00sm*t)d~QNTxb?YTM{*j3v@*u`k>x33MMbRJp5?(6S8o?6e5@?|CQt1H?Mif~wx0_h@pNS}vaFIiC<v_Y`2ZY^K9*C<>~DF%KBe}4@g}4a!N&m-<f3nZ<iQ9t<Xd?ycIr3XojND-oiAow?i7*!OJKy9yd#mQzGSp%dy7r$7~FnopRAvul18d9zKPW&AA$pU8#V69HuX)uMHD9$hd~}T?K}JmO8qb+*xGR^q7P+A>q>S|BKokPEykf;WNYedrpZlW7l1ZcxRBA1*T#9yus@!TSqPS)o#1f?lU$lK#GN6UAtCc+a)`>TxiX;4d#@)cv5|)WSr2D|q|RR~mP$+{8PQ!Q<4wm2{B{@C>YE=;v0;FXS{0RM1a}v`^dz%S<#t<>?j%H@Zm=z2%^OBhB<jm4z`#(=sJPde?x0KNBZ2R;%~=z@-U-rT8RY3}@~4%^H%aEGe=Li>`C%Rkp+O2MrBB#^QxteH5QJu6EWA;JO768Y6L$?;yOn<DXrM%<j~}_wZj)38QIO;E1!Zh$y66y(48((-TTQ&p%f(WGy+47;njIDZj|^muUD+e&0g9uvJTFcEVvKAg&AH=DhST~D0*(G{`#MMN(bPll+sy&hjM4TLAAG{VwU}Xc3;$DfiMu}wuL7+lLBxGibBma!u&qK6Af}IR%POQ}Kde=%$Q^92c3}ACX|VVN46pAN)t=q@UA+9I>d9pDbXCK=!uk$~+jUl0^IHk!7*dR^vCY5Jw{=ZI5XpOgqE2TAI6M!hPtpfUs3@4$y!2aQ1nM4q`)RjvkTeJ$p;el9Y(L)sM>M-QL8aXHq9<Y4i?Hrg`xYL>rL5Jmm&|HS(IhoRPBi1&V3-cy0k;rW6kwhOLC27O>6I8Y8%z`opDd$uO2OUICBB^T3wU2QBw5vQ1!{?lkqE)CIcuqE^$3&0Yng15+utMYrO^CL!2j~^#R`?(rI|65F=*?Kk_?YrlQ2KdolA35qb&S|7$mn*qtCRFihe^aa=9cSpfI;vjpi)pq(%LYYo3d1H<%D^x6FXLzkAQZY%!6M;*2f|5^hh_)8HVX0d+pz13af&*;50%*<@7O;I#iIai6eKD$TojdrW24jLkITUclAq2T5AY0J8(qn{k}Np&Do<!R+ts!zme%D`5u83nLm{_tFf@K!W2@>E%IMnUV0Mq$j6&ti<V%KB7|yqOdTqjN$UB!P`Ee7625-rrk#{zaf<??_G>u0VS-Bcrt5Xm5*Xu*^+aaprPG%i)KCzW%U!Y&5=3M2WV`4VhH7=6(7!>&dR!>orRgRu*_~XRtfnZH6R8mdehMI!XMhT5qzwWEWK<>D8}e%k`hs-mm1l!evzp!@~rpKS=%SuM1-Va@2G_atDSo=+=Ha|Wc_Xh4TB2~7Je4eqI=+aY4JIxjwzHy2bw`ZRfYz23>>~Wx?|T1XSpiKX#0FU)t<lH-KA|BPhjUSC%dda-@RHi26DA3B%3Kb$}dEiDfNKY(I#}8;Sy+FEEWOg-ZHk_G+VLePT=?QdIs91co{M{6b(^$Rex@lU?RQ-2J1mOdo8{`I+LW3SO77InMY5YANmtio@!ZGn>4?=1SY{G#N?mmw_Tk(@-WpLNtT>OAw`P(!bvU&N)Xq}0VUb(iz=M`m^|&wp8x*@Lk+-CA`U~SAW0rWklSl_%MzI4__~0N@`Zv3j4>Qv4v0om<(cjAxuox!^9l?89iIjFqk|f0u6!Sr{n(ks)hrtAC4wknqk7-r<N@n;L+~OiK`>!{q7|kvbET&D$d+yg!dmz1*N#wQkyk=M=G;=KYM<e_sro}3rn3dN`?3sObvDAjk@GX{%?0g08o0Z<x;)a2hmxltunNR<<2L_Sm>OX%e=6jDJ3&A^3&f5Mr{k%*c$=9tT_~e?-FY}eclk7*JVmOj+R^2DU?!}}&Y7qVr@7%On-U|l#Jf}-@dv?PO>gs3V)8Z!wInq@o38q#6)Yz&dPfxCTD#?;?;;}%rapgPtj)wVO)&0-B=y5GhH(p4lJ?>6-%YuNMvQt#4n#6W(&F+e(vC~{cUIw3M>%HV5}MV~8pW2ReXUzof<M(=sSdj(F)g7yEV{<?Y%bDfE&n&N7?&<6mP~{m2_W?Vwb6~<cv1}t57dnPLY1bOC;(=dGE+A1BI1e^xSs@Cc*c`HK!=Qnj$qjZU*V^Kzz?p}Yaob&fq>!vAJdN<0kb>F$CVd+iUgfH45hTYqJ4ZCXk!6RII@);u$HHw|MXR!*wHd$ozNwvN`>l@sZ-{!hx;qwtT%ITX@N}u!w{J9QH3XGqWK4XpQ3xMfJG@C>U_vMF?{wfPgMm5%RoX=MT6;Mj5}1Z*v>Fh-;TLAVMjE-KvJg(eWv~9Uw>H)VlVlslPo%1#<H<3f)e+_d+2~NiH)$-SMmkXVMKAju5FvxxZJ;342@zs5mV{_EfZ~JR75Eq6mIfo)~4+CcSsjvOtKNTMDAD(op$WlJ=A)FmB=I-^}{kP+yv+sD51K)rx<ah+69lr$w5ov_`KhyjPCDxABz5&vX*}vYH;6i950G#)+VPaqVXrZNaP(ap}2#j%dP{p9g~n%&oZf|aPA-TIlr2UduZ9%oG=Pr@$($Xy67X~dJzG8xh2%WC4qI?ZQi08RSFauJ;J^qiCZO8HYG+dQ%(|j7k)!isV=DI_xcE(L_<K#&G2nil}GziciTrAC7ZSC*eLLh5NE0>e^s?sr!%uZ&@TxQPEPHma-T3LEaJ?0h7^Cz#BYJ+phRUni(sj>z@H1a;gS2w{GJ(YU(u9K*;-#$1@xv;HdpGOKof%0scP*k=Cx?uq*9<YVDO?mBGOL--4(r$E2=cjE(Rov?6le)`X~lIPsc#X8%N)kH@GOLh#9x2^hCo<<9>8%1sQTz_`p@CI-mo`4-MWcl@Bz{Y7fjyZ-1_*iw~|~7z$quK#Arcp}5mP5hea>phm}V)TB;{FmIr6S~U6fhe2h|T#hbjGj*tXUKpF?xW`VzT1sdQpm@pJ2#RpL6RmK);oNP30>uHHWHq`bN*Cyxu8g`j$y4QZK{}^eMnBjA&R@>uYARp+?@7h@HXkql0^)6rfw^eMEwRt_CX@>VIPe~hQz=%QbF8}JD4miYHfI1zW!&xh`vZ}A@4GyzT-t}^3QRdcEhba0>a49QinPD8%T<Mq3bey!Q>vSM0L}=QjCC`6aUN-S!2vwMQ46|9J~3N{i6H(;Ygn|bx|VhK{`8gmWtfUVXU}3-3s{@QS-ul_@4?d1A%?0SkNWstV=0JE2n|NySc+L<UATz*6mijd{1*w{a{Y0rZj$eJp6IZfl+QITLW%v~UCblm-b2n)I4(ZWxDsYiNS7R33D0&|Ktw2?<0z7+r~{ix_jy7S{YscS_FK$w9uuPRT{HH2{f8<a=mk(*y#lHW0sFqp1vWpwn5gHN0uu!dK_~94bJ2j7TRiW@8#+#O`f8Y|bLs+;ef)OJH!TW3{90-{Q`v1kcL{$G$H`Y?zT)>&RY)v|Sp)(ej=cFEXN+^VT{VuYNaEb@d8B0~cxb0e4yi$OzJm<x<rb-mukeUJ*q)!QS~R0J(PrM{RDZpr5G3F!4BpVLA<2)QT8sgTI5=<q!pdzx^|LZ11!Ut#hWX$*V;bq*0pil(mR^SI5nK=r=gQrBGCm6qhbK)ZG3k!0GTVMRQ)WCa_CJ#|OF?JouUq|9EgjqGPw5jd05A!d-5X!IC(%rLbLD>w{PBw($24D`Ewz|NyB6g|lP%~B64#DK_9t&R=nx&nhD(JfJ_DncEZP-ZJ^Lc*qx1B?VU=VZbpF)Ahqdx2XW%zLBo+PvcclQ&@N6Xa7^crSgdDdrB?iZNQBx7)N#iJ5s2WOyp?flZ?xzD`3S922dLvU)yaH)l!cO?x_rJNC`Y)XRZU6uPV&IPeMjJG+00G-A!<+#Cx#AZKvBYQl0ssI200dcD")).decode("utf-8")


    def apply_unified(text, patch):
        """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
        else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored."""
        src = text.splitlines(True)
        hunks, cur = [], None
        for ln in patch.splitlines(True):
            if ln.startswith("@@"):
                m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
                if not m:
                    raise ValueError("bad hunk header: " + ln[:80])
                cur = [int(m.group(1)), int(m.group(2) or 1), []]
                hunks.append(cur)
            elif cur is not None and ln[:1] in (" ", "-", "+"):
                cur[2].append(ln)
            elif cur is not None and ln.startswith("\\"):
                raise ValueError("'No newline at end of file' hunks are not supported")
        if not hunks:
            raise ValueError("patch has no hunks")
        out, pos = [], 0
        for start, n_old, body in hunks:
            i = start - 1
            if i < pos:
                raise ValueError(f"overlapping hunk at line {start}")
            out += src[pos:i]
            old_seen = 0
            for ln in body:
                tag, s = ln[0], ln[1:]
                if tag in " -":
                    if i >= len(src) or src[i] != s:
                        raise ValueError(f"context mismatch at line {i + 1}: {s[:60]!r}")
                    i += 1
                    old_seen += 1
                if tag in " +":
                    out.append(s)
            if old_seen != n_old:
                raise ValueError(f"hunk at line {start}: {old_seen} old lines, header says {n_old}")
            pos = i
        return "".join(out + src[pos:])


    def apply_60490():
        """After merlin_install, before B1: patch the installed tree's qsa.py (pre-sha must be Merlin ad42e37's), verify the
        post-sha, the marker, a py_compile with the Merlin venv and that `vllm` resolves to the tree; write the read-only
        worker extension next to the tree's vllm/ (on the server's PYTHONPATH). Any miss raises: no boot is started."""
        if FAKE:
            tree = os.environ.get("NB1_FAKE_TREE")
            if not tree:
                raise RuntimeError("PATCH_60490_NO_TREE: NB1_FAKE_TREE is not set (local harness)")
            tree, py = Path(tree), sys.executable
        else:
            eng = merlin_install()
            tree, py = Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"
        f = tree / P60490_REL
        raw = f.read_bytes()
        h0 = hashlib.sha256(raw).hexdigest()
        rec = {"file": str(f), "sha_before": h0, "pre_expected": P60490_PRE, "post_expected": P60490_POST}
        if h0 == P60490_POST:
            rec["applied"] = "already_applied"
        elif h0 != P60490_PRE:
            raise RuntimeError(f"PATCH_60490_PRE_SHA_MISMATCH: {h0} (expected Merlin ad42e37 {P60490_PRE})")
        else:
            new = apply_unified(raw.decode("utf-8"), P60490).encode("utf-8")
            if hashlib.sha256(new).hexdigest() != P60490_POST:
                raise RuntimeError("PATCH_60490_POST_SHA_MISMATCH: " + hashlib.sha256(new).hexdigest())
            tmp = f.with_name(f.name + ".nb1tmp")
            tmp.write_bytes(new)
            os.replace(tmp, f)
            rec["applied"] = "applied"
        for pyc in (f.parent / "__pycache__").glob("qsa.*.pyc"):
            pyc.unlink()
        data = f.read_bytes()
        rec.update(sha_after=hashlib.sha256(data).hexdigest(), marker=P60490_MARKER.encode() in data)
        ext = tree / "nb1_worker_ext.py"
        ext.write_text(NB1_EXT_SRC)
        rec["worker_ext"] = {"file": str(ext), "sha256": hashlib.sha256(ext.read_bytes()).hexdigest()}
        cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)", str(f), str(ext)],
                            capture_output=True, text=True, timeout=300)
        rec["py_compile"] = {"rc": cp.returncode, "err": cp.stderr[-400:]}
        origin_ok = True
        if not FAKE:
            envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            envp["PYTHONPATH"] = str(tree)            # what Merlin's build() exports for the server
            cp2 = subprocess.run([py, "-c", "import importlib.util as u; print(u.find_spec('vllm').origin); print(u.find_spec('nb1_worker_ext').origin)"],
                                 capture_output=True, text=True, timeout=300, env=envp)
            rec["vllm_origin"] = cp2.stdout.strip()[-400:]
            origin_ok = all(l.startswith(str(tree)) for l in cp2.stdout.strip().splitlines()[-2:]) and len(cp2.stdout.strip().splitlines()) >= 2
        if rec["sha_after"] != P60490_POST or not rec["marker"] or cp.returncode or not origin_ok:
            raise RuntimeError("PATCH_60490_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("sha_after", "marker", "py_compile", "vllm_origin")})[:600])
        rec["ok"] = True
        rec["forward_test"] = "B1's greedy oracle vs RC2's recorded FP8 rows (unpatched RC2 FP8; B1 adds text-only): B1 oracle.vs_rc2_fp8"
        return rec


    def cgroup_snapshot(tag):
        """memory.current / stat / events / pressure (first 12 kB each) to hostmem/<tag>.txt; key numbers returned."""
        rec, txt = {"tag": tag}, []
        for n in ("memory.current", "memory.stat", "memory.events", "memory.pressure"):
            try:
                s = (Path("/sys/fs/cgroup") / n).read_text()[:12000]
            except Exception:
                continue
            txt.append(f"--- {n} ---\n{s}")
            if n == "memory.current":
                rec["current_mib"] = int(s.strip()) >> 20
            elif n == "memory.events":
                rec["events"] = {a: int(b) for a, b in (l.split() for l in s.splitlines() if len(l.split()) == 2)}
            elif n == "memory.pressure":
                rec["pressure"] = s.strip().splitlines()[:2]
        try:
            (HOSTMEM / f"{tag}.txt").write_text("\n".join(txt))
        except Exception:
            pass
        rec["host_mem"] = host_mem()
        return rec


    ENGINE_RX = re.compile(r"vllm|PleOffload|ple_offload|EngineCore|merlin_off/venv", re.I)


    def engine_procs():
        """Processes of an engine (vLLM server, workers, PLE offload) other than this notebook; None off Linux."""
        if not Path("/proc/self/stat").exists():
            return None
        me, out = os.getpid(), []
        for d in Path("/proc").glob("[0-9]*"):
            try:
                pid = int(d.name)
                if pid == me:
                    continue
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if ENGINE_RX.search(cmd) and "py_compile" not in cmd:
                    out.append([pid, cmd[:160]])
            except Exception:
                pass
        return out


    def clear_engine_procs():
        left0 = engine_procs()
        rec = {"found": left0}
        if left0:
            for pid, _ in left0:
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
            time.sleep(10)
            rec["after_kill"] = engine_procs()
        rec["ok"] = not (rec.get("after_kill") if left0 else left0)
        return rec


    def drop_page_cache():
        """posix_fadvise(DONTNEED) on every checkpoint file (merged dir symlinks resolved + the Model mounts); only after the
        previous server is dead (never on the pinned PLE pages of a live server). A measured attempt, not an assumed fix."""
        if not hasattr(os, "posix_fadvise"):
            return {"available": False}
        t, seen, n, b, err = time.time(), set(), 0, 0, 0
        roots = [CKPT] + [v[0] for v in PARTS.values() if v] + ([PLE_DIR] if PLE_DIR else [])
        for root in roots:
            for dp, _, fns in os.walk(str(root)):
                for x in fns:
                    p = os.path.realpath(os.path.join(dp, x))
                    if p in seen:
                        continue
                    seen.add(p)
                    try:
                        fd = os.open(p, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                            b += os.fstat(fd).st_size
                            n += 1
                        finally:
                            os.close(fd)
                    except OSError:
                        err += 1
        return {"available": True, "files": n, "gib": round(b / 2 ** 30, 1), "errors": err, "sec": round(time.time() - t, 1)}


    def host_ram_gate(tag):
        before = cgroup_snapshot(tag + "_before_drop")
        drops, tries = [drop_page_cache()], 0
        after = host_mem()
        while True:
            av = after.get("avail_mib", after.get("MemAvailable_mib"))
            if av is None:
                ok = bool(FAKE)               # local harness only; on Kaggle an unreadable host is never "enough"
                why = "no cgroup / meminfo reading" + (" (local harness: skipped)" if FAKE else ": HOST_RAM_GATE fails closed")
                break
            if av >= RAM_NEED_MIB:
                ok, why = True, f"avail {av} MiB >= {RAM_NEED_MIB}"
                break
            if FAKE:                          # local / CPU-kit harness only (a 30 GiB CPU box): recorded, never enforced
                ok, why = True, f"avail {av} MiB < {RAM_NEED_MIB} MiB (local harness: recorded, not enforced)"
                break
            tries += 1
            if tries > 3:
                ok, why = False, f"HOST_RAM_GATE: avail {av} MiB < {RAM_NEED_MIB} MiB after {tries} page-cache drops"
                break
            time.sleep(20)
            drops.append(drop_page_cache())
            after = host_mem()
        return {"ok": ok, "why": why, "need_mib": RAM_NEED_MIB, "before": before, "after_drop": cgroup_snapshot(tag + "_after_drop"),
                "drops": drops}


    _boot_gate_rc2, _shm_cleanup_rc2 = boot_gate, shm_cleanup


    BOOT = {"deadline": None, "fired": None, "tag": "boot"}


    def boot_gate():
        """RC2's GPU release gate (< 150 MiB per card), the previous engine's processes gone, then the host-RAM gate."""
        pre = clear_engine_procs()                     # E25 klee100 R1-6: before the release wait, not only after it
        g = _boot_gate_rc2()
        g["engine_procs_before_wait"] = pre
        if g.get("ok") is not True:
            return g
        g["engine_procs"] = clear_engine_procs()
        if not g["engine_procs"]["ok"]:
            g["ok"], g["why"] = False, "ENGINE_PROCS_LEFT: " + json.dumps(g["engine_procs"].get("after_kill"))[:300]
            return g
        h = host_ram_gate(BOOT["tag"])
        g["host_ram"] = h
        if not h["ok"]:
            g["ok"], g["why"] = False, h["why"]
        return g


    def shm_cleanup(before):
        """RC2's /dev/shm cleanup (this boot's torch_/psm_/vllm/ple/nccl/sem.mp-/cuda.shm entries) + the loky semaphores the
        RC2 log reported leaked at shutdown (sem.loky-*), only when created during this boot."""
        out = _shm_cleanup_rc2(before)
        out["removed_loky"] = []
        try:
            for p in sorted(SHM.iterdir()):
                if p.name not in before and re.match(r"sem\.loky-", p.name):
                    try:
                        p.unlink()
                        out["removed_loky"].append(p.name)
                    except Exception:
                        pass
        except Exception:
            pass
        return out


    def boot_left():
        d = BOOT["deadline"]
        return min(left(), (d - time.time()) if d else float("inf"))


    def due(sec):
        return boot_left() > sec and not ABORT["why"]


    def watched(body, cap_sec, name):
        """The boot body gets `cap_sec`; every step checks due(); at cap + 300 s the server is killed (hard backstop)."""
        def b(ss):
            BOOT.update(deadline=time.time() + cap_sec, fired=None)

            def fire():
                BOOT["fired"] = name
                for x in ss:
                    try:
                        x.kill()
                    except Exception:
                        pass
            tm = threading.Timer(cap_sec + 300, fire)
            tm.daemon = True
            tm.start()
            try:
                o = body(ss)
                LIVE["out"] = o if isinstance(o, dict) else None   # R2: step() keeps it if anything after the body raises
            finally:
                tm.cancel()
                BOOT["deadline"] = None
            if BOOT["fired"]:
                o["boot_watchdog"] = f"FIRED after {cap_sec + 300:.0f} s: server killed"
            return o
        return b


    def _pick(t, rx, n=6):
        return [l.strip()[:300] for l in t.splitlines() if re.search(rx, l)][:n]


    def server_caps(t, argv=()):
        na = next((l for l in t.splitlines() if "non-default args:" in l), "")
        caps = {"source": "log" if na else "argv"}
        for k in ("max_num_seqs", "max_num_batched_tokens", "block_size", "max_model_len", "kv_cache_memory_bytes", "num_speculative_tokens"):
            m = re.search(rf"'{k}': (\d+)", na)
            caps[k] = int(m.group(1)) if m else None
        for k in ("kv_cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode", "worker_extension_cls"):
            m = re.search(rf"'{k}': '([^']+)'", na)
            caps[k] = m.group(1) if m else None
        m = re.search(r"'language_model_only': (True|False)", na)
        caps["language_model_only"] = m.group(1) == "True" if m else False
        m = re.search(r"'cudagraph_capture_sizes': \[([\d, ]+)\]", na)
        caps["capture_sizes"] = [int(x) for x in m.group(1).split(",")] if m else None
        m = re.findall(r"Setting attention block size to (\d+) tokens", t)
        if m:
            caps["block_size"] = int(m[-1])
        if not na:                                     # fallback (the E23 fake has no args line): the argv, last wins
            a = [str(x) for x in argv]
            for flag, key in (("--max-num-seqs", "max_num_seqs"), ("--block-size", "block_size")):
                idx = [i for i, x in enumerate(a) if x == flag]
                if idx and idx[-1] + 1 < len(a):
                    caps[key] = int(a[idx[-1] + 1])
        caps["block_size"] = caps.get("block_size") or BLOCK
        sizes = caps.get("capture_sizes") or []
        k = caps.get("num_speculative_tokens") or 3
        caps["captures_cover_seqs_x_1pk"] = bool(sizes) and caps.get("max_num_seqs") is not None and max(sizes) >= caps["max_num_seqs"] * (1 + k)
        return caps


    def pool_facts(t):
        f = {}
        m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", t)
        f["kv_tokens"] = int(m[-1].replace(",", "")) if m else None
        m = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", t)
        f["max_concurrency"] = float(m[-1][1]) if m else None
        f["max_concurrency_tokens"] = int(m[-1][0].replace(",", "")) if m else None     # R2: M of the line (F_log)
        m = re.findall(r"Padding mamba page size by ([\d.]+)%", t)
        f["mamba_pad_pct"] = float(m[-1]) if m else None
        f["min_block_for_mamba"] = math.ceil(BLOCK / (1 + f["mamba_pad_pct"] / 100)) if m else None
        f["kv_group_sizes"] = (re.findall(r"kv cache group sizes \(([^)]*)\)", t) or [None])[-1]
        f["side_cache_layout"] = _pick(t, r"Side-cache layout", 2)
        f["initial_free_reserved_gib"] = [[float(a), float(b)] for a, b in re.findall(r"Initial free memory ([\d.]+) GiB, reserved ([\d.]+) GiB", t)][:4]
        f["model_load_gib_sec"] = [[float(a), float(b)] for a, b in re.findall(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", t)][:4]
        f["graph_capture_sec_gib"] = [[int(a), float(b)] for a, b in re.findall(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", t)][:4]
        f["rank_load_gib"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Model loading took ([\d.]+) GiB memory", t)}
        f["rank_load_sec"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Model loading took [\d.]+ GiB memory and ([\d.]+) seconds", t)}
        f["rank_graph_gib"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Graph capturing finished in \d+ secs, took ([\d.]+) GiB", t)}
        f["ssm_dtype_lines"] = _pick(t, r"mamba_ssm_dtype|mamba-ssm-cache-dtype")
        f["gdn_fallback_lines"] = _pick(t, r"Falling back to the Triton GDN spec decode path")
        f["share_workspace_lines"] = _pick(t, r"MERLIN_SHARE_DRAFT_WORKSPACE")
        f["mm_warmup_lines"] = _pick(t, r"Multi-modal warmup completed")
        f["host_ram_lines"] = _pick(t, r"host memory available now|Cgroup memory preflight|Skipping auto-prefetch", 6)
        f["sched_warn_lines"] = _pick(t, r"max_num_scheduled_tokens", 2)
        f["oom_lines"] = _pick(t, r"OutOfMemoryError|CUDA out of memory", 4)
        f["static_scale_lines"] = _pick(t, r"applied static K/V scales", 8)
        f["default_scale_lines"] = _pick(t, r"DEFAULT 1\.0|no static calibration", 4)    # audit 2: the MTP layer's QSA FP8 KV
        f["ext_lines"] = _pick(t, r"Injected .*NB1WorkerExt|nb1_worker_ext", 2)
        f["shm_leak_lines"] = _pick(t, r"leaked (semaphore|shared_memory) objects", 4)
        return f


    def graph_facts(t):
        caps = sorted({(a, int(c)) for a, b, c in re.findall(r"Capturing CUDA graphs \((\w+)\):\s*100%\|[^|]*\|\s*(\d+)/(\d+)", t)})
        return {"captured": caps, "resolved_modes": _pick(t, r"resolved cudagraph modes before capture", 8),
                "draft_identity": _pick(t, r"e2-identity\[.*role=draft", 4), "recoverssm_commit": _pick(t, r"\[recoverssm-commit\]"),
                "recoverssm_path": _pick(t, r"RecoverSSM path taken", 4), "jit": jit_kernels(t)}


    def jit_kernels(t):
        return re.findall(r"JIT compilation during inference: (\w+)", t)


    def rpc_cache_info(port):
        """The read-only worker extension over /collective_rpc (dev mode): actual cache tensor dtypes, specs, free memory."""
        body = {"method": "nb1_cache_info", "timeout": 120}
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=180).read())
            res = r.get("results") or []
        except Exception as e:
            return {"error": repr(e)[:300]}
        out = {"ranks": res}
        dts = sorted({x["dtype"] for w in res if isinstance(w, dict) for g in (w.get("cache_tensors") or [])
                      if "mamba" in g.get("module", "").lower() or "gdn" in g.get("module", "").lower() or "linear" in g.get("module", "").lower()
                      for x in g.get("tensors") or []})
        out["recurrent_tensor_dtypes"] = dts
        out["num_gpu_blocks"] = sorted({w.get("num_gpu_blocks") for w in res if isinstance(w, dict)} - {None})
        out["free_mib_per_rank"] = [((w or {}).get("mem") or {}).get("free_mib") for w in res if isinstance(w, dict)]
        pads = []
        for w in res:
            for s in ((w or {}).get("kv_cache_specs") or []) if isinstance(w, dict) else []:
                if s.get("unpadded_page_bytes") and s.get("page_size_bytes"):
                    pads.append({"rank": w.get("rank"), "type": s.get("type"), "layers": s.get("layers"), "state_dtypes": s.get("state_dtypes"),
                                 "padded": s["page_size_bytes"], "unpadded": s["unpadded_page_bytes"],
                                 "pad_pct": round(100 * (s["page_size_bytes"] - s["unpadded_page_bytes"]) / s["unpadded_page_bytes"], 2)})
        out["mamba_padding"] = pads[:16]
        return out


    def rpc(port, method, *args, timeout=120):
        """POST /collective_rpc (dev mode): a worker method by name with string args; the per-rank results or the error."""
        body = {"method": method, "args": [str(a) for a in args], "timeout": timeout}
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout + 60).read())
            return {"results": r.get("results")}
        except Exception as e:
            return {"error": repr(e)[:300]}


    def free_per_card(rec, name):
        """MiB free per card = total - the highest used reading of this boot (after ready, end, every window peak)."""
        srv = ((rec or {}).get("servers") or {}).get(name) or {}
        rows = [r for r in (srv.get("gpu_mem_after_ready"), srv.get("gpu_mem_end")) if r]
        if not rows:
            return None
        tot = {i: t for row in rows for i, _, t in row}
        used = {i: max(u for row in rows for j, u, _ in row if j == i) for i in tot}
        st = (rec or {}).get("steady")             # R2: nb1_ladder's steady holds "_plan" (list), "_fit" (dict), "_seqs_cap" (int)
        ws = list(st.values()) if isinstance(st, dict) else []
        for w in ws + [(rec or {}).get("profile")]:
            if not isinstance(w, dict):            # R2: only window dicts carry gpu_mem_peak_mib
                continue
            pk = w.get("gpu_mem_peak_mib")
            for i, u in enumerate(pk if isinstance(pk, list) else []):
                if i in used and isinstance(u, (int, float)):
                    used[i] = max(used[i], u)
        return [tot[i] - used[i] for i in sorted(tot)]


    def free_after_ready(rec, name):
        srv = ((rec or {}).get("servers") or {}).get(name) or {}
        row = srv.get("gpu_mem_after_ready")
        return [t - u for _, u, t in row] if row else None


    def kvu(port):
        return L.prom(port).get(L.KVU)


    def pool_blocks_from_usage(us):
        vals = sorted({round(u, 9) for u in us if isinstance(u, (int, float)) and 0 < u < 1})
        if len(vals) < 2:
            return None
        for p in range(20, 3001):
            if all(abs(u * p - round(u * p)) < 0.02 for u in vals):
                return p
        return None


    def reset_cache(port):
        """POST /reset_prefix_cache (dev mode): the 'caches cleared' state of the audits; fails while requests run."""
        try:
            r = L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST"), timeout=120).read()
            return {"ok": True, "body": r[:200].decode(errors="replace")}
        except Exception as e:
            return {"ok": False, "error": repr(e)[:200]}


    def _hits(port):
        try:
            return L.prom(port).get("vllm:prefix_cache_hits_total", 0.0)
        except Exception:
            return None


    def n_fit(P, F, B, band, idle_blocks=0, seqs=None, n_prompts=None):
        """Streams of `band` whose final length (prompt + ~(admission + window + 60 s) x GEN_RATE) all fit in the pool."""
        if P is None or F is None:
            return None
        L0 = BANDS[band]
        final = L0 + (L0 * 8 / max(1.0, PREFILL["tok_s"]) + WIN + 60) * GEN_RATE
        n = int((P - idle_blocks) // (math.ceil(final / B) + F))
        for cap in (seqs, n_prompts):
            if cap:
                n = min(n, cap)
        return max(0, n)


    def admission_test(port, s, band="90k", n=5, max_new=4096):
        """Fresh-cache admission of n distinct long prompts (audit 2: 90k x 5 before any pin change). Reset the cache, start
        n streams, wait until every one has its first token (admission + cold prefill, ~450k tokens: Merlin's cost-aware
        admission may hold a cold miss up to 120 s), then: a resident snapshot (running == n, waiting 0, no preemption, blocks)
        and a short decode window measured on the counters while all n stay resident. max_new 4096 keeps 5 x (ceil(94,096 /
        4096) + 6) = 145 of 149 blocks: the streams cannot outgrow the pool (a lab steady window cannot be used: the first
        stream would finish its max_new inside it, since the cold prefills are staggered by ~20 s each)."""
        if not due(max(240 + 150 + n * BANDS[band] / max(2000.0, PREFILL["tok_s"]),          # E25 klee100 R1-16
                       max(420.0, 180.0 + n * BANDS[band] / max(1000.0, PREFILL["tok_s"])) + 60)):   # E25 klee100 R2-2
            return {"status": "SKIPPED_BOOT_BUDGET"}
        if not L.wait_idle(port, 120):
            return {"status": "INCONCLUSIVE_NOT_IDLE"}
        rec = {"band": band, "n": n, "max_new": max_new, "reset": reset_cache(port)}
        time.sleep(1)
        rec["idle_usage"] = kvu(port)
        m0, l0, t0 = L.prom(port), _req_lines(), time.time()
        ss = [L.Stream(port, PROMPTS[band][i], max_new, temperature=TEMP, seed=500 + i, ignore_eos=True) for i in range(n)]
        for g in ss:
            g.start()
        try:
            limit = max(420.0, 180.0 + n * BANDS[band] / max(1000.0, PREFILL["tok_s"]))
            while time.time() - t0 < limit:
                time.sleep(2)
                if all(g.t_first for g in ss) or any(not g.is_alive() for g in ss):
                    break
            m1 = L.prom(port)
            rec.update(admission_sec=round(time.time() - t0, 1), first_token=[round(g.t_first - g.t_start, 1) if g.t_first else None for g in ss],
                       snapshot={"running": m1.get(L.RUN), "waiting": m1.get(L.WAIT), "usage": m1.get(L.KVU),
                                 "tokens": [len(PROMPTS[band][i]) + len(g.tokens) for i, g in enumerate(ss)]},
                       preemptions=m1.get(L.PREEMPT, 0) - m0.get(L.PREEMPT, 0),
                       prefill_uncached_tok_s=round(((m1.get(L.PROMPT, 0) - m0.get(L.PROMPT, 0)) - (m1.get("vllm:prefix_cache_hits_total", 0)
                                                     - m0.get("vllm:prefix_cache_hits_total", 0))) / max(1e-3, time.time() - t0), 1))
            rec["all_resident"] = (all(g.t_first for g in ss) and rec["snapshot"]["running"] == n and not rec["snapshot"]["waiting"]
                                   and not rec["preemptions"])
            if rec["all_resident"]:
                secs = min(20.0, WIN)
                g0, w0, runs = L.prom(port), time.time(), []
                while time.time() - w0 < secs:
                    time.sleep(min(2.0, secs))
                    runs.append(L.prom(port).get(L.RUN))
                g1 = L.prom(port)
                dt = time.time() - w0
                steps = g1.get(L.ITER, 0) - g0.get(L.ITER, 0)
                rec["window"] = {"sec": round(dt, 1), "tok_s": round((g1.get(L.GEN, 0) - g0.get(L.GEN, 0)) / dt, 1), "running_min": min(runs or [0]),
                                 "running_max": max(runs or [0]), "tick_ms": round(dt * 1000 / steps, 1) if steps else None,
                                 "preemptions": g1.get(L.PREEMPT, 0) - g0.get(L.PREEMPT, 0)}
                if rec["window"]["tick_ms"]:
                    rec["window"]["cycle_ms"] = round(PP * rec["window"]["tick_ms"], 1)
                rec["status"] = ("ALL_RESIDENT" if rec["window"]["running_min"] == n and not rec["window"]["preemptions"]
                                 else "INCONCLUSIVE_RUNNING_CHANGED")
            else:
                rec["status"] = "NOT_ALL_RESIDENT"
        finally:
            for g in ss:
                g.stop()
            for g in ss:
                g.join(10)
            rec["stream_errors"] = sorted({g.err for g in ss if g.err})[:3]
        rec["ttft_per_request"] = _ttfts(port, l0)
        return rec


    def residency_bound(P, F, band, max_new=0, seqs=None):
        """Arithmetic bound: running requests of total context band + max_new that fit P blocks at F fixed blocks each."""
        if not P or F is None:
            return None
        n = int(P // (math.ceil((BANDS[band] + max_new) / BLOCK) + F))
        return min(n, seqs) if seqs else n


    F_MAX_BLOCKS = 40


    def _f_valid(F):
        return isinstance(F, (int, float)) and not isinstance(F, bool) and math.isfinite(F) and 0 <= F <= F_MAX_BLOCKS


    def f_from_log(o, P):
        """R2: F_log = P / X - ceil(M / B) from this boot's line 'GPU KV cache size: N tokens, Maximum concurrency for M tokens
        per request: X x' (X = N / M when N is there: the printed X has 2 decimals); F = ceil(F_log - 0.05) like
        nb5_lib.capacity_plan (RC2 / B1: 149 x 131,072 / 513,940 - 32 = 5.999 -> 6)."""
        pool = o.get("pool") or {}
        N, X, M = pool.get("kv_tokens"), pool.get("max_concurrency"), pool.get("max_concurrency_tokens")
        B = (o.get("caps") or {}).get("block_size") or BLOCK
        if not (isinstance(M, int) and M > 0 and ((isinstance(N, int) and N > 0) or (isinstance(X, (int, float)) and X > 0))):
            return {"ok": False, "why": f"no max-concurrency line in this boot's log (N {N}, M {M}, X {X})"}
        xx = N / M if isinstance(N, int) and N > 0 else float(X)
        f_log = P / xx - math.ceil(M / B)
        F = max(0, int(math.ceil(f_log - 0.05)))
        ok = math.isfinite(f_log) and -0.05 <= f_log <= F_MAX_BLOCKS
        return {"ok": ok, "F": F if ok else None, "F_log": round(f_log, 3), "kv_tokens": N, "max_concurrency": X, "tokens_per_request": M,
                "block_size": B, "x_used": round(xx, 5), "why": None if ok else f"F_log {f_log:.3f} outside 0..{F_MAX_BLOCKS}"}


    def nb1_complete(port, ids, max_new, temperature=0.0, seed=None, logprobs=0, timeout=1800):
        """Non-streaming /v1/completions with token ids: ids, usage, cached tokens (--enable-prompt-tokens-details) and, with
        logprobs, the chosen-token logprobs + top-1/top-2 margins. Every request is saved in traces/requests.jsonl."""
        body = {"model": "q27", "prompt": list(ids), "max_tokens": int(max_new), "temperature": temperature,
                "top_p": 0.95 if temperature else 1.0, "ignore_eos": True, "return_token_ids": True}
        if temperature:
            body["top_k"] = 20
        if seed is not None:
            body["seed"] = int(seed)
        if logprobs:
            body["logprobs"] = int(logprobs)
            body["return_tokens_as_token_ids"] = True
        t = time.time()
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout).read())
            ch = r["choices"][0]
            u = r.get("usage") or {}
            out = {"ids": list(ch.get("token_ids") or []), "text": ch.get("text") or "", "usage": u, "error": None,
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
            lp = ch.get("logprobs") or {}
            if logprobs:
                out["lps"] = list(lp.get("token_logprobs") or [])
                out["margins"] = []
                for top in lp.get("top_logprobs") or []:
                    v = sorted((x for x in (top or {}).values() if isinstance(x, (int, float))), reverse=True)
                    out["margins"].append(round(v[0] - v[1], 4) if len(v) >= 2 else None)
        except Exception as e:
            out = {"ids": [], "text": "", "usage": None, "cached": None, "error": repr(e)[:300]}
        out["sec"] = round(time.time() - t, 2)
        try:
            with _SAVE_LOCK, open(REQ_LOG, "a") as fh:
                fh.write(json.dumps({"kind": "nb1_complete", "port": port, "prompt_len": len(ids), "max_new": max_new,
                                     "prompt_sha": psha(ids)[:16], "text": out["text"], "tokens": out["ids"], "lps": out.get("lps"),
                                     "usage": out["usage"], "error": out["error"], "sec": out["sec"], "t": time.time()}) + "\n")
        except Exception:
            pass
        return out


    def lp_compare(a, b):
        """Greedy rows a (reference) vs b: first differing token, the reference's top-1 - top-2 margin there (L.FAIL_MARGIN),
        mean / max |delta chosen-token logprob| over the identical prefix."""
        if not a or not b or a.get("error") or b.get("error") or not a.get("ids") or not b.get("ids"):
            return {"status": "INCONCLUSIVE"}
        x, y = a["ids"], b["ids"]
        d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        la, lb = a.get("lps") or [], b.get("lps") or []
        k = d if d is not None else min(len(x), len(y))
        dl = [abs(p - q) for p, q in zip(la[:k], lb[:k]) if isinstance(p, (int, float)) and isinstance(q, (int, float))]
        if d is None:
            st, m = ("EXACT" if len(x) == len(y) else "INCONCLUSIVE"), None
        else:
            mm = a.get("margins") or []
            m = mm[d] if d < len(mm) else None
            st = "INCONCLUSIVE" if m is None else ("FAIL_HIGH_MARGIN" if m >= L.FAIL_MARGIN else "NEAR_TIE")
        return {"status": st, "first_diff": d, "ref_margin": m, "mean_abs_dlogprob_prefix": round(statistics.mean(dl), 5) if dl else None,
                "max_abs_dlogprob_prefix": round(max(dl), 5) if dl else None, "tokens": [len(x), len(y)],
                "cached": [a.get("cached"), b.get("cached")]}


    def warmup(port, n, secs=None):
        """Multi-stream 2k decode, ramped 1 -> n streams, excluded from every metric: compiles the kernels RC2 saw compiled
        during inference (_fill_num_accepted_kernel, _causal_conv1d_update_kernel, _gdn_recoverssm_verify_kernel,
        _ple_conv_kernel, _ple_conv_writeback_kernel) before any measured window."""
        secs = secs or (45.0 if not FAKE else 4.0)
        srv = next((x for x in list(L.LIVE) if x.port == port), None)
        j0 = jit_kernels(srv.log()) if srv else []
        ss, t0 = [], time.time()
        try:
            for i in range(n):
                g = L.Stream(port, PROMPTS["2k"][i % len(PROMPTS["2k"])], 4096, temperature=TEMP, seed=900 + i, ignore_eos=True)
                g.start()
                ss.append(g)
                time.sleep(min(3.0, secs / max(1, 2 * n)))
            while time.time() - t0 < secs:
                time.sleep(1)
        finally:
            for g in ss:
                g.stop()
            for g in ss:
                g.join(10)
        L.wait_idle(port, 60)
        j1 = jit_kernels(srv.log()) if srv else []
        return {"streams": n, "sec": round(time.time() - t0, 1), "jit_before": j0, "jit_compiled_in_warmup": j1[len(j0):],
                "errors": sorted({g.err for g in ss if g.err})[:3]}


    def accept_pos(port):
        try:
            txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
        except Exception:
            return None
        pos = {}
        for m in re.finditer(r"^vllm:spec_decode_num_accepted_tokens_per_pos_total\{([^}]*)\}\s+(\S+)", txt, re.M):
            p = re.search(r'position="(\d+)"', m.group(1))
            if p:
                pos[int(p.group(1))] = pos.get(int(p.group(1)), 0.0) + float(m.group(2))
        return {"pos": pos, "drafts": L.prom(port).get(L.DRF, 0.0)}


    def _prefill_done(r, n):
        tail = ((r.get("admission") or {}).get("tail") or [])
        t = tail[-1] if tail else None
        return bool(t) and t[1] == n and t[2] == 0 and t[3] >= n


    def _req_lines():
        try:
            with open(REQ_LOG) as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 0


    def _ttfts(port, start_line):
        out = []
        try:
            with open(REQ_LOG) as fh:
                for i, l in enumerate(fh):
                    if i >= start_line:
                        r = json.loads(l)
                        if r.get("port") == port and r.get("ttft") is not None:
                            out.append(r["ttft"])
        except Exception:
            pass
        return out


    def _counters(port):
        try:
            m = L.prom(port)
            return m.get(L.PROMPT, 0.0), m.get("vllm:prefix_cache_hits_total", 0.0)
        except Exception:
            return None, None


    def window(port, s, band, n, max_new=None, secs=None):
        """One steady window, timed only after every stream finished admission + prefill (the lab's settle needs all running
        streams to have emitted and be gaining, then an explicit check of the last admission sample; Merlin's cost-aware
        admission may hold a cold miss up to 120 s, inside the lab's >= 420 s admission limit); a window with prefill inside is
        retried once. Recorded: tick_ms (one engine step = one micro-batch of C/2 requests under PP2) AND cycle_ms (= PP x
        tick, the per-request advance time), per-request TTFT, uncached prefill tok/s (prompt minus prefix-hit tokens),
        retained (idle) blocks before the window, per-position acceptance."""
        retained_before = None
        if L.wait_idle(port, 60):
            retained_before = kvu(port)
        def run():
            if max_new is None:                 # RC2 / E23 levels(): MAX_NEW[band], WIN, the budget / KV-bound guards
                return levels(port, band, [n], server=s, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"]).get(f"{band}@{n}") or {}
            if not due((secs or WIN) + 210 + n * BANDS[band] / 1500):        # E25 klee100 R1-1
                return {"status": "SKIPPED_BOOT_BUDGET", "boot_left_sec": round(boot_left())}
            return L.steady(port, PROMPTS[band], n, secs=secs or WIN, max_new=max_new, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"])
        a0, c0, l0 = accept_pos(port), _counters(port), _req_lines()
        r = run()
        if r.get("status") == "INCONCLUSIVE_PREFILL_IN_WINDOW" and due(WIN + 120 + n * BANDS[band] / max(500.0, PREFILL["tok_s"])):
            c0, l0 = _counters(port), _req_lines()
            r = dict(run(), retried_after="INCONCLUSIVE_PREFILL_IN_WINDOW")
        a1, c1 = accept_pos(port), _counters(port)
        r["prefill_done_before_window"] = _prefill_done(r, n)
        r["running_eq_level"] = r.get("running_min") == r.get("running_max") == n
        if r.get("status") == "STEADY" and not (r["running_eq_level"] and r["prefill_done_before_window"]):
            r["status"] = "INCONCLUSIVE_RUNNING_NE_LEVEL" if not r["running_eq_level"] else "INCONCLUSIVE_PREFILL_NOT_DONE"
        if r.get("step_ms"):
            r["tick_ms"] = r["step_ms"]
            r["cycle_ms"] = round(PP * r["step_ms"], 1)
            r["requests_per_tick"] = round(n / PP, 2)
            r["microbatches_per_s"] = round(1000.0 / r["step_ms"], 2)
            r["wall_clock_tok_s"] = r.get("tok_s")
            r["tick_note"] = ("tick_ms = step_ms = one PP tick (one completed micro-batch of C/2 requests); cycle_ms = 2 ticks = the "
                              "per-request advance time; microbatches_per_s = completed micro-batches per second; wall_clock_tok_s = "
                              "generated tokens / window seconds")
        adm = (r.get("admission") or {}).get("sec")
        if None not in (c0[0], c1[0]) and adm:
            unc = (c1[0] - c0[0]) - (c1[1] - c0[1])
            r["prefill_uncached_tokens"] = int(unc)
            r["prefill_tok_s_uncached"] = round(unc / adm, 1)
            r["prefill_note"] = "admission.prefill_tok_s counts cached prompt tokens; prefill_tok_s_uncached does not (incl. a retry)"
        r["ttft_per_request"] = _ttfts(port, l0)
        if retained_before is not None:
            r["retained_usage_before"] = retained_before
        if a0 and a1 and a1["drafts"] > a0["drafts"]:
            r["accept_rate_per_position_incl_admission"] = {p: round((a1["pos"].get(p, 0) - a0["pos"].get(p, 0)) / (a1["drafts"] - a0["drafts"]), 3)
                                                            for p in sorted(a1["pos"])}
        return r


    def free_valid(fr):
        """R2 audit N1-3: a valid per-rank free memory reading = 4 finite numbers in 0..24,576 MiB (one L4 card)."""
        return (isinstance(fr, list) and len(fr) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                                                              and 0 <= x <= 24576 for x in fr))
    save()


if not RERUN:
    # ==================================================================== CELL 7 — E25 klee100: boots, pin ladder, TF vs W4, analysis (NEW)
    K100 = {"ready_sec": float(os.environ.get("K100_READY_SEC") or 1500), "fracs": (0.9, 0.8, 0.7), "safety_mib": 1024,
            "mtp_reserve_mib": 1448, "k2_min_free_mib": 512, "long_max_new": 8192, "long_cap_sec": 900,
            "seqs_k": 8, "seqs_k2": 16, "seqs_k3": 32, "admission_max_new": 4096, "n_cap": 16, "k2_body_min_min": 16, "k2_ready_default_sec": 780,
            "ready_factor_k2": 1.5, "teardown_sec": 180, "step_margin_sec": 300}
    # R1-2: measured ready times 423-777 s for MTP boots, 297-642 s otherwise; teardown 1.5-2.5 min. K2's minimum = its
    # shortest body (16 min) + 1.5 x K's measured ready time + teardown + step()'s margin (780 s default before K is ready)
    K100["k2_reserve_sec"] = K100["k2_body_min_min"] * 60 + K100["ready_factor_k2"] * K100["k2_ready_default_sec"] + 180 + 300
    K1["params"] = dict(K100, mtp_reserve_basis="klee100 MTP tensors / TP2 per PP1 card (680 MiB) + 768 MiB draft workspace / "
                                                "graphs (NB0 B -> D on the W4 PP1 cards: +1,004 MiB at seqs 8; NB2 SMOKE seqs 16 graphs 0.52 GiB)",
                        safety_basis="1 GiB beyond the K0 peak (after a 90k teacher-forced request): NB1's 0.5 GiB rule + 0.5 GiB",
                        k2_reserve_basis="K-10 / R1-2: K's optional items (the 120k row, the 90k TF) and extra K ladder rungs run only if "
                                         "left() - K2's minimum (16 min body + 1.5 x K's ready + teardown + margin) covers them")
    KV_DEFAULT = KV_BF16                           # 2,464,415,744 B: Merlin's own L4 BF16 pin (RC2 BF16, NB0 A / C)
    FP8_CAPTURES = flags(RC2_REF["fp8"]["argv_tail"])["--compilation-config"]      # NB0 boot A: RC2 FP8's capture list (to 32)
    CAPTURE_HI = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]                     # NB1: Merlin capture_sizes_for(16, K=3)
    COMPILE_HI = json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": CAPTURE_HI})
    CAPS_LIST = {FP8_CAPTURES: [1, 2, 3, 4, 6, 8, 12, 16, 24, 32], COMPILE_HI: CAPTURE_HI}
    PORTS = {"K0": 8020, "K": 8021, "K2": 8022, "K3": 8023}
    CAP_MIN = {"K0": 12, "K": 24, "K2": 34, "K2_MAX": 34, "K3": 60}   # K3: its sweep is gated by the reserves   # boot body caps (watched: killed at cap + 5 min); K2 from R1-2
    EST_MIN = {"K0": 27, "K": 33, "K2": 43, "K3": 30}   # R1-2: body + measured load + teardown; V-3: K3 = a short core-first diagnostic
    BODY_MIN = {"K0": 12, "K": 16, "K2": 16}           # V-3: the shortest useful body of each later boot (its reserve below)
    K3_GATE_D_GIB = {"PP0": 11.973, "PP1": 12.026}            # the klee100 CPU kit v2 Gate D per-stage GPU weights estimate (language-model-only, calibrated)
    K3_PINS = KL.k3_ladder(KL.k3_pin_steps(K3_GATE_D_GIB))     # rev 3: the measured-v1 raised rungs + the B3-based ladder from step 1
    KV_FP8_DTYPE = flags(RC2_REF["fp8"]["argv_tail"])["--kv-cache-dtype"]          # RC2 FP8: fp8_e4m3
    FP8_BLOCK_BYTES = 2479838700 / 150             # RC2 FP8 block (the kit's Gate D unit: 463 blocks = its estimate)
    BF16_BLOCK_BYTES = 2464415744 / 79             # K-11: RC2 BF16 block = 29.75 MiB (FP8 15.77): K / K2 run BF16 KV
    RC2_WINDOWS = {"rc2_fp8_8k@8": {"tok_s": 261.8, "tick_ms": 48.2, "mal": 3.154}, "rc2_fp8_40k@5": {"tok_s": 190.5},
                   "nb1_r1_b1_fp8_8k@8": {"tok_s": 280.0, "tick_ms": 45.1}, "nb1_r1_b1_fp8_40k@8": {"tok_s": 248.6},    # R2-9: NB1 R1
                   "nb1_r2_b1_fp8_8k@8": {"tok_s": 261.0, "tick_ms": 48.2, "mal": 3.148}, "nb1_r2_b1_fp8_40k@5": {"tok_s": 196.3, "tick_ms": 40.5},
                   "nb1_r2_b1_fp8_40k@7": {"tok_s": 227.9, "tick_ms": 46.7}, "nb1_r2_b3_fp8_40k@8": {"tok_s": 249.5, "tick_ms": 49.9},
                   "note": "W4 Merlin, FP8 KV, MTP K3 (RC2 / NB1 R1 B1 / NB1 R2 B1 + B3): no BF16-KV load window exists for W4"}
    K1BOOTS, REF, GATES, RECS = {}, {}, {}, {}      # RECS: the boot records by role (K0 / K / K2), not saved twice
    # prompts (RC2's prompt(); distinct variants share no prefix block): 16 x 40k, 16 x 60k (variants 16-31, cold vs the 40k
    # window), one 120k (variant 40)
    PROMPTS["40k"] += [prompt(BANDS["40k"], v) for v in range(len(PROMPTS["40k"]), 16)]
    BANDS["60k"], BANDS["120k"] = 60000, 120000
    PROMPTS["60k"] = [prompt(BANDS["60k"], v) for v in range(16, 32)]
    PROMPTS["120k"] = [prompt(BANDS["120k"], 40)]
    MAX_NEW.update({"60k": MAX_LEN - BANDS["60k"] - 512, "120k": MAX_LEN - BANDS["120k"] - 512})
    R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}


    def k1_boot(key, name, mtp, seqs, captures, kv_bytes, title, port=None, ladder=False, kv="bf16"):
        """One boot plan. R1-17: --language-model-only (text only: the 333 model.visual.* tensors, ~440 MiB on PP0, are not
        loaded; NB1 R2 adopted it with a forward test EXACT vs RC2). R1-9: `port` = a fresh port per ladder try. R1-4: a ladder
        boot gets no inner OOM retry (the ladder itself steps down)."""
        b = {"key": key, "name": name, "kv": kv, "mtp": mtp, "seqs": seqs, "captures": captures, "port": port or PORTS[key],
             "kv_bytes": int(kv_bytes), "title": title}
        K1BOOTS[name] = b
        NB1_EXTRA[name], NB1_ENV[name], NB1_KV[name] = list(EXT_FLAG) + ["--language-model-only"], dict(DIAG_ENV), int(kv_bytes)
        if ladder:
            NO_INNER_RETRY.add(name)
        return b


    _merlin_fn_server_nb1 = merlin_fn_server       # RC2 CELL 2 (launch.build + host_phase) with NB1's per-boot hooks


    def merlin_fn_server(name, kv, port, kv_bytes):
        """E25 klee100: K6 before every launch (one missing indexed shard -> no server is started), RC2's Merlin plan, NB0's
        argv edits (MTP off = the same command without --speculative-config; seqs / captures = the existing flags' values),
        the plan + config diff vs RC2 BF16 recorded, NB0's load monitor (launch -> ready)."""
        b = K1BOOTS[name]
        k6 = KL.k6_check(CKPT)
        K1.setdefault("k6_before_launch", {})[name] = {k: k6.get(k) for k in ("ok", "why", "index_files", "missing")}
        if not k6["ok"]:
            save()
            raise RuntimeError("K6_INDEXED_SHARD_MISSING before launch: " + str(k6["why"]))
        s = _merlin_fn_server_nb1(name, kv, port, kv_bytes)
        s.argv = nb0_argv(s.argv, b)
        LIVE_PIN.update(name=name, kv_bytes=int(kv_bytes))      # R1-4: the pin THIS launch uses (after an inner retry too)
        if kv == "fp8":                                          # R3-1: K3's static K/V scales (Merlin also sets it from SCALES)
            s.env["VLLM_QSA_KV_SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
        # K-9: vLLM's own `serve --help=all` (cached per engine) against THIS argv: a missing flag -> no launch (FAIL_CLI_FLAGS);
        # help itself failing is recorded and the launch continues (the NB1 R2 boots with the same argv family worked)
        HELP[name] = L.help_check("merlin", s.argv[:s.argv.index("serve") + 1], s.env, s.argv)
        cli = K1.setdefault("cli_flags", {})[name] = KL.cli_flags_check(HELP[name])
        if cli["ok"] is False:
            save()
            raise RuntimeError("FAIL_CLI_FLAGS: " + str(cli["why"]))
        R["setup"].setdefault("k100_plans", {})[name] = {
            "boot": b["key"], "title": b["title"], "argv_tail": s.argv[-64:], "kv_bytes": kv_bytes,
            "env": {k: s.env[k] for k in sorted(s.env) if k.startswith(ENV_KEYS + ("MERLIN_",))},
            "diff_vs_rc2_" + kv: config_diff(s.argv, s.env, kv_bytes, kv)}
        save()
        if not os.environ.get("RC2_CPU_KIT"):
            deadline = time.time() + max(120, min(K100["ready_sec"] + 120, left() - 240))
            threading.Thread(target=load_monitor, args=(s, name, deadline), daemon=True).start()   # R2-5: keyed by the launch name
        return s


    LIVE_PIN = {"name": None, "kv_bytes": None}


    def live_pin(b):
        """R1-4: the pin of the running launch of boot b (an inner OOM retry runs 0.25 GiB lower than planned)."""
        n = LIVE_PIN.get("name") or ""
        return int(LIVE_PIN["kv_bytes"]) if LIVE_PIN.get("kv_bytes") and (n == b["name"] or n.startswith(b["name"] + "_t")) else int(b["kv_bytes"])


    def merlin_boot(name, kv, port, timeout, body, need=(), expected_rows=4, require_sampled=True):
        """E25 klee100 (rebinds RC2 CELL 5's merlin_boot; same contract and gates) with the second-round audit fixes:
        R2-5 every retry is its own launch `<name>_t<k>` (log, plan, HELP, load monitor, server record kept per launch);
        R1-9 a retry uses a fresh port (+10: Merlin derives MASTER_PORT / VLLM_PORT from it); R1-4 no inner retry for ladder
        boots (NO_INNER_RETRY); R1-8 OOM / device-fault lines are searched in the launch's WHOLE log, not the last 4,000 chars;
        R1-10 / R2-2 / R2-3 a server death is named by what was running (KL.death_label: BOOT_CAP_KILL when the boot watchdog
        fired, TF_OOM only for a death in a TF request, BODY_OOM_<item> otherwise) and never hides a more specific body status."""
        kvb = NB1_KV.get(name) or KV_BF16
        tries = []
        while True:
            lname = name if not tries else f"{name}_t{len(tries)}"
            if lname != name:
                NB1_EXTRA[lname], NB1_ENV[lname], NB1_KV[lname] = list(NB1_EXTRA.get(name) or []), dict(NB1_ENV.get(name) or {}), kvb
                K1BOOTS[lname] = dict(K1BOOTS[name], name=lname, port=port, kv_bytes=kvb)
            gate = boot_gate()
            if gate["ok"] is not True:
                return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries, "launch_name": lname}
            shm0 = shm_names()
            srv_obj = merlin_fn_server(lname, kv, port, kvb)
            o = with_servers([srv_obj], timeout, body)
            try:
                full = srv_obj.log() or ""
            except Exception:
                full = ""
            o.update(kv_bytes=kvb, boot_gate=gate, shm_cleanup=shm_cleanup(shm0), launch_name=lname, port=port,
                     ready_sec=getattr(srv_obj, "ready_sec", None), oom_in_log=bool(OOM_ERR.search(full)), xid_in_log=bool(XID.search(full)))
            oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]
            tries.append({"launch_name": lname, "port": port, "kv_bytes": kvb, "status": o.get("status"), "oom": oom,
                          "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"]})
            if not oom or o["xid_in_log"] or len(tries) > 1 or name in NO_INNER_RETRY or left() < timeout + 900:
                break
            kvb -= OOM_STEP
            port += 10
            log("boot OOM:", lname, "retry with KV_BYTES", kvb, "port", port)
        o["tries"] = tries
        fired = bool(o.get("boot_watchdog")) or BOOT.get("fired") in (name, lname)      # V-13b: the watchdog is checked first
        if fired:
            o["death"], o["status_before_kill"], o["status"] = "BOOT_CAP_KILL", o.get("status"), "BOOT_CAP_KILL"
        elif o.get("server_status") == "SERVER_DIED" or o.get("status") == "SERVER_DIED":
            o["death"] = KL.death_label(False, o["oom_in_log"], o["xid_in_log"], o.get("died_in"))
            if o.get("status") in (None, "DONE", "NOT_RUN", "SERVER_DIED", "BODY_ERROR"):
                o["status"] = o["death"]
        g = gated(o, need=need, expected_rows=expected_rows, require_sampled=require_sampled)
        if (g.get("status") == "TF_OOM" or str(g.get("status")).startswith("BODY_OOM_")) and g.get("correctness") != "CLEAN":
            g["status"] = g["correctness"]
        lg = long_gate(g)                                  # RC2's long rows (none here: the 120k row has its own verdict)
        g["long_verdict"] = lg or ("PASS_LONG" if isinstance(g.get("long"), dict) else None)
        if lg and g.get("status") in ("DONE", "TF_OOM"):
            g["status"] = lg
        return g


    def watched(body, cap_sec, name):
        """E25 klee100 (rebinds NB1's watched; same backstop: at cap + 300 s the server is killed). R1-2: the cap is clipped
        at body start to left() - 480 s (teardown + verdict). R2-2: the boot record carries body_cap_sec / boot_watchdog."""
        def b(ss):
            cap = max(60.0, min(float(cap_sec), left() - 480))
            BOOT.update(deadline=time.time() + cap, fired=None)

            def fire():
                BOOT["fired"] = name
                for x in ss:
                    try:
                        x.kill()
                    except Exception:
                        pass
            tm = threading.Timer(cap + 300, fire)
            tm.daemon = True
            tm.start()
            try:
                o = body(ss)
                LIVE["out"] = o if isinstance(o, dict) else None
            finally:
                tm.cancel()
                BOOT["deadline"] = None
            o["body_cap_sec"] = round(cap)
            if BOOT["fired"]:
                o["boot_watchdog"] = f"FIRED after {cap + 300:.0f} s: server killed"
            return o
        return b


    class GpuSampler(threading.Thread):
        """nvidia-smi every `every` s while a boot body runs: the highest used MiB per card (a transient between readings)."""
        def __init__(self, every=5.0):                      # R1-14: 5 s like RC2 / NB1 (2 s ran nvidia-smi inside the windows)
            super().__init__(daemon=True)
            self.every, self.ev, self.max, self.tot, self.n = every, threading.Event(), {}, {}, 0

        def run(self):
            while not self.ev.is_set():
                for i, u, t in (L.gpu_mem() or []):
                    self.max[i], self.tot[i] = max(self.max.get(i, 0), u), t
                self.n += 1
                self.ev.wait(self.every)

        def rows(self):
            return [[i, self.max[i], self.tot[i]] for i in sorted(self.max)]


    def k100_facts(s, b, o):
        """Config read-back from the server's own log + the worker extension: KV dtype, seqs, pin, spec (MTP on / off),
        captures, prefix caching, quantization, model dir; per-rank memory; pool; humming / method / PLE lines.
        config_ok only when every check holds (else CONFIG_MISMATCH: the boot stops)."""
        t = s.log() or ""
        o["caps"], o["pool"], o["graphs"] = server_caps(t, s.argv), pool_facts(t), graph_facts(t)
        o["workers"] = rpc_cache_info(s.port)
        lines = t.splitlines()
        nd = next((l for l in lines if "non-default args:" in l), "")
        eng = next((l for l in lines if "Initializing a V1 LLM engine" in l), "")
        g = lambda rx, txt: (re.search(rx, txt) or [None, None])[1]
        f = {"kv_cache_dtype": g(r"'kv_cache_dtype': '(\w+)'", nd), "engine_kv": g(r"kv_cache_dtype=(\w+)", eng),
             "max_num_seqs": g(r"'max_num_seqs': (\d+)", nd), "kv_cache_memory_bytes": g(r"'kv_cache_memory_bytes': (\d+)", nd),
             "argv_kv_bytes": flags(s.argv).get("--kv-cache-memory-bytes"), "spec_in_args": "'speculative_config':" in nd,
             "spec_k_args": g(r"'num_speculative_tokens': (\d+)", nd), "engine_spec": g(r"speculative_config=(None|SpeculativeConfig\([^)]*\))", eng),
             "quantization": g(r"quantization=([\w\-]+)", eng), "dtype": g(r"dtype=torch\.(\w+)", eng), "model": g(r"'model': '([^']+)'", nd),
             "captures": o["caps"].get("capture_sizes"), "prefix_caching": "'enable_prefix_caching': True" in nd,
             "mtp_off_warning": bool(re.search(r"needs speculative decoding; num_speculative_tokens=0", t)),
             "engine_line": eng[:2000], "language_model_only": o["caps"].get("language_model_only")}   # R1-17 (recorded)
        es = f["engine_spec"] or ""
        want_kv = "bfloat16" if b.get("kv", "bf16") == "bf16" else KV_FP8_DTYPE      # R3-1: K3 is FP8 E4M3
        ck = {"kv_dtype_args": f["kv_cache_dtype"] == want_kv, "kv_dtype_engine": f["engine_kv"] == want_kv,
              "seqs": f["max_num_seqs"] == str(b["seqs"]), "prefix_caching": f["prefix_caching"], "dtype_bf16": f["dtype"] == "bfloat16",
              "pin": f["kv_cache_memory_bytes"] is not None and f["kv_cache_memory_bytes"] == str(f["argv_kv_bytes"]),
              "quantized": f["quantization"] not in (None, "None"), "captures": f["captures"] == CAPS_LIST.get(b["captures"]),
              "model_dir": f["model"] == str(CKPT),
              "mtp": ((f["spec_in_args"] and f["spec_k_args"] == "3" and es.startswith("SpeculativeConfig") and "num_spec_tokens=3" in es)
                      if b["mtp"] else (not f["spec_in_args"] and es == "None" and f["mtp_off_warning"]))}
        # K-7: the GPU quant method map (worker extension, every rank) vs the checkpoint map; gated on K and K2, recorded on K0
        part = KL.parse_partition((getattr(s, "env", None) or {}).get("VLLM_PP_LAYER_PARTITION"))   # V-1: the boot's own 25/23
        # rev 3: the raw per-rank rows are kept in the boot record (k100_results.json), never only the summary
        o["method_map_raw"] = raw_mm = rpc(s.port, "k100_method_map")
        o["method_map"] = mm = KL.method_map_check((raw_mm or {}).get("results"), need_mtp=b["mtp"], partition=part)
        f["method_map_verdict"] = mm.get("verdict")
        if b["key"] in ("K", "K2", "K3"):     # rev 3: METHOD_CLASS_ONLY (bits unreadable, every class right) is not a mismatch
            ck["GPU_METHOD_MAP_PASS"] = mm.get("gate_ok") is True
        if b.get("kv") == "fp8":                                  # R3-1 / V-4: every rank applied its stage's QSA scales
            try:
                side = json.loads((CKPT / "qsa_kv_scales_262k.json").read_text())
            except Exception as e:
                side = {}
                o["scales_sidecar_error"] = repr(e)[:200]
            o["scales"] = KL.scales_check(t, side, part)          # per rank: 4 ranks, the stage subset, the sidecar values
            o["scales"]["calibration"] = f["scales_calibration"] = SCALES_CALIBRATION       # V-7
            o["scales_nb0"] = scales_facts(t)                     # NB0's aggregate reading (recorded)
            o["scales_rpc"] = rpc(s.port, "k100_kv_scales")       # the modules' own k / v scales on every rank (recorded only)
            ck["STATIC_SCALES_APPLIED"] = o["scales"].get("ok") is True
        f["checks"] = ck
        f["humming_lines"] = _pick(t, r"(?i)humming", 60)
        f["method_lines"] = _pick(t, r"Using \w+ for \w+|Using \w+(MoE)?Method|MoE backend|WNA16|MarlinExperts|INC|auto.?round", 40)
        f["unquantized_lines"] = _pick(t, r"Unquantized(FusedMoE|Linear)Method|falling back to unquantized", 10)
        f["ple_lines"] = _pick(t, r"(?i)PLE offload|Fp8Embedding|ple_embedding_dtype|PLE.*ready", 12)
        f["nan_lines"] = _pick(t, r"(?i)\bnan\b|non-finite|inf detected", 10)
        f["traceback_lines"] = _pick(t, r"Traceback|Error:", 12)
        o["readback"] = f
        o["config_ok"] = all(ck.values())
        pl = o["pool"]
        o["memory_census"] = {"rank_load_gib": pl.get("rank_load_gib"), "rank_load_sec": pl.get("rank_load_sec"),
                              "rank_graph_gib": pl.get("rank_graph_gib"), "initial_free_reserved_gib": pl.get("initial_free_reserved_gib"),
                              "workers_mem": [[w.get("rank"), w.get("mem")] for w in ((o["workers"] or {}).get("ranks") or []) if isinstance(w, dict)],
                              "kv_tokens": pl.get("kv_tokens"), "max_concurrency": [pl.get("max_concurrency_tokens"), pl.get("max_concurrency")]}
        return o


    def warmup_gauge(port, n):
        """NB1's warm-up (excluded from every metric) with the kv_cache_usage gauge sampled every 0.5 s: the usage
        denominators give the pool in blocks (pool_blocks_from_usage), the cross-check of the worker num_gpu_blocks."""
        us, ev = [], threading.Event()

        def samp():
            while not ev.is_set():
                try:
                    us.append(kvu(port))
                except Exception:
                    pass
                ev.wait(0.5)
        th = threading.Thread(target=samp, daemon=True)
        th.start()
        try:
            w = warmup(port, n)
        finally:
            ev.set()
            th.join(5)
        w["kv_usage_samples"] = len(us)
        return w, us


    def k100_capacity(o, us, kv_bytes, kv="bf16"):
        """This boot's pool P (worker num_gpu_blocks - 1, cross-checked with the gauge), F from the max-concurrency line
        (f_from_log), bytes per block, the FP8-equivalent AND the BF16 estimate of the pin vs the kit's 463-block estimate
        (K-11: the measured worker blocks are authoritative), the stream counts. K-3: capacity_ok only when the worker and the
        gauge agree (pool_check MATCH) and F is measured; otherwise n40 / n64 / resident_bound are None (why KV_POOL_<check> /
        F_NOT_MEASURED) and K2 runs neither 40k@N nor the 64k admission."""
        wtot = sorted({x for x in ((o.get("workers") or {}).get("num_gpu_blocks") or []) if isinstance(x, int) and not isinstance(x, bool) and x > 1})
        wb = [x - 1 for x in wtot]
        pg = KL.pool_from_usage(us or [], wb[0] if len(wb) == 1 else None)       # R2-7: the worker's P first, >= 4 distinct samples
        P = wb[0] if len(wb) == 1 else pg
        chk = ("MATCH" if len(wb) == 1 and pg == wb[0] else "WORKER_ONLY" if len(wb) == 1 and pg is None else
               "GAUGE_ONLY" if not wb and pg else "MISMATCH" if len(wb) == 1 and pg else "UNKNOWN")
        fl = f_from_log(o, P) if P else {"ok": False, "why": "pool unknown"}
        F = fl.get("F") if fl.get("ok") else None
        seqs = (o.get("caps") or {}).get("max_num_seqs")
        tot = P + 1 if P else None                    # R2-8: physical blocks = usable + the null block
        out = {"pool_blocks": P, "num_gpu_blocks_total": tot, "worker_usable_blocks": wb, "gauge_pool_blocks": pg, "pool_check": chk,
               "F_log": fl, "F": F, "kv_bytes": kv_bytes, "bytes_per_block": round(kv_bytes / P) if P else None,
               "bytes_per_block_usable": round(kv_bytes / P) if P else None, "bytes_per_block_total": round(kv_bytes / tot) if tot else None,
               "kv_tokens": (o.get("pool") or {}).get("kv_tokens"),
               "basis": "R2-8: usable blocks (num_gpu_blocks - 1) against 463 / 244; RC2's block bytes are pin / TOTAL blocks (150 FP8, 79 BF16)",
               "fp8_equivalent_blocks_of_pin": round(kv_bytes / FP8_BLOCK_BYTES, 1),
               "bf16_estimate_blocks_of_pin": round(kv_bytes / BF16_BLOCK_BYTES, 1),
               "pin_fp8_equivalent_usable": round(kv_bytes / FP8_BLOCK_BYTES - 1, 1), "pin_bf16_estimate_usable": round(kv_bytes / BF16_BLOCK_BYTES - 1, 1),
               "kv_dtype": kv,
               "measured_fp8_equivalent_usable": (P if kv == "fp8" else round(tot * BF16_BLOCK_BYTES / FP8_BLOCK_BYTES - 1, 1)) if tot else None,
               "kit_estimate": {"fp8_blocks": 463, "fp8_usable_blocks": 463, "kv_bytes_equiv": int((463 + 1) * FP8_BLOCK_BYTES),
                                "bf16_usable_blocks_est": int((463 + 1) * FP8_BLOCK_BYTES / BF16_BLOCK_BYTES) - 1,
                                "note": "kit Gate D: RC2's 149 FP8 blocks + the bytes freed by the low-bit experts on the binding stage, "
                                        "all given to KV (MTP on); 463 is FP8-EQUIVALENT usable blocks: at BF16 KV (29.75 vs 15.77 MiB per block) "
                                        "the same bytes are ~244 usable blocks; this pin leaves the safety / MTP reserve / 1 - frac unused"},
               "authoritative": "pool_blocks (the measured worker num_gpu_blocks - 1, checked with the gauge); the estimates are arithmetic",
               "max_num_seqs": seqs, "n40": None, "n64": None, "resident_bound": None}
        out["capacity_ok"] = chk == "MATCH" and F is not None and bool(P)
        if out["capacity_ok"]:
            out["n40"] = n_fit(P, F, BLOCK, "40k", 0, seqs, len(PROMPTS["40k"]))
            out["n64"] = residency_bound(P, F, "60k", max_new=K100["admission_max_new"], seqs=seqs)
            out["resident_bound"] = {L_: P // (math.ceil(L_ / BLOCK) + F) for L_ in (40000, 64096, 90000, 120000)}
        else:
            out["why"] = f"KV_POOL_{chk}" if chk != "MATCH" else "F_NOT_MEASURED"
        return out


    def method_map_summary(o):
        mm = (o or {}).get("method_map") or {}
        return {k: mm.get(k) for k in ("ok", "verdict", "gate_ok", "layers_matching", "layers_class_only", "layers_expected", "mtp_ok",
                                       "need_mtp", "bits_readable", "bits_sources", "drafter_paths", "why")}


    def arith_probe(port):
        """K-12: 8 fixed exact-answer arithmetic prompts (multi-digit add / sub / mul, carries) through the lab's chat_probes
        (greedy, thinking off, 32 tokens); exact-match recorded, never gated."""
        try:
            return KL.arith_score(L.chat_probes(port, cases=KL.arith_cases()))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:300], "gated": False}


    # K-8: every stream of every load window (lab steady() via NB1 window(), and NB1 admission_test()) is written in full
    # (text, token ids, finish) to traces/<boot>_<window>.jsonl after the window returned (outside the timed part). The shared
    # E23 lab file is not changed: L.Stream.__init__ is wrapped here so streams created inside traced() are collected.
    TRACE_CTX = {"collect": None, "label": None, "write": True}


    def k100_trace_install():
        """K-8 / V-8: streams created inside traced() are registered with (boot, window, stream id); each stream's line is
        appended as soon as ITS request ends (the wrapped run()), so a SIGKILL loses at most the unfinished streams; traced()
        writes the rest at the window end."""
        if getattr(L.Stream, "_k100_traced", False):
            return
        init = L.Stream.__init__

        def _init(self, *a, **k):
            init(self, *a, **k)
            c = TRACE_CTX["collect"]
            if c is not None:
                self._k100_ctx = (BOOT.get("tag"), TRACE_CTX.get("label"), len(c), TRACE_CTX.get("write", True))
                c.append(self)
        L.Stream.__init__ = _init
        run0 = getattr(L.Stream, "run", None)
        if run0 is not None:
            def _run(self, *a, **k):
                try:
                    return run0(self, *a, **k)
                finally:
                    if getattr(self, "_k100_ctx", None):
                        k100_trace_one(self)
            L.Stream.run = _run
        L.Stream._k100_traced = True


    def k100_trace_one(g):
        """One stream's full record (text, token ids, finish), once: key (boot, window, stream id)."""
        boot, label, i, write = g._k100_ctx
        if getattr(g, "_k100_written", False) or not write:
            return
        g._k100_written = True
        path = TRACES / f"{boot or 'boot'}_{label}.jsonl"
        try:
            row = g.row()
            body = getattr(g, "body", None) or {}
            with _SAVE_LOCK, open(path, "a") as fh:
                fh.write(json.dumps({"boot": boot, "window": label, "i": i, "port": getattr(g, "port", None),
                                     "prompt_tokens": len(body.get("prompt") or []), "prompt_sha": psha(body.get("prompt") or [])[:16],
                                     "seed": body.get("seed"), "max_tokens": body.get("max_tokens"), "temperature": body.get("temperature"),
                                     "text": row["text"], "tokens": row["tokens"], "finish_reason": row["finish_reason"],
                                     "error": row["error"], "done": row["done"], "t": time.time()}) + "\n")
            K1.setdefault("traces", {})[path.name] = K1.get("traces", {}).get(path.name, 0) + 1
        except Exception as e:
            K1.setdefault("trace_errors", []).append(f"{label}: {type(e).__name__}: {e}"[:200])


    def k100_trace(label, streams):
        for g in streams:
            if getattr(g, "_k100_ctx", None):
                k100_trace_one(g)


    def trace_disk_ok(min_gb=2.0):
        """V-8: free disk space under the traces dir (recorded); below min_gb the window's traces are skipped with a note."""
        try:
            free = shutil.disk_usage(str(TRACES)).free / 1e9
        except Exception:
            return True, None
        return free >= min_gb, round(free, 2)


    def traced(label, fn):
        got = []
        ok, free = trace_disk_ok()
        K1.setdefault("trace_disk_gb", {})[label] = free
        if not ok:
            K1.setdefault("trace_notes", []).append(f"{BOOT.get('tag')} {label}: {free} GB free < 2 GB: traces skipped")
        TRACE_CTX.update(collect=got, label=label, write=ok)
        try:
            return fn()
        finally:
            TRACE_CTX.update(collect=None, label=None, write=True)
            k100_trace(label, got)


    k100_trace_install()


    def long_prefill(port):
        """One ~120k-token request (RC2 prompt(120000, 40)), temperature 0.6, its own stop or the cap (8,192 new tokens /
        900 s); TTFT = the prefill, the 2,000-token garbage-window check over the generated ids, NaN margins (logprobs 2).
        A row ended by a cap is incomplete, never a failure by itself; garbage / a request error / NaN is a failure."""
        ids = PROMPTS["120k"][0]
        n = min(K100["long_max_new"], MAX_LEN - len(ids) - 512)
        cap = max(120, min(K100["long_cap_sec"], boot_left() - 240))
        g = L.Stream(port, ids, n, temperature=TEMP, seed=17, logprobs=2, ignore_eos=False)
        g.start()
        g.join(cap)
        if g.is_alive():
            g.stop()
            g.join(10)
        row = g.row()
        (TRACES / f"long120k_{port}.txt").write_text(row["text"])
        toks = row["tokens"]
        ttft = (g.t_first - g.t_start) if g.t_first and g.t_start else None
        dec = (g.t_end - g.t_first) if g.t_end and g.t_first else None
        wg = window_garbage(toks)
        nonfin = sum(1 for m in row["margins"] if m is not None and not (isinstance(m, (int, float)) and math.isfinite(m)))
        out = {"prompt_tokens": len(ids), "prompt_sha256": ids_sha(ids), "max_new": n, "cap_sec": round(cap), "tokens": len(toks),
               "finish": row["finish_reason"], "error": row["error"], "ttft_sec": round(ttft, 1) if ttft else None,
               "prefill_tok_s": round(len(ids) / ttft, 1) if ttft else None, "decode_tok_s": round(len(toks) / dec, 1) if dec else None,
               "window_garbage": wg, "tail_garbage": L.degenerate(TOK.decode(toks[-2000:])) if len(toks) >= 200 else None,
               "think_closed": "</think>" in row["text"], "nonfinite_margins": nonfin, "tail": row["text"][-400:]}
        out["verdict"] = ("FAIL_LONG_REQUEST" if row["error"] else "FAIL_LONG_NONFINITE" if nonfin else
                          "FAIL_LONG_GARBAGE" if (wg or out["tail_garbage"]) else
                          "FINISHED_NATURALLY_CLEAN" if row["finish_reason"] == "stop" else
                          "CAPPED_BY_TOKENS_CLEAN" if row["finish_reason"] == "length" else "CAPPED_BY_TIME_CLEAN")
        return out


    def tf_gate(tf):
        """NaN / request error / an incomplete prompt-logprob response (valid < expected) in any teacher-forced request of
        this boot -> the fail-fast label (else None)."""
        for band, x in (tf or {}).items():
            for run in ("fresh", "repeat"):
                m = (x or {}).get(run) or {}
                if m.get("error"):
                    return f"NO_GO_TF_REQUEST_ERROR ({band} {run}: {str(m['error'])[:80]})"
                if m.get("nonfinite"):
                    return f"NO_GO_TF_NONFINITE ({band} {run}: {m['nonfinite']} positions)"
                if KL.tf_incomplete(m):
                    return f"NO_GO_TF_INCOMPLETE ({band} {run}: valid {m.get('valid')} of {m.get('expected')})"
        return None


    def chat_probes_full(port, cases=None, tag="probes"):
        """The lab's chat_probes request (thinking off, T 0, 32 tokens, the same contains-match) with R2-10: the FULL content
        and reasoning of every case go to traces/chat_probes.jsonl (the lab record keeps content[:80])."""
        cases = cases or (("Reply with exactly one number: 7 * 6 =", "42"), ("Reply with exactly one number: 2 + 2 =", "4"),
                          ("Reply with exactly one number: 9 - 3 =", "6"))
        out = []
        for q, want in cases:
            body = {"model": "q27", "messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 32,
                    "chat_template_kwargs": {"enable_thinking": False}}
            rec, trace = {"q": q, "want": want}, {}
            try:
                r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                                       headers={"Content-Type": "application/json"}), timeout=300).read())
                msg = r["choices"][0]["message"]
                full = (msg.get("content") or "").strip()
                rec.update(content=full[:200], reasoning_chars=len(msg.get("reasoning") or ""), finish=r["choices"][0].get("finish_reason"),
                           ok=re.search(r"(?<!\d)" + re.escape(want) + r"(?!\d)", full) is not None, garbage=L.degenerate(full))
                trace = {"content_full": full, "reasoning_full": msg.get("reasoning")}
            except Exception as e:
                rec.update(error=repr(e)[:200], ok=False)
            out.append(rec)
            try:
                with _SAVE_LOCK, open(TRACES / "chat_probes.jsonl", "a") as fh:
                    fh.write(json.dumps(dict(rec, **trace, boot=BOOT.get("tag"), tag=tag, t=time.time())) + "\n")
            except Exception:
                pass
        return {"ok": all(x["ok"] for x in out), "cases": out}


    _template_probe_lab = L.template_probe


    def k100_template_probe(port, messages, want_ids, model="q27", timeout=120):
        """R2-10: the lab's template_probe (MATCH / MISMATCH / ERROR, unchanged) + the server's own token ids saved in
        traces/template_probe_<boot>.json (a second /tokenize call with the same body)."""
        res = _template_probe_lab(port, messages, want_ids, model=model, timeout=timeout)
        try:
            body = {"model": model, "messages": messages, "add_generation_prompt": True}
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout).read())
            p = TRACES / f"template_probe_{BOOT.get('tag') or 'boot'}.json"
            p.write_text(json.dumps({"server_tokens": list(r.get("tokens") or []), "want_tokens": list(want_ids), "result": res}))
            res["server_tokens_file"] = p.name
        except Exception as e:
            res["server_tokens_error"] = repr(e)[:200]
        return res


    L.template_probe = k100_template_probe


    def run_item(o, s, label, fn):
        """R1-10: o["died_in"] = the first item that began with the server alive and ended with it dead (it names the death:
        TF_OOM only for a TF request); o["items"] = every item started, in order."""
        alive0 = s.alive()
        o.setdefault("items", []).append(label)
        try:
            return fn()
        finally:
            if alive0 and not s.alive() and "died_in" not in o:
                o["died_in"] = label


    def _start(s, b, o):
        if LIVE.get("rec") is not None:                       # R1-12: a mid-boot save() keeps the body's record
            LIVE["rec"]["live_body"] = o
        o["kv_bytes_live"] = live_pin(b)
        o["census"] = kv_census(s)
        run_item(o, s, "facts", lambda: k100_facts(s, b, o))      # read-back + method map first (the user's K3 order)
        o["probes"] = KL.probe_engine_view(run_item(o, s, "probes", lambda: chat_probes_full(s.port)))   # R1-5
        o["sampled"] = run_item(o, s, "sampled", lambda: sampled_check(s))
        o["gpu_after_ready"] = L.gpu_mem()
        if short_gate_failed(o):
            o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
            return False
        if o["config_ok"] is not True:
            o.update(early_stop="CONFIG_MISMATCH", status="CONFIG_MISMATCH",
                     config_mismatch=[k for k, v in o["readback"]["checks"].items() if not v])
            return False
        return True


    def _finish(s, o, smp):
        smp.ev.set()
        try:
            o["gpu_end"], o["gpu_peak_rows"], o["gpu_samples"] = L.gpu_mem(), smp.rows(), smp.n
            o["census_end"] = kv_census(s)
            o["workers_end"] = rpc_cache_info(s.port) if s.alive() else {"error": "server dead"}
            o["free_mib_by_card"] = KL.free_by_card([o.get("gpu_after_ready"), o.get("gpu_end"), o.get("gpu_peak_rows")])
            o["tf_died"] = not s.alive()
        except Exception as e:
            o["finish_error"] = f"{type(e).__name__}: {e}"[:300]
        return o


    def _oracle_gate(o, s, key, refs):
        o["oracle"] = run_item(o, s, "oracle", lambda: oracle(s.port, key=key, refs=refs))
        if not oracle_ok(o["oracle"]):
            o["early_stop"] = "ORACLE_INCOMPLETE_OR_GARBAGE"
            return False
        return True


    def _tf_stop(o, tf):
        g = tf_gate(tf)
        if g:
            o.update(early_stop=g, status=g.split(" ")[0])
        return bool(g)


    def k_body(b, fn):
        """A boot body: the GPU sampler, the shared short gates (_start), fn(s, o) and the closing census (_finish). An
        exception inside keeps everything measured so far (status BODY_ERROR on the record)."""
        def body(ss):
            s, o, smp = ss[0], {}, GpuSampler()
            smp.start()
            try:
                if _start(s, b, o):
                    fn(s, o)
            except Exception as e:
                o.update(status="BODY_ERROR", body_error=f"{type(e).__name__}: {e}"[:800], body_tb=traceback.format_exc()[-2500:])
            finally:
                _finish(s, o, smp)
            return o
        return body


    def body_k0(b):
        """K0 probe at Merlin's BF16 pin (MTP off) = NB0 A's exact configuration: fail fast on the short gates, the memory
        peaks the pin ladder needs (warm-up, a cold 40k prefill, a 90k teacher-forced request) and, R2-4, TF 2k / 8k / 40k
        fresh (~20 s, due()-gated): K0 vs A isolates the weights from the pin."""
        def fn(s, o):
            if not _oracle_gate(o, s, "k0", ("rc2_bf16",)):
                return
            o["warmup"], us = run_item(o, s, "warmup", lambda: warmup_gauge(s.port, 8))
            o["capacity"] = k100_capacity(o, us, live_pin(b))
            o["prefill"] = run_item(o, s, "prefill", lambda: prefill_probe(s.port))
            o["tf"] = run_item(o, s, "tf_2k-8k-40k", lambda: tf_suite("K0", s, ["2k", "8k", "40k"], runs=("fresh",)))
            if _tf_stop(o, o["tf"]):
                return
            o["tf"].update(run_item(o, s, "tf_90k", lambda: tf_suite("K0", s, ["90k"])))
            _tf_stop(o, o["tf"])
        return k_body(b, fn)


    def k_ready_sec():
        """R1-2: K's measured ready time (the load monitor of the running K launch, else K's record), else 780 s."""
        for rec in (LOADMON.get(BOOT.get("tag") or ""), RECS.get("K")):
            v = (rec or {}).get("ready_sec")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return float(v)
        return float(K100["k2_ready_default_sec"])


    def k2_reserve_sec():
        """R1-2 / K-10: K2's minimum = its shortest body + 1.5 x K's ready time + teardown + step()'s margin."""
        return (K100["k2_body_min_min"] * 60 + K100["ready_factor_k2"] * k_ready_sec() + K100["teardown_sec"]
                + K100["step_margin_sec"])


    def k2_reserve_ok(need_sec):
        """K-10: an optional K item runs only if the run budget minus K2's minimum covers it."""
        return left() - k2_reserve_sec() >= need_sec


    def body_k(b):
        """K: the correctness suite at the ladder pin (MTP off). K-10: the 120k row and the 90k TF are optional: each runs
        only if left() minus K2's reserve covers it (else SKIPPED_K2_RESERVE). R2-1: the T 0.6 120k row's verdict is a
        quality reading (o["quality"], appended to the final label), never the transport status K2 depends on."""
        def fn(s, o):
            if not _oracle_gate(o, s, "k", ("k0", "rc2_bf16", "k3")):        # k3: FP8 vs BF16 on klee100 (diagnostic)
                return
            o["arith"] = run_item(o, s, "arith", lambda: arith_probe(s.port))      # K-12: recorded, not gated
            o["warmup"], us = run_item(o, s, "warmup", lambda: warmup_gauge(s.port, 8))
            o["capacity"] = k100_capacity(o, us, live_pin(b))
            o["tf"] = run_item(o, s, "tf_2k-8k-40k", lambda: tf_suite("K", s, ["2k", "8k", "40k"]))
            if _tf_stop(o, o["tf"]):
                return
            o["core_done"] = KL.tf_core_done(o["tf"])          # R2-3: probes + oracle + TF 2k / 8k / 40k before the optional items
            o["sc"] = run_item(o, s, "sc", lambda: sc_suite("K", s))
            if not s.alive():
                o["long_120k"] = {"verdict": "SKIPPED_SERVER_DEAD"}
            elif not k2_reserve_ok(K100["long_cap_sec"] + 240):
                o["long_120k"] = {"verdict": "SKIPPED_K2_RESERVE", "left_sec": round(left()), "reserve_sec": round(k2_reserve_sec())}
            elif due(420):
                o["long_120k"] = run_item(o, s, "long_120k", lambda: long_prefill(s.port))
            else:
                o["long_120k"] = {"verdict": "SKIPPED_BOOT_BUDGET"}
            if k2_reserve_ok(2 * TF_EST["90k"] + 300):        # the largest prompt-logits request last (NB0 / RC2 rule)
                o["tf"].update(run_item(o, s, "tf_90k", lambda: tf_suite("K", s, ["90k"])))
            else:
                o["tf"]["90k"] = {"status": "SKIPPED_K2_RESERVE", "left_sec": round(left()), "reserve_sec": round(k2_reserve_sec())}
            _tf_stop(o, o["tf"])
            lv = str(o["long_120k"].get("verdict", ""))
            if lv.startswith("FAIL"):
                o.setdefault("quality", []).append("LONG_" + lv)
        return k_body(b, fn)


    def body_k2(b):
        """K2: MTP K3 at the K2 pin: read-back, oracle, arithmetic, TF 2k + 8k fresh (V-6: the K3 vs K2 pair) and 40k vs C
        (before the load windows: the comparisons survive a tight budget), warm-up, pool / F; V-14: the stream sweep at seqs
        16 (8k@16, 40k@16, each only when resident by the measured pool / F), the 64k admission at the largest resident level."""
        def fn(s, o):
            if not _oracle_gate(o, s, "k2", ("k", "rc2_bf16")):
                return
            o["arith"] = run_item(o, s, "arith", lambda: arith_probe(s.port))      # K-12: recorded, not gated
            o["tf"] = run_item(o, s, "tf_2k-8k", lambda: tf_suite("K2", s, ["2k", "8k"], runs=("fresh",)))   # V-6: K3 vs K2 pair
            if _tf_stop(o, o["tf"]):
                return
            o["tf"].update(run_item(o, s, "tf_40k", lambda: tf_suite("K2", s, ["40k"])))
            if _tf_stop(o, o["tf"]):
                return
            o["warmup"], us = run_item(o, s, "warmup", lambda: warmup_gauge(s.port, 8))
            cap = k100_capacity(o, us, live_pin(b), kv=b.get("kv", "bf16"))
            kv_sweep(o, s, cap, tuple(SWEEP_BANDS), levels=(16,))           # V-14: K2 (BF16, seqs 16) sweeps 16 per band
            admission_at_best(o, s, cap)
        return k_body(b, fn)


    # V-14 (user correction: long reasoning, no 8k band): the sweep bands 20k / 40k / 60k / 100k, levels 16 -> 24 -> 32. Each
    # band has its own DISTINCT RC2-style xhigh prompts (prompt(L, v), v from the band's base): identical prompts would share
    # prefix blocks and fake the residency; built lazily per level and released after the band (100k x 32 is ~3.2 M ids)
    SWEEP_BANDS = {"sw20k": (20000, 100), "sw40k": (40000, 140), "sw60k": (60000, 180), "sw100k": (100000, 220)}
    for _k, (_L, _v) in SWEEP_BANDS.items():
        BANDS[_k], MAX_NEW[_k] = _L, MAX_LEN - _L - 512


    def sweep_prompts(band, n):
        have = PROMPTS.get(band) or []
        if len(have) < n:
            L_, v0 = SWEEP_BANDS[band]
            PROMPTS[band] = have + [prompt(L_, v0 + i) for i in range(len(have), n)]
        return PROMPTS[band]


    def gen_during_window(band):
        """Tokens a stream generates during the admission + the window + 60 s (NB1 n_fit's growth rule)."""
        L0 = BANDS[band]
        return (L0 * 8 / max(1.0, PREFILL["tok_s"]) + WIN + 60) * GEN_RATE


    def level_need_sec(band, n):
        """V-14: the prefill-aware estimate of one level: the admission prefill (n x L / the measured prefill rate, >= 1,500
        tok/s) + the window + idle / teardown margins."""
        return n * BANDS[band] / max(1500.0, PREFILL["tok_s"]) + WIN + 180


    def kv_sweep(o, s, cap, bands, levels=KL.SWEEP_LEVELS, gate=None, skipped=None):
        """V-14: the stream sweep. Per band, levels 16 -> 24 -> 32 (within the boot's max_num_seqs): a level runs only when the
        pool / F are verified (capacity_ok) and it is resident at that context (KL.sweep_fit: prompt + the tokens generated
        during the window, the 0.95 rule), else SKIPPED_KV_BOUND with the numbers; `gate(need_sec)` (K3 in ALL: the K0 + K +
        K2 reserves) must hold for the prefill-aware need, else SKIPPED_K3_RESERVE; a band stops at the first level that
        preempts / runs below the level / is CAPACITY_LIMITED. Every window traced (K-8) and item-marked (R1-10); o["sweep"]
        one row per level (tok/s, tick, cycle, acceptance per position, running == level, preemptions, kv usage, the label,
        the prefill seconds), o["sweep_best"] = the highest clean level per band."""
        o["capacity"] = cap
        cok = cap.get("capacity_ok") is True
        P, F, seqs = cap.get("pool_blocks"), cap.get("F"), cap.get("max_num_seqs")
        st, sw, best = o.setdefault("steady", {}), o.setdefault("sweep", {}), o.setdefault("sweep_best", {})
        gate = gate or (lambda need: True)
        for band in bands:
            for n in levels:
                key = f"{band}@{n}"
                if isinstance(seqs, int) and n > seqs:
                    sw[key] = {"status": "SKIPPED_ABOVE_MAX_NUM_SEQS", "max_num_seqs": seqs}
                    break
                if not cok:
                    sw[key] = {"status": "SKIPPED_CAPACITY_NOT_VERIFIED", "why": cap.get("why")}
                    break
                fit = KL.sweep_fit(P, F, BANDS[band], n, gen_during_window(band), block=BLOCK)
                if not fit["fits"]:
                    sw[key] = dict(fit, status="SKIPPED_KV_BOUND")
                    break
                need = level_need_sec(band, n)
                if not due(need):                         # the boot body cap (a level never starts after it)
                    sw[key] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
                    break
                if not gate(need):
                    sw[key] = {"status": "SKIPPED_K3_RESERVE", "need_sec": round(need)}
                    if skipped is not None:
                        skipped.append(key)
                    break
                sweep_prompts(band, n)
                w = st[key] = run_item(o, s, "window_" + key, lambda: traced(key, lambda: window(s.port, s, band, n)))
                stop = KL.sweep_stop(w, n)
                sw[key] = {x: w.get(x) for x in ("status", "tok_s", "tick_ms", "cycle_ms", "running_min", "running_max", "running_eq_level",
                                                 "preemptions", "kv_usage_max", "mean_acceptance_length", "accept_rate_per_position_incl_admission",
                                                 "prefill_tok_s_uncached")}
                sw[key].update(label=w.get("status"), fit=fit, stop=stop, need_sec=round(need),
                               prefill_sec=(w.get("admission") or {}).get("sec"))
                if stop:
                    break
                if w.get("status") == "STEADY" and w.get("running_eq_level"):
                    best[band] = n
            if band in SWEEP_BANDS:
                PROMPTS.pop(band, None)                   # free the band's ids


    def admission_at_best(o, s, cap, gate=None, skipped=None):
        """V-14: the 64k admission at the largest resident level found at 60k (else 40k), min with the pool's 60k bound."""
        cok = cap.get("capacity_ok") is True
        bst = o.get("sweep_best") or {}
        n = min(cap.get("n64") or 0, bst.get("sw60k") or bst.get("sw40k") or 0, len(PROMPTS["60k"])) if cok else 0
        need = max(390 + n * BANDS["60k"] / max(2000.0, PREFILL["tok_s"]), max(420.0, 180.0 + n * BANDS["60k"] / max(1000.0, PREFILL["tok_s"])) + 60)
        if not cok:
            o["admission_64k"] = {"status": "SKIPPED_CAPACITY_NOT_VERIFIED", "why": cap.get("why")}
        elif n < 1:
            o["admission_64k"] = {"status": "SKIPPED_NO_RESIDENT_40K_LEVEL", "n64": cap.get("n64"), "sweep_best": o.get("sweep_best")}
        elif gate is not None and not gate(need):
            o["admission_64k"] = {"status": "SKIPPED_K3_RESERVE", "need_sec": round(need), "n": n}
            if skipped is not None:
                skipped.append("admission_64k")
        else:
            o["admission_64k"] = run_item(o, s, f"admission_60k@{n}", lambda: traced(f"admission_60k@{n}", lambda: admission_test(
                s.port, s, band="60k", n=n, max_new=K100["admission_max_new"])))


    def k3_ready_sec():
        """V-3: K3's measured ready time (its running launch's load monitor, else its record), else 780 s."""
        for rec in (LOADMON.get(BOOT.get("tag") or ""), RECS.get("K3")):
            v = (rec or {}).get("ready_sec")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return float(v)
        return float(K100["k2_ready_default_sec"])


    def later_reserve_sec(ready=None):
        """V-3: K0 + K + K2 minimums = each boot's shortest body + 1.5 x the measured K3 ready time + teardown + step margin."""
        r = k3_ready_sec() if ready is None else float(ready)
        later = ("K0", "K", "K2") if K100_PROFILE != "K3_ONLY" else ()          # V-9: K3_ONLY may use the whole budget
        return sum(BODY_MIN[k] * 60 + K100["ready_factor_k2"] * r + K100["teardown_sec"] + K100["step_margin_sec"] for k in later)


    def k3_optional_ok(need_sec):
        """V-3: a K3 optional item runs only if, after it, left() still covers the K0 + K + K2 reserves."""
        return left() - need_sec >= later_reserve_sec()


    def k3_core_marker(o, b, cap):
        """V-15: the moment K3's core is done: one log line `@@K3_CORE_DONE {json}` (pin, pool blocks, F, method map, scales on
        all ranks, oracle, TF 2k / 8k labels vs NB0 A, ready time), saved in K1 too."""
        def lab(band):
            try:
                return cmp_pair(_arr("A", band), _arr("K3", band)).get("label_full")
            except Exception as e:
                return f"ERROR {type(e).__name__}"
        mk = {"pin": live_pin(b), "pool_blocks": cap.get("pool_blocks"), "F": cap.get("F"),
              "method_map_pass": (o.get("method_map") or {}).get("ok") is True, "method_map_verdict": (o.get("method_map") or {}).get("verdict"),
              "scales_all_ranks": (o.get("scales") or {}).get("ok") is True,
              "oracle": bool(oracle_ok(o.get("oracle"))), "tf_2k": lab("2k"), "tf_8k": lab("8k"), "ready_sec": k3_ready_sec(),
              "core_done": o.get("core_done") is True}
        o["core_marker"] = K1["k3_core_done"] = mk
        print("@@K3_CORE_DONE " + json.dumps(mk, default=str), flush=True)
        save()
        return mk


    def body_k3(b):
        """K3, the FIRST GPU boot (user decision): klee100, FP8 E4M3 QSA KV (static scales), MTP K3, seqs 32 (V-14), captures
        to 64. V-3: CORE first: read-back + method map + per-rank scales (_start), probes (engine faults only) / sampled / the
        4-row oracle, TF 2k + 8k vs NB0 A, the measured pool / F. Then, each gated by k3_optional_ok (the K0 + K + K2 reserves
        in ALL; the whole budget in K3_ONLY): the 20k and 40k sweeps 16 -> 24 -> 32 (core-plus), then 60k and 100k, TF 40k,
        the 64k admission at the largest resident level, the arithmetic probe."""
        def fn(s, o):
            if not _oracle_gate(o, s, "k3", ("rc2_fp8",)):
                return
            o["tf"] = run_item(o, s, "tf_2k-8k", lambda: tf_suite("K3", s, ["2k", "8k"]))
            if _tf_stop(o, o["tf"]):
                return
            o["warmup"], us = run_item(o, s, "warmup", lambda: warmup_gauge(s.port, 8))
            cap = k100_capacity(o, us, live_pin(b), kv="fp8")
            o["capacity"] = cap
            o["core_done"] = KL.k3_core_done(o["tf"], cap)       # V-16: TF 2k + 8k AND a verified pool (MATCH) AND F
            k3_core_marker(o, b, cap)                     # V-15: the live log shows load OK / not OK within ~30 min
            skipped = o["k3_optional_skipped"] = []
            kv_sweep(o, s, cap, ("sw20k", "sw40k"), gate=k3_optional_ok, skipped=skipped)   # core-plus priority
            kv_sweep(o, s, cap, ("sw60k", "sw100k"), gate=k3_optional_ok, skipped=skipped)
            if k3_optional_ok(2 * TF_EST["40k"] + 300):
                o["tf"].update(run_item(o, s, "tf_40k", lambda: tf_suite("K3", s, ["40k"])))
                if _tf_stop(o, o["tf"]):
                    return
            else:
                o["tf"]["40k"] = {"status": "SKIPPED_K3_RESERVE"}
                skipped.append("tf_40k")
            admission_at_best(o, s, cap, gate=k3_optional_ok, skipped=skipped)
            if k3_optional_ok(120):
                o["arith"] = run_item(o, s, "arith", lambda: arith_probe(s.port))   # K-12: recorded, not gated
            else:
                o["arith"] = {"status": "SKIPPED_K3_RESERVE"}
                skipped.append("arith")
        return k_body(b, fn)


    def gpu_health_gate(timeout=180):
        """V-11: after a device fault, before the next boot: a fresh CUDA context on every GPU in a subprocess (a 1024 x 1024
        matmul + synchronize per card, with a timeout), nvidia-smi's volatile uncorrected ECC count, and the memory released
        (< 150 MiB used per card). ok False -> ABORT_DEVICE_FAULT: no later boot."""
        rec, py = {}, None
        kv = (MERLIN_ENG or {}).get("kv") or {}
        if kv.get("MERLIN_VENV"):
            py = kv["MERLIN_VENV"] + "/bin/python"
        code = ("import torch\nn = torch.cuda.device_count()\nfor i in range(n):\n    a = torch.ones(1024, 1024, device=f'cuda:{i}')\n"
                "    b = a @ a\n    torch.cuda.synchronize(i)\n    assert float(b[0, 0]) == 1024.0\nprint('HEALTH_OK', n)\n")
        try:
            r = subprocess.run([py or sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"})
            rec["probe"] = {"rc": r.returncode, "out": (r.stdout + r.stderr)[-600:]}
            probe_ok = r.returncode == 0 and "HEALTH_OK 4" in r.stdout
        except Exception as e:
            rec["probe"], probe_ok = {"error": repr(e)[:300]}, False
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,ecc.errors.uncorrected.volatile.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=60).stdout
            ecc = [x.split(",")[1].strip() for x in q.strip().splitlines() if "," in x]
            rec["ecc_uncorrected"] = ecc
            ecc_ok = len(ecc) == 4 and all(not v.isdigit() or int(v) == 0 for v in ecc)
        except Exception as e:
            rec["ecc_error"], ecc_ok = repr(e)[:200], False
        mem = L.gpu_mem() or []
        rec["gpu_mem"] = mem
        released = len(mem) == 4 and all(u < 150 for _, u, _ in mem)
        rec.update(probe_ok=probe_ok, ecc_ok=ecc_ok, released=released, ok=bool(probe_ok and ecc_ok and released))
        return rec


    def health_after(rec, tries, who):
        """V-11: a device fault in `who` -> the health gate; a failure sets ABORT_DEVICE_FAULT (every later boot skipped)."""
        if not ((rec or {}).get("device_fault") or any(t.get("device_fault") for t in (tries or []))):
            return True
        h = gpu_health_gate()
        K1.setdefault("gpu_health", {})[who] = h
        if h.get("ok") is not True:
            ABORT["why"] = f"ABORT_DEVICE_FAULT after {who}"
            GATES["ABORT_DEVICE_FAULT"] = (False, {"after": who, **{k: h.get(k) for k in ("probe_ok", "ecc_ok", "released")}})
            return False
        K1.setdefault("device_fault_recovered", []).append(who)
        return True


    def run_k(b, body):
        """One klee100 boot through merlin_boot (boot gate: GPU empty, engine processes gone, page cache dropped, Merlin's
        host-RAM rule; the inner 0.25 GiB OOM retry only outside the ladders; evidence + oracle gates) with the per-boot cap.
        R1-3: free memory per card = min(nvidia-smi, the worker's own torch free memory of the same rank), both recorded."""
        BOOT["tag"] = b["name"]
        o = merlin_boot(b["name"], b.get("kv", "bf16"), b["port"], K100["ready_sec"], watched(body, CAP_MIN[b["key"]] * 60, b["name"]),
                        need=(), expected_rows=4)
        LIVE["out"] = o
        if LIVE.get("rec") is not None:
            LIVE["rec"].pop("live_body", None)                 # R1-12: the full record replaces the live copy
        try:
            why = str((o.get("boot_gate") or {}).get("why") or "")
            if o.get("status") == "GPU_NOT_EMPTY" and why.startswith(("HOST_RAM", "ENGINE_PROCS")):
                o["status"] = "HOST_RAM_GATE_FAILED" if why.startswith("HOST_RAM") else "ENGINE_PROCS_LEFT"
            ln = o.get("launch_name") or b["name"]
            o.update(boot=b["key"], title=b["title"], plan=(R["setup"].get("k100_plans") or {}).get(ln),
                     load_monitor=LOADMON.get(ln), planned_kv_bytes=b["kv_bytes"], boot_name=b["name"])
            srv = (o.get("servers") or {}).get(ln) or {}
            smi = KL.free_by_card([srv.get("gpu_mem_after_ready"), srv.get("gpu_mem_end"),
                                   o.get("gpu_after_ready"), o.get("gpu_end"), o.get("gpu_peak_rows")])
            wf = KL.worker_free([o.get("workers"), o.get("workers_end")])
            o["free_mib_by_card_smi"], o["free_mib_by_card_worker"] = smi, wf
            o["free_gap_mib"] = {i: round(smi[i] - wf[i], 1) for i in smi if i in wf}
            k0_gap = None if b["key"] == "K0" else (RECS.get("K0") or {}).get("free_gap_mib")
            o["free_mib_by_card_all"], o["free_source"] = KL.combine_free(smi, wf, k0_gap)
            o["oom_at_launch"] = any(t.get("oom") for t in (o.get("tries") or []))
            # K-6: a device fault (Xid / illegal memory access / misaligned address) stops the ladder; R1-8: the whole log
            o["device_fault"] = bool(o.get("xid_in_log")) or bool(srv.get("xid"))
            o["oom_in_body"] = o.get("status") != "LAUNCH_FAILED" and bool(o.get("oom_in_log"))
        except Exception as e:
            keep_after_error(o, e)
        return o


    def ref_load():
        """NB0's W4 reference (dataset q27-e25-nb0-tf-ref): every file's sha256 vs the build-time SHA256SUMS, each array's
        ids sha vs NB0's recorded prompt sha; loaded as NB0 TF records (TFR['A'] / TFR['C']) and decode rows (SCR['A'])."""
        out = {"dir": str(REF_DIR), "ok": False}
        if REF_DIR is None:
            out["why"] = "the dataset q27-e25-nb0-tf-ref is not attached"
            return out
        bad = []
        for n, want in K100_PIN["ref_sha256"].items():
            p = REF_DIR / n
            if not p.is_file():
                bad.append(f"missing {n}")
            elif hashlib.sha256(p.read_bytes()).hexdigest() != want:
                bad.append(f"sha256 {n}")
        if bad:
            return dict(out, bad=bad, why=f"{len(bad)} reference files missing / changed")
        meta = json.loads((REF_DIR / "nb0_reference.json").read_text())
        for key, bands in meta["tf"].items():
            for band, rec in bands.items():
                for run in ("fresh", "repeat"):
                    z = np.load(REF_DIR / rec[run]["file"])
                    arr = {k: z[k] for k in z.files}
                    if ids_sha(arr["ids"]) != rec["prompt_sha256"]:
                        bad.append(f"ids {rec[run]['file']}")
                    TFR.setdefault(key, {}).setdefault(band, {})[run] = {
                        "meta": {"boot": key, "band": band, "run": run, "k": rec[run]["k"], "prompt_len": rec["prompt_len"],
                                 "prompt_sha256": rec["prompt_sha256"], "error": None, "source": "NB0 (dataset)"}, "arr": arr}
        for band, rec in meta["sc"]["A"].items():
            z = np.load(REF_DIR / rec["cold"])
            SCR.setdefault("A", {})[band] = {"P": rec["prompt_len"], "ids": z["ids"], "cold": {
                "meta": {"error": None, "prompt_sha256": rec["prompt_sha256"], "source": "NB0 (dataset)"}, "tok": [int(x) for x in z["tok"]],
                "lp": z["lp"].astype(np.float64), "top1": z["top1"].astype(np.int64), "top1_lp": z["top1_lp"].astype(np.float64),
                "top2_lp": z["top2_lp"].astype(np.float64)}}
        REF["meta"] = meta
        return dict(out, ok=not bad, bad=bad, why="; ".join(bad) or None, files=len(K100_PIN["ref_sha256"]),
                    source=meta.get("source"), missing=meta.get("missing"))


    def prompt_identity():
        """The prompts the klee100 boots send = NB0's (ids sha). The klee100 tokenizer.json differs in bytes from Merlin's (its
        BPE merges are serialised differently), so the rendered ids are compared; a mismatch puts NB0's exact ids in place
        (the comparison needs identical token sequences, and the reference ids are what A / C saw)."""
        out = {"tf": {}, "sc": {}}
        for band in NB0["tf_bands"]:
            ref = ((TFR.get("A") or {}).get(band) or {}).get("fresh")
            mine = PROMPTS[band][NB0["tf_variant"]]
            same = bool(ref) and ids_sha(mine) == ref["meta"]["prompt_sha256"]
            if ref and not same:
                PROMPTS[band][NB0["tf_variant"]] = [int(x) for x in ref["arr"]["ids"]]
            out["tf"][band] = {"rendered_equals_nb0": same, "used": "rendered" if same else ("nb0_ids" if ref else "rendered_no_reference")}
        for band, v in NB0["sc_variant"].items():
            ref = (SCR.get("A") or {}).get(band)
            mine = PROMPTS[band][v]
            same = bool(ref) and ids_sha(mine) == ref["cold"]["meta"]["prompt_sha256"]
            if ref and not same:
                PROMPTS[band][v] = [int(x) for x in ref["ids"]]
            out["sc"][band] = {"rendered_equals_nb0": same, "used": "rendered" if same else ("nb0_ids" if ref else "rendered_no_reference")}
        osha = [psha(p)[:16] for p in ORACLE_P + LONG_P]            # RC2's own prompt_sha formula (16 hex)
        out["rc2_oracle_prompts_match"] = {k: [r["prompt_sha"] for r in v] == osha for k, v in RC2_REF["oracle_rows"].items()}
        for k, v in RC2_REF["oracle_rows"].items():                  # RC2's recorded W4 rows: references only for identical prompts
            if out["rc2_oracle_prompts_match"][k]:
                ROWS[k] = [{"text": r["text"], "tokens": r["tokens"], "margins": r["margins"], "error": r["error"]} for r in v]
        out["all_rendered_equal_nb0"] = all(x["rendered_equals_nb0"] for d in (out["tf"], out["sc"]) for x in d.values())
        return out


    def _arr(key, band, run="fresh"):
        x = ((TFR.get(key) or {}).get(band) or {}).get(run)
        return x if x and x.get("arr") is not None else None


    def cmp_pair(ref, cand, y=None):
        if not ref or not cand:
            return {"status": "INCONCLUSIVE_MISSING_TF", "label_full": "INCONCLUSIVE_MISSING", "label_last256": "INCONCLUSIVE_MISSING"}
        lst, ful = cmp_last256(ref, cand), cmp_full(ref["arr"], cand["arr"])
        e = {"last256": lst, "full": ful, "shift": shift_check(ref["arr"], cand["arr"]),
             "label_full": KL.tf_label(ful), "label_last256": KL.tf_label(lst)}
        if y is not None:
            e["yardstick_full"] = KL.yardstick(ful, (y or {}).get("full"))
            e["yardstick_last256"] = KL.yardstick(lst, (y or {}).get("last256"))
        return e


    def tf_skips(tf):
        """K-5: {band: the SKIPPED_* status of boot K's TF record (band level, else fresh / repeat)} for tf_reading."""
        out = {}
        for band, x in (tf or {}).items():
            x = x or {}
            st = [x.get("status")] + [(x.get(r) or {}).get("status") for r in ("fresh", "repeat")]
            st = next((v for v in st if isinstance(v, str) and v.startswith("SKIPPED")), None)
            if st:
                out[band] = st
        return out


    def k100_analysis():
        """TF divergence of klee100 vs W4 (NB0 A / C) per band with NB0's A/A floor and its FP8-vs-BF16 yardstick, klee100's
        own A/A, MTP on vs off and cross-boot determinism, self-consistency and greedy-decode parity vs A, capacity, speed."""
        meta = REF.get("meta") or {}
        Y = (meta.get("yardstick") or {})
        pairs, perb = Y.get("pairs") or {}, Y.get("per_boot") or {}
        yb = lambda pk, band: ((pairs.get(pk) or {}).get("bands") or {}).get(band)
        an = {"reference": {k: K1.get("reference", {}).get(k) for k in ("ok", "why", "bad", "missing")},
              "prompt_identity": K1.get("prompt_identity"), "tf_k_vs_A": {}, "tf_k0_vs_A": {}, "tf_k0_vs_k": {}, "tf_headline": {},
              "thresholds": {"screen": KL.SCREEN, "triage": KL.TRIAGE}}
        pa = perb.get("A") or {}
        for band in NB0["tf_bands"]:
            a_, k0_, k_ = _arr("A", band), _arr("K0", band), _arr("K", band)
            e = cmp_pair(a_, k_, yb("AB", band))                       # K vs A: the weights AND the ladder pin
            kr = _arr("K", band, "repeat")
            floor = {"nb0_A_aa_last256": (pa.get("aa") or {}).get(band), "nb0_A_aa_full": (pa.get("aa_full") or {}).get(band),
                     "k_aa_last256": cmp_last256(k_, kr) if k_ and kr else None,
                     "k_aa_full": cmp_full(k_["arr"], kr["arr"], bins=False) if k_ and kr else None,
                     "nb0_cross_request_2k_prefix": (pa.get("prefix_consistency") or {}).get("2k~8k"),
                     "nb0_prefix_consistency": pa.get("prefix_consistency")}
            e["noise_floor"] = floor
            an["tf_k_vs_A"][band] = e
            an["tf_k0_vs_A"][band] = w = cmp_pair(a_, k0_, yb("AB", band))     # R2-4: K0 = A's exact config: the weights only
            an["tf_k0_vs_k"][band] = cmp_pair(k0_, k_)                         # K0 vs K: the pin only (same weights)
            an["tf_headline"][band] = dict(w if k0_ else e, noise_floor=floor,
                                           headline_source="K0_vs_A (weights only)" if k0_ else "K_vs_A (weights + ladder pin)")
        an["tf_k3_vs_A"], an["tf_k3_vs_k_fp8mtp_vs_bf16nomtp"], an["tf_k3_vs_k2"] = {}, {}, {}
        for band in ("2k", "8k", "40k"):                     # R3-1 / V-6: K3 (FP8, MTP K3, seqs 16) vs A / K / K2, the same ids
            an["tf_k3_vs_A"][band] = dict(cmp_pair(_arr("A", band), _arr("K3", band), yb("AB", band)), nb0_fp8_vs_bf16_AB=yb("AB", band))
            an["tf_k3_vs_k_fp8mtp_vs_bf16nomtp"][band] = dict(cmp_pair(_arr("K", band), _arr("K3", band), yb("AB", band)),
                                                              meaning="FP8 KV + MTP K3 / seqs 16 vs BF16 no-MTP / seqs 8 (not the KV effect alone)")
            an["tf_k3_vs_k2"][band] = dict(cmp_pair(_arr("K2", band), _arr("K3", band), yb("CD", band) or yb("AB", band)),
                                           nb0_fp8_vs_bf16_AB=yb("AB", band), nb0_fp8_vs_bf16_CD=yb("CD", band),
                                           meaning="the FP8 KV effect under MTP (same MTP K3 / seqs 16 / captures; only the KV dtype and pin differ)")
        an["tf_k3_vs_A_reading"] = KL.tf_reading(an["tf_k3_vs_A"], tf_skips((RECS.get("K3") or {}).get("tf")), bands=("2k", "8k"))  # V-2 / V-3
        an["tf_k3_vs_k_fp8mtp_vs_bf16nomtp_reading"] = KL.tf_reading(an["tf_k3_vs_k_fp8mtp_vs_bf16nomtp"], None, bands=("2k", "8k"))
        an["tf_k3_vs_k2_reading"] = KL.tf_reading(an["tf_k3_vs_k2"], None, bands=("2k", "8k"))
        r_ = an["tf_k3_vs_k2_reading"]
        an["fp8_headline"] = dict(r_, verdict=(r_["reading"] if r_.get("reading") in KL.TF_RANK else
                                               "INCONCLUSIVE_NO_K2" if not RECS.get("K2") or not _arr("K2", "2k") else "INCONCLUSIVE_" + str(r_.get("reading"))),
                                  source="K3 vs K2 (V-6)", mtp_parity_k2_vs_k_40k=None)
        an["nb0_fp8_vs_bf16_CD_40k"] = yb("CD", "40k")
        skips = {**tf_skips((RECS.get("K0") or {}).get("tf")), **tf_skips((RECS.get("K") or {}).get("tf"))}
        an["tf_reading"] = dict(KL.tf_reading(an["tf_headline"], skips),
                                sources={b: v["headline_source"] for b, v in an["tf_headline"].items()})
        an["tf_k2_vs_C_40k"] = cmp_pair(_arr("C", "40k"), _arr("K2", "40k"), yb("CD", "40k"))
        an["tf_k2_vs_k_40k_mtp_effect"] = dict(cmp_pair(_arr("K", "40k"), _arr("K2", "40k")), nb0_AC_40k=yb("AC", "40k"),
                                               meaning="MTP on / off parity check (K2 vs K)")
        k0, k = _arr("K0", "90k"), _arr("K", "90k")
        an["tf_k0_vs_k_90k_cross_boot"] = cmp_full(k0["arr"], k["arr"], bins=False) if k0 and k else {"status": "INCONCLUSIVE_MISSING_TF"}
        sc = {}
        for band in NB0["sc_bands"]:
            a, x = (SCR.get("A") or {}).get(band), (SCR.get("K") or {}).get(band)
            r = {"nb0_A_decode_vs_tf": ((perb.get("A") or {}).get("sc") or {}).get(band)}
            if x:
                r["k_decode_vs_tf"] = sc_compare(x["cold"], x["tf"], x["P"])
                r["k_cold_vs_cached"] = decode_compare(x["cold"], x["cached"])
                same_prompt = a is not None and x["cold"]["meta"].get("prompt_sha256") == a["cold"]["meta"]["prompt_sha256"]
                r["greedy_vs_A"] = decode_compare(a["cold"], x["cold"]) if same_prompt else {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
            sc[band] = r
        an["self_consistency"] = sc
        an["capacity"] = {k: (RECS.get(k) or {}).get("capacity") for k in ("K3", "K0", "K", "K2")}
        an["memory_census"] = {k: dict((RECS.get(k) or {}).get("memory_census") or {}, free_mib_by_card=(RECS.get(k) or {}).get("free_mib_by_card_all"),
                                       kv_bytes=(RECS.get(k) or {}).get("kv_bytes")) for k in ("K0", "K", "K2")}
        k2 = RECS.get("K2") or {}
        an["speed"] = {"windows": {w: {x: (v or {}).get(x) for x in ("status", "tok_s", "gen_tokens", "running_min", "running_max",
                                                                     "running_eq_level", "step_ms", "tick_ms", "cycle_ms", "mean_acceptance_length",
                                                                     "accept_per_draft", "accept_rate_per_position_incl_admission", "preemptions",
                                                                     "kv_usage_max", "prefill_tok_s_uncached")}
                                   for w, v in (k2.get("steady") or {}).items()},
                       "admission_64k": {x: (k2.get("admission_64k") or {}).get(x) for x in ("status", "n", "all_resident", "admission_sec",
                                                                                              "preemptions", "snapshot", "window")},
                       "long_120k": {x: ((RECS.get("K") or {}).get("long_120k") or {}).get(x) for x in
                                     ("verdict", "tokens", "finish", "ttft_sec", "prefill_tok_s", "decode_tok_s", "window_garbage")},
                       "references": RC2_WINDOWS}
        an["speed_k3"] = {w: {x: (v or {}).get(x) for x in ("status", "tok_s", "gen_tokens", "running_eq_level", "tick_ms", "cycle_ms",
                                                            "mean_acceptance_length", "accept_rate_per_position_incl_admission",
                                                            "preemptions", "kv_usage_max")}
                          for w, v in ((RECS.get("K3") or {}).get("steady") or {}).items()}
        an["speed_k3"]["admission_64k"] = {x: ((RECS.get("K3") or {}).get("admission_64k") or {}).get(x) for x in ("status", "n", "all_resident", "window")}
        return an


    def boot_summary(o):
        o = o or {}
        return {k: o.get(k) for k in ("status", "correctness", "config_ok", "early_stop", "config_mismatch", "kv_bytes", "tries",
                                      "oom_at_launch", "oom_in_body", "device_fault", "error", "status_note", "postprocess_error")}


    REF.update(ok=False)
    try:                                     # K-4: a corrupt reference is a failed gate (CELL 8), never a crash or a boot
        K1["reference"] = ref_load()
    except Exception as e:
        K1["reference"] = {"ok": False, "why": f"REF_LOAD_ERROR {type(e).__name__}: {e}"[:400]}
    try:
        K1["prompt_identity"] = prompt_identity()
    except Exception as e:
        K1["prompt_identity"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    log("K100 reference", json.dumps({k: K1["reference"].get(k) for k in ("ok", "why", "files")}), "| prompts",
        json.dumps({k: v for k, v in K1["prompt_identity"].items() if k != "sc"})[:600])
    save()


if not RERUN:
    # ==================================================================== CELL 8 — E25 klee100 main: gates, K0 -> pin ladder -> K -> K2, report
    def tf_brief(e):
        e = e or {}
        return {"label_full": e.get("label_full"), "label_last256": e.get("label_last256"), "yardstick_full": e.get("yardstick_full"),
                "headline_source": e.get("headline_source"),
                "full": {y: (e.get("full") or {}).get(y) for y in ("status", "n", "expected", "mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")},
                "last256": {y: (e.get("last256") or {}).get(y) for y in ("status", "mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")},
                "noise_floor": {k: v for k, v in (e.get("noise_floor") or {}).items() if k in ("nb0_A_aa_full", "nb0_A_aa_last256", "nb0_prefix_consistency")}}


    def boot_brief(o):
        o = o or {}
        return dict(boot_summary(o), **{k: o.get(k) for k in ("launch_name", "port", "ready_sec", "death", "died_in", "server_status",
                                                               "body_cap_sec", "boot_watchdog", "quality", "core_done", "free_source",
                                                               "free_mib_by_card_smi", "free_mib_by_card_worker", "free_mib_by_card_all")})


    if not RERUN and not os.environ.get("RC2_UNIT"):
        K_REC, K2_REC, K3_REC, MAIN_ERROR, an = None, None, None, None, {}
        FINAL, verdict = {"label": "NOT_REACHED"}, {}
        try:                                       # R1-11: whatever happens, the verdict, save() and the fallback are written
            m = R["sections"]["A0"]["merge"]
            GATES["INPUTS"] = (bool(MERLIN_OK), None if MERLIN_OK else json.dumps(R["inputs"])[:400])
            GATES["OVERLAY_MERGE_K6"] = (m.get("ok") is True, m.get("why") or m.get("bad"))
            # K-4: the NB0 reference and the prompt identity gate the boots (before install / K0); R2-11: RC2 oracle ids recorded
            GATES.update(KL.reference_gates(K1.get("reference"), K1.get("prompt_identity"), NB0["tf_bands"], list(NB0["sc_variant"])))
            K1["rc2_oracle_identity"] = KL.oracle_identity(K1.get("prompt_identity"))
            if all(ok for ok, _ in GATES.values()):
                rec = step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")})(merlin_install())})
                GATES["MERLIN_INSTALL"] = (rec.get("status") == "DONE", rec.get("error"))
            if all(ok for ok, _ in GATES.values()):
                rec = step("A0", "patch_60490", 3, apply_60490)
                GATES["PATCH_60490"] = (rec.get("status") == "DONE" and rec.get("ok") is True, rec.get("error"))
            if all(ok for ok, _ in GATES.values()):
                ok, why = sha_gate(max(60, min(2400, left() - 86 * 60)))        # R1-7: the wait never starves K / K2
                R["setup"]["sha_gate"] = {"ok": ok, "why": why}
                GATES["DIGESTS_K7_K3"] = (ok, why)
                save()
            RUN_K3, RUN_BF16 = K100_PROFILE in ("ALL", "K3_ONLY"), K100_PROFILE in ("ALL", "BF16_ONLY")   # V-9
            if not RUN_K3:
                RECS["K3"] = K3_REC = {"status": "NOT_RUN", "why": f"profile {K100_PROFILE}"}
            if RUN_K3 and all(ok for ok, _ in GATES.values()):     # K3 FIRST (user decision): FP8 KV, its own ladder
                K1["k3_pins"] = K3_PINS
                try:                                   # R3-1: the scales sidecar (the installed Merlin tree first); not a gate
                    K1["scales"] = k100_scales(((MERLIN_ENG or {}).get("kv") or {}).get("MERLIN_TREE"))
                except Exception as e:
                    K1["scales"] = {"ok": False, "why": f"{type(e).__name__}: {e}"[:300]}
                if (K1.get("scales") or {}).get("ok") is not True:
                    RECS["K3"] = K3_REC = {"status": "SKIPPED_NO_SCALES", "why": (K1.get("scales") or {}).get("why")}
                else:
                    def k3_launch(kvb, i):         # R1-4 / R1-9: no inner retry, a fresh name and port per rung
                        b3 = k1_boot("K3", f"k100_K3_fp8_mtp3_pin{int(round(kvb / 2 ** 30 * 100))}_r{i}", True, K100["seqs_k3"], COMPILE_HI,
                                     kvb, f"K3: klee100, FP8 E4M3 QSA KV, MTP K3, seqs 32 + captures to 64, pin {round(kvb / 2 ** 30, 3)} GiB "
                                          f"(rung {i} of the K3 ladder)", port=PORTS["K3"] + 10 * i, ladder=True, kv="fp8")
                        return step("K3", b3["name"], EST_MIN["K3"], lambda: run_k(b3, body_k3(b3)))
                    # rev 3: every step of K3_PINS is tried from index 0 (the two raised rungs from the v1 GPU reading, then v1's
                    # 5,270,142,976, B3, the FP8 base); an extra rung (launch OOM) or ONE body-OOM retry before the core only
                    # while a K3 core + the K0 + K + K2 reserves still fit
                    k3_fits = lambda: left() >= EST_MIN["K3"] * 60 + later_reserve_sec()
                    K1["k3_first_rung"] = K3_PINS["steps"][0]
                    K3_REC, K1["k3_tries"] = KL.ladder_run(K3_PINS["steps"], k3_launch, can_try=lambda i, p: k3_fits(),
                                                           body_retry=lambda r: KL.k3_body_oom_retry_ok(r, k3_fits))
                    RECS["K3"] = K3_REC
                    K1["k3_booted_pin"] = K3_REC.get("kv_bytes") if K3_REC.get("status") not in ("LAUNCH_FAILED", None) else None
                    if K3_REC.get("core_done") is not True:   # V-15: the core failed (rungs exhausted / device fault / a core gate)
                        K1["k3_core_failed"] = {k: K3_REC.get(k) for k in ("status", "correctness", "early_stop", "config_mismatch", "death",
                                                                            "device_fault", "kv_bytes")}
                        print("@@K3_CORE_FAILED " + json.dumps(K1["k3_core_failed"], default=str), flush=True)
                        if K100_PROFILE == "K3_ONLY":         # the load smoke: no further boot or attempt; verdict + fallback below
                            K1["k3_early_exit"] = "K3 core failed in K3_ONLY: no further boots; the verdict and the fallback are written"
                    if RUN_BF16:
                        health_after(K3_REC, K1.get("k3_tries"), "K3")             # V-11
                save()
            if RUN_BF16 and all(ok for ok, _ in GATES.values()):
                b0 = k1_boot("K0", "k100_K0_probe_bf16_mtp0", False, K100["seqs_k"], FP8_CAPTURES, KV_DEFAULT,
                             "K0 probe: klee100, BF16 KV, MTP off, Merlin's BF16 pin, seqs 8 + FP8 captures (= NB0 A's config)")
                r0 = step("K", b0["name"], EST_MIN["K0"], lambda: run_k(b0, body_k0(b0)))
                RECS["K0"] = r0
                cli0 = (K1.get("cli_flags") or {}).get(b0["name"]) or {}
                if cli0.get("ok") is False:        # K-9: the first boot's --help lacks a required flag: no launch happened
                    GATES["CLI_FLAGS"] = (False, cli0.get("why"))
                GATES["BOOT_K0_PROBE"] = (KL.boot_clean(r0), {k: r0.get(k) for k in ("status", "correctness", "early_stop", "config_mismatch", "error")})
                health_after(r0, None, "K0")                                       # V-11
                save()
            if RUN_BF16 and all(ok for ok, _ in GATES.values()):
                lad0 = KL.pin_ladder(r0.get("free_mib_by_card_all") or r0.get("free_mib_by_card"), KV_DEFAULT, K100["fracs"],
                                     K100["safety_mib"], K100["mtp_reserve_mib"])
                k3m = (K3_REC or {}).get("free_mib_by_card_all")
                lad3 = (KL.pin_ladder(k3m, int(K3_REC["kv_bytes"]), K100["fracs"], K100["safety_mib"], 0)      # K3 holds the MTP draft
                        if KL.free_valid4(k3m) and K3_REC.get("status") != "LAUNCH_FAILED" and K3_REC.get("kv_bytes") else None)
                K1["pin_ladder"] = lad = KL.ladder_min(lad0, lad3)          # R1-3 readings of K0 AND K3: per frac the lower pin
                log("K100 pin ladder", json.dumps({k: lad.get(k) for k in ("ok", "steps", "headroom_mib_per_card", "binding_card", "why")}))
                steps_ = lad["steps"] or [{"frac": None, "kv_bytes": KV_DEFAULT, "gib": round(KV_DEFAULT / 2 ** 30, 3), "note": "no ladder gain: default pin"}]

                def k_launch(kvb, i):              # R1-9: a fresh port per rung; R1-4: no inner retry (ladder=True)
                    st = next(x for x in steps_ if x["kv_bytes"] == kvb)
                    b = k1_boot("K", f"k100_K_bf16_mtp0_pin{int(round(kvb / 2 ** 30 * 100))}_r{i}", False, K100["seqs_k"], FP8_CAPTURES,
                                kvb, f"K: klee100, BF16 KV, MTP off, pin {st['gib']} GiB (frac {st['frac']} of the K0 headroom)",
                                port=PORTS["K"] + 10 * i, ladder=True)
                    return step("K", b["name"], EST_MIN["K"], lambda: run_k(b, body_k(b)))
                # K-6 / R1-4: next rung on a launch OOM only, never >= a pin that OOMed, an extra rung only with K + K2 still in budget
                K_REC, K1["k_tries"] = KL.ladder_run([x["kv_bytes"] for x in steps_], k_launch,
                                                     can_try=lambda i, p: left() >= EST_MIN["K"] * 60 + k2_reserve_sec())
                RECS["K"] = K_REC
                save()
                k_oom = KL.k_oom_after_core(K_REC)
                if not health_after(K_REC, K1.get("k_tries"), "K"):                # V-11: no K2 after an unhealthy device fault
                    k_oom = False
                    K_REC = dict(K_REC, k2_blocked="ABORT_DEVICE_FAULT")                                 # R2-3: TF_OOM / BODY_OOM after the core
                if KL.boot_clean(K_REC) or k_oom:
                    lower = None
                    if k_oom:                      # K died of OOM after its core: K2 takes the next lower step (else Merlin's pin)
                        lower = next((x["kv_bytes"] for x in steps_ if x["kv_bytes"] < (K_REC.get("kv_bytes") or 0)), KV_DEFAULT)
                    pin2 = KL.k2_pin_from_k(K_REC.get("free_mib_by_card_all") or K_REC.get("free_mib_by_card"), K_REC.get("kv_bytes"),
                                             K100["mtp_reserve_mib"], K100["k2_min_free_mib"], KV_DEFAULT, lower_pin=lower)
                    K1["k2_pin"] = pin2
                    K1["k2_pins"] = pins2 = KL.k2_pins(pin2["kv_bytes"], KV_DEFAULT)              # K-6: K2's own ladder
                    k_ready = k_ready_sec()

                    def k2_launch(kvb, i):
                        b2 = k1_boot("K2", f"k100_K2_bf16_mtp3_pin{int(round(kvb / 2 ** 30 * 100))}_r{i}", True, K100["seqs_k2"], COMPILE_HI,
                                     kvb, f"K2: klee100, BF16 KV, MTP K3, seqs 16 + captures to 64, pin {round(kvb / 2 ** 30, 3)} GiB ({pin2['rule']})",
                                     port=PORTS["K2"] + 10 * i, ladder=True)
                        plan = KL.k2_plan(left(), k_ready, K100["k2_body_min_min"] * 60, CAP_MIN["K2_MAX"] * 60, K100["ready_factor_k2"],
                                          K100["teardown_sec"], K100["step_margin_sec"])      # R1-2: from K's measured ready time
                        K1.setdefault("k2_budget", []).append(dict(plan, kv_bytes=kvb))
                        if plan["fits"] is not True:       # V-10: K2's minimum body does not fit: no launch at all
                            return {"status": "SKIPPED_BUDGET_K2_MINIMUM", "plan": plan}
                        CAP_MIN["K2"] = plan["body_cap_sec"] / 60
                        return step("K2", b2["name"], plan["est_min"], lambda: run_k(b2, body_k2(b2)))   # step(): SKIPPED_BUDGET
                    K2_REC, K1["k2_tries"] = KL.ladder_run(pins2, k2_launch, known_oom=[t["kv_bytes"] for t in K1["k_tries"]
                                                                                        if t.get("oom_at_launch")])
                    RECS["K2"] = K2_REC
                else:
                    RECS["K2"] = K2_REC = {"status": "SKIPPED_DEPENDENCY", "why": f"boot K not clean: {boot_summary(K_REC)}"}
                save()
        except Exception as e:
            MAIN_ERROR = {"error": f"{type(e).__name__}: {e}"[:800], "tb": traceback.format_exc()[-2500:]}
            R["errors"].append("CELL 8 main: " + MAIN_ERROR["error"])
        finally:
            try:
                try:
                    R["analysis"] = an = k100_analysis()
                except Exception as e:
                    R["analysis"] = an = {"error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-2500:]}
                FINAL = KL.final_label(GATES, K_REC, K2_REC, an.get("tf_reading"), RECS.get("K3") or {"status": "NOT_RUN"},
                                       an.get("tf_k3_vs_A_reading"), profile=K100_PROFILE)
                if MAIN_ERROR:
                    FINAL = dict(FINAL, label="MAIN_ERROR + " + FINAL["label"], main_error=MAIN_ERROR["error"])
                R["minutes"] = round((time.time() - T_START) / 60, 1)
                R["host_phase"] = HOST_PHASE
                verdict = {"label": FINAL, "main_error": MAIN_ERROR, "gates": {k: {"ok": v[0], "why": v[1]} for k, v in GATES.items()},
                           "rc2_oracle_identity": K1.get("rc2_oracle_identity"),
                           "boots": {"K3": boot_brief(RECS.get("K3")), "K0": boot_brief(RECS.get("K0")), "K": boot_brief(K_REC),
                                     "K2": boot_brief(K2_REC)},
                           "k3_pins": K1.get("k3_pins"), "k3_tries": K1.get("k3_tries"), "k3_booted_pin": K1.get("k3_booted_pin"),
                           "k3_first_rung": K1.get("k3_first_rung"), "k3_core_done": K1.get("k3_core_done"),
                           "k3_core_failed": K1.get("k3_core_failed"), "k3_early_exit": K1.get("k3_early_exit"), "k3_optional_skipped": (RECS.get("K3") or {}).get("k3_optional_skipped"),
                           "sweep": {k: {"rows": (RECS.get(k) or {}).get("sweep"), "best": (RECS.get(k) or {}).get("sweep_best")} for k in ("K3", "K2")},
                           "k3_scales": {"file": K1.get("scales"), "applied": {x: ((RECS.get("K3") or {}).get("scales") or {}).get(x) for x in
                                         ("ok", "per_rank", "missing_ranks", "duplicated_ranks", "bad_values", "mtp_default_scale", "why")}},
                           "tf_k3_vs_A_reading": an.get("tf_k3_vs_A_reading"),
                           "fp8_headline": {x: (an.get("fp8_headline") or {}).get(x) for x in ("verdict", "reading", "missing", "source")},
                           "tf_k3_vs_k2_reading": an.get("tf_k3_vs_k2_reading"),
                           "tf_k3_vs_k_fp8mtp_vs_bf16nomtp_reading": an.get("tf_k3_vs_k_fp8mtp_vs_bf16nomtp_reading"),
                           "tf_k2_vs_k_40k_mtp_parity": tf_brief(an.get("tf_k2_vs_k_40k_mtp_effect")),
                           "tf_k3_vs_A": {b: tf_brief(e) for b, e in (an.get("tf_k3_vs_A") or {}).items()},
                           "tf_k3_vs_k2": {b: tf_brief(e) for b, e in (an.get("tf_k3_vs_k2") or {}).items()},
                           "tf_k3_vs_k_fp8mtp_vs_bf16nomtp": {b: tf_brief(e) for b, e in (an.get("tf_k3_vs_k_fp8mtp_vs_bf16nomtp") or {}).items()},
                           "scales_calibration": SCALES_CALIBRATION, "profile": K100_PROFILE, "hard_stop_min": round(BUDGET / 60),
                           "device_fault_recovered": K1.get("device_fault_recovered"), "gpu_health": K1.get("gpu_health"),
                           "predicates": {"k3": KL.boot_predicates(RECS.get("K3"), an.get("tf_k3_vs_A_reading"), fp8=True),
                                          "k": KL.boot_predicates(K_REC, an.get("tf_reading")),
                                          "k2": KL.boot_predicates(K2_REC, {"reading": (an.get("tf_k2_vs_C_40k") or {}).get("label_full")})},
                           "nb0_fp8_vs_bf16_CD_40k": an.get("nb0_fp8_vs_bf16_CD_40k"),
                           "k_tries": K1.get("k_tries"), "pin_ladder": K1.get("pin_ladder"), "k2_pin": K1.get("k2_pin"),
                           "k2_headroom_verified": (K1.get("k2_pin") or {}).get("headroom_verified"), "k2_pins": K1.get("k2_pins"),
                           "k2_tries": K1.get("k2_tries"), "k2_budget": K1.get("k2_budget"), "cli_flags": K1.get("cli_flags"),
                           "gpu_method_map": {k: method_map_summary(RECS.get(k)) for k in ("K3", "K0", "K", "K2")},
                           "probes_correct": {k: ((RECS.get(k) or {}).get("probes") or {}).get("probes_correct") for k in ("K3", "K0", "K", "K2")},
                           "arith_exact": {k: {x: ((RECS.get(k) or {}).get("arith") or {}).get(x) for x in ("exact", "contains", "n", "error")}
                                           for k in ("K3", "K", "K2")},
                           "quality": {k: (RECS.get(k) or {}).get("quality") for k in ("K", "K2")},
                           "traces": K1.get("traces"), "trace_errors": K1.get("trace_errors"),
                           "tf_reading": an.get("tf_reading"),
                           "tf_headline": {b: tf_brief(e) for b, e in (an.get("tf_headline") or {}).items()},
                           "tf_k0_vs_A": {b: tf_brief(e) for b, e in (an.get("tf_k0_vs_A") or {}).items()},
                           "tf_k0_vs_k": {b: tf_brief(e) for b, e in (an.get("tf_k0_vs_k") or {}).items()},
                           "tf_k_vs_A": {b: tf_brief(e) for b, e in (an.get("tf_k_vs_A") or {}).items()},
                           "tf_k2_vs_C_40k": tf_brief(an.get("tf_k2_vs_C_40k")),
                           "capacity": an.get("capacity"), "speed": an.get("speed"), "speed_k3": an.get("speed_k3"),
                           "minutes": R["minutes"], "abort": ABORT["why"],
                           "accuracy": "NOT_TESTED: an ARC accuracy run is a separate step"}
            except Exception as e:
                FINAL = {"label": "VERDICT_ERROR", "why": f"{type(e).__name__}: {e}"[:400]}
                verdict = {"label": FINAL, "main_error": MAIN_ERROR, "verdict_error": traceback.format_exc()[-2500:]}
            finally:
                R["verdict"] = verdict
                save()
                try:
                    (OUT / "k100_verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
                except Exception as e:
                    print("k100_verdict.json not written:", repr(e)[:200], flush=True)
                write_fallback_submission()
                print(json.dumps(verdict, default=str)[:12000])
                print(f"@@K100 {FINAL['label']}@@", flush=True)
