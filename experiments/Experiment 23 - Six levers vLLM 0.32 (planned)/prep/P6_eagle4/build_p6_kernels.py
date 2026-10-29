"""E23 P6 (EAGLE-4 feasibility on 4x L4) kernel builder: two Kaggle CPU kernels, internet ON, pinned image.
P6a foysalemonshanto/q27-e23-p6-eagle-env: the EAGLE-4 vLLM fork (vllm==0.32.0rc1.dev283+eagle.5 + eagle-harness[vllm],
    docs/install.md) as an offline wheelhouse, installed from it with no index, then CPU checks: versions, import,
    `vllm serve --help=all` flags (--linear-backend flashinfer_cutedsl, --attention-backend CUSTOM, speculative method
    eagle_harness), plugin entry points, SASS archs of the compiled libraries (sm_89 or a binary-compatible sm_80 cubin /
    PTX), max GLIBC of every .so (image: 2.35), arch gates in the eagle_harness source. Output: eagle_offline_tar.bin
    (never unpacked by Kaggle) + p6a_report.json.
P6b foysalemonshanto/q27-e23-p6-eagle-drafter: EAGLE-Corp/Qwen3.8-27B-EAGLE4 at a pinned revision (3 files; the
    .gitattributes dot-file is skipped: Kaggle Models refuse it), size + sha256 checked, flat output for a Model."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
GLIBCFIX = (HERE.parent / "P2_merlin_env" / "glibc235_fix.py").read_text(encoding="utf-8").replace("\r\n", "\n")
FLAGCHECK = Path(r"D:/ARC AGI 2/experiments/Experiment 21 - Duck config B on vLLM 0.19 + 0.32 (planned)/src/e21_flagcheck.py"
                 ).read_text(encoding="utf-8").replace("\r\n", "\n")

P6A = r'''# E23 P6a: EAGLE-4 vLLM fork offline env + CPU feasibility checks (internet ON, pinned image)
import hashlib, json, os, re, shutil, subprocess, sys, tarfile, time
from pathlib import Path
T0 = time.time()
GLIBCFIX = ''' + repr(GLIBCFIX) + r'''
FLAGCHECK = ''' + repr(FLAGCHECK) + r'''
REQS = ["eagle-harness[vllm]", "vllm==0.32.0rc1.dev283+eagle.5"]
IDX = ["https://eagle-hq.github.io/EAGLE-4/whl/cu130/", "https://download.pytorch.org/whl/cu130"]
W, T = Path("/kaggle/working"), Path("/tmp/e23p6")
PAY, LOGS = T / "eagle_offline", T / "logs"
for p in (PAY / "wheelhouse", LOGS):
    p.mkdir(parents=True, exist_ok=True)
R = {"steps": {}, "errors": [], "facts": {}}
VPY = "/tmp/eagle-off/bin/python"
STUB = T / "stub"


def sh(name, cmd, timeout=3600, ok_rc=(0,), env=None):
    t = time.time()
    with open(LOGS / f"{name}.log", "w") as f:
        r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                           env={**os.environ, **(env or {})})
    txt = (LOGS / f"{name}.log").read_text(errors="replace")
    R["steps"][name] = {"rc": r.returncode, "sec": round(time.time() - t, 1), "tail": txt[-3000:]}
    print(f"[{time.time() - T0:7.1f}s] {name}: rc {r.returncode}", flush=True)
    if r.returncode not in ok_rc:
        print(txt[-2500:], flush=True)
    return r.returncode, txt


def fail(m):
    R["errors"].append(m)
    print("ERROR:", m, flush=True)


try:
    sh("env", "python3 --version; uv --version; ldd --version | head -1; nproc; df -h /kaggle/working /tmp | cat")
    (PAY / "glibc235_fix.py").write_text(GLIBCFIX)
    (PAY / "reqs.txt").write_text("\n".join(REQS) + "\n")
    idx = " ".join(f"--extra-index-url {u}" for u in IDX)
    # 1. wheelhouse (resolved for this image's CPython 3.12 / linux x86_64, as the GPU notebook will use)
    rc, _ = sh("download", f"python3 -m pip download {' '.join(repr(x) for x in REQS)} {idx} -d {PAY}/wheelhouse "
                           f"--progress-bar off -q", timeout=5400)
    whl = sorted(p.name for p in (PAY / "wheelhouse").iterdir())
    R["facts"]["wheels"] = {"n": len(whl), "bytes": sum(p.stat().st_size for p in (PAY / "wheelhouse").iterdir()),
                            "vllm": [n for n in whl if n.startswith("vllm-")], "eagle": [n for n in whl if "eagle" in n.lower()],
                            "torch": [n for n in whl if n.startswith("torch-")], "flashinfer": [n for n in whl if "flashinfer" in n],
                            "triton": [n for n in whl if n.startswith("triton")], "sdist": [n for n in whl if not n.endswith(".whl")]}
    if rc or not R["facts"]["wheels"]["vllm"]:
        fail(f"wheelhouse download rc {rc}")
    # 2. offline install from the wheelhouse only (the GPU notebook's path): uv venv on the system CPython, no index
    (PAY / "install_eagle_offline.sh").write_text(
        "#!/usr/bin/env bash\n# usage: install_eagle_offline.sh <payload-dir> <venv-dir>  (offline: no index)\nset -euo pipefail\n"
        'PAY="$1"; V="$2"\nuv venv --python "$(command -v python3)" "$V"\n'
        'UV_OFFLINE=1 UV_PRERELEASE=allow uv pip install --python "$V/bin/python" --no-index --find-links "$PAY/wheelhouse" '
        '-r "$PAY/reqs.txt"\n'
        'echo "EAGLE_VENV=$V"\necho INSTALL OK\n')
    shutil.rmtree("/tmp/eagle-off", ignore_errors=True)
    rc, txt = sh("offline_install", f"bash {PAY}/install_eagle_offline.sh {PAY} /tmp/eagle-off", timeout=3600)
    if rc or "INSTALL OK" not in txt:
        fail("offline install failed")
    else:
        STUB.mkdir(exist_ok=True)
        if not (STUB / "libcuda.so.1").exists():
            os.symlink("/usr/local/cuda-12.8/lib64/stubs/libcuda.so", STUB / "libcuda.so.1")
        E = {"LD_LIBRARY_PATH": str(STUB), "VLLM_TARGET_DEVICE": "cuda"}
        # 3. versions + import
        rc, txt = sh("versions", f"cd /tmp && {VPY} -c \"import importlib.metadata as m, json; "
                     f"print('VERS', json.dumps({{k: m.version(k) for k in ['vllm','torch','triton','eagle-harness'] }}))\" ; "
                     f"{VPY} -m pip list 2>/dev/null | grep -i -E 'flashinfer|cutlass|cuda-|nvidia-nccl|eagle' ; "
                     f"uv pip list --python {VPY} | grep -i -E 'flashinfer|cutlass|nccl|eagle|torch|triton|vllm'", env=E, ok_rc=(0, 1))
        m = re.search(r"^VERS (.*)$", txt, re.M)
        R["facts"]["versions"] = json.loads(m.group(1)) if m else None
        rc, txt = sh("import", f"cd /tmp && {VPY} -c \"import torch, vllm, eagle_harness; print('IMPORT OK', vllm.__version__, "
                     f"torch.__version__, torch.version.cuda, eagle_harness.__file__)\"", env=E, ok_rc=(0, 1))
        R["facts"]["import_ok"] = rc == 0 and "IMPORT OK" in txt
        if not R["facts"]["import_ok"]:
            fail("import vllm / eagle_harness failed")
        # 4. CLI (v3): `vllm serve --help` cannot build its parser on a CPU box ("Failed to infer device type", v1), so the
        # EAGLE-4 launch commands are parsed with E21's flag checker (stub "cuda" platform, the fork's real parser +
        # AsyncEngineArgs), with our FP8 target and the docs' quickstart flags
        (T / "e21_flagcheck.py").write_text(FLAGCHECK)
        fp8 = "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"
        drafter = "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-eagle4/transformers/default/1"
        spec = json.dumps({"method": "eagle_harness", "model": drafter, "method_config": {"mode": "static", "node_top_k": 4,
                                                                                         "tree_depth": 14, "max_verify_width": 16}})
        base = [fp8, "--tensor-parallel-size", "4", "--max-model-len", "106496", "--gpu-memory-utilization", "0.90",
                "--max-num-seqs", "16", "--max-num-batched-tokens", "4096", "--attention-backend", "CUSTOM",
                "--speculative-config", spec, "--no-enable-prefix-caching", "--reasoning-parser", "qwen3",
                "--language-model-only", "--disable-custom-all-reduce", "--served-model-name", "q27"]
        V = (R["facts"].get("versions") or {}).get("vllm") or "0.32.0rc1"
        flags = {"eagle_fp8_tp4": {"vllm": V, "flags": base},
                 "eagle_fp8_tp4_cutedsl": {"vllm": V, "flags": base + ["--linear-backend", "flashinfer_cutedsl"]},
                 "eagle_fp8_tp4_fp8kv": {"vllm": V, "flags": base + ["--kv-cache-dtype", "fp8_e4m3"]},
                 "mtp3_fp8_tp4_control": {"vllm": V, "flags": [fp8, "--tensor-parallel-size", "4", "--max-model-len", "106496",
                                          "--max-num-seqs", "16", "--kv-cache-dtype", "fp8_e4m3", "--attention-backend",
                                          "FLASHINFER", "--speculative-config",
                                          json.dumps({"method": "mtp", "num_speculative_tokens": 3}),
                                          "--enable-prefix-caching", "--disable-custom-all-reduce"]}}
        (T / "eagle_flags.json").write_text(json.dumps(flags, indent=1))
        rc, txt = sh("flagcheck", f"cd /tmp && VLLM_USE_V2_MODEL_RUNNER=1 {VPY} {T}/e21_flagcheck.py {T}/eagle_flags.json {V}",
                     env=E, ok_rc=(0, 1, 2))
        m = re.search(r"^FLAGCHECK (.*)$", txt, re.M)
        R["facts"]["flagcheck"] = json.loads(m.group(1)) if m else None
        R["facts"]["flagcheck_lines"] = [l[:700] for l in txt.splitlines() if l.startswith("{")][:8]
        # 5. plugin registration: entry points, SpeculativeConfig method values, attention backend names
        probe = (
            "import importlib.metadata as m, json, typing\n"
            "out = {'entry_points': {g: [str(e) for e in m.entry_points(group=g)] for g in ('vllm.general_plugins', 'vllm.platform_plugins')}}\n"
            "try:\n from vllm.config import SpeculativeConfig\n hints = typing.get_type_hints(SpeculativeConfig)\n"
            " out['spec_method_type'] = str(hints.get('method'))[:1500]\n"
            " out['spec_fields'] = sorted(getattr(SpeculativeConfig, '__dataclass_fields__', {}))\n"
            "except Exception as e:\n out['spec_error'] = repr(e)[:300]\n"
            "try:\n from vllm.v1.attention.backends.registry import AttentionBackendEnum as A\n out['attn_backends'] = [x.name for x in A]\n"
            "except Exception as e:\n out['attn_error'] = repr(e)[:300]\n"
            "print('PLUGIN ' + json.dumps(out))\n")
        (T / "probe.py").write_text(probe)
        rc, txt = sh("plugin", f"cd /tmp && {VPY} {T}/probe.py", env=E, ok_rc=(0, 1))
        m = re.search(r"^PLUGIN (.*)$", txt, re.M)
        R["facts"]["plugin"] = json.loads(m.group(1)) if m else None
        sp = json.dumps(R["facts"]["plugin"] or {})
        R["facts"]["eagle_method_visible"] = "eagle_harness" in sp
        R["facts"]["custom_attention_visible"] = "CUSTOM" in sp
        # 6. compiled-library archs (sm_89, or sm_80 cubins / PTX that run on sm_89) and max GLIBC
        sh("cuobjdump_get", "python3 -m pip download --no-deps -q -d /tmp/cuobj nvidia-cuda-cuobjdump==13.2.51 && "
                            "cd /tmp/cuobj && python3 -c \"import zipfile,glob; zipfile.ZipFile(glob.glob('*.whl')[0]).extractall('x')\" && "
                            "chmod +x x/nvidia/cu13/bin/cuobjdump && x/nvidia/cu13/bin/cuobjdump --version | tail -1", ok_rc=(0, 1))
        cuo = "/tmp/cuobj/x/nvidia/cu13/bin/cuobjdump"
        site = subprocess.run([VPY, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True).stdout.strip()
        R["facts"]["site"] = site
        sos = [p for d in ("vllm", "eagle_harness", "flashinfer") for p in Path(site, d).rglob("*.so*") if p.is_file()] if site else []
        archs = {}
        for p in sos:
            r = subprocess.run([cuo, "--list-elf", str(p)], capture_output=True, text=True)
            e = sorted(set(re.findall(r"sm_(\d+)", r.stdout)))
            r2 = subprocess.run([cuo, "--list-ptx", str(p)], capture_output=True, text=True)
            x = sorted(set(re.findall(r"sm_(\d+)", r2.stdout)))
            if e or x:
                archs[str(p.relative_to(site))] = {"elf": e, "ptx": x}
        R["facts"]["archs"] = archs
        R["facts"]["archs_runnable_on_sm89"] = {k: any(a in ("80", "86", "89") for a in v["elf"]) or any(int(a) <= 89 for a in v["ptx"])
                                                for k, v in archs.items()}
        rc, txt = sh("glibc", f"python3 -I {PAY}/glibc235_fix.py --dry-run " + " ".join(str(p) for p in sos[:400]), ok_rc=(0, 2))
        m = re.search(r"^GLIBCFIX (.*)$", txt, re.M)
        g = json.loads(m.group(1)) if m else {}
        R["facts"]["glibc_over_2_35"] = {k: v.get("high_symbols") for k, v in (g.get("files") or {}).items() if v.get("high")}
        # 7. arch gates and backend code in the sources (informational, for the GPU smoke plan)
        sh("source_gates", f"cd {site} && grep -rn -E 'get_device_capability|is_device_capability|compute_capability|sm_?(80|89|90|100|103)\\b' "
                           f"eagle_harness | head -60; echo ----; grep -rln -E 'flashinfer_cutedsl|CUTEDSL|cutedsl' vllm | head -20; "
                           f"echo ----; grep -rn -E 'class .*Custom.*Backend|CUSTOM *=|\"CUSTOM\"' vllm | head -20", ok_rc=(0, 1, 2))
    # 8. payload as ONE .bin (Kaggle unpacks .zip/.tar on Model or dataset creation and checks every inner name)
    if not R["errors"]:
        files = sorted(p for p in PAY.rglob("*") if p.is_file())
        out = W / "eagle_offline_tar.bin"
        with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as t:
            for p in files:
                t.add(p, "eagle_offline/" + str(p.relative_to(PAY)), recursive=False)
        with tarfile.open(out) as t:
            got = {m.name: m.size for m in t.getmembers() if m.isfile()}
        want = {"eagle_offline/" + str(p.relative_to(PAY)): p.stat().st_size for p in files}
        h = hashlib.sha256()
        with open(out, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        R["payload"] = {"file": out.name, "bytes": out.stat().st_size, "sha256": h.hexdigest(), "members": len(got),
                        "mismatch": sorted(n for n in set(want) | set(got) if want.get(n) != got.get(n))[:20]}
        if R["payload"]["mismatch"]:
            fail("tar member mismatch")
        if R["payload"]["bytes"] > 19e9:
            fail("payload too big")
except Exception as e:
    import traceback
    fail(f"fatal: {e!r}")
    R["traceback"] = traceback.format_exc()[-3000:]
finally:
    with tarfile.open(W / "p6a_logs_tar.bin", "w") as t:
        for p in sorted(LOGS.glob("*")):
            t.add(p, "p6a_logs/" + p.name, recursive=False)
    bad = sorted(str(p.relative_to(W)) for p in W.rglob("*") if p.is_dir() or (p.is_file() and (
        len(p.relative_to(W).parts) > 1 or re.search(r"\.(zip|tar|gz|tgz|bz2|xz|7z)$", p.name) or re.search(r"[^A-Za-z0-9._-]", p.name))))
    R["bad_output_names"] = bad
    if bad:
        fail(f"output not flat / safe: {bad[:10]}")
    f = R["facts"]
    R["feasibility"] = {"offline_install": "offline_install" in R["steps"] and R["steps"]["offline_install"]["rc"] == 0,
                        "import": f.get("import_ok"), "flagcheck": f.get("flagcheck"), "eagle_method_visible": f.get("eagle_method_visible"),
                        "custom_attention_visible": f.get("custom_attention_visible"),
                        "libs_not_runnable_on_sm89": sorted(k for k, v in (f.get("archs_runnable_on_sm89") or {}).items() if not v),
                        "glibc_over_2_35": f.get("glibc_over_2_35")}
    R["all_ok"] = not R["errors"]
    R["sec"] = round(time.time() - T0)
    (W / "p6a_report.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({"feasibility": R["feasibility"], "versions": f.get("versions"), "wheels": f.get("wheels"),
                      "payload": R.get("payload"), "errors": R["errors"]}, default=str)[:8000], flush=True)
    print("ALL OK" if R["all_ok"] else "NOT OK", "|", R["sec"], "s", flush=True)
'''

P6B = r'''# E23 P6b: EAGLE-Corp/Qwen3.8-27B-EAGLE4 drafter, pinned revision, sha-checked, flat output (internet ON)
import hashlib, json, os, re, time, urllib.request
from pathlib import Path
T0 = time.time()
for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE"):
    os.environ.pop(k, None)
os.environ.update(HF_HOME="/tmp/e23p6b_hf", HF_XET_CHUNK_CACHE_SIZE_BYTES="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
REPO, REV = "EAGLE-Corp/Qwen3.8-27B-EAGLE4", "f58a5985223040552ad404fba98f0dd27f5f676a"
SKIP = {".gitattributes"}
W = Path("/kaggle/working")
R = {"repo": REPO, "revision": REV, "files": {}, "errors": []}
try:
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REV}?recursive=true", timeout=60))
    want = {e["path"]: (e.get("size"), (e.get("lfs") or {}).get("oid")) for e in tree if e.get("type") == "file" and e["path"] not in SKIP}
    R["remote"] = want
    from huggingface_hub import hf_hub_download
    for n, (size, oid) in sorted(want.items(), key=lambda kv: kv[1][0] or 0):
        t = time.time()
        p = Path(hf_hub_download(REPO, n, revision=REV, local_dir=str(W)))
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        rec = {"size": p.stat().st_size, "sha256": h.hexdigest(), "sec": round(time.time() - t, 1)}
        rec["ok"] = rec["size"] == size and (oid is None or rec["sha256"] == oid)
        R["files"][n] = rec
        print(n, rec, flush=True)
        if not rec["ok"]:
            R["errors"].append(f"{n}: size/sha mismatch")
    cfg = json.loads((W / "config.json").read_text())
    R["config"] = {k: cfg.get(k) for k in ("architectures", "model_type", "hidden_size", "num_hidden_layers", "vocab_size", "torch_dtype")}
except Exception as e:
    R["errors"].append(f"fatal: {e!r}")
finally:
    import shutil
    shutil.rmtree(W / ".cache", ignore_errors=True)
    bad = sorted(str(p.relative_to(W)) for p in W.rglob("*") if p.is_dir() or (p.is_file() and (
        len(p.relative_to(W).parts) > 1 or p.name.startswith(".") or re.search(r"[^A-Za-z0-9._-]", p.name))))
    R["bad_output_names"] = bad
    if bad:
        R["errors"].append(f"output not flat / safe: {bad}")
    R["all_ok"] = not R["errors"] and bool(R["files"]) and all(v["ok"] for v in R["files"].values())
    R["sec"] = round(time.time() - T0)
    (W / "p6b_report.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R.get(k) for k in ("config", "errors", "bad_output_names")}), flush=True)
    print("ALL OK" if R["all_ok"] else "NOT OK", "|", R["sec"], "s", flush=True)
'''

for slug, title, code in (("q27-e23-p6-eagle-env", "Q27 E23 P6 Eagle Env", P6A), ("q27-e23-p6-eagle-drafter", "Q27 E23 P6 Eagle Drafter", P6B)):
    compile(code, slug, "exec")
    kd = HERE / ("kernel_" + slug.split("-")[-1])
    kd.mkdir(exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (kd / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"foysalemonshanto/{slug}", "title": title, "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "docker_image": IMAGE, "machine_shape": "None"}, indent=2), encoding="utf-8", newline="\n")
    print("kernel", kd, len(code), "chars")
