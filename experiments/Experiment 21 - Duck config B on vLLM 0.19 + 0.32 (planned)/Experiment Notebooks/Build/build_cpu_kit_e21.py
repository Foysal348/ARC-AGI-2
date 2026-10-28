"""E21 Kaggle CPU kit (memory: Kaggle CPU validation before any GPU run), on the GPU notebooks' pinned image (Python 3.12):
1. the fake-vLLM notebook harness of both E21 notebooks, every mode (src/harness/harness_e21_nb.py);
2. every launch command of both notebooks (CONFIG["e21"]["flags"], read from the commit-mode run's e21_report.json)
   parsed by the REAL vLLM argument parser: vllm==0.19.0 installed into the kernel from the S49 wheelhouse (as the
   notebook's CELL 2), vLLM 0.32.0 in /tmp/vllm-venv from its wheelhouse (as the notebook's e18_install);
3. the real Kaggle model folders (FP8 repack + RedHat INT4, attached as model sources): sha256 of chat_template.jinja,
   tokenizer.json, tokenizer_config.json, generation_config.json, config.json, and tokenizer encode equality on samples.
Writes Build/cpu_kit/q27x_e21_cpu_kit_<sha12>.bin + dataset-metadata.json (dataset foysalemonshanto/q27-e21-cpu-kit)
and Build/cpu_kernel/ (q27-e21-cpu-harness.ipynb + kernel-metadata.json).
    python build_cpu_kit_e21.py"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E21 = HERE.parents[1]
E15 = E21.parent / "Experiment 15 - Duck 2x35k independent attempts (planned)"
E14SRC = Path(r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/src")
FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
KIT_SLUG = "foysalemonshanto/q27-e21-cpu-kit"
KERNEL_SLUG = "foysalemonshanto/q27-e21-cpu-harness"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
NBS = {"nb1": "Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb", "nb2": "Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16.ipynb",
       "nb1s": "Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20_SMOKE.ipynb", "nb2s": "Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16_SMOKE.ipynb"}
MODES = ["commit", "rerun", "fallbackA", "halfAfail", "death", "reasoningkey", "gateAB", "wdhold", "overrun", "kvoutage",
         "gpuheldB"]
RUNS = [(f"{k}_{m}", k, m) for k in ("nb1", "nb2") for m in MODES] + [("nb1s_commit", "nb1s", "commit"), ("nb2s_commit", "nb2s", "commit")]
# --part NAME --runs a,b,c (user 2026-10-28: run the kit as several kernels in parallel): this kernel runs only those
# harness runs; --no-flags skips the vLLM installs + flag / tokenizer checks (one part does them)
def _arg(k, d=None):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
PART = _arg("--part")
if _arg("--runs"):
    _want = _arg("--runs").split(",")
    RUNS = [r for r in RUNS if r[0] in _want]
    assert len(RUNS) == len(_want), (_want, RUNS)
if _arg("--modes"):
    MODES = _arg("--modes").split(",")
    RUNS = [r for r in RUNS if r[2] in MODES]
NO_FLAGS = "--no-flags" in sys.argv
# --flags-only (user 2026-10-28: everything in parallel): no harness; the real-vLLM parse of src/e21_flags.json (every launch
# command, generated from the notebooks' own flag functions with the Kaggle paths) + the model-file checks
FLAGS_ONLY = "--flags-only" in sys.argv
if FLAGS_ONLY:
    RUNS = []
    KERNEL_SLUG = "foysalemonshanto/q27-e21-cpu-flags"
elif PART:
    KERNEL_SLUG = f"foysalemonshanto/q27-e21-cpu-{PART}"
OUT_DIR = HERE / "cpu_kit"

files = {}
for p in sorted((E21 / "src").rglob("*.py")):
    if "__pycache__" not in p.parts:
        files["e21/" + p.relative_to(E21 / "src").as_posix()] = p.read_bytes()
files["e21/e21_flags.json"] = (E21 / "src" / "e21_flags.json").read_bytes()
for p in sorted((E21 / "src" / "harness").glob("*.md")):
    files["e21/harness/" + p.name] = p.read_bytes()
for p in sorted(E14SRC.glob("e14_*.py")):
    files["e14src/" + p.name] = p.read_bytes()
for p in sorted((E14SRC / "harness").glob("*.py")):
    files["e14src/harness/" + p.name] = p.read_bytes()
for p in sorted((E15 / "src" / "harness").glob("*.py")):
    files["e15/harness/" + p.name] = p.read_bytes()
for p in sorted(FIX.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        files["fixtures/" + p.relative_to(FIX).as_posix()] = p.read_bytes()
for n in NBS.values():
    files["nb/" + n] = (HERE.parent / n).read_bytes()
for p in sorted((E15 / "Experiment Notebooks" / "Build" / "payload").glob("q27x_*.bin")):
    files["nb/Build/payload/" + p.name] = p.read_bytes()
manifest = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
files["MANIFEST.json"] = json.dumps(manifest, indent=0).encode()
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for k in sorted(files):
        zi = zipfile.ZipInfo(k, date_time=(2026, 10, 28, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, files[k])
kit = buf.getvalue()
KIT_SHA = hashlib.sha256(kit).hexdigest()
KIT_NAME = f"q27x_e21_cpu_kit_{KIT_SHA[:12]}.bin"
OUT_DIR.mkdir(exist_ok=True)
for old in OUT_DIR.glob("q27x_e21_cpu_kit_*.bin"):
    old.unlink()
(OUT_DIR / KIT_NAME).write_bytes(kit)
(OUT_DIR / "dataset-metadata.json").write_text(json.dumps(
    {"title": "Q27 E21 CPU kit", "id": KIT_SLUG, "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")

CODE = r'''# E21 CPU kit on Kaggle (pinned image = the GPU notebooks'): harness modes, real-vLLM flag parsing, real model files
import hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
KIT_NAME, KIT_SHA = "''' + KIT_NAME + '''", "''' + KIT_SHA + '''"
NBS = ''' + json.dumps(NBS) + r'''
RUNS = ''' + json.dumps(RUNS) + r'''
WH = "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
WH032 = "/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"
FP8 = "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"
INT4 = "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1"
OUT = Path("/kaggle/working"); K = Path("/tmp/e21_kit"); T0 = time.time()
REPORT = {"kit": KIT_NAME, "python": sys.version, "nb_sha": {}, "harness": {}, "flagcheck": {}, "models": {}}


def save():
    (OUT / "cpu_kit_report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")


def find(name, sha):
    for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name, "*/*/*/*/" + name):
        for f in sorted(Path("/kaggle/input").glob(pat)):
            if hashlib.sha256(f.read_bytes()).hexdigest() == sha:
                return f.read_bytes()
    raise FileNotFoundError(f"{name} ({sha[:12]}) not under /kaggle/input: attach foysalemonshanto/q27-e21-cpu-kit")


shutil.rmtree(K, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(find(KIT_NAME, KIT_SHA))) as z:
    z.extractall(K)
man = json.loads((K / "MANIFEST.json").read_text())
bad = [k for k, h in man.items() if hashlib.sha256((K / k).read_bytes()).hexdigest() != h]
assert not bad, bad
REPORT["nb_sha"] = {k: hashlib.sha256((K / "nb" / v).read_bytes()).hexdigest() for k, v in NBS.items()}
print("kit OK:", len(man), "files |", sys.version.split()[0], "|", REPORT["nb_sha"], flush=True)

# ---- 3. the real model folders (FP8 vs INT4): template / tokenizer files and encode equality
try:
    _m = {}
    for tag, d in (("fp8", FP8), ("int4", INT4)):
        _m[tag] = {f: (hashlib.sha256((Path(d) / f).read_bytes()).hexdigest()[:16] if (Path(d) / f).is_file() else None)
                   for f in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json")}
    REPORT["models"]["files"] = _m
    _tc = json.loads((Path(FP8) / "tokenizer_config.json").read_text(encoding="utf-8"))
    _j8 = (Path(FP8) / "chat_template.jinja").read_text(encoding="utf-8") if (Path(FP8) / "chat_template.jinja").is_file() else None
    _j4 = (Path(INT4) / "chat_template.jinja").read_text(encoding="utf-8") if (Path(INT4) / "chat_template.jinja").is_file() else None
    REPORT["models"]["template_equal"] = {"fp8_jinja_eq_int4_jinja": _j8 == _j4 and _j8 is not None,
                                          "fp8_tokenizer_config_eq_jinja": _tc.get("chat_template") in (None, _j8)}
    from tokenizers import Tokenizer
    ta, tb = Tokenizer.from_file(FP8 + "/tokenizer.json"), Tokenizer.from_file(INT4 + "/tokenizer.json")
    _s = ["0 0 1 1 2\n3 4 5 6 7 8 9\n" * 40, "def transform(grid):\n    return [row[::-1] for row in grid]\n",
          "<think>\nLet me check the 3x3 block at (2, 5)...\n</think>\n\n", _j8 or ""]
    REPORT["models"]["encode_equal"] = all(ta.encode(x).ids == tb.encode(x).ids for x in _s)
    _q = json.loads((Path(INT4) / "config.json").read_text(encoding="utf-8")).get("quantization_config") or {}
    REPORT["models"]["int4_quant"] = {"method": _q.get("quant_method"), "format": _q.get("format"),
                                      "ignores_mtp": "re:^mtp.*" in (_q.get("ignore") or [])}
except Exception as e:
    REPORT["models"]["error"] = repr(e)[:500]
save()
print("models:", json.dumps(REPORT["models"])[:1500], flush=True)

# ---- the GPU notebooks' runtime pins (CELL 2): transformers 4.57.6 + huggingface-hub 0.36.2 for the harness
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH,
                    "transformers==4.57.6", "huggingface-hub==0.36.2", "tokenizers==0.22.2"], capture_output=True, text=True)
REPORT["pins"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}

# ---- 1. harness modes
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", E14_FIXTURES_DIR=str(K / "fixtures"), E14_HARNESS_DIR=str(K / "e14src/harness"),
           Q27X_PAYLOAD_DIR=str(K / "nb/Build/payload"), E21_FAKE_SERVER=str(K / "e21/harness/fake_vllm_duck_server.py"))
for label, nbk, mode in RUNS:
    env = dict(ENV, E21_HARNESS_ROOT=f"/tmp/e21_h_{label}", E21_MODE=mode)
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(K / "e21/harness/harness_e21_nb.py"), str(K / "nb" / NBS[nbk])],
                           cwd=K / "e21/harness", env=env, capture_output=True, text=True, timeout=5400)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", "timeout"
    (OUT / f"harness_{label}.log").write_text(out, encoding="utf-8")
    lines = out.splitlines()
    REPORT["harness"][label] = {"rc": rc, "sec": round(time.time() - t), "ok": rc == 0 and "@@E21_HARNESS_OK@@" in out,
                                "summary": [x for x in lines if x.startswith("[e21")][-40:],
                                "fails": [x[:1500] for x in lines if "FAIL" in x][:40],
                                "tail": out[-3000:] if rc != 0 else ""}
    if mode == "commit":       # the launch commands this notebook makes (all halves, primary + fallback)
        for rp in sorted(Path(f"/tmp/e21_h_{label}").rglob("e21_report.json")):
            try:
                REPORT["flagcheck"].setdefault(nbk, {})["flags"] = json.loads(rp.read_text())["config"]["flags"]
            except Exception as e:
                REPORT["flagcheck"].setdefault(nbk, {})["flags_error"] = repr(e)[:300]
    save()
    print(label, "->", "ok" if REPORT["harness"][label]["ok"] else "FAIL", REPORT["harness"][label]["summary"][-6:],
          REPORT["harness"][label]["fails"][:6], flush=True)

# ---- 2. real vLLM argument parsing of every launch command
_fl = {f"{nbk}:{k}": v for nbk, d in REPORT["flagcheck"].items() for k, v in (d.get("flags") or {}).items()}
if not _fl and (K / "e21/e21_flags.json").exists():      # --flags-only kit: the commands generated at build time
    _fl = json.loads((K / "e21/e21_flags.json").read_text(encoding="utf-8"))
(Path("/tmp/e21_flags.json")).write_text(json.dumps(_fl), encoding="utf-8")
REPORT["flagcheck"]["commands"] = len(_fl)
t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH, "vllm==0.19.0"],
                   capture_output=True, text=True, timeout=3600)
REPORT["flagcheck"]["install_019"] = {"rc": r.returncode, "sec": round(time.time() - t), "tail": (r.stdout + r.stderr)[-600:]}
r = subprocess.run([sys.executable, str(K / "e21/e21_tokcheck.py"), FP8, INT4], capture_output=True, text=True, timeout=900)
REPORT["tokcheck_kernel"] = next((json.loads(l[9:]) for l in r.stdout.splitlines() if l.startswith("TOKCHECK ")),
                                 {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]})
print("tokcheck kernel (transformers of the 0.19 server):", json.dumps(REPORT["tokcheck_kernel"])[:800], flush=True)
r = subprocess.run([sys.executable, str(K / "e21/e21_flagcheck.py"), "/tmp/e21_flags.json", "0.19.0"],
                   capture_output=True, text=True, timeout=1800)
REPORT["flagcheck"]["v019"] = {"rc": r.returncode, "out": (r.stdout + r.stderr)[-6000:],
                               "summary": next((json.loads(l[10:]) for l in r.stdout.splitlines() if l.startswith("FLAGCHECK ")), None)}
save()
print("flagcheck 0.19:", json.dumps(REPORT["flagcheck"]["v019"]["summary"]), flush=True)
t = time.time()
inst = next(iter(sorted(Path(WH032).rglob("install_wheelhouse.py"))), None)
if inst is None:
    REPORT["flagcheck"]["v032"] = {"error": "install_wheelhouse.py not found: attach " + WH032}
else:
    r = subprocess.run([sys.executable, str(inst), "/tmp/vllm-venv"], capture_output=True, text=True, timeout=2400)
    REPORT["flagcheck"]["install_032"] = {"rc": r.returncode, "sec": round(time.time() - t), "tail": (r.stdout + r.stderr)[-600:]}
    vpy = next((l.split("=", 1)[1] for l in r.stdout.splitlines() if l.startswith("VENV_PYTHON=")), "/tmp/vllm-venv/bin/python")
    r = subprocess.run([vpy, str(K / "e21/e21_tokcheck.py"), FP8, INT4], capture_output=True, text=True, timeout=900)
    REPORT["tokcheck_v032"] = next((json.loads(l[9:]) for l in r.stdout.splitlines() if l.startswith("TOKCHECK ")),
                                   {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]})
    print("tokcheck 0.32 venv:", json.dumps(REPORT["tokcheck_v032"])[:800], flush=True)
    r = subprocess.run([vpy, str(K / "e21/e21_flagcheck.py"), "/tmp/e21_flags.json", "0.32.0"],
                       capture_output=True, text=True, timeout=1800)
    REPORT["flagcheck"]["v032"] = {"rc": r.returncode, "out": (r.stdout + r.stderr)[-6000:],
                                   "summary": next((json.loads(l[10:]) for l in r.stdout.splitlines() if l.startswith("FLAGCHECK ")), None)}
save()
print("flagcheck 0.32:", json.dumps((REPORT["flagcheck"].get("v032") or {}).get("summary")), flush=True)
_mods = REPORT["models"]
REPORT["all_ok"] = (all(v["ok"] for v in REPORT["harness"].values())
                    and all(((REPORT["flagcheck"].get(x) or {}).get("summary") or {}).get("all_ok") for x in ("v019", "v032"))
                    and bool((_mods.get("template_equal") or {}).get("fp8_jinja_eq_int4_jinja")) and bool(_mods.get("encode_equal"))
                    and bool((REPORT.get("tokcheck_kernel") or {}).get("ok")) and bool((REPORT.get("tokcheck_v032") or {}).get("ok")))
REPORT["sec"] = round(time.time() - T0)
save()
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
if NO_FLAGS:      # this part runs the harness only: cut section 2 (vLLM installs, flag + tokenizer checks)
    _i0, _i1 = CODE.index("# ---- 2. real vLLM argument parsing"), CODE.index("_mods = REPORT")
    _skip = 'REPORT["flagcheck"] = {"skipped": "another kit part runs the flag / tokenizer checks"}' + chr(10)
    CODE = CODE[:_i0] + _skip + CODE[_i1:]
    _a0 = CODE.index('REPORT["all_ok"] = ')
    _a1 = CODE.index('REPORT["sec"] = ')
    CODE = CODE[:_a0] + 'REPORT["all_ok"] = all(v["ok"] for v in REPORT["harness"].values())' + chr(10) + CODE[_a1:]
compile(CODE, "<cpu kit>", "exec")
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
          "# E21 CPU kit (Kaggle pinned image)\n\nHarness modes of both E21 notebooks, real vLLM 0.19 / 0.32 parsing of every "
          "launch command, and the real FP8 / INT4 model files. Kit: `" + KIT_SLUG + "` (" + KIT_NAME + ").\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "cpu_kernel"
KD.mkdir(exist_ok=True)
KD = HERE / ("cpu_kernel_flags" if FLAGS_ONLY else f"cpu_kernel_{PART}" if PART else "cpu_kernel")
KD.mkdir(exist_ok=True)
(KD / "q27-e21-cpu-harness.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
(KD / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 E21 CPU Flags" if FLAGS_ONLY else f"Q27 E21 CPU {PART.upper()}" if PART else "Q27 E21 CPU Harness", "code_file": "q27-e21-cpu-harness.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3", "foysalemonshanto/vllm-0-32-0-cu13-wheelhouse",
                                        KIT_SLUG],
    "kernel_sources": [], "competition_sources": [],
    "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1",
                      "foysalemonshanto/qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1"],
    "docker_image": IMAGE, "machine_shape": "None"}, indent=2), encoding="utf-8")
print("kit", OUT_DIR / KIT_NAME, round(len(kit) / 1e6, 1), "MB,", len(manifest), "files")
print("kernel", KD, {k: hashlib.sha256((HERE.parent / v).read_bytes()).hexdigest()[:16] for k, v in NBS.items()})
