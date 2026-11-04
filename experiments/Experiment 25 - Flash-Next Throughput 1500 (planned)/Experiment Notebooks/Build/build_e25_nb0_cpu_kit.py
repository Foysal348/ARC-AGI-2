"""E25 NB0 Kaggle CPU kits (user GO 2026-10-31: parallel CPU kernels on foysalemonshanto, pinned Py3.12 image, private, no
GPU, no internet, exact Model paths). Pattern: build_e25_nb3_cpu_kit.py (NB3 kits v1 ALL OK) and Experiment 13
build_e13_rc2_cpu_kit.py (RC2 kit v1 ALL OK). Each kernel is fail-soft, writes its own JSON report and prints exactly one
line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`. Embedded files travel zlib + base64 (one JSON blob).

  K1 q27-e25-nb0-kit-real   NB0's own code (NB0_UNIT + RC2_CPU_KIT, TMP = /tmp exactly as the GPU run) on the REAL mounts:
                            inputs, merge, full sha256, prompt ids (all 25 RC2 Kaggle prompts reproduced, 40k v5 = TF_P),
                            the real Merlin offline install, merlin_fn_server() for every boot D / C / E / B / A
                            (= launch.build + host_phase + NB0's argv / env edits; no server started), each boot's config
                            diff vs RC2 exactly as designed, the vLLM 0.32 serve parser + AsyncEngineArgs on every final
                            argv inside the Merlin venv (gated when the parser imports; `--help=all` is recorded only: RC2
                            kit v1 showed it is not an argument list on CPU), the boot-E audit on the INSTALLED tree
                            (sha 23c01a20... / a23d5f90..., 10 anchors, 3 references per file, not a build product,
                            py_compile in the venv, patch apply, the patched function imported in the venv returns False
                            under the env and prints the marker, restore + sha re-check; the on-GPU 256-codepoint decoder
                            check is stubbed and recorded GPU-only), the scales sidecar sha, NB0's host preparation on the
                            real cgroup (process reap, psm cleanup, fadvise drop with file / inactive_file before-after,
                            Merlin's gate computed; on a ~30 GiB CPU box it is expected False and is recorded, not gated).
                            Inputs: the 7 RC2 Models + ARC.
  K2 q27-e25-nb0-kit-units  nb0_unit_tests.py on Py3.12 (fake tree from the real part-1 small files, the embedded Merlin
                            ad42e37 files it needs and RC2's two real server logs), the fake end-to-end in clean mode on
                            the REAL Linux /proc (session scan, reap and load monitor on real pids), the competition-rerun
                            flow (fallback submission only). Inputs: part 1 + ARC.
  K3 q27-e25-nb0-kit-e2ea   fake end-to-end fault modes native, kv_long, mtp (each asserts its verdict + findings).
  K4 q27-e25-nb0-kit-e2eb   fake end-to-end fault modes path, shift, repeat.

Run: python build_e25_nb0_cpu_kit.py  ->  D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb0-kit-{real,units,e2ea,e2eb}__foysalemonshanto/
Local dry run (KIT_INPUT = the RC2 local fake input; K1 is expected NOT OK there: no Merlin payload, fake PLE shards):
     python build_e25_nb0_cpu_kit.py --dry
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
E13 = Path(r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)")
MERLIN_SRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
RC2_LOGS = Path(r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2/logs")
NB = HERE.parent / "Q27-XHIGH-E25_NB0_FP8_QUALITY.ipynb"
NB_SHA = hashlib.sha256(NB.read_bytes()).hexdigest()
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
RC2_MODELS = ([f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
              ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
               "foysalemonshanto/e23-merlin-env/Other/v1/1"])


def lf(p):
    return Path(p).read_text(encoding="utf-8").replace("\r\n", "\n")


EXPORT = lf(HERE / "nb0_code_export.py")
assert EXPORT.count("def analyze(") == 1
_vllm_cfg = lf(MERLIN_SRC / "vllm/config/vllm.py").splitlines(True)
_i = next(i for i, l in enumerate(_vllm_cfg) if "def validate_mamba_cached_kernel" in l)
VLLM_EXCERPT = ("# EXCERPT of Merlin ad42e37 vllm/config/vllm.py (NB0 CPU kit: the MTP-off / RecoverSSM branch only)\n"
                + "".join(_vllm_cfg[_i - 1:_i + 70]))
assert "num_speculative_tokens=0, so the stock GDN path is used" in VLLM_EXCERPT.replace('"\n                "', "")
COMMON = {"nb0_code_export.py": EXPORT, "nb0_fake_server.py": lf(HERE / "nb0_fake_server.py"),
          "nb0_local_e2e.py": lf(HERE / "nb0_local_e2e.py"), "parts.json": lf(E13 / "prep/merlin_fn_ckpt/kernels/parts.json")}
UNIT_FILES = {"nb0_unit_tests.py": lf(HERE / "nb0_unit_tests.py"),
              "merlin_src/tools/merlin/launch.py": lf(MERLIN_SRC / "tools/merlin/launch.py"),
              "merlin_src/tools/merlin/registry.py": lf(MERLIN_SRC / "tools/merlin/registry.py"),
              "merlin_src/tools/merlin/profiles/flash-next.json": lf(MERLIN_SRC / "tools/merlin/profiles/flash-next.json"),
              "merlin_src/vllm/models/qwen4_exp/nvidia/ops/qsa.py": lf(MERLIN_SRC / "vllm/models/qwen4_exp/nvidia/ops/qsa.py"),
              "merlin_src/vllm/models/qwen4_exp/nvidia/qsa.py": lf(MERLIN_SRC / "vllm/models/qwen4_exp/nvidia/qsa.py"),
              "merlin_src/vllm/config/vllm.py": VLLM_EXCERPT,
              "rc2_logs/server_merlin_fp8.log": lf(RC2_LOGS / "server_merlin_fp8.log"),
              "rc2_logs/server_merlin_bf16.log": lf(RC2_LOGS / "server_merlin_bf16.log")}
for _k in ("ops/qsa.py", "nvidia/qsa.py"):        # the audited bytes (LF, ad42e37) are what K2's unit tests see
    _p = next(v for k, v in UNIT_FILES.items() if k.endswith(_k) and ("ops" in k) == (_k == "ops/qsa.py"))
    assert hashlib.sha256(_p.encode()).hexdigest() in ("23c01a2069097129ae927bdf4623a3837b7403d4a1cad548505769c05eb6792d",
                                                         "a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653")


def blob(files):
    return base64.b64encode(zlib.compress(json.dumps(files).encode("utf-8"), 9)).decode("ascii")


HEAD = r'''# E25 NB0 CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, zlib
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/nb0kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E25 NB0 CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}}
def save():
    (W / f"nb0_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
FILES = json.loads(zlib.decompress(base64.b64decode("__BLOB__")).decode("utf-8"))
B = K / "build"
for rel, txt in FILES.items():
    (B / rel).parent.mkdir(parents=True, exist_ok=True)
    (B / rel).write_text(txt, encoding="utf-8", newline="\n")
REP["files_sha256"] = {k: hashlib.sha256(v.encode("utf-8")).hexdigest()[:16] for k, v in FILES.items()}
def sh(name, args, env=None, timeout=3600, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})}, cwd=str(cwd or K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    return {"rc": rc, "sec": round(time.time() - t), "tail": out[-1500:], "_out": out}
def verdict(ok, recorded=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "recorded": recorded or {}, "minutes": round((time.time() - T0) / 60, 1)}
    save()
    bad = [k for k, v in ok.items() if not v]
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:6000], flush=True)
    print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
MAKE_FAKE = r"""
import hashlib, json, shutil, sys
from pathlib import Path
INP_REAL, INP = Path(sys.argv[1]), Path(sys.argv[2])
PARTS = json.loads(Path(sys.argv[3]).read_text())
P1 = INP_REAL / "models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1"
shutil.rmtree(INP, ignore_errors=True)
MD = INP / "models/foysalemonshanto"
PD = {k: MD / f"qwen3-8-flash-next-w4a16-merlin-p{k}/transformers/default/1" for k in range(1, 6)}
d1 = PD[1]; d1.mkdir(parents=True)
man_real = json.loads((P1 / "q27_fn_merlin_part1_manifest.json").read_text())
small = {n: v for n, v in man_real["files"].items() if not n.endswith(".safetensors")}
for n in small:
    shutil.copyfile(P1 / n, d1 / n)
idx = json.loads((d1 / "model.safetensors.index.json").read_text())
ple = set(PARTS["ple_from_overlay"])
shards = sorted(set(idx["weight_map"].values()) - ple - set(small))
def fake(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (n * 997).to_bytes(8, "little") * 16
    path.write_bytes(data)
    return [len(data), hashlib.sha256(data).hexdigest()]
groups = [shards[i::5] for i in range(5)]
man1 = dict(man_real, files=dict(small))
for i, n in enumerate(groups[0]):
    man1["files"][n] = fake(d1 / n, i)
(d1 / "q27_fn_merlin_part1_manifest.json").write_text(json.dumps(man1))
for k in range(2, 6):
    man = {"what": f"fake part {k}", "part": k, "parts": 5, "model_subdir": ".",
           "files": {n: fake(PD[k] / n, 100 * k + i) for i, n in enumerate(groups[k - 1])}}
    (PD[k] / f"q27_fn_merlin_part{k}_manifest.json").write_text(json.dumps(man))
pd = MD / "qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1"
for i, n in enumerate(sorted(ple)):
    fake(pd / n, 9000 + i)
mp = MD / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin"
mp.parent.mkdir(parents=True); mp.write_bytes(b"")
c = INP / "competitions/arc-prize-2026-arc-agi-2"; c.mkdir(parents=True)
ev = INP_REAL / "competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
shutil.copyfile(ev, c / "arc-agi_evaluation_challenges.json"); shutil.copyfile(ev, c / "arc-agi_test_challenges.json")
print("FAKE_TREE_OK shards", len(shards), "small", len(small), "ple", len(ple))
"""
def fake_tree():
    (K / "make_fake.py").write_text(MAKE_FAKE)
    r = sh("fake_tree", [sys.executable, str(K / "make_fake.py"), INPUT_ROOT, str(K / "fin"), str(B / "parts.json")], timeout=600)
    return {"rc": r["rc"], "ok": r["rc"] == 0 and "FAKE_TREE_OK" in r["_out"], "tail": r["tail"][-400:]}
def clean_env():
    for k_ in list(os.environ):
        if k_.startswith(("NB0_", "E23_", "RC2_", "FAKE_")) or k_ == "KAGGLE_IS_COMPETITION_RERUN":
            os.environ.pop(k_, None)
PORT_BASE = os.environ.get("KIT_PORT_BASE", "8710")
def e2e(modes, real_proc=False):
    """nb0_local_e2e.py on the fake tree; one result per mode from its PASS / FAIL lines."""
    env = {"NB0_PORT_BASE": PORT_BASE, **({"E2E_REAL_PROC": "1"} if real_proc else {})}
    r = sh("e2e_" + "_".join(modes), [sys.executable, "-u", str(B / "nb0_local_e2e.py"), str(K / "fin"), str(K / "e2e")] + modes,
           env, 3600, cwd=B)
    lines = r["_out"].splitlines()
    per = {}
    for m in modes:
        ml = [l for l in lines if f"[{m}]" in l]
        per[m] = {"pass": sum(l.strip().startswith("PASS") for l in ml), "fail": [l.strip()[:200] for l in ml if l.strip().startswith("FAIL")],
                  "header": next((l for l in lines if l.startswith(f"== mode {m}:")), None)}
        per[m]["ok"] = bool(ml) and not per[m]["fail"] and per[m]["header"] is not None and per[m]["header"].endswith("s") and " rc 0," in per[m]["header"]
    return {"rc": r["rc"], "sec": r["sec"], "all_pass": r["rc"] == 0 and "E2E ALL PASS" in r["_out"], "modes": per,
            "gates_lines": [l.strip()[:400] for l in lines if l.strip().startswith("gates:")], "tail": r["tail"][-800:]}
'''

K1 = r'''
# ---------------------------------------------------------------- K1: NB0 code on the REAL mounts (no server is started)
R1, PL, PAR, BYTE, HOST = {}, {}, {}, {}, {}
G = None
try:
    clean_env()
    (K / "out").mkdir(exist_ok=True); (K / "shm").mkdir(exist_ok=True)
    os.environ.update(NB0_UNIT="1", RC2_CPU_KIT="1", E23_OUT=str(K / "out"), E23_INPUT=INPUT_ROOT,
                      RC2_SUBMISSION=str(K / "out/submission.json"))
    if os.environ.get("KIT_LOCAL_TMP"):                         # local dry run only; Kaggle = /tmp exactly as the GPU run
        os.environ["E23_TMP"] = os.environ["KIT_LOCAL_TMP"]
        Path(os.environ["KIT_LOCAL_TMP"]).mkdir(parents=True, exist_ok=True)
    REH = os.environ.get("KIT_DRY_REHEARSAL")                    # local dry run only (never set on Kaggle), see below
    if REH:
        os.environ["E23_FAKE_SERVER"] = "rehearsal"             # FAKE import: the merge copies files (Windows: no symlinks)
    t = time.time()
    G = runpy.run_path(str(B / "nb0_code_export.py"), run_name="nb0kit")["analyze"].__globals__
    if REH:                                                      # then the REAL code paths: Merlin launch.build from the local
        import importlib.util                                   # source tree, a temp copy of the audited QSA files, no host_phase
        src = Path(REH)
        tree = K / "rehearsal_tree"
        for rel in ("tools/merlin/launch.py", "tools/merlin/registry.py", "tools/merlin/profiles/flash-next.json",
                    "vllm/models/qwen4_exp/nvidia/ops/qsa.py", "vllm/models/qwen4_exp/nvidia/qsa.py"):
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_bytes((src / rel).read_bytes())
        (tree / ".build-products.sha256").write_text("0" * 64 + "  vllm/_C.abi3.so" + chr(10))
        sys.path.insert(0, str(tree / "tools/merlin"))
        import launch as ML
        ML.host_phase = lambda plan: None
        venv = K / "rehearsal_venv"; (venv / "bin").mkdir(parents=True, exist_ok=True)
        try:                                                     # a runnable <venv>/bin/python for py_compile / the import probe
            os.symlink(sys.executable, venv / "bin" / ("python.exe" if os.name == "nt" else "python"))
        except Exception:
            shutil.copyfile(sys.executable, venv / "bin" / ("python.exe" if os.name == "nt" else "python"))
        G["FAKE"] = None
        G["MERLIN_ENG"] = {"kv": {"MERLIN_TREE": str(tree), "MERLIN_VENV": str(venv), "MERLIN_SHIM": "", "MERLIN_CC": ""}, "ML": ML, "cuda_home": ""}
        G["merlin_install"] = lambda: G["MERLIN_ENG"]
        REP["rehearsal"] = "LOCAL DRY REHEARSAL: FAKE merge, local Merlin source, host_phase stubbed, no venv"
    R = G["R"]
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"],
              host_mem=R["sections"]["A0"]["host_mem"], prompt_render={k: v for k, v in R["prompt_render"].items() if "text" not in k},
              tmp=str(G["TMP"]), ckpt=str(G["CKPT"]))
    log("R1 inputs", json.dumps(R["inputs"])[:500]); log("R1 merge", json.dumps(R1["merge"])[:300]); save()
    R1["prompt_check"] = G["prompt_check"]()
    log("R1 prompts", json.dumps({k: R1["prompt_check"][k] for k in ("ok", "reproduced", "rc2_recorded", "rc2_tf_prompt_is_40k_v5")}))
    sc = G["CKPT"] / "qsa_kv_scales_262k.json"
    R1["scales_sha256"] = hashlib.sha256(sc.read_bytes()).hexdigest() if sc.is_file() else None
    t = time.time()
    try:
        eng = G["merlin_install"]()
        R1["install"] = {"ok": True, "sec": round(time.time() - t), "kv": dict(eng["kv"]), "cuda_home": eng["cuda_home"]}
    except Exception as e:
        R1["install"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:800], "tb": traceback.format_exc()[-1500:]}
    log("R2 install", json.dumps(R1["install"])[:500]); save()
except Exception as e:
    R1["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R1["tb"] = traceback.format_exc()[-2500:]
    log("R1 FATAL", R1["fatal"])
REP["checks"]["R1"] = R1; save()

# ---------------------------------------------------------------- every boot's plan: build + host_phase + NB0 edits + diffs
EXPECT = {   # boot -> (flags that differ from the RC2 reference, env keys that differ, reference)
    "D": (set(), set(), "fp8"), "C": (set(), set(), "bf16"), "E": (set(), {"NB0_QSA_FORCE_BYTE_DECODER", "VLLM_CACHE_ROOT"}, "fp8"),
    "B": ({"--speculative-config"}, set(), "fp8"),
    "A": ({"--speculative-config", "--kv-cache-dtype", "--kv-cache-memory-bytes"}, {"KV_BYTES", "SCALES", "VLLM_QSA_KV_SCALES"}, "fp8")}
SRV = {}
if G is not None and (R1.get("install") or {}).get("ok"):
    for key in G["ORDER"]:
        b = G["BOOTS"][key]
        kvb = G["KV_FP8"] if b["kv"] == "fp8" else G["KV_BF16"]
        o = {}
        try:
            s = G["merlin_fn_server"](b["name"], b["kv"], b["port"], kvb)
            SRV[key] = s
            f = G["flags"](s.argv)
            fl_ref, env_ref, ref = EXPECT[key]
            tmp_ = str(G["TMP"])                    # identity on Kaggle (TMP = /tmp); the kit's own NB0_UNIT switch is not boot config
            envn = {k: (v.replace(tmp_, "/tmp").replace("\\", "/") if isinstance(v, str) else v) for k, v in s.env.items() if k != "NB0_UNIT"}
            d = G["config_diff"](s.argv, envn, kvb, ref)
            o.update(host_phase=G["HOST_PHASE"].get(b["name"]), help=G["HELP"].get(b["name"]), server_started=s.proc is not None,
                     argv_tail=s.argv[-66:], diff=d,
                     static={"mtp": ("--speculative-config" in s.argv) == b["mtp"], "seqs": f.get("--max-num-seqs") == str(b["want_seqs"]),
                             "kv": f.get("--kv-cache-dtype") == ("fp8_e4m3" if b["kv"] == "fp8" else "bfloat16"),
                             "prefix_caching": "--enable-prefix-caching" in s.argv,
                             "captures": (f.get("--compilation-config") == G["FP8_CAPTURES"]) if (b["captures"] or b["kv"] == "fp8") else True,
                             "flag_diff_as_designed": set(d["flags"]) == fl_ref, "env_diff_as_designed": set(d["env"]) == env_ref,
                             "byte_env": (s.env.get("NB0_QSA_FORCE_BYTE_DECODER") == "1") == b["byte"]})
            o["ok"] = o["host_phase"] == "ok" and all(o["static"].values()) and not o["server_started"]
        except Exception as e:
            o.update(ok=False, error=f"{type(e).__name__}: {e}"[:1200], host_phase=G["HOST_PHASE"].get(b["name"]), tb=traceback.format_exc()[-1500:])
        PL[key] = o
        log("R3 boot", key, json.dumps({k: v for k, v in o.items() if k not in ("argv_tail", "tb", "help")}, default=str)[:900]); save()
REP["checks"]["plans"] = PL; save()

# ---------------------------------------------------------------- the vLLM 0.32 serve parser + AsyncEngineArgs on every final argv
PARSE = r"""
import json, sys
out = {}
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.engine.arg_utils import AsyncEngineArgs
except BaseException as e:
    print("PARSE_JSON " + json.dumps({"_import": f"{type(e).__name__}: {e}"[:400]})); sys.exit(0)
for name, a in json.load(open(sys.argv[1])).items():
    o = {}
    try:
        ns = make_arg_parser(FlexibleArgumentParser()).parse_args(a)
        o["parse"] = True
        if getattr(ns, "model_tag", None):
            ns.model = ns.model_tag
        ea = AsyncEngineArgs.from_cli_args(ns)
        cc = ea.compilation_config
        caps = cc.get("cudagraph_capture_sizes") if isinstance(cc, dict) else getattr(cc, "cudagraph_capture_sizes", None)
        o.update(engine_args=True, speculative_config=ea.speculative_config, kv_cache_dtype=ea.kv_cache_dtype,
                 max_num_seqs=ea.max_num_seqs, enable_prefix_caching=ea.enable_prefix_caching, captures=caps,
                 use_replayssm=getattr(ea, "use_replayssm", None), max_model_len=ea.max_model_len, block_size=ea.block_size)
    except BaseException as e:
        o["error"] = f"{type(e).__name__}: {e}"[:500]
    out[name] = o
print("PARSE_JSON " + json.dumps(out, default=str))
"""
try:
    if SRV:
        argvs = {k: s.argv[s.argv.index("serve") + 1:] for k, s in SRV.items()}
        (K / "argvs.json").write_text(json.dumps(argvs)); (K / "parse.py").write_text(PARSE)
        s0 = SRV.get("D") or next(iter(SRV.values()))
        r = sh("parser", [G["MERLIN_ENG"]["kv"]["MERLIN_VENV"] + "/bin/python", str(K / "parse.py"), str(K / "argvs.json")], s0.env, 900)
        PAR = next((json.loads(l[11:]) for l in r["_out"].splitlines() if l.startswith("PARSE_JSON ")), {"_error": r["tail"][-800:]})
        sem = {}
        for k, o in PAR.items():
            if k.startswith("_") or not isinstance(o, dict):
                continue
            b = G["BOOTS"][k]
            spec = o.get("speculative_config")
            sem[k] = bool(o.get("parse")) and (not o.get("engine_args") or (
                ((spec is None) if not b["mtp"] else (isinstance(spec, dict) and spec.get("num_speculative_tokens") == 3)) and
                o.get("kv_cache_dtype") == ("fp8_e4m3" if b["kv"] == "fp8" else "bfloat16") and o.get("max_num_seqs") == b["want_seqs"]
                and o.get("enable_prefix_caching") is True))
        PAR["_semantics_ok"] = sem
        PAR["_available"] = "_import" not in PAR and "_error" not in PAR
except Exception as e:
    PAR = {"_fatal": f"{type(e).__name__}: {e}"[:600]}
log("R4 parser", json.dumps(PAR, default=str)[:1500])
REP["checks"]["parser"] = PAR; save()

# ---------------------------------------------------------------- boot E: audit + patch + import + restore on the INSTALLED tree
IMP = r"""
import json, os, sys
os.environ["NB0_QSA_FORCE_BYTE_DECODER"] = "1"
from vllm.models.qwen4_exp.nvidia.ops import qsa as O
v = O.qsa_e4m3_uses_native_cast()
print("BYTE_JSON " + json.dumps({"native_under_env": v, "file": O.__file__, "marker_in_file": "NB0-E25 boot E" in open(O.__file__).read()}))
try:
    import vllm.models.qwen4_exp.nvidia.qsa as Q
    print("QSA_IMPORT ok " + Q.__file__)
except BaseException as e:
    print("QSA_IMPORT " + f"{type(e).__name__}: {e}"[:300])
"""
try:
    if G is not None and (R1.get("install") or {}).get("ok"):
        real_dev = G["byte_decoder_device_check"]
        G["byte_decoder_device_check"] = lambda txt: {"ok": True, "stub": "GPU-only: the 256-codepoint decoder check runs on the L4 before boot E",
                                                      "decoder_found": "_decode_e4m3_to_bf16" in txt}
        tree = Path(G["MERLIN_ENG"]["kv"]["MERLIN_TREE"])
        QP = G["QSA_PATCH"]
        BYTE["before"] = {k: hashlib.sha256((tree / QP[k + "_rel"]).read_bytes()).hexdigest() for k in ("ops", "qsa")}
        pre = G["byte_decoder_prepare"]()
        BYTE["prepare"] = pre
        if pre.get("ok"):
            BYTE["markers"] = {k: (tree / QP[k + "_rel"]).read_text(encoding="utf-8").count(QP["marker"]) for k in ("ops", "qsa")}
            (K / "imp.py").write_text(IMP)
            env = dict((SRV.get("E") or SRV.get("D")).env)
            r = sh("byte_import", [G["MERLIN_ENG"]["kv"]["MERLIN_VENV"] + "/bin/python", str(K / "imp.py")], env, 900)
            BYTE["import"] = next((json.loads(l[10:]) for l in r["_out"].splitlines() if l.startswith("BYTE_JSON ")), {"error": r["tail"][-800:]})
            BYTE["import"]["marker_printed"] = "NB0 boot E: QSA E4M3 byte decoder forced" in r["_out"]
            BYTE["import"]["qsa_import"] = next((l for l in r["_out"].splitlines() if l.startswith("QSA_IMPORT")), None)
            BYTE["restore"] = G["byte_decoder_restore"]()
        BYTE["after"] = {k: hashlib.sha256((tree / QP[k + "_rel"]).read_bytes()).hexdigest() for k in ("ops", "qsa")}
        G["byte_decoder_device_check"] = real_dev
except Exception as e:
    BYTE["fatal"] = f"{type(e).__name__}: {e}"[:800]; BYTE["tb"] = traceback.format_exc()[-1500:]
    try:
        BYTE["restore_after_error"] = G["byte_decoder_restore"]()
    except Exception:
        pass
log("R5 byte", json.dumps({k: v for k, v in BYTE.items() if k != "tb"}, default=str)[:1500])
REP["checks"]["byte_decoder"] = BYTE; save()

# ---------------------------------------------------------------- sha256 of the checkpoint (background since the import) + host prep
try:
    if G is not None:
        t = time.time()
        ok, why = G["sha_gate"](3600)
        R1["sha"] = {"ok": ok, "why": why, "wait_sec": round(time.time() - t), "detail": G["R"]["setup"].get("sha256")}
        log("R1 sha", json.dumps(R1["sha"])[:400])
        HOST = G["host_prep"]()
        HOST["fadvise_effect_mib"] = {k: [(HOST.get("cg_before") or {}).get("stat_mib", {}).get(k), (HOST.get("cg_after") or {}).get("stat_mib", {}).get(k)]
                                      for k in ("file", "inactive_file", "active_file", "anon", "shmem")}
except Exception as e:
    HOST = {"fatal": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1500:]}
log("R6 host", json.dumps({k: HOST.get(k) for k in ("ok", "why", "host_gate", "fadvise", "fadvise_effect_mib", "procs")}, default=str)[:1500])
REP["checks"]["R1"] = R1; REP["checks"]["host"] = {k: v for k, v in HOST.items()}; save()

pc = R1.get("prompt_check") or {}
imp = BYTE.get("import") or {}
QA = (G or {}).get("QSA_PATCH") or {}
par_ok = (not PAR.get("_available")) or (len(PAR.get("_semantics_ok") or {}) == 5 and all(PAR["_semantics_ok"].values()))
verdict({"inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")), "merge_ok": bool((R1.get("merge") or {}).get("ok")),
         "sha256_all_ok": bool((R1.get("sha") or {}).get("ok")),
         "prompts_25_of_25_and_tf_p": pc.get("ok") is True and pc.get("reproduced") == 25 and pc.get("rc2_tf_prompt_is_40k_v5") is True,
         "merlin_install_ok": bool((R1.get("install") or {}).get("ok")),
         "plans_all_5_ok (build + host_phase + edits + diffs)": len(PL) == 5 and all(o.get("ok") for o in PL.values()),
         "no_server_started": not any(o.get("server_started") for o in PL.values()),
         "vllm_parser_on_final_argv (gated when importable)": par_ok,
         "installed_qsa_sha_matches_audit": BYTE.get("before") == {"ops": QA.get("ops_sha"), "qsa": QA.get("qsa_sha")} and bool(QA),
         "byte_patch_apply_ok": (BYTE.get("prepare") or {}).get("ok") is True and BYTE.get("markers") == {"ops": 1, "qsa": 1},
         "byte_patched_function_false_under_env": imp.get("native_under_env") is False and imp.get("marker_printed") is True and imp.get("marker_in_file") is True,
         "byte_restore_sha_ok": (BYTE.get("restore") or {}).get("ok") is True and BYTE.get("after") == BYTE.get("before"),
         "scales_sha_ok": R1.get("scales_sha256") == "5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67",
         "host_prep_code_ok (reap + fadvise + cgroup gate computed)": (HOST.get("procs") or {}).get("ok") is True
             and (HOST.get("fadvise") or {}).get("supported") is True and (HOST.get("fadvise") or {}).get("files", 0) >= 38
             and (HOST.get("host_gate") or {}).get("source") == "cgroup" and (HOST.get("host_gate") or {}).get("avail_mib") is not None},
        recorded={"host_gate_ok (expected False on a ~30 GiB CPU box)": (HOST.get("host_gate") or {}).get("ok"),
                  "host_gate_avail_mib": (HOST.get("host_gate") or {}).get("avail_mib"), "parser_available": PAR.get("_available"),
                  "help_check (CPU: not an argument list, RC2 kit v1)": {k: (o.get("help") or {}).get("missing", [None])[:3] if isinstance(o.get("help"), dict) else o.get("help") for k, o in PL.items()},
                  "device_check": "GPU-only, stubbed in this kit", "qsa_module_import": imp.get("qsa_import")})
'''

K2 = r'''
# ---------------------------------------------------------------- K2: unit tests on Py3.12 + fake e2e clean on the real /proc + rerun
U, E, RR = {}, {}, {}
try:
    clean_env()
    U["fake_tree"] = fake_tree()
    r = sh("units", [sys.executable, str(B / "nb0_unit_tests.py"), str(K / "fin"), str(K / "unit")],
           {"NB0_TEST_MERLIN_SRC": str(B / "merlin_src"), "NB0_TEST_RC2_LOGS": str(B / "rc2_logs")}, 2400, cwd=B)
    lines = r["_out"].splitlines()
    U.update(rc=r["rc"], sec=r["sec"], fails=[l[:200] for l in lines if l.startswith("FAIL ")][:40],
             summary=next((l for l in lines if re.match(r"^\d+/\d+ PASS", l)), None))
    m = re.match(r"^(\d+)/(\d+) PASS", U["summary"] or "")
    U["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not U["fails"]
except Exception as e:
    U.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
log("units", json.dumps({k: v for k, v in U.items() if k != "tb"})[:1200]); REP["checks"]["units"] = U; save()
try:
    E = e2e(["clean"], real_proc=True)
except Exception as e:
    E = {"all_pass": False, "fatal": f"{type(e).__name__}: {e}"[:600]}
log("e2e clean (real /proc)", json.dumps(E, default=str)[:1500]); REP["checks"]["e2e_clean_real_proc"] = E; save()
try:
    o = K / "rerun_out"; o.mkdir(exist_ok=True); (K / "rerun_tmp").mkdir(exist_ok=True)
    r = sh("rerun", [sys.executable, str(B / "nb0_code_export.py")],
           {"KAGGLE_IS_COMPETITION_RERUN": "1", "E23_FAKE_SERVER": str(B / "nb0_fake_server.py"), "E23_OUT": str(o),
            "E23_TMP": str(K / "rerun_tmp"), "E23_INPUT": str(K / "fin"), "RC2_SUBMISSION": str(o / "submission.json")}, 600, cwd=K)
    test = json.loads((K / "fin/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").read_text())
    sub = json.loads((o / "submission.json").read_text()) if (o / "submission.json").exists() else {}
    RR = {"rc": r["rc"], "tasks": len(sub), "want": len(test), "attempts_ok": all(len(v) == len(test[k]["test"]) and
          all(set(a) == {"attempt_1", "attempt_2"} for a in v) for k, v in sub.items()) if sub else False,
          "no_boot": not (o / "nb0_results.json").exists() or "NB0" not in json.loads((o / "nb0_results.json").read_text()).get("sections", {}),
          "tail": r["tail"][-300:]}
    RR["ok"] = RR["rc"] == 0 and RR["tasks"] == RR["want"] > 0 and RR["attempts_ok"] and RR["no_boot"]
except Exception as e:
    RR = {"ok": False, "fatal": f"{type(e).__name__}: {e}"[:600]}
log("rerun", json.dumps(RR)[:600]); REP["checks"]["rerun"] = RR
verdict({"fake_tree_ok": (U.get("fake_tree") or {}).get("ok") is True, "units_all_pass": U.get("all_pass") is True,
         "e2e_clean_real_proc": E.get("all_pass") is True, "rerun_flow_ok": RR.get("ok") is True},
        recorded={"units_summary": U.get("summary"), "e2e_gates": E.get("gates_lines")})
'''

K_E2E = r'''
# ---------------------------------------------------------------- fake end-to-end fault modes __MODES__
FT, E = {}, {}
try:
    clean_env()
    FT = fake_tree()
    E = e2e(__MODES__)
except Exception as e:
    E = {"all_pass": False, "fatal": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1500:]}
log("e2e", json.dumps(E, default=str)[:2500]); REP["checks"]["fake_tree"] = FT; REP["checks"]["e2e"] = E
verdict({"fake_tree_ok": FT.get("ok") is True, **{f"e2e_{m}": bool(((E.get("modes") or {}).get(m) or {}).get("ok")) for m in __MODES__},
         "e2e_all_pass_line": E.get("all_pass") is True}, recorded={"gates": E.get("gates_lines")})
'''

KITS = {
    "real": {"body": K1, "title": "Q27 E25 NB0 Kit Real", "models": RC2_MODELS, "files": COMMON},
    "units": {"body": K2, "title": "Q27 E25 NB0 Kit Units", "models": [RC2_MODELS[0]], "files": {**COMMON, **UNIT_FILES}},
    "e2ea": {"body": K_E2E.replace("__MODES__", json.dumps(["native", "kv_long", "mtp"])), "title": "Q27 E25 NB0 Kit E2E A",
             "models": [RC2_MODELS[0]], "files": COMMON},
    "e2eb": {"body": K_E2E.replace("__MODES__", json.dumps(["path", "shift", "repeat"])), "title": "Q27 E25 NB0 Kit E2E B",
             "models": [RC2_MODELS[0]], "files": COMMON},
}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__BLOB__", blob(spec["files"])), ("__NBSHA__", NB_SHA), ("__NAME__", name)):
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e25-nb0-kit-{name}"
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
            "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB0", NB_SHA[:8], flush=True)
    return out, code


if __name__ == "__main__":
    built = {n: build(n) for n in KITS}
    if "--dry" in sys.argv:
        base = Path(os.environ.get("KIT_LOCAL_ROOT", r"D:/ARC AGI 2/_scratch/e25_nb0_local/kit_dry"))
        inp = os.environ.get("KIT_LOCAL_INPUT", r"D:/ARC AGI 2/_scratch/e13_rc2_local/input")
        procs = {}
        for i, (n, (out, code)) in enumerate(built.items()):
            if n not in (sys.argv[sys.argv.index("--dry") + 1:] or list(built)):
                continue
            d = base / n
            d.mkdir(parents=True, exist_ok=True)
            (d / "kit.py").write_text(code, encoding="utf-8")
            env = {**os.environ, "KIT_INPUT": inp, "KIT_TMP": str(d / "tmp"), "KIT_WORK": str(d / "work"),
                   "KIT_PORT_BASE": str(19100 + 100 * i), "KIT_LOCAL_TMP": str(d / "nb0tmp"),
                   **({"KIT_DRY_REHEARSAL": str(MERLIN_SRC)} if n == "real" else {})}
            procs[n] = (subprocess.Popen([sys.executable, str(d / "kit.py")], stdout=open(d / "stdout.txt", "w"),
                                         stderr=subprocess.STDOUT, env=env), d)
        for n, (p, d) in procs.items():
            p.wait(timeout=7200)
            txt = (d / "stdout.txt").read_text(encoding="utf-8", errors="replace")
            line = next((l for l in txt.splitlines() if l.startswith("@@KIT")), None)
            print(f"DRY {n}: rc {p.returncode} | {line}", flush=True)
