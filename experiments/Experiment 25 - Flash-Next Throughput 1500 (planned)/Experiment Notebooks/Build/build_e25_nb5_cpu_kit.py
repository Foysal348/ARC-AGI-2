"""E25 NB5 Kaggle CPU kits (user GO 2026-10-31): independent CPU kernels in parallel on foysalemonshanto, pinned Py3.12
image, no GPU, no internet. Pattern: build_e25_nb3_cpu_kit.py / Experiment 13 build_e13_rc2_cpu_kit.py (RC2 kit v1 ALL OK).
Each kernel is fail-soft, writes its own JSON report and prints exactly one line `@@KIT <name> ALL OK@@` or
`@@KIT <name> NOT OK: <items>@@`. The NB5 notebook under test is Q27-XHIGH-E25_NB5_QUALITY.ipynb (profile NB5QUALITY_R2, the
paired FP8 / BF16 KV accuracy test; lead 2026-10-31), embedded (lzma + base64, sha-checked). Kit slugs q27-e25-nb5q-kit-*
(the earlier q27-e25-nb5-kit-* folders of the frozen six stay as they are).
R2 additions: KIT-5 (the payload is accepted at ANY sha-OK path under /kaggle/input, the path recorded; the notebook's own
search order is mirrored), KIT-6 (the e2e sets NB5_FAKE_HOST_FLOOR_MIB=1024, FAKE only), K1 applies + verifies the #60490
backport on the real Merlin tree and builds the R2 boots (+1.0 GiB pin, table seqs, lever env, CAPS16), e2e mode 'quality'.

  K1 q27-e25-nb5-kit-real   the notebook's own cells on the REAL mounts: the Duck payload from
                            datasets/foysalemonshanto/q27-e15b-payload (lookup order foysalemonshanto -> esfersami50, sha256,
                            the path used recorded), CELL 3 unpack + manifest sha + sandbox self-test (incl. the memory limit);
                            RC2 inputs at the exact Model paths, merge (sizes) + the full background sha256, the prompt render
                            with the pinned tokenizer / template hashes; the real Merlin offline install; launch.build() +
                            host_phase() for the FP8, BF16 and FP8 + MTP-off (--speculative-config None) boots, no server
                            started; the MTP-off argv differs from FP8 only in --speculative-config (last value None); the
                            commit-turn A/C test on the real checkpoint tokenizer + template (98.3 % = 12,248 cached tokens
                            reproduced, variant_c_ids == the re-tokenized C); the per-request budget guard on real renders of
                            long tool histories; CELL 0's fallback submission = full schema on the real test file, commit and
                            rerun (subprocess, KAGGLE_IS_COMPETITION_RERUN=1 -> RERUN_ALL on the test source).
                            Recorded only: `vllm serve --help=all` (on CPU it is not the argument list: RC2 kit v1), the
                            argument parse in the Merlin venv (speculative_config of each boot), FP8 argv vs RC2's measured tail.
                            Inputs: the 7 RC2 Models + foysalemonshanto/q27-e15b-payload + ARC.
  K2 q27-e25-nb5-kit-units  nb5_unit_tests.py on Py3.12 in kit mode (NB5T_KIT=1: the RC2 GPU data shipped in the kit,
                            requests.jsonl reduced to token counts; the 11 build-time checks SKIP, 126 run; NB5Q-ENV-1:
                            Merlin's tools/merlin launch.py + registry.py + profiles/flash-next.json of ad42e37 shipped,
                            sha-pinned, so the real launch.build() test runs here too).
                            Inputs: part 1 + foysalemonshanto/q27-e15b-payload + ARC.
  K3-K7 q27-e25-nb5-kit-e2e-{commit,rerun,mtp,notok,floor}: one fake-server e2e mode each (user rule 2026-10-31: long tests
                            only on Kaggle, as parallel kernels) on a fake tree made from the real part-1 small files, the
                            real competition eval challenges + solutions and the real payload.
                            Inputs: part 1 + foysalemonshanto/q27-e15b-payload + ARC.

Run: python build_e25_nb5_cpu_kit.py  ->  D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb5q-kit-<name>__foysalemonshanto/ (8 kits)
Local dry run (KIT_INPUT / KIT_TMP / KIT_WORK / KIT_E23_TMP overridden; a local input tree is assembled):
     python build_e25_nb5_cpu_kit.py --dry [kit names]    (user rule: long kits run on Kaggle only)
"""
import base64
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E25 = HERE.parents[1]
E13 = Path(r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)")
NB_PATH = HERE.parent / "Q27-XHIGH-E25_NB5_QUALITY.ipynb"
NB_TXT = NB_PATH.read_text(encoding="utf-8")
NB_SHA = hashlib.sha256(NB_PATH.read_bytes()).hexdigest()
RC2_RUN = Path(r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2")
PARTS = (E13 / "prep" / "merlin_fn_ckpt" / "kernels" / "parts.json").read_text(encoding="utf-8")
CT_RESULT = (E25 / "Experiment Results" / "commit_turn_prefix_test_2026-10-31.json").read_text(encoding="utf-8")
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
PAYLOAD_DS = "foysalemonshanto/q27-e15b-payload"     # lead 2026-10-31: foysalemonshanto cannot read esfersami50's copy
RC2_MODELS = ([f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
              ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
               "foysalemonshanto/e23-merlin-env/Other/v1/1"])
UNITS_EXPECT = {"run": 126, "skipped": 11}          # 137 checks locally (+9 Q-1/Q-2/Q-3/Q-5); 11 are build-time (builder + donors)
MERLIN_TOOLS = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src/tools/merlin")   # NB5Q-ENV-1: ad42e37 (= the e23-merlin-env tree)
MERLIN_FILES = ("launch.py", "registry.py", "profiles/flash-next.json")
SLUG = "q27-e25-nb5q-kit"


def src(name):
    return (HERE / name).read_text(encoding="utf-8")


def reduced_requests():
    """requests.jsonl without the token / text payloads (the unit tests read prompt_len, t, sec and the token COUNT)."""
    out = []
    for line in open(RC2_RUN / "traces" / "requests.jsonl", encoding="utf-8"):
        r = json.loads(line)
        n = len(r.get("tokens") or [])
        r = {k: v for k, v in r.items() if k not in ("tokens", "text", "margins")}
        r["n_tokens"] = n
        out.append(json.dumps(r))
    return "\n".join(out) + "\n"


MAKE_FAKE = r'''
import hashlib, json, shutil, sys
from pathlib import Path
INP_REAL, INP, COMP_REAL = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4])
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
ev = COMP_REAL / "arc-agi_evaluation_challenges.json"
shutil.copyfile(ev, c / "arc-agi_evaluation_challenges.json"); shutil.copyfile(ev, c / "arc-agi_test_challenges.json")
print("FAKE_TREE_OK shards", len(shards), "small", len(small), "ple", len(ple))
'''

HEAD = r'''# E25 NB5 CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, lzma, io, types
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/nb5kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E25 NB5 CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}}
def save():
    (W / f"nb5_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
FILES = json.loads(lzma.decompress(base64.b64decode("__BLOB__")).decode("utf-8"))
FILE_SHAS = __SHAS__
REP["files_ok"] = {n: hashlib.sha256(FILES[n].encode("utf-8")).hexdigest() == h for n, h in FILE_SHAS.items()}
assert all(REP["files_ok"].values()), REP["files_ok"]
assert hashlib.sha256(FILES["nb5.ipynb"].encode("utf-8")).hexdigest() == "__NBSHA__", "embedded notebook sha256"
def sh(name, args, env=None, timeout=3600, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})}, cwd=str(cwd or K),
                           encoding="utf-8", errors="replace")
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out, encoding="utf-8")
    return {"rc": rc, "sec": round(time.time() - t), "tail": out[-1500:], "_out": out}
def verdict(ok, recorded=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "recorded": recorded or {}, "minutes": round((time.time() - T0) / 60, 1)}
    save()
    bad = [k for k, v in ok.items() if not v]
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:8000], flush=True)
    print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
PAYLOAD_NAME, PAYLOAD_SHA = "q27x_e15b_duck_9e4b89285136.bin", "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566"
PAYLOAD_DIRS = ["datasets/foysalemonshanto/q27-e15b-payload", "datasets/esfersami50/q27-e15b-payload"]
def find_payload():
    """The lead's order: foysalemonshanto's copy, then esfersami50's; then (KIT-5) ANY sha-OK copy under the input root in the
    notebook's own search order (name, */name, */*/name, */*/*/name; e.g. the legacy /kaggle/input/q27-e15b-payload/);
    sha256 every time; every candidate recorded."""
    rec = {"order": [], "used": None}
    for d in PAYLOAD_DIRS:
        dd = Path(INPUT_ROOT) / d
        e = {"dir": str(dd), "exists": dd.is_dir(), "files": sorted(os.listdir(dd)) if dd.is_dir() else None}
        f = dd / PAYLOAD_NAME
        if f.is_file():
            e["sha_ok"] = hashlib.sha256(f.read_bytes()).hexdigest() == PAYLOAD_SHA
            if e["sha_ok"] and rec["used"] is None:
                rec["used"] = str(f)
        rec["order"].append(e)
    r_ = Path(INPUT_ROOT)
    rec["all_copies_under_input"] = sorted(str(p) for pat in (PAYLOAD_NAME, "*/" + PAYLOAD_NAME, "*/*/" + PAYLOAD_NAME,
                                                             "*/*/*/" + PAYLOAD_NAME) for p in r_.glob(pat))
    if rec["used"] is None:                  # KIT-5: the mount path varies per session
        for pat in (PAYLOAD_NAME, "*/" + PAYLOAD_NAME, "*/*/" + PAYLOAD_NAME, "*/*/*/" + PAYLOAD_NAME):
            for f in sorted(r_.glob(pat)):
                ok_ = hashlib.sha256(f.read_bytes()).hexdigest() == PAYLOAD_SHA
                rec.setdefault("generic", []).append({"path": str(f), "sha_ok": ok_})
                if ok_ and rec["used"] is None:
                    rec["used"] = str(f)
    rec["used_kind"] = (None if rec["used"] is None else "datasets_dir" if "/datasets/" in rec["used"].replace("\\", "/")
                        else "other_mount")
    rec["datasets_dirs"] = sorted(str(p.relative_to(r_)) for p in r_.glob("datasets/*/*")) if (r_ / "datasets").is_dir() else None
    return rec
P1R = Path(INPUT_ROOT) / "models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1"
COMPD = Path(INPUT_ROOT) / "competitions/arc-prize-2026-arc-agi-2"
if not COMPD.is_dir() and Path(INPUT_ROOT, "arc-prize-2026-arc-agi-2").is_dir():
    COMPD = Path(INPUT_ROOT, "arc-prize-2026-arc-agi-2")
def copy_tok(dst):
    dst.mkdir(parents=True, exist_ok=True)
    for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(P1R / n, dst / n)
    return {n: hashlib.sha256((dst / n).read_bytes()).hexdigest() for n in ("tokenizer.json", "chat_template.jinja")}
def full_schema(sub, test):
    return (set(sub) == set(test) and all(len(sub[t]) == len(test[t]["test"]) for t in test)
            and all(isinstance(a.get(k), list) and a[k] and all(isinstance(r, list) for r in a[k])
                    for t in sub for a in sub[t] for k in ("attempt_1", "attempt_2")))
if os.name == "nt":      # local dry run only: the Exp 21 harness shims (65 kB sandbox bootstrap as a file, no os.killpg)
    import tempfile as _tf, signal as _sig
    _real = subprocess.Popen
    _bdir = Path(_tf.gettempdir()) / "e14_harness_bootstrap"; _bdir.mkdir(parents=True, exist_ok=True)
    class _Shim(_real):
        def __init__(self, args, *a, **kw):
            if isinstance(args, list) and len(args) >= 5 and args[1:4] == ["-I", "-S", "-c"]:
                f = _bdir / "sandbox_bootstrap.py"
                if not f.exists() or f.read_text(encoding="utf-8") != args[4]:
                    f.write_text(args[4], encoding="utf-8")
                args = [args[0], "-I", "-S", str(f)] + args[5:]
            super().__init__(args, *a, **kw)
    subprocess.Popen = _Shim
    if not hasattr(os, "killpg"):
        os.killpg = lambda pid, sig: os.kill(pid, 9)
        if not hasattr(_sig, "SIGKILL"):
            _sig.SIGKILL = 9
'''

K1 = r'''
# ---------------------------------------------------------------- the NB5 notebook's own cells on the real mounts
PAY = find_payload(); REP["checks"]["payload"] = PAY; save()
log("payload", json.dumps(PAY)[:1200])
NB = json.loads(FILES["nb5.ipynb"])
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
HEADS = [c.splitlines()[0][:100] for c in CODE]
WORK = K / "working"; WORK.mkdir()
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5_", "NB5H_", "E12_", "E15B_", "Q27X_", "RC2_", "E23_"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
os.environ.update(E23_INPUT=INPUT_ROOT, E23_TMP=os.environ.get("KIT_E23_TMP", "/tmp"), Q27X_WORKING=str(WORK), Q27X_ARC_ROOT=str(COMPD),
                  RC2_CPU_KIT="1", RC2_SHM=str(K / "shm"), E12_DIR=str(K / "e12_duck"))
(K / "shm").mkdir()
if INPUT_ROOT != "/kaggle/input":
    os.environ["Q27X_PAYLOAD_DIR"] = INPUT_ROOT          # dry run: the notebook's own search root (Kaggle: /kaggle/input)
ENV0 = dict(os.environ)                               # before CELL 1 sets its profile defaults
G = {"__name__": "__main__"}
CELLS = {}
for i in (0, 1, 2, 3, 4):        # CELL 0, 1, 3 (Duck), task order, 5 (engine definitions); never 6/7 (run) or 8/9 (boots)
    t = time.time()
    try:
        exec(compile(CODE[i], f"<nb5 cell {i}>", "exec"), G)
        CELLS[i] = {"ok": True, "sec": round(time.time() - t, 1), "head": HEADS[i]}
    except BaseException as e:
        CELLS[i] = {"ok": False, "head": HEADS[i], "error": f"{type(e).__name__}: {e}"[:1500], "tb": traceback.format_exc()[-3000:]}
        if i == 2 and "E12A" not in G:   # the engine cell needs E12A.ARC2_TOOL_DESCRIPTION only: stub, recorded
            G["E12A"] = types.SimpleNamespace(ARC2_TOOL_DESCRIPTION="(kit stub: the Duck package did not load)")
            CELLS[i]["stub"] = "E12A"
    log("cell", i, json.dumps({k: v for k, v in CELLS[i].items() if k != "tb"})[:1200])
REP["checks"]["cells"] = CELLS
D3 = {"payload_used": G.get("Q27X_PAYLOAD_USED"), "payload_slug": G.get("Q27X_PAYLOAD_SLUG"),
      "sandbox_selftest": G.get("E10_SANDBOX_SELFTEST"), "manifest_files": len(G.get("_E12_MANIFEST") or {}),
      "config_payload_used_is_record": (G.get("CONFIG") or {}).get("nb5", {}).get("payload_used") is G.get("Q27X_PAYLOAD_USED")}
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
step("inputs", lambda: (G["nb5_rc2_inputs"](), {"inputs": G["R"]["inputs"]})[1])
step("prompts", lambda: (G["nb5_rc2_prompts"](), {"merge": G["R"]["sections"]["A0"]["merge"], "host": G["R"]["sections"]["A0"]["host"],
                                                   "prompt_render": {k: v for k, v in G["R"]["prompt_render"].items() if "text" not in k}})[1])
step("install", lambda: (lambda e: {"kv": dict(e["kv"]), "cuda_home": e["cuda_home"]})(G["merlin_install"]()))
step("patch_60490", lambda: G["nb5_apply_60490"]())       # R2 lever a: pre-sha, strict apply, post-sha, marker, py_compile, origin
PARSE = """
import json, sys
argv = json.load(open(sys.argv[1]))
try:
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except Exception:
        from vllm.utils import FlexibleArgumentParser
    from vllm.engine.arg_utils import AsyncEngineArgs
    p = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
    rest = argv[argv.index("serve") + 1:]
    if rest and not rest[0].startswith("--"):
        rest = ["--model", rest[0]] + rest[1:]
    ns, unknown = p.parse_known_args(rest)
    out = {"ok": True, "speculative_config": repr(getattr(ns, "speculative_config", "ABSENT")),
           "kv_cache_dtype": getattr(ns, "kv_cache_dtype", None), "max_model_len": getattr(ns, "max_model_len", None),
           "unknown": unknown[:60]}
except BaseException as e:
    import traceback
    out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1200:]}
print("PARSE_JSON " + json.dumps(out, default=str))
"""
(K / "parse_argv.py").write_text(PARSE)
BOOTS = {}
G["NB5_BOOT_LEVERS"] = dict(G.get("NB5_LEVER_ENV") or {})        # R2: the lever env of the first ladder step
for name, kv, port, extra in (("kit_fp8", "fp8", 8001, []), ("kit_bf16", "bf16", 8000, []),
                              ("kit_fp8_nomtp", "fp8", 8002, list(G.get("NB5_MTP_OFF_ARGS") or ["--speculative-config", "None"]))):
    b = {}
    try:
        # R2: the ladder's first step = RC2 pin + 1.0 GiB per card, launch --max-num-seqs from the table, CAPS16 (NB5_EXTRA_ARGS)
        _seqs = G["NB5L"].seqs_for(kv, 1.0)
        G["NB5_BOOT_EXTRA"] = list(G["NB5_EXTRA_ARGS"]) + extra + ["--max-num-seqs", str(_seqs)]
        b["r2"] = {"pin_gib": 1.0, "max_num_seqs": _seqs, "levers": dict(G["NB5_BOOT_LEVERS"])}
        s = G["merlin_fn_server"](name, kv, port, (G["KV_FP8"] if kv == "fp8" else G["KV_BF16"]) + 2 ** 30)
        b.update(ok=True, host_phase=G["HOST_PHASE"].get(name), help=G["HELP"].get(name), argv=s.argv, server_started=s.proc is not None)
        _al = G["NB5L"].argv_last(s.argv)
        b["r2"].update(argv_kv_bytes=_al.get("--kv-cache-memory-bytes"), argv_seqs=_al.get("--max-num-seqs"),
                       argv_captures=json.loads(_al.get("--compilation-config") or "{}").get("cudagraph_capture_sizes"),
                       env_lever=s.env.get("MERLIN_SHARE_DRAFT_WORKSPACE"),
                       ok=(str(_al.get("--kv-cache-memory-bytes")) == str((G["KV_FP8"] if kv == "fp8" else G["KV_BF16"]) + 2 ** 30)
                           and str(_al.get("--max-num-seqs")) == str(_seqs) and s.env.get("MERLIN_SHARE_DRAFT_WORKSPACE") == "1"
                           and G["NB5L"].captures_cover(json.loads(_al.get("--compilation-config") or "{}").get("cudagraph_capture_sizes"), _seqs)))
        (K / f"argv_{name}.json").write_text(json.dumps(s.argv))
        penv = {k: v for k, v in s.env.items() if k != "LD_PRELOAD"}
        penv["CUDA_VISIBLE_DEVICES"] = ""
        r = sh("parse_" + name, [G["MERLIN_ENG"]["kv"]["MERLIN_VENV"] + "/bin/python", str(K / "parse_argv.py"), str(K / f"argv_{name}.json")],
               penv, 600)
        b["vllm_parse"] = next((json.loads(l[11:]) for l in r["_out"].splitlines() if l.startswith("PARSE_JSON ")), {"ok": False, "tail": r["tail"]})
    except BaseException as e:
        b.update(ok=False, error=f"{type(e).__name__}: {e}"[:1500], host_phase=(G.get("HOST_PHASE") or {}).get(name), tb=traceback.format_exc()[-2500:])
    BOOTS[name] = b
    log("boot", name, json.dumps({k: v for k, v in b.items() if k not in ("argv", "tb")}, default=str)[:1500])
G["NB5_BOOT_EXTRA"] = []
try:
    R1["help_text_head"] = (G["L"]._HELP.get("merlin") or "")[:2500]
except Exception as e:
    R1["help_text_head"] = repr(e)
REP["checks"]["boots"] = BOOTS; save()
NB5L = G.get("NB5L")
A8, AN = (BOOTS.get("kit_fp8") or {}).get("argv"), (BOOTS.get("kit_fp8_nomtp") or {}).get("argv")
MTP_DIFF = NB5L.argv_diff(A8, AN) if (A8 and AN) else None
RC2_DIFF = NB5L.argv_diff(G["NB5_RC2_FP8_ARGV_TAIL"], A8[-60:]) if A8 else None
REP["checks"]["argv"] = {"mtp_off_vs_fp8": MTP_DIFF, "fp8_vs_rc2_tail": RC2_DIFF,
                         "nomtp_tail": AN[-4:] if AN else None}
step("sha256", lambda: (lambda ok_why: {"sha_ok": ok_why[0], "why": ok_why[1], "detail": G["R"]["setup"].get("sha256")})(G["sha_gate"](3600)))

# ---------------------------------------------------------------- commit-turn A / C on the real checkpoint tokenizer + template
CTR = {}
try:
    tok = K / "fn_tok_p1"
    CTR["p1_shas"] = copy_tok(tok)
    if (Path(str(G.get("TOKDIR", ""))) / "tokenizer.json").is_file():
        CTR["merged_copy_same"] = all(hashlib.sha256((Path(G["TOKDIR"]) / n).read_bytes()).hexdigest() == CTR["p1_shas"][n] for n in CTR["p1_shas"])
    (K / "commit_turn_prefix_test.py").write_text(FILES["commit_turn_prefix_test.py"], encoding="utf-8")
    _argv, _so = sys.argv, sys.stdout
    sys.argv, sys.stdout = [str(K / "commit_turn_prefix_test.py"), str(tok)], io.StringIO()
    try:
        CT = runpy.run_path(str(K / "commit_turn_prefix_test.py"), run_name="ct")
    finally:
        sys.argv, sys.stdout = _argv, _so
    rec = json.loads(FILES["ct_result.json"])
    TA_, TC_ = NB5L.tails(lambda s_: CT["TOK"].encode(s_, add_special_tokens=False).ids)
    CTR.update(reproduced=CT["res"] == rec, rows=CT["res"]["rows"], tails=[TA_, TC_],
               c_ids_equal=NB5L.variant_c_ids(CT["A"], TA_, TC_) == CT["C"],
               c_common=CT["common"](CT["cached"], NB5L.variant_c_ids(CT["A"], TA_, TC_)), b_common=CT["common"](CT["cached"], CT["B"]))
    # budget guard on real renders of Duck-shaped long tool histories (the notebook's own lib + config)
    def hist(n, chars):
        m = [dict(x) for x in CT["msgs"][:2]]
        for k in range(n):
            m += [{"role": "assistant", "content": "", "reasoning_content": f"Step {k}: rows and colours.",
                   "tool_calls": [{"type": "function", "function": {"name": "run_python", "arguments": {"code": f"print({k})"}}}]},
                  {"role": "tool", "content": " ".join(str((k * 7 + j) % 10) for j in range(chars // 2))}]
        return m
    plen = lambda m: len(CT["TOK"].encode(CT["render"](m), add_special_tokens=False).ids)
    b0, rs, cap, fl = G["NB5_ATTEMPT"]["b0"], G["NB5_RESERVE"], G["NB5_CAP"], G["NB5_THINK_FLOOR"]
    ps, pm, pl = plen(hist(1, 2000)), plen(hist(20, 3300)), plen(hist(36, 3300))
    g1 = NB5L.context_guard(ps, cap, 0, b0, rs, False, floor=fl)
    g2 = NB5L.context_guard(pm, cap, 20000, b0, rs, False, floor=fl)
    g3 = NB5L.context_guard(pl, cap, 10000, b0, rs, False, floor=fl)
    g4 = NB5L.context_guard(pl, 4000, 108000, b0, rs, True, floor=fl)
    def formula(p, g, req):
        return g["safe_output_budget"] == max(0, min(req, 131072 - p - 2048)) and p + g["cap"] <= 131072
    CTR["guard"] = {"prompts": [ps, pm, pl], "g1": g1, "g2": g2, "g3": g3, "g4": g4, "floor": fl, "cap": cap,
                    "ok": (fl == 60000 and not g1["clamped"] and g1["cap"] == cap and not g1["floor_event"]
                           and g2["clamped"] and formula(pm, g2, cap) and not g2["floor_event"]
                           and g3["budget_limited"] and formula(pl, g3, cap) and g3["new_budget"] == min(b0, 10000 + g3["cap"] + rs)
                           and not g4["floor_event"] and g4["cap"] <= 4000)}
except BaseException as e:
    CTR.update(error=f"{type(e).__name__}: {e}"[:1200], tb=traceback.format_exc()[-2500:])
REP["checks"]["commit_turn"] = CTR; save()
log("commit turn", json.dumps({k: v for k, v in CTR.items() if k not in ("tb", "rows")}, default=str)[:1800])

# ---------------------------------------------------------------- dual mode: CELL 0's fallback on the real test file
SUB = {}
try:
    test = json.load(open(COMPD / "arc-agi_test_challenges.json"))
    sub = json.load(open(WORK / "submission.json"))
    SUB["commit"] = {"tasks": len(sub), "test_tasks": len(test), "full_schema": full_schema(sub, test), "profile": G.get("PROFILE")}
    RR = K / "rerun"; RR.mkdir()
    (RR / "c0.py").write_text(CODE[0], encoding="utf-8"); (RR / "c1.py").write_text(CODE[1], encoding="utf-8")
    (RR / "run.py").write_text("import json, sys\nG = {'__name__': '__main__'}\nfor f in sys.argv[1:3]:\n"
                               "    exec(compile(open(f, encoding='utf-8').read(), f, 'exec'), G)\n"
                               "print('RERUN_JSON ' + json.dumps({'profile': G['PROFILE'], 'source': G['P']['source'], "
                               "'phases': G['P']['nb5_phases'], 'n_tasks': len(G['test_challenges'])}, default=str))\n")
    env = dict(ENV0)
    env.update(KAGGLE_IS_COMPETITION_RERUN="1", Q27X_WORKING=str(RR / "working"))
    r = subprocess.run([sys.executable, str(RR / "run.py"), str(RR / "c0.py"), str(RR / "c1.py")], capture_output=True, text=True,
                       env=env, timeout=600, encoding="utf-8", errors="replace")
    (W / "kit_rerun_cells.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    info = next((json.loads(l[11:]) for l in r.stdout.splitlines() if l.startswith("RERUN_JSON ")), {"tail": (r.stdout + r.stderr)[-1200:]})
    sub2 = json.load(open(RR / "working" / "submission.json")) if (RR / "working" / "submission.json").exists() else {}
    SUB["rerun"] = {**info, "rc": r.returncode, "full_schema": full_schema(sub2, test), "tasks": len(sub2),
                    "ok": r.returncode == 0 and info.get("profile") == "RERUN_ALL" and info.get("source") == "test"
                    and info.get("phases") == [["EVAL", "all", None, "on", "C"]] and full_schema(sub2, test)}
    SUB["comp_files"] = {p.name: [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()] for p in sorted(COMPD.glob("*.json"))}
except BaseException as e:
    SUB.update(error=f"{type(e).__name__}: {e}"[:1200], tb=traceback.format_exc()[-2000:])
REP["checks"]["submission"] = SUB; save()
log("submission", json.dumps(SUB, default=str)[:1500])

B_ = BOOTS
pu = (D3.get("payload_used") or {}).get(PAYLOAD_NAME) or {}
verdict({
    "payload_sha_ok_under_input": PAY["used"] is not None,
    "duck_cell_loaded": bool(CELLS.get(2, {}).get("ok")) and D3["manifest_files"] > 0,
    "notebook_used_that_payload_path": pu.get("path") is not None and Path(pu["path"]).resolve() == Path(PAY["used"] or "/none").resolve(),
    "sandbox_selftest_ok_incl_memory_limit": bool((D3.get("sandbox_selftest") or {}).get("ok")),
    "cells_0_1_3_5_ran": all(CELLS.get(i, {}).get("ok") for i in (0, 1, 3, 4)),
    "inputs_found": bool(((R1.get("inputs") or {}).get("inputs") or {}).get("merlin_inputs_ok")),
    "merge_ok": bool(((G.get("R") or {}).get("sections", {}).get("A0", {}).get("merge") or {}).get("ok")),
    "prompt_hashes_ok": bool(((R1.get("prompts") or {}).get("prompt_render") or {}).get("hashes_ok")),
    "sha256_all_ok": bool((R1.get("sha256") or {}).get("sha_ok")),
    "merlin_install_ok": bool((R1.get("install") or {}).get("ok")),
    "host_phase_fp8_ok": (B_.get("kit_fp8") or {}).get("host_phase") == "ok",
    "host_phase_bf16_ok": (B_.get("kit_bf16") or {}).get("host_phase") == "ok",
    "host_phase_fp8_mtp_off_ok": (B_.get("kit_fp8_nomtp") or {}).get("host_phase") == "ok",
    "r2_boot_argv_pin_seqs_captures_lever": all(((B_.get(n) or {}).get("r2") or {}).get("ok") for n in ("kit_fp8", "kit_bf16", "kit_fp8_nomtp")),
    "patch_60490_applied_verified": bool((R1.get("patch_60490") or {}).get("ok")) and (R1.get("patch_60490") or {}).get("sha_after")
        == "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b",
    "mtp_off_argv_only_speculative_config_None": bool(MTP_DIFF) and list(MTP_DIFF["changed"]) == ["--speculative-config"]
        and MTP_DIFF["changed"]["--speculative-config"][1] == "None" and not MTP_DIFF["added"] and not MTP_DIFF["removed"],
    "no_server_started": bool(B_) and not any((b or {}).get("server_started") for b in B_.values()),
    "commit_turn_A_C_reproduced_real_ckpt": CTR.get("reproduced") is True and CTR.get("c_ids_equal") is True and CTR.get("c_common") == 12248
        and CTR.get("b_common") == 3,
    "budget_guard_real_renders": bool((CTR.get("guard") or {}).get("ok")),
    "submission_commit_full_schema": bool((SUB.get("commit") or {}).get("full_schema")),
    "submission_rerun_profile_full_schema": bool((SUB.get("rerun") or {}).get("ok")),
}, recorded={"payload_used": PAY.get("used"), "payload_used_kind": PAY.get("used_kind"), "esfersami50_visible": any(e["exists"] for e in PAY["order"] if "esfersami50" in e["dir"]),
             "help_check_cpu": {n: (b or {}).get("help") for n, b in B_.items()},
             "vllm_parse_speculative_config": {n: ((b or {}).get("vllm_parse") or {}).get("speculative_config") or ((b or {}).get("vllm_parse") or {}).get("error")
                                               for n, b in B_.items()},
             "fp8_argv_vs_rc2_tail": RC2_DIFF})
'''

K2_SETUP = r'''
# ---------------------------------------------------------------- shared setup: the kit files, the payload, the competition data
PAY = find_payload(); REP["checks"]["payload"] = PAY; save()
log("payload", json.dumps(PAY)[:1200])
B = K / "build"; B.mkdir()
for n in ("nb5_lib.py", "nb5_unit_tests.py", "nb5_e2e.py", "nb5_fake_server.py", "commit_turn_prefix_test.py", "ct_result.json"):
    if n in FILES:                                        # each kit carries only what it runs
        (B / n).write_text(FILES[n], encoding="utf-8", newline="\n")
(B / "nb.ipynb").write_text(FILES["nb5.ipynb"], encoding="utf-8", newline="\n")
RC = K / "rc2"
for rel in [r_[4:] for r_ in FILES if r_.startswith("rc2/")]:
    (RC / rel).parent.mkdir(parents=True, exist_ok=True)
    (RC / rel).write_text(FILES["rc2/" + rel], encoding="utf-8", newline="\n")
(K / "parts.json").write_text(FILES["parts.json"]); (K / "make_fake.py").write_text(FILES["make_fake.py"])
MT = K / "merlin_tools"                               # NB5Q-ENV-1: Merlin's launcher for the real launch.build() unit test
for rel in [r_[7:] for r_ in FILES if r_.startswith("merlin/")]:
    (MT / rel).parent.mkdir(parents=True, exist_ok=True)
    (MT / rel).write_text(FILES["merlin/" + rel], encoding="utf-8", newline="\n")
U, E = {}, {}
try:
    U["comp_files"] = {p.name: [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()] for p in sorted(COMPD.glob("*.json"))}
    U["comp_has_eval_solutions"] = (COMPD / "arc-agi_evaluation_solutions.json").is_file()
    for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5", "E12_", "E15B_", "Q27X_", "RC2_", "E23_", "FAKE_"))] + [
            "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
        os.environ.pop(k_, None)
except BaseException as e:
    U.update(setup_fatal=f"{type(e).__name__}: {e}"[:800])
PAY_OK = PAY["used"] is not None                    # KIT-5: any sha-OK copy under the input root (path + kind recorded)
'''

K2_UNITS = r'''
# ---------------------------------------------------------------- nb5_unit_tests.py on this image (kit mode)
try:
    U["tok_shas"] = copy_tok(K / "fn_tok")
    r = sh("units", [sys.executable, str(B / "nb5_unit_tests.py")],
           {"NB5T_KIT": "1", "NB5T_RC2": str(RC), "NB5T_TOKDIR": str(K / "fn_tok"), "NB5T_PAYLOAD": str(PAY["used"]),
            "NB5T_NB": str(B / "nb.ipynb"), "NB5T_CT_RESULT": str(B / "ct_result.json"), "NB5T_MERLIN_TOOLS": str(MT),
            "PYTHONIOENCODING": "utf-8"}, 3600, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS(?: \((\d+) build-time)?", l) for l in lines if re.match(r"\d+/\d+ PASS", l)), None)
    U.update(rc=r["rc"], sec=r["sec"], fails=[l for l in lines if l.startswith("FAIL ")][:40], skips=[l for l in lines if l.startswith("SKIP ")],
             summary=m.group(0) if m else r["tail"][-800:], passed=int(m.group(1)) if m else None, total=int(m.group(2)) if m else None,
             skipped=int(m.group(3) or 0) if m else None)
    U["all_pass"] = bool(m) and r["rc"] == 0 and U["passed"] == U["total"] == __UEXP_RUN__ and U["skipped"] == __UEXP_SKIP__ and not U["fails"]
except BaseException as e:
    U.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
REP["checks"]["units"] = U; save()
log("units", json.dumps({k: v for k, v in U.items() if k not in ("tb", "skips")}, default=str)[:1800])
verdict({"payload_sha_ok_under_input": PAY_OK, "units_all_pass": U.get("all_pass") is True},
        recorded={"units": U.get("summary"), "skips": U.get("skips"), "payload_used": PAY.get("used")})
'''

K2_E2E = r'''
# ---------------------------------------------------------------- one fake-server e2e mode (__MODE__) on a fake tree from the real part 1
mode = "__MODE__"
U["fake_tree"] = sh("fake_tree", [sys.executable, str(K / "make_fake.py"), INPUT_ROOT, str(K / "fin"), str(K / "parts.json"), str(COMPD)])
U["fake_tree"].pop("_out")
log("fake tree", json.dumps(U["fake_tree"])[:600])
try:
    r = sh("e2e_" + mode, [sys.executable, str(B / "nb5_e2e.py"), str(B / "nb.ipynb"), mode],
           {"NB5_E2E_INPUT": str(K / "fin"), "NB5_E2E_DATA": str(COMPD), "NB5_E2E_PAYSRC": str(Path(PAY["used"]).parent),
            "NB5_E2E_ROOT": str(K / "e2e"), "NB5_E2E_MAX_SEC": "3600" if mode == "quality" else "2400", "PYTHONIOENCODING": "utf-8"},
           4200 if mode == "quality" else 3000, cwd=B)
    lines = r["_out"].splitlines()
    E[mode] = {"rc": r["rc"], "sec": r["sec"], "ok": r["rc"] == 0 and "@@NB5_E2E_OK@@" in r["_out"],
               "summary": next((l for l in lines if re.match(r"\d+/\d+ PASS", l)), None),
               "fails": [l[:400] for l in lines if "e2e FAIL]" in l][:30], "tail": r["tail"][-1200:]}
except BaseException as e:
    E[mode] = {"ok": False, "fatal": f"{type(e).__name__}: {e}"[:600]}
log("e2e", mode, json.dumps({k: v for k, v in E[mode].items() if k != "tail"})[:1500]); REP["checks"]["e2e"] = E; REP["checks"]["setup"] = U; save()
verdict({"payload_sha_ok_under_input": PAY_OK,
         "competition_eval_challenges_and_solutions": bool(U.get("comp_has_eval_solutions")),
         "fake_tree_from_real_part1": (U.get("fake_tree") or {}).get("rc") == 0,
         f"e2e_{mode}": bool(E[mode].get("ok"))},
        recorded={"e2e": E[mode].get("summary"), "comp_files": U.get("comp_files"), "payload_used": PAY.get("used")})
'''

UNIT_FILES = {"nb5.ipynb": NB_TXT, "ct_result.json": CT_RESULT, "parts.json": PARTS, "make_fake.py": MAKE_FAKE,
                        **{n: src(n) for n in ("nb5_lib.py", "nb5_unit_tests.py", "nb5_e2e.py", "nb5_fake_server.py",
                                               "commit_turn_prefix_test.py")},
                        "rc2/logs/server_merlin_fp8.log": (RC2_RUN / "logs/server_merlin_fp8.log").read_text(encoding="utf-8", errors="replace"),
                        "rc2/logs/server_merlin_bf16.log": (RC2_RUN / "logs/server_merlin_bf16.log").read_text(encoding="utf-8", errors="replace"),
                        "rc2/rc2_results.json": (RC2_RUN / "rc2_results.json").read_text(encoding="utf-8"),
                        "rc2/telemetry.jsonl": (RC2_RUN / "telemetry.jsonl").read_text(encoding="utf-8"),
                        "rc2/traces/requests.jsonl": reduced_requests(),
                        **{"merlin/" + n: (MERLIN_TOOLS / n).read_bytes().decode("utf-8") for n in MERLIN_FILES}}
E2E_FILES = {"nb5.ipynb": NB_TXT, "parts.json": PARTS, "make_fake.py": MAKE_FAKE,
             **{n: src(n) for n in ("nb5_e2e.py", "nb5_fake_server.py")}}
E2E_MODES = ("quality", "commit", "rerun", "mtp", "notok", "floor")
KITS = {
    "real": {"body": K1, "title": "Q27 E25 NB5Q Kit Real", "models": RC2_MODELS,
             "files": {"nb5.ipynb": NB_TXT, "commit_turn_prefix_test.py": src("commit_turn_prefix_test.py"), "ct_result.json": CT_RESULT}},
    "units": {"body": K2_SETUP + K2_UNITS, "title": "Q27 E25 NB5Q Kit Units", "models": [RC2_MODELS[0]], "files": UNIT_FILES},
    **{f"e2e-{m}": {"body": K2_SETUP + K2_E2E.replace("__MODE__", m), "title": f"Q27 E25 NB5Q Kit E2E {m.capitalize()}",
                   "models": [RC2_MODELS[0]], "files": E2E_FILES} for m in E2E_MODES},
}


def build(name):
    spec = KITS[name]
    files = spec["files"]
    assert "\r" not in "".join(v for k, v in files.items() if not k.startswith("rc2/")), name
    blob = base64.b64encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    shas = {n: hashlib.sha256(v.encode("utf-8")).hexdigest() for n, v in files.items()}
    code = HEAD + spec["body"]
    for k, v in (("__SHAS__", json.dumps(shas)), ("__NBSHA__", NB_SHA), ("__NAME__", name),
                 ("__UEXP_RUN__", str(UNITS_EXPECT["run"])), ("__UEXP_SKIP__", str(UNITS_EXPECT["skipped"])), ("__BLOB__", blob)):
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
            "keywords": [], "dataset_sources": [PAYLOAD_DS], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB5", NB_SHA[:8], flush=True)
    return out, code


def dry_input(base):
    """A local /kaggle/input look-alike: the RC2 local harness tree (real part-1 small files, fake shards, empty Merlin tar),
    the eval solutions next to the challenges, and the payload in the foysalemonshanto dataset folder."""
    inp = base / "input"
    if not inp.exists():
        shutil.copytree(r"D:/ARC AGI 2/_scratch/e13_rc2_local/input", inp)
        data = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
                    r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
        shutil.copy(data / "arc-agi_evaluation_solutions.json", inp / "competitions/arc-prize-2026-arc-agi-2/")
        pd = inp / "datasets/foysalemonshanto/q27-e15b-payload"
        pd.mkdir(parents=True)
        shutil.copy(Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/"
                         r"Build/payload/q27x_e15b_duck_9e4b89285136.bin"), pd)
    return inp


if __name__ == "__main__":
    built = {n: build(n) for n in KITS}
    if "--dry" in sys.argv:
        base = Path(os.environ.get("KIT_LOCAL_ROOT", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/nb5_kit_dry"))
        inp = dry_input(base)
        for n, (out, code) in built.items():
            if len(sys.argv) > 2 and n not in sys.argv[2:]:
                continue
            d = base / n
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            (d / "kit.py").write_text(code, encoding="utf-8")
            env = {**os.environ, "KIT_INPUT": str(inp), "KIT_TMP": str(d / "tmp"), "KIT_WORK": str(d / "work"),
                   "KIT_E23_TMP": str(d / "e23tmp"), "PYTHONIOENCODING": "utf-8"}
            (d / "e23tmp").mkdir()
            r = subprocess.run([sys.executable, str(d / "kit.py")], capture_output=True, text=True, env=env, timeout=7200,
                               encoding="utf-8", errors="replace")
            (d / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            line = next((l for l in r.stdout.splitlines() if l.startswith("@@KIT")), None)
            print(f"DRY {n}: rc {r.returncode} | {line}", flush=True)
