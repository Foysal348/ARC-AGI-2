"""E25 NB2 Kaggle CPU kits (user GO 2026-10-31): two independent CPU kernels on foysalemonshanto, pinned Py3.12 image, no
GPU, no internet, exact Model mount paths. Pattern: build_e25_nb3_cpu_kit.py (NB3 kits v1 ALL OK) and the RC2 kit. Each
kernel is fail-soft, writes its own JSON report and prints exactly one line `@@KIT <name> ALL OK@@` or
`@@KIT <name> NOT OK: <items>@@`.

  K1 q27-e25-nb2-kit-real   NB2's own code (NB2_UNIT + NB2_CPU_KIT) on the REAL mounts:
                            R1 inputs at the exact paths, merge (sizes), full background sha256, prompt render + pinned
                               hashes on the real tokenizer, the embedded RC2 FP8 oracle rows installed (prompt-sha match);
                            R2 the real Merlin offline install;
                            R3 merlin_fn_server() for A, B, C, K0, CTRL, B at seqs 8 and BH (the BF16-head K2 fallback)
                               = Merlin build() + host_phase()
                               with the trailing overrides; exact diff vs RC2 FP8; `vllm serve --help=all` (recorded);
                               no server is started;
                            R4 the REAL vllm serve argparse parser (+ validate + EngineArgs.from_cli_args) inside the
                               Merlin venv on every final argv: K2/K1 + draft_lm_head_quant, --max-num-seqs 16, the extended
                               --compilation-config, --profiler-config, K0 without speculation; SpeculativeConfig field /
                               Literal checks; VLLM_SERVER_DEV_MODE -> envs flag + the /reset_prefix_cache dev route;
                               create_engine_config() attempted and recorded (CPU box, no GPU);
                            R5 capture plan from the INSTALLED registry + FULL-graph / micro-batch coverage per boot;
                            R6 page-cache drop (posix_fadvise) on the real mounts + Merlin's host gate (recorded).
                            Inputs: the 7 RC2 Models + ARC.
  K2 q27-e25-nb2-kit-units  nb2_unit_tests.py on Py3.12 + the fake-server end-to-end (nb2_e2e_check.py; FAKE-only host
                            floor override NB2_FAKE_HOST_FLOOR_MIB, KIT-1) + the rerun flow + the no-Merlin flow + the SMOKE
                            profile flow (boot B only, smoke verdict) + the two DIAG flows: SMOKE_SEED2 (boot B, window
                            seed 4321) and SMOKE_CTRL (the exact RC2 control, seed 1234), both with the fake server's
                            NB2_FAKE_LOOP_SEEDS=1234 (a seed-1234 request loops on "row? "): SEED2 must count 0 loops (the
                            seed override reached the window), CTRL 1 loop at stream 0 with the window run to its end;
                            both DIAG_DONE with the audit checks prompt_identity / stream_integrity / trace_status /
                            no_duplicate_streams True, logprobs 5 on the window streams, the interpretation rows
                            (SEED2 clean -> [1, 2], CTRL loop -> [1, 3]) and CTRL's loop record (period-1 onset,
                            64 top-logprob positions, MODEL_REPETITION_OBSERVED);
                            on a fake tree made from the real part-1 small files (RC2 kit's MAKE_FAKE,
                            verbatim). The Merlin launcher files (launch.py, registry.py, flash-next.json) the harness's
                            real build() needs, the RC2 FP8 server log, the NB2 builder and the SMOKE_SEED2 notebook ride in
                            one zlib blob (the SMOKE_CTRL notebook and nb2_code_export.py = its code with profile FULL
                            are re-derived from it and sha-checked; the pushed FULL /
                            SMOKE notebooks are not carried). Inputs: part 1 + ARC.

Run: python build_e25_nb2_cpu_kit.py  ->  D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb2-kit-{real,units}__foysalemonshanto/
Local dry run (paths overridden by KIT_INPUT / KIT_TMP / KIT_WORK): python build_e25_nb2_cpu_kit.py --dry
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
MSRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
RC2_OUT = Path(r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2")
NB = HERE.parent / "Q27-XHIGH-E25_NB2_DECODE_SMOKE_SEED2.ipynb"      # the current code (FULL / SMOKE are pushed, old code)
NB_SMOKE = HERE.parent / "Q27-XHIGH-E25_NB2_DECODE_SMOKE_CTRL.ipynb"  # re-derived in the kit from NB (two profile edits)
EXPORT = (HERE / "nb2_code_export.py").read_text(encoding="utf-8")
PARTS = (E13 / "prep" / "merlin_fn_ckpt" / "kernels" / "parts.json").read_text(encoding="utf-8")
NB_SHA = hashlib.sha256(NB.read_bytes()).hexdigest()
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
RC2_MODELS = ([f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
              ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
               "foysalemonshanto/e23-merlin-env/Other/v1/1"])
_kit = (E13 / "Experiment Notebooks" / "Build" / "build_e13_rc2_cpu_kit.py").read_text(encoding="utf-8").replace("\r\n", "\n")
_i = _kit.index("MAKE_FAKE = r'''") + len("MAKE_FAKE = r'''")
MAKE_FAKE = _kit[_i:_kit.index("\n'''\n\nCODE", _i)]            # RC2 kit v1 (ALL OK on Kaggle), verbatim


def blob(files):
    """{relative path: local file} -> one zlib + base64 string (the kit re-creates the files byte for byte)."""
    d = {rel: Path(p).read_bytes().decode("utf-8") for rel, p in files.items()}         # all text (utf-8), exact bytes
    return base64.b64encode(zlib.compress(json.dumps(d).encode(), 9)).decode()


UNITS_FILES = {"kitb/Build/nb2_unit_tests.py": HERE / "nb2_unit_tests.py",
               "kitb/Build/nb2_fake_server.py": HERE / "nb2_fake_server.py", "kitb/Build/nb2_e2e_check.py": HERE / "nb2_e2e_check.py",
               "kitb/Build/build_e25_nb2.py": HERE / "build_e25_nb2.py", f"kitb/{NB.name}": NB,
               "merlin-src/tools/merlin/launch.py": MSRC / "tools/merlin/launch.py",
               "merlin-src/tools/merlin/registry.py": MSRC / "tools/merlin/registry.py",
               "merlin-src/tools/merlin/profiles/flash-next.json": MSRC / "tools/merlin/profiles/flash-next.json",
               "rc2/logs/server_merlin_fp8.log": RC2_OUT / "logs/server_merlin_fp8.log"}
FILE_SHAS = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in UNITS_FILES.items()}
# the SMOKE_CTRL notebook is NOT embedded (the blob would pass 400 kB): the kit re-derives it from the SMOKE_SEED2 notebook
# with the two one-line profile edits and checks its sha256 against the real file (derivation proven here at build time)
SMOKE_EDITS = [['NB2_DEFAULT_PROFILE = "SMOKE_SEED2"', 'NB2_DEFAULT_PROFILE = "SMOKE_CTRL"'],
               ["**This file: profile SMOKE_SEED2** (35-min hard stop, boot B as SMOKE, 8k@8 window seed 4321; "
                "diagnostic, verdict DIAG_DONE + loop count).",
                "**This file: profile SMOKE_CTRL** (35-min hard stop, the exact RC2 FP8 control boot on the failed "
                "SMOKE's 8k@8 prompts, seed 1234; diagnostic, verdict DIAG_DONE + loop count)."]]
SMOKE_SHA = hashlib.sha256(NB_SMOKE.read_bytes()).hexdigest()


def derive_smoke(full_bytes):
    nbj = json.loads(full_bytes.decode("utf-8"))
    for c in nbj["cells"]:
        out = []
        for line in c["source"]:
            for a, b in SMOKE_EDITS:
                line = line.replace(a, b)
            out.append(line)
        c["source"] = out
    return json.dumps(nbj, indent=1).encode("utf-8")


assert hashlib.sha256(derive_smoke(NB.read_bytes())).hexdigest() == SMOKE_SHA, "SMOKE_CTRL notebook is not SMOKE_SEED2 + the two profile edits"
# nb2_code_export.py is NOT embedded either (size): it is the SMOKE_SEED2 notebook's code cells joined by a blank line with
# the baked profile set back to FULL; the kit re-derives it and checks its sha256 (derivation proven here at build time)
EXPORT_EDIT = ['NB2_DEFAULT_PROFILE = "SMOKE_SEED2"', 'NB2_DEFAULT_PROFILE = "FULL"']
EXPORT_SHA = hashlib.sha256((HERE / "nb2_code_export.py").read_bytes()).hexdigest()


def derive_export(nb_bytes):
    codes = ["".join(c["source"]) for c in json.loads(nb_bytes.decode("utf-8"))["cells"] if c["cell_type"] == "code"]
    return "\n\n".join(codes).replace(*EXPORT_EDIT).encode("utf-8")


assert hashlib.sha256(derive_export(NB.read_bytes())).hexdigest() == EXPORT_SHA, "nb2_code_export.py is not the SEED2 code with profile FULL"

HEAD = r'''# E25 NB2 CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, zlib
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/nb2kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E25 NB2 CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}}
def save():
    (W / f"nb2_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def sh(name, args, env=None, timeout=3600, cwd=None, inp=None):
    t = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})} if env is not None
                           else None, cwd=str(cwd or K), input=inp)
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    return {"rc": rc, "sec": round(time.time() - t), "tail": out[-1500:], "_out": out}
def verdict(ok, recorded=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "recorded": recorded or {}, "minutes": round((time.time() - T0) / 60, 1)}
    save()
    bad = [k for k, v in ok.items() if not v]
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:8000], flush=True)
    print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
'''

K1 = r'''
EXPORT_SRC = __EXPORT__
(K / "nb2_code_export.py").write_text(EXPORT_SRC, encoding="utf-8", newline="\n")
REP["export_sha256"] = hashlib.sha256(EXPORT_SRC.encode("utf-8")).hexdigest()
R1, R2, R3, R4, R5, R6 = {}, {}, {}, {}, {}, {}
G = None
# ---------------------------------------------------------------- R1: NB2 code on the real mounts
try:
    os.environ.update(NB2_UNIT="1", NB2_CPU_KIT="1", E23_OUT=str(K / "nb2_out"), E23_TMP=str(K / "nb2_tmp"), E23_INPUT=INPUT_ROOT,
                      NB2_SUBMISSION=str(K / "nb2_out/submission.json"), NB2_SHM=str(K / "nb2_shm"), NB2_BUDGET_SEC=str(6 * 3600))
    for k_ in ("E23_FAKE_SERVER", "NB2_MERLIN_SRC", "KAGGLE_IS_COMPETITION_RERUN"):
        os.environ.pop(k_, None)
    for d in ("nb2_out", "nb2_tmp", "nb2_shm"):
        (K / d).mkdir(exist_ok=True)
    t = time.time()
    SIM = os.environ.get("KIT_DRY_SIM")                # local dry run only: a Merlin source tree stands in for the install
    if SIM:
        os.environ["E23_FAKE_SERVER"] = "dry"          # the fake-tree merge (copies, fake sizes); FAKE is cleared below
    G = runpy.run_path(str(K / "nb2_code_export.py"), run_name="nb2kit")["nb2_arm"].__globals__
    if SIM:
        sys.path.insert(0, str(Path(SIM) / "tools" / "merlin"))
        import launch as _ML
        _ML.host_phase = lambda plan: None
        G["FAKE"] = None
        G["MERLIN_ENG"] = {"kv": {"MERLIN_VENV": str(K / "dry_venv"), "MERLIN_TREE": SIM, "MERLIN_SHIM": "", "MERLIN_CC": "", "MERLIN_CXX": ""},
                           "ML": _ML, "cuda_home": str(K / "dry_cuda")}
        REP["dry_sim"] = "Merlin source tree " + SIM + " (host_phase stubbed, no venv): R2-R5 code paths only"
    R = G["R"]
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"], host=R["sections"]["A0"]["host"],
              host_mem=R["sections"]["A0"]["host_mem"], prompt_render={k: v for k, v in R["prompt_render"].items() if "text" not in k},
              rc2_refs={k: v for k, v in R["rc2_refs"].items()}, rc2_rows_installed="rc2_fp8" in G["ROWS"], prompts=R["prompts"],
              sources=R.get("sources"))
    log("R1 inputs", json.dumps(R["inputs"])[:500]); log("R1 merge", json.dumps(R1["merge"])[:400])
    log("R1 rc2 refs", json.dumps(R1["rc2_refs"])[:400]); save()
except Exception as e:
    R1["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R1["tb"] = traceback.format_exc()[-2500:]
    log("R1 FATAL", R1["fatal"])
REP["checks"]["R1"] = R1; save()
# ---------------------------------------------------------------- R2: the real Merlin offline install
try:
    t = time.time()
    eng = G["merlin_install"]()
    R2 = {"ok": True, "sec": round(time.time() - t), "kv": dict(eng["kv"]), "cuda_home": eng["cuda_home"]}
except Exception as e:
    R2 = {"ok": False, "error": f"{type(e).__name__}: {e}"[:800], "tb": traceback.format_exc()[-1500:]}
log("R2 install", json.dumps(R2)[:600]); REP["checks"]["R2"] = R2; save()
# ---------------------------------------------------------------- R3: build() + host_phase() + --help for every boot
BOOTS3 = [("A", None), ("B", None), ("C", None), ("K0", None), ("CTRL", None), ("B", 8), ("BH", None)]
SERVERS = {}
for bk, seqs in BOOTS3:
    tag = bk + (str(seqs) if seqs else "")
    b = {}
    try:
        cfg = G["make_cfg"](G["BOOTS"][bk], seqs)
        s = G["merlin_fn_server"](cfg["name"], "fp8", cfg["port"], G["KV_FP8"], cfg)
        SERVERS[tag] = (cfg, s)
        b.update(ok=True, name=cfg["name"], host_phase=G["HOST_PHASE"].get(cfg["name"]), help=G["HELP"].get(cfg["name"]),
                 argv_tail=s.argv[-40:], exact_diff=cfg.get("exact_diff"), diff=cfg["diff_vs_rc2_fp8"], server_started=s.proc is not None,
                 commit_env=cfg.get("recoverssm_commit_graph_env"), env_dev_mode=s.env.get("VLLM_SERVER_DEV_MODE"),
                 env_draft_head=s.env.get("DRAFT_HEAD"))
    except Exception as e:
        b.update(ok=False, error=f"{type(e).__name__}: {e}"[:1200], tb=traceback.format_exc()[-1500:],
                 host_phase=(G or {}).get("HOST_PHASE", {}).get(f"nb2_{bk}") if G else None)
    R3[tag] = b
    log("R3", tag, json.dumps({k: v for k, v in b.items() if k not in ("argv_tail", "tb", "exact_diff")}, default=str)[:900]); save()
REP["checks"]["R3"] = R3; save()
# ---------------------------------------------------------------- R4: the real vllm serve parser on every final argv
PARSE = r"""
import json, sys, typing, traceback
req = json.loads(sys.stdin.read())
out = {}
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    import vllm.entrypoints.cli.serve as S
    p = FlexibleArgumentParser(description="kit")
    sub = p.add_subparsers(required=False, dest="subparser")
    cmd = S.cmd_init()[0]
    cmd.subparser_init(sub).set_defaults(dispatch_function=cmd.cmd)
    args = p.parse_args(["serve"] + req["argv"])
    out["parsed"] = True
    try:
        cmd.validate(args)
        out["validated"] = True
    except BaseException as e:
        out["validated"] = False; out["validate_error"] = f"{type(e).__name__}: {e}"[:400]
    from vllm.engine.arg_utils import AsyncEngineArgs
    ea = AsyncEngineArgs.from_cli_args(args)
    cc = ea.compilation_config
    pc = ea.profiler_config
    out["engine_args"] = {"max_num_seqs": ea.max_num_seqs, "max_num_batched_tokens": ea.max_num_batched_tokens,
                          "speculative_config": ea.speculative_config, "kv_cache_dtype": str(ea.kv_cache_dtype),
                          "block_size": ea.block_size, "enable_prefix_caching": ea.enable_prefix_caching,
                          "mamba_cache_mode": str(getattr(ea, "mamba_cache_mode", None)), "use_replayssm": getattr(ea, "use_replayssm", None),
                          "max_model_len": ea.max_model_len, "tp": ea.tensor_parallel_size, "pp": ea.pipeline_parallel_size,
                          "capture_sizes": list(getattr(cc, "cudagraph_capture_sizes", None) or []),
                          "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
                          "profiler": getattr(pc, "profiler", None) if pc is not None else None,
                          "torch_profiler_dir": getattr(pc, "torch_profiler_dir", None) if pc is not None else None}
    import re as _re
    from vllm.config import SpeculativeConfig
    fdict = dict(getattr(SpeculativeConfig, "__dataclass_fields__", {}))
    fields = set(fdict)
    sc = ea.speculative_config or {}
    try:
        hints = typing.get_type_hints(SpeculativeConfig)
    except BaseException:
        hints = {}
    lit = {}
    for k in ("draft_sample_method", "draft_lm_head_quant", "method"):
        ann = hints.get(k) if k in hints else getattr(fdict.get(k), "type", None)
        if isinstance(ann, str):                     # string annotation: the quoted Literal values
            vals = [a or b for a, b in _re.findall(r"'([^']+)'|\"([^\"]+)\"", ann)]
            if not vals and ann.strip():             # an alias name (e.g. DraftSampleMethod): resolve it in the module
                mod = sys.modules.get(SpeculativeConfig.__module__)
                ann = getattr(mod, ann.split("|")[0].strip(), None)
        if not isinstance(ann, str):
            flat, todo = [], list(typing.get_args(ann))
            while todo:
                x = todo.pop(0)
                sub = typing.get_args(x)
                if sub:
                    todo += list(sub)
                else:
                    flat.append(x)
            vals = [str(x) for x in flat if x is not type(None)]
        lit[k] = vals
    out["spec_check"] = {"unknown_keys": sorted(set(sc) - fields), "fields_known": len(fields),
                         "literal_ok": {k: (str(sc[k]) in lit[k]) if (k in sc and lit[k]) else None for k in ("draft_sample_method", "draft_lm_head_quant")},
                         "literals": lit}
    import vllm.envs as envs
    out["dev_mode"] = bool(envs.VLLM_SERVER_DEV_MODE)
    try:
        from vllm.entrypoints.serve import register_vllm_dev_api_routers
        from vllm.entrypoints.serve.dev.cache.api_router import router
        out["dev_route_reset_prefix_cache"] = any(getattr(r, "path", "") == "/reset_prefix_cache" for r in router.routes)
        out["dev_register_fn"] = callable(register_vllm_dev_api_routers)
    except BaseException as e:
        out["dev_route_error"] = f"{type(e).__name__}: {e}"[:300]
    if req.get("engine_config"):
        try:
            vc = ea.create_engine_config()
            out["engine_config"] = {"ok": True, "spec": str(vc.speculative_config)[:300],
                                    "capture_sizes": list(vc.compilation_config.cudagraph_capture_sizes or []),
                                    "max_num_seqs": vc.scheduler_config.max_num_seqs}
        except BaseException as e:
            out["engine_config"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
except BaseException as e:
    out["fatal"] = f"{type(e).__name__}: {e}"[:600]; out["tb"] = traceback.format_exc()[-1500:]
print("R4_JSON " + json.dumps(out, default=str))
"""
(K / "r4_parse.py").write_text(PARSE)
for tag, (cfg, s) in SERVERS.items():
    r4 = {}
    try:
        i = s.argv.index("serve")
        py = str(Path(s.argv[i - 1]).parent / "python")
        r = sh("r4_" + tag, [py, str(K / "r4_parse.py")], None, 900, inp=json.dumps({"argv": s.argv[i + 1:], "engine_config": True}))
        env_r = sh("r4_" + tag + "_env", [py, str(K / "r4_parse.py")], s.env, 900,
                   inp=json.dumps({"argv": s.argv[i + 1:], "engine_config": True}))     # with the plan's real environment
        r4 = next((json.loads(l[8:]) for l in env_r["_out"].splitlines() if l.startswith("R4_JSON ")), {"error": env_r["tail"]})
        r4["rc"] = env_r["rc"]
        r4["no_env_run_parsed"] = "\"parsed\": true" in r["_out"]
        ea = r4.get("engine_args") or {}
        sc = ea.get("speculative_config") or {}
        want_head = None if cfg["head"] in ("bf16", "none") else cfg["head"]
        exp = {"max_num_seqs": ea.get("max_num_seqs") == cfg["seqs"], "capture_sizes": ea.get("capture_sizes") == cfg["captures"],
               "cudagraph_mode": "FULL_AND_PIECEWISE" in str(ea.get("cudagraph_mode")),
               "profiler": (ea.get("profiler") == "torch" and ea.get("torch_profiler_dir") == cfg["prof_dir"]) if not cfg["exact_rc2"]
               else ea.get("profiler") is None,
               "dev_mode": r4.get("dev_mode") is (not cfg["exact_rc2"]),
               "dev_route": r4.get("dev_route_reset_prefix_cache") is True,
               "unknown_spec_keys": not (r4.get("spec_check") or {}).get("unknown_keys")}
        if cfg["k"]:
            exp["k"] = sc.get("num_speculative_tokens") == cfg["k"]
            exp["head"] = sc.get("draft_lm_head_quant") == want_head
            exp["literals"] = all(v is not False for v in ((r4.get("spec_check") or {}).get("literal_ok") or {}).values())
        else:
            exp["no_spec"] = not ea.get("speculative_config")
        r4["expect"] = exp
        r4["ok"] = r4.get("parsed") is True and r4.get("validated") is True and all(exp.values())
    except Exception as e:
        r4 = {"ok": False, "fatal": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1500:]}
    R4[tag] = r4
    log("R4", tag, json.dumps({k: r4.get(k) for k in ("ok", "parsed", "validated", "expect", "engine_config", "fatal")}, default=str)[:1200]); save()
REP["checks"]["R4"] = R4; save()
# ---------------------------------------------------------------- R5: capture plan from the INSTALLED registry
try:
    R5["capture_sizes_for"] = {f"{s}x{k}": G["capture_sizes_for"](s, k) for s, k in ((8, 3), (16, 3), (16, 2), (16, 1), (16, 0))}
    cov = {}
    for tag, (cfg, s) in SERVERS.items():
        gp = G["graph_plan"](cfg["captures"], cfg["k"], cfg["seqs"])
        mb = {f"{b_}@{n}": G["microbatch_cover"](gp, n, cfg["k"]) for b_, n in cfg["ladder"]}
        cov[tag] = {"full": gp["full_decode_sizes"], "covers_all": gp["covers_all_request_counts"], "microbatch_all_covered": all(v["covered"] for v in mb.values()),
                    "capture_source": cfg["capture_source"]}
    R5["coverage"] = cov
    R5["ok"] = (all(v[1] == "merlin_registry" for v in R5["capture_sizes_for"].values())
                and R5["capture_sizes_for"]["16x2"][0][-1] == 48 and R5["capture_sizes_for"]["8x3"][0] == [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
                and all(v["covers_all"] and v["microbatch_all_covered"] for v in cov.values()))
except Exception as e:
    R5 = {"ok": False, "fatal": f"{type(e).__name__}: {e}"[:600]}
log("R5", json.dumps(R5, default=str)[:1500]); REP["checks"]["R5"] = R5; save()
# ---------------------------------------------------------------- R1b: the full sha256, then R6 page-cache drop + host gate
try:
    t = time.time()
    ok, why = G["sha_gate"](4 * 3600)
    R1["sha"] = {"ok": ok, "why": why, "wait_sec": round(time.time() - t), "detail": G["R"]["setup"].get("sha256")}
except Exception as e:
    R1["sha"] = {"ok": False, "why": f"{type(e).__name__}: {e}"[:300]}
log("R1 sha", json.dumps(R1["sha"])[:500]); save()
try:
    h0 = G["host_mem"]()
    R6["page_cache"] = G["drop_page_cache"]()
    R6["host_before"], R6["host_after"] = h0, G["host_mem"]()
    R6["host_gate"] = G["host_gate"](0)
except Exception as e:
    R6["fatal"] = f"{type(e).__name__}: {e}"[:400]
log("R6", json.dumps({k: R6.get(k) for k in ("page_cache", "host_gate")}, default=str)[:900]); REP["checks"]["R6"] = R6
pr = R1.get("prompt_render") or {}
ok = {"inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")), "merge_ok": bool((R1.get("merge") or {}).get("ok")),
      "sha256_all_ok": bool((R1.get("sha") or {}).get("ok")), "prompt_hashes_ok": pr.get("hashes_ok") is True,
      "rc2_oracle_rows_installed": R1.get("rc2_rows_installed") is True and (R1.get("rc2_refs") or {}).get("prompt_sha_match") is True,
      "merlin_install_ok": R2.get("ok") is True,
      "host_phase_all_boots": len(R3) == len(BOOTS3) and all(b.get("host_phase") == "ok" for b in R3.values()),
      "no_server_started": not any(b.get("server_started") for b in R3.values()),
      "ctrl_exact_rc2": not any(((R3.get("CTRL") or {}).get("exact_diff") or {}).get(k) for k in ("flags_changed", "flags_added", "flags_removed", "env_changed")),
      "parser_all_boots": len(R4) == len(BOOTS3) and all(v.get("ok") for v in R4.values()),
      "registry_capture_plan_ok": R5.get("ok") is True,
      "page_cache_drop_ok": bool((R6.get("page_cache") or {}).get("supported")) and (R6.get("page_cache") or {}).get("files", 0) >= 27}
verdict(ok, recorded={"help_check": {t: b.get("help") for t, b in R3.items()},
                      "engine_config_on_cpu": {t: (v.get("engine_config") or {}).get("ok") for t, v in R4.items()},
                      "engine_config_errors": {t: (v.get("engine_config") or {}).get("error") for t, v in R4.items()},
                      "host_gate": (R6.get("host_gate") or {}).get("ok"), "commit_env": {t: b.get("commit_env") for t, b in R3.items()}})
'''

K2 = r'''
B64 = "__BLOB__"
FILE_SHAS = json.loads(__FILESHAS__)
R7 = {}
try:
    files = json.loads(zlib.decompress(base64.b64decode(B64)).decode())
    bad = []
    for rel, b in files.items():
        p = K / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b.encode("utf-8"))
        if hashlib.sha256(p.read_bytes()).hexdigest() != FILE_SHAS[rel]:
            bad.append(rel)
    R7["files"] = {"n": len(files), "sha_mismatch": bad}
    nbj = json.loads((K / "kitb" / "__NBNAME__").read_text(encoding="utf-8"))       # SMOKE_CTRL = SMOKE_SEED2 + two profile edits
    for c in nbj["cells"]:
        out = []
        for line in c["source"]:
            for a, b in json.loads(__SMOKEEDITS__):
                line = line.replace(a, b)
            out.append(line)
        c["source"] = out
    sdata = json.dumps(nbj, indent=1).encode("utf-8")
    (K / "kitb" / "__NBSMOKENAME__").write_bytes(sdata)
    R7["smoke_notebook"] = {"sha256": hashlib.sha256(sdata).hexdigest(), "want": "__SMOKESHA__"}
    R7["smoke_notebook"]["ok"] = R7["smoke_notebook"]["sha256"] == R7["smoke_notebook"]["want"]
    nbs = json.loads((K / "kitb" / "__NBNAME__").read_text(encoding="utf-8"))          # export = SEED2 code, profile FULL
    edata = "\n\n".join("".join(c["source"]) for c in nbs["cells"] if c["cell_type"] == "code").replace(*json.loads(__EXPORTEDIT__)).encode("utf-8")
    (K / "kitb" / "Build" / "nb2_code_export.py").write_bytes(edata)
    R7["export"] = {"sha256": hashlib.sha256(edata).hexdigest(), "want": "__EXPORTSHA__"}
    R7["export"]["ok"] = R7["export"]["sha256"] == R7["export"]["want"]
    (K / "make_fake.py").write_text(__MAKEFAKE__)
    (K / "parts.json").write_text(__PARTS__)
    for k_ in list(os.environ):
        if k_.startswith(("NB2_", "E23_", "RC2_", "FAKE_")) or k_ == "KAGGLE_IS_COMPETITION_RERUN":
            os.environ.pop(k_, None)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    R7["fake_tree"] = {k: v for k, v in sh("fake_tree", [sys.executable, str(K / "make_fake.py"), INPUT_ROOT, str(K / "fin"), str(K / "parts.json")],
                                            {}, 900).items() if k != "_out"}
    BD = K / "kitb" / "Build"
    r = sh("units", [sys.executable, str(BD / "nb2_unit_tests.py"), str(K / "fin"), str(K / "unit"), str(K / "merlin-src"), str(K / "rc2")], {}, 2400, cwd=BD)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    R7["units"] = {"rc": r["rc"], "sec": r["sec"], "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40],
                   "summary": m.group(0) if m else None, "all_pass": bool(m) and m.group(1) == m.group(2) and r["rc"] == 0}
    log("units", json.dumps(R7["units"])[:900]); REP["checks"]["R7"] = R7; save()
    r = sh("e2e", [sys.executable, str(BD / "nb2_e2e_check.py"), str(K / "fin"), str(K / "e2e"), str(K / "merlin-src")], {}, 5400, cwd=BD)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    try:
        st = json.loads((K / "e2e" / "out" / "nb2_results.json").read_text())["sections"].get("S2") or {}
        steps = {n: [v.get("status"), v.get("correctness")] for n, v in st.items() if isinstance(v, dict)}
    except Exception as e:
        steps = repr(e)[:200]
    R7["e2e"] = {"rc": r["rc"], "sec": r["sec"], "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40], "summary": m.group(0) if m else None,
                 "all_pass": bool(m) and m.group(1) == m.group(2) and r["rc"] == 0, "steps": steps,
                 "notebook_tail": (K / "e2e" / "run.log").read_text(errors="replace")[-1500:] if (K / "e2e" / "run.log").exists() else None}
    log("e2e", json.dumps({k: v for k, v in R7["e2e"].items() if k != "notebook_tail"})[:900]); save()
    flows = {}
    for case, inp, extra in (("rerun", "fin", {"KAGGLE_IS_COMPETITION_RERUN": "1"}), ("nomerlin", "fin_nomerlin", {}),
                             ("smoke", "fin", {"NB2_PROFILE": "SMOKE", "NB2_FAKE_HOST_FLOOR_MIB": "1024"}),
                             ("smoke_seed2", "fin", {"NB2_PROFILE": "SMOKE_SEED2", "NB2_FAKE_HOST_FLOOR_MIB": "1024",
                                                     "NB2_FAKE_LOOP_SEEDS": "1234"}),
                             ("smoke_ctrl", "fin", {"NB2_PROFILE": "SMOKE_CTRL", "NB2_FAKE_HOST_FLOOR_MIB": "1024",
                                                    "NB2_FAKE_LOOP_SEEDS": "1234"})):
        o = K / f"flow_{case}"
        for d in ("out", "tmp", "shm"):
            (o / d).mkdir(parents=True, exist_ok=True)
        env = dict(E23_FAKE_SERVER=str(BD / "nb2_fake_server.py"), NB2_MERLIN_SRC=str(K / "merlin-src"), FAKE_SPEED="300",
                   E23_OUT=str(o / "out"), E23_TMP=str(o / "tmp"), E23_INPUT=str(K / inp), NB2_SUBMISSION=str(o / "out/submission.json"),
                   NB2_SHM=str(o / "shm"), E23_WIN_SEC="5", E23_EST_SCALE="0.01", **extra)
        if not case.startswith("smoke"):
            env["NB2_BUDGET_SEC"] = "3000"             # the smoke / DIAG flows keep their own 35-min default budget
        r = sh("flow_" + case, [sys.executable, str(BD / "nb2_code_export.py")], env, 1800, cwd=BD)
        f = {"rc": r["rc"], "sec": r["sec"]}
        try:
            f["submission_tasks"] = len(json.loads((o / "out" / "submission.json").read_text()))
            if case == "nomerlin":
                Rf = json.loads((o / "out" / "nb2_results.json").read_text())
                f["s2"] = {n: v.get("status") for n, v in (Rf["sections"].get("S2") or {}).items()}
                f["source"] = Rf["prompt_render"]["source"]
            if case.startswith("smoke"):
                Rf = json.loads((o / "out" / "nb2_results.json").read_text())
                f["s2"] = {n: [v.get("status"), v.get("correctness")] for n, v in (Rf["sections"].get("S2") or {}).items()}
                f["smoke"] = Rf.get("smoke")
                f["plan"] = Rf.get("plan")
                f["budget_min"] = Rf.get("budget_min")
                f["diag_prompt_check"] = Rf.get("diag_prompt_check")
        except Exception as e:
            f["parse_error"] = repr(e)[:200]
        flows[case] = f
    R7["flows"] = flows
except Exception as e:
    R7["fatal"] = f"{type(e).__name__}: {e}"[:800]; R7["tb"] = traceback.format_exc()[-1500:]
log("flows", json.dumps(R7.get("flows"))[:600]); REP["checks"]["R7"] = R7
fl = R7.get("flows") or {}
_d2 = (fl.get("smoke_seed2") or {}).get("smoke") or {}
_dc = (fl.get("smoke_ctrl") or {}).get("smoke") or {}
_N2_CHECKS = ("prompt_identity", "stream_integrity", "trace_status", "no_duplicate_streams")   # audit N2-1 / N2-2
verdict({"embedded_files_sha_ok": (R7.get("files") or {}).get("n") == __NFILES__ and not (R7.get("files") or {}).get("sha_mismatch"),
         "ctrl_notebook_derived_sha_ok": (R7.get("smoke_notebook") or {}).get("ok") is True,
         "export_derived_sha_ok": (R7.get("export") or {}).get("ok") is True,
         "fake_tree_ok": (R7.get("fake_tree") or {}).get("rc") == 0,
         "units_all_pass": (R7.get("units") or {}).get("all_pass") is True,
         "e2e_all_pass": (R7.get("e2e") or {}).get("all_pass") is True,
         "flow_rerun": (fl.get("rerun") or {}).get("rc") == 0 and (fl.get("rerun") or {}).get("submission_tasks") == 120,
         "flow_nomerlin": (fl.get("nomerlin") or {}).get("rc") == 0 and (fl.get("nomerlin") or {}).get("submission_tasks") == 120
                          and (fl.get("nomerlin") or {}).get("s2") == {"skipped": "SKIPPED_MERGE_FAILED"}
                          and (fl.get("nomerlin") or {}).get("source") == "intel_fallback",
         "flow_smoke": (fl.get("smoke") or {}).get("rc") == 0 and (fl.get("smoke") or {}).get("submission_tasks") == 120
                       and list((fl.get("smoke") or {}).get("s2") or {}) == ["B_k2_int8_s16"]
                       and ((fl.get("smoke") or {}).get("smoke") or {}).get("verdict") == "SMOKE_PASS"
                       and (fl.get("smoke") or {}).get("budget_min") == 35,
         "flow_smoke_seed2": (fl.get("smoke_seed2") or {}).get("rc") == 0 and (fl.get("smoke_seed2") or {}).get("submission_tasks") == 120
                             and list((fl.get("smoke_seed2") or {}).get("s2") or {}) == ["B_k2_int8_s16"]
                             and (fl.get("smoke_seed2") or {}).get("budget_min") == 35
                             and _d2.get("verdict") == "DIAG_DONE" and _d2.get("loop_count") == 0 and _d2.get("seed") == 4321
                             and [x.get("seed") for x in _d2.get("streams") or []] == list(range(4321, 4329))
                             and (_d2.get("checks") or {}).get("window_ran_full") is True
                             and all((_d2.get("checks") or {}).get(k) is True for k in _N2_CHECKS)
                             and (_d2.get("logprobs") or {}).get("requested") == 5
                             and all(((x.get("logprobs") or {}).get("requested") == 5 and (x.get("logprobs") or {}).get("aligned") is True)
                                     for x in _d2.get("streams") or [{}])
                             and ((_d2.get("interpretation") or {}).get("rows_consistent_with_this_result")) == [1, 2],
         "flow_smoke_ctrl": (fl.get("smoke_ctrl") or {}).get("rc") == 0 and (fl.get("smoke_ctrl") or {}).get("submission_tasks") == 120
                            and list((fl.get("smoke_ctrl") or {}).get("s2") or {}) == ["CTRL_k3_bf16_s8"]
                            and (fl.get("smoke_ctrl") or {}).get("budget_min") == 35
                            and _dc.get("verdict") == "DIAG_DONE" and _dc.get("loop_count") == 1 and _dc.get("looped_streams") == [0]
                            and _dc.get("seed") == 1234 and (_dc.get("window") or {}).get("status") == "FAIL_GARBAGE_UNDER_LOAD"
                            and (_dc.get("checks") or {}).get("window_ran_full") is True
                            and (_dc.get("checks") or {}).get("ctrl_exact_rc2_diff_empty") is True
                            and all((_dc.get("checks") or {}).get(k) is True for k in _N2_CHECKS)
                            and (_dc.get("logprobs") or {}).get("requested") == 5
                            and ((((_dc.get("streams") or [{}])[0].get("loop") or {}).get("onset") or {}).get("period")) == 1
                            and len(((_dc.get("streams") or [{}])[0].get("loop") or {}).get("around_onset") or []) == 64
                            and ((_dc.get("streams") or [{}])[0].get("loop") or {}).get("label") == "MODEL_REPETITION_OBSERVED"
                            and ((_dc.get("interpretation") or {}).get("rows_consistent_with_this_result")) == [1, 3]
                            and _dc.get("admission_16") == "SKIPPED_SEQS_CAP"},
        recorded={"units": (R7.get("units") or {}).get("summary"), "e2e": (R7.get("e2e") or {}).get("summary"),
                  "smoke": ((fl.get("smoke") or {}).get("smoke") or {}).get("checks"),
                  "diag_seed2": {k: _d2.get(k) for k in ("verdict", "loop_count", "looped_streams", "checks", "prompt_check")},
                  "diag_ctrl": {k: _dc.get(k) for k in ("verdict", "loop_count", "looped_streams", "checks", "prompt_check")}})
'''

KITS = {
    "real": {"body": K1, "title": "Q27 E25 NB2 Kit Real", "models": RC2_MODELS},
    "units": {"body": K2, "title": "Q27 E25 NB2 Kit Units", "models": [RC2_MODELS[0]]},
}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    subs = (("__EXPORT__", json.dumps(EXPORT)), ("__MAKEFAKE__", json.dumps(MAKE_FAKE)), ("__PARTS__", json.dumps(PARTS)),
            ("__FILESHAS__", json.dumps(json.dumps(FILE_SHAS))), ("__NFILES__", str(len(UNITS_FILES))), ("__NBSHA__", NB_SHA),
            ("__NAME__", name), ("__NBNAME__", NB.name), ("__NBSMOKENAME__", NB_SMOKE.name), ("__SMOKESHA__", SMOKE_SHA),
            ("__SMOKEEDITS__", json.dumps(json.dumps(SMOKE_EDITS))), ("__EXPORTEDIT__", json.dumps(json.dumps(EXPORT_EDIT))),
            ("__EXPORTSHA__", EXPORT_SHA))
    if name == "units":
        subs = (("__BLOB__", blob(UNITS_FILES)),) + subs
    for k, v in subs:
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e25-nb2-kit-{name}"
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
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB2", NB_SHA[:8], flush=True)
    return out, code


if __name__ == "__main__":
    built = {n: build(n) for n in KITS}
    if "--dry" in sys.argv:
        # local dry run: each kit's code on the local fake input tree (real-mount checks fail-soft here: no real Merlin
        # payload / checkpoint locally, so K1 is expected NOT OK on those items; K2 must be ALL OK)
        base = Path(os.environ.get("KIT_LOCAL_ROOT", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/nb2_kit_dry"))
        inp = os.environ.get("KIT_LOCAL_INPUT", r"D:/ARC AGI 2/_scratch/e13_rc2_local/input")
        only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
        for n, (out, code) in built.items():
            if only and n not in only:
                continue
            d = base / n
            d.mkdir(parents=True, exist_ok=True)
            (d / "kit.py").write_text(code, encoding="utf-8")
            env = {**os.environ, "KIT_INPUT": inp, "KIT_TMP": str(d / "tmp"), "KIT_WORK": str(d / "work"), "PYTHONIOENCODING": "utf-8",
                   "KIT_DRY_SIM": str(Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src"))}
            r = subprocess.run([sys.executable, str(d / "kit.py")], capture_output=True, text=True, env=env, timeout=7200, encoding="utf-8", errors="replace")
            (d / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            line = next((l for l in r.stdout.splitlines() if l.startswith("@@KIT")), None)
            print(f"DRY {n}: rc {r.returncode} | {line}", flush=True)
