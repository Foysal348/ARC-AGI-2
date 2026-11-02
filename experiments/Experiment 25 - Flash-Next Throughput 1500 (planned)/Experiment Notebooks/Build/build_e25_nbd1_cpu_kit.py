"""E25 NB-D1 Kaggle CPU REAL kit (pinned Py3.12 image, no GPU, no internet, private; account foysalemonshanto: the klee100 +
drafter Models and the tf-ref dataset are foysal's). The units kit is build_e25_nbd1_units_kit.py.

  real   q27-e25-nbd1-kit-real: NB-S2's real kit (build_e25_nbs2_cpu_kit.py: the NBDF-K100 real kit template of
         build_e27_nbdf_cpu_kit.py, k100 variant, read-only, taken from that module's namespace) RETARGETED to the NB-D1 notebook's
         export (every NB-S2 kit edit copied here, NB-S2's files are not edited), with marked "E25 NB-D1" edits:
           - the notebook's own pre-launch functions in nbd1_main's order: merge (CELL 3) -> merlin_install -> apply_60490 ->
             k100_setup (scales sidecar) -> sha_gate -> PSEQ.before_arm(B) -> SSEQ.apply -> per arm SSEQ.check + PSEQ.before_arm
             -> arm_boot's first try (tree_check, NB1_EXTRA / ENV / KV at the K100 pin, ARM_OF) -> merlin_fn_server (K6, Merlin
             build + host_phase, NBDF launch_check, the --help flag gate with VLLM_TARGET_DEVICE=cpu) for ALL 11 arms;
           - the spawned-child env over the arm's whole expect (QSA / GDN / EMBED_HOST keys included: NB-S1's kit edit), the
             launch env's NB-D1 keys per arm (QSA 128 / 64, Q256 256 / 256, the GDN switch, EMBED_HOST) -> nbd1_qsa_env_final_ok;
           - the argv fields (seqs per arm), R10 spec preflight (D4s K4, DYN's schedule field), R12 drafter load;
           - R9 get_kv_cache_configs: Q 480 calibration, B = K3_swblock1024, D4s / DYN = K4_swblock1024, B_idxfp8 (IDX's pool),
             max concurrency at 21 / 24 / 28 / 32 sequences (S28 / S32) -> nbd1_kv_probes_ok;
           - NBD1PROBE in the Merlin venv: every arm's FINAL argv through vLLM's own serve parser + EngineArgs (one change vs B at
             the argparser level; env-only arms Q256 / G1 / G2 parse identical to B), Merlin's build_dynamic_sd_schedule_lookup ==
             nbd1_lib.dense_schedule, the V2 FULL-graph rule, AttentionConfig fp8 resolution, the shared-prefix checkpoint field,
             the long-prefill fields; Merlin's launch.py QSA 256 lines + the env readers (recorded);
           - the captures per arm (D4s multiples of 5, DYN per tier, S28 / S32 up to 112 / 128) + the Q-defer pack views up to 32;
           - the NB-D1 plan check on the notebook's ARMS, the switch default, the step-coverage table NBD1_STEPS.
         Prints `@@KIT real ALL OK@@` or `@@KIT real NOT OK: <items>@@`; the report nbd1_kit_real_report.json is the input of
         `python build_e25_nbd1.py --kv-gate-report <it>` (the gated rebuild).
Run: python build_e25_nbd1_cpu_kit.py [--build-dir DIR] [--out-root DIR]
     -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbd1-kit-real__foysalemonshanto/
"""
import base64
import hashlib
import json
import lzma
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
MERLIN_SRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
BDIR = Path(sys.argv[sys.argv.index("--build-dir") + 1]) if "--build-dir" in sys.argv else HERE
NBDIR = BDIR if "--build-dir" in sys.argv else HERE.parent
ROOT = Path(sys.argv[sys.argv.index("--out-root") + 1]) if "--out-root" in sys.argv else Path(r"D:/ARC AGI 2/_scratch/push_e25")
OWNER = "foysalemonshanto"
NB_PATH = NBDIR / "Q27-XHIGH-E25_NBD1.ipynb"
FACTS = json.loads((BDIR / "nbd1_build_facts.json").read_text(encoding="utf-8"))
NB_SHA = hashlib.sha256(NB_PATH.read_bytes()).hexdigest()
assert FACTS["sha256"] == NB_SHA, "nbd1 build facts are not for this notebook: rebuild"
assert FACTS["gpu_ready"] is False and FACTS["closed_build_sha256"] == NB_SHA, "the real kit runs on the CLOSED build (its sha gates the rebuild)"
EXPORT = (BDIR / "nbd1_code_export.py").read_text(encoding="utf-8")
assert hashlib.sha256(EXPORT.encode("utf-8")).hexdigest() == FACTS["export_sha256"]
BODY = (BDIR / "nbd1_cell7_body.py").read_text(encoding="utf-8")
LIBS = (HERE / "nbd1_lib.py").read_text(encoding="utf-8")
assert hashlib.sha256(BODY.encode()).hexdigest() == FACTS["body_file_sha256"] and hashlib.sha256(LIBS.encode()).hexdigest() == FACTS["nbd1_lib_sha256"]
ARMS11 = list(FACTS["d1_cfg"]["arms"])
sys.path.insert(0, str(HERE))
import nbd1_lib as D1  # noqa: E402
assert ARMS11 == list(D1.ARM_KEYS)


def blob(files):
    return base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


def sub(text, old, new, where):
    n = text.count(old)
    assert n == 1, f"NB-D1 kit anchor {where}: {n} matches"
    return text.replace(old, new)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


FROZEN = sorted(p for p in E27B.glob("*") if p.is_file()) + [HERE / "build_e25_nbs2_cpu_kit.py", HERE / "build_e25_nbs1_cpu_kit.py"]
FROZEN_SHA = {str(p): sha(p.read_bytes()) for p in FROZEN}

# ---------------------------------------------------------------- the NBDF-K100 kit templates (read-only module namespace)
os.environ["NBDF_VARIANT"] = "k100"
sys.dont_write_bytecode = True
_argv = sys.argv[:]
sys.argv = [str(E27B / "build_e27_nbdf_cpu_kit.py")]
KG = runpy.run_path(str(E27B / "build_e27_nbdf_cpu_kit.py"), run_name="nbdfkit_for_nbd1")
sys.argv = _argv
HEAD, REAL0 = KG["HEAD"], KG["REAL"]
FIXTURE_BLOB, IMAGE = KG["FIXTURE_BLOB"], KG["IMAGE"]
SRC_FILES = tuple(KG["SRC_FILES"]) + ("vllm/v1/spec_decode/dynamic/utils.py", "vllm/config/attention.py", "vllm/config/scheduler.py",
                                      "vllm/config/cache.py", "vllm/v1/core/kv_cache_manager.py", "vllm/config/vllm.py")
SRC_SHAS = {rel: hashlib.sha256((MERLIN_SRC / rel).read_bytes()).hexdigest() for rel in SRC_FILES}

HEAD = HEAD.replace("# E27 NBDF CPU kit __NAME__", "# E25 NB-D1 CPU kit __NAME__ (from the E27 NBDF-K100 kit via E25 NB-S2's)").replace(
    'REP = {"what": f"E27 NBDF CPU kit {KIT}"', 'REP = {"what": f"E25 NB-D1 CPU kit {KIT}"').replace(
    '(W / f"nbdf_kit_{KIT}_report.json")', '(W / f"nbd1_kit_{KIT}_report.json")')
assert "E25 NB-D1 CPU kit" in HEAD and "nbd1_kit_{KIT}_report.json" in HEAD

# ---------------------------------------------------------------- REAL: NB-S2's edits (copied) + the marked NB-D1 edits
R = REAL0
R = sub(R, '''FILES = unpack("__EXPORTBLOB__")
''', '''FILES = unpack("__EXPORTBLOB__")
# ---- E25 NB-D1: the payload dataset (CELL 7 body + nbd1_lib, sha-pinned): the mount first; an embedded copy only when absent
PAY = unpack("__PAYBLOB__")
PAYLOAD = {"mount": None, "embedded_used": False}
for _d in (Path(INPUT_ROOT) / "datasets/foysalemonshanto/q27-e25-nbd1-payload", Path(INPUT_ROOT) / "q27-e25-nbd1-payload"):
    if all((_d / n).is_file() and hashlib.sha256((_d / n).read_bytes()).hexdigest() == h for n, h in PAY["shas"].items()):
        PAYLOAD["mount"] = str(_d)
        break
if PAYLOAD["mount"] is None:
    (K / "payload").mkdir(exist_ok=True)
    for n, txt in PAY["files"].items():
        (K / "payload" / n).write_text(txt, encoding="utf-8", newline="\\n")
    os.environ["NBD1_PAYLOAD_DIR"] = str(K / "payload")
    PAYLOAD["embedded_used"] = True
REP["checks"]["NBD1_PAYLOAD"] = PAYLOAD
''', "payload")
R = sub(R, '''    _okp, _whyp = G["PSEQ"].before_arm(G["ARMS"]["Q"])          # E27 K100: nbdf_main's own call (PatchSeq -> _ppack_apply)''',
        '''    _okp, _whyp = G["PSEQ"].before_arm(G["ARMS"]["B"])          # E25 NB-D1: nbd1_main's own call (PatchSeq -> _ppack_apply)''', "pseq B")
R = sub(R, '''if lv(x, "--max-num-seqs") != "24": bad.append("seqs")''',
        '''if lv(x, "--max-num-seqs") != str(a["expect_cfg"]["max_num_seqs"]): bad.append("seqs")   # E25 NB-D1: per arm''', "R5 seqs")
R = sub(R, '''_xa = next((plans[k_].get("argv") for k_ in ("D3s", "D4s") if (plans.get(k_) or {}).get("argv")), None) or []''',
        '''_xa = next((plans[k_].get("argv") for k_ in ("B",) if (plans.get(k_) or {}).get("argv")), None) or []   # E25 NB-D1''', "kx argv")
R = sub(R, '''K100_KX = {"argv_from": next((k_ for k_ in ("D3s", "D4s") if (plans.get(k_) or {}).get("argv")), None), "seqs": 24,''',
        '''K100_KX = {"argv_from": next((k_ for k_ in ("B",) if (plans.get(k_) or {}).get("argv")), None), "seqs": 24,''', "kx from")
R = sub(R, '''            for s_ in (21, 24):''', '''            for s_ in (21, 24, 28, 32):          # E25 NB-D1: S28 / S32''', "mc by seqs")
R = sub(R, '''                        "q_pages": out["runs"]["Q"].get("pages"), "q_idxfp8_pages": qi.get("pages"), "error": qi.get("error")}
''', '''                        "q_pages": out["runs"]["Q"].get("pages"), "q_idxfp8_pages": qi.get("pages"), "error": qi.get("error")}
        if out.get("q_calibrated"):            # E25 NB-D1 (NB-S2): IDX on B (the D3s SW-1024 layout, MTP layer dropped)
            run("B_idxfp8", worker_specs(ranks_i, 3, dict(bf16, block_size=1024), drop_mtp=True), D_PIN, "FLASH_ATTN")
            KO["r9_idx_b"] = {"b_blocks": (out["runs"].get("K3_swblock1024") or {}).get("num_blocks"),
                              "b_idxfp8_blocks": out["runs"]["B_idxfp8"].get("num_blocks"), "error": out["runs"]["B_idxfp8"].get("error")}
''', "R9 B idx")
R = sub(R, '''         "q_i8_draft_head_build_env": (_pl.get("Q") or {}).get("build_env") == ({"DRAFT_HEAD": "int8"} if (_pl.get("Q") or {}).get("q_i8", {}).get("on") else {}),''',
        '''         "no_draft_head_build_env": all(not (_pl.get(k) or {}).get("build_env") for k in ARMS5),   # E25 NB-D1: no I8 on DFlash2''', "steps i8")
# NB-S1's kit edits: the spawned child reads the arm's WHOLE expect; every plan records the arm's NB-D1 launch-env keys
R = sub(R, 'sp = venv_py(SPAWN_ENV, [",".join(G["N"].WATCHED)], name=',
        'sp = venv_py(SPAWN_ENV, [",".join(list(a["expect"]))], name=', "spawn keys = the arm's expect")
R = sub(R, 'qsa_scales_env=s.env.get("VLLM_QSA_KV_SCALES"),',
        'qsa_scales_env=s.env.get("VLLM_QSA_KV_SCALES"), d1_env={k_: s.env.get(k_) for k_ in tuple(G["D1"].D1_ENV_KEYS) + ("DFLASH_DRAFT_EMBED_HOST", "DFLASH_DRAFT_SW_BLOCK")},',
        "d1 env")
R = sub(R, '''("K100_CAPTURES", k100_captures, "k100_captures_argv_packviews_seqs24_ok")):''',
        '''("K100_CAPTURES", nbd1_captures, "nbd1_captures_argv_packviews_ok")):   # E25 NB-D1: per arm (D4s K4, DYN per tier, S28 / S32), 32 requests''',
        "captures item")
R = sub(R, '''for _key, _fn, _vk in (("K100_SETUP_CHK",''', r'''def nbd1_captures():
    """E25 NB-D1 (replaces item 9): every arm's captures = the plan (multiples of 1 + K up to seqs x (1 + K); D4s K4 -> 5 .. 120;
    DYN: b x (1 + K(b)) per tier; S28 / S32 -> 4 .. 112 / 128), every decode batch an exact FULL graph at its own K (nbd1_lib /
    nbdf_lib coverage), the argv's captures / seqs / KV bytes / spec, cudagraph_mode FULL_AND_PIECEWISE; the Q-defer pack views
    at [n <= 32, K 3 / 4] (nb4_ppack nb4_pack_views)."""
    D1_, lv = G["D1"], G["T"].last_value
    c = {}
    for k in ARMS5:
        a = G["ARMS"].get(k) or {}
        s_ = int(a.get("streams") or 0)
        kk_ = 4 if k in ("D4s", "DYN") else 3
        exp = D1_.dyn_captures(s_) if k == "DYN" else [(kk_ + 1) * b for b in range(1, s_ + 1)]
        cov = D1_.dyn_coverage(a.get("captures") or [], s_) if k == "DYN" else G["N"].coverage(a.get("captures") or [], s_, kk_)
        x = (plans.get(k) or {}).get("argv") or []
        try:
            cm = json.loads(lv(x, "--compilation-config") or "{}").get("cudagraph_mode")
        except Exception:
            cm = None
        bad = []
        if a.get("captures") != exp: bad.append(f"arm captures {a.get('captures')}")
        if G["N"].captures_from_argv(x) != exp: bad.append(f"argv captures {G['N'].captures_from_argv(x)}")
        if lv(x, "--max-num-seqs") != str(s_): bad.append(f"seqs {lv(x, '--max-num-seqs')}")
        if s_ != {"S28": 28, "S32": 32}.get(k, 24): bad.append(f"streams {s_}")
        if lv(x, "--kv-cache-memory-bytes") != str(K100_PIN_B) or a.get("kv_bytes") != K100_PIN_B: bad.append("kv bytes")
        if G["T"].spec_from_argv(x) != a.get("spec"): bad.append(f"spec {G['T'].spec_from_argv(x)} != {a.get('spec')}")
        sp = a.get("spec") or {}
        if sp.get("method") != "dflash" or sp.get("num_speculative_tokens") != kk_: bad.append("spec method / K")
        if cm not in ("FULL_AND_PIECEWISE", "FULL", "FULL_DECODE_ONLY"): bad.append(f"cudagraph_mode {cm}")
        if not cov.get("covers_all") or cov.get("padded_b"): bad.append(f"coverage {cov.get('uncovered_or_padded_b') or cov.get('uncovered_b') or cov.get('padded_b')}")
        c[k] = {"streams": s_, "captures": a.get("captures"), "coverage": cov, "cudagraph_mode": cm, "argv_bad": bad, "ok": not bad}
    import torch
    src = G["P"].PATCHES["vllm/v1/worker/gpu/pp_utils.py"][3]
    seg, keep_ = [], False
    for ln in src.splitlines():
        if ln.startswith("+def nb4_pack_views("):
            keep_ = True
        elif keep_ and (not ln.startswith("+") or ln.startswith(("+def ", "+class ", "+_"))):
            break
        if keep_:
            seg.append(ln[1:] if ln.startswith("+") else ln)
    nsv = {"torch": torch}
    exec("\n".join(seg), nsv)
    pv, badv = nsv["nb4_pack_views"], []
    for kk in (3, 4):
        for n in range(1, 33):
            flat = torch.arange(n * (2 * kk + 3), dtype=torch.int64)
            a_, b_, c_ = pv(flat, n, kk)
            if not (tuple(a_.shape) == (n, kk + 1) and tuple(b_.shape) == (2, n) and tuple(c_.shape) == (n, kk)
                    and a_.is_contiguous() and b_.is_contiguous() and c_.is_contiguous()
                    and a_.numel() + b_.numel() + c_.numel() == flat.numel()
                    and c_.data_ptr() == flat.data_ptr() + 8 * (n * (kk + 1) + 2 * n)
                    and torch.equal(torch.cat([a_.reshape(-1), b_.reshape(-1), c_.reshape(-1)]), flat)):
                badv.append([n, kk])
    c["pack_views"] = {"cases": 64, "bad": badv[:8]}
    c["ok"] = all(c[k]["ok"] for k in ARMS5) and not badv
    return c


for _key, _fn, _vk in (("K100_SETUP_CHK",''', "nbd1_captures def")

NBD1_ITEMS = r'''
# ==================================================================== E25 NB-D1 items (this kit only)
NBD1_OK, NBD1_REC = {}, {}
_D1 = G["D1"]
NBD1PROBE = r"""
import json, sys, inspect, typing, os
out = {"platform": PLAT}
plans = json.loads(open(sys.argv[1]).read())
dyn = json.loads(sys.argv[2])
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    try:
        from vllm.entrypoints.launchers.cli_args import make_arg_parser
    except ImportError:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.engine.arg_utils import EngineArgs
    res = {}
    for k, av in plans.items():
        r = {}
        try:
            args = av[av.index("serve") + 1:]
            ns = make_arg_parser(FlexibleArgumentParser()).parse_args(args)
            if getattr(ns, "model_tag", None) is not None:      # = vllm serve (cli/serve.py ServeSubcommand.cmd): the positional wins
                ns.model = ns.model_tag
            r.update(model=str(ns.model), model_path_exists=os.path.exists(str(ns.model)), hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"))
            ea = EngineArgs.from_cli_args(ns)
            g = lambda o, f: (o.get(f) if isinstance(o, dict) else getattr(o, f, None))
            ac, cc = ea.attention_config, ea.compilation_config
            r.update(indexer_kv_dtype=str(g(ac, "indexer_kv_dtype")), max_num_seqs=ea.max_num_seqs, max_num_batched_tokens=ea.max_num_batched_tokens,
                     long_prefill_token_threshold=ea.long_prefill_token_threshold,
                     long_prefill_token_threshold_adaptive=ea.long_prefill_token_threshold_adaptive,
                     enable_mamba_shared_prefix_checkpoint=ea.enable_mamba_shared_prefix_checkpoint, tp=ea.tensor_parallel_size,
                     pp=ea.pipeline_parallel_size, kv_cache_memory_bytes=ea.kv_cache_memory_bytes, enable_prefix_caching=ea.enable_prefix_caching,
                     spec=ea.speculative_config if isinstance(ea.speculative_config, dict) else str(ea.speculative_config),
                     captures=g(cc, "cudagraph_capture_sizes"), cudagraph_mode=str(g(cc, "cudagraph_mode")), mamba_cache_mode=ea.mamba_cache_mode)
        except BaseException as e:
            r["error"] = f"{type(e).__name__}: {e}"[:500]
        res[k] = r
    out["argparse"] = res
except BaseException as e:
    out["argparse_error"] = f"{type(e).__name__}: {e}"[:500]
try:
    from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
    out["dyn_dense_merlin"] = list(build_dynamic_sd_schedule_lookup(dyn["schedule"], dyn["seqs"], dyn["k"]))
    from vllm.config import vllm as VCm
    src = inspect.getsource(VCm.VllmConfig._maybe_override_dynamic_sd_cudagraph_mode)
    out["dyn_v2_keeps_full"] = "use_v2_model_runner" in src and "PIECEWISE" in src
    from vllm.config import speculative as S
    out["spec_field_per_batch_size"] = "num_speculative_tokens_per_batch_size" in typing.get_type_hints(S.SpeculativeConfig)
    out["use_eagle_includes_dflash"] = '"dflash"' in inspect.getsource(S.SpeculativeConfig.use_eagle)
except BaseException as e:
    out["dyn_error"] = f"{type(e).__name__}: {e}"[:500]
try:
    from vllm.config.attention import AttentionConfig
    out["idx_resolve_fp8"] = AttentionConfig(indexer_kv_dtype="fp8").resolve_indexer_kv_dtype("bf16")
    import vllm
    iq = os.path.join(os.path.dirname(vllm.__file__), "models/qwen4_exp/nvidia/indexer_qsa.py")
    t = open(iq).read()
    out["idx_qsa_accepts_fp8"] = 'self.indexer_kv_dtype == "fp8"' in t and "only 'bf16' or 'fp8'" in t
except BaseException as e:
    out["idx_error"] = f"{type(e).__name__}: {e}"[:500]
try:
    from vllm.config.cache import CacheConfig
    import dataclasses
    out["spc_field"] = "enable_mamba_shared_prefix_checkpoint" in {f.name for f in dataclasses.fields(CacheConfig)}
    import vllm
    km = open(os.path.join(os.path.dirname(vllm.__file__), "v1/core/kv_cache_manager.py")).read()
    out["spc_condition"] = ("enable_mamba_shared_prefix_checkpoint" in km and "self.coordinator.eagle_group_ids" in km
                            and "num_reprefillable_tokens == 0" in km)
    from vllm.config.scheduler import SchedulerConfig
    out["lp_fields"] = all(f in {x.name for x in dataclasses.fields(SchedulerConfig)} for f in ("long_prefill_token_threshold", "long_prefill_token_threshold_adaptive"))
except BaseException as e:
    out["spc_error"] = f"{type(e).__name__}: {e}"[:500]
try:                                         # the QSA workspace readers (recorded): vLLM / Merlin envs read the two names
    import vllm.envs as VE
    import vllm.merlin.envs as ME
    out["qsa_env_readers"] = {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" in inspect.getsource(VE),
                              "VLLM_E2_QSA_LOGITS_GRANULE_MB": "VLLM_E2_QSA_LOGITS_GRANULE_MB" in inspect.getsource(ME)}
except BaseException as e:
    out["qsa_env_error"] = f"{type(e).__name__}: {e}"[:500]
print("@JSON " + json.dumps(out, default=str))
"""
(K / "nbd1_plans.json").write_text(json.dumps({k: (plans.get(k) or {}).get("argv") or [] for k in ARMS5}))
_dyn = {"schedule": _D1.DYN_SCHEDULE, "seqs": 24, "k": _D1.K4}
NP = venv_py(NBD1PROBE, [str(K / "nbd1_plans.json"), json.dumps(_dyn)], name="nbd1_probe", timeout=1800, platform=True)
REP["checks"]["NBD1_PROBE"] = NP
npj = NP.get("json") or {}
ap = npj.get("argparse") or {}
CHANGE = {"Q256": set(), "G1": set(), "G2": set(), "D4s": {"spec", "captures"}, "DYN": {"spec", "captures"}, "IDX": {"indexer_kv_dtype"},
          "LP": {"max_num_batched_tokens", "long_prefill_token_threshold", "long_prefill_token_threshold_adaptive"},
          "S28": {"max_num_seqs", "captures"}, "S32": {"max_num_seqs", "captures"}, "SPC": {"enable_mamba_shared_prefix_checkpoint"}}
dd = _D1.dense_schedule(_dyn["schedule"], 24, 4)
ONE = {}
_b = ap.get("B") or {}
for k in ARMS5:
    r = ap.get(k) or {}
    a = G["ARMS"][k]
    if k == "B":
        ONE[k] = {"error": r.get("error"), "ok": not r.get("error") and r.get("spec") == a["spec"] and r.get("captures") == a["captures"]
                  and r.get("max_num_seqs") == 24 and r.get("tp") == 2 and r.get("pp") == 2 and r.get("kv_cache_memory_bytes") == K100_PIN_B
                  and r.get("enable_prefix_caching") is True and r.get("mamba_cache_mode") == "align"
                  and r.get("indexer_kv_dtype") != "fp8" and r.get("enable_mamba_shared_prefix_checkpoint") is not True, "parsed": r}
        continue
    diff = sorted(f for f in set(r) | set(_b) if f not in ("error",) and r.get(f) != _b.get(f))
    sp_ = r.get("spec") if isinstance(r.get("spec"), dict) else {}
    want = {"Q256": True, "G1": True, "G2": True,
            "D4s": sp_.get("num_speculative_tokens") == 4 and not sp_.get("num_speculative_tokens_per_batch_size"),
            "DYN": sp_.get("num_speculative_tokens") == 4 and sp_.get("num_speculative_tokens_per_batch_size") == [[1, 8, 4], [9, 24, 3]]
                   and npj.get("dyn_dense_merlin") == dd and npj.get("dyn_v2_keeps_full") is True,
            "IDX": r.get("indexer_kv_dtype") == "fp8", "S28": r.get("max_num_seqs") == 28, "S32": r.get("max_num_seqs") == 32,
            "LP": r.get("max_num_batched_tokens") == 2048 and r.get("long_prefill_token_threshold") == 1024 and r.get("long_prefill_token_threshold_adaptive") is True,
            "SPC": r.get("enable_mamba_shared_prefix_checkpoint") is True}[k]
    ONE[k] = {"diff_vs_B": diff, "expected_diff": sorted(CHANGE[k]), "value_ok": want, "error": r.get("error"),
              "spec_equals_plan": r.get("spec") == a["spec"], "captures_equal_plan": r.get("captures") == a["captures"]}
    ONE[k]["ok"] = not r.get("error") and set(diff) == CHANGE[k] and bool(want) and ONE[k]["spec_equals_plan"] and ONE[k]["captures_equal_plan"]
REP["checks"]["NBD1_ONE_CHANGE_ARGPARSE"] = ONE
NBD1_OK["nbd1_argparse_one_change_per_arm"] = len(ONE) == len(ARMS5) and all(v["ok"] for v in ONE.values())
NBD1_OK["nbd1_idx_fp8_resolves"] = npj.get("idx_resolve_fp8") == "fp8" and npj.get("idx_qsa_accepts_fp8") is True
NBD1_OK["nbd1_spc_lp_fields"] = npj.get("spc_field") is True and npj.get("spc_condition") is True and npj.get("lp_fields") is True
NBD1_REC["dyn_schedule_merlin"] = {"equals_lib": npj.get("dyn_dense_merlin") == dd, "v2_keeps_full": npj.get("dyn_v2_keeps_full"),
                                   "dyn_source_check": _D1.DYN_SOURCE_CHECK}
NBD1_REC["use_eagle_includes_dflash"] = npj.get("use_eagle_includes_dflash")
NBD1_REC["qsa_env_readers"] = npj.get("qsa_env_readers") or npj.get("qsa_env_error")
# the KV probes (Merlin get_kv_cache_configs at the pin): B / Q256 / G1 / G2 / LP / SPC = K3_swblock1024, D4s / DYN = K4_swblock1024,
# IDX = B_idxfp8, S28 / S32 = the max concurrency at 28 / 32 sequences (EMBED_HOST changes no KV spec at the same pin)
_rb = r9k.get("r9_idx_b") or {}
_mc3 = ((r9k.get("sim") or {}).get("K3_swblock1024") or {}).get("mc_by_seqs") or {}
_mc4 = ((r9k.get("sim") or {}).get("K4_swblock1024") or {}).get("mc_by_seqs") or {}
_k3, _k4 = _minb9("K3_swblock1024"), _minb9("K4_swblock1024")
_kv = {"B_K3_swblock1024": _k3, "D4s_DYN_K4_swblock1024": _k4, "IDX_B_idxfp8": _rb.get("b_idxfp8_blocks"), "idx_error": _rb.get("error"),
       "mc_by_seqs_K3": _mc3, "mc_by_seqs_K4": _mc4, "q_480": _minb9("Q")}
NBD1_REC["kv_pool"] = _kv
NBD1_OK["nbd1_kv_probes_ok"] = (_isint(_k3) and _k3 > 0 and _isint(_k4) and _k4 > 0
                                and _isint(min(_rb.get("b_idxfp8_blocks") or [0])) and min(_rb.get("b_idxfp8_blocks") or [0]) > 0 and not _rb.get("error")
                                and all(isinstance(_mc3.get(str(s_)) or _mc3.get(s_), (int, float)) for s_ in (24, 28, 32))
                                and all(isinstance(_mc4.get(str(s_)) or _mc4.get(s_), (int, float)) for s_ in (24,)))
# the QSA / GDN / EMBED_HOST keys in every arm's FINAL launch env (Merlin build() sets QSA 256 / 256; NB1_ENV after it wins)
_pl1 = R4.get("plans") or {}
_qe = {}
for k in ARMS5:
    a_ = G["ARMS"][k]
    e_ = (_pl1.get(k) or {}).get("d1_env") or {}
    want_ = {kk: a_["env"].get(kk) for kk in tuple(_D1.D1_ENV_KEYS) + ("DFLASH_DRAFT_EMBED_HOST", "DFLASH_DRAFT_SW_BLOCK")}
    _qe[k] = {"launch_env": e_, "planned": want_, "ok": bool(e_) and e_ == want_, "spawn_ok": (_pl1.get(k) or {}).get("spawn_ok")}
NBD1_REC["qsa_env_final"] = _qe
NBD1_OK["nbd1_qsa_env_final_ok"] = (len(_qe) == len(ARMS5) and all(v["ok"] and v["spawn_ok"] is True for v in _qe.values())
                                    and _qe["B"]["launch_env"].get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB") == "128"
                                    and _qe["B"]["launch_env"].get("VLLM_E2_QSA_LOGITS_GRANULE_MB") == "64"
                                    and _qe["Q256"]["launch_env"].get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB") == "256"
                                    and _qe["Q256"]["launch_env"].get("VLLM_E2_QSA_LOGITS_GRANULE_MB") == "256")
try:                                         # Merlin's own 256 / 256 in the installed launcher (recorded: the override is what makes 128 / 64)
    _lt = (Path(eng["kv"]["MERLIN_TREE"]) / "tools/merlin/launch.py").read_text()
    NBD1_REC["merlin_launch_qsa_256"] = ('env["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "256"' in _lt, 'env["VLLM_E2_QSA_LOGITS_GRANULE_MB"] = "256"' in _lt)
except Exception as e:
    NBD1_REC["merlin_launch_qsa_256"] = f"{type(e).__name__}: {e}"[:200]
# the NB-D1 plan on the notebook's own ARMS + the switch default + the launch path of all 11 arms
try:
    _D1.check_arm_plan(G["N"], G["T"], G["ARMS"], G["NBDF_ARMS"]["D3s"])
    _ca = True
except Exception as e:
    _ca = f"{type(e).__name__}: {e}"[:300]
NBD1_REC["check_arm_plan"] = _ca
NBD1_OK["nbd1_arm_plan_ok"] = _ca is True and tuple(G["ARMS"]) == tuple(_D1.ARM_KEYS) == tuple(ARMS5)
NBD1_REC["switch"] = (G.get("NBD1") or {}).get("switch")
NBD1_OK["nbd1_switch_default_all_on"] = (G.get("NBD1_SWITCH") == {k: True for k in _D1.SWITCH_KEYS}
                                         and ((G.get("NBD1") or {}).get("switch") or {}).get("absent") is False)
NBD1_OK["nbd1_help_flag_gate_real_ok"] = K100_OK.get("k100_help_flag_gate_real_ok") is True
_per = {}
for k in ARMS5:
    p_ = _pl1.get(k) or {}
    _per[k] = {"sseq_check + pseq_before_arm (nbd1_main loop)": p_.get("sseq_check", [None])[0] is True and p_.get("pseq_before_arm", [None])[0] is True,
               "tree_check": bool((p_.get("tree_check") or {}).get("ok")),
               "arm_boot knobs (NB1_EXTRA / NB1_ENV / NB1_KV at q_pin, ARM_OF, no DRAFT_HEAD)": p_.get("pin") == G["K100_CFG"]["q_pin"] and not p_.get("build_env"),
               "K6 before launch": (p_.get("k6_before_launch") or {}).get("ok") is True,
               "Merlin build + host_phase": p_.get("host_phase") == "ok",
               "--help flag gate (L.help_check + KL.cli_flags_check)": ((p_.get("cli_flags") or {}).get("ok") is True),
               "NBDF launch check + nbdf_lib.launch_check at the pin": p_.get("ok") is True and p_.get("launch_check") == [],
               "VLLM_QSA_KV_SCALES": str(p_.get("qsa_scales_env") or "").endswith("qsa_kv_scales_262k.json"),
               "env in a spawned child (the arm's expect)": p_.get("spawn_ok") is True,
               "the NB-D1 launch-env keys as planned": _qe[k]["ok"],
               "argv spec / captures / seqs / KV bytes (R5 + captures item)": not (R5.get("field_check") or {}).get(k) and ((REP["checks"].get("K100_CAPTURES") or {}).get(k) or {}).get("ok") is True,
               "vLLM argparse: B + exactly one change": (ONE.get(k) or {}).get("ok") is True}
NBD1_STEPS = {"per_arm": _per, "table": REP["checks"].get("NBD1_STEP_TABLE"),
              "gpu_only_not_covered": ["boot_gate (GPU empty)", "LoadMonitor / with_servers / readiness", "the NB-D1 read-back on the "
                                       "server log + worker env RPC (QSA / GDN in 4 workers, EMBED_HOST, indexer dtype, logged flags, DYN "
                                       "fallback lines)", "k100_boot_gates", "smoke / windows / PROF profiles / sibling probe"]}
NBD1_STEPS["all_covered"] = all(all(v.values()) for v in _per.values())
REP["checks"]["NBD1_STEPS"] = NBD1_STEPS
NBD1_OK["nbd1_launch_path_all_arms_ok"] = (len(_pl1) == len(ARMS5) and NBD1_STEPS["all_covered"] and R5.get("ok") is True
                                           and K100_OK.get("k100_notebook_launch_steps_all_ok") is True)
log("NBD1", json.dumps({"ok": NBD1_OK, "rec": {k: v for k, v in NBD1_REC.items() if k != "qsa_env_final"}}, default=str)[:3000])
save()
'''
R = sub(R, '''K100_OK["k100_notebook_launch_steps_all_ok"] = all(v is True for v in STEPS.values())
save()
''', '''K100_OK["k100_notebook_launch_steps_all_ok"] = all(v is True for v in STEPS.values())
save()
''' + NBD1_ITEMS, "nbd1 items")
STEP_TABLE = [
    ["CELL 0 identity submission (E18 R2)", "R0 cell0_identity_submission (the export run with RC2_UNIT + RC2_CPU_KIT)"],
    ["CELL 1-3 inputs, lab, merge (overlay + K6 + digests)", "R1 inputs / merge / sha256 (the notebook's own merge_ckpt + sha_gate)"],
    ["CELL 2b k100q_payload + klee100 model path", "the export's CELL 2b runs as is (tf-ref dataset attached)"],
    ["CELL 7 stub: the payload dataset body + nbd1_lib (sha)", "NBD1_PAYLOAD (mount checked by sha; embedded copy only when absent, recorded)"],
    ["CELL 7 drafter_facts (drafter Model pins)", "R1 drafter_pins_ok"],
    ["CELL 7 D1.arm_plan + D1.check_arm_plan (11 arms)", "nbd1_arm_plan_ok (check_arm_plan on the notebook's ARMS)"],
    ["CELL 1 NBD1_ON (default every key on)", "nbd1_switch_default_all_on"],
    ["nbd1_main: merlin_install", "R2 merlin_install_ok + Merlin source shas"],
    ["nbd1_main: apply_60490", "R3 patch_60490_ok"],
    ["nbd1_main: k100_setup (scales sidecar)", "K100_SETUP / k100_setup_scales_sidecar_ok"],
    ["nbd1_main: sha_gate", "R1 sha256_all_ok"],
    ["nbd1_main: PSEQ.before_arm(B)", "R3P before_arm_q (B)"],
    ["nbd1_main: SSEQ.apply", "R3S seam_apply_verify_revert_ok"],
    ["nbd1_main per arm: SSEQ.check + PSEQ.before_arm", "R4 plans[arm].sseq_check / pseq_before_arm (all 11)"],
    ["arm_boot first try: tree_check, NB1_EXTRA / ENV / KV at 7,933,526,016, ARM_OF, EMBED_HOST + QSA in NB1_ENV", "R4 plan_one = arm_boot"],
    ["merlin_fn_server: K6 + Merlin build + host_phase + launch_check + --help flag gate", "R4 host_phase / launch_check / k6 / cli_flags"],
    ["the spawned engine env (the arm's whole expect: QSA / GDN / EMBED_HOST)", "R4 env_in_spawned_child + nbd1_qsa_env_final_ok"],
    ["argv: seqs / KV bytes / dtype / spec / captures per arm", "R5 argv_fields_ok + nbd1_captures (D4s K4, DYN per tier, S28 / S32)"],
    ["vLLM's own parsing of the final argv (the arm's real config)", "NBD1_ONE_CHANGE_ARGPARSE (serve parser + EngineArgs, diff vs B)"],
    ["IDX: AttentionConfig fp8 + the QSA indexer accepts fp8; IDX pool", "nbd1_idx_fp8_resolves + nbd1_kv_probes_ok (R9 B_idxfp8)"],
    ["LP / SPC config fields + the SPC activation condition", "nbd1_spc_lp_fields"],
    ["the spec config (D4s K4, DYN's schedule field) + Merlin's dense schedule", "R10 spec_config_preflight_ok + NBD1 argparse DYN value"],
    ["the drafter load (draft_load_check)", "R12 drafter_loaded_weights_ok"],
    ["KV pool / F / block drop at the pin (B, D4s / DYN = K4), max concurrency at 28 / 32", "R9 kv_cache_configs_ok + nbd1_kv_probes_ok"],
    ["GPU only: boot, smoke, NB-D1 read-back, windows, PROF, sibling, ledger, LIVE", "NOT ON CPU (GPU run); their pure logic = the units kit"],
    ["finally: seam revert, ppack revert", "R3S pristine / K100_SEAM"]]
R = sub(R, '''REP["checks"]["K100_STEPS"] = STEPS''', '''REP["checks"]["K100_STEPS"] = STEPS
REP["checks"]["NBD1_STEP_TABLE"] = json.loads(r"""__STEPTABLE__""")''', "step table")
R = sub(R, '''"drafter_loaded_weights_ok": (R12.get("json") or {}).get("ok") is True, **K100_OK},''',
        '''"drafter_loaded_weights_ok": (R12.get("json") or {}).get("ok") is True, **K100_OK, **NBD1_OK},''', "verdict keys")
R = sub(R, '''recorded={**K100_REC, "patched_model_import": R11.get("json"),''',
        '''recorded={**K100_REC, **NBD1_REC, "payload": PAYLOAD, "patched_model_import": R11.get("json"),''', "verdict recorded")
REAL = R
for _k in D1.KIT_KEYS:
    assert f'NBD1_OK["{_k}"]' in REAL or f'nbd1_captures, "{_k}")' in REAL, f"kit key {_k} never set"   # the captures key: the K100 item loop

# ---------------------------------------------------------------- build
assert hashlib.sha256((E27B / "nbdf_lib.py").read_bytes()).hexdigest() == FACTS["nbdf_lib_sha256"], "nbdf_lib != the build's"
EXPORT_BLOB = blob({"nbdf_code_export.py": EXPORT})
PAY_BLOB = blob({"files": {"nbd1_cell7_body.py": BODY, "nbd1_lib.py": LIBS}, "shas": dict(FACTS["payload_dataset"]["files"])})
MODELS = list(FACTS["model_sources"])
DATASETS = [d for d in FACTS["dataset_sources"] if d != FACTS["payload_dataset"]["slug"]]   # the payload is embedded (not uploaded yet)
KITS = {"real": {"body": REAL, "title": "Q27 E25 NBD1 Kit Real", "models": MODELS, "datasets": DATASETS}}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__EXPORTBLOB__", EXPORT_BLOB), ("__FIXBLOB__", FIXTURE_BLOB), ("__PAYBLOB__", PAY_BLOB),
                 ("__NBSHA__", NB_SHA), ("__NAME__", name), ("__SRCSHAS__", json.dumps(SRC_SHAS)), ("__KITARMS__", json.dumps(ARMS11)),
                 ("__QBLOCKS__", "480"), ("__QPIN__", "7933526016"), ("__DPIN__", "7933526016"), ("__VARIANT__", "k100"),
                 ("__STEPTABLE__", json.dumps(STEP_TABLE))):
        code = code.replace(k, v)
    assert "__" + "EXPORTBLOB" not in code and "__PAYBLOB__" not in code and "__KITARMS__" not in code and "__STEPTABLE__" not in code
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e25-nbd1-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, separators=(",", ":")).encode("utf-8")   # compact (the size margin under the 400 kB rule)
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__{OWNER}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        if f.is_file():
            f.unlink()
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"{OWNER}/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": spec["datasets"], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB-D1", NB_SHA[:8], flush=True)
    return out, code, data


if __name__ == "__main__":
    for n in KITS:
        build(n)
    for p in FROZEN:
        assert sha(p.read_bytes()) == FROZEN_SHA[str(p)], f"FROZEN FILE CHANGED: {p}"
