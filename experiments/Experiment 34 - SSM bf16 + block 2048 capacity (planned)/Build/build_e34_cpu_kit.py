"""E34 Kaggle CPU REAL kit (pinned Py3.12 image, no GPU, no internet, private; account foysalemonshanto: the klee100 + drafter
Models, the tf-ref dataset and the E34 TF-trace dataset are foysal's). The units kit is build_e34_units_kit.py.

  real   q27-e34-kit-real: NB-D1's real kit (build_e25_nbd1_cpu_kit.py: NB-S2's real kit = the NBDF-K100 real kit template,
         taken from that module's namespace; read-only) RETARGETED to the E34 FULL notebook's export, with marked "E34" edits:
           - the notebook's own pre-launch sequence in nbd1_main's order for ALL 4 arms (merge -> merlin_install -> apply_60490 ->
             k100_setup -> sha_gate -> PSEQ.before_arm(B) -> SSEQ.apply -> per arm SSEQ.check + PSEQ.before_arm -> arm_boot's
             first try (tree_check, NB1_EXTRA / ENV / KV, ARM_OF) -> merlin_fn_server (K6, Merlin build + host_phase, the
             notebook's launch check = E34.launch_check, the --help flag gate with VLLM_TARGET_DEVICE=cpu)), the spawned-child env
             over the arm's whole expect (SW block included);
           - the payload dataset q27-e34-payload (mount first; an embedded copy only when absent, E34_PAYLOAD_DIR);
           - E34ARGS in the Merlin venv: every arm's FINAL argv through vLLM's own serve parser + EngineArgs (B + exactly its
             change: mamba_ssm_cache_dtype / block_size / max_num_seqs / captures) + the installed Merlin source facts of the
             SSM bf16 / block 2048 path + MambaStateDtypeCalculator's GDN dtypes;
           - E34KV in the Merlin venv: Merlin's own get_kv_cache_configs / scheduler config / max concurrency on the GPU worker
             record (NBDF-K100's KVPROBE spec builder) for B, S16, S16B2 (SW 512), S16B2 at SW 1024 and fp32 at 2048 (the
             platform rule: raised), the engine block drop, the scheduler / hash block;
           - the never-do lift (nbdf_lib flags exactly the lifted entry on the SSM arms, the notebook's check passes), the launch
             env SW block per arm, the TF-trace dataset (mounted, sha), the plan check (FULL / SMOKE / every W 24 .. 40).
         Prints `@@KIT real ALL OK@@` or `@@KIT real NOT OK: <items>@@`; the report e34_kit_real_report.json is the input of
         `python -B build_e34.py --kv-gate-report <it>` (the gated rebuild).
Run: python -B build_e34_cpu_kit.py [--build-dir DIR] [--out-root DIR]   -> Build/push/q27-e34-kit-real__foysalemonshanto/ (not pushed)
Prerequisite on Kaggle: the TF-trace dataset foysalemonshanto/q27-e34-tf-long uploaded (it does not depend on the gate; too big to
embed in a < 400 kB kit), else e34_tf_long_ok is False (the payload dataset may be absent: its embedded copy is used).
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
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
BDIR = Path(sys.argv[sys.argv.index("--build-dir") + 1]) if "--build-dir" in sys.argv else HERE
NBDIR = BDIR if "--build-dir" in sys.argv else HERE.parent / "Experiment Notebooks"
ROOT = Path(sys.argv[sys.argv.index("--out-root") + 1]) if "--out-root" in sys.argv else HERE / "push"
OWNER = "foysalemonshanto"
FACTS = json.loads((BDIR / "e34_build_facts.json").read_text(encoding="utf-8"))
NB_PATH = NBDIR / FACTS["notebook"]
NB_SHA = hashlib.sha256(NB_PATH.read_bytes()).hexdigest()
assert FACTS["sha256"] == NB_SHA, "e34 build facts are not for this notebook: rebuild"
assert FACTS["gpu_ready"] is False and FACTS["closed_build_sha256"] == NB_SHA, "the real kit runs on the CLOSED FULL build (its sha gates the rebuild)"
EXPORT = (BDIR / "e34_code_export.py").read_text(encoding="utf-8")
assert hashlib.sha256(EXPORT.encode("utf-8")).hexdigest() == FACTS["export_sha256"]
BODY = (BDIR / "e34_cell7_body.py").read_text(encoding="utf-8")
LIB_E34 = (HERE / "e34_lib.py").read_text(encoding="utf-8")
LIB_D1 = (E25B / "nbd1_lib.py").read_text(encoding="utf-8")
assert hashlib.sha256(BODY.encode()).hexdigest() == FACTS["body_file_sha256"] and hashlib.sha256(LIB_E34.encode()).hexdigest() == FACTS["e34_lib_sha256"]
assert hashlib.sha256(LIB_D1.encode()).hexdigest() == FACTS["nbd1_lib_sha256"]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import e34_lib as E  # noqa: E402
ARMS4 = list(E.ARM_KEYS)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def sub(text, old, new, where):
    n = text.count(old)
    assert n == 1, f"E34 kit anchor {where}: {n} matches"
    return text.replace(old, new)


def blob(files):
    return base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


FROZEN = sorted(p for p in E25B.glob("*") if p.is_file()) + sorted(p for p in E27B.glob("*") if p.is_file())
FROZEN_SHA = {str(p): sha(p.read_bytes()) for p in FROZEN}

# ---------------------------------------------------------------- NB-D1's real kit (read-only module namespace; its argv sanitised)
_argv = sys.argv[:]
sys.argv = [str(E25B / "build_e25_nbd1_cpu_kit.py")]
KN = runpy.run_path(str(E25B / "build_e25_nbd1_cpu_kit.py"), run_name="nbd1kit_for_e34")
sys.argv = _argv
HEAD, REAL0, FIXTURE_BLOB, IMAGE = KN["HEAD"], KN["REAL"], KN["FIXTURE_BLOB"], KN["IMAGE"]
SRC_SHAS = dict(KN["SRC_SHAS"])
NBD1_ITEMS = KN["NBD1_ITEMS"]

HEAD = sub(HEAD, "# E25 NB-D1 CPU kit __NAME__ (from the E27 NBDF-K100 kit via E25 NB-S2's)",
           "# E34 CPU kit __NAME__ (from the E27 NBDF-K100 kit via E25 NB-S2's and E25 NB-D1's)", "head name")
HEAD = sub(HEAD, 'REP = {"what": f"E25 NB-D1 CPU kit {KIT}"', 'REP = {"what": f"E34 CPU kit {KIT}"', "head what")
HEAD = sub(HEAD, '(W / f"nbd1_kit_{KIT}_report.json")', '(W / f"e34_kit_{KIT}_report.json")', "head report")

R = REAL0
R = sub(R, '''# ---- E25 NB-D1: the payload dataset (CELL 7 body + nbd1_lib, sha-pinned): the mount first; an embedded copy only when absent
PAY = unpack("__PAYBLOB__")
PAYLOAD = {"mount": None, "embedded_used": False}
for _d in (Path(INPUT_ROOT) / "datasets/foysalemonshanto/q27-e25-nbd1-payload", Path(INPUT_ROOT) / "q27-e25-nbd1-payload"):''',
        '''# ---- E34: the payload dataset (CELL 7 body + nbd1_lib + e34_lib, sha-pinned): the mount first; an embedded copy only when absent
PAY = unpack("__PAYBLOB__")
PAYLOAD = {"mount": None, "embedded_used": False}
for _d in (Path(INPUT_ROOT) / "datasets/foysalemonshanto/q27-e34-payload", Path(INPUT_ROOT) / "q27-e34-payload"):''', "payload mount")
R = sub(R, '''    os.environ["NBD1_PAYLOAD_DIR"] = str(K / "payload")''', '''    os.environ["E34_PAYLOAD_DIR"] = str(K / "payload")''', "payload env")
R = sub(R, '''        if s_ != {"S28": 28, "S32": 32}.get(k, 24): bad.append(f"streams {s_}")''',
        '''        if s_ != {"S16B2W": G["E34"].W_PRED}.get(k, 24): bad.append(f"streams {s_}")   # E34: S16B2W at the planned W''', "captures streams")
R = sub(R, '''            for s_ in (21, 24, 28, 32):          # E25 NB-D1: S28 / S32''', '''            for s_ in (21, 24, 28, 29, 32):      # E25 NB-D1 / E34: S16B2W's planned W 29''', "mc by seqs")

E34_ITEMS = r'''
# ==================================================================== E34 items (this kit only)
NBD1_OK, NBD1_REC = {}, {}                  # (NB-D1's names: the verdict merges them) E34's keys
_E = G["E34"]
E34ARGS = r"""
import json, sys, inspect, os
out = {"platform": PLAT}
plans = json.loads(open(sys.argv[1]).read())
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
            if getattr(ns, "model_tag", None) is not None:
                ns.model = ns.model_tag
            ea = EngineArgs.from_cli_args(ns)
            g = lambda o, f: (o.get(f) if isinstance(o, dict) else getattr(o, f, None))
            cc = ea.compilation_config
            r.update(mamba_ssm_cache_dtype=str(ea.mamba_ssm_cache_dtype), block_size=ea.block_size, max_num_seqs=ea.max_num_seqs,
                     prefix_match_unit=ea.prefix_match_unit, mamba_cache_mode=ea.mamba_cache_mode, enable_prefix_caching=ea.enable_prefix_caching,
                     kv_cache_memory_bytes=ea.kv_cache_memory_bytes, tp=ea.tensor_parallel_size, pp=ea.pipeline_parallel_size,
                     max_num_batched_tokens=ea.max_num_batched_tokens, kv_cache_dtype=str(ea.kv_cache_dtype),
                     spec=ea.speculative_config if isinstance(ea.speculative_config, dict) else str(ea.speculative_config),
                     captures=g(cc, "cudagraph_capture_sizes"), cudagraph_mode=str(g(cc, "cudagraph_mode")))
        except BaseException as e:
            r["error"] = f"{type(e).__name__}: {e}"[:500]
        res[k] = r
    out["argparse"] = res
except BaseException as e:
    out["argparse_error"] = f"{type(e).__name__}: {e}"[:500]
try:
    import vllm
    root = os.path.dirname(vllm.__file__)
    rd = lambda p: open(os.path.join(root, p)).read()
    out["src"] = {
        "arg_mamba_ssm_cache_dtype": '"--mamba-ssm-cache-dtype"' in rd("engine/arg_utils.py"),
        "mambadtype_bf16": 'MambaDType = Literal["auto", "float32", "float16", "bfloat16"]' in rd("config/cache.py"),
        "recoverssm_bf16_fp32": "RecoverSSM supports bfloat16/float32" in rd("config/vllm.py"),
        "recoverssm_replay_fp32_only": ('replay_cache.dtype == torch.float32' in rd("model_executor/layers/mamba/gdn/recoverssm_gdn.py")
                                        and "checkpoint_state.dtype == torch.float32" not in rd("model_executor/layers/mamba/gdn/recoverssm_gdn.py")),
        "qwen35_hook_explicit_wins": "Using the user-specified value." in rd("model_executor/models/config.py"),
        "platform_block_rule": ("attn_block_size = kernel_block_alignment_size * cdiv(" in rd("platforms/interface.py")
                                and "Padding mamba page size by" in rd("platforms/interface.py")
                                and "cache_config.mamba_block_size = cache_config.block_size" in rd("platforms/interface.py")),
        "qsa_ring_divides_block": "must divide the attention block" in rd("models/qwen4_exp/common/qsa_cache.py"),
        "fa_multiple_of_16": "return [MultipleOf(16)]" in rd("v1/attention/backends/flash_attn.py"),
        "engine_block_min_prefix_groups": "if g.kv_cache_spec.prefix_cacheable" in rd("v1/engine/core.py")}
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as MSC
    import torch
    out["gdn_dtypes"] = {d: [str(x) for x in MSC.gated_delta_net_state_dtype(torch.bfloat16, "auto", d)] for d in ("auto", "float32", "bfloat16")}
except BaseException as e:
    out["src_error"] = f"{type(e).__name__}: {e}"[:500]
print("@JSON " + json.dumps(out, default=str))
"""
_pl1 = R4.get("plans") or {}
(K / "e34_plans.json").write_text(json.dumps({k: (_pl1.get(k) or {}).get("argv") or [] for k in ARMS5}))
EA = venv_py(E34ARGS, [str(K / "e34_plans.json")], name="e34_args", timeout=1800, platform=True)
REP["checks"]["E34_ARGS"] = EA
eaj = EA.get("json") or {}
ap = eaj.get("argparse") or {}
CHANGE = {"S16": {"mamba_ssm_cache_dtype"}, "S16B2": {"mamba_ssm_cache_dtype", "block_size"},
          "S16B2W": {"mamba_ssm_cache_dtype", "block_size", "max_num_seqs", "captures"}}
ONE, _b = {}, ap.get("B") or {}
for k in ARMS5:
    r = ap.get(k) or {}
    a = G["ARMS"][k]
    pl = a["e34"]
    base_ok = (not r.get("error") and r.get("tp") == 2 and r.get("pp") == 2 and r.get("kv_cache_memory_bytes") == K100_PIN_B
               and r.get("enable_prefix_caching") is True and r.get("mamba_cache_mode") == "align" and r.get("prefix_match_unit") == 64
               and r.get("spec") == a["spec"] and r.get("captures") == a["captures"] and r.get("max_num_seqs") == a["streams"])
    val_ok = (r.get("block_size") == pl["block"] and r.get("mamba_ssm_cache_dtype") == ("auto" if pl["ssm"] == "float32" else pl["ssm"]))
    if k == "B":
        ONE[k] = {"ok": base_ok and val_ok, "parsed": r}
        continue
    diff = sorted(f for f in set(r) | set(_b) if f != "error" and r.get(f) != _b.get(f))
    ONE[k] = {"diff_vs_B": diff, "expected_diff": sorted(CHANGE[k]), "ok": base_ok and val_ok and set(diff) == CHANGE[k], "parsed": r}
REP["checks"]["E34_CHANGE_ARGPARSE"] = ONE
NBD1_OK["e34_argparse_change_per_arm"] = len(ONE) == len(ARMS5) and all(v["ok"] for v in ONE.values())
_src = eaj.get("src") or {}
NBD1_REC["merlin_src"] = _src or eaj.get("src_error")
NBD1_REC["gdn_dtypes"] = eaj.get("gdn_dtypes")
NBD1_OK["e34_merlin_source_ok"] = (bool(_src) and all(v is True for v in _src.values())
                                   and (eaj.get("gdn_dtypes") or {}).get("bfloat16") == ["torch.bfloat16", "torch.bfloat16"]
                                   and (eaj.get("gdn_dtypes") or {}).get("float32") == ["torch.bfloat16", "torch.float32"])
# ---- E34KV: Merlin's planner on the GPU worker record (NBDF-K100's KVPROBE spec builder + E34's dtype / block / SW transform)
_kvp = KVPROBE.split('    ranks = fx["q_workers"]["ranks"]\n')[0]
E34KV = _kvp + r"""
    import math, os
    from vllm.v1.core import kv_cache_utils as KU2
    ranks = fx["q_workers"]["ranks"]
    def e34_specs(block, ssm, sw):
        rr = copy.deepcopy(ranks)
        mx = 0
        for w in rr:
            for s in w["kv_cache_specs"]:
                ex = str(s.get("example") or "")
                if s["type"] == "MambaSpec" and "linear_attn" in ex and ssm != "float32":
                    s["state_dtypes"] = [s["state_dtypes"][0], ssm] + list(s["state_dtypes"][2:])
                if s.get("block_size") == 4096 and s["type"] in ("FullAttentionSpec", "MLAAttentionSpec", "MambaSpec"):
                    if s["type"] != "MambaSpec" and isinstance(s.get("page_size_bytes"), int):
                        s["page_size_bytes"] = s["page_size_bytes"] * block // 4096
                    s["block_size"] = block
                if s["type"] == "MambaSpec":
                    import math as _m
                    tdt_ = lambda v: getattr(torch, str(v).replace("torch.", ""))
                    sh = k_shapes(s, 3)
                    mx = max(mx, sum(_m.prod(a_) * tdt_(d_).itemsize for a_, d_ in zip(sh["state_shapes"], sh["state_dtypes"])))
                    s["page_size_padded"] = block * 512          # interface.py: the Mamba page padded to the attention page
                    s["page_size_bytes"] = block * 512
        dr = {"type": "SlidingWindowSpec", "block_size": sw, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.bfloat16", "sliding_window": 2048}
        return worker_specs(rr, 3, dr, drop_mtp=True), mx
    def vc_for(block, seqs=24):
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        vc = VllmConfig()
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        vc.model_config = NS(max_model_len=131072, original_max_model_len=131072, hf_config=NS(model_type="qwen4_exp"), use_mla=False,
                             is_hybrid=True, get_num_layers=lambda *a: 48)
        cc, pc, sc = vc.cache_config, vc.parallel_config, vc.scheduler_config
        cc.block_size, cc.enable_prefix_caching, cc.cache_dtype = block, True, "fp8_e4m3"
        cc.prefix_match_unit, cc.mamba_cache_mode = 64, "align"
        pc.pipeline_parallel_size, pc.tensor_parallel_size = 2, 2
        sc.async_scheduling, sc.max_num_batched_tokens, sc.max_num_seqs = True, 1024, int(seqs)
        return vc
    for tag, block, ssm, sw in (("B", 4096, "float32", 1024), ("S16", 4096, "bfloat16", 1024), ("S16B2", 2048, "bfloat16", 512),
                                ("S16B2_SW1024", 2048, "bfloat16", 1024), ("fp32_block2048", 2048, "float32", 512)):
        rec = {"block": block, "ssm": ssm, "sw": sw}
        try:
            ws, mx = e34_specs(block, ssm, sw)
            rec["max_mamba_state_page"], rec["attn_page"] = mx, block * 512
            if mx > block * 512:
                rec["BLOCK_RAISED"] = True
                out["runs"][tag] = rec
                continue
            vc = vc_for(block)
            bks = list(BK.get("target") or []) + ([BK["FLASH_ATTN"]] if BK.get("FLASH_ATTN") else [])
            sup = [l.name for l in AU.get_supported_kv_cache_layouts(bks)] if bks else ["BLNHC", "BLHNC"]
            AU.resolve_kv_cache_layout(vc, [sup] * len(ws), [s for w in ws for s in w.values()])
            cfgs = KU2.get_kv_cache_configs(vc, ws, [Q_PIN] * len(ws))
            sched = KU2.generate_scheduler_kv_cache_config(cfgs)
            rec["num_blocks"] = [c.num_blocks for c in cfgs]
            rec["stride"] = [KU2._pool_bytes_per_block(vc, c.kv_cache_groups) for c in cfgs]
            rec["group_blocks"] = [g.kv_cache_spec.block_size for g in sched.kv_cache_groups]
            rec["mc_by_seqs"] = {str(s_): KU2.get_max_concurrency_for_kv_cache_config(vc_for(block, s_), sched) for s_ in (24, 29, 32, 40)}
            part = [g.kv_cache_spec.block_size for g in sched.kv_cache_groups if g.kv_cache_spec.prefix_cacheable]
            vc.cache_config.block_size = min(part)
            rec["engine_block"] = vc.cache_config.block_size
            sbs, hbs = KU2.resolve_kv_cache_block_sizes(sched, vc)
            rec["scheduler_block"], rec["hash_block"] = sbs, hbs
            pool, mc = min(rec["num_blocks"]), rec["mc_by_seqs"]["24"]
            rec["F"] = math.ceil(round(pool / mc - 131072 / block, 3))
        except BaseException as e:
            rec.update(error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1200:])
        out["runs"][tag] = rec
    out["ok"] = all("error" not in r for r in out["runs"].values())
except BaseException as e:
    out.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
print("@JSON " + json.dumps(out, default=str))
"""
KV34 = venv_py(E34KV, [str(K / "q_fixture.json"), "__QPIN__", "__DPIN__", str(G["CKPT"])], name="e34_kv", timeout=1800, platform=True)
REP["checks"]["E34_KV"] = KV34
_r = ((KV34.get("json") or {}).get("runs") or {})
_cp = _E.CPU_PLANNER
_okb = lambda t, pool: isinstance(min((_r.get(t) or {}).get("num_blocks") or [0]), int) and abs(min((_r.get(t) or {}).get("num_blocks") or [0]) - pool) <= 2
NBD1_REC["kv"] = {t: {x: v.get(x) for x in ("num_blocks", "stride", "group_blocks", "mc_by_seqs", "engine_block", "scheduler_block", "hash_block", "F",
                                             "max_mamba_state_page", "attn_page", "BLOCK_RAISED", "error")} for t, v in _r.items()}
NBD1_OK["e34_kv_probes_ok"] = (bool(_r) and _okb("B", _cp["B"]["pool"]) and _okb("S16", _cp["S16"]["pool"]) and _okb("S16B2", _cp["S16B2"]["pool"])
                               and _okb("S16B2_SW1024", _cp["S16B2_SW1024"]["pool"]) and (_r.get("fp32_block2048") or {}).get("BLOCK_RAISED") is True
                               and (_r.get("S16B2") or {}).get("scheduler_block") == 2048 and (_r.get("S16B2") or {}).get("hash_block") == 64
                               and (_r.get("S16B2") or {}).get("engine_block") == 512 and (_r.get("B") or {}).get("engine_block") == 1024
                               and abs(((_r.get("B") or {}).get("mc_by_seqs") or {}).get("24", 0) - 12.25) < 0.05
                               and (_r.get("S16B2") or {}).get("F") in (17, 18))
# ---- the never-do lift, the launch env, the plan, the switch, the TF traces, the launch path
_nd = {}
for k in ARMS5:
    p_ = _pl1.get(k) or {}
    raw = G["T"].never_do_violations({}, p_.get("argv") or [])
    _nd[k] = {"raw": raw, "notebook_launch_check": p_.get("launch_check"), "ok": raw == ([_E.LIFTED] if k in _E.SSM_ARMS else []) and p_.get("launch_check") == []}
NBD1_REC["never_do"] = _nd
NBD1_OK["e34_neverdo_lift_exact"] = len(_nd) == len(ARMS5) and all(v["ok"] for v in _nd.values())
_qe = {}
for k in ARMS5:
    a_ = G["ARMS"][k]
    e_ = (_pl1.get(k) or {}).get("d1_env") or {}
    want_ = {kk: a_["env"].get(kk) for kk in tuple(G["D1"].D1_ENV_KEYS) + ("DFLASH_DRAFT_EMBED_HOST", "DFLASH_DRAFT_SW_BLOCK")}
    _qe[k] = {"launch_env": e_, "planned": want_, "ok": bool(e_) and e_ == want_ and e_.get("DFLASH_DRAFT_SW_BLOCK") == str(a_["e34"]["sw"]),
              "spawn_ok": (_pl1.get(k) or {}).get("spawn_ok")}
NBD1_REC["launch_env"] = _qe
NBD1_OK["e34_launch_env_ok"] = (len(_qe) == len(ARMS5) and all(v["ok"] and v["spawn_ok"] is True for v in _qe.values())
                                and _qe["B"]["launch_env"].get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB") == "128")
try:
    _E.check_arm_plan(G["D1"], G["N"], G["T"], G["ARMS"], G["NBDF_ARMS"]["D3s"], G["NBD1_SWITCH"])
    _sm = _E.arm_plan(G["D1"], G["N"], G["NBDF_ARMS"]["D3s"], seam_env_k4=G["SEAM_ENVS"].get("D4s"), switch=_E.COPIES["smoke"])
    _E.check_arm_plan(G["D1"], G["N"], G["T"], _sm, G["NBDF_ARMS"]["D3s"], _E.COPIES["smoke"])
    for _w in range(_E.W_MIN, _E.W_MAX + 1):
        _a = dict(G["ARMS"])
        _a["S16B2W"] = _E.lever_arm(G["D1"], G["N"], G["ARMS"]["B"], "S16B2W", G["NBD1_SWITCH"], _w)
        _E.check_arm_plan(G["D1"], G["N"], G["T"], _a, G["NBDF_ARMS"]["D3s"], G["NBD1_SWITCH"])
    _ca = True
except Exception as e:
    _ca = f"{type(e).__name__}: {e}"[:300]
NBD1_REC["check_arm_plan"] = _ca
NBD1_OK["e34_arm_plan_ok"] = _ca is True and tuple(G["ARMS"]) == tuple(_E.ARM_KEYS) == tuple(ARMS5)
NBD1_OK["e34_switch_default_ok"] = G.get("NBD1_SWITCH") == _E.COPIES["full"] and ((G.get("E34R") or {}).get("switch") or {}).get("absent") is False
NBD1_OK["e34_help_flag_gate_real_ok"] = K100_OK.get("k100_help_flag_gate_real_ok") is True
NBD1_OK["e34_captures_argv_packviews_ok"] = K100_OK.get("nbd1_captures_argv_packviews_ok") is True
_tf = G.get("TF_P") or {}
NBD1_REC["tf"] = {"bands": {b: len(v) for b, v in _tf.items()}, "record": (G.get("E34R") or {}).get("tf")}
NBD1_OK["e34_tf_long_ok"] = (G.get("E34_TF_LONG") is not None and list(_tf) == ["8k", "40k", "tf69k", "tf94k", "tf113k"]
                             and all(60000 <= len(_tf[k]) < 131072 for k in ("tf69k", "tf94k", "tf113k")))
_per = {}
for k in ARMS5:
    p_ = _pl1.get(k) or {}
    _per[k] = {"sseq_check + pseq_before_arm (nbd1_main loop)": p_.get("sseq_check", [None])[0] is True and p_.get("pseq_before_arm", [None])[0] is True,
               "tree_check": bool((p_.get("tree_check") or {}).get("ok")),
               "arm_boot knobs (NB1_EXTRA / NB1_ENV / NB1_KV at q_pin, ARM_OF, no DRAFT_HEAD)": p_.get("pin") == G["K100_CFG"]["q_pin"] and not p_.get("build_env"),
               "K6 before launch": (p_.get("k6_before_launch") or {}).get("ok") is True,
               "Merlin build + host_phase": p_.get("host_phase") == "ok",
               "--help flag gate (L.help_check + KL.cli_flags_check)": ((p_.get("cli_flags") or {}).get("ok") is True),
               "the notebook's launch check (E34.launch_check) at the pin": p_.get("ok") is True and p_.get("launch_check") == [],
               "VLLM_QSA_KV_SCALES": str(p_.get("qsa_scales_env") or "").endswith("qsa_kv_scales_262k.json"),
               "env in a spawned child (the arm's expect)": p_.get("spawn_ok") is True,
               "the launch-env keys as planned (SW block)": _qe[k]["ok"],
               "argv spec / captures / seqs / KV bytes (R5 + captures item)": not (R5.get("field_check") or {}).get(k) and ((REP["checks"].get("K100_CAPTURES") or {}).get(k) or {}).get("ok") is True,
               "vLLM argparse: B + exactly its change": (ONE.get(k) or {}).get("ok") is True}
E34_STEPS = {"per_arm": _per, "table": REP["checks"].get("NBD1_STEP_TABLE"),
             "gpu_only_not_covered": ["boot_gate", "readiness", "the E34 read-back on the worker RPC (dtype / block / SW per rank) + the boot log",
                                      "k100_boot_gates", "smoke / oracle / long oracle / prefix / windows / TF / W re-plan"]}
E34_STEPS["all_covered"] = all(all(v.values()) for v in _per.values())
REP["checks"]["E34_STEPS"] = E34_STEPS
NBD1_OK["e34_launch_path_all_arms_ok"] = (len(_pl1) == len(ARMS5) and E34_STEPS["all_covered"] and R5.get("ok") is True
                                          and K100_OK.get("k100_notebook_launch_steps_all_ok") is True)
log("E34", json.dumps({"ok": NBD1_OK, "rec": {k: v for k, v in NBD1_REC.items() if k not in ("launch_env", "kv")}}, default=str)[:3000])
save()
'''
compile(E34_ITEMS, "<e34 kit items>", "exec")
NL = chr(10)
E34_ITEMS_PACKED = (NL + "# ==================================================================== E34 items (this kit only; packed to keep the kit < 400 kB:" + NL
                    + "# the text is build_e34_cpu_kit.py's E34_ITEMS, sha256 " + sha(E34_ITEMS) + ")" + NL
                    + 'exec(compile(__import__("lzma").decompress(__import__("base64").b85decode("'
                    + base64.b85encode(lzma.compress(E34_ITEMS.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
                    + '")).decode("utf-8"), "<e34 kit items>", "exec"))' + NL)
R = sub(R, NBD1_ITEMS, E34_ITEMS_PACKED, "items")
REAL = R
_kv_i = REAL.index('KVPROBE = r"""')
_kv_t = REAL[_kv_i:REAL.index('"""', _kv_i + 14)]
assert _kv_t.count('    ranks = fx["q_workers"]["ranks"]\n') == 1, "E34KV: the KVPROBE prefix anchor"
assert all(x in _kv_t for x in ("def worker_specs(", "def k_shapes(", "BK = {}", "from vllm.v1.attention.backends import utils as AU"))
for _k in E.KIT_KEYS:
    assert f'NBD1_OK["{_k}"]' in E34_ITEMS, f"kit key {_k} never set"
STEP_TABLE = [
    ["CELL 0 identity submission (E18 R2)", "R0 cell0_identity_submission"],
    ["CELL 1-3 inputs, lab, merge (overlay + K6 + digests)", "R1 inputs / merge / sha256"],
    ["CELL 7 stub: the payload dataset body + nbd1_lib + e34_lib (sha)", "NBD1_PAYLOAD (mount by sha; embedded copy only when absent)"],
    ["CELL 7: NB-D1's setup, then E34.arm_plan + E34.check_arm_plan (4 arms)", "e34_arm_plan_ok (+ the SMOKE plan, every W 24 .. 40)"],
    ["CELL 1 copy block E34_ON (the FULL switch)", "e34_switch_default_ok"],
    ["CELL 7: the TF set (the trace dataset, sha)", "e34_tf_long_ok"],
    ["nbd1_main: merlin_install / apply_60490 / k100_setup / sha_gate / PSEQ / SSEQ", "R2 / R3 / K100_SETUP / R1 / R3P / R3S"],
    ["per arm: SSEQ.check + PSEQ.before_arm + arm_boot first try + merlin_fn_server (K6, build, host_phase, E34.launch_check, --help gate)",
     "R4 plans (all 4) + e34_launch_path_all_arms_ok + e34_help_flag_gate_real_ok"],
    ["the never-do lift (exactly the SSM bf16 entry on S16 / S16B2 / S16B2W)", "e34_neverdo_lift_exact"],
    ["the spawned engine env (SW block 1024 / 512, EMBED_HOST, QSA)", "e34_launch_env_ok"],
    ["argv: seqs / KV bytes / spec / captures per arm", "R5 + e34_captures_argv_packviews_ok"],
    ["vLLM's own parsing of the final argv (SSM dtype, block, seqs, captures)", "e34_argparse_change_per_arm"],
    ["the installed Merlin's SSM bf16 / block 2048 source path + GDN dtypes", "e34_merlin_source_ok"],
    ["KV pool / F / blocks at the pin (B, S16, S16B2, SW 1024 control, fp32 x 2048 raise)", "e34_kv_probes_ok"],
    ["GPU only: boot, read-back, smoke, oracle, long oracle, prefix, windows, TF, W re-plan, LIVE", "NOT ON CPU; their pure logic = the units kit"],
    ["finally: seam revert, ppack revert", "R3S pristine / K100_SEAM"]]

# ---------------------------------------------------------------- build
assert hashlib.sha256((E27B / "nbdf_lib.py").read_bytes()).hexdigest() == FACTS["nbdf_lib_sha256"], "nbdf_lib != the build's"
EXPORT_BLOB = blob({"nbdf_code_export.py": EXPORT})
PAY_BLOB = blob({"files": {"e34_cell7_body.py": BODY, "nbd1_lib.py": LIB_D1, "e34_lib.py": LIB_E34}, "shas": dict(FACTS["payload_dataset"]["files"])})
MODELS = list(FACTS["model_sources"])
DATASETS = [d for d in FACTS["dataset_sources"] if d != FACTS["payload_dataset"]["slug"]]   # the payload is embedded; the TF traces are mounted
KITS = {"real": {"body": REAL, "title": "Q27 E34 Kit Real", "models": MODELS, "datasets": DATASETS}}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__EXPORTBLOB__", EXPORT_BLOB), ("__FIXBLOB__", FIXTURE_BLOB), ("__PAYBLOB__", PAY_BLOB),
                 ("__NBSHA__", NB_SHA), ("__NAME__", name), ("__SRCSHAS__", json.dumps(SRC_SHAS)), ("__KITARMS__", json.dumps(ARMS4)),
                 ("__QBLOCKS__", "480"), ("__QPIN__", "7933526016"), ("__DPIN__", "7933526016"), ("__VARIANT__", "k100"),
                 ("__STEPTABLE__", json.dumps(STEP_TABLE))):
        code = code.replace(k, v)
    assert "__" + "EXPORTBLOB" not in code and "__PAYBLOB__" not in code and "__KITARMS__" not in code and "__STEPTABLE__" not in code
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e34-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, separators=(",", ":")).encode("utf-8")
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
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| E34", NB_SHA[:8], flush=True)
    return out, code, data


if __name__ == "__main__":
    for n in KITS:
        build(n)
    for p in FROZEN:
        assert sha(p.read_bytes()) == FROZEN_SHA[str(p)], f"FROZEN FILE CHANGED: {p}"
