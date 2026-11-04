"""E25 NB-S2 Kaggle CPU REAL kit (pinned Py3.12 image, no GPU, no internet, private; account foysalemonshanto: the klee100 +
drafter Models and the tf-ref dataset are foysal's). The units kit is built by build_e25_nbs2_units_kit.py (it ships the
export + facts + body + libs; the switch tests read the export).

  real   q27-e25-nbs2-kit-real: the NBDF-K100 real kit (build_e27_nbdf_cpu_kit.py, k100 variant: its REAL template, read-only,
         taken from that module's namespace) re-pointed at the NB-S2 notebook's export, with marked "E25 NB-S2" edits:
           - the notebook's own pre-launch functions in nbs2_main's order: merge (CELL 3) -> merlin_install -> apply_60490 ->
             k100_setup (scales sidecar) -> sha_gate -> PSEQ.before_arm(B) -> SSEQ.apply -> per arm SSEQ.check + PSEQ.before_arm
             -> arm_boot's first try (tree_check, NB1_EXTRA / ENV / KV at the K100 pin, ARM_OF) -> merlin_fn_server (K6, Merlin
             build + host_phase, NBDF launch_check, the --help flag gate with VLLM_TARGET_DEVICE=cpu) for all 7 arms;
           - the spawned-child env, the argv fields (seqs per arm), R10 spec preflight (DYN's schedule field), R12 drafter load;
           - R9 get_kv_cache_configs: Q 480 calibration, B = K3_swblock1024, DYN ~ K4_swblock1024, + B_idxfp8 (IDX's pool) and
             max concurrency at 21 / 24 / 28 / 32 sequences;
           - NBS2PROBE in the Merlin venv: every arm's FINAL argv through vLLM's own serve parser + EngineArgs (one change vs B
             at the argparser level), Merlin's build_dynamic_sd_schedule_lookup == nbs2_lib.dense_schedule, the V2 FULL-graph
             rule, AttentionConfig fp8 resolution, the shared-prefix checkpoint field / conditions;
           - the captures per arm (DYN per tier) + the Q-defer pack views up to 32 requests; then seam + ppack reverted.
         Prints `@@KIT real ALL OK@@` or `@@KIT real NOT OK: <items>@@`; the step-coverage table is REP["checks"]["NBS2_STEPS"].
         S2-4 (kit only): the argparse probe maps the serve positional model_tag to model exactly as vLLM's own
         `vllm serve` does (entrypoints/cli/serve.py ServeSubcommand.cmd) before EngineArgs.from_cli_args, with HF_HUB_OFFLINE=1
         (venv_py): EngineArgs.__post_init__ then resolves the LOCAL model path (repo_utils.get_model_path: os.path.exists)
         instead of looking up the default HF repo id offline (the LocalEntryNotFoundError of kits_wave2).
         NB-S2 switches: the export's CELL 1 line NBS2_ON (default every arm on) -> nbs2_switch_default_all_on; every arm is
         kit-checked whatever the switch says.
Run: python build_e25_nbs2_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbs2-kit-real__foysalemonshanto/
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
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
OWNER = "foysalemonshanto"
NB_PATH = HERE.parent / "Q27-XHIGH-E25_NBS2_CAPACITY.ipynb"
FACTS = json.loads((HERE / "nbs2_build_facts.json").read_text(encoding="utf-8"))
NB_SHA = hashlib.sha256(NB_PATH.read_bytes()).hexdigest()
assert FACTS["sha256"] == NB_SHA, "nbs2 build facts are not for this notebook: rebuild"
EXPORT = (HERE / "nbs2_code_export.py").read_text(encoding="utf-8")
assert hashlib.sha256(EXPORT.encode("utf-8")).hexdigest() == FACTS["export_sha256"]
BODY = (HERE / "nbs2_cell7_body.py").read_text(encoding="utf-8")
LIBS = (HERE / "nbs2_lib.py").read_text(encoding="utf-8")
assert hashlib.sha256(BODY.encode()).hexdigest() == FACTS["body_file_sha256"] and hashlib.sha256(LIBS.encode()).hexdigest() == FACTS["nbs2_lib_sha256"]
ARMS7 = list(FACTS["s2_cfg"]["arms"])


def blob(files):
    return base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


def sub(text, old, new, where):
    n = text.count(old)
    assert n == 1, f"NB-S2 kit anchor {where}: {n} matches"
    return text.replace(old, new)


# ---------------------------------------------------------------- the NBDF-K100 kit templates (read-only module namespace)
os.environ["NBDF_VARIANT"] = "k100"
_argv = sys.argv[:]
sys.argv = [str(E27B / "build_e27_nbdf_cpu_kit.py")]
KG = runpy.run_path(str(E27B / "build_e27_nbdf_cpu_kit.py"), run_name="nbdfkit_for_nbs2")
sys.argv = _argv
HEAD, REAL0 = KG["HEAD"], KG["REAL"]
FIXTURE_BLOB, IMAGE = KG["FIXTURE_BLOB"], KG["IMAGE"]
SRC_FILES = tuple(KG["SRC_FILES"]) + ("vllm/v1/spec_decode/dynamic/utils.py", "vllm/config/attention.py", "vllm/config/scheduler.py",
                                      "vllm/config/cache.py", "vllm/v1/core/kv_cache_manager.py", "vllm/config/vllm.py")
SRC_SHAS = {rel: hashlib.sha256((MERLIN_SRC / rel).read_bytes()).hexdigest() for rel in SRC_FILES}

HEAD = HEAD.replace("# E27 NBDF CPU kit __NAME__", "# E25 NB-S2 CPU kit __NAME__ (from the E27 NBDF-K100 kit)").replace(
    'REP = {"what": f"E27 NBDF CPU kit {KIT}"', 'REP = {"what": f"E25 NB-S2 CPU kit {KIT}"').replace(
    '(W / f"nbdf_kit_{KIT}_report.json")', '(W / f"nbs2_kit_{KIT}_report.json")')

# ---------------------------------------------------------------- REAL: the marked NB-S2 edits
R = REAL0
R = sub(R, '''FILES = unpack("__EXPORTBLOB__")
''', '''FILES = unpack("__EXPORTBLOB__")
# ---- E25 NB-S2: the payload dataset (CELL 7 body + nbs2_lib, sha-pinned): the mount first; an embedded copy only when absent
PAY = unpack("__PAYBLOB__")
PAYLOAD = {"mount": None, "embedded_used": False}
for _d in (Path(INPUT_ROOT) / "datasets/foysalemonshanto/q27-e25-nbs2-payload", Path(INPUT_ROOT) / "q27-e25-nbs2-payload"):
    if all((_d / n).is_file() and hashlib.sha256((_d / n).read_bytes()).hexdigest() == h for n, h in PAY["shas"].items()):
        PAYLOAD["mount"] = str(_d)
        break
if PAYLOAD["mount"] is None:
    (K / "payload").mkdir(exist_ok=True)
    for n, txt in PAY["files"].items():
        (K / "payload" / n).write_text(txt, encoding="utf-8", newline="\\n")
    os.environ["NBS2_PAYLOAD_DIR"] = str(K / "payload")
    PAYLOAD["embedded_used"] = True
REP["checks"]["NBS2_PAYLOAD"] = PAYLOAD
''', "payload")
R = sub(R, '''    _okp, _whyp = G["PSEQ"].before_arm(G["ARMS"]["Q"])          # E27 K100: nbdf_main's own call (PatchSeq -> _ppack_apply)''',
        '''    _okp, _whyp = G["PSEQ"].before_arm(G["ARMS"]["B"])          # E25 NB-S2: nbs2_main's own call (PatchSeq -> _ppack_apply)''', "pseq B")
R = sub(R, '''if lv(x, "--max-num-seqs") != "24": bad.append("seqs")''',
        '''if lv(x, "--max-num-seqs") != str(a["expect_cfg"]["max_num_seqs"]): bad.append("seqs")   # E25 NB-S2: per arm''', "R5 seqs")
R = sub(R, '''_xa = next((plans[k_].get("argv") for k_ in ("D3s", "D4s") if (plans.get(k_) or {}).get("argv")), None) or []''',
        '''_xa = next((plans[k_].get("argv") for k_ in ("B",) if (plans.get(k_) or {}).get("argv")), None) or []   # E25 NB-S2''', "kx argv")
R = sub(R, '''K100_KX = {"argv_from": next((k_ for k_ in ("D3s", "D4s") if (plans.get(k_) or {}).get("argv")), None), "seqs": 24,''',
        '''K100_KX = {"argv_from": next((k_ for k_ in ("B",) if (plans.get(k_) or {}).get("argv")), None), "seqs": 24,''', "kx from")
R = sub(R, '''            for s_ in (21, 24):''', '''            for s_ in (21, 24, 28, 32):          # E25 NB-S2: S28 / S32''', "mc by seqs")
R = sub(R, '''                        "q_pages": out["runs"]["Q"].get("pages"), "q_idxfp8_pages": qi.get("pages"), "error": qi.get("error")}
''', '''                        "q_pages": out["runs"]["Q"].get("pages"), "q_idxfp8_pages": qi.get("pages"), "error": qi.get("error")}
        if out.get("q_calibrated"):            # E25 NB-S2: IDX on B (the D3s SW-1024 layout, MTP layer dropped)
            run("B_idxfp8", worker_specs(ranks_i, 3, dict(bf16, block_size=1024), drop_mtp=True), D_PIN, "FLASH_ATTN")
            KO["r9_idx_b"] = {"b_blocks": (out["runs"].get("K3_swblock1024") or {}).get("num_blocks"),
                              "b_idxfp8_blocks": out["runs"]["B_idxfp8"].get("num_blocks"), "error": out["runs"]["B_idxfp8"].get("error")}
''', "R9 B idx")
R = sub(R, '''         "q_i8_draft_head_build_env": (_pl.get("Q") or {}).get("build_env") == ({"DRAFT_HEAD": "int8"} if (_pl.get("Q") or {}).get("q_i8", {}).get("on") else {}),''',
        '''         "no_draft_head_build_env": all(not (_pl.get(k) or {}).get("build_env") for k in ARMS5),   # E25 NB-S2: no I8 on DFlash2''', "steps i8")
R = sub(R, '''("K100_CAPTURES", k100_captures, "k100_captures_argv_packviews_seqs24_ok")):''',
        '''("K100_CAPTURES", nbs2_captures, "nbs2_captures_argv_packviews_ok")):   # E25 NB-S2: per arm (DYN per tier), 32 requests''', "captures item")
R = sub(R, '''for _key, _fn, _vk in (("K100_SETUP_CHK",''', r'''def nbs2_captures():
    """E25 NB-S2 (replaces item 9): every arm's captures = the plan (multiples of 4 up to seqs x 4; DYN: b x (1 + K(b)) per
    tier), every decode batch an exact FULL graph at its own K (nbs2_lib coverage), the argv's captures / seqs / KV bytes /
    spec, cudagraph_mode FULL_AND_PIECEWISE; the Q-defer pack views at [n <= 32, K 3 / 4] (nb4_ppack nb4_pack_views)."""
    S2_, lv = G["S2"], G["T"].last_value
    c = {}
    for k in ARMS5:
        a = G["ARMS"].get(k) or {}
        s_ = int(a.get("streams") or 0)
        exp = S2_.dyn_captures(S2_.DYN_SCHEDULE, s_, S2_.DYN_K) if k == "DYN" else [4 * b for b in range(1, s_ + 1)]
        cov = S2_.dyn_coverage(a.get("captures") or [], S2_.DYN_SCHEDULE, s_, S2_.DYN_K) if k == "DYN" else G["N"].coverage(a.get("captures") or [], s_, 3)
        x = (plans.get(k) or {}).get("argv") or []
        try:
            cm = json.loads(lv(x, "--compilation-config") or "{}").get("cudagraph_mode")
        except Exception:
            cm = None
        bad = []
        if a.get("captures") != exp: bad.append(f"arm captures {a.get('captures')}")
        if G["N"].captures_from_argv(x) != exp: bad.append(f"argv captures {G['N'].captures_from_argv(x)}")
        if lv(x, "--max-num-seqs") != str(s_): bad.append(f"seqs {lv(x, '--max-num-seqs')}")
        if lv(x, "--kv-cache-memory-bytes") != str(K100_PIN_B) or a.get("kv_bytes") != K100_PIN_B: bad.append("kv bytes")
        if G["T"].spec_from_argv(x) != a.get("spec"): bad.append(f"spec {G['T'].spec_from_argv(x)} != {a.get('spec')}")
        sp = a.get("spec") or {}
        if sp.get("method") != "dflash" or sp.get("num_speculative_tokens") != (4 if k == "DYN" else 3): bad.append("spec method / K")
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


for _key, _fn, _vk in (("K100_SETUP_CHK",''', "nbs2_captures def")

NBS2_ITEMS = r'''
# ==================================================================== E25 NB-S2 items (this kit only)
NBS2_OK, NBS2_REC = {}, {}
NBS2PROBE = r"""
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
            if getattr(ns, "model_tag", None) is not None:      # S2-4: = vllm serve (cli/serve.py ServeSubcommand.cmd): the positional wins
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
    out["dyn_dense_merlin"] = build_dynamic_sd_schedule_lookup(dyn["schedule"], dyn["seqs"], dyn["k"])
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
print("@JSON " + json.dumps(out, default=str))
"""
(K / "nbs2_plans.json").write_text(json.dumps({k: (plans.get(k) or {}).get("argv") or [] for k in ARMS5}))
_dyn = {"schedule": G["S2"].DYN_SCHEDULE, "seqs": 24, "k": G["S2"].DYN_K}
NP = venv_py(NBS2PROBE, [str(K / "nbs2_plans.json"), json.dumps(_dyn)], name="nbs2_probe", timeout=1800, platform=True)
REP["checks"]["NBS2_PROBE"] = NP
npj = NP.get("json") or {}
ap = npj.get("argparse") or {}
CHANGE = {"IDX": {"indexer_kv_dtype"}, "S28": {"max_num_seqs", "captures"}, "S32": {"max_num_seqs", "captures"}, "DYN": {"spec", "captures"},
          "LP": {"max_num_batched_tokens", "long_prefill_token_threshold", "long_prefill_token_threshold_adaptive"},
          "SPC": {"enable_mamba_shared_prefix_checkpoint"}}
ONE = {}
_b = ap.get("B") or {}
for k in ARMS5:
    r = ap.get(k) or {}
    if k == "B":
        a = G["ARMS"]["B"]
        ONE[k] = {"error": r.get("error"), "ok": not r.get("error") and r.get("spec") == a["spec"] and r.get("captures") == a["captures"]
                  and r.get("max_num_seqs") == 24 and r.get("tp") == 2 and r.get("pp") == 2 and r.get("kv_cache_memory_bytes") == K100_PIN_B
                  and r.get("enable_prefix_caching") is True and r.get("mamba_cache_mode") == "align", "parsed": r}
        continue
    diff = sorted(f for f in set(r) | set(_b) if f not in ("error",) and r.get(f) != _b.get(f))
    a = G["ARMS"][k]
    want = {"IDX": r.get("indexer_kv_dtype") == "fp8", "S28": r.get("max_num_seqs") == 28, "S32": r.get("max_num_seqs") == 32,
            "DYN": (r.get("spec") or {}).get("num_speculative_tokens_per_batch_size") == [[1, 8, 4], [9, 24, 3]] if isinstance(r.get("spec"), dict) else False,
            "LP": r.get("max_num_batched_tokens") == 2048 and r.get("long_prefill_token_threshold") == 1024 and r.get("long_prefill_token_threshold_adaptive") is True,
            "SPC": r.get("enable_mamba_shared_prefix_checkpoint") is True}[k]
    ONE[k] = {"diff_vs_B": diff, "expected_diff": sorted(CHANGE[k]), "value_ok": want, "error": r.get("error"),
              "spec_equals_plan": r.get("spec") == a["spec"], "captures_equal_plan": r.get("captures") == a["captures"]}
    ONE[k]["ok"] = not r.get("error") and set(diff) == CHANGE[k] and want and ONE[k]["spec_equals_plan"] and ONE[k]["captures_equal_plan"]
REP["checks"]["NBS2_ONE_CHANGE_ARGPARSE"] = ONE
NBS2_OK["nbs2_argparse_one_change_per_arm"] = len(ONE) == len(ARMS5) and all(v["ok"] for v in ONE.values())
dd = G["S2"].dense_schedule(_dyn["schedule"], 24, 4)

NBS2_REC["dyn_schedule_merlin_record"] = {"equals_lib": npj.get("dyn_dense_merlin") == dd, "v2_keeps_full": npj.get("dyn_v2_keeps_full"),
                                         "note": "DYN moved to NB-S1: record only"}
NBS2_OK["nbs2_idx_fp8_resolves"] = npj.get("idx_resolve_fp8") == "fp8" and npj.get("idx_qsa_accepts_fp8") is True
NBS2_OK["nbs2_spc_lp_fields"] = npj.get("spc_field") is True and npj.get("spc_condition") is True and npj.get("lp_fields") is True
NBS2_REC["use_eagle_includes_dflash"] = npj.get("use_eagle_includes_dflash")
_rb = r9k.get("r9_idx_b") or {}
NBS2_REC["kv_pool"] = {"B_K3_swblock1024": _rb.get("b_blocks"), "IDX_B_idxfp8": _rb.get("b_idxfp8_blocks"), "error": _rb.get("error"),
                       "DYN_K4_swblock1024": _minb9("K4_swblock1024"), "q_480": _minb9("Q"),
                       "mc_by_seqs_K3": ((r9k.get("sim") or {}).get("K3_swblock1024") or {}).get("mc_by_seqs"),
                       "mc_by_seqs_K4": ((r9k.get("sim") or {}).get("K4_swblock1024") or {}).get("mc_by_seqs")}
NBS2_OK["nbs2_idx_b_kv_config_ok"] = (_isint(min(_rb.get("b_idxfp8_blocks") or [0])) and min(_rb.get("b_idxfp8_blocks") or [0]) > 0
                                      and not _rb.get("error"))
NBS2_REC["switch"] = (G.get("NBS2") or {}).get("switch")              # E25 NB-S2: CELL 1's NBS2_ON as the body read it
NBS2_OK["nbs2_switch_default_all_on"] = (G.get("NBS2_SWITCH") == {k: True for k in ARMS5}
                                         and ((G.get("NBS2") or {}).get("switch") or {}).get("absent") is False)
log("NBS2", json.dumps({"ok": NBS2_OK, "rec": NBS2_REC}, default=str)[:2500])
save()
'''
R = sub(R, '''REP["checks"]["K100_R9_IDX"] = dict(r9k.get("r9_idx") or {"error": "no R9_IDX record"}, record_only=True)''',
        '''REP["checks"]["K100_R9_IDX"] = dict(r9k.get("r9_idx") or {"error": "no R9_IDX record"}, record_only=True)''' + NBS2_ITEMS.replace('\\n"', '\\n"'),
        "nbs2 items")
STEP_TABLE = [
    ["CELL 0 identity submission (E18 R2)", "R0 cell0_identity_submission (the export run with RC2_UNIT + RC2_CPU_KIT)"],
    ["CELL 1-3 inputs, lab, merge (overlay + K6 + digests)", "R1 inputs / merge / sha256 (the notebook's own merge_ckpt + sha_gate)"],
    ["CELL 2b k100q_payload + klee100 model path", "the export's CELL 2b runs as is (tf-ref dataset attached)"],
    ["CELL 7 stub: the payload dataset body + nbs2_lib (sha)", "NBS2_PAYLOAD (mount checked by sha; embedded copy only when absent, recorded)"],
    ["CELL 7 drafter_facts (drafter Model pins)", "R1 drafter_pins_ok"],
    ["CELL 7 S2.arm_plan + S2.check_arm_plan", "R1 arms == the 7 arms (the body raised otherwise)"],
    ["nbs2_main: merlin_install", "R2 merlin_install_ok + Merlin source shas (flag sources incl. the NB-S2 files)"],
    ["nbs2_main: apply_60490", "R3 patch_60490_ok"],
    ["nbs2_main: k100_setup (scales sidecar)", "K100_SETUP / k100_setup_scales_sidecar_ok"],
    ["nbs2_main: sha_gate", "R1 sha256_all_ok"],
    ["nbs2_main: PSEQ.before_arm(B)", "R3P before_arm_q (B) / K100_STEPS ppack_pseq_before_arm_q"],
    ["nbs2_main: SSEQ.apply", "R3S seam_apply_verify_revert_ok"],
    ["nbs2_main per arm: SSEQ.check + PSEQ.before_arm", "R4 plans[arm].sseq_check / pseq_before_arm (all 7)"],
    ["arm_boot first try: tree_check, NB1_EXTRA / ENV / KV at 7,933,526,016, ARM_OF", "R4 plan_one = arm_boot (tree_check_all_5 = all 7)"],
    ["merlin_fn_server: K6 + Merlin build + host_phase + launch_check + --help flag gate", "R4 host_phase / launch_check / k6 / cli_flags (VLLM_TARGET_DEVICE=cpu for --help)"],
    ["the spawned engine env", "R4 env_in_spawned_child"],
    ["argv: seqs / KV bytes / dtype / spec / captures per arm", "R5 argv_fields_ok + NBS2 nbs2_captures (DYN per tier)"],
    ["vLLM's own parsing of the final argv (the arm's real config)", "NBS2_ONE_CHANGE_ARGPARSE (serve parser + EngineArgs, diff vs B)"],
    ["IDX: AttentionConfig fp8 + the QSA indexer accepts fp8; IDX pool", "nbs2_idx_fp8_resolves + nbs2_idx_b_kv_config_ok (R9 B_idxfp8)"],
    ["LP / SPC config fields + the SPC activation condition", "nbs2_spc_lp_fields"],
    ["the spec config (DYN's schedule field)", "R10 spec_config_preflight_ok"],
    ["the drafter load (draft_load_check)", "R12 drafter_loaded_weights_ok + K100_R12"],
    ["KV pool / F / block drop at the pin (B, DYN ~ K4), residents", "R9 kv_cache_configs_ok + K100_R9 / K100_F / K100_BLOCKDROP"],
    ["GPU only: boot, smoke, oracle, TF, windows, probes, ledger", "NOT ON CPU (GPU run); their pure logic = the units kit"],
    ["finally: seam revert, ppack revert", "R3S pristine / K100_SEAM"]]
R = sub(R, '''REP["checks"]["K100_STEPS"] = STEPS''', '''REP["checks"]["K100_STEPS"] = STEPS
REP["checks"]["NBS2_STEPS"] = {"table": json.loads(r"""__STEPTABLE__"""), "launch_steps": STEPS}''', "step table")
R = sub(R, '''"drafter_loaded_weights_ok": (R12.get("json") or {}).get("ok") is True, **K100_OK},''',
        '''"drafter_loaded_weights_ok": (R12.get("json") or {}).get("ok") is True, **K100_OK, **NBS2_OK,
         "nbs2_payload_dataset_mounted": PAYLOAD.get("mount") is not None},''', "verdict keys")
R = sub(R, '''recorded={**K100_REC, "patched_model_import": R11.get("json"),''',
        '''recorded={**K100_REC, **NBS2_REC, "payload": PAYLOAD, "patched_model_import": R11.get("json"),''', "verdict recorded")
REAL = R

# ---------------------------------------------------------------- the units kit: build_e25_nbs2_units_kit.py
assert hashlib.sha256((E27B / "nbdf_lib.py").read_bytes()).hexdigest() == FACTS["nbdf_lib_sha256"], "nbdf_lib != the build's"
EXPORT_BLOB = blob({"nbdf_code_export.py": EXPORT})
PAY_BLOB = blob({"files": {"nbs2_cell7_body.py": BODY, "nbs2_lib.py": LIBS},
                 "shas": dict(FACTS["payload_dataset"]["files"])})
MODELS = list(FACTS["model_sources"])
KITS = {"real": {"body": REAL, "title": "Q27 E25 NBS2 Kit Real", "models": MODELS, "datasets": list(FACTS["dataset_sources"])}}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__EXPORTBLOB__", EXPORT_BLOB), ("__FIXBLOB__", FIXTURE_BLOB), ("__PAYBLOB__", PAY_BLOB),
                 ("__NBSHA__", NB_SHA), ("__NAME__", name), ("__SRCSHAS__", json.dumps(SRC_SHAS)), ("__KITARMS__", json.dumps(ARMS7)),
                 ("__QBLOCKS__", "480"), ("__QPIN__", "7933526016"), ("__DPIN__", "7933526016"), ("__VARIANT__", "k100"),
                 ("__STEPTABLE__", json.dumps(STEP_TABLE))):
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e25-nbs2-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__{OWNER}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"{OWNER}/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": spec["datasets"], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB-S2", NB_SHA[:8], flush=True)
    return out, code, data


if __name__ == "__main__":
    for n in KITS:
        build(n)
