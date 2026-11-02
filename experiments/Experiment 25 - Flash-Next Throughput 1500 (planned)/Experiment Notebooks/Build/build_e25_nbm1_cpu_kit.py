"""E25 NB-M1 real CPU kit (pinned Py3.12 image, no GPU, no internet, private, foysalemonshanto): the NBDF-K100 real kit,
derived IN MEMORY from build_e27_nbdf_cpu_kit.py (NBDF_VARIANT=k100: every K100 substitution kept -- the scales sidecar
step, VLLM_TARGET_DEVICE=cpu for the notebook's own --help, PSEQ.before_arm(Q), arm_boot's first try, launch_check at the
K100 pin) and pointed at the NB-M1 kit notebook (every arm on) / facts / export. The NBDF files are read, never written
(frozen-file hash assert before and after).

The kit runs the NOTEBOOK's own functions: merge, merlin_install, apply_60490, k100_setup, sha_gate, PSEQ / SSEQ, then for
ALL 12 NB-M1 arms the notebook's own pre-launch sequence verbatim: nbdf_main's NB-M1 lever loop (the CELL 1 switch, M1.pre_gate
with the GATED build's kit gate (the only literal the gated rebuild changes) + the vocab table, PSEQ.before_arm), then
arm_boot's first try (tree_check, Q's I8 head / M1.apply_head -> NB4T_BUILD_ENV DRAFT_HEAD, NB1_EXTRA / NB1_ENV / NB1_KV at
the K100 pin, ARM_OF) -> merlin_fn_server (K6, Merlin build + host_phase, the --help flag gate, the NBDF launch check,
VLLM_QSA_KV_SCALES) -> nbdf_lib.launch_check at the pin, the env in a spawned child (the arm's whole expect: the M1 keys
included), the argv fields. NB-M1 changes to the derived template:
  - the arms = the NB-M1 plan (KIT_ARMS = nbm1_lib.ARM_KEYS); R5's seqs = each arm's own; R10's specs = each lever's MTP
    spec with its head (an MTP spec has no kv_cache_dtype: not a field error); the captures item per NB-M1 arm (the expected
    captures from the INSTALLED Merlin registry's capture_sizes_for, K4 multiples of 5, DYN tier-aware);
  - the KV probe (Merlin get_kv_cache_configs): + K4 (the K-dependent state rows at K 4, the MTP layer kept), the IDX run
    (the template's Q_idxfp8, 480 -> ~508), S28 / S32 (Q's configs: pool + max concurrency at 131,072 tokens at seqs 28 / 32);
  - vLLM's own argparse (Merlin venv, VLLM_TARGET_DEVICE=cpu) of EVERY arm's full launch argv: seqs, batched tokens, the
    long-prefill threshold + adaptive, indexer_kv_dtype, the spec dict (= launch_check's want_sp), captures, pin, layout;
  - the vocab tables: Merlin's own load_draft_vocab_ids(path, 248320, tp_size=2) on both mounted tables with
    VLLM_DRAFT_VOCAB_SHA256 (+ a wrong sha refused), the notebook's resolve_vocab found both, and the klee100 checkpoint's
    lm_head (BF16 [248320, 2560], untied) that select_head_rows needs;
  - the verdict keys nbm1_arm_plan_ok, nbm1_launch_path_all_arms_ok, nbm1_help_flag_gate_real_ok, nbm1_argparse_all_arms_ok,
    nbm1_kv_probes_ok, nbm1_vocab_tables_ok (the build gate's six) and the step-coverage table NBM1_STEPS.
Run: python build_e25_nbm1_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbm1-kit-real__foysalemonshanto/
     --light: the E25 NB-M1L (light) real kit on the light kit notebook (every arm on) / nbm1l facts / export / body
              -> q27-e25-nbm1l-kit-real__foysalemonshanto/ (the same checks: the light build changes no launch path)
     --out-root DIR: write the kit folder under DIR instead of D:/ARC AGI 2/_scratch/push_e25 (reproduction checks)"""
import hashlib
import json
import os
import sys
from pathlib import Path

LIGHT = "--light" in sys.argv
PFX = "nbm1l" if LIGHT else "nbm1"
HERE = Path(__file__).resolve().parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
NB = HERE.parent / ("Q27-XHIGH-E25_NBM1L.ipynb" if LIGHT else "Q27-XHIGH-E25_NBM1.ipynb")
TEMPLATE = E27B / "build_e27_nbdf_cpu_kit.py"
sys.path.insert(0, str(HERE))
import nbm1_lib as M1  # noqa: E402


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


FROZEN = sorted(p for p in E27B.glob("*") if p.is_file()) + sorted(p for p in E27B.parent.glob("*.ipynb"))
FROZEN_SHA = {str(p): sha(p.read_bytes()) for p in FROZEN}
ARM_KEYS = tuple(M1.ARM_KEYS)


def sub(text, old, new, where, n=1):
    c = text.count(old)
    assert c == n, f"NB-M1 kit anchor {where}: {c} matches (want {n})"
    return text.replace(old, new)


# ---------------------------------------------------------------- the outer builder, pointed at NB-M1 (text, in memory)
src = TEMPLATE.read_text(encoding="utf-8")
src = sub(src, 'NB_PATH = HERE.parent / ("Q27-XHIGH-E27_NBDF_K100.ipynb" if K100 else "Q27-XHIGH-E27_NBDF_DFLASH2.ipynb")',
          f'NB_PATH = Path(r"{NB}")', "nb path")
src = sub(src, 'FACTS = json.loads((HERE / ("nbdf_k100_build_facts.json" if K100 else "nbdf_build_facts.json")).read_text(encoding="utf-8"))',
          f'FACTS = json.loads(Path(r"{HERE / f"{PFX}_build_facts.json"}").read_text(encoding="utf-8"))', "facts")
src = sub(src, 'EXPORT = (HERE / ("nbdf_k100_code_export.py" if K100 else "nbdf_code_export.py")).read_text(encoding="utf-8")',
          f'EXPORT = Path(r"{HERE / f"{PFX}_code_export.py"}").read_text(encoding="utf-8")', "export")
src = sub(src, 'KV_Q_PIN, KV_D_PIN, KV_Q_BLOCKS, KIT_ARMS = _cfg["q_pin"], _cfg["q_pin"], 480, tuple(_cfg["arms"])',
          f'KV_Q_PIN, KV_D_PIN, KV_Q_BLOCKS, KIT_ARMS = _cfg["q_pin"], _cfg["q_pin"], 480, {ARM_KEYS!r}   # E25 NB-M1', "kit arms")
src = sub(src, 'slug = f"q27-e27-nbdf-k100-kit-{name}" if K100 else f"q27-e27-nbdf-kit-{name}"', 'slug = f"q27-e25-' + PFX + '-kit-{name}"', "slug")
src = sub(src, '"Q27 E27 NBDF K100 Kit Real" if K100 else "Q27 E27 NBDF Kit Real"', '"Q27 E25 NBM1L Kit Real"' if LIGHT else '"Q27 E25 NBM1 Kit Real"', "title")
src = sub(src, 'UNIT_FILES[_BODY_KEY] = (HERE / "nbdf_k100_cell7_body.py").read_text(encoding="utf-8")',
          f'UNIT_FILES[_BODY_KEY] = Path(r"{HERE / f"{PFX}_cell7_body.py"}").read_text(encoding="utf-8")   # E25 NB-M1 body', "units body")
os.environ["NBDF_VARIANT"] = "k100"
sys.dont_write_bytecode = True
ns = {"__file__": str(TEMPLATE), "__name__": "nbm1_kit_template"}
exec(compile(src, str(TEMPLATE) + " (NB-M1, in memory)", "exec"), ns)
assert ns["K100"] is True and ns["KIT_ARMS"] == ARM_KEYS and ns["NB_PATH"] == NB
assert M1.VOCAB_DS in ns["KITS"]["real"]["datasets"], ns["KITS"]["real"]["datasets"]

# ---------------------------------------------------------------- the REAL template (after the K100 substitutions): NB-M1 edits
R = ns["KITS"]["real"]["body"]
# R4 plan_one = nbdf_main's NB-M1 lever loop + arm_boot's first try (the notebook's own lines)
R = sub(R, '''    b = {"arm": key}
    if key != "Q":                                            # E27 K100: nbdf_main's per-D-arm checks before step(...)
        b["sseq_check"] = list(G["SSEQ"].check(key))
        b["pseq_before_arm"] = list(G["PSEQ"].before_arm(a))
    b["tree_check"] = G["tree_check"](a)                       # arm_boot: the first try
    kvb = int(G["K100_CFG"]["q_pin"])
    if key == "Q":
        a["draft_head"] = "int8" if G["Q_I8"]["on"] else None
        a["expect_cfg"]["draft_head"] = a["draft_head"]
        G["NB4T_BUILD_ENV"][nm] = {"DRAFT_HEAD": "int8"} if G["Q_I8"]["on"] else {}
''', '''    b = {"arm": key}
    if key != "Q":                                            # E25 NB-M1: nbdf_main's lever loop before step(...)
        b["m1_switch"] = G["M1_SWITCH"].get(key, True)
        b["m1_pre_gate"] = G["M1"].pre_gate(a, {"passed": True})   # the GATED build's kit gate + the vocab table found
        b["pseq_before_arm"] = list(G["PSEQ"].before_arm(a))
    b["tree_check"] = G["tree_check"](a)                       # arm_boot: the first try
    kvb = int(G["K100_CFG"]["q_pin"])
    if key == "Q":
        a["draft_head"] = "int8" if G["Q_I8"]["on"] else None
        a["expect_cfg"]["draft_head"] = a["draft_head"]
        G["NB4T_BUILD_ENV"][nm] = {"DRAFT_HEAD": "int8"} if G["Q_I8"]["on"] else {}
    elif a.get("m1_lever"):                                   # E25 NB-M1 arm_boot: the lever's draft head into build()
        G["M1"].apply_head(G["N"], a, G["Q_I8"]["on"])
        G["NB4T_BUILD_ENV"][nm] = {"DRAFT_HEAD": a["draft_head"]} if a["draft_head"] else {}
''', "plan_one = the NB-M1 loop + arm_boot")
R = sub(R, '''    b.update(pin=kvb, q_i8=dict(G["Q_I8"]) if key == "Q" else None, build_env=G["NB4T_BUILD_ENV"].get(nm))
    _gate_bad = [x for x in ("sseq_check", "pseq_before_arm") if x in b and b[x][0] is not True]''',
        '''    b.update(pin=kvb, q_i8=dict(G["Q_I8"]) if key == "Q" else None, build_env=G["NB4T_BUILD_ENV"].get(nm), head=a.get("draft_head"))
    _gate_bad = ([x for x in ("pseq_before_arm",) if x in b and b[x][0] is not True] + (["m1_pre_gate"] if b.get("m1_pre_gate") else [])
                 + (["m1_switch"] if b.get("m1_switch") is False else []))''', "plan_one gate")
R = sub(R, 'qsa_scales_env=s.env.get("VLLM_QSA_KV_SCALES"),',
        'qsa_scales_env=s.env.get("VLLM_QSA_KV_SCALES"), m1_env={k_: s.env.get(k_) for k_ in G["M1"].M1_ENV_KEYS},', "m1 env")
R = sub(R, 'sp = venv_py(SPAWN_ENV, [",".join(G["N"].WATCHED)], name=',
        'sp = venv_py(SPAWN_ENV, [",".join(list(a["expect"]))], name=', "spawn keys = the arm's expect")
# R5: each arm's own seqs
R = sub(R, '''if lv(x, "--max-num-seqs") != "24": bad.append("seqs")''',
        '''if lv(x, "--max-num-seqs") != str(a["expect_cfg"]["max_num_seqs"]): bad.append("seqs")   # E25 NB-M1: S28 / S32''', "R5 seqs")
# R10: every lever's MTP spec with its head; an MTP spec has no kv_cache_dtype key
R = sub(R, '''R10 = venv_py(SPECPROBE, [json.dumps({k: G["ARMS"][k]["spec"] for k in ARMS5 if k != "Q"})], name="r10_spec_preflight", platform=True)''',
        '''R10 = venv_py(SPECPROBE, [json.dumps({k: dict(G["ARMS"][k]["spec"], **({"draft_lm_head_quant": G["ARMS"][k]["draft_head"]} if G["ARMS"][k].get("draft_head") else {}))
                                      for k in ARMS5 if k != "Q"})], name="r10_spec_preflight", platform=True)   # E25 NB-M1: MTP specs + head''',
        "R10 specs")
R = sub(R, '''    out["kv_dtype_ok"] = {k: sp.get("kv_cache_dtype") in typing.get_args(CacheDType) for k, sp in specs.items()}''',
        '''    out["kv_dtype_ok"] = {k: ("kv_cache_dtype" not in sp) or sp.get("kv_cache_dtype") in typing.get_args(CacheDType) for k, sp in specs.items()}
    out["head_ok"] = {k: sp.get("draft_lm_head_quant") in (None,) + tuple(x for a in typing.get_args(hints.get("draft_lm_head_quant")) for x in (typing.get_args(a) or (a,)))
                      for k, sp in specs.items()}   # E25 NB-M1: the Literal int4 / int8''', "R10 kv dtype")
R = sub(R, '''                 and all(out["kv_dtype_ok"].values()) and all(out["backend_ok"].values()) and out["draft_tp_unset_is_target_tp"])''',
        '''                 and all(out["kv_dtype_ok"].values()) and all(out["backend_ok"].values()) and out["draft_tp_unset_is_target_tp"]
                 and all(out["head_ok"].values()))''', "R10 ok")
# the KV probe's argv inputs: Q's own argv
R = sub(R, 'for k_ in ("D3s", "D4s") if (plans.get(k_) or {}).get("argv")', 'for k_ in ("Q",) if (plans.get(k_) or {}).get("argv")', "kx argv", n=2)
# the KV probe: + K4 / IDX / S28 / S32
R = sub(R, '''    for tag in ("K3_swblock1024", "K4_swblock1024"):
        rec = {}
        if tag not in CF:''', '''    # ---- E25 NB-M1: the MTP lever KV probes (IDX = Q_idxfp8 above; K4 = the K-dependent state rows at K 4 with the MTP
    # layer kept; S28 / S32 = Q's KV configs at max_num_seqs 28 / 32: pool + max concurrency at 131,072 tokens)
    M1K = out.setdefault("nbm1", {})
    try:
        run("M1_K4", worker_specs(ranks, 4), Q_PIN)
        M1K["K4"] = {k_: out["runs"]["M1_K4"].get(k_) for k_ in ("num_blocks", "error", "kv_group_count", "max_concurrency_131072")}
    except BaseException as e:
        M1K["K4"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    M1K["IDX"] = {k_: (out["runs"].get("Q_idxfp8") or {}).get(k_) for k_ in ("num_blocks", "error", "kv_group_count")}
    M1K["IDX"]["q_blocks"] = (out["runs"].get("Q") or {}).get("num_blocks")
    for s_ in (24, 28, 32):
        try:
            cfgs_, ws_ = CF["Q"]
            v_ = real_vc(s_)
            sch_ = KU.generate_scheduler_kv_cache_config(cfgs_)
            M1K[f"seqs{s_}"] = {"pool": sch_.num_blocks, "mc_131072": KU.get_max_concurrency_for_kv_cache_config(v_, sch_)}
        except BaseException as e:
            M1K[f"seqs{s_}"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    for tag in ("K3_swblock1024", "K4_swblock1024"):
        rec = {}
        if tag not in CF:''', "R9 nbm1 runs")
# the captures item: every NB-M1 arm (expected captures from the INSTALLED registry; K4 multiples of 5; DYN tier-aware)
R = sub(R, '''    exp_caps = {"Q": ns_["capture_sizes_for"](K100_SEQS, 3), "D3s": [4 * b for b in range(1, K100_SEQS + 1)], "D4s": [5 * b for b in range(1, K100_SEQS + 1)]}''',
        '''    exp_caps = {k_: (ns_["capture_sizes_for"]({"S28": 28, "S32": 32}.get(k_, K100_SEQS), 3) if k_ not in ("K4", "DYN") else
                     [5 * b for b in range(1, K100_SEQS + 1)] if k_ == "K4" else
                     sorted({5 * b for b in range(1, 9)} | {4 * b for b in range(9, K100_SEQS + 1)})) for k_ in ARMS5}   # E25 NB-M1''', "exp caps")
R = sub(R, '''    for k in ("Q", "D3s", "D4s"):
        a = G["ARMS"].get(k) or {}
        q = int(a.get("k") or 0) + 1
        g = graphs(a.get("captures") or [], K100_SEQS, q)
        unc = [b for b in range(1, K100_SEQS + 1) if not any(x >= b for x in g)]''', '''    for k in ARMS5:
        a = G["ARMS"].get(k) or {}
        q = int(a.get("k") or 0) + 1
        _sq = int((a.get("expect_cfg") or {}).get("max_num_seqs") or K100_SEQS)   # E25 NB-M1: S28 / S32 at their seqs
        g = graphs(a.get("captures") or [], _sq, q)
        unc = ([b for b in range(1, _sq + 1) if b * (5 if b <= 8 else 4) not in (a.get("captures") or [])] if k == "DYN"
               else [b for b in range(1, _sq + 1) if not any(x >= b for x in g)])''', "caps loop")
R = sub(R, '''        if lv(x, "--max-num-seqs") != str(K100_SEQS): bad.append(f"seqs {lv(x, '--max-num-seqs')}")''',
        '''        if lv(x, "--max-num-seqs") != str(_sq): bad.append(f"seqs {lv(x, '--max-num-seqs')}")''', "caps seqs")
R = sub(R, '''        if sp.get("method") != ("mtp" if k == "Q" else "dflash") or sp.get("num_speculative_tokens") != {"Q": 3, "D3s": 3, "D4s": 4}[k]:''',
        '''        if sp.get("method") != "mtp" or sp.get("num_speculative_tokens") != (4 if k in ("K4", "DYN") else 3):''', "caps spec K")
R = sub(R, '''                "padded_b": [b for b in range(1, K100_SEQS + 1) if b not in g and b not in unc], "cudagraph_mode": cm,
                "lib_coverage": G["N"].coverage(a.get("captures") or [], K100_SEQS, q - 1).get("covers_all"), "argv_bad": bad}
        c[k]["ok"] = not unc and not bad and c[k]["lib_coverage"] is True''', '''                "padded_b": [b for b in range(1, _sq + 1) if b not in g and b not in unc], "cudagraph_mode": cm, "seqs": _sq,
                "lib_coverage": G["N"].coverage(a.get("captures") or [], _sq, q - 1).get("covers_all"), "argv_bad": bad}
        c[k]["ok"] = not unc and not bad and (c[k]["lib_coverage"] is True or k == "DYN")   # DYN: two tiers (checked above)''', "caps ok")
R = sub(R, '''    c["ok"] = all(c[k]["ok"] for k in ("Q", "D3s", "D4s")) and not badv''', '''    c["ok"] = all(c[k]["ok"] for k in ARMS5) and not badv''', "caps all")
# the Q-defer pack views (nb4_ppack nb4_pack_views) for every micro-batch size up to S32's 32 sequences (K 3 and K 4)
R = sub(R, '''        for n in range(1, K100_SEQS + 1):
            flat = torch.arange(n * (2 * kk + 3), dtype=torch.int64)''', '''        for n in range(1, 32 + 1):                            # E25 NB-M1: S32 (32 sequences)
            flat = torch.arange(n * (2 * kk + 3), dtype=torch.int64)''', "pack views 32")
R = sub(R, '''    c["pack_views"] = {"cases": 2 * K100_SEQS, "bad": badv[:8]}''', '''    c["pack_views"] = {"cases": 2 * 32, "bad": badv[:8]}''', "pack views cases")
# STEPS: the NB-M1 lever loop (no seam check on an MTP arm) + the lever heads
R = sub(R, '''         "d_arm_sseq_check_pseq_before_arm": all((_pl.get(k) or {}).get("sseq_check", [None])[0] is True
                                                 and (_pl.get(k) or {}).get("pseq_before_arm", [None])[0] is True for k in ARMS5 if k != "Q"),''',
        '''         "m1_lever_switch_pre_gate_pseq_before_arm": all((_pl.get(k) or {}).get("m1_switch") is True and (_pl.get(k) or {}).get("m1_pre_gate") is None
                                                         and (_pl.get(k) or {}).get("pseq_before_arm", [None])[0] is True for k in ARMS5 if k != "Q"),
         "m1_lever_head_build_env": all((_pl.get(k) or {}).get("build_env") == ({"DRAFT_HEAD": "int4"} if k == "I4" else
                                        {"DRAFT_HEAD": "int8"} if G["Q_I8"]["on"] else {}) for k in ARMS5 if k != "Q"),''', "steps lever loop")

PROBES = r'''
NBM1_ARGPARSE = r"""
import json, os, sys
argvs = json.load(open(sys.argv[1]))
out = {"_env": {k: os.environ.get(k) for k in ("VLLM_TARGET_DEVICE", "CUDA_VISIBLE_DEVICES")}}
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.engine.arg_utils import AsyncEngineArgs
except BaseException as e:
    print("@JSON " + json.dumps({"import_error": repr(e)[:400]})); raise SystemExit(0)
try:
    from vllm.platforms import current_platform
    out["_platform"] = type(current_platform).__name__
except BaseException as e:
    out["_platform"] = "ERROR " + repr(e)[:300]
def g(o, k):
    v = getattr(o, k, None)
    return v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)
for label, argv in argvs.items():
    try:
        a = argv[argv.index("serve") + 1:]
        if a and not a[0].startswith("--"):
            a = ["--model", a[0]] + a[1:]
        p = make_arg_parser(FlexibleArgumentParser())
        ns, unknown = p.parse_known_args(a)
        r = {"unknown": list(unknown)[:40]}
        try:
            ea, r["source"] = AsyncEngineArgs.from_cli_args(ns), "engine_args"
        except BaseException as e:
            ea, r["source"], r["from_cli_args_error"] = ns, "namespace", f"{type(e).__name__}: {e}"[:300]
        cc = getattr(ea, "compilation_config", None)
        caps = cc.get("cudagraph_capture_sizes") if isinstance(cc, dict) else getattr(cc, "cudagraph_capture_sizes", None)
        r.update({k: g(ea, k) for k in ("model", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes", "kv_cache_dtype",
                                         "worker_extension_cls", "max_model_len", "block_size", "mamba_cache_mode", "enable_prefix_caching",
                                         "tensor_parallel_size", "pipeline_parallel_size", "enable_expert_parallel", "language_model_only",
                                         "long_prefill_token_threshold", "long_prefill_token_threshold_adaptive")})
        r["capture_sizes"] = list(caps) if caps else None
        sc = getattr(ea, "speculative_config", None)
        r["speculative_config"] = sc if isinstance(sc, (dict, type(None))) else str(sc)[:300]
        _ac = getattr(ea, "attention_config", None)
        r["indexer_kv_dtype"] = (_ac.get("indexer_kv_dtype") if isinstance(_ac, dict) else getattr(_ac, "indexer_kv_dtype", None))
        r["attention_config"] = str(_ac)[:300]
        r["ok"] = True
    except BaseException as e:
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
    out[label] = r
print("@JSON " + json.dumps(out, default=str))
"""
NBM1_VOCABPROBE = r"""
import json, os, sys
out = {}
try:
    import vllm.models.qwen4_exp.nvidia.draft_head as DH
    out["env_names"] = {"file": getattr(DH, "ENV_DRAFT_VOCAB_FILE", None), "sha": getattr(DH, "ENV_DRAFT_VOCAB_SHA256", None)}
    os.environ[out["env_names"]["sha"] or "VLLM_DRAFT_VOCAB_SHA256"] = sys.argv[2]
    ids = DH.load_draft_vocab_ids(sys.argv[1], 248320, tp_size=2)
    out.update(ok=True, n=int(ids.numel()), dtype=str(ids.dtype), first=int(ids[0]), last=int(ids[-1]),
               sorted_unique=bool((ids[1:] > ids[:-1]).all()), contiguous=bool(ids.is_contiguous()),
               fingerprint=DH.draft_vocab_fingerprint(ids), align=int(DH.DRAFT_VOCAB_ALIGN))
    os.environ[out["env_names"]["sha"] or "VLLM_DRAFT_VOCAB_SHA256"] = "0" * 64
    try:
        DH.load_draft_vocab_ids(sys.argv[1], 248320, tp_size=2)
        out["wrong_sha_refused"] = False
    except ValueError as e:
        out["wrong_sha_refused"] = "sha256 mismatch" in str(e)
except BaseException as e:
    import traceback
    out.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1200:])
print("@JSON " + json.dumps(out, default=str))
"""
'''

NBM1_ITEMS = r'''
# ---- E25 NB-M1: the arm plan, the launch path of all 12 arms, the help gate, argparse, the KV probes, the vocab tables
_M1 = G["M1"]
try:
    _M1.check_arms(G["N"], G["T"], G["ARMS"], G["_nbm1_q0"])
    _ca = True
except Exception as e:
    _ca = f"{type(e).__name__}: {e}"[:300]
_pl = R4.get("plans") or {}
CPU_ENV_M1 = {"VLLM_TARGET_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": ""}
# vLLM's own argparse of EVERY arm's full launch argv (the argv the notebook would start)
RM1A = {}
try:
    _argvs = {k: (_pl.get(k) or {}).get("argv") for k in ARMS5 if (_pl.get(k) or {}).get("argv")}
    (K / "nbm1_argvs.json").write_text(json.dumps(_argvs))
    _r = venv_py(NBM1_ARGPARSE, [str(K / "nbm1_argvs.json")], name="nbm1_argparse", env_extra=CPU_ENV_M1)
    RM1A["parsed"] = _r["json"] or {"error": _r["tail"]}
    _fb = {}
    for k in ARMS5:
        a, x, bad = G["ARMS"][k], (RM1A["parsed"] or {}).get(k) or {}, []
        if x.get("ok") is not True: bad.append("parse " + str(x.get("error"))[:200])
        if x.get("unknown"): bad.append(f"unknown {x.get('unknown')}")
        if x.get("from_cli_args_error"): bad.append("from_cli_args " + str(x.get("from_cli_args_error"))[:200])
        if x.get("max_num_seqs") != a["expect_cfg"]["max_num_seqs"]: bad.append(f"max_num_seqs {x.get('max_num_seqs')}")
        if x.get("max_num_batched_tokens") != (2048 if k == "LP" else 1024): bad.append(f"max_num_batched_tokens {x.get('max_num_batched_tokens')}")
        if k == "LP" and (x.get("long_prefill_token_threshold"), x.get("long_prefill_token_threshold_adaptive")) != (1024, True):
            bad.append(f"long prefill {x.get('long_prefill_token_threshold')} {x.get('long_prefill_token_threshold_adaptive')}")
        if k != "LP" and (x.get("long_prefill_token_threshold") not in (None, 0) or x.get("long_prefill_token_threshold_adaptive") not in (None, False)):
            bad.append("long prefill set on a non-LP arm")
        if (k == "IDX") != (x.get("indexer_kv_dtype") == "fp8") or (k != "IDX" and x.get("indexer_kv_dtype") not in (None, "auto")):
            bad.append(f"indexer_kv_dtype {x.get('indexer_kv_dtype')}")
        _want = dict(a["spec"], draft_lm_head_quant=a["draft_head"]) if a.get("draft_head") else dict(a["spec"])
        _sp = x.get("speculative_config")
        if not isinstance(_sp, dict) or {kk: (list(map(list, vv)) if kk == "num_speculative_tokens_per_batch_size" and vv else vv) for kk, vv in _sp.items()} != _want:
            bad.append(f"spec {str(_sp)[:160]} != {_want}")
        if (x.get("capture_sizes") or []) != a["captures"]: bad.append(f"captures {x.get('capture_sizes')}")
        if x.get("kv_cache_memory_bytes") != int(G["K100_CFG"]["q_pin"]): bad.append(f"kv {x.get('kv_cache_memory_bytes')}")
        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")
        if x.get("language_model_only") is not True: bad.append("language_model_only")
        if x.get("kv_cache_dtype") != "fp8_e4m3": bad.append(f"kv_cache_dtype {x.get('kv_cache_dtype')}")
        if (x.get("tensor_parallel_size"), x.get("pipeline_parallel_size"), x.get("enable_expert_parallel")) != (2, 2, True): bad.append("layout")
        if x.get("worker_extension_cls") != "nb1_worker_ext.NB1WorkerExt": bad.append("worker_extension_cls")
        _fb[k] = bad
    RM1A["field_check"] = _fb
    RM1A["ok"] = len(_fb) == len(ARMS5) and len(_argvs) == len(ARMS5) and all(not v for v in _fb.values())
except Exception as e:
    RM1A.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
log("NB-M1 argparse", json.dumps(RM1A.get("field_check"), default=str)[:3000])
# the KV probes (R9's nbm1 runs + the template's Q_idxfp8)
_r9n = (R9.get("json") or {}).get("nbm1") or {}
_q9 = ((R9.get("json") or {}).get("runs") or {}).get("Q") or {}
_qb = min(_q9.get("num_blocks") or [0])
_kvm1 = {"K4": dict(_r9n.get("K4") or {}), "IDX": dict(_r9n.get("IDX") or {}),
         "S28": dict(_r9n.get("seqs28") or {}), "S32": dict(_r9n.get("seqs32") or {}), "S24": dict(_r9n.get("seqs24") or {})}
_kvm1["K4"]["ok"] = bool(_kvm1["K4"].get("num_blocks")) and not _kvm1["K4"].get("error") and isinstance(_kvm1["K4"].get("kv_group_count"), int)
_ib = min(_kvm1["IDX"].get("num_blocks") or [0])
_kvm1["IDX"]["ok"] = _ib > _qb > 0 and not _kvm1["IDX"].get("error")
for _s in ("S24", "S28", "S32"):
    _kvm1[_s]["ok"] = bool(_kvm1[_s].get("pool")) and isinstance(_kvm1[_s].get("mc_131072"), (int, float)) and _kvm1[_s]["mc_131072"] > 0 and not _kvm1[_s].get("error")
_kvm1["note"] = {"q_blocks": _qb, "idx_blocks": _ib, "kit_r9_ref": "R9_IDX 480 -> 508 (K100 kit)"}
# the vocab tables: the notebook's own resolve_vocab + Merlin's load_draft_vocab_ids on both + the klee100 lm_head
RM1V = {"resolved": G.get("M1_VOCAB")}
try:
    for k, (name, sh, n) in _M1.VOCAB_TABLES.items():
        v_ = (G.get("M1_VOCAB") or {}).get(k) or {}
        if not v_.get("found"):
            RM1V[k] = {"ok": False, "why": v_.get("why")}
            continue
        _r = venv_py(NBM1_VOCABPROBE, [str(v_["path"]), sh], name=f"nbm1_vocab_{k.lower()}", env_extra=CPU_ENV_M1)
        j = _r["json"] or {"error": _r["tail"]}
        j["ok_all"] = (j.get("ok") is True and j.get("n") == n and j.get("sorted_unique") is True and j.get("wrong_sha_refused") is True
                       and j.get("env_names") == {"file": _M1.ENV_VOCAB, "sha": _M1.ENV_VOCAB_SHA} and j.get("last", 10 ** 9) < 248077
                       and n % (int(j.get("align") or 128) * _M1.TP) == 0)
        j["arm_env"] = {kk: G["ARMS"][k]["env"].get(kk) for kk in (_M1.ENV_VOCAB, _M1.ENV_VOCAB_SHA)}
        j["arm_env_ok"] = j["arm_env"] == {_M1.ENV_VOCAB: v_["path"], _M1.ENV_VOCAB_SHA: sh}
        RM1V[k] = j
    _ck = Path(str(G["CKPT"]))
    _cj = json.loads((_ck / "config.json").read_text(encoding="utf-8"))
    _wm = json.loads((_ck / "model.safetensors.index.json").read_text())["weight_map"]
    _cand = [n_ for n_ in _wm if n_.endswith(("lm_head.weight", "shared_head.head.weight"))]
    _heads = {}
    for n_ in _cand:
        h_, _ = _hdr(_ck / _wm[n_])
        _heads[n_] = {"dtype": (h_.get(n_) or {}).get("dtype"), "shape": (h_.get(n_) or {}).get("shape"), "file": _wm[n_]}
    RM1V["lm_head"] = {"candidates": _heads, "tie_top": _cj.get("tie_word_embeddings"), "tie_text": (_cj.get("text_config") or {}).get("tie_word_embeddings")}
    RM1V["lm_head"]["ok"] = (bool(_heads) and all(v["dtype"] == "BF16" and list(v["shape"] or []) == [248320, 2560] for v in _heads.values())
                             and RM1V["lm_head"]["tie_top"] in (None, False) and RM1V["lm_head"]["tie_text"] in (None, False))
except Exception as e:
    RM1V.update(error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
log("NB-M1 vocab", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "tb"}) for k, v in RM1V.items()}, default=str)[:3000])
_chg = {}
for k in ARMS5:
    a_, e_ = G["ARMS"][k], (_pl.get(k) or {}).get("m1_env") or {}
    _chg[k] = {"launch_env": e_, "planned": {kk: a_["env"].get(kk) for kk in _M1.M1_ENV_KEYS}, "change": a_.get("m1_change")}
    _chg[k]["as_planned"] = e_ == _chg[k]["planned"] and len(e_) == len(_M1.M1_ENV_KEYS)
_capc = (REP["checks"].get("K100_CAPTURES") or {})
_kvof = {"K4": "K4", "IDX": "IDX", "S28": "S28", "S32": "S32"}
_per = {}
for k in ARMS5:
    p_ = _pl.get(k) or {}
    _per[k] = {"nbdf_main NB-M1 loop (switch, pre_gate, PSEQ.before_arm)": True if k == "Q" else (p_.get("m1_switch") is True and p_.get("m1_pre_gate") is None and p_.get("pseq_before_arm", [None])[0] is True),
               "tree_check": bool((p_.get("tree_check") or {}).get("ok")),
               "arm_boot knobs (head -> NB4T_BUILD_ENV, NB1_EXTRA / NB1_ENV / NB1_KV at q_pin, ARM_OF)": p_.get("pin") == G["K100_CFG"]["q_pin"]
               and p_.get("build_env") == ({"DRAFT_HEAD": p_.get("head")} if p_.get("head") else {}),
               "K6 before launch": (p_.get("k6_before_launch") or {}).get("ok") is True,
               "Merlin build + host_phase": p_.get("host_phase") == "ok",
               "--help flag gate (L.help_check + KL.cli_flags_check)": ((p_.get("cli_flags") or {}).get("ok") is True),
               "NBDF launch check + nbdf_lib.launch_check at the pin": p_.get("ok") is True and p_.get("launch_check") == [],
               "VLLM_QSA_KV_SCALES": str(p_.get("qsa_scales_env") or "").endswith("qsa_kv_scales_262k.json"),
               "env in a spawned child (the arm's expect)": p_.get("spawn_ok") is True,
               "the M1 env keys in the launch env as planned": _chg[k]["as_planned"],
               "argv spec / captures / seqs / KV bytes (R5 + captures item)": not (R5.get("field_check") or {}).get(k) and (_capc.get(k) or {}).get("ok") is True,
               "vLLM argparse of the full argv": not (RM1A.get("field_check") or {}).get(k, ["missing"]),
               "KV probe": (_kvm1.get(_kvof[k]) or {}).get("ok") is True if k in _kvof else None,
               "vocab table (Merlin's loader)": (RM1V.get(k) or {}).get("ok_all") is True and (RM1V.get(k) or {}).get("arm_env_ok") is True if k in _M1.VOCAB_TABLES else None}
NBM1_STEPS = {"global": [
    {"step": "merge (CELL 3)", "notebook": "R['sections']['A0']['merge']", "kit": "R1 merge", "covered": bool((R1.get("merge") or {}).get("ok"))},
    {"step": "install", "notebook": "merlin_install()", "kit": "R2 merlin_install()", "covered": R2.get("ok") is True},
    {"step": "#60490 + worker extension", "notebook": "apply_60490()", "kit": "R3 apply_60490()", "covered": R3.get("ok") is True},
    {"step": "k100_setup (overlay merge + K6 + scales sidecar)", "notebook": "k100_setup()", "kit": "K100_SETUP k100_setup()",
     "covered": K100_SETUP.get("ok") is True and K100_SETUP.get("sidecar_exists") is True},
    {"step": "sha gate", "notebook": "sha_gate()", "kit": "R1 sha_gate()", "covered": bool((R1.get("sha") or {}).get("ok"))},
    {"step": "nb4_ppack (Q-defer)", "notebook": "pseq.before_arm(ARMS['Q'])", "kit": "R3P PSEQ.before_arm(ARMS['Q'])", "covered": (R3P.get("before_arm_q") or [None])[0] is True},
    {"step": "dflash_seam apply (the NBDF-K100 sequence; inert on MTP)", "notebook": "sseq.apply()", "kit": "R3S SSEQ.apply()", "covered": R3S.get("apply_ok") is True},
    {"step": "arm plan (m1_arms / install / check_arms) + resolve_vocab", "notebook": "CELL 7 NB-M1 block", "kit": "check_arms on the notebook's ARMS",
     "covered": _ca is True and all((G.get("M1_VOCAB") or {}).get(k, {}).get("found") for k in _M1.VOCAB_TABLES)},
    {"step": "seam / ppack revert", "notebook": "sseq.finish() / pseq.finish()", "kit": "R3S / R3P revert (pristine)", "covered": R3S.get("pristine") is True}],
    "per_arm": _per,
    "gpu_only_not_covered": ["boot_gate (GPU empty)", "LoadMonitor / with_servers / readiness", "M1.readback (server log, worker env RPC, the REDUCED "
                             "DRAFT VOCAB / Draft lm_head lines, the indexer cache spec, the logged seqs / batched tokens / long prefill / spec)",
                             "k100_boot_gates (method map / static scales on GPU)", "the windows / oracle / TF / profiles / PROF"]}
NBM1_STEPS["all_covered"] = all(x["covered"] for x in NBM1_STEPS["global"]) and all(all(v is not False for v in d.values()) for d in _per.values())
REP["checks"]["NBM1"] = {"check_arms": _ca, "arms": list(G["ARMS"]), "argparse": RM1A, "kv": _kvm1, "vocab": RM1V, "env_change": _chg,
                         "steps": NBM1_STEPS}
K100_OK["nbm1_arm_plan_ok"] = (_ca is True and tuple(G["ARMS"]) == tuple(_M1.ARM_KEYS) == tuple(ARMS5)
                               and bool((G["NBDF"].get("nbm1") or {}).get("arms")))
K100_OK["nbm1_launch_path_all_arms_ok"] = (len(_pl) == len(ARMS5) and NBM1_STEPS["all_covered"] and R5.get("ok") is True
                                           and K100_OK.get("k100_notebook_launch_steps_all_ok") is True)
K100_OK["nbm1_help_flag_gate_real_ok"] = K100_OK.get("k100_help_flag_gate_real_ok") is True
K100_OK["nbm1_argparse_all_arms_ok"] = RM1A.get("ok") is True
K100_OK["nbm1_kv_probes_ok"] = all((_kvm1.get(x) or {}).get("ok") is True for x in ("K4", "IDX", "S24", "S28", "S32"))
K100_OK["nbm1_vocab_tables_ok"] = (all((RM1V.get(k) or {}).get("ok_all") is True and (RM1V.get(k) or {}).get("arm_env_ok") is True for k in _M1.VOCAB_TABLES)
                                   and (RM1V.get("lm_head") or {}).get("ok") is True)
save()
'''
R = sub(R, '''K100_OK["k100_notebook_launch_steps_all_ok"] = all(v is True for v in STEPS.values())
''', '''K100_OK["k100_notebook_launch_steps_all_ok"] = all(v is True for v in STEPS.values())
''' + PROBES + NBM1_ITEMS, "nbm1 items")
ns["KITS"]["real"]["body"] = R
if "--out-root" in sys.argv:                  # reproduction checks: the kit folder elsewhere (build() reads the global ROOT)
    ns["ROOT"] = Path(sys.argv[sys.argv.index("--out-root") + 1])
out, code = ns["build"]("real")

for p in FROZEN:
    assert sha(p.read_bytes()) == FROZEN_SHA[str(p)], f"FROZEN FILE CHANGED: {p}"
data = (out / f"q27-e25-{PFX}-kit-real.ipynb").read_bytes()
meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
assert M1.VOCAB_DS in meta["dataset_sources"] and meta["id"] == f"foysalemonshanto/q27-e25-{PFX}-kit-real", meta
print("NB-M1L real kit" if LIGHT else "NB-M1 real kit", out, len(data), sha(data), "| notebook", sha(NB.read_bytes())[:8], "| datasets", meta["dataset_sources"])
