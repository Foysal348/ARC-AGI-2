"""E25 NB-S1 real CPU kit (pinned Py3.12 image, no GPU, no internet, private, foysalemonshanto): the NBDF-K100 real kit,
derived IN MEMORY from build_e27_nbdf_cpu_kit.py (NBDF_VARIANT=k100: every K100 substitution kept -- the scales sidecar
step, VLLM_TARGET_DEVICE=cpu for the notebook's own --help, PSEQ.before_arm(Q) / SSEQ.check + PSEQ.before_arm per arm,
arm_boot's first try, launch_check at the K100 pin) and pointed at the NB-S1 notebook / facts / export. The NBDF files are
read, never written (frozen-file hash assert before and after).

The kit runs the NOTEBOOK's own functions: merge, merlin_install, apply_60490, k100_setup, sha_gate, PSEQ / SSEQ, then for
ALL 6 NB-S1 arms (B under key Q, G1, G2, EH, F8, DYN) tree_check -> arm_boot's knobs -> merlin_fn_server (K6, Merlin
build + host_phase, the --help flag gate, the NBDF launch check, VLLM_QSA_KV_SCALES) -> nbdf_lib.launch_check at the pin,
the env in a spawned child (the GDN keys included: the arm's expect), the argv fields against the arm's own spec /
captures (nbdf_lib.launch_check's want_sp: no kit-only expectation). NB-S1 changes to the derived template:
  - the arms = the NB-S1 plan; the captures item per NB-S1 arm (DYN: tier-aware coverage b x 5 for b <= 8, b x 4 for 9-24);
  - the KV probe (Merlin get_kv_cache_configs) gets an F8 run: fp8 drafter KV, SW block 2048, TRITON_ATTN; B / G1 / G2 / EH
    = K3_swblock1024 (EH changes no KV spec at the same pin), DYN = K4_swblock1024;
  - each plan records the arm's own env change (GDN / EMBED_HOST / SW block) from the launch env;
  - the verdict keys nbs1_arm_plan_ok, nbs1_launch_path_all_arms_ok, nbs1_help_flag_gate_real_ok (the build gate's three)
    + nbs1_kv_probes_ok, and the step-coverage table NBS1_STEPS (every notebook step from merge to the launch argv, per arm).
Run: python build_e25_nbs1_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbs1-kit-real__foysalemonshanto/"""
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
NB = HERE.parent / "Q27-XHIGH-E25_NBS1.ipynb"
TEMPLATE = E27B / "build_e27_nbdf_cpu_kit.py"


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


FROZEN = [E27B / n for n in ("nbdf_lib.py", "build_e27_nbdf.py", "build_e27_nbdf_k100.py", "build_e27_nbdf_cpu_kit.py",
                             "build_e27_nbdf_k100_cpu_kit.py", "dflash_seam.py", "nbdf_k100_cell7_body.py", "nbdf_cell7_body.py",
                             "nbdf_k100_code_export.py", "nbdf_k100_build_facts.json", "nbdf_build_facts.json")]
FROZEN += [E27B.parent / "Q27-XHIGH-E27_NBDF_K100.ipynb", E27B.parent / "Q27-XHIGH-E27_NBDF_DFLASH2.ipynb"]
FROZEN_SHA = {str(p): sha(p.read_bytes()) for p in FROZEN}
ARM_KEYS = ("Q", "G1", "G2", "EH", "F8", "DYN")


def sub(text, old, new, where, n=1):
    c = text.count(old)
    assert c == n, f"NB-S1 kit anchor {where}: {c} matches (want {n})"
    return text.replace(old, new)


# ---------------------------------------------------------------- the outer builder, pointed at NB-S1 (text, in memory)
src = TEMPLATE.read_text(encoding="utf-8")
src = sub(src, 'NB_PATH = HERE.parent / ("Q27-XHIGH-E27_NBDF_K100.ipynb" if K100 else "Q27-XHIGH-E27_NBDF_DFLASH2.ipynb")',
          f'NB_PATH = Path(r"{NB}")', "nb path")
src = sub(src, 'FACTS = json.loads((HERE / ("nbdf_k100_build_facts.json" if K100 else "nbdf_build_facts.json")).read_text(encoding="utf-8"))',
          f'FACTS = json.loads(Path(r"{HERE / "nbs1_build_facts.json"}").read_text(encoding="utf-8"))', "facts")
src = sub(src, 'EXPORT = (HERE / ("nbdf_k100_code_export.py" if K100 else "nbdf_code_export.py")).read_text(encoding="utf-8")',
          f'EXPORT = Path(r"{HERE / "nbs1_code_export.py"}").read_text(encoding="utf-8")', "export")
src = sub(src, 'KV_Q_PIN, KV_D_PIN, KV_Q_BLOCKS, KIT_ARMS = _cfg["q_pin"], _cfg["q_pin"], 480, tuple(_cfg["arms"])',
          f'KV_Q_PIN, KV_D_PIN, KV_Q_BLOCKS, KIT_ARMS = _cfg["q_pin"], _cfg["q_pin"], 480, {ARM_KEYS!r}   # E25 NB-S1', "kit arms")
src = sub(src, 'slug = f"q27-e27-nbdf-k100-kit-{name}" if K100 else f"q27-e27-nbdf-kit-{name}"', 'slug = f"q27-e25-nbs1-kit-{name}"', "slug")
src = sub(src, '"Q27 E27 NBDF K100 Kit Real" if K100 else "Q27 E27 NBDF Kit Real"', '"Q27 E25 NBS1 Kit Real"', "title")

src = sub(src, 'UNIT_FILES[_BODY_KEY] = (HERE / "nbdf_k100_cell7_body.py").read_text(encoding="utf-8")',
          f'UNIT_FILES[_BODY_KEY] = Path(r"{HERE / "nbs1_cell7_body.py"}").read_text(encoding="utf-8")   # E25 NB-S1 body', "units body")
os.environ["NBDF_VARIANT"] = "k100"
sys.dont_write_bytecode = True
ns = {"__file__": str(TEMPLATE), "__name__": "nbs1_kit_template"}
exec(compile(src, str(TEMPLATE) + " (NB-S1, in memory)", "exec"), ns)
assert ns["K100"] is True and ns["KIT_ARMS"] == ARM_KEYS and ns["NB_PATH"] == NB

# ---------------------------------------------------------------- the REAL template (after the K100 substitutions): NB-S1 edits
R = ns["KITS"]["real"]["body"]
# the KV probe: + the F8 run (fp8 drafter KV, SW block 2048, TRITON_ATTN)
R = sub(R, '''("K3_swblock1024", 3, dict(bf16, block_size=1024), "FLASH_ATTN")):''',
        '''("K3_swblock1024", 3, dict(bf16, block_size=1024), "FLASH_ATTN"),
                               ("K3_fp8_sw2048", 3, dict(fp8, block_size=2048), "TRITON_ATTN")):   # E25 NB-S1 F8''', "R9 F8 run")
# the KV probe's argv inputs: B's own argv (B = the D3s config under key Q)
R = sub(R, 'for k_ in ("D3s", "D4s") if (plans.get(k_) or {}).get("argv")', 'for k_ in ("Q", "G1") if (plans.get(k_) or {}).get("argv")', "kx argv", n=2)
# every plan records the arm's own env change from the FINAL launch env; the spawned child reads the arm's whole expect
R = sub(R, 'qsa_scales_env=s.env.get("VLLM_QSA_KV_SCALES"),',
        'qsa_scales_env=s.env.get("VLLM_QSA_KV_SCALES"), s1_env={k_: s.env.get(k_) for k_ in tuple(G["S1"].S1_ENV_KEYS) + ("DFLASH_DRAFT_EMBED_HOST", "DFLASH_DRAFT_SW_BLOCK")},',
        "s1 env")
R = sub(R, 'sp = venv_py(SPAWN_ENV, [",".join(G["N"].WATCHED)], name=',
        'sp = venv_py(SPAWN_ENV, [",".join(list(a["expect"]))], name=', "spawn keys = the arm's expect")
# the captures item: the NB-S1 arms (the notebook's own arm captures / spec; DYN tier-aware)
R = sub(R, '''    exp_caps = {"Q": ns_["capture_sizes_for"](K100_SEQS, 3), "D3s": [4 * b for b in range(1, K100_SEQS + 1)], "D4s": [5 * b for b in range(1, K100_SEQS + 1)]}''',
        '''    exp_caps = {k_: (sorted({5 * b for b in range(1, 9)} | {4 * b for b in range(9, K100_SEQS + 1)}) if k_ == "DYN"
                     else [4 * b for b in range(1, K100_SEQS + 1)]) for k_ in ARMS5}      # E25 NB-S1: B + one change each''', "exp caps")
R = sub(R, '''    for k in ("Q", "D3s", "D4s"):
        a = G["ARMS"].get(k) or {}''', '''    for k in ARMS5:
        a = G["ARMS"].get(k) or {}''', "caps loop")
R = sub(R, '''        unc = [b for b in range(1, K100_SEQS + 1) if not any(x >= b for x in g)]''',
        '''        unc = ([b for b in range(1, K100_SEQS + 1) if b * (5 if b <= 8 else 4) not in (a.get("captures") or [])] if k == "DYN"
               else [b for b in range(1, K100_SEQS + 1) if not any(x >= b for x in g)])''', "dyn coverage")
R = sub(R, '''        if sp.get("method") != ("mtp" if k == "Q" else "dflash") or sp.get("num_speculative_tokens") != {"Q": 3, "D3s": 3, "D4s": 4}[k]:''',
        '''        if sp.get("method") != "dflash" or sp.get("num_speculative_tokens") != (4 if k == "DYN" else 3):''', "caps spec K")
R = sub(R, '''        c[k]["ok"] = not unc and not bad and c[k]["lib_coverage"] is True''',
        '''        c[k]["ok"] = not unc and not bad and (c[k]["lib_coverage"] is True or k == "DYN")   # DYN: two tiers (checked above)''', "caps ok")
R = sub(R, '''    c["ok"] = all(c[k]["ok"] for k in ("Q", "D3s", "D4s")) and not badv''', '''    c["ok"] = all(c[k]["ok"] for k in ARMS5) and not badv''', "caps all")
# the NB-S1 verdict keys + the step-coverage table (after the K100 launch-steps check)
NBS1_ITEMS = r'''
# ---- E25 NB-S1: the arm plan, the launch path of all 6 arms, the help gate, the KV probes, the step-coverage table
_S1 = G["S1"]
try:
    _S1.check_arms(G["N"], G["T"], G["ARMS"])
    _ca = True
except Exception as e:
    _ca = f"{type(e).__name__}: {e}"[:300]
_runs = (R9.get("json") or {}).get("runs") or {}
_kvrun = {"Q": "K3_swblock1024", "G1": "K3_swblock1024", "G2": "K3_swblock1024", "EH": "K3_swblock1024", "F8": "K3_fp8_sw2048",
          "DYN": "K4_swblock1024"}
_kv = {k: {"run": r, "num_blocks": (_runs.get(r) or {}).get("num_blocks"), "error": (_runs.get(r) or {}).get("error"),
           "kv_group_count": (_runs.get(r) or {}).get("kv_group_count"),
           "ok": bool((_runs.get(r) or {}).get("num_blocks")) and "error" not in (_runs.get(r) or {})} for k, r in _kvrun.items()}
_chg = {}
for k in ARMS5:
    a_, e_ = G["ARMS"][k], (_pl.get(k) or {}).get("s1_env") or {}
    _chg[k] = {"launch_env": e_, "planned": {kk: a_["env"].get(kk) for kk in e_}, "change": a_.get("s1_change")}
    _chg[k]["as_planned"] = bool(e_) and e_ == _chg[k]["planned"]
_capc = (REP["checks"].get("K100_CAPTURES") or {})
_per = {}
for k in ARMS5:
    p_ = _pl.get(k) or {}
    _per[k] = {"sseq_check_pseq_before_arm (nbdf_main D loop)": True if k == "Q" else (p_.get("sseq_check", [None])[0] is True and p_.get("pseq_before_arm", [None])[0] is True),
               "tree_check": bool((p_.get("tree_check") or {}).get("ok")),
               "arm_boot knobs (NB1_EXTRA / NB1_ENV / NB1_KV at q_pin, ARM_OF, NB4T_BUILD_ENV)": p_.get("pin") == G["K100_CFG"]["q_pin"] and (p_.get("build_env") in ({}, None)),
               "K6 before launch": (p_.get("k6_before_launch") or {}).get("ok") is True,
               "Merlin build + host_phase": p_.get("host_phase") == "ok",
               "--help flag gate (L.help_check + KL.cli_flags_check)": ((p_.get("cli_flags") or {}).get("ok") is True),
               "NBDF launch check + nbdf_lib.launch_check at the pin": p_.get("ok") is True and p_.get("launch_check") == [],
               "VLLM_QSA_KV_SCALES": str(p_.get("qsa_scales_env") or "").endswith("qsa_kv_scales_262k.json"),
               "env in a spawned child (the arm's expect)": p_.get("spawn_ok") is True,
               "the arm's own env change in the launch env": _chg[k]["as_planned"],
               "argv spec / captures / seqs / KV bytes (R5 + captures item)": not (R5.get("field_check") or {}).get(k) and (_capc.get(k) or {}).get("ok") is True,
               "KV probe (get_kv_cache_configs)": _kv[k]["ok"]}
NBS1_STEPS = {"global": [
    {"step": "merge (CELL 3)", "notebook": "R['sections']['A0']['merge']", "kit": "R1 merge", "covered": bool((R1.get("merge") or {}).get("ok"))},
    {"step": "install", "notebook": "merlin_install()", "kit": "R2 merlin_install()", "covered": R2.get("ok") is True},
    {"step": "#60490 + worker extension", "notebook": "apply_60490()", "kit": "R3 apply_60490()", "covered": R3.get("ok") is True},
    {"step": "k100_setup (overlay merge + K6 + scales sidecar)", "notebook": "k100_setup()", "kit": "K100_SETUP k100_setup()",
     "covered": K100_SETUP.get("ok") is True and K100_SETUP.get("sidecar_exists") is True},
    {"step": "sha gate", "notebook": "sha_gate()", "kit": "R1 sha_gate()", "covered": bool((R1.get("sha") or {}).get("ok"))},
    {"step": "nb4_ppack (Q-defer)", "notebook": "pseq.before_arm(ARMS['Q'])", "kit": "R3P PSEQ.before_arm(ARMS['Q'])", "covered": (R3P.get("before_arm_q") or [None])[0] is True},
    {"step": "dflash_seam apply + ring", "notebook": "sseq.apply() + N.ring_status", "kit": "R3S SSEQ.apply()", "covered": R3S.get("apply_ok") is True},
    {"step": "B base gate (drafter / kit gate / seam)", "notebook": "S1.base_gate", "kit": "R1 drafter pins + R3S seam (the kit gate = this kit)",
     "covered": R1.get("drafter_ok") is True and R3S.get("apply_ok") is True},
    {"step": "arm plan (s1_arms / install / check_arms)", "notebook": "CELL 7 NB-S1 block", "kit": "check_arms on the notebook's ARMS", "covered": _ca is True},
    {"step": "seam / ppack revert", "notebook": "sseq.finish() / pseq.finish()", "kit": "R3S / R3P revert (pristine)", "covered": R3S.get("pristine") is True}],
    "per_arm": _per,
    "gpu_only_not_covered": ["boot_gate (GPU empty)", "LoadMonitor / with_servers / readiness", "S1.readback (server log, worker env RPC, "
                             "dflash_readback: GDN in the workers, EMBED_HOST, TRITON_ATTN + 2048 block, the DYN fallback line)",
                             "k100_boot_gates (method map / static scales on GPU)", "the windows / oracle / TF / profile"]}
NBS1_STEPS["all_covered"] = all(x["covered"] for x in NBS1_STEPS["global"]) and all(all(v.values()) for v in _per.values())
REP["checks"]["NBS1"] = {"check_arms": _ca, "arms": list(G["ARMS"]), "kv": _kv, "env_change": _chg, "steps": NBS1_STEPS}
K100_OK["nbs1_arm_plan_ok"] = (_ca is True and tuple(G["ARMS"]) == tuple(_S1.ARM_KEYS) == tuple(ARMS5)
                               and bool((G["NBDF"].get("nbs1") or {}).get("arms")))
K100_OK["nbs1_launch_path_all_arms_ok"] = (len(_pl) == len(ARMS5) and NBS1_STEPS["all_covered"] and R5.get("ok") is True
                                           and K100_OK.get("k100_notebook_launch_steps_all_ok") is True)
K100_OK["nbs1_help_flag_gate_real_ok"] = K100_OK.get("k100_help_flag_gate_real_ok") is True
K100_OK["nbs1_kv_probes_ok"] = all(v["ok"] for v in _kv.values())
save()
'''
R = sub(R, '''K100_OK["k100_notebook_launch_steps_all_ok"] = all(v is True for v in STEPS.values())
''', '''K100_OK["k100_notebook_launch_steps_all_ok"] = all(v is True for v in STEPS.values())
''' + NBS1_ITEMS, "nbs1 items")
ns["KITS"]["real"]["body"] = R
out, code = ns["build"]("real")

for p in FROZEN:
    assert sha(p.read_bytes()) == FROZEN_SHA[str(p)], f"FROZEN FILE CHANGED: {p}"
data = (out / "q27-e25-nbs1-kit-real.ipynb").read_bytes()
print("NB-S1 real kit", out, len(data), sha(data), "| notebook", sha(NB.read_bytes())[:8])
