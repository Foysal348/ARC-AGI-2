"""E25 NB4T LEVERS (2026-11-01): a GPU measurement notebook for the next speed levers on top of the Q-defer winner of NB4S
(notebook fa9f35ca, GPU run v1: QD 309.1 tok/s at 8k@8 = +21.1 %, 297.1 at 40k@8 = +19.8 %, oracle EXACT; after Q-defer
stage-1 compute (33.0 ms) binds, stage 0 is 21.6 ms).

  CELL 0  E18 R2's cell 1 VERBATIM (Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb, sha 116d2d41; = NB6's CELL 0): mode, clock,
          the identity fallback submission.json of every test task. A competition rerun writes only that file.
  CELL 1-6  the Kaggle-run NB4S notebook's cells (sha fa9f35ca) verbatim, with these marked "E25 NB4T" edits only:
          CELL 1  title, budget 120 min, OUT /kaggle/working/nb4t, R["what"], nb4t_results.json, the rerun branch (no [[0]]
                  placeholder: CELL 0's identity file stays) and write_fallback_submission() = CELL 0's identity writer;
          CELL 2  merlin_fn_server: Merlin launcher knobs of the launch (NB4T_BUILD_ENV[name], I8's DRAFT_HEAD=int8) go into
                  build()'s input env (launch.py:319 reads DRAFT_HEAD there; the NB1 env export after build() is too late),
                  and the recorded env keys include DRAFT_HEAD.
  CELL 7  NEW: the arms Q / I8 / P26 / P27 / C12 (nb4t_lib.arm_plan: Q + exactly one change each), boot / read-backs /
          oracle / windows / profile / analysis, the plan (Q > I8 > P26 > P27 > C12, an arm starts only with its full
          minimum left; P27 skipped after a P26 OOM), the parity-aware verdict in a finally block.
  CELL 8  NEW: report + the identity fallback submission (commit).
  nb4t_lib.py (new), nb4_timeline.py (agent C, = NB4S) and nb4_ppack.py (rev 2, = NB4S) are embedded zlib + base64 with
  their sha256, checked at load.

Nothing outside the NB4T files is written (NB4S, its builder and every other notebook are checked byte for byte).
Run: python build_e25_nb4t.py -> ../Q27-XHIGH-E25_NB4T_LEVERS.ipynb + nb4t_code_export.py + nb4t_build_facts.json + the GPU
push folder (kernel-metadata.json + the notebook; nothing is pushed).
"""
import ast
import base64
import builtins
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
EXP = NBDIR.parent.parent
NB4S_NB = NBDIR / "Q27-XHIGH-E25_NB4S_STALL.ipynb"
NB4S_SHA = "fa9f35ca43ec0c8108327a840a9f6c28d2891e1313c131b3ec56b379048f964a"       # NB4S rev 3, GPU run v1 2026-11-01
E18R2_NB = EXP / "Experiment 18 - S1-60K on vLLM 0.32 (planned)" / "Experiment Notebooks" / "Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb"
E18R2_SHA = "116d2d4101a53eb4ff833e9c97e5a779f0f2bae5735129a4663ad1b42ad79a1d"
NB6_NB = NBDIR / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb"
LIB = HERE / "nb4t_lib.py"
TIMELINE = HERE / "nb4_timeline.py"
PPACK = HERE / "nb4_ppack.py"
OUT_NB = NBDIR / "Q27-XHIGH-E25_NB4T_LEVERS.ipynb"
EXPORT = HERE / "nb4t_code_export.py"
FACTS = HERE / "nb4t_build_facts.json"
SLUG = "q27-xhigh-e25-nb4t-levers"
PUSH = Path(r"D:/ARC AGI 2/_scratch/push_e25") / f"{SLUG}__foysalemonshanto"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODELS = ([f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
          ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
           "foysalemonshanto/e23-merlin-env/Other/v1/1"])
PUSHED = set()          # shas of NB4T versions that ran on Kaggle (never overwrite one; rebuilds then go to *_REBUILD)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")

# Merlin 3.0.0 (ad42e37) source facts behind every arm (D:/ARC AGI 2/_scratch/e23/p2/merlin-src), recorded in the notebook
FLAGS = {
    "Q": {"change": "nb4_ppack rev 2 patch + NB4_PPDEFER=1 (NB4S QD)",
          "language_model_only": "already in Q: NB4S B = NB1 R2 B3 carries --language-model-only (caps.language_model_only true "
                                 "in the NB4S run), so P26 / P27 change the partition only"},
    "I8": {"knob": "env DRAFT_HEAD=int8, an input of Merlin's build()",
           "merlin": ["tools/merlin/launch.py:319-322 (DRAFT_HEAD read: bf16 | int8 | int4; draft_spec = ',\"draft_lm_head_quant\":\"int8\"')",
                      "tools/merlin/launch.py:441 (--speculative-config {method mtp, K, probabilistic + draft_spec})",
                      "tools/merlin/registry.py:405 (DRAFT_HEAD_DEFAULT = 'bf16')"],
           "engine": ["vllm/config/speculative.py:610 (draft_lm_head_quant: 'int4' | 'int8' | None)",
                      "vllm/v1/worker/gpu/spec_decode/speculator.py:345 + 547-603 (_maybe_quantize_draft_lm_head: last PP stage "
                      "only; refuses local argmax / synthetic / adaptive verification)",
                      "vllm/model_executor/layers/draft_lm_head_quant.py:75-125 (QuantizedDraftLMHead; logs 'Draft lm_head: "
                      "INT8 weight-only copy of the NxK shard (M MiB; the target keeps its torch.bfloat16 head)')",
                      "vllm/models/qwen4_exp/nvidia/mtp.py:497 (supports_draft_lm_head_quant = True)"],
           "repack_preflight": ("not needed: the INT8 copy is packed on the CPU by marlin_quantize "
                                "(model_executor/layers/quantization/utils/marlin_utils_test.py:127-161: gptq_quantize_weights + "
                                "marlin_weights + marlin_permute_scales, pure torch) and run by apply_gptq_marlin_linear; "
                                "draft_lm_head_quant.py never calls gptq_marlin_repack. NB2 SMOKE ran K2 + INT8 on GPU (head "
                                "read-back PASS, E25_GPU_wave1_NB3_NB2SMOKE). The arm's read-back is the proof it is used."),
           "readback": "argv spec draft_lm_head_quant=int8 (launch check) + non-default args spec + 'Draft lm_head: INT8' lines"},
    "P26": {"knob": "env VLLM_PP_LAYER_PARTITION=26,22 exported after build() (NB1_ENV)",
            "merlin": ["tools/merlin/launch.py:373-374 (the profile layout sets 25,23; flash-next has no PP_PARTITION knob)",
                       "vllm/envs.py:895", "vllm/distributed/utils.py:126-170 (get_pp_indices; the sum must be 48)",
                       "vllm/model_executor/models/config.py:1030-1046 (every PLE layer must live on PP rank 0)",
                       "vllm/compilation/caching.py:572-576 (the partition is a compile-cache factor: a recompile boot)"],
            "readback": "worker env (nb4s_env RPC) + 'pipeline layers a-b' per stage + pool blocks, per-rank free, timeline"},
    "P27": {"knob": "env VLLM_PP_LAYER_PARTITION=27,21", "skip_rule": "skipped when P26 OOMs (any launch try, or BODY_OOM)"},
    "C12": {"knob": "trailing --max-num-seqs 12 (Merlin's profile passes 8 at launch.py:453; extra args come last, "
                    "launch.py:111, so the last occurrence wins)",
            "readback": "launch check + non-default args max_num_seqs 12 + running == 12 in every window"},
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def b64z(text):
    return base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: x{text.count(old)}"
    return text.replace(old, new)


def other_files_shas():
    """sha of every notebook in Experiment Notebooks except NB4T + the NB4S builder / lib files (byte-identity check)."""
    d = {p.name: sha(p.read_bytes()) for p in sorted(NBDIR.glob("*.ipynb")) if p.name != OUT_NB.name}
    for n in ("build_e25_nb4s.py", "nb4s_lib.py", "nb4s_unit_tests.py", "nb4s_code_export.py", "nb4_ppack.py", "nb4_timeline.py"):
        d[n] = sha((HERE / n).read_bytes())
    return d


SHAS_BEFORE = other_files_shas()

# ---------------------------------------------------------------- inputs: the GPU-run NB4S notebook + E18 R2's CELL 0
_nb4s_bytes = NB4S_NB.read_bytes()
assert sha(_nb4s_bytes) == NB4S_SHA, "NB4S notebook is not the GPU-run fa9f35ca"
_nb4s = json.loads(_nb4s_bytes.decode("utf-8"))
NB4S_CODE = ["".join(c["source"]) for c in _nb4s["cells"] if c["cell_type"] == "code"]
assert len(NB4S_CODE) == 8 and NB4S_CODE[0].startswith("# ====") and all(c.startswith("if not RERUN:\n") for c in NB4S_CODE[1:])
_e18 = E18R2_NB.read_bytes()
assert sha(_e18) == E18R2_SHA, "E18 R2 notebook changed"
CELL0 = "".join(json.loads(_e18.decode("utf-8"))["cells"][1]["source"]).replace("\r\n", "\n")
assert CELL0.startswith("# ==================================================================== CELL 0"), CELL0[:80]
_nb6_code = ["".join(c["source"]) for c in json.loads(NB6_NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
assert _nb6_code[0] == CELL0, "NB6's CELL 0 != E18 R2 cell 1 (the user's rule: copy NB6's CELL 0 = E18 R2's cell 1)"
E18_CELL0_SHA = sha(CELL0.encode("utf-8"))

# ---------------------------------------------------------------- embedded modules (LF, verbatim, sha-checked at run time)
LIB_SRC = LIB.read_bytes().decode("utf-8")
TL_SRC = TIMELINE.read_bytes().decode("utf-8")
PP_SRC = PPACK.read_bytes().decode("utf-8")
for _n, _s in (("nb4t_lib.py", LIB_SRC), ("nb4_timeline.py", TL_SRC), ("nb4_ppack.py", PP_SRC)):
    assert "\r" not in _s, f"{_n} must be LF"
    compile(_s, _n, "exec")
LIB_SHA, TL_SHA, PP_SHA = sha(LIB_SRC.encode()), sha(TL_SRC.encode()), sha(PP_SRC.encode())
assert f'TL_SHA = "{TL_SHA}"' in NB4S_CODE[6] and f'PPACK_SHA = "{PP_SHA}"' in NB4S_CODE[6], \
    "nb4_timeline.py / nb4_ppack.py differ from the modules the NB4S GPU run embedded"
_pp_defs = {n.name for n in ast.parse(PP_SRC).body if isinstance(n, ast.FunctionDef)}
assert {"apply_ppack_patch", "revert_ppack_patch", "verify_ppack_patch", "server_env", "readback_label", "parse_log"} <= _pp_defs
_tl_defs = {n.name for n in ast.parse(TL_SRC).body if isinstance(n, ast.FunctionDef)}
assert {"analyze", "strip_private", "format_report", "load_traces"} <= _tl_defs
assert "nb4s_env" in NB4S_CODE[5] or any("def nb4s_env" in c for c in NB4S_CODE), "NB4S worker extension (nb4s_env) missing"

# ---------------------------------------------------------------- CELL 1 / CELL 2: marked NB4T edits on NB4S's cells
CELL1 = NB4S_CODE[0]
CELL1 = once(CELL1, "(RC2 verbatim + E25 NB1 / NB4S marked edits)", "(RC2 verbatim + E25 NB1 / NB4S / NB4T marked edits)", "c1 title")
CELL1 = once(CELL1, 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 95 * 60)      # E25 NB4S: hard stop 95 min (B > S > QP > QD)',
             'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 120 * 60)     # E25 NB4T: hard stop 120 min (Q > I8 > P26 > P27 > C12)', "c1 budget")
CELL1 = once(CELL1, 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb4s"))   # E25 NB4S',
             'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb4t"))   # E25 NB4T', "c1 out")
CELL1 = once(CELL1, 'R = {"what": "E25 NB4S STALL: CUDA_DEVICE_MAX_CONNECTIONS=32 vs the stage-0 in-graph stall (Flash-Next on Merlin)",   # E25 NB4S',
             'R = {"what": "E25 NB4T LEVERS: INT8 draft head / PP partition 26,22 + 27,21 / 12 streams on the Q-defer base (Flash-Next on Merlin)",   # E25 NB4T',
             "c1 what")
CELL1 = once(CELL1, '        tmp = OUT / "nb4s_results.json.tmp"                      # E25 NB4S\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nb4s_results.json")                   # E25 NB4S',
             '        tmp = OUT / "nb4t_results.json.tmp"                      # E25 NB4T\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nb4t_results.json")                   # E25 NB4T', "c1 save")
CELL1 = once(CELL1, '''def write_fallback_submission():
    tp = COMP / "arc-agi_test_challenges.json"
    if tp.is_file():
        tasks = json.load(open(tp))
        sub = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]] for k, v in tasks.items()}
        Path(os.environ.get("RC2_SUBMISSION", "/kaggle/working/submission.json")).write_text(json.dumps(sub))


if RERUN:
    write_fallback_submission()
    print("E25 NB4S stall diagnostic: competition rerun -> fallback submission.json only")
''', '''def write_fallback_submission():
    """E25 NB4T: the proven E18 R2 format (CELL 0's writer): the identity submission of every test task (attempt_1 =
    attempt_2 = the test input), written atomically to CELL 0's SUBMISSION_PATH; never a [[0]] placeholder."""
    write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))


if RERUN:                                                         # E25 NB4T: CELL 0 already wrote the identity file
    print("E25 NB4T speed levers: competition rerun -> CELL 0's identity fallback submission.json only (E18 R2 format)")
''', "c1 fallback + rerun")
CELL2 = NB4S_CODE[1]
CELL2 = once(CELL2, '''        plan = eng["ML"].build("flash-next", PROFILE, str(CKPT), base, extra=extra, kv=kv, checkpoint="flash-next", status="untested")
''', '''        base.update((globals().get("NB4T_BUILD_ENV") or {}).get(name) or {})   # E25 NB4T: launcher knobs (DRAFT_HEAD) into build()
        plan = eng["ML"].build("flash-next", PROFILE, str(CKPT), base, extra=extra, kv=kv, checkpoint="flash-next", status="untested")
''', "c2 build env")
CELL2 = once(CELL2, '"env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "CUDA_DEVICE_MAX_CONNECTIONS"))}}   # E25 NB4S',
             '"env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "CUDA_DEVICE_MAX_CONNECTIONS", "DRAFT_HEAD"))}}   # E25 NB4S / NB4T',
             "c2 env keys")
CELL3_6 = NB4S_CODE[2:6]                                    # NB4S CELL 3-6 verbatim (already wrapped in `if not RERUN:`)

# ---------------------------------------------------------------- CELL 7: the arms, the boot, the analysis, the plan (NEW code)
CELL7 = r'''# ==================================================================== CELL 7 — E25 NB4T: the arms Q / I8 / P26 / P27 / C12, the plan (NEW code)
import importlib, base64, zlib
NB4T_LIB_SHA = "__LIBSHA__"                 # nb4t_lib.py (new), nb4_timeline.py (agent C) and nb4_ppack.py (rev 2) = NB4S's:
NB4T_LIB_SRC = zlib.decompress(base64.b64decode("__LIBB64__")).decode("utf-8")   # verbatim, zlib + base64, sha-checked
TL_SHA = "__TLSHA__"
TL_SRC = zlib.decompress(base64.b64decode("__TLB64__")).decode("utf-8")
PPACK_SHA = "__PPSHA__"
PPACK_SRC = zlib.decompress(base64.b64decode("__PPB64__")).decode("utf-8")
for _src, _sha, _n in ((NB4T_LIB_SRC, NB4T_LIB_SHA, "nb4t_lib"), (TL_SRC, TL_SHA, "nb4_timeline"), (PPACK_SRC, PPACK_SHA, "nb4_ppack")):
    if hashlib.sha256(_src.encode("utf-8")).hexdigest() != _sha:
        raise RuntimeError(f"EMBED_SHA_MISMATCH: {_n}")
    (TMP / f"{_n}.py").write_text(_src, encoding="utf-8")
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))
N = importlib.import_module("nb4t_lib")
TL = importlib.import_module("nb4_timeline")
P = importlib.import_module("nb4_ppack")
NB4T = R.setdefault("nb4t", {})
NB4T.update(question="on top of Q (NB4S's B + Q-defer), which next lever speeds decode: the INT8 draft LM head at K3 (I8), the PP "
                     "layer partition 26,22 / 27,21 (P26 / P27), and what do 12 streams give (C12, measure only)?",
            lib=N.LIB_VERSION, lib_sha256=NB4T_LIB_SHA, timeline_sha256=TL_SHA, ppack_sha256=PPACK_SHA,
            ppack_file_shas={"pp_pre": P.PP_PRE, "pp_post": P.PP_POST, "mr_pre": P.MR_PRE, "mr_post": P.MR_POST,
                             "gw_pre": P.GW_PRE, "gw_post": P.GW_POST},
            nb4s_notebook_sha256="__NB4SSHA__", e18_r2_cell0_sha256="__E18C0SHA__", merlin_flags=json.loads(r"""__FLAGS__"""))
B3_PIN = KV_FP8 + (1 << 30)                 # NB1 R2 B3 = NB4S B: 3,553,580,524 B (FP8 KV, 215 blocks in the NB4S run)
if B3_PIN != 3553580524:
    raise RuntimeError(f"B3_PIN {B3_PIN} != 3553580524")
Q_EXTRA = EXT_FLAG + ["--language-model-only", "--profiler-config", PROFILER_CFG]   # = NB4S B_EXTRA (NB1 R2 B3)
Q_BASE_ENV = dict(DIAG_ENV)                                                      # = NB4S B_ENV
ARMS = N.arm_plan(Q_EXTRA, Q_BASE_ENV, B3_PIN, q_env=P.server_env("Q-defer"))
N.check_arm_plan(ARMS)                      # never-do list + "each arm = Q + exactly one change" (raises: no boot)
NB4T["arms"] = {k: {x: a[x] for x in ("name", "port", "env", "build_env", "change", "expect", "expect_cfg", "extra", "kv_bytes",
                                      "needs_patch", "streams", "windows", "est_min", "cap_min", "title", "lever", "measure_only")}
                for k, a in ARMS.items()}
NB4T["order"] = list(N.ORDER)
NB4T["oom_policy"] = ("NB4S's one policy for every arm: a launch OOM (OOM line anywhere in the launch's full log, no device fault) -> "
                      "one retry 0.25 GiB lower on a fresh port and launch name; every later arm starts at Q's final pin; a body "
                      "OOM never retries (BODY_OOM_<item>); a P26 OOM (any try or BODY_OOM) skips P27")
NB4T["never_do"] = {"env": sorted(N.NEVER_DO_ENV), "argv": list(N.NEVER_DO_ARGS), "argv_values": N.NEVER_DO_ARG_VALUES}
NB4T["budget"] = {"hard_stop_min": BUDGET / 60, "start_reserve_sec": N.START_RESERVE_SEC, "body_reserve_sec": N.BODY_RESERVE_SEC,
                  "priority": "Q > I8 > P26 > P27 > C12 (an arm starts only with est_min x 60 + 300 s left)"}
ARM_OF = {}                                 # launch name (retries included) -> arm
NB4T_BUILD_ENV = {}                         # launch name -> Merlin build() knobs (CELL 2's E25 NB4T edit reads it)
PIN = {"kv_bytes": B3_PIN, "source": "NB1 R2 B3 = NB4S B"}
NB4T_READY_SEC = 20 * 60                    # readiness gate per boot (NB4S: 630 s for the first, compiling boot; 231-249 s later)
CUR = {"item": None, "kv_bytes": None}
_HIST = {}


def _tree_py():
    eng = merlin_install()
    return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"


def _ppack_apply():
    tree, py = _tree_py()
    return P.apply_ppack_patch(tree, py)


def _ppack_revert():
    tree, py = _tree_py()
    rec = P.revert_ppack_patch(tree, py)
    v = P.verify_ppack_patch(tree)              # the pre-shas, verified again after the revert
    if v["state"] != "not_applied":
        raise RuntimeError("REVERT_NOT_PRISTINE: " + json.dumps(v)[:400])
    rec["verify_after"] = v
    return rec


PSEQ = N.PatchSeq(_ppack_apply, _ppack_revert)


def _prom_nb4t(port):
    """e23_lab.prom (same dict, one HTTP read) + the iteration-token histogram buckets with their read time (NB4S)."""
    txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
    agg, buckets = N.prom_parse(txt)
    h = _HIST.setdefault(port, [])
    h.append((time.perf_counter(), buckets))
    del h[:-600]
    return agg


L.prom = _prom_nb4t


def _hist_at(port, t):
    """The latest histogram snapshot read at or before perf_counter time t (steady() reads m0 then sets t0)."""
    best = None
    for ts, b in _HIST.get(port, []):
        if ts <= t + 0.01:
            best = (ts, b)
    return best if best and t - best[0] < 1.0 else None


def ensure_prompts(n):
    """C12: n distinct prompt variants per measured band (L.steady / profiled_load take prompts[i % len]); variants 0-7 stay
    NB4S's, so the 8-stream windows of every other arm see exactly the NB4S prompts."""
    for b in ("8k", "40k"):
        while len(PROMPTS[b]) < n:
            PROMPTS[b].append(prompt(BANDS[b], len(PROMPTS[b])))
    return {b: len(PROMPTS[b]) for b in ("8k", "40k")}


_merlin_fn_server_nb1 = merlin_fn_server       # RC2 CELL 2 + NB1's per-boot hooks + NB4T's build-knob edit


def merlin_fn_server(name, kv, port, kv_bytes):
    """E25 NB4T: NB1's Merlin plan, then (fail closed: no server is started) nb4t_lib.launch_check on the FULL final env /
    argv of THIS launch: never-do items, every watched env key (NB4_PPDEFER=1, the partition), the effective
    --speculative-config draft head, the last --max-num-seqs, --language-model-only, the pin."""
    s = _merlin_fn_server_nb1(name, kv, port, kv_bytes)
    arm = ARM_OF.get(name)
    popped = {}
    for w, want in ((arm or {}).get("expect") or {}).items():
        if want is None and w in s.env:              # inherited from the image env, not planned: removed (recorded)
            popped[w] = s.env.pop(w)
    bad = N.launch_check(arm, s.env, s.argv, kv_bytes) if arm else [f"launch {name} has no arm"]
    NB4T.setdefault("launch", {})[name] = {
        "arm": (arm or {}).get("key"), "port": port, "kv_bytes": kv_bytes, "never_do_and_plan": bad,
        "build_env": NB4T_BUILD_ENV.get(name), "spec": N.spec_from_argv(s.argv), "max_num_seqs": N.last_value(s.argv, "--max-num-seqs"),
        "env": {k: s.env[k] for k in sorted(s.env) if k.startswith(("CUDA_", "NCCL_", "NB4_", "VLLM_SERVER_DEV", "VLLM_USE_V2",
                                                                    "VLLM_PP_LAYER_PARTITION", "DRAFT_HEAD", "P2P"))},
        "argv_tail": s.argv[-14:], "help": HELP.get(name), "popped_unplanned_env": popped}
    save()
    if bad:
        raise RuntimeError("NB4T_LAUNCH_CHECK: " + "; ".join(bad))
    return s


def free_min(port):
    """Free MiB per card = min(nvidia-smi, the worker's own torch free memory of the same rank); both recorded."""
    smi = L.gpu_mem() or []
    w = rpc_cache_info(port)
    wf = w.get("free_mib_per_rank") if isinstance(w, dict) else None
    sf = [t_ - u for _, u, t_ in smi] if smi else None
    mins = None
    if sf and wf and len(sf) == len(wf):
        mins = [min(a, b) if isinstance(b, (int, float)) else a for a, b in zip(sf, wf)]
    return {"nvidia_smi": sf, "worker": wf, "min": mins or sf or wf}


def proc_environ():
    """Side record: the watched keys in /proc/<pid>/environ of every engine process (server, workers, PLE)."""
    blobs = {}
    for pid, _ in (engine_procs() or [])[:24]:
        try:
            blobs[pid] = Path(f"/proc/{pid}/environ").read_bytes()
        except Exception:
            blobs[pid] = None
    return blobs


def _live(o, key, val):
    """Every item lands in the record and in the live step record before save() (a hard kill keeps it)."""
    o[key] = val
    rec = LIVE.get("rec")
    if rec is not None:
        rec.setdefault("live", {})[key] = val
    save()
    return val


def q_readback(s, arm):
    """The 4 'NB4_PPQ init' lines (every rank must agree) + the Q-defer flush ratios, via nb4_ppack.readback_label on the
    launch's whole log (every NB4T arm runs Q-defer)."""
    try:
        lab, det = P.readback_label(s.log() or "", arm["q_label"])
    except Exception as e:
        lab, det = "NOT_INTEGRATED", {"error": f"{type(e).__name__}: {e}"[:300]}
    return {"label": lab, "detail": det}


def hist_window(port, s, band, n):
    """NB1's window() (RC2 levels / lab steady; running == level, prefill done, per-position acceptance, tick / cycle) + the
    window's micro-batch split from the iteration-token histogram at steady()'s own t0 / t1 (recorded, not gating)."""
    r = window(port, s, band, n)
    h0, h1 = (_hist_at(port, r["t0"]) if r.get("t0") else None), (_hist_at(port, r["t1"]) if r.get("t1") else None)
    d = N.hist_delta(h0[1], h1[1]) if h0 and h1 else None
    r["parity"] = N.parity_from_hist(d, n, PP, 3)
    return r


def profile_window(port, arm):
    """One profiled 8k window at the arm's stream count: lab profiled_load (settled decode, then the bounded POST
    /start_profile {"delay_iterations": 20, "max_iterations": 40}, 6 s, /stop_profile); the traces are kept per arm."""
    n = arm["streams"]
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    rec = L.profiled_load(port, PROMPTS["8k"], n, PROF_DIR, secs=6.0, max_new=MAX_NEW["8k"], spec=True, prefill_tok_s=PREFILL["tok_s"])
    rec["profile_request"] = {"delay_iterations": 20, "max_iterations": 40, "secs": 6.0, "level": n, "band": "8k"}
    if "trace" in rec:
        rec["lab_trace_summary"] = rec.pop("trace")
        rec["lab_trace_note"] = "e23_lab rule: its PIECEWISE verdict is a parser artefact here (graphs research §0.1); nb4_timeline decides"
    g = L.gpu_mem()
    if g:
        rec["gpu_mem_peak_mib"] = [x[1] for x in g]
    kept = TRACES / f"profile_{arm['key'].lower()}"
    shutil.rmtree(kept, ignore_errors=True)
    kept.mkdir(parents=True, exist_ok=True)
    rec["trace_files"] = []
    for f in sorted(PROF_DIR.rglob("*.pt.trace.json*")):
        if f.stat().st_size < 120e6:
            shutil.copy2(f, kept / f.name)
            rec["trace_files"].append([f.name, f.stat().st_size])
    rec["trace_dir"] = str(kept)
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    return rec


def win_need(wk, n):
    """Seconds a window needs (NB4S's WIN_NEED at n streams)."""
    if wk.startswith("profile"):
        return 360 + 10 * max(0, n - 8)
    return WIN + 120 + n * BANDS[wk.split("@")[0]] / max(500.0, PREFILL["tok_s"])


RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P,DRAFT_HEAD"


def arm_body(arm, ref, store):
    """Every arm, the same items in the same order at its stream count; every item gated by due() on the boot's deadline."""
    def body(ss):
        s = ss[0]
        p = s.port
        n = arm["streams"]
        o = {"arm": arm["key"], "items_skipped": [], "streams": n}
        CUR["item"] = "facts"
        o = boot_facts(s, dict(o, census=kv_census(s), probes=L.chat_probes(p), sampled=sampled_check(s)), s.name)
        LIVE["out"] = o
        CUR["item"] = "readback"
        _live(o, "env_readback_raw", rpc(p, "nb4s_env", RPC_KEYS))
        _live(o, "env_readback", N.parse_env_readback(o["env_readback_raw"], arm["expect"]))
        _live(o, "proc_environ", N.parse_proc_environ(proc_environ(), arm["expect"]))
        _live(o, "config_readback", N.config_readback(arm, s.log() or "", CUR["kv_bytes"]))
        _live(o, "q_readback", q_readback(s, arm))
        _live(o, "free_after_ready_min", free_min(p))
        _live(o, "capacity_after_ready", N.capacity_facts(dict(o, pin=CUR["kv_bytes"])))
        if short_gate_failed(o):
            o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                     quality_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
            return o
        if o["env_readback"].get("definitive"):
            o.update(status="NOT_ENGAGED", early_stop="the workers do not hold the planned env: " + str(o["env_readback"].get("why")))
            return o                          # definitive: the arm's change is not in effect, nothing to measure
        if o["config_readback"].get("definitive"):
            o.update(status="NOT_ENGAGED", early_stop="the server log contradicts the plan: " + str(o["config_readback"].get("why")))
            return o
        if (o["q_readback"] or {}).get("label") == "NOT_INTEGRATED":
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="nb4_ppack init lines: " + json.dumps(o["q_readback"])[:300])
            return o
        if due(300):                          # every arm (the 40k probe prompt is variant 7 of the 40k window: same cache state)
            CUR["item"] = "prefill_probe"
            _live(o, "prefill", prefill_probe(p))
        if not due(240):
            o["items_skipped"].append("oracle")
            o["status"] = "SKIPPED_BOOT_BUDGET_ORACLE"
            return o
        CUR["item"] = "oracle"
        _live(o, "oracle", oracle(p, key=store, refs=(ref,)))
        o["oracle_vs"] = (o["oracle"].get("vs_" + ref) or {}).get("status")
        o["quality_gate"] = quality_gate(o, ref)
        if o["quality_gate"] != "PASS":
            o["load_tests_skipped"] = f"oracle gate vs {ref}: {o['quality_gate']}"
            o["status"] = "NO_GO_QUALITY" if o["quality_gate"].startswith("FAIL") else "INCONCLUSIVE_QUALITY"
            o["census_end"] = kv_census(s)
            return o
        _live(o, "q_readback", q_readback(s, arm))         # the counters moved by now (probes + sampled + oracle decode)
        if (o["q_readback"] or {}).get("label") not in ("ENGAGED", "ENGAGED_FALLBACK"):   # NB4S: a FALLBACK is not final here
            o.update(status=o["q_readback"]["label"], not_integrated=o["q_readback"]["label"] == "NOT_INTEGRATED",
                     early_stop="nb4_ppack read-back: " + json.dumps(o["q_readback"])[:300])
            return o
        if n > len(PROMPTS["8k"]) or n > len(PROMPTS["40k"]):
            o["prompts"] = ensure_prompts(n)
        if due(150):
            CUR["item"] = "warmup"
            _live(o, "warmup", warmup(p, n))
        else:
            o["items_skipped"].append("warmup")
        cap = o["capacity_after_ready"]
        pool = cap["num_gpu_blocks"][0] if len(cap.get("num_gpu_blocks") or []) == 1 else None
        o["steady"] = {}                      # RC2's _windows() gate reads "steady": garbage under load is NO_GO here too
        for wk in arm["windows"]:
            need = win_need(wk, n)
            if not due(need):
                o["steady"][wk] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
                o["items_skipped"].append(wk)
                continue
            if not s.alive():
                o["steady"][wk] = {"status": "SKIPPED_SERVER_DEAD"}
                continue
            CUR["item"] = wk
            band = wk.split("@")[0].replace("profile_", "")
            gen_est = int((WIN + n * BANDS[band] / max(500.0, PREFILL["tok_s"])) * 300.0 / n)   # ~300 tok/s aggregate (NB4S QD)
            cp = N.capacity_predict(pool, cap.get("F_blocks"), n, BANDS[band], gen_est)     # recorded, never gating
            try:
                r = profile_window(p, arm) if wk.startswith("profile") else hist_window(p, s, band, n)
            except Exception as e:
                r = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
            r["capacity_predict"] = cp
            o["steady"][wk] = r
            _live(o, "steady", o["steady"])
        CUR["item"] = "end"
        o["census_end"] = kv_census(s)
        o["workers_end"] = rpc_cache_info(p)
        o["free_end_min"] = free_min(p)
        o["q_readback_end"] = q_readback(s, arm)
        o["ppq_stats_last"] = [e for e in P.parse_log(s.log() or "") if e.get("event") == "stats"][-8:]
        o["jit_after_warmup"] = jit_kernels(s.log())[-12:]
        o["config_readback_end"] = N.config_readback(arm, s.log() or "", CUR["kv_bytes"])
        return o
    return body


def arm_watched(body, cap_min, name):
    """NB4S's watched(): the cap fixed at body START = min(cap, left - BODY_RESERVE_SEC); at cap + 300 s the server is killed."""
    def b(ss):
        cap = N.body_cap_sec(cap_min, left())
        BOOT.update(deadline=time.time() + cap, fired=None)

        def fire():
            BOOT["fired"] = name
            for x in ss:
                try:
                    x.kill()
                except Exception:
                    pass
        tm = threading.Timer(cap + 300, fire)
        tm.daemon = True
        tm.start()
        try:
            o = body(ss)
            LIVE["out"] = o if isinstance(o, dict) else None
        finally:
            tm.cancel()
            BOOT["deadline"] = None
        o["body_cap_sec"] = round(cap)
        if BOOT["fired"]:
            o["boot_watchdog"] = f"FIRED after {cap + 300:.0f} s: server killed"
        return o
    return b


def arm_boot(arm, body):
    """One arm: the boot gate, one Merlin launch (its build knobs set for this launch name), the /dev/shm cleanup; NB4S's
    ONE OOM policy (launch OOM in the launch's full log, no device fault -> one retry 0.25 GiB lower on a fresh port)."""
    kvb, port, tries, o = PIN["kv_bytes"], arm["port"], [], {}
    ref = "rc2_fp8" if arm["key"] == "Q" else "q"
    while True:
        lname = arm["name"] if not tries else f"{arm['name']}_t{len(tries)}"
        NB1_EXTRA[lname], NB1_ENV[lname], NB1_KV[lname] = list(arm["extra"]), dict(arm["env"]), kvb
        NB4T_BUILD_ENV[lname] = dict(arm["build_env"])
        ARM_OF[lname] = arm
        CUR["kv_bytes"] = kvb
        BOOT["tag"] = lname
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries, "launch_name": lname, "pin": kvb}
        shm0 = shm_names()
        mon = LoadMonitor(lname)
        mon.start()
        srv, CUR["item"] = None, "launch"
        try:
            srv = merlin_fn_server(lname, "fp8", port, kvb)
            o = with_servers([srv], NB4T_READY_SEC, arm_watched(body, arm["cap_min"], lname))
        finally:
            mon.stop_ev.set()
        try:
            full = srv.log() or ""
        except Exception:
            full = ""
        lines = full.splitlines()
        o.update(kv_bytes=kvb, pin=kvb, launch_name=lname, port=port, boot_gate=gate, shm_cleanup=shm_cleanup(shm0),
                 ready_sec=getattr(srv, "ready_sec", None), load_monitor_last=mon.last, oom_in_log=bool(OOM_ERR.search(full)),
                 xid_in_log=bool(XID.search(full)), oom_lines=[l[:240] for l in lines if OOM_ERR.search(l)][:6],
                 xid_lines=[l[:240] for l in lines if XID.search(l)][:6], died_in=CUR["item"])
        if o.get("q_readback") is None and srv is not None:
            o["q_readback"] = q_readback(srv, arm)        # a launch that failed: the init lines (if any) are still in the log
        if srv is not None:
            o["q_readback_final"] = q_readback(srv, arm)  # the whole log after the server stopped (F1's final stats line)
            o.setdefault("config_readback", N.config_readback(arm, full, kvb))
        oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]
        tries.append({"launch_name": lname, "port": port, "kv_bytes": kvb, "status": o.get("status"), "oom": oom,
                      "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"]})
        if not oom or o["xid_in_log"] or len(tries) > 1 or left() < NB4T_READY_SEC + arm["cap_min"] * 60:
            break
        kvb -= OOM_STEP
        port += 1
        log("boot OOM:", lname, "-> one retry with KV_BYTES", kvb, "port", port)
    o["tries"] = tries
    if o.get("status") == "SERVER_DIED":
        o["death"] = ("BOOT_CAP_KILL" if o.get("boot_watchdog") else
                      f"BODY_OOM_{o.get('died_in')}" if o["oom_in_log"] and not o["xid_in_log"] else
                      f"DEVICE_FAULT_{o.get('died_in')}" if o["xid_in_log"] else f"SERVER_DIED_IN_{o.get('died_in')}")
        o["status"] = o["death"]
    o["capacity"] = N.capacity_facts(o)
    return gated(o, need=(ref,), expected_rows=4)


def analyze_arm(o, key):
    """After the boot (GPUs idle): nb4t_lib.graph_stalls + nb4_timeline.analyze on the arm's profiled traces."""
    pk = ARMS[key]["windows"][2]
    d = ((o.get("steady") or {}).get(pk) or {}).get("trace_dir")
    if not d or not list(Path(d).glob("*.pt.trace.json*")):
        o["stall"] = {"error": "NO_TRACE_FILE"}
        o["timeline"] = {"error": "NO_TRACE_FILE"}
        return o
    t = time.time()
    try:
        st = N.graph_stalls(d)
        (OUT / f"stall_{key.lower()}.json").write_text(json.dumps(st, indent=1, default=str))
        o["stall"] = {k: v for k, v in st.items() if k != "by_rank"}
        o["stall"]["by_rank"] = {r: {x: y for x, y in v.items() if x != "rows"} for r, v in (st.get("by_rank") or {}).items()}
    except Exception as e:
        o["stall"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
    try:
        rep = TL.strip_private(TL.analyze(d))
        (OUT / f"timeline_{key.lower()}.json").write_text(json.dumps(rep, indent=1, default=str))
        o["timeline"] = N.summarize_timeline(rep)
        o["stage_compute"] = N.stage_compute(o["timeline"])
        try:
            (OUT / f"timeline_{key.lower()}.md").write_text(TL.format_report(rep))
        except Exception as e:
            o["timeline"]["format_report_error"] = repr(e)[:200]
    except Exception as e:
        o["timeline"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
    o["parity_record"] = N.parity_of(o)
    o["analysis_sec"] = round(time.time() - t, 1)
    return o


def run_arm(key):
    arm = ARMS[key]
    o = arm_boot(arm, arm_body(arm, "rc2_fp8" if key == "Q" else "q", "q" if key == "Q" else None))
    LIVE["out"] = o
    try:
        o["arm"], o["planned"] = key, NB4T["arms"][key]
        o["free_mib_per_card"] = free_per_card(o, o.get("launch_name"))
        if left() > 120:
            analyze_arm(o, key)
    except Exception as e:
        keep_after_error(o, e)
    return o


def _arm_rec(key):
    r = (R["sections"].get(key) or {}).get(ARMS[key]["name"])
    return r if isinstance(r, dict) else None


def arm_clean(o):
    return isinstance(o, dict) and o.get("status") == "DONE" and o.get("correctness") == "CLEAN" and o.get("quality_gate") == "PASS"


def skip_arm(key, why, status="SKIPPED_DEPENDENCY", **kw):
    R["sections"].setdefault(key, {})[ARMS[key]["name"]] = dict({"status": status, "why": why}, **kw)
    save()


def finalize():
    """Always (finally): analysis of any arm with traces but no analysis yet, then the verdict + nb4t_verdict.json."""
    try:
        recs = {}
        for k in ARMS:
            r = _arm_rec(k)
            if r is not None and "stall" not in r and ((r.get("steady") or {}).get(ARMS[k]["windows"][2]) or {}).get("trace_dir"):
                try:
                    analyze_arm(r, k)
                except Exception as e:
                    r["stall"] = {"error": repr(e)[:300]}
            recs[k] = r
        v = N.verdict({k: r for k, r in recs.items() if r is not None}, ARMS)
    except Exception as e:
        v = {"verdict": "ERROR_IN_VERDICT", "why": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
    v["arms_status"] = {k: (_arm_rec(k) or {}).get("status") for k in ARMS}
    v["patch"] = NB4T.get("patch")
    v["minutes"] = round((time.time() - T_START) / 60, 1)
    NB4T["verdict"] = v
    try:
        (OUT / "nb4t_verdict.json").write_text(json.dumps(v, indent=1, default=str))
    except Exception:
        pass
    save()
    log("NB4T VERDICT", v.get("verdict"), "|", str(v.get("why"))[:400])
    return v


def _patch_rec(pseq):
    return {"state": pseq.state, "log": pseq.log, "error": pseq.error, "apply": pseq.apply_rec, "revert": pseq.revert_rec}


def nb4t_main(runner=None, pseq=None):
    """install -> #60490 + extension -> sha gate -> nb4_ppack patch -> Q -> I8 -> P26 -> P27 -> C12 (each later arm needs a
    clean, engaged Q and its full minimum of time; P27 needs P26 without an OOM). The patch is applied before Q, kept for
    every arm, reverted at the end (pre-shas verified); the verdict is written in the finally block whatever happens."""
    runner = runner or run_arm
    pseq = pseq or PSEQ
    try:
        m = R["sections"]["A0"]["merge"]
        if not m.get("ok"):
            NB4T["stop"] = "MERGE_FAILED: no boot started"
            for k in ARMS:
                skip_arm(k, NB4T["stop"])
            return
        step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        pt = step("A0", "patch_60490", 3, apply_60490)
        if pt.get("status") != "DONE" or not pt.get("ok"):
            for k in ARMS:
                skip_arm(k, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
            return
        if "rc2_fp8" not in ROWS and not FAKE:
            for k in ARMS:
                skip_arm(k, "RC2's recorded FP8 oracle rows do not match this notebook's prompts (no gate reference)")
            return
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            for k in ARMS:
                skip_arm(k, "sha gate: " + why, "SKIPPED_SHA_GATE")
            return
        ok_p, why_p = pseq.before_arm(ARMS["Q"])
        NB4T["patch"] = _patch_rec(pseq)
        if not ok_p:
            skip_arm("Q", why_p, "NOT_INTEGRATED", not_integrated=True)
            for k in N.ORDER[1:]:
                skip_arm(k, "no base: Q " + why_p)
            return
        q = step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q"))
        bq = N.base_check(q)
        if not arm_clean(q) or not bq[0]:
            for k in N.ORDER[1:]:
                skip_arm(k, f"Q not a clean base (status {q.get('status')}, correctness {q.get('correctness')}, gate "
                            f"{q.get('quality_gate')}, base {bq[1]}: {bq[2]})")
            return
        PIN.update(kv_bytes=q.get("pin") or PIN["kv_bytes"], source="Q's final pin")
        for k in N.ORDER[1:]:
            a = ARMS[k]
            if k == "P27":
                run_ok, why_d = N.p27_dependency(_arm_rec("P26"))
                if not run_ok:
                    skip_arm(k, why_d, "SKIPPED_P26_OOM")
                    continue
            if not N.arm_fits(left(), a):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(priority Q > I8 > P26 > P27 > C12)", "SKIPPED_BUDGET")
                continue
            ok_p, why_p = pseq.before_arm(a)
            NB4T["patch"] = _patch_rec(pseq)
            if not ok_p:
                skip_arm(k, why_p, "NOT_INTEGRATED", not_integrated=True)
                continue
            step(k, a["name"], a["est_min"], lambda k=k: runner(k))
    finally:
        try:
            pseq.finish()
        except Exception as e:
            pseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        NB4T["patch"] = _patch_rec(pseq)
        finalize()


def _gpu_util():
    while not _gpu_util_stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                with open(OUT / "nb4t_gpu_util.jsonl", "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - T_START, 1), "gpus": r.stdout.strip().splitlines()}) + "\n")
        except Exception:
            pass
        _gpu_util_stop.wait(5)


_gpu_util_stop = threading.Event()
if not RERUN and not os.environ.get("RC2_UNIT"):
    if NVSMI_OK:
        threading.Thread(target=_gpu_util, daemon=True).start()
    nb4t_main()
    _gpu_util_stop.set()
    R["host_phase"] = HOST_PHASE
    R["prefill"] = PREFILL
    R["minutes"] = round((time.time() - T_START) / 60, 1)
    save()
save()
'''

# ---------------------------------------------------------------- CELL 8: report + the identity fallback submission
CELL8 = r'''# ==================================================================== CELL 8 — report + the identity fallback submission (E18 R2 format)
if not RERUN and not os.environ.get("RC2_UNIT"):
    write_fallback_submission()
    def _arm_brief(k):
        r = _arm_rec(k) or {}
        w = r.get("steady") or {}
        return {"status": r.get("status"), "why": r.get("why"), "correctness": r.get("correctness"), "quality_gate": r.get("quality_gate"),
                "oracle_vs": r.get("oracle_vs"), "pin": r.get("pin"), "tries": r.get("tries"), "ready_sec": r.get("ready_sec"),
                "env_readback": (r.get("env_readback") or {}).get("why"), "config_readback": (r.get("config_readback") or {}).get("why"),
                "config_checks": (r.get("config_readback") or {}).get("checks"), "q_readback_final": (r.get("q_readback_final") or {}).get("label"),
                "capacity": r.get("capacity"), "stage_compute": r.get("stage_compute"), "parity": r.get("parity_record"),
                "windows": {k2: N.window_brief(v) for k2, v in w.items() if isinstance(v, dict) and not k2.startswith("profile")},
                "stall_stage0": {x: ((r.get("stall") or {}).get("stage0") or {}).get(x) for x in ("stall_ms_median", "share_below_1ms")},
                "stall_stage1": {x: ((r.get("stall") or {}).get("stage1") or {}).get(x) for x in ("stall_ms_median", "sendrecv_ms_per_tick_mean")},
                "timeline": {x: (r.get("timeline") or {}).get(x) for x in ("tick_ms_mean", "stages", "parity_verdict", "error")},
                "error": r.get("error")}
    R["summary"] = {k: _arm_brief(k) for k in ARMS}
    save()
    _v = NB4T.get("verdict") or {}
    print("NB4T THROUGHPUT WINNER:", _v.get("best_arm"), _v.get("best_verdict"), _v.get("best_gain_8k"), "|", _v.get("why"))
    print("C12 (measure only):", json.dumps(_v.get("c12"), default=str)[:1500])
    print(json.dumps(_v, default=str)[:6000])
    print(json.dumps(R["summary"], default=str)[:14000])
'''

MD0 = """# E25 NB4T LEVERS: the next speed levers on the Q-defer base

NB4S (fa9f35ca, GPU run v1): Q-defer (deferred PP receive) = 309.1 tok/s at 8k@8 (+21.1 %), 297.1 at 40k@8 (+19.8 %), oracle
EXACT. After Q-defer, stage-1 compute (33.0 ms) binds; stage 0 is 21.6 ms.

| Arm (order) | Config = Q + exactly one change | Read-back |
|---|---|---|
| Q | NB4S B (FP8 KV, pin 3,553,580,524 B, text-only, #60490, MTP K3, seqs 8, profiler) + nb4_ppack rev 2 + `NB4_PPDEFER=1` | env, Q-defer ENGAGED, oracle vs RC2 FP8 |
| I8 | `DRAFT_HEAD=int8` (Merlin build knob → `draft_lm_head_quant: int8`), K3 | spec + `Draft lm_head: INT8` lines |
| P26 | `VLLM_PP_LAYER_PARTITION=26,22` | worker env, stage layer ranges, pool blocks, free per rank, per-stage compute |
| P27 | `VLLM_PP_LAYER_PARTITION=27,21` (skipped after a P26 OOM) | the same |
| C12 | `--max-num-seqs 12`, 12 streams: 8k@12 / 40k@12 vs Q 8k@8 / 40k@8 (measure only) | seqs 12, running == 12, preemptions, KV usage |

Per arm: boot + read-backs, greedy oracle vs Q (EXACT / NEAR_TIE), 8k ×2 and 40k steady windows, one profiled 8k window
(nb4_timeline: per-stage compute, stall, parity). Parity is recorded for every arm; a stall comparison needs the same
parity label or modal split (else `PARITY_DIFFERS`); the base is never rejected for parity. Throughput winner (I8 / P26 /
P27): 8k speed PASS on both windows, both 40k windows valid, no 40k regression. Hard stop 120 min; the patch is reverted at
the end. Commit run only; a competition rerun writes CELL 0's identity fallback submission (E18 R2 format).
"""

# ---------------------------------------------------------------- assemble
subs = {"__LIBB64__": b64z(LIB_SRC), "__LIBSHA__": LIB_SHA, "__TLB64__": b64z(TL_SRC), "__TLSHA__": TL_SHA,
        "__PPB64__": b64z(PP_SRC), "__PPSHA__": PP_SHA, "__NB4SSHA__": NB4S_SHA, "__E18C0SHA__": E18_CELL0_SHA,
        "__FLAGS__": json.dumps(FLAGS)}
new7, new8 = CELL7, CELL8
for k, v in subs.items():
    new7, new8 = new7.replace(k, v), new8.replace(k, v)
assert "__" + "LIB" not in new7 and '"""__' not in new7
cells_src = [CELL0, CELL1, CELL2] + list(CELL3_6) + ["if not RERUN:\n" + textwrap.indent(c, "    ") for c in (new7, new8)]
for c in cells_src:
    compile(c, "<nb4t cell>", "exec")
assert cells_src[0] == CELL0 and sha(cells_src[0].encode()) == E18_CELL0_SHA
assert all(c.startswith("if not RERUN:\n") for c in cells_src[2:])
assert cells_src[3:7] == NB4S_CODE[2:6]

# every name loaded in the notebook is defined somewhere in it (assign / def / class / import / for / with / except) or builtin
_defined, _loaded = set(dir(builtins)) | {"__file__", "__name__"}, set()
for c in cells_src:
    for node in ast.walk(ast.parse(c)):
        if isinstance(node, ast.Name):
            (_loaded if isinstance(node.ctx, ast.Load) else _defined).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            _defined.add(node.name)
            _defined.update(a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs) if not isinstance(node, ast.ClassDef) else None
            if not isinstance(node, ast.ClassDef):
                if node.args.vararg:
                    _defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    _defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            _defined.update(a.arg for a in node.args.args + node.args.kwonlyargs)
            if node.args.vararg:
                _defined.add(node.args.vararg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            _defined.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            _defined.add(node.name)
        elif isinstance(node, ast.arg):
            _defined.add(node.arg)
_undef = sorted(_loaded - _defined)
assert not _undef, f"names loaded but never defined in the notebook: {_undef}"
for nm in ("write_json_atomic", "identity_submission", "test_challenges", "SUBMISSION_PATH", "prompt", "BANDS", "PROMPTS", "window",
           "warmup", "oracle", "quality_gate", "gated", "boot_facts", "kv_census", "rpc", "rpc_cache_info", "free_per_card"):
    assert nm in _defined, nm

# the arm plan is valid at build time too (never-do list, one change per arm)
_libns = {"__name__": "nb4t_lib_build"}
exec(compile(LIB_SRC, "nb4t_lib.py", "exec"), _libns)
_ppns = {"__name__": "nb4_ppack_build"}
exec(compile(PP_SRC, "nb4_ppack.py", "exec"), _ppns)
_arms = _libns["arm_plan"](["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}"],
                           {"VLLM_SERVER_DEV_MODE": "1"}, 3553580524, q_env=_ppns["server_env"]("Q-defer"))
_libns["check_arm_plan"](_arms)
assert _arms["Q"]["env"].get("NB4_PPDEFER") == "1" and _arms["I8"]["build_env"] == {"DRAFT_HEAD": "int8"}
assert _arms["P26"]["env"]["VLLM_PP_LAYER_PARTITION"] == "26,22" and _arms["P27"]["env"]["VLLM_PP_LAYER_PARTITION"] == "27,21"
assert _arms["C12"]["extra"][-2:] == ["--max-num-seqs", "12"] and _arms["C12"]["streams"] == 12

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
               [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cells_src],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
data = json.dumps(nb, indent=1).encode("utf-8")
assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
OUT_NB.write_bytes(data)
EXPORT.write_bytes("\n\n".join(cells_src).encode("utf-8"))
PUSH.mkdir(parents=True, exist_ok=True)
(PUSH / f"{SLUG}.ipynb").write_bytes(data)
meta = {"id": f"foysalemonshanto/{SLUG}", "title": "Q27 Xhigh E25 NB4T Levers", "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": MODELS, "docker_image": IMAGE, "machine_shape": "NvidiaL4"}
(PUSH / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
SHAS_AFTER = other_files_shas()
_changed = sorted(k for k in set(SHAS_BEFORE) | set(SHAS_AFTER) if SHAS_BEFORE.get(k) != SHAS_AFTER.get(k))
facts = {"notebook": OUT_NB.name, "sha256": sha(data), "bytes": len(data), "cells": len(nb["cells"]),
         "cell_bytes": [len(c.encode()) for c in cells_src], "lib_sha256": LIB_SHA, "timeline_sha256": TL_SHA, "ppack_sha256": PP_SHA,
         "nb4s_sha256": NB4S_SHA, "e18_r2_sha256": E18R2_SHA, "e18_r2_cell0_sha256": E18_CELL0_SHA,
         "export_sha256": sha(EXPORT.read_bytes()), "push_dir": str(PUSH), "other_files_checked": len(SHAS_BEFORE),
         "other_files_changed": _changed, "nb4s_code_cell_sha256": [sha(c.encode()) for c in NB4S_CODE],
         "nb4t_code_cell_sha256": [sha(c.encode()) for c in cells_src], "flags": FLAGS}
FACTS.write_bytes((json.dumps(facts, indent=1) + "\n").encode("utf-8"))
print(json.dumps({k: v for k, v in facts.items() if k != "flags"}, indent=1))
if _changed:
    raise SystemExit(f"OTHER FILES CHANGED DURING THE BUILD (not by this builder?): {_changed}")
