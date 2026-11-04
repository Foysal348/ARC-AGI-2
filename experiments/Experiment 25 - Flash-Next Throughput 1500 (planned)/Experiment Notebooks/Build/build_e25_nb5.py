"""E25 NB5 FN-Duck (spec: Experiment Review/NB_BUILD_SPEC_2026-10-31.md section 3 NB5 + the two coordinator addenda):
the Duck ARC-AGI-2 harness (config B = E15B R3K2, as run on vLLM by E21) ported to the Merlin Flash-Next server (the
Kaggle-proven RC2 FP8 boot path). Build + local tests only; nothing here pushes or runs on Kaggle.

Reuse (verbatim; every piece is cut from its source and asserted):
- RC2 (experiments/Experiment 13 .../Build/build_e13_rc2.py, Kaggle run 12eaa96c): the builder's own cell strings
  (its definitions are executed up to its notebook write): the e23_lab module, the LAB load + traced Stream
  (CELL 1), the input discovery block (CELL 1, wrapped as nb5_rc2_inputs()), CELL 2 up to the stock fallback (run +
  -lcuda shim, host_facts, host_mem, telemetry, /dev/shm cleanup, MServer, boot_gate, merge_ckpt, sha_gate,
  prefetch_bg + merlin_install, the KV pins, merlin_fn_server), CELL 3 (A0 + the Flash-Next tokenizer / xhigh template
  checks + prompts, wrapped as nb5_rc2_prompts()), the E23 step runner (CELL 4) and kv_census / sampled_check /
  prefill_probe (CELL 5).
- E21 NB2 notebook (Kaggle-run, sha cbbfd01b) = E15B R3K2 (c058d58b) cells with the vLLM fixes: CELL 0, CELL 3 (Duck
  package payload 9e4b8928 + the 'reasoning' history key + the finish_reason 'length' restore + sandbox self-test),
  the task-order cell, CELL 6/7 wrapped (e21_cell6 / e21_cell7), from CELL 1 the time block and the E12 / E15B config
  block, from CELL 1b q27x_payload; from E21's engine code: the vLLM metrics / stats-line parsers, counters, content
  probes, contract probe, cache probe (build_e21.segment()).
New (marked NB5): config (Flash-Next values), nb5_lib.py (capacity, live-block reservation, page-cache drop + RAM gate,
commit-turn variant C, request tracer, reports), the Merlin relaunch for E15B's watchdog, the warm-up, the pure
decode window, the commit / parser / cache-state probes, the phase runner (SMOKE -> metric + workload reset -> EVAL,
optional MTP-off EVAL), the report.

usage: python build_e25_nb5.py [--profile NB5EVAL120|NB5EVAL30|NB5EVAL30MTP]   ->  ../Q27-XHIGH-E25_NB5_FNDUCK[...].ipynb
"""
import hashlib
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
EXP = Path(r"D:/ARC AGI 2/experiments")
RC2_BUILDER = EXP / "Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)" / "Experiment Notebooks" / "Build" / "build_e13_rc2.py"
E21_BUILD = EXP / "Experiment 21 - Duck config B on vLLM 0.19 + 0.32 (planned)" / "Experiment Notebooks" / "Build"
E21_NB = E21_BUILD.parent / "Q27-XHIGH-E21_DUCKB_VLLM_NB2_FP8_SSM16_S16.ipynb"
E21_NB_SHA = "cbbfd01b14dd9f3726c83ab137947174e8d3e84bc6528fcb20e381f2138aaa3d"
RC2_RESULTS = Path(r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2/rc2_results.json")
PROFILES = {"NB5EVAL120": "Q27-XHIGH-E25_NB5_FNDUCK.ipynb", "NB5EVAL30": "Q27-XHIGH-E25_NB5_FNDUCK_EVAL30.ipynb",
            "NB5EVAL30MTP": "Q27-XHIGH-E25_NB5_FNDUCK_EVAL30MTP.ipynb", "NB5QUALITY": "Q27-XHIGH-E25_NB5_FNDUCK_QUALITY.ipynb",
            "NB5SMOKEDIAG": "Q27-XHIGH-E25_NB5_FNDUCK_SMOKEDIAG.ipynb", "NB5BUDGET": "Q27-XHIGH-E25_NB5_FNDUCK_BUDGET.ipynb",
            # the paired FP8-vs-BF16 KV ARC accuracy test (lead 2026-10-31, user-approved; spec change R2: ONE notebook,
            # two phases, 3 h, equal windows, three capacity levers)
            "NB5QUALITY_R2": "Q27-XHIGH-E25_NB5_QUALITY.ipynb"}
PUSHED: set = {            # sha256 of every NB5 notebook that ran on Kaggle or is frozen (never overwrite one: rebuild -> *_REBUILD)
    # the six pre-R2 notebooks (frozen 2026-10-31 by the lead: they keep the pre-fix code; a rebuild goes to *_REBUILD)
    "9887c63cbbad10bd45deb47170b60a02cdb34a36c3c676ef9f68644fe31b2bc6",   # Q27-XHIGH-E25_NB5_FNDUCK.ipynb
    "52a38c7e496641bd0d8156b057b4e71461c267aab676ad0bc18413801a8569a8",   # ..._FNDUCK_BUDGET.ipynb
    "ff52ccea5f90baba22255557529cbd773203bbeddee2a2e541818c8e8fa48e32",   # ..._FNDUCK_EVAL30.ipynb
    "50abce270b86faf1badd677362012c0e498fe3fb08997da6e89ffe09381b663a",   # ..._FNDUCK_EVAL30MTP.ipynb
    "1a9462c4f3e6c9eec4b0714c4f22fefac6dd6b2e707f6ae8c168fa6c0b1c9779",   # ..._FNDUCK_QUALITY.ipynb
    "db766836b60e940ceb923cbc043f972c7e5eec99d07963d6cf760a1e29e3bf18"}   # ..._FNDUCK_SMOKEDIAG.ipynb
EXPORT_PROFILE = "NB5QUALITY_R2"   # nb5_code_export.py = the code cells of this build (the frozen six keep the older code)
# ---- vLLM #60490 backport (NB1's strict applier + patch, sha-pinned; lead R2 lever a: the QSA indexer workspace is
# reserved at profile time, so a larger KV pin fails at launch instead of OOM at the first requests)
NB1_BUILDER = HERE / "build_e25_nb1.py"
_nb1_src = NB1_BUILDER.read_text(encoding="utf-8").replace("\r\n", "\n")
APPLIER = _nb1_src[_nb1_src.index("APPLIER = r'''\n") + len("APPLIER = r'''\n"):_nb1_src.index("'''\nPATCH_TEXT = PATCH.read_bytes()")]
assert APPLIER.startswith("def apply_unified(text, patch):") and APPLIER.rstrip().endswith('return "".join(out + src[pos:])')
PATCH_TEXT = (HERE / "patches" / "60490_qsa_indexer_workspace_merlin_ad42e37.patch").read_bytes().decode("utf-8")
assert "\r" not in PATCH_TEXT
P60490_PRE = "a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653"   # = NB1 P_PRE (Merlin ad42e37 qsa.py)
P60490_POST = "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b"  # = NB1 P_POST (NB1 GPU: sha_after 88fe0a8f)
assert f'P_PRE = "{P60490_PRE}"' in _nb1_src and f'P_POST = "{P60490_POST}"' in _nb1_src
assert f"pre-patch sha256  {P60490_PRE}" in PATCH_TEXT and f"post-patch sha256 {P60490_POST}" in PATCH_TEXT
CAPS16 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]   # = NB1 CAPTURE_HI = shipped U Merlin capture_sizes_for(16, K=3)
assert "CAPTURE_HI = " + json.dumps(CAPS16) in _nb1_src


def sha(b):
    return hashlib.sha256(b).hexdigest()


def norm(t):
    return t.replace("\r\n", "\n")


# ------------------------------------------------------------------ the RC2 builder's definitions (no write)
_rc2_src = norm(RC2_BUILDER.read_text(encoding="utf-8"))
_cut_at = _rc2_src.index('subs = {"__LAB__": json.dumps(LAB), "__PLE__": json.dumps(PLE)}')
RC2 = {"__file__": str(RC2_BUILDER), "__name__": "rc2_builder_defs"}
exec(compile(_rc2_src[:_cut_at], str(RC2_BUILDER), "exec"), RC2)
cut, once = RC2["cut"], RC2["once"]                   # RC2's own helpers (count == 1 asserted)
assert RC2["PUSHED"] == {"12eaa96cea2ced510ef54bca83586a3f545cb0609ccdb5634cf32ecfb9fcafa5"}, "RC2 builder changed"
LAB = RC2["LAB"]
PLE = RC2["PLE"]

# ------------------------------------------------------------------ the E21 builder's helpers + the E21 NB2 notebook
E21 = {"__file__": str(E21_BUILD / "build_e21.py"), "__name__": "e21_builder_defs"}
exec(compile(norm((E21_BUILD / "build_e21.py").read_text(encoding="utf-8")), "build_e21.py", "exec"), E21)
segment, wrap = E21["segment"], E21["wrap"]
_b = E21_NB.read_bytes()
assert sha(_b) == E21_NB_SHA, ("E21 NB2 donor changed", sha(_b))
D = [norm("".join(c["source"])) for c in json.loads(_b.decode("utf-8"))["cells"]]
assert D[1] == E21["E15B"][1], "CELL 0 differs from E15B R3K2"
E21_ENGINE_NEW = norm((E21_BUILD / "e21_engine_new.py").read_text(encoding="utf-8"))
NB5_LIB = norm((HERE / "nb5_lib.py").read_text(encoding="utf-8"))
RC2_FP8_ARGV_TAIL = json.loads(RC2_RESULTS.read_text(encoding="utf-8"))["setup"]["merlin_plans"]["merlin_fp8"]["argv_tail"]


def dedent_block(t):
    return textwrap.dedent(t)


def packed(text, label):
    """A verbatim module as a zlib + base64 literal with its sha256 checked at load (E13 Duck builder pattern; keeps the
    commit notebook < 400 kB). The decompressed text is byte-identical to the source file."""
    import base64
    import zlib
    b64 = base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode()
    h = sha(text.encode("utf-8"))
    return (f'(lambda _t: _t if __import__("hashlib").sha256(_t.encode()).hexdigest() == "{h}" else '
            f'(_ for _ in ()).throw(RuntimeError("{label}: embedded source sha256 mismatch")))('
            f'__import__("zlib").decompress(__import__("base64").b64decode("{b64}")).decode("utf-8"))')


# ------------------------------------------------------------------ RC2 pieces
RC2_CELL1, RC2_CELL2, RC2_CELL3, RC2_CELL5 = RC2["CELL1"], RC2["CELL2"], RC2["CELL3"], RC2["CELL5"]
RC2_LAB_LOAD = dedent_block(cut(RC2_CELL1, "    LAB_SRC = __LAB__", "    PLE_EXPECTED = json.loads", "rc2 lab load"))
RC2_LAB_LOAD = once(RC2_LAB_LOAD, "LAB_SRC = __LAB__", "LAB_SRC = " + packed(LAB, "e23_lab"), "lab literal")
RC2_INPUTS = dedent_block(cut(RC2_CELL1, "    PLE_EXPECTED = json.loads", None, "rc2 inputs"))
RC2_INPUTS = once(RC2_INPUTS, "L.start_watchdog(HARD_STOP,", "L.start_watchdog(HARD_STOP_EPOCH,", "watchdog epoch")
RC2_INPUTS = RC2_INPUTS.replace("__PLE__", json.dumps(PLE))
RC2_ENGINE = cut(RC2_CELL2, "def run(name, cmd", "STOCK = None", "rc2 cell2 to the stock fallback")
RC2_ENGINE = once(RC2_ENGINE, 'TELEMETRY = OUT / "telemetry.jsonl"', 'TELEMETRY = ENG_OUT / "telemetry.jsonl"   # NB5: engine folder',
                  "telemetry path")
RC2_ENGINE = once(RC2_ENGINE, 'PROFILE = "fn-quad-24g-sm89"', 'MERLIN_PROFILE = "fn-quad-24g-sm89"   # NB5: PROFILE is the Duck profile name',
                  "merlin profile name")
RC2_ENGINE = once(RC2_ENGINE, 'eng["ML"].build("flash-next", PROFILE, str(CKPT)', 'eng["ML"].build("flash-next", MERLIN_PROFILE, str(CKPT)',
                  "merlin profile use")
RC2_ENGINE = once(RC2_ENGINE, '"--default-chat-template-kwargs", XHIGH]\n    if FAKE:',
                  '"--default-chat-template-kwargs", XHIGH] + list(NB5_BOOT_EXTRA)   # NB5: NB1/NB2 levers, MTP off\n    if FAKE:',
                  "extra args")
RC2_ENGINE = once(RC2_ENGINE, '"--port", str(port), "--max-model-len", str(MAX_LEN)], os.environ, port, LOGS)',
                  '"--port", str(port), "--max-model-len", str(MAX_LEN),\n'
                  '                              "--kv-cache-dtype", NB5L.KV_DTYPE_NAME[kv], "--kv-cache-memory-bytes", str(kv_bytes)]'
                  ' + list(NB5_BOOT_EXTRA),   # NB5: the fake reports dtype / pin (C-11 read-back, the ladder)\n'
                  '                       dict(os.environ, **NB5_BASE_ENV, **NB5_BOOT_LEVERS), port, LOGS)',
                  "fake extra args")
RC2_ENGINE = once(RC2_ENGINE, '                CUDA_HOME=eng["cuda_home"])\n    if kv == "fp8":',
                  '                CUDA_HOME=eng["cuda_home"])\n    base.update(NB5_BASE_ENV)                          # NB5: Merlin knobs from NB1 / NB2 (build() reads DRAFT_HEAD etc.)\n'
                  '    base.update(NB5_BOOT_LEVERS)                       # NB5 R2: this boot try\'s lever env (lever b)\n    if kv == "fp8":',
                  "base env")
# NB5Q-ENV-1 (Kaggle real kit 2026-10-31: env_lever null in all three boots): launch.build() drops every MERLIN_* variable
# (merlin-src tools/merlin/launch.py:377) and the server env is dict(plan.env), so the base update alone never reached the
# server. The NB5 env is applied again AFTER build() (= NB1 nb1_code_export.py:449) and its final values are recorded.
RC2_ENGINE = once(RC2_ENGINE, '    argv = plan.argv()\n    HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)\n',
                  '    env.update(NB5_BASE_ENV)                           # NB5Q-ENV-1: after build(), which strips every MERLIN_* variable (= NB1)\n'
                  '    env.update(NB5_BOOT_LEVERS)                        # NB5Q-ENV-1: lever b reaches the server env\n'
                  '    argv = plan.argv()\n    HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)\n',
                  "env after build")
RC2_ENGINE = once(RC2_ENGINE, '"env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES"))}}',
                  '"env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES"))},\n'
                  '                                                       "lever_env": {k: env.get(k) for k in sorted({**NB5_BASE_ENV, **NB5_BOOT_LEVERS})}}   # NB5Q-ENV-1',
                  "lever env record")
RC2_PROMPTS = RC2_CELL3
RC2_PROMPTS = once(RC2_PROMPTS, 'BANDS = {"2k": 2048, "8k": 8192, "40k": 40000, "90k": 90000, "116k": 116000}',
                   'BANDS = {"2k": 2048, "8k": 8192, "40k": 40000}      # NB5: sampled check, warm-up, pure decode window',
                   "bands")
RC2_PROMPTS = once(RC2_PROMPTS, 'TF_P = {"40k": PROMPTS["40k"][5]}', 'TF_P = {}', "no teacher-forced rows")
STEP_RUNNER = RC2["STEP_RUNNER"]
RC2_CHECKS = cut(RC2_CELL5, "def kv_census(s):", "def window_garbage(", "kv_census + sampled_check + prefill_probe")

# ------------------------------------------------------------------ E21 / E15B pieces
C1 = D[2]
TIME_BLOCK = cut(C1, "TOTAL_BUDGET_SEC = float(", "\n\n# v1.3.5 (run 2 OOM)", "time block")
E12_BLOCK = cut(C1, "# ---- E12: Duck REPL agent", "\n\n\n# ====", "E12 + E15B block")
for old, new, lab in (
        ('E12_MAX_GEN_PER_BRANCH = int(os.environ.get("E12_MAX_GEN", 50000 if _E12_REV4 else 0)) or None',
         'E12_MAX_GEN_PER_BRANCH = int(os.environ.get("E12_MAX_GEN", NB5_ATTEMPT["b0"])) or None   # NB5: Flash-Next attempt 1',
         "attempt 1"),
        ('(12288 if _E12_REV6 else 49152 if _E12_REV5 else 60000)', '(NB5_ATTEMPT["b0"] - NB5_RESERVE)', "first cap"),
        ('E12_REASONING_EFFORT = os.environ.get("E12_REASONING_EFFORT", "").strip().lower()',
         'E12_REASONING_EFFORT = os.environ.get("E12_REASONING_EFFORT", "xhigh").strip().lower()   # NB5: forced xhigh', "effort"),
        ('E12_R6 = {"E12_R6_CAP_FIRST": 12288, "E12_R6_CAP_PROGRESS": 24576, "E12_R6_CAP_RESCUE": 32768,',
         'E12_R6 = {"E12_R6_CAP_FIRST": NB5_CAP, "E12_R6_CAP_PROGRESS": NB5_CAP, "E12_R6_CAP_RESCUE": NB5_CAP,   # NB5: no 12k/24k/32k',
         "rev6 caps"),
        ('"E12_R6_KV_ADMIT": 0.95, "E12_R6_KV_POLL": 5.0, "E12_R6_CAP_FIRST_B1": 24576,',
         '"E12_R6_KV_ADMIT": NB5_GATE, "E12_R6_KV_POLL": 5.0, "E12_R6_CAP_FIRST_B1": NB5_CAP,', "b1 cap"),
        ('"E12_R6_NUDGE_TOKENS": 24000, "E12_R6_B1_GRID_MAX_TOKENS": 12000, "E12_R6_MIN_CHUNK": 12288,',
         '"E12_R6_NUDGE_TOKENS": 24000, "E12_R6_B1_GRID_MAX_TOKENS": 12000, "E12_R6_MIN_CHUNK": NB5_CAP,', "min chunk"),
        ('"E12_R6_CACHE_DIAG": 1}', '"E12_R6_CACHE_DIAG": 0}   # NB5: the SGLang cache diagnostic is not run on vLLM', "cache diag"),
        ('E15B = {"E12_B_MODE": 1, "E12_B_ATTEMPT_TOKENS": E12_MAX_GEN_PER_BRANCH or 50000, "E12_B_RESERVE": 4096,\n'
         '        "E12_B_WINDOW": 100000, "E12_B_ATTEMPT_TOKENS_B1": 45000,',
         'E15B = {"E12_B_MODE": 1, "E12_B_ATTEMPT_TOKENS": E12_MAX_GEN_PER_BRANCH or NB5_ATTEMPT["b0"], "E12_B_RESERVE": NB5_RESERVE,\n'
         '        "E12_B_WINDOW": NB5_WINDOW, "E12_B_ATTEMPT_TOKENS_B1": NB5_ATTEMPT["b1"],', "E15B budgets")):
    E12_BLOCK = once(E12_BLOCK, old, new, lab)
PAYLOAD_FN = segment(D[3], "Q27X_PAYLOAD_SLUG") + "\n\n\n" + segment(D[3], "q27x_payload")
# NB5 (lead 2026-10-31, verified): foysalemonshanto owns foysalemonshanto/q27-e15b-payload (byte-identical 9e4b8928 + e14
# modules) and cannot read esfersami50's copy -> the two exact dataset folders first (foysal, then esfersami50 so the notebook
# still runs there), then E21's generic search; sha256 checked either way; the path used is recorded.
PAYLOAD_FN = once(PAYLOAD_FN, 'Q27X_PAYLOAD_SLUG = "esfersami50/q27-e15b-payload"',
                  'Q27X_PAYLOAD_SLUG = "foysalemonshanto/q27-e15b-payload"      # NB5: the foysalemonshanto copy (esfersami50 second)\n'
                  'Q27X_PAYLOAD_DIRS = ["datasets/foysalemonshanto/q27-e15b-payload", "datasets/esfersami50/q27-e15b-payload"]\n'
                  'Q27X_PAYLOAD_USED = {}                                       # NB5: payload name -> the file that was used',
                  "payload slug")
PAYLOAD_FN = once(PAYLOAD_FN, '        for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name):',
                  '        for pat in [d_ + "/" + name for d_ in Q27X_PAYLOAD_DIRS] + [name, "*/" + name, "*/*/" + name, "*/*/*/" + name]:   # NB5',
                  "payload order")
PAYLOAD_FN = once(PAYLOAD_FN, '                if hashlib.sha256(b).hexdigest() == sha:\n                    return b',
                  '                if hashlib.sha256(b).hexdigest() == sha:\n'
                  '                    Q27X_PAYLOAD_USED[name] = {"path": str(f), "sha256": sha, "rejected_before": list(seen)}   # NB5\n'
                  '                    print("payload", name, "<-", f, flush=True)\n'
                  '                    return b',
                  "payload record")
E21_DONOR_FNS = [(D[7], n) for n in ("_e12v_request_json", "_E12V_PROM_RE", "_E12V_ALIAS", "metrics_snapshot")] + \
                [(E21_ENGINE_NEW, n) for n in ("_E12V_STAT_RE", "e21_stat_rows", "e11_decode_summary", "_E21_COUNTERS",
                                               "e21_counters", "e21_snapshot_counters", "E21_CONTENT_PROBES",
                                               "E21_NUMBER_WORDS", "e21_garbage", "e21_content_ok", "e21_content_probes",
                                               "e21_prom", "E21_PROBE_TOOLS", "e21_contract_probe", "e21_soft",
                                               "e21_cache_probe")]
E21_DONORS = []
for txt, n in E21_DONOR_FNS:
    s = segment(txt, n)
    assert s in D[7], ("not in the Kaggle-run E21 engine cell", n)
    E21_DONORS.append((n, s))
CELL67 = D[8]
CELL67 = once(CELL67, '        out["mamba_usage"] = None\n        return out',
              '        out["mamba_usage"] = None\n        out["kv_usage_server"] = out.get("kv_usage")      # NB5: recorded\n'
              '        out["kv_usage"] = nb5_gate_kv(out.get("kv_usage"))  # NB5: the live-block reservation fraction\n'
              '        return out', "capacity probe -> reservation")
_ORD = "    E12_ORDER = sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t]))\n"
CELL67 = once(CELL67, _ORD, _ORD + '    if P.get("task_order"):      # NB5 R2: the lead\'s fixed task order (the same in both paired phases)\n'
                                   '        E12_ORDER = [t for t in P["task_order"] if t in TASK_IDS] + [t for t in E12_ORDER if t not in P["task_order"]]\n',
              "R2 fixed task order")

# ------------------------------------------------------------------ NB5 cells
MD0 = """# E25 NB5 · FN-Duck: Duck config B on Qwen3.8-Flash-Next (Merlin, RC2 FP8 boot path) · __PROFILE__

- **Engine:** Merlin v3.0.0 (vLLM 0.32 fork) `fn-quad-24g-sm89`, TP2×PP2+EP, MTP K3, KV dtype parameter (fp8 default,
  provisional), `--max-model-len 131072`, xhigh forced. Boot = RC2 FP8 (sha-checked merge, offline install, host_phase,
  page-cache drop + Merlin RAM floor, GPU < 150 MiB, ready ≤ 30 min).
- **Duck:** config B (E15B R3K2 as run by E21 on vLLM): attempt 1 = 112k, attempt 2 = 100k generated tokens, long
  uninterrupted thinking (one request may think up to the attempt budget − 4,096), commit turn variant C
  (thinking-on render, `<think>\\n` → `<think>\\n\\n</think>\\n\\n`, token ids), variant A switchable.
- **Admission:** live-block reservation (projected peak blocks of every admitted attempt ≤ 0.95 × pool; pool / F from
  the server, NB1's table, placeholders 149 / 7 only as labelled defaults) + a live server hold; no step-down.
- **Phases:** warm-up (JIT) → pure decode window → SMOKE (2 tasks, A/C split) → metric + workload reset → EVAL
  [→ MTP-off EVAL]. 11:50 hard stop. Rerun = all hidden tasks; commit = evaluation subset + score.
- Report: `q27x/nb5_report.json` (pure decode / steady mixed / solver window / whole notebook averages, prefix + recompute,
  commit-turn tool-call rate, solve-vs-tokens bands, running == in-flight).
"""

CELL1 = r'''# ==================================================================== CELL 1 — NB5 configuration (E25 FN-Duck)
# E15B / E21 profile mechanics; Flash-Next values. Every NB1 / NB2 output that changes the engine is a parameter here.
NB5_DEFAULT_PROFILE = "__PROFILE__"
PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", NB5_DEFAULT_PROFILE).strip().upper()
E11_HARNESS = os.environ.get("E11_HARNESS") == "1"            # local CPU harness only (fake server, no GPU)
_NB5_T = {"time_budget_sec": 710 * 60, "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60,
          "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60}   # 11:50 hard stop
# SMOKE = real evaluation tasks, the Duck's real multi-turn continuation, exact-match scored (addendum 4): two fast ones
# (S1COMMIT4) + the known-answer 0934a4d8 (RC2's FP8 116k answer was wrong, 15/27 cells; 27B xhigh solves it). Never
# tuned on: it is a check, recorded per KV dtype.
NB5_SMOKE_TASKS = ["2ba387bc", "1818057f", "0934a4d8"]
# paired quality A/B (addendum 5, user-approved; OFF by default; runs only after NB0's FP8 gate = PASS; GPU needs the
# user's go): the 10 tasks the 27B solved on every one of >= 3 of our attempts (Puzzle Capability Ledger,
# ledger_all120.csv) + the known-answer 0934a4d8; the same seeds, budgets, prompts and harness in both configs
# revision 4: SMOKE = a 37-min solve window inside a hard 40-min gate (the paired replay must fit before it); the
# gate decides EVAL / switch to variant A / abort; it never moves the 11:45 solve end or the 11:50 watchdog
NB5_SMOKE_GATE_SEC = int(os.environ.get("NB5_SMOKE_GATE_SEC", 40 * 60)) if E11_HARNESS else 40 * 60
NB5_QUALITY_TASKS = ["1ae2feb7", "38007db0", "3e6067c3", "65b59efc", "d8e07eb2", "1818057f", "409aa875", "5961cc34",
                     "7491f3cf", "b99e7126", "0934a4d8"]
NB5_QUALITY_WINDOW = int(os.environ.get("NB5_QUALITY_WINDOW_SEC", 150 * 60))   # a bound per config (expected 1.3 / 1.8 h)
NB5_NB0_FP8 = os.environ.get("NB5_NB0_FP8_GATE", "").strip().upper()             # NB0's FP8 verdict: PASS | FAIL | ""
# ---- the paired FP8-vs-BF16 KV ARC accuracy test NB5QUALITY_R2 (lead 2026-10-31, user-approved; spec R2): the 15 tasks
# the lead chose from resources/Puzzle Capability Ledger/tokens_to_solve.csv (our 27B Duck + Poetiq runs on the public
# eval set), in this order, with their group labels; ONE notebook, phase Q_FP8_K3 then Q_BF16_K3 (separate boots), the
# same seeds / budgets / prompts / Duck config / task order / capacity levers, equal solve windows inside 3 h
NB5_QAB_GROUPS = {"solid": ["1818057f", "45a5af55", "bf45cf4b", "2ba387bc", "8f215267", "38007db0", "58f5dbd5"],
                  "sometimes": ["247ef758", "f931b4a8", "b5ca7ac4", "65b59efc", "221dfab4"],
                  "never": ["2b83f449", "16b78196", "64efde09"]}
NB5_QAB_TASKS = [t for g in ("solid", "sometimes", "never") for t in NB5_QAB_GROUPS[g]]
NB5_QAB_GROUP_OF = {t: g for g, ts in NB5_QAB_GROUPS.items() for t in ts}
# A-9: R2 never requires NB5_NB0_FP8_GATE == PASS; NB0's verdict is baked and recorded, the FP8 run is labelled
NB5_QAB_NB0 = {"verdict": "NO_GO_FP8_TF_DIVERGENCE", "full_prompt_rank1": {"2k": 0.90, "8k": 0.93, "40k": 0.749, "90k": 0.661},
               "source": "E25 Experiment Results/E25_GPU_wave1_NB0_NB1_2026-10-31.md (NB0 esfersami50 sha 37a427a2)"}
NB5_QAB_FP8_LABEL = "UNVALIDATED_FP8_UNDER_TEST"
NB5_CAPS16 = __CAPS16__                                        # NB1 CAPTURE_HI (FULL decode covers 16 x (1 + K3) = 64)
# phase = [name, tasks ("smoke" | "all"), solve window sec (None = until the hard stop), MTP ("on" | "off"), commit
# variant, KV dtype (None = NB5_KV_DTYPE), options {"early_exit": SMOKE exits as soon as its gates pass, "budget":
# both attempts' generated-token budget for this phase}]
PROFILE_TABLE = {
    "NB5EVAL120": dict(_NB5_T, source="eval", n_tasks=120,
                       nb5_phases=[["SMOKE", "smoke", 37 * 60, "on", "split", None, {"early_exit": True}], ["EVAL", "all", None, "on", "C"]]),
    "NB5EVAL30": dict(_NB5_T, source="eval", n_tasks=30,
                      nb5_phases=[["SMOKE", "smoke", 37 * 60, "on", "split", None, {"early_exit": True}], ["EVAL", "all", None, "on", "C"]]),
    # paired accuracy run (PLAN section 0): the same 30 tasks, MTP on then off, equal windows
    "NB5EVAL30MTP": dict(_NB5_T, source="eval", n_tasks=30,
                         nb5_phases=[["SMOKE", "smoke", 37 * 60, "on", "split", None, {"early_exit": True}], ["EVAL", "all", 280 * 60, "on", "C"],
                                     ["EVALOFF", "all", 280 * 60, "off", "C"]]),
    # the paired quality A/B (addendum 5): (FP8 KV, MTP K3) vs (BF16 KV, MTP K3); streams = the no-preemption ceiling at
    # a 120k worst-case context = pool // (ceil(120000 / B) + F) (RC2 pools 149 / 78, F 6 -> 4 / 2); attempts of
    # 104k (8k prompt + 104k + 8k tools = 120k), gate 1.0 (the ceiling itself), two attempts per task, no stage 3, one
    # fixed per-request seed. MTP off is NOT in this A/B (NB0 measures it by teacher forcing)
    "NB5QUALITY": dict(_NB5_T, source="eval", task_list=NB5_QUALITY_TASKS, nb5_gate=1.0,
                       nb5_attempt={"b0": 104000, "b1": 104000},
                       nb5_env={"E12_B_EXTRA_ATTEMPTS": "0", "LOCAL_ANALYZER_SEED": str(20261005)},
                       nb5_phases=[["Q_FP8_K3", "all", NB5_QUALITY_WINDOW, "on", "C", "fp8"],
                                   ["Q_BF16_K3", "all", NB5_QUALITY_WINDOW, "on", "C", "bf16"]]),
    # the long multi-task SMOKE (PLAN v3.6): diagnostic only, never in a production / EVAL profile; no early exit, no EVAL
    "NB5SMOKEDIAG": dict(_NB5_T, source="eval", n_tasks=120,
                         nb5_phases=[["SMOKE", "smoke", 37 * 60, "on", "split", None, {"early_exit": False}]]),
    # budget sweep (PLAN v3.6, off by default): EVAL30, the same 30 tasks per budget, equal 160-min windows
    "NB5BUDGET": dict(_NB5_T, source="eval", n_tasks=30,
                      nb5_phases=[[f"B{k // 1000:03d}", "all", 160 * 60, "on", "C", None, {"budget": k}]
                                  for k in (60000, 80000, 100000, 112000)]),
    "RERUN_ALL": dict(_NB5_T, source="test", n_tasks=None, nb5_phases=[["EVAL", "all", None, "on", "C"]]),
    # spec R2 (lead 2026-10-31): 3 h hard runtime incl. install and both boots; equal solve windows (fixed when the first
    # phase starts); commit variant A (C is not validated, A-11); no SMOKE (the boot's fail-fast probes stay); 104k per
    # attempt, 2 attempts, no stage 3, one fixed seed; levers a (KV pin ladder + #60490) and b (non-numeric smaller F,
    # measured F), the same in both phases; concurrency (simplified, user decision): workers = max_num_seqs =
    # max(1, floor(pool / (ceil(65536 / B) + F))) of the boot, launch seqs from SEQS_TABLE per pin step, CAPS16 captures
    "NB5QUALITY_R2": dict(_NB5_T, time_budget_sec=180 * 60, source="eval", task_list=NB5_QAB_TASKS, task_order=NB5_QAB_TASKS,
                          nb5_gate=1.0, nb5_attempt={"b0": 104000, "b1": 104000},
                          nb5_env={"E12_B_EXTRA_ATTEMPTS": "0", "LOCAL_ANALYZER_SEED": str(20261005)},
                          quality_ab=True, equal_windows=True, no_pure_decode=True, admit_mode="slots", f_rule="measured",
                          pin_ladder=True, patch_60490=True, seqs_rule="table", caps16=True,
                          lever_env={"MERLIN_SHARE_DRAFT_WORKSPACE": "1"},
                          continue_after_failed_boot=True,
                          nb5_phases=[["Q_FP8_K3", "all", None, "on", "A", "fp8"], ["Q_BF16_K3", "all", None, "on", "A", "bf16"]]),
}
if E11_HARNESS and os.environ.get("NB5H_TASKS"):                 # local harness only: a few tasks, short windows
    PROFILE_TABLE["NB5HARNESS"] = dict(_NB5_T, source="eval", task_list=os.environ["NB5H_TASKS"].split(","),
                                       nb5_phases=json.loads(os.environ["NB5H_PHASES"]))
if PROFILE not in PROFILE_TABLE:
    raise ValueError(f"unknown profile {PROFILE}; choose {list(PROFILE_TABLE)}")
P = PROFILE_TABLE[PROFILE]
if PROFILE == "NB5QUALITY" and NB5_NB0_FP8 != "PASS":
    # addendum 5: only after NB0's FP8 gate; a clear FP8 FAIL skips the A/B (production then uses BF16)
    raise RuntimeError(f"NB5QUALITY needs NB5_NB0_FP8_GATE=PASS (got {NB5_NB0_FP8!r}): A/B skipped")
assert not any(_p.get("parity") for _p in PROFILE_TABLE.values()), "NB5-2: the E12 DFlash parity path is not part of NB5"
NB5_QAB = ({"nb0": NB5_QAB_NB0, "nb0_gate_env": NB5_NB0_FP8 or None, "nb0_gate_required": False, "fp8_label": NB5_QAB_FP8_LABEL,
            "groups": NB5_QAB_GROUPS, "tasks": NB5_QAB_TASKS} if P.get("quality_ab") else None)   # A-9: recorded, never a gate
for _k, _v in (P.get("nb5_env") or {}).items():
    os.environ.setdefault(_k, _v)                             # read by the E15B block below / by Duck at import
SEED = 20261005
TASK_SAMPLE_SEED = 20261005
POETIQ_NUM_EXPERTS = POETIQ_MAX_ITERATIONS = 0                # (printed by the E10 task-order cell; no Poetiq here)

# ---- time (E15B / E21 CELL 1 block, verbatim)
''' + TIME_BLOCK + r'''

# ---- NB5 engine + workload parameters. Values marked PLACEHOLDER must be replaced by NB1 / NB2 measurements.
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
NB5_KV_DTYPE = os.environ.get("NB5_KV_DTYPE", "fp8").strip().lower()   # PROVISIONAL: RC2 TF FP8-vs-BF16 under study in NB1
if NB5_NB0_FP8 == "FAIL":
    NB5_KV_DTYPE = "bf16"                                     # addendum 5: NB0 failed FP8 clearly -> BF16
assert NB5_KV_DTYPE in ("fp8", "bf16"), NB5_KV_DTYPE
NB5_KV_BYTES = {"fp8": 2479838700, "bf16": 2464415744}         # RC2 pins (Merlin L4 rule); NB1 may change them
# NB1 capacity table (addendum 2), JSON: {"fp8": {"pool_blocks": n, "f_blocks": n, "kv_bytes": n}, "bf16": {...}};
# the live server values (cache_config_info, the max-concurrency line) override it; then the PLACEHOLDERS below
NB5_NB1_CAPACITY = json.loads(os.environ.get("NB5_NB1_CAPACITY_JSON", "{}") or "{}")
NB5_POOL_PLACEHOLDER = {"fp8": 149, "bf16": 78}               # PLACEHOLDER: RC2 (513,940 / 269,042 tok, block 4096)
NB5_F_PLACEHOLDER = 6                                         # PLACEHOLDER = RC2 measured (149 / 3.92 - 32 = 6.0); NB1 refines
NB5_ADMIT_MODE = os.environ.get("NB5_ADMIT_MODE", P.get("admit_mode", "projected"))  # projected (default) | live | future | slots
# R2 'slots' (lead 2026-10-31, simplified): the engine schedules; the reservation is only a safety cap of `workers`
# attempts in flight (1 slot each); C-7 still denies an attempt whose projected blocks exceed the pool
NB5_F_RULE = P.get("f_rule", "max")                          # 'measured' (R2, lever b): F from the server's own line
NB5_SEQS_RULE = P.get("seqs_rule")                           # R2: launch --max-num-seqs from nb5_lib.SEQS_TABLE[kv][pin]
NB5_LEVER_ENV = dict(P.get("lever_env") or {})               # lever b (R2): non-numeric Merlin env; dropped if it fails
# 'future' = a PLACEHOLDER hook for a later NB1-memory-telemetry mode (nb5_lib.future_admission_blocks returns None):
# it behaves exactly as projected now and is recorded as such
assert NB5_ADMIT_MODE in ("projected", "live", "future", "slots"), NB5_ADMIT_MODE
NB5_THINK_FLOOR = int(os.environ.get("NB5_THINK_FLOOR", 60000)) if E11_HARNESS else 60000   # user floor, every request
NB5_CTX_MARGIN = 2048                                         # PLAN v3.6: safe output = min(cap, 131072 - prompt - 2048)
NB5_GATE = float(P.get("nb5_gate", 0.95))                    # one gate: projected blocks <= 0.95 x pool
_att = P.get("nb5_attempt") or {"b0": 112000, "b1": 100000}
NB5_ATTEMPT = {"b0": int(os.environ.get("NB5_ATTEMPT_B0", _att["b0"])), "b1": int(os.environ.get("NB5_ATTEMPT_B1", _att["b1"]))}
NB5_RESERVE = int(os.environ.get("NB5_RESERVE", 4096))        # the commit turn's reserve (E15B #48)
NB5_CAP = NB5_ATTEMPT["b0"] - NB5_RESERVE                     # the longest single request (uninterrupted thinking)
NB5_WINDOW = 131072 - 512                                     # Duck context window = max-model-len - Duck's safety margin
NB5_PROMPT_ALLOW = NB5_TOOL_ALLOW = 8192                      # projected peak = prompt + budget + tool results
NB5_COMMIT_MODE = os.environ.get("NB5_COMMIT_MODE", "")      # "" = per phase (SMOKE split, EVAL C); else C | A | split
NB5_EXTRA_ARGS = json.loads(os.environ.get("NB5_EXTRA_ARGS_JSON", "[]") or "[]")   # NB1 / NB2 levers (trailing vllm args)
if P.get("caps16"):                                           # trailing args win (NB2 CAPS16 rule: FULL covers seqs x (1 + K))
    NB5_EXTRA_ARGS = NB5_EXTRA_ARGS + ["--compilation-config", json.dumps(
        {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": NB5_CAPS16}, separators=(",", ":"))]
NB5_BASE_ENV = json.loads(os.environ.get("NB5_BASE_ENV_JSON", "{}") or "{}")       # Merlin knobs, e.g. {"DRAFT_HEAD": "int8"}
NB5_BOOT_LEVERS = {}                                          # the lever env of the current boot try (set by nb5_boot)
NB5_MTP_OFF_ARGS = ["--speculative-config", "None"]           # vLLM optional_type: "None" -> no speculation (last wins)
NB5_BOOT_EXTRA = []                                           # set per boot by the runner
NB5_READY_TIMEOUT = 30 * 60                                   # spec section 2: readiness gate <= 30 min per boot
NB5_RELAUNCH_MIN_SEC = 30 * 60                                # a Merlin relaunch needs ~17-20 min before it serves
NB5_RAM_NEED_MIB = (69 + 4) * 1024                            # Merlin's floor 69 GiB + 4 GiB margin
NB5_PORT = int(os.environ.get("NB5_PORT", 8001))                # RC2 FP8 port; the local harness picks a free one
if not E11_HARNESS:                                           # user floor: thinking >= 60k (never below without approval)
    assert min(NB5_ATTEMPT.values()) >= 60000, NB5_ATTEMPT
assert NB5_CAP <= 131072 - 512 and max(NB5_ATTEMPT.values()) <= 120000, NB5_ATTEMPT
SGL_MODEL_PATH = INP / "models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1"   # tokenizer
SGL_DRAFT_PATH = Path("/nonexistent/nb5-no-separate-drafter")  # (read by the rev 6 preflight; MTP is built in)
SGL_MAX_RUNNING = 8                                           # Duck env placeholder; the phase sets the real streams
SGL_RELAUNCH_TIMEOUT_SEC = NB5_READY_TIMEOUT
SGL_URL = None
CONFIG = {"PROFILE": PROFILE, "RERUN": RERUN, "profile_row": P}
os.environ["E12_RAW_REASONING_HISTORY"] = "1"                 # E14-P (E15B CELL 1b): Duck keeps the raw reasoning
''' + E12_BLOCK + r'''
for _k in ("E12_B_ATTEMPT_TOKENS", "E12_B_ATTEMPT_TOKENS_B1"):
    assert E11_HARNESS or E15B[_k] >= 60000, (_k, E15B[_k])   # the 60k floor holds after any env override too
E21_VARIANT = "NB5"
E21_SERVED = {"fp8": "q27", "bf16": "q27"}                     # RC2's served model name
E21_VLLM_MAX_MODEL_LEN = 131072
E21_HALF = None
E21_HALF_END_PC = HARD_STOP - E12_STOP_BEFORE_HARD_SEC
E21_STATE = {"halves": {}}
E15B_SOLVE_END_PC = float("inf")


''' + PAYLOAD_FN + r'''


# ---- NB5 library (new code, E25 Build/nb5_lib.py; verbatim)
NB5_LIB_SRC = __NB5LIBPACK__
(TMP / "nb5_lib.py").write_text(NB5_LIB_SRC, encoding="utf-8")
sys.path.insert(0, str(TMP))
import nb5_lib as NB5L
CONFIG["nb5"] = {"kv_dtype": NB5_KV_DTYPE, "kv_bytes": NB5_KV_BYTES, "nb1_capacity": NB5_NB1_CAPACITY,
                 "pool_placeholder": NB5_POOL_PLACEHOLDER, "f_placeholder": NB5_F_PLACEHOLDER, "gate": NB5_GATE,
                 "attempt": NB5_ATTEMPT, "reserve": NB5_RESERVE, "admit_mode": NB5_ADMIT_MODE, "think_floor": NB5_THINK_FLOOR,
                 "smoke_gate_sec": NB5_SMOKE_GATE_SEC, "cap": NB5_CAP, "window": NB5_WINDOW,
                 "prompt_allow": NB5_PROMPT_ALLOW, "tool_allow": NB5_TOOL_ALLOW, "commit_mode": NB5_COMMIT_MODE,
                 "extra_args": NB5_EXTRA_ARGS, "base_env": NB5_BASE_ENV, "phases": P["nb5_phases"],
                 "smoke_tasks": NB5_SMOKE_TASKS, "lib_sha256": hashlib.sha256(NB5_LIB_SRC.encode()).hexdigest(),
                 "payload_used": Q27X_PAYLOAD_USED,   # filled when CELL 3 loads the Duck package
                 "f_rule": NB5_F_RULE, "seqs_rule": NB5_SEQS_RULE, "lever_env": NB5_LEVER_ENV,
                 "pin_ladder": bool(P.get("pin_ladder")), "patch_60490": bool(P.get("patch_60490")),
                 "equal_windows": bool(P.get("equal_windows")), "quality_ab": NB5_QAB}
print("NB5:", json.dumps(CONFIG["nb5"], default=str), flush=True)
'''

CELL3_PATCH = r'''

# ==================================================================== NB5 (E25): commit-turn variant C, request trace, admission
# _E21_CC_ORIG is the name E21's wrapper (installed above as ToolAgent._chat_completion) calls: NB5 rebinds it to a
# dispatcher, so E21's finish_reason restore stays around BOTH paths. A commit request (E15B #48: after the commit text)
# goes as variant C (CONFIG nb5 commit mode; 'split' = a stable (task, branch) hash, the paired SMOKE comparison).
_NB5_CC_BASE = _E21_CC_ORIG                      # Duck's own ToolAgent._chat_completion
NB5_GUARD = {"requests": 0, "tokenized": 0, "estimated": 0, "clamps": 0, "clamped_tokens": 0, "floor_events": 0,
             "events": [], "tokenize_retries": 0, "local_tokenizer": 0, "no_request": {}, "estimates": []}
NB5_LOCAL_TOK = {"verified": False, "why": "not loaded yet"}     # A-14: set in CELL 8 from RC2's pinned-hash check
NB5_LOCAL_MARGIN = 256                           # A-14: + tokens on a local count (the server render is the reference)


class NB5NoRequest(Exception):
    """A request that must not be sent (A-13 / A-14 / C-7); carries the attempt verdict."""
    def __init__(self, verdict, info=None):
        super().__init__(verdict)
        self.verdict, self.info = verdict, info or {}


def _nb5_peak_len(self):
    """C-9: the attempt's own projected peak tokens min(131072, prompt_allow + budget + tool_allow) (CELL 5 capacity)."""
    pk = NB5_BLOCKS.get("peak_b0" if int(getattr(self, "arc2_branch", 0) or 0) == 0 else "peak_b1")
    return int(min(MAX_LEN, pk)) if pk else MAX_LEN


def _nb5_guard(self, messages, kw):
    """Revision 3 + A-13 / A-14 / C-9, on EVERY request (first turn and every tool turn): the server's own render length
    (/tokenize, 2 retries; + 2 for variant C's longer tail); if /tokenize still fails, the LOCAL Flash-Next tokenizer +
    template only when their pinned hashes were verified at load (+ NB5_LOCAL_MARGIN), else NO request
    (INCONCLUSIVE_CONTEXT_LENGTH); every estimate is recorded. max_tokens <= the attempt's projected peak (C-9: so the
    reservation is a true upper bound) - prompt - margin; no room -> NO request (CONTEXT_EXHAUSTED, A-13). The 60k
    thinking floor (nb5_lib.context_guard): an attempt that can no longer reach it ends its thinking path through Duck's
    own budget-exhaustion route."""
    ids, how, tries = None, None, []
    payload = NB5L.chat_payload(_E12_TA, self, messages, kw)
    for _k in range(3):                              # A-14: /tokenize + 2 retries
        try:
            ids = NB5L.tokenize_chat(SGL_URL + "/v1", payload, timeout=120)
            how = "tokenize"
            break
        except Exception as _e:
            tries.append(f"{type(_e).__name__}: {_e}"[:160])
            if _k < 2:
                NB5_GUARD["tokenize_retries"] += 1
                time.sleep(0.25 * (_k + 1))
    NB5_GUARD["requests"] += 1
    est = None
    if ids is not None:
        plen = len(ids) + 2
        NB5_GUARD["tokenized"] += 1
    else:
        try:
            est = int(self._estimate_request_input_tokens(messages, tools=kw.get("tools")))
        except Exception:
            est = None
        local = None
        if NB5_LOCAL_TOK.get("verified"):
            try:
                local = len(NB5L.local_chat_ids(payload, NB5_LOCAL_TOK["hf"], NB5_LOCAL_TOK["template"], NB5_LOCAL_TOK["encode"],
                                                NB5_LOCAL_TOK.get("default_kwargs")))
            except Exception as _e:
                tries.append(f"local render: {type(_e).__name__}: {_e}"[:160])
        NB5_GUARD["estimated"] += 1
        if len(NB5_GUARD["estimates"]) < 500:
            NB5_GUARD["estimates"].append({"task": self.arc2_task_id, "branch": self.arc2_branch, "duck_estimate": est,
                                           "local_tokens": local, "verified": bool(NB5_LOCAL_TOK.get("verified")),
                                           "tries": tries, "t": round(time.time(), 1)})
        if local is None:
            raise NB5NoRequest("INCONCLUSIVE_CONTEXT_LENGTH", {"tries": tries, "duck_estimate": est,
                                                               "local_why": NB5_LOCAL_TOK.get("why")})
        plen, how = local + 2 + NB5_LOCAL_MARGIN, "local_tokenizer"
        NB5_GUARD["local_tokenizer"] += 1
    req = int(kw.get("max_tokens") or self._max_output_tokens)
    inc = bool(getattr(self, "_arc2_b", False) and self._arc2_b_in_commit()) or NB5L.is_commit(self)
    g = NB5L.context_guard(plen, req, int(self.generated_tokens), int(getattr(self, "_arc2_b_budget", 10 ** 9)),
                           int(getattr(self, "_arc2_b_reserve", NB5_RESERVE)), inc, _nb5_peak_len(self), NB5_THINK_FLOOR,
                           NB5_CTX_MARGIN)
    g["how"] = how
    if g["no_room"]:
        raise NB5NoRequest("CONTEXT_EXHAUSTED", dict(g, task=self.arc2_task_id, branch=self.arc2_branch))
    if g["clamped"]:
        kw["max_tokens"] = g["cap"]
        NB5_GUARD["clamps"] += 1
        NB5_GUARD["clamped_tokens"] += g["requested"] - g["cap"]
    if g["floor_event"]:
        NB5_GUARD["floor_events"] += 1
        self._arc2_b_budget = g["new_budget"]          # Duck's own route: last thinking request, then the commit turn
        self._nb5_budget_limited = True                # the attempt's verdict (record + report): BUDGET_LIMITED
    if (g["clamped"] or g["floor_event"]) and len(NB5_GUARD["events"]) < 500:
        NB5_GUARD["events"].append(dict(g, task=self.arc2_task_id, branch=self.arc2_branch, generated=int(self.generated_tokens),
                                        t=round(time.time(), 1)))
    return (ids if how == "tokenize" else None), g


def _nb5_no_request(self, verdict, info):
    """The request is NOT sent. The attempt ends through Duck's own route without a request: its budget is set to what it
    generated and the commit flag is raised, so Duck's done_now() is true at the next should_stop() check (Duck checks it
    before every request); the session cap follows. A synthetic empty reply (0 tokens) goes back to Duck. The verdict is
    the attempt's record verdict; the tracer counts it apart from the requests."""
    NB5_GUARD["no_request"][verdict] = NB5_GUARD["no_request"].get(verdict, 0) + 1
    self._nb5_verdict = getattr(self, "_nb5_verdict", None) or verdict
    self._nb5_budget_limited = True
    n = getattr(self, "_nb5_no_request_n", 0) + 1
    self._nb5_no_request_n = n
    try:
        self._arc2_b_budget = int(self.generated_tokens)
        self._arc2_b_commit_sent = True
        s_ = NB5_ACTIVE.get((self.arc2_task_id, self.arc2_branch))
        if s_ is not None:
            s_.max_generated_tokens = max(1, int(self.generated_tokens))
    except Exception:
        pass
    try:
        NB5_TRACE.note_no_request(self, verdict, info)
    except Exception:
        pass
    if n > 3:                                        # never a silent loop: the attempt ends through Duck's failure path
        raise RuntimeError(f"nb5: {verdict}: no request sent ({n} times)")
    return _E12_TA._ChatCompletionResult(message={"role": "assistant", "content": "", "reasoning": None}, finish_reason="stop",
                                         usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                         served_by="nb5_no_request")


NB5_TRACE = NB5L.Tracer()
NB5_C = {"enabled": True, "mode": "C", "n": 0, "probe": None}
NB5_TAILS = None                                 # (TAIL_A, TAIL_C) from the Flash-Next tokenizer (CELL 5)
NB5_RES = None                                   # the phase's Reservation (CELL 5)
NB5_BLOCKS = {"b0": None, "b1": None, "start": None, "f": None, "block": 4096, "peak_b0": None, "peak_b1": None}
NB5_ACTIVE = {}                                  # (task, branch) -> the running BranchSession (SMOKE early exit)
NB5_EARLY = {"armed": False, "fired": None}
NB5_POOL_CAP = {"blocks": None}                  # 'slots' (R2): the boot's gate x pool, for the C-7 check


def _nb5_cc(self, messages, **kw):
    t0, res, err, info, kind = time.time(), None, None, None, "chat"
    try:
        _ids, _g = _nb5_guard(self, messages, kw)
    except NB5NoRequest as _nr:
        return _nb5_no_request(self, _nr.verdict, _nr.info)
    NB5_TRACE.enter()
    try:
        info = {"guard": _g}
        if NB5L.is_commit(self):
            v = NB5L.commit_variant(NB5_C["mode"], self)
            kind = "commit_A"
            try:
                NB5_TRACE.keep_commit_sample(v, NB5L.chat_payload(_E12_TA, self, messages, kw), self)
            except Exception:
                pass
            if v == "C" and NB5_C["enabled"] and NB5_TAILS:
                NB5_C["n"] += 1
                res, info = NB5L.commit_c_call(_E12_TA, self, messages, kw, SGL_URL + "/v1", NB5_TAILS[0], NB5_TAILS[1],
                                               kw.get("request_timeout_seconds") or self._timeout, NB5_C["n"], ids=_ids)
                info["guard"] = _g
                kind = "commit_C" if res is not None else "commit_A_fallback"
        if res is None:
            res = _NB5_CC_BASE(self, messages, **kw)
        return res
    except BaseException as _e:
        err = _e
        raise
    finally:
        NB5_TRACE.exit()
        try:
            NB5_TRACE.record(self, kind, kw, res, t0, time.time(), error=err, info=info, messages=messages)
            if NB5_ADMIT_MODE == "live" and NB5_RES is not None and res is not None:
                _u = res.usage if isinstance(res.usage, dict) else {}
                _ctx = int(_u.get("prompt_tokens") or 0) + int(_u.get("completion_tokens") or 0)
                NB5_RES.update((self.arc2_task_id, self.arc2_branch),
                               NB5L.cdiv(max(1, _ctx), NB5_BLOCKS["block"]) + NB5_BLOCKS["f"])
        except Exception:
            pass


_E21_CC_ORIG = _nb5_cc
_NB5_RUN0 = E12A.BranchSession.run


def _nb5_session_run(self, pause_at_tokens=None, tag=None):
    """C2: an attempt starts only when its projected peak blocks fit the reservation (deadline = the solve end minus
    the min-admit time); running attempts are never touched; the blocks are released when the attempt's run ends.
    C-7: a projected need above the whole capacity never starts (KV_NEED_ABOVE_POOL, no request). 'slots' (R2): the
    engine schedules; the reservation only caps the attempts in flight at `workers` (1 slot each), C-7 checks the
    projected blocks against NB5_POOL_CAP (the boot's gate x pool)."""
    res = NB5_RES
    if res is None or self.status == "done":
        return _NB5_RUN0(self, pause_at_tokens=pause_at_tokens, tag=tag)
    key = (self.task_id, self.branch)
    if NB5_EARLY.get("fired"):                     # PLAN v3.6: SMOKE passed its gates -> no new attempt
        self.status = "done"
        self.record.update(note="nb5: SMOKE early exit (gates passed)", nb5_reservation="early_exit")
        self.write()
        return self.status
    slots = NB5_ADMIT_MODE == "slots"
    need = (int(NB5_BLOCKS["b0" if self.branch == 0 else "b1"] or 1) if NB5_ADMIT_MODE != "live"
            else int(NB5_BLOCKS["start"] or 1))
    if (need > int(NB5_POOL_CAP["blocks"] or 0)) if slots else res.above_pool(need):
        self.status = "done"
        self.record.update(note="nb5: KV_NEED_ABOVE_POOL (projected need above the whole pool)", nb5_reservation="denied",
                           nb5_verdict="KV_NEED_ABOVE_POOL")
        self.write()
        return self.status
    if not res.acquire(key, 1 if slots else need, self.deadline - float(E12_R6["E12_R6_MIN_ADMIT_SEC"])):
        self.status = "done"
        self.record.update(note="nb5: no KV reservation before the admit deadline", nb5_reservation="denied")
        self.write()
        return self.status
    self.record["nb5_reservation"] = {"blocks": need, "slots": slots, "used_after_grant": res.used}
    NB5_ACTIVE[key] = self
    try:
        return _NB5_RUN0(self, pause_at_tokens=pause_at_tokens, tag=tag)
    finally:
        NB5_ACTIVE.pop(key, None)
        res.release(key)
        _v = getattr(self.agent, "_nb5_verdict", None)
        if _v:
            self.record["nb5_verdict"] = _v                    # CONTEXT_EXHAUSTED / INCONCLUSIVE_CONTEXT_LENGTH / KV_...
        elif getattr(self.agent, "_nb5_budget_limited", False):
            self.record["nb5_verdict"] = "BUDGET_LIMITED"     # published by the scheduler right after this returns


E12A.BranchSession.run = _nb5_session_run
print("NB5: commit dispatcher + request trace + reservation hook installed", flush=True)
'''

ENGINE_HEAD = r'''# ==================================================================== CELL 5 — NB5 engine: Merlin Flash-Next (RC2 FP8 boot path)
# Verbatim donors (build_e25_nb5.py cuts and asserts them): RC2 builder (e23_lab, CELL 1 LAB load + inputs, CELL 2 to the
# stock fallback, CELL 3 prompts, E23 step runner, kv_census / sampled_check / prefill_probe); E21 (metrics, stats
# lines, counters, content / contract / cache probes). New: the NB5 glue at the end of this cell.
import urllib.request as _urlreq, signal as _signal, statistics as _statistics
import tarfile                                   # NB5-FATAL-1: RC2's merlin_install opens the Merlin payload tar
MODE = "FULL"                                   # E23 step runner (NTOK 128); RC2's MODE
FAKE = os.environ.get("E23_FAKE_SERVER") if E11_HARNESS else None
T_START = time.time() - (time.perf_counter() - NOTEBOOK_T0)      # RC2's epoch clock on the Duck notebook's origin
HARD_STOP_EPOCH = T_START + TOTAL_BUDGET_SEC                      # 11:50: the e23 watchdog kills every live server
BUDGET = int(TOTAL_BUDGET_SEC)                                   # RC2's name (its input log line)
MAX_LEN = 131072
ENG_OUT = OUT / "engine"
ENG_OUT.mkdir(parents=True, exist_ok=True)
LOGS = ENG_OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = ENG_OUT / "traces"
TRACES.mkdir(exist_ok=True)
R = {"what": "E25 NB5 FN-Duck engine", "max_len": MAX_LEN, "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()


def save():
    with _SAVE_LOCK:
        tmp = ENG_OUT / "nb5_engine.json.tmp"
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(ENG_OUT / "nb5_engine.json")


def left():
    return HARD_STOP_EPOCH - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"
ENGINE_READY = False; llm = None; engine = None; CONCURRENCY = 0; E11_SERVER = None; E12_VLLM_CONCURRENCY = 0
ENGINE_INFO = {"backend": None}
HELP = {}
# ---- RC2 CELL 1: the lab module + every request traced (verbatim)
'''

NB5_ENGINE = r'''

# ==================================================================== NB5 glue (new)
TEMP = 0.6                                       # RC2's sampled-check temperature (the Duck itself runs at T 1.0)
PREFILL = {"tok_s": 700.0, "measured": False}
NB5_SERVERS = []                                 # with_servers' list; a relaunch replaces the element (same name)
NB5_BOOTS = []
NB5_RC2_FP8_ARGV_TAIL = json.loads(r"""__RC2ARGV__""")
NB5_CAPACITY = {}
NB5_FINAL_KILL = False
NB5_BOOT_KV = NB5_KV_DTYPE
NB5_BOOT_DECODE = {"tok_s": None}               # this boot's pure decode rate (the token-weighted duty factor)
NB5_F_MARGIN = 0                                 # PLAN v3.6: + blocks per admission after a real preemption (carried)


class NB5Server(MServer):
    """The E21 cells call log_text() and kill(fast=...); RC2's MServer has log() and kill()."""
    def log_text(self):
        return self.log()

    def kill(self, fast=False, wait=None):
        super().kill()


MServer = NB5Server


def E11Server(*_a, **_k):
    """NB5-2: E21 CELL 6/7's E12 DFlash parity relaunch (P['parity']) names E11Server; no NB5 profile sets 'parity'
    (asserted in CELL 1), so that path is unreachable. Defined explicitly so it can never become a NameError."""
    raise RuntimeError("NB5-2: the E12 DFlash parity relaunch (E11Server) is not part of NB5")


# ---- vLLM #60490 backport (R2 lever a; NB1's strict applier + the sha-pinned patch, verified before and after)
P60490 = __P60490__
P60490_REL = "vllm/models/qwen4_exp/nvidia/qsa.py"
P60490_PRE, P60490_POST = "__PPRE__", "__PPOST__"
P60490_MARKER = "# E25 NB1: backport of vLLM PR #60490 (issue #60489)."
__APPLIER__


def nb5_apply_60490():
    """After merlin_install, before the first boot (NB1 apply_60490 without the worker extension): the installed tree's
    qsa.py must have Merlin ad42e37's pre-sha; the strict applier's result must have the post-sha; atomic replace; marker,
    py_compile with the Merlin venv and the vllm origin in the tree are checked. Any miss raises (the file is untouched
    unless the post-sha matched): the caller then keeps the RC2 pin (no ladder above it)."""
    if FAKE:
        return {"ok": None, "fake": True, "why": "local harness: no Merlin tree"}
    eng = merlin_install()
    tree, py = Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"
    f = tree / P60490_REL
    raw = f.read_bytes()
    h0 = hashlib.sha256(raw).hexdigest()
    rec = {"file": str(f), "sha_before": h0, "pre_expected": P60490_PRE, "post_expected": P60490_POST}
    if h0 == P60490_POST:
        rec["applied"] = "already_applied"
    elif h0 != P60490_PRE:
        raise RuntimeError(f"PATCH_60490_PRE_SHA_MISMATCH: {h0} (expected Merlin ad42e37 {P60490_PRE})")
    else:
        new = apply_unified(raw.decode("utf-8"), P60490).encode("utf-8")
        if hashlib.sha256(new).hexdigest() != P60490_POST:
            raise RuntimeError("PATCH_60490_POST_SHA_MISMATCH: " + hashlib.sha256(new).hexdigest())
        tmp = f.with_name(f.name + ".nb5tmp")
        tmp.write_bytes(new)
        os.replace(tmp, f)
        rec["applied"] = "applied"
    for pyc in (f.parent / "__pycache__").glob("qsa.*.pyc"):
        pyc.unlink()
    data = f.read_bytes()
    rec.update(sha_after=hashlib.sha256(data).hexdigest(), marker=P60490_MARKER.encode() in data)
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)", str(f)],
                        capture_output=True, text=True, timeout=300)
    rec["py_compile"] = {"rc": cp.returncode, "err": cp.stderr[-400:]}
    envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    envp["PYTHONPATH"] = str(tree)
    cp2 = subprocess.run([py, "-c", "import importlib.util as u; print(u.find_spec('vllm').origin)"], capture_output=True,
                         text=True, timeout=300, env=envp)
    rec["vllm_origin"] = cp2.stdout.strip()[-400:]
    origin_ok = rec["vllm_origin"].startswith(str(tree))
    if rec["sha_after"] != P60490_POST or not rec["marker"] or cp.returncode or not origin_ok:
        raise RuntimeError("PATCH_60490_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("sha_after", "marker", "py_compile",
                                                                                              "vllm_origin")})[:600])
    rec["ok"] = True
    return rec


def nb5_host_floor():
    """KIT-6: the host RAM floor. Real runs: always Merlin's 69 GiB + 4 GiB. FAKE mode only (the CPU-kit fake server;
    the Kaggle CPU kit has a 30 GiB cgroup): NB5_FAKE_HOST_FLOOR_MIB overrides it. Ignored without FAKE."""
    v = os.environ.get("NB5_FAKE_HOST_FLOOR_MIB")
    if FAKE and v:
        try:
            return int(v), "NB5_FAKE_HOST_FLOOR_MIB (fake server only)"
        except ValueError:
            pass
    return NB5_RAM_NEED_MIB, "Merlin 69 GiB + 4 GiB margin"


def e11_kill_server(wait=True):
    """NB5: CELL 7 calls this after every phase; the server must live on for the next phase (the boot's with_servers
    kills it). Counters are snapshot first (E21)."""
    e21_snapshot_counters()
    if NB5_FINAL_KILL and E11_SERVER is not None:
        E11_SERVER.kill(fast=not wait)


def nb5_gate_kv(kv_server):
    """Duck's KV gate value (E15B G4-1): the reservation's projected fraction. Merlin's raw usage gauge also counts the
    retained prefixes of finished turns (RC2: 0.30 with one 47k request; 0.97 at 8k x 8 with 0 preemptions), so the
    raw gauge would hold admissions that fit; it is recorded (kv_usage_server) and drives the live hold instead."""
    return NB5_RES.frac() if NB5_RES is not None else kv_server


def nb5_metrics_text():
    try:
        return _urlreq.urlopen(SGL_URL + "/metrics", timeout=15).read().decode("utf-8", "replace")
    except Exception:
        return ""


def nb5_host_gate(tag):
    """Spec section 2: drop the page cache of the checkpoint files (merged symlinks -> Model mounts), then Merlin's
    floor (max - (current - inactive_file) >= 69 + 4 GiB). Fail closed on a measured shortfall; recorded either way."""
    roots = [CKPT] + [d for d, _ in [v for v in (PARTS or {}).values() if v]] + ([PLE_DIR] if PLE_DIR else [])
    _need, _src = nb5_host_floor()
    rec = NB5L.ram_gate(host_mem, _need, drop_fn=lambda: NB5L.drop_page_cache(roots),
                        wait_sec=0 if FAKE else 120)
    rec.update(tag=tag, floor_source=_src)
    R["setup"].setdefault("host_gates", []).append(rec)
    save()
    return rec


def nb5_capacity(srv, kv):
    """Capacity from measured inputs (addendum 2): the server's caps (max_num_seqs, block, max-concurrency line), the
    usable pool from cache_config_info (live) -> NB1's table -> the gauge quantization -> the PLACEHOLDER; F = max(NB1's F
    or the PLACEHOLDER, F_log). Every source is recorded."""
    caps = NB5L.server_caps(srv.log())
    mt = nb5_metrics_text()
    live = NB5L.pool_from_metrics(mt)
    nb1 = (NB5_NB1_CAPACITY or {}).get(kv) or {}
    if live:
        pool, src = live["usable"], "cache_config_info"
    elif nb1.get("pool_blocks"):
        pool, src = int(nb1["pool_blocks"]), "nb1_table"
    else:
        pool, src = NB5_POOL_PLACEHOLDER[kv], "PLACEHOLDER_rc2"
    f_par = int(nb1.get("f_blocks") or NB5_F_PLACEHOLDER)
    f_src = "nb1_table" if nb1.get("f_blocks") else "PLACEHOLDER_rc2_measured_6"
    block = caps.get("block_size") or 4096
    plan = NB5L.capacity_plan(pool, block, NB5_ATTEMPT, NB5_PROMPT_ALLOW, NB5_TOOL_ALLOW, MAX_LEN, f_par, NB5_GATE,
                              seqs=caps.get("max_num_seqs"), conc_tokens=caps.get("conc_tokens"), conc_x=caps.get("conc_x"),
                              f_rule=NB5_F_RULE)
    plan.update(pool_source=src, f_source=("f_log (server max-concurrency line)" if NB5_F_RULE == "measured" and plan.get("f_log")
                                           is not None else f_src), caps=caps, kv_dtype=kv, live_pool=live, nb1=nb1)
    if NB5_ADMIT_MODE == "slots":                  # R2 (user decision): workers = max_num_seqs = the 64k-context rule
        _wr = NB5L.workers_rule(pool, block, plan["f_eff"])
        _seq = caps.get("max_num_seqs")
        _w = min(_wr, int(_seq)) if _seq else _wr
        plan["r2_workers"] = {"rule": _wr, "rule_text": "max(1, floor(pool / (ceil(65536 / B) + F)))", "pool": pool,
                              "f_eff": plan["f_eff"], "launch_max_num_seqs": _seq, "workers": _w,
                              "rule_vs_launch": None if not _seq else ("EQUAL" if _wr == int(_seq) else
                                                                       ("RULE_BELOW_LAUNCH" if _wr < int(_seq) else "RULE_ABOVE_LAUNCH")),
                              "projected_120k_workers": plan["workers"],
                              "captures_cover_workers": NB5L.captures_cover(NB5_CAPS16 if P.get("caps16") else [32], _w)}
        if plan["workers"] >= 1:                   # C-8 stays: no attempt kind fits the pool -> NO_GO (workers 0)
            plan["workers"] = _w
    if caps.get("max_num_seqs") is None:
        plan["warning"] = "max_num_seqs not found in the server log: workers = pool streams"
    plan["other_dtype_plan"] = {k: NB5L.capacity_plan(NB5_POOL_PLACEHOLDER[k] if not (NB5_NB1_CAPACITY.get(k) or {}).get("pool_blocks")
                                                      else int(NB5_NB1_CAPACITY[k]["pool_blocks"]), block, NB5_ATTEMPT,
                                                      NB5_PROMPT_ALLOW, NB5_TOOL_ALLOW, MAX_LEN, f_par, NB5_GATE,
                                                      seqs=8 if k == "fp8" else 3)["per_attempt"]
                                for k in ("fp8", "bf16")}
    return plan


def nb5_chat(body, timeout=600):
    return _e12v_request_json(SGL_URL + "/v1/chat/completions", payload=body, timeout=timeout)


def nb5_complete(ids, max_tokens, timeout=600, **extra):
    body = dict({"model": E21_SERVED[NB5_KV_DTYPE], "prompt": list(ids), "max_tokens": int(max_tokens), "temperature": 0.0,
                 "stream": False}, **extra)
    return _e12v_request_json(SGL_URL + "/v1/completions", payload=body, timeout=timeout)


def nb5_commit_probe(served):
    """W3 GPU check before the Duck (addendum 1 + 2): a real tool turn, then the commit request rendered by the server
    (/tokenize) -> tail check -> render parity with the chat path (prompt_tokens) -> variant C via /v1/completions (cache
    hit, tool call, think markup, finish stop with a closed think block) -> the SAME ids through the chat path with
    continue_final_message (the server's qwen3 reasoning + qwen3_coder tool parsers) -> parser agreement -> the same C
    ids with a cache_salt (cold prefix) vs warm: recurrent-state correctness under prefix reuse (greedy)."""
    out = {"c_usable": False}
    kw = {"enable_thinking": True, "reasoning_effort": "low"}
    sys_ = {"role": "system", "content": "You are a careful assistant. When asked, use the python tool."}
    u1 = {"role": "user", "content": "NB5 commit probe: use the python tool exactly once to run print(6 * 7). Keep your thinking very short."}
    r1 = nb5_chat({"model": served, "messages": [sys_, u1], "tools": E21_PROBE_TOOLS, "tool_choice": "auto",
                   "temperature": 0.0, "max_tokens": 3072, "chat_template_kwargs": kw})
    m1 = r1["choices"][0]["message"]
    calls = m1.get("tool_calls") or []
    out["turn1"] = {"tool_calls": len(calls), "finish": r1["choices"][0].get("finish_reason"),
                    "usage": r1.get("usage")}
    if not calls:          # thinking ran long: our own scripted turn (the render / cache / parser checks still apply)
        calls = [{"id": "chatcmpl-tool-nb5probe", "type": "function",
                  "function": {"name": "python", "arguments": json.dumps({"code": "print(6 * 7)"})}}]
        m1 = {"content": "", "reasoning": "I will run print(6 * 7) with the python tool."}
        out["turn1"]["scripted"] = True
    hist = [sys_, u1, {"role": "assistant", "content": m1.get("content") or "", "reasoning": m1.get("reasoning") or "",
                       "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": "42"},
            {"role": "user", "content": "Your budget is almost spent. In this reply, call python once with print(7 * 6) and nothing else."}]
    payload = {"model": served, "messages": hist, "tools": E21_PROBE_TOOLS, "chat_template_kwargs": kw}
    ids = NB5L.tokenize_chat(SGL_URL + "/v1", payload)
    ids_c = NB5L.variant_c_ids(ids, NB5_TAILS[0], NB5_TAILS[1])
    out.update(render_tokens=len(ids), tail=ids[-4:], tail_ok=ids_c is not None)
    r2 = nb5_chat({"model": served, "messages": hist, "tools": E21_PROBE_TOOLS, "temperature": 0.0, "max_tokens": 1,
                   "chat_template_kwargs": kw})
    out["render_parity"] = (r2.get("usage") or {}).get("prompt_tokens") == len(ids)
    out["chat_prompt_tokens"] = (r2.get("usage") or {}).get("prompt_tokens")
    if ids_c is None:
        return out
    r3 = nb5_complete(ids_c, 512)
    c3, u3 = r3["choices"][0], r3.get("usage") or {}
    msg3, n3, leak3 = NB5L.parse_c_reply(_E12_TA, c3.get("text") or "")
    out["variant_c"] = {"tool_calls": n3, "finish": c3.get("finish_reason"), "think_markup_in_reply": leak3,
                        "cached": ((u3.get("prompt_tokens_details") or {}).get("cached_tokens")),
                        "prompt": u3.get("prompt_tokens"), "completion": u3.get("completion_tokens"),
                        "args": [(c["function"] or {}).get("arguments") for c in (msg3.get("tool_calls") or [])],
                        "natural_finish": c3.get("finish_reason") == "stop" and not leak3, "text_head": (c3.get("text") or "")[:300]}
    try:          # the server's own parsers on the hand-closed think: continue_final_message with a fixed prefill renders
        # C + the prefill (HF / vLLM trim only the end of a NON-empty final message); the SAME ids go to /v1/completions
        # (Duck's markup parser) and to chat (the server's qwen3 reasoning + qwen3_coder tool parsers), greedy
        pre = {"role": "assistant", "content": "I will call the tool now.", "reasoning": ""}
        tb = {"model": served, "messages": hist + [pre], "tools": E21_PROBE_TOOLS, "chat_template_kwargs": kw,
              "add_generation_prompt": False, "continue_final_message": True}
        ids_cfm = list(_e12v_request_json(SGL_URL + "/tokenize", payload=tb, timeout=120).get("tokens") or [])
        out["cfm_starts_with_c"] = ids_cfm[:len(ids_c)] == ids_c
        if out["cfm_starts_with_c"]:
            r4 = nb5_chat(dict(tb, temperature=0.0, max_tokens=512))
            m4 = r4["choices"][0]["message"]
            a4 = [(c.get("function") or {}).get("arguments") for c in (m4.get("tool_calls") or [])]
            r4c = nb5_complete(ids_cfm, 512)
            m4c, n4c, _ = NB5L.parse_c_reply(_E12_TA, r4c["choices"][0].get("text") or "")
            a4c = [(c.get("function") or {}).get("arguments") for c in (m4c.get("tool_calls") or [])]
            norm_ = lambda xs: [json.loads(x) if isinstance(x, str) and x.strip().startswith("{") else x for x in xs]
            out["server_parsers"] = {"tool_calls": len(a4), "client_tool_calls": n4c, "reasoning_chars": len(m4.get("reasoning") or ""),
                                     "content_head": (m4.get("content") or "")[:200], "finish": r4["choices"][0].get("finish_reason"),
                                     "args_equal_client_parse": norm_(a4) == norm_(a4c)}
    except Exception as _e:
        out["server_parsers"] = {"error": repr(_e)[:300]}
    try:          # recurrent-state correctness under prefix reuse: a cold prefix (cache_salt) vs the warm one, greedy
        r5 = nb5_complete(ids_c, 128, cache_salt=f"nb5-cold-{time.time_ns()}")
        r6 = nb5_complete(ids_c, 128)
        t5, t6 = r5["choices"][0].get("text") or "", r6["choices"][0].get("text") or ""
        cw = ((r6.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        cc = ((r5.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        k = next((i for i, (a, b) in enumerate(zip(t5, t6)) if a != b), None)
        out["cold_vs_warm"] = {"cold_cached": cc, "warm_cached": cw, "equal": t5 == t6, "first_diff_char": k,
                               "verdict": ("INCONCLUSIVE_NO_COLD" if cc else ("EQUAL" if t5 == t6 else "DIFFER"))}
    except Exception as _e:
        out["cold_vs_warm"] = {"error": repr(_e)[:300], "verdict": "INCONCLUSIVE_ERROR"}
    out["c_usable"] = bool(out["tail_ok"] and out["render_parity"])
    return out


def nb5_paired_commit_replay(tag, deadline_pc=None):
    """Addendum 2: the first REAL Duck commit request of the phase (kept by the tracer), replayed greedily as variant A
    (chat, thinking on) and variant C (ids): cache hit, generated tokens, tool call and code validity of each; with an
    MTP-off boot the same C replay is compared later (target-only parity)."""
    samp = NB5_TRACE.commit_samples.get("C") or NB5_TRACE.commit_samples.get("A")
    if not samp:
        return {"skipped": "no commit request in this phase"}
    pl = json.loads(json.dumps(samp["payload"]))
    _left = lambda: (deadline_pc - time.perf_counter()) if deadline_pc else 1200.0
    if _left() < 60:
        return {"skipped": "less than 60 s before the SMOKE gate", "task": samp["task"]}
    out = {"task": samp["task"], "branch": samp["branch"]}
    import ast as _ast

    def _code_ok(args):
        try:
            a = json.loads(args) if isinstance(args, str) else (args or {})
            _ast.parse(a.get("code") or "")
            return bool(a.get("code"))
        except Exception:
            return False
    try:
        ids = NB5L.tokenize_chat(SGL_URL + "/v1", pl)
        ids_c = NB5L.variant_c_ids(ids, NB5_TAILS[0], NB5_TAILS[1])
        if ids_c is not None:
            r = nb5_complete(ids_c, min(int(pl.get("max_tokens") or 2048), 2048), timeout=max(10, _left() / 2))
            c = r["choices"][0]
            m, n, lk = NB5L.parse_c_reply(_E12_TA, c.get("text") or "")
            u = r.get("usage") or {}
            out["C"] = {"cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "prompt": u.get("prompt_tokens"),
                        "completion": u.get("completion_tokens"), "tool_calls": n, "finish": c.get("finish_reason"),
                        "code_valid": any(_code_ok(x["function"].get("arguments")) for x in (m.get("tool_calls") or [])),
                        "text": (c.get("text") or "")[:4000]}
        ra = nb5_chat(dict(pl, temperature=0.0, max_tokens=min(int(pl.get("max_tokens") or 4096), 4096)), timeout=max(10, _left()))
        ca, ua = ra["choices"][0], ra.get("usage") or {}
        ma = ca.get("message") or {}
        out["A"] = {"cached": (ua.get("prompt_tokens_details") or {}).get("cached_tokens"), "prompt": ua.get("prompt_tokens"),
                    "completion": ua.get("completion_tokens"), "tool_calls": len(ma.get("tool_calls") or []),
                    "finish": ca.get("finish_reason"),
                    "code_valid": any(_code_ok((x.get("function") or {}).get("arguments")) for x in (ma.get("tool_calls") or [])),
                    "reasoning_chars": len(ma.get("reasoning") or "")}
        if "C" in out:
            out["same_tool_call_presence"] = bool(out["A"]["tool_calls"]) == bool(out["C"]["tool_calls"])
    except Exception as _e:
        out["error"] = repr(_e)[:400]
    return out


def nb5_warmup(port, W):
    """Addendum 1.3: the JIT-during-inference kernels (RC2: _fill_num_accepted, causal_conv1d_update, gdn_recoverssm
    verify, ple_conv, ple_conv_writeback) compile at the first sampled decode / first concurrent verify. Every metric
    window starts after this warm-up; JIT lines inside a later window are reported per phase."""
    j0 = len(NB5L.jit_kernels(E11_SERVER.log()))
    t0 = time.time()
    rows = []
    for level, band, n_new in ((1, "2k", 256), (W, "8k", 384), (min(W, 2), "40k", 64), (W, "2k", 256)):
        ss = [L.Stream(port, PROMPTS[band][i % len(PROMPTS[band])], n_new, temperature=1.0, seed=900 + i) for i in range(level)]
        for g in ss:
            g.start()
        for g in ss:
            g.join(max(30, min(900, left() - 600)))
        for g in ss:
            if g.is_alive():
                g.stop()
                g.join(10)
        rows.append({"level": level, "band": band, "tokens": [len(g.tokens) for g in ss], "errors": [g.err for g in ss if g.err]})
    ks = NB5L.jit_kernels(E11_SERVER.log())
    return {"sec": round(time.time() - t0, 1), "rows": rows, "jit_before": j0, "jit_after": len(ks), "jit_kernels": ks,
            "log_offset_after": len(E11_SERVER.log())}


def nb5_pure_decode(port, W, plan):
    """Pure decode: L.steady (the window starts only after prefill settled; STEADY needs running == level the whole
    window) at the Duck's own sampling (T 1.0) and level = workers, 40k prompts when they fit, else 8k."""
    band = "40k" if W * (NB5L.cdiv(40000 + 8192, plan["block_size"]) + plan["f_eff"]) <= plan["capacity_blocks"] else "8k"
    r = L.steady(port, PROMPTS[band], W, secs=WIN, max_new=MAX_NEW[band], temperature=1.0, prefill_tok_s=PREFILL["tok_s"])
    r["band"] = band
    r["engine_rates"] = NB5L.engine_rates(r, pp=2)
    r["note"] = ("decode-rate window on RC2's ARC-text prompts (concatenated real tasks): a rate, never an acceptance / K "
                 "decision input (PLAN never-do: synthetic prompts for acceptance)")
    return r


def nb5_probes(srv):
    """The short correctness gates before any Duck work (RC2 + E21): census, thinking-off arithmetic, a sampled request,
    content probes, the tool contract (history 'reasoning' key, turn-2 growth, cache probe), the commit probe.
    Returns (ok, record); garbage / dead is fatal everywhere, a tool-contract issue only outside the rerun."""
    o = {"census": kv_census(srv), "probes": L.chat_probes(srv.port), "sampled": sampled_check(srv)}
    o["fatal"] = None
    if (not FAKE and o["probes"].get("ok") is not True) or o["sampled"].get("ok") is not True or o["sampled"].get("garbage"):
        o["fatal"] = "SHORT_CORRECTNESS_GATE_FAILED"
        return False, o
    served = E21_SERVED[NB5_KV_DTYPE]
    try:
        o["content_probes"] = e21_content_probes(served)
    except Exception as _e:
        o["content_probes"] = {"error": repr(_e)[:600]}
        o["fatal"] = "CONTENT_PROBES_FAILED"
        return False, o
    try:
        o["contract"] = e21_contract_probe(served)
    except Exception as _e:
        o["contract"] = {"error": repr(_e)[:600]}
        if not RERUN:
            o["fatal"] = "CONTRACT_PROBE_FAILED"
            return False, o
    try:
        o["commit_probe"] = nb5_commit_probe(served)
    except Exception as _e:
        o["commit_probe"] = {"error": repr(_e)[:600], "c_usable": False}
    NB5_C["enabled"] = bool((o["commit_probe"] or {}).get("c_usable"))
    NB5_C["probe"] = o["commit_probe"]
    if not PREFILL["measured"]:
        o["prefill"] = prefill_probe(srv.port)
    return True, o


def nb5_reset_duck_state():
    """E21 gotchas: Duck's priority gate is a process-wide singleton built with ARC3_MAX_ACTIVE_STREAMS (fresh per
    phase); the scheduler / server-down globals of arc2_duck are cleared."""
    with _E12_TA._PRIORITY_GATE_LOCK:
        _E12_TA._PRIORITY_GATE = None
    _E12_TA._GATE_STATS.update(enqueued=0, blocked=0, blocked_ms=0, handover_ms=0)
    E12A.SERVER_DOWN.clear(); E12A.SERVER_WAITS.update(waits=0, sec=0.0); E12A.SERVER_LAST.clear(); E12A.SCHED_LAST.clear()
    os.environ["ARC3_REASONING_HISTORY_KEY"] = "reasoning"
    os.environ["LOCAL_ANALYZER_BASE_URL"] = os.environ["OPENAI_BASE_URL"] = SGL_URL + "/v1"
    os.environ["ARC3_MAX_ACTIVE_STREAMS"] = str(int(CONCURRENCY))


def nb5_sampler(stop, rows, path, workers):
    """Every 15 s during a phase: server running / waiting / raw KV gauge / generated / preemptions, the client's
    in-flight requests and the reservation. Live observation (addendum 2): server waiting > 0 while the client has no
    more requests in flight than run -> the reservation holds new grants (two samples in a row); a real preemption ->
    F + 1 for later admissions. Never touches running work."""
    last_pre, bad = None, 0
    while not stop.wait(float(os.environ.get("NB5_SAMPLER_SEC", "15"))):
        try:
            rr = e21_prom()
            g = lambda n: sum(float(x.value) for x in rr if x.name == n)
            s = {"t": round(time.time(), 1), "running": int(g("vllm:num_requests_running")), "waiting": int(g("vllm:num_requests_waiting")),
                 "kv_server": round(max([float(x.value) for x in rr if x.name == "vllm:kv_cache_usage_perc"] or [0.0]), 4),
                 "gen": g("vllm:generation_tokens_total"), "preempt": g("vllm:num_preemptions_total")}
        except Exception as _e:
            s = {"t": round(time.time(), 1), "error": repr(_e)[:120]}
        s["inflight"] = NB5_TRACE.inflight
        s["workers"] = workers
        if NB5_RES is not None:                   # PLAN v3.6: does Merlin's defer and Duck's projected gate restrict each other?
            _need = min(int(NB5_BLOCKS["b0"] or 1), int(NB5_BLOCKS["b1"] or 1))
            _done = sum(1 for v in (globals().get("E12_RECORDS") or {}).values() for r in v if r.get("status") == "done")
            s.update(duck_waiting=NB5_RES.waiting, duck_held=len(NB5_RES.held), free_blocks=NB5_RES.cap - NB5_RES.used,
                     work_left=(_done + len(NB5_RES.held)) < 2 * len(TASK_IDS))
            s["free_not_admitted"] = bool(s["free_blocks"] >= _need and s["duck_held"] < workers and s["work_left"]
                                          and not NB5_EARLY.get("fired") and not NB5_RES.waiting)
        if NB5_EARLY.get("armed") and not NB5_EARLY.get("fired"):
            try:
                if nb5_smoke_ready(s):
                    nb5_smoke_early_exit()
            except Exception as _e:
                NB5_EARLY["error"] = repr(_e)[:200]
        if NB5_RES is not None:
            s["reserved"] = NB5_RES.used
            if "waiting" in s and NB5_ADMIT_MODE != "slots":     # R2 'slots': the engine schedules (no client hold)
                bad = bad + 1 if (s["waiting"] > 0 and s["inflight"] <= s["running"] + s["waiting"] and s["running"] < s["inflight"]) else 0
                NB5_RES.set_live_hold(f"server waiting {s['waiting']} running {s['running']} in-flight {s['inflight']}" if bad >= 2 else None)
            if "preempt" in s:
                if last_pre is not None and s["preempt"] > last_pre and NB5_ADMIT_MODE == "slots":
                    NB5_EARLY["preempted"] = True          # R2: preemptions allowed (counted, speed only); no F correction
                elif last_pre is not None and s["preempt"] > last_pre:
                    NB5_RES.correct(1, f"server preemptions {last_pre} -> {s['preempt']}")
                    NB5_EARLY["preempted"] = True
                last_pre = s["preempt"]
        rows.append(s)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(s) + "\n")
        except Exception:
            pass


def nb5_smoke_ready(sample):
    """PLAN v3.6 SMOKE early exit: the functional checks already passed before the phase (probes, render parity, one
    short tool turn); the task-level gates: requests served without errors, >= 1 attempt finished, >= 1 variant-C
    commit turn with a tool call (unless C is disabled), no preemption, the server alive."""
    tr = NB5_TRACE.summary()
    done = [r for v in (globals().get("E12_RECORDS") or {}).values() for r in v if r.get("status") == "done"]
    cv = (tr.get("commit_by_variant") or {}).get("C") or {}
    return bool(tr.get("requests") and not tr.get("errors") and done and (cv.get("with_tool_call") or not NB5_C["enabled"])
                and not NB5_EARLY.get("preempted") and E11_SERVER is not None and E11_SERVER.alive())


def nb5_smoke_early_exit():
    """No new attempt (the reservation hook refuses), and every running attempt ends at its next request boundary
    (Duck's own done_now: generated >= the session cap; later requests clamped to now). In-flight requests finish."""
    NB5_EARLY["fired"] = round(time.perf_counter() - NOTEBOOK_T0, 1)
    for s_ in list(NB5_ACTIVE.values()):
        try:
            s_.max_generated_tokens = max(1, int(getattr(s_.agent, "generated_tokens", 0) or 0))
            if s_.agent is not None:
                s_.agent._arc2_deadline = time.monotonic()
        except Exception:
            pass
    log("NB5 SMOKE early exit: gates passed at", NB5_EARLY["fired"], "s")


def nb5_interaction(rows, interval):
    """Per phase (PLAN v3.6): the server waiting while Duck holds attempts, free projected capacity with work left but no
    admission, Duck holding with the server idle-able; in samples and seconds."""
    rr = [r for r in rows if "duck_waiting" in r and "waiting" in r]
    f = lambda c: sum(1 for r in rr if c(r))
    out = {"samples": len(rr), "interval_sec": interval,
           "server_waiting": f(lambda r: r["waiting"] > 0),
           "duck_holding": f(lambda r: r["duck_waiting"] > 0),
           "server_waiting_while_duck_holds": f(lambda r: r["waiting"] > 0 and r["duck_waiting"] > 0),
           "free_projected_capacity_no_admission": f(lambda r: r.get("free_not_admitted"))}
    out.update({k + "_sec": round(v * interval, 1) for k, v in list(out.items()) if k not in ("samples", "interval_sec")})
    return out


def nb5_qab_phase_record(name, rep, rows, preemptions):
    """R2 per phase: the actual solve window, the capacity levers of this boot (pool blocks, pin, F, workers, max_num_seqs,
    the lever env), speed (max streams reached, mean running, preemptions -> speed PREEMPTED, accuracy untouched,
    tok/s), the per-task accuracy table (nb5_lib.qab_tasks) and its per-group summary, the FP8 label."""
    run = dict(E12_RUN or {})
    ws = (run.get("window_shift") or {}).get("window_start_sec") or run.get("started_sec")
    we_pc = E15B_SOLVE_END_PC if E15B_SOLVE_END_PC != float("inf") else None
    to_epoch = lambda pc: time.time() - (time.perf_counter() - pc)
    ws_ep = to_epoch(NOTEBOOK_T0 + float(ws)) if ws is not None else None
    we_ep = to_epoch(we_pc) if we_pc else None
    win = {"start_sec": ws, "end_sec": round(we_pc - NOTEBOOK_T0) if we_pc else None,
           "actual_sec": round(we_pc - NOTEBOOK_T0 - float(ws), 1) if (we_pc and ws is not None) else None,
           "planned_sec": (NB5_REPORT["phases"].get(name, {}).get("window_plan") or {}).get("planned_sec"),
           "solver_returned_sec": run.get("finished_sec")}
    b = NB5_BOOTS[-1] if NB5_BOOTS else {}
    tr = (b.get("tries") or [{}])[-1]
    cap = NB5_CAPACITY or {}
    busy = [r_ for r_ in rows if isinstance(r_.get("running"), int) and (r_.get("inflight") or 0) > 0]
    tasks = NB5L.qab_tasks(E12_RECORDS, SOLUTIONS, rep.get("eval_unit_records"), NB5_QAB_GROUP_OF, P["task_list"],
                           {"b0": E15B["E12_B_ATTEMPT_TOKENS"], "b1": E15B["E12_B_ATTEMPT_TOKENS_B1"]}, ws_ep, we_ep,
                           (NB5_TRACE.summary() or {}).get("by_attempt"))
    return {"window": win, "order": rep.get("order"),
            "fp8_status": NB5_QAB_FP8_LABEL if NB5_BOOT_KV == "fp8" else "BF16_REFERENCE",
            "capacity_levers": {"pool_blocks": cap.get("pool_blocks"), "pool_source": cap.get("pool_source"),
                                "capacity_blocks": cap.get("capacity_blocks"), "f_eff": cap.get("f_eff"), "f_log": cap.get("f_log"),
                                "f_source": cap.get("f_source"), "pin_gib": tr.get("pin_gib"), "kv_bytes": tr.get("kv_bytes"),
                                "lever_env": tr.get("levers"), "max_num_seqs_launch": tr.get("max_num_seqs_launch"),
                                "workers": int(CONCURRENCY), "r2_workers": cap.get("r2_workers"),
                                "captures": NB5_CAPS16 if P.get("caps16") else None,
                                "ladder_tries": [{"name": t_.get("name"), "status": t_.get("status"),
                                                  "failed": (t_.get("ladder") or {}).get("failed")} for t_ in b.get("tries") or []],
                                "levers_dropped": b.get("levers_dropped"), "config_readback": (tr.get("config_readback") or {}).get("ok")},
            "speed": {"max_running": max([r_["running"] for r_ in rows if isinstance(r_.get("running"), int)] or [0]),
                      "max_inflight": max([int(r_.get("inflight") or 0) for r_ in rows] or [0]),
                      "mean_running_busy": round(sum(r_["running"] for r_ in busy) / len(busy), 2) if busy else None,
                      "preemptions": int(preemptions or 0), "speed_status": "PREEMPTED" if preemptions else "CLEAN",
                      "accuracy_affected_by_preemption": False,
                      "solver_tok_s": (NB5_REPORT["phases"].get(name, {}).get("solver_window") or {}).get("tok_s"),
                      "phase_tok_s": (NB5_REPORT["phases"].get(name, {}).get("phase_window") or {}).get("tok_s")},
            "qab": {"tasks": tasks, "summary": NB5L.qab_summary(tasks)}}


def nb5_phase(spec, mtp_tag):
    """One Duck run on the live server: fresh Duck state (workload reset), counters / log offset at the start
    (timestamp-delta metric reset), E15B CELL 6 + 7 (E21's wrapped functions), the NB5 phase record."""
    global TASK_IDS, CHALLENGES, SOLUTIONS, OUT, E21_HALF, E21_HALF_END_PC, E12_SOLVE_WINDOW_SEC, E15B_SOLVE_END_PC, NB5_RES
    global NB5_F_MARGIN, E12_MAX_GEN_PER_BRANCH, CONCURRENCY, E12_STREAMS, E12_VLLM_CONCURRENCY, E12_RECORDS
    name, sel, window, _mtp, cmode = spec[:5]
    opts = (spec[6] if len(spec) > 6 else None) or {}
    rec = NB5_REPORT["phases"].setdefault(name, {"spec": spec, "boot": mtp_tag})
    ids = [t for t in NB5_SMOKE_TASKS if t in NB5_EVAL_CH] if sel == "smoke" else list(NB5_ALL_IDS)
    if not ids:
        rec["status"] = "SKIPPED_NO_TASKS"
        return rec
    if P.get("equal_windows") and sel != "smoke":
        # R2: equal solve windows, fixed when the FIRST paired phase starts (after its boot: a slow first boot shortens
        # every window equally); a later phase gets the same window unless the hard stop leaves less (recorded UNEQUAL)
        if NB5_EQ["window_sec"] is None:
            _left_ph = [s_[0] for s_ in P["nb5_phases"] if s_[0] == name or s_[0] not in NB5_REPORT["phases"]]
            NB5_EQ["plan"] = NB5L.equal_windows(time.perf_counter(), HARD_STOP - E12_STOP_BEFORE_HARD_SEC,
                                                time.perf_counter() - float(NB5_EQ["try_t0_pc"] or time.perf_counter()),
                                                float(os.environ.get("NB5_PHASE_TAIL_SEC", 300)) if E11_HARNESS else 300.0,
                                                max(1, len(_left_ph)))
            NB5_EQ["window_sec"] = NB5_EQ["plan"]["window_sec"]
        window = float(NB5_EQ["window_sec"])
        _room = (HARD_STOP - E12_STOP_BEFORE_HARD_SEC) - time.perf_counter() - 60.0
        rec["window_plan"] = {"planned_sec": window, "room_sec": round(_room, 1), "plan": NB5_EQ["plan"], "unequal": False}
        if _room < window:
            rec["window_plan"].update(unequal=True, short_by_sec=round(window - _room, 1))
            window = max(0.0, _room)
        if window < (60 if E11_HARNESS else 5 * 60):
            rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), window_sec=round(window, 1))
            return rec
    else:
        need = (window or 0) + 10 * 60
        if left() < need + 5 * 60:
            rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), need_min=round(need / 60, 1))
            return rec
    src_ch, src_sol = (NB5_EVAL_CH, NB5_EVAL_SOL) if sel == "smoke" else (NB5_ALL_CH, NB5_ALL_SOL)
    TASK_IDS = sorted(ids)
    CHALLENGES = {t: src_ch[t] for t in TASK_IDS}
    SOLUTIONS = {t: src_sol[t] for t in TASK_IDS} if src_sol is not None else None
    OUT = BASE_OUT / f"phase_{name}"
    OUT.mkdir(parents=True, exist_ok=True)
    E21_HALF, E21_HALF_END_PC, E12_SOLVE_WINDOW_SEC = name, HARD_STOP - E12_STOP_BEFORE_HARD_SEC, window
    E15B_SOLVE_END_PC = float("inf")
    nb5_reset_duck_state()
    NB5_C["mode"] = NB5_COMMIT_MODE or cmode
    _base = (dict(E15B), E12_MAX_GEN_PER_BRANCH, dict(NB5_BLOCKS), CONCURRENCY)
    if opts.get("budget"):                         # PLAN v3.6 NB5BUDGET: this phase's attempt budget (both attempts)
        _bud = int(opts["budget"])
        assert E11_HARNESS or _bud >= 60000, _bud
        E15B["E12_B_ATTEMPT_TOKENS"] = E15B["E12_B_ATTEMPT_TOKENS_B1"] = E12_MAX_GEN_PER_BRANCH = _bud
        os.environ["E12_B_ATTEMPT_TOKENS"] = os.environ["E12_B_ATTEMPT_TOKENS_B1"] = str(_bud)
        _pl = NB5L.capacity_plan(NB5_CAPACITY["pool_blocks"], NB5_CAPACITY["block_size"], {"b0": _bud, "b1": _bud},
                                 NB5_PROMPT_ALLOW, NB5_TOOL_ALLOW, MAX_LEN, NB5_CAPACITY["f_eff"], NB5_GATE,
                                 seqs=(NB5_CAPACITY.get("caps") or {}).get("max_num_seqs"))
        NB5_BLOCKS.update(b0=_pl["per_attempt"]["b0"]["blocks"], b1=_pl["per_attempt"]["b1"]["blocks"])
        CONCURRENCY = E12_STREAMS = E12_VLLM_CONCURRENCY = int(_pl["workers"])
        rec["budget_plan"] = {"budget": _bud, "workers": _pl["workers"], "blocks": _pl["per_attempt"]["b0"]["blocks"]}
        if int(_pl["workers"]) < 1:                # C-8: this budget fits no stream -> no solver for this phase
            rec.update(status="NO_GO_INSUFFICIENT_KV", budget_plan=dict(rec["budget_plan"], verdict=_pl.get("verdict")))
            E15B.update(_base[0])
            E12_MAX_GEN_PER_BRANCH = _base[1]
            NB5_BLOCKS.update(_base[2])
            CONCURRENCY = E12_STREAMS = E12_VLLM_CONCURRENCY = _base[3]
            os.environ["E12_B_ATTEMPT_TOKENS"] = str(_base[0]["E12_B_ATTEMPT_TOKENS"])
            os.environ["E12_B_ATTEMPT_TOKENS_B1"] = str(_base[0]["E12_B_ATTEMPT_TOKENS_B1"])
            return rec
        nb5_reset_duck_state()
    # 'slots' (R2): the reservation is only a safety cap of `workers` attempts in flight (the engine schedules)
    NB5_RES = NB5L.Reservation(int(CONCURRENCY) if NB5_ADMIT_MODE == "slots" else NB5_CAPACITY["capacity_blocks"])
    NB5_RES.f_extra = NB5_F_MARGIN                 # PLAN v3.6: a preemption margin carries into later phases
    NB5_EARLY.update(armed=bool(opts.get("early_exit")), fired=None, preempted=False)
    NB5_TRACE.new_phase(name, OUT / "duck")
    c0, off0, t0 = e21_counters(), len(E11_SERVER.log()), time.time()
    h0 = NB5L.hist_sums(e21_prom())
    g0 = dict(NB5_GUARD, events=[])
    _pc0 = time.perf_counter()
    # revision 4: SMOKE's solve window ends >= 3 min before its 40-min gate (the replay fits in between); never past the
    # notebook's own solve end
    _gate_pc = min(_pc0 + NB5_SMOKE_GATE_SEC, HARD_STOP - E12_STOP_BEFORE_HARD_SEC) if name == "SMOKE" else None
    if name == "SMOKE":
        window = max(60.0, min(float(window or NB5_SMOKE_GATE_SEC), _gate_pc - _pc0 - 180.0))
        E12_SOLVE_WINDOW_SEC = window
        rec["smoke_gate_deadline_min"] = round((_gate_pc - _pc0) / 60, 1)
    E12_RECORDS = {}                               # the sampler never reads the previous phase's records
    rows, stop = [], threading.Event()
    th = threading.Thread(target=nb5_sampler, args=(stop, rows, OUT / "nb5_sampler.jsonl", int(CONCURRENCY)), daemon=True,
                          name="nb5_sampler")
    th.start()
    rec.update(status="RUNNING", tasks=len(TASK_IDS), start_sec=round(time.perf_counter() - NOTEBOOK_T0), commit_mode=NB5_C["mode"],
               commit_c_enabled=NB5_C["enabled"], workers=int(CONCURRENCY))
    try:
        e21_cell6()
        e21_cell7()
        rec["status"] = "DONE"
    except BaseException:
        rec.update(status="ERROR", error=traceback.format_exc()[-3000:])
    finally:
        stop.set()
        th.join(timeout=30)
        c1, t1 = e21_counters(), time.time()
        txt = E11_SERVER.log()
        sl = txt[off0:]
        run = dict(E12_RUN or {})
        solve_sec = (run.get("finished_sec") or 0) - (run.get("started_sec") or 0)
        bs = run.get("b_summary") or {}
        rep = {}
        try:
            rep = json.loads((OUT / "e12_run_report.json").read_text(encoding="utf-8"))
        except Exception as _e:
            rep = {"error": repr(_e)[:200]}
        rec.update(end_sec=round(time.perf_counter() - NOTEBOOK_T0), counters=NB5L.counter_delta(c0, c1),
                   tracer=NB5_TRACE.summary(), reservation=NB5_RES.snapshot(),
                   running_check=NB5L.running_check(rows, int(CONCURRENCY)),
                   steady_mixed=NB5L.steady_mixed(e21_stat_rows(sl), int(CONCURRENCY)),
                   jit_kernels_in_phase=NB5L.jit_kernels(sl), deferred=NB5L.deferred_lines(sl),
                   latency_means=NB5L.hist_delta(h0, NB5L.hist_sums(e21_prom())),
                   solver_window={"solve_sec": solve_sec, "server_gen_tokens": bs.get("server_gen_tokens"),
                                  "tok_s": NB5L.rate(bs.get("server_gen_tokens"), solve_sec)},
                   phase_window={"sec": round(t1 - t0, 1), "gen_tokens": NB5L.counter_delta(c0, c1).get("vllm:generation_tokens_total"),
                                 "tok_s": NB5L.rate(NB5L.counter_delta(c0, c1).get("vllm:generation_tokens_total"), t1 - t0)},
                   bands=NB5L.solve_bands(E12_RECORDS, SOLUTIONS),
                   task_solve=NB5L.task_solve_times(E12_RECORDS, SOLUTIONS), kv_dtype=NB5_BOOT_KV,
                   quality=(NB5L.quality_table(E12_RECORDS, SOLUTIONS, NB5_TRACE.summary().get("turns"),
                                               ((NB5L.hist_delta(h0, NB5L.hist_sums(e21_prom())).get("time_to_first_token_seconds") or {})
                                                .get("mean_sec"))) if name.startswith("Q_") else None),
                   score={k: rep.get(k) for k in ("eval_query_hits", "eval_query_total", "eval_pass1_hits", "eval_pass2_hits",
                                                  "eval_task_norm_pct")},
                   b_summary={k: bs.get(k) for k in ("attempts", "attempt2_started", "accepted", "at_budget", "commit_turns",
                                                     "commit_with_tool_call", "weak_accepts", "server_running_mean",
                                                     "server_gen_tokens", "counted_gen_tokens", "extra_started")},
                   rev6_workers=run.get("rev6_workers"), streams=run.get("streams"), solver_error=run.get("error"),
                   samples=len(rows),
                   guard={k: NB5_GUARD[k] - g0[k] for k in ("requests", "tokenized", "estimated", "clamps", "clamped_tokens",
                                                            "floor_events")} | {"events": NB5_GUARD["events"][-50:]},
                   duty=NB5L.duty_factor(sl, t1 - t0, NB5L.counter_delta(c0, c1).get("vllm:generation_tokens_total"),
                                         NB5_BOOT_DECODE.get("tok_s")),
                   interaction=nb5_interaction(rows, float(os.environ.get("NB5_SAMPLER_SEC", "15"))),
                   budget_limited_attempts=sum(1 for v in (E12_RECORDS or {}).values() for r in v
                                               if r.get("nb5_verdict") == "BUDGET_LIMITED"),
                   early_exit={k: NB5_EARLY.get(k) for k in ("armed", "fired", "preempted", "error")},
                   paired_split=(NB5L.variant_outcomes(E12_RECORDS, SOLUTIONS, NB5_C["mode"]) if NB5_C["mode"] == "split" else None))
        _pre = int(rec["counters"].get("vllm:num_preemptions_total") or 0)
        if _pre and name == "SMOKE":
            NB5_F_MARGIN += 1                          # PLAN v3.6 gate: any SMOKE preemption -> +1 F for every later admission
        NB5_F_MARGIN = max(NB5_F_MARGIN, NB5_RES.f_extra)
        rec["preemption_gate"] = {"preemptions": _pre, "verdict": "PREEMPTED" if _pre else "PASS",
                                  "f_margin_after": NB5_F_MARGIN}
        if P.get("quality_ab") and name.startswith("Q_"):
            try:
                rec.update(nb5_qab_phase_record(name, rep, rows, _pre))
            except Exception as _e:
                rec["qab_error"] = traceback.format_exc()[-1500:]
        if opts.get("budget"):                     # restore the base budgets / capacity for any later phase
            E15B.update(_base[0])
            E12_MAX_GEN_PER_BRANCH = _base[1]
            NB5_BLOCKS.update(_base[2])
            CONCURRENCY = E12_STREAMS = E12_VLLM_CONCURRENCY = _base[3]
            os.environ["E12_B_ATTEMPT_TOKENS"] = str(_base[0]["E12_B_ATTEMPT_TOKENS"])
            os.environ["E12_B_ATTEMPT_TOKENS_B1"] = str(_base[0]["E12_B_ATTEMPT_TOKENS_B1"])
        NB5_EARLY.update(armed=False)
        try:
            rec["paired_commit_replay"] = (nb5_paired_commit_replay(name, _gate_pc) if name == "SMOKE" or mtp_tag.endswith("nomtp")
                                           else None)
        except Exception as _e:
            rec["paired_commit_replay"] = {"error": repr(_e)[:300]}
        NB5_RES = None
        if name == "SMOKE":
            rec["smoke_gate_at_min"] = round((time.perf_counter() - _pc0) / 60, 1)
            rec["smoke_gate_within_bound"] = time.perf_counter() <= _gate_pc + 30
        nb5_write_report()
    return rec


def nb5_smoke_gate(rec):
    """SMOKE -> EVAL (fail closed): the phase ran, requests were served, the server lived, no preemption, the request
    error share < 50 %. A broken commit variant C (>= 4 C commits, none with a tool call) switches EVAL to variant A."""
    tr = rec.get("tracer") or {}
    why = []
    if rec.get("status") != "DONE":
        why.append(f"status {rec.get('status')}")
    if not tr.get("requests"):
        why.append("no request served")
    elif tr.get("errors", 0) / max(1, tr["requests"]) >= 0.5:
        why.append(f"errors {tr['errors']} of {tr['requests']}")
    if (rec.get("preemption_gate") or {}).get("verdict") == "PREEMPTED":
        rec["preemption_action"] = f"+1 F margin for every later admission (now {NB5_F_MARGIN})"   # reported, not an abort
    if E11_SERVER is None or not E11_SERVER.alive():
        why.append("server not alive")
    cv = (tr.get("commit_by_variant") or {}).get("C") or {}
    av = (tr.get("commit_by_variant") or {}).get("A") or {}
    sp = rec.get("paired_split") or {}
    # PLAN v3.6: the C -> A decision = the >= 4-commits-without-a-call safety trigger, OR corruption only under C
    # (unparsable code / think markup in a closed-think reply / non-grid candidates), OR >= 2 fewer exact-match attempts
    # under C with >= 3 commit-turn attempts in each variant (small samples: recorded either way)
    reasons = []
    if cv.get("requests", 0) >= 4 and not cv.get("with_tool_call"):
        reasons.append("no tool call in any of >= 4 variant-C commit requests")
    c_bad = cv.get("corrupt", 0) + (sp.get("C") or {}).get("invalid_grid_candidates", 0)
    a_bad = av.get("corrupt", 0) + (sp.get("A") or {}).get("invalid_grid_candidates", 0)
    if c_bad and not a_bad:
        reasons.append(f"corruption under C only ({c_bad})")
    if ((sp.get("C") or {}).get("attempts", 0) >= 3 and (sp.get("A") or {}).get("attempts", 0) >= 3
            and sp["C"]["exact"] <= sp["A"]["exact"] - 2):       # a 1-attempt difference on 3 + 3 is noise
        reasons.append(f"exact-match C {sp['C']['exact']}/{sp['C']['attempts']} vs A {sp['A']['exact']}/{sp['A']['attempts']}")
    rec["variant_decision"] = {"C": {"commit": cv, "split": sp.get("C"), "corrupt": c_bad},
                               "A": {"commit": av, "split": sp.get("A"), "corrupt": a_bad}, "switch_reasons": reasons}
    if reasons:
        NB5_C["enabled"] = False
        rec["commit_c_disabled_for_eval"] = "; ".join(reasons)
    if rec.get("smoke_gate_within_bound") is False:
        why.append("SMOKE gate passed its 40-min bound")
    rec["gate_decision"] = "ABORT" if why else ("EVAL_SWITCH_TO_A" if rec.get("commit_c_disabled_for_eval") else "EVAL")
    return why


def e21_relaunch():
    """E15B's watchdog relaunch on Merlin (NB5): the dead / hung server is killed, the GPUs released, the host gate
    passed, the same plan booted under the same name (its old log is copied first), at most once per boot, only with
    >= NB5_RELAUNCH_MIN_SEC before the solve end. The new server replaces the element of with_servers' list."""
    global E11_SERVER, SGL_URL
    t = globals().get("E15B_SOLVE_END_PC", float("inf")) - time.perf_counter() - 5 * 60
    if t < NB5_RELAUNCH_MIN_SEC:
        return False, f"only {round(t)} s left for a relaunch"
    e21_snapshot_counters()
    old = E11_SERVER
    n = len(NB5_REPORT.setdefault("relaunches", []))
    try:
        cp = old.log_path.with_name(old.log_path.stem + f"_dead{n}.log")
        shutil.copyfile(old.log_path, cp)
        wd = (E12_RUN or {}).get("watchdog") or {}
        if wd.get("old_logs"):
            wd["old_logs"][-1] = str(cp)
        old.kill()
        rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus="0,1,2,3", free_mib=150)
        gate = nb5_host_gate(f"relaunch{n}")
        if rel is False or gate.get("ok") is False:
            NB5_REPORT["relaunches"].append({"ok": False, "gpu_release": rel, "host_gate": gate.get("ok")})
            return False, "GPU release / host RAM gate failed"
        srv = merlin_fn_server(old.name, NB5_BOOT_KV, old.port, NB5_KV_BYTES[NB5_BOOT_KV])
        srv.start()
        NB5_SERVERS[NB5_SERVERS.index(old)] = srv
        ok, why = srv.wait_ready(max(60.0, min(NB5_READY_TIMEOUT, t)))
        sc = sampled_check(srv) if ok else None
        ok = bool(ok and sc and sc.get("ok"))
        NB5_REPORT["relaunches"].append({"ok": ok, "why": why, "sampled": sc, "old_log": str(cp)})
        if ok:
            E11_SERVER, SGL_URL = srv, srv.url
        return ok, why
    except Exception as _e:
        NB5_REPORT["relaunches"].append({"ok": False, "error": repr(_e)[:400]})
        return False, repr(_e)[:300]


NB5_PIN = {"start_gib": None, "levers": None, "patch_60490": None}   # R2: the next boot starts at this boot's working step
NB5_EQ = {"window_sec": None, "plan": None, "try_t0_pc": None}        # R2: the equal solve window (set at the first phase)


def nb5_boot_try(trec, name, kv_bytes, phases, mtp_on):
    """One launch of one boot (fail closed): GPU release < 150 MiB, page-cache drop + Merlin RAM floor, ready <= 30 min,
    the C-11 config read-back, the short gates, capacity (C-8: 0 workers = NO_GO), warm-up, the pure decode window
    (commit only, not R2), then the phases of this boot (A-12: a phase that is not DONE or has a solver error ends the
    boot as PHASE_FAILED, the partial submission / report stay). Everything is recorded in trec."""
    global E11_SERVER, SGL_URL, ENGINE_READY, CONCURRENCY, E12_STREAMS, E12_VLLM_CONCURRENCY, ENGINE_INFO, NB5_CAPACITY
    trec["t0_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
    NB5_EQ["try_t0_pc"] = time.perf_counter()
    trec["boot_gate"] = boot_gate()
    if trec["boot_gate"]["ok"] is not True:
        trec["status"] = "GPU_NOT_EMPTY"
        return trec
    trec["host_gate"] = nb5_host_gate(name)
    if trec["host_gate"].get("ok") is False:
        trec["status"] = "HOST_RAM_GATE_FAILED"
        return trec
    shm0 = shm_names()
    srv = merlin_fn_server(name, NB5_BOOT_KV, NB5_PORT, kv_bytes)
    trec["argv_tail"] = srv.argv[-60:]
    trec["server_env_levers"] = {k: srv.env.get(k) for k in sorted({**NB5_BASE_ENV, **NB5_BOOT_LEVERS})}   # NB5Q-ENV-1: final server env
    trec["diff_vs_rc2_fp8"] = NB5L.argv_diff(NB5_RC2_FP8_ARGV_TAIL, srv.argv[-60:])
    NB5_SERVERS[:] = [srv]

    def body(ss):
        global E11_SERVER, SGL_URL, ENGINE_READY, CONCURRENCY, E12_STREAMS, E12_VLLM_CONCURRENCY, ENGINE_INFO, NB5_CAPACITY
        s = ss[0]
        E11_SERVER, SGL_URL = s, s.url
        trec["ready_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
        trec["config_readback"] = NB5L.config_readback(s.log(), s.argv, NB5_BOOT_KV, mtp_on)    # C-11
        if not trec["config_readback"]["ok"] and not RERUN:
            trec["status"] = "INCONCLUSIVE_CONFIG_MISMATCH"
            for sp in phases:
                NB5_REPORT["phases"].setdefault(sp[0], {"spec": sp, "boot": name}).update(
                    status="INCONCLUSIVE_CONFIG_MISMATCH", config_readback=trec["config_readback"]["checks"])
            return {}
        ok, pr = nb5_probes(s)
        trec["probes"] = pr
        if not ok:
            trec["status"] = "NO_GO_" + str(pr.get("fatal"))
            return {}
        NB5_CAPACITY = nb5_capacity(s, NB5_BOOT_KV)
        trec["capacity"] = NB5_CAPACITY
        W = int(NB5_CAPACITY["workers"])
        if W < 1 or NB5_CAPACITY.get("verdict") == "NO_GO_INSUFFICIENT_KV":                     # C-8
            trec["status"] = "NO_GO_INSUFFICIENT_KV"
            return {}
        NB5_BLOCKS["b0"] = NB5_CAPACITY["per_attempt"]["b0"]["blocks"]
        NB5_BLOCKS["b1"] = NB5_CAPACITY["per_attempt"]["b1"]["blocks"]
        NB5_BLOCKS.update(f=NB5_CAPACITY["f_eff"], block=NB5_CAPACITY["block_size"],
                          start=NB5L.cdiv(NB5_PROMPT_ALLOW, NB5_CAPACITY["block_size"]) + NB5_CAPACITY["f_eff"],
                          peak_b0=NB5_CAPACITY["per_attempt"]["b0"]["projected_peak_tokens"],          # C-9
                          peak_b1=NB5_CAPACITY["per_attempt"]["b1"]["projected_peak_tokens"])
        NB5_POOL_CAP["blocks"] = NB5_CAPACITY["capacity_blocks"]
        CONCURRENCY = E12_STREAMS = E12_VLLM_CONCURRENCY = W
        ENGINE_INFO = {"backend": f"merlin-{NB5_BOOT_KV}-{'mtp3' if mtp_on else 'nomtp'}", "kv_tokens": (NB5_CAPACITY.get("caps") or {}).get("kv_tokens"),
                       "concurrency": W, "capacity": NB5_CAPACITY, "max_model_len": MAX_LEN, "cfg": {"weights": NB5_BOOT_KV}}
        ENGINE_READY = True
        log("NB5 boot", name, "| workers", W, "| capacity", json.dumps({k: NB5_CAPACITY.get(k) for k in
                                                                         ("pool_blocks", "pool_source", "f_eff", "f_source", "capacity_blocks",
                                                                          "r2_workers")}))
        trec["admission"] = {"mode": NB5_ADMIT_MODE, "future_hook": NB5L.future_admission_blocks(None, None),
                             "note": ("future = placeholder, behaves as projected" if NB5_ADMIT_MODE == "future" else
                                      "slots = a safety cap of workers attempts in flight; the engine schedules (preemptions "
                                      "allowed, counted)" if NB5_ADMIT_MODE == "slots" else None)}
        trec["warmup"] = nb5_warmup(s.port, W)
        if not RERUN and not P.get("no_pure_decode"):
            try:
                trec["pure_decode"] = nb5_pure_decode(s.port, W, NB5_CAPACITY)
                if trec["pure_decode"].get("status") == "STEADY":
                    NB5_BOOT_DECODE["tok_s"] = trec["pure_decode"].get("tok_s")
            except Exception as _e:
                trec["pure_decode"] = {"error": repr(_e)[:300]}
        trec["counters_after_warmup"] = e21_counters()
        trec["phases_started"] = True
        for spec in phases:
            r = nb5_phase(spec, name)
            if spec[0] == "SMOKE":
                why = nb5_smoke_gate(r)
                r["gate"] = why or "PASS"
                if why and not RERUN:
                    log("NB5 SMOKE gate failed:", why)
                    trec["status"] = "SMOKE_GATE_FAILED"
                    break
            elif r.get("status") != "DONE" or r.get("solver_error"):                                # A-12
                trec.update(status="PHASE_FAILED", failed_phase=spec[0],
                            phase_error=str(r.get("error") or r.get("solver_error") or r.get("status"))[-2000:])
                log("NB5 PHASE_FAILED:", spec[0], trec["phase_error"][-300:])
                break
            if not s.alive() and E11_SERVER is s:
                break
        trec["counters_end"] = e21_counters()
        return {}
    o = with_servers(NB5_SERVERS, NB5_READY_TIMEOUT, body)
    ENGINE_READY = False
    trec.update(servers=o.get("servers"), gpu_released=o.get("gpu_released"), shm_cleanup=shm_cleanup(shm0))
    trec["log_text_for_ladder"] = srv.log() or ""
    trec["status"] = trec.get("status") or o.get("status") or "DONE"
    return trec


def nb5_boot(tag, phases, mtp_on, kv=None):
    """One Merlin boot. R2 lever a (KV pin ladder, NB1's descending rule): kv_cache_memory_bytes = the RC2 pin + 1.0 /
    0.75 / 0.5 / 0 GiB per card, highest first (the second boot starts at the first boot's working step: the same
    settings in both phases); a step down only on a launch failure or an OOM line before any phase. Lever b: the
    non-numeric lever env (MERLIN_SHARE_DRAFT_WORKSPACE=1) is dropped when an error line names it, or once all steps
    failed with it (then the RC2 pin without it). The launch --max-num-seqs follows SEQS_TABLE[kv][pin]. Every try is
    recorded (pin, pool, F, failure); the working try's record is merged into the boot record."""
    global NB5_BOOT_EXTRA, NB5_BOOT_KV, NB5_BOOT_LEVERS
    NB5_BOOT_KV = kv or NB5_KV_DTYPE
    brec = {"tag": tag, "mtp": mtp_on, "kv_dtype": NB5_BOOT_KV, "kv_bytes": NB5_KV_BYTES[NB5_BOOT_KV], "tries": []}
    NB5_BOOTS.append(brec)
    base_extra = list(NB5_EXTRA_ARGS) + ([] if mtp_on else list(NB5_MTP_OFF_ARGS))
    if P.get("pin_ladder"):
        start = NB5_PIN["start_gib"]
        if start is None and P.get("patch_60490") and not (NB5_PIN.get("patch_60490") or {}).get("ok") and not FAKE:
            start = 0.0                            # #60490 not verified: no step above the RC2 pin (fail safe)
        tries = NB5L.pin_tries(NB5_KV_BYTES[NB5_BOOT_KV], start)
    else:
        tries = [{"gib": 0.0, "kv_bytes": NB5_KV_BYTES[NB5_BOOT_KV]}]
    levers = dict(NB5_PIN["levers"] if NB5_PIN["levers"] is not None else NB5_LEVER_ENV)
    nolever_left = bool(levers)
    i, trec = 0, {}
    while i < len(tries):
        tr = tries[i]
        if brec["tries"] and left() < 25 * 60:
            brec["ladder_stop"] = f"only {round(left() / 60, 1)} min left: no further boot try"
            break
        NB5_BOOT_LEVERS = dict(levers)
        seqs = NB5L.seqs_for(NB5_BOOT_KV, tr["gib"]) if NB5_SEQS_RULE == "table" else None
        NB5_BOOT_EXTRA = base_extra + (["--max-num-seqs", str(seqs)] if seqs else [])
        name = (tag if not P.get("pin_ladder") else f"{tag}_pin{int(round(tr['gib'] * 100))}" + ("" if levers or not NB5_LEVER_ENV else "_nolever"))
        trec = {"name": name, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"], "max_num_seqs_launch": seqs, "levers": dict(levers)}
        brec["tries"].append(trec)
        nb5_boot_try(trec, name, tr["kv_bytes"], phases, mtp_on)
        fail = NB5L.boot_failure(trec.get("status"), trec.pop("log_text_for_ladder", ""), levers, trec.get("phases_started", False))
        trec["ladder"] = fail
        log("NB5 boot try", name, trec.get("status"), "| ladder", json.dumps({k: fail[k] for k in ("failed", "lever_suspect")}))
        nb5_write_report()
        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why"):
            break
        if fail["lever_suspect"]:
            for k_ in fail["lever_suspect"]:
                levers.pop(k_, None)
            brec.setdefault("levers_dropped", []).append({"keys": fail["lever_suspect"], "why": "named in an error line", "try": name})
            continue                               # the same pin step without that lever
        if i == len(tries) - 1 and nolever_left and levers:
            brec.setdefault("levers_dropped", []).append({"keys": sorted(levers), "why": "every pin step failed with the lever env",
                                                          "try": name})
            levers, nolever_left = {}, False
            continue                               # the RC2 pin without the levers
        i += 1
    for k_ in ("boot_gate", "host_gate", "argv_tail", "server_env_levers", "diff_vs_rc2_fp8", "config_readback", "probes", "capacity", "admission",
               "warmup", "pure_decode", "counters_after_warmup", "counters_end", "servers", "gpu_released", "shm_cleanup",
               "failed_phase", "phase_error", "phases_started", "ready_sec", "t0_sec"):
        if k_ in trec:
            brec[k_] = trec[k_]
    brec["pin_used"] = {"gib": trec.get("pin_gib"), "kv_bytes": trec.get("kv_bytes"), "levers": trec.get("levers"),
                        "server_env_levers": trec.get("server_env_levers"),
                        "max_num_seqs_launch": trec.get("max_num_seqs_launch"),
                        "pool_blocks": (trec.get("capacity") or {}).get("pool_blocks"), "f_eff": (trec.get("capacity") or {}).get("f_eff"),
                        "failures": [{"name": t_["name"], "status": t_.get("status"), "oom_lines": (t_.get("ladder") or {}).get("oom_lines")}
                                     for t_ in brec["tries"] if (t_.get("ladder") or {}).get("failed")]}
    if trec and not (trec.get("ladder") or {}).get("failed") and trec.get("status") not in ("GPU_NOT_EMPTY", "HOST_RAM_GATE_FAILED"):
        NB5_PIN.update(start_gib=trec.get("pin_gib"), levers=dict(trec.get("levers") or {}))
    brec["status"] = trec.get("status") or brec.get("ladder_stop") and "SKIPPED_BUDGET" or "NO_TRY"
    return brec


def nb5_write_report():
    try:
        NB5_REPORT["minutes"] = round((time.perf_counter() - NOTEBOOK_T0) / 60, 1)
        write_json_atomic(BASE_OUT / "nb5_report.json", NB5_REPORT)
    except Exception as _e:
        print("NB5 report write failed:", repr(_e)[:200], flush=True)
'''

RUNNER = r'''# ==================================================================== CELL 8 — NB5: boots and phases
# Commit: warm-up -> pure decode window -> SMOKE (2 tasks, A/C split) -> SMOKE gate -> EVAL (eval subset, scored)
# [-> MTP-off boot -> EVALOFF]. Hidden rerun: warm-up -> EVAL on every hidden task (CELL 0's fallback stays otherwise).
BASE_OUT = OUT
NB5_ALL_IDS, NB5_ALL_CH, NB5_ALL_SOL = list(TASK_IDS), dict(CHALLENGES), (dict(SOLUTIONS) if SOLUTIONS is not None else None)
NB5_EVAL_CH = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json") if not RERUN else {}
NB5_EVAL_SOL = (_load_json(ARC_ROOT / "arc-agi_evaluation_solutions.json")
                if not RERUN and (ARC_ROOT / "arc-agi_evaluation_solutions.json").exists() else None)
NB5_REPORT = {"run_id": RUN_ID, "profile": PROFILE, "rerun": RERUN, "config": CONFIG.get("nb5"), "boots": NB5_BOOTS,
              "phases": {}, "errors": []}
try:
    if not E10_SANDBOX_SELFTEST.get("ok"):
        raise RuntimeError("sandbox self-test failed: " + json.dumps(E10_SANDBOX_SELFTEST)[:300])
    nb5_rc2_inputs()
    if not MERLIN_OK:
        raise RuntimeError("Merlin inputs missing: " + json.dumps(R.get("inputs"), default=str)[:600])
    nb5_rc2_prompts()            # RC2 A0 (host facts, merge + background sha256) + the Flash-Next tokenizer / xhigh template
    NB5_TAILS = NB5L.tails(lambda s_: TOK.encode(s_, add_special_tokens=False).ids)
    NB5_REPORT["tails"] = NB5_TAILS
    NB5_REPORT["prompt_render"] = {k: v for k, v in R.get("prompt_render", {}).items() if "text" not in k}
    # A-14: the local tokenizer + template are a /tokenize fallback only when RC2's pinned hashes were verified at load
    NB5_LOCAL_TOK.update(verified=bool(MERLIN_PROMPTS and R.get("prompt_render", {}).get("hashes_ok")), hf=_HF, template=_tpl,
                         encode=lambda s_: TOK.encode(s_, add_special_tokens=False).ids, default_kwargs=json.loads(XHIGH),
                         why=None if (MERLIN_PROMPTS and R.get("prompt_render", {}).get("hashes_ok")) else
                         "tokenizer / template hashes not verified (or not the Merlin checkpoint)")
    NB5_REPORT["local_tokenizer"] = {"verified": NB5_LOCAL_TOK["verified"], "why": NB5_LOCAL_TOK["why"]}
    step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")}
                                                           if not FAKE else {"fake": True})(merlin_install())})
    _ok, _why = sha_gate(min(1500, left() - 900))
    R["setup"]["sha_gate"] = {"ok": _ok, "why": _why}
    if not _ok:
        raise RuntimeError("sha256 gate: " + _why)
    if P.get("patch_60490"):                       # R2 lever a: #60490 before the first boot (fail safe: RC2 pin only)
        try:
            NB5_PIN["patch_60490"] = nb5_apply_60490()
        except Exception as _e:
            NB5_PIN["patch_60490"] = {"ok": False, "error": f"{type(_e).__name__}: {_e}"[:800]}
        R["setup"]["patch_60490"] = NB5_REPORT["patch_60490"] = NB5_PIN["patch_60490"]
        log("NB5 #60490:", json.dumps(NB5_PIN["patch_60490"], default=str)[:600])
    _groups = []
    for _sp in P["nb5_phases"]:
        _sp = list(_sp) + [None] * (7 - len(_sp))
        _key = (_sp[3], _sp[5] or NB5_KV_DTYPE)
        if not _groups or _groups[-1][0] != _key:
            _groups.append([_key, []])
        _groups[-1][1].append(_sp)
    for (_mtp, _kv), _phs in _groups:
        _b = nb5_boot(f"merlin_{_kv}_{'mtp' if _mtp == 'on' else 'nomtp'}", _phs, _mtp == "on", _kv)
        log("NB5 boot", _b["tag"], _b.get("status"))
        nb5_write_report()
        if _b.get("status") not in ("DONE",):
            # R2: the second paired phase still runs after a failed first boot (recorded); a GPU-release abort stops all
            if not P.get("continue_after_failed_boot") or ABORT.get("why"):
                break
except BaseException:
    NB5_REPORT["errors"].append(traceback.format_exc()[-3000:])
    print("NB5 error (submission.json keeps what was written):", traceback.format_exc()[-2000:], flush=True)
NB5_FINAL_KILL = True
for _s in list(NB5_SERVERS):
    try:
        _s.kill()
    except Exception:
        pass
nb5_write_report()
'''

REPORT = r'''# ==================================================================== CELL 9 — NB5 report: the four averages, counters, bundle
try:
    _ph = NB5_REPORT.get("phases") or {}
    _main = _ph.get("EVAL") or next(iter(_ph.values()), {})
    _gen_total = sum(float(((b.get("counters_end") or {}).get("vllm:generation_tokens_total")) or 0) for b in NB5_BOOTS)
    _nb_sec = time.perf_counter() - NOTEBOOK_T0
    NB5_REPORT["throughput"] = {
        "pure_decode": [{k: (b.get("pure_decode") or {}).get(k) for k in ("band", "level", "status", "tok_s", "running_mean",
                                                                         "mean_acceptance_length", "step_ms", "engine_rates")} | {"boot": b["tag"]}
                        for b in NB5_BOOTS if b.get("pure_decode")],
        "steady_mixed": {n: p.get("steady_mixed") for n, p in _ph.items()},
        "solver_window": {n: p.get("solver_window") for n, p in _ph.items()},
        "whole_notebook": {"gen_tokens_all_boots": _gen_total, "sec": round(_nb_sec, 1), "tok_s": NB5L.rate(_gen_total, _nb_sec)},
        # revision 1: productive decode time / wall time and the token-weighted version, next to the rates
        "duty_factor": {**{n: p.get("duty") for n, p in _ph.items()},
                        "whole_notebook": NB5L.duty_factor("\n".join(f.read_text(encoding="utf-8", errors="replace")
                                                                      for f in sorted(LOGS.glob("server_*.log"))),
                                                           _nb_sec, _gen_total,
                                                           next((b["pure_decode"].get("tok_s") for b in NB5_BOOTS
                                                                 if (b.get("pure_decode") or {}).get("status") == "STEADY"), None))},
        "note": "pure decode = a settled decode-only window at running == workers; steady mixed = stats lines with every slot "
                "running (turn prefill included); solver window = server tokens / solve seconds; whole notebook = every boot's "
                "tokens / the notebook's seconds (boots, probes, warm-ups included). All after the JIT warm-up."}
    if "EVAL" in _ph and "EVALOFF" in _ph:     # the paired MTP on / off accuracy comparison (same tasks, equal windows)
        _a, _b = _ph["EVAL"], _ph["EVALOFF"]
        NB5_REPORT["mtp_pair"] = {"score_on": _a.get("score"), "score_off": _b.get("score"),
                                  "accepted_on": (_a.get("bands") or {}).get("accepted"), "accepted_off": (_b.get("bands") or {}).get("accepted"),
                                  "correct_on": (_a.get("bands") or {}).get("correct"), "correct_off": (_b.get("bands") or {}).get("correct"),
                                  "solver_tok_s_on": (_a.get("solver_window") or {}).get("tok_s"),
                                  "solver_tok_s_off": (_b.get("solver_window") or {}).get("tok_s"),
                                  "target_only_commit_replay": _b.get("paired_commit_replay")}
    # revision 5: the gates only a GPU run can pass, with what THIS run measured; a fake-server pass is not evidence
    _cp = ((NB5_BOOTS[0].get("probes") or {}).get("commit_probe") or {}) if NB5_BOOTS else {}
    _vc, _sp, _cw = _cp.get("variant_c") or {}, _cp.get("server_parsers") or {}, _cp.get("cold_vs_warm") or {}
    _v = lambda ok: None if ok is None else ("PASS" if ok else "FAIL")
    NB5_REPORT["gpu_only_gates"] = {
        "evidence": "FAKE_SERVER (local harness): NOT production evidence" if FAKE else "GPU run",
        "token_id_completion_semantics": {"rule": "server render ends '<think>\\n', /tokenize = chat prompt_tokens, the C reply ends "
                                                  "with finish stop, a closed think, a parsed call",
                                          "tail_ok": _cp.get("tail_ok"), "render_parity": _cp.get("render_parity"),
                                          "natural_finish": _vc.get("natural_finish"), "calls": _vc.get("tool_calls"),
                                          "verdict": _v(bool(_cp.get("tail_ok") and _cp.get("render_parity") and _vc.get("natural_finish")
                                                             and _vc.get("tool_calls"))) if _cp else None},
        "parser_correctness": {"rule": "the same continue_final_message ids: server qwen3 / qwen3_coder parse == Duck's markup parse",
                               "cfm_starts_with_c": _cp.get("cfm_starts_with_c"), "args_equal": _sp.get("args_equal_client_parse"),
                               "verdict": _v(_sp.get("args_equal_client_parse")) if "args_equal_client_parse" in _sp else "INCONCLUSIVE"},
        "mtp_on_state_consistency": {"rule": "greedy C reply with a cold prefix (cache_salt) == with the warm prefix; MTP-off replay "
                                             "(NB5EVAL30MTP) for target-only parity", "cold_vs_warm": _cw.get("verdict"),
                                     "target_only_replay": (_ph.get("EVALOFF") or {}).get("paired_commit_replay") or "not run",
                                     "verdict": {"EQUAL": "PASS", "DIFFER": "FAIL"}.get(_cw.get("verdict"), "INCONCLUSIVE")},
        "real_prefix_reuse": {"rule": "whole-run prefix hit >= 88 % (PLAN W2), commit turns cached", "phases": {
            n: {"client_prefix_hit_pct": (p.get("tracer") or {}).get("prefix_hit_pct"),
                "server_prefix_hit_pct": (p.get("counters") or {}).get("prefix_hit_pct"),
                "block_floor_pct": (p.get("tracer") or {}).get("prefix_hit_pct_at_block_floor"),
                "retained_session_hit_share": (p.get("tracer") or {}).get("retained_session_hit_share"),
                "commit_C_cached_pct": (((p.get("tracer") or {}).get("commit_by_variant") or {}).get("C") or {}).get("cached_pct")}
            for n, p in _ph.items()},
            "verdict": _v(((_main.get("counters") or {}).get("prefix_hit_pct") or 0) >= 88.0) if _main.get("counters") else None}}
    if FAKE:
        for _k in ("token_id_completion_semantics", "parser_correctness", "mtp_on_state_consistency", "real_prefix_reuse"):
            NB5_REPORT["gpu_only_gates"][_k]["verdict"] = "LOCAL_ONLY (" + str(NB5_REPORT["gpu_only_gates"][_k]["verdict"]) + ")"
    _bs = {n: p for n, p in _ph.items() if (p.get("budget_plan") or {}).get("budget")}
    if _bs:                                    # PLAN v3.6 NB5BUDGET: solve rate, concurrency, tok/s per budget
        NB5_REPORT["budget_sweep"] = {n: {"budget": p["budget_plan"]["budget"], "workers": p["budget_plan"]["workers"],
                                          "task_norm_pct": (p.get("score") or {}).get("eval_task_norm_pct"),
                                          "query_hits": (p.get("score") or {}).get("eval_query_hits"),
                                          "server_running_mean": (p.get("b_summary") or {}).get("server_running_mean"),
                                          "solver_tok_s": (p.get("solver_window") or {}).get("tok_s"),
                                          "budget_limited_attempts": p.get("budget_limited_attempts")} for n, p in _bs.items()}
    _q = {n: p for n, p in _ph.items() if n.startswith("Q_")}
    if _q:                                     # addendum 5: per config and the paired task x config table
        NB5_REPORT["quality_ab"] = {
            "configs": {n: {"kv": p.get("kv_dtype"), "workers": p.get("workers"), "score": p.get("score"),
                            "exact_tasks": sum(1 for v in (p.get("quality") or {}).values() if v.get("exact_all_tests")),
                            "complete_tasks": sum(1 for v in (p.get("quality") or {}).values() if v.get("complete")),
                            "output_tokens": sum(v.get("output_tokens") or 0 for v in (p.get("quality") or {}).values()),
                            "tool_call_rate": (p.get("tracer") or {}).get("tool_call_rate"),
                            "ttft_mean_sec": ((p.get("latency_means") or {}).get("time_to_first_token_seconds") or {}).get("mean_sec"),
                            "phase_sec": (p.get("phase_window") or {}).get("sec")} for n, p in _q.items()},
            "paired": {t: {n: {k: (p.get("quality") or {}).get(t, {}).get(k) for k in ("exact_all_tests", "complete", "output_tokens",
                                                                                         "wall_sec", "tool_call_rate", "first_request_latency_sec")}
                           for n, p in _q.items()} for t in (P.get("task_list") or NB5_QUALITY_TASKS)}}
    if P.get("quality_ab"):                    # R2: per phase + the paired FP8 / BF16 summary (all / finished in both)
        _pc = {n: {"kv": p.get("kv_dtype"), "status": p.get("status"), "fp8_status": p.get("fp8_status"),
                   "window": p.get("window"), "window_plan": p.get("window_plan"), "capacity_levers": p.get("capacity_levers"),
                   "speed": p.get("speed"), "order": p.get("order"), "score": p.get("score"),
                   "tasks": (p.get("qab") or {}).get("tasks") or {}, "summary": (p.get("qab") or {}).get("summary")}
               for n, p in _q.items()}
        _qa = NB5_REPORT.setdefault("quality_ab", {})
        _qa.update(nb0=NB5_QAB_NB0, fp8_label=NB5_QAB_FP8_LABEL, groups=NB5_QAB_GROUPS, per_config=_pc)
        # Q-1 / Q-2 / Q-3 / Q-5: windows check (both DONE, numeric, |delta| <= 2 s, none unequal), paired, deployment_comparison
        # (all tasks / finished in both / missing-incomplete-failed matrix), precision_comparison UNVERIFIED, phase_diff
        _qa.update(NB5L.qab_report(_pc))
        for n, v in _pc.items():
            print(NB5L.qab_table_text(v["tasks"], n), flush=True)
            print(n, "summary:", json.dumps(v.get("summary"), default=str), "| levers:", json.dumps(v.get("capacity_levers"), default=str)[:1500],
                  "| speed:", json.dumps(v.get("speed"), default=str), "| window:", json.dumps(v.get("window"), default=str), flush=True)
        _pa = _qa["paired"]
        for _blk in ("all", "finished_in_both"):
            for _g, _s in ((_pa.get(_blk) or {}).items() if isinstance(_pa.get(_blk), dict) else []):
                print(f"PAIRED [{_blk}] {_g}: tasks {_s['tasks']} | solved FP8 {_s['solved_FP8']} BF16 {_s['solved_BF16']} | both "
                      f"{_s['both']} FP8-only {_s['FP8_only']} BF16-only {_s['BF16_only']} neither {_s['neither']} | discordant "
                      f"{_s['discordant']} (McNemar p {_s['mcnemar_p']}; paired n {_s['paired_n']}, min attainable p "
                      f"{_s['min_attainable_p_paired_n']} / {_s['min_attainable_p_at_discordant']} at this discordant count; "
                      f"{_s['reading']}) {_s['discordant_tasks']}", flush=True)
        if "skipped" in _pa:
            print("PAIRED:", _pa["skipped"], flush=True)
        print("deployment_comparison:", _qa["deployment_comparison"]["status"], json.dumps(_qa["windows_check"]), "| matrix:",
              json.dumps({n: (p["outcomes"], p["tasks_missing"]) for n, p in _qa["deployment_comparison"]
                          ["missing_incomplete_failed"]["phases"].items()}), "| precision_comparison: UNVERIFIED | phase_diff:",
              json.dumps(_qa["phase_diff"]["differences"], default=str)[:2500], flush=True)
    nb5_write_report()
    print("NB5 REPORT:", json.dumps({"throughput": NB5_REPORT["throughput"], "phases": {n: {k: p.get(k) for k in (
        "status", "score", "workers", "running_check", "commit_mode")} for n, p in _ph.items()}}, default=str)[:6000], flush=True)
except BaseException:
    print("NB5 report error:", traceback.format_exc()[-2000:], flush=True)
try:
    _pack = WORKING / "Q27X_NB5_BUNDLE.zip"
    with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in [BASE_OUT / "nb5_report.json", SUBMISSION_PATH, ENG_OUT / "nb5_engine.json", ENG_OUT / "telemetry.jsonl"] \
                + sorted(LOGS.glob("*.log")) + sorted(TRACES.glob("*")) + sorted(BASE_OUT.glob("phase_*/e12_run_report.json")) \
                + sorted(BASE_OUT.glob("phase_*/nb5_sampler.jsonl")):
            if fp.exists() and fp.is_file():
                z.write(fp, arcname=fp.relative_to(WORKING).as_posix() if WORKING in fp.parents else fp.name)
    print("NB5 bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
except BaseException:
    print("NB5 bundle error:", traceback.format_exc()[-1500:], flush=True)
'''


def engine_cell():
    donors = "\n\n\n".join(f"# ---- E21 donor, verbatim: {n}\n{s}" for n, s in E21_DONORS)
    inputs_fn = wrap(RC2_INPUTS, "nb5_rc2_inputs", "RC2 CELL 1 input discovery (exact Model paths), GPU preflight, watchdog")
    prompts_fn = wrap(RC2_PROMPTS, "nb5_rc2_prompts", "RC2 CELL 3: A0 (host facts, merge, sha256 thread), tokenizer, template, prompts")
    return (ENGINE_HEAD + RC2_LAB_LOAD + "\n# ---- RC2 CELL 2 (verbatim to the stock fallback)\n" + RC2_ENGINE
            + "\n# ---- RC2 CELL 1 input block, wrapped\n" + inputs_fn
            + "\n\n# ---- RC2 CELL 3, wrapped\n" + prompts_fn
            + "\n\n# ---- RC2 CELL 4 (= the E23 step runner, verbatim)\n" + STEP_RUNNER
            + "\n\n\n# ---- RC2 CELL 5 checks (verbatim)\n" + RC2_CHECKS
            + "\n\n# ---- E21 engine functions\n" + donors
            + NB5_ENGINE.replace("__RC2ARGV__", json.dumps(RC2_FP8_ARGV_TAIL)).replace("__P60490__", json.dumps(PATCH_TEXT))
            .replace("__PPRE__", P60490_PRE).replace("__PPOST__", P60490_POST).replace("__APPLIER__", APPLIER.rstrip("\n")))


MD0_R2 = """
**NB5QUALITY_R2 (lead 2026-10-31, user-approved): the paired FP8-vs-BF16 KV ARC accuracy test.** ONE notebook, 180-min hard
limit (install + both boots included): phase Q_FP8_K3 (FP8 QSA KV, label UNVALIDATED_FP8_UNDER_TEST; NB0 verdict
NO_GO_FP8_TF_DIVERGENCE recorded, not a gate) then Q_BF16_K3 (BF16 KV), separate boots, MTP K3 on, commit variant A, the 15
lead-chosen eval tasks (7 solid / 5 sometimes / 3 never) in a fixed order, 104k per attempt, 2 attempts, one fixed seed, no
SMOKE, EQUAL solve windows (fixed at the first phase). Levers in BOTH phases: KV pin ladder (RC2 pin + 1.0 / 0.75 / 0.5 / 0
GiB per card, #60490 backport), MERLIN_SHARE_DRAFT_WORKSPACE=1 (numerics-neutral; F measured from the server line), workers =
max_num_seqs = max(1, floor(pool / (ceil(65536 / B) + F))) (launch seqs from the per-pin table), CAPS16 captures; the engine
schedules (preemptions counted: speed only). Report: per phase pool / pin / F / streams / preemptions / tok/s + the per-task
accuracy table, then the paired summary (all tasks and finished-in-both). Audit fixes Q-1/2/3/5: an attempt is finished
only on ACCEPTED / BUDGET_END (errors, timeouts, no terminal evidence = FAILED_OR_INTERRUPTED); the comparison is a
`deployment_comparison` (same wall-clock, production config; INCONCLUSIVE_UNEQUAL_WINDOWS unless both phases are DONE with
numeric windows within 2 s) with all-task, finished-in-both and missing / incomplete / failed results, paired n and the
minimum attainable p next to McNemar; `precision_comparison` = UNVERIFIED (workers / pool / pin differ: `phase_diff`).
"""


def cells(profile):
    c1 = (CELL1.replace("__PROFILE__", profile).replace("__NB5LIBPACK__", packed(NB5_LIB, "nb5_lib"))
          .replace("__CAPS16__", json.dumps(CAPS16)))
    return [("markdown", MD0.replace("__PROFILE__", profile) + (MD0_R2 if profile == "NB5QUALITY_R2" else "")),
            ("code", D[1]),                                   # CELL 0 (E15B R3K2 = E21), verbatim
            ("code", c1),
            ("code", D[5] + CELL3_PATCH),                     # CELL 3 (E21), verbatim + the NB5 hooks
            ("code", D[6]),                                   # task order (E21 = E15B), verbatim
            ("code", engine_cell()),
            ("code", CELL67),                                 # CELL 6/7 (E21 wrapped functions) + the gate patch
            ("code", RUNNER),
            ("code", REPORT)]


def undefined_names_check(cl):
    """NB5-2: pyflakes over all code cells as ONE module (the notebook namespace): no undefined name may remain."""
    try:
        from pyflakes import api as _pf_api, reporter as _pf_rep
    except ImportError:
        print("WARNING: pyflakes not installed: the undefined-name check did not run", flush=True)
        return None
    import io
    out, err = io.StringIO(), io.StringIO()
    _pf_api.check("\n\n".join(norm(t) for k, t in cl if k == "code"), "<nb5 cells>", _pf_rep.Reporter(out, err))
    bad = [ln for ln in out.getvalue().splitlines() if "undefined name" in ln]
    assert not bad and not err.getvalue().strip(), ("NB5-2: undefined names in the notebook", bad[:20], err.getvalue()[:500])
    return True


def build(profile):
    out = NBDIR / PROFILES[profile]
    nb_cells = []
    for kind, text in cells(profile):
        text = norm(text)
        if kind == "code":
            compile(text, f"<nb5 {profile}>", "exec")
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    undefined_names_check(cells(profile))
    data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    if out.exists() and sha(out.read_bytes()) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
        assert not (out.exists() and sha(out.read_bytes()) in PUSHED), out
    out.write_bytes(data)
    if profile == EXPORT_PROFILE:   # one export = the code cells of the R2 build (the frozen six keep the older code)
        (HERE / "nb5_code_export.py").write_text("\n\n".join(norm(t) for k, t in cells(profile) if k == "code"),
                                                 encoding="utf-8", newline="\n")
    print("wrote", out.name, "sha256", sha(data), "|", len(data), "bytes |", len(nb_cells), "cells")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    prof = args[args.index("--profile") + 1] if "--profile" in args else "NB5EVAL120"
    build(prof)
