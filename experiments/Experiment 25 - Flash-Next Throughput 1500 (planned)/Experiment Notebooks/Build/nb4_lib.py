"""E25 NB4 TICK: the pure parts (stdlib only), embedded in the notebook (sha-checked) and imported by the unit tests.
Spec: Experiment Review/NB4_TICK_SPEC_2026-11-01.md + the lead's addenda (2026-11-01: agents B / C / D research, the
folder sweep): the stage-0 gap is an in-graph stall on the PP sampled-token broadcast, decode already runs FULL graphs.

  BASE / make_arms / validate_arms / never_do   the base (NB1 R2 B3 + NB5 QUALITY seqs 9 / CAPS16) and the arms
        T0 (+ the C runtime A/B inside its boot) -> S -> S2 -> G -> P -> T0b (+ optional K4 / INT8), each = base + exactly
        one change; the measured never-do list fails the build / the gate
  compile_json / profiler_json / arm_extra / arm_env / pin_ladder / port_for / launch_name   per-launch argv / env pieces
  retry_decision / death_label    one OOM policy (klee100): a launch OOM steps down the pin ladder (B3 -> -0.25 GiB -> the
        FP8 base pin) on non-base arms; a device fault stops; a fast non-OOM launch failure is retried once
  free_by_card / worker_free / combine_free / probe_engine_view   klee100 R1-3 / R1-5
  parse_nondefault / readback_checks / diff_vs_ref / one_change_ok   config read-back, fail closed
  cudagraph_table / split_label   --cudagraph-metrics rows per window -> FULL / NONE decode counts, the n histogram and
        the micro-batch split (BALANCED / IMBALANCED), compared only between equal splits
  engaged_log / mechanism / commit_split / stall_fraction   the change visible (log / env) AND proven (trace / counters)
  tf_label / correctness          NB0's screening thresholds; greedy oracle EXACT / NEAR_TIE
  window_ok / aa_spread / speed_verdict   winner rule: > max(2 %, the measured A/A spread) on the arm's decisive windows
  final_label / next_arm          labels (WIN_SPEED / WIN_DIAGNOSTIC ...) and the planner (T0b protected)
  op_placement / iteration_summary / core_timer_summary / CORE_TIMER_SRC   the T0 timeline extras
"""
import json
import math
import re
import statistics

GiB, MiB = 1 << 30, 1 << 20
KV_FP8_RC2 = 2479838700                        # Merlin's own L4 FP8 pin (RC2 FP8, 149 usable blocks)
B3_PIN = 3553580524                            # NB1 R2 B3 = KV_FP8_RC2 + 1 GiB (214 blocks; NB5 QUALITY's FP8 pin)
PIN_LADDER = (B3_PIN, B3_PIN - 256 * MiB, KV_FP8_RC2)   # lead item 8: B3, -0.25 GiB, the FP8 base pin (non-base arms)
SEQS = 9                                       # NB5 QUALITY: nb5_lib.SEQS_TABLE["fp8"][1.0]
CAPS16 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]   # NB1 CAPTURE_HI = NB5 CAPS16
G_CAPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 36]   # agent D §1.4: FULL for every n = 1..9 at seqs 9, K3
K4_CAPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 35, 40, 45]   # K4: (C/2) x 5 tokens, n = 1..9
K_SPEC = 3
PP, TP = 2, 2
P5_MAX_BYTES = 184320                          # agent D §4 / lead: P5 for <= 36 tokens x 5120 B; NCCL above
BASE = {"kv": "fp8", "kv_dtype": "fp8_e4m3", "kv_bytes": B3_PIN, "seqs": SEQS, "mtp": True, "k": K_SPEC,
        "compile": {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": CAPS16},
        "language_model_only": True, "patch_60490": True,
        "env": {"VLLM_SERVER_DEV_MODE": "1", "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1"},
        "flags": ["--language-model-only", "--cudagraph-metrics"],
        "what": "W4 Merlin, FP8 E4M3 QSA KV, pin 3,553,580,524 B, MTP K3, --language-model-only, #60490, seqs 9, CAPS16; "
                "identical on every arm (lead item 6 / D item 3): NB1's read-only worker extension + dev mode, "
                "--cudagraph-metrics, VLLM_CUSTOM_SCOPES_FOR_PROFILING=1, the idle torch profiler config"}
ARM_ORDER = ("T0", "S", "S2", "S3", "G", "P", "T0b", "K4", "INT8")
OPTIONAL = ("K4", "INT8")                      # lead item 12: only after T0b, only if the budget allows
BASE_ARMS = ("T0", "T0b")
WINDOWS = ("8k@8", "40k@8", "8k@9", "40k@9")   # D: level 9 or G cannot engage; real xhigh prompts (RC2 prompt())
MIN_GAIN_PCT = 2.0
# minutes. EST = what the planner / step() need before an arm starts (ready + body + teardown, realistic: NB1 R2 ready 297-642 s,
# teardown 1.5-2.5 min); CAP = the body cap (watched: killed at cap + 5 min)
EST_MIN = {"T0": 38, "S": 24, "S2": 24, "S3": 24, "G": 25, "P": 30, "T0b": 28, "K4": 25, "INT8": 24}
CAP_MIN = {"T0": 30, "S": 19, "S2": 19, "S3": 19, "G": 19, "P": 19, "T0b": 25, "K4": 19, "INT8": 19}
READY_SEC = 1500
PORT_BASE = 8040                               # Merlin's PORT 8000..8355 scheme (MASTER_PORT / VLLM_PORT derive from it)
PORT_SLOT = {k: i for i, k in enumerate(ARM_ORDER)}
SCREEN = {"rank1_min": 0.90, "mean_max": 0.15}       # NB0 (external audit 2): conservative screening
TRIAGE = {"mean_max": 1.0, "rank1_min": 0.5}         # NB0 (external audit 3): severe-discrepancy triage
CLEAN_ORACLE = ("EXACT", "NEAR_TIE")                 # NB0 / NB1: L.FAIL_MARGIN 0.5 at the first differing token
WIN_NOTE = "adoption needs paired ARC accuracy + PLAN §4.1-7 evidence"
# the measured never-do list (lead 2026-11-01, folder sweep §2 item 9): the build fails if any arm sets one
NEVER_DO = [("env", "P2P", lambda v: str(v).lower() not in ("off", "0", "")), ("env", "NCCL_P2P_DISABLE", lambda v: str(v) == "0"),
            ("env", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", lambda v: str(v) == "1"),
            ("env", "NCCL_MIN_CTAS", None), ("env", "NCCL_MAX_CTAS", None), ("env", "NCCL_MIN_NCHANNELS", None),
            ("env", "NCCL_MAX_NCHANNELS", None), ("env", "NCCL_SHM_USE_CUDA_MEMCPY", None),
            ("env", "NCCL_PROTO", lambda v: "LL128" in str(v).upper()), ("env", "COUNTERS", lambda v: str(v) == "1"),
            ("env", "NCCL_GRAPH_MIXING_SUPPORT", lambda v: str(v) == "0"), ("env", "NCCL_GRAPH_REGISTER", lambda v: str(v) == "1"),
            ("flag", "--no-async-scheduling", None), ("flag", "--no-enable-prefix-caching", None), ("flag", "--enable-hc-sp", None),
            ("flag", "--mamba-ssm-cache-dtype", lambda v: "bf" in str(v).lower()), ("flag", "--enable-custom-all-reduce", None)]


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def spec_json(k):
    """Merlin's own --speculative-config value (launch.py:441) with K = k."""
    return json.dumps({"method": "mtp", "num_speculative_tokens": int(k), "draft_sample_method": "probabilistic"}, separators=(",", ":"))


# ===================================================================================== arms (base + exactly one change)
def make_arms(s2=None, g=None, p=None, s3=None, disabled=()):
    """The arms (data-driven: `disabled` = arm keys switched off by a flag -> DISABLED, never booted). s2 / s3 / p = the
    install result of the S2 (Q-pack) / S3 (Q-defer) (nb4_ppack.py) and P (P5, nb4_p5_engine.py) patch hooks
    {"ok", "env", "init_rx", "counters_rpc", "why"}; None -> NOT_INTEGRATED / P5_NOT_INTEGRATED (no GPU spent). g = an
    override of G's default (agent D's capture list). C is not a boot: it is the runtime A/B inside T0 (lead item 4)."""
    arms = {
        "T0": {"key": "T0", "role": "base", "dims": ["timeline"], "change": "full host timeline (torch profiler CPU + CUDA on 4 "
               "workers at 8k@8 and 8k@9, vLLM iteration log, the engine-core step timer) + the C runtime A/B (booted with "
               "VLLM_RECOVERSSM_COMMIT_GRAPH=graph, flipped to eager before any measurement)",
               "compile": {}, "extra": ["--enable-logging-iteration-details"], "env": {"VLLM_RECOVERSSM_COMMIT_GRAPH": "graph"},
               "profiles": ["8k@8", "8k@9"], "core_timer": True, "commit_ab": True, "integrated": True, "mechanism": "timeline",
               "decisive": WINDOWS, "ladder": False, "reps": 1, "k": K_SPEC, "prebuild_env": {}},
        "S": {"key": "S", "role": "arm", "dims": ["max_connections"], "change": "CUDA_DEVICE_MAX_CONNECTIONS=32 (hardware "
              "work-queue false-dependency test of the stage-0 in-graph stall)", "compile": {}, "extra": [],
              "env": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}, "profiles": ["8k@8"], "integrated": True, "mechanism": "stall",
              "decisive": WINDOWS, "ladder": True, "reps": 1, "k": K_SPEC, "prebuild_env": {}},
        "S2": {"key": "S2", "role": "arm", "dims": ["ppack"], "change": "Q-pack: one packed PP broadcast instead of three "
               "(nb4_ppack.py patch, env NB4_PPACK=1; applied before the boot, reverted after)", "compile": {}, "extra": [],
               "env": {}, "profiles": ["8k@8"], "integrated": False, "mechanism": "stall", "decisive": WINDOWS, "ladder": True,
               "reps": 1, "k": K_SPEC, "prebuild_env": {}, "patch": "ppack", "variant": "Q-pack",
               "why_not": "NOT_INTEGRATED: nb4_ppack.py not built / not installed"},
        "S3": {"key": "S3", "role": "arm", "dims": ["ppdefer"], "change": "Q-defer: stage 0 posts the parked sampled-token "
               "receive behind the NEXT forward launch (nb4_ppack.py patch, env NB4_PPDEFER=1; applied before the boot, reverted "
               "after)", "compile": {}, "extra": [], "env": {}, "profiles": ["8k@8"], "integrated": False, "mechanism": "stall",
               "decisive": WINDOWS, "ladder": True, "reps": 1, "k": K_SPEC, "prebuild_env": {}, "patch": "ppack", "variant": "Q-defer",
               "why_not": "NOT_INTEGRATED: nb4_ppack.py not built / not installed"},
        "G": {"key": "G", "role": "arm", "dims": ["captures"], "change": "capture-coverage: the capture list " + json.dumps(G_CAPS)
              + " (FULL for every n = 1..9 at seqs 9; agent D §1.4); diagnostic", "compile": {"cudagraph_capture_sizes": list(G_CAPS)},
              "extra": [], "env": {}, "profiles": ["8k@9"], "integrated": True, "mechanism": "capture_coverage",
              "decisive": ("8k@9", "40k@9"), "ladder": True, "reps": 1, "k": K_SPEC, "prebuild_env": {}, "diagnostic": True},
        "P": {"key": "P", "role": "arm", "dims": ["p5"], "change": "P5 world-2 host-staged all-reduce in the engine's TP "
              f"all-reduce (<= {P5_MAX_BYTES} B, NCCL above; nb4_p5_engine.py); diagnostic", "compile": {}, "extra": [], "env": {},
              "profiles": ["8k@8"], "integrated": False, "mechanism": "p5", "decisive": WINDOWS, "ladder": True, "reps": 1,
              "k": K_SPEC, "prebuild_env": {}, "patch": "p5", "diagnostic": True,
              "why_not": "P5_NOT_INTEGRATED: nb4_p5_engine.py not built / not installed"},
        "T0b": {"key": "T0b", "role": "aa", "dims": ["aa_repeat"], "change": "the base again (identical flag set, no profiled "
                "window): the A/A, last; 2 replicates of every window", "compile": {}, "extra": [], "env": {}, "profiles": [],
                "integrated": True, "mechanism": None, "decisive": WINDOWS, "ladder": False, "reps": 2, "k": K_SPEC,
                "prebuild_env": {}},
        "K4": {"key": "K4", "role": "arm", "dims": ["spec_k4"], "change": "MTP K4 (--speculative-config num_speculative_tokens 4) "
               "with its capture list (C/2) x 5 " + json.dumps(K4_CAPS), "compile": {"cudagraph_capture_sizes": list(K4_CAPS)},
               "extra": [], "env": {}, "spec": spec_json(4), "profiles": [], "integrated": True, "mechanism": "spec_k4",
               "decisive": WINDOWS, "ladder": True, "reps": 1, "k": 4, "prebuild_env": {}, "optional": True},
        "INT8": {"key": "INT8", "role": "arm", "dims": ["draft_head_int8"], "change": "DRAFT_HEAD=int8 (Merlin build() knob, K3)",
                 "compile": {}, "extra": [], "env": {}, "profiles": [], "integrated": True, "mechanism": "log_line",
                 "engaged_log_rx": [r"INT8 weight-only copy"], "decisive": WINDOWS, "ladder": True, "reps": 1, "k": K_SPEC,
                 "prebuild_env": {"DRAFT_HEAD": "int8"}, "optional": True},
    }
    if g:
        arms["G"].update({k: v for k, v in g.items() if k in ("compile", "change", "decisive", "profiles")})
    for k, s, lab in (("S2", s2, "NOT_INTEGRATED"), ("S3", s3, "NOT_INTEGRATED"), ("P", p, "P5_NOT_INTEGRATED")):
        if s is None:
            continue
        a = arms[k]
        ok = s.get("ok") is True
        a["integrated"] = ok
        a["why_not"] = None if ok else (lab + ": " + str(s.get("why") or "install not ok"))[:400]
        if ok:
            a["env"] = {str(x): str(y) for x, y in (s.get("env") or {}).items()}
            for f in ("init_rx", "counters_rpc", "install"):
                if f in s:
                    a[f] = s[f]
    for k in disabled or ():
        if k in arms:
            arms[k].update(integrated=False, disabled=True, why_not=f"DISABLED by flag ({k})")
    return arms


def never_do(arm):
    """The never-do items an arm would set (lead 2026-11-01): env keys / values and flags (extra + spec)."""
    hits = []
    env = dict(arm.get("env") or {}, **(arm.get("prebuild_env") or {}))
    fl = [str(x) for x in (arm.get("extra") or [])]
    for kind, key, bad in NEVER_DO:
        if kind == "env" and key in env and (bad is None or bad(env[key])):
            hits.append(f"env {key}={env[key]}")
        if kind == "flag" and key in fl:
            i = fl.index(key)
            v = fl[i + 1] if i + 1 < len(fl) else ""
            if bad is None or bad(v):
                hits.append(f"flag {key} {v}".strip())
    if arm.get("k") == 2:
        hits.append("K2 (a -7 % lever at 8 streams)")
    return hits


ALLOWED_KINDS = {"T0": {"extra", "env"}, "S": {"env"}, "S2": {"env"}, "S3": {"env"}, "G": {"compile"}, "P": {"env"}, "T0b": set(),
                 "K4": {"compile", "spec"}, "INT8": {"prebuild_env"}}   # the setting kinds each arm's one change may touch


def validate_arms(arms):
    """Every arm = base + exactly one change dimension; nothing on the never-do list; the base settings (pin, KV dtype,
    seqs, profiler, prefix caching) are never an arm's change. Returns the problems (empty = valid)."""
    bad = []
    forbidden = ("--kv-cache-memory-bytes", "--kv-cache-dtype", "--max-num-seqs", "--speculative-config", "--max-model-len",
                 "--block-size", "--profiler-config", "--cudagraph-metrics", "--worker-extension-cls")
    for k in ARM_ORDER:
        a = arms.get(k)
        if not a:
            bad.append(f"{k}: missing")
            continue
        if len(a.get("dims") or []) != 1:
            bad.append(f"{k}: {len(a.get('dims') or [])} change dimensions (exactly one)")
        bad += [f"{k}: never-do {h}" for h in never_do(a)]
        for f in a.get("extra") or []:
            if str(f) in forbidden:
                bad.append(f"{k}: extra flag {f} changes a base setting")
        for e in (a.get("env") or {}):
            if str(e).startswith(("KV_", "NB4_CORE")) or e in ("SCALES", "VLLM_PP_LAYER_PARTITION", "VLLM_SERVER_DEV_MODE",
                                                                 "VLLM_CUSTOM_SCOPES_FOR_PROFILING"):
                bad.append(f"{k}: env {e} changes a base setting")
        for ck in (a.get("compile") or {}):
            if ck not in ("cudagraph_mode", "cudagraph_capture_sizes"):
                bad.append(f"{k}: compile key {ck} not a capture setting")
        if a.get("spec") and a.get("dims") != ["spec_k4"]:
            bad.append(f"{k}: speculative-config change outside the K4 arm")
        kinds = {f for f in ("compile", "extra", "env", "prebuild_env", "spec") if a.get(f)}
        if not kinds <= ALLOWED_KINDS.get(k, set()):
            bad.append(f"{k}: changes {sorted(kinds - ALLOWED_KINDS.get(k, set()))} (allowed {sorted(ALLOWED_KINDS.get(k, set()))})")
    t0b = arms.get("T0b") or {}
    if t0b.get("compile") or t0b.get("extra") or t0b.get("env") or t0b.get("prebuild_env") or t0b.get("profiles"):
        bad.append("T0b: must be the base repeated (identical flags, no profiled window)")
    return bad


def compile_dict(arm):
    d = json.loads(json.dumps(BASE["compile"]))
    d.update(arm.get("compile") or {})
    return d


def compile_json(arm):
    """The --compilation-config value (Merlin's own compact JSON format; NB5 CAPS16 rule)."""
    return json.dumps(compile_dict(arm), separators=(",", ":"))


def profiler_json(prof_dir):
    """vLLM ProfilerConfig (field names checked in Merlin vllm/config/profiler.py, agent C): torch on every worker,
    CPU + CUDA, stack / shapes / memory / flops off, frontend off; idle until /start_profile. Identical on every arm."""
    return json.dumps({"profiler": "torch", "torch_profiler_dir": str(prof_dir), "torch_profiler_activities": ["CPU", "CUDA"],
                       "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False,
                       "torch_profiler_with_memory": False, "torch_profiler_with_flops": False, "ignore_frontend": True})


def arm_extra(arm, ext_flag, prof_dir):
    """Trailing vllm flags of one launch: the base flag set (identical on every arm), then the arm's own flags."""
    return list(ext_flag) + list(BASE["flags"]) + ["--profiler-config", profiler_json(prof_dir)] + [str(x) for x in (arm.get("extra") or [])]


def arm_env(arm, diag_env, core_timer_path=None):
    env = dict(diag_env)
    env.update(BASE["env"])
    env.update({k: str(v) for k, v in (arm.get("env") or {}).items()})
    if arm.get("core_timer") and core_timer_path:
        env["NB4_CORE_TIMER"] = str(core_timer_path)
    return env


def pin_for(arm, rung):
    """The KV pin of a launch: the base pin; non-base arms step down PIN_LADDER on a launch OOM (lead item 8)."""
    return PIN_LADDER[min(int(rung), len(PIN_LADDER) - 1)] if arm.get("ladder") else B3_PIN


def port_for(arm_key, try_i):
    """A fresh port per launch try (klee100 R1-9): 8040 + 10 x slot + try, inside Merlin's 8000..8355 scheme."""
    p = PORT_BASE + 10 * PORT_SLOT[arm_key] + int(try_i)
    assert 8000 <= p <= 8355, p
    return p


def launch_name(arm_key, try_i):
    """A fresh launch name per try (klee100 R2-5): log, plan, HELP, load monitor and server record kept per launch."""
    return f"nb4_{arm_key.lower()}" + (f"_t{try_i}" if try_i else "")


# ===================================================================================== one OOM policy, deaths, memory
def retry_decision(arm, status, oom_in_log, xid_in_log, launch_sec, tries, rung, left_sec, need_sec, log_tail=""):
    """ONE OOM policy (klee100 rules, lead item 8). Returns (retry, rung_next, drop_commit_env, why).
      device fault -> stop; launch OOM -> the next lower pin rung (non-base arms only; a base arm never changes its pin);
      a launch failure that names the RecoverSSM commit capture on an arm booted with it (T0) -> one retry without
      VLLM_RECOVERSSM_COMMIT_GRAPH (NB1 run_boot rule: the C A/B is then NOT_INTEGRATED, T0 stays the base);
      a fast (< 600 s) non-OOM launch failure -> one retry at the same pin; never past the budget."""
    if status != "LAUNCH_FAILED":
        return False, rung, False, "no retry: not a launch failure"
    if xid_in_log:
        return False, rung, False, "no retry: device fault in the launch log"
    if left_sec < need_sec:
        return False, rung, False, f"no retry: {round(left_sec)} s left < {round(need_sec)} s"
    if tries >= 4:
        return False, rung, False, "no retry: 4 launches"
    if oom_in_log:
        if arm.get("ladder") and rung + 1 < len(PIN_LADDER):
            return True, rung + 1, False, f"launch OOM: next pin rung {rung + 1} ({PIN_LADDER[rung + 1]} B)"
        return False, rung, False, "no retry: launch OOM at the last allowed pin (base arms keep the base pin)"
    if arm.get("commit_ab") and (arm.get("env") or {}).get("VLLM_RECOVERSSM_COMMIT_GRAPH") and \
            re.search(r"recoverssm|commit", log_tail or "", re.I):
        return True, rung, True, "commit-graph capture failure: one retry without VLLM_RECOVERSSM_COMMIT_GRAPH (C A/B dropped)"
    if tries >= 2:
        return False, rung, False, "no retry: the one same-pin retry is used"
    if not _fin(launch_sec) or launch_sec >= 600:
        return False, rung, False, f"no retry: the launch took {launch_sec} s (a readiness timeout is not repeated)"
    return True, rung, False, "retry once: fast non-OOM launch failure, fresh port and launch name"


def death_label(watchdog, oom, xid, died_in):
    """klee100 R1-10 / R2-2: a server death is named by what was running."""
    item = str(died_in or "unknown")
    if watchdog:
        return "BOOT_CAP_KILL"
    if xid:
        return "DEVICE_FAULT_IN_" + item
    if oom:
        return "TF_OOM" if item.startswith("tf") else "BODY_OOM_" + item
    return "TF_SERVER_DIED" if item.startswith("tf") else "SERVER_DIED_IN_" + item


def free_by_card(readings):
    """nvidia-smi rows [[index, used_mib, total_mib], ...] (several readings) -> {index: total - max used}."""
    used, tot = {}, {}
    for row in readings or []:
        for r in row or []:
            if not isinstance(r, (list, tuple)) or len(r) < 3:
                continue
            i, u, t = int(r[0]), float(r[1]), float(r[2])
            used[i] = max(used.get(i, 0.0), u)
            tot[i] = t
    return {i: tot[i] - used[i] for i in sorted(tot)}


def worker_free(readings):
    """rpc_cache_info results -> {rank: lowest free MiB} (the worker's own torch free memory; rank r runs on card r)."""
    out = {}
    for rd in readings or []:
        for w in ((rd.get("ranks") or []) if isinstance(rd, dict) else []):
            if not isinstance(w, dict) or not isinstance(w.get("rank"), int) or isinstance(w.get("rank"), bool):
                continue
            f = (w.get("mem") or {}).get("free_mib")
            if _fin(f):
                out[w["rank"]] = min(out.get(w["rank"], f), f)
    return out


def combine_free(smi, worker, default_gap_mib=1024):
    """klee100 R1-3: per card min(nvidia-smi free, the worker's free of the same rank); no worker reading -> nvidia-smi
    minus 1,024 MiB. Returns (free, source)."""
    out, src = {}, {}
    for i, v in (smi or {}).items():
        w = (worker or {}).get(i)
        if _fin(w):
            out[i], src[i] = min(v, w), "min(nvidia-smi, worker)"
        else:
            out[i], src[i] = v - default_gap_mib, f"nvidia-smi - {default_gap_mib} MiB (no worker reading)"
    return out, src


def probe_engine_view(raw):
    """klee100 R1-5: the short arithmetic probes gate on ENGINE faults only; the arithmetic itself is recorded."""
    cases = list((raw or {}).get("cases") or [])
    faults = [c.get("q") for c in cases if c.get("error") or not str(c.get("content") or "").strip() or c.get("garbage")]
    return {"ok": bool(cases) and not faults, "engine_faults": faults,
            "probes_correct": f"{sum(1 for c in cases if c.get('ok'))}/{len(cases)}",
            "arith_all_correct": (raw or {}).get("ok"), "cases": cases, "gate": "engine faults only (klee100 R1-5)"}


# ===================================================================================== read-back (fail closed)
def parse_nondefault(line):
    """The facts NB4 checks from vLLM's 'non-default args:' line."""
    g = lambda rx: (re.search(rx, line or "") or [None, None])[1]
    caps = g(r"'cudagraph_capture_sizes': \[([\d, ]*)\]")
    return {"kv_cache_dtype": g(r"'kv_cache_dtype': '(\w+)'"), "kv_cache_memory_bytes": g(r"'kv_cache_memory_bytes': (\d+)"),
            "max_num_seqs": g(r"'max_num_seqs': (\d+)"), "num_speculative_tokens": g(r"'num_speculative_tokens': (\d+)"),
            "spec_method": g(r"'speculative_config': \{'method': '(\w+)'"),
            "language_model_only": g(r"'language_model_only': (True|False)"),
            "enable_prefix_caching": g(r"'enable_prefix_caching': (True|False)"),
            "worker_extension_cls": g(r"'worker_extension_cls': '([^']+)'"),
            "capture_sizes": [int(x) for x in caps.split(",") if x.strip()] if caps is not None else None,
            "cudagraph_mode": g(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)"), "profiler": g(r"ProfilerConfig\(profiler='(\w+)'"),
            "iteration_details": g(r"'enable_logging_iteration_details': (True|False)"),
            "cudagraph_metrics": g(r"'cudagraph_metrics': (True|False)")}


D8_KEYS = ("VLLM_TARGET_SEQ_LENS_ASYNC", "VLLM_E2_GUARD_MODE", "VLLM_GDN_DECODE_FASTPATH", "VLLM_GDN_SHARED_ALIGN_INDEX",
           "VLLM_GDN_BA_AUX_STREAM", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_DEVICE_MAX_CONNECTIONS", "OMP_NUM_THREADS",
           "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_BREAKABLE_CUDAGRAPH")


def d8_switches(env):
    """Lead item 10: the Merlin D8 host switches, the allocator, max connections, OMP and every NCCL_* (record only)."""
    env = env or {}
    out = {k: env.get(k) for k in D8_KEYS}
    out.update({k: v for k, v in sorted(env.items()) if str(k).startswith("NCCL_")})
    return out


def readback_checks(nd, env_keys, arm, kv_bytes):
    """The base settings must be exactly the base, the arm's own change present (fail closed: unreadable = False)."""
    cd = compile_dict(arm)
    env_keys = env_keys or {}
    ck = {"kv_dtype": nd.get("kv_cache_dtype") == BASE["kv_dtype"],
          "pin": nd.get("kv_cache_memory_bytes") == str(int(kv_bytes)),
          "seqs": nd.get("max_num_seqs") == str(BASE["seqs"]),
          "mtp_k": nd.get("num_speculative_tokens") == str(arm.get("k") or K_SPEC) and nd.get("spec_method") == "mtp",
          "language_model_only": nd.get("language_model_only") == "True",
          "prefix_caching": nd.get("enable_prefix_caching") == "True",
          "worker_extension": nd.get("worker_extension_cls") == "nb1_worker_ext.NB1WorkerExt",
          "captures": nd.get("capture_sizes") == list(cd.get("cudagraph_capture_sizes") or []),
          "cudagraph_mode": nd.get("cudagraph_mode") == str(cd.get("cudagraph_mode")),
          "profiler": nd.get("profiler") == "torch", "cudagraph_metrics": nd.get("cudagraph_metrics") == "True",
          "iteration_details": (nd.get("iteration_details") == "True") if arm["key"] == "T0" else (nd.get("iteration_details") != "True")}
    for k, v in dict(BASE["env"], **(arm.get("env") or {})).items():
        ck[f"env {k}"] = env_keys.get(k) == str(v)
    if "VLLM_RECOVERSSM_COMMIT_GRAPH" not in (arm.get("env") or {}):
        ck["env VLLM_RECOVERSSM_COMMIT_GRAPH absent"] = env_keys.get("VLLM_RECOVERSSM_COMMIT_GRAPH") in (None, "", "eager")
    if "CUDA_DEVICE_MAX_CONNECTIONS" not in (arm.get("env") or {}):
        ck["env CUDA_DEVICE_MAX_CONNECTIONS absent"] = env_keys.get("CUDA_DEVICE_MAX_CONNECTIONS") in (None, "")
    return ck


ARGV_IGNORE = ("--port", "--host")
ENV_IGNORE = ("VLLM_PORT", "MASTER_PORT", "NCCL_VERSION", "NB4_CORE_TIMER", "PORT")


def _flags(argv):
    out = {}
    a = [str(x) for x in argv or []]
    for i, x in enumerate(a):
        if x.startswith("--"):
            nxt = a[i + 1] if i + 1 < len(a) else None
            out[x] = nxt if (nxt is not None and not nxt.startswith("--")) else True
    return out


def diff_vs_ref(ref_argv, ref_env, argv, env):
    """Effective flag (last occurrence wins) and env differences of a launch vs the reference launch (the base arm)."""
    rf, cf = _flags(ref_argv), _flags(argv)
    fl = {k: [rf.get(k), cf.get(k)] for k in sorted(set(rf) | set(cf)) if k not in ARGV_IGNORE and rf.get(k) != cf.get(k)}
    re_, ce = ref_env or {}, env or {}
    ev = {k: [re_.get(k), ce.get(k)] for k in sorted(set(re_) | set(ce)) if k not in ENV_IGNORE and re_.get(k) != ce.get(k)}
    return {"flags": fl, "env": ev}


def expected_diff(arm, ref_arm, pin_differs=False):
    """The flags / env keys an arm may differ in from the reference arm (both = base + their own change)."""
    fl, ev = set(), set()
    for a in (arm, ref_arm):
        if a.get("compile"):
            fl.add("--compilation-config")
        if a.get("spec") or (a.get("prebuild_env") or {}).get("DRAFT_HEAD"):
            fl.add("--speculative-config")     # Merlin build() writes DRAFT_HEAD into the speculative config
        fl.update(str(x) for x in (a.get("extra") or []) if str(x).startswith("--"))
        ev.update(a.get("env") or {})
        ev.update(a.get("prebuild_env") or {})
    if pin_differs:
        fl.add("--kv-cache-memory-bytes")
        ev.add("KV_BYTES")
    return fl, ev


def one_change_ok(diff, arm, ref_arm, pin_differs=False):
    """True when the launch differs from the reference launch only where the two arms' planned changes allow."""
    fl, ev = expected_diff(arm, ref_arm, pin_differs)
    extra_f = sorted(k for k in (diff.get("flags") or {}) if k not in fl)
    extra_e = sorted(k for k in (diff.get("env") or {}) if k not in ev)
    return {"ok": not extra_f and not extra_e, "unexpected_flags": extra_f, "unexpected_env": extra_e,
            "allowed_flags": sorted(fl), "allowed_env": sorted(ev)}


# ===================================================================================== --cudagraph-metrics, the split
CG_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|", re.M)


def cudagraph_table(log_text, k=K_SPEC, max_n=SEQS):
    """vLLM --cudagraph-metrics tables (compilation/cuda_graph.py: Unpadded / Padded / Paddings / Runtime Mode / Count,
    logged every stats interval) in a log slice -> decode rows (unpadded = n x (1 + K), n <= max_n) by n and mode."""
    rows = [(int(a), int(b), int(c), m, int(n)) for a, b, c, m, n in CG_ROW.findall(log_text or "")]
    q = int(k) + 1
    dec, other = {}, 0
    for u, p, pad, mode, cnt in rows:
        if u % q == 0 and 1 <= u // q <= max_n:
            d = dec.setdefault(str(u // q), {})
            d[mode] = d.get(mode, 0) + cnt
            d.setdefault("padded", {})[str(p)] = d.get("padded", {}).get(str(p), 0) + cnt
        else:
            other += cnt
    by_mode = {}
    for d in dec.values():
        for m, c in d.items():
            if m != "padded":
                by_mode[m] = by_mode.get(m, 0) + c
    return {"rows": len(rows), "decode_by_n": dec, "decode_by_mode": by_mode, "non_decode_steps": other}


def split_label(table, level):
    """The micro-batch split of a window (lead item 3 / D item 8): BALANCED when >= 90 % of the decode steps carry
    floor(L/2) or ceil(L/2) requests; IMBALANCED otherwise (e.g. 8/0 parity lock); UNKNOWN without rows."""
    dec = (table or {}).get("decode_by_n") or {}
    tot = {n: sum(v for m, v in d.items() if m != "padded") for n, d in dec.items()}
    s = sum(tot.values())
    if not s:
        return {"label": "UNKNOWN", "steps": 0, "n_hist": tot}
    want = {str(level // 2), str(level - level // 2)}
    frac = sum(v for n, v in tot.items() if n in want) / s
    return {"label": "BALANCED" if frac >= 0.9 else "IMBALANCED", "steps": s, "n_hist": tot, "balanced_frac": round(frac, 3),
            "want_n": sorted(want)}


# ===================================================================================== engaged (log) + mechanism
def commit_graph_ns(log_text):
    return sorted({int(x) for x in re.findall(r"\[recoverssm-commit\] captured commit graph n=(\d+)", log_text or "")})


def engaged_log(arm, log_text, env_keys=None, worker_env=None, readback=None):
    """The arm's change must be visible BEFORE any measurement (fail closed): the server env read-back, the env read in
    every worker / session process (S: CUDA reads it at context creation), the arm's log lines."""
    t, k = log_text or "", arm["key"]
    ev = {"rx": {rx: bool(re.search(rx, t)) for rx in (arm.get("engaged_log_rx") or [])}}
    ck = (readback or {})
    if k in ("T0b",):
        return {"ok": True, "evidence": {"why": "A/A: no change to see"}}
    if k == "T0":
        ev["commit_captured_n"] = commit_graph_ns(t)
        ev["iteration_details_flag"] = ck.get("iteration_details")
        return {"ok": bool(ck.get("iteration_details")), "evidence": ev}
    if k == "S":
        want = (arm.get("env") or {}).get("CUDA_DEVICE_MAX_CONNECTIONS")
        vals = [v for v in (worker_env or {}).values()]
        ev["server_env"] = (env_keys or {}).get("CUDA_DEVICE_MAX_CONNECTIONS")
        ev["worker_env"] = worker_env
        ok = ev["server_env"] == want and bool(vals) and all(v == want for v in vals)
        return {"ok": ok, "evidence": ev}
    if k == "G":
        ev["captures_checked"] = ck.get("captures")
        return {"ok": bool(ck.get("captures")), "evidence": ev}
    if k == "K4":
        return {"ok": bool(ck.get("mtp_k")) and bool(ck.get("captures")), "evidence": ev}
    if k in ("P", "S2", "S3"):
        rx = arm.get("init_rx")
        ev["init_rx"], ev["init_lines"] = rx, (len(re.findall(rx, t)) if rx else 0)
        ok = bool(rx) and ev["init_lines"] > 0
        for e_, v_ in (arm.get("env") or {}).items():
            ok = ok and (env_keys or {}).get(e_, (worker_env or {}).get(e_)) in (str(v_), None)
        return {"ok": ok, "evidence": ev}
    ok = bool(ev["rx"]) and all(ev["rx"].values())
    return {"ok": ok, "evidence": ev}


def commit_split(before, after, steps=None):
    """RecoverSSM commits per rank between two get_recoverssm_commit_stats reads (lead item 5): graph replays / staged /
    eager deltas and the fallback reasons (pp-deferred / not_captured / no_commit ...), per step when steps is given."""
    out = []
    for i, (b, a) in enumerate(zip(before or [], after or [])):
        b, a = b or {}, a or {}
        d = {k: (a.get(k) or 0) - (b.get(k) or 0) for k in ("replays", "staged_steps", "eager_steps", "captures")}
        fb = {r: (a.get("fallbacks") or {}).get(r, 0) - (b.get("fallbacks") or {}).get(r, 0)
              for r in set(a.get("fallbacks") or {}) | set(b.get("fallbacks") or {})}
        r = {"rank": i, "mode": a.get("mode"), "delta": d, "fallbacks_delta": {k: v for k, v in fb.items() if v}}
        if steps:
            r["per_step"] = {k: round(v / steps, 3) for k, v in d.items()}
        out.append(r)
    tot = {k: sum(x["delta"][k] for x in out) for k in ("replays", "staged_steps", "eager_steps")}
    return {"ranks": out, "total": tot, "steps": steps}


def stall_fraction(analysis):
    """From nb4_timeline.analyze (agent C): the share of steady stage-0 graphs with an in-graph stall > 1 ms and the mean
    stall; None when not measurable."""
    st = (analysis or {}).get("stalls") or {}
    n, w = st.get("steady_ticks"), st.get("ticks_with_stall_over_1ms")
    if not (_fin(n) and _fin(w)) or not n:
        return None
    return {"frac_over_1ms": round(w / n, 3), "steady": n, "stall_ms_mean": st.get("graph_stall_ms_mean"),
            "tick_ms_mean": ((analysis or {}).get("summary") or {}).get("tick_ms_mean")}


def full_steps(table):
    by = (table or {}).get("decode_by_n") or {}
    return {n: {m: c for m, c in d.items() if m != "padded"} for n, d in by.items()}


def mechanism(arm, ev):
    """The trace / counters must prove the change engaged (fail closed). ev keys: tl (this arm's nb4_timeline analysis at
    its compare band), tl_ref (T0's at the same band), cg (cudagraph tables by window), log, p5_label, p5_delta, accept."""
    m = arm.get("mechanism")
    if m is None:
        return {"ok": True, "why": "A/A: no mechanism"}
    if m == "timeline":
        tl = ev.get("tl") or {}
        ok = len(tl.get("ranks") or []) == 4 and bool(tl.get("summary"))
        return {"ok": ok, "trace_ranks": tl.get("ranks"), "iteration_lines": ev.get("iteration_lines"),
                "core_timer_rows": ev.get("core_timer_rows"), "why": None if ok else "timeline incomplete (4 rank traces + analysis)"}
    if m == "stall":
        f, f0 = stall_fraction(ev.get("tl")), stall_fraction(ev.get("tl_ref"))
        ok = bool(f) and f["frac_over_1ms"] <= 0.05
        out = {"ok": ok, "stall": f, "stall_t0": f0, "rule": "stage-0 in-graph stall > 1 ms in <= 5 % of steady graphs",
               "why": None if ok else f"stall not removed: {f}"}
        if arm.get("key") in ("S2", "S3"):         # the Q-pack / Q-defer read-back: every PP worker's init + packed send / recv counters
            pl = ev.get("ppack_label") or ["MISSING", {}]
            out["ppack_label"] = pl
            if pl[0] != "ENGAGED":
                out.update(ok=False, why=f"{arm.get('variant')} read-back {pl[0]}: {pl[1]}")
        return out
    if m == "capture_coverage":
        tabs = ev.get("cg") or {}
        rows = {w: full_steps(tabs.get(w)) for w in ("8k@9", "40k@9")}
        full9 = sum((d.get("9") or {}).get("FULL", 0) for d in rows.values())
        none_dec = sum(c for d in rows.values() for n in d.values() for mm, c in n.items() if mm == "NONE")
        ok = full9 > 0 and none_dec == 0
        return {"ok": ok, "level9_decode_by_n": rows, "full_n9_steps": full9, "none_decode_steps": none_dec,
                "why": None if ok else f"36-token steps FULL {full9}, eager decode rows {none_dec}"}
    if m == "commit_replays":
        cs = ev.get("commit") or {}
        reps = (cs.get("total") or {}).get("replays")
        ok = _fin(reps) and reps > 0
        return {"ok": ok, "replays": reps, "split": cs, "captured_n": commit_graph_ns(ev.get("log")),
                "why": None if ok else f"no commit replays in the graph windows ({reps})"}
    if m == "p5":
        lab = ev.get("p5_label") or ["MISSING", []]
        d = ev.get("p5_delta") or {}
        ok = lab[0] == "ENGAGED" and _fin(d.get("executed")) and d["executed"] > 0
        return {"ok": ok, "label": lab, "delta": d, "why": None if ok else f"P5 not engaged: {lab}"}
    if m == "spec_k4":
        acc = ev.get("accept") or {}
        ok = any(str(p) in {"3", 3} or p == 3 for p in acc) if acc else False
        return {"ok": ok, "accept_positions": sorted(acc, key=str), "why": None if ok else "no 4th acceptance position"}
    if m == "log_line":
        rx = arm.get("engaged_log_rx") or []
        ok = bool(rx) and all(re.search(x, ev.get("log") or "") for x in rx)
        return {"ok": ok, "why": None if ok else f"log line missing: {rx}"}
    return {"ok": False, "why": f"unknown mechanism {m}"}


# ===================================================================================== correctness
def tf_label(c):
    """One cmp_full / compare_tf record -> TF_CLOSE (NB0 screening: mean <= 0.15 and rank-1 >= 0.90) / TF_DIVERGENT /
    TF_DIVERGENT_SEVERE (NB0 triage) / INCONCLUSIVE_* (klee100 K-5: PARTIAL is not a reading)."""
    if not isinstance(c, dict):
        return "INCONCLUSIVE_MISSING"
    if c.get("status") != "MEASURED":
        return "INCONCLUSIVE_" + str(c.get("status") or "MISSING").replace("INCONCLUSIVE_", "")
    if c.get("expected") is not None and c.get("n") != c.get("expected"):
        return "INCONCLUSIVE_INCOMPLETE_POSITIONS"
    m, r1 = c.get("mean_abs_dlogprob"), c.get("rank1_agreement")
    if not (_fin(m) and _fin(r1)):
        return "INCONCLUSIVE_NONFINITE"
    if m > TRIAGE["mean_max"] or r1 < TRIAGE["rank1_min"]:
        return "TF_DIVERGENT_SEVERE"
    if m <= SCREEN["mean_max"] and r1 >= SCREEN["rank1_min"]:
        return "TF_CLOSE"
    return "TF_DIVERGENT"


def correctness(evidence_corr, oracle_status, tf_labels, bands=("2k", "8k", "40k"), need_exact=False):
    """Per arm vs the base: the boot's evidence gate (CLEAN / NO_GO_* / INCONCLUSIVE_*), the greedy oracle vs the base
    (EXACT / NEAR_TIE pass; need_exact: EXACT only) and TF 2k / 8k / 40k vs the base (every band TF_CLOSE). PASS only
    when all hold; FAIL on any failure; otherwise INCONCLUSIVE (never a WIN)."""
    labs = {b: (tf_labels or {}).get(b) for b in bands}
    why = []
    ec = str(evidence_corr or "MISSING")
    if ec.startswith("NO_GO") or oracle_status == "FAIL_HIGH_MARGIN" or any(str(l).startswith("TF_DIVERGENT") for l in labs.values()):
        if ec.startswith("NO_GO"):
            why.append(f"evidence {ec}")
        if oracle_status == "FAIL_HIGH_MARGIN":
            why.append("greedy oracle FAIL_HIGH_MARGIN vs base")
        why += [f"TF {b} {l}" for b, l in labs.items() if str(l).startswith("TF_DIVERGENT")]
        return {"verdict": "FAIL", "why": why, "oracle": oracle_status, "tf": labs, "evidence": ec}
    if ec != "CLEAN":
        why.append(f"evidence {ec}")
    if oracle_status not in (("EXACT",) if need_exact else CLEAN_ORACLE):
        why.append(f"oracle {oracle_status}")
    why += [f"TF {b} {l}" for b, l in labs.items() if l != "TF_CLOSE"]
    return {"verdict": "PASS" if not why else "INCONCLUSIVE", "why": why or None, "oracle": oracle_status, "tf": labs, "evidence": ec}


# ===================================================================================== speed (winner rule)
def window_ok(w):
    """A usable steady window: STEADY, real running == level, prefill done before it, 0 preemptions."""
    return (isinstance(w, dict) and w.get("status") == "STEADY" and w.get("running_eq_level") is True
            and not w.get("preemptions") and _fin(w.get("tok_s")) and w["tok_s"] > 0)


def reps_of(wins, key):
    """All usable replicates of window `key` ('8k@8', '8k@8#2', ...)."""
    return [v for k, v in (wins or {}).items() if (k == key or k.startswith(key + "#")) and window_ok(v)]


def mean_tok(wins, key):
    xs = reps_of(wins, key)
    return round(sum(x["tok_s"] for x in xs) / len(xs), 3) if xs else None


def split_of(wins, key):
    xs = reps_of(wins, key)
    labs = {((x.get("split") or {}).get("label")) for x in xs}
    return labs.pop() if len(labs) == 1 else ("MIXED" if labs else None)


def aa_spread(t0_w, t0b_w, windows=WINDOWS):
    """Per window: the A/A spread in % = max(|T0 - T0b| / mean, the within-boot replicate spread of T0b) (lead item 9)."""
    out = {}
    for k in windows:
        a, b = mean_tok(t0_w, k), mean_tok(t0b_w, k)
        if a is None or b is None:
            out[k] = None
            continue
        m = (a + b) / 2
        reps = [x["tok_s"] for x in reps_of(t0b_w, k)]
        within = 100 * (max(reps) - min(reps)) / statistics.mean(reps) if len(reps) >= 2 else None
        cross = 100 * abs(a - b) / m
        out[k] = {"t0": a, "t0b": b, "mean": round(m, 2), "cross_pct": round(cross, 3),
                  "within_t0b_pct": round(within, 3) if within is not None else None,
                  "spread_pct": round(max(cross, within or 0.0), 3), "split_t0": split_of(t0_w, k), "split_t0b": split_of(t0b_w, k)}
    return out


def speed_verdict(arm_w, t0_w, t0b_w, windows=WINDOWS, decisive=WINDOWS, min_gain_pct=MIN_GAIN_PCT):
    """The winner rule (spec + lead): every DECISIVE window beats the base by more than max(2 %, the measured A/A
    spread); no window SLOWER. Base = the mean of the T0 / T0b usable windows; a window whose split differs from the
    base's is not compared (INCONCLUSIVE_SPLIT). No A/A -> INCONCLUSIVE_NO_AA (provisional gains kept)."""
    aa = aa_spread(t0_w, t0b_w, windows)
    per, labs = {}, {}
    for k in windows:
        at = mean_tok(arm_w, k)
        bases = [x for x in (mean_tok(t0_w, k), mean_tok(t0b_w, k)) if x is not None]
        if at is None or not bases:
            per[k] = {"label": "INCONCLUSIVE_WINDOW", "arm_tok_s": at, "base_windows": len(bases)}
            labs[k] = "INCONCLUSIVE_SPEED"
            continue
        sa, sb = split_of(arm_w, k), split_of(t0_w, k) or split_of(t0b_w, k)
        base = sum(bases) / len(bases)
        gain = 100 * (at - base) / base
        sp = (aa.get(k) or {}).get("spread_pct")
        thr = max(min_gain_pct, sp) if sp is not None else None
        d = {"arm_tok_s": at, "base_tok_s": round(base, 2), "gain_pct": round(gain, 3), "aa_spread_pct": sp, "threshold_pct": thr,
             "split_arm": sa, "split_base": sb}
        if sa and sb and sa != sb:
            d["label"] = "INCONCLUSIVE_SPLIT"
        elif thr is None:
            d["label"] = "INCONCLUSIVE_NO_AA"
            d["provisional_vs_2pct"] = "WIN" if gain > min_gain_pct else "SLOWER" if gain < -min_gain_pct else "NO_GAIN"
        else:
            d["label"] = "WIN" if gain > thr else "SLOWER" if gain < -thr else "NO_GAIN"
        per[k] = d
        labs[k] = d["label"] if d["label"] in ("WIN", "SLOWER", "NO_GAIN", "INCONCLUSIVE_NO_AA") else "INCONCLUSIVE_SPEED"
    dl = [labs.get(k) for k in decisive]
    if "SLOWER" in labs.values():
        lab = "SLOWER"
    elif "INCONCLUSIVE_SPEED" in dl or None in dl:
        lab = "INCONCLUSIVE_SPEED"
    elif "INCONCLUSIVE_NO_AA" in dl:
        lab = "INCONCLUSIVE_NO_AA"
    elif dl and all(x == "WIN" for x in dl):
        lab = "WIN"
    else:
        lab = "NO_GAIN"
    return {"label": lab, "windows": per, "aa": aa, "decisive": list(decisive),
            "rule": f"every decisive window gain > max({min_gain_pct} %, A/A spread), no window SLOWER, equal splits only"}


C_WINDOWS = ("8k@8", "40k@8")                  # the C runtime A/B windows (n = 4 per micro-batch: captured with CAPS16)


def c_verdict(graph_w, eager_w, aa, greedy, commit, evidence_corr, windows=C_WINDOWS, min_gain_pct=MIN_GAIN_PCT):
    """Arm C = the runtime A/B inside T0's boot (lead item 4): the same boot's eager windows (T0's own) vs the graph
    windows after set_recoverssm_commit_mode graph. Correctness: the greedy row graph vs eager EXACT (bit-identity claim)
    and T0's evidence CLEAN; mechanism: PP1 replays > 0; speed: every window > max(2 %, the A/A spread)."""
    corr = ("PASS" if (greedy or {}).get("status") == "EXACT" and evidence_corr == "CLEAN" else
            "FAIL" if (greedy or {}).get("status") == "FAIL_HIGH_MARGIN" else "INCONCLUSIVE")
    mech = mechanism({"mechanism": "commit_replays"}, {"commit": commit})
    per, labs = {}, []
    for k in windows:
        g, e = mean_tok(graph_w, k), mean_tok(eager_w, k)
        if g is None or e is None:
            per[k] = {"label": "INCONCLUSIVE_WINDOW", "graph": g, "eager": e}
            labs.append("INCONCLUSIVE_SPEED")
            continue
        gain = 100 * (g - e) / e
        sp = ((aa or {}).get(k) or {}).get("spread_pct")
        thr = max(min_gain_pct, sp) if sp is not None else min_gain_pct
        sg, se = split_of(graph_w, k), split_of(eager_w, k)
        lab = ("INCONCLUSIVE_SPEED" if sg and se and sg != se else "WIN" if gain > thr else "SLOWER" if gain < -thr else "NO_GAIN")
        per[k] = {"graph_tok_s": g, "eager_tok_s": e, "gain_pct": round(gain, 3), "threshold_pct": thr, "aa_measured": sp is not None,
                  "split_graph": sg, "split_eager": se, "label": lab}
        labs.append(lab)
    sp_lab = ("SLOWER" if "SLOWER" in labs else "INCONCLUSIVE_SPEED" if "INCONCLUSIVE_SPEED" in labs else
              "WIN" if labs and all(x == "WIN" for x in labs) else "NO_GAIN")
    if corr == "FAIL":
        lab = "CORRECTNESS_FAIL"
    elif not mech["ok"]:
        lab = "NOT_ENGAGED"
    elif corr != "PASS":
        lab = "CORRECTNESS_INCONCLUSIVE"
    else:
        lab = "WIN_SPEED" if sp_lab == "WIN" else sp_lab
    return {"label": lab, "correctness": corr, "greedy_graph_vs_eager": greedy, "mechanism": mech, "speed": sp_lab, "windows": per,
            "note": "same-boot runtime A/B (stage 1 only: stage-0 commits are PP-deferred and always eager); expected <= 1 %"}


# ===================================================================================== labels + plan
BOOT_FAIL_PREFIX = ("LAUNCH_FAILED", "GPU_NOT_EMPTY", "HOST_RAM_GATE_FAILED", "ENGINE_PROCS_LEFT", "SERVER_DIED", "BOOT_CAP_KILL",
                    "DEVICE_FAULT", "TF_OOM", "BODY_OOM", "TF_SERVER_DIED", "FAIL_GPU_RELEASE", "BODY_ERROR", "ERROR",
                    "POSTPROCESS_ERROR", "PARTIAL_HARD_STOP", "SKIPPED_ABORT", "PATCH_", "P5_GATE")


def boot_failed(rec):
    st = str((rec or {}).get("status") or "")
    return st.startswith(BOOT_FAIL_PREFIX) or bool((rec or {}).get("death"))


def win_label(arm):
    """Lead item 11: a speed WIN reads WIN_SPEED (adoption needs paired ARC accuracy + PLAN §4.1-7); P is WIN_DIAGNOSTIC
    (2 repeats, 40k / 90k rows and contention evidence are not in NB4)."""
    return "WIN_DIAGNOSTIC" if arm["key"] == "P" else "WIN_SPEED"


def final_label(arm, rec, corr=None, speed=None, engaged=None, mech=None):
    """Label precedence (fail closed): not integrated > skipped > boot failed > config mismatch > not engaged (log) >
    correctness FAIL > mechanism not proven (NOT_ENGAGED) > correctness inconclusive > speed. Base arms: BASE / AA."""
    k = arm["key"]
    rec = rec or {}
    st = str(rec.get("status") or "")
    if arm.get("disabled"):
        return {"label": "DISABLED", "why": arm.get("why_not")}
    if not arm.get("integrated"):
        return {"label": "P5_NOT_INTEGRATED" if k == "P" else "NOT_INTEGRATED", "why": arm.get("why_not")}
    if st.startswith("P5_NOT_INTEGRATED") or st.startswith("NOT_INTEGRATED"):
        return {"label": "P5_NOT_INTEGRATED" if k == "P" else "NOT_INTEGRATED", "why": rec.get("why") or st}
    if st.startswith("SKIPPED"):
        return {"label": "SKIPPED_NO_BASE" if st == "SKIPPED_NO_BASE" else "SKIPPED_BUDGET", "why": rec.get("why") or st}
    if not rec:
        return {"label": "SKIPPED_BUDGET", "why": "not run"}
    if boot_failed(rec):
        return {"label": "BASE_FAILED" if k == "T0" else "AA_FAILED" if k == "T0b" else "BOOT_FAILED", "why": rec.get("death") or st}
    if rec.get("early_stop") == "CONFIG_MISMATCH" or st == "CONFIG_MISMATCH":
        return {"label": "CONFIG_MISMATCH", "why": rec.get("config_mismatch")}
    if k in BASE_ARMS:
        good = st == "DONE" and rec.get("correctness") == "CLEAN" and not rec.get("early_stop")
        lab = ("BASE" if good else "BASE_FAILED") if k == "T0" else ("AA" if good else "AA_FAILED")
        return {"label": lab, "why": None if good else (rec.get("early_stop") or rec.get("correctness") or st),
                "aa_correctness": (corr or {}).get("verdict") if k == "T0b" else None}
    if (engaged is not None and not engaged.get("ok")) or rec.get("early_stop") == "NOT_ENGAGED":
        return {"label": "NOT_ENGAGED", "why": "log / env: " + json.dumps((engaged or {}).get("evidence") or rec.get("engaged_why"),
                                                                        default=str)[:300]}
    c = (corr or {}).get("verdict")
    if c == "FAIL":
        return {"label": "CORRECTNESS_FAIL", "why": (corr or {}).get("why")}
    if mech is not None and not mech.get("ok"):
        return {"label": "NOT_ENGAGED", "why": "mechanism: " + str(mech.get("why"))}
    if c != "PASS":
        return {"label": "CORRECTNESS_INCONCLUSIVE", "why": (corr or {}).get("why")}
    s = (speed or {}).get("label") or "INCONCLUSIVE_SPEED"
    if s == "WIN":
        return {"label": win_label(arm), "why": WIN_NOTE, "diagnostic": bool(arm.get("diagnostic"))}
    return {"label": s, "why": (speed or {}).get("windows"), "diagnostic": bool(arm.get("diagnostic"))}


def base_ok(rec):
    """A base (reference) launch: DONE, evidence CLEAN, no early stop, its oracle rows and TF arrays stored."""
    rec = rec or {}
    return (rec.get("status") == "DONE" and rec.get("correctness") == "CLEAN" and not rec.get("early_stop")
            and rec.get("reference_stored") is True)


def next_arm(arms, done, left_sec, margin_sec=300):
    """The next arm (None = finished). Order T0 -> S -> S2 -> G -> P -> T0b (lead 2026-11-01), then the optional K4 /
    INT8 only if they fit. Not integrated arms are never booted. T0 not a clean base -> T0b next (it becomes the base);
    no clean base -> stop. T0b is protected: before any other arm X, if X + T0b do not both fit, T0b runs now."""
    todo = [k for k in ARM_ORDER if k not in done and arms[k].get("integrated")]
    t0 = done.get("T0")
    if t0 is not None and not base_ok(t0):
        if "T0b" in todo:
            return "T0b", "T0 is not a clean base: T0b runs next as the base"
        if not base_ok(done.get("T0b")):
            return None, "no clean base (T0 and T0b): the remaining arms are SKIPPED_NO_BASE"
    pre = [k for k in todo if k not in OPTIONAL]
    if pre:
        nxt = pre[0]
        if nxt not in BASE_ARMS and "T0b" in pre and left_sec < (EST_MIN[nxt] + EST_MIN["T0b"]) * 60 + 2 * margin_sec:
            return "T0b", f"{nxt} + T0b do not both fit: T0b runs now (protected A/A)"
        return nxt, "lead order"
    for k in todo:
        if left_sec >= EST_MIN[k] * 60 + margin_sec:
            return k, "optional arm, budget allows"
    return None, "all arms done (optional arms: no budget)" if todo else "all arms done"


def ref_arm(done):
    """The reference (correctness + config diff) arm: T0 when it is a clean base, else T0b."""
    for k in BASE_ARMS:
        if base_ok(done.get(k)):
            return k
    return None


# ===================================================================================== T0 extras: op placement, engine-core timing
PLACE = (("ple", r"(?i)ple"), ("embedding", r"(?i)embed"), ("lm_head_logits", r"(?i)logits|lm_head|vocab"),
         ("sampler", r"(?i)sampl|rejection|argmax|topk|top_k|topp"), ("recoverssm_commit", r"(?i)recoverssm|commit_|boundary_fold|final_fold"),
         ("pp_broadcast", r"Broadcast"), ("pp_sendrecv", r"SendRecv|ncclDevKernel_Send|ncclDevKernel_Recv"),
         ("tp_allreduce", r"AllReduce"), ("allgather", r"AllGather"), ("h2d_copy", r"(?i)Memcpy HtoD|HtoD"),
         ("draft_mtp", r"(?i)draft|propose|mtp|eagle"))


def op_placement(events):
    """Lead item 3: which rank runs what. Per bucket: GPU kernel / memcpy ms (by name) and host op / annotation count
    (cpu_op / user_annotation / python names: draft passes, sampler, commit, broadcast origin)."""
    out = {b: {"gpu_ms": 0.0, "gpu_n": 0, "host_n": 0} for b, _ in PLACE}
    for e in events or []:
        if e.get("ph") != "X":
            continue
        cat, name = str(e.get("cat", "")).lower(), str(e.get("name", ""))
        gpu = cat in ("kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset")
        host = cat in ("cpu_op", "user_annotation", "python_function")
        if not (gpu or host):
            continue
        for b, rx in PLACE:
            if re.search(rx, name):
                if gpu:
                    out[b]["gpu_ms"] += float(e.get("dur", 0.0)) / 1000.0
                    out[b]["gpu_n"] += 1
                else:
                    out[b]["host_n"] += 1
                break
    return {b: {"gpu_ms": round(v["gpu_ms"], 3), "gpu_n": v["gpu_n"], "host_n": v["host_n"]} for b, v in out.items()}


ITER_RX = re.compile(r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, "
                     r"(\d+) generation tokens, iteration elapsed time: ([\d.]+) ms")


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def iteration_summary(log_text):
    """vLLM's --enable-logging-iteration-details lines (T0): decode-only steps by generation requests (n per step, in
    step order: the parity sequence) with the blocking wait ('iteration elapsed time')."""
    rows = [tuple(float(x) for x in m.groups()) for m in ITER_RX.finditer(log_text or "")]
    out = {"lines": len(rows)}
    dec = {}
    seq = []
    for it, cr, ct, gr, gt, ms in rows:
        if cr == 0:
            seq.append(int(gr))
            if gr > 0:
                dec.setdefault(int(gr), []).append(ms)
    out["decode_by_gen_requests"] = {str(k): {"n": len(v), "elapsed_ms_median": round(statistics.median(v), 3),
                                              "elapsed_ms_p90": round(_pct(v, 0.9), 3)} for k, v in sorted(dec.items())}
    even, odd = seq[0::2], seq[1::2]
    out["parity"] = {"even_mean_n": round(statistics.mean(even), 3) if even else None,
                     "odd_mean_n": round(statistics.mean(odd), 3) if odd else None, "steps": len(seq), "tail": seq[-40:]}
    return out


def core_timer_summary(rows):
    """The env-gated step timer rows [t_in_ns, t_out_ns, dur_ns, sched_ns, update_ns, q_in, q_out, executed, outputs,
    kind] -> per-step and between-step times (ms) of the engine core."""
    rows = sorted([r for r in rows or [] if isinstance(r, (list, tuple)) and len(r) >= 10], key=lambda r: r[0])
    if not rows:
        return {"rows": 0}
    f = lambda xs: {"n": len(xs), "median": round(statistics.median(xs), 3), "p90": round(_pct(xs, 0.9), 3),
                    "mean": round(statistics.mean(xs), 3)} if xs else {"n": 0}
    return {"rows": len(rows), "step_ms": f([r[2] / 1e6 for r in rows]), "schedule_ms": f([r[3] / 1e6 for r in rows]),
            "update_ms": f([r[4] / 1e6 for r in rows]),
            "between_steps_ms": f([(rows[i + 1][0] - rows[i][1]) / 1e6 for i in range(len(rows) - 1)]),
            "steps_with_outputs_ms": f([r[2] / 1e6 for r in rows if r[8] > 0]),
            "queue_fill_steps_ms": f([r[2] / 1e6 for r in rows if r[8] == 0]),
            "span_s": round((rows[-1][1] - rows[0][0]) / 1e9, 3), "kinds": sorted({r[9] for r in rows})}


def cpu_per_process(before, after, dt, roles=None):
    """Lead item 2: CPU use per server-session process over a window (NB0 session_procs snapshots: utime + stime ticks)
    -> cores used per process (role from the server log), and the total."""
    tck = 100.0
    b = {p["pid"]: p for p in before or [] if isinstance(p, dict) and "pid" in p}
    out = []
    for p in after or []:
        q = b.get(p.get("pid"))
        if not q or not dt:
            continue
        c = ((p.get("utime", 0) + p.get("stime", 0)) - (q.get("utime", 0) + q.get("stime", 0))) / tck / dt
        out.append({"pid": p["pid"], "role": (roles or {}).get(p["pid"], p.get("comm")), "cores": round(c, 3)})
    out.sort(key=lambda x: -x["cores"])
    return {"procs": out[:24], "total_cores": round(sum(x["cores"] for x in out), 3), "sec": round(dt, 1)}


CORE_TIMER_MODULE = "nb4_core_timer"
CORE_TIMER_DIST = {
    "nb4_core_timer-0.1.dist-info/METADATA": "Metadata-Version: 2.1\nName: nb4-core-timer\nVersion: 0.1\n",
    "nb4_core_timer-0.1.dist-info/entry_points.txt": "[vllm.general_plugins]\nnb4_core_timer = nb4_core_timer:register\n",
    "nb4_core_timer-0.1.dist-info/RECORD": "",
}
CORE_TIMER_SRC = r'''"""E25 NB4 engine-core step timer: a vLLM general plugin (entry point vllm.general_plugins), DORMANT unless the env
NB4_CORE_TIMER is set (T0 only). In the EngineCore process only (process name EngineCore*): wraps EngineCore.step and
EngineCore.step_with_batch_queue at class level (load_general_plugins runs at the top of EngineCore.__init__, before
step_fn is bound) and, on the first call, the instance's scheduler.schedule / update_from_output. One row per engine step:
[t_in_ns (time.time_ns, wall clock), t_out_ns, dur_ns, schedule_ns, update_ns, batch_queue_in, batch_queue_out, executed,
outputs, kind]; appended to $NB4_CORE_TIMER.<pid>.jsonl every 2 s (daemon thread) and at exit. Never raises into vLLM:
the wrapped call's own exceptions propagate unchanged."""
import json
import os
import threading
import time

_BUF, _LOCK, _CUR = [], threading.Lock(), {}
_ST = {"installed": False, "path": None, "rows": 0, "errors": []}


def _flush():
    with _LOCK:
        rows, _BUF[:] = list(_BUF), []
    if not rows or not _ST["path"]:
        return
    try:
        with open(_ST["path"], "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        _ST["rows"] += len(rows)
    except Exception as e:
        if len(_ST["errors"]) < 5:
            _ST["errors"].append(repr(e)[:160])


def _periodic():
    while True:
        time.sleep(2.0)
        _flush()


def _timed(f, nm):
    def w(*a, **k):
        t = time.perf_counter_ns()
        try:
            return f(*a, **k)
        finally:
            _CUR[nm] = _CUR.get(nm, 0) + time.perf_counter_ns() - t
    return w


def _wrap_sched(core):
    sch = getattr(core, "scheduler", None)
    if sch is None or getattr(sch, "_nb4_wrapped", False):
        return
    for nm in ("schedule", "update_from_output"):
        f = getattr(sch, nm, None)
        if f is not None:
            setattr(sch, nm, _timed(f, nm))
    sch._nb4_wrapped = True


def _qlen(core):
    q = getattr(core, "batch_queue", None)
    return len(q) if q is not None else -1


def _wrap_step(orig, kind):
    def step(self, *a, **k):
        try:
            _wrap_sched(self)
            qi = _qlen(self)
        except Exception:
            qi = -2
        _CUR.clear()
        t0, p0 = time.time_ns(), time.perf_counter_ns()
        out = orig(self, *a, **k)
        p1 = time.perf_counter_ns()
        try:
            outs, ex = out if isinstance(out, tuple) and len(out) == 2 else (None, None)
            row = [t0, t0 + (p1 - p0), p1 - p0, _CUR.get("schedule", 0), _CUR.get("update_from_output", 0), qi, _qlen(self),
                   bool(ex), 0 if not outs else len(outs), kind]
            with _LOCK:
                _BUF.append(row)
        except Exception as e:
            if len(_ST["errors"]) < 5:
                _ST["errors"].append(repr(e)[:160])
        return out
    step.__wrapped__ = orig
    return step


def register():
    try:
        path = os.environ.get("NB4_CORE_TIMER")
        if not path:
            return
        import multiprocessing as mp
        if not str(mp.current_process().name).startswith("EngineCore"):
            return
        import sys
        core = sys.modules.get("vllm.v1.engine.core")
        if core is None:
            import importlib
            core = importlib.import_module("vllm.v1.engine.core")
        ec = core.EngineCore
        if getattr(ec, "_nb4_timer", False):
            return
        for nm, kind in (("step", "step"), ("step_with_batch_queue", "batch_queue")):
            if hasattr(ec, nm):
                setattr(ec, nm, _wrap_step(getattr(ec, nm), kind))
        ec._nb4_timer = True
        _ST.update(installed=True, path=f"{path}.{os.getpid()}.jsonl")
        import atexit
        atexit.register(_flush)
        threading.Thread(target=_periodic, daemon=True, name="nb4-core-timer").start()
        print(f"NB4 core timer installed pid={os.getpid()} -> {_ST['path']}", flush=True)
    except Exception as e:
        try:
            print("NB4 core timer NOT installed: " + repr(e)[:200], flush=True)
        except Exception:
            pass
'''
